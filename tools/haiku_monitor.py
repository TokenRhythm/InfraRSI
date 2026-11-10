"""Haiku 5.5 monitor loop (design §5). Every 10 min: read the deterministic monitor outputs
(execution/monitor/state.json, alerts/OPEN.json, alerts/SUMMARY.md), ask Haiku for a short reading, write
execution/monitor/notes/<ts>.md, and escalate.

Haiku gets read-only tools (Read/Grep/Glob) and answers with one JSON object; this script writes every file.
A Haiku CRIT pauses new launches through the dispatch QueueStore (running jobs are never stopped from here) and
asks Sonnet 5.5 for a root-cause note in alerts/RCA_<ts>.md. Lines starting with 'EVENT ' on stdout are for the
orchestrator's watcher. `python -m infrarsi monitor --loop` (poll + rules, every 5 min) runs separately.

Usage: python tools/haiku_monitor.py [--once] [--interval 600] [--no-sonnet]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from infrarsi.contracts import EXECUTION_DIR, PROJECT_ROOT, utc_now  # noqa: E402

MON, ALERTS = EXECUTION_DIR / 'monitor', EXECUTION_DIR / 'alerts'
NOTES = MON / 'notes'
LOG = MON / 'haiku_monitor.jsonl'
CLAUDE = Path.home() / '.local' / 'bin' / 'claude.exe'
TOOLS = 'Read,Grep,Glob'
HAIKU, SONNET = 'claude-haiku-5-5', 'claude-sonnet-5-5'
STATE_CAP = 40_000          # characters of state.json inlined into the brief
QUIET_EVERY = 6             # with no active attempt and unchanged open alerts, call Haiku only every 6th tick

SYSTEM = ('You are the read-only experiment monitor of InfraRSI, an infra-RSI benchmark running on eight '
          'shared 8xH100 nodes (WINH0-7). You may read local files; you must not modify files or run commands. '
          'Be terse and factual; cite the field or file behind every claim.')

BRIEF = """Monitor tick {utc}. Read the inputs below (and any file under {root} if needed), compare with the
previous note, and answer with ONE JSON object and nothing else:
{{"headline": "<= 120 chars", "note_md": "<= 15 lines of markdown",
  "findings": [{{"level": "WARN|CRIT", "code": "snake_case", "run": "run_id or null", "message": "...",
                "evidence": "field/file + value", "suggested_action": "..."}}],
  "escalate": false, "escalation_question": ""}}

What is being run: probes, screens, scored runs and lifecycle checks of fixed training tasks; each attempt
writes heartbeat.json (every update; every 60 s while loading), per-rank signals and a receipt. The rules engine
(rules.py) already raised the alerts in OPEN.json and wrote SUMMARY.md; do not repeat them unless you add new
evidence. Experiment rules: exactly one seed per experiment group (input plans A/B are data orders, not seeds);
numerics tolerances vs the reference: main CE <= 0.005 relative per step, grad norm <= 0.02, window mean
<= 0.002; other users' jobs share these nodes and must never be touched (report only).

CRIT only when no rule caught it and one of these holds: a run's science may be invalid (numerics beyond
tolerance, invariant / data-order / token-count mismatch, duplicate seeds or repeats of one group); a foreign
process shares GPUs with a running scored run; a run shows no progress for more than 2x its step time plus
10 min; the queue has been paused or idle with pending work for more than 30 min; /data free space < 1 TB
(design 5.2) or the LLM budget is exceeded. Do not audit the monitor code itself unless asked; judge the runs. Otherwise use WARN (slow, straggler, idle GPUs while work is pending, repeated
failures) or only mention it in the note. Set "escalate" true only for a CRIT that needs root-cause analysis.

## SUMMARY.md
{summary}

## OPEN.json
{open_alerts}

## state.json (truncated to {cap} chars)
{state}

## Previous note
{prev_note}
"""

RCA = """Root-cause analysis requested by the Haiku monitor at {utc}.
Finding(s): {findings}
Question: {question}
Read execution/monitor/state.json, execution/alerts/, execution/queue/ and the relevant infrarsi code under
{root}. Write <= 40 lines of markdown: what happened, evidence (file + field + value), most likely cause,
recommended action for the orchestrator, and whether a running attempt should be stopped (a proposal only;
nothing is stopped automatically). Do not modify anything."""


def _read(p: Path, default=''):
    try:
        return p.read_text(encoding='utf-8')
    except FileNotFoundError:
        return default


def _trim_state(text: str) -> str:
    """Drop bulky log tails first, then hard-truncate."""
    try:
        st = json.loads(text)
        for run in st.get('runs', []) or []:
            if isinstance(run, dict) and run.get('log_tails'):
                run['log_tails'] = [t[-600:] for t in run['log_tails']]
        text = json.dumps(st, indent=1, default=str)
    except (json.JSONDecodeError, AttributeError):
        pass
    return text[:STATE_CAP]


def ask(model: str, prompt: str, timeout: int = 900, budget_usd: float = 0.5) -> dict:
    """One headless Claude CLI call with read-only tools; returns the CLI's json result object."""
    cmd = [str(CLAUDE), '-p', '--model', model, '--output-format', 'json', '--max-budget-usd', str(budget_usd),
           '--safe-mode', '--tools', TOOLS,
           '--allowedTools', TOOLS, '--strict-mcp-config', '--permission-mode', 'plan', '--permission-prompts',
           'none', '--no-session-persistence', '--disable-slash-commands', '--no-chrome', '--system-prompt', SYSTEM]
    p = subprocess.run(cmd, input=prompt.encode('utf-8'), capture_output=True, timeout=timeout, cwd=str(PROJECT_ROOT))
    out = p.stdout.decode('utf-8', 'replace').strip()
    try:
        res = json.loads(out.splitlines()[-1]) if out else {}
    except json.JSONDecodeError:
        res = {}
    res['_exit'] = p.returncode
    if p.returncode or res.get('is_error'):
        res['_stderr'] = p.stderr.decode('utf-8', 'replace')[-2000:]
    return res


def parse_verdict(text: str) -> dict:
    a, b = text.find('{'), text.rfind('}')
    v = json.loads(text[a:b + 1])
    v.setdefault('findings', [])
    for f in v['findings']:
        if f.get('level') not in ('WARN', 'CRIT'):
            f['level'] = 'WARN'
    return v


def _quiet_key() -> str | None:
    """Fingerprint of 'nothing is happening': None while any attempt is unfinished."""
    try:
        st = json.loads(_read(MON / 'state.json', '{}'))
    except json.JSONDecodeError:
        return None
    runs = st.get('runs') or st.get('attempts') or []
    if any(isinstance(r, dict) and not r.get('finished') for r in runs):
        return None
    try:
        open_keys = sorted(a.get('key', '') for a in json.loads(_read(ALERTS / 'OPEN.json', '{}')).get('open', []))
    except json.JSONDecodeError:
        open_keys = []
    return json.dumps([open_keys, sorted(str(r.get('attempt_id')) for r in runs if isinstance(r, dict))])


def tick(use_sonnet: bool) -> dict:
    utc = utc_now()
    ts = utc[:19].replace(':', '').replace('-', '')
    notes = sorted(NOTES.glob('*.md'))
    prompt = BRIEF.format(utc=utc, root=PROJECT_ROOT, cap=STATE_CAP, summary=_read(ALERTS / 'SUMMARY.md', '(none yet)'),
                          open_alerts=_read(ALERTS / 'OPEN.json', '(none yet)')[:8000],
                          state=_trim_state(_read(MON / 'state.json', '{}')),
                          prev_note=_read(notes[-1]) if notes else '(first tick)')
    res = ask(HAIKU, prompt, budget_usd=0.5)
    rec = {'utc': utc, 'model': HAIKU, 'exit': res.get('_exit'), 'cost_usd': res.get('total_cost_usd'),
           'is_error': res.get('is_error')}
    try:
        v = parse_verdict(res.get('result') or '')
    except (ValueError, json.JSONDecodeError):
        rec['error'] = (res.get('_stderr') or res.get('result') or 'no output')[-800:]
        _log(rec)
        return rec
    NOTES.mkdir(parents=True, exist_ok=True)
    (NOTES / f'{ts}.md').write_text(f'# {utc} {v.get("headline", "")}\n\n{v.get("note_md", "")}\n', encoding='utf-8')
    rec.update(headline=v.get('headline'), findings=v['findings'])
    crit = [f for f in v['findings'] if f['level'] == 'CRIT']
    for f in v['findings']:
        print(f'EVENT {f["level"]} haiku:{f.get("code")} {f.get("run") or "-"} {f.get("message", "")[:300]}', flush=True)
    if v['findings']:
        lines = [f'# Haiku monitor findings {utc}', '']
        lines += [f'- **{f["level"]}** `{f.get("code")}` run={f.get("run")}: {f.get("message")}\n  - evidence: '
                  f'{f.get("evidence")}\n  - suggested: {f.get("suggested_action")}' for f in v['findings']]
        level = 'CRIT' if crit else 'WARN'
        (ALERTS / f'HAIKU_{level}_{ts}.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if crit:
        try:
            from infrarsi.dispatch.queue import QueueStore
            QueueStore().pause('; '.join(f.get('message', '') for f in crit)[:2000], by='haiku_monitor')
            print('EVENT CRIT queue paused by haiku_monitor', flush=True)
        except Exception as e:                                           # noqa: BLE001
            print(f'EVENT CRIT queue pause failed: {e!r}', flush=True)
    if use_sonnet and (crit or v.get('escalate')):
        r = ask(SONNET, RCA.format(utc=utc, findings=json.dumps(crit or v['findings'], ensure_ascii=False),
                                   question=v.get('escalation_question') or '-', root=PROJECT_ROOT), timeout=1800,
                budget_usd=3.0)
        (ALERTS / f'RCA_{ts}.md').write_text(f'# Sonnet RCA {utc}\n\n{r.get("result") or r.get("_stderr", "")}\n',
                                             encoding='utf-8')
        rec['rca'] = {'exit': r.get('_exit'), 'cost_usd': r.get('total_cost_usd')}
        print(f'EVENT RCA written alerts/RCA_{ts}.md', flush=True)
    _log(rec)
    return rec


def _log(rec: dict):
    MON.mkdir(parents=True, exist_ok=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--interval', type=float, default=600)
    ap.add_argument('--no-sonnet', action='store_true')
    a = ap.parse_args()
    fails, quiet, last_key = 0, 0, None
    while True:
        t0 = time.time()
        key = _quiet_key()
        quiet = quiet + 1 if key is not None and key == last_key else 0
        last_key = key
        if quiet % QUIET_EVERY and not a.once:
            time.sleep(max(0.0, a.interval - (time.time() - t0)))
            continue
        try:
            rec = tick(not a.no_sonnet)
            fails = fails + 1 if rec.get('error') else 0
        except Exception as e:                                           # noqa: BLE001
            fails += 1
            _log({'utc': utc_now(), 'error': repr(e)})
        if fails >= 3:
            print(f'EVENT WARN haiku_monitor failing ({fails} ticks in a row); see {LOG}', flush=True)
        if a.once:
            return
        time.sleep(max(0.0, a.interval - (time.time() - t0)))


if __name__ == '__main__':
    os.environ.setdefault('PYTHONUTF8', '1')
    main()
