"""Builds tests/cpu/search/data/hls3_r3.json, the fixture of the round-3 tests (FABLE_HLS3_FIXES_R3; run by hand, never by
a test or at run time) from qa_trials/trials_tb:

- arms: G4U-, Q9H- and M35H2-code-env-opus-h5 as build_hls3_arms builds an hls3 arm, every launch with all its
  per-update intervals (`intervals_all`: F23's scatter) and the hypotheses' and review findings' self-proof fields
  (evidence_note, active_from_update, warmup_updates);
- probes: the proof and job-log-note lines (timestamp, text) of the patch probes F10 judged on Q9H-h5 (c2, c5, c6, c8) and
  M35H2-h5 (c1), and of their bases' runs - F13 / F21;
- jitter: the per-update timestamps of appendix A's runs (thrash, truly slow, valid) - F23's calibration;
- llm: the failed hypothesis calls of appendix B (their stream-json events; the CLI's thinking-token counters and the
  thinking blocks' signatures left out) and the incumbent's patch file - F14.

    python build_hls3_r3.py <trials_tb dir> <out json>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_hls3_arms as B  # noqa: E402

ARMS = {'G4U-code-env-opus-h5': 'G4U-code-env-opus-h5', 'Q9H-code-env-opus-h5': 'Q9H-code-env-opus-h5',
        'M35H2-code-env-opus-h5': 'M35H2-code-env-opus-h5'}                     # (M35H2-h5: F22's out-of-memory runs)
PROBES = [('Q9H-code-env-opus-h5', 'c2', 'probe', 'c0'), ('Q9H-code-env-opus-h5', 'c5', 'probe', 'c4'),
          ('Q9H-code-env-opus-h5', 'c6', 'probe', 'c4'), ('Q9H-code-env-opus-h5', 'c8', 'probe', 'c7'),
          ('M35H2-code-env-opus-h5', 'c1', 'probe', 'c0')]
JITTER = [('G4U-code-env-opus-h3', 'c2', 'thrash'), ('G4U-rnd', 'c2', 'thrash'), ('G4U-code-env-opus-h5', 'c13', 'thrash'),
          ('G4U-code-env-opus-h5', 'c8', 'thrash'), ('G4U-code-env-opus-h5', 'c1', 'thrash'),
          ('G4U-code-env-opus-h5', 'c3', 'thrash'), ('Q9H-code-env-opus-h3', 'c5', 'possible'),
          ('G4U-code-env-opus-h5', 'c11', 'slow'), ('Q9H-code-env-opus-h3', 'c1', 'slow'),
          ('Q9H-code-env-opus-h5', 'c1', 'slow'), ('Q27U-code-env-opus-h3', 'c7', 'slow'),
          ('G4U-code-env-opus-h5', 'c0', 'valid'), ('G4U-code-env-opus-h5', 'c6', 'valid'),
          ('G4U-code-env-opus-h5', 'c12', 'valid'), ('G4U-code-env-opus-h5', 'c15', 'valid'), ('G4U-rnd', 'c7', 'valid'),
          ('Q9H-code-env-opus-h5', 'c0', 'valid'), ('Q9H-code-env-opus-h5', 'c9', 'valid'),
          ('Q27U-code-env-opus-h3', 'c0', 'valid'), ('G4U-code-env-opus-h3', 'c0', 'valid'),
          ('Q9H-code-env-opus-h3', 'c0', 'valid'), ('G4U-rnd', 'c0', 'valid')]
LLM = [('G4U-code-env-opus-h5', 'hypotheses_18e63e00e994', ''), ('G4U-code-env-opus-h5', 'hypotheses_f22819980c1c', ''),
       ('G4U-code-env-opus-h5', 'hypotheses_f22819980c1c', 'retry_1'),
       ('G4U-code-env-opus-h5', 'hypotheses_1b88026a1c09', ''), ('Q9H-code-env-opus-h5', 'hypotheses_2e462c3009a2', ''),
       ('Q9H-code-env-opus-h5', 'hypotheses_2e462c3009a2', 'retry_1')]
PROOF_KEYS = ('evidence_note', 'active_from_update', 'warmup_updates', 'site')


def run_dir(root: Path, arm: str) -> Path:
    return next((root / arm / 'repo' / '.infrarsi' / 'runs').glob('*'))


_ATTEMPTS = B.attempts


def attempts_all(run: Path) -> list:
    """build_hls3_arms.attempts, each launch with every per-update interval (`intervals_all`)."""
    rows = _ATTEMPTS(run)
    full = {}
    for rec in (run / 'attempts').glob('*/*/receipt.json'):
        r = json.loads(rec.read_text(encoding='utf-8'))
        ps = [x for x in ((r.get('evidence') or {}).get('per_step') or []) if isinstance(x.get('t'), (int, float))]
        full[(r.get('candidate_id'), r.get('purpose'), r.get('utc_start'))] = [round(y['t'] - x['t'], 4)
                                                                              for x, y in zip(ps, ps[1:])]
    for a in rows:
        a['intervals_all'] = full.get((a['cid'], a['purpose'], a['utc_start']))
    return rows


def proof_fields(run: Path) -> dict:
    """mechanism (hypotheses) / issue (review findings) -> the self-proof fields the model gave."""
    out = {}
    for d in sorted((run / 'llm_calls').glob('*_*')):
        res = B.last_result(d) or {}
        for h in (res.get('hypotheses') or []) + (res.get('findings') or []):
            if isinstance(h, dict):
                key = str(h.get('mechanism') or h.get('issue') or '')[:200]
                out.setdefault(key, {k: h.get(k) for k in PROOF_KEYS if h.get(k) is not None})
    return out


def log_lines(path: Path) -> list:
    """The stamped proof and job-log-note lines of a run's log, as (time, text) (signals.scan_events reads them)."""
    from infrarsi.project.signals import PATCH_LINE_RE, perf_note
    out = []
    for ln in path.read_text(encoding='utf-8', errors='replace').splitlines():
        t, sep, text = ln.partition('\t')
        try:
            t = float(t)
        except ValueError:
            continue
        if sep and (PATCH_LINE_RE.search(text) or perf_note(text)):
            out.append([t, text])
    return out


def probes(root: Path) -> list:
    out = []
    for arm, cid, purpose, base in PROBES:
        run = run_dir(root, arm)
        logs = sorted((run / 'attempts').glob(f'*/{cid}-{purpose}-*/train.log'))
        blog = sorted((run / 'attempts').glob(f'*/{base}-{"reference" if base == "c0" else "screen"}-*/train.log'))
        rec = json.loads(sorted((run / 'attempts').glob(f'*/{cid}-{purpose}-*/receipt.json'))[0].read_text(encoding='utf-8'))
        out.append({'arm': arm, 'cid': cid, 'base': base, 'lines': log_lines(logs[0]),
                    'base_lines': log_lines(blog[0]) if blog else [],
                    'steps': (rec.get('evidence') or {}).get('steps_completed'),
                    'recorded': {k: ((rec.get('evidence') or {}).get('events') or {}).get(k)
                                 for k in ('patch_lines', 'patch_proof', 'perf_notes')}})
    return out


def jitter(root: Path) -> list:
    out = []
    for arm, cid, cls in JITTER:
        run = run_dir(root, arm)
        r = json.loads((run / 'search' / 'receipts' / f'{cid}.json').read_text(encoding='utf-8'))
        ev = r.get('evidence') or {}
        out.append({'arm': arm, 'cid': cid, 'class': cls, 'stopped_early': ev.get('stopped_early'),
                    'per_step': [[x['step'], x['t']] for x in ev.get('per_step') or [] if isinstance(x.get('t'), (int, float))]})
    return out


def llm(root: Path) -> list:
    out = []
    for arm, call, sub in LLM:
        d = run_dir(root, arm) / 'llm_calls' / call / sub if sub else run_dir(root, arm) / 'llm_calls' / call
        events = []
        for ln in (d / 'stdout.jsonl').read_text(encoding='utf-8', errors='replace').splitlines():
            try:
                e = json.loads(ln)
            except ValueError:
                continue
            if e.get('type') == 'system' and e.get('subtype') == 'thinking_tokens':
                continue
            for b in (e.get('message') or {}).get('content') or [] if e.get('type') == 'assistant' else []:
                if isinstance(b, dict) and b.get('type') == 'thinking':
                    b['signature'] = ''                     # (the encrypted thinking: no part of the answer)
            events.append(e)
        rec = json.loads((d / 'receipt.json').read_text(encoding='utf-8'))
        out.append({'arm': arm, 'call': call, 'attempt': sub or 'first', 'model': rec.get('requested_model'),
                    'error': rec.get('error'), 'parse_mode': rec.get('parse_mode'),
                    'stdout': '\n'.join(json.dumps(e, ensure_ascii=False) for e in events) + '\n'})
    files = {}
    for arm in ('G4U-code-env-opus-h5', 'Q9H-code-env-opus-h5'):
        tree = sorted((run_dir(root, arm) / 'attempts').glob('*/c0-reference-*/tree'))[0]
        files[arm] = {'train_patches.py': (tree / 'train_patches.py').read_text(encoding='utf-8')}
    return out, files


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    B.attempts = attempts_all                                   # (every interval of every launch)
    B.H3.update(ARMS)
    data = {'arms': {}}
    for name, rel in ARMS.items():
        a = B.arm(root, name, rel)
        fields = proof_fields(run_dir(root, rel))
        for call in a['calls']:
            for h in call.get('hypotheses') or []:
                h.update(fields.get(str(h.get('mechanism'))[:200]) or {})
            for f in call.get('findings') or []:
                f.update({k: v for k, v in (fields.get(str(f.get('issue'))[:200]) or {}).items()})
        res = json.loads(next((root / rel).glob('*__result.json')).read_text(encoding='utf-8'))
        a['hls3_bundles'] = [{k: b.get(k) for k in ('bundle', 'base', 'candidate', 'origin', 'status')}
                             for b in ((res.get('search') or {}).get('hls3') or {}).get('bundles') or []]
        data['arms'][name] = a                          # (hls3_bundles: the base each memory bundle ran on)
    data['probes'] = probes(root)
    data['jitter'] = jitter(root)
    data['llm'], data['files'] = llm(root)
    out.write_text(json.dumps(data, indent=0, sort_keys=True, default=str, ensure_ascii=False) + '\n', encoding='utf-8')
    print('wrote', out, out.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
