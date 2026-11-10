"""Alert rules of design section 5.2 over a monitor state (and the previous one for rates/growth).

Each alert: {level: INFO|WARN|CRIT, code, key, message, run?, attempt?, node?, lanes?, data?, stop_proposal?}.
key is stable across polls so escalate.py can de-duplicate. The monitor only reports; STOP proposals need Opus.
"""
from __future__ import annotations

import datetime as dt
import re

GIB = 2 ** 30
TB = 1e12
HB_WARN_MIN_S, HB_CRIT_S = 600.0, 1800.0
FIRST_STEP_S = {'dense8': 25 * 60, 'moe': 40 * 60, 'gpu32': 60 * 60, 'small': 25 * 60}
DISK_WARN, DISK_CRIT = 3 * TB, 1 * TB
BUDGET_WARN, BUDGET_CRIT = 0.90, 1.10
DOWNLOAD_STALL_S = 20 * 60
NO_PROGRESS_MIN_S, NO_PROGRESS_SLACK_S = 15 * 60, 10 * 60
LLM_USD_WARN, LLM_USD_CRIT = 1200.0, 1500.0      # design section 3: stop P1 LLM runs at $1,200; cap $1,500
DELIBERATE_STOPS = ('request', 'external', 'caller')
DAY0 = dt.date(2026, 10, 20)            # design Day 0; D1 = 2026-10-21 (Asia/Singapore)
SGT = dt.timezone(dt.timedelta(hours=8))
LOG_CRIT = [('oom', re.compile(r'CUDA out of memory|OutOfMemoryError')),
            ('nccl', re.compile(r'ncclInternalError|ncclSystemError|ncclUnhandledCudaError|ncclRemoteError|NCCL error|'
                                r'Watchdog caught collective|ProcessGroupNCCL.*(?:timed out|Timeout)')),
            ('nan_inf', re.compile(r'(?i)\b(?:loss|grad[_ ]?norm)\b[^\n]{0,40}\b(?:nan|inf)\b|non-finite|NaN detected')),
            ('traceback', re.compile(r'Traceback \(most recent call last\)')),
            ('segfault', re.compile(r'Segmentation fault|SIGSEGV'))]


def _alert(level, code, key, message, **kw):
    return dict(level=level, code=code, key=key, message=message, **kw)


def _parse(ts):
    try:
        return dt.datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def first_step_limit(run: dict) -> float:
    gpus = run.get('task_gpus') or run.get('world_size') or 8
    fam = run.get('task_family') or ''
    if gpus >= 32:
        return FIRST_STEP_S['gpu32']
    if 'moe' in fam or 'next' in fam or 'deepseek' in fam:
        return FIRST_STEP_S['moe']
    return FIRST_STEP_S['dense8'] if gpus >= 8 else FIRST_STEP_S['small']


def run_rules(state: dict, matrix: dict | None) -> list[dict]:
    out = []
    gpuh_by_run: dict[str, float] = {}
    for r in state.get('runs', []):
        tag = f'{r["run_id"]}/{r["attempt_id"]}'
        el = r.get('elapsed_s') or 0.0
        if r.get('world_size'):
            gpuh_by_run[r['run_id']] = gpuh_by_run.get(r['run_id'], 0.0) + r['world_size'] * el / 3600.0
        probe = r.get('purpose') == 'probe'
        for i, tail in enumerate(r.get('log_tails') or []):
            # an expected FR-chain probe OOM (log or receipt) and the launcher traceback it causes are INFO
            probe_oom = probe and (bool(r.get('receipt_oom')) or LOG_CRIT[0][1].search(tail) is not None)
            for code, rx in LOG_CRIT:
                m = rx.search(tail)
                if not m:
                    continue
                level = 'INFO' if (probe_oom and code in ('oom', 'traceback')) else 'CRIT'
                line = tail[max(0, m.start() - 120): m.end() + 120].replace('\n', ' ')
                out.append(_alert(level, f'log_{code}', f'log:{code}:{tag}', f'{tag} node{i}: {code} in log: ...{line}...',
                                  run=r['run_id'], attempt=r['attempt_id'], node_index=i))
        if r.get('finished'):
            continue
        if not r.get('launched_by_harness', True):
            out.append(_alert('INFO', 'unmanaged_attempt', f'unmanaged:{tag}', f'{tag}: attempt not launched by harness.run_screen '
                              '(no launch specs); liveness rules skipped', run=r['run_id'], attempt=r['attempt_id']))
            continue
        hb = r.get('heartbeat_age_s')
        step_s = r.get('last_step_time_s') or 0.0
        warn = max(HB_WARN_MIN_S, 5 * step_s)
        if hb is not None and hb > HB_CRIT_S:
            out.append(_alert('CRIT', 'heartbeat_lost', f'hb:{tag}', f'{tag}: heartbeat {hb / 60:.0f} min old (> 30 min)',
                              run=r['run_id'], attempt=r['attempt_id'],
                              stop_proposal={'reason': 'heartbeat lost > 30 min', 'attempt_id': r['attempt_id'], 'run_id': r['run_id']}))
        elif hb is not None and hb > warn:
            out.append(_alert('WARN', 'heartbeat_stale', f'hb:{tag}', f'{tag}: heartbeat {hb / 60:.1f} min old (> {warn / 60:.0f} min)',
                              run=r['run_id'], attempt=r['attempt_id']))
        if (r.get('step') or 0) < 1:                                     # no update completed yet
            lim = first_step_limit(r)
            if el > 2 * lim:
                out.append(_alert('CRIT', 'first_step_missing', f'first:{tag}', f'{tag}: no first update after {el / 60:.0f} min (limit {lim / 60:.0f})',
                                  run=r['run_id'], attempt=r['attempt_id']))
            elif el > lim:
                out.append(_alert('WARN', 'first_step_slow', f'first:{tag}', f'{tag}: no first update after {el / 60:.0f} min (limit {lim / 60:.0f})',
                                  run=r['run_id'], attempt=r['attempt_id']))
        cap = r.get('wall_seconds_cap')
        if cap and el > 1.2 * cap:
            out.append(_alert('CRIT', 'attempt_over_wall', f'wall:{tag}', f'{tag}: {el / 60:.0f} min > wall cap {cap / 60:.0f} min + 20%',
                              run=r['run_id'], attempt=r['attempt_id'],
                              stop_proposal={'reason': 'run over its wall cap by 20%', 'attempt_id': r['attempt_id'], 'run_id': r['run_id']}))
    if matrix:
        for run_id, used in gpuh_by_run.items():
            m = matrix['by_id'].get(run_id)
            if not m or not m.get('gpu_hours_budget'):
                continue
            frac = used / float(m['gpu_hours_budget'])
            if frac > BUDGET_CRIT:
                out.append(_alert('CRIT', 'budget_gpuh', f'budget:{run_id}', f'{run_id}: {used:.1f} GPUh > 110% of {m["gpu_hours_budget"]}', run=run_id))
            elif frac >= BUDGET_WARN:
                out.append(_alert('WARN', 'budget_gpuh', f'budget:{run_id}', f'{run_id}: {used:.1f} GPUh >= 90% of {m["gpu_hours_budget"]}', run=run_id))
    return out


def _expected_invalid(r: dict) -> str | None:
    """Invalid by design or by decision: FR/ED probe OOMs, red-team runs, operator/caller stops."""
    if r.get('purpose') == 'probe' and r.get('receipt_oom'):
        return 'expected probe OOM (escalation chain)'
    if str(r['run_id']).startswith('E4-'):
        return 'red-team run (invalid receipts are the point)'
    if (r.get('stop') or {}).get('kind') in DELIBERATE_STOPS:
        return f'stopped deliberately ({(r.get("stop") or {}).get("kind")})'
    return None


def finish_rules(state: dict) -> list[dict]:
    """WARN per invalid finish; CRIT for the run's lane when its last two counted finishes are both invalid."""
    out, by_run = [], {}
    for r in state.get('runs', []):
        if not r.get('finished') or r.get('receipt_valid') is not False or not r.get('launched_by_harness', True):
            if r.get('finished') and r.get('launched_by_harness', True):
                by_run.setdefault(r['run_id'], []).append(r)
            continue
        tag = f'{r["run_id"]}/{r["attempt_id"]}'
        why = _expected_invalid(r)
        out.append(_alert('INFO' if why else 'WARN', 'invalid_finish', f'invalid:{tag}',
                          f'{tag} finished with receipt valid=false' + (f' ({why})' if why else ''),
                          run=r['run_id'], attempt=r['attempt_id']))
        by_run.setdefault(r['run_id'], []).append(r)
    for run_id, rs in by_run.items():
        counted = sorted((r for r in rs if not _expected_invalid(r)), key=lambda r: r.get('created_utc') or '')
        last2 = counted[-2:]
        if len(last2) == 2 and all(r.get('receipt_valid') is False for r in last2):
            lanes = sorted({n for r in last2 for n in (r.get('nodes') or [])})
            out.append(_alert('CRIT', 'invalid_streak', f'streak:{run_id}',
                              f'{run_id}: two invalid finishes in a row ({last2[0]["attempt_id"]}, {last2[1]["attempt_id"]}); '
                              f'lane {lanes} paused so a failure loop does not burn the queue', run=run_id, lanes=lanes))
    return out


def progress_stall_rules(state: dict, prev: dict | None) -> list[dict]:
    """WARN when the update counter is stuck while the heartbeat stays fresh (phase train)."""
    out = []
    now = _parse(state.get('polled_utc'))
    prev_runs = {r['attempt_id']: r for r in (prev or {}).get('runs', [])}
    for r in state.get('runs', []):
        if r.get('finished') or not r.get('launched_by_harness', True) or r.get('step') is None:
            continue
        p = prev_runs.get(r['attempt_id'])
        same = p is not None and p.get('step') == r['step']
        r['step_since_utc'] = (p.get('step_since_utc') or (prev or {}).get('polled_utc')) if same else state.get('polled_utc')
        if r.get('phase') != 'train':
            continue
        step_s = r.get('last_step_time_s') or 0.0
        fresh = r.get('heartbeat_age_s') is not None and r['heartbeat_age_s'] <= max(HB_WARN_MIN_S, 5 * step_s)
        since = _parse(r['step_since_utc'])
        limit = max(2 * step_s + NO_PROGRESS_SLACK_S, NO_PROGRESS_MIN_S)
        if same and fresh and now and since and (now - since).total_seconds() >= limit:
            tag = f'{r["run_id"]}/{r["attempt_id"]}'
            out.append(_alert('WARN', 'no_progress', f'stall:{tag}', f'{tag}: step {r["step"]} unchanged for '
                              f'{(now - since).total_seconds() / 60:.0f} min (limit {limit / 60:.0f}) with a fresh heartbeat',
                              run=r['run_id'], attempt=r['attempt_id']))
    return out


def spend_rules(state: dict, matrix: dict | None) -> list[dict]:
    s = state.get('llm_spend') or {}
    out = []
    cap = float(((matrix or {}).get('drivers') or {}).get('llm_cost_cap_usd') or LLM_USD_CRIT)
    usd = float(s.get('total_usd') or 0.0)
    if usd >= cap:
        out.append(_alert('CRIT', 'llm_spend_cap', 'llm:spend', f'LLM spend ${usd:,.0f} >= cap ${cap:,.0f}: stop all LLM runs'))
    elif usd >= LLM_USD_WARN:
        out.append(_alert('WARN', 'llm_spend_p1_stop', 'llm:spend',
                          f'LLM spend ${usd:,.0f} >= ${LLM_USD_WARN:,.0f}: stop P1 LLM runs (design section 3)'))
    for key in s.get('incomplete') or []:
        out.append(_alert('WARN', 'llm_accounting_incomplete', f'llm:acct:{key}',
                          f'search {key} reports accounting_complete=false (cost ledger incomplete)'))
    return out


def node_rules(state: dict) -> list[dict]:
    out = []
    for name, n in state.get('nodes', {}).items():
        if not n.get('reachable'):
            out.append(_alert('WARN', 'node_unreachable', f'unreach:{name}', f'{name} unreachable: {n.get("error")}', node=name))
            continue
        foreign = n.get('foreign_apps') or []
        if foreign and n.get('lease'):
            out.append(_alert('CRIT', 'foreign_on_leased', f'foreign:{name}',
                              f'{name} leased by {n["lease"].get("run_id")} has non-project GPU processes {[a["pid"] for a in foreign]} (report only)',
                              node=name, lanes=[name]))
        elif foreign:
            out.append(_alert('INFO', 'foreign_on_free', f'foreign:{name}', f'{name}: other users\' GPU processes {[a["pid"] for a in foreign]}',
                              node=name))
    return out


def disk_rules(state: dict) -> list[dict]:
    d = state.get('disk') or {}
    free = d.get('free_bytes')
    if free is None:
        return []
    if free < DISK_CRIT:
        return [_alert('CRIT', 'disk_low', 'disk:/data', f'/data free {free / TB:.2f} TB < 1 TB (queue paused)', lanes='all')]
    if free < DISK_WARN:
        return [_alert('WARN', 'disk_low', 'disk:/data', f'/data free {free / TB:.2f} TB < 3 TB')]
    return []


def validity_rules(state: dict) -> list[dict]:
    out = []
    valid = {r['attempt_id']: r.get('receipt_valid') for r in state.get('runs', [])}
    for name, led in (state.get('ledger') or {}).items():
        for line in led.get('tail') or []:
            try:
                import json
                e = json.loads(line)
            except ValueError:
                continue
            if isinstance(e, dict) and e.get('accepted') is True and valid.get(e.get('attempt_id')) is False:
                out.append(_alert('CRIT', 'invalid_accepted', f'accept:{e.get("attempt_id")}',
                                  f'ledger {name}: attempt {e.get("attempt_id")} accepted but its receipt is valid=false'))
    for p, v in (state.get('packages') or {}).items():
        if v.get('current') != v.get('frozen'):
            out.append(_alert('CRIT', 'package_changed', f'pkg:{p}', f'package {p} digest changed after G3 ({v.get("frozen")} -> {v.get("current")})'))
    return out


def download_rules(state: dict, prev: dict | None) -> list[dict]:
    out = []
    pd = (prev or {}).get('downloads') or {}
    now = _parse(state.get('polled_utc'))
    for name, e in (state.get('downloads') or {}).items():
        if e.get('done'):
            continue
        last = pd.get(name, {})
        same = bool(last) and last.get('bytes') == e.get('bytes')
        e['unchanged_since_utc'] = (last.get('unchanged_since_utc') or (prev or {}).get('polled_utc')) if same else state.get('polled_utc')
        since = _parse(e['unchanged_since_utc'])
        if same and now and since and (now - since).total_seconds() >= DOWNLOAD_STALL_S:
            out.append(_alert('WARN', 'download_stalled', f'dl:{name}', f'download {name}: no byte growth for '
                              f'{(now - since).total_seconds() / 60:.0f} min ({e.get("bytes", 0) / 1e9:.1f} GB)'))
    for name, lg in (state.get('download_logs') or {}).items():
        if lg.get('failed'):
            out.append(_alert('WARN', 'download_failed', f'dlfail:{name}', f'download log {name} reports FAILED: {lg.get("last")}'))
    return out


def progress_rules(state: dict, prev: dict | None, matrix: dict | None, now: dt.datetime | None = None) -> list[dict]:
    out = []
    q = state.get('queue') or {}
    nodes = state.get('nodes', {})
    idle = [n for n, v in nodes.items() if v.get('reachable') and not v.get('lease') and not v.get('compute_apps')]
    waiting = q.get('runnable_waiting_nodes') or []
    runnable = [r for r in (q.get('runnable') or []) if r not in waiting]
    pidle = (prev or {}).get('idle_since') or {}
    state['idle_since'] = {n: pidle.get(n, state['polled_utc']) for n in idle}
    t = _parse(state['polled_utc'])
    long_idle = [n for n, s in state['idle_since'].items() if t and _parse(s) and (t - _parse(s)).total_seconds() > 1800]
    if long_idle and runnable and not q.get('paused'):
        out.append(_alert('WARN', 'lane_idle', 'idle:' + ','.join(sorted(long_idle)),
                          f'{long_idle} idle > 30 min while {len(runnable)} runs are runnable (e.g. {runnable[:3]})'))
    if matrix:
        now = now or dt.datetime.now(SGT)
        st = {}
        for r in matrix['runs']:
            d = str(r['planned_day'])
            mm = re.match(r'D(\d)', d)
            if not mm or r['priority'][:2] != 'P0':
                continue
            day = int(mm.group(1)) + (1 if re.match(r'D\d-\d', d) else 0)
            due = dt.datetime.combine(DAY0 + dt.timedelta(days=day + 1), dt.time(0), SGT) + dt.timedelta(hours=4)
            if now > due:
                st.setdefault(str(r['node_lane']), []).append(r['run_id'])
        # terminal outcomes that need no intervention (dispatch terminal states); 'invalid' / 'failed' stay late
        done = {k for k, v in (q.get('statuses') or {}).items() if v in ('done', 'infeasible', 'deferred', 'running', 'skipped')}
        for lane, runs in st.items():
            late = [x for x in runs if x not in done]
            if late:
                out.append(_alert('WARN', 'lane_behind', f'behind:{lane}', f'lane {lane}: {len(late)} P0 runs > 4 h behind schedule '
                                  f'(e.g. {late[:3]})', lanes=[lane]))
    return out


def llm_rules(state: dict, prev: dict | None) -> list[dict]:
    out = []
    h = state.get('llm_health') or {}
    if h.get('error_rate_30min') is not None and h['error_rate_30min'] > 0.2:
        out.append(_alert('WARN', 'llm_errors', 'llm:errors', f'LLM backend error rate {h["error_rate_30min"]:.0%} over 30 min'))
    w = state.get('weak_endpoint')
    if w is not None:
        pw = (prev or {}).get('weak_down_since')
        if not w.get('reachable'):
            state['weak_down_since'] = pw or state['polled_utc']
            t, s = _parse(state['polled_utc']), _parse(state['weak_down_since'])
            if t and s and (t - s).total_seconds() >= 3600:
                out.append(_alert('CRIT', 'weak_endpoint_down', 'weak:down', f'weak endpoint {w["url"]} unavailable for > 1 h'))
            else:
                out.append(_alert('WARN', 'weak_endpoint_down', 'weak:down', f'weak endpoint {w["url"]} unreachable: {w.get("error")}'))
    return out


def evaluate(state: dict, prev: dict | None = None, matrix: dict | None = None, now: dt.datetime | None = None) -> list[dict]:
    alerts = []
    for f in (lambda: run_rules(state, matrix), lambda: finish_rules(state), lambda: progress_stall_rules(state, prev),
              lambda: spend_rules(state, matrix), lambda: node_rules(state), lambda: disk_rules(state),
              lambda: validity_rules(state), lambda: download_rules(state, prev),
              lambda: progress_rules(state, prev, matrix, now), lambda: llm_rules(state, prev)):
        try:
            alerts.extend(f())
        except Exception as e:                                         # noqa: BLE001 - one broken rule must not hide others
            alerts.append(_alert('WARN', 'rule_error', f'rule_error:{getattr(f, "__name__", "rule")}', f'rule evaluation failed: {e!r}'))
    for e in state.get('errors') or []:
        alerts.append(_alert('WARN', 'poll_error', f'poll:{e[:60]}', f'poll error: {e}'))
    order = {'CRIT': 0, 'WARN': 1, 'INFO': 2}
    alerts.sort(key=lambda a: (order[a['level']], a['code'], a['key']))
    return alerts
