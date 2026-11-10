"""Alerts to execution/alerts/: <level>_<ts>.md + .json per poll (de-duplicated by key with cooldowns),
OPEN.json / SUMMARY.md for the Haiku loop, QUEUE_PAUSE.json on CRIT (blocks new launches only), and
STOP_REQUEST_<attempt>.json proposals that take effect only after Opus confirms (dispatch --confirm-stop).
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

from infrarsi.contracts import EXECUTION_DIR, utc_now

ALERTS_DIR = EXECUTION_DIR / 'alerts'
COOLDOWN_S = {'INFO': 6 * 3600, 'WARN': 1800, 'CRIT': 900}


def _write(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str), encoding='utf-8')
    os.replace(tmp, p)


def _read(p: Path, default):
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return default


def escalate(alerts: list[dict], state: dict, *, alerts_dir: Path = ALERTS_DIR, store=None, now: str | None = None) -> dict:
    alerts_dir = Path(alerts_dir)
    now = now or utc_now()
    t = dt.datetime.fromisoformat(now)
    index = _read(alerts_dir / 'index.json', {})
    emitted, seen = [], set()
    for a in alerts:
        seen.add(a['key'])
        rec = index.get(a['key'])
        last = dt.datetime.fromisoformat(rec['last_emitted_utc']) if rec else None
        escalated = rec is not None and {'INFO': 0, 'WARN': 1, 'CRIT': 2}[a['level']] > {'INFO': 0, 'WARN': 1, 'CRIT': 2}[rec['level']]
        if rec is None or escalated or (t - last).total_seconds() >= COOLDOWN_S[a['level']]:
            emitted.append(a)
            index[a['key']] = {'level': a['level'], 'code': a['code'], 'first_utc': (rec or {}).get('first_utc', now),
                               'last_emitted_utc': now, 'count': (rec or {}).get('count', 0) + 1, 'message': a['message']}
        else:
            index[a['key']].update(level=a['level'], message=a['message'], last_seen_utc=now)
    for k in [k for k in index if k not in seen]:
        index[k]['resolved_utc'] = index[k].get('resolved_utc') or now
    stamp = t.strftime('%Y%m%dT%H%M%SZ')
    files = []
    for level in ('CRIT', 'WARN'):
        batch = [a for a in emitted if a['level'] == level]
        if not batch:
            continue
        md = [f'# {level} {now}', ''] + [f'- `{a["code"]}` {a["message"]}' for a in batch]
        p = alerts_dir / f'{level}_{stamp}.md'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('\n'.join(md) + '\n', encoding='utf-8')
        _write(p.with_suffix('.json'), {'schema': 'infrarsi-alerts-v1', 'level': level, 'utc': now, 'alerts': batch})
        files.append(str(p))
    info = [a for a in emitted if a['level'] == 'INFO']
    if info:
        with open(alerts_dir / 'info.log', 'a', encoding='utf-8') as f:
            for a in info:
                f.write(f'{now} {a["code"]} {a["message"]}\n')
    paused = None
    crit = [a for a in alerts if a['level'] == 'CRIT']
    if crit and store is not None:
        new = [a for a in crit if a in emitted]
        if new or not store.paused():
            batch = new or crit
            scoped = all(a.get('lanes') and a.get('lanes') != 'all' for a in batch)
            lanes = sorted({str(x) for a in batch for x in a['lanes']}) if scoped else None
            store.pause('; '.join(a['message'] for a in batch)[:2000], lanes=lanes, by='monitor.escalate')
        paused = store.paused()
    proposals = []
    for a in alerts:
        sp = a.get('stop_proposal')
        if not sp:
            continue
        p = alerts_dir / f'STOP_REQUEST_{sp["attempt_id"]}.json'
        if not p.exists():
            _write(p, {'schema': 'infrarsi-stop-request-v1', 'status': 'proposed', 'proposed_utc': now, **sp,
                       'confirm_with': f'python -m infrarsi dispatch --confirm-stop {sp["run_id"]}/{sp["attempt_id"]}'})
        proposals.append(str(p))
    open_alerts = [dict(key=k, **v) for k, v in index.items() if not v.get('resolved_utc') and v['level'] != 'INFO']
    _write(alerts_dir / 'index.json', index)
    _write(alerts_dir / 'OPEN.json', {'schema': 'infrarsi-open-alerts-v1', 'utc': now, 'open': open_alerts,
                                       'paused': paused, 'stop_proposals': proposals})
    lines = [f'# Monitor summary {now}', '', f'queue paused: {"YES - " + paused["reasons"][-1]["reason"][:300] if paused else "no"}',
             f'open CRIT: {sum(1 for a in open_alerts if a["level"] == "CRIT")}, open WARN: {sum(1 for a in open_alerts if a["level"] == "WARN")}', '']
    lines += [f'- **{a["level"]}** `{a["code"]}` {a["message"]} (since {a["first_utc"]}, x{a["count"]})'
              for a in sorted(open_alerts, key=lambda a: (a['level'] != 'CRIT', a['first_utc']))]
    (alerts_dir / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return {'emitted': emitted, 'files': files, 'paused': paused, 'stop_proposals': proposals, 'open': open_alerts}


def run_once(remote=None, *, store=None, state_path=None, alerts_dir: Path = ALERTS_DIR) -> dict:
    """poll -> save state.json -> rules -> escalate (the 5-minute cycle)."""
    from infrarsi.dispatch.queue import QueueStore, load_matrix

    from . import poll as P
    from . import rules as R
    store = store or QueueStore()
    state_path = state_path or P.STATE_PATH
    prev = P.load(state_path)
    st = P.poll(remote, store=store)
    try:
        matrix = load_matrix()
    except Exception:                                                  # noqa: BLE001
        matrix = None
    alerts = R.evaluate(st, prev, matrix)
    st['alerts'] = alerts
    P.save(st, state_path)
    return escalate(alerts, st, alerts_dir=alerts_dir, store=store)
