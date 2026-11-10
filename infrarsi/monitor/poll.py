"""Read-only collection (every 5 min) into execution/monitor/state.json for the Haiku monitor and rules.py.

One gpu_status per node (staggered), one scan of /data/infrarsi/runs, leases, disk, downloads and ledger via
the first reachable node. Nothing on WINH is written, started or stopped from here.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

from infrarsi.contracts import EXECUTION_DIR, RemotePaths, load_all_tasks, utc_now

MONITOR_DIR = EXECUTION_DIR / 'monitor'
SEARCHES_DIR = EXECUTION_DIR / 'searches'           # agent SearchSession out_dirs: searches/<run_id>/
STATE_PATH = MONITOR_DIR / 'state.json'
PACKAGE_FREEZE = EXECUTION_DIR / 'G3_PACKAGE_FREEZE.json'
WEAK_ENDPOINT = MONITOR_DIR / 'weak_endpoint.json'
LLM_HEALTH = MONITOR_DIR / 'llm_health.json'


def expected_downloads() -> dict:
    """Directory name -> expected bytes, from the frozen task specs."""
    out = {}
    for t in load_all_tasks().values():
        m, d = t.raw['model'], t.raw['data']
        if not m['repo_id'].startswith('local:') and m.get('size_gb'):
            out[m['repo_id'].replace('/', '__')] = {'kind': 'model', 'expected_bytes': int(float(m['size_gb']) * 1e9)}
        src = (d.get('source') or '').split()
        if src and '/' in src[-1] and d.get('bytes'):
            out[src[-1].replace('/', '__')] = {'kind': 'dataset', 'expected_bytes': int(d['bytes'])}
    return out


def _attempt_view(snap: dict) -> dict:
    req = snap.get('request') or {}
    hb = snap.get('heartbeat') or {}
    r0 = snap.get('rank0_last') or {}
    exits = [n.get('exit') for n in snap.get('nodes', [])]
    created = req.get('created_utc')
    elapsed = None
    if created:
        import datetime as dt
        elapsed = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(created)).total_seconds()
    return {'run_id': snap['run_id'], 'attempt_id': snap['attempt_id'], 'purpose': req.get('purpose'), 'created_utc': created,
            'task_id': snap.get('task_id'), 'task_gpus': snap.get('task_gpus'), 'task_family': snap.get('task_family'),
            'nodes': req.get('nodes'), 'world_size': req.get('world_size'), 'steps': req.get('steps'),
            'wall_seconds_cap': req.get('wall_seconds_cap'), 'phase': hb.get('phase'), 'step': hb.get('step'),
            'heartbeat_age_s': snap.get('heartbeat_age_s'), 'elapsed_s': elapsed, 'finished': snap.get('finished'),
            'exit_codes': [(e or {}).get('exit_code') for e in exits], 'exit_reasons': [(e or {}).get('reason') for e in exits],
            'stop': snap.get('stop'), 'receipt_exists': snap.get('receipt_exists'), 'receipt_valid': snap.get('receipt_valid'),
            'launched_by_harness': snap.get('launched_by_harness', True), 'receipt_oom': snap.get('receipt_oom'),
            'last_step_time_s': r0.get('whole_update_s'),
            'pids': [{'node_index': n['node_index'], 'pgid': (n.get('pid') or {}).get('pgid'),
                      'hostname': (n.get('pid') or {}).get('hostname')} for n in snap.get('nodes', [])],
            'wrapper_alive_age_s': [n.get('alive_age_s') for n in snap.get('nodes', [])],
            'log_tails': [n.get('log_tail') or '' for n in snap.get('nodes', [])]}


def _queue_summary(store) -> dict:
    st = store.state()
    counts = {}
    for v in st['runs'].values():
        counts[v.get('status', 'pending')] = counts.get(v.get('status', 'pending'), 0) + 1
    out = {'counts': counts, 'statuses': {k: v.get('status') for k, v in st['runs'].items()},
           'running': {k: v.get('nodes') for k, v in st['runs'].items() if v.get('status') == 'running'},
           'paused': store.paused(), 'markers': sorted(store.markers())}
    try:
        from infrarsi.dispatch.queue import load_matrix, plan
        p = plan(load_matrix(), store)
        out['runnable'] = [a['run_id'] for a in p['assign']]
        out['runnable_waiting_nodes'] = [a['run_id'] for a in p['assign'] if a['nodes'] is None and a['where'].startswith('waiting')]
        out['matrix_total'] = len(load_matrix()['runs'])
    except Exception as e:                                             # noqa: BLE001
        out['plan_error'] = repr(e)
    return out


def _search_spend(d: Path) -> dict | None:
    """Latest cumulative LLM spend of one search: trace.json 'used' when finished, else the events.jsonl stream
    (latest cumulative 'used' snapshot if any event carries one, otherwise the sum of llm_call payloads)."""
    trace, events = d / 'trace.json', d / 'events.jsonl'
    if trace.exists():
        try:
            t = json.loads(trace.read_text(encoding='utf-8'))
            used = t.get('used') or {}
            return {'usd': float(used.get('llm_usd') or 0.0), 'tokens': int(used.get('llm_tokens') or 0),
                    'accounting_complete': bool(t.get('accounting_complete', used.get('accounting_complete', True))),
                    'source': 'trace.json', 'finished': True, 'updated_utc': t.get('ended_utc')}
        except (ValueError, TypeError):
            pass
    if not events.exists():
        return None
    usd, tok, complete, latest, last_utc = 0.0, 0, True, None, None
    for line in events.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        p = e.get('payload') or {}
        last_utc = e.get('utc', last_utc)
        if isinstance(p.get('used'), dict) and p['used'].get('llm_usd') is not None:
            latest = p['used']
        if e.get('kind') == 'llm_call':
            usd += float(p.get('llm_usd') or 0.0)
            tok += int(p.get('llm_tokens') or 0)
            if p.get('accounting_complete') is False:
                complete = False
    if latest is not None:
        usd, tok = max(usd, float(latest.get('llm_usd') or 0.0)), max(tok, int(latest.get('llm_tokens') or 0))
        complete = complete and latest.get('accounting_complete', True) is not False
    return {'usd': usd, 'tokens': tok, 'accounting_complete': complete, 'source': 'events.jsonl', 'finished': False,
            'updated_utc': last_utc}


def llm_spend(root: Path = SEARCHES_DIR) -> dict:
    out = {'root': str(root), 'searches': {}, 'total_usd': 0.0, 'total_tokens': 0, 'incomplete': []}
    if not Path(root).exists():
        return out
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir():
            continue
        dirs = [d] if ((d / 'trace.json').exists() or (d / 'events.jsonl').exists()) else sorted(x for x in d.iterdir() if x.is_dir())
        for x in dirs:
            s = _search_spend(x)
            if s is None:
                continue
            key = x.relative_to(root).as_posix()
            out['searches'][key] = s
            out['total_usd'] += s['usd']
            out['total_tokens'] += s['tokens']
            if not s['accounting_complete']:
                out['incomplete'].append(key)
    return out


def _weak_endpoint() -> dict | None:
    if not WEAK_ENDPOINT.exists():
        return None
    cfg = json.loads(WEAK_ENDPOINT.read_text(encoding='utf-8'))
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(cfg['url'].rstrip('/') + '/models', timeout=5) as r:
            ok = r.status == 200
    except Exception as e:                                             # noqa: BLE001
        return {'url': cfg['url'], 'reachable': False, 'error': repr(e)[:200], 'utc': utc_now()}
    return {'url': cfg['url'], 'reachable': ok, 'latency_s': time.monotonic() - t0, 'utc': utc_now()}


def poll(remote=None, *, store=None, stagger_s: float = 0.3, recent_hours: float = 12.0) -> dict:
    from infrarsi.dispatch.queue import QueueStore
    from infrarsi.harness.remote import default_remote
    remote = remote or default_remote()
    store = store or QueueStore()
    t0 = time.monotonic()
    st = {'schema': 'infrarsi-monitor-state-v1', 'polled_utc': utc_now(), 'errors': [], 'nodes': {}, 'runs': [],
          'downloads': {}, 'disk': None, 'leases': {}, 'ledger': {}, 'packages': None, 'weak_endpoint': None,
          'llm_health': None}
    fs_node = None
    for k in range(8):
        name = f'WINH{k}'
        try:
            g = remote.call(k, 'gpu_status', timeout=60)
            st['nodes'][name] = {'reachable': True, 'hostname': g['hostname'], 'error': g.get('error'),
                                 'compute_apps': g['compute_apps'],
                                 'gpu_mem_used_mib': [x['mem_used_mib'] for x in g['gpus']],
                                 'gpu_util_pct': [x['util_pct'] for x in g['gpus']]}
            fs_node = k if fs_node is None else fs_node
        except Exception as e:                                         # noqa: BLE001
            st['nodes'][name] = {'reachable': False, 'error': repr(e)[:300]}
            st['errors'].append(f'{name}: {e!r}'[:300])
        time.sleep(stagger_s)
    if fs_node is not None:
        def call(func, **kw):
            try:
                return remote.call(fs_node, func, timeout=900, **kw)
            except Exception as e:                                     # noqa: BLE001
                st['errors'].append(f'{func}: {e!r}'[:300])
                return None
        st['leases'] = call('read_leases', lease_dir=RemotePaths.leases) or {}
        st['disk'] = call('disk_free', path='/data')
        scans = call('scan_runs', runs_root=RemotePaths.runs, recent_hours=recent_hours) or []
        st['runs'] = [_attempt_view(s) for s in scans]
        dl = call('download_status', models_root=RemotePaths.models, datasets_root=f'{RemotePaths.root}/datasets/raw',
                  log_dir=f'{RemotePaths.root}/logs/download') or {'entries': {}, 'logs': {}}
        exp = expected_downloads()
        for name, e in dl['entries'].items():
            want = exp.get(name, {}).get('expected_bytes')
            st['downloads'][name] = dict(e, expected_bytes=want, fraction=(e['bytes'] / want) if want else None)
        st['download_logs'] = dl['logs']
        st['ledger'] = call('ledger_status', ledger_root=RemotePaths.ledger) or {}
        if PACKAGE_FREEZE.exists():
            frozen = json.loads(PACKAGE_FREEZE.read_text(encoding='utf-8'))
            st['packages'] = {p: {'frozen': sha, 'current': call('tree_digest', root=p)}
                              for p, sha in (frozen.get('packages') or {}).items()}
    _mark_ownership(st)
    st['queue'] = _queue_summary(store)
    st['weak_endpoint'] = _weak_endpoint()
    try:
        st['llm_spend'] = llm_spend()
    except OSError as e:
        st['errors'].append(f'llm_spend: {e!r}'[:300])
    if LLM_HEALTH.exists():
        st['llm_health'] = json.loads(LLM_HEALTH.read_text(encoding='utf-8'))
    st['poll_duration_s'] = round(time.monotonic() - t0, 2)
    return st


def _mark_ownership(st: dict) -> None:
    """A compute process is ours when its session id is the pgid of a live attempt's torchrun on that host."""
    own = {}
    for r in st['runs']:
        if r.get('finished'):
            continue
        for p in r['pids']:
            if p.get('pgid') and p.get('hostname'):
                own[(p['hostname'], p['pgid'])] = f'{r["run_id"]}/{r["attempt_id"]}'
    for name, n in st['nodes'].items():
        if not n.get('reachable'):
            continue
        lease = st['leases'].get(name)
        for a in n['compute_apps']:
            a['owner'] = own.get((n['hostname'], a.get('sid')))
            if not a['owner'] and lease and lease.get('owner') == 'infrarsi' and 'infrarsi' in (a.get('cmdline') or ''):
                a['owner'] = f'infrarsi lease {lease.get("run_id")} (by command line)'
        n['foreign_apps'] = [a for a in n['compute_apps'] if not a.get('owner')]
        n['lease'] = lease


def save(state: dict, path: Path = STATE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prev = path.with_name('state.prev.json')
    if path.exists():
        os.replace(path, prev)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True, default=str), encoding='utf-8')
    os.replace(tmp, path)
    return path


def load(path: Path = STATE_PATH) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
