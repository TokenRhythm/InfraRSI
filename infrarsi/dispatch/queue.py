"""Run queue over design/RUN_MATRIX.yaml: dependencies (runs, gates, markers), priorities, advisory lanes,
whole-node exclusivity with reservations for multi-node runs, leases + occupancy re-check, QUEUE_PAUSE.json.

State lives in execution/queue/: state.json (per run status), markers.json (gates and markers set by the
orchestrator: G0:..., G1:..., DL:<model key>, PKG:A0, SHIM:..., ENV:..., USER_APPROVAL..., P0_CORE_COMPLETE),
QUEUE_PAUSE.json (written by monitor/escalate on CRIT; blocks new launches only).

Run states: pending -> running -> done | infeasible | invalid | failed (handlers.terminal_state); a run dependency is
satisfied only by 'done', or by 'infeasible' of an FR / ED run (design 1.6 reference fallback). Arms that v0.1 defers
(handlers.DEFERRED_ARMS) are listed as deferred, never as missing handlers, and are never launched.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

import yaml

from infrarsi.contracts import DESIGN_DIR, EXECUTION_DIR, TaskSpec, utc_now

from .handlers import DEFERRED_NOTE, INFEASIBLE_OK_ARMS, is_deferred

MATRIX = DESIGN_DIR / 'RUN_MATRIX.yaml'
QUEUE_DIR = EXECUTION_DIR / 'queue'
PRIORITY_RANK = {'P0': 0, 'P1': 1, 'P2': 2}
DAY_RANK = {'D1': 1.0, 'D1-2': 1.5, 'D2': 2.0, 'D2-3': 2.5, 'D3': 3.0, 'D3-4': 3.5, 'D4': 4.0, 'D4-5': 4.5, 'D5': 5.0,
            'D5+': 5.5, 'none': 9.0}
MARKER_PREFIXES = ('G0', 'G1', 'G2', 'G3', 'G4', 'G5', 'DL', 'PKG', 'SHIM', 'ENV', 'USER_APPROVAL', 'P0_CORE_COMPLETE')
NODE_GROUPS = {1: [[k] for k in range(8)], 2: [[0, 1], [2, 3], [4, 5], [6, 7]], 4: [[0, 1, 2, 3], [4, 5, 6, 7]],
               8: [list(range(8))]}
TERMINAL = ('done', 'infeasible', 'invalid', 'failed', 'deferred', 'skipped')


def load_matrix(path: Path = MATRIX) -> dict:
    m = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if m.get('schema') not in ('infrarsi-run-matrix-v1', 'infra' + 'smith-run-matrix-v1'):
        raise ValueError(f'unexpected run matrix schema {m.get("schema")!r}')
    for i, r in enumerate(m['runs']):
        r['_index'] = i
    m['by_id'] = {r['run_id']: r for r in m['runs']}
    return m


def parse_priority(p: str) -> dict:
    level = re.match(r'P[0-2]', str(p)).group(0)
    return {'level': level, 'rank': PRIORITY_RANK[level], 'condition': str(p)[len(level):].strip(' ()') or None,
            'default_enabled': 'not executed by default' not in str(p)}


def lane_nodes(lane: str):
    """'WINH2' -> [2]; 'WINH4+5' -> [4, 5]; 'WINH0-3' -> [0..3]; free-lane -> None; driver-PC -> 'driver'; pools -> 'external'."""
    lane = str(lane)
    if lane.startswith('driver'):
        return 'driver'
    if lane.startswith(('WIN0', 'unicom')):
        return 'external'
    m = re.search(r'WINH(\d)(?:([+-])(\d))?', lane)
    if not m:
        return None
    a = int(m.group(1))
    if m.group(2) == '+':
        return [a, int(m.group(3))]
    if m.group(2) == '-':
        return list(range(a, int(m.group(3)) + 1))
    return [a]


def is_marker(dep: str) -> bool:
    return dep.split(':', 1)[0] in MARKER_PREFIXES


def model_key_dirs() -> dict[str, str]:
    """DL:<model key> -> /data/infrarsi/models/<owner>__<name> directory name (from the task specs)."""
    from infrarsi.contracts import load_all_tasks
    out = {}
    for t in load_all_tasks().values():
        repo = t.raw['model']['repo_id']
        if not repo.startswith('local:'):
            out[t.raw['model']['key']] = repo.replace('/', '__')
    return out


class _FileLock:
    """Cross-process exclusive lock file (O_EXCL); stale after `stale_s`."""

    def __init__(self, path: Path, stale_s: float = 120.0):
        self.path, self.stale_s = Path(path), stale_s
        self._t = threading.Lock()

    def __enter__(self):
        self._t.acquire()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        end = time.monotonic() + 60
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f'{os.getpid()} {utc_now()}'.encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > self.stale_s:
                        self.path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() > end:
                    self._t.release()
                    raise TimeoutError(f'lock {self.path} held')
                time.sleep(0.2)

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)
        self._t.release()


class QueueStore:
    def __init__(self, qdir: Path = QUEUE_DIR):
        self.dir = Path(qdir)
        self.state_path, self.markers_path = self.dir / 'state.json', self.dir / 'markers.json'
        self.pause_path, self.refs_path = self.dir / 'QUEUE_PAUSE.json', self.dir / 'references.json'
        self.lock = _FileLock(self.dir / '.lock')

    def _read(self, p: Path, default):
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except FileNotFoundError:
            return default

    def _write(self, p: Path, obj):
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + f'.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding='utf-8')
        os.replace(tmp, p)

    def state(self) -> dict:
        return self._read(self.state_path, {'schema': 'infrarsi-queue-state-v1', 'runs': {}})

    def markers(self) -> dict:
        return self._read(self.markers_path, {})

    def references(self) -> dict:
        return self._read(self.refs_path, {})

    def paused(self) -> dict | None:
        return self._read(self.pause_path, None)

    def update_run(self, run_id: str, **fields) -> dict:
        with self.lock:
            st = self.state()
            rec = st['runs'].setdefault(run_id, {'status': 'pending', 'history': []})
            if 'status' in fields and fields['status'] != rec.get('status'):
                rec.setdefault('history', []).append({'utc': utc_now(), 'from': rec.get('status'), 'to': fields['status']})
            rec.update(fields)
            st['updated_utc'] = utc_now()
            self._write(self.state_path, st)
            return rec

    def set_marker(self, name: str, note: str = '', by: str = 'orchestrator') -> None:
        with self.lock:
            m = self.markers()
            m[name] = {'set_utc': utc_now(), 'by': by, 'note': note}
            self._write(self.markers_path, m)

    def clear_marker(self, name: str) -> None:
        with self.lock:
            m = self.markers()
            m.pop(name, None)
            self._write(self.markers_path, m)

    def set_reference(self, task_id: str, **fields) -> None:
        with self.lock:
            r = self.references()
            r.setdefault(task_id, {}).update(fields)
            self._write(self.refs_path, r)

    def pause(self, reason: str, lanes=None, by='monitor') -> None:
        cur = self.paused() or {'schema': 'infrarsi-queue-pause-v1', 'reasons': []}
        cur['reasons'].append({'utc': utc_now(), 'reason': reason, 'lanes': lanes, 'by': by})
        self._write(self.pause_path, cur)

    def unpause(self) -> None:
        self.pause_path.unlink(missing_ok=True)


def dep_ok(dep: str, state: dict, markers: dict, matrix: dict, downloads: dict | None = None) -> tuple[bool, str]:
    if dep in matrix['by_id']:
        st = state['runs'].get(dep, {}).get('status', 'pending')
        if st == 'done' or (st == 'infeasible' and matrix['by_id'][dep].get('arm') in INFEASIBLE_OK_ARMS):
            return True, ''
        return False, f'run {dep} is {st}'
    if dep in markers:
        return True, ''
    head = dep.split(':', 1)[0]
    if head in ('G0', 'G1', 'G2', 'G3', 'G4', 'G5') and any(k.split(':', 1)[0] == head for k in markers):
        return True, ''                                          # gate markers may be set by short name ('G1')
    if head == 'DL' and downloads is not None:
        d = downloads.get(dep.split(':', 1)[1])
        return bool(d), f'download {dep[3:]} not complete'
    return False, f'marker {dep} not set'


def plan(matrix: dict, store: QueueStore, *, include_p2: bool = False, enabled_conditional: set | None = None,
         downloads: dict | None = None, busy_nodes: set | None = None, live_free: set | None = None,
         handlers: dict | None = None) -> dict:
    """Pure scheduling decision: runnable runs (priority order) and node assignments; nothing is launched."""
    state, markers = store.state(), store.markers()
    runnable, blocked, deferred = [], [], []
    for r in matrix['runs']:
        st = state['runs'].get(r['run_id'], {}).get('status', 'pending')
        if st != 'pending':
            continue
        if handlers is not None and is_deferred(handlers.get(r['arm'])):
            deferred.append(r['run_id'])
            blocked.append({'run_id': r['run_id'], 'missing': [DEFERRED_NOTE], 'deferred': True})
            continue
        pr = parse_priority(r['priority'])
        if not pr['default_enabled'] and r['run_id'] not in (enabled_conditional or set()):
            blocked.append({'run_id': r['run_id'], 'missing': ['not executed by default']})
            continue
        if pr['level'] == 'P2' and not include_p2 and r['run_id'] not in (enabled_conditional or set()):
            blocked.append({'run_id': r['run_id'], 'missing': ['P2 disabled']})
            continue
        miss = [why for ok, why in (dep_ok(d, state, markers, matrix, downloads) for d in r.get('depends_on') or []) if not ok]
        (blocked if miss else runnable).append(r if not miss else {'run_id': r['run_id'], 'missing': miss})
    runnable.sort(key=lambda r: (parse_priority(r['priority'])['rank'], 0 if r['run_id'].startswith('E6-R-') else 1,
                                 DAY_RANK.get(str(r['planned_day']), 9.0), r['_index']))
    running = {k: v for k, v in state['runs'].items() if v.get('status') == 'running'}
    taken = set(busy_nodes or set())
    for v in running.values():
        taken |= set(v.get('node_indices') or [])
    free = set(range(8)) - taken if live_free is None else (set(live_free) - taken)
    # Runs with a handler are placed first (priority order); handler-less runs only get leftovers (display only).
    launchable = [r for r in runnable if handlers is None or handlers.get(r['arm'])]
    rest = [r for r in runnable if r not in launchable]
    reserved: set[int] = set()
    assign = []
    for group, can_launch in ((launchable, True), (rest, False)):
        for i, r in enumerate(group):
            a = _place(r, group[i + 1:], free, reserved, can_launch)
            a.update(handler=bool((handlers or {}).get(r['arm'])), launchable=can_launch and a['nodes'] is not None)
            if a['nodes']:
                free -= {int(n[4:]) for n in a['nodes']}
            assign.append(a)
    order = {r['run_id']: i for i, r in enumerate(runnable)}
    assign.sort(key=lambda a: order[a['run_id']])
    return {'utc': utc_now(), 'paused': store.paused(), 'runnable': [r['run_id'] for r in runnable], 'assign': assign,
            'blocked': blocked, 'deferred': deferred, 'running': {k: v.get('nodes') for k, v in running.items()},
            'free_nodes': sorted(free), 'reserved_nodes': sorted(reserved), 'markers': sorted(markers)}


def _place(r: dict, later: list[dict], free: set[int], reserved: set[int], can_reserve: bool) -> dict:
    """Preferred lane if free; else nodes no later run prefers (aligned groups first); else wait (and reserve)."""
    lane = lane_nodes(r['node_lane'])
    if r['gpus'] == 0 or lane == 'driver':
        return {'run_id': r['run_id'], 'nodes': [], 'where': 'driver'}
    if lane == 'external':
        return {'run_id': r['run_id'], 'nodes': None, 'where': 'external pool (not WINH)'}
    need = int(r['nodes'])
    avail = free - reserved
    wanted_later = set()
    for x in later:
        ln = lane_nodes(x['node_lane'])
        if isinstance(ln, list):
            wanted_later |= set(ln)
    pick = None
    if isinstance(lane, list) and len(lane) == need and set(lane) <= avail:
        pick = lane
    elif isinstance(lane, list) and need == 1 and lane[0] in avail:
        pick = [lane[0]]
    if pick is None:
        groups = [g for g in NODE_GROUPS.get(need, []) if set(g) <= avail]
        groups.sort(key=lambda g: (len(set(g) & wanted_later), g))
        pick = groups[0] if groups else None
    if pick is None and need <= len(avail):
        pick = sorted(avail, key=lambda k: (k in wanted_later, k))[:need]
    if pick is None:
        if need > 1 and can_reserve:       # hold what is free of the preferred group so later runs cannot starve it
            want = lane if isinstance(lane, list) and len(lane) == need else next(iter(NODE_GROUPS.get(need, [[]])), [])
            reserved |= set(want) & free
        return {'run_id': r['run_id'], 'nodes': None, 'where': f'waiting for {need} free node(s)'}
    return {'run_id': r['run_id'], 'nodes': [f'WINH{k}' for k in sorted(pick)], 'where': 'WINH',
            'lane_match': isinstance(lane, list) and set(pick) <= set(lane)}


def paused_nodes(pause: dict | None) -> set[int] | None:
    """None = no pause; set() = global pause (some reason without lanes); else the node indices paused."""
    if not pause:
        return None
    nodes = set()
    for r in pause.get('reasons') or []:
        lanes = r.get('lanes')
        if not lanes or lanes == 'all':
            return set()
        for lane in ([lanes] if isinstance(lanes, str) else lanes):
            ln = lane_nodes(lane)
            if not isinstance(ln, list):
                return set()
            nodes |= set(ln)
    return nodes


def downloads_status(remote) -> dict:
    """{model key: bool} from DOWNLOAD_DONE.json markers (read-only remote call)."""
    from infrarsi.contracts import RemotePaths
    st = remote.call('WINH0', 'download_status', models_root=RemotePaths.models, datasets_root=f'{RemotePaths.root}/datasets/raw',
                     log_dir=f'{RemotePaths.root}/logs/download', timeout=600)
    done = {name for name, e in st['entries'].items() if e.get('done')}
    return {key: d in done for key, d in model_key_dirs().items()}


def task_of(run: dict) -> TaskSpec | None:
    return TaskSpec.load(run['task']) if run.get('task') not in (None, 'none') else None
