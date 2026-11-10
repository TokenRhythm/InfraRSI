"""Parser of one flagship screen's collected outputs (AMENDMENT_EXT_003 Q3; X5_F1_ASSESSMENT §2.3).

    collect(local_dir) -> parsed         per-rank per-step scalars, completeness, launch records (pure, CPU)
    from_status_cell(status, group)      the same shape from a stored formal-54 cell (STATUS_AT_FINISH_*.json)
    window_metric(steps)                 the metric: mean over updates 17..30 of the per-step max over 32 ranks

Layout read by collect() (FlashNextRunner._collect; recursive fallbacks to the production names in brackets):
  signals/rankNN.jsonl          [<run>.signals.rank<r>/signals.jsonl]   32 per-rank signal logs (train_step rows)
  ranks/rankNN.json             [<run>.rank<r>.json]                   the trainer's per-rank result (runtime echo)
  nodes/node<n>.{exit,release,pid}.json, nodes/node<n>.log             the four node launchers' records and log tails
  attempt.json, job.json, fn_spec.json, run.json                       launch record, command + env, node map

Per rank, the checks of the production collector's signal reader (collect_family.py read_rank, signal part): every
record newline-terminated JSON with sequence == line index; identity (attempt_id, run_id = plan.run_id, rank); exactly
the train_step updates 1..30 in order; values.timing.update_seconds finite > 0; progress.horizon == 1024 and
consumed_packs == 8 x update; every values.losses entry finite with main_ce present; gradients.global_norm finite >= 0;
learning_rates.base_lr finite >= 0. Metric (collect_family.py:274-283): per update the max over the 32 ranks of
update_seconds, then statistics.mean over updates 17..30 (exactly 14), null unless 32 ranks and 4 node exits are
complete (the W family excludes update 1 from timing, outside the window). Numerics scalars are rank 0's, as the
collector's rows.rank0; the other ranks' copies of the global scalars are compared and reported.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path

RANKS, NODES, UPDATES = 32, 4, 30
WINDOW = (17, 30)
HORIZON, PACKS_PER_UPDATE = 1024, 8
SCHEMA = 'flashnext-parsed-v1'
# numerics scalar -> where it lives in a train_step row's values (numerics_gates.json 'scalars')
SCALARS = {'main_ce': ('losses', 'main_ce'), 'mtp_ce': ('losses', 'mtp_ce'), 'aux_loss': ('losses', 'main_aux_weighted'),
           'mtp_aux': ('losses', 'mtp_aux_weighted'), 'grad_norm': ('gradients', 'global_norm'),
           'lr': ('learning_rates', 'base_lr')}
GLOBAL_SCALARS = ('main_ce', 'mtp_ce', 'aux_loss', 'mtp_aux', 'grad_norm', 'lr')
VIEW_SKIP = ('microbatches', 'learning_rates.groups')       # nested per-micro / per-group detail, not scalar families
MAX_VIEW_SCALARS = 64
MAX_LOG_BYTES = 8 << 20
OOM_RE = re.compile(r'CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|cudaErrorMemoryAllocation|'
                    r'CUDA error: out of memory|NCCL WARN .*Failed to allocate|Cuda failure .*out of memory', re.I)
NCCL_RE = re.compile(r'ncclInternalError|ncclSystemError|ncclUnhandledCudaError|ncclRemoteError|NCCL error|'
                     r'Watchdog caught collective operation timeout|ProcessGroupNCCL.*(?:timed out|Timeout)|'
                     r'NCCL communicator was aborted')
_SIG_DIR = re.compile(r'\.signals\.rank(\d+)$')


def finite(v) -> bool:
    return type(v) in (int, float) and math.isfinite(v)


def _epoch(s) -> float | None:
    """A signal row's time_utc (ISO-8601, written by the frozen signal writer after the update's CPU values are read)
    as epoch seconds; None if absent or unparseable. This is the independent record clock the timer guard checks."""
    if not isinstance(s, str):
        return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except (ValueError, OSError, OverflowError):
        return None


def _read_json(path: Path | None):
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return None


def _sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def window_metric(steps: list[dict], window=WINDOW, excluded=()) -> tuple[float | None, int]:
    """(mean over the window of the per-update max over ranks, number of window updates); None unless exactly
    window length updates are present (collect_family.py:281-283, statistics.mean as the collector)."""
    a, b = int(window[0]), int(window[1])
    vals = [s['max_rank_s'] for s in steps if a <= s['update'] <= b and s['update'] not in excluded
            and finite(s.get('max_rank_s'))]
    n_expected = len([u for u in range(a, b + 1) if u not in excluded])
    return (statistics.mean(vals) if len(vals) == n_expected and vals else None), len(vals)


# ---------------------------------------------------------------- discovery
def _single(root: Path, preferred: Path | None, pattern: str, what: str, problems: list) -> Path | None:
    """The one file of a singleton artifact: the preferred path and every `pattern` match below root are candidates;
    more than one distinct file is a problem (the screen is then incomplete, never a silent pick)."""
    cands = {}
    if preferred is not None and preferred.is_file():
        cands[str(preferred.resolve())] = preferred
    for p in root.rglob(pattern):
        if p.is_file():
            cands.setdefault(str(p.resolve()), p)
    if len(cands) > 1:
        problems.append(f'{what}: {len(cands)} copies under the collection '
                        f'({sorted(str(p.relative_to(root)) for p in cands.values())[:3]})')
        return None
    return next(iter(cands.values()), None)


def discover(root: Path, run: str | None = None) -> dict:
    """Where each collected file is (X5's runner layout first, then the production names anywhere below root); every
    singleton artifact must exist at most once."""
    out = {'signals': {}, 'ranks': {}, 'nodes': {}, 'logs': [], 'problems': []}
    probs = out['problems']
    for name in ('attempt.json', 'job.json', 'fn_spec.json', 'run.json', 'materialization.json'):
        out[name.split('.')[0]] = _single(root, root / name, name, name, probs)
    sdir = root / 'signals'
    for r in range(RANKS):
        p = sdir / f'rank{r:02d}.jsonl'
        if p.is_file():
            out['signals'][r] = [p]
    for p in sorted(root.rglob('signals.jsonl')):
        m = _SIG_DIR.search(p.parent.name)
        if m and int(m.group(1)) < RANKS:
            out['signals'].setdefault(int(m.group(1)), []).append(p)
    for r in range(RANKS):
        pattern = f'{run}.rank{r}.json' if isinstance(run, str) and run else f'rank{r:02d}.json'
        p = _single(root, root / 'ranks' / f'rank{r:02d}.json', pattern, f'trainer result of rank {r}', probs)
        if p is not None:
            out['ranks'][r] = p
    for n in range(NODES):
        out['nodes'][n] = {kind: _single(root, root / 'nodes' / f'node{n}.{kind}.json', f'node{n}.{kind}.json',
                                         f'node{n}.{kind}.json', probs) for kind in ('exit', 'release', 'pid')}
    out['logs'] = sorted(p for p in root.rglob('*.log') if p.is_file())
    return out


# ---------------------------------------------------------------- one rank
def _get(values: dict, path: tuple):
    v = values
    for k in path:
        if not isinstance(v, dict):
            return None
        v = v.get(k)
    return v


def flat_scalars(values: dict, prefix: str = '', depth: int = 0) -> dict:
    """Numeric leaves of a train_step row's values (dotted names, depth <= 3), lists and VIEW_SKIP subtrees left out."""
    out = {}
    if not isinstance(values, dict) or depth > 3:
        return out
    for k, v in values.items():
        name = f'{prefix}{k}'
        if name in VIEW_SKIP:
            continue
        if isinstance(v, dict):
            out.update(flat_scalars(v, name + '.', depth + 1))
        elif finite(v) and type(v) is not bool:
            out[name] = float(v)
    return out


def read_signals(path: Path, rank: int, *, attempt_id=None, run_id=None, updates: int = UPDATES) -> dict:
    """One rank's signals.jsonl -> {'ok', 'problems', 'rows', 'failures', 'other_kinds'} (production reader checks)."""
    problems, rows, failures, other = [], [], [], {}
    expected = list(range(1, updates + 1))
    try:
        data = path.read_bytes()
    except OSError as exc:
        return {'ok': False, 'problems': [f'unreadable: {exc!r}'], 'rows': [], 'failures': [], 'other_kinds': {}}
    lines = data.split(b'\n')
    if lines and lines[-1] == b'':
        lines = lines[:-1]
    elif data:
        problems.append('incomplete signal record (no trailing newline)')
        lines = lines[:-1]
    for seq, raw in enumerate(lines):
        try:
            row = json.loads(raw)
        except ValueError:
            problems.append(f'record {seq}: not JSON')
            break
        if not isinstance(row, dict):
            problems.append(f'record {seq}: not an object')
            break
        if row.get('rank') != rank:
            problems.append(f'record {seq}: rank {row.get("rank")!r} != {rank}')
        if attempt_id is not None and row.get('attempt_id') != attempt_id:
            problems.append(f'record {seq}: attempt_id {str(row.get("attempt_id"))[:40]!r} is not this attempt')
        if run_id is not None and row.get('run_id') != run_id:
            problems.append(f'record {seq}: run_id {str(row.get("run_id"))[:60]!r} is not this run')
        if row.get('sequence') != seq:
            problems.append(f'record {seq}: sequence {row.get("sequence")!r}')
        kind = row.get('kind')
        if kind == 'failure':
            failures.append({'sequence': seq, 'values': json.dumps(row.get('values'), default=str)[:400]})
        if kind != 'train_step':
            other[str(kind)] = other.get(str(kind), 0) + 1
            continue
        u, values = row.get('completed_updates'), row.get('values')
        if type(u) is not int or u not in expected:
            problems.append(f'record {seq}: update {u!r} outside 1..{updates}')
            continue
        if not isinstance(values, dict):
            problems.append(f'update {u}: values missing')
            continue
        wall = _get(values, ('timing', 'update_seconds'))
        if not (finite(wall) and wall > 0):
            problems.append(f'update {u}: timing.update_seconds {wall!r}')
        prog = values.get('progress') if isinstance(values.get('progress'), dict) else {}
        if prog.get('horizon') != HORIZON or prog.get('consumed_packs') != u * PACKS_PER_UPDATE:
            problems.append(f'update {u}: progress {prog.get("horizon")!r}/{prog.get("consumed_packs")!r} != '
                            f'{HORIZON}/{u * PACKS_PER_UPDATE}')
        losses = values.get('losses')
        if not isinstance(losses, dict) or 'main_ce' not in losses:
            problems.append(f'update {u}: losses.main_ce missing')
            losses = losses if isinstance(losses, dict) else {}
        bad = sorted(k for k, v in losses.items() if not finite(v))
        if bad:
            problems.append(f'update {u}: non-finite losses {bad[:4]}')
        gn, lr = _get(values, ('gradients', 'global_norm')), _get(values, ('learning_rates', 'base_lr'))
        if not (finite(gn) and gn >= 0):
            problems.append(f'update {u}: gradients.global_norm {gn!r}')
        if not (finite(lr) and lr >= 0):
            problems.append(f'update {u}: learning_rates.base_lr {lr!r}')
        rec = {'update': u, 'seconds': float(wall) if finite(wall) else None, 't_utc': _epoch(row.get('time_utc'))}
        for name, where in SCALARS.items():
            v = _get(values, where)
            rec[name] = v if finite(v) else None
        rec['scalars'] = flat_scalars(values)
        rows.append(rec)
    got = [r['update'] for r in rows]
    if got != expected:
        problems.append(f'train_step updates {got[:4]}..{got[-2:]} ({len(got)}) != 1..{updates}')
    return {'ok': not problems, 'problems': problems[:12], 'n_problems': len(problems), 'rows': rows,
            'failures': failures, 'other_kinds': other}


# ---------------------------------------------------------------- external clock (NM18 P1-2)
def read_extclock(root) -> dict:
    """{by_rank: {rank: {update: t_monotonic}}, n_files, problems} from the out-of-process watcher files (NM18 P1-2,
    strict per NM21 P1-3). The runner collects exactly four canonical files extclock/node00.jsonl .. node03.jsonl (from
    the remote fn_extclock.node<r>.jsonl). ONLY those four are read; the node number comes from the canonical filename
    alone. Unconditional containment: every file must exist, hold exactly its eight ranks (r*8 .. r*8+7) with node_rank
    equal to the file's node, the complete update sequence 1..30 for each of those ranks, no duplicates, and
    t_monotonic strictly increasing with the update index. Any missing file, any extra file under extclock/, or any
    violation is a problem; timer_consistency fail-closes on a non-empty `problems`. The watcher's clock is in a
    different interpreter than the editable trainer, so an edit cannot advance it; the only subtraction the guard makes
    is of two stamps of the SAME rank (one node clock), so no cross-node comparison is made. Defense in depth - correctly
    partitioned forged rows would pass this, so the kernel-and-constants code space, which prevents an overlay from
    writing any file at all, is the primary protection."""
    root = Path(root)
    d = root / 'extclock'
    by_rank, problems = {}, []
    per_node = RANKS // NODES
    if not d.is_dir():
        return {'by_rank': {}, 'n_files': 0, 'problems': ['no extclock/ directory (a code-arm screen needs the external clock)']}
    present = sorted(p.name for p in d.glob('*') if p.is_file())
    canonical = [f'node{r:02d}.jsonl' for r in range(NODES)]
    extra = [n for n in present if n not in canonical]
    if extra:
        problems.append(f'unexpected file(s) under extclock/: {extra[:4]} (only {canonical} are read)')
    for r in range(NODES):
        f = d / f'node{r:02d}.jsonl'
        if not f.is_file():
            problems.append(f'extclock/node{r:02d}.jsonl missing')
            continue
        lo, hi = r * per_node, r * per_node + per_node
        per = {}
        try:
            raw = f.read_bytes()
        except OSError as exc:
            problems.append(f'node{r:02d}: unreadable ({exc!r})')
            continue
        malformed = False
        for line in raw.split(b'\n'):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                problems.append(f'node{r:02d}: a record is not JSON')
                malformed = True
                continue
            if not isinstance(row, dict):
                problems.append(f'node{r:02d}: a record is not an object')
                malformed = True
                continue
            rr, u, t, nr = row.get('rank'), row.get('update'), row.get('t_monotonic'), row.get('node_rank')
            if not (type(rr) is int and type(u) is int and finite(t)):
                problems.append(f'node{r:02d}: a record lacks a finite (rank, update, t_monotonic)')
                malformed = True
                continue
            if nr != r:
                problems.append(f'node{r:02d}: node_rank {nr!r} != {r}')
                malformed = True
                continue
            if not (lo <= rr < hi):
                problems.append(f'node{r:02d}: rank {rr} is not one of this node\'s ranks {lo}..{hi - 1}')
                malformed = True
                continue
            per.setdefault(rr, []).append((u, float(t)))
        if malformed:
            continue
        for rr in range(lo, hi):
            seq = per.get(rr, [])
            updates = [u for u, _ in seq]
            if updates != list(range(1, UPDATES + 1)):
                problems.append(f'node{r:02d} rank {rr}: update set {updates[:3]}..{updates[-2:] if updates else []} '
                                f'is not the complete 1..{UPDATES} (duplicates / gaps / wrong order)')
                continue
            ts = [t for _, t in seq]
            if any(ts[i] <= ts[i - 1] for i in range(1, len(ts))):
                problems.append(f'node{r:02d} rank {rr}: t_monotonic is not strictly increasing with the update')
                continue
            for u, t in seq:
                by_rank.setdefault(rr, {})[u] = t
    return {'by_rank': by_rank, 'n_files': len(present), 'problems': problems[:16]}


# ---------------------------------------------------------------- nodes, logs, launch records
def _node(rec: dict) -> dict:
    ex, rel, pid = (_read_json(rec.get(k)) for k in ('exit', 'release', 'pid'))
    problems = []
    if not isinstance(ex, dict):
        problems.append('exit file missing')
        ex = {}
    elif ex.get('exit_code') != 0 or ex.get('error'):
        problems.append(f'exit_code {ex.get("exit_code")!r} error {json.dumps(ex.get("error"), default=str)[:200]}')
    if not isinstance(rel, dict):
        problems.append('release file missing')
        rel = {}
    elif not (rel.get('idle') is True and rel.get('process_group_gone') is True and rel.get('processes') == ''):
        problems.append(f'not released clean (idle {rel.get("idle")!r}, group gone {rel.get("process_group_gone")!r}, '
                        f'processes {str(rel.get("processes"))[:80]!r})')
    if ex and rel and ex.get('hostname') != rel.get('hostname'):
        problems.append(f'exit host {ex.get("hostname")!r} != release host {rel.get("hostname")!r}')
    pid = pid if isinstance(pid, dict) else {}
    return {'ok': not problems, 'problems': problems, 'exit_code': ex.get('exit_code'), 'error': ex.get('error'),
            'hostname': ex.get('hostname') or rel.get('hostname'), 'node': ex.get('node') or pid.get('node'),
            'started_unix': pid.get('started_unix'),                  # the launch record: the launcher's own start
            'exit_started_unix': ex.get('started_unix'), 'finished_unix': ex.get('finished_unix'),
            'started_utc': pid.get('started_utc'), 'finished_utc': ex.get('finished_utc'),
            'command': pid.get('command') if isinstance(pid.get('command'), list) else None,
            'release': {k: rel.get(k) for k in ('idle', 'process_group_gone', 'processes')}}


def scan_logs(paths, root: Path) -> dict:
    """OOM / NCCL error lines in the collected launcher logs (harness.validate patterns)."""
    hits = {'oom': [], 'nccl': []}
    for p in paths:
        try:
            text = p.read_bytes()[-MAX_LOG_BYTES:].decode('utf-8', errors='replace')
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for kind, rx in (('oom', OOM_RE), ('nccl', NCCL_RE)):
                if rx.search(line) and len(hits[kind]) < 20:
                    hits[kind].append({'file': str(p.relative_to(root)).replace('\\', '/'), 'line': i, 'text': line.strip()[:240]})
    return hits


def _rank_result(path: Path | None) -> dict | None:
    r = _read_json(path)
    if not isinstance(r, dict):
        return None
    fam = r.get('infra_family') if isinstance(r.get('infra_family'), dict) else {}
    code = fam.get('bundle_code_sha256')
    return {'status': r.get('status'), 'attempt_id': r.get('attempt_id'), 'rank': r.get('rank'), 'steps': r.get('steps'),
            'resume': r.get('resume'), 'aux_readback_mode': r.get('aux_readback_mode'), 'log_run_id': r.get('log_run_id'),
            'variant': fam.get('variant'), 'training_seed': fam.get('training_seed'),
            'data_order_policy': fam.get('data_order_policy'),
            'bundle_code_sha256': _sha_obj(code) if isinstance(code, dict) else None}


def launch_records(attempt: dict | None, job: dict | None, spec: dict | None) -> dict:
    """The fields of the launch record the guards read (knob binding, frozen hashes), kept small."""
    a, j = attempt or {}, job or {}
    plan = a.get('plan') if isinstance(a.get('plan'), dict) else {}
    code = a.get('code_sha256')
    keep = ('attempt_id', 'run', 'infra_variant', 'training_seed', 'run_limit', 'screen_stop', 'resume', 'stop',
            'training_source_sha256', 'data_manifest_sha256', 'sources_sha256', 'update_supervision_sha256',
            'data_order_policy', 'systems_policy', 'g2_mode', 'replay_mode', 'precision_arm', 'allocator_policy')
    env, cmd = j.get('env'), j.get('command')
    return {'attempt': {**{k: a.get(k) for k in keep}, 'plan_run_id': plan.get('run_id'),
                        'uc_contract': isinstance(a.get('uc_contract'), dict),
                        'w_family_contract': isinstance(a.get('w_family_contract'), dict),
                        'code_sha256': code if isinstance(code, dict) else None,
                        'code_sha256_digest': _sha_obj(code) if isinstance(code, dict) else None} if attempt else None,
            'job': {'env': {str(k): v for k, v in env.items()} if isinstance(env, dict) else None,
                    'command': [str(x) for x in cmd] if isinstance(cmd, list) else None, 'name': j.get('name'),
                    'group': j.get('group'), 'master_port': j.get('master_port'),
                    'metadata_directory': j.get('metadata_directory'),
                    'timeout_seconds': j.get('timeout_seconds')} if job else None,
            'fn_spec': {k: (spec or {}).get(k) for k in ('nodes', 'hostnames', 'host_numbers', 'master_addr', 'master_port',
                                                         'timeout_s', 'out_root')} if spec else None}


# ---------------------------------------------------------------- the screen
def _view_scalars(ok_ranks: dict, updates: int, window=WINDOW) -> dict:
    """Every values.* scalar present in all ranks' rows: per update the max (and min) over ranks, then the window mean
    of the max, the max and the min over the run (identical summary for every arm, D-F1-5)."""
    names = None
    for res in ok_ranks.values():
        for row in res['rows']:
            names = set(row['scalars']) if names is None else names & set(row['scalars'])
    out = {}
    for name in sorted(names or ())[:MAX_VIEW_SCALARS]:
        per_u_max, per_u_min = {}, {}
        for res in ok_ranks.values():
            for row in res['rows']:
                v = row['scalars'][name]
                u = row['update']
                per_u_max[u] = max(per_u_max.get(u, v), v)
                per_u_min[u] = min(per_u_min.get(u, v), v)
        win = [per_u_max[u] for u in range(window[0], window[1] + 1) if u in per_u_max]
        out[name] = {'window_mean_of_rank_max': statistics.fmean(win) if win else None,
                     'max': max(per_u_max.values()), 'min': min(per_u_min.values())}
    return out


def collect(local_dir, *, updates: int = UPDATES) -> dict:
    """Never raises: a malformed collection is an incomplete screen (metric null) with its problems listed."""
    try:
        return _collect(Path(local_dir), updates)
    except Exception as exc:                          # noqa: BLE001 - a parse failure is an incomplete screen
        return {'schema': SCHEMA, 'source': {'kind': 'signals', 'dir': str(local_dir)}, 'identity': {},
                'expected': {'ranks': RANKS, 'nodes': NODES, 'updates': updates, 'window': list(WINDOW)},
                'complete': False, 'completeness': {'n_ranks_ok': 0, 'nodes_ok': False, 'rank_problems': [],
                                                    'node_problems': [], 'problems': [f'parse error: {exc!r}'[:300]]},
                'nodes': [], 'steps': [], 'excluded_timing_steps': [], 'per_step': [],
                'metric': {'value': None, 'unchecked_value': None, 'n_window': 0, 'window': list(WINDOW)},
                'scalar_consistency': {'checked_ranks': 0, 'mismatches': []}, 'rank_window_mean_s': [], 'values': {},
                'peak_reserved_gib': None, 'failure_rows': [], 'other_record_kinds': [], 'logs': {'oom': [], 'nccl': []},
                'rank_results': [None] * RANKS, 'launch': {'attempt': None, 'job': None, 'fn_spec': None},
                'ext_clock': {'by_rank': {}, 'n_files': 0, 'problems': []}, 'runner': None}


def _collect(root: Path, updates: int) -> dict:
    head = []
    attempt = _read_json(_single(root, root / 'attempt.json', 'attempt.json', 'attempt.json', head))
    attempt = attempt if isinstance(attempt, dict) else None
    run = attempt.get('run') if attempt and isinstance(attempt.get('run'), str) else None
    found = discover(root, run)
    job, spec, runinfo = (_read_json(found.get(k)) for k in ('job', 'fn_spec', 'run'))
    job = job if isinstance(job, dict) else None
    spec = spec if isinstance(spec, dict) else None
    problems = list(found['problems'])
    if attempt is None:
        problems.append('attempt.json missing or not an object: rank identity cannot be checked')
    plan = (attempt or {}).get('plan')
    if attempt is not None and not isinstance(plan, dict):
        problems.append('attempt.json plan is not an object')
        plan = {}
    attempt_id = attempt.get('attempt_id') if attempt else None
    run_id = (plan or {}).get('run_id') if attempt else None
    resume = (attempt or {}).get('resume', 0)
    if type(resume) is not int:
        problems.append(f'attempt resume {resume!r} is not an integer')
        resume = 0
    if attempt is not None and (resume != 0 or attempt.get('run_limit') != updates):
        problems.append(f'attempt resume {resume!r} run_limit {attempt.get("run_limit")!r}: a screen is resume 0, {updates} updates')
    if job is not None and (not isinstance(job.get('env'), dict) or not isinstance(job.get('command'), list)):
        problems.append('job.json env / command malformed')
    per_rank, failures = {}, []
    for r in range(RANKS):
        paths = found['signals'].get(r, [])
        if not paths:
            per_rank[r] = {'ok': False, 'problems': ['signal file missing'], 'rows': [], 'failures': [], 'other_kinds': {}}
            continue
        if len(paths) > 1:
            per_rank[r] = {'ok': False, 'problems': [f'{len(paths)} signal files for this rank'], 'rows': [], 'failures': [],
                           'other_kinds': {}}
            continue
        res = read_signals(paths[0], r, attempt_id=attempt_id, run_id=run_id, updates=updates)
        if attempt is None:
            res['ok'] = False
        per_rank[r] = res
        failures += [{'rank': r, **f} for f in res['failures']]
    nodes = [_node(found['nodes'][n]) for n in range(NODES)]
    ok_ranks = {r: res for r, res in per_rank.items() if res['ok']}
    nodes_ok = all(nd['ok'] for nd in nodes)
    complete = len(ok_ranks) == RANKS and nodes_ok and attempt is not None and not problems
    excluded = [resume + 1] if (attempt or {}).get('infra_variant') == 'W' else []
    steps = []
    if len(ok_ranks) == RANKS:
        for i, u in enumerate(range(1, updates + 1)):
            times = [(per_rank[r]['rows'][i]['seconds'], r) for r in range(RANKS)]
            mx, arg = max(times)
            mn = min(t for t, _ in times)
            # the independent record-clock gap of this update: per rank time_utc(u) - time_utc(u-1), then the max over
            # ranks (consistent with the metric's per-update max over ranks); None unless every rank's gap is finite.
            gap = None
            if i >= 1:
                pairs = [(per_rank[r]['rows'][i].get('t_utc'), per_rank[r]['rows'][i - 1].get('t_utc'))
                         for r in range(RANKS)]
                if all(finite(a) and finite(b) for a, b in pairs):
                    gap = max(a - b for a, b in pairs)
            steps.append({'update': u, 'max_rank_s': mx, 'argmax_rank': arg, 'min_rank_s': mn, 'spread_s': mx - mn,
                          'timing_eligible': u not in excluded, 'record_gap_max_rank_s': gap})
    metric, n_window = window_metric(steps, excluded=excluded) if steps else (None, 0)
    rank0 = per_rank[0]['rows'] if per_rank[0]['ok'] else []
    per_step = [{'update': row['update'], **{k: row[k] for k in SCALARS}} for row in rank0]
    mismatches = []
    if rank0:
        for r, res in ok_ranks.items():
            if r == 0:
                continue
            for a, b in zip(rank0, res['rows']):
                diff = [k for k in GLOBAL_SCALARS if a[k] != b[k]]
                if diff and len(mismatches) < 12:
                    mismatches.append({'rank': r, 'update': a['update'], 'scalars': diff})
    rank_means = []
    if len(ok_ranks) == RANKS:
        a, b = WINDOW
        for r in range(RANKS):
            vals = [row['seconds'] for row in per_rank[r]['rows'] if a <= row['update'] <= b]
            rank_means.append(statistics.fmean(vals) if vals else None)
    values_view = _view_scalars(ok_ranks, updates) if len(ok_ranks) == RANKS else {}
    peak = (values_view.get('memory.peak_reserved_gib') or {}).get('max')
    return {
        'schema': SCHEMA,
        'source': {'kind': 'signals', 'dir': str(root)},
        'identity': {'attempt_id': attempt_id, 'run_id': run_id, 'run': run, 'variant': (attempt or {}).get('infra_variant')},
        'expected': {'ranks': RANKS, 'nodes': NODES, 'updates': updates, 'window': list(WINDOW)},
        'complete': complete,
        'completeness': {'ranks_ok': sorted(ok_ranks), 'n_ranks_ok': len(ok_ranks),
                         'rank_problems': [{'rank': r, 'problems': res['problems'][:4]} for r, res in sorted(per_rank.items())
                                           if not res['ok']],
                         'n_train_step_rows': [len(per_rank[r]['rows']) for r in range(RANKS)],
                         'nodes_ok': nodes_ok, 'node_problems': [{'node': n, 'problems': nd['problems']}
                                                                 for n, nd in enumerate(nodes) if not nd['ok']],
                         'problems': problems},
        'nodes': nodes,
        'steps': steps,
        'excluded_timing_steps': excluded,
        'metric': {'value': metric if complete else None, 'unchecked_value': metric, 'n_window': n_window,
                   'window': list(WINDOW),
                   'formula': 'statistics.mean over updates 17..30 of max over 32 ranks of values.timing.update_seconds '
                              '(collect_family.py:274-283); null unless 32 ranks and 4 node exits are complete'},
        'per_step': per_step,
        'scalar_consistency': {'checked_ranks': len(ok_ranks), 'mismatches': mismatches},
        'rank_window_mean_s': rank_means,
        'values': values_view,
        'peak_reserved_gib': peak,
        'failure_rows': failures[:20],
        'other_record_kinds': [{'rank': r, 'kinds': res['other_kinds']} for r, res in sorted(per_rank.items())
                               if res['other_kinds']],
        'logs': scan_logs(found['logs'], root),
        'rank_results': [_rank_result(found['ranks'].get(r)) for r in range(RANKS)],
        'ext_clock': read_extclock(root),
        'launch': launch_records(attempt, job, spec),
        'runner': {k: (runinfo or {}).get(k) for k in ('timed_out', 'runner_stop', 'cap_s')} if isinstance(runinfo, dict) else None,
    }


# ---------------------------------------------------------------- stored formal-54 cells
def from_status_cell(status, group: str = 'A', *, path: str | None = None) -> dict:
    """A stored cell (STATUS_AT_FINISH_*.json collections.<group>.summary) in the parsed shape. The stored rows keep the
    per-update max over ranks and rank 0's scalars (collect_family.py:276-279); the metric is re-derived from them."""
    sha = None
    if not isinstance(status, dict):
        path = str(status)
        raw = Path(status).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        status = json.loads(raw.decode('utf-8-sig'))
    summ = status['collections'][group]['summary']
    rows = summ['rows']
    excluded = list(summ.get('excluded_timing_steps') or [])
    steps = [{'update': r['step'], 'max_rank_s': r['max_rank_whole_seconds'], 'timing_eligible': r['step'] not in excluded,
              'record_gap_max_rank_s': None}             # stored cells carry no time_utc (the timer guard is code-arm only)
             for r in rows]
    per_step = []
    for r in rows:
        r0 = r['rank0']
        values = {'losses': r0.get('losses') or {}, 'gradients': {'global_norm': r0.get('grad_norm')},
                  'learning_rates': {'base_lr': r0.get('base_lr')}}
        per_step.append({'update': r['step'], **{k: (_get(values, w) if finite(_get(values, w)) else None)
                                                 for k, w in SCALARS.items()}})
    metric, n_window = window_metric(steps, excluded=excluded)
    complete = bool(summ.get('complete_verified'))
    return {
        'schema': SCHEMA,
        'source': {'kind': 'stored_status', 'wave': status.get('wave'), 'group': group, 'path': path, 'sha256': sha,
                   'attempt_id': summ.get('attempt_id'), 'variant': summ.get('variant'), 'name': summ.get('name')},
        'identity': {'attempt_id': summ.get('attempt_id'), 'run_id': summ.get('run_id'), 'run': summ.get('name'),
                     'variant': summ.get('variant')},
        'expected': {'ranks': RANKS, 'nodes': NODES, 'updates': UPDATES, 'window': list(WINDOW)},
        'complete': complete,
        'completeness': {'stored_complete_verified': complete, 'nodes': summ.get('nodes')},
        'steps': steps,
        'excluded_timing_steps': excluded,
        'metric': {'value': metric if complete else None, 'unchecked_value': metric, 'n_window': n_window,
                   'window': list(WINDOW), 'stored_value': summ.get('primary_window_17_30_mean_max32')},
        'per_step': per_step,
    }
