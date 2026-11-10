"""FlashNextRunner: the F1 EnvRunner (AMENDMENT_EXT_003). One 30-step screen of the flagship production path on four
whole leased WINH nodes -> an ext-receipt-v1 (assembled and signed by the ext.common.env_runner base).

    run(task, point, None, purpose, run_id, candidate_id=, wall_cap=, cap_s=, role=, reference_parsed=) -> receipt

Per attempt:
  1. The S_F1 point is validated first (switch_space.validate); a point outside the space never takes a node: signed
     receipt, zero GPU cost, `invalid_proposal`. A runner that still holds a node from an unverified cleanup refuses
     every later screen the same way (`nodes_held`).
  2. Cap: 1,800 s (Q3); 3,600 s only for the two reference screens (purpose 'reference' on U+C or F-HT = S_match) when
     the caller asks for it; never more than the admitted wall_cap.
  3. Bundle (no lease held): variant.build_bundle on the re-rooted custom runtime, variant.ext_bind (single-group phase
     gate, the screen's cap as the launcher timeout, LP_CACHE_ROOT), per-node fn_spec.json + fn_node<r>.sh, upload to
     the bundle's metadata directory (shared /data).
  4. Four-node lease, all or nothing: every node probed (no foreign compute process, no unexplained GPU memory, no
     lease held by someone else), then leases/WINH<k>.json (owner infrarsi-ext, all four nodes named) taken in node
     order; any failure releases what was taken and the screen is not launched (OccupiedError -> invalid receipt).
  5. Launch node/fn_launch_node.py on the four nodes (detached; INFRARSI_TOKEN in its environment and inherited by
     torchrun and the 32 ranks); poll the node exit files. The launchers stop themselves at one screen-wide deadline
     (first launcher start + cap - stop reserve, so that first launcher start -> last node exit <= cap including the
     TERM/KILL sequence), on a peer failure or on STOP.json; the runner writes STOP.json at the cap as a backstop and
     gives up stop_margin_s later. In `finally`, on all four nodes at once and in rounds until verified or
     CLEANUP_BUDGET_S: SIGKILL this screen's job group and token carriers, then (once the job is gone) the launcher's
     session, which holds the host locks - each group only when a live member carries this screen's token; re-list
     both; probe the GPUs (no process with the token or a cwd / command line under the flashnext run directories). A
     node is clean only when all of it ran and came back empty; any other node keeps its lease (held_nodes).
  6. Leases are released with retries, in parallel, within RELEASE_BUDGET_S (release_status: removed / absent; a
     foreign holder is never touched); a lease whose release is not proven stays in held_nodes. Every node call has
     its own timeout, capped by the time left, and no round or attempt starts after its deadline, so with a working
     transport lease-held time <= cap + screen_grace_s; a node whose transport fails can add the transport's own retry
     bound to one call in flight (WinhRemote: 3 tries x (connect <= 45 s + session <= 20 s + the call's timeout) +
     12 s backoff, about 4-5 min), which the receipt charges like any other held second. GPU cost = 8 GPUs x the seconds each node lease was held, from
     its acquisition to its proven release; a node still leased is charged up to the receipt, listed as open-ended,
     and the receipt says accounting_complete false.
  7. Collect to <out_dir>/attempts/<cand>-<attempt>/: the 32 signals.jsonl and rank receipts, node exit / release / pid
     files, node log tails, attempt / job / fn_spec, the materialization receipt; delete bulky outputs on WINH. Each
     file is read on the node, lzma-compressed there (~86x on the signal logs) and returned in replies of at most
     COLLECT_BUDGET_BYTES compressed (below the 2 MiB SSH window: harness/remote.py WinhRemote._exec returns on the
     exit status, which can overtake the last MBs of a larger stdout), and must decompress to its sha256 computed on
     the node. A file too large for one reply comes over SFTP against the same sha256. Read-only on WINH.
  8. Judge: X6's pipeline - parse.collect(local_dir), judge.evaluate(parsed, reference_parsed, gates, role=...)
     (judge_valid, numerics_class, reasons, metric_s, details), guards.check(parsed, point, cap_s=...) (the Q3 hard
     checks) and judge.receipt_extra(verdict) (receipt fields). The runner adds only what it alone observes:
     process_cleanup, runner_timeout and lease_release. Guards veto only a run the judge accepted; otherwise they are
     reported.
Admission (SearchSession._admission): bounded_cap / worst_case_gpu_hours as agent.harness_runner.HarnessScreenRunner,
on the 1,800 s screen cap with screen_grace_s after it; pair_wall_cap and stop_grace_s are 0 (Q5: each arm's locked pair
runs right after its search, outside its budget, as the ext base class). Driver-side code: no torch import.

Code overlays (setting 2, only for a task of the code arm, code_space.is_code_task): step 1 also runs code_space.validate
on the overlay (the hooks' validator; an error = invalid_proposal, no node); step 3 builds the bundle from a per-attempt
copy of the local variant runtime with the overlay files replaced (code_space.apply, in the attempt's temporary build
directory; the local _variant_runtime and the deployed FN_SRC runtime are never written) and checks that the bundle's
code inventory is exactly the manifest's with the overlay files replaced; the bundle's code/ directory, uploaded to
FN_ROOT/metadata/<name>/code (shared /data) and hash-verified, is what the job runs. The judge's frozen_hashes guard gets
the overlay's file hashes (every other file must still be the frozen one), the candidate digest includes the overlay
digest, and the receipt carries `code_overlay` (code_space.record; None for a code-arm screen without an overlay). Any
other task keeps refusing an overlay ('F1 has no file actions (switches only)') and its receipts are unchanged.
"""
from __future__ import annotations

import base64
import hashlib
import json
import lzma
import shlex
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ...contracts import digest, new_attempt_id, utc_now
from ...harness import signing
from ...harness.leases import OccupiedError, make_lease
from ..common import guards as cg
from ..common import node_ext
from ..common.env_runner import EnvRun, EnvRunner, jsonable
from ..common.remote_exec import OWNER, is_ours, new_token, node_lease_path, occupancy_problems, q
from . import code_space as C
from . import switch_space as S
from . import variant as V

NODE_DIR = Path(__file__).resolve().parent / 'node'
LAUNCHER = 'fn_launch_node.py'
DEFAULT_NODES = ('WINH0', 'WINH1', 'WINH3', 'WINH4')
ALT_NODES = ('WINH1', 'WINH3', 'WINH4', 'WINH6')
# reports/low_precision_winh_20261014/launch_lp_node.py:7-8 (physical WINH k -> hostname / internal IP)
HOSTNAMES = ['is-ddkt5zq62vhpqy3a-devmachine-0', 'is-ddlhqgltfd2k3e5p-devmachine-0', 'is-ddmenq5g4h3txhhx-devmachine-0',
             'is-ddmenxvc7gwyrrln-devmachine-0', 'is-ddmen6iq3r252hfg-devmachine-0', 'is-ddmeodo6ak3rab5j-devmachine-0',
             'is-ddmeoiewou2jf53g-devmachine-0', 'is-ddmeolyc4esxbguc-devmachine-0']
IPS = ['172.27.7.98', '172.27.96.243', '172.27.72.155', '172.27.67.6', '172.27.25.20', '172.27.14.136', '172.27.54.119',
       '172.27.15.180']
CAP_S = 1800                  # AMENDMENT_EXT_003 Q3: launcher start -> last node exit
COLD_REFERENCE_CAP_S = 3600   # the two reference screens, if the compile cache turns out cold
REFERENCE_RECIPES = ('U_plus_C', 'S_match')   # the two reference screens: U+C and F-HT = S_match
WORLD = 32
GPUS = tuple(range(8))
GPUS_PER_NODE = 8
DEVIATIONS = ('D-F1-1', 'D-F1-2', 'D-F1-3', 'D-F1-4', 'D-F1-5', 'D-F1-6')
BULKY_BYTES = 256 << 20       # outputs larger than this are deleted on WINH after collection (none expected at 30 steps)
STOP_RESERVE_S = 60.0         # launchers stop at first start + cap - this: TERM 20 s + KILL 20 s + poll fit inside
# The formal queue's and the historical launcher's host locks, also held by node/fn_launch_node.py (same tuple there)
LEGACY_LOCK_DIRS = ('/data/low_precision_winh_20261014/locks',
                    '/data/zehua_flashnext_replay_20261013/source_tree/qwen-sft/runs/h100_scale_20260923_v1/locks')
MIN_CAP_S = 2 * STOP_RESERVE_S
RELEASE_ATTEMPTS, RELEASE_BACKOFF_S = 4, 1.0
SWEEP_SETTLE_S = 2.0
COLLECT_ATTEMPTS = 3          # failed replies (or sha256 mismatches) in a row before the collection gives up
LOG_TAIL_BYTES = 256 << 10    # node / launch logs keep their last 256 KiB (as before)
COLLECT_BUDGET_BYTES = 1 << 20  # compressed bytes per reply: base64 ~1.4 MB, below the 2 MiB SSH window
COLLECT_PRESET = 1            # lzma preset: ~86x on the signal logs, ~0.07 s per 4.4 MB file
COLLECT_CALL_TIMEOUT_S = 300.0
COLLECT_MARK = 'FNCOLLECT'

# Node-side collection (python3 -c, stdlib, read-only). argv: base64 JSON {files: [[path, tail_bytes], ...], budget,
# preset} ('-': the same on stdin). In order, for each file: the whole file (tail_bytes 0) or its last tail_bytes, its sha256, size and
# mtime, and its lzma-compressed bytes; stops before the file that would push the compressed total past `budget` (a first file larger
# than the budget is returned as too_big: hash only, fetched by SFTP); a missing file is reported as missing.
COLLECT_PY = r'''
import base64, hashlib, json, lzma, os, sys
req = json.loads(base64.b64decode(sys.stdin.read() if sys.argv[1] == '-' else sys.argv[1]))
out, used = [], 0
for path, tail in req['files']:
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as f:
            if tail and size > tail:
                f.seek(size - tail)
            data = f.read()
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        out.append({'path': path, 'missing': True})
        continue
    blob = lzma.compress(data, preset=req['preset'])
    entry = {'path': path, 'size': size, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
             'mtime': os.stat(path).st_mtime}
    if len(blob) > req['budget']:
        if out:
            break
        entry['too_big'] = True
        out.append(entry)
        break
    if used + len(blob) > req['budget']:
        break
    entry['xz'] = base64.b64encode(blob).decode('ascii')
    out.append(entry)
    used += len(blob)
print('FNCOLLECT' + json.dumps({'files': out}))
'''
EXTCLOCK_POLL_S = 0.1         # the external-clock watcher polls each rank's signals.jsonl this often (NM18 P1-2)
# fn_extclock watcher (NM18 P1-2): a stdlib process the runner starts on each node (from fn_node<r>.sh, before the
# frozen launcher) for a CODE-ARM screen only. It tails the node's 8 local ranks' signals.jsonl and appends, to its
# own file under the run's output root, (node_rank, rank, update, its own time.monotonic(), wall) the first time it
# sees each train_step record. Its interpreter is not the editable trainer runtime, so an edit cannot advance its
# clock (signals time_utc is written inside the trainer and can be forged in lockstep with the timer). Uploaded per
# screen (its sha256 is recorded in the receipt), never pinned: no frozen_manifest / preflight / launcher change.
EXTCLOCK_PY = r'''
import base64, json, os, sys, time
cfg = json.loads(base64.b64decode(sys.argv[1]))
life, run, out = cfg['life'], cfg['run'], cfg['out']
first, nproc = int(cfg['first_rank']), int(cfg['nproc'])
node_rank, stop_update = int(cfg['node_rank']), int(cfg['stop_update'])
poll = float(cfg.get('poll_s', 0.1))
deadline = time.monotonic() + float(cfg['deadline_s'])
ranks = list(range(first, first + nproc))
paths = {r: '%s/%s.signals.rank%d/signals.jsonl' % (life, run, r) for r in ranks}
offs = {r: 0 for r in ranks}
seen = {r: set() for r in ranks}
try:
    os.makedirs(os.path.dirname(out), exist_ok=True)
except OSError:
    pass
sink = open(out, 'a', buffering=1)


def emit(rank, update):
    rec = {'node_rank': node_rank, 'rank': rank, 'update': update,
           't_monotonic': time.monotonic(), 't_wall': time.time()}
    sink.write(json.dumps(rec, sort_keys=True) + '\n')
    sink.flush()
    try:
        os.fsync(sink.fileno())
    except OSError:
        pass
    seen[rank].add(update)


def poll_rank(r):
    try:
        size = os.path.getsize(paths[r])
    except OSError:
        return
    if size <= offs[r]:
        return
    try:
        fh = open(paths[r], 'rb')
    except OSError:
        return
    with fh:
        fh.seek(offs[r])
        data = fh.read()
    cut = data.rfind(b'\n')
    if cut < 0:
        return
    offs[r] += cut + 1
    for raw in data[:cut].split(b'\n'):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(row, dict) or row.get('kind') != 'train_step':
            continue
        u = row.get('completed_updates')
        if isinstance(u, int) and u not in seen[r]:
            emit(r, u)


while True:
    for r in ranks:
        poll_rank(r)
    if all(stop_update in seen[r] for r in ranks) or time.monotonic() >= deadline:
        break
    time.sleep(poll)
sink.close()
'''
EXTCLOCK_SHA = hashlib.sha256(EXTCLOCK_PY.encode()).hexdigest()
EXTCLOCK_NAME = 'fn_extclock.py'
SWEEP_DRAIN_S = 10.0          # the sweep waits this long for the job to go before it kills the lock-holding launcher
CALL_TIMEOUT_S = 30.0         # one node call (the transport may retry it; the deadlines below bound the wait)
CLEANUP_BUDGET_S = 120.0      # kill + verification, every round and node (nodes in parallel); then: kept
RELEASE_BUDGET_S = 40.0       # lease release, every node in parallel; then: unresolved (kept)
LAUNCH_ALLOWANCE_S = 20.0     # lease acquisition and launch calls before the job starts
SWEEP_MARK = 'FNSWEEP'
COST_BASIS = ('8 GPUs x seconds each node lease was held, from its acquisition to its proven release; a node still '
              'leased is charged up to this receipt and listed in open_ended_nodes')

# Node-side cleanup (python3 -c, stdlib). argv: token, settle seconds, drain seconds, the node's pid file ('-' for none:
# its 'pid' is the job's process group, start_new_session), the launcher's pid (0 for unknown: node_ext.launch runs it
# under setsid, so it is also its session's process group). Phase 1 SIGKILLs the job: each such group except the
# launcher's in which a live member carries INFRARSI_TOKEN=<token> (a process group cannot be joined from outside its
# session and its id is not reused while it has members, so such a group is this screen's), and every token carrier
# outside the launcher's group; phase 2 waits up to `drain` seconds for all of it to go, so that the launcher, which
# holds the host locks, dies last; phase 3 SIGKILLs the launcher's group (same token rule) and every remaining token
# carrier - unless part of the job survived SIGKILL that long: then the launcher is spared (it keeps the host locks;
# everything left is reported, so the node stays leased); then it waits `settle` seconds and lists what is left: live (non-zombie) members of the two groups, with or
# without the token, and token carriers. Nothing else is signalled.
SWEEP_PY = r'''
import json, os, signal, sys, time
token, settle, drain, pidfile, launcher = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], int(sys.argv[5])
if len(token) < 8:
    raise SystemExit('refusing an empty or short token')
job = 0
if pidfile != '-':
    try:
        with open(pidfile) as f:
            value = json.load(f).get('pid')
        job = value if isinstance(value, int) and value > 1 else 0
    except FileNotFoundError:
        pass
launcher = launcher if launcher > 1 else 0
groups = sorted({g for g in (job, launcher) if g})
own = {os.getpid(), os.getpgid(0)}
VAR = b'INFRARSI_TOKEN='


def tok(pid):
    try:
        with open('/proc/%d/environ' % pid, 'rb') as f:
            for item in f.read().split(b'\0'):
                if item.startswith(VAR):
                    return item[len(VAR):].decode('utf-8', 'replace')
    except OSError:
        pass
    return None


def scan():
    members, carriers = {}, {}
    for name in os.listdir('/proc'):
        if not name.isdigit() or int(name) in own:
            continue
        pid = int(name)
        try:
            with open('/proc/%d/stat' % pid) as f:
                fields = f.read().rsplit(')', 1)[1].split()
            state, pgrp = fields[0], int(fields[2])
        except (OSError, ValueError, IndexError):
            continue
        if state in ('Z', 'X'):
            continue
        if pgrp in groups:
            members.setdefault(pgrp, []).append(pid)
        if tok(pid) == token:
            carriers[pid] = pgrp
    return members, carriers


def kill(members, carriers, spare):
    done = []
    for g, pids in sorted(members.items()):
        if g != spare and g not in own and any(tok(p) == token for p in pids):
            try:
                os.killpg(g, signal.SIGKILL)
                done.append(-g)
            except OSError:
                pass
    for pid, pgrp in sorted(carriers.items()):
        if not (spare and pgrp == spare):
            try:
                os.kill(pid, signal.SIGKILL)
                done.append(pid)
            except OSError:
                pass
    return done


def left_outside(spare):
    members, carriers = scan()
    return [p for g, pids in members.items() if g != spare for p in pids] + [p for p, g in carriers.items() if g != spare]


members, carriers = scan()
killed = kill(members, carriers, launcher)
end = time.time() + drain
while launcher and left_outside(launcher) and time.time() < end:
    time.sleep(0.5)
spared = bool(launcher and left_outside(launcher))
if not spared:
    members, carriers = scan()
    killed += kill(members, carriers, None)
time.sleep(settle)
members, carriers = scan()
print('FNSWEEP' + json.dumps({'killed': killed, 'token_pids': sorted(carriers)[:64], 'groups_checked': groups,
                              'spared_launcher': spared,
                              'groups': {str(g): sorted(p)[:64] for g, p in members.items()}}))
'''


def node_index(node: str) -> int:
    k = int(str(node).upper().removeprefix('WINH'))
    if not 0 <= k < len(HOSTNAMES):
        raise ValueError(f'unknown node {node!r}')
    return k


def node_spec(nodes, *, master_port: int, timeout_s: float, out_root: str = V.FN_ROOT + '/runs',
              locks: str = V.FN_LOCKS, stop_reserve_s: float = STOP_RESERVE_S) -> dict:
    """fn_spec.json: the node map the launchers use (rank r runs on nodes[r]; rank 0 is the rendezvous leader)."""
    nodes = [str(n).upper() for n in nodes]
    if len(nodes) != 4 or len(set(nodes)) != 4:
        raise ValueError(f'a screen needs four distinct nodes, got {nodes}')
    idx = [node_index(n) for n in nodes]
    return {'schema': 'flashnext-node-spec-v1', 'nodes': nodes, 'hostnames': [HOSTNAMES[k] for k in idx],
            'ips': [IPS[k] for k in idx], 'host_numbers': [k + 1 for k in idx], 'master_addr': IPS[idx[0]],
            'master_port': int(master_port), 'timeout_s': float(timeout_s), 'stop_reserve_s': float(stop_reserve_s),
            'out_root': out_root, 'locks': locks}


def layout_from_records(attempt: dict, job: dict) -> dict:
    """variant.remote_layout from a collected attempt.json + job.json (no bundle needed): where a screen wrote on WINH."""
    life, run = attempt['lifecycle_run_root'], attempt['run']
    return {'name': job['name'], 'run': run, 'attempt_id': attempt['attempt_id'], 'metadata': job['metadata_directory'],
            'output_root': job['output_root'], 'lifecycle_run_root': life, 'job': job['metadata_directory'] + '/job.json',
            'signals': [f'{life}/{run}.signals.rank{r}/signals.jsonl' for r in range(32)],
            'rank_receipts': [f'{life}/{run}.rank{r}.json' for r in range(32)]}


def master_port(attempt_id: str) -> int:
    """Deterministic rendezvous port in [43000, 47000) (away from the formal 48270/48271 and the ext tools' ports)."""
    return 43000 + int(hashlib.sha256(attempt_id.encode()).hexdigest()[:8], 16) % 4000


def node_script(src: str, job: str, rank: int, spec: str, *, watcher: dict | None = None) -> str:
    """The script a node runs; a repeated launch (e.g. a transport retry) is a no-op (mkdir marker next to fn_spec).
    watcher (code arm only): {'path', 'log', 'cfg'} starts the external-clock watcher in the background before exec'ing
    the launcher. watcher=None reproduces the switch-only script byte-for-byte."""
    marker = spec.rsplit('/', 1)[0] + f'/.launched{int(rank)}'
    lines = ['#!/usr/bin/env bash', 'set -u',
             f'mkdir {q(marker)} 2>/dev/null || {{ echo "rank {int(rank)} was already launched"; exit 0; }}']
    if watcher is not None:
        args = base64.b64encode(json.dumps(watcher['cfg'], sort_keys=True).encode()).decode('ascii')
        lines.append(f'/usr/bin/python3 -B {q(watcher["path"])} {q(args)} >{q(watcher["log"])} 2>&1 &')
    lines.append(f'exec /usr/bin/python3 -B {q(src + "/node/" + LAUNCHER)} {q(job)} {int(rank)} {q(spec)}')
    return '\n'.join(lines) + '\n' 


def sweep_command(token: str, pidfile: str | None, launcher_pid: int | None = None, settle_s: float = SWEEP_SETTLE_S,
                  drain_s: float = SWEEP_DRAIN_S) -> str:
    if pidfile is not None and not str(pidfile).startswith(V.FN_ROOT + '/'):
        raise ValueError(f'pid file outside the flashnext root: {pidfile}')
    launcher = int(launcher_pid) if isinstance(launcher_pid, int) and launcher_pid > 1 else 0
    return (f'python3 -c {q(SWEEP_PY)} {q(token)} {float(settle_s):g} {float(drain_s):g} {q(pidfile or "-")} {launcher}')


def parse_sweep(res: dict) -> dict:
    out = (res or {}).get('stdout') or ''
    if (res or {}).get('code') != 0 or SWEEP_MARK not in out:
        raise RuntimeError(f'sweep exit {(res or {}).get("code")}: {((res or {}).get("stderr") or "")[-300:]}')
    return json.loads(out.split(SWEEP_MARK, 1)[1].strip().splitlines()[0])


def sweep_args(cmd: str) -> tuple[str, str, int]:
    """(token, pid file, launcher pid) of a sweep_command (for simulated nodes)."""
    parts = shlex.split(cmd)
    return parts[3], parts[6], int(parts[7])


class FlashNextRunner(EnvRunner):
    env = 'flashnext'
    deviations = DEVIATIONS
    # SearchSession admission API (as ext.common.env_runner.EnvRunner): the locked pair is not part of an F1 arm's budget
    # (AMENDMENT_EXT_003 Q5), so no pair time is reserved: SearchSession reserves 2 x (pair_wall_cap + stop_grace_s) of
    # wall and 2 x worst_case_gpu_hours(world, pair_wall_cap) of GPU time, both 0 here.
    stop_grace_s = 0.0
    # one screen's allowance after its cap (bounded_cap / worst_case_gpu_hours): the STOP backstop margin (60 s), the
    # cleanup and release deadlines, the lease + launch calls
    screen_grace_s = 60.0 + CLEANUP_BUDGET_S + RELEASE_BUDGET_S + LAUNCH_ALLOWANCE_S
    KILL_WAITS = (0.0, 5.0, 10.0, 20.0, 30.0)   # a SIGKILLed 32-rank job with pinned host memory leaves nvidia-smi slowly

    def __init__(self, ops_by_node: dict, *, nodes=DEFAULT_NODES, runtime_dir, cap_s: float = CAP_S,
                 reference_parsed: dict | None = None, gates: dict | None = None, out_dir=None, src: str = V.FN_SRC,
                 build_fn=None, judge_fn=None, poll_s: float = 15.0, stop_margin_s: float = 60.0,
                 stop_reserve_s: float = STOP_RESERVE_S, keep_remote: bool = False, clock=time.monotonic,
                 sleep=time.sleep, log=None):
        self.nodes = [str(n).upper() for n in nodes]
        node_spec(self.nodes, master_port=43000, timeout_s=1)               # validates the node map
        missing = [n for n in self.nodes if n not in ops_by_node]
        if missing:
            raise ValueError(f'no NodeOps for {missing}')
        self.ops_by_node = ops_by_node
        super().__init__(ops_by_node[self.nodes[0]], gpus=GPUS, out_dir=out_dir, clock=clock, log=log)
        self.node = '+'.join(self.nodes)
        self.runtime_dir = str(runtime_dir)
        self.cap_s = float(cap_s)
        self.reference_parsed, self.gates = reference_parsed, gates
        self.src = src
        self.build_fn = build_fn or self._build
        self.judge_fn = judge_fn
        self.poll_s, self.stop_margin_s, self.keep_remote = poll_s, stop_margin_s, keep_remote
        self.screen_grace_s = float(stop_margin_s) + CLEANUP_BUDGET_S + RELEASE_BUDGET_S + LAUNCH_ALLOWANCE_S
        self.stop_reserve_s = float(stop_reserve_s)
        self.sleep = sleep
        self.history: list[dict] = []
        self.held_nodes: list[str] = []          # nodes left leased: cleanup or lease release not proven

    # ------------------------------------------------------------ admission (SearchSession._admission)
    def _screen_cap(self) -> float:
        return min(float(self.cap_s), float(CAP_S))

    def worst_case_gpu_hours(self, world: int, wall_cap: float | None = None) -> float:
        cap = self._screen_cap() if wall_cap is None else float(wall_cap)
        return int(world) * (cap + (self.screen_grace_s if cap else 0.0)) / 3600.0

    def bounded_cap(self, gpu_hours_left: float, world: int, wall_left: float, expected_s: float) -> int | None:
        """Largest wall cap whose worst case fits the GPU-hours and wall left; None when shorter than the expected
        screen (a screen cut by its cap is wasted)."""
        cap = min(self._screen_cap(), gpu_hours_left * 3600.0 / max(1, int(world)) - self.screen_grace_s,
                  wall_left - self.screen_grace_s)
        return int(cap) if cap >= max(60.0, float(expected_s)) else None

    def pair_wall_cap(self, expected_s: float) -> int:
        return 0                                   # Q5: the locked pair runs outside the arm's budget

    def cap_limit(self, purpose: str, point: dict) -> float:
        """Q3: 1,800 s per screen; 3,600 s only for the two reference screens (U+C and F-HT = S_match, purpose
        'reference') when the compile cache is cold (the caller asks for it with cap_s)."""
        refs = {S.key(S.registered_points()[name]) for name in REFERENCE_RECIPES}
        return float(COLD_REFERENCE_CAP_S if purpose == 'reference' and S.key(point) in refs else CAP_S)

    def describe(self) -> dict:
        return {**super().describe(), 'nodes': list(self.nodes), 'cap_s': self.cap_s, 'runtime_dir': self.runtime_dir,
                'stop_reserve_s': self.stop_reserve_s, 'switch_space_sha256': S.sha256(),
                'held_nodes': list(self.held_nodes)}

    # ------------------------------------------------------------ ScreenRunner
    def run(self, task, config: dict, overlay: dict | None, purpose: str, run_id: str, **kw) -> dict:
        candidate_id = str(kw.pop('candidate_id', None) or 'c')
        wall_cap = kw.pop('wall_cap', None)
        requested = kw.pop('cap_s', None)
        role = kw.pop('role', 'candidate')
        reference = kw.pop('reference_parsed', self.reference_parsed)
        point, errors, warnings = S.validate(config)
        code = C.is_code_task(task)
        ov = {'overlay': overlay} if code else {}      # only the code arm's receipts carry code_overlay
        if overlay:
            if code:                                   # the hooks' validator again: a refused edit never takes a node
                errors = errors + C.validate(overlay)[0]
            else:
                errors = errors + ['F1 has no file actions (switches only)']
        if errors:
            receipt = self.not_launched(task, config, purpose, run_id, candidate_id, errors, **ov)
        elif self.held_nodes:
            receipt = self.not_launched(
                task, config, purpose, run_id, candidate_id,
                [f'{", ".join(self.held_nodes)} still leased by this runner after an unproven cleanup or lease release '
                 '(see the earlier receipt); release them by hand after checking the node'], kind='nodes_held',
                basis='refused before any build or lease: nodes still held by this runner', **ov)
        else:
            limit = self.cap_limit(purpose, point)
            cap = min(float(requested or self.cap_s), limit)
            if wall_cap is not None:
                cap = min(cap, float(wall_cap))
            info = {'requested_s': requested, 'limit_s': limit, 'wall_cap_s': wall_cap, 'stop_reserve_s': self.stop_reserve_s}
            if cap < MIN_CAP_S:
                receipt = self.not_launched(task, config, purpose, run_id, candidate_id,
                                            [f'cap {cap:.0f} s is below the minimum {MIN_CAP_S:.0f} s'], kind='cap_too_small',
                                            basis='refused before any build or lease: cap too small', **ov)
            else:
                receipt = self._run(task, point, purpose, run_id, candidate_id, cap, role, reference, warnings, info,
                                    **({'overlay': overlay} if code and overlay else {}))
        self.history.append(receipt)
        return receipt

    def not_launched(self, task, config, purpose, run_id, candidate_id, errors, *, kind: str = 'invalid_proposal',
                     basis: str = 'refused by the S_F1 validator before any build or lease', **code) -> dict:
        """code: {'overlay': the candidate's overlay} for a code-arm task (its receipt then records code_overlay)."""
        attempt_id = new_attempt_id()
        now = utc_now()
        r = {'schema': 'ext-receipt-v1', 'env': self.env, 'task_id': task.id, 'arm': getattr(task, 'arm', None),
             'run_id': run_id, 'attempt_id': attempt_id, 'candidate_id': candidate_id, 'purpose': purpose,
             'candidate_digest': None, 'frozen_digest': task.frozen_digest, 'task_digest': task.digest, 'node': None,
             'gpus': [], 'world_size': int(task.world_size), 'deviations': list(self.deviations), 'valid': False,
             'invalid_reasons': [f'{kind}: {e}' for e in errors], 'step_time': {'mean_window_max_rank_s': None},
             'metric': {'name': task.metric_name, 'value': None, 'direction': 'min', 'raw': {}}, 'judge': None,
             'judge_valid': False, 'guard_rejections': [], 'guard_reports': [], 'phases': {'total': 0.0},
             'point': jsonable(config) if isinstance(config, dict) else None, 'not_launched': kind,
             'held_nodes': list(self.held_nodes), 'switch_space_sha256': S.sha256(), 'utc_start': now, 'utc_end': now,
             'wall_s': 0.0, 'cost': {'gpu_seconds': 0.0, 'wall_seconds': 0.0, 'accounting_complete': True, 'basis': basis}}
        if 'overlay' in code:                      # what the refused candidate's code was (no validation implied)
            ovl = code['overlay']
            r['code_overlay'] = (C.record(ovl) if ovl and not C.validate(ovl)[0] else C.loose_record(ovl))
            r['code_space_sha256'] = C.sha256()
        signing.sign(r)
        self._store(r)
        return r

    # ------------------------------------------------------------ one screen
    def _build(self, point: dict, *, workdir: Path, cap_s: float, overlay: dict | None = None) -> Path:
        """overlay (code arm): the bundle is built from a per-attempt copy of the variant runtime with the overlay files
        replaced (code_space.apply, under the attempt's own build directory); its code inventory must then be exactly
        the overlay's (code_space.expected_inventory)."""
        runtime = self.runtime_dir
        if overlay:
            runtime = C.apply(self.runtime_dir, overlay, Path(workdir) / 'runtime_overlay')['runtime']
        res = V.build_bundle(S.to_knobs(point), group='A', port=48270, output=workdir, runtime_dir=runtime,
                             root=V.FN_ROOT)
        bundle = Path(res['bundle'])
        if overlay:
            got = V.read(bundle / 'attempt.json').get('code_sha256')
            if got != C.expected_inventory(overlay):
                raise RuntimeError('the bundle code inventory is not the variant runtime with this code overlay')
        V.ext_bind(bundle, cap_s=cap_s)
        return bundle

    def _run(self, task, point, purpose, run_id, candidate_id, cap, role, reference, warnings, cap_info,
             overlay: dict | None = None) -> dict:
        attempt_id = new_attempt_id()
        t0, utc_start = self.clock(), utc_now()
        result, failure = None, None
        occupancy, leases = {}, {}
        ov = C.record(overlay) if overlay else None
        local = None
        try:
            with tempfile.TemporaryDirectory(prefix='fn') as tmp:
                bundle = self.build_fn(point, workdir=Path(tmp), cap_s=cap, **({'overlay': overlay} if overlay else {}))
                lay = V.remote_layout(bundle)
                port = master_port(attempt_id)
                job = V.read(bundle / 'job.json')
                job['master_port'] = port
                (bundle / 'job.json').write_bytes(V.encode(job))
                spec = node_spec(self.nodes, master_port=port, timeout_s=cap, stop_reserve_s=self.stop_reserve_s)
                t_pre = self.clock()
                self._upload(bundle, lay, spec, code_arm=bool(overlay), cap_s=cap)
                t_up = self.clock()
                taken = self._lease(run_id, attempt_id, occupancy, leases)
                cleanup: dict = {}
                try:
                    run = self._launch_and_wait(lay, cap, cleanup)
                finally:                                       # unverified (or never verified) nodes keep their lease
                    verified = set(cleanup.get('verified_nodes') or ())
                    self._release(taken, attempt_id, leases, keep=[n for n in taken if n not in verified],
                                  why=cleanup.get('survivors_by_node'))
                run['release_by_node'] = self.release_view(leases)
                t_coll = self.clock()
                local = self._collect(lay, run, candidate_id, attempt_id, code_arm=bool(overlay))
                t_done = self.clock()
                run['phases'] = {'build_s': round(t_pre - t0, 3), 'upload_s': round(t_up - t_pre, 3),
                                 'held_s': round(self.lease_cost(leases, t_coll)['held_s'], 3),
                                 'collect_s': round(t_done - t_coll, 3)}
                result = self._judge(run, local, lay, point, candidate_id, reference, cap, warnings, role,
                                     **({'code_overlay': ov} if ov else {}))
        except OccupiedError as exc:
            failure = ('occupied', exc)
        except Exception as exc:                     # noqa: BLE001 - recorded and charged, never raised
            failure = ('runner_error', exc)
        now = self.clock()
        lc = self.lease_cost(leases, now)
        receipt = self.receipt(task, point, overlay or None, purpose, run_id, attempt_id, candidate_id, result, failure,
                               utc_start=utc_start, wall=max(0.0, now - t0), held=lc['held_s'],
                               occupancy={'problems': occupancy.get('problems'), 'lease': None, 'probe': None})
        receipt.update(point=point, switch_space_sha256=S.sha256(), numerics_role=role, nodes=list(self.nodes),
                       cap_s=cap, cap=cap_info, held_nodes=list(self.held_nodes),
                       occupancy_by_node=occupancy.get('by_node'), release_by_node=self.release_view(leases))
        if C.is_code_task(task):                     # setting 2: which code ran (None = the frozen runtime)
            receipt.update(code_overlay=ov, code_space_sha256=C.sha256())
            if ov:                                   # an overlay screen also ran the external-clock watcher (NM18 P1-2)
                receipt['external_clock'] = {'watcher_sha256': EXTCLOCK_SHA, 'poll_s': EXTCLOCK_POLL_S,
                                             'files': [f'fn_extclock.node{r}.jsonl' for r in range(4)]}
            if result is not None and not result.judge_valid and local is not None:
                try:                                 # the Python error a crashed edit ended with (agent feedback)
                    receipt['failure_tail'] = C.failure_tail(local)
                except Exception as exc:             # noqa: BLE001 - feedback only, never fails the receipt
                    receipt['failure_tail'] = {'error': f'{type(exc).__name__}: {exc}'[:300]}
        receipt['cost'] = {'gpu_seconds': round(lc['gpu_seconds'], 3), 'wall_seconds': receipt['cost']['wall_seconds'],
                           'accounting_complete': not lc['open_ended_nodes'], 'basis': COST_BASIS,
                           'by_node': lc['by_node'], 'open_ended_nodes': lc['open_ended_nodes']}
        signing.sign(receipt)
        self._store(receipt)
        self.attempts.append({'attempt_id': attempt_id, 'candidate_id': candidate_id, 'purpose': purpose,
                              'valid': receipt['valid'], 'metric': receipt['metric']['value'], 'utc': receipt['utc_end']})
        self.log(f'{run_id}/{candidate_id}/{attempt_id}: {purpose} valid={receipt["valid"]} step_time_s={receipt["metric"]["value"]}'
                 + (f' HELD {self.held_nodes}' if self.held_nodes else ''))
        return receipt

    # ------------------------------------------------------------ steps
    @staticmethod
    def extclock_files(lay: dict) -> dict:
        """{remote fn_extclock.node<r>.jsonl -> local extclock/node<rr>.jsonl} (code-arm screens only)."""
        out = lay['output_root']
        return {f'{out}/{EXTCLOCK_NAME[:-3]}.node{r}.jsonl': f'extclock/node{r:02d}.jsonl' for r in range(4)}

    def _upload(self, bundle: Path, lay: dict, spec: dict, *, code_arm: bool = False, cap_s: float | None = None) -> None:
        ops = self.ops
        meta = lay['metadata']
        files = {remote: Path(local).read_bytes() for local, remote in lay['files']}
        files[meta + '/fn_spec.json'] = V.encode(spec)
        if code_arm:
            files[f'{meta}/{EXTCLOCK_NAME}'] = EXTCLOCK_PY.encode()
        for r in range(4):
            watcher = None
            if code_arm:
                cfg = {'node_rank': r, 'first_rank': r * GPUS_PER_NODE, 'nproc': GPUS_PER_NODE,
                       'life': lay['lifecycle_run_root'], 'run': lay['run'],
                       'out': f"{lay['output_root']}/{EXTCLOCK_NAME[:-3]}.node{r}.jsonl",
                       'deadline_s': float(cap_s if cap_s is not None else self.cap_s), 'stop_update': 30,
                       'poll_s': EXTCLOCK_POLL_S}
                watcher = {'path': f'{meta}/{EXTCLOCK_NAME}', 'log': f'{meta}/fn_extclock.node{r}.log', 'cfg': cfg}
            files[f'{meta}/fn_node{r}.sh'] = node_script(self.src, lay['job'], r, meta + '/fn_spec.json',
                                                         watcher=watcher).encode()
        ops.mkdirs(sorted({str(Path(p).parent).replace('\\', '/') for p in files}))
        batch, size = {}, 0
        for path, data in sorted(files.items()):           # bounded batches (each call is one SSH round trip)
            batch[path] = data
            size += len(data)
            if size > (4 << 20):
                ops.write_files(batch)
                batch, size = {}, 0
        if batch:
            ops.write_files(batch)
        want = {p: hashlib.sha256(d).hexdigest() for p, d in files.items()}
        got = ops.hash_files(sorted(want))
        bad = sorted(p for p in want if got.get(p) != want[p])
        if bad:
            raise RuntimeError(f'upload verification failed for {bad[:4]}')
        launcher = f'{self.src}/node/{LAUNCHER}'
        if ops.hash_files([launcher]).get(launcher) != hashlib.sha256((NODE_DIR / LAUNCHER).read_bytes()).hexdigest():
            raise RuntimeError(f'{launcher} on WINH is not this version of the launcher (deploy it first)')

    def _lease(self, run_id: str, attempt_id: str, occupancy: dict, leases: dict) -> list:
        """All or nothing: probe all four, then take the four node leases in order; roll back on any failure. A node
        whose acquire call failed in transport counts as possibly held (the rollback's release_status settles it)."""
        lease = make_lease(run_id, attempt_id, self.nodes, purpose='ext:flashnext', scope='attempt')
        lease.update(owner=OWNER, gpus=list(GPUS))
        problems, by_node = {}, {}
        for n in self.nodes:
            ops = self.ops_by_node[n]
            current = ops.call('read_json', path=node_lease_path(n), default=None, _timeout=CALL_TIMEOUT_S)
            probe = ops.call('gpu_probe', _timeout=CALL_TIMEOUT_S)
            why = occupancy_problems(probe, GPUS, lease=current, lease_ok=lambda c: c.get('attempt_id') == attempt_id)
            by_node[n] = {'hostname': probe.get('hostname'), 'apps': len(probe.get('apps') or []), 'problems': why}
            if probe.get('hostname') != HOSTNAMES[node_index(n)]:
                why.append(f'hostname {probe.get("hostname")} is not {n} ({HOSTNAMES[node_index(n)]})')
            if why:
                problems[n] = '; '.join(why)
        occupancy.update(problems=problems, by_node=by_node)
        if problems:
            raise OccupiedError(problems)
        taken = []
        try:
            for n in self.nodes:
                leases[n] = {'state': 'acquiring', 'acquired_t': self.clock(), 'released_t': None}
                taken.append(n)
                res = self.ops_by_node[n].call('acquire', path=node_lease_path(n), lease=lease, _timeout=CALL_TIMEOUT_S)
                if not res['ok']:                              # someone else's lease: never ours, never released by us
                    taken.pop()
                    del leases[n]
                    raise OccupiedError({n: f'lease taken: {res.get("existing")}'})
                leases[n].update(state='held', acquired_t=self.clock())
        except BaseException:
            self._release(taken, attempt_id, leases)
            raise
        return taken

    def _release(self, taken: list, attempt_id: str, leases: dict, keep=(), why=None) -> None:
        """Release every node lease in `taken` except `keep`, idempotently, with retries, all nodes in parallel; no new
        attempt starts after RELEASE_BUDGET_S and each call's timeout is capped by the time left; never raises. A kept
        node or an unproven release stays leased and is added to held_nodes; a foreign holder is never touched."""
        keep = set(keep or ())
        order = list(reversed(list(taken)))
        for n in order:
            rec = leases.setdefault(n, {'acquired_t': self.clock(), 'released_t': None})
            if n in keep:
                rec.update(state='kept', reason=jsonable((why or {}).get(n) or 'cleanup not verified'))
        todo = [n for n in order if n not in keep]
        deadline = self.clock() + RELEASE_BUDGET_S
        results = self._each(lambda n: self._release_one(n, attempt_id, deadline), todo)
        for n in todo:
            rec, (kind, value) = leases[n], results[n]
            if kind == 'ok' and value.get('state') in ('removed', 'absent', 'foreign'):
                rec.update(state=value['state'], released_t=value['t'])
                if value['state'] == 'foreign':
                    rec['holder'] = jsonable(value.get('holder'))
                continue
            if kind == 'ok':
                err = value.get('error') or f'unexpected release status {value.get("state")!r}'
            else:
                err = f'{type(value).__name__}: {value}'
            rec.update(state='unresolved', error=str(err)[:300])
        for n in order:
            rec = leases[n]
            if rec.get('state') in ('kept', 'unresolved') and n not in self.held_nodes:
                self.held_nodes.append(n)
                self.log(f'{n}: lease kept ({rec["state"]}): {str(rec.get("reason") or rec.get("error"))[:300]}')

    def _release_one(self, n: str, attempt_id: str, deadline: float) -> dict:
        err = None
        for i in range(RELEASE_ATTEMPTS):
            if deadline - self.clock() <= 0:
                break
            try:
                st = self.ops_by_node[n].call('release_status', path=node_lease_path(n), attempt_id=attempt_id,
                                              _timeout=self._call_timeout(deadline))
                return {'state': (st or {}).get('state'), 'holder': (st or {}).get('holder'), 't': self.clock()}
            except Exception as exc:                          # noqa: BLE001 - transport failure: retried, then recorded
                err = f'{type(exc).__name__}: {str(exc)[:300]}'
                if i + 1 < RELEASE_ATTEMPTS:
                    self.sleep(min(RELEASE_BACKOFF_S * (2 ** i), max(0.0, deadline - self.clock())))
        return {'state': None, 'error': err or f'release deadline ({RELEASE_BUDGET_S:.0f} s)'}

    def _call_timeout(self, deadline: float) -> float:
        return max(1.0, min(CALL_TIMEOUT_S, deadline - self.clock()))

    def _each(self, fn, nodes) -> dict:
        """fn(node) on every node at once (independent, idempotent node calls, each with its own timeout) and wait for
        all of them: no call is abandoned, so every result is final when this returns and no late call can change a
        node or a lease afterwards. {node: ('ok', value) | ('error', exception)}."""
        nodes = list(nodes)
        if not nodes:
            return {}
        with ThreadPoolExecutor(max_workers=len(nodes), thread_name_prefix='flashnext') as pool:
            futures = {n: pool.submit(fn, n) for n in nodes}
        out = {}
        for n, f in futures.items():
            exc = f.exception()
            out[n] = ('error', exc) if exc is not None else ('ok', f.result())
        return out

    def _read(self, paths, max_bytes: int) -> dict:
        raw = self.ops.call('read_files', paths=list(paths), max_bytes=max_bytes, _timeout=CALL_TIMEOUT_S)
        return {p: (None if v is None else base64.b64decode(v)) for p, v in raw.items()}

    def _write(self, files: dict) -> None:
        enc = {p: base64.b64encode(d).decode('ascii') for p, d in files.items()}
        self.ops.call('write_files', files=enc, mode=None, uid=None, gid=None, _timeout=CALL_TIMEOUT_S)

    @staticmethod
    def release_view(leases: dict) -> dict:
        return {n: {k: v for k, v in rec.items() if k not in ('acquired_t', 'released_t')} for n, rec in leases.items()}

    @staticmethod
    def lease_cost(leases: dict, now: float) -> dict:
        """8 GPUs x seconds per node lease, acquisition -> proven release; still-held nodes up to `now` (open-ended)."""
        by_node, open_ended, gpu_s, first, last = {}, [], 0.0, None, None
        for n, rec in leases.items():
            a = rec.get('acquired_t')
            if a is None:
                continue
            r = rec.get('released_t')
            end = now if r is None else r
            held = max(0.0, end - a)
            gpu_s += GPUS_PER_NODE * held
            by_node[n] = {'held_s': round(held, 3), 'state': rec.get('state'), 'open_ended': r is None}
            if r is None:
                open_ended.append(n)
            first = a if first is None else min(first, a)
            last = end if last is None else max(last, end)
        return {'gpu_seconds': gpu_s, 'by_node': by_node, 'open_ended_nodes': open_ended,
                'held_s': (last - first) if first is not None else 0.0}

    def _launch_and_wait(self, lay: dict, cap: float, cleanup: dict) -> dict:
        """cleanup is filled in `finally` (kill + verification) even when launching or polling raises."""
        out, meta = lay['output_root'], lay['metadata']
        token = new_token('isxf')
        exits_paths = [f'{out}/node{r}.exit.json' for r in range(4)]
        launched, exits, stopped, timed_out, poll_errors = {}, {}, False, False, 0
        t_launch = self.clock()
        try:
            for r, n in enumerate(self.nodes):
                launched[n] = self.ops_by_node[n].call('launch', script=f'{meta}/fn_node{r}.sh',
                                                       log=f'{meta}/fn_node{r}.launch.log',
                                                       env={node_ext.TOKEN_VAR: token}, _timeout=CALL_TIMEOUT_S)
            stop_at = t_launch + cap                       # backstop: the launchers stop at cap - stop reserve
            give_up = stop_at + self.stop_margin_s
            while True:
                try:
                    raw = self._read(exits_paths, 1 << 16)
                    exits = {p: json.loads(b) for p, b in raw.items() if b}
                except Exception as exc:                  # noqa: BLE001 - a transport hiccup must not end the screen
                    poll_errors += 1
                    self.log(f'poll failed ({poll_errors}): {exc!r}'[:300])
                if len(exits) == 4:
                    break
                now = self.clock()
                if now >= stop_at and not stopped:
                    try:
                        self._write({f'{out}/STOP.json': V.encode({'reason': 'runner wall cap', 'cap_s': cap,
                                                                  'utc': utc_now()})})
                        stopped = True
                    except Exception as exc:              # noqa: BLE001 - retried at the next poll; killed at give_up
                        self.log(f'STOP.json write failed: {exc!r}'[:300])
                if now >= give_up:
                    timed_out = True
                    break
                self.sleep(self.poll_s)
        finally:
            cleanup.update(self._kill_everywhere(token, lay, launched))
        return {'token': token, 'launched': launched, 'exits': {Path(p).name: v for p, v in exits.items()},
                'runner_stop': stopped, 'timed_out': timed_out, 'poll_errors': poll_errors, 'cap_s': cap, **cleanup}

    def _check_node(self, n: str, r: int, token: str, lay: dict, launched: dict | None, roots, deadline: float):
        """One cleanup round on one node: the sweep (job first, launcher last), then the GPU probe. (killed, problems);
        a check that fails or cannot run is a problem."""
        ops = self.ops_by_node[n]
        problems, killed = [], []
        pid = ((launched or {}).get(n) or {}).get('pid')
        cmd = sweep_command(token, f'{lay["output_root"]}/node{r}.pid.json', pid if isinstance(pid, int) else None)
        try:
            t = self._call_timeout(deadline)
            sw = parse_sweep(ops.call('sh', cmd=cmd, timeout=t, _timeout=t + 15))
            killed = list(sw.get('killed') or [])
            if sw.get('token_pids'):
                problems.append({'token_pids': sw['token_pids'][:16]})
            alive = {g: m for g, m in (sw.get('groups') or {}).items() if m}
            if alive:
                problems.append({'process_groups': alive})
        except Exception as exc:                              # noqa: BLE001 - unverifiable counts as surviving
            problems.append(f'sweep failed: {exc!r}'[:300])
        try:
            probe = ops.call('gpu_probe', _timeout=self._call_timeout(deadline))
            if probe.get('error'):
                problems.append(f'probe error: {probe["error"]}'[:300])
            ours = [a for a in probe.get('apps') or []
                    if a.get('token') == token or any(is_ours(a, root) for root in roots)]
            if ours:
                problems.append({'gpu': [{k: a.get(k) for k in ('pid', 'gpu_index', 'cwd')} for a in ours][:8]})
        except Exception as exc:                              # noqa: BLE001 - unverifiable counts as surviving
            problems.append(f'probe failed: {exc!r}'[:300])
        return killed, problems

    def _kill_everywhere(self, token: str, lay: dict, launched: dict | None = None) -> dict:
        """Rounds (KILL_WAITS) until every node is verified; no round starts after CLEANUP_BUDGET_S and each call's
        timeout is capped by the time left. In each round, all unverified nodes at once: _check_node (SIGKILL this screen's job group and token carriers, then the launcher's session -
        each group only where a live member carries the token - re-list both, then probe the GPUs for any process of
        ours: this screen's token, or a cwd / command line under any flashnext run directory). A node is verified only
        when the sweep and the probe both ran and found nothing; anything else (a failed or unfinished check included)
        is a survivor, and the node keeps its lease."""
        roots = [lay['output_root'], lay['metadata'], V.FN_ROOT]
        deadline = self.clock() + CLEANUP_BUDGET_S
        killed = {n: [] for n in self.nodes}
        rank = {n: r for r, n in enumerate(self.nodes)}
        pending, survivors = list(self.nodes), {}
        for wait_s in self.KILL_WAITS:
            if wait_s:
                if self.clock() + wait_s + 1.0 >= deadline:
                    break
                self.sleep(wait_s)
            results = self._each(lambda n: self._check_node(n, rank[n], token, lay, launched, roots, deadline), pending)
            survivors = {}
            for n in pending:
                kind, value = results[n]
                if kind == 'ok':
                    killed[n].extend(value[0])
                    if value[1]:
                        survivors[n] = value[1]
                else:
                    survivors[n] = [f'check failed: {value!r}'[:300]]
            pending = [n for n in pending if n in survivors]
            if not pending or self.clock() >= deadline:
                break
        return {'killed_by_node': killed, 'survivors_by_node': survivors,
                'verified_nodes': [n for n in self.nodes if n not in survivors]}

    def _collect(self, lay: dict, run: dict, candidate_id: str, attempt_id: str, *, code_arm: bool = False) -> Path:
        base = Path(self.out_dir) if self.out_dir else Path(tempfile.mkdtemp(prefix='fncollect'))
        local = base / 'attempts' / f'{candidate_id}-{attempt_id}'
        local.mkdir(parents=True, exist_ok=True)
        run['collection'] = self.fetch_outputs(lay, local,
                                               extra_wanted=self.extclock_files(lay) if code_arm else None)
        (local / 'run.json').write_text(json.dumps(jsonable({**run, 'layout': {k: v for k, v in lay.items() if k != 'files'}}),
                                                   indent=1), encoding='utf-8')
        if not self.keep_remote:
            self._delete_bulky(lay['output_root'])
        return local

    @staticmethod
    def output_files(lay: dict) -> tuple[dict, dict]:
        """({remote path: local name} of the files parse.collect reads, {remote path: local name} of the logs)."""
        out, meta = lay['output_root'], lay['metadata']
        wanted = {p: f'signals/rank{r:02d}.jsonl' for r, p in enumerate(lay['signals'])}
        wanted.update({p: f'ranks/rank{r:02d}.json' for r, p in enumerate(lay['rank_receipts'])})
        for r in range(4):
            for kind in ('exit', 'release', 'pid'):
                wanted[f'{out}/node{r}.{kind}.json'] = f'nodes/node{r}.{kind}.json'
        wanted.update({meta + '/attempt.json': 'attempt.json', meta + '/job.json': 'job.json',
                       meta + '/fn_spec.json': 'fn_spec.json',      # the worker writes it under launch['output_root']
                       f'{out}/family_materialization/materialization.json': 'materialization.json'})
        logs = {f'{out}/node{r}.log': f'nodes/node{r}.log' for r in range(4)}
        logs.update({f'{meta}/fn_node{r}.launch.log': f'nodes/fn_node{r}.launch.log' for r in range(4)})
        return wanted, logs

    def _get(self, path: str) -> bytes | None:
        """One whole remote file over SFTP when the transport has it (WinhRemote.get_bytes: request / response, no
        exit-status race), else the node helper's read_files for this one file (in-process stand-ins have no channel)."""
        getter = getattr(getattr(self.ops, 'remote', None), 'get_bytes', None)
        if getter is not None:
            return getter(self.ops.node, path)
        return self.ops.read_files([path]).get(path)

    def _collect_call(self, items: list) -> list:
        arg = base64.b64encode(json.dumps({'files': items, 'budget': COLLECT_BUDGET_BYTES,
                                           'preset': COLLECT_PRESET}).encode()).decode('ascii')
        res = self.ops.call('sh', cmd=f'python3 -c {q(COLLECT_PY)} {q(arg)}', timeout=COLLECT_CALL_TIMEOUT_S,
                            _timeout=COLLECT_CALL_TIMEOUT_S + 30)
        out = (res or {}).get('stdout') or ''
        if (res or {}).get('code') != 0 or COLLECT_MARK not in out:
            raise RuntimeError(f'collect exit {(res or {}).get("code")}: {((res or {}).get("stderr") or "")[-300:]}')
        return json.loads(out.split(COLLECT_MARK, 1)[1].strip().splitlines()[0])['files']

    def fetch_outputs(self, lay: dict, local: Path, *, extra_wanted: dict | None = None) -> dict:
        """Copy a screen's outputs from WINH into `local` in the collected layout parse.collect reads; read-only on WINH.
        Every file (a log: its last LOG_TAIL_BYTES) is read, hashed and lzma-compressed on the node and comes back in
        replies of at most COLLECT_BUDGET_BYTES compressed; it must decompress to the node's sha256 and size. A failed
        reply or a mismatch is asked again from that file on; COLLECT_ATTEMPTS replies in a row without progress raise
        RuntimeError. A file missing on the node leaves no local copy (a stale one is removed). Returns what was copied
        (per file: remote path, size, sha256, mtime)."""
        local = Path(local)
        wanted, logs = self.output_files(lay)
        wanted = {**wanted, **(extra_wanted or {})}      # code arm: the external-clock watcher's records (whole files)
        name_of = {**wanted, **logs}
        pending = [[p, 0] for p in sorted(wanted)] + [[p, LOG_TAIL_BYTES] for p in sorted(logs)]
        files, calls, failures, sftp = {}, 0, 0, 0
        t0 = self.clock()
        remote = getattr(self.ops, 'remote', None) is not None
        while pending:
            if not remote:                                # in-process stand-in: no channel, read the file itself
                path, tail = pending.pop(0)
                data = self.ops.read_files([path], max_bytes=tail or None).get(path)
                calls += 1
                dst = local / name_of[path]
                if data is None:
                    dst.unlink(missing_ok=True)               # no stale copy may stand for a missing file
                    files[name_of[path]] = None
                    continue
                files[name_of[path]] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
                continue
            error, done = None, 0
            try:
                calls += 1
                got = self._collect_call(pending)
                if not got:
                    raise RuntimeError('collect: empty reply')
                for entry in got:
                    path, tail = pending[done]
                    if entry.get('path') != path:
                        raise RuntimeError(f'collect: reply out of order at {path}')
                    if entry.get('missing'):
                        (local / name_of[path]).unlink(missing_ok=True)   # no stale copy may stand for a missing file
                        files[name_of[path]] = None
                        done += 1
                        continue
                    if entry.get('too_big'):
                        sftp += 1
                        whole = self._get(path)
                        data = whole[-tail:] if (whole is not None and tail) else whole
                    else:
                        data = lzma.decompress(base64.b64decode(entry['xz']))
                    if data is None or len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
                        raise RuntimeError(f'collect: {path} does not match its sha256 / size on the node')
                    dst = local / name_of[path]
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(data)
                    files[name_of[path]] = {'remote': path, 'bytes': len(data), 'source_bytes': entry['size'],
                                            'sha256': entry['sha256'], 'mtime': entry.get('mtime'),
                                            **({'tail_bytes': len(data)} if tail and entry['size'] > tail else {})}
                    done += 1
            except Exception as exc:                          # noqa: BLE001 - asked again from the failed file on
                error = f'{type(exc).__name__}: {str(exc)[:300]}'
            pending = pending[done:]
            if done:
                failures = 0                                  # progress: count failures from the new head file
            if error is None:
                continue
            failures += 1
            self.log(f'collection retry {failures}/{COLLECT_ATTEMPTS}: {error}')
            if failures >= COLLECT_ATTEMPTS:
                raise RuntimeError(f'collection: {error} ({COLLECT_ATTEMPTS} failed replies in a row)')
        return {'method': 'node_lzma' if remote else 'read_files', 'calls': calls, 'sftp_files': sftp, 'files': files,
                'seconds': round(self.clock() - t0, 3)}

    def _delete_bulky(self, out: str) -> None:
        if not out.startswith(V.FN_ROOT + '/runs/'):
            raise ValueError(f'refusing to clean {out}')
        try:
            self.ops.sh(f'find {q(out)} -xdev -type f -size +{BULKY_BYTES // (1 << 20)}M -delete', timeout=300)
        except Exception as exc:                              # noqa: BLE001 - reported, not fatal
            self.log(f'bulky cleanup of {out} failed: {exc!r}')

    # ------------------------------------------------------------ judge
    RECOLLECTED_COPY = ('task_id', 'arm', 'frozen_digest', 'task_digest', 'deviations', 'world_size', 'gpus', 'node',
                        'switch_space_sha256', 'numerics_role', 'nodes', 'cap_s', 'cap', 'held_nodes', 'occupancy',
                        'occupancy_by_node', 'release_by_node', 'utc_start', 'utc_end', 'wall_s', 'phases',
                        'code_overlay', 'code_space_sha256')      # the last two only on code-arm receipts

    def judge_recollected(self, local: Path, lay: dict, original: dict, reference, task, extra_rejections=()) -> dict:
        """The receipt this runner would have written for `original`'s attempt, judged by the unchanged pipeline (_judge:
        X6's parse / judge / guards + the runner guards) over outputs re-collected read-only into `local`; the attempt's
        own records (identity, nodes, release states, times) come from `original`. Unsigned: the supersede step builds and
        signs the record (supersede.supersede_record). extra_rejections [(id, detail)]: identity checks the caller
        made (the attempt binding) - vetoes like the guards' own, reported instead when the judge already rejects."""
        point, cap = original['point'], float(original.get('cap_s') or CAP_S)
        exits = {}
        for r in range(4):
            p = Path(local) / 'nodes' / f'node{r}.exit.json'
            exits[f'node{r}.exit.json'] = json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
        run = {'launched': None, 'exits': exits, 'runner_stop': False, 'timed_out': False, 'cap_s': cap,
               'killed_by_node': None, 'survivors_by_node': {}, 'verified_nodes': list(original.get('nodes') or []),
               'release_by_node': original.get('release_by_node') or {}, 'poll_errors': None,
               'phases': dict(original.get('phases') or {})}
        ov = original.get('code_overlay') if isinstance(original.get('code_overlay'), dict) else None
        result = self._judge(run, Path(local), lay, point, original['candidate_id'], reference, cap, [],
                             original.get('numerics_role') or 'candidate', **({'code_overlay': ov} if ov else {}))
        for gid, detail in extra_rejections:
            if result.judge_valid:
                result.guards.reject(gid, detail)
            else:
                result.guards.report(gid, would_reject=detail, note='judge invalid: not a veto')
        rec = self.receipt(task, point, None, original['purpose'], original['run_id'], original['attempt_id'],
                           original['candidate_id'], result, None, utc_start=original['utc_start'],
                           wall=float(original.get('wall_s') or 0.0), held=0.0, occupancy=None)
        rec.update({k: original[k] for k in self.RECOLLECTED_COPY if k in original})
        rec['point'] = point
        rec.pop(signing.SIGNATURE_FIELD, None)
        return rec

    def _judge(self, run: dict, local: Path, lay: dict, point: dict, candidate_id: str, reference, cap: float,
               warnings, role: str, code_overlay: dict | None = None) -> EnvRun:
        """judge_fn(local_dir, reference, gates, role, point, cap) -> (verdict, GuardResult, receipt fields). Default:
        X6's pipeline (E4) - parse.collect, judge.evaluate (the production metric and the Q3 numerics gate),
        guards.check (the Q3 hard checks: complete_verified, frozen_hashes, knob_binding, wall_cap, oom_nccl) and
        judge.receipt_extra. The runner adds what only it sees: process_cleanup, runner_timeout and lease_release.
        code_overlay (a code-arm screen with an overlay, code_space.record): judge_fn also gets code_overlay = its file
        hashes (frozen_hashes then expects exactly those files replaced) and the candidate digest includes its digest."""
        kw = {'code_overlay': dict(code_overlay.get('files') or {})} if code_overlay else {}
        verdict, guards, fields = (self.judge_fn or self.x6_judge)(local, reference, self.gates, role, point, cap, **kw)
        guards = cg.combine(guards, self.runner_guards(run))
        judge_valid = bool(verdict.get('judge_valid'))
        if not judge_valid:                                    # guards veto only what the judge accepted
            guards = cg.GuardResult(reports=list(guards.reports) + [{'id': g['id'], 'would_reject': g['detail'],
                                                                     'note': 'judge invalid: not a veto'} for g in guards.rejections])
        return EnvRun(candidate_id=candidate_id, judge=jsonable(verdict), judge_valid=judge_valid,
                      metric_value=verdict.get('metric_s'), metric_raw={'numerics_class': verdict.get('numerics_class')},
                      judge_reasons=list(verdict.get('reasons') or []), guards=guards, phases=run.get('phases') or {},
                      candidate_digest=digest({'switch_space': S.sha256(), 'point': point,
                                               **({'code_overlay': code_overlay.get('digest')} if code_overlay else {})}),
                      extra={**(fields or {}),
                             'runner': {k: run.get(k) for k in ('launched', 'exits', 'runner_stop', 'timed_out', 'cap_s',
                                                                'poll_errors', 'killed_by_node', 'survivors_by_node',
                                                                'verified_nodes', 'release_by_node')},
                             'bundle': {k: lay[k] for k in ('name', 'run', 'metadata', 'output_root', 'lifecycle_run_root')},
                             'numerics_class': verdict.get('numerics_class'), 'collected': str(local),
                             'validator_warnings': list(warnings)})

    @staticmethod
    def x6_judge(local: Path, reference, gates, role: str, point: dict, cap: float, code_overlay: dict | None = None):
        """code_overlay {'code/<name>': sha256} (code arm only): the files guards.frozen_hashes accepts replaced."""
        from . import guards as G
        from . import judge as J
        from . import parse as P
        parsed = P.collect(local)
        verdict = J.evaluate(parsed, reference, gates, role=role)
        kw = {'code_overlay': code_overlay} if code_overlay else {}
        return verdict, G.check(parsed, point, cap_s=cap, **kw), J.receipt_extra(verdict)

    def runner_guards(self, run: dict) -> cg.GuardResult:
        """Only what the runner itself observes; the screen's hard checks are X6's guards.check."""
        g = cg.GuardResult()
        if run.get('survivors_by_node'):
            g.reject('process_cleanup', f'cleanup not verified (processes of ours or failed checks): {run["survivors_by_node"]}')
        if run.get('timed_out'):
            g.reject('runner_timeout', f'not all four node exits within cap + {self.stop_margin_s:.0f} s; killed by token')
        bad = {n: s for n, s in (run.get('release_by_node') or {}).items() if s.get('state') in ('unresolved', 'foreign')}
        if bad:
            g.reject('lease_release', f'node lease release not proven: {bad}')
        g.report('runner', stopped=bool(run.get('runner_stop')), exits=sorted(run.get('exits') or {}),
                 poll_errors=run.get('poll_errors'))
        return g

    def dry_run(self, point: dict, *, workdir, cap_s: float | None = None, purpose: str = 'screen',
                overlay: dict | None = None) -> dict:
        """Build and bind the bundle of `point` locally and return exactly what one screen would upload and run; nothing
        is uploaded, leased or launched. overlay (code arm): the bundle of the per-attempt runtime copy with it."""
        point, errors, _ = S.validate(point)
        if overlay:
            errors = errors + C.validate(overlay)[0]
        if errors:
            raise ValueError('; '.join(errors))
        cap = min(float(cap_s or self.cap_s), self.cap_limit(purpose, point))
        bundle = self.build_fn(point, workdir=Path(workdir), cap_s=cap, **({'overlay': overlay} if overlay else {}))
        lay, job = V.remote_layout(bundle), V.read(Path(bundle) / 'job.json')
        spec = node_spec(self.nodes, master_port=master_port(lay['attempt_id']), timeout_s=cap,
                         stop_reserve_s=self.stop_reserve_s)
        out = {'bundle': str(bundle), 'metadata': lay['metadata'], 'output_root': lay['output_root'],
               'node_scripts': {self.nodes[r]: node_script(self.src, lay['job'], r, lay['metadata'] + '/fn_spec.json')
                                for r in range(4)},
               'job_command': job['command'], 'job_env': job['env'], 'timeout_seconds': job['timeout_seconds'],
               'phase_gate': V.read(Path(bundle) / 'attempt.json')['phase_gate'], 'fn_spec': spec}
        if overlay:
            out.update(code_overlay=C.record(overlay), code_sha256=V.read(Path(bundle) / 'attempt.json')['code_sha256'],
                       remote_code_dir=lay['metadata'] + '/code')
        return out


if __name__ == '__main__':
    # python -m infrarsi.ext.flashnext.runner dry-run [--nodes WINH1,WINH3,WINH4,WINH6] [--cap 1800]
    #                                                   [--purpose reference] [--point JSON]
    import argparse
    from .deploy import custom_runtime_local
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['dry-run'])
    ap.add_argument('--nodes', default=','.join(DEFAULT_NODES))
    ap.add_argument('--cap', type=float, default=CAP_S)
    ap.add_argument('--purpose', default='screen', help="'reference' allows 3,600 s for U+C / S_match")
    ap.add_argument('--point', default=None, help='S_F1 point as JSON (default: the start, U+C)')
    a = ap.parse_args()
    nodes = a.nodes.split(',')
    with tempfile.TemporaryDirectory(prefix='fndry') as tmp:
        rt = custom_runtime_local(Path(tmp) / 'rt')
        from ..common.remote_exec import NodeOps         # abstract: any call would raise; a dry run makes none
        r = FlashNextRunner({n: NodeOps(n) for n in nodes}, nodes=nodes, runtime_dir=rt['runtime'], cap_s=a.cap)
        out = r.dry_run(json.loads(a.point) if a.point else S.start_point(), workdir=Path(tmp) / 'b', cap_s=a.cap,
                        purpose=a.purpose)
    print(json.dumps(out, indent=1))
