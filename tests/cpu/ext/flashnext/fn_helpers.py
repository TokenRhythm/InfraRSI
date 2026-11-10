"""Four simulated WINH nodes for FlashNextRunner tests: LocalNodeOps sharing one /data root, programmable probes,
exit codes, hangs and surviving processes, a fake clock, a fake bundle builder and a stub judge."""
from __future__ import annotations

import json
import os
import shlex
import threading
from pathlib import Path

from infrarsi.ext.common.env_task import EnvTask
from infrarsi.ext.common.remote_exec import LocalNodeOps, NodeError
from infrarsi.ext.flashnext import runner as R
from infrarsi.ext.flashnext import variant as V


class FakeClock:
    """Shared by the runner's per-node threads (cleanup and release run in parallel): updates under a lock."""

    def __init__(self):
        self.t = 1000.0
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            return self.t

    def sleep(self, s):
        with self._lock:
            self.t += float(s)


class Cluster:
    def __init__(self, root: Path, nodes=R.DEFAULT_NODES, clock: FakeClock | None = None):
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        # Mirrored /data paths (bundle names ~90 chars) exceed MAX_PATH under pytest's temp dirs: extended-length root.
        self.root = Path('\\\\?\\' + str(root)) if os.name == 'nt' else root
        self.nodes = list(nodes)
        self.clock = clock or FakeClock()
        self.apps = {n: [] for n in self.nodes}           # compute apps each probe reports
        self.exit_codes = {r: 0 for r in range(4)}         # per node rank
        self.hang = False                                  # launchers never write exit files
        self.survivors = {}               # node -> GPU apps of ours left after every sweep
        self.token_survivors = {}         # node -> pids carrying the token left after every sweep (not on a GPU)
        self.group_survivors = {}         # node -> {pgid: [pids]} live group members left after every sweep
        self.sweep_fail = set()           # nodes whose sweep cannot run (exit 1)
        self.probe_error_after_launch = {}  # node -> nvidia-smi error reported once the node was launched
        self.release_failures = {}        # node -> number of release_status calls that fail in transport
        self.block = {}                   # (node, 'sweep' | node_ext function) -> threading.Event: the call hangs
        self.launched, self.kills, self.sweeps = [], [], []
        self.ops = {n: self._ops(n) for n in self.nodes}
        # the deployed launcher (the runner refuses another version)
        self.write(f'{V.FN_SRC}/node/{R.LAUNCHER}', (R.NODE_DIR / R.LAUNCHER).read_bytes())

    def local(self, remote: str) -> Path:
        return self.root / remote.lstrip('/')

    def write(self, remote: str, data) -> None:
        p = self.local(remote)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())

    def _ops(self, n):
        cluster = self

        class Ops(LocalNodeOps):
            def call(self, func, _timeout: float = 300, **kw):
                sweep = func == 'sh' and R.SWEEP_MARK in kw.get('cmd', '')
                blocker = cluster.block.get((n, 'sweep' if sweep else func))
                if func == 'gpu_probe' and not any(m == n for m, _ in cluster.launched):
                    blocker = None                        # the pre-lease occupancy probe answers
                if blocker is not None:                   # a node call that hangs until its own timeout (as WinhRemote)
                    blocker.wait(timeout=min(float(_timeout), 60.0))
                    raise NodeError(f'{n} {func}: simulated hang, timed out after {_timeout} s')
                if sweep:                                 # the runner's node-side cleanup (runner.SWEEP_PY)
                    return cluster._sweep(n, kw['cmd'])
                if func == 'release_status' and cluster.release_failures.get(n, 0) > 0:
                    cluster.release_failures[n] -= 1
                    raise NodeError(f'{n} release_status: simulated transport failure')
                return super().call(func, _timeout, **kw)

        return Ops(self.root, n, probe_fn=lambda: self._probe(n), launcher=lambda ops, script, log: self._launch(n, script))

    def _sweep(self, n, cmd):
        token, pidfile, launcher = R.sweep_args(cmd)
        groups = [launcher] if launcher > 1 else []
        self.sweeps.append((n, token, pidfile, groups))
        if n in self.sweep_fail:
            return {'code': 1, 'stdout': '', 'stderr': 'simulated: python3 not found'}
        self.kills.append((n, token))
        self.apps[n] = list(self.survivors.get(n, []))
        job = json.loads(self.local(pidfile).read_text()).get('pid') if pidfile != '-' and self.local(pidfile).exists() else None
        checked = sorted({*groups, *([job] if job else [])})
        left = self.group_survivors.get(n, {})
        res = {'killed': [], 'token_pids': list(self.token_survivors.get(n, [])), 'groups_checked': checked,
               'groups': {str(g): left[g] for g in checked if left.get(g)}}
        return {'code': 0, 'stdout': R.SWEEP_MARK + json.dumps(res) + '\n', 'stderr': ''}

    def _probe(self, n):
        launched = any(m == n for m, _ in self.launched)
        return {'hostname': R.HOSTNAMES[R.node_index(n)], 'utc': 'now',
                'error': self.probe_error_after_launch.get(n) if launched else None,
                'gpus': [{'index': g, 'uuid': f'GPU-{g}', 'mem_used_mib': 0} for g in range(8)], 'apps': list(self.apps[n])}

    def _launch(self, n, script_local):
        args = shlex.split(Path(script_local).read_text().strip().splitlines()[-1])
        job_remote, rank = args[-3], int(args[-2])
        self.launched.append((n, rank))
        job = json.loads(self.local(job_remote).read_text())
        attempt = json.loads(self.local(job['metadata_directory'] + '/attempt.json').read_text())
        out, life, run = job['output_root'], attempt['lifecycle_run_root'], attempt['run']
        self.apps[n] = [{'pid': 4000 + rank, 'gpu_index': 0, 'cwd': out, 'cmdline': 'python3 -m torch.distributed.run',
                         'token': 'unknown'}]
        t = self.clock()
        self.write(f'{out}/node{rank}.pid.json', {'pid': 4000 + rank, 'started_unix': t, 'node': n})
        for k in range(8 * rank, 8 * rank + 8):
            self.write(f'{life}/{run}.signals.rank{k}/signals.jsonl',
                       b''.join(json.dumps({'kind': 'train_step', 'step': s}).encode() + b'\n' for s in range(1, 31)))
            self.write(f'{life}/{run}.rank{k}.json', {'rank': k})
        if not self.hang:
            self.write(f'{out}/node{rank}.exit.json', {'exit_code': self.exit_codes[rank], 'started_unix': t,
                                                       'finished_unix': t + 600.0, 'node': n})
            self.write(f'{out}/node{rank}.release.json', {'idle': True})
        return {'pid': 3000 + rank, 'hostname': R.HOSTNAMES[R.node_index(n)], 'utc': 'now'}

    def leases(self):
        d = self.local('/data/infrarsi/leases')
        return sorted(p.name for p in d.glob('WINH*.json')) if d.exists() else []


def fake_build(cluster: Cluster):
    """A minimal bundle with the real layout: attempt/job/deployment files that variant.remote_layout reads."""
    counter = {'n': 0}

    def build(point, *, workdir: Path, cap_s: float) -> Path:
        counter['n'] += 1
        name = f'flash32_joint_tp8_p1_lp_rsi_unified_test_a_0000000{counter["n"]}_a{counter["n"]:032x}'
        local = workdir / name
        (local / 'code').mkdir(parents=True)
        meta, out = f'{V.FN_ROOT}/metadata/{name}', f'{V.FN_ROOT}/runs/{name}'
        attempt = {'run': name, 'attempt_id': f'a{counter["n"]:032x}', 'lifecycle_run_root': out + '/training'}
        job = {'name': name, 'group': 'A', 'metadata_directory': meta, 'output_root': out, 'master_port': 48270,
               'command': ['bash', V.FORMAL_SHELL], 'env': {}, 'timeout_seconds': int(cap_s)}
        (local / 'attempt.json').write_bytes(V.encode(attempt))
        (local / 'job.json').write_bytes(V.encode(job))
        (local / 'code' / 'run_precision_screen.py').write_text('# test\n')
        files = [{'local': str(local / n), 'remote': f'{meta}/{n}'} for n in ('attempt.json', 'job.json')]
        files.append({'local': str(local / 'code' / 'run_precision_screen.py'), 'remote': f'{meta}/code/run_precision_screen.py'})
        (local / 'deployment.json').write_bytes(V.encode({'files': files}))
        return local
    return build


def judge_ok(local, reference, gates, role, point, cap):
    """Stub of X6's pipeline: always numerics E0; completeness as guards.complete_verified (4 node exits 0, 32 logs)."""
    from infrarsi.ext.common.guards import GuardResult
    g = GuardResult()
    exits = [json.loads(p.read_text()) for p in sorted(Path(local, 'nodes').glob('node*.exit.json'))]
    if len(exits) != 4 or any(e.get('exit_code') != 0 for e in exits):
        g.reject('complete_verified', f'node exits {[e.get("exit_code") for e in exits]}')
    if len(list(Path(local, 'signals').glob('rank*.jsonl'))) != 32:
        g.reject('complete_verified', 'signal logs missing')
    verdict = {'judge_valid': True, 'numerics_class': 'E0', 'reasons': [], 'metric_s': 12.5, 'details': {'role': role}}
    return verdict, g, {'summary': {'stub': True}, 'memory': {'max_peak_reserved_gib': None}}


def task():
    return EnvTask.from_dict({'id': 'F1-FN', 'env': 'flashnext', 'split': 'external', 'arm': 'F-RND',
                              'metric_name': 'step_time_s', 'action_space': 'knobs',
                              'knob_schema': {'properties': {'implementation_stack': {'enum': ['native_rollbacks']}}},
                              'frozen_files': {}, 'mutable_globs': [], 'max_patch_lines': 0,
                              'budget': {'screens': 12, 'llm_tokens': 0, 'wall_seconds': 18000, 'gpu_hours': 192},
                              'margin': {'kind': 'rsd', 'floor': 0.03, 'k': 3}, 'tau': 0.0953, 'world_size': 32,
                              'facts': {}})


def runner(cluster: Cluster, tmp: Path, **kw):
    kw.setdefault('build_fn', fake_build(cluster))
    kw.setdefault('judge_fn', judge_ok)
    return R.FlashNextRunner(cluster.ops, nodes=cluster.nodes, runtime_dir=str(tmp / 'rt'), out_dir=tmp / 'out',
                             clock=cluster.clock, sleep=cluster.clock.sleep, **kw)
# Provenance: X6 overwrote this file by accident at 08:14 and restored it at 08:20 from its bytecode (source of 08:10:35).
# X5 checked it at 10:55 against its own session record: the code is identical; the six comments the restore could not
# recover are back. X6's helpers live in x6_helpers.py.
