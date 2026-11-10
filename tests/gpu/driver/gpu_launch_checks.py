"""GPU launch checks for the driver harness with dummy_entry.py (seconds of GPU time). Run from the repo root:

    python tests/gpu/driver/gpu_launch_checks.py single      # WINH2: 8-GPU screen (D2 shape), receipt + leases
    python tests/gpu/driver/gpu_launch_checks.py stop        # WINH2: long run stopped through run_screen's STOP path
    python tests/gpu/driver/gpu_launch_checks.py probe       # WINH2: fr_v0 chain on S2, simulated OOM at TP1 -> TP2
    python tests/gpu/driver/gpu_launch_checks.py refuse      # WINH3: a foreign lease makes run_screen refuse to launch
    python tests/gpu/driver/gpu_launch_checks.py two_node    # WINH2+WINH3: 16-rank rendezvous (D5 shape)

Every launch goes through run_screen, which checks occupancy and leases first and never touches other processes.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from infrarsi.contracts import RemotePaths, RunRequest, TaskSpec   # noqa: E402
from infrarsi.harness import OccupiedError, default_remote, run_screen   # noqa: E402
from infrarsi.harness import leases as Lz   # noqa: E402
from infrarsi.harness.probes import run_probe_chain   # noqa: E402

ENTRY = ('script', str(Path(__file__).with_name('dummy_entry.py')))
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / 'tests' / 'gpu' / 'driver' / '_receipts'


def req(task_id, nodes, run_id, **kw):
    t = TaskSpec.load(task_id)
    gpn = t.gpus if t.nodes == 1 else t.gpus // t.nodes
    dummy = kw.pop('dummy', {})
    cfg = kw.pop('config', {'tensor_model_parallel_size': 1})
    cfg = dict(cfg, dummy=dummy)
    kw.setdefault('numerics_role', 'reference')          # standalone base runs: no numerics reference to compare with
    return RunRequest(run_id=run_id, purpose=kw.pop('purpose', 'screen'), task=t.raw, config=cfg, nodes=nodes,
                      gpus_per_node=gpn, wall_seconds_cap=kw.pop('wall_seconds_cap', 900), **kw)


def summary(rec):
    h = rec['harness']
    return {'attempt_id': rec['attempt_id'], 'valid': rec['valid'], 'failure_kind': h['failure_kind'],
            'steps_completed': rec['steps_completed'], 'step_time': rec.get('step_time'),
            'invalid_reasons': rec['invalid_reasons'][:6], 'exits': [(e or {}).get('exit_code') for e in h['exits']],
            'exit_reasons': [(e or {}).get('reason') for e in h['exits']], 'leases_released': h.get('leases_released'),
            'stored': h.get('stored'), 'gpu_hours': h.get('gpu_hours'), 'master': f'{h.get("master_addr")}:{h.get("master_port")}',
            'warnings': h.get('warnings'), 'env': rec.get('env')}


def leases_now(r):
    return r.call('WINH2', 'read_leases', lease_dir=RemotePaths.leases)


def check_single(r):
    rec = run_screen(req('D2', ['WINH2'], 'GPUCHECK-D-single', dummy={'step_sleep_s': 0.15}), remote=r, entry=ENTRY,
                     poll_s=5, local_dir=OUT)
    return {'single': summary(rec), 'leases_after': leases_now(r)}


def check_stop(r):
    t0 = time.time()
    rec = run_screen(req('D2', ['WINH2'], 'GPUCHECK-D-stop', steps=400, dummy={'step_sleep_s': 0.5}), remote=r, entry=ENTRY,
                     poll_s=5, local_dir=OUT, stop_check=lambda: 'gpu check: driver STOP after 60 s' if time.time() - t0 > 60 else None)
    stop = r.read_json('WINH2', f'{RemotePaths.runs}/GPUCHECK-D-stop/{rec["attempt_id"]}/STOP.json')
    return {'stop': summary(rec), 'stop_file': stop, 'leases_after': leases_now(r)}


def check_probe(r):
    from infrarsi.discover import workload_facts
    from infrarsi.planner import FRChain
    t = TaskSpec.load('S2')
    chain = FRChain(t, workload_facts(t, remote=r))

    class Dummied:
        """Adds the dummy block (simulated OOM below TP2) to every probe config of the real fr_v0 chain."""
        def current(self):
            c = chain.current()
            return None if c is None else dict(c, dummy={'step_sleep_s': 0.1, 'oom_below_tp': 2})

        def advance(self, outcome, rec):
            chain.advance(outcome, rec)

    recs = run_probe_chain(Dummied(), t, ['WINH2'], 'GPUCHECK-D-probe', runner=lambda q, **kw: run_screen(
        q, remote=r, entry=ENTRY, poll_s=5, local_dir=OUT), wall_seconds_cap=600)
    return {'probe_chain': chain.result()['history'], 'status': chain.status,
            'fr_tp': (chain.config or {}).get('tensor_model_parallel_size'), 'probes': [summary(x) for x in recs],
            'leases_after': leases_now(r)}


def check_refuse(r):
    lease = {'schema': 'infrarsi-lease-v1', 'owner': 'gpu-check-foreign', 'run_id': 'FOREIGN', 'attempt_id': 'foreign1',
             'since': 'now', 'nodes': ['WINH3']}
    path = f'{RemotePaths.leases}/WINH3.json'
    r.write_json('WINH3', path, lease, exclusive=True)
    try:
        run_screen(req('D2', ['WINH3'], 'GPUCHECK-D-refuse'), remote=r, entry=ENTRY, poll_s=5, local_dir=OUT)
        outcome = 'LAUNCHED (unexpected)'
    except OccupiedError as e:
        outcome = f'refused: {e.report}'
    finally:
        r.call('WINH3', 'release_leases', lease_dir=RemotePaths.leases, nodes=['WINH3'], attempt_id='foreign1')
    return {'refuse': outcome, 'leases_after': leases_now(r)}


def check_two_node(r):
    rec = run_screen(req('D5', ['WINH2', 'WINH3'], 'GPUCHECK-D-two-node', dummy={'step_sleep_s': 0.15, 'allreduce_mb': 64}),
                     remote=r, entry=ENTRY, poll_s=5, local_dir=OUT)
    return {'two_node': summary(rec), 'leases_after': leases_now(r)}


if __name__ == '__main__':
    which = sys.argv[1]
    r = default_remote()
    for n in ('WINH2', 'WINH3'):
        print(n, json.dumps(Lz.occupancy(r, [n])[n], default=str)[:400], flush=True)
    res = {'single': check_single, 'stop': check_stop, 'probe': check_probe, 'refuse': check_refuse,
           'two_node': check_two_node}[which](r)
    print(json.dumps(res, indent=1, default=str))
