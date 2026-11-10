import base64
import io
import json
import os
import contextlib
import shlex
import sys
from pathlib import Path

import pytest

from infrarsi.ext.common import node_ext, remote_exec
from infrarsi.ext.common.env_runner import SCHEMA_RECEIPT, jsonable
from infrarsi.ext.common.remote_exec import (GpuLease, LocalNodeOps, OWNER, WinhNodeOps, gpu_claim_path, node_lease_path,
                                               nobody_cmd, occupancy_problems)
from infrarsi.harness import signing
from infrarsi.harness.leases import OccupiedError
from x1_common_helpers import ToyRunner, local_ops, toy_task

REQUIRED = ('schema', 'env', 'task_id', 'run_id', 'attempt_id', 'purpose', 'candidate_digest', 'frozen_digest', 'node', 'gpus',
            'world_size', 'valid', 'invalid_reasons', 'step_time', 'metric', 'judge', 'judge_valid', 'guard_rejections',
            'guard_reports', 'phases', 'utc_start', 'utc_end', 'wall_s', 'cost', 'deviations', 'harness_signature')


# ---------------------------------------------------------------- receipts of the base runner
def test_receipt_shape_signature_and_storage(tmp_path):
    ops = local_ops(tmp_path)
    runner = ToyRunner(ops, out_dir=tmp_path / 'local')
    rec = runner.run(toy_task(), {'workers': 4, 'flash': True}, None, 'screen', 'run1', candidate_id='c3')
    assert all(k in rec for k in REQUIRED) and rec['schema'] == SCHEMA_RECEIPT and signing.verify(rec)
    assert rec['valid'] and rec['metric'] == {'name': 'exec_time_s', 'value': 20.0, 'direction': 'min',
                                              'raw': {'score': pytest.approx(2.9957, abs=1e-4), 'judge_value': 20.0}}
    assert rec['step_time'] == {'mean_window_max_rank_s': 20.0} and rec['candidate_id'] == 'c3' and rec['deviations'] == ['D-T-1']
    assert rec['cost'] == {'gpu_seconds': 160.0, 'wall_seconds': 40.0, 'accounting_complete': True,
                           'basis': rec['cost']['basis']} and rec['phases']['total'] == 40.0
    assert runner.seen[0]['candidate_id'] == 'c3' and rec['summary'] == {'t': 20.0}
    stored = json.loads((tmp_path / 'local' / 'receipts' / f'{rec["attempt_id"]}.json').read_text(encoding='utf-8'))
    assert stored == rec
    # the attempt's node lease and GPU claims are gone afterwards
    assert not ops.local(node_lease_path('WINH0')).exists()
    assert not any(ops.local(gpu_claim_path('WINH0', g)).exists() for g in range(4))


def test_occupied_node_is_never_launched(tmp_path):
    foreign = [{'pid': 77, 'gpu_index': 5, 'used_mib': 900, 'cwd': '/root/other', 'cmdline': 'python train.py'}]
    runner = ToyRunner(local_ops(tmp_path, apps=foreign))
    rec = runner.run(toy_task(), {}, None, 'screen', 'r')
    assert not rec['valid'] and rec['runner_error']['kind'] == 'occupied' and runner.seen == []
    assert rec['cost']['accounting_complete'] is False and rec['cost']['gpu_seconds'] == 0.0 and signing.verify(rec)
    assert 'foreign compute processes [77]' in rec['invalid_reasons'][0]
    busy = ToyRunner(local_ops(tmp_path, gpu_mem={2: 30000})).run(toy_task(), {}, None, 'screen', 'r')
    assert 'without a visible process on [2]' in busy['invalid_reasons'][0]
    ops = local_ops(tmp_path)
    ops.write_files({node_lease_path('WINH0'): json.dumps({'owner': 'infrarsi', 'run_id': 'E7', 'attempt_id': 'a1'})})
    leased = ToyRunner(ops).run(toy_task(), {}, None, 'screen', 'r')
    assert 'leased by infrarsi:E7/a1' in leased['invalid_reasons'][0]


def test_runner_errors_become_invalid_receipts(tmp_path):
    class Broken(ToyRunner):
        def execute(self, *a, **k):
            self.vclock.advance(30)
            raise RuntimeError('ssh dropped')
    rec = Broken(local_ops(tmp_path)).run(toy_task(), {}, None, 'screen', 'r')
    assert not rec['valid'] and rec['runner_error'] == {'kind': 'runner_error', 'type': 'RuntimeError', 'message': 'ssh dropped'}
    assert rec['cost'] == {**rec['cost'], 'gpu_seconds': 120.0, 'accounting_complete': False}


def test_held_mode_and_arm_scoped_claims(tmp_path):
    ops = local_ops(tmp_path)
    task = toy_task()
    with pytest.raises(OccupiedError):                    # held mode needs the run's node lease
        GpuLease(ops, env='toyenv', gpus=[2], mode='held', held_run_id='grp').acquire('arm1', 'a1')
    ops.write_files({node_lease_path('WINH0'): json.dumps({'owner': OWNER, 'run_id': 'grp', 'attempt_id': 'run:grp'})})
    arm = GpuLease(ops, env='toyenv', gpus=[2], mode='held', held_run_id='grp')
    arm.acquire('arm1', 'run:arm1', scope='run')
    claim = json.loads(ops.local(gpu_claim_path('WINH0', 2)).read_text(encoding='utf-8'))
    assert claim['owner'] == OWNER and claim['scope'] == 'run' and claim['purpose'] == 'ext:toyenv' and claim['gpus'] == [2]
    # attempts of the same arm run under its claim; another arm cannot take GPU 2
    runner = ToyRunner(ops, lease_mode='held', held_run_id='grp')
    runner.gpus = [2]
    assert runner.run(task, {}, None, 'screen', 'arm1')['valid']
    assert ops.local(gpu_claim_path('WINH0', 2)).exists()
    other = ToyRunner(ops, lease_mode='held', held_run_id='grp')
    other.gpus = [2]
    assert 'GPU 2 claimed' in other.run(task, {}, None, 'screen', 'arm2')['invalid_reasons'][0]
    assert arm.release() == [gpu_claim_path('WINH0', 2)]


def test_occupancy_rules():
    gpus = [{'index': i, 'uuid': f'G{i}', 'mem_used_mib': 0} for i in range(8)]
    ours = {'pid': 5, 'gpu_index': 4, 'cwd': '/data/infrarsi/ext/runs/autoresearch/r/c1/a/repo', 'cmdline': 'python'}
    ok = lambda lease: True                              # noqa: E731
    assert occupancy_problems({'gpus': gpus, 'apps': [ours]}, [0], lease=None, lease_ok=ok) == []
    assert 'target GPUs' in occupancy_problems({'gpus': gpus, 'apps': [ours]}, [4], lease=None, lease_ok=ok)[0]
    assert occupancy_problems({'gpus': gpus, 'apps': []}, [9], lease=None, lease_ok=ok) == ['target GPUs [9] not present on the node']
    assert occupancy_problems({'gpus': gpus, 'apps': [], 'error': 'smi'}, [0], lease=None, lease_ok=ok)[0].startswith('probe error')


# ---------------------------------------------------------------- commands and node functions
def test_nobody_command():
    cmd = nobody_cmd('python train.py', home='/r/h', cwd='/r/repo', path_dirs=('/env/bin',), gpus=(2,), token='isx-0123456789',
                     extra={'HF_HOME': '/r/h/hf'}, timeout=600)
    assert cmd.startswith('timeout --kill-after=10 600 setpriv --reuid=65534 --regid=65534 --clear-groups --no-new-privs env -i ')
    for part in ('HOME=/r/h', 'PATH=/env/bin:/usr/bin:/bin', 'CUDA_VISIBLE_DEVICES=2', 'HF_HUB_OFFLINE=1', 'INFRARSI_TOKEN=isx-0123456789',
                 'HF_HOME=/r/h/hf', "bash -c 'export PATH=/env/bin:/usr/bin:/bin; cd /r/repo && python train.py'"):
        assert part in cmd
    assert shlex.split(nobody_cmd("echo it's", home='/h'))[-1] == "export PATH=/usr/bin:/bin; echo it's"
    assert " bash -lc 'export" in nobody_cmd('true', home='/h', login=True)


def test_node_ext_functions(tmp_path):
    a, b = str(tmp_path / 'x' / 'a.txt'), str(tmp_path / 'x' / 'b.bin')
    node_ext.write_files({a: base64.b64encode(b'hello').decode(), b: base64.b64encode(b'\0' * 10).decode()})
    assert base64.b64decode(node_ext.read_files([a])[a]) == b'hello'
    assert base64.b64decode(node_ext.read_files([b], max_bytes=3)[b]) == b'\0' * 3
    assert node_ext.hash_files([a, str(tmp_path / 'none')])[str(tmp_path / 'none')] is None
    assert node_ext.sizes([a]) == {a: 5}
    lease = str(tmp_path / 'leases' / 'WINH0.json')
    assert node_ext.acquire(lease, {'attempt_id': 'a1'})['ok'] and node_ext.acquire(lease, {'attempt_id': 'a1'})['ok']
    assert not node_ext.acquire(lease, {'attempt_id': 'a2'})['ok']
    assert not node_ext.release(lease, 'a2') and node_ext.release(lease, 'a1') and not os.path.exists(lease)
    with pytest.raises(ValueError):
        node_ext.remove_tree(str(tmp_path), str(tmp_path / 'x'))
    node_ext.remove_tree(str(tmp_path / 'x'), str(tmp_path))
    assert not (tmp_path / 'x').exists()
    with pytest.raises(ValueError):
        node_ext.kill_token('short')
    assert node_ext.listdir(str(tmp_path / 'leases')) == [] and node_ext.listdir(a) is None   # PROPOSED_CHANGES_X3 #3
    assert node_ext.listdir(str(tmp_path / 'nope')) is None


class ShippingRemote:
    """Executes WinhNodeOps' shipped command in-process: decodes the module source and runs it as __main__."""

    def __init__(self):
        self.commands, self.uploads = [], {}

    def put_bytes(self, node, data, path):
        self.uploads[path] = data
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)

    def sh(self, node, cmd, timeout=120):
        self.commands.append(cmd)
        if cmd.startswith(('mkdir -p ', 'rm -f ')):              # request-file housekeeping
            if cmd.startswith('rm -f '):
                Path(shlex.split(cmd)[2]).unlink(missing_ok=True)
            return {'code': 0, 'stdout': '', 'stderr': ''}
        parts = cmd.split()
        src = base64.b64decode(parts[1]).decode('utf-8')
        assert parts[2:7] == ['|', 'base64', '-d', '|', 'python3'] and parts[7] == '-'
        out = io.StringIO()
        old = sys.argv
        sys.argv = ['-', shlex.split(parts[8])[0]]
        try:
            with contextlib.redirect_stdout(out):
                exec(compile(src, 'node_ext', 'exec'), {'__name__': '__main__'})
        finally:
            sys.argv = old
        return {'code': 0, 'stdout': out.getvalue(), 'stderr': ''}


def test_winh_node_ops_protocol(tmp_path):
    ops = WinhNodeOps('WINH3', remote=ShippingRemote())
    p = str(tmp_path / 'f.txt')
    ops.write_files({p: 'abc'})
    assert ops.read_text(p) == 'abc' and ops.hash_files([p])[p] == node_ext.hash_files([p])[p]
    with pytest.raises(remote_exec.NodeError):
        ops.call('remove_tree', path=str(tmp_path), root=str(tmp_path / 'sub'))
    assert ops.remote.commands and all(c.startswith('echo ') for c in ops.remote.commands)
    big = 'x' * 300_000                                                  # beyond one command line (MAX_ARG_STRLEN 128 KiB)
    ops = WinhNodeOps('WINH3', remote=ShippingRemote(), request_dir=str(tmp_path / 'req'))
    ops.write_files({p: big})
    assert ops.read_text(p) == big and ops.remote.uploads                 # the request travelled as a file
    assert all(len(c) < remote_exec.ARGV_BUDGET for c in ops.remote.commands)
    assert any(' @' in c or "'@" in c for c in ops.remote.commands) and not list((tmp_path / 'req').glob('*.json'))   # removed


def test_jsonable_sanitizes_non_finite():
    assert jsonable({'a': float('nan'), 'b': [float('inf'), (1, 2)], 3: object}) == {'a': 'nan', 'b': ['inf', [1, 2]], '3': str(object)}


def test_local_node_ops_maps_data_paths(tmp_path):
    ops = LocalNodeOps(tmp_path)
    ops.write_files({'/data/infrarsi/ext/runs/x/y.txt': 'z'})
    assert (tmp_path / 'data' / 'infrarsi' / 'ext' / 'runs' / 'x' / 'y.txt').read_text() == 'z'
    assert ops.hash_files(['/data/infrarsi/ext/runs/x/y.txt']) == {'/data/infrarsi/ext/runs/x/y.txt': node_ext.hash_files(
        [str(tmp_path / 'data' / 'infrarsi' / 'ext' / 'runs' / 'x' / 'y.txt')]).popitem()[1]}
