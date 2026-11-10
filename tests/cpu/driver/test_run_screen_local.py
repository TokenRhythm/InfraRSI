"""run_screen end to end against LocalRemote: code sync (and re-hashing of cached trees), leases, occupancy refusal,
STOP, collection, storage, fail-closed data resolution and reference loading before any lease, site env."""
import json

import pytest

from driver_helpers import DUMMY_MOD, fake_index, indexed_remote, make_request, sim_launcher, simulated, wait_for_stop_then

from infrarsi.harness import OccupiedError, codesync, run_screen
from infrarsi.harness import launch as L
from infrarsi.harness.remote import LocalRemote
from infrarsi.harness.runner import ReferenceUnusable, observe


def _leases(remote):
    d = remote.local('/data/infrarsi/leases')
    return sorted(p.name for p in d.glob('*.json')) if d.exists() else []


def _launched(remote):
    return [c for c in remote.calls if c[1] in ('launch_wrapper', 'acquire_leases')]


def test_run_screen_happy_path(tmp_path, fsroot):
    r = indexed_remote(fsroot, launcher=sim_launcher())
    req = make_request('D2', nodes=['WINH2'])
    rec = run_screen(req, remote=r, poll_s=0.01, local_dir=tmp_path / 'receipts', log=lambda m: None)
    assert rec['valid'], rec['invalid_reasons']
    assert rec['code_digest'] == rec['harness']['code_sync']['code_digest']   # other builders edit the tree live
    assert r.local(f'/data/infrarsi/receipts/{req.attempt_id}.json').exists()
    assert (tmp_path / 'receipts' / f'{req.attempt_id}.json').exists()
    assert _leases(r) == []                                     # released after a clean exit
    spec = json.loads(r.local(f'{req.out_dir}/launch/node0.json').read_text())
    assert spec['argv'][:3] == ['/mnt/base/qwen-sft/.venv-megatron-qwen35/bin/python', '-m', 'torch.distributed.run']
    assert '-m' in spec['argv'] and 'infrarsi.adapters.megatron_core.generic_train' in spec['argv']
    assert spec['env']['PYTHONPATH'].startswith(rec['harness']['code_sync']['code_root'] + ':')
    assert spec['env']['NCCL_SOCKET_IFNAME'] == 'eth0' and spec['env']['GLOO_SOCKET_IFNAME'] == 'eth0'
    assert 'CUDA_VISIBLE_DEVICES' not in spec['env']


def test_two_node_launch_specs(tmp_path, fsroot):
    r = indexed_remote(fsroot, launcher=sim_launcher())
    req = make_request('D5', nodes=['WINH2', 'WINH3'])
    rec = run_screen(req, remote=r, poll_s=0.01, local_dir=None, log=lambda m: None)
    assert rec['valid'], rec['invalid_reasons']
    s0 = json.loads(r.local(f'{req.out_dir}/launch/node0.json').read_text())
    s1 = json.loads(r.local(f'{req.out_dir}/launch/node1.json').read_text())
    i = s0['argv'].index('--node_rank')
    assert s0['argv'][i + 1] == '0' and s1['argv'][i + 1] == '1'
    assert s0['argv'][s0['argv'].index('--master_addr') + 1] == s1['argv'][s1['argv'].index('--master_addr') + 1] == '172.27.0.12'
    assert s0['argv'][s0['argv'].index('--nnodes') + 1] == '2'
    assert all(s['env']['GLOO_SOCKET_IFNAME'] == s['env']['NCCL_SOCKET_IFNAME'] == 'eth0' for s in (s0, s1))


def test_site_network_env_is_immutable(monkeypatch):
    """Codex R1 8: GLOO_SOCKET_IFNAME=eth0 next to NCCL_SOCKET_IFNAME; a request (or anything that slipped through the
    whitelist) cannot override either."""
    for key in ('GLOO_SOCKET_IFNAME', 'GLOO_DEVICE_TRANSPORT', 'NCCL_SOCKET_IFNAME', 'TP_SOCKET_IFNAME'):
        with pytest.raises(ValueError):
            L.check_env({key: 'lo'})
    req = make_request('D5', nodes=['WINH2', 'WINH3'], config={'env': {'NCCL_ALGO': 'Ring', 'CUDA_DEVICE_MAX_CONNECTIONS': 1}})
    env = L.node_env(code_root='/c', request=req, node_rank=1)
    assert env['GLOO_SOCKET_IFNAME'] == env['NCCL_SOCKET_IFNAME'] == 'eth0' and env['NCCL_ALGO'] == 'Ring'
    monkeypatch.setattr(L, 'check_env', lambda e: {'GLOO_SOCKET_IFNAME': 'docker0', 'NCCL_SOCKET_IFNAME': 'lo'})
    env = L.node_env(code_root='/c', request=req, node_rank=0)              # site values are applied last
    assert env['GLOO_SOCKET_IFNAME'] == env['NCCL_SOCKET_IFNAME'] == 'eth0'


def _frozen_snapshot(fsroot):
    import shutil
    snap = fsroot / 'pkg'                                       # frozen copy: the live tree changes under other builders
    shutil.copytree(codesync.PACKAGE_DIR, snap, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.md'))
    return snap


def test_code_uploaded_once_per_digest(tmp_path, fsroot):
    snap = _frozen_snapshot(fsroot)
    r = LocalRemote(fsroot)
    a = codesync.ensure_code(r, 'WINH0', package_dir=snap)
    b = codesync.ensure_code(r, 'WINH0', package_dir=snap)
    assert a['uploaded'] and not b['uploaded'] and a['code_root'] == b['code_root'] and b['rehashed_files'] > 50
    root = r.local(a['code_root'])
    assert (root / 'infrarsi' / 'harness' / 'node_wrapper.py').exists() and (root / 'MANIFEST.json').exists()
    man = codesync.tree_manifest(snap)
    assert all(not k.endswith(('.pyc', '.md')) and '__pycache__' not in k for k in man['files'])
    assert man['code_digest'] == codesync.code_digest(snap) == a['code_digest']
    assert codesync.build_tar(man, snap) == codesync.build_tar(man, snap)       # deterministic bytes


@pytest.mark.parametrize('damage', ['modified', 'missing', 'extra_module', 'pycache'])
def test_cached_code_tree_is_rehashed_before_reuse(tmp_path, fsroot, damage):
    """Codex R1 5 (codesync half): a cached tree is never trusted from its MANIFEST.json alone."""
    snap = _frozen_snapshot(fsroot)
    r = LocalRemote(fsroot)
    a = codesync.ensure_code(r, 'WINH0', package_dir=snap)
    pkg = r.local(a['code_root']) / 'infrarsi'
    target = pkg / 'harness' / 'validate.py'
    if damage == 'modified':
        target.write_text(target.read_text(encoding='utf-8') + '\nREQUIRE_UTC_FIELDS = False\n', encoding='utf-8')
    elif damage == 'missing':
        target.unlink()
    elif damage == 'extra_module':
        (pkg / 'harness' / 'sitecustomize.py').write_text('import os\n', encoding='utf-8')
    else:
        (pkg / 'harness' / '__pycache__').mkdir()
        (pkg / 'harness' / '__pycache__' / 'validate.cpython-312.pyc').write_bytes(b'\x00' * 16)
    with pytest.raises(RuntimeError, match='re-hashing'):
        codesync.ensure_code(r, 'WINH0', package_dir=snap)
    rep = r.call('WINH0', 'verify_code', code_dir=a['code_root'], files=codesync.tree_manifest(snap)['files'])
    assert not rep['ok'] and (rep['mismatched'] or rep['missing'] or rep['extra'])


def test_failed_install_leaves_nothing(tmp_path, fsroot):
    from infrarsi.harness.remote import RemoteError
    snap = {'a.py': b'x = 1\n'}
    man = codesync.tree_manifest(snap=snap)
    bad = dict(man, files={'a.py': '0' * 64})                    # file changed after hashing
    r = LocalRemote(fsroot)
    tar = '/data/infrarsi/code/.incoming/t.tar'
    r.put_bytes('WINH0', codesync.build_tar(man, snap=snap), tar)
    with pytest.raises(RemoteError):
        r.call('WINH0', 'install_code', tar_path=tar, code_dir='/data/infrarsi/code/' + man['code_digest'], manifest=bad)
    code = r.local('/data/infrarsi/code')
    assert sorted(p.name for p in code.iterdir()) == ['.incoming'] and not list((code / '.incoming').iterdir())
    with pytest.raises(RuntimeError):
        codesync.build_tar(bad, snap=snap)                      # the driver refuses an inconsistent snapshot too


def test_refuses_foreign_process(tmp_path, fsroot):
    r = indexed_remote(fsroot, launcher=sim_launcher(),
                       gpu_apps={2: [{'pid': 777, 'name': 'python', 'used_mib': 30000, 'sid': 777}]})
    with pytest.raises(OccupiedError) as e:
        run_screen(make_request('D2', nodes=['WINH2']), remote=r, poll_s=0.01, local_dir=None, log=lambda m: None)
    assert 'WINH2' in e.value.report
    assert not [c for c in r.calls if c[1] == 'launch_wrapper'] and _leases(r) == []


def test_refuses_unexplained_gpu_memory(tmp_path, fsroot):
    r = indexed_remote(fsroot, launcher=sim_launcher())
    r.gpu_mem = {2: {5: 40000}}
    with pytest.raises(OccupiedError):
        run_screen(make_request('D2', nodes=['WINH2']), remote=r, poll_s=0.01, local_dir=None, log=lambda m: None)


def test_refuses_foreign_lease(tmp_path, fsroot):
    r = indexed_remote(fsroot, launcher=sim_launcher())
    p = r.local('/data/infrarsi/leases/WINH2.json')
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({'owner': 'someone', 'run_id': 'x', 'attempt_id': 'y'}))
    with pytest.raises(OccupiedError):
        run_screen(make_request('D2', nodes=['WINH2']), remote=r, poll_s=0.01, local_dir=None, log=lambda m: None)
    assert p.exists()                                           # never removes a lease it does not own


def test_run_scope_lease_is_honoured_and_kept(tmp_path, fsroot):
    from infrarsi.harness import leases
    r = indexed_remote(fsroot, launcher=sim_launcher())
    req = make_request('D2', nodes=['WINH2'])
    leases.acquire(r, ['WINH2'], leases.make_lease(req.run_id, 'run:' + req.run_id, ['WINH2'], scope='run'))
    rec = run_screen(req, remote=r, poll_s=0.01, local_dir=None, log=lambda m: None)
    assert rec['valid'] and _leases(r) == ['WINH2.json']       # the dispatcher releases its own run lease


def test_local_stop_request(tmp_path, fsroot):
    r = indexed_remote(fsroot, launcher=sim_launcher(wait_for_stop_then(stop_at=4), threaded=True))
    req = make_request('D2', nodes=['WINH2'])
    stop_dir = tmp_path / 'stop'
    stop_dir.mkdir()
    (stop_dir / f'{req.attempt_id}.json').write_text(json.dumps({'reason': 'operator test'}))
    rec = run_screen(req, remote=r, poll_s=0.01, stop_dir=stop_dir, local_dir=None, log=lambda m: None)
    stop = json.loads(r.local(f'{req.out_dir}/STOP.json').read_text())
    assert stop['kind'] == 'request' and stop['reason'] == 'operator test'
    assert not rec['valid'] and rec['steps_completed'] == 4 and rec['harness']['failure_kind'] == 'stopped'
    assert _leases(r) == []


def test_caller_stop_check_and_wall_cap(tmp_path, fsroot):
    r = indexed_remote(fsroot, launcher=sim_launcher(wait_for_stop_then(stop_at=2), threaded=True))
    req = make_request('D2', nodes=['WINH2'], wall_seconds_cap=0)
    rec = run_screen(req, remote=r, poll_s=0.01, local_dir=None, log=lambda m: None)
    assert rec['harness']['stop']['kind'] == 'wall_cap' and rec['harness']['failure_kind'] == 'timeout'


def test_bad_env_rejected_before_anything(tmp_path, fsroot):
    r = indexed_remote(fsroot, launcher=sim_launcher())
    for env in ({'LD_PRELOAD': 'x.so'}, {'NCCL_SOCKET_IFNAME': 'lo'}, {'GLOO_SOCKET_IFNAME': 'lo'}):
        with pytest.raises(ValueError):
            run_screen(make_request('D2', nodes=['WINH2'], config={'env': env}), remote=r, log=lambda m: None)
    assert r.calls == []


def test_partial_multinode_launch_failure_stops_and_finishes(tmp_path, fsroot, monkeypatch):
    import threading
    import time as _t
    from infrarsi.harness import runner
    from infrarsi.harness.remote import RemoteError
    monkeypatch.setattr(runner, 'PARTIAL_LAUNCH_SETTLE_S', 0.05)

    def launcher(k, kw, remote):
        spec = json.loads(remote.local(kw['spec_path']).read_text())
        out = remote.local(spec['out_dir'])
        if spec['node_index'] == 1:
            raise RemoteError('WINH3: transport failed')

        def wrapper0():                                       # node 0 wrapper: honours STOP.json, writes its exit
            while not (out / 'STOP.json').exists():
                _t.sleep(0.01)
            (out / 'exit_node0.json').write_text(json.dumps({'exit_code': 0, 'reason': 'stop', 'wall_s': 1.0,
                                                             'release': {'process_group_gone': True}}))
        threading.Thread(target=wrapper0, daemon=True).start()
        return {'pid': 1, 'hostname': 'h', 'utc': ''}

    r = indexed_remote(fsroot, launcher=launcher)
    req = make_request('D5', nodes=['WINH2', 'WINH3'])
    rec = run_screen(req, remote=r, poll_s=0.01, local_dir=None, log=lambda m: None)
    ex1 = json.loads(r.local(f'{req.out_dir}/exit_node1.json').read_text())
    assert ex1['reason'] == 'launch_failed' and ex1['synthesized_by_driver']
    assert not rec['valid'] and rec['harness']['failure_kind'] == 'launch_error'
    assert _leases(r) == []


# ---------------------------------------------------------------- finding 2: data resolution fails closed

def test_data_index_checks(tmp_path, fsroot):
    r = LocalRemote(fsroot)
    facts = fake_index(r)['D2']

    def run(data_for, plan='A'):
        r.launcher = sim_launcher(lambda req, out: DUMMY_MOD.simulate_attempt(req, out, data=data_for(req)))
        return run_screen(make_request('D2', nodes=['WINH2'], plan=plan, purpose='screen' if plan == 'A' else 'score'),
                          remote=r, poll_s=0.01, local_dir=None, log=lambda m: None)
    rec = run(lambda req: facts[req['input_plan']])
    ok = {c['name'] for c in rec['harness']['checks'] if c['ok']}
    assert rec['valid'] and {'data.index_task', 'data.manifest_sha256', 'data.index_order_digest', 'data.physical_tokens',
                             'data.supervised_tokens', 'data.order_digest', 'data.consumed_digest',
                             'data.consumed_per_update', 'data.receipt_supervised_vector'} <= ok
    assert run(lambda req: facts[req['input_plan']], plan='B')['valid']
    bad = run(lambda req: dict(facts['A'], sup=[x + 1 for x in facts['A']['sup']]))
    assert not bad['valid'] and any('data.supervised_tokens' in x for x in bad['invalid_reasons'])
    swapped = run(lambda req: dict(facts['A'], order=facts['B']['order'], order_digest=facts['B']['order_digest']))
    assert not swapped['valid'] and any('data.order_digest' in x for x in swapped['invalid_reasons'])
    assert any('data.consumed_digest' in x for x in swapped['invalid_reasons'])


@pytest.mark.parametrize('broken', ['no_index', 'task_missing', 'no_manifest_bytes', 'malformed_entry', 'sha_mismatch'])
def test_unresolved_data_refuses_before_any_lease(fsroot, broken):
    """Codex R1 2: an absent, unreadable, malformed or unverifiable INDEX.json / manifest stops a real-data launch
    before occupancy checks, leases or launch."""
    from infrarsi.harness.dataindex import INDEX_PATH
    r = LocalRemote(fsroot, launcher=sim_launcher())
    if broken != 'no_index':
        fake_index(r)
        idx = json.loads(r.local(INDEX_PATH).read_text())
        entry = idx['tasks']['D2']
        if broken == 'task_missing':
            idx['tasks'].pop('D2')
        elif broken == 'no_manifest_bytes':
            r.local(entry['manifest']).unlink()
        elif broken == 'malformed_entry':
            entry['data_order_digest'] = {}
        elif broken == 'sha_mismatch':
            entry['manifest_sha256'] = '0' * 64
        r.local(INDEX_PATH).write_text(json.dumps(idx))
    req = make_request('D2', nodes=['WINH2'])
    with pytest.raises(Exception):
        run_screen(req, remote=r, poll_s=0.01, local_dir=None, log=lambda m: None)
    assert req.data_manifest is None and not _launched(r) and not [c for c in r.calls if c[1] == 'gpu_status']
    assert _leases(r) == []


def test_synthetic_task_runs_without_index(tmp_path, fsroot):
    from infrarsi.dispatch.handlers import synthetic_task
    from infrarsi.contracts import TaskSpec
    r = LocalRemote(fsroot, launcher=sim_launcher())
    req = make_request('S1', nodes=['WINH0'])
    req.task = synthetic_task(TaskSpec.load('S1')).raw
    rec = run_screen(req, remote=r, poll_s=0.01, local_dir=None, log=lambda m: None)
    assert rec['valid'], rec['invalid_reasons']
    assert req.data_manifest is None and rec['harness']['data_index'] == {'skipped': 'synthetic data'}


# ---------------------------------------------------------------- finding 3: references authenticated before launch

def test_unusable_reference_refuses_before_any_lease(tmp_path, fsroot):
    r = indexed_remote(fsroot, launcher=sim_launcher())
    fr = run_screen(make_request('D2', nodes=['WINH2']), remote=r, poll_s=0.01, local_dir=None, log=lambda m: None)
    path = fr['harness']['stored']
    good = run_screen(make_request('D2', nodes=['WINH2'], numerics_role='candidate', reference_receipt=path), remote=r,
                      poll_s=0.01, local_dir=None, log=lambda m: None)
    assert good['valid'] and good['harness']['numerics']['compared'] and good['harness']['numerics']['class'] == 'E0'
    stored = json.loads(r.local(path).read_text())
    stored['per_step'] = stored['per_step'][:1]                                  # one-row reference at the path
    r.local(path).write_text(json.dumps(stored))
    calls = len(r.calls)
    with pytest.raises(ReferenceUnusable):
        run_screen(make_request('D2', nodes=['WINH2'], numerics_role='candidate', reference_receipt=path), remote=r,
                   poll_s=0.01, local_dir=None, log=lambda m: None)
    assert not [c for c in r.calls[calls:] if c[1] in ('launch_wrapper', 'acquire_leases', 'gpu_status')]
    other = simulated(tmp_path, make_request('D2'))                             # dict disagreeing with the named path
    with pytest.raises(ReferenceUnusable):
        run_screen(make_request('D2', nodes=['WINH2'], numerics_role='candidate', reference_receipt='/data/elsewhere.json'),
                   reference=dict(other, harness=dict(other['harness'], stored='/data/x.json')), remote=r, log=lambda m: None)


def test_observe_reattaches(tmp_path, fsroot):
    r = indexed_remote(fsroot, launcher=sim_launcher())
    req = make_request('D2', nodes=['WINH2'])
    first = run_screen(req, remote=r, poll_s=0.01, local_dir=None, log=lambda m: None)
    again = observe(req.run_id, req.attempt_id, remote=r, poll_s=0.01, local_dir=None, log=lambda m: None)
    assert again['valid'] == first['valid'] and again['step_time'] == first['step_time']
