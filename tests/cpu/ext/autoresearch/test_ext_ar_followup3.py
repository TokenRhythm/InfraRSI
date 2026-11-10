"""X1 follow-up 3, Codex W9 round 2 M-02: lease cleanup state in every ENV-C arm result, fail closed when incomplete."""
import json

import pytest

from infrarsi.ext.autoresearch import arms, cli, source
from infrarsi.ext.autoresearch.runner import AutoresearchRunner
from infrarsi.ext.common.remote_exec import ContainmentError, LeaseError, NodeError, gpu_claim_path, node_lease_path
from x1_ar_helpers import Node, make_assets


def _no_backoff(monkeypatch):
    import infrarsi.ext.common.remote_exec as rx
    orig = rx.GpuLease.release
    monkeypatch.setattr(rx.GpuLease, 'release', lambda self, **kw: orig(self, **{'sleep': lambda s: None, **kw}))


def _stuck_release(ops):
    real = ops.call

    def call(func, _timeout=300, **kw):
        if func == 'release_status':
            raise NodeError('ssh: connection reset')
        return real(func, _timeout, **kw)
    ops.call = call
    return real


def test_arm_lease_fails_closed_when_claims_stay(tmp_path, monkeypatch):
    _no_backoff(monkeypatch)
    node = Node(tmp_path)
    events = []
    with pytest.raises(LeaseError) as e:
        with arms.arm_lease(node.ops, 'arm1', [0], uids=[64004], events=events):
            _stuck_release(node.ops)
    assert sorted(e.value.kept) == sorted([gpu_claim_path('WINH0', 0), node_lease_path('WINH0')]) and len(e.value.failures) == 2
    final = events[-1]
    assert final['kind'] == 'release' and final['complete'] is False and final['kept'] == e.value.kept
    assert node.ops.local(node_lease_path('WINH0')).exists()           # still held: nothing pretends it was released


def test_arm_lease_records_a_failed_acquire(tmp_path, monkeypatch):
    _no_backoff(monkeypatch)
    node = Node(tmp_path)
    real = node.ops.call

    def call(func, _timeout=300, **kw):                                  # the GPU claim fails, then its rollback too
        if func == 'acquire' and kw.get('path') == gpu_claim_path('WINH0', 0):
            raise NodeError('ssh: connection reset')
        if func == 'release_status':
            raise NodeError('ssh: connection reset')
        return real(func, _timeout, **kw)
    node.ops.call = call
    events = []
    with pytest.raises(LeaseError) as e:                                # the rollback could not confirm: fail closed
        with arms.arm_lease(node.ops, 'arm2', [0], uids=[64004], events=events):
            pytest.fail('started')
    assert events[-1]['kind'] == 'acquire_failed' and events[-1]['complete'] is False
    assert node_lease_path('WINH0') in e.value.kept and isinstance(e.value.__cause__, NodeError)


def test_cli_result_records_the_final_lease_state_on_failure(tmp_path, monkeypatch):
    _no_backoff(monkeypatch)
    assets = make_assets(tmp_path)
    assets['reference'] = {**assets['reference'], 'val_bpb': 0.996365, 'num_steps': 936, 'attempt_id': 'a21767e9dbd4f',
                           'node': 'WINH0', 'gpu': [0], 'prepare_variant': 'h2d_sync', 'first_losses': source.R2['first_losses']}
    from infrarsi.ext.autoresearch import task as taskmod
    real_task = taskmod.env_c_task
    monkeypatch.setattr(cli, 'env_c_task', lambda arm, **kw: real_task(arm, assets=assets, **kw))
    monkeypatch.setattr(cli, 'EXEC_ROOT', tmp_path / 'exec')
    node = Node(tmp_path)
    orig_init = AutoresearchRunner.__init__

    def fast_init(self, *a, **kw):
        kw.update(clock=node.clock, sleep=node.clock.advance, poll_s=5)
        orig_init(self, *a, **kw)
    monkeypatch.setattr(AutoresearchRunner, '__init__', fast_init)
    real_run = AutoresearchRunner.run

    def run_then_lose_the_node(self, *a, **kw):                          # the attempt works, the final release does not
        rec = real_run(self, *a, **kw)
        _stuck_release(node.ops)
        return rec
    monkeypatch.setattr(AutoresearchRunner, 'run', run_then_lose_the_node)
    with pytest.raises(LeaseError):
        cli.main(['c-base', '--run-id', 'cbx'], ops_factory=lambda n: node.ops)
    r = json.loads((tmp_path / 'exec' / 'cbx' / 'result.json').read_text(encoding='utf-8'))
    assert r['valid'] is True and r['lease_complete'] is False and r['error'].startswith('LeaseError')
    assert r['lease_final']['kind'] == 'release' and r['lease_final']['kept']


def test_cli_ok_result_and_group_release_status(tmp_path, monkeypatch, capsys):
    node = Node(tmp_path)
    assert cli.main(['group-lease', '--run-id', 'grp'], ops_factory=lambda n: node.ops) == 0
    lease = node.ops.local(node_lease_path('WINH0'))
    held = json.loads(lease.read_text(encoding='utf-8'))
    lease.write_text(json.dumps({**held, 'attempt_id': 'run:someone-else'}), encoding='utf-8')        # taken over
    capsys.readouterr()
    assert cli.main(['group-release', '--run-id', 'grp'], ops_factory=lambda n: node.ops) == 3          # fail closed
    out = json.loads(capsys.readouterr().out)
    assert out['released'] is False and out['state'] == 'foreign' and lease.exists()


def test_group_lease_acquire_with_a_lost_answer(tmp_path):
    node = Node(tmp_path)
    real = node.ops.call

    def lossy(func, _timeout=300, **kw):
        if func == 'acquire' and kw.get('path') == node_lease_path('WINH0'):
            real(func, _timeout, **kw)
            raise NodeError('ssh: answer lost')
        return real(func, _timeout, **kw)
    node.ops.call = lossy
    with pytest.raises(NodeError):
        cli.group_lease(node.ops, 'grp2', [0, 2, 4, 6])
    assert not node.ops.local(node_lease_path('WINH0')).exists()       # rolled back, owner-checked


def test_c_rnd_receipts_carry_the_frozen_space_and_failure_classes(tmp_path):
    """AMENDMENT_EXT_007 §4.4: every C-RND receipt names the frozen knob space; every receipt classifies a failed run;
    the C-RND trace reports its space coverage."""
    from infrarsi.ext.autoresearch import knobs
    from x1_ar_helpers import ar_task
    node = Node(tmp_path, [{}, {'crash': True}])
    rnd = ar_task(tmp_path, 'C-RND')
    with arms.arm_lease(node.ops, 'rnd9', [0], uids=[64004]):
        r = arms.make_runner(node.ops, rnd, gpu=0, run_id='rnd9', clock=node.clock, sleep=node.clock.advance, poll_s=5)
        ok = r.run(rnd, {}, None, 'screen', 'rnd9', candidate_id='c0')
        bad = r.run(rnd, knobs.sample_order(42)[0], None, 'screen', 'rnd9', candidate_id='c1')
    sha = knobs.frozen_reference()['sha256']
    assert ok['knob_space_sha256'] == sha == bad['knob_space_sha256'] and ok['failure_class'] is None
    assert bad['failure_class'] == 'oom' and not bad['valid']             # the synthetic crash log is a CUDA OOM
    assert 'failure_class' not in (ok.get('summary') or {})                 # never part of an agent's receipt digest
    node2 = Node(tmp_path / 'b')
    cis = ar_task(tmp_path / 'b', 'C-IS')
    with arms.arm_lease(node2.ops, 'is9', [0], uids=[64003]):
        rec = arms.make_runner(node2.ops, cis, gpu=0, run_id='is9', clock=node2.clock, sleep=node2.clock.advance,
                               poll_s=5).run(cis, {}, None, 'screen', 'is9', candidate_id='c0')
    assert 'knob_space_sha256' not in rec and rec['failure_class'] is None
    node3 = Node(tmp_path / 'c')
    rnd3 = ar_task(tmp_path / 'c', 'C-RND')
    with arms.arm_lease(node3.ops, 'rnd10', [0], uids=[64004]):
        tr = arms.run_c_rnd(rnd3, arms.make_runner(node3.ops, rnd3, gpu=0, run_id='rnd10', clock=node3.clock,
                                                   sleep=node3.clock.advance, poll_s=5), run_id='rnd10',
                            clock=node3.clock).to_dict()
    ex = tr['extra']
    screened = sum(1 for s in tr['screens'] if s['candidate_id'] != 'c0')
    assert ex['space_size'] == 47 and ex['knob_space_sha256'] == sha and ex['space_coverage'] == round(screened / 47, 4)


def test_d20_locked_skips_a_baseline_selection(tmp_path, monkeypatch):
    """O decision D20 (AMENDMENT_EXT_003 Q2 principle): an arm whose final selection is the unchanged baseline gets no
    locked pair - nothing runs, the result says skipped, y = 0, invalid_accept_locked 0, both halves null, exit 0."""
    monkeypatch.setattr(cli, 'EXEC_ROOT', tmp_path / 'exec')
    base = source.base_train()
    cases = {
        'agent.py': (tmp_path / 'best.train.py', base),                                         # a train.py file
        'cis.json': (tmp_path / 'trace.json', {'schema': 'infrarsi-search-trace-v1', 'selected': 'c0',
                                              'selected_overlay': {}, 'selected_config': {}}),  # C-IS / C-RND chose c0
        'nat.json': (tmp_path / 'nat' / 'trace.json', {'schema': 'ext-basic-agent-trace-v1', 'selected': None}),
    }
    for name, (path, content) in cases.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content if isinstance(content, str) else json.dumps(content), encoding='utf-8')
        rc = cli.main(['locked', '--run-id', f'lk-{name}', '--arm', 'C-IS', '--from', str(path), '--arm-index', '1'],
                      ops_factory=lambda n: pytest.fail('a baseline selection must not touch the node'),
                      driver_factory=lambda: None)
        r = json.loads((tmp_path / 'exec' / f'lk-{name}' / 'result.json').read_text(encoding='utf-8'))
        assert rc == 0 and r['skipped'] == 'selection_is_baseline' and r['y'] == 0.0 and r['invalid_accept_locked'] == 0
        assert r['baseline'] is None and r['candidate'] is None and r['order'] is None and r['lease_complete'] is True
        assert r['selection_sha256'] == source.BASE_TRAIN_SHA256
    changed = tmp_path / 'changed.train.py'
    changed.write_text(base.replace('DEVICE_BATCH_SIZE = 128', 'DEVICE_BATCH_SIZE = 64'), encoding='utf-8')
    assert arms.baseline_skip(changed.read_text(encoding='utf-8')) is None                  # a real change is re-scored
    node = Node(tmp_path / 'n')
    from x1_ar_helpers import ar_task
    task = ar_task(tmp_path / 'n', 'C-IS')
    with arms.arm_lease(node.ops, 'lk', [0], uids=[64003]):
        r = arms.make_runner(node.ops, task, gpu=0, run_id='lk', clock=node.clock, sleep=node.clock.advance, poll_s=5)
        assert arms.locked_pair(r, task, base, run_id='lk', arm_index=2)['skipped'] == 'selection_is_baseline'
    assert node.scripts == []                                                                  # nothing ran
