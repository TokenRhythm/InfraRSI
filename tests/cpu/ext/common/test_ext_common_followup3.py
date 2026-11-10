"""X1 follow-up 3 (PROPOSED_CHANGES_X3 items 6-8): held time includes the release, kept claims make the cost
incomplete; incomplete accounting reaches arm rows and labels; a claim whose acquire answer was lost is rolled back
owner-checked."""
import json

import pytest

from infrarsi.agent.search import SearchSession
from infrarsi.ext.common import trajectory as tj
from infrarsi.ext.common.env_judge import EnvJudge
from infrarsi.ext.common.remote_exec import GpuLease, NodeError, gpu_claim_path, node_lease_path
from infrarsi.ext.common.session_hooks import SessionHooks
from infrarsi.harness import signing
from infrarsi.harness.leases import OccupiedError
from x1_common_helpers import BASE_FILES, ToyRunner, local_ops, toy_task


# ---------------------------------------------------------------- item 6
def test_held_time_includes_the_release_rpcs(tmp_path):
    task = toy_task()
    ops = local_ops(tmp_path)
    r = ToyRunner(ops)
    real = ops.call

    def slow_release(func, _timeout=300, **kw):                           # each release RPC takes 5 s
        if func == 'release_status':
            r.vclock.advance(5)
        return real(func, _timeout, **kw)
    ops.call = slow_release
    rec = r.run(task, {'workers': 2}, None, 'screen', 'run1', candidate_id='c1')
    # 70 s in execute + 5 release RPCs (node lease and 4 GPU claims) x 5 s, on 4 GPUs
    assert rec['valid'] and rec['cost']['gpu_seconds'] == pytest.approx(4 * (70 + 5 * 5))
    assert rec['cost']['accounting_complete'] is True and 'incomplete_reasons' not in rec['cost']


def test_kept_claims_make_the_cost_incomplete(tmp_path):
    task = toy_task()
    ops = local_ops(tmp_path)
    real = ops.call

    def stuck(func, _timeout=300, **kw):                                  # the node cannot be reached at release time
        if func == 'release_status':
            raise NodeError('ssh: connection reset')
        return real(func, _timeout, **kw)
    ops.call = stuck
    r = ToyRunner(ops)
    import infrarsi.ext.common.remote_exec as rx
    orig = rx.GpuLease.release
    rx.GpuLease.release = lambda self, **kw: orig(self, sleep=lambda s: None, **kw)                    # no real backoff
    try:
        rec = r.run(task, {'workers': 2}, None, 'screen', 'run1', candidate_id='c1')
    finally:
        rx.GpuLease.release = orig
    assert rec['valid'] and rec['cost']['accounting_complete'] is False and signing.verify(rec)
    assert 'the release failed' in rec['cost']['incomplete_reasons'][0] and len(rec['lease_release']['kept']) == 5
    assert len(rec['lease_release']['failures']) == 5
    ops.call = real
    hooks = SessionHooks(task, base_files=BASE_FILES)                      # the session charges it as unknown accounting
    s = SearchSession(task=task, budget=task.search_budget(), runner=ToyRunner(local_ops(tmp_path / 'b')), judge=EnvJudge(),
                      run_id='r', arm='T-RND', start_config={}, layers=('L0',), hooks=hooks)
    s.ledger.charge_screen(rec['cost']['gpu_seconds'] / 3600.0, rec['cost']['wall_seconds'], rec['cost']['accounting_complete'] is True)
    assert s.ledger.accounting_complete is False


def test_containment_failure_reason(tmp_path):
    class Domain(ToyRunner):
        def domain_uids(self):
            return [64011]
    ops = local_ops(tmp_path)
    ops.uid_survivors[64011] = [5]
    rec = Domain(ops).run(toy_task(), {'workers': 2}, None, 'screen', 'run1', candidate_id='c1')
    assert not rec['valid'] and rec['cost']['accounting_complete'] is False
    assert rec['cost']['incomplete_reasons'][0].startswith('containment failed, the claims are kept')


# ---------------------------------------------------------------- item 7
def _trace(complete=True, used_complete=True, notes=None):
    cum = lambda i, ok: {'screens': i, 'llm_tokens': 100 * i, 'wall_seconds': 100.0 * i, 'gpu_hours': 0.1 * i,  # noqa: E731
                         'llm_usd': 0.0, 'accounting_complete': ok}
    screens = [{'index': 0, 'candidate_id': 'c0', 'valid': True, 'step_time': 0.9977, 'cum': cum(1, True)},
               {'index': 1, 'candidate_id': 'c1', 'valid': True, 'step_time': 0.9940, 'cum': cum(2, complete)}]
    tr = {'arm': 'X', 'screens': screens, 'used': {**cum(2, used_complete)}, 'stop_reason': 'budget:screens'}
    if notes:
        tr['extra'] = {'accounting_notes': notes}
    return tr


def test_arm_row_reports_accounting():
    task = toy_task(env='autoresearch', metric_name='val_bpb', tau=0.002, margin={'kind': 'fixed', 'value': 0.002},
                    budget={'screens': 4, 'llm_tokens': 1000, 'wall_seconds': 1000.0, 'gpu_hours': 4.0})
    ok = tj.arm_row(_trace(), task, 0.9977)
    assert ok['accounting_complete'] is True and ok['accounting_reasons'] == []
    bad = tj.arm_row(_trace(complete=False, notes=['no node stamp for row 3']), task, 0.9977)
    assert bad['accounting_complete'] is False and bad['accounting_reasons'][0] == 'screens [1]'
    assert 'no node stamp for row 3' in bad['accounting_reasons']
    used = tj.arm_row(_trace(used_complete=False), task, 0.9977)
    assert used['accounting_complete'] is False and used['accounting_reasons'] == ['used']
    assert tj.accounting({'accounting_complete': False, 'screens': []}) == (False, ['trace'])
    assert tj.accounting({'screens': [{'cum': {}}]}) == (True, [])        # no flag at all: nothing claims incompleteness


def test_compare_labels_incomplete_pairs():
    task = toy_task(env='autoresearch', metric_name='val_bpb', tau=0.002, margin={'kind': 'fixed', 'value': 0.002})
    is_row = {'locked_valid': True, 'y_locked': 0.005, 'AUC_budget': 0.002, 'invalid_accept_locked': 0, 'accounting_complete': True}
    nat = {'locked_valid': True, 'y_locked': 0.002, 'AUC_budget': 0.001, 'accounting_complete': True}
    assert tj.compare(task, is_row, nat)['label'] == 'advantage'                       # complete: labels unchanged
    r = tj.compare(task, is_row, {**nat, 'accounting_complete': False, 'accounting_reasons': ['used']})
    assert r['label'] == 'incomplete' and r['label_if_complete'] == 'advantage' and r['incomplete'] == ['native']
    assert r['accounting_reasons'] == {'native': ['used']} and r['delta_y'] == pytest.approx(0.003)
    r = tj.compare(task, {**is_row, 'locked_valid': False, 'accounting_complete': False}, nat)
    assert r['label'] == 'incomplete' and r['label_if_complete'] == 'invalid' and r['incomplete'] == ['IS']


# ---------------------------------------------------------------- item 8
def test_lost_acquire_answer_is_rolled_back(tmp_path):
    ops = local_ops(tmp_path)
    real = ops.call
    lost = gpu_claim_path('WINH0', 2)

    def lossy(func, _timeout=300, **kw):                                   # the claim is created, the answer is lost
        if func == 'acquire' and kw.get('path') == lost:
            real(func, _timeout, **kw)
            raise NodeError('ssh: connection reset after the claim was written')
        return real(func, _timeout, **kw)
    ops.call = lossy
    lease = GpuLease(ops, env='toyenv', gpus=[0, 1, 2, 3])
    with pytest.raises(NodeError):
        lease.acquire('r', 'a1')
    for p in [node_lease_path('WINH0')] + [gpu_claim_path('WINH0', g) for g in range(4)]:
        assert not ops.local(p).exists(), p                                 # the lost claim too (owner-checked)
    assert lease.acquired == [] and lease.release_failures == [] and lease.ambiguous == set()


def test_lost_answer_never_touches_a_foreign_claim(tmp_path):
    ops = local_ops(tmp_path)
    real = ops.call
    foreign = gpu_claim_path('WINH0', 1)
    ops.local(foreign).parent.mkdir(parents=True, exist_ok=True)

    def lossy(func, _timeout=300, **kw):                                   # another attempt claims GPU 1 meanwhile
        if func == 'acquire' and kw.get('path') == foreign:
            ops.local(foreign).write_text(json.dumps({'attempt_id': 'other', 'owner': 'infrarsi-ext'}), encoding='utf-8')
            raise NodeError('ssh: timeout')
        return real(func, _timeout, **kw)
    ops.call = lossy
    lease = GpuLease(ops, env='toyenv', gpus=[0, 1])
    with pytest.raises(NodeError):
        lease.acquire('r', 'a2')
    assert json.loads(ops.local(foreign).read_text(encoding='utf-8'))['attempt_id'] == 'other'    # untouched
    assert not ops.local(gpu_claim_path('WINH0', 0)).exists() and not ops.local(node_lease_path('WINH0')).exists()
    assert lease.release_failures == [] and lease.acquired == []
    ops.call = real
    with pytest.raises(OccupiedError):                                      # an answered refusal is not ambiguous
        GpuLease(ops, env='toyenv', gpus=[1]).acquire('r', 'a3')
    assert json.loads(ops.local(foreign).read_text(encoding='utf-8'))['attempt_id'] == 'other'


# ---------------------------------------------------------------- Codex W9 round 2, M-02: the rollback carries its state
def test_acquire_rollback_failures_fail_closed(tmp_path):
    from infrarsi.ext.common.remote_exec import LeaseError
    ops = local_ops(tmp_path)
    real = ops.call

    def call(func, _timeout=300, **kw):                                  # GPU 1 is refused, and the rollback cannot reach the node
        if func == 'acquire' and kw.get('path') == gpu_claim_path('WINH0', 1):
            return {'ok': False, 'existing': {'attempt_id': 'other'}}
        if func == 'release_status':
            raise NodeError('ssh: connection reset')
        return real(func, _timeout, **kw)
    ops.call = call
    lease = GpuLease(ops, env='toyenv', gpus=[0, 1])
    waits = []
    lease.sleep = waits.append                                            # backoff without real sleeping
    with pytest.raises(LeaseError) as e:
        lease.acquire('r', 'a1')
    assert waits == [1.0, 2.0, 4.0, 8.0] * 2
    assert isinstance(e.value.__cause__, OccupiedError) and sorted(e.value.kept) == sorted(lease.acquired)
    assert len(e.value.failures) == 2 and lease.rollback['complete'] is False
    ops.call = real                                                       # a complete rollback re-raises the original
    lease2 = GpuLease(ops, env='toyenv', gpus=[2, 1])
    ops.local(gpu_claim_path('WINH0', 1)).write_text(json.dumps({'attempt_id': 'other', 'owner': 'x'}), encoding='utf-8')
    for p in (node_lease_path('WINH0'), gpu_claim_path('WINH0', 0)):     # the first attempt's leftovers, cleaned up
        ops.local(p).unlink(missing_ok=True)
    with pytest.raises(OccupiedError):
        lease2.acquire('r', 'a2')
    assert lease2.rollback == {'error': lease2.rollback['error'], 'complete': True, 'kept': [], 'failures': [],
                               'taken': [node_lease_path('WINH0'), gpu_claim_path('WINH0', 2)], 'ambiguous': False}
    assert lease2.state() == {'complete': True, 'kept': [], 'failures': []}


def test_partial_acquire_is_charged(tmp_path):
    """Codex W9 round 3, R3-03: claims taken before a refused GPU are held until the rollback ends - charged as world x
    (rollback end - first claim), accounting incomplete, the rollback recorded in the receipt."""
    task = toy_task()
    ops = local_ops(tmp_path)
    r = ToyRunner(ops)
    stale = gpu_claim_path('WINH0', 2)                                     # an earlier attempt's leftover claim
    ops.local(stale).parent.mkdir(parents=True, exist_ok=True)
    ops.local(stale).write_text(json.dumps({'attempt_id': 'stale', 'owner': 'infrarsi-ext'}), encoding='utf-8')
    real = ops.call

    def slow(func, _timeout=300, **kw):                                   # every claim RPC and rollback RPC takes 2 s
        if func in ('acquire', 'release_status'):
            r.vclock.advance(2)
        return real(func, _timeout, **kw)
    ops.call = slow
    rec = r.run(task, {'workers': 2}, None, 'screen', 'run1', candidate_id='c1')
    assert not rec['valid'] and rec['runner_error']['kind'] == 'occupied'
    # node lease, GPU 0, GPU 1 taken (3 x 2 s), GPU 2 refused (2 s), 3 rollback releases (3 x 2 s) = 14 s on 4 GPUs
    assert rec['cost']['gpu_seconds'] == pytest.approx(4 * 14) and rec['cost']['accounting_complete'] is False
    assert rec['lease_rollback']['complete'] is True and len(rec['lease_rollback']['taken']) == 3
    assert not ops.local(node_lease_path('WINH0')).exists() and ops.local(stale).exists()

