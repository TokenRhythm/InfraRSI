"""Whole-codebase review fixes in the verifier (2026-10-24): equal optimizer-step counters do not hide another learning
rate (C2 #1); a required gradient check without data evidence is unchecked, not information (C2 #2); a run stopped as
planned after k updates of the full schedule completes (C1 #5 / C2 #4, Fable 5.1's horizon design)."""
from infrarsi.harness import signing
from infrarsi.verify import DEFAULT_REQUIRED, Verifier
from test_verifier import receipt, status


def _with(r, **cols):
    for row in r['evidence']['per_step']:
        row.update({k: f(row['step']) for k, f in cols.items()})
    return signing.sign(r)


def _detail(v, cid):
    return {c['id']: c['detail'] for c in v.checks}[cid]


def test_learning_rate_is_compared_next_to_equal_step_counters():
    ref = _with(receipt('ref', sign=False), optim_step=lambda s: s)
    other = _with(receipt('c', t=0.9, lr=lambda s: 1e-2 * s, sign=False), optim_step=lambda s: s)
    v = Verifier()(other, ref, ref)
    assert status(v, 'optimizer_steps') == 'fail' and not v.valid
    same = _with(receipt('d', t=0.9, sign=False), optim_step=lambda s: s)
    v = Verifier()(same, ref, ref)
    assert status(v, 'optimizer_steps') == 'pass' and 'learning rate' in _detail(v, 'optimizer_steps')
    lr_only = Verifier()(receipt('e', t=0.9), receipt('ref2'), receipt('ref2'))
    assert 'proxy' in _detail(lr_only, 'optimizer_steps')


def test_a_gradient_check_without_data_evidence_is_unchecked():
    ref = receipt('ref', data=False)
    cand = receipt('c', t=0.9, data=False, gn=lambda s: 2 * (1.0 + 0.01 * s))
    v = Verifier()(cand, ref, ref)
    assert status(v, 'grad_norm_within_tolerance') == 'unchecked' and v.valid      # a gap in the report, not a veto
    assert 'grad_norm_within_tolerance' in v.coverage['unchecked']
    v = Verifier(required=(*DEFAULT_REQUIRED, 'grad_norm_within_tolerance'))(cand, ref, ref)
    assert not v.valid                                                              # the user asked for it


def test_a_planned_stop_completes_and_others_do_not():
    ref = receipt('ref', steps=30)
    p = receipt('p', steps=2, sign=False)
    p['evidence'].update(steps=30, stop_after=2, planned_stop=True, exit_code=-15)
    v = Verifier()(signing.sign(p), ref, ref)
    assert v.valid, v.reasons
    assert {status(v, c) for c in ('completes', 'loss_within_tolerance')} == {'pass'}
    assert status(v, 'optimizer_steps') == 'unchecked'          # the learning rate agrees, but no counter is exposed
    for change in ({'steps_completed': 1}, {'timed_out': True}, {'stop_after': None}, {'planned_stop': False}):
        q = receipt('q', steps=2, sign=False)
        q['evidence'].update({'steps': 30, 'stop_after': 2, 'planned_stop': True, 'exit_code': -15, **change})
        assert status(Verifier()(signing.sign(q), ref, ref), 'completes') == 'fail', change
