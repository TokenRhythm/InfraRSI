"""U-1-21: the loss and gradient-norm checks keep the deviation from the reference they measured as structured evidence
with its sign (`signed_deviation` = candidate / reference - 1; `deviation` stays its absolute value, the number compared
with the tolerance, and the check's result is unchanged), and a candidate row of the trace carries it from the
candidate's full-length screen only (`Candidate.deviations`)."""
import pytest

from infrarsi.agent.search import Candidate
from infrarsi.verify import Verifier

from test_verifier import receipt

LOSS = lambda s: 2.0 - 0.01 * s                     # noqa: E731 - the curves of test_verifier.receipt
GN = lambda s: 1.0 + 0.01 * s                       # noqa: E731
NONE = {'loss_deviation': None, 'grad_norm_deviation': None}


def check(v, cid):
    return next(c for c in v.checks if c['id'] == cid)


def cand(cid, rec, verdict, status='screened'):
    return Candidate(id=cid, parent='c0', kind='L0', origin='test', config={}, overlay={}, overlay_digest=None, key=cid,
                     status=status, verdict=verdict.to_dict(), receipt=rec)


def test_the_window_mean_deviation_keeps_its_sign():
    ref = receipt('ref', data=False)                 # no data digest: the window means are compared
    for f in (1.02, 0.98):
        r = check(Verifier()(receipt('c', t=0.9, data=False, loss=lambda s: LOSS(s) * f), ref, ref), 'loss_within_tolerance')
        assert r['status'] == 'pass' and 'window-mean deviation 2.0000%' in r['detail']     # the result as before
        assert r['evidence']['signed_deviation'] == pytest.approx(f - 1)
        assert r['evidence']['deviation'] == abs(r['evidence']['signed_deviation'])


def test_with_the_data_order_proven_it_is_the_worst_update():
    ref = receipt('ref')                             # a data digest per update: compared per update
    low = receipt('c', t=0.9, loss=lambda s: LOSS(s) * (0.96 if s == 7 else 1.01))
    r = check(Verifier()(low, ref, ref), 'loss_within_tolerance')
    assert r['status'] == 'fail' and r['evidence']['paired'] and '(update 7,' in r['detail']
    assert r['evidence']['signed_deviation'] == pytest.approx(-0.04) and r['evidence']['deviation'] == pytest.approx(0.04)


def test_the_first_update_gradient_norm_has_a_number_only_where_it_is_checked():
    ref = receipt('ref')
    r = check(Verifier()(receipt('c', t=0.9, gn=lambda s: GN(s) * 0.99), ref, ref), 'grad_norm_within_tolerance')
    assert r['status'] == 'pass' and r['evidence']['signed_deviation'] == pytest.approx(-0.01)
    ref = receipt('ref', data=False)                 # the first batch is not proven the same: reported, not compared
    r = check(Verifier()(receipt('c', t=0.9, data=False, gn=lambda s: GN(s) * 0.99), ref, ref), 'grad_norm_within_tolerance')
    assert r['status'] == 'unchecked' and '(1.0000%)' in r['detail'] and 'evidence' not in r


def test_no_number_without_a_comparison():
    ref = receipt('ref')
    assert 'evidence' not in check(Verifier()(ref, None, ref), 'loss_within_tolerance')      # the reference itself
    r = check(Verifier()(receipt('c', loss=lambda s: None), ref, ref), 'loss_within_tolerance')
    assert r['status'] == 'unchecked' and 'evidence' not in r
    r = check(Verifier(disabled=['loss_within_tolerance'])(receipt('c'), ref, ref), 'loss_within_tolerance')
    assert r['status'] == 'off' and 'evidence' not in r


def test_a_candidate_row_carries_the_deviation_of_its_full_length_screen_only():
    v, ref = Verifier(), receipt('ref')
    full = receipt('c1', t=0.9, loss=lambda s: LOSS(s) * 1.02, gn=lambda s: GN(s) * 0.99)
    c1 = cand('c1', full, v(full, ref, ref))
    assert c1.deviations() == {'loss_deviation': pytest.approx(0.02), 'grad_norm_deviation': pytest.approx(-0.01)}
    assert not set(NONE) & set(c1.summary({}))       # the rows the model is shown leave them out
    cut = receipt('c2', t=0.9, completed=6, loss=lambda s: LOSS(s) * 1.02)    # 6 of its 10 updates ran
    assert check(v(cut, ref, ref), 'loss_within_tolerance')['evidence']['signed_deviation'] == pytest.approx(0.02)
    assert cand('c2', cut, v(cut, ref, ref), 'aborted').deviations() == NONE        # a mean of fewer updates
    assert cand('c3', None, v(cut, ref, ref), 'infeasible').deviations() == NONE    # a failed probe's verdict
    assert cand('c0', ref, v(ref, None, ref)).deviations() == NONE                  # the reference itself
