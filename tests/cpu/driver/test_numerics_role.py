"""A run is a numerics reference only when launched as one (RunRequest.numerics_role); candidates need a reference."""
import copy
import importlib

import pytest

from driver_helpers import HF, make_request, simulated

from infrarsi.contracts import TaskSpec
from infrarsi.dispatch import QueueStore
from infrarsi.harness.remote import LocalRemote
from infrarsi.harness.runner import recheck_against
from infrarsi.reward import evaluate

H = importlib.import_module('infrarsi.dispatch.handlers')   # the package re-exports a function named `handlers`


def _viol(rv):
    return {v.split(':')[0] for v in rv['violations']}


def test_candidate_without_reference_fails_harness_and_reward(tmp_path):
    rec = simulated(tmp_path, make_request('D2', numerics_role='candidate'))
    num = rec['harness']['numerics']
    assert num == dict(num, compared=False, is_reference=False, violations=['no numerics reference'])
    assert not rec['valid'] and any('numerics.verified' in x for x in rec['invalid_reasons'])
    rv = evaluate(rec)
    assert not rv['valid'] and rv['metrics']['NumericsClass'] is None
    assert {'numerics_compared', 'numerics_class_matches_allowed', 'loss_guard'} <= _viol(rv)
    assert any('numerics not verified' in v for v in rv['violations'])


def test_fr_reference_passes_as_ref(tmp_path):
    fr = simulated(tmp_path, make_request('D2', numerics_role='reference'))
    assert fr['valid'] and fr['numerics_role'] == 'reference' and fr['harness']['numerics']['is_reference'] is True
    rv = evaluate(fr)
    assert rv['valid'] and rv['metrics']['NumericsClass'] == 'REF', rv['violations']
    # a reference-role run that does carry a reference is compared normally
    again = simulated(tmp_path, make_request('D2', numerics_role='reference'), reference=fr)
    assert again['harness']['numerics']['compared'] is True and again['harness']['numerics']['class'] == 'E0'


def test_flipped_reference_flag_on_a_candidate_is_rejected(tmp_path):
    from infrarsi.harness import signing
    rec = simulated(tmp_path, make_request('D2', numerics_role='candidate'))
    forged = copy.deepcopy(rec)
    forged['harness']['numerics'].update(is_reference=True, violations=[])    # the whole receipt is HMAC-signed ...
    forged['valid'], forged['invalid_reasons'] = True, []
    rv = evaluate(forged)
    assert not rv['valid'] and 'receipt_signature' in _viol(rv)
    signing.sign(forged)                                                       # ... and even a key holder cannot
    rv = evaluate(forged)                                                      # turn a candidate role into a reference
    assert not rv['valid'] and 'numerics_compared' in _viol(rv) and rv['metrics']['NumericsClass'] is None
    no_role = copy.deepcopy(forged)                                            # and a receipt without a role is no reference
    no_role.pop('numerics_role')
    assert not evaluate(signing.sign(no_role))['valid']


def test_candidate_scored_first_recovers_after_recheck(tmp_path):
    """Odd task parity: the candidate's plan-B run precedes FR's plan-B run, then recheck_against compares them."""
    cand = simulated(tmp_path, make_request('D2', plan='B', purpose='score', numerics_role='candidate'))
    assert not cand['valid']
    fr_b = simulated(tmp_path, make_request('D2', plan='B', purpose='score', numerics_role='reference'))
    ok = recheck_against(cand, fr_b)
    assert ok['valid'], ok['invalid_reasons']
    assert ok['harness']['numerics']['compared'] and not any('numerics.verified' in x for x in ok['invalid_reasons'])
    assert evaluate(ok)['valid']


# ---------------------------------------------------------------- dispatch handlers set the role

class FakeRunner:
    def __init__(self):
        self.requests = []

    def __call__(self, req, **kw):
        self.requests.append(req)
        rec = {'attempt_id': req.attempt_id, 'valid': True, 'harness': {'failure_kind': 'ok'},
               'step_time': {'mean_window_max_rank_s': 1.0, 'rsd_window': 0.01}}
        if req.purpose == 'probe':
            rec['per_step'] = [{'update': u, 'max_rank_whole_s': 1.0} for u in range(1, 6)]
        return rec


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    from infrarsi.discover import workload_facts
    monkeypatch.setattr(H, '_facts', lambda task, ctx: workload_facts(task, cache_dir=HF))
    return H.Context(store=QueueStore(tmp_path / 'queue'), runner=FakeRunner(), remote=LocalRemote(tmp_path / 'fs'))


def _main(ctx):
    return [r for r in ctx.runner.requests if r.purpose != 'probe']


def _stored_fr(ctx, tmp_path, task='D2', config=None):
    """Signed FR plan-A screen and plan-B score receipts at their remote paths (as run_fr records them)."""
    cfg = config or {'tensor_model_parallel_size': 1}
    a = simulated(tmp_path, make_request(task, config=cfg))
    b = simulated(tmp_path, make_request(task, plan='B', purpose='score', config=cfg))
    paths = {}
    for k, rec in (('fr_screen', a), ('fr_score', b)):
        paths[k] = f'/data/infrarsi/receipts/{rec["attempt_id"]}.json'
        ctx.remote.write_json('WINH0', paths[k], rec)
    return paths, a, b


def test_fr_runs_are_references_and_probes_unaffected(ctx):
    res = H.run_fr({'run_id': 'E0-CAL-D2-FR', 'task': 'D2', 'wall_hours_cap': 4}, ['WINH2'], ctx)
    assert res['status'] == 'feasible' and H.terminal_state('FR', res) == ('done', None)
    assert [(r.purpose, r.input_plan, r.numerics_role, r.reference_receipt) for r in _main(ctx)] == [
        ('screen', 'A', 'reference', None), ('score', 'B', 'reference', None)]
    assert all(r.numerics_role == 'candidate' for r in ctx.runner.requests if r.purpose == 'probe')
    ref = ctx.store.references()['D2']
    assert ref['numerics_reference'] == 'FR' and H.reference_paths(ref)[2] == 'FR'


def test_ed_compares_with_fr_when_fr_is_feasible(ctx, tmp_path):
    paths, _, _ = _stored_fr(ctx, tmp_path)
    ctx.store.set_reference('D2', fr_status='feasible', **paths)
    res = H.run_ed({'run_id': 'E1-D2-ED', 'task': 'D2', 'wall_hours_cap': 4}, ['WINH2'], ctx)
    assert res['numerics_role'] == 'candidate'
    assert [(r.input_plan, r.numerics_role, r.reference_receipt) for r in _main(ctx)] == [
        ('A', 'candidate', paths['fr_screen']), ('B', 'candidate', paths['fr_score'])]
    ref = ctx.store.references()['D2']
    assert ref['ed_numerics_role'] == 'candidate' and 'numerics_reference' not in ref       # ED did not take over
    assert H.reference_paths(ref) == (paths['fr_screen'], paths['fr_score'], 'FR')


def test_ed_refuses_unusable_fr_receipts_before_probing(ctx, tmp_path):
    paths, a, _ = _stored_fr(ctx, tmp_path)
    bad = copy.deepcopy(a)
    bad['per_step'] = bad['per_step'][:1]                                     # unsigned, incomplete reference
    ctx.remote.write_json('WINH0', paths['fr_screen'], bad)
    ctx.store.set_reference('D2', fr_status='feasible', **paths)
    with pytest.raises(RuntimeError, match='FR reference receipts unusable'):
        H.run_ed({'run_id': 'E1-D2-ED', 'task': 'D2', 'wall_hours_cap': 4}, ['WINH2'], ctx)
    assert ctx.runner.requests == []                                          # not even a probe


def test_ed_becomes_the_reference_when_fr_is_infeasible(ctx, tmp_path):
    ctx.store.set_reference('H6', fr_status='infeasible')
    res = H.run_ed({'run_id': 'E1-H6-ED', 'task': 'H6', 'wall_hours_cap': 3}, ['WINH2'], ctx)
    assert res['status'] == 'feasible' and res['numerics_role'] == 'reference'
    assert [(r.input_plan, r.numerics_role, r.reference_receipt) for r in _main(ctx)] == [
        ('A', 'reference', None), ('B', 'reference', None)]
    ref = ctx.store.references()['H6']
    assert ref['numerics_reference'] == 'ED'
    screen, score, kind = H.reference_paths(ref)
    assert kind == 'ED' and screen == ref['ed_screen'] and score == ref['ed_score']
    # the agent arms then get ED's plan-A screen as their numerics reference
    ctx.remote.write_json('WINH0', screen, {'attempt_id': 'ed-a'})
    seen = {}
    H.search_handler(lambda **kw: seen.update(kw) or {})({'run_id': 'E1-H6-A0', 'task': 'H6', 'arm': 'A0'}, ['WINH2'], ctx)
    assert seen['reference_receipt'] == {'attempt_id': 'ed-a'}


def test_ed_refuses_without_an_fr_reference(ctx):
    ctx.store.set_reference('D2', fr_status='invalid', fr_screen='/r/a.json', fr_score='/r/b.json')
    with pytest.raises(RuntimeError, match='no FR numerics reference'):
        H.run_ed({'run_id': 'E1-D2-ED', 'task': 'D2', 'wall_hours_cap': 4}, ['WINH2'], ctx)
    assert ctx.runner.requests == []                                     # nothing launched, not even probes


def test_fr_lifecycle_needs_a_verified_feasible_fr(ctx, tmp_path):
    """Codex R1 10: the lifecycle starts only from a verified, feasible FR receipt and exactly its configuration."""
    cfg = {'tensor_model_parallel_size': 2}
    paths, _, b = _stored_fr(ctx, tmp_path, config=cfg)
    run = {'run_id': 'E0-LIFECYCLE-DENSE8', 'task': 'D2', 'wall_hours_cap': 2}
    for refs, match in (({'fr_status': 'invalid', 'fr_config': cfg, **paths}, 'not verified feasible'),
                        ({'fr_status': 'feasible', 'fr_config': cfg}, 'not verified feasible'),
                        ({'fr_status': 'feasible', 'fr_config': {'tensor_model_parallel_size': 4}, **paths}, 'config_digest')):
        ctx.store.set_reference('D2', **refs)
        with pytest.raises(RuntimeError, match=match):
            H.run_lifecycle(run, ['WINH2'], ctx)
        ctx.store.refs_path.unlink()
    bad = copy.deepcopy(b)
    bad['valid'] = False                                                  # invalid FR plan-B receipt at the path
    ctx.remote.write_json('WINH0', paths['fr_score'], bad)
    ctx.store.set_reference('D2', fr_status='feasible', fr_config=cfg, **paths)
    with pytest.raises(RuntimeError, match='unusable'):
        H.run_lifecycle(run, ['WINH2'], ctx)
    assert ctx.runner.requests == []
    ctx.remote.write_json('WINH0', paths['fr_score'], b)
    res = H.run_lifecycle(run, ['WINH2'], ctx)
    (req,) = ctx.runner.requests
    assert req.purpose == 'lifecycle' and req.reference_receipt is None and req.numerics_role == 'reference'
    assert req.config == cfg and res['config_digest'] == b['config_digest'] and res['status'] == 'pass'


def test_invalid_role_rejected_by_the_contract():
    with pytest.raises(ValueError):
        make_request('D2', numerics_role='baseline')
    assert TaskSpec.load('D2').id == 'D2'
