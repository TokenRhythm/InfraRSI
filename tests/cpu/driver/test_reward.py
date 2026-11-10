"""reward.evaluate on synthetic validated receipts: hard constraints, the speed-only R, the checklist view, the exact
contract, breakeven, and the fail-closed rules (no InterStep fallback, authenticated parent / reference, promotion
receipts bound to the candidate)."""
import copy
import json
import math

import pytest

from driver_helpers import DUMMY_MOD, data_facts, expected_data, make_request, pin_overlay, simulated, validate_dir

from infrarsi.harness.validate import verify_receipt_digest
from infrarsi.reward import breakeven, evaluate, exact, lenient, load_reward_cfg, resolve
from infrarsi.harness import signing


@pytest.fixture
def pair(tmp_path):
    parent = simulated(tmp_path, make_request('D2'), step_time=lambda r, u: 1.0)
    cand = simulated(tmp_path, make_request('D2', config={'tensor_model_parallel_size': 2}, numerics_role='candidate'),
                     reference=parent, step_time=lambda r, u: 0.8)
    return parent, cand


def _resign(r):
    signing.sign(r)
    assert verify_receipt_digest(r)
    return r


def _names(rv):
    return {v.split(':')[0] for v in rv['violations']}


def test_faster_e0_candidate(pair):
    parent, cand = pair
    rv = evaluate(cand, parent)
    assert rv['valid'] and rv['metrics']['NumericsClass'] == 'E0', rv['violations']
    assert math.isclose(rv['components']['g_t'], math.log(1.0 / 0.8))
    assert math.isclose(rv['R'], math.log(1.25))
    assert {c['id'] for c in rv['checks']} >= {'completes', 'loss_within_tolerance'} and not rv['feedback']
    assert rv['metrics']['InputTokensPerSecond'] == 262144 / rv['metrics']['StepTime']


def test_e1_tolerance_speed_only(tmp_path):
    parent = simulated(tmp_path, make_request('D2'), step_time=lambda r, u: 1.0)
    cand = simulated(tmp_path, make_request('D2', numerics_role='candidate'), reference=parent,
                     step_time=lambda r, u: 0.9, ce_scale=1.0015)
    rv = evaluate(cand, parent)
    assert rv['valid'] and rv['metrics']['NumericsClass'] == 'E1', rv['violations']
    assert math.isclose(rv['R'], math.log(1 / 0.9), rel_tol=1e-6)        # within tolerance: no loss term in R
    drift = simulated(tmp_path, make_request('D2', numerics_role='candidate'), reference=parent, ce_scale=1.0025)
    rv = evaluate(drift, parent)
    assert not rv['valid']                                            # 0.25 % > 0.2 % window mean
    assert [c['id'] for c in rv['checks'] if c['status'] == 'fail'] == ['loss_within_tolerance']
    assert 'FAILED loss_within_tolerance' in rv['feedback']
    ex = evaluate(cand, parent, exact(load_reward_cfg()))
    assert not ex['valid'] and ex['R'] == -math.inf
    assert any('numerics_class_matches_allowed' in v for v in ex['violations'])


def test_unverified_numerics_invalid(tmp_path, pair):
    parent, _ = pair
    nocmp = simulated(tmp_path, make_request('D2', plan='B', purpose='score', numerics_role='candidate'), reference=parent)
    rv = evaluate(nocmp, parent)
    assert not rv['valid'] and any('numerics not verified' in v for v in rv['violations'])
    assert 'parent_receipt' in _names(rv)                              # a plan-A screen is no parent of a plan-B score


def test_inter_step_inconsistency(pair):
    parent, cand = pair
    bad = copy.deepcopy(cand)
    bad['step_time']['inter_step_mean_s'] = parent['step_time']['inter_step_mean_s']   # wall clock did not improve
    rv = evaluate(_resign(bad), parent)
    assert not rv['valid'] and any(v.startswith('inter_step_consistency') for v in rv['violations'])


def test_missing_inter_step_is_a_violation_not_a_skip(pair):
    """Codex R1 12: no fallback to the trainer's own InterStep summary, and no ok/skipped result."""
    parent, cand = pair
    bad = copy.deepcopy(cand)
    bad['step_time']['inter_step_mean_s'] = None
    bad['harness']['reported']['step_time']['inter_step_mean_s'] = 0.8                 # what the trainer claimed
    rv = evaluate(_resign(bad), parent)
    hit = [h for h in rv['hard'] if h['name'] == 'inter_step_consistency']
    assert not rv['valid'] and hit and hit[0]['ok'] is False and hit[0]['skipped'] is False
    assert rv['metrics']['InterStep'] is None
    no_st = copy.deepcopy(cand)
    no_st['step_time'] = None
    assert 'step_time_derived' in _names(evaluate(_resign(no_st), parent))


def test_parent_receipt_is_authenticated(tmp_path, pair):
    parent, cand = pair
    forged = copy.deepcopy(parent)
    forged['step_time']['mean_window_max_rank_s'] = 5.0                                 # inflate T_parent, unsigned
    rv = evaluate(cand, forged)
    assert not rv['valid'] and 'parent_receipt' in _names(rv)
    invalid = _resign(dict(copy.deepcopy(parent), valid=False))
    assert 'parent_receipt' in _names(evaluate(cand, invalid))
    short = copy.deepcopy(parent)
    short['per_step'] = short['per_step'][:5]
    assert 'parent_receipt' in _names(evaluate(cand, _resign(short)))
    other_task = simulated(tmp_path, make_request('D1'))
    assert 'parent_receipt' in _names(evaluate(cand, other_task))
    # a valid candidate parent (search chain) is fine without the reference role
    assert evaluate(cand, cand)['valid']


def test_unsigned_or_altered_receipt_rejected(pair):
    parent, cand = pair
    forged = copy.deepcopy(cand)
    forged['step_time']['mean_window_max_rank_s'] = 0.1
    rv = evaluate(forged, parent)
    assert not rv['valid'] and any(v.startswith('receipt_signature') for v in rv['violations'])


def test_memory_and_oom_constraints(pair):
    parent, cand = pair
    big = copy.deepcopy(cand)
    big['memory']['max_peak_reserved_gib'] = 79.0
    big['events']['alloc_ooms'] = 1
    rv = evaluate(_resign(big), parent)
    assert {'peak_reserved_gib_max', 'alloc_ooms_max'} <= _names(rv)


def test_speed_is_the_only_objective(pair):
    parent, cand = pair
    c = copy.deepcopy(cand)
    c['memory']['max_peak_reserved_gib'] = parent['memory']['max_peak_reserved_gib'] / math.e ** 5
    c = _resign(c)
    rv = evaluate(c, parent)
    assert rv['valid'] and rv['components'] == {'g_t': math.log(1.0 / 0.8)} and math.isclose(rv['R'], math.log(1.25))


def _lifecycle_receipt(tmp_path, config, overlay_digest=None):
    """A validated lifecycle attempt of `config` (plan B on D2: save_at 30 + resume 4); overlay_digest forges the
    runtime's reported effective overlay set."""
    req = make_request('D2', purpose='lifecycle', plan='B', config=config,
                       lifecycle={'save_at': 30, 'resume_steps': 4, 'export': True, 'forward_compare': True})
    expected_data(req)
    pin_overlay(req)
    out = tmp_path / req.attempt_id
    DUMMY_MOD.simulate_lifecycle(req.to_dict(), str(out), data=data_facts(req), step_time=lambda r, u: 0.8)
    if overlay_digest:
        rec = json.loads((out / 'receipt.json').read_text())
        rec['overlay_digest'] = overlay_digest
        (out / 'receipt.json').write_text(json.dumps(rec))
    return validate_dir(out, req)


def test_promotion_binds_lifecycle_and_plan_b_receipts(tmp_path, pair):
    """A item 9 / Codex finding 11: evaluate(promotion=True) itself binds the auxiliary receipts to the candidate."""
    parent, cand = pair
    rv = evaluate(cand, parent, promotion=True)
    assert {'lifecycle_for_promotion', 'second_input_plan_consistency'} <= _names(rv)
    cfg = {'tensor_model_parallel_size': 2}
    plan_b = simulated(tmp_path, make_request('D2', plan='B', purpose='score', config=cfg), step_time=lambda r, u: 0.81)
    life = _lifecycle_receipt(tmp_path, cfg)
    assert life['valid'], life['invalid_reasons']
    ok = evaluate(cand, parent, promotion=True, lifecycle_receipt=life, plan_b_receipt=plan_b, margin=0.03)
    assert ok['valid'], ok['violations']
    # receipts of another candidate (config), unsigned receipts, or the wrong purpose / plan are rejected
    other_cfg = {'tensor_model_parallel_size': 4}
    other_b = simulated(tmp_path, make_request('D2', plan='B', purpose='score', config=other_cfg), step_time=lambda r, u: 0.81)
    rv = evaluate(cand, parent, promotion=True, lifecycle_receipt=life, plan_b_receipt=other_b, margin=0.03)
    assert 'second_input_plan_consistency' in _names(rv) and any('config_digest' in v for v in rv['violations'])
    other_life = _lifecycle_receipt(tmp_path, other_cfg)
    rv = evaluate(cand, parent, promotion=True, lifecycle_receipt=other_life, plan_b_receipt=plan_b, margin=0.03)
    assert 'lifecycle_for_promotion' in _names(rv)
    unsigned = copy.deepcopy(life)
    unsigned['lifecycle']['forward_ok'] = True
    unsigned['valid'] = True
    unsigned['steps_completed'] = 99
    rv = evaluate(cand, parent, promotion=True, lifecycle_receipt=unsigned, plan_b_receipt=plan_b, margin=0.03)
    assert any('harness signature' in v for v in rv['violations'])
    rv = evaluate(cand, parent, promotion=True, lifecycle_receipt=plan_b, plan_b_receipt=life, margin=0.03)
    assert {'lifecycle_for_promotion', 'second_input_plan_consistency'} <= _names(rv)
    overlay = _lifecycle_receipt(tmp_path, cfg, overlay_digest='9' * 64)
    rv = evaluate(cand, parent, promotion=True, lifecycle_receipt=overlay, plan_b_receipt=plan_b, margin=0.03)
    assert any('overlay_digest' in v for v in rv['violations'])


def test_breakeven_reference_is_authenticated(pair):
    parent, cand = pair
    forged = copy.deepcopy(parent)
    forged['harness_signature']['mac'] = '0' * 64
    rv = evaluate(cand, parent, ref_receipt=forged)
    assert not rv['valid'] and 'reference_receipt' in _names(rv)
    assert evaluate(cand, parent, ref_receipt=parent)['valid']
    rv = evaluate(cand, parent, ref_receipt=cand)                          # a candidate is no reference
    assert 'reference_receipt' in _names(rv)


def test_resolve_memory_cap_by_gpu_type():
    cfg = load_reward_cfg()
    assert resolve(cfg, gpu_type='H20-96GB')['hard']['peak_reserved_gib_max'] == 95
    assert resolve(cfg)['hard']['peak_reserved_gib_max'] == cfg['hard']['peak_reserved_gib_max']
    ex = exact(cfg)
    assert ex['numerics_class_allowed'] == 'E0' and set(ex['tolerance'].values()) == {0.0}


def test_breakeven_math():
    b = breakeven(10.0, ref_per_update=0.02, cand_per_update=0.015, planned_training_updates=1024)
    assert math.isclose(b['n_breakeven'], 2000.0) and b['breaks_even'] and not b['worthwhile']
    assert breakeven(1.0, 0.02, 0.02)['status'] == 'does not break even'


def test_lenient_reward_accepts_what_full_rejects(pair):
    parent, cand = pair
    forged = copy.deepcopy(cand)
    forged['invariants']['physical_tokens_per_update'] = 1
    assert lenient(forged, parent)['valid'] and not evaluate(forged, parent)['valid']


def test_promotion_accepts_the_resolved_form_of_the_screened_config(tmp_path, pair):
    """The search's start candidate screened the planner's sparse config, its plan-B / lifecycle runs the fully resolved
    one: different config digests, same effective_config (schema defaults + config) and performance env -> bound.
    A different effective configuration or recorded performance env is not."""
    parent, cand = pair                                               # cand screened {'tensor_model_parallel_size': 2}
    resolved = dict(DUMMY_MOD.l0_defaults(), tensor_model_parallel_size=2)
    plan_b = simulated(tmp_path, make_request('D2', plan='B', purpose='score', config=resolved), step_time=lambda r, u: 0.81)
    life = _lifecycle_receipt(tmp_path, resolved)
    assert plan_b['config_digest'] != cand['config_digest'] and plan_b['effective_config'] == cand['effective_config']
    ok = evaluate(cand, parent, promotion=True, lifecycle_receipt=life, plan_b_receipt=plan_b, margin=0.03)
    assert ok['valid'], ok['violations']
    tuned = copy.deepcopy(plan_b)
    tuned['env'] = dict(tuned['env'], env={'NCCL_ALGO': 'Ring', 'GLOO_SOCKET_IFNAME': 'eth0'})
    rv = evaluate(cand, parent, promotion=True, lifecycle_receipt=life, plan_b_receipt=_resign(tuned), margin=0.03)
    assert 'second_input_plan_consistency' in _names(rv)
    mixed = simulated(tmp_path, make_request('D2', plan='B', purpose='score', config=dict(resolved, micro_batch_size=2)))
    rv = evaluate(cand, parent, promotion=True, lifecycle_receipt=life, plan_b_receipt=mixed, margin=0.03)
    assert 'second_input_plan_consistency' in _names(rv)


def test_amendment_001_loss_guard(tmp_path):
    """reward.yaml AMENDMENT-001 keys: the grad-norm gates are the first update and the window log bias; a later
    single-step excursion is reported, not penalised; p_q stays CE-based; missing grad-norm evidence fails closed."""
    parent = simulated(tmp_path, make_request('D2'), step_time=lambda r, u: 1.0)
    spike = simulated(tmp_path, make_request('D2', numerics_role='candidate'), reference=parent, step_time=lambda r, u: 0.9,
                      gn_scale=lambda u: 1.56 if u == 25 else 1.0)
    rv = evaluate(spike, parent)
    assert rv['valid'], rv['violations']
    assert rv['metrics']['NumericsClass'] == 'E1'                                     # CE identical
    assert rv['metrics']['LossDeviation']['grad_norm_rel_max'] == pytest.approx(0.56)
    biased = simulated(tmp_path, make_request('D2', numerics_role='candidate'), reference=parent, step_time=lambda r, u: 0.9,
                       gn_scale=lambda u: 1.0 if u == 1 else 0.7)
    rv = evaluate(biased, parent)
    assert not rv['valid'] and 'loss_guard' in _names(rv)
    old = copy.deepcopy(spike)                                     # a receipt without the amended grad-norm block
    old['harness']['numerics'].pop('grad_norm')
    rv = evaluate(_resign(old), parent)
    assert not rv['valid'] and any('grad_norm.first_update_rel missing' in v for v in rv['violations'])
    ex = evaluate(spike, parent, exact(load_reward_cfg()))          # all gates 0: only a bitwise E0 candidate passes
    assert not ex['valid'] and any('E0 (bitwise) required' in v for v in ex['violations'])
