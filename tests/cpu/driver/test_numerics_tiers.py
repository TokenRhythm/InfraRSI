"""AMENDMENT-002 numerics tiers in the harness and the reward engine (builder X4).

The tier follows the task's model facts; each tier's frozen table comes from the reward config; a spec off its tier
fails every run closed; screens gate the first-update grad norm at 0.03 on MoE tiers and 0.01 on dense ones;
lifecycles follow the tier's deterministic-kernel rule (deterministic + overrides, or deterministic_unavailable for
Gated DeltaNet with first resumed update CE 1e-3 / grad norm 0.03); a deterministic lifecycle compares backends_flags
without deterministic_algorithms / cudnn.deterministic; model-defined fp32 tensors must be exactly fp32 (Codex R3).
Tests run on a copy of the design in a known state (amendment_helpers.design_world), whatever reports/ holds.
"""
import copy
import importlib
import math

import pytest

import amendment_helpers as AH
from driver_helpers import DUMMY_MOD, make_request, rewrite_receipt, attempt_dir, validate_dir

from infrarsi.contracts import TaskSpec, load_all_tasks
from infrarsi.harness import recheck_against, signing, validate as V
from infrarsi.reward import evaluate, exact

R = importlib.import_module('infrarsi.reward.evaluate')     # the module (the package re-exports the function)


@pytest.fixture
def amended(tmp_path, monkeypatch):
    w = AH.design_world(tmp_path, monkeypatch, 'amended')
    yield w
    w.close()


@pytest.fixture
def legacy(tmp_path, monkeypatch):
    w = AH.design_world(tmp_path, monkeypatch, 'legacy')
    yield w
    w.close()


def _resign(r):
    signing.sign(r)
    return r


# ---------------------------------------------------------------- tiers and frozen tables

def test_tier_of_every_task_follows_its_model_facts():
    tiers = {tid: V.numerics_tier(t.raw) for tid, t in load_all_tasks().items()}
    for tier, ids in AH.TIER_TASKS.items():
        assert {t for t, k in tiers.items() if k == tier} == ids, tier
    raw = copy.deepcopy(TaskSpec.load('D4').raw)
    raw['model']['bridge_type'] = 'qwen3_next'                   # the facts decide, never a spec numbers field
    assert V.numerics_tier(raw) == 'moe_nondeterministic'
    raw['model'].update(bridge_type='qwen3', family='qwen3-dense')
    assert V.numerics_tier(raw) == 'dense'


def test_live_design_is_consistent_with_itself():
    """Before or after the orchestrator applies PROPOSED_CHANGES_X4.md, every live spec equals its tier's table."""
    for tid, t in load_all_tasks().items():
        assert V.numerics_tier_check(t.raw)['problems'] == [], tid


def test_proposed_reward_yaml_gives_exactly_the_ruled_tables():
    t = V.tier_tables(AH.amended_cfg())
    assert t['dense']['tolerance'] == AH.A001_TOLERANCE
    for tier in ('moe_deterministic', 'moe_nondeterministic'):
        assert t[tier]['tolerance'] == dict(AH.A001_TOLERANCE, grad_norm_rel_max_first_update=0.03)
    assert t['dense']['lifecycle_tolerance'] == t['moe_deterministic']['lifecycle_tolerance'] == \
        dict(AH.A001_LIFECYCLE, deterministic='required')
    assert t['moe_nondeterministic']['lifecycle_tolerance'] == dict(
        AH.A001_LIFECYCLE, deterministic='unavailable', first_resumed_update={'main_ce_rel_max': 1.0e-3, 'grad_norm_rel_max': 0.03})
    old = V.tier_tables(AH.legacy_cfg())                          # the AMENDMENT-001 design: one table for all tiers
    assert all(old[k] == {'tolerance': AH.A001_TOLERANCE, 'lifecycle_tolerance': AH.A001_LIFECYCLE} for k in V.NUMERICS_TIERS)


def test_proposed_spec_blocks_make_all_18_specs_consistent_and_partial_edits_fail():
    amended, legacy = AH.amended_cfg(), AH.legacy_cfg()
    want = {'dense': 1, 'moe_deterministic': 2, 'moe_nondeterministic': 3}  # deterministic; + gn 0.03; + first resumed
    for tid in load_all_tasks():
        new, old = AH.spec_raw(tid, 'amended'), AH.spec_raw(tid, 'legacy')
        tier = V.numerics_tier(new)
        assert V.numerics_tier_check(new, amended)['problems'] == [], tid
        assert V.numerics_tier_check(old, legacy)['problems'] == [], tid
        assert len(V.numerics_tier_check(old, amended)['problems']) == want[tier], tid      # specs not edited yet
        assert V.numerics_tier_check(new, legacy)['problems'], tid                            # reward.yaml not edited yet


def test_a_spec_cannot_loosen_tighten_or_omit_its_gates():
    cfg, raw = AH.amended_cfg(), AH.spec_raw('D2', 'amended')
    assert V.numerics_tier_check(raw, cfg)['problems'] == []
    loose = copy.deepcopy(raw)
    loose['numerics']['tolerance']['grad_norm_rel_max_first_update'] = 0.03          # a dense task claiming the MoE gate
    tight = copy.deepcopy(raw)
    tight['numerics']['tolerance']['main_ce_rel_max_per_step'] = 0.004
    gone = copy.deepcopy(raw)
    del gone['numerics']['lifecycle_tolerance']['deterministic']
    extra = copy.deepcopy(raw)
    extra['numerics']['tolerance']['grad_norm_rel_max_per_step'] = 0.02             # the removed AMENDMENT-001 key
    nondet = copy.deepcopy(raw)
    nondet['numerics']['lifecycle_tolerance']['deterministic'] = 'unavailable'      # a dense task opting out
    for case in (loose, tight, gone, extra, nondet):
        assert V.numerics_tier_check(case, cfg)['problems']
    same = copy.deepcopy(raw)
    same['numerics']['tolerance']['grad_norm_log_bias_window_max'] = 0.1            # 0.10 == 0.1: YAML spelling only
    assert V.numerics_tier_check(same, cfg)['problems'] == []
    broken = AH.amended_cfg()
    broken['tolerance']['grad_norm_rel_max_first_update'] = {'dense': 0.01}         # no MoE entry: no table at all
    chk = V.numerics_tier_check(AH.spec_raw('D4', 'amended'), broken)
    assert chk['table'] is None and chk['problems']
    odd = AH.amended_cfg()
    odd['lifecycle_tolerance']['deterministic'] = 'sometimes'
    assert V.numerics_tier_check(raw, odd)['table'] is None
    assert V.numerics_tier_check(raw, {'tolerance': None})['problems']


def test_lifecycle_tolerance_reads_the_tier_table(amended):
    assert V.lifecycle_tolerance(TaskSpec.load('F1').raw)['first_resumed_update'] == {'main_ce_rel_max': 1e-3, 'grad_norm_rel_max': 0.03}
    assert V.lifecycle_tolerance(TaskSpec.load('D4').raw)['first_resumed_update'] == {'main_ce_rel_max': 1e-4, 'grad_norm_rel_max': 5e-3}
    assert V.lifecycle_tolerance(TaskSpec.load('S2').raw)['deterministic'] == 'required'


# ---------------------------------------------------------------- screens

def test_screen_of_a_spec_off_its_tier_fails_closed(amended, monkeypatch):
    ok = AH.run(amended, make_request('D2'))
    assert ok['valid'] and ok['harness']['numerics']['tier'] == 'dense', ok['invalid_reasons']
    monkeypatch.setattr(V, 'reward_cfg', AH.legacy_cfg)            # reward.yaml and specs out of step
    rec = AH.run(amended, make_request('D2'))
    assert not rec['valid'] and AH.reasons(rec, 'numerics.tier') and rec['harness']['numerics']['tier_problems']
    probe = AH.run(amended, make_request('D2', purpose='probe'))
    assert not any(c['name'] == 'numerics.tier' for c in probe['harness']['checks'])    # probes compare no numerics


@pytest.mark.parametrize('tid,scale,ok', [('D4', 1.025, True), ('D4', 1.035, False), ('D6', 1.025, True),
                                          ('D2', 1.015, False), ('D2', 1.008, True)])
def test_first_update_grad_norm_gate_by_tier(amended, tid, scale, ok):
    """MoE tiers: update-1 grad norm within 3 % (D4 same-config spread 1.4 %); dense tasks keep 1 %."""
    ref = AH.run(amended, make_request(tid))
    cand = AH.run(amended, make_request(tid, numerics_role='candidate', config={'tensor_model_parallel_size': 2}),
                  reference=ref, gn_scale=lambda u: scale if u == 1 else 1.0)
    num = cand['harness']['numerics']
    assert num['tier'] == V.numerics_tier(TaskSpec.load(tid).raw)
    assert num['tolerance']['grad_norm_rel_max_first_update'] == (0.01 if num['tier'] == 'dense' else 0.03)   # recorded
    assert cand['valid'] is ok, cand['invalid_reasons']
    assert num['class'] == ('E1' if ok else 'FAIL')
    assert ok or any('rel dev at update 1' in v for v in num['violations'])


def test_moe_relaxation_keeps_the_window_bias_gate(amended):
    """Ruling point 3: a systematic gradient change (s03-like local norm, ln ratio -1.04 under DP8) is still caught by
    the window log bias on MoE tiers, and the first update is far beyond 0.03."""
    ref = AH.run(amended, make_request('D4'))
    s03 = AH.run(amended, make_request('D4', numerics_role='candidate', config={'tensor_model_parallel_size': 2}),
                 reference=ref, gn_scale=lambda u: 8 ** -0.5)
    v = s03['harness']['numerics']['violations']
    assert not s03['valid'] and any('window log bias' in x for x in v) and any('rel dev at update 1' in x for x in v)
    later = AH.run(amended, make_request('D4', numerics_role='candidate', config={'tensor_model_parallel_size': 2}),
                   reference=ref, gn_scale=lambda u: 1.0 if u == 1 else math.exp(-0.15))
    assert not later['valid'] and any('window log bias' in x for x in later['harness']['numerics']['violations'])


def test_recheck_against_uses_the_tier_and_rechecks_the_spec(amended, monkeypatch):
    fr_b = AH.run(amended, make_request('D4', plan='B', purpose='score'))
    cand = AH.run(amended, make_request('D4', plan='B', purpose='score', numerics_role='candidate',
                                        config={'tensor_model_parallel_size': 2}), gn_scale=lambda u: 1.02 if u == 1 else 1.0)
    assert not cand['valid'] and AH.reasons(cand, 'numerics.verified')        # candidate-first score: no reference yet
    again = recheck_against(cand, fr_b)
    assert again['valid'], again['invalid_reasons']
    assert again['harness']['numerics']['tier'] == 'moe_deterministic' and again['harness']['numerics']['class'] == 'E1'
    assert any(c['name'] == 'numerics.tier' and c['ok'] for c in again['harness']['checks'])
    monkeypatch.setattr(V, 'reward_cfg', AH.legacy_cfg)
    stale = recheck_against(cand, fr_b)                                         # 0.02 > 0.01 and the spec is off-tier
    assert not stale['valid'] and AH.reasons(stale, 'numerics.tier') and AH.reasons(stale, 'numerics.class')


def test_reward_engine_resolves_per_tier_gates(amended):
    cfg = AH.amended_cfg()
    parent = AH.run(amended, make_request('D4'), step_time=lambda r, u: 1.0)
    cand = AH.run(amended, make_request('D4', numerics_role='candidate', config={'tensor_model_parallel_size': 2}),
                  reference=parent, step_time=lambda r, u: 0.9, gn_scale=lambda u: 1.02 if u == 1 else 1.0)
    assert cand['valid'], cand['invalid_reasons']
    assert evaluate(cand, parent, cfg)['valid']                                # MoE: 0.02 <= 0.03
    assert not evaluate(cand, parent, AH.legacy_cfg())['valid']                # the AMENDMENT-001 scalar 0.01
    old = copy.deepcopy(cand)
    old['harness']['numerics'].pop('tier')                                     # validated before the tiers existed:
    assert R.receipt_tier(old) == 'moe_deterministic' and evaluate(_resign(old), parent, cfg)['valid']   # the spec says
    forged = copy.deepcopy(cand)
    forged['harness']['numerics']['tier'] = 'dense'                            # signed tier against the spec's facts
    assert R.receipt_tier(forged) is None
    rv = evaluate(_resign(forged), parent, cfg)
    assert not rv['valid'] and 'numerics_tier' in {v.split(':')[0] for v in rv['violations']}
    assert any('grad_norm.first_update_rel 0.02 > 0.0' in v for v in rv['violations'])                    # gates 0 too
    stale = copy.deepcopy(cand)                                                # spec changed since: the signed tier
    stale['task_digest'] = '0' * 64                                            # alone is no authority
    assert R.receipt_tier(stale) is None
    assert 'numerics_tier' in {v.split(':')[0] for v in evaluate(_resign(stale), None, cfg)['violations']}
    assert 'numerics_tier' not in {v.split(':')[0] for v in evaluate(_resign(stale), None, AH.legacy_cfg())['violations']}
    assert R.tier_tolerance(cfg['tolerance'], None)['grad_norm_rel_max_first_update'] is None
    ex = evaluate(cand, parent, exact(cfg))                                 # per-tier gates zeroed like the others
    assert not ex['valid']
    e0 = AH.run(amended, make_request('D4', numerics_role='candidate', config={'tensor_model_parallel_size': 2}),
                reference=parent, step_time=lambda r, u: 0.9)
    assert e0['harness']['numerics']['class'] == 'E0' and evaluate(e0, None, exact(cfg))['valid']
    stale_e0 = _resign(dict(copy.deepcopy(e0), task_digest='0' * 64))        # E0 checks no gate: the tier check still binds
    rv = evaluate(stale_e0, None, exact(cfg))
    assert not rv['valid'] and 'numerics_tier' in {v.split(':')[0] for v in rv['violations']}


# ---------------------------------------------------------------- lifecycles

LIFE = {'save_at': 6, 'resume_steps': 4, 'export': True, 'forward_compare': True, 'deterministic': True}


def _life(world, tid='S1', flags=None, resume_fn=None, **kw):
    """A lifecycle attempt whose trainer block starts from what the trainer reports (dummy deterministic_fields);
    flags replace entries (None stands for a missing one); resume_fn edits the first resumed update (step 7)."""
    req = make_request(tid, purpose='lifecycle', plan='B', steps=6, lifecycle=dict(LIFE))
    edit = None
    if resume_fn:
        edit = lambda rank, row: resume_fn(row) if rank == 0 and row['step'] == 7 else row        # noqa: E731
    return AH.run_life(world, req, flags={**DUMMY_MOD.LIFECYCLE_FLAGS, **(flags or {})}, resume_fn=edit, **kw)


def test_deterministic_lifecycle_required_outside_gated_deltanet(amended):
    ok = _life(amended)
    assert ok['valid'], ok['invalid_reasons']
    rep = ok['harness']['lifecycle']
    assert rep['deterministic']['rule'] == 'required' and rep['tier'] == 'dense'
    assert rep['resumed_bitwise_updates'] == rep['resumed_updates'] == 4         # report only
    for flags in ({'deterministic': False}, {'deterministic_overrides': None}, {'deterministic_overrides': {}}):
        rec = _life(amended, flags=flags)
        assert not rec['valid'] and AH.reasons(rec, 'lifecycle.deterministic'), flags


def test_gated_deltanet_lifecycle_states_unavailable_and_uses_its_gates(amended):
    ok = _life(amended, 'D6')
    assert ok['valid'], ok['invalid_reasons']
    rep = ok['harness']['lifecycle']
    assert rep['deterministic']['rule'] == 'unavailable' and rep['first_resumed_tolerance'] == {'main_ce_rel_max': 1e-3, 'grad_norm_rel_max': 0.03}
    for flags in ({'deterministic_unavailable': None}, {'deterministic': True, 'deterministic_unavailable': None}):
        rec = _life(amended, 'D6', flags=flags)
        assert not rec['valid'] and AH.reasons(rec, 'lifecycle.deterministic'), flags
    assert not _life(amended, 'D6', flags={'deterministic': True})['valid']    # GDN cannot be deterministic
    ce = lambda f: (lambda r: dict(r, main_ce=r['main_ce'] * f))               # noqa: E731
    gn = lambda f: (lambda r: dict(r, grad_norm=r['grad_norm'] * f))           # noqa: E731
    assert _life(amended, 'D6', resume_fn=ce(1 + 5e-4))['valid']
    assert _life(amended, 'D6', resume_fn=gn(1.02))['valid']
    for tid, fn in (('D6', ce(1 + 2e-3)), ('D6', gn(1.04)), ('S1', ce(1 + 5e-4)), ('S1', gn(1.02))):
        rec = _life(amended, tid, resume_fn=fn)
        assert not rec['valid'] and AH.reasons(rec, 'lifecycle.first_resumed_update'), tid


def test_pre_amendment_design_has_no_deterministic_rule(legacy):
    rec = _life(legacy, flags={'deterministic': False, 'deterministic_overrides': None})
    assert rec['valid'], rec['invalid_reasons']
    assert rec['harness']['lifecycle']['deterministic']['rule'] is None
    assert not any(c['name'] == 'lifecycle.deterministic' for c in rec['harness']['checks'])


def test_deterministic_lifecycle_compares_backends_flags_without_its_two_keys(amended):
    fr_b = AH.run(amended, make_request('S1', plan='B', purpose='score'))
    life = {'save_at': 40, 'resume_steps': 8, 'export': True, 'forward_compare': True, 'deterministic': True}

    def flip(**extra):
        return lambda r: r['invariants']['backends_flags'].update({'deterministic_algorithms': True,
                                                                  'cudnn.deterministic': True, **extra})

    def run(flags=None, mutate=None):
        req = make_request('S1', purpose='lifecycle', plan='B', numerics_role='candidate', lifecycle=dict(life))
        return AH.run_life(amended, req, reference=fr_b, flags={**DUMMY_MOD.LIFECYCLE_FLAGS, **(flags or {})}, mutate=mutate)
    ok = run(mutate=flip())
    assert ok['valid'], ok['invalid_reasons']
    assert any(c['name'] == 'ref.backends_flags' and c['ok'] for c in ok['harness']['checks'])
    plain = run()
    assert ok['harness']['bindings']['precision_digest'] == plain['harness']['bindings']['precision_digest']
    nondet = run(flags={'deterministic': False}, mutate=flip())                 # not a deterministic lifecycle: equal flags
    assert not nondet['valid'] and AH.reasons(nondet, 'ref.backends_flags')
    tf32 = run(mutate=flip(**{'cuda.matmul.allow_tf32': True}))               # every other precision flag still binds
    assert not tf32['valid'] and AH.reasons(tf32, 'ref.backends_flags')


def test_driver_asks_every_lifecycle_for_deterministic_kernels():
    from infrarsi.agent.harness_runner import HarnessScreenRunner
    from infrarsi.dispatch.handlers import _lifecycle_block
    for tid in ('S2', 'D4', 'F1'):
        task = TaskSpec.load(tid)
        assert _lifecycle_block(task)['deterministic'] is True
        req = HarnessScreenRunner(lambda r, **k: {}, ['WINH0']).request(task, {}, 'lifecycle', 'R', role='reference')
        assert req.lifecycle['deterministic'] is True


# ---------------------------------------------------------------- model-defined fp32 tensors (Codex R3 HIGH)

GDN = {'param': {'torch.bfloat16': 296}, 'grad': {'torch.float32': 300}, 'master': {'torch.float32': 150},
       'exp_avg': {'torch.float32': 150}, 'exp_avg_sq': {'torch.float32': 150}, 'model_fp32': {'torch.float32': 4}}
NAMES = ['decoder.layers.0.self_attention.A_log:torch.float32', 'decoder.layers.0.self_attention.dt_bias:torch.float32',
         'decoder.layers.1.self_attention.A_log:torch.float32', 'decoder.layers.1.self_attention.dt_bias:torch.float32']


def _census(tmp_path, census, names, request=None, reference=None):
    req = request or make_request('D2')
    out = attempt_dir(tmp_path, req)
    rewrite_receipt(out, lambda r: r['invariants'].update(dtype_census=copy.deepcopy(census), model_defined_fp32_params=list(names)))
    return validate_dir(out, req, reference)


def test_model_defined_fp32_tensors_must_be_exactly_fp32(tmp_path):
    ok = _census(tmp_path, GDN, NAMES)
    assert ok['valid'], ok['invalid_reasons']
    half = _census(tmp_path, GDN, NAMES[:1] + ['decoder.layers.0.self_attention.dt_bias:torch.float16'] + NAMES[2:])
    assert not half['valid'] and AH.reasons(half, 'dtype.model_defined_fp32')
    wide = _census(tmp_path, dict(GDN, model_fp32={'torch.float32': 3, 'torch.float64': 1}), NAMES)
    assert not wide['valid'] and AH.reasons(wide, 'dtype.model_defined_fp32')
    other = _census(tmp_path, GDN, NAMES + ['decoder.layers.0.mlp.linear_fc1.weight:torch.float32'])
    assert not other['valid'] and AH.reasons(other, 'dtype.model_defined_fp32')
    hidden = _census(tmp_path, dict(GDN, param={'torch.bfloat16': 296, 'torch.float32': 4}), NAMES)
    assert not hidden['valid'] and AH.reasons(hidden, 'dtype.param')            # the fixed census: no fp32 allowance
    legacy = _census(tmp_path, {k: v for k, v in dict(GDN, param={'torch.bfloat16': 298, 'torch.float32': 2}).items()
                                if k != 'model_fp32'}, ['A_log', 'dt_bias'])
    assert legacy['valid'], legacy['invalid_reasons']                            # pre-fix receipts keep their allowance


def test_dropping_a_model_defined_fp32_tensor_differs_from_the_reference(tmp_path):
    ref = _census(tmp_path, GDN, NAMES)
    cast = {k: v for k, v in dict(GDN, param={'torch.bfloat16': 300}).items() if k != 'model_fp32'}
    cand = _census(tmp_path, dict(cast, model_fp32={}), [], request=make_request('D2', numerics_role='candidate'), reference=ref)
    assert not cand['valid'] and AH.reasons(cand, 'ref.dtype_census')


def test_harness_and_runtime_agree_on_the_fp32_names():
    from infrarsi.runtime import gates
    assert V.MODEL_FP32_FRAGMENTS == gates.FP32_PARAM_FRAGMENTS and V.MODEL_FP32_KEY in gates.EXPECTED_DTYPES
    assert gates.OPTIONAL_CENSUS == (V.MODEL_FP32_KEY,)
