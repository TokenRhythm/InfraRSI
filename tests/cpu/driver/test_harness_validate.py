"""Signals re-derivation, receipt validation, numerics classes (E0/E1/FAIL), tamper detection, and the fail-closed
rules of Codex R1: the reference loader (finding 3), mandatory invariants (4), consumed packs (6), signals binding (7)."""
import copy
import json
import math
import statistics

import pytest

from driver_helpers import (DUMMY_MOD, attempt_dir, data_facts, expected_data, make_request, pin_overlay,
                            rewrite_receipt, rewrite_signals, simulated, validate_dir)

from infrarsi.harness import derive, signing, validate as V
from infrarsi.harness.dataindex import per_update_digests


TOL = {'main_ce_rel_max_per_step': 0.005, 'main_ce_rel_mean_over_window': 0.002, 'grad_norm_rel_max_first_update': 0.01,
       'grad_norm_log_bias_window_max': 0.10, 'grad_norm_rel_per_step': 'report_only', 'repeated_adjacent_scalars': 'invalid'}


def _reasons(rec, prefix):
    return [x for x in rec['invalid_reasons'] if x.startswith(f'harness: {prefix}')]


def _resign(r):
    signing.sign(r)                                   # the tests hold the session key; WINH never does
    assert V.verify_receipt_digest(r)
    return r


def test_valid_screen_rederived_from_signals(tmp_path):
    req = make_request('D2')
    st = lambda r, u: 0.5 + 0.01 * r + 0.001 * (u % 3)                     # noqa: E731
    rec = simulated(tmp_path, req, step_time=st)
    assert rec['valid'], rec['invalid_reasons']
    a, b = req.window
    want = [max(st(r, u) for r in range(8)) for u in range(a, b + 1)]
    assert abs(rec['step_time']['mean_window_max_rank_s'] - statistics.fmean(want)) < 1e-12
    assert abs(rec['step_time']['rsd_window'] - statistics.stdev(want) / statistics.fmean(want)) < 1e-12
    assert rec['per_step'][0]['slowest_rank'] == 7 and rec['steps_completed'] == 30
    assert rec['step_time']['inter_step_source'] == 'signals.inter_step_s'
    assert rec['step_time']['inter_step_mean_s'] == pytest.approx(statistics.fmean(want[:-1]))   # start(k) - start(k-1)
    assert V.verify_receipt_digest(rec) and rec['harness']['bindings']['signals_bound']
    assert rec['harness']['numerics']['is_reference'] is True


def test_e0_e1_fail_against_reference(tmp_path):
    ref = simulated(tmp_path, make_request('D2'))
    same = simulated(tmp_path, make_request('D2', config={'tensor_model_parallel_size': 2}), reference=ref)
    assert same['harness']['numerics']['class'] == 'E0' and same['valid']
    near = simulated(tmp_path, make_request('D2'), reference=ref, ce_scale=1.001)
    assert near['harness']['numerics']['class'] == 'E1' and near['valid']
    far = simulated(tmp_path, make_request('D2'), reference=ref, ce_scale=1.01)
    assert far['harness']['numerics']['class'] == 'FAIL' and not far['valid']
    assert any('numerics.class' in r for r in far['invalid_reasons'])


def test_plan_b_run_cannot_use_a_plan_a_reference(tmp_path):
    ref = simulated(tmp_path, make_request('D2', plan='A'))
    cand = simulated(tmp_path, make_request('D2', plan='B', purpose='score', numerics_role='candidate'), reference=ref)
    assert cand['harness']['numerics']['compared'] is False and cand['harness']['numerics']['is_reference'] is False
    assert not cand['valid'] and _reasons(cand, 'numerics.verified') and _reasons(cand, 'ref.usable')
    # a reference-role run handed an unusable reference is invalid too; without one it is the reference
    assert not simulated(tmp_path, make_request('D2', plan='B', purpose='score'), reference=ref)['valid']
    fr_b = simulated(tmp_path, make_request('D2', plan='B', purpose='score'))
    assert fr_b['valid'] and fr_b['harness']['numerics']['is_reference'] is True


def test_stopped_attempt_is_invalid(tmp_path):
    rec = simulated(tmp_path, make_request('D2'), stop_at=5, stop={'kind': 'request', 'reason': 'test'})
    assert not rec['valid'] and rec['steps_completed'] == 5
    assert rec['harness']['failure_kind'] == 'stopped'


def test_oom_probe_classified(tmp_path):
    rec = simulated(tmp_path, make_request('D2', purpose='probe'), oom=True)
    assert not rec['valid'] and rec['harness']['failure_kind'] == 'oom'


def test_missing_receipt_is_synthesized(tmp_path):
    req = make_request('D2')
    out = attempt_dir(tmp_path, req, write_receipt=False)
    rec = validate_dir(out, req)
    assert rec['synthesized_by_harness'] and not rec['valid'] and rec['step_time'] is None


def _tamper(tmp_path, mutate, request=None, reference=None):
    req = request or make_request('D2')
    out = attempt_dir(tmp_path, req)
    rewrite_receipt(out, lambda r: mutate(r, out))
    return validate_dir(out, req, reference)


def test_receipt_summary_tamper_detected(tmp_path):
    rec = _tamper(tmp_path, lambda r, out: r['step_time'].update(mean_window_max_rank_s=r['step_time']['mean_window_max_rank_s'] * 0.8))
    assert not rec['valid'] and any('receipt.step_time' in x for x in rec['invalid_reasons'])
    # the validated copy carries the signals-derived value, not the forged one
    assert rec['step_time']['mean_window_max_rank_s'] > rec['harness']['reported']['step_time']['mean_window_max_rank_s']


def test_silent_precision_drop_detected(tmp_path):
    rec = _tamper(tmp_path, lambda r, out: r['invariants']['dtype_census'].update(exp_avg_sq={'torch.bfloat16': 100}))
    assert not rec['valid'] and any('dtype.exp_avg_sq' in x for x in rec['invalid_reasons'])


def test_census_count_maps_and_model_defined_fp32(tmp_path):
    counts = {'param': {'torch.bfloat16': 300}, 'grad': {'torch.float32': 300}, 'master': {'torch.float32': 150},
              'exp_avg': {'torch.float32': 150}, 'exp_avg_sq': {'torch.float32': 150}}
    rec = _tamper(tmp_path, lambda r, out: r['invariants'].update(dtype_census=counts))
    assert rec['valid'], rec['invalid_reasons']
    mixed = json.loads(json.dumps(counts))
    mixed['exp_avg_sq'] = {'torch.float32': 140, 'torch.bfloat16': 10}
    rec = _tamper(tmp_path, lambda r, out: r['invariants'].update(dtype_census=mixed))
    assert not rec['valid'] and any('dtype.exp_avg_sq' in x for x in rec['invalid_reasons'])
    gdn = json.loads(json.dumps(counts))
    gdn['param'] = {'torch.bfloat16': 298, 'torch.float32': 2}
    rec = _tamper(tmp_path, lambda r, out: r['invariants'].update(dtype_census=gdn, model_defined_fp32_params=['A_log', 'dt_bias']))
    assert rec['valid'], rec['invalid_reasons']
    rec = _tamper(tmp_path, lambda r, out: r['invariants'].update(dtype_census=gdn, model_defined_fp32_params=['A_log']))
    assert not rec['valid']


def _clip(values):
    def mutate(r, out):
        rewrite_signals(out, 0, lambda row: dict(row, clip_calls=values[row['step'] - 1]), 8)
        r['signals_digest'] = DUMMY_MOD.signals_digest(str(out), 8)
        r['invariants']['clip_calls_total'] = sum(values)
    return mutate


def test_clip_calls_per_update_counts(tmp_path):
    assert _tamper(tmp_path, _clip([2] * 30))['valid']                         # hybrid offload: 2 per update
    skipped = _tamper(tmp_path, _clip([1] * 10 + [0] + [1] * 19))
    assert not skipped['valid'] and _reasons(skipped, 'signals.clip_calls')
    cumulative = _tamper(tmp_path, _clip(list(range(1, 31))))                    # INTERFACES: per update, not cumulative
    assert not cumulative['valid'] and _reasons(cumulative, 'signals.clip_calls')
    total = _tamper(tmp_path, lambda r, out: r['invariants'].update(clip_calls_total=29))
    assert not total['valid'] and _reasons(total, 'inv.clip_calls_total')


def test_token_count_change_detected(tmp_path):
    rec = _tamper(tmp_path, lambda r, out: r['invariants'].update(physical_tokens_per_update=131072))
    assert not rec['valid'] and any('physical_tokens_per_update' in x for x in rec['invalid_reasons'])


def test_layout_that_ran_must_be_the_requested_one(tmp_path):
    rec = _tamper(tmp_path, lambda r, out: r['effective_config'].update(tensor_model_parallel_size=4))
    assert not rec['valid'] and _reasons(rec, 'inv.layout')
    gone = _tamper(tmp_path, lambda r, out: r.pop('effective_config'))
    assert not gone['valid'] and _reasons(gone, 'inv.layout') and _reasons(gone, 'inv.schema')


# ---------------------------------------------------------------- finding 7: signals binding

def test_signals_rewritten_after_receipt_detected(tmp_path):
    def mutate(r, out):
        p = out / 'signals' / 'rank3.jsonl'
        p.write_text(p.read_text().replace('"data_s": 0.013', '"data_s": 0.02'))
    rec = _tamper(tmp_path, mutate)
    assert _reasons(rec, 'receipt.signals_digest') and not rec['valid']
    assert rec['step_time'] is None and rec['profile'] is None           # nothing scored from unbound files


@pytest.mark.parametrize('bad', [None, '', 'ab' * 31, 'AB' * 32, 12345])
def test_null_or_malformed_signals_digest_rejected(tmp_path, bad):
    rec = _tamper(tmp_path, lambda r, out: r.update(signals_digest=bad))
    assert not rec['valid'] and _reasons(rec, 'receipt.signals_digest') and rec['step_time'] is None
    assert rec['harness']['bindings']['signals_bound'] is False


def test_alternate_digest_conventions_rejected(tmp_path):
    """Only sha256 of the rank files concatenated in increasing rank order binds the signals (INTERFACES)."""
    import hashlib

    def per_file(r, out):
        per = [hashlib.sha256((out / 'signals' / f'rank{k}.jsonl').read_bytes()).hexdigest() for k in range(8)]
        r['signals_digest'] = hashlib.sha256(json.dumps(per, separators=(',', ':')).encode()).hexdigest()

    def reversed_order(r, out):
        h = hashlib.sha256()
        for k in reversed(range(8)):
            h.update((out / 'signals' / f'rank{k}.jsonl').read_bytes())
        r['signals_digest'] = h.hexdigest()
    for mutate in (per_file, reversed_order):
        rec = _tamper(tmp_path, mutate)
        assert not rec['valid'] and _reasons(rec, 'receipt.signals_digest')
    texts = {f'rank{k}.jsonl': f'{{"step": {k}}}\n' for k in (10, 2, 1)}
    assert derive.signals_digest(texts) == hashlib.sha256(''.join(texts[f'rank{k}.jsonl'] for k in (1, 2, 10)).encode()).hexdigest()


# ---------------------------------------------------------------- finding 3: the fail-closed reference loader

@pytest.fixture
def ref_and_req(tmp_path):
    return simulated(tmp_path, make_request('D2')), make_request('D2', numerics_role='candidate')


def _ref_case(ref, edit, resign=True):
    r = copy.deepcopy(ref)
    edit(r)
    return _resign(r) if resign else r


def test_one_row_reference_cannot_authorize_a_candidate(tmp_path, ref_and_req):
    """Codex R1 3A: a one-row reference matching update 1 must not give allowed_ok."""
    ref, req = ref_and_req
    one = _ref_case(ref, lambda r: r.update(per_step=r['per_step'][:1]))
    probs = V.reference_problems(one, task=req.task, plan='A', steps=30, window=req.window)
    assert any('lacks updates' in p for p in probs) and any('scoring window' in p for p in probs)
    res = V.compare_numerics(simulated(tmp_path, make_request('D2'))['per_step'], one['per_step'], {}, (17, 30), steps=30)
    assert res['class'] == 'FAIL' and not res['allowed_ok'] and any('reference lacks updates' in v for v in res['violations'])
    cand = simulated(tmp_path, req, reference=one)
    assert not cand['valid'] and _reasons(cand, 'ref.usable') and _reasons(cand, 'numerics.verified')


def test_nan_reference_is_rejected(tmp_path, ref_and_req):
    """Codex R1 3B: NaN CE / grad norm in the reference must not pass as E1."""
    ref, req = ref_and_req
    rows = [dict(p, main_ce=math.nan, grad_norm=math.nan) for p in ref['per_step']]
    res = V.compare_numerics(simulated(tmp_path, make_request('D2'))['per_step'], rows,
                             TOL, (17, 30), steps=30)
    assert res['class'] == 'FAIL' and any('non-finite reference' in v for v in res['violations'])
    nan_ref = copy.deepcopy(ref)
    nan_ref['per_step'] = [dict(p, main_ce=None) for p in nan_ref['per_step']]      # stored NaN reads back as null
    probs = V.reference_problems(_resign(nan_ref), task=req.task, plan='A', steps=30, window=req.window)
    assert any('non-finite' in p for p in probs)


@pytest.mark.parametrize('case', ['unsigned', 'invalid', 'plan_swapped', 'stale_task', 'candidate_role', 'wrong_purpose',
                                  'incomplete', 'missing_invariant'])
def test_reference_authenticity_and_identity(tmp_path, ref_and_req, case):
    """Codex R1 3C: unsigned, invalid, plan-swapped, stale-task, candidate-role, wrong-purpose or incomplete receipts
    at the reference path are unusable."""
    ref, req = ref_and_req
    edits = {'unsigned': (lambda r: r['step_time'].update(mean_window_max_rank_s=0.1), False),
             'invalid': (lambda r: r.update(valid=False), True),
             'plan_swapped': (lambda r: r.update(input_plan='B'), True),
             'stale_task': (lambda r: r.update(task_digest='0' * 64), True),
             'candidate_role': (lambda r: r.update(numerics_role='candidate'), True),
             'wrong_purpose': (lambda r: r.update(purpose='probe'), True),
             'incomplete': (lambda r: r.update(steps_completed=29), True),
             'missing_invariant': (lambda r: r['invariants'].pop('loss_definition_digest'), True)}
    edit, resign = edits[case]
    bad = _ref_case(ref, edit, resign)
    assert V.reference_problems(bad, task=req.task, plan='A', steps=30, window=req.window)
    cand = simulated(tmp_path, make_request('D2', numerics_role='candidate'), reference=bad)
    assert not cand['valid'] and _reasons(cand, 'ref.usable')
    assert not V.reference_problems(ref, task=req.task, plan='A', steps=30, window=req.window)


def test_recheck_against_fails_closed(tmp_path):
    from infrarsi.harness.runner import ReferenceUnusable, recheck_against
    cand = simulated(tmp_path, make_request('D2', plan='B', purpose='score', numerics_role='candidate'))
    fr_b = simulated(tmp_path, make_request('D2', plan='B', purpose='score'))
    assert recheck_against(cand, fr_b)['valid']
    with pytest.raises(ReferenceUnusable):
        recheck_against(cand, _ref_case(fr_b, lambda r: r.update(per_step=r['per_step'][:3])))
    with pytest.raises(ReferenceUnusable):
        recheck_against(cand, simulated(tmp_path, make_request('D2', plan='A')))      # plan-A reference for plan B
    forged = copy.deepcopy(cand)
    forged['step_time']['mean_window_max_rank_s'] = 0.01                                 # altered after signing
    with pytest.raises(ValueError):
        recheck_against(forged, fr_b)


# ---------------------------------------------------------------- finding 4: mandatory, task-aware invariants

@pytest.mark.parametrize('field', ['loss_definition_digest', 'trainable_param_digest', 'lr_schedule_digest',
                                   'model_config_digest', 'backends_flags', 'clip_calls_total', 'dtype_census.exp_avg'])
def test_omitted_invariant_on_both_sides_is_not_equality(tmp_path, field):
    """Codex R1 4: a reference and a candidate that both omit a scored invariant do not validate."""
    def drop(r, out=None):
        if '.' in field:
            a, b = field.split('.')
            r['invariants'][a].pop(b)
        else:
            r['invariants'].pop(field)
    ref_req = make_request('D2')
    ref = _tamper(tmp_path, drop, request=ref_req)
    assert not ref['valid'] and _reasons(ref, 'inv.schema')
    assert V.reference_problems(_resign(dict(ref, valid=True)), task=ref_req.task, plan='A', steps=30, window=ref_req.window)
    cand = _tamper(tmp_path, drop, request=make_request('D2', numerics_role='candidate'))
    assert not cand['valid']


def test_moe_router_and_capacity_fields_are_required(tmp_path):
    req = make_request('D4')
    ok = simulated(tmp_path, req)
    assert ok['valid'], ok['invalid_reasons']
    for k in ('router_topk', 'num_experts', 'capacity_factor'):
        rec = _tamper(tmp_path, lambda r, out, k=k: r['invariants'].pop(k), request=make_request('D4'))
        assert not rec['valid'] and _reasons(rec, 'inv.schema'), k
    cf = _tamper(tmp_path, lambda r, out: r['invariants'].update(capacity_factor=1.25), request=make_request('D4'))
    assert not cf['valid'] and _reasons(cf, 'inv.capacity_factor')
    dense = simulated(tmp_path, make_request('D2'))                            # dense tasks: router fields are None
    assert dense['valid'] and dense['invariants']['router_topk'] is None


def test_lr_schedule_must_match_the_task(tmp_path):
    rec = _tamper(tmp_path, lambda r, out: r['invariants'].update(lr_schedule_expected_digest='1' * 64))
    assert not rec['valid'] and _reasons(rec, 'inv.schema')


def test_candidate_invariants_must_equal_the_reference(tmp_path):
    ref = simulated(tmp_path, make_request('D2'))
    for key, val in (('trainable_param_digest', '2' * 64), ('backends_flags', {'cuda.matmul.allow_tf32': True}),
                     ('loss_definition_digest', '3' * 64)):
        cand = _tamper(tmp_path, lambda r, out, key=key, val=val: r['invariants'].update({key: val}),
                       request=make_request('D2', numerics_role='candidate'), reference=ref)
        assert not cand['valid'] and _reasons(cand, f'ref.{key}'), key


# ---------------------------------------------------------------- finding 6: consumed packs and token vectors

def test_consumed_pack_digests_are_mandatory(tmp_path):
    """Codex R1 6: full-plan order digest, correct totals and no consumed digest is not enough."""
    rec = _tamper(tmp_path, lambda r, out: [r['invariants'].pop(k) for k in ('data_consumed_digest', 'data_consumed_per_update')])
    assert not rec['valid'] and _reasons(rec, 'data.consumed_digest') and _reasons(rec, 'data.consumed_per_update')


def test_repeated_or_reordered_packs_detected(tmp_path):
    req = make_request('D2')
    facts = data_facts(req)
    order = list(facts['order'])
    ppu = facts['ppu']
    repeated = order[:ppu] * 2 + order[2 * ppu:]                                   # update 2 replays update 1's packs
    rec = simulated(tmp_path, req, data=dict(facts, order=repeated))
    assert not rec['valid'] and _reasons(rec, 'data.consumed_digest') and _reasons(rec, 'data.consumed_per_update')
    # the INDEX order digest is the only accepted data_order_digest (no per-update fallback)
    alt = _tamper(tmp_path, lambda r, out: r['invariants'].update(data_order_digest=r['invariants']['data_consumed_digest']))
    assert not alt['valid'] and _reasons(alt, 'data.order_digest')


def test_per_update_digest_vector_must_cover_every_update(tmp_path):
    def short(r, out):
        r['invariants']['data_consumed_per_update'] = r['invariants']['data_consumed_per_update'][:-1]
    rec = _tamper(tmp_path, short)
    assert not rec['valid'] and _reasons(rec, 'data.consumed_per_update')

    def extra(r, out):
        r['invariants']['data_consumed_per_update'] = r['invariants']['data_consumed_per_update'] + ['0' * 64]
    assert _reasons(_tamper(tmp_path, extra), 'data.consumed_per_update')
    req = make_request('D2')
    exp = expected_data(req)
    assert per_update_digests(exp['order'], exp['packs_per_update'], 2)[1] != per_update_digests(exp['order'], exp['packs_per_update'], 2)[0]


def test_supervised_token_vectors_cover_every_update(tmp_path):
    req = make_request('D2')
    exp = dict(expected_data(req))
    short = dict(exp, supervised_tokens_per_update=exp['supervised_tokens_per_update'][:20])     # INDEX vector too short
    out = attempt_dir(tmp_path, req)
    from infrarsi.harness import remote_side
    rec = V.validate(remote_side.collect_run(str(out), 1), req, None, data=short)
    assert not rec['valid'] and _reasons(rec, 'data.supervised_vector')
    rec = _tamper(tmp_path, lambda r, out: r['invariants']['supervised_tokens_per_update'].update(
        A=r['invariants']['supervised_tokens_per_update']['A'][:-2]))
    assert not rec['valid'] and _reasons(rec, 'data.receipt_supervised_vector')
    rec = _tamper(tmp_path, lambda r, out: r['invariants']['supervised_tokens_per_update'].update(
        A=r['invariants']['supervised_tokens_per_update']['A'] + [5]))
    assert not rec['valid'] and _reasons(rec, 'data.receipt_supervised_vector')


def test_missing_data_facts_fail_closed(tmp_path):
    req = make_request('D2')
    out = attempt_dir(tmp_path, req)
    from infrarsi.harness import remote_side
    for data in (None, {'error': 'INDEX.json unavailable'}):
        rec = V.validate(remote_side.collect_run(str(out), 1), req, None, data=data)
        assert not rec['valid'] and _reasons(rec, 'data.index')


# ---------------------------------------------------------------- derivation details

def test_derive_handles_missing_rank_and_zero_base():
    texts = {'rank0.jsonl': '\n'.join(json.dumps({'step': s, 'whole_update_s': 1.0, 'main_ce': 1.0, 'grad_norm': 1.0, 'lr': 1e-5,
                                                  'physical_tokens': 1, 'supervised_tokens': 1}) for s in range(0, 4))}
    d = derive.derive(texts, world_size=2, steps=4, window=(2, 4))
    assert d['steps_completed'] == 0 and any('missing for ranks [1]' in p for p in d['problems'])
    texts['rank1.jsonl'] = '\n'.join(json.dumps({'step': s, 'whole_update_s': 2.0}) for s in range(0, 4))
    d = derive.derive(texts, world_size=2, steps=4, window=(2, 4))
    assert d['step_base'] == 0 and d['steps_completed'] == 4 and d['step_time']['mean_window_max_rank_s'] == 2.0
    assert d['per_step'][0]['slowest_rank'] == 1
    assert d['step_time']['inter_step_mean_s'] is None and any('InterStep not derivable' in p for p in d['problems'])


def test_inter_step_from_runtime_signals_convention():
    """runtime/signals.py rows carry inter_step_s; InterStep = mean over a+1..b of the max-rank value."""
    rows = {r: [] for r in range(2)}
    for u in range(1, 6):
        for r in range(2):
            row = {'step': u, 'whole_update_s': 1.0 + r, 'inter_step_s': None if u == 1 else 1.1 + 0.1 * r + 0.01 * u}
            if r == 0:
                row.update(main_ce=1.0, grad_norm=1.0, lr=1e-5, physical_tokens=1, supervised_tokens=1)
            rows[r].append(row)
    texts = {f'rank{r}.jsonl': '\n'.join(json.dumps(x) for x in v) for r, v in rows.items()}
    d = derive.derive(texts, world_size=2, steps=5, window=(2, 5))
    want = statistics.fmean([1.2 + 0.01 * u for u in (3, 4, 5)])
    assert d['step_time']['inter_step_source'] == 'signals.inter_step_s'
    assert abs(d['step_time']['inter_step_mean_s'] - want) < 1e-12
    rows[1][3].pop('inter_step_s')                                     # one rank silent at update 4: not derivable
    texts = {f'rank{r}.jsonl': '\n'.join(json.dumps(x) for x in v) for r, v in rows.items()}
    assert derive.derive(texts, world_size=2, steps=5, window=(2, 5))['step_time']['inter_step_mean_s'] is None


LIFE_FLAGS = {'saved_at': 6, 'resumed_ok': True, 'resume_scalars_match': True, 'state_exact': True, 'export_ok': True,
              'forward_ok': True, 'forward': {'rel_l2': 0.018, 'top1_agreement': 0.996, 'forward_ok': True}}


def _lifecycle_attempt(tmp_path, resume_ce_scale=1.0, flags=None, lifecycle=None, resume_fn=None, bind=True):
    """runtime trainer layout: signals/ runs 1..save_at+resume_steps, signals_resume/ the resumed save_at+1..
    resume_fn(row) edits the resumed rank-0 rows (steps 7..10); bind=False omits lifecycle.signals_resume_digest."""
    req = make_request('S1', purpose='lifecycle', plan='B', steps=6,
                       lifecycle=lifecycle or {'save_at': 6, 'resume_steps': 4, 'export': True, 'forward_compare': True})
    expected_data(req)
    pin_overlay(req)
    out = tmp_path / req.attempt_id

    def edit(rank, row):
        row = dict(row, main_ce=row['main_ce'] * resume_ce_scale) if rank == 0 else row
        return (resume_fn or (lambda x: x))(row) if rank == 0 else row
    DUMMY_MOD.simulate_lifecycle(req.to_dict(), str(out), flags=copy.deepcopy(LIFE_FLAGS) if flags is None else flags,
                                 resume_fn=edit, bind=bind, data=data_facts(req))
    return validate_dir(out, req)


def test_resume_signals_must_be_bound(tmp_path):
    """Codex R2 finding 3: no lifecycle.signals_resume_digest, or signals_resume/ replaced after the process wrote its
    digest (here with a copy of the uninterrupted rows hiding a failed restore), is invalid; continuity is not derived."""
    unbound = _lifecycle_attempt(tmp_path, bind=False)
    assert not unbound['valid'] and _reasons(unbound, 'lifecycle.signals_resume_digest')
    assert unbound['harness']['lifecycle'] is None
    req = make_request('S1', purpose='lifecycle', plan='B', steps=6,
                       lifecycle={'save_at': 6, 'resume_steps': 4, 'export': True, 'forward_compare': True})
    expected_data(req)
    pin_overlay(req)
    out = tmp_path / req.attempt_id
    DUMMY_MOD.simulate_lifecycle(req.to_dict(), str(out), flags=copy.deepcopy(LIFE_FLAGS), data=data_facts(req),
                                 resume_fn=lambda r, row: dict(row, main_ce=row['main_ce'] * 1.3))   # a failed restore
    p = out / 'signals_resume' / 'rank0.jsonl'
    rows = [json.loads(x) for x in (out / 'signals' / 'rank0.jsonl').read_text().splitlines()]
    p.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows if r['step'] > 6))    # swapped afterwards
    rec = validate_dir(out, req)
    assert not rec['valid'] and _reasons(rec, 'lifecycle.signals_resume_digest')


def test_lifecycle_validation(tmp_path):
    ok = _lifecycle_attempt(tmp_path)
    assert ok['valid'], ok['invalid_reasons']
    assert {c['name'] for c in ok['harness']['checks'] if c['ok']} >= {
        'lifecycle.resume_complete', 'lifecycle.state_exact', 'lifecycle.first_resumed_update', 'lifecycle.continuation',
        'lifecycle.exact_scalars', 'lifecycle.flags', 'lifecycle.forward', 'data.supervised_tokens_total'}
    assert ok['harness']['lifecycle']['first_resumed_update'] == 7 and len(ok['harness']['lifecycle']['steps']) == 4
    drift = _lifecycle_attempt(tmp_path, resume_ce_scale=1.02)
    assert not drift['valid'] and any('lifecycle.continuation' in x for x in drift['invalid_reasons'])
    noexp = _lifecycle_attempt(tmp_path, flags=dict(LIFE_FLAGS, export_ok=False))
    assert not noexp['valid'] and any('lifecycle.flags' in x for x in noexp['invalid_reasons'])


def test_lifecycle_state_must_be_restored_exactly(tmp_path):
    """AMENDMENT-001: saved and restored state digests bitwise equal (receipt lifecycle.state_exact)."""
    for flags in (dict(LIFE_FLAGS, state_exact=False), {k: v for k, v in LIFE_FLAGS.items() if k != 'state_exact'}):
        rec = _lifecycle_attempt(tmp_path, flags=flags)
        assert not rec['valid'] and _reasons(rec, 'lifecycle.state_exact')


def test_lifecycle_first_resumed_update_uses_the_tight_gate(tmp_path):
    """The first resumed update: CE <= 1e-4 and grad norm <= 5e-3 against the uninterrupted run; later updates: CE
    within the task per-step tolerance, grad norm reported only (T-train's S2 case: 11.5 % at the second resumed step)."""
    first = lambda f: (lambda r: f(r) if r['step'] == 7 else r)          # noqa: E731
    later = lambda f: (lambda r: f(r) if r['step'] == 8 else r)          # noqa: E731
    rec = _lifecycle_attempt(tmp_path, resume_fn=first(lambda r: dict(r, main_ce=r['main_ce'] * (1 + 2e-4))))
    assert not rec['valid'] and _reasons(rec, 'lifecycle.first_resumed_update')
    rec = _lifecycle_attempt(tmp_path, resume_fn=first(lambda r: dict(r, grad_norm=r['grad_norm'] * 1.01)))
    assert not rec['valid'] and _reasons(rec, 'lifecycle.first_resumed_update')
    rec = _lifecycle_attempt(tmp_path, resume_fn=first(lambda r: dict(r, grad_norm=r['grad_norm'] * 1.004,
                                                                       main_ce=r['main_ce'] * (1 + 5e-5))))
    assert rec['valid'], rec['invalid_reasons']
    s2 = _lifecycle_attempt(tmp_path, resume_fn=later(lambda r: dict(r, grad_norm=r['grad_norm'] * 1.115)))
    assert s2['valid'], s2['invalid_reasons']
    assert s2['harness']['lifecycle']['grad_norm_rel_max_later'] == pytest.approx(0.115)
    rec = _lifecycle_attempt(tmp_path, resume_fn=later(lambda r: dict(r, main_ce=r['main_ce'] * 1.006)))
    assert not rec['valid'] and _reasons(rec, 'lifecycle.continuation')
    rec = _lifecycle_attempt(tmp_path, resume_fn=later(lambda r: dict(r, main_ce=r['main_ce'] * 1.004)))
    assert rec['valid'], rec['invalid_reasons']


@pytest.mark.parametrize('key,value', [('lr', 2e-5), ('supervised_tokens', 1), ('clip_calls', 2), ('physical_tokens', 7)])
def test_lifecycle_lr_tokens_and_clip_calls_are_exact(tmp_path, key, value):
    rec = _lifecycle_attempt(tmp_path, resume_fn=lambda r: dict(r, **{key: value}) if r['step'] == 9 else r)
    assert not rec['valid'] and _reasons(rec, 'lifecycle.exact_scalars')


def test_lifecycle_forward_metrics(tmp_path):
    bad = _lifecycle_attempt(tmp_path, flags=dict(LIFE_FLAGS, forward={'rel_l2': 0.06, 'top1_agreement': 0.99}))
    assert not bad['valid'] and _reasons(bad, 'lifecycle.forward')
    bad = _lifecycle_attempt(tmp_path, flags=dict(LIFE_FLAGS, forward={'rel_l2': 0.01, 'top1_agreement': 0.85}))
    assert not bad['valid'] and _reasons(bad, 'lifecycle.forward')
    gone = _lifecycle_attempt(tmp_path, flags={k: v for k, v in LIFE_FLAGS.items() if k != 'forward'})
    assert not gone['valid'] and _reasons(gone, 'lifecycle.forward')


def test_lifecycle_flags_only_for_enabled_phases(tmp_path):
    """T-train item 7: export_ok / forward_ok (and the forward metrics) only when the request enables them."""
    off = {'save_at': 6, 'resume_steps': 4, 'export': False, 'forward_compare': False}
    minimal = {'resumed_ok': True, 'state_exact': True}
    rec = _lifecycle_attempt(tmp_path, lifecycle=off, flags=minimal)
    assert rec['valid'], rec['invalid_reasons']
    rec = _lifecycle_attempt(tmp_path, lifecycle=dict(off, export=True), flags=minimal)
    assert not rec['valid'] and any("['export_ok']" in x for x in rec['invalid_reasons'])


def test_lifecycle_against_a_plan_b_reference_uses_its_own_step_counts(tmp_path):
    """A item 7: a lifecycle attempt runs save_at + resume_steps updates; its optimizer_steps, LR digest, supervised
    total and consumed digest are checked against its own expected counts, everything else against the reference."""
    fr_b = simulated(tmp_path, make_request('S1', plan='B', purpose='score'))
    req = make_request('S1', purpose='lifecycle', plan='B', numerics_role='candidate',
                       lifecycle={'save_at': 40, 'resume_steps': 8, 'export': True, 'forward_compare': True})
    expected_data(req)
    pin_overlay(req)
    out = tmp_path / req.attempt_id
    DUMMY_MOD.simulate_lifecycle(req.to_dict(), str(out), data=data_facts(req))
    life = validate_dir(out, req, reference=fr_b)
    assert life['valid'], life['invalid_reasons']
    assert life['invariants']['optimizer_steps'] == 48 and fr_b['invariants']['optimizer_steps'] == 40
    names = {c['name'] for c in life['harness']['checks']}
    assert 'ref.optimizer_steps' not in names and 'ref.lr_schedule_digest' not in names and 'ref.trainable_param_digest' in names
    assert life['harness']['numerics']['compared'] and life['harness']['numerics']['class'] == 'E0'


def test_compare_numerics_lr_and_aux_vanished():
    ref = [{'update': u, 'main_ce': 2.0 + 0.1 * u, 'aux_loss': 0.01, 'grad_norm': 1.0 + 0.1 * u, 'lr': 1e-5} for u in range(1, 4)]
    cand = [dict(p) for p in ref]
    cand[1]['aux_loss'] = 0.0
    cand[2]['lr'] = 2e-5
    res = V.compare_numerics(cand, ref, TOL, (1, 3), steps=3)
    assert res['class'] == 'FAIL'
    assert any('aux loss vanished' in v for v in res['violations']) and any('LR differs' in v for v in res['violations'])
    gap = V.compare_numerics([dict(p) for p in ref], ref[:1] + ref[2:], TOL, (1, 3), steps=3)
    assert gap['class'] == 'FAIL' and gap['compared'] and any('reference lacks updates [2]' in v for v in gap['violations'])


# ---------------------------------------------------------------- AMENDMENT-001 grad-norm gates and self-check

def _gn_case(tmp_path, gn_scale, **kw):
    ref = simulated(tmp_path, make_request('D2'))
    return simulated(tmp_path, make_request('D2', numerics_role='candidate', config={'tensor_model_parallel_size': 2}),
                     reference=ref, gn_scale=gn_scale, **kw)


@pytest.mark.parametrize('scale', [1.56, 0.44])
def test_single_step_grad_norm_excursion_passes_under_the_window_bias(tmp_path, scale):
    """S2-like noise: one +-56 % excursion inside the 14-update window moves the mean log ratio by about 0.03-0.06."""
    rec = _gn_case(tmp_path, lambda u: scale if u == 20 else 1.0)
    num = rec['harness']['numerics']
    assert rec['valid'], rec['invalid_reasons']
    assert num['class'] == 'E1' and num['grad_norm']['rel_max_all_steps'] == pytest.approx(abs(scale - 1))
    assert num['grad_norm']['log_bias_window'] == pytest.approx(math.log(scale) / 14) and num['grad_norm']['n_window'] == 14
    assert len(num['grad_norm']['per_step_rel']) == 30 and num['grad_norm']['first_update_rel'] == 0.0


def test_systematic_grad_norm_bias_fails(tmp_path):
    """s03-like local norm under DP2: ln(1/sqrt(2)) = -0.35 on every update after the first fails the window gate."""
    rec = _gn_case(tmp_path, lambda u: 1.0 if u == 1 else math.exp(-0.35))
    num = rec['harness']['numerics']
    assert not rec['valid'] and num['class'] == 'FAIL'
    assert num['grad_norm']['log_bias_window'] == pytest.approx(-0.35)
    assert any('window log bias' in v for v in num['violations']) and not any('update 1' in v for v in num['violations'])
    whole = _gn_case(tmp_path, lambda u: 2 ** -0.5)                       # the full s03 also fails the first-update gate
    assert any('rel dev at update 1' in v for v in whole['harness']['numerics']['violations'])


def test_first_update_grad_norm_deviation_fails(tmp_path):
    rec = _gn_case(tmp_path, lambda u: 1.02 if u == 1 else 1.0)
    num = rec['harness']['numerics']
    assert not rec['valid'] and num['class'] == 'FAIL' and num['grad_norm']['first_update_rel'] == pytest.approx(0.02)
    assert any('rel dev at update 1' in v for v in num['violations'])
    ok = _gn_case(tmp_path, lambda u: 1.008 if u == 1 else 1.0)            # within 1 %
    assert ok['valid'] and ok['harness']['numerics']['class'] == 'E1'


def test_adjacent_repeated_scalars_invalidate_any_run(tmp_path):
    """Reference-free self-check: a reference run (no comparison) with update 12 replaying update 11's scalars."""
    req = make_request('D2')
    out = attempt_dir(tmp_path, req)
    rows = {json.loads(x)['step']: json.loads(x) for x in (out / 'signals' / 'rank0.jsonl').read_text().splitlines()}
    copy_11 = lambda r: dict(r, main_ce=rows[11]['main_ce'], grad_norm=rows[11]['grad_norm']) if r['step'] == 12 else r  # noqa: E731
    rewrite_signals(out, 0, copy_11, 8)
    rec = validate_dir(out, req)
    assert not rec['valid'] and _reasons(rec, 'self_check.repeated_pairs')
    assert rec['harness']['numerics']['self_check'] == {'repeated_pairs': [[11, 12]]} and rec['harness']['numerics']['is_reference']
    from infrarsi.reward import evaluate
    assert 'repeated_adjacent_scalars' in {v.split(':')[0] for v in evaluate(rec)['violations']}
    only_ce = lambda r: dict(r, main_ce=rows[11]['main_ce']) if r['step'] == 12 else r          # noqa: E731
    out2 = attempt_dir(tmp_path, make_request('D2'))
    rewrite_signals(out2, 0, only_ce, 8)
    assert not validate_dir(out2, make_request('D2'))['harness']['numerics']['self_check']['repeated_pairs']


def test_microbatch_counts_are_an_invariant(tmp_path):
    """AMENDMENT-001 risk note: every rank runs packs_per_update / (DP x MBS) backward micro-batches every update."""
    req = make_request('D2', config={'tensor_model_parallel_size': 2})               # DP 4, MBS 1, 32 packs -> 8
    ok = simulated(tmp_path, req)
    assert ok['valid'] and ok['harness']['microbatches']['values'] == {'8': 240}
    skip = simulated(tmp_path, make_request('D2', config={'tensor_model_parallel_size': 2}),
                     microbatches=lambda r, u: 4 if (r == 5 and u == 23) else 8)          # half the backward, once
    assert not skip['valid'] and _reasons(skip, 'signals.microbatches')
    missing = simulated(tmp_path, make_request('D2'), microbatches=lambda r, u: None if u == 3 else 4)
    assert not missing['valid'] and missing['harness']['microbatches']['n_missing'] == 8
    assert V.expected_microbatches(32, 8, {'tensor_model_parallel_size': 1, 'micro_batch_size': 2}) == 2
    assert V.expected_microbatches(16, 16, {'pipeline_model_parallel_size': 2, 'micro_batch_size': 1}) == 2
    assert V.expected_microbatches(32, 8, {'tensor_model_parallel_size': 3}) is None


def test_nan_signals_do_not_crash_validation(tmp_path):
    """A NaN loss row is an invalid receipt (canonical digests reject NaN, so it is stored as null), not an exception."""
    req = make_request('D2')
    out = attempt_dir(tmp_path, req)
    rewrite_signals(out, 0, lambda r: dict(r, main_ce=float('nan')) if r['step'] == 7 else r, 8)
    rec = validate_dir(out, req)
    assert not rec['valid'] and V.verify_receipt_digest(rec) and rec['per_step'][6]['main_ce'] is None
    assert any('stability.nan_inf' in x for x in rec['invalid_reasons'])


def test_sharpness_statistics_are_reported(tmp_path):
    """AMENDMENT-001 E0 record: clip-active share and adjacent |d ln gn| (report only, never gated)."""
    rec = simulated(tmp_path, make_request('D2'))
    sh = rec['harness']['numerics']['sharpness']
    # dummy grad norm 1.5 / (1 + 0.05 u) exceeds clip_grad 1.0 on updates 1..9 of 30
    assert sh['clip_grad'] == 1.0 and sh['clip_active_fraction'] == pytest.approx(9 / 30) and sh['n_updates'] == 30
    assert sh['adjacent_abs_dlog_gn_max'] == pytest.approx(math.log(1.10 / 1.05))
    assert rec['valid']
