"""Verifier on synthetic project receipts: speed-only reward, the tolerance knob, each checklist item, feedback and
the promotion items."""
import math

import pytest

from infrarsi.harness import signing
from infrarsi.verify import SCHEMA, Verifier
from infrarsi.verify import checklist as C


def receipt(attempt='a', steps=10, loss=lambda s: 2.0 - 0.01 * s, gn=lambda s: 1.0 + 0.01 * s, lr=lambda s: 1e-4 * s,
            tokens=4096, t=1.0, rep=None, data=True, exit_code=0, completed=None, events=None, frozen=None, meta=None,
            mem=None, static=None, sign=True, **extra):
    per = []
    for s in range(1, steps + 1):
        row = {'step': s, 'loss': loss(s), 'grad_norm': gn(s), 'lr': lr(s)}
        if tokens is not None:
            row['tokens'] = tokens(s) if callable(tokens) else tokens
        if data:
            row['data'] = data(s) if callable(data) else f'batch{s}'
        per.append(row)
    ev = {'steps': steps, 'steps_completed': steps if completed is None else completed, 'exit_code': exit_code,
          'per_step': per, 'window': [3, steps], 'step_time': t, 'clock_step_time': t, 'reported_step_time': rep,
          'rsd_window': 0.005, 'events': events or {}, 'meta': meta or {}, 'memory_peak_gib': mem,
          'frozen': {'checked': True, 'n': 2, 'changed': [], 'missing': []} if frozen is None else frozen,
          'static_workload': static or {}, **extra}
    r = {'schema': SCHEMA, 'attempt_id': attempt, 'evidence': ev}
    return signing.sign(r) if sign else r


def status(v, cid):
    return {c['id']: c['status'] for c in v.checks}[cid]


@pytest.fixture
def ref():
    return receipt('ref', t=1.0)


def test_reference_run_passes_trivially(ref):
    v = Verifier()(ref, None, ref)
    assert v.valid, v.reasons
    assert status(v, 'loss_within_tolerance') == 'pass' and v.numerics_class == 'reference'
    assert v.step_time == 1.0 and v.reward is None


def test_faster_candidate_speed_only_reward(ref):
    cand = receipt('c1', t=0.8, mem=70.0)
    v = Verifier()(cand, ref, ref)
    assert v.valid, v.reasons
    assert math.isclose(v.reward, math.log(1 / 0.8))
    assert all(c['status'] != 'fail' for c in v.checks) and v.feedback == ''
    assert v.coverage['unchecked'] == ['memory_cap', 'optimizer_steps', 'same_model']   # optimizer_steps: lr only, no counter


@pytest.mark.parametrize('dev,tol,ok', [(0.02, 0.03, True), (0.05, 0.03, False), (0.05, 0.10, True)])
def test_tolerance_knob(ref, dev, tol, ok):
    cand = receipt('c', t=0.9, loss=lambda s: (2.0 - 0.01 * s) * (1 + dev))
    v = Verifier(tolerance=tol)(cand, ref, ref)
    assert v.valid is ok
    if not ok:
        assert status(v, 'loss_within_tolerance') == 'fail'
        assert 'FAILED loss_within_tolerance' in v.feedback and 'revert it or make it exact' in v.feedback


def test_tolerance_zero_is_exact(ref):
    assert Verifier(tolerance=0)(receipt('c', t=0.9), ref, ref).valid
    off = receipt('c2', t=0.9, loss=lambda s: (2.0 - 0.01 * s) * (1 + 1e-9))
    assert not Verifier(tolerance=0)(off, ref, ref).valid
    with pytest.raises(ValueError):
        Verifier(tolerance=1.5)


def test_workload_change_fails_and_is_explained(ref):
    v = Verifier()(receipt('c', t=0.5, tokens=2048), ref, ref)
    assert not v.valid and status(v, 'same_workload') == 'fail'
    assert 'micro-batch x gradient accumulation' in v.feedback


def test_missing_workload_evidence_is_required(ref):
    bare_ref, cand = receipt('r', tokens=None), receipt('c', t=0.9, tokens=None)
    v = Verifier()(cand, bare_ref, bare_ref)
    assert not v.valid and status(v, 'same_workload') == 'unchecked'
    assert 'NOT CHECKABLE (required) same_workload' in v.feedback
    static_ref = receipt('r2', tokens=None, static={'global_batch': 64})
    assert Verifier()(receipt('c2', t=0.9, tokens=None, static={'global_batch': 64}), static_ref, static_ref).valid
    v = Verifier()(receipt('c3', t=0.9, tokens=None, static={'global_batch': 32}), static_ref, static_ref)
    assert status(v, 'same_workload') == 'fail' and 'from the configuration' in v.reasons[0]


def test_data_order_drives_pairing(ref):
    shuffled = receipt('c', t=0.9, data=lambda s: f'other{s}')
    v = Verifier()(shuffled, ref, ref)
    assert status(v, 'same_data') == 'fail' and not v.valid
    no_data_ref = receipt('r', data=False)
    noisy = receipt('c2', t=0.9, data=False, loss=lambda s: (2.0 - 0.01 * s) * (1.06 if s % 2 else 0.95),
                    gn=lambda s: 2.0)
    v = Verifier()(noisy, no_data_ref, no_data_ref)            # unpaired: window mean within 3 %, grad norm report only
    assert v.valid, v.reasons
    assert status(v, 'same_data') == 'unchecked' and status(v, 'grad_norm_within_tolerance') == 'unchecked'
    paired = receipt('c3', t=0.9, gn=lambda s: 2.0)
    assert status(Verifier()(paired, ref, ref), 'grad_norm_within_tolerance') == 'fail'


def test_run_health(ref):
    nan = receipt('c', t=0.9, loss=lambda s: 'nan' if s == 4 else 2.0 - 0.01 * s)
    assert status(Verifier()(nan, ref, ref), 'no_errors') == 'fail'
    frozen = receipt('c2', t=0.9, loss=lambda s: 1.5 if s > 5 else 2.0 - 0.01 * s, gn=lambda s: 1.0 if s > 5 else 1.0 + 0.01 * s)
    v = Verifier(tolerance=0.5)(frozen, ref, ref)
    assert status(v, 'no_errors') == 'fail' and 'bitwise identical' in v.reasons[0]
    crash = receipt('c3', t=0.9, exit_code=1, completed=6)
    v = Verifier()(crash, ref, ref)
    assert not v.valid and status(v, 'completes') == 'fail'
    oom = receipt('c4', t=0.9, events={'oom': 2, 'samples': ['CUDA out of memory']})
    assert status(Verifier()(oom, ref, ref), 'no_errors') == 'fail'
    big = receipt('c5', t=0.9, mem=79.5)
    assert status(Verifier(memory_cap_gib=78)(big, ref, ref), 'memory_cap') == 'fail'


def test_lr_model_frozen_files(ref):
    assert status(Verifier()(receipt('c', t=0.9, lr=lambda s: 2e-4 * s), ref, ref), 'optimizer_steps') == 'fail'
    mref = receipt('r', meta={'trainable_params': 100})
    assert status(Verifier()(receipt('c2', t=0.9, meta={'trainable_params': 90}), mref, mref), 'same_model') == 'fail'
    touched = receipt('c3', t=0.9, frozen={'checked': True, 'n': 2, 'changed': ['loss.py'], 'missing': []})
    v = Verifier()(touched, ref, ref)
    assert status(v, 'frozen_files') == 'fail' and not v.valid


def test_timing_consistency():
    ref = receipt('r', t=1.0, rep=0.9)
    honest = receipt('c', t=0.8, rep=0.72)
    assert status(Verifier()(honest, ref, ref), 'timing_consistent') == 'pass'
    moved = receipt('c2', t=0.95, rep=0.6)              # the logged step time improved, the wall clock did not
    v = Verifier()(moved, ref, ref)
    assert status(v, 'timing_consistent') == 'fail' and not v.valid


def test_signature_and_schema(ref):
    unsigned = receipt('c', sign=False)
    assert not Verifier()(unsigned, ref, ref).valid
    forged = receipt('c2', t=0.1)
    forged['evidence']['step_time'] = 0.01
    assert not Verifier()(forged, ref, ref).valid


def test_disabled_items_are_off(ref):
    v = Verifier(disabled=('frozen_files',))(receipt('c', t=0.9, frozen={'checked': True, 'changed': ['x']}), ref, ref)
    assert v.valid and status(v, 'frozen_files') == 'off'


def test_promotion_items(ref):
    vf = Verifier()
    cand = receipt('c', t=0.8)
    ok = vf.promotion(cand, ref, rescore=(receipt('r2', t=1.0), receipt('c2', t=0.8)),
                      resume=receipt('res', t=0.8), tests=[{'cmd': 'pytest', 'exit_code': 0}])
    assert {r['id']: r['status'] for r in ok} == {'speed_reproduces': 'pass', 'resume_continues': 'pass',
                                                  'eval_matches': 'unchecked', 'user_tests': 'pass'}
    noise = vf.promotion(cand, ref, rescore=(receipt('r3', t=1.0), receipt('c3', t=0.99)), tests=[{'cmd': 't', 'exit_code': 1}])
    by = {r['id']: r for r in noise}
    assert by['speed_reproduces']['status'] == 'fail' and by['user_tests']['status'] == 'fail'
    assert by['resume_continues']['status'] == 'unchecked'
    assert C.blocking(noise) == [by['speed_reproduces'], by['user_tests']]


# ---------------------------------------------------------------- Codex review A1 (accepted findings)
def test_every_workload_dimension_is_compared(ref):
    sref = receipt('r')
    for row in sref['evidence']['per_step']:
        row['seq_len'] = 4096
    signing.sign(sref)
    cand = receipt('c', t=0.9)
    for row in cand['evidence']['per_step']:
        row['seq_len'] = 2048                      # same tokens per update, different sequences
    signing.sign(cand)
    v = Verifier()(cand, sref, sref)
    assert status(v, 'same_workload') == 'fail' and 'seq_len' in v.reasons[0]


def test_partial_loss_or_data_is_not_a_pass(ref):
    cand = receipt('c', t=0.9)
    cand['evidence']['per_step'][4].pop('loss')
    cand['evidence']['per_step'][5].pop('data')
    signing.sign(cand)
    v = Verifier()(cand, ref, ref)
    assert status(v, 'loss_within_tolerance') == 'fail' and status(v, 'same_data') == 'fail' and not v.valid


def test_nonfinite_grad_norm_and_first_update(ref):
    nan_gn = receipt('c', t=0.9, gn=lambda s: 'nan' if s == 3 else 1.0 + 0.01 * s)
    assert status(Verifier()(nan_gn, ref, ref), 'no_errors') == 'fail'
    late = receipt('c2', t=0.9)
    late['evidence']['per_step'][0].pop('grad_norm')
    signing.sign(late)
    assert status(Verifier()(late, ref, ref), 'grad_norm_within_tolerance') == 'fail'


def test_zero_step_time_is_no_measurement(ref):
    zero = receipt('c', t=0.0, rep=0.0)
    v = Verifier()(zero, ref, ref)
    assert not v.valid and v.step_time is None


def test_optimizer_step_counter(ref):
    r2 = receipt('r2')
    c2 = receipt('c2', t=0.9)
    for i, (x, y) in enumerate(zip(r2['evidence']['per_step'], c2['evidence']['per_step'])):
        x['optim_step'] = i + 1
        y['optim_step'] = i + 1 if i != 4 else i              # one update without an optimizer step
    signing.sign(r2)
    signing.sign(c2)
    v = Verifier()(c2, r2, r2)
    assert status(v, 'optimizer_steps') == 'fail' and 'optim_step' in v.reasons[0]


def test_resume_must_continue_after_the_checkpoint(ref):
    cand = receipt('c', t=0.8)
    restarted = receipt('res', t=0.8)                        # logs updates 1..10 again: the checkpoint was ignored
    items = {i['id']: i for i in Verifier().promotion(cand, ref, resume=restarted, save_at=5)}
    assert items['resume_continues']['status'] == 'fail' and 'not loaded' in items['resume_continues']['detail']
    cont = receipt('res2', t=0.8)
    cont['evidence']['per_step'] = [r for r in cont['evidence']['per_step'] if r['step'] > 5]
    signing.sign(cont)
    items = {i['id']: i for i in Verifier().promotion(cand, ref, resume=cont, save_at=5)}
    assert items['resume_continues']['status'] == 'pass'


# ---------------------------------------------------------------- review A2 (gpt-6.1-sol) and B2-5b
def test_static_invariant_is_compared_with_per_update_dimensions(ref):
    r = receipt('ref2', static={'seq_len': 4096})
    cand = receipt('c', t=0.8, static={'seq_len': 2048})          # same tokens per update, a declared invariant changed
    v = Verifier()(cand, r, r)
    assert status(v, 'same_workload') == 'fail' and 'seq_len' in v.reasons[0]


def test_missing_data_digest_fails_when_the_reference_has_one(ref):
    cand = receipt('c', t=0.8, data=False)
    v = Verifier()(cand, ref, ref)
    assert status(v, 'same_data') == 'fail' and not v.valid


def test_unmeasured_candidate_cannot_rest_on_its_own_report(ref):
    cand = receipt('c', t=0.5, rep=0.5, clock_step_time=None, clock_note='log lines arrive in bursts')
    v = Verifier()(cand, ref, ref)
    assert status(v, 'timing_consistent') == 'fail' and not v.valid


def test_resume_must_cover_every_update_after_the_checkpoint(ref):
    cand = receipt('c', t=0.8)
    short = receipt('res', t=0.8)
    short['evidence']['per_step'] = [r for r in short['evidence']['per_step'] if r['step'] == 6]
    signing.sign(short)
    items = {i['id']: i for i in Verifier().promotion(cand, ref, resume=short, save_at=5)}
    assert items['resume_continues']['status'] == 'fail' and 'missing' in items['resume_continues']['detail']


def test_optim_step_reaches_project_evidence():
    from infrarsi.project import signals as sig
    rows = sig.LogParser().parse([(0.1 * s, f'step {s} loss 2.0 optimizer_step {s} lr 0.001') for s in range(1, 5)])
    ev = sig.evidence(rows, steps=4, window=[2, 4], exit_code=0, timed_out=False, events={}, frozen={})
    assert [r.get('optim_step') for r in ev['per_step']] == [1, 2, 3, 4]


def test_eval_parser_ignores_auxiliary_losses():
    from infrarsi.project.runner import EVAL_RE
    lines = ['eval_loss: 1.20', 'aux_loss: 0.10', 'moe_aux_loss=0.05']
    hits = [(m.group('k').lower(), m.group('v')) for line in lines for m in EVAL_RE.finditer(line)]
    assert hits == [('eval_loss', '1.20')]


# ---------------------------------------------------------------- review R3a (gpt-6.1-sol)
def test_candidate_without_the_reference_optimizer_counter_fails():
    r = receipt('ref3', t=1.0, sign=False)
    for row in r['evidence']['per_step']:
        row['optim_step'] = row['step']
    signing.sign(r)
    v = Verifier()(receipt('c', t=0.8), r, r)                     # same learning rates, no optimizer-step counter
    assert status(v, 'optimizer_steps') == 'fail' and not v.valid


def test_failed_reference_rerun_is_not_a_noise_sample(tmp_path, ref):
    import json
    from infrarsi.project import Project
    from infrarsi.project.flow import promote
    (tmp_path / 'train.py').write_text('print(1)\n', encoding='utf-8')
    p = Project({'name': 'n', 'launch': {'command': '{python} train.py {args}'}, 'knobs': {},
                 'workload': {'steps': 10, 'window': [3, 10]}}, tmp_path / 'infrarsi.yaml')
    (tmp_path / 'rc').mkdir()
    (tmp_path / 'rc' / 'c1.json').write_text(json.dumps({'config': {}, 'overlay': {}}), encoding='utf-8')
    runs = iter([receipt('rerun', t=1.3, exit_code=1), receipt('c1', t=0.8)])

    class Runner:
        def run(self, *a, **kw):
            return next(runs)

        def run_tests(self, *a, **kw):
            return []
    trace = {'candidates': [{'id': 'c1', 'valid': True, 'step_time': 0.8}]}
    assert promote(p, None, Runner(), Verifier(), trace, ref, 'r', tmp_path / 'rc')['ref_offsets'] == []


# ---------------------------------------------------------------- measured noise (Fable H1 decision 3)
def test_promotion_margin_follows_the_measured_noise(ref):
    rerun, cand = receipt('rr', t=1.0), receipt('cc', t=0.99)            # a 1 % gain on a quiet node
    old = {i['id']: i for i in Verifier().promotion(cand, ref, rescore=(rerun, cand))}
    new = {i['id']: i for i in Verifier().promotion(cand, ref, rescore=(rerun, cand), margin=math.expm1(2 * 0.0014))}
    assert old['speed_reproduces']['status'] == 'fail'                   # the 3 % floor of the per-update rule
    assert new['speed_reproduces']['status'] == 'pass'                   # 2 sigma_pair = 0.28 %


def test_doctor_run_of_the_same_configuration_is_a_noise_pair(tmp_path, ref):
    import json
    from infrarsi.project import Project
    from infrarsi.project.flow import doctor_pair
    (tmp_path / 'train.py').write_text('print(1)\n', encoding='utf-8')
    p = Project({'name': 'n', 'launch': {'command': '{python} train.py {args}'}, 'knobs': {},
                 'workload': {'steps': 10, 'window': [3, 10]}}, tmp_path / 'infrarsi.yaml')
    d = tmp_path / '.infrarsi' / 'doctor' / 'doctor' / 'c0-doctor-1'
    d.mkdir(parents=True)
    doc = receipt('doc', t=1.002, sign=False)
    doc.update(config_digest='cfg', overlay_digest='ov')
    signing.sign(doc)
    (d / 'receipt.json').write_text(json.dumps(doc), encoding='utf-8')
    run_ref = receipt('ref4', t=1.0, sign=False)
    run_ref.update(config_digest='cfg', overlay_digest='ov')
    signing.sign(run_ref)
    assert doctor_pair(p, Verifier(), run_ref, 1.0) == pytest.approx(math.log(1.002))
    assert doctor_pair(p, Verifier(), dict(run_ref, host='another-node'), 1.0) is None   # another machine: no pair
    run_ref['config_digest'] = 'other'                                   # another configuration: no pair
    assert doctor_pair(p, Verifier(), run_ref, 1.0) is None


def test_promotion_margin_is_recomputed_with_the_promotion_runs(tmp_path, ref):
    import json
    from infrarsi.project import Project
    from infrarsi.project.flow import promote
    (tmp_path / 'train.py').write_text('print(1)\n', encoding='utf-8')
    p = Project({'name': 'n', 'launch': {'command': '{python} train.py {args}'}, 'knobs': {},
                 'workload': {'steps': 10, 'window': [3, 10]}}, tmp_path / 'infrarsi.yaml')
    (tmp_path / 'rc').mkdir()
    (tmp_path / 'rc' / 'c1.json').write_text(json.dumps({'config': {}, 'overlay': {}}), encoding='utf-8')
    runs = iter([receipt('rerun', t=1.0, split_half=0.001, rsd_window=0.0001),
                 receipt('c1', t=1 / 1.036, split_half=0.001, rsd_window=0.0001)])

    class Runner:
        def run(self, *a, **kw):
            return next(runs)

        def run_tests(self, *a, **kw):
            return []
    noise = {'mode': 'measured', 'sigma_pair': 0.02, 'pairs': [0.02], 'split_halves': [0.001] * 4}
    trace = {'candidates': [{'id': 'c1', 'valid': True, 'step_time': 1 / 1.036}], 'extra': {'noise': noise}}
    promo = promote(p, None, Runner(), Verifier(), trace, ref, 'r', tmp_path / 'rc')
    item = next(i for i in promo['tried'][0]['items'] if i['id'] == 'speed_reproduces')
    assert item['status'] == 'pass' and item['evidence']['margin'] == pytest.approx(math.expm1(2 * 0.02 / math.sqrt(2)))
