"""C4 - the verifier as a table: synthetic receipts, one perturbation each, the item and status it must produce; the
required / disabled semantics; the tolerance boundaries; the short-run and repeat rules; the promotion items."""
import json
import math

import pytest

from infrarsi.harness import signing
from infrarsi.verify import SCHEMA, Verifier
from infrarsi.verify import checklist as C

STEPS = 12
WINDOW = [5, 12]


def rows(*, steps=STEPS, loss=lambda s: 2.0 - 0.01 * s, gn=lambda s: 1.0 + 0.01 * s, lr=lambda s: 1e-4 * s, tokens=4096,
         data=True, step_time=None, optim=None, seq_len=None, mem=None):
    out = []
    for s in range(1, steps + 1):
        r = {'step': s, 'loss': loss(s), 'grad_norm': gn(s), 'lr': lr(s), 't': 10.0 + s}
        if tokens is not None:
            r['tokens'] = tokens(s) if callable(tokens) else tokens
        if data:
            r['data'] = data(s) if callable(data) else f'batch{s}'
        if step_time is not None:
            r['step_time'] = step_time
        if optim is not None:
            r['optim_step'] = optim(s)
        if seq_len is not None:
            r['seq_len'] = seq_len
        if mem is not None:
            r['mem_gib'] = mem
        out.append(r)
    return out


def receipt(attempt='a', *, per_step=None, t=1.0, rep=None, exit_code=0, completed=None, timed_out=False, events=None,
            frozen=None, meta=None, mem=None, static=None, stop_after=None, planned_stop=False, window=None, sign=True,
            cap=None, **extra):
    per = rows() if per_step is None else per_step
    done = completed if completed is not None else (per[-1]['step'] if per else 0)
    ev = {'steps': STEPS, 'steps_completed': done, 'exit_code': exit_code, 'timed_out': timed_out, 'per_step': per,
          'window': window or WINDOW, 'step_time': t, 'clock_step_time': t, 'reported_step_time': rep, 'rsd_window': 0.005,
          'events': {'oom': 0, 'collective': 0, 'traceback': 0, 'samples': [], 'nan_inf_steps': [], **(events or {})},
          'meta': meta or {}, 'memory_peak_gib': mem, 'memory_cap_gib': cap,
          'frozen': {'checked': True, 'n': 1, 'changed': [], 'missing': []} if frozen is None else frozen,
          'static_workload': static or {}, 'stop_after': stop_after, 'planned_stop': planned_stop, **extra}
    r = {'schema': SCHEMA, 'attempt_id': attempt, 'evidence': ev}
    return signing.sign(r) if sign else r


def status(v) -> dict:
    return {c['id']: c for c in v.checks}


@pytest.fixture
def ref():
    return receipt('ref')


# ---------------------------------------------------------------- one perturbation, one item
CASES = {
    'faster_clean': (dict(t=0.8), True, {}),
    'loss_2pct': (dict(per_step=rows(loss=lambda s: (2.0 - 0.01 * s) * 1.02)), True, {'loss_within_tolerance': 'pass'}),
    'loss_just_under_3pct': (dict(per_step=rows(loss=lambda s: (2.0 - 0.01 * s) * 1.0299)), True, {'loss_within_tolerance': 'pass'}),
    'loss_just_over_3pct': (dict(per_step=rows(loss=lambda s: (2.0 - 0.01 * s) * 1.0300001)), False, {'loss_within_tolerance': 'fail'}),
    'loss_off_at_one_update': (dict(per_step=rows(loss=lambda s: (2.0 - 0.01 * s) * (1.5 if s == 7 else 1.0))), False,
                               {'loss_within_tolerance': 'fail'}),                 # paired: per update, not the mean
    'model_digest_differs': (dict(meta={'model_digest': 'b', 'trainable_params': 10}), False, {'same_model': 'fail'},
                             dict(meta={'model_digest': 'a', 'trainable_params': 10})),
    'trainable_params_differ': (dict(meta={'model_digest': 'a', 'trainable_params': 99}), False, {'same_model': 'fail'},
                                dict(meta={'model_digest': 'a', 'trainable_params': 10})),
    'model_equal': (dict(meta={'model_digest': 'a', 'trainable_params': 10}), True, {'same_model': 'pass'},
                    dict(meta={'model_digest': 'a', 'trainable_params': 10})),
    'optim_step_counter_differs': (dict(per_step=rows(optim=lambda s: s + 1)), False, {'optimizer_steps': 'fail'},
                                   dict(per_step=rows(optim=lambda s: s))),
    'optim_step_counter_missing': (dict(), False, {'optimizer_steps': 'fail'}, dict(per_step=rows(optim=lambda s: s))),
    'static_invariant_changed': (dict(static={'global_batch': 16}), False, {'same_workload': 'fail'}, dict(static={'global_batch': 8})),
    'static_invariant_equal': (dict(static={'global_batch': 8}), True, {'same_workload': 'pass'}, dict(static={'global_batch': 8})),
    'loss_nan_mid_run': (dict(per_step=rows(loss=lambda s: 'nan' if s == 8 else 2.0 - 0.01 * s),
                              events={'nan_inf_steps': [8]}), False, {'no_errors': 'fail', 'loss_within_tolerance': 'fail'}),
    'loss_missing_at_an_update': (dict(per_step=[r for r in rows() if r['step'] != 6]), False, {'loss_within_tolerance': 'fail'}),
    'grad_norm_first_update_off': (dict(per_step=rows(gn=lambda s: (1.0 + 0.01 * s) * (1.1 if s == 1 else 1.0))), False,
                                   {'grad_norm_within_tolerance': 'fail'}),
    'grad_norm_later_update_off': (dict(per_step=rows(gn=lambda s: (1.0 + 0.01 * s) * (1.1 if s == 9 else 1.0))), True,
                                   {'grad_norm_within_tolerance': 'pass'}),        # only the first update is held
    'tokens_halved': (dict(per_step=rows(tokens=2048)), False, {'same_workload': 'fail'}),
    'tokens_missing': (dict(per_step=rows(tokens=None)), False, {'same_workload': 'fail'}),
    'seq_len_changed': (dict(per_step=rows(seq_len=256)), False, {'same_workload': 'fail'}, dict(per_step=rows(seq_len=512))),
    'seq_len_not_in_reference': (dict(per_step=rows(seq_len=256)), True, {'same_workload': 'pass'}),   # nothing to compare
    'data_order_changed': (dict(per_step=rows(data=lambda s: f'other{s}')), False, {'same_data': 'fail'}),
    'data_digest_missing': (dict(per_step=rows(data=False)), False, {'same_data': 'fail'}),
    'lr_schedule_changed': (dict(per_step=rows(lr=lambda s: 2e-4 * s)), False, {'optimizer_steps': 'fail'}),
    'lr_equal_within_1e-6': (dict(per_step=rows(lr=lambda s: 1e-4 * s * (1 + 5e-7))), True, {'optimizer_steps': 'unchecked'}),  # lr only: a proxy
    'optim_step_counter_equal': (dict(per_step=rows(optim=lambda s: s)), True, {'optimizer_steps': 'pass'},
                                 dict(per_step=rows(optim=lambda s: s))),
    'frozen_file_changed': (dict(frozen={'checked': True, 'n': 1, 'changed': ['loss.py'], 'missing': []}), False, {'frozen_files': 'fail'}),
    'frozen_file_missing': (dict(frozen={'checked': True, 'n': 1, 'changed': [], 'missing': ['loss.py']}), False, {'frozen_files': 'fail'}),
    'exit_code_1': (dict(exit_code=1), False, {'completes': 'fail'}),
    'timed_out': (dict(timed_out=True, completed=4), False, {'completes': 'fail'}),
    'fewer_updates_exit_0': (dict(completed=9, per_step=rows(steps=9)), False, {'completes': 'fail'}),
    'oom_line': (dict(events={'oom': 1}), False, {'no_errors': 'fail'}),
    'nccl_line_mid_run_exit_1': (dict(events={'collective': 2}, exit_code=1), False, {'completes': 'fail', 'no_errors': 'fail'}),
    'nccl_line_at_a_clean_end': (dict(events={'collective': 2}), True, {'no_errors': 'pass'}),
    'traceback_line': (dict(events={'traceback': 1}), False, {'no_errors': 'fail'}),
    'frozen_scalars': (dict(per_step=rows(loss=lambda s: 2.0 if s in (6, 7) else 2.0 - 0.01 * s,
                                          gn=lambda s: 1.0 if s in (6, 7) else 1.0 + 0.01 * s)), False, {'no_errors': 'fail'}),
    'memory_over_cap': (dict(mem=20.0, cap=16.0), False, {'memory_cap': 'fail'}),
    'memory_within_cap': (dict(mem=15.0, cap=16.0), True, {'memory_cap': 'pass'}),
    'reported_time_disagrees': (dict(t=0.8, rep=0.4), False, {'timing_consistent': 'fail'}),
    'reported_time_agrees': (dict(t=0.8, rep=0.81), True, {'timing_consistent': 'pass'}),
    'no_clock_but_reference_had_one': (dict(clock_step_time=None, rep=0.5, t=0.5), False, {'timing_consistent': 'fail'}),
}


@pytest.mark.parametrize('name', sorted(CASES))
def test_one_perturbation(name):
    kw, valid, items = CASES[name][:3]
    ref_kw = CASES[name][3] if len(CASES[name]) > 3 else ({'rep': 1.0} if 'reported' in name or 'clock' in name else {})
    ref = receipt('ref', **ref_kw)
    v = Verifier()(receipt('c', **kw), ref, ref)
    assert v.valid == valid, (name, v.reasons)
    st = status(v)
    for cid, s in items.items():
        assert st[cid]['status'] == s, (name, cid, st[cid])


# ---------------------------------------------------------------- required / disabled / unchecked
def test_unchecked_vetoes_only_when_required(ref):
    bare = receipt('c', per_step=rows(tokens=None, data=False))
    v = Verifier()(bare, ref, receipt('ref', per_step=rows(tokens=None, data=False)))
    assert status(v)['same_workload']['status'] == 'unchecked' and not v.valid       # same_workload is required
    v2 = Verifier(required=('completes', 'loss_within_tolerance'))(bare, ref, receipt('ref', per_step=rows(tokens=None, data=False)))
    assert v2.valid and status(v2)['same_workload']['status'] == 'unchecked'


def test_disabled_item_is_off_and_never_vetoes(ref):
    v = Verifier(disabled=('loss_within_tolerance',))(receipt('c', per_step=rows(loss=lambda s: 5.0)), ref, ref)
    assert status(v)['loss_within_tolerance']['status'] == 'off' and v.valid


def test_a_failed_optional_item_still_vetoes(ref):
    v = Verifier()(receipt('c', events={'traceback': 1}), ref, ref)
    assert not v.valid and 'no_errors' not in Verifier().required


def test_unsigned_or_tampered_receipts_are_invalid(ref):
    assert not Verifier()(receipt('c', sign=False), ref, ref).valid
    r = receipt('c')
    r['evidence']['step_time'] = 0.1
    assert not Verifier()(r, ref, ref).valid
    assert Verifier(require_signature=False)(r, ref, ref).valid
    assert not Verifier()({'schema': 'other'}, ref, ref).valid
    assert not Verifier()('not a receipt', ref, ref).valid


def test_dropping_the_model_metadata_the_reference_had_is_a_failure():
    """F-SYS-15 (fixed): as for same_data, a candidate that stops reporting what the reference reported fails."""
    ref = receipt('ref', meta={'model_digest': 'a', 'trainable_params': 10})
    v = Verifier()(receipt('c', meta={}), ref, ref)
    assert status(v)['same_model']['status'] == 'fail' and 'not reported' in status(v)['same_model']['detail']
    v = Verifier()(receipt('c', meta={'model_digest': 'a'}), ref, ref)
    assert status(v)['same_model']['status'] == 'fail' and 'trainable_params' in status(v)['same_model']['detail']


def test_a_collective_error_in_the_middle_of_a_run_fails_even_with_exit_0():
    """F-SYS-16 (fixed): the end-of-run exemption (DF18) applies only to lines after the last update."""
    ref = receipt('ref')
    mid = receipt('c', events={'collective': 1, 'collective_at_end': 0})
    assert status(Verifier()(mid, ref, ref))['no_errors']['status'] == 'fail'
    end = receipt('d', events={'collective': 2, 'collective_at_end': 2})
    assert status(Verifier()(end, ref, ref))['no_errors']['status'] == 'pass'
    mixed = receipt('e', events={'collective': 2, 'collective_at_end': 1})
    assert status(Verifier()(mixed, ref, ref))['no_errors']['status'] == 'fail'


def test_tolerance_is_bounded():
    for bad in (-0.1, 1.0, 1.5, 'x', None):
        with pytest.raises(ValueError):
            Verifier(tolerance=bad)
    assert Verifier(tolerance=0).tolerance == 0.0


# ---------------------------------------------------------------- unpaired loss: window mean, short runs, repeats
def test_unpaired_loss_uses_the_window_mean():
    r = receipt('ref', per_step=rows(data=False))
    spike = receipt('c', per_step=rows(data=False, loss=lambda s: (2.0 - 0.01 * s) * (1.15 if s == 7 else 1.0)))   # 1.9 % on the mean
    v = Verifier()(spike, r, r)
    item = status(v)['loss_within_tolerance']
    assert item['status'] == 'pass' and 'window-mean' in item['detail'] and item['evidence']['paired'] is False


def test_a_short_run_is_held_to_the_noise_of_its_few_updates():
    noisy = lambda s: 2.0 * (1 + 0.3 * (-1) ** s)                     # a loss that swings 30 % between updates
    r = receipt('ref', per_step=rows(data=False, loss=noisy))
    probe = receipt('p', per_step=rows(data=False, loss=lambda s: noisy(s) * 1.1, steps=2), completed=2, stop_after=2, planned_stop=True)
    v = Verifier()(probe, r, r)
    item = status(v)['loss_within_tolerance']
    assert item['status'] == 'pass' and item['evidence']['tolerance'] > 0.03 and 'held to' in item['detail']
    quiet = receipt('ref2', per_step=rows(data=False))
    probe2 = receipt('p2', per_step=rows(data=False, loss=lambda s: (2.0 - 0.01 * s) * 1.1, steps=2), completed=2, stop_after=2, planned_stop=True)
    assert status(Verifier()(probe2, quiet, quiet))['loss_within_tolerance']['status'] == 'fail'   # a quiet loss: 3 %


def test_a_planned_stop_completes_and_an_abort_does_not():
    r = receipt('ref')
    planned = receipt('p', per_step=rows(steps=3), completed=3, stop_after=3, planned_stop=True, exit_code=-15)
    assert status(Verifier()(planned, r, r))['completes']['status'] == 'pass'
    aborted = receipt('a', per_step=rows(steps=3), completed=3, stop_after=None, planned_stop=False, exit_code=-15, stopped_early=True)
    assert status(Verifier()(aborted, r, r))['completes']['status'] == 'fail'


def test_the_repeat_of_the_reference_decides_whether_draws_vary():
    noisy = lambda s: 2.0 * (1 + 0.3 * (-1) ** s)
    r = receipt('ref', per_step=rows(data=False, loss=noisy))
    same_draws = receipt('rep', per_step=rows(data=False, loss=noisy))
    other_draws = receipt('rep2', per_step=rows(data=False, loss=lambda s: 2.0 * (1 + 0.3 * (-1) ** (s + 1))))
    v = Verifier(repeat=same_draws['evidence'])
    lr = v.loss_repeat(r['evidence'])
    assert lr['draws_vary'] is False and lr['repeat'] < 0.001
    v2 = Verifier(repeat=other_draws['evidence'])
    lr2 = v2.loss_repeat(r['evidence'])
    assert lr2['draws_vary'] is True and lr2['tolerance'] > 0.03
    full = receipt('c', per_step=rows(data=False, loss=lambda s: noisy(s) * 1.06))      # 6 % off on the mean
    assert status(v(full, r, r))['loss_within_tolerance']['status'] == 'fail'           # same draws: the manifest 3 %
    assert status(v2(full, r, r))['loss_within_tolerance']['status'] == 'pass'          # draws vary: 3 x the noise


def test_the_feedback_names_failed_items_with_hints(ref):
    v = Verifier()(receipt('c', per_step=rows(tokens=2048)), ref, ref)
    assert 'same_workload' in v.feedback and 'global batch' in v.feedback
    assert v.coverage['checked'] and 'same_model' in v.coverage['unchecked']


# ---------------------------------------------------------------- promotion items
def test_speed_reproduces_margin_boundary(ref):
    v = Verifier()
    cand = receipt('c', t=0.9)
    fresh_ref, fresh_c = receipt('r2', t=1.0), receipt('c2', t=0.9)
    item = {i['id']: i for i in v.promotion(cand, ref, rescore=(fresh_ref, fresh_c), margin=0.05)}['speed_reproduces']
    assert item['status'] == 'pass' and math.isclose(item['evidence']['speedup'], 1 / 0.9)
    item = {i['id']: i for i in v.promotion(cand, ref, rescore=(fresh_ref, fresh_c), margin=0.12)}['speed_reproduces']
    assert item['status'] == 'fail' and 'noise margin' in item['detail']
    item = {i['id']: i for i in v.promotion(cand, ref, rescore=(fresh_ref, receipt('c3', t=0.9, exit_code=1)))}['speed_reproduces']
    assert item['status'] == 'fail' and 're-run not valid' in item['detail']
    assert {i['id']: i for i in v.promotion(cand, ref)}['speed_reproduces']['status'] == 'unchecked'


def test_resume_continues_detects_a_restart_and_a_divergence(ref):
    v = Verifier()
    cand = receipt('c', t=0.9)
    resumed = receipt('rs', per_step=[r for r in rows() if r['step'] > 6])
    ok = {i['id']: i for i in v.promotion(cand, ref, resume=resumed, save_at=6)}['resume_continues']
    assert ok['status'] == 'pass'
    restarted = receipt('rs2', per_step=rows())                      # started again at update 1
    bad = {i['id']: i for i in v.promotion(cand, ref, resume=restarted, save_at=6)}['resume_continues']
    assert bad['status'] == 'fail' and 'started again' in bad['detail']
    diverged = receipt('rs3', per_step=[dict(r, loss=r['loss'] * 1.2) for r in rows() if r['step'] > 6])
    assert {i['id']: i for i in v.promotion(cand, ref, resume=diverged, save_at=6)}['resume_continues']['status'] == 'fail'


def test_eval_and_user_tests_items(ref):
    v = Verifier()
    cand = receipt('c', t=0.9)
    items = {i['id']: i for i in v.promotion(cand, ref, evals=({'exit_code': 0, 'value': 2.0}, {'exit_code': 0, 'value': 2.05}),
                                             tests=[{'cmd': 'x', 'exit_code': 0, 'tail': ''}])}
    assert items['eval_matches']['status'] == 'pass' and items['user_tests']['status'] == 'pass'
    items = {i['id']: i for i in v.promotion(cand, ref, evals=({'exit_code': 0, 'value': 2.0}, {'exit_code': 0, 'value': 2.2}),
                                             tests=[{'cmd': 'x', 'exit_code': 1, 'tail': 'boom'}])}
    assert items['eval_matches']['status'] == 'fail' and items['user_tests']['status'] == 'fail' and 'boom' in items['user_tests']['detail']
    items = {i['id']: i for i in v.promotion(cand, ref, evals=({'exit_code': 1, 'value': None}, {'exit_code': 0, 'value': 2.0}))}
    assert items['eval_matches']['status'] == 'fail' and 'reference' in items['eval_matches']['detail']


def test_every_checklist_item_appears_once_in_a_screen_verdict(ref):
    v = Verifier()(receipt('c'), ref, ref)
    ids = [c['id'] for c in v.checks]
    assert len(ids) == len(set(ids)) and set(ids) == set(C.SCREEN_IDS)
    assert json.dumps(v.to_dict())                                    # serialisable
