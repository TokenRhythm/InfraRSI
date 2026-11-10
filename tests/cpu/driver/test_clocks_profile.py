"""profile block, independent clocks (per step, training span, external wrapper clock) and data_manifest pinning."""
import json
import os

import pytest

from driver_helpers import (DUMMY_MOD, attempt_dir, fake_index, indexed_remote, make_request, rewrite_signals,
                            sim_launcher, simulated, validate_dir)

from infrarsi.harness import run_screen, validate as V
from infrarsi.harness.remote import LocalRemote


def names(rec, ok=None):
    return {c['name'] for c in rec['harness']['checks'] if ok is None or c['ok'] == ok}


# ---------------------------------------------------------------- profile

def test_profile_block_from_all_ranks(tmp_path):
    rec = simulated(tmp_path, make_request('D2'), step_time=lambda r, u: 0.5 + 0.01 * r)
    prof = rec['profile']
    assert set(prof) == {'whole_update_s', 'fwd_bwd_s', 'optim_s', 'data_s'}
    assert all(set(v) == {'window_mean_max_rank_s', 'window_mean_min_rank_s'} for v in prof.values())
    assert prof['whole_update_s']['window_mean_max_rank_s'] == pytest.approx(0.57)
    assert prof['whole_update_s']['window_mean_min_rank_s'] == pytest.approx(0.50)
    assert prof['fwd_bwd_s']['window_mean_max_rank_s'] == pytest.approx(0.8 * 0.57)
    assert prof['data_s']['window_mean_max_rank_s'] == pytest.approx(0.017)
    assert prof['data_s']['window_mean_min_rank_s'] == pytest.approx(0.010)
    assert V.verify_receipt_digest(rec)                                       # profile is inside the signed part


def test_profile_null_for_probes_and_incomplete_windows(tmp_path):
    assert simulated(tmp_path, make_request('D2', purpose='probe'))['profile'] is None
    stopped = simulated(tmp_path, make_request('D2'), stop_at=10, stop={'kind': 'request', 'reason': 't'})
    assert stopped['profile']['whole_update_s']['window_mean_max_rank_s'] is None


# ---------------------------------------------------------------- clocks

def test_honest_clocks_pass(tmp_path):
    rec = simulated(tmp_path, make_request('D2'), step_time=lambda r, u: 2.0 + 0.01 * r)
    assert rec['valid'], rec['invalid_reasons']
    assert {'clock_mismatch', 'clock_order', 'train_clock_mismatch', 'external_clock_mismatch'} <= names(rec, ok=True)
    assert 'clock_fields' not in names(rec)                                      # every rank-update row has UTC stamps
    wc = rec['wallclock']
    assert set(wc) == {'train_utc_s', 'process_train_s'}
    assert wc['train_utc_s'] == pytest.approx(30 * 2.07) and wc['process_train_s'] == pytest.approx(30 * 2.07)


def test_forged_perf_counter_rejected(tmp_path):
    """perf_counter scaled by 0.8 (every whole_update_s and phase looks 20 % faster) while UTC is untouched."""
    rec = simulated(tmp_path, make_request('D2'), step_time=lambda r, u: 2.0, perf_scale=0.8)
    assert not rec['valid']
    assert any(x.startswith('harness: clock_mismatch') for x in rec['invalid_reasons'])
    assert any(x.startswith('harness: train_clock_mismatch') for x in rec['invalid_reasons'])   # 60 s vs 48 s
    assert len(rec['harness']['clocks']['per_step']['outliers']) == 8                             # every rank
    from infrarsi.reward import evaluate
    assert not evaluate(rec)['valid']


def _validated(tmp_path, mutate, request=None, **sim):
    req = request or make_request('D2')
    out = attempt_dir(tmp_path, req, **sim)
    mutate(out)
    return validate_dir(out, req)


def _perf_skew(steps):
    """The perf-counter time of these steps disagrees with their UTC span (UTC order stays intact)."""
    return lambda r: dict(r, whole_update_s=r['whole_update_s'] * 0.6) if r['step'] in steps else r


def test_window_outliers_are_never_tolerated_on_screens(tmp_path):
    one = _validated(tmp_path, lambda out: rewrite_signals(out, 3, _perf_skew({5}), 8))        # outside 17-30
    assert one['valid'], one['invalid_reasons']
    assert one['harness']['clocks']['per_step']['outliers'] == {'3': [5]}
    inside = _validated(tmp_path, lambda out: rewrite_signals(out, 3, _perf_skew({20}), 8))     # inside the window
    assert not inside['valid'] and any('clock_mismatch' in x for x in inside['invalid_reasons'])
    two = _validated(tmp_path, lambda out: rewrite_signals(out, 3, _perf_skew({5, 9}), 8))      # two outside
    assert not two['valid'] and any('clock_mismatch' in x for x in two['invalid_reasons'])
    # a probe keeps the per-rank allowance of one step anywhere (no window)
    probe = _validated(tmp_path, lambda out: rewrite_signals(out, 3, _perf_skew({3}), 8), request=make_request('D2', purpose='probe'))
    assert probe['valid'], probe['invalid_reasons']


def test_missing_utc_on_some_rows_is_invalid(tmp_path):
    strip = lambda r: {k: v for k, v in r.items() if k not in ('utc_start_s', 'utc_end_s')} if r['step'] in (7, 8) else r  # noqa: E731
    rec = _validated(tmp_path, lambda out: rewrite_signals(out, 2, strip, 8))
    assert not rec['valid'] and any(x.startswith('harness: clock_fields') for x in rec['invalid_reasons'])
    assert rec['harness']['clocks']['per_step']['missing'] == {'2': [7, 8]}


def test_missing_utc_fields_are_invalid(tmp_path):
    """REQUIRE_UTC_FIELDS: the training process stamps every row (T-train, runtime/signals.py)."""
    assert V.REQUIRE_UTC_FIELDS is True
    rec = simulated(tmp_path, make_request('D2'), clocks=False)
    assert not rec['valid'] and any(x.startswith('harness: clock_fields') for x in rec['invalid_reasons'])
    assert any(x.startswith('harness: train_clock') for x in rec['invalid_reasons'])     # no UTC span on a screen
    assert rec['wallclock']['train_utc_s'] is None and rec['wallclock']['process_train_s'] is not None


def test_utc_stamps_must_be_positive_and_ordered(tmp_path):
    back = lambda r: dict(r, utc_start_s=r['utc_start_s'] - 5.0) if r['step'] == 12 else r     # noqa: E731 - overlaps step 11
    rec = _validated(tmp_path, lambda out: rewrite_signals(out, 4, back, 8))
    assert not rec['valid'] and rec['harness']['clocks']['per_step']['disordered'] == {'4': [12]}
    inverted = lambda r: dict(r, utc_end_s=r['utc_start_s'] - 0.1) if r['step'] == 3 else r   # noqa: E731
    rec = _validated(tmp_path, lambda out: rewrite_signals(out, 1, inverted, 8))
    assert not rec['valid'] and any('clock_order' in x for x in rec['invalid_reasons'])
    zero = lambda r: dict(r, utc_start_s=0.0, utc_end_s=r['whole_update_s']) if r['step'] == 1 else r   # noqa: E731
    rec = _validated(tmp_path, lambda out: rewrite_signals(out, 0, zero, 8))
    assert not rec['valid'] and '0' in rec['harness']['clocks']['per_step']['disordered']


def test_non_positive_elapsed_times_rejected(tmp_path):
    rec = _validated(tmp_path, lambda out: rewrite_signals(out, 5, lambda r: dict(r, whole_update_s=0.0) if r['step'] == 2 else r, 8))
    assert not rec['valid'] and any('not positive' in x for x in rec['invalid_reasons'])
    neg = _validated(tmp_path, lambda out: rewrite_signals(out, 5, lambda r: dict(r, inter_step_s=-0.5) if r['step'] == 25 else r, 8))
    assert not neg['valid'] and any('InterStep not derivable' in x for x in neg['invalid_reasons'])


def test_inter_step_is_only_derived_from_signals(tmp_path):
    """A rank without inter_step_s inside the window: no InterStep, an invalid screen, and no fallback to the
    trainer's summary in the reward engine."""
    drop = lambda r: {k: v for k, v in r.items() if k != 'inter_step_s'} if r['step'] in (18, 19) else r  # noqa: E731
    rec = _validated(tmp_path, lambda out: rewrite_signals(out, 6, drop, 8))
    assert rec['step_time']['inter_step_mean_s'] is None and rec['harness']['reported']['step_time']['inter_step_mean_s']
    assert not rec['valid'] and any('InterStep not derivable' in x for x in rec['invalid_reasons'])
    from infrarsi.reward.evaluate import inter_step
    assert inter_step(rec) is None


def test_external_clock_against_the_wrapper(tmp_path):
    # wrapper saw an hour, the process claims ~20 s and nothing explains the gap
    bad = simulated(tmp_path, make_request('D2'), exit_fields={'wall_s': 3600.0})
    assert not bad['valid'] and any(x.startswith('harness: external_clock_mismatch') for x in bad['invalid_reasons'])

    def explained(out):                     # the same hour, but almost all of it is teardown after the receipt
        m = os.stat(out / 'receipt.json').st_mtime
        p = out / 'exit_node0.json'
        ex = json.loads(p.read_text())
        ex.update(wall_s=3600.0, wrapper_start_utc_s=m - 30.0, child_start_utc_s=m - 28.0, first_heartbeat_utc_s=m - 25.0,
                  child_exit_utc_s=m + 3560.0, wrapper_end_utc_s=m + 3561.0)
        p.write_text(json.dumps(ex))
    ok = _validated(tmp_path, explained)
    ext = ok['harness']['clocks']['external']
    assert ok['valid'], ok['invalid_reasons']
    assert ext['components']['teardown_after_receipt_s'] == pytest.approx(3560.0)
    assert ext['components']['startup_before_heartbeat_s'] == pytest.approx(3.0)


def test_missing_external_or_training_clock_fails_screens_only(tmp_path):
    def no_wall(out):
        p = out / 'exit_node0.json'
        ex = json.loads(p.read_text())
        ex.pop('wall_s')
        p.write_text(json.dumps(ex))
    screen = _validated(tmp_path, no_wall)
    assert not screen['valid'] and any(x.startswith('harness: external_clock:') for x in screen['invalid_reasons'])
    probe = _validated(tmp_path, no_wall, request=make_request('D2', purpose='probe'))
    assert probe['valid'] and any(w.startswith('external_clock') for w in probe['harness']['warnings'])
    no_train = simulated(tmp_path, make_request('D2', plan='B', purpose='score'), phases_override={'train': None})
    assert not no_train['valid'] and any(x.startswith('harness: train_clock:') for x in no_train['invalid_reasons'])


def test_scaled_phases_caught_externally(tmp_path):
    """Both process clocks forged together (perf_counter and time.time run at 0.8x, so the per-step and training-span
    checks agree with each other); the wrapper's own clock saw the real 600 s and still disagrees."""
    rec = simulated(tmp_path, make_request('D2'), step_time=lambda r, u: 16.0, exit_fields={'wall_s': 610.0})
    assert {'clock_mismatch', 'train_clock_mismatch'} <= names(rec, ok=True)
    assert rec['wallclock']['process_train_s'] == pytest.approx(480.0)
    assert [x for x in rec['invalid_reasons'] if 'clock' in x] == [
        x for x in rec['invalid_reasons'] if x.startswith('harness: external_clock_mismatch')] != []


# ---------------------------------------------------------------- data_manifest pinned before upload

def test_request_carries_the_validated_manifest(tmp_path, fsroot):
    r = LocalRemote(fsroot)
    facts = fake_index(r)['D2']
    r.launcher = sim_launcher(lambda req, out: DUMMY_MOD.simulate_attempt(req, out, data=facts[req['input_plan']]))
    req = make_request('D2', nodes=['WINH2'])
    rec = run_screen(req, remote=r, poll_s=0.01, local_dir=None, log=lambda m: None)
    uploaded = json.loads(r.local(f'{req.out_dir}/request.json').read_text())
    assert uploaded['data_manifest'] == facts['A']['manifest'] == req.data_manifest
    assert rec['valid'] and {'data.manifest_used', 'data.manifest_digest', 'data.request_manifest'} <= names(rec, ok=True)
    # a trainer that read another manifest is caught
    other = dict(facts['A'], manifest='/data/infrarsi/datasets/packed/OTHER/manifest.json')
    r.launcher = sim_launcher(lambda req, out: DUMMY_MOD.simulate_attempt(req, out, data=other))
    bad = run_screen(make_request('D2', nodes=['WINH2']), remote=r, poll_s=0.01, local_dir=None, log=lambda m: None)
    assert not bad['valid'] and any('data.manifest_used' in x for x in bad['invalid_reasons'])


def test_manifest_conflicts_refuse_to_launch(tmp_path, fsroot):
    r = indexed_remote(fsroot, launcher=sim_launcher())
    with pytest.raises(ValueError):
        run_screen(make_request('D2', nodes=['WINH2'], data_manifest='/data/x/manifest.json'), remote=r, log=lambda m: None)
    m = next(r.local('/data/infrarsi/datasets/packed').glob('FAKE-D2__*/manifest.json'))
    m.write_text(m.read_text() + ' ')                                  # sha256 no longer matches INDEX.json
    with pytest.raises(RuntimeError):
        run_screen(make_request('D2', nodes=['WINH2']), remote=r, log=lambda m: None)
    assert not [c for c in r.calls if c[1] in ('launch_wrapper', 'acquire_leases')]


def test_synthetic_task_leaves_manifest_unset(fsroot):
    from infrarsi.harness.runner import resolve_data
    req = make_request('S1')
    req.task = dict(req.task, data=dict(req.task['data'], synthetic=True))
    assert resolve_data(LocalRemote(fsroot), ['WINH0'], req) == {'skipped': 'synthetic data'} and req.data_manifest is None
    req.data_manifest = '/data/infrarsi/datasets/packed/x/manifest.json'
    with pytest.raises(ValueError):
        resolve_data(LocalRemote(fsroot), ['WINH0'], req)
