from types import SimpleNamespace

import pytest

from infrarsi.contracts import file_digest
from infrarsi.runtime import overlay_loader as l1


def test_builtin_overlays_are_noops():
    ov = l1.load(None)
    assert ov.source == 'builtin' and set(ov.files) == {f'{m}.py' for m in l1.HOOK_MODULES}
    ctx = SimpleNamespace()
    assert ov.overrides('comm_groups', 'parallel_init_kwargs', ctx) == {}
    assert ov.overrides('recompute_policy', 'model_config_overrides', ctx) == {}
    assert ov.call('schedule', 'microbatch_order', ctx, 1, [[1], [2]]) == [[1], [2]]
    assert ov.call('schedule', 'forward_backward', ctx, lambda **kw: kw['x'], x=7) == 7
    assert ov.digest == l1.load(None).digest


def test_request_overlay_loaded_after_sha_check(tmp_path):
    src = "def microbatch_order(ctx, step, microbatches):\n    return list(reversed(microbatches))\n"
    (tmp_path / 'schedule.py').write_text(src)
    spec = {'dir': str(tmp_path), 'files': {'schedule.py': file_digest(tmp_path / 'schedule.py')}}
    ov = l1.load(spec)
    assert ov.source == 'request' and ov.digest != l1.load(None).digest
    assert ov.call('schedule', 'microbatch_order', None, 1, [[1], [2]]) == [[2], [1]]
    assert ov.call('schedule', 'forward_backward', None, lambda **kw: 3) == 3      # missing hook -> default
    spec['files']['schedule.py'] = '0' * 64
    with pytest.raises(l1.OverlayViolation, match='sha256'):
        l1.load(spec)


def test_overlay_cannot_leave_l1_zone(tmp_path):
    (tmp_path / 'timers.py').write_text('x = 1\n')
    with pytest.raises(l1.OverlayViolation, match='outside the L1 zone'):
        l1.load({'dir': str(tmp_path), 'files': {'timers.py': file_digest(tmp_path / 'timers.py')}})
    (tmp_path / 'recompute_policy.py').write_text(
        "def model_config_overrides(ctx):\n    return {'calculate_per_token_loss': False, 'recompute_num_layers': 2}\n")
    ov = l1.load({'dir': str(tmp_path), 'files': {'recompute_policy.py': file_digest(tmp_path / 'recompute_policy.py')}})
    with pytest.raises(l1.OverlayViolation, match='calculate_per_token_loss'):
        ov.overrides('recompute_policy', 'model_config_overrides', None)


def test_overlay_digest_matches_patch_guard():
    pg = pytest.importorskip('infrarsi.actions.patch_guard')
    ov = l1.load(None)
    assert ov.digest == pg.overlay_digest(ov.files)


def test_loader_lives_in_the_frozen_zone():
    from infrarsi.runtime import invariants as inv
    rels = [r for r, _ in inv.frozen_zone_files(inv.Path(inv.__file__).resolve().parents[1])]
    assert 'runtime/overlay_loader.py' in rels


FROZEN = dict(forward_step_func=len, data_iterator=iter(()), model=['m'], num_microbatches=2, seq_length=8,
              micro_batch_size=1, forward_only=False)


def _with_schedule(fn):
    ov = l1.load(None)
    ov.modules = dict(ov.modules, schedule=SimpleNamespace(forward_backward=fn))
    return ov


def test_schedule_hook_runs_the_frozen_schedule_once():
    calls = []

    def fb(**kw):
        calls.append(kw)
        return ['store']
    assert l1.load(None).forward_backward(None, fb, FROZEN) == ['store'] and calls == [FROZEN]

    def wrapper(ctx, run, **kw):          # allowed: work around the schedule, call it once, return its result
        return run(**kw)
    calls.clear()
    assert _with_schedule(wrapper).forward_backward(None, fb, FROZEN) == ['store'] and len(calls) == 1


@pytest.mark.parametrize('hook, match', [
    (lambda ctx, run, **kw: (run(**kw), run(**kw))[0], 'twice'),
    (lambda ctx, run, **kw: None, '0 times'),
    (lambda ctx, run, **kw: run(**dict(kw, num_microbatches=1)), 'altered'),
    (lambda ctx, run, **kw: list(run(**kw)), 'replaced'),
])
def test_schedule_hook_violations(hook, match):
    calls = []

    def fb(**kw):
        calls.append(1)
        return ['store']
    with pytest.raises(l1.OverlayViolation, match=match):
        _with_schedule(hook).forward_backward(None, fb, FROZEN)
    assert len(calls) <= 1                  # a second schedule execution is refused before it runs


def test_schedule_guard_runs_inside_run_and_a_threaded_second_call_is_refused():
    import threading
    seen, calls = [], []
    ov = _with_schedule(lambda ctx, run, **kw: run(**kw))
    ov.guard = seen.append

    def fb(**kw):
        calls.append(1)
        return ['store']
    assert ov.forward_backward(None, fb, FROZEN) == ['store']
    assert seen == ['before schedule.forward_backward', 'before the schedule', 'after the schedule',
                    'after schedule.forward_backward']
    results, inside, gate = [], threading.Event(), threading.Event()

    def slow(**kw):                               # the first schedule is still running when the second call comes
        inside.set()
        gate.wait(5)
        calls.append(2)
        return ['slow']

    def threaded(ctx, run, **kw):
        th = threading.Thread(target=lambda: results.append(_call(run, kw)))
        th.start()
        inside.wait(5)
        try:
            return run(**kw)
        finally:
            gate.set()
            th.join()
    calls.clear()
    with pytest.raises(l1.OverlayViolation, match='twice'):
        _with_schedule(threaded).forward_backward(None, slow, FROZEN)
    assert calls == [2] and results == [['slow']]


def _call(run, kw):
    try:
        return run(**kw)
    except l1.OverlayViolation as e:
        return e


def test_effective_digest_merges_request_files_over_defaults(tmp_path):
    from infrarsi.contracts import overlay_digest
    builtin = l1.load(None)
    assert l1.effective_files(None) == builtin.files and l1.effective_digest(None) == builtin.digest
    (tmp_path / 'schedule.py').write_text('def forward_backward(ctx, run, **kw):\n    return run(**kw)\n')
    spec = {'dir': str(tmp_path), 'files': {'schedule.py': file_digest(tmp_path / 'schedule.py')}}
    files = l1.effective_files(spec)
    assert set(files) == {f'{m}.py' for m in l1.HOOK_MODULES} and files['schedule.py'] == spec['files']['schedule.py']
    assert l1.effective_digest(spec) == overlay_digest(files) == l1.load(spec).digest
    (tmp_path / 'schedule.py').write_text('def forward_backward(ctx, run, **kw):\n    return None\n')
    with pytest.raises(l1.OverlayViolation, match='sha256'):
        l1.effective_files(spec)
