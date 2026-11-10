import pytest

from infrarsi.ext.autoresearch import knobs, source
from infrarsi.ext.autoresearch.hooks import APPENDIX, AutoresearchHooks
from infrarsi.ext.autoresearch.native import task_text
from infrarsi.ext.autoresearch.task import ARMS, env_c_task, load_assets
from x1_ar_helpers import ar_task


def test_tasks_per_arm(tmp_path):
    for arm, (space, track) in ARMS.items():
        t = ar_task(tmp_path, arm)
        assert (t.id, t.env, t.arm, t.action_space, t.raw['track'], t.world_size, t.metric_name) == ('ENV-C', 'autoresearch', arm, space, track, 1, 'val_bpb')
        assert t.margin == {'kind': 'fixed', 'value': 0.002} and t.tau == 0.002 and t.mutable_globs == ('train.py',)
        assert t.budget['screens'] == 36 and t.budget['wall_seconds'] == 14400 and t.budget['gpu_hours'] == 4.0
        assert {'D-AR-1', 'D-AR-2', 'D-AR-3'} <= set(t.raw['deviations'])
        assert ('D-AR-5' in t.raw['deviations']) == (track == 'infra') and ('D-AR-6' in t.raw['deviations']) == arm.startswith('C-NAT')
    rnd = ar_task(tmp_path, 'C-RND')
    assert rnd.budget['llm_tokens'] == 0 and rnd.knob_schema == knobs.KNOB_SCHEMA and ar_task(tmp_path, 'C-IS').knob_schema == {}
    ff = rnd.frozen_files
    assert len([k for k in ff if k.startswith('data/')]) == 11 and 'tokenizer/tokenizer.pkl' in ff and 'kernel/_ops.py' in ff
    assert ff['repo/prepare.py'] == source.sha256_text(source.candidate_prepare('h2d_sync'))           # D-AR-7 (default)
    assert ff['judge/prepare.py'] == source.UPSTREAM_SHA256['prepare.py'] and 'judge/model_ref.py' in ff and 'tools/run_train.py' in ff
    with pytest.raises(ValueError):
        env_c_task('C-XX', assets={})


def test_committed_assets_match_the_recorded_r2_tokenizer():
    assets = load_assets()
    assert assets['tokenizer'] == source.R2['tokenizer_sha256'] and set(assets['data']) == set(source.DATA_SHARDS)
    assert assets['prep']['tokenizer_determinism']['deterministic'] is True
    assert env_c_task('C-IS').frozen_digest == env_c_task('C-RND').frozen_digest


def test_anchor_is_the_solo_d_ar_7_baseline_on_winh0():
    """AMENDMENT_EXT_002 (c): tau, AUC_budget, check (4) and the calibration gate use a21767e9dbd4f."""
    t = env_c_task('C-IS')
    a = t.raw['anchor']
    assert (a['attempt_id'], a['node'], a['num_steps'], a['prepare_variant']) == ('a21767e9dbd4f', 'WINH0', 936, 'h2d_sync')
    assert abs(a['val_bpb'] - 0.996365) < 1e-6 and t.facts['baseline']['val_bpb'] == a['val_bpb']
    assert t.raw['reference_losses'] == a['first_losses'] and len(a['first_losses']) == 10 and a['first_losses'][1] == 8.854702
    assert t.raw['expected_runtime']['optimizer_groups'] and len(t.raw['expected_runtime']['optimizer_groups']) == 9


def test_hooks_patch_guard_is_the_prelaunch_infra_check(tmp_path):
    h = AutoresearchHooks(ar_task(tmp_path, 'C-IS'))
    base = source.base_train()
    ok = base.replace('DEVICE_BATCH_SIZE = 128', 'DEVICE_BATCH_SIZE = 64')
    assert h.validate_patch({'train.py': ok}) == ([], [])
    errs, _ = h.validate_patch({'train.py': base.replace('MATRIX_LR = 0.04', 'MATRIX_LR = 0.05')})
    assert errs and errs[0].startswith('infra-only: constant MATRIX_LR changed')
    errs, _ = h.validate_patch({'prepare.py': 'x'})
    assert any('not mutable' in e for e in errs)
    errs, _ = h.validate_patch({'train.py': base + ''.join(f'# pad {i}\n' for i in range(301))})
    assert any('lines > limit 300' in e for e in errs)
    errs, _ = h.validate_patch({'train.py': base.replace('import gc', 'import gc\nimport subprocess')})
    assert any('import of subprocess' in e for e in errs)
    assert 'syntax error' in h.validate_patch({'train.py': 'def (:'})[0][0]
    native = AutoresearchHooks(ar_task(tmp_path, 'C-NAT-infra'), native=True)
    assert native.validate_patch({'train.py': base.replace('MATRIX_LR = 0.04', 'MATRIX_LR = 0.05')}) == ([], [])
    assert native.validate_patch({'prepare.py': 'x'})[0]
    free = AutoresearchHooks(ar_task(tmp_path, 'C-NAT-free'))
    assert free.validate_patch({'train.py': base.replace('DEPTH = 8 ', 'DEPTH = 10 ')}) == ([], [])


def test_hooks_views_and_knobs(tmp_path):
    h = AutoresearchHooks(ar_task(tmp_path, 'C-IS'))
    assert h.layers == ('L1',) and h.validate_config({}) == ({}, [], [])
    v = h.task_view()
    assert v['facts']['baseline']['val_bpb'] == 0.9977 and v['facts']['prepare_constants']['TIME_BUDGET'] == 300
    assert 'Infra-only track' in v['contract'] and APPENDIX.splitlines()[0] in v['contract'] and v['read_only_files'] == ['README.md', 'prepare.py']
    rf = h.read_file('train.py', 1, 30)
    assert rf['total_lines'] == len(source.base_train().splitlines()) and 'Autoresearch pretraining script' in rf['text']
    assert 'evaluate_bpb' in h.read_file('prepare.py', 340, 365)['text'] and 'error' in h.read_file('uv.lock')
    assert any('TOTAL_BATCH_SIZE' in x for x in h.verifier_checklist())
    r = AutoresearchHooks(ar_task(tmp_path, 'C-RND'))
    assert r.layers == ('L0',)
    resolved, errs, _ = r.validate_config({'DEVICE_BATCH_SIZE': 64})
    assert not errs and resolved == {**knobs.DEFAULTS, 'DEVICE_BATCH_SIZE': 64}
    assert r.validate_config({'DEVICE_BATCH_SIZE': 100})[1] and r.validate_config({'mode': 'x'})[1]


def test_native_task_text(tmp_path):
    t = task_text(ar_task(tmp_path, 'C-NAT-infra'), repo='/r/repo', tag='oct21')
    assert t.startswith(source.upstream('program.md')[:200]) and APPENDIX in t and 'run_experiment' in t and 'autoresearch/oct21' in t
    f = task_text(ar_task(tmp_path, 'C-NAT-free'), repo='/r/repo', tag='oct21')
    assert APPENDIX not in f and 'val_bpb it prints' in f and '36 experiments' in f and '4 hours' in f


def test_knob_space_only_reaches_c_rnd(tmp_path):
    """AMENDMENT_EXT_007: the frozen space is C-RND's; the LLM arms' task specs and views carry nothing of it."""
    import json
    from infrarsi.ext.autoresearch.task import env_c_task
    from x1_ar_helpers import make_assets
    assets = make_assets(tmp_path)
    rnd = env_c_task('C-RND', assets=assets)
    assert rnd.raw['knob_space'] == knobs.frozen_reference() and rnd.knob_schema == knobs.KNOB_SCHEMA
    for arm in ('C-IS', 'C-NAT-infra', 'C-NAT-free', 'C-BASE'):
        t = env_c_task(arm, assets=assets)
        assert 'knob_space' not in t.raw and t.knob_schema == {}
        view = json.dumps(AutoresearchHooks(t).task_view(), default=str)
        assert 'reduce-overhead' not in view and 'knob_space' not in view and 'max-autotune' not in view
