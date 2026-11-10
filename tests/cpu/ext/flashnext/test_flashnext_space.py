"""S_F1 (switch_space.py / switch_space.json) and the frozen manifest (deploy.py). AMENDMENT_EXT_003 Q1, Q5, Q6."""
import json

from infrarsi.ext.flashnext import switch_space as S


def test_frozen_file_is_the_generated_space():
    assert S.PATH.read_bytes() == S.generate()
    doc = S.load()
    assert doc['space_id'] == 'S_F1' and [s['name'] for s in doc['switches']] == list(S.PROPOSABLE)
    # AMENDMENT_EXT_008 Q1: corrected once; the 677-point v1 is superseded; the fast mask is fixed, cited under excluded
    assert 'AMENDMENT_EXT_008' in doc['ruling'] and doc['superseded_sha256'] == [
        'ea48b5226ab558341f35c7942aa8d787cbe1adb99900c91ef9a48f4bdff94b77'] and S.sha256() not in doc['superseded_sha256']
    assert doc['fixed']['qsa_dkdv_fastmask'] == '0' and 'qsa_dkdv_fastmask' not in S.PROPOSABLE
    ex = next(x for x in doc['excluded'] if x.startswith('qsa_dkdv_fastmask'))
    assert 'qsa_fastmask_training.py:17' in ex and 'w_family_runtime.py:123' in ex and '324 points' in ex


def test_size_and_reachable_human_recipes():
    """Q6 floor: >= 128 admissible points and every exact recipe admissible; start = U+C."""
    doc = S.load()
    assert doc['size'] == {'admissible_points': 353, 'systems_rows': 12,
                           'by_stack': {'native_rollbacks': 1, 'production': 1, 'production_qsa_memo_segsum': 27,
                                        'systems_policy': 324}}
    reg = doc['registered_points_not_for_task_views']
    assert set(reg) == {'U_plus_C', 'B', 'W', 'S0', 'S_match', 'S_match_full'}
    for name, p in reg.items():
        c, errors, _ = S.validate(p)
        assert not errors and c == p, name
    assert reg['U_plus_C'] == S.start_point() == doc['start']
    keys = {S.key(p) for p in S.points()}
    assert len(keys) == 353 and all(S.key(p) in keys for p in reg.values())
    assert all(p['qsa_dkdv_fastmask'] == ('0' if p['implementation_stack'] == 'systems_policy' else None) for p in S.points())


def test_task_view_names_no_recipe_and_no_timing():
    text = json.dumps(S.task_view())
    for word in ('S_match', 'S0', 'U_plus_C', 'U+C', 'Q_full', 'Qo', '"W"', '"B"', '12.49', '8.429', 'provenance', 'vendored'):
        assert word not in text, word


def test_fixed_and_excluded_values_are_not_proposable():
    p = dict(S.registered_points()['S_match'], moment_policy='scaled_fp16_dual_cuda')
    assert any('not switches' in e for e in S.validate(p)[1])
    assert any('not switches' in e for e in S.validate(dict(S.start_point(), aux_readback_mode='deferred'))[1])
    assert any('not switches' in e for e in S.validate(dict(S.start_point(), tensor_model_parallel_size=4))[1])


def test_validator_rules():
    start = S.start_point()
    assert S.validate(dict(start, qkv_subgroup=True))[1]                       # not read by the rollback stack
    assert S.validate({'implementation_stack': 'systems_policy'})[1]           # systems switches have no default
    bad = dict(S.registered_points()['S_match_full'], retain_expert_layers=True)
    assert any('native_full_accounting.py:15' in e for e in S.validate(bad)[1])
    w, errors, _ = S.validate({'implementation_stack': 'production_qsa_memo_segsum'})
    assert not errors and w['qsa_fwd_launch'] == '4,1' and w['qsa_dkdv_fastmask'] is None
    assert S.validate(dict(w, qsa_dkdv_fastmask='1'))[1]                        # pinned inside that block
    assert S.validate(dict(w, qsa_dkdv_fastmask='0'))[0] == w                    # EXT_008: '0' admissible everywhere
    k = S.to_knobs(w)
    assert k['family'] == 'W' and k['qsa_dkdv_fastmask'] == '0' and k['aux_readback_mode'] == 'control'
    assert S.from_knobs(k) == w


def test_random_control_sampler_is_deterministic_and_admissible():
    a, b = S.sample_order(42, 12), S.sample_order(42, 12)
    assert a == b and len(a) == 12
    keys = [S.key(p) for p in a]
    assert len(set(keys)) == 12 and S.key(S.start_point()) not in keys
    assert all(not S.validate(p)[1] for p in a)
    rest = S.sample_order(42, 3, seen=keys[:5])
    assert not set(S.key(p) for p in rest) & set(keys[:5])
    assert S.sample_order(43, 12) != a


def test_frozen_manifest_matches_the_variant_layer(frozen):
    from infrarsi.ext.flashnext import deploy as D
    m = D.load_manifest()
    assert m['formal54']['code_freeze_sha256'] == frozen.CODE_FREEZE_SHA256
    assert m['switch_space_sha256'] == S.sha256()
    assert m['variant_runtime']['root'] == frozen.FN_ROOT
    assert m['variant_runtime']['edits']['family_worker.py:run_root'] == 1
    assert m['variant_runtime']['changed_files'] == ['collect_family.py', 'family_policy.py', 'family_runtime.py',
                                                    'family_transform.py', 'family_worker.py', 'systems_policy.py']
    assert all(p.startswith('/data/infrarsi/ext/') for p in m['remote_layout'].values() if p.startswith('/data/infrarsi'))


def test_node_launcher_compiles():
    """fn_launch_node.py runs on Linux nodes only (fcntl, killpg); here it must at least compile."""
    compile((S.PATH.parent / 'node' / 'fn_launch_node.py').read_text(encoding='utf-8'), 'fn_launch_node.py', 'exec')


def test_ext008_fast_mask_is_fixed_for_every_arm():
    """AMENDMENT_EXT_008 Q1: '0' or unset -> '0' where the stack reads it (shape kept); anything else is an
    invalid_proposal without a why; not in the switch table, listed under fixed; never drawn by F-RND."""
    sm, start = S.registered_points()['S_match'], S.start_point()
    for stack_point in (sm, S.registered_points()['W'], start, S.registered_points()['B']):
        errors = S.validate(dict(stack_point, qsa_dkdv_fastmask='1'))[1]
        assert errors == ["qsa_dkdv_fastmask='1' not in ['0']: fixed"]
        assert S.validate(dict(stack_point, qsa_dkdv_fastmask='0'))[0] == stack_point
    assert S.validate({k: v for k, v in sm.items() if k != 'qsa_dkdv_fastmask'})[0] == sm      # unset -> '0'
    assert set(sm) == set(start) == set(S.SWITCHES)                                              # canonical shape
    view = S.task_view()
    assert 'qsa_dkdv_fastmask' not in [r['name'] for r in view['switches']]
    assert view['fixed'].endswith('30 updates, QSA dK/dV fast mask off (qsa_dkdv_fastmask = 0)')
    text = json.dumps(view)
    for word in ('assert', 'crash', 'qsa_fastmask_training', 'w_family_runtime', 'EXT_008', 'superseded'):
        assert word not in text, word                                                           # nothing about why
    draws = S.RandomSwitchSampler(42)
    for _ in range(200):
        assert 'qsa_dkdv_fastmask' not in draws.draw()
    assert all(p['qsa_dkdv_fastmask'] in ('0', None) for p in S.sample_order(42, 12))
