"""ext/flashnext guards (X6; AMENDMENT_EXT_003 Q3 hard checks and Q2 locked-pair guards): every check vetoes with its
id, passes a clean screen, and is what FlashNextRunner's default judge (x6_judge) applies."""
import ast
import json

import pytest

import x6_helpers as H
from infrarsi.ext.flashnext import guards as G
from infrarsi.ext.flashnext import judge as J
from infrarsi.ext.flashnext import knobs as K
from infrarsi.ext.flashnext import parse as P
from infrarsi.ext.flashnext import runner as R
from infrarsi.ext.flashnext import switch_space as S
from infrarsi.ext.flashnext import variant as V

REG = S.registered_points()
SMATCH = REG['S_match']
CUSTOM = dict(SMATCH, retain_expert_layers=False, qkv_subgroup=False, qsa_fwd_launch='8,2')   # fast mask fixed at 0 (EXT_008)


def _parsed(tmp_path, point=SMATCH, **kw):
    return P.collect(H.write_screen(tmp_path / 'scr', H.profile_of(H.uc_a()), point, **kw))


def _ids(g):
    return [r['id'] for r in g.rejections]


@pytest.mark.parametrize('name', sorted(REG) + ['custom'])
def test_a_clean_screen_passes_every_check(tmp_path, name):
    point = CUSTOM if name == 'custom' else REG[name]
    assert not S.validate(point)[1]
    g = G.check(_parsed(tmp_path, point), point, local_check=False)
    assert g.ok, g.rejections
    reports = {r['id']: r for r in g.reports}
    assert reports['inter_step_consistency']['status'] == 'not_applicable'
    assert reports['wall_span']['span_s'] < G.CAP_S and reports['scalar_rank_consistency']['mismatches'] == []


def test_custom_point_is_a_custom_variant():
    assert K.variant_name(S.to_knobs(CUSTOM)).startswith('SX_') and G.expected_launch(CUSTOM)['variant'].startswith('SX_')


# ---------------------------------------------------------------- complete_verified
def test_complete_verified(tmp_path):
    g = G.check(_parsed(tmp_path, drop_ranks=(30,)), SMATCH)
    assert 'complete_verified' in _ids(g) and '31/32' in g.rejections[0]['detail']
    g = G.check(_parsed(tmp_path / 'n', node_exit={3: 1}), SMATCH)
    assert 'complete_verified' in _ids(g) and 'node 3' in next(r for r in g.rejections if r['id'] == 'complete_verified')['detail']


# ---------------------------------------------------------------- frozen_hashes
@pytest.mark.parametrize('field', G.PINS)
def test_frozen_pins(tmp_path, field):
    g = G.frozen_hashes(_parsed(tmp_path, attempt_overrides={field: '0' * 64}), local_check=False)
    assert _ids(g) == ['frozen_hashes'] and field in g.rejections[0]['detail']


def test_frozen_code_inventory_and_trainer_echo(tmp_path):
    m = G.load_manifest()
    code = dict(m['variant_runtime']['inventory'])
    code['systems_policy.py'] = 'f' * 64
    g = G.frozen_hashes(_parsed(tmp_path, attempt_overrides={'code_sha256': code}), local_check=False)
    assert any('systems_policy.py' in r['detail'] for r in g.rejections)
    other = dict(m['variant_runtime']['inventory'], **{'uc_contract.py': 'e' * 64})
    g = G.frozen_hashes(_parsed(tmp_path / 'b', rank_result_overrides={
        5: {'infra_family': {'variant': G.expected_launch(SMATCH)['variant'], 'training_seed': 42, 'bundle_code_sha256': other}}}),
        local_check=False)
    assert any('ranks [5]' in r['detail'] for r in g.rejections)


def test_frozen_manifest_must_exist(tmp_path, monkeypatch):
    parsed = _parsed(tmp_path)
    monkeypatch.setattr(G, 'MANIFEST_PATH', tmp_path / 'absent.json')
    g = G.frozen_hashes(parsed)
    assert _ids(g) == ['frozen_hashes'] and 'frozen_manifest.json' in g.rejections[0]['detail']


def test_local_variant_layer_against_the_manifest(tmp_path):
    """The mechanism, on a manifest made consistent with the files on this PC now (X5 may be mid-edit; X5's own test
    test_frozen_manifest_matches_the_variant_layer checks that the shipped manifest is current)."""
    m = json.loads(G.MANIFEST_PATH.read_text())
    m = dict(m, switch_space_sha256=S.sha256(),
             node_files={n: G._sha_file(G.NODE_DIR / n) for n in m['node_files']})
    assert G.local_layer_problems(m) == []
    assert G.frozen_hashes(_parsed(tmp_path), m).ok
    bad = dict(m, switch_space_sha256='0' * 64)
    assert any('switch_space.json' in r['detail'] for r in G.frozen_hashes(_parsed(tmp_path / 'b'), bad).rejections)
    bad = dict(m, node_files={'fn_launch_node.py': '0' * 64})
    assert any('fn_launch_node.py' in r['detail'] for r in G.frozen_hashes(_parsed(tmp_path / 'c'), bad).rejections)
    inv = dict(m['variant_runtime']['inventory'], **{'uc_contract.py': '1' * 64})
    bad = dict(m, variant_runtime=dict(m['variant_runtime'], inventory=inv))
    if G.load_manifest() and (G.MANIFEST_PATH.parent / 'variant.py').exists():
        from infrarsi.ext.flashnext import variant as V
        if (V.FROZEN_DIR / 'CODE_FREEZE.json').exists():
            assert any('formal-54 freeze' in p for p in G.local_layer_problems(bad))


def test_manifest_runtime_is_the_formal54_freeze_outside_its_edits(frozen):
    """The variant runtime's unchanged files are the formal-54 CODE_FREEZE inventory (d6a8f3...)."""
    m = G.load_manifest()
    common = json.loads((V.FROZEN_DIR / 'CODE_FREEZE.json').read_text(encoding='utf-8-sig'))['common_code_sha256']
    inv, changed = m['variant_runtime']['inventory'], set(m['variant_runtime']['changed_files'])
    assert set(inv) == set(common) and len(inv) == 86
    assert all(inv[n] == common[n] for n in inv if n not in changed) and changed


# ---------------------------------------------------------------- knob_binding
def test_knob_binding_requested_point_vs_bundle(tmp_path):
    g = G.knob_binding(_parsed(tmp_path, SMATCH), REG['S0'])
    assert g.rejections and all(r['id'] == 'knob_binding' for r in g.rejections)
    assert any('infra_variant' in r['detail'] for r in g.rejections)


@pytest.mark.parametrize('env,needle', [({'FLASH_QSA_FWD_LAUNCH': '8,2'}, 'FLASH_QSA_FWD_LAUNCH'),
                                        ({'FLASH_G2_MODE': 'control'}, 'FLASH_G2_MODE'),
                                        ({'FLASH_COMBINE_SEGSUM': '2'}, 'FLASH_COMBINE_SEGSUM'),
                                        ({'FLASH_AUX_READBACK_MODE': 'deferred'}, 'FLASH_AUX_READBACK_MODE'),
                                        ({'FLASH_STOP': '30'}, 'FLASH_STOP'),
                                        ({'FLASH_QSA_DKDV_FASTMASK': None}, 'FLASH_QSA_DKDV_FASTMASK')])
def test_knob_binding_launch_env(tmp_path, env, needle):
    g = G.knob_binding(_parsed(tmp_path, env_overrides=env), SMATCH)
    assert any(needle in r['detail'] for r in g.rejections), g.rejections


@pytest.mark.parametrize('over,needle', [({'systems_policy': dict(G.expected_launch(SMATCH)['systems_policy'], qkv_subgroup=False)}, 'systems_policy'),
                                         ({'training_seed': 43}, 'training_seed'),
                                         ({'run_limit': 1024}, 'run_limit'),
                                         ({'g2_mode': 'control'}, 'g2_mode')])
def test_knob_binding_attempt(tmp_path, over, needle):
    g = G.knob_binding(_parsed(tmp_path, attempt_overrides=over), SMATCH)
    assert any(needle in r['detail'] for r in g.rejections)


def test_knob_binding_runtime_echo(tmp_path):
    g = G.knob_binding(_parsed(tmp_path, rank_result_overrides={7: {'aux_readback_mode': 'deferred'}}), SMATCH)
    assert any('rank 7' in r['detail'] for r in g.rejections)
    g = G.knob_binding(_parsed(tmp_path / 'v', variant_override='S0'), SMATCH)
    assert len([r for r in g.rejections if 'trainer result' in r['detail']]) == 32
    g = G.knob_binding(_parsed(tmp_path / 'm', skip_rank_results=True), SMATCH)
    assert any('not collected' in r['detail'] for r in g.rejections)


def test_knob_binding_launched_commands(tmp_path):
    root = H.write_screen(tmp_path / 's', H.profile_of(H.uc_a()), SMATCH)
    pid = json.loads((root / 'nodes' / 'node2.pid.json').read_text())
    pid['command'][-1] = '/data/elsewhere/run_precision_screen.py'
    (root / 'nodes' / 'node2.pid.json').write_text(json.dumps(pid))
    g = G.knob_binding(P.collect(root), SMATCH)
    assert [r['detail'] for r in g.rejections] == ['node 2: launched command differs from the bundle job command']
    (root / 'fn_spec.json').unlink()
    assert any('fn_spec.json' in r['detail'] for r in G.knob_binding(P.collect(root), SMATCH).rejections)


def test_knob_binding_signal_identity(tmp_path):
    root = H.write_screen(tmp_path / 's', H.profile_of(H.uc_a()), SMATCH)
    path = root / 'signals' / 'rank11.jsonl'
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    for r in rows:
        r['attempt_id'] = 'aforeign'
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    g = G.check(P.collect(root), SMATCH)
    assert {'complete_verified', 'knob_binding'} <= set(_ids(g))


def test_expected_launch_matches_the_registered_formal54_bundles(registered):
    """Real attempt.json / job.json of the six E1 recipes (formal-54 bundles, read only) bind to their S_F1 points."""
    for entry, name in [('A1_03_U_plus_C', 'U_plus_C'), ('A1_08_B', 'B'), ('A1_06_W', 'W'), ('A1_01_S0', 'S0'),
                        ('A1_04_S_match', 'S_match'), ('A1_09_S_match_full', 'S_match_full')]:
        d = V.registered_bundle(entry)
        rec = P.launch_records(V.read(d / 'attempt.json'), V.read(d / 'job.json'), None)
        exp = G.expected_launch(REG[name])
        assert rec['attempt']['infra_variant'] == exp['variant'] == name
        assert {k: rec['job']['env'].get(k) for k in G.KNOB_ENV} == exp['env'], name
        for k in ('systems_policy', 'g2_mode', 'replay_mode', 'uc_contract', 'w_family_contract'):
            assert rec['attempt'][k] == exp[k], (name, k)
        assert {k: rec['job']['env'].get(k) for k in G.FIXED_ENV} == G.FIXED_ENV
        pins = G.load_manifest()['source_pins']
        assert all(rec['attempt'][k] == pins[k] for k in ('training_source_sha256', 'data_manifest_sha256', 'sources_sha256'))


def test_w_flags_are_the_frozen_contract(frozen):
    tree = ast.parse((V.FROZEN_RUNTIME / 'w_family_contract.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', None) == 'FLAGS')
    assert ast.literal_eval(node.value) == G.W_FLAGS


# ---------------------------------------------------------------- wall_cap, oom_nccl
def test_wall_cap(tmp_path):
    assert 'wall_cap' in _ids(G.check(_parsed(tmp_path, wall_s=2000.0), SMATCH, local_check=False))
    assert G.check(_parsed(tmp_path / 'c', wall_s=2000.0), SMATCH, cap_s=G.COLD_REFERENCE_CAP_S, local_check=False).ok
    g = G.wall_cap(_parsed(tmp_path / 't', node_exit={1: 124}))
    assert any('exit 124' in r['detail'] for r in g.rejections)
    p = _parsed(tmp_path / 'm')
    p['nodes'][0]['started_unix'] = None
    assert 'unproven' in G.wall_cap(p).rejections[0]['detail']


def test_oom_and_nccl(tmp_path):
    g = G.oom_nccl(_parsed(tmp_path, log_lines={1: 'x\ntorch.OutOfMemoryError: CUDA out of memory\n'}))
    assert _ids(g) == ['oom_nccl'] and 'OOM' in g.rejections[0]['detail'] and 'nodes/node1.log:2' in g.rejections[0]['detail']
    g = G.oom_nccl(_parsed(tmp_path / 'n', log_lines={0: '[rank3] ProcessGroupNCCL watchdog: Timeout after 600000 ms\n'}))
    assert 'NCCL' in g.rejections[0]['detail']
    g = G.oom_nccl(_parsed(tmp_path / 'f', failure_rank=4))
    assert any('failure signal row' in r['detail'] for r in g.rejections)


def test_skip_ids():
    p = {'complete': True, 'completeness': {}, 'launch': {}, 'nodes': [], 'logs': {}}
    g = G.check(p, SMATCH, skip=G.IDS)
    assert g.ok and {r['id'] for r in g.reports} == {'inter_step_consistency', 'scalar_rank_consistency'}


# ---------------------------------------------------------------- locked-pair guards
def test_locked_self_mismatch_and_void_pair():
    gates = J.load_gates()
    own = H.parsed_cell(H.uc_a())['per_step']
    assert G.locked_self_mismatch(own, own).ok
    drift = [dict(p, main_ce=p['main_ce'] * 1.003) for p in own]
    g = G.locked_self_mismatch(drift, own)
    assert _ids(g) == ['locked_self_mismatch'] and g.reports[0]['numerics_class'] == 'FAIL'
    assert _ids(G.locked_self_mismatch(None, own)) == ['locked_self_mismatch']
    assert gates['tolerance']['main_ce_rel_mean_over_window'] == 0.002          # the frozen gate it always uses
    assert _ids(G.void_pair({'valid': False, 'invalid_reasons': ['numerics FAIL']})) == ['void_pair']
    assert G.void_pair({'valid': True}).ok


# ---------------------------------------------------------------- X5's runner applies them (its default judge)
def test_runner_default_judge_is_this_pipeline(tmp_path):
    root = H.write_screen(tmp_path / 's', H.profile_of(H.uc_a()), SMATCH, env_overrides={'FLASH_G2_MODE': 'control'})
    verdict, guards, fields = R.FlashNextRunner.x6_judge(root, H.stored_ref(H.uc_a()), J.load_gates(), 'candidate',
                                                        SMATCH, 1800.0)
    assert verdict['judge_valid'] and verdict['numerics_class'] == 'E0'
    assert 'knob_binding' in _ids(guards) and fields['summary'] is verdict['details']['view']
