"""ext/flashnext task contracts and hooks (X6; AMENDMENT_EXT_003 Q1, Q5, D-F1-5): budgets exactly as the Q5 table, the
task view (no recipe names, no stored timings), the S_F1 validator, the receipt digest, the F-PROMPTS start check and
the F-OPTICO codec."""
import json
import math

import pytest

import x6_helpers as H
from infrarsi.agent.budget import VirtualClock
from infrarsi.agent.context import estimate_tokens
from infrarsi.baselines.optico_style.optico import parse_change_para
from infrarsi.ext.flashnext import deviations as dv
from infrarsi.ext.flashnext import guards as G
from infrarsi.ext.flashnext import switch_space as S
from infrarsi.ext.flashnext import task as T
from infrarsi.ext.flashnext.hooks import FlashNextHooks, SwitchCodec
from infrarsi.harness import signing

Q5 = {'F-PROMPTS': (3, 200_000, 10.0, 7200, 48), 'F-OPTICO': (12, 1_500_000, 40.0, 18000, 192),
      'F-A0': (12, 1_500_000, 40.0, 18000, 192), 'F-RND': (12, 0, 0.0, 18000, 192)}


@pytest.fixture(scope='module')
def uc(tmp_path_factory):
    return H.uc_reference_receipt(tmp_path_factory.mktemp('uc'), clock=VirtualClock())


@pytest.mark.parametrize('arm', sorted(Q5))
def test_budgets_are_the_q5_table(arm):
    t = T.make_task(arm)
    screens, tokens, usd, wall, gpuh = Q5[arm]
    assert (t.budget['screens'], t.budget['llm_tokens'], t.budget['wall_seconds'], t.budget['gpu_hours']) == (screens, tokens, wall, gpuh)
    b = t.search_budget()
    assert b.scoring_reserve_screens == 0.0 and b.screens == screens
    if arm == 'F-RND':
        assert 'llm_usd' not in t.budget and b.llm_usd is None and t.raw['llm_usd_cap_recorded'] == 0.0
    else:
        assert t.budget['llm_usd'] == usd == b.llm_usd
    assert (t.env, t.metric_name, t.world_size, t.margin) == ('flashnext', 'step_time_s', 32, {'kind': 'rsd', 'floor': 0.03, 'k': 3})
    assert t.tau == pytest.approx(math.log(1.10)) and t.raw['arm_index'] == T.ARM_INDEX[arm]
    assert t.raw['start_point'] == S.start_point() and t.raw['switch_space']['sha256'] == S.sha256()
    assert t.raw['deviations'] == list(dv.COMMON) + (['D-OC-15'] if arm == 'F-OPTICO' else [])
    assert t.raw['lifecycle'] == dv.SCREEN_LEVEL_LABEL


def test_same_space_and_judge_for_every_arm():
    tasks = [T.make_task(a) for a in Q5]
    for key in ('switch_space', 'numerics_gates', 'start_point', 'screen', 'facts', 'knob_schema'):
        assert len({json.dumps(t.raw[key], sort_keys=True) for t in tasks}) == 1, key
    assert len({t.frozen_digest for t in tasks}) == 1
    assert [T.ARM_INDEX[a] for a in ('F-PROMPTS', 'F-OPTICO', 'F-A0', 'F-RND')] == [0, 1, 2, 3]


def test_references_and_frozen_files():
    for ref in ('UC-REF', 'F-HT'):
        t = T.make_task(ref)
        assert t.budget['screens'] == 1 and t.raw['role'] == 'reference'
    files = T.frozen_files()
    m = G.load_manifest()
    assert len(files) == 86 + len(m['remote_read_only'])
    assert all(k.startswith('code/') or k.startswith('data/') for k in files)
    with pytest.raises(ValueError):
        T.make_task('F-OE')


def test_task_view_has_every_q1_part_and_no_leak(uc):
    view = FlashNextHooks(T.make_task('F-A0'), start_receipt=uc).task_view()
    assert {'hardware_profile', 'workload', 'fixed_layout', 'start_configuration', 'start_measured', 'switch_space',
            'gates', 'budget', 'contract'} <= set(view)
    assert view['hardware_profile']['summary']['gpu_type'] == 'H100-80GB' and view['hardware_profile']['summary']['nodes'] == 4
    assert view['fixed_layout']['tp'] == 8 and view['fixed_layout']['ep'] == 32
    assert view['switch_space'] == S.task_view() and view['start_configuration'] == S.start_point()
    assert view['start_measured']['step_time_s'] == uc['metric']['value'] and len(view['start_measured']['per_step_max_rank_s']) == 30
    assert view['gates']['numerics']['tolerance']['grad_norm_rel_max_first_update'] == 0.03
    assert view['budget'] == T.make_task('F-A0').budget
    text = json.dumps(view)
    assert T.task_view_leaks(view) == [], T.task_view_leaks(view)
    assert 'provenance' not in text and 'registered_points' not in text


def test_the_leak_check_bites():
    assert T.task_view_leaks({'x': 'retain like S_match'}) == ['S_match']
    assert T.task_view_leaks({'stack': 'W'}) == ['"W"']
    assert T.task_view_leaks({'t': 12.490}) == ['12.49'] and T.task_view_leaks({'t': 8.4291}) == []
    assert T.task_view_leaks({'start': 'U+C'}) == ['U+C']


@pytest.mark.parametrize('point', [S.start_point(), S.registered_points()['S_match'], {'implementation_stack': 'production'},
                                   {'implementation_stack': 'systems_policy', 'retain_attention_prefix': False,
                                    'retain_expert_layers': True, 'qkv_subgroup': True, 'shared_master_guard': True},
                                   {'implementation_stack': 'production', 'qkv_subgroup': True},
                                   dict(S.start_point(), moment_policy='scaled_fp16_dual_cuda'), {'tp': 4}, 'x'])
def test_validator_is_switch_space_validate(point):
    h = FlashNextHooks(T.make_task('F-RND'))
    got = h.validate_config(point)
    if not isinstance(point, dict):
        assert got[1] and got[0] == {}
        return
    want = S.validate(point)
    assert got[1] == want[1] and got[2] == want[2] and got[0] == (want[0] if not want[1] else {})


def test_hooks_basics():
    h = FlashNextHooks(T.make_task('F-A0'))
    assert h.layers == ('L0',) and h.tools_enabled is False
    assert h.skill_tags() == ['external', 'flashnext', 'gdn', 'moe', 'multi-node']
    assert h.validate_patch({'a.py': 'x'})[0] and 'switches' in h.validate_patch({})[0][0]
    for text in [h.contract_text(), *h.verifier_checklist(), h.fixed_parameters_text(), h.skill_query()]:
        assert T.task_view_leaks(text) == [], text


@pytest.mark.parametrize('limit', [1500, 600, 300])
def test_receipt_digest_is_bounded_and_keeps_the_view(uc, limit):
    h = FlashNextHooks(T.make_task('F-A0'))
    d = h.receipt_digest(uc, None, max_tokens=limit)
    assert estimate_tokens(json.dumps(d, separators=(',', ':'), default=str)) <= limit or len(d) <= 2
    assert d['id']['valid'] is True
    if limit == 1500:
        assert d['numerics']['class'] == 'E0' and d['metric']['step_time_s'] == uc['metric']['value']
        assert d['point'] == S.start_point() and 'summary' in d
    d = h.receipt_digest(uc, None, max_tokens=600, fields=('id', 'verdict', 'metric', 'summary', 'guards'))
    assert 'point' not in d and 'cost' not in d


def test_start_receipt_check(uc, tmp_path):
    h = FlashNextHooks(T.make_task('F-PROMPTS'))
    assert h.start_receipt_problems(uc, S.start_point()) == []
    unsigned = {k: v for k, v in uc.items() if k != signing.SIGNATURE_FIELD}
    assert any('signature' in p for p in h.start_receipt_problems(unsigned, S.start_point()))
    other = H.copy_receipt(uc, point=S.registered_points()['S_match'])
    assert any('its point is not the UC-REF point' in p for p in h.start_receipt_problems(other, S.start_point()))
    assert any('not the S_F1 start point' in p for p in h.start_receipt_problems(uc, S.registered_points()['B']))
    bad = H.copy_receipt(uc, valid=False)
    assert any('not a valid screen' in p for p in h.start_receipt_problems(bad, S.start_point()))


def test_optico_codec_round_trips_every_point():
    codec = SwitchCodec()
    assert codec.names() == list(S.SWITCHES)
    for p in S.points():
        lines = codec.to_lines(p)
        assert ',' not in lines                                    # OptiCo's change_para parser splits on commas
        current = dict(line.split(': ', 1) for line in lines.splitlines())
        reply = 'Thinking...\nThe modified parameters are:\n' + lines
        back = codec.to_config(parse_change_para(reply, current))
        point, errors, _ = S.validate(back)
        assert not errors and point == p


def test_fixed_parameters_text_carries_the_domain():
    text = FlashNextHooks(T.make_task('F-OPTICO')).fixed_parameters_text()
    for s in S.PROPOSABLE:
        assert f'\n{s}: ' in text
    assert '\nqsa_dkdv_fastmask: ' not in text                     # EXT_008: fixed, not a parameter to change
    assert '4/1 | 4/2 | 8/2' in text and 'Fixed layout: TP 8, EP 32' in text
