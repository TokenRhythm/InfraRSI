"""The transfer judgment with a fake LLM (experience decision 3, 6.5) and the write-back of a run as a pack that the next
run's store loads (6.6)."""
import json
from types import SimpleNamespace

import pytest

from expjob import KNOBS, REF, megatron_log, project
from infrarsi.experience import transfer
from infrarsi.experience.pack import read_lines, validate
from infrarsi.experience.runs import plan_for, summary, write_back
from infrarsi.experience.store import Store, fingerprint, plan


class FakeDriver:
    model = 'fake-llm'

    def __init__(self, action):
        self.action, self.calls = action, []

    def complete(self, system, prompt, *, schema=None, role=None, max_output_tokens=None):
        self.calls.append((role, json.loads(prompt)))
        return SimpleNamespace(action=self.action, model=self.model, llm_tokens=100, llm_usd=0.0, accounting_complete=True)


@pytest.fixture
def fp(tmp_path):
    return fingerprint(project('NVIDIA A100-SXM4-80GB', 80), {'log': str(megatron_log(tmp_path / 'train.log'))})


def test_the_judgment_sets_signs_feasibility_and_a_start(fp):
    pl = plan(Store.load(), fp, KNOBS, REF)
    assert pl['coverage']['fp8']['level'] == 'partial' and transfer.needed(pl)
    pl['composite'] = None                                          # as when the data give no start point
    action = {'assessment': [
        {'knob': 'fp8', 'change': {'fp8': '1'}, 'sign': 'unknown', 'p_feasible': 0.5, 'why': 'no fp8 tensor cores'},
        {'knob': 'micro_batch_size', 'change': {'micro_batch_size': '2'}, 'sign': '+', 'p_feasible': 0.4},   # a string
        {'knob': 'num_workers', 'change': {'num_workers': 8}, 'sign': 'infeasible', 'p_feasible': 0.1},
        {'knob': 'TORCH_NCCL_AVOID_RECORD_STREAMS', 'change': {'TORCH_NCCL_AVOID_RECORD_STREAMS': '1'}, 'sign': '+'},
        {'knob': 'overlap_grad_reduce', 'change': {'overlap_grad_reduce': True}, 'sign': '+', 'p_feasible': 2},
        {'knob': 'cross_entropy_fusion_impl', 'change': {'cross_entropy_fusion_impl': 'te'}, 'sign': '+'},
        {'knob': 'overlap_param_gather', 'change': {'overlap_param_gather': '1'}, 'sign': '-', 'flip': True},
        {'knob': 'not_a_knob', 'change': {'not_a_knob': 1}, 'sign': '+'}],
        'start_config': {'micro_batch_size': '2', 'TORCH_NCCL_AVOID_RECORD_STREAMS': '1', 'overlap_grad_reduce': 'true',
                         'cross_entropy_fusion_impl': 'te', 'fp8': '1'}}
    drv, session = FakeDriver(action), SimpleNamespace(charged=[], charge_llm=lambda r, role: session.charged.append(role))
    rec = transfer.judge(drv, pl, KNOBS, REF, session=session)
    assert session.charged == ['transfer'] and drv.calls[0][0] == 'transfer'
    assert {k['knob'] for k in drv.calls[0][1]['knobs']} == set(KNOBS)
    seeds = {s['knob']: s for s in pl['seeds']}
    assert seeds['fp8']['sign'] == 'unknown' and seeds['micro_batch_size']['sign'] == '+'
    assert seeds['overlap_grad_reduce']['p_feasible'] == 0.9         # clipped to [0.1, 0.9]
    assert any('judged infeasible' in w for w in pl['check']({**REF, 'num_workers': 8}))
    assert pl['composite']['members'] == ['micro_batch_size', 'TORCH_NCCL_AVOID_RECORD_STREAMS', 'overlap_grad_reduce',
                                          'cross_entropy_fusion_impl']   # judged '+', at most four; fp8 was not '+'
    assert seeds['TORCH_NCCL_AVOID_RECORD_STREAMS']['gain'] == 0.0      # a member without data: a zero-size seed
    assert pl['composite']['change']['micro_batch_size'] == 2 and pl['composite']['change']['overlap_grad_reduce'] is True
    assert 'sign' not in seeds['overlap_param_gather']                 # V0: an opposite sign keeps the data's
    assert [r['used'] for r in rec['assessment'] if r['knob'] == 'overlap_param_gather'] == [False]
    assert len(rec['assessment']) == 7 and rec['model'] == 'fake-llm'


def test_the_judgment_is_scored_by_the_run():
    rec = {'assessment': [{'change': {'a': 1}, 'sign': '+', 'p_feasible': 0.8},
                          {'change': {'b': 1}, 'sign': '+', 'p_feasible': 0.9},
                          {'change': {'c': 1}, 'sign': '-', 'p_feasible': 0.6},
                          {'change': {'d': 1}, 'sign': '+', 'p_feasible': 0.5}]}
    ledger = [{'change': {'a': 1}, 'status': 'confirmed', 'measured': 0.03},
              {'change': {'b': 1}, 'status': 'refuted', 'measured': -0.02},
              {'change': {'c': 1}, 'status': 'unresolved', 'measured': 0.001},       # within noise: no sign score
              {'change': {'d': 1}, 'status': 'invalid', 'failed_checks': ['completes']}]
    sc = transfer.score(rec, ledger, 0.002, {'status': 'screened', 'gain': 0.05})
    assert sc['sign_acc'] == 0.5 and sc['n_resolved'] == 2 and sc['n_tried'] == 4 and sc['start_gain'] == 0.05
    assert sc['feasibility_brier'] == pytest.approx(((0.8 - 1) ** 2 + (0.9 - 1) ** 2 + (0.6 - 1) ** 2 + 0.5 ** 2) / 4)


def trace():
    cands = [{'id': 'c0', 'parent': None, 'change': None, 'valid': True, 'step_time': 3.0},
             {'id': 'c1', 'parent': 'c0', 'change': {'micro_batch_size': [1, 2]}, 'valid': True, 'step_time': 2.9},
             {'id': 'c2', 'parent': 'c0', 'change': {'fp8': ['0', '1']}, 'valid': False, 'step_time': 2.2},
             {'id': 'c3', 'parent': 'c1', 'change': {'nan_check_off': [False, True]}, 'valid': True, 'step_time': 2.88},
             {'id': 'c4', 'parent': 'c1', 'change': {'manual_gc': [False, True]}, 'valid': True, 'step_time': 2.7}]
    full = lambda c, ok, t, why=(): {'candidate_id': c, 'fidelity': 'full', 'valid': ok, 'step_time': t, 'reasons': list(why)}  # noqa: E731
    screens = [full('c1', True, 2.9), full('c2', False, 2.2, ['loss_within_tolerance: 21 % off']), full('c3', True, 2.885),
               full('c3', True, 2.875), full('c4', True, 2.7),
               {'candidate_id': 'c4', 'fidelity': 'probe', 'valid': True, 'step_time': 9.9}]
    return {'candidates': cands, 'screens': screens, 'extra': {'strategy': {'name': 'hls', 'experience': {'seeded': 2}},
                                                                 'ledger': []}}


def test_a_run_is_written_back_and_loaded_as_experience(tmp_path, monkeypatch):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    proj = project()
    pl = plan_for(proj, {'log': str(megatron_log(tmp_path / 'train.log'))}, log=lambda m: None)
    promo = {'tried': [{'candidate': 'c4', 'items': [{'id': 'speed_reproduces', 'status': 'fail'}]}]}
    path = write_back(proj, pl, trace(), promo, 'r1', 0.0012, summary(pl, trace(), 0.0012))
    lines = read_lines(path)
    assert validate(lines)[0] == [] and path.parent.name == 'runs'
    obs = {x['id']: x for x in lines if x['kind'] == 'observation'}
    assert set(obs) == {'c0', 'c1', 'c2', 'c3'}                     # c4's speed did not reproduce at promotion
    assert obs['c2']['status'] == 'invalid' and obs['c3']['n'] == 2
    assert obs['c3']['step_time_s'] == pytest.approx((2.885 * 2.875) ** 0.5)
    assert obs['c1']['config']['micro_batch'] == '2' and obs['c3']['config']['nan_check'] == 'off'
    env = next(x for x in lines if x['kind'] == 'env')
    assert env['gpu'] == 'H200-141GB' and env['noise_pct'] == 0.12
    st = Store.load(include=['run-*'])
    eff = {tuple(e['change']): e for e in st.effects}
    assert eff[('micro_batch',)]['speedup'] == pytest.approx(3.0 / 2.9)
    assert eff[('nan_check',)]['change']['nan_check'] == ['on', 'off'] and len(st.effects) == 2
    again = plan(st, fingerprint(proj, {'log': str(tmp_path / 'train.log')}), KNOBS, REF)
    assert again['coverage']['nan_check_off']['level'] == 'covered'  # 0.5 % against a 0.12 % noise: readable


def test_a_knob_the_experience_cannot_place_is_named(tmp_path):
    # F-D2-R205-4: the user's sharding knob `accel` matched no mechanism, got no data and was never tried
    accel = {'type': 'str', 'flag': '--accel', 'choices': ['zero2', 'zero3'], 'default': 'zero3'}
    seen = []
    plan_for(project(knobs={**KNOBS, 'accel': accel}), {'log': str(megatron_log(tmp_path / 'train.log'))}, log=seen.append)
    assert 'no mechanism known for accel' in seen[-1]
    named = {**accel, 'mechanism': 'zero_stage', 'states': {'zero2': '2', 'zero3': '3'}}
    pl = plan_for(project(knobs={**KNOBS, 'accel': named}), {'log': str(megatron_log(tmp_path / 't2.log'))}, log=seen.append)
    assert pl['coverage']['accel']['mechanism'] == 'zero_stage' and 'accel' not in seen[-1]
