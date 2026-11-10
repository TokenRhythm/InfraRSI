"""R9 review (gpt-6.1-sol) of the experience layer: one test per finding (each fails on the pre-fix code)."""
from types import SimpleNamespace

import pytest

from expjob import KNOBS, REF, megatron_log, project
from infrarsi.experience import transfer
from infrarsi.experience.pack import derive_effects
from infrarsi.experience.runs import write_back
from infrarsi.experience.store import Store, fingerprint, plan, read_lines


def head(pid, *rest):
    return [{'kind': 'pack', 'id': 'pack', 'schema': 'infrarsi-exp/1', 'pack_id': pid, 'title': pid, 'license': 'CC0',
             'tier': 'measured'},
            {'kind': 'env', 'id': 'e', 'gpu': 'H200-141GB', 'gpus_per_node': 8, 'nodes': 1, 'framework': 'megatron',
             'model': {'params_b': 8.0}, 'workload': {'seq_len': 4096}}, *rest]


def obs(i, cfg, **kw):
    return {'kind': 'observation', 'id': i, 'env': 'e', 'config': cfg, 'status': 'ok', **kw}


def test_1_a_pair_is_compared_in_one_unit():
    mixed = head('p', obs('a', {'compile': 'off'}, step_time_s=2.0), obs('b', {'compile': 'on'}, tokens_per_s_per_gpu=1e4))
    assert derive_effects(mixed) == []
    both = head('p', obs('a', {'compile': 'off'}, step_time_s=2.0, tokens_per_s_per_gpu=4000),
                obs('b', {'compile': 'on'}, tokens_per_s_per_gpu=5000))
    assert derive_effects(both)[0]['speedup'] == pytest.approx(5000 / 4000)


def test_2_confirmation_reruns_are_in_the_written_observation(tmp_path, monkeypatch):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    proj = project()
    fp = fingerprint(proj, {'log': str(megatron_log(tmp_path / 'train.log'))})
    pl = plan(Store.load(include=['h200-*']), fp, KNOBS, REF)
    tr = {'candidates': [{'id': 'c0', 'parent': None, 'change': None, 'valid': True, 'step_time': 3.0},
                         {'id': 'c1', 'parent': 'c0', 'change': {'micro_batch_size': [1, 2]}, 'valid': True, 'step_time': 2.9}],
          'screens': [{'candidate_id': 'c1', 'fidelity': 'full', 'valid': True, 'step_time': 2.9},
                      {'candidate_id': 'c0', 'fidelity': 'rescore', 'valid': True, 'step_time': 3.02},
                      {'candidate_id': 'c1', 'fidelity': 'rescore', 'valid': True, 'step_time': 3.2}],
          'extra': {}}
    o = {x['id']: x for x in read_lines(write_back(proj, pl, tr, None, 'r2', 0.001)) if x['kind'] == 'observation'}
    assert o['c1']['step_time_s'] == pytest.approx((2.9 * 3.2) ** 0.5) and o['c1']['n'] == 2
    assert o['c0']['step_time_s'] == pytest.approx((3.0 * 3.02) ** 0.5) and o['c0']['n'] == 2
    eff = Store.load(include=['run-r2']).effects
    assert len(eff) == 1 and eff[0]['speedup'] < 1                      # the confirmation reversed the screen's sign


def small_plan():
    knobs = {'mbs': {'type': 'int', 'choices': [1, 2, 4]}, 'gc': {'type': 'bool'}}
    seed = {'knob': 'mbs', 'mechanism': 'micro_batch', 'change': {'mbs': 2}, 'level': 'partial', 'gain': 0.02,
            'evidence_ids': [], 'warnings': []}
    pl = {'coverage': {'mbs': {'level': 'partial', 'mechanism': 'micro_batch'}, 'gc': {'level': 'none', 'mechanism': 'manual_gc'}},
          'seeds': [seed], 'composite': None, 'check': lambda cfg: []}
    return pl, knobs, {'mbs': 1, 'gc': False}


def test_3_judged_moves_and_start_members_match_the_whole_change():
    pl, knobs, ref = small_plan()
    transfer.apply(pl, {'assessment': [{'knob': 'mbs', 'change': {'mbs': 4}, 'sign': '+'},
                                       {'knob': 'gc', 'change': {'gc': True}, 'sign': '+'}],
                        'start_config': {'mbs': 4, 'gc': True}}, knobs, ref)
    assert pl['composite'] is None and pl['seeds'][0]['change'] == {'mbs': 2}   # 4 is not what the bundle would run
    pl, knobs, ref = small_plan()
    transfer.apply(pl, {'assessment': [{'knob': 'mbs', 'change': {'mbs': 4}, 'sign': 'infeasible'}]}, knobs, ref)
    assert pl['check']({'mbs': 4, 'gc': False}) and not pl['check']({'mbs': 2, 'gc': False})


def test_5_the_layout_comes_from_the_knobs_without_an_argument_dump(store):
    knobs = {**KNOBS, 'tp': {'type': 'int', 'choices': [1, 2], 'flag': '--tensor-model-parallel-size', 'default': 2}}
    p = project(knobs=knobs, model={'params_b': 8.03}, workload={'seq_len': 4096})
    pl = plan(store, fingerprint(p), knobs, p.reference_config())
    assert pl['fingerprint']['layout']['tp'] == '2'
    c = pl['coverage']['micro_batch_size']                               # the TP2 measurement, not TP1's +2.1 %
    assert c['evidence'][0] == 'h200-harvest-2026-10/HM22~HM07' and c['gain'] == pytest.approx(0.0777, abs=1e-3)


def test_6_balance_coverage_needs_the_same_gpu_count(store, tmp_path):
    p = project()
    p.gpus = 1
    pl = plan(store, fingerprint(p, {'log': str(megatron_log(tmp_path / 't.log'))}), KNOBS, REF)
    assert pl['coverage']['micro_batch_size']['level'] == 'partial'
    assert pl['coverage']['attention_backend']['level'] == 'covered'     # kernel class: any GPU count


def test_7_constraint_ids_are_local_to_their_pack(tmp_path):
    a = head('a', {'kind': 'constraint', 'id': 'c1', 'scope': {}, 'when': {'nan_check': 'on'}, 'symptom': 'crash'})
    b = head('b', {'kind': 'constraint', 'id': 'c1', 'scope': {}, 'when': {'precision': 'fp8_delayed'}, 'symptom': 'crash'})
    pl = plan(Store([a, b]), fingerprint(project(), {'log': str(megatron_log(tmp_path / 't.log'))}), KNOBS, REF)
    assert pl['at_reference'] == ['a/c1'] and pl['check']({**REF, 'fp8': '1'})


def test_8_a_joint_effect_is_partial_context_not_a_size(tmp_path):
    joint = {'kind': 'effect', 'id': 'j', 'env': 'e', 'change': {'micro_batch': ['1', '2'], 'manual_gc': ['off', 'on']},
             'speedup': 1.05, 'method': 'paired_runs', 'base': {'tp': '1'}}
    st = Store([head('p', joint)])
    pl = plan(st, fingerprint(project(), {'log': str(megatron_log(tmp_path / 't.log'))}), KNOBS, REF)
    c = pl['coverage']['manual_gc']
    assert c['level'] == 'partial' and c['gain'] == 0.0 and c['joint']
    assert 'manual_gc' not in {s['knob'] for s in pl['seeds']}
    assert any(isinstance(e['change'], dict) and 'manual_gc' in e['change'] for e in pl['payload']['effects'])


def test_10_a_bundled_member_was_not_tried_alone():
    rec = {'assessment': [{'change': {'a': 1}, 'sign': '+', 'p_feasible': 0.1}]}
    sc = transfer.score(rec, [{'change': {'a': 1}, 'status': 'bundled', 'measured': None}], 0.002)
    assert sc['n_tried'] == 0 and sc['feasibility_brier'] is None


@pytest.fixture(scope='module')
def store():
    return Store.load()


def test_r10_1_a_member_measured_alone_still_scores():
    rec = {'assessment': [{'change': {'a': 1}, 'sign': '+', 'p_feasible': 0.1}]}
    sc = transfer.score(rec, [{'change': {'a': 1}, 'status': 'bundled', 'measured': 0.01}], 0.02)
    assert sc['n_tried'] == 1 and sc['feasibility_brier'] == pytest.approx(0.81)


def test_r10_2_the_citation_is_the_record_that_gave_the_sign(tmp_path):
    joint = head('near', {'kind': 'effect', 'id': 'j', 'env': 'e', 'speedup': 1.20, 'method': 'paired_runs',
                          'change': {'manual_gc': ['off', 'on'], 'micro_batch': ['1', '2']}, 'base': {'tp': '1'}})
    alone = head('far', {'kind': 'effect', 'id': 'g', 'env': 'e', 'speedup': 0.95, 'method': 'paired_runs',
                         'change': {'manual_gc': ['off', 'on']}, 'base': {'tp': '1'}})
    alone[1] = dict(alone[1], gpu='H20-96GB')
    fp = fingerprint(project(), {'log': str(megatron_log(tmp_path / 't.log', seq_length=16384))})
    c = plan(Store([joint, alone]), fp, KNOBS, REF)['coverage']['manual_gc']
    assert c['level'] == 'partial' and c['gain'] < 0 and c['evidence'] == ['far/g'] and c['nearest']['gpu'] == 'H20-96GB'


def test_r11_a_compound_record_is_context_for_the_frameworks_own_flag(tmp_path):
    rec = {'kind': 'effect', 'id': 'ov', 'env': 'e', 'speedup': 1 / 0.9, 'method': 'paired_runs', 'base': {'tp': '2'},
           'change': {'tp_comm_overlap': ['off', 'on']}, 'with': {'sequence_parallel': ['off', 'on']}}
    st = Store([head('p', rec)])
    fp = fingerprint(project(), {'log': str(megatron_log(tmp_path / 't.log', tensor_model_parallel_size=2))})
    own = {'overlap': {'type': 'bool', 'flag': '--tp-comm-overlap'}}               # the flag alone: no sequence parallel
    c = plan(st, fp, own, {'overlap': False})['coverage']['overlap']
    assert c['level'] == 'partial' and c['gain'] == 0.0 and c['joint']
    switch = {'overlap': {'type': 'str', 'choices': ['0', '1'], 'env': 'TP_OVERLAP', 'default': '0',
                          'mechanism': 'tp_comm_overlap', 'states': {'0': 'off', '1': 'on'}}}   # a script that sets both
    c = plan(st, fp, switch, {'overlap': '0'})['coverage']['overlap']
    assert c['level'] == 'covered' and c['gain'] == pytest.approx(0.1054, abs=1e-3)
