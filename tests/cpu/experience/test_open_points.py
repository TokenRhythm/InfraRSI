"""The harvest session's open points for the experience layer (reports/h20_prior_20261021/experience/README.md): fabric,
unregistered mechanisms, the memory look-up's scale, rule order under the cap, version keys, fp16 in the argument dump."""
import pytest

from expjob import KNOBS, REF, megatron_log, project
from infrarsi.experience.store import Store, fingerprint, megatron_states, plan, states_of, labels, versions


def pack(pid, env, *rest):
    return [{'kind': 'pack', 'id': 'pack', 'schema': 'infrarsi-exp/1', 'pack_id': pid, 'title': pid, 'license': 'CC0',
             'tier': 'measured'},
            {'kind': 'env', 'id': 'e', 'gpu': 'H200-141GB', 'gpus_per_node': 8, 'nodes': 1, 'framework': 'megatron',
             'model': {'params_b': 8.0}, 'workload': {'seq_len': 4096}, **env}, *rest]


def eff(mech, a, b, speedup, **base):
    return {'kind': 'effect', 'id': f'{mech}-{a}-{b}', 'env': 'e', 'change': {mech: [a, b]}, 'speedup': speedup,
            'method': 'paired_runs', 'base': {'tp': '1', **base}}


def two_node(tmp_path, inter):
    p = project()
    p.nodes = 2
    if inter:
        p.raw['hardware']['interconnect'] = {'intra': 'nvlink', 'inter': inter}
    return fingerprint(p, {'log': str(megatron_log(tmp_path / 't.log'))})


def test_1_balance_evidence_across_nodes_needs_the_same_fabric(tmp_path):
    st = Store([pack('tcp', {'nodes': 2, 'interconnect': {'intra': 'nvlink', 'inter': 'tcp'}},
                     eff('dp_grad_overlap', 'off', 'on', 1.2),
                     {'kind': 'rule', 'id': 'r', 'scope': {'interconnect.inter': 'tcp'}, 'text': 'over TCP: ...'})])
    levels = {}
    for inter in (None, 'tcp', 'ib'):
        fp = two_node(tmp_path, inter)
        pl = plan(st, fp, KNOBS, REF)
        levels[inter] = pl['coverage']['overlap_grad_reduce']['level']
        assert (inter == 'tcp') == bool(pl['payload']['rules'])     # a TCP rule reaches a TCP target only
    assert levels == {None: 'partial', 'tcp': 'covered', 'ib': 'partial'}


def test_2_an_unregistered_mechanism_is_context_not_a_prior(tmp_path):
    knobs = {**KNOBS, 'fp8_attn': {'type': 'bool', 'flag': '--fp8-dot-product-attention', 'mechanism': 'x-fp8-attention',
                                   'states': {'false': 'off', 'true': 'on'}}}
    st = Store([pack('p', {}, eff('x-fp8-attention', 'off', 'on', 1.06))])
    pl = plan(st, fingerprint(project(knobs=knobs), {'log': str(megatron_log(tmp_path / 't.log'))}), knobs,
              {**REF, 'fp8_attn': False})
    c = pl['coverage']['fp8_attn']
    assert c['level'] == 'partial' and c['gain'] == 0.0 and c['unregistered']
    assert 'fp8_attn' not in {s['knob'] for s in pl['seeds']}


def test_3_the_memory_lookup_matches_nodes_and_expert_parallelism(tmp_path):
    obs = {'kind': 'observation', 'id': 'o', 'env': 'e', 'status': 'ok', 'step_time_s': 3.0, 'mem_peak_gib': 120.0,
           'config': {'tp': '1', 'pp': '1', 'cp': '1', 'ep': '1', 'micro_batch': '2', 'precision': 'bf16',
                      'recompute': 'none', 'bf16_main_grads': 'fp32', 'distributed_optimizer': 'on'}}
    st = Store([pack('p', {}, obs)])
    labs = labels(KNOBS, 'megatron')
    one = fingerprint(project(), {'log': str(megatron_log(tmp_path / 'a.log'))})
    assert st.fits(one, states_of(one, labs, {**REF, 'micro_batch_size': 2}))['fits'] is True
    ep2 = fingerprint(project(), {'log': str(megatron_log(tmp_path / 'b.log', expert_model_parallel_size=2))})
    assert st.fits(ep2, states_of(ep2, labs, {**REF, 'micro_batch_size': 2})) is None
    p = project()
    p.nodes, p.gpus = 2, 4                                            # the same 8 GPUs on two nodes
    two = fingerprint(p, {'log': str(megatron_log(tmp_path / 'c.log'))})
    assert st.fits(two, states_of(two, labs, {**REF, 'micro_batch_size': 2})) is None


def test_4_the_most_specific_and_confident_rules_come_first(tmp_path):
    rules = [{'kind': 'rule', 'id': f'r{i}', 'scope': scope, 'confidence': conf, 'text': f'rule {i}'}
             for i, (scope, conf) in enumerate([({}, 'high'), ({'framework': 'megatron'}, 'low'),
                                                ({'framework': 'megatron', 'gpu': ['H200-141GB']}, 'medium'),
                                                ({'framework': 'megatron'}, 'high')])]
    pl = plan(Store([pack('p', {}, *rules)]), fingerprint(project(), {'log': str(megatron_log(tmp_path / 't.log'))}), KNOBS, REF)
    assert [r['id'] for r in pl['payload']['rules']] == ['p/r2', 'p/r3', 'p/r1', 'p/r0']


def test_5_version_keys_are_normalized():
    assert versions({'transformer_engine': '2.3.0', 'megatron_core': '0.13.0', 'cudnn': 90100, 'torch': '2.6.0'},
                    'megatron') == {'te': '2.3.0', 'framework': '0.13.0', 'cudnn': '9.1.0', 'torch': '2.6.0'}
    assert versions({'cudnn': '8902'}, 'fsdp') == {'cudnn': '8.9.2'}
    assert versions({'deepspeed': '0.19.8'}, 'deepspeed') == {'deepspeed': '0.19.8', 'framework': '0.19.8'}


def test_6_fp16_is_read_from_the_argument_dump():
    assert megatron_states({'fp16': 'True', 'bf16': 'False', 'fp8': 'None'})['precision'] == 'fp16'
    assert megatron_states({'fp16': 'False', 'bf16': 'True', 'fp8': 'None'})['precision'] == 'bf16'
    assert megatron_states({'bf16': 'True', 'fp8': 'hybrid', 'fp8_recipe': 'delayed'})['precision'] == 'fp8_delayed'


def test_r12_1_a_written_back_two_node_run_keeps_its_fabric(tmp_path, monkeypatch):
    from infrarsi.experience.runs import write_back
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    p = project()
    p.nodes = 2
    p.raw['hardware']['interconnect'] = {'intra': 'nvlink', 'inter': 'tcp'}
    fp = fingerprint(p, {'log': str(megatron_log(tmp_path / 't.log'))})
    tr = {'candidates': [{'id': 'c0', 'parent': None, 'change': None, 'valid': True, 'step_time': 3.0},
                         {'id': 'c1', 'parent': 'c0', 'change': {'overlap_grad_reduce': [False, True]}, 'valid': True}],
          'screens': [{'candidate_id': 'c1', 'fidelity': 'full', 'valid': True, 'step_time': 2.7}], 'extra': {}}
    write_back(p, plan(Store([]), fp, KNOBS, REF), tr, None, 'two', 0.001)
    pl = plan(Store.load(include=['run-two']), fp, KNOBS, REF)
    assert pl['coverage']['overlap_grad_reduce']['level'] == 'covered'


def test_r12_2_a_state_outside_the_registry_is_unknown():
    from infrarsi.experience.registry import label
    lab = label('replicate', {'type': 'int', 'choices': [1, 2], 'flag': '--parallelism.data-parallel-replicate-degree'}, 'torchtitan')
    assert lab is None                                               # no scalar alias for hsdp any more
    named = label('hsdp', {'type': 'int', 'choices': [1, 2]}, 'torchtitan')
    assert named.state(2) is None                                    # '2' is not one of none / 2x4 / 2x8
    mapped = label('replicate', {'type': 'int', 'choices': [1, 2], 'mechanism': 'hsdp', 'states': {1: 'none', 2: '2x4'}}, 'torchtitan')
    assert mapped.state(2) == '2x4' and mapped.state(1) == 'none'


def test_r13_a_value_without_a_registry_state_is_not_written_as_the_reference(tmp_path, monkeypatch):
    from infrarsi.experience.runs import write_back
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    knobs = {'attn': {'type': 'str', 'choices': ['auto', 'fused'], 'default': 'auto', 'flag': '--attention-backend',
                      'mechanism': 'attention_backend'},                  # 'fused' is not a state: cudnn would be
             'gc': {'type': 'bool', 'flag': '--manual-gc'}}
    p = project(knobs=knobs)
    fp = fingerprint(p, {'log': str(megatron_log(tmp_path / 't.log'))})
    labs = labels(knobs, 'megatron')
    assert states_of(fp, labs, {'attn': 'fused', 'gc': True})['attention_backend'] == '?'
    tr = {'candidates': [{'id': 'c0', 'parent': None, 'change': None, 'valid': True, 'step_time': 3.0},
                         {'id': 'c1', 'parent': 'c0', 'change': {'attn': ['auto', 'fused'], 'gc': [False, True]}, 'valid': True}],
          'screens': [{'candidate_id': 'c1', 'fidelity': 'full', 'valid': True, 'step_time': 2.7}], 'extra': {}}
    write_back(p, plan(Store([]), fp, knobs, {'attn': 'auto', 'gc': False}), tr, None, 'r13', 0.001)
    st = Store.load(include=['run-r13'])
    assert [o['id'] for o in st.observations] == ['c0'] and st.effects == []
