"""The experience store on the shipped seeds (experience decision, sections 1-2 and 6.4): labels, mechanism states,
coverage levels for one job on three GPU classes, the memory look-up, constraints, the start composite, pack loading."""
import json

import pytest

from expjob import KNOBS, REF, megatron_log, project
from infrarsi.experience.pack import derive_effects, validate, write_lines
from infrarsi.experience.registry import label
from infrarsi.experience.store import UNKNOWN, Store, fingerprint, labels, plan, states_of


@pytest.fixture(scope='module')
def store():
    return Store.load()


@pytest.fixture
def log(tmp_path):
    return {'log': str(megatron_log(tmp_path / 'train.log'))}


def levels(pl):
    out = {}
    for k, c in pl['coverage'].items():
        out.setdefault(c['level'], set()).add(k)
    return out


def test_every_knob_of_the_job_is_labelled():
    labs = labels(KNOBS, 'megatron')
    assert all(labs[k] is not None for k in KNOBS)
    assert labs['nan_check_off'].mechanism == 'nan_check' and labs['nan_check_off'].state(True) == 'off'
    assert labs['nan_check_off'].own and not labs['bf16_main_grads'].own      # a framework flag vs a project switch
    assert labs['fp8'].state('1') == 'fp8_delayed' and labs['NCCL_NVLS_ENABLE'].state('0') == 'off'
    assert label('learning_rate', {'type': 'float', 'min': 1e-5, 'max': 1e-3}, 'megatron') is None


def test_fingerprint_from_the_argument_dump(log):
    fp = fingerprint(project(), log)
    assert fp['gpu'] == 'H200-141GB' and fp['model'] == {'params_b': 8.03, 'family': 'dense', 'layers': 32, 'hidden': 4096,
                                                         'ffn': 14336, 'heads': 32, 'kv_heads': 8, 'head_dim': 128,
                                                         'vocab': 128256, 'untie': True}   # the shape: the accounting's
    assert fp['workload'] == {'seq_len': 4096, 'global_batch_tokens': 262144, 'data': 'mock'}
    assert fp['layout'] == {'tp': '1', 'pp': '1', 'cp': '1', 'ep': '1'}
    assert fp['states']['bf16_main_grads'] == 'fp32' and fp['states']['precision'] == 'bf16'


def test_a_project_switch_may_set_what_it_requires(log):
    fp = fingerprint(project(), log)
    labs = labels(KNOBS, 'megatron')
    assert states_of(fp, labs, {**REF, 'bf16_main_grads': '1'})['precision_aware_optimizer'] == UNKNOWN
    assert states_of(fp, labs, {**REF, 'bf16_main_grads': '1'}, ok=True)['precision_aware_optimizer'] == 'on'
    own = labels({'grads': {'type': 'str', 'choices': ['fp32', 'bf16'], 'flag': '--main-grads-dtype'}}, 'megatron')
    assert states_of(fp, own, {'grads': 'bf16'})['precision_aware_optimizer'] == 'off'   # the flag alone sets no more
    assert states_of(fp, {}, {'mystery': True})['x-mystery'] == 'true'


def test_coverage_on_the_h200_class_node(store, log):
    pl = plan(store, fingerprint(project(), log), KNOBS, REF)
    lv = levels(pl)
    assert lv['covered'] == {'micro_batch_size', 'bf16_main_grads', 'attention_backend', 'fp8', 'recompute_activations',
                             'cross_entropy_loss_fusion', 'nan_check_off', 'overlap_grad_reduce'}
    assert lv['null'] == {'manual_gc', 'NCCL_NVLS_ENABLE', 'CUDA_DEVICE_MAX_CONNECTIONS', 'PYTORCH_CUDA_ALLOC_CONF'}
    assert lv['partial'] == {'overlap_param_gather'}                  # measured only on TP2, together with grad overlap
    assert lv['none'] == {'TORCH_NCCL_AVOID_RECORD_STREAMS', 'empty_unused_memory_level', 'cross_entropy_fusion_impl',
                          'ddp_average_in_collective', 'num_workers'}
    cov = pl['coverage']
    assert cov['micro_batch_size']['evidence'][0].endswith('/pb-micro_batch')   # InfraRSI's own run before the harvest
    assert cov['attention_backend']['gain'] < 0 and cov['bf16_main_grads']['gain'] == pytest.approx(0.043, abs=1e-4)


def test_coverage_moves_with_the_gpu_class(store, log):
    h20 = levels(plan(store, fingerprint(project('NVIDIA H20', 96), log), KNOBS, REF))
    assert {'attention_backend', 'bf16_main_grads', 'cross_entropy_loss_fusion', 'nan_check_off'} <= h20['covered']
    assert 'overlap_grad_reduce' in h20['partial']                   # balance class: the H200 evidence is elsewhere
    a100 = levels(plan(store, fingerprint(project('NVIDIA A100-SXM4-80GB', 80), log), KNOBS, REF))
    assert {'attention_backend', 'bf16_main_grads'} <= a100['covered']  # kernel class: the sign transfers
    assert {'fp8', 'micro_batch_size', 'overlap_grad_reduce'} <= a100['partial']
    other = levels(plan(store, fingerprint(project(framework='deepspeed'), log), KNOBS, REF))
    assert set(other) == {'none'}


def test_memory_lookup(store, log):
    labs = labels(KNOBS, 'megatron')
    mbs2 = {**REF, 'micro_batch_size': 2, 'cross_entropy_loss_fusion': True}   # HX06's memory regime
    big, small = fingerprint(project(), log), fingerprint(project('NVIDIA H20', 96), log)
    assert store.fits(big, states_of(big, labs, mbs2))['fits'] is True
    assert store.fits(small, states_of(small, labs, mbs2))['fits'] is False       # 123 GiB peak on the H200 node
    assert store.fits(small, states_of(small, labs, REF))['fits'] is True
    assert store.fits(small, states_of(small, labs, {**REF, 'recompute_activations': True})) is None   # no such shape


def test_constraints_and_the_start_composite(store, log):
    fp = fingerprint(project(), log)
    pl = plan(store, fp, KNOBS, REF)
    warn = {s['knob']: s['warnings'] for s in pl['seeds'] if s['warnings']}
    assert set(warn) == {'fp8'} and any(w.startswith('fp8-mock-loss') for w in warn['fp8'])
    assert pl['composite']['members'] == ['bf16_main_grads', 'micro_batch_size', 'cross_entropy_loss_fusion',
                                          'overlap_grad_reduce']                # fp8 left out: known to fail here
    assert not pl['check']({**REF, 'bf16_main_grads': '1'})           # the switch sets the optimizer it needs
    assert not pl['check']({**REF, 'empty_unused_memory_level': 1})   # a 'slow' constraint orders, never warns
    own = {**KNOBS, 'grads': {'type': 'str', 'choices': ['fp32', 'bf16'], 'flag': '--main-grads-dtype', 'default': 'fp32'}}
    pl2 = plan(store, fp, own, {**REF, 'grads': 'fp32'})
    assert any(w.startswith('bf16-grads-needs-pao') for w in pl2['check']({**REF, 'grads': 'bf16'}))
    unknown = plan(store, fingerprint(project(versions={}), log), KNOBS, REF)
    assert not any(w.startswith('bf16-grads') for w in unknown['check']({**REF, 'grads': 'bf16'}))


def test_payload_is_small_and_names_the_differences(store, log):
    pl = plan(store, fingerprint(project('NVIDIA H20', 96), log), KNOBS, REF)
    text = json.dumps(pl['payload'], default=str)
    assert len(text) <= 4 * 3000 + 2000
    near = pl['payload']['nearest'][0]
    assert near['gpu'] == 'H20-96GB' or any(d.startswith('gpu ') for d in near['differences'])


def pack(pid, *rest, tier='measured'):
    return [{'kind': 'pack', 'id': 'pack', 'schema': 'infrarsi-exp/1', 'pack_id': pid, 'title': pid, 'license': 'CC0',
             'tier': tier},
            {'kind': 'env', 'id': 'e', 'gpu': 'H200', 'gpus_per_node': 8, 'nodes': 1, 'framework': 'megatron',
             'model': {'params_b': 8}, 'workload': {'seq_len': 4096}}, *rest]


def test_loading_skips_bad_and_repeated_packs(tmp_path, monkeypatch):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path))
    d = tmp_path / 'experience'
    d.mkdir()
    write_lines(d / 'a.jsonl', pack('mine-a'))
    write_lines(d / 'b.jsonl', pack('mine-a'))
    write_lines(d / 'c.jsonl', pack('mine-c')[:1] + [{'kind': 'effect', 'id': 'x', 'env': 'nowhere'}])
    st = Store.load(include=['mine-*'])
    assert [p['pack_id'] for p in st.packs] == ['mine-a'] and len(st.skipped) == 2
    assert [p['pack_id'] for p in Store.load(include=['h20-*']).packs] == ['h20-harvest-2026-10']


def test_a_compound_move_derives_as_one_mechanism():
    obs = [{'kind': 'observation', 'id': 'a', 'env': 'e', 'config': {'bf16_main_grads': 'fp32'}, 'status': 'ok', 'step_time_s': 1.0},
           {'kind': 'observation', 'id': 'b', 'env': 'e', 'status': 'ok', 'step_time_s': 0.95,
            'config': {'bf16_main_grads': 'bf16', 'precision_aware_optimizer': 'on'}},
           {'kind': 'observation', 'id': 'c', 'env': 'e', 'status': 'ok', 'step_time_s': 0.9,
            'config': {'bf16_main_grads': 'fp32', 'manual_gc': 'on', 'nvls': 'off'}}]
    lines = pack('p', *obs)
    assert validate(lines) == ([], [])
    eff = derive_effects(lines)
    assert len(eff) == 1 and eff[0]['change'] == {'bf16_main_grads': ['fp32', 'bf16']}
    assert eff[0]['with'] == {'precision_aware_optimizer': ['off', 'on']} and eff[0]['speedup'] == pytest.approx(1 / 0.95)


def test_the_seeds_are_clean(store):
    import re
    assert not [o['id'] for o in store.observations if re.search('contaminated|invalid|dup', o['id'])]
    assert len(store.envs) >= 80 and len(store.effects) >= 250 and len(store.constraints) >= 20
