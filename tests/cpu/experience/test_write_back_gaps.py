"""Write-back gaps (2026-10-24, FABLE_DESIGN_CHECK 5.4): a candidate whose 2-update probe ran out of memory becomes an
`oom` observation (the memory look-up's evidence), ok observations carry their measured peak, and `init` drafts the
model size / shape and the sequence length from the model directory a non-Megatron job loads, so its runs write back."""
import json

import pytest

from expjob import KNOBS, REF, megatron_log, project
from infrarsi.experience.runs import write_back
from infrarsi.experience.store import Store, fingerprint, plan, read_lines
from infrarsi.project import detect


def test_probe_oom_becomes_an_observation_and_ok_rows_carry_their_peak(tmp_path, monkeypatch):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    proj = project()
    fp = fingerprint(proj, {'log': str(megatron_log(tmp_path / 'train.log'))})
    pl = plan(Store([]), fp, KNOBS, REF)
    tr = {'candidates': [{'id': 'c0', 'parent': None, 'change': None, 'valid': True, 'step_time': 3.0,
                          'peak_reserved_gib': 61.25},
                         {'id': 'c1', 'parent': 'c0', 'change': {'micro_batch_size': [1, 2]}, 'valid': False,
                          'status': 'infeasible'},
                         {'id': 'c2', 'parent': 'c0', 'change': {'manual_gc': [False, True]}, 'valid': False,
                          'status': 'infeasible'}],
          'screens': [{'candidate_id': 'c1', 'fidelity': 'probe', 'valid': False,
                       'reasons': ['out of memory: CUDA out of memory. Tried to allocate 4.00 GiB']},
                      {'candidate_id': 'c2', 'fidelity': 'probe', 'valid': False, 'reasons': ['exit code 1: Traceback']}],
          'extra': {}}
    lines = read_lines(write_back(proj, pl, tr, None, 'w1', 0.001))
    o = {x['id']: x for x in lines if x['kind'] == 'observation'}
    assert o['c1']['status'] == 'oom' and 'window_steps' not in o['c1']
    assert 'c2' not in o                                        # a crashed probe says nothing about memory
    assert o['c0']['x-torch-reserved-gib'] == 61.25 and 'mem_peak_gib' not in o['c0']   # the allocator's, not smi's
    assert o['c0']['window_steps'] == 20
    assert lines[0]['summary'].endswith('1 configurations at full fidelity, 1 out of memory at a probe')
    again = plan(Store.load(include=['run-w1']), fp, KNOBS, REF)     # the next run on this card: probe-first
    assert any(w.startswith('memory: run-w1/c1') for w in again['check']({**REF, 'micro_batch_size': 2}))
    assert not again['check'](REF)


def _safetensors(path, tensors: dict):
    head = json.dumps({k: {'dtype': 'BF16', 'shape': s, 'data_offsets': [0, 0]} for k, s in tensors.items()}).encode()
    path.write_bytes(len(head).to_bytes(8, 'little') + head)


HF_TRAIN = ("import argparse\nimport torch.distributed as dist\nfrom transformers import AutoModelForCausalLM\n"
            "def main():\n    ap = argparse.ArgumentParser()\n    ap.add_argument('--model_name_or_path', default='ckpt/tiny')\n"
            "    ap.add_argument('--max_seq_length', type=int, default=2048)\n    ap.add_argument('--max-steps', type=int)\n"
            "    ap.add_argument('--micro-batch-size', type=int, default=4)\n    dist.init_process_group('nccl')\n"
            "if __name__ == '__main__':\n    main()\n")


def test_init_drafts_model_and_seq_len_from_the_model_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(detect, 'gpu_facts', lambda: {'gpus': 8, 'memory_gib': 77.3, 'gpu_type': 'NVIDIA H100 80GB HBM3'})
    (tmp_path / 'train.py').write_text(HF_TRAIN, encoding='utf-8')
    m = tmp_path / 'ckpt' / 'tiny'
    m.mkdir(parents=True)
    (m / 'config.json').write_text(json.dumps({'model_type': 'qwen2', 'hidden_size': 896, 'num_hidden_layers': 24,
                                               'intermediate_size': 4864, 'num_attention_heads': 14,
                                               'num_key_value_heads': 2, 'vocab_size': 151936,
                                               'tie_word_embeddings': True}), encoding='utf-8')
    _safetensors(m / 'model.safetensors', {'embed': [151936, 896], 'layer.w': [24, 896, 4864], 'norm': [896]})
    man, notes = detect.draft(tmp_path)
    exact = (151936 * 896 + 24 * 896 * 4864 + 896) / 1e9
    assert man['model']['params_b'] == pytest.approx(round(exact, 3))
    assert man['model']['layers'] == 24 and man['model']['kv_heads'] == 2 and man['model']['vocab'] == 151936
    assert man['workload']['seq_len'] == 2048
    assert any(n.startswith('model: ') and 'safetensors headers' in n for n in notes)


def test_init_names_what_is_missing_when_the_model_is_not_here(tmp_path, monkeypatch):
    monkeypatch.setattr(detect, 'gpu_facts', lambda: {'gpus': 8, 'memory_gib': 77.3, 'gpu_type': 'NVIDIA H100 80GB HBM3'})
    (tmp_path / 'train.py').write_text(HF_TRAIN.replace("'ckpt/tiny'", "'/nowhere/model'"), encoding='utf-8')
    man, notes = detect.draft(tmp_path)
    assert 'model' not in man and man['workload']['seq_len'] == 2048
    assert any(n.startswith('Add `model: {params_b: ') and 'does not write its measurements back' in n for n in notes)   # (U-1-4)


def _pipe(tmp_path, cls, patch, files=('diffusion_pytorch_model.safetensors',)):
    d = tmp_path / 'pipe'
    (d / 'transformer').mkdir(parents=True)
    (d / 'model_index.json').write_text('{}', encoding='utf-8')
    (d / 'transformer' / 'config.json').write_text(json.dumps({'_class_name': cls, 'patch_size': patch, 'num_layers': 28}),
                                                   encoding='utf-8')
    for f in files:
        _safetensors(d / 'transformer' / f, {'w': [1000, 675]})
    return d


def test_diffusion_transformer_seq_len_is_its_latent_tokens(tmp_path):
    _pipe(tmp_path, 'DiTTransformer2DModel', 2, files=('diffusion_pytorch_model.safetensors',
                                                       'diffusion_pytorch_model.fp16.safetensors'))
    model, seq, src = detect.model_facts(tmp_path, {'model_path': 'pipe', 'resolution': 256})
    assert model == {'params_b': 0.001} and seq == 256 and 'safetensors' in src   # the fp16 copy is not counted twice


def test_flux_packs_two_by_two_latents_and_an_unknown_class_gets_no_seq_len(tmp_path):
    _pipe(tmp_path, 'FluxTransformer2DModel', 1)
    assert detect.model_facts(tmp_path, {'model_path': 'pipe', 'resolution': 1024})[1] == 4096
    other = tmp_path / 'o'
    other.mkdir()
    _pipe(other, 'SomeNewTransformer', 2)
    assert detect.model_facts(other, {'model_path': 'pipe', 'resolution': 1024})[1] is None


def test_verifier_worded_probe_oom_and_a_candidates_own_oom_wins_over_the_reference_peak(tmp_path, monkeypatch):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    proj = project()
    fp = fingerprint(proj, {'log': str(megatron_log(tmp_path / 'train.log'))})
    tr = {'candidates': [{'id': 'c0', 'parent': None, 'change': None, 'valid': True, 'step_time': 3.0,
                          'peak_reserved_gib': 61.25},
                         {'id': 'c1', 'parent': 'c0', 'change': {'cross_entropy_loss_fusion': [False, True]},
                          'valid': False, 'status': 'infeasible'}],
          'screens': [{'candidate_id': 'c1', 'fidelity': 'probe', 'valid': False,      # as the verifier words them
                       'reasons': ['completes: exit code 1', 'no_errors: 1 out-of-memory line(s); 2 traceback line(s)']}],
          'extra': {}}
    o = {x['id']: x for x in read_lines(write_back(proj, plan(Store([]), fp, KNOBS, REF), tr, None, 'w2', 0.001))
         if x['kind'] == 'observation'}
    assert o['c1']['status'] == 'oom'
    again = plan(Store.load(include=['run-w2']), fp, KNOBS, REF)     # c0 has the same shape and a peak that fits
    assert any(w.startswith('memory: run-w2/c1') for w in again['check']({**REF, 'cross_entropy_loss_fusion': True}))


def test_checkpoint_sets_shards_and_revisions(tmp_path, monkeypatch):
    d = tmp_path / 'm'
    d.mkdir()
    _safetensors(d / 'model-00001-of-00002.safetensors', {'a': [10, 10]})
    assert detect.safetensors_params(d) is None                  # shard 2 of 2 missing
    _safetensors(d / 'model-00002-of-00002.safetensors', {'b': [5]})
    assert detect.safetensors_params(d) == 105
    (d / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'model-00001-of-00002.safetensors',
                                                                               'b': 'gone.safetensors'}}), encoding='utf-8')
    assert detect.safetensors_params(d) is None                  # the index names a file that is not there
    hub = tmp_path / 'hub'
    for h, layers in (('aaaa', 24), ('ffff', 2)):                # hash order is not the revision order
        s = hub / 'models--org--tiny' / 'snapshots' / h
        s.mkdir(parents=True)
        (s / 'config.json').write_text(json.dumps({'model_type': 'qwen2', 'hidden_size': 64, 'num_hidden_layers': layers}),
                                       encoding='utf-8')
    (hub / 'models--org--tiny' / 'refs').mkdir()
    (hub / 'models--org--tiny' / 'refs' / 'main').write_text('aaaa', encoding='utf-8')
    monkeypatch.setenv('HF_HUB_CACHE', str(hub))
    assert detect.model_facts(tmp_path, {'model_name_or_path': 'org/tiny'})[0]['layers'] == 24
    (hub / 'models--org--tiny' / 'refs' / 'v2').write_text('ffff', encoding='utf-8')
    assert detect.model_facts(tmp_path, {'model_name_or_path': 'org/tiny', 'revision': 'v2'})[0]['layers'] == 2
    assert detect.model_facts(tmp_path, {'model_name_or_path': 'org/tiny', 'revision': 'ffff'})[0]['layers'] == 2
    assert detect.model_facts(tmp_path, {'model_name_or_path': 'org/tiny', 'revision': 'v3'}) == ({}, None, None)


def test_a_pre_quantized_checkpoint_is_estimated_from_its_shape(tmp_path):
    q = tmp_path / 'q4'
    q.mkdir()
    (q / 'config.json').write_text(json.dumps({'model_type': 'qwen3', 'hidden_size': 4096, 'num_hidden_layers': 36,
                                               'intermediate_size': 12288, 'num_attention_heads': 32,
                                               'num_key_value_heads': 8, 'head_dim': 128, 'vocab_size': 151936,
                                               'tie_word_embeddings': False,
                                               'quantization_config': {'quant_method': 'bitsandbytes', 'load_in_4bit': True}}),
                                   encoding='utf-8')
    _safetensors(q / 'model.safetensors', {'packed': [1000, 2]})            # packed elements: not the parameter count
    m, _, src = detect.model_facts(tmp_path, {'model_path': 'q4'})
    assert 8.0 < m['params_b'] < 8.4 and 'estimate' in src


def test_moe_family_and_no_estimate_for_a_two_matrix_mlp(tmp_path):
    moe = tmp_path / 'moe'
    moe.mkdir()
    (moe / 'config.json').write_text(json.dumps({'model_type': 'qwen3_moe', 'hidden_size': 2048, 'num_hidden_layers': 48,
                                                 'moe_intermediate_size': 768, 'num_attention_heads': 32,
                                                 'num_key_value_heads': 4, 'vocab_size': 151936, 'num_experts': 128,
                                                 'num_experts_per_tok': 8}), encoding='utf-8')
    m = detect.model_facts(tmp_path, {'model_path': 'moe'})[0]
    assert m['family'] == 'moe' and m['experts'] == 128 and m['topk'] == 8 and m['params_b'] > 25
    neox = tmp_path / 'neox'
    neox.mkdir()
    (neox / 'config.json').write_text(json.dumps({'model_type': 'gpt_neox', 'hidden_size': 6144, 'num_hidden_layers': 44,
                                                  'intermediate_size': 24576, 'num_attention_heads': 64,
                                                  'vocab_size': 50432}), encoding='utf-8')
    assert 'params_b' not in detect.model_facts(tmp_path, {'model_path': 'neox'})[0]
