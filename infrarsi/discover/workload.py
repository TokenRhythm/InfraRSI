"""workload_facts(task): model/data facts from the HF config.json (+ safetensors index metadata).

Parameter counts follow each architecture's weight layout (gated MLPs, GQA/MLA/GDN attention, routed + shared
experts, PLE n-gram tables). params_total is the model as trained (language model only; MTP only where the task
trains it); params_checkpoint is the index total_size / 2 when present, for cross-checking.
"""
from __future__ import annotations

import json
from pathlib import Path

from infrarsi.contracts import EXECUTION_DIR, RemotePaths, TaskSpec, utc_now

CACHE = EXECUTION_DIR / 'discover' / 'hf_configs'


def model_dir(task: TaskSpec) -> str:
    repo = task.raw['model']['repo_id']
    if repo.startswith('local:'):
        return repo[len('local:'):]
    return f'{RemotePaths.models}/{repo.replace("/", "__")}'


def cache_name(task: TaskSpec) -> str:
    return model_dir(task).rstrip('/').split('/')[-1] + '.json'


def text_config(cfg: dict) -> dict:
    return cfg.get('text_config') or cfg


def _layer_kinds(c: dict, n: int) -> list[str]:
    if c.get('layer_types'):
        return ['linear' if t == 'linear_attention' else 'full' for t in c['layer_types']][:n]
    if c.get('full_attention_interval') and c.get('linear_num_key_heads'):
        k = int(c['full_attention_interval'])
        return ['full' if (i + 1) % k == 0 else 'linear' for i in range(n)]
    return ['full'] * n


def _moe_layers(c: dict, mtype: str, n: int) -> list[bool]:
    if mtype == 'deepseek_v2':
        first, freq = int(c.get('first_k_dense_replace', 0)), int(c.get('moe_layer_freq', 1))
        return [i >= first and i % freq == 0 for i in range(n)]
    experts = c.get('num_experts') or c.get('n_routed_experts')
    if not experts:
        return [False] * n
    only = set(c.get('mlp_only_layers') or [])
    step = int(c.get('decoder_sparse_step', 1) or 1)
    return [i not in only and (i + 1) % step == 0 for i in range(n)]


def count_params(cfg: dict, *, include_mtp: bool = False) -> dict:
    c = text_config(cfg)
    mtype = c.get('model_type') or cfg.get('model_type')
    H = int(c['hidden_size'])
    n = int(c.get('num_hidden_layers') or c.get('num_layers'))
    V = int(c.get('vocab_size') or c.get('padded_vocab_size'))
    nq = int(c['num_attention_heads'])
    nkv = int(c.get('num_key_value_heads') or c.get('multi_query_group_num') or nq)
    hd = int(c.get('head_dim') or c.get('kv_channels') or H // nq)
    I = int(c.get('intermediate_size') or c.get('ffn_hidden_size') or 4 * H)
    tied = bool(cfg.get('tie_word_embeddings', c.get('tie_word_embeddings', False)))
    kinds = _layer_kinds(c, n)
    moe = _moe_layers(c, mtype, n)
    E = int(c.get('num_experts') or c.get('n_routed_experts') or 0)
    topk = int(c.get('num_experts_per_tok') or 0)
    Im = int(c.get('moe_intermediate_size') or 0)
    if mtype == 'deepseek_v2':
        Ish = int(c.get('n_shared_experts') or 0) * Im
    else:
        Ish = int(c.get('shared_expert_intermediate_size') or 0)
    gated_q = mtype in ('qwen3_next', 'qwen4_exp', 'qwen4_exp_text')       # output-gated full attention

    def attn_full():
        if mtype == 'deepseek_v2':
            nope, rope, vh = int(c['qk_nope_head_dim']), int(c['qk_rope_head_dim']), int(c['v_head_dim'])
            kvl, ql = int(c['kv_lora_rank']), c.get('q_lora_rank')
            q = H * nq * (nope + rope) if not ql else H * int(ql) + int(ql) + int(ql) * nq * (nope + rope)
            return q + H * (kvl + rope) + kvl + kvl * nq * (nope + vh) + nq * vh * H
        q = H * nq * hd * (2 if gated_q else 1)
        p = q + 2 * H * nkv * hd + nq * hd * H
        if mtype in ('qwen3', 'qwen3_moe', 'qwen3_next', 'qwen4_exp', 'qwen4_exp_text'):
            p += 2 * hd                                                      # q/k norms
        if c.get('add_qkv_bias') or c.get('attention_bias'):
            p += nq * hd + 2 * nkv * hd
        return p

    def attn_linear():
        kd = int(c['linear_num_key_heads']) * int(c['linear_key_head_dim'])
        vh = int(c['linear_num_value_heads'])
        vd = vh * int(c['linear_value_head_dim'])
        conv = (2 * kd + vd) * int(c.get('linear_conv_kernel_dim', 4))
        return H * (2 * kd + 2 * vd) + H * 2 * vh + conv + 2 * vh + int(c['linear_value_head_dim']) + vd * H

    def moe_block():
        experts = E * 3 * H * Im
        shared = 3 * H * Ish + (H if Ish and mtype != 'deepseek_v2' else 0)
        return experts, shared + H * E

    br = {'embedding': V * H, 'output': 0 if tied else V * H, 'attention': 0, 'dense_mlp': 0, 'expert': 0,
          'moe_other': 0, 'norms': 0}
    active_moe = 0
    for i in range(n):
        br['attention'] += attn_full() if kinds[i] == 'full' else attn_linear()
        br['norms'] += 2 * H
        if moe[i]:
            ex, other = moe_block()
            br['expert'] += ex
            br['moe_other'] += other
            active_moe += topk * 3 * H * Im
        else:
            ff = I if mtype != 'deepseek_v2' else int(c['intermediate_size'])
            br['dense_mlp'] += 3 * H * ff
    br['norms'] += H
    extras = {}
    if include_mtp and int(c.get('mtp_num_hidden_layers') or 0) > 0:
        ex, other = moe_block()
        extras['mtp'] = int(c['mtp_num_hidden_layers']) * (attn_full() + ex + other + 2 * H * H + 4 * H)
        extras['mtp_expert'] = int(c['mtp_num_hidden_layers']) * ex
        active_moe += int(c['mtp_num_hidden_layers']) * topk * 3 * H * Im
    if c.get('ngram_vocab_size_base') and c.get('ple_embed_dim'):
        extras['ple'] = int(c['ngram_vocab_size_base']) * int(c['ple_embed_dim'])
    total = sum(br.values()) + extras.get('mtp', 0)
    expert = br['expert'] + extras.get('mtp_expert', 0)
    active = total - expert + active_moe
    return {'model_type': mtype, 'hidden': H, 'layers': n, 'vocab': V, 'heads': nq, 'kv_heads': nkv, 'head_dim': hd,
            'intermediate': I, 'tied_embeddings': tied, 'layer_kinds': {'full': kinds.count('full'), 'linear': kinds.count('linear')},
            'moe': ({'num_experts': E, 'topk': topk, 'moe_intermediate': Im, 'shared_intermediate': Ish,
                     'moe_layers': sum(moe), 'dense_layers': n - sum(moe)} if E else None),
            'linear_heads': ({'key': int(c['linear_num_key_heads']), 'value': int(c['linear_num_value_heads']),
                              'key_dim': int(c['linear_key_head_dim']), 'value_dim': int(c['linear_value_head_dim'])}
                             if c.get('linear_num_key_heads') else None),
            'breakdown': br, 'extras': extras, 'params_total': total, 'params_expert': expert,
            'params_dense': total - expert, 'params_active': active, 'cpu_resident_params': extras.get('ple', 0)}


def tp_candidates(facts: dict, gpus_per_node: int = 8) -> list[int]:
    """Powers of two <= gpus_per_node that divide attention heads, KV groups and linear-attention heads."""
    out = []
    for tp in (1, 2, 4, 8):
        if tp > gpus_per_node:
            break
        if facts['heads'] % tp or facts['kv_heads'] % tp:
            continue
        lh = facts.get('linear_heads')
        if lh and (lh['key'] % tp or lh['value'] % tp):
            continue
        out.append(tp)
    return out


def facts_from_config(task: TaskSpec, cfg: dict, index_metadata: dict | None = None) -> dict:
    flagship = task.raw['model'].get('bridge_type') == 'qwen4_exp'
    f = count_params(cfg, include_mtp=flagship)
    wl, hw = task.raw['workload'], task.raw['hardware']
    total_size = (index_metadata or {}).get('total_size')
    f.update({'schema': 'infrarsi-workload-facts-v1', 'task_id': task.id, 'repo_id': task.raw['model']['repo_id'],
              'family': task.raw['model'].get('family'), 'bridge_type': task.raw['model'].get('bridge_type'),
              'params_checkpoint': int(total_size) // 2 if total_size else None, 'seq_len': int(wl['seq_len']),
              'packs_per_update': int(wl['packs_per_update']), 'physical_tokens_per_update': int(wl['physical_tokens_per_update']),
              'data_kind': task.raw['data'].get('kind'), 'data_key': task.raw['data'].get('key'), 'gpus': int(hw['gpus']),
              'nodes': int(hw['nodes']), 'created_utc': utc_now()})
    f['tp_candidates'] = tp_candidates(f, min(8, f['gpus']))
    return f


def workload_facts(task: TaskSpec | str, *, remote=None, hf_config: dict | None = None, cache_dir: Path = CACHE,
                   refresh: bool = False) -> dict:
    """HF facts for a task: explicit hf_config > local cache > read-only remote read of <model_dir>/config.json."""
    task = TaskSpec.load(task) if isinstance(task, str) else task
    meta = None
    if hf_config is None:
        p = Path(cache_dir) / cache_name(task)
        if p.exists() and not refresh:
            cached = json.loads(p.read_text(encoding='utf-8'))
            hf_config, meta = cached['config'], cached.get('index_metadata')
        else:
            from infrarsi.harness.remote import default_remote
            r = (remote or default_remote()).call('WINH0', 'hf_model_facts', model_dir=model_dir(task))
            if not r.get('config'):
                raise FileNotFoundError(f'{model_dir(task)}/config.json not available yet (download in progress?)')
            hf_config, meta = r['config'], r.get('index_metadata')
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({'config': hf_config, 'index_metadata': meta, 'model_dir': model_dir(task)}, indent=1),
                         encoding='utf-8')
    return facts_from_config(task, hf_config, meta)
