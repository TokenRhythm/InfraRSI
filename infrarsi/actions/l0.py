"""L0 action space: Megatron-Core 0.18.2 config keys plus the env whitelist (design 1.8).

`validate_l0` checks types, domains and the cross-key rules MCore enforces (schema `constraints`, with
source lines). Task- and model-dependent rules run only when a task / model facts are given.
"""
from __future__ import annotations

import dataclasses
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).resolve().parent / 'schema' / 'l0_megatron_core_0_18_2.json'
MOE_ONLY = ('moe_token_dispatcher_type', 'moe_enable_deepep', 'moe_deepep_num_sms', 'moe_grouped_gemm',
            'moe_permute_fusion', 'moe_shared_expert_overlap', 'overlap_moe_expert_parallel_comm',
            'moe_layer_recompute', 'delay_wgrad_compute', 'expert_tensor_parallel_size')
OFFLOAD_ONLY = ('optimizer_offload_fraction', 'overlap_cpu_optimizer_d2h_h2d', 'pin_cpu_grads', 'pin_cpu_params')
# SFT packs (data D-SFT-*) pad every segment to a multiple of 8 tokens; TE THD context parallelism needs segment
# lengths divisible by 2*CP, so SFT tasks allow CP in {1, 2, 4}. PT streams (D-PT-*) allow any CP the layout admits.
SFT_CP_ALLOWED = (1, 2, 4)
# Keys with a single legal value under the frozen objective (C22) or no effect without VPP: never sampled.
NO_SEARCH = ('average_in_collective', 'align_param_gather', 'use_distributed_optimizer')


def is_sft_task(task_raw: dict) -> bool:
    data = task_raw.get('data') or {}
    return str(data.get('key', '')).upper().startswith('D-SFT') or data.get('kind') == 'sft_packed'


@lru_cache(maxsize=1)
def load_schema(path: str | None = None) -> dict:
    return json.loads(Path(path or SCHEMA_PATH).read_text(encoding='utf-8'))


def l0_keys() -> list[str]:
    return list(load_schema()['keys'])


def defaults() -> dict:
    """MCore defaults for every L0 key (the meaning of an omitted key)."""
    return {k: spec['default'] for k, spec in load_schema()['keys'].items()}


@dataclasses.dataclass
class L0Report:
    ok: bool
    errors: list[str]
    warnings: list[str]
    resolved: dict                      # every L0 key explicit, plus normalized 'env'
    derived: dict                       # world, dp, edp, ga when a task is given

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _task_raw(task) -> dict | None:
    if task is None:
        return None
    return task.raw if hasattr(task, 'raw') else task


def _check_value(key: str, spec: dict, v: Any) -> str | None:
    t = spec['type']
    if v is None:
        if spec.get('nullable') or None in spec.get('values', []):
            return None
        return f'{key}: null not allowed'
    if t == 'bool':
        if type(v) is not bool:
            return f'{key}: expected bool, got {v!r}'
        if 'values' in spec and v not in spec['values']:
            return f'{key}: only {spec["values"]} is legal (task constant)'
        return None
    if t == 'int':
        if type(v) is not int:
            return f'{key}: expected int, got {v!r}'
        if 'values' in spec and v not in spec['values']:
            return f'{key}: {v} not in {spec["values"]}'
        if 'range' in spec and not spec['range'][0] <= v <= spec['range'][1]:
            return f'{key}: {v} outside {spec["range"]}'
        return None
    if t == 'float':
        if type(v) not in (int, float) or not math.isfinite(v):
            return f'{key}: expected finite number, got {v!r}'
        lo, hi = spec['range']
        return None if lo <= v <= hi else f'{key}: {v} outside {spec["range"]}'
    if t == 'enum':
        return None if v in spec['values'] else f'{key}: {v!r} not in {spec["values"]}'
    if t == 'list_enum':
        if type(v) is not list or any(type(x) is not str for x in v):
            return f'{key}: expected list of strings, got {v!r}'
        bad = [x for x in v if x not in spec['values']]
        if bad:
            return f'{key}: unknown entries {bad}; allowed {spec["values"]}'
        if len(set(v)) != len(v):
            return f'{key}: duplicate entries'
        return None
    return f'{key}: unsupported schema type {t}'


def _check_env_value(name: str, spec: dict, raw: Any) -> tuple[str | None, str | None]:
    """Returns (normalized string value, error)."""
    if raw is None:
        return None, None
    if type(raw) is bool or type(raw) not in (int, str):
        return None, f'env {name}: value must be a string or int'
    v = str(raw).strip()
    t = spec['type']
    if t == 'int':
        try:
            n = int(v)
        except ValueError:
            return None, f'env {name}: {v!r} is not an int'
        lo, hi = spec['range']
        return (str(n), None) if lo <= n <= hi else (None, f'env {name}: {n} outside [{lo}, {hi}]')
    if t == 'enum':
        return (v, None) if v in spec['values'] else (None, f'env {name}: {v!r} not in {spec["values"]}')
    if t == 'str_list':
        items = v[1:].split(',') if v.startswith('^') else v.split(',')
        bad = [x for x in items if x not in spec['values']]
        return (v, None) if items and not bad else (None, f'env {name}: unknown entries {bad}')
    if t == 'alloc_conf':
        seen = set()
        for part in v.split(','):
            if ':' not in part:
                return None, f'env {name}: malformed option {part!r}'
            k, val = part.split(':', 1)
            opt = spec['options'].get(k)
            if opt is None:
                return None, f'env {name}: option {k!r} not whitelisted'
            if k in seen:
                return None, f'env {name}: duplicate option {k!r}'
            seen.add(k)
            if opt['type'] == 'enum' and val not in opt['values']:
                return None, f'env {name}: {k}:{val} not in {opt["values"]}'
            if opt['type'] in ('int', 'float'):
                try:
                    x = int(val) if opt['type'] == 'int' else float(val)
                except ValueError:
                    return None, f'env {name}: {k}:{val} not numeric'
                if not opt['range'][0] <= x <= opt['range'][1]:
                    return None, f'env {name}: {k}:{val} outside {opt["range"]}'
        return v, None
    return None, f'env {name}: unsupported type {t}'


def validate_env(env: Any) -> tuple[dict, list[str]]:
    schema = load_schema()
    if env is None:
        return {}, []
    if type(env) is not dict:
        return {}, ['env must be an object']
    out, errors = {}, []
    for name, raw in env.items():
        spec = schema['env'].get(name)
        if spec is None:
            errors.append(f'env {name}: not in the L0 env whitelist')
            continue
        v, err = _check_env_value(name, spec, raw)
        if err:
            errors.append(err)
        elif v is not None:
            out[name] = v
    return out, errors


def validate_l0(config: dict, task=None, model_facts: dict | None = None) -> L0Report:
    """Validate an L0 config (all or some keys; omitted keys take MCore defaults)."""
    schema = load_schema()
    keys = schema['keys']
    errors, warnings = [], []
    if type(config) is not dict:
        return L0Report(False, ['config must be an object'], [], {}, {})
    planner_only = sorted(k for k in config if type(k) is str and k.startswith('_'))
    if planner_only:                    # planner bookkeeping (e.g. planner.FLAGSHIP_B placements); never sent to WINH
        warnings.append(f'ignored planner-only keys: {planner_only}')
        config = {k: v for k, v in config.items() if k not in planner_only}
    for k in config:
        if k != 'env' and k not in keys:
            errors.append(f'{k}: not an L0 key' + (' (fixed by the task)' if k in schema['fixed_by_task']['keys'] else ''))
    c = defaults()
    for k, v in config.items():
        if k in keys:
            err = _check_value(k, keys[k], v)
            if err:
                errors.append(err)
            else:
                c[k] = list(v) if type(v) is list else v
    env, env_errors = validate_env(config.get('env'))
    errors += env_errors
    if errors:                          # cross-key rules assume well-typed values
        return L0Report(False, errors, warnings, {**c, 'env': env}, {})

    tp, pp, cp, ep = c['tensor_model_parallel_size'], c['pipeline_model_parallel_size'], c['context_parallel_size'], c['expert_model_parallel_size']
    etp = c['expert_tensor_parallel_size'] or tp
    gran, method, nrec, mods = c['recompute_granularity'], c['recompute_method'], c['recompute_num_layers'], c['recompute_modules']
    disp = c['moe_token_dispatcher_type']

    if c['sequence_parallel'] and tp <= 1:
        errors.append('C01 sequence_parallel requires tensor_model_parallel_size > 1')
    if c['tp_comm_overlap'] and not c['sequence_parallel']:
        errors.append('C02 tp_comm_overlap requires sequence_parallel')
    if gran == 'full':
        if method is None:
            errors.append('C03 full recompute needs recompute_method uniform|block')
        if nrec is None:
            errors.append('C03 full recompute needs recompute_num_layers >= 1')
    elif gran == 'selective':
        if nrec is not None:
            errors.append('C03 selective recompute requires recompute_num_layers null')
        if method is not None:
            errors.append('C03 selective recompute: recompute_method must be null in L0')
    else:
        if method is not None or nrec is not None:
            errors.append('C03 recompute_method/num_layers set without recompute_granularity')
    if mods is not None:
        if gran != 'selective' and not c['moe_layer_recompute']:
            warnings.append('C04 recompute_modules has no effect unless granularity is selective')
        if 'moe_act' in mods and not c['moe_grouped_gemm']:
            errors.append('C04 recompute_modules moe_act requires moe_grouped_gemm')
        if mods == []:
            warnings.append('C04 empty recompute_modules recomputes nothing')
    if c['moe_layer_recompute'] and gran == 'full':
        errors.append('C05 moe_layer_recompute cannot be combined with full recompute')
    if c['cpu_offloading']:
        if pp > 1:
            errors.append('C06 cpu_offloading requires pipeline_model_parallel_size == 1')
        if gran is not None or c['moe_layer_recompute']:
            errors.append('C06 cpu_offloading does not work with activation recomputation')
    if c['overlap_param_gather'] and not c['overlap_grad_reduce']:
        errors.append('C07 overlap_param_gather requires overlap_grad_reduce')
    if c['moe_enable_deepep'] and disp != 'flex':
        errors.append("C08 moe_enable_deepep requires moe_token_dispatcher_type 'flex'")
    if c['moe_shared_expert_overlap'] and disp not in ('alltoall', 'flex'):
        errors.append('C09 moe_shared_expert_overlap requires alltoall or flex dispatcher')
    if c['overlap_moe_expert_parallel_comm']:            # trainer preflight rejects it (combined-1F1B schedule plan)
        errors.append('C10 overlap_moe_expert_parallel_comm is not supported by the v0.1 generic trainer')
        if ep <= 1:
            errors.append('C10 overlap_moe_expert_parallel_comm requires expert_model_parallel_size > 1')
        if disp not in ('alltoall', 'flex'):
            errors.append('C10 overlap_moe_expert_parallel_comm requires alltoall or flex dispatcher')
        if gran == 'full' or method is not None or nrec is not None:
            errors.append('C10 overlap_moe_expert_parallel_comm forbids full recompute / recompute method / num_layers')
        if (mods and 'moe' in mods) or c['moe_layer_recompute']:
            errors.append("C10 overlap_moe_expert_parallel_comm forbids 'moe' recompute")
        if c['moe_shared_expert_overlap']:
            errors.append('C10 overlap_moe_expert_parallel_comm forbids moe_shared_expert_overlap')
        if pp > 1:
            errors.append('C10 overlap_moe_expert_parallel_comm with PP > 1 needs VPP, which is not in L0')
    if c['delay_wgrad_compute'] and not c['overlap_moe_expert_parallel_comm']:
        errors.append('C11 delay_wgrad_compute requires overlap_moe_expert_parallel_comm')
    if c['cross_entropy_loss_fusion'] and c['cross_entropy_fusion_impl'] == 'te':
        errors.append("C12 cross_entropy_loss_fusion with cross_entropy_fusion_impl 'te' is rejected (stability)")
    if c['average_in_collective']:
        errors.append('C22 average_in_collective must be false: Megatron-Core asserts it off under calculate_per_token_loss '
                      '(core/distributed/distributed_data_parallel.py:169-172), which the frozen objective uses')
    if c['cp_comm_type'] is not None and cp <= 1:
        warnings.append('C19 cp_comm_type has no effect with context_parallel_size 1')
    if c['moe_deepep_num_sms'] % 2:
        errors.append('C20 moe_deepep_num_sms must be even')
    if c['align_param_gather']:
        warnings.append('align_param_gather only acts with interleaved PP (not in L0)')
    if not c['optimizer_cpu_offload'] and (c['optimizer_offload_fraction'] or c['overlap_cpu_optimizer_d2h_h2d']):
        warnings.append('optimizer offload options have no effect without optimizer_cpu_offload')
    if c['optimizer_cpu_offload'] and c['optimizer_offload_fraction'] == 0.0:
        warnings.append('optimizer_cpu_offload with optimizer_offload_fraction 0.0 keeps all state on GPU')

    derived = {}
    raw = _task_raw(task)
    if raw is not None:
        world = int(raw['hardware']['gpus'])
        nodes = int(raw['hardware']['nodes'])
        per_node = world // max(nodes, 1)
        packs = int(raw['workload']['packs_per_update'])
        derived.update(world=world, gpus_per_node=per_node, etp=etp)
        if world % (tp * pp * cp):
            errors.append(f'C13 world {world} not divisible by TP*PP*CP = {tp * pp * cp}')
        else:
            dp = world // (tp * pp * cp)
            derived['dp'] = dp
            if packs % (dp * c['micro_batch_size']):
                errors.append(f'C14 packs_per_update {packs} not divisible by DP*MBS = {dp * c["micro_batch_size"]}')
            else:
                derived['ga'] = packs // (dp * c['micro_batch_size'])
        if world % (etp * ep * pp):
            errors.append(f'C13 world {world} not divisible by ETP*EP*PP = {etp * ep * pp}')
        else:
            derived['edp'] = world // (etp * ep * pp)
        if tp > per_node or per_node % tp:
            errors.append(f'C15 TP {tp} must divide GPUs per node ({per_node})')
        if is_sft_task(raw) and cp not in SFT_CP_ALLOWED:
            errors.append(f'C19 context_parallel_size={cp} not allowed on SFT task (data {raw["data"]["key"]}): packed '
                          f'segments are padded to multiples of 8 tokens, so CP must be in {list(SFT_CP_ALLOWED)} '
                          f'(rejected before any screen)')
        seq = int(raw['workload']['seq_len'])
        if cp > 1 and seq % (2 * cp):
            errors.append(f'C19 seq_len {seq} not divisible by 2*CP')

    if model_facts:
        mf = model_facts
        heads, groups = mf.get('num_attention_heads'), mf.get('num_query_groups')
        if heads and heads % tp:
            errors.append(f'C16 attention heads {heads} not divisible by TP {tp}')
        if groups:
            if groups >= tp and groups % tp:
                errors.append(f'C16 query groups {groups} not divisible by TP {tp}')
            if groups < tp and tp % groups:
                errors.append(f'C16 TP {tp} not a multiple of query groups {groups}')
        layers = mf.get('num_layers')
        if layers:
            if layers % pp:
                errors.append(f'C18 num_layers {layers} not divisible by PP {pp}')
            elif gran == 'full' and nrec is not None and nrec > layers // pp:
                errors.append(f'C18 recompute_num_layers {nrec} > layers per stage {layers // pp}')
            if c['cpu_offloading'] and not 0 <= c['cpu_offloading_num_layers'] < layers:
                errors.append('C06 cpu_offloading_num_layers must be < num_layers')
        experts = mf.get('num_experts') or 0
        if experts:
            if experts % ep:
                errors.append(f'C17 num_experts {experts} not divisible by EP {ep}')
            if c['moe_shared_expert_overlap'] and not mf.get('has_shared_experts'):
                warnings.append('moe_shared_expert_overlap has no effect without shared experts')
        else:
            if ep != 1:
                errors.append('C17 dense model requires expert_model_parallel_size 1')
            touched = [k for k in MOE_ONLY if k in config and config[k] != keys[k]['default']]
            if touched:
                warnings.append(f'C21 MoE-only keys have no effect on a dense model: {touched}')
            if mods and set(mods) & {'moe', 'moe_act', 'shared_experts'}:
                errors.append('C04 MoE recompute modules on a dense model')
        if mods and 'mla_up_proj' in mods and not mf.get('multi_latent_attention'):
            errors.append('C04 recompute_modules mla_up_proj requires multi-latent attention')
        if c['apply_rope_fusion'] and mf.get('multi_latent_attention') and mf.get('rope_type') not in (None, 'yarn'):
            errors.append('apply_rope_fusion with MLA needs YaRN RoPE (transformer_config.py:2637)')
        if c['bias_activation_fusion'] and mf.get('activation') not in (None, 'silu', 'swiglu', 'gelu', 'quick_gelu'):
            errors.append(f'bias_activation_fusion unsupported for activation {mf.get("activation")!r}')

    resolved = {**c, 'env': env}
    return L0Report(not errors, errors, warnings, resolved, derived)


def model_facts_from_hf_config(hf: dict) -> dict:
    """Facts validate_l0 and the planner need, from a HuggingFace config.json dict. Multimodal wrappers (the flagship
    qwen4_exp) nest the language model under text_config, which is what trains here (discover.workload.text_config)."""
    top = hf
    hf = hf.get('text_config') or hf
    experts = hf.get('num_experts') or hf.get('n_routed_experts') or hf.get('num_local_experts') or 0
    shared = (hf.get('shared_expert_intermediate_size') or 0) > 0 or (hf.get('n_shared_experts') or 0) > 0
    rope = hf.get('rope_scaling') or {}
    return {
        'model_type': top.get('model_type') or hf.get('model_type'),
        'num_layers': hf.get('num_hidden_layers'),
        'num_attention_heads': hf.get('num_attention_heads'),
        'num_query_groups': hf.get('num_key_value_heads') or hf.get('num_attention_heads'),
        'hidden_size': hf.get('hidden_size'),
        'ffn_hidden_size': hf.get('intermediate_size'),
        'moe_ffn_hidden_size': hf.get('moe_intermediate_size'),
        'vocab_size': hf.get('vocab_size'),
        'num_experts': int(experts),
        'moe_router_topk': hf.get('num_experts_per_tok'),
        'has_shared_experts': bool(shared),
        'multi_latent_attention': hf.get('kv_lora_rank') is not None,
        'rope_type': rope.get('type') or rope.get('rope_type'),
        'activation': hf.get('hidden_act'),
        'linear_attention': bool(hf.get('layer_types') or hf.get('full_attention_interval')),
    }


def l0_space(task=None, model_facts: dict | None = None, include_env: bool = True) -> dict[str, list]:
    """Search grid for samplers (random/TPE): applicable keys only, values from `search_values`/`values`."""
    schema = load_schema()
    raw = _task_raw(task)
    moe = bool(model_facts and model_facts.get('num_experts'))
    per_node = world = None
    if raw is not None:
        world = int(raw['hardware']['gpus'])
        per_node = world // int(raw['hardware']['nodes'])
    space = {}
    for k, spec in schema['keys'].items():
        if k in MOE_ONLY and model_facts is not None and not moe:
            continue
        if k in NO_SEARCH:
            continue
        if spec['type'] == 'bool':
            vals = [False, True]
        else:
            vals = list(spec.get('search_values', spec.get('values', [])))
        if k == 'tensor_model_parallel_size' and per_node:
            vals = [v for v in vals if v <= per_node and per_node % v == 0]
        if k in ('pipeline_model_parallel_size', 'context_parallel_size') and world:
            vals = [v for v in vals if world % v == 0]
        if k == 'context_parallel_size' and raw is not None and is_sft_task(raw):
            vals = [v for v in vals if v in SFT_CP_ALLOWED]
        if k == 'expert_model_parallel_size':
            if not moe and model_facts is not None:
                continue
            if world:
                vals = [v for v in vals if v <= world and world % v == 0]
            if model_facts and model_facts.get('num_experts'):
                vals = [v for v in vals if model_facts['num_experts'] % v == 0]
        if k == 'recompute_modules' and model_facts is not None and not moe:
            vals = [v for v in vals if v is None or not set(v) & {'moe', 'moe_act', 'shared_experts'}]
        if k == 'cp_comm_type' and world == 1:
            continue
        if vals:
            space[k] = vals
    if include_env:
        for name, spec in schema['env'].items():
            if 'search_values' in spec:
                space['env:' + name] = list(spec['search_values'])
    return space


def config_from_point(point: dict) -> dict:
    """Turn a flat sampler point ('env:NAME' keys) into an L0 config with an 'env' block."""
    config, env = {}, {}
    for k, v in point.items():
        if k.startswith('env:'):
            if v is not None:
                env[k[4:]] = v
        else:
            config[k] = v
    if env:
        config['env'] = env
    return config


def config_diff(parent: dict, child: dict) -> dict:
    """{key: [old, new]} over resolved L0 keys and env (both sides resolved with defaults)."""
    d0, out = defaults(), {}
    a = {**d0, **{k: v for k, v in parent.items() if k != 'env'}}
    b = {**d0, **{k: v for k, v in child.items() if k != 'env'}}
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            out[k] = [a.get(k), b.get(k)]
    ea, eb = parent.get('env') or {}, child.get('env') or {}
    for k in sorted(set(ea) | set(eb)):
        if str(ea.get(k)) != str(eb.get(k)) or (k in ea) != (k in eb):
            out['env:' + k] = [ea.get(k), eb.get(k)]
    return out
