"""Conditional sampling of the L0 space for random/TPE: a key is drawn only when it can matter, so draws respect the
MCore cross-key rules (constraints C01-C21) by construction. Remaining rejections (world/batch divisibility, model
facts) are handled by validate_l0. `suggest(name, choices)` is rng.choice or optuna's trial.suggest_categorical;
names with '@' are conditional sub-spaces with their own fixed choice lists (optuna needs static choices).
"""
from __future__ import annotations

from ..actions import config_from_point, load_schema
from .common import decode_value, encode_value

ON = {'recompute_num_layers@full': [1, 2, 4, 8], 'recompute_method@full': ['uniform', 'block'],
      'recompute_modules@selective': [['core_attn'], ['layernorm'], ['mlp'], ['core_attn', 'mlp'], ['layernorm', 'mlp'],
                                      ['moe_act'], ['moe'], ['layernorm', 'moe_act'], ['core_attn', 'moe_act']],
      'cpu_offloading_num_layers@on': [1, 2, 4, 8], 'optimizer_offload_fraction@on': [0.25, 0.5, 0.75, 1.0]}
FUSIONS = ('apply_rope_fusion', 'bias_activation_fusion', 'masked_softmax_fusion', 'gradient_accumulation_fusion',
           'cross_entropy_loss_fusion')


LAYOUT_KEYS = ('tensor_model_parallel_size', 'pipeline_model_parallel_size', 'context_parallel_size',
               'expert_model_parallel_size', 'expert_tensor_parallel_size', 'micro_batch_size')
LAYOUT_TAGS = ('tp', 'pp', 'cp', 'ep', 'etp', 'mbs')


def layout_choices(task, model_facts: dict | None, space: dict) -> list[str]:
    """Every (TP, PP, CP, EP, ETP, MBS) the task and model admit (C13-C19), encoded 'tp2.pp1.cp1.ep8.etp1.mbs1'."""
    import itertools
    from ..actions import validate_l0
    grids = [space.get(k, [None]) if k != 'expert_tensor_parallel_size' else [v for v in space.get(k, [None])] for k in LAYOUT_KEYS]
    out = []
    for combo in itertools.product(*grids):
        cfg = {k: v for k, v in zip(LAYOUT_KEYS, combo) if v is not None or k == 'expert_tensor_parallel_size'}
        if validate_l0(cfg, task, model_facts).ok:
            out.append('.'.join(f'{t}{"n" if v is None else v}' for t, v in zip(LAYOUT_TAGS, combo)))
    return out


def decode_layout(code: str) -> dict:
    out = {}
    for part, key, tag in zip(code.split('.'), LAYOUT_KEYS, LAYOUT_TAGS):
        v = part[len(tag):]
        out[key] = None if v in ('n', 'None') else int(v)
    return {k: v for k, v in out.items() if v is not None or k == 'expert_tensor_parallel_size'}


def sample_config(suggest, space: dict, model_facts: dict | None = None) -> tuple[dict, dict]:
    """Returns (point {name: encoded choice}, L0 config)."""
    point, cfg = {}, {}
    moe = bool((model_facts or {}).get('num_experts'))
    dense_mods = lambda mods: not set(mods) & {'moe', 'moe_act', 'shared_experts'}

    def draw(name: str, choices: list, key: str | None = None):
        enc = [encode_value(c) for c in choices]
        v = suggest(name, enc)
        point[name] = v
        cfg[key or name.split('@')[0]] = decode_value(v)
        return cfg[key or name.split('@')[0]]

    def s(key: str):
        if key in space:
            return draw(key, space[key])
        return None

    if space.get('__layout__'):
        code = suggest('layout', list(space['__layout__']))
        point['layout'] = code
        cfg.update(decode_layout(code))
        tp, pp, cp = cfg['tensor_model_parallel_size'], cfg['pipeline_model_parallel_size'], cfg['context_parallel_size']
        ep = cfg.get('expert_model_parallel_size') or 1
    else:
        tp = s('tensor_model_parallel_size') or 1
        pp = s('pipeline_model_parallel_size') or 1
        cp = s('context_parallel_size') or 1
        ep = (s('expert_model_parallel_size') or 1) if moe else 1
        if moe:
            s('expert_tensor_parallel_size')
        s('micro_batch_size')
    if tp > 1 and s('sequence_parallel'):
        s('tp_comm_overlap')
    if cp > 1:
        s('cp_comm_type')
    disp, grouped, shared = None, False, False
    if moe:                                                   # MoE kernels first: recompute choices depend on them
        disp = s('moe_token_dispatcher_type')
        if disp == 'flex':
            s('moe_enable_deepep')
            s('moe_deepep_num_sms')
        grouped = bool(s('moe_grouped_gemm'))
        s('moe_permute_fusion')
        if disp in ('alltoall', 'flex') and (model_facts or {}).get('has_shared_experts'):
            shared = bool(s('moe_shared_expert_overlap'))
    gran = s('recompute_granularity')
    if gran == 'full':
        draw('recompute_method@full', ON['recompute_method@full'])
        n = draw('recompute_num_layers@full', ON['recompute_num_layers@full'])
        per_stage = ((model_facts or {}).get('num_layers') or 0) // pp
        if per_stage and n > per_stage:                       # deterministic repair: at most every layer of a stage
            cfg['recompute_num_layers'] = per_stage
    elif gran == 'selective':
        variant = '_dense' if not moe else ('_grouped' if grouped else '_ungrouped')
        mods = [m for m in ON['recompute_modules@selective'] if (moe or dense_mods(m)) and (grouped or 'moe_act' not in m)]
        draw('recompute_modules@selective' + variant, mods, key='recompute_modules')
    elif pp == 1 and s('cpu_offloading'):
        draw('cpu_offloading_num_layers@on', ON['cpu_offloading_num_layers@on'])
    if s('optimizer_cpu_offload'):
        draw('optimizer_offload_fraction@on', ON['optimizer_offload_fraction@on'])
        for k in ('overlap_cpu_optimizer_d2h_h2d', 'pin_cpu_grads', 'pin_cpu_params'):
            s(k)
    if s('overlap_grad_reduce'):
        s('overlap_param_gather')
    s('bucket_size')
    s('average_in_collective')
    # overlap_moe_expert_parallel_comm (and delay_wgrad_compute, which needs it) is not supported by the v0.1 generic
    # trainer (combined-1F1B schedule plan); validate_l0 rejects it, so the sampler never draws it
    for k in FUSIONS:
        s(k)
    s('attention_backend')
    env_point = {}
    for k in sorted(space):
        if k.startswith('env:'):
            v = suggest(k, [encode_value(c) for c in space[k]])
            point[k] = v
            env_point[k] = decode_value(v)
    return point, {**cfg, **config_from_point(env_point)}


def config_to_trial_params(config: dict, space: dict, model_facts: dict | None = None) -> tuple[dict, dict]:
    """Best-effort projection of a config (the ED start) onto the conditional parameter names, for optuna.

    Returns (params, {name: choices}); values outside a choice list fall back to its first choice.
    """
    from ..actions import validate_l0
    c = validate_l0(config).resolved
    replay, choices_of = {}, {}

    def suggest(name, choices):
        key = name.split('@')[0]
        if name.startswith('env:'):
            v = (c.get('env') or {}).get(name[4:])
            v = encode_value(None if v is None else (int(v) if str(v).isdigit() else v))
        elif name == 'layout':
            v = '.'.join(f'{t}{"n" if c.get(k) is None else c.get(k)}' for k, t in zip(LAYOUT_KEYS, LAYOUT_TAGS))
        else:
            v = encode_value(c.get(key))
        replay[name] = v if v in choices else choices[0]
        choices_of[name] = list(choices)
        return replay[name]

    sample_config(suggest, space, model_facts)
    return replay, choices_of


def schema_keys() -> list[str]:
    return list(load_schema()['keys'])
