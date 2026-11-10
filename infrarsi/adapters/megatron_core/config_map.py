"""RunRequest (task + L0 config) -> Megatron-Core / bridge construction kwargs. Pure Python, no torch.

Missing L0 keys take the schema default (the Megatron-Core field default); the FR program writes every
key explicitly. Task constants (design 1.3 / 1.4: distributed optimizer, fp32 grad reduce, the objective,
calculate_per_token_loss) are applied last and cannot be set from L0. Structural checks that would
otherwise fail deep inside Megatron-Core (divisibility, GA, CP vs. packing) raise ConfigError here.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from infrarsi.runtime import invariants as inv

SCHEMA_PATH = Path(__file__).resolve().parents[2] / 'actions' / 'schema' / 'l0_megatron_core_0_18_2.json'

# Megatron-Core 0.18.2 field defaults; used only when the L0 schema file is not shipped with the code.
FALLBACK_DEFAULTS = {
    'tensor_model_parallel_size': 1, 'pipeline_model_parallel_size': 1, 'context_parallel_size': 1,
    'expert_model_parallel_size': 1, 'expert_tensor_parallel_size': None, 'sequence_parallel': False,
    'micro_batch_size': 1, 'recompute_granularity': None, 'recompute_method': None, 'recompute_num_layers': None,
    'recompute_modules': None, 'moe_layer_recompute': False, 'cpu_offloading': False, 'cpu_offloading_num_layers': 0,
    'optimizer_cpu_offload': False, 'optimizer_offload_fraction': 0.0, 'overlap_cpu_optimizer_d2h_h2d': False,
    'pin_cpu_grads': True, 'pin_cpu_params': True, 'overlap_grad_reduce': False, 'overlap_param_gather': False,
    'align_param_gather': False, 'bucket_size': None, 'average_in_collective': False, 'delay_wgrad_compute': False,
    'tp_comm_overlap': False, 'cp_comm_type': None, 'moe_token_dispatcher_type': 'allgather',
    'moe_enable_deepep': False, 'moe_deepep_num_sms': 20, 'moe_grouped_gemm': False, 'moe_permute_fusion': False,
    'moe_shared_expert_overlap': False, 'overlap_moe_expert_parallel_comm': False, 'apply_rope_fusion': False,
    'bias_activation_fusion': False, 'masked_softmax_fusion': False, 'gradient_accumulation_fusion': False,
    'cross_entropy_loss_fusion': False, 'cross_entropy_fusion_impl': 'native', 'attention_backend': 'auto',
}
FALLBACK_ENV = ('PYTORCH_CUDA_ALLOC_CONF', 'CUDA_DEVICE_MAX_CONNECTIONS', 'OMP_NUM_THREADS')
FIXED_KEYS = ('global_batch_size', 'seq_length', 'train_iters', 'moe_router_topk', 'moe_expert_capacity_factor', 'fp8',
              'main_params_dtype', 'exp_avg_dtype', 'exp_avg_sq_dtype', 'use_precision_aware_optimizer',
              'use_distributed_optimizer', 'grad_reduce_in_fp32', 'virtual_pipeline_model_parallel_size', 'lr',
              'clip_grad')
# FR environment (design 1.4): only CUDA_DEVICE_MAX_CONNECTIONS is pinned; OMP keeps torchrun's default.
ENV_DEFAULTS = {'CUDA_DEVICE_MAX_CONNECTIONS': '1', 'OMP_NUM_THREADS': '1'}

TRANSFORMER_KEYS = ('recompute_granularity', 'recompute_method', 'recompute_num_layers', 'recompute_modules',
                    'cpu_offloading', 'cpu_offloading_num_layers', 'tp_comm_overlap', 'apply_rope_fusion',
                    'bias_activation_fusion', 'masked_softmax_fusion', 'gradient_accumulation_fusion',
                    'cross_entropy_loss_fusion', 'cross_entropy_fusion_impl', 'attention_backend')
MOE_KEYS = ('moe_token_dispatcher_type', 'moe_enable_deepep', 'moe_deepep_num_sms', 'moe_grouped_gemm',
            'moe_permute_fusion', 'moe_shared_expert_overlap', 'overlap_moe_expert_parallel_comm',
            'delay_wgrad_compute', 'moe_layer_recompute')
DDP_KEYS = ('overlap_grad_reduce', 'overlap_param_gather', 'align_param_gather', 'bucket_size')
OFFLOAD_KEYS = ('optimizer_cpu_offload', 'optimizer_offload_fraction', 'overlap_cpu_optimizer_d2h_h2d',
                'pin_cpu_grads', 'pin_cpu_params')


class ConfigError(ValueError):
    pass


def load_schema(path: str | Path = SCHEMA_PATH) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def model_facts(model_dir_or_config: str | Path | dict) -> dict:
    """Shape facts needed for layout checks, read from the HF config.json (no transformers import)."""
    if isinstance(model_dir_or_config, dict):
        cfg = model_dir_or_config
    else:
        p = Path(model_dir_or_config)
        cfg = json.loads((p / 'config.json' if p.is_dir() else p).read_text(encoding='utf-8'))
    cfg = cfg.get('text_config', cfg)
    experts = next((cfg[k] for k in ('num_experts', 'n_routed_experts', 'num_local_experts') if cfg.get(k)), None)
    heads = int(cfg['num_attention_heads'])
    return {'model_type': cfg.get('model_type'), 'num_layers': int(cfg['num_hidden_layers']),
            'num_attention_heads': heads, 'num_query_groups': int(cfg.get('num_key_value_heads') or heads),
            'num_experts': int(experts) if experts else None, 'vocab_size': int(cfg['vocab_size']),
            'multi_latent_attention': cfg.get('kv_lora_rank') is not None,
            'gated_delta_net': cfg.get('model_type') == 'qwen3_next' or cfg.get('linear_num_value_heads') is not None,
            'tie_word_embeddings': bool(cfg.get('tie_word_embeddings', False))}


@dataclasses.dataclass
class Plan:
    world: int
    nodes: int
    gpus_per_node: int
    tp: int
    pp: int
    cp: int
    ep: int
    etp: int
    dp: int
    mbs: int
    ga: int
    sp: bool
    seq_len: int
    packs_per_update: int
    data_kind: str
    is_moe: bool
    effective: dict
    model_kwargs: dict
    ddp_kwargs: dict
    optimizer_kwargs: dict
    env: dict
    objective: dict

    def summary(self) -> dict:
        return {k: getattr(self, k) for k in ('world', 'tp', 'pp', 'cp', 'ep', 'etp', 'dp', 'mbs', 'ga', 'sp',
                                               'seq_len', 'packs_per_update', 'is_moe')}


def _schema_parts(schema: dict | None):
    if schema is None:
        return dict(FALLBACK_DEFAULTS), set(FALLBACK_ENV), set(FIXED_KEYS)
    defaults = {k: spec.get('default') for k, spec in schema['keys'].items()}
    return defaults, set(schema.get('env', {})), set(schema.get('fixed_by_task', {}).get('keys', FIXED_KEYS))


def make_plan(request, facts: dict, schema: dict | None | str = 'auto') -> Plan:
    """Validate the request against the model/task and derive every construction kwarg."""
    req = request.to_dict() if hasattr(request, 'to_dict') else dict(request)
    task, workload = req['task'], req['task']['workload']
    schema = load_schema() if schema == 'auto' else schema
    defaults, env_whitelist, fixed = _schema_parts(schema)
    cfg = dict(req['config'] or {})
    env_req = {str(k): str(v) for k, v in (cfg.pop('env', None) or {}).items() if v is not None}
    if set(cfg) & fixed:
        raise ConfigError(f'task constants cannot be set from L0: {sorted(set(cfg) & fixed)}')
    unknown = set(cfg) - set(defaults)
    if unknown:
        raise ConfigError(f'unknown L0 keys: {sorted(unknown)}')
    bad_env = sorted(k for k in env_req if k not in env_whitelist)
    if bad_env:
        raise ConfigError(f'env keys outside the L0 whitelist: {bad_env}')
    eff = dict(defaults)
    eff.update(cfg)

    world = len(req['nodes']) * int(req['gpus_per_node'])
    gpn = int(req['gpus_per_node'])
    tp, pp, cp = int(eff['tensor_model_parallel_size']), int(eff['pipeline_model_parallel_size']), int(eff['context_parallel_size'])
    mbs = int(eff['micro_batch_size'])
    is_moe = facts.get('num_experts') is not None
    ep = int(eff['expert_model_parallel_size'])
    etp = tp if eff['expert_tensor_parallel_size'] is None else int(eff['expert_tensor_parallel_size'])
    sp = bool(eff['sequence_parallel'])
    seq_len, ppu = int(workload['seq_len']), int(workload['packs_per_update'])
    data_kind = (task.get('data') or {}).get('kind', 'sft_packed')

    def need(cond, msg):
        if not cond:
            raise ConfigError(msg)

    need(world % (tp * pp * cp) == 0, f'world {world} not divisible by TP*PP*CP={tp * pp * cp}')
    dp = world // (tp * pp * cp)
    need(tp <= gpn and gpn % tp == 0, f'TP {tp} must divide GPUs per node {gpn}')
    need(not sp or tp > 1, 'sequence_parallel requires TP > 1')
    need(ppu % (dp * mbs) == 0, f'packs_per_update {ppu} not divisible by DP*MBS={dp * mbs}')
    ga = ppu // (dp * mbs)
    if bool(eff.get('overlap_moe_expert_parallel_comm')):
        raise ConfigError('overlap_moe_expert_parallel_comm selects Megatron-Core\'s combined-1F1B schedule-plan path, '
                          'which the generic trainer does not support in v0.1')
    need(facts['num_attention_heads'] % tp == 0, f'attention heads {facts["num_attention_heads"]} % TP {tp} != 0')
    groups = facts['num_query_groups']
    need(groups % tp == 0 or tp % groups == 0, f'query groups {groups} incompatible with TP {tp}')
    need(facts['num_layers'] % pp == 0, f'num_layers {facts["num_layers"]} % PP {pp} != 0')
    if is_moe:
        need(facts['num_experts'] % ep == 0, f'num_experts {facts["num_experts"]} % EP {ep} != 0')
        need(world % (etp * ep * pp) == 0, f'world {world} not divisible by ETP*EP*PP={etp * ep * pp}')
    else:
        need(ep == 1, 'dense model requires EP == 1')
        etp = tp
    if cp > 1:
        granularity = 8 if data_kind == 'sft_packed' else seq_len
        need(granularity % (2 * cp) == 0, f'CP {cp}: packed segments (multiple of {granularity}) not divisible by 2*CP')
    tokens_local = mbs * seq_len // cp
    need(not sp or tokens_local % tp == 0, f'micro-batch tokens per CP rank {tokens_local} % TP {tp} != 0')
    need(eff.get('use_distributed_optimizer', True) is True, 'use_distributed_optimizer is a task constant (true)')
    need(not eff['average_in_collective'], 'average_in_collective conflicts with the frozen calculate_per_token_loss')
    need(not eff['overlap_param_gather'] or eff['overlap_grad_reduce'], 'overlap_param_gather requires overlap_grad_reduce')
    if eff['recompute_num_layers'] is not None and eff['recompute_granularity'] == 'full':
        need(int(eff['recompute_num_layers']) <= facts['num_layers'] // pp, 'recompute_num_layers exceeds layers per stage')

    objective = inv.objective_from_task(task)
    model_kwargs = {'tensor_model_parallel_size': tp, 'pipeline_model_parallel_size': pp,
                    'context_parallel_size': cp, 'expert_model_parallel_size': ep if is_moe else 1,
                    'expert_tensor_parallel_size': etp, 'sequence_parallel': sp,
                    'cp_comm_type': eff['cp_comm_type'] or 'p2p'}
    model_kwargs.update({k: eff[k] for k in TRANSFORMER_KEYS})
    if is_moe:
        model_kwargs.update({k: eff[k] for k in MOE_KEYS})
        moe = objective['moe']
        model_kwargs.update(moe_router_load_balancing_type=moe['router_load_balancing'],
                            moe_aux_loss_coeff=moe['aux_loss_coeff'], moe_router_dtype=moe['router_dtype'],
                            moe_expert_capacity_factor=moe['capacity_factor'], moe_z_loss_coeff=None)
    model_kwargs.update(calculate_per_token_loss=True, hidden_dropout=0.0, perform_initialization=False, bf16=True)

    ddp_kwargs = {k: eff[k] for k in DDP_KEYS}
    ddp_kwargs.update(grad_reduce_in_fp32=True, use_distributed_optimizer=True, average_in_collective=False,
                      check_for_nan_in_grad=False, delay_wgrad_compute=bool(is_moe and eff['delay_wgrad_compute']))

    offload = {k: eff[k] for k in OFFLOAD_KEYS}
    lr = objective['lr_schedule']
    optimizer_kwargs = dict(optimizer='adam', lr=lr['peak'], min_lr=lr['min'], weight_decay=objective['weight_decay'],
                            adam_beta1=objective['betas'][0], adam_beta2=objective['betas'][1],
                            adam_eps=objective['eps'], clip_grad=objective['clip_grad'], decoupled_weight_decay=True,
                            bf16=True, use_distributed_optimizer=True,
                            overlap_param_gather=bool(eff['overlap_param_gather']), **offload)
    # The hybrid CPU/GPU optimizer reuses the precision-aware code path; masters and moments stay fp32.
    optimizer_kwargs['use_precision_aware_optimizer'] = bool(offload['optimizer_cpu_offload'])

    env = dict(ENV_DEFAULTS)
    env.update(env_req)
    return Plan(world=world, nodes=len(req['nodes']), gpus_per_node=gpn, tp=tp, pp=pp, cp=cp, ep=ep if is_moe else 1,
                etp=etp, dp=dp, mbs=mbs, ga=ga, sp=sp, seq_len=seq_len, packs_per_update=ppu, data_kind=data_kind,
                is_moe=is_moe, effective=eff, model_kwargs=model_kwargs, ddp_kwargs=ddp_kwargs,
                optimizer_kwargs=optimizer_kwargs, env={'set': env, 'whitelist': sorted(env_whitelist)},
                objective=objective)


def model_dir_for(task: dict) -> str:
    from infrarsi.contracts import RemotePaths
    model = task['model']
    return model.get('dir') or f"{RemotePaths.models}/{model['repo_id'].replace('/', '__')}"


def code_digest_problem(expected: str, package_dir: str | Path) -> str | None:
    """None when the imported package tree hashes to the request's code_digest (or none was given)."""
    if not expected:
        return None
    from infrarsi.contracts import code_digest
    got = code_digest(package_dir)
    return None if got == expected else f'code_digest_mismatch: request {expected} != imported tree {got} ({package_dir})'


def deterministic_lifecycle(request, facts: dict) -> bool:
    """Opt-in bit-faithful lifecycle (request.lifecycle.deterministic). Megatron-Core's Gated DeltaNet has no
    deterministic path for packed THD sequences, so those families keep the default kernels."""
    return (request.purpose == 'lifecycle' and bool((request.lifecycle or {}).get('deterministic'))
            and not facts.get('gated_delta_net'))


def child_env(parent: dict, plan: Plan) -> dict:
    """Trainer environment: whitelisted performance variables come only from the request (plus FR defaults)."""
    env = {k: v for k, v in parent.items() if k not in plan.env['whitelist'] and k not in ENV_DEFAULTS}
    env.update(plan.env['set'])
    return env
