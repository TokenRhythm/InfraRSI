"""Frozen objective contract, LR schedule and invariant digests (pure Python, no torch)."""
from __future__ import annotations

import enum
import math
from pathlib import Path
from typing import Any

from infrarsi.contracts import digest, file_digest

# Design section 1.3: identical for every public task. Task specs repeat it; a mismatch is a task error.
OBJECTIVE = {
    'optimizer': 'adamw',
    'betas': [0.9, 0.95],
    'weight_decay': 0.1,
    'eps': 1e-08,
    'clip_grad': 1.0,
    'lr_schedule': {'type': 'cosine', 'peak': 1e-05, 'min': 1e-06, 'warmup_updates': 0, 'horizon_updates': 1024},
    'moe': {'router_load_balancing': 'aux_loss', 'aux_loss_coeff': 0.001, 'router_dtype': 'fp32', 'capacity_factor': None},
}

LOSS_CONTRACT = {
    'main': 'cross entropy over supervised positions (labels != -100), token mean over the whole update',
    'reduction': 'calculate_per_token_loss: backward on per-micro-batch CE sums; finalize_model_grads divides '
                 'gradients by the global supervised-token count of the update',
    'aux': 'MCore aux_loss load balancing per MoE layer, fp32 router, dropless (no capacity factor)',
    'aux_reported': 'mean over MoE layers and over global micro-batches of load_balancing_loss (aux / coeff)',
    'labels': 'already aligned next-token targets from the packed data; never shifted again',
    'padding': 'pack padding runs through the whole model incl. the MoE router (counted in aux loss); '
               'only the CE excludes it (labels == -100)',
}

# Semantic (non-parallel, non-performance) model fields that enter model_config_digest.
MODEL_SEMANTIC_KEYS = (
    'hf_model_type', 'llm_model_type', 'num_layers', 'hidden_size', 'ffn_hidden_size', 'num_attention_heads',
    'num_query_groups', 'kv_channels', 'padded_vocab_size', 'max_position_embeddings', 'position_embedding_type',
    'rotary_base', 'rotary_percent', 'rotary_interleaved', 'rope_scaling', 'partial_rotary_factor',
    'attention_scaling', 'layernorm_epsilon', 'normalization', 'layernorm_zero_centered_gamma', 'qk_layernorm',
    'add_qkv_bias', 'add_bias_linear', 'untie_embeddings_and_output_weights', 'swiglu', 'gated_linear_unit',
    'hidden_dropout', 'attention_dropout', 'attention_output_gate', 'num_moe_experts', 'moe_ffn_hidden_size',
    'moe_shared_expert_intermediate_size', 'moe_shared_expert_gate', 'moe_router_topk', 'moe_router_pre_softmax',
    'moe_router_score_function', 'moe_router_topk_scaling_factor', 'moe_router_num_groups',
    'moe_router_group_topk', 'moe_router_enable_expert_bias', 'moe_layer_freq', 'multi_latent_attention',
    'q_lora_rank', 'kv_lora_rank', 'qk_head_dim', 'qk_pos_emb_head_dim', 'v_head_dim',
    'experimental_attention_variant', 'linear_attention_freq', 'linear_num_key_heads', 'linear_num_value_heads',
    'linear_key_head_dim', 'linear_value_head_dim', 'linear_conv_kernel_dim', 'mtp_num_layers',
)

# Live config fields that define the loss; read back from the built model after overlays ran.
LOSS_LIVE_KEYS = (
    'calculate_per_token_loss', 'moe_aux_loss_coeff', 'moe_router_load_balancing_type', 'moe_router_dtype',
    'moe_expert_capacity_factor', 'moe_z_loss_coeff', 'moe_token_drop_policy', 'mtp_num_layers',
)

FROZEN_DIRS = ('contracts', 'runtime', 'adapters')   # overlays (runtime/overlays) are digested separately


def jsonable(value: Any) -> Any:
    """Deterministic JSON view of config values (enums by name, dtypes and callables by qualified name)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if callable(value):
        return f'{getattr(value, "__module__", "")}.{getattr(value, "__qualname__", type(value).__name__)}'
    return str(value)


def objective_from_task(task: dict) -> dict:
    """Return the frozen objective after checking the task spec agrees with it."""
    spec = task.get('objective') or {}
    problems = []
    for key in ('optimizer', 'weight_decay', 'eps', 'clip_grad'):
        if spec.get(key) != OBJECTIVE[key]:
            problems.append(f'{key}={spec.get(key)!r}')
    if [float(b) for b in spec.get('betas') or []] != OBJECTIVE['betas']:
        problems.append(f'betas={spec.get("betas")!r}')
    sched = spec.get('lr_schedule') or {}
    for key, want in OBJECTIVE['lr_schedule'].items():
        if sched.get(key) != want:
            problems.append(f'lr_schedule.{key}={sched.get(key)!r}')
    moe = spec.get('moe') or {}
    for key, want in OBJECTIVE['moe'].items():
        if moe.get(key) != want:
            problems.append(f'moe.{key}={moe.get(key)!r}')
    if problems:
        raise ValueError(f'task {task.get("id")} objective deviates from the frozen contract: {problems}')
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in OBJECTIVE.items()}


def lr_at(update: int, schedule: dict | None = None) -> float:
    """LR applied by optimizer update ``update`` (1-based).

    Mirrors megatron.core OptimizerParamScheduler.get_lr with num_steps = update - 1 (the scheduler is
    stepped by one after every successful update), init_lr 0, identical float operation order.
    """
    s = schedule or OBJECTIVE['lr_schedule']
    if s['type'] != 'cosine':
        raise ValueError(f'unsupported schedule {s["type"]}')
    num_steps = update - 1
    max_lr, min_lr = float(s['peak']), float(s['min'])
    warmup, decay = int(s['warmup_updates']), int(s['horizon_updates'])
    if warmup > 0 and num_steps <= warmup:
        return 0.0 + (max_lr - 0.0) * float(num_steps) / float(warmup)
    if num_steps > decay:
        return min_lr
    decay_ratio = float(num_steps - warmup) / float(decay - warmup)
    coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
    return min_lr + coeff * (max_lr - min_lr)


def lr_schedule_digest(lrs: list[float]) -> str:
    return digest([float(x) for x in lrs])


def data_order_digest(update_packs: list[list[int]]) -> str:
    """Layout-invariant: the global pack set of every update, in update order."""
    return digest([sorted(int(p) for p in packs) for packs in update_packs])


def loss_definition(live: dict) -> dict:
    return {'contract': LOSS_CONTRACT, 'live': {k: jsonable(live.get(k)) for k in LOSS_LIVE_KEYS}}


def model_config_view(values: dict) -> dict:
    return {k: jsonable(values.get(k)) for k in MODEL_SEMANTIC_KEYS}


def param_inventory_digest(shape_counts: dict[str, int]) -> str:
    """Digest of the global multiset of trainable parameter shapes (layout invariant by construction)."""
    return digest(sorted([str(k), int(v)] for k, v in shape_counts.items()))


def frozen_zone_files(package_root: str | Path) -> list[list[str]]:
    root = Path(package_root)
    rows = []
    for sub in FROZEN_DIRS:
        for path in sorted((root / sub).rglob('*.py')):
            rel = path.relative_to(root).as_posix()
            if rel.startswith('runtime/overlays/') or '__pycache__' in rel:
                continue
            rows.append([rel, file_digest(path)])
    return rows


def frozen_zone_digest(package_root: str | Path) -> str:
    return digest(frozen_zone_files(package_root))
