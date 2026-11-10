"""Bridge model, DDP, optimizer and LR scheduler construction (Megatron-Core 0.18.2 + vendored mcore-bridge).

Mirrors flash-next external_train.py: ``hf_to_mcore_config(AutoConfig)`` + layout/L0 kwargs -> bridge
``ModelConfig`` -> ``get_mcore_model`` -> ``bridge.load_weights`` -> MCore DDP -> ``get_megatron_optimizer``,
and Megatron-LM's training.py for the schedule hooks on the model config.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
# Megatron-Core / bridge entry points are bound when the trainer imports this module, before any overlay runs.
from megatron.core.distributed import (DistributedDataParallel as _DDP, DistributedDataParallelConfig as _DDPConfig,
                                       finalize_model_grads as _finalize_model_grads)
from megatron.core.optimizer import OptimizerConfig as _OptimizerConfig, get_megatron_optimizer as _get_optimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler as _Scheduler
from megatron.core.transformer.enums import AttnBackend as _AttnBackend
from mcore_bridge import ModelConfig as _ModelConfig, get_mcore_model as _get_mcore_model, \
    hf_to_mcore_config as _hf_to_mcore_config

from infrarsi.adapters.megatron_core.config_map import model_dir_for  # noqa: F401  (re-export)

# Overlays may not touch these; they are re-applied after every overlay hook.
FROZEN_MODEL_KEYS = ('calculate_per_token_loss', 'moe_router_load_balancing_type', 'moe_aux_loss_coeff',
                     'moe_router_dtype', 'moe_expert_capacity_factor', 'moe_z_loss_coeff', 'bf16',
                     'perform_initialization', 'hidden_dropout')


def initialize_parallel(plan, extra: dict | None = None) -> None:
    from megatron.core import parallel_state as ps
    kwargs = dict(tensor_model_parallel_size=plan.tp, pipeline_model_parallel_size=plan.pp,
                  context_parallel_size=plan.cp, expert_model_parallel_size=plan.ep,
                  expert_tensor_parallel_size=plan.etp, distributed_timeout_minutes=30)
    kwargs.update(extra or {})
    ps.initialize_model_parallel(**kwargs)


def model_config(plan, model_dir: str, overrides: dict | None = None, deterministic: bool = False):
    """Bridge ModelConfig: HF-derived architecture, then layout/L0 kwargs, then the frozen objective."""
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
    kw = _hf_to_mcore_config(hf_config)
    bridge_derived = {k: kw.get(k) for k in ('moe_router_load_balancing_type', 'moe_router_score_function',
                                              'moe_router_topk', 'moe_router_pre_softmax', 'moe_shared_expert_gate',
                                              'moe_router_bias_update_rate', 'moe_router_enable_expert_bias')}
    kw.update(plan.model_kwargs)
    kw.update(overrides or {})
    kw.update({k: plan.model_kwargs[k] for k in FROZEN_MODEL_KEYS if k in plan.model_kwargs})
    kw['deterministic_mode'] = bool(deterministic)     # lifecycle runs only (runtime/lifecycle.DETERMINISTIC_ENV)
    if deterministic and not kw.get('multi_latent_attention'):
        # cuDNN's deterministic fused-attention backward asks for a >100 GiB workspace on 8K THD packs;
        # flash-attn's deterministic varlen backward needs < 1 GiB.
        kw['attention_backend'] = 'flash'
    kw['attention_backend'] = _AttnBackend[kw['attention_backend']]
    kw.update(params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16)
    cfg = _ModelConfig(**kw)
    return cfg, bridge_derived


def build_models(cfg) -> list:
    models = _get_mcore_model(cfg)
    device = torch.cuda.current_device()
    return [m.cuda(device).train() for m in models]


def load_hf_weights(cfg, models, model_dir: str) -> None:
    cfg.bridge.load_weights(models, model_dir)


def wrap_ddp(cfg, models, ddp_kwargs: dict):
    ddp_cfg = _DDPConfig(**ddp_kwargs)
    ddp_models = [_DDP(config=cfg, ddp_config=ddp_cfg, module=m, disable_bucketing=i > 0)
                  for i, m in enumerate(models)]
    return ddp_models, ddp_cfg


def build_optimizer(ddp_models, optimizer_kwargs: dict, objective: dict):
    opt_cfg = _OptimizerConfig(params_dtype=torch.bfloat16, **optimizer_kwargs)
    optimizer = _get_optimizer(opt_cfg, ddp_models)
    s, wd = objective['lr_schedule'], objective['weight_decay']
    scheduler = _Scheduler(optimizer, init_lr=0.0, max_lr=s['peak'], min_lr=s['min'],
                                        lr_warmup_steps=s['warmup_updates'], lr_decay_steps=s['horizon_updates'],
                                        lr_decay_style=s['type'], start_wd=wd, end_wd=wd,
                                        wd_incr_steps=s['horizon_updates'], wd_incr_style='constant')
    return optimizer, scheduler, opt_cfg


def install_schedule_hooks(cfg, ddp_models, ddp_cfg, optimizer) -> None:
    """Same hooks Megatron-LM's training.py installs on the model config before the first update."""
    def one(fns):
        return fns[0] if len(fns) == 1 else fns

    cfg.grad_scale_func = optimizer.scale_loss
    cfg.finalize_model_grads_func = _finalize_model_grads
    if ddp_cfg.overlap_grad_reduce:
        cfg.no_sync_func = one([m.no_sync for m in ddp_models])
        cfg.grad_sync_func = one([m.start_grad_sync for m in ddp_models])
    if ddp_cfg.overlap_param_gather and ddp_cfg.align_param_gather:
        cfg.param_sync_func = one([m.start_param_sync for m in ddp_models])


def hf_param_count(model_dir: str, tied: bool = False, exclude_prefixes: tuple = ()) -> dict:
    """Parameter count of the HF checkpoint from the safetensors headers only (no tensor reads).

    Tied checkpoints may still store ``lm_head.weight``; it is the embedding and is not counted twice.
    ``exclude_prefixes`` drops tensors the Megatron model does not instantiate (e.g. ``mtp.``).
    """
    import json
    import struct
    total, tensors = 0, 0
    for path in sorted(Path(model_dir).glob('*.safetensors')):
        with open(path, 'rb') as f:
            header = json.loads(f.read(struct.unpack('<Q', f.read(8))[0]))
        for name, meta in header.items():
            skip = name == '__metadata__' or (tied and name == 'lm_head.weight')
            if skip or name.startswith(tuple(exclude_prefixes)):
                continue
            n = 1
            for d in meta['shape']:
                n *= int(d)
            total += n
            tensors += 1
    return {'numel': total, 'tensors': tensors}


def unwrap(ddp_models) -> list:
    return [getattr(m, 'module', m) for m in ddp_models]


def env_versions() -> dict:
    import megatron.core
    import mcore_bridge
    out = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'megatron_core': megatron.core.__version__,
           'mcore_bridge': getattr(mcore_bridge, '__version__', None),
           'megatron_core_path': os.path.dirname(megatron.core.__file__),
           'mcore_bridge_path': os.path.dirname(mcore_bridge.__file__)}
    try:
        out['nccl'] = '.'.join(map(str, torch.cuda.nccl.version()))
    except Exception:
        out['nccl'] = None
    try:
        import transformer_engine
        out['transformer_engine'] = transformer_engine.__version__
    except Exception:
        out['transformer_engine'] = None
    try:
        import transformers
        out['transformers'] = transformers.__version__
    except Exception:
        out['transformers'] = None
    return out
