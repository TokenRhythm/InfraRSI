"""Validated L1 patch templates (weak-model protocol: templates before free patches, FINAL_PLAN 7.2).

Each template appends hook definitions to the pristine overlay file (a later `def` shadows the default), written
against the runtime's hook contract (infrarsi/runtime/overlays/__init__.py): override hooks return only keys from
overlays.ALLOWED, hooks keep their signatures, values and dtypes are preserved. Templates whose overrides are
L0-equivalent expose `l0_overrides(params)` so the caller validates the merged config before spending a screen.
"""
from __future__ import annotations

import dataclasses
from string import Template as _Text
from typing import Callable

from . import patch_guard

HP_GROUPS = ['dp', 'dp_cp', 'tp', 'cp', 'pp', 'ep', 'ep_dp']        # Megatron-Core process-group names (parallel_state.py)


@dataclasses.dataclass(frozen=True)
class Template:
    id: str
    file: str
    description: str
    params: dict                              # name -> {'type', 'values'|'range', 'default'}
    body: str                                 # hook source with $name fields (filled with Python literals)
    overrides: Callable | None = None         # params -> L0-equivalent overrides (validated by the caller)

    def schema(self) -> dict:
        return {'id': self.id, 'file': self.file, 'description': self.description, 'params': self.params}

    def values(self, params: dict | None) -> dict:
        return {k: (params or {}).get(k, spec['default']) for k, spec in self.params.items()}

    def l0_overrides(self, params: dict | None) -> dict:
        return self.overrides(self.values(params)) if self.overrides else {}


TEMPLATES = {t.id: t for t in [
    Template('host_sync.manual_gc', 'host_sync.py',
             'Disable automatic Python GC during training; collect every `interval` updates, between updates.',
             {'interval': {'type': 'int', 'range': [1, 1000], 'default': 10}},
             'import gc\n\n\ndef on_train_start(ctx):\n    gc.collect()\n    gc.disable()\n\n\n'
             'def after_update(ctx, step):\n    if step % $interval == 0:\n        gc.collect()\n\n\n'
             'def on_train_end(ctx):\n    gc.enable()\n'),
    Template('data_path.pinned_nonblocking', 'data_path.py',
             'Copy each micro-batch through pinned host memory with non-blocking host-to-device transfers.', {},
             'def to_device(ctx, arrays):\n    import torch\n'
             '    return {k: torch.from_numpy(v).pin_memory().to(ctx.device, non_blocking=True) for k, v in arrays.items()}\n'),
    Template('comm_groups.high_priority_streams', 'comm_groups.py',
             'Create the listed communicators on high-priority CUDA streams (overlapped collectives preempt compute).',
             {'groups': {'type': 'list_enum', 'values': HP_GROUPS, 'default': ['dp_cp']}},
             'def parallel_init_kwargs(ctx):\n    return {"high_priority_stream_groups": $groups}\n'),
    Template('comm_groups.pad_dp_buckets', 'comm_groups.py',
             'Pad gradient buckets to a large power of two for higher NCCL bus bandwidth at large DP.', {},
             'def ddp_config_overrides(ctx):\n    return {"pad_buckets_for_high_nccl_busbw": True}\n'),
    Template('state_placement.optimizer_offload', 'state_placement.py',
             'Offload a fraction of the optimizer state and its Adam update to the CPU (HybridDeviceOptimizer).',
             {'fraction': {'type': 'float', 'range': [0.05, 1.0], 'default': 1.0}, 'overlap': {'type': 'bool', 'default': True}},
             'def optimizer_config_overrides(ctx):\n    return {"optimizer_cpu_offload": True, "optimizer_offload_fraction": $fraction,'
             ' "overlap_cpu_optimizer_d2h_h2d": $overlap}\n',
             lambda v: {'optimizer_cpu_offload': True, 'optimizer_offload_fraction': v['fraction'], 'overlap_cpu_optimizer_d2h_h2d': v['overlap']}),
    Template('recompute_policy.selective_modules', 'recompute_policy.py',
             'Selective activation recompute of the listed submodules instead of full-layer recompute.',
             {'modules': {'type': 'list_enum', 'values': ['core_attn', 'layernorm', 'mlp', 'moe', 'moe_act', 'shared_experts'],
                          'default': ['layernorm', 'mlp']}},
             'def model_config_overrides(ctx):\n    return {"recompute_granularity": "selective", "recompute_method": None,'
             ' "recompute_num_layers": None, "recompute_modules": $modules}\n',
             lambda v: {'recompute_granularity': 'selective', 'recompute_method': None, 'recompute_num_layers': None,
                        'recompute_modules': list(v['modules'])}),
    Template('schedule.moe_ep_a2a_overlap', 'schedule.py',
             'Overlap MoE expert-parallel all-to-all with another micro-batch and delay expert wgrad (MoE, EP > 1).', {},
             'def model_config_overrides(ctx):\n    return {"overlap_moe_expert_parallel_comm": True, "delay_wgrad_compute": True}\n',
             lambda v: {'overlap_moe_expert_parallel_comm': True, 'delay_wgrad_compute': True}),
]}


def check_params(t: Template, params: dict) -> list[str]:
    errors = []
    if type(params) is not dict:
        return ['params must be an object']
    for k in params:
        if k not in t.params:
            errors.append(f'unknown param {k}')
    for k, spec in t.params.items():
        v = params.get(k, spec['default'])
        if spec['type'] == 'bool' and type(v) is not bool:
            errors.append(f'{k}: expected bool')
        elif spec['type'] == 'int' and (type(v) is not int or not spec['range'][0] <= v <= spec['range'][1]):
            errors.append(f'{k}: expected int in {spec["range"]}')
        elif spec['type'] == 'float' and (type(v) not in (int, float) or type(v) is bool or not spec['range'][0] <= v <= spec['range'][1]):
            errors.append(f'{k}: expected number in {spec["range"]}')
        elif spec['type'] == 'enum' and v not in spec['values']:
            errors.append(f'{k}: expected one of {spec["values"]}')
        elif spec['type'] == 'list_enum' and (type(v) is not list or not v or any(x not in spec['values'] for x in v)):
            errors.append(f'{k}: expected a non-empty subset of {spec["values"]}')
    return errors


def render(template_id: str, params: dict | None, base: dict[str, str] | None = None) -> tuple[dict[str, str] | None, list[str]]:
    """Returns ({overlay file: new full text}, errors)."""
    t = TEMPLATES.get(template_id)
    if t is None:
        return None, [f'unknown template {template_id!r}; available: {sorted(TEMPLATES)}']
    errors = check_params(t, params or {})
    if errors:
        return None, errors
    base = patch_guard.base_overlays() if base is None else base
    text = base.get(t.file)
    if text is None:
        return None, [f'{t.file}: no pristine overlay shipped; template unavailable']
    body = _Text(t.body).substitute({k: repr(v) for k, v in t.values(params).items()})
    return {t.file: text.rstrip('\n') + f'\n\n\n# --- infrarsi template {t.id} ---\n' + body}, []


def menu() -> list[dict]:
    return [t.schema() for t in TEMPLATES.values()]
