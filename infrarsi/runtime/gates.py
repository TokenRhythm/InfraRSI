"""Gates independent of the loss tolerance: dtype census, torch.backends flag audit, first-use exactness.

A silent precision drop is invalid even when the loss stays within tolerance (design section 1.7), so the
census and the flag audit are checked on every receipt. torch is imported lazily; the comparison helpers
are pure Python.
"""
from __future__ import annotations

import functools
from collections import Counter

# Functions whose identity must not change once overlays can run (module attribute paths). Clocks, collectives,
# serialization and hashing, the Megatron schedule / loss / grad finalisation / optimizer step / LR scheduler,
# and this package's own timers and writers. Checked after setup, after every update and before the parts file.
SENTINEL_PATHS = (
    'time.perf_counter', 'time.time', 'time.monotonic', 'json.dumps', 'json.loads', 'json.JSONEncoder',
    'json.encoder.JSONEncoder.encode', 'json.encoder.JSONEncoder.iterencode', 'hashlib.sha256', 'builtins.open',
    'builtins.sorted', 'torch.cuda.synchronize', 'torch.cuda.Event', 'torch.cuda.Event.record',
    'torch.cuda.Event.elapsed_time', 'torch.cuda.max_memory_allocated', 'torch.cuda.max_memory_reserved',
    'torch.cuda.memory_stats', 'torch.cuda.reset_peak_memory_stats', 'torch.distributed.barrier',
    'torch.distributed.all_reduce', 'torch.distributed.all_gather_into_tensor', 'torch.distributed.all_gather_object',
    'torch.distributed.broadcast', 'torch.distributed.all_gather', 'torch.distributed.reduce_scatter_tensor',
    'torch.distributed.reduce_scatter', 'torch.distributed.all_to_all_single', 'torch.distributed.all_to_all',
    'mcore_bridge.model.gpt_model.GPTModel.forward', 'mcore_bridge.model.gpt_model.GPTModel._postprocess',
    'megatron.core.models.gpt.gpt_model.GPTModel.forward',
    'megatron.core.pipeline_parallel.get_forward_backward_func',
    'megatron.core.pipeline_parallel.schedules.get_forward_backward_func',
    'megatron.core.pipeline_parallel.schedules.forward_backward_no_pipelining',
    'megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving',
    'megatron.core.pipeline_parallel.schedules.forward_step',
    'megatron.core.pipeline_parallel.schedules.forward_step_calc_loss',
    'megatron.core.pipeline_parallel.schedules.backward_step',
    'megatron.core.distributed.finalize_model_grads',
    'megatron.core.distributed.finalize_model_grads.finalize_model_grads',
    'megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.finish_grad_sync',
    'megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.scale_gradients',
    'megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.zero_grad_buffer',
    'megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.step',
    'megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.prepare_grads',
    'megatron.core.optimizer.optimizer.ChainedOptimizer.step',
    'megatron.core.optimizer.optimizer.MegatronOptimizer.clip_grad_norm',
    'megatron.core.optimizer.optimizer.MegatronOptimizer.get_grad_norm',
    'megatron.core.optimizer.optimizer.clip_grad_by_total_norm_fp32',
    'megatron.core.optimizer.optimizer.get_grad_norm_fp32',
    'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.step_with_ready_grads',
    'megatron.core.optimizer_param_scheduler.OptimizerParamScheduler.step',
    'megatron.core.optimizer_param_scheduler.OptimizerParamScheduler.get_lr',
    'megatron.core.tensor_parallel.vocab_parallel_cross_entropy',
    'megatron.core.models.common.language_module.language_module.LanguageModule.compute_language_model_loss',
    'megatron.core.transformer.moe.moe_utils.switch_load_balancing_loss_func',
    'megatron.core.transformer.moe.router.TopKRouter.routing',
    'torch.cuda.current_device', 'torch.cuda.current_stream',
    'infrarsi.runtime.timers._clock', 'infrarsi.runtime.timers._utc', 'infrarsi.runtime.timers._sync',
    'infrarsi.runtime.timers._current_device', 'infrarsi.runtime.timers._current_stream',
    'infrarsi.runtime.timers._reset_peak', 'infrarsi.runtime.timers._record',
    'infrarsi.runtime.timers.UpdateTimer.mark',
    'infrarsi.runtime.timers.UpdateTimer.begin', 'infrarsi.runtime.timers.UpdateTimer.end',
    'infrarsi.runtime.signals.dump_row', 'infrarsi.runtime.signals.SignalWriter.write',
    'infrarsi.runtime.receipt_writer.write_json_atomic', 'infrarsi.runtime.receipt_writer._json_dumps',
    'infrarsi.adapters.megatron_core.data.local_arrays', 'infrarsi.adapters.megatron_core.data.zigzag_index',
    'infrarsi.adapters.megatron_core.model_build.model_config',
    'infrarsi.adapters.megatron_core.model_build.build_models',
    'infrarsi.adapters.megatron_core.model_build.wrap_ddp',
    'infrarsi.adapters.megatron_core.model_build.build_optimizer',
    'infrarsi.adapters.megatron_core.model_build.install_schedule_hooks',
    'infrarsi.adapters.megatron_core.trainer._loss', 'infrarsi.adapters.megatron_core.trainer.Trainer.update',
    'infrarsi.adapters.megatron_core.trainer._forward_step',
    'infrarsi.adapters.megatron_core.trainer.Trainer.reduce_scalars',
    'infrarsi.adapters.megatron_core.trainer.Trainer.aux_loss',
    'infrarsi.adapters.megatron_core.trainer.Trainer.frozen_problems',
    'infrarsi.adapters.megatron_core.trainer.Trainer.invariants',
    'infrarsi.adapters.megatron_core.trainer.Trainer.write_parts',
    'infrarsi.adapters.megatron_core.trainer.micro_batches',
    'infrarsi.adapters.megatron_core.trainer._same_layout',
    'infrarsi.adapters.megatron_core.trainer.Trainer.train',
    'infrarsi.adapters.megatron_core.trainer.Trainer.check_tamper',
    'infrarsi.runtime.lifecycle.save_checkpoint', 'infrarsi.runtime.lifecycle.load_checkpoint',
    'infrarsi.runtime.lifecycle.state_digest', 'infrarsi.runtime.lifecycle.compare_continuation',
    'infrarsi.runtime.gates.dtype_census', 'infrarsi.runtime.gates.census_problems',
    'infrarsi.runtime.gates.backends_flags', 'infrarsi.runtime.invariants.lr_at',
    'infrarsi.runtime.overlay_loader.Overlays.call', 'infrarsi.runtime.overlay_loader.Overlays.overrides',
    'infrarsi.runtime.overlay_loader.Overlays.forward_backward',
)
# Instance attributes that would shadow the frozen class methods on the live optimizer / model objects.
OPTIMIZER_METHODS = ('step', 'prepare_grads', 'step_with_ready_grads', 'get_grad_norm', 'clip_grad_norm', 'zero_grad',
                     'scale_loss', 'get_loss_scale')
INNER_OPTIMIZER_METHODS = ('step', 'zero_grad', 'load_state_dict', 'state_dict')
MODEL_METHODS = ('forward', 'finish_grad_sync', 'start_grad_sync', 'scale_gradients', 'zero_grad_buffer', 'no_sync',
                 'start_param_sync')
RAW_MODEL_METHODS = ('forward', '_postprocess', 'compute_language_model_loss', '_preprocess')
TRAINER_METHODS = ('update', 'train', 'save', 'check_tamper', 'current_lr', 'reduce_scalars', 'aux_loss',
                   'frozen_problems', 'invariants', 'clock_check', 'expected', 'write_parts', 'finish',
                   'finish_resume', 'deep_check', 'plan_orders', 'verify_batches', 'model_structure')


def resolve_parent(path: str):
    """(object holding the last attribute, attribute name), walking attributes from the top-level package.

    An attribute wins over a same-named submodule, so a package re-export (megatron.core.distributed.
    finalize_model_grads, the function) resolves to what ``from package import name`` would bind.
    """
    import importlib
    parts = path.split('.')
    obj = importlib.import_module(parts[0])
    for i, name in enumerate(parts[1:-1], start=1):
        nxt = getattr(obj, name, None)
        if nxt is None:
            nxt = importlib.import_module('.'.join(parts[:i + 1]))
        obj = nxt
    if not hasattr(obj, parts[-1]) and isinstance(obj, type(importlib)):
        importlib.import_module(path)
    return obj, parts[-1]


def resolve(path: str):
    parent, name = resolve_parent(path)
    return getattr(parent, name)


def make_checker(paths=SENTINEL_PATHS):
    """Closure built before overlay code runs: parents, names and original objects are captured here, and later
    checks use only captured builtins, so replacing this module's functions cannot disable the check."""
    entries, _getattr = [], getattr
    for p in paths:
        try:
            parent, name = resolve_parent(p)
            entries.append((p, parent, name, _getattr(parent, name)))
        except (ImportError, AttributeError):
            continue

    def check(instances=()) -> list:
        bad = []
        for p, parent, name, original in entries:
            if _getattr(parent, name, None) is not original:
                bad.append(p)
        for label, obj, names in instances:
            own = _getattr(obj, '__dict__', None) or {}
            bad.extend(f'{label}.{n}' for n in names if n in own)
        return bad
    check.paths = tuple(e[0] for e in entries)
    return check


def make_pins(entries):
    """Closure over live objects captured now. Each entry is (label, holder, key, mode): mode 'is' -> the holder
    must still hold the identical object at attribute / index ``key``; 'eq' -> a value of the identical type that
    compares equal (the type is checked first, so a replacement object never gets to answer the comparison);
    'len' -> a list of the same length. Returns the labels that changed."""
    captured, _getattr, _len, _type = [], getattr, len, type

    def read(holder, key, mode):
        if mode == 'len':
            return _len(holder)
        return holder[key] if isinstance(key, int) else _getattr(holder, key, None)
    for label, holder, key, mode in entries:
        captured.append((label, holder, key, mode, read(holder, key, mode)))

    def check() -> list:
        bad = []
        for label, holder, key, mode, original in captured:
            try:
                now = read(holder, key, mode)
            except (IndexError, KeyError, TypeError):
                bad.append(label)
                continue
            if mode == 'is':
                changed = now is not original
            elif mode == 'eq':
                changed = _type(now) is not _type(original) or now != original
            else:
                changed = now != original
            if changed:
                bad.append(label)
        return bad
    return check


def make_module_pins(models):
    """Identity of the whole module tree of the model chunks: every holder's ``_modules`` dict, its exact ordered
    names (ModuleList order is execution order) and every child must stay the same, and no instance attribute may
    shadow a child. Dict lookups only (~1k modules per model), cheap enough to run around every overlay hook and
    around the schedule itself."""
    entries, seen, stack = [], set(), [(f'models[{i}]', m) for i, m in enumerate(models)]
    while stack:
        label, mod = stack.pop()
        if id(mod) in seen:
            continue
        seen.add(id(mod))
        own = mod.__dict__
        children = own.get('_modules')
        if children is None:
            continue
        entries.append((label, own, children, tuple(children.items()), tuple(children)))
        stack += [(f'{label}.{n}', c) for n, c in children.items() if c is not None]
    _type, _str, _tuple, _len, _zip, _any = type, str, tuple, len, zip, any

    def check() -> list:
        bad = []
        for label, own, children, items, names in entries:
            if own.get('_modules') is not children:
                bad.append(f'{label}._modules')
                continue
            keys = _tuple(children)          # exact str keys in the original order (str checked before comparing)
            if _len(keys) != _len(names) or _any(_type(k) is not _str or k != n for k, n in _zip(keys, names)):
                bad.append(f'{label} children')
                continue
            for name, child in items:
                if children.get(name) is not child or name in own:
                    bad.append(f'{label}.{name}')
        return bad
    return check


def sentinel_snapshot(paths=SENTINEL_PATHS) -> dict:
    """{path: object}; unresolvable paths (optional modules) are skipped and listed under None."""
    out = {}
    for p in paths:
        try:
            out[p] = resolve(p)
        except (ImportError, AttributeError):
            continue
    return out


def sentinel_changes(snapshot: dict) -> list[str]:
    changed = []
    for p, obj in snapshot.items():
        try:
            now = resolve(p)
        except (ImportError, AttributeError):
            now = None
        if now is not obj:
            changed.append(p)
    return changed


def shadowed(obj, names) -> list[str]:
    own = getattr(obj, '__dict__', {})
    return [n for n in names if n in own]


EXPECTED_DTYPES = {'param': 'torch.bfloat16', 'grad': 'torch.float32', 'master': 'torch.float32',
                   'exp_avg': 'torch.float32', 'exp_avg_sq': 'torch.float32', 'model_fp32': 'torch.float32'}
# Parameters and buffers the model definition itself keeps in fp32 (GDN decay terms A_log / dt_bias, router expert
# bias). Codex R3 HIGH: they are exempt from the bf16 parameter contract only when exactly torch.float32 and are then
# censused apart ('model_fp32', empty on models without them); any other non-bf16 dtype is censused as a parameter and
# fails. Their gradients are censused like every other parameter's.
FP32_PARAM_FRAGMENTS = ('A_log', 'dt_bias', 'expert_bias')
OPTIONAL_CENSUS = ('model_fp32',)


def model_defined_fp32(name: str) -> bool:
    return any(f in name for f in FP32_PARAM_FRAGMENTS)


def backends_flags() -> dict:
    import torch
    b = torch.backends
    flags = {'float32_matmul_precision': torch.get_float32_matmul_precision(),
             'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
             'default_dtype': str(torch.get_default_dtype())}
    probes = {
        'cuda.matmul.allow_tf32': lambda: b.cuda.matmul.allow_tf32,
        'cuda.matmul.allow_bf16_reduced_precision_reduction': lambda: b.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'cuda.matmul.allow_fp16_reduced_precision_reduction': lambda: b.cuda.matmul.allow_fp16_reduced_precision_reduction,
        'cuda.matmul.fp32_precision': lambda: b.cuda.matmul.fp32_precision,
        'cudnn.allow_tf32': lambda: b.cudnn.allow_tf32,
        'cudnn.enabled': lambda: b.cudnn.enabled,
        'cudnn.deterministic': lambda: b.cudnn.deterministic,
        'cudnn.benchmark': lambda: b.cudnn.benchmark,
        'sdp.flash': lambda: b.cuda.flash_sdp_enabled(),
        'sdp.mem_efficient': lambda: b.cuda.mem_efficient_sdp_enabled(),
        'sdp.math': lambda: b.cuda.math_sdp_enabled(),
        'sdp.cudnn': lambda: b.cuda.cudnn_sdp_enabled(),
    }
    for name, probe in probes.items():
        try:
            value = probe()
            flags[name] = value if isinstance(value, (bool, int, str)) else str(value)
        except Exception:   # attribute absent in this torch build
            flags[name] = None
    return flags


def flags_problems(baseline: dict, start: dict, end: dict) -> list[str]:
    problems = []
    for label, snap in (('train start', start), ('train end', end)):
        for k in sorted(set(baseline) | set(snap)):
            if baseline.get(k) != snap.get(k):
                problems.append(f'torch.backends {k} changed at {label}: {baseline.get(k)!r} -> {snap.get(k)!r}')
    return problems


def leaf_optimizers(optimizer):
    if optimizer is None:
        return []
    if hasattr(optimizer, 'chained_optimizers'):
        return [leaf for o in optimizer.chained_optimizers for leaf in leaf_optimizers(o)]
    return [optimizer]


def _inner_states(inner):
    """Optimizer state dicts as they physically live (HybridDeviceOptimizer: its CPU/GPU sub-optimizers)."""
    subs = getattr(inner, 'sub_optimizers', None)
    if subs:
        return [st for sub in subs for st in sub.state.values()]
    return list(getattr(inner, 'state', {}).values())


def dtype_census(models, optimizer) -> dict:
    import torch
    counts = {k: Counter() for k in EXPECTED_DTYPES}
    devices = {k: Counter() for k in EXPECTED_DTYPES}
    exempt = []

    def add(kind, t):
        counts[kind][str(t.dtype)] += 1
        devices[kind][t.device.type] += 1

    for model in models:
        for name, p in model.named_parameters():
            if model_defined_fp32(name) and p.dtype == torch.float32:
                exempt.append(f'{name}:{p.dtype}')
                add('model_fp32', p)
            else:
                add('param', p)                      # an fp16 / fp64 A_log lands here and fails the bf16 contract
            grad = getattr(p, 'main_grad', None)
            if grad is not None:
                add('grad', grad)
            elif p.requires_grad:
                counts['grad']['missing'] += 1
        for name, b in model.named_buffers():         # router expert_bias is a buffer (fp32 in Megatron-Core)
            if model_defined_fp32(name):
                exempt.append(f'{name}:{b.dtype}')
                add('model_fp32', b)
    for leaf in leaf_optimizers(optimizer):
        groups = list(getattr(leaf, 'shard_fp32_from_float16_groups', None) or [])
        groups += list(getattr(leaf, 'shard_fp32_groups', None) or [])
        for group in groups:
            for t in group:
                if t is not None:
                    add('master', t)
        inner = getattr(leaf, 'optimizer', None)
        for sub in getattr(inner, 'sub_optimizers', None) or []:
            if not groups:   # hybrid offload keeps the fp32 masters as the sub-optimizers' params
                for g in sub.param_groups:
                    for t in g['params']:
                        add('master', t)
        for st in _inner_states(inner):
            for key in ('exp_avg', 'exp_avg_sq'):
                if torch.is_tensor(st.get(key)):
                    add(key, st[key])
            if torch.is_tensor(st.get('master_param')):
                add('master', st['master_param'])
    return {'counts': {k: dict(v) for k, v in counts.items()},
            'devices': {k: dict(v) for k, v in devices.items()},
            'model_defined_fp32_params': sorted(exempt)[:64], 'model_defined_fp32_count': len(exempt)}


def census_problems(census: dict) -> list[str]:
    problems = []
    for kind, want in EXPECTED_DTYPES.items():
        got = census['counts'].get(kind, {})
        if not got and kind not in OPTIONAL_CENSUS:
            problems.append(f'dtype census: no {kind} tensors found')
        bad = {d: n for d, n in got.items() if d != want}
        if bad:
            problems.append(f'dtype census: {kind} expected {want}, found {bad}')
    return problems


class FirstUseGate:
    """Wrap an L2 replacement op: on its first real input compare with an fp64 reference, once per process.

    ``reference`` receives the same arguments cast to float64 and must return the reference output.
    """

    def __init__(self, name: str, reference, atol: float, rtol: float):
        self.name, self.reference, self.atol, self.rtol = name, reference, atol, rtol
        self.report = None

    def __call__(self, fn):
        import torch

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            out = fn(*args, **kwargs)
            if self.report is None:
                ref_args = [a.detach().double() if torch.is_tensor(a) else a for a in args]
                with torch.no_grad():
                    ref = self.reference(*ref_args, **kwargs)
                err = (out.detach().double() - ref).abs()
                bound = self.atol + self.rtol * ref.abs()
                self.report = {'op': self.name, 'max_abs_err': float(err.max()),
                               'ok': bool((err <= bound).all())}
                if not self.report['ok']:
                    raise AssertionError(f'first-use exactness failed for {self.name}: {self.report}')
            return out
        return wrapped
