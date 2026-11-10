"""Loader of the L1 overlays (frozen zone): hook dispatch, allowed override keys, sha256 checks.

The six modules in ``runtime/overlays/`` are documented no-ops. An optimizer agent may replace any of them via
``RunRequest.overlay = {'dir': <remote dir>, 'files': {'<module>.py': sha256}}``; a replaced file only needs to
define the hooks it changes (missing hooks fall back to the defaults below). The loader verifies every
sha256, and the ``*_overrides`` hooks may only return keys from ``ALLOWED``. Objective, data order, LR
schedule, timers, receipts, gates and invariants are not reachable from here; the trainer re-reads the live
configs after the overlays ran and records violations as invalid receipts.

Call points (trainer order):
  comm_groups.parallel_init_kwargs(ctx)            before initialize_model_parallel
  recompute_policy.model_config_overrides(ctx)     before the bridge ModelConfig is built
  schedule.model_config_overrides(ctx)             before the bridge ModelConfig is built
  recompute_policy.after_model_build(ctx, models)  after HF weights are loaded, before DDP wrapping
  comm_groups.ddp_config_overrides(ctx)            before DDP wrapping
  state_placement.optimizer_config_overrides(ctx)  before get_megatron_optimizer
  state_placement.after_optimizer_build(ctx, opt)  after the optimizer exists
  host_sync.on_train_start(ctx) / on_train_end(ctx)
  host_sync.before_update(ctx, step) / after_update(ctx, step)   outside the update timer
  data_path.prefetch(ctx, step, microbatches)       start of every update (inside the timer)
  data_path.to_device(ctx, arrays)                  once per micro-batch (inside the timer)
  schedule.microbatch_order(ctx, step, microbatches)   order of this rank's micro-batches in an update
  schedule.forward_backward(ctx, run, **kwargs)     wraps Megatron-Core's forward_backward_func call
"""
from __future__ import annotations

import hashlib
import importlib
import threading
import types
from pathlib import Path

from infrarsi.contracts import digest, file_digest

PACKAGE = 'infrarsi.runtime.overlays'
HOOK_MODULES = ('schedule', 'recompute_policy', 'comm_groups', 'state_placement', 'host_sync', 'data_path')

ALLOWED = {
    'comm_groups.parallel_init_kwargs': {'nccl_communicator_config_path', 'use_sharp', 'high_priority_stream_groups'},
    'comm_groups.ddp_config_overrides': {'overlap_grad_reduce', 'overlap_param_gather', 'align_param_gather',
                                         'bucket_size', 'num_buckets', 'pad_buckets_for_high_nccl_busbw'},
    'recompute_policy.model_config_overrides': {
        'recompute_granularity', 'recompute_method', 'recompute_num_layers', 'recompute_modules',
        'moe_layer_recompute', 'distribute_saved_activations', 'cpu_offloading', 'cpu_offloading_num_layers',
        'cpu_offloading_activations', 'cpu_offloading_weights'},
    'schedule.model_config_overrides': {
        'moe_shared_expert_overlap', 'overlap_moe_expert_parallel_comm', 'delay_wgrad_compute', 'tp_comm_overlap',
        'moe_deepep_num_sms', 'batch_p2p_comm', 'overlap_p2p_comm', 'defer_embedding_wgrad_compute'},
    'state_placement.optimizer_config_overrides': {
        'optimizer_cpu_offload', 'optimizer_offload_fraction', 'overlap_cpu_optimizer_d2h_h2d', 'pin_cpu_grads',
        'pin_cpu_params', 'use_torch_optimizer_for_cpu_offload'},
}


class OverlayViolation(RuntimeError):
    pass


class Overlays:
    def __init__(self, modules: dict, defaults: dict, files: dict[str, str], source: str):
        self.modules, self.defaults, self.files, self.source = modules, defaults, files, source
        self.guard = None        # trainer tamper check, run before and after every request-supplied hook

    @property
    def digest(self) -> str:
        """Effective hook set (defaults replaced by request files): contracts.overlay_digest of {name: sha256}."""
        return digest([[name, sha] for name, sha in sorted(self.files.items())])

    def hook(self, module: str, name: str):
        """(function, supplied): the request's hook when its file defines it, else the built-in default."""
        mod = self.modules[module]
        fn = getattr(mod, name, None)
        if callable(fn) and mod is not self.defaults[module]:
            return fn, True
        return getattr(self.defaults[module], name), False

    def call(self, module: str, name: str, *args, **kwargs):
        fn, supplied = self.hook(module, name)
        guard = self.guard if supplied else None     # built-in defaults are frozen code; no check around them
        if guard is not None:
            guard(f'before {module}.{name}')
        out = fn(*args, **kwargs)
        if guard is not None:
            guard(f'after {module}.{name}')
        return out

    def forward_backward(self, ctx, fb_func, frozen: dict):
        """schedule.forward_backward(ctx, run, **frozen): the hook may wrap the schedule, but ``run`` executes
        Megatron-Core's forward_backward_func exactly once with the frozen arguments, and the hook must return
        its result unchanged."""
        produced, started, lock, guard = [], [], threading.Lock(), self.guard

        def run(**kwargs):
            if kwargs.keys() != frozen.keys() or any(kwargs[k] is not v and kwargs[k] != v for k, v in frozen.items()):
                raise OverlayViolation('schedule.forward_backward altered the schedule arguments')
            with lock:                        # claimed before the schedule starts: a second call never runs it
                if started:
                    raise OverlayViolation('schedule.forward_backward ran the schedule twice')
                started.append(True)
            if guard is not None:
                guard('before the schedule')
            produced.append(fb_func(**frozen))
            if guard is not None:
                guard('after the schedule')
            return produced[0]
        returned = self.call('schedule', 'forward_backward', ctx, run, **frozen)
        if len(produced) != 1 or returned is not produced[0]:
            raise OverlayViolation(f'schedule.forward_backward ran the schedule {len(produced)} times or replaced '
                                   f'its result')
        return produced[0]

    def overrides(self, module: str, name: str, ctx) -> dict:
        out = dict(self.call(module, name, ctx) or {})
        extra = set(out) - ALLOWED[f'{module}.{name}']
        if extra:
            raise OverlayViolation(f'{module}.{name} may not set {sorted(extra)}')
        return out


def effective_files(spec: dict | None) -> dict[str, str]:
    """{'<hook>.py': sha256} of the hook set a request runs: the code tree's six default files, each replaced by the
    request's file of the same name after checking its sha256 on disk. Hashes only; no overlay code is executed."""
    root = Path(__file__).resolve().parent / 'overlays'
    files = {f'{m}.py': file_digest(root / f'{m}.py') for m in HOOK_MODULES}
    if spec:
        for fname, want in sorted((spec.get('files') or {}).items()):
            mod = fname[:-3] if fname.endswith('.py') else fname
            if mod not in HOOK_MODULES or Path(fname).name != fname:
                raise OverlayViolation(f'overlay file {fname!r} is outside the L1 zone')
            got = hashlib.sha256((Path(spec['dir']) / f'{mod}.py').read_bytes()).hexdigest()
            if got != want:
                raise OverlayViolation(f'overlay {fname} sha256 {got} != request {want}')
            files[f'{mod}.py'] = got
    return files


def effective_digest(spec: dict | None) -> str:
    """contracts.overlay_digest of effective_files(spec): what RunRequest.effective_overlay_digest pins."""
    return digest([[name, sha] for name, sha in sorted(effective_files(spec).items())])


def load(spec: dict | None) -> Overlays:
    """Built-in no-op hooks, or the request's overlay files after sha256 verification."""
    defaults = {m: importlib.import_module(f'{PACKAGE}.{m}') for m in HOOK_MODULES}
    files = {f'{m}.py': file_digest(defaults[m].__file__) for m in HOOK_MODULES}
    if not spec:
        return Overlays(dict(defaults), defaults, files, 'builtin')
    root = Path(spec['dir'])
    modules, sources = dict(defaults), {}
    for fname, want in sorted((spec.get('files') or {}).items()):      # verify every file before running any
        mod = fname[:-3] if fname.endswith('.py') else fname
        if mod not in HOOK_MODULES or Path(fname).name != fname:
            raise OverlayViolation(f'overlay file {fname!r} is outside the L1 zone')
        source = (root / f'{mod}.py').read_bytes()
        got = hashlib.sha256(source).hexdigest()
        if got != want:
            raise OverlayViolation(f'overlay {fname} sha256 {got} != request {want}')
        sources[mod] = source
        files[f'{mod}.py'] = got
    try:                                          # the static L1 guard re-checks exactly these bytes
        from infrarsi.actions.patch_guard import static_scan
    except ImportError as e:
        raise OverlayViolation(f'static overlay guard unavailable in this code tree: {e}') from e
    for mod, source in sources.items():
        findings = static_scan(source.decode('utf-8'), f'{mod}.py')
        if findings:
            raise OverlayViolation(f'{mod}.py rejected by the static guard: {findings[:3]}')
    for mod, source in sources.items():           # executes exactly the bytes that were hashed
        module = types.ModuleType(f'infrarsi_overlay_{mod}')
        module.__file__ = str(root / f'{mod}.py')
        exec(compile(source, module.__file__, 'exec'), module.__dict__)
        modules[mod] = module
    return Overlays(modules, defaults, files, 'request')
