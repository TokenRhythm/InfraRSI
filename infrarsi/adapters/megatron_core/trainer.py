"""The training process proper: one per rank, spawned by the supervisor for each phase.

Order: CUDA + NCCL -> parallel state -> bridge model (HF weights, or a checkpoint when resuming) -> DDP ->
optimizer + scheduler -> update loop (frozen timers, per-rank signals, heartbeat, STOP) -> gates and
invariants -> receipt parts for the supervisor. The update itself is Megatron-Core's forward_backward_func
with calculate_per_token_loss (loss_func returns CE sum + supervised-token count) and optimizer.step().
"""
from __future__ import annotations

import datetime
import gc
import json
import math
import os
import random
import socket
import subprocess
import sys
import traceback
from collections import Counter
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from infrarsi.contracts import RunRequest, digest, utc_now
from infrarsi.runtime import gates, lifecycle, invariants as inv, receipt_writer as rw
from infrarsi.runtime import overlay_loader as l1
from infrarsi.runtime.heartbeat import Heartbeat
from infrarsi.runtime.signals import SignalWriter, read_signals
from infrarsi.runtime.stop import StopMonitor
from infrarsi.runtime.timers import PhaseClock, UpdateTimer
from infrarsi.adapters.megatron_core import config_map, model_build as mb
from infrarsi.adapters.megatron_core.data import DataSource, local_arrays

# Bound at import (before any L1 overlay is loaded): monkeypatching time/torch later cannot reach the timers.
from time import perf_counter as _clock  # noqa: E402
_barrier = dist.barrier
_all_gather_into_tensor = dist.all_gather_into_tensor
_max_alloc, _max_reserved = torch.cuda.max_memory_allocated, torch.cuda.max_memory_reserved
_memory_stats = torch.cuda.memory_stats
_current_device = torch.cuda.current_device
from megatron.core.packed_seq_params import PackedSeqParams as _PackedSeqParams  # noqa: E402
from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker as _moe_tracker  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed as _cuda_seed  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RECORDED_ENV = ('USE_MCORE_GDN', 'NCCL_SOCKET_IFNAME', 'GLOO_SOCKET_IFNAME', 'NCCL_IB_HCA', 'NCCL_NET_PLUGIN',
                'PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'TORCH_NCCL_ASYNC_ERROR_HANDLING') + tuple(lifecycle.DETERMINISTIC_ENV)
STAT_FIELDS = ('whole_update_s', 'fwd_bwd_s', 'optim_s', 'data_s', 'peak_alloc_gib', 'peak_reserved_gib', 'stop')


class ClipCounter:
    """Counts clip_grad_by_total_norm_fp32 calls (once per non-empty optimizer per step in MCore 0.18)."""

    def __init__(self):
        import megatron.core.optimizer.optimizer as mo
        self.count = 0
        original = mo.clip_grad_by_total_norm_fp32

        def counted(*args, **kwargs):
            self.count += 1
            return original(*args, **kwargs)
        mo.clip_grad_by_total_norm_fp32 = counted

    def take(self) -> int:
        n, self.count = self.count, 0
        return n


def micro_batches(t, order: tuple, consumed: list, sums: list, counts: dict):
    """This rank's micro-batches of one update, staged on the device lazily (inside the update timer).

    A plain generator over values taken from the trainer here: the schedule overlay receives it as
    ``data_iterator`` and must not be able to reach the trainer, its timer or its writers through it.
    ``consumed`` receives the pack ids of every micro-batch handed out; ``sums`` the checksums (with this
    process's secret weights) of the host arrays before ``data_path.to_device`` and of what it returned, which
    the trainer compares after the update, outside the timed region.
    """
    timer, source, plan, cp_rank, overlays, ctx = t.timer, t.source, t.plan, t.cp_rank, t.overlays, t.ctx
    w_host, w_dev = t.data_key

    def batches():
        for pids in order:
            with timer.data():
                arrays = local_arrays([source.pack(pid) for pid in pids], plan.cp, cp_rank)
                cu = arrays['cu_seqlens']
                max_seqlen = int(np.diff(cu).max())
                layout = _layout(arrays)
                host = [_host_sum(v, w_host) for v in arrays.values()]
                batch = overlays.call('data_path', 'to_device', ctx, arrays)
                if not _same_layout(batch, layout):
                    raise l1.OverlayViolation('data_path.to_device changed keys, shapes or dtypes of a micro-batch')
                sums.append((host, [_dev_sum(batch[k], w_dev) for k in layout]))
                batch = dict(batch, _counts=counts)        # our copy: the overlay keeps no handle on it
                padded = batch['cu_seqlens'] if plan.cp > 1 else None
                batch['packed_seq_params'] = _PackedSeqParams(
                    qkv_format='thd', cu_seqlens_q=batch['cu_seqlens'], cu_seqlens_kv=batch['cu_seqlens'],
                    max_seqlen_q=max_seqlen, max_seqlen_kv=max_seqlen,
                    cu_seqlens_q_padded=padded, cu_seqlens_kv_padded=padded)
            consumed.append(pids)
            yield batch
    return batches()


def _layout(arrays: dict) -> dict:
    return {k: (tuple(v.shape), torch.from_numpy(v[:0]).dtype) for k, v in arrays.items()}


def _same_layout(batch, layout: dict) -> bool:
    """to_device may change placement only: the same keys, shapes and dtypes as the host arrays."""
    return isinstance(batch, dict) and batch.keys() == layout.keys() and all(
        torch.is_tensor(batch[k]) and tuple(batch[k].shape) == s and batch[k].dtype == d for k, (s, d) in layout.items())


# Weighted integer sums with per-process random weights < 2**24: values are non-negative and < 2**18 (labels -100
# are already 0) and a micro-batch has < 2**17 positions, so neither side overflows int64.
def _host_sum(v, w) -> int:
    flat = v.reshape(-1).astype(np.int64)
    return int((flat * w[:flat.size]).sum())


def _dev_sum(x, w):
    flat = x.reshape(-1).to(torch.int64)
    return (flat * w[:flat.numel()]).sum()


def _forward_step(data_iterator, model):
    """forward_step_func of the schedule (a plain function: nothing of the trainer hangs off it). Counts the
    forward and, through a gradient hook on its output, the backward of every micro-batch (signals 'microbatches')."""
    b = next(data_iterator)
    counts = b['_counts']
    out = model(input_ids=b['tokens'], position_ids=b['position_ids'], attention_mask=None,
                labels=b['labels'], packed_seq_params=b['packed_seq_params'])
    counts['forward'] += 1
    if torch.is_tensor(out) and out.requires_grad:
        out.register_hook(partial(_count_backward, counts))
    return out, partial(_loss, b['loss_mask'])


def _count_backward(counts, grad):
    counts['backward'] += 1


def _loss(loss_mask, output_tensor):
    """calculate_per_token_loss contract: (sum of CE over supervised positions, their count, report)."""
    losses = output_tensor.float().view(-1)
    mask = loss_mask.view(-1)
    loss_sum = torch.sum(losses * mask.float())
    n = mask.sum().to(torch.int)
    return loss_sum, n, {'ce_sum': loss_sum.detach().clone(), 'n': n.detach().clone(), 'phys': mask.numel()}


class Trainer:
    def __init__(self, request_path: str, phase: str):
        self.clock = PhaseClock()
        self.phase = phase
        self.req = RunRequest.from_dict(json.loads(Path(request_path).read_text(encoding='utf-8')))
        self.out = Path(self.req.out_dir)
        self.rank, self.world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.task = self.req.task
        self.model_dir = mb.model_dir_for(self.task)
        self.facts = config_map.model_facts(self.model_dir)
        self.plan = config_map.make_plan(self.req, self.facts)
        life = self.req.lifecycle or {}
        self.save_at = int(life.get('save_at', self.req.steps)) if self.req.purpose == 'lifecycle' else None
        self.resume_steps = int(life.get('resume_steps', self.task['workload'].get('resume_continuation_steps', 10))) \
            if self.req.purpose == 'lifecycle' else 0
        if phase == 'train':
            self.first_step, self.last_step = 1, (self.save_at + self.resume_steps if self.save_at else self.req.steps)
        else:
            self.first_step, self.last_step = self.save_at + 1, self.save_at + self.resume_steps
        self.heartbeat = Heartbeat(self.out, enabled=self.rank == 0)
        self.heartbeat.beat('resume' if phase == 'resume' else 'load')
        self.heartbeat.start()
        self.stop = StopMonitor(self.out, self.req.wall_seconds_cap)
        self.events = {'exceptions': [], 'nccl_errors': 0, 'stopped': None}
        self.problems: list[str] = []
        self.rows: list[dict] = []
        self.consumed: dict[int, list[int]] = {}     # update -> pack ids this rank consumed
        self.deterministic = config_map.deterministic_lifecycle(self.req, self.facts)
        self.pins: list = []                         # identity / value pins of the live objects (gates.make_pins)
        self.optimizer_steps = 0
        self.parts_name = rw.PARTS_FILE if phase == 'train' else 'lifecycle/resume_parts.json'

    # ------------------------------------------------------------ setup

    def setup(self) -> None:
        torch.cuda.set_device(self.local_rank)
        # Lazy communicator creation as in Megatron-LM's initialize (no device_id): unused subgroups cost no memory.
        dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30))
        import megatron.core.parallel_state as ps
        import mcore_bridge  # noqa: F401  (applies the bridge's Megatron patches before any model code)
        from megatron.core.pipeline_parallel import get_forward_backward_func
        self.ps = ps
        # Everything the frozen update calls is bound or snapshotted before overlay code can run.
        self.fb_func = get_forward_backward_func(pp_size=self.plan.pp, vp_size=None)
        self.clip = ClipCounter()
        self.checker = gates.make_checker()          # closure over the pre-overlay identities
        if self.deterministic:                       # lifecycle: bit-faithful reference vs resumed updates
            wrong = {k: os.environ.get(k) for k, v in lifecycle.DETERMINISTIC_ENV.items() if os.environ.get(k) != v}
            if wrong:
                raise RuntimeError(f'lifecycle run without the deterministic environment: {wrong}')
            torch.use_deterministic_algorithms(True, warn_only=True)
        self.flags_baseline = gates.backends_flags()
        self.optimizer, self.ddp_models, self.models = None, [], []
        self.overlays = l1.load(self.req.overlay)
        want = getattr(self.req, 'effective_overlay_digest', None)
        if want and self.overlays.digest != want:
            raise l1.OverlayViolation(f'overlay_digest_mismatch: loaded {self.overlays.digest} != request {want}')
        self.overlays.guard = self.check_tamper
        self.ctx = SimpleNamespace(rank=self.rank, world_size=self.world, local_rank=self.local_rank,
                                   device=torch.device('cuda', self.local_rank), task_id=self.task['id'],
                                   plan=self.plan.summary(), config=dict(self.plan.effective), step=0,
                                   models=None, optimizer=None, phase=self.phase)
        mb.initialize_parallel(self.plan, self.overlays.overrides('comm_groups', 'parallel_init_kwargs', self.ctx))
        seed = int(self.req.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        _cuda_seed(seed)
        self.dp_rank, self.dp_size = ps.get_data_parallel_rank(), ps.get_data_parallel_world_size()
        self.cp_rank = ps.get_context_parallel_rank()
        self.clock.add('startup', _clock() - self.clock.t0)

        self.source = DataSource(self.task, self.req.input_plan, self.facts['vocab_size'],
                                 data_manifest=getattr(self.req, 'data_manifest', None))
        if self.last_step > self.source.max_updates:
            raise ValueError(f'{self.last_step} updates requested, data holds {self.source.max_updates}')

        with self.clock.phase('build'):
            self.heartbeat.beat('build')
            os.environ.setdefault('USE_MCORE_GDN', '1')
            overrides = self.overlays.overrides('recompute_policy', 'model_config_overrides', self.ctx)
            overrides.update(self.overlays.overrides('schedule', 'model_config_overrides', self.ctx))
            self.cfg, self.bridge_derived = mb.model_config(self.plan, self.model_dir, overrides, self.deterministic)
            self.models = mb.build_models(self.cfg)
        if self.phase == 'train':
            with self.clock.phase('load'):
                self.heartbeat.beat('load')
                mb.load_hf_weights(self.cfg, self.models, self.model_dir)
        self.pins.append(gates.make_pins([('models', self.models, None, 'len')] +
                                         [(f'models[{i}]', self.models, i, 'is') for i in range(len(self.models))]))
        self.pins.append(gates.make_module_pins(self.models))          # the whole tree, checked around every hook
        with self.clock.phase('build'):
            self.heartbeat.beat('build')
            self.overlays.call('recompute_policy', 'after_model_build', self.ctx, tuple(self.models))
            ddp_kwargs = dict(self.plan.ddp_kwargs)
            ddp_kwargs.update(self.overlays.overrides('comm_groups', 'ddp_config_overrides', self.ctx))
            ddp_kwargs.update(grad_reduce_in_fp32=True, use_distributed_optimizer=True, average_in_collective=False)
            self.ddp_models, self.ddp_cfg = mb.wrap_ddp(self.cfg, self.models, ddp_kwargs)
            opt_kwargs = dict(self.plan.optimizer_kwargs)
            opt_kwargs.update(self.overlays.overrides('state_placement', 'optimizer_config_overrides', self.ctx))
            opt_kwargs['use_precision_aware_optimizer'] = bool(opt_kwargs['optimizer_cpu_offload'])
            for k in ('lr', 'min_lr', 'weight_decay', 'adam_beta1', 'adam_beta2', 'adam_eps', 'clip_grad'):
                opt_kwargs[k] = self.plan.optimizer_kwargs[k]
            self.optimizer, self.scheduler, self.opt_cfg = mb.build_optimizer(self.ddp_models, opt_kwargs,
                                                                               self.plan.objective)
            mb.install_schedule_hooks(self.cfg, self.ddp_models, self.ddp_cfg, self.optimizer)
            self.pins.append(gates.make_pins(self.object_pins()))
            self.ctx.models, self.ctx.optimizer = tuple(self.ddp_models), self.optimizer
            self.overlays.call('state_placement', 'after_optimizer_build', self.ctx, self.optimizer)
        if self.phase == 'resume':
            with self.clock.phase('load'):
                self.heartbeat.beat('resume')
                step, self.load_info = lifecycle.load_checkpoint(self.ckpt_dir, self.ddp_models, self.optimizer,
                                                                 self.scheduler)
            if step != self.save_at or self.scheduler.num_steps != self.save_at:
                raise RuntimeError(f'checkpoint holds update {step} (scheduler {self.scheduler.num_steps}), '
                                   f'expected {self.save_at}')
            self.optimizer_steps = step
            self.resume_fingerprint = self.gathered_fingerprint()
            restores = [None] * self.world
            dist.all_gather_object(restores, self.load_info['steps_restored'])
            self.gathered_restores = restores
        self.pins.append(gates.make_pins(self.sub_optimizer_pins()))     # HDO rebuilds them on load
        self.structure = self.model_structure()
        rng = np.random.default_rng(int.from_bytes(os.urandom(8), 'little'))
        w = rng.integers(1, 1 << 24, size=self.plan.mbs * self.plan.seq_len + 2, dtype=np.int64)
        self.data_key = (w, torch.from_numpy(w).to(self.ctx.device))
        gc.collect()
        torch.cuda.empty_cache()
        self.check_tamper('setup')
        self.problems += self.frozen_problems('train start')
        self.env = self.environment()
        self.write_parts(final=False)

    @property
    def ckpt_dir(self) -> Path:
        return self.out / 'ckpt' / f'step{self.save_at}'

    # ------------------------------------------------------------ update loop

    def current_lr(self) -> float:
        lrs = {float(g['lr']) for g in self.optimizer.param_groups if g['params']}
        if len(lrs) != 1:
            self.problems.append(f'param groups disagree on LR: {sorted(lrs)}')
        return lrs.pop() if lrs else float('nan')

    def update(self, step: int) -> dict:
        t, plan = self.timer, self.plan
        t.begin()
        lr = self.current_lr()
        for m in self.ddp_models:
            m.zero_grad_buffer()
        self.optimizer.zero_grad()
        order = self.orders[step]
        self.overlays.call('data_path', 'prefetch', self.ctx, step, order)
        consumed, sums, counts = [], [], {'forward': 0, 'backward': 0}
        frozen = dict(forward_step_func=_forward_step, data_iterator=micro_batches(self, order, consumed, sums, counts),
                      model=self.ddp_models, num_microbatches=len(order), seq_length=plan.mbs * plan.seq_len,
                      micro_batch_size=1, forward_only=False)
        store = self.overlays.forward_backward(self.ctx, self.fb_func, frozen)
        if len(consumed) != len(order) or counts['forward'] != len(order) or counts['backward'] != len(order):
            raise l1.OverlayViolation(f'update {step}: {len(consumed)} micro-batches consumed, {counts} executed, '
                                      f'{len(order)} planned')
        self.consumed[step] = [p for pids in consumed for p in pids]
        t.mark('fwd_bwd')
        ok, grad_norm, _ = self.optimizer.step()
        lr_used = self.current_lr()                   # what the step used, before the scheduler moves it
        if ok:
            self.scheduler.step(increment=1)
            self.optimizer_steps += 1
        t.mark('optim')
        times = t.end()
        self.verify_batches(sums)
        if lr_used != lr:
            raise l1.OverlayViolation(f'learning rate changed inside update {step}: {lr} -> {lr_used}')
        # Adam counters as the real step left them, before any overlay hook runs again (deep_check compares)
        self.steps_after_update = lifecycle.optimizer_steps(self.optimizer)
        times.update(lr=lr, ok=bool(ok), grad_norm=float(grad_norm) if grad_norm is not None else float('nan'),
                     clip_calls=self.clip.take(), store=store, microbatches=counts['backward'])
        return times

    def reduce_scalars(self, store) -> tuple[float, int, int]:
        ps = self.ps
        red = torch.zeros(3, dtype=torch.float64, device='cuda')
        for d in store or []:
            red[0] += d['ce_sum'].double()
            red[1] += d['n'].double()
            red[2] += float(d['phys'])
        dist.all_reduce(red, group=ps.get_data_parallel_group(with_context_parallel=True))
        if self.plan.pp > 1:
            dist.all_reduce(red, group=ps.get_pipeline_model_parallel_group())
        ce_sum, n, phys = red.tolist()
        return (ce_sum / n if n > 0 else float('nan')), int(round(n)), int(round(phys))

    def aux_loss(self) -> float | None:
        if not self.plan.is_moe:
            return None
        ps, cfg = self.ps, self.cfg
        tracker = _moe_tracker()
        size = cfg.num_layers + (cfg.mtp_num_layers or 0)
        v = torch.zeros(size, dtype=torch.float64, device='cuda')
        entry = tracker.metrics.get('load_balancing_loss')
        if entry is not None:
            v[:entry.values.numel()] += entry.values.detach().double()
        dist.all_reduce(v, group=ps.get_tensor_and_context_parallel_group())
        dist.all_reduce(v, group=ps.get_data_parallel_group())
        if self.plan.pp > 1:
            dist.all_reduce(v, group=ps.get_pipeline_model_parallel_group())
        for e in tracker.metrics.values():
            e.values.zero_()
        return float(v.sum()) / (self.num_moe_layers() * (self.plan.packs_per_update // self.plan.mbs))

    def num_moe_layers(self) -> int:
        """Mirrors megatron.core MoEMetricsTracker._count_moe_layers (None / int / list) plus MTP layers."""
        freq, n = self.cfg.moe_layer_freq, self.cfg.num_layers
        count = n if freq is None else sum(1 for i in range(n) if i % freq == 0) if isinstance(freq, int)             else int(sum(freq))
        return count + (self.cfg.mtp_num_layers or 0)

    def train(self) -> None:
        self.timer = UpdateTimer(barrier=_barrier, device=self.local_rank)
        self.orders = self.plan_orders()
        sig_name = 'signals' if self.phase == 'train' else 'signals_resume'
        self.signals = SignalWriter(self.out, self.rank, sig_name)
        self.flags_start = gates.backends_flags()
        self.heartbeat.beat('train', self.first_step - 1)
        self.overlays.call('host_sync', 'on_train_start', self.ctx)
        train_start = _clock()
        stats = torch.empty(self.world * len(STAT_FIELDS), dtype=torch.float64, device='cuda')
        for step in range(self.first_step, self.last_step + 1):
            self.ctx.step = step
            self.overlays.call('host_sync', 'before_update', self.ctx, step)
            r = self.update(step)
            if step == self.first_step:
                self.clock.add('compile_first_step', r['whole_update_s'])
            main_ce, n_sup, phys = self.reduce_scalars(r['store'])
            aux = self.aux_loss()
            mem = _memory_stats(self.local_rank)
            row = {'step': step, 'whole_update_s': r['whole_update_s'], 'fwd_bwd_s': r['fwd_bwd_s'],
                   'optim_s': r['optim_s'], 'data_s': r['data_s'], 'inter_step_s': r['inter_step_s'],
                   'utc_start_s': r['utc_start_s'], 'utc_end_s': r['utc_end_s'], 'cuda_whole_s': r['cuda_whole_s'],
                   'peak_alloc_gib': _max_alloc(self.local_rank) / 2**30,
                   'peak_reserved_gib': _max_reserved(self.local_rank) / 2**30,
                   'alloc_retries': int(mem.get('num_alloc_retries', 0)), 'alloc_ooms': int(mem.get('num_ooms', 0)),
                   'microbatches': r['microbatches']}
            flag = self.stop.flag(poll_file=self.rank == 0)
            local = torch.tensor([row['whole_update_s'], row['fwd_bwd_s'] or float('nan'), row['optim_s'] or float('nan'),
                                  row['data_s'], row['peak_alloc_gib'], row['peak_reserved_gib'], float(flag)],
                                 dtype=torch.float64, device='cuda')
            _all_gather_into_tensor(stats, local)
            table = stats.view(self.world, len(STAT_FIELDS)).cpu()
            stop = int(table[:, -1].max())
            if self.rank == 0:
                row.update(main_ce=main_ce, aux_loss=aux, grad_norm=r['grad_norm'], lr=r['lr'], physical_tokens=phys,
                           supervised_tokens=n_sup, optimizer_steps=self.optimizer_steps, clip_calls=r['clip_calls'])
                slow = int(table[:, 0].argmax())
                print(f'INFRARSI_STEP phase={self.phase} step={step} max_rank_whole_s={float(table[slow, 0]):.4f} '
                      f'slowest={slow} ce={main_ce:.6f} gn={r["grad_norm"]:.6f} lr={r["lr"]:.4e} '
                      f'peak_reserved_gib={float(table[:, 5].max()):.2f}', flush=True)
            if not r['ok']:
                self.problems.append(f'step {step}: optimizer step skipped')
            if r['clip_calls'] < 1:
                self.problems.append(f'step {step}: gradient clipping did not run')
            self.signals.write(row)
            self.rows.append(row)
            self.heartbeat.beat('train', step)
            self.overlays.call('host_sync', 'after_update', self.ctx, step)
            self.check_tamper(f'after update {step}')
            self.deep_check(step)
            if self.save_at and self.phase == 'train' and step == self.save_at:
                self.save()
            if stop:
                self.events['stopped'] = StopMonitor.reason(stop)
                break
        self.overlays.call('host_sync', 'on_train_end', self.ctx)
        self.clock.add('train', _clock() - train_start)
        self.signals.close()
        self.flags_end = gates.backends_flags()

    def save(self) -> None:
        self.heartbeat.beat('save', self.save_at)
        with self.clock.phase('save'), lifecycle.synced_params(self.ddp_models):
            info = lifecycle.save_checkpoint(self.ckpt_dir, self.ddp_models, self.optimizer, self.scheduler,
                                             self.save_at)
            info['fingerprint'] = self.gathered_fingerprint()
        if self.rank == 0:
            rw.write_json_atomic(self.out / 'lifecycle' / 'save.json', dict(info, utc=utc_now()))

    def gathered_fingerprint(self) -> list:
        out = [None] * self.world
        dist.all_gather_object(out, lifecycle.state_digest(self.ddp_models, self.optimizer))
        return out

    # ------------------------------------------------------------ gates and invariants

    def check_tamper(self, when: str) -> None:
        """Overlays may not replace frozen functions or shadow methods of the live optimizers / models.

        Runs before and after every request-supplied overlay hook, around Megatron's forward_backward_func, after
        setup, after every update and at the end.
        """
        inst = []
        if self.optimizer is not None:
            leaves = gates.leaf_optimizers(self.optimizer)
            inst += [('optimizer', o, gates.OPTIMIZER_METHODS) for o in [self.optimizer] + leaves]
            for leaf in leaves:
                inner = getattr(leaf, 'optimizer', None)
                for o in [inner] + list(getattr(inner, 'sub_optimizers', None) or []):
                    if o is not None:
                        inst.append(('inner_optimizer', o, gates.INNER_OPTIMIZER_METHODS))
        inst += [('model', m, gates.MODEL_METHODS) for m in self.ddp_models]
        inst += [('raw_model', m, gates.RAW_MODEL_METHODS) for m in self.models]
        inst.append(('trainer', self, gates.TRAINER_METHODS))
        bad = self.checker(inst)
        for pin in self.pins:
            bad += pin()
        if _current_device() != self.local_rank:
            bad.append(f'cuda current device {_current_device()} != {self.local_rank}')
        if bad:
            raise l1.OverlayViolation(f'{when}: frozen dependencies replaced or shadowed: {bad}')

    def object_pins(self) -> list:
        """Live DDP wrappers, optimizer leaves and inner optimizers, schedule callbacks and frozen config values,
        captured before any overlay hook receives the model wrappers or the optimizer."""
        e = [('ddp_models', self.ddp_models, None, 'len')]
        for i, d in enumerate(self.ddp_models):
            e += [(f'ddp_models[{i}]', self.ddp_models, i, 'is'), (f'ddp_models[{i}].module', d, 'module', 'is'),
                  (f'ddp_models[{i}].config', d, 'config', 'is'), (f'models[{i}].config', self.models[i], 'config', 'is')]
        e += [(f'cfg.{k}', self.cfg, k, 'is') for k in ('finalize_model_grads_func', 'grad_scale_func', 'no_sync_func',
                                                          'grad_sync_func', 'param_sync_func')]
        e += [(f'cfg.{k}', self.cfg, k, 'eq') for k in mb.FROZEN_MODEL_KEYS + ('deterministic_mode',)]
        chained = getattr(self.optimizer, 'chained_optimizers', None)
        if chained is not None:
            e += [('chained_optimizers', chained, None, 'len')]
            e += [(f'chained_optimizers[{i}]', chained, i, 'is') for i in range(len(chained))]
        for j, leaf in enumerate(gates.leaf_optimizers(self.optimizer)):
            e += [(f'leaf[{j}].optimizer', leaf, 'optimizer', 'is'), (f'leaf[{j}].config', leaf, 'config', 'is'),
                  (f'leaf[{j}].is_stub_optimizer', leaf, 'is_stub_optimizer', 'eq')]
            e += [(f'leaf[{j}].config.{k}', leaf.config, k, 'eq')
                  for k in ('lr', 'min_lr', 'weight_decay', 'adam_beta1', 'adam_beta2', 'adam_eps', 'clip_grad',
                            'main_params_dtype', 'main_grads_dtype', 'exp_avg_dtype', 'exp_avg_sq_dtype')]
        e += [('scheduler.optimizer', self.scheduler, 'optimizer', 'is')]
        e += [(f'scheduler.{k}', self.scheduler, k, 'eq')
              for k in ('lr_decay_style', 'lr_decay_steps', 'lr_warmup_steps', 'max_lr', 'min_lr', 'start_wd',
                        'end_wd', 'wd_incr_style')]
        e += [(f'ddp_cfg.{k}', self.ddp_cfg, k, 'eq')
              for k in ('grad_reduce_in_fp32', 'average_in_collective', 'use_distributed_optimizer')]
        return e

    def sub_optimizer_pins(self) -> list:
        """HybridDeviceOptimizer's CPU / GPU optimizers (its load_state_dict rebuilds them, so pinned after it)."""
        e = []
        for j, leaf in enumerate(gates.leaf_optimizers(self.optimizer)):
            inner = getattr(leaf, 'optimizer', None)
            cpu = getattr(inner, 'cpu_optimizers', None)
            if isinstance(cpu, list):
                e += [(f'leaf[{j}].cpu_optimizers', inner, 'cpu_optimizers', 'is'),
                      (f'leaf[{j}].gpu_optimizer', inner, 'gpu_optimizer', 'is'),
                      (f'leaf[{j}].cpu_optimizers.len', cpu, None, 'len')]
                e += [(f'leaf[{j}].cpu_optimizers[{i}]', cpu, i, 'is') for i in range(len(cpu))]
        return e

    def model_structure(self) -> tuple:
        return tuple((i, n, id(m)) for i, chunk in enumerate(self.models) for n, m in chunk.named_modules())

    def deep_check(self, step: int) -> None:
        """After every update, outside the timed region: full module tree, Adam step counters, frozen configs."""
        bad = []
        if self.model_structure() != self.structure:
            bad.append('model module tree changed')
        if lifecycle.optimizer_steps(self.optimizer) != self.steps_after_update:
            bad.append('optimizer step counters changed after the update')
        bad += lifecycle.step_counter_problems(self.optimizer, self.optimizer_steps)[:4]
        if bad:
            raise l1.OverlayViolation(f'after update {step}: {bad}')
        self.problems += self.frozen_problems(f'after update {step}')

    def plan_orders(self) -> dict:
        """Every update's micro-batch order, fixed before the timed loop: a permutation of the planned packs and
        identical on all ranks of one data-parallel replica (TP / CP / PP peers consume the same micro-batches)."""
        orders = {}
        for step in range(self.first_step, self.last_step + 1):
            lists = self.source.microbatches(step - 1, self.dp_rank, self.dp_size, self.plan.mbs)
            got = self.overlays.call('schedule', 'microbatch_order', self.ctx, step, [list(mb) for mb in lists])
            order = tuple(tuple(int(p) for p in mb) for mb in got)
            if sorted(order) != sorted(tuple(mb) for mb in lists):
                raise l1.OverlayViolation(f'schedule.microbatch_order changed the micro-batch set of update {step}')
            orders[step] = order
        mine = (self.dp_rank, digest([orders[s] for s in sorted(orders)]))
        seen = [None] * self.world
        dist.all_gather_object(seen, mine)
        if any(d != mine[1] for r, d in seen if r == self.dp_rank):
            raise l1.OverlayViolation('micro-batch order differs between ranks of one data-parallel replica')
        return orders

    def verify_batches(self, sums: list) -> None:
        """Values the model received == the planned host arrays (checksums with this process's secret weights)."""
        if not sums:
            return
        host = [h for hs, _ in sums for h in hs]
        dev = torch.stack([d for _, ds in sums for d in ds]).tolist()
        if dev != host:
            raise l1.OverlayViolation('data_path.to_device changed micro-batch values')

    def frozen_problems(self, when: str) -> list[str]:
        """Live configs after overlays vs the frozen contract (objective, precision, reduction semantics)."""
        p, cfg, plan = [], self.cfg, self.plan
        if cfg.calculate_per_token_loss is not True:
            p.append('calculate_per_token_loss disabled')
        if plan.is_moe:
            for k in ('moe_router_load_balancing_type', 'moe_aux_loss_coeff', 'moe_router_dtype',
                      'moe_expert_capacity_factor', 'moe_z_loss_coeff'):
                if getattr(cfg, k, None) != plan.model_kwargs[k]:
                    p.append(f'{k}={getattr(cfg, k, None)!r} != {plan.model_kwargs[k]!r}')
        if cfg.params_dtype != torch.bfloat16 or not cfg.bf16:
            p.append(f'model precision {cfg.params_dtype} bf16={cfg.bf16}')
        want = plan.optimizer_kwargs
        for leaf in gates.leaf_optimizers(self.optimizer):
            oc = leaf.config
            for k in ('lr', 'min_lr', 'weight_decay', 'adam_beta1', 'adam_beta2', 'adam_eps', 'clip_grad'):
                if getattr(oc, k) != want[k]:
                    p.append(f'optimizer {k}={getattr(oc, k)!r} != {want[k]!r}')
            for k in ('main_params_dtype', 'main_grads_dtype', 'exp_avg_dtype', 'exp_avg_sq_dtype'):
                if getattr(oc, k) != torch.float32:
                    p.append(f'optimizer {k}={getattr(oc, k)}')
            if not (oc.decoupled_weight_decay and oc.use_distributed_optimizer and oc.bf16 and oc.optimizer == 'adam'):
                p.append('optimizer is not bf16 distributed AdamW')
        d = self.ddp_cfg
        if not d.grad_reduce_in_fp32 or d.average_in_collective or not d.use_distributed_optimizer:
            p.append('DDP reduction differs from fp32 sum-reduce with distributed optimizer')
        s, obj = self.scheduler, plan.objective['lr_schedule']
        if (s.lr_decay_style, s.lr_decay_steps, s.lr_warmup_steps, s.max_lr, s.min_lr) != \
                (obj['type'], obj['horizon_updates'], obj['warmup_updates'], obj['peak'], obj['min']) or \
                (s.start_wd, s.end_wd, s.wd_incr_style) != (plan.objective['weight_decay'],) * 2 + ('constant',):
            p.append('LR / weight-decay scheduler differs from the frozen schedule')
        for g in self.optimizer.param_groups:
            if g.get('lr_mult', 1.0) != 1.0 or g.get('wd_mult', 1.0) not in (0.0, 1.0) or \
                    g.get('max_lr') != obj['peak'] or g.get('min_lr') != obj['min']:
                p.append('param group overrides LR/WD multipliers')
                break
        return [f'{when}: {x}' for x in p]

    def param_inventory(self) -> dict:
        """Global multiset of trainable parameter shapes, each counted once (layout invariant)."""
        ps = self.ps
        tp_rank, dpcp_rank = ps.get_tensor_model_parallel_rank(), ps.get_data_parallel_rank(with_context_parallel=True)
        etp_rank, edp_rank = ps.get_expert_tensor_parallel_rank(), ps.get_expert_data_parallel_rank()
        first_stage = ps.is_pipeline_first_stage()
        shapes, frozen = Counter(), []
        for m in mb.unwrap(self.ddp_models):
            for name, p in m.named_parameters():
                if not p.requires_grad:
                    frozen.append(name)
                    continue
                if getattr(p, 'shared_embedding', False) and not first_stage:
                    continue                      # tied output copy on the last PP stage
                expert = not getattr(p, 'allreduce', True)
                own = (edp_rank == 0 and etp_rank == 0) if expert else (dpcp_rank == 0 and tp_rank == 0)
                if not own:
                    continue
                shape = list(p.shape)
                if getattr(p, 'tensor_model_parallel', False):
                    shape[p.partition_dim] *= self.plan.etp if expert else self.plan.tp
                shapes[('expert' if expert else 'dense', tuple(shape))] += 1
        gathered = [None] * self.world
        dist.all_gather_object(gathered, (dict(shapes), frozen))
        total, frozen_all = Counter(), []
        for s, f in gathered:
            total.update(s)
            frozen_all += f
        counts = {f'{kind}:{list(shape)}': n for (kind, shape), n in total.items()}
        numel = sum(n * math.prod(shape) for (_, shape), n in total.items())
        return {'trainable_params': numel, 'trainable_param_digest': inv.param_inventory_digest(counts),
                'frozen_params': sorted(set(frozen_all))[:32], 'param_tensors': sum(total.values())}

    def environment(self) -> dict:
        info = (socket.gethostname(), torch.cuda.get_device_name(self.local_rank))
        gathered = [None] * self.world
        dist.all_gather_object(gathered, info)
        env = mb.env_versions()
        env.update(hostnames=sorted({h for h, _ in gathered}), gpu_names=sorted({g for _, g in gathered}),
                   python=sys.version.split()[0], bridge_derived=inv.jsonable(self.bridge_derived),
                   env={k: os.environ.get(k) for k in sorted(set(self.plan.env['whitelist']) | set(RECORDED_ENV))})
        if self.rank == 0:
            try:
                env['driver'] = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                               capture_output=True, text=True, timeout=30).stdout.split()[0]
            except Exception:
                env['driver'] = None
        return env

    def invariants(self) -> dict:
        ppu, seq = self.plan.packs_per_update, self.plan.seq_len
        census = gates.dtype_census(mb.unwrap(self.ddp_models), self.optimizer)
        gathered = [None] * self.world
        dist.all_gather_object(gathered, census['counts'])
        counts = {k: Counter() for k in gates.EXPECTED_DTYPES}
        for c in gathered:
            for k, v in c.items():
                counts[k].update(v)
        census_all = {'counts': {k: dict(v) for k, v in counts.items()}}
        self.problems += gates.census_problems(census_all)
        self.problems += gates.flags_problems(self.flags_baseline, self.flags_start, self.flags_end)
        self.problems += self.frozen_problems('train end')
        inventory = self.param_inventory()
        if inventory['frozen_params']:
            self.problems.append(f'frozen parameters present: {inventory["frozen_params"][:4]}')
        steps = list(range(self.first_step, self.last_step + 1))
        ps = self.ps
        owner = (ps.get_tensor_model_parallel_rank() == 0 and ps.get_context_parallel_rank() == 0
                 and ps.get_pipeline_model_parallel_rank() == 0)
        consumed = [None] * self.world
        dist.all_gather_object(consumed, self.consumed if owner else None)
        done = [r['step'] for r in self.rows]
        update_packs = [sorted(p for c in consumed if c for p in c.get(s, [])) for s in done]
        lrs = [r['lr'] for r in self.rows if 'lr' in r]
        cfg = self.cfg
        loss_def = inv.loss_definition({k: getattr(cfg, k, None) for k in inv.LOSS_LIVE_KEYS})
        model_view = inv.model_config_view({k: getattr(cfg, k, None) for k in inv.MODEL_SEMANTIC_KEYS})
        return {
            'gbs_packs': ppu, 'physical_tokens_per_update': ppu * seq,
            'supervised_tokens_total': sum(int(r.get('supervised_tokens', 0)) for r in self.rows),
            'optimizer_steps': self.optimizer_steps, 'clip_calls_total': sum(int(r.get('clip_calls', 0)) for r in self.rows),
            'lr_schedule_digest': inv.lr_schedule_digest(lrs),
            'lr_schedule_expected_digest': inv.lr_schedule_digest([inv.lr_at(s, self.plan.objective['lr_schedule'])
                                                                   for s in steps[:len(lrs)]]),
            **inventory, 'hf_param_count': mb.hf_param_count(self.model_dir, self.facts['tie_word_embeddings']),
            'data_order_digest': self.source.order_digest,           # INDEX.json data_order_digest[<plan>]
            'data_consumed_digest': inv.data_order_digest(update_packs),  # global pack set of each update run
            'data_consumed_per_update': [digest(u) for u in update_packs],  # update k -> digest(sorted pack ids)
            'supervised_tokens_per_update': {self.req.input_plan: [int(r.get('supervised_tokens', -1))
                                                                   for r in self.rows]},
            'data': self.source.describe(),
            'loss_definition_digest': digest(loss_def), 'loss_definition': loss_def,
            'router_topk': cfg.moe_router_topk if self.plan.is_moe else None,
            'capacity_factor': cfg.moe_expert_capacity_factor if self.plan.is_moe else None,
            'num_experts': cfg.num_moe_experts, 'frozen_zone_digest': inv.frozen_zone_digest(PACKAGE_ROOT),
            'model_config_digest': digest(model_view), 'model_config': model_view,
            'dtype_census': census_all['counts'], 'dtype_devices_rank0': census['devices'],
            'model_defined_fp32_params': census['model_defined_fp32_params'],
            'backends_flags': self.flags_end, 'backends_flags_baseline': self.flags_baseline,
            'clock_check': self.clock_check(),
        }

    def clock_check(self) -> dict:
        """This rank's CUDA-event span / host perf_counter span per update, and UTC vs host interval error."""
        ratios = [r['cuda_whole_s'] / r['whole_update_s'] for r in self.rows if r.get('cuda_whole_s')]
        utc_err = [abs((r['utc_end_s'] - r['utc_start_s']) - r['whole_update_s']) for r in self.rows]
        return {'rank': self.rank, 'updates': len(ratios), 'cuda_over_host_min': min(ratios, default=None),
                'cuda_over_host_max': max(ratios, default=None), 'utc_minus_host_max_abs_s': max(utc_err, default=None),
                'utc_span_s': (self.rows[-1]['utc_end_s'] - self.rows[0]['utc_start_s']) if self.rows else None}

    def expected(self, steps: list[int]) -> dict:
        """Frozen per-step expectations: LR schedule values and supervised tokens from the data metadata."""
        sup = {s: self.source.update_supervised_tokens(s - 1) for s in steps} if self.rank == 0 else {}
        return {'lr': {s: inv.lr_at(s, self.plan.objective['lr_schedule']) for s in steps},
                'supervised_tokens': sup, 'physical_tokens': self.plan.packs_per_update * self.plan.seq_len,
                'microbatches': self.plan.ga}

    # ------------------------------------------------------------ parts and lifecycle

    def write_parts(self, final: bool, **extra) -> None:
        if self.rank != 0:
            return
        parts = {'final': final, 'phase': self.phase, 'overlay_digest': self.overlays.digest if hasattr(self, 'overlays') else None,
                 'target_steps': self.last_step if self.phase == 'train' else self.resume_steps,
                 'phases': self.clock.snapshot(), 'env': getattr(self, 'env', {}), 'events': self.events,
                 'effective_config': self.plan.effective, 'plan': self.plan.summary(),
                 'problems': list(self.problems), 'utc': utc_now()}
        parts.update(extra)
        rw.write_json_atomic(self.out / self.parts_name, parts)

    def finish(self) -> None:
        self.check_tamper('train end')
        steps = [r['step'] for r in self.rows]
        invariants = self.invariants()
        expected = self.expected(steps)
        if self.phase == 'resume':
            self.finish_resume(invariants, expected)
            return
        self.write_parts(final=True, invariants=invariants, expected=expected)
        self.heartbeat.close('done')

    def finish_resume(self, invariants: dict, expected: dict) -> None:
        tol = (self.task.get('numerics') or {}).get('tolerance') or {'main_ce_rel_max_per_step': 0.005}
        ltol = lifecycle.lifecycle_tolerance(self.task)
        requested = bool((self.req.lifecycle or {}).get('deterministic'))
        result = {'saved_at': self.save_at, 'resumed_ok': len(self.rows) == self.resume_steps,
                  'deterministic_unavailable': 'packed-THD Gated DeltaNet has no deterministic Megatron-Core path'
                  if requested and not self.deterministic else None,
                  'deterministic': self.deterministic and bool(self.cfg.deterministic_mode)
                  and torch.are_deterministic_algorithms_enabled(),
                  'attention_backend': getattr(self.cfg.attention_backend, 'name', str(self.cfg.attention_backend))}
        if self.deterministic:      # AMENDMENT-002: what the deterministic phases replaced, with the requested values
            result['deterministic_overrides'] = lifecycle.deterministic_overrides(
                self.plan.effective, self.plan.env['set'], result['attention_backend'])
        if self.rank == 0:
            ref = read_signals(self.out / 'signals').get(0, [])
            result['continuation'] = lifecycle.compare_continuation(
                [r for r in ref if r['step'] > self.save_at], [r for r in self.rows], tol, ltol)
            result['resumed_bitwise_updates'] = result['continuation'].get('bitwise_updates')   # report only
            saved = json.loads((self.out / 'lifecycle' / 'save.json').read_text(encoding='utf-8'))
            result['state_exact'] = saved.get('fingerprint') == self.resume_fingerprint
            result['state_digest_saved'] = digest(saved.get('fingerprint'))
            result['state_digest_restored'] = digest(self.resume_fingerprint)
            result['fingerprint_restored'] = self.resume_fingerprint
            result['optimizer_steps_restored'] = self.gathered_restores
            result['state_digest_mismatch'] = sorted({f'rank{r}:{cat}' for r, (a, b) in
                                                      enumerate(zip(saved.get('fingerprint') or [], self.resume_fingerprint))
                                                      for cat in set(a) | set(b) if a.get(cat) != b.get(cat)})
            # AMENDMENT-001 self-check on the resumed trajectory, from the saved update on
            pairs = rw.repeated_pairs([r for r in ref if r['step'] == self.save_at] + list(self.rows))
            result['self_check'] = {'repeated_pairs': pairs}
            if pairs:
                self.problems.append(f'repeated adjacent scalars (main_ce, grad_norm) at resumed updates {pairs[:4]}')
            result['resume_scalars_match'] = result['continuation']['match'] and not pairs and (
                result['state_exact'] or not ltol.get('state_digest_exact', True))
            sup_ok = all(r.get('supervised_tokens') == expected['supervised_tokens'].get(r['step']) for r in self.rows)
            result['resume_tokens_match'] = sup_ok
        export, life = None, self.req.lifecycle or {}
        with lifecycle.synced_params(self.ddp_models):     # full parameters after the last optimizer step
            if life.get('export', True):
                self.heartbeat.beat('export')
                with self.clock.phase('export'):
                    export = lifecycle.export_hf(self.cfg, mb.unwrap(self.ddp_models), self.model_dir,
                                                 self.out / 'hf_export', self.rank, self.facts['tie_word_embeddings'])
            if life.get('forward_compare', True):
                self.heartbeat.beat('compare')
                tokens = np.asarray(self.source.pack(int(self.source.order[0]))['tokens'][:lifecycle.FIXED_BATCH_TOKENS])
                logits = lifecycle.mcore_logits(self.ddp_models, self.plan, tokens, self.fb_func)
                if logits is not None:
                    (self.out / 'lifecycle').mkdir(parents=True, exist_ok=True)
                    torch.save({'tokens': torch.as_tensor(tokens), 'logits': logits},
                               self.out / 'lifecycle' / 'mcore_logits.pt')
                dist.barrier()
        if self.rank == 0:
            result['export'] = export
            result['export_ok'] = bool(export and export['export_ok'])
            self.write_parts(final=True, invariants=invariants, expected=expected, lifecycle=result,
                             rows=self.rows)
        self.heartbeat.close('done')

    def fail(self, exc: BaseException) -> None:
        tb = traceback.format_exc()
        kind = 'oom' if isinstance(exc, torch.cuda.OutOfMemoryError) else \
            'nccl' if 'nccl' in (str(exc) + tb).lower() else type(exc).__name__
        record = {'rank': self.rank, 'phase': self.phase, 'kind': kind, 'error': repr(exc)[:2000],
                  'traceback': tb[-6000:], 'step': self.rows[-1]['step'] if self.rows else None, 'utc': utc_now()}
        rw.write_json_atomic(self.out / 'failures' / f'{self.phase}_rank{self.rank}.json', record)
        self.events['exceptions'].append({k: record[k] for k in ('rank', 'phase', 'kind', 'error', 'step')})
        if kind == 'nccl':
            self.events['nccl_errors'] += 1
        try:
            self.write_parts(final=False)
        finally:
            self.heartbeat.close('failed')


def run(request_path: str, phase: str) -> int:
    code, trainer = 0, None
    try:
        trainer = Trainer(request_path, phase)
        trainer.setup()
        trainer.train()
        trainer.finish()
    except BaseException as exc:      # noqa: BLE001 - the receipt must record every failure
        print(f'INFRARSI_FAILED rank={os.environ.get("RANK")} phase={phase} {exc!r}', file=sys.stderr, flush=True)
        traceback.print_exc()
        if trainer is not None:
            trainer.fail(exc)
        code = 1
    finally:
        if dist.is_initialized():
            try:
                if code == 0:
                    dist.barrier()
                dist.destroy_process_group()
            except Exception:
                pass
    return code
