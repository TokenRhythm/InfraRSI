"""ScreenRunner protocol and a synthetic fake. The real runner is agent/harness_runner.py::HarnessScreenRunner over
the dispatcher's request-level runner (harness.run_screen); it returns harness-validated receipts unchanged.

Contract for real runners:
  run(task, config, overlay, purpose, run_id) -> receipt dict (infrarsi-receipt-v1, validated by the harness)
  - task: contracts.TaskSpec; config: resolved L0 config with an 'env' block; overlay: None or {overlay file: full
    text} (complete set or changed files; the runner uploads and digests them); purpose: 'screen' | 'probe' |
    'score' | 'lifecycle'; run_id: the RUN_MATRIX run id (attempt ids are the runner's).
  - Always returns a receipt, also on failure (valid=false, invalid_reasons), never raises for training failures.
  - Adds a driver-measured cost block: receipt['cost'] = {'gpu_seconds', 'wall_seconds'} (world_size x occupancy
    seconds, including load/compile/failure; excluding queue time before the lease).
  - Optional: receipt['numerics'] = {'class','passed','main_ce_rel_max','grad_norm': {first_update_rel, log_bias_window},
    'reasons'} vs the FR
    plan-A reference, and receipt['profile'] = {phase: {'window_mean_max_rank_s','window_mean_min_rank_s'}} for
    phases whole_update_s, fwd_bwd_s, optim_s, data_s (re-derived from signals).
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Protocol

from ..actions import validate_l0
from ..contracts import SCHEMA_RECEIPT, digest


class ScreenRunner(Protocol):
    def run(self, task, config: dict, overlay: dict | None, purpose: str, run_id: str) -> dict: ...


def _noise(key: str, i: int, scale: float) -> float:
    h = hashlib.sha256(f'{key}:{i}'.encode()).digest()
    u = int.from_bytes(h[:8], 'big') / 2 ** 64
    return (u - 0.5) * 2 * scale


def reference_invariants(task_raw: dict, model_facts: dict | None) -> dict:
    w = task_raw['workload']
    mf = model_facts or {}
    return {'gbs_packs': w['packs_per_update'], 'physical_tokens_per_update': w['physical_tokens_per_update'],
            'supervised_tokens_total': int(w['physical_tokens_per_update'] * w['screen_steps'] * 0.6),
            'optimizer_steps': w['screen_steps'], 'lr_schedule_digest': digest(task_raw['objective']['lr_schedule']),
            'trainable_params': int(mf.get('params', 1_700_000_000)), 'trainable_param_digest': digest(['all-params', task_raw['model']['repo_id']]),
            'data_order_digest': digest([task_raw['data']['key'], 'plan-A']), 'loss_definition_digest': digest(task_raw['objective']['loss']),
            'router_topk': mf.get('moe_router_topk'), 'capacity_factor': None, 'num_experts': mf.get('num_experts') or None,
            'frozen_zone_digest': 'fz-' + digest('frozen')[:16], 'model_config_digest': digest([task_raw['model']['repo_id'], mf.get('num_layers')]),
            'dtype_census': {'param': 'bfloat16', 'grad': 'float32', 'master': 'float32', 'exp_avg': 'float32', 'exp_avg_sq': 'float32'},
            'backends_flags': {'cuda.matmul.allow_tf32': False, 'cudnn.allow_tf32': False,
                               'cuda.matmul.allow_bf16_reduced_precision_reduction': True,
                               'cuda.matmul.allow_fp16_reduced_precision_reduction': True}}


def synthetic_receipt(task_raw: dict, config: dict, *, step_time_s: float, peak_reserved_gib: float, run_id: str,
                      attempt_id: str, purpose: str = 'screen', model_facts: dict | None = None, valid: bool = True,
                      invalid_reasons=(), rsd: float = 0.01, startup_s: float = 240.0, events: dict | None = None,
                      loss_base: float = 2.0, inter_step_ratio: float = 1.005, overlay_digest: str | None = None) -> dict:
    """A complete infrarsi-receipt-v1 for tests (deterministic noise keyed by attempt_id)."""
    w = task_raw['workload']
    steps = int(w['screen_steps'])
    a, b = w['scoring_window']
    world = int(task_raw['hardware']['gpus'])
    per_step = []
    for s in range(1, steps + 1):
        t = step_time_s * (1 + (0.25 if s == 1 else 0.0) + _noise(attempt_id, s, rsd * 1.7))
        per_step.append({'step': s, 'max_rank_whole_s': round(t, 6), 'slowest_rank': s % world,
                         'main_ce': round(loss_base - 0.004 * s + _noise('ce' + task_raw['id'], s, 1e-4), 6),
                         'aux_loss': 0.0 if not (model_facts or {}).get('num_experts') else round(0.012 + _noise('aux', s, 1e-5), 7),
                         'grad_norm': round(1.1 - 0.003 * s + _noise('gn' + task_raw['id'], s, 1e-3), 6),
                         'lr': round(1e-6 + 0.5 * 9e-6 * (1 + math.cos(math.pi * (s - 1) / 1024)), 12),
                         'physical_tokens': w['physical_tokens_per_update'],
                         'supervised_tokens': int(w['physical_tokens_per_update'] * 0.6)})
    win = [r['max_rank_whole_s'] for r in per_step if a <= r['step'] <= b]
    mean = sum(win) / len(win)
    sd = math.sqrt(sum((x - mean) ** 2 for x in win) / max(1, len(win) - 1))
    ev = {'alloc_retries': 0, 'alloc_ooms': 0, 'nccl_errors': 0, 'nan_inf_steps': 0, 'exceptions': []}
    ev.update(events or {})
    train_s = sum(r['max_rank_whole_s'] for r in per_step)
    return {
        'schema': SCHEMA_RECEIPT, 'run_id': run_id, 'attempt_id': attempt_id, 'purpose': purpose, 'task_id': task_raw['id'],
        'task_digest': digest(task_raw), 'config_digest': digest(config), 'code_digest': 'code-synthetic',
        'overlay_digest': overlay_digest, 'input_plan': 'A', 'world_size': world, 'nodes': [f'WINH{k}' for k in range(int(task_raw['hardware']['nodes']))],
        'steps': steps, 'steps_completed': steps if valid or not events else steps, 'window': [a, b],
        'step_time': {'mean_window_max_rank_s': round(mean, 6), 'median_window_max_rank_s': round(sorted(win)[len(win) // 2], 6),
                      'rsd_window': round(sd / mean, 6), 'inter_step_mean_s': round(mean * inter_step_ratio, 6)},
        'per_step': per_step, 'invariants': reference_invariants(task_raw, model_facts),
        'memory': {'per_rank': [{'rank': r, 'peak_alloc_gib': round(peak_reserved_gib * 0.9, 3), 'peak_reserved_gib': round(peak_reserved_gib - 0.01 * r, 3)} for r in range(world)],
                   'max_peak_alloc_gib': round(peak_reserved_gib * 0.9, 3), 'max_peak_reserved_gib': round(peak_reserved_gib, 3)},
        'events': ev,
        'phases': {'startup': 20.0, 'load': startup_s * 0.5, 'build': startup_s * 0.3, 'compile_first_step': startup_s * 0.2,
                   'train': round(train_s, 3), 'save': 0.0, 'export': 0.0, 'total': round(20 + startup_s + train_s, 3)},
        'env': {'torch': '2.10.0+cu128', 'te': '2.16', 'mcore': '0.18.2', 'whitelisted': config.get('env', {})},
        'valid': valid, 'invalid_reasons': list(invalid_reasons), 'signals_digest': digest([attempt_id, 'signals']),
        'cost': {'gpu_seconds': round(world * (20 + startup_s + train_s), 3), 'wall_seconds': round(20 + startup_s + train_s, 3)},
        # Proposed (PROPOSED_CHANGES_A): train span from per-update UTC stamps, a clock independent of perf_counter.
        'wallclock': {'train_utc_s': round(train_s, 3)},
    }


class FakeScreenRunner:
    """Deterministic synthetic performance/memory model for CPU tests of search loops (not a planner)."""

    def __init__(self, task, model_facts: dict, *, clock=None, time_scale: float = 1.0, hbm_gib: float = 80.0,
                 mutate=None, fail=None):
        self.task_raw = task.raw if hasattr(task, 'raw') else task
        self.mf, self.clock, self.hbm = model_facts, clock, hbm_gib
        self.time_scale = time_scale
        self.mutate, self.fail = mutate, fail     # mutate(receipt, config, overlay) -> receipt; fail(config) -> reason|None
        self.calls = []

    def model(self, config: dict) -> tuple[float, float]:
        """(step seconds, peak reserved GiB) under a coarse cost model."""
        c = validate_l0(config).resolved
        raw, mf = self.task_raw, self.mf
        world = int(raw['hardware']['gpus'])
        tp, pp, cp = c['tensor_model_parallel_size'], c['pipeline_model_parallel_size'], c['context_parallel_size']
        ep, mbs = c['expert_model_parallel_size'], c['micro_batch_size']
        etp = c['expert_tensor_parallel_size'] or tp
        dp = max(1, world // (tp * pp * cp))
        params = mf.get('params', 1.7e9)
        frac_e = mf.get('expert_param_fraction', 0.0)
        active = mf.get('active_params', params)
        dense_p, exp_p = params * (1 - frac_e), params * frac_e
        w_per_rank = dense_p / (tp * pp) + exp_p / (ep * etp * pp)
        state_per_rank = 12 * w_per_rank / dp * (1 - (c['optimizer_offload_fraction'] if c['optimizer_cpu_offload'] else 0.0))
        seq, hidden, layers = raw['workload']['seq_len'], mf.get('hidden_size', 2048), mf.get('num_layers', 28)
        act_layer = 34 * seq * hidden * mbs / (tp if c['sequence_parallel'] else 1) / cp
        gran = c['recompute_granularity']
        act_factor = {'full': 2 / 34 * (c['recompute_num_layers'] or 1), 'selective': 0.55}.get(gran, 1.0)
        logits = seq * mbs * mf.get('vocab_size', 151936) * 4 / tp
        mem = (6 * w_per_rank + state_per_rank + act_layer * act_factor * layers / pp + logits) / 2 ** 30 * 1.08 + 1.5
        tokens = raw['workload']['physical_tokens_per_update']
        flops = 6 * active * tokens * (1 + 0.33 * (gran == 'full') + 0.08 * (gran == 'selective'))
        t = flops / (world * 989e12 * (0.42 + 0.03 * min(mbs, 4) / 4))
        t *= 1 + 0.035 * (tp - 1) * (0.5 if c['tp_comm_overlap'] else 1.0) + 0.06 * (cp - 1)
        ga = max(1, raw['workload']['packs_per_update'] // (dp * mbs))
        t *= 1 + (pp - 1) / (ga + pp - 1)
        if dp > 1:
            t *= 1 + (0.02 if c['overlap_grad_reduce'] else 0.07)
        if frac_e:
            t *= 1 + 0.05 * (ep > 1) * (0.5 if c['moe_token_dispatcher_type'] in ('alltoall', 'flex') else 1.0)
            t *= 0.93 if c['moe_grouped_gemm'] else 1.0
        if c['optimizer_cpu_offload']:
            t += 12 * w_per_rank / dp * c['optimizer_offload_fraction'] / 20e9
        for k in ('apply_rope_fusion', 'bias_activation_fusion', 'gradient_accumulation_fusion', 'cross_entropy_loss_fusion'):
            t *= 0.985 if c[k] else 1.0
        return t * self.time_scale, mem

    def run(self, task, config: dict, overlay: dict | None, purpose: str, run_id: str) -> dict:
        n = len(self.calls)
        attempt = 'a' + hashlib.sha256(json.dumps([run_id, n, config], sort_keys=True, default=str).encode()).hexdigest()[:12]
        self.calls.append({'config': config, 'overlay': overlay, 'purpose': purpose, 'run_id': run_id, 'attempt_id': attempt})
        rep = validate_l0(config, self.task_raw, self.mf)
        ov_digest = None if not overlay else digest(sorted(overlay))
        if not rep.ok:
            r = synthetic_receipt(self.task_raw, config, step_time_s=1.0, peak_reserved_gib=0.0, run_id=run_id, attempt_id=attempt,
                                  purpose=purpose, model_facts=self.mf, valid=False, invalid_reasons=['config rejected at build: ' + rep.errors[0]],
                                  startup_s=60.0, overlay_digest=ov_digest)
            r['steps_completed'] = 0
            r['cost'] = {'gpu_seconds': 60.0 * r['world_size'], 'wall_seconds': 60.0}
        else:
            t, mem = self.model(config)
            reason = self.fail(config) if self.fail else None
            if mem > self.hbm - 2 or reason:
                r = synthetic_receipt(self.task_raw, config, step_time_s=t, peak_reserved_gib=min(mem, self.hbm), run_id=run_id,
                                      attempt_id=attempt, purpose=purpose, model_facts=self.mf, valid=False,
                                      invalid_reasons=[reason or 'CUDA out of memory'], events={'alloc_ooms': 0 if reason else 1},
                                      overlay_digest=ov_digest)
                r['steps_completed'] = 1
                r['cost'] = {'gpu_seconds': round(r['world_size'] * 300.0, 3), 'wall_seconds': 300.0}
            else:
                r = synthetic_receipt(self.task_raw, config, step_time_s=t, peak_reserved_gib=mem, run_id=run_id, attempt_id=attempt,
                                      purpose=purpose, model_facts=self.mf, overlay_digest=ov_digest)
        if self.mutate:
            r = self.mutate(r, config, overlay)
        if self.clock is not None:
            self.clock.advance(r['cost']['wall_seconds'])
        return r
