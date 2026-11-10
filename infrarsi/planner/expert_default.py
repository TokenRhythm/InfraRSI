"""expert_default (ED, design 1.5): memory feasibility + coarse roofline + hardware rules -> ranked L0 configs,
top-3 diverse candidates probed for 5 updates each, fastest valid probe wins. Non-LLM; every choice is explained.

Rules: NVLink inside a node lets TP go up to the node size; without NVLink prefer DP/EP, small TP, more recompute;
low-compute high-memory GPUs: less recompute, larger MBS; without IB keep EP inside a node.
"""
from __future__ import annotations

import copy
import itertools

from infrarsi.contracts import TaskSpec, digest

from .feasibility import estimate

VERSION = 'ed_v0'
ED_ENV = {'CUDA_DEVICE_MAX_CONNECTIONS': '1', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
MAJOR = ('tensor_model_parallel_size', 'context_parallel_size', 'expert_model_parallel_size', 'optimizer_cpu_offload')


def candidate_space(task: TaskSpec, facts: dict, hw: dict) -> list[dict]:
    s = hw['summary']
    gpn = min(s['gpus_per_node'], task.gpus)
    world, seq = task.gpus, task.seq_len
    tps = [t for t in facts['tp_candidates'] if t <= gpn]
    if not s.get('nvlink_intra', True):
        tps = [t for t in tps if t <= 2]                         # rule: no NVLink -> small TP
    cps = [c for c in (1, 2, 4) if c == 1 or (seq >= 16384 and seq % (2 * c) == 0)]
    moe = facts.get('moe')
    eps = [1]
    if moe:
        lim = world if s.get('ib_inter', True) else gpn          # rule: no IB -> EP inside a node
        eps = [e for e in (1, 2, 4, 8, 16, 32, 64) if e <= lim and moe['num_experts'] % e == 0 and world % e == 0]
    mbss = (1, 2)
    recs = ('full', 'selective', None)
    if (s.get('peak_bf16_tflops') or 989) < 300 and (s.get('gpu_mem_gib') or 80) > 90:
        recs = ('selective', None, 'full')                       # rule: low compute, high memory -> less recompute
    out = []
    for tp, cp, ep, mbs, rec, off in itertools.product(tps, cps, eps, mbss, recs, (False, True)):
        if world % (tp * cp):
            continue
        dp = world // (tp * cp)
        if task.raw['workload']['packs_per_update'] % (dp * mbs):
            continue
        cfg = {'tensor_model_parallel_size': tp, 'pipeline_model_parallel_size': 1, 'context_parallel_size': cp,
               'sequence_parallel': tp > 1, 'micro_batch_size': mbs, 'use_distributed_optimizer': True,
               'overlap_grad_reduce': dp > 1, 'overlap_param_gather': dp > 1, 'apply_rope_fusion': True,
               'bias_activation_fusion': True, 'masked_softmax_fusion': True, 'gradient_accumulation_fusion': True,
               'cross_entropy_loss_fusion': True, 'cross_entropy_fusion_impl': 'native', 'attention_backend': 'auto',
               'optimizer_cpu_offload': off, 'env': dict(ED_ENV)}
        if rec == 'full':
            cfg.update(recompute_granularity='full', recompute_method='uniform', recompute_num_layers=1)
        elif rec == 'selective':
            cfg.update(recompute_granularity='selective')
        if off:
            cfg.update(optimizer_offload_fraction=1.0, overlap_cpu_optimizer_d2h_h2d=True)
        if moe:
            cfg.update(expert_model_parallel_size=ep, expert_tensor_parallel_size=1, moe_token_dispatcher_type='alltoall',
                       moe_grouped_gemm=True, moe_permute_fusion=True, moe_enable_deepep=False)
        out.append(cfg)
    return out


def _explain(e: dict) -> str:
    c, L, t = e['config'], e['layout'], e['step_time_s'] or {}
    rec = c.get('recompute_granularity') or 'none'
    knobs = (f"TP{L['tp']} CP{L['cp']} EP{L['ep']} DP{L['dp']} MBS{L['mbs']} GA{L['ga']} recompute={rec} "
             f"offload={bool(c.get('optimizer_cpu_offload'))}")
    mem = f"GPU {e['gpu_gib']:.1f}/{e['gpu_budget_gib']:.1f} GiB, host {e['host_gib_per_node']:.0f}/{e['host_budget_gib']:.0f} GiB/node"
    if not t:
        return f'{knobs}: {mem}'
    parts = ', '.join(f'{k} {t[k]:.2f}s' for k in ('compute', 'tp_comm', 'ep_comm', 'dp_comm', 'offload') if t.get(k))
    return f"{knobs}: {mem}; est {t['total']:.2f}s/update ({parts})"


def rank(task: TaskSpec, facts: dict, hw: dict) -> list[dict]:
    ranked = []
    for cfg in candidate_space(task, facts, hw):
        e = estimate(task, cfg, hw, facts)
        if not e['fits']:
            continue
        ranked.append({'config': cfg, 'estimate': e, 'est_step_s': e['step_time_s']['total'], 'explanation': _explain(e)})
    ranked.sort(key=lambda r: (r['est_step_s'], r['estimate']['gpu_gib'], digest(r['config'])))
    return ranked


def pick_diverse(ranked: list[dict], k: int = 3) -> list[dict]:
    chosen = []
    for r in ranked:
        key = tuple(r['config'].get(m) for m in MAJOR)
        if all(key != tuple(c['config'].get(m) for m in MAJOR) for c in chosen):
            chosen.append(r)
        if len(chosen) == k:
            return chosen
    for r in ranked:                                   # not enough distinct layouts: fill in rank order
        if len(chosen) == k:
            break
        if r not in chosen:
            chosen.append(r)
    return chosen


class EDChain:
    """Probe the top-3 (then further ranked candidates up to max_probes if none is valid); fastest valid wins."""

    def __init__(self, task: TaskSpec, facts: dict, hw: dict, k: int = 3, max_probes: int = 6):
        self.task, self.facts, self.hw, self.k, self.max_probes = task, facts, hw, k, max_probes
        self.ranked = rank(task, facts, hw)
        top = pick_diverse(self.ranked, k)
        self.queue = top + [r for r in self.ranked if r not in top]
        self.history: list[dict] = []
        self.status = 'probing' if self.queue else 'infeasible'
        self.config = None

    def current(self) -> dict | None:
        if self.status != 'probing':
            return None
        n = len(self.history)
        return copy.deepcopy(self.queue[n]['config']) if n < len(self.queue) else None

    def advance(self, outcome: str, receipt: dict | None = None) -> None:
        from infrarsi.harness.probes import probe_step_time
        n = len(self.history)
        cand = self.queue[n]
        self.history.append({'rank': self.ranked.index(cand) + 1, 'config_digest': digest(cand['config']), 'outcome': outcome,
                             'probe_step_s': probe_step_time(receipt) if (receipt and outcome == 'ok') else None,
                             'est_step_s': cand['est_step_s'], 'attempt_id': (receipt or {}).get('attempt_id'),
                             'explanation': cand['explanation']})
        valid = [h for h in self.history if h['outcome'] == 'ok' and h['probe_step_s'] is not None]
        done_top = len(self.history) >= min(self.k, len(self.queue))
        if (done_top and valid) or len(self.history) >= min(self.max_probes, len(self.queue)):
            if valid:
                best = min(valid, key=lambda h: h['probe_step_s'])
                self.config = copy.deepcopy(next(r['config'] for r in self.queue if digest(r['config']) == best['config_digest']))
                self.status = 'feasible'
            else:
                self.status = 'infeasible'

    def result(self) -> dict:
        chosen = next((h for h in self.history if self.config and h['config_digest'] == digest(self.config)), None)
        return {'version': VERSION, 'task': self.task.id, 'status': self.status, 'config': self.config,
                'config_digest': digest(self.config) if self.config else None, 'history': self.history,
                'top': [{'config': r['config'], 'est_step_s': r['est_step_s'], 'explanation': r['explanation']}
                        for r in pick_diverse(self.ranked, self.k)],
                'n_candidates_fitting': len(self.ranked),
                'explanation': (f"chose rank {chosen['rank']} (fastest valid probe {chosen['probe_step_s']:.3f}s): "
                                f"{chosen['explanation']}") if chosen else 'no valid probe'}


def expert_default(task: TaskSpec | str, hw: dict | None = None, probe=None, facts: dict | None = None, k: int = 3) -> dict:
    """Without probe: the ranked plan and top-k. With probe(config) -> (outcome, receipt): run the probes, pick ED."""
    from infrarsi.discover import default_profile, workload_facts
    task = TaskSpec.load(task) if isinstance(task, str) else task
    hw = hw or default_profile(task.nodes)
    chain = EDChain(task, facts or workload_facts(task), hw, k=k)
    if probe is not None:
        while (cfg := chain.current()) is not None:
            outcome, receipt = probe(cfg)
            chain.advance(outcome, receipt)
    return chain.result()
