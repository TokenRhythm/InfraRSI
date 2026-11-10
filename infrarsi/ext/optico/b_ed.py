"""B-ED: InfraRSI's expert-default planner (planner/expert_default + planner/feasibility, design §1.5) projected onto
OptiCo's parameter set (design §1.1 / §3.4): one non-LLM trial per workload.

Projection (every rule is the planner's own; only the mapping is ours):
  - facts: the script's model shape through discover.workload.count_params / tp_candidates (HF-style config built
    from run_<model>.sh); hardware = the default WINH profile (8 x H100 80GB, NVLink, IB);
  - candidate space = expert_default.candidate_space's rules restricted to OptiCo's knobs: TP from tp_candidates up to
    the node (no NVLink -> TP <= 2), PP 1 (the v0 planner's layouts; PP > 1 is outside it), CP 1 and EP 1 (dense, not
    OptiCo knobs), MBS in {1, 2}, recompute full (1 layer per group) / selective / none -> AC full,AC_LAYERS 1 / sel /
    none, overlap grad-reduce and param-gather iff DP > 1, fused cross-entropy on; optimizer CPU offload is not an
    OptiCo knob (off only); first/last PP split none;
  - feasibility.estimate (memory fit within min(78 GiB, 0.92 x HBM) and the coarse roofline) ranks the fitting
    candidates exactly as expert_default.rank does (estimated step time, then GPU GiB, then digest);
  - B-ED = the top-ranked candidate that passes validate_optico_params (one trial; ED's top-3 probing is not part of
    the ENV-B protocol).
"""
from __future__ import annotations

import itertools
import types

from ...contracts import digest
from ...discover.hardware import default_profile
from ...discover.workload import count_params, tp_candidates
from ...planner.feasibility import estimate
from . import params as pm
from . import workloads as wl

VERSION = 'b_ed_v1 (ed_v0 rules projected onto OptiCo knobs)'
ED_REC_TO_AC = {'full': 'full', 'selective': 'sel', None: 'none'}


def planner_task(workload) -> types.SimpleNamespace:
    """The TaskSpec surface feasibility.estimate reads, for one OptiCo workload (fixed GBS/seq of the scripts)."""
    w = wl.get(workload)
    raw = {'id': w.task_id, 'hardware': {'gpus': wl.GPUS, 'nodes': 1, 'gpu_type': 'H100-80GB'},
           'workload': {'seq_len': wl.SEQ, 'packs_per_update': wl.GBS, 'physical_tokens_per_update': wl.GBS * wl.SEQ},
           'data': {'kind': 'pt_stream', 'key': 'optico-mmap'}}
    return types.SimpleNamespace(id=w.task_id, raw=raw, gpus=wl.GPUS, nodes=1, seq_len=wl.SEQ)


def planner_facts(workload) -> dict:
    w = wl.get(workload)
    f = count_params(w.hf_like_config())
    f['tp_candidates'] = tp_candidates(f, wl.GPUS)
    return f


def candidate_space(workload, hw: dict | None = None) -> list[tuple[dict, dict]]:
    """[(planner L0-style config, OptiCo knobs)] under the ED rules (see module docstring)."""
    w = wl.get(workload)
    s = (hw or default_profile(1))['summary']
    facts = planner_facts(w)
    gpn = min(s['gpus_per_node'], wl.GPUS)
    tps = [t for t in facts['tp_candidates'] if t <= gpn]
    if not s.get('nvlink_intra', True):
        tps = [t for t in tps if t <= 2]                         # ED rule: no NVLink -> small TP
    recs = ('full', 'selective', None)
    if (s.get('peak_bf16_tflops') or 989) < 300 and (s.get('gpu_mem_gib') or 80) > 90:
        recs = ('selective', None, 'full')                       # ED rule: low compute, high memory -> less recompute
    out = []
    for tp, mbs, rec in itertools.product(tps, (1, 2), recs):
        dp = wl.GPUS // tp
        if wl.GPUS % tp or wl.GBS % (dp * mbs):
            continue
        cfg = {'tensor_model_parallel_size': tp, 'pipeline_model_parallel_size': 1, 'context_parallel_size': 1,
               'sequence_parallel': tp > 1, 'micro_batch_size': mbs, 'use_distributed_optimizer': True,
               'overlap_grad_reduce': dp > 1, 'overlap_param_gather': dp > 1, 'cross_entropy_loss_fusion': True,
               'optimizer_cpu_offload': False}
        if rec == 'full':
            cfg.update(recompute_granularity='full', recompute_method='uniform', recompute_num_layers=1)
        elif rec == 'selective':
            cfg.update(recompute_granularity='selective')
        knobs = w.knobs(MICRO_BATCH_SIZE=mbs, TP=tp, PP=1, OVERLAP_GRAD_REDUCE=dp > 1, OVERLAP_PARAM_GATHER=dp > 1,
                        AC=ED_REC_TO_AC[rec], AC_LAYERS=1, FUSED_CROSS_ENTROPY=True, FIRST_PP_NUM_LAYERS=None,
                        LAST_PP_NUM_LAYERS=None)
        out.append((cfg, knobs))
    return out


def rank(workload, hw: dict | None = None) -> list[dict]:
    w = wl.get(workload)
    hw = hw or default_profile(1)
    task, facts = planner_task(w), planner_facts(w)
    ranked = []
    for cfg, knobs in candidate_space(w, hw):
        e = estimate(task, cfg, hw, facts)
        if not e['fits']:
            continue
        norm, errs, _ = pm.validate_optico_params(knobs, w)
        t = e['step_time_s'] or {}
        ranked.append({'params': norm, 'valid': not errs, 'errors': errs, 'est_step_s': t.get('total'),
                       'gpu_gib': round(e['gpu_gib'], 2), 'gpu_budget_gib': round(e['gpu_budget_gib'], 2),
                       'est': {k: round(v, 4) for k, v in t.items() if isinstance(v, float) and k != 'flops'},
                       'explanation': f'{pm.format_params(norm)}: GPU {e["gpu_gib"]:.1f}/{e["gpu_budget_gib"]:.1f} GiB, '
                                      f'est {t.get("total", float("nan")):.2f} s/update'})
    ranked.sort(key=lambda r: (r['est_step_s'], r['gpu_gib'], digest(r['params'])))
    return ranked


def pick(workload, hw: dict | None = None) -> dict:
    """{'params', 'explanation', 'ranked', 'version'}; params None when nothing fits (then B-ED is not run)."""
    w = wl.get(workload)
    ranked = rank(w, hw)
    pick = next((r for r in ranked if r['valid']), None)
    return {'version': VERSION, 'workload': w.key, 'task_id': w.task_id, 'params': pick['params'] if pick else None,
            'explanation': (f'rank {ranked.index(pick) + 1} of {len(ranked)} fitting: {pick["explanation"]}' if pick
                            else 'no candidate fits the memory model'),
            'is_default': bool(pick) and pm.is_default(pick['params'], w),
            'ranked': [{k: r[k] for k in ('params', 'valid', 'est_step_s', 'gpu_gib', 'explanation')} for r in ranked[:8]]}


b_ed = pick
