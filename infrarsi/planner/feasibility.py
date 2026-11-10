"""feasibility.estimate(task, config, hw) -> per-GPU and per-node host memory, plus a coarse step-time roofline.

Memory (design 1.5): bf16 params + fp32 grad buffer + fp32 master/m/v (distributed optimizer shards states over
DP for dense and over expert-DP for experts; offload moves a fraction of master/m/v plus an fp32 grad copy to
pinned host memory); activations by layer kind, tokens, MBS and recompute mode; MoE experts by EP rank;
vocab-parallel logits; 8 % of GPU memory kept for communication workspace. GPU budget = min(78 GiB hard cap,
0.92 x device memory). The flagship recipe adds three placements outside L0: PLE tables on the host, expert Adam-v
on the host and lossless 16-bit master encoding (keys starting with '_').
"""
from __future__ import annotations

import dataclasses
import math

from infrarsi.contracts import TaskSpec

GIB = 2 ** 30
COMM_RESERVE = 0.08
FIXED_GPU_GIB = 3.0            # CUDA context, NCCL/TE workspaces, allocator slack
HOST_PER_RANK_GIB = 6.0        # dataloader, CUDA host context, Python
HARD_CAP_GIB = 78.0            # reward.yaml peak_reserved_gib_max (H100 tier)


@dataclasses.dataclass
class Layout:
    tp: int
    pp: int
    cp: int
    ep: int
    etp: int
    dp: int
    edp: int
    mbs: int
    ga: int
    sp: bool
    problems: list


def layout(task: TaskSpec, config: dict, facts: dict, gpus_per_node: int = 8, ib_inter: bool = True) -> Layout:
    c = config
    world = task.gpus
    tp, pp, cp = int(c.get('tensor_model_parallel_size', 1)), int(c.get('pipeline_model_parallel_size', 1)), int(c.get('context_parallel_size', 1))
    moe = facts.get('moe')
    ep = int(c.get('expert_model_parallel_size', 1)) if moe else 1
    etp = int(c.get('expert_tensor_parallel_size', tp if moe else 1)) if moe else 1
    mbs = int(c.get('micro_batch_size', 1))
    sp = bool(c.get('sequence_parallel', tp > 1))
    probs = []
    if world % (tp * pp * cp):
        probs.append(f'world {world} not divisible by TP*PP*CP {tp * pp * cp}')
    dp = max(1, world // (tp * pp * cp))
    edp = max(1, world // (ep * etp * pp)) if moe else dp
    if moe and world % (ep * etp * pp):
        probs.append(f'world {world} not divisible by EP*ETP*PP {ep * etp * pp}')
    if moe and moe['num_experts'] % ep:
        probs.append(f'{moe["num_experts"]} experts not divisible by EP {ep}')
    packs = task.raw['workload']['packs_per_update']
    if packs % (dp * mbs):
        probs.append(f'{packs} packs/update not divisible by DP*MBS {dp * mbs}')
    ga = max(1, packs // (dp * mbs))
    if tp > gpus_per_node:
        probs.append(f'TP {tp} spans nodes (no NVLink across nodes)')
    if facts['heads'] % tp or (facts['kv_heads'] % tp and not c.get('_kv_heads_replicated')):
        probs.append(f'TP {tp} does not divide heads {facts["heads"]} / KV groups {facts["kv_heads"]}')
    lh = facts.get('linear_heads')
    if lh and (lh['key'] % tp or lh['value'] % tp):
        probs.append(f'TP {tp} does not divide linear-attention heads {lh}')
    if moe and tp > 1 and not sp:
        probs.append('MoE with TP > 1 needs sequence_parallel')
    if moe and ep * etp > gpus_per_node and not ib_inter:
        probs.append('EP spans nodes without IB')
    if task.seq_len % (2 * cp) and cp > 1:
        probs.append(f'seq {task.seq_len} not divisible by 2*CP')
    data = task.raw.get('data') or {}
    if (data.get('kind') == 'sft_packed' or str(data.get('key', '')).startswith('D-SFT')) and cp not in (1, 2, 4):
        probs.append(f'CP {cp} invalid for SFT data (segments padded to multiples of 8: CP in 1, 2, 4)')
    if pp > 1:
        probs.append('PP > 1 is outside the v0 planner')
    return Layout(tp, pp, cp, ep, etp, dp, edp, mbs, ga, sp, probs)


def _attn_act_bytes(facts: dict, kind: str, tokens: float, L: Layout) -> float:
    """bf16 activations one attention block keeps for backward without recompute (bytes per GPU)."""
    H, tp = facts['hidden'], L.tp
    sp = tp if (L.sp or tp == 1) else 1
    nq, nkv, hd = facts['heads'], facts['kv_heads'], facts['head_dim']
    if kind == 'linear' and facts.get('linear_heads'):
        lh = facts['linear_heads']
        kd, vd = lh['key'] * lh['key_dim'], lh['value'] * lh['value_dim']
        return 2 * tokens * (2 * H / sp + (4 * kd + 4 * vd) / tp)
    q = nq * hd * (2 if facts['model_type'] in ('qwen3_next', 'qwen4_exp', 'qwen4_exp_text') else 1)
    return 2 * tokens * (2 * H / sp + (q + 2 * nkv * hd + nq * hd) / tp)


def _mlp_act_bytes(facts: dict, moe_layer: bool, tokens: float, L: Layout) -> float:
    H, tp = facts['hidden'], L.tp
    sp = tp if (L.sp or tp == 1) else 1
    if moe_layer and facts.get('moe'):
        m = facts['moe']
        assign = tokens * m['topk'] / sp
        return (2 * assign * (2 * H + 3 * m['moe_intermediate']) + 8 * tokens / sp * m['num_experts']
                + 6 * tokens * m.get('shared_intermediate', 0) / tp + 2 * tokens * H / sp)
    return 2 * tokens * (2 * H / sp + 3 * facts['intermediate'] / tp)


def activation_bytes(facts: dict, config: dict, L: Layout, seq: int) -> dict:
    tokens = seq / L.cp * L.mbs
    sp = L.tp if (L.sp or L.tp == 1) else 1
    n = facts['layers']
    kinds = facts['layer_kinds']
    n_moe = (facts.get('moe') or {}).get('moe_layers', 0)
    attn = {k: _attn_act_bytes(facts, k, tokens, L) for k in ('full', 'linear')}
    mlp = {m: _mlp_act_bytes(facts, m, tokens, L) for m in (False, True)}
    total_none = kinds['full'] * attn['full'] + kinds['linear'] * attn['linear'] + n_moe * mlp[True] + (n - n_moe) * mlp[False]
    worst = max(attn[k] for k in ('full', 'linear') if kinds[k]) + max(mlp[m] for m in (False, True) if (n_moe if m else n - n_moe))
    gran = config.get('recompute_granularity')
    if gran == 'full':
        k = int(config.get('recompute_num_layers', 1))
        stored = (n / k) * tokens * facts['hidden'] * 2 / sp
        working = 2 * worst * k
    elif gran == 'selective':
        stored, working = 0.8 * total_none, worst
    else:
        stored, working = total_none, worst
    fused = bool(config.get('cross_entropy_loss_fusion'))
    logits = tokens * facts['vocab'] / L.tp * (4 if fused else 10)
    if facts.get('extras', {}).get('mtp'):
        logits *= 1.5
    return {'stored': stored, 'working': working, 'logits': logits}


def estimate(task: TaskSpec, config: dict, hw: dict | None = None, facts: dict | None = None) -> dict:
    from infrarsi.discover import default_profile, workload_facts
    hw = (hw or default_profile(task.nodes))['summary']
    facts = facts or workload_facts(task)
    gpn = min(hw['gpus_per_node'], task.gpus)
    L = layout(task, config, facts, gpn, hw.get('ib_inter', True))
    c = config
    cpu_params = facts.get('cpu_resident_params', 0) if c.get('_ple_cpu_offload') else 0
    p_exp = facts['params_expert']
    p_dense = facts['params_total'] - p_exp
    if not c.get('_ple_cpu_offload') and facts.get('cpu_resident_params'):
        p_dense += facts['cpu_resident_params']           # PLE tables on GPU unless the recipe offloads them
    dense_local = p_dense / (L.tp * L.pp)
    exp_local = p_exp / (L.ep * L.etp * L.pp) if facts.get('moe') else 0.0
    master_b = float(c.get('_master_param_bytes', 4))
    off = float(c.get('optimizer_offload_fraction', 1.0)) if c.get('optimizer_cpu_offload') else 0.0
    exp_v_cpu = bool(c.get('_expert_exp_avg_sq_cpu'))
    dist_opt = c.get('use_distributed_optimizer', True)
    sd = L.dp if dist_opt else 1
    se = L.edp if dist_opt else 1
    states_dense = dense_local / sd * (master_b + 8)
    states_exp = exp_local / se * (master_b + (4 if exp_v_cpu else 8))
    b = {'params': 2 * (dense_local + exp_local), 'grads': 4 * (dense_local + exp_local),
         'optimizer_states': (1 - off) * (states_dense + states_exp)}
    act = activation_bytes(facts, c, L, task.seq_len)
    b.update(act_stored=act['stored'], act_working=act['working'], logits=act['logits'], fixed=FIXED_GPU_GIB * GIB)
    gpu = sum(b.values())
    budget = min(HARD_CAP_GIB, (1 - COMM_RESERVE) * hw['gpu_mem_gib'])
    host_rank = off * (states_dense + states_exp + 4 * (dense_local / sd + exp_local / se))
    if exp_v_cpu:
        host_rank += 4 * exp_local / se
    host_node = (host_rank / GIB + HOST_PER_RANK_GIB) * gpn
    if cpu_params:
        host_node += 2 * cpu_params / GIB      # assumption: one bf16 PLE table per node (flagship PLE_CPU_OFFLOAD)
    est = {'config': dict(c), 'layout': dataclasses.asdict(L), 'valid_layout': not L.problems,
           'gpu_gib': gpu / GIB, 'gpu_budget_gib': budget, 'gpu_fits': gpu / GIB <= budget,
           'host_gib_per_node': host_node, 'host_budget_gib': hw['pinned_budget_gib'],
           'host_fits': host_node <= hw['pinned_budget_gib'], 'breakdown_gib': {k: v / GIB for k, v in b.items()}}
    est['fits'] = est['valid_layout'] and est['gpu_fits'] and est['host_fits']
    est['step_time_s'] = roofline(task, c, facts, L, hw, off) if est['valid_layout'] else None
    est['notes'] = list(L.problems)
    if not est['gpu_fits']:
        est['notes'].append(f'GPU estimate {gpu / GIB:.1f} GiB > budget {budget:.1f} GiB')
    if not est['host_fits']:
        est['notes'].append(f'host estimate {host_node:.0f} GiB/node > pinned budget {hw["pinned_budget_gib"]:.0f} GiB')
    return est


def roofline(task: TaskSpec, c: dict, facts: dict, L: Layout, hw: dict, off: float) -> dict:
    """Coarse per-update time (seconds) used only to rank candidates; probes decide."""
    peak = (hw.get('peak_bf16_tflops') or 989.0) * 1e12
    world, tokens, H = task.gpus, task.raw['workload']['physical_tokens_per_update'], facts['hidden']
    moe = facts.get('moe')
    embed = facts['breakdown']['embedding'] + facts['breakdown']['output']
    flops = 6 * (facts['params_active'] - embed) * tokens + 6 * facts['vocab'] * H * tokens
    seg = task.seq_len if task.raw['data'].get('kind') == 'pt_stream' else min(task.seq_len, 4096) / 2
    flops += 12 * facts['layer_kinds']['full'] * facts['heads'] * facts['head_dim'] * tokens * seg / 2
    rec = {'full': 4 / 3, 'selective': 1.05}.get(c.get('recompute_granularity'), 1.0)
    eff = 0.30 if moe else 0.45
    if facts['layer_kinds']['linear']:
        eff *= 0.85
    eff *= 1.05 if L.mbs > 1 else 1.0
    eff *= 1 - 0.03 * math.log2(L.tp)
    if c.get('bias_activation_fusion') and c.get('apply_rope_fusion'):
        eff *= 1.05
    t_comp = flops * rec / (world * peak * eff)
    tok_local = task.seq_len / L.cp * L.mbs
    gpn = min(hw['gpus_per_node'], world)
    ops = 8 + (4 if c.get('recompute_granularity') == 'full' else 0)
    t_tp = facts['layers'] * L.ga * ops * tok_local * H * 2 * (L.tp - 1) / L.tp / 180e9 if L.tp > 1 else 0.0
    t_ep = 0.0
    if moe and L.ep > 1:
        assign = tok_local * moe['topk'] / (L.tp if L.sp else 1)
        bw = 120e9 if L.ep * L.etp <= gpn else 40e9
        n_ops = 4 + (2 if c.get('recompute_granularity') == 'full' else 0)
        t_ep = moe['moe_layers'] * L.ga * n_ops * assign * H * 2 * (L.ep - 1) / L.ep / bw
    dense_local = (facts['params_total'] - facts['params_expert']) / L.tp
    exp_local = facts['params_expert'] / (L.ep * L.etp) if moe else 0.0
    bw_dp = 180e9 if world <= gpn else 45e9
    t_dp = (6 * dense_local * (L.dp - 1) / L.dp + 6 * exp_local * (L.edp - 1) / L.edp) / bw_dp if world > 1 else 0.0
    if c.get('overlap_grad_reduce') and c.get('overlap_param_gather'):
        t_dp *= 0.25
    elif c.get('overlap_grad_reduce'):
        t_dp *= 0.6
    t_off = 0.0
    if off:
        local = off * (dense_local / L.dp + exp_local / L.edp)
        pcie = 6 * off * (dense_local + exp_local) / 20e9
        t_off = 16 * local / 12e9 + pcie * (0.5 if c.get('overlap_cpu_optimizer_d2h_h2d') else 1.0)
    t_over = 0.01 * L.ga
    total = t_comp + t_tp + t_ep + t_dp + t_off + t_over
    return {'total': total, 'compute': t_comp, 'tp_comm': t_tp, 'ep_comm': t_ep, 'dp_comm': t_dp, 'offload': t_off,
            'overhead': t_over, 'flops': flops}
