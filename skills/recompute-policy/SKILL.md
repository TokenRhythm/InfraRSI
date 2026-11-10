---
name: recompute-policy
description: Spend recomputation only where memory requires it - full vs selective vs none, which layers and which submodules, and the single-update lifetime rule for any replay cache.
license: Apache-2.0
when_to_use: FR or ED uses full recompute and headroom exists; OOM after removing recompute; choosing recompute_modules; MoE layers dominate activations; 'activation checkpointing', 'selective recompute', 'recompute_num_layers', 'replay cache'.
metadata:
  author: InfraRSI A0 (human-authored)
  source: human-authored (A0)
  version: 1
  tags: [recompute, memory, activations, moe, dense, long-context]
---

# Recompute policy

## Trigger conditions
- recompute_granularity is full and the tightest rank has > ~10 GiB headroom.
- A candidate without recompute OOMs, and you need the cheapest partial recompute that fits.
- MoE models: expert activations and dispatch buffers dominate.

## Required observations (receipt fields)
- `memory.max_peak_reserved_gib` and per-rank spread; `profile.fwd_bwd_s` (recompute cost lands there); `events.alloc_retries`.
- Model facts: num_layers, MoE or dense, shared experts.

## Rules: MCore 0.18.2 options
- full: recompute_method uniform (recompute_num_layers = layers per checkpoint unit) or block (recompute_num_layers = how many layers per pipeline stage are recomputed; the rest keep activations). block with k < layers/PP is the main knob between "all" and "none".
- selective: recompute_modules from core_attn, layernorm, mlp, moe, moe_act, shared_experts (mla_up_proj only for latent attention). moe_act and layernorm use output-discarding checkpointing and are cheap; moe_act requires moe_grouped_gemm. With TE fused/flash attention, core_attn recompute rarely saves much (MCore warns about it).
- moe_layer_recompute is a deprecated alias for selective + moe and is rejected together with full recompute.
- cpu_offloading (activation offload) does not combine with any recompute.
- overlap_moe_expert_parallel_comm forbids full recompute, recompute_method/num_layers and 'moe' in recompute_modules.

## Rules: cost model
- Full recompute repeats each layer's forward: about one third more transformer FLOPs, and it repeats forward collectives (TP all-gathers, MoE dispatch/combine). On the flagship, recompute meant 98 + 98 non-cached dispatches per update.
- Each layer you stop recomputing adds roughly its full activation footprint (order 34*s*b*h bytes, divided by TP with SP and by CP). Remove recompute from as many layers as headroom allows, keeping >= 4 GiB margin.
- Different parts of a layer have different cost/memory ratios: attention prefixes are cheap to keep relative to their recompute cost in some models, MoE suffixes the opposite. Measure; do not assume.

## Rules: replay caches (L1)
- A cache that lets recompute reuse forward results must be keyed by (update, micro-batch, layer, generation), consumed exactly once, cleared after each micro-batch, and checked on first use against the native path. Anything that survives across updates is the cross-update cache cheat and is invalid.
- In this benchmark (v0) the static guard rule G19 requires every replaced forward/step to call the captured original exactly once and return its result, so a replay cache that skips a forward is rejected before any screen. Do not propose one; reach the same memory/time trade with recompute_method block, recompute_num_layers or recompute_modules.

## Allowed actions
- L0: recompute_granularity, recompute_method, recompute_num_layers, recompute_modules, moe_layer_recompute, cpu_offloading(+num_layers), micro_batch_size.
- L1: recompute_policy.py (template recompute_policy.first_k_layers_full: full recompute only for the first k layers per stage).
- Typical sequence from FR (full/uniform/1): try block with a few layers recomputed, or selective with cheap modules; then none if headroom remains.

## Failure signals
- OOM at update 2 after removing recompute (moments now resident).
- Selective recompute with higher peak and lower throughput than full (core_attn recompute under fused attention).
- Recompute removed but StepTime unchanged: the update is bound elsewhere (communication or optimizer).

## Verification
- StepTime beyond the margin with 0 OOMs, peak within cap; for replay caches, a first-use digest gate and a lifetime assertion (cache empty after each micro-batch).

## Success case (real)
- Flagship: keeping the attention prefix of backbone layers 3, 7, ..., 43 and checkpointing only the MoE suffix (layer 47 and MTP unchanged): 10.373877 -> 9.766195 s (+6.22%), exact. A single-update replay cache for the MoE combine in recompute: 8.831999 -> 8.504627 s (+3.85%), exact. (MANUSCRIPT.md 4.3, Table 6)

## Failure case (real)
- NeMo, 64 GPUs, 2K tokens: selective recompute v2 gave 30,143.33 tok/s versus full recompute 30,528.63 tok/s with a higher peak (30.29 vs 22.44 GiB). Micro-batch 2 recomputing only 24 layers OOMed at step 2; with 31 layers it ran at 47,210.24 tok/s and 61.82 GiB, still below micro-batch 1 without recompute (49,237.07 tok/s). Selective is not automatically cheaper than full. (reports/winh8_20260925/REPORT.md)
