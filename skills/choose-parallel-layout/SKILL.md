---
name: choose-parallel-layout
description: Choose TP/PP/CP/EP/ETP/DP, micro-batch and gradient accumulation for a fixed global batch from a per-rank memory budget and the cost of each collective; know which process groups are legal.
license: Apache-2.0
when_to_use: First proposals of a task; OOM or a tight rank (<4 GiB headroom); ample headroom with recompute still on; MoE expert placement; long sequences; multi-node scaling; 'which TP', 'EP or DP', 'PP bubble', 'micro batch', 'out of memory'.
metadata:
  author: InfraRSI A0 (human-authored)
  source: human-authored (A0)
  version: 1
  tags: [parallel, layout, memory, tp, pp, cp, ep, dp, micro-batch, dense, moe, long-context, multi-node, single-node]
---

# Choose a parallel layout

GBS (packs per update), sequence length and the data order are task constants. You choose how the same update is
split: `DP = world / (TP*PP*CP)`, `GA = packs_per_update / (DP*MBS)` (must be an integer), experts use
`EDP = world / (ETP*EP*PP)`.

## Trigger conditions
- Start config (ED) receipt shows OOM, allocator retries, or < 4 GiB headroom on the tightest rank.
- Headroom > ~20 GiB while full recompute or offload is still on (memory is being bought with time).
- MoE model, or sequence length >= 32K, or more than one node.

## Required observations (receipt fields)
- `memory.max_peak_reserved_gib`, `memory.per_rank[]` (tightest rank), `events.alloc_retries`, `events.alloc_ooms`.
- `step_time.mean_window_max_rank_s`, `profile` phases (fwd_bwd_s vs optim_s), `derived` dp/ga from propose_config feedback.

## Rules: memory budget per rank (bf16 params, fp32 grads, distributed optimizer)
- Weights 2 B/param + fp32 main grads 4 B/param on every rank holding the shard: dense parameters divided by TP*PP, expert parameters by EP*ETP*PP.
- Optimizer state (fp32 master + exp_avg + exp_avg_sq) 12 B/param, sharded over DP for dense and over EDP for experts. Large EP shrinks EDP and raises per-rank expert optimizer state and work.
- Activations without recompute are of order 34*s*b*h bytes per layer for a standard block (Korthikanti et al. 2022), more for wide SwiGLU MLPs; divided by TP when sequence parallel is on, and by CP. Full recompute keeps roughly the layer input (2*s*b*h) per layer plus one layer's activations during backward.
- Logits: s*b*V*4 bytes in fp32 for the loss, divided by TP (vocab-parallel). A 152K vocabulary at 32K tokens is about 19.9 GB per sequence before TP.
- Keep >= 4 GiB free on the tightest rank and ~8% for NCCL/cuBLAS workspaces; component peaks are not additive, so trust receipts over the ledger. An ideal shard ledger alone could not decide capacity in this project.

## Rules: cost of each dimension
- TP: two all-gathers and two reduce-scatters per layer per pass with SP; only inside an NVLink node (TP <= 8). Use the smallest TP that fits; turn on sequence_parallel whenever TP > 1 (MCore warns when EP and TP > 1 run without SP).
- PP: 1F1B without VPP idles (PP-1)/(GA+PP-1) of the update; with GA of 2-8 it is rarely worth it unless nothing else fits. PP needs num_layers % PP == 0 here.
- CP: splits the sequence; worth it when activations dominate at >= 32K. On SFT tasks (data D-SFT-*) every packed segment is padded to a multiple of 8 tokens and THD context parallelism needs segment lengths divisible by 2*CP, so CP must be 1, 2 or 4; CP=8 is rejected by the L0 validator before any screen. PT tasks (data D-PT-*) allow any CP the layout admits (seq_len % (2*CP) == 0).
- DP with the distributed optimizer: cheapest scaling; overlap_grad_reduce hides the reduce-scatter (see moe-ep-communication and profile-whole-update for verification).
- EP: keep the all-to-all on NVLink (EP <= 8 on one node) unless per-rank expert memory forces more; num_experts % EP == 0. ETP > 1 splits each expert GEMM and is usually slower than larger EP for small experts.
- MBS > 1 raises GEMM efficiency but multiplies activations; GA must stay an integer.

## Rules: legal process groups
- TP*CP*DP*PP = world for attention/dense, ETP*EP*EDP*PP = world for experts. These are two factorizations of the same ranks; EP is not multiplied with DP. world % (TP*PP*CP) == 0 and world % (ETP*EP*PP) == 0.
- Query groups (KV heads) and heads must divide TP, or TP must be a multiple of the query groups when groups < TP (KV heads are replicated).

## Allowed actions
- L0: tensor_model_parallel_size, sequence_parallel, pipeline_model_parallel_size, context_parallel_size (+ cp_comm_type), expert_model_parallel_size, expert_tensor_parallel_size, micro_batch_size; memory levers from manage-memory-lifetimes and recompute-policy.
- Order: make it fit with the least time cost (layout and selective savings before full recompute and offload), then remove savings while headroom allows.

## Failure signals
- OOM at update 2 although update 1 passed (optimizer state is allocated lazily after the first step).
- Allocator retries with tiny free memory: the layout is at the edge even if it completes.
- TP increase that cuts memory but raises StepTime more than the recompute it allowed you to drop.

## Verification
- Fit: a full screen with 0 OOMs, 0 retries preferred, >= 4 GiB headroom on the tightest rank.
- Speed: whole-update StepTime vs parent beyond the margin; check that GA/DP in the feedback match your intent.

## Success case (real)
- NeMo, 4 nodes, GBS 64, MoE: EP32 and EP8 without activation checkpointing both OOMed at step 2. With full activation checkpointing, EP32 reached 22,217.02 tok/s at 44.98 GiB while EP8 reached 7,491.88 tok/s at 73.11 GiB: more expert parallelism cut both memory and time. (reports/winh8_20260925/REPORT.md, completed experiments)
- Fine-grained variant: the QKV all-gather of a narrow projection (width 1,664 gathered over TP8 to 13,312, then sliced) was replaced by two fixed size-4 subgroups inside TP8, while weights, DDP, optimizer and SP stayed TP8: 7.390488 -> 7.317272 s (+1.00%), exact. (MANUSCRIPT.md 4.5, Table 6)

## Failure case (real)
- flash-next MCore on 32 GPUs with expert-DP 1: the shipped config OOMed at step 2 in the MTP forward. Each rank owned 1,568 expert weights and the Muon FP32 master+momentum (~28.7 GiB) stayed resident; EDP1 also doubled per-rank expert optimizer work compared with EDP2 on 64 GPUs. It ran only with lossless state offload, at 20.572 s/step. Lesson: count expert optimizer state per EDP shard before shrinking EDP. (reports/flash_next_mfu_20261003/PROGRESS.md 1; EVIDENCE_PRIOR64.md)
