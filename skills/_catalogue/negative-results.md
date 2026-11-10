# Negative results catalogue (human-authored, A0)

Retrievable data file: ideas that were measured and did not pay off on the whole update, with numbers.

## Neg-1 FP8 batch-quantization fusion
57 component cases, launches 64 -> 3, seven 32-rank 30-update runs. Group A: stock FP8 11.121953 s, fused FP8 10.876133 s, BF16 10.373877 s (fused vs stock +2.26%, vs BF16 -4.62%). Group B: 11.138189 / 11.110150 / 10.678406 s (+0.25%, -3.89%). Stock and fused scalars matched for 30 updates. Transpose/workspace caches were not reused, so expert weights were re-quantized under gradient accumulation and recompute. Decision: keep BF16 for speed. (MANUSCRIPT.md 6.6, Table 9)

## Neg-2 Integer packing frontend
Medians before -> after (ms): offset 0 11.435618 -> 8.913383 (1,786 -> 1,306 kernels); 8192 19.226326 -> 14.551279; 16384 19.253847 -> 14.532480; 24576 19.143477 -> 14.441861. Whole-update opportunity 20 x sum / 4 = 0.083101325 s/update against a frozen 0.16 s gate (51.9%). Exact, not adopted, no 64-GPU A/B spent. (mfu_core_20260929/QSA_PACKING_FINAL_DISPOSITION_V1.md)

## Neg-3 Cohort allocation of state buffers
13 updates, then 1 allocator retry on rank 28 at update 14 (before restore), 0 OOM. Live memory only ~0.383 MiB lower; peak 70.951833 vs 70.952207 GiB; reserved fell 62 MiB. No snapshot taken, so the fragmentation hypothesis stayed unproven. (cohort_training_v1/FAILURE_AND_OPTIONS.md)

## Neg-4 Resume OOM inside NCCL
Continuous updates 1-6 with a save at 3 succeeded; resume from 3 failed at update 4 in the first grad-norm all-reduce (NCCL "Cuda failure 2 'out of memory'" on 4 ranks) with torch num_ooms = 0. At before_optimizer on rank 28 allocated memory was identical, reserved 72.332 vs 70.943 GiB, free 0.167 vs 1.156 GiB. NCCL allocates outside the PyTorch allocator at first use. (diagnostics/q_resume_nccl_oom_v1/SUMMARY.md)

## Smaller negatives
- DeepEP SMs 20 -> 16: -1.30%; 40 and 24: slower. Keep 20 unless measured.
- Selective recompute v2 (NeMo 64 GPUs, 2K): 30,143 vs 30,529 tok/s for full recompute with higher peak (30.29 vs 22.44 GiB).
- Mesh-grouped fused AdamW: +5.6% but BF16 moments and no stochastic rounding; loss 0.57656 vs 0.16340 after 12 steps. Invalid semantics.
- QSA selector CUDA graph: exact, missed the 0.16 s gate in all 48 full-cost scenarios.
- Async DeepEP dispatch: +0.44% in a single run, below noise; not adopted.
