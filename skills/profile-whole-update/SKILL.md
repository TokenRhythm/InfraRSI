---
name: profile-whole-update
description: Measure and attribute time with the whole-update contract - window mean of max-rank update time, inter-step consistency, same-rank phase comparisons, and noise-aware acceptance.
license: Apache-2.0
when_to_use: Reading any receipt; deciding whether a candidate is faster; a component got faster but the update did not; stragglers; noisy windows; 'is this real', 'where does the time go', 'phase breakdown', 'why no speedup'.
metadata:
  author: InfraRSI A0 (human-authored)
  source: human-authored (A0)
  version: 1
  tags: [profile, measurement, timing, stragglers, noise, dense, moe, single-node, multi-node]
---

# Profile the whole update

## Trigger conditions
- Every screen result. Every claim that a change helped.
- `profile` shows one phase improving while StepTime does not.
- `step_time.rsd_window` above ~2%, or the tail (last 3 steps) above the window mean.

## Required observations (receipt fields)
- `step_time.mean_window_max_rank_s` (the metric), `median_window_max_rank_s`, `rsd_window`, `inter_step_mean_s`.
- `per_step[].max_rank_whole_s`, `per_step[].slowest_rank`; `profile` (per-phase window means of max-rank and min-rank: whole_update_s, fwd_bwd_s, optim_s, data_s) when the harness attaches it.

## Rules: the contract
- StepTime = mean over the scoring window (steps 17-30 of 30, or 21-40 of 40) of the maximum over ranks of the native whole-update time: all micro-batch forward/backward, gradient sync, optimizer step, and any park/restore. Startup, compile, checkpoint and export are reported separately (JobTime) and never enter StepTime.
- InterStep (observed wall time between updates) must move with StepTime: if |delta log InterStep - delta log StepTime| > 2% the candidate is invalid (work was moved outside the timed region).
- Updates and ranks inside one run are not independent samples; one screen is one observation. The locked scoring pair decides; a win needs log(t_FR/t) > log(1 + margin), margin = max(3%, 3 x RSD of the FR window).

## Rules: attribution
- Phase maxima over ranks are taken independently and are not additive: forward/backward 6.3967 -> 5.2221 s and optimizer 1.8655 -> 1.5273 s did not decompose the 64-GPU total.
- Compare phases on the same rank and step (parent vs child). Communication time includes waiting for the slowest peer: a 1.42 s "notify" cost was 196 calls x 7.3 ms of mostly waiting; one DP rank spending 2.31 s vs 0.68 s on attention made every other rank wait in the dispatcher.
- Kernel launches are not time: removing 2,400 launches and 384 syncs gave only +1.16%; a change that kept its launch count gained in host scope (549.05 -> 361.16 ms).
- Report speedup as t_parent/t_child - 1 on the same window: 10.373877 -> 8.504627 s is -18.02% time and +21.98% throughput.

## Rules: noise
- Two equivalent baseline runs differed by 0.363% at 64 GPUs. With two baselines, J = max(|delta full|, |delta half1|, |delta half2|); accept a gain only if it exceeds max(threshold, 2J) and both half-windows improve. In this benchmark the margin rule above is the acceptance rule.

## Allowed actions
- read_receipt / profile_ranges on parent and child before deciding the next change.
- retrieve_skill for the phase that dominates (memory, recompute, moe-ep-communication, optimize-data-path).
- finish with the best valid candidate; do not re-screen identical configs to "average out" noise (one screen per candidate).

## Failure signals
- Component metric improves, StepTime flat or worse (count it as a failure, record why).
- InterStep improves much less than StepTime (deferred work) or more (timer artifact).
- `slowest_rank` fixed on one rank: a straggler or load skew bounds the update; balance or overlap it instead of speeding up average ranks.

## Verification
- Same window, same task, fresh process: StepTime delta beyond the margin, InterStep consistent within 2%, tail ratio <= 1.25, no change in invariants.

## Success case (real)
- 64-GPU MCore flagship, four runtime changes against a repeated original: 8.2926 -> 6.7707 s (+22.48%); against the repeat original 8.2625 s, +22.03%. train_step wall 8.2790 -> 6.7591 s. The gain held across windows (1.2248x / 1.2326x / 1.2378x for windows 4-32 / 9-32 / 17-32). (MANUSCRIPT.md 6.3, Table 5)

## Failure case (real)
- FP8 batch-quantization fusion cut component launches from 64 to 3, yet the whole update stayed -4.62% / -3.89% versus BF16 on two node groups. A same-rank, same-step diff placed the excess in forward/backward (+445.3 / +454.4 ms) and park (+50.5 / +71.6 ms): weights were re-quantized under GA and recompute. Component speed did not survive into the update. (reports/low_precision_winh_20261014/quant_fusion_20261015; MANUSCRIPT.md 6.6)
