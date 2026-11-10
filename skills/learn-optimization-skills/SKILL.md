---
name: learn-optimization-skills
description: How the optimizer improves itself between development tasks - what it may and may not edit, the promotion gate for a new package, memory entry format, and how to distill applicability conditions and counter-examples from a trajectory.
license: Apache-2.0
when_to_use: Reflection after a development task; writing memory entries during a search; deciding whether a lesson is general or task-specific; 'what did we learn', 'update skills', 'memory entry', 'reflect'.
metadata:
  author: InfraRSI A0 (human-authored)
  source: human-authored (A0)
  version: 1
  tags: [rsi, reflection, memory, skills, package, learning]
---

# Learn optimization skills

The optimizer A (skills, memory, retrieval and analysis tools, policy, prompts) is distinct from the target S (the
training job). Only A transfers between tasks. Training patches never transfer, and private retest results never
feed back into A.

## Trigger conditions
- After each development task (reflect), and during a search when a result explains something reusable (write_memory).

## Required observations (receipt fields)
- The finished task's trace: candidates with changes and verdicts, screens with StepTime and costs, invalid reasons, skill/memory/tool hit logs.

## Rules: what may change
- May edit: skills (conditions, rules, counter-examples, real cases), memory.db entries, retrieval and policy settings within caps, prompts, receipt-analysis tools (JSON analysis subset).
- May not edit: harness, reward, invariants, frozen zone, budgets, tolerances, model weights, the L0 schema, patch guard. They are outside the package.
- A new package must pass selfcheck (schema; skills parse with all sections; tools run on historical receipts; prompts keep placeholders; policy within caps; answer-isolation audit). Otherwise the previous package is kept and the reflection is recorded as invalid_fallback.

## Rules: memory entries
- Format: tags {hardware: [...], model: [...], data: [...]}, slow (config delta, StepTime), fast (config delta, StepTime), outcome, failure_reason, lesson. Tags make retrieval conditional on context; a lesson without tags generalizes blindly.
- Record failures with their mechanism (e.g. "OOM at update 2: lazy moments"), not just "failed".

## Rules: distilling skills
- Write conditions, not answers: "when headroom > X and recompute is full, remove recompute from k layers" rather than "use config c7".
- Keep counter-examples next to the rule: Muon three-phase optimizations have no AdamW counterpart; fastmask needs zero bases or falls back; NUMA binding depends on the node's NUMA layout; an external TP4 suggestion was reordered after timings contradicted it.
- A tool that changed but was never invoked is "unconsumed": it is not evidence of learning. Inheritance counts only when retrieval or tool logs show use.
- Re-solving a task seen before is an answer-isolated re-solve, not an unseen task.

## Allowed actions
- write_memory during search; in reflection: edit skills/policy/prompts/tools, add memory.

## Failure signals
- Package unchanged after a task with clear lessons; selfcheck failures; lessons that name one task's final config; held-out identifiers anywhere in the package (audit failure).

## Verification
- selfcheck passes; audit passes; later tasks' logs show the new skill sections or tools being retrieved/invoked.

## Success case (real)
- Tool inheritance (T5): compare_same_rank_stages.py, written for the FP8 investigation, read the BF16 anchor queue, paired (rank, step) and checked 448 pairs, then produced QSA/prefix/QKV/segsum analyses for later candidates. The row-wise dual-FP64 gate (T2), frozen before a kernel repair, rejected v1 and was reused on two more layer sets. (agentic_moe_sft_tech_report_20261018/RSI_TRACK.md T5, T2)

## Failure case (real)
- RSI pilots of the previous project: pilot-1 10.441060 -> 10.448436 s (retest 10.490263), pilot-2 10.556310 -> 10.607184 s. The A0 package came out unchanged, pilot-2 was interrupted, 0 of 6 formal lineages ran, and the evidence was judged insufficient (pilot-1 had 1 reflection cycle where 2 were needed). Without completed cycles and recorded inheritance, no self-improvement claim is possible. (RSI_TRACK.md 3, 6; rsi_pilot_v1/D1_PILOT_COMPLETION_V1.md)
