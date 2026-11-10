---
name: weak-model-protocol
description: Protocol that lets a cheaper LLM drive the optimizer safely - JSON-schema actions, one action per turn from an explicit menu, configs before patches, templates before free patches, small injected context, verifier checklist and bounded retries.
license: Apache-2.0
when_to_use: The driver is a weaker or cheaper model (API or self-hosted); invalid or malformed actions; context overflow; provider errors; 'weak model', 'DeepSeek API driver', 'json schema', 'retries', 'menu'.
metadata:
  author: InfraRSI A0 (human-authored)
  source: human-authored (A0)
  version: 1
  tags: [weak-model, protocol, json, actions, retries, context, verifier]
---

# Weak-model protocol

The same harness guarantees validity for every driver; the protocol only makes a weaker model productive and cheap.
Numerics, budgets and applicability are checked by programs; the model chooses and explains.

## Trigger conditions
- Weak mode (policy.modes.weak), or any driver producing malformed actions, oversize patches or repeated invalid configs.

## Required observations (receipt fields)
- Receipt digests (<= 1.5K tokens) instead of raw receipts; the STATE block (budget, best, L1 gate, templates).

## Rules (FINAL_PLAN 7.2)
1. Actions are JSON constrained by a schema (response_format json_object for APIs, guided decoding for vLLM/SGLang), temperature 0.
2. One action per turn from an explicit menu: inspect / propose_config / apply_patch_template / propose_patch / run_trial / write_memory / finish.
3. Configs first: L1 stays locked until enough L0 screens ran (default 6) or L0 stalls (3 screens without a new best). Validated templates before any free patch; free patches <= 80 changed lines. L2 is closed.
4. Inject only the top-2 relevant skill sections (<= 4K tokens) and a <= 1.5K-token receipt digest.
5. The verifier re-checks with the checklist and may veto.
6. Context <= 32K tokens with a rolling summary; structured state (candidates table) is persistent and re-rendered each turn.
7. At most 3 retries per turn after a rejected reply, each with the validation error fed back; every attempt is charged.
8. Never escalate silently to a stronger model; the served model id is checked on every reply.

## Allowed actions
- The menu above; configs as small deltas (<= 6 keys) on a named base candidate.

## Failure signals
- Same invalid config proposed twice (duplicate detector), schema errors after 3 retries, provider errors (HTTP 429/5xx), context near the limit.

## Verification
- Accounting complete for every call (tokens and cost known, model id matches); JSON actions valid; InvalidAccept = 0.

## Success case (real)
- Closest project record: the 32-GPU first RSI pilot (pilot-1) drove the model with "exactly one JSON action, without markdown" over seven actions (read_source, define_analysis, run_analysis, patch_target, submit_candidate, propose_package, finish); every call was a fresh tools-disabled session and every call's cost was accounted. The protocol worked; the search itself found no speedup (B0 / c1 / retest 10.441060 / 10.448436 / 10.490263 s). (rsi_infra_execution_20261018/rsi_pilot_v1/D1_PILOT_COMPLETION_V1.md)

## Failure case (real)
- The second pilot (pilot-2) driver hit 23 consecutive provider 429 responses (calls 11-33); call 34 only reached STARTED, no session end, selection unfrozen, no retest. The known lower bound of spent cost was 132,677 tokens and USD 2.7731617499999994. Lesson: malformed or failed calls cost budget and must be visible; there is no hidden retry, and an exception without a cost receipt closes accounting. Model self-claims also need program checks: a claim that "the AST blocks in-place mutation" was refuted by dict |= and nested-list +=. (rsi_pilot_v1/D2_SEARCH_EXTERNAL_TERMINATION_V1.md; agentic_moe_sft_tech_report_20261018/RSI_TRACK.md 9)
