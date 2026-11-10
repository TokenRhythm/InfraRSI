"""F-PROMPTS (external bench design 5.1, W6): the PROMPTS / ASAP method ported to the flagship task as a one-shot
recommendation on the InfraRSI harness.

PROMPTS / ASAP (Google, TPU production workloads) publish no code or prompts, so what is ported is the method's shape:
ONE call to the driver model every arm uses, given the HardwareProfile summary, the task's workload facts and the
receipt summary of the start configuration (U+C on F1), returns exactly three L0 configurations (JSON schema checked).
Each one is refused before any screen when the L0 validator or the planner feasibility model rejects it; the accepted
ones are screened once each on plan A (formal reward against the start receipt) and the fastest valid one goes to the
locked pair {reference, selected} on plan B. No iteration, no skills, no memory, no verifier: the original method has
none. The call and the screens are charged to the search ledger like every arm's; this goal runs no promotion
lifecycle (design 5.2: screening-level verification only).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

from ..actions import load_schema
from ..actions.l0 import MOE_ONLY, NO_SEARCH
from ..agent.context import estimate_tokens, profile_ranges, receipt_digest
from ..agent.jsonschema_lite import validate as schema_errors
from ..agent.optimize import MIN_CALL_WALL_S, output_allowance
from ..agent.search import SearchSession, SearchTrace
from ..contracts import TaskSpec, digest
from ..planner.feasibility import estimate

ARM = 'F-PROMPTS'
N_CONFIGS = 3
LLM_TOKENS_CAP = 200_000          # design 6.2: F-PROMPTS 0.2M tokens
LLM_USD_CAP = 10.0                # design 6.2: per-arm hard cap (a single call costs well under $1)
CALL_TIMEOUT_S = 1800.0
RATIONALE_CHARS = 1200
RECEIPT_TOKENS = 2500             # receipt summary of the start configuration (agent/context.receipt_digest)

CONFIG_ITEM = {'type': 'object', 'properties': {'config': {'type': 'object'},
                                                'rationale': {'type': 'string', 'maxLength': RATIONALE_CHARS}},
               'required': ['config', 'rationale'], 'additionalProperties': False}
SCHEMA = {'type': 'object', 'properties': {'configs': {'type': 'array', 'minItems': N_CONFIGS, 'maxItems': N_CONFIGS,
                                                       'items': CONFIG_ITEM},
                                           'notes': {'type': 'string', 'maxLength': 2000}},
          'required': ['configs'], 'additionalProperties': False}
SYSTEM = ('You are a training-performance engineer who recommends Megatron-Core training configurations in one pass. '
          'Reply with exactly one JSON object and nothing else.')
FACT_KEYS = ('model_type', 'family', 'bridge_type', 'layers', 'hidden', 'heads', 'kv_heads', 'head_dim', 'vocab',
             'intermediate', 'tied_embeddings', 'layer_kinds', 'moe', 'linear_heads', 'breakdown', 'extras',
             'params_total', 'params_active', 'params_expert', 'params_dense', 'params_checkpoint',
             'cpu_resident_params', 'tp_candidates')
DEVIATIONS = [
    'PROMPTS / ASAP target internal TPU production workloads and publish no code or prompts: the prompt text is ours and '
    'reproduces the method shape (one-shot top-k recommendation from hardware, workload and profile facts).',
    'Driver: the model every arm uses (claude-opus-5-5 through the common tools-disabled CLI transport); one call, no '
    'retry; temperature is not settable.',
    'Action space: the InfraRSI L0 schema (Megatron-Core 0.18.2 keys and the env whitelist), recommendations given as '
    'changes on the start configuration (U+C on F1); planner-only "_" keys in a reply are ignored.',
    'Refusals: a recommendation the L0 validator, the duplicate check or the planner feasibility model rejects costs no '
    'screen and is not replaced (the method has no second round).',
    'Screening, judging and the locked pair run through the InfraRSI harness and formal reward like every arm '
    '(numerics E1 with the task tier\'s gates, AMENDMENT-002); no promotion lifecycle in this goal.',
    'No skills, memory, verifier or analysis tools (the original method has none).',
]


# ---------------------------------------------------------------- views shown to the model

def hardware_view(hardware: dict | None, nodes: int) -> dict:
    """The HardwareProfile summary planners read (discover/hardware.py), with its source; the WINH default profile
    when the dispatcher holds none."""
    from ..discover import default_profile
    prof = hardware if isinstance(hardware, dict) and isinstance(hardware.get('summary'), dict) else default_profile(nodes)
    return {'source': prof.get('source') or 'discover.hardware_profile (read-only probe)', 'summary': prof['summary'],
            'digest': prof.get('digest')}


def workload_view(task_raw: dict, facts: dict | None) -> dict:
    w, m, d, hw = task_raw['workload'], task_raw['model'], task_raw['data'], task_raw['hardware']
    return {'task_id': task_raw['id'], 'purpose': task_raw.get('purpose'),
            'model': {k: m.get(k) for k in ('key', 'repo_id', 'family', 'bridge_type', 'size_gb')},
            'data': {'key': d.get('key'), 'kind': d.get('kind'), 'format': d.get('format')},
            'workload': {k: w.get(k) for k in ('seq_len', 'packs_per_update', 'physical_tokens_per_update', 'screen_steps',
                                               'scoring_window')},
            'hardware': {k: hw.get(k) for k in ('gpus', 'nodes', 'gpu_type', 'placement')},
            'objective': task_raw.get('objective'),
            'model_facts': {k: (facts or {}).get(k) for k in FACT_KEYS if k in (facts or {})}}


def l0_view(model_facts: dict | None) -> dict:
    """The L0 action space as every arm's validator reads it (actions/schema): per key type, legal values or range,
    default and the schema's note; the env whitelist; the keys the task fixes and the cross-key rules."""
    s = load_schema()
    moe = bool((model_facts or {}).get('num_experts'))
    keys = {}
    for k, spec in s['keys'].items():
        if k in MOE_ONLY and not moe:
            continue
        row = {f: spec[f] for f in ('type', 'values', 'range', 'nullable', 'default') if spec.get(f) is not None}
        if k in NO_SEARCH:
            row['single_legal_value'] = True
        if spec.get('notes'):
            row['notes'] = str(spec['notes'])[:240]
        keys[k] = row
    env = {}
    for name, spec in s['env'].items():
        row = {f: spec[f] for f in ('type', 'values', 'range') if spec.get(f) is not None}
        if isinstance(spec.get('options'), dict):
            row['options'] = sorted(spec['options'])
        if spec.get('notes'):
            row['notes'] = str(spec['notes'])[:160]
        env[name] = row
    return {'l0_keys': keys, 'env_whitelist': env, 'fixed_by_task': s['fixed_by_task']['keys'],
            'rules': [f'{c["id"]}: {c["text"]}' for c in s['constraints']]}


def start_view(start_config: dict, start_receipt: dict, start_verdict: dict | None) -> dict:
    """The current configuration (resolved L0 + env), the placements outside L0 it was planned with, and the summary
    of its measured screen: the receipt digest the agent reads plus the per-phase profile."""
    view = receipt_digest(start_receipt, start_verdict, max_tokens=RECEIPT_TOKENS)
    view['profile'] = profile_ranges(start_receipt)
    view['effective_config'] = start_receipt.get('effective_config')
    return view


def build_prompt(task_raw: dict, hardware: dict, workload: dict, current: dict, placements: dict, measured: dict,
                 action_space: dict, *, tolerance: dict, peak_cap_gib) -> str:
    w = task_raw['workload']
    a, b = w['scoring_window']
    dump = lambda o: json.dumps(o, ensure_ascii=False, default=str)            # noqa: E731
    parts = [
        '# METHOD',
        f'One-shot configuration recommendation: from the hardware, the workload and the measured screen of the current '
        f'configuration, recommend the {N_CONFIGS} configurations most likely to lower the whole-update step time. This '
        'is the only round: you will see no result and nothing is retried.',
        '# OBJECTIVE AND RULES',
        f'- StepTime = mean over updates {a}..{b} (of {w["screen_steps"]}) of the max-over-ranks whole-update time, '
        'measured in a fresh process on a fixed input plan. Lower is better; only a gain beyond noise (at least 3 %) counts.',
        '- Fixed by the task, never change: global batch (packs per update), sequence length, data and its order, loss, '
        'optimizer and LR schedule, precision (bf16 parameters, fp32 master weights, moments and gradient reduction), '
        'trainable parameters, router top-k and capacity, model definition. Only the L0 keys and the environment '
        'variables under ACTION SPACE may change.',
        f'- A run is invalid on an out-of-memory error, on peak reserved memory above {peak_cap_gib} GiB per GPU, or on '
        f'numerics outside the task tolerance {dump(tolerance)} (relative to the current configuration\'s run).',
        '- Each recommendation lists only the L0 keys it changes; every other key keeps its value in CURRENT '
        'CONFIGURATION. "env" entries are merged into the current env (null removes one). Keys starting with "_" are '
        'ignored.',
        f'- The {N_CONFIGS} recommendations must differ from the current configuration and from each other. A '
        'recommendation the validator or the memory model rejects is dropped without a run and is not replaced.',
        '# HARDWARE', dump(hardware),
        '# WORKLOAD', dump(workload),
        '# CURRENT CONFIGURATION', dump(current),
        '# PLACEMENTS OUTSIDE L0 (fixed for every recommendation)', dump(placements),
        '# CURRENT CONFIGURATION: MEASURED SCREEN', dump(measured),
        '# ACTION SPACE', dump(action_space),
        '# ANSWER',
        f'Exactly one JSON object: {{"configs": [{{"config": {{"<L0 key>": <value>, "env": {{"<NAME>": "<value>"}}}}, '
        f'"rationale": "<why it is faster, at most 600 characters>"}}, ...]}} with exactly {N_CONFIGS} entries, best first.',
    ]
    return '\n'.join(parts)


# ---------------------------------------------------------------- reply, refusals, feasibility

def parse_reply(res) -> tuple[list | None, list[str]]:
    """(configs, problems): the reply's exactly-N_CONFIGS entries when the call succeeded and the object validates
    against SCHEMA; otherwise no configuration at all (one call, nothing is repaired or re-asked)."""
    if res.error or not isinstance(res.action, dict):
        return None, [res.error or 'the reply is not a JSON object']
    problems = schema_errors(res.action, SCHEMA)
    if problems:
        return None, problems[:8]
    return list(res.action['configs']), []


def placements_of(start_config: dict | None) -> dict:
    """Planner-only placements of the start configuration ('_' keys: never sent to WINH, no L0 key moves them)."""
    return {k: v for k, v in (start_config or {}).items() if str(k).startswith('_')}


def _for_estimate(config: dict, placements: dict) -> dict:
    """A resolved L0 config as the feasibility model reads it: ETP null means 'same as TP' (Megatron-Core default)."""
    cfg = {**config, **placements}
    if cfg.get('expert_tensor_parallel_size') is None:
        cfg.pop('expert_tensor_parallel_size', None)
    return cfg


def feasibility(task: TaskSpec, config: dict, placements: dict, hardware: dict | None, facts: dict,
                start_estimate: dict | None = None) -> tuple[bool, dict]:
    """(ok, detail) of planner.feasibility.estimate for a resolved L0 config with the start's placements. Refused when
    the layout is invalid or the memory estimate does not fit. When the model cannot fit even the start configuration
    (which ran), memory is judged relative to that start estimate instead of the absolute budget ('relative_to_start')."""
    est = estimate(task, _for_estimate(config, placements), hardware, facts)
    relative = bool(start_estimate) and start_estimate['valid_layout'] and not start_estimate['fits']
    if relative:
        gpu_ok = est['gpu_gib'] <= max(est['gpu_budget_gib'], start_estimate['gpu_gib'])
        host_ok = est['host_gib_per_node'] <= max(est['host_budget_gib'], start_estimate['host_gib_per_node'])
    else:
        gpu_ok, host_ok = est['gpu_fits'], est['host_fits']
    ok = bool(est['valid_layout'] and gpu_ok and host_ok)
    st = est.get('step_time_s') or {}
    return ok, {'fits': ok, 'basis': 'relative_to_start' if relative else 'absolute', 'valid_layout': est['valid_layout'],
                'gpu_gib': round(est['gpu_gib'], 2), 'gpu_budget_gib': round(est['gpu_budget_gib'], 2),
                'host_gib_per_node': round(est['host_gib_per_node'], 1), 'host_budget_gib': round(est['host_budget_gib'], 1),
                'est_step_s': round(st['total'], 3) if st.get('total') else None, 'notes': list(est['notes'])[:6]}


def _propose(session: SearchSession, index: int, item: dict, task: TaskSpec, placements: dict, hardware, facts: dict,
             start_estimate: dict) -> dict:
    changes = dict(item.get('config') or {})
    rec = {'index': index, 'config': item.get('config'), 'rationale': str(item.get('rationale') or '')[:RATIONALE_CHARS]}
    ignored = sorted(k for k in changes if str(k).startswith('_'))
    if ignored:                                      # placements are the start's, never the model's
        rec['ignored_keys'] = ignored
        changes = {k: v for k, v in changes.items() if k not in ignored}
    fb = session.propose_config(changes=changes, base='c0', origin='prompts', rationale=rec['rationale'])
    if 'candidate_id' not in fb:
        dup = 'duplicate_of' in fb
        rec.update(outcome='refused_duplicate' if dup else 'refused_l0',
                   detail=f'duplicate of {fb["duplicate_of"]}' if dup else (fb.get('errors') or [fb.get('error')])[:6])
        return rec
    cid = fb['candidate_id']
    cand = session.candidates[cid]
    ok, detail = feasibility(task, cand.config, placements, hardware, facts, start_estimate)
    rec.update(candidate_id=cid, change=fb.get('change'), feasibility=detail)
    if not ok:                                       # no screen is spent on a configuration that cannot fit
        cand.status = 'vetoed'
        session.count_invalid('infeasible')
        session.event('feasibility_refused', {'candidate_id': cid, 'detail': detail})
        rec['outcome'] = 'refused_infeasible'
        return rec
    rec['outcome'] = 'accepted'
    return rec


# ---------------------------------------------------------------- the method

def run_prompts(task, budget, start_config: dict, runner, *, llm_backend, judge=None, model_facts=None, facts=None,
                hardware=None, run_id: str = ARM, out_dir=None, start_receipt=None, reference_receipt=None, clock=None,
                tolerance: dict | None = None, peak_cap_gib=78.0, hooks=None) -> SearchTrace:
    """hooks (an ext environment's SessionHooks, e.g. ext/flashnext F-PROMPTS): run_prompts_ext - the environment's task
    view and validator replace the L0 views and the planner feasibility precheck; hooks=None is the core path, unchanged."""
    if hooks is not None:
        return run_prompts_ext(task, budget, start_config, runner, llm_backend=llm_backend, judge=judge, hooks=hooks,
                               run_id=run_id, out_dir=out_dir, start_receipt=start_receipt,
                               reference_receipt=reference_receipt, clock=clock)
    spec = task if hasattr(task, 'raw') else TaskSpec(raw=task)
    raw = spec.raw
    if not isinstance(start_receipt, dict):
        raise ValueError('F-PROMPTS needs the start configuration\'s screen receipt (its summary is the method input)')
    describe = getattr(llm_backend, 'describe', None)
    session = SearchSession(task=spec, budget=budget, runner=runner, judge=judge, run_id=run_id, arm=ARM, out_dir=out_dir,
                            start_config=start_config, start_receipt=start_receipt, reference_receipt=reference_receipt,
                            model_facts=model_facts, mode='strong', layers=('L0',), clock=clock,
                            driver_info={'kind': ARM, 'llm': describe() if callable(describe) else type(llm_backend).__name__})
    c0 = session.candidates['c0']
    placements = placements_of(start_config)
    if not facts:
        from ..discover import workload_facts
        facts = workload_facts(spec)
    start_estimate = estimate(spec, _for_estimate(c0.config, placements), hardware, facts)   # raises before any LLM cost
    if tolerance is None:
        from ..harness.validate import numerics_tier_check
        tolerance = (numerics_tier_check(raw)['table'] or {}).get('tolerance') or (raw.get('numerics') or {}).get('tolerance')
    hw_view = hardware_view(hardware, spec.nodes)
    prompt = build_prompt(raw, hw_view, workload_view(raw, facts), c0.config, placements,
                          start_view(start_config, start_receipt, c0.verdict), l0_view(model_facts),
                          tolerance=tolerance, peak_cap_gib=peak_cap_gib)
    call = {'role': 'prompts_recommend', 'prompt_sha256': hashlib.sha256((SYSTEM + '\x00' + prompt).encode('utf-8')).hexdigest(),
            'prompt_tokens_est': estimate_tokens(SYSTEM + prompt), 'made': False}
    out = Path(out_dir) if out_dir else None
    if out:
        (out / 'prompts').mkdir(parents=True, exist_ok=True)
        (out / 'prompts' / 'system.txt').write_text(SYSTEM, encoding='utf-8')
        (out / 'prompts' / 'prompt.txt').write_text(prompt, encoding='utf-8')
    items, stop = None, None
    cap = output_allowance(budget, session.ledger, llm_backend, SYSTEM + prompt)
    wall_left = budget.wall_seconds - session.ledger.wall()
    if session.ledger.exhausted():
        stop = 'budget:' + session.ledger.exhausted()[0]
    elif cap is None:
        stop = 'budget:llm_tokens'
    elif wall_left < MIN_CALL_WALL_S:
        stop = 'budget:wall_seconds'
    else:
        usd_left = None if budget.llm_usd is None else max(0.0, budget.llm_usd - session.ledger.used['llm_usd'])
        res = llm_backend.complete(SYSTEM, prompt, schema=SCHEMA, call_dir=(out / 'calls' / '0001_prompts_recommend') if out else None,
                                   role='prompts_recommend', max_output_tokens=cap, max_usd=usd_left,
                                   timeout=max(1.0, min(CALL_TIMEOUT_S, wall_left)))
        session.charge_llm(res, 'prompts_recommend')
        items, problems = parse_reply(res)
        call.update(made=True, error=res.error, problems=problems, llm_tokens=res.llm_tokens, llm_usd=res.llm_usd,
                    accounting_complete=res.accounting_complete, call_id=res.meta.get('call_id'))
        if out:
            (out / 'prompts' / 'reply.json').write_text(json.dumps({'action': res.action, 'text': res.text, 'error': res.error,
                                                                   'problems': problems}, ensure_ascii=False, indent=1,
                                                                  default=str), encoding='utf-8')
        if items is None:
            stop = 'llm_error' if res.error else 'llm_reply_invalid'
    recs = [_propose(session, i, item, spec, placements, hardware, facts, start_estimate)
            for i, item in enumerate(items or [], 1)]
    for rec in recs:                                 # accepted recommendations, in the model's order: one screen each
        if rec['outcome'] != 'accepted':
            continue
        fb = session.screen(rec['candidate_id'])
        if 'error' in fb:
            rec.update(outcome='not_screened', detail=fb['error'])
            continue
        rec.update(outcome='screened', valid=fb.get('valid'), step_time=fb.get('step_time'), reasons=fb.get('reasons'),
                   vs_start=fb.get('vs_start'))
    if stop is None:
        unscreened = [r for r in recs if r['outcome'] == 'not_screened']
        stop = (session.stop_flag or 'budget') if unscreened else (
            'done' if any(r['outcome'] == 'screened' for r in recs) else 'no_screenable_recommendation')
    # The arm's selection follows the rule of every arm (SearchSession.finalize: the start wins unless a valid candidate
    # is faster, and then nothing is promoted); the best valid recommendation is reported for the budget-matched
    # comparison with F-A0's first three screens (design 5.2), whether or not it beat the start.
    valid = [r for r in recs if r['outcome'] == 'screened' and r.get('valid') and r.get('step_time')]
    best = min(valid, key=lambda r: (r['step_time'], r['index']), default=None)
    extra = {'method': 'PROMPTS / ASAP one-shot top-3 recommendation', 'deviations': DEVIATIONS,
             'prompts': {'call': call, 'recommendations': recs, 'placements': placements,
                         'best_recommendation': best and {k: best.get(k) for k in ('index', 'candidate_id', 'step_time',
                                                                                    'vs_start')},
                         'start_estimate': {k: start_estimate.get(k) for k in ('fits', 'valid_layout', 'gpu_gib',
                                                                              'gpu_budget_gib', 'host_gib_per_node')},
                         'refused': sum(1 for r in recs if r['outcome'].startswith('refused')),
                         'screened': sum(1 for r in recs if r['outcome'] == 'screened')}}
    return session.finalize(stop_reason=stop, extra=extra)


# ---------------------------------------------------------------- the method on an external environment (hooks)

SYSTEM_EXT = ('You are a training-performance engineer who recommends training configurations in one pass. '
              'Reply with exactly one JSON object and nothing else.')
EXT_DEVIATIONS = [d for d in DEVIATIONS if d.startswith(('PROMPTS / ASAP', 'Driver:', 'No skills'))] + [
    "Action space: the external environment's own switch / knob table and validator (its SessionHooks; on F1 the S_F1 "
    'switch space, D-F1-2); recommendations are changes on the start configuration (U+C on F1).',
    "Refusals: a recommendation the environment's validator or the duplicate check rejects costs no screen and is not "
    'replaced (the method has no second round); the L0 feasibility model does not apply (on F1 the layout is fixed and '
    'memory is proven by the start configuration itself).',
    "Screening and judging through the environment's runner and judge like every arm of that environment (on F1: E1 "
    'numerics against the U+C reference screen, receipts signed); the locked pair is run by the environment\'s arm code; '
    'no promotion lifecycle.',
]


def build_prompt_ext(task_view: dict, current: dict, measured: dict) -> str:
    """The method's prompt from an environment's task view (hardware profile, workload / fixed facts, switch table,
    gates, budget), the start configuration and the digest of its measured screen."""
    dump = lambda o: json.dumps(o, ensure_ascii=False, default=str)            # noqa: E731
    view = {k: v for k, v in task_view.items() if k not in ('start_measured', 'contract')}
    metric = (task_view.get('metric') or {})
    parts = [
        '# METHOD',
        f'One-shot configuration recommendation: from the hardware, the workload and the measured screen of the current '
        f'configuration, recommend the {N_CONFIGS} configurations most likely to lower the whole-update step time. This '
        'is the only round: you will see no result and nothing is retried.',
        '# OBJECTIVE AND RULES',
        f'- {metric.get("name", "StepTime")}: {metric.get("definition", "the task metric")}; measured in a fresh process. '
        'Lower is better; only a gain beyond noise (at least 3 %) counts.',
        '- Everything not listed as a switch or knob in the TASK block is fixed by the task and never changes.',
        '- A run is invalid when it does not complete or when its numerics leave the gates in the TASK block (relative to '
        "the current configuration's run).",
        '- Each recommendation lists only the switches it changes; every other switch keeps its value in CURRENT '
        'CONFIGURATION.',
        f'- The {N_CONFIGS} recommendations must differ from the current configuration and from each other. A '
        'recommendation the validator rejects is dropped without a run and is not replaced.',
        '# CONTRACT', str(task_view.get('contract') or ''),
        '# TASK', dump(view),
        '# CURRENT CONFIGURATION', dump(current),
        '# CURRENT CONFIGURATION: MEASURED SCREEN', dump(measured),
        '# ANSWER',
        f'Exactly one JSON object: {{"configs": [{{"config": {{"<switch>": <value>}}, '
        f'"rationale": "<why it is faster, at most 600 characters>"}}, ...]}} with exactly {N_CONFIGS} entries, best first.',
    ]
    return '\n'.join(parts)


def _propose_ext(session: SearchSession, index: int, item: dict) -> dict:
    changes = dict(item.get('config') or {})
    rec = {'index': index, 'config': item.get('config'), 'rationale': str(item.get('rationale') or '')[:RATIONALE_CHARS]}
    fb = session.propose_config(changes=changes, base='c0', origin='prompts', rationale=rec['rationale'])
    if 'candidate_id' not in fb:
        dup = 'duplicate_of' in fb
        rec.update(outcome='refused_duplicate' if dup else 'refused_invalid',
                   detail=f'duplicate of {fb["duplicate_of"]}' if dup else (fb.get('errors') or [fb.get('error')])[:6])
        return rec
    rec.update(candidate_id=fb['candidate_id'], change=fb.get('change'), outcome='accepted')
    return rec


def run_prompts_ext(task, budget, start_config: dict, runner, *, llm_backend, hooks, judge=None, run_id: str = ARM,
                    out_dir=None, start_receipt=None, reference_receipt=None, clock=None) -> SearchTrace:
    """F-PROMPTS on an external environment: the start check is the environment's (hooks.start_receipt_problems, e.g.
    the signed U+C reference receipt on F1), the validator replaces the L0 validator and the feasibility precheck; one
    call, exactly three recommendations, one screen per accepted one, selection by the common rule."""
    if not isinstance(start_receipt, dict):
        raise ValueError("F-PROMPTS needs the start configuration's screen receipt (its summary is the method input)")
    check = getattr(hooks, 'start_receipt_problems', None)
    problems = check(start_receipt, start_config) if callable(check) else []
    if problems:
        raise ValueError('the start receipt cannot be the F-PROMPTS input: ' + '; '.join(problems[:4]))
    describe = getattr(llm_backend, 'describe', None)
    session = SearchSession(task=task, budget=budget, runner=runner, judge=judge, run_id=run_id, arm=ARM, out_dir=out_dir,
                            start_config=start_config, start_receipt=start_receipt, reference_receipt=reference_receipt,
                            mode='strong', layers=('L0',), clock=clock, hooks=hooks,
                            driver_info={'kind': ARM, 'llm': describe() if callable(describe) else type(llm_backend).__name__})
    c0 = session.candidates['c0']
    measured = hooks.receipt_digest(start_receipt, c0.verdict, max_tokens=RECEIPT_TOKENS)
    prompt = build_prompt_ext(hooks.task_view(), c0.config, measured)
    call = {'role': 'prompts_recommend', 'prompt_sha256': hashlib.sha256((SYSTEM_EXT + '\x00' + prompt).encode('utf-8')).hexdigest(),
            'prompt_tokens_est': estimate_tokens(SYSTEM_EXT + prompt), 'made': False}
    out = Path(out_dir) if out_dir else None
    if out:
        (out / 'prompts').mkdir(parents=True, exist_ok=True)
        (out / 'prompts' / 'system.txt').write_text(SYSTEM_EXT, encoding='utf-8')
        (out / 'prompts' / 'prompt.txt').write_text(prompt, encoding='utf-8')
    items, stop = None, None
    cap = output_allowance(budget, session.ledger, llm_backend, SYSTEM_EXT + prompt)
    wall_left = budget.wall_seconds - session.ledger.wall()
    if session.ledger.exhausted():
        stop = 'budget:' + session.ledger.exhausted()[0]
    elif cap is None:
        stop = 'budget:llm_tokens'
    elif wall_left < MIN_CALL_WALL_S:
        stop = 'budget:wall_seconds'
    else:
        usd_left = None if budget.llm_usd is None else max(0.0, budget.llm_usd - session.ledger.used['llm_usd'])
        res = llm_backend.complete(SYSTEM_EXT, prompt, schema=SCHEMA,
                                   call_dir=(out / 'calls' / '0001_prompts_recommend') if out else None,
                                   role='prompts_recommend', max_output_tokens=cap, max_usd=usd_left,
                                   timeout=max(1.0, min(CALL_TIMEOUT_S, wall_left)))
        session.charge_llm(res, 'prompts_recommend')
        items, problems = parse_reply(res)
        call.update(made=True, error=res.error, problems=problems, llm_tokens=res.llm_tokens, llm_usd=res.llm_usd,
                    accounting_complete=res.accounting_complete, call_id=res.meta.get('call_id'))
        if out:
            (out / 'prompts' / 'reply.json').write_text(json.dumps({'action': res.action, 'text': res.text, 'error': res.error,
                                                                   'problems': problems}, ensure_ascii=False, indent=1,
                                                                  default=str), encoding='utf-8')
        if items is None:
            stop = 'llm_error' if res.error else 'llm_reply_invalid'
    recs = [_propose_ext(session, i, item) for i, item in enumerate(items or [], 1)]
    for rec in recs:                                 # accepted recommendations, in the model's order: one screen each
        if rec['outcome'] != 'accepted':
            continue
        fb = session.screen(rec['candidate_id'])
        if 'error' in fb:
            rec.update(outcome='not_screened', detail=fb['error'])
            continue
        rec.update(outcome='screened', valid=fb.get('valid'), step_time=fb.get('step_time'), reasons=fb.get('reasons'),
                   vs_start=fb.get('vs_start'))
    if stop is None:
        unscreened = [r for r in recs if r['outcome'] == 'not_screened']
        stop = (session.stop_flag or 'budget') if unscreened else (
            'done' if any(r['outcome'] == 'screened' for r in recs) else 'no_screenable_recommendation')
    valid = [r for r in recs if r['outcome'] == 'screened' and r.get('valid') and r.get('step_time')]
    best = min(valid, key=lambda r: (r['step_time'], r['index']), default=None)
    extra = {'method': 'PROMPTS / ASAP one-shot top-3 recommendation', 'deviations': EXT_DEVIATIONS,
             'prompts': {'call': call, 'recommendations': recs,
                         'best_recommendation': best and {k: best.get(k) for k in ('index', 'candidate_id', 'step_time',
                                                                                    'vs_start')},
                         'refused': sum(1 for r in recs if r['outcome'].startswith('refused')),
                         'screened': sum(1 for r in recs if r['outcome'] == 'screened'),
                         'hooks': hooks.describe() if callable(getattr(hooks, 'describe', None)) else None}}
    return session.finalize(stop_reason=stop, extra=extra)


# ---------------------------------------------------------------- dispatcher arm

def prompts_budget(task, run: dict):
    """The task's search budget lowered by the run row (agent.arms.budget_for), then capped by the arm: 3 screens,
    0.2M LLM tokens, $10 (a run row may lower llm_usd_cap further)."""
    from ..agent import arms as A
    b = A.budget_for(task, run)
    usd = LLM_USD_CAP if run.get('llm_usd_cap') is None else min(LLM_USD_CAP, float(run['llm_usd_cap']))
    return dataclasses.replace(b, screens=min(b.screens, N_CONFIGS), llm_tokens=min(b.llm_tokens, LLM_TOKENS_CAP),
                               llm_usd=usd if b.llm_usd is None else min(b.llm_usd, usd))


def start_problems(start_receipt, start_config: dict, task) -> list[str]:
    """The receipt summarised for the model must be the signed, valid, numerics-verified plan-A screen of exactly the
    start configuration (as sent: raw or without planner-only keys)."""
    from ..agent import arms as A
    from ..agent.harness_runner import wire_config
    p = A.authenticate(start_receipt, task, purpose='screen', plan='A')
    if isinstance(start_receipt, dict) and start_receipt.get('config_digest') not in (digest(start_config),
                                                                                      digest(wire_config(start_config))):
        p.append('the start receipt is not a screen of the recorded start configuration (config_digest)')
    return p


def locked_summary(pair: dict) -> dict:
    """The locked plan-B numbers of the pair against its reference (U+C on F1): log speedup and the margin rule."""
    t_ref = pair.get('t_fr_locked') or pair.get('t_ed_locked')
    t = pair.get('t_cand_locked')
    g = math.log(t_ref / t) if t_ref and t else None
    m = pair.get('margin')
    return {'reference_kind': pair.get('reference_kind'), 't_ref_locked': t_ref, 't_cand_locked': t, 'log_speedup': g,
            'margin': m, 'beyond_margin': (g > math.log(1 + m)) if g is not None and m is not None else None,
            'promoted': pair.get('promoted'), 'invalid_accept': pair.get('invalid_accept')}


def prompts_search(*, run, task, nodes, start_config, start_receipt, reference_receipt, runner, out_dir, ctx) -> dict:
    """search_fn for dispatch.handlers.search_handler(..., 'ED'): the start slot holds the task's start configuration
    (U+C for F1, recorded with its plan-A screen); every screen goes through HarnessScreenRunner on the run's nodes."""
    from ..agent import arms as A
    from ..agent.harness_runner import HarnessScreenRunner
    from ..discover import default_profile, workload
    out_dir = Path(out_dir)
    refs = ctx.store.references().get(task.id, {})
    ref_path, reference_receipt, ref_kind = A.numerics_reference(refs, reference_receipt, start_receipt)
    problems = start_problems(start_receipt, start_config, task)
    if problems:
        raise RuntimeError(f'{run["run_id"]}: the start receipt cannot be the F-PROMPTS input: ' + '; '.join(problems[:4]))
    mf = A.model_facts_for(task, ctx.remote)
    facts = workload.workload_facts(task, remote=ctx.remote, cache_dir=workload.CACHE)     # the cache model_facts_for read
    hw = ctx.hw or default_profile(task.nodes)
    cfg = A.task_reward(task)
    hr = HarnessScreenRunner(runner, nodes, reference_receipt_path=ref_path, reference_receipt=reference_receipt,
                             wall_seconds_cap=A.attempt_cap(run), log=ctx.log)
    trace = run_prompts(task, prompts_budget(task, run), start_config, hr, llm_backend=A.DRIVERS['strong'](),
                        judge=A.formal_judge(task), model_facts=mf, facts=facts, hardware=hw, run_id=run['run_id'],
                        out_dir=out_dir, start_receipt=start_receipt, reference_receipt=reference_receipt,
                        peak_cap_gib=(cfg.get('hard') or {}).get('peak_reserved_gib_max', 78.0)).to_dict()
    result = {'arm': run['arm'], 'task': task.id, 'numerics_reference': ref_kind, 'trace': A.trace_summary(trace),
              'prompts': (trace.get('extra') or {}).get('prompts')}
    ed_score, ed_problems = A.load_ed_score(task, refs, ctx, nodes)
    res = A.finish(result, task, run, hr, refs, trace, out_dir, ed_score=ed_score, ed_problems=ed_problems,
                   start_config=start_config, lifecycle=False)
    res['locked_summary'] = locked_summary(res.get('locked_pair') or {})
    A._write_json(out_dir / 'result.json', res)
    return res
