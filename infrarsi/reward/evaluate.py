"""evaluate(receipt, parent_receipt, reward_cfg) -> metric vector, hard constraints, checklist, scalar R.

Speed is the only objective (revamp design A): R = log(T_parent / T) for a valid candidate, -inf for an invalid one,
None without a parent. There are no presets, weights, memory or loss terms. Reads only harness-validated receipts
(step_time/per_step re-derived from signals); any hard violation gives valid=False. The hard items are also returned
in the generic checklist format (verify.core.from_core) with feedback for the search.

Numerics gates follow AMENDMENT-001 (reward.yaml tolerance): CE per-step and window-mean deviation, the grad-norm
deviation of the first update and the scoring-window mean log ratio (later per-step grad norm is reported only);
every run, reference or candidate, must be free of adjacent bitwise-repeated (main_ce, grad_norm). AMENDMENT-002: a
gate given per numerics tier ({tier | 'moe': value}) takes the value of the receipt's tier: the model facts of its
frozen task spec (bound by task_digest), which the harness-signed harness.numerics.tier must match when present.
Without a resolvable tier the receipt is invalid (numerics_tier) and every per-tier gate is 0. These predeclared
tiers are the benchmark tasks' tolerance.

Fail closed: StepTime and InterStep come only from the harness-derived values (no fallback to what the training
process reported) and a missing required time is a violation, never a skipped check. The parent receipt (and the
breakeven reference) go through the harness's one reference loader (validate.reference_problems: signed, valid, same
task and plan, same purpose, complete and finite). With promotion=True the lifecycle and plan-B receipts are bound to
the candidate: harness signature, same task / code / config / overlay digests, purpose 'lifecycle' resp. 'score' on
plan B (the candidate itself is the plan-A screen).
"""
from __future__ import annotations

import math

from infrarsi.contracts import TaskSpec, digest
from infrarsi.harness.validate import (NUMERICS_ORDER, NUMERICS_TIERS, is_tier_map, numerics_tier, reference_problems,
                                         tier_value, verify_receipt_digest)

from ..verify import checklist as _checklist
from ..verify.core import from_core
from .breakeven import breakeven_from_receipts
from .presets import load_reward_cfg, resolve

INVARIANT_PREFIXES = ('identity.', 'inv.', 'ref.', 'signals', 'dtype.', 'steps.', 'receipt.', 'world_size.', 'data.', 'lifecycle.',
                      'self_check.')


def _tol(tol: dict, key: str) -> float:
    """A numeric gate of the resolved reward config; missing or non-numeric (report_only) gates are 0 (fail closed)."""
    v = (tol or {}).get(key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else 0.0


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def tier_tolerance(tol: dict, tier) -> dict:
    """The reward config's gates for one numerics tier (AMENDMENT-002): per-tier entries resolve to the tier's value;
    one that cannot be resolved (unknown tier, no entry) becomes None, which _tol treats as 0 (fail closed)."""
    out = {}
    for k, v in (tol or {}).items():
        if is_tier_map(v):
            try:
                out[k] = tier_value(v, tier) if tier in NUMERICS_TIERS else None
            except KeyError:
                out[k] = None
        else:
            out[k] = v
    return out


def _d(x) -> dict:
    return x if isinstance(x, dict) else {}


def step_time(r: dict | None):
    return _num(_d(_d(r).get('step_time')).get('mean_window_max_rank_s'))


def inter_step(r: dict | None):
    """InterStep re-derived by the harness from the signals; None when it could not be derived (never the trainer's)."""
    return _num(_d(_d(r).get('step_time')).get('inter_step_mean_s'))


def peak_reserved(r: dict | None):
    return _num(_d(_d(r).get('memory')).get('max_peak_reserved_gib'))


def numerics(r: dict) -> dict:
    h = r.get('harness') if isinstance(r, dict) else None
    n = h.get('numerics') if isinstance(h, dict) else None
    return n if isinstance(n, dict) else {}


def _rejected(receipt, cfg: dict, name: str, detail: str) -> dict:
    """Result for a receipt that is not trusted at all: nothing else in it is read (no crash on forged structure)."""
    item = {'name': name, 'ok': False, 'detail': detail, 'skipped': False}
    checks = from_core([item], None)
    return {'schema': 'infrarsi-reward-v2', 'valid': False, 'R': -math.inf, 'components': None, 'metrics': {},
            'hard': [item], 'violations': [f'{name}: {detail}'], 'checks': checks, 'feedback': _checklist.feedback(checks),
            'attempt_id': receipt.get('attempt_id') if isinstance(receipt, dict) else None, 'parent_attempt_id': None,
            'numerics_class_allowed': cfg.get('numerics_class_allowed', 'E1')}


def is_reference_run(r: dict) -> bool:
    """A numerics reference only when the harness found no reference AND the run was launched as one: the role sits
    in the signed part of the receipt; a receipt without a role is not a reference."""
    return bool(numerics(r).get('is_reference')) and r.get('numerics_role') == 'reference'


AUX_IDENTITY = ('task_id', 'task_digest', 'code_digest', 'overlay_digest')


def _perf_env(r: dict) -> dict:
    """The whitelisted performance env the training process recorded (receipt env.env), unset entries dropped."""
    import fnmatch
    from infrarsi.harness.launch import ENV_DENY, ENV_WHITELIST
    env = ((r.get('env') or {}).get('env') or {}) if isinstance(r.get('env'), dict) else {}
    return {k: v for k, v in env.items() if v is not None and any(fnmatch.fnmatchcase(k, p) for p in ENV_WHITELIST)
            and not any(fnmatch.fnmatchcase(k, p) for p in ENV_DENY)}


def same_config(a: dict, b: dict) -> bool:
    """The same configuration ran: equal config_digest, or - when one side sent the planner's sparse config and the
    other the fully resolved one (the search's start candidate) - the same effective_config (schema defaults + config,
    the layout the trainer ran, checked against the request by the harness) and the same recorded performance env."""
    if a.get('config_digest') == b.get('config_digest'):
        return True
    ea, eb = a.get('effective_config'), b.get('effective_config')
    return isinstance(ea, dict) and bool(ea) and digest(ea) == digest(eb) and _perf_env(a) == _perf_env(b)


def aux_binding_problems(anchor: dict, aux: dict | None, *, purpose: str, plan: str, name: str) -> list[str]:
    """An auxiliary receipt (lifecycle, second input plan) must be harness-signed and belong to the candidate:
    same task, code and overlay digests, the same configuration (same_config), expected purpose and plan."""
    if not isinstance(aux, dict):
        return [f'{name} receipt missing']
    p = [] if verify_receipt_digest(aux) else [f'{name}: harness signature does not verify']
    for k in AUX_IDENTITY:
        if aux.get(k) != anchor.get(k):
            p.append(f'{name}: {k} {str(aux.get(k))[:16]} != candidate {str(anchor.get(k))[:16]}')
    if not same_config(anchor, aux):
        p.append(f'{name}: config_digest {str(aux.get("config_digest"))[:16]} != candidate '
                 f'{str(anchor.get("config_digest"))[:16]} and the effective configurations differ')
    if aux.get('purpose') != purpose or aux.get('input_plan') != plan:
        p.append(f'{name}: purpose/plan {aux.get("purpose")}/{aux.get("input_plan")} != {purpose}/{plan}')
    return p


def _task_raw(receipt: dict) -> dict | None:
    """The frozen task spec of a receipt when it still hashes to the receipt's task_digest (else None)."""
    try:
        raw = TaskSpec.load(receipt['task_id']).raw
    except Exception:                                   # noqa: BLE001 - unknown task id: identity is checked by digest
        return None
    return raw if digest(raw) == receipt.get('task_digest') else None


def receipt_tier(receipt: dict) -> str | None:
    """AMENDMENT-002 numerics tier of a receipt: from the model facts of its frozen task spec (the spec must still hash to
    the receipt's task_digest); the tier the harness signed into harness.numerics.tier must agree when present. None
    when the spec cannot be matched or the two disagree: the receipt's own fields are never the authority."""
    raw = _task_raw(receipt)
    if raw is None:
        return None
    tier = numerics_tier(raw)
    signed = numerics(receipt).get('tier')
    return None if signed is not None and signed != tier else tier


def parent_problems(receipt: dict, parent: dict, *, role: str | None = None) -> list[str]:
    """validate.reference_problems for a reward parent (any role) or breakeven reference (role 'reference'): same task,
    plan, purpose, steps and window as the candidate, signed, valid, complete and finite. A reference is also held to
    the mandatory invariant schema; a parent's invariants were enforced when the harness validated it (valid=True)."""
    return reference_problems(parent, task=_task_raw(receipt) if role == 'reference' else None,
                              task_id=receipt.get('task_id'), task_digest=receipt.get('task_digest'),
                              plan=receipt.get('input_plan'), steps=receipt.get('steps'), window=receipt.get('window'),
                              role=role, purpose=receipt.get('purpose'))


def metric_vector(r: dict) -> dict:
    st = r.get('step_time') or {}
    t = step_time(r)
    inv = r.get('invariants') or {}
    per = r.get('per_step') or []
    win = r.get('window')
    sup = [p.get('supervised_tokens') for p in per if win and win[0] <= p.get('update', p.get('step', 0)) <= win[1]]
    sup_mean = sum(sup) / len(sup) if sup and all(isinstance(x, int) for x in sup) else None
    num = numerics(r)
    h = r.get('harness') or {}
    ev = r.get('events') or {}
    lc = r.get('lifecycle') or {}
    return {
        'StepTime': t, 'StepTimeMedian': _num(st.get('median_window_max_rank_s')), 'RSD': _num(st.get('rsd_window')),
        'InterStep': inter_step(r),
        'InputTokensPerSecond': inv.get('physical_tokens_per_update') / t if t and inv.get('physical_tokens_per_update') else None,
        'SupervisedTokensPerSecond': sup_mean / t if t and sup_mean else None,
        'JobTime': r.get('phases') or {}, 'PeakMemory': {'max_reserved_gib': peak_reserved(r),
                                                          'max_alloc_gib': _num((r.get('memory') or {}).get('max_peak_alloc_gib'))},
        'AllocRetries': ev.get('alloc_retries'), 'AllocOOMs': ev.get('alloc_ooms'), 'NCCLErrors': ev.get('nccl_errors'),
        'NumericsClass': 'REF' if is_reference_run(r) else num.get('class'),
        'LossDeviation': dict({k: num.get(k) for k in ('main_ce_rel_max', 'main_ce_rel_mean_window', 'grad_norm_rel_max', 'aux_rel_max')},
                              grad_norm_first_update_rel=(num.get('grad_norm') or {}).get('first_update_rel'),
                              grad_norm_log_bias_window=(num.get('grad_norm') or {}).get('log_bias_window')),
        'Stability': {'complete': r.get('steps_completed') == r.get('steps'), 'nan_inf': ev.get('nan_inf_steps'),
                      'tail_step_ratio': (h.get('derived') or {}).get('tail_step_ratio')},
        'Invariants': {'violations': [c for c in h.get('checks', []) if not c['ok'] and c['severity'] == 'invalid'
                                      and c['name'].startswith(INVARIANT_PREFIXES)]},
        'Lifecycle': {k: lc.get(k) for k in ('resumed_ok', 'resume_scalars_match', 'export_ok', 'forward_ok')} if lc else None,
        'GPUHours': h.get('gpu_hours'), 'MFU': None,
    }


class _Hard:
    def __init__(self):
        self.items = []

    def add(self, name, ok, detail='', skipped=False):
        self.items.append({'name': name, 'ok': bool(ok), 'detail': detail, 'skipped': skipped})


def evaluate(receipt: dict, parent_receipt: dict | None = None, reward_cfg: dict | None = None, *,
             promotion: bool = False, lifecycle_receipt: dict | None = None, plan_b_receipt: dict | None = None,
             margin: float | None = None, gpu_type: str | None = None, search_gpuh: float | None = None,
             ref_receipt: dict | None = None) -> dict:
    base = reward_cfg or load_reward_cfg()
    tiered = any(is_tier_map(v) for v in (base.get('tolerance') or {}).values())
    cfg = resolve(base, gpu_type=gpu_type)
    hard_cfg = cfg['hard']
    if not verify_receipt_digest(receipt):          # HMAC over the whole receipt (harness/signing.py): read nothing else
        return _rejected(receipt, cfg, 'receipt_signature', 'receipt not validated by this harness, or altered')
    tier = receipt_tier(receipt)
    tol = tier_tolerance(cfg['tolerance'], tier)
    m = metric_vector(receipt)
    hd = _Hard()
    hd.add('receipt_signature', True, 'HMAC verifies')
    if tiered:                                       # per-tier gates need the receipt's tier
        hd.add('numerics_tier', tier is not None, 'numerics tier unresolvable: the frozen task spec does not hash to the '
               f'receipt task_digest, or harness.numerics.tier {numerics(receipt).get("tier")!r} disagrees with it')
    hd.add('receipt_valid', receipt.get('valid') is True, '; '.join(receipt.get('invalid_reasons') or [])[:500])
    hd.add('complete_updates', m['Stability']['complete'], f'{receipt.get("steps_completed")}/{receipt.get("steps")}')
    hd.add('step_time_derived', m['StepTime'] is not None and m['StepTime'] > 0,
           'no harness-derived StepTime (scoring window incomplete or signals unbound)')
    hd.add('invariants', not m['Invariants']['violations'], '; '.join(f'{c["name"]}: {c["detail"]}' for c in m['Invariants']['violations'])[:500])
    num = numerics(receipt)
    allowed = cfg.get('numerics_class_allowed', 'E1')
    cls = num.get('class')
    ref_run = is_reference_run(receipt)
    if ref_run:
        hd.add('numerics_class_matches_allowed', True, 'reference run (no comparison)')
        hd.add('loss_guard', True, 'reference run')
    else:
        if not num.get('compared'):
            why = '; '.join(str(x) for x in (num.get('violations') or [])) or 'no comparable reference'
            if num.get('is_reference'):
                why = f'reference flag on a {receipt.get("numerics_role")}-role run'
            hd.add('numerics_compared', False, f'numerics not verified: not compared against a reference ({why})')
        ok = cls in NUMERICS_ORDER and NUMERICS_ORDER[cls] <= NUMERICS_ORDER.get(allowed, 1)
        hd.add('numerics_class_matches_allowed', ok, f'class {cls} vs allowed {allowed}' if cls else 'numerics not verified')
        # Loss guard against the task's predeclared tolerance (AMENDMENT-001/002 gates); missing evidence fails.
        viol = [str(x) for x in (num.get('violations') or [])]
        if cls == 'E1' and allowed == 'E0':
            viol.append('E0 (bitwise) required')
        if cls and cls != 'E0':
            gn = num.get('grad_norm') if isinstance(num.get('grad_norm'), dict) else {}
            bias = _num(gn.get('log_bias_window'))
            for name, val, key in (('main_ce_rel_max', _num(num.get('main_ce_rel_max')), 'main_ce_rel_max_per_step'),
                                   ('main_ce_rel_mean_window', _num(num.get('main_ce_rel_mean_window')), 'main_ce_rel_mean_over_window'),
                                   ('grad_norm.first_update_rel', _num(gn.get('first_update_rel')), 'grad_norm_rel_max_first_update'),
                                   ('|grad_norm.log_bias_window|', None if bias is None else abs(bias), 'grad_norm_log_bias_window_max')):
                lim = _tol(tol, key)
                if val is None or val > lim:
                    viol.append(f'{name} {val if val is None else f"{val:.3g}"} > {lim}' if val is not None else f'{name} missing')
        hd.add('loss_guard', bool(cls) and not viol, '; '.join(viol) if viol else ('' if cls else 'numerics not verified'))
    repeated = (numerics(receipt).get('self_check') or {}).get('repeated_pairs')
    hd.add('repeated_adjacent_scalars', not repeated,
           f'adjacent updates with bitwise-equal (main_ce, grad_norm): {(repeated or [])[:8]}')
    t, tp = m['StepTime'], step_time(parent_receipt)
    pp = parent_problems(receipt, parent_receipt) if parent_receipt is not None else []
    if parent_receipt is not None:
        hd.add('parent_receipt', not pp, '; '.join(pp[:6]))
    if parent_receipt is not None and not pp:                     # an untrusted parent is not read any further
        i, ip = m['InterStep'], inter_step(parent_receipt)
        if None in (t, tp, i, ip) or 0 in (t, tp, i, ip):
            hd.add('inter_step_consistency', False, f'required times missing (StepTime {t}, parent {tp}; InterStep {i}, '
                   f'parent {ip}): the timing evidence is incomplete')
        else:
            d = abs((1 - i / ip) - (1 - t / tp))
            hd.add('inter_step_consistency', d <= hard_cfg['inter_step_consistency'],
                   f'|dInterStep - dStepTime| = {d:.4f} (> {hard_cfg["inter_step_consistency"]} means work moved outside the timer)')
        fz, fzp = (receipt.get('invariants') or {}).get('frozen_zone_digest'), (parent_receipt.get('invariants') or {}).get('frozen_zone_digest')
        hd.add('frozen_zone_hash_unchanged', fz is not None and fz == fzp, f'{fz} vs parent {fzp}')
    pr = m['PeakMemory']['max_reserved_gib']
    hd.add('peak_reserved_gib_max', pr is not None and pr <= hard_cfg['peak_reserved_gib_max'],
           f'{pr} GiB > {hard_cfg["peak_reserved_gib_max"]}' if pr is not None else 'peak memory missing')
    hd.add('alloc_ooms_max', (m['AllocOOMs'] or 0) <= hard_cfg['alloc_ooms_max'], f'{m["AllocOOMs"]} allocator OOMs')
    hd.add('nccl_errors_max', (m['NCCLErrors'] or 0) <= hard_cfg['nccl_errors_max'], f'{m["NCCLErrors"]} NCCL errors')
    stab = hard_cfg['stability']
    tail = m['Stability']['tail_step_ratio']
    hd.add('stability', (m['Stability']['nan_inf'] or 0) <= stab['nan_inf'] and (tail is None or tail <= stab['tail_step_ratio_max']),
           f'nan_inf {m["Stability"]["nan_inf"]}, tail ratio {tail}')
    if promotion:
        # The auxiliary receipts must be signed and bound to this candidate (A finding 11 / item 9).
        lc_bind = aux_binding_problems(receipt, lifecycle_receipt, purpose='lifecycle', plan='B', name='lifecycle')
        lc = _d(_d(lifecycle_receipt).get('lifecycle'))
        flags = [k for k in ('resumed_ok', 'resume_scalars_match', 'export_ok', 'forward_ok') if lc.get(k) is not True]
        lc_ok = not lc_bind and _d(lifecycle_receipt).get('valid') is True and bool(lc) and not flags
        hd.add('lifecycle_for_promotion', lc_ok, '; '.join(lc_bind + ([f'lifecycle flags not true: {flags}'] if flags else [])
                                                           + ([] if _d(lifecycle_receipt).get('valid') is True
                                                              else ['lifecycle receipt not valid']))[:500])
        b_bind = aux_binding_problems(receipt, plan_b_receipt, purpose='score', plan='B', name='plan B')
        if isinstance(lifecycle_receipt, dict) and isinstance(plan_b_receipt, dict) and                 lifecycle_receipt.get('config_digest') != plan_b_receipt.get('config_digest'):
            b_bind.append('the lifecycle and plan-B receipts ran different configurations')
        if receipt.get('input_plan') != 'A':
            b_bind.append(f'the promoted candidate receipt is on plan {receipt.get("input_plan")!r}, not A')
        if b_bind:
            hd.add('second_input_plan_consistency', False, '; '.join(b_bind)[:500])
        else:
            tb, mg = step_time(plan_b_receipt), margin if margin is not None else 0.03
            inv_a, inv_b = receipt.get('invariants') or {}, plan_b_receipt.get('invariants') or {}
            keys = ('gbs_packs', 'physical_tokens_per_update', 'trainable_params', 'trainable_param_digest', 'loss_definition_digest',
                    'router_topk', 'capacity_factor', 'num_experts', 'frozen_zone_digest', 'model_config_digest', 'dtype_census')
            diff = [k for k in keys if inv_a.get(k) != inv_b.get(k)]
            ok = plan_b_receipt.get('valid') is True and t and tb and abs(math.log(tb / t)) <= math.log(1 + mg) and not diff
            hd.add('second_input_plan_consistency', ok, f'plan B {tb} vs plan A {t} (margin {mg}); invariant diffs {diff}')
    if ref_receipt is not None:
        rp = parent_problems(receipt, ref_receipt, role='reference')
        hd.add('reference_receipt', not rp, '; '.join(rp[:6]))
    violations = [x for x in hd.items if not x['ok']]
    valid = not violations
    comp = None
    R = None
    if not valid:
        R = -math.inf
    elif parent_receipt is not None:
        g_t = math.log(tp / t) if t and tp else 0.0
        R, comp = g_t, {'g_t': g_t}
    checks = from_core(hd.items, receipt)
    out = {'schema': 'infrarsi-reward-v2', 'valid': valid, 'R': R, 'components': comp, 'metrics': m, 'hard': hd.items,
           'violations': [f'{x["name"]}: {x["detail"]}' for x in violations], 'checks': checks,
           'feedback': _checklist.feedback(checks), 'coverage': _checklist.coverage(checks), 'attempt_id': receipt.get('attempt_id'),
           'parent_attempt_id': (parent_receipt or {}).get('attempt_id'), 'numerics_class_allowed': allowed}
    if search_gpuh is not None and (ref_receipt or parent_receipt) and valid:
        out['breakeven'] = breakeven_from_receipts(search_gpuh, ref_receipt or parent_receipt, receipt,
                                                   cfg.get('planned_training_updates'))
    return out


def lenient(receipt: dict, parent_receipt: dict | None = None) -> dict:
    """E4 'lenient reward': StepTime and completion only (the red-team contrast); never used for formal tables."""
    t, tp = step_time(receipt), step_time(parent_receipt)
    ok = receipt.get('steps_completed') == receipt.get('steps') and t is not None
    return {'schema': 'infrarsi-reward-lenient-v1', 'valid': ok, 'R': (math.log(tp / t) if ok and tp else (None if ok else -math.inf))}
