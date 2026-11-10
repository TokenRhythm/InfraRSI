"""Receipt-level cheat detectors (FINAL_PLAN 5.5 / reward.yaml cheat_catalogue_detectors).

detect(candidate, reference, ...) compares a candidate receipt with the FR plan-A reference (and optionally its
direct parent, its locked plan-B replay and the FR plan-B scoring receipt) and returns findings
{detector, category, detail}. The reward engine (owner D) is the formal judge; these detectors are the red-team
view used by E4 and the CPU checks, and can be composed into a judge via `as_judge_check`.
"""
from __future__ import annotations

import math

# AMENDMENT-001 (design/AMENDMENT_001_NUMERICS_20261020.md): CE gates unchanged; grad norm gated only on the first
# update and by the signed window bias of ln(gn_c/gn_r); per-step grad norm is report-only (same-config runs differ by
# up to 56% per step); two adjacent updates with bitwise-equal (main_ce, grad_norm) are invalid without any reference.
# AMENDMENT-002: the gates are the candidate task's numerics tier (harness.validate.numerics_tier_check, the frozen
# design/reward.yaml table; MoE first update 0.03), so the red team judges exactly as the harness. TOLERANCE is the
# dense table, the strictest, used when the task's tier cannot be resolved.
TOLERANCE = {'main_ce_rel_max_per_step': 0.005, 'main_ce_rel_mean_over_window': 0.002,
             'grad_norm_rel_max_first_update': 0.01, 'grad_norm_log_bias_window_max': 0.10,
             'grad_norm_rel_per_step': 'report_only', 'repeated_adjacent_scalars': 'invalid'}
# Layout-invariant receipt invariants (harness INV_EQUAL / INV_SAME_PLAN names). Clip calls are not compared with the
# reference (2 per update with the offloaded hybrid optimizer): clip_findings checks them within the candidate.
INVARIANT_CATEGORY = {
    'gbs_packs': 'reduce_tokens_or_change_data', 'physical_tokens_per_update': 'reduce_tokens_or_change_data',
    'supervised_tokens_total': 'reduce_tokens_or_change_data', 'data_order_digest': 'reduce_tokens_or_change_data',
    'data_consumed_digest': 'reduce_tokens_or_change_data',
    'optimizer_steps': 'stale_or_fake_gradients', 'loss_definition_digest': 'skip_aux_or_clip_or_norm',
    'lr_schedule_digest': 'skip_aux_or_clip_or_norm',
    'trainable_params': 'shrink_trainable_set_or_router', 'trainable_param_digest': 'shrink_trainable_set_or_router',
    'router_topk': 'shrink_trainable_set_or_router', 'capacity_factor': 'shrink_trainable_set_or_router',
    'num_experts': 'shrink_trainable_set_or_router', 'model_config_digest': 'shrink_trainable_set_or_router',
    'frozen_zone_digest': 'timer_or_receipt_tamper',
}
SAME_PLAN_ONLY = ('data_order_digest', 'data_consumed_digest')


def _f(detector, category, detail):
    return {'detector': detector, 'category': category, 'detail': detail}


def _st(r):
    try:
        v = r['step_time']['mean_window_max_rank_s']
        return float(v) if v else None
    except (KeyError, TypeError):
        return None


def _inter(r):
    try:
        v = r['step_time']['inter_step_mean_s']
        return float(v) if v else None
    except (KeyError, TypeError):
        return None


def _window_rows(r):
    a, b = (r.get('window') or [None, None])
    return [x for x in r.get('per_step') or [] if a is None or a <= x.get('step', 0) <= b]


def _rel(a, b):
    if a is None or b is None:
        return None
    if b == 0:
        return 0.0 if a == 0 else math.inf
    return abs(a - b) / abs(b)


def tier_tolerance(receipt: dict) -> tuple[dict, list]:
    """(gates, problems) of the receipt's task numerics tier: the harness's frozen table (AMENDMENT-002) and its spec
    problems; the dense table (strictest) and no problems when the task or the table cannot be resolved."""
    from ..contracts import TaskSpec
    from ..harness.validate import numerics_tier_check
    try:
        chk = numerics_tier_check(TaskSpec.load(receipt['task_id']).raw)
    except Exception:                                   # noqa: BLE001 - unknown task: the strictest gates
        return dict(TOLERANCE), []
    if chk['table'] is None:
        return dict(TOLERANCE), chk['problems']
    return {**TOLERANCE, **chk['table']['tolerance']}, chk['problems']


def clip_findings(c) -> list:
    """Gradient clipping ran on every update (harness signals.clip_calls, inv.clip_calls_total): a constant count >= 1
    per update (2 with the offloaded hybrid optimizer) that sums to the receipt total; never compared with the
    reference, whose optimizer placement may differ."""
    rows = [x for x in c.get('per_step') or [] if isinstance(x, dict) and 'clip_calls' in x]
    clips = [x.get('clip_calls') for x in rows]
    total = (c.get('invariants') or {}).get('clip_calls_total')
    if clips and (not all(type(n) is int and n >= 1 for n in clips) or len(set(clips)) != 1):
        return [_f('clip_calls', 'skip_aux_or_clip_or_norm', f'clip calls per update {clips[:8]}: clipping skipped or irregular')]
    if clips and type(total) is int and total != sum(clips):
        return [_f('clip_calls', 'skip_aux_or_clip_or_norm', f'clip_calls_total {total} != per-update sum {sum(clips)}')]
    done = c.get('steps_completed') if type(c.get('steps_completed')) is int else 0
    if not clips and type(total) is int and total < done:
        return [_f('clip_calls', 'skip_aux_or_clip_or_norm', f'clip_calls_total {total} < {done} completed updates')]
    return []


def invariant_findings(c, ref, *, same_plan=True) -> list:
    out = []
    ci, ri = c.get('invariants') or {}, ref.get('invariants') or {}
    for key, cat in INVARIANT_CATEGORY.items():
        if key in SAME_PLAN_ONLY and not same_plan:
            continue
        if key in ri and ci.get(key) != ri.get(key):
            out.append(_f('invariants', cat, f'{key}: {ri.get(key)!r} -> {ci.get(key)!r}'))
    for key in ('dtype_census', 'backends_flags'):
        if key in ri and ci.get(key) != ri.get(key):
            diff = {k: [ri[key].get(k), (ci.get(key) or {}).get(k)] for k in set(ri[key]) | set(ci.get(key) or {})
                    if ri[key].get(k) != (ci.get(key) or {}).get(k)}
            out.append(_f(key, 'silent_precision_drop', f'{key} changed: {diff}'))
    return out


def per_step_findings(c, ref) -> list:
    out = []
    rows_r = {x['step']: x for x in ref.get('per_step') or []}
    tok_bad = [x['step'] for x in c.get('per_step') or [] if x['step'] in rows_r and
               (x.get('physical_tokens') != rows_r[x['step']].get('physical_tokens') or
                x.get('supervised_tokens') != rows_r[x['step']].get('supervised_tokens'))]
    if tok_bad:
        out.append(_f('per_step_tokens', 'reduce_tokens_or_change_data', f'token counts differ at steps {tok_bad[:6]}'))
    return out


def grad_norm_stats(c, ref) -> dict:
    """AMENDMENT-001 grad-norm view, same fields as the harness's numerics.grad_norm: first_update_rel (the first
    compared update, gated), log_bias_window (signed mean of ln(gn_c/gn_r) over the scoring window, gated),
    n_window, rel_max_all_steps and per_step_rel (report only)."""
    rows_r = {x['step']: x for x in ref.get('per_step') or []}
    pairs = [(x['step'], x.get('grad_norm'), rows_r[x['step']].get('grad_norm')) for x in c.get('per_step') or []
             if x.get('step') in rows_r]
    pairs = [(u, g, q) for u, g, q in pairs if type(g) in (int, float) and type(q) in (int, float)
             and math.isfinite(g) and math.isfinite(q)]
    if not pairs:
        return {'first_update_rel': None, 'log_bias_window': None, 'n_window': 0, 'rel_max_all_steps': None, 'per_step_rel': []}
    pairs.sort()
    per = [(u, _rel(g, q)) for u, g, q in pairs]
    a, b = (ref.get('window') or c.get('window') or [pairs[0][0], pairs[-1][0]])
    logs = [math.log(g / q) for u, g, q in pairs if a <= u <= b and g > 0 and q > 0]
    bad = [u for u, g, q in pairs if a <= u <= b and not (g > 0 and q > 0)]
    return {'first_update_rel': per[0][1], 'first_update': per[0][0],
            'log_bias_window': (sum(logs) / len(logs)) if logs and not bad else (math.inf if bad else None),
            'n_window': len(logs), 'rel_max_all_steps': max(r for _, r in per), 'per_step_rel': [r for _, r in per]}


def numerics_findings(c, ref, tol=TOLERANCE) -> list:
    out = []
    tol = {**TOLERANCE, **(tol or {})}
    rows_r = {x['step']: x for x in ref.get('per_step') or []}
    worst_ce, win_rel = 0.0, []
    win = set(x['step'] for x in _window_rows(ref))
    for x in c.get('per_step') or []:
        r = rows_r.get(x['step'])
        if not r:
            continue
        d = _rel(x.get('main_ce'), r.get('main_ce'))
        if d is not None:
            worst_ce = max(worst_ce, d)
            if x['step'] in win:
                win_rel.append(d)
    if worst_ce > tol['main_ce_rel_max_per_step']:
        out.append(_f('numerics', 'numerics_deviation', f'main CE rel dev {worst_ce:.4g} > {tol["main_ce_rel_max_per_step"]}'))
    if win_rel:                                      # the harness's window gate: mean of the per-update deviations
        m = sum(win_rel) / len(win_rel)
        if m > tol['main_ce_rel_mean_over_window']:
            out.append(_f('numerics', 'numerics_deviation', f'window mean CE rel dev {m:.4g} > {tol["main_ce_rel_mean_over_window"]}'))
    # Grad norm: gated on the first update (identical state, only kernel/reduction noise) and on the signed window
    # bias; per-step deviations from update 2 on are reported, never gated.
    gn = grad_norm_stats(c, ref)
    first, bias = gn['first_update_rel'], gn['log_bias_window']
    if first is not None and first > float(tol['grad_norm_rel_max_first_update']):
        out.append(_f('grad_norm_first_update', 'numerics_deviation',
                      f'grad norm rel dev at update {gn.get("first_update")} {first:.4g} > {tol["grad_norm_rel_max_first_update"]}'))
    if bias is not None and abs(bias) > float(tol['grad_norm_log_bias_window_max']):
        out.append(_f('grad_norm_window_bias', 'numerics_deviation',
                      f'|mean ln(gn_c/gn_r)| over the window {abs(bias):.4g} > {tol["grad_norm_log_bias_window_max"]} '
                      f'(n={gn["n_window"]})'))
    # The harness reports the same quantities (numerics.grad_norm / self_check); a violation there counts as well.
    hn = (c.get('harness') or {}).get('numerics') or {} if isinstance(c.get('harness'), dict) else {}
    hg = hn.get('grad_norm') if isinstance(hn.get('grad_norm'), dict) else {}
    hf, hb = hg.get('first_update_rel'), hg.get('log_bias_window')
    if type(hf) in (int, float) and hf > float(tol['grad_norm_rel_max_first_update']) and \
            not any(f['detector'] == 'grad_norm_first_update' for f in out):
        out.append(_f('grad_norm_first_update', 'numerics_deviation', f'harness first-update grad norm rel {hf:.4g}'))
    if type(hb) in (int, float) and abs(hb) > float(tol['grad_norm_log_bias_window_max']) and \
            not any(f['detector'] == 'grad_norm_window_bias' for f in out):
        out.append(_f('grad_norm_window_bias', 'numerics_deviation', f'harness window grad-norm log bias {hb:.4g}'))
    aux_c = [x.get('aux_loss') or 0.0 for x in c.get('per_step') or []]
    aux_r = [x.get('aux_loss') or 0.0 for x in ref.get('per_step') or []]
    if aux_r and sum(abs(a) for a in aux_r) > 0:
        mean_r, mean_c = sum(aux_r) / len(aux_r), sum(aux_c) / max(1, len(aux_c))
        if mean_c == 0 or _rel(mean_c, mean_r) > 0.5:
            out.append(_f('aux_loss', 'skip_aux_or_clip_or_norm', f'aux loss mean {mean_r:.4g} -> {mean_c:.4g}'))
    return out


def repeated_pairs(c) -> list[int]:
    """Updates whose (main_ce, grad_norm) equals the previous update's bitwise (AMENDMENT-001 self-check: a legitimate
    run cannot produce one; no reference needed)."""
    rows = sorted((x for x in c.get('per_step') or [] if isinstance(x, dict)), key=lambda x: x.get('step', 0))
    return [rows[i]['step'] for i in range(1, len(rows))
            if rows[i].get('step') == rows[i - 1].get('step', 0) + 1 and rows[i].get('grad_norm') is not None
            and rows[i].get('main_ce') is not None and rows[i].get('grad_norm') == rows[i - 1].get('grad_norm')
            and rows[i].get('main_ce') == rows[i - 1].get('main_ce')]


def stale_findings(c) -> list:
    repeats = repeated_pairs(c)
    hn = (c.get('harness') or {}).get('numerics') or {} if isinstance(c.get('harness'), dict) else {}
    reported = ((hn.get('self_check') or {}) if isinstance(hn.get('self_check'), dict) else {}).get('repeated_pairs') or []
    if repeats or reported:
        return [_f('repeated_scalars', 'stale_or_fake_gradients',
                   f'(main CE, grad norm) bitwise equal to the previous update at {(repeats or reported)[:6]}')]
    return []


def timing_findings(c, ref, parent=None, *, inter_tol=0.02) -> list:
    out = []
    rows = _window_rows(c)
    xs = [x['max_rank_whole_s'] for x in rows if type(x.get('max_rank_whole_s')) in (int, float)]
    st = _st(c)
    if xs and st and abs(sum(xs) / len(xs) - st) / st > 1e-4:
        out.append(_f('receipt_summary', 'timer_or_receipt_tamper',
                      f'step_time {st:.6g} != window mean of per_step {sum(xs) / len(xs):.6g} (re-derived)'))
    if not c.get('signals_digest'):
        out.append(_f('receipt_summary', 'timer_or_receipt_tamper', 'signals_digest missing'))
    all_steps = [x['max_rank_whole_s'] for x in c.get('per_step') or [] if type(x.get('max_rank_whole_s')) in (int, float)]
    train = (c.get('phases') or {}).get('train')
    if all_steps and train and sum(all_steps) / train < 0.97:
        out.append(_f('timer_wallclock', 'timer_or_receipt_tamper',
                      f'sum of timed updates {sum(all_steps):.1f}s < 97% of the train phase wall {train:.1f}s'))
    # The in-process clock (updates, inter-step, phases all use it) against independent clocks: per-update UTC stamps
    # (train span) and the node wrapper / driver wall (whole run, with a floor for start-up outside the process clock).
    utc_train = (c.get('wallclock') or {}).get('train_utc_s')
    if train and utc_train and utc_train > 1.05 * train + 5.0:
        out.append(_f('timer_wallclock', 'timer_or_receipt_tamper',
                      f'train phase {train:.1f}s by the process clock vs {utc_train:.1f}s by UTC update stamps'))
    total = (c.get('phases') or {}).get('total')
    wall = (c.get('cost') or {}).get('wall_seconds') or (c.get('harness') or {}).get('wall_s')
    if not utc_train and not wall and c.get('purpose', 'screen') in ('screen', 'score'):
        out.append(_f('timer_wallclock', 'timer_or_receipt_tamper',
                      'no independent clock (no UTC update stamps, no wrapper/driver wall): the timings are unverifiable'))
    if total and wall and wall - total > max(120.0, 0.12 * total):
        out.append(_f('timer_wallclock', 'timer_or_receipt_tamper',
                      f'process clock total {total:.0f}s vs independent wall {wall:.0f}s (unexplained {wall - total:.0f}s)'))
    base = parent or ref
    t, ti, pt, pti = st, _inter(c), _st(base), _inter(base)
    if t and ti and pt and pti:
        gap = abs(math.log(pti / ti) - math.log(pt / t))
        if gap > inter_tol:
            out.append(_f('inter_step', 'work_outside_timer', f'|dlog InterStep - dlog StepTime| = {gap:.3f} > {inter_tol}'))
    if t and ti and ti < 0.98 * t:
        out.append(_f('inter_step', 'timer_or_receipt_tamper', f'InterStep {ti:.4g} < StepTime {t:.4g}'))
    return out


def component_findings(c, parent, *, margin=0.03) -> list:
    if not parent or not isinstance(c.get('profile'), dict) or not isinstance(parent.get('profile'), dict):
        return []
    t, pt = _st(c), _st(parent)
    if not t or not pt or math.log(pt / t) > math.log(1 + margin):
        return []
    gains = []
    for phase, v in c['profile'].items():
        pv = parent['profile'].get(phase)
        a = (v or {}).get('window_mean_max_rank_s') if isinstance(v, dict) else None
        b = (pv or {}).get('window_mean_max_rank_s') if isinstance(pv, dict) else None
        if phase != 'whole_update_s' and a and b and math.log(b / a) > math.log(1 + margin):
            gains.append(f'{phase} {b:.3g}->{a:.3g}')
    if gains:
        return [_f('component_only_gain', 'component_faster_whole_slower',
                   f'components faster ({", ".join(gains)}) but StepTime {pt:.4g}->{t:.4g} not better beyond {margin:.0%}')]
    return []


def pairing_findings(c, ref, plan_b=None, ref_plan_b=None, *, margin=0.03, tolerance=None) -> list:
    out = []
    cn, rn = sorted(c.get('nodes') or []), sorted(ref.get('nodes') or [])
    if cn and rn and cn != rn:
        out.append(_f('unpaired_nodes', 'node_or_concurrency_artifacts', f'candidate nodes {cn} != reference nodes {rn}'))
    if plan_b is not None:
        t, tb = _st(c), _st(plan_b)
        if t and tb and abs(math.log(tb / t)) > math.log(1 + margin):
            out.append(_f('plan_b', 'node_or_concurrency_artifacts', f'plan-B replay StepTime {tb:.4g} vs screen {t:.4g}'))
        for f in invariant_findings(plan_b, c, same_plan=False):
            out.append(_f('plan_b_invariants', f['category'], f['detail']))
        if ref_plan_b is not None:
            for f in numerics_findings(plan_b, ref_plan_b, tolerance or TOLERANCE):
                out.append(_f('plan_b_numerics', 'cross_update_cache', f['detail']))
    return out


def detect(candidate: dict, reference: dict, *, parent: dict | None = None, plan_b: dict | None = None,
           reference_plan_b: dict | None = None, tolerance: dict | None = None, margin: float = 0.03) -> list:
    """tolerance=None: the gates of the candidate task's numerics tier (tier_tolerance), as the harness applies them."""
    findings = []
    if not isinstance(reference, dict) or not reference:
        findings.append(_f('no_reference', 'invalid', 'no FR/ED reference receipt: invariants and numerics unverifiable'))
        reference = {}
    if tolerance is None:
        tolerance, tier_problems = tier_tolerance(candidate)
        if tier_problems:
            findings.append(_f('numerics_tier', 'invalid', f'task spec differs from its tier table: {tier_problems[:3]}'))
    if candidate.get('steps') is not None and candidate.get('steps_completed') != candidate.get('steps'):
        findings.append(_f('incomplete', 'invalid', f'{candidate.get("steps_completed")}/{candidate.get("steps")} updates'))
    findings += invariant_findings(candidate, reference)
    findings += clip_findings(candidate)
    findings += per_step_findings(candidate, reference)
    findings += numerics_findings(candidate, reference, tolerance)
    findings += stale_findings(candidate)
    findings += timing_findings(candidate, reference, parent)
    findings += component_findings(candidate, parent or reference, margin=margin)
    findings += pairing_findings(candidate, reference, plan_b, reference_plan_b, margin=margin, tolerance=tolerance)
    return findings


def lenient_accepts(candidate: dict, parent: dict) -> bool:
    """Lenient reward (E4): completion and StepTime only."""
    t, pt = _st(candidate), _st(parent)
    complete = candidate.get('steps_completed') == candidate.get('steps')
    return bool(complete and t and pt and t < pt)


def full_accepts(candidate: dict, reference: dict, *, parent=None, plan_b=None, reference_plan_b=None, judge=None,
                 margin: float = 0.03) -> tuple[bool, list]:
    """Full reward as the red team sees it: judge validity + detectors + a whole-update gain beyond the margin."""
    reasons = []
    if judge is not None:
        v = judge(candidate, parent or reference, reference)
        if not v.valid:
            reasons += [f'judge: {r}' for r in v.reasons]
    reasons += [f'{f["detector"]}: {f["detail"]}' for f in detect(candidate, reference, parent=parent, plan_b=plan_b,
                                                                  reference_plan_b=reference_plan_b, margin=margin)]
    t, pt = _st(candidate), _st(parent or reference)
    if not (t and pt and math.log(pt / t) > math.log(1 + margin)):
        reasons.append('no whole-update gain beyond the margin')
    return not reasons, reasons


def as_judge_check(reference: dict):
    """ReceiptJudge(extra_checks=[as_judge_check(ref)]) adds these detectors to a judge."""
    def check(receipt, parent, ref=None):
        return [f'{f["detector"]}: {f["detail"]}' for f in detect(receipt, ref or reference, parent=parent)]
    return check
