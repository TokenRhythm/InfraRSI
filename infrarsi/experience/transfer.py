"""The LLM transfer judgment (experience decision, section 3): once per run, after the reference run, when an LLM is
configured and some knob is covered only partially or not at all (or no start composite came from the data). The model
sees the target, the knobs with their coverage, the nearest environments with their computed differences, the effects,
constraints and rules in scope and its earlier judgments for these GPU classes; it returns per knob the sign it expects on
the target and whether the change runs at all, and at most a four-knob start configuration. Only the sign, the feasibility
and the start configuration are used (sizes come from the data; the model's numbers are recorded and scored, T7). The
run's measurements score the judgment afterwards: sign accuracy over the judged moves the run resolves, the Brier score of
the feasibility, the start composite's gain.

V0 (126 H20 -> H200 pairs, Opus 5.5, hw/V0/): the model caught 4 of 14 sign flips (pre-registered: >= 7) and flagged 6 of 97
agreements, so, as pre-registered, a judged sign is taken only when it agrees with the data's or says "unknown"; a "0" or
an opposite sign keeps the data's sign. "infeasible" still makes the move probe-first."""
from __future__ import annotations

import json

from .registry import gpus

SIGNS = ('+', '-', '0', 'infeasible', 'unknown')
TRANSFER_SCHEMA = {'type': 'object', 'required': ['assessment'], 'properties': {
    'assessment': {'type': 'array', 'items': {'type': 'object', 'required': ['knob', 'change', 'sign'], 'properties': {
        'knob': {'type': 'string'}, 'change': {}, 'mechanism': {'type': ['string', 'null']},
        'sign': {'enum': list(SIGNS)}, 'flip': {'type': 'boolean'}, 'p_feasible': {'type': ['number', 'null']},
        'expected_gain_pct': {'type': ['number', 'null']}, 'why': {'type': 'string'}, 'evidence_ids': {'type': 'array'}}}},
    'start_config': {'type': 'object'}, 'memory_note': {'type': 'string'}}}

TRANSFER_SYSTEM = """You judge how measured training-speed effects transfer to a new environment, for InfraRSI, which
makes one existing training job faster. The input has the target environment; its knobs, each with the mechanism it sets,
its reference value and what the experience store knows about it (covered: measured in a matching environment; null:
measured there within noise; partial: measured only in other environments or together with another change; none: never
measured); the nearest measured environments with their differences from the target (computed, not judged); the effects,
constraints (known failures) and rules in scope; and earlier judgments for these GPU classes with their scores.
For each knob that is partial or none (at most 12, the most promising first) judge the change from its reference value
on the target (change: {"<knob>": <its new value>}): sign "+" faster, "-" slower, "0" no measurable change, "infeasible" it will not run or will be rejected
(memory, crash, loss), "unknown"; flip true when your sign differs from the nearest evidence; p_feasible, the probability
that it runs at all; expected_gain_pct (recorded and scored, not used to order the search); why (at most 200 characters:
which difference matters); the evidence_ids you relied on. start_config: at most 4 knob values that differ from the
reference, each judged "+", that you expect to compose ({} when none). memory_note: one sentence on memory headroom.
Reply with exactly one JSON object: {"assessment":[{"knob","change","mechanism","sign","flip","p_feasible",
"expected_gain_pct","why","evidence_ids"}],"start_config":{},"memory_note":""}."""


def needed(plan: dict) -> bool:
    cov = plan.get('coverage') or {}
    return any(c.get('mechanism') and c['level'] in ('partial', 'none') for c in cov.values()) or not plan.get('composite')


def request(plan: dict, knobs: dict, reference: dict, judgments: list | None = None) -> dict:
    cov, pl = plan['coverage'], plan['payload']
    near = {n.get('gpu') for n in pl.get('nearest') or []} | {plan['fingerprint'].get('gpu')}
    return {**{k: pl[k] for k in ('target', 'nearest', 'effects', 'constraints', 'rules')},
            'knobs': [{'knob': n, 'mechanism': c.get('mechanism'), 'class': c.get('class'), 'reference': reference.get(n),
                       'choices': (knobs.get(n) or {}).get('choices') or ([False, True] if (knobs.get(n) or {}).get('type') == 'bool' else None),
                       'level': c['level'], **({'gain_pct': round(100 * c['gain'], 2), 'value': c.get('value')}
                                               if c.get('gain') is not None else {})} for n, c in cov.items()],
            'noise': {'store_noise_pct_this_gpu': plan.get('noise_pct')},
            'gpu_classes': {g: gpus().get(g) for g in near if g in gpus()},
            'earlier_judgments': (judgments or [])[-10:]}


def _choice(knob: dict, v):
    """The knob's own value for what the model wrote ('2' for 2, "true" for True); None when it is not a choice."""
    opts = list(knob.get('choices') or ([False, True] if knob.get('type') == 'bool' else []))
    return next((c for c in opts if str(c).lower() == str(v).lower()), None)


def apply(plan: dict, out: dict, knobs: dict, reference: dict) -> dict:
    """Merge a judgment into the plan: a partial seed row takes the sign and the feasibility; an 'infeasible' move
    without a seed row becomes a warning of the plan's check; the start configuration becomes the composite when the
    data gave none (members: knobs judged '+' whose value differs from the reference). Returns the judgment record."""
    seeds = {s['knob']: s for s in plan['seeds']}
    record, judged, bad = [], {}, []
    for a in (out.get('assessment') or [])[:12]:
        n, ch = a.get('knob'), a.get('change')
        if isinstance(ch, list) and len(ch) == 2:      # [reference, new], the form of the input's effects (D1-15, Opus)
            ch = {n: ch[1]}
        elif not isinstance(ch, (dict, list)):         # the new value itself (R2-03, Qwen3-Next: "change": 2)
            ch = {n: ch}
        if n not in knobs or not isinstance(ch, dict) or set(ch) != {n} or a.get('sign') not in SIGNS:
            continue
        v = _choice(knobs[n], ch[n])
        if v is None or v == reference.get(n):
            continue
        ch = {n: v}
        p = float(a.get('p_feasible', 0.7)) if isinstance(a.get('p_feasible'), (int, float)) else 0.7
        row = {'knob': n, 'change': ch, 'sign': a['sign'], 'p_feasible': round(min(0.9, max(0.1, p)), 3),
               'flip': bool(a.get('flip')), 'why': str(a.get('why') or '')[:200],
               'expected_gain_pct': a.get('expected_gain_pct') if isinstance(a.get('expected_gain_pct'), (int, float)) else None,
               'level': plan['coverage'].get(n, {}).get('level')}
        record.append(row)
        judged[n] = row
        s = seeds.get(n)
        same = s is not None and s['change'] == ch
        if same and s['level'] == 'partial':
            data = '+' if s['gain'] > 0 else '-'
            keep = row['sign'] in (data, 'unknown', 'infeasible')     # V0: the model's flips were mostly wrong
            s.update(p_feasible=row['p_feasible'], why=row['why'], llm_transfer_mu=row['expected_gain_pct'],
                     **({'sign': row['sign']} if keep else {}))
            row['used'] = keep
        elif not same and row['sign'] == 'infeasible':
            bad.append(ch)
    if bad:
        base = plan['check']
        plan['check'] = lambda cfg: base(cfg) + [f'transfer: {k}={v} judged infeasible' for ch in bad for k, v in ch.items()
                                                 if cfg.get(k) == v and reference.get(k) != v]
    start = {n: judged[n]['change'][n] for n, v in (out.get('start_config') or {}).items()
             if n in judged and judged[n]['sign'] == '+' and _choice(knobs[n], v) == judged[n]['change'][n]
             and (n not in seeds or seeds[n]['change'] == judged[n]['change'])}   # the bundle runs the seed's value
    if plan.get('composite') is None and len(start) >= 2:
        for n, v in list(start.items())[:4]:
            if n not in seeds:                          # a judged move without data: a zero-size seed the bundle can hold
                seeds[n] = {'knob': n, 'mechanism': plan['coverage'][n].get('mechanism') or n, 'change': {n: v},
                            'level': 'partial', 'gain': 0.0, 'evidence_ids': [], 'warnings': [], 'sign': '+',
                            'p_feasible': judged[n]['p_feasible'], 'why': judged[n]['why'], 'valid_on': {n: reference.get(n)}}
                plan['seeds'].append(seeds[n])
        members = list(start)[:4]
        plan['composite'] = {'members': members, 'change': {n: start[n] for n in members}, 'gain': None, 'source': 'llm',
                             'warnings': plan['check']({**reference, **{n: start[n] for n in members}})}
    return {'assessment': record, 'start_config': start, 'memory_note': str(out.get('memory_note') or '')[:300]}


def judge(driver, plan: dict, knobs: dict, reference: dict, *, session=None, judgments=None,
          max_output_tokens: int = 4000) -> dict | None:
    """One JSON call (role 'transfer', charged to the session like the proposer's); None when the call failed."""
    prompt = json.dumps(request(plan, knobs, reference, judgments), ensure_ascii=False, default=str)
    kw = {'max_output_tokens': max_output_tokens}
    if getattr(session, 'ledger', None) is not None:   # within the search budget, as the proposer's calls
        from ..agent.optimize import call_allowances
        kw = call_allowances(session, driver, TRANSFER_SYSTEM + prompt, max_output_tokens)
        if kw is None:
            return None
    res = driver.complete(TRANSFER_SYSTEM, prompt, schema=TRANSFER_SCHEMA, role='transfer', **kw)
    if session is not None:
        session.charge_llm(res, 'transfer')
    if not res.action:
        return None
    rec = apply(plan, res.action, knobs, reference)
    rec['model'] = getattr(res, 'model', None)
    return rec


def _ran(h: dict) -> bool:
    return h.get('status') != 'invalid' or not set(h.get('failed_checks') or ()) & {'completes', 'no_errors', 'memory'}


def score(record: dict, ledger: list[dict], sigma_pair: float, composite_fate: dict | None = None) -> dict:
    """The judgment against this run's measurements: sign accuracy over the judged moves measured beyond 2 sigma_pair
    (within noise does not count), the Brier score of p_feasible over the judged moves that were tried, the start
    composite's gain (or why it failed)."""
    first = {}
    for h in ledger:                                    # the first measurement of each single-knob move (a member
        ch = h.get('change') or {}                      # retired by a bundle was not measured alone)
        alone = h.get('status') != 'bundled' or h.get('measured') is not None
        if len(ch) == 1 and h.get('status') not in ('open', 'superseded') and alone and json.dumps(ch, default=str) not in first:
            first[json.dumps(ch, default=str)] = h
    right = n = 0
    brier = []
    for a in record.get('assessment') or []:
        h = first.get(json.dumps(a['change'], default=str))
        if h is None:
            continue
        brier.append((a['p_feasible'] - float(_ran(h))) ** 2)
        g = h.get('measured')
        if a['sign'] in ('+', '-', '0') and g is not None and abs(g) > 2 * sigma_pair:
            n += 1
            right += (a['sign'] == '+' and g > 0) or (a['sign'] == '-' and g < 0)
    fate = composite_fate or {}
    start = (round(fate['gain'], 5) if fate.get('gain') is not None and fate.get('status') == 'screened'
             else f'failed: {fate.get("status")} {fate.get("failed_checks") or ""}'.strip() if fate else None)
    return {'sign_acc': round(right / n, 3) if n else None, 'n_resolved': n,
            'feasibility_brier': round(sum(brier) / len(brier), 4) if brier else None, 'n_tried': len(brier),
            'start_gain': start}
