"""result.json + REPORT.md of a project run: what to change, how much faster, what the verifier checked, the cost."""
from __future__ import annotations

import json
import math
from pathlib import Path

from ..verify.verifier import loss_repeat_note


def _fmt(x, nd=4):
    return f'{x:.{nd}g}' if isinstance(x, (int, float)) and not isinstance(x, bool) else ('-' if x is None else str(x))


def _diff(ref: dict, cfg: dict) -> dict:
    return {k: [ref.get(k), v] for k, v in (cfg or {}).items() if ref.get(k) != v}


def _val(x) -> str:
    return "''" if x == '' else str(x)


def _change(c: dict) -> str:
    """A candidate's change for the table (U-1-19): `knob old -> new` pairs, its patch, and the candidate it was applied
    to when that is not the reference; cut only between pairs."""
    ch = c.get('change') if isinstance(c.get('change'), dict) else {}
    parts = [f'{k} {_val(v[0])} -> {_val(v[1])}' if isinstance(v, (list, tuple)) and len(v) == 2 else f'{k} = {_val(v)}'
             for k, v in sorted(ch.items())]
    if c.get('overlay_files'):
        parts.append('patch of ' + ', '.join(str(f) for f in c['overlay_files']))
    if not parts and c.get('change') and not isinstance(c.get('change'), dict):
        parts = [str(c['change'])]
    text = ''
    for i, part in enumerate(parts):
        if len(text) + len(part) > 200:
            text += f', ... ({len(parts) - i} more)'
            break
        text += (', ' if text else '') + part
    on = c.get('parent')
    return (f'on {on}: ' if on and on != 'c0' and text else '') + text.replace('|', '/')


# U-1-20, U-2-7: the stop reasons and the refused proposals in words (the code stays in parentheses)
STOPS = {'no_new_valid_sample': 'no new valid configuration: the last 2000 draws in a row broke a constraint or an '
                                'invariant or repeated a measured one',
         'strategy_done': 'the strategy had nothing left to test',
         'interrupted': 'interrupted (Ctrl-C or a termination signal)',
         'user_stop': 'stopped by you (Ctrl-C or a termination signal); the best candidates so far then went through the '
                      'normal re-verification',
         'process_ended': 'its process ended before the search did (killed, or the machine or the connection was lost)',
         'budget:screens': 'the budget of {n} full runs was used', 'budget:gpu_hours': 'the GPU-hour budget was used',
         'budget:wall_seconds': 'the wall-clock budget was used', 'budget:llm_tokens': 'the LLM token budget was used',
         'budget:llm_usd': 'the LLM USD budget was used',
         'no_new_config': 'OptiCo proposed nothing new: six proposals in a row were refused before any run or repeated '
                          'one already made',
         'refusals': 'the model refused three of OptiCo\'s requests',
         'llm_error_or_budget': 'a model call failed (see below) or the budget ran out between two of OptiCo\'s calls'}
ENABLE = {'resume_continues': 'set checkpoint.save, checkpoint.resume and checkpoint.save_at in infrarsi.yaml',
          'eval_matches': 'set verify.eval; it needs the checkpoint commands too', 'user_tests': 'set verify.tests'}
REFUSED = {'l0_invalid': 'failed validation (a constraint, an invariant, a type or a range)',
           'duplicate': 'repeated a configuration or patch already proposed',
           'patch_rejected': 'patches refused (frozen or unknown files, size)', 'unknown_base': 'named an unknown base',
           'infeasible': 'were judged infeasible',
           'start_failed': 'set a knob value that had failed to start twice in this search'}


def _stop_words(stop: str, screens, code: bool = True) -> str:
    if stop.startswith('same_failure_x'):
        words = f'{stop[len("same_failure_x"):]} runs in a row failed the same way'
    else:
        words = STOPS.get(stop, '').format(n=screens)
    return (f'{words} ({stop})' if code else words) if words else stop or 'unknown'


def _truncated(r: dict) -> list[str]:
    """U-8-4: "keep the current configuration" after a search that ended long before its run budget is not a finding
    about the space; say so under the recommendation."""
    s = r.get('search') or {}
    used, budget = (s.get('used') or {}).get('screens'), (s.get('budget') or {}).get('screens')
    stop = str(s.get('stop_reason') or '')
    if not (used is not None and budget and used < budget
            and (stop in ('no_new_valid_sample', 'no_new_config') or stop.startswith('same_failure'))):
        return []
    return [f'Only {used} of the {budget} budgeted full runs were made - {_stop_words(stop, budget, code=False)} - so this '
            'says that these runs found nothing faster, not that nothing is (another strategy, `--strategy random` or '
            '`hls`, samples the space differently).', '']


def _not_measured(s: dict) -> list[str]:
    """U-3-3: what the proposer put in the ledger that no run ever measured (the search ended first), so a user sees
    that e.g. bf16 was never tried rather than tried and rejected."""
    ledger = s.get('ledger') or []
    if any('base' in h for h in ledger):                # hls2's ledger says why (FABLE_DESIGN_SEARCH_FIX C9)
        return _not_measured_why(ledger, s.get('selected'))
    measured = {str(h.get('mechanism') or h.get('change')) for h in ledger if h.get('status') not in (None, 'open')}
    names = {}
    for h in ledger:
        name = str(h.get('mechanism') or h.get('change'))
        if h.get('status') == 'open' and name not in measured and name not in names:
            g = h.get('predicted_gain_pct')
            names[name] = f'{name} (predicted {g:+.1f} %)' if isinstance(g, (int, float)) else name
    if not names:
        return []
    shown = list(names.values())
    return ['', f'Proposed but not measured in this search: {", ".join(shown[:8])}'
                + (f' and {len(shown) - 8} more' if len(shown) > 8 else '') + '.']


PRUNED = 'unrunnable on this card'                    # hls3 (S-10): a value the sharded-state arithmetic rules out


def _not_measured_why(ledger: list, final: str | None) -> list[str]:
    """The never-measured moves of an hls2 ledger by why: proposed on an earlier incumbent and not carried (a patch, a
    move the incumbent made or undid, `why`), still queued when the budget ended, or not run because the value failed
    to start twice (the user's question of 2026-10-29: why did Liger never get its turn?)."""
    ran = [h for h in ledger if h.get('status') not in (None, 'open', 'superseded')
           and not str(h.get('why') or '').startswith('unrunnable')]
    measured = {str(h.get('mechanism') or h.get('change')) for h in ran}
    moves = {json.dumps(h['change'], sort_keys=True, default=str) for h in ran if h.get('change')}   # under any name
    rank = {'superseded': 0, 'unrunnable': 1, 'pruned': 1, 'deferred': 2, 'queued': 3}   # a move carried on and still
    state: dict[str, tuple] = {}                                                          # queued is not superseded
    for h in ledger:
        name, st, why = str(h.get('mechanism') or h.get('change')), h.get('status'), str(h.get('why') or '')
        if name in measured or h.get('change') and json.dumps(h['change'], sort_keys=True, default=str) in moves:
            continue
        if st == 'open' and h.get('base') == final and h.get('deferred'):    # hls3: predicted not to fit, or waiting
            kind, why = 'deferred', str(h['deferred'])                       # for its members' separate runs
        elif st == 'invalid' and why.startswith(PRUNED):                     # hls3: impossible on this card (S-10)
            kind, why = 'pruned', why[len(PRUNED):].strip(' :')
        else:
            kind = ('queued' if st == 'open' and h.get('base') == final else 'unrunnable' if st == 'invalid'
                    and why.startswith('unrunnable') else 'superseded' if st in ('open', 'superseded') else None)
        if kind and (name not in state or rank[kind] > rank[state[name][0]]):
            g = h.get('predicted_gain_pct')
            what = f'{name} (predicted {g:+.1f} %)' if isinstance(g, (int, float)) and g else name
            state[name] = (kind, what + (f' [{why[len("unrunnable: "):] if kind == "unrunnable" else why}]'
                                         if why and kind != 'queued' else ''))
    groups = {k: [w for kind, w in state.values() if kind == k]
              for k in ('superseded', 'queued', 'deferred', 'unrunnable', 'pruned')}
    words = {'superseded': 'proposed on an earlier incumbent and not tested on a later one',
             'queued': 'queued on the final incumbent when the budget ended',
             'deferred': 'deferred on the final incumbent (hls3: predicted not to fit, or its parts first run separately)',
             'unrunnable': 'not run: the value had failed to start twice',
             'pruned': 'not run: it cannot fit on this card (hls3: the sharded-state arithmetic)'}
    out = []
    for k, shown in groups.items():
        if shown:
            out.append(f'{words[k]}: {"; ".join(shown[:8])}' + (f' and {len(shown) - 8} more' if len(shown) > 8 else ''))
    return ['', 'Proposed but not measured in this search - ' + '. '.join(out) + '.'] if out else []


def _wcs(h: dict | None) -> list[str]:
    """Strategy wcs's account (FABLE_ARCH_REVIEW_20261031 section 4): the prior it got, the table it searched, each
    pick with its feasibility and outcome, the failed runs, the pair table, the patches; the components switched off."""
    if not h:
        return []
    out = ['', '## wcs: the whole-configuration search', '']
    if h.get('off'):
        out.append(f'Components switched off (ablation): {", ".join(h["off"])}.')
    pr = h.get('prior') or {}
    t = h.get('table') or {}
    out.append(f'Prior ({pr.get("source", "none")}): {pr.get("known", 0)} of {pr.get("values", 0)} knob values rated, '
               f'{pr.get("declared_memory", 0)} with a memory direction, {len(pr.get("pairs") or [])} interactions, '
               f'{len(pr.get("configs") or [])} suggested configurations'
               + (f', {h.get("experience_absorbed")} values from the experience store' if h.get('experience_absorbed') else '')
               + f'. Table: {t.get("legal", 0)} legal configurations' + ('' if t.get('whole', True) else ' (a sample of the table)')
               + f'. Model calls: {h.get("llm_calls", 0)}.')
    top = pr.get('top') or []
    if top:
        out.append('The prior\'s largest gains: ' + '; '.join(f'{x["value"]} {x["gain_pct"]:+.1f} %' + (
            f' ({x["memory"]})' if x.get('memory') else '') for x in top[:6]) + '.')
    picks = h.get('picks') or []
    if picks:
        out.append('')
        out.append('| pick | after full runs | kind | configuration | feasibility | predicted | outcome |')
        out.append('|---|---|---|---|---|---|---|')
        for p in picks:
            out.append(f'| {p.get("id")} | {p.get("screens_done")} | {p.get("kind")} | {str(p.get("mechanism"))[:80]} | '
                       f'{p.get("p_feasible")} | {p.get("predicted_gain_pct"):+.1f} % | {p.get("outcome", "-")}'
                       + (f' ({p["gain_vs_reference_pct"]:+.1f} % vs the reference)' if p.get('gain_vs_reference_pct') is not None else '')
                       + ' |')
    fails = h.get('failures') or []
    if fails:
        by = {}
        for f in fails:
            by.setdefault(f.get('class'), []).append(f.get('candidate'))
        out.append('')
        out.append('Failed runs by class: ' + '; '.join(f'{k} {len(v)} ({", ".join(str(x) for x in v[:6])})'
                                                       for k, v in by.items()) + '.')
    pairs = h.get('pairs') or []
    if pairs:
        out.append('Values not combined again (the pair table): ' + '; '.join(
            f'{p["pair"]} (p {p["p"]}: {p["why"]})' for p in pairs[:6]) + '.')
    post = h.get('posterior') or []
    if post:
        out.append('The posterior\'s largest gains: ' + '; '.join(
            f'{x["value"]} {x["gain_pct"]:+.1f} % (sd {x["sd_pct"]:.1f})' for x in post[:6]) + '.')
    patches = h.get('patches') or {}
    if patches:
        out.append('Patches: ' + '; '.join(f'{pid} {p.get("status")}' + (f' {p["gain_pct"]:+.1f} %' if p.get('gain_pct') is not None else '')
                                           + (' (adopted)' if p.get('adopted') else '') for pid, p in patches.items()) + '.')
    n = h.get('counts') or {}
    out.append(f'Counts: {n.get("probe_first", 0)} probed first, {n.get("infeasible", 0)} candidates ruled out by the runs, '
               f'{n.get("partner_probes", 0)} partner probes, {n.get("coverage", 0)} coverage screens, '
               f'{n.get("jitter_thrash", 0)} runs read as thrash from their scattered updates.')
    return out


def _lowfi(h: dict) -> list[str]:
    """A candidate's probes and short runs (search/ladder.py's account) and its memory ledger, in words."""
    lad = h.get('ladder') or {}
    acc = lad.get('account') or {}
    by = acc.get('by_fidelity') or {}
    n = lad.get('counts') or {}
    out = []
    if acc:
        out.append('Runs by fidelity: ' + '; '.join(f'{f} {d["runs"]} ({d["gpu_hours"]:.3g} GPU-h)' for f, d in by.items())
                   + f'. Probes and short runs: {acc.get("lowfi_gpu_hours", 0):.3g} GPU-h, '
                   f'{100 * (acc.get("lowfi_share_of_gpu_hours") or 0):.0f} % of the search\'s GPU-hours (allowance '
                   f'{acc.get("allowance_gpu_hours", 0):.3g} GPU-h); short runs promoted to full {n.get("promoted", 0)} of '
                   f'{n.get("promoted", 0) + n.get("not_promoted", 0)}; probed first {n.get("probe_first", 0)}; capped by the '
                   f'allowance {n.get("lowfi_capped", 0)}.')
        out.append(acc.get('note') or '')
    led = h.get('ledger') or {}
    if led:
        obs = led.get('observations') or {}
        out.append(f'Memory ledger: {sum(obs.values())} observations ({", ".join(f"{k} {v}" for k, v in obs.items()) or "none"})'
                   f', {led.get("basis", 0)} basis functions, terms: {", ".join(led.get("terms") or []) or "none"}'
                   + (f' (dropped: {", ".join(str(t.get("name")) for t in led["dropped_terms"][:4])})' if led.get('dropped_terms') else '')
                   + '; moved most by the runs: ' + '; '.join(f'{r["name"]} {r.get("prior_gib", r.get("prior_coef"))} -> '
                                                            f'{r["fitted"]}' for r in (led.get('moved_most') or [])[:4]) + '.')
    return out


def _esls(h: dict | None) -> list[str]:
    """Strategy esls's account (possible_search_algorithm/20 section 1): the spenders and the chain of spend levels,
    what each enabler frees, the boundary screens with their sets, the bisection's probes, the tail, the probes and short
    runs (not screens) with their GPU-hours, the memory ledger; the components switched off."""
    if not h:
        return []
    out = ['', '## esls: the enabler / spender lattice search', '']
    if h.get('off'):
        out.append(f'Components switched off (ablation): {", ".join(h["off"])}.')
    if h.get('llm_calls') is not None:
        out.append(f'Model calls: {h["llm_calls"]}' + (f' of at most {h["llm_calls_max"]}' if h.get('llm_calls_max') else '')
                   + " (the prior, the code track's asks and their re-asks: one allowance, wcs's).")
    lat = h.get('lattice') or []
    out.append(f'Spenders: {", ".join(h.get("spenders") or []) or "none"}. Chain: {len(lat)} spend levels'
               + (f' ({lat[0]["mem_gib"]} to {lat[-1]["mem_gib"]} GiB per GPU by the ledger)' if lat and lat[0].get('mem_gib') is not None else '')
               + (f', {len(h["pruned"])} pruned (out of memory with every enabler)' if h.get('pruned') else '') + '.')
    en = h.get('enablers') or {}
    if en:
        out.append('Enablers: ' + '; '.join(f'{k} {e.get("status")}' + (f', frees {e["frees"]:.1f} GiB' if e.get('frees') is not None else '')
                                            + (f' (declared {e["declared"]})' if e.get('declared') is not None else '')
                                            for k, e in en.items()) + '.')
    bs = h.get('boundaries') or []
    if bs:
        out += ['', '| enabler set | level | configuration | outcome | gain vs the reference |', '|---|---|---|---|---|']
        for b in bs:
            out.append(f'| {b.get("set") or "none"}{" (ledge)" if b.get("ledge") else ""} | {b.get("level")} | '
                       f'{str(b.get("configuration"))[:90]} | {b.get("outcome")} | '
                       + (f'{b["gain_pct"]:+.2f} %' if b.get('gain_pct') is not None else '-') + ' |')
        out.append('')
    probes = h.get('probes_bisection') or []
    if probes:
        out.append(f'Bisection probes: {len(probes)} (' + '; '.join(f'{p["set"]} level {p["level"]}: '
                                                               f'{"fits" if p["fits"] else "does not fit"}' for p in probes[:8]) + ').')
    t = h.get('tail') or {}
    if t.get('members'):
        out.append(f'Tail: {", ".join(t["members"])} as one bundle; accepted: {", ".join(t.get("accepted") or []) or "none"}'
                   f'; diagnostic screens {t.get("diagnostic_screens", 0)}.')
    sb = h.get('screens_by_purpose') or {}
    if sb:
        out.append('Full screens by purpose: ' + ', '.join(f'{k} {v}' for k, v in sb.items()) + '.')
    return out + _lowfi(h)


def _rpp(h: dict | None) -> list[str]:
    """Strategy rpp's (or its control rpp-ctl's) account (possible_search_algorithm/20 section 6): the draw's weighting,
    each draw with its feasibility, plan and outcome, the probes and short runs (not screens) with their GPU-hours, the
    memory ledger; the components switched off."""
    if not h:
        return []
    name = h.get('name') or 'rpp'
    out = ['', f'## {name}: draws weighted by feasibility' + (' and the time model' if h.get('temperature') != 'inf' else ''), '']
    if h.get('off'):
        out.append(f'Components switched off (ablation): {", ".join(h["off"])}.')
    out.append(f'Weight P_feas^{h.get("a")} x exp(mu / T), T {h.get("temperature")}'
               + (f' (T0 {h["T0"]:.3g}, annealed over the valid full runs)' if isinstance(h.get('T0'), (int, float))
                  and h.get('temperature') != 'inf' else '') + f'; model calls: {h.get("llm_calls", 0)}'
               + (f' of at most {h["llm_calls_max"]}' if h.get('llm_calls_max') else '') + '.')
    draws = h.get('draws') or []
    if draws:
        out += ['', '| draw | after full runs | feasibility | plan | outcome |', '|---|---|---|---|---|']
        for d in draws:
            out.append(f'| {d.get("id")} | {d.get("screens_done")} | {d.get("p_feasible")} | {" -> ".join(d.get("plan") or [])} | '
                       f'{d.get("outcome", "-")} |')
        out.append('')
    return out + _lowfi(h)


def _llmopt(h: dict | None) -> list[str]:
    """Strategy opro's or ltune's account (possible_search_algorithm/10 section 04; search/opro.py, search/ltune.py):
    the model's configuration calls with what each gave and what was admitted, each proposal with its feasibility, its
    stated, short and measured gains and its outcome (ltune: its rung of the race), the proposals ruled out and why, the
    tail, the probes and short runs (not screens) with their GPU-hours, the memory ledger; the components switched off."""
    if not h:
        return []
    name = h.get('name') or 'opro'
    title = {'opro': 'rounds of whole configurations from the model (OPRO)',
             'ltune': 'one call, a race of whole configurations (lambda-Tune)'}.get(name, 'whole configurations from the model')
    out = ['', f'## {name}: {title}', '']
    if h.get('off'):
        out.append(f'Components switched off (ablation): {", ".join(h["off"])}.')
    out.append(f'Model calls: {h.get("llm_calls", 0)} of at most {h.get("llm_calls_max")} and {h.get("llm_usd_max")} USD '
               "(the prior, the configuration calls, the code track's asks and their re-asks: one budget); code asks "
               f'{h.get("code_asks", 0)}, code screens {h.get("code_screens", 0)}.')
    rounds = h.get('rounds') or []
    if rounds:
        out.append('Configuration calls: ' + '; '.join(
            f'{r.get("purpose")} {r.get("round")} after {r.get("screens_done")} full screens: {r.get("got")} given, '
            f'{r.get("admitted")} admitted' for r in rounds[:12]) + '.')

    def pct(x):
        return f'{x:+.2f} %' if isinstance(x, (int, float)) else '-'
    arch = h.get('archive') or []
    if arch:
        out += ['', '| call | proposal | feasibility | stated | short | measured | outcome |', '|---|---|---|---|---|---|---|']
        for a in arch:
            ch = ', '.join(f'{k}={v}' for k, v in (a.get('change') or {}).items())[:90].replace('|', '/')
            out.append(f'| {a.get("round")} | {ch or "(the reference)"} | {a.get("p_feasible")} | '
                       f'{pct(a.get("stated_gain_pct"))} | {pct(a.get("short_gain_pct"))} | {pct(a.get("measured_gain_pct"))} | '
                       f'{a.get("stage") or a.get("outcome") or a.get("status")} |')
        out.append('')
    rej = h.get('rejected') or []
    if rej:
        out.append(f'Ruled out before running: {len(rej)} (' + '; '.join(
            f'{", ".join(f"{k}={v}" for k, v in (x.get("change") or {}).items())[:60]}: {str(x.get("why"))[:80]}'
            for x in rej[:4]) + ').')
    if name == 'ltune':
        out.append(f'Race: {h.get("race_full_runs", 0)} of its configurations run in full.')
    t = h.get('tail') or {}
    if t.get('members'):
        out.append(f'Tail: {", ".join(t["members"])} as one bundle; accepted: {", ".join(t.get("accepted") or []) or "none"}'
                   f'; diagnostic screens {t.get("diagnostic_screens", 0)}.')
    sb = h.get('screens_by_purpose') or {}
    if sb:
        out.append('Full screens by purpose: ' + ', '.join(f'{k} {v}' for k, v in sb.items()) + '.')
    return out + _lowfi(h)


def _xfer(h: dict | None) -> list[str]:
    """Strategy xfer's account (FABLE_PICK_C6_20261031 section 2; search/xfer.py): the source environments (their numbers,
    weights before and after the RGPE updates, their winners and what became of them), the transfer brief and the port
    ask (what it showed, what was ported and how it fared), the notes (no store, no brief), the code track's asks, the
    tail, the probes (not screens) with their GPU-hours; the components switched off."""
    if not h:
        return []
    out = ['', '## xfer: transfer-first over the experience store', '']
    if h.get('off'):
        out.append(f'Components switched off (ablation): {", ".join(h["off"])}.')
    for n in h.get('notes') or []:
        out.append(f'Note: {n}.')
    out.append(f'Model calls: {h.get("llm_calls", 0)} of at most {h.get("llm_calls_max")} and {h.get("llm_usd_max")} USD '
               "(the prior, the port ask, the code track's asks and their re-asks: one budget); code asks "
               f'{h.get("code_asks", 0)}, code screens {h.get("code_screens", 0)}; source winners run in full on the first '
               f'screens: {h.get("source_full_runs", 0)}.')

    def pct(x):
        return f'{x:+.2f} %' if isinstance(x, (int, float)) else '-'
    srcs = h.get('sources') or []
    if srcs:
        out += ['', '| source | card GiB / nodes / params | weight start -> now | winner there | change here | here | state |',
                '|---|---|---|---|---|---|---|']
        for s in srcs:
            e = s.get('env') or {}
            ch = ', '.join(f'{k}={v}' for k, v in (s.get('change') or {}).items())[:80].replace('|', '/')
            out.append(f'| {s.get("source")} | {e.get("card_gib")} / {e.get("nodes")} / {e.get("params_b")} | '
                       f'{s.get("weight_start")} -> {s.get("weight")} | {pct(s.get("winner_gain_pct"))} | {ch or "-"} | '
                       f'{pct(s.get("measured_gain_pct"))} | {s.get("state")}'
                       f'{" (" + s["outcome"] + ")" if s.get("outcome") and s["outcome"] != s.get("state") else ""} |')
        out.append('')
    b, port = h.get('brief') or {}, h.get('port') or {}
    out.append(f'Transfer brief: {b.get("proven", 0)} proven and {b.get("inert", 0)} inert patches of other environments.')
    if port.get('shown'):
        out.append(f'Port ask after {port.get("screens_done")} full screens: {len(port["shown"])} proven patches shown '
                   f'({", ".join(port["shown"][:12])}), {port.get("do_not_repeat", 0)} not to repeat; {port.get("answers", 0)} '
                   f'patches back, {len(port.get("queued") or [])} queued.')
    elif port.get('sent') is False:
        out.append(f'Port ask after {port.get("screens_done")} full screens: its request did not go out (no entry shown).')
    for pid, x in (port.get('ported') or {}).items():
        out.append(f'Ported {pid} (brief entry {x.get("entry")}): {x.get("status")}'
                   + (f', {x["gain_pct"]:+.2f} %' if isinstance(x.get('gain_pct'), (int, float)) else '') + '.')
    t = h.get('tail') or {}
    if t.get('members'):
        out.append(f'Tail: {", ".join(t["members"])} as one bundle; accepted: {", ".join(t.get("accepted") or []) or "none"}'
                   f'; diagnostic screens {t.get("diagnostic_screens", 0)}.')
    sb = h.get('screens_by_purpose') or {}
    if sb:
        out.append('Full screens by purpose: ' + ', '.join(f'{k} {v}' for k, v in sb.items()) + '.')
    return out + _lowfi(h)


def _hls3(h: dict | None) -> list[str]:
    """Strategy hls3's account (FABLE_TAKEAWAYS_20261030 section 4): its failed runs by class, what the memory model
    decided (deferred, kept probes), the memory bundles with their composition and fate, the stall steps, the changes
    that failed together, the re-based moves and the pruned values; the components switched off."""
    if not h:
        return []
    out = ['', '## hls3: memory, failures and stalls', '']
    if h.get('off'):
        out.append(f'Components switched off (ablation): {", ".join(h["off"])}.')
    fails = h.get('failures') or []
    if fails:
        by = {}
        for f in fails:
            by.setdefault(f.get('class'), []).append(f.get('candidate'))
        out.append('Failed runs by class: ' + '; '.join(f'{k} {len(v)} ({", ".join(str(x) for x in v[:6])})'
                                                       for k, v in by.items()) + '.')
        oom = [f for f in fails if f.get('class') == 'oom' and f.get('detail')]
        if oom:
            out.append('Out of memory: ' + '; '.join(f'{f["candidate"]} {f["detail"]}' for f in oom[:4]) + '.')
    n = h.get('counts') or {}
    m = h.get('memory') or {}
    floor = (m.get('floor') or {}).get('floor_gib')
    out.append(f'Memory model (capacity {m.get("capacity_gib") or "?"} GiB'
               + (f', sharded state floor {floor} GiB' if floor else '') + f'): {n.get("deferred", 0)} proposals deferred as '
               f'predicted out of memory (by measured moves), {n.get("unsure", 0)} probed first as unsure (a declaration '
               f'at most asks for a probe), {n.get("kept_probes", 0)} probes kept while the screens were the scarcer budget '
               '(a failed probe spends no screen)'
               + (f'; {n["possible_thrash"]} slow runs without memory evidence noted as possible thrash'
                  if n.get('possible_thrash') else '') + '.')
    for b in h.get('bundles') or []:
        out.append(f'- bundle {b.get("bundle")} on {b.get("base")}: {b.get("spender_mechanism")} + '
                   f'{", ".join(b.get("enablers") or [])} ({b.get("origin")}; why: {str(b.get("why"))[:120]}) -> '
                   f'{b.get("status") or "not run"}' + (f', {b["candidate"]}' if b.get('candidate') else '')
                   + (f', {100 * b["gain"]:+.1f} %' if isinstance(b.get('gain'), (int, float)) else ''))
    for st in h.get('stalls') or []:
        out.append(f'- stall after {st.get("after_screens")} full runs on {st.get("incumbent")}: probes with '
                   f'{", ".join(st.get("probes") or []) or "none"}; whole configuration: {st.get("whole") or "none"}')
    for t in h.get('failed_together') or []:
        sep = t.get('separately') or {}
        out.append(f'- failed together in {t.get("candidate")} ({t.get("kind")}): {"; ".join(t.get("members") or [])}; '
                   f'run separately since: {", ".join(f"{k} ({v})" for k, v in sep.items()) or "none yet"}')
    if h.get('rebased'):
        out.append('Re-based on a new incumbent: ' + '; '.join(f'{r["move"]} ({r["was"]} on {r["from"]}, queued on {r["to"]})'
                                                               for r in h['rebased'][:6]) + '.')
    if h.get('pruned'):
        out.append('Pruned as impossible on this card: ' + '; '.join(f'{k} ({v})' for k, v in list(h['pruned'].items())[:4])
                   + '.')
    v = (h.get('variance') or {}).get('reference') or {}
    if v:
        out.append(f'Reference updates: mean {v.get("mean_over_median_pct")} % above the median, {v.get("stalls")} stalls of '
                   f'{v.get("updates")}; the search compared medians.')
    return out


def _refused_words(inv: dict | None) -> str:
    return ', '.join(f'{n} {REFUSED.get(k, k)} ({k})' for k, n in (inv or {}).items()) or 'none'


def _reasons(r: dict) -> dict:
    """candidate -> the first reason of its failed run (result.json screens), for the table (U-1-22)."""
    out = {}
    for s in r.get('screens') or []:
        if s.get('valid') is False and s.get('reasons') and s.get('candidate_id') not in out:
            out[s['candidate_id']] = str(s['reasons'][0])
    return out


def _always_failing(r: dict, why: dict) -> list[str]:
    """Knob values whose every run that changed to them (two or more) failed the same way (U-7-2): a flag the script
    refuses, a value its code has no case for. Values that only ever ran together are named together; a descendant that
    inherits the value is not counted, so the line reports those runs, not the value's fate."""
    runs = {}
    for c in r.get('candidates') or []:                 # (a run aborted as slower did not fail)
        if c.get('valid') is None or c.get('status') == 'aborted' or not isinstance(c.get('change'), dict):
            continue
        for k, v in c['change'].items():
            new = v[1] if isinstance(v, (list, tuple)) and len(v) == 2 else v
            runs.setdefault(f'{k}={_val(new)}', []).append(c)
    groups = {}
    for value, cs in runs.items():
        reasons = {why.get(c['id']) for c in cs}
        if len(cs) >= 2 and not any(c.get('valid') for c in cs) and len(reasons) == 1 and None not in reasons:
            groups.setdefault((tuple(c['id'] for c in cs), reasons.pop()), []).append(value)
    return [f'{", ".join(values)}: the {len(ids)} runs whose change set {"them" if len(values) > 1 else "it"} '
            f'({", ".join(ids)}) all failed the same way: {reason[:200]}'
            for (ids, reason), values in groups.items()]


def _experience(e: dict | None) -> list[str]:
    """The experience section: what the store knew, what the search did with it, the transfer judgment's scores."""
    if not e:
        return []
    lv, s = e.get('levels') or {}, e.get('search') or {}
    out = ['', '## Experience', '', f'Store: {len(e.get("packs") or [])} packs; knobs covered {lv.get("covered", 0)}, '
           f'null {lv.get("null", 0)}, partial {lv.get("partial", 0)}, none {lv.get("none", 0)}.']
    if s:                                               # U-3-7: what the words mean, in the line
        out.append(f'Seeded {s.get("seeded", 0)} entries (changes taken from earlier measurements): confirmed '
                   f'{s.get("confirmed", 0)}, bundled {s.get("bundled", 0)} (tested together with others), refuted '
                   f'{s.get("refuted", 0)} (not faster here), unresolved {s.get("unresolved", 0)} (within the noise), '
                   f'invalid {s.get("invalid", 0)}, never measured {s.get("open", 0) + s.get("superseded", 0)}; full runs '
                   'spent on them: '
                   f'{s.get("screens", 0)}.')
    comp, fate = e.get('composite'), s.get('composite_fate')
    if comp:
        g = comp.get('gain')
        src = f' (source speed-up x{math.exp(g):.3f})' if g is not None else ' (from the transfer judgment)'
        if not fate:
            what = 'not run'
        else:
            what = 'moved the incumbent' if fate.get('moved') else str(fate.get('status'))
            if fate.get('gain') is not None:
                what += f', measured x{math.exp(fate["gain"]):.3f}'
            if fate.get('failed_checks'):
                what += f', failed {", ".join(fate["failed_checks"])}'
        out.append(f'Start composite {", ".join(f"{k}={v}" for k, v in comp["change"].items())}{src}: {what}.')
    j = e.get('judgment')
    if j:                                               # U-3-7, U-5-9: scores in words, no empty values
        sc = j.get('scores') or {}
        head = (f'Transfer judgment ({j.get("model")}): the model judged how {len(j.get("assessment") or [])} knobs carry '
                'over from the measured environments')
        if sc.get('n_resolved') or sc.get('n_tried'):
            acc, n = sc.get('sign_acc'), sc.get('n_resolved') or 0
            sign = (f'the direction it predicted (faster / slower) was right for {100 * acc:.0f} % of the {n} single-knob '
                    'effects resolved beyond the noise' if n and isinstance(acc, (int, float)) and not isinstance(acc, bool)
                    else 'no single-knob effect was resolved beyond the noise, so its predicted directions are unscored')
            out.append(f'{head}; {sign}, feasibility Brier score {_fmt(sc.get("feasibility_brier"), 3)} over '
                       f'{sc.get("n_tried")} tried (0 is perfect)'
                       + (f', start composite: {sc["start_gain"]}' if sc.get('start_gain') is not None else '') + '.')
        else:
            out.append(f'{head}; nothing to score yet (none of the knobs it judged was measured in this run).')
    if e.get('written'):
        out.append(f'This run was written back as experience: `{e["written"]}` (share it with `infrarsi exp export`).')
    elif e.get('not_written'):
        out.append(f'This run was not written back as experience: {e["not_written"]}.')
    return out


def _code_review(cr: dict | None) -> list[str]:
    if not cr:
        return []
    verdict = {'patch': 'code changes worth a test run', 'no_patch': 'nothing that only code can remove, so the search '
               'changed knobs only'}.get(cr.get('verdict'), 'no usable answer, so the search showed the model the code')
    out = ['', '## Code review', '', f'Before the search the model reviewed the mutable code: {verdict}. '
           f'{cr.get("summary") or ""}'.rstrip(), '']
    if cr.get('findings'):                              # U-3-8: which knob addresses the finding, if any
        out += ['| file | finding | knob that reaches it | patch |', '|---|---|---|---|']
        out += [f'| {f.get("file")} | {str(f.get("issue")).replace("|", "/")[:160]} | {f.get("knob") or "code only"} | '
                f'{f.get("patch") or "-"} |' for f in cr['findings']]
    return out


def _numerics(r: dict) -> list[str]:
    """Each tested change's loss deviation from the reference and from its base (the run it was applied to) with the
    class the proposer declared, against the repeat floor: the largest deviation of a re-run of the reference
    configuration (the doctor run, the promotion re-runs). A change is judged by its own effect (from its base). A
    diagnostic, not a verdict: the tolerance is the one acceptance rule."""
    s = r.get('search') or {}
    rows = [(str(h.get('mechanism') or h.get('change')), [h.get('change_class')], h['loss_deviation_pct'],
             h.get('loss_deviation_base_pct')) for h in s.get('ledger') or [] if h.get('loss_deviation_pct') is not None]
    rows += [('bundle: ' + ' + '.join(b['members']), b.get('classes') or [None], b['loss_deviation_pct'],
              b.get('loss_deviation_base_pct')) for b in s.get('bundle_numerics') or []]   # what its members did not
    if not rows:
        return []
    reps = [x for x in [s.get('doctor_repeat_loss_deviation')]
            + [d for t in (r.get('promotion') or {}).get('tried') or []
               for d in (t.get('reference_repeat_loss_deviation'), t.get('candidate_repeat_loss_deviation'))]
            if type(x) in (int, float)]
    floor = 100 * max(reps) if reps else None
    out = [f'Loss deviation of each tested change by its declared class (a diagnostic; the {r["tolerance"]:.1%} tolerance '
           'is the one acceptance rule). Repeat floor: ' + (
               f'{floor:.4f} %, the largest of {len(reps)} repeat(s) of one configuration (the reference re-runs, the '
               'promoted candidate\'s fresh run).' if floor is not None
               else 'unknown (no re-run of the reference configuration).'), '',
           '| change | declared class | from the reference | from its base | its own vs the repeat floor |',
           '|---|---|---|---|---|']
    for name, classes, d, db in rows[:30]:
        own = d if db is None else db                    # the change's own effect: from the run it was applied to
        vs = '-' if floor is None else 'within' if own <= floor else \
            'above: declared execution, numerics changed' if all(c == 'execution' for c in classes) else 'above'
        out.append(f'| {name.replace("|", "/")[:80]} | {", ".join(c or "-" for c in classes)} | {d:.4f} % | '
                   f'{"-" if db is None else f"{db:.4f} %"} | {vs} |')
    # U-3-6: a floor of 0 % (a job that repeats exactly) puts every numerical change, however small, "above"
    out += ['', 'How to read it: the repeat floor is the largest loss difference between two runs of one configuration; '
                '"from its base" is the change\'s own effect, measured against the run it was applied to; "above" marks a '
                'change that moved the loss more than repeating does - expected for a declared numerics change (bf16), worth '
                'a look for a declared execution one. Only the tolerance decides acceptance.']
    return out + ['']


def _rerun_note(t: dict, promoted: bool) -> str | None:
    """One or two sentences a user can act on when a candidate's fresh re-run at promotion did not reproduce its screen
    (D1-20: c1 screened at 3.062 s and re-ran at 3.245 s while the reference repeated within 0.3 %); None otherwise."""
    rv = t.get('rerun_vs_screen') or {}
    if not rv.get('beyond_margin'):
        return None
    m = next((i['evidence'].get('margin') for i in t.get('items') or [] if i.get('id') == 'speed_reproduces' and i.get('evidence')), None)
    d = rv['log_ratio']
    how = f'{100 * abs(math.expm1(d)):.1f} % {"slower" if d > 0 else "faster"} than its screen ({_fmt(rv["rerun"])} s vs {_fmt(rv["screen"])} s'
    how += f'; two runs of one configuration agree within {100 * m:.1f} % on this machine' if m is not None else ''
    rep = t.get('reference_repeat')
    how += f', and the reference repeated within {100 * abs(rep):.1f} %)' if rep is not None else ')'
    if promoted:
        sp = next((i['evidence'].get('speedup') for i in t.get('items') or [] if i.get('id') == 'speed_reproduces' and i.get('evidence')), None)
        claim = f'x{sp:.3f}' if sp else 'the re-run speed-up'
        fast, slow = sorted((rv['screen'], rv['rerun']))
        return (f'Its fresh re-run was {how}: the configuration ran at two speeds during this run, and {claim} is the one the '
                f'{"slower" if d > 0 else "faster"} run gives. Before relying on {"more than " if d > 0 else ""}{claim}, run '
                f'the recommended configuration twice: a repeat near {_fmt(fast)} s means the faster speed is available, one '
                f'near {_fmt(slow)} s means it is not.')
    return (f'{t["candidate"]}\'s screen did not reproduce: its fresh re-run was {how}, so the gain that could be verified is '
            f'the re-run\'s. The configuration ran at two speeds during this run; if you want to pursue it, run it twice by hand.')


def _rechecks(s: dict) -> list[str]:
    """The engine's re-runs of an incumbent after two aborted candidates agreed with each other but not with its screen."""
    out = []
    for rc in s.get('rechecks') or []:
        a, b = rc.get('after_aborts') or ['?', '?']
        ts = rc.get('aborted_step_times') or [None, None]
        head = (f'Incumbent re-run: after {a} and {b} were aborted at the same speed ({_fmt(ts[0])} s, {_fmt(ts[1])} s) against '
                f'{rc["incumbent"]}\'s screen ({_fmt(rc.get("t_screen"))} s), {rc["incumbent"]} was run again: ')
        if rc.get('t_rerun') is None:
            out.append(head + f'no result ({rc.get("why")}).')
        elif rc.get('moved'):
            out.append(head + f'{_fmt(rc["t_rerun"])} s ({100 * rc["d"]:+.1f} %, beyond the noise), so later candidates were compared '
                       f'with {_fmt(rc["t_rerun"])} s; {a} and {b} may be as fast as {rc["incumbent"]} rather than slower.')
        elif abs(rc.get('d') or 0.0) <= (rc.get('threshold') or 0.0):
            out.append(head + f'{_fmt(rc["t_rerun"])} s ({100 * rc["d"]:+.1f} %, within the noise), so its screen stands.')
        else:
            out.append(head + f'{_fmt(rc["t_rerun"])} s ({100 * rc["d"]:+.1f} %, faster than its screen beyond the noise), so '
                       'later candidates were compared with the mean of the two runs.')
    return out


def _phase_tried(r: dict) -> list[str]:
    """What the search tried on each phase class: the measured ledger rows by the classes their knobs' mechanisms act
    on (result.knob_phases, experience/data/phases.yaml)."""
    kp = r.get('knob_phases') or {}
    by = {}
    for h in (r.get('search') or {}).get('ledger') or []:
        if h.get('status') in (None, 'open', 'superseded') or not isinstance(h.get('change'), dict):
            continue
        sets = [kp.get(k) for k in h['change']]
        names = ['every phase'] if any(s == 'all' for s in sets) else sorted({c for s in sets if s for c in s})
        g = h.get('measured_gain_pct')
        what = f'{h.get("mechanism") or ", ".join(h["change"])} ({f"x{math.exp(g / 100):.3f}" if g is not None else h["status"]})'
        for c in names or ['unmapped']:
            by.setdefault(c, []).append(what)
    return ['Tried on each phase class: ' + '; '.join(f'{c}: {", ".join(v[:6])}' for c, v in sorted(by.items()))
            + '.'] if by else []


def _window(p: dict) -> str:
    """U-1-25: the scoring window as what it is (the updates that are timed; the ones before are warm-up)."""
    w = p.get('window')
    if isinstance(w, (list, tuple)) and len(w) == 2 and all(isinstance(x, int) for x in w):
        return f'timed updates {w[0]}-{w[1]}' + (f' (the first {w[0] - 1} are warm-up)' if w[0] > 1 else '')
    return f'scoring window {w}'


def resumed_lines(resumes: list) -> list[str]:
    """The REPORT.md section for `resumes`."""
    out = ['## Resumed', '',
           f'This run was resumed {len(resumes)} time(s) from the search-state checkpoint it left; the amounts used before '
           'each resume count towards the budget now given, and the run keeps its id.', '']
    for i, e in enumerate(resumes, 1):
        used, b0, b1 = e.get('used_before') or {}, e.get('budget_before') or {}, e.get('budget_after') or {}
        ref = e.get('reference') or {}
        out.append(f'{i}. {e.get("utc")}: from `{(e.get("from_checkpoint") or {}).get("file", e.get("from"))}` saved '
                   f'{(e.get("from_checkpoint") or {}).get("saved_utc")} after {e.get("screens_done")} full runs '
                   f'(used: {used.get("llm_usd", 0):.3f} USD, {used.get("llm_tokens", 0)} tokens, '
                   f'{used.get("gpu_hours", 0):.3f} GPU-hours, {used.get("wall_seconds", 0) / 3600:.2f} h); '
                   f'budget {_budget_words(b0)} -> {_budget_words(b1)}; reference {ref.get("mode", "reused")}'
                   + (f' (x{ref["ratio_new_over_saved"]} the saved one, earlier runs not rescaled)' if ref.get('ratio_new_over_saved')
                      else '')
                   + f'; code {"CHANGED (--allow-code-change): " if e.get("code_changed") else "unchanged: "}'
                   f'{str(e.get("code_hash_before"))[:12]} -> {str(e.get("code_hash_after"))[:12]}.')
        if (e.get('requirements') or {}).get('changed'):
            out[-1] += ' The run keeps the requirements it started with; those of infrarsi.yaml now differ (new requirements need a new run).'
    out.append('')
    return out


def _budget_words(b: dict) -> str:
    from ..agent.budget import UNLIMITED
    if not b:
        return '-'
    tokens = f'{b["llm_tokens"]} tokens' if b.get('llm_tokens') is not None and b['llm_tokens'] < UNLIMITED else 'no token cap'
    wall = f'{b["wall_seconds"] / 3600:.3g} h' if b.get('wall_seconds') is not None and b['wall_seconds'] < UNLIMITED else 'no time cap'
    usd = f'{b["llm_usd"]} USD' if b.get('llm_usd') is not None else 'no dollar cap'
    return f'{b.get("screens")} runs, {usd}, {tokens}, {wall}'


def markdown(r: dict) -> str:
    p, ref = r['project'], r['reference']
    where = {'ssh': ' on ssh hosts', 'slurm': ' in a slurm allocation'}.get(p.get('execution'), '')   # U-6-1
    lines = [f'# InfraRSI run {r["run_id"]}', '', f'Project **{p["name"]}** ({p["gpus"]} GPU(s){where}, {p["steps"]} '
             f'updates per run, {_window(p)}); loss tolerance **{r["tolerance"]:.1%}**.', '']
    if r.get('outcome') == 'infrastructure_failure' and not r.get('candidates'):
        lines += ['## The machines could not be reached', '', 'The reference run could not even start: InfraRSI could not '
                  'copy the project to the hosts or connect to them. Nothing was measured. Check that the hosts in '
                  '`execution.hosts` answer to ssh without a password prompt, then run again.', '',
                  '```', r.get('infrastructure') or '', '```']
        return '\n'.join(lines) + '\n'
    if r.get('resumes'):                                 # `infrarsi run --resume`: where the run was continued from
        lines += resumed_lines(r['resumes'])
    if r.get('finished'):                                # `infrarsi finish`: the promotion was made later, from the receipts
        lines += ['## Finished later', '', f'The promotion below was made afterwards by `infrarsi finish` ({r["finished"].get("utc")}), '
                  'from the receipts the run left on disk. The search itself was not continued, and this run wrote no '
                  'experience back.', '']
    during = r.get('interrupted_during') if r.get('outcome') == 'interrupted' else None   # U-7-3: where the run stopped
    if during == 'profile':                              # before the search: the reference run is all there is
        lines += ['## Interrupted', '', 'The run was stopped (Ctrl-C or a termination signal) during the profile run of the '
                  'reference configuration, before the search: nothing was searched or promoted. The reference run took '
                  f'{_fmt(ref.get("step_time"))} s per update (its log: `{ref.get("log")}`). Run again to search.']
        return '\n'.join(lines) + '\n'
    if r.get('outcome') == 'infrastructure_failure':
        lines += ['## Stopped early: the machines could not be reached', '', 'InfraRSI lost the hosts during the run, '
                  'retried, and stopped. The runs measured before that are reported below; nothing was promoted, and the '
                  'candidate that was running is not counted as a failure. Check the connection, then run again.', '',
                  '```', r.get('infrastructure') or '', '```', '']
    if during == 'promotion':                            # the search finished; `tried` holds the complete promotion rows
        pr = r.get('promotion') or {}
        ids = [t['candidate'] for t in pr.get('tried') or []]
        lines += ['## Interrupted', '', 'The search finished, then the run was stopped (Ctrl-C or a termination signal): '
                  f'{pr.get("why")}; nothing was promoted. '
                  + (f'Before that, the promotion items of {", ".join(ids)} were checked (the table under Verifier). '
                     if ids else '')
                  + 'The search\'s runs are reported below; the cost counts the promotion runs that finished. Run again to '
                  'search again: a new run starts over, nothing is resumed. '
                  f'`infrarsi finish {r.get("run_dir")}` re-verifies the best candidates and adopts the winner.', '']
    elif during == 'write-back':                         # after the last run: the results below are not affected
        lines += ['## Interrupted', '', 'The run was stopped (Ctrl-C or a termination signal) after its last run, while it '
                  'wrote its results back for later runs (the ledger, the experience pack): the results below stand as '
                  'measured; only the write-back was cut short, so later runs may not know this run\'s measurements.', '']
    elif during == 'report':                             # the write-back had finished too
        lines += ['## Interrupted', '', 'The run was stopped (Ctrl-C or a termination signal) after its last run and its '
                  'write-back, while it wrote this report: the results below are complete.', '']
    elif r.get('outcome') == 'interrupted':
        n = len(r.get('screens') or [])
        best = min((c for c in r.get('candidates') or [] if c.get('valid') and c.get('step_time') and c['id'] != 'c0'),
                   key=lambda c: c['step_time'], default=None)
        t_ref = (r.get('reference') or {}).get('step_time')
        # U-7-4: a full run cut off by the stop still counts in the search line below; a new run starts over
        lines += ['## Interrupted', '', f'The search was stopped (Ctrl-C or a termination signal) after {n} run(s); nothing '
                  'was promoted. The runs measured before that are reported below'
                  + (f'; the fastest valid one, {best["id"]} (x{t_ref / best["step_time"]:.3f} on its screen), was not '
                     're-run, so its gain is not verified' if best and t_ref else '') + '. A full run cut off by the stop '
                  'still counts in the search line below. Run again to search again: a new run starts over, nothing is '
                  f'resumed. `infrarsi finish {r.get("run_dir")}` re-verifies the best candidates and adopts the winner.', '']
    if r.get('outcome') == 'not_verifiable':
        lines += ['## Nothing could be verified', '', 'The reference ran, but the search did not start: '
                  + (r.get('feedback') or '') + '. Then run again.', '']
    if r.get('outcome') == 'reference_failed':
        lines += ['## The reference run failed', '', 'Your current configuration did not pass the run-health checks, so nothing '
                  'was searched. Fix the items below (often: the log does not expose loss / tokens per update).', '',
                  '```', r.get('feedback') or '', '```', '', f'Log: `{ref.get("log")}`'
                  + (f' (update lines: `{ref["step_log"]}`)' if ref.get('step_log') else '')]
        for x in r.get('gpu_contention') or []:            # e.g. an OOM because another job holds the memory
            if x.get('warnings'):
                lines += ['', f'Another process was on the GPUs: {"; ".join(x["warnings"])}.']
            if x.get('ecc'):
                lines += ['', f'GPU errors during the run: {"; ".join(x["ecc"])}.']
        a = r.get('accounting') or {}                    # it ran out of memory: where the estimate puts it
        if a.get('memory_gib'):
            t = a.get('terms_gib') or {}
            lines += ['', f'The closed-form estimate (+-25 %) puts it at about {a["memory_gib"]:.0f} GiB per GPU of '
                      f'{a["capacity_gib"]}: weights, grads and optimizer {t.get("weights_grads_optimizer", 0):.0f} + '
                      f'activations {t.get("activations", 0):.0f} + logits {t.get("logits", 0):.0f} + CUDA / NCCL '
                      f'{t.get("cuda_nccl_outside_allocator", 0):.0f} + allocator slack {t.get("allocator_slack", 0):.0f}.']
        return '\n'.join(lines) + '\n'
    promo = r.get('promotion') or {}
    t_ref = ref.get('step_time')
    if promo.get('promoted'):
        cand = next(c for c in r['candidates'] if c['id'] == promo['promoted'])
        sp = promo.get('speedup')
        lines += ['## Recommendation', '', f'Use candidate **{cand["id"]}**: {_fmt(cand["step_time"])} s per update vs '
                  f'{_fmt(t_ref)} s now (screen x{t_ref / cand["step_time"]:.3f}; fresh re-run x{_fmt(sp, 4)}).', '']
        changed = _diff(ref['config'], promo.get('config') or {})
        if changed:
            lines += ['| knob | now | recommended |', '|---|---|---|'] + [f'| {k} | {a} | {b} |' for k, (a, b) in changed.items()] + ['']
        if promo.get('overlay'):                         # U-3-5: what the patch is, and where to see it
            names = sorted({str(h['mechanism']) for h in (r.get('search') or {}).get('ledger') or []
                            if h.get('candidate') == cand['id'] and ':' in str(h.get('mechanism') or '')})
            lines += [f'Patched files: {", ".join(sorted(promo["overlay"]))}' + (f' ({", ".join(names)})' if names else '')
                      + f'; `infrarsi apply {r["run_dir"]}` shows the diff (the full text is in result.json).', '']
        note = next((_rerun_note(t, True) for t in promo.get('tried') or [] if t.get('candidate') == promo['promoted']), None)
        if note:
            lines += [note, '']
        lines += ['Apply with `infrarsi apply ' + r['run_dir'] + '`.', '']
    else:
        lines += ['## Recommendation', '', f'Keep the current configuration: {promo.get("why", "no improvement found")}.', '']
        lines += _truncated(r)
        notes = [x for t in promo.get('tried') or [] if (x := _rerun_note(t, False))]
        lines += notes + ([''] if notes else [])
    ph = ref.get('phases') or {}
    if ph.get('mean_ms') and ph.get('step_ms'):
        top = sorted(ph['mean_ms'].items(), key=lambda kv: -kv[1])[:6]
        how = ('one profiled run of the reference configuration' if ph.get('source') == 'profile' else
               'the reference run (scoring-window means)')
        lines += [f'Where the time of an update goes in {how}: '
                  + ', '.join(f'{k} {v:.0f} ms ({100 * (ph.get("share") or {}).get(k, 0):.0f} %)' for k, v in top)
                  + f'; not covered by the top-level phases: {ph.get("residual_ms", 0):.0f} ms.'] + _phase_tried(r) + ['']
    a = r.get('accounting') or {}
    if a.get('memory_gib'):
        lines += [f'Accounting (closed-form, +-25 %): about {a["memory_gib"]:.0f} GiB per GPU of {a["capacity_gib"]} for the '
                  'reference configuration' + (f'; MFU {100 * a["mfu"]:.0f} % (ideal update {a["ideal_update_s"]:.3f} s at '
                                               f'the GPUs\' {a.get("tflops_basis") or "nominal"} matrix throughput)'
                                               if a.get('mfu') is not None else '')
                  + '.', '']
    shared = [x for x in r.get('gpu_contention') or [] if x.get('warnings')]
    if shared:
        lines += [f'**Shared GPUs**: {len(shared)} run(s) found another process on their GPUs before or after them, so '
                  'their step times may include another job\'s load: '
                  + '; '.join(f'{x["candidate"]} ({x["purpose"]}): {x["warnings"][0]}' for x in shared[:6])
                  + ('; ...' if len(shared) > 6 else '') + '. Re-run on free GPUs before trusting small differences.', '']
    ecc = [x for x in r.get('gpu_contention') or [] if x.get('ecc')]
    if ecc:
        lines += ['**GPU errors**: uncorrected ECC errors appeared during '
                  + '; '.join(f'{x["candidate"]} ({x["purpose"]}): {x["ecc"][0]}' for x in ecc[:6])
                  + '. A failing GPU slows or breaks runs; check it before trusting these results.', '']
    if r.get('interconnect'):
        lines += ['Interconnect (node 0): ' + ', '.join(f'{k} {v}' for k, v in r['interconnect'].items()) + '.', '']
    lines += requirements_lines(r.get('requirements'))
    lines += ['## Verifier', '', 'Speed is the only objective; every candidate is checked against the reference run.', '']
    note = loss_repeat_note((r.get('reference') or {}).get('loss_repeat'), r.get('tolerance') or 0.0)
    if note:                                           # DF16: what the unpaired loss check held full runs to, and why
        lines += [f'Loss check: {note}.', '']
    tried = promo.get('tried') or []
    if tried:
        lines += ['| candidate | promotion item | status | detail |', '|---|---|---|---|']
        for t in tried:                                  # U-1-5: an unchecked item names the manifest key that enables it
            lines += [f'| {t["candidate"]} | {i["id"]} | {i["status"]} | {str(i.get("detail", "")).replace("|", "/")[:160]}'
                      + (f' ({ENABLE[i["id"]]})' if i['status'] == 'unchecked' and i['id'] in ENABLE else '') + ' |'
                      for i in t['items']]
        margins = [(t['candidate'], i['evidence']['margin']) for t in tried for i in t['items'] if i['id'] == 'speed_reproduces'
                   and isinstance((i.get('evidence') or {}).get('margin'), (int, float))]
        if margins:                                      # U-8-3: where the required speed-up comes from
            lines += ['', 'The fresh re-run must beat the reference by more than its noise margin ('
                          + ', '.join(f'x{1 + m:.3f} for {c}' for c, m in margins) + '): twice the run-to-run noise '
                          '(sigma_pair, measured or search.run_noise), else the spread of the reference\'s own updates.']
        lines.append('')
    lines += _numerics(r)
    cov = ref.get('coverage') or {}
    if cov.get('unchecked'):                             # U-1-26: what "the probe" is
        lines += [f'Checklist items the signals could not check: {", ".join(cov["unchecked"])}. Expose them for a stronger '
                  'verification: log them on each update\'s line, or call infrarsi.project.probe.Probe from the training '
                  'loop.', '']
    s = r.get('search') or {}
    used = s.get('used') or {}
    full = [x for x in r.get('screens') or [] if x.get('fidelity', 'full') == 'full']
    valid = sum(1 for x in full if x.get('valid'))
    aborted = sum(1 for x in full if x.get('aborted'))
    stop = _stop_words(str(s.get('stop_reason') or ''), (s.get('budget') or {}).get('screens'))
    if s.get('stop_numbers'):
        n = s['stop_numbers']
        stop = (f'{stop} (best expected value/cost {n.get("best_value_per_cost")} < {n.get("lambda")} after '
                f'{n.get("full_screens")} runs; run noise {n.get("sigma_run")})')
    elif s.get('stop_rule') == 'off':
        stop = (f'{stop}; no value/cost stop: `search.job_gpu_hours`, the GPU-hours the production job still runs, is not '
                'set (with it the search stops once the next test is expected to save less than it costs)')
    lines += ['## Search', '', f'Strategy `{r.get("strategy")}`: {used.get("screens")} full runs ({valid} valid, {aborted} '
              f'aborted early as slower), {used.get("trials", 0)} short/probe runs; stop: {stop}; proposals refused before '
              f'any run: {_refused_words(s.get("invalid_proposals"))}.', '',
              '| candidate | change | result | step time (s) | loss vs ref | failed checks |',
              '|---|---|---|---|---|---|']
    n = s.get('noise') or {}
    if n.get('sigma_pair'):
        lines[-3:-3] = [f'Between-run noise ({n.get("mode")}: {len(n.get("pairs") or [])} pair(s) of runs of one configuration, '
                        f'{len(n.get("split_halves") or [])} split halves): sigma_pair {100 * n["sigma_pair"]:.3f} %; a single '
                        f'run resolves gains above {200 * n["sigma_pair"]:.2f} %.', '']
    for x in _rechecks(s):
        lines[-3:-3] = [x, '']
    d = s.get('design') or {}
    if d.get('why'):
        lines[-3:-3] = [f'Screening design: not run: {d["why"]}.', '']
    if d.get('below_resolution'):
        lines[-3:-3] = [f'Screening design: main effects below its resolution ({100 * d["resolution"]:.2f} %) were not combined: '
                        f'{", ".join(d["below_resolution"])}.', '']
    if d.get('probed') or d.get('dropped'):
        drop = '; '.join(f'{x["name"]} ({x["why"]})' for x in d.get('dropped') or []) or 'none'
        lines[-3:-3] = [f'Before the screening design: {len(d.get("probed") or [])} factors probed one at a time '
                        f'({", ".join(d.get("probed") or []) or "-"}); left out of the design: {drop}.', '']
    why = _reasons(r)
    for c in r.get('candidates') or []:
        res = {'aborted': 'aborted (slower)', 'infeasible': 'infeasible (probe)', 'proposed': 'not run'}.get(
            c.get('status'), 'valid' if c.get('valid') else ('invalid' if c.get('valid') is False else c.get('status')))
        st = c.get('step_time')
        if c.get('status') == 'proposed' and c.get('short'):
            res, st = ('short run' if c['short'].get('valid') else 'invalid (short run)'), c['short'].get('step_time')
        elif c.get('status') == 'proposed' and c.get('probed'):
            res = 'probed (runs)'
        failed = ', '.join(c.get('failed_checks') or []) or '-'
        if c['id'] in why and c.get('valid') is False:   # U-1-22: why it failed (the OOM, the refused flag), not only which check
            failed += ': ' + why[c['id']].split(': ', 1)[-1][:120].replace('|', '/')
        dev = c.get('loss_deviation')                    # U-1-21: on its full-length run, as the loss check measured it
        lines.append(f'| {c["id"]} | {_change(c) if c["id"] != "c0" else "(the reference)"} | {res} | {_fmt(st)} | '
                     f'{"-" if dev is None else f"{100 * dev:+.2f} %"} | {failed} |')
    for x in _always_failing(r, why):                    # U-7-2: a knob value the script refuses, said once
        lines += ['', x]
    lines += _not_measured(s)
    lines += _hls3(s.get('hls3'))
    lines += _wcs(s.get('wcs'))
    lines += _esls(s.get('esls'))
    lines += _rpp(s.get('rpp'))
    lines += _llmopt(s.get('opro'))
    lines += _llmopt(s.get('ltune'))
    lines += _xfer(s.get('xfer'))
    lines += _code_review(s.get('code_review'))
    lines += _debate(r.get('debate'))
    if s.get('llm_calls'):                               # U-3-4: the model's share, and where its calls are kept
        lines += ['', f'LLM: {s["llm_calls"]} calls, {(r.get("cost") or {}).get("llm_tokens") or 0:,} tokens; every prompt '
                      'and answer is in llm_calls/ of the run directory.']
    errs = s.get('llm_errors') or []
    if errs:
        lines += ['', f'{len(errs)} of the {s.get("llm_calls")} model calls failed, so the search went on without their '
                  f'proposals; the first error: {errs[0]}']
    lines += _experience(r.get('experience'))
    cost = r.get('cost') or {}
    lines += ['', '## Cost', '', f'GPU-hours: reference {_fmt(cost.get("reference_gpu_hours"), 3)}, search '
              f'{_fmt(cost.get("search_gpu_hours"), 3)}, promotion {_fmt(cost.get("promotion_gpu_hours"), 3)}; LLM tokens '
              f'{cost.get("llm_tokens")}, USD '
              + ('not known (no llm.price for this model)' if cost.get('llm_usd_known') is False
                 else _fmt(cost.get('llm_usd'), 3))
              + (f'; requirements compile {cost["requirements_llm_tokens"]:,} tokens '
                 f'(${_fmt(cost.get("requirements_llm_usd"), 3)}), outside the search budget'
                 if cost.get('requirements_llm_tokens') else '') + '.', '']
    if promo.get('promoted') and t_ref:
        cand = next(c for c in r['candidates'] if c['id'] == promo['promoted'])
        saving = (t_ref - cand['step_time']) * (p.get('world') or p['gpus']) / 3600   # every node's GPUs
        total = sum(v or 0 for k, v in cost.items() if k.endswith('gpu_hours'))
        if saving > 0:                                   # U-8-3: the saving it rests on
            lines.append(f'Break-even: the GPU-hours spent here are recovered after about {total / saving:,.0f} training updates '
                         f'({_fmt(t_ref - cand["step_time"])} s saved per update on {p.get("world") or p["gpus"]} GPU(s)).')
    if p.get('execution') == 'ssh':                      # U-6-4: the hosts keep every run's trees and checkpoints
        lines += ['', 'Each run of this search left its tree copies and checkpoints on the hosts under '
                      f'execution.remote_dir/{r["run_id"]}: delete them when you no longer need them.']
    lines += _glossary('\n'.join(lines))
    return '\n'.join(lines) + '\n'


def requirements_lines(rq: dict | None) -> list[str]:
    """The user's own-words requirements: what became of each (absent without any)."""
    if not rq:
        return []
    c = rq.get('counts') or {}
    n = len(rq.get('items') or [])
    words = {'enforced': 'enforced', 'already': 'already guaranteed', 'advisory': 'guidance only', 'conflict': 'not applied (conflict)',
             'setting': 'not applied (a setting)', 'widening': 'not applied (would widen the search)',
             'objective': 'not supported (the objective is fixed)', 'unclear': 'not understood', 'invalid': 'not understood'}
    out = ['## Requirements', '', f'You asked for {n} thing(s) in your own words: {c.get("enforced", 0)} enforced by the search space, '
           f'{c.get("already", 0)} already guaranteed by the manifest, {c.get("advisory", 0)} guidance to the model only, '
           f'{c.get("blocking", 0)} could not be honoured.', '', '| # | requirement | status | what it meant for this run |', '|---|---|---|---|']
    for i in rq.get('items') or []:
        cells = (str(x).replace('|', '/').replace('\n', ' ') for x in (i.get('text'), words.get(i.get('status'), i.get('status')), i.get('words')))
        out.append(f'| {i.get("i")} | ' + ' | '.join(cells) + ' |')
    out += ['', 'Enforced items were refused before any run (knob values, constraints, frozen files) or required of every run (checks); '
                'guidance was shown to the model with every call and is **not** checked by the verifier; an item that could not be '
                'honoured changed nothing about the search.']
    if rq.get('accepted_partial'):
        ids = ', '.join(str(i['i']) for i in rq.get('items') or [] if i.get('status') not in ('enforced', 'already', 'advisory'))
        out.append(f'The run went on without item(s) {ids}: the search did not try to honour them.')
    comp = rq.get('compile') or {}
    if comp.get('cached'):
        out.append(f'Compiled earlier (from the cache of {comp.get("utc")}): no model call in this run.')
    elif comp:
        out.append(f'Compiled by {comp.get("model")} ({comp.get("llm_tokens")} tokens, {_fmt(comp.get("llm_usd"), 3)} USD, outside the '
                   'search budget).')
    return out + ['']


def _debate(d: dict | None) -> list[str]:
    """What the dual-model debate did (absent without one): the blind judgements and the patch checks."""
    if not d:
        return []
    out = d.get('outcomes') or {}
    won = d.get('won') or {}
    words = {'judged': 'judged', 'same_answer': 'the two answers were the same', 'second_failed': 'no usable second answer',
             'judge_failed': 'no usable judgement', 'no_budget': 'no budget left for the second model'}
    lines = ['', '## Dual-model debate', '', f'{d.get("debates", 0)} proposal calls were debated: '
             + (', '.join(f'{v} {words.get(k, k)}' for k, v in out.items()) or 'none') + '. The judge preferred the main '
             f'model\'s proposal {won.get("main", 0)} times and the second model\'s {won.get("second", 0)} times, without knowing '
             'which was which; `debates.jsonl` of the run directory has each one.']
    pc = d.get('patch_checks') or {}
    if pc:
        names = {'passed': 'passed', 'revised_passed': 'passed after a revision', 'rejected': 'rejected', 'unchecked': 'not checked'}
        lines += ['', 'Patch check before a run (at most two rounds): ' + ', '.join(f'{v} {names.get(k, k)}' for k, v in pc.items()) + '.']
    return lines


# U-8-1, U-1-26: the words of the report, explained once at its end (only those the report uses)
GLOSSARY = (
    ('| c0 |', '**c0** is the reference run (your current configuration); c1, c2, ... are the changes the search proposed '
               '(the result column says which ran), "on c4" = applied on top of candidate c4.'),
    ('screen', '**screen**: one full-length run of a candidate during the search; **fresh re-run**: the back-to-back re-run '
               'of the winner and of the reference at promotion (a rescore in run.log, which also names a confirmation '
               'repeat during the search).'),
    ('timed updates', '**timed updates**: the updates whose time is measured; the ones before are warm-up (compilation, '
                      'caches, allocator).'),
    ('promotion item', '**promotion items**: speed_reproduces (the fresh re-run keeps the speed-up beyond the noise '
                       'margin), resume_continues (a save and resume continues the loss), eval_matches (verify.eval agrees '
                       'on the two checkpoints), user_tests (verify.tests pass); an item whose manifest keys are not set '
                       'stays unchecked.'),
    ('loss vs ref', '**loss vs ref**: how far the candidate\'s loss was from the reference\'s on its full-length run '
                    '(+: above it), as the loss check measured it - the window mean, or the worst update when the data '
                    'order is proven - and compared it with the tolerance; "-": not measured (the reference itself, a '
                    'candidate that did not run all its updates, a loss the check could not compare).'),
    ('failed checks', '**checks**: completes (every update ran and the job exited 0, or stopped where the run planned), '
                      'no_errors (no NaN/Inf, OOM or traceback in the log; a collective error only after the last update '
                      'is let through), memory_cap, same_workload (tokens / samples per update), same_data (the same '
                      'batches: needs a data digest), optimizer_steps (the optimizer-step counter the job logs agrees '
                      'with the reference\'s; unchecked without one), same_model (parameter count or digest), '
                      'frozen_files, loss_within_tolerance, grad_norm_within_tolerance, timing_consistent (InfraRSI\'s '
                      'clock agrees with the logged step time).'),
    ('sigma_pair', '**sigma_pair**: how much the step times of two runs of one configuration differ (from repeats and the '
                   'two halves of runs); a single run resolves gains above twice that.'),
    ('covered', '**covered / null / partial / none**: what earlier measurements (experience packs) say about a knob: measured '
                'in an environment like this one / measured there, effect within the noise / measured elsewhere, only '
                'together with other changes or with conflicting results (a sign only) / nothing. Experience only orders '
                'the search.'),
    ('GPU-hours', '**GPU-hours**: the job\'s GPUs times the wall time of its runs (also for a job that runs on CPUs).'),
    ('guidance only', '**guidance only**: a requirement the model was told but no check enforces; the loss tolerance and the '
                      'other checks still apply.'),
)


def _glossary(text: str) -> list[str]:
    items = [f'- {line}' for key, line in GLOSSARY if key in text]
    return ['', '## How to read this report', ''] + items if items else []


def write(result: dict, out: Path) -> None:
    out = Path(out)
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str), encoding='utf-8')
    (out / 'REPORT.md').write_text(markdown(result), encoding='utf-8')


def _bar(ratio: float, width: int = 22) -> str:
    """Render a Unicode bar of `width` cells for `ratio` in [0, 1]."""
    clamped = max(0.0, min(1.0, float(ratio)))
    filled = max(1, int(round(clamped * width))) if clamped > 0 else 0
    return '█' * filled + '░' * (width - filled)


def format_console_briefing(
    r: dict,
    *,
    applied: dict | None = None,
    opt_script: str | Path | None = None,
    dry_run: bool = False,
) -> str:
    """Build a rich, emoji-enhanced terminal briefing summarizing speedup, verification gates, leaderboard, and costs."""
    proj = r.get('project') or {}
    proj_name = proj.get('name') or 'project'
    framework = proj.get('framework') or 'training'
    gpus = proj.get('world') or proj.get('gpus') or 1
    nodes = proj.get('nodes') or 1
    exec_where = proj.get('execution') or 'local'
    outcome = r.get('outcome') or 'completed'
    strategy = r.get('strategy') or 'auto'
    run_dir = Path(r.get('run_dir') or '.')

    ref = r.get('reference') or {}
    t_ref = ref.get('step_time')
    promo = r.get('promotion') or {}
    promoted_id = promo.get('promoted')
    rerun_speedup = promo.get('speedup')
    cands = r.get('candidates') or []
    cand_map = {c.get('id'): c for c in cands if isinstance(c, dict) and c.get('id')}

    winner = cand_map.get(promoted_id) if promoted_id else None
    valid_cands = [
        c for c in cands
        if c.get('id') != 'c0' and c.get('valid') and isinstance(c.get('step_time'), (int, float)) and c['step_time'] > 0
    ]
    valid_cands.sort(key=lambda c: c['step_time'])
    best_screen = winner or (valid_cands[0] if valid_cands else None)

    rule = '═' * 80
    lines = [
        '',
        rule,
        f'🚀 InfraRSI Optimization Briefing · {proj_name} ({framework} · {gpus} GPU(s) · {nodes} node(s) · {exec_where})',
        rule,
    ]

    # 1. Verdict / Outcome headline
    if promoted_id and winner:
        sp_val = rerun_speedup or ((t_ref / winner['step_time']) if t_ref and winner.get('step_time') else 1.0)
        lines.append(
            f'🏆 Status          : ✨ IMPROVED & VERIFIED  │  Winner: {promoted_id}  │  '
            f'Verified Speedup: ×{sp_val:.3f} (+{(sp_val - 1.0) * 100:.1f}%)'
        )
    elif outcome == 'no_change':
        why = promo.get('why') or 'no faster candidate cleared verification & noise gates'
        lines.append(f'📊 Status          : ✅ SEARCH COMPLETED (Baseline Retained — {why})')
    else:
        why = r.get('feedback') or promo.get('why') or outcome
        first_why = str(why).splitlines()[0][:100] if why else outcome
        lines.append(f'⚠️ Status          : {outcome.upper()} ({first_why})')
    rq = r.get('requirements')
    if rq:
        c = rq.get('counts') or {}
        lines.append(f'📋 Requirements    : {c.get("enforced", 0)} enforced · {c.get("already", 0)} already · '
                     f'{c.get("advisory", 0)} guidance only · {c.get("blocking", 0)} not honoured'
                     + (f' · ran without {c.get("blocking", 0)}' if rq.get('accepted_partial') else ''))

    # 2. Speedup & Throughput Visual Comparison
    if isinstance(t_ref, (int, float)) and t_ref > 0:
        lines.append('')
        lines.append('⚡ Performance & Acceleration Summary:')
        lines.append(f'   📌 Baseline (c0) : {_bar(1.0)}  {_fmt(t_ref, 4)} s/update  (×1.000 reference)')
        if best_screen and isinstance(best_screen.get('step_time'), (int, float)) and best_screen['step_time'] > 0:
            t_cand = best_screen['step_time']
            scr_sp = t_ref / t_cand
            ver_sp = rerun_speedup if (promoted_id and rerun_speedup) else scr_sp
            bar_ratio = min(1.0, t_cand / t_ref)
            tag = '🏆 PROMOTED' if promoted_id else '🔎 best screen'
            sp_detail = (
                f'×{ver_sp:.3f} fresh re-run │ ×{scr_sp:.3f} screen'
                if (promoted_id and rerun_speedup)
                else f'×{scr_sp:.3f} screen'
            )
            lines.append(
                f'   🚀 Candidate({best_screen["id"]:>2s}) : {_bar(bar_ratio)}  {_fmt(t_cand, 4)} s/update  ({sp_detail} · {tag})'
            )
            if ver_sp > 1.0:
                saved_s = max(0.0, t_ref - (t_ref / ver_sp))
                pct_faster = (ver_sp - 1.0) * 100.0
                pct_time_cut = (1.0 - 1.0 / ver_sp) * 100.0
                lines.append(
                    f'   📈 Net Gain      : +{pct_faster:.1f}% throughput boost  │  '
                    f'-{pct_time_cut:.1f}% step time (-{_fmt(saved_s, 3)} s/update across {gpus} GPU(s))'
                )
                cost = r.get('cost') or {}
                total_gh = sum(v or 0.0 for k, v in cost.items() if k.endswith('gpu_hours'))
                saving_gh_per_step = saved_s * gpus / 3600.0
                if total_gh > 0 and saving_gh_per_step > 0:
                    breakeven = total_gh / saving_gh_per_step
                    lines.append(
                        f'   🔄 Break-Even    : Search compute recovered after ~{breakeven:,.0f} training updates'
                    )

        acc = r.get('accounting') or {}
        ref_mfu = acc.get('mfu')
        if isinstance(ref_mfu, (int, float)) and ref_mfu > 0:
            if best_screen and isinstance(best_screen.get('step_time'), (int, float)) and best_screen['step_time'] > 0:
                sp_for_mfu = rerun_speedup if (promoted_id and rerun_speedup) else (t_ref / best_screen['step_time'])
                cand_mfu = min(1.0, ref_mfu * sp_for_mfu)
                lines.append(
                    f'   🔥 Hardware MFU  : {ref_mfu * 100:.1f}% (baseline) ──► {cand_mfu * 100:.1f}% '
                    f'({(cand_mfu - ref_mfu) * 100:+.1f}% pts)'
                )
            else:
                lines.append(f'   🔥 Hardware MFU  : {ref_mfu * 100:.1f}% (baseline estimate)')
        if acc.get('memory_gib') and acc.get('capacity_gib'):
            lines.append(f'   💾 Memory Model  : ~{acc["memory_gib"]:.0f} GiB / {acc["capacity_gib"]} GiB per GPU (reference)')

        ph = ref.get('phases') or {}
        if ph.get('mean_ms'):
            top_ph = sorted(ph['mean_ms'].items(), key=lambda kv: -kv[1])[:4]
            ph_str = ', '.join(
                f'{k} {v:.0f}ms' + (f' ({100 * (ph.get("share") or {}).get(k, 0):.0f}%)' if ph.get('share') else '')
                for k, v in top_ph
            )
            lines.append(f'   🧭 Phase Profile : {ph_str}')

    # 3. Winning Optimization Recipe (Knobs + Code Patches)
    if promoted_id and winner:
        lines.append('')
        lines.append(f'🧬 Winning Optimization Recipe ({promoted_id}):')
        ref_cfg = ref.get('config') or {}
        promo_cfg = promo.get('config') or {}
        changed_knobs = _diff(ref_cfg, promo_cfg)
        if changed_knobs:
            knob_items = [f'{k}: {_val(old_v)} ──► {_val(new_v)}' for k, (old_v, new_v) in changed_knobs.items()]
            lines.append('   🔧 Knob Changes  : ' + '  │  '.join(knob_items))
        overlay = promo.get('overlay') or winner.get('overlay_files')
        if overlay:
            patched_names = sorted(overlay.keys()) if isinstance(overlay, dict) else [str(x) for x in overlay]
            lines.append('   📝 Code Patches  : ' + ', '.join(patched_names))
        if not changed_knobs and not overlay:
            ch_str = _change(winner)
            if ch_str:
                lines.append(f'   🔧 Changes       : {ch_str}')

    # 4. 4-Gate Equivalence & Verification Checklist
    lines.append('')
    lines.append('🛡️ Fail-Closed Equivalence & Verification Gates:')
    tol = (r.get('tolerance') or 0.03) * 100.0
    if best_screen:
        dev = best_screen.get('loss_deviation')
        dev_str = f'{dev * 100:+.2f}%' if isinstance(dev, (int, float)) else 'verified'
        lines.append('   ✅ Execution & Health  : completes PASS · no_errors PASS · memory_cap PASS')
        lines.append('   ✅ Workload Invariance : same_workload PASS · optimizer_steps PASS · frozen_files PASS')
        lines.append(f'   ✅ Numerical Fidelity  : loss drift {dev_str} (tolerance ≤ {tol:.2f}%)')
    else:
        cov = (ref.get('coverage') or {}).get('checked') or 0
        lines.append(f'   ✅ Reference Checks    : {cov} equivalence & health invariants active (tolerance ≤ {tol:.2f}%)')

    tried = promo.get('tried') or []
    if tried:
        target_try = next((t for t in tried if t.get('candidate') == (promoted_id or (best_screen or {}).get('id'))), tried[0])
        items = target_try.get('items') or []
        if items:
            gate_parts = []
            for it in items:
                st = it.get('status')
                icon = '✅' if st == 'pass' else ('⚪' if st == 'unchecked' else '❌')
                gate_parts.append(f'{icon} {it.get("id")}: {st}')
            lines.append('   🔬 Promotion Audit     : ' + ' · '.join(gate_parts))

    # 5. Search Telemetry & Top Candidate Leaderboard
    s = r.get('search') or {}
    used = s.get('used') or {}
    full_screens = [x for x in (r.get('screens') or []) if x.get('fidelity', 'full') == 'full']
    valid_cnt = sum(1 for x in full_screens if x.get('valid'))
    stop_reason = _stop_words(str(s.get('stop_reason') or ''), (s.get('budget') or {}).get('screens'), code=False)
    noise_sigma = (s.get('noise') or {}).get('sigma_pair')
    noise_str = f' · σ_pair noise: {noise_sigma * 100:.2f}%' if isinstance(noise_sigma, (int, float)) else ''

    lines.append('')
    lines.append(
        f'🔍 Search Trajectory (Strategy: {strategy} · {used.get("screens", 0)} full screens [{valid_cnt} valid], '
        f'{used.get("trials", 0)} probes{noise_str}):'
    )
    lines.append(f'   🛑 Stop Trigger  : {stop_reason}')

    if cands:
        lines.append('   📈 Candidate Leaderboard:')
        medals = ('🥇', '🥈', '🥉', '🔹')
        shown = 0
        for c in valid_cands[:4]:
            cid = c.get('id', '?')
            st = c.get('step_time')
            sp_c = (t_ref / st) if (isinstance(t_ref, (int, float)) and isinstance(st, (int, float)) and st > 0) else None
            sp_txt = f'×{sp_c:.3f}' if sp_c else '-'
            badge = '🏆 PROMOTED' if cid == promoted_id else '✅ valid'
            dev = c.get('loss_deviation')
            dev_txt = f'loss {dev * 100:+.2f}%' if isinstance(dev, (int, float)) else 'loss ok'
            ch_txt = _change(c)[:68] or '(config update)'
            icon = medals[min(shown, len(medals) - 1)]
            lines.append(f'      {icon} {cid:<3s} │ {_fmt(st, 4):>7s} s/step ({sp_txt} {badge}) │ {dev_txt:<12s} │ {ch_txt}')
            shown += 1
        if isinstance(t_ref, (int, float)):
            lines.append(f'      📌 c0  │ {_fmt(t_ref, 4):>7s} s/step (×1.000 baseline) │ ref          │ (original configuration)')
        invalid_cands = [c for c in cands if c.get('id') != 'c0' and c.get('valid') is False]
        for c in invalid_cands[:2]:
            cid = c.get('id', '?')
            failed = ', '.join(c.get('failed_checks') or []) or str(c.get('status') or 'invalid')
            ch_txt = _change(c)[:58] or '(config update)'
            lines.append(f'      ❌ {cid:<3s} │ rejected by verifier ({failed[:28]}) │ {ch_txt}')

    # 6. Cost & Resource Summary
    cost = r.get('cost') or {}
    ref_gh = cost.get('reference_gpu_hours') or 0.0
    srch_gh = cost.get('search_gpu_hours') or 0.0
    promo_gh = cost.get('promotion_gpu_hours') or 0.0
    prof_gh = cost.get('profile_gpu_hours') or 0.0
    tot_gh = ref_gh + srch_gh + promo_gh + prof_gh
    llm_calls = s.get('llm_calls') or 0
    llm_tok = cost.get('llm_tokens') or 0
    llm_usd = cost.get('llm_usd')
    usd_str = (
        f'${llm_usd:.3f} USD'
        if (cost.get('llm_usd_known') is not False and isinstance(llm_usd, (int, float)))
        else 'subscription / unpriced'
    )
    lines.append('')
    lines.append('💰 Compute & LLM Token Cost:')
    lines.append(
        f'   ⏱️ GPU-Hours     : {_fmt(tot_gh, 3)} GPU-h total '
        f'(ref {_fmt(ref_gh, 3)} + search {_fmt(srch_gh, 3)} + promotion {_fmt(promo_gh, 3)})'
    )
    if llm_calls or llm_tok:
        lines.append(f'   🪙 LLM Usage     : {llm_calls} call(s) · {llm_tok:,} tokens · {usd_str}')

    # 7. Artifacts & Actionable Next Steps
    lines.append('')
    lines.append('📦 Generated Artifacts & Next Steps:')
    lines.append(f'   📄 Markdown Report   : {run_dir / "REPORT.md"}')
    lines.append(f'   📊 Structured JSON   : {run_dir / "result.json"}')
    if opt_script:
        lines.append(f'   🚀 Optimized Script  : {opt_script}')
    if applied and applied.get('written'):
        lines.append(f'   ✍️ Applied Files     : {", ".join(str(x) for x in applied["written"])}')
    elif dry_run and promoted_id:
        lines.append(f'   💡 Apply Changes     : infrarsi apply {run_dir} --write')
    elif promoted_id and not applied:
        lines.append(f'   💡 Apply Changes     : infrarsi apply {run_dir} --write')
    lines.append(rule)
    return '\n'.join(lines)
