"""ENV-B (OptiCo-public workloads W1 qwen2.5-7B, W3 qwen2.5-0.5B, W2 llama3-8B): design §3.6 / §0.3 with
AMENDMENT_EXT_002 Q1 applied.

Per arm and workload: tflops_locked (own locked pair), speedup_vs_default (own pair), tflops_best_search,
trials_to_first_improvement (first new trial with T >= (1 + margin_w) x the arm's own screen-0 default), n_trials,
n_failed_runs (OOM / crash / timeout), n_invalid_prelaunch, n_guard_rejections, invalid_accept_locked, CTG(+5 %),
AUC_budget, costs (+ n_refusals for B-NAT, EXT_001 Q5).
margin_w: ext.optico.margin.compute over the protocol runs of the ledger only (abandoned runs never enter S_w), written
as margin_<W>.json. Labels: trajectory.compare on the arms' raw locked TFLOPS (AMENDMENT_EXT_002: y = ln(T_locked /
T_B-DEF), a common reference per workload, so d = ln(T_locked(a) / T_locked(b))); the ratio of the two pairs' default
halves is printed as a drift diagnostic. D23: an arm whose runs are on another node than the workload's B-DEF (B-ISW W1 on
WINH1) uses a same-node ratio, y = ln(T_locked / its locked pair's default half), also for its search-phase y; the
node factor of the default is printed. D20: a locked pair recorded `skipped` (the selection is the workload default) has
locked value = the start reference (B-DEF), y = 0, InvalidAccept 0. Sensitivity (reported, never applied): the label
under the raw ratio T_locked(a) / T_locked(b) and under the paired ratio speedup_vs_default(a) / speedup_vs_default(b).
ENV-B `advantage` needs >= 2 workloads `advantage` and 0 `disadvantage` (`invalid` counts as a disadvantage, §0.3).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from ..common import trajectory as TJ
from ..optico import margin as OM
from ..optico import params as PM
from ..optico import task as OT
from ..optico import workloads as OW
from . import render as R
from .cells import cell, missing, na, ok, pending, val
from .records import (ScoreTask, Store, accounting_d26, findings, finite, harness_refused, in_ids, not_run_reason,
                      refusal_kind, run_cost, run_id_of, src, trials_used)

DIR = 'optico'
REFERENCE_ARMS = ('B-DEF', 'B-ED')
SEARCH_ARMS = ('B-NAT', 'B-IS', 'B-RND', 'B-ISW', 'B-IS2', 'B-IS2-noexp')
OPTIONAL_ARMS = ('B-ISW', 'B-IS2', 'B-IS2-noexp')   # listed only where the ledger names them (B-IS2: phase E of 10-24)
ARMS = REFERENCE_ARMS + SEARCH_ARMS
LOCKED_ARMS = ('B-ED', 'B-NAT', 'B-IS', 'B-RND', 'B-ISW', 'B-IS2', 'B-IS2-noexp')
MAIN_PAIRS = (('B-IS', 'B-NAT'), ('B-ISW', 'B-NAT'), ('B-IS2', 'B-NAT'), ('B-IS2', 'B-IS'), ('B-IS2', 'B-IS2-noexp'))
RND_PAIRS = (('B-IS', 'B-RND'), ('B-NAT', 'B-RND'), ('B-ISW', 'B-RND'), ('B-IS2', 'B-RND'))
BAD_FOR_IS = ('disadvantage', 'invalid')


def workloads(ledger: dict) -> list[str]:
    return [k for k, v in (ledger.get('ENV-B') or {}).items() if re.fullmatch(r'W\d+', k) and isinstance(v, dict)]


def score_task(w: str, arm: str) -> ScoreTask:
    b = OT.make_task(w, arm).budget
    return ScoreTask(id=OW.get(w).task_id, env='optico', arm=arm, margin=dict(OT.MARGIN), tau=OT.TAU,
                     budget={'screens': b.get('screens'), 'llm_tokens': b.get('llm_tokens') or None,
                             'wall_seconds': b.get('wall_seconds')})


def tflops(r: dict | None) -> float | None:
    v = (((r or {}).get('metric') or {}).get('raw') or {}).get('tflops_per_gpu')
    return float(v) if finite(v) and v > 0 else None


def _key(p, w):
    """X3's identity key of a configuration (optico.arms.is_default_selection): the knobs normalized as the validator
    launches them (AC_LAYERS -> 1 under AC none / sel, overlap flags off under offload), typed and canonical."""
    if not isinstance(p, dict):
        return None
    try:
        norm, errs, _ = PM.validate_optico_params(dict(p), OW.get(w))
        return PM.params_key(PM.typed(norm if not errs else p, strict=True))
    except (ValueError, TypeError, AttributeError, KeyError):
        return None


def _same_params(a, b, w) -> bool:
    ka = _key(a, w)
    return ka is not None and ka == _key(b, w)


def _is_default(p, w) -> bool:
    return _same_params(p, OW.get(w).defaults, w)


def protocol_receipts(store: Store, root: Path, ledger: dict, abandoned_ids: set) -> list[dict]:
    """Every receipt of the ledger's ENV-B protocol runs (arms and locked pairs, all workloads), with `_path` for
    ext.optico.margin; abandoned runs are skipped."""
    out = []
    for w in workloads(ledger):
        spec = ledger['ENV-B'][w]
        ids = [run_id_of(spec.get(a)) for a in ARMS] + [run_id_of(x) for x in (spec.get('locked') or {}).values()]
        for rid in ids:
            if not rid or in_ids(rid, abandoned_ids):
                continue
            d = root / DIR / rid
            for att, r in store.receipts(d).items():
                out.append({**r, '_path': str(d / 'receipts' / f'{att}.json')})
    return out


def _load(store: Store, root: Path, spec, ab_ids: set = frozenset()) -> dict:
    """spec = the ledger value: a run id, 'not run: <reason>' (sanctioned, final) or null (not started: pending)."""
    why = not_run_reason(spec)
    if why:
        return {'status': 'not run', 'run_id': None, 'why': why}
    rid = run_id_of(spec)
    if not rid:
        return {'status': 'not started', 'run_id': None, 'why': 'no run id in the ledger yet'}
    if in_ids(rid, ab_ids):
        return {'status': 'abandoned (ledger conflict)', 'run_id': rid}
    d = root / DIR / rid
    if not d.is_dir():
        return {'status': 'not started', 'run_id': rid}
    res = store.json(d / 'result.json') if (d / 'result.json').is_file() else None
    tr = store.json(d / 'trace.json') if (d / 'trace.json').is_file() else None
    recs = store.receipts(d)
    finished = isinstance(tr, dict) and tr.get('stop_reason') and tr.get('ended_utc')
    if isinstance(res, dict) and res.get('schema') in ('infrarsi-ext-optico-native-v1', 'infra' + 'smith-ext-optico-native-v1'):
        finished = finished and res.get('status') == 'done'
    return {'status': 'finished' if finished else 'running', 'run_id': rid, 'dir': d, 'result': res, 'trace': tr,
            'receipts': recs, 'store': store}


def _single(store: Store, root: Path, spec, purpose: str, problems: list, w: str, ab_ids: set = frozenset()) -> dict:
    """B-DEF / B-ED: the one protocol trial of the run."""
    a = _load(store, root, spec, ab_ids)
    rid = a.get('run_id')
    if a['status'] not in ('finished', 'running'):
        return {'status': a['status'], 'run_id': rid, 'why': a.get('why'), 'T': missing(a['status'], a.get('why'))}
    rs = [r for r in a['receipts'].values() if r.get('purpose') == purpose]
    if len(rs) != 1:
        if len(rs) > 1:
            problems.append(f'ENV-B {w} {rid}: {len(rs)} {purpose} receipts in a single-trial run')
        return {'status': 'running' if not rs else 'finished', 'run_id': rid,
                'T': pending(f'{len(rs)} {purpose} receipts in {rid}')}
    r = rs[0]
    if r.get('task_id') != OW.get(w).task_id:
        problems.append(f'ENV-B {w} {rid}: receipt task {r.get("task_id")} is not {OW.get(w).task_id}')
    s = src(rid, r)
    return {'status': 'finished', 'run_id': rid, 'receipt': r, 'node': r.get('node'),
            'T': cell(tflops(r), s) if r.get('valid') and tflops(r) else na('trial invalid'),
            'valid': cell(bool(r.get('valid')), s), 'iter_s': cell((r.get('metric') or {}).get('value'), s),
            'params': cell(PM.format_params(PM.typed(r.get('params') or {}, strict=False)) if r.get('params') else None, s),
            'gpu_hours': cell(((r.get('cost') or {}).get('gpu_seconds') or 0) / 3600.0, s)}


def _selection(a: dict) -> dict | None:
    tr = a.get('trace') or {}
    if isinstance(tr.get('selected_config'), dict):
        return tr['selected_config']
    return None


def _search(a: dict, w: str, arm: str, margin_w: float | None, t_ref: dict, iter_ref: float | None, problems: list) -> dict:
    """Search-phase numbers; t_ref / iter_ref = the y reference (the workload's B-DEF; D23: the arm's same-node default)."""
    tr, recs, rid = a['trace'], a['receipts'], a['run_id']
    scr = list(tr.get('screens') or [])
    out = {}
    Ts, failed, guard, prelaunch, infra = [], {'oom': 0, 'timeout': 0, 'crash': 0}, 0, 0, {}
    for i, s in enumerate(scr):
        r = recs.get(s.get('attempt_id'))
        refused = harness_refused(r)                              # D24: a pre-launch refusal is not an invalid proposal
        if s.get('sanitizer_rejected') or ((r or {}).get('not_launched') and not refused):
            prelaunch += 1
            Ts.append(None)
            continue
        if r is None:
            problems.append(f'ENV-B {w} {arm}: screen {i} ({s.get("candidate_id")}) has no receipt {s.get("attempt_id")}')
            Ts.append(None)
            continue
        if bool(r.get('valid')) != bool(s.get('valid')) or (s.get('valid') and (r.get('metric') or {}).get('value') != s.get('step_time')):
            problems.append(f'ENV-B {w} {arm}: screen {i} ({s.get("candidate_id")}) differs from its receipt {s.get("attempt_id")}')
        if refused:                                              # refused / lost by our harness, not the candidate
            k = refusal_kind(r)
            infra[k] = infra.get(k, 0) + 1
            Ts.append(None)
            continue
        Ts.append(tflops(r) if r.get('valid') else None)
        guard += bool(r.get('guard_rejections'))
        if not r.get('judge_valid'):
            timed_out = bool(((r.get('runner') or {}).get('status') or {}).get('timed_out'))
            failed['oom' if r.get('oom') else 'timeout' if timed_out else 'crash'] += 1
    srcs = [src(rid, recs.get(s.get('attempt_id'))) for s in scr]
    inv = tr.get('invalid_proposals') if isinstance(tr.get('invalid_proposals'), dict) else {}
    prelaunch += sum(int(v) for v in inv.values())
    out['n_trials'] = cell(len(scr) - sum(1 for s in scr if s.get('sanitizer_rejected')) - sum(infra.values()), srcs,
                           note='launched trials incl. the default start')
    out['n_runner_errors'] = cell(infra, srcs)
    out['n_failed_runs'] = cell(sum(failed.values()), srcs, note=', '.join(f'{k} {v}' for k, v in failed.items() if v) or None)
    out['n_invalid_prelaunch'] = cell(prelaunch, [f'{rid}/trace.json'],
                                      note=', '.join(f'{k} {v}' for k, v in inv.items()) or None)
    out['n_guard_rejections'] = cell(guard, srcs)
    good = [(t, i) for i, t in enumerate(Ts) if t]
    if good:
        t, i = max(good)
        out['tflops_best_search'] = cell(t, srcs[i])
    else:
        out['tflops_best_search'] = na('no valid trial')
    t0, i0 = (Ts[0], 0) if Ts else (None, None)
    if not t0 and arm == 'B-NAT':                       # OptiCo RETRYs a failed default row: its first valid retry
        for i, s in enumerate(scr):
            nat = (recs.get(s.get('attempt_id')) or {}).get('native') or {}
            if nat.get('kind') == 'retry' and nat.get('retry_of') == 0 and Ts[i]:
                t0, i0 = Ts[i], i
                break
    r0 = recs.get((scr[0] if scr else {}).get('attempt_id')) or {}
    why0 = (f'screen 0 (the default start) was refused by our harness ({refusal_kind(r0)}; D24 when prelaunch): no own '
            'default value' if harness_refused(r0) else 'screen 0 (the default start) invalid')
    out['T_own_default'] = cell(t0, srcs[i0]) if t0 else na(why0)
    if margin_w is None:
        out['trials_to_first_improvement'] = pending('margin_w pending')
    elif not t0:
        out['trials_to_first_improvement'] = na(why0)
    else:
        hit = next((i for i, t in enumerate(Ts) if i > i0 and t and t >= (1.0 + margin_w) * t0), None)
        out['trials_to_first_improvement'] = cell(hit if hit is not None else f'none in {len(scr) - 1}',
                                                  srcs[hit] if hit is not None else srcs)
    if arm == 'B-NAT' and isinstance(a.get('result'), dict):
        res = a['result']
        llm = res.get('llm') or {}
        out['n_refusals'] = cell(llm.get('refused'), f'{rid}/result.json')
        b = res.get('best_success_row') or {}
        out['published_best'] = cell(b.get('tflops_per_gpu'), f'{rid}:{b.get("attempt_id")}', note=b.get('params'))
    used = tr.get('used') or {}
    us = f'{rid}/trace.json'
    out['gpu_hours'] = cell(used.get('gpu_hours'), us)
    out['llm_tokens'] = cell(used.get('llm_tokens'), us)
    out['llm_usd'] = cell(used.get('llm_usd'), us)
    out['wall_hours'] = cell(used['wall_seconds'] / 3600.0, us) if finite(used.get('wall_seconds')) else na('no wall')
    out['stop_reason'] = cell(tr.get('stop_reason'), us)
    out['trials_used'] = trials_used(tr, us)
    want = score_task(w, arm).budget
    got = tr.get('budget') or {}
    diff = {k: (got.get(k), v) for k, v in want.items() if v and got.get(k) != v}
    if diff:
        problems.append(f'ENV-B {w} {arm}: trace budget differs from the protocol caps {diff}')
    task = score_task(w, arm)
    if ok(t_ref) and finite(iter_ref):
        Tdef = val(t_ref)
        values = {k: (s.get('step_time'), {'tflops': T}) for k, (s, T) in enumerate(zip(scr, Ts)) if T}   # per row
        evs = TJ.events(tr, task, float(iter_ref), ref_raw={'tflops': Tdef}, values=values)
        au, ct = TJ.auc(evs, task), TJ.ctg(evs, task)
        rs = (t_ref.get('src') or []) + [us]
        out['AUC_budget'] = cell(au['auc'], rs)
        out['x_end'] = cell(au['x_end'], rs)
        out['CTG_reached'] = cell(bool(ct['reached']), rs)
        out['CTG_gpu_hours'] = cell(ct['gpuh'], rs) if ct['reached'] else na('+5 % not reached')
        out['CTG_usd'] = cell(ct['usd'], rs) if ct['reached'] else na('+5 % not reached')
        out['CTG_wall_hours'] = cell(ct['wall_s'] / 3600.0, rs) if ct['reached'] else na('+5 % not reached')
    else:
        for k in ('AUC_budget', 'x_end', 'CTG_reached', 'CTG_gpu_hours', 'CTG_usd', 'CTG_wall_hours'):
            out[k] = pending(f"the y reference is missing ({(t_ref or {}).get('why') or 'B-DEF missing or invalid'})")
    producer = 'native' if arm == 'B-NAT' else 'infrarsi'         # R6-N2: the arm decides, never the trace
    ev = (a['store'].event_sequence(a['dir'] / 'events.jsonl')      # InfraRSI: the explicit call-to-row boundary
          if producer == 'infrarsi' and a.get('store') is not None else None)
    acc = accounting_d26(tr, recs, ev, producer=producer)  # checks for every arm; D26 only on a producer flag
    out['accounting'] = acc[:2]
    out['accounting_d26'] = acc[2]
    out['accounting_reconciliation'] = acc[3]
    out['findings'] = findings(f'ENV-B {w}', arm, rid, tr, infra, accounting=acc)
    return out


def _locked(store: Store, root: Path, rid: str | None, w: str, arm: str, selection: dict | None, problems: list,
            t_def: dict | None = None) -> dict:
    if not rid:
        return {'status': pending('no locked run id in the ledger')}
    d = root / DIR / rid
    lk = store.json(d / 'locked.json') if (d / 'locked.json').is_file() else None
    if not isinstance(lk, dict):
        return {'status': pending(f'locked pair {rid} not run' if not d.is_dir() else f'locked pair {rid} running')}
    if lk.get('skipped'):                   # D20: the selection is the start (workload defaults): no locked run, y = 0
        ls = f'{rid}/locked.json'
        if lk.get('skipped') != 'selection_is_default' or lk.get('arm') != arm or not _is_default(selection, w):
            problems.append(f'ENV-B {w} {arm} locked {rid}: locked.json is `skipped` ({lk.get("skipped")}) but the arm\'s '
                            'selection is not the workload default (D20 not confirmed)')
            return {'status': pending('D20 skip not confirmed by the arm\'s selection')}
        if not ok(t_def):
            return {'status': pending('D20: the locked value is the start reference (B-DEF), which is not available')}
        return {'run_id': rid, 'skipped': 'selection_is_default (D20)', 'order': na('D20: no locked run'),
                'tflops_locked': cell(val(t_def), (t_def.get('src') or []) + [ls], note='D20: the start reference (B-DEF)'),
                'tflops_default_locked': na('D20: no locked run'),
                'locked_valid': cell(True, ls, note='D20: the selection is the start'),
                'invalid_accept_locked': cell(0, ls, note='D20'), 'speedup_vs_default': cell(1.0, ls, note='D20'),
                'gpu_hours': cell(0.0, ls, note='D20: no locked run'), 'node': None}
    recs = store.receipts(d)
    halves = {}
    for who, cid in (('default', 'locked-default'), ('candidate', 'locked-candidate')):
        att = (lk.get(who) or {}).get('attempt_id')
        r = recs.get(att)
        if r is None:
            problems.append(f'ENV-B {w} {arm} locked {rid}: {who} receipt {att} missing')
            return {'status': pending(f'locked {who} receipt missing')}
        if (r.get('purpose') != 'locked' or r.get('candidate_id') != cid or r.get('arm') != arm
                or r.get('task_id') != OW.get(w).task_id):
            problems.append(f'ENV-B {w} {arm} locked {rid}: {who} receipt {att} is not this arm\'s {cid} run')
        if (lk.get(who) or {}).get('tflops_per_gpu') != tflops(r) or bool((lk.get(who) or {}).get('valid')) != bool(r.get('valid')):
            problems.append(f'ENV-B {w} {arm} locked {rid}: locked.json {who} differs from its receipt {att}')
        halves[who] = r
    d_r, c_r = halves['default'], halves['candidate']
    ident = selection is None or _same_params(c_r.get('params') or {}, selection, w)
    if not ident:
        problems.append(f'ENV-B {w} {arm} locked {rid}: the candidate half is not the arm\'s final selection')
    sd, sc = src(rid, d_r), src(rid, c_r)
    valid = bool(c_r.get('valid')) and ident
    out = {'run_id': rid, 'order': cell(lk.get('order'), f'{rid}/locked.json'),
           'tflops_locked': cell(tflops(c_r), sc) if valid and tflops(c_r) else na('locked candidate invalid'),
           'tflops_default_locked': cell(tflops(d_r), sd) if d_r.get('valid') and tflops(d_r) else na('default half invalid'),
           'locked_valid': cell(valid, sc, note=None if ident else 'identity: not the final selection'),
           'invalid_accept_locked': cell(0 if valid else 1, sc),
           'gpu_hours': cell(sum(((r.get('cost') or {}).get('gpu_seconds') or 0) for r in halves.values()) / 3600.0, [sd, sc]),
           'node': d_r.get('node'), '_iter_default': (d_r.get('metric') or {}).get('value') if d_r.get('valid') else None}
    if d_r.get('node') != c_r.get('node'):
        problems.append(f'ENV-B {w} {arm} locked {rid}: the halves ran on different nodes ({d_r.get("node")}, {c_r.get("node")})')
    if ok(out['tflops_locked']) and ok(out['tflops_default_locked']):
        out['speedup_vs_default'] = cell(val(out['tflops_locked']) / val(out['tflops_default_locked']), [sd, sc])
    else:
        out['speedup_vs_default'] = na('a half is invalid')
    return out


def _y_locked(row: dict, ref: dict | None) -> float | None:
    """ln(T_locked / ref); D20 (skipped pair): 0."""
    lk = row.get('locked') or {}
    if lk.get('skipped'):
        return 0.0
    if ok(ref) and ok(lk.get('tflops_locked')):
        return math.log(val(lk['tflops_locked']) / val(ref))
    return None


def _compare(w: str, a: str, b: str, arms: dict, t_def: dict, rsd: float | None) -> dict:
    for x in (a, b):
        if arms[x]['status'] == 'not run':
            return {'label': 'not run', 'why': f'{x}: {arms[x].get("why") or "no run id in the ledger (optional arm / cut)"}'}
    for x in (a, b):
        if arms[x]['status'] != 'finished':
            return {'label': 'pending', 'why': f'{x} {arms[x]["status"]}'}
    if rsd is None:
        return {'label': 'pending', 'why': 'margin_w not computable'}
    rows = {}
    for x in (a, b):
        lk = arms[x].get('locked') or {}
        if not ok(lk.get('locked_valid')):
            return {'label': 'pending', 'why': f'{x} locked pair: {(lk.get("status") or {}).get("why")}'}
        auc = arms[x]['search'].get('AUC_budget')
        if not ok(auc):
            return {'label': 'pending', 'why': f'{x} AUC_budget pending ({(auc or {}).get("why")})'}
        v = bool(val(lk['locked_valid']))
        row = {'locked_valid': v, 'invalid_accept_locked': val(lk['invalid_accept_locked']), 'AUC_budget': val(auc),
               'accounting_complete': arms[x]['search']['accounting'][0],
               'accounting_reasons': arms[x]['search']['accounting'][1]}
        if v:
            y = _y_locked(arms[x], arms[x].get('y_ref'))
            if y is None:
                return {'label': 'pending', 'why': f'{x}: y reference missing ({(arms[x].get("y_ref") or {}).get("why")})'}
            row['y_locked'] = y
        rows[x] = row
    task = score_task(w, a)
    res = TJ.compare(task, rows[a], rows[b], rsd=rsd)
    da, db = (arms[x]['locked'].get('tflops_default_locked') for x in (a, b))
    res['drift_default_halves'] = (val(da) / val(db)) if ok(da) and ok(db) else None
    # sensitivity (reported, never applied): the label under the raw ratio of the two locked selections (one common
    # reference: B-DEF) and under the paired ratio (each pair's own default half)
    ta, tb = (arms[x]['locked'].get('tflops_locked') for x in (a, b))
    sa, sb = (arms[x]['locked'].get('speedup_vs_default') for x in (a, b))
    res['raw_ratio'] = val(ta) / val(tb) if ok(ta) and ok(tb) else None
    res['paired_ratio'] = val(sa) / val(sb) if ok(sa) and ok(sb) else None
    base = res.get('label_if_complete', res['label'])
    flags = []
    for name, ys in (('raw', {x: _y_locked(arms[x], t_def) for x in (a, b)}),
                     ('paired', {x: (math.log(val(arms[x]['locked']['speedup_vs_default']))
                                     if ok(arms[x]['locked'].get('speedup_vs_default')) else None) for x in (a, b)})):
        if any(rows[x]['locked_valid'] and ys[x] is None for x in (a, b)):
            res[f'label_if_{name}'] = None
            continue
        alt = TJ.compare(task, *({**rows[x], **({'y_locked': ys[x]} if rows[x]['locked_valid'] else {})} for x in (a, b)),
                         rsd=rsd)
        res[f'label_if_{name}'] = alt.get('label_if_complete', alt['label'])
        if res[f'label_if_{name}'] != base:
            flags.append(f'{name}: {res[f"label_if_{name}"]}')
    res['sensitivity_flag'] = flags
    res['src'] = sorted({s for x in (a, b) for c in (arms[x]['locked'].get('locked_valid'), arms[x]['locked'].get('tflops_locked'),
                                                      arms[x]['search'].get('AUC_budget'), arms[x].get('y_ref'))
                         for s in ((c or {}).get('src') or [])})
    return res


def _random_sentence(w: str, a: str, res: dict, coverage: str = '') -> str | None:
    lab = res.get('label')
    if lab == 'advantage':
        return f'ENV-B {w}: LLM search ({a}) beats random search (B-RND) beyond margin_w on every condition: advantage.'
    if lab == 'not run':
        return f'ENV-B {w}: {a} vs B-RND not compared ({res.get("why")}).'
    if lab == 'incomplete':
        return f'ENV-B {w}: {a} vs B-RND incomplete (accounting); the label if complete would be {res.get("label_if_complete")}.'
    if lab in ('parity', 'disadvantage', 'invalid'):
        return f'ENV-B {w}: there is no evidence that LLM search ({a}) beats random search (B-RND): {lab}.'
    if lab == 'no_advantage':
        return f'ENV-B {w}: {a} beats B-RND beyond margin_w on the locked TFLOPS but not on every condition: no_advantage.'
    return None


def build(store: Store, root: Path, ledger: dict, *, abandoned: list) -> dict:
    problems: list[str] = []
    ab_ids = {x['run_id'] for x in abandoned if x['env'] == 'ENV-B'}
    rec_all = protocol_receipts(store, root, ledger, ab_ids)
    out = {'workloads': {}, 'problems': problems, 'pending': [], 'abandoned': []}
    for w in workloads(ledger):
        spec = ledger['ENV-B'][w]
        if store.key_present and store.verify:
            mrec = OM.compute(w, rec_all)
        else:                                          # S_w needs verified receipts: never a silent calibration-only value
            mrec = {'workload': w, 'estimator': 'unavailable (no receipt key: S_w cannot be verified)', 'n': None,
                    'S_w': [], 'excluded': [], 'margin': None, 'rsd_used': None}
        ex = store.json(root / DIR / f'margin_{w}.json') if (root / DIR / f'margin_{w}.json').is_file() else None
        if isinstance(ex, dict) and mrec.get('margin') is not None:
            mine = sorted(x['attempt_id'] for x in mrec['S_w'])
            theirs = sorted(x.get('attempt_id') for x in ex.get('S_w') or [])
            if mine != theirs or ex.get('estimator') != mrec.get('estimator') or ex.get('margin') != mrec.get('margin'):
                extra = [x for x in ex.get('S_w') or [] if x.get('attempt_id') not in set(mine)]
                ab_extra = sorted({str(x.get('run_id')) for x in extra if in_ids(x.get('run_id'), ab_ids)})
                problems.append(f'ENV-B {w}: records margin_{w}.json differs from the recomputation over the ledger\'s '
                                f'protocol runs: it has {len(extra)} sample(s) the ledger excludes'
                                + (f' (from abandoned runs {ab_extra})' if ab_extra else '')
                                + f' and lacks {len(set(mine) - set(theirs))}; margin {ex.get("margin")} '
                                f'({ex.get("estimator")}, n {ex.get("n")}) vs recomputed {mrec.get("margin")} '
                                f'({mrec.get("estimator")}, n {mrec.get("n")})'
                                + ('; no numeric effect' if ex.get('margin') == mrec.get('margin') else '')
                                + ' - regenerate it over the ledger\'s protocol runs (optico.cli margin --receipts)')
        margin_w, rsd = mrec.get('margin'), mrec.get('rsd_used')
        msrc = [f'{x["run_id"]}:{x["attempt_id"]}' for x in mrec['S_w']] or ['AMENDMENT_EXT_002 W3 calibration group']
        other = [x for x in mrec.get('S_w') or [] if spec.get('node') and x.get('node') and x['node'] != spec['node']]
        s_w_nodes = None
        if other and margin_w is not None:              # D26(c): kept (EXT_002: protocol-produced default runs)
            drop = {x['attempt_id'] for x in other}
            alt = OM.compute(w, [r for r in rec_all if r.get('attempt_id') not in drop])
            same = alt.get('margin') is not None and abs(alt['margin'] - margin_w) < 1e-12
            floor = same and abs(margin_w - OM.FLOOR) < 1e-12           # the estimator's own floor (margin.py)
            s_w_nodes = {'nodes': sorted({x.get('node') for x in mrec['S_w'] if x.get('node')}),
                         'other_node_samples': [{k: x.get(k) for k in ('run_id', 'attempt_id', 'arm', 'kind', 'node', 'T')}
                                                for x in other],
                         'margin_without_them': alt.get('margin'), 'estimator_without_them': alt.get('estimator'),
                         'statement': (f"S_{w} keeps {len(other)} default run(s) from "
                                       f"{', '.join(sorted({x['node'] for x in other}))} "
                                       f"({', '.join(sorted({str(x.get('arm')) for x in other}))}) per D26(c) (EXT_002: "
                                       'protocol-produced default runs of the workload); '
                                       + ('no numeric effect (3 % floor)' if floor else 'no numeric effect' if same else
                                          f"margin_w {alt.get('margin')} without them vs {margin_w} with them"))}
        W = {'spec': {k: spec.get(k) for k in ARMS}, 'node': spec.get('node'), 'margin_record': mrec,
             'margin_w': None, '_msrc': msrc, 's_w_nodes': s_w_nodes,   # the margin cell is set once every slot is known
             'n_w': cell(mrec.get('n'), msrc), 'rsd_used': cell(rsd, msrc), 'arms': {}}
        bdef = _single(store, root, spec.get('B-DEF'), 'b_def', problems, w, ab_ids)
        bed = _single(store, root, spec.get('B-ED'), 'b_ed', problems, w, ab_ids)
        t_def, def_node = bdef['T'], bdef.get('node')
        iter_def = val(bdef['iter_s']) if ok(bdef.get('iter_s')) else None
        W['T_def'] = t_def
        W['arms']['B-DEF'] = {'arm': 'B-DEF', **bdef}
        W['arms']['B-ED'] = {'arm': 'B-ED', **bed}
        if ok(bed.get('T')) and ok(t_def):
            W['arms']['B-ED']['speedup_trial_vs_bdef'] = cell(val(bed['T']) / val(t_def),
                                                              (bed['T'].get('src') or []) + (t_def.get('src') or []))

        def locked_of(arm, row, sel):
            lspec = (spec.get('locked') or {}).get(arm)
            if row.get('status') != 'finished':
                return {'status': missing(row.get('status') or '', row.get('why'))}
            if in_ids(run_id_of(lspec), ab_ids):
                return {'status': pending(f'locked run {lspec} is abandoned (ledger conflict)')}
            return _locked(store, root, run_id_of(lspec), w, arm, sel, problems, t_def=t_def)
        W['arms']['B-ED']['locked'] = locked_of('B-ED', bed, bed['receipt'].get('params') if isinstance(bed.get('receipt'), dict) else None)
        for arm in SEARCH_ARMS:
            if arm in OPTIONAL_ARMS and arm not in spec:
                continue
            a = _load(store, root, spec.get(arm), ab_ids)
            row = {'arm': arm, 'run_id': a.get('run_id'), 'status': a['status'], 'why': a.get('why'), 'selection': None}
            if a['status'] == 'finished':
                row['selection'] = _selection(a)
                nodes = sorted({r.get('node') for r in a['receipts'].values() if r.get('node')})
                row['node'] = nodes[0] if len(nodes) == 1 else None
                if len(nodes) > 1:
                    problems.append(f'ENV-B {w} {arm} {a["run_id"]}: receipts on several nodes {nodes}')
                row['locked'] = locked_of(arm, row, row['selection'])
                row['y_ref'], it_ref = t_def, iter_def
                row['y_ref_basis'] = 'B-DEF (same node)'
                if def_node and row['node'] and row['node'] != def_node:          # D23: a same-node ratio
                    lk = row['locked']
                    dh = lk.get('tflops_default_locked')
                    row['y_ref_basis'] = (f'D23: the locked pair default half on {row["node"]} (B-DEF ran on {def_node}); '
                                          'y is a same-node ratio')
                    if ok(dh) and lk.get('node') == row['node'] and finite(lk.get('_iter_default')):
                        row['y_ref'] = cell(val(dh), dh.get('src'), note=f'D23: default half on {row["node"]}')
                        it_ref = float(lk['_iter_default'])
                        row['node_factor'] = (cell(val(dh) / val(t_def), (dh.get('src') or []) + (t_def.get('src') or []),
                                                   note=f'workload default {row["node"]} / {def_node} (D23)')
                                              if ok(t_def) else na('B-DEF value missing'))
                    else:
                        row['y_ref'] = pending(f'D23: {arm} ran on {row["node"]}, B-DEF on {def_node}: its y reference is '
                                               f'its locked pair default half on {row["node"]} '
                                               f'({(lk.get("status") or {}).get("why") or "not available"})')
                        row['node_factor'] = pending('the same-node default half is not available')
                row['search'] = _search(a, w, arm, margin_w, row['y_ref'], it_ref, problems)
            else:
                row['locked'] = {'status': missing(row['status'], row.get('why'))}
            W['arms'][arm] = row
        labels = {}
        for a_, b_ in MAIN_PAIRS + RND_PAIRS:
            if a_ in W['arms'] and b_ in W['arms']:
                labels[f'{a_} vs {b_}'] = _compare(w, a_, b_, W['arms'], t_def, rsd)
        W['labels'] = labels
        W['random_sentences'] = [s for k, res in labels.items() if k.endswith('vs B-RND')
                                 for s in [_random_sentence(w, k.split(' vs ')[0], res)] if s]
        nat = W['arms'].get('B-NAT') or {}
        tti = (nat.get('search') or {}).get('trials_to_first_improvement')
        W['bnat_beats_default'] = (missing(nat.get('status') or 'not started', nat.get('why')) if nat.get('status') != 'finished' else
                                   tti if not ok(tti) else cell(isinstance(val(tti), int), tti.get('src')))
        W['slots_final'] = (all(r.get('status') in ('finished', 'not run') for r in W['arms'].values())
                            and not any(isinstance((r.get('locked') or {}).get('status'), dict)
                                        for r in W['arms'].values() if r.get('status') == 'finished' and r['arm'] in LOCKED_ARMS))
        out['workloads'][w] = W
        for arm, row in W['arms'].items():
            if row.get('status') == 'not run':
                continue                                # a sanctioned final state (ledger), not pending
            if row.get('status') != 'finished':
                out['pending'].append(f'ENV-B {w} {arm}: {row.get("run_id")} {row["status"]}'
                                      + (f' ({row["why"]})' if row.get('why') else ''))
            lk = row.get('locked') or {}
            if isinstance(lk.get('status'), dict) and row.get('status') == 'finished':
                out['pending'].append(f'ENV-B {w} {arm} locked: {lk["status"].get("why")}')
        if mrec.get('margin') is None:
            out['pending'].append(f'ENV-B {w} margin_w: {mrec["estimator"]}')
    # AMENDMENT_EXT_002 rule 3: the pooled fallback is the final estimator when n_w < 4 at G-FIN (cut arms, invalid
    # defaults); it pools every workload's S, so it is final once every ENV-B slot is final
    every = all(W['slots_final'] for W in out['workloads'].values())
    for w, W in out['workloads'].items():
        m = W['margin_record']
        msrc = W.pop('_msrc')
        if m.get('margin') is None:
            W['margin_w'] = pending(m['estimator'])
        else:                                          # R3-N6: the note follows the gate (numbers unchanged)
            if m['estimator'] == 'run_to_run':
                note = 'run_to_run' + ('' if W['slots_final'] else ' (provisional until the workload\'s protocol runs are final)')
            else:
                note = m['estimator'] + (' (final pooled fallback under AMENDMENT_EXT_002)' if every
                                         else ' (provisional until every ENV-B slot is final)')
            W['margin_w'] = cell(m['margin'], msrc, note=note)
        W['margin_final'] = m.get('margin') is not None and (W['slots_final'] if m['estimator'] == 'run_to_run' else every)
        for r in W['labels'].values():                 # a label computed on a margin that may still move says so
            if r.get('label') not in ('pending', 'not run'):
                r['margin_final'] = W['margin_final']
        if m.get('margin') is not None and m['estimator'] != 'run_to_run' and not every:
            out['pending'].append(f'ENV-B {w} margin_w: {m["estimator"]} (n_w {m["n"]}) - provisional until every ENV-B '
                                  'slot is final')
    # environment label (§0.3 / AMENDMENT_EXT_002: >= 2 workloads advantage and 0 disadvantage, absolute count)
    main = {w: W['labels'].get('B-IS vs B-NAT', {'label': 'pending'})['label'] for w, W in out['workloads'].items()}
    counts = {k: sum(1 for v in main.values() if v == k) for k in sorted(set(main.values()))}
    if any(v == 'pending' for v in main.values()):
        env_label = 'pending'
    elif any(v == 'incomplete' for v in main.values()):
        env_label = 'incomplete'
    elif sum(v == 'advantage' for v in main.values()) >= 2 and not any(v in BAD_FOR_IS for v in main.values()):
        env_label = 'advantage'
    else:
        env_label = 'no advantage (rule: >= 2 workloads advantage and 0 disadvantage not met)'
    out['env_label'] = {'label': env_label, 'per_workload': main, 'counts': counts}
    gate = {w: W['bnat_beats_default'] for w, W in out['workloads'].items()}
    yes = sum(1 for c in gate.values() if ok(c) and val(c) is True)
    no = sum(1 for c in gate.values() if ok(c) and val(c) is False)
    need = 2 if len(gate) >= 3 else len(gate)
    decided = all(ok(c) or (isinstance(c, dict) and c.get('status') == 'n/a') for c in gate.values())
    out['method_behaviour_reproduced'] = {
        'per_workload': gate, 'value': 'yes' if yes >= need else ('no' if (no > len(gate) - need or decided) else 'pending'),
        'margins_final': all(W.get('margin_final') for W in out['workloads'].values()),
        'rule': 'B-NAT beats its own default by margin_w within 20 iterations on >= 2/3 workloads (§0.3, AMENDMENT_EXT_002)'}
    for x in abandoned:
        if x['env'] == 'ENV-B':
            out['abandoned'].append({**x, 'cost': run_cost(store, root / DIR / x['run_id'], world_size=OW.GPUS)})
    out['findings'] = [f for W in out['workloads'].values() for a in W['arms'].values()
                       for f in (a.get('search') or {}).get('findings') or []]
    return out


# ---------------------------------------------------------------- markdown
def markdown(sec: dict) -> str:
    L = ['# ENV-B — OptiCo-public workloads (8×H100, judge = `run_monitor.sh`, TFLOPS/GPU of iterations 3–5)', '',
         'Design §3.6 / §0.3 with AMENDMENT_EXT_002 Q1 (margin_w = max(3 %, 3 × run-to-run RSD of the protocol default '
         'runs; n_w ≥ 4, else pooled fallback). B-NAT runs OptiCo\'s own loop "with its reasoning-format lines removed '
         'because the driver refuses to emit tagged reasoning; the parsers accept untagged answers" (D-OC-15, '
         'AMENDMENT_EXT_001 Q5). Not compared with the paper\'s "+1.84 % vs expert" (expert configurations unpublished): '
         'only gains over the script default are reported. y for AUC / CTG = ln(TFLOPS / TFLOPS_B-DEF).', '',
         f"**ENV-B label: {sec['env_label']['label']}** (B-IS vs B-NAT per workload: {sec['env_label']['per_workload']}).", '',
         f"Method behaviour reproduced (§0.3 gate): {sec['method_behaviour_reproduced']['value']}"
         + ('' if sec['method_behaviour_reproduced'].get('margins_final') else ' (provisional: margin_w not final on every '
                                                                              'workload)') + f" — "
         f"{sec['method_behaviour_reproduced']['rule']}; per workload: "
         + ', '.join(f'{w} {R.fmt(c)}' for w, c in sec['method_behaviour_reproduced']['per_workload'].items()), '']
    for w, W in sec['workloads'].items():
        m = W['margin_record']
        L += [f'## {w} ({OW.get(w).model_type} {OW.get(w).model_size}, node {W["node"]})', '',
              f"margin_w = {R.fmt(W['margin_w'], 4)} ({m['estimator']}; n_w = {m['n']}, RSD used = {R.fmt(m.get('rsd_used'), 4)}"
              + (f", pooled df {m.get('pooled_df')}" if m.get('pooled_df') else '') + '). W3 judge property: iteration 4 is the '
              'slowest in every default run (issue #21) — a within-run pattern, not noise; it cancels in comparisons.'
              + (f" {W['s_w_nodes']['statement']}." if W.get('s_w_nodes') else ''), '']
        t = []
        for arm, a in W['arms'].items():
            st = a.get('status')
            lk = a.get('locked') or {}
            lk_why = lk['status'] if isinstance(lk.get('status'), dict) else na('B-DEF has no locked pair')
            q = {k: lk.get(k, lk_why) for k in ('tflops_locked', 'tflops_default_locked', 'speedup_vs_default',
                                                  'invalid_accept_locked')}
            if arm in REFERENCE_ARMS:
                ref = na('reference trial (no search)')
                s = {k: ref for k in ('trials_to_first_improvement', 'n_trials', 'n_failed_runs', 'n_invalid_prelaunch',
                                      'n_guard_rejections', 'CTG_gpu_hours', 'AUC_budget', 'llm_tokens', 'llm_usd',
                                      'wall_hours', 'trials_used')}
                s['tflops_best_search'] = a.get('T', missing(st or '', a.get('why')))
                s['gpu_hours'] = a.get('gpu_hours', missing(st or '', a.get('why')))
            else:
                s = a.get('search') or {}
            p = lambda k: s.get(k, missing(st or '', a.get('why')))  # noqa: E731
            acc = s.get('accounting') if arm not in REFERENCE_ARMS else None
            acc_txt = ('n/a' if not acc else 'complete' + (f" ({s['accounting_d26']})" if s.get('accounting_d26') else '')
                       if acc[0] else 'incomplete (' + '; '.join(map(str, acc[1])) + ')')
            t.append([arm, a.get('run_id'), st, q['tflops_locked'], q['tflops_default_locked'], q['speedup_vs_default'],
                      p('tflops_best_search'), p('trials_to_first_improvement'), p('n_trials'), p('n_failed_runs'),
                      p('n_invalid_prelaunch'), p('n_guard_rejections'), q['invalid_accept_locked'], p('CTG_gpu_hours'),
                      p('AUC_budget'), p('gpu_hours'), p('llm_tokens'), p('llm_usd'), p('wall_hours'),
                      p('n_refusals') if arm == 'B-NAT' else na('—'), p('trials_used'), acc_txt])
        L += [R.table(['arm', 'run', 'status', 'tflops_locked', 'default half', 'speedup_vs_default', 'tflops_best_search',
                       'trials_to_first_improvement', 'n_trials', 'n_failed_runs', 'n_invalid_prelaunch',
                       'n_guard_rejections', 'invalid_accept_locked', 'CTG(+5 %) GPUh', 'AUC_budget', 'GPUh',
                       'LLM tokens', 'USD', 'wall h', 'n_refusals', 'trials used / budget (stop reason)', 'accounting'], t,
                      {3: 1, 4: 1, 5: 4, 6: 1, 13: 3, 14: 4, 15: 2, 17: 2, 18: 2}), '']
        for arm, a in W['arms'].items():
            if a.get('node_factor') is not None:
                L += [f"{arm} ran on {a.get('node')} (B-DEF on {W['arms']['B-DEF'].get('node')}): {a.get('y_ref_basis')}; "
                      f"y reference {R.fmt(a.get('y_ref'), 1)} TFLOPS; node factor of the default {R.fmt(a['node_factor'], 4)}.", '']
            if (a.get('locked') or {}).get('skipped'):
                L += [f"{arm}: locked pair {a['locked'].get('run_id')} `skipped` (D20: the selection is the workload default); "
                      f"locked value = B-DEF {R.fmt(a['locked'].get('tflops_locked'), 1)}, y = 0, InvalidAccept 0.", '']
        bed = W['arms'].get('B-ED') or {}
        if bed.get('speedup_trial_vs_bdef'):
            L += [f"B-ED single trial vs B-DEF: ×{R.fmt(bed['speedup_trial_vs_bdef'], 3)} ({R.fmt(bed.get('params'))}).", '']
        L += ['Labels: the pre-declared quantity (AMENDMENT_EXT_002: the two arms\' raw locked TFLOPS, Δy = ln of their '
              'ratio; D23: a cross-node arm uses its same-node ratio). Sensitivity, reported and never applied: the label '
              'under the raw ratio T_locked(a) / T_locked(b) and under the paired ratio speedup_vs_default(a) / '
              'speedup_vs_default(b) (each pair\'s own default half); "DIFFERS" marks a label that would change.', '']
        t = [[k, R.label_text(r) + ('' if r.get('margin_final', True) else ' (margin_w provisional)'), r.get('delta_y'),
              r.get('margin_y'), r.get('raw_ratio'), r.get('label_if_raw') or 'n/a',
              r.get('paired_ratio'), r.get('label_if_paired') or 'n/a', r.get('drift_default_halves'),
              ('DIFFERS (' + '; '.join(r['sensitivity_flag']) + ')') if r.get('sensitivity_flag') else
              ('same' if r.get('label_if_raw') or r.get('label_if_paired') else 'n/a'),
              ', '.join(f'{c}={"yes" if v else "no"}' for c, v in (r.get('conditions') or {}).items()) or r.get('why') or '']
             for k, r in W['labels'].items()]
        L += [R.table(['pair', 'label', 'Δy', 'margin_y', 'raw T_locked a/b', 'label under raw', 'paired speedup a/b',
                       'label under paired', 'drift (default halves a/b)', 'sensitivity', 'conditions / why'], t,
                      {2: 4, 3: 4, 4: 4, 6: 4, 8: 4}), '']
        L += ['LLM vs random: ' + (' '.join(W['random_sentences']) or 'pending (' + '; '.join(
            f"{k} {r.get('label')}" for k, r in W['labels'].items() if k.endswith('vs B-RND')) + ').'), '']
    L += ['## Abandoned runs (pipeline-fix line; not results; never in S_w)', '']
    t = [[x['scope'], x['run_id'], x['reason'], x['cost'].get('gpu_hours'), x['cost'].get('llm_tokens'), x['cost'].get('llm_usd'),
          x['cost'].get('wall_hours'), x['cost'].get('basis')] for x in sec['abandoned']]
    L += [R.table(['workload', 'run', 'reason', 'GPUh', 'LLM tokens', 'USD', 'wall h', 'basis'], t, {3: 2, 5: 2, 6: 2}), '']
    L += ['## Findings in the records (facts for O; not tool problems)', '', R.bullets(sec['findings']), '']
    L += ['## Pending and problems', '', '**Pending**', '', R.bullets(sec['pending']), '', '**Problems**', '',
          R.bullets(sec['problems']), '']
    return '\n'.join(L)
