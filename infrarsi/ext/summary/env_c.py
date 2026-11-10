"""ENV-C (autoresearch, 1×H100, 300 s, trusted judge_bpb): design §4.4 / §0.3 with AMENDMENT_EXT_002 Q2 and
AMENDMENT_EXT_007 applied.

Per arm: val_bpb_locked, val_bpb_best_search, steps / tok_per_s / mfu / peak_vram (the locked candidate run),
n_experiments, n_crash, n_invalid_check, n_failed_by_class (cuda_graph / oom / timeout / other, AMENDMENT_EXT_007),
invalid_accept_locked, CTG(-0.002), AUC_budget, costs; C-RND space_coverage of its pruned frozen space.
The advantage rule runs on the infra track only: C-IS vs C-NAT-infra (fixed 0.002 bpb margin; y = bpb_C-BASE - bpb with
the D-AR-7 C-BASE anchor of the arms' node, AMENDMENT_EXT_002 Q2). C-NAT-free is the published-protocol reproduction
(free track, printed val_bpb), described, never compared. LLM vs random: C-IS / C-NAT-infra vs C-RND with C-RND's
coverage printed beside the sentence.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

from ..autoresearch import knobs as AK
from ..autoresearch import task as AT
from ..common import trajectory as TJ
from . import render as R
from .cells import cell, missing, na, ok, pending, val
from .records import (ScoreTask, Store, accounting_text, accounting_verified, anchor_ref, findings, finite, in_ids,
                      not_run_reason, refusal_kind, run_cost, run_id_of, src, trials_used, utc)

DIR = 'autoresearch'
ARMS = ('C-IS', 'C-NAT-infra', 'C-NAT-free', 'C-RND')
MAIN = ('C-IS', 'C-NAT-infra')
RND_PAIRS = (('C-IS', 'C-RND'), ('C-NAT-infra', 'C-RND'))
README_FIGURE = 'README figure: 83 experiments, 15 kept, val_bpb ≈ 0.998 → ≈ 0.977 (RELATED_WORK_EVAL group B; design §4.4)'


def score_task(arm: str) -> ScoreTask:
    b = AT.BUDGET
    return ScoreTask(id='ENV-C', env='autoresearch', arm=arm, margin=dict(AT.MARGIN), tau=AT.TAU,
                     budget={'screens': b['screens'], 'llm_tokens': None if arm == 'C-RND' else b['llm_tokens'],
                             'wall_seconds': b['wall_seconds']})


def _bpb(r: dict | None) -> float | None:
    v = ((r or {}).get('metric') or {}).get('value')
    return float(v) if finite(v) and v > 0 else None


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
    tr = None
    for name in ('trace.json', 'agent/trace.json'):
        if (d / name).is_file():
            tr = store.json(d / name)
            break
    res = store.json(d / 'result.json') if (d / 'result.json').is_file() else None
    finished = isinstance(tr, dict) and tr.get('stop_reason') and tr.get('ended_utc') and isinstance(res, dict)
    return {'status': 'finished' if finished else 'running', 'run_id': rid, 'dir': d, 'trace': tr, 'result': res,
            'receipts': store.receipts(d), 'store': store}


def _search(a: dict, arm: str, base: dict, problems: list) -> dict:
    tr, recs, rid = a['trace'], a['receipts'], a['run_id']
    scr = list(tr.get('screens') or [])
    out = {}
    classes, infra = collections.Counter(), collections.Counter()
    n_crash = n_check = 0
    for i, s in enumerate(scr):
        r = recs.get(s.get('attempt_id'))
        if r is None:
            problems.append(f'ENV-C {arm}: screen {i} ({s.get("candidate_id")}) has no receipt {s.get("attempt_id")}')
            continue
        if bool(r.get('valid')) != bool(s.get('valid')) or (s.get('valid') and _bpb(r) != s.get('step_time')):
            problems.append(f'ENV-C {arm}: screen {i} ({s.get("candidate_id")}) differs from its receipt {s.get("attempt_id")}')
        if isinstance(r.get('runner_error'), dict):              # refused / failed by our harness, not by the candidate
            infra[refusal_kind(r)] += 1
            continue
        fc = r.get('failure_class')
        if fc:
            classes[fc] += 1
        rej = [str(g.get('id')) for g in r.get('guard_rejections') or [] if isinstance(g, dict)]
        n_crash += bool(fc) or 'exit_ok' in rej
        n_check += any(g.startswith('C-') or g == 'frozen_files' for g in rej)
    srcs = [src(rid, recs.get(s.get('attempt_id'))) for s in scr]
    good = [s for s in scr if s.get('valid') and finite(s.get('step_time'))]
    out['n_experiments'] = cell(len(scr) - sum(infra.values()), srcs,
                                note='launched screens incl. the start (C-IS / C-RND: the baseline c0)')
    out['n_runner_errors'] = cell(dict(infra), srcs, note='screens our harness refused or lost (e.g. occupancy): not '
                                                          'experiments, counted by the trace as screens')
    out['n_crash'] = cell(n_crash, srcs)
    out['n_invalid_check'] = cell(n_check, srcs)
    out['n_failed_by_class'] = cell(dict(classes), srcs)
    if good:
        b = min(good, key=lambda s: s['step_time'])
        out['val_bpb_best_search'] = cell(b['step_time'], src(rid, recs.get(b.get('attempt_id'))))
    else:
        out['val_bpb_best_search'] = na('no valid experiment')
    ex = tr.get('extra') if isinstance(tr.get('extra'), dict) else {}
    if arm == 'C-RND':
        out['space_coverage'] = cell(ex.get('space_coverage'), f'{rid}/trace.json', note=f'space size {ex.get("space_size")}')
        out['knob_space_sha256'] = cell(ex.get('knob_space_sha256'), f'{rid}/trace.json')
    used = tr.get('used') or {}
    us = f'{rid}/trace.json'
    for k in ('gpu_hours', 'llm_tokens', 'llm_usd'):
        out[k] = cell(used.get(k), us)
    out['wall_hours'] = cell(used['wall_seconds'] / 3600.0, us) if finite(used.get('wall_seconds')) else na('no wall')
    out['stop_reason'] = cell(tr.get('stop_reason'), us)
    out['trials_used'] = trials_used(tr, us)
    if ok(base):
        task = score_task(arm)
        evs = TJ.events(tr, task, val(base))
        au, ct = TJ.auc(evs, task), TJ.ctg(evs, task)
        rs = (base.get('src') or []) + [us]
        out['AUC_budget'] = cell(au['auc'], rs)
        out['x_end'] = cell(au['x_end'], rs)
        out['CTG_reached'] = cell(bool(ct['reached']), rs)
        out['CTG_gpu_hours'] = cell(ct['gpuh'], rs) if ct['reached'] else na('-0.002 not reached')
        out['CTG_usd'] = cell(ct['usd'], rs) if ct['reached'] else na('-0.002 not reached')
        out['CTG_wall_hours'] = cell(ct['wall_s'] / 3600.0, rs) if ct['reached'] else na('-0.002 not reached')
    else:
        for k in ('AUC_budget', 'x_end', 'CTG_reached', 'CTG_gpu_hours', 'CTG_usd', 'CTG_wall_hours'):
            out[k] = pending('C-BASE anchor receipt missing')
    # D27: the accounting checks; the basic-agent arms (C-NAT-free / C-NAT-infra) record no call-to-row boundary:
    # their per-row LLM attribution is producer-asserted; C-IS / C-RND: the events.jsonl order
    producer = 'agent' if arm in ('C-NAT-free', 'C-NAT-infra') else 'infrarsi'   # R6-N2: the arm decides
    ev = (a['store'].event_sequence(a['dir'] / 'events.jsonl')
          if a.get('store') is not None and producer == 'infrarsi' else None)
    acc = accounting_verified(tr, recs, ev, producer=producer)
    out['accounting'] = acc[:2]
    out['accounting_reconciliation'] = acc[3]
    out['findings'] = findings('ENV-C', arm, rid, tr, dict(infra), accounting=acc)
    return out


def _train_sha(r: dict | None) -> str | None:
    return ((r or {}).get('files') or {}).get('train.py')


def _d20(rid: str, arm: str, res: dict, a: dict | None, anchor: dict | None, problems: list) -> dict:
    """D20: no locked pair when the arm's final train.py is the baseline; locked value = the C-BASE anchor (y = 0),
    InvalidAccept 0. Confirmed from the records: the selection's own receipt digest and the skip record's
    selection_sha256 are both the anchor's (baseline) train.py digest."""
    ls = f'{rid}/result.json'
    base_sha = (anchor or {}).get('train_sha')
    sel_sha = None
    if a is not None and isinstance(a.get('trace'), dict):
        sel = a['trace'].get('selected')
        if sel is None:
            sel_sha = base_sha                          # nothing selected: the baseline
        else:
            scr = next((x for x in a['trace'].get('screens') or [] if x.get('candidate_id') == sel), None)
            sel_sha = _train_sha(a['receipts'].get((scr or {}).get('attempt_id'))) if scr else None
    if not (res.get('skipped') == 'selection_is_baseline' and res.get('arm') in (None, arm) and base_sha
            and sel_sha == base_sha and res.get('selection_sha256') == base_sha):
        problems.append(f'ENV-C {arm} locked {rid}: result.json is `skipped` ({res.get("skipped")}) but the selection is not '
                        f'confirmed as the baseline train.py (selection digest {str(sel_sha)[:12]}, skip record '
                        f'{str(res.get("selection_sha256"))[:12]}, anchor {str(base_sha)[:12]}; D20 not confirmed)')
        return {'status': pending('D20 skip not confirmed by the records')}
    base = (anchor or {}).get('val_bpb')
    if not ok(base):
        return {'status': pending('D20: the locked value is the C-BASE anchor, which is not available')}
    return {'run_id': rid, 'skipped': 'selection_is_baseline (D20)', 'order': na('D20: no locked run'),
            'val_bpb_locked': cell(val(base), (base.get('src') or []) + [ls], note='D20: the C-BASE anchor (y = 0)'),
            'val_bpb_baseline_half': na('D20: no locked run'),
            'locked_valid': cell(True, ls, note='D20: the selection is the baseline'),
            'invalid_accept_locked': cell(0, ls, note='D20'),
            **{k: na('D20: no locked run') for k in ('steps', 'tok_per_s', 'mfu', 'peak_vram_mb')},
            'gpu_hours': cell(0.0, ls, note='D20: no locked run')}


def _locked(store: Store, root: Path, rid: str | None, arm: str, problems: list, a: dict | None = None,
            anchor: dict | None = None) -> dict:
    if not rid:
        return {'status': pending('no locked run id in the ledger')}
    d = root / DIR / rid
    res = store.json(d / 'result.json') if (d / 'result.json').is_file() else None
    if isinstance(res, dict) and res.get('skipped'):
        return _d20(rid, arm, res, a, anchor, problems)
    if not isinstance(res, dict) or not isinstance(res.get('candidate'), dict):
        return {'status': pending(f'locked pair {rid} ' + ('running' if d.is_dir() else 'not run'))}
    recs = store.receipts(d)
    halves = {}
    for who, cid in (('baseline', 'lock-base'), ('candidate', 'lock-cand')):
        att = (res.get(who) or {}).get('attempt_id')
        r = recs.get(att)
        if r is None:
            problems.append(f'ENV-C {arm} locked {rid}: {who} receipt {att} missing')
            return {'status': pending(f'locked {who} receipt missing')}
        want = {'env': 'autoresearch', 'task_id': 'ENV-C', 'arm': arm, 'run_id': rid, 'purpose': 'score', 'candidate_id': cid}
        off = {k: r.get(k) for k, v in want.items() if r.get(k) != v}
        if off:
            problems.append(f"ENV-C {arm} locked {rid}: {who} receipt {att} is not this arm's {cid} run ({off})")
            return {'status': pending(f'locked {who} receipt does not identify as {want}')}
        if bool((res.get(who) or {}).get('valid')) != bool(r.get('valid')) or (res.get(who) or {}).get('val_bpb') != _bpb(r):
            problems.append(f'ENV-C {arm} locked {rid}: result.json {who} differs from its receipt {att}')
        halves[who] = r
    b, c = halves['baseline'], halves['candidate']
    sb, sc = src(rid, b), src(rid, c)
    sm, raw = c.get('summary') or {}, (c.get('metric') or {}).get('raw') or {}
    # identity: the candidate half trains the arm's final train.py (the selected experiment's; the baseline when the arm
    # selected nothing / its start)
    ident = None
    if a is not None and isinstance(a.get('trace'), dict):
        sel = a['trace'].get('selected')
        if sel is None:                                 # nothing selected: the arm falls back to the baseline train.py
            exp = _train_sha(b)
        else:                                           # the selected experiment's own receipt (unknown id: unverifiable)
            scr = next((x for x in a['trace'].get('screens') or [] if x.get('candidate_id') == sel), None)
            exp = _train_sha(a['receipts'].get((scr or {}).get('attempt_id'))) if scr else None
        if exp and _train_sha(c):
            ident = exp == _train_sha(c)
    if ident is False:
        problems.append(f"ENV-C {arm} locked {rid}: the candidate half does not train the arm's final train.py")
    elif ident is None:
        problems.append(f"ENV-C {arm} locked {rid}: the candidate half's identity cannot be checked (train.py digests)")
    valid = bool(c.get('valid')) and ident is not False
    known = ident is not None
    out = {'run_id': rid, 'order': cell(res.get('order'), f'{rid}/result.json'),
           'val_bpb_locked': cell(_bpb(c), sc) if valid and _bpb(c) else na('locked candidate invalid'),
           'val_bpb_baseline_half': cell(_bpb(b), sb) if b.get('valid') and _bpb(b) else na('baseline half invalid'),
           'locked_valid': (cell(valid, sc, note=None if ident is not False else 'identity: not the final train.py')
                            if known else pending('identity of the locked candidate unverifiable')),
           'invalid_accept_locked': cell(0 if valid else 1, sc) if known else pending('identity unverifiable'),
           'steps': cell(sm.get('num_steps', raw.get('num_steps')), sc),
           'tok_per_s': cell(sm.get('tok_per_sec', raw.get('median_tok_per_sec')), sc),
           'mfu': cell(sm.get('mfu_percent', raw.get('mfu_percent')), sc),
           'peak_vram_mb': cell(sm.get('peak_vram_mb', raw.get('peak_vram_mb')), sc),
           'gpu_hours': cell(sum(((r.get('cost') or {}).get('gpu_seconds') or 0) for r in halves.values()) / 3600.0, [sb, sc])}
    t_b, t_c = utc(b.get('utc_start')), utc(c.get('utc_start'))
    order = (['baseline', 'candidate'] if t_b < t_c else ['candidate', 'baseline']) if t_b and t_c else None
    if order and res.get('order') and list(res['order']) != order:
        problems.append(f'ENV-C {arm} locked {rid}: recorded order {res.get("order")} differs from the receipts\' start '
                        f'times {order}')
    return out


def anchor_cells(store: Store, root: Path, spec: dict, problems: list, ab_ids: set = frozenset()) -> dict:
    """AMENDMENT_EXT_002 Q2: the anchor is the solo D-AR-7 C-BASE run on the arms' node, valid by the trusted judge.
    An abandoned anchor run is never read (R3-N5)."""
    att, arid = anchor_ref(spec.get('anchor'))
    if arid and in_ids(arid, ab_ids):
        problems.append(f'ENV-C anchor: run {arid} is abandoned (ledger conflict); no anchor is used')
        return {'val_bpb': pending(f'anchor run {arid} is abandoned (ledger conflict)'), 'steps': pending('anchor abandoned')}
    ar = store.receipts(root / DIR / arid).get(att) if arid else None
    if ar is None:
        return {'val_bpb': pending(f'anchor receipt {att} ({arid}) missing'), 'steps': pending('anchor missing')}
    if not spec.get('node'):
        problems.append("ENV-C: the ledger names no arms' node (ENV-C.node); the anchor cannot be bound to it")
        return {'val_bpb': pending('ledger ENV-C.node missing'), 'steps': pending('anchor not bound')}
    want = {'env': 'autoresearch', 'arm': 'C-BASE', 'task_id': 'ENV-C', 'valid': True, 'run_id': arid, 'node': spec['node']}
    bad = {k: ar.get(k) for k, v in want.items() if ar.get(k) != v}
    if 'D-AR-7' not in (ar.get('deviations') or []):
        bad['deviations'] = ar.get('deviations')
    if bad:
        problems.append(f'ENV-C anchor {att} ({arid}) is not the D-AR-7 C-BASE run on the arms\' node: {bad}')
        return {'val_bpb': pending(f'anchor {att} is not the D-AR-7 C-BASE run on the arms\' node'),
                'steps': pending('anchor not usable')}
    s = src(arid, ar)
    return {'val_bpb': cell(_bpb(ar), s), 'steps': cell((ar.get('summary') or {}).get('num_steps'), s),
            'node': ar.get('node'), 'deviations': ar.get('deviations'), 'train_sha': _train_sha(ar)}


def frozen_space(store: Store, problems: list) -> dict:
    """AMENDMENT_EXT_007: the frozen knob_space_C.json (its sha256 is in every C-RND receipt)."""
    p = Path(AK.__file__).resolve().parent / AK.KNOB_SPACE_FILE
    b = store.read_bytes(p, role='tool data')
    if b is None:
        problems.append(f'ENV-C: frozen C-RND space {p.name} missing')
        return {}
    d = json.loads(b.decode('utf-8'))
    return {'sha256': hashlib.sha256(b).hexdigest(), 'size_without_baseline': d.get('size_without_baseline'),
            'size': d.get('size')}


def rnd_space_checks(a: dict, space: dict, problems: list) -> dict:
    """C-RND's trace and launched receipts must name the frozen space; coverage recomputed as distinct launched points
    (harness-refused screens excluded) / the frozen space size without the baseline."""
    tr, recs, rid = a['trace'], a['receipts'], a['run_id']
    if not space:
        return {'space_coverage_recomputed': pending('frozen space missing')}
    ex = tr.get('extra') if isinstance(tr.get('extra'), dict) else {}
    names = {ex.get('knob_space_sha256')} | {r.get('knob_space_sha256') for r in recs.values()
                                              if not isinstance(r.get('runner_error'), dict)}
    if names != {space['sha256']}:
        problems.append(f'ENV-C C-RND {rid}: knob space digests {sorted(map(str, names))} are not the frozen '
                        f'knob_space_C.json {space["sha256"][:16]}')
    cand = {c.get('id'): c.get('config_digest') for c in tr.get('candidates') or [] if isinstance(c, dict)}
    pts = {cand.get(s.get('candidate_id')) for s in tr.get('screens') or []
           if s.get('candidate_id') != 'c0' and not isinstance((recs.get(s.get('attempt_id')) or {}).get('runner_error'), dict)}
    pts.discard(None)
    n = space.get('size_without_baseline')
    cov = len(pts) / n if n else None
    out = {'space_coverage_recomputed': cell(cov, [f'{rid}/trace.json', 'knob_space_C.json'],
                                             note=f'{len(pts)} distinct launched points of {n}'),
           'knob_space_frozen_sha256': cell(space['sha256'], 'knob_space_C.json')}
    tc = ex.get('space_coverage')
    if cov is not None and finite(tc) and abs(tc - cov) > 1e-9:
        out['space_coverage_note'] = (f'trace says {tc} (screens incl. refused ones); recomputed {cov:.4f} from distinct '
                                      'launched points')
    return out


def _compare(a: str, b: str, arms: dict, base: dict) -> dict:
    for x in (a, b):
        if arms[x]['status'] == 'not run':
            return {'label': 'not run', 'why': f'{x}: {arms[x].get("why")}'}
    for x in (a, b):
        if arms[x]['status'] != 'finished':
            return {'label': 'pending', 'why': f'{x} {arms[x]["status"]}'}
    rows = {}
    for x in (a, b):
        lk = arms[x].get('locked') or {}
        if not ok(lk.get('locked_valid')):
            return {'label': 'pending', 'why': f'{x} locked pair: {(lk.get("status") or {}).get("why")}'}
        auc = arms[x]['search'].get('AUC_budget')
        if not ok(auc):
            return {'label': 'pending', 'why': f'{x} AUC_budget pending'}
        v = bool(val(lk['locked_valid']))
        row = {'locked_valid': v, 'invalid_accept_locked': val(lk['invalid_accept_locked']), 'AUC_budget': val(auc),
               'accounting_complete': arms[x]['search']['accounting'][0],
               'accounting_reasons': arms[x]['search']['accounting'][1]}
        if v:
            row['y_locked'] = val(base) - val(lk['val_bpb_locked'])
        rows[x] = row
    res = TJ.compare(score_task(a), rows[a], rows[b])
    da, db = (arms[x]['locked'].get('val_bpb_baseline_half') for x in (a, b))
    res['drift_baseline_halves'] = (val(da) - val(db)) if ok(da) and ok(db) else None
    res['src'] = sorted({s for x in (a, b) for c in (arms[x]['locked'].get('locked_valid'), arms[x]['locked'].get('val_bpb_locked'),
                                                      arms[x]['search'].get('AUC_budget'), base)
                         for s in ((c or {}).get('src') or [])})
    return res


def calibration(store: Store, root: Path, spec: dict, anchor: dict, problems: list, ab_ids: set = frozenset()) -> dict:
    """G-C (§4.3, AMENDMENT_EXT_002 Q2): C-BASE steps within 2 % of the anchor's, the judge-loader equivalence, and
    the pre-arm feasibility evidence behind the pruned C-RND space (AMENDMENT_EXT_007)."""
    rid = spec.get('calibration')
    out = {'run_id': rid}
    if not rid:
        return {**out, 'gate': pending('no calibration run in the ledger')}
    if in_ids(rid, ab_ids):                             # R3-N5: an abandoned run is never gate evidence
        problems.append(f'ENV-C calibration: run {rid} is abandoned (ledger conflict); not used as G-C evidence')
        return {**out, 'gate': pending(f'calibration run {rid} is abandoned (ledger conflict)')}
    recs = store.receipts(root / DIR / rid)
    base = next((r for r in recs.values() if r.get('arm') == 'C-BASE'), None)
    if base is None or not ok(anchor.get('steps')):
        return {**out, 'gate': pending('calibration C-BASE receipt or anchor missing')}
    steps = (base.get('summary') or {}).get('num_steps')
    rel = abs(steps - val(anchor['steps'])) / val(anchor['steps']) if finite(steps) else None
    s = src(rid, base)
    le = next((g for g in base.get('guard_reports') or [] if g.get('id') == 'judge_loader_equivalence'), None)
    out.update(c_base_steps=cell(steps, s), anchor_steps=anchor['steps'], rel_dev=cell(rel, [s] + anchor['steps'].get('src', [])),
               gate=cell(bool(rel is not None and rel <= AT.CALIB_GATE), [s] + anchor['steps'].get('src', []),
                         note=f'|Δsteps| <= {AT.CALIB_GATE:.0%}'),
               loader_equivalence=cell((le or {}).get('abs_dev'), s, note='ok' if (le or {}).get('ok') else 'FAILED / missing'),
               removed_values=[{'knob': x['knob'], 'value': x['value'], 'error_class': x['error_class'],
                                'evidence': x['evidence']} for x in AK.REMOVED],
               space_size=len(AK.points()), space_size_without_baseline=len(AK.sample_order()))
    if le is None or not le.get('ok'):
        problems.append('ENV-C calibration: judge-loader equivalence missing or failed')
    return out


def build(store: Store, root: Path, ledger: dict, *, abandoned: list) -> dict:
    spec = ledger.get('ENV-C') or {}
    problems: list[str] = []
    ab_ids = {x['run_id'] for x in abandoned if x['env'] == 'ENV-C'}
    anchor = anchor_cells(store, root, spec, problems, ab_ids)
    base = anchor['val_bpb']
    space = frozen_space(store, problems)
    arms = {}
    for arm in ARMS:
        a = _load(store, root, (spec.get('arms') or {}).get(arm), ab_ids)
        row = {'arm': arm, 'run_id': a.get('run_id'), 'status': a['status'], 'why': a.get('why'),
               'track': 'free' if arm == 'C-NAT-free' else 'infra'}
        if a['status'] == 'finished':
            row['search'] = _search(a, arm, base, problems)
            if arm == 'C-RND':
                row['search'].update(rnd_space_checks(a, space, problems))
            lspec = (spec.get('locked') or {}).get(arm)
            row['locked'] = (_locked(store, root, run_id_of(lspec), arm, problems, a, anchor)
                             if not in_ids(run_id_of(lspec), ab_ids)
                             else {'status': pending(f'locked run {lspec} is abandoned (ledger conflict)')})
        else:
            row['locked'] = {'status': missing(row['status'], row.get('why'))}
        arms[arm] = row
    labels = {f'{a} vs {b}': _compare(a, b, arms, base) for a, b in (MAIN,) + RND_PAIRS}
    cov = (arms['C-RND'].get('search') or {}).get('space_coverage_recomputed')
    cov_txt = f' (C-RND covered {R.fmt(cov, 3)} of its pruned space)' if cov and ok(cov) else ''
    sentences = []
    for k in ('C-IS vs C-RND', 'C-NAT-infra vs C-RND'):
        lab = labels[k].get('label')
        if lab == 'advantage':
            sentences.append(f'ENV-C: LLM search ({k.split(" vs ")[0]}) beats random search (C-RND) beyond 0.002 bpb on every '
                             f'condition: advantage{cov_txt}.')
        elif lab == 'not run':
            sentences.append(f'ENV-C: {k} not compared ({labels[k].get("why")}).')
        elif lab == 'incomplete':
            sentences.append(f'ENV-C: {k} incomplete (accounting); the label if complete would be '
                             f'{labels[k].get("label_if_complete")}{cov_txt}.')
        elif lab in ('parity', 'disadvantage', 'invalid'):
            sentences.append(f'ENV-C: there is no evidence that LLM search ({k.split(" vs ")[0]}) beats random search '
                             f'(C-RND): {lab}{cov_txt}.')
        elif lab == 'no_advantage':
            sentences.append(f'ENV-C: {k.split(" vs ")[0]} beats C-RND beyond 0.002 bpb but not on every condition: '
                             f'no_advantage{cov_txt}.')
    pend = [f'ENV-C anchor: {c.get("why")}' for c in (anchor['val_bpb'],) if c.get('status') == 'pending']
    cal = calibration(store, root, spec, anchor, problems, ab_ids)
    if not ok(cal.get('gate')) or val(cal.get('gate')) is not True:
        pend.append(f"ENV-C G-C calibration: {R.fmt(cal.get('gate'))} ({(cal.get('gate') or {}).get('why') or 'gate not passed'})")
    for arm, row in arms.items():
        if row['status'] == 'not run':
            continue                                    # a sanctioned final state (ledger), not pending
        if row['status'] != 'finished':
            pend.append(f'ENV-C {arm}: {row["run_id"]} {row["status"]}' + (f' ({row["why"]})' if row.get('why') else ''))
        elif isinstance((row.get('locked') or {}).get('status'), dict):
            pend.append(f'ENV-C {arm} locked: {row["locked"]["status"].get("why")}')
    ab = [{**x, 'cost': run_cost(store, root / DIR / x['run_id'], world_size=1)} for x in abandoned if x['env'] == 'ENV-C']
    fnd = [f for r in arms.values() for f in (r.get('search') or {}).get('findings') or []]
    return {'anchor': anchor, 'calibration': cal, 'arms': arms,
            'labels': labels, 'random_sentences': sentences, 'abandoned': ab, 'problems': problems, 'pending': pend,
            'findings': fnd}


# ---------------------------------------------------------------- markdown
def markdown(sec: dict) -> str:
    a, cal = sec['anchor'], sec['calibration']
    lab = sec['labels']['C-IS vs C-NAT-infra']
    L = ['# ENV-C — autoresearch (1×H100 per arm, 300 s training, trusted `judge_bpb`)', '',
         'Design §4.4 / §0.3 with AMENDMENT_EXT_002 Q2 and AMENDMENT_EXT_007. All arms incl. C-NAT-free run with the '
         'one-line race fix D-AR-7 (`h2d_sync`); the published example is compared against the D-AR-7 C-BASE, not R2; '
         'the H2D race is an upstream bug we report (REPRO_ISSUES #20). Margin 0.002 bpb (fixed); y = bpb_C-BASE − bpb.', '',
         f"C-BASE anchor (arms' node {a.get('node')}): val_bpb {R.fmt(a['val_bpb'], 6)}, {R.fmt(a['steps'])} steps.", '',
         f"**Infra-track label (C-IS vs C-NAT-infra): {R.label_text(lab)}**"
         + (f" — Δy {R.fmt(lab.get('delta_y'), 5)}, margin {R.fmt(lab.get('margin_y'), 3)}" if lab.get('delta_y') is not None else '')
         + (f" ({lab.get('why')})" if lab.get('why') else ''), '',
         'LLM vs random: ' + (' '.join(sec['random_sentences']) or 'pending (' + '; '.join(
             f"{k} {r.get('label')}" for k, r in sec['labels'].items() if k.endswith('vs C-RND')) + ').'), '']
    t = []
    for arm, r in sec['arms'].items():
        s, lk = r.get('search') or {}, r.get('locked') or {}
        lw = lk['status'] if isinstance(lk.get('status'), dict) else na('—')
        p = lambda k: s.get(k, missing(r['status'], r.get('why')))  # noqa: E731
        q = lambda k: lk.get(k, lw)  # noqa: E731
        t.append([arm, r['track'], r['run_id'], r['status'], q('val_bpb_locked'), q('val_bpb_baseline_half'),
                  p('val_bpb_best_search'), q('steps'), q('tok_per_s'), q('mfu'), q('peak_vram_mb'), p('n_experiments'),
                  p('n_crash'), p('n_invalid_check'), p('n_failed_by_class'), q('invalid_accept_locked'),
                  p('CTG_gpu_hours'), p('AUC_budget'), p('gpu_hours'), p('llm_tokens'), p('llm_usd'), p('wall_hours'),
                  s.get('space_coverage_recomputed', na('—')) if arm == 'C-RND' else na('—'), p('trials_used'),
                  accounting_text((s['accounting'][0], s['accounting'][1], None, s.get('accounting_reconciliation')))
                  if s.get('accounting') else 'n/a'])
    L += [R.table(['arm', 'track', 'run', 'status', 'val_bpb_locked', 'baseline half', 'val_bpb_best_search', 'steps',
                   'tok/s', 'MFU %', 'peak VRAM MB', 'n_experiments', 'n_crash', 'n_invalid_check', 'n_failed_by_class',
                   'invalid_accept_locked', 'CTG(−0.002) GPUh', 'AUC_budget', 'GPUh', 'LLM tokens', 'USD', 'wall h',
                   'C-RND coverage', 'trials used / budget (stop reason)', 'accounting'], t,
                  {4: 6, 5: 6, 6: 6, 16: 3, 17: 5, 18: 3, 20: 2, 21: 2, 22: 3}), '']
    for arm, r in sec['arms'].items():
        if (r.get('locked') or {}).get('skipped'):
            L += [f"{arm}: locked pair {r['locked'].get('run_id')} `skipped` (D20: the final train.py is the baseline); "
                  f"locked value = the C-BASE anchor {R.fmt(r['locked'].get('val_bpb_locked'), 6)}, y = 0, InvalidAccept 0.", '']
    t = [[k, R.label_text(r), r.get('delta_y'), r.get('margin_y'), r.get('drift_baseline_halves'),
          ', '.join(f'{c}={"yes" if v else "no"}' for c, v in (r.get('conditions') or {}).items()) or r.get('why') or '']
         for k, r in sec['labels'].items()]
    L += [R.table(['pair', 'label', 'Δy (bpb)', 'margin', 'drift (baseline halves a−b)', 'conditions / why'], t,
                  {2: 5, 3: 3, 4: 5}), '']
    free = sec['arms']['C-NAT-free']
    fs = free.get('search') or {}
    fm = missing(free['status'], free.get('why'))       # R3-N4: a sanctioned cut reads n/a, never pending
    L += ['## C-NAT-free — published-protocol reproduction (not an infra comparison)', '',
          'Free track (`program.md` verbatim, model semantics may change, printed val_bpb as in the published protocol); it '
          'changes model semantics, so it is never compared with the infra arms. ' + README_FIGURE + '.', '',
          f"Status {free['status']}" + (f" ({free.get('why')})" if free.get('why') else '')
          + f"; experiments {R.fmt(fs.get('n_experiments', fm))}; best printed val_bpb "
          f"{R.fmt(fs.get('val_bpb_best_search', fm), 6)}; locked "
          f"{R.fmt((free.get('locked') or {}).get('val_bpb_locked', fm), 6)}; trials {R.fmt(fs.get('trials_used', fm))}.", '']
    L += ['## G-C calibration and the pruned C-RND space (AMENDMENT_EXT_007)', '']
    if 'c_base_steps' in cal:
        L += [f"Calibration {cal['run_id']}: C-BASE {R.fmt(cal['c_base_steps'])} steps vs anchor {R.fmt(cal['anchor_steps'])} "
              f"(|Δ| {R.fmt(cal['rel_dev'], 4)}; gate {R.fmt(cal['gate'])}); judge-loader equivalence |Δval_bpb| "
              f"{R.fmt(cal['loader_equivalence'], 8)} ({cal['loader_equivalence'].get('note')}). C-RND space: "
              f"{cal['space_size']} points ({cal['space_size_without_baseline']} without the baseline) after removing:", '']
        L += [R.bullets(f"{x['knob']} = {x['value']} ({x['error_class']}): {'; '.join(x['evidence'])}"
                        for x in cal['removed_values']), '']
    else:
        L += [f"Calibration: {R.fmt(cal.get('gate'))} ({(cal.get('gate') or {}).get('why')})", '']
    L += ['## Abandoned runs (pipeline-fix line; not results)', '']
    t = [[x['run_id'], x['reason'], x['cost'].get('gpu_hours'), x['cost'].get('llm_tokens'), x['cost'].get('llm_usd'),
          x['cost'].get('wall_hours'), x['cost'].get('basis')] for x in sec['abandoned']]
    L += [R.table(['run', 'reason', 'GPUh', 'LLM tokens', 'USD', 'wall h', 'basis'], t, {2: 3, 4: 2, 5: 2}), '']
    L += ['## Findings in the records (facts for O; not tool problems)', '', R.bullets(sec['findings']), '']
    L += ['## Pending and problems', '', '**Pending**', '', R.bullets(sec['pending']), '', '**Problems**', '',
          R.bullets(sec['problems']), '']
    return '\n'.join(L)
