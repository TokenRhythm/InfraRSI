"""F1 fallback (AMENDMENT_EXT_003): X6's own summary step, called as is.

`ext.flashnext.cli summary` loads the U+C reference screen (`<UC-REF run>/uc_ref.json`), F-HT (`<F-HT run>/ht.json`)
and each arm's run directory (sealed trace.json + locked*.json + receipts), recomputes margin_F1.json from the window's
receipts and writes summary_F1.json (rows, Q5 labels, matched prefix, F-HT line, drift, claims). Here it is called with
--out = the summary's output directory (and --allow-nonformal only under --allow-partial); its result is rendered and
its numbers are cited as 'summary_F1.json:<arm>'. Nothing of the F1 rule is re-implemented.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

from . import render as R
from .cells import cell, na, pending
from .records import Store, accounting_text, accounting_verified, in_ids, not_run_reason, run_cost, run_id_of, trials_used

NO_COMPARISON = 'no comparison (harness)'                    # EXT_009 Q4 (X6's arms.NO_COMPARISON)


def disclosure() -> str:
    """AMENDMENT_EXT_009 Q5's fixed sentence, from X6's arms.EXT009_DISCLOSURE (one source)."""
    from ..flashnext import arms as FA
    return FA.EXT009_DISCLOSURE

DIR = 'flashnext'
ARMS = ('F-PROMPTS', 'F-OPTICO', 'F-A0', 'F-RND')
ROW_KEYS = ('step_time_locked', 'speedup_vs_UC', 'recovered_fraction', 'reached_1_30x', 'numerics_class', 'InvalidAccept',
            'AUC_budget', 'n_invalid_proposal', 'locked_status', 't_uc_pair', 'drift_uc_pair_over_ref')


def _hash_run(store: Store, d: Path) -> None:
    for p in sorted(d.glob('receipts/*.json')) + sorted(d.glob('*.json')):
        store.read_bytes(p, role='f1 record')


def build(store: Store, root: Path, ledger: dict, out_dir: Path, *, allow_partial: bool, runner=None) -> dict:
    spec = ledger.get('F1') or {}
    refs, arms = spec.get('references') or {}, spec.get('arms') or {}
    out = {'nodes': spec.get('nodes'), 'pending': [], 'problems': [], 'summary': None, 'refs': refs, 'arms_ledger': arms,
           'not_run': None, 'arms_not_run': {}, 'trials': {}, 'accounting': {}}
    ab = [x for x in ledger_abandoned(ledger, root) if x['env'] == 'F1']
    ab_ids = {x['run_id'] for x in ab}
    out['abandoned'] = [{**x, 'cost': run_cost(store, root / DIR / x['run_id'], world_size=32)} for x in ab]   # R3-N7
    st = str(spec.get('status') or '')
    if st.lower().replace('_', ' ').startswith('not run'):
        out['not_run'] = st                             # AMENDMENT_EXT_003 Q6: F1 dropped, every arm `not_run (...)`
        return out
    for k in ('UC-REF', 'F-HT'):
        if in_ids(run_id_of(refs.get(k)), ab_ids):
            out['problems'].append(f'F1 {k}: run {run_id_of(refs.get(k))} is abandoned (ledger conflict)')
            out['pending'].append(f'F1 {k}: the ledger slot holds an abandoned run')
            return out
    uc_dir = root / DIR / run_id_of(refs.get('UC-REF')) if run_id_of(refs.get('UC-REF')) else None
    if uc_dir is None or not (uc_dir / 'uc_ref.json').is_file():
        out['pending'].append('F1: the U+C reference screen has not run (RUN_LEDGER F1.references.UC-REF '
                              f'{refs.get("UC-REF")!r}); every F1 arm is pending')
        return out
    ht = root / DIR / run_id_of(refs.get('F-HT')) / 'ht.json' if run_id_of(refs.get('F-HT')) else None
    if ht is None or not ht.is_file():
        out['pending'].append('F1: F-HT (S_match on the node set) not run')
    dirs = {}
    for arm in ARMS:
        why = not_run_reason(arms.get(arm))
        rid = run_id_of(arms.get(arm))
        if why:
            out['arms_not_run'][arm] = why              # sanctioned (ledger), not pending
        elif not rid:
            out['pending'].append(f'F1 {arm}: not started (ledger null)')
        elif in_ids(rid, ab_ids):
            out['problems'].append(f'F1 {arm}: run {rid} is abandoned (ledger conflict)')
        elif not (root / DIR / rid / 'trace.json').is_file():
            out['pending'].append(f'F1 {arm}: {rid} running or not started')
        else:
            dirs[arm] = root / DIR / rid
            out['trials'][arm] = trials_used(store.json(root / DIR / rid / 'trace.json'), f'{rid}/trace.json')
            acc = _accounting(store, root / DIR / rid, uc_dir)
            out['accounting'][arm] = {'complete': acc[0], 'reasons': acc[1], 'reconciliation': acc[3],
                                      'text': accounting_text(acc)}
            if not acc[0]:                              # D27: an unverified F1 search cannot be formal
                out['pending'].append(f'F1 {arm} ({rid}): accounting ' + accounting_text(acc))
    for d in [uc_dir] + ([ht.parent] if ht is not None and ht.is_file() else []) + list(dirs.values()):
        _hash_run(store, d)
    out['references'] = _references(store, spec, uc_dir, ht)
    out['ext008'] = _ext008(store, root, spec, out['problems'])
    out['ext009'] = [{'arm': arm, **x} for arm, d in dirs.items() for x in _superseding(store, d)]
    if not dirs:
        return out
    args = ['summary', '--uc-ref', str(uc_dir / 'uc_ref.json')] + (['--ht', str(ht)] if ht is not None and ht.is_file() else [])
    for arm, d in dirs.items():
        args += ['--arm', f'{arm}={d}']
    cut = sorted(out['arms_not_run'])
    args += ['--out', str(out_dir)] + (['--allow-nonformal'] if allow_partial or cut else [])
    if runner is None:
        from ..flashnext import cli as runner
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = runner.main(args)
    out['cli'] = {'args': args, 'exit': rc}
    p = Path(out_dir) / 'summary_F1.json'
    if rc != 0 or not p.is_file():
        out['pending'].append(f'F1: X6 summary refused (exit {rc}): {buf.getvalue().strip()[-1500:]}')
        return out
    s = json.loads(p.read_text(encoding='utf-8'))
    out['summary'] = s
    reasons = list(s.get('nonformal_reasons') or [])
    if cut:                                             # X6 requires all four arms; a sanctioned cut is the only allowed gap
        allowed = f'arms missing from the window: {cut}'
        out['cut_reason_accepted'] = allowed if allowed in reasons else None
        reasons = [x for x in reasons if x != allowed]
    if not s.get('formal') and reasons:
        out['pending'] += [f'F1 not formal: {x}' for x in reasons]
    elif not s.get('formal') and not (cut and out.get('cut_reason_accepted')):
        out['pending'].append('F1: X6 summary not formal')
    return out


def _canon(r) -> str:
    return json.dumps(r, sort_keys=True, default=str)


def _receipt_of(store: Store, p: Path | None) -> dict:
    d = store.json(p) if p is not None and p.is_file() else None
    if not isinstance(d, dict):
        return {}
    return d['receipt'] if isinstance(d.get('receipt'), dict) else d


def _references(store: Store, spec: dict, uc_dir: Path | None, ht: Path | None) -> list[dict]:
    """The kept references (EXT_008 Q3): their receipts carry the superseded S_F1 sha; printed beside the current one."""
    out = []
    for kind, p in (('UC-REF', uc_dir / 'uc_ref.json' if uc_dir is not None else None), ('F-HT', ht)):
        r = _receipt_of(store, p)
        v = (r.get('metric') or {}).get('value') if r else None
        out.append({'kind': kind, 'run_id': run_id_of((spec.get('references') or {}).get(kind)),
                    'attempt_id': r.get('attempt_id'), 'valid': r.get('valid'), 'step_time_s': v,
                    'receipt_switch_space_sha256': r.get('switch_space_sha256'),
                    'current_switch_space_sha256': spec.get('switch_space_sha256'),
                    'superseded_switch_space_sha256': spec.get('switch_space_superseded_sha256')})
    return out


def _ext008(store: Store, root: Path, spec: dict, problems: list) -> dict | None:
    """AMENDMENT_EXT_008's freeze record (flashnext/FREEZE_EXT008.json), hashed into FREEZE.json."""
    if not spec.get('freeze_ext008'):
        return None
    p = root / DIR / Path(str(spec['freeze_ext008'])).name
    if store.read_bytes(p, role='f1 freeze (AMENDMENT_EXT_008)') is None:
        problems.append(f'F1: the ledger names {spec["freeze_ext008"]}, which is not in the records')
        return None
    return {'file': f'{DIR}/{p.name}', 'sha256': store.files[str(p.resolve())]['sha256']}


def _superseding(store: Store, d: Path) -> list[dict]:
    """The EXT_009 superseding records of an arm's directory (FREEZE.json cites their attempt ids)."""
    out = []
    for att, recs in sorted(store.superseding(d).items()):
        for x in recs:
            r = x['record']
            out.append({'run_id': r.get('run_id'), 'attempt_id': att, 'candidate_id': r.get('candidate_id'),
                        'collection': r.get('collection'), 'file': f'{d.name}/receipts/{Path(x["path"]).name}',
                        'sha256': (store.files.get(x['path']) or {}).get('sha256')})
    return out


def _accounting(store: Store, d: Path, uc_dir: Path | None) -> tuple:
    """D27 for an F1 arm (X6's records; every arm runs on SearchSession: GPU = the counted screens' receipts, the
    events.jsonl order is the call-to-row boundary). Declared non-screen charges in the arm's receipts (R6-N4): the
    U+C start reference only as receipts/c0.json, SearchSession's copy of the UC-REF receipt (same attempt, same
    content; recorded, not charged to the arm); the halves named by locked*.json, each a purpose-`locked` receipt named
    for its half (X6's half_id: locked-uc / locked-candidate[-aN])."""
    tr = store.json(d / 'trace.json')
    tr = tr if isinstance(tr, dict) else {}
    recs = store.receipts(d)
    ns, errs = {}, []
    sup = [x['record'] for v in store.superseding(d).values() for x in v]
    if sup:                                              # EXT_009: X6's formal rule on the superseding records
        from ..flashnext import supersede as SU
        errs += [f'EXT_009: {x}' for x in SU.index(list(recs.values()) + sup)[2]]
    u = store.json(uc_dir / 'uc_ref.json') if uc_dir is not None and (uc_dir / 'uc_ref.json').is_file() else None
    if isinstance(u, dict):                              # R7-N1: with a U+C reference, its c0.json copy is mandatory
        uc_rec = u['receipt'] if isinstance(u.get('receipt'), dict) else u
        att = u.get('receipt_attempt') or uc_rec.get('attempt_id')
        c0 = store.json(d / 'receipts' / 'c0.json') if (d / 'receipts' / 'c0.json').is_file() else None
        if c0 is None:
            errs.append(f'receipts/c0.json, the copy of the U+C reference receipt ({att}), is missing')
        elif isinstance(c0, dict) and att and c0.get('attempt_id') == att and _canon(c0) == _canon(uc_rec):
            ns[att] = 'start reference (U+C, charged to UC-REF)'
        else:
            errs.append(f'receipts/c0.json is not the copy of the U+C reference receipt ({att})')
    for p in sorted(d.glob('locked*.json')):
        lk = store.json(p)
        if not isinstance(lk, dict):
            errs.append(f'{p.name}: not a JSON object')
            continue
        if not (lk.get('locked_run') is True or 'uc_half' in lk or 'candidate_half' in lk):
            continue                                     # the ruled no-improvement case: no locked run, no pair
        for half, prefix in (('uc_half', 'locked-uc'), ('candidate_half', 'locked-candidate')):
            att = (lk.get(half) or {}).get('attempt_id') if isinstance(lk.get(half), dict) else None
            if not att:                                  # R7-N1: a locked run declares both halves
                errs.append(f'{p.name}: a locked run without its {half} attempt id')
                continue
            ns[att] = f'locked pair {half.replace("_", " ")} ({p.name})'
            r = recs.get(att)
            if r is not None and not (r.get('purpose') == 'locked' and str(r.get('candidate_id') or '').startswith(prefix)):
                errs.append(f'{p.name} {half} names {att}, which is not the {prefix} half '
                            f'({r.get("purpose")} / {r.get("candidate_id")})')
    return accounting_verified(tr, recs, store.event_sequence(d / 'events.jsonl'), non_screen=ns, producer='infrarsi',
                               extra_fails=errs)


def ledger_abandoned(ledger: dict, root=None) -> list:
    from .records import abandoned
    return abandoned(ledger, root)


def status(sec: dict, arm: str) -> str:
    """R3 / table status of one F arm: sanctioned not run, formal result, or pending X6 output."""
    if sec.get('not_run'):
        return sec['not_run']                           # the ledger's own wording, e.g. 'not_run (G-F1 failed: ...)'
    if arm in (sec.get('arms_not_run') or {}):
        return f"not run ({sec['arms_not_run'][arm]})"
    s = sec.get('summary') or {}
    if arm in (s.get('rows') or {}):
        return 'migrated (formal F1 result)' if s.get('formal') else 'migrated (F1 result not formal yet)'
    return 'migrated (F1 window pending)'


def rows(sec: dict) -> dict:
    """{arm: {key: cell}} from summary_F1.json (source: summary_F1.json:<arm>)."""
    s = sec.get('summary') or {}
    out = {}
    for arm in ARMS:
        r = (s.get('rows') or {}).get(arm)
        acc = (sec.get('accounting') or {}).get(arm)
        if sec.get('not_run') or arm in (sec.get('arms_not_run') or {}):
            out[arm] = {k: na(status(sec, arm)) for k in ROW_KEYS + ('costs', 'CTG', 'trials_used', 'accounting')}
            continue
        if r is None:
            out[arm] = {k: pending('F1 arm not summarized') for k in ROW_KEYS + ('costs',)}
            out[arm]['trials_used'] = (sec.get('trials') or {}).get(arm) or pending('F1 arm not run yet')
            out[arm]['accounting'] = cell(acc['text'], 'D27') if acc else pending('F1 arm not run yet')
            continue
        sid = f'summary_F1.json:{arm}'
        out[arm] = {k: cell(r.get(k), sid) for k in ROW_KEYS}
        out[arm]['CTG'] = cell(r.get('CTG'), sid)
        out[arm]['costs'] = cell(r.get('costs'), sid)
        out[arm]['trials_used'] = (sec.get('trials') or {}).get(arm) or na('no trace')
        out[arm]['accounting'] = cell(acc['text'], 'D27') if acc else na('no trace')
    return out


def markdown(sec: dict) -> str:
    s = sec.get('summary') or {}
    L = ['# F1 fallback — Qwen3.8-Flash-Next production path, implementation-switch track (32×H100)', '',
         'AMENDMENT_EXT_003: one switch space S_F1 for every arm, start U+C, plan-A locked pairs, E1 numerics gates incl. '
         'MTP CE, margin = max(3 %, 3 × RSD(S_UC)); all numbers below come from X6\'s summary step '
         '(`ext.flashnext.cli summary`), called unchanged.', '']
    if sec.get('not_run'):                              # R3-N7: abandoned F1 runs keep their cost line
        if sec.get('abandoned'):
            L += [f"**F1: {sec['not_run']}** — every F1 arm `not_run`; no accepted F1 result runs (AMENDMENT_EXT_003 Q6). "
                  'The abandoned F1 runs below are pipeline-fix cost, not results.', ''] + _abandoned_md(sec)
        else:
            L += [f"**F1: {sec['not_run']}** — every F1 arm `not_run`; no GPU spent on F1 arms (AMENDMENT_EXT_003 Q6).", '']
        return '\n'.join(L)
    note = [f'EXT_009 disclosure: {disclosure()}.', '']
    L += _references_md(sec, note)
    if not s:
        L += ['**Pending.** ' + ' '.join(sec['pending'] + sec['problems']), ''] + _abandoned_md(sec, note)
        rws = rows(sec)
        L += [R.table(['arm', 'step_time_locked s', 'speedup_vs_UC', 'recovered_fraction', 'reached_1.30x', 'numerics_class',
                       'InvalidAccept', 'AUC_budget', 'trials used / budget (stop reason)', 'accounting'],
                      [[a] + [rws[a][k] for k in ('step_time_locked', 'speedup_vs_UC', 'recovered_fraction', 'reached_1_30x',
                                                  'numerics_class', 'InvalidAccept', 'AUC_budget', 'trials_used',
                                                  'accounting')]
                       for a in ARMS]), ''] + note
        return '\n'.join(L)
    L += [f"Formal: {'yes' if s.get('formal') else 'no'}" + ('' if s.get('formal') else f" ({'; '.join(s.get('nonformal_reasons') or [])})"),
          f"U+C reference: {R.fmt(s.get('t_uc_ref'), 3)} s (attempt {s.get('uc_ref_attempt')}); margin {s.get('margin')}.", '']
    rws = rows(sec)
    t = [[a] + [rws[a][k] for k in ('step_time_locked', 't_uc_pair', 'speedup_vs_UC', 'recovered_fraction', 'reached_1_30x',
                                    'numerics_class', 'InvalidAccept', 'AUC_budget', 'n_invalid_proposal', 'locked_status',
                                    'trials_used', 'accounting')]
         for a in ARMS]
    L += [R.table(['arm', 'step_time_locked s', 'U+C half s', 'speedup_vs_UC', 'recovered_fraction', 'reached_1.30x',
                   'numerics_class', 'InvalidAccept', 'AUC_budget', 'n_invalid_proposal', 'locked',
                   'trials used / budget (stop reason)', 'accounting'], t,
                  {1: 3, 2: 3, 3: 4, 4: 3, 8: 4}), ''] + note
    for arm in ARMS:                                    # EXT_009 Q4 / Q5: harness fields and the reading of an S_match pick
        r = (s.get('rows') or {}).get(arm) or {}
        h = r.get('harness') or {}
        if h:
            rec = '; '.join(f"{x.get('candidate_id')} {R.fmt(x.get('step_time'), 3)} s"
                            + (' (descriptive only)' if x.get('descriptive_only') else '') for x in h.get('recovered') or [])
            L += [f"{arm} — harness (EXT_009): n_harness_lost {h.get('n_harness_lost')}, n_recovered {h.get('n_recovered')}"
                  + (f' ({rec})' if rec else '') + f", harness_cost_selection {R.fmt(h.get('harness_cost_selection'))}"
                  + f", comparison withheld {R.fmt(h.get('comparison_withheld'))}; failed by class {h.get('n_failed_by_class')}.", '']
        if r.get('recovered_fraction_reading'):
            L += [f"{arm}: {r['recovered_fraction_reading']}.", '']
        if r.get('screens'):
            t = [[x.get('index'), x.get('candidate_id'), x.get('attempt_id'), x.get('collection'), x.get('class'),
                  x.get('valid'), x.get('step_time'), x.get('descriptive_only')] for x in r['screens']]
            L += [f'### {arm} screens', '', R.table(['#', 'candidate', 'attempt', 'collection', 'class', 'valid', 'step_time s',
                                                    'descriptive only'], t, {6: 3}), ''] + note
    t = []
    for k, v in (s.get('labels') or {}).items():
        why = (f"withheld by {v.get('withheld_by')}: {v.get('reason')} (computed: {(v.get('computed') or {}).get('label')})"
               if v.get('label') == NO_COMPARISON else
               ', '.join(f'{c}={"yes" if x else "no"}' for c, x in (v.get('conditions') or {}).items()) or v.get('reason') or '')
        t.append([k, v.get('label'), v.get('delta_y'), v.get('margin'), why])
    L += [R.table(['pair', 'label', 'Δy', 'margin', 'conditions / why'], t, {2: 4, 3: 4}), ''] + note
    if s.get('F-PROMPTS matched prefix'):
        m = s['F-PROMPTS matched prefix']
        L += [f"F-PROMPTS matched prefix (outside the main criterion): {m.get('label')} (F-PROMPTS best "
              f"{R.fmt(m.get('F-PROMPTS_best'), 3)} s vs F-A0 first-3 best {R.fmt(m.get('F-A0_first3_best'), 3)} s).", '']
    if s.get('F-HT'):
        h = s['F-HT']
        L += [f"F-HT — {h.get('line')}: {R.fmt(h.get('step_time_s'), 3)} s (×{R.fmt(h.get('speedup_vs_uc_ref'), 3)} vs the U+C "
              f"reference, {h.get('numerics_class')}).", '']
    for arm, txt in (s.get('random_control') or {}).items():
        L += [f'Random control: {arm} — {txt}.']
    L += ['', f"Drift of the pairs' U+C halves: {s.get('drift')}", '', f"Claims: {s.get('claims')}", '',
          f"Screen level: {s.get('screen_level')}", '']
    if sec.get('arms_not_run'):
        L += ['Arms not run (ledger): ' + '; '.join(f'{a}: {w}' for a, w in sec['arms_not_run'].items()), '']
    L += _abandoned_md(sec, note) + ['## Pending and problems', '', R.bullets(sec['pending'] + sec['problems']), '']
    return '\n'.join(L)


def _references_md(sec: dict, note: list) -> list:
    refs = sec.get('references') or []
    if not refs:
        return []
    t = [[x['kind'], x.get('run_id'), x.get('attempt_id'), x.get('valid'), x.get('step_time_s'),
          x.get('receipt_switch_space_sha256'), x.get('current_switch_space_sha256')] for x in refs]
    sup = (refs[0] or {}).get('superseded_switch_space_sha256')
    out = ['## References (kept under AMENDMENT_EXT_008)', '',
           R.table(['reference', 'run', 'attempt', 'valid', 'step_time s', 'S_F1 sha in its receipt',
                    'current S_F1 sha (EXT_008)'], t, {4: 3}), '']
    if sup:
        out += [f'S_F1 {sup[:12]}… is superseded by EXT_008 (677 → 353 admissible points); the references were kept '
                'because only the space file, pin, manifest sha and CPU tests changed (D28).', '']
    ext8 = sec.get('ext008')
    if ext8:
        out += [f"EXT_008 freeze record: `{ext8['file']}` (sha256 {ext8['sha256'][:16]}…).", '']
    return out + note


def _abandoned_md(sec: dict, note: list | None = None) -> list:
    if not sec.get('abandoned'):
        return []
    t = [[x['run_id'], x['reason'], x['cost'].get('gpu_hours'), x['cost'].get('llm_usd'), x['cost'].get('wall_hours'),
          x['cost'].get('basis')] for x in sec['abandoned']]
    return ['## Abandoned runs (pipeline-fix line; not results)', '',
            R.table(['run', 'reason', 'GPUh', 'USD', 'wall h', 'basis'], t, {2: 2, 3: 2, 4: 2}), ''] + list(note or [])
