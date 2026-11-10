"""Cross-checks of one finished run: REPORT.md against result.json, result.json against the receipts on disk, the
budget against what was used. Every journey and search-oracle test calls `check_run(run_dir)`."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from infrarsi.harness import signing
from infrarsi.verify import Verifier


def _fmt(x, nd=4):
    return f'{x:.{nd}g}' if isinstance(x, (int, float)) and not isinstance(x, bool) else ('-' if x is None else str(x))


def check_run(run_dir: Path, *, budget_screens: int | None = None) -> dict:
    run_dir = Path(run_dir)
    r = json.loads((run_dir / 'result.json').read_text(encoding='utf-8'))
    text = (run_dir / 'REPORT.md').read_text(encoding='utf-8')
    problems = []
    outcome = r['outcome']
    promo = r.get('promotion') or {}
    # --- the outcome and its section
    section = {'improved': '## Recommendation', 'no_change': '## Recommendation', 'reference_failed': '## The reference run failed',
               'not_verifiable': '## Nothing could be verified', 'interrupted': '## Interrupted',
               'infrastructure_failure': 'could not be reached'}[outcome]
    if section not in text:
        problems.append(f'outcome {outcome}: section {section!r} missing from REPORT.md')
    if promo.get('promoted'):                          # improved, or interrupted after a passed promotion (U-7-3)
        cand = promo['promoted']
        if f'Use candidate **{cand}**' not in text:
            problems.append('the recommended candidate is not the promoted one')
        if f'fresh re-run x{_fmt(promo["speedup"], 4)}' not in text:
            problems.append(f'the report\'s re-run speedup is not promotion.speedup {promo["speedup"]}')
        ref_cfg, cfg = r['reference']['config'], promo['config']
        for k, v in cfg.items():
            if ref_cfg.get(k) != v and f'| {k} | {ref_cfg.get(k)} | {v} |' not in text:
                problems.append(f'knob {k} {ref_cfg.get(k)} -> {v} missing from the recommendation table')
        tried = {t['candidate'] for t in promo.get('tried') or []}
        if cand not in tried or not any(t['passed'] for t in promo['tried'] if t['candidate'] == cand):
            problems.append('the promoted candidate did not pass its promotion items')
        # the promoted candidate is the fastest valid candidate that passed; nothing faster passed
        passed = [t for t in promo['tried'] if t['passed']]
        if passed and passed[0]['candidate'] != cand:
            problems.append('a candidate passed promotion before the promoted one')
    if outcome == 'no_change' and promo.get('promoted'):
        problems.append('no_change with a promoted candidate')
    # an interrupt during the promotion, the write-back or the report after it (U-7-3) leaves the search complete
    if outcome in ('improved', 'no_change') or r.get('interrupted_during') in ('promotion', 'write-back', 'report'):
        # --- candidates: the table lists each once; receipts on disk verify; valid ones have a step time
        cands = r.get('candidates') or []
        ids = [c['id'] for c in cands]
        if len(ids) != len(set(ids)):
            problems.append('duplicate candidate ids')
        rows = re.findall(r'^\| (c\d+) \| ', text, re.M)
        if sorted(set(rows)) != sorted(ids):
            problems.append(f'candidate table rows {sorted(set(rows))} != candidates {sorted(ids)}')
        receipts = {p.stem: json.loads(p.read_text(encoding='utf-8')) for p in (run_dir / 'search' / 'receipts').glob('*.json')}
        for c in cands:
            if c.get('status') in ('screened', 'aborted') and c['id'] not in receipts:
                problems.append(f'candidate {c["id"]} ran but has no receipt on disk')
            if c.get('valid') and not c.get('step_time'):
                problems.append(f'valid candidate {c["id"]} without a step time')
        for cid, rec in receipts.items():
            if not signing.verify(rec):
                problems.append(f'receipt {cid} does not verify')
            if rec.get('candidate_id') != cid:
                problems.append(f'receipt {cid} is for candidate {rec.get("candidate_id")}')
        # --- the decisions follow from the receipts: a fresh judgement of every full screen receipt against the stored
        # reference receipt must agree with the table (validity and step time); the promoted candidate's fresh pair must
        # give the reported speed-up
        vd = r.get('verifier') or {}
        judge = Verifier(r['tolerance'], required=tuple(vd.get('required') or ()), disabled=vd.get('disabled') or (),
                         memory_cap_gib=vd.get('memory_cap_gib'))
        ref_rec = receipts.get('c0')
        if ref_rec is None:
            problems.append('no reference receipt (search/receipts/c0.json)')
        else:
            by_id = {c['id']: c for c in cands}
            for cid, rec in receipts.items():
                c = by_id.get(cid)
                ev = rec.get('evidence') or {}
                full = type(ev.get('steps_completed')) is int and ev.get('steps_completed') == ev.get('steps') and not ev.get('stop_after')
                if cid == 'c0' or c is None or c.get('status') != 'screened' or not full:
                    continue
                v = judge(rec, None, ref_rec)
                if bool(c.get('valid')) != v.valid:
                    problems.append(f'{cid}: the table says valid={c.get("valid")}, a fresh judgement of its receipt says {v.valid}: {v.reasons[:2]}')
                if c.get('step_time') and v.step_time and not math.isclose(c['step_time'], v.step_time, rel_tol=1e-9):
                    problems.append(f'{cid}: table step time {c["step_time"]} != receipt {v.step_time}')
            # U-1-21: a row's deviation from the reference is what its loss / gradient-norm check measured on its
            # full-length screen (a fresh judgement of the receipt), none without one (c0 included); the report's
            # candidate table shows the loss one as "loss vs ref"
            table = text.split('## Search', 1)[-1]
            for c in cands:
                ev = (receipts.get(c['id']) or {}).get('evidence') or {}
                full = type(ev.get('steps')) is int and ev.get('steps_completed') == ev['steps']
                checks = {x['id']: x for x in judge(receipts[c['id']], None, ref_rec).checks} if full else {}
                for key, check in (('loss_deviation', 'loss_within_tolerance'),
                                   ('grad_norm_deviation', 'grad_norm_within_tolerance')):
                    want = (checks.get(check, {}).get('evidence') or {}).get('signed_deviation')
                    if key not in c or c[key] != want:
                        problems.append(f'{c["id"]}: {key} {c.get(key, "missing")} != {want} (a fresh judgement of its receipt)')
                cell = '-' if c.get('loss_deviation') is None else f'{100 * c["loss_deviation"]:+.2f} %'
                row = re.search(rf'^\| {c["id"]} \| .*$', table, re.M)
                if row is None or row.group(0).split(' | ')[4] != cell:
                    problems.append(f'{c["id"]}: the candidate table does not show loss vs ref {cell}')
            if promo.get('promoted'):
                tried = next((t for t in promo.get('tried') or [] if t['candidate'] == promo['promoted']), None)
                ids = (tried or {}).get('rescore') or []
                pair = {}
                for f in (run_dir / 'attempts').glob(f'*/*-rescore-*/receipt.json'):
                    rec = json.loads(f.read_text(encoding='utf-8'))
                    if rec.get('attempt_id') in ids:
                        pair[rec['candidate_id']] = rec
                if set(pair) != {'c0', promo['promoted']}:
                    problems.append(f'the promoted candidate\'s fresh pair of receipts is not on disk: found {sorted(pair)}')
                else:
                    a, b = judge(pair['c0'], None, None), judge(pair[promo['promoted']], None, pair['c0'])
                    if not (a.valid and b.valid and signing.verify(pair['c0']) and signing.verify(pair[promo['promoted']])):
                        problems.append(f'the fresh pair does not re-judge as valid: {a.reasons[:1]} {b.reasons[:1]}')
                    elif not math.isclose(a.step_time / b.step_time, promo['speedup'], rel_tol=1e-6):
                        problems.append(f'promotion.speedup {promo["speedup"]} != the pair\'s {a.step_time / b.step_time}')
                    if pair[promo['promoted']].get('config') != promo.get('config'):
                        problems.append('the promoted configuration is not the one the fresh run used')
                    if (pair[promo['promoted']].get('overlay') or {}) != (promo.get('overlay') or {}):
                        problems.append('the promoted patch is not the one the fresh run used')
        digests = [c.get('config_digest') for c in cands if c.get('status') == 'screened' and not c.get('overlay_files')]
        # (two screened candidates with one configuration happen only as a deliberate re-run; flag as information)
        if len(digests) != len(set(digests)):
            problems.append('two screened candidates share a configuration digest')
        # --- budget
        used = (r.get('search') or {}).get('used') or {}
        b = (r.get('search') or {}).get('budget') or {}
        if budget_screens is not None and (used.get('screens') or 0) > budget_screens:
            problems.append(f'used {used.get("screens")} screens of a budget of {budget_screens}')
        if b.get('screens') and (used.get('screens') or 0) > b['screens']:
            problems.append(f'used {used.get("screens")} screens of a budget of {b["screens"]}')
        if b.get('gpu_hours') and (used.get('gpu_hours') or 0) > b['gpu_hours'] * 1.05:
            problems.append(f'used {used.get("gpu_hours")} GPU-hours of a budget of {b["gpu_hours"]}')
        full = [s for s in r.get('screens') or [] if s.get('fidelity', 'full') == 'full']
        if used.get('screens') is not None and len(full) != used['screens']:
            problems.append(f'{len(full)} full screens recorded, used.screens says {used["screens"]}')
        # --- the search section's numbers
        if f'Strategy `{r.get("strategy")}`: {used.get("screens")} full runs' not in text:
            problems.append('the search line does not state the strategy and the full runs')
        cost = r.get('cost') or {}
        if f'GPU-hours: reference {_fmt(cost.get("reference_gpu_hours"), 3)}' not in text:
            problems.append('the cost line does not match result.cost')
        # --- the run log exists and names the milestones
        log = (run_dir / 'run.log').read_text(encoding='utf-8') if (run_dir / 'run.log').is_file() else ''
        for m in ('reference run of the current configuration', 'search: strategy'):
            if m not in log:
                problems.append(f'run.log lacks the milestone {m!r}')
    assert not problems, '\n'.join(problems)
    return r
