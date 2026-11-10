"""`infrarsi finish [RUN_DIR]`: the promotion of a run that ended before its own, from what the run left on disk.

A run stopped with Ctrl-C re-verifies its best candidates by itself (`search.on_interrupt: verify`, project/flow.py). This is
the same step for a run that could not: its process was killed or lost its machine or connection, a second Ctrl-C stopped
the re-verification, or `search.on_interrupt: stop` was set. It reads the candidates' receipts in `search/receipts/`, so it
works for every strategy and needs no search-state checkpoint: the fastest valid candidates get the fresh back-to-back re-runs
and the other promotion items of a normal run, and result.json and REPORT.md of the run directory are written again.

The experience write-back is not made: the strategy's ledger went with the process. The search is not continued
(`infrarsi run --resume` does that).
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from ..actions import config_diff
from ..agent import state as S
from ..agent.debate import logged, summarize
from ..verify import evidence
from . import report as report_mod
from .flow import (BASE_FILES, RUN_META, _brief, doctor_run, gpu_contention, make_verifier, promote, remember_noise,
                   repeat_admissible, run_log)
from .manifest import Project
from .runner import ProjectRunner

PROCESS_ENDED = 'process_ended'                        # the stop reason of a search whose process ended without a report
FINISHABLE = (None, 'interrupted', 'infrastructure_failure')   # a promotion that ran to its end stands: no second try at it


class FinishError(RuntimeError):
    """A run that cannot be finished (nothing measured, already decided, the project changed since)."""


def _load(path: Path) -> dict | None:
    try:
        doc = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _events(run_dir: Path) -> list[dict]:
    rows = []
    try:
        text = (run_dir / 'search' / 'events.jsonl').read_text(encoding='utf-8', errors='replace')
    except OSError:
        return rows
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue                                   # (a line the killed process did not finish)
    return rows


def _measured(receipts: Path, verifier, ref: dict) -> list:
    """A candidate row per receipt, judged as the search judged it. An aborted run is not valid here: its step time covers the
    updates it reached."""
    rows = []
    ref_cfg = ref.get('config') or {}
    for f in sorted(receipts.glob('c*.json'), key=lambda p: (len(p.stem), p.stem)):
        r = _load(f)
        if r is None:
            continue
        v = verifier(r, None, ref)
        ev = r.get('evidence') if isinstance(r.get('evidence'), dict) else {}
        aborted = bool(ev.get('stopped_early'))
        cfg = r.get('config') or {}
        valid = v.valid and not aborted
        rows.append({'id': f.stem, 'parent': None if f.stem == 'c0' else 'c0', 'status': 'aborted' if aborted else 'screened',
                     'change': config_diff(ref_cfg, cfg), 'overlay_files': sorted(r.get('overlay') or {}), 'valid': valid,
                     'step_time': v.step_time, 'reasons': v.reasons[:3], 'attempt_id': r.get('attempt_id')})
    return rows


def _merged(known: list, measured: list, stale: bool) -> list:
    """The saved candidate rows with what the receipts add: a candidate run after the result was written, a row that had no
    verdict yet; when the result is stale (the search went on after it), the receipts decide."""
    rows = list(known)
    by_id = {c.get('id'): c for c in rows}
    for m in measured:
        if m['id'] not in by_id:
            rows.append(m)
        elif stale or by_id[m['id']].get('valid') is None:
            by_id[m['id']].update(m)
    return rows


def _caught_up(result: dict, fresh: dict) -> dict:
    """A result older than the receipts with the search's accounting from the events (the run went on after it)."""
    cost = {**(result.get('cost') or {}), **{k: v for k, v in (fresh.get('cost') or {}).items()
                                             if k in ('search_gpu_hours', 'llm_tokens', 'llm_usd') and v is not None}}
    return {**result, 'screens': fresh.get('screens') or result.get('screens'), 'cost': cost,
            'search': {**(result.get('search') or {}), 'used': (fresh.get('search') or {}).get('used'), 'stop_reason': PROCESS_ENDED,
                       'llm_calls': (fresh.get('search') or {}).get('llm_calls')}}


def _rebuilt(project: Project, out: Path, verifier, ref: dict, vref, rows: list) -> dict:
    """What a run's result holds at this point, for a run whose process ended before it wrote one."""
    events = _events(out)
    runs = [e['payload'] for e in events if e.get('kind') in ('screen', 'trial', 'rescore') and isinstance(e.get('payload'), dict)]
    used = next((r['cum'] for r in reversed(runs) if isinstance(r.get('cum'), dict)), {})
    calls = [e['payload'] for e in events if e.get('kind') == 'llm_call' and isinstance(e.get('payload'), dict)]
    if calls:                                          # (the model calls after the last run are in no run's running total)
        used = {**used, 'llm_tokens': sum(c.get('llm_tokens') or 0 for c in calls)}
        if any(c.get('llm_usd') is not None for c in calls):
            used['llm_usd'] = sum(c.get('llm_usd') or 0.0 for c in calls)
    try:
        strategy = S.read(out / 'search' / S.STATE_DIR)[0]['header'].get('strategy')
    except S.CheckpointError:
        strategy = None
    brief = _brief(ref, vref)
    return {'schema': 'infrarsi-run-v1', 'run_id': out.name, 'run_dir': str(out), 'project': project.describe(),
            'manifest': str(project.path) if project.path else None, 'tolerance': verifier.tolerance,
            'verifier': verifier.describe(), 'reference': {'config': ref.get('config'), **brief}, 'strategy': strategy,
            'search': {'stop_reason': PROCESS_ENDED, 'used': used, 'llm_calls': len(calls)},
            'screens': runs, 'candidates': rows,
            'cost': {'reference_gpu_hours': brief['gpu_hours'], 'search_gpu_hours': used.get('gpu_hours'),
                     'promotion_gpu_hours': 0.0, 'profile_gpu_hours': 0.0, 'llm_tokens': used.get('llm_tokens'),
                     'llm_usd': used.get('llm_usd')}}


def _meta(out: Path) -> dict:
    return _load(out / RUN_META) or {}


def _stale(out: Path, receipts: Path) -> bool:
    """A result.json older than a receipt: the search went on after it (resumed) and ended without a new one."""
    try:
        written = (out / 'result.json').stat().st_mtime
    except OSError:
        return False
    return any(f.stat().st_mtime > written for f in receipts.glob('c*.json'))


def project_of(run_dir) -> Project:
    """The project of a run directory: the manifest the run recorded, else the one of <root>/.infrarsi/runs/<run>."""
    out = Path(run_dir).resolve()
    name = _meta(out).get('manifest') or (_load(out / 'result.json') or {}).get('manifest')
    return Project.load(name or out.parents[2] / 'infrarsi.yaml')


def finish_run(run_dir, *, project: Project | None = None, log=print) -> dict:
    """The promotion of the run of `run_dir` from its receipts; result.json and REPORT.md of the directory are written again
    with the outcome. Raises FinishError for a run with nothing to verify or one already decided."""
    out = Path(run_dir).resolve()
    receipts = out / 'search' / 'receipts'
    ref = _load(receipts / 'c0.json')
    if ref is None:
        raise FinishError(f'{out}: no reference run was recorded (search/receipts/c0.json), so there is nothing to verify')
    prev = _load(out / 'result.json') or {}
    stale = bool(prev) and _stale(out, receipts)
    if prev and not stale:
        if (prev.get('promotion') or {}).get('promoted'):
            raise FinishError(f'{out}: {prev["promotion"]["promoted"]} is already verified: `infrarsi apply {out}` shows the change')
        if prev.get('outcome') not in FINISHABLE:
            raise FinishError(f'{out} ended as {prev.get("outcome")}: its re-verification ran to its end, so there is nothing to '
                              'finish; run again to search again')
    project = project or project_of(out)
    saved = _load(out / 'requirements.json') or prev.get('requirements')
    if saved:                                          # the run's own requirements hold here too (its checks, its frozen files)
        from .requirements import RequirementsError, apply
        try:
            project = apply(project, saved)
        except RequirementsError as exc:
            raise FinishError(f'{out}: {exc}') from None
    tolerance = _meta(out).get('tolerance', prev.get('tolerance'))   # (the tolerance the run was given: --tolerance too)
    verifier = make_verifier(project, tolerance)
    if json.dumps(ref.get('config'), sort_keys=True, default=str) != json.dumps(project.reference_config(), sort_keys=True, default=str):
        raise FinishError(f'{project.path}: the current configuration is not the one this run measured against; restore it, or '
                          'run again')
    vref = verifier(ref, None, ref)
    if not vref.valid:
        raise FinishError(f'{out}: the reference run did not pass the run checks: ' + '; '.join(vref.reasons[:3]))
    log = run_log(out / 'run.log', log)
    doc = doctor_run(project, ref)                     # (the loss check reads from the repeat whether the draws differ)
    if doc and repeat_admissible(verifier(doc, None, ref)):
        verifier.repeat = evidence(doc)
    measured = _measured(receipts, verifier, ref)
    rows = _merged(prev.get('candidates') or [], measured, stale)
    # what the normal promotion reads: the result's noise while that result is current, else the search's own as it stood after
    # its last run (search/noise.json, written by the Engine); a strategy outside the Engine has none, as in its normal promotion
    current = None if stale else (prev.get('search') or {}).get('noise')
    noise = current or _load(out / 'search' / 'noise.json') or (prev.get('search') or {}).get('noise') or {}
    trace = {'candidates': rows, 'extra': {'noise': noise}}
    fresh = _rebuilt(project, out, verifier, ref, vref, rows)
    result = _caught_up(dict(prev), fresh) if stale else (dict(prev) if prev else fresh)
    result['search'] = {**(result.get('search') or {}), 'noise': noise}   # (what a later `finish` of this run reads)
    if saved:
        result.setdefault('requirements', saved)
    if (out / 'debates.jsonl').is_file():              # (its log is the whole account: a killed run wrote no result to carry it)
        result['debate'] = summarize(logged(out / 'debates.jsonl'))
    earlier_gpu = (result.get('cost') or {}).get('promotion_gpu_hours') or 0.0
    # a report on disk before the re-runs (as the run itself writes one): a hard kill during them leaves it
    result.update(outcome='interrupted', interrupted_during='promotion', candidates=rows,
                  promotion={'promoted': None, 'tried': [], 'gpu_hours': 0.0, 'ref_offsets': [], 'interrupted': True,
                             'why': '`infrarsi finish` did not complete its re-verification'})
    report_mod.write(result, out)
    runner = ProjectRunner(project, out_dir=out / 'attempts', log=log, skip=out.parent)
    faster = any(c['id'] != 'c0' and c.get('valid') and c.get('step_time') and c['step_time'] < vref.step_time for c in rows)
    log(f'[infrarsi] finish {prev.get("run_id") or out.name}: ' + (
        'fresh re-runs of the fastest valid candidates against the reference' if faster
        else 'no valid candidate is faster than the reference, nothing to re-run'))
    promo = promote(project, project.task(), runner, verifier, trace, ref, prev.get('run_id') or out.name, receipts,
                    max_candidates=int(project.raw['search'].get('promotion_candidates') or 3))
    if promo.get('promoted'):                          # the files the patches were written against: apply merges later edits
        base = _load(out / 'search' / BASE_FILES) or {}
        promo['base'] = {rel: base.get(rel) for rel in promo.get('overlay') or {}}
    if not promo.get('interrupted'):
        remember_noise(project, promo.get('ref_offsets') or [])
    outcome = ('interrupted' if promo.get('interrupted') else 'improved' if promo.get('promoted') else
               'infrastructure_failure' if promo.get('infrastructure') else 'no_change')
    result.update(outcome=outcome, promotion=promo, gpu_contention=gpu_contention(runner),
                  finished={'utc': _dt.datetime.now(_dt.timezone.utc).isoformat(), 'from': 'receipts',
                            'experience_written': False})
    if not promo.get('interrupted'):
        result.pop('interrupted_during', None)
    if promo.get('infrastructure'):
        result['infrastructure'] = promo['infrastructure']
    result['cost'] = {**(result.get('cost') or {}), 'promotion_gpu_hours': earlier_gpu + promo.get('gpu_hours', 0.0)}
    log(f'[infrarsi] {outcome}' + (f', x{promo["speedup"]:.3f} on the fresh re-run' if promo.get('speedup') else '')
        + f'; report: {out / "REPORT.md"}')
    report_mod.write(result, out)
    return result
