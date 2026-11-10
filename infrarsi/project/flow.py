"""`infrarsi run`: the one flow (revamp design B) - reference run, search, promotion, report.

    run_project(project, strategy=None, driver=None) -> result dict   (also written to <run dir>/result.json, REPORT.md)

1. Reference: the user's current configuration runs once; it must pass the run-health items, else nothing is searched.
2. Search: the chosen strategy (infrarsi.search) proposes and screens candidates; the Verifier judges each run
   against the reference and its checklist feedback goes back to the strategy.
3. Promotion: down the ranking of valid candidates faster than the reference, each gets a fresh back-to-back re-run
   with the reference, the checkpoint resume check (when configured) and the project's own tests; the first that
   passes is the recommendation. No candidate passing means "keep your configuration".
The reference and promotion runs are reported with the search cost but are outside the search budget.
"""
from __future__ import annotations

import contextlib
import copy
import datetime as _dt
import json
import math
import os
import signal
import threading
from pathlib import Path

from .. import search
from ..agent.search import USER_STOP
from ..contracts import InfrastructureError
from ..search import checkpoint as ckpt
from ..search.engine import loss_deviation, pooled_sigma_pair, white_floor
from ..verify import DEFAULT_REQUIRED, Verifier, blocking, coverage, evidence
from ..verify.checklist import UNVERIFIABLE_WORKLOAD
from ..verify.verifier import loss_repeat_note
from . import report as report_mod
from . import signals as sig
from .hooks import ProjectHooks
from .llm import UNPRICED
from .manifest import Project
from .runner import PROFILE_UPDATES, PROFILE_WARMUP, ProjectRunner


def home() -> Path:
    return Path(os.environ.get('INFRARSI_HOME') or Path.home() / '.infrarsi')


def user_package():
    """The user's optimizer package (skills, prompts, policy, cross-run memory), built from A0 on first use."""
    from ..rsi.package import OptimizerPackage, build_a0
    path = home() / 'package'
    if (path / 'manifest.json').exists():
        return OptimizerPackage.load(path)
    return build_a0(path)


def ledger_path() -> Path:
    return home() / 'ledger.jsonl'


def prior_results(project: Project, limit: int = 30) -> list[dict]:
    """Measured outcomes of earlier runs (any project) on the same GPU type, newest first: the cross-run memory the
    LLM sees as priors."""
    p = ledger_path()
    if not p.is_file():
        return []
    gpu = project.raw['hardware'].get('gpu_type')
    rows = []
    for line in p.read_text(encoding='utf-8').splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get('status') in ('confirmed', 'bundled', 'refuted', 'unresolved', 'invalid') and (gpu is None or r.get('gpu_type') in (gpu, None)):
            rows.append({k: r.get(k) for k in ('project', 'framework', 'gpus', 'mechanism', 'change', 'predicted_gain_pct',
                                                'measured_gain_pct', 'status', 'failed_checks', 'promotion')})
    return rows[::-1][:limit]


def run_noise(project: Project) -> float | None:
    """A fixed between-run noise floor from the manifest (search.run_noise), else None: the engine measures the noise
    (pairs of runs of one configuration and the split halves of every run, `noise_pairs`, Engine.sigma_pair)."""
    set_ = project.raw['search'].get('run_noise')
    return float(set_) if set_ is not None else None


def noise_pairs(project: Project) -> list[float]:
    """Log ratios of two runs of one configuration recorded by earlier runs of this project on this GPU type
    (reference re-runs at promotion, confirmation re-runs, doctor vs reference)."""
    xs = []
    p = ledger_path()
    if p.is_file():
        for line in p.read_text(encoding='utf-8').splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('kind') == 'noise' and r.get('project') == project.name and                     r.get('gpu_type') == project.raw['hardware'].get('gpu_type') and isinstance(r.get('offset'), (int, float)):
                xs.append(float(r['offset']))
    return xs


def doctor_run(project: Project, ref: dict) -> dict | None:
    """The receipt of `infrarsi doctor`'s run of this run's reference configuration on the same machine in the last
    day; None without one."""
    runs = sorted((project.root / '.infrarsi' / 'doctor' / 'doctor').glob('c0-doctor-*/receipt.json'),
                  key=lambda f: f.stat().st_mtime)
    if not runs or _dt.datetime.now().timestamp() - runs[-1].stat().st_mtime > 86400:
        return None
    try:
        doc = json.loads(runs[-1].read_text(encoding='utf-8'))
    except ValueError:
        return None
    if any(doc.get(k) != ref.get(k) for k in ('config_digest', 'overlay_digest', 'host', 'executor')):
        return None                                   # another configuration, or another machine
    return doc


def repeat_admissible(v) -> bool:
    """The doctor's run is the reference's repeat when nothing but the loss sets it apart: every blocking check other
    than loss_within_tolerance passes (a run of another workload or model under the same knobs - DF16_r1: half the
    tokens per update - must not widen a seeded job's tolerance); an unsigned or foreign receipt is no repeat.
    (Collective lines at the end of a complete run - D2-R204e's doctor run - do not fail no_errors: DF18.)"""
    return bool(v.checks) and {r.split(':', 1)[0] for r in v.reasons} <= {'loss_within_tolerance'}


def repeat_valid(v) -> bool:
    """The doctor run as the reference's repeat for the speed pair and the repeat floor: valid as judged with the
    repeat installed (DF17_r1: an admitted repeat must not leave the noise pool and the floor empty)."""
    return bool(v.checks) and v.valid


def doctor_floor(verifier: Verifier, doc: dict | None, ref: dict) -> float | None:
    """The repeat floor from the doctor run: its loss deviation from the reference, judged at the tolerance the repeat
    set; None without a doctor run or when it is no repeat."""
    if not doc:
        return None
    v = verifier(doc, None, ref)
    return loss_deviation(v.to_dict()) if repeat_valid(v) else None


def doctor_pair(project: Project, verifier: Verifier, ref: dict, t_ref: float | None) -> float | None:
    """log(doctor run / this run's reference) when `infrarsi doctor` ran the same configuration in the last day:
    one more pair of runs of one configuration for the noise estimate, at no cost."""
    doc = doctor_run(project, ref) if t_ref else None
    if doc is None:
        return None
    v = verifier(doc, None, ref)
    return math.log(v.step_time / t_ref) if repeat_valid(v) and v.step_time else None


def repeat_deviation(v) -> float | None:
    """The loss deviation in the verdict of a re-run of the reference configuration against the reference: the floor a
    change's own deviation is read against (non-bitwise kernels, reduction order); None when it was not checked."""
    return loss_deviation(v.to_dict()) if v is not None and v.valid else None


def own_repeat(verifier, v, fresh: dict, screen: dict) -> float | None:
    """The promoted candidate's fresh run against its own screen run, as the loss check measures it: the run-to-run
    spread of the incumbent configuration (D1-10: compiled runs differ by 1e-6 where the reference repeats bit for
    bit); None when the fresh run is not valid."""
    if not (v is not None and v.valid and hasattr(verifier, 'screen_checks')):
        return None
    try:
        return loss_deviation({'checks': verifier.screen_checks(fresh.get('evidence') or {}, screen.get('evidence') or {})})
    except Exception:                                   # noqa: BLE001 - a diagnostic never fails the promotion
        return None


def _write_whole(fd: int, data: bytes) -> None:
    # every byte before any interrupt (U-7-3): os.write may write fewer bytes than asked, so the rest goes in further
    # calls; a Ctrl-C / SIGTERM meanwhile is held (signal handlers are main-thread only) and a KeyboardInterrupt raised
    # by a call (which then wrote nothing) kept, each delivered once the last byte is out - no row is left half written
    view, held, cut, old = memoryview(data), [], False, {}
    try:
        for s in (signal.SIGINT, signal.SIGTERM) if threading.current_thread() is threading.main_thread() else ():
            old[s] = signal.signal(s, lambda n, frame: held.append(n))
        while view:
            try:
                view = view[os.write(fd, view):]
            except KeyboardInterrupt:
                cut = True
    finally:
        for s, handler in old.items():
            signal.signal(s, handler)
    if cut:
        raise KeyboardInterrupt
    for n in held[:1]:                                  # through the handler the process had
        signal.raise_signal(n)


def append_ledger(text: str) -> None:
    """Append rows, each ending in a newline, to the ledger; a fragment an older version or a crash left at its end is
    closed first, so that the new rows start on their own line (and are read)."""
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode('utf-8')
    if path.is_file() and path.stat().st_size:
        with path.open('rb') as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b'\n':
                data = b'\n' + data
    # an unbuffered descriptor, not a file object, so that _write_whole sees every write; on Windows in text mode,
    # as the old open('a') was: rows still end in '\r\n'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    try:
        _write_whole(fd, data)
    finally:
        os.close(fd)


def remember_noise(project: Project, offsets) -> int:
    """Append this run's pairs of runs of one configuration, |log ratio| (promotion re-runs of the reference,
    confirmation re-runs, doctor vs reference), to the ledger."""
    xs = [x for x in offsets or [] if isinstance(x, (int, float))]
    if not xs:
        return 0
    stamp, gpu = _dt.datetime.now(_dt.timezone.utc).isoformat(), project.raw['hardware'].get('gpu_type')
    append_ledger(''.join(json.dumps({'utc': stamp, 'kind': 'noise', 'project': project.name, 'gpu_type': gpu,
                                      'offset': round(x, 5)}) + '\n' for x in xs))
    return len(xs)


def remember(project: Project, trace: dict, promo: dict | None = None, skip=()) -> int:
    """Append the resolved hypotheses of this run to the cross-run ledger, each with the promotion outcome of its
    candidate (for a bundled one: the composite's) when that was tried, and 'not tried' for a confirmed or bundled one
    that promotion did not reach (a change that did not reproduce, or was never checked, must not read as reliable);
    returns how many rows were written. `skip`: ids of the rows an earlier completion of this run wrote (a resumed run)."""
    rows = [h for h in ((trace.get('extra') or {}).get('ledger') or []) if h.get('status') not in ('open', 'superseded')
            and h.get('id') not in set(skip)]
    if not rows:
        return 0
    outcome = {}
    for t in (promo or {}).get('tried') or []:
        failed = [i['id'] for i in t.get('items') or [] if i.get('status') == 'fail']
        outcome[t['candidate']] = 'passed' if t.get('passed') else 'failed: ' + ', '.join(failed)
    hw = project.raw['hardware']
    stamp = _dt.datetime.now(_dt.timezone.utc).isoformat()
    lines = []
    for h in rows:
        cand = h.get('bundle_candidate') or h.get('candidate')    # a bundled change: the composite that moved
        extra = ({'promotion': outcome[cand]} if cand in outcome else
                 {'promotion': 'not tried'} if promo is not None and h.get('status') in ('confirmed', 'bundled') else {})
        lines.append(json.dumps({'utc': stamp, 'project': project.name, 'framework': project.raw.get('framework'),
                                 'gpu_type': hw.get('gpu_type'), 'gpus': project.gpus, **h, **extra}, default=str) + '\n')
    append_ledger(''.join(lines))
    return len(rows)                                    # counted once the write returned (run_project's `remembered`)


def make_verifier(project: Project, tolerance: float | None = None) -> Verifier:
    v = project.raw['verify']
    return Verifier(project.tolerance if tolerance is None else tolerance, required=v.get('required') or DEFAULT_REQUIRED,
                    disabled=v.get('disabled') or (), memory_cap_gib=project.raw['hardware'].get('memory_gib'))


def _brief(receipt: dict, verdict) -> dict:
    ev = receipt.get('evidence') or {}
    return {'attempt_id': receipt.get('attempt_id'), 'valid': verdict.valid, 'step_time': verdict.step_time,
            'reasons': verdict.reasons[:6], 'coverage': verdict.coverage, 'checks': verdict.checks,
            'step_time_source': ev.get('step_time_source'), 'rsd_window': ev.get('rsd_window'), 'log': receipt.get('log'),
            'phases': ev.get('phases'), 'step_log': receipt.get('step_log'), 'log_tail': (receipt.get('log_tail') or [])[-40:],
            'gpu_hours': (receipt.get('cost') or {}).get('gpu_seconds', 0) / 3600}


def promote(project: Project, task, runner: ProjectRunner, verifier: Verifier, trace: dict, reference: dict, run_id: str,
            receipts_dir: Path, *, max_candidates: int = 3) -> dict:
    gpu_s = [0.0]
    tried, offsets = [], []
    at = {'runs': 0}                                   # for an interrupt: the candidate and check under way, runs made
    try:
        return _promote(project, task, runner, verifier, trace, reference, run_id, receipts_dir, max_candidates,
                        tried, offsets, gpu_s, at)
    except InfrastructureError as exc:                 # the hosts went away: what was tried stays, nothing is promoted
        return {'promoted': None, 'tried': tried, 'gpu_hours': gpu_s[0] / 3600, 'ref_offsets': offsets,
                'why': f'promotion stopped: the hosts could not be reached ({str(exc)[:200]})', 'infrastructure': str(exc)}
    except KeyboardInterrupt:                          # Ctrl-C / SIGTERM (U-7-3): likewise, and the run ends as interrupted
        where = f' of {at["candidate"]} ({at["purpose"]})' if 'candidate' in at else ''
        return {'promoted': None, 'tried': tried, 'gpu_hours': gpu_s[0] / 3600, 'ref_offsets': offsets, 'interrupted': True,
                'why': f'interrupted during the promotion{where}, after {at["runs"]} re-run(s)'}


def _promote(project, task, runner, verifier, trace, reference, run_id, receipts_dir, max_candidates, tried, offsets,
             gpu_acc, at) -> dict:
    def run(*a, **kw):                                  # each run is charged when it returns (a later one may fail)
        r = runner.run(*a, **kw)
        gpu_acc[0] += (r.get('cost') or {}).get('gpu_seconds', 0)
        at['runs'] += 1
        return r

    t_ref = verifier(reference, None, reference).step_time
    ranked = sorted((c for c in trace['candidates'] if c['id'] != 'c0' and c.get('valid') and c.get('step_time')
                     and c['step_time'] < t_ref), key=lambda c: c['step_time'])
    ref_saved = []
    noise = (trace.get('extra') or {}).get('noise') or {}           # the search's noise estimate (Engine.noise)
    pairs = list(noise.get('pairs') or [])
    # both forms of the split halves: the robust ones (medians) count only once the pool has a pair of runs of one
    # configuration, which the reference re-run below adds when it is valid (DF4_r2); older traces have the plain ones
    plain = list(noise.get('split_halves_plain', noise.get('split_halves')) or [])
    robust = list(noise.get('split_halves_robust') or plain)
    ref_cfg = project.reference_config()
    ck = project.raw.get('checkpoint')
    for c in ranked[:max_candidates]:
        at.update(candidate=c['id'], purpose='fresh re-run')
        cand = json.loads((receipts_dir / f'{c["id"]}.json').read_text(encoding='utf-8'))
        cfg, overlay = cand.get('config') or {}, cand.get('overlay') or {}
        runs = [run(task, ref_cfg, None, 'rescore', run_id, candidate_id='c0'),
                run(task, cfg, overlay, 'rescore', run_id, candidate_id=c['id'])]
        resume, evals = None, None
        if ck:
            at['purpose'] = 'resume'
            save_at = project.save_at
            shared = runner.shared_dir(run_id, f'ckpt-{c["id"]}')
            runs.append(run(task, cfg, overlay, 'save', run_id, candidate_id=c['id'], steps=save_at,
                            extra_args=ck['save'], ckpt_dir=shared))
            resume = run(task, cfg, overlay, 'resume', run_id, candidate_id=c['id'], extra_args=ck['resume'], ckpt_dir=shared)
            runs.append(resume)
            if project.raw['verify'].get('eval'):
                at['purpose'] = 'eval'
                ref_ckpt = runner.shared_dir(run_id, 'ckpt-c0')
                if not ref_saved:
                    runs.append(run(task, ref_cfg, None, 'save', run_id, candidate_id='c0', steps=save_at,
                                    extra_args=ck['save'], ckpt_dir=ref_ckpt))
                    ref_saved.append(ref_ckpt)
                evals = (runner.run_eval(ref_cfg, None, run_id, 'c0', ref_ckpt),
                         runner.run_eval(cfg, overlay, run_id, c['id'], shared))
        rerun, fresh = verifier(runs[0], None, reference), verifier(runs[1], None, reference)
        if rerun.valid and rerun.step_time and t_ref:
            pairs.append(math.log(rerun.step_time / t_ref))          # one more pair of runs of the reference
            offsets.append(abs(pairs[-1]))
        for r, v in ((runs[0], rerun), (runs[1], fresh)):
            ev = r.get('evidence') or {}
            h, hr = ev.get('split_half'), ev.get('split_half_robust')
            if v.valid and type(h) in (int, float):                   # and the two fresh runs' split halves
                plain.append(h)
                robust.append(hr if type(hr) in (int, float) else h)
        halves = robust if pairs else plain
        margin = None                                                 # 2 sigma_pair, this promotion's runs included
        if noise.get('mode') == 'fixed' and noise.get('sigma_pair'):
            margin = math.expm1(2 * noise['sigma_pair'])
        elif noise.get('mode') == 'measured':
            sp = pooled_sigma_pair(pairs, halves)
            if sp is not None:
                margin = math.expm1(2 * max(sp, white_floor(runs[0].get('evidence'), robust=bool(pairs))))
        at['purpose'] = 'tests'
        tests = runner.run_tests(cfg, overlay, run_id, c['id'])
        items = verifier.promotion(runs[1], reference, rescore=(runs[0], runs[1]), resume=resume, tests=tests, evals=evals,
                                   save_at=project.save_at, margin=margin)
        m = next((i['evidence'].get('margin') for i in items if i['id'] == 'speed_reproduces' and i.get('evidence')), None)
        d = math.log(fresh.step_time / c['step_time']) if fresh.valid and fresh.step_time and c.get('step_time') else None
        row = {'candidate': c['id'], 'step_time': c['step_time'], 'items': items, 'passed': not blocking(items, verifier.required),
               'rescore': [r['attempt_id'] for r in runs[:2]],
               # the candidate's fresh run against its screen (two runs of one configuration), and the reference's own repeat
               'rerun_vs_screen': None if d is None else {'screen': c['step_time'], 'rerun': fresh.step_time, 'log_ratio': round(d, 5),
                                                          'beyond_margin': m is not None and abs(d) > math.log1p(m)},
               'reference_repeat': round(pairs[-1], 5) if rerun.valid and rerun.step_time and t_ref else None,
               # repeats, not deviations from the reference (U-8-2): the reference's re-run against the reference (the
               # repeat floor), and the candidate's own: its fresh run against its screen
               'reference_repeat_loss_deviation': repeat_deviation(rerun),
               'candidate_repeat_loss_deviation': own_repeat(verifier, fresh, runs[1], cand)}
        tried.append(row)
        if row['passed']:
            return {'promoted': c['id'], 'config': cfg, 'overlay': overlay, 'tried': tried, 'gpu_hours': gpu_acc[0] / 3600,
                    'ref_offsets': offsets,
                    'speedup': next((i['evidence']['speedup'] for i in items if i['id'] == 'speed_reproduces'
                                     and i.get('evidence')), None)}
    return {'promoted': None, 'tried': tried, 'gpu_hours': gpu_acc[0] / 3600, 'ref_offsets': offsets,
            'why': 'no valid candidate faster than the reference' if not ranked else 'no candidate passed promotion'}


def profile(runner, task, ref_cfg: dict, ref: dict, run_id: str, log, off, purpose: str = 'profile') -> tuple:
    """The reference configuration's phases from one short profiled run (A11, project/profhook) when its log shows
    none (Megatron's timers and signals.phases give them for free): (phases in phase_summary's shape, GPU-hours);
    (None, 0) with the 'profile' ablation, on Windows or under srun, or when the hook could not run."""
    if (ref.get('evidence') or {}).get('phases') or 'profile' in off or not runner.can_profile():
        return None, 0.0
    log('[infrarsi] the log shows no phases: one short profiled run of the reference configuration')
    try:
        pr = runner.run(task, ref_cfg, None, purpose, run_id, candidate_id='c0', profile=True, bench=False,
                        steps=PROFILE_WARMUP + PROFILE_UPDATES + 3)
    except InfrastructureError as exc:
        log(f'[infrarsi] profile run: the hosts could not be reached ({exc}); no phases')
        return None, 0.0
    gh = (pr.get('cost') or {}).get('gpu_seconds', 0) / 3600
    p = (pr.get('evidence') or {}).get('profile') or {}
    ph = sig.profile_phases(p)
    if ph is None:                                      # U-6-5: the hook profiles PyTorch; a job without it gives nothing
        log('[infrarsi] profile run: no phases (' + str(p.get('error') or 'no phases.json: the hook did not load - the '
            'job may not use PyTorch, or it sets PYTHONPATH without keeping it') + ')')
    else:
        log('[infrarsi] phases (profile run): ' + ', '.join(f'{k} {100 * v:.1f} %' for k, v in ph['share'].items())
            + f', idle {100 * ph["residual_ms"] / ph["step_ms"]:.1f} %')
    return ph, gh


def apply_links(project: Project, runner, log=print) -> dict:
    """The job's interconnect as node 0 shows it (runner.link_facts): fills the manifest's `hardware.interconnect` in
    memory when the user did not set it, so the experience store scopes multi-node knowledge by the real fabric
    (TCP-only containers vs IB); a set value is kept, and a disagreement is logged."""
    found = runner.link_facts()
    have = project.raw['hardware'].get('interconnect')
    if found and not have:
        project.raw['hardware']['interconnect'] = dict(found)
        log(f'[infrarsi] interconnect (detected): ' + ', '.join(f'{k} {v}' for k, v in found.items()))
        return found
    if found.get('inter') and (have.get('inter') if isinstance(have, dict) else have) not in (None, found['inter']):
        log(f'[infrarsi] interconnect: the manifest says {have}, node 0 shows {found} (the manifest is used)')
    ic = dict(have) if isinstance(have, dict) else {'inter': have} if have else {}
    if found.get('bandwidth_gbs') and not ic.get('bandwidth_gbs'):      # a measured number the manifest did not give
        project.raw['hardware']['interconnect'] = {**ic, 'bandwidth_gbs': found['bandwidth_gbs']}
        log(f'[infrarsi] all-reduce bandwidth (measured): {found["bandwidth_gbs"]} GB/s bus bandwidth')
    return found


def gpu_contention(runner) -> list[dict]:
    """The runs that saw another process on their GPUs, or uncorrected ECC errors appear during them (runner
    snapshots before / after each run)."""
    return [{'candidate': a['candidate_id'], 'purpose': a['purpose'], 'warnings': a.get('gpu_contention') or [],
             'ecc': a.get('gpu_ecc') or []}
            for a in getattr(runner, 'attempts', []) if a.get('gpu_contention') or a.get('gpu_ecc')]


def _design(trace: dict) -> dict | None:
    """The screening design's resolution, the factors below it, and the factors probed or dropped before it (doe, or
    the doe part of doe+hls)."""
    st = (trace.get('extra') or {}).get('strategy')
    if not isinstance(st, dict):                         # knob-space strategies name themselves only
        return None
    for part in [st] + list(st.get('parts') or []):
        if isinstance(part, dict) and part.get('name') == 'doe' and (part.get('resolution') is not None
                                                                     or part.get('probed') or part.get('dropped')
                                                                     or part.get('why')):
            return {'resolution': part.get('resolution'), 'below_resolution': part.get('below_resolution') or [],
                    'probed': part.get('probed') or [], 'dropped': part.get('dropped') or [], 'why': part.get('why')}
    return None


def _bundle_numerics(trace: dict) -> list:
    """HLS's full bundle runs with their members' declared classes and the loss deviation (hls, or the hls part of
    doe+hls)."""
    st = (trace.get('extra') or {}).get('strategy')
    parts = [st] + list(st.get('parts') or []) if isinstance(st, dict) else []
    return [b for x in parts if isinstance(x, dict) for b in x.get('bundle_numerics') or []]


def _hls3(trace: dict) -> dict | None:
    """Strategy hls3's account (its describe()['hls3']: failures, memory model, bundles, stalls, ...); None otherwise."""
    st = (trace.get('extra') or {}).get('strategy')
    return st.get('hls3') if isinstance(st, dict) else None


def _wcs(trace: dict) -> dict | None:
    """Strategy wcs's account (its describe()['wcs']: the prior, the table, the picks, the memory model, the pair table,
    the patches); None otherwise."""
    st = (trace.get('extra') or {}).get('strategy')
    return st.get('wcs') if isinstance(st, dict) else None


# the control of the candidates (search/rpp.py; D-C1, CAND_r1 C9): no experience store - no plan read, no earlier
# results as priors, no write-back of the store or of the cross-run ledger
NO_EXPERIENCE = ('rpp-ctl',)


def _cand(trace: dict, name: str) -> dict | None:
    """A candidate strategy's account (search/esls.py, search/rpp.py: describe()['esls'] / ['rpp'] - its phases or draws,
    the memory ledger, the rung ladder and the low-fidelity account); None otherwise."""
    st = (trace.get('extra') or {}).get('strategy')
    return st.get(name) if isinstance(st, dict) else None


def _code_review(trace: dict) -> dict | None:
    """The proposer's code review (hls, greedy, the hls part of doe+hls): verdict, summary, findings."""
    st = (trace.get('extra') or {}).get('strategy')
    for part in ([st] + list(st.get('parts') or [])) if isinstance(st, dict) else []:
        review = ((part or {}).get('proposer') or {}).get('code_review') if isinstance(part, dict) else None
        if review:
            return review
    return None


def capability_of(project: Project, driver, override: float | None = None) -> float:
    """The capability knob: --capability, else manifest llm.capability, else inferred from the model name."""
    from ..agent.capability import clamp, infer
    from .llm import parse
    if override is not None:
        return clamp(override)
    llm = parse(project.raw.get('llm'))               # `llm: none` and `llm: vllm:<model>` are strings
    if llm.get('capability') is not None:
        return clamp(llm['capability'])
    return infer(getattr(driver, 'model', None))


def model_words(project: Project, strategy: str, driver, override: float | None, capability: float) -> str:
    """The run's model in the search line (U-1-15, U-3-2): none, not used by this strategy, or its name and capability
    with where the capability came from."""
    from .llm import parse
    if driver is None:
        return "no model (none in --llm or the manifest's llm block)"
    name = getattr(driver, 'model', None) or 'the model'
    if strategy not in search.LLM_STRATEGIES:
        return f'model {name} not used (strategy {strategy} does not call one)'
    how = '--capability' if override is not None else 'llm.capability' if parse(project.raw.get('llm')).get(
        'capability') is not None else 'inferred from the model name'
    return f'model {name}, capability {capability:.2f} ({how})'


def patch_status(project: Project, strategy: str | None = None, capability: float | None = None,
                 has_llm: bool | None = True) -> str:
    """Whether the search can patch code, in one phrase (doctor, the run log): only an LLM proposes patches, and not in
    menu mode (capability < 0.4, except the agent strategy), and only to the files code.mutable lists. has_llm None: not
    known yet (`run --llm` decides)."""
    from ..agent.capability import search_params
    files = project.files(project.raw['code']['mutable'])
    if not files:
        return 'off: code.mutable lists no file, so only knobs are searched'
    if strategy in ('random', 'tpe', 'doe', 'optico', 'rpp-ctl'):
        return f'off: strategy {strategy} changes knobs only'
    if has_llm is False:
        return 'off: no LLM (only an LLM proposes patches)'
    if strategy != 'agent' and capability is not None and search_params(capability)['menu']:
        return f'off: a model of capability {capability:.2f} (below 0.4) only rates knob moves'
    frozen = project.files(project.raw['code']['frozen'])
    return (('on' if has_llm else 'on once the run has an LLM (--llm or llm.backend)') + ': the search may edit '
            + ', '.join(files[:6]) + (f' and {len(files) - 6} more' if len(files) > 6 else '')
            + (f'; frozen: {", ".join(frozen[:6])}' if frozen else ''))


def configured_patch_status(project: Project) -> str:
    """patch_status from the manifest alone (doctor): its search strategy and llm block; without an llm backend there,
    the run's --llm decides."""
    from ..agent.capability import clamp, infer
    from .llm import parse
    raw = project.raw.get('llm')
    llm = parse(raw)
    backend = 'none' if raw == 'none' else llm.get('backend')
    cap = clamp(llm['capability']) if llm.get('capability') is not None else infer(llm.get('model')) if backend else None
    return patch_status(project, project.raw['search'].get('strategy'), cap, backend != 'none' if backend else None)


def run_log(path: Path, echo=print):
    """Every progress message goes to the run's own log file, and to `echo` (the console) when that still works: a run
    must not die because the terminal that started it went away."""
    def log(msg: str) -> None:
        try:
            with path.open('a', encoding='utf-8') as fh:
                fh.write(f'{_dt.datetime.now().strftime("%H:%M:%S")} {msg}\n')
        except OSError:
            pass
        try:
            echo(msg)
        except (OSError, ValueError):
            pass
    return log


BASE_FILES = 'base_files.json'                         # the mutable files as the search started from them (`infrarsi finish`)
RUN_META = 'run_meta.json'                             # what `infrarsi finish` needs when the run wrote no result


@contextlib.contextmanager
def counted_interrupts(hits: list):
    """Ctrl-C and SIGTERM in the with-block are appended to `hits` and raise KeyboardInterrupt as before: the executor swallows
    a second interrupt while it stops the job, and the search must still know that there were two."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def count(signum, frame):
        hits.append(signum)
        raise KeyboardInterrupt
    old = {s: signal.signal(s, count) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for s, handler in old.items():
            signal.signal(s, handler)


def run_project(project: Project, *, strategy: str | None = None, driver=None, out_root=None, tolerance: float | None = None,
                budget: dict | None = None, seed: int = 42, keep_trees: bool = False, log=print,
                capability: float | None = None, experience=None, ablate=None, resume=None,
                on_interrupt: str | None = None, requirements: dict | None = None, live: bool = False) -> dict:
    """`resume`: project/resume.Resume - the run of its directory goes on from its search-state checkpoint (the reference,
    the profile and the other pre-search results are the saved ones); everything after the search is a normal run's.
    `on_interrupt`: what Ctrl-C / SIGTERM does during the search, over `search.on_interrupt`: 'verify' (default) stops the
    search and sends the best candidates so far through the promotion, a second interrupt stopping that; 'stop' ends the
    run there, nothing promoted. `requirements`: the record of project/requirements.py the project was narrowed by; None: none."""
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%d-%H%M%S')
    if resume is not None:                              # the run keeps its directory and its id
        out, run_id = Path(resume.run_dir).resolve(), resume.run_id
    else:
        runs_root = Path(out_root or project.root / '.infrarsi' / 'runs').resolve()      # the job runs in its tree:
        runs_root.mkdir(parents=True, exist_ok=True)                                      # its paths are absolute
        for k in range(1, 100):                         # two runs started within one second get two directories
            run_id = f'{project.name}-{stamp}' + (f'-{k}' if k > 1 else '')
            out = runs_root / run_id
            try:
                out.mkdir(exist_ok=False)
                break
            except FileExistsError:
                continue
        else:
            raise RuntimeError(f'{runs_root}: a hundred runs of {project.name} in one second')
    from .live import LiveConsole
    console = LiveConsole(driver, enabled=live, log=log)
    log = run_log(out / 'run.log', console.write_log)
    if requirements:                                   # a copy next to the run, for whoever does not read result.json
        (out / 'requirements.json').write_text(json.dumps(requirements, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    if driver is not None and getattr(driver, 'out_root', False) is None:
        driver.out_root = out / 'llm_calls'            # what the model was sent and answered, next to the run
    if callable(getattr(driver, 'attach', None)):      # the debate's own calls and log
        driver.attach(out, log)
    task = project.task()
    verifier = make_verifier(project, tolerance)
    (out / RUN_META).write_text(json.dumps({'run_id': run_id, 'manifest': str(project.path) if project.path else None,
                                            'tolerance': verifier.tolerance}), encoding='utf-8')
    runner = ProjectRunner(project, out_dir=out / 'attempts', log=log, keep_trees=keep_trees, skip=out.parent)
    ref_cfg = project.reference_config()
    ex = project.raw.get('execution') or {}               # U-6-1: say where the runs happen when it is not here
    where = f' on {", ".join(str(h) for h in ex.get("hosts") or [])} (ssh)' if ex.get('where') == 'ssh' else \
        ' (slurm)' if ex.get('where') == 'slurm' else ''
    sc = project.raw['search']
    off = sorted(set(sc.get('ablate') or []) if ablate is None else set(ablate))   # research switches (search.ablate)
    if resume is not None:
        result, ref, vref, phases, profile_gh = resume.begin(project=project, task=task, runner=runner, verifier=verifier,
                                                             run_id=run_id, out=out, log=log, ref_cfg=ref_cfg, off=off)
        apply_links(project, runner, log)               # (the interconnect the first run filled in, from the saved link)
    else:
        log(f'[infrarsi] {run_id}: reference run of the current configuration{where} (progress: {out / "run.log"})')
        result = {'schema': 'infrarsi-run-v1', 'run_id': run_id, 'run_dir': str(out), 'project': project.describe(),
                  'manifest': str(project.path) if project.path else None, 'tolerance': verifier.tolerance,
                  'verifier': verifier.describe(), 'reference': {'config': ref_cfg},
                  **({'requirements': requirements} if requirements else {})}
        try:
            ref = runner.run(task, ref_cfg, None, 'reference', run_id, candidate_id='c0', bench='bandwidth' not in off)
        except InfrastructureError as exc:
            log(f'[infrarsi] stopped: the hosts could not be reached ({exc})')
            result.update(outcome='infrastructure_failure', infrastructure=str(exc))
            report_mod.write(result, out)
            return result
        vref = verifier(ref, None, ref)
        result['reference'].update(_brief(ref, vref))
        if isinstance(ref.get('evidence'), dict):       # which comparisons a candidate's signals support (the reference
            ev = ref['evidence']                          # against itself passes them trivially), as doctor reports it
            result['reference']['coverage'] = coverage(verifier.screen_checks(ev, ev))
        if not vref.valid:
            result.update(outcome='reference_failed', feedback=vref.feedback, gpu_contention=gpu_contention(runner))
            crlf = project.crlf_scripts() if project.posix_hosts() else []
            if crlf:                                        # a likely cause: a checkout with Windows line endings
                result['feedback'] = (result['feedback'] + f'\n{", ".join(crlf)}: Windows (CRLF) line endings: bash on the '
                                      'hosts reads every line with a trailing \\r (convert with dos2unix)').strip()
            if ((ref.get('evidence') or {}).get('events') or {}).get('oom') and 'accounting' not in off:
                from ..experience import runs as exp_runs     # out of memory: where the estimate puts the memory
                acc = exp_runs.accounting_for(project, ref)
                result['accounting'] = acc(ref_cfg, None) if acc else None
            report_mod.write(result, out)
            return result
        if verifier.workload_unverifiable(ref.get('evidence') or {}):
            result.update(outcome='not_verifiable', feedback=UNVERIFIABLE_WORKLOAD)   # D2-R211: no candidate could pass
            log(f'[infrarsi] stopped before the search: {UNVERIFIABLE_WORKLOAD}')
            report_mod.write(result, out)
            return result
        result['interconnect'] = apply_links(project, runner, log) or None   # before the experience plan scopes by it
        try:
            phases, profile_gh = profile(runner, task, ref_cfg, ref, run_id, log, off)
        except KeyboardInterrupt:                          # Ctrl-C / SIGTERM (U-7-3): the reference run is still reported
            log('[infrarsi] interrupted during the profile run, before the search; nothing is searched')
            result.update(outcome='interrupted', interrupted_during='profile')
            report_mod.write(result, out)
            return result
        if phases:
            result['reference']['phases'] = phases
    name = search.resolve(strategy or project.raw['search']['strategy'], has_llm=driver is not None, knobs=project.knobs)
    hooks = ProjectHooks(project, task, tolerance=verifier.tolerance,
                         requirements=(requirements or {}).get('guidance'))
    if resume is None and hooks.base_files:            # what `infrarsi finish` merges later edits against, as apply does
        (out / 'search').mkdir(parents=True, exist_ok=True)
        (out / 'search' / BASE_FILES).write_text(json.dumps(hooks.base_files, ensure_ascii=False), encoding='utf-8')
    hooks.reference_phases = phases
    hooks.reference_notes = ((ref.get('evidence') or {}).get('events') or {}).get('perf_notes')
    from ..experience import runs as exp_runs
    if resume is not None:                             # the values the run had before its search
        pairs, prior, doc_dev, exp, prior_saved = resume.pre_search(verifier)
    else:
        pairs = noise_pairs(project)                       # earlier runs' pairs; this run's start after them
        doc = doctor_run(project, ref)                     # the doctor run: a repeat of the reference - the loss check reads
        if doc and repeat_admissible(verifier(doc, None, ref)):   # from it whether this job's draws differ between runs
            verifier.repeat = evidence(doc)                # (DF16, D2-R204d) before it judges the repeat or any candidate
        result['reference']['loss_repeat'] = rep = verifier.loss_repeat(evidence(ref))
        note = loss_repeat_note(rep, verifier.tolerance)
        if note:
            log(f'[infrarsi] loss check: {note}')
        prior, d_doc = len(pairs), doctor_pair(project, verifier, ref, vref.step_time)
        if d_doc is not None:
            pairs.append(d_doc)
        doc_dev = doctor_floor(verifier, doc, ref)         # the repeat floor (at the tolerance the repeat set)
        if name in NO_EXPERIENCE:                          # the control: experience off whatever the manifest says
            experience = 'off'
            log(f'[infrarsi] experience: off (strategy {name} is the control: no experience store)')
        exp = None if name in NO_EXPERIENCE else \
            exp_runs.plan_for(project, ref, select=experience, driver=driver, log=log, ablate=off)   # None: experience off
    acc = None if 'accounting' in off else exp_runs.accounting_for(project, ref)   # (config, step time) -> estimate
    # earlier runs' results are experience too: only with every pack (`auto`), not with `off` or a pack selection
    every_pack = exp_runs.selection(experience if experience is not None else sc.get('experience')) is None
    options = {'full_steps': project.steps, 'short_steps': sc.get('short_steps'), 'probe_steps': sc.get('probe_steps', 2),
               'warmup': project.window[0] - 1,         # the updates before the scoring window are warm-up
               'multi_fidelity': sc.get('multi_fidelity', True),
               # the GPU-hours the production job still runs: the exchange rate of the value/cost stop; unset, the budget
               # ends the search (a default of 300 runs' worth stopped H2b's arm A after 6 of its 24 runs)
               'job_gpu_hours': float(sc['job_gpu_hours']) if sc.get('job_gpu_hours') else None,
               'capability': capability_of(project, driver, capability),
               'prior_results': prior_saved if resume is not None else prior_results(project) if every_pack else [],
               'run_noise': run_noise(project), 'noise_pairs': pairs, 'experience': exp, 'log': log, 'ablate': off,
               'accounting': acc, 'phase_cap': bool(sc.get('phase_cap')),
               'progress': log,                         # a console line per finished run (U-1-14; only this flow)
               'on_interrupt': on_interrupt or sc.get('on_interrupt') or 'verify',
               'control': console.poll if live else None}
    if options['ablate']:
        log(f'[infrarsi] ablation: {", ".join(options["ablate"])} off')
    ctx = search.Context(task=task, budget=task.search_budget(**(budget or {})), start_config=ref_cfg, runner=runner,
                         judge=verifier, hooks=hooks, run_id=run_id, out_dir=out / 'search', start_receipt=ref,
                         reference_receipt=ref, driver=driver, package=user_package() if name == 'agent' else None, seed=seed,
                         options=options)
    resumes = resume.resumes if resume is not None else []
    result_base = copy.deepcopy(result)                 # the pre-search results a resume starts from

    def flow_state() -> dict:                           # what a search-state checkpoint keeps of this flow (project/resume.py)
        return {'run_id': run_id, 'strategy': name, 'seed': seed, 'result': result_base, 'reference': ref, 'phases': phases,
                'profile_gh': profile_gh, 'pairs0': pairs0, 'prior': prior, 'doc_dev': doc_dev, 'repeat': verifier.repeat,
                'exp': {k: v for k, v in (exp or {}).items() if not callable(v)} if exp is not None else None,
                'prior_results': options['prior_results'], 'capability': options['capability'], 'resumes': resumes,
                'runner': {'attempts': runner.attempts, 'link': runner.link}, 'written': written}
    pairs0 = list(pairs)
    written = resume.flow.get('written') if resume is not None else None
    options['checkpoint'] = ckpt.Writer(out / 'search' / 'state', flow=flow_state)
    if resume is not None:
        options['resume'] = resume.apply
    b = ctx.budget                                      # U-1-13, U-1-15, U-3-2: the budget and the model in words
    from ..agent.budget import UNLIMITED
    wall_label = f'{b.wall_seconds / 3600:.3g} h' if b.wall_seconds < UNLIMITED else 'no time cap'
    gpu_label = f'{b.gpu_hours:.3g} GPU-hours' if b.gpu_hours < UNLIMITED else 'no GPU-hour cap'
    token_label = f'{b.llm_tokens:,} LLM tokens' if b.llm_tokens < UNLIMITED else 'no token cap'
    log(f'[infrarsi] search: strategy {name}; budget {b.screens} full runs, {wall_label}, '
        f'{gpu_label}, {token_label}' + (f', {b.llm_usd} USD' if b.llm_usd is not None else '')
        + f'; {model_words(project, name, driver, capability, options["capability"])}')
    if driver is not None and name in search.LLM_STRATEGIES and getattr(driver, 'unpriced', False):
        log(f'[infrarsi] ! {UNPRICED}')               # U-5-2: a USD budget would never stop this run
    log(f'[infrarsi] code patches {patch_status(project, name, options["capability"], driver is not None)}')
    if callable(getattr(driver, 'summary', None)):
        d = driver.describe()['debate']
        log(f'[infrarsi] debate on: second model {d["second"]}'
            + (f', patch check by {d["verifier"]} (at most {d["rounds"]} rounds)' if d['rounds'] else ', no patch check'))
    wrap_up = options['on_interrupt'] == 'verify'
    if wrap_up:
        log('[infrarsi] Ctrl-C stops the search and re-verifies the best candidate so far; Ctrl-C again stops without that')
    hits: list = []
    try:
        with counted_interrupts(hits), console:
            trace = search.run(name, ctx).to_dict()
    except KeyboardInterrupt:                          # Ctrl-C / SIGTERM: what the search measured is reported (D1-18)
        if not ctx.sessions:
            raise
        trace = ctx.sessions[-1].finalize(stop_reason=USER_STOP if wrap_up else 'interrupted').to_dict()
    if (out / 'search' / 'state' / 'checkpoint.pkl').is_file():
        import shlex
        log('[infrarsi] 💾 Resume: infrarsi run --resume ' + shlex.quote(str(out))
            + ' --screens TOTAL (TOTAL includes completed runs; increase other exhausted caps too)')
    if trace.get('stop_reason') == USER_STOP and len(hits) > 1:   # interrupted twice (the second while the job was stopped)
        trace['stop_reason'] = 'interrupted'
    # U-7-3: an interrupt (Ctrl-C / SIGTERM) after the search ends in the same result and report, as far as the run got:
    # `promo` is the promotion reached (marked interrupted until it is over), `remembered` / `exp_sum` the write-back's,
    # `phase` the step after the promotion
    promo = {'promoted': None, 'tried': [], 'gpu_hours': 0.0, 'ref_offsets': [], 'interrupted': True,
             'why': 'interrupted before the promotion started'}
    remembered, exp_sum, phase = 0, None, 'write-back'

    def finish(at: str | None, quiet: bool = False) -> dict:
        """result.json and REPORT.md from the state reached; `at`: the step a caught interrupt cut, None if none came;
        `quiet`: no console line (the report written before a re-verification, replaced when that ends)."""
        during = ('search' if trace.get('stop_reason') == 'interrupted' else
                  'promotion' if promo.get('interrupted') else at)
        if promo.get('promoted'):                      # the files the patches were written against: apply merges
            promo['base'] = {rel: hooks.base_files.get(rel) for rel in promo.get('overlay') or {}}   # later edits
        if exp_sum is not None and 'written' not in exp_sum:   # cut while its pack was written: none was published
            exp_sum.update(written=None, not_written='the run was interrupted before its pack was written')
        outcome = 'interrupted' if during else 'improved' if promo.get('promoted') else \
            'infrastructure_failure' if promo.get('infrastructure') else 'no_change'
        if promo.get('infrastructure'):
            result['infrastructure'] = promo['infrastructure']
        if during:                                     # where Ctrl-C / SIGTERM stopped the run (the report's Interrupted)
            result['interrupted_during'] = during
        else:
            result.pop('interrupted_during', None)     # (the report written before a re-verification that then finished)
        if not quiet:
            log(f'[infrarsi] {outcome}' + (f' ({during})' if during else '')
                + (f', x{promo["speedup"]:.3f} on the fresh re-run' if promo.get('speedup') else '')
                + f'; report: {out / "REPORT.md"}')
        extra = trace.get('extra') or {}
        result.update(outcome=outcome, strategy=name, promotion=promo,
                      experience=exp_sum, research={'ablate': off, 'phase_cap': options['phase_cap']},
                      accounting=acc(ref_cfg, vref.step_time) if acc else None, knob_phases=hooks.knob_phases,
                      search={**{k: trace.get(k) for k in ('stop_reason', 'used', 'budget', 'selected',
                                                           'selected_step_time', 'invalid_proposals', 'verifier',
                                                           'accounting_complete')},
                              'stop_numbers': extra.get('stop_numbers'), 'noise': extra.get('noise') or None,
                              'rechecks': extra.get('rechecks') or None,
                              'stop_rule': extra.get('stop_rule'), 'job_gpu_hours': options['job_gpu_hours'],
                              'design': _design(trace), 'code_review': _code_review(trace),
                              'residual': extra.get('residual'), 'remembered': remembered,
                              'ledger': extra.get('ledger'), 'bundle_numerics': _bundle_numerics(trace),
                              'doctor_repeat_loss_deviation': doc_dev, 'llm_calls': len(trace.get('llm_calls') or []),
                              'llm_errors': [str(c['error'])[:300] for c in trace.get('llm_calls') or []
                                             if c.get('error')],
                              **({'hls3': _hls3(trace)} if _hls3(trace) else {}),
                              **({'wcs': _wcs(trace)} if _wcs(trace) else {}),
                              **{n: _cand(trace, n) for n in ('opro', 'ltune') if _cand(trace, n)},
                              **{n: _cand(trace, n) for n in ('xfer',) if _cand(trace, n)},
                              # the finalists (search/finalist_base and their modules): describe()[name]
                              **{n: _cand(trace, n) for n in ('bcomp', 'beam', 'cards') if _cand(trace, n)},
                              **{n: _cand(trace, n) for n in ('esls', 'rpp') if _cand(trace, n)}},
                      **({'resumes': resumes} if resumes else {}),
                      **({'debate': driver.summary()} if callable(getattr(driver, 'summary', None)) else {}),
                      candidates=trace['candidates'], screens=trace['screens'], gpu_contention=gpu_contention(runner),
                      cost={'reference_gpu_hours': result['reference']['gpu_hours'],
                            'search_gpu_hours': trace['used']['gpu_hours'],
                            'promotion_gpu_hours': promo.get('gpu_hours', 0.0), 'profile_gpu_hours': profile_gh,
                            'llm_tokens': trace['used'].get('llm_tokens'),
                            'llm_usd': trace['used'].get('llm_usd'),
                            'llm_usd_known': not getattr(driver, 'unpriced', False),   # an API without llm.price: unknown
                            **({'requirements_llm_tokens': requirements['compile'].get('llm_tokens'),   # (outside the budget)
                                'requirements_llm_usd': requirements['compile'].get('llm_usd')}
                               if requirements and (requirements.get('compile') or {}).get('cached') is False else {})})
        report_mod.write(result, out)
        return result

    try:                                               # one guard from the end of the search to the report (U-7-3)
        stop = str(trace.get('stop_reason') or '')
        if stop == 'interrupted':
            promo = {'promoted': None, 'tried': [], 'gpu_hours': 0.0, 'ref_offsets': [],
                     'why': 'the run was interrupted during the search, so nothing was promoted'}
            log('[infrarsi] interrupted during the search; nothing is promoted')
        elif stop.startswith('infrastructure'):
            promo = {'promoted': None, 'tried': [], 'gpu_hours': 0.0, 'ref_offsets': [], 'infrastructure': stop,
                     'why': 'the search stopped because the hosts could not be reached, so nothing was promoted'}
            log(f'[infrarsi] stopped: {stop}; nothing is promoted')
        else:                                          # U-1-17: only candidates faster than the reference are re-run
            if stop == USER_STOP:                      # stopped by the user: what was measured goes through the promotion;
                promo['why'] = 'the search was stopped and the re-verification of the best candidate did not finish'
                finish(None, quiet=True)               # a report on disk first: a hard kill during the re-runs leaves it
                log('[infrarsi] search stopped by you: re-verifying the best candidate so far (Ctrl-C again stops without '
                    f'that; `infrarsi finish {out}` does it later)')
            faster = any(c['id'] != 'c0' and c.get('valid') and c.get('step_time') and vref.step_time
                         and c['step_time'] < vref.step_time for c in trace['candidates'])
            log('[infrarsi] promotion: fresh re-runs of the fastest valid candidates against the reference' if faster
                else '[infrarsi] promotion: no valid candidate is faster than the reference, nothing to re-run')
            promo = promote(project, task, runner, verifier, trace, ref, run_id, out / 'search' / 'receipts',
                            max_candidates=int(sc.get('promotion_candidates') or 3))
            if promo.get('interrupted'):
                log(f'[infrarsi] {promo["why"]}; nothing is promoted')
        noise = (trace.get('extra') or {}).get('noise') or {}
        remembered = (remember(project, trace, promo, skip=(written or {}).get('ledger_ids') or ())
                      if name not in NO_EXPERIENCE else 0)   # (the control writes none; a resumed run skips the rows written)
        mine = (noise.get('pairs') or pairs)[prior:][(written or {}).get('noise') or 0:]   # the doctor pair and the
        #                                                                                    confirmation pairs of this run
        remember_noise(project, [abs(x) for x in mine] + list(promo.get('ref_offsets') or []))
        if exp is not None:                            # coverage, the composite's fate, the judgment; the run as a pack
            exp_sum = exp_runs.summary(exp, trace, noise.get('sigma_pair'))
            written = exp_runs.write_back(project, exp, trace, promo, run_id, noise.get('sigma_pair'), exp_sum)
            exp_sum['written'] = str(written) if written else None
            if not written:                            # (the toy of R2-09: no model facts, and nothing said so)
                exp_sum['not_written'] = exp_runs.unwritable(exp['fingerprint']) or (
                    'its pack did not validate: ' + '; '.join(exp.get('write_back_errors') or ['no detail']))
        phase = 'report'
        return finish(None)
    except KeyboardInterrupt:                          # anywhere after the search: the same report, as far as it got
        return finish(phase)

