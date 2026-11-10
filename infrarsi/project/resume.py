"""`infrarsi run --resume <run dir>`: a run continued from the search-state checkpoint it left (search/checkpoint.py,
agent/state.py: <run>/search/state/checkpoint.pkl, written after every observed run and at the end of the search).

    resume_project(project, run_dir, driver=..., budget=...)    the run's search goes on from where it stopped under the budget
                                     given now; the promotion, the report and the write-back are a normal run's

What carries over, and what is the resuming invocation's:

    carried over   the session (candidates with their configuration, overlay files, receipts and verdicts; screens, events, the
                   model-call log, the amounts used - screens, dollars, tokens, calls, GPU-hours, wall clock), the engine (the
                   incumbent, the noise pairs, the keys tried ...), the strategy's whole state (ledgers, queues, stacks, the
                   code track, the fix log, counters, histories) and its random generators, the proposer's and the budget
                   driver's counters, the reference and the flow's pre-search results (its profile, the doctor's repeat, the
                   earlier runs' noise pairs, the experience plan, the priors) - the run keeps its id and its directory
    the new run's  the budget (the used amounts count towards it: the run stops when the TOTAL reaches the budget now
                   given), the caps the budget and the manifest set (derived again, `rebudget`), the hosts, the model, the
                   handles (runner, hooks, driver, loggers)

The reference is the saved one (an exact continuation); `--remeasure-reference` runs it again first, records both and the ratio
and leaves the earlier screens as they were measured (they are never rescaled). A tree whose package hash differs from the
checkpoint's is refused unless `--allow-code-change`. Every resume adds an entry to `resumes` in result.json, REPORT.md and
the search's trace.
"""
from __future__ import annotations

import copy
import dataclasses
import datetime as _dt
import json
from pathlib import Path

from .. import search
from ..agent import state as S
from ..search import checkpoint as ckpt


class ResumeError(RuntimeError):
    """A run that cannot be resumed (no checkpoint, another code version, a strategy that keeps no state ...)."""


@dataclasses.dataclass
class Resume:
    """What `flow.run_project(resume=...)` reads: the run's directory and id, the checkpoint (`data`: its plain parts and the
    pickled state), the flow's saved state and the list of resumes (the earlier ones and this one)."""
    run_dir: Path
    run_id: str
    data: dict
    path: Path
    flow: dict
    remeasure: bool = False
    allow_code_change: bool = False
    resumes: list = dataclasses.field(default_factory=list)
    entry: dict = dataclasses.field(default_factory=dict)
    applied: bool = False
    requirements: dict | None = None                     # the run's requirements record, as it started with them

    # ------------------------------------------------------------ the flow's pre-search results
    def begin(self, *, project, task, runner, verifier, run_id, out, log, ref_cfg, off):
        """(result, reference receipt, its verdict, phases, profile GPU-hours): the pre-search results the run saved, the
        runner's attempts restored; with `remeasure` the reference runs once more first and the ratio is recorded."""
        fl = self.flow
        result = copy.deepcopy(fl['result'])
        result.update(run_dir=str(out), project=project.describe(),
                      manifest=str(project.path) if project.path else result.get('manifest'))
        ref = fl['reference']
        runner.attempts[:] = copy.deepcopy(fl['runner']['attempts'])
        runner.link = copy.deepcopy(fl['runner']['link'])
        vref = verifier(ref, None, ref)
        self.entry['reference'] = {'mode': 'reused', 'attempt': ref.get('attempt_id'), 'step_time': vref.step_time}
        if self.remeasure:
            log(f'[infrarsi] {run_id}: resuming - the reference configuration runs once more (--remeasure-reference)')
            new = runner.run(task, ref_cfg, None, 'reference', run_id, candidate_id='c0', bench='bandwidth' not in off)
            vnew = verifier(new, None, new)
            if not vnew.valid or not vnew.step_time or not vref.step_time:
                raise ResumeError('the reference configuration does not pass the run checks now: ' + '; '.join(vnew.reasons[:3]))
            ratio = vnew.step_time / vref.step_time
            self.entry['reference'] = {
                'mode': 'remeasured', 'attempt': new.get('attempt_id'), 'step_time': vnew.step_time,
                'saved_attempt': ref.get('attempt_id'), 'saved_step_time': vref.step_time,
                'ratio_new_over_saved': round(ratio, 5), 'gpu_hours': (new.get('cost') or {}).get('gpu_seconds', 0) / 3600,
                'note': 'the search goes on against the saved reference; the runs before the resume are not rescaled - '
                        f'multiply their step times by {ratio:.4f} to read them against the new reference'}
            log(f'[infrarsi] reference again: {vnew.step_time:.4g} s per update, {ratio:.4f} x the saved {vref.step_time:.4g} s')
        return result, ref, vref, fl['phases'], fl['profile_gh']

    def pre_search(self, verifier):
        """(noise pairs at the search's start, how many of them earlier runs' were, the doctor repeat's loss deviation,
        the experience plan, the prior results): the values the run computed before its search, as it had them; the
        verifier gets the doctor's repeat back."""
        fl = self.flow
        verifier.repeat = copy.deepcopy(fl['repeat'])
        return list(fl['pairs0']), fl['prior'], fl['doc_dev'], copy.deepcopy(fl['exp']), copy.deepcopy(fl['prior_results'])

    # ------------------------------------------------------------ the search
    def apply(self, session, engine, strategy) -> None:
        """The search's saved state onto the objects the resumed run built (search/__init__ passes it as options['resume']
        to the strategy's registration); the caps follow the new budget and manifest; the entry for `resumes` is made."""
        live = ckpt.live_objects(session, engine, strategy)
        state = S.state_of_file(self.data, live)
        session.out_dir = Path(self.run_dir) / 'search'
        ckpt.restore(state, session, engine, strategy)
        if session.hooks is not None:
            session.hooks.live_notes = list(getattr(session, 'live_notes', []))
        if not hasattr(strategy, 'rebudget'):
            raise ResumeError(f'the strategy {getattr(strategy, "name", "?")} cannot be resumed')
        strategy.rebudget(engine)
        h = self.data['header']
        self.entry.update(
            budget_before=h.get('budget'), budget_after=session.budget.as_dict(), used_before=h.get('budget_used'),
            stop_before=h.get('stop'), screens_done=h.get('screens_done'),
            from_checkpoint={'file': self.path.name, 'saved_utc': h.get('saved_utc'), 'why': h.get('why'),
                             'sequence': h.get('sequence'), 'incumbent': h.get('incumbent'), 'bytes': h.get('bytes')},
            cleared_stop_flag=h.get('stop'),
            caps_after={k: getattr(strategy, k, None) for k in ('calls_max', 'usd_max', 'code_asks', 'history_max',
                                                                'patches_per_ask')
                        if getattr(strategy, k, None) is not None})
        self.resumes.append(self.entry)
        engine.resumes = self.resumes
        self.applied = True


def checkpoint_dir(run_dir: Path) -> Path:
    return Path(run_dir) / 'search' / S.STATE_DIR


def prepare(run_dir, *, remeasure: bool = False, allow_code_change: bool = False) -> Resume:
    """The checkpoint of a run directory, checked: it exists, its package hash is this tree's (or `allow_code_change`), its
    strategy keeps a resumable state."""
    run_dir = Path(run_dir).resolve()
    if not (run_dir / 'search').is_dir():
        raise ResumeError(f'{run_dir} is not a run directory (no search/ in it)')
    try:
        data, path = S.read(checkpoint_dir(run_dir))
    except S.CheckpointError as exc:
        raise ResumeError(f'{run_dir} has no checkpoint to resume from ({exc}); runs made before search-state checkpoints '
                          'existed cannot be resumed') from exc
    h, flow = data['header'], data.get('flow')
    if not isinstance(flow, dict):
        raise ResumeError(f'{path}: the checkpoint carries no flow state')
    if flow.get('strategy') not in search.RESUMABLE:
        raise ResumeError(f'the run used strategy {flow.get("strategy")!r}; only {", ".join(search.RESUMABLE)} keep a '
                          'resumable state')
    prev = run_dir / 'result.json'
    if prev.is_file():                                   # completed once: the write-back must not repeat what it wrote
        try:
            part = json.loads(prev.read_text(encoding='utf-8')).get('search') or {}
        except ValueError:
            part = {}
        if part.get('remembered'):
            flow['written'] = {'ledger_ids': [h.get('id') for h in part.get('ledger') or []
                                              if isinstance(h, dict) and h.get('status') not in ('open', 'superseded')],
                               'noise': max(0, len((part.get('noise') or {}).get('pairs') or []) - int(flow.get('prior') or 0))}
    now = S.package_hash()
    r = Resume(run_dir=run_dir, run_id=flow['run_id'], data=data, path=path, flow=flow, remeasure=remeasure,
               allow_code_change=allow_code_change, resumes=list(flow.get('resumes') or []),
               requirements=(flow.get('result') or {}).get('requirements'))
    changed = h.get('package_hash') != now
    if changed and not allow_code_change:
        raise ResumeError(f'the checkpoint was made by another version of the code (package hash {str(h.get("package_hash"))[:12]}, '
                          f'this tree {now[:12]}); resume with the tree that made it, or pass --allow-code-change')
    r.entry = {'utc': _dt.datetime.now(_dt.timezone.utc).isoformat(), 'from': 'checkpoint',
               'code_hash_before': h.get('package_hash'), 'code_hash_after': now, 'code_changed': changed,
               'allow_code_change': allow_code_change}
    return r


def resume_project(project, run_dir, *, prepared: Resume | None = None, driver=None, tolerance=None,
                   budget: dict | None = None, log=print, capability=None, experience=None, ablate=None,
                   remeasure_reference: bool = False, allow_code_change: bool = False, seed: int | None = None,
                   keep_trees: bool = False, on_interrupt: str | None = None, live: bool = False) -> dict:
    """The run of `run_dir` continued (flow.run_project with the checkpoint): result.json, REPORT.md and the trace of the
    same directory, with the resume recorded in `resumes`. `prepared`: `prepare`'s result, when the caller made it."""
    from .flow import run_project
    r = prepared or prepare(run_dir, remeasure=remeasure_reference, allow_code_change=allow_code_change)
    res = run_project(project, strategy=r.flow['strategy'], driver=driver, tolerance=tolerance, budget=budget,
                      seed=r.flow.get('seed', 42) if seed is None else seed, keep_trees=keep_trees, log=log,
                      capability=capability, experience=experience, ablate=ablate, resume=r, on_interrupt=on_interrupt,
                      requirements=r.requirements, live=live)
    if not r.applied and res.get('outcome') not in ('reference_failed', 'infrastructure_failure', 'interrupted'):
        raise ResumeError('the search did not take the checkpoint: the run was not resumed (its result is not trustworthy)')
    return res


def latest_run(target='.', out_root=None) -> Path:
    """Newest saved search for this project (also works with a custom --out directory)."""
    from .manifest import Project
    project = Project.load(target)
    root = Path(out_root or project.root / '.infrarsi' / 'runs')
    candidates = []
    for checkpoint in root.glob('*/search/state/checkpoint.json'):
        run = checkpoint.parents[2]
        if not any((checkpoint.parent / name).is_file() for name in ('checkpoint.pkl', 'checkpoint.prev.pkl')):
            continue
        try:
            meta = json.loads((run / 'run_meta.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if meta.get('manifest') and Path(meta['manifest']).resolve() == project.path.resolve():
            candidates.append((checkpoint.stat().st_mtime_ns, run))
    if not candidates:
        raise ResumeError(f'no saved search for {project.path} in {root}; pass a RUN_DIR explicitly')
    return max(candidates)[1]
