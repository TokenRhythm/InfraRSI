"""F1 arm launchers (AMENDMENT_EXT_003 Q2, Q5). Each call runs one piece of the window on the four leased nodes; the
order (U+C ref -> F-HT -> F-PROMPTS -> F-OPTICO -> F-A0 -> F-RND, each arm's locked pair right after its search) is the
orchestrator's.

  run_uc_reference   the registered U+C recipe (numerics_role reference), judged against the stored U+C A cell
                     rsi_perf_v1_c1_p03; a FAIL stops the window (the runner does not reproduce the path)
  run_ht             F-HT = S_match, judged against the new U+C reference screen (recovered_fraction denominator)
  run_f_prompts      baselines.prompts_oneshot.run_prompts(hooks=FlashNextHooks): one call, top-3, three screens
  run_f_optico       baselines.optico_style.run_optico_style(hooks=...): OptiCo's 4 prompt classes + pool, D-OC-15
  run_f_a0           agent.optimize(hooks=FlashNextHooks, layers=('L0',), mode='strong'), verifier on, tools off
  run_f_rnd          SearchSession + switch_space.RandomSwitchSampler(seed 42): the same validator, no LLM
  run_locked         Q2 plan-A locked pair {U+C, final selection}: arm-index parity order, no run when the selection
                     is U+C, locked_self_mismatch and void_pair guards; attempt 1, and attempt 2 only after a void
                     attempt 1 of the same arm and selection (run it with out_dir = the arm's run directory)
  margin / summarize margin_F1.json and summary_F1.json (Q5: S_UC estimator, labels, recovered_fraction, CTG, AUC);
                     formal only for a complete window: four arms with Q5 budgets, sealed traces whose screens are the
                     arm's signed screen receipts, sealed locked results whose two halves are one back-to-back run in
                     the parity order and re-derive every number, U+C reference and F-HT screens, one node set for
                     every launched screen, no differing copies of a receipt; the margin is recomputed from the
                     window's receipts (a supplied margin file must match)
Each arm's trace.json and each locked*.json are sealed (harness.signing, the receipt key) when written: an edited file
is not formal.
Screens the harness lost after the path measured them (AMENDMENT_EXT_009, supersede.py): for F-PROMPTS a signed
superseding record replaces the `runner_error` receipt (retrace re-derives and re-seals the trace; the locked pair's own
screen is the record); for the multi-step arms the screen stays as the method saw it and a recovered value is
descriptive only; a pair label is withheld ('no comparison (harness)') when an arm has n_harness_lost >= 2 or a recovered
value beats its own selection.
Every arm starts from U+C with the U+C reference screen as its start receipt (not re-screened), the same driver
(agent.arms.strong_driver, Opus 5.5 CLI), the same judge (EnvJudge on the runner's signed receipts) and the same space.
Every entry point refuses a uc_ref that is not the U+C reference screen (guards.uc_reference_problems).

Setting 2 (code edits): run_f_a0_code = run_f_a0 with the code arm's task (F-A0-CODE: S_F1 + C_F1, layers L0 + L1) and
deviation D-F1-7; its trace adds extra.code_space (the space hash and the selection's overlay digest and files). Its
locked pair is run_locked's: the candidate half runs the selection's point WITH its overlay (the trace's
selected_overlay, re-validated and bound to the own screen's code_overlay digest), the U+C half runs U+C without one;
locked.json records the overlay. A selection = U+C with an overlay is not U+C (it is locked like any other). The
switch-only window (ARM_INDEX, its four arms, labels and formal checks) is unchanged; a summary given the code arm too
adds its row and the label 'F-A0-CODE vs F-A0' (and its locked U+C half joins S_UC).

The v4 extras (task.V4_ARMS: F-OPEN4-BCOMP) run as F-OPEN-BCOMP (run_f_is2 with bcomp,
V4_STRATEGY; run_locked in their v3 counterpart's parity order) from the v4 trees, which read the v4 caps off the task
(task.V4_MANIFEST). They stay out of the window's counts: their locked U+C halves are listed under the margin's excluded,
not in S_UC; a summary given them marks their rows (extra_arm = task.V4_LABEL) and leaves them out of every pair label
and of the drift's max / min.

The same-money extras (task.BUD_ARMS: F-BUD-BCOMP, user 11-03) are extras of the same kind
(task.EXTRA_ARMS: outside S_UC, the pair labels and the drift; their rows carry task.extra_label), run from the v4.1 trees
(pc_code_o2b bcomp; the user dropped v4.2 11-04) until they have spent what F-OPEN-AR spent on this job
(task.BUD_MANIFEST caps). They keep search-state checkpoints (run_f_is2 checkpoint=True: the tree's search/checkpoint.py Writer on
<out>/state) and can be continued from the last one (resume=True; the CLI's `arm ... --resume`).
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import statistics
from pathlib import Path

from ...harness import signing
from ..common import trajectory as TJ
from ..common.env_judge import EnvJudge
from ..common.guards import GuardResult
from . import code_space as C
from . import deviations as dv
from . import guards as G
from . import judge as J
from . import parse as P
from . import supersede as SU
from . import switch_space as S
from . import task as T
from .hooks import FlashNextHooks

ARM_INDEX = T.ARM_INDEX                 # the switch-only window (setting 1); the code arm: T.CODE_ARM_INDEX
ALL_ARM_INDEX = {**T.ARM_INDEX, **T.CODE_ARM_INDEX, **T.OPEN_ARM_INDEX}   # + the open-space wave (section 6 F8)
CODE_PAIRS = (('F-A0-CODE', 'F-A0'),)   # labelled only when the code arm is in the summary
# the open-tree search strategy -> its open-wave arm (arms.run_f_is2 default; the CLI may pass arm explicitly)
STRATEGY_ARM = {'bcomp': 'F-OPEN-BCOMP', 'autoresearch': 'F-OPEN-AR', 'rebench': 'F-OPEN-RB'}
OURS_STRATEGIES = ('bcomp',)     # the proposed-method arms that read cross-job experience (D4); AR / RB read none
# the v4 extras' strategies: the v4 trees register them under the v3 names (pc_code_o2 bcomp)
V4_STRATEGY = {'F-OPEN4-BCOMP': 'bcomp'}
# every extra's strategy: + the same-money extras (the v4.1 trees pc_code_o2b bcomp)
EXTRA_STRATEGY = {**V4_STRATEGY, 'F-BUD-BCOMP': 'bcomp'}


def _code_arm(arm: str, trace=None) -> bool:
    """True when the arm's locked pair / overlay machinery is in force: the setting-2 arm F-A0-CODE always, and an
    open-wave arm whose trace ran with C_F1 open (its sealed trace carries the extra.code_space block, written by
    run_f_is2 / run_f_a0_code). A switch-only run (code=False, or an unknown trace) is not a code arm."""
    if arm in T.CODE_ARMS:
        return True
    if arm not in T.OPEN_ARMS or trace is None:
        return False
    tr = trace if isinstance(trace, dict) else _load(trace)
    return bool((tr.get('extra') or {}).get('code_space'))
REACHED = 1.30                          # reached_1.30x: locked speedup over the arm's own U+C half
MIN_N = 4
FLOOR, K = 0.03, 3.0
EXT009_DISCLOSURE = "two F-PROMPTS screens (c1, c2 of `f1_prompts_20261021b`) completed on the nodes but were lost by our collection transport (#37, a reply-truncation race in our SSH read loop); their values come from a read-only re-collection of the intact node files judged by the unchanged judge, recorded as signed superseding receipts beside the retained `runner_error` receipts; no GPU was re-spent and no LLM call repeated; c1's value was known when the rule was written, and the rule is value-blind"
NO_COMPARISON = 'no comparison (harness)'


def _write(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    return path


def _load(x):
    return json.loads(Path(x).read_text(encoding='utf-8-sig')) if isinstance(x, (str, Path)) else x


def _seal(obj: dict, path: Path) -> dict:
    """Sign a finished trace or locked result with the harness receipt key (as written: JSON round trip first) and
    write it; the formal summary accepts only sealed traces and locked results (an edited file fails verification)."""
    d = json.loads(json.dumps(obj, ensure_ascii=False, default=str))
    signing.sign(d)
    _write(path, d)
    return d


def _ts(x):
    try:
        return _dt.datetime.fromisoformat(str(x))
    except (TypeError, ValueError):
        return None


def ht_point() -> dict:
    """F-HT: the best human recipe (registered S_match), never shown to an arm."""
    return S.registered_points()['S_match']


def _metric(receipt: dict | None) -> float | None:
    v = ((receipt or {}).get('metric') or {}).get('value')
    return float(v) if P.finite(v) and v > 0 else None


def _uc_ref_value(uc_ref: dict) -> float:
    """The U+C reference screen's metric, after its identity checks (guards.uc_reference_problems)."""
    problems = G.uc_reference_problems(uc_ref)
    if problems:
        raise ValueError(f'not the U+C reference screen receipt: {problems}')
    return _metric(uc_ref)


# ---------------------------------------------------------------- the runner (X5's FlashNextRunner; its default judge
# FlashNextRunner.x6_judge = parse.collect -> judge.evaluate -> guards.check -> judge.receipt_extra)
def make_runner(nodes, *, out_dir, reference=None, cap_s: float = G.CAP_S, runtime_dir=None, log=None, ops_by_node=None):
    """The F1 runner on WINH (WinhNodeOps per node) with the frozen gates; runtime_dir = the local re-rooted variant
    runtime (built once by deploy.custom_runtime_local and checked against frozen_manifest.json)."""
    from ..common.remote_exec import WinhNodeOps
    from . import deploy as D
    from . import runner as R
    from . import variant as V
    nodes = [str(n).upper() for n in nodes]
    runtime_dir = Path(runtime_dir) if runtime_dir else Path(out_dir).parent / '_variant_runtime'
    if not (runtime_dir / 'systems_policy.py').exists():
        D.custom_runtime_local(runtime_dir)
    want = G.load_manifest()['variant_runtime']['inventory']
    if V.inventory(runtime_dir) != want:
        raise RuntimeError(f'{runtime_dir}: not the frozen variant runtime (frozen_manifest.json)')
    ops = ops_by_node or {n: WinhNodeOps(n) for n in nodes}
    return R.FlashNextRunner(ops, nodes=nodes, runtime_dir=runtime_dir, cap_s=cap_s, reference_parsed=reference,
                             gates=J.load_gates(), out_dir=out_dir, log=log or (lambda m: print(m, flush=True)))


# ---------------------------------------------------------------- references (plan A, before the arms)
def run_uc_reference(runner, *, run_id: str, out_dir, cap_s: float | None = None, stored=None) -> dict:
    """U+C through the new runner, judged against the stored U+C A cell. Returns {receipt, stop_window, ...}."""
    task = T.make_task('UC-REF')
    stored = stored if stored is not None else J.stored_reference()
    r = runner.run(task, S.start_point(), None, 'reference', run_id, candidate_id='uc-ref', role='reference',
                   reference_parsed=stored, **({'cap_s': cap_s} if cap_s else {}))
    res = {'kind': 'UC-REF', 'receipt_attempt': r.get('attempt_id'), 'valid': r.get('valid'),
           'numerics_class': (r.get('judge') or {}).get('numerics_class'), 'step_time_s': _metric(r),
           'judge_value': ((r.get('metric') or {}).get('raw') or {}).get('judge_value'),
           'stored_cell': {'wave': 'rsi_perf_v1_c1_p03', 'group': 'A', 'value': J.STORED_UC_A_METRIC},
           'stop_window': r.get('valid') is not True,
           'note': 'a FAIL stops the window: the runner does not reproduce the path (AMENDMENT_EXT_003 Q5)'}
    out = Path(out_dir)
    _write(out / 'uc_ref.json', r)
    _write(out / 'uc_ref_result.json', res)
    return {**res, 'receipt': r}


def run_ht(runner, *, run_id: str, out_dir, uc_ref: dict, cap_s: float | None = None) -> dict:
    t_ref = _uc_ref_value(uc_ref)
    task = T.make_task('F-HT')
    r = runner.run(task, ht_point(), None, 'reference', run_id, candidate_id='f-ht', role='candidate',
                   reference_parsed=J.reference_from_receipt(uc_ref), **({'cap_s': cap_s} if cap_s else {}))
    t = _metric(r)
    res = {'kind': 'F-HT', 'receipt_attempt': r.get('attempt_id'), 'valid': r.get('valid'),
           'numerics_class': (r.get('judge') or {}).get('numerics_class'), 'step_time_s': t,
           'speedup_vs_uc_ref': (t_ref / t) if t and r.get('valid') else None,
           'line': 'best known human recipe on this node set'}
    out = Path(out_dir)
    _write(out / 'ht.json', r)
    _write(out / 'ht_result.json', res)
    return {**res, 'receipt': r}


# ---------------------------------------------------------------- arms
def _prepare(arm: str, runner, uc_ref: dict, overrides: dict, *, code: bool = False):
    _uc_ref_value(uc_ref)                                  # the start receipt and reference must be the U+C reference
    task = T.make_task(arm, code=code)
    budget = task.search_budget(**{k: v for k, v in overrides.items() if v is not None})
    runner.reference_parsed = J.reference_from_receipt(uc_ref)
    return task, budget


def run_f_rnd(runner, *, run_id: str, out_dir, uc_ref: dict, seed: int = S.SEED, arm: str = 'F-RND', clock=None,
              **overrides) -> dict:
    from ...agent.search import SearchSession
    task, budget = _prepare(arm, runner, uc_ref, overrides)       # arm = 'F-RND' (window) or 'F-RND-20-s42/s43' (open)
    hooks = FlashNextHooks(task, start_receipt=uc_ref)
    s = SearchSession(task=task, budget=budget, runner=runner, judge=EnvJudge(env=T.ENV), run_id=run_id, arm=arm,
                      out_dir=out_dir, start_config=S.start_point(), start_receipt=uc_ref, reference_receipt=uc_ref,
                      layers=('L0',), hooks=hooks, driver_info={'kind': arm, 'llm': None, 'sampler': 'switch_space.RandomSwitchSampler'},
                      **({'clock': clock} if clock else {}))
    sampler = S.RandomSwitchSampler(seed)
    seen = {S.key(c.config) for c in s.candidates.values()}
    drawn, stop = [], None
    while stop is None:
        if s.stop_flag or s.ledger.exhausted():
            stop = s.stop_flag or 'budget:' + s.ledger.exhausted()[0]
            break
        p = sampler.next(seen)
        if p is None:
            stop = 'no_valid_sample'
            break
        seen.add(S.key(p))
        drawn.append(p)
        fb = s.propose_config(config=p, base='c0', origin='random')
        if 'candidate_id' in fb:
            s.screen(fb['candidate_id'])
    tr = s.finalize(stop_reason=stop, extra={'seed': seed, 'draws': sampler.draws, 'proposed': len(drawn),
                                             'sampler': 'ext.flashnext.switch_space.RandomSwitchSampler',
                                             'deviations': dv.for_arm(arm)}).to_dict()
    return _seal(tr, Path(out_dir) / 'trace.json')


def _f1_project_view(task):
    """A minimal project view of the Flash-Next job for experience.runs.plan_for (as ext.optico.arms.project_view is for
    an OptiCo workload): its fingerprint and the S_F1 switch table as knobs, so ours read the designated Flash-Next home's
    cross-job records and patches (the home excludes every Flash-Next record - D4). The switches carry no registry
    mechanism, so the knob-side coverage is 'none'; the value for ours is the transfer brief / patches the finalist reads
    from the home (finalist_base._fb_transfer_init, gated on the plan being present)."""
    from types import SimpleNamespace
    f = task.facts
    hw, lay = f['hardware'], f['fixed_layout']
    tv = S.task_view()
    knobs = {}
    for row in tv['switches']:
        k = {'type': 'str', 'default': tv['start'].get(row['name'])}
        if row.get('values'):
            k['choices'] = [str(v) for v in row['values']]
        knobs[row['name']] = k
    raw = {'hardware': {'gpu_type': hw.get('gpu_type', 'H100-80GB'), 'memory_gib': 80, 'interconnect': 'infiniband'},
           'framework': 'megatron', 'versions': {},
           'model': {'name': f['model']['name'], 'params_b': 125.0, 'family': 'moe'},
           'workload': {'seq_len': lay['seq_len']}, 'search': {}, 'states': {}}
    return SimpleNamespace(raw=raw, gpus=hw.get('gpus_per_node', 8), nodes=hw.get('nodes', 4), knobs=knobs,
                           name='flashnext-f1', window=list(P.WINDOW), reference_config=S.start_point)


def run_f_is2(strategy: str, runner, *, run_id: str, out_dir, uc_ref: dict, arm: str | None = None,
              screens: int | None = None,
              seed: int = S.SEED, code: bool = False, experience=None, backend=None, log=None, checkpoint: bool = False,
              resume: bool = False, **overrides) -> dict:
    """An open-wave search strategy on the F1 task (section 6 F8), modelled on ext.optico.arms.run_b_is2: search.run
    (strategy, ctx) with FlashNextHooks on S_F1 (the switch table is the task view) and, when code=True, the C_F1 code
    space open through validate_patch (layers L0 + L1, deviation D-F1-7). Every arm starts from U+C with the U+C
    reference screen as its start receipt (not re-screened), the same driver (Opus 5.5 CLI), judge, start and space as
    the other F1 arms. `strategy` is the registered search strategy to run - the open trees register `bcomp` (pc_code_o1),
    `autoresearch` (pc_code_ar3) and `rebench` (pc_code_rb2), the v4 trees `bcomp` (pc_code_o2) for the v4 extras (arm F-OPEN4-BCOMP: their task carries the v4 caps); `arm` is
    the EnvTask arm (default: STRATEGY_ARM[strategy]).
      experience: for the proposed-method arms (bcomp) the CLI passes 'auto' -> build the F1 cross-job plan from
      the arm's INFRARSI_HOME (the Flash-Next home, Flash-Next records excluded; driver=None, so no HLS transfer call -
      the finalist reads the transfer brief from the home itself); a dict is used as given; None (AR / RB / OptiCo /
      random) reads no experience.
      Fidelity: the F1 runner's run_limit is 30 and there is no probe or short fidelity, so multi_fidelity is off - the
      engine runs every candidate at full fidelity (its ('probe','full') plan reduces to ('full',)) and the finalists'
      FinalistLadder (short=False) never schedules or charges a low-fidelity run; every patch proves itself in a full
      screen (as OptiCo's B-IS2 has no shorter run). (The ladder's lowfi_share is the finalists' constructor default; it
      is never consumed here. A literal zero would require a change in the open trees' finalist_base, which is not ours.)
      screens: None = the arm's Q5 entry (20 for the open-wave arms and the v4 extras, 105 for the same-money extras).
      checkpoint / resume (the same-money extras; trees with search-state checkpoints, v4.1 on): checkpoint=True gives the
      engine the tree's Writer on <out_dir>/state (after a paid proposal, after an observed run that spent something, at the
      end); resume=True continues the search from that checkpoint, in the same out_dir (see _resume_state / _resumer: the
      saved experience plan, the state onto the objects built here, the caps derived again, extra.resumes in the trace)."""
    from ... import search
    from ...agent.capability import infer
    arm = arm or STRATEGY_ARM.get(strategy, strategy)
    saved = _resume_state(out_dir, run_id=run_id, arm=arm, strategy=strategy, code=code, uc_ref=uc_ref) if resume else None
    task, budget = _prepare(arm, runner, uc_ref, {'screens': screens, **overrides}, code=code)
    if backend is None:
        from ...agent.arms import strong_driver
        backend = strong_driver()
        if hasattr(backend, 'max_retries'):                # as the product's claude-cli driver (project.llm.make_driver)
            backend.max_retries = 1
    hooks = FlashNextHooks(task, start_receipt=uc_ref)     # hooks.code follows is_code_task(task) (code=True -> L0+L1)
    say = log or (lambda m: None)
    if saved is not None:                                  # a resume: the plan the search started with, as it had it
        experience = saved['flow'].get('experience')
    elif experience in (True, 'auto', 'build'):            # ours: build the cross-job plan from INFRARSI_HOME (the FN
        from ...experience import runs as exp_runs         # home); driver=None so no HLS transfer call - the finalist
        experience = exp_runs.plan_for(_f1_project_view(task), None, driver=None, log=say)   # reads the brief itself
    options = {'full_steps': P.UPDATES, 'warmup': P.WINDOW[0] - 1, 'short_steps': None, 'probe_steps': 2,
               'multi_fidelity': False, 'job_gpu_hours': None, 'capability': infer(getattr(backend, 'model', None)),
               'prior_results': [], 'run_noise': None, 'noise_pairs': [], 'experience': experience, 'log': say,
               'ablate': [], 'accounting': None, 'arm': arm, 'count_rescores': True}
    applied = []
    if checkpoint or resume:
        from ...agent import state as St
        from ...search import checkpoint as ckpt
        resumes = list(saved['flow'].get('resumes') or []) if saved is not None else []
        flow = {'run_id': run_id, 'arm': arm, 'strategy': strategy, 'code': bool(code), 'seed': seed,
                'uc_ref_attempt': uc_ref.get('attempt_id'), 'flashnext_sha256': _fn_sha256(), 'resumes': resumes,
                'experience': ({k: v for k, v in experience.items() if not callable(v)} if isinstance(experience, dict)
                               else experience)}
        options['checkpoint'] = ckpt.Writer(Path(out_dir) / St.STATE_DIR, flow=lambda: flow)
        if saved is not None:
            options['resume'] = _resumer(saved, Path(out_dir), resumes, applied, say)
    prev = _with_deviation(runner, 'D-F1-7') if code else None
    try:
        ctx = search.Context(task=task, budget=budget, start_config=S.start_point(), runner=runner,
                             judge=EnvJudge(env=T.ENV), hooks=hooks, run_id=run_id, out_dir=Path(out_dir),
                             start_receipt=uc_ref, reference_receipt=uc_ref, driver=backend, seed=seed, options=options)
        trace = search.run(strategy, ctx)
    finally:
        if prev is not None:
            runner.deviations = prev
    if saved is not None and not applied:
        raise RuntimeError(f'{strategy} did not take the checkpoint: the run was not resumed (its trace is not trustworthy)')
    tr = trace.to_dict()
    if code:                                               # setting 2: record the code space and the selection's overlay
        tr.setdefault('extra', {})['code_space'] = code_block(tr)
    return _seal(tr, Path(out_dir) / 'trace.json')


def _fn_sha256() -> str:
    """sha256 over this ext/flashnext directory's files (as fn_launch's provenance line): the harness a checkpoint ran with."""
    import hashlib
    d, h = Path(__file__).resolve().parent, hashlib.sha256()
    for p in sorted(x for x in d.rglob('*') if x.is_file() and '__pycache__' not in x.parts):
        h.update(p.relative_to(d).as_posix().encode() + b'\0' + p.read_bytes() + b'\0')
    return h.hexdigest()


def _resume_state(out_dir, *, run_id: str, arm: str, strategy: str, code: bool, uc_ref: dict) -> dict:
    """The checkpoint in <out_dir>/state a resume continues from, checked: it exists and is readable, it is this run's
    (run id, arm, strategy, code flag, the U+C reference attempt it was started against: the restored candidates keep
    their receipts, so the live reference must be the same one), and the tree's package is the one it was made by
    (agent/state.package_hash; the trees are frozen while their cells run). {'data', 'path', 'flow'}; ValueError
    otherwise (nothing launched)."""
    from ...agent import state as St
    try:
        data, path = St.read(Path(out_dir) / St.STATE_DIR)
    except St.CheckpointError as exc:
        raise ValueError(f'nothing to resume: {exc}') from exc
    h, fl = data.get('header') or {}, data.get('flow') or {}
    want = {'run_id': run_id, 'arm': arm, 'strategy': strategy, 'code': bool(code)}
    got = {'run_id': fl.get('run_id'), 'arm': fl.get('arm'), 'strategy': fl.get('strategy'), 'code': fl.get('code')}
    if got != want:
        raise ValueError(f'{path} is not this run\'s checkpoint: {got} (this run {want})')
    if fl.get('uc_ref_attempt') != (uc_ref or {}).get('attempt_id'):
        raise ValueError(f'{path} was started against the U+C reference attempt {fl.get("uc_ref_attempt")}, not '
                         f'{(uc_ref or {}).get("attempt_id")}: a resume keeps its reference')
    if h.get('package_hash') != St.package_hash():
        raise ValueError(f'{path} was made by another version of the tree (package hash {str(h.get("package_hash"))[:12]}, '
                         f'this tree {St.package_hash()[:12]})')
    return {'data': data, 'path': path, 'flow': fl}


def _resumer(saved: dict, out_dir: Path, resumes: list, applied: list, say):
    """options['resume'] of a resumed run_f_is2 (as project/resume.Resume.apply): the saved state onto the session, engine
    and strategy this call built, the session's directory set back (it was built without one, so building it wrote nothing
    to the events it continues), the caps derived again from the budget and the task (strategy.rebudget), the entry made in
    `resumes` (the flow's list, so the next checkpoints keep it; the trace's extra.resumes) with the session's model calls
    and screens at the restore (calls_before / screens_before: what a classifier of this invocation skips)."""
    def apply(session, engine, strategy) -> None:
        from ...agent import state as St
        from ...search import checkpoint as ckpt
        if not hasattr(strategy, 'rebudget'):
            raise ValueError(f'the strategy {getattr(strategy, "name", "?")} cannot be resumed')
        state = St.state_of_file(saved['data'], ckpt.live_objects(session, engine, strategy))
        session.out_dir = out_dir
        ckpt.restore(state, session, engine, strategy)
        strategy.rebudget(engine)
        h, fl = saved['data'].get('header') or {}, saved['flow']
        resumes.append({'utc': _dt.datetime.now(_dt.timezone.utc).isoformat(), 'from': 'checkpoint',
                        'from_checkpoint': {'file': saved['path'].name, 'saved_utc': h.get('saved_utc'), 'why': h.get('why'),
                                            'sequence': h.get('sequence'), 'incumbent': h.get('incumbent')},
                        'screens_done': h.get('screens_done'), 'used_before': h.get('budget_used'),
                        'stop_before': h.get('stop'), 'budget_after': session.budget.as_dict(),
                        'calls_before': len(session.llm_calls), 'screens_before': len(session.screens),
                        'flashnext_changed': fl.get('flashnext_sha256') != _fn_sha256()})
        engine.resumes = resumes
        applied.append(True)
        say(f'[flashnext] resumed {fl.get("run_id")} from {saved["path"].name} ({h.get("why")}, {h.get("screens_done")} '
            f'screens done, incumbent {h.get("incumbent")})')
    return apply


def run_f_a0(runner, *, run_id: str, out_dir, uc_ref: dict, package=None, backend=None, clock=None, **overrides) -> dict:
    from ...agent.optimize import optimize
    task, budget = _prepare('F-A0', runner, uc_ref, overrides)
    if package is None:
        from ...agent.arms import a0_package
        package = a0_package()
    if backend is None:
        from ...agent.arms import strong_driver
        backend = strong_driver()
    hooks = FlashNextHooks(task, start_receipt=uc_ref, frozen_package_digest=package.digest)
    trace = optimize(task, package, backend, budget, S.start_point(), runner, judge=EnvJudge(env=T.ENV), run_id=run_id,
                     arm='F-A0', out_dir=out_dir, mode='strong', start_receipt=uc_ref, reference_receipt=uc_ref,
                     use_verifier=True, use_skills=True, layers=('L0',), hooks=hooks, **({'clock': clock} if clock else {}))
    return _seal(trace.to_dict(), Path(out_dir) / 'trace.json')


def code_block(trace: dict) -> dict:
    """extra.code_space of a code-arm trace: the space, and the selection's overlay (digest = its candidate's
    overlay_digest, per-file sha256) or None when the selection has no code edit."""
    sel = trace.get('selected_overlay') or {}
    rec = C.record(sel) if sel else None
    cand = next((c for c in trace.get('candidates') or [] if c.get('id') == trace.get('selected')), {})
    return {'space_id': C.SPACE_ID, 'sha256': C.sha256(), 'mutable_files': list(C.MUTABLE_PATHS),
            'max_patch_lines': C.MAX_PATCH_LINES, 'selected_overlay': rec, 'selected_overlay_digest': cand.get('overlay_digest'),
            'candidates_with_code': sum(1 for c in trace.get('candidates') or [] if c.get('overlay_files'))}


def run_f_a0_code(runner, *, run_id: str, out_dir, uc_ref: dict, package=None, backend=None, clock=None,
                  **overrides) -> dict:
    """F-A0-CODE (setting 2): run_f_a0 with the code space open - the same package (A0), driver, verifier, skills, judge,
    start receipt and budget; candidates are S_F1 points plus overlays of C_F1 (layers L0 + L1)."""
    from ...agent.optimize import optimize
    task, budget = _prepare('F-A0-CODE', runner, uc_ref, overrides)
    if package is None:
        from ...agent.arms import a0_package
        package = a0_package()
    if backend is None:
        from ...agent.arms import strong_driver
        backend = strong_driver()
    hooks = FlashNextHooks(task, start_receipt=uc_ref, frozen_package_digest=package.digest)
    prev = _with_deviation(runner, 'D-F1-7')
    try:
        trace = optimize(task, package, backend, budget, S.start_point(), runner, judge=EnvJudge(env=T.ENV),
                         run_id=run_id, arm='F-A0-CODE', out_dir=out_dir, mode='strong', start_receipt=uc_ref,
                         reference_receipt=uc_ref, use_verifier=True, use_skills=True, layers=('L0', 'L1'), hooks=hooks,
                         **({'clock': clock} if clock else {}))
    finally:
        runner.deviations = prev
    tr = trace.to_dict()
    tr.setdefault('extra', {})['code_space'] = code_block(tr)
    return _seal(tr, Path(out_dir) / 'trace.json')


def _with_deviation(runner, dev: str):
    prev = tuple(getattr(runner, 'deviations', ()) or ())
    runner.deviations = tuple(sorted(set(prev) | {dev}))
    return prev


def run_f_optico(runner, *, run_id: str, out_dir, uc_ref: dict, arm: str = 'F-OPTICO', backend=None, optico_root=None,
                 clock=None, **overrides) -> dict:
    from ...baselines.optico_style import optico as oc
    task, budget = _prepare(arm, runner, uc_ref, overrides)       # arm = 'F-OPTICO' (window) or 'F-OPTICO-20' (open wave)
    if backend is None:
        from ...agent.arms import strong_driver
        backend = strong_driver()
    hooks = FlashNextHooks(task, start_receipt=uc_ref)
    prev = _with_deviation(runner, 'D-OC-15')
    try:
        trace = oc.run_optico_style(task, budget, S.start_point(), runner, llm_backend=backend, judge=EnvJudge(env=T.ENV),
                                    run_id=run_id, out_dir=out_dir, start_receipt=uc_ref, reference_receipt=uc_ref,
                                    clock=clock, optico_root=optico_root or oc.OPTICO_ROOT, hooks=hooks, arm=arm)
    finally:
        runner.deviations = prev
    return _seal(trace.to_dict(), Path(out_dir) / 'trace.json')


def run_f_prompts(runner, *, run_id: str, out_dir, uc_ref: dict, backend=None, clock=None, **overrides) -> dict:
    from ...baselines.prompts_oneshot import run_prompts
    task, budget = _prepare('F-PROMPTS', runner, uc_ref, overrides)
    if backend is None:
        from ...agent.arms import strong_driver
        backend = strong_driver()
    hooks = FlashNextHooks(task, start_receipt=uc_ref)
    trace = run_prompts(task, budget, S.start_point(), runner, llm_backend=backend, judge=EnvJudge(env=T.ENV),
                        run_id=run_id, out_dir=out_dir, start_receipt=uc_ref, reference_receipt=uc_ref, clock=clock,
                        hooks=hooks)
    return _seal(trace.to_dict(), Path(out_dir) / 'trace.json')


ARM_FUNCS = {'F-PROMPTS': run_f_prompts, 'F-OPTICO': run_f_optico, 'F-A0': run_f_a0, 'F-RND': run_f_rnd,
             'F-A0-CODE': run_f_a0_code}


# ---------------------------------------------------------------- locked pairs (Q2)
def selection_from(trace) -> tuple[dict, dict | None, str]:
    """(final S_F1 point, its own search-screen receipt or None for the start, candidate id) from a trace dict or the
    path of a trace.json (the receipt is read from the run's receipts/<cid>.json)."""
    path = Path(trace) if isinstance(trace, (str, Path)) else None
    tr = _load(trace)
    cid = tr.get('selected')
    point, errors, _ = S.validate(tr.get('selected_config') or {})
    if errors:
        raise ValueError(f'the trace selection is not an S_F1 point: {errors}')
    if point == S.start_point() and not tr.get('selected_overlay'):     # U+C itself (a code edit on U+C is not U+C)
        return point, None, cid or 'c0'
    rec = None
    base = path.parent if path else (Path(tr['out_dir']) if tr.get('out_dir') else None)
    row = next((r for r in tr.get('screens') or [] if r.get('candidate_id') == cid), {})
    if base is not None and row.get('collection') == 'recollected_readonly':      # EXT_009: the superseding record
        _, records, _ = SU.index(load_receipts([base]))                       # admitted records only
        r = records.get(row.get('attempt_id'))
        if r is not None and r.get('collection') == 'recollected_readonly' and SU.digest(r) == row.get('record_digest'):
            rec = {k: v for k, v in r.items() if k != '_path'}
    elif base is not None and (base / 'receipts' / f'{cid}.json').exists():
        rec = _load(base / 'receipts' / f'{cid}.json')
    return point, rec, cid


def own_screen_problems(own, cid: str, point: dict) -> list[str]:
    """The selection's own search screen must be authentic before its numerics anchor locked_self_mismatch: signed, a
    valid flashnext screen of exactly this candidate and point, with updates 1..30 in its judge block."""
    if not isinstance(own, dict):
        return ['no receipt']
    p = []
    if not signing.verify(own):
        p.append('harness signature missing or invalid')
    if own.get('env') != T.ENV or own.get('valid') is not True or own.get('judge_valid') is not True:
        p.append('not a valid flashnext screen')
    if own.get('purpose') != 'screen' or own.get('candidate_id') != cid:
        p.append(f'not the search screen of {cid} ({own.get("purpose")!r}, {own.get("candidate_id")!r})')
    if S.validate(own.get('point') or {})[0] != point:
        p.append('its point is not the selection')
    steps = ((own.get('judge') or {}).get('details') or {}).get('per_step')
    if not isinstance(steps, list) or [x.get('update') if isinstance(x, dict) else None for x in steps] != list(range(1, P.UPDATES + 1)):
        p.append('its judge block does not carry updates 1..30')
    return p


def selection_overlay(arm: str, trace) -> dict | None:
    """The code arm's selected overlay (the trace's selected_overlay, re-validated by code_space); None for a
    switch-only arm or a selection without a code edit."""
    if not _code_arm(arm, trace):
        return None
    ov = _load(trace).get('selected_overlay') or {}
    if not ov:
        return None
    errors, _ = C.validate(ov)
    if errors:
        raise ValueError(f'{arm}: the selected overlay is not a code edit of {C.SPACE_ID}: {errors[:3]}')
    return ov


def overlay_problems(receipt, overlay: dict | None) -> list[str]:
    """A code-arm receipt ran exactly this overlay (its code_overlay digest), or none when overlay is None."""
    if not isinstance(receipt, dict):
        return ['no receipt']
    got = (receipt.get('code_overlay') or {}).get('digest') if isinstance(receipt.get('code_overlay'), dict) else None
    want = (C.record(overlay) or {}).get('digest') if overlay else None
    return [] if got == want else [f'its code overlay {str(got)[:12]} is not the selection\'s {str(want)[:12]}']


def pair_order(arm: str) -> list[str]:
    """Q2: even arm index (F-PROMPTS 0, F-A0 2; F-A0-CODE 4) runs the U+C half first, odd (F-OPTICO 1, F-RND 3) the
    candidate."""
    return ['uc', 'candidate'] if ALL_ARM_INDEX[arm] % 2 == 0 else ['candidate', 'uc']


def locked_numbers(uc: dict, cand: dict, own: dict | None, attempt: int) -> tuple[dict, GuardResult]:
    """A locked pair's status and numbers from the receipts of its two halves (and, for the self check, the selection's
    own search screen): run_locked writes them, formal_problems re-derives them from the signed receipts."""
    guards = GuardResult()
    guards.extend(G.void_pair(uc))
    void = not guards.ok
    own_steps = ((own.get('judge') or {}).get('details') or {}).get('per_step') if isinstance(own, dict) else None
    cand_steps = ((cand.get('judge') or {}).get('details') or {}).get('per_step')
    self_g = G.locked_self_mismatch(cand_steps, own_steps) if cand.get('valid') else GuardResult()
    guards.extend(self_g)
    cand_ok = cand.get('valid') is True and self_g.ok
    t_pair, t_cand = _metric(uc), _metric(cand)
    out = {'status': ('void_twice' if attempt >= 2 else 'void') if void else 'ok',
           'cost': {'gpu_seconds': sum(((r.get('cost') or {}).get('gpu_seconds') or 0.0) for r in (uc, cand)),
                    'wall_seconds': sum(((r.get('cost') or {}).get('wall_seconds') or 0.0) for r in (uc, cand))},
           'guard_rejections': guards.rejections, 'guard_reports': guards.reports,
           'numerics_class': (cand.get('judge') or {}).get('numerics_class') if cand_ok else None}
    if void:
        out.update(t_uc_pair=None, t_locked=None, locked_valid=None, invalid_accept=None, speedup_vs_uc=None, y_pair=None)
    elif cand_ok and t_pair and t_cand:
        out.update(t_uc_pair=t_pair, t_locked=t_cand, locked_valid=True, invalid_accept=0, speedup_vs_uc=t_pair / t_cand,
                   y_pair=math.log(t_pair / t_cand))
    else:
        out.update(t_uc_pair=t_pair, t_locked=None, locked_valid=False, invalid_accept=1, speedup_vs_uc=1.0, y_pair=0.0)
    return out, self_g


LOCKED_FIELDS = ('status', 't_uc_pair', 't_locked', 'locked_valid', 'invalid_accept', 'speedup_vs_uc', 'y_pair',
                 'numerics_class', 'cost')


def half_id(who: str, attempt: int) -> str:
    return f'locked-{who}' + (f'-a{attempt}' if attempt > 1 else '')


def run_locked(runner, *, arm: str, run_id: str, out_dir, trace, uc_ref: dict, attempt: int = 1,
               own_receipt: dict | None = None) -> dict:
    """The plan-A locked pair of one arm, back to back on the same nodes; attempt 2 = the one re-run of a void attempt 1
    (same arm, same selection) at the end of the window. Writes locked.json (locked_attempt2.json)."""
    if arm not in ALL_ARM_INDEX:
        raise ValueError(f'unknown F1 arm {arm!r}')
    if attempt not in (1, 2):
        raise ValueError('a locked pair runs once; only a void pair is re-run, once (attempt 2)')
    point, own, cid = selection_from(trace)
    own = own_receipt if own_receipt is not None else own
    code = _code_arm(arm, trace)                       # setting 2 (F-A0-CODE) or an open arm whose trace ran with C_F1
    overlay = selection_overlay(arm, trace)            # the code arm's selected code edit (None: none / switch-only arm)
    ov_rec = C.record(overlay) if overlay else None
    idx = ALL_ARM_INDEX[arm]
    order = pair_order(arm)
    t_ref = _uc_ref_value(uc_ref)
    out = Path(out_dir)
    name = 'locked.json' if attempt == 1 else 'locked_attempt2.json'
    if (out / name).exists():
        raise ValueError(f'{out / name} exists: this locked attempt already ran')
    if attempt == 2:
        first = _load(out / 'locked.json') if (out / 'locked.json').exists() else None
        if not (isinstance(first, dict) and first.get('arm') == arm and first.get('status') == 'void'
                and first.get('point') == point
                and (not code or ((first.get('code_overlay') or {}).get('digest') == (ov_rec or {}).get('digest')))):
            raise ValueError('attempt 2 re-runs a void attempt 1 of the same arm and selection; none found')
    devs = list(dv.for_arm(arm))                       # an open code arm adds D-F1-7 per-run (not in dv.ARM_EXTRA)
    if code and 'D-F1-7' not in devs:
        devs.append('D-F1-7')
    base = {'schema': 'flashnext-locked-v1', 'arm': arm, 'arm_index': idx, 'attempt': attempt, 'selected': cid,
            'point': point, 'run_id': run_id, 'uc_ref_attempt': uc_ref.get('attempt_id'), 't_uc_ref': t_ref,
            'deviations': devs, 'lifecycle': dv.SCREEN_LEVEL_LABEL}
    if code:                                           # setting 2: the selection's code edit is part of what is locked
        base['code_overlay'] = ov_rec
    if point == S.start_point() and not overlay:
        res = {**base, 'locked_run': False, 'status': 'no_improvement', 'order': [],
               'reason': 'the final selection is U+C itself: no locked run (Q2, no A/A block); locked value = the U+C '
                         'reference', 't_uc_pair': t_ref, 't_locked': t_ref, 'locked_valid': True, 'invalid_accept': 0,
               'speedup_vs_uc': 1.0, 'y_pair': 0.0, 'numerics_class': None, 'cost': {'gpu_seconds': 0.0, 'wall_seconds': 0.0}}
        return _seal(res, out / name)
    problems = own_screen_problems(own, cid, point)
    if code:
        problems += overlay_problems(own, overlay)
    if problems:
        raise ValueError(f'{arm}: the search screen of the selection {cid} cannot anchor the locked pair: {problems}')
    task = T.make_task(arm, code=code)
    reference = J.reference_from_receipt(uc_ref)
    prev = (_with_deviation(runner, 'D-OC-15') if arm in ('F-OPTICO', 'F-OPTICO-20') else
            _with_deviation(runner, 'D-F1-7') if code else None)
    got = {}
    try:
        for who in order:
            p = S.start_point() if who == 'uc' else point
            got[who] = runner.run(task, p, overlay if (who == 'candidate' and overlay) else None, 'locked', run_id,
                                  role='candidate', reference_parsed=reference, candidate_id=half_id(who, attempt))
    finally:
        if prev is not None:
            runner.deviations = prev
    uc, cand = got['uc'], got['candidate']
    nums, self_g = locked_numbers(uc, cand, own, attempt)
    t_pair, t_cand = _metric(uc), _metric(cand)
    res = {**base, 'locked_run': True, 'order': order, **nums,
           'uc_half': {'attempt_id': uc.get('attempt_id'), 'valid': uc.get('valid'), 'step_time_s': t_pair,
                       'numerics_class': (uc.get('judge') or {}).get('numerics_class'),
                       'invalid_reasons': (uc.get('invalid_reasons') or [])[:4]},
           'candidate_half': {'attempt_id': cand.get('attempt_id'), 'valid': cand.get('valid'), 'step_time_s': t_cand,
                              'numerics_class': (cand.get('judge') or {}).get('numerics_class'),
                              'self_match': self_g.reports[0] if self_g.reports else None,
                              'invalid_reasons': (cand.get('invalid_reasons') or [])[:4]},
           'own_screen_attempt': (own or {}).get('attempt_id'), 'own_screen_digest': SU.digest(own) if own else None,
           'own_screen_collection': (own or {}).get('collection', 'collected'), 'nodes': uc.get('nodes')}
    if code:                                           # which code each half ran (the U+C half: the frozen runtime)
        for half, r in (('uc_half', uc), ('candidate_half', cand)):
            res[half]['code_overlay_digest'] = ((r.get('code_overlay') or {}).get('digest')
                                                if isinstance(r.get('code_overlay'), dict) else None)
        res['code_overlay_bound'] = not overlay_problems(cand, overlay) and not overlay_problems(uc, None)
    if res['status'] != 'ok':
        res['note'] = ('void pair: re-run once at the end of the window (run_locked attempt=2)' if attempt < 2 else
                       'second void pair: the arm is reported without a locked value (label invalid)')
    elif not res['locked_valid']:
        res['note'] = 'the final selection is invalid when locked: the arm falls back to U+C (y = 0), InvalidAccept 1'
    return _seal(res, out / name)


def retrace(run_dir) -> dict:
    """EXT_009 mechanics (the harness, no hand edit): the one-shot arm's trace re-derived from its receipts including the
    admitted superseding records (supersede.rederive_trace) and re-sealed; the sealed as-run trace is kept, unchanged, as
    trace.as_run.json and is always the source (idempotent). Refused for a multi-step arm or any record problem."""
    d = Path(run_dir)
    as_run = d / 'trace.as_run.json'
    src = _load(as_run if as_run.exists() else d / 'trace.json')
    if not signing.verify(src):
        raise ValueError(f'{d}: the as-run trace is not sealed')
    originals, records, problems = SU.index(load_receipts([d]))
    if problems:
        raise ValueError(f'{d}: superseding records not admitted: {problems}')
    missing = SU.undisposed(originals, records, arms=[src.get('arm')])
    if missing:
        raise ValueError(f'{d}: collection losses without their EXT_009 record (supersede or unrecoverable): {missing}')
    new = SU.rederive_trace(src, records)
    if not as_run.exists():
        _write(as_run, src)
    return _seal(new, d / 'trace.json')


def final_locked(locked_results) -> dict | None:
    """The locked result that counts for an arm: attempt 1, or attempt 2 when it re-ran a void attempt 1 of the same arm
    and selection. Any other sequence (attempt 2 alone, a non-void attempt 1 before it, attempts beyond 2, duplicates)
    raises."""
    rows = [_load(x) for x in locked_results if x is not None]
    rows = [r for r in rows if isinstance(r, dict)]
    if not rows:
        return None
    by = {}
    for r in rows:
        a = r.get('attempt')
        if a not in (1, 2) or a in by:
            raise ValueError(f'locked attempts {[x.get("attempt") for x in rows]}: only 1, or 1 then 2')
        by[a] = r
    if 2 in by:
        first, second = by.get(1), by[2]
        if not (first and first.get('status') == 'void' and first.get('arm') == second.get('arm')
                and first.get('point') == second.get('point')):
            raise ValueError('locked attempt 2 without a void attempt 1 of the same arm and selection')
        return second
    return by[1]


# ---------------------------------------------------------------- margin_F1.json (Q5)
def load_receipts(paths) -> list[dict]:
    out, seen = [], set()
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(x for x in p.rglob('*.json') if x.parent.name == 'receipts' or x.name in ('uc_ref.json', 'ht.json'))
        elif p.is_file():
            files.append(p)
    for f in files:
        try:
            r = json.loads(f.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if isinstance(r, dict) and r.get('schema') == 'ext-receipt-v1' and r.get('env') == T.ENV:
            key = json.dumps(r, sort_keys=True, default=str)       # identical copies (c0.json, uc_ref.json) once;
            if key not in seen:                                    # differing copies of one attempt are all kept
                seen.add(key)
                out.append({**r, '_path': str(f)})
    return out


def s_uc_kind(r: dict) -> str | None:
    """S_UC membership by run kind only: the U+C reference screen and the U+C half of every locked pair."""
    if r.get('task_id') != T.TASK_ID:
        return None
    if r.get('arm') == 'UC-REF' and r.get('purpose') == 'reference' and r.get('candidate_id') == 'uc-ref':
        return 'uc_reference'
    if r.get('arm') in ALL_ARM_INDEX and r.get('purpose') == 'locked' and str(r.get('candidate_id', '')).startswith('locked-uc'):
        return 'locked_uc_half'                        # the code arm's U+C half runs U+C without code edits
    return None


def window_rsd(receipt: dict) -> float | None:
    times = ((receipt.get('judge') or {}).get('details') or {}).get('step_max_rank_s') or []
    a, b = P.WINDOW
    win = [t for i, t in enumerate(times, 1) if a <= i <= b and P.finite(t)]
    if len(win) < 2:
        return None
    return statistics.stdev(win) / statistics.fmean(win)


def margin(receipts) -> dict:
    members, excluded, anomalies, seen = [], [], [], {}
    for r in receipts:
        kind = s_uc_kind(r)
        if kind is None:
            continue
        clean = {k: v for k, v in r.items() if k != '_path'}
        row = {'attempt_id': r.get('attempt_id'), 'kind': kind, 'arm': r.get('arm'), 'run_id': r.get('run_id'),
               'T': _metric(r), 'path': r.get('_path')}
        if r.get('attempt_id') in seen:                 # one run counts once (its copies: c0.json, uc_ref.json, ...)
            if seen[r.get('attempt_id')] != clean:
                anomalies.append({**row, 'why': 'a second, different receipt for the same attempt'})
            continue
        seen[r.get('attempt_id')] = clean
        if not signing.verify(clean):
            excluded.append({**row, 'why': 'harness signature missing or invalid'})
        elif r.get('arm') in T.EXTRA_ARMS:              # post hoc: never moves the pre-registered window's margin
            excluded.append({**row, 'why': ('a v4 extra' if r.get('arm') in T.V4_ARMS else 'a same-money extra')
                             + ' (post hoc): outside S_UC'})
        elif r.get('valid') is not True or r.get('judge_valid') is not True:
            excluded.append({**row, 'why': 'not valid (judge or guards)'})
        elif _metric(r) is None or S.validate(r.get('point') or {})[0] != S.start_point() or r.get('code_overlay'):
            # a signed valid receipt of this run kind that is not a U+C measurement: an upstream inconsistency, shown
            anomalies.append({**row, 'why': 'run kind of a U+C half but not a U+C measurement (point or metric)'})
        else:
            members.append({**row, 'window_rsd': window_rsd(r)})
    vals = [m['T'] for m in members]
    n = len(vals)
    mean = statistics.fmean(vals) if n else None
    s = statistics.stdev(vals) if n >= 2 else None
    rsd = (s / mean) if s is not None and mean else None
    rec = {'schema': 'flashnext-margin-v1', 'task_id': T.TASK_ID,
           'rule': 'AMENDMENT_EXT_003 Q5: S_UC = the U+C reference screen + the U+C half of every locked pair; margin = '
                   'max(0.03, 3 x RSD(S_UC)), sample SD (n-1); n < 4 -> per arm max(0.03, 3 x RSD_window) of the '
                   "arm's own locked U+C half; membership by run kind only",
           'S_UC': members, 'excluded': excluded, 'anomalies': anomalies, 'n': n, 'mean': mean, 's': s, 'rsd': rsd,
           'diagnostic_within_window_rsd': {m['attempt_id']: m['window_rsd'] for m in members}}
    if n >= MIN_N:
        rec.update(estimator='run_to_run', margin=max(FLOOR, K * rsd), rsd_used=rsd, per_arm_margin=None)
    else:
        per_arm = {}
        for m in members:
            if m['kind'] == 'locked_uc_half' and m['window_rsd'] is not None:
                per_arm[m['arm']] = max(per_arm.get(m['arm'], 0.0), max(FLOOR, K * m['window_rsd']))
        rec.update(estimator='fallback_within_window', margin=None, rsd_used=None, per_arm_margin=per_arm,
                   note='n < 4: each comparison uses the larger of the two arms\' fallback margins (an arm without a '
                        'locked U+C half uses the floor 0.03)')
    return rec


def margin_for(mrec: dict, *arms) -> float:
    if mrec.get('margin') is not None:
        return float(mrec['margin'])
    per = mrec.get('per_arm_margin') or {}
    return max([per.get(a, FLOOR) for a in arms] + [FLOOR])


# ---------------------------------------------------------------- per-arm rows and labels (Q5)
def arm_row(arm: str, trace: dict, locked: dict | None, uc_ref: dict, ht: dict | None) -> dict:
    task = T.make_task(arm)
    t_ref = _uc_ref_value(uc_ref)
    t_ht = _metric(ht) if isinstance(ht, dict) and ht.get('valid') else None
    status = (locked or {}).get('status')
    no_value = locked is None or status in ('void', 'void_twice')
    lv = bool((locked or {}).get('locked_valid'))
    t_locked = (locked or {}).get('t_locked') if lv else None
    y_ref = (math.log(t_ref / t_locked) if t_locked else 0.0) if not no_value else None
    row = TJ.arm_row(trace, task, t_ref, locked=t_locked if lv else None, locked_valid=lv if not no_value else None,
                     invalid_accept_locked=int((locked or {}).get('invalid_accept') or 0))
    t_pair = (locked or {}).get('t_uc_pair')
    y_pair = (locked or {}).get('y_pair')
    denom = math.log(t_ref / t_ht) if t_ht else None
    row.update(arm=arm, arm_index=ALL_ARM_INDEX[arm], locked_status=status if locked else 'not_run',
               step_time_locked=t_locked, t_uc_pair=t_pair, speedup_vs_UC=(locked or {}).get('speedup_vs_uc'),
               y_pair=y_pair, y_ref=y_ref,
               recovered_fraction=(y_pair / denom) if (y_pair is not None and denom) else None,
               reached_1_30x=bool(((locked or {}).get('speedup_vs_uc') or 0) >= REACHED) if not no_value else None,
               numerics_class=(locked or {}).get('numerics_class'), InvalidAccept=(locked or {}).get('invalid_accept'),
               n_invalid_proposal=sum(int(v) for v in (trace.get('invalid_proposals') or {}).values()),
               drift_uc_pair_over_ref=(t_pair / t_ref) if t_pair else None,
               costs={'search': trace.get('used'), 'locked': (locked or {}).get('cost')},
               deviations=dv.for_arm(arm), screen_level=dv.SCREEN_LEVEL_LABEL)
    if _code_arm(arm, trace):                          # setting 2: never one table with the switch-only arms unmarked
        row.update(setting='code edits (setting 2): S_F1 switches + C_F1 code edits',
                   code_overlay=(locked or {}).get('code_overlay'))
    if arm in T.EXTRA_ARMS:                            # a labelled extra, never one of the pre-registered rows
        row['extra_arm'] = T.extra_label(arm)
    return row


def label(row_is: dict, row_other: dict, m: float) -> dict:
    """Q5 rule (trajectory.compare made exact): d = ln(t_locked(other) / t_locked(is)) on y against the U+C reference
    (an arm invalid when locked falls back to U+C, y = 0); advantage <=> is valid locked, InvalidAccept 0,
    d >= ln(1 + m), AUC_budget(is) >= AUC_budget(other); disadvantage <=> -d >= ln(1 + m); parity within the margin;
    no_advantage beyond it with another condition failing; invalid when an arm has no locked value (two void pairs)."""
    lm = math.log1p(m)
    if row_is.get('y_ref') is None or row_other.get('y_ref') is None:
        return {'label': 'invalid', 'margin': m, 'margin_y': lm, 'delta_y': None,
                'reason': 'an arm has no locked value (void pair twice or not run)'}
    d = row_is['y_ref'] - row_other['y_ref']
    eps = 1e-12
    cond = {'valid_locked': bool(row_is.get('locked_valid')), 'invalid_accept_zero': int(row_is.get('InvalidAccept') or 0) == 0,
            'beyond_margin': d >= lm - eps,
            'auc_not_worse': (row_is.get('AUC_budget') or 0.0) >= (row_other.get('AUC_budget') or 0.0)}
    if all(cond.values()):
        lab = 'advantage'
    elif -d >= lm - eps:
        lab = 'disadvantage'
    elif abs(d) < lm:
        lab = 'parity'
    else:
        lab = 'no_advantage'
    return {'label': lab, 'conditions': cond, 'delta_y': d, 'margin': m, 'margin_y': lm}


def prefix_best(trace: dict, n: int = 3) -> float | None:
    ok = [s['step_time'] for s in (trace.get('screens') or [])[:n] if s.get('valid') and P.finite(s.get('step_time'))]
    return min(ok) if ok else None


def locked_receipt_problems(arm: str, lk: dict, by_attempt: dict, trace=None, uc_ref: dict | None = None,
                            records: dict | None = None) -> list[str]:
    """A locked result checked against the window's signed receipts: it locks the arm's final selection; its halves are
    this arm's locked runs of U+C and of that point; the own search screen is authentic; and status, t_uc_pair,
    t_locked, locked_valid, invalid_accept, speedup, y_pair and numerics class re-derive from those receipts."""
    def signed(r):
        return isinstance(r, dict) and signing.verify({k: v for k, v in r.items() if k != '_path'})
    p = []
    try:
        sel, _, cid = selection_from(trace) if trace is not None else (None, None, None)
    except (ValueError, OSError, TypeError) as exc:
        return [f'{arm}: the trace has no usable selection ({exc})']
    if sel is not None and lk.get('point') != sel:
        p.append(f'{arm}: the locked point is not the arm\'s final selection')
    if not signing.verify(lk):
        p.append(f'{arm}: locked result (attempt {lk.get("attempt")}) not sealed by the controller (edited?)')
    if not lk.get('locked_run'):
        t_ref = _metric(uc_ref) if isinstance(uc_ref, dict) else None
        want = {'status': 'no_improvement', 't_uc_pair': t_ref, 't_locked': t_ref, 'locked_valid': True,
                'invalid_accept': 0, 'speedup_vs_uc': 1.0, 'y_pair': 0.0, 'numerics_class': None,
                'cost': {'gpu_seconds': 0.0, 'wall_seconds': 0.0}}
        if lk.get('point') != S.start_point() or any(lk.get(k) != v for k, v in want.items()) or lk.get('code_overlay'):
            p.append(f'{arm}: a locked result without a run must be the U+C selection with the U+C reference value')
        return p
    if _code_arm(arm, trace):                          # setting 2: the locked code edit is the selection's, run as such
        try:
            sel_ov = selection_overlay(arm, trace) if trace is not None else None
        except (ValueError, OSError, TypeError) as exc:
            sel_ov = None
            p.append(f'{arm}: the trace\'s selected overlay is not usable ({exc})')
        want_ov = (C.record(sel_ov) or {}).get('digest') if sel_ov else None
        if trace is not None and (lk.get('code_overlay') or {}).get('digest') != want_ov:
            p.append(f'{arm}: the locked code overlay is not the arm\'s final selection\'s')
        for half, ov in (('uc_half', None), ('candidate_half', sel_ov)):
            r = by_attempt.get((lk.get(half) or {}).get('attempt_id'))
            if isinstance(r, dict) and overlay_problems(r, ov):
                p.append(f'{arm}: {half} receipt: {overlay_problems(r, ov)[0]}')
        own_r = by_attempt.get(lk.get('own_screen_attempt'))
        if isinstance(own_r, dict) and overlay_problems(own_r, sel_ov):
            p.append(f'{arm}: own search screen: {overlay_problems(own_r, sel_ov)[0]}')
    if lk.get('order') != pair_order(arm):
        p.append(f'{arm}: locked order {lk.get("order")} is not the Q2 parity order {pair_order(arm)}')
    halves = {}
    attempt = lk.get('attempt') if lk.get('attempt') in (1, 2) else 1
    for half, who, point in (('uc_half', 'uc', S.start_point()), ('candidate_half', 'candidate', lk.get('point'))):
        att = (lk.get(half) or {}).get('attempt_id')
        r = by_attempt.get(att)
        if not signed(r):
            p.append(f'{arm}: {half} receipt {att} missing or not signed')
            continue
        if (r.get('arm') != arm or r.get('purpose') != 'locked' or r.get('task_id') != T.TASK_ID
                or S.validate(r.get('point') or {})[0] != point):
            p.append(f'{arm}: {half} receipt {att} is not this arm\'s locked run of its point')
        if not lk.get('run_id') or r.get('run_id') != lk.get('run_id') or r.get('candidate_id') != half_id(who, attempt):
            p.append(f'{arm}: {half} receipt {att} is not the {half_id(who, attempt)} run of locked run {lk.get("run_id")!r}')
        if bool(r.get('valid')) != bool((lk.get(half) or {}).get('valid')) or _metric(r) != (lk.get(half) or {}).get('step_time_s'):
            p.append(f'{arm}: {half} in locked.json differs from its receipt {att}')
        halves[half] = r
    own_att = lk.get('own_screen_attempt')
    rec = (records or {}).get(own_att)
    own = rec if (arm in SU.ONE_SHOT and rec is not None and rec.get('collection') == 'recollected_readonly') \
        else by_attempt.get(own_att)
    own_p = own_screen_problems({k: v for k, v in own.items() if k != '_path'} if isinstance(own, dict) else None,
                                cid if cid is not None else lk.get('selected'), lk.get('point'))
    if own_p or (own or {}).get('arm') != arm:
        p.append(f'{arm}: own search screen {lk.get("own_screen_attempt")} not in the window or not authentic: '
                 f'{own_p or ["another arm"]}')
    if isinstance(own, dict) and (lk.get('own_screen_digest') is not None or SU.is_record(own)) \
            and lk.get('own_screen_digest') != SU.digest(own):
        p.append(f'{arm}: the locked pair was anchored on another own-screen receipt than the window\'s '
                 f'{"superseding record" if SU.is_record(own) else "receipt"} {own_att}')
    if len(halves) == 2:
        by_who = {'uc': halves['uc_half'], 'candidate': halves['candidate_half']}
        first, second = (by_who[w] for w in pair_order(arm))
        t1s, t1e, t2s = _ts(first.get('utc_start')), _ts(first.get('utc_end')), _ts(second.get('utc_start'))
        if not (t1s and t1e and t2s and t1s <= t1e <= t2s):
            p.append(f'{arm}: the two halves did not run back to back in the order {pair_order(arm)} '
                     f'({first.get("utc_start")} .. {first.get("utc_end")}, then {second.get("utc_start")})')
        nums, _ = locked_numbers(halves['uc_half'], halves['candidate_half'], own if not own_p else None, attempt)
        bad = [k for k in LOCKED_FIELDS if lk.get(k) != nums.get(k)]
        if bad:
            p.append(f'{arm}: locked.json {bad} differ from what its receipts give')
    return p


def trace_problems(arm: str, trace, by_attempt: dict, have_receipts: bool = True, records: dict | None = None) -> list[str]:
    """A formal trace is the one the controller sealed after the search (or, for the one-shot arm, re-derived and
    re-sealed by retrace, EXT_009), and every screen in it is this arm's signed screen receipt (candidate, validity,
    metric, GPU cost) - for a superseded one-shot screen, its admitted record; no screen receipt of the arm in the window
    is missing from it."""
    records = records or {}
    if not isinstance(trace, dict):
        return [f'{arm}: no trace']
    p = []
    if not signing.verify(trace):
        p.append(f'{arm}: trace not sealed by the controller (edited after the search?)')
    if not have_receipts:
        return p
    seen = set()
    one_shot = arm in SU.ONE_SHOT
    for i, sc in enumerate(trace.get('screens') or []):
        att = sc.get('attempt_id')
        seen.add(att)
        rec = records.get(att)
        rec_ok = rec is not None and rec.get('collection') == 'recollected_readonly'
        if sc.get('collection') == 'recollected_readonly':
            if not one_shot or not rec_ok:
                p.append(f'{arm}: screen {i} ({sc.get("candidate_id")}) claims a re-collection without an admitted '
                         f'superseding record' + ('' if one_shot else ' (a multi-step arm keeps what it saw, EXT_009 Q4)'))
                continue
            if sc.get('record_digest') != SU.digest(rec):
                p.append(f'{arm}: screen {i} ({sc.get("candidate_id")}) applied another record than the admitted one')
            r = rec
        else:
            if one_shot and rec_ok:
                p.append(f'{arm}: screen {i} ({sc.get("candidate_id")}) has an admitted superseding record the trace does '
                         'not apply (retrace the one-shot arm, EXT_009 Q1)')
            r = by_attempt.get(att)
        ok = isinstance(r, dict) and signing.verify({k: x for k, x in r.items() if k != '_path'})
        if not ok or r.get('arm') != arm or r.get('purpose') != 'screen' or r.get('candidate_id') != sc.get('candidate_id'):
            p.append(f'{arm}: screen {i} ({sc.get("candidate_id")}, {att}) has no signed screen receipt of this arm')
            continue
        gpu = (r.get('cost') or {}).get('gpu_seconds')
        if (bool(r.get('valid')) != bool(sc.get('valid'))
                or (sc.get('valid') and _metric(r) != sc.get('step_time'))
                or not (P.finite(gpu) and P.finite(sc.get('gpu_hours')) and abs(gpu / 3600.0 - sc['gpu_hours']) <= 1e-9)):
            p.append(f'{arm}: screen {i} ({sc.get("candidate_id")}) differs from its receipt {att}')
    stray = sorted(str(a) for a, r in by_attempt.items()
                   if r.get('arm') == arm and r.get('purpose') == 'screen' and a not in seen)
    if stray:
        p.append(f'{arm}: screen receipts in the window that its trace does not list: {stray[:8]}')
    if one_shot:                                   # its selection is the best valid screen as finally recorded
        cands = [(trace.get('start_step_time'), -1, trace.get('start') or 'c0')] + [
            (sc.get('step_time'), sc.get('index'), sc.get('candidate_id')) for sc in trace.get('screens') or []
            if sc.get('valid') and P.finite(sc.get('step_time'))]
        cands = [c for c in cands if P.finite(c[0])]
        if cands and min(cands)[2] != trace.get('selected'):
            p.append(f'{arm}: selection {trace.get("selected")} is not the best valid screen {min(cands)[2]}')
    return p


def formal_problems(*, uc_ref: dict, ht: dict | None, arms: dict, receipts=None, margin_rec=None) -> list[str]:
    """What keeps a window from being the formal F1 result: a missing arm, reference or locked result, a void pair not
    re-run yet, a budget other than the Q5 table, a node set that changed within the window, a locked result that its
    receipts do not confirm, a margin file that the window's receipts do not reproduce."""
    p = [f'U+C reference: {x}' for x in G.uc_reference_problems(uc_ref)]
    p += [f'F-HT: {x}' for x in G.ht_reference_problems(ht)] if isinstance(ht, dict) else ['F-HT receipt missing']
    missing = sorted(set(ARM_INDEX) - set(arms))
    if missing:
        p.append(f'arms missing from the window: {missing}')
    by_attempt, records, idx_problems = SU.index(receipts or [])    # originals; admitted superseding records (EXT_009)
    p += idx_problems
    missing = SU.undisposed(by_attempt, records, arms=set(arms))
    if missing:
        p.append(f'collection losses (#37 class) without their EXT_009 record (supersede or unrecoverable): {missing}')
    for arm, v in arms.items():
        lk = v.get('locked')
        locked_history = [h for h in (v.get('locked_history') or ([lk] if lk else [])) if isinstance(h, dict)]
        if lk is None:
            p.append(f'{arm}: no locked result')
        elif lk.get('arm') != arm:
            p.append(f'{arm}: the locked result is arm {lk.get("arm")!r}')
        else:
            p += locked_receipt_problems(arm, lk, by_attempt, trace=v.get('trace'), uc_ref=uc_ref, records=records)
            if lk.get('attempt') == 2:              # the re-run is formal only after a void attempt 1 its receipts confirm
                first = next((h for h in locked_history if h.get('attempt') == 1), None)
                if first is None:
                    p.append(f'{arm}: locked attempt 2 without its attempt 1 in the window')
                else:
                    p += [f'(attempt 1) {x}' for x in locked_receipt_problems(arm, first, by_attempt,
                                                                              trace=v.get('trace'), uc_ref=uc_ref,
                                                                              records=records)]
        if receipts is not None:                    # every locked run of the arm belongs to one of its locked attempts
            used = {(h.get(k) or {}).get('attempt_id') for h in locked_history for k in ('uc_half', 'candidate_half')}
            stray = sorted(a for a, r in by_attempt.items()
                           if r.get('arm') == arm and r.get('purpose') == 'locked' and a not in used)
            if stray:
                p.append(f'{arm}: locked runs in the window that no locked result accounts for: {stray}')
        lk = v.get('locked')
        if lk and lk.get('status') == 'void':
            p.append(f'{arm}: void locked pair pending its re-run at the end of the window (run_locked attempt=2)')
        want = T.make_task(arm).search_budget().as_dict()
        got = (v.get('trace') or {}).get('budget')
        if got != want:
            p.append(f'{arm}: budget {got} is not the Q5 table {want}')
        p += trace_problems(arm, v.get('trace'), by_attempt, receipts is not None, records=records)
    sets = {'UC-REF': uc_ref.get('nodes')}
    if isinstance(ht, dict):
        sets['F-HT'] = ht.get('nodes')
    for arm, v in arms.items():
        if (v.get('locked') or {}).get('locked_run'):
            sets[f'{arm} locked'] = v['locked'].get('nodes')
    if len({json.dumps(sorted(x or [])) for x in sets.values()}) > 1 or any(not x for x in sets.values()):
        p.append(f'node sets differ within the window or are unrecorded: {sets}')
    if receipts is not None:
        want_nodes = sorted(uc_ref.get('nodes') or [])
        off = sorted(str(a) for a, r in by_attempt.items()
                     if not r.get('not_launched') and sorted(r.get('nodes') or []) != want_nodes)
        if off:
            p.append(f'launched screens on another node set than the U+C reference screen: {off[:8]}')
        m = margin(receipts)
        if uc_ref.get('attempt_id') not in [x['attempt_id'] for x in m['S_UC']]:
            p.append('the U+C reference screen is not in S_UC (margin computed from other receipts)')
        bad = [x['attempt_id'] for x in m['anomalies']] + [x['attempt_id'] for x in m['excluded'] if 'signature' in x['why']]
        if bad:
            p.append(f'S_UC receipts with a bad signature or not a U+C measurement: {bad}')
        if margin_rec is not None and _margin_key(margin_rec) != _margin_key(m):
            p.append('the margin file is not what the window\'s receipts give (members, estimator or margin differ)')
    else:
        p.append('no receipts: the margin and the locked values cannot be checked')
    return p


def _margin_key(m: dict) -> str:
    return json.dumps({'members': sorted((str(x.get('attempt_id')), x.get('T')) for x in m.get('S_UC') or []),
                       'estimator': m.get('estimator'), 'margin': m.get('margin'), 'per_arm': m.get('per_arm_margin')},
                      sort_keys=True, default=str)


def summarize(*, uc_ref: dict, ht: dict | None, arms: dict, receipts=None, margin_rec: dict | None = None,
              allow_nonformal: bool = False) -> dict:
    """arms: {arm: {'trace': trace dict, 'locked': final locked result or None, 'locked_history': every locked*.json
    row of the arm (optional; default [locked])}}; receipts: every signed receipt of the
    window (load_receipts over its run directories). The margin is recomputed from the receipts (a supplied margin file
    must match it). Refused while a void pair waits for its re-run, and when the window is not formal (all four arms, both
    references, Q5 budgets, one node set, locked values confirmed by their receipts) unless allow_nonformal."""
    problems = formal_problems(uc_ref=uc_ref, ht=ht, arms=arms, receipts=receipts, margin_rec=margin_rec)
    pending = [x for x in problems if 'pending its re-run' in x]
    if pending or (problems and not allow_nonformal):
        raise ValueError('summary refused: ' + '; '.join(problems))
    if receipts is not None:
        margin_rec = margin(receipts)
    elif margin_rec is None:
        raise ValueError('summary needs the window receipts (or, non-formal only, a margin record)')
    t_ref = _uc_ref_value(uc_ref)
    originals, records, _ = SU.index(receipts or [])
    rows = {a: arm_row(a, v['trace'], v.get('locked'), uc_ref, ht) for a, v in arms.items() if v.get('trace')}
    harness = {a: SU.harness_view(a, v['trace'], originals, records) for a, v in arms.items() if v.get('trace')}
    for a, row in rows.items():
        hv = harness[a]
        row['harness'] = {k: hv[k] for k in ('n_harness_lost', 'n_recovered', 'recovered', 'harness_cost_selection',
                                             'n_failed_by_class', 'comparison_withheld')}
        row['screens'] = hv['rows']
        lk = arms[a].get('locked') or {}
        if lk.get('point') is not None and lk.get('point') == ht_point():
            t_sel = (arms[a]['trace'] or {}).get('selected_step_time')
            t_ht = _metric(ht) if isinstance(ht, dict) and ht.get('valid') else None
            rel = (t_sel / t_ht - 1.0) if (t_sel and t_ht) else None
            row['recovered_fraction_reading'] = (
                'the selection is the registered S_match point, the F-HT point: recovered_fraction ~ 1 says the method '
                'found the best human recipe, not that it beat it'
                + (f' (its screen {t_sel:.3f} s vs F-HT {t_ht:.3f} s, {rel:+.1%}, '
                   f'{"under" if abs(rel) < FLOOR else "not under"} the {FLOOR:.0%} floor)' if rel is not None else ''))
    labels = {}
    for a, b in (('F-A0', 'F-OPTICO'), ('F-A0', 'F-RND'), ('F-OPTICO', 'F-RND')) + CODE_PAIRS:
        if a in rows and b in rows:
            lab = label(rows[a], rows[b], margin_for(margin_rec, a, b))
            withheld = [x for x in (a, b) if harness[x]['comparison_withheld']]
            if withheld:
                lab = {'label': NO_COMPARISON, 'computed': lab, 'withheld_by': withheld,
                       'reason': 'EXT_009 Q4: an arm with n_harness_lost >= 2 or harness_cost_selection'}
            labels[f'{a} vs {b}'] = lab
    out = {'schema': 'flashnext-summary-v1', 'task_id': T.TASK_ID, 't_uc_ref': t_ref,
           'uc_ref_attempt': uc_ref.get('attempt_id'), 'rows': rows, 'labels': labels,
           'margin': {k: margin_rec.get(k) for k in ('estimator', 'n', 'rsd', 'margin', 'per_arm_margin')},
           'switch_space': {'space_id': S.SPACE_ID, 'sha256': S.sha256(), 'admissible_points': len(S.points()),
                            'switches': list(S.SWITCHES)},
           'claims': T.CLAIMS, 'screen_level': dv.SCREEN_LEVEL_LABEL,
           'formal': not problems, 'nonformal_reasons': problems}
    code_arms = [a for a in rows if a in T.CODE_ARMS]
    if code_arms:                                      # setting 2 rows: their space and claims, marked as such
        out['code_space'] = {'space_id': C.SPACE_ID, 'sha256': C.sha256(), 'arms': code_arms, 'claims': T.CODE_CLAIMS,
                             'note': 'code-edit arms (setting 2) are compared with switch-only arms only as the value of '
                                     'code edits for the same method (F-A0-CODE vs F-A0); switch-only methods cannot edit '
                                     'code'}
    v4 = [a for a in rows if a in T.V4_ARMS]
    if v4:                                             # the v4 extras: marked rows, outside the window's counts
        out['v4_extras'] = {'arms': v4, 'label': T.V4_LABEL,
                            'note': 'post hoc: in no pair label, not in S_UC (the margin) and not in the drift\'s max / min'}
    bud = [a for a in rows if a in T.BUD_ARMS]
    if bud:                                            # the same-money extras: the same treatment
        out['bud_extras'] = {'arms': bud, 'label': T.BUD_LABEL,
                             'note': 'post hoc: in no pair label, not in S_UC (the margin) and not in the drift\'s max / min'}
    if isinstance(ht, dict):
        t_ht = _metric(ht) if ht.get('valid') else None
        out['F-HT'] = {'line': 'best known human recipe on this node set', 'valid': ht.get('valid'), 'step_time_s': t_ht,
                       'speedup_vs_uc_ref': (t_ref / t_ht) if t_ht else None,
                       'numerics_class': (ht.get('judge') or {}).get('numerics_class')}
    if 'F-PROMPTS' in arms and 'F-A0' in arms:
        tp, ta = prefix_best(arms['F-PROMPTS']['trace']), prefix_best(arms['F-A0']['trace'])
        yp = math.log(t_ref / tp) if tp else 0.0
        ya = math.log(t_ref / ta) if ta else 0.0
        m = margin_for(margin_rec, 'F-PROMPTS', 'F-A0')
        d = ya - yp
        lm = math.log1p(m)
        out['F-PROMPTS matched prefix'] = {
            'note': 'best valid of F-PROMPTS\'s 3 screens vs the best valid of F-A0\'s first 3 screens (search side), '
                    'outside the main criterion', 'F-PROMPTS_best': tp, 'F-A0_first3_best': ta, 'delta_y': d, 'margin': m,
            'label': 'F-A0 ahead' if d >= lm - 1e-12 else ('F-PROMPTS ahead' if -d >= lm - 1e-12 else 'parity')}
    for other in ('F-A0', 'F-OPTICO'):
        lab = labels.get(f'{other} vs F-RND')
        if lab and lab['label'] not in ('advantage', NO_COMPARISON):
            out.setdefault('random_control', {})[other] = 'no evidence in this space that LLM search beats random'
    superseded = [{'arm': a, **v} for a, hv in harness.items() for v in hv['recovered']]
    if superseded:
        out['ext009'] = {'disclosure': EXT009_DISCLOSURE, 'superseded_or_recovered_screens': superseded,
                         'collection_values': ['collected', 'recollected_readonly', 'rescreen'],
                         'classes': ['failed', 'runner_error', 'infra_refusal', 'harness_lost', 'recovered']}
    pairs = [r['t_uc_pair'] for a, r in rows.items() if r.get('t_uc_pair') and a not in T.EXTRA_ARMS]
    out['drift'] = {'uc_halves_over_ref': {a: r['drift_uc_pair_over_ref'] for a, r in rows.items()},
                    'max_over_min_uc_halves': (max(pairs) / min(pairs)) if len(pairs) >= 2 else None}
    return out
