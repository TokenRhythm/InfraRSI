"""The system side of a search (revamp design C, after the Fable 5.1 consult): strategies only propose and observe;
the engine owns evaluation at three fidelities, noise estimates, the incumbent, the value/cost decision and the
numeric stop rule.

    Engine(session, ...).run(strategy) -> SearchTrace
    Strategy.propose(engine, k) -> [Proposal];  Strategy.observe(engine, proposal, outcome);  Strategy.residual(engine)

Fidelities: probe (1-2 updates: feasibility, memory), short (a step-time estimate), full (the verifier-complete run).
A probe or short run gets the job's full {steps} horizon (its warm-up and decay) and is stopped at an update boundary
after its updates; a short run's step time is compared with the same updates of its base's full run.
A full run can be aborted at an update boundary once it is significantly slower than the incumbent (one-sided z-test
on the post-warm-up per-update times). The incumbent changes only for a gain beyond the pair noise (2 sigma_pair).

Decision: for an open proposal p with predicted log gain mu (already calibrated by the strategy), uncertainty sigma
and feasibility q: EI = q (mu Phi(z) + s phi(z)), s^2 = sigma^2 + sigma_m^2, z = mu / s. Value V = EI x H_job, where
H_job is the GPU-hours the production job still runs (a log gain g saves about g x H_job); cost C = expected GPU-hours
of evaluating p. Run argmax V/C while V/C >= lambda or the exploration floor is not met; otherwise stop and report the
numbers (best V/C, residual, kappa, sigma), never "the model thinks we are done". Without a declared H_job (None) V/C
only orders the proposals: the search runs until the budget is spent or nothing is left to propose.

Robust mode (a strategy's `robust_stats`, hls3 S-9; off for every other strategy): the search's own decisions read the
median update instead of the window mean - the incumbent's and each screen's step time, the early abort, the re-runs -
and the robust split halves; a proposal with stat 'mean' (a move that removes stalls) is judged by the means throughout
(its early abort, its gain, its confirmation, their noise); a short run is still read by its mean (a trial keeps no
per-update times). A GC stall moves the mean of a 20-update window by 1.6-14 % and the median by almost nothing
(FABLE_TAKEAWAYS T8). What leaves the search is unchanged: `noise()` exports the window means' pairs and their
sigma_pair (the promotion's margin and the project's noise ledger read them); the median pairs stay in the search
(`pairs_robust`, and the trace's noise.robust). The verifier, the promotion and its margin do not change.
"""
from __future__ import annotations

import dataclasses
import json
import math
import statistics

from ..agent.search import START_FAILED, USER_STOP, SearchSession, SearchTrace
from ..agent.state import Stateful
from ..project.signals import clock_time, intervals


@dataclasses.dataclass
class Proposal:
    change: dict = dataclasses.field(default_factory=dict)   # knob values to set on the base config
    patch: dict | None = None                                # {file: full text}, merged onto the base candidate's files
    base: str | None = None                                  # base candidate id; None = the incumbent
    mu: float = 0.0                                          # predicted log gain vs the base
    sigma: float = 0.05
    p_feasible: float = 0.9
    mechanism: str = ''
    hypothesis: str | None = None
    origin: str = ''
    fidelity: tuple = ('full',)                              # the evaluation plan, e.g. ('probe', 'full'), ('short',)
    abortable: bool = True
    absolute: bool = False                                   # change is a complete config (random / TPE)
    confirm: bool = False                                    # a composite: confirm a near-threshold gain by a re-run pair
    change_class: str | None = None                          # the proposer's label: execution | numerics | algorithmic
    keep_probe: bool = False                                 # hls3 (S-1): probe even when a probe costs half a full run
    stat: str | None = None                                  # hls3 (S-9): 'mean' judges it by the mean (a stall remover)
    probe_cap: float | None = None                           # hls3 (F1a): the probe's wall cap (s), not the job's timeout
    gate: object = None                                      # hls3 (F1b): (engine, cid, base, probe row) -> why the full
    #                                                          run after a passed probe is not worth it, else None
    retry: bool = False                                      # hls3 (F1a): the one full retry of a probe its own start cap
    #                                                          stopped - not a duplicate, its candidate reopened
    probe_steps: int | None = None                           # hls3 (F12): this probe's updates, when more than the engine's
    #                                                          (a speed-gated probe compares its updates after a warm-up)


@dataclasses.dataclass
class Outcome:
    cid: str | None
    status: str          # screened | infeasible | aborted | invalid | duplicate | rejected | short | probed | infra
                         # (infra: no verdict; probed: a probe-only plan that ran, e.g. DOE's feasibility check)
    valid: bool
    step_time: float | None
    gain: float | None   # measured log gain vs the base (full or short fidelity)
    fidelity: str
    gpu_hours: float
    reasons: list
    failed_checks: list
    loss_deviation: float | None = None    # the loss check's relative deviation from the reference (a full run)
    loss_deviation_base: float | None = None   # ... from the base the change was applied to (its own effect)
    gain_mean: float | None = None         # robust mode (hls3 S-9): the gain on the window means, next to the medians'


class BoundGate:
    """A Proposal.gate a search-state checkpoint can keep (agent/state.py; a lambda or a closure cannot be pickled): the
    method `name` of `owner` (a strategy) called with the engine's (engine, cid, base, row) and then `args`; `base`, when
    given, replaces the base the engine passes."""
    __slots__ = ('owner', 'name', 'args', 'base')

    def __init__(self, owner, name: str, *args, base=None):
        self.owner, self.name, self.args, self.base = owner, name, args, base

    def __getstate__(self):
        return (self.owner, self.name, self.args, self.base)

    def __setstate__(self, state):
        self.owner, self.name, self.args, self.base = state

    def __call__(self, engine, cid, base, row):
        return getattr(self.owner, self.name)(engine, cid, base if self.base is None else self.base, row, *self.args)


def no_gate(*args, **kw):
    """A gate that never refuses (a probe-only plan: the engine passes the detailed row, the plan asks nothing of it)."""
    return None


def loss_deviation(verdict: dict | None) -> float | None:
    """The loss check's measured deviation in a verdict (per update when the data order is proven, else the window
    mean); None when it was not checked."""
    for x in (verdict or {}).get('checks') or []:
        if isinstance(x, dict) and x.get('id') == 'loss_within_tolerance':
            d = (x.get('evidence') or {}).get('deviation')
            return float(d) if type(d) in (int, float) else None
    return None


class Strategy(Stateful):
    name = 'strategy'
    uses_engine_stop = True          # False: the strategy decides when it is done (returns no proposals)
    CHECKPOINT_LIVE = ('engine',)    # search-state checkpoints (agent/state.py): the engine it was bound to is the resuming run's

    def propose(self, engine: 'Engine', k: int) -> list[Proposal]:
        raise NotImplementedError

    def observe(self, engine: 'Engine', proposal: Proposal, outcome: Outcome) -> None:
        pass

    def residual(self, engine: 'Engine') -> dict:
        return {}

    def describe(self) -> dict:
        return {'name': self.name}

    @classmethod
    def proposer_features(cls, off=()) -> dict:
        """What this strategy asks of its LLMProposer (proposers.FEATURES: robust parsing, the patch self-proof, the
        library source; search/__init__._engine passes it as `features`): nothing by default - a strategy that probes code
        opts in (hls3 by its components; `off`: the ablated ones)."""
        return {}


def _phi(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _Phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def expected_improvement(mu: float, s: float, q: float = 1.0) -> float:
    if s <= 0:
        return q * max(0.0, mu)
    z = mu / s
    return q * (mu * _Phi(z) + s * _phi(z))


def pooled_sigma_pair(pairs, halves) -> float | None:
    """sqrt(max(mean d^2 over pairs of runs of one configuration, the split halves' spread)): the halves' RMS with fewer
    than three of them, else their sample sd about their mean - a within-run trend common to every run (D2-R205b: the
    second half 5 % slower in all eight runs, halves 0.044..0.066) cancels in any comparison of two runs and is not
    between-run noise, while one odd half among many (a random stall) keeps its full weight; None without samples."""
    if not pairs and not halves:
        return None
    h2 = (statistics.variance(halves) if len(halves) >= 3 else sum(x * x for x in halves) / len(halves)) if halves else 0.0
    return math.sqrt(max(sum(x * x for x in pairs) / len(pairs) if pairs else 0.0, h2))


def median_update(ev, window=None) -> float | None:
    """The median per-update interval over `window` (default: the run's scoring window) on InfraRSI's clock, else the
    median of the reported step times (robust mode); None below three intervals."""
    if not isinstance(ev, dict):
        return None
    w = list(window or ev.get('window') or [])
    if len(w) != 2:
        return None
    rows = ev.get('per_step') or []
    xs = intervals(rows, w)[1]
    if len(xs) < 3:
        xs = [r['step_time'] for r in rows if w[0] <= r.get('step', 0) <= w[1] and type(r.get('step_time')) in (int, float)]
    return statistics.median(xs) if len(xs) >= 3 and statistics.median(xs) > 0 else None


def white_floor(ev, robust: bool = False) -> float:
    """sqrt 2 x per-update RSD / sqrt(updates in the scoring window): the noise of two window means if the per-update
    times were independent, a lower bound of sigma_pair (0 when the run has no RSD). With `robust` (the pool has a pair
    of runs of one configuration, which shows whether a stall repeats) the robust RSD (`rsd_robust`, MAD based) when
    the receipt has it: a stall that recurs at the same update in every run (D2-05) puts the plain RSD at 0.5 and the
    floor at 16 % while the repeats of the configuration agreed within 0.5 %; one run alone cannot tell such a stall
    from a random one, which a window mean does carry (DF4_r1)."""
    key = 'rsd_robust' if robust and isinstance(ev, dict) and type(ev.get('rsd_robust')) in (int, float) else 'rsd_window'
    rsd = ev.get(key) if isinstance(ev, dict) else None
    w = ev.get('window') if isinstance(ev, dict) else None
    if type(rsd) not in (int, float) or rsd <= 0 or not (isinstance(w, (list, tuple)) and len(w) == 2):
        return 0.0
    return math.sqrt(2) * rsd / math.sqrt(max(1, w[1] - w[0] + 1))


class Engine(Stateful):
    recheck_agree = 1.0      # two aborted runs 'agree' within this many sigma_pair (see recheck_after)
    # search-state checkpoints (agent/state.py): the session, the log, the writer and the settings of a screen (steps, the
    # job's GPU-hours, the noise floor, the floor of screens) are the resuming run's; the rest is saved
    CHECKPOINT_LIVE = ('s', 'log', 'checkpoint', 'resumes', 'full_steps', 'probe_steps', 'warmup', 'short_steps', 'job_gpu_hours',
                       'lam', 'floor_screens', 'multi_fidelity', 'early_abort', 'run_noise', 'accounting', '_active_strategy')

    def __init__(self, session: SearchSession, *, full_steps: int, short_steps: int | None = None, probe_steps: int = 2,
                 warmup: int = 3, job_gpu_hours: float | None = None, lam: float = 1.0, floor_screens: int | None = None,
                 multi_fidelity: bool = True, early_abort: bool = True, run_noise: float | None = 0.015,
                 noise_pairs=None, log=None, robust: bool | None = None, checkpoint=None, accounting=None):
        self.s = session
        self.accounting = accounting
        self.checkpoint = checkpoint                     # (engine, strategy, why, stop) -> None after each observed run, or None
        self.resumes: list[dict] = []                    # the resumes of this search (the flow's list), in the trace's extra
        self._evaluation = None                          # continuation between a candidate's fidelity runs
        self._pending = None                             # the proposal chosen and not yet observed: a checkpoint keeps it
        #                                                  (the model calls that chose it are paid), a resumed search runs it first
        self.robust = robust                             # None: the strategy's `robust_stats` decides at run()
        self.full_steps, self.probe_steps, self.warmup = int(full_steps), int(probe_steps), int(warmup)
        self.short_steps = int(short_steps or max(self.warmup + 5, full_steps // 3))
        self.job_gpu_hours = None if job_gpu_hours is None else float(job_gpu_hours)     # None: no value/cost stop
        self.lam = float(lam)
        b = session.budget.screens
        self.floor_screens = floor_screens if floor_screens is not None else min(b, max(4, math.ceil(0.25 * b)))
        self.multi_fidelity, self.early_abort = multi_fidelity, early_abort
        # between-run noise: a fixed floor (manifest search.run_noise, the simulator), or None = measured from the runs
        self.run_noise = None if run_noise is None else float(run_noise)
        self.pairs = [float(x) for x in noise_pairs or () if type(x) in (int, float)]   # log ratios, runs of one config
        self._prior = len(self.pairs)                    # the earlier runs' pairs (the project's noise ledger)
        self.pairs_robust: list[float] = []              # robust mode: this search's pairs on the median updates (its own
        self.log = log or (lambda m: None)               # noise only; `pairs` keeps the window means' offsets)
        self._inc = 'c0'
        self.decisions: list[dict] = []
        self.stop_numbers: dict | None = None
        self.tried: set = set()
        self.retried: set = set()                       # hls3 (F1a): keys whose one full retry ran
        self._empty = 0
        self.confirmations: list[dict] = []
        self.t_conf: dict[str, float] = {}               # step time of a confirmed candidate (mean of two runs)
        self.rechecks: list[dict] = []                   # re-runs of the incumbent after aborts that agreed with each other
        self._aborted: list[tuple] = []                  # (incumbent, cid, post-warm-up mean) of the aborts against it
        self._rerun: dict[str, tuple] = {}               # (step time, evidence) of the re-run that replaced a screen
        self._rerun_mean: dict[str, float] = {}          # robust mode: that re-run's window mean (its step time: the median)

    # ------------------------------------------------------------ state
    def t(self, cid: str) -> float | None:
        return self.t_conf[cid] if cid in self.t_conf else self.t_screen(cid)

    def t_window(self, cid: str, window) -> float | None:
        """cid's full run's step time over `window`, from its per-update timestamps: the like-for-like base of a short
        run (the same early updates); its full-window time when the receipt has no timestamps there."""
        c = self.s.candidates.get(cid)
        ev = self._rerun[cid][1] if cid in self._rerun else (c.receipt or {}).get('evidence') if c else None
        t = clock_time(ev.get('per_step') or [], list(window))[0] if isinstance(ev, dict) and self.t(cid) else None
        return t or self.t(cid)                         # (a short run's own time is its mean: so is this, in robust mode)

    def t_partial(self, cid: str) -> float | None:
        """The post-warm-up mean step time of an aborted run (what the abort rule compared), from its timestamps (robust
        mode: the median update)."""
        c = self.s.candidates.get(cid)
        ev = (c.receipt or {}).get('evidence') if c else None
        done = ev.get('steps_completed') if isinstance(ev, dict) else None
        if type(done) is not int or done <= self.warmup + 1:
            return None
        if self.robust:
            m = median_update(ev, [self.warmup + 1, done])
            if m:
                return m
        return clock_time(ev.get('per_step') or [], [self.warmup + 1, done])[0]

    def t_screen(self, cid: str) -> float | None:
        """Step time of the candidate's full-length screen (not the mean of a confirmation pair); robust mode: the median
        update of the screen's scoring window."""
        c = self.s.candidates.get(cid) or None
        v = c and c.verdict
        t = v.get('step_time') if v and v.get('valid') and v.get('step_time') else None
        if t and self.robust:
            t = median_update((c.receipt or {}).get('evidence')) or t
        return t

    def t_mean(self, cid: str) -> float | None:
        """The window mean of the candidate's screen (robust mode judges a stall remover by it), else its time."""
        return self._screen_mean(cid) or self.t(cid)

    def _screen_mean(self, cid: str) -> float | None:
        """The window mean of the candidate's full-length screen: what the verifier reported (None without a valid one)."""
        v = (self.s.candidates.get(cid) or None)
        v = v and v.verdict
        return v.get('step_time') if v and v.get('valid') and v.get('step_time') else None

    def incumbent(self) -> str:
        return self._inc

    def full_screens(self) -> int:
        return int(self.s.ledger.used['screens'])

    def sigma_step(self, robust: bool | None = None) -> float:
        """Relative sd of per-update times inside a run (the incumbent's scoring window); `robust` (default: robust
        mode): the robust one (MAD based), which a stall does not inflate."""
        robust = bool(self.robust) if robust is None else robust
        c = self.s.candidates.get(self._inc)
        ev = (c.receipt or {}).get('evidence') if c and isinstance((c.receipt or {}).get('evidence'), dict) else {}
        rsd = ev.get('rsd_window') if ev else None
        if robust and ev and type(ev.get('rsd_robust')) in (int, float) and ev['rsd_robust'] > 0:
            rsd = ev['rsd_robust']
        return float(rsd) if type(rsd) in (int, float) and rsd > 0 else 0.02

    def split_halves(self, robust: bool | None = None) -> list[float]:
        """The split-half values (signals.split_half) of the valid full-length runs made so far, screens and re-runs:
        the robust ones (medians, `split_half_robust`) once the pool has a pair of runs of one configuration, which
        shows whether a stall repeats, else the plain ones (DF4_r1 #2); a receipt without the robust value gives the
        plain one. Robust mode compares medians, so the medians' halves are its noise from the start."""
        robust = (bool(self.pairs) or bool(self.robust)) if robust is None else robust
        key = 'split_half_robust' if robust else 'split_half'
        out = []
        for c in self.s.candidates.values():
            ev = (c.receipt or {}).get('evidence')
            rows = ([ev] if isinstance(ev, dict) and (c.verdict or {}).get('valid') else []) + \
                [r for r in c.results.get('rescore') or [] if r.get('valid')]
            hs = [r.get(key) if type(r.get(key)) in (int, float) else r.get('split_half') for r in rows]
            out += [float(h) for h in hs if type(h) in (int, float)]
        return out

    def sigma_pair(self, robust: bool | None = None) -> float:
        """Noise of the comparison of two runs (log step time). Fixed: sqrt 2 x max(run_noise, per-update RSD), the
        simulator's and a manifest search.run_noise. Measured (run_noise None): sqrt(max(mean d^2 over the pairs of runs
        of one configuration, mean h^2 over the split halves of every valid full run)), recomputed after each run and
        never below the white noise of two window means; no measurement at all: the old default (2 % run noise).
        `robust` (default: robust mode): the noise of two median updates - the earlier runs' pairs with this search's
        median pairs, the robust split halves - which only the search's own decisions read."""
        robust = bool(self.robust) if robust is None else robust
        if self.run_noise is not None:
            return math.sqrt(2) * max(self.run_noise, self.sigma_step(robust))
        pairs = self.pairs[:self._prior] + self.pairs_robust if robust else self.pairs
        s = pooled_sigma_pair(pairs, self.split_halves(bool(pairs) or robust))
        if s is None:
            return math.sqrt(2) * max(0.02, self.sigma_step(robust))
        c = self.s.candidates.get(self._inc)
        return max(s, white_floor((c.receipt or {}).get('evidence') if c else None, robust=bool(pairs) or robust))

    def sigma_run(self, robust: bool | None = None) -> float:
        """Between-run sd of a run's mean step time (`robust`: of its median update)."""
        return (self.sigma_pair() if robust is None else self.sigma_pair(robust)) / math.sqrt(2)

    def noise(self) -> dict:
        """The noise as the promotion, the report and the project's noise ledger read it - the window means' pairs and
        the sigma_pair they give, whatever statistic the search compared; robust mode adds its own numbers under `robust`
        (the median pairs and their sigma_pair: the trace's, no consumer reads them)."""
        out = {'mode': 'fixed' if self.run_noise is not None else 'measured',
               'sigma_pair': round(self.sigma_pair(False) if self.robust else self.sigma_pair(), 6),
               'pairs': [round(x, 6) for x in self.pairs],
               'split_halves': [round(x, 6) for x in (self.split_halves(bool(self.pairs)) if self.robust
                                                      else self.split_halves())],
               'split_halves_plain': [round(x, 6) for x in self.split_halves(robust=False)],   # the promotion picks one
               'split_halves_robust': [round(x, 6) for x in self.split_halves(robust=True)]}   # by its own pairs
        if self.robust:
            out['robust'] = {'statistic': 'median update', 'sigma_pair': round(self.sigma_pair(True), 6),
                             'pairs': [round(x, 6) for x in self.pairs_robust]}
        return out

    def costs(self) -> dict:
        """Mean GPU-hours of a full screen, a short trial and a probe (measured, else priors from a full screen)."""
        rows = self.s.screens
        full = [r['gpu_hours'] for r in rows if r.get('fidelity', 'full') == 'full' and r.get('valid')]
        start = self.s.candidates['c0'].receipt
        f0 = (((start or {}).get('cost') or {}).get('gpu_seconds') or 0) / 3600 if start else 0
        f = sum(full) / len(full) if full else (f0 or 1.0)
        out = {'full': f}
        for name, share in (('short', 0.5), ('probe', 0.35)):
            xs = [r['gpu_hours'] for r in rows if r.get('fidelity') == name]
            out[name] = sum(xs) / len(xs) if xs else share * f
        return out

    # ------------------------------------------------------------ proposals -> candidates -> outcomes
    def _fresh(self, p: Proposal) -> bool:
        """Not measured yet - or hls3's one full retry of a probe its start cap stopped (`retry`, once per key)."""
        k = self.key(p)
        return k not in self.tried or bool(p.retry) and k not in self.retried

    def key(self, p: Proposal) -> str:
        """The proposal's identity in `tried`: a configuration measured short only (DOE's design rows) still gets its
        full screen."""
        base = p.base or self._inc
        return repr((None if p.absolute else base, sorted(p.change.items(), key=str), sorted((p.patch or {}).items()),
                     tuple(p.fidelity)[-1:] == ('short',)))

    def materialize(self, p: Proposal) -> tuple[str | None, str | None, dict]:
        base = 'c0' if p.absolute else (p.base or self._inc)
        fb, cid = {}, base
        if p.change or p.absolute:
            fb = (self.s.propose_config(config=dict(p.change), base=base, origin=p.origin or 'engine', rationale=p.mechanism)
                  if p.absolute else self.s.propose_config(changes=dict(p.change), base=base, origin=p.origin or 'engine',
                                                           rationale=p.mechanism))
            if 'candidate_id' not in fb and fb.get('duplicate_of') and p.retry and \
                    getattr(self.s, 'reopen_start_slow', None) is not None and self.s.reopen_start_slow(fb['duplicate_of']):
                fb = {**fb, 'status': 'proposed'}            # hls3 (F1a): the cap-stopped candidate, reopened for its retry
            if 'candidate_id' not in fb and fb.get('duplicate_of') and (fb.get('status') == 'proposed' or p.patch):
                # only probed or run short so far (DOE), or the base of this patch (measured already): reuse it
                fb = {**fb, 'candidate_id': fb['duplicate_of']}
                if self.s.invalid.get('duplicate'):
                    self.s.invalid['duplicate'] -= 1             # not a rejected proposal
            if 'candidate_id' not in fb:
                return None, 'duplicate' if 'duplicate_of' in fb else 'rejected', fb
            cid = fb['candidate_id']
        if p.patch:
            fb = self.s.propose_patch(files=dict(p.patch), base=cid, origin=p.origin or 'engine', rationale=p.mechanism)
            if 'candidate_id' not in fb and fb.get('duplicate_of') and p.retry and \
                    getattr(self.s, 'reopen_start_slow', None) is not None and self.s.reopen_start_slow(fb['duplicate_of']):
                fb = {**fb, 'candidate_id': fb['duplicate_of']}   # hls3 (F1a): reopened for its retry
            if 'candidate_id' not in fb:
                return None, 'duplicate' if 'duplicate_of' in fb else 'rejected', fb
            cid = fb['candidate_id']
        hit = self.s.unrunnable_hit(cid) if hasattr(self.s, 'unrunnable_hit') else None
        if hit:                                          # it would fail as before: a package it needs is not installed
            self.s.count_invalid('unrunnable')
            return None, 'rejected', {'error': f'this change needs a package that is not installed: {hit}'}
        dead = self.s.start_failed_hit(cid) if getattr(self.s, 'unrunnable_values', None) else None
        if dead:                                         # hls2 (C5): a value whose runs failed to start twice
            self.s.count_invalid('start_failed')
            return None, 'rejected', {'error': f'{START_FAILED} {dead}'}
        return cid, None, fb

    def plan(self, p: Proposal) -> tuple:
        """The proposal's fidelity plan; a probe only when it is cheaper than half a full run (load-dominated jobs make
        probes almost as expensive as a full run). A probe-only plan (a feasibility check) stays a probe."""
        if tuple(p.fidelity) == ('probe',):
            return ('probe',)
        plan = tuple(f for f in p.fidelity if f != 'probe' or self.multi_fidelity) or ('full',)
        if 'probe' in plan and not p.keep_probe:        # (hls3 keeps it while the screens are the scarcer budget: a
            c = self.costs()                            # failed probe spends no screen)
            if c['probe'] >= 0.5 * c['full']:
                plan = tuple(f for f in plan if f != 'probe') or ('full',)
        return plan

    def abort_rule(self, t_ref: float, robust: bool | None = None):
        """Abort when the post-warm-up mean is slower than t_ref (1 + delta) by a one-sided z of 1.645; delta covers the
        run-to-run offset (sigma_run), so a candidate as fast as the incumbent is not killed by an unlucky run. Robust
        (default: robust mode): the median update so far against t_ref (a median), its standard error 1.2533 x the robust
        spread / sqrt k, so that one stall does not kill a candidate within the noise (Q9U-h2 c5 / c8: fused AdamW at
        -1.7 %); robust=False in robust mode: the means' rule (a stall remover's, against the incumbent's window mean)."""
        robust = bool(self.robust) if robust is None else robust
        s_rel, delta = (self.sigma_step(), max(0.01, self.sigma_run())) if robust == bool(self.robust) else \
            (self.sigma_step(robust), max(0.01, self.sigma_run(robust)))   # (robust mode, a stall remover: the means')

        def stop(rows) -> bool:
            pts = [(r['step'], r['t']) for r in rows if isinstance(r.get('t'), (int, float)) and r['step'] > self.warmup]
            if len(pts) < 6:
                return False
            k = pts[-1][0] - pts[0][0]
            if robust:
                gaps = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(pts, pts[1:]) if b[0] > a[0]]
                med = statistics.median(gaps)
                return (med - t_ref * (1 + delta)) / (1.2533 * s_rel * t_ref / math.sqrt(len(gaps))) > 1.645
            mean = (pts[-1][1] - pts[0][1]) / k
            return (mean - t_ref * (1 + delta)) / (s_rel * t_ref / math.sqrt(k)) > 1.645
        return stop

    def evaluate(self, p: Proposal) -> Outcome:
        out = self._evaluate(p)
        if getattr(self.s, 'stop_flag', None) == USER_STOP and out.status == 'rejected':
            # A live stop between fidelities keeps this proposal and its completed-trial costs.
            out.status = 'paused'
        else:
            self._evaluation = None
        return out

    def _evaluate(self, p: Proposal) -> Outcome:
        if tuple(p.fidelity) != ('probe',):              # a feasibility probe leaves the config open for a real run
            if p.retry:
                self.retried.add(self.key(p))
            self.tried.add(self.key(p))
        state = getattr(self, '_evaluation', None)
        if state is None:
            base = 'c0' if p.absolute else (p.base or self._inc)
            cid, why, fb = self.materialize(p)
            if cid is None:
                return Outcome(None, why, False, None, None, 'none', 0.0, [str(fb.get('error') or fb)[:300]], [])
            state = {'base': base, 'cid': cid, 't_base': self.t(base), 'gpu': 0.0, 'index': 0, 'plan': self.plan(p)}
            self._evaluation = state
        base, cid, t_base, gpu = state['base'], state['cid'], state['t_base'], state['gpu']
        for index, fid in enumerate(state['plan']):
            if index < state['index']:
                continue
            state.update(index=index, gpu=gpu)
            active = getattr(self, '_active_strategy', None)
            checkpoint = (lambda: self._checkpoint(active, 'interaction')) if active is not None else None
            self._poll_control(checkpoint=checkpoint)
            if fid in ('probe', 'short'):
                steps = max(self.probe_steps, int(p.probe_steps or 0)) if fid == 'probe' else self.short_steps
                window = [self.warmup + 1, steps] if fid == 'short' and steps > self.warmup + 1 else None
                kw = {}                                  # hls3 (F1): the probe's own start cap, its per-update times
                if fid == 'probe' and p.probe_cap:
                    kw['wall_cap'] = p.probe_cap
                if fid == 'probe' and p.gate is not None:
                    kw['detail'] = True
                row = self.s.trial(cid, steps=steps, fidelity=fid, window=window, horizon=self.full_steps, **kw)
                gpu += row.get('gpu_hours', 0.0)
                if 'error' in row:
                    return Outcome(cid, 'infra' if row.get('infrastructure') else 'rejected', False, None, None, fid, gpu,
                                   [row['error']], [])
                if not row['valid']:
                    return Outcome(cid, 'infeasible' if fid == 'probe' else 'invalid', False, None, None, fid, gpu,
                                   row['reasons'], (self.s.candidates[cid].results.get(fid) or {}).get('failed_checks', []))
                if fid == 'short' and p.fidelity[-1] == 'short':
                    st = row['step_time']
                    tb = self.t_window(base, window) if window else t_base   # the base over the same updates
                    gain = math.log(tb / st) if st and tb else None
                    return Outcome(cid, 'short', True, st, gain, 'short', gpu, [], [])
                if fid == 'probe' and p.fidelity[-1] == 'probe':  # it ran: feasible (no step time worth reading)
                    return Outcome(cid, 'probed', True, None, None, 'probe', gpu, [], [])
                if fid == 'probe' and p.gate is not None:   # hls3 (F1b, F10): a probe after which the full run is not
                    why = p.gate(self, cid, base, row)      # worth it (much slower than its base's, memory-bound; a patch
                    if why:                                 # that does not prove it took effect), with its check's name
                        why, check = why if isinstance(why, tuple) else (why, 'probe_speed')
                        return Outcome(cid, 'infeasible', False, None, None, 'probe', gpu, [why], [check])
                continue
            mean = bool(self.robust) and p.stat == 'mean'    # a stall remover: the means judge it, its early abort too
            t_ref = (self.t_mean(self._inc) or self.t_mean(base)) if mean else (self.t(self._inc) or t_base)
            stop = self.abort_rule(t_ref, robust=False if mean else None) if (
                self.early_abort and p.abortable and self.multi_fidelity and t_ref) else None
            n_rows = len(self.s.screens)
            res = self.s.screen(cid, stop=stop, **({'check_targets': False} if hasattr(self.s, 'check_targets') else {}))
            self._save_noise()
            c = self.s.candidates[cid]
            if len(self.s.screens) > n_rows:                 # a refused screen adds no row and costs nothing
                gpu += self.s.screens[-1].get('gpu_hours') or 0.0
            if 'error' in res:
                return Outcome(cid, 'infra' if res.get('infrastructure') else 'rejected', False, None, None, 'full', gpu,
                               [res['error']], [])
            if c.status == 'aborted':
                gpu += self.recheck_after(cid)
                return Outcome(cid, 'aborted', False, None, None, 'full', gpu, ['aborted: slower than the incumbent'], [])
            failed = [x['id'] for x in (c.verdict or {}).get('checks') or [] if x.get('status') == 'fail']
            dev = loss_deviation(c.verdict)
            dev_base = dev if base == 'c0' else self.deviation_vs(cid, base)
            if not res['valid']:
                return Outcome(cid, 'invalid', False, None, None, 'full', gpu, res['reasons'], failed, dev, dev_base)
            st = res['step_time']
            gain_mean = None
            if self.robust:                           # the medians decide; a stall remover is judged by the means
                gain_mean = math.log(self.t_mean(base) / st) if st and self.t_mean(base) else None
                if mean:
                    t_base, t_inc = self.t_mean(base), self.t_mean(self._inc)
                else:
                    st = self.t_screen(cid) or st
                    t_inc = self.t(self._inc)
            else:
                t_inc = self.t(self._inc)
            gain = math.log(t_base / st) if st and t_base else None
            g = math.log(t_inc / st) if t_inc and st else None
            sp = self.sigma_pair(False) if mean else self.sigma_pair()   # the noise of the statistic compared
            if t_inc is None or (g is not None and g > 2 * sp):
                self._inc, self._aborted = cid, []    # only a gain beyond the pair noise moves the incumbent
            elif p.confirm and g is not None and g > sp:
                self.confirm(cid, g, stat=p.stat)     # a composite near the threshold: one back-to-back re-run pair
                gpu += self.confirmations[-1].get('gpu_hours') or 0.0
            if st and hasattr(self.s, 'check_targets'):
                self.s.check_targets(cid, st, reference_time=self.t('c0') or self._screen_mean('c0'),
                                     accounting=self.accounting, log=self.log)
            return Outcome(cid, 'screened', True, st, gain, 'full', gpu, [], [], dev, dev_base, gain_mean)
        return Outcome(cid, 'invalid', False, None, None, 'none', gpu, ['empty evaluation plan'], [])

    def deviation_vs(self, cid: str, base: str) -> float | None:
        """cid's loss deviation from base's full run, as the verifier's loss check measures it (the change's own effect on
        the numerics, not what its base already changed); None without both runs or a judge that checks the loss."""
        c, b = self.s.candidates.get(cid), self.s.candidates.get(base)
        ra, rb = (c.receipt if c else None), (b.receipt if b else None)
        if not (hasattr(self.s.judge, 'screen_checks') and isinstance(ra, dict) and isinstance(rb, dict)):
            return None
        try:
            return loss_deviation({'checks': self.s.judge.screen_checks(ra.get('evidence') or {}, rb.get('evidence') or {})})
        except Exception:                               # noqa: BLE001 - a diagnostic never fails the search
            return None

    def recheck_after(self, cid: str) -> float:
        """After an abort: two aborted runs in a row that agree with each other (within `recheck_agree` sigma_pair, 1 by
        default) but not with the incumbent's screen are as likely a screen the configuration no longer reproduces as two slower candidates
        (D1-20: the c1 family ran at 3.06 s and then at 3.24 s; four candidates were aborted against the 3.06 s screen).
        Then the incumbent is re-run once; a re-run slower than its screen by more than 2 sigma_pair becomes the step time
        the next runs are compared with, an agreeing one is averaged in (as `confirm` does). At most two re-checks per
        incumbent and none after one moved its time; the re-run's pair stays out of the noise pool (a configuration with
        two speeds is not the machine's noise: pooled, it would reject its own real gain at promotion). Returns the
        re-run's GPU-hours (0 without one)."""
        inc, t = self._inc, self.t_partial(cid)
        prev = self._aborted[-1] if self._aborted and self._aborted[-1][0] == inc else None
        self._aborted.append((inc, cid, t))
        mine = [r for r in self.rechecks if r['incumbent'] == inc]
        t0 = self.t_screen(inc)
        if prev is None or not (t and prev[2] and t0) or abs(math.log(t / prev[2])) > self.recheck_agree * self.sigma_pair():
            return 0.0
        if len(mine) >= 2 or any(r['moved'] for r in mine):
            return 0.0
        n0 = len(self.s.screens)
        r = self.s.rescore(inc)
        self._save_noise()
        t_mean = r.get('step_time') if r.get('valid') else None
        if self.robust and r.get('valid') and r.get('step_time'):     # robust mode: the re-run's median update
            r = {**r, 'step_time': median_update(r.get('evidence')) or r['step_time']}
        self._aborted = []
        rec = {'incumbent': inc, 'after_aborts': [prev[1], cid], 'aborted_step_times': [round(prev[2], 5), round(t, 5)],
               't_screen': t0, 'gpu_hours': sum(x.get('gpu_hours') or 0.0 for x in self.s.screens[n0:]), 'moved': False,
               **({'statistic': 'median update'} if self.robust else {})}
        if not (r.get('valid') and r.get('step_time')):
            rec['why'] = r.get('error') or 'the re-run was not valid'
        else:
            d, thr = math.log(r['step_time'] / t0), 2 * self.sigma_pair()
            rec.update(t_rerun=r['step_time'], d=round(d, 5), threshold=round(thr, 5), moved=d > thr)
            self.t_conf[inc] = r['step_time'] if d > thr else math.sqrt(t0 * r['step_time'])
            if d > thr:                                  # its later pairs and short-window bases come from the re-run
                self._rerun[inc] = (r['step_time'], r.get('evidence') or {})
                if self.robust:
                    self._rerun_mean[inc] = t_mean
            self.log(f'[search] incumbent {inc} re-run after {prev[1]} and {cid} were aborted at the same speed: '
                     f'{r["step_time"]:.4g} s vs its screen {t0:.4g} s ({100 * d:+.2f} %, '
                     + ('beyond the noise: the next runs are compared with the re-run' if d > thr else 'within the noise'
                        if abs(d) <= thr else 'faster than its screen beyond the noise: the two are averaged') + ')')
        self.rechecks.append(rec)
        return rec['gpu_hours']

    def confirm(self, cid: str, d1: float, stat: str | None = None) -> bool:
        """Re-run the incumbent and the candidate back to back; move when the mean of the two paired log differences
        (sd sigma_pair / sqrt 2 = sigma_run) exceeds 1.645 sigma_run (one-sided 5 %). Robust mode: the re-runs' median
        updates decide (a stall remover, stat 'mean': the means, as its d1 and its noise); either way the pairs of runs
        of one configuration keep the window means' offsets (`pairs`: what the promotion and the noise ledger read) and
        the medians' go to the search's own pool."""
        inc, n0 = self._inc, len(self.s.screens)
        r_inc = self.s.rescore(inc)
        self._save_noise()
        r_c = self.s.rescore(cid) if r_inc.get('valid') else {}
        self._save_noise()

        def med(r):                                      # robust mode: a re-run's median update
            return (median_update(r.get('evidence')) or r['step_time']) if self.robust else r['step_time']
        for c, r in ((inc, r_inc), (cid, r_c)):          # each re-run with its own screen (or the re-run that replaced
            if not (r.get('valid') and r.get('step_time')):   # it): a pair of runs of one config
                continue
            if c in self._rerun:                         # the window means: the pair the promotion's margin reads
                t0 = self._rerun_mean.get(c) if self.robust else self._rerun[c][0]
            else:
                t0 = self._screen_mean(c)
            if t0:
                self.pairs.append(math.log(r['step_time'] / t0))
            t0 = self._rerun[c][0] if c in self._rerun else self.t_screen(c)
            if self.robust and t0:
                self.pairs_robust.append(math.log(med(r) / t0))
        gpu = sum(r.get('gpu_hours') or 0.0 for r in self.s.screens[n0:])
        rec = {'cid': cid, 'incumbent': inc, 'd1': round(d1, 5), 'gpu_hours': gpu}
        use_mean = not self.robust or stat == 'mean'
        if self.robust:
            rec['statistic'] = 'mean' if use_mean else 'median update'
        if not (r_inc.get('valid') and r_c.get('valid') and r_inc.get('step_time') and r_c.get('step_time')):
            rec.update(moved=False, why=r_inc.get('error') or r_c.get('error') or 'a re-run was not valid')
            self.confirmations.append(rec)
            return False
        d2 = math.log(r_inc['step_time'] / r_c['step_time']) if use_mean else math.log(med(r_inc) / med(r_c))
        d, thr = (d1 + d2) / 2, 1.645 * (self.sigma_run(False) if self.robust and use_mean else self.sigma_run())
        rec.update(d2=round(d2, 5), d=round(d, 5), threshold=round(thr, 5), moved=d > thr)
        if d > thr:
            self.t_conf[cid] = math.sqrt(self.t(cid) * med(r_c))     # the search's statistic of the candidate
            self._inc, self._aborted = cid, []
        self.confirmations.append(rec)
        return d > thr

    # ------------------------------------------------------------ decision and stop
    def expected_cost(self, p: Proposal) -> float:
        c = self.costs()
        plan = self.plan(p)
        cost, survive = 0.0, 1.0
        for f in plan:
            cf = c[f]
            if f == 'probe' and p.probe_steps and p.probe_steps > self.probe_steps:   # hls3: a longer probe's updates
                cf += (p.probe_steps - self.probe_steps) * c['full'] / max(1, self.full_steps)
            cost += survive * cf
            if f == 'probe':
                survive *= p.p_feasible
        return max(cost, 1e-9)

    def value(self, p: Proposal) -> float:
        s = math.sqrt(p.sigma ** 2 + self.sigma_run() ** 2)
        h = 1.0 if self.job_gpu_hours is None else self.job_gpu_hours
        return expected_improvement(p.mu, s, p.p_feasible) * h

    def decide(self, proposals: list[Proposal]) -> tuple[Proposal | None, dict]:
        scored = sorted(((self.value(p) / self.expected_cost(p), p) for p in proposals), key=lambda x: -x[0])
        best_ratio, best = scored[0]
        floor_met = self.full_screens() >= self.floor_screens
        on = self.job_gpu_hours is not None
        numbers = {'best_value_per_cost': round(best_ratio, 4), 'lambda': self.lam, 'floor_screens': self.floor_screens,
                   'full_screens': self.full_screens(), 'floor_met': floor_met, 'sigma_run': round(self.sigma_run(), 4),
                   'n_open': len(proposals), 'best_mechanism': best.mechanism, 'best_mu': round(best.mu, 4),
                   'job_gpu_hours': self.job_gpu_hours, 'stop_rule': 'on' if on else 'off'}
        self.decisions.append(numbers)
        if self.stops_at(best_ratio):
            return None, numbers
        return best, numbers

    def stops_at(self, best_ratio: float) -> bool:
        """The numeric stop rule decide() applies: with a declared H_job, the best value/cost below lambda once the
        exploration floor is met (a strategy that narrows its proposals can ask it first: hls2's backstop)."""
        return self.job_gpu_hours is not None and best_ratio < self.lam and self.full_screens() >= self.floor_screens

    def _poll_control(self, checkpoint=None):
        poll = getattr(self.s, 'poll_control', None)
        if poll is not None:
            poll(checkpoint=checkpoint)

    # ------------------------------------------------------------ driver
    def run(self, strategy: Strategy, *, max_iters: int = 100000) -> SearchTrace:
        self._active_strategy = strategy
        self.s.ledger.allow_token_free = bool(getattr(strategy, 'supports_token_free', False))
        if self.s.ledger.allow_token_free and self.s.stop_flag == 'budget:llm_tokens':
            self.s.stop_flag = None
        if self.robust is None:                         # the strategy's statistic (hls3 S-9; every other: the means)
            self.robust = bool(getattr(strategy, 'robust_stats', False))
        try:
            reason = self._loop(strategy, max_iters)
        except KeyboardInterrupt:                       # with the flow's `verify` policy (Ctrl-C / SIGTERM): the search ends
            if getattr(self.s, 'on_interrupt', None) != 'verify':   # here as a normal one, its noise and ledger for the
                raise                                   # promotion that follows; otherwise the flow reports the interruption
            reason = USER_STOP
        if reason != USER_STOP:                         # (cut mid-run, the state may hold a candidate no run has finished:
            self._checkpoint(strategy, 'end', reason)   #  the checkpoint before that run stays the one a resume starts from)
        extra = self._extra(strategy, reason)
        return self.s.finalize(selected=self._inc, stop_reason=reason or 'unknown', extra=extra, override=False)

    def _loop(self, strategy: Strategy, max_iters: int) -> str | None:
        """Propose, evaluate, observe until a stop; returns why the search stopped."""
        reason = None
        for _ in range(max_iters):
            self._poll_control(checkpoint=lambda: self._checkpoint(strategy, 'interaction'))
            if self.s.stop_flag:
                reason = self.s.stop_flag
                break
            ex = self.s.ledger.exhausted()
            if ex:
                reason = 'budget:' + ex[0]
                break
            if self._pending is not None:                # a resumed search: the proposal its checkpoint was about to run
                p = self._pending
            else:
                raw = strategy.propose(self, 16)
                props = [p for p in raw if self._fresh(p)]
                for p in raw:
                    if not self._fresh(p):         # already measured: the strategy learns it, the loop goes on
                        strategy.observe(self, p, Outcome(None, 'duplicate', False, None, None, 'none', 0.0, ['already tried'],
                                                          []))
                if not props:
                    if raw and self._empty < 50:
                        self._empty += 1
                        continue
                    from .token_governor import TokenGovernor
                    reason = ('budget:llm_tokens' if TokenGovernor(self.s.ledger).exhausted
                              else 'strategy_done')
                    break
                self._empty = 0
                if strategy.uses_engine_stop:
                    p, numbers = self.decide(props)
                    if p is None:
                        reason, self.stop_numbers = 'stop_rule', numbers
                        break
                else:
                    p = props[0]
                self._pending = p
                self._checkpoint(strategy, 'proposed')   # (after the calls that chose it, before its run)
            self._poll_control(checkpoint=lambda: self._checkpoint(strategy, 'interaction'))
            if self.s.stop_flag == USER_STOP:
                self._checkpoint(strategy, 'interaction', USER_STOP)
                return USER_STOP
            out = self.evaluate(p)
            if out.status == 'paused':
                self._checkpoint(strategy, 'interaction', USER_STOP)
                return USER_STOP
            what = (p.mechanism or str(p.change))[:120]
            self.log(f'[search] {strategy.name}: {what} -> {out.status}'
                     + (f', {100 * out.gain:+.2f} % vs its base' if out.gain is not None else '')
                     + (f' ({out.reasons[0][:160]})' if out.reasons and not out.valid else '')
                     + (' -> new incumbent' if out.cid is not None and out.cid == self._inc else ''))
            strategy.observe(self, p, out)
            self._pending = None
            self._checkpoint(strategy, 'observe')
            self._save_noise()
        else:
            reason = 'max_iters'
        return reason

    def _save_noise(self) -> None:
        """The noise summary after each run and re-run, for `infrarsi finish` when the process ends without a report: the
        promotion then holds a candidate to the margin this search measured."""
        out = getattr(self.s, 'out_dir', None)          # (a session that keeps no files has none)
        if out is None:
            return
        try:
            (out / 'noise.json').write_text(json.dumps(self.noise()), encoding='utf-8')
        except OSError:
            pass

    def _extra(self, strategy: Strategy, reason: str | None) -> dict:
        """The trace's extra section. A search interrupted inside a strategy's update can leave it half made: the summaries
        then give way to what the promotion needs (the noise and the incumbent) rather than cost the run its report."""
        try:
            ledger = getattr(strategy, 'ledger_view', None)
            return {'strategy': strategy.describe(), 'incumbent': self._inc, 'stop_numbers': self.stop_numbers,
                    'stop_rule': self.decisions[-1]['stop_rule'] if self.decisions else None,
                    'ledger': ledger() if callable(ledger) else None,
                    'decisions': self.decisions[-50:], 'residual': strategy.residual(self),
                    'confirmations': self.confirmations, 'rechecks': self.rechecks, 'noise': self.noise(),
                    'fidelity_steps': {'probe': self.probe_steps, 'short': self.short_steps, 'full': self.full_steps},
                    'multi_fidelity': self.multi_fidelity, 'early_abort': self.early_abort,
                    **({'statistic': 'median update (robust mode)'} if self.robust else {}),
                    **({'resumes': list(self.resumes)} if self.resumes else {})}
        except Exception as exc:                         # noqa: BLE001
            if reason != USER_STOP:
                raise
            self.log(f'[search] the strategy summary was cut by the interrupt ({type(exc).__name__}: {str(exc)[:200]})')
            return {'incumbent': self._inc, 'noise': self.noise(), 'confirmations': self.confirmations}

    def _checkpoint(self, strategy: Strategy, why: str, stop: str | None = None) -> None:
        """The search-state checkpoint (agent/state.py, project/resume.py) after a proposal is chosen ('proposed', with
        `_pending`), after an observed run and at the end of the search: passive - the writer only reads the state - and a
        failed write never stops the search."""
        if self.checkpoint is None:
            return
        try:
            self.checkpoint(self, strategy, why, stop)
        except Exception as exc:                         # noqa: BLE001 - a checkpoint that cannot be written is logged
            self.log(f'[search] checkpoint not written: {type(exc).__name__}: {str(exc)[:300]}')
