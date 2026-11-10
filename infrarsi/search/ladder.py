"""The rung ladder: probe -> short -> full for any Engine strategy (possible_search_algorithm/20 section 5) - a shared
component (esls and rpp / rpp-ctl use it; wcs may later).

A configuration climbs at most three rungs, each a fidelity the engine already has:

    probe   a few updates (feasibility, the measured peak, the speed gate past the warm-up): first when its feasibility
            is below `probe_below` (the cliff is judged here: short runs carry no per-update times)
    short   about a third of the updates (the engine compares them with the same updates of its base's full run)
    full    the verifier-complete run: the only rung that counts as a screen and that can move the incumbent

After a short run the full run follows only when the configuration may still beat the incumbent: its short gain over the
reference plus `promote_z` x its spread reaches the incumbent's gain (one-sided; the engine's own early abort still
guards every full run). A configuration whose declared warm-up is longer than the engine's skips the short rung (its
first updates would mislead).

Budget: probes and short runs are not counted as screens; their GPU-hours are counted in the search's GPU-hours (the
session's trials). The low-fidelity allowance keeps them at most `lowfi_share` of the screens' GPU-hours (the budget's
screens x the mean full run's GPU-hours), so an arm's GPU-hours stay within (1 + lowfi_share) x those of a search of
full screens only; when the GPU-hour budget binds before the screens do, the screens' GPU-hours are the part of it that
keeps that ratio (GPU-hours / (1 + lowfi_share)), so that the full runs keep the rest. A strategy with a code track
splits the allowance in two pools (CAND_r1 C8): its code proofs' probes have `code_reserve` of it, at least one probe's
cost, charged with their own runs only (`code_cids`); the knob side's probes and short runs have the rest - so the knob
side cannot starve the code track, and the two together stay within the allowance. When neither pool can fund a probe
alone but the two remainders together can, the other pool's stranded remainder (below one probe) moves over once
(`moved`, logged in `transfers`): the total allowance is unchanged (CAND_r2 1). `account(engine)` is what a strategy's
report shows.
"""
from __future__ import annotations

import math

PROBE_BELOW = 0.7               # a configuration below this feasibility is probed first
PROMOTE_Z = 1.0                 # a short run is promoted when its gain + z x its spread reaches the incumbent's
LOWFI_SHARE = 0.5               # probes and short runs: at most this share of the screens' GPU-hours
CODE_RESERVE = 0.2              # ... of which this part (at least one probe) is the code proofs' pool (a code track)
NOTE = ('Probes and short runs are not counted as screens; their GPU-hours are counted in the search\'s GPU-hours '
        '(at most {share:.0%} of the screens\' GPU-hours by the low-fidelity allowance).')


def screens_budget(engine) -> int:
    return int(getattr(engine.s.budget, 'screens', 0) or 0)


def lowfi_spent(engine) -> float:
    """GPU-hours spent on probes and short runs so far."""
    return sum(r.get('gpu_hours') or 0.0 for r in engine.s.screens if r.get('fidelity') in ('probe', 'short'))


def allowance_of(engine, share: float) -> float:
    """The low-fidelity allowance in GPU-hours: `share` x the screens' GPU-hours - the budget's screens x the engine's
    mean valid full run, or, when the GPU-hour budget binds first, GPU-hours / (1 + share)."""
    screens = screens_budget(engine) * engine.costs()['full']
    gpu = getattr(engine.s.budget, 'gpu_hours', None)
    if isinstance(gpu, (int, float)) and not isinstance(gpu, bool) and gpu > 0:
        screens = min(screens, gpu / (1.0 + share))
    return share * screens


def account(engine, share: float | None = None) -> dict:
    """Runs and GPU-hours per fidelity (the session's rows), the low-fidelity share and the allowance - the numbers a
    report states next to its screens."""
    by: dict[str, dict] = {}
    for r in engine.s.screens:
        f = r.get('fidelity') or 'full'
        d = by.setdefault(f, {'runs': 0, 'gpu_hours': 0.0})
        d['runs'] += 1
        d['gpu_hours'] += r.get('gpu_hours') or 0.0
    total = sum(d['gpu_hours'] for d in by.values())
    low = sum(by.get(f, {}).get('gpu_hours', 0.0) for f in ('probe', 'short'))
    out = {'by_fidelity': {f: {'runs': d['runs'], 'gpu_hours': round(d['gpu_hours'], 4)} for f, d in sorted(by.items())},
           'lowfi_gpu_hours': round(low, 4), 'lowfi_share_of_gpu_hours': round(low / total, 3) if total else 0.0}
    if share is not None:
        out['allowance_gpu_hours'] = round(allowance_of(engine, share), 4)
        out['note'] = NOTE.format(share=share)
    return out


class RungLadder:
    """The ladder's decisions for one strategy (the module docstring); its counters are part of the strategy's report."""

    def __init__(self, *, short: bool = True, probe_below: float = PROBE_BELOW, promote_z: float = PROMOTE_Z,
                 lowfi_share: float = LOWFI_SHARE, code_reserve: float = 0.0):
        self.short, self.probe_below, self.promote_z = bool(short), float(probe_below), float(promote_z)
        self.lowfi_share = float(lowfi_share)
        self.code_reserve = min(1.0, max(0.0, float(code_reserve)))
        self.code_cids: set = set()                      # the candidates of code proofs (their runs: the code pool's)
        self.moved = 0.0                                 # allowance moved to the code pool (< 0: to the knob side)
        self.transfers: list[dict] = []                  # each stranded remainder moved over (CAND_r2 1)
        self.counts = {'probe_first': 0, 'short': 0, 'promoted': 0, 'not_promoted': 0, 'lowfi_capped': 0}
        self.decisions: list[dict] = []

    def reserve(self, engine) -> float:
        """The code proofs' pool in GPU-hours: `code_reserve` of the allowance, at least one probe's cost (0 without a
        code track)."""
        if not self.code_reserve:
            return 0.0
        a = allowance_of(engine, self.lowfi_share)
        return min(a, max(self.code_reserve * a, self.next_cost(engine, 'probe')))

    def allowance(self, engine, *, code: bool = False) -> float:
        """A pool of the low-fidelity allowance in GPU-hours: the code proofs' (`code`) or the knob side's (the rest);
        the two always sum to the allowance (a transfer moves a remainder, it never adds one)."""
        a = allowance_of(engine, self.lowfi_share)
        r = min(a, max(0.0, self.reserve(engine) + self.moved))
        return r if code else a - r

    def spent(self, engine, *, code: bool = False) -> float:
        """What a pool has spent: the code proofs' own probes and short runs (`code`), or every other one."""
        c = sum(r.get('gpu_hours') or 0.0 for r in engine.s.screens
                if r.get('fidelity') in ('probe', 'short') and r.get('candidate_id') in self.code_cids) if self.code_cids else 0.0
        return c if code else lowfi_spent(engine) - c

    @staticmethod
    def next_cost(engine, fidelity: str) -> float:
        """The GPU-hours one more run at this fidelity is expected to cost: the engine's estimate, at least the mean of
        the runs at it that passed (a run that fails at its start costs little and would make the estimate short)."""
        est = engine.costs()[fidelity]
        xs = [r.get('gpu_hours') or 0.0 for r in engine.s.screens if r.get('fidelity') == fidelity and r.get('valid')]
        return max(est, sum(xs) / len(xs)) if xs else est

    def _fund(self, engine, cost: float, *, code: bool = False) -> bool:
        """Whether a pool can pay `cost`: from its own remainder; else, when the other pool's remainder can no longer pay a
        probe of its own (stranded) and the two together can pay, that remainder moves over once (CAND_r2 1)."""
        own = self.allowance(engine, code=code) - self.spent(engine, code=code)
        if own + 1e-12 >= cost:
            return True
        if not self.code_reserve:
            return False
        other = self.allowance(engine, code=not code) - self.spent(engine, code=not code)
        if 1e-12 < other < self.next_cost(engine, 'probe') - 1e-12 and own + other + 1e-12 >= cost:
            self.moved += other if code else -other
            self.transfers.append({'to': 'code' if code else 'knobs', 'gpu_hours': round(other, 4)})
            return True
        return False

    def affordable(self, engine, fidelity: str, *, code: bool = False) -> bool:
        """Whether one more run at this fidelity ('probe' / 'short') stays within its pool of the low-fidelity allowance
        (`code`: a code proof's probe), a stranded remainder of the other pool moved over if that is what it takes."""
        return self._fund(engine, self.next_cost(engine, fidelity), code=code)

    def plan(self, engine, p_feasible: float, *, warmup: int | None = None, probe: bool | None = None,
             short: bool | None = None) -> tuple:
        """The first rungs of a configuration: ('probe', 'short') / ('short',) / ('probe', 'full') / ('full',).
        `probe`: force (True) or forbid (False) the probe; None: by `probe_below`. `short`: False skips the short rung
        (a screen the strategy wants in full); None: the ladder's. The plan's low-fidelity rungs fit the allowance
        together (the engine runs them without asking again; CAND_r1 C8): the probe first, the short run only if both do."""
        want_probe = (p_feasible < self.probe_below) if probe is None else bool(probe)
        c_probe = self.next_cost(engine, 'probe')
        do_probe = want_probe and self._fund(engine, c_probe)
        long_warm = isinstance(warmup, int) and warmup > int(getattr(engine, 'warmup', 3) or 3)
        want_short = self.short if short is None else bool(short) and self.short
        do_short = want_short and not long_warm and \
            self._fund(engine, self.next_cost(engine, 'short') + (c_probe if do_probe else 0.0))
        if want_probe and not do_probe or want_short and not long_warm and not do_short:
            self.counts['lowfi_capped'] += 1
        if do_probe:
            self.counts['probe_first'] += 1
        if do_short:
            self.counts['short'] += 1
            return ('probe', 'short') if do_probe else ('short',)
        return ('probe', 'full') if do_probe else ('full',)

    def sigma_short(self, engine) -> float:
        """A short run's spread as an estimate of the full run's log gain: the between-run noise and the per-update
        noise over the short window's updates."""
        n = max(1, int(engine.short_steps) - int(engine.warmup))
        return math.sqrt(engine.sigma_run() ** 2 + engine.sigma_step() ** 2 / n)

    def incumbent_gain(self, engine) -> float:
        inc = engine.incumbent()
        t0, ti = engine.t('c0'), engine.t(inc)
        return math.log(t0 / ti) if t0 and ti and inc != 'c0' else 0.0

    def promote(self, engine, out, *, label: str = '') -> bool:
        """After a short outcome (status 'short', gain over the reference on the same updates): the full run follows
        when gain + z x sigma_short >= the incumbent's gain. A failed short run is not promoted."""
        ok = out.status == 'short' and out.gain is not None
        y_inc = self.incumbent_gain(engine)
        s = self.sigma_short(engine)
        go = bool(ok and out.gain + self.promote_z * s >= y_inc)
        self.counts['promoted' if go else 'not_promoted'] += 1
        self.decisions.append({'what': label[:120], 'candidate': out.cid, 'status': out.status,
                               **({'short_gain_pct': round(100 * math.expm1(out.gain), 2)} if out.gain is not None else {}),
                               'incumbent_gain_pct': round(100 * math.expm1(y_inc), 2),
                               'sigma_short_pct': round(100 * s, 2), 'promoted': go})
        return go

    def describe(self, engine=None) -> dict:
        out = {'short_rung': self.short, 'probe_below': self.probe_below, 'promote_z': self.promote_z,
               'lowfi_share': self.lowfi_share, 'code_reserve': self.code_reserve, 'counts': dict(self.counts),
               'transfers': self.transfers[-10:], 'promotions': self.decisions[-20:]}
        if engine is not None:
            out['account'] = account(engine, self.lowfi_share)
        return out
