"""What the candidate strategies (esls, rpp, rpp-ctl; possible_search_algorithm/20) share on top of wcs: a subclass of
`wcs.WCS` (no edit to wcs) that keeps wcs's prior call, table, surrogate, pair table, run dominance, cliff rule and code
track, and adds the memory ledger (search/ledger.py) as the feasibility where the runs do not decide, the rung ladder
(search/ladder.py), short runs as surrogate observations (with doubled noise), and the low-fidelity account. Each
candidate overrides `propose` (one proposal per step, `uses_engine_stop = False`: its own order until the screens are
spent) and keeps wcs's `observe` for the bookkeeping.

On the final wcs (pc_code_wcs6): every model call goes through wcs's own methods (the prior in `_init`, the code track's
asks in `_code`), so wcs's one allowance of LLM_CALLS_MAX calls over every model call (`calls_left`, the re-asks
included) holds for the candidates as it does for wcs; the prior's whole configurations enter the surrogate as wide,
tagged observations (wcs's `_init`) and a short run retires those about the values it measured, as a full run does
(WCS_r2 4); the store's warnings lower the feasibility by what they say (wcs's `_warn_level`). wcs's own propose steps
(`_prior_pick`, `_rescue`, `_partner`, `_coverage`, `_acquire` with `_price`) are not used: each candidate has its own
order.

From the review CAND_r1: the ledger's identity and features hold the code patches a configuration carries (wcs's
pseudo-knobs: a run's own result never stands for the same knobs under another overlay); the ledger takes wcs's final
class of a run (its jitter rule: thrash) from wcs's own record, not a second reading; wcs's measured-only dominance (an
out-of-memory or thrashing run plus moves the runs measured to spend or be neutral) rules a configuration out before
the ledger is asked, a run's own result first; a code proof's probe-first plan runs only while the low-fidelity
allowance's code pool has a probe (with a code track: CODE_RESERVE of the allowance, at least one probe, charged with the
code proofs' own runs; the knob side has the rest).

    _pf(cfg, ids)              feasibility: the ledger (a run: its own result), else wcs's; x the pair table
    _knob_q(engine, cfg, ...)  a proposal of a whole configuration with a plan of rungs
    _evidence(engine, p, out)  the run's memory evidence into the ledger (peak / out of memory / thrash)
"""
from __future__ import annotations

import math
import statistics

from .engine import Outcome, Proposal
from .ladder import CODE_RESERVE, LOWFI_SHARE, RungLadder, account
from .ledger import for_wcs, vk                  # (wcs's key form, memoised: the same strings as wcs.vk)
from .wcs import P_ALONE_FAILED, P_UNSURE, SD_UNKNOWN, WCS, Surrogate


class CandidateBase(WCS):
    uses_engine_stop = False

    def __init__(self, proposer, *, off_wcs=(), ledger: bool = True, short: bool = True, lowfi_share: float = LOWFI_SHARE,
                 facts=None, accounting=None, experience=None, kappa0: float = 0.5, seed: int = 42):
        super().__init__(proposer, off=list(off_wcs), facts=facts, accounting=accounting, experience=experience,
                         kappa0=kappa0, seed=seed)
        self.use_ledger = bool(ledger)
        self.mledger = None
        self.ladder = RungLadder(short=short, lowfi_share=lowfi_share,      # (a code track: part held for its proofs)
                                 code_reserve=CODE_RESERVE if self.on['wcs_code'] else 0.0)
        self.pending: list[Proposal] = []                # proposals waiting for their turn (promotions, ledges, ...)
        self.engine = None                               # (for the report's low-fidelity account)
        self.lowfi_share = float(lowfi_share)
        self.screens_by: dict[str, int] = {}             # full screens by purpose (the pick's kind)
        self._cls_memo: dict = {}                        # the outcome being observed -> its class (wcs's one reading)
        self._dom: dict = {}                             # measured dominance by configuration, between observations
        self._dom_stamp = None
        self._dom_idx: dict | None = None                # its lookup data, built once per observation (CAND_r2 2)

    # ------------------------------------------------------------ set-up
    def _init(self, engine) -> None:
        super()._init(engine)
        self.mledger = for_wcs(self) if self.use_ledger else None

    def _left(self, engine) -> int:
        return int(getattr(engine.s.budget, 'screens', 0) or 0) - engine.full_screens()

    # ------------------------------------------------------------ feasibility
    def _pf(self, cfg: dict, ids=()) -> tuple[float, str]:
        """P(the configuration fits): the ledger (a configuration that ran: its own result; the others by the censored
        model, which every run's peak, out-of-memory bound and cliff has updated) - without it, wcs's feasibility (the
        runs' dominance, the declared arithmetic); the experience store's warning lowers it by what it says (wcs's
        `_warn_level`); the pair table and the values that failed alone multiply it (as wcs)."""
        if self.mledger is None:
            return self._p_feasible(cfg, ids)
        m = self._m_of(cfg, ids)
        p, why = 1.0, ''
        if self.on['wcs_feasibility']:
            kcfg = {k: cfg.get(k) for k in self.knobs}
            lids = self._led_ids(ids)
            own = self.mledger.ran_status(kcfg, ids=lids)  # a run's own result (same knobs, same patches) first; then
            out = self._measured_out(m) if own is None else None    # wcs's measured dominance (CAND_r1 C5)
            if out:
                return 0.0, out
            p = self.mledger.p_feasible(kcfg, ids=lids)
            if self.mledger.cap:
                mu, sd = self.mledger.predict(kcfg, ids=lids)
                why = f'the ledger: {mu:.1f} +- {sd:.1f} GiB per GPU of {self.mledger.cap:.0f}'
            if p > P_UNSURE and own is None:             # (a run's own evidence outranks the store)
                warn = self._warned(kcfg)
                if warn:                                 # (wcs's rule: as low as what the warning says, _warn_level)
                    p, why = min(p, self._warn_level(warn)), (why + '; ' if why else '') + f'the store warns: {str(warn[0])[:120]}'
        if self.on['wcs_antagonism']:
            present = {(k, vk(v)) for k, v in m.items() if k.startswith('patch:') or vk(v) != vk(self.ref.get(k))}
            for pair, rec in self.pairs.items():
                if pair <= present and rec['p'] < 1.0:
                    p *= rec['p']
                    why = (why + '; ' if why else '') + f'{self._pair_words(pair)}: {rec["why"]}'
            for key, n in self.alone.items():
                if key in present and n:
                    p *= P_ALONE_FAILED if n == 1 else 0.0
                    why = (why + '; ' if why else '') + f'{key[0]}={key[1]} failed to start alone {n}x'
        return p, why

    def _ckey(self, cfg: dict, ids) -> tuple:
        """wcs's configuration key, its values' keys memoised (the same tuple as wcs's; a draw scores the whole table)."""
        return tuple(sorted((k, vk(cfg.get(k))) for k in self.knobs)) + tuple(sorted(f'patch:{x}' for x in ids))

    def _led_ids(self, ids) -> tuple:
        """The code patches a configuration carries as the ledger's pseudo-knobs (wcs's patch:<id>), each added once with
        the memory its idea declared (CAND_r1 C3)."""
        ids = tuple(sorted({str(x) for x in ids or ()}))
        if self.mledger is not None:
            for pid in ids:
                if pid not in self.mledger.patch_ids:
                    rec = self.patches.get(pid) or {}
                    self.mledger.add_patch(pid, rec.get('mem_dir'), rec.get('mem_gib'))
        return ids

    def _dom_index(self) -> dict:
        """What the measured-only dominance needs, built once per observation (CAND_r2 2): the keys some measured move
        names (`hk`) - a run can dominate a configuration only if every key they differ in is one of them, since wcs's
        split reads any other change as unknown - every other key (`other`), and the failed and fitted runs by their
        values on those other keys."""
        stamp = (len(self.mem.runs), len(self.mem.measured))
        if self._dom_idx is not None and self._dom_stamp == stamp:
            return self._dom_idx
        hk = {k for move in self.mem.measured for k, _, _ in move}
        keys = set(self.knobs)
        for r in self.mem.runs.values():
            keys |= set(r['cfg'])
        other = tuple(sorted(keys - hk))
        failed: dict = {}
        fitted: dict = {}
        for cid, r in self.mem.runs.items():
            proj = tuple(vk(r['cfg'].get(k)) for k in other)
            if r['status'] in ('oom', 'thrash'):
                failed.setdefault(proj, []).append((cid, r['cfg']))
            elif r['status'] == 'ok':
                fitted.setdefault(proj, []).append(r['cfg'])
        self._dom, self._dom_stamp = {}, stamp
        self._dom_idx = {'hk': hk, 'hks': tuple(sorted(hk)), 'keys': keys, 'other': other, 'failed': failed,
                         'fitted': fitted}
        return self._dom_idx

    def _measured_out(self, m: dict) -> str | None:
        """wcs's measured-only dominance (MemoryModel.predict with hard = the moves the runs measured): the out-of-memory
        or thrashing run whose memory this configuration needs at least - unless a run that fitted needs at least as
        much too (the two disagree: wcs's 'unsure', left to the ledger). None when no run rules it out. Only the runs
        that agree with it outside the measured keys are compared (wcs's own `dominates`), from the index built once
        per observation (CAND_r1 C5, CAND_r2 2)."""
        idx = self._dom_index()
        if not idx['failed']:
            return None
        if any(k not in idx['keys'] and k not in idx['hk'] and vk(v) != 'null' for k, v in m.items()):
            return None                                  # (a key no run holds and no measured move names: unknown)
        proj = tuple(vk(m.get(k)) for k in idx['other'])
        cands = idx['failed'].get(proj)
        if not cands:
            return None
        key = (proj, tuple(vk(m.get(k)) for k in idx['hks']))
        if key in self._dom:
            return self._dom[key]
        hard = self.mem.measured
        bad = next((cid for cid, cfg in cands if self.mem.dominates(m, cfg, hard)), None)
        if bad is not None and any(self.mem.dominates(cfg, m, hard) for cfg in idx['fitted'].get(proj, [])):
            bad = None
        out = None if bad is None else (f'it needs at least the memory of {bad} ({self.mem.runs[bad]["status"]}) by '
                                        'moves the runs measured')
        self._dom[key] = out
        return out

    def _lowfi_ok(self, engine, q) -> bool:
        """A proposal whose plan starts with a probe (a code proof's) runs only while the allowance's code pool has a
        probe (CAND_r1 C8)."""
        return tuple(q.fidelity or ('full',))[0] != 'probe' or self.ladder.affordable(engine, 'probe', code=True)

    def _warmup(self, cfg: dict) -> int:
        """The longest warm-up the prior declared for the configuration's values (0: none)."""
        return max([e.get('warmup') or 0 for (k, key), e in (self.prior.levels.items() if self.prior else [])
                    if vk(cfg.get(k)) == key and vk(cfg.get(k)) != vk(self.ref.get(k))] + [0])

    def _mu(self, engine, inc, cfg: dict, ids=()) -> tuple[float, float]:
        """The surrogate's mean gain over the incumbent and its spread."""
        mean = self.surr.mean()
        x = self._feats(cfg, ids)
        mu = Surrogate.value(x, mean) - Surrogate.value(self._incumbent_feats(engine, inc), mean)
        return mu, math.sqrt(max(1e-6, self.surr.var(x)))

    def _knob_q(self, engine, inc, cfg: dict, *, kind: str, plan: tuple, p: float | None = None, why: str = '',
                origin: str | None = None) -> Proposal:
        """A proposal of a whole configuration (knob values; the proven patch chain rides on it) with a plan of rungs:
        ('probe',) a feasibility probe (no full run), ('probe', 'short'), ('short',), ('probe', 'full'), ('full',)."""
        ids = tuple(self.chain)
        if p is None:
            p, why = self._pf(cfg, ids)
        mu, sigma = self._mu(engine, inc, cfg, ids)
        q = self._proposal(engine, inc, cfg, ids, mu=mu, sigma=sigma, p=p, why=why, probe=plan[0] == 'probe',
                           origin=origin or self.name, kind=kind)
        q.fidelity = tuple(plan)
        if plan[0] != 'probe':
            q.gate, q.probe_cap, q.probe_steps, q.keep_probe = None, None, None, False
        else:
            q.keep_probe = True                          # (the candidates' probes are the point, never dropped)
        self.ledger[q.hypothesis]['plan'] = list(plan)
        return q

    # ------------------------------------------------------------ outcomes
    @staticmethod
    def _peak_of(engine, out) -> float | None:
        """The run's measured peak (D3's harness field first, then the log's) at its own fidelity."""
        c = engine.s.candidates.get(out.cid) if out.cid else None
        if c is None:
            return None
        if out.fidelity == 'full' or out.status in ('screened', 'aborted'):
            ev = ((c.receipt or {}).get('evidence') or {}) if isinstance(c.receipt, dict) else {}
            peak = ev.get('memory_peak_measured_gib') or ev.get('memory_peak_gib') or (c.verdict or {}).get('peak_reserved_gib')
        else:
            res = (c.results or {}).get(out.fidelity) or (c.results or {}).get('probe') or {}
            peak = res.get('memory_peak_measured_gib') or res.get('memory_peak_gib')
        return float(peak) if isinstance(peak, (int, float)) and peak > 0 else None

    def _cfg_of(self, p) -> dict:
        return {k: (p.change or {}).get(k, self.ref.get(k)) for k in self.knobs} if p.absolute else dict(self.ref)

    def classify(self, engine, p, out) -> dict:
        """wcs's (hls3's) failure classifier, read once per outcome: wcs's observe and the ledger see the same class."""
        key = (id(p), out.cid, out.status, out.fidelity)
        got = self._cls_memo.get(key)
        if got is None:
            got = WCS.classify(self, engine, p, out)
            self._cls_memo = {key: got}
        return got

    def _evidence(self, engine, p, out, kind: str | None = None) -> dict | None:
        """The run's memory evidence into the ledger: the configuration that ran and its patches as wcs recorded them,
        wcs's final class (after its jitter rule: thrash; CAND_r1 C4) unless the strategy reads it as `kind`, the peak at
        the run's fidelity; the class."""
        if out.status in ('infra', 'duplicate', 'rejected') or not out.cid:
            return None
        cls = self.classify(engine, p, out)
        if self.mledger is not None:
            run = self.runs.get(out.cid) or {}
            cfg_m = run.get('cfg') or (self._m_of(self._cfg_of(p), self._patch_ideas(p)) if p.absolute else None)
            if cfg_m is not None:
                k = kind or run.get('kind') or cls['kind']
                peak = self._peak_of(engine, out) or cls.get('peak_gib')
                ids = self._led_ids(x[len('patch:'):] for x, v in cfg_m.items()
                                    if isinstance(x, str) and x.startswith('patch:') and v)
                self.mledger.observe_run({kk: cfg_m.get(kk) for kk in self.knobs}, k, peak=peak, oom=cls.get('oom'),
                                         source=out.cid, ids=ids)
        return cls

    def observe(self, engine, p, out: Outcome):
        self._dom, self._dom_idx = {}, None
        super().observe(engine, p, out)
        self.engine = engine
        if (p.hypothesis or '').startswith('patch|') and out.cid:
            self.ladder.code_cids.add(out.cid)           # (a code proof's runs: the code pool's)
        cls = self._evidence(engine, p, out)
        if out.status == 'short' and out.gain is not None and p.absolute:
            ids = tuple(self._patch_ideas(p))
            feats = self._feats(self._cfg_of(p), ids)
            measured = set(feats)                        # a prior suggestion about a value now measured is superseded
            self.surr.retire(lambda tag, fs, _m=measured: tag[0] == 'config' and bool(set(fs) & _m))   # (as wcs: WCS_r2 4)
            self.surr.observe(feats, out.gain, 2 * self.ladder.sigma_short(engine))
        if out.status == 'screened':
            kind = (p.hypothesis or '').split('|', 1)[0]
            self.screens_by[kind] = self.screens_by.get(kind, 0) + 1
        return cls

    # ------------------------------------------------------------ report
    def candidate_view(self) -> dict:
        """wcs's account (the prior, the table, the picks, the runs, the pair table, the memory model, the patches) with
        the ledger, the ladder and the low-fidelity account."""
        d = WCS.describe(self)
        w = d.get('wcs') or {}
        out = {'wcs_off': w.get('off'), 'prior': w.get('prior'), 'table': w.get('table'), 'counts': w.get('counts'),
               'llm_calls': w.get('llm_calls'), 'llm_calls_max': w.get('llm_calls_max'), 'picks': w.get('picks'),
               'runs': w.get('runs'), 'failures': w.get('failures'),
               'pairs': w.get('pairs'), 'patches': w.get('patches'), 'chain': w.get('chain'), 'inert': w.get('inert'),
               'from_store': w.get('from_store'), 'posterior': w.get('posterior'),
               'ledger': self.mledger.describe() if self.mledger is not None else None,
               'ladder': self.ladder.describe(self.engine) if self.engine is not None else self.ladder.describe(),
               'screens_by_purpose': dict(self.screens_by)}
        return out, d.get('learned'), d.get('proposer')


def median_sd(levels) -> float:
    sds = [e['sd'] for e in (levels or {}).values() if isinstance(e.get('sd'), (int, float)) and e['sd'] > 0]
    return statistics.median(sds) if sds else SD_UNKNOWN
