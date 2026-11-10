"""Search strategies over the shared SearchSession (revamp design C). A strategy run is fn(ctx) -> SearchTrace, where
ctx holds the task, budget, start config, runner, verifier (judge), hooks, start/reference receipts, out_dir, run_id and
an optional LLM driver. Every strategy produces the same trace format, so they can be compared or combined.

    random, tpe          knob-space baselines (no LLM)
    hls                  headroom-ledger search on the Engine: LLM hypotheses as priors, calibrated, value/cost scheduling,
                         multi-fidelity racing, numeric stop (without an LLM: neighbourhood exploration only)
    hls2                 hls with the search fix (FABLE_DESIGN_SEARCH_FIX, 2026-10-29): untested knob ideas follow the
                         incumbent, the model sees what is still queued, judged moves get honest priors and their turn,
                         kappa per origin, a value that fails to start twice is not proposed again
    hls3                 hls2 with the general strategies of FABLE_TAKEAWAYS_20261030 (search/hls3.py): memory as a
                         budget (feasibility from the runs, kept probes), spenders bundled with enablers, failures as
                         information, stall probes and whole-configuration steps, re-basing, phase re-ranking, code sites,
                         median decisions, card-size pruning; each a component `ablate` switches off (all off: hls2)
    doe                  Plackett-Burman screening of the two-level knobs at short fidelity, then their best combination
    doe+hls              the screening design first, then the ledger (default with an LLM when the knobs are mostly
                         two-level: at least four, and at least three quarters of the schema; otherwise hls)
    greedy               the old agent behaviour on the Engine (best prediction first, finishes when nothing clears
                         the noise) - a baseline
    agent                the original one-action-per-turn LLM loop (agent.optimize)
    optico               OptiCo as a baseline: its four LLM agents and message pool over the knobs (search/optico.py,
                         baselines.optico_style; its prompts are read at run time, INFRARSI_OPTICO_ROOT)
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

from ..agent.search import SearchSession, SearchTrace
from . import knob_space

# research switches (`search.ablate` / `run --ablate`): the components an ablation turns off; the product runs all of them
ABLATIONS = {'probe_first': 'DOE probes each factor (2 updates) before its screening design',
             'accounting': "the closed-form memory / MFU accounting: the proposer's field and the memory warning where "
                           'the experience has no measured match',
             'bandwidth': "a multi-node reference / doctor run's all-reduce bandwidth benchmark",
             'profile': 'the one-off profiled run that gives a job without logged phases its phases'}
# hls3's components (search/hls3.py; FABLE_TAKEAWAYS_20261030 section 4): name -> (strategy, what turning it off removes)
HLS3_COMPONENTS = {
    'hls3_feasibility': ('S-1', 'the memory feasibility model: a candidate that needs at least what an out-of-memory run '
                                'needed by measured moves is deferred, one bounded only by declarations or nearly bounded '
                                'probed first, probes kept while screens bind'),
    'hls3_memory_schema': ('S-1', "the proposer's declared memory direction and GiB of each move (they order and build, "
                                  'at most ask for a probe)'),
    'hls3_bundles': ('S-2', 'a memory-blocked spender proposed together with the best freeing moves'),
    'hls3_failures': ('S-3', 'failure classes (OOM, compiler, collective, start cap, thrash ...) and their summary for '
                             'the proposer; thrash (a measured peak near the card, or an out-of-memory bound) read as the '
                             'memory edge, not a slow move'),
    'hls3_antagonism': ('S-3', 'changes that failed together run separately before they are combined again'),
    'hls3_stall_probes': ('S-4', "at a stall, feasibility probes of the blocked spender with each candidate enabler"),
    'hls3_rebase': ('S-5', "after an incumbent change, the strongest moves rejected earlier queued on it again"),
    'hls3_explore': ('S-6', 'at a stall, one whole-configuration step (the experience composite or the proposer\'s best '
                            'moves, made coherent)'),
    'hls3_phases': ('S-7', "the incumbent's open moves re-ranked by its own phase split"),
    'hls3_sites': ('S-8', 'the code prompts ask for the site first and the gain per changed line'),
    'hls3_variance': ('S-9', 'median decisions (incumbent, early abort, gains) and stall removers judged by the mean'),
    'hls3_prune': ('S-10', 'configurations whose training state alone is above the card (the shared accounting, while '
                           'every run that fitted agrees) are not proposed, values a declaration puts above the sharded-'
                           "state floor are probed first, and the proposer's accounting shards the states"),
    # FABLE_HLS3_FIXES_R1 (the first hls3 arms): one component per fix
    'hls3_start_cap': ('F1', "a probe stopped at 4x its base's time to the first update (start_slow: not a start failure), "
                             'kept for an unsure, memory-spending or start-risky move while the GPU-hours and the wall left '
                             'pay every screen left (a start-risky one also when its expected saving pays for it), priced at '
                             'its cap'),
    'hls3_probe_gate': ('F1', "no full run after a probe 2x slower than its base's first updates on a memory-spending "
                              'configuration (thrash only with a measured near-cap peak or an out-of-memory bound)'),
    'hls3_coverage': ('F2', 'the compatible start-safe knob moves below 2 sigma_pair that are predicted to help and spend no '
                            'memory as one bundle, probed first, not screened singly while it is proposed'),
    'hls3_need': ('F3', "a spender's next bundle frees more than every failed one; re-based only after a freeing move as "
                        'big; no unknown-direction enabler in a bundle; at most three tries per spender'),
    'hls3_complete': ('F4', "a change to part of a workload invariant's knobs completed with the others"),
    'hls3_pairs': ('F5', 'every pair of a small set that failed to start is suspect: probed first, deferred after a second '
                         'failure; a value whose every failed start had a partner is probed alone before C5 rules it out; a '
                         'proposal blocked by an incumbent member is a swap'),
    'hls3_remainder': ('F6', 'the part of a multi-knob entry the incumbent did not take carried to it'),
    'hls3_warn': ('F7', "a memory warning of the store or the estimate caps feasibility only where hls3's runs bound it"),
    'hls3_dead_patch': ('F8', 'the same fatal start line (an exception) twice rules out the new elements both runs shared, '
                              "their common code lines included; an inert patch reported and its site's next patch without a "
                              'self-check line waits'),
    # FABLE_HLS3_FIXES_R2_Q9H_M35H2 (the Q9H and M35H2 arms): one component per fix
    'hls3_self_proof': ('F10', 'a patch proves at its probe that it took effect - its `found -> set` and `active` lines, '
                               'found != set, the job-log note it targets gone - else inert at start: no full run, the '
                               'reason back to the proposer; post-hoc inert below max(sigma_pair, kappa x mu / 4)'),
    'hls3_fear_prior': ('F11', 'a declared memory spender no run bounds is ranked by the most this run declared for it and, '
                               'once a feared move (a declared spender below the probe rule) ran, the share of them that '
                               "fitted (Beta(1, 1)), probed first; the ledger keeps the proposer's p"),
    'hls3_gate_low': ('F12', "the probe speed gate at 1.6x its base's first updates (2.0 without), the proposal's own "
                             'declared patch spend counted, a change of unknown memory direction a possible spend'),
    # FABLE_HLS3_FIXES_R3 (the G4U and Q9H -h5 arms): one component per fix
    'hls3_scan_whole': ('F13', "F10 reads a patch's self-proof lines whole (not cut at 300 characters; a site with '::' "
                               'kept), and its feedback shows the found / set pair it read'),
    'hls3_parse': ('F14', "the proposer's hypothesis answer read robustly: the text blocks of a reply split over the "
                          "CLI's resume turns joined, each hypothesis validated on its own (a bad one dropped and reported, "
                          'non-schema `edits` taken from the `patch` text), retries that name the exact validation errors'),
    'hls3_seed_mu': ('F15', 'an experience seed with no prior that ran out of memory or thrashed is a spender at the most '
                            "this run predicted for the move (the transfer judgment's, the proposer's), else at 2 sigma_pair "
                            '(spender identification only; the ledger keeps its zero prior)'),
    'hls3_pair_probes': ('F16', 'a memory-blocked spender no freeing bundle covers is probed with each other member of a '
                                'composite declared to run that died at its start, else with up to two moves of no '
                                'measured memory effect (moves slow alone included): undeclared enablers'),
    'hls3_cov_unresolved': ('F17', 'the members of a coverage bundle measured within 2 sigma_pair are resolved by it '
                                   '(unresolved, not screened one by one) unless the proposer predicts one again above '
                                   '2 sigma_pair'),
    'hls3_carry_patches': ('F18', "an open patch entry follows a knob-only change of the incumbent (the same code)"),
    'hls3_pair_set': ('F19', "pairs are suspect in a failed start of up to the transfer judgment's start composite (at "
                             'most 5 changes); a pair probe is exempt from the pair it tests'),
    'hls3_deferred': ('F21', "a patch's third proof line, `deferred (<reason>)`, and `active` printed after the new path "
                             'ran: a replacement that hands calls back to the original is reported as such'),
    'hls3_jitter_thrash': ('F23', 'a slower run of a memory-spending configuration whose update intervals scatter (MAD / '
                                  "median at max(2 %, 5x its base's)) is thrash - the memory edge - without a peak"),
    'hls3_shared_spender': ('F22', 'the change every out-of-memory run on a base shared (two runs or more, none of it in a '
                                   'run that fitted) is a spender of its own: its need and tries counted as F3 counts them'),
    # the user's decisions of round 3 (D3, D4, D7)
    'hls3_peaks': ('F9a', "each run's peak memory as the harness measured it (every process of every node, D3) read by "
                          "the memory model, F23's thrash rule and F16's enablers"),
    'hls3_writeback': ('F9b', 'what the run learned of its starts that a later arm can start from - the sets that failed '
                              'to start together (with what each did alone), the pairs two or more failed starts held, '
                              'the values that failed alone - as the strategy-neutral `learned` section written back '
                              'into its experience pack, and read back from a shared store by a later arm (D4)'),
    'hls3_library_source': ('D7', "read-only excerpts of the library source the job imports, at the names its job-log "
                                  'notes and its patch sites give (at most 400 lines), in the code prompts')}
ABLATIONS.update({k: f'hls3 {s}: {d}' for k, (s, d) in HLS3_COMPONENTS.items()})
HLS3_R1 = tuple(k for k, (s, _) in HLS3_COMPONENTS.items() if s in {f'F{i}' for i in range(1, 9)})   # round-1 fixes
HLS3_R2 = tuple(k for k, (s, _) in HLS3_COMPONENTS.items() if s in ('F10', 'F11', 'F12'))          # round-2 fixes
HLS3_R3 = tuple(k for k, (s, _) in HLS3_COMPONENTS.items()                                       # round-3 fixes
                if s in ('F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F21', 'F22', 'F23', 'F9a', 'F9b', 'D7'))
# wcs's components (search/wcs.py; FABLE_ARCH_REVIEW_20261031 section 4): name -> what turning it off removes
WCS_COMPONENTS = {
    'wcs_prior': "the model's one-call prior over the knob table (each value's gain, memory direction and GiB, warm-up, "
                 'the pairs that enable or conflict, up to four whole configurations); off: a flat prior, no declarations',
    'wcs_feasibility': 'the memory feasibility of a whole configuration: the sharded-state arithmetic, the declared memory '
                       'against the measured peak, dominance by the runs that fitted or ran out of memory (MemoryModel), '
                       'the cliff read from scattered update times; off: every configuration is taken to fit',
    'wcs_antagonism': 'the pair table: values that failed to start together are not combined again until one of them ran '
                      'alone or a run holding both was valid; declared conflicts start suspect; the start failures earlier '
                      'arms wrote back (D4) enter it',
    'wcs_thompson': 'Thompson sampling of the surrogate (one joint draw of its posterior picks the next configuration); off: '
                    'the posterior mean (greedy expected improvement)',
    'wcs_diversity': 'the first three full screens differ from every run so far in at least two knobs the prior calls big',
    'wcs_partner_probe': 'a spender the runs block gets one probe together with each knob of unknown memory direction that '
                         'the prior does not call harmful (an undeclared enabler)',
    'wcs_probe_first': 'a configuration whose feasibility is below 0.7 is probed first (kept while the screens are the scarcer '
                       'budget, the start cap, the speed gate with the compile warm-up guard); off: full runs only',
    'wcs_code': 'the code track: the model asked for patches only (at most three per ask, at most three asks) when two or '
                'more screens are left and no patch is pending; a patch proves at its probe that it took effect; a proven '
                'patch is a pseudo-knob tested on the best feasible configuration',
    'wcs_coverage': 'the last screen as one bundle of every value the posterior rates positive on the incumbent'}
ABLATIONS.update({k: f'wcs: {d}' for k, d in WCS_COMPONENTS.items()})
# the candidates (search/esls.py, search/rpp.py; possible_search_algorithm/20): name -> what turning it off removes
ESLS_COMPONENTS = {
    'esls_prior': "the model's one prior call (wcs's), asking also for memory terms and each value's family; off: a flat "
                  'prior, no declarations',
    'esls_ledger': 'the censored memory ledger (search/ledger) as the feasibility where the runs do not decide; off: '
                   "wcs's (dominance, arithmetic, pairs) - the arm esls-noledger",
    'esls_enabler_probe': 'one probe per enabler on the reference measures what it frees; off: its declaration or the '
                          "ledger's estimate - the arm esls-noprobe",
    'esls_ledge': 'a boundary screen slower than predicted near the card, or failing, gets one screen a level down',
    'esls_lateral': "one other composition at the boundary's memory, predicted faster, through the ladder",
    'esls_tail': 'the small knobs rated positive as one bundle on the incumbent, a bounded bisection after a regression',
    'esls_ladder': 'the short rung for the lateral, hedge and fill draws (probe -> short -> full); off: probe -> full',
    'esls_hedge': "one feasibility-weighted random draw when the lattice's screens all landed within 2 sigma_pair",
    'esls_code': "wcs's code track (patches only, the self-proof, proven patches as pseudo-knobs)"}
RPP_COMPONENTS = {
    'rpp_prior': "the model's prior call (wcs's): the time model's means; off: a flat prior",
    'rpp_ledger': "the censored memory ledger as the feasibility where the runs do not decide; off: wcs's",
    'rpp_anneal': 'the draw weighted by exp(mu / T), T annealed; off: T = inf (feasibility only)',
    'rpp_ladder': 'the short rung (probe -> short -> full); off: probe first, then full',
    'rpp_code': "wcs's code track (rpp: on; rpp-ctl: off)"}
ABLATIONS.update({k: f'esls: {d}' for k, d in ESLS_COMPONENTS.items()})
ABLATIONS.update({k: f'rpp: {d}' for k, d in RPP_COMPONENTS.items()})


@dataclasses.dataclass
class Context:
    task: object
    budget: object
    start_config: dict
    runner: object
    judge: object
    hooks: object
    run_id: str
    out_dir: Path
    start_receipt: dict | None = None
    reference_receipt: dict | None = None
    driver: object = None
    package: object = None
    seed: int = 42
    options: dict = dataclasses.field(default_factory=dict)
    sessions: list = dataclasses.field(default_factory=list)   # the strategy's session: an interrupted run reports it

    def session(self, arm: str, **kw) -> SearchSession:
        # a resumed run (project/resume.py) builds its session without the directory, so that building it writes nothing
        # to the events of the run it continues; the resume sets the directory once the saved state is on it
        s = SearchSession(task=self.task, budget=self.budget, runner=self.runner, judge=self.judge, run_id=self.run_id,
                          arm=arm, out_dir=None if self.options.get('resume') else self.out_dir,
                          start_config=self.start_config, start_receipt=self.start_receipt,
                          reference_receipt=self.reference_receipt, hooks=self.hooks,
                          driver_info=getattr(self.driver, 'describe', lambda: None)() if self.driver else None, **kw)
        s.progress = self.options.get('progress')     # the project flow's log: a line per finished run (U-1-14)
        s.control = self.options.get('control')
        s.on_interrupt = self.options.get('on_interrupt')   # the project flow's policy for Ctrl-C / SIGTERM
        self.sessions.append(s)
        return s


def _knob(name, fn):
    def run(ctx: Context) -> SearchTrace:
        s = ctx.session(name)
        stop = fn(s, seed=ctx.seed)
        return s.finalize(stop_reason=stop, extra={'strategy': name, 'seed': ctx.seed})
    return run


def _agent(ctx: Context) -> SearchTrace:
    from ..agent.optimize import optimize
    if ctx.driver is None:
        raise ValueError('the agent strategy needs an LLM backend (manifest llm.backend or --llm)')
    # with the debate on, the agent's reviewer is the model that did not write the candidate (or the explicit verifier)
    checker = ctx.driver.checker if getattr(ctx.driver, 'rounds', 0) else None
    return optimize(ctx.task, ctx.package, ctx.driver, ctx.budget, ctx.start_config, ctx.runner, judge=ctx.judge,
                    run_id=ctx.run_id, arm='agent', out_dir=ctx.out_dir, start_receipt=ctx.start_receipt,
                    reference_receipt=ctx.reference_receipt, hooks=ctx.hooks, capability=ctx.options.get('capability', 0.6),
                    sessions=ctx.sessions, progress=ctx.options.get('progress'), checker=checker, control=ctx.options.get('control'))


def _optico(ctx: Context) -> SearchTrace:
    from .optico import run as run_optico
    return run_optico(ctx)


def _engine(name):
    def run(ctx: Context) -> SearchTrace:
        from .engine import Engine
        from .proposers import LLMProposer
        from .strategies import DOE, HLS, HLS2, Chain, Greedy
        if name == 'greedy' and ctx.driver is None:
            raise ValueError('the greedy strategy needs an LLM backend (manifest llm.backend or --llm)')
        from ..agent.capability import mode_for, policy_overrides, search_params
        o = ctx.options
        c = o.get('capability', 0.6)
        kp = search_params(c)
        exp = o.get('experience')                      # experience.store.plan for this job, or None
        ablate = set(o.get('ablate') or ())             # research switches: components turned off for an ablation
        extra = {'ledger_states': True} if name in ('hls2', 'hls3') else {}   # the prompt explains hls2's ledger states
        acc, h3 = o.get('accounting'), {}
        if name == 'hls3':                             # hls3: the memory facts, sharded states
            from .hls3 import HLS3, sharded_accounting
            facts = getattr(ctx.hooks, 'memory_facts', lambda: None)() or {}
            acc = acc if 'hls3_prune' in ablate else sharded_accounting(acc, facts)
            h3 = {'off': sorted(x for x in ablate if x in HLS3_COMPONENTS), 'facts': facts,
                  'accounting': o.get('accounting')}          # (S-10's state arithmetic: the shared one, unwrapped)
        ledger = HLS3 if name == 'hls3' else HLS2 if name == 'hls2' else HLS
        if name == 'wcs':                              # wcs: the whole-configuration search (its own class, its features)
            from .wcs import WCS
            ledger = WCS
            facts = getattr(ctx.hooks, 'memory_facts', lambda: None)() or {}
        if name in ('esls', 'rpp', 'rpp-ctl'):         # the candidates: on wcs, their own classes and features
            from .esls import ESLS
            from .rpp import RPP
            ledger = ESLS if name == 'esls' else RPP
            facts = getattr(ctx.hooks, 'memory_facts', lambda: None)() or {}
            if name == 'rpp-ctl':                      # the control: no model call, no experience store, no code
                ablate = set(ablate) | {'rpp_prior', 'rpp_anneal', 'rpp_code'}
        prop = LLMProposer(ctx.driver, ctx.hooks, max_hypotheses=kp['max_hypotheses'], menu=kp['menu'],
                           sigma_scale=kp['sigma_scale'], prior=o.get('prior_results'), experience=(exp or {}).get('payload'),
                           search_calibrates=name in ('hls', 'hls2', 'hls3', 'doe+hls', 'wcs', 'esls', 'rpp', 'rpp-ctl'),
                           accounting=acc, features=ledger.proposer_features(ablate),   # (what the strategy opts in to:
                           **extra)                                                     # none for hls / hls2)
        hls = lambda: ledger(prop, kappa0=kp['kappa0'], probe_below=kp['probe_below'], experience=exp,   # noqa: E731
                             phase_cap=bool(o.get('phase_cap')), **h3)
        doe = lambda: DOE(prop, experience=exp, probe_first='probe_first' not in ablate)                  # noqa: E731
        wcs = lambda: ledger(prop, off=sorted(x for x in ablate if x in WCS_COMPONENTS), facts=facts,    # noqa: E731
                             accounting=acc, experience=exp, kappa0=kp['kappa0'], seed=ctx.seed)
        cand = lambda: ledger(prop, off=sorted(x for x in ablate if x in ESLS_COMPONENTS), facts=facts,  # noqa: E731
                              accounting=acc, experience=exp, kappa0=kp['kappa0'], seed=ctx.seed) if name == 'esls' else \
            ledger(prop, off=sorted(x for x in ablate if x in RPP_COMPONENTS), name=name, a=1.0,
                   temperature=float('inf') if name == 'rpp-ctl' else None, facts=facts, accounting=acc,
                   experience=None if name == 'rpp-ctl' else exp, kappa0=kp['kappa0'], seed=ctx.seed)
        strategy = {'hls': hls, 'hls2': hls, 'hls3': hls, 'doe': doe, 'doe+hls': lambda: Chain(doe(), hls()),
                    'greedy': lambda: Greedy(prop), 'wcs': wcs, 'esls': cand, 'rpp': cand, 'rpp-ctl': cand}[name]()
        session = ctx.session(o.get('arm') or name, stop_after_same_failures=10 ** 6, mode=mode_for(c),
                              patch_line_limit=policy_overrides(c)['patch_line_limit'],
                              layers=tuple(getattr(ctx.hooks, 'layers', None) or ('L0', 'L1')),
                              count_rescores=bool(o.get('count_rescores')))
        if ctx.start_receipt is None:                  # no reference run before the search (an environment arm, as
            session.screen('c0')                       # B-IS / B-RND): the start is its first counted screen
        engine = Engine(session, full_steps=o['full_steps'], short_steps=o.get('short_steps'),
                        probe_steps=o.get('probe_steps', 2), warmup=o.get('warmup', 3), job_gpu_hours=o['job_gpu_hours'],
                        multi_fidelity=o.get('multi_fidelity', True), run_noise=o.get('run_noise', 0.015),
                        noise_pairs=o.get('noise_pairs'), log=o.get('log'), accounting=o.get('accounting'))
        return engine.run(strategy)
    return run


LLM_STRATEGIES = ('agent', 'hls', 'hls2', 'doe+hls', 'greedy')   # the strategies that call the model (doe, tpe, random do not)
STRATEGIES = {'random': _knob('random', knob_space.random_search), 'tpe': _knob('tpe', knob_space.tpe_search),
              'agent': _agent, 'hls': _engine('hls'), 'hls2': _engine('hls2'), 'doe': _engine('doe'),
              'doe+hls': _engine('doe+hls'), 'greedy': _engine('greedy')}
# hls3 (search/hls3.py): hls2 with FABLE_TAKEAWAYS_20261030's general strategies, each component ablatable
LLM_STRATEGIES += ('hls3',)
STRATEGIES['hls3'] = _engine('hls3')
# the OptiCo baseline (search/optico.py): its agents call the model
LLM_STRATEGIES += ('optico',)
STRATEGIES['optico'] = _optico
# wcs (search/wcs.py): the whole-configuration search of FABLE_ARCH_REVIEW_20261031 section 4, each component ablatable
LLM_STRATEGIES += ('wcs',)
STRATEGIES['wcs'] = _engine('wcs')
# the candidates (search/esls.py, search/rpp.py; possible_search_algorithm/20): rpp-ctl calls no model
LLM_STRATEGIES += ('esls', 'rpp')
STRATEGIES.update({'esls': _engine('esls'), 'rpp': _engine('rpp'), 'rpp-ctl': _engine('rpp-ctl')})


def resolve(name: str | None, *, has_llm: bool, knobs: dict | None = None) -> str:
    if name in (None, '', 'auto'):
        if has_llm:
            # a screening design pays when the space is mostly toggles; its main effects go stale once a layout moves
            two = [k for k in (knobs or {}).values() if (k.get('type') == 'bool' and not k.get('narrowed'))
                   or len(k.get('choices') or []) == 2]
            return 'doe+hls' if len(two) >= 4 and len(two) >= 0.75 * len(knobs) else 'hls'
        import importlib.util
        return 'tpe' if importlib.util.find_spec('optuna') else 'random'      # optuna is an optional dependency
    name = {'bcompv4.1': 'bcomp', 'bcomp-v4.1': 'bcomp', 'bcomp_v4.1': 'bcomp'}.get(name, name)
    if name not in STRATEGIES:
        raise ValueError(f'unknown search strategy {name!r} (known: {sorted(STRATEGIES)})')
    return name


def run(name: str, ctx: Context) -> SearchTrace:
    name = {'bcompv4.1': 'bcomp', 'bcomp-v4.1': 'bcomp', 'bcomp_v4.1': 'bcomp'}.get(name, name)
    return STRATEGIES[name](ctx)


# opro and ltune (search/opro.py, search/ltune.py; possible_search_algorithm/10 section 04): name -> what turning it off
# removes. One LLM budget for the knob and the code side (search/llmopt: at most 10 calls and 8 USD).
OPRO_COMPONENTS = {
    'opro_prior': "the model's prior call (wcs's): its whole configurations are the first round, its means the time "
                  'model; off: a flat prior, the first round a configuration call',
    'opro_history': 'the history in every configuration call (each measured configuration best first with its outcome '
                    'and peak, the failures, the proposals ruled out); off: the task and the knob table only',
    'opro_filter': 'the feasibility filter and order (a proposal the runs rule out is not run and is shown to the '
                   "model; feasibility x the stated gain orders a batch); off: legality only, the model's order",
    'opro_ledger': "the censored memory ledger as the feasibility where the runs do not decide; off: wcs's",
    'opro_probe': 'a proposal below 0.7 feasibility is probed first (within the low-fidelity allowance); off: full runs '
                  'only (a code proof keeps its probe)',
    'opro_code': "wcs's code track (patches only, the self-proof, proven patches as pseudo-knobs), asked when the knob "
                 'side stalls (the incumbent unchanged over two full screens)',
    'opro_tail': 'the tail bundle on the incumbent before the feasibility-weighted draw when no proposal is left; off: '
                 'the draw only'}
LTUNE_COMPONENTS = {
    'ltune_one_call': "the knob side's one model call (the race's diverse configurations), no prior call; off: wcs's "
                      'prior call too (its means and memory declarations, its configurations added to the race)',
    'ltune_probe': "the race's probe-only runs of the configurations below 0.7 feasibility; off: no probe in the race",
    'ltune_short': "the short rung (the race's first run, the local refinement's ladder); off: full runs only",
    'ltune_cap': "the race's capping: a short gain below the best one's by max(2 sigma_short, 2 %) is not extended",
    'ltune_ledger': "the censored memory ledger as the feasibility where the runs do not decide; off: wcs's",
    'ltune_code': "wcs's code track after the race, alternating with the knob side",
    'ltune_tail': 'the tail bundle on the winner before its neighbours',
    'ltune_local': "the winner's neighbours (one or two knobs off it) by feasibility x the time model, through the rung "
                   'ladder; off: the feasibility-weighted draw'}
ABLATIONS.update({k: f'opro: {d}' for k, d in OPRO_COMPONENTS.items()})
ABLATIONS.update({k: f'ltune: {d}' for k, d in LTUNE_COMPONENTS.items()})


def _engine_c4(name):
    """opro / ltune on the Engine: _engine's set-up (the proposer, the session, the engine) with their own classes."""
    def run(ctx: Context) -> SearchTrace:
        from ..agent.capability import mode_for, policy_overrides, search_params
        from .engine import Engine
        from .ltune import LTune
        from .opro import OPRO
        from .proposers import LLMProposer
        o = ctx.options
        c = o.get('capability', 0.6)
        kp = search_params(c)
        exp = o.get('experience')                      # experience.store.plan for this job, or None
        ablate = set(o.get('ablate') or ())
        cls, comps = (OPRO, OPRO_COMPONENTS) if name == 'opro' else (LTune, LTUNE_COMPONENTS)
        facts = getattr(ctx.hooks, 'memory_facts', lambda: None)() or {}
        acc = o.get('accounting')
        prop = LLMProposer(ctx.driver, ctx.hooks, max_hypotheses=kp['max_hypotheses'], menu=kp['menu'],
                           sigma_scale=kp['sigma_scale'], prior=o.get('prior_results'), experience=(exp or {}).get('payload'),
                           search_calibrates=True, accounting=acc, features=cls.proposer_features(ablate))
        strategy = cls(prop, off=sorted(x for x in ablate if x in comps), facts=facts, accounting=acc, experience=exp,
                       kappa0=kp['kappa0'], seed=ctx.seed)
        session = ctx.session(o.get('arm') or name, stop_after_same_failures=10 ** 6, mode=mode_for(c),
                              patch_line_limit=policy_overrides(c)['patch_line_limit'],
                              layers=tuple(getattr(ctx.hooks, 'layers', None) or ('L0', 'L1')),
                              count_rescores=bool(o.get('count_rescores')))
        if ctx.start_receipt is None:                  # no reference run before the search: the start is its first
            session.screen('c0')                       # counted screen (as _engine)
        engine = Engine(session, full_steps=o['full_steps'], short_steps=o.get('short_steps'),
                        probe_steps=o.get('probe_steps', 2), warmup=o.get('warmup', 3), job_gpu_hours=o['job_gpu_hours'],
                        multi_fidelity=o.get('multi_fidelity', True), run_noise=o.get('run_noise', 0.015),
                        noise_pairs=o.get('noise_pairs'), log=o.get('log'), accounting=o.get('accounting'))
        return engine.run(strategy)
    return run


LLM_STRATEGIES += ('opro', 'ltune')
STRATEGIES.update({'opro': _engine_c4('opro'), 'ltune': _engine_c4('ltune')})


# xfer (search/xfer.py; FABLE_PICK_C6_20261031 section 2): transfer-first over the experience store, the knob side and the
# code side. One LLM budget for both (search/llmopt: at most 10 calls and 8 USD); six asks on its own instance.
XFER_COMPONENTS = {
    'xfer_knob': "the other environments' best configurations from the arm's home (their run packs, weighed by the "
                 'similarity of their numbers and the RGPE update): screen 1, the last screen, a step with nothing else; '
                 "off: the prior's configuration first",
    'xfer_rgpe': "after each full screen a source's weight x max(0.25, 1 + agreements - disagreements) of its single-move "
                 'effects with the pairs measured here; off: the similarity weights stay',
    'xfer_port': "the port ask: the transfer brief's proven patches (best weighted gain first, one per site) and its inert "
                 'ones as do-not-repeat in the first patch call, the model porting at most 3; off: ordinary asks only',
    'xfer_prior': "after the source screens, the prior's best configuration when it differs from the incumbent in two or "
                  'more big knobs and the time model rates it above; off: code',
    'xfer_code': "wcs's code track (patches only, the self-proof, proven patches as pseudo-knobs), six asks",
    'xfer_tail': 'the tail bundle on the incumbent on the screen before the last; off: code',
    'xfer_ledger': "the censored memory ledger as the feasibility where the runs do not decide; off: wcs's",
    'xfer_probe': 'a configuration below 0.7 feasibility is probed first (within the low-fidelity allowance); off: full runs '
                  'only (a code proof keeps its probe)'}
ABLATIONS.update({k: f'xfer: {d}' for k, d in XFER_COMPONENTS.items()})


def _engine_x6(name):
    """xfer on the Engine: _engine_c4's set-up (the proposer, the session, the engine) with its own class."""
    def run(ctx: Context) -> SearchTrace:
        from ..agent.capability import mode_for, policy_overrides, search_params
        from .engine import Engine
        from .proposers import LLMProposer
        from .xfer import XFer
        o = ctx.options
        c = o.get('capability', 0.6)
        kp = search_params(c)
        exp = o.get('experience')                      # experience.store.plan for this job, or None
        ablate = set(o.get('ablate') or ())
        facts = getattr(ctx.hooks, 'memory_facts', lambda: None)() or {}
        acc = o.get('accounting')
        prop = LLMProposer(ctx.driver, ctx.hooks, max_hypotheses=kp['max_hypotheses'], menu=kp['menu'],
                           sigma_scale=kp['sigma_scale'], prior=o.get('prior_results'), experience=(exp or {}).get('payload'),
                           search_calibrates=True, accounting=acc, features=XFer.proposer_features(ablate))
        strategy = XFer(prop, off=sorted(x for x in ablate if x in XFER_COMPONENTS), facts=facts, accounting=acc,
                        experience=exp, kappa0=kp['kappa0'], seed=ctx.seed)
        session = ctx.session(o.get('arm') or name, stop_after_same_failures=10 ** 6, mode=mode_for(c),
                              patch_line_limit=policy_overrides(c)['patch_line_limit'],
                              layers=tuple(getattr(ctx.hooks, 'layers', None) or ('L0', 'L1')),
                              count_rescores=bool(o.get('count_rescores')))
        if ctx.start_receipt is None:                  # no reference run before the search: the start is its first
            session.screen('c0')                       # counted screen (as _engine)
        engine = Engine(session, full_steps=o['full_steps'], short_steps=o.get('short_steps'),
                        probe_steps=o.get('probe_steps', 2), warmup=o.get('warmup', 3), job_gpu_hours=o['job_gpu_hours'],
                        multi_fidelity=o.get('multi_fidelity', True), run_noise=o.get('run_noise', 0.015),
                        noise_pairs=o.get('noise_pairs'), log=o.get('log'), accounting=o.get('accounting'))
        return engine.run(strategy)
    return run


LLM_STRATEGIES += ('xfer',)
STRATEGIES['xfer'] = _engine_x6('xfer')

# bcomp (search/bcomp.py; FINALISTS_IMPL_SPEC_20261101 A.1): bundle completion over the memory ledger, on the finalists'
# shared base (search/finalist_base.py: S1-S6, the ladder). One LLM budget for the knob and the code side (10 calls, 8 USD).
BCOMP_COMPONENTS = {
    'bcomp_ledger': "the censored memory ledger as the only feasibility (S4: the card less 2 %, declared GiB floored at "
                    "0.15 until measured); off: wcs's",
    'bcomp_complete': "the configuration screens' completions: a run that failed for memory (a probe too) and is predicted "
                      'faster than the incumbent, plus the partner that frees the most GiB (measured, then declared, then '
                      "of unknown direction by the guide's gain), carrying its patches, probed first below 0.7; two per run",
    'bcomp_challenge': "the configuration screens' challenges: the guide's vetoed value with the largest |declared gain| that "
                       'no run holds, on the incumbent alone, in full (probed first below 0.5); two per arm',
    'bcomp_isolated_blame': 'S4: a value is lowered alone (x0.25) only when every other value of each run that failed for '
                            'memory holding it fitted somewhere',
    'bcomp_peak_thrash': 'S5: a probe whose measured peak reaches 95 % of the card is thrash (no full run; the ledger and '
                         'the completion material)'}
# bcomp v2 (V2_SEARCH_DESIGN_20261101 M1, M2, M4; search/explore.py's ExploreMixin): its new switches
BCOMP_COMPONENTS.update({
    'bcomp_feedback': "M2: the failure attribution's rules on every knob candidate (a value in two memory failures and no "
                      'fitted run needs a saver covering its shortfall, excluded after three failures with a saver; a set '
                      'that failed together otherwise is never held whole again) and the completion queue re-ranked by '
                      "the updated ledger's P; off: v1's completions",
    'bcomp_revise': 'M2: one model call revising the guide\'s levels (gain, sd, p_runs, memory) of the values the failures '
                    'involve, after the second failed run or probe (or one failure against a declared p_runs >= 0.7), '
                    'merged into the guide; two per arm within the model budget',
    'ex_contrarian': 'M1: contrarian moves (a doubtful low-rated value on the incumbent, its conflicts resolved, a saver '
                     "added when it spends memory) in the challenge's slot, two per arm; off: v1's challenges",
    'ex_escape': 'M4: while stalled every knob candidate admissible (trust region, tabu, never a failure repeated) and '
                 'the escape candidates (radius, warm restart, momentum)'})
ABLATIONS.update({k: f'bcomp: {d}' for k, d in BCOMP_COMPONENTS.items()})


def _engine_f1(ctx: Context) -> SearchTrace:
    """bcomp on the Engine: finalist_base.run_engine (the proposer with its features, the strategy with its components
    switched off, the session with random's same-failure stop, the engine)."""
    from .bcomp import BComp
    from .finalist_base import run_engine
    return run_engine(ctx, BComp, BCOMP_COMPONENTS)


LLM_STRATEGIES += ('bcomp',)
STRATEGIES['bcomp'] = _engine_f1


# Strategies whose full state is saved for `infrarsi run --resume`.
RESUMABLE = ('bcomp',)
