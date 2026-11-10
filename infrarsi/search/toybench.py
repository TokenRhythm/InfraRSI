"""Toy experiments on ToyTrain (revamp design C; Fable 5.1 consult §5): strategies x tasks x budgets, one run each.

    python -m infrarsi.search.toybench --out DIR [--tasks 36] [--budgets 16,40] [--arms random,tpe,greedy,hls,...]

Budget = B full-screen equivalents of GPU-hours (B x the reference run's cost); the reference run itself is free.
Metrics on true (noiseless) values: frac = log(t_ref / t_selected) / log(t_ref / t_optimum) (the fraction of the
attainable gain), GPU-hours used, invalid share (GPU-hours on failed, infeasible or aborted runs), stop reason, and the
anytime curve (best measured candidate's true frac at 25/50/75/100 % of the budget). One seed per (task, arm, budget):
generality comes from the task grid; arms are paired by task through common random numbers.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from ..agent.budget import VirtualClock
from ..agent.capability import search_params
from ..agent.search import SearchSession
from ..ext.common.env_task import EnvTask
from ..ext.common.session_hooks import SessionHooks
from ..verify import Verifier
from . import HLS3_COMPONENTS, HLS3_R1, HLS3_R2, HLS3_R3, WCS_COMPONENTS
from .engine import Engine
from .proposers import SimProposer
from .strategies import DOE, HLS, HLS2, Chain, Greedy, RandomKnobs, TPEKnobs
from .toytrain import ToyRunner, _patches, _u, couple_tasks, enabler_tasks, make_sim, mem_tasks, tasks, toy_task_spec

FULL, SHORT, PROBE = 30, 10, 2


class MemoryHooks(SessionHooks):
    """The toy's hooks with the job's memory facts (hls3 reads the card, the GPUs and the parameters from them)."""

    def __init__(self, task, facts: dict, **kw):
        super().__init__(task, **kw)
        self._facts = dict(facts)

    def memory_facts(self) -> dict:
        return dict(self._facts)


def arms(sim, *, q: float = 0.7, b: float = 1.5, rho: float = 0.5, memory: bool = False) -> dict:
    """name -> (strategy factory, engine options). `memory`: the simulated model declares each move's memory (hls3's
    schema field; the other strategies ignore it, so their runs are unchanged)."""
    from .hls3 import HLS3
    from .wcs import WCS
    sp = lambda: SimProposer(sim, q=q, b=b, rho=rho, memory=memory)          # noqa: E731
    facts = getattr(sim, 'memory_facts', None)
    return {
        'random': (lambda: RandomKnobs(seed=sim.task.seed % 2 ** 31), {'multi_fidelity': False}),
        'tpe': (lambda: TPEKnobs(seed=42), {'multi_fidelity': False}),
        'greedy': (lambda: Greedy(sp()), {'multi_fidelity': False}),
        'hls-1f': (lambda: HLS(sp()), {'multi_fidelity': False}),
        # ablations of the composition machinery (Fable decision 2.A-2.C)
        'hls-nobundle': (lambda: HLS(sp(), bundle=False), {'multi_fidelity': True}),
        'hls-nosplit': (lambda: HLS(sp(), split=False), {'multi_fidelity': True}),
        'hls-noconfirm': (lambda: HLS(sp(), confirm=False), {'multi_fidelity': True}),
        'hls-nocarry': (lambda: HLS(sp(), carry=False), {'multi_fidelity': True}),
        'hls': (lambda: HLS(sp()), {'multi_fidelity': True}),
        'hls2': (lambda: HLS2(sp()), {'multi_fidelity': True}),         # hls with the 2026-10 search fix
        'hls3': (lambda: HLS3(sp(), facts=facts), {'multi_fidelity': True}),   # hls2 + FABLE_TAKEAWAYS section 4
        'hls3-off': (lambda: HLS3(sp(), facts=facts, off=sorted(HLS3_COMPONENTS)), {'multi_fidelity': True}),   # = hls2
        # hls3 as of a round: the fixes of that round's FABLE_HLS3_FIXES and of every later one off
        'hls3-r0': (lambda: HLS3(sp(), facts=facts, off=list(HLS3_R1 + HLS3_R2 + HLS3_R3)), {'multi_fidelity': True}),
        'hls3-r1': (lambda: HLS3(sp(), facts=facts, off=list(HLS3_R2 + HLS3_R3)), {'multi_fidelity': True}),
        'hls3-r2': (lambda: HLS3(sp(), facts=facts, off=list(HLS3_R3)), {'multi_fidelity': True}),   # before the R3 fixes
        'doe+hls': (lambda: Chain(DOE(sp()), HLS(sp())), {'multi_fidelity': True}),
        'random+fmr': (lambda: RandomKnobs(seed=sim.task.seed % 2 ** 31, probes=True), {'multi_fidelity': True}),
        # the capability knob set to the simulated model's knowledge coverage q (design D)
        'hls-knob': (lambda: HLS(sp(), kappa0=search_params(q)['kappa0'], probe_below=search_params(q)['probe_below']),
                     {'multi_fidelity': True}),
        # hls3 with one component off (its ablation)
        **{f'hls3-no-{c[5:]}': ((lambda c=c: HLS3(sp(), facts=facts, off=[c])), {'multi_fidelity': True})
           for c in HLS3_COMPONENTS},
        # wcs (search/wcs.py; FABLE_ARCH_REVIEW_20261031 section 4): the whole-configuration search; without the model's
        # prior (a flat prior: the feasibility model over random's table), without the feasibility model, one component off
        'wcs': (lambda: WCS(sp(), facts=facts, seed=sim.task.seed % 2 ** 31), {'multi_fidelity': True}),
        'wcs-noprior': (lambda: WCS(sp(), facts=facts, off=['wcs_prior'], seed=sim.task.seed % 2 ** 31), {'multi_fidelity': True}),
        'wcs-nofeas': (lambda: WCS(sp(), facts=facts, off=['wcs_feasibility', 'wcs_antagonism'], seed=sim.task.seed % 2 ** 31),
                       {'multi_fidelity': True}),
        **{f'wcs-no-{c[4:]}': ((lambda c=c: WCS(sp(), facts=facts, off=[c], seed=sim.task.seed % 2 ** 31)), {'multi_fidelity': True})
           for c in WCS_COMPONENTS},
        **candidate_arms(sim, sp, facts),
    }


def candidate_arms(sim, sp, facts) -> dict:
    """The candidates (search/esls.py, search/rpp.py; possible_search_algorithm/20) and their controls: esls, its ablations
    (esls-noprobe: the enablers' GiB from the declarations; esls-noledger: wcs's feasibility; esls-no-<component>); rpp;
    rpp-ctl (a = 1, T = inf, no model, no store, no code: random + the same filter + the same rungs); rpp-ctl-prior (the
    same with the prior's memory declarations in its filter); rpp-a0 (a = 0, T = inf: random + the rungs)."""
    from . import ESLS_COMPONENTS
    from .esls import ESLS
    from .rpp import RPP
    seed = sim.task.seed % 2 ** 31
    inf = float('inf')
    mf = {'multi_fidelity': True}
    ctl = ['rpp_prior', 'rpp_anneal', 'rpp_code']
    return {
        'esls': (lambda: ESLS(sp(), facts=facts, seed=seed), mf),
        'esls-noprobe': (lambda: ESLS(sp(), facts=facts, off=['esls_enabler_probe'], seed=seed), mf),
        'esls-noledger': (lambda: ESLS(sp(), facts=facts, off=['esls_ledger'], seed=seed), mf),
        **{f'esls-no-{c[5:]}': ((lambda c=c: ESLS(sp(), facts=facts, off=[c], seed=seed)), mf) for c in ESLS_COMPONENTS},
        'rpp': (lambda: RPP(sp(), facts=facts, seed=seed), mf),
        'rpp-ctl': (lambda: RPP(sp(), facts=facts, off=ctl, temperature=inf, name='rpp-ctl', seed=seed), mf),
        'rpp-ctl-prior': (lambda: RPP(sp(), facts=facts, off=['rpp_anneal', 'rpp_code'], temperature=inf,
                                      name='rpp-ctl-prior', seed=seed), mf),
        'rpp-a0': (lambda: RPP(sp(), facts=facts, off=ctl + ['rpp_ledger'], a=0.0, temperature=inf, name='rpp-a0',
                               seed=seed), mf),
    }


def run_one(task, arm: str, budget: float, *, job_factor: float | None = 300.0, proposer: dict | None = None,
            bind: str = 'gpu', inert: float = 0.0) -> dict:
    """One arm on one task. `bind`: 'gpu' - B full screens' worth of GPU-hours with 4B screens (the toybench's budget);
    'screens' - B screens with 4B screens' worth of GPU-hours (the real arms' budget: 10 screens bound, GPU-hours not).
    `inert` (hls3 F10): each working patch is inert with this probability - it installs but its hook never runs; the
    simulated model predicts the gain it would give if it ran; the row counts the full screens of a new inert patch."""
    sim = make_sim(task)
    if inert and getattr(sim, 'patch_ok', None):
        sim.inert = frozenset(p for p, ok in sim.patch_ok.items() if ok and _u(task.seed, 'inert', p) < inert)
    clock = VirtualClock()
    runner = ToyRunner(sim, clock)
    ref_cfg = sim.reference()
    t0 = time.perf_counter()
    probe_task = EnvTask.from_dict(toy_task_spec(sim, {'screens': 1, 'llm_tokens': 0, 'wall_seconds': 1e12, 'gpu_hours': 1e9}))
    ref = runner.run(probe_task, ref_cfg, None, 'reference', 'toy', candidate_id='c0', steps=FULL)
    c_full = ref['cost']['gpu_seconds'] / 3600
    screens, gpu_h = (int(budget), 4 * budget * c_full) if bind == 'screens' else (int(4 * budget), budget * c_full)
    task_env = EnvTask.from_dict(toy_task_spec(sim, {'screens': screens, 'llm_tokens': 0, 'wall_seconds': 1e12,
                                                     'gpu_hours': gpu_h}))
    facts = getattr(sim, 'memory_facts', None)
    hooks = MemoryHooks(task_env, facts, base_files={'patches.txt': ''}) if facts else \
        SessionHooks(task_env, base_files={'patches.txt': ''})
    session = SearchSession(task=task_env, budget=task_env.search_budget(), runner=runner,
                            judge=Verifier(0.03, memory_cap_gib=sim.hw['mem']), run_id=f'toy-{arm}', arm=arm,
                            start_config=ref_cfg, start_receipt=ref, reference_receipt=ref, hooks=hooks, clock=clock,
                            stop_after_same_failures=10 ** 6)
    factory, opts = arms(sim, **(proposer or {}))[arm]
    engine = Engine(session, full_steps=FULL, short_steps=SHORT, probe_steps=PROBE,
                    job_gpu_hours=None if job_factor is None else job_factor * c_full, **opts)
    trace = engine.run(factory()).to_dict()
    t_ref, _ = sim.truth(ref_cfg)
    t_opt, _, _ = sim.optimum()
    sel = session.candidates[trace['selected']]
    t_sel, _ = sim.truth(sel.config, _patches(sel.overlay))
    attainable = math.log(t_ref / t_opt)
    frac = lambda t: (math.log(t_ref / t) / attainable) if t and attainable > 0 else 0.0     # noqa: E731
    used = session.ledger.used['gpu_hours']
    bad = sum(r.get('gpu_hours', 0) for r in session.screens if not r.get('valid') and not r.get('aborted'))
    aborted = sum(r.get('gpu_hours', 0) for r in session.screens if r.get('aborted'))
    curve = {}
    for share in (0.25, 0.5, 0.75, 1.0):
        best = None
        for r in session.screens:
            if r['cum']['gpu_hours'] <= share * budget * c_full + 1e-9 and r.get('best_valid'):
                best = r['best_valid']
        c = session.candidates[best] if best else session.candidates['c0']
        curve[str(share)] = round(frac(sim.truth(c.config, _patches(c.overlay))[0]), 4)
    curve_screens = {}                              # (the candidates: the anytime curve by full screens, design 30 4.3)
    for k in (1, 3, 5, 10):
        best = None
        for r in session.screens:
            if (r.get('cum') or {}).get('screens', 0) <= k and r.get('best_valid'):
                best = r['best_valid']
        c = session.candidates[best] if best else session.candidates['c0']
        curve_screens[str(k)] = round(frac(sim.truth(c.config, _patches(c.overlay))[0]), 4)
    full_runs = [x for x in runner.log[1:] if not x.get('stop_after')]         # (the first row: the reference)
    dead = getattr(sim, 'inert', None)

    def _new_inert(cid) -> bool:                    # a full run of a candidate that added an inert patch to its base
        c = session.candidates.get(cid)
        par = session.candidates.get(getattr(c, 'parent', None) or '') if c is not None else None
        mine = {x for x in _patches(c.overlay) if not x.startswith(('proof:', 'defer:'))} if c is not None else set()
        return bool((mine - set(_patches(par.overlay) if par is not None else ())) & dead)
    return {'task': task.name, 'arm': arm, 'budget': budget, 'proposer': proposer or {}, 'job_factor': job_factor,
            'bind': bind, 'frac': round(frac(t_sel), 4),
            'failed_screens': sum(1 for x in full_runs if x['why'] and not x['stopped']),
            'oom_screens': sum(1 for x in full_runs if x['why'] == 'oom'),
            'probes': sum(1 for x in runner.log[1:] if x.get('stop_after')),
            'shorts': sum(1 for x in runner.log[1:] if x.get('stop_after') == SHORT),     # (of the probes: the short rung)
            'lowfi_gpu_h_share': round(sum(x.get('gpu_h', 0) for x in runner.log[1:] if x.get('stop_after'))
                                       / (budget * c_full), 3),
            'speedup_true': round(t_ref / t_sel, 4) if t_sel else None, 'attainable_x': round(t_ref / t_opt, 4),
            'gpu_h_share': round(used / (budget * c_full), 3), 'invalid_share': round(bad / used, 3) if used else 0.0,
            'aborted_share': round(aborted / used, 3) if used else 0.0,
            'screens': session.ledger.used['screens'], 'trials': session.ledger.used.get('trials', 0),
            'stop': trace['stop_reason'], 'curve': curve, 'curve_screens': curve_screens, 'schema': task.schema,
            'model': task.model, 'gpu': task.gpu,
            'hidden': task.hidden, 'noise': task.noise, 'wall_s': round(time.perf_counter() - t0, 2),
            **({'inert_patches': sorted(dead), 'inert_screens': sum(1 for x in full_runs if _new_inert(x['candidate_id']))}
               if dead else {})}


def summarize(rows: list[dict]) -> str:
    """Markdown: per (budget, arm) the median frac, mean frac, tasks won vs greedy (paired), invalid share, stops."""
    out = ['| budget | arm | median frac | mean frac | wins vs greedy | mean GPU-h share | invalid share | early stops |',
           '|---|---|---|---|---|---|---|---|']
    by = {}
    for r in rows:
        by.setdefault((r['budget'], r['arm']), []).append(r)
    for (b, arm), rs in sorted(by.items(), key=lambda x: (x[0][0], x[0][1])):
        g = {x['task']: x['frac'] for x in by.get((b, 'greedy'), [])}
        wins = sum(1 for x in rs if x['task'] in g and x['frac'] > g[x['task']] + 1e-6)
        ties = sum(1 for x in rs if x['task'] in g and abs(x['frac'] - g[x['task']]) <= 1e-6)
        early = sum(1 for x in rs if not str(x['stop']).startswith('budget'))
        out.append(f'| {b:g} | {arm} | {statistics.median(x["frac"] for x in rs):.3f} | {statistics.mean(x["frac"] for x in rs):.3f} | '
                   f'{wins}/{len(rs)} (ties {ties}) | {statistics.mean(x["gpu_h_share"] for x in rs):.2f} | '
                   f'{statistics.mean(x["invalid_share"] for x in rs):.2f} | {early}/{len(rs)} |')
    return '\n'.join(out)


def main(argv=None):
    import os
    import tempfile
    ap = argparse.ArgumentParser(prog='python -m infrarsi.search.toybench')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tasks', type=int, default=36)
    ap.add_argument('--budgets', default='16,40')
    ap.add_argument('--arms', default='random,tpe,greedy,hls-1f,hls,doe+hls,random+fmr')
    ap.add_argument('--q', type=float, default=0.7)
    ap.add_argument('--b', type=float, default=1.5)
    ap.add_argument('--rho', type=float, default=0.5)
    ap.add_argument('--job-factor', default='300', help="the production job's GPU-hours in reference runs; 'none': no "
                    'value/cost stop')
    ap.add_argument('--mem', action='store_true', help="the memory-coupled tasks (toytrain.mem_tasks), the simulated model "
                    "declaring memory, the screens binding (hls3's comparison)")
    ap.add_argument('--inert', type=float, default=0.0, help='each working patch inert (installs, its hook never runs) '
                    "with this probability (hls3's patch self-proof, F10)")
    ap.add_argument('--enabler', action='store_true', help='the undeclared-enabler tasks (toytrain.enabler_tasks; implies '
                    "--mem's model and budget: hls3's pair probes, F16)")
    ap.add_argument('--couple', action='store_true', help='the coupling tasks (toytrain.couple_tasks: the memory-coupled '
                    "job with a real table's noise knobs and one pair that only pays together), as --mem (wcs's comparison)")
    a = ap.parse_args(argv)
    os.environ.setdefault('INFRARSI_RECEIPT_KEY', str(Path(tempfile.gettempdir()) / 'infrarsi_toybench.key'))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    mem = a.mem or a.enabler or a.couple
    prop = {'q': a.q, 'b': a.b, 'rho': a.rho, **({'memory': True} if mem else {})}
    jf = None if a.job_factor == 'none' else float(a.job_factor)
    for task in (couple_tasks(a.tasks) if a.couple else enabler_tasks(a.tasks) if a.enabler else mem_tasks(a.tasks) if a.mem
                 else tasks(a.tasks)):
        for b in [float(x) for x in a.budgets.split(',')]:
            for arm in a.arms.split(','):
                r = run_one(task, arm, b, job_factor=jf, proposer=prop, bind='screens' if mem else 'gpu',
                            inert=a.inert)
                rows.append(r)
                print(json.dumps(r), flush=True)
    (out / 'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
    (out / 'summary.md').write_text(summarize(rows) + '\n', encoding='utf-8')
    print(summarize(rows))


# ---------------------------------------------------------------- opro / ltune (possible_search_algorithm/10 section 04)
def c4_arms(sim, sp, facts) -> dict:
    """opro and ltune (search/opro.py, search/ltune.py) on the simulated model (its configuration call:
    SimProposer.config_batch); opro-nohist: the rounds without the history (opro_history off); ltune-prior: with wcs's
    prior call too (ltune_one_call off)."""
    from .ltune import LTune
    from .opro import OPRO
    seed = sim.task.seed % 2 ** 31
    mf = {'multi_fidelity': True}
    return {'opro': (lambda: OPRO(sp(), facts=facts, seed=seed), mf),
            'opro-nohist': (lambda: OPRO(sp(), facts=facts, off=['opro_history'], seed=seed), mf),
            'ltune': (lambda: LTune(sp(), facts=facts, seed=seed), mf),
            'ltune-prior': (lambda: LTune(sp(), facts=facts, off=['ltune_one_call'], seed=seed), mf)}


_arms_before_c4 = arms


def arms(sim, *, q: float = 0.7, b: float = 1.5, rho: float = 0.5, memory: bool = False) -> dict:   # noqa: F811
    """The arms above and c4_arms (an addition: the arms above are unchanged)."""
    out = _arms_before_c4(sim, q=q, b=b, rho=rho, memory=memory)
    out.update(c4_arms(sim, lambda: SimProposer(sim, q=q, b=b, rho=rho, memory=memory), getattr(sim, 'memory_facts', None)))
    return out


# ---------------------------------------------------------------- xfer (FABLE_PICK_C6_20261031 section 2)
TOY_SOURCES = 5                 # xfer-src: the other tasks of the family that stand for the other environments


def toy_env(sim) -> dict:
    """A memory toy's environment numbers as a run pack's `env` line gives them (its card, one node, its state's size)."""
    return {'card_gib': getattr(sim, 'cap', None), 'nodes': 1, 'gpus_per_node': 8,
            'params_b': (getattr(sim, 'memory_facts', None) or {}).get('params_b'), 'framework': 'other'}


def toy_packs(task, n: int = TOY_SOURCES) -> list:
    """D4 for the toybench (xfer-src): run packs of `n` other tasks of the task's family (a fixed hash order; never the
    task itself), as if earlier arms had run them - each its reference, its optimum and every single move from the
    reference, at their noiseless times (an out-of-memory move as such), configurations as the packs write them (x-<knob>,
    the value as text). [] for a task without a family of memory toys."""
    import hashlib
    from .toytrain import CoupleTask, MemTask, couple_tasks, enabler_tasks
    if not isinstance(task, MemTask):
        return []
    fam = couple_tasks() if isinstance(task, CoupleTask) else enabler_tasks() if getattr(task, 'enabler', False) else mem_tasks()
    others = sorted((t for t in fam if t.name != task.name),
                    key=lambda t: hashlib.sha256(f'{task.name}|{t.name}'.encode()).hexdigest())[:n]
    out = []
    for t in others:
        s = make_sim(t)

        def state(cfg):
            return {'x-' + k.replace('_', '-'): (str(v).lower() if isinstance(v, bool) else str(v)) for k, v in cfg.items()}
        ref = s.reference()
        rows = [('c0', ref)]
        _, opt, _ = s.optimum()
        if opt:
            rows.append(('opt', dict(opt)))
        for k, vals in s.knobs().items():
            rows += [(f'm-{k}-{v}', {**ref, k: v}) for v in vals if v != ref[k]]
        lines = [{'kind': 'pack', 'id': 'pack', 'schema': 'infrarsi-exp/1', 'pack_id': f'run-toy-{t.name}',
                  'title': f'toy {t.name}', 'license': 'unspecified', 'tier': 'verified'},
                 {'kind': 'env', 'id': 'env', 'gpu': 'toy', 'gpus_per_node': 8, 'nodes': 1, 'memory_gib': s.cap,
                  'framework': 'other', 'model': {'params_b': s.memory_facts['params_b']}, 'workload': {'seq_len': 4096},
                  'noise_pct': 1.0}]
        for oid, cfg in rows:
            tt, why = s.truth(cfg)
            lines.append({'kind': 'observation', 'id': oid, 'env': 'env', 'method': 'infrarsi', 'config': state(cfg),
                          **({'status': 'ok', 'step_time_s': round(tt, 6)} if tt else {'status': 'oom' if why == 'oom' else 'crash'})})
        out.append(lines)
    return out


def x6_arms(sim, sp, facts) -> dict:
    """xfer (search/xfer.py) on the simulated model: xfer - no store and no brief (the toybench has no home: the prior's
    configuration first, ordinary asks); xfer-src - the run packs of five other tasks of the family as its store
    (toy_packs: their optima and single moves), the task's own numbers as the target's."""
    from .xfer import XFer
    seed = sim.task.seed % 2 ** 31
    mf = {'multi_fidelity': True}
    return {'xfer': (lambda: XFer(sp(), facts=facts, seed=seed), mf),
            'xfer-src': (lambda: XFer(sp(), facts=facts, seed=seed, packs=toy_packs(sim.task), env=toy_env(sim)), mf)}


_arms_before_x6 = arms


def arms(sim, *, q: float = 0.7, b: float = 1.5, rho: float = 0.5, memory: bool = False) -> dict:   # noqa: F811
    """The arms above and x6_arms (an addition: the arms above are unchanged)."""
    out = _arms_before_x6(sim, q=q, b=b, rho=rho, memory=memory)
    out.update(x6_arms(sim, lambda: SimProposer(sim, q=q, b=b, rho=rho, memory=memory), getattr(sim, 'memory_facts', None)))
    return out


# ---------------------------------------------------------------- the finalists (FINALISTS_IMPL_SPEC_20261101 A.1: bcomp)
def f1_arms(sim, sp, facts) -> dict:
    """bcomp (search/bcomp.py) on the simulated model: bcomp - no home in the toybench, so no experience source (start one
    is the guide's best configuration); bcomp-src - the run packs of five other tasks of the family as its experience
    (toy_packs, as xfer-src: their references, optima and single moves), the task's own numbers as the target's."""
    from .bcomp import BComp
    seed = sim.task.seed % 2 ** 31
    mf = {'multi_fidelity': True}
    return {'bcomp': (lambda: BComp(sp(), facts=facts, seed=seed), mf),
            'bcomp-src': (lambda: BComp(sp(), facts=facts, seed=seed, packs=toy_packs(sim.task), env=toy_env(sim)), mf)}


_arms_before_f1 = arms


def arms(sim, *, q: float = 0.7, b: float = 1.5, rho: float = 0.5, memory: bool = False) -> dict:   # noqa: F811
    """The arms above and f1_arms (an addition: the arms above are unchanged)."""
    out = _arms_before_f1(sim, q=q, b=b, rho=rho, memory=memory)
    out.update(f1_arms(sim, lambda: SimProposer(sim, q=q, b=b, rho=rho, memory=memory), getattr(sim, 'memory_facts', None)))
    return out


if __name__ == '__main__':
    main()
