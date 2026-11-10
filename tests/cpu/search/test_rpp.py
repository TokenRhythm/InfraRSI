"""rpp and rpp-ctl (search/rpp.py; possible_search_algorithm/20 section 6): the registration, the draw (a = 0 and T = inf:
uniform over the legal table less what ran; a = 1: feasibility filter and weight; the annealed weight on the
feasibility-weighted gain; only what is predicted to fit once no probe is left), the control's missing model call, the
rung ladder on the memory toy (probes and short runs counted apart from the screens), every component off, and no knob,
model or library name in the candidates' modules."""
import math
import re
from collections import Counter
from pathlib import Path

import pytest

from cand_helpers import run_kept, toy
from infrarsi.search import ABLATIONS, ESLS_COMPONENTS, LLM_STRATEGIES, RPP_COMPONENTS, STRATEGIES, resolve
from infrarsi.search.ladder import NOTE
from infrarsi.search.proposers import SimProposer
from infrarsi.search.rpp import P_MIN, RPP, weighted_draw
from infrarsi.search.toytrain import mem_tasks
from test_wcs import KNOBS, START, PriorProposer, engine

CTL = ['rpp_prior', 'rpp_anneal', 'rpp_code']
SRC = Path(__file__).resolve().parents[3] / 'infrarsi' / 'search'


def test_registration_ablations_and_resolve():
    assert {'esls', 'rpp', 'rpp-ctl'} <= set(STRATEGIES) and {'esls', 'rpp'} <= set(LLM_STRATEGIES)
    assert 'rpp-ctl' not in LLM_STRATEGIES and resolve('rpp-ctl', has_llm=False) == 'rpp-ctl'
    assert len(RPP_COMPONENTS) == 5 and all(ABLATIONS[k].startswith('rpp: ') for k in RPP_COMPONENTS)
    assert len(ESLS_COMPONENTS) == 9 and all(ABLATIONS[k].startswith('esls: ') for k in ESLS_COMPONENTS)
    assert RPP.proposer_features(['rpp_code'])['self_proof'] is False and RPP.proposer_features()['self_proof'] is True


def flat(seed=3, **kw):
    """rpp on the wcs test landscape with a flat prior (no model answer), initialised; (strategy, engine)."""
    eng = engine()
    r = RPP(PriorProposer(None), facts={'cap_gib': 100.0}, seed=seed, **kw)
    r._init(eng)
    return r, eng


def key(cfg):
    return tuple(sorted((k, repr(cfg[k])) for k in KNOBS))


def test_a0_tinf_is_uniform_over_the_legal_table_less_what_ran():
    r, eng = flat()
    n_legal = len(r._candidates(eng, 'c0'))
    assert n_legal == 143                                             # 144 legal, less the reference that ran
    n = 50 * n_legal
    counts = Counter(key(weighted_draw(r, eng, 'c0', a=0.0, T=math.inf, salt=i)[0]) for i in range(n))
    assert key(START) not in counts and len(counts) == n_legal
    chi2 = sum((c - n / n_legal) ** 2 / (n / n_legal) for c in counts.values())
    assert chi2 < (n_legal - 1) + 4 * math.sqrt(2 * (n_legal - 1))   # a chi-square within 4 sd of its mean


def test_a1_drops_the_unlikely_weights_by_feasibility_and_keeps_to_the_likely_without_probes():
    r, eng = flat()

    def pf(cfg, ids=()):
        return (0.01 if cfg['rc'] == 'none' else 0.9 if cfg['mb'] == 1 else 0.3), 'planted'
    r._pf = pf
    draws = [weighted_draw(r, eng, 'c0', a=1.0, T=math.inf, salt=i)[0] for i in range(3000)]
    assert not [c for c in draws if c['rc'] == 'none'] and P_MIN > 0.01   # below P_MIN: never drawn
    ratio = sum(c['mb'] == 1 for c in draws) / sum(c['mb'] == 2 for c in draws)
    assert 2.5 < ratio < 3.6                                          # 0.9 / 0.3 over equally many configurations
    likely = [weighted_draw(r, eng, 'c0', a=1.0, T=math.inf, salt=i, p_min=0.7)[0] for i in range(300)]
    assert all(c['mb'] == 1 for c in likely)                          # no probe left: only what is predicted to fit
    r._pf = lambda cfg, ids=(): (0.3, 'planted')                      # ... unless nothing is: the draw still happens
    assert weighted_draw(r, eng, 'c0', a=1.0, T=math.inf, salt=0, p_min=0.7) is not None


def test_annealing_weights_the_feasibility_weighted_gain():
    r, eng = flat()
    r.surr.observe(r._feats({**START, 'rc': 'none'}), 0.30, 0.001)   # predicted fast, but unlikely to fit
    r.surr.observe(r._feats({**START, 'rc': 'half'}), 0.10, 0.001)   # predicted faster than the rest, fits
    r._pf = lambda cfg, ids=(): ((0.06 if cfg['rc'] == 'none' else 1.0), 'planted')

    def share(T, rc):
        draws = [weighted_draw(r, eng, 'c0', a=1.0, T=T, salt=i)[0] for i in range(600)]
        return sum(c['rc'] == rc for c in draws) / len(draws)
    assert share(math.inf, 'half') == pytest.approx(48 / (48 + 48 + 48 * 0.06), abs=0.06)   # feasibility only
    assert share(0.01, 'half') > 0.95 and share(0.01, 'none') < 0.03   # annealed: the fast one that fits
    r.k_valid = 3
    assert r._temperature() == pytest.approx(r.T0 / 4)
    assert RPP(PriorProposer(None), temperature=math.inf)._temperature() == math.inf
    assert RPP(PriorProposer(None), off=['rpp_anneal'])._temperature() == math.inf


def test_the_control_makes_no_model_call(monkeypatch):
    calls = Counter()
    orig = SimProposer.knob_prior

    def counted(self, *a, **kw):
        calls['prior'] += 1
        return orig(self, *a, **kw)
    monkeypatch.setattr(SimProposer, 'knob_prior', counted)
    t = mem_tasks()[0]
    row, ctl = run_kept(monkeypatch, t, lambda sp, facts, sim: RPP(sp, facts=facts, off=CTL, temperature=math.inf,
                                                                   name='rpp-ctl', seed=sim.task.seed % 2 ** 31))
    assert calls['prior'] == 0 and row['screens'] == 10 and ctl.describe()['rpp']['temperature'] == 'inf'
    assert ctl.describe()['name'] == 'rpp-ctl' and ctl.prior.source != 'model'
    row, full = run_kept(monkeypatch, t, lambda sp, facts, sim: RPP(sp, facts=facts, seed=sim.task.seed % 2 ** 31))
    assert calls['prior'] == 1 and full.prior.source == 'model'


def test_the_ladder_on_the_memory_toy_counts_probes_and_short_runs_apart():
    t = mem_tasks()[1]
    engine_, sim, sp, facts = toy(t)
    r = RPP(sp(), facts=facts, seed=sim.task.seed % 2 ** 31)
    trace = engine_.run(r)
    s = engine_.s
    assert trace.stop_reason.startswith('budget') and s.ledger.used['screens'] == 10
    rows = Counter(x.get('fidelity', 'full') for x in s.screens)
    d = r.describe()['rpp']
    lad = d['ladder']
    assert rows['short'] >= 1 and lad['counts']['short'] >= rows['short'] and s.ledger.used['trials'] >= rows['short']
    acc = lad['account']
    assert acc['by_fidelity']['short']['runs'] == rows['short'] and acc['note'] == NOTE.format(share=0.5)
    assert acc['lowfi_gpu_hours'] <= acc['allowance_gpu_hours'] + max(x['gpu_hours'] for x in s.screens)
    assert lad['counts']['promoted'] + lad['counts']['not_promoted'] >= 1
    assert all(x['plan'] for x in d['draws'])


@pytest.mark.parametrize('off', [[c] for c in RPP_COMPONENTS] + [CTL])
def test_every_component_off_runs(monkeypatch, off):
    row, r = run_kept(monkeypatch, mem_tasks()[2], lambda sp, facts, sim: RPP(sp, facts=facts, off=off, seed=5))
    assert row['screens'] == 10 and r.describe()['rpp']['off'] == sorted(off)


def test_a0_rpp_runs_as_random_with_the_rungs(monkeypatch):
    row, r = run_kept(monkeypatch, mem_tasks()[0], lambda sp, facts, sim: RPP(
        sp, facts=facts, off=CTL + ['rpp_ledger'], a=0.0, temperature=math.inf, name='rpp-a0', seed=5))
    assert row['screens'] == 10 and r.describe()['rpp']['a'] == 0.0


@pytest.mark.parametrize('module', ['ledger', 'ladder', 'tail', 'cand_base', 'rpp', 'esls'])
def test_the_candidates_modules_name_no_knob_model_or_library(module):
    src = (SRC / f'{module}.py').read_text(encoding='utf-8')
    code = '\n'.join(ln for ln in src.splitlines() if not ln.strip().startswith('#'))
    for word in ('recompute', 'micro_batch', 'liger', 'compile', 'fsdp', 'deepspeed', 'megatron', 'torch', 'qwen', 'gemma',
                 'llama', 'flash', 'expandable', 'checkpointing'):
        assert not re.search(rf'\b{word}\b', code, re.I), (module, word)


def test_rpp_keeps_to_wcs_s_one_model_call_allowance():
    from infrarsi.agent.backends.fake import ScriptedBackend
    from infrarsi.search.proposers import LLMProposer
    from infrarsi.search.wcs import LLM_CALLS_MAX
    from test_search import TOGGLE, setup
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    seen = []

    def script(role, system, prompt):
        seen.append(role)
        return {'levels': [{'knob': 'fused_adam', 'value': True, 'gain_pct': 3, 'sd_pct': 2,
                            'memory': {'direction': 'neutral', 'gib': 0}}], 'interactions': [], 'configs': [], 'notes': ''}
    prop = LLMProposer(ScriptedBackend(script), s.hooks, search_calibrates=True, features=RPP.proposer_features())
    r = RPP(prop, facts={'cap_gib': 80.0})
    r._init(eng)
    assert seen == ['prior'] and prop.calls_left == LLM_CALLS_MAX - 1
    d = r.describe()['rpp']
    assert d['llm_calls'] == 1 and d['llm_calls_max'] == LLM_CALLS_MAX
    spent = LLMProposer(ScriptedBackend(script), s.hooks, search_calibrates=True, features=RPP.proposer_features())
    spent.driver_calls = LLM_CALLS_MAX                                # the allowance already spent: no prior call
    r2 = RPP(spent, facts={'cap_gib': 80.0})
    r2._init(eng)
    assert seen == ['prior'] and r2.prior.source == 'flat' and r2.propose(eng, 16)


def test_a_short_run_supersedes_the_priors_configuration_suggestions(monkeypatch):
    from types import SimpleNamespace
    from infrarsi.search.engine import Outcome
    from test_wcs import answer
    a = answer()
    a['configs'] = [{'name': 'aggressive', 'change': {'rc': 'none', 'mb': 2, 'liger': True}, 'gain_pct': 40, 'sd_pct': 1}]
    eng = engine()
    r = RPP(PriorProposer(a), facts={'cap_gib': 100.0}, seed=3)
    r._init(eng)
    assert [o[3][0] for o in r.surr.obs if o[3] is not None] == ['config']   # wcs's: the stated gain as a wide observation
    monkeypatch.setattr(r.ladder, 'sigma_short', lambda engine: 0.01)        # (the test engine has no short rung)
    monkeypatch.setattr(r.ladder, 'promote', lambda engine, out, label='': False)
    cfg = {**START, 'mb': 2}
    p = r._knob_q(eng, 'c0', cfg, kind='draw', plan=('short',), p=1.0)
    eng.s.candidates['s1'] = SimpleNamespace(id='s1', parent='c0', config=dict(cfg), overlay={}, status='short',
                                             results={'short': {'valid': True}}, verdict={'valid': True}, receipt={})
    r.observe(eng, p, Outcome('s1', 'short', True, None, 0.02, 'short', 0.1, [], []))
    assert not [o for o in r.surr.obs if o[3] is not None]            # mb=2 measured: the suggestion holding it went
    assert [o[1] for o in r.surr.obs if o[3] is None] == [pytest.approx(0.02)]   # the short run, with doubled noise
    assert [o[2] for o in r.surr.obs if o[3] is None] == [pytest.approx(0.02)]


def test_the_stores_warning_lowers_the_ledger_feasibility_by_what_it_says():
    r, eng = flat()
    cfg = {**START, 'mb': 2}
    p0, _ = r._pf(cfg)
    assert r.mledger is not None and p0 > 0.4                         # (above P_UNSURE: the store may lower it)
    r.exp = {'check': lambda c: ['memory (estimate): about 145 GiB of 80 (closed-form accounting, +-25 %)']
             if c.get('mb') == 2 else []}
    p, why = r._pf(cfg)
    assert p == pytest.approx(min(p0, 0.03)) and 'the store warns' in why   # wcs's level: far beyond the card
    r.exp = {'check': lambda c: ['memory: run-x/c3 ran out of memory']}
    assert r._pf(cfg)[0] == pytest.approx(min(p0, 0.1))
    r.mledger.observe_run(cfg, 'ok', peak=50.0, source='m1')          # it ran and fitted: its own result outranks the store
    assert r._pf(cfg)[0] == 1.0
