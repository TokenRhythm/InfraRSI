"""opro and ltune (search/opro.py, search/ltune.py, search/llmopt.py; possible_search_algorithm/10 section 04): the
registration and the ablation switches, the one budget of 10 calls and 8 USD, the admission (values matched to the
table, a coupled knob completed, an illegal or ruled-out proposal not run and shown to the model, one proposed again
moved to the newest call, a configuration holding every change of a non-memory failure ruled out), the history (best
first, every outcome), opro's rounds (round 0 the prior's configurations, the rounds spread over the screens, the code
track's calls held back, the code track asked only when the knob side stalls), ltune's race (one call, probes and
short runs, the capping, at most two extended, the promotion rule) and its local refinement, the LLMProposer's
configuration call and the simulated one, every component off on the memory toy, and no knob, model or library name
in the modules."""
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from cand_helpers import run_kept, toy
from infrarsi.search import ABLATIONS, LLM_STRATEGIES, LTUNE_COMPONENTS, OPRO_COMPONENTS, STRATEGIES, resolve
from infrarsi.search.engine import Outcome
from infrarsi.search.llmopt import CALLS_MAX, USD_MAX
from infrarsi.search.ltune import CAP_MIN, FULL_MAX, LTune
from infrarsi.search.opro import OPRO
from infrarsi.search.proposers import CONFIG_RACE_RULE, CONFIG_ROUND_RULE, CONFIG_SYSTEM, LLMProposer, SimProposer
from infrarsi.search.toytrain import ToyTask, mem_tasks
from infrarsi.search.wcs import vk
from test_wcs import CAP, GAIN, KNOBS, START, PriorProposer, answer, engine, landscape, mem_of

SRC = Path(__file__).resolve().parents[3] / 'infrarsi' / 'search'


class BatchProposer(PriorProposer):
    """A scripted model: the prior (test_wcs.answer) and fixed configuration batches, one per call (the last repeats);
    keeps what each call was given."""

    def __init__(self, prior=None, batches=()):
        super().__init__(prior)
        self.batches, self.seen = list(batches), []

    def config_batch(self, engine, base, *, k=5, history=None, purpose='round'):
        self.calls += 1
        self.seen.append({'k': k, 'history': json.loads(json.dumps(history or [], default=str)), 'purpose': purpose})
        if not self.batches:
            return None
        return {'configs': self.batches[min(len(self.seen) - 1, len(self.batches) - 1)]}


def opro(prior=None, batches=(), **kw):
    eng = engine(**{k: kw.pop(k) for k in ('screens', 'effect', 'validate') if k in kw})
    st = OPRO(BatchProposer(prior, batches), facts={'cap_gib': 100.0}, seed=3, **kw)
    return st, eng


# ---------------------------------------------------------------- registration
def test_registration_ablations_and_resolve():
    assert {'opro', 'ltune'} <= set(STRATEGIES) and {'opro', 'ltune'} <= set(LLM_STRATEGIES)
    assert resolve('opro', has_llm=True) == 'opro' and resolve('ltune', has_llm=False) == 'ltune'
    assert len(OPRO_COMPONENTS) == 7 and all(ABLATIONS[k].startswith('opro: ') for k in OPRO_COMPONENTS)
    assert len(LTUNE_COMPONENTS) == 8 and all(ABLATIONS[k].startswith('ltune: ') for k in LTUNE_COMPONENTS)
    assert OPRO.proposer_features(['opro_code'])['self_proof'] is False and OPRO.proposer_features()['self_proof'] is True
    assert LTune.proposer_features(['ltune_code'])['self_proof'] is False and LTune.proposer_features()['self_proof'] is True
    assert (CALLS_MAX, USD_MAX) == (10, 8.0)
    on = OPRO(PriorProposer(None))
    assert on.on['wcs_prior'] and on.on['wcs_code'] and on.mledger is None and on.use_ledger
    off = OPRO(PriorProposer(None), off=sorted(OPRO_COMPONENTS))
    assert not off.on['wcs_prior'] and not off.on['wcs_code'] and not off.use_ledger and off.ladder.probe_below == 0.0
    assert not off.history_on and not off.filter_on
    lt = LTune(PriorProposer(None))
    assert not lt.on['wcs_prior'] and lt.on['wcs_code'] and lt.ladder.short       # one call: no prior call
    lt2 = LTune(PriorProposer(None), off=['ltune_one_call', 'ltune_short'])
    assert lt2.on['wcs_prior'] and not lt2.ladder.short


# ---------------------------------------------------------------- the budget
def test_one_budget_of_ten_calls_and_eight_dollars():
    st, eng = opro(answer())
    st._init(eng)
    assert st.proposer.calls == 1 and st._calls_left() == CALLS_MAX - 1 and st._budget_ok(eng)
    st.proposer.calls = CALLS_MAX
    assert not st._budget_ok(eng) and st._ask(eng, 'c0') == [] and st.proposer.seen == []    # spent: no call made
    st.proposer.calls = 3
    eng.s.ledger.used['llm_usd'] = USD_MAX                           # the dollars spent: no call either
    assert not st._budget_ok(eng) and st._ask(eng, 'c0') == []
    eng.s.ledger.used['llm_usd'] = 1.0
    assert st._budget_ok(eng) and st._budget_ok(eng, reserve=6) and not st._budget_ok(eng, reserve=7)
    st.proposer.calls_left = None
    st._allowance()
    assert st.proposer.calls_left == CALLS_MAX - 3                   # the proposer's re-asks stay within it too


def test_the_llm_proposer_configuration_call_and_its_allowance():
    from infrarsi.agent.backends.fake import ScriptedBackend
    from test_search import TOGGLE, setup
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    seen = []

    def script(role, system, prompt):
        seen.append((role, system, json.loads(prompt)))
        return {'configs': [{'change': {'fused_adam': True}, 'gain_pct': 3, 'sd_pct': 1, 'peak_gib': None, 'why': 'x'}],
                'notes': ''}
    prop = LLMProposer(ScriptedBackend(script), s.hooks, search_calibrates=True, features=OPRO.proposer_features())
    history_rows = [{'change': {'fused_adam': True}, 'outcome': 'x1.020 the reference speed'}]
    a = prop.config_batch(eng, 'c0', k=4, history=history_rows, purpose='round')
    assert a['configs'][0]['change'] == {'fused_adam': True} and s.llm_calls[-1]['role'] == 'configs'
    role, system, payload = seen[0]
    assert role == 'configs' and system == CONFIG_SYSTEM + CONFIG_ROUND_RULE
    assert payload['k'] == 4 and payload['history'] == history_rows and 'knob_table' in payload and 'task' in payload
    assert payload['knob_table']['fused_adam']['reference'] is False and prop.driver_calls == 1
    prop.config_batch(eng, 'c0', k=5, purpose='race')
    assert seen[-1][1] == CONFIG_SYSTEM + CONFIG_RACE_RULE and seen[-1][2]['history'] == []
    prop.calls_left = 0                                              # the allowance spent: the call is not made
    assert prop.config_batch(eng, 'c0') is None and len(seen) == 2
    assert LLMProposer(None, s.hooks).config_batch(eng, 'c0') is None    # no model: no call


# ---------------------------------------------------------------- admission and history
def test_admission_matches_completes_and_refuses():
    def check(cfg, c0):                                              # a coupled knob: micro-batch 2 needs tf32 on; recompute
        if cfg['mb'] == 2 and not cfg['tf32']:                       # off never with the loader at 4
            return (None,)
        return (None,) if cfg['rc'] == 'none' and cfg['wk'] == 4 else (cfg,)
    st, eng = opro(answer())
    eng.s.check_config = check
    st._init(eng)
    n0 = len(st.archive)                                             # (round 0: the prior's configuration)
    a = st._admit(eng, {'rc': 'half', 'mb': '2', 'nope': 1}, source='round', rnd=1, rank=0, gain=12.0, why='w')
    assert a['cfg'] == {**START, 'rc': 'half', 'mb': 2, 'tf32': True} and a['status'] == 'open'   # coerced, completed
    assert a['gain'] == pytest.approx(math.log1p(0.12)) and 0 < a['p'] <= 1
    assert [o[3][0] for o in st.surr.obs if o[3] is not None].count('config') >= 1   # the stated gain observed (wide)
    again = st._admit(eng, {'rc': 'half', 'mb': 2, 'tf32': True}, source='round', rnd=2, rank=3, gain=None)
    assert again is a and a['round'] == 2 and a['rank'] == 3 and len(st.archive) == n0 + 1   # moved
    assert st._admit(eng, {'rc': 'full'}, source='round', rnd=2, rank=0) is None       # the reference: nothing to run
    assert st._admit(eng, {'rc': 'none', 'wk': 4}, source='round', rnd=2, rank=1) is None
    assert st.rejected[-1]['why'] == 'no legal configuration holds these values'
    st._pf = lambda cfg, ids=(): (0.0, 'planted')
    assert st._admit(eng, {'liger': True}, source='round', rnd=2, rank=2) is None
    assert st.rejected[-1]['why'] == 'ruled out by the runs: planted' and st.rejected[-1]['change'] == {'liger': True}


def test_a_non_memory_failure_rules_out_what_holds_all_its_changes_and_retires_its_stated_gain():
    def effect(base, cfg, overlay):
        if cfg['tf32'] and cfg['wk'] == 0:
            return None, 'crash'
        return landscape(base, cfg, overlay)
    batch = [{'change': {'tf32': True, 'wk': 0}, 'gain_pct': 30}, {'change': {'liger': True}, 'gain_pct': 5}]
    st, eng = opro(None, [batch], screens=2, effect=effect, off=['opro_prior'])
    eng.run(st)
    bad = [a for a in st.archive if a['cfg']['tf32'] and a['cfg']['wk'] == 0][0]
    assert bad['outcome'] == 'crash' and not [o for o in st.surr.obs if o[3] == ('config', st._ckey(bad['cfg'], ()))]
    p, why = st._pf({**START, 'tf32': True, 'wk': 0, 'liger': True})
    assert p == 0.0 and 'which failed (crash)' in why
    assert st._pf({**START, 'tf32': True})[0] > 0 and st._pf({**START, 'wk': 0})[0] > 0   # one of them alone: open


def test_the_history_is_best_first_with_every_outcome():
    batches = [[{'change': {'rc': 'none'}, 'gain_pct': 25},                                  # out of memory
                {'change': {'rc': 'half', 'mb': 2, 'liger': True}, 'gain_pct': 20},          # fits, +21 %
                {'change': {'liger': True}, 'gain_pct': 5},                                  # +5 %
                {'change': {'comp': True}, 'gain_pct': 3}]]                                  # -5 %: aborted
    st, eng = opro(None, batches, screens=4, off=['opro_prior', 'opro_probe', 'opro_code'])
    eng.run(st)
    h = st._history(eng)
    measured = [r for r in h if r['outcome'].startswith('x')]
    assert len(measured) >= 2 and measured == sorted(measured, key=lambda r: -float(r['outcome'][1:6]))
    assert measured[0]['change'] == {'rc': 'half', 'mb': 2, 'liger': True}
    assert any(r['outcome'].startswith('out of memory') and r['change'] == {'rc': 'none'} for r in h)
    assert any(r['outcome'].startswith('aborted') and r['change'] == {'comp': True} for r in h)
    assert h.index(measured[-1]) < min(i for i, r in enumerate(h) if not r['outcome'].startswith('x'))
    st.rejected.append({'change': {'mb': 2}, 'why': 'planted', 'source': 'round'})
    assert st._history(eng)[-1] == {'change': {'mb': 2}, 'outcome': 'not run: planted'}
    assert st.proposer.seen[0]['history'] == [] and st.proposer.seen[0]['purpose'] == 'round'


# ---------------------------------------------------------------- opro's rounds
def test_opro_starts_with_the_priors_configurations_and_spreads_its_rounds():
    a = answer()
    a['configs'] = [{'name': 'aggressive', 'change': {'rc': 'none', 'mb': 2, 'liger': True}},
                    {'name': 'safe', 'change': {'liger': True}}]
    batches = [[{'change': {'rc': 'half', 'mb': 2, 'liger': True}, 'gain_pct': 20},
                {'change': {'rc': 'none', 'comp': True}, 'gain_pct': 25},
                {'change': {'tf32': True, 'liger': True}, 'gain_pct': 6}],
               [{'change': {'rc': 'none', 'comp': True, 'mb': 2}, 'gain_pct': 30},
                {'change': {'rc': 'half', 'mb': 2, 'liger': True, 'wk': 4}, 'gain_pct': 21}]]
    st, eng = opro(a, batches, screens=10)
    st._init(eng)
    assert [r['purpose'] for r in st.rounds] == ['prior'] and {x['source'] for x in st.archive} == {'prior'}
    assert st.round_left == math.ceil(10 / (CALLS_MAX - 1 + 1))         # 10 screens over this round and 9 more
    eng.run(st)
    d = st.describe()['opro']
    assert eng.s.ledger.used['screens'] == 10 and d['llm_calls'] <= CALLS_MAX
    rounds = [r for r in d['rounds'] if r['purpose'] == 'round']
    assert rounds and len(rounds) == st.proposer.calls - 1                # (no code track: no calls held back)
    history_rows = [s['history'] for s in st.proposer.seen]
    assert history_rows[0] == [] or history_rows[0][0]['outcome'] and any(history_rows[1:])
    first = [s for s in d['steps'] if s['pick'] == 'round'][0]
    assert first['round'] == 0                                       # round 0: the prior's configuration ran first
    st.proposer.hypotheses = lambda engine, base, ledger=None: []    # a model that writes code: two calls held back
    st.asks, st.proposer.calls = 0, 1
    assert st._rounds_payable(eng, 3) == CALLS_MAX - 1 - 2 and st._rounds_payable(eng, 2) == CALLS_MAX - 1
    st.asks = 2
    assert st._rounds_payable(eng, 3) == CALLS_MAX - 1 - 1               # (one ask of wcs's three left)
    st2, eng2 = opro(a, batches, screens=10, off=['opro_code'])
    st2.proposer.hypotheses = lambda engine, base, ledger=None: []
    st2._init(eng2)
    assert st2._rounds_payable(eng2, 10) == CALLS_MAX - 1                # no code track: none held back


def test_opro_orders_a_round_by_feasibility_times_the_stated_gain_and_keeps_the_unlikely_last():
    st, eng = opro(None, [[{'change': {'liger': True}, 'gain_pct': 5},
                           {'change': {'rc': 'half'}, 'gain_pct': 30},
                           {'change': {'tf32': True}, 'gain_pct': 40}]], off=['opro_prior'])
    st._init(eng)
    st._ask(eng, 'c0')
    p = {'liger': 0.9, 'rc': 0.5, 'tf32': 0.02}
    st._pf = lambda cfg, ids=(): (next((v for k, v in p.items() if vk(cfg[k]) != vk(START[k])), 1.0), 'planted')
    order = [next(k for k in p if vk(a['cfg'][k]) != vk(START[k])) for a in st._ranked(eng)]
    assert order == ['rc', 'liger']                                  # 0.5 x 30 % > 0.9 x 5 %; tf32 below P_MIN: left out
    st._pf = lambda cfg, ids=(): (0.02, 'planted')
    assert len(st._ranked(eng)) == 3                                 # nothing reaches P_MIN: all of them
    off = OPRO(BatchProposer(None, [[{'change': {'liger': True}, 'gain_pct': 5}, {'change': {'rc': 'half'}, 'gain_pct': 30}]]),
               off=['opro_prior', 'opro_filter'])
    off._init(eng)
    off._ask(eng, 'c0')
    assert [a['rank'] for a in off._ranked(eng)] == [0, 1] and all(a['p'] == 1.0 for a in off.archive)   # the model's order


def test_opro_asks_the_code_track_only_when_the_knob_side_stalls():
    st, eng = opro(answer())
    st._init(eng)
    st.inc_hist, st.knob_since_ask = ['c0', 'c0', 'c0'], 2
    assert st._code_due()
    st.inc_hist = ['c0', 'c1', 'c1']
    assert not st._code_due()                                        # it moved at the screen before the last one
    st.inc_hist, st.knob_since_ask = ['c1', 'c1', 'c1'], 1
    assert not st._code_due()                                        # two knob screens since the last ask, not one
    from test_search import BOTH, setup
    sim, s, eng2 = setup(BOTH, budget=12, job_gpu_hours=None)
    o = OPRO(SimProposer(sim, q=1.0, b=1.0, rho=0.0), seed=2)
    o.table_cap = 4000                                               # (the toy's 49k-row table sampled)
    eng2.run(o)
    d = o.describe()['opro']
    asks = [x for x in d['steps'] if x['pick'] == 'code ask']
    assert d['code_asks'] == len(asks) and all(x['stalled'] for x in asks) and d['llm_calls'] <= CALLS_MAX
    assert d['code_asks'] <= o.code_asks


# ---------------------------------------------------------------- ltune's race
def test_ltune_races_one_calls_configurations_on_the_memory_toy(monkeypatch):
    t = mem_tasks()[1]
    eng, sim, sp, facts = toy(t)
    calls = []
    orig = SimProposer.knob_prior
    monkeypatch.setattr(SimProposer, 'knob_prior', lambda self, *a, **kw: calls.append('prior') or orig(self, *a, **kw))
    st = LTune(sp(), facts=facts, seed=sim.task.seed % 2 ** 31)
    trace = eng.run(st)
    s = eng.s
    d = st.describe()['ltune']
    assert trace.stop_reason.startswith('budget') and s.ledger.used['screens'] == 10 and calls == []
    assert [r['purpose'] for r in d['rounds']] == ['race'] and d['rounds'][0]['asked'] == 5
    assert d['llm_calls'] <= 1 + st.code_asks
    rungs = [x['rung'] for x in d['race_log']]
    assert 'shorting' in rungs and st.fulls_raced <= FULL_MAX
    lows = [x for x in s.screens if x.get('fidelity') in ('probe', 'short')]
    assert lows and s.ledger.used['trials'] >= len([x for x in lows if x.get('fidelity') == 'short'])
    acc = d['ladder']['account']
    assert acc['lowfi_gpu_hours'] <= acc['allowance_gpu_hours'] + max(x['gpu_hours'] for x in s.screens)
    assert any(x['pick'] == 'race done' for x in d['steps'])
    assert {x['pick'] for x in d['steps']} - {'race done'} <= {'local', 'tail', 'code', 'draw'}


def test_ltune_caps_the_short_runs_and_extends_at_most_two():
    t = mem_tasks()[0]
    eng, sim, sp, facts = toy(t)
    st = LTune(PriorProposer(None), facts=facts, seed=1)
    st._init(eng)
    vals = [(k, v) for k in st.knobs for v in st.knobs[k] if vk(v) != vk(st.ref.get(k))][:4]
    items = [st._admit(eng, {k: v}, source='race', rnd=0, rank=i, gain=10.0) for i, (k, v) in enumerate(vals)]
    items = [a for a in items if a is not None]
    assert len(items) == 4
    gs = [0.10, 0.09, 0.07, -0.05]
    for a, g in zip(items, gs):
        a.update(stage='raced', gs=g, p=1.0)
    st.race = items
    st._refresh = lambda it: it['p']
    st.ladder.sigma_short = lambda engine: 0.005
    margin = max(2 * 0.005, math.log1p(CAP_MIN))
    q1 = st._race_step(eng, 'c0', 10)
    assert [a['stage'] for a in items] == ['full', 'raced', 'capped', 'capped'] and 0.10 - 0.07 > margin
    assert tuple(q1.fidelity) == ('full',) and st.by_hid[q1.hypothesis] is items[0]
    items[0]['stage'] = 'done'
    q2 = st._race_step(eng, 'c0', 9)
    assert st.by_hid[q2.hypothesis] is items[1] and st.fulls_raced == 2
    items[1]['stage'] = 'done'
    assert st._race_step(eng, 'c0', 8) is None                       # two extended at most; the capped stay out
    assert st.ladder.counts['promoted'] == 2
    nocap = LTune(PriorProposer(None), facts=facts, seed=1, off=['ltune_cap'])
    nocap._init(eng)
    its = [nocap._admit(eng, {k: v}, source='race', rnd=0, rank=i, gain=10.0) for i, (k, v) in enumerate(vals)]
    for a, g in zip(its, [0.10, -0.20, 0.07, 0.02]):
        a.update(stage='raced', gs=g, p=1.0)
    nocap.race, nocap._refresh = its, (lambda it: it['p'])
    nocap.ladder.sigma_short = lambda engine: 0.005
    nocap.ladder.incumbent_gain = lambda engine: 0.08                # an incumbent at +8 %: the promotion rule cuts
    q = nocap._race_step(eng, 'c0', 10)
    assert nocap.by_hid[q.hypothesis] is its[0] and its[1]['stage'] == 'raced'   # no capping
    its[0]['stage'] = 'done'
    assert nocap._race_step(eng, 'c0', 9) is None
    assert [a['stage'] for a in its[1:]] == ['not promoted', 'not promoted', 'not promoted']


def test_ltune_local_refinement_takes_the_best_neighbour_by_feasibility_and_gain():
    eng = engine()
    st = LTune(PriorProposer(answer()), facts={'cap_gib': 100.0}, seed=1, off=['ltune_one_call'])
    st._init(eng)
    assert st.prior.source == 'model'
    q = st._local(eng, 'c0')
    cfg = {k: q.change.get(k) for k in KNOBS}
    d = sum(1 for k in KNOBS if vk(cfg[k]) != vk(START[k]))
    assert 1 <= d <= 2 and q.hypothesis.startswith('local|')
    st._pf = lambda c, ids=(): ((1.0 if mem_of(c) <= CAP else 0.0), 'planted')
    q2 = st._local(eng, 'c0')
    c2 = {k: q2.change.get(k) for k in KNOBS}
    assert mem_of(c2) <= CAP and sum(GAIN.get((k, v), 0.0) for k, v in c2.items() if v != START[k]) > 0.05


# ---------------------------------------------------------------- the simulated configuration call
def test_the_simulated_configuration_call_reads_the_history():
    t = mem_tasks()[0]
    eng, sim, sp, facts = toy(t)
    p = sp()
    a = p.config_batch(eng, 'c0', k=5, history=[], purpose='round')
    b = sp().config_batch(eng, 'c0', k=5, history=[], purpose='round')
    assert a == b and 1 <= len(a['configs']) <= 5                    # deterministic per task and call
    first = a['configs'][0]['change']
    history_rows = [{'change': first, 'outcome': 'out of memory, short by 3.0 GiB'},
            {'change': dict(list(first.items())[:1]), 'outcome': 'failed: crash'}]
    c = sp().config_batch(eng, 'c0', k=5, history=history_rows, purpose='round')
    keys = [json.dumps(sorted(x['change'].items()), default=str) for x in c['configs']]
    assert json.dumps(sorted(first.items()), default=str) not in keys   # nothing in the history again
    kv = list(first.items())[0]
    assert not any(x['change'].get(kv[0]) == kv[1] for x in c['configs'])   # a value two failures held: left out
    r = sp().config_batch(eng, 'c0', k=5, history=[], purpose='race')
    assert r['configs'] and r != a and all(x['why'] == 'simulated' for x in r['configs'])


# ---------------------------------------------------------------- every component off, the names
@pytest.mark.parametrize('cls,off', [(OPRO, [c]) for c in OPRO_COMPONENTS] + [(OPRO, sorted(OPRO_COMPONENTS))]
                         + [(LTune, [c]) for c in LTUNE_COMPONENTS] + [(LTune, sorted(LTUNE_COMPONENTS))])
def test_every_component_off_runs(monkeypatch, cls, off):
    row, st = run_kept(monkeypatch, mem_tasks()[2], lambda sp, facts, sim: cls(sp, facts=facts, off=off, seed=5), budget=6)
    d = st.describe()[st.name]
    assert row['screens'] == 6 and d['off'] == sorted(off) and d['llm_calls'] <= CALLS_MAX


@pytest.mark.parametrize('module', ['llmopt', 'opro', 'ltune'])
def test_the_modules_name_no_knob_model_or_library(module):
    src = (SRC / f'{module}.py').read_text(encoding='utf-8')
    code = '\n'.join(ln for ln in src.splitlines() if not ln.strip().startswith('#'))
    for word in ('recompute', 'micro_batch', 'liger', 'compile', 'fsdp', 'deepspeed', 'megatron', 'torch', 'qwen', 'gemma',
                 'llama', 'flash', 'expandable', 'checkpointing'):
        assert not re.search(rf'\b{word}\b', code, re.I), (module, word)


def test_the_toybench_runs_the_new_arms_and_keeps_the_old_ones():
    from infrarsi.search.toybench import arms, run_one
    sim = SimpleNamespace(task=SimpleNamespace(seed=1), memory_facts=None)
    names = set(arms(sim))
    assert {'opro', 'opro-nohist', 'ltune', 'ltune-prior', 'random', 'wcs', 'esls', 'rpp', 'rpp-ctl'} <= names
    for arm in ('opro', 'ltune'):
        r = run_one(mem_tasks()[3], arm, 6, job_factor=None, proposer={'memory': True}, bind='screens')
        assert r['arm'] == arm and r['screens'] == 6 and r['stop'].startswith('budget')
