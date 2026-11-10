"""The review C4_r1 of opro / ltune (reports/infrarsi_newmodels_20261028/reviews/C4_r1_out.md): one regression test per
finding, each failing on the code before the fix.

    1 (P1)  ltune: a local short run promoted after a patch entered the chain runs on its own candidate (no resubmitted
            configuration + overlay coming back as a duplicate of it)
    2 (P1)  every model call is capped at min(the session's dollars left, 8 USD - spent), the session without a USD cap
    3 (P1)  the backend's own JSON re-asks count against the 10 calls, and with one attempt left no second request is made
    4 (P2)  admission keeps every value the model named (the reference-valued ones too) and refuses a value the knob table
            does not have
    5 (P2)  every race full run of ltune, the unraced fallback included, counts against full_max

and of its round 2 (C4_r2_out.md): a legal null value is admitted (not taken for a value the table lacks) and held
through the legality completion; a value the table lacks is still refused.
"""
import dataclasses

from cand_helpers import toy
from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.search.llmopt import CALLS_MAX, USD_MAX
from infrarsi.search.ltune import FULL_MAX, LTune
from infrarsi.search.opro import OPRO
from infrarsi.search.proposers import LLMProposer
from infrarsi.search.toytrain import mem_tasks
from infrarsi.search.wcs import vk
from test_c4 import opro
from test_wcs import PriorProposer, engine


class PricedBackend(ScriptedBackend):
    """A scripted backend that charges `usd` a call - at most the call's max_usd, as the real backends keep to it - and,
    with `reask`, re-asks a malformed first reply internally as often as its `max_retries` allows (each re-ask an
    attempt, LLMResult.attempts); keeps every call's keywords and the max_retries each call saw."""

    def __init__(self, script, *, usd: float = 4.5, max_retries: int = 0, reask: bool = False):
        super().__init__(script, tokens_per_call=2000)
        self.usd, self.max_retries, self.reask = usd, max_retries, reask
        self.kw, self.retries_seen, self.attempts_made = [], [], 0

    def complete(self, system, prompt, **kw):
        self.kw.append(dict(kw))
        self.retries_seen.append(self.max_retries)
        n = 1 + (self.max_retries if self.reask else 0)
        res = super().complete(system, prompt, **{k: v for k, v in kw.items() if k in ('schema', 'call_dir', 'role',
                                                                                       'expect_json')})
        cap = kw.get('max_usd')
        self.attempts_made += n
        return dataclasses.replace(res, llm_usd=n * self.usd if cap is None else min(n * self.usd, float(cap)), attempts=n)


def script(role, system, prompt):
    if role == 'prior':
        return {'levels': [{'knob': 'fused_adam', 'value': True, 'gain_pct': 3, 'sd_pct': 2}], 'interactions': [],
                'configs': [], 'notes': ''}
    return {'configs': [{'change': {'fused_adam': True}, 'gain_pct': 3, 'sd_pct': 1, 'peak_gib': None, 'why': 'x'}],
            'notes': ''}


def llm_opro(backend):
    from test_search import TOGGLE, setup
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    prop = LLMProposer(backend, s.hooks, search_calibrates=True, features=OPRO.proposer_features())
    return OPRO(prop, facts={'cap_gib': 80.0}, seed=1), s, eng


def test_r1_1_a_promoted_local_short_run_runs_on_its_own_candidate_after_a_patch_is_adopted():
    eng, sim, sp, facts = toy(mem_tasks()[0], 10)
    st = LTune(PriorProposer(None), facts=facts, seed=1)
    st._init(eng)
    st.race, st.race_done = [], True
    st.chain, st.overlay_best = ['p9'], {'patches.txt': 'adopted_patch\n'}
    st.patches['p9'] = {'files': dict(st.overlay_best), 'status': 'proven', 'mechanism': 'an adopted patch'}
    cfg = next(c for c in st._table(eng) if c != st.ref and st._pf(c, ('p9',))[0] >= 0.9)
    q = st._knob_q(eng, 'c0', cfg, kind='local', plan=('short',), p=1.0)
    assert q.patch == st.overlay_best                                # the chain's overlay rides on it
    out = eng.evaluate(q)
    assert out.status == 'short'
    st.ladder.promote = lambda engine, o, label='': True             # it may beat the incumbent
    st.observe(eng, q, out)
    q2 = st.pending.pop(0)
    assert eng._fresh(q2)
    out2 = eng.evaluate(q2)
    assert out2.status in ('screened', 'aborted', 'invalid') and out2.cid == out.cid, out2   # its own candidate in full
    st.observe(eng, q2, out2)
    assert eng.full_screens() == 1 and st.runs[out.cid]['fidelity'] == 'full'
    assert st._ckey(cfg, ('p9',)) in st.tried and st.ideas[q2.hypothesis] == frozenset({'p9'})


def test_r1_2_every_model_call_is_capped_at_the_dollars_left():
    be = PricedBackend(script, usd=4.5)
    st, s, eng = llm_opro(be)
    assert getattr(s.ledger.budget, 'llm_usd', None) is None         # (the manifest's default: no session USD cap)
    st._init(eng)                                                    # the prior call
    st._ask(eng, 'c0')                                               # a configuration call
    st._ask(eng, 'c0')                                               # nothing left: not sent
    assert [k.get('max_usd') for k in be.kw] == [USD_MAX, USD_MAX - 4.5]
    assert s.ledger.used['llm_usd'] <= USD_MAX + 1e-9 and len(be.kw) == 2


def test_r1_3_the_backends_re_asks_count_against_the_calls_and_stop_at_the_allowance():
    be = PricedBackend(script, usd=0.001, max_retries=1, reask=True)     # every reply re-asked once while allowed
    st, s, eng = llm_opro(be)
    st._init(eng)
    for _ in range(3 * CALLS_MAX):
        if not st._budget_ok(eng):
            break
        st._ask(eng, 'c0')
    assert be.attempts_made <= CALLS_MAX and st._llm_used() == be.attempts_made
    be2 = PricedBackend(script, usd=0.001, max_retries=1, reask=True)
    st2, s2, eng2 = llm_opro(be2)
    st2._init(eng2)
    st2._bdriver.attempts = CALLS_MAX - 1                            # one attempt left
    n0 = be2.attempts_made
    st2._ask(eng2, 'c0')
    assert be2.attempts_made - n0 == 1 and be2.retries_seen[-1] == 0 and be2.max_retries == 1   # no second request
    assert st2._llm_used() == CALLS_MAX and not st2._budget_ok(eng2)


def test_r1_4_admission_keeps_every_named_value_and_refuses_values_the_table_lacks():
    st, eng = opro(None, off=['opro_prior'])
    eng.s.check_config = lambda cfg, c0: (None,) if cfg['mb'] == 2 and not cfg['tf32'] else (cfg,)   # mb 2 needs tf32
    st._init(eng)
    assert st.ref['tf32'] is False
    assert st._admit(eng, {'mb': 2, 'tf32': False}, source='round', rnd=1, rank=0, gain=8.0) is None   # named: kept
    assert st.rejected[-1]['why'] == 'no legal configuration holds these values'
    a = st._admit(eng, {'mb': 2}, source='round', rnd=1, rank=1, gain=8.0)
    assert a['cfg']['mb'] == 2 and a['cfg']['tf32'] is True                  # not named: completed
    assert st._admit(eng, {'rc': 'half', 'mb': 7}, source='round', rnd=1, rank=2, gain=8.0) is None
    assert st.rejected[-1]['why'] == 'not a value of the knob table: mb=7'
    assert not [x for x in st.archive if vk(x['cfg']['rc']) == vk('half') and x['cfg']['mb'] == 1]


def test_r1_5_every_race_full_run_counts_against_full_max():
    eng, sim, sp, facts = toy(mem_tasks()[0], 10)
    st = LTune(PriorProposer(None), facts=facts, seed=1)
    st._init(eng)
    vals = [(k, v) for k in st.knobs for v in st.knobs[k] if vk(v) != vk(st.ref.get(k))]
    items = []
    for i, (k, v) in enumerate(vals):
        a = st._admit(eng, {k: v}, source='race', rnd=0, rank=i, gain=10.0)
        if a is not None:
            items.append(a)
        if len(items) == 5:
            break
    for a in items:
        a.update(stage='new', p=1.0)
    st.race, st._refresh = items, (lambda it: it['p'])
    st.ladder.affordable = lambda engine, fid, code=False: False     # the low-fidelity allowance spent: no short rung
    fulls = []
    for _ in range(10):
        q = st._race_step(eng, 'c0', 10)
        if q is None:
            break
        fulls.append(q)
        st.by_hid[q.hypothesis]['stage'] = 'done'
    assert len(fulls) == FULL_MAX == 2 and all(tuple(q.fidelity)[-1] == 'full' for q in fulls)
    assert sorted(a['stage'] for a in items) == ['done', 'done', 'not extended', 'not extended', 'not extended']


def test_r2_a_legal_null_value_is_admitted_and_held_through_completion():
    knobs = {'fa': [False, True], 'wk': [0, 2], 'nw': [None, 1, 8]}       # nw: a knob with a legal null (its reference)
    eng = engine(knobs=knobs, start={'fa': False, 'wk': 0, 'nw': None}, effect=lambda base, cfg, overlay: (0.0, None))
    eng.s.check_config = lambda cfg, c0: (cfg,) if not cfg['fa'] or cfg['wk'] == 2 or cfg['nw'] == 1 else (None,)
    st = OPRO(PriorProposer(None), facts={'cap_gib': 100.0}, seed=3, off=['opro_prior'])
    st._init(eng)
    assert st.ref['nw'] is None
    a = st._admit(eng, {'nw': None, 'fa': True}, source='round', rnd=1, rank=0, gain=5.0)
    assert a is not None and a['cfg'] == {'fa': True, 'wk': 2, 'nw': None}   # the null held; the completion moved wk
    assert st._admit(eng, {'nw': 'null', 'fa': True, 'wk': 2}, source='round', rnd=2, rank=0, gain=5.0) is a
    c = st._admit(eng, {'fa': True}, source='round', rnd=2, rank=1, gain=5.0)
    assert c['cfg'] == {'fa': True, 'wk': 0, 'nw': 1}                  # (not named: the completion may move it)
    assert st._admit(eng, {'nw': 5, 'fa': True}, source='round', rnd=2, rank=2, gain=5.0) is None
    assert st.rejected[-1]['why'] == 'not a value of the knob table: nw=5'
    assert not [x for x in st.rejected if 'nw=None' in x['why']]
