"""The finalists' shared parts (search/finalist_base.py; FINALISTS_IMPL_SPEC_20261101 A.0 S1-S5) through a minimal finalist
registered only in the tests (finalist_helpers.Dummy): the base and its constants, its composition with another
LLMOptBase strategy (the OPRO mixin form); S1 - an ask only with no patch pending, a screen after this one and a run since the
last, held back by a strong knob candidate unless the screen is protected, asks by budget beyond wcs's three, the last two
calls the code track's, three patches per ask, the three prompt fields, a no_patch review that keeps the track open; S2 -
the brief and the packs' patch sections read from the home only (own numbers left out), the patch section as the brief
builder reads an arm, its write-back field; S3 - a candidate derived from a failed run carries that run's patches with the
chain, the rescue too; S4 - the ledger alone at the card less 2 %, the declared-memory floor until a measurement,
isolated_blame by combination, a measured failure deciding its own configuration and its supersets probed first (never in
full without a probe); S5 - a probe at 95 % of the card is thrash and not extended; base f1.1's probe accounting (B.9
P2: the code-probe pool of four proofs outside the allowance, repairs included, the knob side never on it, reported
apart; the probe / full ratio rule; S6 still gating; a whole run whose knob probes spend the allowance and whose patches
still get their proofs); an engine-level ten-screen run with its accounting (screens, probes, short runs, model calls,
dollars); every part off; the other strategies' files byte for byte; no knob, model or library name in the module."""
import difflib
import hashlib
import itertools
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from cand_helpers import run_kept, toy
from finalist_helpers import DUMMY_COMPONENTS, CodeProposer, Dummy, fin, llm_toy, llm_toy_setup
from hls2_helpers import BASE_TEXT, PATCH_FILE, SimEngine, SimSession, snippet
from infrarsi.search import STRATEGIES
from infrarsi.search.cand_base import CandidateBase
from infrarsi.search.engine import Outcome
from infrarsi.search.finalist_base import (BLOCKED_VALUE, CARD_RESERVE, CODE_CALLS, CODE_PROBES_MAX, FB_PARTS,
                                             FB_VERSION, P_FLOOR_DECLARED, PATCHES_PER_ASK, RATIO_MAX, RATIO_PROBE_BELOW,
                                             REVIEW_VERDICT, THRASH_PEAK, FinalistBase, FinalistLadder,
                                             pack_patch_entries, site_of)
from infrarsi.search.ladder import CODE_RESERVE, LOWFI_SHARE, PROBE_BELOW, RungLadder, account
from infrarsi.search.ledger import SD_DECLARED, THRASH_LO
from infrarsi.search.llmopt import CALLS_MAX, HISTORY_MAX, USD_MAX, LLMOptBase
from infrarsi.search.opro import OPRO
from infrarsi.search.toytrain import MEM_KNOBS, make_sim, mem_tasks
from infrarsi.search.wcs import CODE_ASKS as WCS_CODE_ASKS
from infrarsi.search.wcs import WCS
from test_wcs import KNOBS, START, answer
from test_xfer import FAR, NEAR, SAME_CARD, brief, pack

SRC = Path(__file__).resolve().parents[3] / 'infrarsi' / 'search'
FIELDS = {'measured_history', 'transfer', 'fix_inert'}


def ph(name, mech=None, mu=0.03):
    """A patch of train_patches.py (the simulated engine's file) as the proposer returns one."""
    return {'mechanism': mech or f'patch {name}', 'change': {}, 'patch': {PATCH_FILE: BASE_TEXT + snippet(name)}, 'mu': mu,
            'sigma': 0.02, 'p_feasible': 0.9, 'site': name}


def run(st, eng, q):
    o = eng.evaluate(q)
    st.observe(eng, q, o)
    return o


# ---------------------------------------------------------------- the base
def test_the_base_is_a_cooperative_llmopt_subclass_with_the_spec_constants():
    assert issubclass(FinalistBase, LLMOptBase) and issubclass(FinalistLadder, RungLadder)
    assert (CODE_CALLS, PATCHES_PER_ASK, HISTORY_MAX, P_FLOOR_DECLARED, BLOCKED_VALUE, THRASH_PEAK) == (2, 3, 40, 0.15, 0.25, 0.95)
    assert (SD_DECLARED, LOWFI_SHARE, CODE_RESERVE, PROBE_BELOW, CARD_RESERVE) == (0.5, 0.5, 0.2, 0.7, 0.02)
    st = Dummy(CodeProposer())
    assert (st.calls_max, st.usd_max, st.code_asks, st.patches_per_ask) == (CALLS_MAX, USD_MAX, CALLS_MAX, 3)
    assert st.code_asks > WCS_CODE_ASKS                                     # S1: bounded by the budget, not by three
    assert isinstance(st.ladder, FinalistLadder) and st.ladder.short is False and st.ladder.probe_below == 0.7
    assert st.ladder.lowfi_share == 0.5 and st.use_ledger and all(st.fb_on.values())
    assert FB_VERSION == 'f1.1' and (CODE_PROBES_MAX, RATIO_MAX, RATIO_PROBE_BELOW) == (4, 0.6, 0.5)   # B.9 P2
    assert st.ladder.pool and st.ladder.code_probes_max == 4 and st.ladder.code_reserve == 0.2 and st.ladder.borrow
    nob = Dummy(CodeProposer(), off=['fb_code_borrow'])                      # (the pool, then f1.0's share - its borrow
    assert nob.ladder.pool and not nob.ladder.borrow and nob.ladder.code_reserve == 0.2   # switched off)
    off = Dummy(CodeProposer(), off=sorted(DUMMY_COMPONENTS))
    assert not off.on['wcs_code'] and not off.use_ledger and not any(off.fb_on.values())
    assert isinstance(off.ladder, FinalistLadder) and not off.ladder.borrow and off.ladder.code_reserve == 0.0
    assert not FinalistLadder(pool=False).pool and FinalistLadder(pool=False).code_probes_max == 0   # (f1.0's accounting)
    assert not any(n in STRATEGIES for n in ('finalist', 'fbdummy'))       # (the base registers nothing)


def test_a_finalist_can_put_the_base_before_another_llmopt_strategy(monkeypatch):
    class H(FinalistBase, OPRO):                                            # the OPRO mixin form
        name = 'h'
    assert H.__mro__[1:4] == (FinalistBase, OPRO, LLMOptBase)
    row, st = run_kept(monkeypatch, mem_tasks()[1], lambda sp, facts, sim: H(sp, facts=facts, seed=2), budget=5)
    assert row['screens'] == 5 and st.mledger.margin == pytest.approx(0.98) and st.code_asks == CALLS_MAX
    assert st.opro_on['opro_code'] and isinstance(st.ladder, FinalistLadder) and st.ladder.short is False
    assert H(CodeProposer(), off=['opro_probe']).ladder.probe_below == 0.0   # (a strategy's own probe switch is kept)


# ---------------------------------------------------------------- S1: the code track by budget
def test_s1_an_ask_needs_no_pending_patch_a_screen_after_this_one_and_a_run_since_the_last():
    st, eng = fin(None, answers=lambda ctx, n: [ph(f'a{n}')])               # a flat prior: no knob candidate holds back
    st._init(eng)
    assert st._code(eng, 'c0', 1) == [] and st.asks == 0                    # the last screen: nothing after it
    q = st._code(eng, 'c0', 2)                                               # a screen after this one: an ask
    assert st.asks == 1 and len(q) == 1 and q[0].hypothesis.startswith('patch|')
    assert st.fb_asks[-1]['why'].startswith('no knob candidate') and st.fb_asks[-1]['sent']
    assert st._code(eng, 'c0', 5) == q and st.asks == 1                     # a patch pending: no ask, the queue back
    st.queue = []
    assert st._code(eng, 'c0', 5) == [] and st.asks == 1                    # no run since the last ask: none
    st.fb_runs += 1
    assert len(st._code(eng, 'c0', 5)) == 1 and st.asks == 2


def test_s1_asks_go_on_by_budget_beyond_wcs_three_and_stop_when_it_is_spent():
    st, eng = fin(None, answers=lambda ctx, n: [ph(f'b{n}')])
    st._init(eng)
    assert st.proposer.calls == 1                                            # the prior call
    for _ in range(20):
        st.queue = []
        st.fb_runs += 1
        st._code(eng, 'c0', 5)
    assert st.asks == CALLS_MAX - 1 > WCS_CODE_ASKS and st.proposer.calls == CALLS_MAX == st._llm_used()
    assert not st._budget_ok(eng)                                            # spent: no more asks
    assert [a['queued'] for a in st.fb_asks] == [[f'p{i}'] for i in range(1, CALLS_MAX)]


def test_s1_a_strong_knob_candidate_holds_the_ask_back_unless_the_screen_is_protected(monkeypatch):
    st, eng = fin(answer(), answers=lambda ctx, n: [ph(f'c{n}')])           # the prior: recompute half +8 %, ...
    st._init(eng)
    strong = st._knob_strong(eng, 'c0')
    assert strong and strong['predicted_gain_pct'] >= 2.0 and strong['p_feasible'] >= 0.5
    assert st._code(eng, 'c0', 5) == [] and st.asks == 0 and st.fb_counts['held_by_knob_side'] == 1
    assert st.finalist_view()['held_back_by'] == strong
    st.protected_screens = frozenset({1})                                    # the next screen is protected: an ask
    assert len(st._code(eng, 'c0', 5)) == 1 and 'protected' in st.fb_asks[-1]['why']
    st.queue, st.protected_screens = [], frozenset({5})
    st.fb_runs += 1
    assert st._code(eng, 'c0', 5) == [] and st.asks == 1                    # (screen 1 again, not protected)
    monkeypatch.setattr(Dummy, '_knob_strong', lambda self, engine, inc: None)   # nothing strong: an ask
    assert len(st._code(eng, 'c0', 5)) == 1 and st.asks == 2


def test_s1_the_last_two_calls_are_the_code_tracks_while_it_can_ask():
    def configs(n):
        return [{'change': {'liger': True, 'wk': 4 if n % 2 else 0}, 'gain_pct': 5}]
    st, eng = fin(None, answers=lambda ctx, n: [ph(f'd{n}')], configs=configs)
    st._init(eng)
    st.proposer.calls = CALLS_MAX - 3                                        # three calls left: a configuration call
    assert st._ask(eng, 'c0') and st.proposer.calls == CALLS_MAX - 2
    assert st._ask(eng, 'c0') == [] and st.last_got is None and st.proposer.calls == CALLS_MAX - 2   # two left: code's
    assert st.fb_counts['knob_calls_held'] == 1
    assert st._knob_strong(eng, 'c0')                                        # (the call's +5 % entered the time model)
    st.fb_runs, st.protected_screens = st.fb_runs + 1, frozenset({1})
    assert len(st._code(eng, 'c0', 5)) == 1 and st.proposer.calls == CALLS_MAX - 1   # ... and the code track uses them
    eng.s.ledger.used['screens'] = 9                                         # one screen left: no ask can come
    assert st._fb_knob_call_ok(eng)
    eng.s.ledger.used['screens'] = 0
    st.fb_code_idle = True                                                   # an ask that could make no call: released
    assert st._fb_knob_call_ok(eng)
    off, eng2 = fin(None, configs=configs, off=['fbd_code'])                 # no code track: the knob side's
    off._init(eng2)
    off.proposer.calls = CALLS_MAX - 2
    assert off._fb_knob_call_ok(eng2) and off._ask(eng2, 'c0')


def test_s1_at_most_three_patches_per_ask_patches_only():
    many = [ph(x) for x in 'abcde']
    knob = {'mechanism': 'a knob', 'change': {'liger': True}, 'patch': None}
    st, eng = fin(None, answers=lambda ctx, n: [knob] + many)
    st._init(eng)
    q = st._code(eng, 'c0', 5)
    assert len(q) == 3 == len(st.queue) and all(x.hypothesis.startswith('patch|') for x in q)


def test_s1_every_patch_call_carries_the_measured_history_the_transfer_and_the_fix():
    """On the memory toy (the real engine, the toy's proof lines): an out-of-memory run and a patch that does not prove
    itself feed the next asks' `measured_history` (the shortfall, the patch's result) and `fix_inert` (its record, once);
    the transfer field shows the brief's proven entries by weighted gain, one per site; none of the three outside an ask,
    and wcs's own patch call has none of them."""
    from infrarsi.search.toybench import toy_env
    task = next(t for t in mem_tasks() if not t.antagonism)
    eng, sim, sp, facts = toy(task, 10)
    near, far = (sim.cap, 1, facts['params_b'] * 1.1), (sim.cap * 2, 2, 999.0)
    bf = brief(proven=[('p001', 'fn_a', 6.0, near), ('p002', 'fn_b', 9.0, far), ('p003', 'fn_a', 4.0, near)],
               inert=[('i001', 'fn_c', 'not activated: no `active` line', near)], framework='other')
    box = {}

    def answers(ctx, n):
        if n == 1:
            return [{'mechanism': 'patch na', 'change': {}, 'patch': {'patches.txt': 'na\n'}, 'mu': 0.03, 'site': 'na'}]
        if n == 2:
            return [{'mechanism': f'[fix {box["pid"]}] patch nb', 'change': {}, 'patch': {'patches.txt': 'nb\nproof:nb\n'},
                     'mu': 0.03, 'site': 'nb'}]
        return []
    st = Dummy(CodeProposer(None, answers), facts=facts, seed=1, brief=bf, env=toy_env(sim))
    st._init(eng)
    big = max((c for c in st._table(eng) if sim.memory(c) > sim.cap), key=lambda c: sim.memory(c))
    o = run(st, eng, st._knob_q(eng, 'c0', big, kind='t', plan=('full',), p=0.9))
    assert o.status == 'invalid' and st.runs[o.cid]['kind'] == 'oom'
    q = st._code(eng, 'c0', 9)[0]                                           # the first ask
    box['pid'] = q.hypothesis.split('|')[1]
    ctx = st.proposer.seen[-1]
    rows = ctx['measured_history']['rows']
    assert ctx['measured_history']['rule'].startswith('MEASURED HISTORY') and len(rows) <= HISTORY_MAX
    assert any(r['outcome'].startswith('out of memory, short by') for r in rows)
    view = ctx['transfer']
    assert view['rule'].startswith('TRANSFER') and [r['id'] for r in view['proven']] == ['p001', 'p002']
    assert view['proven'][0]['runs'] == 2 and view['do_not_repeat'][0]['site'] == 'fn_c'   # fn_a twice: p003 counted
    assert 'fix_inert' not in ctx and st.fb_asks[-1]['transfer']['proven'] == ['p001', 'p002']
    o = run(st, eng, q)                                                      # no proof line: inert at its probe
    assert o.status == 'infeasible' and st.runs[o.cid]['kind'] == 'inert_at_start' and eng.full_screens() == 1
    assert st.fb_inert_log == [{'pid': box['pid'], 'cid': o.cid, 'base': 'c0', 'shown': False, 'screens_done': 1}]
    q2 = st._code(eng, 'c0', 9)[0]                                          # the second ask: fix this one
    ctx = st.proposer.seen[-1]
    fix = ctx['fix_inert']
    assert fix['rule'].startswith('FIX THIS ONE') and fix['id'] == box['pid'] and fix['mechanism'] == 'patch na'
    assert fix['reason'].startswith('it printed no') and fix['added_lines'] == ['patches.txt: na']
    assert any(r.get('patch_result', '').startswith('did not prove at its probe') for r in ctx['measured_history']['rows'])
    assert st.fb_inert_log[0]['shown'] and st.fb_fixes == {q2.hypothesis.split('|')[1]: box['pid']}
    o = run(st, eng, q2)                                                     # its fix proves itself
    assert o.status == 'screened'
    st._code(eng, 'c0', 8)
    assert 'fix_inert' not in st.proposer.seen[-1]                           # shown once
    assert not FIELDS & set(st.context(eng, 'c0'))                          # (only during an ask)
    w = WCS(CodeProposer(None), facts=facts, seed=1)
    w._init(eng)
    assert not FIELDS & set(w.context(eng, 'c0'))                           # wcs's call as it is


def _review_script(verdict):
    calls = {'n': 0}

    def script(role, system, payload):
        if role == 'prior':
            return {'levels': [], 'interactions': [], 'configs': [], 'notes': ''}
        if role == 'code_review':
            return {'verdict': verdict, 'summary': 'nothing only code can remove', 'findings': []}
        if role == 'configs':
            return {'configs': [], 'notes': ''}
        calls['n'] += 1
        nm = f'r{calls["n"]}'
        return {'hypotheses': [{'mechanism': f'patch {nm}', 'change': {}, 'patch': {'patches.txt': f'{nm}\nproof:{nm}\n'},
                                'predicted_gain_pct': 3, 'site': nm}]}
    return script


def test_s1_a_no_patch_review_does_not_close_the_code_track():
    r = llm_toy(_review_script('no_patch'), screens=6, usd_per_call=0.5, protected=range(1, 7))
    roles = [x[0] for x in r.seen]
    hyp = [p for role, p, *_ in r.seen if role == 'hypotheses']
    assert roles.count('code_review') == 1 and len(hyp) >= 2                # asks went on after the no_patch review
    assert all('code' in p and p['code_review']['verdict'] == REVIEW_VERDICT for p in hyp)
    assert r.prop.review['verdict'] == 'no_patch' and '_review' not in r.prop.__dict__   # put back as it answered
    assert 'complete' not in r.st._bdriver.__dict__                           # (the dispatch spy undone)
    w = llm_toy(_review_script('no_patch'), screens=6, usd_per_call=0.5, protected=range(1, 7), off=['fb_review_lift'])
    assert [x[0] for x in w.seen].count('hypotheses') == 0 and w.st.fb_code_idle   # wcs's behaviour: the track closes


# ---------------------------------------------------------------- S2: patch memory across jobs
SECTION = {'proven': [{'mechanism': 'chunked loss', 'site': 'LossFn.forward', 'site_text': 'mod.LossFn.forward',
                       'gain_pct': 7.0, 'proof': {'found': 'a', 'set': 'b'}}, {'mechanism': 'bad', 'gain_pct': -1.0}],
           'inert': [{'mechanism': 'cache flag', 'site': 'attn_cache', 'reason': 'not activated: no `active` line',
                      'printed': ['[infrarsi-patch] attn_cache: found a -> set b']}, 'junk']}


def test_s2_the_brief_and_the_packs_patch_sections_come_from_the_home_with_the_plan_only(tmp_path, monkeypatch):
    from infrarsi.experience.pack import write_lines
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path))
    runs = tmp_path / 'experience' / 'runs'
    runs.mkdir(parents=True)
    a = pack('run-a', NEAR, [('c0', {}, 1.0), ('c1', {'mb': 2}, 0.9)])
    a[0]['patches'] = SECTION
    own = pack('run-own', (141, 1, 27.0), [('c0', {}, 1.0)])               # this job's own numbers: left out
    own[0]['patches'] = {'proven': [{'mechanism': 'mine', 'site': 'x_fn', 'gain_pct': 30.0}]}
    skip = pack('run-skip', FAR, [('c0', {}, 1.0)])                          # not in the plan's packs: not read
    skip[0]['patches'] = {'proven': [{'mechanism': 'skipped', 'site': 'y_fn', 'gain_pct': 50.0}]}
    for pk in (a, own, skip):
        write_lines(runs / f'{pk[0]["pack_id"]}.jsonl', pk)
    bf = brief(proven=[('p001', 'Norm.forward', 5.0, NEAR), ('p002', 'own_fn', 40.0, (141, 1, 27.0))],
               inert=[('i001', 'conv_fn', 'self-proof failed: again', SAME_CARD)])
    (tmp_path / 'experience' / 'transfer_brief.json').write_text(json.dumps(bf), encoding='utf-8')
    plan = {'fingerprint': {'memory_gib': 141, 'nodes': 1, 'gpus_per_node': 8, 'model': {'params_b': 27.0},
                            'framework': 'hf_trainer'}, 'packs': ['run-a', 'run-own']}
    st, eng = fin(answer(), experience=plan)
    st._init(eng)
    assert [e['id'] for e in st.fb_proven] == ['p001', 'ph001'] and [e['id'] for e in st.fb_inert] == ['i001', 'ih001']
    assert st.fb_sources == {'brief': 3, 'packs': 3} and st.fb_target['card_gib'] == 141.0
    assert any("this job's own numbers left out" in n for n in st.fb_notes)
    assert st.fb_inert[1]['printed'] == ['[infrarsi-patch] attn_cache: found a -> set b']
    view = st._fb_transfer_view()
    assert [r['id'] for r in view['proven']] == ['ph001', 'p001']           # 4.5 x 7 before 4.5 x 5
    assert view['proven'][0]['proof'] == {'found': 'a', 'set': 'b'}
    assert [x['mechanism'] for x in view['do_not_repeat']] == ['cache flag', 'mechanism i001']   # by weight: 4.5, 3
    assert view['do_not_repeat'][0]['printed'] and 'run-a' not in json.dumps(view)   # never a source's name
    none, eng2 = fin(answer())                                               # no experience plan: nothing read
    none._init(eng2)
    assert none.fb_proven == none.fb_inert == [] and any('no experience plan' in n for n in none.fb_notes)
    assert none._fb_transfer_view() is None
    off, eng3 = fin(answer(), experience=plan, off=['fb_transfer'])
    off._init(eng3)
    assert off.fb_proven == [] and any('transfer part is off' in n for n in off.fb_notes)
    proven, inert, n = pack_patch_entries([a, [{'kind': 'pack', 'patches': SECTION}]])   # (a pack without numbers: none)
    assert n == 1 and [e['mechanism'] for e in proven] == ['chunked loss'] and len(inert) == 1


def test_s2_the_patch_section_reads_as_the_brief_builder_reads_an_arm():
    from infrarsi.experience.pack import validate
    from infrarsi.experience.runs import _patches
    st, eng = fin(answer())
    st._init(eng)
    site = 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5RMSNorm.forward'
    st.patches = {'p1': {'status': 'proven', 'adopted': True, 'gain_pct': 4.5, 'mechanism': 'review: m.py: fused norm',
                         'site': site},
                  'p2': {'status': 'proven', 'adopted': False, 'gain_pct': 0.4, 'mechanism': 'tiny', 'site': 'tiny_fn'},
                  'p3': {'status': 'proven', 'adopted': False, 'gain_pct': 1.8, 'mechanism': 'inconclusive'},
                  'p4': {'status': 'inert at its probe', 'mechanism': 'cache flag', 'site': 'attn_cache'},
                  'p5': {'status': 'oom', 'mechanism': 'memory'}, 'p6': {'status': 'queued', 'mechanism': 'never ran'},
                  'p7': {'status': 'slower', 'mechanism': 'slow one'}, 'p8': {'status': 'crash', 'mechanism': 'broken'}}
    st.inert = [{'candidate': 'c9', 'mechanism': 'cache flag', 'site': 'attn_cache', 'at': 'probe',
                 'reason': 'attn_cache: no `active` line - the patched code never ran within the probe',
                 'printed': ['[infrarsi-patch] attn_cache: found a -> set b'],
                 'parsed': {'attn_cache': {'found': 'a', 'set': 'b', 'active': False}}}]
    st.fb_patch_cid = {'p4': 'c9'}
    sec = st.patch_section()
    assert [x['mechanism'] for x in sec['proven']] == ['fused norm'] and sec['proven'][0]['site'] == 'RMSNorm.forward'
    assert sec['proven'][0]['gain_pct'] == 4.5 and sec['proven'][0]['site_text'] == site
    inert = {x['mechanism']: x for x in sec['inert']}
    assert set(inert) == {'tiny', 'cache flag', 'slow one', 'broken'}       # oom, queued, inconclusive left out
    assert inert['tiny']['reason'] == 'no measurable gain: +0.40 % at its full run'
    assert inert['cache flag']['reason'].startswith('not activated:') and inert['cache flag']['printed']
    assert inert['cache flag']['parsed'] == st.inert[0]['parsed']
    assert inert['slow one']['reason'] == 'slower: aborted against the incumbent'
    assert inert['broken']['reason'].startswith('failed at its run (crash)')
    d = st.describe()
    assert d['patches'] == sec and _patches({'extra': {'strategy': d}}) == sec
    head = {'kind': 'pack', 'id': 'pack', 'schema': 'infrarsi-exp/1', 'pack_id': 'run-x', 'title': 'x', 'license': 'x',
            'tier': 'verified', 'patches': sec}
    assert validate([head])[0] == []                                         # (a header field: every tree reads the pack)
    w = WCS(CodeProposer(answer()), facts={'cap_gib': 100.0})
    w._init(eng)
    assert _patches({'extra': {'strategy': w.describe()}}) is None           # the other strategies write none
    assert site_of(None, 'kernels.attention_cost', 'x') == 'attention_cost'


# ---------------------------------------------------------------- S3: candidates derived from a failed run
TWO = 'def one():\n    pass\n\n\ndef two():\n    pass\n'
PATCH_A = {PATCH_FILE: 'def one():\n    _m_a()\n\n\ndef two():\n    pass\n'}
PATCH_B = {PATCH_FILE: 'def one():\n    pass\n\n\ndef two():\n    _m_b()\n'}


def patch_oom(base, cfg, overlay):
    """Patch `a` runs out of memory unless the fused loss frees what it needs; the rest fit (+1 % per value, patch `b`
    +5 %)."""
    def g(c, ov):
        return 0.01 * sum(1 for k, v in c.items() if v != START[k]) + (0.05 if '_m_b(' in ''.join((ov or {}).values()) else 0)
    if '_m_a(' in ''.join((overlay or {}).values()) and not cfg['liger']:
        return None, 'oom'
    return g(cfg, overlay) - g(base.config, base.overlay), None


def test_s3_a_candidate_derived_from_a_failed_patch_run_carries_its_patches_with_the_chain(monkeypatch):
    s = SimSession(START, knobs=KNOBS, screens=10, base_files={PATCH_FILE: TWO})
    eng = SimEngine(s, patch_oom, sigma_runs=[0.01 / math.sqrt(2)])
    st = Dummy(CodeProposer(answer()), facts={'cap_gib': 100.0}, seed=3)
    st._init(eng)
    qa = st._patch_proposal(eng, 'c0', {'mechanism': 'patch a', 'change': {}, 'patch': dict(PATCH_A), 'mu': 0.03})
    oa = run(st, eng, qa)
    assert oa.status == 'infeasible' and oa.fidelity == 'probe' and st.runs[oa.cid]['kind'] == 'oom'
    assert st.derived_ids(oa.cid) == ('p1',) and tuple(st.chain) == ()      # wcs's _rescue carried the chain only
    q = st.derived_q(eng, 'c0', {**START, 'liger': True}, oa.cid, kind='complete', why='the fused loss frees what it needs')
    assert q is not None and st.ideas[q.hypothesis] == frozenset({'p1'}) and '_m_a()' in q.patch[PATCH_FILE]
    assert st.ledger[q.hypothesis]['derived_from'] == oa.cid and st.ledger[q.hypothesis]['patches'] == ['p1']
    assert tuple(q.fidelity)[0] == 'probe'                                  # it holds the failed combination: a probe
    o = run(st, eng, q)
    assert o.status == 'screened' and st.pids[o.cid] == frozenset({'p1'})
    qb = st._patch_proposal(eng, eng.incumbent(), {'mechanism': 'patch b', 'change': {}, 'patch': dict(PATCH_B), 'mu': 0.03})
    run(st, eng, qb)                                                         # patch b proves and is adopted
    assert st.chain == ['p3']
    q2 = st.derived_q(eng, eng.incumbent(), {**START, 'liger': True, 'tf32': True}, oa.cid, kind='repair')
    assert st.ideas[q2.hypothesis] == frozenset({'p1', 'p3'})                # the run's patch with the chain, merged
    assert '_m_a()' in q2.patch[PATCH_FILE] and '_m_b()' in q2.patch[PATCH_FILE]
    edge = FinalistBase.failed_material(st)[0]
    monkeypatch.setattr(Dummy, 'failed_material', lambda self: [dict(edge, kind='thrash', probe_peak=True)])
    qr = st._rescue(eng, eng.incumbent())                                    # the edge rescue: the run's patch too
    assert qr is not None and qr.hypothesis.startswith('rescue|') and st.ideas[qr.hypothesis] == frozenset({'p1', 'p3'})
    assert st.counts['rescue_probes'] == 1 and st.ledger[qr.hypothesis]['derived_from'] == oa.cid
    gone = st.derived_q(eng, eng.incumbent(), {**START, 'liger': True, 'wk': 4}, 'c-none', kind='repair')
    assert gone is None                                                      # (a run that left no candidate: nothing)


# ---------------------------------------------------------------- S4: memory as a price
def test_s4_the_ledger_alone_decides_at_the_card_less_two_percent():
    st, eng = fin(answer(), cap=40.0, ref_peak=30.0)
    st._init(eng)
    led = st.mledger
    assert led.margin == pytest.approx(1 - CARD_RESERVE) and led.cap == 40.0
    cfg = {**START, 'tf32': True}
    assert st._pf(cfg)[0] == pytest.approx(led.p_feasible(cfg)) and 'the line 39.2 of 40' in st._pf(cfg)[1]
    st.exp = {'check': lambda c: ['memory: a measured neighbour ran out of memory at about 50 GiB of 40 GiB']}
    assert st._pf(cfg)[0] == pytest.approx(led.p_feasible(cfg))             # the store's warning is not a source
    assert CandidateBase._pf(st, cfg, ())[0] <= 0.1                         # (the candidates' reading lowers it)
    assert st._p_feasible(cfg, ()) == st._pf(cfg, ())                       # wcs's steps read S4's feasibility too
    o = run(st, eng, st._knob_q(eng, 'c0', {**START, 'rc': 'half'}, kind='t', plan=('full',), p=0.9))
    assert o.status == 'screened' and st._pf({**START, 'rc': 'half'}) == (1.0, 'it ran and fitted')


def test_s4_declared_gib_never_push_p_below_the_floor_until_measured():
    st, eng = fin(answer(), cap=40.0, ref_peak=35.0)                        # recompute none declared +60 GiB, half +20
    st._init(eng)
    led = st.mledger
    none, half = {**START, 'rc': 'none', 'wk': 4}, {**START, 'rc': 'half', 'wk': 4}
    assert led.p_feasible(none) < P_FLOOR_DECLARED and st._pf(none)[0] == pytest.approx(P_FLOOR_DECLARED)
    assert 'declared memory of 1 unmeasured value' in st._pf(none)[1]
    assert led.p_feasible(half) < P_FLOOR_DECLARED and st._pf(half)[0] == pytest.approx(P_FLOOR_DECLARED)
    led.observe_run({**START, 'rc': 'half'}, 'ok', peak=31.0, source='cX')   # a measured peak: the measurement decides
    led.observe_run({**START, 'rc': 'none'}, 'oom', oom={'need_gib': 52.0}, source='cY')
    assert st._pf(half)[0] == pytest.approx(led.p_feasible(half)) and st._pf(half)[0] > 0.9
    assert st._pf(none)[0] == pytest.approx(led.p_feasible(none)) and st._pf(none)[0] < P_FLOOR_DECLARED


def mb_ooms(base, cfg, overlay):
    """Micro-batch 2 runs out of memory with recompute half or none or with tf32; everything else fits."""
    if cfg['mb'] == 2 and (cfg['rc'] != 'full' or cfg['tf32']):
        return None, 'oom'
    return 0.01 * sum(1 for k, v in cfg.items() if v != START[k]), None


def test_s4_isolated_blame_lowers_a_value_alone_only_when_its_failures_others_fitted():
    st, eng = fin(answer(), effect=mb_ooms, off=['fb_s6'])                  # (three OOMs in a row on purpose: S6 off)
    st._init(eng)
    for extra in ({'rc': 'half'}, {'rc': 'none'}, {'tf32': True}):          # three OOMs sharing micro-batch 2
        o = run(st, eng, st._knob_q(eng, 'c0', {**START, 'mb': 2, **extra}, kind='t', plan=('full',), p=0.9))
        assert st.runs[o.cid]['kind'] == 'oom'
    probe = {**START, 'mb': 2, 'liger': True}
    assert st.isolated_blame() == {}                                         # the others never fitted: no blame
    on = st._pf(probe)[0]
    st.fb_on['fb_isolated_blame'] = False
    assert on == pytest.approx(st._pf(probe)[0])                            # its P unchanged by the rule
    st.fb_on['fb_isolated_blame'] = True
    assert ('mb', '2') in WCS._blocked_values(st)                            # (wcs's rule would have lowered it)
    for extra in ({'rc': 'half'}, {'rc': 'none'}, {'tf32': True}):          # the others fit alone
        run(st, eng, st._knob_q(eng, 'c0', {**START, **extra}, kind='t', plan=('full',), p=0.9))
    assert set(st.isolated_blame()) == {('mb', '2')}                          # now micro-batch 2 alone is blamed
    p, why = st._pf(probe)
    st.fb_on['fb_isolated_blame'] = False
    assert p == pytest.approx(BLOCKED_VALUE * st._pf(probe)[0]) and 'failed alone' in why


def test_s4_a_failure_decides_its_own_configuration_and_its_supersets_need_a_passed_probe(monkeypatch):
    import infrarsi.search.llmopt as llmopt
    from test_xfer import pair_oom
    st, eng = fin(answer(), effect=pair_oom)
    st._init(eng)
    pair = {**START, 'rc': 'half', 'mb': 2}
    o = run(st, eng, st._knob_q(eng, 'c0', pair, kind='t', plan=('full',), p=0.9))
    assert st.runs[o.cid]['kind'] == 'oom' and st._pf(pair)[0] == 0.0       # its own configuration: ruled out
    sup = {**pair, 'liger': True}
    p, why = st._pf(sup)
    assert p > 0.0 and 'failed for memory' in why
    combos = st.oom_combinations
    st.oom_combinations = lambda: []
    assert st._pf(sup)[0] == pytest.approx(p)                               # P stays the ledger's (S4: the only source)
    st.oom_combinations = combos
    sub = {**START, 'rc': 'half', 'liger': True}                             # a subset: not of the combination
    assert st._fb_combo(sub) is None and 'failed for memory' not in st._pf(sub)[1]
    q = st._knob_q(eng, 'c0', sup, kind='t', plan=('full',), p=p)           # a full plan: its probe forced first
    assert q is not None and tuple(q.fidelity) == ('probe', 'full') and q.gate is not None
    assert st.fb_counts['combo_probed'] == 1
    st.ladder._fund = lambda engine, cost, code=False: False                 # the allowance spent: not run at all
    assert st._knob_q(eng, 'c0', sup, kind='t', plan=('full',), p=p) is None and st.fb_counts['combo_refused'] == 1
    item = {'cfg': sup, 'key': st._ckey(sup, ()), 'round': 0, 'rank': 0, 'gain': None, 'why': 'x', 'p': p, 'why_p': '',
            'status': 'open', 'source': 'test'}
    st.archive.append(item)
    assert st._run_item(eng, 'c0', item, kind='round') is None and item['status'] == 'refused'
    assert st.rejected[-1]['why'].startswith('it holds a combination measured to fail')
    other = {**START, 'liger': True, 'wk': 4}
    drawn = []

    def draw(s, engine, inc, **kw):                                          # the draw prefers the member
        cfg = sup if s._pf(sup)[0] > 0 else other
        drawn.append(cfg is sup)
        return cfg, (), s._pf(cfg)[0], 'drawn', {}
    monkeypatch.setattr(llmopt, 'weighted_draw', draw)
    q = st._fallback(eng, 'c0')
    assert drawn == [True, False] and q is not None and {k: q.change[k] for k in START} == other
    st.tail_base, st.tail.done = None, False
    st.surr.mean = lambda: {('rc', '"half"'): 0.05, ('mb', '2'): 0.05, ('liger', 'true'): 0.05}
    st._knob_cfg = lambda engine, inc: dict(START)
    st.mem.split = lambda base, cfg, known: {'spends': [], 'frees': [], 'unknown': []}
    assert st._tail_step(eng, 'c0', 5) is None and st.tail.done              # a tail holding the pair: refused, ended


def test_s4_a_failed_combination_stays_off_full_screens_over_a_whole_run(monkeypatch):
    """The brief's hard rule 5 over ten screens, the ledger's own P pinned at 0.95 (only the rule can catch them):
    configuration calls keep offering supersets of a pair that always runs out of memory; a configuration holding the whole
    combination of an earlier failure never runs in full without its probe first, and each is caught at its probe."""
    from infrarsi.search.ledger import MemoryLedger
    from test_xfer import pair_oom
    monkeypatch.setattr(MemoryLedger, 'p_feasible', lambda self, cfg, ids=(): 0.95)

    def configs(n):
        return [{'change': {'rc': 'half', 'mb': 2, **extra}, 'gain_pct': 20 - i} for i, extra in
                enumerate(({'liger': True}, {'tf32': True}, {'wk': 0}, {'wk': 4}, {'liger': True, 'tf32': True}))]
    st, eng = fin(answer(), configs=configs, effect=pair_oom)
    log, ev = [], eng.evaluate

    def evaluate(p):
        o = ev(p)
        log.append((tuple(p.fidelity), {k: p.change.get(k) for k in START} if p.absolute else None, o.status, o.fidelity))
        return o
    eng.evaluate = evaluate
    eng.run(st)
    failed, members = [], []
    for plan, cfg, status, fidelity in log:
        if cfg is None:
            continue
        held = {(k, v) for k, v in cfg.items() if v != START[k]}
        if any(f <= held for f in failed):
            members.append((plan, status, fidelity))
        if status in ('invalid', 'infeasible'):
            failed.append(held)
    assert members and all(plan[0] == 'probe' and fidelity == 'probe' for plan, _, fidelity in members), members
    assert eng.full_screens() == 10 and st.fb_counts['combo_probed'] >= len(members)


# ---------------------------------------------------------------- S5: thrash at a probe by its peak
def _edge_task():
    """A memory toy without the crash pair with two configurations at 95 % of the card or more (below the card)."""
    rows = [dict(zip(MEM_KNOBS, v)) for v in itertools.product(*MEM_KNOBS.values())]
    for t in mem_tasks():
        if t.antagonism:
            continue
        sim = make_sim(t)
        hi = [c for c in rows if THRASH_PEAK * sim.cap <= sim.memory(c) < sim.cap]
        if len(hi) >= 2:
            return t, hi
    raise AssertionError('no edge task')


def test_s5_a_probe_at_95_percent_of_the_card_is_thrash_and_not_extended():
    task, hi = _edge_task()
    eng, sim, sp, facts = toy(task, 10)
    st = Dummy(CodeProposer(None), facts=facts, seed=1)
    st._init(eng)
    q = st._knob_q(eng, 'c0', hi[0], kind='t', plan=('probe', 'full'), p=0.5)
    o = run(st, eng, q)
    assert o.status == 'infeasible' and o.fidelity == 'probe' and o.failed_checks == ['probe_thrash']
    assert eng.full_screens() == 0 and st.runs[o.cid]['kind'] == 'thrash'   # not extended: no screen spent
    assert st.mledger.ran_status({k: hi[0].get(k) for k in st.knobs}) == 'thrash' and st._pf(hi[0])[0] == 0.0
    [t] = st.fb_thrash
    assert t['candidate'] == o.cid and t['share'] >= THRASH_PEAK
    assert st.ledger[q.hypothesis]['status'] == 'thrash at its probe'
    m = st.failed_material()[0]
    assert m['run'] == o.cid and m['kind'] == 'thrash' and m['at_probe'] and m['probe_peak']
    assert m['gap_gib'] == pytest.approx(t['peak_gib'] - THRASH_LO * sim.cap)
    assert any(r['outcome'].startswith('thrash at its probe') for r in st._history(eng))
    q2 = st._knob_q(eng, 'c0', hi[1], kind='t', plan=('probe',), p=0.5)
    o2 = run(st, eng, q2)                                                    # a probe-only plan: classified the same
    assert o2.status == 'probed' and st.runs[o2.cid]['kind'] == 'thrash' and len(st.fb_thrash) == 2
    off = Dummy(CodeProposer(None), facts=facts, seed=1, off=['fb_peak_thrash'])
    eng2, *_ = toy(task, 10)
    off._init(eng2)
    o3 = run(off, eng2, off._knob_q(eng2, 'c0', hi[0], kind='t', plan=('probe', 'full'), p=0.5))
    assert 'probe_thrash' not in (o3.failed_checks or []) and off.fb_thrash == []   # the part off: no peak rule
    gate = st._fb_gated(lambda e, cid, base, row: ('inner', 'patch_inert'))  # a patch's gate: the peak rule first
    assert gate(eng, 'cx', 'c0', {'memory_peak_gib': 0.96 * sim.cap})[1] == 'probe_thrash'
    assert gate(eng, 'cx', 'c0', {'memory_peak_gib': 0.5 * sim.cap}) == ('inner', 'patch_inert')


# ---------------------------------------------------------------- the ladder
def test_f11_the_code_probe_pool_holds_four_proofs_repairs_included_outside_the_allowance():
    """Base f1.1 (B.9 P2): a patch's self-proof probe, a derived candidate's that carries an unadopted patch (a repair)
    and a repair patch's (`[fix <id>]`) are paid first by the code-probe pool - four per arm, outside the low-fidelity
    allowance, their GPU-hours reported apart; the fifth falls back to the allowance's code share (F1 (b), as in f1.0)."""
    s = SimSession(START, knobs=KNOBS, screens=10, base_files={PATCH_FILE: TWO})
    eng = SimEngine(s, patch_oom, sigma_runs=[0.01 / math.sqrt(2)], probe_cost=0.25)
    fix = {}
    st = Dummy(CodeProposer(answer(), answers=lambda ctx, n: [dict(fix)]), facts={'cap_gib': 100.0}, seed=3)
    st._init(eng)
    lad = st.ladder
    share = lad.allowance(eng, code=True)
    assert lad.pool and lad.code_probes_left() == CODE_PROBES_MAX == 4 and share == pytest.approx(0.3)   # (CODE_RESERVE)
    assert lad.allowance(eng) + share == pytest.approx(lad.base_allowance(eng))   # (f1.0's split of the allowance)
    oa = run(st, eng, st._patch_proposal(eng, 'c0', {'mechanism': 'patch a', 'change': {}, 'patch': dict(PATCH_A), 'mu': 0.03}))
    assert st.runs[oa.cid]['kind'] == 'oom'                                  # 1: patch a's self-proof (out of memory)
    qd = st.derived_q(eng, 'c0', {**START, 'liger': True}, oa.cid, kind='complete', p=0.95)
    assert tuple(qd.fidelity) == ('probe', 'full') and st.ledger[qd.hypothesis]['probe_pool'] == 'code'
    od = run(st, eng, qd)                                                    # 2: its completion re-proves it (a repair)
    assert od.status == 'screened'
    qb = st._patch_proposal(eng, eng.incumbent(), {'mechanism': 'patch b', 'change': {}, 'patch': dict(PATCH_B), 'mu': 0.03})
    ob = run(st, eng, qb)                                                    # 3: patch b's self-proof
    assert ob.status == 'screened' and eng.incumbent() == ob.cid
    pb = qb.hypothesis.split('|')[1]
    fix.update(mechanism=f'[fix {pb}] patch b at the hook the job calls', change={}, mu=0.03,
               patch={PATCH_FILE: 'def one():\n    pass\n\n\ndef two():\n    _m_b()\n    _m_c()\n'})
    st.protected_screens, st.fb_runs = frozenset(range(1, 11)), st.fb_runs + 1
    q4 = st._code(eng, eng.incumbent(), 5)                                   # 4: the repair patch, asked for
    assert len(q4) == 1 and st.fb_fixes[q4[0].hypothesis.split('|')[1]] == pb and st._lowfi_ok(eng, q4[0])
    run(st, eng, q4[0])
    assert [x['kind'] for x in lad.code_log] == ['patch', 'derived', 'patch', 'patch'] and lad.code_probes_left() == 0
    assert lad.spent_pool() == pytest.approx(4 * 0.25) and lad.spent(eng, code=True) == lad.spent(eng) == 0.0
    acc = lad.describe(eng)['account']                                       # reported apart, the allowance untouched
    assert acc['code_pool_probes'] == 4 and acc['code_pool_gpu_hours'] == pytest.approx(1.0)
    assert acc['knob_lowfi_gpu_hours'] == acc['code_share_gpu_hours'] == 0.0 and acc['lowfi_gpu_hours'] == pytest.approx(1.0)
    assert acc['allowance_gpu_hours'] == pytest.approx(lad.base_allowance(eng), abs=1e-4) and lad.affordable(eng, 'probe')
    inc = eng.incumbent()
    five = {f: t + '\n\ndef three():\n    _m_e()\n' for f, t in eng.s.candidates[inc].overlay.items()}
    q5 = st._patch_proposal(eng, inc, {'mechanism': 'patch e', 'change': {}, 'patch': five, 'mu': 0.03})
    assert st._fb_code_probe_ok(eng) and st._lowfi_ok(eng, q5)               # the fifth: the allowance's code share
    run(st, eng, q5)
    assert len(lad.code_log) == 4 and [x['id'] for x in lad.code_share_log] == [q5.hypothesis]
    assert lad.spent(eng, code=True) == pytest.approx(0.25) and lad.spent(eng) == 0.0 and lad.spent_pool() == pytest.approx(1.0)
    assert st.finalist_view()['probe_accounting'] == {'code_pool': True, 'code_probes_used': 4, 'code_probes_max': 4,
                                                      'code_share_probes': 1, 'probe_below': 0.5,
                                                      'probe_ratio': pytest.approx(0.25 / 0.3, abs=1e-3)}


def test_f11_the_knob_side_never_draws_on_the_code_probe_pool():
    """Base f1.1: knob probes are funded by the low-fidelity allowance alone; spent, a doubtful knob candidate runs in
    full or not at all (hard rule 5 refuses a failed combination) while the pool keeps its four probes for patches, whose
    probes do not touch the allowance."""
    st, eng = fin(answer(), effect=s6_effect, answers=lambda ctx, n: [ph(f'w{n}')], cap=40.0, ref_peak=30.0, off=['fb_s6'])
    eng.probe_cost = 0.3                                                     # (a probe as dear as a full run)
    st._init(eng)
    lad = st.ladder
    for wk in (0, 4, 2):                                                     # micro-batch 2 runs out of memory at its probes
        for extra in ({}, {'tf32': True}, {'liger': True}):
            plan = lad.plan(eng, 0.3, short=False)
            if plan[0] != 'probe':
                break
            run(st, eng, st._knob_q(eng, 'c0', {**START, 'mb': 2, 'wk': wk, **extra}, kind='t', plan=plan, p=0.3))
    assert not lad.affordable(eng, 'probe') and lad.spent(eng) >= lad.allowance(eng) - 0.3
    assert lad.code_log == [] and lad.code_probes_left() == 4 and st._fb_code_probe_ok(eng)   # the pool untouched
    assert lad.plan(eng, 0.3, short=False) == ('full',) and not lad.transfers   # a doubtful candidate: no probe from it
    assert st._knob_q(eng, 'c0', {**START, 'mb': 2, 'wk': 0, 'rc': 'half'}, kind='t', plan=('full',), p=0.3) is None
    assert st.fb_counts['combo_refused'] >= 1                                # (hard rule 5: no probe left, not run)
    knob = lad.spent(eng)
    st.protected_screens, st.fb_runs = frozenset({1}), st.fb_runs + 1
    q = st._code(eng, 'c0', 5)                                               # a patch: its proof from the pool
    assert len(q) == 1 and st._lowfi_ok(eng, q[0])
    run(st, eng, q[0])
    assert len(lad.code_log) == 1 and lad.spent(eng) == pytest.approx(knob) and lad.spent_pool() == pytest.approx(0.3)
    assert lad.spent(eng, code=True) == 0.0 and lad.code_share_log == []    # (the pool's: neither the share nor the knob side)


def _prove(st, eng):
    """One code-track step on a protected screen: the ask's patch, run when its proof can be paid (else None)."""
    st.queue, st.fb_runs = [], st.fb_runs + 1
    q = st._code(eng, eng.incumbent(), 5)
    return run(st, eng, q[0]) if q and st._lowfi_ok(eng, q[0]) else None


def test_f11_beyond_the_pool_a_proof_falls_back_to_the_code_share_then_its_borrow_then_none():
    """F1 (b): the pool's four proofs first; the fifth from the allowance's code share, the sixth borrowed from the knob
    side's unspent allowance (f1.0's rules); with no room left, no seventh proof and no ask; the GPU-hours in three
    separate lines (the pool, the code share, the knob side) that add up to every probe's."""
    st, eng = fin(answer(), effect=s6_effect, answers=lambda ctx, n: [ph(f'u{n}')], cap=40.0, ref_peak=30.0, off=['fb_s6'])
    eng.probe_cost = 0.3                                                     # (the allowance 1.5: the share 0.3, the knob
    st._init(eng)                                                            #  side 1.2)
    lad = st.ladder
    st.protected_screens = frozenset(range(1, 11))
    assert all(_prove(st, eng) is not None for _ in range(4))
    assert len(lad.code_log) == 4 and lad.code_share_log == [] and lad.code_probes_left() == 0
    for wk in (0, 4, 2):                                                     # the knob side spends 0.9 of its 1.2
        run(st, eng, st._knob_q(eng, 'c0', {**START, 'mb': 2, 'wk': wk}, kind='t', plan=lad.plan(eng, 0.3, short=False), p=0.3))
    assert lad.spent(eng) == pytest.approx(0.9) and len(lad.code_log) == 4   # (the pool: the code track's only)
    assert _prove(st, eng) is not None and len(lad.code_share_log) == 1 and not lad.transfers      # 5: the code share
    assert _prove(st, eng) is not None and len(lad.code_share_log) == 2 and lad.transfers[-1]['borrowed']   # 6: borrowed
    assert not st._fb_code_probe_ok(eng) and _prove(st, eng) is None and len(lad.code_share_log) == 2   # no room: none
    acc = lad.describe(eng)['account']
    assert (acc['code_pool_gpu_hours'], acc['code_share_gpu_hours'], acc['knob_lowfi_gpu_hours']) == \
        (pytest.approx(1.2), pytest.approx(0.6), pytest.approx(0.9))         # three lines ...
    assert acc['lowfi_gpu_hours'] == pytest.approx(2.7) and (acc['code_pool_probes'], acc['code_share_probes']) == (4, 2)
    assert acc['knob_allowance_gpu_hours'] + acc['code_share_allowance_gpu_hours'] == \
        pytest.approx(acc['allowance_gpu_hours'], abs=1e-3)                  # ... the allowance split as f1.0 split it


def test_f11_with_the_pool_spent_and_no_room_in_the_allowance_there_is_no_fifth_proof():
    """F1 (b): a probe dearer than the whole low-fidelity allowance (a starved two-node job): f1.0's code share and borrow
    pay at most the first proof (priced at the engine's estimate before any probe ran); f1.1's pool pays four, and with
    no room in the allowance there is no fifth and no ask."""
    st, eng = fin(None, answers=lambda ctx, n: [ph(f'v{n}')])
    eng.probe_cost = 2.0                                                     # (the allowance: 0.5 x 10 x 0.3 = 1.5)
    st._init(eng)
    lad = st.ladder
    st.protected_screens = frozenset(range(1, 11))
    assert all(_prove(st, eng) is not None for _ in range(4)) and len(lad.code_log) == 4
    assert not st._fb_code_probe_ok(eng) and not lad.affordable(eng, 'probe', code=True)
    assert _prove(st, eng) is None and lad.code_share_log == [] and not lad.transfers and lad.spent(eng, code=True) == 0.0
    assert lad.spent_pool() == pytest.approx(8.0) and lad.spent(eng) == 0.0  # (outside the allowance, apart)
    st0, eng0 = fin(None, answers=lambda ctx, n: [ph(f'v{n}')])
    eng0.probe_cost = 2.0
    st0.ladder.code_probes_max = 0                                           # f1.0's accounting
    st0._init(eng0)
    st0.protected_screens = frozenset(range(1, 11))
    assert sum(_prove(st0, eng0) is not None for _ in range(5)) <= 1 < len(lad.code_log)


def test_f11_the_ratio_rule_runs_p_from_05_in_full_when_a_probe_costs_over_06_of_a_full_run():
    """Base f1.1: the probe / full ratio from the first valid probe against the first valid full run (before any probe,
    the reference run's start share); above 0.6 the knob side probes only below P 0.5 - its threshold reads 0.5 in the
    plans, the draw's floor and a strategy's own checks; at 0.6 or below, the old rule (0.7)."""
    st, eng = fin(None)
    eng.probe_cost = 0.24                                                    # 0.8 of the 0.3 GPU-hour full run
    st._init(eng)
    lad = st.ladder
    assert lad.probe_ratio()[0] is None and lad.probe_below == 0.7           # no valid probe, no timed reference run
    assert lad.plan(eng, 0.6, short=False) == ('probe', 'full')
    o = run(st, eng, st._knob_q(eng, 'c0', {**START, 'tf32': True}, kind='t', plan=('probe', 'full'), p=0.6))
    assert o.status == 'screened'
    r, src = lad.probe_ratio()
    assert r == pytest.approx(0.8) and src.startswith('the first valid probe') and lad.probe_below == RATIO_PROBE_BELOW
    assert lad.plan(eng, 0.6, short=False) == ('full',) and lad.plan(eng, 0.45, short=False) == ('probe', 'full')
    q = st._knob_q(eng, 'c0', {**START, 'wk': 4}, kind='t', plan=lad.plan(eng, 0.55, short=False), p=0.55)
    assert tuple(q.fidelity) == ('full',)                                    # P >= 0.5: in full
    d = lad.describe(eng)
    assert d['probe_ratio']['applies'] and d['probe_below'] == 0.5 and d['probe_below_base'] == 0.7
    st2, eng2 = fin(None)
    eng2.probe_cost = 0.18                                                   # 0.6 of a full run: the old rule
    st2._init(eng2)
    run(st2, eng2, st2._knob_q(eng2, 'c0', {**START, 'tf32': True}, kind='t', plan=('probe', 'full'), p=0.6))
    assert st2.ladder.probe_ratio()[0] == pytest.approx(0.6) and st2.ladder.probe_below == 0.7
    assert st2.ladder.plan(eng2, 0.6, short=False) == ('probe', 'full')
    st3, eng3 = fin(None)                                                    # before any probe: the reference's start -
    eng3.s.candidates['c0'].receipt['evidence'] = {'wall_s': 200.0, 'step_time': 2.0,   # 142 s to its first update, two
                                                   'per_step': [{'t': 2.0 * i} for i in range(1, 31)]}   # updates of 2 s
    st3._init(eng3)
    r3, src3 = st3.ladder.probe_ratio()
    assert r3 == pytest.approx(144.0 / 200.0) and src3.startswith("the reference run's start") and st3.ladder.probe_below == 0.5
    st4 = Dummy(CodeProposer(None))                                          # (a strategy's own lower threshold stays)
    st4.ladder.probe_below = 0.4
    st4._init(eng3)
    assert st4.ladder.probe_below == 0.4


def test_f11_s6_still_gates_under_the_ratio_rule():
    """Base f1.1 changes only the probe accounting: after two knob-side runs out of memory a candidate below P 0.9 is
    still refused (the screen goes to the code track), one at 0.9 or more runs - in full, the ratio rule applying."""
    from infrarsi.search.finalist_base import S6_P
    st, eng = fin(answer(), effect=s6_effect, answers=lambda ctx, n: [ph(f'z{n}')], cap=40.0, ref_peak=30.0)
    eng.probe_cost = 0.24
    st._init(eng)
    run(st, eng, st._knob_q(eng, 'c0', {**START, 'tf32': True}, kind='t', plan=('probe', 'full'), p=0.95))
    assert st.ladder.probe_below == RATIO_PROBE_BELOW                        # (the ratio, 0.8: the rule applies)
    for x in ({'wk': 0}, {'wk': 4}):
        run(st, eng, st._knob_q(eng, 'c0', {**START, 'mb': 2, **x}, kind='t', plan=('full',), p=0.9))
    assert st.s6_active()
    low, high = {**START, 'rc': 'half'}, {**START, 'tf32': True, 'wk': 0}
    assert st._pf(low)[0] < S6_P <= st._pf(high)[0]
    assert st._knob_q(eng, 'c0', low, kind='t', plan=st.ladder.plan(eng, st._pf(low)[0], short=False)) is None
    assert st.fb_counts['s6_refused'] >= 1 and st._fb_ask_due(eng, 'c0', 5).endswith('(S6)')
    q = st._knob_q(eng, 'c0', high, kind='t', plan=st.ladder.plan(eng, st._pf(high)[0], short=False))
    assert q is not None and tuple(q.fidelity) == ('full',)


def test_the_allowance_is_the_screens_full_runs_share_and_a_raised_gpu_cap_buys_no_probe():
    """D5: the probe / short-run allowance is LOWFI_SHARE x the screens x the mean full run; RungLadder's (ladder.py
    allowance_of) takes the smaller of that and gpu_hours / 1.5, so raising a binding cap grows it - the finalists' does
    not move."""
    from infrarsi.search.ladder import allowance_of
    st, eng = fin(None)
    st._init(eng)
    full = eng.costs()['full']
    eng.s.budget.gpu_hours = 1.0                                             # a cap below the ten screens' 3 GPU-hours
    rung, mine = allowance_of(eng, LOWFI_SHARE), st.ladder.base_allowance(eng)
    assert rung == pytest.approx(0.5 * 1.0 / 1.5) and mine == pytest.approx(LOWFI_SHARE * 10 * full)
    eng.s.budget.gpu_hours = 192.0                                           # D5's raise
    assert allowance_of(eng, LOWFI_SHARE) > rung and st.ladder.base_allowance(eng) == pytest.approx(mine)
    assert st.ladder.allowance(eng) + st.ladder.allowance(eng, code=True) == pytest.approx(mine)
    assert st.ladder.describe(eng)['account']['allowance_gpu_hours'] == pytest.approx(round(mine, 4))


# ---------------------------------------------------------------- S6: no out-of-memory streaks
def s6_effect(base, cfg, overlay):
    """Micro-batch 2 always runs out of memory, and so does a patch `big`; the fused loss with tf32 crashes; the rest fit
    (+1 % per value off the reference)."""
    if cfg['mb'] == 2 or '_m_big(' in ''.join((overlay or {}).values()):
        return None, 'oom'
    if cfg['liger'] and cfg['tf32']:
        return None, 'crash'
    return 0.01 * sum(1 for k, v in cfg.items() if v != START[k]), None


def test_s6_the_counter_counts_knob_side_ooms_at_probes_and_full_runs_and_resets_on_any_other_result():
    st, eng = fin(answer(), effect=s6_effect, cap=40.0, ref_peak=30.0)
    st._init(eng)
    o = run(st, eng, st._knob_q(eng, 'c0', {**START, 'mb': 2, 'wk': 0}, kind='t', plan=('probe', 'full'), p=0.9))
    assert o.fidelity == 'probe' and st.runs[o.cid]['kind'] == 'oom' and st.fb_oom_streak == 1   # a probe counts
    run(st, eng, st._patch_proposal(eng, 'c0', ph('z')))                     # a patch run: not the knob side's
    assert st.fb_oom_streak == 1
    o = run(st, eng, st._knob_q(eng, 'c0', {**START, 'mb': 2, 'wk': 4}, kind='t', plan=('full',), p=0.9))
    assert o.fidelity == 'full' and st.fb_oom_streak == 2 and st.s6_active()   # a full run counts
    q = st._knob_q(eng, 'c0', {**START, 'tf32': True}, kind='t', plan=('full',))
    st.observe(eng, q, Outcome(None, 'duplicate', False, None, None, 'none', 0.0, ['already tried'], []))
    assert st.fb_oom_streak == 2                                             # (no run: no change)
    o = run(st, eng, st.uniform_start_q(eng, 'c0', {**START, 'liger': True, 'tf32': True}))
    assert st.runs[o.cid]['kind'] == 'crash' and st.fb_oom_streak == 0       # any other result resets it
    assert st.fb_oom_streak_max == 2 and st.finalist_view()['s6']['max_streak'] == 2


def test_s6_after_two_ooms_a_candidate_below_09_is_refused_and_the_screen_goes_to_code(monkeypatch):
    import infrarsi.search.llmopt as llmopt
    from infrarsi.search.finalist_base import S6_P
    st, eng = fin(answer(), effect=s6_effect, answers=lambda ctx, n: [ph(f'g{n}')], cap=40.0, ref_peak=30.0)
    st._init(eng)
    cids = [run(st, eng, st._knob_q(eng, 'c0', {**START, 'mb': 2, **x}, kind='t', plan=('full',), p=0.9)).cid
            for x in ({'wk': 0}, {'wk': 4})]
    low, high = {**START, 'rc': 'half'}, {**START, 'tf32': True}            # recompute half declared +20 GiB: low P
    assert st.s6_active() and st._pf(low)[0] < S6_P <= st._pf(high)[0]
    assert st._knob_q(eng, 'c0', low, kind='t', plan=('full',)) is None and st.fb_counts['s6_refused'] == 1
    assert st.derived_q(eng, 'c0', low, cids[-1], kind='complete') is None and st.fb_counts['s6_refused'] == 2
    item = {'cfg': low, 'key': st._ckey(low, ()), 'round': 0, 'rank': 0, 'gain': None, 'why': 'x', 'p': 0.5,
            'why_p': '', 'status': 'open', 'source': 'test'}
    assert st._run_item(eng, 'c0', item, kind='round') is None and item['status'] == 'open'   # (kept for later)
    monkeypatch.setattr(Dummy, '_knob_strong', lambda self, engine, inc: {'change': {'rc': 'half'}, 'p_feasible': 0.9,
                                                                          'predicted_gain_pct': 9.0})
    q = st._code(eng, 'c0', 5)                                               # the screen goes to code, whatever the knob
    assert len(q) == 1 and '(S6)' in st.fb_asks[-1]['why'] and st.fb_counts['s6_code_asks'] == 1   # side holds
    seen = {}

    def draw(s, engine, inc, **kw):
        seen['p_min'] = kw.get('p_min')
        return high, (), s._pf(high)[0], 'drawn', {}
    monkeypatch.setattr(llmopt, 'weighted_draw', draw)
    assert st._fallback(eng, 'c0') is not None and seen['p_min'] == S6_P    # the draw: only P(fits) >= 0.9
    qu = st.uniform_start_q(eng, 'c0', {**START, 'rc': 'half', 'wk': 0})    # the uniform start: in full regardless
    assert qu is not None and tuple(qu.fidelity) == ('full',) and qu.hypothesis.startswith('uniform|')
    assert st._knob_q(eng, 'c0', high, kind='t', plan=('full',)) is not None
    o = run(st, eng, st._knob_q(eng, 'c0', {**START, 'tf32': True, 'wk': 4}, kind='t', plan=('full',)))
    assert o.status == 'screened' and not st.s6_active()                    # a fit ends the streak: the gate lifts
    assert st._knob_q(eng, 'c0', low, kind='t', plan=('full',)) is not None
    st.fb_oom_streak, st.fb_on['fb_s6'] = 2, False                           # (the part off: no gate)
    assert st._knob_q(eng, 'c0', {**low, 'wk': 0}, kind='t', plan=('full',)) is not None


def test_s6_a_patch_that_ran_out_of_memory_is_never_retried():
    calls = {}

    def answers(ctx, n):
        calls[n] = True
        return [ph('big3', mech='A big cache 3'), ph('other', mech='a fused optimizer')]
    st, eng = fin(None, effect=s6_effect, answers=answers)
    st._init(eng)
    o = run(st, eng, st._patch_proposal(eng, 'c0', ph('big', mech='a big cache 1')))
    assert o.status == 'infeasible' and o.fidelity == 'probe' and st.runs[o.cid]['kind'] == 'oom'
    assert [b['key'] for b in st.fb_s6_bans] == ['a big cache'] and st.fb_oom_streak == 0   # (the knob counter: no)
    q2 = st._patch_proposal(eng, 'c0', ph('big2', mech='[fix p1] a big cache 2'))
    st.queue.append(q2)
    assert st._code(eng, 'c0', 5, ask=False) == [] and st.queue == []       # a queued retry: dropped
    assert st.ledger[q2.hypothesis]['status'] == 'dropped' and '(S6)' in st.ledger[q2.hypothesis]['why']
    q = st._code(eng, 'c0', 5)                                               # an ask: its retry not queued, another is
    assert [st.patches[x.hypothesis.split('|')[1]]['mechanism'] for x in q] == ['a fused optimizer']
    assert st.fb_counts['s6_patches_dropped'] == 2 and st._fb_s6_banned('x', ph('big')['patch']) is not None   # (files)


def test_s6_over_a_whole_run_no_knob_candidate_below_09_follows_two_ooms(monkeypatch):
    """Ten screens where every micro-batch-2 configuration runs out of memory and the configuration calls keep offering
    them: every knob candidate proposed while two knob-side runs in a row had run out of memory had P(fits) >= 0.9 - no
    exception but the uniform start (SHARED_r1 P1-3: no last resort)."""
    from infrarsi.search.finalist_base import S6_P

    def configs(n):
        return [{'change': {'mb': 2, **x}, 'gain_pct': 15 - i} for i, x in
                enumerate(({}, {'liger': True}, {'tf32': True}, {'wk': 0}, {'rc': 'half'}))]
    st, eng = fin(answer(), configs=configs, effect=s6_effect, cap=40.0, ref_peak=30.0)
    made = []
    real = Dummy._knob_q

    def knob_q(self, engine, inc, cfg, **kw):
        active, p = self.s6_active(), self._pf(cfg, tuple(self.chain))[0]
        q = real(self, engine, inc, cfg, **kw)
        if q is not None:
            made.append((active, p, bool(kw.get('s6_exempt'))))
        return q
    monkeypatch.setattr(Dummy, '_knob_q', knob_q)
    eng.run(st)
    gated = [(p, exempt) for active, p, exempt in made if active]
    assert all(p >= S6_P for p, exempt in gated if not exempt), gated
    assert st.fb_oom_streak_max >= 2 and st.fb_counts['s6_refused'] >= 1 and 's6_last_resort' not in st.fb_counts


def test_shared_r1_p1_3_no_sub_09_fallback_after_two_ooms_the_code_track_or_nothing(monkeypatch):
    """SHARED_r1 P1-3's scenario: two knob-side runs out of memory, no configuration of the table at P >= 0.9: the draw
    proposes nothing below it; with the code budget spent the step proposes nothing at all, with a call left the code
    track's patch takes the screen."""
    st, eng = fin(answer(), effect=s6_effect, answers=lambda ctx, n: [ph(f'h{n}')])
    st._init(eng)
    for x in ({'wk': 0}, {'wk': 4}):
        run(st, eng, st._knob_q(eng, 'c0', {**START, 'mb': 2, **x}, kind='t', plan=('full',), p=0.9))
    assert st.s6_active()
    monkeypatch.setattr(Dummy, '_pf', lambda self, cfg, ids=(): (0.366, 'planted'))   # (none reaches 0.9)
    calls = st.proposer.calls
    st.proposer.calls = CALLS_MAX                                            # the code budget spent
    assert st._fallback(eng, 'c0') is None and st.fb_counts['s6_no_knob'] == 1
    st.proposer.calls = calls
    q = st._fallback(eng, 'c0')                                              # a call left: the code track's patch
    assert q is not None and q.hypothesis.startswith('patch|') and '(S6)' in st.fb_asks[-1]['why']


def test_shared_r1_p1_1_a_derived_candidate_proves_its_unadopted_patch_before_any_full_run():
    """SHARED_r1 P1-1's scenario: a patch's first probe runs out of memory; the completion carrying it gets a probe (even
    at a high P) whose gate runs the patch's self-proof after the peak and speed gates: a probe row with no proof line is
    `patch_inert` (as the patch's own gate reads it), one with its found -> set and active lines passes; without an
    affordable probe the completion is not proposed."""
    s = SimSession(START, knobs=KNOBS, screens=10, base_files={PATCH_FILE: TWO})
    eng = SimEngine(s, patch_oom, sigma_runs=[0.01 / math.sqrt(2)])
    st = Dummy(CodeProposer(answer()), facts={'cap_gib': 100.0}, seed=3)
    st._init(eng)
    oa = run(st, eng, st._patch_proposal(eng, 'c0', {'mechanism': 'patch a', 'change': {}, 'patch': dict(PATCH_A), 'mu': 0.03}))
    assert st.runs[oa.cid]['kind'] == 'oom' and st.chain == []
    q = st.derived_q(eng, 'c0', {**START, 'liger': True}, oa.cid, kind='complete', p=0.95, plan=('full',))
    assert q is not None and tuple(q.fidelity) == ('probe', 'full') and st.ledger[q.hypothesis]['proof_owed'] == ['p1']
    silent = {'patch_lines': [], 'intervals': []}
    why = q.gate(eng, 'cz', 'c0', silent)
    assert isinstance(why, tuple) and why[1] == 'patch_inert' and why[0].startswith('it printed no')
    proved = {'patch_lines': ['[infrarsi-patch] one: found pass -> set _m_a', '[infrarsi-patch] one: active'],
              'intervals': []}
    assert q.gate(eng, 'cz2', 'c0', proved) is None
    for _ in range(st.ladder.code_probes_left()):                            # the code-probe pool spent (f1.1): beyond it
        st.ladder.note_code_probe('cx', 'patch|px', 0.1, 'patch')            # the knob side's probe, as in f1.0
    q2 = st.derived_q(eng, 'c0', {**START, 'liger': True, 'wk': 2, 'tf32': True}, oa.cid, kind='complete', p=0.95)
    assert tuple(q2.fidelity)[0] == 'probe' and 'probe_pool' not in st.ledger[q2.hypothesis]
    st.ladder._fund = lambda engine, cost, code=False: False                 # no affordable probe at all: not proposed
    assert st.derived_q(eng, 'c0', {**START, 'liger': True, 'wk': 4}, oa.cid, kind='complete', p=0.95) is None


def test_shared_r1_p1_2_a_derived_incumbent_keeps_patch_ids_aligned_with_its_files():
    """SHARED_r1 P1-2's scenario: patch p1 runs out of memory; a completion carrying it with the fused loss becomes the
    incumbent; a later patch p3 written against the incumbent's files carries p1 in its ids too, and a configuration
    without the fused loss that carries the chain (both patches' files) holds p1's measured failure: probed first."""
    def effect(base, cfg, overlay):
        def g(c, ov):
            text = ''.join((ov or {}).values())
            return 0.05 * bool(c['liger']) + (0.05 if '_m_b(' in text else 0.0) + 0.01 * bool(c['tf32'])
        if '_m_a(' in ''.join((overlay or {}).values()) and not cfg['liger']:
            return None, 'oom'
        return g(cfg, overlay) - g(base.config, base.overlay), None
    s = SimSession(START, knobs=KNOBS, screens=10, base_files={PATCH_FILE: TWO})
    eng = SimEngine(s, effect, sigma_runs=[0.01 / math.sqrt(2)])
    st = Dummy(CodeProposer(answer()), facts={'cap_gib': 100.0}, seed=3)
    st._init(eng)
    oa = run(st, eng, st._patch_proposal(eng, 'c0', {'mechanism': 'patch a', 'change': {}, 'patch': dict(PATCH_A), 'mu': 0.03}))
    comp = st.derived_q(eng, 'c0', {**START, 'liger': True}, oa.cid, kind='complete', p=0.95)
    oc = run(st, eng, comp)
    assert oc.status == 'screened' and eng.incumbent() == oc.cid
    assert st.chain == ['p1'] and '_m_a()' in st.overlay_best[PATCH_FILE] and st.patches['p1']['adopted_via'] == oc.cid
    both = {PATCH_FILE: 'def one():\n    _m_a()\n\n\ndef two():\n    _m_b()\n'}   # (written against the incumbent's files)
    qb = st._patch_proposal(eng, oc.cid, {'mechanism': 'patch b', 'change': {}, 'patch': both, 'mu': 0.03})
    assert st.ideas[qb.hypothesis] == frozenset({'p1', qb.hypothesis.split('|')[1]})
    run(st, eng, qb)
    assert set(st.chain) == {'p1', qb.hypothesis.split('|')[1]}
    q = st._knob_q(eng, eng.incumbent(), {**START, 'tf32': True}, kind='t', plan=('full',), p=0.99)
    assert q is not None and tuple(q.fidelity)[0] == 'probe'                # p1's failure held: a probe before any full
    assert '_m_a()' in q.patch[PATCH_FILE] and set(st.ideas[q.hypothesis]) == set(st.chain)   # run; ids = files


def test_shared_r1_p1_4_a_retrying_configuration_call_cannot_eat_the_code_calls():
    """SHARED_r1 P1-4's scenario: a backend that uses every re-ask it is granted; a configuration call starting at 7/10
    attempts gets one attempt (its re-asks capped at the calls left less CODE_CALLS); the code track then has its two (the
    review made at an earlier ask: the patch call and its re-ask)."""
    def script(role, system, payload):
        if role == 'prior':
            return {'levels': [], 'interactions': [], 'configs': [], 'notes': ''}
        if role == 'configs':
            return {'configs': [{'change': {'fused_optimizer': True}, 'gain_pct': 2, 'why': 'x'}], 'notes': ''}
        if role == 'code_review':
            return {'verdict': 'patch', 'summary': 'x', 'findings': []}
        return {'hypotheses': [{'mechanism': 'patch q', 'change': {}, 'patch': {'patches.txt': 'q\nproof:q\n'},
                                'predicted_gain_pct': 3, 'site': 'q'}]}
    r = llm_toy_setup(script, retries=2, protected=range(1, 11))
    st, eng, be = r.st, r.eng, r.be
    st._init(eng)
    r.prop.review = {'verdict': 'patch', 'summary': 'x', 'findings': []}     # (made at an earlier ask)
    st._bdriver.attempts = 7
    st._ask(eng, 'c0')
    assert be.granted[-1] == ('configs', 0) and st._bdriver.attempts == 8 and st._calls_left() == CODE_CALLS
    st.fb_runs += 1
    assert st._code(eng, 'c0', 5) and st._bdriver.attempts == CALLS_MAX     # the patch call and its re-ask: the two
    assert be.granted[-1] == ('hypotheses', 1) and st.fb_asks[-1]['sent'] and st.fb_asks[-1]['calls'] == [8, CALLS_MAX]


def test_shared_r2_the_proof_carrier_is_the_run_that_adopted_the_chain():
    """SHARED_r2 P1: a patch adopted through a later completion keeps its first candidate (a run that carried it alone)
    and gains adopted_via; the proof baseline is the adopting run, so the rest of the chain's proof lines are not new to
    a later patch's self-proof."""
    from types import SimpleNamespace as NS
    from infrarsi.search.finalist_base import FinalistBase
    eng = NS(s=NS(candidates={'c0': 1, 'c1': 1, 'c4': 1}))
    me = NS(chain=['pA'], patches={'pA': {'candidate': 'c1', 'adopted_via': 'c4'}})
    assert FinalistBase._fb_chain_carrier(me, eng) == 'c4'
    del me.patches['pA']['adopted_via']
    assert FinalistBase._fb_chain_carrier(me, eng) == 'c1'
    me.patches['pA']['adopted_via'] = 'c9'                      # (a run the session does not know: the reference)
    assert FinalistBase._fb_chain_carrier(me, eng) == 'c0'
    assert FinalistBase._fb_chain_carrier(NS(chain=[], patches={}), eng) == 'c0'


def _blame_me(runs):
    from types import SimpleNamespace as NS
    from infrarsi.search.finalist_base import FinalistBase
    me = NS(runs=runs, _levels=lambda cfg: sorted(cfg.items()), fb_obs=1, _fb_iso_stamp=None, _fb_iso=None, _fb_iso_ctx={})
    me.isolated_blame = lambda: FinalistBase.isolated_blame(me)
    return me, (lambda held: FinalistBase._fb_blamed(me, sorted(held.items())))


def test_isolated_blame_waives_a_value_in_a_context_where_it_fitted():
    """Finalist 1's base finding (SHARED_r4 / r5): a failed run whose values each fitted alone elsewhere is a combination
    failure: both values are blamed, but a candidate holding a value's whole fitted context (here: nothing else) is not
    priced down for it; the combination itself keeps hard rule 5's probe."""
    me, blamed = _blame_me({'c1': {'kind': 'ok', 'cfg': {'a': '1'}}, 'c2': {'kind': 'ok', 'cfg': {'b': '1'}},
                            'c3': {'kind': 'oom', 'cfg': {'a': '1', 'b': '1'}}})
    assert set(me.isolated_blame()) == {('a', '1'), ('b', '1')}
    assert blamed({'a': '1'}) == [] and blamed({'a': '1', 'z': '9'}) == [] and blamed({'b': '1'}) == []


def test_a_failed_run_inside_a_fitted_run_blames_nothing():
    """Finalist 2's base finding: a fitted run held all of a failed run's values plus savers; the failure is the savers'
    absence, so no member is blamed alone (it starved the arm: every untried row at P <= 0.25)."""
    me, blamed = _blame_me({'c1': {'kind': 'oom', 'cfg': {'a': '1', 'b': '1', 'c': '1'}},
                            'c5': {'kind': 'ok', 'cfg': {'a': '1', 'b': '1', 'c': '1', 'e': '1', 'f': '1'}}})
    assert me.isolated_blame() == {} and blamed({'a': '1', 'b': '1'}) == []


def test_a_rescued_fit_keeps_the_blame_outside_its_rescue():
    """SHARED_r4 P2: recompute=none failed beside a neutral tf32 (which fitted alone) and then fitted only with compile (a
    saver); a candidate with recompute=none but without compile is still priced down (its probe runs first), one with
    compile is not; tf32 is never priced down (it fitted alone)."""
    me, blamed = _blame_me({'c1': {'kind': 'ok', 'cfg': {'tf32': 'true'}},
                            'c2': {'kind': 'oom', 'cfg': {'recompute': 'none', 'tf32': 'true'}},
                            'c3': {'kind': 'ok', 'cfg': {'recompute': 'none', 'compile': 'true'}}})
    assert blamed({'recompute': 'none', 'liger': 'true'}) == [('recompute', 'none')]
    assert blamed({'recompute': 'none', 'compile': 'true', 'liger': 'true'}) == []
    assert blamed({'tf32': 'true', 'liger': 'true'}) == []


def test_the_base_value_key_is_wcs_key_for_every_value():
    from infrarsi.search import finalist_base, wcs
    vals = (None, 0, 1, 1.0, True, False, 0.0, -0.0, 'x', '1', [1, 2], {'b': 1, 'a': [2]}, (1,), (1.0,), (True,), (1, 'y'),
            ((1,), 2), float('inf'))
    for _ in range(2):                                  # (the second pass reads the memo)
        for v in vals:
            assert finalist_base.vk(v) == wcs.vk(v), v


def test_the_base_levels_are_wcs_levels():
    """The base's _levels (memoised keys) gives exactly wcs's list."""
    from types import SimpleNamespace as NS
    from infrarsi.search.finalist_base import FinalistBase
    from infrarsi.search.wcs import WCS
    me = NS(knobs={'a': [1, 2], 'b': [None, 'x'], 'c': [[1], [2]], 'd': [(1,), (True,)]},
            ref={'a': 1, 'b': None, 'c': [1], 'd': (1,)})
    for cfg in ({'a': 1, 'b': None, 'c': [1]}, {'a': 2, 'b': 'x', 'c': [2], 'patch:p1': True, 'z': 5},
                {'a': 2.0, 'b': None}, {'d': (True,), 'patch:p1': True}, {'d': (1,)}, {}):
        assert FinalistBase._levels(me, cfg) == WCS._levels(me, cfg)


def test_six_identical_runner_failures_stop_a_finalist_session(tmp_path):
    """D5 (Fable on SHARED_r1 P2): run_engine's session stops at SAME_FAILURE_STOP = 6 full runs in a row failing with one
    signature (the same rule as random) - a crash at start, six times."""
    from infrarsi.agent.backends.fake import ScriptedBackend
    from infrarsi.search import Context
    from infrarsi.search import toybench as tb
    from infrarsi.search.finalist_base import SAME_FAILURE_STOP, run_engine
    assert SAME_FAILURE_STOP == 6
    task = next(t for t in mem_tasks() if t.antagonism)                      # compile with the fused loss crashes
    sim = tb.make_sim(task)
    runner = tb.ToyRunner(sim, tb.VirtualClock())
    ref_cfg = sim.reference()
    probe_task = tb.EnvTask.from_dict(tb.toy_task_spec(sim, {'screens': 1, 'llm_tokens': 0, 'wall_seconds': 1e12,
                                                             'gpu_hours': 1e9}))
    ref = runner.run(probe_task, ref_cfg, None, 'reference', 'toy', candidate_id='c0', steps=tb.FULL)
    env = tb.EnvTask.from_dict(tb.toy_task_spec(sim, {'screens': 10, 'llm_tokens': 10 ** 7, 'wall_seconds': 1e12,
                                                      'gpu_hours': 1e9}))
    made = []

    class Crashing(Dummy):
        def propose(self, engine, k):
            if self.prior is None:
                self._init(engine)
            for cfg in self._table(engine):
                if cfg['compile'] and cfg['fused_loss'] and self._ckey(cfg, ()) not in self.tried:
                    q = self._knob_q(engine, engine.incumbent(), cfg, kind='t', plan=('full',), p=0.99)
                    if q is not None:
                        made.append(q)
                        return [q]
            return []

    def script(role, system, prompt):
        return {'levels': [], 'interactions': [], 'configs': [], 'notes': ''} if role == 'prior' else {}
    ctx = Context(task=env, budget=env.search_budget(), start_config=ref_cfg, runner=runner,
                  judge=tb.Verifier(0.03, memory_cap_gib=sim.hw['mem']), run_id='toy-s', out_dir=tmp_path,
                  hooks=tb.MemoryHooks(env, sim.memory_facts, base_files={'patches.txt': ''}), start_receipt=ref,
                  reference_receipt=ref, driver=ScriptedBackend(script, tokens_per_call=2000), seed=1,
                  options={'full_steps': tb.FULL, 'short_steps': tb.SHORT, 'probe_steps': tb.PROBE, 'job_gpu_hours': None})
    trace = run_engine(ctx, Crashing, DUMMY_COMPONENTS)
    s = ctx.sessions[-1]
    assert trace.stop_reason == 'same_failure_x6' and s.ledger.used['screens'] == 6 == len(made)
    assert all(x.get('fidelity') == 'full' and not x['valid'] for x in s.screens)


# ---------------------------------------------------------------- an engine-level run and its accounting
def _ten_screen_script(patches_until=None):
    state = {'n': 0}

    def script(role, system, payload):
        if role == 'prior':
            return {'levels': [{'knob': 'fused_loss', 'value': True, 'gain_pct': 2, 'sd_pct': 2,
                                'memory': {'direction': 'frees', 'gib': 3}}],
                    'interactions': [], 'configs': [{'name': 'a', 'change': {'fused_loss': True}, 'gain_pct': 2}],
                    'notes': ''}
        if role == 'code_review':
            return {'verdict': 'patch', 'summary': 'x', 'findings': []}
        if role == 'configs':
            return {'configs': [{'change': {'loader_workers': 4, 'fused_optimizer': True}, 'gain_pct': 3, 'why': 'x'},
                                {'change': {'expandable_segments': True}, 'gain_pct': 1, 'why': 'y'}], 'notes': ''}
        state['n'] += 1
        n = state['n']
        if patches_until is not None and n > patches_until:
            return {'hypotheses': []}
        return {'hypotheses': [{'mechanism': f'patch x{n}a', 'change': {}, 'patch': {'patches.txt': f'x{n}a\nproof:x{n}a\n'},
                                'predicted_gain_pct': 3, 'site': f'x{n}a'},
                               {'mechanism': f'patch x{n}b', 'change': {}, 'patch': {'patches.txt': f'x{n}b\n'},
                                'predicted_gain_pct': 3, 'site': f'x{n}b'}]}
    return script


def test_ten_screens_on_the_memory_toy_and_their_accounting():
    """The dummy with an LLMProposer over a priced scripted backend (0.5 USD a call) on the memory toy, every screen
    protected: ten full screens, probes (not screens), no short run, every backend attempt within the ten calls and every
    dollar within the eight, more asks than wcs's three while the budget lasts, at most three patches each, the fields in
    the patch calls, each inert patch's fix asked once, the low-fidelity account within its allowance."""
    r = llm_toy(_ten_screen_script(), usd_per_call=0.5, protected=range(1, 11))
    st, s, be, seen = r.st, r.s, r.be, r.seen
    d = st.describe()['fbdummy']
    fid = [x.get('fidelity') for x in s.screens]
    assert r.trace.stop_reason.startswith('budget') and s.ledger.used['screens'] == 10 == fid.count('full')
    assert fid.count('short') == 0 and fid.count('probe') >= 2 and s.ledger.used.get('trials', 0) >= fid.count('probe')
    assert st._llm_used() == be.attempts_made == d['llm_calls'] <= CALLS_MAX
    assert s.ledger.used['llm_usd'] <= USD_MAX + 1e-9
    assert all(k.get('max_usd') is not None and k['max_usd'] <= USD_MAX + 1e-9 for k in be.kw)
    roles = [x[0] for x in seen]
    assert roles[0] == 'prior' and roles.count('code_review') == 1
    assert st.asks > WCS_CODE_ASKS and d['finalist']['asks_made'] == st.asks == d['code_asks']
    assert all(len(a['queued']) <= PATCHES_PER_ASK for a in st.fb_asks)
    hyp = [p for role, p, *_ in seen if role == 'hypotheses']
    assert len(hyp) >= 2 and all('measured_history' in p for p in hyp[1:]) and any('fix_inert' in p for p in hyp)
    fixes = [p['fix_inert']['id'] for p in hyp if 'fix_inert' in p]
    assert len(fixes) == len(set(fixes))                                     # each inert patch's fix asked once
    acc = d['ladder']['account']                                             # reported; funded by the ladder's estimates,
    probe = max(x.get('gpu_hours') or 0.0 for x in s.screens if x.get('fidelity') == 'probe')   # so within one probe of the
    inside = acc['knob_lowfi_gpu_hours'] + acc['code_share_gpu_hours']      # end's allowance (its mean full run moved);
    assert inside <= acc['allowance_gpu_hours'] + probe                       # f1.1: the code-probe pool apart
    code = [x for x in s.screens if x.get('fidelity') == 'probe' and x.get('candidate_id') in st.ladder.code_cids]
    assert acc['code_pool_probes'] == len(st.ladder.code_log) == CODE_PROBES_MAX   # (the pool first, then the share)
    assert acc['code_share_probes'] == len(st.ladder.code_share_log) == len(code) - CODE_PROBES_MAX
    assert acc['code_pool_gpu_hours'] + acc['code_share_gpu_hours'] == pytest.approx(sum(x['gpu_hours'] for x in code), abs=1e-3)
    assert acc['lowfi_gpu_hours'] == pytest.approx(inside + acc['code_pool_gpu_hours'], abs=1e-3)   # three lines
    assert acc['by_fidelity']['full']['runs'] == 10 and 0 < acc['lowfi_share_of_gpu_hours'] < 1


def _hungry_script():
    """A model whose prior and configuration calls favour the memory-hungry values (most of them out of memory at their
    probes) and whose patch calls bring three patches each."""
    state = {'n': 0}

    def script(role, system, payload):
        if role == 'prior':
            return {'levels': [
                {'knob': 'micro_batch', 'value': 4, 'gain_pct': 20, 'sd_pct': 5, 'memory': {'direction': 'spends', 'gib': 30}},
                {'knob': 'recompute', 'value': 'none', 'gain_pct': 15, 'sd_pct': 5, 'memory': {'direction': 'spends', 'gib': 25}},
                {'knob': 'keep_params', 'value': True, 'gain_pct': 4, 'sd_pct': 2, 'memory': {'direction': 'spends', 'gib': 10}},
                {'knob': 'micro_batch', 'value': 2, 'gain_pct': 10, 'sd_pct': 4, 'memory': {'direction': 'spends', 'gib': 12}}],
                'interactions': [], 'configs': [], 'notes': ''}
        if role == 'configs':
            return {'configs': [{'change': {'micro_batch': 4, 'recompute': 'none'}, 'gain_pct': 30, 'why': 'x'},
                                {'change': {'micro_batch': 4, 'keep_params': True}, 'gain_pct': 25, 'why': 'y'},
                                {'change': {'recompute': 'none', 'keep_params': True}, 'gain_pct': 20, 'why': 'z'},
                                {'change': {'micro_batch': 2, 'recompute': 'none'}, 'gain_pct': 22, 'why': 'w'},
                                {'change': {'micro_batch': 4, 'recompute': 'partial'}, 'gain_pct': 21, 'why': 'v'}], 'notes': ''}
        if role == 'code_review':
            return {'verdict': 'patch', 'summary': 'x', 'findings': []}
        state['n'] += 1
        return {'hypotheses': [{'mechanism': f'patch y{state["n"]}{c}', 'change': {},
                                'patch': {'patches.txt': f'y{state["n"]}{c}\nproof:y{state["n"]}{c}\n'},
                                'predicted_gain_pct': 3, 'site': f'y{state["n"]}{c}'} for c in 'abc']}
    return script


def test_f11_a_whole_run_proves_at_least_as_many_patches_as_f10_and_keeps_the_pool_apart():
    """Base f1.1 over ten screens of a memory-coupled toy whose probes cost about 0.6-0.7 of a full run (loading
    dominates, as on the two-node jobs) and whose model favours memory-hungry configurations: the code track's proofs
    come from the pool first (four), then from the allowance's code share and its borrow (F1 (b)) - never fewer proofs
    than the same run under f1.0's accounting (`code_probes_max = 0`); the knob side and the share stay within the
    allowance, the pool apart, and the three lines add up to every probe's GPU-hours."""
    task = mem_tasks()[26]
    r = llm_toy(_hungry_script(), task=task, usd_per_call=0.25, protected=range(5, 11))
    st, s = r.st, r.s
    lad = st.ladder
    probe = max(x['gpu_hours'] for x in s.screens if x.get('fidelity') == 'probe')
    assert s.ledger.used['screens'] == 10 and lad.pool and len(lad.code_log) == CODE_PROBES_MAX and lad.code_share_log
    assert lad.spent(r.eng) + lad.spent(r.eng, code=True) <= lad.base_allowance(r.eng) + probe
    lowfi = sum(x['gpu_hours'] for x in s.screens if x.get('fidelity') == 'probe')
    assert lad.spent_pool() + lad.spent(r.eng, code=True) + lad.spent(r.eng) == pytest.approx(lowfi, abs=1e-6)
    old = llm_toy_setup(_hungry_script(), task=task, usd_per_call=0.25, protected=range(5, 11))
    old.st.ladder.code_probes_max = 0                                        # f1.0's accounting
    old.eng.run(old.st)
    assert not old.st.ladder.pool and old.st.ladder.code_log == [] and old.s.ledger.used['screens'] == 10
    assert len(lad.code_log) + len(lad.code_share_log) >= len(old.st.ladder.code_share_log)   # never fewer proofs


def test_the_last_two_calls_go_to_the_code_track_in_a_run():
    """Without protected screens and with patch calls that soon bring nothing, the dummy's configuration rounds compete for
    the calls (0.25 USD a call: the calls bind before the dollars): no configuration call takes one of the last two while
    two or more screens are left."""
    r = llm_toy(_ten_screen_script(patches_until=1), usd_per_call=0.25)
    configs = [(before, screens) for role, payload, screens, before in r.seen if role == 'configs']
    assert configs and all(CALLS_MAX - before > CODE_CALLS or 10 - screens < 2 for before, screens in configs), configs
    assert r.st._llm_used() <= CALLS_MAX and r.s.ledger.used['screens'] == 10


@pytest.mark.parametrize('off', [[c] for c in DUMMY_COMPONENTS] + [sorted(DUMMY_COMPONENTS)])
def test_every_part_off_runs(monkeypatch, off):
    row, st = run_kept(monkeypatch, mem_tasks()[2], lambda sp, facts, sim: Dummy(sp, facts=facts, off=off, seed=5),
                       budget=6)
    d = st.describe()['fbdummy']
    assert row['screens'] == 6 and d['finalist']['parts_off'] == sorted(x for x in off if x in FB_PARTS)
    assert d['llm_calls'] <= CALLS_MAX and row['shorts'] == 0


def test_a_failed_patch_call_degrades_and_never_crashes():
    def boom(ctx, n):
        raise RuntimeError('the backend fell over')
    st, eng = fin(None, answers=boom)
    st._init(eng)
    assert st._code(eng, 'c0', 5) == [] and st.asks == 1 and st.fb_asks[-1]['sent'] is False
    st.fb_runs += 1
    assert st._code(eng, 'c0', 5) == [] and st.asks == 2                    # tried again after a run, not in a loop
    assert st.proposer.calls == 3 and not st.fb_code_idle


# ---------------------------------------------------------------- isolation of the other strategies, names
def test_every_other_file_is_the_pristine_trees_and_the_shared_ones_only_gained_lines():
    """Against qa_trials/_pc_code_f_base: every module of the package byte for byte, but the new one and the two
    write-back / result hooks, which only inserted lines."""
    root = SRC.parents[1]
    base = root.parent / '_pc_code_f_base'
    if not base.is_dir():
        pytest.skip('the pristine tree is not next to this one')
    added = {'infrarsi/experience/runs.py', 'infrarsi/project/flow.py',
             'infrarsi/search/__init__.py', 'infrarsi/search/toybench.py',   # (the finalists' registrations: inserted)
             'infrarsi/agent/search.py',      # (v4 F4: the session's opt-in own-lines count, inserted)
             'infrarsi/search/proposers.py'}  # (v4: the patch request's cap from the spec, inserted)
    finalists = ('bcomp', 'beam', 'cards', 'explore')                    # (each finalist's own new modules)
    v41 = {'infrarsi/agent/budget.py', 'infrarsi/agent/search.py', 'infrarsi/agent/state.py',   # (v4.1 trees: the
           'infrarsi/cli/main.py', 'infrarsi/project/flow.py', 'infrarsi/project/report.py',   # search-state
           'infrarsi/project/resume.py', 'infrarsi/search/__init__.py', 'infrarsi/search/checkpoint.py',   # checkpoints,
           'infrarsi/search/engine.py', 'infrarsi/search/fold.py', 'infrarsi/search/llmopt.py',   # the bounded asks:
           'infrarsi/search/proposers.py', 'infrarsi/search/wcs.py'}      # test_v41.py and the Codex review cover them)
    v41_tree = root.name in ('pc_code_o4b', 'pc_code_o2b')
    for p in sorted((root / 'infrarsi').rglob('*.py')):
        rel = p.relative_to(root).as_posix()
        if rel == 'infrarsi/search/finalist_base.py' or (rel.startswith('infrarsi/search/')
                                                          and p.stem.startswith(finalists) and not (base / rel).exists()):
            continue
        if v41_tree and rel in v41:
            continue
        b = base / rel
        assert b.is_file(), rel
        if rel in added:
            ops = difflib.SequenceMatcher(None, b.read_text(encoding='utf-8').splitlines(),
                                          p.read_text(encoding='utf-8').splitlines(), autojunk=False).get_opcodes()
            assert all(tag in ('equal', 'insert') for tag, *_ in ops), rel
            continue
        assert hashlib.sha256(p.read_bytes()).hexdigest() == hashlib.sha256(b.read_bytes()).hexdigest(), rel


def test_the_toybench_arms_are_unchanged():
    from infrarsi.search.toybench import arms
    names = set(arms(SimpleNamespace(task=SimpleNamespace(seed=1), memory_facts=None)))
    assert {'random', 'wcs', 'opro', 'ltune', 'xfer', 'xfer-src', 'esls', 'rpp'} <= names
    assert not {'fbdummy', 'finalist'} & names


def test_the_module_names_no_knob_model_or_library():
    src = (SRC / 'finalist_base.py').read_text(encoding='utf-8')
    code = '\n'.join(ln for ln in src.splitlines() if not ln.strip().startswith('#'))
    for word in ('recompute', 'micro_batch', 'liger', 'compile', 'fsdp', 'deepspeed', 'megatron', 'torch', 'qwen', 'gemma',
                 'llama', 'flash', 'expandable', 'checkpointing', 'nccl', 'cudnn', 'triton', 'sdpa', 'hf_trainer', 'nemo',
                 'deepseek', 'transformers'):
        assert not re.search(rf'\b{word}\b', code, re.I), word
    assert not re.search(r'\b(Q9U|Q9H|G4U|M35H[24]|Q27U|Q27H2|E16)\b', code)   # nor an environment
