"""v4 (FABLE_CODE_DIAGNOSIS_20261102 F1-F10; tree pc_code_o2 - bcomp v4): the fixes' rules
one by one on the shared base (finalist_base, explore, the session's line accounting in agent/search.py), the gate (the
code track open and a budget beyond ten screens; a knob-only arm and every ten-screen arm unchanged), the scope of the v4
tree against its open tree, and a scripted twenty-screen run of BComp with v4 on and off.
The replays of the real records are in test_v4_replays.py."""
import dataclasses
import difflib
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from finalist_helpers import CodeProposer, Dummy, fin
from infrarsi.agent.backends.base import LLMResult
from infrarsi.search import finalist_base as FBM
from infrarsi.search.finalist_base import (FLOOR_SD, OVERLAY_CEILING, SLOW_START, V4_PARTS, CallGuard, stream_charge)
from infrarsi.search.toytrain import mem_tasks
from infrarsi.search.wcs import vk
from test_explore import ExB, base_levels, cfg, prior, put, setup
from test_wcs import KNOBS, START

from infrarsi.search.bcomp import BComp

TREE = Path(__file__).resolve().parents[3]
OPEN_TREE = {'pc_code_o2': 'pc_code_o1'}
V4_FILES = {'infrarsi/search/finalist_base.py', 'infrarsi/search/explore.py', 'infrarsi/agent/search.py',
            'infrarsi/search/bcomp.py', 'infrarsi/search/proposers.py'}
ALL_OFF = sorted(V4_PARTS)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ran_one(st, eng):
    """One knob run through the base's proposal (a full screen done: the floor and the asks may hold from now on)."""
    return put(st, eng, {'liger': True})


# ---------------------------------------------------------------- the gate
@pytest.mark.parametrize('screens,applies', [(5, False), (10, False), (11, True), (20, True)])
def test_v4_applies_only_with_the_code_track_open_beyond_ten_screens(screens, applies):
    st, eng = fin(None, screens=screens)
    st._init(eng)
    assert all(st._v4(p, eng) == applies for p in V4_PARTS)
    assert st.finalist_view()['v4']['applies'] == applies
    st2, eng2 = fin(None, screens=screens, v4_off=['v4_floor'])
    st2._init(eng2)
    assert not st2._v4('v4_floor', eng2) and st2._v4('v4_stack', eng2) == applies
    assert st2.finalist_view()['v4']['parts_off'] == ['v4_floor']


def test_a_knob_only_arm_never_runs_v4():
    st, eng = fin(None, screens=20)
    st.proposer.hooks = SimpleNamespace(layers=('L0',))         # (code.mutable empty: the knob-only gate)
    st._init(eng)
    assert not st._can_code() and not any(st._v4(p, eng) for p in V4_PARTS)
    assert st.finalist_view()['v4']['applies'] is False


# ---------------------------------------------------------------- the trees
def test_the_v4_tree_differs_from_its_open_tree_only_in_the_v4_files():
    base = TREE.parent / OPEN_TREE.get(TREE.name, '')
    if TREE.name not in OPEN_TREE or not base.is_dir():
        pytest.skip('not a v4 tree next to its open tree')
    changed = set()
    for p in sorted((TREE / 'infrarsi').rglob('*.py')):
        rel = p.relative_to(TREE).as_posix()
        b = base / rel
        assert b.is_file(), rel                                  # (no module added)
        if sha(p) != sha(b):
            changed.add(rel)
    assert changed and changed <= V4_FILES, changed - V4_FILES
    assert {p.relative_to(base).as_posix() for p in (base / 'infrarsi').rglob('*.py')} == \
        {p.relative_to(TREE).as_posix() for p in (TREE / 'infrarsi').rglob('*.py')}
    old = (base / 'infrarsi/agent/search.py').read_text(encoding='utf-8').splitlines()   # (F4: lines inserted only)
    new = (TREE / 'infrarsi/agent/search.py').read_text(encoding='utf-8').splitlines()
    assert all(t in ('equal', 'insert') for t, *_ in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes())


# ---------------------------------------------------------------- F1: the floor
def _floored(screens=20, **kw):
    st, eng = setup(ExB, prior(*base_levels()), screens=screens)
    for k, v in kw.items():
        setattr(st, k, v)
    ran_one(st, eng)
    return st, eng


def test_f1_the_floor_refuses_a_held_pick_below_its_two_sigma_upside_or_p():
    st, eng = _floored()
    inc = eng.incumbent()
    thr = st._fb_floor_thr(eng)
    assert thr == pytest.approx(max(math.log1p(0.02), 2 * eng.sigma_pair()))
    c = cfg(wk=4)
    st._mu = lambda e, i, x, ids=(): (thr - FLOOR_SD * 0.004 - 1e-4, 0.004)      # (upside just below the floor)
    assert 'two-sigma upside' in st._fb_floor_why(eng, inc, c, (), 0.9, 'draw') and st.fb_floor_refused
    assert st._fb_floor_why(eng, inc, c, (), 0.9, 'complete') is None             # (not a held kind)
    st._mu = lambda e, i, x, ids=(): (-0.01, 0.02)                                # (negative, upside above: passes)
    assert st._fb_floor_why(eng, inc, c, (), 0.9, 'tail') is None
    assert 'P 0.40 below 0.5' in st._fb_floor_why(eng, inc, c, (), 0.4, 'tail')
    assert st.v4_counts['floor_refused'] == 2 and st.finalist_view()['v4']['floor'][-1]['kind'] == 'tail'
    st._fb_floor_lift = True                                                      # (lifted for a step: nothing held)
    assert st._fb_floor_why(eng, inc, c, (), 0.4, 'tail') is None


@pytest.mark.parametrize('why', ['ten screens', 'no screen ran', 'one screen left', 'no budget', 'part off'])
def test_f1_the_floor_holds_only_while_the_code_track_can_take_the_screen(why):
    st, eng = setup(ExB, prior(*base_levels()), screens=10 if why == 'ten screens' else 20)
    if why != 'no screen ran':
        ran_one(st, eng)
    if why == 'one screen left':
        st._left = lambda e: 1
    if why == 'no budget':
        st._budget_ok = lambda e, reserve=0: False
    if why == 'part off':
        st.v4_on['v4_floor'] = False
    st._mu = lambda e, i, x, ids=(): (-0.5, 0.001)
    assert st._fb_floor_why(eng, eng.incumbent(), cfg(wk=4), (), 0.9, 'draw') is None


def test_f1_a_step_the_floor_empties_goes_to_the_code_track_else_runs_with_the_floor_lifted():
    st, eng = _floored()
    held = []

    def knob_side():                                  # (a draw the floor refuses unless it is lifted)
        if st._fb_floor_why(eng, eng.incumbent(), cfg(wk=4), (), 0.9, 'draw'):
            held.append('refused')
            return None
        return 'knob pick'
    st._mu = lambda e, i, x, ids=(): (-0.5, 0.001)
    st._fb_floor_code = lambda e, i, left, why: 'code pick'
    assert st._fb_v4_step(eng, eng.incumbent(), 5, knob_side) == 'code pick' and st.v4_counts['floor_to_code'] == 1
    st._fb_floor_code = lambda e, i, left, why: None   # (the code track has nothing: the floor lifted for the step)
    assert st._fb_v4_step(eng, eng.incumbent(), 5, knob_side) == 'knob pick' and st.v4_counts['floor_lifted'] == 1
    assert st.v4_floor[-1]['lifted'] and not st._fb_floor_lift and held == ['refused', 'refused']


# ---------------------------------------------------------------- F2: the stack
class _C:
    def __init__(self, cid, text, parent):
        self.id, self.overlay, self.parent = cid, {'f.py': text}, parent


def _stack_state(limit=None, gains=(1.5, 1.2), sp=0.01, extra=None):
    """A finalist with a chain p1 (candidate c1, the incumbent) and proven, unadopted patches p2 / p3 that each changed
    one other line of f.py on c1 (a fake engine: candidates with overlays, no hooks)."""
    base = 'a\nb\nc\nd\ne\n'
    cands = {'c0': _C('c0', base, None), 'c1': _C('c1', 'A\nb\nc\nd\ne\n', 'c0'),
             'c2': _C('c2', 'A\nB\nc\nd\ne\n', 'c1'), 'c3': _C('c3', 'A\nb\nc\nD\ne\n', 'c1')}
    if extra:
        cands.update(extra)
    s = SimpleNamespace(candidates=cands, hooks=None, patch_line_limit=limit, budget=SimpleNamespace(screens=20))
    eng = SimpleNamespace(s=s, sigma_pair=lambda: sp, full_screens=lambda: 6, _fresh=lambda q: True,
                          incumbent=lambda: 'c1')
    st = Dummy(CodeProposer(None), facts={'cap_gib': 100.0}, seed=3)
    st.engine = eng
    st.chain, st.overlay_best = ['p1'], dict(cands['c1'].overlay)
    st.patches = {'p1': {'status': 'proven', 'adopted': True, 'candidate': 'c1', 'gain_pct': 9.0},
                  'p2': {'status': 'proven', 'adopted': False, 'candidate': 'c2', 'gain_pct': gains[0]},
                  'p3': {'status': 'proven', 'adopted': False, 'candidate': 'c3', 'gain_pct': gains[1]}}
    st.v4_stack = ['p2', 'p3']
    st._pf = lambda c, ids=(): (0.9, '')
    st._knob_cfg = lambda e, cid: {}
    st._left = lambda e: 5
    return st, eng


def test_f2_proven_patches_within_the_noise_are_tested_together_once_their_sum_passes_the_floor():
    st, eng = _stack_state()
    q = st._fb_stack_q(eng, 'c1')
    assert q is not None and q.hypothesis.startswith('stack|') and q.fidelity == ('full',) and q.base == 'c1'
    assert q.patch == {'f.py': 'A\nB\nc\nD\ne\n'}                 # (both edits merged onto the chain's files)
    assert st.ideas[q.hypothesis] == frozenset({'p1', 'p2', 'p3'}) and st.v4_stacks[-1]['members'] == ['p2', 'p3']
    assert st._fb_stack_q(eng, 'c1') is q and st.v4_stack == [] and st.v4_counts['stack_tests'] == 1
    assert q.mu == pytest.approx(math.log1p(0.015) + math.log1p(0.012))
    # measured above sigma_pair over its base: the members join the chain, the run's files carried
    st.runs = {'c1': {'y': 0.10}, 'c9': {'y': 0.13}}
    eng.s.candidates['c9'] = _C('c9', q.patch['f.py'], 'c1')
    st._fb_stack_seen(eng, q, SimpleNamespace(cid='c9', status='screened'), 'ok')
    assert st.chain == ['p1', 'p2', 'p3'] and st.overlay_best == {'f.py': 'A\nB\nc\nD\ne\n'}
    assert st.patches['p2']['adopted'] and st.patches['p3']['adopted_via'] == 'c9' and st._v4_stack_q is None
    assert st.v4_stacks[-1]['adopted'] == 'measured above sigma_pair' and st.v4_counts['stack_adopted'] == 1


def test_f2_no_stack_below_the_floor_or_beyond_the_line_limit_and_a_conflict_leaves_the_stack():
    st, eng = _stack_state(gains=(0.6, 0.5))                     # (sum below max(2 %, 2 sigma_pair))
    assert st._fb_stack_q(eng, 'c1') is None and st.v4_stack == ['p2', 'p3']
    st, eng = _stack_state(limit=3)                              # (2 + 2 own lines over a limit of 3: one fits)
    assert st._fb_stack_q(eng, 'c1') is None
    st, eng = _stack_state(limit=4)
    assert st._fb_stack_q(eng, 'c1') is not None
    st, eng = _stack_state(extra={'c4': _C('c4', 'A\nX\nc\nd\ne\n', 'c1')})   # (p4 edits p2's line otherwise)
    st.patches['p4'] = {'status': 'proven', 'adopted': False, 'candidate': 'c4', 'gain_pct': 1.0}
    st.v4_stack = ['p2', 'p3', 'p4']
    q = st._fb_stack_q(eng, 'c1')
    assert q is not None and st.v4_stacks[-1]['members'] == ['p2', 'p3'] and 'p4' not in st.v4_stack
    assert st.patches['p4']['stack'] == 'its edit does not merge onto the chain' and st.v4_counts['stack_dropped'] == 1


def test_f2_a_stack_test_within_the_noise_adopts_nothing_unless_the_engine_made_it_the_incumbent():
    st, eng = _stack_state()
    q = st._fb_stack_q(eng, 'c1')
    st.runs = {'c1': {'y': 0.10}, 'c9': {'y': 0.105}}
    eng.s.candidates['c9'] = _C('c9', q.patch['f.py'], 'c1')
    st._fb_stack_seen(eng, q, SimpleNamespace(cid='c9', status='screened'), 'ok')
    assert st.chain == ['p1'] and st.v4_stacks[-1]['adopted'] is False
    st, eng = _stack_state()
    q = st._fb_stack_q(eng, 'c1')
    st.runs = {'c1': {'y': 0.10}, 'c9': {'y': 0.105}}
    eng.s.candidates['c9'] = _C('c9', q.patch['f.py'], 'c1')
    eng.incumbent = lambda: 'c9'
    st._fb_stack_seen(eng, q, SimpleNamespace(cid='c9', status='screened'), 'ok')
    assert st.chain == ['p1', 'p2', 'p3'] and st.v4_stacks[-1]['adopted'] == 'the engine made it the incumbent'


def test_f2_two_patches_appended_at_the_same_place_stack_and_the_session_accepts_the_test():
    """On a real session (the memory toy's hooks: files through candidate_files, F4's own-lines count on): a chain patch
    and two patches each appended after it on the chain's candidate merge (both insertions kept) and the stack's files are
    a candidate the session accepts on the incumbent."""
    s = _session(True)
    base = s.hooks.candidate_files(s.candidates['c0'].overlay)['patches.txt']
    c1 = s.propose_patch(files={'patches.txt': base + 'chain1\nproof:chain1\n'}, base='c0')['candidate_id']
    t1 = s.hooks.candidate_files(s.candidates[c1].overlay)['patches.txt']
    c2 = s.propose_patch(files={'patches.txt': t1 + 'pa\nproof:pa\n'}, base=c1)['candidate_id']
    c3 = s.propose_patch(files={'patches.txt': t1 + 'pb\nproof:pb\n'}, base=c1)['candidate_id']
    eng = SimpleNamespace(s=s, sigma_pair=lambda: 0.005, full_screens=lambda: 6, _fresh=lambda q: True,
                          incumbent=lambda: c1)
    st = Dummy(CodeProposer(None), facts={'cap_gib': 100.0}, seed=3)
    st.engine, st.chain, st.overlay_best = eng, ['p1'], dict(s.candidates[c1].overlay)
    st.patches = {'p1': {'status': 'proven', 'adopted': True, 'candidate': c1, 'gain_pct': 5.0},
                  'p2': {'status': 'proven', 'adopted': False, 'candidate': c2, 'gain_pct': 1.2},
                  'p3': {'status': 'proven', 'adopted': False, 'candidate': c3, 'gain_pct': 1.0}}
    st.v4_stack = ['p2', 'p3']
    st._pf, st._knob_cfg, st._left = (lambda c, ids=(): (0.9, '')), (lambda e, cid: {}), (lambda e: 5)
    q = st._fb_stack_q(eng, c1)
    assert q is not None and q.patch['patches.txt'] == t1 + 'pa\nproof:pa\npb\nproof:pb\n' and q.base == c1
    got = s.propose_patch(files=dict(q.patch), base=q.base)
    assert 'candidate_id' in got and got['diff_lines'] == 4      # (its own lines: the two patches' four, the toy's proofs)


# ---------------------------------------------------------------- F3: never-repeat and a patch's failure
def _with_patch_failure(screens=20):
    st, eng = setup(ExB, prior(*base_levels()), screens=screens)
    ran_one(st, eng)
    inc_cfg = st._knob_cfg(eng, eng.incumbent())
    st.runs['c90'] = {'cfg': {**inc_cfg, 'patch:p9': True}, 'kind': 'thrash', 'y': None, 'status': 'infeasible',
                      'fidelity': 'probe', 'origin': 'wcs-patch', 'base': eng.incumbent(), 'bound': False}
    st.pids['c90'] = frozenset({'p9'})                            # (a patch's probe on the incumbent's values)
    st.fb_obs += 1
    st._fb_forget()
    return st, eng, inc_cfg


def test_f3_a_failure_that_carried_an_unproven_patch_is_not_a_knob_repeat():
    st, eng, inc_cfg = _with_patch_failure()
    again = {**inc_cfg, 'wk': 4}                                  # (the same big knobs, no saver added)
    assert st._ex_admissible(again, inc_cfg)[0]                   # v4: the failure is p9's
    st.v4_on['v4_repeat'] = False                                 # v3: refused for it
    st.fb_obs += 1
    ok, why = st._ex_admissible(again, inc_cfg)
    assert not ok and why.startswith('it repeats c90')
    st, eng, inc_cfg = _with_patch_failure(screens=10)            # (ten screens: v3's rule)
    assert not st._ex_admissible({**inc_cfg, 'wk': 4}, inc_cfg)[0]


def test_f3_a_knob_failure_is_still_never_repeated_and_a_carried_patch_counts():
    st, eng, inc_cfg = _with_patch_failure()
    st.chain.append('p9')                                         # (p9 carried now: its failure would repeat)
    st.fb_obs += 1
    assert not st._ex_admissible({**inc_cfg, 'wk': 4}, inc_cfg)[0]
    st, eng, inc_cfg = _with_patch_failure()
    del st.pids['c90']
    st.runs['c90']['cfg'] = dict(inc_cfg)                         # (a knob-only failure)
    st.fb_obs += 1
    ok, why = st._ex_admissible({**inc_cfg, 'wk': 4}, inc_cfg)
    assert not ok and why.startswith('it repeats c90')


def test_f3_a_failure_isolated_blame_gives_a_patch_is_that_patchs():
    st, eng, inc_cfg = _with_patch_failure()
    st.chain.append('p9')
    st.fb_obs += 1
    st.isolated_blame = lambda: {('patch:p8', 'true'): ['c90']}   # (blamed on p8, which the candidate does not carry)
    assert st._ex_patches_of('c90', st.runs['c90']) == frozenset({'p8'})
    assert st._ex_admissible({**inc_cfg, 'wk': 4}, inc_cfg)[0]


# ---------------------------------------------------------------- F4: the session's own-lines count
def _session(own: bool):
    from infrarsi.agent.budget import VirtualClock
    from infrarsi.search import toybench as tb
    sim = tb.make_sim(mem_tasks()[1])
    clock = VirtualClock()
    runner = tb.ToyRunner(sim, clock)
    ref_cfg = sim.reference()
    probe = tb.EnvTask.from_dict(tb.toy_task_spec(sim, {'screens': 1, 'llm_tokens': 0, 'wall_seconds': 1e12,
                                                        'gpu_hours': 1e9}))
    ref = runner.run(probe, ref_cfg, None, 'reference', 'toy', candidate_id='c0', steps=tb.FULL)
    env = tb.EnvTask.from_dict(tb.toy_task_spec(sim, {'screens': 20, 'llm_tokens': 10 ** 7, 'wall_seconds': 1e12,
                                                      'gpu_hours': 1e9}))
    hooks = tb.MemoryHooks(env, sim.memory_facts, base_files={'patches.txt': ''})
    s = tb.SearchSession(task=env, budget=env.search_budget(), runner=runner,
                         judge=tb.Verifier(0.03, memory_cap_gib=sim.hw['mem']), run_id='v4-f4', arm='f4',
                         start_config=ref_cfg, start_receipt=ref, reference_receipt=ref, hooks=hooks, clock=clock)
    if own:
        s.patch_lines_own, s.patch_overlay_ceiling = True, OVERLAY_CEILING
    return s


@pytest.mark.parametrize('own', [False, True])
def test_f4_the_line_limit_counts_the_new_patchs_own_lines_under_an_overlay_ceiling(own):
    s = _session(own)
    lim = min(int(s.hooks.patch_line_limit), int(s.patch_line_limit or 300))
    first = ''.join(f'a{i}\n' for i in range(lim - 2))
    r1 = s.propose_patch(files={'patches.txt': first}, base='c0')
    assert 'candidate_id' in r1, r1
    more = first + ''.join(f'b{i}\n' for i in range(5))           # (5 new lines; the overlay over the limit)
    r2 = s.propose_patch(files={'patches.txt': more}, base=r1['candidate_id'])
    if not own:
        assert 'candidate_id' not in r2 and r2['diff_lines'] > lim                 # v3: the cumulative count
        return
    assert 'candidate_id' in r2 and r2['diff_lines'] == 5                          # v4: its own lines
    proofs = more + ''.join(f'print("[infrarsi-patch] s{i}: found 1 -> set 2")\n' for i in range(lim + 5)) + 'c\n'
    r3 = s.propose_patch(files={'patches.txt': proofs}, base=r2['candidate_id'])
    assert 'candidate_id' in r3 and r3['diff_lines'] == 1                          # (the self-proof's lines left out)
    big = ''.join(f'z{i}\n' for i in range(OVERLAY_CEILING * lim + 1))
    r4 = s.propose_patch(files={'patches.txt': big}, base='c0')                     # (over the limit on its own and
    assert 'candidate_id' not in r4 and any('own lines' in f for f in r4['findings'])   # over the ceiling)
    assert any('the whole overlay' in f for f in r4['findings'])


# ---------------------------------------------------------------- F5: a timed-out call
def _stream(tmp_path, *, msgs=2, thinking=3000, text='abc'):
    lines = [{'type': 'system', 'subtype': 'init'}]
    for i in range(msgs):
        lines.append({'type': 'assistant', 'message': {'id': f'm{i}', 'content': [{'type': 'thinking'}],
                                                       'usage': {'input_tokens': 2, 'cache_creation_input_tokens': 1000,
                                                                 'cache_read_input_tokens': 500, 'output_tokens': 5,
                                                                 'cache_creation': {'ephemeral_1h_input_tokens': 1000}}}})
    lines += [{'type': 'system', 'subtype': 'thinking_tokens', 'estimated_tokens_delta': thinking // 3}] * 3
    if text:
        lines.append({'type': 'assistant', 'message': {'id': 'm0', 'content': [{'type': 'text', 'text': text}]}})
    (tmp_path / 'stdout.jsonl').write_text('\n'.join(json.dumps(x) for x in lines) + '\n', encoding='utf-8')
    return tmp_path


def test_f5_a_timed_out_call_is_charged_what_its_stream_reported(tmp_path):
    from infrarsi.agent.backends.anthropic_api import PRICES
    p = PRICES['claude-opus-5-5']
    usd, how = stream_charge(_stream(tmp_path), 'claude-opus-5-5')
    out = 3000 + math.ceil(3 / 3.5)
    want = (4 * p[0] + out * p[1] + 1000 * p[2] + 2000 * FBM.CACHE_1H * p[0]) / 1e6
    assert usd == pytest.approx(want) and '2 message(s)' in how and f'{out} output tokens' in how
    assert stream_charge(tmp_path / 'none', 'claude-opus-5-5')[0] is None
    assert stream_charge(tmp_path, 'not-a-priced-model')[0] is None


@pytest.mark.parametrize('applies', [True, False])
def test_f5_the_guard_reprices_a_timeout_only_and_only_under_v4(tmp_path, applies):
    d = _stream(tmp_path)
    timeout = LLMResult(action=None, text=None, llm_tokens=67000, llm_usd=5.0, wall_seconds=900.0,
                        accounting_complete=False, error='CLI failed or incomplete result', model='claude-opus-5-5',
                        observed_models=[], meta={'status': 'TIMEOUT', 'call_dir': str(d), 'call_id': 'x'})
    ok = dataclasses.replace(timeout, meta={'status': 'RETURNED', 'call_dir': str(d)}, llm_usd=1.0)
    inner = SimpleNamespace(answers=[timeout, ok], timeout_seconds=900, max_retries=1)
    inner.complete = lambda system, prompt, **kw: inner.answers.pop(0)
    g = CallGuard(inner, SimpleNamespace(_v4=lambda part, engine=None: applies))
    r = g.complete('s', 'p')
    want = stream_charge(d, 'claude-opus-5-5')[0]
    assert g.last_status == 'TIMEOUT' and r.llm_usd == pytest.approx(want if applies else 5.0)
    if applies:
        assert r.meta['upper_bound_usd'] == 5.0 and g.repriced[-1]['charged_usd'] == round(want, 4)
    assert g.complete('s', 'p').llm_usd == 1.0 and g.last_status == 'RETURNED'
    g.max_retries = 0                                             # (attributes are the backend's: read and set)
    g.timeout_seconds = 1200.0
    assert inner.max_retries == 0 and inner.timeout_seconds == 1200.0 and g.timeout_seconds == 1200.0


def test_f5_the_slim_reask_follows_a_timeout():
    st, eng = fin(None, screens=20)
    st._init(eng)
    guard = CallGuard(SimpleNamespace(timeout_seconds=900.0), st)
    st._bdriver = SimpleNamespace(_inner=guard, attempts=0, usd=0.0, usd_left=lambda: 100.0)   # (a driver's stand-in)
    st._budget_ok = lambda e, reserve=0: True
    assert not st._fb_slim_due(eng)
    guard.__dict__['last_status'] = 'TIMEOUT'
    assert st._fb_slim_due(eng)
    st._fb_slim = True                                            # (the slim re-ask itself: never a second)
    assert not st._fb_slim_due(eng)
    st._fb_slim = False


# ---------------------------------------------------------------- the spec's budget (FABLE_V4_BUDGET_20261102 section 1)
PARITY = {'search': {'budget': {'llm_usd': 40, 'llm_tokens': 200000000, 'llm_calls': 60, 'report_usd': 16},
                     'caps': {'fix_calls_per_screen': 2, 'fix_calls': 20, 'patches_per_ask': 4, 'code_probes': 12}},
          'llm': {'max_usd_per_call': 8, 'timeout_s': 1200}}


def _with_manifest(screens, manifest, knob_only=False, **kw):
    """The dummy on the simulated engine whose task carries `manifest` (as a project's task does), a backend stand-in
    inside a budget driver stand-in, started."""
    st, eng = fin(None, screens=screens, **kw)
    eng.s.task.raw = {'manifest': manifest}                          # (the simulated session's task: a namespace)
    if knob_only:
        st.proposer.hooks = SimpleNamespace(layers=('L0',))
    backend = SimpleNamespace(timeout_seconds=900.0, max_usd_per_call=5.0)
    st._bdriver = SimpleNamespace(_inner=CallGuard(backend, st), attempts=0, usd=0.0, usd_left=lambda: 100.0)
    st._init(eng)
    return st, eng, backend


def caps_of(st, be) -> tuple:
    return (st.calls_max, st.code_asks, st.usd_max, st.patches_per_ask, st.ladder.code_probes_max, st.history_max,
            st.v4_fix_max, st.v4_fix_per_screen, be.timeout_seconds, be.max_usd_per_call)


def test_a_spec_with_the_parity_keys_yields_these_caps_and_one_without_them_v3s():
    st, eng, be = _with_manifest(20, PARITY)
    assert caps_of(st, be) == (60, 60, 40.0, 4, 12, 80, 20, 2, 1200.0, 8.0)
    assert st.finalist_view()['v4']['spec_caps'] == {'usd': 40.0, 'calls': 60, 'patches_per_ask': 4, 'code_probes': 12,
                                                     'fix_calls': 20, 'fix_calls_per_screen': 2, 'call_usd': 8.0,
                                                     'call_wall_s': 1200.0}
    st, eng, be = _with_manifest(20, {})                            # (the open v3 tree's at twenty screens)
    assert caps_of(st, be) == (20, 20, 16.0, 3, 8, 80, 0, 0, 900.0, 5.0) and st.fb_spec_caps == {}
    st, eng, be = _with_manifest(10, {})                            # (v3's at ten)
    assert caps_of(st, be) == (10, 10, 8.0, 3, 4, 40, 0, 0, 900.0, 5.0)
    st, eng, be = _with_manifest(20, PARITY, knob_only=True)        # (gated on the code track: a knob-only arm ignores it)
    assert caps_of(st, be)[:8] == (20, 20, 16.0, 3, 8, 80, 0, 0) and st.fb_spec_caps == {}
    st, eng, be = _with_manifest(20, PARITY, calls_max=7)           # (a cap given to the constructor is kept)
    assert (st.calls_max, st.usd_max) == (7, 40.0)
    st, eng, be = _with_manifest(10, PARITY)                        # (the spec's caps hold at any budget)
    assert caps_of(st, be)[:4] == (60, 60, 40.0, 4)
    bad = {'search': {'budget': {'llm_calls': 'x', 'llm_usd': -1}, 'caps': {'fix_calls': True}}, 'llm': {'timeout_s': 0}}
    assert FBM.spec_budget(SimpleNamespace(s=SimpleNamespace(task=SimpleNamespace(raw={'manifest': bad}),
                                                             budget=SimpleNamespace(llm_usd=None)))) == {}
    assert FBM.spec_budget(SimpleNamespace(s=SimpleNamespace(budget=SimpleNamespace(llm_usd=33.0)))) == {'usd': 33.0}


FIXCAPS = {'search': {'caps': {'fix_calls_per_screen': 2, 'fix_calls': 3}}}


def test_a_refused_or_failed_patch_waits_for_one_fix_call_with_its_finding():
    st, eng, _ = _with_manifest(20, FIXCAPS)
    eng.s.events = [{'kind': 'invalid_proposal', 'payload': {'findings': ["diff 312 lines > limit 300 (this patch's own "
                                                                          'lines against its base)']}}]
    refused = SimpleNamespace(status='rejected', cid=None, fidelity='none', reasons=['patch rejected'])
    st._fb_fix_note(eng, SimpleNamespace(hypothesis='patch|p7', base='c0'), refused, None)
    st._fb_fix_note(eng, SimpleNamespace(hypothesis='patch|p7', base='c0'), refused, None)   # (each patch once)
    assert [r['pid'] for r in st.v4_fix_pending] == ['p7'] and st.v4_fix_pending[0]['stage'] == 'guard'
    assert st.v4_fix_pending[0]['finding'][0].startswith('diff 312 lines > limit 300')
    oom = SimpleNamespace(status='infeasible', cid='c99', fidelity='probe',
                          reasons=['completes: exit code 1, 0/30 updates; the log: OutOfMemoryError: out of memory'])
    st._fb_fix_note(eng, SimpleNamespace(hypothesis='patch|p8', base='c0'), oom, {'kind': 'oom', 'line': 'oom'})
    assert st.v4_fix_pending[-1]['stage'] == 'probe' and st.v4_fix_pending[-1]['finding'][0] == 'oom at its probe'
    assert 'OutOfMemoryError' in ' '.join(st.v4_fix_pending[-1]['finding'])
    slower = SimpleNamespace(status='aborted', cid='c98', fidelity='full', reasons=['slower'])
    st._fb_fix_note(eng, SimpleNamespace(hypothesis='patch|p9', base='c0'), slower, {'kind': 'slower'})
    assert len(st.v4_fix_pending) == 2                               # (a full run is not fixed)
    st0, eng0, _ = _with_manifest(20, {})                            # (no fix calls set: nothing waits)
    st0._fb_fix_note(eng0, SimpleNamespace(hypothesis='patch|p7', base='c0'), oom, {'kind': 'oom'})
    assert not st0.v4_fix_pending and not st0._fb_fix_due(eng0, 5)


def test_the_fix_call_carries_the_finding_queues_one_patch_and_keeps_to_its_caps():
    from test_finalist_base import ph
    st, eng, _ = _with_manifest(20, FIXCAPS)
    st._bdriver = None                                               # (the scripted model: no driver)
    ran_one(st, eng)
    st._fb_code_probe_ok = lambda e: True
    st._budget_ok = lambda e, reserve=0: True
    st.proposer.answers = lambda ctx, n: [ph(f'f{n}', '[fix p7] fewer lines'), ph(f'g{n}')]
    for pid in ('p7', 'p8', 'p9', 'p10'):
        st.patches[pid] = {'mechanism': f'patch {pid}', 'site': 'mod.f', 'files': {}}
        st.v4_fix_pending.append({'pid': pid, 'stage': 'guard', 'kind': 'rejected', 'base': 'c0',
                                  'finding': ['diff 312 lines > limit 300']})
    inc, n = eng.incumbent(), eng.full_screens()
    got = st._fb_fix_ask(eng, inc, 5)
    field = st.proposer.seen[-1]['fix_refused']
    assert len(got) == 1 and field['id'] == 'p7' and field['stage'] == 'guard' and field['rule'] == FBM.FIX_REFUSED_RULE
    new = got[0].hypothesis.split('|')[1]
    assert field['finding'] == ['diff 312 lines > limit 300'] and st.fb_fixes[new] == 'p7'
    assert st.fb_asks[-1]['fix_refused'] == 'p7' and st.v4_fix_log[-1]['queued'] == [new]
    assert st._fb_fix_due(eng, 5) and st._fb_fix_ask(eng, inc, 5)    # (the second of this screen)
    assert not st._fb_fix_due(eng, 5)                                # (two per screen)
    eng.full_screens = lambda: n + 1
    assert st._fb_fix_due(eng, 5) and st._fb_fix_ask(eng, inc, 5)    # (the arm's third)
    eng.full_screens = lambda: n + 2
    assert not st._fb_fix_due(eng, 5) and st.v4_counts['fix_calls'] == 3 and 'fix_refused' not in st.context(eng, inc)


# ---------------------------------------------------------------- F6: the code track before the arm ends
@pytest.mark.parametrize('never_done', [True, False])
def test_f6_an_empty_step_asks_again_while_an_ask_brings_nothing_and_makes_a_call(never_done):
    st, eng = _floored()
    st.v4_on['v4_never_done'] = never_done
    calls = {'n': 0}
    answers = [None, None, SimpleNamespace(hypothesis='patch|q')]

    def code_step(e, inc, left, ask=True):
        if not ask:
            return None
        st.asks += 1
        calls['n'] += 1
        return answers.pop(0)
    st._code_step = code_step
    st._llm_used = lambda: calls['n']
    got = st._fb_floor_code(eng, eng.incumbent(), 5, 'test')
    if never_done:
        assert got.hypothesis == 'patch|q' and st.asks == 3 and st.v4_counts['reasks'] == 2
    else:
        assert got is None and st.asks == 1
    assert st.v4_code[-1]['asks'] == (3 if never_done else 1)


def test_f6_the_ask_gate_lets_an_ask_follow_one_whose_patches_ended_before_a_run():
    st, eng = _floored()
    inc, left = eng.incumbent(), 5
    st.fb_runs_at_ask = st.fb_runs                                # (no run since the last ask)
    st._fb_code_probe_ok = lambda e: True
    assert st._fb_ask_due(eng, inc, left) is None
    st._fb_force_ask, st._fb_force_why = True, 'the knob side has nothing to run (F6)'
    assert 'handed the screen' in st._fb_ask_due(eng, inc, left)
    st.queue = [object()]                                         # (a patch still pending: no ask)
    assert st._fb_ask_due(eng, inc, left) is None


# ---------------------------------------------------------------- F7: a declared slow start
@pytest.mark.parametrize('h,screens,cap', [({'warmup_updates': SLOW_START}, 20, None), ({'active_from_update': 3}, 20, None),
                                           ({'warmup_updates': 1}, 20, 250.0), ({'warmup_updates': 3}, 10, 250.0)])
def test_f7_a_patch_declaring_a_slow_start_probes_without_a_start_cap(h, screens, cap):
    st, eng = fin(None, screens=screens)
    st._init(eng)
    q = SimpleNamespace(fidelity=('probe', 'full'), probe_cap=250.0)
    st._fb_slow_start(eng, q, h)
    assert q.probe_cap == cap
    full = SimpleNamespace(fidelity=('full',), probe_cap=250.0)
    st._fb_slow_start(eng, full, h)
    assert full.probe_cap == 250.0


# ---------------------------------------------------------------- r3: the runs' time limit as Auto-Research's
@pytest.mark.parametrize('screens,off,none', [(20, (), True), (20, ('v4_no_start_cap',), False), (10, (), False)])
def test_r3_with_v4_on_no_probe_has_a_start_cap_the_launch_timeout_bounds_it(screens, off, none, monkeypatch):
    from infrarsi.search.wcs import WCS
    st, eng = fin(None, screens=screens)
    st._init(eng)
    for p in off:
        st.v4_on[p] = False
    monkeypatch.setattr(WCS, '_probe_cap', lambda self, e, base, steps=None: 280.0)   # (a timed reference: 280 s)
    got = st._probe_cap(eng, 'c0')
    assert (got is None) if none else (got == 280.0)


# ---------------------------------------------------------------- F8: an earlier arm's failed start
def test_f8_a_contrarian_move_holding_a_value_that_failed_to_start_elsewhere_is_probed_first():
    st, eng = _floored()
    st.from_store = [{'members': [{'comp': True}, {'rc': 'none'}], 'id': 's1', 'message': 'failed to start'}]
    st._v4_store = None
    inc = eng.incumbent()
    assert st._fb_store_sets() == [frozenset({('comp', vk(True)), ('rc', vk('none'))})]
    plan = st._fb_store_plan(eng, inc, cfg(liger=True, comp=True), ('full',))
    assert plan[0] == 'probe' and st.v4_counts['store_probes'] == 1
    assert st._fb_store_plan(eng, inc, cfg(liger=True, wk=4), ('full',)) == ('full',)   # (no such value)
    st.ladder.plan = lambda *a, **kw: ('full',)                   # (no probe affordable: not run now, eligible)
    assert st._fb_store_plan(eng, inc, cfg(liger=True, comp=True), ('full',)) is None and st.v4_counts['store_held'] == 1


# ---------------------------------------------------------------- F9: the ledger's P
def test_f9_m1_ranks_its_moves_by_the_optimistic_gain_times_the_ledger_p():
    st, eng = _floored()
    e = {'mu': 0.0, 'sd': 0.1, 'p_runs': 1.0}
    st._ex_contra_values = lambda engine, inc_cfg: [(0.2, ('comp', 'true'), True, e, ['r'], 0.2),
                                                    (0.1, ('tf32', 'true'), True, e, ['r'], 0.1)]
    p = {'comp': 0.2, 'tf32': 0.9}
    st._ex_move = lambda engine, inc, k, key, why='', inc_cfg=None: {
        'cfg': cfg(**{k: True}), 'p': p[k], 'note': '', 'saver': None, 'k': k}
    for name in ('_ex_untested', '_ex_kept_failure', '_ex_rules'):
        setattr(st, name, lambda *a, _n=name, **kw: _n == '_ex_untested')
    st._ex_adm = lambda *a, **kw: (True, '')
    moves = st._ex_contra_list(eng, eng.incumbent())
    assert [m['k'] for m in moves] == ['tf32', 'comp'] and moves[0]['score_v4_pct'] == pytest.approx(9.0)
    st.v4_on['v4_ledger_p'] = False
    assert [m['k'] for m in st._ex_contra_list(eng, eng.incumbent())] == ['comp', 'tf32']




# ---------------------------------------------------------------- F10: the do-not-repeat list once
def test_f10_the_do_not_repeat_list_goes_in_the_first_ask_only():
    st, eng = fin(None, screens=20)
    st._init(eng)
    st.fb_target = {'card_gib': 80.0, 'nodes': 1, 'params_b': 9.0, 'framework': 'hf_trainer'}
    env = {'card_gib': 80.0, 'nodes': 1, 'params_b': 8.0, 'framework': 'hf_trainer'}
    st.fb_proven = [{'id': 'x1', 'mechanism': 'a fused step', 'site': 'mod.step', 'gain_pct': 4.0, 'env': env}]
    st.fb_inert = [{'id': 'y1', 'mechanism': 'a cast', 'site': 'mod.cast', 'reason': 'inert: no effect', 'env': env}]
    first = st._fb_transfer_view()
    assert first and first['proven'] and first['do_not_repeat']
    later = st._fb_transfer_view(later=True)
    assert later['proven'] == first['proven'] and 'do_not_repeat' not in later
    st.fb_proven = []
    assert st._fb_transfer_view(later=True) is None


# ---------------------------------------------------------------- r2: the review's findings (reviews/V4_r1_out.md)
def test_r2_an_unaffordable_queued_patch_does_not_hold_back_a_due_stack():
    """Finding 1: a queued patch whose probe cannot be paid (the pool and the allowance spent) no longer blocks the stack
    test that needs no probe; the patch stays queued. v4's stack off: v3's queue as before."""
    from infrarsi.search.engine import Proposal
    for stack_on in (True, False):
        st, eng = _stack_state()
        st.v4_on['v4_stack'] = stack_on
        st.ideas['patch|p9'] = frozenset({'p1', 'p9'})
        st.patches['p9'] = {'status': 'queued', 'mechanism': 'patch p9', 'files': {'f.py': 'A\nb\nc\nd\nE\n'}}
        q9 = Proposal(change={}, patch={'f.py': 'A\nb\nc\nd\nE\n'}, base='c1', hypothesis='patch|p9',
                      fidelity=('probe', 'full'))
        st.queue = [q9]
        st._lowfi_ok = lambda e, q: tuple(q.fidelity)[0] != 'probe'   # (no probe affordable now)
        got = st._code_step(eng, 'c1', 5, ask=False)
        if stack_on:
            assert got is not None and got.hypothesis.startswith('stack|') and got.fidelity == ('full',)
        else:
            assert got is None
        assert st.queue == [q9]


@pytest.mark.parametrize('manifest,want', [(PARITY, 4), ({}, 3)])
def test_r2_the_patch_request_asks_for_the_specs_patches_per_ask(manifest, want):
    """Finding 2: the real proposer's patch-only request carries the spec's patches per ask (absent: three, v3's)."""
    from test_bcomp import _toy_script
    r = _bcomp_toy(_toy_script(), 20, manifest=manifest)
    asks = [pl for role, pl in r.seen if role == 'hypotheses']
    assert asks and {pl['max_hypotheses'] for pl in asks} == {want} and r.st.patches_per_ask == want


def _fix_ready(manifest=FIXCAPS):
    st, eng, _ = _with_manifest(20, manifest)
    st._bdriver = None                                               # (the scripted model: no driver)
    ran_one(st, eng)
    st._fb_code_probe_ok = lambda e: True
    st._budget_ok = lambda e, reserve=0: True
    return st, eng


def test_r2_a_fix_calls_revision_passes_the_failed_patchs_s6_ban_unless_its_files_repeat():
    """Finding 3: the revision the fix call asked for (`[fix p1]`, files changed) is queued and kept by the queue's filter
    although p1 ran out of memory (S6's ban on its mechanism); a reply repeating p1's files is still dropped."""
    from test_finalist_base import ph
    st, eng = _fix_ready()
    orig = ph('m1', 'reduce activation memory')
    st.patches['p1'] = {'mechanism': 'reduce activation memory', 'site': 'm1', 'files': dict(orig['patch'])}
    st.fb_s6_bans.append({'pid': 'p1', 'key': st._fb_mech_key('reduce activation memory'),
                          'files': st._fb_files_key(orig['patch']), 'mechanism': 'reduce activation memory'})
    waiting = {'pid': 'p1', 'stage': 'probe', 'kind': 'oom', 'base': 'c0', 'finding': ['oom at its probe']}
    st.v4_fix_pending.append(dict(waiting))
    st.proposer.answers = lambda ctx, n: [ph('m2', '[fix p1] reduce activation memory')]
    got = st._fb_fix_ask(eng, eng.incumbent(), 5)
    new = got[0].hypothesis.split('|')[1] if got else None
    assert new in st.v4_fix_authorized and st.fb_counts['s6_patches_dropped'] == 0
    assert got[0] in st._code(eng, eng.incumbent(), 5, ask=False)   # (the queue keeps it)
    st.queue.clear()
    st.v4_fix_pending.append(dict(waiting))
    st.proposer.answers = lambda ctx, n: [{**orig, 'mechanism': '[fix p1] the same files again'}]
    assert st._fb_fix_ask(eng, eng.incumbent(), 5) == [] and st.fb_counts['s6_patches_dropped'] == 1
    st.proposer.answers = lambda ctx, n: [ph('m3', 'reduce activation memory')]   # (an ordinary ask's repeat: banned)
    assert st._fb_ask(eng, eng.incumbent(), 5, 'an ask') == [] and st.fb_counts['s6_patches_dropped'] == 2


def test_r2_the_stack_counts_own_lines_without_the_proof_lines():
    """Finding 4: three mergeable patches of 90 lines and 15 proof lines each stack within a 300-line limit (270 own
    lines, F4's count), where counting the proof lines (315) would leave the third out and the sum below the floor."""
    def block(i):
        return ''.join(f'x{i}_{k}\n' for k in range(90)) + ''.join(
            f'print("[infrarsi-patch] s{i}_{k}: found 1 -> set 2")\n' for k in range(15))
    chain = 'a\nCHAIN\n'
    extra = {'c0': _C('c0', 'a\n', None), 'c1': _C('c1', chain, 'c0'), 'c2': _C('c2', chain + block(2), 'c1'),
             'c3': _C('c3', chain + block(3), 'c1'), 'c4': _C('c4', chain + block(4), 'c1')}
    st, eng = _stack_state(limit=300, gains=(0.9, 0.9), extra=extra)
    st.patches['p4'] = {'status': 'proven', 'adopted': False, 'candidate': 'c4', 'gain_pct': 0.9}
    st.v4_stack = ['p2', 'p3', 'p4']
    q = st._fb_stack_q(eng, 'c1')
    assert q is not None and sorted(st.v4_stacks[-1]['members']) == ['p2', 'p3', 'p4']
    assert st.v4_stacks[-1]['own_lines'] == 270


def test_r2_a_derived_probe_keeps_the_patchs_declared_warm_up_for_f7():
    """Finding 5: the patch record keeps warmup_updates, so a derived candidate's proof probe of that patch runs without
    a start cap as the patch's own probe did."""
    from hls2_helpers import BASE_TEXT, PATCH_FILE, snippet
    from infrarsi.search.engine import Proposal
    st, eng = fin(None, screens=20)
    st._init(eng)
    st._probe_cap = lambda e, base, steps=None: 280.0                # (a timed reference: a 280 s start cap)
    h = {'mechanism': 'a slow-starting patch', 'change': {}, 'patch': {PATCH_FILE: BASE_TEXT + snippet('w')}, 'mu': 0.03,
         'sigma': 0.02, 'p_feasible': 0.9, 'site': 'w', 'warmup_updates': 3, 'active_from_update': 1}
    q = st._patch_proposal(eng, eng.incumbent(), h)
    pid = q.hypothesis.split('|')[1]
    assert q.probe_cap is None and st.patches[pid]['warmup_updates'] == 3
    dq = Proposal(change={}, patch=dict(h['patch']), fidelity=('probe', 'full'), probe_cap=280.0)
    st._fb_proof_probe(eng, eng.incumbent(), dq, dict(h['patch']), pid)
    assert dq.probe_cap is None
    st.patches[pid].pop('warmup_updates')                            # (r1's record: the derived probe kept the cap)
    dq2 = Proposal(change={}, patch=dict(h['patch']), fidelity=('probe', 'full'), probe_cap=280.0)
    st._fb_proof_probe(eng, eng.incumbent(), dq2, dict(h['patch']), pid)
    assert dq2.probe_cap == 280.0


def test_r2_in_a_fix_call_the_requested_fix_is_queued_even_when_it_comes_second():
    """The live probe's finding: the model put the `[fix p19]` patch second of three; the fix call queues it, not the
    unrelated first one."""
    from test_finalist_base import ph
    st, eng = _fix_ready()
    st.patches['p7'] = {'mechanism': 'patch p7', 'site': 'mod.f', 'files': {}}
    st.v4_fix_pending.append({'pid': 'p7', 'stage': 'guard', 'kind': 'rejected', 'base': 'c0',
                              'finding': ['diff 314 lines > limit 300']})
    st.proposer.answers = lambda ctx, n: [ph(f'g{n}', 'an unrelated patch'), ph(f'f{n}', '[fix p7] compact rewrite'),
                                          ph(f'h{n}', 'another one')]
    got = st._fb_fix_ask(eng, eng.incumbent(), 5)
    assert len(got) == 1 and st.patches[got[0].hypothesis.split('|')[1]]['mechanism'].startswith('[fix p7]')


# ---------------------------------------------------------------- a scripted twenty-screen run, v4 on and off
def _bcomp_toy(script, screens, manifest=None, **kw):
    """test_bcomp's bcomp_llm_toy with keywords for BComp (the v4 switches), the task's manifest (`manifest`) and the
    payloads the model was sent (`seen`: (role, payload))."""
    from infrarsi.agent.backends.fake import ScriptedBackend
    from infrarsi.agent.budget import VirtualClock
    from infrarsi.search import toybench as tb
    from infrarsi.search.proposers import LLMProposer
    task = mem_tasks()[1]
    sim = tb.make_sim(task)
    clock = VirtualClock()
    runner = tb.ToyRunner(sim, clock)
    ref_cfg = sim.reference()
    probe = tb.EnvTask.from_dict(tb.toy_task_spec(sim, {'screens': 1, 'llm_tokens': 0, 'wall_seconds': 1e12,
                                                        'gpu_hours': 1e9}))
    ref = runner.run(probe, ref_cfg, None, 'reference', 'toy', candidate_id='c0', steps=tb.FULL)
    c_full = ref['cost']['gpu_seconds'] / 3600
    env = tb.EnvTask.from_dict(tb.toy_task_spec(sim, {'screens': screens, 'llm_tokens': 10 ** 7, 'wall_seconds': 1e12,
                                                      'gpu_hours': 4 * screens * c_full}))
    hooks = tb.MemoryHooks(env, sim.memory_facts, base_files={'patches.txt': ''})
    s = tb.SearchSession(task=env, budget=env.search_budget(), runner=runner,
                         judge=tb.Verifier(0.03, memory_cap_gib=sim.hw['mem']), run_id='toy-b4', arm='b4',
                         start_config=ref_cfg, start_receipt=ref, reference_receipt=ref, hooks=hooks, clock=clock,
                         stop_after_same_failures=10 ** 6)
    eng = tb.Engine(s, full_steps=tb.FULL, short_steps=tb.SHORT, probe_steps=tb.PROBE, job_gpu_hours=None,
                    multi_fidelity=True)
    if manifest is not None:
        s.task.raw['manifest'] = manifest
    seen = []

    def wrapped(role, system, prompt):
        seen.append((role, json.loads(prompt)))
        return script(role, system, json.loads(prompt))

    class Priced(ScriptedBackend):
        def complete(self, system, prompt, **kwargs):
            res = super().complete(system, prompt, **{k: v for k, v in kwargs.items()
                                                      if k in ('schema', 'call_dir', 'role', 'expect_json')})
            cap = kwargs.get('max_usd')
            return dataclasses.replace(res, llm_usd=0.5 if cap is None else min(0.5, float(cap)))
    prop = LLMProposer(Priced(wrapped, tokens_per_call=2000), hooks, search_calibrates=True,
                       features=BComp.proposer_features(()))
    st = BComp(prop, facts=sim.memory_facts, seed=1, packs=tb.toy_packs(task, 2), env=tb.toy_env(sim), **kw)
    trace = eng.run(st)
    return SimpleNamespace(st=st, s=s, trace=trace, seen=seen)


@pytest.mark.parametrize('off', [(), ALL_OFF])
def test_a_scripted_twenty_screen_run_of_the_trees_finalist_with_v4_on_and_off(off):
    from test_bcomp import _toy_script
    r = _bcomp_toy(_toy_script(), 20, v4_off=off)
    trace, st, s = r.trace, r.st, r.s
    d = st.describe()['bcomp']
    fid = Counter(x.get('fidelity') for x in s.screens)
    assert trace.stop_reason.startswith('budget') and s.ledger.used['screens'] == 20 == fid['full']
    assert st._llm_used() <= 20 and s.ledger.used['llm_usd'] <= 16.0 + 1e-9
    v4 = d['finalist']['v4']
    assert v4['applies'] and v4['parts_off'] == sorted(off)
    assert s.patch_lines_own is (not off)                         # (F4 set on the session at the start, unless off)
    if off:
        assert not any(v for k, v in v4['counts'].items()) and not v4['stack']['tests']
    json.dumps(d, default=str)
