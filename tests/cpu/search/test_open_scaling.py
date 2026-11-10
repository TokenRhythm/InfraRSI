"""F1 of the open wave (FABLE_OPEN_DESIGN_20261102 sections 1 and 6; tree pc_code_o1 - bcomp v3):
the per-arm caps as per-screen rates of the screen budget B, B counted at least ten. At ten screens or
fewer every cap is today's constant and nothing is set (the v3 suites run unchanged); at twenty the design's table - 20
model calls / 16 USD and the asks they bound, 80 history rows, 8 code-proof probes, 4 contrarian / repair / revise, caps given or changed before the start are kept; the registered strategy reads B from its
session; a scripted 20-screen run of the tree's finalist completes twenty full screens with the code track open, and
bcomp's report keeps every one of them (v3's at ten screens); the open tree differs from its v3 tree only in the rate
files (and bcomp's report)."""
import hashlib
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from finalist_helpers import CodeProposer, fin
from infrarsi.search import STRATEGIES
from infrarsi.search import explore as X
from infrarsi.search.explore import ExploreMixin
from infrarsi.search.finalist_base import (CALLS_RATE, CODE_PROBES_MAX, CODE_PROBES_RATE, HISTORY_RATE, RATE_SCREENS,
                                             USD_RATE, per_screen, rate_screens)
from infrarsi.search.llmopt import CALLS_MAX, HISTORY_MAX, USD_MAX
from infrarsi.search.toytrain import mem_tasks
from test_explore import ExB, MemEngine, mem_of, table_gain

from infrarsi.search.bcomp import BComp

TREE = Path(__file__).resolve().parents[3]
NAME = 'bcomp'
TODAY = {'calls': 10, 'usd': 8.0, 'asks': 10, 'history_rows': 40, 'code_probes': 4}
TABLE = {'calls': 20, 'usd': 16.0, 'asks': 20, 'history_rows': 80, 'code_probes': 8}   # the design's section 1 at 20
V3_TREE = {'pc_code_o1': 'pc_code_n1v3'}
RATE_FILES = {'infrarsi/search/finalist_base.py', 'infrarsi/search/explore.py',
              'infrarsi/search/bcomp.py'}                                   # (bcomp: its report's per-screen record)


def caps(st) -> dict:
    return {'calls': st.calls_max, 'usd': st.usd_max, 'asks': st.code_asks, 'history_rows': st.history_max,
            'code_probes': st.ladder.code_probes_max}


def ex_caps(st) -> tuple:
    return st.EX_CONTRA_MAX, st.EX_REPAIR_MAX, st.EX_REVISE_MAX


def ex_setup(screens, cls=ExB):
    """test_explore's set-up of a finalist with the shared module (bcomp's bases) on a `screens` budget."""
    eng = MemEngine(table_gain({}), mem_of({}), screens=screens)
    st = cls(CodeProposer(None), facts={'cap_gib': 100.0}, seed=3)
    st._init(eng)
    st._ex_init(eng)
    return st, eng


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- the rates
def test_the_rates_give_todays_constants_at_ten_screens_and_the_designs_table_at_twenty():
    assert RATE_SCREENS == 10
    assert (per_screen(CALLS_RATE, 10), USD_RATE * 10, per_screen(HISTORY_RATE, 10), per_screen(CODE_PROBES_RATE, 10)) == \
        (CALLS_MAX, USD_MAX, HISTORY_MAX, CODE_PROBES_MAX) == (10, 8.0, 40, 4)
    assert (per_screen(CALLS_RATE, 20), USD_RATE * 20, per_screen(HISTORY_RATE, 20), per_screen(CODE_PROBES_RATE, 20)) == \
        (20, 16.0, 80, 8)
    for rate, cap in ((X.EX_CONTRA_RATE, X.EX_CONTRA_MAX), (X.EX_REPAIR_RATE, X.EX_REPAIR_MAX),
                      (X.EX_REVISE_RATE, X.EX_REVISE_MAX)):
        assert per_screen(rate, 10) == cap == 2 and per_screen(rate, 20) == 4
    assert [per_screen(0.2, b) for b in (11, 15, 19, 25, 30)] == [2, 3, 3, 5, 6]          # (rounded down in between)
    # the module constants and the mixin's class caps stay today's (what the v3 suites pin)
    assert (X.EX_CONTRA_MAX, X.EX_STALL, X.EX_RADIUS_MAX, X.EX_REPAIR_MAX, X.EX_REVISE_MAX) == (2, 2, 3, 2, 2)
    assert (ExploreMixin.EX_CONTRA_MAX, ExploreMixin.EX_REPAIR_MAX, ExploreMixin.EX_REVISE_MAX) == (2, 2, 2)
    # B: the session's screens, counted at least ten (none known: ten)
    assert rate_screens(None) == rate_screens(SimpleNamespace(s=SimpleNamespace())) == 10
    assert [rate_screens(SimpleNamespace(s=SimpleNamespace(budget=SimpleNamespace(screens=b)))) for b in (2, 10, 20, 30)] \
        == [10, 10, 20, 30]


@pytest.mark.parametrize('screens', [2, 5, 10])
def test_ten_screens_or_fewer_keep_todays_caps_and_set_nothing(screens):
    st, eng = fin(None, screens=screens)
    st._init(eng)
    assert caps(st) == TODAY and st.fb_rate_screens == 10
    assert 'history_max' not in st.__dict__ and st.ladder.pool_size == CODE_PROBES_MAX
    assert st.finalist_view()['caps'] == {'rate_screens': 10, **TODAY}
    ex, _ = ex_setup(screens)
    assert ex_caps(ex) == (2, 2, 2) and not [k for k in ex.__dict__ if k.startswith('EX_')]
    assert ex._ex_view()['limits'] == {'contrarian': 2, 'repair': 2, 'stall': 2, 'radius_max': 3, 'revise': 2}


def test_twenty_screens_give_the_designs_table():
    st, eng = fin(None, screens=20)
    assert caps(st) == TODAY                                                 # (built: no budget known yet)
    st._init(eng)
    assert caps(st) == TABLE and st.fb_rate_screens == 20
    assert st.finalist_view()['caps'] == {'rate_screens': 20, **TABLE}
    assert st.finalist_view()['probe_accounting']['code_probes_max'] == 8 and st.ladder.code_probes_left() == 8
    assert st.ladder.describe(eng)['code_pool']['max'] == 8
    ex, _ = ex_setup(20)
    assert ex_caps(ex) == (4, 4, 4)
    assert (ExploreMixin.EX_CONTRA_MAX, ExploreMixin.EX_REPAIR_MAX, ExploreMixin.EX_REVISE_MAX) == (2, 2, 2)
    assert ex._ex_view()['limits'] == {'contrarian': 4, 'repair': 4, 'stall': 2, 'radius_max': 3, 'revise': 4}
    st15, eng15 = fin(None, screens=15)                                      # (a budget in between: the rates)
    st15._init(eng15)
    assert caps(st15) == {'calls': 15, 'usd': pytest.approx(12.0), 'asks': 15, 'history_rows': 60, 'code_probes': 6}


def test_caps_given_or_changed_before_the_start_are_kept_and_a_second_start_changes_nothing():
    st, eng = fin(None, screens=20, calls_max=7, usd_max=5.0)
    st.ladder.code_probes_max = 0                                            # (f1.0's accounting: the pool off)
    st._init(eng)
    assert (st.calls_max, st.usd_max, st.code_asks, st.history_max) == (7, 5.0, 7, 80)
    assert st.ladder.code_probes_max == 0 and not st.ladder.pool and st.ladder.pool_size == 8
    st.ladder.pool = True                                                    # (switched on again: the arm's pool size)
    assert st.ladder.code_probes_max == 8
    st._fb_rates(eng)                                                        # a second start moves nothing
    assert (st.calls_max, st.usd_max, st.history_max, st.ladder.code_probes_max) == (7, 5.0, 80, 8)

    class One(ExB):
        EX_CONTRA_MAX = 1                                                    # (a class's own cap)
    ex, eng2 = ex_setup(20, One)
    assert ex_caps(ex) == (1, 4, 4)
    ex._ex_rates(eng2)
    assert ex_caps(ex) == (1, 4, 4)


def test_the_history_rows_follow_the_budget():
    """A finalist's measured history - every patch call's `measured_history` and every configuration call's history -
    shows up to its cap: 40 rows at ten screens, 80 at twenty."""
    for screens, cap in ((10, 40), (20, 80)):
        st, eng = fin(None, screens=screens)
        st._init(eng)
        for i in range(100):
            st.runs[f'x{i}'] = {'cfg': dict(st.ref), 'kind': 'ok', 'status': 'screened', 'y': 0.001 * i, 'fidelity': 'full'}
        rows = st._history(eng)
        if getattr(st, 'bounded_asks', False):           # (v4.1: the rows beyond the cap folded into a last summary row)
            assert rows[-1]['folded']['rows'] == 100 - cap
            rows = rows[:-1]
        assert len(rows) == cap and rows[0]['outcome'] == f'x{math.exp(0.099):.3f} the reference speed'


# ---------------------------------------------------------------- the registered strategy
def _registered_script():
    n = {'p': 0}

    def script(role, system, prompt):
        if role == 'prior':
            return {'levels': [{'knob': 'fused_loss', 'value': True, 'gain_pct': 2, 'sd_pct': 2,
                                'memory': {'direction': 'frees', 'gib': 3}}],
                    'interactions': [], 'configs': [{'name': 'a', 'change': {'fused_loss': True}, 'gain_pct': 2}],
                    'notes': ''}
        if role == 'code_review':
            return {'verdict': 'patch', 'summary': 'x', 'findings': []}
        if role == 'configs':
            return {'configs': [{'change': {'fused_optimizer': True}, 'gain_pct': 2, 'why': 'x'}], 'notes': ''}
        if role == 'revise':
            return {'levels': [], 'notes': ''}
        n['p'] += 1
        k = n['p']
        return {'hypotheses': [{'mechanism': f'patch r{k}', 'change': {}, 'patch': {'patches.txt': f'r{k}\nproof:r{k}\n'},
                                'predicted_gain_pct': 3, 'site': f'r{k}'}]}
    return script


def run_registered(tmp_path, screens):
    """The tree's finalist as the CLI runs it (STRATEGIES -> run_engine) on the memory toy with a scripted model."""
    from infrarsi.agent.backends.fake import ScriptedBackend
    from infrarsi.search import Context
    from infrarsi.search import toybench as tb
    sim = tb.make_sim(mem_tasks()[1])
    runner = tb.ToyRunner(sim, tb.VirtualClock())
    ref_cfg = sim.reference()
    probe_task = tb.EnvTask.from_dict(tb.toy_task_spec(sim, {'screens': 1, 'llm_tokens': 0, 'wall_seconds': 1e12,
                                                             'gpu_hours': 1e9}))
    ref = runner.run(probe_task, ref_cfg, None, 'reference', 'toy', candidate_id='c0', steps=tb.FULL)
    env = tb.EnvTask.from_dict(tb.toy_task_spec(sim, {'screens': screens, 'llm_tokens': 10 ** 7, 'wall_seconds': 1e12,
                                                      'gpu_hours': 1e9}))
    ctx = Context(task=env, budget=env.search_budget(), start_config=ref_cfg, runner=runner,
                  judge=tb.Verifier(0.03, memory_cap_gib=sim.hw['mem']), run_id=f'toy-o{screens}', out_dir=tmp_path,
                  hooks=tb.MemoryHooks(env, sim.memory_facts, base_files={'patches.txt': ''}), start_receipt=ref,
                  reference_receipt=ref, driver=ScriptedBackend(_registered_script(), tokens_per_call=2000), seed=1,
                  options={'full_steps': tb.FULL, 'short_steps': tb.SHORT, 'probe_steps': tb.PROBE, 'job_gpu_hours': None})
    trace = STRATEGIES[NAME](ctx)
    return trace, ctx.sessions[-1]


@pytest.mark.parametrize('screens,expect', [(10, TODAY), (20, TABLE)])
def test_the_registered_strategy_reads_the_screen_budget_from_its_session(tmp_path, screens, expect):
    trace, s = run_registered(tmp_path, screens)
    view = trace.extra['strategy'][NAME]
    assert s.ledger.used['screens'] == screens and trace.stop_reason.startswith('budget')
    assert view['finalist']['caps'] == {'rate_screens': max(10, screens), **expect}
    assert (view['llm_calls_max'], view['llm_usd_max']) == (expect['calls'], expect['usd'])
    assert view['llm_calls'] <= expect['calls'] and s.ledger.used['llm_usd'] <= expect['usd'] + 1e-9
    assert view['finalist']['asks'] and any(a['sent'] for a in view['finalist']['asks'])   # (the code track asked)


# ---------------------------------------------------------------- a scripted 20-screen run of the tree's finalist
@pytest.mark.skipif(BComp is None, reason='bcomp is the other open tree')
def test_a_scripted_twenty_screen_bcomp_run_completes_with_the_code_track_open():
    """bcomp v3 end to end on the memory toy (test_bcomp's priced scripted model, 0.5 USD a call, another task's packs) at
    twenty screens: twenty full screens, no short run; the table's caps; more model calls than ten-screen bcomp's ten,
    within the twenty and the sixteen dollars (every call capped by it); the knob side's five screens then fifteen code
    screens; the code track asks, proves its patches on the pool's eight probes first and runs them in full."""
    from infrarsi.search import toybench as tb
    from test_bcomp import _toy_script, bcomp_llm_toy
    task = mem_tasks()[1]
    r = bcomp_llm_toy(_toy_script(), screens=20, usd_per_call=0.5, packs=tb.toy_packs(task, 2), task=task)
    st, s, be, seen = r.st, r.s, r.be, r.seen
    d = st.describe()['bcomp']
    fid = Counter(x.get('fidelity') for x in s.screens)
    assert r.trace.stop_reason.startswith('budget') and s.ledger.used['screens'] == 20 == fid['full'] and not fid['short']
    assert caps(st) == TABLE and ex_caps(st) == (4, 4, 4) and d['bcomp']['explore']['limits']['contrarian'] == 4
    assert st._llm_used() == be.attempts_made == d['llm_calls'] and CALLS_MAX < d['llm_calls'] <= 20
    assert s.ledger.used['llm_usd'] <= 16.0 + 1e-9 and all(k['max_usd'] <= 16.0 + 1e-9 for k in be.kw)
    assert max(k['max_usd'] for k in be.kw) > USD_MAX                       # (the calls see the twenty-screen dollars)
    assert st.bc_knob_screens == 5 and d['finalist']['protected_screens'] == list(range(6, 21))
    sent = [a for a in st.fb_asks if a['sent']]
    assert len(sent) > 6 and all(len(a['queued']) <= 3 for a in st.fb_asks)  # (ten-screen bcomp's toy run: six asks)
    assert len(st.ladder.code_log) == 8 and st.ladder.code_share_log          # the pool's eight proofs first, then the share
    proven = [p for p in st.patches.values() if p.get('status') == 'proven']
    with_patch = [x for x in s.screens if x.get('fidelity') == 'full'
                  and any(str(k).startswith('patch:') and v for k, v in (st.runs.get(x.get('candidate_id')) or {})
                          .get('cfg', {}).items())]
    assert proven and len(with_patch) >= 10                                  # patches proven, and run in full
    hyp = [p for role, p, _ in seen if role == 'hypotheses']
    assert all(len(p['measured_history']['rows']) <= 80 for p in hyp if 'measured_history' in p)
    view = d['bcomp']                                                        # the report: every screen of the twenty
    assert len(view['screens']) == 20 == len(st.bc_screens) and view['schedule'].startswith('code track open: screens 6-20')
    assert view['steps'] == st.bc_steps[-120:]


@pytest.mark.skipif(BComp is None, reason='bcomp is the other open tree')
@pytest.mark.parametrize('screens', [5, 6, 10])
def test_bcomps_report_at_ten_screens_or_fewer_is_v3s(screens):
    """At ten screens or fewer bcomp's report holds what v3's held: the per-screen record (every full screen; v3 kept the
    last twelve), the last sixty steps, the schedule's words - v3's 'screens 6-10' whatever the budget (OPEN_F1_r1 P2)."""
    from infrarsi.search import toybench as tb
    from test_bcomp import _toy_script, bcomp_llm_toy
    task = mem_tasks()[1]
    r = bcomp_llm_toy(_toy_script(), screens=screens, usd_per_call=0.5, packs=tb.toy_packs(task, 2), task=task)
    view = r.st.describe()['bcomp']['bcomp']
    assert view['screens'] == r.st.bc_screens[-12:] and len(view['screens']) == screens
    assert view['steps'] == r.st.bc_steps[-60:]
    assert view['schedule'] == "code track open: screens 6-10 are the code track's (v1)"
    r.st._can_code = lambda: False                                           # (the knob-only branch's words too)
    assert r.st.bcomp_view()['schedule'] == ("knob-only: screens 6-10 are completions, contrarian moves, escapes, the "
                                             "ledger's best, then the tail and the draw")






# ---------------------------------------------------------------- what the open tree changes
def test_the_open_tree_differs_from_its_v3_tree_only_in_the_rate_files():
    v3 = TREE.parent / V3_TREE.get(TREE.name, '')
    if TREE.name not in V3_TREE or not v3.is_dir():
        pytest.skip('not an open tree next to its v3 tree')
    changed = set()
    for p in sorted((TREE / 'infrarsi').rglob('*.py')):
        rel = p.relative_to(TREE).as_posix()
        b = v3 / rel
        assert b.is_file(), rel
        if sha(p) != sha(b):
            changed.add(rel)
    assert {rel for rel in (p.relative_to(v3).as_posix() for p in (v3 / 'infrarsi').rglob('*.py'))
            if not (TREE / rel).is_file()} == set()
    assert changed <= RATE_FILES and 'infrarsi/search/llmopt.py' not in changed
    assert {'infrarsi/search/finalist_base.py', 'infrarsi/search/explore.py'} <= changed
    assert 'infrarsi/search/bcomp.py' in changed
