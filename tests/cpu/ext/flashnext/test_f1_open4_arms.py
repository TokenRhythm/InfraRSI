"""The v4 extras on the F1 task (FABLE_FN4_RESUME_20261103; FABLE_V4_BUDGET_20261102 section 1):
F-OPEN4-BCOMP runs exactly as F-OPEN-BCOMP (the CLI's run_f_is2 with bcomp, ours' cross-job
experience, C_F1 with --code, the ordinary locked pair in the v3 counterpart's parity order) except for their tree
(pc_code_o2, through fn_launch) and the v4 caps: F-OPEN-AR's Q5 entry outside, task.V4_MANIFEST inside
(raw['manifest'], where the v4 trees' finalist_base.spec_budget reads a spec's caps). Labelled extras, outside the
window's counts (S_UC, the drift, every pair label). Every other arm's contract is unchanged. CPU only, no network, no
node, no real model.

The last test runs the registered strategy itself, so it runs only where the package registers it: the main repo's
tests under scripts/fn_overlay_plugin.py with FN_TREE = qa_trials/pc_code_o2 (F-OPEN4-BCOMP), and, for the v3
counterpart's caps, pc_code_o1; elsewhere it is skipped."""
import json

import pytest

import x6_helpers as H
from infrarsi import search
from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.agent.budget import VirtualClock
from infrarsi.ext.flashnext import arms
from infrarsi.ext.flashnext import cli
from infrarsi.ext.flashnext import code_space as C
from infrarsi.ext.flashnext import switch_space as S
from infrarsi.ext.flashnext import task as T
from infrarsi.harness import signing
from test_flashnext_code_arm import CodeFakeRunner, overlay

V4 = ('F-OPEN4-BCOMP',)
V3_OF = {'F-OPEN4-BCOMP': 'F-OPEN-BCOMP'}
STRATEGY = {'F-OPEN4-BCOMP': 'bcomp', 'F-OPEN-BCOMP': 'bcomp'}
CAPS = {'search': {'budget': {'llm_calls': 60, 'report_usd': 16},
                   'caps': {'fix_calls_per_screen': 2, 'fix_calls': 20, 'patches_per_ask': 4, 'code_probes': 12}},
        'llm': {'max_usd_per_call': 8, 'timeout_s': 1200}}
# what a v4 tree's spec_budget makes of the v4 task (llm_usd: the session budget's, F-OPEN-AR's 40)
SPEC = {'usd': 40.0, 'calls': 60, 'patches_per_ask': 4, 'code_probes': 12, 'fix_calls': 20, 'fix_calls_per_screen': 2,
        'call_usd': 8.0, 'call_wall_s': 1200.0}
AR_BUDGET = (20, 3_000_000, 40.0, 21600.0, 320.0)
# every other arm, as before the v4 extras: (screens, llm_tokens, llm_usd, wall_seconds, gpu_hours), index, pair order
OLD = {'F-PROMPTS': ((3, 200_000, 10.0, 7200.0, 48.0), 0), 'F-OPTICO': ((12, 1_500_000, 40.0, 18000.0, 192.0), 1),
       'F-A0': ((12, 1_500_000, 40.0, 18000.0, 192.0), 2), 'F-RND': ((12, 0, 0.0, 18000.0, 192.0), 3),
       'F-A0-CODE': ((12, 1_500_000, 40.0, 18000.0, 192.0), 4),
       'F-OPEN-BCOMP': (AR_BUDGET, 6), 'F-OPEN-AR': (AR_BUDGET, 7), 'F-OPEN-RB': (AR_BUDGET, 8),
       'F-OPTICO-20': (AR_BUDGET, 9), 'F-RND-20-s42': ((20, 0, 0.0, 21600.0, 320.0), 10),
       'F-RND-20-s43': ((20, 0, 0.0, 21600.0, 320.0), 11)}
SMATCH = S.registered_points()['S_match']
OTHER = S.sample_order(seed=7, n=3)[1]


@pytest.fixture
def uc(tmp_path):
    return H.uc_reference_receipt(tmp_path / 'uc', clock=VirtualClock())


def _entry(arm: str) -> tuple:
    a = T.ARMS[arm]
    return a['screens'], a['llm_tokens'], a['llm_usd'], a['wall_seconds'], a['gpu_hours']


# ---------------------------------------------------------------- the contract
def test_v4_extras_take_f_open_ar_budget_exactly():
    for arm in V4:
        assert _entry(arm) == _entry('F-OPEN-AR') == AR_BUDGET and T.ARMS[arm]['llm'] is True
        assert T.budget_for(arm) == T.budget_for('F-OPEN-AR')
        for code in (False, True):
            assert T.make_task(arm, code=code).search_budget().as_dict() == \
                T.make_task('F-OPEN-AR', code=code).search_budget().as_dict()


def test_the_v4_caps_ride_on_the_v4_tasks_only():
    assert T.V4_ARMS == V4 and T.V4_MANIFEST == CAPS
    for arm in V4:
        for code in (False, True):
            t = T.make_task(arm, code=code)
            assert t.raw['manifest'] == CAPS and t.arm == arm
            assert t.action_space == ('knobs+patch' if code else 'knobs')        # --code opens C_F1 as for F-OPEN-*
        coded = T.make_task(arm, code=True)
        assert tuple(coded.mutable_globs) == C.MUTABLE_PATHS and C.is_code_task(coded)
        assert T.make_task(arm).raw['manifest'] is not T.V4_MANIFEST           # a copy: the table is not shared
    for arm in [a for a in T.ARMS if a not in V4 and a not in T.BUD_ARMS] + list(T.REFERENCES):   # (BUD_ARMS:
        #                                                    test_f1_bud_arms)
        assert 'manifest' not in T.make_task(arm).raw, arm
        if arm in T.OPEN_ARMS:
            assert 'manifest' not in T.make_task(arm, code=True).raw, arm


def test_every_other_arm_is_unchanged():
    assert T.ARM_INDEX == {'F-PROMPTS': 0, 'F-OPTICO': 1, 'F-A0': 2, 'F-RND': 3} and T.CODE_ARM_INDEX == {'F-A0-CODE': 4}
    assert set(T.ARMS) == set(OLD) | set(V4) | set(T.BUD_ARMS)
    for arm, (budget, idx) in OLD.items():
        assert _entry(arm) == budget and arms.ALL_ARM_INDEX[arm] == idx, arm
    assert arms.pair_order('F-OPEN-BCOMP') == ['uc', 'candidate']
    assert arms.STRATEGY_ARM == {'bcomp': 'F-OPEN-BCOMP', 'autoresearch': 'F-OPEN-AR',
                                 'rebench': 'F-OPEN-RB'}                     # run_f_is2's default arm per strategy
    assert arms.OURS_STRATEGIES == ('bcomp',)


def test_v4_extras_are_open_arms_with_their_v3_counterparts_pair_order():
    assert T.OPEN_ARM_INDEX['F-OPEN4-BCOMP'] == 12
    assert len(set(arms.ALL_ARM_INDEX.values())) == len(arms.ALL_ARM_INDEX)
    for arm in V4:
        assert arm in T.OPEN_ARMS and arm not in T.ARM_INDEX and arm not in T.CODE_ARMS
        assert arms.pair_order(arm) == arms.pair_order(V3_OF[arm]) and arms.V4_STRATEGY[arm] == STRATEGY[arm]


# ---------------------------------------------------------------- the CLI dispatch
def test_cli_runs_the_v4_extras_as_their_v3_counterparts(tmp_path, uc, monkeypatch, capsys):
    """`arm F-OPEN4-*` -> run_f_is2 with the v3 counterpart's strategy, ours' experience ('auto': the arm's
    INFRARSI_HOME), --code and no budget override: the arm's Q5 entry stands."""
    seen = []

    def fake_is2(strategy, runner, **kw):
        seen.append((strategy, kw['arm'], kw['code'], kw['experience'],
                     sorted(k for k in kw if k in ('screens', 'llm_tokens', 'llm_usd', 'wall_seconds', 'gpu_hours'))))
        return {'arm': kw['arm'], 'stop_reason': 'stub'}
    monkeypatch.setattr(arms, 'run_f_is2', fake_is2)
    monkeypatch.setattr(cli, '_runner', lambda args, out, reference=None, cap_s=None: object())
    path = tmp_path / 'uc_ref.json'
    path.write_text(json.dumps(uc), encoding='utf-8')
    for arm in V4 + ('F-OPEN-BCOMP',):
        assert cli.main(['arm', arm, '--run-id', 'r', '--uc-ref', str(path), '--nodes', ','.join(H.NODES), '--out',
                         str(tmp_path / arm), '--code']) == 0
    capsys.readouterr()
    assert seen == [('bcomp', 'F-OPEN4-BCOMP', True, 'auto', []),
                    ('bcomp', 'F-OPEN-BCOMP', True, 'auto', [])]


# ---------------------------------------------------------------- run_f_is2 and the locked pair
def _probe_strategy(seen: dict, ov: dict):
    """test_f1_open_arms._code_strategy (a switch point, then a C_F1 overlay on it) recording what a v4 tree reads off
    the session: its task's raw manifest and the budget's dollars (finalist_base.spec_budget)."""
    def run(ctx):
        s = ctx.session(ctx.options.get('arm') or 'scripted', layers=tuple(ctx.hooks.layers))
        seen.update(manifest=s.task.raw.get('manifest'), usd=s.budget.llm_usd, layers=tuple(ctx.hooks.layers))
        fb = s.propose_config(config=SMATCH, base='c0', origin='scripted')
        s.screen(fb['candidate_id'])
        pb = s.propose_patch(files=ov, base=fb['candidate_id'], origin='scripted')
        assert 'candidate_id' in pb, pb
        s.screen(pb['candidate_id'])
        return s.finalize(stop_reason='scripted_done', extra={'strategy': 'scripted'})
    return run


def _run(tmp_path, uc, arm, clock, runner=None):
    """One arm through run_f_is2 (the probe strategy, C_F1 open) and its locked pair: (trace, locked, seen, runner)."""
    seen = {}
    r = runner or CodeFakeRunner(tmp_path / f'run-{arm}', clock=clock)
    search.STRATEGIES['fn_probe'] = _probe_strategy(seen, overlay())
    try:
        tr = arms.run_f_is2('fn_probe', r, run_id=f'rid-{arm}', out_dir=tmp_path / arm, uc_ref=uc, arm=arm, code=True)
    finally:
        search.STRATEGIES.pop('fn_probe', None)
    lk = arms.run_locked(r, arm=arm, run_id=f'rid-{arm}', out_dir=tmp_path / arm, trace=tmp_path / arm / 'trace.json',
                         uc_ref=uc)
    return tr, lk, seen, r


def test_run_f_is2_carries_the_v4_caps_to_the_session_and_locks_as_its_v3_counterpart(tmp_path, uc):
    clock = VirtualClock()
    for arm in V4:
        tr, lk, seen, r = _run(tmp_path, uc, arm, clock)
        assert seen == {'manifest': CAPS, 'usd': 40.0, 'layers': ('L0', 'L1')}
        assert tr['arm'] == arm and tr['budget'] == T.make_task('F-OPEN-AR').search_budget().as_dict()
        assert signing.verify(json.loads((tmp_path / arm / 'trace.json').read_text(encoding='utf-8')))
        assert tr['extra']['code_space']['selected_overlay'] == C.record(overlay())
        assert lk['status'] == 'ok' and lk['order'] == arms.pair_order(V3_OF[arm]) and lk['code_overlay_bound'] is True
        assert lk['arm_index'] == T.OPEN_ARM_INDEX[arm] and 'D-F1-7' in lk['deviations'] and signing.verify(lk)
        by = {x['attempt_id']: x for x in arms.load_receipts([tmp_path / arm, r.out_dir]) + [uc]}
        assert arms.locked_receipt_problems(arm, lk, by, trace=tmp_path / arm / 'trace.json', uc_ref=uc) == []
        assert arms.trace_problems(arm, tr, by) == []
    _, _, seen3, _ = _run(tmp_path, uc, 'F-OPEN-BCOMP', clock)                    # a v3 arm: no manifest, as before
    assert seen3['manifest'] is None and seen3['usd'] == 40.0


def test_v4_extras_stay_out_of_the_window_counts(tmp_path, uc):
    """Adding the v4 extra to a summary changes no number of the v3 arms: its U+C half is excluded from S_UC (listed
    with why), its row is marked, it is in no label and not in the drift's max / min."""
    clock = VirtualClock()
    runs, dirs = {}, {}
    for arm in ('F-OPEN-BCOMP', 'F-OPEN4-BCOMP'):
        tr, lk, _, r = _run(tmp_path, uc, arm, clock)
        runs[arm], dirs[arm] = {'trace': tr, 'locked': lk}, [tmp_path / arm, r.out_dir]
    v3 = {a: runs[a] for a in ('F-OPEN-BCOMP',)}
    rec_v3 = arms.load_receipts(dirs['F-OPEN-BCOMP']) + [uc]
    rec_all = rec_v3 + arms.load_receipts(dirs['F-OPEN4-BCOMP'])
    m3, m_all = arms.margin(rec_v3), arms.margin(rec_all)
    v4_half = runs['F-OPEN4-BCOMP']['locked']['uc_half']['attempt_id']
    assert arms._margin_key(m3) == arms._margin_key(m_all) and m_all['n'] == 2
    assert [(x['attempt_id'], x['why']) for x in m_all['excluded']] == [(v4_half, 'a v4 extra (post hoc): outside S_UC')]
    assert v4_half not in [x['attempt_id'] for x in m_all['S_UC']] and not m_all['anomalies']
    s3 = arms.summarize(uc_ref=uc, ht=None, arms=v3, receipts=rec_v3, allow_nonformal=True)
    s = arms.summarize(uc_ref=uc, ht=None, arms={**v3, 'F-OPEN4-BCOMP': runs['F-OPEN4-BCOMP']}, receipts=rec_all,
                       allow_nonformal=True)
    assert s['labels'] == s3['labels'] == {} and s['margin'] == s3['margin'] and 'v4_extras' not in s3
    assert s['drift']['max_over_min_uc_halves'] == s3['drift']['max_over_min_uc_halves'] is None  # one retained wave arm
    assert all(s['rows'][a] == s3['rows'][a] for a in v3)                     # the v3 rows: unchanged, unmarked
    assert all('extra_arm' not in s3['rows'][a] for a in v3)
    row = s['rows']['F-OPEN4-BCOMP']
    assert row['extra_arm'] == T.V4_LABEL == 'fixed after the diagnosis, out-of-sample on Flash-Next'
    assert row['arm_index'] == 12 and row['step_time_locked'] == runs['F-OPEN4-BCOMP']['locked']['t_locked']
    assert s['v4_extras']['arms'] == ['F-OPEN4-BCOMP'] and s['v4_extras']['label'] == T.V4_LABEL
    assert not [x for x in s['nonformal_reasons'] if x.startswith('F-OPEN4-BCOMP')], s['nonformal_reasons']


# ---------------------------------------------------------------- the registered strategy (a v4 / v3 tree only)
def _role_script():
    """fn_phasec_check's role script: the prior and the configuration calls answer with S_F1 points, the code review
    with no_patch, every patch call with no hypothesis."""
    lv = [{'knob': k, 'value': v, 'gain_pct': 8, 'sd_pct': 3, 'memory': {'direction': 'none', 'gib': 0}}
          for k, v in SMATCH.items() if v is not None][:4]

    def script(role, system, prompt):
        if role == 'prior':
            return {'levels': lv, 'interactions': [], 'configs': [{'name': 'a', 'change': dict(SMATCH), 'gain_pct': 20}],
                    'notes': ''}
        if role == 'code_review':
            return {'verdict': 'no_patch', 'summary': 'x', 'findings': []}
        if role == 'configs':
            return {'configs': [{'change': dict(OTHER), 'gain_pct': 5, 'why': 'x'}], 'notes': ''}
        if role == 'revise':
            return {'levels': [], 'notes': ''}
        return {'hypotheses': []}
    return script


def _finalist_base(v4: bool):
    """The package's finalist base when it is the v4 one (`v4`: spec_budget, V4_PARTS) or the v3 one, else None."""
    try:
        from infrarsi.search import finalist_base as fb
    except ImportError:
        return None
    return fb if hasattr(fb, 'spec_budget') == v4 else None


@pytest.mark.parametrize('arm', V4 + ('F-OPEN-BCOMP',))
def test_the_tree_strategy_runs_the_arm_with_its_caps(tmp_path, uc, monkeypatch, capsys, arm):
    """The real CLI path (`arm ARM --code`, 20 screens) with the tree's registered strategy on the fake runner and a
    scripted model that has the CLI backend's per-call caps: a v4 extra on its v4 tree has every v4 part on
    (`_v4(part, engine)`: the code track open, more than ten screens) and the v4 caps - 60 calls, 40 USD, 4 patches per
    ask, a pool of 12 code probes, fix calls 20 / 2 per screen, 8 USD / 1,200 s per call on the backend; a v3 arm on its
    v3 tree keeps v3's rates (20 calls, 16 USD, 3 patches per ask) and the backend's own per-call caps."""
    v4 = arm in V4
    fb = _finalist_base(v4)
    if fb is None or STRATEGY[arm] not in search.STRATEGIES:
        pytest.skip(f'{STRATEGY[arm]} on the {"v4" if v4 else "v3"} finalist base is registered in its tree only '
                    '(fn_overlay_plugin.py, FN_TREE)')
    import infrarsi.agent.arms as agent_arms
    home = tmp_path / 'home'
    (home / 'experience' / 'runs').mkdir(parents=True)
    monkeypatch.setenv('INFRARSI_HOME', str(home))                     # ours read the (empty) home's plan
    backend = ScriptedBackend(_role_script(), tokens_per_call=2000)
    backend.timeout_seconds, backend.max_usd_per_call = 900.0, 5.0       # ClaudeCLIBackend's defaults
    monkeypatch.setattr(agent_arms, 'strong_driver', lambda: backend)
    clock = VirtualClock()
    r = CodeFakeRunner(tmp_path / 'run', clock=clock)
    monkeypatch.setattr(cli, '_runner', lambda args, out, reference=None, cap_s=None: r)
    got = {}
    orig = fb.FinalistBase._init

    def spy(self, engine):
        orig(self, engine)
        got.update(strategy=self, engine=engine)
    monkeypatch.setattr(fb.FinalistBase, '_init', spy)
    path = tmp_path / 'uc_ref.json'
    path.write_text(json.dumps(uc), encoding='utf-8')
    out = tmp_path / 'arm'
    assert cli.main(['arm', arm, '--run-id', 'rid', '--uc-ref', str(path), '--nodes', ','.join(H.NODES), '--out', str(out),
                     '--code']) == 0
    capsys.readouterr()
    tr = json.loads((out / 'trace.json').read_text(encoding='utf-8'))
    st, eng = got['strategy'], got['engine']
    assert tr['arm'] == arm and tr['budget']['screens'] == 20 and eng.s.task.arm == arm
    fin = tr['extra']['strategy'][STRATEGY[arm]]['finalist']
    if v4:
        assert {p: st._v4(p, eng) for p in fb.V4_PARTS} == {p: True for p in fb.V4_PARTS}
        assert st.fb_spec_caps == SPEC and (st.calls_max, st.usd_max, st.patches_per_ask) == (60, 40.0, 4)
        assert st.ladder.pool_size == 12 and (st.v4_fix_max, st.v4_fix_per_screen) == (20, 2)
        assert (backend.timeout_seconds, backend.max_usd_per_call) == (1200.0, 8.0)
        assert fin['v4']['applies'] is True and fin['v4']['parts_off'] == [] and fin['v4']['spec_caps'] == SPEC
        assert (fin['caps']['calls'], fin['caps']['usd']) == (60, 40.0)
    else:
        assert not hasattr(st, 'v4_on') and 'v4' not in fin
        assert (st.calls_max, st.usd_max, st.patches_per_ask) == (20, 16.0, 3)
        assert (backend.timeout_seconds, backend.max_usd_per_call) == (900.0, 5.0)
    assert cli.main(['locked', arm, '--run-id', 'rid', '--uc-ref', str(path), '--nodes', ','.join(H.NODES), '--out',
                     str(out), '--from', str(out / 'trace.json')]) in (0, 3)
    capsys.readouterr()
    lk = json.loads((out / 'locked.json').read_text(encoding='utf-8'))
    assert lk['arm'] == arm and signing.verify(lk) and (not lk['locked_run'] or lk['order'] == arms.pair_order(arm))
