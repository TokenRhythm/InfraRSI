"""The open-space wave on the F1 task (FABLE_OPEN_DESIGN_20261102 section 6 F8): arms.run_f_is2 runs a registered search
strategy over S_F1 (and, with code=True, C_F1) on the fake four-node runner, every screen a sealed receipt, and its
locked pair is the ordinary arms.run_locked. The switch-only window (F-PROMPTS / F-OPTICO / F-A0 / F-RND) and the
setting-2 arm (F-A0-CODE) are untouched - their tests (test_fnx6_arms.py, test_flashnext_code_arm.py) still pass.
CPU only, no network, no node. A scripted strategy stands in for the open trees' bcomp / autoresearch / rebench,
which are registered in those trees' search packages (the F1 wave runs from a tree copy that carries them)."""
import json
from pathlib import Path

import pytest

import x6_helpers as H
from infrarsi import search
from infrarsi.agent.budget import VirtualClock
from infrarsi.ext.flashnext import arms
from infrarsi.ext.flashnext import code_space as C
from infrarsi.ext.flashnext import switch_space as S
from infrarsi.ext.flashnext import task as T
from infrarsi.ext.flashnext.hooks import FlashNextHooks
from infrarsi.harness import signing
from test_flashnext_code_arm import CodeFakeRunner, overlay

START = S.start_point()
SMATCH = S.registered_points()['S_match']
D_F1 = ['D-F1-1', 'D-F1-2', 'D-F1-3', 'D-F1-4', 'D-F1-5', 'D-F1-6']
REPLAY = 'code/replay_training.py'


@pytest.fixture
def uc(tmp_path):
    return H.uc_reference_receipt(tmp_path / 'uc', clock=VirtualClock())


def _receipts(d: Path):
    return [json.loads(p.read_text()) for p in sorted((d / 'receipts').glob('*.json'))]


def _switch_strategy(points):
    """A scripted engine-style strategy: start from the U+C receipt (c0 not re-screened), screen each switch point in
    turn until the screen budget is spent (as arms.run_f_rnd drives SearchSession)."""
    def run(ctx):
        s = ctx.session(ctx.options.get('arm') or 'scripted')
        if ctx.start_receipt is None:
            s.screen('c0')
        for p in points:
            if s.stop_flag or s.ledger.exhausted():
                break
            fb = s.propose_config(config=p, base='c0', origin='scripted')
            if 'candidate_id' in fb:
                s.screen(fb['candidate_id'])
        stop = s.stop_flag or ('budget:' + s.ledger.exhausted()[0] if s.ledger.exhausted() else 'scripted_done')
        return s.finalize(stop_reason=stop, extra={'strategy': 'scripted', 'seed': ctx.seed,
                                                   'exp_present': ctx.options.get('experience') is not None})
    return run


def _code_strategy(point, ov):
    """A scripted code-arm strategy: a switch point (c1), then a C_F1 overlay on it (c2, 3 % faster on the fake runner)."""
    def run(ctx):
        s = ctx.session(ctx.options.get('arm') or 'scripted', layers=tuple(ctx.hooks.layers))
        fb = s.propose_config(config=point, base='c0', origin='scripted')
        s.screen(fb['candidate_id'])
        pb = s.propose_patch(files=ov, base=fb['candidate_id'], origin='scripted')
        assert 'candidate_id' in pb, pb
        s.screen(pb['candidate_id'])
        return s.finalize(stop_reason='scripted_done', extra={'strategy': 'scripted'})
    return run


# ---------------------------------------------------------------- run_f_is2: a strategy runs 20 screens, then run_locked
def test_run_f_is2_scripted_runs_twenty_screens_sealed_and_locks(tmp_path, uc, monkeypatch):
    monkeypatch.setitem(search.STRATEGIES, 'fn_scripted', _switch_strategy(S.sample_order(seed=42, n=20)))
    clock = VirtualClock()
    r = H.FakeRunner(tmp_path / 'run', clock=clock)
    tr = arms.run_f_is2('fn_scripted', r, run_id='f1_open_bcomp_20261103', out_dir=tmp_path / 'b', uc_ref=uc,
                        arm='F-OPEN-BCOMP')
    assert tr['arm'] == 'F-OPEN-BCOMP' and len(tr['screens']) == 20 and tr['budget']['screens'] == 20
    assert tr['start_step_time'] == uc['metric']['value'] and tr['selected'] != 'c0'
    assert signing.verify(json.loads((tmp_path / 'b' / 'trace.json').read_text(encoding='utf-8')))   # the trace is sealed
    assert 'code_space' not in (tr.get('extra') or {})                                  # switches only (code=False)
    recs = [x for x in _receipts(r.out_dir) if x.get('purpose') == 'screen']
    assert len(recs) == 20 and all(signing.verify(x) and x['valid'] and x['deviations'] == D_F1 for x in recs)
    assert [x['point'] for x in r.launched] == S.sample_order(seed=42, n=20) and START not in [x['point'] for x in r.launched]
    # the locked pair is the ordinary run_locked (F-OPEN-BCOMP index 6 -> U+C half first)
    res = arms.run_locked(r, arm='F-OPEN-BCOMP', run_id='f1_open_bcomp_20261103', out_dir=tmp_path / 'b',
                          trace=tmp_path / 'b' / 'trace.json', uc_ref=uc)
    assert res['order'] == ['uc', 'candidate'] and res['status'] == 'ok' and res['locked_valid'] is True
    assert res['deviations'] == D_F1 and 'code_overlay' not in res and signing.verify(res)
    receipts = arms.load_receipts([tmp_path / 'b', r.out_dir]) + [uc]     # search screens + the locked halves
    by = {x['attempt_id']: x for x in receipts}
    assert arms.locked_receipt_problems('F-OPEN-BCOMP', res, by, trace=tmp_path / 'b' / 'trace.json', uc_ref=uc) == []
    assert res['uc_half']['attempt_id'] in [x['attempt_id'] for x in arms.margin(receipts)['S_UC']]


def test_run_f_is2_code_arm_records_and_locks_the_overlay(tmp_path, uc, monkeypatch):
    ov = overlay()                                                        # a trivial C_F1 edit on replay_training.py
    monkeypatch.setitem(search.STRATEGIES, 'fn_scripted_code', _code_strategy(SMATCH, ov))
    clock = VirtualClock()
    r = CodeFakeRunner(tmp_path / 'run', clock=clock)
    tr = arms.run_f_is2('fn_scripted_code', r, run_id='f1_open_bcomp_20261103', out_dir=tmp_path / 'h', uc_ref=uc,
                        arm='F-OPEN-BCOMP', code=True)
    assert tr['arm'] == 'F-OPEN-BCOMP' and tr['selected_overlay'] == ov
    block = tr['extra']['code_space']
    assert block['sha256'] == C.sha256() and block['selected_overlay'] == C.record(ov)
    assert block['selected_overlay_digest'] == C.record(ov)['digest'] and arms._code_arm('F-OPEN-BCOMP', tr) is True
    code_recs = [x for x in _receipts(r.out_dir) if (x.get('code_overlay') or {}).get('digest')]
    assert code_recs and all('D-F1-7' in x['deviations'] for x in code_recs) and list(r.deviations) == list(H.FakeRunner.deviations)
    res = arms.run_locked(r, arm='F-OPEN-BCOMP', run_id='f1_open_bcomp_20261103', out_dir=tmp_path / 'h',
                          trace=tmp_path / 'h' / 'trace.json', uc_ref=uc)
    assert res['status'] == 'ok' and res['code_overlay'] == C.record(ov) and res['code_overlay_bound'] is True
    assert res['order'] == ['uc', 'candidate'] and 'D-F1-7' in res['deviations']       # F-OPEN-BCOMP index 6 -> uc first
    assert res['candidate_half']['code_overlay_digest'] == C.record(ov)['digest']
    assert res['uc_half']['code_overlay_digest'] is None                               # the U+C half runs no overlay
    receipts = arms.load_receipts([tmp_path / 'h', r.out_dir]) + [uc]
    by = {x['attempt_id']: x for x in receipts}
    assert arms.locked_receipt_problems('F-OPEN-BCOMP', res, by, trace=tmp_path / 'h' / 'trace.json', uc_ref=uc) == []


# ---------------------------------------------------------------- run_f_rnd / run_f_optico carry the open arm name
def test_run_f_rnd_open_seed_and_arm(tmp_path, uc):
    clock = VirtualClock()
    r = H.FakeRunner(tmp_path / 'run', clock=clock)
    tr = arms.run_f_rnd(r, run_id='f1_open_rnd_20261103', out_dir=tmp_path / 'rnd', uc_ref=uc, arm='F-RND-20-s43',
                        seed=43, clock=clock, screens=4)
    assert tr['arm'] == 'F-RND-20-s43' and tr['extra']['seed'] == 43
    assert [x['point'] for x in r.launched] == S.sample_order(seed=43, n=4) and len(tr['screens']) == 4
    res = arms.run_locked(r, arm='F-RND-20-s43', run_id='f1_open_rnd_20261103', out_dir=tmp_path / 'rnd',
                          trace=tmp_path / 'rnd' / 'trace.json', uc_ref=uc)
    assert res['arm'] == 'F-RND-20-s43' and res['status'] in ('ok', 'no_improvement') and signing.verify(res)


# ---------------------------------------------------------------- the ARMS entries and indices (section 6 F8)
def test_open_arm_budgets_and_deviations():
    for a in ('F-OPEN-BCOMP', 'F-OPEN-AR', 'F-OPEN-RB', 'F-OPTICO-20'):
        b = T.make_task(a).budget
        assert (b['screens'], b['wall_seconds'], b['gpu_hours'], b['llm_usd'], b['llm_tokens']) == (20, 21600.0, 320.0, 40.0, 3_000_000)
    for a in ('F-RND-20-s42', 'F-RND-20-s43'):
        b = T.make_task(a).budget
        assert b['screens'] == 20 and b.get('llm_usd') is None and b['llm_tokens'] == 0
    from infrarsi.ext.flashnext import deviations as dv
    assert dv.for_arm('F-OPTICO-20') == D_F1 + ['D-OC-15'] and dv.for_arm('F-OPEN-BCOMP') == D_F1


def test_open_arms_are_outside_the_switch_only_window():
    assert T.ARM_INDEX == {'F-PROMPTS': 0, 'F-OPTICO': 1, 'F-A0': 2, 'F-RND': 3}        # unchanged (setting 1)
    assert T.CODE_ARM_INDEX == {'F-A0-CODE': 4} and T.CODE_ARMS == ('F-A0-CODE',)
    assert set(T.OPEN_ARM_INDEX) == set(T.OPEN_ARMS) and not (set(T.OPEN_ARMS) & (set(T.ARM_INDEX) | set(T.CODE_ARM_INDEX)))
    assert set(arms.ALL_ARM_INDEX) == set(T.ARM_INDEX) | set(T.CODE_ARM_INDEX) | set(T.OPEN_ARM_INDEX)
    assert len({arms.ALL_ARM_INDEX[a] for a in arms.ALL_ARM_INDEX}) == len(arms.ALL_ARM_INDEX)   # distinct indices


def test_make_task_code_flag_opens_the_space_only_for_open_arms():
    plain = T.make_task('F-OPEN-BCOMP')
    assert plain.action_space == 'knobs' and not C.is_code_task(plain)
    coded = T.make_task('F-OPEN-BCOMP', code=True)
    assert coded.action_space == 'knobs+patch' and C.is_code_task(coded)
    assert tuple(coded.mutable_globs) == C.MUTABLE_PATHS and coded.max_patch_lines == C.MAX_PATCH_LINES
    assert T.make_task('F-A0', code=True).action_space == 'knobs'        # the flag never opens a switch-only window arm


# ---------------------------------------------------------------- OPEN_F8 r1 review fixes
def test_run_f_is2_builds_cross_job_experience_for_ours_only(tmp_path, uc, monkeypatch):
    """P1: ours (bcomp) build the F1 cross-job experience plan from INFRARSI_HOME and pass it into the search;
    the baselines' path (experience=None) reads none. The scripted strategy records whether ctx.options['experience']
    was present."""
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'fnhome'))     # an empty home: plan_for reads the shipped seeds
    (tmp_path / 'fnhome' / 'experience' / 'runs').mkdir(parents=True)
    monkeypatch.setitem(search.STRATEGIES, 'fn_scripted', _switch_strategy(S.sample_order(seed=42, n=3)))
    r = H.FakeRunner(tmp_path / 'r1', clock=VirtualClock())
    tr = arms.run_f_is2('fn_scripted', r, run_id='f1_open_bcomp_20261103', out_dir=tmp_path / 'ours', uc_ref=uc,
                        arm='F-OPEN-BCOMP', experience='auto')
    assert tr['extra']['exp_present'] is True                           # ours: a plan was built and passed
    r2 = H.FakeRunner(tmp_path / 'r2', clock=VirtualClock())
    tr2 = arms.run_f_is2('fn_scripted', r2, run_id='f1_open_ar_20261103', out_dir=tmp_path / 'base', uc_ref=uc,
                         arm='F-OPEN-AR', experience=None)
    assert tr2['extra']['exp_present'] is False                         # AR / RB / baselines: no experience


def test_f1_project_view_yields_a_plan_with_a_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'h'))
    (tmp_path / 'h' / 'experience' / 'runs').mkdir(parents=True)
    from infrarsi.experience import runs as exp_runs
    pv = arms._f1_project_view(T.make_task('F-OPEN-BCOMP'))
    assert pv.gpus == 8 and pv.nodes == 4 and pv.reference_config() == START and 'implementation_stack' in pv.knobs
    pl = exp_runs.plan_for(pv, None, driver=None, log=lambda m: None)
    assert isinstance(pl, dict) and pl.get('fingerprint')


def test_hooks_expose_the_within_16_usd_report_basis():
    """P2: FlashNextHooks supplies search.budget.report_usd = 16 (the open wave's equal-budget basis) through a project
    shim, so the AR / RB adapters' report_basis reads 16, not their 8 default; the open-tree AR / RB code is unchanged."""
    for arm in ('F-OPEN-AR', 'F-OPEN-RB'):
        h = FlashNextHooks(T.make_task(arm, code=True))
        assert h.project.raw['search']['budget']['report_usd'] == 16.0, arm
    # explicit override still flows
    assert FlashNextHooks(T.make_task('F-OPEN-AR'), report_usd=8.0).project.raw['search']['budget']['report_usd'] == 8.0


def test_run_f_is2_honours_an_explicit_llm_usd_override(tmp_path, uc, monkeypatch):
    """P3: an explicit --llm-usd smoke override reaches the task budget (the CLI no longer pops it for the engine
    arms); run_f_is2 forwards it through _prepare."""
    monkeypatch.setitem(search.STRATEGIES, 'fn_scripted', _switch_strategy(S.sample_order(seed=42, n=2)))
    r = H.FakeRunner(tmp_path / 'r', clock=VirtualClock())
    tr = arms.run_f_is2('fn_scripted', r, run_id='f1_open_bcomp_20261103', out_dir=tmp_path / 'b', uc_ref=uc,
                        arm='F-OPEN-BCOMP', llm_usd=1.0, screens=2)
    assert tr['budget']['llm_usd'] == 1.0 and tr['budget']['screens'] == 2
