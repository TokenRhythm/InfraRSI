"""The same-money extras on the F1 task (user 11-03 17:4x; FABLE_EMPTY_ASKS_20261103): F-BUD-BCOMP run as
F-OPEN-BCOMP (run_f_is2 with bcomp, ours' cross-job experience, C_F1 with --code, the locked pair in
the v3 counterpart's parity order) from the v4.1 tree (pc_code_o2b, through fn_launch) until it has
spent what F-OPEN-AR spent on this job (23.71 USD, its 3 M token cap; 105 screens as a safety cap), with their caps
(task.BUD_MANIFEST) and search-state checkpoints (<out>/state), resumable with `arm ... --resume`. Labelled extras, outside
the window's counts. CPU only, no network, no node, no real model.

The last test runs the registered strategy itself, crashes it and resumes it, so it runs only where the package has
search-state checkpoints and registers the strategy: the main repo's tests under scripts/fn_overlay_plugin.py with
FN_TREE = qa_trials/pc_code_o2b (F-BUD-BCOMP) - the v4.2 tree o2c passes too;
elsewhere it is skipped."""
import json

import pytest

import x6_helpers as H
from infrarsi import search
from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.agent.budget import VirtualClock
from infrarsi.ext.flashnext import arms
from infrarsi.ext.flashnext import cli
from infrarsi.ext.flashnext import task as T
from infrarsi.harness import signing
from test_f1_open4_arms import _role_script, _run
from test_flashnext_code_arm import CodeFakeRunner

BUD = ('F-BUD-BCOMP',)
V3_OF = {'F-BUD-BCOMP': 'F-OPEN-BCOMP'}
STRATEGY = {'F-BUD-BCOMP': 'bcomp'}
BUDGET = (105, 3_000_000, 23.71, 72000.0, 700.0)
CAPS = {'search': {'budget': {'llm_calls': 315, 'report_usd': 23.71},
                   'caps': {'fix_calls_per_screen': 2, 'fix_calls': 105, 'patches_per_ask': 4, 'code_probes': 63}},
        'llm': {'max_usd_per_call': 8, 'timeout_s': 1200}}


@pytest.fixture
def uc(tmp_path):
    return H.uc_reference_receipt(tmp_path / 'uc', clock=VirtualClock())


def _uc_file(tmp_path, uc):
    path = tmp_path / 'uc_ref.json'
    path.write_text(json.dumps(uc), encoding='utf-8')
    return path


# ---------------------------------------------------------------- the contract
def test_bud_extras_budget_caps_and_index():
    assert T.BUD_ARMS == BUD and T.EXTRA_ARMS == T.V4_ARMS + BUD and T.BUD_MANIFEST == CAPS
    for arm in BUD:
        a = T.ARMS[arm]
        assert (a['screens'], a['llm_tokens'], a['llm_usd'], a['wall_seconds'], a['gpu_hours']) == BUDGET and a['llm']
        for code in (False, True):
            t = T.make_task(arm, code=code)
            assert t.raw['manifest'] == CAPS and t.arm == arm and t.raw['manifest'] is not T.BUD_MANIFEST
            assert t.action_space == ('knobs+patch' if code else 'knobs')
            assert t.search_budget().llm_usd == 23.71 and t.search_budget().screens == 105
        assert arm in T.OPEN_ARMS and arm not in T.ARM_INDEX and arm not in T.CODE_ARMS
        assert arms.pair_order(arm) == arms.pair_order(V3_OF[arm]) and arms.EXTRA_STRATEGY[arm] == STRATEGY[arm]
        assert T.extra_label(arm) == T.BUD_LABEL
    assert T.OPEN_ARM_INDEX['F-BUD-BCOMP'] == 14
    assert len(set(arms.ALL_ARM_INDEX.values())) == len(arms.ALL_ARM_INDEX)
    assert T.extra_label('F-OPEN4-BCOMP') == T.V4_LABEL
    assert {a: arms.EXTRA_STRATEGY[a] for a in T.V4_ARMS} == arms.V4_STRATEGY


# ---------------------------------------------------------------- the CLI dispatch
def test_cli_runs_the_bud_extras_with_checkpoints(tmp_path, uc, monkeypatch, capsys):
    """`arm F-BUD-*` -> run_f_is2 with the v3 counterpart's strategy, ours' experience, checkpoints on; --resume passes
    resume=True; every other arm (open-wave, v4, the setting-1 window, F-A0-CODE) keeps no checkpoint and refuses --resume
    before anything is dispatched (exit 6); a refused resume exits 7."""
    seen = []

    def fake_is2(strategy, runner, **kw):
        seen.append((strategy, kw['arm'], kw['experience'], kw['checkpoint'], kw['resume']))
        if kw['resume'] and kw['run_id'] == 'other':
            raise ValueError('nothing to resume')
        return {'arm': kw['arm'], 'stop_reason': 'stub'}
    monkeypatch.setattr(arms, 'run_f_is2', fake_is2)
    monkeypatch.setattr(cli, '_runner', lambda args, out, reference=None, cap_s=None: object())
    path = _uc_file(tmp_path, uc)

    def go(arm, *extra, rid='r'):
        rc = cli.main(['arm', arm, '--run-id', rid, '--uc-ref', str(path), '--nodes', ','.join(H.NODES), '--out',
                       str(tmp_path / arm), '--code', *extra])
        capsys.readouterr()
        return rc
    assert [go(a) for a in BUD + ('F-OPEN-BCOMP',)] == [0, 0]
    assert go('F-BUD-BCOMP', '--resume') == 0
    assert go('F-OPEN-BCOMP', '--resume') == 6 and go('F-OPEN4-BCOMP', '--resume') == 6
    monkeypatch.setitem(arms.ARM_FUNCS, 'F-A0', lambda *a, **kw: seen.append('F-A0 dispatched') or {})
    monkeypatch.setitem(arms.ARM_FUNCS, 'F-A0-CODE', lambda *a, **kw: seen.append('F-A0-CODE dispatched') or {})
    assert all(go(a, '--resume') == 6 for a in ('F-A0', 'F-A0-CODE', 'F-RND', 'F-OPTICO', 'F-PROMPTS', 'F-OPTICO-20',
                                                 'F-RND-20-s42'))
    assert go('F-BUD-BCOMP', '--resume', rid='other') == 7
    assert seen == [('bcomp', 'F-BUD-BCOMP', 'auto', True, False),
                    ('bcomp', 'F-OPEN-BCOMP', 'auto', False, False), ('bcomp', 'F-BUD-BCOMP', 'auto', True, True),
                    ('bcomp', 'F-BUD-BCOMP', 'auto', True, True)]


# ---------------------------------------------------------------- the summary
def test_bud_extras_stay_out_of_the_window_counts(tmp_path, uc):
    clock = VirtualClock()
    runs, dirs = {}, {}
    for arm in ('F-OPEN-BCOMP', 'F-BUD-BCOMP'):
        tr, lk, seen, r = _run(tmp_path, uc, arm, clock)
        runs[arm], dirs[arm] = {'trace': tr, 'locked': lk}, [tmp_path / arm, r.out_dir]
        if arm == 'F-BUD-BCOMP':
            assert seen['manifest'] == CAPS and seen['usd'] == 23.71
            assert lk['order'] == arms.pair_order('F-OPEN-BCOMP') and lk['arm_index'] == 14
    v3 = {a: runs[a] for a in ('F-OPEN-BCOMP',)}
    rec_v3 = arms.load_receipts(dirs['F-OPEN-BCOMP']) + [uc]
    rec_all = rec_v3 + arms.load_receipts(dirs['F-BUD-BCOMP'])
    m3, m_all = arms.margin(rec_v3), arms.margin(rec_all)
    half = runs['F-BUD-BCOMP']['locked']['uc_half']['attempt_id']
    assert arms._margin_key(m3) == arms._margin_key(m_all)
    assert [(x['attempt_id'], x['why']) for x in m_all['excluded']] == [(half, 'a same-money extra (post hoc): outside S_UC')]
    s3 = arms.summarize(uc_ref=uc, ht=None, arms=v3, receipts=rec_v3, allow_nonformal=True)
    s = arms.summarize(uc_ref=uc, ht=None, arms={**v3, 'F-BUD-BCOMP': runs['F-BUD-BCOMP']}, receipts=rec_all,
                       allow_nonformal=True)
    assert s['labels'] == s3['labels'] == {} and s['margin'] == s3['margin'] and 'bud_extras' not in s3
    assert s['drift']['max_over_min_uc_halves'] == s3['drift']['max_over_min_uc_halves'] is None  # one retained wave arm
    assert all(s['rows'][a] == s3['rows'][a] for a in v3)
    assert s['rows']['F-BUD-BCOMP']['extra_arm'] == T.BUD_LABEL and 'v4_extras' not in s
    assert s['bud_extras']['arms'] == ['F-BUD-BCOMP'] and s['bud_extras']['label'] == T.BUD_LABEL
    assert not [x for x in s['nonformal_reasons'] if x.startswith('F-BUD-BCOMP')], s['nonformal_reasons']


# ---------------------------------------------------------------- crash and resume (a v4.2 tree only)
class Crash(BaseException):
    """The process dying under the search (not an Exception: nothing in the search catches it)."""


class CrashingRunner(CodeFakeRunner):
    """CodeFakeRunner that dies at its `crash_at`-th launched screen (before running it)."""

    def __init__(self, *a, crash_at=None, **kw):
        super().__init__(*a, **kw)
        self.crash_at = crash_at

    def run(self, task, config, overlay, purpose, run_id, **kw):
        if self.crash_at is not None and self.n + 1 >= self.crash_at:
            raise Crash(f'killed at screen {self.n + 1}')
        return super().run(task, config, overlay, purpose, run_id, **kw)


def _tree_ready(arm):
    try:
        from infrarsi.agent import state  # noqa: F401
        from infrarsi.search import checkpoint  # noqa: F401
    except ImportError:
        return False
    return STRATEGY[arm] in search.STRATEGIES and STRATEGY[arm] in getattr(search, 'RESUMABLE', ())


@pytest.mark.parametrize('arm', BUD)
def test_the_tree_strategy_crashes_and_resumes(tmp_path, uc, monkeypatch, capsys, arm):
    """The real CLI path (`arm ARM --code`, 12 screens --nonformal) with the tree's strategy, a scripted model and a fake
    runner that dies at its 6th screen: the checkpoint in <out>/state is this run's; `--resume` continues it on a new
    runner (the screens and model calls before the crash kept, not asked or run again; the trace's extra.resumes has the
    entry) and finishes; a resume under another run id is refused (exit 7, nothing launched)."""
    if not _tree_ready(arm):
        pytest.skip(f'{STRATEGY[arm]} with search-state checkpoints is in its v4.1+ tree only (fn_overlay_plugin.py, FN_TREE)')
    import infrarsi.agent.arms as agent_arms
    home = tmp_path / 'home'
    (home / 'experience' / 'runs').mkdir(parents=True)
    monkeypatch.setenv('INFRARSI_HOME', str(home))
    backends = []

    def driver():
        b = ScriptedBackend(_role_script(), tokens_per_call=2000)
        b.timeout_seconds, b.max_usd_per_call = 900.0, 5.0
        backends.append(b)
        return b
    monkeypatch.setattr(agent_arms, 'strong_driver', driver)
    clock = VirtualClock()
    runners = [CrashingRunner(tmp_path / 'run1', clock=clock, crash_at=6), CrashingRunner(tmp_path / 'run2', clock=clock)]
    monkeypatch.setattr(cli, '_runner', lambda args, out, reference=None, cap_s=None: runners[0] if not runners[0].used
                        else runners[1])
    for r in runners:
        r.used = False
    path, out = _uc_file(tmp_path, uc), tmp_path / 'arm'
    argv = ['arm', arm, '--run-id', 'rid', '--uc-ref', str(path), '--nodes', ','.join(H.NODES), '--out', str(out),
            '--code', '--screens', '12', '--nonformal']
    with pytest.raises(Crash):
        cli.main(argv)
    runners[0].used = True
    capsys.readouterr()
    head = json.loads((out / 'state' / 'checkpoint.json').read_text(encoding='utf-8'))
    assert head['run_id'] == 'rid' and head['arm'] == arm and head['screens_done'] >= 4 and runners[0].n == 5
    assert not (out / 'trace.json').exists()
    calls_before = len(backends[0].calls) if hasattr(backends[0], 'calls') else None
    assert cli.main(argv[:3] + ['other'] + argv[4:] + ['--resume']) == 7          # another run id: refused
    uc2 = H.uc_reference_receipt(tmp_path / 'uc2', clock=VirtualClock())
    path2 = tmp_path / 'uc_ref2.json'
    path2.write_text(json.dumps(uc2), encoding='utf-8')
    i = argv.index('--uc-ref')
    assert uc2['attempt_id'] != uc['attempt_id']
    assert cli.main(argv[:i + 1] + [str(path2)] + argv[i + 2:] + ['--resume']) == 7   # another U+C reference: refused
    assert runners[1].n == 0
    assert cli.main(argv + ['--resume']) == 0
    capsys.readouterr()
    tr = json.loads((out / 'trace.json').read_text(encoding='utf-8'))
    assert signing.verify(tr) and tr['arm'] == arm
    res = tr['extra']['resumes']
    assert len(res) == 1 and res[0]['from_checkpoint']['sequence'] == head['sequence']
    assert res[0]['screens_done'] == head['screens_done'] and res[0]['flashnext_changed'] is False
    assert res[0]['calls_before'] == head['llm_calls'] and 0 < res[0]['screens_before'] <= len(tr['screens'])
    assert tr['used']['screens'] >= head['screens_done'] and len(tr['llm_calls'] or []) >= head['llm_calls']
    assert runners[1].n >= 1                                                       # it went on, on the new runner
    assert runners[0].n + runners[1].n == len([c for c in tr['candidates'] if c.get('screened') or c.get('verdict')]) \
        or runners[0].n + runners[1].n >= tr['used']['screens']
    if calls_before is not None:
        assert len(backends[-1].calls) <= len(tr['llm_calls'] or []) - head['llm_calls'] + 1
