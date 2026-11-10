"""ext/flashnext command line (X6): the no-GPU commands, and the GPU commands end to end with the runner factory
replaced by the fake four-node runner (no node, no lease, no network)."""
import json

import pytest

import x6_helpers as H
from infrarsi.agent.budget import VirtualClock
from infrarsi.ext.flashnext import arms
from infrarsi.ext.flashnext import cli
from infrarsi.ext.flashnext import switch_space as S
from infrarsi.ext.flashnext import task as T


def _json(capsys):
    return json.loads(capsys.readouterr().out)


def test_view_space_and_random_order(capsys):
    assert cli.main(['view', 'F-PROMPTS']) == 0
    view = _json(capsys)
    assert view['arm'] == 'F-PROMPTS' and view['switch_space'] == S.task_view() and T.task_view_leaks(view) == []
    assert cli.main(['space']) == 0
    sp = _json(capsys)
    assert sp['admissible_points'] == 353 and sp['sha256'] == S.sha256() and sp['start'] == S.start_point()
    assert {'B', 'S0', 'S_match', 'S_match_full', 'W'} <= set(sp['registered_reachable'])
    assert cli.main(['rnd-seq', '--n', '4']) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert [json.loads(x.split(' ', 1)[1]) for x in lines] == S.sample_order(seed=42, n=4)


def test_rederive_the_20_stored_cells(capsys):
    if not H.F54.exists():
        pytest.skip('formal-54 final tree not on this machine')
    assert cli.main(['rederive']) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert json.loads(lines[-1]) == {'cells': 20, 'all_ok': True} and len(lines) == 21
    assert sum('MISMATCH' in x for x in lines) == 0 and sum('class=FAIL (expected FAIL)' in x for x in lines) == 8


def test_window_commands_with_the_fake_runner(tmp_path, monkeypatch, capsys):
    clock = VirtualClock()
    fake = H.FakeRunner(tmp_path / 'run', clock=clock,
                        base=H.scale_profile(H.profile_of(H.uc_a()), H.UC_REF_TIME_FACTOR))
    monkeypatch.setattr(arms, 'make_runner', lambda nodes, **kw: fake)
    monkeypatch.setattr(arms.J, 'stored_reference', lambda: H.stored_ref(H.uc_a()))
    root = tmp_path / 'win'
    assert cli.main(['ref-uc', '--run-id', 'ucref', '--out', str(root / 'ucref')]) == 0
    res = _json(capsys)
    assert res['valid'] and res['numerics_class'] == 'E0' and res['stop_window'] is False
    uc = str(root / 'ucref' / 'uc_ref.json')
    assert cli.main(['ref-ht', '--run-id', 'ht', '--uc-ref', uc, '--out', str(root / 'ht')]) == 0
    assert _json(capsys)['line'] == 'best known human recipe on this node set'
    assert cli.main(['arm', 'F-RND', '--run-id', 'frnd', '--uc-ref', uc, '--out', str(root / 'frnd'), '--screens', '3', '--nonformal']) == 0
    tr = _json(capsys)
    assert tr['arm'] == 'F-RND' and tr['stop_reason'] == 'budget:screens' and (root / 'frnd' / 'trace.json').exists()
    assert cli.main(['locked', 'F-RND', '--run-id', 'lk', '--uc-ref', uc, '--from', str(root / 'frnd' / 'trace.json'),
                     '--out', str(root / 'frnd')]) == 0
    lk = _json(capsys)
    assert lk['order'] == ['candidate', 'uc'] and lk['status'] == 'ok'
    assert cli.main(['margin', '--runs', str(tmp_path), '--out', str(root)]) == 0
    m = _json(capsys)
    assert m['n'] == 2 and m['estimator'] == 'fallback_within_window' and (root / 'margin_F1.json').exists()
    assert cli.main(['summary', '--uc-ref', uc, '--ht', str(root / 'ht' / 'ht.json'), '--arm', f'F-RND={root / "frnd"}',
                     '--margin', str(root / 'margin_F1.json'), '--out', str(root), '--allow-nonformal']) == 0
    out = _json(capsys)
    assert out['rows']['F-RND']['step_time_locked'] == lk['t_locked'] and (root / 'summary_F1.json').exists()


def test_a_failed_uc_reference_stops_the_window(tmp_path, monkeypatch, capsys):
    fake = H.FakeRunner(tmp_path / 'run', clock=VirtualClock(),
                        drift={S.key(S.start_point()): {'ce_rel': 0.01}})
    monkeypatch.setattr(arms, 'make_runner', lambda nodes, **kw: fake)
    monkeypatch.setattr(arms.J, 'stored_reference', lambda: H.stored_ref(H.uc_a()))
    assert cli.main(['ref-uc', '--run-id', 'ucref', '--out', str(tmp_path / 'ucref')]) == 2
    out = capsys.readouterr().out
    assert '"stop_window": true' in out and 'STOP THE WINDOW' in out
