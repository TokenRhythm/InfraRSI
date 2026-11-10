"""Compact onboarding, explicit recovery consent, budgeted live chat, and actual checkpoint continuation."""
import json
import threading
import time
from types import SimpleNamespace

import pytest

from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.agent.budget import UNLIMITED
from infrarsi.agent.search import SearchSession, USER_STOP
from infrarsi.cli.main import _retry_launch, main
from infrarsi.project import Project
from infrarsi.project.flow import run_project
from infrarsi.project.hooks import ProjectHooks
from infrarsi.project.interview import Interview, questions
from infrarsi.project.live import LiveConsole
from infrarsi.project.recovery import recover, diagnose, _read
from infrarsi.project.resume import latest_run, ResumeError
from test_project import toy  # noqa: F401
from test_interview import Script, namespace


def test_default_checklist_is_six_items_and_one_enter(toy):
    script = Script('')
    qa = Interview(toy, namespace(), read=script, write=script.say)
    qa.run()
    text = '\n'.join(script.said)
    assert len(script.prompts) == 1 and 'Start?' in script.prompts[0]
    assert sum(line.strip()[:1].isdigit() for line in script.said) == 6
    assert all(word in text for word in ('⚠️', 'already runs successfully', 'more', 'Ctrl-G', '--resume'))
    assert 'Training command' not in text and 'On Ctrl-C during' not in text
    assert not qa.changes


def test_compact_number_edits_the_displayed_setting(toy):
    script = Script('3', '2%', '')
    a = namespace()
    Interview(toy, a, read=script, write=script.say).run()
    assert a.tolerance == .02


def test_advanced_caps_can_be_cleared(toy):
    a = namespace()
    ctx = Interview(toy, a)
    items = [q for q in questions() if q.when(ctx)]
    wall = next(i for i, q in enumerate(items, 1) if 'wall clock' in q.label)
    tokens = next(i for i, q in enumerate(items, 1) if 'LLM tokens' in q.label)
    script = Script('more', str(wall), 'none', str(tokens), 'none', '')
    qa = Interview(toy, a, read=script, write=script.say)
    qa.run()
    qa.save()
    b = Project.load(toy.path).task().search_budget()
    assert b.wall_seconds == UNLIMITED and b.llm_tokens == UNLIMITED


def test_fresh_defaults_have_only_one_budget_cap(tmp_path):
    p = Project({'launch': {'command': 'true'}, 'knobs': {'x': {'env': 'X', 'type': 'bool', 'default': False}}}, tmp_path / 'infrarsi.yaml')
    b = p.task().search_budget(screens=40)
    assert (b.screens, b.wall_seconds, b.gpu_hours, b.llm_tokens, b.llm_usd) == (40, UNLIMITED, UNLIMITED, UNLIMITED, None)


def test_no_ambiguous_all_three(toy):
    item = next(q for q in questions() if q.label.startswith('Knobs'))
    shown = item.show(Interview(toy, namespace()))
    assert 'detected parameters:' in shown and 'workers' in shown
    assert 'not a limit' in item.explain(Interview(toy, namespace()))


def test_recovery_decline_never_calls_model_or_writes(toy):
    drv = ScriptedBackend([])
    before = toy.path.read_bytes()
    assert recover(toy, {'reasons': ['failed']}, drv, read=Script('q'), write=lambda _: None) is None
    assert not drv.calls and toy.path.read_bytes() == before


def test_diagnosis_reads_then_proposes_but_needs_approval(toy):
    command = toy.raw['launch']['command'] + ' --example'
    drv = ScriptedBackend([{'diagnosis': 'Inspect launch', 'read_files': ['train.py'], 'command': ''},
                           {'diagnosis': 'Try this', 'read_files': [], 'command': command}])
    before = toy.path.read_bytes()
    assert recover(toy, {'log_tail': ['bad launch']}, drv, read=Script('d', 'n', 'q'), write=lambda _: None) is None
    assert len(drv.calls) == 2 and 'train.py' in drv.calls[1]['prompt']
    assert toy.path.read_bytes() == before
    assert (toy.root / '.infrarsi/launch-recovery.jsonl').exists()


def test_approved_command_is_saved_with_backup(toy):
    p = recover(toy, {}, None, read=Script('m', 'python train.py {args}', 'y'), write=lambda _: None)
    assert p.raw['launch']['command'] == 'python train.py {args}'
    assert Project.load(toy.path).raw['launch']['command'] == 'python train.py {args}'
    assert toy.path.with_suffix('.yaml.bak').is_file()


def test_diagnosis_cannot_read_outside_or_hidden_files(toy, tmp_path):
    (tmp_path / 'secret').write_text('PRIVATE')
    (toy.root / 'link.txt').symlink_to(tmp_path / 'secret')
    for name in ('../secret', 'link.txt', '.env', '.git/config'):
        assert 'refused' in _read(toy.root, name)


def test_diagnosis_read_loop_is_bounded(toy):
    drv = ScriptedBackend(lambda *args: {'diagnosis': 'More', 'read_files': ['train.py'], 'command': ''})
    assert diagnose(toy, {}, drv, write=lambda _: None) is None
    assert len(drv.calls) == 4


def test_failed_baseline_retries_after_user_request(toy, monkeypatch):
    answers = iter(['r'])
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    calls = []
    def operation(p):
        calls.append(p)
        return {'valid': len(calls) == 2, 'reasons': ['failed']}
    p, result = _retry_launch(toy, namespace(interactive=True), None, operation, doctor=True)
    assert result['valid'] and len(calls) == 2


def test_unattended_failure_never_prompts(toy, monkeypatch):
    monkeypatch.setattr('builtins.input', lambda _: pytest.fail('unattended input'))
    _, result = _retry_launch(toy, namespace(yes=True), None, lambda _: {'valid': False}, doctor=True)
    assert not result['valid']


def session(toy, **budget):
    task = toy.task()
    hooks = ProjectHooks(toy, task)
    return SearchSession(task=task, budget=task.search_budget(llm_tokens=30000, **budget), runner=None, hooks=hooks,
                         start_config=toy.reference_config(), run_id='live', arm='test')


def test_chat_charges_shared_budget_and_notes_reach_task_view(toy):
    s = session(toy)
    drv = ScriptedBackend([{'reply': '目前还没有完成试跑。'}], tokens_per_call=100)
    c = LiveConsole(drv, enabled=False, log=lambda _: None)
    c.submit('/note 优先减少通信')
    c.submit('当前进度？')
    c.poll(s)
    assert s.ledger.used['llm_tokens'] == 100 and s.llm_calls[-1]['role'] == 'live_chat'
    assert s.hooks.task_view()['user_guidance'] == ['优先减少通信']
    assert any(e['kind'] == 'live_reply' for e in s.events)
    assert 'control' not in str(s.state_dict().keys())


def test_chat_respects_exhausted_budget(toy):
    s = session(toy)
    s.ledger.used['llm_tokens'] = 30000
    drv = ScriptedBackend([])
    c = LiveConsole(drv, enabled=False, log=lambda _: None)
    c.submit('hello')
    c.poll(s)
    assert not drv.calls


def test_pause_can_resume_and_stop_at_boundary(toy):
    s = session(toy)
    c = LiveConsole(enabled=False, log=lambda _: None)
    c.submit('/pause')
    t = threading.Thread(target=c.poll, args=(s,))
    t.start()
    time.sleep(.05)
    assert t.is_alive()
    c.submit('/resume')
    t.join(1)
    assert not t.is_alive()
    c.submit('/stop')
    c.poll(s)
    assert s.stop_flag == USER_STOP


def test_pipe_does_not_take_over_stdin():
    stream = SimpleNamespace(isatty=lambda: False)
    with LiveConsole(stdin=stream, stdout=stream) as console:
        assert console.thread is None


def test_resume_latest_uses_metadata_with_custom_out(toy, tmp_path, monkeypatch):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    out = tmp_path / 'custom-runs'
    from infrarsi.project.resume import Resume
    def enter(console):
        console.submit('/note Keep communication overhead low')
        return console
    monkeypatch.setattr(LiveConsole, '__enter__', enter)
    original_apply = Resume.apply
    def apply(resume, session, engine, strategy):
        original_apply(resume, session, engine, strategy)
        assert session.hooks.task_view()['user_guidance'] == ['Keep communication overhead low']
    monkeypatch.setattr(Resume, 'apply', apply)
    result = run_project(toy, strategy='bcomp', budget={'screens': 1}, out_root=out, experience='off', log=lambda _: None, live=True)
    run = latest_run(toy.path, out)
    assert str(run) == result['run_dir']
    monkeypatch.setattr(LiveConsole, '__enter__', lambda console: console)
    main(['run', str(toy.root), '--resume', 'latest', '--out', str(out), '--screens', '2', '--llm', 'none', '--yes'])
    resumed = json.loads((run / 'result.json').read_text())
    assert resumed['resumes'] and resumed['search']['used']['screens'] == 2
    # An explicit run directory works without relying on the conventional directory layout.
    main(['run', '--resume', str(run), '--screens', '3', '--llm', 'none', '--yes'])
    assert json.loads((run / 'result.json').read_text())['search']['used']['screens'] == 3


def test_latest_does_not_silently_pick_another_project(toy, tmp_path):
    with pytest.raises(ResumeError, match='no saved search'):
        latest_run(toy.path, tmp_path / 'missing')


@pytest.mark.skipif(__import__('os').name != 'posix', reason='POSIX terminal')
def test_terminal_restores_after_interrupt_and_accepts_unicode():
    import os
    import pty
    import termios
    master, slave = pty.openpty()
    before = termios.tcgetattr(slave)
    try:
        with os.fdopen(os.dup(slave), 'r') as inp, os.fdopen(os.dup(slave), 'w', buffering=1) as out:
            c = LiveConsole(stdin=inp, stdout=out, log=lambda _: None)
            with pytest.raises(KeyboardInterrupt):
                with c:
                    assert not termios.tcgetattr(slave)[3] & termios.ICANON
                    os.write(master, '\x07/note 保持配置\n'.encode())
                    deadline = time.monotonic() + 2
                    while c.messages.empty() and time.monotonic() < deadline:
                        time.sleep(.01)
                    assert c.messages.get_nowait() == '/note 保持配置'
                    raise KeyboardInterrupt
            assert termios.tcgetattr(slave) == before
            assert not c.thread.is_alive()
    finally:
        os.close(master)
        os.close(slave)


def test_post_search_infrastructure_failure_does_not_restart(toy, monkeypatch):
    monkeypatch.setattr('infrarsi.project.recovery.recover', lambda *a, **kw: pytest.fail('not a baseline failure'))
    report = {'outcome': 'infrastructure_failure', 'reference': {'valid': True},
              'search': {'used': {'screens': 16}}, 'promotion': {'infrastructure': 'SSH lost'}}
    _, result = _retry_launch(toy, namespace(interactive=True), None, lambda _: report)
    assert result is report


def test_diagnosis_explicit_zero_budget_sends_nothing(toy):
    from infrarsi.project.recovery import DiagnosisBudget
    drv = ScriptedBackend([])
    limits = DiagnosisBudget.for_run(toy, namespace(budget_tokens=0, budget_usd=0))
    assert diagnose(toy, {}, drv, allowance=limits, write=lambda _: None) is None
    assert not drv.calls


def test_diagnosis_gets_known_failed_log_outside_project(toy, tmp_path):
    log = tmp_path / 'custom-output' / 'train.log'
    log.parent.mkdir()
    log.write_text('Set DATASET_ROOT before launching this job')
    drv = ScriptedBackend([{'diagnosis': 'Missing dataset', 'command': '', 'read_files': []}])
    diagnose(toy, {'reference': {'log': str(log)}}, drv, write=lambda _: None)
    assert 'Set DATASET_ROOT' in drv.calls[0]['prompt']


@pytest.mark.skipif(__import__('os').name != 'posix', reason='POSIX terminal')
def test_second_interrupt_while_joining_always_restores_terminal(monkeypatch):
    import os
    import pty
    import termios
    master, slave = pty.openpty()
    before = termios.tcgetattr(slave)
    try:
        with os.fdopen(os.dup(slave), 'r') as inp, os.fdopen(os.dup(slave), 'w', buffering=1) as out:
            c = LiveConsole(stdin=inp, stdout=out, log=lambda _: None)
            c.__enter__()
            join = c.thread.join
            def interrupt(**kw):
                raise KeyboardInterrupt
            monkeypatch.setattr(c.thread, 'join', interrupt)
            with pytest.raises(KeyboardInterrupt):
                c.__exit__(None, None, None)
            assert termios.tcgetattr(slave) == before
            join(timeout=1)
    finally:
        os.close(master)
        os.close(slave)


@pytest.mark.skipif(__import__('os').name != 'posix', reason='POSIX terminal')
def test_interrupt_during_chat_banner_cleans_up_partial_entry():
    import os
    import pty
    import termios
    master, slave = pty.openpty()
    before = termios.tcgetattr(slave)
    try:
        with os.fdopen(os.dup(slave), 'r') as inp, os.fdopen(os.dup(slave), 'w', buffering=1) as out:
            def interrupted_log(_):
                raise KeyboardInterrupt
            c = LiveConsole(stdin=inp, stdout=out, log=interrupted_log)
            with pytest.raises(KeyboardInterrupt):
                c.__enter__()
            assert termios.tcgetattr(slave) == before
            assert c.closed.is_set() and not c.thread.is_alive()
    finally:
        os.close(master)
        os.close(slave)
