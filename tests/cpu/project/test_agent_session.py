"""`infrarsi agent` (project/agent_session.py) with the session backend on the toy CPU job: a bcomp run whose model calls
are requests a stand-in session answers - in one process through run_project, and as the commands run it (start: the
engine detached, optimize's stages, nothing written to the project; next / answer until the report); a stop at a boundary
with the re-verification; a note that reaches later prompts; an engine killed mid-search found dead and resumed from its
checkpoint; the requests the CLI refuses."""
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))

from infrarsi.agent.backends import session as S  # noqa: E402
from infrarsi.project import Project  # noqa: E402
from infrarsi.project import agent_session as A  # noqa: E402
from infrarsi.project.flow import run_project  # noqa: E402
from session_helpers import Answerer, wait_for  # noqa: E402
from test_bcomp_project import scripted, toy  # noqa: E402,F401  (the toy copy and its scripted model)

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    monkeypatch.delenv(S.ENV_DIR, raising=False)
    monkeypatch.delenv(S.ENV_MODEL, raising=False)
    return tmp_path / 'home'


def drive(d, reply, *, tokens=2000, timeout=300.0, before=None):
    """The session's loop - next, answer, next - until the run needs no more answers: (the last status, the roles
    answered). `before(request)` runs when a request is seen and before it is answered, that is while the engine is
    held waiting for it (so what a test does there cannot race the engine); a true value ends the loop there, the
    request left unanswered."""
    t0, roles = time.monotonic(), []
    while time.monotonic() - t0 < timeout:
        st = A.wait_next(d, wait=20, poll=0.05)
        if st['state'] == 'request':
            r = S.find(d, st['request']['id'])
            if before is not None and before(r):
                return st, roles
            item = reply(r.role, r.text('system.txt'), r.text('prompt.txt'))
            Path(st['request']['answer']).write_text(json.dumps(item), encoding='utf-8')
            got = A.answer(d, r.seq, tokens=tokens)
            if not got['ok'] and (d / S.CONTROL / 'stop').is_file():   # a stop closes the call that was waiting
                assert got['closed'] and 'closed without an answer' in got['error'], got
                continue
            assert got['ok'], got
            roles.append(r.role)
        elif st['state'] != 'running':
            return st, roles
    raise AssertionError('the run did not end: ' + json.dumps(A.status(d), default=str)[:3000])


def end_engine(d):
    """Kill the session's engine and what it started, as a crash or a reboot would."""
    pid = A.load(d)['engine']['pid']
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], capture_output=True)
    else:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
        try:                                             # (started from this process, it is this process's child: without
            os.waitpid(pid, 0)                           #  this it would stay as an ended process nobody collected)
        except ChildProcessError:
            pass
    wait_for(lambda: not A.pid_alive(pid), timeout=30)


@pytest.fixture(autouse=True)
def no_engine_left_behind(toy):  # noqa: F811
    """A test that fails half way must not leave its detached engine waiting for an answer."""
    yield
    d = A.current(toy)
    if d is not None and (A.load(d).get('engine') or {}).get('pid') and A.engine_alive(d):
        end_engine(d)


def test_a_bcomp_run_answered_through_the_session(toy, tmp_path, home):  # noqa: F811
    d = toy / '.infrarsi' / 'sessions' / 'one'
    d.mkdir(parents=True)
    S.write_json(d / 'session.json', {'schema': A.SESSION_SCHEMA, 'id': 'one', 'state': 'running',
                                      'engine': {'pid': os.getpid(), 'host': __import__('socket').gethostname()}})
    drv, calls, _ = scripted(toy)
    be = S.SessionBackend(directory=d, model='claude-opus-5-5', poll_seconds=0.01)
    with Answerer(d, drv.script, tokens=2000) as a:
        res = run_project(Project.load(toy), strategy='bcomp', driver=be, budget={'screens': 4, 'llm_tokens': 10 ** 6},
                          out_root=tmp_path / 'out', log=lambda m: None)
    assert res['strategy'] == 'bcomp' and res['search']['used']['screens'] == 4 and res['outcome'] in ('improved', 'no_change')
    reqs = S.requests(d)
    assert calls[0] == 'prior' and [r.role for r in reqs] == calls and all(r.state == 'answered' for r in reqs)
    assert all(h['ok'] for h in a.handed) and len(a.handed) == len(calls)
    assert res['search']['llm_calls'] == len(calls) and res['cost']['llm_tokens'] == 2000 * len(calls)
    assert res['cost']['llm_usd'] == 0
    assert A.run_dir_of(d) == Path(res['run_dir'])       # the flow named the run directory to the session
    kept = sorted(p.name for p in (Path(res['run_dir']) / 'llm_calls').iterdir())   # and the run keeps its calls itself
    assert kept == [r.path.name for r in reqs]
    first = Path(res['run_dir']) / 'llm_calls' / kept[0]
    assert S.read_json(first / 'request.json')['prompt'] == reqs[0].text('prompt.txt')
    assert json.loads((first / 'result.txt').read_text(encoding='utf-8'))['levels']
    assert S.read_json(first / 'receipt.json')['llm_tokens'] == 2000
    p = S.read_json(d / 'progress.json')                 # and wrote where the search stood at its boundaries
    assert p['screens_budget'] == 4 and 1 <= p['screens_done'] <= 4
    assert A.summary(d)['outcome'] == res['outcome'] and A.summary(d)['screens'] == 4
    v = res['search']['bcomp']['bcomp']                  # the search is bcomp's, as with any backend (test_bcomp_project):
    assert v['starts']['experience']['change'] == {'workers': 4}   # no home to learn from - the guide's best stands in


def test_start_next_answer_to_the_report(toy, tmp_path, home, monkeypatch):  # noqa: F811
    monkeypatch.chdir(tmp_path)
    before = {p.name: p.read_bytes() for p in toy.iterdir() if p.is_file()}
    drv, calls, seen = scripted(toy)
    st = A.start(toy, model='claude-opus-5-5', options=['--screens', '4', '--budget-tokens', '1000000'])
    assert st['state'] in ('running', 'request'), st
    d = A.current(toy)
    assert d is not None and st['session'] == str(d) and A.locate(None, toy) == d
    assert A.current(toy / 'a' / 'sub') == d and A.current(tmp_path) is None   # (found from inside the project only)
    with pytest.raises(A.AgentError, match='is active'):
        A.start(toy)                                     # one run of a project at a time
    final, roles = drive(d, drv.script)
    assert final['state'] == 'finished' and final['exit_code'] == 0, final
    res = final['result']
    assert res['outcome'] in ('improved', 'no_change') and res['screens'] == 4 and res['strategy'] == 'bcomp'
    assert Path(res['report']).is_file() and roles == calls and roles[0] == 'prior'
    assert f'infrarsi report {res["run_dir"]}' in final['do'] and 'when the user agrees' in final['do']
    log = (d / 'engine.log').read_text(encoding='utf-8', errors='replace')
    for stage in ('stage 1/5 (init)', 'stage 2/5 (doctor)', 'stage 3/5 (run): searching with strategy bcomp',
                  'stage 4/5 (report)', 'stage 5/5 (apply)'):
        assert stage in log, log[-3000:]
    assert 'model claude-opus-5-5, capability 0.90 (inferred from the model name)' in log
    after = {p.name: p.read_bytes() for p in toy.iterdir() if p.is_file()}
    assert after == before                               # apply stays the user's: the project is as it was
    state = A.load(d)
    assert state['argv'][:2] == ['optimize', str(toy)] and '--dry-run' in state['argv'] and '--yes' in state['argv']
    assert state['argv'][state['argv'].index('--llm') + 1] == 'session:claude-opus-5-5'
    assert not A.engine_alive(d) and A.status(d)['state'] == 'finished'
    assert A.stop(d)['ok'] is False                      # nothing to stop
    with pytest.raises(A.AgentError, match='no request 99'):
        A.answer(d, 99)
    hyp = [p for role, _, p in seen if role == 'hypotheses']
    assert hyp and all('measured_history' in p for p in hyp)   # the prompts are the search's own


def test_a_stop_at_a_boundary_and_a_note(toy, tmp_path, home, monkeypatch):  # noqa: F811
    monkeypatch.chdir(tmp_path)
    drv, calls, seen = scripted(toy)
    A.start(toy, model='claude-sonnet-5-5', options=['--screens', '12', '--budget-tokens', '1000000', '--no-doctor'])
    d = A.current(toy)
    box = {}

    def before(r):                                       # (the engine is held waiting for this request's answer)
        if r.seq == 1:                                   # the first request: guidance for every later proposal
            assert A.note(d, 'keep the dataloader as it is')['ok']
        if r.role == 'hypotheses' and calls.count('hypotheses') == 1 and 'stop' not in box:
            box['stop'] = A.stop(d)                      # the second request for patches: stop instead of answering;
            wait_for(lambda: r.state != 'pending')       # the engine ends the call that was waiting (an answer handed
        return False                                     # in before it looked would still count: not raced here)
    final, roles = drive(d, drv.script, before=before)
    assert box['stop']['ok'] and final['state'] == 'finished', final
    res = final['result']
    assert res['stop_reason'] == 'user_stop' and 1 <= res['screens'] < 12 and res['outcome'] in ('improved', 'no_change')
    closed = [r for r in S.requests(d) if (r.accepted() or {}).get('status') == 'STOPPED']
    assert len(closed) == 1 and closed[0].role == 'hypotheses'   # the call that was waiting ended with the stop
    assert 'stage 2/5 (doctor): skipped' in (d / 'engine.log').read_text(encoding='utf-8', errors='replace')
    run = Path(res['run_dir'])
    events = [json.loads(x) for x in (run / 'search' / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert any(e['kind'] == 'live_message' and e['payload']['text'] == '/note keep the dataloader as it is' for e in events)
    later = [p for role, _, p in seen[1:] if role == 'hypotheses']
    assert later and all(p['task']['user_guidance'] == ['keep the dataloader as it is'] for p in later)   # in later prompts


def test_an_engine_that_died_is_found_dead_and_resumed(toy, tmp_path, home, monkeypatch):  # noqa: F811
    monkeypatch.chdir(tmp_path)
    drv, calls, _ = scripted(toy)
    A.start(toy, model='claude-opus-5-5', options=['--screens', '6', '--budget-tokens', '1000000', '--no-doctor'])
    d = A.current(toy)

    def before(r):                                       # a request that comes once the search has a checkpoint and two
        if r.seq == 1:                                   # runs behind it: left unanswered, the engine waiting for it
            A.note(d, 'leave the loss alone')            # (and a note at the start: in the checkpoint by then)
        run = A.run_dir_of(d)
        done = (S.read_json(d / 'progress.json') or {}).get('screens_done') or 0
        return bool(run and (run / 'search' / 'state' / 'checkpoint.json').is_file() and done >= 2)
    mid, _ = drive(d, drv.script, before=before)
    assert mid['state'] == 'request', mid
    waiting = mid['request']['id']
    end_engine(d)
    dead = A.status(d)
    assert dead['state'] == 'dead' and 'infrarsi agent resume' in dead['do'] and not A.engine_alive(d)
    assert A.wait_next(d, wait=0)['state'] == 'dead'
    # two resumes at once (a command sent twice): one engine, the other is told the run is active
    got = []

    def resume():
        try:
            got.append(A.resume(d)['state'])
        except A.AgentError as exc:
            got.append(str(exc))
    threads = [threading.Thread(target=resume) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert sorted(x in ('running', 'request') for x in got) == [False, True], got
    assert any('is active' in x for x in got)
    engine = A.load(d)['engine']['pid']
    assert A.hold(d.parent / A.ENGINE_LOCK) is None      # the engine has the project
    code = A.serve(d)                                    # an engine started by hand finds it taken and touches nothing
    assert code == A.BUSY and A.load(d)['engine']['pid'] == engine and A.load(d)['state'] == 'running'
    final, _ = drive(d, drv.script)
    assert final['state'] == 'finished' and final['exit_code'] == 0, final
    res = json.loads((Path(final['result']['run_dir']) / 'result.json').read_text(encoding='utf-8'))
    assert res['search']['used']['screens'] == 6 and len(res['resumes']) == 1 and res['outcome'] in ('improved', 'no_change')
    states = {r.seq: r.state for r in S.requests(d)}
    assert states[waiting] == 'abandoned' and set(states.values()) == {'answered', 'abandoned'}   # asked again by the new one
    # the note came back with the checkpoint, together with how far the notes were read: in the prompts after the
    # resume, and taken once (not read a second time by the engine that resumed)
    run = Path(final['result']['run_dir'])
    events = [json.loads(x) for x in (run / 'search' / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert sum(1 for e in events if e['kind'] == 'live_message') == 1
    after = [r for r in S.requests(d) if r.seq > waiting and r.role == 'hypotheses']
    assert after and all(json.loads(r.text('prompt.txt'))['task']['user_guidance'] == ['leave the loss alone'] for r in after)
    lock = A.hold(d.parent / A.ENGINE_LOCK)              # the engine is gone, and its lock with it
    assert lock is not None and A.hold(d.parent / A.ENGINE_LOCK) is None
    lock.close()
    state = A.load(d)
    assert state['mode'] == 'resume' and state['resumes'] == 1 and state['argv'][0] == 'run' and '--resume' in state['argv']
    assert state['options'] == ['--screens', '6', '--budget-tokens', '1000000']   # the budget it started with (not --no-doctor)
    assert A._carried(['--screens', '6', '--no-doctor', '--requirements', 'a b', '--wall-hours', '2h', '--verbose'],
                      ['--screens', '9']) == ['--wall-hours', '2h', '--verbose', '--screens', '9']
    with pytest.raises(A.AgentError, match='is active|nothing to resume'):   # (a finished run with its checkpoint could
        A.resume(tmp_path / 'nowhere')                                        #  be resumed with a larger budget)


def test_stop_now_ends_the_engine_and_says_what_it_reached(toy, tmp_path, home, monkeypatch):  # noqa: F811
    monkeypatch.chdir(tmp_path)
    drv, calls, _ = scripted(toy)
    A.start(toy, model='claude-opus-5-5', options=['--screens', '6', '--budget-tokens', '1000000', '--no-doctor'])
    d = A.current(toy)
    box = {}

    def before(r):                                       # at the first request (the engine waiting): end it at once
        box['pid'] = A.load(d)['engine']['pid']
        box['out'] = A.stop(d, now=True)
        return True
    drive(d, drv.script, before=before)
    out = box['out']
    assert out['ok'] and out['state'] in ('failed', 'finished') and not A.pid_alive(box['pid']) and not A.engine_alive(d)
    assert 'told to end at once' in out['do'] and ('Nothing was promoted' in out['do']) == (not (out.get('result') or {}).get('promoted'))
    assert A.status(d)['state'] == out['state'] and S.pending(d) == []
    assert A.stop(d)['ok'] is False                      # nothing left to stop
    lock = A.hold(d.parent / A.ENGINE_LOCK)              # and the project is free for the next run
    assert lock is not None
    lock.close()


def test_the_engine_names_its_session_as_the_current_one(toy, tmp_path, home, monkeypatch):  # noqa: F811
    """A launcher killed while it waits for its engine (a tool's time limit) must not leave an engine `status` cannot
    find: the engine publishes `current` itself, once the project is its."""
    monkeypatch.chdir(tmp_path)
    base = A.sessions_root(toy)
    d = base / 'by-hand'
    d.mkdir(parents=True)
    S.write_json(d / 'session.json', {'schema': A.SESSION_SCHEMA, 'id': 'by-hand', 'root': str(toy), 'state': 'starting',
                                      'created_utc': A.utc(), 'spawned_utc': A.utc(),
                                      'argv': ['optimize', str(toy), '--llm', 'session', '--yes', '--dry-run',
                                               '--strategy', 'no-such-strategy']})
    assert A.current(toy) is None
    monkeypatch.setenv(S.ENV_DIR, str(d))
    assert A.serve(d) == 2                               # (the engine's command fails at once: a request error)
    assert A.current(toy) == d and A.load(d)['state'] == 'failed' and A.load(d)['exit_code'] == 2


def test_start_says_at_once_when_the_engine_cannot_use_its_home_or_its_key(toy, tmp_path, home, monkeypatch):  # noqa: F811
    """Inside a sandbox the engine may be unable to write its home directory, or to use the key it signs receipts with
    (a real Codex session on Windows met the second and made a key itself): the start refuses before anything runs, in
    words for the user, and says that the key is not the agent's to make."""
    from infrarsi.harness import signing
    monkeypatch.chdir(tmp_path)
    blocker = tmp_path / 'a-file'
    blocker.write_text('x', encoding='utf-8')
    monkeypatch.setenv('INFRARSI_HOME', str(blocker / 'home'))          # a directory that cannot be made
    with pytest.raises(A.AgentError, match='cannot write its home directory'):
        A.start(toy)
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))

    def denied():
        raise PermissionError('[WinError 5] Access is denied')
    monkeypatch.setattr(signing, 'key_id', denied)
    with pytest.raises(A.AgentError, match='do not create, copy or edit it') as err:
        A.start(toy)
    assert 'outside the sandbox' in str(err.value) and 'PermissionError' in str(err.value)
    assert '`infrarsi agent start`' in str(err.value)
    assert A.current(toy) is None and not A.sessions_root(toy).exists()   # nothing was started, nothing left behind
    with pytest.raises(A.AgentError, match='`infrarsi agent resume` run outside the sandbox'):
        A.preflight('resume')                            # a resume names itself: its run must not be started anew
    monkeypatch.undo()                                   # (the real key check again: the test session's throwaway key)
    monkeypatch.delenv(S.ENV_DIR, raising=False)
    kept = tmp_path / 'kept-home'                        # no more is asked than a run asks: a ledger it can append to
    kept.mkdir()
    (kept / 'ledger.jsonl').write_text('{"kind": "noise"}\n', encoding='utf-8')
    monkeypatch.setenv('INFRARSI_HOME', str(kept))
    if os.name != 'nt':                                  # a home that takes no new entries, with its ledger writable
        os.chmod(kept, 0o555)
    try:
        A.preflight()
        assert (kept / 'ledger.jsonl').read_text(encoding='utf-8') == '{"kind": "noise"}\n'   # opened, not written
        os.chmod(kept / 'ledger.jsonl', 0o444)           # a ledger that cannot be appended to: refused
        if os.name == 'nt' or os.geteuid() != 0:         # (root writes anything)
            with pytest.raises(A.AgentError, match='at its ledger'):
                A.preflight()
    finally:
        os.chmod(kept / 'ledger.jsonl', 0o644)
        os.chmod(kept, 0o755)


def test_a_run_that_cannot_start_says_why(toy, tmp_path, home, monkeypatch):  # noqa: F811
    monkeypatch.chdir(tmp_path)
    st = A.start(toy, options=['--strategy', 'no-such-strategy'])
    d = A.current(toy)
    final = st if st['state'] == 'failed' else A.wait_next(d, wait=60, poll=0.05)
    assert final['state'] == 'failed' and final['exit_code'] == 2, final
    assert any('no-such-strategy' in line for line in final['log_tail'])
    assert 'infrarsi agent start' in final['do']
    with pytest.raises(A.AgentError, match='nothing to resume'):
        A.resume(d)
    with pytest.raises(A.AgentError, match='no such file or directory'):
        A.start(tmp_path / 'missing')


def test_the_commands_as_a_session_types_them(toy, tmp_path, home):  # noqa: F811
    """`python -m infrarsi agent ...` as separate processes: one JSON object each, the exit status for a script."""
    env = {**os.environ, 'PYTHONPATH': str(REPO)}

    def cmd(*argv, cwd=toy):
        p = subprocess.run([sys.executable, '-m', 'infrarsi', 'agent', *map(str, argv)], cwd=cwd, env=env, capture_output=True,
                           text=True, encoding='utf-8', errors='replace', timeout=300)
        return p.returncode, (json.loads(p.stdout) if p.stdout.strip().startswith('{') else None), p.stderr
    code, out, err = cmd('status')
    assert code == 2 and 'no session under' in err and out is None
    drv, calls, _ = scripted(toy)
    code, out, err = cmd('start', '.', '--model', 'claude-opus-5-5', '--screens', '3', '--budget-tokens', '1000000', '--no-doctor')
    assert code == 0 and out['state'] in ('running', 'request'), (out, err)
    for _ in range(200):
        code, out, err = cmd('next', '--wait', '20')
        if out['state'] == 'request':
            r = S.find(out['session'], out['request']['id'])
            assert r.meta['max_attempts'] == 2 and out['request']['attempts_left'] == 2   # one re-ask, as the CLI backends
            Path(out['request']['answer']).write_text(json.dumps(drv.script(r.role, r.text('system.txt'), r.text('prompt.txt'))),
                                                      encoding='utf-8')
            code, got, err = cmd('answer', r.seq, '--tokens', '1500', '--by', 'claude-opus-5-5')
            assert code == 0 and got['ok'], (got, err)
        elif out['state'] != 'running':
            break
    assert out['state'] == 'finished' and code == 0, (out, err)
    code, st, _ = cmd('status', '--session', out['session'], cwd=tmp_path)
    assert code == 0 and st['state'] == 'finished' and st['result']['screens'] == 3
    code, got, err = cmd('answer', 1)
    assert code == 1 and got['ok'] is False and 'already answered' in got['error']
    code, _, err = cmd('start', '.', '--llm', 'claude-cli')   # the model is the session: not an option of start
    assert code == 2 and 'unrecognized arguments' in err
