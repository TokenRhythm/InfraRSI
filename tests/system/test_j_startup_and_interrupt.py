"""Journeys of the startup questions and of the interrupted run on a real terminal and with real signals (Linux): the questions
under a pseudo-terminal, a pipe that is asked nothing, Ctrl-C (SIGINT) and SIGTERM during the search, a killed process and
`infrarsi finish`."""
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from _support.toy import latest_run, make_toy

pytestmark = pytest.mark.posix


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    return {**os.environ, 'INFRARSI_HOME': str(tmp_path / 'home'), 'TOY_TIME_SCALE': '0.03', 'PYTHONUNBUFFERED': '1'}


def toy_project(tmp_path):
    def edit(doc):
        doc['knobs']['precision']['choices'] = ['fp32', 'bf16']         # (no trap: every candidate valid and faster)
        doc['knobs']['micro_batch_size']['choices'] = [1, 2, 4]
        doc['search']['run_noise'] = 0.01
    return make_toy(tmp_path / 'toy', steps=12, window=(5, 12), save_at=6, screens=12, edit=edit)


class Terminal:
    """`python -m infrarsi ...` on a real pseudo-terminal: what a person sees and types."""

    def __init__(self, argv, cwd, env):
        import pty
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.chdir(cwd)
            os.environ.clear()
            os.environ.update(env)
            os.execv(sys.executable, [sys.executable, '-m', 'infrarsi', *map(str, argv)])
        self.text = ''

    def read_until(self, *words, timeout=120.0) -> str:
        """Everything printed up to one of `words` (the first that appears), or the end of the process."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(w in self.text for w in words):
                return next(w for w in words if w in self.text)
            ready, _, _ = select.select([self.fd], [], [], 0.2)
            if ready:
                try:
                    chunk = os.read(self.fd, 65536).decode('utf-8', 'replace')
                except OSError:
                    return ''
                if not chunk:
                    return ''
                self.text += chunk
        raise AssertionError(f'timed out waiting for {words}:\n{self.text[-1500:]}')

    def type(self, line: str) -> None:
        os.write(self.fd, (line + '\n').encode('utf-8'))

    def wait(self, timeout=240.0) -> int:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            done, status = os.waitpid(self.pid, os.WNOHANG)
            ready, _, _ = select.select([self.fd], [], [], 0.2)
            if ready:
                try:
                    self.text += os.read(self.fd, 65536).decode('utf-8', 'replace')
                except OSError:
                    pass
            if done:
                return os.waitstatus_to_exitcode(status)
        os.kill(self.pid, signal.SIGKILL)
        raise AssertionError('timed out:\n' + self.text[-1500:])

    def press_enter_through(self) -> int:
        """Enter at every question and at the go-ahead; how many questions were asked."""
        n = 0
        while True:
            seen = self.read_until(']: ', 'Start? [Enter', timeout=120)
            if seen == 'Start? [Enter':
                self.type('')
                self.text = self.text.replace('Start? [Enter', 'Start?-')
                return n
            self.type('')
            self.text = self.text.replace(']: ', '] ')
            n += 1


# ---------------------------------------------------------------- the questions on a real terminal
def test_enter_through_the_questions_on_a_terminal_runs_the_defaults(tmp_path, env):
    p = toy_project(tmp_path)
    t = Terminal(['run', p, '--llm', 'none', '--strategy', 'random', '--screens', '2'], tmp_path, env)
    asked = t.press_enter_through()
    code = t.wait()
    assert code == 0, t.text[-2000:]
    assert asked == 0 and 'Loss tolerance: 0.03' in t.text and 'six settings' in t.text
    r = json.loads((latest_run(p) / 'result.json').read_text(encoding='utf-8'))
    assert r['strategy'] == 'random' and r['search']['used']['screens'] == 2
    assert not (p / 'infrarsi.yaml.bak').exists()        # Enter changed nothing


def test_a_at_the_first_question_accepts_everything(tmp_path, env):
    p = toy_project(tmp_path)
    t = Terminal(['run', p, '--llm', 'none', '--strategy', 'random', '--screens', '1'], tmp_path, env)
    t.read_until('Start? [Enter')
    t.type('walk')
    t.text = t.text.replace('Start? [Enter', 'Start?-')
    t.read_until(']: ')
    t.type('a')
    t.read_until('Start? [Enter')
    t.type('')
    assert t.wait() == 0, t.text[-1500:]


def test_q_on_a_terminal_runs_nothing(tmp_path, env):
    p = toy_project(tmp_path)
    t = Terminal(['run', p, '--llm', 'none', '--strategy', 'random', '--screens', '2'], tmp_path, env)
    t.read_until('Start? [Enter')
    t.type('q')
    assert t.wait() == 2 and 'stopped at the questions' in t.text
    assert not (p / '.infrarsi' / 'runs').exists()


def test_an_answer_on_a_terminal_reaches_the_run(tmp_path, env):
    p = toy_project(tmp_path)
    t = Terminal(['run', p, '--llm', 'none', '--strategy', 'random'], tmp_path, env)
    t.read_until('Start? [Enter')
    t.type('4')
    t.read_until(']: ')
    t.type('2')
    t.text = t.text.replace('Start? [Enter', 'Start?-')
    t.read_until('Start? [Enter')
    t.type('')
    assert t.wait() == 0, t.text[-1500:]
    assert json.loads((latest_run(p) / 'result.json').read_text(encoding='utf-8'))['search']['used']['screens'] == 2


def test_a_pipe_is_asked_nothing(tmp_path, env):
    p = toy_project(tmp_path)
    got = subprocess.run([sys.executable, '-m', 'infrarsi', 'run', str(p), '--llm', 'none', '--strategy', 'random', '--screens', '1'],
                         input='', capture_output=True, text=True, env=env, timeout=200)
    assert got.returncode == 0 and '[1/' not in got.stdout and 'Start?' not in got.stdout, got.stdout[-1000:]


def test_yes_skips_the_questions_on_a_terminal(tmp_path, env):
    p = toy_project(tmp_path)
    t = Terminal(['run', p, '--llm', 'none', '--strategy', 'random', '--screens', '1', '-y'], tmp_path, env)
    assert t.wait() == 0 and 'Start?' not in t.text


# ---------------------------------------------------------------- signals during the search
def start(p, env, *extra):
    return subprocess.Popen([sys.executable, '-m', 'infrarsi', 'optimize', str(p), '--llm', 'none', '--strategy', 'random', '--screens', '12',
                             '--no-doctor', '-y', *extra], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, start_new_session=True)


def read_to(proc, word, seen: list, timeout=180.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        seen.append(line)
        if word in line:
            return
    raise AssertionError(f'never saw {word!r}:\n' + ''.join(seen[-40:]))


def finish(proc, seen: list, timeout=300) -> int:
    out, _ = proc.communicate(timeout=timeout)
    seen.append(out or '')
    return proc.returncode


@pytest.mark.parametrize('sig', [signal.SIGINT, signal.SIGTERM])
def test_a_signal_during_the_search_reverifies_and_adopts(tmp_path, env, sig):
    p = toy_project(tmp_path)
    proc, seen = start(p, env), []
    read_to(proc, 'run 3 of', seen)
    proc.send_signal(sig)
    code = finish(proc, seen)
    text = ''.join(seen)
    assert code == 0, text[-2500:]
    assert 're-verifying the best candidate' in text and 'stage 5/5 (apply): written' in text and (p / 'infrarsi_optimized.sh').is_file()
    r = json.loads((latest_run(p) / 'result.json').read_text(encoding='utf-8'))
    assert r['search']['stop_reason'] == 'user_stop' and r['outcome'] == 'improved' and r['promotion']['promoted']


def test_the_job_of_the_cut_run_is_gone(tmp_path, env):
    # the toy runs in its own session: the interrupt must stop it, not leave it running
    p = toy_project(tmp_path)
    proc, seen = start(p, env), []
    read_to(proc, 'run 2 of', seen)
    proc.send_signal(signal.SIGINT)
    finish(proc, seen)
    time.sleep(1)
    left = subprocess.run(['pgrep', '-f', str(p / '.infrarsi')], capture_output=True, text=True).stdout.split()
    assert not [x for x in left if int(x) != os.getpid()]


def test_two_interrupts_stop_without_the_reverification(tmp_path, env):
    p = toy_project(tmp_path)
    proc, seen = start(p, env), []
    read_to(proc, 'run 3 of', seen)
    proc.send_signal(signal.SIGINT)
    time.sleep(0.2)
    proc.send_signal(signal.SIGINT)
    code = finish(proc, seen)
    text = ''.join(seen)
    assert code == 130, text[-2000:]
    assert not (p / 'infrarsi_optimized.sh').exists()    # (the second signal may land before or during the re-verification)
    r = json.loads((latest_run(p) / 'result.json').read_text(encoding='utf-8'))
    assert r['outcome'] == 'interrupted' and r['promotion']['promoted'] is None


def test_a_killed_run_is_finished_later(tmp_path, env):
    p = toy_project(tmp_path)
    proc, seen = start(p, env), []
    read_to(proc, 'run 3 of', seen)
    proc.kill()
    finish(proc, seen)
    run = latest_run(p)
    assert not (run / 'result.json').exists()            # the process had no chance to write one
    got = subprocess.run([sys.executable, '-m', 'infrarsi', 'finish', str(run), '--dry-run'], capture_output=True, text=True, env=env, timeout=300)
    assert got.returncode == 0, got.stdout[-2000:] + got.stderr[-500:]
    assert 'improved' in got.stdout and 'dry run' in got.stdout
    r = json.loads((run / 'result.json').read_text(encoding='utf-8'))
    assert r['finished'] and r['search']['stop_reason'] == 'process_ended' and r['promotion']['promoted']


def test_ctrl_g_live_conversation_during_a_real_search(tmp_path, env):
    p = toy_project(tmp_path)
    t = Terminal(['run', p, '--llm', 'none', '--strategy', 'bcomp', '--screens', '4'], tmp_path,
                 {**env, 'TOY_TIME_SCALE': '0.08'})
    t.press_enter_through()
    t.read_until('Ctrl-G: open chat')
    os.write(t.fd, b'\x07')
    t.read_until('chat> ')
    t.type('/status')
    assert t.read_until('Completed ') == 'Completed ', t.text[-2000:]
    t.type('/pause')
    t.read_until('Pause requested:')
    t.type('/note Prefer lower communication overhead')
    t.read_until('Guidance queued')
    t.type('/resume')
    t.read_until('Continuing the search.')
    t.read_until('Guidance saved for future')
    t.type('/stop')
    t.read_until('Stop requested:')
    os.write(t.fd, b'\x1b')
    assert t.wait() == 0, t.text[-2000:]
    run = latest_run(p)
    events = [json.loads(line) for line in (run / 'search/events.jsonl').read_text().splitlines()]
    assert any(e['kind'] == 'live_message' and 'communication' in str(e) for e in events)
    assert (run / 'search/state/checkpoint.pkl').exists()
    assert 'Resume: infrarsi run --resume' in t.text


def test_failed_launch_can_be_repaired_and_retried_in_same_invocation(tmp_path, env):
    import yaml
    p = toy_project(tmp_path)
    manifest = p / 'infrarsi.yaml'
    doc = yaml.safe_load(manifest.read_text())
    working = doc['launch']['command']
    doc['launch']['command'] = 'python missing_training_script.py'
    manifest.write_text(yaml.safe_dump(doc))
    t = Terminal(['optimize', p, '--llm', 'none', '--strategy', 'random', '--screens', '1', '--no-doctor', '--dry-run'], tmp_path, env)
    t.press_enter_through()
    assert t.read_until('Recovery [r:') == 'Recovery [r:', t.text[-2000:]
    t.type('m')
    assert t.read_until('Working launch command') == 'Working launch command'
    t.type(working)
    assert t.read_until('Save this command and retry the baseline?') == 'Save this command and retry the baseline?'
    t.type('y')
    assert t.wait() == 0, t.text[-2000:]
    assert yaml.safe_load(manifest.read_text())['launch']['command'] == working
    results = [json.loads(f.read_text()) for f in (p / '.infrarsi/runs').glob('*/result.json')]
    assert len(results) == 2 and sum(r['outcome'] == 'reference_failed' for r in results) == 1
    assert any((r.get('search') or {}).get('used', {}).get('screens') == 1 for r in results)
