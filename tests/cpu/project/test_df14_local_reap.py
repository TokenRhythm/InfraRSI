"""D2-R204d (2026-10-25): a Wan job launched through `accelerate launch` ran its workers in sessions of their own; the
timeout killed the launcher's session, the workers' orphans kept the output pipe (and a GPU), and the run's read never
ended - 80 min past the timeout until they were stopped by hand. The local executor tags the job's processes
(INFRARSI_RUN_TREE) and reaps what left the session: after a kill, and after the run."""
import os
import subprocess
import threading
import time

import pytest

from infrarsi.project import executor as ex

pytestmark = pytest.mark.skipif(os.name == 'nt' or not os.path.isdir('/proc'), reason='POSIX sessions and /proc')


def _run(tmp_path, cmd, timeout_s):
    tree = tmp_path / 'tree'
    tree.mkdir(exist_ok=True)
    out = {}

    def go():
        out.update(ex.LocalExecutor().run(cmd, cwd=str(tree), env={}, timeout_s=timeout_s, log_path=tmp_path / 'train.log'))
    th = threading.Thread(target=go, daemon=True)
    t0 = time.monotonic()
    th.start()
    th.join(60)
    return out, time.monotonic() - t0


def _alive(marker):
    r = subprocess.run(['pgrep', '-f', marker], capture_output=True, text=True)
    return r.stdout.split()


def test_a_timeout_ends_the_run_when_an_escaped_process_holds_its_output(tmp_path):
    marker = f'sleep 3{os.getpid() % 1000:03d}'                  # a sleep length of its own: found again by pgrep
    res, wall = _run(tmp_path, f'setsid {marker} & echo started; sleep 600', timeout_s=2)
    assert res, 'the run never ended'
    assert res['timed_out'] and wall < 40 and not _alive(marker)


def test_what_outlives_a_finished_run_is_reaped(tmp_path):
    marker = f'sleep 4{os.getpid() % 1000:03d}'
    res, wall = _run(tmp_path, f'setsid {marker} > /dev/null 2>&1 < /dev/null & echo done', timeout_s=60)
    assert res['exit_code'] == 0 and not res['timed_out'] and wall < 30 and not _alive(marker)


def test_other_processes_are_left_alone(tmp_path):
    (tmp_path / 'tree').mkdir()
    others = [subprocess.Popen(['sleep', '30'], cwd=str(tmp_path)),           # outside the tree, without the tag
              subprocess.Popen(['sleep', '30'], cwd=str(tmp_path / 'tree'))]  # the user's shell in the tree: no tag
    try:
        res, _ = _run(tmp_path, 'echo hi', timeout_s=30)
        assert res['exit_code'] == 0 and all(o.poll() is None for o in others)
    finally:
        for o in others:
            o.kill()


def test_a_reap_reaches_its_own_launch_only(tmp_path):
    # DF14_r1 #2: a timeout callback still in its kill when the run has returned reaps later; another launch in the same
    # tree (the next command) carries another tag
    mine = subprocess.Popen(['sleep', '30'], cwd=str(tmp_path), env={**os.environ, 'INFRARSI_LAUNCH': 'b'})
    try:
        ex._reap_local('a')
        assert mine.poll() is None
        ex._reap_local('b')
        assert mine.wait(10) is not None
    finally:
        mine.kill()


def test_an_interrupt_in_the_grace_kills_at_once(tmp_path, monkeypatch):
    # DF14_r1 #3: Ctrl-C while the reap waits for a TERM-resistant worker: SIGKILL before the interrupt goes on
    w = subprocess.Popen(['bash', '-c', "trap '' TERM; while :; do sleep 1; done"], cwd=str(tmp_path),
                         env={**os.environ, 'INFRARSI_LAUNCH': 'c'})
    time.sleep(0.5)

    def interrupt(s):
        raise KeyboardInterrupt
    monkeypatch.setattr(ex.time, 'sleep', interrupt)
    try:
        with pytest.raises(KeyboardInterrupt):
            ex._reap_local('c')
        monkeypatch.undo()
        assert w.wait(10) is not None
    finally:
        w.kill()
