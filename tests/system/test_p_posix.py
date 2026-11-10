"""P1/P2 - POSIX process semantics and the ssh executor, on a Linux node: a termination signal during the search kills the
job and leaves a report; a timed-out job's whole session is gone; orphans of a launch are reaped by their tag; the ssh
executor against localhost; an unreachable host is an infrastructure failure; a CRLF launcher is named."""
import os
import shutil
import signal
import subprocess
import sys
import time

import pytest
import yaml

from _support import cli
from _support.faultjob import make_project
from _support.toy import latest_run, make_toy, manifest, report, result, write_manifest

pytestmark = pytest.mark.posix


def _jobs_under(root) -> list[str]:
    """Processes whose working directory lies under `root` (the job's candidate trees): the launched command names
    train.py and config.json by relative paths, so the command line alone cannot tell this test's jobs apart."""
    out = []
    for d in os.listdir('/proc'):
        if not d.isdigit() or int(d) == os.getpid():
            continue
        try:
            cwd = os.readlink(f'/proc/{d}/cwd')
            cmd = open(f'/proc/{d}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            continue
        if cwd.startswith(str(root)) and 'pytest' not in cmd:
            out.append(f'{d} {cmd[:100]}')
    return out


def test_sigterm_during_the_search_kills_the_job_and_writes_the_report(tmp_path, home):
    root = make_project(tmp_path / 'sig', modes=('ok', 'fast'), only_mode=True, steps=40, timeout_s=60, screens=6)
    env = {**os.environ, 'FAULT_STEP_S': '0.2'}
    proc = subprocess.Popen([sys.executable, '-m', 'infrarsi', 'run', str(root), '--llm', 'none', '--strategy', 'random'],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    deadline = time.time() + 120
    log = None
    while time.time() < deadline:
        runs = sorted((root / '.infrarsi' / 'runs').glob('*/run.log'))
        if runs and 'screen ->' in runs[-1].read_text(encoding='utf-8'):
            log = runs[-1]
            break
        time.sleep(0.5)
    assert log is not None, 'the search never started'
    time.sleep(1.0)                                                  # the screen's job is running now
    before = _jobs_under(root)
    assert before, 'no job process to kill'
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 130, out
    assert 'interrupted' in out
    time.sleep(1.0)
    left = _jobs_under(root)
    assert not left, f'job processes survived the termination: {left} (before: {before})'
    run = latest_run(root)
    r = result(run)
    assert r['outcome'] == 'interrupted' and '## Interrupted' in report(run)
    assert cli.run('report', run).code == 0


def test_a_timed_out_hang_leaves_no_process(tmp_path, home):
    root = make_project(tmp_path / 'hang', default_mode='hang', timeout_s=3)
    res = cli.run('doctor', root)
    assert res.code == 1 and 'killed at the' in res.out
    time.sleep(0.5)
    left = _jobs_under(root)
    assert not left, left


def test_reap_local_kills_a_child_that_left_the_session(tmp_path):
    from infrarsi.project.executor import _reap_local
    tag = 'systest-' + os.urandom(4).hex()
    # the child leaves the launch's session by itself (as accelerate / torchrun workers do) and keeps the launch tag
    code = 'import os, time\nos.setsid()\ntime.sleep(300)\n'
    child = subprocess.Popen([sys.executable, '-c', code], env={**os.environ, 'INFRARSI_LAUNCH': tag})
    time.sleep(0.8)
    assert child.poll() is None, 'the stand-in orphan did not start'
    _reap_local(tag, grace_s=2.0)
    assert child.wait(timeout=10) is not None


def _ssh_localhost_ok() -> bool:
    if shutil.which('ssh') is None:
        return False
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', 'localhost', 'true'], capture_output=True, timeout=20)
    return r.returncode == 0


@pytest.mark.skipif(not _ssh_localhost_ok(), reason='ssh localhost without a password is not set up here')
def test_ssh_executor_to_localhost_runs_doctor_and_a_short_search(tmp_path, home):
    root = make_toy(tmp_path / 'toy', screens=2)
    doc = manifest(root)
    doc['execution'] = {'where': 'ssh', 'hosts': ['localhost'], 'remote_dir': str(tmp_path / 'remote'), 'python': sys.executable}
    doc['launch']['env'] = {'TOY_TIME_SCALE': os.environ.get('TOY_TIME_SCALE', '0.004')}
    write_manifest(root, doc)
    res = cli.run('doctor', root)
    assert res.code == 0, res.text
    assert 'reference run: OK' in res.out
    res = cli.run('run', root, '--llm', 'none', '--strategy', 'random')
    assert res.code == 0, res.text
    r = result(latest_run(root))
    assert r['outcome'] in ('improved', 'no_change') and r['project']['execution'] == 'ssh'
    assert (tmp_path / 'remote').is_dir()


@pytest.mark.skipif(not _ssh_localhost_ok(), reason='ssh localhost without a password is not set up here')
def test_two_hosts_render_the_multi_node_placeholders(tmp_path, home):
    root = tmp_path / 'mn'
    root.mkdir()
    (root / 'job.py').write_text('import sys, time\nrank, n, master = sys.argv[1:4]\n'
                                 'for s in range(1, 13):\n    time.sleep(0.02)\n'
                                 '    print(f"step {s} loss {2.5 - 0.01 * s:.4f} tokens 4096 rank {rank} of {n} master {master}", flush=True)\n',
                                 encoding='utf-8')
    doc = {'schema': 'infrarsi-project-v1', 'name': 'mn', 'launch': {'command': '{python} job.py {node_rank} {nnodes} {master_addr} {args}', 'timeout_s': 60},
           'knobs': {'w': {'type': 'int', 'choices': [1, 2], 'flag': '--w', 'default': 1}}, 'workload': {'steps': 12, 'window': [5, 12]},
           'hardware': {'gpus': 1}, 'execution': {'where': 'ssh', 'hosts': ['localhost', '127.0.0.1'], 'remote_dir': str(tmp_path / 'remote'),
                                                  'python': sys.executable, 'master_addr': '127.0.0.1'},
           'verify': {'required': ['completes', 'loss_within_tolerance']}}
    (root / 'infrarsi.yaml').write_text(yaml.safe_dump(doc, sort_keys=False), encoding='utf-8')
    res = cli.run('doctor', root)
    assert 'Traceback' not in res.text, res.text
    logs = sorted((root / '.infrarsi' / 'doctor').glob('**/train.log'))
    assert logs
    text = logs[-1].read_text(encoding='utf-8')
    assert 'rank 0 of 2 master 127.0.0.1' in text


def test_an_unreachable_host_is_an_infrastructure_failure(tmp_path, home, monkeypatch):
    from infrarsi.project import runner as runner_mod
    from infrarsi.project.flow import run_project
    monkeypatch.setattr(runner_mod.ProjectRunner, 'infra_retry_waits', (0.1, 0.1))
    root = make_toy(tmp_path / 'toy')
    doc = manifest(root)
    doc['execution'] = {'where': 'ssh', 'hosts': ['no-such-host.invalid'], 'remote_dir': '/tmp/x'}
    write_manifest(root, doc)
    from infrarsi.project import Project
    r = run_project(Project.load(root), strategy='random', budget={'screens': 2}, log=lambda m: None)
    assert r['outcome'] == 'infrastructure_failure'
    assert 'could not be reached' in report(r['run_dir'])
    res = cli.run('doctor', root)
    assert res.code != 0 and 'could not be reached' in res.text and 'Traceback' not in res.text


def test_a_crlf_launcher_is_named_with_the_cause(tmp_path, home):
    root = make_toy(tmp_path / 'toy')
    (root / 'run.sh').write_bytes(b'#!/bin/bash\r\nexec python train.py "$@"\r\n')
    doc = manifest(root)
    doc['launch']['command'] = 'bash run.sh --config {config} --steps {steps} {args}'
    write_manifest(root, doc)
    res = cli.run('doctor', root)
    assert res.code == 1 and 'CRLF' in res.out and 'dos2unix' in res.out, res.text
    res = cli.run('run', root, '--llm', 'none', '--strategy', 'random', '--screens', '2')
    assert res.code == 1
    assert 'CRLF' in report(latest_run(root))
