"""D2-R201 (2026-10-24, two unicom H200 nodes over TCP, remote_dir on the /data both nodes share): the candidate tree
was extracted once per host, so host 1's copy replaced the files host 0 had already looked up - node 0's launcher
failed at once ('Stale file handle', exit 126) - and node 1 waited 30 minutes at the rendezvous for it. A host that
already sees the copy is not copied to again; after node 0's job failed, the other nodes are stopped and reaped."""
import subprocess
import sys
import time

from infrarsi.project import executor as ex


def test_a_tree_on_storage_the_hosts_share_is_extracted_once(tmp_path, monkeypatch):
    fs = {'h1': tmp_path / 'shared', 'h2': tmp_path / 'shared', 'h3': tmp_path / 'own'}   # h1 and h2 share the disk
    copies, real_popen, real_run = [], subprocess.Popen, subprocess.run

    def popen(argv, **kw):                             # `ssh host 'mkdir -p R && tar -xzf - -C R'`, extracted locally
        copies.append(argv[-2])
        fs[argv[-2]].mkdir(parents=True, exist_ok=True)
        prog = 'import sys, tarfile; tarfile.open(fileobj=sys.stdin.buffer, mode="r|gz").extractall(sys.argv[1])'
        return real_popen([sys.executable, '-c', prog, str(fs[argv[-2]])], **kw)

    def run(argv, **kw):                               # `ssh host 'cat R/.infrarsi_placed'`
        p = fs[argv[-2]] / ex.PLACED
        return subprocess.CompletedProcess(argv, 0, stdout=p.read_text() if p.exists() else '', stderr='')
    monkeypatch.setattr(ex.subprocess, 'Popen', popen)
    monkeypatch.setattr(ex.subprocess, 'run', run)
    tree = tmp_path / 'tree'
    tree.mkdir()
    (tree / 'run.sh').write_text('echo hi\n')
    remote = ex.SSHExecutor(['h1', 'h2', 'h3'], '/r').place(tree, 'run1/c0')
    assert remote == '/r/run1/c0' and copies == ['h1', 'h3']        # h2 reads h1's token: no second extraction
    assert (fs['h1'] / 'run.sh').exists() and (fs['h3'] / ex.STAMPER).exists()
    copies.clear()
    ex.SSHExecutor(['h1', 'h2'], '/r').place(tree, 'run1/c1')       # the next copy has its own token
    assert copies == ['h1']


def test_after_node_0_failed_the_other_nodes_are_stopped_and_reaped(tmp_path, monkeypatch):
    progs, killed, reaped = iter([f'print("{ex.END_MARK} 126")',                       # node 0: its launcher failed
                                  f'import time; time.sleep(60); print("{ex.END_MARK} 0")']), [], []   # node 1: waits

    def fake_popen(argv, **kw):
        return subprocess.Popen([sys.executable, '-c', next(progs)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)

    def kill(p, grace_s=10.0):
        killed.append(p.pid)
        p.kill()
    monkeypatch.setattr(ex, '_popen', fake_popen)
    monkeypatch.setattr(ex, '_kill_tree', kill)
    monkeypatch.setattr(ex.SSHExecutor, '_reap', lambda self, i, cwd, grace_s=30: reaped.append(i))
    monkeypatch.setattr(ex, 'WORKER_GRACE_S', 1.0)
    t0 = time.monotonic()
    res = ex.SSHExecutor(['h1', 'h2'], '/r').run('train', cwd='/r/t', env={}, timeout_s=600, log_path=tmp_path / 'a.log')
    assert res['exit_code'] == 126 and len(killed) == 1 and reaped == [0, 1]
    assert time.monotonic() - t0 < 30                                # not node 1's own minute (30 in the trial)


def test_a_silent_node_0_connection_is_still_a_transport_failure(tmp_path, monkeypatch):
    # DF7_r2 #1: stopping the waiting worker must not hide that node 0's ssh died without a word (exit 255)
    import pytest
    from infrarsi.contracts import InfrastructureError
    progs, killed, reaped = iter(['import sys; sys.exit(255)',
                                  f'import time; time.sleep(60); print("{ex.END_MARK} 0")']), [], []

    def fake_popen(argv, **kw):
        return subprocess.Popen([sys.executable, '-c', next(progs)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)

    def kill(p, grace_s=10.0):
        killed.append(p.pid)
        p.kill()
    monkeypatch.setattr(ex, '_popen', fake_popen)
    monkeypatch.setattr(ex, '_kill_tree', kill)
    monkeypatch.setattr(ex.SSHExecutor, '_reap', lambda self, i, cwd, grace_s=30: reaped.append(i))
    monkeypatch.setattr(ex, 'WORKER_GRACE_S', 1.0)
    with pytest.raises(InfrastructureError, match='ssh to h1 failed: no output'):
        ex.SSHExecutor(['h1', 'h2'], '/r').run('train', cwd='/r/t', env={}, timeout_s=600, log_path=tmp_path / 'a.log')
    assert killed                                                   # the waiting worker was stopped
