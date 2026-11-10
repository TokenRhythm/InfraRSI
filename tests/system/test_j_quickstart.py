"""J1 / J5 / J10 - the README quick start as a user runs it on a repository without a manifest; the mistakes a user makes
on the way (one line on stderr, exit 2, nothing run); what `report` and `apply` accept; the lock a run holds (another run
may be active in the project)."""
import importlib
import json
import os
import pathlib
import re
import socket
import threading
import time

import pytest

from _support import cli
from _support.consistency import check_run
from _support.toy import latest_run, make_toy, manifest, report, runs, write_manifest
from infrarsi.project import Project


def test_readme_quick_start_end_to_end(tmp_path, home):
    root = make_toy(tmp_path / 'repo', manifest=False)
    # 1. init drafts the manifest and tells the user what to do next
    res = cli.run('init', root)
    assert res.code == 0, res.text
    assert res.out.startswith('wrote ') and 'next: review the file, then `infrarsi doctor`' in res.out
    assert (root / 'infrarsi.yaml').is_file()
    p = Project.load(root)
    assert len(p.knobs) >= 3 and p.raw['launch']['command']
    again = cli.run('init', root)
    assert again.code == 2 and 'exists' in again.err and 'Traceback' not in again.err
    # a short job for the test: the draft keeps its own steps otherwise (and saves its checkpoint half-way through them)
    doc = manifest(root)
    doc['workload'] = {'steps': 12, 'window': [5, 12], **({'invariants': doc['workload'].get('invariants', {})})}
    doc['checkpoint']['save_at'] = 6
    write_manifest(root, doc)
    # 2. doctor: the reference runs and the checklist coverage is shown
    res = cli.run('doctor', root)
    assert res.code == 0, res.text
    assert 'reference run: OK' in res.out and 'signals seen (updates of' in res.out and 'code patches' in res.out   # (U-1-9)
    assert re.search(r'^\s+pass\s+completes', res.out, re.M) and 'log:' in res.out
    # 3. run without a model (auto -> tpe with optuna installed)
    res = cli.run('run', root, '--llm', 'none', '--screens', '4')
    assert res.code == 0, res.text
    run = latest_run(root)
    r = check_run(run, budget_screens=4)
    assert r['strategy'] == 'tpe' and r['outcome'] in ('improved', 'no_change')
    assert res.out.count('[infrarsi]') >= 4 and str(run / 'REPORT.md') in res.out
    assert [ln for ln in res.out.splitlines() if ln.startswith(f'{p.name}: {r["outcome"]}')]
    # 4. report prints REPORT.md; without an argument, in the project directory, the newest run's (named on stderr)
    named = cli.run('report', run)
    assert named.out.strip() == report(run).strip() and named.err == ''
    res = cli.run('report', cwd=root)
    assert res.code == 0 and res.out == named.out and res.err == f'run: {run.relative_to(root)}\n', res.text
    # 5. apply: a dry run first, then --write when something was promoted
    dry = cli.run('apply', run)
    assert dry.code == 0, dry.text
    if r['outcome'] == 'improved':
        assert 'dry run: add --write' in dry.out
        wr = cli.run('apply', run, '--write')
        assert wr.code == 0 and ('written:' in wr.out or '(no file changes)' in wr.out), wr.text
        changed = {k: v for k, v in r['promotion']['config'].items() if r['reference']['config'].get(k) != v}
        path_knobs = [k for k in changed if 'path' in p.knobs.get(k, {})]
        if path_knobs:                                               # 6. the applied config is the new reference
            new = root / 'config.infrarsi.yaml'
            assert new.is_file()
            doc = manifest(root)
            doc['config']['file'] = new.name
            write_manifest(root, doc)
            ref2 = Project.load(root).reference_config()
            assert all(ref2[k] == changed[k] for k in path_knobs)
            res = cli.run('doctor', root)
            assert res.code == 0 and 'reference run: OK' in res.out
    else:
        assert dry.out.startswith('nothing to apply')


def test_init_with_the_jobs_command_line(tmp_path, home):
    root = make_toy(tmp_path / 'repo', manifest=False)
    res = cli.run('init', root, '--command', 'python train.py --config config.yaml --steps 30 --workers 2')
    assert res.code == 0, res.text
    launch = (root / 'infrarsi_launch.sh').read_text(encoding='utf-8')
    assert launch.startswith('#!/bin/bash') and 'python train.py --config config.yaml --steps 30 --workers 2 "$@"' in launch
    assert 'holds your command' in res.out
    p = Project.load(root)
    assert 'infrarsi_launch.sh' in p.raw['launch']['command']
    for bad, msg in (('python a.py | tee log', 'pipeline'), ('bash run.sh --x', 'shell script'), ('python "a.py', 'cannot be read')):
        res = cli.run('init', tmp_path / 'other', '--command', bad)
        assert res.code == 2 and msg in res.err and 'Traceback' not in res.err, (bad, res.text)
    res = cli.run('init', root, '--command', 'python train.py', '--entry', 'train.py')
    assert res.code == 2 and 'not both' in res.err


MISTAKES = {
    'stale_constraint': (lambda d: d.update(constraints=['gpus % (tp * pp) == 0']), ['doctor', 'run'], "unknown name 'tp'"),
    'window_outside_steps': (lambda d: d['workload'].update(window=[5, 99]), ['doctor', 'run'], 'must lie within'),
    'two_targets': (lambda d: d['knobs']['workers'].update(path='train.workers'), ['run'], 'exactly one target'),
    'tolerance_out_of_range': (lambda d: d['verify'].update(tolerance=3), ['run'], 'verify.tolerance'),
    'ssh_without_hosts': (lambda d: d.update(execution={'where': 'ssh'}), ['doctor'], 'execution.hosts is required'),
}


@pytest.mark.parametrize('name', sorted(MISTAKES))
def test_a_manifest_mistake_is_one_line_before_any_run(toy, name):
    edit, cmds, message = MISTAKES[name]
    doc = manifest(toy)
    edit(doc)
    write_manifest(toy, doc)
    for cmd in cmds:
        res = cli.run(cmd, toy, *(['--llm', 'none'] if cmd == 'run' else []))
        assert res.code == 2, (cmd, res.text)
        assert message in res.err and res.err.startswith(f'infrarsi {cmd}:') and 'Traceback' not in res.err, res.err
        assert res.err.count('\n') == 1, 'more than one line'
    assert runs(toy) == [] and not (toy / '.infrarsi' / 'doctor').exists()


@pytest.mark.parametrize('args,message', [
    (['--ablate', 'nothing'], '--ablate: unknown'),
    (['--llm', 'bogus:m'], 'unknown llm backend'),
    (['--llm', 'openai-compat'], 'needs a model name'),
])
def test_a_bad_request_is_one_line_before_any_run(toy, args, message):
    res = cli.run('run', toy, *args)
    assert res.code == 2, res.text
    assert message in res.err and 'Traceback' not in res.err, res.err
    assert runs(toy) == [], 'a run directory was created for a request that cannot be served'


def test_an_unknown_strategy_is_one_line_before_any_run(toy):
    """F-SYS-10 (fixed): the strategy name is checked before the reference run costs its GPU time."""
    res = cli.run('run', toy, '--llm', 'none', '--strategy', 'bogus')
    assert res.code == 2 and 'unknown search strategy' in res.err and 'Traceback' not in res.err
    assert runs(toy) == [], 'the reference ran before the strategy name was checked'


def test_a_tolerance_outside_the_range_is_one_line(toy):
    res = cli.run('run', toy, '--llm', 'none', '--strategy', 'random', '--tolerance', '1.5')
    assert res.code == 2 and '--tolerance must be in [0, 1)' in res.err and 'Traceback' not in res.err
    assert runs(toy) == []


def test_broken_or_missing_manifests(tmp_path, home, toy):
    res = cli.run('doctor', tmp_path / 'nowhere')
    assert res.code == 2 and 'does not exist' in res.err and '`infrarsi init`' in res.err
    (toy / 'infrarsi.yaml').write_text('name: [unclosed\n', encoding='utf-8')
    res = cli.run('run', toy, '--llm', 'none')
    assert res.code == 2 and 'not valid YAML' in res.err and 'Traceback' not in res.err
    (toy / 'infrarsi.yaml').write_text('schema: infrarsi-project-v9\nname: x\nlaunch: {command: x}\n', encoding='utf-8')
    res = cli.run('run', toy, '--llm', 'none')
    assert res.code == 2 and 'schema' in res.err


def test_debug_mode_shows_the_traceback(toy, monkeypatch):
    doc = manifest(toy)
    doc.update(constraints=['gpus % (tp * pp) == 0'])
    write_manifest(toy, doc)
    monkeypatch.setenv('INFRARSI_DEBUG', '1')
    with pytest.raises(Exception):
        cli.run('doctor', toy)


def test_a_project_path_with_spaces_and_non_ascii(tmp_path, home):
    root = make_toy(tmp_path / 'my repo (试验)' / 'toy')
    res = cli.run('doctor', root)
    assert res.code == 0, res.text
    res = cli.run('run', root, '--llm', 'none', '--strategy', 'random', '--screens', '2')
    assert res.code == 0, res.text
    check_run(latest_run(root), budget_screens=2)


def test_out_directory_inside_the_project_is_not_copied_into_candidate_trees(toy):
    res = cli.run('run', toy, '--llm', 'none', '--strategy', 'random', '--screens', '2', '--out', toy / 'myruns')
    assert res.code == 0, res.text
    run = sorted((toy / 'myruns').iterdir())[-1]
    check_run(run, budget_screens=2)
    trees = list(run.glob('attempts/**/tree/myruns'))
    assert not trees, 'the runs directory was copied into a candidate tree'


def _fake_run(root, name, files=('REPORT.md', 'result.json', 'run.log'), mtime=None):
    """A run directory under root/.infrarsi/runs holding `files` (its REPORT.md names it), modified at `mtime`."""
    run = root / '.infrarsi' / 'runs' / name
    run.mkdir(parents=True)
    for f in files:
        (run / f).write_text(f'# {name}\n' if f == 'REPORT.md' else '{}\n', encoding='utf-8')
    if mtime is not None:
        os.utime(run, (mtime, mtime))
    return run


def test_report_and_apply_find_the_run(toy):
    """U-1-18: without an argument the newest run (by modification time) under .infrarsi/runs of the current
    directory, for a project directory its newest run, for a file of a run that run; a run picked as the newest is
    named."""
    (toy / 'run.log').write_text('step 1 loss 2.0\n', encoding='utf-8')     # the job's own log: not a run
    for cmd in ('report', 'apply'):                                     # no run yet: one line, exit 2
        res = cli.run(cmd, cwd=toy)
        assert res.code == 2 and res.out == '' and res.err == (
            f'infrarsi {cmd}: no run under .infrarsi/runs: run `infrarsi run` first, or give a run directory\n')
    newer = _fake_run(toy, 'toy-cpu-20261026-090000', mtime=1_700_000_100)   # the earlier name, modified later
    older = _fake_run(toy, 'toy-cpu-20261026-100000', mtime=1_700_000_000)
    for args, cwd, named in (((), toy, newer.relative_to(toy)), ((toy,), None, newer)):   # in the project, from outside
        res = cli.run('report', *args, cwd=cwd)
        assert res.code == 0 and res.out == f'# {newer.name}\n\n' and res.err == f'run: {named}\n', res.text
    for f in ('REPORT.md', 'result.json', 'run.log'):                  # a file of a run: that run, named by the user
        res = cli.run('report', older / f)
        assert res.code == 0 and res.out == f'# {older.name}\n\n' and res.err == '', (f, res.text)
    running = _fake_run(toy, 'toy-cpu-20261026-110000', files=('run.log',))   # the newest has not finished
    res = cli.run('apply', cwd=toy)
    assert res.code == 2 and res.out == '' and res.err.count('\n') == 1, res.text
    assert f'{running.relative_to(toy)} did not finish: no result.json was written' in res.err
    res = cli.run('report', running / 'run.log')
    assert res.code == 2 and f'{running} did not finish: no REPORT.md was written' in res.err


WARNING = '! another run may be active in this project: '


def _lock(root):
    return root.resolve() / '.infrarsi' / 'run.lock'


def _write_lock(root, lock: dict) -> None:
    _lock(root).parent.mkdir(exist_ok=True)
    _lock(root).write_text(json.dumps(lock), encoding='utf-8')


def test_a_run_replaces_a_stale_lock_and_removes_its_own(toy, monkeypatch):
    """U-7-9: the lock of a run that crashed is named in one warning line and replaced: while this run measures the lock
    is its own (pid, host, start), and when the run ends it is gone."""
    from infrarsi.project import runner as runner_mod
    _write_lock(toy, {'pid': 999999, 'host': 'elsewhere', 'started_utc': '2026-10-25T08:00:00Z'})
    real, held = runner_mod.ProjectRunner.run, []

    def run(self, task, config, overlay, purpose, run_id, **kw):
        held.append(json.loads(_lock(toy).read_text(encoding='utf-8')) if _lock(toy).is_file() else None)
        return real(self, task, config, overlay, purpose, run_id, **kw)
    monkeypatch.setattr(runner_mod.ProjectRunner, 'run', run)
    res = cli.run('run', toy, '--llm', 'none', '--strategy', 'random', '--screens', '2')
    assert res.code == 0, res.text
    assert res.lines(WARNING) == [f'{WARNING}pid 999999 on elsewhere, started 2026-10-25T08:00:00Z - its load changes '
                                  f'the step times this run measures; delete {_lock(toy)} if that run is gone'], res.out
    assert held and all(h and set(h) == {'pid', 'host', 'started_utc'} and (h['pid'], h['host']) == (os.getpid(), socket.gethostname())
                        for h in held), held
    assert not _lock(toy).exists(), 'the run left its lock behind'


def test_an_interrupted_run_removes_its_lock(toy, monkeypatch):
    """Ctrl-C (or `kill`) during the reference run: exit status 130, and no lock left behind."""
    from infrarsi.project import runner as runner_mod
    held = []

    def run(self, task, config, overlay, purpose, run_id, **kw):
        held.append(_lock(toy).is_file())
        raise KeyboardInterrupt
    monkeypatch.setattr(runner_mod.ProjectRunner, 'run', run)
    res = cli.run('run', toy, '--llm', 'none', '--strategy', 'random')
    assert res.code == 130 and 'interrupted' in res.text, res.text
    assert held == [True] and not _lock(toy).exists()


def test_a_run_keeps_the_lock_of_a_run_started_after_it(toy, monkeypatch):
    """Two runs overlap and the earlier one ends first: the later run's lock stays (that run still measures)."""
    later = {'pid': 4343, 'host': 'elsewhere', 'started_utc': '2026-10-26T10:00:00Z'}

    def run_project(project, **kw):                     # another run starts meanwhile and writes its lock
        _write_lock(toy, later)
        return {'outcome': 'no_change', 'run_dir': str(toy), 'strategy': 'random', 'promotion': {}, 'search': {}}
    monkeypatch.setattr('infrarsi.project.flow.run_project', run_project)
    res = cli.run('run', toy, '--llm', 'none', '--strategy', 'random')
    assert res.code == 0, res.text
    assert json.loads(_lock(toy).read_text(encoding='utf-8')) == later


@pytest.mark.parametrize('stale', [False, True])
def test_a_lock_written_while_a_run_ends_survives(toy, monkeypatch, stale):
    """Review U6 r1 / r2: run B writes its lock between run A's ownership check and A's delete (both on the real lock
    path, B in a thread). The write and the check-and-delete each hold .infrarsi/run.lock.mutex, so B's write waits
    for A's delete and B's lock survives. stale: a crashed run's mutex (ten seconds old) is there when B starts and A
    ends; both find it old, B first, and B goes on from what it saw only once A has removed it and holds the section."""
    cli_main = importlib.import_module('infrarsi.cli.main')
    root, real_read, real_stat, me = toy.resolve(), cli_main._read_lock, pathlib.Path.stat, socket.gethostname()
    mutex, b = _lock(toy).with_name('run.lock.mutex'), []
    a_holds, b_saw_mutex, b_wrote, b_may_end = (threading.Event() for _ in range(4))

    def run_b():
        with cli_main._run_lock(root):
            b_wrote.set()
            b_may_end.wait(10)

    def start_b():
        monkeypatch.setattr(socket, 'gethostname', lambda: 'host-b')   # what tells B's lock from A's
        b.append(threading.Thread(target=run_b))
        b[0].start()

    def read(r):                                          # A's ownership check, in its section: B is to write now
        lock = real_read(r)
        if lock and lock.get('host') == me and threading.current_thread() is threading.main_thread() and not a_holds.is_set():
            a_holds.set()
            if not b:
                start_b()
            b_wrote.wait(0.3)                             # B waits for A's section (without one it writes now)
        return lock

    def stat(self, *a, **kw):                             # B's first look at the mutex: it goes on once A holds the section
        st = real_stat(self, *a, **kw)
        if self == mutex and b and threading.current_thread() is b[0] and not b_saw_mutex.is_set():
            b_saw_mutex.set()
            a_holds.wait(5)
        return st
    monkeypatch.setattr(cli_main, '_read_lock', read)
    monkeypatch.setattr(pathlib.Path, 'stat', stat)
    with cli_main._run_lock(root):                        # run A
        if stale:
            mutex.write_text('', encoding='utf-8')
            os.utime(mutex, (time.time() - 10,) * 2)
            start_b()
            assert b_saw_mutex.wait(5)
    assert b_wrote.wait(5), 'run B never wrote its lock'
    assert json.loads(_lock(toy).read_text(encoding='utf-8'))['host'] == 'host-b', 'run A deleted the lock of run B'
    b_may_end.set()
    b[0].join(5)
    assert not _lock(toy).exists() and not list(_lock(toy).parent.glob('run.lock.mutex*'))


def test_a_busy_lock_section_skips_the_lock_operation(toy):
    """Review U6 r2: a mutex that stays fresh for the whole wait (another run's; here its mtime is ahead of the clock, as
    on a file server whose clock runs ahead) makes the run skip that lock operation - the write, or the check and
    delete - and go on; the mutex is left in place. The lock is advisory."""
    cli_main = importlib.import_module('infrarsi.cli.main')
    root, mutex = toy.resolve(), _lock(toy).with_name('run.lock.mutex')

    def busy():
        mutex.write_text('', encoding='utf-8')
        os.utime(mutex, (time.time() + 60,) * 2)
    _lock(toy).parent.mkdir()
    busy()
    with cli_main._run_lock(root):                        # the write is skipped
        assert not _lock(toy).exists()
    assert mutex.exists() and not _lock(toy).exists()
    mutex.unlink()
    with cli_main._run_lock(root):                        # the lock is written; its check and delete are skipped
        assert json.loads(_lock(toy).read_text(encoding='utf-8'))['pid'] == os.getpid()
        busy()
    assert mutex.exists() and json.loads(_lock(toy).read_text(encoding='utf-8'))['pid'] == os.getpid()


def test_doctor_warns_about_a_run_lock_and_leaves_it(toy):
    """doctor started while a run's lock exists: the warning line first, the exit status of its run, the lock as it was
    (doctor's one short run writes none)."""
    lock = {'pid': 4242, 'host': 'elsewhere', 'started_utc': '2026-10-26T09:15:02Z'}
    _write_lock(toy, lock)
    res = cli.run('doctor', toy)
    assert res.code == 0 and 'reference run: OK' in res.out, res.text
    assert res.out.startswith(f'{WARNING}pid 4242 on elsewhere, started 2026-10-26T09:15:02Z - its load changes'), res.out
    assert json.loads(_lock(toy).read_text(encoding='utf-8')) == lock


@pytest.mark.parametrize('promoted', ['c3', None])
def test_an_interrupted_run_says_whether_its_recommendation_stands(toy, monkeypatch, promoted):
    """Finding 3 of U4's review: a run stopped while it wrote its results back, after its promotion, is `interrupted`
    (exit 130) and its recommendation stands: the console says so and names `apply` as the next step. A run stopped
    before any promotion keeps its words and gets no apply hint."""
    run_dir = toy / '.infrarsi' / 'runs' / 'toy-cpu-20261026-120000'
    monkeypatch.setattr('infrarsi.project.flow.run_project', lambda project, **kw: {
        'outcome': 'interrupted', 'interrupted_during': 'write-back' if promoted else 'search', 'run_dir': str(run_dir),
        'strategy': 'random', 'promotion': {'promoted': promoted, 'speedup': 1.5 if promoted else None}, 'search': {}})
    res = cli.run('run', toy, '--llm', 'none', '--strategy', 'random')
    assert res.code == 130, res.text
    said = (', x1.500 (re-run) (stopped after the promotion: the verified recommendation stands)' if promoted else
            ' (stopped before the end: nothing was promoted)')
    nxt = f'next: infrarsi report {run_dir}' + (f'; infrarsi apply {run_dir} shows the change (--write writes it)'
                                                  if promoted else '')
    assert res.out.splitlines() == [f'toy-cpu: interrupted{said} -> {run_dir / "REPORT.md"}', nxt], res.out
