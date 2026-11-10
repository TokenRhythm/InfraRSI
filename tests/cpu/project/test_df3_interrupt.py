"""The failure bundle R1-18 (2026-10-24): Ctrl-C during a search left the job running (it runs in its own session, so
the terminal's SIGINT never reaches it) and the interrupted run had no report (`infrarsi report` refused it)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from infrarsi.project import Project
from infrarsi.project import executor as ex
from infrarsi.project import runner as runner_mod
from infrarsi.project.flow import run_project

from test_project import toy  # noqa: F401 - the toy project fixture


def test_ctrl_c_while_a_job_runs_kills_it(tmp_path):
    proc = ex._popen([sys.executable, '-c', 'import time; print("step 1 loss 2.0", flush=True); time.sleep(60)'])

    def stop(lines):                                    # the interrupt arrives while the job's output is read
        if lines:
            raise KeyboardInterrupt
        return False
    with pytest.raises(KeyboardInterrupt):
        ex.stream(proc, timeout_s=60, log_path=tmp_path / 'job.log', stop=stop, poll_s=0)
    assert proc.wait(timeout=15) is not None


class Interrupting(ex.LocalExecutor):
    """Ctrl-C arrives as the search's second run starts."""

    def __init__(self, state):
        super().__init__()
        self.state = state

    def place(self, tree, rel):
        if self.state['search']:
            self.state['n'] += 1
            if self.state['n'] == 2:
                raise KeyboardInterrupt
        return super().place(tree, rel)


def test_an_interrupted_search_is_reported(toy, tmp_path, monkeypatch):
    state = {'search': False, 'n': 0}
    monkeypatch.setattr(runner_mod.ex, 'make', lambda execution: Interrupting(state))

    def log(m):
        if m.startswith('[infrarsi] search:'):
            state['search'] = True
    res = run_project(toy, strategy='random', budget={'screens': 4}, out_root=tmp_path / 'out', log=log, on_interrupt='stop')
    assert res['outcome'] == 'interrupted' and res['promotion']['promoted'] is None
    assert res['interrupted_during'] == 'search'
    text = (Path(res['run_dir']) / 'REPORT.md').read_text(encoding='utf-8')
    assert '## Interrupted' in text and 'after 1 run(s)' in text and (Path(res['run_dir']) / 'result.json').is_file()


def _without_traps(toy, writes_back=False):
    """The toy without its traps (fp8 moves the loss, a micro-batch of 8 runs out of memory): every candidate is valid
    and faster than the reference, so the promotion has candidates to re-run whatever the strategy draws. writes_back:
    also a 2 % noise margin (search.run_noise: the fastest candidate passes its promotion) and the facts a pack needs."""
    doc = yaml.safe_load((toy.root / 'infrarsi.yaml').read_text(encoding='utf-8'))
    doc['knobs']['precision']['choices'] = ['fp32', 'bf16']
    doc['knobs']['micro_batch_size']['choices'] = [1, 2, 4]
    if writes_back:
        doc['search']['run_noise'] = 0.01
        doc['hardware']['gpu_type'] = 'H100-SXM-80GB'
        doc['model'] = {'params_b': 0.001}
        doc['workload']['seq_len'] = 512
    (toy.root / 'infrarsi.yaml').write_text(yaml.safe_dump(doc), encoding='utf-8')
    return Project.load(toy.root)


def _ranked(res) -> list:
    """The candidates the promotion re-runs, fastest first (flow._promote's order)."""
    t_ref = res['reference']['step_time']
    return [c['id'] for c in sorted((c for c in res['candidates'] if c['id'] != 'c0' and c.get('valid')
                                     and c.get('step_time') and c['step_time'] < t_ref), key=lambda c: c['step_time'])]


def _run(project, tmp_path, screens, log=lambda m: None, strategy='random') -> dict:
    """A search of `screens` runs; an interrupt that escapes run_project fails the test (pytest would stop)."""
    try:
        return run_project(project, strategy=strategy, budget={'screens': screens}, out_root=tmp_path / 'out', log=log)
    except KeyboardInterrupt:
        pytest.fail('the interrupt propagated out of run_project: no REPORT.md, no result.json')


def test_an_interrupted_promotion_is_reported(toy, tmp_path, monkeypatch):
    # U-7-3: Ctrl-C during the promotion propagated out of run_project: no REPORT.md, no result.json. Here the first
    # candidate fails its tests and Ctrl-C arrives as the second one's fresh re-run starts.
    real_run = runner_mod.ProjectRunner.run
    state, rescores, tested = {'promotion': False}, [], []

    def run(self, task, config, overlay, purpose, run_id, **kw):
        if state['promotion'] and purpose == 'rescore':
            rescores.append(kw.get('candidate_id'))
            if len(rescores) == 3:                      # the reference's re-run for the second candidate
                raise KeyboardInterrupt
        return real_run(self, task, config, overlay, purpose, run_id, **kw)

    def run_tests(self, config, overlay, run_id, candidate_id):
        tested.append(candidate_id)
        return [{'cmd': 'check', 'exit_code': 1, 'tail': 'a failing test'}]
    monkeypatch.setattr(runner_mod.ProjectRunner, 'run', run)
    monkeypatch.setattr(runner_mod.ProjectRunner, 'run_tests', run_tests)
    logs = []

    def log(m):
        logs.append(m)
        state['promotion'] = state['promotion'] or m.startswith('[infrarsi] promotion:')
    res = _run(_without_traps(toy), tmp_path, 4, log)
    first, second = _ranked(res)[:2]
    promo = res['promotion']
    assert res['outcome'] == 'interrupted' and res['interrupted_during'] == 'promotion'
    assert promo['promoted'] is None and promo['interrupted'] is True and tested == [first]
    assert [t['candidate'] for t in promo['tried']] == [first] and not promo['tried'][0]['passed']
    # the first candidate's five runs: the fresh pair, its save and resume, the reference's save for the eval
    assert promo['why'] == f'interrupted during the promotion of {second} (fresh re-run), after 5 re-run(s)'
    assert promo['gpu_hours'] > 0 and res['cost']['promotion_gpu_hours'] == promo['gpu_hours']
    assert len(promo['ref_offsets']) == 1 and f'[infrarsi] {promo["why"]}; nothing is promoted' in logs
    run_dir = Path(res['run_dir'])
    saved = json.loads((run_dir / 'result.json').read_text(encoding='utf-8'))
    ids = [c['id'] for c in res['candidates']]
    assert saved['outcome'] == 'interrupted' and [t['candidate'] for t in saved['promotion']['tried']] == [first]
    assert [c['id'] for c in saved['candidates']] == ids and len(saved['screens']) == 4 and {first, second} <= set(ids)
    text = (run_dir / 'REPORT.md').read_text(encoding='utf-8')
    section = text.split('## Interrupted', 1)[1].split('## Recommendation', 1)[0]
    assert promo['why'] in section and f'the promotion items of {first} were checked' in section
    assert 'nothing was promoted' in section and f'| {first} | user_tests | fail |' in text
    assert all(f'| {c} |' in text.split('## Search', 1)[1] for c in ids)


@pytest.mark.parametrize('point,check,runs', [('resume', 'resume', 3), ('run_eval', 'eval', 5), ('run_tests', 'tests', 5)])
def test_the_interrupted_promotion_names_its_check(toy, tmp_path, monkeypatch, point, check, runs):
    real_run = runner_mod.ProjectRunner.run

    def stop(*a, **kw):
        raise KeyboardInterrupt

    def run(self, task, config, overlay, purpose, run_id, **kw):
        if purpose == 'resume':
            raise KeyboardInterrupt
        return real_run(self, task, config, overlay, purpose, run_id, **kw)
    if point == 'resume':
        monkeypatch.setattr(runner_mod.ProjectRunner, 'run', run)
    else:                                               # the evals, the project's tests
        monkeypatch.setattr(runner_mod.ProjectRunner, point, stop)
    res = _run(_without_traps(toy), tmp_path, 2)
    promo = res['promotion']
    assert res['outcome'] == 'interrupted' and promo['promoted'] is None and promo['tried'] == []
    assert promo['why'] == f'interrupted during the promotion of {_ranked(res)[0]} ({check}), after {runs} re-run(s)'


def test_an_interrupted_profile_run_is_reported(toy, tmp_path, monkeypatch):
    real_run = runner_mod.ProjectRunner.run

    def run(self, task, config, overlay, purpose, run_id, **kw):
        if purpose == 'profile':
            raise KeyboardInterrupt
        return real_run(self, task, config, overlay, purpose, run_id, **kw)
    monkeypatch.setattr(runner_mod.ProjectRunner, 'run', run)
    monkeypatch.setattr(runner_mod.ProjectRunner, 'can_profile', lambda self: True)   # (none on Windows)
    res = _run(toy, tmp_path, 2)
    assert res['outcome'] == 'interrupted' and res['interrupted_during'] == 'profile' and 'search' not in res
    run_dir = Path(res['run_dir'])
    assert json.loads((run_dir / 'result.json').read_text(encoding='utf-8'))['interrupted_during'] == 'profile'
    text = (run_dir / 'REPORT.md').read_text(encoding='utf-8')
    assert '## Interrupted' in text and 'during the profile run' in text and '## Search' not in text


def test_an_interrupted_write_back_keeps_the_finished_promotion(toy, tmp_path, monkeypatch):
    # Codex review r1: Ctrl-C inside a writer left a header-only pack at its final path and ledger rows result.json did
    # not count. A promotion that passed, then Ctrl-C once the pack's first line is on disk
    from infrarsi.experience import runs as exp_runs
    from infrarsi.project.apply import plan
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))   # this run's ledger and packs only
    cut = []

    def write_lines(path, lines):
        Path(path).write_text(json.dumps(lines[0]) + '\n', encoding='utf-8')
        cut.append(Path(path).name)
        raise KeyboardInterrupt
    monkeypatch.setattr(exp_runs, 'write_lines', write_lines)
    res = _run(_without_traps(toy, writes_back=True), tmp_path, 4, strategy='hls')
    promo, home = res['promotion'], tmp_path / 'home'
    assert res['outcome'] == 'interrupted' and res['interrupted_during'] == 'write-back'
    assert promo['promoted'] and 'interrupted' not in promo and 'base' in promo   # it passed: its result stands
    assert cut == [f'{res["run_id"]}.part'] and not list((home / 'experience' / 'runs').iterdir())   # no pack at all
    assert res['experience']['written'] is None and 'interrupted' in res['experience']['not_written']
    text = (home / 'ledger.jsonl').read_text(encoding='utf-8')
    rows = [json.loads(line) for line in text.splitlines()]          # whole rows, each counted
    assert text.endswith('\n') and res['search']['remembered'] == sum(r.get('kind') != 'noise' for r in rows) > 0
    report = (Path(res['run_dir']) / 'REPORT.md').read_text(encoding='utf-8')
    assert 'while it wrote its results back' in report and f'Use candidate **{promo["promoted"]}**' in report
    assert 'not written back as experience: the run was interrupted before its pack was written' in report
    assert plan(res['run_dir'])['candidate'] == promo['promoted']   # `apply` takes the run


@pytest.mark.parametrize('at,during', [('[infrarsi] promotion:', 'promotion'), ('; report: ', 'report')])
def test_an_interrupt_between_the_steps_after_the_search_is_reported(toy, tmp_path, at, during):
    # Codex review r1: the promotion's announcement and the final log line were outside every guard
    hits = []

    def log(m):
        if at in m and not hits:
            hits.append(m)
            raise KeyboardInterrupt
    res = _run(_without_traps(toy, writes_back=True), tmp_path, 2, log)
    promo = res['promotion']
    assert hits and res['outcome'] == 'interrupted' and res['interrupted_during'] == during
    saved = json.loads((Path(res['run_dir']) / 'result.json').read_text(encoding='utf-8'))
    assert saved['interrupted_during'] == during and len(saved['screens']) == 2
    text = (Path(res['run_dir']) / 'REPORT.md').read_text(encoding='utf-8')
    if during == 'promotion':                           # before the promotion started: nothing re-run, nothing written back
        assert promo['promoted'] is None and promo['tried'] == [] and promo['gpu_hours'] == 0.0
        assert promo['why'] == 'interrupted before the promotion started' and promo['why'] in text
        assert res['experience'] is None and res['search']['remembered'] == 0
    else:                                               # after the write-back: the promotion that passed stands
        assert promo['promoted'] and 'base' in promo and 'while it wrote this report' in text
        assert Path(res['experience']['written']).is_file() and f'Use candidate **{promo["promoted"]}**' in text


def test_the_ledger_rows_are_written_whole_or_not_at_all(toy, tmp_path, monkeypatch):
    # Codex review r1: Ctrl-C between two rows left the first (result.json counted none), and half a row joined the next
    # run's; the rows are now assembled first and appended in one write
    from infrarsi.project import flow
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    trace = {'extra': {'ledger': [{'candidate': 'c1', 'mechanism': 'a', 'status': 'confirmed'},
                                  {'candidate': 'c2', 'mechanism': 'b', 'status': 'refuted'}]}}
    calls = []

    def dumps(obj, **kw):                               # Ctrl-C while the second row is formatted
        calls.append(obj)
        if len(calls) % 2 == 0:
            raise KeyboardInterrupt
        return json.dumps(obj, **kw)
    monkeypatch.setattr(flow, 'json', SimpleNamespace(dumps=dumps, loads=json.loads))
    for write in (lambda: flow.remember(toy, trace, {'tried': []}), lambda: flow.remember_noise(toy, [0.01, 0.02])):
        with pytest.raises(KeyboardInterrupt):
            write()
    assert not flow.ledger_path().exists()               # neither writer left a row
    monkeypatch.setattr(flow, 'json', json)
    assert flow.remember(toy, trace, {'tried': []}) == 2 and flow.remember_noise(toy, [0.01, 0.02]) == 2
    text = flow.ledger_path().read_text(encoding='utf-8')
    assert text.endswith('\n') and [json.loads(x).get('kind') for x in text.splitlines()] == [None, None, 'noise', 'noise']


@pytest.mark.parametrize('rows', [b'"kind": "noise"', b'"mechanism"'])
def test_an_interrupt_during_a_ledger_write_comes_after_its_rows(toy, tmp_path, monkeypatch, rows):
    # Codex reviews r2-r3, replayed on the noise rows and on the hypothesis rows: the write call writes one row and part
    # of the next, the next call raises KeyboardInterrupt having written nothing. The rest is written first, then the
    # interrupt stops the run: every line on disk is a whole row
    from infrarsi.project import flow
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    real_write, calls, tested = flow.os.write, [], []

    def write(fd, data):
        data = bytes(data)
        if rows in data and len(calls) < 2:
            calls.append(data)
            if len(calls) == 1:                         # one row and part of the next
                return real_write(fd, data[:data.index(b'\n') + 10])
            raise KeyboardInterrupt
        return real_write(fd, data)

    def run_tests(self, config, overlay, run_id, candidate_id):   # the first candidate fails its tests: two reference
        tested.append(candidate_id)                                # re-runs, two noise rows
        return [{'cmd': 'check', 'exit_code': int(len(tested) == 1), 'tail': ''}]
    monkeypatch.setattr(flow.os, 'write', write)
    monkeypatch.setattr(runner_mod.ProjectRunner, 'run_tests', run_tests)
    res = _run(_without_traps(toy, writes_back=True), tmp_path, 4, strategy='hls')
    lines = [json.loads(x) for x in flow.ledger_path().read_text(encoding='utf-8').splitlines()]   # whole rows only
    hypotheses = sum(h.get('status') != 'open' for h in res['search']['ledger'])
    assert len(calls) == 2 and res['outcome'] == 'interrupted' and res['interrupted_during'] == 'write-back'
    assert sum(x.get('kind') != 'noise' for x in lines) == hypotheses > 0
    if rows == b'"mechanism"':      # cut inside remember(): its rows are out, the count it returns was not assigned
        assert res['search']['remembered'] == 0 and not any(x.get('kind') == 'noise' for x in lines)
    else:                           # remember() had returned: the count is the hypothesis rows on disk
        assert res['search']['remembered'] == hypotheses and sum(x.get('kind') == 'noise' for x in lines) >= 2


def test_the_ledger_write_finishes_a_short_write_before_an_interrupt(tmp_path, monkeypatch):
    # Codex review r3: os.write may write fewer bytes than asked. The first call writes one row and part of the next:
    # alone (the rest follows), with Ctrl-C arriving during that call (held until the rows are out), with the next call
    # raising KeyboardInterrupt itself (Codex's replay); the rows end up whole each time
    import signal

    from infrarsi.project import flow
    rows = ''.join(json.dumps({'utc': 't', 'kind': 'noise', 'offset': x}) + '\n' for x in (0.1, 0.2, 0.3))
    real_write = flow.os.write
    for case in ('short', 'signal', 'raise'):
        monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / case))
        calls = []

        def write(fd, data):
            calls.append(len(data))
            if len(calls) == 1:
                if case == 'signal':
                    signal.raise_signal(signal.SIGINT)
                return real_write(fd, bytes(data)[:rows.index('\n') + 10])
            if len(calls) == 2 and case == 'raise':
                raise KeyboardInterrupt
            return real_write(fd, data)
        monkeypatch.setattr(flow.os, 'write', write)
        if case == 'short':
            flow.append_ledger(rows)
        else:
            with pytest.raises(KeyboardInterrupt):
                flow.append_ledger(rows)
        assert flow.ledger_path().read_text(encoding='utf-8') == rows and len(calls) == (3 if case == 'raise' else 2)
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler   # the handlers are back


def test_rows_appended_after_a_fragment_start_on_their_own_line(toy, tmp_path, monkeypatch):
    # Codex review r2: a row an older version or a crash left without its newline joined the next run's first row
    from infrarsi.project import flow
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    flow.ledger_path().parent.mkdir(parents=True)
    flow.ledger_path().write_bytes(b'{"utc": "2026-10-25", "project": "toy-cpu", "status": "conf')
    trace = {'extra': {'ledger': [{'candidate': 'c1', 'mechanism': 'a', 'status': 'confirmed'}]}}
    assert flow.remember(toy, trace, {'tried': []}) == 1
    assert [r['mechanism'] for r in flow.prior_results(toy)] == ['a']     # the fragment skipped, the new row read
    assert flow.ledger_path().read_text(encoding='utf-8').splitlines()[0].endswith('"conf')


def test_a_second_ctrl_c_during_the_grace_period_kills_at_once(monkeypatch):
    calls = []

    def kill(proc, grace_s=10.0):
        calls.append(grace_s)
        if len(calls) == 1:
            raise KeyboardInterrupt                     # pressed again while the job is given time to exit
    monkeypatch.setattr(ex, '_kill_tree', kill)
    with pytest.raises(KeyboardInterrupt):
        ex._stop_tree(SimpleNamespace(pid=1))
    assert calls == [10.0, 0.0]


def _ssh_run(monkeypatch, exc, kill_interrupts=(), reap_interrupts=(), raises=None):
    """An ssh run of three nodes that raises `exc` after launching them; the kills and reaps it caused, a further Ctrl-C
    arriving during the listed ones."""
    killed, reaped = [], []

    def kill(proc, grace_s=10.0):
        killed.append((proc, grace_s))
        if len(killed) in kill_interrupts:
            raise KeyboardInterrupt

    def reap(self, i, cwd, grace_s=30):
        reaped.append((i, grace_s))
        if len(reaped) in reap_interrupts:
            raise KeyboardInterrupt
    monkeypatch.setattr(ex, '_kill_tree', kill)
    monkeypatch.setattr(ex.SSHExecutor, '_reap', reap)

    def run(self, cmd, cwd, exports, master, python, timeout_s, log_path, stop, procs, side):
        procs += ['p0', 'p1', 'p2']
        raise exc
    monkeypatch.setattr(ex.SSHExecutor, '_run', run)
    with pytest.raises(raises or type(exc)):
        ex.SSHExecutor(['a', 'b', 'c'], '/r').run('job', cwd='/r/t', env={}, timeout_s=60, log_path=Path('log'))
    return killed, reaped


def test_an_interrupt_anywhere_in_an_ssh_run_stops_and_reaps_every_host(monkeypatch):
    killed, reaped = _ssh_run(monkeypatch, KeyboardInterrupt())
    assert killed == [('p0', 10.0), ('p1', 10.0), ('p2', 10.0)] and reaped == [(0, 30), (1, 30), (2, 30)]
    # Ctrl-C again while client 1 is stopped, and while host 0 is reaped: every step still happens, without grace
    killed, reaped = _ssh_run(monkeypatch, KeyboardInterrupt(), kill_interrupts=(2,), reap_interrupts=(1,))
    assert killed == [('p0', 10.0), ('p1', 10.0), ('p1', 0.0), ('p2', 0.0)]
    assert reaped == [(0, 1), (0, 1), (1, 1), (2, 1)]


def test_an_ordinary_failure_of_an_ssh_run_stops_every_client(monkeypatch):
    killed, reaped = _ssh_run(monkeypatch, OSError(28, 'No space left on device'))
    assert [p for p, _ in killed] == ['p0', 'p1', 'p2'] and reaped == []


def test_the_agent_strategy_registers_its_session(monkeypatch):
    from infrarsi import search
    seen = {}
    import importlib
    monkeypatch.setattr(importlib.import_module('infrarsi.agent.optimize'), 'optimize', lambda *a, **kw: seen.update(kw))
    ctx = search.Context(task=None, budget=None, start_config={}, runner=None, judge=None, hooks=None, run_id='r',
                         out_dir=Path('.'), driver=object())
    search.STRATEGIES['agent'](ctx)
    assert seen['sessions'] is ctx.sessions


def test_the_sigterm_handler_is_scoped_to_a_main_thread_run(monkeypatch, tmp_path):
    import signal
    import threading
    from infrarsi.cli import main as cli_main
    cli = sys.modules['infrarsi.cli.main']
    monkeypatch.setattr(cli, '_run_jobs', lambda *a, **kw: 0)
    before = signal.getsignal(signal.SIGTERM)
    cli_main(['run', str(tmp_path / 'infrarsi.yaml')])                # main thread: installed, then restored
    assert signal.getsignal(signal.SIGTERM) is before
    errors = []
    t = threading.Thread(target=lambda: errors.append(cli.cmd_run(cli.build().parse_args(['run', str(tmp_path)]))))
    t.start()
    t.join()
    assert errors == [0]                                # a worker thread: no handler, no ValueError


def test_a_scripts_argument_error_is_the_logs_error_line():
    # R1-18: a CRLF launch script passed `ok\r`; the useful line was argparse's, not an exception
    from infrarsi.project import signals
    lines = ['usage: fail_train.py [-h] [--max-steps MAX_STEPS] [--mode {ok,crash,oom,hang}]',
             "fail_train.py: error: argument --mode: invalid choice: 'ok\\r' (choose from ok, crash, oom, hang)"]
    assert signals.scan_events([(0.0, x) for x in lines])['error'].startswith('fail_train.py: error: argument --mode')
    assert 'error' not in signals.scan_events([(0.0, 'step 3 loss 2.0 lr_error: 0.1')])


def test_ctrl_c_while_an_ordinary_failure_is_cleaned_up_cancels_everything(monkeypatch):
    # DF3_r3: the full disk stops the clients; Ctrl-C during client 1's stop goes on to stop all and reap every host
    killed, reaped = _ssh_run(monkeypatch, OSError(28, 'No space left on device'), kill_interrupts=(2,),
                              raises=KeyboardInterrupt)
    assert [p for p, _ in killed] == ['p0', 'p1', 'p0', 'p1', 'p2'] and [i for i, _ in reaped] == [0, 1, 2]
