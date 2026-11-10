"""An interrupted search re-verifies and adopts its best candidate (search.on_interrupt: verify, the default), and
`infrarsi finish` does the same later for a run that could not."""
import json
import os
import signal
import time
from pathlib import Path

import pytest
import yaml

from infrarsi.cli.main import main
from infrarsi.project import Project
from infrarsi.project import runner as runner_mod
from infrarsi.project import finish as finish_mod
from infrarsi.project.finish import FinishError, finish_run
from infrarsi.project.flow import run_project
from infrarsi.project.manifest import ManifestError

from test_df3_interrupt import _without_traps
from test_project import toy  # noqa: F401 - the toy project fixture


def _stop_at(monkeypatch, *, purpose, nth):
    """Ctrl-C as the nth run of `purpose` starts; returns the count of the runs started."""
    real, seen = runner_mod.ProjectRunner.run, {'n': 0}

    def run(self, task, config, overlay, p, run_id, **kw):
        if p == purpose:
            seen['n'] += 1
            if seen['n'] == nth:
                raise KeyboardInterrupt
        return real(self, task, config, overlay, p, run_id, **kw)
    monkeypatch.setattr(runner_mod.ProjectRunner, 'run', run)
    return seen


def _project(toy):
    return _without_traps(toy, writes_back=True)


@pytest.mark.parametrize('strategy', ['random', 'hls'])
def test_ctrl_c_during_the_search_reverifies_the_best_candidate(toy, tmp_path, monkeypatch, strategy):
    # random is outside the Engine (the flow finalizes it); hls ends inside the Engine with its noise
    _stop_at(monkeypatch, purpose='screen', nth=4)
    logs = []
    res = run_project(_project(toy), strategy=strategy, budget={'screens': 12}, out_root=tmp_path / 'out', log=logs.append)
    assert res['search']['stop_reason'] == 'user_stop' and 'interrupted_during' not in res
    assert res['outcome'] == 'improved' and res['promotion']['promoted'], res['promotion']
    assert any('search stopped by you: re-verifying' in m for m in logs)
    assert any(m.startswith('[infrarsi] Ctrl-C stops the search') for m in logs)
    saved = json.loads((Path(res['run_dir']) / 'result.json').read_text(encoding='utf-8'))
    assert saved['outcome'] == 'improved' and 'interrupted_during' not in saved
    text = (Path(res['run_dir']) / 'REPORT.md').read_text(encoding='utf-8')
    assert '## Recommendation' in text and 'Use candidate' in text and 'stopped by you' in text and '## Interrupted' not in text


def test_the_engine_keeps_its_noise_estimate_for_the_reverification(toy, tmp_path, monkeypatch):
    _stop_at(monkeypatch, purpose='screen', nth=4)
    res = run_project(_project(toy), strategy='hls', budget={'screens': 12}, out_root=tmp_path / 'out', log=lambda m: None)
    assert (res['search'].get('noise') or {}).get('sigma_pair') is not None


def test_a_report_is_on_disk_before_the_rerun_starts(toy, tmp_path, monkeypatch):
    real, seen = runner_mod.ProjectRunner.run, {'n': 0, 'at_rerun': None, 'report': False}

    def run(self, task, config, overlay, p, run_id, **kw):
        if p == 'screen':
            seen['n'] += 1
            if seen['n'] == 4:
                raise KeyboardInterrupt
        if p == 'rescore' and seen['at_rerun'] is None:
            seen['at_rerun'] = json.loads((self.out_dir.parent / 'result.json').read_text(encoding='utf-8'))
            seen['report'] = 'infrarsi finish' in (self.out_dir.parent / 'REPORT.md').read_text(encoding='utf-8')
        return real(self, task, config, overlay, p, run_id, **kw)
    monkeypatch.setattr(runner_mod.ProjectRunner, 'run', run)
    res = run_project(_project(toy), strategy='random', budget={'screens': 12}, out_root=tmp_path / 'out', log=lambda m: None)
    early = seen['at_rerun']
    assert early['outcome'] == 'interrupted' and early['interrupted_during'] == 'promotion'
    assert 'did not finish' in early['promotion']['why'] and seen['report']
    assert res['outcome'] == 'improved'                 # ... and the finished run's result replaced it


def test_a_second_ctrl_c_stops_the_reverification(toy, tmp_path, monkeypatch):
    real, seen = runner_mod.ProjectRunner.run, {'screen': 0, 'rescore': 0}

    def run(self, task, config, overlay, p, run_id, **kw):
        if p in seen:
            seen[p] += 1
            if (p == 'screen' and seen[p] == 4) or (p == 'rescore' and seen[p] == 1):
                raise KeyboardInterrupt
        return real(self, task, config, overlay, p, run_id, **kw)
    monkeypatch.setattr(runner_mod.ProjectRunner, 'run', run)
    res = run_project(_project(toy), strategy='random', budget={'screens': 12}, out_root=tmp_path / 'out', log=lambda m: None)
    assert res['outcome'] == 'interrupted' and res['interrupted_during'] == 'promotion' and res['promotion']['promoted'] is None
    text = (Path(res['run_dir']) / 'REPORT.md').read_text(encoding='utf-8')
    assert '## Interrupted' in text and 'infrarsi finish' in text


def test_nothing_faster_ends_as_no_change(toy, tmp_path, monkeypatch):
    # the interrupt comes as the very first candidate starts: nothing but the reference was measured
    _stop_at(monkeypatch, purpose='screen', nth=1)
    logs = []
    res = run_project(_project(toy), strategy='random', budget={'screens': 12}, out_root=tmp_path / 'out', log=logs.append)
    assert res['outcome'] == 'no_change' and res['search']['stop_reason'] == 'user_stop'
    assert any('no valid candidate is faster' in m for m in logs)


def test_on_interrupt_stop_ends_at_once(toy, tmp_path, monkeypatch):
    _stop_at(monkeypatch, purpose='screen', nth=4)
    res = run_project(_project(toy), strategy='random', budget={'screens': 12}, out_root=tmp_path / 'out', log=lambda m: None,
                      on_interrupt='stop')
    assert res['outcome'] == 'interrupted' and res['interrupted_during'] == 'search' and res['promotion']['promoted'] is None
    assert res['search']['stop_reason'] == 'interrupted'
    assert 'infrarsi finish' in (Path(res['run_dir']) / 'REPORT.md').read_text(encoding='utf-8')


def test_the_manifest_sets_the_policy(toy, tmp_path, monkeypatch):
    doc = yaml.safe_load((toy.root / 'infrarsi.yaml').read_text(encoding='utf-8'))
    doc['search']['on_interrupt'] = 'stop'
    (toy.root / 'infrarsi.yaml').write_text(yaml.safe_dump(doc), encoding='utf-8')
    _stop_at(monkeypatch, purpose='screen', nth=2)
    res = run_project(Project.load(toy.root), strategy='random', budget={'screens': 6}, out_root=tmp_path / 'out',
                      log=lambda m: None)
    assert res['outcome'] == 'interrupted' and res['interrupted_during'] == 'search'
    doc['search']['on_interrupt'] = 'nonsense'
    (toy.root / 'infrarsi.yaml').write_text(yaml.safe_dump(doc), encoding='utf-8')
    with pytest.raises(ManifestError, match='on_interrupt'):
        Project.load(toy.root)


def test_a_project_without_the_key_is_verified(toy):
    assert toy.raw['search']['on_interrupt'] == 'verify'


# ---------------------------------------------------------------- `infrarsi optimize` and `infrarsi finish`
def test_optimize_applies_the_verified_candidate_after_ctrl_c(toy, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    p = _project(toy)
    _stop_at(monkeypatch, purpose='screen', nth=4)
    main(['optimize', str(p.root), '--llm', 'none', '--strategy', 'random', '--screens', '12', '--no-doctor',
          '--out', str(tmp_path / 'runs')])                # no SystemExit: exit status 0
    out = capsys.readouterr().out
    assert 'stage 5/5 (apply): written:' in out and 'stage 4/5 (report)' in out and 'improved' in out
    run = next((tmp_path / 'runs').iterdir())
    r = json.loads((run / 'result.json').read_text(encoding='utf-8'))
    assert r['search']['stop_reason'] == 'user_stop' and r['promotion']['promoted']
    assert (p.root / 'infrarsi_optimized.sh').is_file()


def _stopped_run(toy, tmp_path, monkeypatch, tolerance=None) -> tuple:
    """A run ended by Ctrl-C with the `stop` policy: (project, its result)."""
    p = _project(toy)
    real = runner_mod.ProjectRunner.run
    seen = {'n': 0}

    def run(self, task, config, overlay, purpose, run_id, **kw):
        if purpose == 'screen':
            seen['n'] += 1
            if seen['n'] == 4:
                raise KeyboardInterrupt
        return real(self, task, config, overlay, purpose, run_id, **kw)
    monkeypatch.setattr(runner_mod.ProjectRunner, 'run', run)
    res = run_project(p, strategy='random', budget={'screens': 12}, out_root=tmp_path / 'out', log=lambda m: None,
                      on_interrupt='stop', tolerance=tolerance)
    monkeypatch.setattr(runner_mod.ProjectRunner, 'run', real)
    assert res['outcome'] == 'interrupted'
    return p, res


def test_finish_verifies_a_stopped_run(toy, tmp_path, monkeypatch):
    p, res = _stopped_run(toy, tmp_path, monkeypatch)
    done = finish_run(res['run_dir'], project=p, log=lambda m: None)
    assert done['outcome'] == 'improved' and done['promotion']['promoted'] and done['finished']['from'] == 'receipts'
    assert 'interrupted_during' not in done and done['cost']['promotion_gpu_hours'] > 0
    text = (Path(res['run_dir']) / 'REPORT.md').read_text(encoding='utf-8')
    assert '## Finished later' in text and 'Use candidate' in text
    with pytest.raises(FinishError, match='already verified'):
        finish_run(res['run_dir'], project=p, log=lambda m: None)


def test_finish_works_from_the_receipts_alone(toy, tmp_path, monkeypatch):
    # the process was killed: no result.json, no report
    p, res = _stopped_run(toy, tmp_path, monkeypatch)
    run = Path(res['run_dir'])
    (run / 'result.json').unlink()
    (run / 'REPORT.md').unlink()
    done = finish_run(run, project=p, log=lambda m: None)
    assert done['outcome'] == 'improved' and done['search']['stop_reason'] == 'process_ended'
    assert (run / 'REPORT.md').is_file() and 'its process ended' in (run / 'REPORT.md').read_text(encoding='utf-8')
    assert [c['id'] for c in done['candidates']][0] == 'c0' and done['candidates'][0]['valid']


def test_finish_reports_the_model_calls_and_the_debate_of_the_cut_run(toy, tmp_path, monkeypatch):
    p, res = _stopped_run(toy, tmp_path, monkeypatch)
    run = Path(res['run_dir'])
    (run / 'result.json').unlink()
    row = {'event': 'debate', 'seq': 0, 'role': 'prior', 'outcome': 'judged', 'winner': 'second'}
    (run / 'debates.jsonl').write_text(json.dumps(row) + '\n{"a line the killed process did not fin', encoding='utf-8')
    with (run / 'search' / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({'kind': 'llm_call', 'payload': {}}) + '\n')
    done = finish_run(run, project=p, log=lambda m: None)
    assert done['search']['llm_calls'] == 1 and done['debate']['debates'] == 1 and done['debate']['won'] == {'second': 1}
    assert '## Dual-model debate' in (run / 'REPORT.md').read_text(encoding='utf-8')


def test_finish_counts_the_model_calls_after_the_last_run(toy, tmp_path, monkeypatch):
    p, res = _stopped_run(toy, tmp_path, monkeypatch)
    run = Path(res['run_dir'])
    (run / 'result.json').unlink()
    with (run / 'search' / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({'kind': 'llm_call', 'payload': {'llm_tokens': 6000, 'llm_usd': 0.3}}) + '\n')
    done = finish_run(run, project=p, log=lambda m: None)
    assert done['cost']['llm_tokens'] == 6000 and done['cost']['llm_usd'] == pytest.approx(0.3)


def test_finish_reads_an_event_log_cut_inside_a_character(toy, tmp_path, monkeypatch):
    p, res = _stopped_run(toy, tmp_path, monkeypatch)
    run = Path(res['run_dir'])
    (run / 'result.json').unlink()
    with (run / 'search' / 'events.jsonl').open('ab') as f:
        f.write('{"kind": "turn", "payload": "\u5012'.encode('utf-8')[:-1])
    assert finish_run(run, project=p, log=lambda m: None)['outcome'] == 'improved'


def test_finish_records_the_files_a_patch_was_written_against(toy, tmp_path, monkeypatch):
    p, res = _stopped_run(toy, tmp_path, monkeypatch)
    base = json.loads((Path(res['run_dir']) / 'search' / 'base_files.json').read_text(encoding='utf-8'))
    assert set(base) == {'kernels.py'}
    monkeypatch.setattr(finish_mod, 'promote', lambda *a, **k: {'promoted': 'c1', 'overlay': {'kernels.py': 'patched'}, 'tried': [],
                                                                  'gpu_hours': 0.0, 'ref_offsets': [], 'config': {}, 'speedup': 1.2})
    done = finish_run(res['run_dir'], project=p, log=lambda m: None)
    assert done['promotion']['base'] == {'kernels.py': base['kernels.py']}


def test_finish_refuses_a_run_whose_promotion_ended(toy, tmp_path):
    p = _project(toy)
    res = run_project(p, strategy='random', budget={'screens': 2}, out_root=tmp_path / 'out', log=lambda m: None)
    if res['outcome'] == 'improved':
        with pytest.raises(FinishError, match='already verified'):
            finish_run(res['run_dir'], project=p, log=lambda m: None)
    else:
        with pytest.raises(FinishError, match='nothing to finish'):
            finish_run(res['run_dir'], project=p, log=lambda m: None)


def test_finish_refuses_a_changed_configuration(toy, tmp_path, monkeypatch):
    p, res = _stopped_run(toy, tmp_path, monkeypatch)
    doc = yaml.safe_load((p.root / p.raw['config']['file']).read_text(encoding='utf-8'))
    doc['train']['precision'] = 'bf16'
    (p.root / p.raw['config']['file']).write_text(yaml.safe_dump(doc), encoding='utf-8')
    with pytest.raises(FinishError, match='not the one this run measured'):
        finish_run(res['run_dir'], project=Project.load(p.root), log=lambda m: None)


def test_finish_on_the_command_line(toy, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    p, res = _stopped_run(toy, tmp_path, monkeypatch)
    main(['finish', res['run_dir'], '--dry-run'])
    out = capsys.readouterr().out
    assert 'dry run' in out and 'improved' in out
    assert not (p.root / 'infrarsi_optimized.sh').exists()


def test_finish_says_when_nothing_was_recorded(tmp_path, capsys):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'run.log').write_text('started\n', encoding='utf-8')
    with pytest.raises(SystemExit) as e:
        main(['finish', str(run)])
    assert e.value.code == 2
    assert 'infrarsi finish' in capsys.readouterr().err


def test_a_second_ctrl_c_while_the_job_stops_skips_the_reverification(toy, tmp_path, monkeypatch):
    # the executor swallows the second interrupt while it stops the job and re-raises one KeyboardInterrupt: both must count
    real, seen = runner_mod.ProjectRunner.run, {'n': 0, 'rescore': 0}

    def run(self, task, config, overlay, p, run_id, **kw):
        if p == 'screen':
            seen['n'] += 1
            if seen['n'] == 4:
                try:
                    signal.raise_signal(signal.SIGINT)
                except KeyboardInterrupt:
                    signal.raise_signal(signal.SIGINT)
        seen['rescore'] += p == 'rescore'
        return real(self, task, config, overlay, p, run_id, **kw)
    monkeypatch.setattr(runner_mod.ProjectRunner, 'run', run)
    res = run_project(_project(toy), strategy='random', budget={'screens': 12}, out_root=tmp_path / 'out', log=lambda m: None)
    assert res['outcome'] == 'interrupted' and res['interrupted_during'] == 'search' and seen['rescore'] == 0
    assert res['search']['stop_reason'] == 'interrupted'


def test_the_checkpoint_before_the_cut_run_is_kept(toy, tmp_path, monkeypatch):
    from test_bcomp_project import scripted
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    drv, _, _ = scripted(toy.root)
    _stop_at(monkeypatch, purpose='screen', nth=3)
    res = run_project(Project.load(toy.root), strategy='bcomp', driver=drv, budget={'screens': 8, 'llm_tokens': 10 ** 7},
                      out_root=tmp_path / 'out', log=lambda m: None)
    assert res['search']['stop_reason'] == 'user_stop'
    state = json.loads((Path(res['run_dir']) / 'search' / 'state' / 'checkpoint.json').read_text(encoding='utf-8'))
    assert state['why'] != 'end' and state['stop'] is None


def test_finish_uses_the_tolerance_the_run_was_given(toy, tmp_path, monkeypatch):
    p, res = _stopped_run(toy, tmp_path, monkeypatch, tolerance=0.02)
    run = Path(res['run_dir'])
    (run / 'result.json').unlink()
    done = finish_run(run, project=p, log=lambda m: None)
    assert done['tolerance'] == 0.02


def test_finish_finds_the_project_of_a_run_written_elsewhere(toy, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    p, res = _stopped_run(toy, tmp_path, monkeypatch)        # (out_root: outside the project)
    run = Path(res['run_dir'])
    (run / 'result.json').unlink()
    (run / 'REPORT.md').unlink()
    main(['finish', str(run), '--dry-run'])
    assert 'improved' in capsys.readouterr().out


def test_finish_writes_a_report_before_its_reruns(toy, tmp_path, monkeypatch):
    p, res = _stopped_run(toy, tmp_path, monkeypatch)
    run = Path(res['run_dir'])
    (run / 'result.json').unlink()
    (run / 'REPORT.md').unlink()
    real, seen = runner_mod.ProjectRunner.run, {}

    def spy(self, task, config, overlay, purpose, run_id, **kw):
        if purpose == 'rescore' and not seen:
            seen['result'] = json.loads((run / 'result.json').read_text(encoding='utf-8'))
            seen['report'] = (run / 'REPORT.md').is_file()
        return real(self, task, config, overlay, purpose, run_id, **kw)
    monkeypatch.setattr(runner_mod.ProjectRunner, 'run', spy)
    finish_run(run, project=p, log=lambda m: None)
    assert seen['report'] and seen['result']['outcome'] == 'interrupted' and 'did not complete' in seen['result']['promotion']['why']


def test_finish_keeps_the_cost_of_an_earlier_promotion(toy, tmp_path, monkeypatch):
    p, res = _stopped_run(toy, tmp_path, monkeypatch)
    real = runner_mod.ProjectRunner.run

    def cut(self, task, config, overlay, purpose, run_id, **kw):
        if purpose == 'save':
            raise KeyboardInterrupt
        return real(self, task, config, overlay, purpose, run_id, **kw)
    monkeypatch.setattr(runner_mod.ProjectRunner, 'run', cut)
    first = finish_run(res['run_dir'], project=p, log=lambda m: None)
    assert first['outcome'] == 'interrupted' and first['cost']['promotion_gpu_hours'] > 0
    monkeypatch.setattr(runner_mod.ProjectRunner, 'run', real)
    done = finish_run(res['run_dir'], project=p, log=lambda m: None)
    assert done['outcome'] == 'improved' and done['cost']['promotion_gpu_hours'] > first['cost']['promotion_gpu_hours']


def test_finish_sees_candidates_measured_after_the_saved_result(toy, tmp_path, monkeypatch):
    # a resumed search that was killed: the receipts are newer than result.json, which says nothing of them
    p, res = _stopped_run(toy, tmp_path, monkeypatch)
    run = Path(res['run_dir'])
    best = min((c for c in res['candidates'] if c['id'] != 'c0' and c.get('valid')), key=lambda c: c['step_time'])
    new = run / 'search' / 'receipts' / 'c99.json'
    new.write_text((run / 'search' / 'receipts' / f'{best["id"]}.json').read_text(encoding='utf-8'), encoding='utf-8')
    os.utime(new, (time.time() + 60, time.time() + 60))
    done = finish_run(run, project=p, log=lambda m: None)
    assert any(c['id'] == 'c99' for c in done['candidates'])


def _capture_promote(monkeypatch, interrupted=False):
    seen = {}

    def fake(project, task, runner, verifier, trace, ref, run_id, receipts, **kw):
        seen['noise'] = trace['extra']['noise']
        seen['rows'] = trace['candidates']
        return {'promoted': None, 'tried': [], 'gpu_hours': 0.0, 'ref_offsets': [], 'why': 'scripted', 'interrupted': interrupted}
    monkeypatch.setattr(finish_mod, 'promote', fake)
    return seen


def test_finish_reads_the_noise_the_engine_saved_after_its_last_run(toy, tmp_path, monkeypatch):
    p = _project(toy)
    _stop_at(monkeypatch, purpose='screen', nth=4)
    res = run_project(p, strategy='hls', budget={'screens': 12}, out_root=tmp_path / 'out', log=lambda m: None, on_interrupt='stop')
    assert res['outcome'] == 'interrupted' and not res['search'].get('noise')   # (the flow's fallback has no Engine summary)
    run = Path(res['run_dir'])
    saved = json.loads((run / 'search' / 'noise.json').read_text(encoding='utf-8'))
    assert saved['mode'] and 'split_halves_plain' in saved
    seen = _capture_promote(monkeypatch)
    finish_run(run, project=p, log=lambda m: None)
    assert seen['noise'] == saved


def test_a_current_result_is_the_noise_finish_uses(toy, tmp_path, monkeypatch):
    # verify policy, the re-verification interrupted: the result has the Engine's final summary; noise.json may be older
    p = _project(toy)
    real, seen = runner_mod.ProjectRunner.run, {'screen': 0, 'rescore': 0}

    def run(self, task, config, overlay, purpose, run_id, **kw):
        if purpose in seen:
            seen[purpose] += 1
            if (purpose == 'screen' and seen[purpose] == 4) or (purpose == 'rescore' and seen[purpose] == 1):
                raise KeyboardInterrupt
        return real(self, task, config, overlay, purpose, run_id, **kw)
    monkeypatch.setattr(runner_mod.ProjectRunner, 'run', run)
    res = run_project(p, strategy='hls', budget={'screens': 12}, out_root=tmp_path / 'out', log=lambda m: None)
    assert res['outcome'] == 'interrupted' and res['search']['noise']
    run_dir = Path(res['run_dir'])
    (run_dir / 'search' / 'noise.json').write_text(json.dumps({'mode': 'measured', 'pairs': []}), encoding='utf-8')
    got = _capture_promote(monkeypatch)
    monkeypatch.setattr(runner_mod.ProjectRunner, 'run', real)
    finish_run(run_dir, project=p, log=lambda m: None)
    assert got['noise'] == res['search']['noise']


def test_a_retried_finish_reads_the_noise_the_first_one_used(toy, tmp_path, monkeypatch):
    p, res = _stopped_run(toy, tmp_path, monkeypatch)
    run = Path(res['run_dir'])
    (run / 'search' / 'noise.json').write_text(json.dumps({'mode': 'measured', 'pairs': [0.3]}), encoding='utf-8')
    f = run / 'search' / 'receipts' / 'c0.json'
    os.utime(f, (time.time() + 60, time.time() + 60))    # (the result is older than the receipts: the snapshot is used)
    seen = _capture_promote(monkeypatch, interrupted=True)   # (the first finish is interrupted: a later one may retry)
    finish_run(run, project=p, log=lambda m: None)
    assert seen['noise'] == {'mode': 'measured', 'pairs': [0.3]}
    os.utime(f, (time.time() - 3600, time.time() - 3600))    # (now current: the result carries what the first finish used)
    (run / 'search' / 'noise.json').write_text(json.dumps({'mode': 'measured', 'pairs': []}), encoding='utf-8')
    finish_run(run, project=p, log=lambda m: None)
    assert seen['noise'] == {'mode': 'measured', 'pairs': [0.3]}


def test_finish_gives_a_strategy_without_noise_the_rule_of_its_normal_promotion(toy, tmp_path, monkeypatch):
    p, res = _stopped_run(toy, tmp_path, monkeypatch)        # random: outside the Engine, no noise summary
    assert not (Path(res['run_dir']) / 'search' / 'noise.json').exists()
    seen = _capture_promote(monkeypatch)
    finish_run(res['run_dir'], project=p, log=lambda m: None)
    assert seen['noise'] == {}


def test_a_newer_receipt_beats_the_stale_result(toy, tmp_path, monkeypatch):
    p, res = _stopped_run(toy, tmp_path, monkeypatch)
    run = Path(res['run_dir'])
    best = min((c for c in res['candidates'] if c['id'] != 'c0' and c.get('valid')), key=lambda c: c['step_time'])
    doc = json.loads((run / 'result.json').read_text(encoding='utf-8'))
    next(c for c in doc['candidates'] if c['id'] == best['id'])['valid'] = False   # what the earlier part of the run said
    (run / 'result.json').write_text(json.dumps(doc), encoding='utf-8')
    f = run / 'search' / 'receipts' / f'{best["id"]}.json'
    os.utime(f, (time.time() + 60, time.time() + 60))
    seen = _capture_promote(monkeypatch)
    done = finish_run(run, project=p, log=lambda m: None)
    assert next(c for c in seen['rows'] if c['id'] == best['id'])['valid'] is True
    assert done['search']['stop_reason'] == 'process_ended'
