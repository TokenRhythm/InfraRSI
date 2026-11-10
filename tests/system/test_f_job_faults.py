"""F1 - fault injection on the stand-in job (_support/faultjob): every failure a training job can show, judged three ways.

    as a candidate   the verifier rejects it with the checklist item that names the fault (table CANDIDATE)
    as the reference `infrarsi doctor` says what is wrong, with the line a user needs (table DOCTOR)
    in a search      the search survives the faulty candidates and still promotes the real gain; a candidate that is
                     fast but breaks resume or eval fails its promotion item, and the report says which
"""
import json

import pytest
import yaml

from _support import cli
from _support.faultjob import make_project
from _support.toy import latest_run, report, result
from infrarsi.project import Project, ProjectRunner
from infrarsi.project.flow import make_verifier

# mode -> (valid as a candidate, {checklist item: status it must have})
CANDIDATE = {
    'fast': (True, {'loss_within_tolerance': 'pass', 'same_workload': 'pass', 'same_data': 'pass', 'timing_consistent': 'pass'}),
    'slow': (True, {}),
    'edge_lossy': (True, {'loss_within_tolerance': 'pass'}),          # 2.5 % at a 3 % tolerance
    'fast_lossy': (False, {'loss_within_tolerance': 'fail'}),         # 5 %
    'crash': (False, {'completes': 'fail', 'no_errors': 'fail'}),
    'oom': (False, {'completes': 'fail', 'no_errors': 'fail'}),
    'hang': (False, {'completes': 'fail'}),
    'nan': (False, {'no_errors': 'fail', 'loss_within_tolerance': 'fail'}),
    'incomplete': (False, {'completes': 'fail'}),
    'exit_nonzero_end': (False, {'completes': 'fail'}),
    'silent': (False, {'completes': 'fail'}),
    'fewer_tokens': (False, {'same_workload': 'fail'}),
    'seq_len_change': (False, {'same_workload': 'fail'}),             # README: "sequence length per update"
    'skip_lr': (False, {'optimizer_steps': 'fail'}),
    'frozen_write': (False, {'frozen_files': 'fail'}),
    'fake_fast': (False, {'timing_consistent': 'fail'}),              # reports 0.6 x the time it takes
    'burst': (False, {'timing_consistent': 'fail'}),                  # its lines arrive all at once: no clock
    'digest_change': (False, {'same_data': 'fail'}),
    'traceback_exit0': (False, {'no_errors': 'fail'}),
    'mem_over': (False, {'memory_cap': 'fail'}),
    'nccl_end': (True, {'no_errors': 'pass'}),                        # DF18: collective lines at a clean end
    'partial_timer': (False, {'timing_consistent': 'fail'}),          # its own timer covers 85 % of the update
    'crlf': (True, {'completes': 'pass'}),
    'dup': (True, {'completes': 'pass'}),                             # each line printed twice: 12 updates, not 24
    'zero_based': (True, {'completes': 'pass'}),
    'noisy': (True, {}),
}

# mode of the reference -> (doctor's exit status, phrases its output must contain)
DOCTOR = {
    'crash': (1, ['NOT VALID', 'exit code 1', 'the log: RuntimeError: CUDA error']),
    'oom': (1, ['NOT VALID', 'out-of-memory']),
    'hang': (1, ['killed at the']),
    'nan': (1, ['NaN/Inf']),
    'silent': (1, ['no step lines were recognized']),
    'incomplete': (1, ['9/12 updates']),
    'exit_nonzero_end': (1, ['exit code 3']),
    'mem_over': (1, ['peak 99.0 GiB vs cap 16.0 GiB']),
    'noisy': (0, ['noisy step times']),
    # F-SYS-4 (fixed): lines that all arrive at once are bursts, the reference keeps its reported step time and a warning
    'burst': (0, ["not on InfraRSI's clock", 'bursts']),
    # F-SYS-1 (fixed): a reference whose own timer covers 85 % of the update runs, with the coverage stated as a warning
    'partial_timer': (0, ['reference run: OK', "covers 85% of the update"]),
    # F-SYS-14 (fixed): a job without an optimizer-step counter is told so
    'ok': (0, ['reference run: OK', 'no optimizer-step counter']),
}


def test_a_tqdm_bar_job_with_the_drafted_patterns(tmp_path, home):
    """diffusers-style `Steps: n/N [..., loss=...]` bars are read through signals.patterns (what `init` drafts for such
    scripts, F-D1-05); the \r-redrawn lines arrive as separate lines and each count's last redraw holds the values."""
    from infrarsi.project.detect import TQDM_STEPS
    root = make_project(tmp_path / 'tqdm', default_mode='tqdm')
    doc = yaml.safe_load((root / 'infrarsi.yaml').read_text(encoding='utf-8'))
    doc['signals'] = {'patterns': dict(TQDM_STEPS)}
    doc['verify'] = {'tolerance': 0.03, 'required': ['completes', 'loss_within_tolerance']}   # no tokens on a bar
    (root / 'infrarsi.yaml').write_text(yaml.safe_dump(doc, sort_keys=False), encoding='utf-8')
    res = cli.run('doctor', root)
    assert res.code == 0, res.text
    assert '12 updates logged' in res.out and 'loss 12' in res.out


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    """One reference run of the stand-in job, shared by the candidate verdicts."""
    root = make_project(tmp_path_factory.mktemp('faults') / 'fault', modes=tuple(CANDIDATE))
    proj = Project.load(root)
    runner = ProjectRunner(proj, out_dir=root / '.infrarsi' / 'systest')
    task = proj.task()
    ref = runner.run(task, proj.reference_config(), None, 'reference', 'ref', candidate_id='c0')
    v = make_verifier(proj)
    assert v(ref, None, ref).valid, v(ref, None, ref).reasons
    return {'project': proj, 'runner': runner, 'task': task, 'ref': ref, 'verifier': v}


def _status(verdict) -> dict:
    return {c['id']: c for c in verdict.checks}


@pytest.mark.parametrize('mode', sorted(CANDIDATE))
def test_candidate_verdict(reference, mode):
    r = reference
    cand = r['runner'].run(r['task'], dict(r['project'].reference_config(), mode=mode), None, 'screen', 'ref', candidate_id=f'c-{mode}')
    v = r['verifier'](cand, r['ref'], r['ref'])
    valid, items = CANDIDATE[mode]
    got = _status(v)
    assert v.valid == valid, f'{mode}: valid {v.valid}, reasons {v.reasons}'
    for cid, st in items.items():
        assert got[cid]['status'] == st, f'{mode}: {cid} is {got[cid]["status"]}: {got[cid]["detail"]}'
    if mode in ('crash', 'oom'):                       # the line a user needs (F-D1-09-1), not only the exit code
        assert 'the log:' in got['completes']['detail'], got['completes']['detail']
    if mode == 'hang':
        assert 'killed at the' in got['completes']['detail']
    if mode in ('dup', 'zero_based', 'crlf'):
        assert cand['evidence']['steps_completed'] == 12
    if mode == 'fast':
        assert v.step_time < r['verifier'](r['ref'], None, r['ref']).step_time * 0.8
    if mode == 'slow':
        assert v.step_time > r['verifier'](r['ref'], None, r['ref']).step_time * 1.2


def test_every_receipt_is_signed_and_verifies(reference):
    from infrarsi.harness import signing
    assert signing.verify(reference['ref'])
    tampered = json.loads(json.dumps(reference['ref']))
    tampered['evidence']['step_time'] = 1e-9
    assert not signing.verify(tampered)
    v = reference['verifier'](tampered, None, reference['ref'])
    assert not v.valid and 'signature' in ' '.join(v.reasons)


@pytest.mark.parametrize('mode', sorted(DOCTOR))
def test_doctor_names_the_fault(tmp_path, home, mode):
    root = make_project(tmp_path / f'doc-{mode}', default_mode=mode)
    res = cli.run('doctor', root)
    code, phrases = DOCTOR[mode]
    assert res.code == code, f'{mode}: exit {res.code}\n{res.text}'
    for p in phrases:
        assert p in res.text, f'{mode}: {p!r} not in\n{res.text}'


def test_search_survives_faulty_candidates_and_promotes_the_real_gain(tmp_path, home):
    modes = ('ok', 'fast', 'crash', 'nan', 'fewer_tokens', 'skip_lr', 'fast_lossy')
    root = make_project(tmp_path / 'search', modes=modes, only_mode=True, screens=10)
    res = cli.run('run', root, '--strategy', 'random', '--llm', 'none')
    assert res.code == 0, res.text
    r = result(latest_run(root))
    assert r['outcome'] == 'improved', r['promotion']
    assert r['promotion']['config']['mode'] == 'fast'
    by_mode = {c['change']['mode'][1]: c for c in r['candidates'] if c.get('change')}
    assert set(by_mode) >= set(modes) - {'ok'}, sorted(by_mode)            # every mode was screened once
    expect = {'crash': 'completes', 'nan': 'no_errors', 'fewer_tokens': 'same_workload', 'skip_lr': 'optimizer_steps',
              'fast_lossy': 'loss_within_tolerance'}
    for mode, item in expect.items():
        c = by_mode[mode]
        assert c['valid'] is False and item in (c.get('failed_checks') or []), (mode, c.get('failed_checks'), c.get('reasons'))
    text = report(latest_run(root))
    assert '## Recommendation' in text and "| mode | ok | fast |" in text
    for item in expect.values():
        assert item in text


@pytest.mark.parametrize('mode,item', [('fast_noresume', 'resume_continues'), ('fast_reset_state', 'resume_continues'),
                                       ('fast_eval_diff', 'eval_matches')])
def test_a_fast_candidate_that_breaks_a_promotion_item_is_not_promoted(tmp_path, home, mode, item):
    root = make_project(tmp_path / mode, modes=('ok', mode), only_mode=True, screens=3)
    res = cli.run('run', root, '--strategy', 'random', '--llm', 'none')
    assert res.code == 0, res.text
    r = result(latest_run(root))
    assert r['outcome'] == 'no_change' and r['promotion']['promoted'] is None
    assert r['promotion']['why'] == 'no candidate passed promotion'
    tried = r['promotion']['tried']
    assert tried and not tried[0]['passed']
    failed = {i['id'] for i in tried[0]['items'] if i['status'] == 'fail'}
    assert failed == {item}, failed
    assert f'| {item} | fail |' in report(latest_run(root))


def test_promotion_falls_back_when_the_projects_tests_reject_the_fastest_candidate(tmp_path, home):
    """check.py rejects any configuration with two data workers: the fastest candidate (fast, 2 workers) fails
    user_tests, the next one (fast, 1 worker) passes and is the recommendation; the report shows both."""
    knobs = {'mode': {'type': 'str', 'choices': ['ok', 'fast'], 'flag': '--mode', 'default': 'ok'},
             'workers': {'type': 'int', 'choices': [1, 2], 'path': 'train.workers'}}
    root = make_project(tmp_path / 'fallback', knobs=knobs, screens=6)
    res = cli.run('run', root, '--strategy', 'random', '--llm', 'none')
    assert res.code == 0, res.text
    r = result(latest_run(root))
    assert r['outcome'] == 'improved', r['promotion'].get('why')
    assert r['promotion']['config'] == {'mode': 'fast', 'workers': 1}
    tried = r['promotion']['tried']
    assert len(tried) == 2 and not tried[0]['passed'] and tried[1]['passed']
    first = {i['id']: i['status'] for i in tried[0]['items']}
    assert first['user_tests'] == 'fail' and first.get('speed_reproduces') == 'pass'
    text = report(latest_run(root))
    assert f'| {tried[0]["candidate"]} | user_tests | fail |' in text and f'| {tried[1]["candidate"]} | user_tests | pass |' in text
    from _support.consistency import check_run
    check_run(latest_run(root), budget_screens=6)


def test_a_swallowed_collective_error_mid_run_is_rejected(reference):
    """F-SYS-16 (fixed): only collective lines after the job's last update are the end's; one in the middle of a run that
    went on and exited 0 fails no_errors."""
    r = reference
    cand = r['runner'].run(r['task'], dict(r['project'].reference_config(), mode='nccl_mid'), None, 'screen', 'ref', candidate_id='c-nccl-mid')
    v = r['verifier'](cand, r['ref'], r['ref'])
    assert not v.valid and _status(v)['no_errors']['status'] == 'fail', _status(v)['no_errors']


def test_a_knob_value_the_script_refuses_is_an_argparse_error_line(tmp_path, home):
    # F-D1-18-4: the useful line is argparse's, and it reaches the verdict as the log's error line
    root = make_project(tmp_path / 'bogus', modes=('ok', 'bogus'), only_mode=True)
    proj = Project.load(root)
    runner = ProjectRunner(proj, out_dir=root / '.infrarsi' / 'systest')
    task = proj.task()
    ref = runner.run(task, proj.reference_config(), None, 'reference', 'r', candidate_id='c0')
    cand = runner.run(task, {'mode': 'bogus'}, None, 'screen', 'r', candidate_id='c1')
    v = make_verifier(proj)(cand, ref, ref)
    assert not v.valid
    detail = _status(v)['completes']['detail']
    assert 'exit code 2' in detail and 'train.py: error: argument --mode: invalid choice' in detail, detail
