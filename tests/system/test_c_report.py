"""C10 - REPORT.md: every outcome renders its section from a sparse result (no crash on missing fields), the numbers in
the report are the numbers in result.json (the consistency checker on a real toy run), and `report` prints it."""
import pytest

from _support import cli
from _support.consistency import check_run
from _support.toy import latest_run, report
from infrarsi.project.report import markdown

PROJECT = {'name': 'p', 'gpus': 1, 'world': 1, 'steps': 12, 'window': [5, 12]}


def base(outcome, **over):
    r = {'schema': 'infrarsi-run-v1', 'run_id': 'p-1', 'run_dir': '/runs/p-1', 'project': PROJECT, 'tolerance': 0.03,
         'outcome': outcome, 'reference': {'config': {'a': 1}, 'step_time': 1.0, 'log': '/l'}, 'promotion': {}, 'candidates': [],
         'screens': [], 'search': {}, 'cost': {}}
    r.update(over)
    return r


def test_infrastructure_failure_before_anything_ran():
    text = markdown(base('infrastructure_failure', infrastructure='ssh to h1 failed: no output'))
    assert '## The machines could not be reached' in text and 'ssh to h1 failed' in text and '## Search' not in text


def test_infrastructure_failure_during_the_search_keeps_the_measured_runs():
    cands = [{'id': 'c0', 'status': 'screened', 'valid': True, 'step_time': 1.0}, {'id': 'c1', 'status': 'screened', 'valid': True, 'step_time': 0.9, 'change': {'a': [1, 2]}}]
    text = markdown(base('infrastructure_failure', infrastructure='lost', candidates=cands, strategy='random',
                         promotion={'promoted': None, 'why': 'stopped', 'tried': []}, search={'used': {'screens': 1}}))
    assert '## Stopped early: the machines could not be reached' in text and '| c1 |' in text and 'Keep the current configuration' in text


def test_interrupted_names_the_unverified_best():
    cands = [{'id': 'c0', 'status': 'screened', 'valid': True, 'step_time': 1.0}, {'id': 'c1', 'status': 'screened', 'valid': True, 'step_time': 0.8}]
    text = markdown(base('interrupted', candidates=cands, screens=[{}], strategy='tpe', promotion={'promoted': None, 'why': 'interrupted', 'tried': []},
                         search={'used': {'screens': 1}}))
    assert '## Interrupted' in text and 'after 1 run(s)' in text and 'c1 (x1.250 on its screen)' in text


def test_interrupted_says_where_the_run_stopped():
    # U-7-3: the promotion, the profile run before the search, the write-back after the promotion
    cands = [{'id': 'c0', 'status': 'screened', 'valid': True, 'step_time': 1.0},
             {'id': 'c1', 'status': 'screened', 'valid': True, 'step_time': 0.8, 'change': {'a': [1, 2]}},
             {'id': 'c2', 'status': 'screened', 'valid': True, 'step_time': 0.9, 'change': {'a': [1, 3]}}]
    why = 'interrupted during the promotion of c2 (resume), after 8 re-run(s)'
    tried = [{'candidate': 'c1', 'passed': False, 'items': [{'id': 'user_tests', 'status': 'fail', 'detail': '0/1 passed'}]}]
    text = markdown(base('interrupted', interrupted_during='promotion', candidates=cands, screens=[{}, {}], strategy='random',
                         promotion={'promoted': None, 'interrupted': True, 'why': why, 'tried': tried, 'gpu_hours': 0.25},
                         search={'used': {'screens': 2}}, cost={'promotion_gpu_hours': 0.25}))
    section = text.split('## Interrupted', 1)[1].split('## Recommendation', 1)[0]
    assert why in section and 'the promotion items of c1 were checked' in section and 'after 2 run(s)' not in section
    assert f'Keep the current configuration: {why}.' in text and '| c1 | user_tests | fail |' in text
    assert '| c2 |' in text and 'promotion 0.25' in text
    text = markdown(base('interrupted', interrupted_during='profile'))
    assert 'during the profile run of the reference configuration, before the search' in text and '## Search' not in text
    promo = {'promoted': 'c1', 'config': {'a': 2}, 'speedup': 1.2, 'tried': [{'candidate': 'c1', 'passed': True, 'items': []}]}
    text = markdown(base('interrupted', interrupted_during='write-back', candidates=cands, promotion=promo, strategy='random'))
    assert 'while it wrote its results back' in text and 'Use candidate **c1**' in text
    text = markdown(base('interrupted', interrupted_during='report', candidates=cands, promotion=promo, strategy='random'))
    assert 'while it wrote this report: the results below are complete' in text and 'Use candidate **c1**' in text


def test_reference_failed_shows_feedback_log_and_memory_estimate():
    text = markdown(base('reference_failed', feedback='completes: exit code 1; the log: RuntimeError: boom',
                         reference={'config': {}, 'log': '/x/train.log', 'step_log': '/x/node1.log'},
                         gpu_contention=[{'candidate': 'c0', 'purpose': 'reference', 'warnings': ['GPU 0 busy'], 'ecc': []}],
                         accounting={'memory_gib': 90.0, 'capacity_gib': 80, 'terms_gib': {'weights_grads_optimizer': 40, 'activations': 30, 'logits': 10, 'cuda_nccl_outside_allocator': 5, 'allocator_slack': 5}}))
    assert '## The reference run failed' in text and 'RuntimeError: boom' in text and '/x/train.log' in text and 'node1.log' in text
    assert 'Another process was on the GPUs' in text and 'about 90 GiB per GPU of 80' in text


def test_not_verifiable_carries_the_advice():
    text = markdown(base('not_verifiable', feedback='no tokens per update: add workload.invariants'))
    assert '## Nothing could be verified' in text and 'workload.invariants' in text


def test_improved_and_no_change_with_sparse_fields():
    cands = [{'id': 'c0', 'status': 'screened', 'valid': True, 'step_time': 1.0}, {'id': 'c1', 'status': 'screened', 'valid': True, 'step_time': 0.8, 'change': {'a': [1, 2]}}]
    promo = {'promoted': 'c1', 'config': {'a': 2}, 'speedup': 1.22, 'tried': [{'candidate': 'c1', 'passed': True, 'items': [
        {'id': 'speed_reproduces', 'status': 'pass', 'detail': 'x1.22'}]}]}
    text = markdown(base('improved', candidates=cands, promotion=promo, strategy='hls', search={'used': {'screens': 1}, 'stop_reason': 'budget'}))
    assert 'Use candidate **c1**' in text and '| a | 1 | 2 |' in text and 'fresh re-run x1.22' in text and '| c1 | speed_reproduces | pass |' in text
    text = markdown(base('no_change', candidates=cands, promotion={'promoted': None, 'why': 'nothing faster', 'tried': []}, strategy='hls'))
    assert 'Keep the current configuration: nothing faster' in text
    assert markdown(base('no_change'))                                        # nothing at all: still renders


def test_a_real_run_is_consistent_and_report_prints_it(toy):
    res = cli.run('run', toy, '--strategy', 'random', '--screens', '4', '--llm', 'none')
    assert res.code == 0, res.text
    run = latest_run(toy)
    r = check_run(run, budget_screens=4)
    printed = cli.run('report', run)
    assert printed.code == 0 and printed.out.strip() == report(run).strip()
    assert res.out.count('[infrarsi]') >= 4 and f'{r["outcome"]}' in res.out        # the milestones, then the summary line
    assert str(run / 'REPORT.md') in res.out
    assert 'interrupted_during' not in r                                              # a complete run: no such key


def test_each_candidate_row_and_the_table_carry_its_deviation_from_the_reference(tmp_path, home):
    """U-1-21: a candidate row of result.json carries the deviation from the reference that its full run's loss check
    measured, and the report's candidate table shows it ("loss vs ref"): bf16 moves the toy's loss +0.2 % (valid), fp8
    +3.5 % (invalid), a micro-batch of 8 in fp32 runs out of memory before its first update (its loss check is
    unchecked: None, "-"). The toy logs no data digest, so its gradient-norm check only reports: None everywhere."""
    from _support.toy import make_toy

    def space(doc):                                  # micro-batch 1 or 8 (accumulation 8 or 1) x precision: 6 configs
        doc['knobs'] = {k: doc['knobs'][k] for k in ('micro_batch_size', 'grad_accum', 'precision')}
        doc['knobs']['micro_batch_size']['choices'] = doc['knobs']['grad_accum']['choices'] = [1, 8]
    root = make_toy(tmp_path / 'toy', edit=space)
    res = cli.run('run', root, '--strategy', 'random', '--screens', '5', '--llm', 'none')
    assert res.code == 0, res.text
    run = latest_run(root)
    r = check_run(run, budget_screens=5)             # every row against a fresh verdict, the table's column against it

    def new(c, knob, now):
        return ((c.get('change') or {}).get(knob) or [None, now])[1]
    rows = {(new(c, 'micro_batch_size', 1), new(c, 'precision', 'fp32')): c for c in r['candidates']}
    assert len(rows) == len(r['candidates']) == 6 and rows[1, 'fp32']['id'] == 'c0'   # the reference, each config once
    for mbs in (1, 8):
        bf16, fp8 = rows[mbs, 'bf16'], rows[mbs, 'fp8']
        assert bf16['valid'] and bf16['loss_deviation'] == pytest.approx(0.002, abs=1e-5)     # moved, within 3 %
        assert not fp8['valid'] and 'loss_within_tolerance' in fp8['failed_checks']
        assert fp8['loss_deviation'] == pytest.approx(0.035, abs=1e-5)
    assert not rows[8, 'fp32']['valid'] and rows[8, 'fp32']['loss_deviation'] is None     # out of memory: nothing ran
    assert rows[1, 'fp32']['loss_deviation'] is None and all(c['grad_norm_deviation'] is None for c in r['candidates'])
    table = report(run).split('## Search', 1)[1]

    def shown(key):                                  # the "loss vs ref" cell of a configuration's row
        return next(ln for ln in table.splitlines() if ln.startswith(f'| {rows[key]["id"]} | ')).split(' | ')[4]
    assert [shown(k) for k in ((1, 'bf16'), (8, 'bf16'), (1, 'fp8'), (8, 'fp8'), (8, 'fp32'), (1, 'fp32'))] == \
        ['+0.20 %', '+0.20 %', '+3.50 %', '+3.50 %', '-', '-']


def test_ctrl_c_during_the_promotion_still_writes_the_report(tmp_path, home, monkeypatch):
    """U-7-3: Ctrl-C during the promotion of a 1-GPU job left no report (`report` said the run did not finish). The
    first candidate re-run fails its tests, and Ctrl-C arrives as the second one's resume run starts. Exit status 130;
    REPORT.md and result.json hold the search and the first candidate's promotion rows."""
    from _support.toy import make_toy
    from infrarsi.project import runner as runner_mod

    def no_traps(doc):                                  # no fp8, no micro-batch 8: every candidate valid and faster
        doc['knobs']['precision']['choices'] = ['fp32', 'bf16']
        doc['knobs']['micro_batch_size']['choices'] = [1, 2, 4]
    root = make_toy(tmp_path / 'toy', edit=no_traps)
    real, resumes, tested = runner_mod.ProjectRunner.run, [], []

    def run(self, task, config, overlay, purpose, run_id, **kw):
        if purpose == 'resume':
            resumes.append(kw.get('candidate_id'))
            if len(resumes) == 2:
                raise KeyboardInterrupt
        return real(self, task, config, overlay, purpose, run_id, **kw)

    def run_tests(self, config, overlay, run_id, candidate_id):
        tested.append(candidate_id)
        return [{'cmd': 'check', 'exit_code': 1, 'tail': 'a failing test'}]
    monkeypatch.setattr(runner_mod.ProjectRunner, 'run', run)
    monkeypatch.setattr(runner_mod.ProjectRunner, 'run_tests', run_tests)
    res = cli.run('run', root, '--strategy', 'random', '--screens', '4', '--llm', 'none')
    assert res.code == 130 and 'interrupted' in res.out, res.text
    run = latest_run(root)
    r = check_run(run, budget_screens=4)
    promo = r['promotion']
    assert r['interrupted_during'] == 'promotion' and promo['promoted'] is None and tested == resumes[:1]
    assert [t['candidate'] for t in promo['tried']] == resumes[:1]
    # the first candidate's fresh pair, save, resume and the reference's save; the second one's fresh pair and save
    assert promo['why'] == f'interrupted during the promotion of {resumes[1]} (resume), after 8 re-run(s)'
    printed = cli.run('report', run)
    assert printed.code == 0 and '## Interrupted' in printed.out and promo['why'] in printed.out


def test_ctrl_c_during_the_write_back_after_a_promotion_stops_the_suite(tmp_path, home, monkeypatch):
    """U-7-3, Codex review r1: a promotion that passed, then Ctrl-C once the experience pack's first line is on disk.
    No pack is published (no partial file either), the ledger keeps whole rows and result.json counts them, and the run
    ends interrupted with its recommendation kept: exit status 130, the suite's second project never runs, `apply`
    takes the run."""
    import json
    from pathlib import Path

    import yaml

    from _support.toy import make_toy, runs
    from infrarsi.experience import runs as exp_runs

    def edit(doc):                                      # no traps, a 2 % noise margin, the facts a pack needs
        doc['knobs']['precision']['choices'] = ['fp32', 'bf16']
        doc['knobs']['micro_batch_size']['choices'] = [1, 2, 4]
        doc['search']['run_noise'] = 0.01
        doc['hardware']['gpu_type'] = 'H100-SXM-80GB'
        doc['model'] = {'params_b': 0.001}
        doc['workload']['seq_len'] = 512
    first, second = make_toy(tmp_path / 'a', edit=edit), make_toy(tmp_path / 'b', edit=edit)
    suite = tmp_path / 'suite.yaml'
    suite.write_text(yaml.safe_dump({'schema': 'infrarsi-suite-v1', 'strategies': ['hls'],
                                     'projects': [str(first / 'infrarsi.yaml'), str(second / 'infrarsi.yaml')]}),
                     encoding='utf-8')

    def write_lines(path, lines):
        Path(path).write_text(json.dumps(lines[0]) + '\n', encoding='utf-8')
        raise KeyboardInterrupt
    monkeypatch.setattr(exp_runs, 'write_lines', write_lines)
    res = cli.run('run', suite, '--screens', '4', '--llm', 'none')
    assert res.code == 130 and 'interrupted' in res.out, res.text
    assert not runs(second)                             # the suite stopped with the interrupted run
    run = latest_run(first)
    r = check_run(run, budget_screens=4)                # the recommendation and its fresh pair of receipts included
    assert r['interrupted_during'] == 'write-back' and r['promotion']['promoted'] and r['experience']['written'] is None
    assert not list((home / 'experience' / 'runs').iterdir())
    text = (home / 'ledger.jsonl').read_text(encoding='utf-8')
    rows = [json.loads(line) for line in text.splitlines()]
    assert text.endswith('\n') and r['search']['remembered'] == sum(x.get('kind') != 'noise' for x in rows) > 0
    applied = cli.run('apply', run)
    assert applied.code == 0 and 'the change:' in applied.out, applied.text
