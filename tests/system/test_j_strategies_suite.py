"""J2 / J3 - every search strategy without a model on the toy (budget kept, report consistent, the claim re-measured
independently), the strategies that need a model when none is given, and a benchmark suite with projects that fail as it
runs or with a broken manifest (checked before any run)."""
import math

import pytest
import yaml

from _support import cli
from _support.consistency import check_run
from _support.fake_llm import FakeLLM, Reply
from _support.toy import latest_run, make_toy, report, result
from infrarsi.project import Project, ProjectRunner
from infrarsi.project.flow import make_verifier


def independent_speedup(root, promoted_config: dict, repeats: int = 2) -> float:
    """The reference and the promoted configuration, each run `repeats` times by a fresh runner outside the search:
    the speed-up a user would measure."""
    p = Project.load(root)
    runner = ProjectRunner(p, out_dir=root / '.infrarsi' / 'independent')
    task, v = p.task(), make_verifier(p)
    times = {'ref': [], 'cand': []}
    for i in range(repeats):
        for name, cfg in (('ref', p.reference_config()), ('cand', promoted_config)):
            rec = runner.run(task, cfg, None, 'rescore', 'indep', candidate_id=f'{name}{i}')
            verdict = v(rec, None, rec)
            assert verdict.valid, (name, verdict.reasons)
            times[name].append(verdict.step_time)
    return (sum(times['ref']) / repeats) / (sum(times['cand']) / repeats)


@pytest.mark.parametrize('strategy', ['random', 'tpe', 'hls', 'doe'])
def test_each_strategy_without_a_model(tmp_path, home, strategy):
    root = make_toy(tmp_path / 'toy')
    res = cli.run('run', root, '--llm', 'none', '--strategy', strategy, '--screens', '6')
    assert res.code == 0, res.text
    run = latest_run(root)
    r = check_run(run, budget_screens=6)
    assert r['strategy'] == strategy and r['outcome'] in ('improved', 'no_change'), r['outcome']
    assert 'code patches off' in (run / 'run.log').read_text(encoding='utf-8')
    for c in r['candidates']:                                 # the contract: no fp8 passes the 3 % loss check
        if (c.get('change') or {}).get('precision', [None, None])[1] == 'fp8':
            assert c['valid'] is False and 'loss_within_tolerance' in (c.get('failed_checks') or [])
    if r['outcome'] == 'improved':
        claimed = r['promotion']['speedup']
        measured = independent_speedup(root, r['promotion']['config'])
        print(f'\n[{strategy}] claimed x{claimed:.3f} (fresh re-run), independent x{measured:.3f}')
        assert measured > 1.0, f'the promoted change is not faster when measured outside the search (x{measured:.3f})'
        assert abs(math.log(measured / claimed)) < math.log(1.5), 'the claim and an independent measurement disagree by over 50 %'


@pytest.mark.parametrize('strategy', ['greedy', 'agent', 'doe+hls'])
def test_a_strategy_that_needs_a_model_says_so_without_one(toy, strategy):
    """greedy / agent without a model: one line before any run (F-SYS-10, fixed); doe+hls runs model-less."""
    res = cli.run('run', toy, '--llm', 'none', '--strategy', strategy, '--screens', '2')
    assert 'Traceback' not in res.text, res.text
    if res.code == 0:                                             # allowed: it ran as a model-less search
        r = result(latest_run(toy))
        assert r['outcome'] in ('improved', 'no_change')
    else:
        assert res.code == 2 and 'needs an LLM backend' in res.err, res.err
        from _support.toy import runs
        assert runs(toy) == [], 'the reference ran before the strategy was checked'


def test_auto_resolves_to_a_knob_space_strategy_without_a_model(toy):
    res = cli.run('run', toy, '--llm', 'none', '--screens', '2')
    assert res.code == 0, res.text
    assert result(latest_run(toy))['strategy'] in ('tpe', 'random')


def test_a_suite_runs_every_project_and_tells_the_broken_ones_apart(tmp_path, home):
    """A project that fails while the suite runs (its model refuses the test call, its reference run crashes) is a row
    of the table and the others run; a broken manifest stops the suite before any run (the test below)."""
    good = make_toy(tmp_path / 'good', screens=3, edit=lambda d: d.update(name='good'))
    crash = make_toy(tmp_path / 'crash', edit=lambda d: d.update(name='crash'))
    (crash / 'train.py').write_text('raise RuntimeError("the job cannot start")\n', encoding='utf-8')
    suite = tmp_path / 'suite.yaml'
    suite.write_text(yaml.safe_dump({'schema': 'infrarsi-suite-v1', 'projects': ['good/infrarsi.yaml', 'refused/infrarsi.yaml', 'crash/infrarsi.yaml']}),
                     encoding='utf-8')                              # no strategies: each manifest's own
    with FakeLLM(lambda req: Reply(403, {'error': {'message': 'not available from this network'}})) as llm:
        refused = make_toy(tmp_path / 'refused', edit=lambda d: d.update(      # hls calls the model: a test call first
            name='refused', llm={'backend': 'openai-compat', 'model': llm.model, 'base_url': llm.base_url},
            search=dict(d['search'], strategy='hls')))
        res = cli.run('run', suite, '--screens', '3')
    assert res.code == 1, res.text                                  # one project could not run: scripts and CI see it
    assert 'Traceback' not in res.text
    table = [ln for ln in res.out.splitlines() if ln.startswith('| ') and 'project' not in ln and '---' not in ln]
    assert len(table) == 3, res.out
    rows = {ln.split('|')[1].strip(): ln for ln in table}
    assert rows['good'].count('|') == 6 and ('improved' in rows['good'] or 'no_change' in rows['good'])
    assert 'error: llm: fake-model refused a test call: HTTP 403' in rows['refused'], rows['refused']
    assert 'reference_failed' in rows['crash']
    check_run(latest_run(good), budget_screens=3)
    r = result(latest_run(crash))
    assert r['outcome'] == 'reference_failed' and 'RuntimeError: the job cannot start' in r['feedback']
    assert '## The reference run failed' in report(latest_run(crash))
    assert not (refused / '.infrarsi').exists()


BROKEN = {                                                          # the fourth manifest of a suite, and its line
    'unknown_strategy': (lambda d: d['search'].update(strategy='bogus'), 'unknown search strategy'),
    'missing_config_file': (lambda d: d['config'].update(file='missing.yaml'), 'missing.yaml does not exist'),
    'tolerance_out_of_range': (lambda d: d['verify'].update(tolerance=3), 'verify.tolerance must be a number'),
    'stale_constraint': (lambda d: d.update(constraints=['gpus % (tp * pp) == 0']), "unknown name 'tp'"),
    'llm_without_model': (lambda d: d.update(llm={'backend': 'openai-compat', 'base_url': 'http://127.0.0.1:9/v1'}),
                          'needs a model name'),
    'config_not_a_mapping': (lambda d: d.update(config='config.yaml'),             # a shape Project.load does not check
                             "AttributeError: 'str' object has no attribute 'get'"),
    'malformed_yaml': ('name: [unclosed\n', "is not valid YAML: while parsing a flow sequence in "),   # the manifest's text
}


@pytest.mark.parametrize('name', sorted(BROKEN))
def test_a_broken_manifest_stops_the_suite_before_any_run(tmp_path, home, name):
    """U-7-8: every manifest of a suite is loaded, its strategy resolved and its model's driver built before the first
    run: a broken fourth one is one line that names it (exit 2), not an error row after three projects' runs. The YAML
    parser's diagnostic spans lines: the line keeps its detail."""
    edit, message = BROKEN[name]
    roots = [make_toy(tmp_path / f'p{i}', edit=edit if i == 4 and callable(edit) else None) for i in (1, 2, 3, 4)]
    if isinstance(edit, str):
        (roots[3] / 'infrarsi.yaml').write_text(edit, encoding='utf-8')
    suite = tmp_path / 'suite.yaml'
    suite.write_text(yaml.safe_dump({'schema': 'infrarsi-suite-v1', 'projects': [f'p{i}/infrarsi.yaml' for i in (1, 2, 3, 4)]}),
                     encoding='utf-8')
    res = cli.run('run', suite, '--screens', '2')
    assert res.code == 2, res.text
    assert res.err.startswith(f'infrarsi run: {(roots[3] / "infrarsi.yaml").resolve()}') and message in res.err, res.err
    assert res.err.count('\n') == 1 and 'Traceback' not in res.err and res.out == '', res.text
    assert not any((r / '.infrarsi').exists() for r in roots), 'a project ran before every manifest was checked'


def test_a_suite_with_a_strategy_override_runs_each_project_once(tmp_path, home):
    a = make_toy(tmp_path / 'a', screens=2)
    suite = tmp_path / 'suite.yaml'
    suite.write_text(yaml.safe_dump({'schema': 'infrarsi-suite-v1', 'projects': ['a/infrarsi.yaml'], 'strategies': ['random', 'tpe']}), encoding='utf-8')
    res = cli.run('run', suite, '--llm', 'none', '--strategy', 'random', '--screens', '2')
    assert res.code == 0, res.text
    from _support.toy import runs
    assert len(runs(a)) == 1 and result(runs(a)[0])['strategy'] == 'random'


def test_a_suite_file_that_is_not_one(tmp_path, home):
    bad = tmp_path / 'suite.yaml'
    bad.write_text('projects: [x\n', encoding='utf-8')
    res = cli.run('run', bad, '--llm', 'none')
    assert res.code == 2 and 'not valid YAML' in res.err
    other = tmp_path / 'notes.yaml'
    other.write_text('schema: something-else\n', encoding='utf-8')
    res = cli.run('run', other, '--llm', 'none')
    assert res.code == 2 and 'Traceback' not in res.err              # treated as a manifest: a one-line manifest error
