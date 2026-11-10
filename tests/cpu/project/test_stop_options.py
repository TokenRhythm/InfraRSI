import pytest

from infrarsi.cli.main import build
from infrarsi.project.manifest import ManifestError, Project
from infrarsi.project.signals import LogParser, evidence


@pytest.mark.parametrize('command', ['run', 'optimize'])
def test_cli_stop_and_budget_units(command):
    opts = build().parse_args([command, '--target-speedup', '1.8', '--target-mfu', '45', '--max-wallclock', '90m',
                              '--token-budget', '1.5M'])
    assert (opts.target_speedup, opts.target_mfu, opts.wall_hours, opts.budget_tokens) == (1.8, .45, 1.5, 1500000)
    opts = build().parse_args([command, '--wall-hours', '2', '--budget-tokens', '250000'])
    assert opts.wall_hours == 2 and opts.budget_tokens == 250000


@pytest.mark.parametrize('command', ['run', 'optimize'])
def test_cli_handlers_forward_limits_to_the_run(command, monkeypatch, tmp_path):
    import importlib
    main = importlib.import_module('infrarsi.cli.main')
    from infrarsi.project import autopilot, flow
    project = Project({'launch': {'command': 'python train.py'}}, tmp_path / 'infrarsi.yaml')
    seen = {}
    args = build().parse_args([command, str(tmp_path), '--llm', 'none', '--target-speedup', '1.8', '--target-mfu', '45',
                              '--max-wallclock', '90m', '--token-budget', '500k',
                              *(['--no-doctor'] if command == 'optimize' else [])])

    def run_jobs(a, jobs, budget, *rest):
        seen.update(budget)
        return 0

    def run_project(project, **kwargs):
        seen.update(kwargs['budget'])
        return {'run_dir': str(tmp_path), 'outcome': 'no_gain', 'promotion': {}}

    monkeypatch.setattr(main, '_run_jobs', run_jobs)
    monkeypatch.setattr(autopilot, 'ensure_manifest', lambda *a, **kw: (project, False, []))
    monkeypatch.setattr(flow, 'run_project', run_project)
    assert getattr(main, 'cmd_' + command)(args) == 0
    assert seen == {'target_speedup': 1.8, 'target_mfu': .45, 'wall_seconds': 5400, 'llm_tokens': 500000}


@pytest.mark.parametrize('location', ['stop', 'budget'])
def test_manifest_targets_reach_the_search_budget(tmp_path, location):
    p = Project({'launch': {'command': 'python train.py'}, 'code': {'mutable': ['train.py']},
                 'search': {location: {'target_speedup': 1.8, 'target_mfu': 45}}}, tmp_path / 'infrarsi.yaml')
    assert p.raw['search']['budget']['target_mfu'] == .45
    budget = p.task().search_budget()
    assert budget.target_speedup == 1.8 and budget.target_mfu == .45
    assert p.task().search_budget(target_mfu=60).target_mfu == .6


@pytest.mark.parametrize('name,value', [('target_speedup', 1), ('target_speedup', float('inf')), ('target_speedup', True),
                                       ('target_mfu', 0), ('target_mfu', 101), ('target_mfu', float('nan'))])
def test_invalid_manifest_targets_fail_before_running(tmp_path, name, value):
    with pytest.raises(ManifestError, match=name):
        Project({'launch': {'command': 'python train.py'}, 'search': {'stop': {name: value}}}, tmp_path / 'infrarsi.yaml')


def test_mfu_log_formats_and_window_mean():
    rows = LogParser().parse([(1, 'step 1 loss=1 MFU: 90%'), (2, 'step 2 loss=1 mfu=0.4'),
                              (3, 'step 3 loss=1 mfu 50'), (4, 'step 4 loss=1 MFU = 60 %')])
    ev = evidence(rows, steps=4, window=[2, 4], exit_code=0, timed_out=False, events={}, frozen={})
    assert [r['mfu'] for r in rows] == [.9, .4, .5, .6] and ev['mfu'] == pytest.approx(.5)
    assert all('mfu' in row for row in ev['per_step'])

