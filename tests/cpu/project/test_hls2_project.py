"""hls2 in a project run (FABLE_DESIGN_SEARCH_FIX C9): `--strategy hls2` and a manifest's `search.strategy: hls2` both
run it; REPORT.md says why a proposed move was not measured (superseded, still queued, unrunnable) while an hls ledger
reads as before; the never-measured 'superseded' rows stay out of the cross-run ledger and the transfer judgment's score."""
import json
import shutil
from pathlib import Path

import pytest
import yaml

from infrarsi.experience.transfer import score
from infrarsi.project import Project
from infrarsi.project.flow import remember, run_project
from infrarsi.project.report import _experience, _not_measured

TOY = Path(__file__).resolve().parents[3] / 'examples' / 'toy_cpu'


@pytest.fixture
def toy(tmp_path, monkeypatch):
    monkeypatch.setenv('TOY_TIME_SCALE', '0.004')
    d = tmp_path / 'toy'
    shutil.copytree(TOY, d, ignore=shutil.ignore_patterns('.infrarsi', '__pycache__'))
    doc = yaml.safe_load((d / 'infrarsi.yaml').read_text(encoding='utf-8'))
    doc['workload'] = {'steps': 12, 'window': [5, 12], 'invariants': {'global_batch': 'micro_batch_size * grad_accum'}}
    doc['checkpoint']['save_at'] = 6
    (d / 'infrarsi.yaml').write_text(yaml.safe_dump(doc), encoding='utf-8')
    return d


@pytest.mark.parametrize('how', ['flag', 'manifest'])
def test_a_project_runs_hls2(toy, tmp_path, how):
    if how == 'manifest':
        doc = yaml.safe_load((toy / 'infrarsi.yaml').read_text(encoding='utf-8'))
        doc['search']['strategy'] = 'hls2'
        (toy / 'infrarsi.yaml').write_text(yaml.safe_dump(doc), encoding='utf-8')
    res = run_project(Project.load(toy), strategy='hls2' if how == 'flag' else None, budget={'screens': 3},
                      out_root=tmp_path / 'out', log=lambda m: None)
    assert res['strategy'] == 'hls2' and res['search']['used']['screens'] == 3
    trace = json.loads((Path(res['run_dir']) / 'search' / 'trace.json').read_text(encoding='utf-8'))
    st = trace['extra']['strategy']
    assert st['name'] == 'hls2' and st['fix']['carry_open'] and 'kappa_by_class' in st
    assert all('base' in h for h in trace['extra']['ledger'])
    assert 'Strategy `hls2`' in (Path(res['run_dir']) / 'REPORT.md').read_text(encoding='utf-8')


def test_report_says_why_a_move_was_not_measured():
    old = [{'mechanism': 'liger', 'change': {'l': True}, 'status': 'open', 'predicted_gain_pct': 3.92},
           {'mechanism': 'mbs2', 'change': {'m': 2}, 'status': 'confirmed', 'measured_gain_pct': 7.7}]
    assert _not_measured({'ledger': old}) == ['', 'Proposed but not measured in this search: liger (predicted +3.9 %).']
    new = [{'mechanism': 'liger', 'change': {'l': True}, 'base': 'c0', 'status': 'superseded', 'predicted_gain_pct': 3.92,
            'why': 'carried to c1'},
           {'mechanism': 'liger', 'change': {'l': True}, 'base': 'c1', 'status': 'open', 'predicted_gain_pct': 3.92},
           {'mechanism': 'sdpa patch', 'change': {}, 'base': 'c0', 'status': 'superseded', 'predicted_gain_pct': 0.7,
            'why': 'a patch of c0, not carried to c1'},
           {'mechanism': 'compile', 'change': {'c': True}, 'base': 'c1', 'status': 'invalid',
            'why': 'unrunnable: c=True: killed at the launch timeout before its first update (2 runs that set it)'},
           {'mechanism': 'mbs2', 'change': {'m': 2}, 'base': 'c0', 'status': 'confirmed', 'measured_gain_pct': 7.7}]
    text = _not_measured({'ledger': new, 'selected': 'c1'})[1]
    assert 'queued on the final incumbent when the budget ended: liger (predicted +3.9 %)' in text
    assert 'proposed on an earlier incumbent and not tested on a later one: sdpa patch (predicted +0.7 %) [a patch' in text
    assert 'not run: the value had failed to start twice: compile [c=True: killed' in text and 'mbs2' not in text


def test_never_measured_rows_stay_out_of_the_cross_run_ledger_and_the_judgment_score(tmp_path, monkeypatch):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    p = Project({'name': 'p', 'launch': {'command': '{python} t.py {args}'}, 'knobs': {'k': {'type': 'bool', 'flag': '--k'}},
                 'workload': {'steps': 4, 'window': [2, 4]}}, tmp_path / 'infrarsi.yaml')
    rows = [{'mechanism': 'a', 'change': {'k': True}, 'status': 'superseded'},
            {'mechanism': 'b', 'change': {'k': True}, 'status': 'refuted', 'measured_gain_pct': -1.0}]
    assert remember(p, {'extra': {'ledger': rows}}) == 1
    judged = {'assessment': [{'knob': 'k', 'change': {'k': True}, 'sign': '+', 'p_feasible': 0.8}]}
    sup = [dict(rows[0], measured=None)]
    assert score(judged, sup, 0.01)['n_tried'] == 0                 # superseded: never tried
    assert score(judged, [dict(rows[1], measured=-0.01)], 0.001)['n_tried'] == 1
    lines = _experience({'levels': {'none': 1}, 'search': {'seeded': 3, 'open': 1, 'superseded': 2}})
    assert any('never measured 3; full runs spent on them: 0.' in x for x in lines)
    assert any('never measured 1; full runs spent on them: 0.' in x               # hls's numbers: as before
               for x in _experience({'levels': {'none': 1}, 'search': {'seeded': 1, 'open': 1}}))
