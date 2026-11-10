"""hls3 in a project run (search/hls3.py): `--strategy hls3` runs it with the manifest's memory facts, result.json and
REPORT.md carry its account (failures, memory model, bundles, stalls), `search.ablate` switches its components off by
name (all off: hls2's ledger and no hls3 account), and the report says why a move was deferred or pruned."""
import json
import shutil
from pathlib import Path

import pytest
import yaml

from infrarsi.project import Project
from infrarsi.project.flow import run_project
from infrarsi.project.report import _hls3, _not_measured
from infrarsi.search import HLS3_COMPONENTS

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


@pytest.mark.parametrize('ablate', [None, sorted(HLS3_COMPONENTS)])
def test_a_project_runs_hls3(toy, tmp_path, ablate):
    res = run_project(Project.load(toy), strategy='hls3', budget={'screens': 3}, out_root=tmp_path / 'out',
                      log=lambda m: None, ablate=ablate)
    assert res['strategy'] == 'hls3' and res['search']['used']['screens'] == 3
    trace = json.loads((Path(res['run_dir']) / 'search' / 'trace.json').read_text(encoding='utf-8'))
    st = trace['extra']['strategy']
    assert st['name'] == 'hls3' and st['fix']['carry_open'] and all('base' in h for h in trace['extra']['ledger'])
    report = (Path(res['run_dir']) / 'REPORT.md').read_text(encoding='utf-8')
    if ablate:                                                      # every component off: hls2, no hls3 account
        assert 'hls3' not in st and 'hls3' not in res['search'] and '## hls3' not in report
        assert res['research']['ablate'] == sorted(HLS3_COMPONENTS) and 'statistic' not in trace['extra']
    else:
        h = res['search']['hls3']
        assert h['off'] == [] and 'counts' in h and isinstance(h['memory'], dict)
        assert trace['extra']['statistic'] == 'median update (robust mode)'
        assert '## hls3: memory, failures and stalls' in report and 'Memory model' in report


def test_the_report_names_deferred_and_pruned_moves():
    ledger = [{'mechanism': 'no recompute', 'change': {'gc': False}, 'base': 'c2', 'status': 'open',
               'predicted_gain_pct': 20.0, 'deferred': 'predicted out of memory: it needs at least the memory of c4 (oom)'},
              {'mechanism': 'unsharded weights', 'change': {'keep': True}, 'base': 'c2', 'status': 'invalid',
               'predicted_gain_pct': 3.0, 'why': 'unrunnable on this card: keep=True: the sharded state alone is 33 GiB'},
              {'mechanism': 'liger', 'change': {'l': True}, 'base': 'c2', 'status': 'open', 'predicted_gain_pct': 4.0}]
    text = _not_measured({'ledger': ledger, 'selected': 'c2'})[1]
    assert 'deferred on the final incumbent' in text and 'no recompute (predicted +20.0 %) [predicted out of memory' in text
    assert 'it cannot fit on this card' in text and 'unsharded weights (predicted +3.0 %) [keep=True' in text
    assert 'queued on the final incumbent when the budget ended: liger' in text
    lines = _hls3({'off': ['hls3_prune'], 'failures': [{'candidate': 'c4', 'class': 'oom', 'detail': 'tried to allocate 8 GiB'}],
                   'counts': {'deferred': 2, 'unsure': 1, 'kept_probes': 3}, 'memory': {'capacity_gib': 139.8},
                   'bundles': [{'bundle': 'bundle|mem|0', 'base': 'c2', 'spender_mechanism': 'no recompute',
                                'enablers': ['liger'], 'origin': 'hls3-bundle', 'why': 'oom on c2', 'status': 'screened',
                                'candidate': 'c9', 'gain': 0.12}],
                   'stalls': [{'after_screens': 5, 'incumbent': 'c2', 'probes': ['compile'], 'whole': 'bundle:...'}]})
    text = '\n'.join(lines)
    assert 'Components switched off (ablation): hls3_prune' in text and 'oom 1 (c4)' in text
    assert 'no recompute + liger' in text and '+12.0 %' in text and 'stall after 5 full runs on c2: probes with compile' in text
    assert _hls3(None) == []
