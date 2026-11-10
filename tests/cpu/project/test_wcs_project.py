"""wcs in a project run (search/wcs.py): `--strategy wcs` runs it through the real Engine -> SearchSession -> runner ->
verifier -> promotion path on the toy CPU job; with a scripted model the prior call, the patch-only hypothesis call, a
patch that prints its proof lines (probed, then run in full), and the promotion of the best candidate; result.json and
REPORT.md carry wcs's account; every component off still runs."""
import json
import shutil
from pathlib import Path

import pytest
import yaml

from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.project import Project
from infrarsi.project.flow import run_project
from infrarsi.project.report import _wcs
from infrarsi.search import WCS_COMPONENTS

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


def scripted(toy: Path):
    """The model: a modest prior over the toy's knobs (so that the patch, predicted large, is worth a screen early), a
    review that patches, one patch of kernels.py that prints its two proof lines and makes the attention cheaper, plus a
    knob hypothesis the patch-only mode must drop."""
    kernels = (toy / 'kernels.py').read_text(encoding='utf-8')
    patched = kernels.replace('return 0.02 * chunks * chunks / 4', 'return 0.015 * chunks * chunks / 4') + \
        "\nprint('[infrarsi-patch] kernels.attention_cost: found 0.02 -> set 0.015', flush=True)\n" \
        "print('[infrarsi-patch] kernels.attention_cost: active', flush=True)\n"
    assert patched != kernels
    calls = []

    def script(role, system, prompt):
        calls.append(role)
        if role == 'prior':
            return {'levels': [{'knob': 'precision', 'value': 'bf16', 'gain_pct': 6, 'sd_pct': 3, 'memory': {'direction': 'frees', 'gib': 1}},
                               {'knob': 'precision', 'value': 'fp8', 'gain_pct': 8, 'sd_pct': 4, 'memory': {'direction': 'frees', 'gib': 2}},
                               {'knob': 'compile', 'value': True, 'gain_pct': 3, 'sd_pct': 3, 'warmup_updates': 2,
                                'memory': {'direction': 'unknown', 'gib': None}},
                               {'knob': 'fused_optimizer', 'value': True, 'gain_pct': 2, 'sd_pct': 2},
                               {'knob': 'micro_batch_size', 'value': 4, 'gain_pct': 2, 'sd_pct': 2, 'memory': {'direction': 'spends', 'gib': 4}},
                               {'knob': 'workers', 'value': 2, 'gain_pct': 1, 'sd_pct': 1}],
                    'interactions': [], 'configs': [{'name': 'aggressive', 'change': {'precision': 'fp8', 'compile': True}}],
                    'notes': 'scripted'}
        if role == 'code_review':
            return {'verdict': 'patch', 'findings': [], 'summary': 'the attention kernel can be cheaper'}
        return {'hypotheses': [{'mechanism': 'cheaper attention chunks', 'change': {}, 'patch': {'kernels.py': patched},
                                'predicted_gain_pct': 60, 'p_feasible': 0.9, 'site': 'kernels.attention_cost',
                                'memory': {'direction': 'neutral', 'gib': 0}},
                               {'mechanism': 'a knob the patch-only mode drops', 'change': {'workers': 4}, 'predicted_gain_pct': 1}]}
    return ScriptedBackend(script, tokens_per_call=2000), calls


def test_a_project_runs_wcs_with_a_scripted_model_through_the_promotion(toy, tmp_path):
    drv, calls = scripted(toy)
    res = run_project(Project.load(toy), strategy='wcs', driver=drv, budget={'screens': 6, 'llm_tokens': 10 ** 6},
                      out_root=tmp_path / 'out', log=lambda m: None)
    assert res['strategy'] == 'wcs' and res['search']['used']['screens'] == 6
    assert calls[0] == 'prior' and 'hypotheses' in calls and calls.count('prior') == 1 and len(calls) <= 6
    w = res['search']['wcs']
    assert w['prior']['source'] == 'model' and w['prior']['known'] == 6 and w['table']['legal'] > 10
    assert w['llm_calls'] >= 2 and any(p['kind'] == 'patch' for p in w['picks'])
    patches = w['patches']
    assert patches and all(p['status'] in ('proven', 'inert at its probe', 'queued', 'slower', 'crash', 'oom', 'start_cap',
                                            'hang', 'numerics', 'refused', 'invalid', 'aborted') for p in patches.values())
    trace = json.loads((Path(res['run_dir']) / 'search' / 'trace.json').read_text(encoding='utf-8'))
    st = trace['extra']['strategy']
    assert st['name'] == 'wcs' and st['proposer']['patch_only'] and all('kind' in h for h in trace['extra']['ledger'])
    kinds = {r.get('fidelity') for r in trace['screens']}
    assert 'probe' in kinds                                              # the patch probed first (its self-proof)
    assert any(p['status'] != 'queued' for p in patches.values())        # ... and judged
    assert res['promotion']['tried'] and res['outcome'] in ('improved', 'no_change')
    report = (Path(res['run_dir']) / 'REPORT.md').read_text(encoding='utf-8')
    assert '## wcs: the whole-configuration search' in report and 'Prior (model)' in report and '| pick |' in report


@pytest.mark.parametrize('ablate', [None, sorted(WCS_COMPONENTS)])
def test_a_project_runs_wcs_without_a_model_and_with_every_component_off(toy, tmp_path, ablate):
    res = run_project(Project.load(toy), strategy='wcs', budget={'screens': 3}, out_root=tmp_path / 'out',
                      log=lambda m: None, ablate=ablate)
    assert res['strategy'] == 'wcs' and res['search']['used']['screens'] == 3
    w = res['search']['wcs']
    assert w['prior']['source'] == 'flat' and w['llm_calls'] == 0
    assert w['off'] == (sorted(WCS_COMPONENTS) if ablate else [])
    assert res['research']['ablate'] == (sorted(WCS_COMPONENTS) if ablate else [])
    report = (Path(res['run_dir']) / 'REPORT.md').read_text(encoding='utf-8')
    assert '## wcs: the whole-configuration search' in report
    if ablate:
        assert 'Components switched off (ablation): wcs_antagonism' in report


def test_the_report_section_reads_the_account():
    lines = _wcs({'off': ['wcs_code'], 'prior': {'source': 'model', 'known': 5, 'values': 8, 'declared_memory': 4,
                                                'pairs': [{'members': ['a=1', 'b=2'], 'kind': 'conflicts'}], 'configs': [],
                                                'top': [{'value': 'rc=none', 'gain_pct': 22.0, 'memory': 'spends'}]},
                  'table': {'legal': 72, 'whole': True}, 'llm_calls': 1,
                  'picks': [{'id': 'thompson|1', 'screens_done': 0, 'kind': 'thompson', 'mechanism': 'rc=none + mb=2',
                             'p_feasible': 0.4, 'predicted_gain_pct': 30.0, 'outcome': 'oom'}],
                  'failures': [{'candidate': 'c1', 'class': 'oom'}], 'pairs': [{'pair': 'a=1 + b=2', 'p': 0.05, 'why': 'c3 failed'}],
                  'posterior': [{'value': 'rc=none', 'gain_pct': 20.0, 'sd_pct': 3.0}],
                  'patches': {'p3': {'status': 'proven', 'gain_pct': 7.5, 'adopted': True}},
                  'counts': {'probe_first': 2, 'infeasible': 4}})
    text = '\n'.join(lines)
    assert 'Components switched off (ablation): wcs_code' in text and '5 of 8 knob values rated' in text
    assert 'rc=none +22.0 % (spends)' in text and '| thompson|1 | 0 | thompson | rc=none + mb=2 | 0.4 | +30.0 % | oom |' in text
    assert 'oom 1 (c1)' in text and 'a=1 + b=2 (p 0.05: c3 failed)' in text and 'p3 proven +7.5 % (adopted)' in text
    assert _wcs(None) == []
