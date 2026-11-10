"""opro and ltune in a project run (search/opro.py, search/ltune.py; possible_search_algorithm/10 section 04):
`--strategy opro | ltune` through the real Engine -> SearchSession -> runner -> verifier -> promotion path on the toy CPU
job with a scripted model. opro: the prior call, configuration rounds that read the history (best first, the outcomes),
a proposal the legal table refuses not run, the code track's patch; ltune: one configuration call for the race (no prior
call), probes and short runs that are not screens, the race's full runs; both within one budget of 10 calls; result.json
and REPORT.md carry their account. Without a model and with every component off they still run."""
import json
import shutil
from pathlib import Path

import pytest
import yaml

from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.project import Project
from infrarsi.project.flow import run_project
from infrarsi.search import LTUNE_COMPONENTS, OPRO_COMPONENTS
from infrarsi.search.llmopt import CALLS_MAX
from infrarsi.search.proposers import CONFIG_RACE_RULE, CONFIG_ROUND_RULE, CONFIG_SYSTEM

TOY = Path(__file__).resolve().parents[3] / 'examples' / 'toy_cpu'
NOT_SCREENS = 'Probes and short runs are not counted as screens; their GPU-hours are counted in the search\'s GPU-hours'


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


CONFIGS = [[{'change': {'precision': 'bf16', 'fused_optimizer': True}, 'gain_pct': 8, 'why': 'cheaper math'},
            {'change': {'precision': 'fp8', 'compile': True}, 'gain_pct': 12, 'why': 'cheapest math'},
            {'change': {'micro_batch_size': 4, 'grad_accum': 4}, 'gain_pct': 5, 'why': 'fewer steps, illegal batch'},
            {'change': {'workers': 2, 'precision': 'bf16'}, 'gain_pct': 6, 'why': 'input'},
            {'change': {'no_such_knob': 1}, 'gain_pct': 50, 'why': 'nothing the table has'}],
           [{'change': {'precision': 'bf16', 'fused_optimizer': True, 'workers': 2}, 'gain_pct': 9, 'why': 'best + input'},
            {'change': {'precision': 'fp8', 'fused_optimizer': True}, 'gain_pct': 10, 'why': 'fp8 + optimizer'},
            {'change': {'micro_batch_size': 4}, 'gain_pct': 4, 'why': 'larger micro-batch'}]]


def scripted(toy: Path):
    """The model: a prior over the toy's knobs, configuration batches (the first with one the legal table refuses and one
    with no knob of the table), a review that patches, one patch of kernels.py that proves itself."""
    kernels = (toy / 'kernels.py').read_text(encoding='utf-8')
    patched = kernels.replace('return 0.02 * chunks * chunks / 4', 'return 0.015 * chunks * chunks / 4') + \
        "\nprint('[infrarsi-patch] kernels.attention_cost: found 0.02 -> set 0.015', flush=True)\n" \
        "print('[infrarsi-patch] kernels.attention_cost: active', flush=True)\n"
    assert patched != kernels
    calls, seen = [], []

    def script(role, system, prompt):
        calls.append(role)
        seen.append((role, system, json.loads(prompt)))
        if role == 'prior':
            return {'levels': [{'knob': 'precision', 'value': 'bf16', 'gain_pct': 6, 'sd_pct': 3},
                               {'knob': 'fused_optimizer', 'value': True, 'gain_pct': 2, 'sd_pct': 2},
                               {'knob': 'workers', 'value': 2, 'gain_pct': 1, 'sd_pct': 1}],
                    'interactions': [], 'configs': [{'name': 'safe', 'change': {'precision': 'bf16'}, 'gain_pct': 6}],
                    'notes': 'scripted'}
        if role == 'configs':
            n = sum(1 for r in calls if r == 'configs')
            return {'configs': CONFIGS[min(n - 1, len(CONFIGS) - 1)], 'notes': 'scripted'}
        if role == 'code_review':
            return {'verdict': 'patch', 'findings': [], 'summary': 'the attention kernel can be cheaper'}
        return {'hypotheses': [{'mechanism': 'cheaper attention chunks', 'change': {}, 'patch': {'kernels.py': patched},
                                'predicted_gain_pct': 60, 'p_feasible': 0.9, 'site': 'kernels.attention_cost',
                                'memory': {'direction': 'neutral', 'gib': 0}}]}
    return ScriptedBackend(script, tokens_per_call=2000), calls, seen


def test_a_project_runs_opro_with_a_scripted_model(toy, tmp_path):
    drv, calls, seen = scripted(toy)
    res = run_project(Project.load(toy), strategy='opro', driver=drv, budget={'screens': 8, 'llm_tokens': 10 ** 6},
                      out_root=tmp_path / 'out', log=lambda m: None)
    assert res['strategy'] == 'opro' and res['search']['used']['screens'] == 8
    assert calls[0] == 'prior' and calls.count('prior') == 1 and calls.count('configs') >= 2
    cfg_calls = [(s, p) for r, s, p in seen if r == 'configs']
    system, first = cfg_calls[0]
    assert system == CONFIG_SYSTEM + CONFIG_ROUND_RULE and first['k'] == 5 and 'knob_table' in first
    assert {'workload_invariants', 'constraints'} & set(first['task']) or first['task']   # the legal table's rules
    later = cfg_calls[-1][1]['history']
    assert later and all('change' in r and 'outcome' in r for r in later)
    measured = [r for r in later if r['outcome'].startswith('x')]
    assert measured and [r['outcome'] for r in measured] == sorted((r['outcome'] for r in measured), reverse=True)
    o = res['search']['opro']
    assert o['llm_calls'] == len(calls) <= CALLS_MAX and o['llm_calls_max'] == CALLS_MAX and o['llm_usd_max'] == 8.0
    assert o['rounds'][0]['purpose'] == 'prior' and o['rounds'][1]['purpose'] == 'round'
    assert any('no legal configuration' in x['why'] for x in o['rejected'])    # micro-batch 4 with accumulation 4
    assert not any('no_such_knob' in json.dumps(a['change']) for a in o['archive'])
    assert any(a.get('measured_gain_pct') is not None for a in o['archive'])
    assert res['promotion']['tried'] and res['outcome'] in ('improved', 'no_change')
    report = (Path(res['run_dir']) / 'REPORT.md').read_text(encoding='utf-8')
    assert '## opro: rounds of whole configurations from the model (OPRO)' in report
    assert f'Model calls: {len(calls)} of at most {CALLS_MAX} and 8.0 USD' in report and '| call | proposal |' in report


def test_a_project_runs_ltune_with_one_configuration_call(toy, tmp_path):
    drv, calls, seen = scripted(toy)
    res = run_project(Project.load(toy), strategy='ltune', driver=drv, budget={'screens': 6, 'llm_tokens': 10 ** 6},
                      out_root=tmp_path / 'out', log=lambda m: None)
    assert res['strategy'] == 'ltune' and res['search']['used']['screens'] == 6
    assert 'prior' not in calls and calls.count('configs') == 1 and calls[0] == 'configs'
    system, payload = [(s, p) for r, s, p in seen if r == 'configs'][0]
    assert system == CONFIG_SYSTEM + CONFIG_RACE_RULE and payload['history'] == []
    t = res['search']['ltune']
    assert t['llm_calls'] == len(calls) <= CALLS_MAX and t['race_full_runs'] >= 1
    rungs = [x['rung'] for x in t['race_log']]
    assert 'shorting' in rungs and 'full' in rungs
    trace = json.loads((Path(res['run_dir']) / 'search' / 'trace.json').read_text(encoding='utf-8'))
    kinds = [r.get('fidelity') for r in trace['screens']]
    assert 'short' in kinds and sum(1 for k in kinds if k in (None, 'full')) >= 6
    acc = t['ladder']['account']
    assert acc['by_fidelity']['short']['runs'] >= 1 and acc['note'].startswith(NOT_SCREENS)
    report = (Path(res['run_dir']) / 'REPORT.md').read_text(encoding='utf-8')
    assert '## ltune: one call, a race of whole configurations (lambda-Tune)' in report and NOT_SCREENS in report
    assert 'Race: ' in report and 'Runs by fidelity: ' in report


@pytest.mark.parametrize('strategy,ablate', [('opro', None), ('opro', sorted(OPRO_COMPONENTS)),
                                             ('ltune', None), ('ltune', sorted(LTUNE_COMPONENTS))])
def test_a_project_runs_them_without_a_model_and_with_every_component_off(toy, tmp_path, strategy, ablate):
    res = run_project(Project.load(toy), strategy=strategy, budget={'screens': 3}, out_root=tmp_path / 'out',
                      log=lambda m: None, ablate=ablate)
    assert res['strategy'] == strategy and res['search']['used']['screens'] == 3
    acc = res['search'][strategy]
    assert acc['prior']['source'] == 'flat' and acc['llm_calls'] == 0 and acc['rounds'] == []
