"""The candidates in a project run (search/esls.py, search/rpp.py): `--strategy esls | rpp | rpp-ctl` through the real
Engine -> SearchSession -> runner -> verifier -> promotion path on the toy CPU job. esls with a scripted model: the one
prior call (asking also for memory terms and families), enabler probes (probe-only, not screens), full screens, the
patch-only call and the patch's self-proof, the promotion; result.json and REPORT.md carry esls's account with its probes
and short runs, their GPU-hours and the sentence that they are not screens. rpp-ctl: no model call at all, the ladder's
probes and short runs, the same sentence. Every component off still runs."""
import json
import shutil
from pathlib import Path

import pytest
import yaml

from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.project import Project
from infrarsi.project.flow import run_project
from infrarsi.project.report import _esls, _rpp
from infrarsi.search import ESLS_COMPONENTS, RPP_COMPONENTS
from infrarsi.search.esls import PRIOR_RULE
from infrarsi.search.wcs import LLM_CALLS_MAX

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


def scripted(toy: Path):
    """The model: a prior over the toy's knobs with families and one memory term (the extended call), a review that
    patches, one patch of kernels.py that prints its two proof lines and makes the attention cheaper."""
    kernels = (toy / 'kernels.py').read_text(encoding='utf-8')
    patched = kernels.replace('return 0.02 * chunks * chunks / 4', 'return 0.015 * chunks * chunks / 4') + \
        "\nprint('[infrarsi-patch] kernels.attention_cost: found 0.02 -> set 0.015', flush=True)\n" \
        "print('[infrarsi-patch] kernels.attention_cost: active', flush=True)\n"
    assert patched != kernels
    calls, systems = [], []

    def script(role, system, prompt):
        calls.append(role)
        systems.append(system)
        if role == 'prior':
            return {'levels': [{'knob': 'precision', 'value': 'bf16', 'gain_pct': 6, 'sd_pct': 3, 'family': 'compute',
                                'memory': {'direction': 'frees', 'gib': 1}},
                               {'knob': 'precision', 'value': 'fp8', 'gain_pct': 8, 'sd_pct': 4, 'family': 'compute',
                                'memory': {'direction': 'frees', 'gib': 2}},
                               {'knob': 'compile', 'value': True, 'gain_pct': 3, 'sd_pct': 3, 'warmup_updates': 2,
                                'family': 'kernels', 'memory': {'direction': 'unknown', 'gib': None}},
                               {'knob': 'fused_optimizer', 'value': True, 'gain_pct': 2, 'sd_pct': 2, 'family': 'optimizer'},
                               {'knob': 'micro_batch_size', 'value': 4, 'gain_pct': 4, 'sd_pct': 2, 'family': 'activations',
                                'memory': {'direction': 'spends', 'gib': 4}},
                               {'knob': 'workers', 'value': 2, 'gain_pct': 1, 'sd_pct': 1, 'family': 'input'}],
                    'interactions': [], 'configs': [], 'notes': 'scripted',
                    'mem_terms': [{'name': 'activations', 'gib_expr': '2.5 * micro_batch_size', 'when': None}]}
        if role == 'code_review':
            return {'verdict': 'patch', 'findings': [], 'summary': 'the attention kernel can be cheaper'}
        return {'hypotheses': [{'mechanism': 'cheaper attention chunks', 'change': {}, 'patch': {'kernels.py': patched},
                                'predicted_gain_pct': 60, 'p_feasible': 0.9, 'site': 'kernels.attention_cost',
                                'memory': {'direction': 'neutral', 'gib': 0}}]}
    return ScriptedBackend(script, tokens_per_call=2000), calls, systems


def test_a_project_runs_esls_with_a_scripted_model_through_the_promotion(toy, tmp_path):
    drv, calls, systems = scripted(toy)
    res = run_project(Project.load(toy), strategy='esls', driver=drv, budget={'screens': 6, 'llm_tokens': 10 ** 6},
                      out_root=tmp_path / 'out', log=lambda m: None)
    assert res['strategy'] == 'esls' and res['search']['used']['screens'] == 6
    assert calls[0] == 'prior' and calls.count('prior') == 1 and PRIOR_RULE.strip() in systems[0]
    e = res['search']['esls']
    assert e['prior']['source'] == 'model' and e['lattice'] and e['spenders']
    assert e['families'] and e['ledger']['basis'] >= 2 and 'activations' in e['ledger']['terms']
    assert 'hypotheses' in calls and e['patches'] and any(x['status'] != 'queued' for x in e['patches'].values())
    trace = json.loads((Path(res['run_dir']) / 'search' / 'trace.json').read_text(encoding='utf-8'))
    kinds = [r.get('fidelity') for r in trace['screens']]
    assert 'probe' in kinds and sum(1 for k in kinds if k in (None, 'full')) >= 6
    assert res['search']['used'].get('trials', 0) >= 1                   # probes / short runs: trials, not screens
    acc = e['ladder']['account']
    assert acc['by_fidelity']['probe']['runs'] >= 1 and acc['note'].startswith(NOT_SCREENS)
    assert any(x['status'] in ('measured', 'ran', 'declared', 'failed') for x in e['enablers'].values())
    assert res['promotion']['tried'] and res['outcome'] in ('improved', 'no_change')
    report = (Path(res['run_dir']) / 'REPORT.md').read_text(encoding='utf-8')
    assert '## esls: the enabler / spender lattice search' in report and NOT_SCREENS in report
    assert 'Runs by fidelity: ' in report and 'Memory ledger: ' in report
    assert e['llm_calls'] == len(calls) <= LLM_CALLS_MAX and e['llm_calls_max'] == LLM_CALLS_MAX   # wcs's one allowance
    assert f'Model calls: {len(calls)} of at most {LLM_CALLS_MAX}' in report


def test_a_project_runs_the_control_without_any_model_call(toy, tmp_path):
    drv, calls, _ = scripted(toy)
    logs = []
    res = run_project(Project.load(toy), strategy='rpp-ctl', driver=drv, budget={'screens': 4, 'llm_tokens': 10 ** 6},
                      out_root=tmp_path / 'out', log=logs.append)
    assert res['strategy'] == 'rpp-ctl' and res['search']['used']['screens'] == 4
    assert calls == [] and drv.calls == []                                # no prior, no hypotheses, no review
    r = res['search']['rpp']
    assert r['name'] == 'rpp-ctl' and r['temperature'] == 'inf' and r['llm_calls'] == 0
    assert sorted(r['off']) == ['rpp_anneal', 'rpp_code', 'rpp_prior'] and r['prior']['source'] == 'flat'
    assert r['ladder']['account']['note'].startswith(NOT_SCREENS) and r['draws']
    assert any('code patches off: strategy rpp-ctl changes knobs only' in m for m in logs)
    report = (Path(res['run_dir']) / 'REPORT.md').read_text(encoding='utf-8')
    assert '## rpp-ctl: draws weighted by feasibility' in report and NOT_SCREENS in report and '| draw |' in report


@pytest.mark.parametrize('strategy,ablate', [('rpp', None), ('rpp', sorted(RPP_COMPONENTS)),
                                             ('esls', None), ('esls', sorted(ESLS_COMPONENTS))])
def test_a_project_runs_the_candidates_without_a_model_and_with_every_component_off(toy, tmp_path, strategy, ablate):
    res = run_project(Project.load(toy), strategy=strategy, budget={'screens': 3}, out_root=tmp_path / 'out',
                      log=lambda m: None, ablate=ablate)
    assert res['strategy'] == strategy and res['search']['used']['screens'] == 3
    acc = res['search']['esls' if strategy == 'esls' else 'rpp']
    assert acc['prior']['source'] == 'flat' and acc['llm_calls'] == 0
    assert acc['off'] == (sorted(ablate) if ablate else [])
    assert res['research']['ablate'] == (sorted(ablate) if ablate else [])
    report = (Path(res['run_dir']) / 'REPORT.md').read_text(encoding='utf-8')
    assert ('## esls: ' if strategy == 'esls' else '## rpp: ') in report


def test_the_report_sections_read_the_accounts():
    ladder = {'counts': {'probe_first': 2, 'short': 3, 'promoted': 1, 'not_promoted': 2, 'lowfi_capped': 0},
              'account': {'by_fidelity': {'full': {'runs': 10, 'gpu_hours': 4.0}, 'probe': {'runs': 4, 'gpu_hours': 0.8},
                                          'short': {'runs': 3, 'gpu_hours': 0.9}},
                          'lowfi_gpu_hours': 1.7, 'lowfi_share_of_gpu_hours': 0.298, 'allowance_gpu_hours': 2.0,
                          'note': NOT_SCREENS + ' (at most 50% of the screens\' GPU-hours by the low-fidelity allowance).'}}
    ledger = {'observations': {'exact': 9, 'lower': 2}, 'basis': 12, 'terms': ['activations'], 'dropped_terms': [],
              'moved_most': [{'name': 'mb=4', 'prior_gib': 20.0, 'fitted': 31.5}]}
    text = '\n'.join(_esls({'off': ['esls_tail'], 'spenders': ['mb=2', 'mb=4'],
                            'lattice': [{'level': 'the reference', 'mem_gib': 40.0, 'gain_pct': 0.0},
                                        {'level': 'mb=2', 'mem_gib': 60.0, 'gain_pct': 5.0}],
                            'pruned': [], 'enablers': {'fl=true': {'status': 'measured', 'frees': 11.6, 'declared': 4.0}},
                            'boundaries': [{'set': 'fl=True', 'level': 1, 'configuration': 'mb=2 + fl=True', 'outcome': 'ok',
                                            'gain_pct': 7.25}],
                            'probes_bisection': [{'set': 'fl=True', 'level': 1, 'fits': True, 'outcome': 'ok'}],
                            'tail': {'members': ['a=1'], 'accepted': ['a=1'], 'diagnostic_screens': 0},
                            'screens_by_purpose': {'boundary': 1}, 'ladder': ladder, 'ledger': ledger}))
    assert 'Components switched off (ablation): esls_tail' in text and 'Chain: 2 spend levels (40.0 to 60.0 GiB' in text
    assert 'fl=true measured, frees 11.6 GiB (declared 4.0)' in text and '| fl=True | 1 | mb=2 + fl=True | ok | +7.25 % |' in text
    assert 'probe 4 (0.8 GPU-h)' in text and '30 % of the search\'s GPU-hours (allowance 2 GPU-h)' in text
    assert 'promoted to full 1 of 3' in text and NOT_SCREENS in text and 'mb=4 20.0 -> 31.5' in text
    rtext = '\n'.join(_rpp({'name': 'rpp-ctl', 'a': 1.0, 'temperature': 'inf', 'T0': 0.03, 'off': ['rpp_prior'],
                            'llm_calls': 0, 'draws': [{'id': 'draw|1', 'screens_done': 0, 'p_feasible': 0.62,
                                                       'plan': ['probe', 'short'], 'outcome': 'short'}],
                            'ladder': ladder, 'ledger': ledger}))
    assert '## rpp-ctl: draws weighted by feasibility' in rtext and 'and the time model' not in rtext
    assert '| draw|1 | 0 | 0.62 | probe -> short | short |' in rtext and NOT_SCREENS in rtext
    assert _esls(None) == [] and _rpp(None) == []


def test_c9_the_control_neither_reads_nor_writes_the_experience_store(toy, tmp_path, monkeypatch):
    # CAND_r1 C9 (D-C1): rpp-ctl runs without the experience store - no plan, no earlier results, no write-back
    from infrarsi.experience import runs as exp_runs
    from infrarsi.project import flow
    calls = []
    real_plan = exp_runs.plan_for
    monkeypatch.setattr(exp_runs, 'plan_for', lambda *a, **k: calls.append('plan_for') or real_plan(*a, **k))
    monkeypatch.setattr(exp_runs, 'write_back', lambda *a, **k: calls.append('write_back'))
    monkeypatch.setattr(flow, 'prior_results', lambda *a, **k: calls.append('prior_results') or [])
    monkeypatch.setattr(flow, 'remember', lambda *a, **k: calls.append('remember') or 0)
    res = run_project(Project.load(toy), strategy='rpp-ctl', budget={'screens': 2}, out_root=tmp_path / 'out',
                      log=lambda m: None)
    assert res['strategy'] == 'rpp-ctl' and res['search']['used']['screens'] == 2
    assert calls == []                                                # before: plan_for, prior_results, remember, ...
    res = run_project(Project.load(toy), strategy='rpp', budget={'screens': 2}, out_root=tmp_path / 'out2',
                      log=lambda m: None)
    assert 'plan_for' in calls and 'remember' in calls                # (the candidate keeps the store: the switch is
    #                                                                   the control's alone)
