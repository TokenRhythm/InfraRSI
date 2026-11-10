"""bcomp in a project run (search/bcomp.py; FINALISTS_IMPL_SPEC_20261101 A.1): `--strategy bcomp` through the real Engine ->
SearchSession -> runner -> verifier -> promotion path on the toy CPU job with a scripted model and a private home that
holds another environment's run pack (its states as the store writes them) and a transfer brief: start one is that
source's winner mapped onto the toy's table, start two the guide's configuration, start three the uniform draw in full;
the code screens' patch calls carry the shared fields; one budget of 10 calls, no configuration call; result.json carries
bcomp's account and the write-back puts the arm's patches on its pack header. Without a home it still runs: start one is
the guide's best."""
import json
import shutil
from pathlib import Path

import pytest
import yaml

from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.experience.pack import read_lines, validate, write_lines
from infrarsi.experience.store import labels, states_of
from infrarsi.project import Project
from infrarsi.project.flow import run_project
from infrarsi.search.llmopt import CALLS_MAX

TOY = Path(__file__).resolve().parents[3] / 'examples' / 'toy_cpu'


@pytest.fixture
def toy(tmp_path, monkeypatch):
    monkeypatch.setenv('TOY_TIME_SCALE', '0.004')
    d = tmp_path / 'toy'
    shutil.copytree(TOY, d, ignore=shutil.ignore_patterns('.infrarsi', '__pycache__'))
    doc = yaml.safe_load((d / 'infrarsi.yaml').read_text(encoding='utf-8'))
    doc['workload'] = {'steps': 12, 'window': [5, 12], 'invariants': {'global_batch': 'micro_batch_size * grad_accum'},
                       'seq_len': 4096}
    doc['model'] = {'params_b': 1.0}                     # (the write-back needs the model's size and the sequence, and
    doc['hardware'] = {**doc['hardware'], 'gpu_type': 'NVIDIA L40S'}   # the pack's environment a GPU)
    doc['checkpoint']['save_at'] = 6
    (d / 'infrarsi.yaml').write_text(yaml.safe_dump(doc), encoding='utf-8')
    return d


def home_with_a_source(root: Path, project) -> dict:
    """A private home: one run pack of another environment (32 GiB card, 1 node, 2.0 B parameters) whose best knob-only
    run is bf16 with the fused optimizer, and a transfer brief with one proven and one inert patch of that environment."""
    labs = labels(project.knobs, None)
    ref = project.reference_config()
    best = {**ref, 'precision': 'bf16', 'fused_optimizer': True}
    obs = []
    for oid, cfg, t in (('c0', ref, 1.0), ('c1', {**ref, 'fused_optimizer': True}, 0.97), ('c2', best, 0.8)):
        obs.append({'kind': 'observation', 'id': oid, 'env': 'env', 'method': 'infrarsi', 'window_steps': 8,
                    'config': {k: v for k, v in states_of({'states': {}}, labs, cfg, ok=True).items() if v != '?'},
                    'status': 'ok', 'step_time_s': t, 'n': 1})
    pack = [{'kind': 'pack', 'id': 'pack', 'schema': 'infrarsi-exp/1', 'pack_id': 'run-other-env', 'title': 'other',
             'contributors': ['infrarsi'], 'license': 'unspecified', 'created': '2026-11-01', 'tier': 'verified',
             'summary': ''},
            {'kind': 'env', 'id': 'env', 'gpu': 'L40S', 'gpus_per_node': 1, 'nodes': 1, 'memory_gib': 32,
             'framework': 'other', 'versions': {}, 'model': {'name': 'other', 'params_b': 2.0},
             'workload': {'seq_len': 4096}, 'noise_pct': 1.0}] + obs
    runs = root / 'experience' / 'runs'
    runs.mkdir(parents=True)
    write_lines(runs / 'run-other-env.jsonl', pack)
    src = {'card_gib': 32.0, 'nodes': 1, 'params_b': 2.0, 'framework': 'other'}
    brief = {'schema': 'infrarsi-transfer-brief/1', 'builder': 'test', 'sources': [{'arm': 'another arm'}],
             'proven': [{'id': 'p001', 'status': 'proven', 'mechanism': 'cheaper attention chunks',
                         'site': 'attention_cost', 'gain_pct': 30.0, 'source': src, 'src': 0}],
             'inert': [{'id': 'i001', 'status': 'inert', 'mechanism': 'a cache flag', 'site': 'attention_cost',
                        'reason': 'not activated: no `active` line', 'source': src, 'src': 0}]}
    (root / 'experience' / 'transfer_brief.json').write_text(json.dumps(brief), encoding='utf-8')
    return best


def scripted(toy: Path):
    """The model: a prior over the toy's knobs (its whole configuration: the fused optimizer with workers 4), a review that
    patches, and per patch call two patches of kernels.py - one that prints no proof line (inert at its probe), one that
    proves itself."""
    kernels = (toy / 'kernels.py').read_text(encoding='utf-8')
    proved = kernels.replace('return 0.02 * chunks * chunks / 4', 'return 0.015 * chunks * chunks / 4') + \
        "\nprint('[infrarsi-patch] kernels.attention_cost: found 0.02 -> set 0.015', flush=True)\n" \
        "print('[infrarsi-patch] kernels.attention_cost: active', flush=True)\n"
    silent = kernels + '\n# a tweak that prints no proof line\n_TWEAK = 1\n'
    assert proved != kernels
    calls, seen = [], []

    def script(role, system, prompt):
        calls.append(role)
        payload = json.loads(prompt)
        seen.append((role, system, payload))
        if role == 'prior':
            return {'levels': [{'knob': 'precision', 'value': 'bf16', 'gain_pct': 6, 'sd_pct': 3},
                               {'knob': 'fused_optimizer', 'value': True, 'gain_pct': 2, 'sd_pct': 2},
                               {'knob': 'workers', 'value': 4, 'gain_pct': 3, 'sd_pct': 2}],
                    'interactions': [], 'configs': [{'name': 'safe', 'change': {'workers': 4}, 'gain_pct': 3}],
                    'notes': 'scripted'}
        if role == 'code_review':
            return {'verdict': 'patch', 'findings': [], 'summary': 'the attention kernel can be cheaper'}
        n = sum(1 for r in calls if r == 'hypotheses')
        return {'hypotheses': [{'mechanism': f'a silent tweak {n}', 'change': {}, 'patch': {'kernels.py': silent},
                                'predicted_gain_pct': 5, 'p_feasible': 0.9, 'site': 'kernels.tweak',
                                'memory': {'direction': 'neutral', 'gib': 0}},
                               {'mechanism': f'cheaper attention chunks {n}', 'change': {}, 'patch': {'kernels.py': proved},
                                'predicted_gain_pct': 60, 'p_feasible': 0.9, 'site': 'kernels.attention_cost',
                                'memory': {'direction': 'neutral', 'gib': 0}}]}
    return ScriptedBackend(script, tokens_per_call=2000), calls, seen


def test_a_project_runs_bcomp_from_its_home(toy, tmp_path, monkeypatch):
    from infrarsi.experience.store import Store
    home = tmp_path / 'home'
    monkeypatch.setenv('INFRARSI_HOME', str(home))
    best = home_with_a_source(home, Project.load(toy))
    drv, calls, seen = scripted(toy)
    res = run_project(Project.load(toy), strategy='bcomp', driver=drv, budget={'screens': 8, 'llm_tokens': 10 ** 6},
                      out_root=tmp_path / 'out', log=lambda m: None)
    assert res['strategy'] == 'bcomp' and res['search']['used']['screens'] == 8
    assert calls[0] == 'prior' and calls.count('prior') == 1 and 'configs' not in calls and len(calls) <= CALLS_MAX
    b = res['search']['bcomp']
    assert b['llm_calls'] == len(calls) and b['llm_usd_max'] == 8.0
    v = b['bcomp']
    picks = [s['pick'] for s in v['steps'] if s['pick'] in ('experience', 'guide', 'uniform')]
    assert picks == ['experience', 'guide', 'uniform'] and v['knob_screens'] == 4
    s1 = v['starts']['experience']
    assert s1['state'] == 'consumed' and s1['source'] == {'card_gib': 32.0, 'nodes': 1, 'params_b': 2.0}
    assert s1['change'] == {k: v for k, v in best.items() if k in ('precision', 'fused_optimizer')}   # mapped by name
    assert v['starts']['guide']['change'] == {'workers': 4}
    assert b['finalist']['protected_screens'] == [5, 6, 7, 8]
    hyp = [p for r, _, p in seen if r == 'hypotheses']
    assert hyp and all('transfer' in p and 'measured_history' in p for p in hyp)
    assert 'another arm' not in json.dumps(hyp[0]['transfer']) and 'run-other-env' not in json.dumps(hyp[0]['transfer'])
    written = res['experience']['written']                                   # S2: the arm's patches on its pack header
    assert written, res['experience']
    lines = read_lines(written)
    assert validate(lines)[0] == [] and lines[0].get('patches')
    assert any(p['pack_id'] == lines[0]['pack_id'] for p in Store.load().packs)


def test_a_project_runs_bcomp_without_a_home(toy, tmp_path, monkeypatch):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'empty-home'))
    drv, calls, seen = scripted(toy)
    res = run_project(Project.load(toy), strategy='bcomp', driver=drv, budget={'screens': 4, 'llm_tokens': 10 ** 6},
                      out_root=tmp_path / 'out', log=lambda m: None)
    v = res['search']['bcomp']['bcomp']
    assert res['search']['used']['screens'] == 4 and res['outcome'] in ('improved', 'no_change')
    assert v['starts']['experience']['change'] == {'workers': 4}             # the guide's best stands in
    assert any('no usable experience source' in n for n in v['notes'])
    assert all('transfer' not in p for r, _, p in seen if r == 'hypotheses')
