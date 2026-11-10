"""xfer in a project run (search/xfer.py; FABLE_PICK_C6_20261031 section 2): `--strategy xfer` through the real Engine ->
SearchSession -> runner -> verifier -> promotion path on the toy CPU job with a scripted model and a private home that
holds another environment's run pack (its states as the store writes them) and a transfer brief: the source's winner on
the first screen, wcs's prior call as it is, the port ask as the first patch call (its `transfer` field, then ordinary
asks), one budget of 10 calls; result.json and REPORT.md carry the account. Without a home it still runs, and says so."""
import json
import shutil
from pathlib import Path

import pytest
import yaml

from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.experience.pack import write_lines
from infrarsi.experience.store import labels, states_of
from infrarsi.project import Project
from infrarsi.project.flow import run_project
from infrarsi.search.llmopt import CALLS_MAX
from infrarsi.search.proposers import PRIOR_SYSTEM

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


def home_with_a_source(root: Path, project) -> dict:
    """A private home: one run pack of another environment (32 GiB card, 1 node, 1.0 B parameters, the toy's framework)
    whose best knob-only run is bf16 with the fused optimizer, and a transfer brief with one proven and one inert patch of
    that environment."""
    labs = labels(project.knobs, None)
    ref = project.reference_config()
    best = {**ref, 'precision': 'bf16', 'fused_optimizer': True}
    obs = []
    for oid, cfg, t in (('c0', ref, 1.0), ('c1', {**ref, 'fused_optimizer': True}, 0.97), ('c2', best, 0.8)):
        obs.append({'kind': 'observation', 'id': oid, 'env': 'env', 'method': 'infrarsi', 'window_steps': 8,
                    'config': {k: v for k, v in states_of({'states': {}}, labs, cfg, ok=True).items() if v != '?'},
                    'status': 'ok', 'step_time_s': t, 'n': 1})
    pack = [{'kind': 'pack', 'id': 'pack', 'schema': 'infrarsi-exp/1', 'pack_id': 'run-other-env', 'title': 'other',
             'contributors': ['infrarsi'], 'license': 'unspecified', 'created': '2026-10-31', 'tier': 'verified',
             'summary': ''},
            {'kind': 'env', 'id': 'env', 'gpu': 'L40S', 'gpus_per_node': 1, 'nodes': 1, 'memory_gib': 32,
             'framework': 'other', 'versions': {}, 'model': {'name': 'other', 'params_b': 1.0},
             'workload': {'seq_len': 4096}, 'noise_pct': 1.0}] + obs
    runs = root / 'experience' / 'runs'
    runs.mkdir(parents=True)
    write_lines(runs / 'run-other-env.jsonl', pack)
    src = {'card_gib': 32.0, 'nodes': 1, 'params_b': 1.0, 'framework': 'other'}
    brief = {'schema': 'infrarsi-transfer-brief/1', 'builder': 'test', 'sources': [{'arm': 'another arm'}],
             'proven': [{'id': 'p001', 'status': 'proven', 'mechanism': 'cheaper attention chunks',
                         'site': 'attention_cost', 'gain_pct': 30.0, 'source': src, 'src': 0}],
             'inert': [{'id': 'i001', 'status': 'inert', 'mechanism': 'a cache flag', 'site': 'attention_cost',
                        'reason': 'not activated: no `active` line', 'source': src, 'src': 0}]}
    (root / 'experience' / 'transfer_brief.json').write_text(json.dumps(brief), encoding='utf-8')
    return best


def scripted(toy: Path):
    """The model: a prior over the toy's knobs, a review that patches, and a patch of kernels.py that proves itself -
    named as the port of brief entry p001 when the call carries the transfer field."""
    kernels = (toy / 'kernels.py').read_text(encoding='utf-8')
    patched = kernels.replace('return 0.02 * chunks * chunks / 4', 'return 0.015 * chunks * chunks / 4') + \
        "\nprint('[infrarsi-patch] kernels.attention_cost: found 0.02 -> set 0.015', flush=True)\n" \
        "print('[infrarsi-patch] kernels.attention_cost: active', flush=True)\n"
    assert patched != kernels
    calls, seen = [], []

    def script(role, system, prompt):
        calls.append(role)
        payload = json.loads(prompt)
        seen.append((role, system, payload))
        if role == 'prior':
            return {'levels': [{'knob': 'precision', 'value': 'bf16', 'gain_pct': 6, 'sd_pct': 3},
                               {'knob': 'fused_optimizer', 'value': True, 'gain_pct': 2, 'sd_pct': 2}],
                    'interactions': [], 'configs': [{'name': 'safe', 'change': {'precision': 'bf16'}, 'gain_pct': 6}],
                    'notes': 'scripted'}
        if role == 'code_review':
            return {'verdict': 'patch', 'findings': [], 'summary': 'the attention kernel can be cheaper'}
        tag = '[port p001] ' if 'transfer' in payload else ''
        return {'hypotheses': [{'mechanism': tag + 'cheaper attention chunks', 'change': {}, 'patch': {'kernels.py': patched},
                                'predicted_gain_pct': 60, 'p_feasible': 0.9, 'site': 'kernels.attention_cost',
                                'memory': {'direction': 'neutral', 'gib': 0}}]}
    return ScriptedBackend(script, tokens_per_call=2000), calls, seen


def test_a_project_runs_xfer_from_its_home(toy, tmp_path, monkeypatch):
    home = tmp_path / 'home'
    monkeypatch.setenv('INFRARSI_HOME', str(home))
    best = home_with_a_source(home, Project.load(toy))
    drv, calls, seen = scripted(toy)
    res = run_project(Project.load(toy), strategy='xfer', driver=drv, budget={'screens': 6, 'llm_tokens': 10 ** 6},
                      out_root=tmp_path / 'out', log=lambda m: None)
    assert res['strategy'] == 'xfer' and res['search']['used']['screens'] == 6
    assert calls[0] == 'prior' and calls.count('prior') == 1
    assert seen[0][1] == PRIOR_SYSTEM or seen[0][1].startswith(PRIOR_SYSTEM)  # wcs's prior call (no extra fields)
    assert 'transfer' not in seen[0][2]
    hyp = [p for r, _, p in seen if r == 'hypotheses']
    assert hyp and 'transfer' in hyp[0] and all('transfer' not in p for p in hyp[1:])
    assert not [r for p in hyp for r in p.get('inert_patches') or [] if r.get('id') == 'i001']   # (wcs's own list only)
    view = hyp[0]['transfer']
    assert [r['id'] for r in view['proven']] == ['p001'] and view['do_not_repeat'][0]['site'] == 'attention_cost'
    assert 'another arm' not in json.dumps(view)                             # never a source's name
    x = res['search']['xfer']
    assert x['llm_calls'] == len(calls) <= CALLS_MAX and x['llm_usd_max'] == 8.0
    assert len(x['sources']) == 1 and x['sources'][0]['env'] == {'card_gib': 32.0, 'nodes': 1, 'params_b': 1.0}
    src = x['sources'][0]
    assert src['change'] == {k: v for k, v in best.items() if k in ('precision', 'fused_optimizer')}
    assert src['state'] == 'screened' and x['steps'][0]['pick'] == 'source'   # the first screen: the source's winner
    assert x['port']['shown'] == ['p001'] and x['port']['queued'] and x['port']['queued'][0]['entry'] == 'p001'
    assert x['brief'] == {'proven': 1, 'inert': 1, 'used': True}, (x['brief'], x['notes'])
    report = (Path(res['run_dir']) / 'REPORT.md').read_text(encoding='utf-8')
    assert '## xfer: transfer-first over the experience store' in report and 'Port ask after' in report
    assert 'Ported p' in report


def test_a_project_runs_xfer_without_a_home(toy, tmp_path, monkeypatch):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'empty-home'))
    drv, calls, seen = scripted(toy)
    res = run_project(Project.load(toy), strategy='xfer', driver=drv, budget={'screens': 4, 'llm_tokens': 10 ** 6},
                      out_root=tmp_path / 'out', log=lambda m: None)
    x = res['search']['xfer']
    assert res['search']['used']['screens'] == 4 and x['sources'] == []
    assert any('no transfer brief' in n for n in x['notes']) and any('no usable source' in n for n in x['notes'])
    assert all('transfer' not in p for r, _, p in seen if r == 'hypotheses')
    assert x['steps'][0]['pick'] == 'prior'                                  # the prior's configuration first
