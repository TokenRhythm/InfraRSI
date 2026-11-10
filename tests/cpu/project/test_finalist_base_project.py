"""The finalists' shared parts in a project run (search/finalist_base.py): a minimal finalist registered only here
(tests/cpu/search/finalist_helpers.Dummy, through finalist_base.run_engine) on the toy CPU job with a scripted model and a
private home that holds a transfer brief and another environment's pack with a `patches` section - the real Engine ->
SearchSession -> runner -> verifier -> promotion path: the patch calls carry `transfer` (the brief's and the pack's
entries) and `measured_history`, an inert patch's fix is asked for, one budget of 10 calls; the write-back puts this arm's
`patches` section on its pack's header, which still validates and which the store still loads. Without a home it runs
and says so."""
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from infrarsi import search
from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.experience.pack import read_lines, validate, write_lines
from infrarsi.project import Project
from infrarsi.project.flow import run_project
from infrarsi.search.finalist_base import run_engine
from infrarsi.search.llmopt import CALLS_MAX

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'search'))   # the search tests' dummy finalist
from finalist_helpers import DUMMY_COMPONENTS, Dummy  # noqa: E402

TOY = Path(__file__).resolve().parents[3] / 'examples' / 'toy_cpu'
OTHER = {'card_gib': 32.0, 'nodes': 1, 'params_b': 1.0, 'framework': 'other'}


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
    monkeypatch.setitem(search.STRATEGIES, 'fbdummy',
                        lambda ctx: run_engine(ctx, Dummy, DUMMY_COMPONENTS, protected=range(1, 7)))
    monkeypatch.setattr(search, 'LLM_STRATEGIES', search.LLM_STRATEGIES + ('fbdummy',))
    return d


def home_with_memory(root: Path) -> None:
    """A private home: a transfer brief with one proven and one inert patch of another environment, and that
    environment's run pack carrying a `patches` section (one proven entry)."""
    runs = root / 'experience' / 'runs'
    runs.mkdir(parents=True)
    pack = [{'kind': 'pack', 'id': 'pack', 'schema': 'infrarsi-exp/1', 'pack_id': 'run-other-env', 'title': 'other',
             'contributors': ['infrarsi'], 'license': 'unspecified', 'created': '2026-11-01', 'tier': 'verified',
             'summary': '', 'patches': {'proven': [{'mechanism': 'a fused attention cost', 'site': 'attention_cost',
                                                    'gain_pct': 12.0, 'proof': {'found': '0.02', 'set': '0.015'}}]}},
            {'kind': 'env', 'id': 'env', 'gpu': 'L40S', 'gpus_per_node': 1, 'nodes': 1, 'memory_gib': 32,
             'framework': 'other', 'versions': {}, 'model': {'name': 'other', 'params_b': 1.0},
             'workload': {'seq_len': 4096}, 'noise_pct': 1.0},
            {'kind': 'observation', 'id': 'c0', 'env': 'env', 'method': 'infrarsi', 'window_steps': 8,
             'config': {'x-workers': '1'}, 'status': 'ok', 'step_time_s': 1.0, 'n': 1}]
    write_lines(runs / 'run-other-env.jsonl', pack)
    brief = {'schema': 'infrarsi-transfer-brief/1', 'builder': 'test', 'sources': [{'arm': 'another arm'}],
             'proven': [{'id': 'p001', 'status': 'proven', 'mechanism': 'cheaper attention chunks',
                         'site': 'kernels_cost', 'gain_pct': 30.0, 'source': OTHER, 'src': 0}],
             'inert': [{'id': 'i001', 'status': 'inert', 'mechanism': 'a cache flag', 'site': 'attention_cost',
                        'reason': 'not activated: no `active` line', 'source': OTHER, 'src': 0}]}
    (root / 'experience' / 'transfer_brief.json').write_text(json.dumps(brief), encoding='utf-8')


def scripted(toy: Path):
    """The model: a prior over the toy's knobs, a review that patches, and per patch call two patches of kernels.py - one
    that proves itself, one that prints no proof line (inert at its probe)."""
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
                               {'knob': 'fused_optimizer', 'value': True, 'gain_pct': 2, 'sd_pct': 2}],
                    'interactions': [], 'configs': [{'name': 'safe', 'change': {'precision': 'bf16'}, 'gain_pct': 6}],
                    'notes': 'scripted'}
        if role == 'code_review':
            return {'verdict': 'patch', 'findings': [], 'summary': 'the attention kernel can be cheaper'}
        if role == 'configs':
            return {'configs': [{'change': {'fused_optimizer': True}, 'gain_pct': 2, 'why': 'x'}], 'notes': ''}
        n = sum(1 for r in calls if r == 'hypotheses')          # (the silent one first: its probe always fits the code
        return {'hypotheses': [{'mechanism': f'a silent tweak {n}', 'change': {}, 'patch': {'kernels.py': silent},   # pool
                                'predicted_gain_pct': 5, 'p_feasible': 0.9, 'site': 'kernels.tweak',
                                'memory': {'direction': 'neutral', 'gib': 0}},
                               {'mechanism': f'cheaper attention chunks {n}', 'change': {}, 'patch': {'kernels.py': proved},
                                'predicted_gain_pct': 60, 'p_feasible': 0.9, 'site': 'kernels.attention_cost',
                                'memory': {'direction': 'neutral', 'gib': 0}}]}
    return ScriptedBackend(script, tokens_per_call=2000), calls, seen


def test_a_project_runs_a_finalist_on_the_shared_parts_and_writes_its_patches_back(toy, tmp_path, monkeypatch):
    from infrarsi.experience.store import Store
    home = tmp_path / 'home'
    monkeypatch.setenv('INFRARSI_HOME', str(home))
    home_with_memory(home)
    drv, calls, seen = scripted(toy)
    res = run_project(Project.load(toy), strategy='fbdummy', driver=drv, budget={'screens': 6, 'llm_tokens': 10 ** 6},
                      out_root=tmp_path / 'out', log=lambda m: None)
    assert res['strategy'] == 'fbdummy' and res['search']['used']['screens'] == 6
    assert calls[0] == 'prior' and calls.count('prior') == 1 and len(calls) <= CALLS_MAX
    hyp = [p for r, _, p in seen if r == 'hypotheses']
    assert hyp and all('transfer' in p for p in hyp)
    view = hyp[0]['transfer']
    assert {r['id'] for r in view['proven']} == {'p001', 'ph001'} and view['do_not_repeat'][0]['site'] == 'attention_cost'
    assert 'another arm' not in json.dumps(view) and 'run-other-env' not in json.dumps(view)   # never a source's name
    assert all('measured_history' in p for p in hyp[1:])                    # (the toy's probes cost more than its full
    written = res['experience']['written']                                   # runs: one or two asks fit the allowance)
    assert written, res['experience']
    lines = read_lines(written)
    sec = lines[0].get('patches')
    assert sec and any(x['reason'].startswith('self-proof failed') for x in sec.get('inert') or [])
    assert validate(lines)[0] == []
    store = Store.load()
    assert any(p['pack_id'] == lines[0]['pack_id'] for p in store.packs)    # the store still reads the pack
    assert not [s for s in store.skipped if Path(s[0]).name == Path(written).name]


def test_a_project_runs_a_finalist_without_a_home(toy, tmp_path, monkeypatch):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'empty-home'))
    drv, calls, seen = scripted(toy)
    res = run_project(Project.load(toy), strategy='fbdummy', driver=drv, budget={'screens': 4, 'llm_tokens': 10 ** 6},
                      out_root=tmp_path / 'out', log=lambda m: None)
    assert res['search']['used']['screens'] == 4 and res['outcome'] in ('improved', 'no_change')
    assert all('transfer' not in p for r, _, p in seen if r == 'hypotheses')
