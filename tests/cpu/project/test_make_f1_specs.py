"""The finalist arms' spec generator (reports/infrarsi_newmodels_20261028/scripts/make_f1_specs.py), on copies only: base
specs copied into a temporary spec folder, small shared homes built in a temporary folder, an empty trials folder for the
brief (never the real nm folder or the real homes). D5 (FINALISTS_IMPL_SPEC §B.7): E16's wall 12 h, 192 GPU-hours and a
50400 s run step; M35H4's wall 5 h (its run step 21600 s); the six earlier jobs untouched; each override recorded in the
spec as `x-budget-override` and printed. The private homes hold no record of the job itself and no ledger."""
import importlib.util
import json
import shutil
from pathlib import Path

import pytest
import yaml

SCRIPT = Path('E:/tr/tyc_sftv2aliyun/reports/infrarsi_newmodels_20261028/scripts/make_f1_specs.py')
NM = Path('E:/tr/tyc_sftv2aliyun/reports/infrarsi_textbook_goal_20261024/scripts/pc/nm')
BASES = ('E16-code', 'M35H4-code-env-opus', 'Q9U-code-env-opus')

pytestmark = pytest.mark.skipif(not SCRIPT.is_file() or not all((NM / f'{b}.yaml').is_file() for b in BASES),
                                reason='the campaign scripts and base specs are not on this PC')


def script():
    spec = importlib.util.spec_from_file_location('make_f1_specs_under_test', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pack(path: Path, pid: str, model: str) -> None:
    lines = [{'kind': 'pack', 'id': 'pack', 'schema': 'infrarsi-exp/1', 'pack_id': pid, 'title': pid, 'license': 'x',
              'tier': 'verified'},
             {'kind': 'env', 'id': 'env', 'gpu': 'H100', 'gpus_per_node': 8, 'nodes': 1, 'framework': 'other',
              'model': {'name': model, 'params_b': 9.0}, 'workload': {'seq_len': 4096}}]
    path.write_text('\n'.join(json.dumps(x) for x in lines) + '\n', encoding='utf-8')


@pytest.fixture
def sandbox(tmp_path):
    nm = tmp_path / 'nm'
    nm.mkdir()
    for b in BASES:
        shutil.copyfile(NM / f'{b}.yaml', nm / f'{b}.yaml')
    homes = tmp_path / 'homes'
    own = {'E16': ['dsv4-v16-20261029-010101', 'dsv4-e16-code-20261101-010101'],
           'M35H4': ['m35-winh4n-code-env-opus-20261102-010101'], 'Q9U': ['q9-unicom1-code-env-opus-20261101-010101']}
    for side, records in own.items():
        runs = homes / side / 'experience' / 'runs'
        runs.mkdir(parents=True)
        pack(runs / 'g4-unicom1-code-env-opus-20261029-075745.jsonl', 'run-g4-unicom1-code-env-opus-20261029-075745',
             'g4-unicom1-code-env-opus')
        for r in records:
            pack(runs / f'{r}.jsonl', f'run-{r}', r)
        pack(runs / 'renamed.jsonl', f'run-{records[0]}-copy', records[0])      # (caught by its content)
        (homes / side / 'ledger.jsonl').write_text('{}\n', encoding='utf-8')
    trials = tmp_path / 'trials'
    trials.mkdir()                                                           # (no finished arm: an empty brief)
    return nm, homes, trials


def test_the_d5_overrides_are_applied_recorded_and_printed(sandbox, capsys):
    nm, homes, trials = sandbox
    mod = script()
    assert mod.BUDGET_OVERRIDE == {'E16': {'manifest.search.budget.wall_hours': 12, 'manifest.search.budget.gpu_hours': 192,
                                           'steps.run.timeout_s': 50400},
                                   'M35H4': {'manifest.search.budget.wall_hours': 5}}
    rc = mod.main(['bcomp', 'b1', 'pc_bin_f1', str(homes), 'E16-code=winh6,winh7', 'M35H4-code-env-opus',
                   'Q9U-code-env-opus', '--nm', str(nm), '--trials', str(trials), '--cards', 'none'])
    assert rc == 0
    out = capsys.readouterr().out
    e16 = yaml.safe_load((nm / 'E16-code-env-opus-b1.yaml').read_text(encoding='utf-8'))
    b = e16['manifest']['search']['budget']
    run = next(s for s in e16['steps'] if s['name'] == 'run')
    assert (b['screens'], b['wall_hours'], b['gpu_hours'], run['timeout_s']) == (10, 12, 192, 50400)
    assert '--strategy bcomp ' in run['cmd'] and 'qa_trials/pc_bin_f1/infrarsi' in run['cmd']
    assert e16['x-budget-override'] == {'source': 'FINALISTS_IMPL_SPEC §B.7 D5', 'rule': 'the screens are the only binding budget',
                                        'changes': {'manifest.search.budget.wall_hours': [4.8, 12],
                                                    'manifest.search.budget.gpu_hours': [77, 192],
                                                    'steps.run.timeout_s': [25200, 50400]}}
    assert e16['id'] == 'E16-code-env-opus-b1' and e16['manifest']['name'] == 'dsv4-e16-code-b1'
    assert e16['x-hosts'] == ['winh6', 'winh7'] and e16['manifest']['execution']['hosts'] == ['winh6', 'winh7']
    m35 = yaml.safe_load((nm / 'M35H4-code-env-opus-b1.yaml').read_text(encoding='utf-8'))
    mb = m35['manifest']['search']['budget']
    assert mb['wall_hours'] == 5 and 'gpu_hours' not in mb
    assert next(s for s in m35['steps'] if s['name'] == 'run')['timeout_s'] == 21600
    assert m35['x-budget-override']['changes'] == {'manifest.search.budget.wall_hours': [3, 5]}
    q9 = yaml.safe_load((nm / 'Q9U-code-env-opus-b1.yaml').read_text(encoding='utf-8'))
    base = yaml.safe_load((NM / 'Q9U-code-env-opus.yaml').read_text(encoding='utf-8'))
    assert 'x-budget-override' not in q9 and q9['manifest']['search']['budget'] == base['manifest']['search']['budget']
    assert next(s for s in q9['steps'] if s['name'] == 'run')['timeout_s'] == \
        next(s for s in base['steps'] if s['name'] == 'run')['timeout_s']
    assert out.count('budget override (FINALISTS_IMPL_SPEC §B.7 D5)') == 2 and 'gpu_hours 77 -> 192' in out
    assert 'no budget override' in out
    for side, prefix in (('E16', 'dsv4-'), ('M35H4', 'm35-winh4n'), ('Q9U', 'q9-unicom')):
        home = homes / f'{side}-b1'
        names = sorted(p.name for p in (home / 'experience' / 'runs').iterdir())
        assert names == ['g4-unicom1-code-env-opus-20261029-075745.jsonl'], names   # only the other job's record
        assert not (home / 'ledger.jsonl').exists() and (home / 'experience' / 'transfer_brief.json').is_file()
        spec = yaml.safe_load((nm / f'{side}-code-env-opus-b1.yaml').read_text(encoding='utf-8'))
        assert spec['env']['INFRARSI_HOME'] == str(home).replace('\\', '/') and spec['manifest']['name'].startswith(prefix)


def test_it_refuses_a_wrong_suffix_an_existing_home_and_a_cards_arm_without_cards(sandbox):
    nm, homes, trials = sandbox
    mod = script()
    with pytest.raises(AssertionError, match='arms are -m'):
        mod.main(['beam', 'b1', 'pc_bin_f2', str(homes), 'Q9U-code-env-opus', '--nm', str(nm), '--trials', str(trials)])
    if not (mod.HERE / 'mine_mechanism_cards.py').is_file():
        with pytest.raises(SystemExit, match='mine_mechanism_cards'):
            mod.main(['cards', 'k1', 'pc_bin_f4', str(homes), 'Q9U-code-env-opus', '--nm', str(nm), '--trials', str(trials)])
        assert not (homes / 'Q9U-k1').exists()                               # (refused before any home is built)
    (homes / 'Q9U-m1').mkdir()
    with pytest.raises(AssertionError, match='never overwritten'):
        mod.main(['beam', 'm1', 'pc_bin_f2', str(homes), 'Q9U-code-env-opus', '--nm', str(nm), '--trials', str(trials)])
    assert not (nm / 'Q9U-code-env-opus-m1.yaml').exists()
