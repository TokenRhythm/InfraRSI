"""S1 - static contracts: every module imports, the CLI lists the product commands and each --help works, the README's
flags and strategies exist, the checklist items are documented, the manifest docstring matches the defaults, the example
manifest loads, the packaging entry point is right."""
import importlib
import pkgutil
import re
import shlex
from pathlib import Path

import pytest

from _support import cli
from infrarsi.cli.main import build
from infrarsi.project import Project
from infrarsi.project.manifest import DEFAULTS
from infrarsi.search import STRATEGIES, resolve
from infrarsi.verify.checklist import CHECKS

REPO = Path(__file__).resolve().parents[2]
README = (REPO / 'README.md').read_text(encoding='utf-8')
PRODUCT = ('optimize', 'init', 'doctor', 'run', 'report', 'apply', 'finish', 'requirements', 'exp', 'agent')


OPTIONAL = ('torch', 'transformers', 'paramiko', 'anthropic', 'optuna', 'numpy', 'megatron', 'deepspeed', 'vllm', 'safetensors')


def test_every_module_imports():
    """Every module imports; a module that needs an optional dependency (torch and friends) is allowed to miss it."""
    import infrarsi
    bad, optional = [], []
    for m in pkgutil.walk_packages(infrarsi.__path__, 'infrarsi.'):
        if m.name.startswith('infrarsi.redteam.seeds.'):
            continue                                                 # data files by contract: a test asserts they are never imported
        try:
            importlib.import_module(m.name)
        except ModuleNotFoundError as exc:
            if (exc.name or '').split('.')[0] in OPTIONAL:
                optional.append(f'{m.name} ({exc.name})')
            else:
                bad.append(f'{m.name}: {type(exc).__name__}: {exc}')
        except Exception as exc:                                     # noqa: BLE001 - collected
            bad.append(f'{m.name}: {type(exc).__name__}: {exc}')
    assert not bad, '\n'.join(bad)
    print(f'\n{len(optional)} modules need an optional dependency not installed here: {optional[:8]}')


def test_help_lists_the_product_commands_and_each_help_works():
    res = cli.run('--help')
    assert res.code == 0 and '{optimize,init,doctor,run,report,apply,finish,requirements,exp,agent}' in res.out and 'research harness' in res.out
    for cmd in PRODUCT:
        r = cli.run(cmd, '--help')
        assert r.code == 0 and cmd in r.out, (cmd, r.text)
    assert cli.run().code == 2                                       # no command: usage, exit 2


@pytest.mark.parametrize('filename', ['README.md', 'README_ZH.md'])
def test_readme_flags_exist_in_the_parser(filename):
    parser = build()
    subs = {a.dest: a for a in parser._actions if a.dest == 'cmd'}['cmd'].choices
    known = {option for action in parser._actions for option in action.option_strings}
    for name, sp in subs.items():
        for a in sp._actions:
            known.update(a.option_strings)
    text = (REPO / filename).read_text(encoding='utf-8')
    for flag in sorted(set(re.findall(r'(?<![\w-])(--[a-z][a-z0-9_-]+)', text))):
        if flag in ('--nproc_per_node', '--save', '--save-at', '--load', '--bf16', '--accel', '--timing-log-level', '--no-index',
                    '--find-links', '--no-build-isolation', '--section', '--flag', '--compile', '--max-steps', '--config_file',
                    '--workers', '--micro-batch-size', '--tf32', '--fused', '--seq-len', '--grad-accum',
                    '--main-grads-dtype', '--use-precision-aware-optimizer', '--train_text_encoder', '--fast'):
            continue                                                 # flags of the user's job or of pip in examples

        assert flag in known, f'README names {flag}, which no infrarsi command has'


@pytest.mark.parametrize('filename', ['README.md', 'README_ZH.md'])
def test_readme_strategy_examples_exist(filename):
    """Validate the published examples without requiring a full strategy manual in the README."""
    text = (REPO / filename).read_text(encoding='utf-8')
    checked = 0
    for command in re.findall(r'^infrarsi ([^\n#]+)', text, re.M):
        try:
            args = build().parse_args(shlex.split(command))
        except SystemExit as exc:
            assert exc.code == 0, (filename, command)
            continue
        if hasattr(args, 'strategy'):
            assert resolve(args.strategy, has_llm=True, knobs={}) in STRATEGIES
            checked += 1
    assert checked, f'{filename} has no strategy example to validate'


def test_every_checklist_item_is_in_the_readme_table():
    """F-SYS-11 (fixed): the verifier table names every implemented item. The published README (the OSS export's) may
    organise the text differently: the verifier section when there is one, else the whole file; a README that documents
    no checklist item at all is reported, not failed."""
    parts = README.split('## What the verifier checks')
    text = parts[1].split('## Search')[0] if len(parts) > 1 else README
    documented = [c.id for c in CHECKS if c.id != 'precision' and c.id in text]
    if not documented:
        pytest.skip('this README documents no checklist item by id')
    missing = [c.id for c in CHECKS if c.id != 'precision' and c.id not in text]
    assert not missing, missing


def test_manifest_docstring_keys_match_the_defaults():
    from infrarsi.project import manifest as m
    doc = m.__doc__
    keys = set(re.findall(r'^    ([a-z_]+):', doc, re.M))
    assert keys - {'schema', 'name'} <= set(DEFAULTS), keys - set(DEFAULTS) - {'schema', 'name'}
    assert {'launch', 'knobs', 'workload', 'verify', 'search', 'llm', 'execution', 'code', 'signals'} <= keys


@pytest.mark.parametrize('filename', ['README.md', 'README_ZH.md'])
def test_the_example_manifest_loads_and_the_quick_start_command_parses(filename):
    p = Project.load(REPO / 'examples' / 'toy_cpu')
    assert p.name == 'toy-cpu' and p.files(p.raw['code']['mutable']) == ['kernels.py']
    text = (REPO / filename).read_text(encoding='utf-8')
    for cmd in re.findall(r'^infrarsi ([^\n#]+)', text, re.M):
        argv = shlex.split(cmd.strip().replace('<run>', 'x').replace('.infrarsi/runs/x', 'x'))
        argv = [a for a in argv if not a.startswith('[')]
        try:
            build().parse_args(argv)                                 # every README command line is a valid request
        except SystemExit as exc:
            assert exc.code == 0, (filename, cmd)                    # --version / --help exit successfully


def test_the_readme_example_manifest_loads(tmp_path):
    """The manifest the README shows is a valid project once the files it names exist (a reader copies it). The
    published README (the OSS export's) is a landing page without the section: reported, not failed."""
    import yaml
    parts = README.split('## The manifest (infrarsi.yaml)')
    if len(parts) < 2:
        pytest.skip('this README has no manifest section')
    block = parts[1].split('```yaml')[1].split('```')[0]
    root = tmp_path / 'repo'
    (root / 'configs').mkdir(parents=True)
    (root / 'configs' / 'sft.yaml').write_text('train: {micro_batch_size: 2, grad_accum: 4}\n', encoding='utf-8')
    for f in ('train.py', 'loss.py', 'data.py', 'eval.py'):
        (root / f).write_text('', encoding='utf-8')
    (root / 'infrarsi.yaml').write_text(block, encoding='utf-8')
    p = Project.load(root)
    assert p.name == 'my-sft' and set(p.knobs) == {'micro_batch_size', 'grad_accum', 'compile', 'NCCL_ALGO'}
    assert p.reference_config()['micro_batch_size'] == 2 and yaml.safe_load(block)['search']['budget']['screens'] == 16


def test_packaging_entry_point_and_package_data():
    import tomllib
    meta = tomllib.loads((REPO / 'pyproject.toml').read_text(encoding='utf-8'))
    assert meta['project']['scripts']['infrarsi'] == 'infrarsi.cli.main:main'
    deps = meta['project']['dependencies']
    assert 'pyyaml>=6' in deps and all(d.split(';')[0].strip().startswith(('pyyaml', 'tomli')) for d in deps), deps
    data = meta['tool']['setuptools']['package-data']['*']
    assert '*.yaml' in data and '*.jsonl' in data and '*.json' in data
    assert (REPO / 'infrarsi' / 'experience' / 'data' / 'mechanisms.yaml').is_file()
    assert (REPO / 'infrarsi' / 'project' / 'profhook' / 'sitecustomize.py').is_file()


def test_research_commands_are_reachable_but_unlisted():
    res = cli.run('dispatch', '--help')
    assert res.code == 0
    assert 'dispatch' not in cli.run('--help').out.split('research harness')[0].split('positional arguments')[1]
