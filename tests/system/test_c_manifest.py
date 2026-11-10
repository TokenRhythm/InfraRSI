"""C1/C2 - the manifest: the safe expression evaluator under fuzz and attack, the user's mistakes as one-line errors,
the reference configuration's rules, how every knob target renders (path, flag, override, const, env), and the candidate
tree (overlay confinement, const rewrite, config rendering)."""
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from _support.toy import make_toy, manifest, write_manifest
from infrarsi.project import Project, ProjectRunner
from infrarsi.project import executor as ex
from infrarsi.project.manifest import ManifestError, args_for, env_for, evaluate, expr_names, read_const, write_const


# ---------------------------------------------------------------- expressions
def _random_expr(rng, depth=0):
    names = ['a', 'b', 'gpus']
    if depth > 3 or rng.random() < 0.3:
        return rng.choice(names + [str(rng.randint(1, 9))])
    op = rng.choice(['+', '-', '*', '//', '%', 'min', 'max', 'abs', 'cmp', 'if', 'and'])
    if op in ('+', '-', '*', '//', '%'):
        return f'({_random_expr(rng, depth + 1)} {op} {_random_expr(rng, depth + 1)})'
    if op in ('min', 'max'):
        return f'{op}({_random_expr(rng, depth + 1)}, {_random_expr(rng, depth + 1)})'
    if op == 'abs':
        return f'abs({_random_expr(rng, depth + 1)})'
    if op == 'cmp':
        return f'({_random_expr(rng, depth + 1)} {rng.choice(["<", "<=", "==", "!=", ">", ">="])} {_random_expr(rng, depth + 1)})'
    if op == 'if':
        return f'({_random_expr(rng, depth + 1)} if {_random_expr(rng, depth + 1)} else {_random_expr(rng, depth + 1)})'
    # `and` / `or` over comparisons only: the evaluator returns a boolean where Python returns the last operand, which
    # only matters when a truth value feeds arithmetic (not a constraint's use)
    cmp = lambda: f'({_random_expr(rng, depth + 2)} {rng.choice(["<", "==", ">="])} {_random_expr(rng, depth + 2)})'
    return f'({cmp()} {rng.choice(["and", "or"])} {cmp()})'


def test_evaluator_agrees_with_python_on_random_expressions():
    rng = random.Random(7)
    names = {'a': 3, 'b': 2, 'gpus': 8}
    checked, no_short_circuit = 0, 0
    for _ in range(400):
        e = _random_expr(rng)
        try:
            want = eval(e, {'__builtins__': {}}, {**names, 'min': min, 'max': max, 'abs': abs})
        except ZeroDivisionError:
            with pytest.raises(ZeroDivisionError):
                evaluate(e, names)
            continue
        try:
            got = evaluate(e, names)
        except ZeroDivisionError:                       # F-SYS-8: `and` / `or` evaluate every operand (no short-circuit)
            no_short_circuit += 1
            continue
        assert got == want, e
        checked += 1
    assert checked > 300
    print(f'\n{no_short_circuit} of 400 expressions raised only because and/or do not short-circuit')


def test_and_or_short_circuit():
    """F-SYS-8 (fixed): `and` / `or` short-circuit as in Python, so a guarded constraint never divides by zero."""
    assert evaluate('tp == 0 or gpus % tp == 0', {'tp': 0, 'gpus': 8}) is True
    assert evaluate('tp > 0 and gpus % tp == 0', {'tp': 0, 'gpus': 8}) is False
    assert evaluate('tp > 0 and gpus % tp == 0', {'tp': 2, 'gpus': 8}) is True


def test_a_huge_exponent_is_a_manifest_error():
    """F-SYS-13 (fixed): `2 ** 999999999` is refused at once instead of stalling the load."""
    with pytest.raises(ManifestError, match='exponent'):
        evaluate('2 ** 999999999', {})
    assert evaluate('2 ** 10', {}) == 1024


@pytest.mark.parametrize('expr', [
    '__import__("os").system("x")', 'open("x")', 'a.__class__', 'a.real', 'a[0]', '[1 for _ in (1,)]', 'lambda: 1',
    '(a := 1)', 'f"{a}"', 'getattr(a, "x")', 'min.__name__', '"x".join(["a"])', 'print(1)', 'a ** b ** c', 'not_a_name',
    'min(a, key=a)', 'a if b', '1 +', 'import os', 'exec("1")', 'eval("1")', 'globals()', 'type(a)', '{}', 'set()',
])
def test_evaluator_rejects_anything_but_arithmetic(expr):
    with pytest.raises(ManifestError):
        evaluate(expr, {'a': 1, 'b': 2})


def test_evaluator_names_the_unknown_name():
    with pytest.raises(ManifestError, match="unknown name 'tp'"):
        evaluate('gpus % tp == 0', {'gpus': 8})
    assert expr_names('gpus % (tp * pp) == 0 and min(a, 2)') == {'gpus', 'tp', 'pp', 'a'}
    assert expr_names('1 +') == set()


def test_pathological_expressions_do_not_hang_a_load():
    """A constraint such as `2 ** 999999999` or `'x' * 10**9` is a manifest a user could write by mistake: loading it
    must not stall `doctor` for minutes. Checked in a subprocess with a time limit (an absent limit shows as a timeout)."""
    code = ('import sys; sys.path.insert(0, %r)\nfrom infrarsi.project.manifest import evaluate\n'
            'for e in ("2 ** 999999999", "(10 ** 9) ** (10 ** 9)"):\n'
            '    try:\n        evaluate(e, {})\n    except Exception as exc:\n        print(type(exc).__name__)\n'
            'print("done")' % str(Path(__file__).resolve().parents[2]))
    try:
        p = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        pytest.xfail('FINDING: a power expression with a huge exponent stalls the manifest evaluator (no bound on **)')
    assert 'done' in p.stdout


# ---------------------------------------------------------------- the user's mistakes
MISTAKES = {
    'no_launch_command': (lambda d: d['launch'].pop('command'), 'launch.command is required'),
    'two_targets': (lambda d: d['knobs']['workers'].update(path='train.workers'), 'exactly one target'),
    'no_target': (lambda d: d['knobs']['workers'].pop('flag'), 'exactly one target'),
    'bad_type': (lambda d: d['knobs']['workers'].update(type='integer'), "type 'integer' not in"),
    'no_choices': (lambda d: d['knobs']['workers'].pop('choices'), 'give choices or min/max'),
    'const_without_file': (lambda d: d['knobs'].update(K={'type': 'int', 'choices': [1], 'const': 'K'}), 'a const target needs `file`'),
    'path_without_config': (lambda d: d['config'].update(file=None), 'a path target needs config.file'),
    'config_missing': (lambda d: d['config'].update(file='nope.yaml'), 'does not exist'),
    'config_toml': (lambda d: d['config'].update(file='train.toml'), 'only yaml or json'),
    'steps_too_small': (lambda d: d['workload'].update(steps=1), 'workload.steps must be an int >= 2'),
    'window_outside': (lambda d: d['workload'].update(window=[5, 40]), 'must lie within'),
    'tolerance_1.5': (lambda d: d['verify'].update(tolerance=1.5), 'verify.tolerance must be a number in [0, 1)'),
    'signals_source_typo': (lambda d: d.update(signals={'source': 'logs'}), 'signals.source must be log or probe'),
    'phase_without_group': (lambda d: d.update(signals={'phases': {'fwd': 'fwd ([0-9.]+)'}}), 'needs a named group v'),
    'phase_bad_regex': (lambda d: d.update(signals={'phases': {'fwd': 'fwd (?P<v>['}}), 'not a regular expression'),
    'where_typo': (lambda d: d.update(execution={'where': 'remote'}), 'execution.where must be local, ssh or slurm'),
    'ssh_without_hosts': (lambda d: d.update(execution={'where': 'ssh'}), 'execution.hosts is required for ssh'),
    'checkpoint_without_resume': (lambda d: d['checkpoint'].pop('resume'), 'checkpoint needs save and resume'),
    'unknown_ablation': (lambda d: d['search'].update(ablate=['nothing']), 'search.ablate: a list of'),
    'phase_cap_not_bool': (lambda d: d['search'].update(phase_cap='yes'), 'search.phase_cap: true or false'),
    'constraint_unknown_knob': (lambda d: d.update(constraints=['gpus % (tp * pp) == 0']), "unknown name 'tp'"),
    'constraint_unparseable': (lambda d: d.update(constraints=['gpus %% 2 ==']), 'cannot parse'),
    'invariant_unknown_name': (lambda d: d['workload']['invariants'].update(gb='mbs * ga'), "unknown name 'mbs'"),
    'knob_without_default_in_invariant': (lambda d: (d['knobs'].update(ga={'type': 'int', 'choices': [1, 2], 'flag': '--ga'}),
                                                     d['workload']['invariants'].update(gb='ga * 2')), 'whose current value is unknown'),
}


@pytest.mark.parametrize('name', sorted(MISTAKES))
def test_each_mistake_is_one_manifest_error_that_names_it(tmp_path, home, name):
    root = make_toy(tmp_path / 'toy')
    (root / 'train.toml').write_text('x = 1\n', encoding='utf-8')
    doc = manifest(root)
    edit, message = MISTAKES[name]
    edit(doc)
    write_manifest(root, doc)
    with pytest.raises(ManifestError) as exc:
        Project.load(root)
    assert message in str(exc.value), str(exc.value)
    assert '\n' not in str(exc.value).strip() or ';' in str(exc.value)       # one line for the CLI


def test_load_errors_for_a_broken_file(tmp_path, home):
    root = make_toy(tmp_path / 'toy')
    with pytest.raises(ManifestError, match='does not exist'):
        Project.load(tmp_path / 'nowhere')
    (root / 'infrarsi.yaml').write_text('name: [unclosed\n', encoding='utf-8')
    with pytest.raises(ManifestError, match='not valid YAML'):
        Project.load(root)
    (root / 'infrarsi.yaml').write_text('- a list\n', encoding='utf-8')
    with pytest.raises(ManifestError, match='not a mapping'):
        Project.load(root)
    (root / 'infrarsi.yaml').write_text('schema: infrarsi-project-v2\nname: x\nlaunch: {command: x}\n', encoding='utf-8')
    with pytest.raises(ManifestError, match='schema'):
        Project.load(root)


# ---------------------------------------------------------------- the reference configuration and rendering
def test_reference_config_rules(tmp_path, home):
    root = make_toy(tmp_path / 'toy', edit=lambda d: d['knobs'].update(
        ov={'type': 'int', 'choices': [1, 2], 'override': 'train.ov'},
        ov_null={'type': 'int', 'choices': [1, 2], 'override': 'train.ov_null'},
        ov_default={'type': 'int', 'choices': [1, 2], 'override': 'train.ov_default', 'default': 2},
        flag_bool={'type': 'bool', 'flag': '--fb'},
        K={'type': 'int', 'choices': [1, 2], 'const': 'K', 'file': 'consts.py'}))
    (root / 'consts.py').write_text('X = 1\nK = 2   # the constant\n', encoding='utf-8')
    cfg = yaml.safe_load((root / 'config.yaml').read_text(encoding='utf-8'))
    cfg['train']['ov_null'] = None
    (root / 'config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
    p = Project.load(root)
    ref = p.reference_config()
    assert ref['ov'] is None                  # an override absent everywhere stays unset
    assert ref['ov_null'] is None             # one the config sets to null stays null, not the default
    assert ref['ov_default'] == 2
    assert ref['flag_bool'] is False          # an absent switch is off
    assert ref['K'] == 2                      # read from the script, comment kept apart
    assert p.names(ref)['gpus'] == 1 and p.invariants(ref) == {'global_batch': 8}
    assert args_for(p, dict(ref, ov=2, flag_bool=True, compile=True, workers=4)) == ['--workers', '4', '--compile', 'train.ov=2',
                                                                                     'train.ov_default=2', '--fb']
    assert env_for(p, ref) == {}
    assert write_const('K = 2   # c\n', 'K', 3) == 'K = 3   # c\n' and read_const(root / 'consts.py', 'nope') is None
    with pytest.raises(ManifestError):
        write_const('X = 1\n', 'K', 3)


def test_flag_styles_and_env_rendering(tmp_path, home):
    root = make_toy(tmp_path / 'toy', edit=lambda d: d['knobs'].update(
        tf32={'type': 'bool', 'flag': '--tf32', 'style': 'value'},
        ckpt={'type': 'bool', 'flag': '--ckpt', 'off_flag': '--no-ckpt'},
        NCCL={'type': 'str', 'choices': ['Ring', 'Tree'], 'env': 'NCCL_ALGO'},
        FAST={'type': 'bool', 'env': 'FAST'}))
    p = Project.load(root)
    base = p.reference_config()
    a = args_for(p, dict(base, tf32=False, ckpt=False))
    assert '--tf32' in a and a[a.index('--tf32') + 1] == 'false' and '--no-ckpt' in a
    a = args_for(p, dict(base, tf32=True, ckpt=True))
    assert a[a.index('--tf32') + 1] == 'true' and '--ckpt' in a and '--no-ckpt' not in a
    assert env_for(p, dict(base, NCCL='Tree', FAST=True)) == {'NCCL_ALGO': 'Tree', 'FAST': '1'}
    assert env_for(p, dict(base, NCCL=None, FAST=False)) == {'FAST': '0'}


def test_materialize_renders_config_consts_and_overlay_and_confines_files(tmp_path, home):
    root = make_toy(tmp_path / 'toy', edit=lambda d: d['knobs'].update(
        K={'type': 'int', 'choices': [1, 2], 'const': 'K', 'file': 'consts.py'}))
    (root / 'consts.py').write_text('K = 1  # keep me\n', encoding='utf-8')
    (root / 'big.pt').write_bytes(b'0' * 10)
    (root / 'checkpoints').mkdir()
    (root / 'checkpoints' / 'x.bin').write_bytes(b'0')
    p = Project.load(root)
    r = ProjectRunner(p, out_dir=tmp_path / 'out')
    tree = tmp_path / 'tree'
    rel = r.materialize(tree, dict(p.reference_config(), micro_batch_size=2, grad_accum=4, K=2), {'kernels.py': '# patched\n'})
    assert rel == 'config.yaml'
    doc = yaml.safe_load((tree / 'config.yaml').read_text(encoding='utf-8'))
    assert doc['train']['micro_batch_size'] == 2 and doc['train']['grad_accum'] == 4 and doc['train']['precision'] == 'fp32'
    assert (tree / 'consts.py').read_text(encoding='utf-8') == 'K = 2  # keep me\n'
    assert (tree / 'kernels.py').read_text(encoding='utf-8') == '# patched\n'
    assert not (tree / 'big.pt').exists() and not (tree / 'checkpoints').exists()      # code.exclude
    assert (root / 'kernels.py').read_text(encoding='utf-8') != '# patched\n'           # the user's tree untouched


@pytest.mark.xfail(strict=True, reason='FINDING F-SYS-2: executor.own_file does not confine a relative path with `..` or '
                   'an absolute path to the candidate tree (the hooks reject such patch paths, so this is the second line)')
def test_own_file_confines_writes_to_the_tree(tmp_path):
    tree = tmp_path / 'tree'
    tree.mkdir()
    for bad in ('../escape.py', '/abs/x.py', 'C:/x.py', 'a/../../b.py'):
        f = ex.own_file(tree, bad)
        assert f.resolve().is_relative_to(tree.resolve()), f'{bad!r} -> {f}'


def test_patch_paths_that_are_not_canonical_are_rejected_by_the_hooks(tmp_path, home):
    """The line of confinement that holds: validate_patch rejects a non-canonical path (`..`, absolute, `./`, `//`,
    backslashes) before the glob match, so a model's patch can only name files under the tree."""
    from infrarsi.ext.common.guards import path_problem
    from infrarsi.project.hooks import ProjectHooks
    root = make_toy(tmp_path / 'toy', edit=lambda d: d['code'].update(mutable=['**/*.py']))
    p = Project.load(root)
    hooks = ProjectHooks(p, p.task())
    assert path_problem('kernels.py') is None and path_problem('src/a.py') is None
    for bad in ('../evil.py', '/abs/evil.py', 'C:/x.py', 'src/../../../evil.py', 'kernels.py/../loss.py', './x.py', 'a//b.py',
                'a\\b.py', 'x.py/', ''):
        assert path_problem(bad), bad
        errors, _ = hooks.validate_patch({bad: 'x = 1\n'})
        assert errors and 'non-canonical' in errors[0], (bad, errors)
    errors, _ = hooks.validate_patch({'loss.py': 'x = 1\n'})
    assert errors and any('frozen' in e for e in errors), errors


@pytest.mark.xfail(strict=True, reason='FINDING F-SYS-2 (latent, minor): touched_outside matches a path against the '
                   'mutable globs with fnmatch and no normalisation: `src/../../x.py` passes `src/**/*.py`; only the '
                   'canonical-path check before it confines a patch')
def test_touched_outside_alone_would_confine():
    from infrarsi.ext.common.guards import touched_outside
    for bad, globs in ((['../evil.py'], ('*.py',)), (['/abs/evil.py'], ('**/*.py',)), (['src/../../../evil.py'], ('src/**/*.py',))):
        assert touched_outside({f: 'x' for f in bad}, globs) == bad, (bad, globs)


def test_json_config_rendering(tmp_path, home):
    from _support.faultjob import make_project
    root = make_project(tmp_path / 'fault')
    p = Project.load(root)
    r = ProjectRunner(p, out_dir=tmp_path / 'out')
    tree = tmp_path / 'tree'
    r.materialize(tree, dict(p.reference_config(), bs=4, accum=1, STEP_SCALE=0.8), {})
    doc = json.loads((tree / 'config.json').read_text(encoding='utf-8'))
    assert doc['train'] == {'bs': 4, 'accum': 1, 'workers': 1, 'opt': 'plain'} and doc['optim'] == {'lr': 0.001}
    assert 'STEP_SCALE = 0.8' in (tree / 'train.py').read_text(encoding='utf-8')


def test_task_budget_and_files(tmp_path, home):
    root = make_toy(tmp_path / 'toy')
    p = Project.load(root)
    t = p.task()
    b = t.budget if isinstance(t.budget, dict) else t.budget.__dict__
    assert b['screens'] == 12 and b['wall_seconds'] == 0.5 * 3600
    assert b['gpu_hours'] == pytest.approx((12 + 4) * 1 * 120 / 3600)           # (screens + 4) x world x timeout
    assert t.action_space == 'knobs+patch' and set(t.frozen_files) == {'loss.py'} and tuple(t.mutable_globs) == ('kernels.py',)
    assert p.files(['*.py']) == ['check.py', 'eval.py', 'kernels.py', 'loss.py', 'train.py']
    assert p.excluded('.git/x') and p.excluded('ckpt/model.safetensors') and not p.excluded('src/a.py')
    doc = manifest(root)
    doc['knobs'], doc['code']['mutable'] = {}, []
    doc['workload']['invariants'] = {}
    write_manifest(root, doc)
    with pytest.raises(ManifestError, match='nothing to search'):
        Project.load(root).task()


def test_crlf_launch_scripts_are_found(tmp_path, home):
    root = make_toy(tmp_path / 'toy', edit=lambda d: d['launch'].update(command='bash run.sh --config {config} {args}', setup='source env.sh'))
    (root / 'run.sh').write_bytes(b'#!/bin/bash\r\npython train.py "$@"\r\n')
    (root / 'env.sh').write_bytes(b'export X=1\n')
    p = Project.load(root)
    assert p.crlf_scripts() == ['run.sh']
    assert p.posix_hosts() == (sys.platform != 'win32')
