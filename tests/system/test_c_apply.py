"""C6 - `infrarsi apply`: a dry run shows the diff and the instructions, --write writes the config copy and the patched
files, twice is the same, a user's later edits are merged or refused, --branch uses git, and a run with nothing to apply
says so."""
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from _support import cli
from infrarsi.project import Project


def fake_run(root: Path, *, changes: dict | None = None, overlay: dict | None = None, base: dict | None = None,
             promoted: str | None = 'c1') -> Path:
    """A run directory with the result.json `apply` reads (the promotion's config and overlay)."""
    p = Project.load(root)
    ref = p.reference_config()
    run = root / '.infrarsi' / 'runs' / 'fake-run'
    run.mkdir(parents=True, exist_ok=True)
    promo = ({'promoted': promoted, 'config': dict(ref, **(changes or {})), 'overlay': overlay or {}, 'base': base or {},
              'tried': [], 'speedup': 1.3} if promoted else {'promoted': None, 'why': 'no valid candidate faster than the reference', 'tried': []})
    res = {'schema': 'infrarsi-run-v1', 'run_id': 'fake-run', 'run_dir': str(run), 'manifest': str(root / 'infrarsi.yaml'),
           'outcome': 'improved' if promoted else 'no_change', 'reference': {'config': ref}, 'promotion': promo}
    (run / 'result.json').write_text(json.dumps(res), encoding='utf-8')
    (run / 'REPORT.md').write_text('# fake\n', encoding='utf-8')
    return run


def test_dry_run_shows_the_diff_and_the_instructions(toy):
    run = fake_run(toy, changes={'precision': 'bf16', 'compile': True, 'workers': 4})
    before = (toy / 'config.yaml').read_bytes()
    res = cli.run('apply', run)
    assert res.code == 0, res.text
    assert '+++ b/config.infrarsi.yaml' in res.out and '+  precision: bf16' in res.out
    assert '* add --compile to your launch command' in res.out and '* flag --workers: 1 -> 4' in res.out   # (U-2-8)
    assert 'point your launch command at config.infrarsi.yaml' in res.out
    assert 'dry run: add --write' in res.out
    assert (toy / 'config.yaml').read_bytes() == before and not (toy / 'config.infrarsi.yaml').exists()


def test_write_is_idempotent_and_leaves_the_original_config(toy):
    patched = (toy / 'kernels.py').read_text(encoding='utf-8').replace('chunks * chunks / 4', 'chunks / 2')
    base = {'kernels.py': (toy / 'kernels.py').read_text(encoding='utf-8')}
    run = fake_run(toy, changes={'precision': 'bf16'}, overlay={'kernels.py': patched}, base=base)
    before = (toy / 'config.yaml').read_bytes()
    res = cli.run('apply', run, '--write')
    assert res.code == 0 and 'written:' in res.out, res.text
    assert (toy / 'kernels.py').read_text(encoding='utf-8') == patched
    new = yaml.safe_load((toy / 'config.infrarsi.yaml').read_text(encoding='utf-8'))
    assert new['train']['precision'] == 'bf16' and new['train']['micro_batch_size'] == 1
    assert (toy / 'config.yaml').read_bytes() == before
    again = cli.run('apply', run, '--write')
    assert again.code == 0 and (toy / 'kernels.py').read_text(encoding='utf-8') == patched
    assert 'written:' in again.out
    # the applied project still loads and the new config is the reference when the user points at it
    doc = yaml.safe_load((toy / 'infrarsi.yaml').read_text(encoding='utf-8'))
    doc['config']['file'] = 'config.infrarsi.yaml'
    (toy / 'infrarsi.yaml').write_text(yaml.safe_dump(doc), encoding='utf-8')
    assert Project.load(toy).reference_config()['precision'] == 'bf16'


def test_a_later_edit_elsewhere_is_merged_and_a_conflict_is_refused(toy):
    original = (toy / 'kernels.py').read_text(encoding='utf-8')
    patched = original.replace('chunks * chunks / 4', 'chunks / 2')
    run = fake_run(toy, overlay={'kernels.py': patched}, base={'kernels.py': original})
    # the user edited another line since the search: the change goes on top of the edit
    edited = original.replace('"""Mutable', '"""Edited by the user. Mutable')
    (toy / 'kernels.py').write_text(edited, encoding='utf-8')
    res = cli.run('apply', run, '--write')
    assert res.code == 0, res.text
    text = (toy / 'kernels.py').read_text(encoding='utf-8')
    assert 'Edited by the user' in text and 'chunks / 2' in text


def _conflicting_run(toy):
    original = (toy / 'kernels.py').read_text(encoding='utf-8')
    patched = original.replace('chunks * chunks / 4', 'chunks / 2')
    run = fake_run(toy, overlay={'kernels.py': patched}, base={'kernels.py': original})
    (toy / 'kernels.py').write_text(original.replace('chunks * chunks / 4', 'chunks * 3'), encoding='utf-8')
    return run, original.replace('chunks * chunks / 4', 'chunks * 3')


def test_a_conflicting_edit_writes_nothing(toy):
    run, edited = _conflicting_run(toy)
    try:
        res = cli.run('apply', run, '--write')
        failed, text = res.code != 0, res.text
    except RuntimeError as exc:                                      # see the next test
        failed, text = True, str(exc)
    assert failed and 'does not merge' in text
    assert (toy / 'kernels.py').read_text(encoding='utf-8') == edited


def test_a_conflicting_apply_is_one_line_not_a_traceback(toy):
    """F-SYS-9 (fixed): a merge conflict is a request error - one line on stderr, exit 2."""
    run, _ = _conflicting_run(toy)
    res = cli.run('apply', run, '--write')
    assert res.code == 2 and 'does not merge' in res.err and 'Traceback' not in res.err


def test_env_and_override_knobs_are_instructions(toy):
    doc = yaml.safe_load((toy / 'infrarsi.yaml').read_text(encoding='utf-8'))
    doc['knobs']['NCCL_ALGO'] = {'type': 'str', 'choices': ['Ring', 'Tree'], 'env': 'NCCL_ALGO'}
    doc['knobs']['ac'] = {'type': 'bool', 'override': 'model.activation_checkpointing'}
    (toy / 'infrarsi.yaml').write_text(yaml.safe_dump(doc), encoding='utf-8')
    run = fake_run(toy, changes={'NCCL_ALGO': 'Tree', 'ac': True})
    res = cli.run('apply', run)
    assert res.code == 0 and '* environment NCCL_ALGO=Tree (was None)' in res.out
    assert '* override model.activation_checkpointing=true (was None)' in res.out and '(no file changes)' in res.out


def test_nothing_to_apply_and_wrong_directories(toy, tmp_path):
    run = fake_run(toy, promoted=None)
    res = cli.run('apply', run)
    assert res.code == 0 and res.out.startswith('nothing to apply: no valid candidate faster than the reference')
    res = cli.run('apply', tmp_path)
    assert res.code == 2 and 'not a run directory (no result.json)' in res.err and 'Traceback' not in res.err
    res = cli.run('report', tmp_path)
    assert res.code == 2 and 'not a run directory (no REPORT.md)' in res.err
    res = cli.run('report', run)
    assert res.code == 0 and res.out.startswith('# fake')


def test_without_an_argument_the_newest_run_is_applied(toy):
    """U-1-18: `apply` in the project directory is `apply <its newest run>`, the run named in the first line."""
    run = fake_run(toy, changes={'precision': 'bf16', 'workers': 4})
    named = cli.run('apply', run)
    assert named.code == 0 and named.out.startswith('the change: '), named.text
    res = cli.run('apply', cwd=toy)
    assert res.code == 0 and res.out == f'run: {run.relative_to(toy)}\n' + named.out, res.text
    assert cli.run('apply', toy).out == f'run: {run}\n' + named.out          # the project directory, from elsewhere


def test_branch_writes_on_a_new_git_branch(toy):
    git = lambda *a: subprocess.run(['git', '-C', str(toy), *a], capture_output=True, text=True)
    if git('--version').returncode != 0:
        pytest.skip('git not installed')
    git('init', '-q')
    git('-c', 'user.email=t@t', '-c', 'user.name=t', 'add', '.')
    git('-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-q', '-m', 'base')
    patched = (toy / 'kernels.py').read_text(encoding='utf-8').replace('chunks * chunks / 4', 'chunks / 2')
    run = fake_run(toy, overlay={'kernels.py': patched})
    res = cli.run('apply', run, '--write', '--branch', 'infrarsi/try-1')
    assert res.code == 0, res.text
    assert git('rev-parse', '--abbrev-ref', 'HEAD').stdout.strip() == 'infrarsi/try-1'
    assert 'kernels.py' in git('status', '--porcelain').stdout
    try:                                                             # the branch exists: refused (F-SYS-9: as a traceback)
        res = cli.run('apply', run, '--write', '--branch', 'infrarsi/try-1')
        failed, text = res.code != 0, res.text
    except RuntimeError as exc:
        failed, text = True, str(exc)
    assert failed and 'git checkout -b' in text
