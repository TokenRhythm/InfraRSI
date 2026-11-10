"""N1/N5 - non-functional: InfraRSI's own overhead (tree copies with the excludes, a chatty job's log, a long log) and a
fresh install without the optional dependencies. Both opt-in (slow / packaging markers)."""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from _support.toy import latest_run, make_toy, result
from infrarsi.project import Project, ProjectRunner
from infrarsi.project import executor as ex
from infrarsi.project.flow import make_verifier

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_tree_copy_skips_checkpoints_and_stays_fast(tmp_path):
    src = tmp_path / 'repo'
    for i in range(3000):
        p = src / f'pkg{i % 30}' / f'm{i}.py'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f'x = {i}\n', encoding='utf-8')
    (src / 'checkpoints').mkdir()
    (src / 'checkpoints' / 'big.pt').write_bytes(b'0' * (50 << 20))
    (src / 'model.safetensors').write_bytes(b'0' * (50 << 20))
    (src / '.git').mkdir()
    (src / '.git' / 'objects').write_bytes(b'0' * (10 << 20))
    excluded = Project({'name': 'x', 'launch': {'command': 'true'}}, src / 'infrarsi.yaml').excluded
    t0 = time.perf_counter()
    ex.copy_tree(src, tmp_path / 'tree', excluded)
    dt = time.perf_counter() - t0
    assert not (tmp_path / 'tree' / 'checkpoints').exists() and not (tmp_path / 'tree' / 'model.safetensors').exists()
    assert not (tmp_path / 'tree' / '.git').exists()
    assert len(list((tmp_path / 'tree').rglob('*.py'))) == 3000
    assert dt < 30, f'copying 3000 small files took {dt:.1f} s'


@pytest.mark.slow
def test_a_chatty_job_does_not_distort_the_clock(tmp_path, home):
    root = make_toy(tmp_path / 'toy')
    train = (root / 'train.py').read_text(encoding='utf-8')
    chatty = train.replace("        print(f'step {step}", "        for _ in range(200): print('debug: a chatty line of no consequence', flush=True)\n        print(f'step {step}")
    (root / 'train.py').write_text(chatty, encoding='utf-8')
    p = Project.load(root)
    runner = ProjectRunner(p, out_dir=root / '.infrarsi' / 't')
    rec = runner.run(p.task(), p.reference_config(), None, 'reference', 'r', candidate_id='c0')
    v = make_verifier(p)(rec, None, rec)
    assert v.valid and rec['evidence']['steps_completed'] == 12
    assert rec['evidence']['rsd_window'] < 0.25, rec['evidence']['rsd_window']


@pytest.mark.slow
def test_a_long_log_is_scanned_in_seconds():
    from infrarsi.project import signals as sig
    lines = [(float(i) * 0.1, f'step {i} | loss {2.0 - i * 1e-7:.7f} | grad_norm 1.1 | lr 1e-4 | tokens 4096') for i in range(1, 200001)]
    t0 = time.perf_counter()
    rows = sig.LogParser().parse(lines)
    dt = time.perf_counter() - t0
    assert len(rows) == 200000 and dt < 120, f'200k lines took {dt:.0f} s'
    t0 = time.perf_counter()
    ev = sig.evidence(rows, steps=200000, window=[1000, 200000], exit_code=0, timed_out=False, events={}, frozen={})
    assert ev['steps_completed'] == 200000 and time.perf_counter() - t0 < 60


@pytest.mark.packaging
def test_fresh_install_without_optional_dependencies(tmp_path):
    venv = tmp_path / 'venv'
    subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
    bindir = venv / ('Scripts' if os.name == 'nt' else 'bin')
    py = bindir / ('python.exe' if os.name == 'nt' else 'python')
    # a real (non-editable) install, run through its console script from a directory outside the checkout with no
    # PYTHONPATH: what pip ships is what runs (the package data - seed packs, the profile hook - included)
    r = subprocess.run([str(py), '-m', 'pip', 'install', '-q', str(REPO)], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    exe = bindir / ('infrarsi.exe' if os.name == 'nt' else 'infrarsi')
    neutral = tmp_path / 'elsewhere'
    neutral.mkdir()
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    env.update(INFRARSI_HOME=str(tmp_path / 'home'), TOY_TIME_SCALE='0.004')
    r = subprocess.run([str(exe), '--help'], capture_output=True, text=True, cwd=neutral, env=env)
    assert r.returncode == 0 and 'doctor' in r.stdout
    r = subprocess.run([str(py), '-c', 'import infrarsi, infrarsi.project.profhook as h, pathlib; '
                        'print(pathlib.Path(infrarsi.__file__).parent.resolve()); '
                        'print(pathlib.Path(h.__file__).with_name("sitecustomize.py").is_file())'], capture_output=True, text=True, cwd=neutral, env=env)
    assert str(REPO.resolve()) not in r.stdout and 'True' in r.stdout, r.stdout + r.stderr   # the installed copy, with its hook
    r = subprocess.run([str(exe), 'exp', 'stats'], capture_output=True, text=True, cwd=neutral, env=env, timeout=600)
    assert r.returncode == 0 and ' packs:' in r.stdout and 'skipped' not in r.stdout, r.stdout + r.stderr   # the seeds shipped
    root = make_toy(tmp_path / 'toy', screens=2)
    r = subprocess.run([str(exe), 'run', str(root), '--llm', 'none', '--strategy', 'random'], capture_output=True, text=True, cwd=neutral, env=env, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert result(latest_run(root))['outcome'] in ('improved', 'no_change')
    r = subprocess.run([str(exe), 'run', str(root), '--llm', 'none', '--strategy', 'tpe'], capture_output=True, text=True, cwd=neutral, env=env, timeout=600)
    assert r.returncode != 0 and 'optuna' in (r.stdout + r.stderr) and 'Traceback' not in (r.stdout + r.stderr)
    r = subprocess.run([str(exe), 'run', str(root), '--llm', 'none'], capture_output=True, text=True, cwd=neutral, env=env, timeout=600)
    assert r.returncode == 0 and result(latest_run(root))['strategy'] == 'random'       # auto without optuna
