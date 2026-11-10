"""The one-off phase profile (textbook design P2 generic phases, A11, 2026-10-24): the profiler hook's trace -> per
update an exclusive split of the GPU timeline into compute, exposed communication, optimizer and idle."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from infrarsi.project.profhook.sitecustomize import MARK, split

HOOK = Path(__file__).resolve().parents[3] / 'infrarsi' / 'project' / 'profhook'


def _k(ts, end, name):
    return {'ph': 'X', 'cat': 'kernel', 'ts': ts, 'dur': end - ts, 'name': name}


def _opt(ts, end, name='Optimizer.step#AdamW.step'):
    return {'ph': 'X', 'cat': 'gpu_user_annotation', 'ts': ts, 'dur': end - ts, 'name': name}


def test_an_update_splits_into_exclusive_parts():
    ev = [_opt(0, 100), _k(150, 600, 'sm90_xmma_gemm_bf16'), _k(650, 800, 'flash_fwd_kernel'),
          _k(500, 700, 'ncclDevKernel_AllReduce_Sum_bf16_RING_LL'), _opt(900, 1100), _k(950, 1050, 'multi_tensor_apply_kernel'),
          _k(1150, 1900, 'sm90_xmma_gemm_bf16'), _opt(1900, 2100), _k(1950, 2050, 'multi_tensor_apply_kernel')]
    out = split(ev, 2)
    first, second = out['per_update']
    assert first == pytest.approx({'optimizer': 0.1, 'compute': 0.6, 'comm': 0.05, 'idle': 0.25, 'update': 1.0})
    assert second == pytest.approx({'optimizer': 0.1, 'compute': 0.75, 'comm': 0.0, 'idle': 0.15, 'update': 1.0})
    assert out['updates'] == 2 and out['mean_ms']['compute'] == pytest.approx(0.675)


def test_two_optimizers_bound_an_update_by_the_one_that_steps_last():
    ev = [_opt(0, 100, 'Optimizer.step#Muon.step'), _opt(100, 200), _k(300, 800, 'gemm'),
          _opt(800, 900, 'Optimizer.step#Muon.step'), _opt(900, 1000), _k(950, 990, 'adam')]
    out = split(ev, 1)
    assert out['updates'] == 1 and out['per_update'][0]['update'] == pytest.approx(0.8)
    assert out['per_update'][0]['optimizer'] == pytest.approx(0.04) and out['per_update'][0]['compute'] == pytest.approx(0.5)


def test_no_boundaries_no_phases():
    assert 'error' in split([_k(0, 10, 'gemm')], 3) and 'error' in split([], 3)


def _mark(ts, end):
    return {'ph': 'X', 'cat': 'gpu_user_annotation', 'ts': ts, 'dur': end - ts, 'name': MARK}


def test_the_hooks_own_marks_bound_the_updates_whatever_the_optimizers():
    ev = [_mark(95, 100), _opt(10, 20), _opt(40, 50), _k(150, 600, 'gemm'), _opt(600, 650), _opt(650, 700),
          _k(610, 640, 'adam'), _k(660, 690, 'adam'), _mark(995, 1000), _k(1100, 1900, 'gemm'), _mark(1995, 2000)]
    out = split(ev, 2)                              # two AdamW instances per update: the marks decide
    assert out['updates'] == 2 and [r['update'] for r in out['per_update']] == pytest.approx([0.9, 1.0])
    assert out['per_update'][0]['optimizer'] == pytest.approx(0.06)


def test_a_shadowed_sitecustomize_keeps_its_identity_and_the_hook_stays_out_unless_asked(tmp_path):
    user = tmp_path / 'user'
    user.mkdir()
    (user / 'sitecustomize.py').write_text('from __future__ import annotations\nimport dataclasses\n\n'
                                           '@dataclasses.dataclass\nclass Cfg:\n    x: int = 1\n\nVALUE = Cfg().x\n',
                                           encoding='utf-8')
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(HOOK), str(user)])}
    env.pop('INFRARSI_PROFILE', None)
    out = subprocess.run([sys.executable, '-c', 'import sitecustomize; print(sitecustomize.__file__, sitecustomize.VALUE)'],
                         env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and str(user) in out.stdout and out.stdout.strip().endswith('1'), out.stderr
    alone = subprocess.run([sys.executable, '-c', 'import sitecustomize; print(sitecustomize.__file__)'],
                           env={**env, 'PYTHONPATH': str(HOOK)}, capture_output=True, text=True, timeout=120)
    # Debian and managed Python installs can have a system sitecustomize. With
    # profiling off, the hook must preserve that module's identity too.
    system = subprocess.run([sys.executable, '-c', 'import sitecustomize; print(sitecustomize.__file__)'],
                            env={**env, 'PYTHONPATH': ''}, capture_output=True, text=True, timeout=120)
    expected = system.stdout.strip() if system.returncode == 0 else str(HOOK / 'sitecustomize.py')
    assert alone.returncode == 0 and alone.stdout.strip() == expected and not alone.stderr


def test_the_shadowed_sitecustomize_keeps_its_sys_path_changes(tmp_path):
    user = tmp_path / 'user2'
    user.mkdir()
    (user / 'sitecustomize.py').write_text('import sys\nsys.path.insert(0, "/custom/dependencies")\n', encoding='utf-8')
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(HOOK), str(user)])}
    env.pop('INFRARSI_PROFILE', None)
    out = subprocess.run([sys.executable, '-c', 'import sys; print("/custom/dependencies" in sys.path, sys.path[1:3])'],
                         env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.startswith('True'), (out.stdout, out.stderr)
