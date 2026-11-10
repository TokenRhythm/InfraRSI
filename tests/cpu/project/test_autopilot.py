"""Unit and end-to-end tests for InfraRSI one-command autopilot (`infrarsi/project/autopilot.py` & `infrarsi/cli/main.py`)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from infrarsi.cli.main import main
from infrarsi.project.autopilot import (
    auto_heal_manifest,
    default_strategy,
    detect_llm_spec,
    ensure_manifest,
    ensure_shell_forwards,
    is_script_path,
    resolve_script_target,
)
from infrarsi.project.manifest import Project
from infrarsi.search import RESUMABLE, STRATEGIES, resolve

REPO = Path(__file__).resolve().parents[3]
TOY = REPO / 'examples' / 'toy_cpu'


@pytest.fixture(autouse=True)
def no_local_login(monkeypatch):
    monkeypatch.setattr('infrarsi.project.autopilot._logged_in_cli', lambda: None)
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)


def _copy_toy(dst: Path) -> Path:
    shutil.copytree(TOY, dst)
    for name in ('infrarsi.yaml', 'infrarsi_launch.sh', 'infrarsi_optimized.sh'):
        p = dst / name
        if p.exists():
            p.unlink()
    if (dst / '.infrarsi').exists():
        shutil.rmtree(dst / '.infrarsi')
    return dst


def test_strategy_aliases_preserve_bcomp_v41():
    assert resolve('bcompv4.1', has_llm=True) == 'bcomp'
    assert RESUMABLE == ('bcomp',)
    assert 'bcomp' in STRATEGIES


def test_resolve_script_target_handles_root_and_subdir_scripts(tmp_path):
    (tmp_path / 'train.py').write_text('print("ok")\n', encoding='utf-8')
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    sh = scripts / 'run_train.sh'
    sh.write_text('#!/usr/bin/env bash\npython train.py --lr 0.01\n', encoding='utf-8')

    assert is_script_path(sh) is True
    assert is_script_path(tmp_path / 'train.py') is True
    assert is_script_path(tmp_path) is False

    root, rel = resolve_script_target(sh)
    assert root == tmp_path.resolve()
    assert rel == 'scripts/run_train.sh'


def test_ensure_shell_forwards_creates_wrapper_when_dollar_at_missing(tmp_path):
    sh = tmp_path / 'run.sh'
    sh.write_text('#!/usr/bin/env bash\nset -e\npython train.py --batch-size 4 \\\n  --lr 0.01\n', encoding='utf-8')
    rel, note = ensure_shell_forwards(tmp_path, 'run.sh')
    assert rel == 'infrarsi_launch.sh'
    assert note is not None
    wrapped = (tmp_path / rel).read_text(encoding='utf-8')
    assert '"$@"' in wrapped

    # A script that already forwards "$@" is left unchanged
    sh2 = tmp_path / 'run_ok.sh'
    sh2.write_text('#!/usr/bin/env bash\npython train.py "$@"\n', encoding='utf-8')
    rel2, note2 = ensure_shell_forwards(tmp_path, 'run_ok.sh')
    assert rel2 == 'run_ok.sh'
    assert note2 is None


def test_auto_heal_manifest_fills_missing_knob_defaults(tmp_path):
    p = _copy_toy(tmp_path / 'proj')
    doc = {
        'schema': 'infrarsi-project-v1',
        'name': 'broken',
        'launch': {'command': '{python} train.py {args}'},
        'knobs': {
            'batch_size': {'flag': '--batch-size', 'type': 'int', 'choices': [2, 4, 8]},
            'accum': {'flag': '--accum', 'type': 'int', 'choices': [1, 2, 4]},
        },
        'workload': {
            'steps': 10,
            'window': [5, 10],
            'invariants': {'global_batch': 'batch_size * accum'},
        },
        'constraints': [
            'batch_size * accum == 2',
            'batch_size * accum == 999',
        ],
    }
    healed = auto_heal_manifest(doc, p)
    assert healed
    proj = Project(doc, p / 'infrarsi.yaml')
    assert proj.knobs['batch_size']['default'] in (2, 4, 8)
    assert proj.knobs['accum']['default'] in (1, 2, 4)
    assert doc['constraints'] == ['batch_size * accum == 2']


def test_detect_llm_spec_and_default_strategy(tmp_path, monkeypatch):
    p = _copy_toy(tmp_path / 'proj')
    proj, drafted, _ = ensure_manifest(p)
    assert drafted is True
    for k in ('INFRARSI_LLM', 'ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'OPENROUTER_API_KEY', 'VLLM_URL'):
        monkeypatch.delenv(k, raising=False)

    spec_none = detect_llm_spec(proj, None)
    assert spec_none['backend'] == 'none'
    assert default_strategy(None, spec_none, proj.knobs) in ('tpe', 'random')

    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test-key')
    spec_openai = detect_llm_spec(proj, None)
    assert spec_openai['backend'] == 'openai'
    assert default_strategy(None, spec_openai, proj.knobs) == 'bcomp'
    assert default_strategy('bcompv4.1', spec_openai, proj.knobs) == 'bcomp'


def test_one_command_autopilot_on_python_script_and_directory(tmp_path, monkeypatch):
    for k in ('INFRARSI_LLM', 'ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'OPENROUTER_API_KEY', 'VLLM_URL'):
        monkeypatch.delenv(k, raising=False)
    p = _copy_toy(tmp_path / 'proj_py')
    out_dir = p / 'runs'
    # Implicit `optimize` when passing train.py directly
    main([str(p / 'train.py'), '--trials', '2', '--strategy', 'random', '--out', str(out_dir)])
    assert (p / 'infrarsi.yaml').is_file()
    runs = sorted(out_dir.iterdir())
    assert len(runs) == 1
    run_dir = runs[0]
    assert (run_dir / 'REPORT.md').is_file()
    assert (run_dir / 'result.json').is_file()
    assert (run_dir / 'search' / 'trace.json').is_file()
    res = json.loads((run_dir / 'result.json').read_text(encoding='utf-8'))
    assert res['schema'] == 'infrarsi-run-v1'
    if (res.get('promotion') or {}).get('promoted'):
        assert (p / 'infrarsi_optimized.sh').is_file()

    # Explicit `optimize` when passing a directory
    p2 = _copy_toy(tmp_path / 'proj_dir')
    out2 = p2 / 'runs'
    main(['optimize', str(p2), '--screens', '2', '--strategy', 'random', '--no-doctor', '--out', str(out2)])
    runs2 = sorted(out2.iterdir())
    assert len(runs2) == 1
    assert (runs2[0] / 'REPORT.md').is_file()


def test_format_console_briefing_renders_emoji_summary():
    from infrarsi.project.report import format_console_briefing
    sample = {
        'project': {'name': 'qwen-sft', 'framework': 'megatron', 'gpus': 8, 'nodes': 1, 'world': 8, 'execution': 'local'},
        'outcome': 'improved',
        'strategy': 'bcomp',
        'tolerance': 0.03,
        'run_dir': '/tmp/runs/qwen-sft-001',
        'reference': {
            'step_time': 1.25,
            'config': {'micro_batch_size': 1, 'grad_accum': 8, 'compile': False},
            'phases': {'mean_ms': {'forward': 500.0, 'backward': 450.0, 'comm': 300.0}, 'share': {'forward': 0.4, 'backward': 0.36, 'comm': 0.24}},
        },
        'accounting': {'mfu': 0.28, 'memory_gib': 58.0, 'capacity_gib': 80},
        'promotion': {
            'promoted': 'c2',
            'speedup': 1.84,
            'config': {'micro_batch_size': 2, 'grad_accum': 4, 'compile': True},
            'overlay': {'kernels.py': 'patched'},
            'tried': [{'candidate': 'c2', 'items': [{'id': 'speed_reproduces', 'status': 'pass'}, {'id': 'eval_matches', 'status': 'pass'}]}],
        },
        'candidates': [
            {'id': 'c0', 'valid': True, 'step_time': 1.25},
            {'id': 'c1', 'valid': False, 'failed_checks': ['loss_within_tolerance'], 'change': {'compile': [False, True]}},
            {'id': 'c2', 'valid': True, 'step_time': 0.67, 'loss_deviation': 0.0018, 'change': {'micro_batch_size': [1, 2], 'grad_accum': [8, 4], 'compile': [False, True]}},
        ],
        'screens': [{'fidelity': 'full', 'valid': False}, {'fidelity': 'full', 'valid': True}],
        'search': {'stop_reason': 'budget:screens', 'used': {'screens': 2, 'trials': 1}, 'budget': {'screens': 2}, 'llm_calls': 3, 'noise': {'sigma_pair': 0.004}},
        'cost': {'reference_gpu_hours': 0.04, 'search_gpu_hours': 0.12, 'promotion_gpu_hours': 0.06, 'llm_tokens': 24500, 'llm_usd': 0.08, 'llm_usd_known': True},
    }
    text = format_console_briefing(sample, opt_script='infrarsi_optimized.sh')
    assert '🚀 InfraRSI Optimization Briefing' in text
    assert '🏆 Status          : ✨ IMPROVED & VERIFIED' in text
    assert '×1.840' in text
    assert '⚡ Performance & Acceleration Summary:' in text
    assert '🔥 Hardware MFU' in text
    assert '🧬 Winning Optimization Recipe (c2):' in text
    assert '🛡️ Fail-Closed Equivalence & Verification Gates:' in text
    assert '📈 Candidate Leaderboard:' in text
    assert '❌ c1' in text
    assert '💰 Compute & LLM Token Cost:' in text
    assert '📦 Generated Artifacts & Next Steps:' in text

