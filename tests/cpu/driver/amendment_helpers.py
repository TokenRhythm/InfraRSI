"""AMENDMENT-002 test helpers (builder X4): the design in a known state, independent of whether the orchestrator has
already applied PROPOSED_CHANGES_X4.md to reports/.

design_world(tmp, monkeypatch, state) copies the live reward.yaml and task specs into tmp, rewrites their numerics
blocks to the 'legacy' (AMENDMENT-001) or 'amended' (AMENDMENT-002, the exact blocks of PROPOSED_CHANGES_X4.md) state,
points contracts.DESIGN_DIR and the reward loader at the copy and builds a LocalRemote with a fake packed-data INDEX
of those specs. run() / run_life() simulate an attempt (tests/gpu/driver/dummy_entry.py) and validate it like
harness.run_screen does. Other test directories load this module by path.
"""
from __future__ import annotations

import copy
import importlib.util
import re
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import yaml

try:                                                   # the module tests/cpu/driver imports (one dummy, one fake fs)
    import driver_helpers as DH
except ImportError:                                    # loaded by path from another test directory
    _spec = importlib.util.spec_from_file_location('driver_helpers_for_amendment_helpers',
                                                   Path(__file__).resolve().parent / 'driver_helpers.py')
    DH = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(DH)

from infrarsi import contracts  # noqa: E402
from infrarsi.harness import dataindex, remote_side, validate as V  # noqa: E402
from infrarsi.harness.remote import LocalRemote  # noqa: E402
from infrarsi.reward import load_reward_cfg, presets  # noqa: E402

PROPOSAL = DH.REPO / 'PROPOSED_CHANGES_X4.md'
BLOCKS = dict(re.findall(r'<!-- x4:([\w.]+) -->\s*```yaml\n(.*?)\n```', PROPOSAL.read_text(encoding='utf-8'), re.S))
TIER_TASKS = {'dense': {'S1', 'S2', 'D1', 'D2', 'H2', 'H3', 'H4', 'H6', 'X1', 'X3'},
              'moe_deterministic': {'D3', 'D4', 'D5', 'H1', 'X2'}, 'moe_nondeterministic': {'D6', 'H5', 'F1'}}
# The frozen AMENDMENT-001 values (design/AMENDMENT_001_NUMERICS_20261020.md).
A001_TOLERANCE = {'main_ce_rel_max_per_step': 0.005, 'main_ce_rel_mean_over_window': 0.002,
                  'grad_norm_rel_max_first_update': 0.01, 'grad_norm_log_bias_window_max': 0.10,
                  'grad_norm_rel_per_step': 'report_only', 'repeated_adjacent_scalars': 'invalid'}
A001_LIFECYCLE = {'state_digest_exact': True, 'first_resumed_update': {'main_ce_rel_max': 1.0e-4, 'grad_norm_rel_max': 5.0e-3},
                  'later_updates': {'main_ce': 'task_per_step_tolerance', 'grad_norm': 'report_only',
                                    'lr_tokens_clip_calls': 'exact'},
                  'forward': {'rel_l2_max': 0.05, 'top1_agreement_min': 0.90}}


def legacy_cfg() -> dict:
    cfg = copy.deepcopy(load_reward_cfg())
    cfg['tolerance'], cfg['lifecycle_tolerance'] = copy.deepcopy(A001_TOLERANCE), copy.deepcopy(A001_LIFECYCLE)
    return cfg


def amended_cfg() -> dict:
    cfg = copy.deepcopy(load_reward_cfg())
    cfg.update(yaml.safe_load(BLOCKS['reward.yaml']))
    return cfg


def spec_text(text: str, state: str) -> str:
    """A spec's text with its numerics block in `state` ('legacy' or 'amended')."""
    tier = V.numerics_tier(yaml.safe_load(text))
    old, new = BLOCKS['spec.old'], BLOCKS[f'spec.{tier}']
    src, dst = (new, old) if state == 'legacy' else (old, new)
    if src in text:
        return text.replace(src, dst)
    assert dst in text, f'numerics block of {tier} spec matches neither state'
    return text


def reward_text(text: str, state: str) -> str:
    lines = text.splitlines(keepends=True)
    a = next(i for i, ln in enumerate(lines) if ln.startswith('tolerance:'))
    b = next(i for i, ln in enumerate(lines) if ln.startswith('metrics:'))
    if state == 'amended':
        block = BLOCKS['reward.yaml'] + '\n'
    else:
        block = yaml.safe_dump({'tolerance': A001_TOLERANCE, 'lifecycle_tolerance': A001_LIFECYCLE}, sort_keys=False)
    return ''.join(lines[:a]) + block + ''.join(lines[b:])


def spec_raw(task_id: str, state: str) -> dict:
    return yaml.safe_load(spec_text((contracts.DESIGN_DIR / 'task_specs' / f'{task_id}.yaml').read_text(encoding='utf-8'), state))


def design_world(tmp_path: Path, monkeypatch, state: str = 'amended') -> SimpleNamespace:
    d = Path(tmp_path) / f'design_{state}'
    (d / 'task_specs').mkdir(parents=True)
    live = contracts.DESIGN_DIR
    for p in sorted((live / 'task_specs').glob('*.yaml')):
        (d / 'task_specs' / p.name).write_text(spec_text(p.read_text(encoding='utf-8'), state), encoding='utf-8')
    (d / 'reward.yaml').write_text(reward_text((live / 'reward.yaml').read_text(encoding='utf-8'), state), encoding='utf-8')
    monkeypatch.setattr(contracts, 'DESIGN_DIR', d)
    monkeypatch.setattr(presets, 'REWARD_YAML', d / 'reward.yaml')
    root = Path(tempfile.mkdtemp(prefix='isx'))          # short root (Windows MAX_PATH)
    remote = LocalRemote(root)
    facts = DH.fake_index(remote, [p.stem for p in (d / 'task_specs').glob('*.yaml') if p.stem != 'F1'])
    return SimpleNamespace(state=state, dir=d, remote=remote, facts=facts, root=root, tmp=Path(tmp_path),
                           cfg=presets.load_reward_cfg(d / 'reward.yaml'),
                           close=lambda: shutil.rmtree(root, ignore_errors=True))


def _prepare(world, req):
    data = dataindex.load(world.remote, ['WINH0'], req.task, req.input_plan)
    req.data_manifest = data['manifest']
    DH.pin_overlay(req)
    return data, world.facts[req.task['id']][req.input_plan]


def run(world, req, reference=None, mutate=None, **sim) -> dict:
    """Validated receipt of a simulated attempt of `req` (mutate(receipt) edits receipt.json before validation)."""
    data, facts = _prepare(world, req)
    out = world.tmp / req.attempt_id
    DH.DUMMY_MOD.simulate_attempt(req.to_dict(), str(out), data=facts, **sim)
    if mutate:
        DH.rewrite_receipt(out, mutate)
    return V.validate(remote_side.collect_run(str(out), len(req.nodes)), req, reference, data=data)


def run_life(world, req, reference=None, flags=None, resume_fn=None, mutate=None, **sim) -> dict:
    """Validated receipt of a simulated lifecycle attempt (dummy_entry.simulate_lifecycle; flags replace its defaults)."""
    data, facts = _prepare(world, req)
    out = world.tmp / req.attempt_id
    DH.DUMMY_MOD.simulate_lifecycle(req.to_dict(), str(out), flags=flags, resume_fn=resume_fn, data=facts, **sim)
    if mutate:
        DH.rewrite_receipt(out, mutate)
    return V.validate(remote_side.collect_run(str(out), len(req.nodes)), req, reference, data=data)


def reasons(rec: dict, prefix: str) -> list[str]:
    return [x for x in rec['invalid_reasons'] if x.startswith(f'harness: {prefix}')]
