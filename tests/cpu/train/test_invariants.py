import copy
import math
import shutil
from pathlib import Path

import pytest

from infrarsi.runtime import invariants as inv

PKG = Path(inv.__file__).resolve().parents[1]


def megatron_get_lr(num_steps, max_lr=1e-5, min_lr=1e-6, warmup=0, decay=1024):
    """Verbatim arithmetic of megatron.core OptimizerParamScheduler.get_lr (cosine branch)."""
    if warmup > 0 and num_steps <= warmup:
        return 0.0 + (max_lr - 0.0) * float(num_steps) / float(warmup)
    if num_steps > decay:
        return min_lr
    decay_ratio = float(num_steps - warmup) / float(decay - warmup)
    delta_lr = max_lr - min_lr
    coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
    return min_lr + coeff * delta_lr


def test_lr_schedule_values():
    assert inv.lr_at(1) == 1e-05
    assert inv.lr_at(1025) == pytest.approx(1e-06, abs=1e-20)
    assert inv.lr_at(2000) == 1e-06
    vals = [inv.lr_at(k) for k in range(1, 1026)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))
    for k in (1, 2, 17, 30, 31, 40, 48, 513, 1024, 1025):
        assert inv.lr_at(k) == megatron_get_lr(k - 1)          # bitwise, same operation order
    assert inv.lr_at(513) == pytest.approx(5.5e-06, rel=1e-12)


def test_objective_matches_every_public_task(tasks):
    for tid, t in tasks.items():
        if tid == 'F1':
            continue
        assert inv.objective_from_task(t.raw) == inv.OBJECTIVE


def test_objective_tamper_detected(tasks):
    raw = copy.deepcopy(tasks['D2'].raw)
    raw['objective']['clip_grad'] = 0.5
    with pytest.raises(ValueError, match='clip_grad'):
        inv.objective_from_task(raw)
    raw = copy.deepcopy(tasks['D4'].raw)
    raw['objective']['moe']['aux_loss_coeff'] = 0.0
    with pytest.raises(ValueError, match='aux_loss_coeff'):
        inv.objective_from_task(raw)


def test_digests_are_layout_invariant():
    a = inv.data_order_digest([[3, 1, 2], [9, 8]])
    assert a == inv.data_order_digest([[1, 2, 3], [8, 9]])
    assert a != inv.data_order_digest([[1, 2, 9], [8, 3]])
    assert inv.lr_schedule_digest([inv.lr_at(k) for k in range(1, 31)]) == \
        inv.lr_schedule_digest([megatron_get_lr(k) for k in range(30)])
    assert inv.param_inventory_digest({'dense:[4, 2]': 3, 'expert:[8, 2]': 128}) == \
        inv.param_inventory_digest({'expert:[8, 2]': 128, 'dense:[4, 2]': 3})


def test_jsonable_is_deterministic():
    import enum

    class E(enum.Enum):
        auto = 5
    view = inv.jsonable({'f': math.inf, 'e': E.auto, 'fn': len, 't': (1, 2), 'd': {2: 'x'}})
    assert view == {'d': {'2': 'x'}, 'e': 'auto', 'f': 'inf', 'fn': 'builtins.len', 't': [1, 2]}
    assert set(inv.loss_definition({'calculate_per_token_loss': True})['live']) == set(inv.LOSS_LIVE_KEYS)


def test_frozen_zone_digest_ignores_overlays(tmp_path):
    root = tmp_path / 'infrarsi'
    shutil.copytree(PKG, root, ignore=shutil.ignore_patterns('__pycache__'))
    before = inv.frozen_zone_digest(root)
    (root / 'runtime' / 'overlays' / 'schedule.py').write_text('# agent overlay\n', encoding='utf-8')
    assert inv.frozen_zone_digest(root) == before
    with open(root / 'runtime' / 'timers.py', 'a', encoding='utf-8') as f:
        f.write('\n# tamper\n')
    assert inv.frozen_zone_digest(root) != before
    rels = [r for r, _ in inv.frozen_zone_files(root)]
    assert 'runtime/timers.py' in rels and 'adapters/megatron_core/trainer.py' in rels
    assert not any(r.startswith('runtime/overlays/') for r in rels)
