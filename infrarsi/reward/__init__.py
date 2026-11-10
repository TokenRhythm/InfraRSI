"""Guarded reward (design/reward.yaml): metric vector -> hard constraints -> checklist -> R = speed gain; breakeven."""
from .breakeven import breakeven, breakeven_from_receipts, per_update_gpuh
from .evaluate import evaluate, lenient, metric_vector
from .presets import MEMORY_CAP_GIB, exact, load_reward_cfg, resolve

__all__ = ['evaluate', 'exact', 'lenient', 'metric_vector', 'load_reward_cfg', 'resolve', 'MEMORY_CAP_GIB', 'breakeven',
           'breakeven_from_receipts', 'per_update_gpuh']
