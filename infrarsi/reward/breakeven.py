"""N_breakeven = C_search / (C_FR_per_update - C_candidate_per_update) (plan section 5.4), same unit on both sides."""
from __future__ import annotations

import math


def per_update_gpuh(step_time_s: float, world_size: int) -> float:
    return step_time_s * world_size / 3600.0


def breakeven(search_cost: float, ref_per_update: float, cand_per_update: float,
              planned_training_updates: int | None = 1024) -> dict:
    """Costs in one unit (GPUh or USD). 'does not break even' when the candidate is not cheaper per update."""
    saving = ref_per_update - cand_per_update
    if saving <= 0 or not math.isfinite(saving):
        return {'n_breakeven': None, 'breaks_even': False, 'status': 'does not break even', 'saving_per_update': saving,
                'planned_training_updates': planned_training_updates, 'worthwhile': False}
    n = search_cost / saving
    worth = planned_training_updates is not None and n <= planned_training_updates
    return {'n_breakeven': n, 'breaks_even': True, 'status': 'breaks even', 'saving_per_update': saving,
            'planned_training_updates': planned_training_updates, 'worthwhile': worth,
            'net_saving_at_plan': (planned_training_updates * saving - search_cost) if planned_training_updates else None}


def breakeven_from_receipts(search_gpuh: float, ref_receipt: dict, cand_receipt: dict,
                            planned_training_updates: int | None = 1024) -> dict:
    def pu(r):
        return per_update_gpuh(r['step_time']['mean_window_max_rank_s'], r['world_size'])
    return dict(breakeven(search_gpuh, pu(ref_receipt), pu(cand_receipt), planned_training_updates), unit='GPUh')
