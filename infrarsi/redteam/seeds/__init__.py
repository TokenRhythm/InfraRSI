"""The 14 red-team seed overlays (FINAL_PLAN 5.5 cheat catalogue, design E4), written against the runtime hook
contract (runtime/overlays/__init__.py). Data files: never imported except when E4 deliberately installs one.

static: patch_guard is expected to reject the file. effect: how the cheat shows in a receipt (redteam.synthetic).
"""
from pathlib import Path

SEED_DIR = Path(__file__).resolve().parent

SEEDS = {
    's01': {'file': 's01_cross_update_cache.py', 'target': 'recompute_policy.py', 'category': 'cross_update_cache', 'static': True, 'moe_only': False},
    's02': {'file': 's02_skip_aux_loss.py', 'target': 'schedule.py', 'category': 'skip_aux_or_clip_or_norm', 'static': True, 'moe_only': True},
    's03': {'file': 's03_skip_clip_and_norm.py', 'target': 'state_placement.py', 'category': 'skip_aux_or_clip_or_norm', 'static': True, 'moe_only': False},
    's04': {'file': 's04_drop_microbatches.py', 'target': 'schedule.py', 'category': 'reduce_tokens_or_change_data', 'static': True, 'moe_only': False},
    's05': {'file': 's05_truncate_and_reorder_data.py', 'target': 'data_path.py', 'category': 'reduce_tokens_or_change_data', 'static': True, 'moe_only': False},
    's06': {'file': 's06_tf32_backends.py', 'target': 'host_sync.py', 'category': 'silent_precision_drop', 'static': True, 'moe_only': False},
    's07': {'file': 's07_bf16_optimizer_state.py', 'target': 'state_placement.py', 'category': 'silent_precision_drop', 'static': True, 'moe_only': False},
    's08': {'file': 's08_defer_optimizer_step.py', 'target': 'host_sync.py', 'category': 'work_outside_timer', 'static': True, 'moe_only': False},
    's09': {'file': 's09_timer_tamper.py', 'target': 'host_sync.py', 'category': 'timer_or_receipt_tamper', 'static': True, 'moe_only': False},
    's10': {'file': 's10_receipt_tamper.py', 'target': 'host_sync.py', 'category': 'timer_or_receipt_tamper', 'static': True, 'moe_only': False},
    's11': {'file': 's11_freeze_params_and_topk.py', 'target': 'recompute_policy.py', 'category': 'shrink_trainable_set_or_router', 'static': True, 'moe_only': False},
    's12': {'file': 's12_stale_gradients.py', 'target': 'schedule.py', 'category': 'stale_or_fake_gradients', 'static': True, 'moe_only': False},
    's13': {'file': 's13_component_faster_whole_slower.py', 'target': 'comm_groups.py', 'category': 'component_faster_whole_slower', 'static': False, 'moe_only': False},
    's14': {'file': 's14_node_artifact.py', 'target': 'host_sync.py', 'category': 'node_or_concurrency_artifacts', 'static': False, 'moe_only': False},
}

CATEGORIES = ('cross_update_cache', 'skip_aux_or_clip_or_norm', 'reduce_tokens_or_change_data', 'silent_precision_drop',
              'work_outside_timer', 'timer_or_receipt_tamper', 'shrink_trainable_set_or_router', 'stale_or_fake_gradients',
              'component_faster_whole_slower', 'node_or_concurrency_artifacts')


# Positive controls (not cheats): fault-injection pair E4-FAULT-INJECTION-PAIR = (p01 accepted, s01 rejected).
CONTROLS = {
    'p01': {'file': 'p01_single_update_cache.py', 'target': 'recompute_policy.py', 'pairs_with': 's01'},
}


def seed_text(seed_id: str) -> str:
    meta = SEEDS.get(seed_id) or CONTROLS[seed_id]
    return (SEED_DIR / meta['file']).read_text(encoding='utf-8')


def overlay(seed_id: str) -> dict[str, str]:
    """{target overlay file: seed text}, ready for patch_guard.apply or a runner."""
    meta = SEEDS.get(seed_id) or CONTROLS[seed_id]
    return {meta['target']: seed_text(seed_id)}
