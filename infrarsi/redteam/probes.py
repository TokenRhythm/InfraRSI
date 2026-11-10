"""Static red-team probes: small overlays that reach an effect the guard must reject before any screen (regression
inputs from the Codex reviews; data, never imported by the training process). `run_redteam.cpu_run` requires every
probe to be rejected by `actions.patch_guard`.
"""
from __future__ import annotations

PROBES = {
    # driver-side review R1, finding 5(a): computed-key and mutator writes into optimizer / runtime containers
    'opt_params_computed_key': ('state_placement.py', 'def after_optimizer_build(ctx, optimizer):\n'
                                '    for group in optimizer.param_groups:\n        key = "pa" + "rams"\n        group[key] = []\n'),
    'opt_group_update': ('state_placement.py', 'def after_optimizer_build(ctx, optimizer):\n'
                         '    for g in optimizer.param_groups:\n        g.update({"params": []})\n'),
    'opt_state_reset': ('state_placement.py', 'def after_optimizer_build(ctx, optimizer):\n'
                        '    for p in list(optimizer.state):\n        optimizer.state[p] = {}\n'),
    'plan_computed_key': ('recompute_policy.py', 'def model_config_overrides(ctx):\n    k = "num" + "_layers"\n'
                          '    ctx.plan[k] = 1\n    return {}\n'),
    # finding 5(b): filesystem mutation of the shared /data tree
    'path_replace': ('host_sync.py', 'from pathlib import Path\n\n\ndef on_train_end(ctx):\n'
                     '    Path("/data/infrarsi/code/x").replace("/data/infrarsi/code/y")\n'),
    'os_renames': ('host_sync.py', 'import os\n\n\ndef on_train_end(ctx):\n    os.renames("/data/infrarsi/a", "/data/infrarsi/b")\n'),
    'os_rename_alias': ('host_sync.py', 'import os\n\n\ndef on_train_end(ctx):\n    mv = os.rename\n    mv("a", "b")\n'),
    'os_chmod': ('host_sync.py', 'import os\n\n\ndef on_train_end(ctx):\n    os.chmod("/data/infrarsi/a", 0o777)\n'),
    'path_symlink': ('host_sync.py', 'from pathlib import Path\n\n\ndef on_train_end(ctx):\n    Path("a").symlink_to("/data/infrarsi/b")\n'),
    'shutil_rmtree': ('host_sync.py', 'import shutil\n\n\ndef on_train_end(ctx):\n    shutil.rmtree("/data/infrarsi/runs")\n'),
}


def static_probe_report() -> dict:
    from ..actions import patch_guard
    rows = {}
    for name, (target, text) in PROBES.items():
        rep = patch_guard.apply({target: text}, mode='strong')
        rows[name] = {'target': target, 'rejected': not rep.accepted, 'codes': sorted(rep.codes())}
    return {'total': len(rows), 'rejected': sum(r['rejected'] for r in rows.values()),
            'accepted': sorted(n for n, r in rows.items() if not r['rejected']), 'rows': rows}
