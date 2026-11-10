"""Deviation ids of the F1 fallback (AMENDMENT_EXT_003 Q4; DEVIATION_IDS.md is the ledger, O's file).

Every F1 receipt lists all that apply: D-F1-1..6 for every screen (the runner records them), D-OC-15 in addition on
F-OPTICO, D-F1-7 in addition on F-A0-CODE (setting 2, code edits). No id for F-RND, F-HT or the margin estimator
(protocol rules, as AMENDMENT_EXT_002).
"""
from __future__ import annotations

DEVIATIONS = {
    'D-F1-1': 'measurement side = the formal-54 production path (launch_lp_node.py safety logic -> portable shell -> '
              'flash_import_snapshot_{parent,worker} -> run_precision_screen.py -> train_flash_strategy_20260925.py, sha '
              '4ab14ced) wrapped as ext/flashnext; outputs, locks and cache under /data/infrarsi/ext; node map '
              'WINH1/3/4/6; per-screen cap 1,800 s instead of the path\'s 7,200 s; metric = the collect_family formula '
              're-derived from signals',
    'D-F1-2': 'action space = switch_space.json (S_F1, implementation switches on the fixed production layout) instead '
              'of the L0 layout grid; lossy moment policies excluded; F-PROMPTS feasibility precheck -> the S_F1 '
              'validator; F-OPTICO codec over the switches',
    'D-F1-3': 'locked pairs on the frozen plan-A data order; no plan B',
    'D-F1-4': 'no InfraRSI runtime invariants, dtype census or inter-step consistency; replaced by the path\'s own '
              'asserts, frozen hashes, knob binding and the numerics compare incl. MTP CE',
    'D-F1-5': 'F-A0 analysis tools and profile_ranges closed (tools_enabled=False, as A-ISW / B-IS / C-IS); receipt '
              'view = per-step max-rank times, per-rank spread, numerics deltas and class, and every values.* scalar '
              'family present in signals, summarized identically for every arm',
    'D-F1-6': 'promotion lifecycle not run (F1.yaml lifecycle.required_for_promotion: true); every F1 result is '
              'screen-level; no checkpoint / resume / export',
    'D-OC-15': 'strip_think_format (decision D7): the lines of OptiCo\'s prompts that demand <think>-tagged reasoning '
               'are dropped (F-OPTICO; B-NAT shim rule); prompts, parsers and loop otherwise unchanged',
    'D-F1-7': 'setting 2 (F-A0-CODE only): the action space is S_F1 plus code edits of the 13 speed-path runtime modules '
              'of C_F1 (code_space.py; complete new file versions, <= 300 changed lines, integrity rules against each '
              'base version); the overlay runs from the attempt\'s own bundle code/ copy on /data; frozen-file and '
              'overlay hashes are checked by the worker, every rank echo and guards.frozen_hashes; to be entered in '
              'DEVIATION_IDS.md',
}

COMMON = ('D-F1-1', 'D-F1-2', 'D-F1-3', 'D-F1-4', 'D-F1-5', 'D-F1-6')
# F-OPTICO-20 is the open wave's OptiCo arm (D-OC-15 as F-OPTICO); the open arms' D-F1-7 is added per-run when C_F1 is
# open (arms.run_f_is2 / arms.run_locked with code=True), not statically, since an open arm can also run switches only.
ARM_EXTRA = {'F-OPTICO': ('D-OC-15',), 'F-OPTICO-20': ('D-OC-15',), 'F-A0-CODE': ('D-F1-7',)}
SCREEN_LEVEL_LABEL = 'screen-level; no checkpoint / resume / export'          # D-F1-6, printed next to every result


def for_arm(arm: str | None) -> list[str]:
    return list(COMMON) + list(ARM_EXTRA.get(arm or '', ()))


def describe(ids) -> dict:
    return {i: DEVIATIONS[i] for i in ids}
