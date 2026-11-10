# Proposed changes from builder X6 (ext/flashnext judge / guards / arms), 2026-10-21

X6 did not edit any of these files. Each item names the owner, the reason (AMENDMENT_EXT_003 section) and the
smallest change.

## 1 `runner.py` (X5): the locked pair is reserved inside the arm's search budget (Q5)

`FlashNextRunner.pair_wall_cap` returns `min(cap_s, max(1.5 x expected, expected + 300))`, so `SearchSession._admission`
reserves two locked-pair members (wall and GPU-hours) inside every F1 arm's budget. In addition
`worst_case_gpu_hours(world, wall_cap)` reads `wall_cap or self.cap_s`, so even a pair cap of 0 would reserve 2 x 32 x
(1,800 + 180) s. AMENDMENT_EXT_003 Q5 runs the locked pair outside the search budget ("each arm's locked pair right after
its search"; `gpu_hours` = 12 x 16 "so that worst-case admission never pre-empts screens"), and the ext base class says
the same (`EnvRunner.pair_wall_cap` returns 0: "the locked re-score is not part of an ext arm's budget").
With 10-15 min screens the reserve (about 27 GPUh and 3,060 s) does not bind; near the caps it removes one or two of
the 12 screens. Proposed:

```python
    def pair_wall_cap(self, expected_s: float) -> int:
        return 0                                   # Q5: the locked pair runs outside the arm's budget

    def worst_case_gpu_hours(self, world: int, wall_cap: float | None = None) -> float:
        cap = self.cap_s if wall_cap is None else float(wall_cap)
        return int(world) * (cap + (self.stop_grace_s if cap else 0.0)) / 3600.0
```

## 2 `frozen_manifest.json` (X5): regenerate after every launcher edit (stale twice; clean at 09:40)

At 08:29 `node/fn_launch_node.py` changed after `frozen_manifest.json` was written (08:15); X6's `frozen_hashes` guard
(Q3: "plus the variant layer's own frozen_manifest.json") rejected every screen with `node/fn_launch_node.py differs
from the frozen manifest` until X5 regenerated the manifest at 08:40. It went stale again before 09:23 (Codex round 2,
critical: launcher sha256 `e607d00f...` vs manifest `4fe91564...`; the full CPU suite failed X5's
`test_frozen_manifest_matches_the_variant_layer` at the same time). X5 regenerated it at 09:40 after the 09:39 launcher
edit; `guards.local_layer_problems(load_manifest())` is `[]` since. Any later edit of the variant layer needs
`python -m infrarsi.ext.flashnext.deploy manifest` and a redeploy before preflight (G-F1 (4)); the guard enforces it,
and X5's test above catches a stale manifest on CPU.

## 3 `tests/cpu/ext/flashnext/fn_helpers.py` (X5): restored after an accidental overwrite

At 08:14 X6 wrote its own helper module under the same name and overwrote X5's (created 08:10, unknown to X6 at that
moment). X6 restored it at 08:20 from X5's own bytecode (`__pycache__/fn_helpers.cpython-312.pyc` of 08:13, source mtime
08:10:35, 7,741 bytes): the recompiled code objects are identical (instructions, constants, names, line and exception
tables), so behaviour is exactly X5's; comments, if there were any, could not be recovered. X6's helpers now live in
`x6_helpers.py`. X5 has since edited the file again (08:35). Please check that nothing is missing.

## 4 Ledgers and interface docs (O)

- `EXT_INTERFACES.md` §1: add `infrarsi/ext/flashnext/` (X5: knobs, variant, switch_space(.json), runner, deploy,
  preflight, frozen_manifest.json, node/; X6: parse, judge, guards, numerics_gates.json, task, hooks, arms, cli,
  deviations) and the baseline seams X6 added (below).
- `ext/common/env_task.py:23` `ENVS` (X1): add `'flashnext'` (cosmetic; `check_spec` accepts any identifier).
- Baseline seams added by X6 (backward compatible; with `hooks=None` the old code path runs unchanged and the existing
  tests stay green): `baselines/common.py:new_session(..., hooks=None)`; `baselines/prompts_oneshot.py:run_prompts(...,
  hooks=None)` -> `run_prompts_ext` (the environment's validator replaces the L0 validator and the planner feasibility
  precheck; the start check is `hooks.start_receipt_problems`); `baselines/optico_style/optico.py:run_optico_style(...,
  hooks=None, arm='optico_style_L0')` (codec `hooks.param_codec()`, FIXED_PARA `hooks.fixed_parameters_text()`,
  `DEVIATIONS_HOOKS`). Owners X4 (F-PROMPTS) and A/X3 (optico_style) should know.
- `DEVIATION_IDS.md`: no change (D-F1-1..6 and D-OC-15 as ruled; `ext/flashnext/deviations.py` mirrors them).
- F1 run directories (Codex round 3): each arm's `trace.json` and every `locked*.json` carry a `harness_signature`
  (receipt key) added by `arms._seal`; the formal summary refuses unsealed or edited files. Readers of `trace.json` see
  one extra key. A window is the set of run directories given to `summary`; the ledger should fix one run id per arm.
