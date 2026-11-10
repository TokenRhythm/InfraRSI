"""ENV-A adapter: RE-Bench `ai_rd_optimize_llm_foundry` (design §2, EXT_INTERFACES §1 owner X2).

Driver-side modules (never import torch): `task` (EnvTask + task view), `knobs` (§2.4 space + templater),
`patch_guard` (validate_patch helpers), `guards` (G1-G7 over node measurements), `runner` (RebenchLlmfRunner,
subclasses common.env_runner.EnvRunner), `tools` (A-NAT score() EnvTool), `hooks` (EnvAHooks subclasses
common.session_hooks.SessionHooks). GPU / torch work lives in `node/score_runner.py`, run on the leased node via the
shared `common.remote_exec.NodeOps`, and is never imported here.
"""
from .guards import evaluate as evaluate_guards
from .hooks import EnvAHooks
from .knobs import KNOBS, knob_schema, render, sample, sample_order, validate
from .patch_guard import validate_patch
from .runner import EnvARunner, RebenchLlmfRunner, SCHEMA_RECEIPT
from .task import arm_deviations, build_env_a_task, frozen_file_hashes, read_base_solution, task_view
from .tools import ScoreState, locked_rescore, make_tools

__all__ = ['RebenchLlmfRunner', 'EnvARunner', 'SCHEMA_RECEIPT', 'EnvAHooks', 'KNOBS', 'knob_schema', 'render',
           'sample', 'sample_order', 'validate', 'validate_patch', 'evaluate_guards', 'build_env_a_task',
           'arm_deviations', 'frozen_file_hashes', 'read_base_solution', 'task_view', 'make_tools', 'locked_rescore',
           'ScoreState']
