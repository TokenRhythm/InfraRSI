"""Strategy optico: OptiCo (github.com/TangZhe96/OptiCo-public) as a baseline arm of `infrarsi run` - its four agents
(monitor, loc_error, give_sugg, change_para) and message pool from baselines.optico_style, run unchanged over the
project's knobs through ProjectHooks (the knob codec, the fixed-parameter block, the performance table's facts), on the
flow's SearchSession: the same budget as every strategy (each model request admitted as the search proposer's calls
are: agent.optimize.call_allowances), the flow's reference run as its start and first pool row (no screen spent on it),
the verifier's verdicts, the promotion's locked pair afterwards. What differs from OptiCo as
published is listed in baselines.optico_style.DEVIATIONS_PROJECT and recorded in the trace.

OptiCo-public has no license: its prompts are read at run time - from INFRARSI_OPTICO_ROOT when it is set, else from
the package's third_party_src/OptiCo-public - and never shipped. The model must answer in free text (OptiCo's
'KEY: VALUE' lines). `problem(driver)` says, before the reference run, why either is missing.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT_ENV = 'INFRARSI_OPTICO_ROOT'


def root() -> Path:
    """Where OptiCo's sources are read from: INFRARSI_OPTICO_ROOT, else baselines.optico_style.OPTICO_ROOT."""
    from ..baselines.optico_style import optico as oc
    env = os.environ.get(ROOT_ENV, '').strip()
    return Path(env) if env else oc.OPTICO_ROOT


def prompts_problem(path: Path | None = None) -> str | None:
    """Why OptiCo's prompt files and analyzer wrappers cannot be read from `path` (default root()); None when they can."""
    from ..baselines.optico_style import optico as oc
    path = root() if path is None else Path(path)
    try:
        oc.load_materials(path)
    except (OSError, ValueError, SyntaxError) as exc:
        return (f"the optico strategy reads OptiCo's prompts at run time and cannot read them from {path} "
                f'({type(exc).__name__}: {exc}): fetch github.com/TangZhe96/OptiCo-public (it has no license, so it is '
                f'never shipped) and set {ROOT_ENV} to its directory (prompt/*.txt, script/analyzer/analyzer.py)')
    return None


def backend_problem(driver) -> str | None:
    """Why `driver` cannot carry OptiCo's free-text requests; None when it can."""
    if driver is None:
        return 'the optico strategy needs an LLM backend (manifest llm.backend or --llm)'
    if getattr(driver, 'kind', None) == 'anthropic_api':
        return ("the optico strategy needs free-text replies (OptiCo's 'KEY: VALUE' lines) and the anthropic backend reads "
                'every reply as JSON: use claude-cli, or an OpenAI-compatible endpoint with llm.structured: none')
    if getattr(driver, 'structured', None) == 'json_object':
        return ('the optico strategy needs free-text replies and this OpenAI-compatible backend asks for a JSON object '
                '(llm.structured json_object, its default): set llm.structured: none')
    return None


def problem(driver) -> str | None:
    """Why the optico strategy cannot run with `driver` (the model, then OptiCo's prompts); None when it can. The CLI asks
    before the reference run; the strategy asks again when it starts."""
    return backend_problem(driver) or prompts_problem()


def run(ctx):
    """search.Context -> SearchTrace (search.STRATEGIES['optico'])."""
    from ..baselines.optico_style import optico as oc
    if not ((getattr(ctx.task, 'knob_schema', None) or {}).get('properties')):
        raise ValueError('the optico strategy changes knobs only, and this project declares none (knobs:)')
    path = root()
    why = backend_problem(ctx.driver) or prompts_problem(path)
    if why:
        raise ValueError(why)
    log = ctx.options.get('log')
    if log:
        log(f"[infrarsi] optico: OptiCo's prompts from {path}")
    session = ctx.session(ctx.options.get('arm') or 'optico', layers=('L0',))
    trace = oc.run_optico_style(ctx.task, ctx.budget, ctx.start_config, ctx.runner, llm_backend=ctx.driver, judge=ctx.judge,
                                run_id=ctx.run_id, start_receipt=ctx.start_receipt, reference_receipt=ctx.reference_receipt,
                                optico_root=path, hooks=ctx.hooks, arm=session.arm, session=session,
                                deviations=oc.DEVIATIONS_PROJECT)
    if trace.stop_reason == 'refusals' and log:
        log(f'[infrarsi] ! optico: the model refused {trace.extra.get("n_refusals")} requests despite D-OC-15 '
            "(OptiCo's think-format lines dropped); the search stopped there")
    return trace
