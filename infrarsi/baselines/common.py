"""Shared scaffolding for baseline arms: same SearchSession (budget, validation, screening, trace) as the agent.

Config track (design E2): same L0 space, same ED start, same screen/wall/GPU budgets. Invalid L0 samples are
rejected by validate_l0 before any screen (as for the agent) and counted in invalid_proposals.
"""
from __future__ import annotations

import json

from ..actions import config_from_point, l0_space
from ..agent.search import SearchSession

NONE = '__none__'          # categorical stand-in for None (lists are JSON strings)


def encode_value(v):
    if v is None:
        return NONE
    if isinstance(v, list):
        return json.dumps(v)
    return v


def decode_value(v):
    if v == NONE:
        return None
    if isinstance(v, str) and v.startswith('['):
        return json.loads(v)
    return v


def space_for(task, model_facts) -> dict[str, list]:
    """L0 grid plus '__layout__': the task's valid (TP, PP, CP, EP, ETP, MBS) tuples as one categorical."""
    from .space import layout_choices
    space = l0_space(task, model_facts, include_env=True)
    if task is not None:
        space['__layout__'] = layout_choices(task, model_facts, space)
    return space


def point_to_config(point: dict) -> dict:
    return config_from_point({k: decode_value(v) for k, v in point.items()})


def config_to_point(config: dict, space: dict) -> dict:
    """Project a (start) config onto the sampler space; values outside the grid fall back to the first choice."""
    from ..actions import validate_l0
    resolved = validate_l0(config).resolved
    point = {}
    for k, choices in space.items():
        if k.startswith('env:'):
            v = (resolved.get('env') or {}).get(k[4:])
            v = None if v is None else (int(v) if str(v).isdigit() else v)
        else:
            v = resolved.get(k)
        enc = [encode_value(c) for c in choices]
        point[k] = encode_value(v) if encode_value(v) in enc else enc[0]
    return point


def new_session(*, task, budget, runner, judge, run_id, arm, out_dir, start_config, start_receipt, reference_receipt,
                model_facts, clock, layers=('L0',), hooks=None) -> SearchSession:
    """hooks (ext.common.session_hooks.SessionHooks): an external environment's validator / digest seam, passed through
    to SearchSession (EXT_INTERFACES §3); None (every core task) changes nothing."""
    return SearchSession(task=task, budget=budget, runner=runner, judge=judge, run_id=run_id, arm=arm, out_dir=out_dir,
                         start_config=start_config, start_receipt=start_receipt, reference_receipt=reference_receipt,
                         model_facts=model_facts, mode='strong', layers=layers, clock=clock,
                         driver_info={'kind': arm, 'llm': None}, hooks=hooks)


def screen_start(session: SearchSession):
    if session.candidates['c0'].status == 'proposed':
        session.screen('c0')
