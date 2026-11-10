"""The driver-capability knob (revamp design D): one number c in [0, 1] instead of the strong / weak split.

c = 1 trusts a frontier model with free-form hypotheses, large patches and little scaffolding; c = 0 treats the model
as a weak advisor whose output is constrained to menus, whose predictions are discounted until measurements confirm
them, and whose feasibility claims are checked by probes. Everything in between is interpolated, so a user can turn
the knob (manifest `llm.capability`, `--capability`) and the system degrades or relaxes continuously. When no value
is given it is inferred from the model name.

    infer(model) -> c
    search_params(c) -> {'kappa0', 'sigma_scale', 'probe_below', 'max_hypotheses', 'menu'}       (Engine strategies)
    policy_overrides(c, doc) -> {field: value} for agent.policy.load (the one-action-per-turn agent loop)
"""
from __future__ import annotations

import re

KNOWN = ((r'fable|mythos|gpt-6|o5|gemini-3.*pro', 0.95), (r'opus', 0.9), (r'gpt-5', 0.85), (r'sonnet|gemini.*flash', 0.7),
         (r'deepseek|qwen3\.?[5-9]|kimi|glm-5', 0.5), (r'haiku|mini|nano|flash-lite', 0.35), (r'7b|8b|9b|14b', 0.2))


def infer(model: str | None) -> float:
    m = (model or '').lower()
    for rx, c in KNOWN:
        if re.search(rx, m):
            return c
    return 0.6


def clamp(c) -> float:
    return min(1.0, max(0.0, float(c)))


def lerp(weak, strong, c: float):
    v = weak + (strong - weak) * clamp(c)
    return int(round(v)) if isinstance(weak, int) and isinstance(strong, int) else v


def search_params(c: float) -> dict:
    """Engine-strategy scaffolding: prior trust in the model's gain predictions (kappa0, before calibration), how much
    its uncertainty is inflated, below which claimed feasibility a probe runs first, how many hypotheses one call
    returns, and whether the model rates a menu of enumerated moves instead of inventing changes."""
    c = clamp(c)
    return {'kappa0': lerp(0.25, 0.7, c), 'sigma_scale': lerp(2.0, 1.0, c), 'probe_below': lerp(0.95, 0.75, c),
            'max_hypotheses': lerp(4, 12, c), 'menu': c < 0.4}


# one-action-per-turn agent loop (agent.policy): the old weak (c=0) and strong (c=1) protocol values are the endpoints
_LOOP = {'max_context_tokens': (32000, 48000), 'keep_last_turns': (6, 10), 'patch_line_limit': (80, 300), 'max_retries': (3, 0),
         'memory_hits': (2, 3), 'config_first_min_screens': (6, 0), 'l0_stall_screens': (3, 0)}


def policy_overrides(c: float) -> dict:
    c = clamp(c)
    out = {k: lerp(w, s, c) for k, (w, s) in _LOOP.items()}
    out['templates_before_free_patch'] = c < 0.5
    return out


def mode_for(c: float) -> str:
    """The discrete part that remains: the action menu / schema form of the agent loop."""
    return 'weak' if clamp(c) < 0.5 else 'strong'
