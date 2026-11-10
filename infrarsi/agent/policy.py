"""Optimizer policy (policy.yaml in an OptimizerPackage): context sizes, verifier/summarizer, weak-model gates.

Reflection may edit policy.yaml, but never beyond these caps (the weak-model protocol and the search budget
are frozen; rsi.selfcheck enforces POLICY_CAPS).
"""
from __future__ import annotations

import copy
import dataclasses

import yaml

DEFAULT_POLICY = {
    'schema': 'infrarsi-policy-v1',
    'modes': {
        'strong': {'max_context_tokens': 48000, 'keep_last_turns': 10, 'max_turns': 120, 'verifier': True,
                   'summarizer': 'program', 'retrieve_top_k': 2, 'skill_tokens': 4000, 'receipt_digest_tokens': 1500,
                   'layers': ['L0', 'L1'], 'patch_line_limit': 300, 'max_retries': 0, 'memory_hits': 3,
                   'config_first_min_screens': 0, 'l0_stall_screens': 0, 'templates_before_free_patch': False},
        'weak': {'max_context_tokens': 32000, 'keep_last_turns': 6, 'max_turns': 120, 'verifier': True,
                 'summarizer': 'program', 'retrieve_top_k': 2, 'skill_tokens': 4000, 'receipt_digest_tokens': 1500,
                 'layers': ['L0', 'L1'], 'patch_line_limit': 80, 'max_retries': 3, 'memory_hits': 2,
                 'config_first_min_screens': 6, 'l0_stall_screens': 3, 'templates_before_free_patch': True},
    },
    'search': {'screen_start_if_missing': True, 'stop_after_same_failures': 6, 'max_model_errors_in_row': 6},
}

POLICY_CAPS = {   # key -> (min, max) per mode; reflection cannot exceed the protocol
    'strong': {'max_context_tokens': (4000, 64000), 'keep_last_turns': (1, 40), 'max_turns': (1, 200), 'retrieve_top_k': (0, 2),
               'skill_tokens': (0, 4000), 'receipt_digest_tokens': (200, 1500), 'patch_line_limit': (1, 300),
               'max_retries': (0, 3), 'memory_hits': (0, 5), 'config_first_min_screens': (0, 16), 'l0_stall_screens': (0, 16)},
    'weak': {'max_context_tokens': (4000, 32000), 'keep_last_turns': (1, 20), 'max_turns': (1, 200), 'retrieve_top_k': (0, 2),
             'skill_tokens': (0, 4000), 'receipt_digest_tokens': (200, 1500), 'patch_line_limit': (1, 80),
             'max_retries': (0, 3), 'memory_hits': (0, 5), 'config_first_min_screens': (0, 16), 'l0_stall_screens': (0, 16)},
}
ALLOWED_LAYERS = ('L0', 'L1')          # L2 is closed on the public tasks (weak: always closed)


@dataclasses.dataclass
class ModePolicy:
    mode: str
    max_context_tokens: int
    keep_last_turns: int
    max_turns: int
    verifier: bool
    summarizer: str
    retrieve_top_k: int
    skill_tokens: int
    receipt_digest_tokens: int
    layers: list
    patch_line_limit: int
    max_retries: int
    memory_hits: int
    config_first_min_screens: int
    l0_stall_screens: int
    templates_before_free_patch: bool
    search: dict


def check_policy(doc: dict) -> list[str]:
    errors = []
    if not isinstance(doc, dict) or doc.get('schema') != 'infrarsi-policy-v1':
        return ['policy.yaml: schema must be infrarsi-policy-v1']
    unknown = set(doc) - {'schema', 'modes', 'search', 'notes'}
    if unknown:
        errors.append(f'policy.yaml: unknown top-level keys {sorted(unknown)} (budgets, reward and tolerances are frozen)')
    modes = doc.get('modes') or {}
    for mode in ('strong', 'weak'):
        m = modes.get(mode)
        if not isinstance(m, dict):
            errors.append(f'policy.yaml: modes.{mode} missing')
            continue
        fields = {f.name for f in dataclasses.fields(ModePolicy)} - {'mode', 'search'}
        missing, extra = fields - set(m), set(m) - fields
        if missing:
            errors.append(f'modes.{mode}: missing {sorted(missing)}')
        if extra:
            errors.append(f'modes.{mode}: unknown keys {sorted(extra)}')
        for k, (lo, hi) in POLICY_CAPS[mode].items():
            v = m.get(k)
            if type(v) is not int or not lo <= v <= hi:
                errors.append(f'modes.{mode}.{k}={v!r} outside [{lo}, {hi}]')
        if type(m.get('verifier')) is not bool or type(m.get('templates_before_free_patch')) is not bool:
            errors.append(f'modes.{mode}: verifier/templates_before_free_patch must be bool')
        if m.get('summarizer') not in ('program', 'llm'):
            errors.append(f'modes.{mode}.summarizer must be program|llm')
        layers = m.get('layers')
        if not isinstance(layers, list) or not layers or any(x not in ALLOWED_LAYERS for x in layers):
            errors.append(f'modes.{mode}.layers must be a non-empty subset of {ALLOWED_LAYERS}')
    if isinstance(modes.get('weak'), dict) and modes['weak'].get('templates_before_free_patch') is not True:
        errors.append('modes.weak.templates_before_free_patch must stay true (weak-model protocol)')
    s = doc.get('search') or {}
    if type(s.get('stop_after_same_failures')) is not int or not 1 <= s['stop_after_same_failures'] <= 6:
        errors.append('search.stop_after_same_failures must be an int in [1, 6]')
    return errors


def load(text_or_dict, mode: str, **overrides) -> ModePolicy:
    doc = yaml.safe_load(text_or_dict) if isinstance(text_or_dict, str) else copy.deepcopy(text_or_dict)
    errors = check_policy(doc)
    if errors:
        raise ValueError('; '.join(errors))
    m = dict(doc['modes'][mode])
    m.update({k: v for k, v in overrides.items() if v is not None})
    return ModePolicy(mode=mode, search=doc.get('search') or {}, **m)


def default_yaml() -> str:
    return yaml.safe_dump(DEFAULT_POLICY, sort_keys=False, allow_unicode=True)
