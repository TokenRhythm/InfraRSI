"""Agent tools as JSON actions (one action per model turn) and the role output schemas.

Strong mode tools (FINAL_PLAN 3.3): read_env, read_receipt, profile_ranges, propose_config, propose_patch,
run_screen, retrieve_skill, write_memory, finish. Weak mode (FINAL_PLAN 7.2) exposes the explicit menu
inspect / propose_config / apply_patch_template / propose_patch / run_trial / write_memory / finish and is
mapped onto the same tools.
"""
from __future__ import annotations

_ID = {'type': 'string', 'minLength': 1, 'maxLength': 64}
_TEXT = {'type': 'string', 'maxLength': 2000}


def _act(name: str, props: dict | None = None, required: tuple = ()) -> dict:
    return {'type': 'object', 'properties': {'action': {'const': name}, **(props or {})},
            'required': ['action', *required], 'additionalProperties': False}


MEMORY_ENTRY = {'type': 'object', 'properties': {
    'tags': {'type': 'object'}, 'slow': {'type': ['object', 'null']}, 'fast': {'type': ['object', 'null']},
    'outcome': {'type': 'string', 'maxLength': 400}, 'failure_reason': {'type': ['string', 'null'], 'maxLength': 400},
    'lesson': {'type': 'string', 'maxLength': 1200}}, 'required': ['lesson'], 'additionalProperties': False}

STRONG_ACTIONS = {
    'read_env': _act('read_env'),
    'read_receipt': _act('read_receipt', {'candidate_id': _ID, 'fields': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 12}}, ('candidate_id',)),
    'profile_ranges': _act('profile_ranges', {'candidate_id': _ID}, ('candidate_id',)),
    'retrieve_skill': _act('retrieve_skill', {'query': {'type': 'string', 'minLength': 1, 'maxLength': 500}, 'skill': {'type': 'string', 'maxLength': 80}}, ('query',)),
    'propose_config': _act('propose_config', {'base': _ID, 'config': {'type': 'object'}, 'changes': {'type': 'object'}, 'rationale': _TEXT}),
    'propose_patch': _act('propose_patch', {'base': _ID, 'files': {'type': 'object', 'additionalProperties': {'type': 'string'}},
                                            'template': {'type': 'string'}, 'params': {'type': 'object'}, 'rationale': _TEXT}),
    'run_screen': _act('run_screen', {'candidate_id': _ID}, ('candidate_id',)),
    'write_memory': _act('write_memory', {'entry': MEMORY_ENTRY}, ('entry',)),
    'finish': _act('finish', {'candidate_id': _ID, 'reason': _TEXT}),
}

WEAK_ACTIONS = {
    'inspect': _act('inspect', {'what': {'enum': ['env', 'receipt', 'profile', 'skill']}, 'candidate_id': _ID,
                                'query': {'type': 'string', 'maxLength': 300}}, ('what',)),
    'propose_config': _act('propose_config', {'base': _ID, 'changes': {'type': 'object', 'maxProperties': 6}, 'rationale': {'type': 'string', 'maxLength': 600}}, ('base', 'changes')),
    'apply_patch_template': _act('apply_patch_template', {'base': _ID, 'template': {'type': 'string'}, 'params': {'type': 'object'}}, ('template',)),
    'propose_patch': _act('propose_patch', {'base': _ID, 'file': {'type': 'string'}, 'content': {'type': 'string', 'maxLength': 20000}, 'rationale': {'type': 'string', 'maxLength': 600}}, ('file', 'content')),
    'run_trial': _act('run_trial', {'candidate_id': _ID}, ('candidate_id',)),
    'write_memory': _act('write_memory', {'entry': MEMORY_ENTRY}, ('entry',)),
    'finish': _act('finish', {'candidate_id': _ID, 'reason': {'type': 'string', 'maxLength': 600}}),
}


def action_schema(mode: str, allowed: tuple[str, ...] | None = None) -> dict:
    table = WEAK_ACTIONS if mode == 'weak' else STRONG_ACTIONS
    names = [n for n in table if allowed is None or n in allowed]
    return {'oneOf': [table[n] for n in names]}


VERIFIER_SCHEMA = {'type': 'object', 'properties': {
    'verdict': {'enum': ['pass', 'veto']}, 'reasons': {'type': 'array', 'items': {'type': 'string', 'maxLength': 400}, 'maxItems': 8},
    'checklist': {'type': 'object'}}, 'required': ['verdict', 'reasons'], 'additionalProperties': False}

SUMMARIZER_SCHEMA = {'type': 'object', 'properties': {'summary': {'type': 'string', 'maxLength': 6000}},
                     'required': ['summary'], 'additionalProperties': False}

REFLECT_SCHEMA = {'type': 'object', 'properties': {
    'skills': {'type': 'object', 'additionalProperties': {'type': 'string'}},
    'policy': {'type': ['string', 'null']},
    'prompts': {'type': 'object', 'additionalProperties': {'type': 'string'}},
    'tools': {'type': 'object', 'additionalProperties': {'type': ['string', 'null']}},
    'memory': {'type': 'array', 'items': MEMORY_ENTRY, 'maxItems': 16},
    'rationale': {'type': 'string', 'maxLength': 4000}}, 'required': ['rationale'], 'additionalProperties': False}

STRONG_MENU = '''Return exactly one JSON object per turn (no markdown). Actions:
{"action":"read_env"}  hardware profile, task and model facts
{"action":"read_receipt","candidate_id":"c3","fields":["memory","events"]}  digest of a screened candidate's receipt
{"action":"profile_ranges","candidate_id":"c3"}  per-phase timing ranges across ranks/steps
{"action":"retrieve_skill","query":"why is the MoE dispatch slow","skill":"moe-ep-communication"}  top-2 skill sections
{"action":"propose_config","base":"c0","changes":{"<L0 key>":value,"env":{"NAME":"value"}},"rationale":"..."}  (or "config": a complete L0 config)
{"action":"propose_patch","base":"c2","files":{"host_sync.py":"<complete new file text>"},"rationale":"..."}  L1 overlay (or "template":"<id>","params":{...})
{"action":"run_screen","candidate_id":"c4"}  one screen = one fresh training run (costs budget)
{"action":"write_memory","entry":{"tags":{...},"slow":{...},"fast":{...},"outcome":"...","failure_reason":null,"lesson":"..."}}
{"action":"finish","candidate_id":"c4","reason":"..."}  stop; the best valid screened candidate is promoted'''

WEAK_MENU = '''Return exactly ONE JSON object, nothing else. Menu (pick one):
1 {"action":"inspect","what":"env|receipt|profile|skill","candidate_id":"c1","query":"..."}
2 {"action":"propose_config","base":"c0","changes":{"<L0 key>":value}}   (at most 6 keys)
3 {"action":"apply_patch_template","base":"c2","template":"<template id>","params":{...}}   (only after configs are exhausted)
4 {"action":"propose_patch","base":"c2","file":"<overlay>.py","content":"<complete file>"}   (only after a template; <= 80 changed lines)
5 {"action":"run_trial","candidate_id":"c4"}
6 {"action":"write_memory","entry":{"lesson":"..."}}
7 {"action":"finish","candidate_id":"c4","reason":"..."}'''


def normalize(action: dict, mode: str) -> dict:
    """Map a weak-mode action onto the strong tool vocabulary."""
    a = dict(action)
    kind = a.get('action')
    if mode != 'weak':
        return a
    if kind == 'inspect':
        what = a.get('what')
        if what == 'env':
            return {'action': 'read_env'}
        if what == 'receipt':
            return {'action': 'read_receipt', 'candidate_id': a.get('candidate_id', '')}
        if what == 'profile':
            return {'action': 'profile_ranges', 'candidate_id': a.get('candidate_id', '')}
        return {'action': 'retrieve_skill', 'query': a.get('query') or 'general'}
    if kind == 'apply_patch_template':
        return {'action': 'propose_patch', 'base': a.get('base'), 'template': a['template'], 'params': a.get('params') or {}}
    if kind == 'propose_patch':
        return {'action': 'propose_patch', 'base': a.get('base'), 'files': {a['file']: a['content']}, 'rationale': a.get('rationale', ''),
                'free_patch': True}
    if kind == 'run_trial':
        return {'action': 'run_screen', 'candidate_id': a['candidate_id']}
    return a
