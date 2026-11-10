"""Roles: proposer (one JSON action per turn), verifier (may only veto), summarizer (rolling history).

System prompts come from the OptimizerPackage's prompts/ (editable by reflection); placeholders are filled here.
Every call is a fresh stateless request; all roles charge the same search budget.
"""
from __future__ import annotations

import json
from pathlib import Path

from .backends.claude_cli import system_arg_units, MAX_SYSTEM_ARG_UTF16_UNITS
from .context import estimate_tokens
from .tools import STRONG_MENU, SUMMARIZER_SCHEMA, VERIFIER_SCHEMA, WEAK_MENU

PLACEHOLDERS = ('{ACTION_MENU}', '{MODE}', '{LAYERS}', '{PATCH_LINE_LIMIT}', '{TEMPLATES}', '{CHECKLIST}')

DEFAULT_CHECKLIST = [
    'Workload unchanged: GBS (packs per update), sequence length, data order, supervised tokens.',
    'Objective unchanged: loss definition, aux loss, grad clipping, LR schedule, optimizer math.',
    'Precision unchanged: bf16 params/activations, fp32 master/moments/grad reduce; no TF32/FP8 or lower-precision state.',
    'No work moved outside the timed update (no deferred optimizer step, no cross-update caching or prefetch of compute).',
    'Trainable set, router top-k / capacity and model config unchanged.',
    'Overlay touches only the mutable zone and keeps every hook.',
    'The change has a concrete expected whole-update benefit given the parent receipt (not only a component metric).',
]


def fill(template: str, **values) -> str:
    out = template
    for key, value in values.items():
        out = out.replace('{' + key + '}', str(value))
    return out


class Roles:
    def __init__(self, prompts: dict[str, str], mode: str, policy, driver, *, verifier_driver=None, summarizer_driver=None,
                 calls_dir: Path | None = None, template_menu: list | None = None):
        self.prompts, self.mode, self.policy = prompts, mode, policy
        self.driver = driver
        self.verifier_driver = verifier_driver or driver
        self.summarizer_driver = summarizer_driver or driver
        self.calls_dir = calls_dir
        self.n = 0
        menu = WEAK_MENU if mode == 'weak' else STRONG_MENU
        tmpl = json.dumps(template_menu or [], separators=(',', ':'))
        key = 'weak_proposer.md' if mode == 'weak' else 'proposer.md'
        self.proposer_system = fill(prompts[key], ACTION_MENU=menu, MODE=mode, LAYERS=','.join(policy.layers),
                                    PATCH_LINE_LIMIT=policy.patch_line_limit, TEMPLATES=tmpl, CHECKLIST='')
        self.verifier_system = fill(prompts['verifier.md'], CHECKLIST='\n'.join(f'- {x}' for x in DEFAULT_CHECKLIST), MODE=mode)
        self.summarizer_system = prompts['summarizer.md']
        for name, text in (('proposer', self.proposer_system), ('verifier', self.verifier_system), ('summarizer', self.summarizer_system)):
            if system_arg_units(text) > MAX_SYSTEM_ARG_UTF16_UNITS:
                raise ValueError(f'{name} system prompt exceeds the argv budget ({system_arg_units(text)} UTF-16 units)')

    def _dir(self, role: str):
        self.n += 1
        return (self.calls_dir / f'{self.n:04d}_{role}') if self.calls_dir else None

    def propose(self, prompt: str, schema: dict, **kw):
        return self.driver.complete(self.proposer_system, prompt, schema=schema, call_dir=self._dir('proposer'), role='proposer', **kw)

    def verify_prompt(self, view: dict) -> str:
        return 'Candidate to review (JSON):\n' + json.dumps(view, ensure_ascii=False, indent=1, default=str)

    def verify(self, view: dict, **kw):
        return self.verifier_driver.complete(self.verifier_system, self.verify_prompt(view), schema=VERIFIER_SCHEMA,
                                             call_dir=self._dir('verifier'), role='verifier', **kw)

    def summarize(self, previous: str, lines: list[str], max_tokens: int = 1500, **kw):
        prompt = json.dumps({'previous_summary': previous, 'new_turns': lines, 'limit_tokens': max_tokens}, ensure_ascii=False)
        res = self.summarizer_driver.complete(self.summarizer_system, prompt, schema=SUMMARIZER_SCHEMA,
                                              call_dir=self._dir('summarizer'), role='summarizer', **kw)
        if res.action and estimate_tokens(res.action.get('summary', '')) <= max_tokens:
            return res, res.action['summary']
        return res, None
