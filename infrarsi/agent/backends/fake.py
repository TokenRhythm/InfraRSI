"""Scripted backend for CPU tests: replays a list of actions (or a callable) with synthetic usage."""
from __future__ import annotations

import json

from ..jsonschema_lite import validate
from .base import LLMResult


class ScriptedBackend:
    kind = 'scripted'

    def __init__(self, script, *, model: str = 'scripted-model', tokens_per_call: int = 1000, usd_per_call: float = 0.001,
                 seconds_per_call: float = 1.0, clock=None, validate_schema: bool = True):
        self.script = script                 # list of dict/str, or callable(role, system, prompt) -> dict/str
        self.model, self.tokens_per_call, self.usd_per_call = model, tokens_per_call, usd_per_call
        self.seconds_per_call, self.clock, self.validate_schema = seconds_per_call, clock, validate_schema
        self.calls = []

    def describe(self) -> dict:
        return {'kind': self.kind, 'model': self.model}

    call_overhead_tokens = 0

    @property
    def max_output_tokens(self) -> int:
        return int(self.tokens_per_call)        # a scripted call is charged exactly this much

    def complete(self, system: str, prompt: str, *, schema=None, call_dir=None, role='proposer', expect_json=True, **_):
        if callable(self.script):
            item = self.script(role, system, prompt)
        else:
            item = self.script.pop(0) if self.script else {'action': 'finish', 'reason': 'script exhausted'}
        self.calls.append({'role': role, 'system': system, 'prompt': prompt, 'reply': item})
        if self.clock is not None:
            self.clock.advance(self.seconds_per_call)
        text = item if type(item) is str else json.dumps(item)
        action, error = None, None
        if expect_json:
            try:
                action = json.loads(text)
                if type(action) is not dict:
                    raise ValueError('not an object')
            except ValueError:
                action, error = None, 'invalid structured JSON action'
            if action is not None and schema and self.validate_schema:
                problems = validate(action, schema)
                if problems:
                    action, error = None, 'schema: ' + '; '.join(problems[:4])
        return LLMResult(action=action, text=text, llm_tokens=self.tokens_per_call, llm_usd=self.usd_per_call,
                         wall_seconds=self.seconds_per_call, accounting_complete=True, error=error, model=self.model,
                         observed_models=[self.model], meta={'call_index': len(self.calls)})
