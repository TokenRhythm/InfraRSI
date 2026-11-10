"""Claude through the Anthropic API (official `anthropic` SDK, an optional dependency: `pip install anthropic`).

One stateless Messages request per call (explicit history replay, like every backend): the system prompt is cached
(automatic prompt caching), thinking is adaptive (the default of current models) with an explicit effort, and the
reply's JSON action is parsed from the text and checked against the role's schema. Server-side refusal fallbacks are
on by default (beta `server-side-fallback-2026-07-01`, `fallbacks: "default"`); a turn served by a fallback model is
priced at that model and marked as incomplete accounting. Credentials: ANTHROPIC_API_KEY or an `ant auth login`
profile (resolved by the SDK).
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from ..jsonschema_lite import validate
from .base import LLMResult, parse_json_object

DEFAULT_MODEL = 'claude-opus-5-5'
FALLBACK_BETA = 'server-side-fallback-2026-07-01'
# USD per 1M tokens: input, output, cache read, cache write (5-minute TTL = 1.25 x input)
PRICES = {'claude-opus-5-5': (4.0, 20.0, 0.20, 5.0), 'claude-sonnet-5-5': (2.0, 10.0, 0.20, 2.5),
          'claude-haiku-5-5': (0.10, 0.50, 0.01, 0.125), 'claude-fable-5-1': (10.0, 50.0, 0.25, 12.5),
          'claude-opus-5': (5.0, 25.0, 0.50, 6.25), 'claude-sonnet-5': (2.0, 10.0, 0.20, 2.5)}


def price_usd(model: str, usage) -> float | None:
    p = PRICES.get(model)
    if p is None or usage is None:
        return None
    g = lambda k: int(getattr(usage, k, 0) or 0)                       # noqa: E731
    return (g('input_tokens') * p[0] + g('output_tokens') * p[1] + g('cache_read_input_tokens') * p[2]
            + g('cache_creation_input_tokens') * p[3]) / 1e6


class AnthropicBackend:
    kind = 'anthropic_api'
    call_overhead_tokens = 64

    def __init__(self, *, model: str = DEFAULT_MODEL, effort: str = 'high', max_tokens: int = 16000,
                 timeout_seconds: float = 600, max_retries: int = 2, fallbacks: bool = True, client=None,
                 out_root: str | Path | None = None, base_url: str | None = None, api_key_env: str | None = None):
        self.model, self.effort, self.max_tokens = model, effort, int(max_tokens)
        self.timeout_seconds, self.max_retries, self.fallbacks = timeout_seconds, max_retries, fallbacks
        self._client = client
        self.out_root = Path(out_root) if out_root else None
        self.price = PRICES.get(model)
        self.base_url, self.api_key_env = base_url, api_key_env

    def describe(self) -> dict:
        return {'kind': self.kind, 'model': self.model, 'effort': self.effort, 'max_tokens': self.max_tokens,
                'fallbacks': 'default' if self.fallbacks else None, 'pricing_usd_per_mtok': self.price}

    def client(self):
        if self._client is None:
            import anthropic
            options = {}
            if self.base_url:
                options['base_url'] = os.path.expandvars(self.base_url)
            if self.api_key_env:
                options['api_key'] = os.environ.get(self.api_key_env)
                if not options['api_key']:
                    raise ValueError(f'llm.api_key_env {self.api_key_env!r} is not set')
            self._client = anthropic.Anthropic(max_retries=self.max_retries, timeout=self.timeout_seconds, **options)
        return self._client

    def complete(self, system: str, prompt: str, *, schema: dict | None = None, call_dir=None, role: str = 'proposer',
                 timeout: float | None = None, max_output_tokens: int | None = None, max_usd: float | None = None,
                 token_allowance: int | None = None, **_ignored) -> LLMResult:
        want = int(min(self.max_tokens, max_output_tokens or self.max_tokens))
        inp = len((system + prompt).encode('utf-8')) // 3              # the input estimate (a byte per 3 tokens)
        p_in = max(self.price[0], self.price[3]) if self.price else 0.0  # the prompt may be written to the cache
        out_cap = want                                                 # the answer cut to what the caller has left
        if token_allowance is not None:
            out_cap = min(out_cap, int(token_allowance) - inp)
        if max_usd is not None and self.price:
            out_cap = min(out_cap, int((float(max_usd) * 1e6 - inp * p_in) / self.price[1]))
        if out_cap < min(256, want):                                   # no room for an answer: not sent, no cost
            return LLMResult(action=None, text=None, llm_tokens=0, llm_usd=0.0, wall_seconds=0.0, accounting_complete=True,
                             error='token or USD allowance exhausted before this call; not sent', model=self.model,
                             observed_models=[], meta={'role': role})
        call_id = uuid.uuid4().hex
        kw = dict(model=self.model, max_tokens=out_cap, system=system, messages=[{'role': 'user', 'content': prompt}],
                  output_config={'effort': self.effort}, cache_control={'type': 'ephemeral'})
        started = time.monotonic()
        meta = {'call_id': call_id, 'role': role}
        resp, error = None, None
        try:
            c = self.client()
            if timeout:
                c = c.with_options(timeout=min(float(timeout), self.timeout_seconds))
            if self.fallbacks:
                resp = c.beta.messages.create(betas=[FALLBACK_BETA], fallbacks='default', **kw)
            else:
                resp = c.messages.create(**kw)
        except Exception as exc:                                    # noqa: BLE001 - classified below, never raised
            name = type(exc).__name__
            error = f'{name}: {str(exc)[:500]}'
            meta['unreachable'] = name == 'APIConnectionError'           # network down; a timeout is not
            meta['status_code'] = getattr(exc, 'status_code', None)
        wall = time.monotonic() - started
        if resp is None:
            # The request may or may not have been served: charge the input estimate plus the output cap.
            bound = inp + out_cap
            usd = ((inp * p_in + out_cap * self.price[1]) / 1e6) if self.price else None
            if call_dir is not None or self.out_root is not None:
                d = Path(call_dir) if call_dir is not None else self.out_root / f'{role}_{call_id[:12]}'
                d.mkdir(parents=True, exist_ok=True)
                (d / 'call.json').write_text(json.dumps({'system': system, 'prompt': prompt, 'text': None, 'error': error,
                                                         'meta': meta}, ensure_ascii=False, default=str), encoding='utf-8')
            return LLMResult(action=None, text=None, llm_tokens=bound, llm_usd=usd, wall_seconds=wall,
                             accounting_complete=False, error=error, model=self.model, observed_models=[], meta=meta)
        text = ''.join(getattr(b, 'text', '') for b in resp.content if getattr(b, 'type', None) == 'text')
        served = getattr(resp, 'model', self.model) or self.model
        usage = getattr(resp, 'usage', None)
        tokens = sum(int(getattr(usage, k, 0) or 0) for k in ('input_tokens', 'output_tokens', 'cache_read_input_tokens',
                                                              'cache_creation_input_tokens')) if usage else None
        usd = price_usd(served, usage)
        fell_back = any(getattr(i, 'type', None) == 'fallback_message' for i in (getattr(usage, 'iterations', None) or []))
        meta.update(stop_reason=resp.stop_reason, served_model=served, fallback=fell_back)
        action = None
        if resp.stop_reason == 'refusal':
            error = f'refusal ({getattr(getattr(resp, "stop_details", None), "category", None)})'
        elif resp.stop_reason == 'max_tokens':
            error = 'invalid structured JSON: the reply hit max_tokens'
        else:
            action, mode = parse_json_object(text)
            meta['parse_mode'] = mode
            if action is None:
                error = 'response is not a JSON object'
            elif schema:
                errs = validate(action, schema)
                if errs:
                    error, action = f'schema: {"; ".join(errs[:4])}', None
        if call_dir is not None or self.out_root is not None:
            d = Path(call_dir) if call_dir is not None else self.out_root / f'{role}_{call_id[:12]}'
            d.mkdir(parents=True, exist_ok=True)
            (d / 'call.json').write_text(json.dumps({'system': system, 'prompt': prompt, 'text': text, 'meta': meta,
                                                     'tokens': tokens, 'usd': usd}, ensure_ascii=False), encoding='utf-8')
        return LLMResult(action=action, text=text, llm_tokens=tokens, llm_usd=usd, wall_seconds=wall,
                         accounting_complete=usage is not None and usd is not None and not fell_back, error=error,
                         model=self.model, observed_models=[served], meta=meta)
