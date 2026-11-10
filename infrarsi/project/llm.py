"""The LLM that drives the agent strategy of a project run, from the manifest's `llm` block or `--llm backend[:model]`.

    anthropic[:model]        Anthropic API, official SDK (ANTHROPIC_API_KEY or an `ant auth login` profile); default claude-opus-5-5
    claude-cli[:model]       the local Claude Code CLI (tools disabled, one fresh session per call)
    codex-cli[:model]        the local Codex CLI login (default: its configured model)
    deepseek[:model]         DeepSeek official API (DEEPSEEK_API_KEY; default deepseek-chat)
    openai:model             OpenAI API (OPENAI_API_KEY)
    openai-compat:model      any OpenAI-compatible server: llm.base_url (and llm.api_key_env); llm.api: responses for
                             an endpoint that speaks only the Responses API, llm.headers for headers it needs;
                             base_url may name environment variables ($VLLM_URL)
    vllm:model               a self-hosted vLLM/SGLang server (llm.base_url, default http://127.0.0.1:8000/v1); llm.extra_body
                             is merged into every request (e.g. {chat_template_kwargs: {enable_thinking: false}} for Qwen3)
    none                     no LLM: the search runs without one (TPE / random)
"""
from __future__ import annotations

import os
import re
import time
import urllib.parse


def parse(spec) -> dict:
    if spec is None or spec == '' or spec == 'none':
        return {}
    if isinstance(spec, dict):
        return {k: v for k, v in spec.items() if v is not None}
    backend, _, model = str(spec).partition(':')
    return {'backend': backend, **({'model': model} if model else {})}


def override(manifest, cli) -> dict:
    """--llm over the manifest's llm block: the backend and model from the command line; the manifest's endpoint
    settings (base_url, api_key_env, context_tokens, extra_body, ...) when it names the same backend, its capability
    only for the same model. `none` turns the model off."""
    c = parse(cli)
    if not c:
        return {'backend': 'none'}
    m = parse(manifest)
    if m.get('backend') != c.get('backend'):
        return c
    keep = {k: v for k, v in m.items() if k != 'capability' or c.get('model') in (None, m.get('model'))}
    return {**keep, **c}


def make_driver(spec, *, out_root=None):
    s = parse(spec)
    kind = s.get('backend')
    if not kind or kind == 'none':
        return None
    from ..agent.backends import AnthropicBackend, ClaudeCLIBackend, CodexCLIBackend, OpenAICompatBackend
    model = s.get('model')
    if kind == 'anthropic':
        return AnthropicBackend(**({'model': model} if model else {}), effort=s.get('effort', 'high'),
                                fallbacks=s.get('fallbacks', True), out_root=out_root,
                                base_url=s.get('base_url'), api_key_env=s.get('api_key_env'))
    if kind in ('claude-cli', 'claude'):                 # subscription login uses the user's model aliases
        return ClaudeCLIBackend(**({'model': model} if model else {}), out_root=out_root, max_retries=1,
                                effort=s.get('effort', 'high'), executable=s.get('executable'), strict_audit=False)
    if kind in ('codex-cli', 'codex'):
        return CodexCLIBackend(model=model or 'default', out_root=out_root, max_retries=1,
                               effort=s.get('effort', 'high'), executable=s.get('executable'))
    if kind in ('openai', 'openai-compat', 'vllm', 'deepseek'):
        if kind == 'deepseek':
            model = model or os.environ.get('INFRARSI_DEEPSEEK_MODEL', 'deepseek-chat')
        if not model:
            raise ValueError(f'llm backend {kind} needs a model name ({kind}:<model>)')
        base = {'openai': os.environ.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1',
                'openai-compat': os.environ.get('OPENAI_BASE_URL') or os.environ.get('OPENROUTER_BASE_URL'),
                'vllm': 'http://127.0.0.1:8000/v1',
                'deepseek': os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')}.get(kind)
        raw = s.get('base_url') or base
        if not raw:
            raise ValueError('openai-compat needs llm.base_url')
        base = os.path.expandvars(str(raw))             # F-F-1: `$VLLM_URL` was sent as written, every call failed
        unset = re.findall(r'\$\{?(\w+)', base)
        if unset:
            raise ValueError(f'llm.base_url {raw!r}: the environment variable {unset[0]} is not set')
        u = urllib.parse.urlsplit(base)
        try:
            port_ok = u.port is None or u.port > 0
        except ValueError:                              # a port that is not a number
            port_ok = False
        if u.scheme not in ('http', 'https') or not u.hostname or not port_ok:
            raise ValueError(f'llm.base_url {raw!r} is not an http(s) URL with a host (and a valid port)')
        key_env = s.get('api_key_env') or {'openai': 'OPENAI_API_KEY', 'deepseek': 'DEEPSEEK_API_KEY'}.get(kind)
        if kind == 'openai-compat' and not key_env:
            key_env = next((k for k in ('OPENAI_API_KEY', 'OPENROUTER_API_KEY') if os.environ.get(k)), None)
        # the output cap is fitted to context_tokens per request; the backend's own 4096 cut hypotheses with code (C1b)
        # a paid API's price from the backend's table unless given (USD unknown when it has none); vLLM: self-hosted
        return OpenAICompatBackend(base_url=base, model=model, api_key_env=key_env,
                                   price=s.get('price', None if kind == 'vllm' else 'auto'),
                                   structured=s.get('structured', 'json_object'), out_root=out_root,
                                   max_tokens=int(s.get('max_tokens') or 16384),
                                   timeout_seconds=float(s.get('timeout_s') or 300),   # per call, its retries included
                                   context_tokens=s.get('context_tokens'), extra_body=s.get('extra_body'),
                                   api=s.get('api', 'chat'), headers=s.get('headers'))
    raise ValueError(f'unknown llm backend {kind!r} (anthropic, claude-cli/claude, codex-cli/codex, deepseek, '
                     'openai, openai-compat, vllm, none)')


BACKENDS = ('anthropic', 'claude-cli', 'claude', 'codex-cli', 'codex', 'deepseek', 'openai', 'openai-compat', 'vllm', 'none')


def debate_settings(raw, *, enabled=None, second=None, verifier=None) -> dict:
    """The debate's settings: the manifest's `debate:` block with the command line over it (`--second-llm` also turns it on)."""
    d = dict(raw) if isinstance(raw, dict) else {}
    if second is not None:
        d['second'] = second
    if verifier is not None:
        d['verifier'] = verifier
    d['enabled'] = bool(d.get('enabled')) if enabled is None and second is None else (enabled is not False)
    return d


def make_debate(driver, settings: dict, *, seed: int = 42):
    """`driver` with the debate's second model around it (agent/debate.py); ValueError when the debate cannot be made."""
    from ..agent.debate import DebateDriver
    second = make_driver(settings.get('second'))
    if second is None:
        raise ValueError('the debate needs a second model: debate.second or --second-llm backend[:model]')
    spec = settings.get('verifier')
    verifier = None if spec in (None, 'none') else make_driver(spec)
    rounds = 0 if spec == 'none' else int(settings.get('verify_rounds', 2))
    return DebateDriver(driver, second, verifier=verifier, rounds=rounds, seed=seed)


# a refusal that retrying will not change: the provider refuses the model or the key, or the endpoint is not there
REFUSED = re.compile(r'(HTTP (400|401|403|404)\b|endpoint unreachable)')
REFUSED_CODES = (400, 401, 403, 404)
# U-5-2 (doctor, run): a model the price table does not know reports 0 USD per call, so a USD budget never triggers
UNPRICED = ('no price is known for this model: USD are not counted and search.budget.llm_usd cannot stop a run '
            '(set llm.price, USD per 1M tokens: {input_miss, input_hit, output})')


def preflight(driver) -> dict:
    """One small JSON call before any GPU time (trial D1-16b: a provider refused the model from this network, every call
    of the run failed after the reference run and the search went on without the model). {'ok', 'refused', 'error',
    'seconds', 'llm_tokens'}; its tokens are not charged to a run."""
    t0 = time.monotonic()
    try:
        res = driver.complete('You check that a model endpoint answers. Reply with exactly one JSON object.',
                              'Reply with the JSON object {"ok": true}.', schema={'type': 'object'}, role='preflight',
                              max_output_tokens=2048)
    except Exception as exc:                            # noqa: BLE001 - reported to the user
        return {'ok': False, 'refused': True, 'error': f'{type(exc).__name__}: {exc}',
                'seconds': round(time.monotonic() - t0, 1), 'llm_tokens': None}
    err, meta = getattr(res, 'error', None), getattr(res, 'meta', None) or {}
    refused = bool(err) and bool(REFUSED.match(err) or meta.get('status_code') in REFUSED_CODES or meta.get('unreachable'))
    return {'ok': not err and isinstance(res.action, dict), 'refused': refused, 'error': err,
            'seconds': round(time.monotonic() - t0, 1), 'llm_tokens': getattr(res, 'llm_tokens', None)}
