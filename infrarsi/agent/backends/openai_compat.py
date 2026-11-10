"""OpenAI-compatible chat/completions backend (DeepSeek official API or a self-hosted vLLM endpoint), or the OpenAI
Responses API (`api='responses'`) for endpoints that speak only that.

Weak-model protocol (FINAL_PLAN 7.2): JSON constrained output (response_format json_object, or vLLM guided
decoding), temperature 0, client-side schema validation, and at most `max_retries` re-asks with the validation
errors fed back. Every attempt is charged. The served model must equal the requested one (no silent escalation).
"""
from __future__ import annotations

import datetime as _dt
import http.client
import json
import math
import os
import socket
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Callable

from ..jsonschema_lite import validate
from .base import LLMResult, parse_json_object

# USD per 1M tokens. deepseek-flash: FINAL_REVIEW_FABLE51.md section 1 (official price page, 2026-10-20).
PRICES = {
    'deepseek-chat': {'input_miss': 0.27, 'input_hit': 0.07, 'output': 1.10},
    'deepseek-reasoner': {'input_miss': 0.55, 'input_hit': 0.14, 'output': 2.19},
    'deepseek-flash': {'input_miss': 0.30, 'input_hit': 0.30, 'output': 1.20, 'note': 'peak prices (off-peak 0.15/0.60); cache-hit price unverified, charged at the miss price'},
    'self-hosted': None,
}

Transport = Callable[[str, dict, dict, float], tuple[int, dict]]
BYTES_PER_TOKEN = 3.5                      # conservative input estimate (agent.context.estimate_tokens uses the same)


def classify_transport_error(exc: BaseException) -> tuple[bool, bool]:
    """(unreachable, maybe_served). Refused/unresolvable/connect-timeout: the request never left this PC, nothing
    was served. A connection closed without any response (ssh tunnel up, server down): unreachable, but whether the
    server saw the request is unknown. Anything else (read timeout, reset mid-response): reachable, maybe served."""
    reason = exc.reason if isinstance(exc, urllib.error.URLError) and not isinstance(exc, urllib.error.HTTPError) else exc
    if isinstance(reason, (ConnectionRefusedError, socket.gaierror)):
        return True, False
    if isinstance(reason, OSError) and getattr(reason, 'errno', None) in (10061, 111, 113, 10065, 101, 10051):
        return True, False                 # ECONNREFUSED / EHOSTUNREACH / ENETUNREACH (Windows and POSIX numbers)
    if isinstance(exc, urllib.error.URLError) and isinstance(reason, (TimeoutError, socket.timeout)):
        return True, False                 # timed out while connecting (urlopen wraps connect errors in URLError)
    if isinstance(reason, http.client.RemoteDisconnected):
        return True, True
    return False, True


def chat_usage(u: dict) -> dict:
    """A Responses API usage block in chat-completions terms (what the accounting reads)."""
    return {'prompt_tokens': u.get('input_tokens'), 'completion_tokens': u.get('output_tokens'),
            'prompt_tokens_details': u.get('input_tokens_details'), 'completion_tokens_details': u.get('output_tokens_details')}


def urllib_transport(url: str, headers: dict, body: dict, timeout: float) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'), headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode('utf-8', errors='replace'))
        except ValueError:
            payload = {'error': {'message': str(exc)}}
        return exc.code, payload


class OpenAICompatBackend:
    kind = 'openai_compat'
    call_overhead_tokens = 64             # chat template around the messages

    def __init__(self, *, base_url: str, model: str, api_key: str | None = None, api_key_env: str | None = 'DEEPSEEK_API_KEY',
                 price: dict | None | str = 'auto', structured: str = 'json_object', temperature: float = 0.0,
                 max_tokens: int = 4096, timeout_seconds: float = 300, max_retries: int = 3,
                 transport: Transport | None = None, out_root: str | Path | None = None, extra_body: dict | None = None,
                 served_model: str | None = None, context_tokens: int | None = None, selfhost: dict | None = None,
                 pinger: Callable[[str, float], dict] | None = None, api: str = 'chat', headers: dict | None = None):
        """selfhost: {'route', 'engine', 'engine_version', 'gpus', 'gpu_type'} for a self-hosted endpoint: API USD is 0
        and each call records gpus x wall seconds as serving GPU time (the serving window itself is accounted by the
        orchestrator). context_tokens: the server's max-model-len; requests are fitted to it or refused unsent.
        api: 'chat' (/chat/completions) or 'responses' (/responses); headers: sent with every request (a gateway that
        wants its own User-Agent)."""
        if structured not in ('json_object', 'json_schema', 'guided_json', 'none'):
            raise ValueError(f'structured={structured!r}')
        if api not in ('chat', 'responses') or (api == 'responses' and structured == 'guided_json'):
            raise ValueError(f'api={api!r} (chat or responses; guided_json is a vLLM chat option)')
        self.api, self.headers = api, {str(k): str(v) for k, v in (headers or {}).items()}
        self.base_url, self.model = base_url.rstrip('/'), model
        self.served_model = served_model or model
        self.api_key = api_key if api_key is not None else (os.environ.get(api_key_env) if api_key_env else None)
        self.price = PRICES.get(model) if price == 'auto' else price
        self.unpriced = price == 'auto' and self.price is None and not selfhost   # a paid API, price not known
        self.structured, self.temperature, self.max_tokens = structured, temperature, max_tokens
        self.timeout_seconds, self.max_retries = timeout_seconds, max_retries
        self.transport = transport or urllib_transport
        self.out_root = Path(out_root) if out_root else None
        self.extra_body = extra_body or {}
        self.context_tokens = context_tokens
        self.pinger = pinger                                   # (url, timeout) -> /models JSON; default urllib GET
        self.selfhost = dict(selfhost) if selfhost else None
        if self.selfhost:
            self.price = None                                  # API USD is 0 for a self-hosted endpoint

    def describe(self) -> dict:
        d = {'kind': self.kind, 'model': self.model, 'base_url': self.base_url, 'api': self.api, 'structured': self.structured,
             'temperature': self.temperature, 'max_retries': self.max_retries, 'max_tokens': self.max_tokens,
             'context_tokens': self.context_tokens, 'extra_body': self.extra_body,
             'pricing': self.price or ('unknown: no price for this model (set llm.price); USD not accounted' if self.unpriced
                                       else 'self-hosted (GPU-hours accounted separately)')}
        if self.selfhost:
            d.update(kind='openai_compat_selfhost', route=self.selfhost.get('route'), engine=self.selfhost.get('engine'),
                     engine_version=self.selfhost.get('engine_version'), serving_gpus=self.selfhost.get('gpus'),
                     gpu_type=self.selfhost.get('gpu_type'), pricing='API USD 0; serving cost in GPU-hours of gpu_type')
        return d

    def ping(self, timeout: float = 5.0) -> dict:
        """GET {base_url}/models: {'ok', 'models' | 'error', 'unreachable'}; the configured model must be served."""
        try:
            if self.pinger is not None:
                payload = self.pinger(f'{self.base_url}/models', timeout)
            else:
                with urllib.request.urlopen(urllib.request.Request(f'{self.base_url}/models', method='GET'), timeout=timeout) as r:
                    payload = json.loads(r.read().decode('utf-8'))
            ids = [m.get('id') for m in payload.get('data', [])]
        except Exception as exc:                                       # noqa: BLE001 - reported to the caller
            unreachable, _ = classify_transport_error(exc)
            return {'ok': False, 'unreachable': unreachable or isinstance(exc, (OSError, urllib.error.URLError)),
                    'error': f'{type(exc).__name__}: {exc}'}
        ok = self.model in ids or self.served_model in ids
        return {'ok': ok, 'models': ids, 'unreachable': False, **({} if ok else {'error': f'{self.model!r} not served: {ids}'})}

    def _think(self) -> int:
        """The thinking budget llm.extra_body sets (OpenRouter's `reasoning: {max_tokens: N}`), else 0."""
        r = self.extra_body.get('reasoning') if isinstance(self.extra_body, dict) else None
        think = r.get('max_tokens') if isinstance(r, dict) else None
        return int(think) if type(think) is int and think > 0 else 0

    def _role_cap(self, max_output_tokens, grow: int) -> int | None:
        """A role's output cap for this attempt; with a thinking budget the cap is the room for the answer on top of it:
        the provider refuses a limit at or below the budget (D2-16b)."""
        return int(max_output_tokens) * grow + self._think() if max_output_tokens else None

    def _fit_max_tokens(self, messages: list) -> int | None:
        """Output cap that keeps input + output within the server context; None if even a short answer cannot fit."""
        if not self.context_tokens:
            return int(self.max_tokens)
        est_in = math.ceil(len(json.dumps(messages, ensure_ascii=False).encode('utf-8')) / BYTES_PER_TOKEN)
        room = int(self.context_tokens) - est_in - 64
        return min(int(self.max_tokens), room) if room >= 256 else None

    def _usd(self, usage: dict) -> float:
        if not self.price:
            return 0.0
        hit, miss = usage.get('prompt_cache_hit_tokens'), usage.get('prompt_cache_miss_tokens')
        if hit is None and miss is None:
            hit, miss = 0, usage['prompt_tokens']
        elif miss is None:
            miss = usage['prompt_tokens'] - hit
        elif hit is None:
            hit = usage['prompt_tokens'] - miss
        return (miss * self.price['input_miss'] + hit * self.price['input_hit'] + usage['completion_tokens'] * self.price['output']) / 1e6

    @staticmethod
    def _usage_ok(usage) -> bool:
        """Verified usage: non-negative integer prompt/completion counts (and cache counts when present)."""
        if type(usage) is not dict:
            return False
        for k in ('prompt_tokens', 'completion_tokens', 'prompt_cache_hit_tokens', 'prompt_cache_miss_tokens'):
            v = usage.get(k)
            if v is None and k.startswith('prompt_cache'):
                continue
            if type(v) is not int or v < 0:
                return False
        hit, miss, total = usage.get('prompt_cache_hit_tokens'), usage.get('prompt_cache_miss_tokens'), usage['prompt_tokens']
        if hit is not None and miss is not None:
            return hit + miss == total                   # the two cache counts must add up to the prompt
        return (hit or 0) <= total and (miss or 0) <= total

    def _upper_bound(self, messages: list, max_out: int | None = None) -> tuple[int, float]:
        """Per-attempt charge when usage is unknown: every input byte as a token plus the attempt's output cap (what it
        asked for; llm.max_tokens when not known), at the input miss / output prices (never zero; accounting stays
        incomplete)."""
        inp = len(json.dumps(messages, ensure_ascii=False).encode('utf-8'))
        out = int(max_out or self.max_tokens)
        if not self.price:
            return inp + out, 0.0
        return inp + out, (inp * self.price['input_miss'] + out * self.price['output']) / 1e6

    def _body(self, messages: list, schema: dict | None, max_tokens: int | None = None) -> dict:
        if self.api == 'responses':                  # no temperature: reasoning models reject it (llm.extra_body may add one)
            body = {**self.extra_body, 'model': self.model, 'instructions': messages[0]['content'],
                    'input': [{'role': m['role'], 'content': m['content']} for m in messages[1:]],
                    'max_output_tokens': int(max_tokens or self.max_tokens), 'store': False, 'stream': False}
            if self.structured == 'json_object':
                body['text'] = {'format': {'type': 'json_object'}}
            elif self.structured == 'json_schema' and schema:
                body['text'] = {'format': {'type': 'json_schema', 'name': 'action', 'schema': schema, 'strict': False}}
            return body
        body = {**self.extra_body, 'model': self.model, 'messages': messages, 'temperature': self.temperature,
                'max_tokens': int(max_tokens or self.max_tokens), 'stream': False}       # fixed fields always win
        if self.structured == 'json_object':
            body['response_format'] = {'type': 'json_object'}
        elif self.structured == 'json_schema' and schema:
            body['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'action', 'schema': schema, 'strict': False}}
        elif self.structured == 'guided_json' and schema:
            body['guided_json'] = schema
            body['response_format'] = {'type': 'json_object'}
        return body

    def _usd_room(self, messages: list, max_usd, usd: float) -> int | None:
        """Output tokens the next attempt may produce within the USD the caller has left (its input charged byte by byte
        at the miss price, as the admission does); None without a USD cap or an output price."""
        if max_usd is None or not self.price or not self.price.get('output'):
            return None
        inp = len(json.dumps(messages, ensure_ascii=False).encode('utf-8'))
        return int(((float(max_usd) - usd) * 1e6 - inp * float(self.price.get('input_miss') or 0.0)) / float(self.price['output']))

    def _answer(self, resp: dict) -> tuple[str | None, bool]:
        """(the reply's text, whether it stopped at the output cap) of a chat or Responses API answer."""
        if self.api == 'responses':
            parts = [c.get('text') or '' for o in resp.get('output') or [] if type(o) is dict and o.get('type') == 'message'
                     for c in o.get('content') or [] if type(c) is dict and c.get('type') == 'output_text']
            cut = resp.get('status') == 'incomplete' and (resp.get('incomplete_details') or {}).get('reason') == 'max_output_tokens'
            return (''.join(parts) if parts else None), cut
        try:
            choice = resp['choices'][0]
            return choice['message']['content'], choice.get('finish_reason') == 'length'
        except (KeyError, IndexError, TypeError):
            return None, False

    def complete(self, system: str, prompt: str, *, schema: dict | None = None, call_dir=None, role: str = 'proposer',
                 expect_json: bool = True, timeout: float | None = None, max_output_tokens: int | None = None,
                 token_allowance: int | None = None, max_usd: float | None = None, **_ignored) -> LLMResult:
        call_id = uuid.uuid4().hex
        directory = Path(call_dir) if call_dir else ((self.out_root / f'{role}_{call_id[:12]}') if self.out_root else None)
        if directory:
            directory.mkdir(parents=True, exist_ok=False)
        headers = {'Content-Type': 'application/json', **self.headers}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}]
        started = time.monotonic()
        budget_s = self.timeout_seconds if timeout is None else max(1.0, min(float(self.timeout_seconds), float(timeout)))
        tokens, usd, complete, attempts, log = 0, 0.0, True, [], []
        action, text, error, observed = None, None, None, []
        terminal = None                              # a transport failure ended the call: 'unreachable' or 'transport'
        grow = 1                                     # doubles after a reply cut at the output cap (F-D1-16-3)
        url = f'{self.base_url}/{"responses" if self.api == "responses" else "chat/completions"}'
        for attempt in range(1 + self.max_retries):
            remaining = budget_s - (time.monotonic() - started)
            if remaining <= 0:                       # a cut or rejected reply: say what ended its retries
                error = (f'{error}; the call\'s time budget ({budget_s:.0f} s, llm.timeout_s) ended the retries' if error
                         else f'timeout: {budget_s:.0f}s call budget used')
                break
            max_out = self._fit_max_tokens(messages)
            if max_out is None:                      # refused unsent: nothing served, nothing to charge
                error = f'context too long for the endpoint (> {self.context_tokens} tokens with the answer); not sent'
                attempts.append({'attempt': attempt, 'status': None, 'seconds': 0.0, 'error': error})
                break
            if max_output_tokens:
                max_out = min(max_out, self._role_cap(max_output_tokens, grow))
            if token_allowance is not None:          # every attempt (re-asks too) must fit what the caller has left
                worst_in = math.ceil(len(json.dumps(messages, ensure_ascii=False).encode('utf-8')) / BYTES_PER_TOKEN)
                room = int(token_allowance) - tokens - worst_in - self.call_overhead_tokens
                if room < 256:
                    error = error or 'token allowance exhausted before this attempt; not sent'
                    break
                max_out = min(max_out, room)
            usd_room = self._usd_room(messages, max_usd, usd)     # and its worst case within the USD left
            if usd_room is not None:
                if usd_room < 256:
                    error = error or 'USD allowance exhausted before this attempt; not sent'
                    break
                max_out = min(max_out, usd_room)
            if self._think() and max_out <= self._think():   # what the provider would refuse: not sent, not charged
                error = (f'the thinking budget (llm.extra_body reasoning.max_tokens {self._think()}) leaves no room for '
                         f'the answer within the output limit of this call ({max_out} tokens: llm.max_tokens, the context, '
                         'the token or USD allowance): lower the budget or raise the limits; not sent')
                attempts.append({'attempt': attempt, 'status': None, 'seconds': 0.0, 'max_tokens': max_out, 'error': error})
                break
            t0 = time.monotonic()
            unreachable = maybe_served = False
            try:
                status, resp = self.transport(url, headers, self._body(messages, schema, max_out), remaining)
            except Exception as exc:                 # transport failure: the request may have been served and billed
                status, resp = 0, {'error': {'message': f'{type(exc).__name__}: {exc}'}}
                unreachable, maybe_served = classify_transport_error(exc)
            if type(resp) is not dict:
                resp = {'error': {'message': f'non-object response {type(resp).__name__}'}}
            if self.api == 'responses' and type(resp.get('usage')) is dict:
                resp = {**resp, 'usage': chat_usage(resp['usage'])}
            row = {'attempt': attempt, 'status': status, 'seconds': time.monotonic() - t0, 'max_tokens': max_out}
            if status == 0:
                # Transport failures are terminal for this call (never resent: the server may already have served it).
                # Unreachable endpoint (refused / connect timeout / closed without response): fail fast, an operator
                # problem rather than a model error. A request that provably never left is free and complete; any other
                # is charged one conservative upper bound and leaves the accounting incomplete.
                msg = resp['error']['message']
                failure = f'endpoint unreachable: {msg}' if unreachable else f'transport error: {msg}'
                terminal = 'unreachable' if unreachable else 'transport'
                if error and time.monotonic() - started >= budget_s - 1.0:   # the retry of a cut or rejected reply
                    error = f"{error}; the call's time budget ({budget_s:.0f} s, llm.timeout_s) ended the retries ({failure})"
                else:
                    error = failure
                row['error'] = failure
                if maybe_served:
                    ub_tokens, ub_usd = self._upper_bound(messages, max_out)
                    tokens, usd, complete = tokens + ub_tokens, usd + ub_usd, False
                    row['usage_upper_bound'] = {'llm_tokens': ub_tokens, 'llm_usd': ub_usd}
                attempts.append(row)
                break
            usage = resp.get('usage')
            if self._usage_ok(usage):
                tokens += usage['prompt_tokens'] + usage['completion_tokens']
                usd += self._usd(usage)
                row['usage'] = usage
            else:
                # Unknown, malformed or negative usage on any attempt (also HTTP errors and transport failures):
                # charge a conservative upper bound and mark the accounting incomplete.
                ub_tokens, ub_usd = self._upper_bound(messages, max_out)
                tokens += ub_tokens
                usd += ub_usd
                complete = False
                row.update(usage_upper_bound={'llm_tokens': ub_tokens, 'llm_usd': ub_usd}, usage_reported=usage)
            if status != 200:
                error = f'HTTP {status}: {str((resp or {}).get("error", resp))[:300]}'
                row['error'] = error
                attempts.append(row)
                if status in (400, 401, 403, 404, 422):
                    break                            # not retryable
                continue
            served = resp.get('model')
            observed.append(served)
            text, cut = self._answer(resp)
            if cut:
                row['cut'] = True                    # stopped at max_out (a reasoning model's thinking counts)
            if directory:                            # every answer, a rejected one before its retry too
                (directory / f'answer_{attempt}.txt').write_text(text or '', encoding='utf-8')
            if served not in (self.model, self.served_model):
                error = f'served model {served!r} != requested {self.model!r}'
                row['error'] = error
                attempts.append(row)
                break                                # never accept a different (possibly stronger) model
            if not expect_json:
                action, error = None, None
                attempts.append(row)
                break
            obj, mode = parse_json_object(text)
            problems = ['response is not a JSON object'] if obj is None else (validate(obj, schema) if schema else [])
            row['parse_mode'] = mode
            if not problems:
                action, error = obj, None
                attempts.append(row)
                break
            if cut:                                  # no answer to correct: the same request with more room
                room = self._fit_max_tokens(messages) or 0
                if max_output_tokens:
                    room = min(room, self._role_cap(max_output_tokens, grow * 2))
                if token_allowance is not None:
                    worst_in = math.ceil(len(json.dumps(messages, ensure_ascii=False).encode('utf-8')) / BYTES_PER_TOKEN)
                    room = min(room, int(token_allowance) - tokens - worst_in - self.call_overhead_tokens)
                usd_room = self._usd_room(messages, max_usd, usd)
                if usd_room is not None:
                    room = min(room, usd_room)
                error = (f'the reply was cut at the output limit ({max_out} tokens) before its JSON ended'
                         + ('' if room > max_out else f'; no larger limit is left (llm.max_tokens {self.max_tokens}, the '
                            'token or USD allowance)')
                         + ': a reasoning model\'s thinking counts: raise llm.max_tokens or limit its reasoning (llm.extra_body)')
                row['error'] = error
                attempts.append(row)
                if room <= max_out:
                    break                            # the same request would be cut again
                grow *= 2
                continue
            error = 'schema: ' + '; '.join(problems[:6])
            row['error'] = error
            attempts.append(row)
            messages = messages + [{'role': 'assistant', 'content': text or '(an empty reply)'},
                                   {'role': 'user', 'content': 'Your reply was rejected: ' + '; '.join(problems[:6]) +
                                    '. Reply again with exactly one JSON object that satisfies the action schema.'}]
        wall = time.monotonic() - started
        complete = complete and not self.unpriced        # its USD are unknown, not zero
        busy = sum(a.get('seconds') or 0.0 for a in attempts)
        serving_gpu_s = round(busy * int(self.selfhost.get('gpus') or 0), 3) if self.selfhost else None
        if directory:
            (directory / 'receipt.json').write_text(json.dumps(
                {'schema': 'infrarsi-openai-call-v1', 'call_id': call_id, 'role': role, 'model': self.model,
                 'base_url': self.base_url, 'utc': _dt.datetime.now(_dt.timezone.utc).isoformat(), 'attempts': attempts,
                 'llm_tokens': tokens, 'llm_usd': usd, 'accounting_complete': complete, 'error': error, 'wall_seconds': wall,
                 'selfhost_gpu_seconds': serving_gpu_s, 'describe': self.describe()},
                indent=1, default=str), encoding='utf-8')
            (directory / 'request.json').write_text(json.dumps({'system': system, 'prompt': prompt, 'schema': schema}), encoding='utf-8')
        # Known usage is always reported (charged); accounting_complete says whether it is the whole cost.
        return LLMResult(action=action if error is None else None, text=text, llm_tokens=tokens,
                         llm_usd=usd, wall_seconds=wall, accounting_complete=complete, error=error,
                         model=self.model, observed_models=sorted({m for m in observed if m}), attempts=len(attempts),
                         meta={'call_id': call_id, 'attempts': attempts, 'partial_tokens': tokens, 'partial_usd': usd,
                               'selfhost_gpu_seconds': serving_gpu_s,
                               'gpu_type': (self.selfhost or {}).get('gpu_type'),
                               'unreachable': terminal == 'unreachable', 'transport_error': terminal is not None})
