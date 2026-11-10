"""LLM metering for the ENV-A arms (A-NAT, A-ISW) -- AMENDMENT_EXT_001 Q1/Q4.

Under D1 tokens are not a budget dimension (sentinel cap, no USD), so there is NO ledger change: `MeteringDriver`
wraps a backend, passes every `complete()` result through UNCHANGED, and only *observes* the per-call usage the
backend already records, so the arm summary can report (never charge): `llm_calls`, `llm_tokens_total`,
`llm_tokens_uncached = Σ(prompt − usage.prompt_tokens_details.cached_tokens + completion)`,
`llm_wall_hours = Σ call seconds`, `llm_gpu_hours_h20 = 4 × llm_wall_hours`, `llm_usd = 0`. Driver-side, no torch.
"""
from __future__ import annotations


def _cached(usage: dict) -> int:
    details = usage.get('prompt_tokens_details')
    if isinstance(details, dict) and type(details.get('cached_tokens')) is int:
        return max(0, details['cached_tokens'])
    for k in ('prompt_cache_hit_tokens', 'cached_tokens'):
        if type(usage.get(k)) is int:
            return max(0, usage[k])
    return 0


def uncached_tokens(result) -> int:
    """Σ over the result's attempts of prompt−cached+completion (known usage) or the attempt's upper bound (unknown)."""
    total = 0
    for a in (result.meta or {}).get('attempts') or []:
        u = a.get('usage') if isinstance(a, dict) else None
        if isinstance(u, dict) and type(u.get('prompt_tokens')) is int and type(u.get('completion_tokens')) is int:
            total += max(0, u['prompt_tokens'] - _cached(u)) + u['completion_tokens']
        else:
            ub = ((a.get('usage_upper_bound') or {}) if isinstance(a, dict) else {}).get('llm_tokens')
            total += ub if type(ub) is int else 0
    return total


class LlmMeter:
    """Accumulates the reported-only ENV-A LLM aggregates (no charging)."""

    def __init__(self, serving_gpus: int = 4, gpu_type: str = 'H20'):
        self.calls = 0
        self.tokens_total = 0
        self.tokens_uncached = 0
        self.wall_seconds = 0.0
        self.usd = 0.0                              # what a paid API charged (0 on the self-hosted endpoint)
        self.serving_gpus = int(serving_gpus)       # the self-hosted endpoint's GPUs (0 on a paid API)
        self.gpu_type = gpu_type

    def observe(self, result) -> None:
        self.calls += 1
        if type(result.llm_tokens) is int:
            self.tokens_total += result.llm_tokens
        self.tokens_uncached += uncached_tokens(result)
        if type(result.wall_seconds) in (int, float):
            self.wall_seconds += float(result.wall_seconds)
        if type(getattr(result, 'llm_usd', None)) in (int, float):
            self.usd += float(result.llm_usd)

    def summary(self) -> dict:
        wall_h = self.wall_seconds / 3600.0
        return {'llm_calls': self.calls, 'llm_tokens_total': self.tokens_total,
                'llm_tokens_uncached': self.tokens_uncached, 'llm_wall_hours': round(wall_h, 4),
                f'llm_gpu_hours_{self.gpu_type.lower()}': round(self.serving_gpus * wall_h, 4),
                'llm_usd': round(self.usd, 6)}


class RequestSeed:
    """AMENDMENT_EXT_006: put a `seed` into the request body of every call (vLLM honours it; an endpoint that ignores
    it is unaffected). per_call (default): seed = base + k on the k-th call -- reproducible and never repeated. A
    constant seed (per_call=False) would make an identical replayed prompt sample identically, i.e. bring back the
    deterministic loop the restart is meant to escape."""

    def __init__(self, inner, base: int, *, per_call: bool = True):
        self._inner, self.base, self.per_call, self.calls = inner, int(base), bool(per_call), 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def describe(self) -> dict:
        d = dict(self._inner.describe()) if hasattr(self._inner, 'describe') else {'kind': type(self._inner).__name__}
        d.update(request_seed=self.base, request_seed_mode='base + call index' if self.per_call else 'constant')
        return d

    def complete(self, *args, **kwargs):
        seed = self.base + self.calls if self.per_call else self.base
        body = getattr(self._inner, 'extra_body', None)
        self._inner.extra_body = {**(body if isinstance(body, dict) else {}), 'seed': int(seed)}
        self.calls += 1
        return self._inner.complete(*args, **kwargs)


class MeteringDriver:
    """Wrap a backend, observe each call for the ENV-A LLM report, return results unchanged."""

    def __init__(self, inner, meter: LlmMeter | None = None):
        self._inner = inner
        self.meter = meter or LlmMeter()

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def describe(self) -> dict:
        d = dict(self._inner.describe()) if hasattr(self._inner, 'describe') else {'kind': type(self._inner).__name__}
        d['metered'] = True
        return d

    def complete(self, *args, **kwargs):
        res = self._inner.complete(*args, **kwargs)
        self.meter.observe(res)
        return res


PHASE_E_MODEL = 'Qwen3-30B-A3B'
PHASE_E_CONTEXT = 40_960
NATIVE_CONTEXT = 32_000          # transcript estimate; leaves room for the native agent's 4096-token answer


def phase_e_driver(*, native: bool = False, out_dir=None, transport=None) -> tuple:
    """Both phase-E arms use the same keyless vLLM service; construction makes no endpoint calls.

    D-RB-17 replaces the historical DeepSeek endpoint/context. A-NAT2 keeps EXT_006 sampling and its
    4096 output cap; HLS uses the product's 16384 cap for hypotheses with code. The backend fits each
    request to the server context. The finite token-budget sentinel is inherited from D1.
    """
    import os
    from pathlib import Path
    from ...agent.backends import OpenAICompatBackend

    url = os.environ.get('VLLM_URL', '').strip()
    if not url:
        raise ValueError('set VLLM_URL to the self-hosted Qwen3-30B-A3B OpenAI-compatible base URL (ending /v1)')
    route = 'vllm'
    backend = OpenAICompatBackend(
        base_url=url, model=PHASE_E_MODEL, api_key_env=None, price=None, structured='json_object',
        temperature=1.0 if native else 0.0, max_tokens=4096 if native else 16_384,
        context_tokens=PHASE_E_CONTEXT, timeout_seconds=600,
        extra_body={'chat_template_kwargs': {'enable_thinking': False}},
        selfhost={'route': route, 'engine': 'vllm', 'engine_version': '0.11', 'gpus': 1, 'gpu_type': 'H100'},
        out_root=Path(out_dir) / 'llm_calls' if out_dir else None, transport=transport)
    meter = LlmMeter(serving_gpus=1, gpu_type='H100')
    # A-NAT2 retains EXT_006's historical base-42, per-call sampling policy.
    return MeteringDriver(RequestSeed(backend, 42) if native else backend, meter), route, meter
