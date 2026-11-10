"""Common LLM backend result type. Every backend call is one fresh, stateless request (explicit history replay)."""
from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, Protocol


@dataclasses.dataclass
class LLMResult:
    action: dict | None             # parsed JSON object, or None
    text: str | None                # raw final text
    llm_tokens: int | None          # input + cache creation + cache read + output (None = unknown)
    llm_usd: float | None
    wall_seconds: float
    accounting_complete: bool
    error: str | None               # validation / transport error; action is None when set
    model: str
    observed_models: list
    attempts: int = 1               # weak-model retries included
    meta: dict = dataclasses.field(default_factory=dict)

    def usage(self) -> dict:
        return {'llm_tokens': self.llm_tokens, 'llm_usd': self.llm_usd, 'wall_seconds': self.wall_seconds}


def is_refusal(result: LLMResult) -> bool:
    """The model declined the request (Claude stop_reason 'refusal', e.g. a safeguard flag such as
    [reasoning_extraction]). The call is an error and is still charged; ClaudeCLIBackend puts stop_reason in meta."""
    return (result.meta or {}).get('stop_reason') == 'refusal'


class Backend(Protocol):
    model: str
    kind: str

    def complete(self, system: str, prompt: str, *, schema: dict | None = None, call_dir=None,
                 role: str = 'proposer') -> LLMResult: ...


class BackendError(RuntimeError):
    pass


_FENCE = re.compile(r'^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$', re.S)
# every fenced block (fences on their own lines, empty blocks included), with its language tag
_FENCES = re.compile(r'^[ \t]*```([^\r\n`]*)\r?\n(.*?)^[ \t]*```[ \t]*\r?$', re.M | re.S)


def _reject_constant(c):
    raise ValueError(f'non-finite JSON constant {c}')


def parse_json_object(text: Any) -> tuple[dict | None, str]:
    """Strict JSON first; then a single fenced block, alone or the only one inside prose (trial C3c: Opus wrote its
    review's reasoning before the block). Returns (object, mode) with mode strict|fenced|embedded|failed."""
    if type(text) is not str:
        return None, 'failed'
    fenced = _FENCE.match(text)
    blocks = [] if fenced else _FENCES.findall(text)
    for mode, candidate in (('strict', text), ('fenced', fenced.group(1) if fenced else None),
                            ('embedded', blocks[0][1] if len(blocks) == 1 and blocks[0][0].strip() in ('', 'json') else None)):
        if candidate is None:
            continue
        try:
            obj = json.loads(candidate, parse_constant=_reject_constant)
            json.dumps(obj, allow_nan=False)          # rejects 1e999 overflowing to inf
        except (ValueError, RecursionError):
            continue
        if type(obj) is dict:
            return obj, mode
    return None, 'failed'
