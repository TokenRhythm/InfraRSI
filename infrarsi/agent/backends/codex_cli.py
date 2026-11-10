"""Stateless Codex CLI requests using the local login, JSONL usage, and structured replies."""
from __future__ import annotations

import dataclasses
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from ..jsonschema_lite import validate
from .base import LLMResult, parse_json_object


def _configured_model() -> str | None:
    """Preserve the default model while exec ignores the rest of the user's configuration."""
    try:
        try:
            import tomllib
        except ImportError:  # Python 3.10
            import tomli as tomllib
        config = Path(os.environ.get('CODEX_HOME') or Path.home() / '.codex') / 'config.toml'
        with config.open('rb') as src:
            model = tomllib.load(src).get('model')
        return model if isinstance(model, str) and model.strip() else None
    except (OSError, ValueError):
        return None


def _find_executable(explicit: str | None = None) -> Path:
    explicit = explicit or os.environ.get('INFRARSI_CODEX_EXE')
    found = explicit or shutil.which('codex') or shutil.which('codex.exe') or shutil.which('codex.cmd')
    if not found:
        raise FileNotFoundError('codex CLI not found')
    return Path(found).resolve(strict=True)


def parse_codex_stream(raw: str, *, exit_code=0, status: str = 'RETURNED', expect_json: bool = True,
                       schema: dict | None = None):
    """Return (action, fields), preserving charged usage even for failed or malformed replies."""
    texts, usages, errors = [], [], []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError('event is not an object')
            json.dumps(event, allow_nan=False)
        except (ValueError, RecursionError):
            errors.append('unexpected stream event shape')
            continue
        kind = event.get('type')
        if kind == 'item.completed':
            item = event.get('item') or {}
            if isinstance(item, dict) and item.get('type') == 'agent_message' and isinstance(item.get('text'), str):
                texts.append(item['text'])
        elif kind == 'turn.completed':
            usages.append(event.get('usage'))
        elif kind in ('turn.failed', 'error'):
            errors.append('CLI failed: ' + str(event.get('error') or event.get('message') or kind)[:300])
    keys = ('input_tokens', 'cached_input_tokens', 'output_tokens', 'reasoning_output_tokens')
    counts = dict.fromkeys(keys, 0)
    usage_ok = bool(usages)
    for usage in usages:
        if not isinstance(usage, dict):
            usage_ok = False
            continue
        # Older CLIs omit the optional cache/reasoning counters.
        row = {k: usage.get(k, 0) if k in ('cached_input_tokens', 'reasoning_output_tokens') else usage.get(k) for k in keys}
        if not all(type(v) is int and v >= 0 for v in row.values()):
            usage_ok = False
            continue
        for k, v in row.items():
            counts[k] += v
    effective = max(0, counts['input_tokens'] - counts['cached_input_tokens']) + int(counts['cached_input_tokens'] * 0.1)
    text = '\n'.join(texts)
    action, mode = parse_json_object(text) if expect_json else (None, 'not_parsed')
    error = None
    if status != 'RETURNED' or exit_code != 0:
        error = 'CLI failed or incomplete result'
    elif errors:
        error = errors[0]
    elif not usage_ok:
        error = 'missing usage accounting'
    elif expect_json and action is None:
        error = 'invalid structured JSON action'
    elif schema is not None and action is not None:
        problems = validate(action, schema)
        if problems:
            error = 'schema: ' + '; '.join(problems[:6])
    fields = {'result_text': text, 'parse_mode': mode, 'validation_error': error,
              'llm_tokens': effective + counts['output_tokens'] + counts['reasoning_output_tokens'] if usage_ok else None,
              'llm_usd': 0.0, 'accounting_complete': usage_ok,
              'token_breakdown': {**counts, 'effective_input_tokens': effective}}
    return (None if error else action), fields


class CodexCLIBackend:
    kind = 'codex_cli'
    call_overhead_tokens = 4000

    def __init__(self, *, model: str = 'default', effort: str = 'high', executable: str | None = None,
                 max_output_tokens: int = 16000, timeout_seconds: float = 900,
                 out_root: str | Path | None = None, max_retries: int = 1):
        self.executable = _find_executable(executable)
        self.model, self.effort = model, effort
        self.configured_model = _configured_model() if model in (None, '', 'default', 'codex') else None
        self.max_output_tokens, self.timeout_seconds = int(max_output_tokens), float(timeout_seconds)
        self.out_root = Path(out_root) if out_root else None
        self.max_retries = max(0, int(max_retries))

    def describe(self) -> dict:
        return {'kind': self.kind, 'model': self.model, 'effort': self.effort, 'max_retries': self.max_retries,
                **({'configured_model': self.configured_model} if self.configured_model else {})}

    def complete(self, system: str, prompt: str, *, schema: dict | None = None, call_dir=None, role: str = 'proposer',
                 expect_json: bool = True, timeout: float | None = None, max_output_tokens: int | None = None,
                 token_allowance: int | None = None, retry_note=None, **_ignored) -> LLMResult:
        limit = min(self.timeout_seconds, timeout if timeout is not None else self.timeout_seconds)
        cap = min(self.max_output_tokens, max_output_tokens or self.max_output_tokens)
        root = Path(call_dir) if call_dir else (self.out_root or Path(tempfile.gettempdir()) / 'infrarsi_codex_calls') / f'{role}_{uuid.uuid4().hex[:12]}'
        attempts, note = [], ''
        for n in range(self.max_retries + 1):
            wall_left = limit - sum(r.wall_seconds for r in attempts)
            left = None if token_allowance is None else token_allowance - sum(r.llm_tokens for r in attempts)
            reserve = math.ceil(math.ceil(len((system + '\n\n' + prompt + note).encode('utf-8')) / 3.5) * 1.25)
            if wall_left <= 0 or (left is not None and left < reserve + self.call_overhead_tokens + min(cap, 256)):
                break
            room = cap if left is None else min(cap, left - reserve - self.call_overhead_tokens)   # (a retry fits what is left)
            directory = root if n == 0 else root / f'retry_{n}'
            result = self._once(system, prompt + note, schema, directory, expect_json, wall_left, room)
            attempts.append(result)
            if not expect_json or not result.error or not (result.error == 'invalid structured JSON action' or result.error.startswith('schema: ')):
                break
            note = (f'\n\nYour previous reply could not be used ({result.error[:300]}). '
                    'Reply again with only the JSON object the instructions ask for.')
            if callable(retry_note):
                note = retry_note(result) or note
        if not attempts:
            return LLMResult(None, None, 0, 0.0, 0.0, True, 'token or time allowance exhausted before this call; not sent',
                             self.model, [], meta={'status': 'NOT_SPAWNED'})
        last = attempts[-1]
        if len(attempts) == 1:
            return last
        breakdown = {k: sum(r.meta['token_breakdown'].get(k, 0) for r in attempts) for k in last.meta['token_breakdown']}
        return dataclasses.replace(last, llm_tokens=sum(r.llm_tokens for r in attempts),
                                   wall_seconds=sum(r.wall_seconds for r in attempts), attempts=len(attempts),
                                   accounting_complete=all(r.accounting_complete for r in attempts),
                                   meta={**last.meta, 'token_breakdown': breakdown,
                                         'attempts': [r.meta for r in attempts]})

    def _once(self, system, prompt, schema, directory, expect_json, timeout, cap):
        directory.mkdir(parents=True, exist_ok=False)
        argv = [str(self.executable), 'exec', '--ephemeral', '--skip-git-repo-check', '--ignore-user-config',
                '--ignore-rules', '--sandbox', 'read-only', '--json', '-c', f'model_reasoning_effort="{self.effort}"',
                *(['-c', 'model=' + json.dumps(self.configured_model)] if self.configured_model else []),
                *(['-m', self.model] if self.model and self.model not in ('default', 'codex') else []), '-']
        (directory / 'request.json').write_text(json.dumps({'system': system, 'prompt': prompt, 'schema': schema}), encoding='utf-8')
        (directory / 'command.json').write_text(json.dumps(argv), encoding='utf-8')
        started, status, exit_code, error = time.monotonic(), 'NOT_SPAWNED', None, None
        try:
            with tempfile.TemporaryDirectory(prefix='infrarsi_codex_empty_') as cwd:
                with (directory / 'stdout.jsonl').open('wb') as out, (directory / 'stderr.txt').open('wb') as err:
                    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=out, stderr=err, cwd=cwd)
                    status = 'STARTED'
                    try:
                        proc.communicate((system + '\n\n' + prompt).encode('utf-8'), timeout=timeout)
                        status = 'RETURNED'
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.communicate()
                        status = 'TIMEOUT'
                    exit_code = proc.returncode
        except (OSError, ValueError) as exc:
            error = f'{type(exc).__name__}: {exc}'
        wall = time.monotonic() - started
        stdout = directory / 'stdout.jsonl'
        raw = stdout.read_text(encoding='utf-8', errors='replace') if stdout.exists() else ''
        action, fields = parse_codex_stream(raw, exit_code=exit_code, status=status, expect_json=expect_json, schema=schema)
        if status == 'NOT_SPAWNED':
            fields.update(llm_tokens=0, accounting_complete=True)
        elif fields['llm_tokens'] is None:
            fields['llm_tokens'] = math.ceil(len((system + prompt).encode('utf-8')) / 3.5) + self.call_overhead_tokens + cap
            fields['usage_upper_bound'] = True
        error = error or fields['validation_error']
        meta = {**fields, 'status': status, 'exit_code': exit_code, 'call_dir': str(directory), 'error': error}
        (directory / 'receipt.json').write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
        return LLMResult(None if error else action, fields['result_text'], fields['llm_tokens'], 0.0, wall,
                         fields['accounting_complete'], error, self.model, [], meta=meta)
