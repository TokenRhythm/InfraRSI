"""Claude Code CLI headless backend: one fresh process and session per call, tools disabled, empty cwd.

Argv, stream parsing and cost accounting follow rsi_optimizer_harness_v1/claude_transport.py and
rsi_gpu_backend_v3_d2_orderfix/review_cost.py. History is replayed explicitly in the prompt (stdin); the system
prompt goes through argv and is size-checked, never truncated. `accounting_complete` holds only when the
terminal result carries full usage, a finite cost, and modelUsage contains exactly the requested model.
"""
from __future__ import annotations

import contextlib
import dataclasses
import datetime as _dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from ..jsonschema_lite import validate
from .base import LLMResult, parse_json_object

DEFAULT_MODEL = 'claude-opus-5-5'
# the family names users write (`--llm claude-cli:opus`), resolved to the ids the CLI reports usage under
ALIASES = {'opus': 'claude-opus-5-5', 'sonnet': 'claude-sonnet-5-5', 'haiku': 'claude-haiku-5-5', 'fable': 'claude-fable-5-1'}
MAX_SYSTEM_ARG_UTF16_UNITS = 20000
MAX_ARGV_UTF16_UNITS = 30000
USAGE_KEYS = ('input_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens', 'output_tokens')
BUILTIN_AGENTS = {'claude', 'Explore', 'general-purpose', 'Plan'}
BUILTIN_PLUGINS = {'cc-plugin-agents-md', 'cc-plugin-telemetry', 'cc-plugin-plugin-authoring'}


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def system_arg_units(system: str) -> int:
    return len(subprocess.list2cmdline(['--system-prompt', system]).encode('utf-16-le')) // 2


def validate_system_size(system: str) -> None:
    n = system_arg_units(system)
    if n > MAX_SYSTEM_ARG_UTF16_UNITS:
        raise ValueError(f'system prompt is {n} UTF-16 units in argv (limit {MAX_SYSTEM_ARG_UTF16_UNITS}); never truncated')


def build_argv(executable: str, *, model: str, effort: str, session_id: str, max_budget_usd: float, system: str,
               max_output_tokens: int) -> list[str]:
    settings = {'env': {'CLAUDE_CODE_EFFORT_LEVEL': effort, 'CLAUDE_CODE_MAX_OUTPUT_TOKENS': str(max_output_tokens)}}
    return [str(executable), '-p', '--model', model, '--effort', effort, '--output-format', 'stream-json', '--verbose',
            '--safe-mode', '--tools', '', '--strict-mcp-config', '--setting-sources', '', '--settings', _canonical(settings),
            '--permission-mode', 'manual', '--permission-prompts', 'none', '--no-session-persistence',
            '--disable-slash-commands', '--no-chrome', '--session-id', session_id,
            '--max-budget-usd', f'{max_budget_usd:.4f}', '--system-prompt', system]


def assistant_text(events: list) -> str:
    """Every text block the assistant wrote in a call, in order (F14a: a reply the CLI split over its own resume
    turns, of which the result event holds only the last block)."""
    out = []
    for e in events:
        m = e.get('message') if e.get('type') == 'assistant' and isinstance(e.get('message'), dict) else {}
        blocks = m.get('content')
        for b in blocks if isinstance(blocks, list) else []:
            if isinstance(b, dict) and b.get('type') == 'text' and isinstance(b.get('text'), str):
                out.append(b['text'])
    return ''.join(out)


def parse_stream(raw: str, *, model: str, exit_code, status: str, expect_json: bool = True, join_text: bool = False,
                 strict_audit: bool = True):
    """Offline-testable parser of one stream-json call. Returns (action | None, fields).

    Known usage is preserved even when the call is otherwise invalid, so failed calls are still charged. `join_text`
    (F14a; off by default): a result that is not the JSON object is read again from every text block the assistant
    wrote, joined (parse_mode 'joined').
    """
    events, shape_error = [], False
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            _canonical(event)               # rejects NaN/inf, including 1e999
        except (ValueError, RecursionError):
            shape_error = True
            continue
        if type(event) is not dict:
            shape_error = True
            continue
        events.append(event)
    models, uses, inventories, resumes = set(), [], [], 0
    for event in events:
        message = event.get('message', {})
        if type(message) is not dict:
            shape_error, message = True, {}
        actual = event.get('model') or message.get('model')
        if actual:
            if type(actual) is str:
                models.add(actual)
            else:
                shape_error = True
        blocks = message.get('content', [])
        if type(blocks) is not list:
            shape_error, blocks = True, []
        for block in blocks:
            if type(block) is not dict:
                shape_error = True
            elif block.get('type') in ('tool_use', 'server_tool_use'):
                uses.append(block.get('name'))
        if event.get('type') == 'user' and event.get('isSynthetic') is True and blocks and all(
                type(b) is dict and b.get('type') == 'text' for b in blocks):
            resumes += 1        # the CLI's own "Output token limit hit. Resume directly" turn (D1-15), not a tool loop
        if event.get('type') == 'system' and event.get('subtype') == 'init':
            inventories.append({k: event.get(k) for k in ('tools', 'mcp_servers', 'skills', 'slash_commands', 'agents', 'plugins')})
    results = [e for e in events if e.get('type') == 'result']
    result = results[-1] if results else {}
    fields = {'observed_models': sorted(models), 'tool_uses': uses, 'init_tool_inventories': [v['tools'] for v in inventories],
              'accounting_complete': False, 'llm_tokens': None, 'llm_usd': None, 'result_events': len(results),
              'resumed_turns': resumes}
    for key in ('is_error', 'subtype', 'modelUsage', 'usage', 'num_turns', 'total_cost_usd', 'duration_ms',
                'duration_api_ms', 'permission_denials', 'terminal_reason', 'stop_reason', 'api_error'):
        if key in result:
            fields[key] = result[key]
    usage = result.get('usage', {})
    if type(usage) is not dict:
        shape_error, usage = True, {}
    model_usage = result.get('modelUsage', {})
    if type(model_usage) is not dict:
        shape_error, model_usage = True, {}
    cost = result.get('total_cost_usd')
    usage_ok = all(type(usage.get(k)) is int and usage[k] >= 0 for k in USAGE_KEYS)
    cost_ok = type(cost) in (int, float) and math.isfinite(cost) and cost >= 0
    if usage_ok:
        fields['llm_tokens'] = sum(usage[k] for k in USAGE_KEYS)
        fields['token_breakdown'] = {k: usage[k] for k in USAGE_KEYS}
    if cost_ok:
        fields['llm_usd'] = float(cost)
    elif not strict_audit and cost is None:
        fields['llm_usd'], cost_ok = 0.0, True  # subscription login has usage but no per-call bill
    fields['accounting_complete'] = bool(usage_ok and cost_ok and set(model_usage) == {model} and len(results) == 1)
    if not strict_audit:
        fields['accounting_complete'] = bool(usage_ok and cost_ok)
    configured = False
    for inv in inventories:
        configured |= any(inv[k] not in (None, []) for k in ('mcp_servers', 'skills', 'slash_commands'))
        agents, plugins = inv['agents'] or [], inv['plugins'] or []
        if type(agents) is not list or any(a not in BUILTIN_AGENTS for a in agents):
            configured = True
        if type(plugins) is not list or any(type(p) is not dict or p.get('path') != 'builtin' or p.get('name') not in BUILTIN_PLUGINS for p in plugins):
            configured = True
    server_uses = usage.get('server_tool_use', {})
    server_bad = type(server_uses) is not dict or any(type(v) is not int or v != 0 for v in server_uses.values())
    model_tool_bad = any(type(v) is not dict or v.get('webSearchRequests', 0) != 0 for v in model_usage.values())
    error = None
    if status != 'RETURNED' or exit_code != 0 or result.get('is_error') is not False:
        error = 'CLI failed or incomplete result'
        if result.get('api_error') and type(result.get('result')) is str:   # the CLI's own words: "... exceeded the
            error += f': {result["result"][:200]}'                          # 8000 output token maximum ...
    elif shape_error:
        error = 'unexpected stream event shape'
    elif len(results) != 1:
        error = 'expected exactly one result event'
    elif strict_audit and (sorted(models) != [model] or set(model_usage) != {model}):
        error = f'actual model mismatch: observed {sorted(models)}, modelUsage {sorted(model_usage)}'
    elif (strict_audit and (fields['init_tool_inventories'] != [[]] or configured)) or uses or server_bad or model_tool_bad:
        error = 'CLI tool inventory/use not empty or unsupported'
    elif result.get('num_turns') != 1 + resumes or result.get('permission_denials') not in (None, []):
        error = 'unexpected turns or permission denials'
    elif not fields['accounting_complete']:
        error = 'missing usage accounting'
    text = result.get('result') if type(result.get('result')) is str else None
    fields['result_text'] = text
    action, mode = None, 'not_parsed'
    if error is None and expect_json:
        action, mode = parse_json_object(text)
        if action is None and join_text:
            joined = assistant_text(events)
            if joined and joined != text:
                action, _ = parse_json_object(joined)
                if action is not None:
                    mode, fields['result_text'] = 'joined', joined
        if action is None:
            error = 'invalid structured JSON action'
    fields['parse_mode'] = mode
    fields['validation_error'] = error
    return action, fields


@contextlib.contextmanager
def _empty_dir():
    d = tempfile.mkdtemp(prefix='infrarsi_cli_empty_')
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _correctable(error: str | None) -> bool:
    """A reply the model can fix when asked again: not the JSON object, or not the schema's (not a transport,
    accounting or model-identity failure)."""
    return bool(error) and (error == 'invalid structured JSON action' or error.startswith('schema: '))


def _find_executable(explicit: str | None, *, strict_audit: bool = True) -> Path:
    # A pinned private copy (INFRARSI_CLAUDE_EXE) keeps the transport fixed for a whole run: the user's own Claude Code
    # auto-updates ~/.local/bin/claude.exe, which made every later call of a running arm fail (external bench #31).
    explicit = explicit or os.environ.get('INFRARSI_CLAUDE_EXE') or None
    if explicit:
        return Path(explicit).resolve(strict=True)
    home = Path.home() / '.local' / 'bin' / ('claude.exe' if sys.platform == 'win32' else 'claude')
    if home.exists():
        return home.resolve()
    found = (shutil.which('claude.exe') if sys.platform == 'win32' else None) or shutil.which('claude')
    if not found:
        raise FileNotFoundError('claude CLI not found')
    if sys.platform == 'win32' and Path(found).suffix.lower() != '.exe' and strict_audit:
        raise ValueError('Windows transport requires the native claude.exe, never a batch shim')
    return Path(found).resolve()


class ClaudeCLIBackend:
    kind = 'claude_cli'

    def __init__(self, *, model: str = DEFAULT_MODEL, effort: str = 'high', executable: str | None = None,
                 max_output_tokens: int = 32000, timeout_seconds: float = 900, max_usd_per_call: float = 5.0,
                 out_root: str | Path | None = None, max_retries: int = 0, strict_audit: bool = True):
        self.strict_audit = strict_audit
        self.executable = _find_executable(executable) if strict_audit else _find_executable(executable, strict_audit=False)
        allowed = ('.exe',) if strict_audit else ('.exe', '.cmd', '.bat')
        if sys.platform == 'win32' and self.executable.suffix.lower() not in allowed:
            raise ValueError('Windows transport requires the native claude.exe, never a batch shim')
        model = ALIASES.get(str(model).lower(), model) if strict_audit else str(model).lower()
        if model != DEFAULT_MODEL and not model.startswith('claude-') and (strict_audit or model not in ALIASES):
            raise ValueError(f'unsupported model {model!r}: a claude-* model id or one of {", ".join(ALIASES)}')
        self.model, self.effort = model, effort
        self.max_output_tokens, self.timeout_seconds, self.max_usd_per_call = max_output_tokens, timeout_seconds, max_usd_per_call
        self.version = subprocess.run([str(self.executable), '--version'], capture_output=True, text=True, timeout=60).stdout.strip()
        self.executable_sha256 = hashlib.sha256(self.executable.read_bytes()).hexdigest()
        self.out_root = Path(out_root) if out_root else None
        self.max_retries = int(max_retries)             # re-asks of a reply that is not the JSON asked for (0: one call)

    def describe(self) -> dict:
        return {'kind': self.kind, 'model': self.model, 'effort': self.effort, 'cli_version': self.version,
                'executable_sha256': self.executable_sha256, **({'max_retries': self.max_retries} if self.max_retries else {})}

    call_overhead_tokens = 4000          # CLI context around the system prompt (cache creation/read), reserved per call

    def complete(self, system: str, prompt: str, *, schema: dict | None = None, call_dir=None, role: str = 'proposer',
                 expect_json: bool = True, max_usd: float | None = None, timeout: float | None = None,
                 max_output_tokens: int | None = None, token_allowance: int | None = None, join_text: bool = False,
                 retry_note=None, **_ignored) -> LLMResult:
        """One call: a fresh process and session. With max_retries, a reply that is not the JSON object asked for
        (E-C-IS2b: the incumbent's one hypothesis call) is asked again in a fresh session, the error named, within what
        the call's tokens, USD and time have left; every attempt is charged, and the result is the last attempt's with
        the summed usage (its directory: call_dir/retry_<k>). F14 (off by default): `join_text` reads a reply split
        over resume turns from all its text blocks; `retry_note(result)` writes the retry's note (the exact errors)."""
        out_cap = int(min(self.max_output_tokens, max_output_tokens or self.max_output_tokens))
        if max_output_tokens and token_allowance is not None:
            # the CLI's limit holds the thinking and the answer (B-IS2 W1: Opus at effort high thought through the 8,000
            # tokens of the hypothesis answer, four turns in a row, in 4 of 6 calls): the room up to the driver's cap
            # that the tokens left allow, never less than the answer the caller admitted; the input reserved as the
            # admission reserves it (agent/optimize.input_reserve: the estimate plus 25 %)
            worst_in = math.ceil(math.ceil(len((system + prompt).encode('utf-8')) / 3.5) * 1.25)
            room = int(token_allowance) - worst_in - self.call_overhead_tokens
            out_cap = max(out_cap, min(int(self.max_output_tokens), room))
        budget = min(self.max_usd_per_call, max_usd if max_usd is not None else self.max_usd_per_call)
        timeout = min(self.timeout_seconds, timeout or self.timeout_seconds)
        call_id = uuid.uuid4().hex
        if call_dir is None:
            base = self.out_root or Path(tempfile.gettempdir()) / 'infrarsi_cli_calls'
            call_dir = base / f'{role}_{call_id[:12]}'
        tries = [self._once(system, prompt, schema, Path(call_dir), role, expect_json, budget, timeout, out_cap, call_id,
                            join_text=join_text)]
        while expect_json and len(tries) <= self.max_retries and _correctable(tries[-1].error):
            usd, wall = budget - sum(t.llm_usd for t in tries), timeout - sum(t.wall_seconds for t in tries)
            if usd <= 0.01 or wall <= 30:
                break
            note = (f'\n\nYour previous reply could not be used ({tries[-1].error[:300]}). Reply again with only the JSON '
                    'object the instructions ask for.')
            if callable(retry_note):
                note = retry_note(tries[-1]) or note
            cap = out_cap
            if token_allowance is not None:            # as the first attempt was admitted: input, context, answer
                worst_in = math.ceil(math.ceil(len((system + prompt + note).encode('utf-8')) / 3.5) * 1.25)
                cap = min(cap, int(token_allowance) - sum(t.llm_tokens for t in tries) - worst_in - self.call_overhead_tokens)
                if cap < 256:
                    break
            tries.append(self._once(system, prompt + note, schema, Path(call_dir) / f'retry_{len(tries)}', role,
                                    expect_json, usd, wall, cap, uuid.uuid4().hex, join_text=join_text))
        if len(tries) == 1:
            return tries[0]
        last = tries[-1]
        return dataclasses.replace(last, llm_tokens=sum(t.llm_tokens for t in tries), llm_usd=sum(t.llm_usd for t in tries),
                                   wall_seconds=sum(t.wall_seconds for t in tries), attempts=len(tries),
                                   accounting_complete=all(t.accounting_complete for t in tries),
                                   meta={**last.meta, 'retried_call_ids': [t.meta.get('call_id') for t in tries[:-1]]})

    def _once(self, system, prompt, schema, directory: Path, role, expect_json, budget, timeout, out_cap,
              call_id, join_text: bool = False) -> LLMResult:
        session_id = str(uuid.uuid4())
        directory.mkdir(parents=True, exist_ok=False)
        meta = {'schema': 'infrarsi-cli-call-v1', 'call_id': call_id, 'session_id': session_id, 'role': role,
                'requested_model': self.model, 'effort': self.effort, 'cli_version': self.version,
                'executable_sha256': self.executable_sha256, 'start_utc': _utc(), 'status': 'STARTED',
                'request_sha256': hashlib.sha256((system + '\x00' + prompt).encode('utf-8')).hexdigest(),
                'max_budget_usd': budget, 'timeout_seconds': timeout, 'accounting_complete': False}
        (directory / 'request.json').write_text(_canonical({'system': system, 'prompt': prompt, 'schema': schema}), encoding='utf-8')
        started = time.monotonic()
        error, raw, exit_code = None, '', None
        try:
            validate_system_size(system)
            if hashlib.sha256(self.executable.read_bytes()).hexdigest() != self.executable_sha256:
                raise ValueError('CLI executable changed after initialization')
            with _empty_dir() as cwd:
                argv = build_argv(self.executable, model=self.model, effort=self.effort, session_id=session_id,
                                  max_budget_usd=budget, system=system, max_output_tokens=out_cap)
                if len(subprocess.list2cmdline(argv).encode('utf-16-le')) // 2 >= MAX_ARGV_UTF16_UNITS:
                    raise ValueError('complete argv budget exceeded')
                (directory / 'command.json').write_text(_canonical(argv[:-1] + ['<system prompt>']), encoding='utf-8')
                with (directory / 'stdout.jsonl').open('wb') as out, (directory / 'stderr.txt').open('wb') as err:
                    env = os.environ.copy()
                    env['DISABLE_AUTOUPDATER'] = '1'        # never let a call update the executable it runs from
                    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=out, stderr=err, cwd=cwd, env=env)
                    meta['pid'] = proc.pid
                    try:
                        proc.communicate(prompt.encode('utf-8'), timeout=timeout)
                        meta['status'] = 'RETURNED'
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.communicate()
                        meta['status'] = 'TIMEOUT'
                    exit_code = proc.returncode
        except (OSError, ValueError) as exc:
            meta['status'] = meta.get('status') if meta.get('pid') else 'NOT_SPAWNED'
            error = f'{type(exc).__name__}: {exc}'
        wall = time.monotonic() - started
        if (directory / 'stdout.jsonl').exists():
            raw = (directory / 'stdout.jsonl').read_text(encoding='utf-8', errors='replace')
        action, fields = parse_stream(raw, model=self.model, exit_code=exit_code, status=meta['status'], expect_json=expect_json,
                                      join_text=join_text, strict_audit=self.strict_audit)
        if meta['status'] == 'NOT_SPAWNED':
            # Nothing ran: zero cost is known, not assumed.
            fields.update(accounting_complete=True, llm_tokens=0, llm_usd=0.0)
        elif fields['llm_tokens'] is None or fields['llm_usd'] is None:
            # Unknown usage (timeout, crash): charge an upper bound, never zero; accounting stays incomplete.
            bound_tokens = math.ceil(len((system + prompt).encode('utf-8')) / 3.5) + 4000 + out_cap
            fields['llm_tokens'] = fields['llm_tokens'] if fields['llm_tokens'] is not None else bound_tokens
            fields['llm_usd'] = fields['llm_usd'] if fields['llm_usd'] is not None else float(budget)
            fields['usage_upper_bound'] = True
        meta.update({k: v for k, v in fields.items() if k != 'result_text'}, exit_code=exit_code, wall_seconds=wall, end_utc=_utc())
        error = error or fields['validation_error']
        if not error and schema and isinstance(action, dict):   # the reply as the action schema asks (types too)
            problems = validate(action, schema)
            error = 'schema: ' + '; '.join(problems[:6]) if problems else None
        meta['error'] = error
        (directory / 'receipt.json').write_text(_canonical(meta), encoding='utf-8')
        if fields.get('result_text') is not None:
            (directory / 'result.txt').write_text(fields['result_text'], encoding='utf-8')
        return LLMResult(action=None if error else action, text=fields.get('result_text'), llm_tokens=fields['llm_tokens'],
                         llm_usd=fields['llm_usd'], wall_seconds=wall, accounting_complete=fields['accounting_complete'],
                         error=error, model=self.model, observed_models=fields['observed_models'],
                         meta={'call_id': call_id, 'session_id': session_id, 'call_dir': str(directory),
                               'parse_mode': fields['parse_mode'], 'status': meta['status'],
                               'token_breakdown': fields.get('token_breakdown'),
                               'stop_reason': fields.get('stop_reason'),            # 'refusal': base.is_refusal
                               'terminal_reason': fields.get('terminal_reason')})
