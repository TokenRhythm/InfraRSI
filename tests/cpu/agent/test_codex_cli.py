"""Codex CLI transport without network access or a local account."""
import json
import subprocess
import sys

import pytest

from infrarsi.agent.backends import CodexCLIBackend, make_backend, parse_codex_stream
from infrarsi.agent.backends import codex_cli


@pytest.fixture(autouse=True)
def private_config(monkeypatch, tmp_path):
    monkeypatch.setenv('CODEX_HOME', str(tmp_path / 'config'))


def stream(text='{"ok": true}', usage=None):
    return '\n'.join(json.dumps(e) for e in [
        {'type': 'thread.started', 'thread_id': 'test'},
        {'type': 'item.completed', 'item': {'type': 'reasoning', 'text': 'not the answer'}},
        {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': text}},
        {'type': 'turn.completed', 'usage': usage if usage is not None else {
            'input_tokens': 10000, 'cached_input_tokens': 9000, 'output_tokens': 100, 'reasoning_output_tokens': 50}},
    ])


def test_stream_discounts_cached_input_and_keeps_raw_counters():
    action, fields = parse_codex_stream(stream())
    assert action == {'ok': True} and fields['validation_error'] is None
    assert fields['llm_tokens'] == 2050 and fields['llm_usd'] == 0.0 and fields['accounting_complete']
    assert fields['token_breakdown'] == {'input_tokens': 10000, 'cached_input_tokens': 9000, 'output_tokens': 100,
                                       'reasoning_output_tokens': 50, 'effective_input_tokens': 1900}


def test_cached_input_never_makes_effective_input_negative():
    action, fields = parse_codex_stream(stream(usage={'input_tokens': 10, 'cached_input_tokens': 20, 'output_tokens': 3}))
    assert action == {'ok': True} and fields['llm_tokens'] == 5


def test_multiple_messages_optional_usage_and_schema_validation():
    raw = '\n'.join(json.dumps({'type': 'item.completed', 'item': {'type': 'agent_message', 'text': t}})
                    for t in ('{"ok":', 'true}'))
    raw += '\n' + json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 20, 'output_tokens': 3}})
    assert parse_codex_stream(raw)[0] == {'ok': True}
    action, fields = parse_codex_stream(raw, schema={'type': 'object', 'required': ['missing']})
    assert action is None and fields['validation_error'].startswith('schema:') and fields['llm_tokens'] == 23


@pytest.mark.parametrize('raw,error', [
    (stream('not json'), 'invalid structured JSON action'),
    (stream(usage={}), 'missing usage accounting'),
    (stream() + '\n[]', 'unexpected stream event shape'),
    (stream() + '\n{"type":"turn.failed","error":{"message":"unavailable"}}', 'CLI failed:'),
])
def test_failed_calls_are_not_accepted(raw, error):
    action, fields = parse_codex_stream(raw)
    assert action is None and fields['validation_error'].startswith(error)


def fake_cli(monkeypatch, replies, seen, *, timeout=False):
    replies = iter(replies)

    class Process:
        def __init__(self, argv, **kwargs):
            self.out, self.returncode = kwargs['stdout'], 0
            self.killed = False
            seen.append({'argv': argv, 'cwd': kwargs['cwd']})

        def communicate(self, data=None, timeout=None):
            if data is not None:
                seen[-1]['stdin'] = data
            if timeout is not None and fail_timeout:
                raise subprocess.TimeoutExpired('codex', timeout)
            if not self.killed:
                self.out.write(next(replies).encode('utf-8'))

        def kill(self):
            self.killed, self.returncode = True, -1

    fail_timeout = timeout
    monkeypatch.setattr(codex_cli.subprocess, 'Popen', Process)


@pytest.mark.parametrize('model', ['default', 'codex', 'specific-model'])
def test_transport_stdin_flags_retries_and_total_usage(tmp_path, monkeypatch, model):
    seen = []
    fake_cli(monkeypatch, [stream('{"ok": "bad"}'), stream()], seen)
    backend = CodexCLIBackend(executable=sys.executable, model=model)
    res = backend.complete('system', 'prompt', call_dir=tmp_path / 'call', token_allowance=50000,
                           schema={'type': 'object', 'properties': {'ok': {'type': 'boolean'}}})
    assert res.action == {'ok': True} and res.attempts == 2 and res.llm_tokens == 4100
    assert res.meta['token_breakdown']['input_tokens'] == 20000
    assert seen[0]['stdin'] == b'system\n\nprompt' and b'schema:' in seen[1]['stdin']
    argv = seen[0]['argv']
    assert argv[1:] == ['exec', '--ephemeral', '--skip-git-repo-check', '--ignore-user-config', '--ignore-rules',
                       '--sandbox', 'read-only', '--json', '-c', 'model_reasoning_effort="high"',
                       *(['-m', model] if model == 'specific-model' else []), '-']
    assert (tmp_path / 'call/retry_1/receipt.json').is_file()


def test_no_retry_after_timeout_or_without_allowance(tmp_path, monkeypatch):
    seen = []
    fake_cli(monkeypatch, [], seen, timeout=True)
    backend = CodexCLIBackend(executable=sys.executable)
    r = backend.complete('system', 'prompt', call_dir=tmp_path / 'call')
    assert r.error and r.meta['status'] == 'TIMEOUT' and not r.accounting_complete
    assert r.llm_tokens > 0 and len(seen) == 1
    r = backend.complete('system', 'prompt', token_allowance=5)
    assert r.meta['status'] == 'NOT_SPAWNED' and r.llm_tokens == 0 and len(seen) == 1


def test_executable_override_and_factory(monkeypatch, tmp_path):
    shim = tmp_path / 'codex.cmd'
    shim.write_text('@echo off\n', encoding='utf-8')
    monkeypatch.setenv('INFRARSI_CODEX_EXE', str(shim))
    assert make_backend('codex').executable == shim
    assert CodexCLIBackend(executable=sys.executable).executable != shim


def test_default_model_is_preserved_without_loading_other_user_settings(tmp_path, monkeypatch):
    config = tmp_path / 'config'
    config.mkdir()
    (config / 'config.toml').write_text('model = "configured-model"\nmodel_reasoning_effort = "low"\n', encoding='utf-8')
    seen = []
    fake_cli(monkeypatch, [stream()], seen)
    backend = CodexCLIBackend(executable=sys.executable)
    assert backend.model == 'default' and backend.describe()['configured_model'] == 'configured-model'
    assert backend.complete('system', 'prompt').error is None
    assert 'model="configured-model"' in seen[0]['argv'] and '-m' not in seen[0]['argv']
    assert 'model_reasoning_effort="high"' in seen[0]['argv']
    assert CodexCLIBackend(executable=sys.executable, model='explicit').configured_model is None

