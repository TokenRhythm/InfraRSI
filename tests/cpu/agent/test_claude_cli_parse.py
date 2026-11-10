import json

import pytest

from infrarsi.agent.backends.claude_cli import (MAX_SYSTEM_ARG_UTF16_UNITS, build_argv, parse_stream,
                                                  system_arg_units, validate_system_size)
from agent_helpers import FIXTURES

HAIKU = (FIXTURES / 'stream_haiku_5_5_real.jsonl').read_text(encoding='utf-8')
FABLE = (FIXTURES / 'stream_fable_5_1_real_20261018.jsonl').read_text(encoding='utf-8')


def _events(raw):
    return [json.loads(x) for x in raw.splitlines() if x.strip()]


def _dump(events):
    return '\n'.join(json.dumps(e) for e in events) + '\n'


def test_family_aliases_resolve_to_the_ids_usage_is_reported_under(monkeypatch, tmp_path):
    import infrarsi.agent.backends.claude_cli as cc
    exe = tmp_path / 'claude.exe'
    exe.write_bytes(b'x')
    monkeypatch.setattr(cc, '_find_executable', lambda explicit: exe)
    monkeypatch.setattr(cc.subprocess, 'run', lambda *a, **k: type('R', (), {'stdout': 'claude 2.1\n'})())
    assert cc.ClaudeCLIBackend(model='opus').model == 'claude-opus-5-5'           # trial C3b: `claude-cli:opus`
    assert cc.ClaudeCLIBackend(model='Sonnet').model == 'claude-sonnet-5-5'
    assert cc.ClaudeCLIBackend(model='claude-haiku-5-5').model == 'claude-haiku-5-5'
    with pytest.raises(ValueError, match='one of opus, sonnet, haiku, fable'):
        cc.ClaudeCLIBackend(model='gpt-6')


def test_real_haiku_sample_parses_with_complete_accounting():
    action, f = parse_stream(HAIKU, model='claude-haiku-5-5', exit_code=0, status='RETURNED')
    assert f['validation_error'] is None and f['accounting_complete'] is True
    assert action['action'] == 'propose_config' and action['base'] == 'c0'
    assert f['observed_models'] == ['claude-haiku-5-5'] and f['init_tool_inventories'] == [[]]
    assert f['llm_tokens'] == 2862 and abs(f['llm_usd'] - 0.0006628) < 1e-12
    assert f['token_breakdown']['output_tokens'] == 302 and f['num_turns'] == 1
    recorded = json.loads((FIXTURES / 'stream_haiku_5_5_real.receipt.json').read_text())
    assert recorded['llm_tokens'] == f['llm_tokens'] and recorded['accounting_complete'] is True


def test_real_fable_sample_from_previous_harness():
    action, f = parse_stream(FABLE, model='claude-fable-5-1', exit_code=0, status='RETURNED')
    assert f['validation_error'] is None and action == {'action': 'run_analysis', 'tool': 'runtime:mean', 'data': [2, 4, 6]}
    assert f['llm_tokens'] == 2 + 3367 + 0 + 146 and f['llm_usd'] == 0.07466


def test_requested_model_must_match_exactly_but_known_usage_is_kept():
    action, f = parse_stream(HAIKU, model='claude-opus-5-5', exit_code=0, status='RETURNED')
    assert action is None and 'model mismatch' in f['validation_error']
    assert f['accounting_complete'] is False and f['llm_tokens'] == 2862      # still charged


def test_model_usage_with_an_extra_model_is_incomplete():
    ev = _events(HAIKU)
    ev[-1]['modelUsage']['claude-opus-5-5'] = dict(ev[-1]['modelUsage']['claude-haiku-5-5'])
    action, f = parse_stream(_dump(ev), model='claude-haiku-5-5', exit_code=0, status='RETURNED')
    assert action is None and f['accounting_complete'] is False


@pytest.mark.parametrize('mutate, expect', [
    (lambda ev: ev[:-1], 'CLI failed'),                                                    # no result event
    (lambda ev: ev + [ev[-1]], 'exactly one result'),                                      # duplicated result
    (lambda ev: [dict(ev[0], tools=['Bash'])] + ev[1:], 'tool inventory'),
    (lambda ev: ev[:-1] + [dict(ev[-1], num_turns=2)], 'turns'),
    (lambda ev: ev[:-1] + [dict(ev[-1], result='not json at all')], 'invalid structured JSON'),
    (lambda ev: ev[:-1] + [dict(ev[-1], is_error=True)], 'CLI failed'),
    (lambda ev: ev[:-1] + [dict(ev[-1], total_cost_usd=None)], 'missing usage accounting'),
])
def test_rejections(mutate, expect):
    action, f = parse_stream(_dump(mutate(_events(HAIKU))), model='claude-haiku-5-5', exit_code=0, status='RETURNED')
    assert action is None and expect in f['validation_error'], f['validation_error']


def test_tool_use_block_is_rejected():
    ev = _events(HAIKU)
    for e in ev:
        if e.get('type') == 'assistant':
            e['message']['content'].append({'type': 'tool_use', 'name': 'Bash', 'input': {}})
            break
    _, f = parse_stream(_dump(ev), model='claude-haiku-5-5', exit_code=0, status='RETURNED')
    assert f['validation_error'] == 'CLI tool inventory/use not empty or unsupported'


def test_fenced_json_and_nonfinite_values():
    ev = _events(HAIKU)
    ev[-1]['result'] = '```json\n{"action": "read_env"}\n```'
    action, f = parse_stream(_dump(ev), model='claude-haiku-5-5', exit_code=0, status='RETURNED')
    assert action == {'action': 'read_env'} and f['parse_mode'] == 'fenced'
    ev[-1]['result'] = '{"action": "finish", "x": 1e999}'
    action, f = parse_stream(_dump(ev), model='claude-haiku-5-5', exit_code=0, status='RETURNED')
    assert action is None
    ev[-1]['result'] = 'Four costs are code-only:\n1. eager attention\n\n```json\n{"verdict": "patch"}\n```\n'   # C3c (Opus)
    action, f = parse_stream(_dump(ev), model='claude-haiku-5-5', exit_code=0, status='RETURNED')
    assert action == {'verdict': 'patch'} and f['parse_mode'] == 'embedded'
    for two in ('a\n```json\n{"a": 1}\n```\nor\n```json\n{"b": 2}\n```',                # two blocks: no guess
                'Example only:\n```json\n{"tool": "submit"}\n```\nRun this instead:\n```python\nprint(1)\n```',  # QA15
                'Example only:\n```json\n{"tool": "submit"}\n```\nRun this instead:\n```python\n```'):          # QA16
        ev[-1]['result'] = two
        action, f = parse_stream(_dump(ev), model='claude-haiku-5-5', exit_code=0, status='RETURNED')
        assert action is None and f['parse_mode'] == 'failed'
    ev[-1]['result'] = 'The ```json``` block holds the action:\n```json\n{"tool": "submit"}\n```'     # QA16: inline span
    action, f = parse_stream(_dump(ev), model='claude-haiku-5-5', exit_code=0, status='RETURNED')
    assert action == {'tool': 'submit'} and f['parse_mode'] == 'embedded'


def test_timeout_and_garbage_lines():
    _, f = parse_stream(HAIKU, model='claude-haiku-5-5', exit_code=-9, status='TIMEOUT')
    assert f['validation_error'] == 'CLI failed or incomplete result' and f['llm_tokens'] == 2862
    _, f = parse_stream('garbage\n' + HAIKU, model='claude-haiku-5-5', exit_code=0, status='RETURNED')
    assert f['validation_error'] == 'unexpected stream event shape'


def test_argv_is_the_audited_headless_invocation():
    argv = build_argv('claude.exe', model='claude-opus-5-5', effort='high', session_id='s-1', max_budget_usd=2.0,
                      system='SYS', max_output_tokens=4096)
    joined = ' '.join(argv)
    for flag in ('-p', '--model claude-opus-5-5', '--output-format stream-json', '--verbose', '--safe-mode', '--strict-mcp-config',
                 '--permission-mode manual', '--no-session-persistence', '--disable-slash-commands', '--no-chrome',
                 '--session-id s-1', '--effort high'):
        assert flag in joined, flag
    assert argv[argv.index('--tools') + 1] == '' and argv[argv.index('--setting-sources') + 1] == ''
    assert argv[-2:] == ['--system-prompt', 'SYS']


def test_system_prompt_size_is_checked_never_truncated():
    validate_system_size('x' * 1000)
    with pytest.raises(ValueError):
        validate_system_size('x' * (MAX_SYSTEM_ARG_UTF16_UNITS + 10))
    assert system_arg_units('ab') < 30


def test_complete_charges_an_upper_bound_when_usage_is_unknown(tmp_path):
    """Offline: the stand-in executable fails (no stream-json); cost is an upper bound, never zero."""
    import sys
    from infrarsi.agent.backends.claude_cli import ClaudeCLIBackend
    b = ClaudeCLIBackend(executable=sys.executable, max_output_tokens=1000, max_usd_per_call=2.5, timeout_seconds=60)
    r = b.complete('system text', 'prompt ' * 100, call_dir=tmp_path / 'c1')
    assert r.action is None and r.error == 'CLI failed or incomplete result'
    assert r.accounting_complete is False and r.llm_usd == 2.5 and r.llm_tokens >= 1000 + 4000
    receipt = json.loads((tmp_path / 'c1' / 'receipt.json').read_text())
    assert receipt['usage_upper_bound'] is True and receipt['status'] == 'RETURNED'
    r = b.complete('x' * (MAX_SYSTEM_ARG_UTF16_UNITS + 100), 'p', call_dir=tmp_path / 'c2')
    assert r.meta['status'] == 'NOT_SPAWNED' and r.accounting_complete is True and r.llm_tokens == 0 and r.llm_usd == 0.0
