"""E-C-IS2b (2026-10-24): the incumbent's one hypothesis call through the CLI came back as text that is not the JSON
object asked for, and the search ended. With max_retries (InfraRSI's own claude-cli driver: 1) the backend asks
again once, in a fresh session with the error named; both calls are charged and the second answer is used. The
default stays one call (the native competitor arms build the backend themselves)."""
import json
import sys

import pytest

from agent_helpers import FIXTURES
from infrarsi.agent.backends import claude_cli

EVENTS = [json.loads(line) for line in (FIXTURES / 'stream_haiku_5_5_real.jsonl').read_text(encoding='utf-8').splitlines()
          if line.strip()]
USD = next(e for e in EVENTS if e.get('type') == 'result')['total_cost_usd']


def _stream(text):
    return '\n'.join(json.dumps(dict(e, result=text) if e.get('type') == 'result' else e) for e in EVENTS) + '\n'


def _cli(monkeypatch, replies, prompts):
    it = iter(replies)

    class FakePopen:
        def __init__(self, argv, stdin=None, stdout=None, stderr=None, cwd=None, env=None, **kw):
            self.pid, self.returncode, self._out = 4242, None, stdout

        def communicate(self, data=None, timeout=None):
            prompts.append(data.decode('utf-8'))
            self._out.write(_stream(next(it)).encode('utf-8'))
            self.returncode = 0
            return b'', b''
    monkeypatch.setattr(claude_cli.subprocess, 'Popen', FakePopen)


def test_a_reply_that_is_not_json_is_asked_again_once(tmp_path, monkeypatch):
    prompts = []
    b = claude_cli.ClaudeCLIBackend(executable=sys.executable, model='claude-haiku-5-5', timeout_seconds=60, max_retries=1)
    _cli(monkeypatch, ['Here are my thoughts first.', '{"ok": true}'], prompts)   # after __init__'s `--version` call
    r = b.complete('system', 'prompt', call_dir=tmp_path / 'c1', schema={'type': 'object'})
    assert r.error is None and r.action == {'ok': True} and r.attempts == 2
    assert prompts[0] == 'prompt' and 'could not be used (invalid structured JSON action)' in prompts[1]
    assert r.llm_usd == pytest.approx(2 * USD) and r.accounting_complete
    assert len(r.meta['retried_call_ids']) == 1 and r.meta['retried_call_ids'][0] != r.meta['call_id']
    first = json.loads((tmp_path / 'c1' / 'receipt.json').read_text(encoding='utf-8'))
    assert first['error'] == 'invalid structured JSON action'
    assert json.loads((tmp_path / 'c1' / 'retry_1' / 'receipt.json').read_text(encoding='utf-8'))['error'] is None
    assert b.describe()['max_retries'] == 1


def test_one_call_by_default_and_at_most_max_retries(tmp_path, monkeypatch):
    prompts = []
    b = claude_cli.ClaudeCLIBackend(executable=sys.executable, model='claude-haiku-5-5', timeout_seconds=60)
    b1 = claude_cli.ClaudeCLIBackend(executable=sys.executable, model='claude-haiku-5-5', timeout_seconds=60, max_retries=1)
    _cli(monkeypatch, ['not json'], prompts)
    r = b.complete('system', 'prompt', call_dir=tmp_path / 'c1')
    assert r.error == 'invalid structured JSON action' and r.attempts == 1 and len(prompts) == 1
    assert 'max_retries' not in b.describe()
    prompts = []
    _cli(monkeypatch, ['no', 'still no'], prompts)
    r = b1.complete('system', 'prompt', call_dir=tmp_path / 'c2')
    assert r.error == 'invalid structured JSON action' and r.attempts == 2 and len(prompts) == 2
    assert r.llm_usd == pytest.approx(2 * USD)


def test_no_retry_without_budget_left_or_for_a_failure_the_model_cannot_fix(tmp_path, monkeypatch):
    prompts = []
    b = claude_cli.ClaudeCLIBackend(executable=sys.executable, model='claude-haiku-5-5', timeout_seconds=60, max_retries=1)
    _cli(monkeypatch, ['not json', '{"ok": true}'], prompts)
    r = b.complete('system', 'prompt', call_dir=tmp_path / 'c1', max_usd=USD + 0.005)     # nothing left after the first
    assert r.attempts == 1 and len(prompts) == 1
    assert not claude_cli._correctable('CLI failed or incomplete result')
    assert not claude_cli._correctable('actual model mismatch: observed [], modelUsage []')
    assert claude_cli._correctable('schema: missing hypotheses')



def _cap(call_dir):
    argv = json.loads((call_dir / 'command.json').read_text(encoding='utf-8'))
    return int(json.loads(argv[argv.index('--settings') + 1])['env']['CLAUDE_CODE_MAX_OUTPUT_TOKENS'])


def test_a_retry_fits_the_callers_token_allowance(tmp_path, monkeypatch):
    # DF7_r1 #1: the re-ask's input, the CLI's context overhead and its answer must fit the tokens the search has left
    prompts = []
    b0 = claude_cli.ClaudeCLIBackend(executable=sys.executable, model='claude-haiku-5-5', timeout_seconds=60)
    b = claude_cli.ClaudeCLIBackend(executable=sys.executable, model='claude-haiku-5-5', timeout_seconds=60, max_retries=1)
    _cli(monkeypatch, ['not json'], prompts)
    one = b0.complete('system', 'prompt', call_dir=tmp_path / 'c0').llm_tokens           # one attempt's tokens
    room = one + b.call_overhead_tokens
    _cli(monkeypatch, ['not json', '{"ok": true}'], prompts)
    r = b.complete('system', 'prompt', call_dir=tmp_path / 'c1', token_allowance=room + 100)
    assert r.attempts == 1 and r.error == 'invalid structured JSON action'              # no room left: no retry
    _cli(monkeypatch, ['not json', '{"ok": true}'], prompts)
    r = b.complete('system', 'prompt', call_dir=tmp_path / 'c2', token_allowance=room + 5000)
    assert r.attempts == 2 and r.action == {'ok': True}
    assert _cap(tmp_path / 'c2') == b.max_output_tokens and 4900 < _cap(tmp_path / 'c2' / 'retry_1') < 5000


def test_the_cli_limit_leaves_room_for_thinking_within_the_allowance(tmp_path, monkeypatch):
    # B-IS2 W1 (10-25): the hypothesis call's 8,000-token answer cap was the CLI's whole limit, and Opus at effort high
    # thought through it four turns in a row; with a token allowance the limit grows to the driver's cap within it
    prompts = []
    b = claude_cli.ClaudeCLIBackend(executable=sys.executable, model='claude-haiku-5-5', timeout_seconds=60)
    _cli(monkeypatch, ['{"ok": true}'] * 5, prompts)
    b.complete('system', 'prompt', call_dir=tmp_path / 'c1', max_output_tokens=8000, token_allowance=10 ** 6)
    assert _cap(tmp_path / 'c1') == b.max_output_tokens
    b.complete('system', 'prompt', call_dir=tmp_path / 'c2', max_output_tokens=8000,
               token_allowance=20000 + b.call_overhead_tokens + 3)
    assert 19990 < _cap(tmp_path / 'c2') <= 20000                       # the tokens left bound the room
    b.complete('system', 'prompt', call_dir=tmp_path / 'c3', max_output_tokens=8000, token_allowance=9000)
    assert _cap(tmp_path / 'c3') == 8000                                # never below the admitted answer
    b.complete('system', 'prompt', call_dir=tmp_path / 'c4', max_output_tokens=8000)
    assert _cap(tmp_path / 'c4') == 8000                                # no allowance given: the caller's cap
    # DF13_r1: near the budget the room keeps the admission's input reserve (estimate + 25 %), as optimize admits it
    from infrarsi.agent.optimize import input_reserve
    prompt = 'x' * (28000 - len('system'))
    b.complete('system', prompt, call_dir=tmp_path / 'c5', max_output_tokens=8000, token_allowance=30000)
    assert _cap(tmp_path / 'c5') == 16000
    assert input_reserve('system' + prompt) + b.call_overhead_tokens + _cap(tmp_path / 'c5') <= 30000


def test_a_reask_keeps_the_input_reserve(tmp_path, monkeypatch):
    # DF13_r2: the re-ask's cap - the grown limit bounded by what the tokens left allow - keeps the admission's input
    # reserve too, so both attempts together fit the allowance
    from infrarsi.agent.optimize import input_reserve
    prompts = []
    b = claude_cli.ClaudeCLIBackend(executable=sys.executable, model='claude-haiku-5-5', timeout_seconds=60, max_retries=1)
    _cli(monkeypatch, ['not json', '{"ok": true}'], prompts)
    prompt = 'x' * (28000 - len('system'))
    r = b.complete('system', prompt, call_dir=tmp_path / 'c1', max_output_tokens=8000, token_allowance=40000)
    assert r.attempts == 2 and _cap(tmp_path / 'c1') == 26000
    first = json.loads((tmp_path / 'c1' / 'receipt.json').read_text(encoding='utf-8'))
    again = json.loads((tmp_path / 'c1' / 'retry_1' / 'request.json').read_text(encoding='utf-8'))
    room = input_reserve('system' + again['prompt']) + b.call_overhead_tokens + _cap(tmp_path / 'c1' / 'retry_1')
    assert first['llm_tokens'] + room <= 40000


def test_a_cut_reply_says_so():
    events = [dict(e, is_error=True, api_error='max_output_tokens',
                   result="API Error: Claude's response exceeded the 8000 output token maximum.")
              if e.get('type') == 'result' else e for e in EVENTS]
    raw = '\n'.join(json.dumps(e) for e in events)
    _, fields = claude_cli.parse_stream(raw, model='claude-haiku-5-5', exit_code=0, status='RETURNED')
    assert fields['validation_error'].startswith('CLI failed or incomplete result: API Error: Claude')
    assert '8000 output token maximum' in fields['validation_error'] and fields['api_error'] == 'max_output_tokens'
