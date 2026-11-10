"""Whole-codebase review fixes in the LLM backends and the RE-Bench arm (2026-10-24, C4 and C2 #7 #8): a paid API keeps
its table price and an unknown one is not free; Claude CLI answers are held to the schema; Anthropic fits its answer to
the token and USD allowances and reserves output at the output price; a named RE-Bench endpoint is used even with a paid key, and its meter reports the route's
costs; a scoring attempt is launched once."""
import json
import types

import pytest

from agent_helpers import FIXTURES
from infrarsi.agent.backends.base import LLMResult
from infrarsi.project.llm import make_driver

def test_a_paid_api_keeps_its_price_and_an_unknown_one_is_not_free():
    d = make_driver({'backend': 'openai-compat', 'model': 'deepseek-flash', 'base_url': 'https://api.deepseek.com',
                     'api_key_env': 'INFRARSI_NO_SUCH_KEY'})
    assert d.price and not d.unpriced
    u = make_driver({'backend': 'openai', 'model': 'gpt-x', 'api_key_env': 'INFRARSI_NO_SUCH_KEY'})
    assert u.price is None and u.unpriced and 'unknown' in u.describe()['pricing']
    u.transport = lambda url, headers, body, timeout: (200, {'model': 'gpt-x', 'usage': {'prompt_tokens': 10, 'completion_tokens': 5},
                                                             'choices': [{'message': {'content': '{"a": 1}'}}]})
    r = u.complete('s', 'p', schema={'type': 'object'})
    assert r.action == {'a': 1} and r.llm_usd == 0.0 and r.accounting_complete is False   # its USD are unknown
    v = make_driver({'backend': 'vllm', 'model': 'local'})
    assert v.price is None and not v.unpriced


def test_claude_cli_holds_the_answer_to_the_schema(tmp_path, monkeypatch):
    import infrarsi.agent.backends.claude_cli as cc
    exe = tmp_path / 'claude.exe'
    exe.write_bytes(b'x')
    monkeypatch.setattr(cc, '_find_executable', lambda explicit: exe)
    monkeypatch.setattr(cc.subprocess, 'run', lambda *a, **k: type('R', (), {'stdout': 'claude 2.1\n'})())
    stream = (FIXTURES / 'stream_haiku_5_5_real.jsonl').read_bytes()

    class FakePopen:
        def __init__(self, argv, stdin=None, stdout=None, stderr=None, cwd=None, env=None):
            stdout.write(stream)
            self.pid, self.returncode = 4242, 0

        def communicate(self, data=None, timeout=None):
            return None, None
    monkeypatch.setattr(cc.subprocess, 'Popen', FakePopen)
    b = cc.ClaudeCLIBackend(model='claude-haiku-5-5')
    ok = b.complete('s', 'p', schema={'type': 'object', 'required': ['action']}, call_dir=tmp_path / 'c1')
    assert ok.action and ok.error is None
    bad = b.complete('s', 'p', schema={'type': 'object', 'properties': {'base': {'type': 'integer'}}}, call_dir=tmp_path / 'c2')
    assert bad.action is None and bad.error.startswith('schema:') and bad.llm_tokens == 2862      # still charged


def test_anthropic_fits_the_answer_to_the_allowances():
    from infrarsi.agent.backends import AnthropicBackend
    calls = []

    def create(**kw):
        calls.append(kw)
        raise TimeoutError('the answer was lost')
    client = types.SimpleNamespace(beta=types.SimpleNamespace(messages=types.SimpleNamespace(create=create)),
                                   with_options=lambda **k: client)
    b = AnthropicBackend(client=client)                     # opus 5.5: input 4, output 20, cache write 5 USD per M
    inp = len('s' + 'p' * 300) // 3
    r = b.complete('s', 'p' * 300)
    assert r.llm_usd == pytest.approx((inp * 5.0 + 16000 * 20.0) / 1e6) and r.accounting_complete is False
    b.complete('s', 'p' * 300, max_usd=0.1)
    assert calls[-1]['max_tokens'] == int((0.1 * 1e6 - inp * 5.0) / 20.0)
    b.complete('s', 'p' * 300, token_allowance=1000)
    assert calls[-1]['max_tokens'] == 1000 - inp
    n = len(calls)
    r = b.complete('s', 'p' * 300, max_usd=0.001)
    assert len(calls) == n and r.error.startswith('token or USD allowance') and r.llm_usd == 0.0


def test_a_named_rebench_endpoint_is_used_even_with_a_paid_key(monkeypatch):
    from infrarsi.ext.rebench_llmf import arms as ra
    for k in ('INFRARSI_WEAK_URL', 'INFRARSI_WEAK_MODEL'):          # weak_driver sets them: restored afterwards
        monkeypatch.setenv(k, 'unset')
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'k')
    drv, route, meter = ra.weak_driver(base_url='http://127.0.0.1:18000/v1', served_model='local-model')
    assert route == 'selfhost' and meter.serving_gpus > 0
    drv, route, meter = ra.weak_driver()
    assert route == 'deepseek' and meter.serving_gpus == 0
    meter.observe(LLMResult(action={}, text=None, llm_tokens=10, llm_usd=0.25, wall_seconds=3600.0, accounting_complete=True,
                            error=None, model='m', observed_models=[]))
    s = meter.summary()
    assert s['llm_usd'] == 0.25 and s['llm_gpu_hours_h20'] == 0


def test_a_scoring_attempt_is_launched_once(monkeypatch):
    from infrarsi.ext.rebench_llmf import runner as rr
    monkeypatch.setattr(rr, 'ship_node_files', lambda ops: None)
    monkeypatch.setattr(rr, 'node_bound', lambda job: 100)
    seen = []

    class Ops:
        def mkdirs(self, paths, mode=None):
            pass

        def write_files(self, files, mode=None):
            pass

        def sh(self, cmd, timeout=None):
            seen.append(cmd)
            return {'stdout': '', 'stderr': '', 'code': 75}

        def read_json(self, path):
            return {'ok': True, 'result': {'score': 1.0}}
    assert rr.winh_scorer(Ops(), {'run_dir': '/r', 'attempt_id': 'a1', 'out': '/r/results/a1.json'}) == {'score': 1.0}
    assert seen[0].startswith('mkdir /r/results/a1.launched 2>/dev/null || exit 75; ')


# ---------------------------------------------------------------- round 2 (Astra CF_r1)
def test_a_retried_scoring_launch_waits_for_the_first_run(monkeypatch):
    from infrarsi.ext.rebench_llmf import runner as rr
    monkeypatch.setattr(rr, 'node_bound', lambda job: 100)
    answers = [None, None, {'ok': True, 'result': {'score': 2.0}}]

    class Ops:
        def mkdirs(self, paths, mode=None):
            pass

        def write_files(self, files, mode=None):
            pass

        def sh(self, cmd, timeout=None):
            return {'stdout': '', 'stderr': '', 'code': 75}       # the marker exists: the first launch still scores

        def read_json(self, path):
            return answers.pop(0)
    job = {'run_dir': '/r', 'attempt_id': 'a1', 'out': '/r/results/a1.json'}
    assert rr.launch_scorer(Ops(), job, wait_s=0) == {'score': 2.0}
