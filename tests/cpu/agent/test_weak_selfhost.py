"""Weak-model route R2 (self-hosted DeepSeek-V4-Flash on WIN0 through the tunnel): request settings, fail-fast on an
unreachable endpoint, context fitting, and serving GPU-time accounting. No network."""
import http.client
import json
import urllib.error

import pytest

from infrarsi.agent import arms
from infrarsi.agent.backends.base import LLMResult
from infrarsi.agent.backends.openai_compat import OpenAICompatBackend, classify_transport_error
from infrarsi.agent.budget import Budget
from infrarsi.agent.judge import ReceiptJudge
from infrarsi.agent.runner import FakeScreenRunner, synthetic_receipt
from infrarsi.agent.tools import action_schema
from agent_helpers import DENSE_8B, FR_DENSE

WEAK = action_schema('weak')
MODEL = 'deepseek-ai/DeepSeek-V4-Flash'


def _selfhost(transport, **kw):
    route, b = arms.weak_driver()
    assert route == 'selfhost'
    b.transport = transport
    for k, v in kw.items():
        setattr(b, k, v)
    return b


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    monkeypatch.delenv('INFRARSI_WEAK_MODEL', raising=False)
    monkeypatch.delenv('INFRARSI_WEAK_URL', raising=False)
    monkeypatch.setattr(arms, 'WEAK_STATUS', tmp_path / 'missing_STATUS.json')        # verified defaults


def test_request_settings_match_the_verified_endpoint():
    seen = []

    def transport(url, headers, body, timeout):
        seen.append((url, headers, body))
        return 200, {'model': MODEL, 'usage': {'prompt_tokens': 900, 'completion_tokens': 40},
                     'choices': [{'message': {'content': '{"action": "finish"}'}}]}
    b = _selfhost(transport)
    r = b.complete('sys', 'prompt', schema=WEAK)
    url, headers, body = seen[0]
    assert url == 'http://127.0.0.1:18000/v1/chat/completions' and 'Authorization' not in headers
    assert body['model'] == MODEL and body['temperature'] == 0.0 and body['response_format'] == {'type': 'json_object'}
    assert body['chat_template_kwargs'] == {'thinking': False} and 'guided_json' not in body
    assert r.action == {'action': 'finish'} and r.llm_usd == 0.0 and r.accounting_complete and r.llm_tokens == 940
    d = b.describe()
    assert d['kind'] == 'openai_compat_selfhost' and d['route'] == 'R2' and d['engine_version'] == '0.26.0'
    assert d['model'] == MODEL and d['serving_gpus'] == 4 and d['gpu_type'] == 'H20-96GB'
    assert r.meta['selfhost_gpu_seconds'] == pytest.approx(4 * sum(a['seconds'] for a in r.meta['attempts']), abs=1e-3)


def test_status_json_overrides_the_defaults(tmp_path, monkeypatch):
    st = {'route': 'R2', 'served_model_name': MODEL, 'engine_version': '0.26.0+cu129', 'gpus_used': [0, 1, 2, 3, 4, 5],
          'base_url_local': 'http://127.0.0.1:18001/v1', 'extra_request_params': {'chat_template_kwargs': {'thinking': True}}}
    (tmp_path / 'STATUS.json').write_text(json.dumps(st))
    monkeypatch.setattr(arms, 'WEAK_STATUS', tmp_path / 'STATUS.json')
    _, b = arms.weak_driver()
    assert b.base_url == 'http://127.0.0.1:18001/v1' and b.selfhost['gpus'] == 6 and b.describe()['engine_version'] == '0.26.0'
    assert b.extra_body['chat_template_kwargs'] == {'thinking': False}       # thinking is always forced off (JSON mode)


@pytest.mark.parametrize('exc, served', [
    (urllib.error.URLError(ConnectionRefusedError(10061, 'refused')), False),   # tunnel down
    (urllib.error.URLError(TimeoutError('timed out')), False),                   # connect timed out
    (http.client.RemoteDisconnected('Remote end closed connection without response'), True),   # tunnel up, server down
])
def test_unreachable_endpoint_fails_fast_without_retries(exc, served):
    calls = []

    def transport(url, headers, body, timeout):
        calls.append(1)
        raise exc
    assert classify_transport_error(exc) == (True, served)
    r = _selfhost(transport).complete('sys', 'prompt', schema=WEAK)
    assert len(calls) == 1 and r.error.startswith('endpoint unreachable') and r.meta['unreachable'] and r.action is None
    if served:
        assert r.accounting_complete is False and r.llm_tokens > 0
    else:
        assert r.accounting_complete is True and r.llm_tokens == 0


def test_read_timeout_is_not_unreachable():
    assert classify_transport_error(TimeoutError('read timed out')) == (False, True)


def test_requests_are_fitted_to_the_32k_context():
    seen = []

    def transport(url, headers, body, timeout):
        seen.append(body['max_tokens'])
        return 200, {'model': MODEL, 'usage': {'prompt_tokens': 10, 'completion_tokens': 5},
                     'choices': [{'message': {'content': '{"action": "finish"}'}}]}
    b = _selfhost(transport)
    b.complete('sys', 'x' * 101500, schema=WEAK)                            # ~29K estimated input tokens
    assert 256 <= seen[0] < 4096
    r = b.complete('sys', 'x' * (33000 * 4), schema=WEAK)                    # cannot fit: refused unsent, zero cost
    assert len(seen) == 1 and r.error.startswith('context too long') and r.llm_tokens == 0 and r.accounting_complete


class _Unreachable:
    model = MODEL

    def __init__(self):
        self.calls = 0

    def describe(self):
        return {'kind': 'openai_compat_selfhost'}

    def complete(self, system, prompt, **kw):
        self.calls += 1
        return LLMResult(action=None, text=None, llm_tokens=0, llm_usd=0.0, wall_seconds=0.1, accounting_complete=True,
                         error='endpoint unreachable: URLError: refused', model=MODEL, observed_models=[],
                         meta={'unreachable': True, 'selfhost_gpu_seconds': 0.0})


def test_search_stops_at_once_when_the_endpoint_goes_away(tasks, a0_package):
    from infrarsi.agent.optimize import optimize
    drv = _Unreachable()
    t = tasks['D2']
    start = synthetic_receipt(t.raw, {}, step_time_s=5.0, peak_reserved_gib=60, run_id='r', attempt_id='ed')
    tr = optimize(t, a0_package, drv, Budget(screens=8, llm_tokens=10 ** 6, wall_seconds=36000, gpu_hours=100),
                  dict(FR_DENSE, tensor_model_parallel_size=2, sequence_parallel=True), FakeScreenRunner(t, DENSE_8B),
                  judge=ReceiptJudge(), model_facts=DENSE_8B, mode='weak', start_receipt=start).to_dict()
    assert tr['stop_reason'] == 'endpoint_unreachable' and drv.calls == 1 and tr['used']['screens'] == 0


def test_weak_arm_preflight_refuses_a_dead_endpoint(monkeypatch):
    class Dead:
        context_tokens, max_tokens = 32768, 4096

        def ping(self):
            return {'ok': False, 'unreachable': True, 'error': 'URLError: refused'}
    monkeypatch.setitem(arms.DRIVERS, 'weak', lambda structured=True: ('selfhost', Dead()))
    with pytest.raises(RuntimeError, match='endpoint not ready'):
        arms.agent_search(arms.AGENT_ARMS['A0_weak_structured'], run={'run_id': 'E5-S2-WEAK-structured', 'arm': 'A0_weak_structured'},
                          task=__import__('infrarsi.contracts', fromlist=['TaskSpec']).TaskSpec.load('S2'), nodes=['WINH6'],
                          start_config={}, start_receipt={'x': 1}, reference_receipt={'x': 1}, runner=None, out_dir='unused',
                          ctx=type('C', (), {'store': type('S', (), {'references': lambda self: {'S2': {'fr_screen': 'p'}}})(),
                                             'remote': None, 'hw': None, 'log': print})())
