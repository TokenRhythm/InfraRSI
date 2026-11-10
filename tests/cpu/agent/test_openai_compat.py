import json

from infrarsi.agent.backends.openai_compat import OpenAICompatBackend
from infrarsi.agent.tools import action_schema


class FakeTransport:
    def __init__(self, replies):
        self.replies, self.bodies = list(replies), []

    def __call__(self, url, headers, body, timeout):
        self.bodies.append({'url': url, 'headers': headers, 'body': json.loads(json.dumps(body))})
        status, content, model, usage = self.replies.pop(0)
        if status != 200:
            return status, {'error': {'message': 'boom'}}
        resp = {'model': model, 'choices': [{'message': {'role': 'assistant', 'content': content}}]}
        if usage is not None:
            resp['usage'] = usage
        return 200, resp


U = {'prompt_tokens': 1000, 'completion_tokens': 100, 'total_tokens': 1100}
OK = json.dumps({'action': 'propose_config', 'base': 'c0', 'changes': {'micro_batch_size': 2}})
WEAK = action_schema('weak')


def backend(replies, **kw):
    t = FakeTransport(replies)
    return OpenAICompatBackend(base_url='https://api.deepseek.com', model='deepseek-flash', api_key='k', transport=t, **kw), t


def test_first_reply_valid_json_schema_and_price(tmp_path):
    b, t = backend([(200, OK, 'deepseek-flash', U)])
    r = b.complete('sys', 'prompt', schema=WEAK, call_dir=tmp_path / 'c1')
    assert r.error is None and r.action['changes'] == {'micro_batch_size': 2} and r.attempts == 1
    assert r.llm_tokens == 1100 and abs(r.llm_usd - (1000 * 0.30 + 100 * 1.20) / 1e6) < 1e-12 and r.accounting_complete
    body = t.bodies[0]['body']
    assert body['response_format'] == {'type': 'json_object'} and body['temperature'] == 0.0 and body['model'] == 'deepseek-flash'
    assert t.bodies[0]['headers']['Authorization'] == 'Bearer k'
    assert json.loads((tmp_path / 'c1' / 'receipt.json').read_text())['accounting_complete'] is True


def test_retries_with_feedback_and_every_attempt_charged():
    b, t = backend([(200, 'I think TP=2', 'deepseek-flash', U), (200, '{"action":"propose_config"}', 'deepseek-flash', U),
                    (200, OK, 'deepseek-flash', U)])
    r = b.complete('sys', 'prompt', schema=WEAK)
    assert r.error is None and r.attempts == 3 and r.llm_tokens == 3300
    last = t.bodies[-1]['body']['messages']
    assert last[-1]['role'] == 'user' and 'rejected' in last[-1]['content'] and last[-2]['role'] == 'assistant'


def test_at_most_three_retries():
    b, t = backend([(200, '{"action":"bogus"}', 'deepseek-flash', U)] * 4)
    r = b.complete('sys', 'prompt', schema=WEAK)
    assert r.action is None and r.error.startswith('schema') and r.attempts == 4 and r.llm_tokens == 4400 and not t.replies


def test_served_model_must_match_no_silent_escalation():
    b, _ = backend([(200, OK, 'deepseek-v4-pro', U), (200, OK, 'deepseek-flash', U)])
    r = b.complete('sys', 'prompt', schema=WEAK)
    assert r.action is None and 'served model' in r.error and r.attempts == 1 and r.llm_tokens == 1100


def test_http_errors_retry_and_auth_errors_do_not():
    b, _ = backend([(500, None, None, None), (503, None, None, None), (200, OK, 'deepseek-flash', U)])
    r = b.complete('sys', 'prompt', schema=WEAK)
    assert r.error is None and r.attempts == 3
    b, t = backend([(401, None, None, None), (200, OK, 'deepseek-flash', U)])
    r = b.complete('sys', 'prompt', schema=WEAK)
    assert r.action is None and 'HTTP 401' in r.error and len(t.replies) == 1


def test_missing_usage_marks_accounting_incomplete():
    b, _ = backend([(200, OK, 'deepseek-flash', None)])
    r = b.complete('sys', 'prompt', schema=WEAK)
    assert r.action is not None and r.accounting_complete is False


def test_self_hosted_guided_json_has_no_usd():
    t = FakeTransport([(200, OK, 'Qwen3-Next-80B-A3B-Instruct', U)])
    b = OpenAICompatBackend(base_url='http://127.0.0.1:18000/v1', model='Qwen3-Next-80B-A3B-Instruct', api_key_env=None,
                            price=None, structured='guided_json', transport=t)
    r = b.complete('sys', 'prompt', schema=WEAK)
    assert r.error is None and r.llm_usd == 0.0 and t.bodies[0]['body']['guided_json'] == WEAK
    assert 'Authorization' not in t.bodies[0]['headers']
