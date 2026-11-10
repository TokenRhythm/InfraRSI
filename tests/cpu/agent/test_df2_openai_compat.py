"""Base-model trials D1-16b (2026-10-24): a reasoning model's thinking used the whole output cap, the reply was empty or
cut and every re-ask kept the cap (DeepSeek V4.1 Flash, Qwen3.8 Flash), or sent the empty reply back as an assistant
turn that the provider refused (Kimi K3.1 Pro: HTTP 400); an endpoint that speaks only the Responses API (Cubence)."""
import json

from infrarsi.agent.backends.openai_compat import OpenAICompatBackend

OK = '{"hypotheses": []}'
SCHEMA = {'type': 'object', 'required': ['hypotheses']}


def usage(out, reasoning=0):
    return {'prompt_tokens': 1000, 'completion_tokens': out, 'completion_tokens_details': {'reasoning_tokens': reasoning}}


class Chat:
    def __init__(self, replies):
        self.replies, self.bodies = list(replies), []

    def __call__(self, url, headers, body, timeout):
        self.bodies.append({'url': url, 'headers': headers, 'body': json.loads(json.dumps(body))})
        content, finish, out = self.replies.pop(0)
        return 200, {'model': 'm', 'usage': usage(out, out if finish == 'length' else 0),
                     'choices': [{'message': {'role': 'assistant', 'content': content}, 'finish_reason': finish}]}


def test_a_reply_cut_at_the_cap_is_sent_again_with_twice_the_room():
    t = Chat([('', 'length', 4000), (OK[:9], 'length', 8000), (OK, 'stop', 900)])
    b = OpenAICompatBackend(base_url='https://x/v1', model='m', api_key='k', transport=t, max_tokens=16384)
    r = b.complete('sys', 'prompt', schema=SCHEMA, max_output_tokens=4000)
    assert r.error is None and r.action == {'hypotheses': []} and r.attempts == 3
    assert [x['body']['max_tokens'] for x in t.bodies] == [4000, 8000, 16000]
    assert all(len(x['body']['messages']) == 2 for x in t.bodies)       # resent as asked, no empty assistant turn
    assert r.llm_tokens == 3000 + 4000 + 8000 + 900                      # every attempt charged


def test_a_cut_reply_without_more_room_stops_and_says_what_bounds_it():
    t = Chat([('', 'length', 16384)])
    b = OpenAICompatBackend(base_url='https://x/v1', model='m', api_key='k', transport=t, max_tokens=16384)
    r = b.complete('sys', 'prompt', schema=SCHEMA)
    assert r.action is None and r.attempts == 1 and 'cut at the output limit (16384 tokens)' in r.error
    assert 'llm.max_tokens' in r.error


def test_larger_caps_stay_within_the_usd_the_caller_has_left():
    # DF2_r1 #1: the admission reserved four attempts at the role's cap; a doubled cap must not overspend max_usd
    t = Chat([('', 'length', 4000), ('', 'length', 8000), ('', 'length', 8000), ('', 'length', 8000)])
    b = OpenAICompatBackend(base_url='https://x/v1', model='m', api_key='k', transport=t, max_tokens=16384,
                            price={'input_miss': 0.0, 'input_hit': 0.0, 'output': 5.0})
    r = b.complete('sys', 'prompt', schema=SCHEMA, max_output_tokens=4000, max_usd=0.1)
    assert [x['body']['max_tokens'] for x in t.bodies] == [4000, 8000]
    assert r.llm_usd <= 0.1 and 'USD allowance' in r.error


def test_a_retryable_error_without_usage_is_charged_its_own_cap_and_retried():
    # DF2_r2: a 429 without usage was charged llm.max_tokens; the USD check then blocked the admitted retry
    replies = [(429, None), (200, OK)]
    sent = []

    def transport(url, headers, body, timeout):
        sent.append(body['max_tokens'])
        status, content = replies.pop(0)
        if status != 200:
            return status, {'error': {'message': 'rate limited'}}
        return 200, {'model': 'm', 'usage': usage(10), 'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]}

    b = OpenAICompatBackend(base_url='https://x/v1', model='m', api_key='k', transport=transport, max_tokens=16384,
                            price={'input_miss': 1.0, 'input_hit': 1.0, 'output': 5.0})
    r = b.complete('sys', 'prompt', schema=SCHEMA, max_output_tokens=4000, max_usd=0.083)
    assert sent == [4000, 4000] and r.error is None and r.action == {'hypotheses': []} and r.llm_usd < 0.083


def test_an_empty_complete_reply_is_corrected_without_an_empty_assistant_turn():
    t = Chat([('', 'stop', 10), (OK, 'stop', 10)])
    b = OpenAICompatBackend(base_url='https://x/v1', model='m', api_key='k', transport=t)
    r = b.complete('sys', 'prompt', schema=SCHEMA)
    assert r.error is None and t.bodies[1]['body']['messages'][2] == {'role': 'assistant', 'content': '(an empty reply)'}


class Responses:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, url, headers, body, timeout):
        self.calls.append({'url': url, 'headers': headers, 'body': json.loads(json.dumps(body))})
        text, status = self.replies.pop(0)
        out = [{'type': 'reasoning', 'summary': []}]
        if text is not None:
            out.append({'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': text}]})
        return 200, {'model': 'gpt-5.6-terra', 'status': status, 'output': out,
                     **({'incomplete_details': {'reason': 'max_output_tokens'}} if status == 'incomplete' else {}),
                     'usage': {'input_tokens': 500, 'output_tokens': 120, 'output_tokens_details': {'reasoning_tokens': 100}}}


def test_the_responses_api():
    t = Responses([(None, 'incomplete'), (OK, 'completed')])
    b = OpenAICompatBackend(base_url='https://gw/v1', model='gpt-5.6-terra', api_key='k', transport=t, api='responses',
                            headers={'User-Agent': 'agent/1.0'}, structured='json_schema', extra_body={'reasoning': {'effort': 'low'}})
    r = b.complete('the system', 'the prompt', schema=SCHEMA, max_output_tokens=2000)
    assert r.error is None and r.action == {'hypotheses': []} and r.llm_tokens == 2 * 620
    first = t.calls[0]
    assert first['url'] == 'https://gw/v1/responses' and first['headers']['User-Agent'] == 'agent/1.0'
    assert first['headers']['Authorization'] == 'Bearer k'
    body = first['body']
    assert body['instructions'] == 'the system' and body['input'] == [{'role': 'user', 'content': 'the prompt'}]
    assert body['text']['format']['type'] == 'json_schema' and body['reasoning'] == {'effort': 'low'}
    assert 'temperature' not in body and 'messages' not in body and body['store'] is False
    assert [c['body']['max_output_tokens'] for c in t.calls] == [2000, 4000]     # incomplete at the cap: more room
