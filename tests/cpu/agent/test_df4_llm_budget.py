"""Base-model trials D2-16 (2026-10-24): Qwen3.8 Flash and Kimi K3.1 Pro reasoned past 8,000 and then 16,000 output
tokens (145 s and 245 s per attempt); the 300 s per-call time budget, which a user could not change, ended the retries
with an error that named only the output limit."""
import itertools
import urllib.error

from infrarsi.agent.backends import openai_compat
from infrarsi.agent.backends.openai_compat import OpenAICompatBackend
from infrarsi.project.llm import make_driver

import pytest

from test_df2_openai_compat import OK, SCHEMA, Chat, usage


def test_the_time_budget_that_ends_the_retries_is_named_and_settable(monkeypatch):
    clock = itertools.count(0.0, 200.0)                       # every look at the clock is 200 s later
    monkeypatch.setattr(openai_compat.time, 'monotonic', lambda: next(clock))
    t = Chat([('', 'length', 4000), (OK, 'stop', 900)])
    b = OpenAICompatBackend(base_url='https://x/v1', model='m', api_key='k', transport=t, max_tokens=16384,
                            timeout_seconds=300)
    r = b.complete('sys', 'prompt', schema=SCHEMA, max_output_tokens=4000)
    assert r.action is None and len(t.bodies) == 1
    assert 'cut at the output limit (4000 tokens)' in r.error
    assert "the call's time budget (300 s, llm.timeout_s) ended the retries" in r.error
    llm = {'backend': 'openai-compat', 'base_url': 'https://x/v1', 'model': 'm'}
    assert make_driver({**llm, 'timeout_s': 900}).timeout_seconds == 900 and make_driver(llm).timeout_seconds == 300


def test_a_retry_that_times_out_keeps_the_cut_diagnosis(monkeypatch):
    # DF4_r1 #5: the first reply was cut after 145 s; the retry got the remaining 155 s and its request timed out
    now = [0.0]
    monkeypatch.setattr(openai_compat.time, 'monotonic', lambda: now[0])
    sent = []

    def transport(url, headers, body, timeout):
        sent.append(timeout)
        if len(sent) == 1:
            now[0] += 145.0
            return 200, {'model': 'm', 'usage': usage(8000, 8000),
                         'choices': [{'message': {'role': 'assistant', 'content': ''}, 'finish_reason': 'length'}]}
        now[0] += timeout
        raise TimeoutError('timed out')
    b = OpenAICompatBackend(base_url='https://x/v1', model='m', api_key='k', transport=transport, max_tokens=16384,
                            timeout_seconds=300)
    r = b.complete('sys', 'prompt', schema=SCHEMA, max_output_tokens=8000)
    assert len(sent) == 2 and sent[1] == pytest.approx(155.0) and 'cut at the output limit (8000 tokens)' in r.error
    assert "the call's time budget (300 s, llm.timeout_s) ended the retries (transport error: TimeoutError: timed out)" in r.error
    assert r.meta['transport_error'] and not r.meta['unreachable']      # DF4_r2: the flags follow the outcome, not the text


def test_an_unreachable_endpoint_after_a_cut_reply_is_still_unreachable(monkeypatch):
    # DF4_r2 #2: a connect timeout on the retry: the preflight and the agent loop must still see an unreachable endpoint
    now = [0.0]
    monkeypatch.setattr(openai_compat.time, 'monotonic', lambda: now[0])

    def transport(url, headers, body, timeout):
        if not now[0]:
            now[0] += 145.0
            return 200, {'model': 'm', 'usage': usage(8000, 8000),
                         'choices': [{'message': {'role': 'assistant', 'content': ''}, 'finish_reason': 'length'}]}
        now[0] += timeout
        raise urllib.error.URLError(TimeoutError('timed out'))
    b = OpenAICompatBackend(base_url='https://x/v1', model='m', api_key='k', transport=transport, max_tokens=16384,
                            timeout_seconds=300)
    r = b.complete('sys', 'prompt', schema=SCHEMA, max_output_tokens=8000)
    assert r.meta['unreachable'] and r.meta['transport_error'] and 'endpoint unreachable' in r.error


def test_a_thinking_budget_gets_its_room_above_the_role_cap():
    # F-D2-16b-1: Qwen3.8 Flash's provider refused max_tokens 2048 with a thinking budget of 24000 (HTTP 400)
    t = Chat([(OK, 'stop', 900)])
    b = OpenAICompatBackend(base_url='https://x/v1', model='m', api_key='k', transport=t, max_tokens=49152,
                            extra_body={'reasoning': {'max_tokens': 24000}})
    r = b.complete('sys', 'prompt', schema=SCHEMA, max_output_tokens=8000)
    assert r.error is None and t.bodies[0]['body']['max_tokens'] == 32000 and t.bodies[0]['body']['reasoning'] == {'max_tokens': 24000}
    small = OpenAICompatBackend(base_url='https://x/v1', model='m', api_key='k', transport=Chat([(OK, 'stop', 9)]),
                                max_tokens=16384, extra_body={'reasoning': {'max_tokens': 24000}})
    r = small.complete('sys', 'prompt', schema=SCHEMA, max_output_tokens=8000)
    # DF5_r1 #6: never above llm.max_tokens, and a limit at or below the budget is refused here, not sent
    assert small.transport.bodies == [] and 'leaves no room for the answer' in r.error and r.llm_tokens == 0
    tight = OpenAICompatBackend(base_url='https://x/v1', model='m', api_key='k', transport=Chat([(OK, 'stop', 9)]),
                                max_tokens=49152, extra_body={'reasoning': {'max_tokens': 24000}})
    r = tight.complete('sys', 'prompt', schema=SCHEMA, max_output_tokens=8000, token_allowance=20000)
    assert tight.transport.bodies == [] and 'leaves no room for the answer' in r.error
