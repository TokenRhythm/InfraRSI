"""The session backend (agent/backends/session.py): a model call written as a request, its answer handed in by the
session and read back as every backend's result; an unusable answer returned with the reason, the last one closing the
call; the wait ended by the call's time and by a stop; a new engine abandoning what the last one asked; the reply read
by the rule of the scripted and CLI backends; the session's stop and notes at the search's boundaries."""
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from infrarsi.agent.backends import session as S
from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.agent.search import USER_STOP
from infrarsi.project import agent_session as A
from infrarsi.project.llm import make_driver, preflight
from session_helpers import Answerer, session_dir, wait_for

SCHEMA = {'type': 'object', 'properties': {'n': {'type': 'integer'}}, 'required': ['n'], 'additionalProperties': False}


def call(be, *args, **kw):
    """backend.complete in a thread: (thread, box); box['res'] once it returned."""
    box = {}
    t = threading.Thread(target=lambda: box.update(res=be.complete(*args, **kw)), daemon=True)
    t.start()
    return t, box


def test_a_call_is_a_request_and_its_answer(tmp_path):
    d = session_dir(tmp_path)
    be = S.SessionBackend(directory=d, model='claude-opus-5-5', poll_seconds=0.01)
    t, box = call(be, 'SYSTEM\nline two', '{"ask": "é"}', schema=SCHEMA, role='hypotheses', max_output_tokens=900)
    r = wait_for(lambda: S.pending(d))[0]
    assert (r.seq, r.role, r.path.name) == (1, 'hypotheses', '0001-hypotheses')
    assert r.text('system.txt') == 'SYSTEM\nline two' and r.text('prompt.txt') == '{"ask": "é"}' and r.schema() == SCHEMA
    assert r.meta['max_output_tokens'] == 900 and r.meta['expect_json'] is True
    assert r.meta['digest'] == S.request_digest('hypotheses', 'SYSTEM\nline two', '{"ask": "é"}', SCHEMA)
    readme = r.text('README.md')
    assert 'answer.json' in readme and 'schema.json' in readme and '900 tokens' in readme
    view = A.status(d)                                   # what `agent next` shows the session
    assert view['state'] == 'request' and view['request']['id'] == 1 and view['request']['attempts_left'] == 2
    assert 'infrarsi agent answer 1' in view['do']
    (r.path / 'answer.json').write_text('{"n": 7}', encoding='utf-8')
    got = A.answer(d, 1)
    assert got['ok'] and got['closed'] and r.state == 'answered'
    t.join(10)
    res = box['res']
    assert res.action == {'n': 7} and res.error is None and res.text == '{"n": 7}' and res.model == 'claude-opus-5-5'
    assert res.llm_usd == 0.0 and res.accounting_complete is False and res.meta['usage_estimated'] is True
    assert res.llm_tokens == S.estimate_tokens('SYSTEM\nline two', '{"ask": "é"}', '{"n": 7}', be.call_overhead_tokens)
    assert res.meta['call_dir'] == str(r.path) and res.meta['status'] == 'RETURNED'
    assert A.answer(d, 1)['ok'] is False                 # handed in once


def test_the_tokens_the_session_reports_are_the_charge(tmp_path):
    d = session_dir(tmp_path)
    be = S.SessionBackend(directory=d, poll_seconds=0.01)
    with Answerer(d, lambda role, system, prompt: {'n': 1}, tokens=4321):
        res = be.complete('s', 'p', schema=SCHEMA, role='prior')
    assert res.llm_tokens == 4321 and res.accounting_complete is True and res.meta['usage_estimated'] is False
    assert be.model == 'session' and be.describe()['kind'] == 'session'


def test_an_unusable_answer_goes_back_with_the_reason(tmp_path):
    d = session_dir(tmp_path)
    be = S.SessionBackend(directory=d, poll_seconds=0.01, max_attempts=3)
    t, box = call(be, 's', 'p', schema=SCHEMA, role='prior')
    r = wait_for(lambda: S.pending(d))[0]
    (r.path / 'answer.json').write_text('not json', encoding='utf-8')
    got = A.answer(d, 1)
    assert not got['ok'] and not got['closed'] and got['attempts_left'] == 2 and got['error'] == 'invalid structured JSON action'
    assert 'feedback.txt' in got['do'] and 'invalid structured JSON action' in r.text('feedback.txt')
    assert A.status(d)['request']['feedback'].startswith('Your answer could not be used')
    (r.path / 'answer.json').write_text('{"n": "seven"}', encoding='utf-8')
    got = A.answer(d, 1)
    assert not got['ok'] and got['attempts_left'] == 1 and got['error'].startswith('schema: $.n: expected integer')
    (r.path / 'answer.json').write_text('Here it is:\n```json\n{"n": 3}\n```\n', encoding='utf-8')   # (one fenced block reads)
    assert A.answer(d, 1)['ok']
    t.join(10)
    res = box['res']
    assert res.action == {'n': 3} and res.attempts == 3 and res.meta['parse_mode'] == 'embedded'
    # all three replies are charged (none came with a count: each estimated from the request's and its own size)
    assert res.llm_tokens == sum(S.estimate_tokens('s', 'p', x, be.call_overhead_tokens) for x in (
        'not json', '{"n": "seven"}', 'Here it is:\n```json\n{"n": 3}\n```\n')) and res.accounting_complete is False


def test_every_reply_handed_in_is_charged(tmp_path):
    """The CLI backends charge each attempt of a call; so does the session: the replies it returned and the one it took."""
    d = session_dir(tmp_path)
    be = S.SessionBackend(directory=d, poll_seconds=0.01, max_attempts=3)
    t, box = call(be, 's', 'p', schema=SCHEMA, role='prior')
    r = wait_for(lambda: S.pending(d))[0]
    assert r.meta['overhead_tokens'] == be.call_overhead_tokens == 4000
    (r.path / 'answer.json').write_text('nope', encoding='utf-8')
    assert not A.answer(d, 1, tokens=700)['ok']
    assert S.read_json(r.path / 'rejected-1.json')['llm_tokens'] == 700 and S.usage(r) == (700, True)
    (r.path / 'answer.json').write_text('{"n": 1}', encoding='utf-8')
    assert A.answer(d, 1, tokens=900)['ok']
    t.join(10)
    assert box['res'].llm_tokens == 1600 and box['res'].accounting_complete is True and box['res'].attempts == 2
    # a call that ends without an answer still pays for the reply it returned (a count for one, none for the other)
    t, box = call(be, 's', 'p', schema=SCHEMA, role='prior', timeout=0.6)
    r = wait_for(lambda: S.pending(d))[0]
    (r.path / 'answer.json').write_text('nope', encoding='utf-8')
    assert not A.answer(d, r.seq, tokens=500)['ok']
    t.join(10)
    assert box['res'].meta['status'] == 'TIMEOUT' and box['res'].llm_tokens == 500 and box['res'].accounting_complete is True
    with Answerer(d, lambda role, system, prompt: '{"n": 0.5}') as a:    # the default: two replies, then the call is over
        res = S.SessionBackend(directory=d, poll_seconds=0.01).complete('s', 'p', schema=SCHEMA, role='prior')
    assert len(a.handed) == 2 and res.attempts == 2 and res.accounting_complete is False
    assert res.llm_tokens == 2 * S.estimate_tokens('s', 'p', '{"n": 0.5}', 4000)


def test_no_second_reply_when_the_tokens_left_do_not_cover_it(tmp_path):
    d = session_dir(tmp_path)
    be = S.SessionBackend(directory=d, poll_seconds=0.01, max_attempts=3)
    prompt = 'p' * 7000                                  # 2000 tokens by size: a re-ask reserves 2500 + 4000 + 256
    t, box = call(be, 's', prompt, schema=SCHEMA, role='prior', token_allowance=12000)
    r = wait_for(lambda: S.pending(d))[0]
    (r.path / 'answer.json').write_text('nope', encoding='utf-8')
    got = A.answer(d, 1, tokens=6000)                    # 6000 left: not enough for another
    assert not got['ok'] and got['closed'] and got['attempts_left'] == 0 and 'last attempt' in r.text('feedback.txt')
    t.join(10)
    assert box['res'].error == 'invalid structured JSON action' and box['res'].llm_tokens == 6000
    t, box = call(be, 's', prompt, schema=SCHEMA, role='prior', token_allowance=12000)
    r = wait_for(lambda: S.pending(d))[0]
    (r.path / 'answer.json').write_text('nope', encoding='utf-8')
    got = A.answer(d, r.seq, tokens=5000)                # 7000 left: one more fits
    assert not got['closed'] and got['attempts_left'] == 2
    (r.path / 'answer.json').write_text('{"n": 2}', encoding='utf-8')
    assert A.answer(d, r.seq, tokens=100)['ok']
    t.join(10)
    assert box['res'].action == {'n': 2} and box['res'].llm_tokens == 5100


def test_a_directory_an_engine_left_half_made_is_no_request_and_its_number_is_taken(tmp_path):
    d = session_dir(tmp_path)
    be = S.SessionBackend(directory=d, poll_seconds=0.01)
    half = d / S.REQUESTS / '0001-hypotheses'            # made, and the engine ended before its request.json
    half.mkdir()
    (half / 'system.txt').write_text('s', encoding='utf-8')
    assert S.requests(d) == [] and A.status(d)['state'] == 'running'
    with Answerer(d, lambda role, system, prompt: {'n': 1}):
        res = S.SessionBackend(directory=d, poll_seconds=0.01).complete('s', 'p', schema=SCHEMA, role='hypotheses')
    assert res.action == {'n': 1} and res.meta['call_id'] == '0002' and [r.seq for r in S.requests(d)] == [2]
    assert be.complete('s', 'p', role='preflight').action == {'ok': True}


def test_the_last_unusable_answer_closes_the_call_with_its_error(tmp_path):
    d = session_dir(tmp_path)
    be = S.SessionBackend(directory=d, poll_seconds=0.01, max_attempts=2)
    with Answerer(d, lambda role, system, prompt: '{"n": 1.5}') as a:
        res = be.complete('s', 'p', schema=SCHEMA, role='prior')
    assert [h['closed'] for h in a.handed] == [False, True] and not any(h['ok'] for h in a.handed)
    assert res.action is None and res.error.startswith('schema: $.n: expected integer') and res.text == '{"n": 1.5}'
    assert res.attempts == 2 and res.llm_tokens > 0     # (the failed call is charged, as the CLI backends charge it)
    r = S.find(d, 1)
    assert r.state == 'answered' and 'last attempt' in r.text('feedback.txt')


def test_a_byte_order_mark_and_a_file_elsewhere(tmp_path):
    d = session_dir(tmp_path)
    be = S.SessionBackend(directory=d, poll_seconds=0.01)
    t, box = call(be, 's', 'p', schema=SCHEMA, role='prior')
    wait_for(lambda: S.pending(d))
    with pytest.raises(A.AgentError, match='cannot be read'):
        A.answer(d, 1)                                   # nothing written yet
    with pytest.raises(A.AgentError, match='no request 9'):
        A.answer(d, 9)
    other = tmp_path / 'elsewhere.json'
    other.write_bytes(b'\xef\xbb\xbf{"n": 2}')           # a file some shells write: UTF-8 with a byte-order mark
    assert A.answer(d, 1, file=other, by='some-model')['ok']
    t.join(10)
    assert box['res'].action == {'n': 2} and box['res'].meta['answered_by'] == 'some-model'


def test_preflight_asks_no_one(tmp_path):
    d = session_dir(tmp_path)
    be = S.SessionBackend(directory=d)
    p = preflight(be)
    assert p['ok'] and not p['refused'] and S.requests(d) == []


def test_the_time_a_call_has_ends_its_wait(tmp_path):
    d = session_dir(tmp_path)
    be = S.SessionBackend(directory=d, poll_seconds=0.01)
    res = be.complete('s', 'p', schema=SCHEMA, role='prior', timeout=0.05)
    assert res.action is None and res.error.startswith('no answer from the session within') and res.llm_tokens == 0
    assert res.meta['status'] == 'TIMEOUT'
    r = S.find(d, 1)
    (r.path / 'answer.json').write_text('{"n": 1}', encoding='utf-8')
    late = A.answer(d, 1)                                # an answer after the close is refused, and says so
    assert not late['ok'] and late['closed'] and 'closed without an answer (no answer from the session' in late['error']
    assert S.SessionBackend(directory=d, answer_timeout=0.05, poll_seconds=0.01).complete('s', 'p').meta['status'] == 'TIMEOUT'


def test_a_stop_ends_the_wait(tmp_path):
    d = session_dir(tmp_path)
    be = S.SessionBackend(directory=d, poll_seconds=0.01)
    t, box = call(be, 's', 'p', schema=SCHEMA, role='hypotheses')
    wait_for(lambda: S.pending(d))
    S.write_text(d / S.CONTROL / 'stop', 'now\n')
    t.join(10)
    assert box['res'].meta['status'] == 'STOPPED' and 'stopped' in box['res'].error and box['res'].llm_tokens == 0
    # until the search reaches its boundary it may call again: those calls end at once, and no request is written
    again = be.complete('s', 'p', schema=SCHEMA, role='hypotheses')
    assert again.meta['status'] == 'STOPPED' and again.action is None and len(S.requests(d)) == 1


def test_a_new_engine_abandons_what_the_last_one_asked(tmp_path):
    d = session_dir(tmp_path)
    old = S.SessionBackend(directory=d, poll_seconds=0.01)
    t, box = call(old, 's', 'p', schema=SCHEMA, role='prior')
    r1 = wait_for(lambda: S.pending(d))[0]
    new = S.SessionBackend(directory=d, poll_seconds=0.01)
    assert r1.state == 'abandoned' and S.pending(d) == []
    t.join(10)                                           # (an engine still there stops waiting for what is no longer its)
    assert box['res'].meta['status'] == 'ABANDONED' and box['res'].action is None and box['res'].llm_tokens == 0
    with Answerer(d, lambda role, system, prompt: {'n': 5}):
        res = new.complete('s', 'p', schema=SCHEMA, role='prior')
    assert res.action == {'n': 5} and res.meta['call_id'] == '0002'
    (r1.path / 'answer.json').write_text('{"n": 1}', encoding='utf-8')
    assert 'already abandoned' in A.answer(d, 1)['error']


@pytest.mark.parametrize('reply', [{'n': 4}, '{"n": 4}', '```json\n{"n": 4}\n```', 'no json here', '{"n": "x"}', '[1, 2]',
                                   '{"n": 4, "more": 1}'])
def test_a_reply_reads_as_the_scripted_backend_reads_it(tmp_path, reply):
    """The same reply through the session and through the scripted backend: the same action, or an error of the same kind."""
    d = session_dir(tmp_path)
    be = S.SessionBackend(directory=d, poll_seconds=0.01, max_attempts=1)
    with Answerer(d, lambda role, system, prompt: reply, tokens=2000):
        got = be.complete('s', 'p', schema=SCHEMA, role='prior')
    want = ScriptedBackend([reply], tokens_per_call=2000).complete('s', 'p', schema=SCHEMA, role='prior')
    fenced = isinstance(reply, str) and reply.startswith('```')     # (the CLI backends' rule reads one fenced block)
    assert got.action == (want.action if not fenced else {'n': 4})
    assert bool(got.error) == (bool(want.error) and not fenced)
    if want.error and not fenced:
        assert got.error.split(':')[0] == want.error.split(':')[0]
    assert got.llm_tokens == want.llm_tokens == 2000


def test_free_text_calls(tmp_path):
    d = session_dir(tmp_path)
    be = S.SessionBackend(directory=d, poll_seconds=0.01)
    with Answerer(d, lambda role, system, prompt: 'plain words') as a:
        res = be.complete('s', 'p', role='optico_propose', expect_json=False)
    assert res.text == 'plain words' and res.action is None and res.error is None
    assert S.find(d, 1).meta['answer_file'] == 'answer.txt' and a.handed[0]['ok']


def test_the_session_backend_needs_a_session(tmp_path, monkeypatch):
    monkeypatch.delenv(S.ENV_DIR, raising=False)
    with pytest.raises(ValueError, match='infrarsi agent start'):
        make_driver('session')
    with pytest.raises(FileNotFoundError):
        make_driver({'backend': 'session', 'dir': str(tmp_path / 'none')})
    d = session_dir(tmp_path)
    monkeypatch.setenv(S.ENV_DIR, str(d))
    be = make_driver('session:gpt-6.1')
    assert be.kind == 'session' and be.model == 'gpt-6.1' and be.out_root is None
    assert be.max_attempts == S.MAX_ATTEMPTS == 2        # one re-ask, as the CLI backends the same function makes
    assert make_driver({'backend': 'session', 'max_attempts': 4}).max_attempts == 4
    be.out_root = tmp_path / 'runs' / 'r1' / 'llm_calls'             # as flow.run_project sets it
    assert S.read_json(d / 'run.json')['run_dir'] == str(tmp_path / 'runs' / 'r1')
    assert A.run_dir_of(d) == tmp_path / 'runs' / 'r1'
    for name in ('second', 'verifier'):                  # the debate's other models, were they session backends too
        other = make_driver('session:another')           # (agent/debate.DebateDriver.attach gives them these directories)
        other.out_root = tmp_path / 'runs' / 'r1' / 'llm_calls' / name
        assert A.run_dir_of(d) == tmp_path / 'runs' / 'r1' and other.out_root.name == name   # the run's pointer stays


def fake_session(screens=3, budget=8):
    events = []
    cands = {'c0': SimpleNamespace(verdict={'step_time': 2.0}), 'c4': SimpleNamespace(verdict={'step_time': 1.25})}
    return SimpleNamespace(
        ledger=SimpleNamespace(snapshot=lambda: {'screens': screens, 'llm_tokens': 10}), budget=SimpleNamespace(screens=budget),
        best_valid=lambda: 'c4', candidates=cands, llm_calls=[1, 2], hooks=SimpleNamespace(live_notes=[]), stop_flag=None,
        event=lambda kind, payload: events.append((kind, payload)), events=events)


def test_notes_and_a_stop_reach_the_search_at_its_boundaries(tmp_path):
    d = session_dir(tmp_path)
    be = S.SessionBackend(directory=d)
    ctl, s, saved = be.session_control, fake_session(), []
    ctl.poll(s, checkpoint=lambda: saved.append(1))
    assert s.stop_flag is None and saved == [] and not hasattr(s, 'live_notes')
    p = S.read_json(d / 'progress.json')
    assert (p['screens_done'], p['screens_budget'], p['best'], p['best_speedup'], p['llm_calls']) == (3, 8, 'c4', 1.6, 2)
    assert A.status(d)['progress']['best_speedup'] == 1.6
    A.note(d, 'try the fused optimizer first')
    A.note(d, 'do not touch the dataloader')
    ctl.poll(s, checkpoint=lambda: saved.append(1))
    assert s.live_notes == s.hooks.live_notes == ['try the fused optimizer first', 'do not touch the dataloader']
    assert [e[1]['text'] for e in s.events] == ['/note try the fused optimizer first', '/note do not touch the dataloader']
    assert saved == [1]
    ctl.poll(s, checkpoint=lambda: saved.append(1))      # read once
    assert len(s.events) == 2 and saved == [1] and s.session_notes_read == 2
    # a resumed run reads on from where its restored state stands. One whose checkpoint has both notes takes none again;
    resumed = fake_session()
    resumed.live_notes, resumed.session_notes_read = list(s.live_notes), 2
    S.SessionBackend(directory=d).session_control.poll(resumed)
    assert resumed.live_notes == s.live_notes and resumed.events == [] and resumed.hooks.live_notes == s.live_notes
    # one whose engine ended after reading the second note and before saving it has only the first: it reads the second
    early = fake_session()
    early.live_notes, early.session_notes_read = ['try the fused optimizer first'], 1
    S.SessionBackend(directory=d).session_control.poll(early)
    assert early.live_notes == s.live_notes and [e[1]['text'] for e in early.events] == ['/note do not touch the dataloader']
    with pytest.raises(A.AgentError):
        A.note(d, '   ')
    S.write_text(d / S.CONTROL / 'stop', 'x\n')
    ctl.before_trial(s)
    assert s.stop_flag == USER_STOP
    s.stop_flag = None
    ctl.poll(s, checkpoint=lambda: saved.append(1))
    assert s.stop_flag == USER_STOP and saved == [1, 1]


def test_a_close_then_an_answer(tmp_path):
    d = session_dir(tmp_path)
    be = S.SessionBackend(directory=d, poll_seconds=0.01)
    call(be, 's', 'p', schema=SCHEMA, role='prior')
    r = wait_for(lambda: S.pending(d))[0]
    assert S.close(r, 'no answer', status='TIMEOUT') is True
    assert S.close(r, 'again', status='TIMEOUT') is False
    assert S.submit(r, '{"n": 1}')['ok'] is False and r.accepted()['error'] == 'no answer'
    assert [p.name for p in r.path.iterdir() if p.name.endswith('.tmp')] == []


def bare_request(d, k, **meta):
    """A request directory as the engine writes one, without an engine waiting on it."""
    path = d / S.REQUESTS / f'{k:04d}-prior'
    path.mkdir(parents=True)
    S.write_json(path / 'request.json', {'schema': S.REQUEST_SCHEMA, 'seq': k, 'role': 'prior', 'expect_json': True,
                                        'max_attempts': 2, 'bytes': {'system': 1, 'prompt': 1}, **meta})
    S.write_json(path / 'schema.json', SCHEMA)
    return S.find(d, k)


def test_many_writers_at_once_and_one_record(tmp_path):
    """Closes and answers racing for one request: exactly one of them becomes its record, whole, and only that one is
    told it did."""
    d = session_dir(tmp_path)
    for k in range(1, 9):
        r, start, won = bare_request(d, k), threading.Barrier(6), []

        def write(i, r=r, start=start, won=won):
            start.wait()
            if i % 2:
                ok = S.close(S.Request(r.path), f'close {i}', status='TIMEOUT')
            else:
                ok = S.submit(S.Request(r.path), json.dumps({'n': i}))['ok']
            if ok:
                won.append(i)
        threads = [threading.Thread(target=write, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        rec = r.accepted()
        assert len(won) == 1, won
        assert (rec['error'] == f'close {won[0]}') if won[0] % 2 else (json.loads(rec['text']) == {'n': won[0]})


def test_unusable_replies_at_once_are_each_counted_and_the_limit_holds(tmp_path):
    """Six unusable replies handed in at the same moment for a request that allows three: three are recorded, each under
    its own number and with its own tokens; the third closes the call; the other three are told the request is over; a
    close arriving in between waits for the reply being recorded, so what the call cost is whole."""
    d = session_dir(tmp_path)
    for k in range(1, 6):
        r, start, got = bare_request(d, k, max_attempts=3), threading.Barrier(7), []

        def hand_in(i, r=r, start=start, got=got):
            start.wait()
            got.append(S.submit(S.Request(r.path), f'not json {i}', tokens=100 + i))

        def closer(r=r, start=start):
            start.wait()
            S.close(S.Request(r.path), 'no answer', status='TIMEOUT')
        threads = [threading.Thread(target=hand_in, args=(i,)) for i in range(6)] + [threading.Thread(target=closer)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(20)
        rows = [S.read_json(p) for p in sorted(r.path.glob('rejected-*.json'))]
        counted = [g for g in got if g['error'] == 'invalid structured JSON action']
        assert len(rows) == len(counted) <= 3 and len({row['text'] for row in rows}) == len(rows)
        assert sorted(p.name for p in r.path.glob('rejected-*.json')) == [f'rejected-{n}.json' for n in range(1, len(rows) + 1)]
        assert S.usage(r) == (sum(row['llm_tokens'] for row in rows), True)
        assert r.state == 'answered' and sum(1 for g in got if 'nothing was handed in' in str(g['error'])) == 6 - len(rows)
        assert sorted(g['attempts'] for g in counted) == list(range(1, len(counted) + 1))   # (each its own number)


def test_a_close_left_half_made_by_a_killed_answer_command_is_finished(tmp_path):
    """Recording a reply as unusable is the step that counts. An answer command killed right after recording the last
    reply a call allows, before closing the call, leaves a request that looks open: the next answer does not get a third
    attempt, and the engine that waits ends the call itself."""
    d = session_dir(tmp_path)
    r = bare_request(d, 1)                               # two replies allowed; both recorded as unusable, no closing record
    for n in (1, 2):
        S.write_json(r.path / f'rejected-{n}.json', {'text': f'bad {n}', 'error': 'invalid structured JSON action',
                                                    'llm_tokens': 100 * n})
    assert r.state == 'pending'
    late = S.submit(r, '{"n": 1}')                       # a good reply now is too late: the call had its two
    assert not late['ok'] and late['closed'] and 'nothing was handed in' in late['error']
    rec = r.accepted()
    assert rec['text'] == 'bad 2' and rec['error'] == 'invalid structured JSON action' and rec['attempts'] == 2
    assert S.usage(r, rec) == (300, True)                # (each recorded reply once)
    be = S.SessionBackend(directory=d, poll_seconds=0.01)
    t, box = call(be, 's', 'p', schema=SCHEMA, role='prior')
    r2 = wait_for(lambda: S.pending(d))[0]
    (r2.path / 'answer.json').write_text('nope', encoding='utf-8')
    assert A.answer(d, r2.seq, tokens=50)['closed'] is False
    S.write_json(r2.path / 'rejected-2.json', {'text': '{}', 'error': 'schema: $: missing n', 'llm_tokens': 70})
    t.join(10)                                           # the engine did not wait for a closing record that never comes
    res = box['res']                                     # (and reads the reply itself: the error is its own check's)
    assert res.error == 'schema: $: missing n' and res.attempts == 2 and res.llm_tokens == 120 and res.action is None
    assert S.find(d, r2.seq).state == 'answered'


def test_a_writer_that_dies_leaves_no_lock_and_one_that_stays_is_waited_for(tmp_path):
    import subprocess
    import sys
    d = session_dir(tmp_path)
    r = bare_request(d, 1)
    code = ('import sys, os; sys.path.insert(0, sys.argv[1]); from infrarsi.agent.backends import session as S; '
            'f = S.hold(sys.argv[2]); assert f is not None; os._exit(0)')      # takes the request's lock and is gone
    repo = Path(S.__file__).resolve().parents[3]
    assert subprocess.run([sys.executable, '-c', code, str(repo), str(r.path / '.lock')], timeout=60).returncode == 0
    assert S.close(r, 'no answer', status='TIMEOUT') is True       # the operating system dropped the dead writer's lock
    r2 = bare_request(d, 2)
    held = S.hold(r2.path / '.lock')                     # a writer that is still there (this process, another file)
    assert held is not None and S.hold(r2.path / '.lock') is None
    with pytest.raises(TimeoutError, match='request 2 has been held'):
        with S.changing(r2, patience=0.2):
            pass
    held.close()
    assert S.submit(r2, '{"n": 1}')['ok'] is True
