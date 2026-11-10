"""C7 - the OpenAI-compatible LLM path end to end over HTTP (a local fake endpoint): the backend's contract (refusals,
retries, cut replies, served model, time budget, accounting), the manifest's llm block (base_url variables, headers,
extra_body, price), and what `doctor` / `run` do with a model that refuses, is unreachable, or drives the search."""
import json
import socket

import pytest

from _support import cli
from _support.consistency import check_run
from _support.fake_llm import Cut, Delay, FakeLLM, Reply, Served, by_role, hypothesis, role_of
from _support.toy import latest_run, manifest, report, result, runs, write_manifest
from infrarsi.project import Project, ProjectRunner
from infrarsi.project.flow import make_verifier
from infrarsi.project.llm import make_driver, override, parse, preflight

SCHEMA = {'type': 'object', 'required': ['hypotheses'], 'properties': {'hypotheses': {'type': 'array'}}}
OK = {'hypotheses': []}


def driver(llm, **extra):
    return make_driver({'backend': 'openai-compat', 'model': llm.model, 'base_url': llm.base_url, 'api_key_env': None,
                        'timeout_s': 5, **extra})


# ---------------------------------------------------------------- the backend over HTTP
def test_preflight_and_a_plain_call():
    with FakeLLM() as llm:
        d = driver(llm)
        p = preflight(d)
        assert p['ok'] and not p['refused'] and p['llm_tokens'] > 0
        r = d.complete('sys', 'say ok', schema=SCHEMA) if False else d.complete('sys', 'say ok', schema={'type': 'object'})
        assert r.error is None and r.action == {'ok': True} and r.accounting_complete is False   # unpriced: USD unknown
        body = llm.requests[0]['body']
        assert body['model'] == llm.model and body['temperature'] == 0.0 and body['response_format'] == {'type': 'json_object'}
        assert body['stream'] is False and llm.requests[0]['path'].endswith('/chat/completions')


@pytest.mark.parametrize('status', [400, 401, 403, 404])
def test_a_refusal_is_not_retried_and_the_preflight_says_so(status):
    with FakeLLM(lambda req: Reply(status, {'error': {'message': 'nope'}})) as llm:
        p = preflight(driver(llm))
        assert p['refused'] and not p['ok'] and f'HTTP {status}' in p['error']
        assert len(llm.requests) == 1


def test_5xx_and_429_are_retried_then_the_answer_counts():
    def script(req):
        return Reply(500, {'error': {'message': 'busy'}}) if req['n'] == 0 else Reply(429, {'error': {'message': 'slow down'}}) if req['n'] == 1 else OK
    with FakeLLM(script) as llm:
        r = driver(llm).complete('sys', 'p', schema=SCHEMA)
        assert r.error is None and r.attempts == 3 and len(llm.requests) == 3
        assert r.accounting_complete is False and r.llm_tokens > 0          # the failed attempts are charged a bound


def test_a_schema_violation_is_fed_back_once():
    with FakeLLM(lambda req: {'hypotheses': 'not a list'} if req['n'] == 0 else OK) as llm:
        r = driver(llm).complete('sys', 'p', schema=SCHEMA)
        assert r.error is None and r.attempts == 2
        msgs = llm.requests[1]['body']['messages']
        assert msgs[-2]['role'] == 'assistant' and msgs[-1]['role'] == 'user' and 'rejected' in msgs[-1]['content']


def test_prose_instead_of_json_is_asked_again_up_to_three_times():
    with FakeLLM(lambda req: 'I would try TP=2 first.') as llm:
        r = driver(llm).complete('sys', 'p', schema=SCHEMA)
        assert r.action is None and r.attempts == 4 and 'not a JSON object' in r.error


def test_a_cut_reply_is_asked_again_with_twice_the_room():
    big = {'hypotheses': [hypothesis({'compile': True}, 10) for _ in range(6)]}
    with FakeLLM(lambda req: Cut(big) if req['n'] == 0 else big) as llm:
        r = driver(llm, max_tokens=16000).complete('sys', 'p', schema=SCHEMA, max_output_tokens=1000)
        assert r.error is None and r.attempts == 2
        m1, m2 = llm.requests[0]['body']['max_tokens'], llm.requests[1]['body']['max_tokens']
        assert m1 == 1000 and m2 == 2000
        assert llm.requests[1]['body']['messages'] == llm.requests[0]['body']['messages']       # the same request, more room


def test_a_cut_reply_without_room_left_is_an_error_naming_the_knob():
    with FakeLLM(lambda req: Cut(OK)) as llm:
        r = driver(llm, max_tokens=1000).complete('sys', 'p', schema=SCHEMA, max_output_tokens=1000)
        assert r.action is None and 'cut at the output limit' in r.error and 'llm.max_tokens' in r.error
        assert len(llm.requests) == 1


def test_a_served_model_that_differs_is_refused():
    with FakeLLM(lambda req: Served(OK, 'bigger-model')) as llm:
        r = driver(llm).complete('sys', 'p', schema=SCHEMA)
        assert r.action is None and 'served model' in r.error and len(llm.requests) == 1


def test_a_slow_model_ends_at_the_call_time_budget():
    with FakeLLM(lambda req: Delay(3, OK)) as llm:
        r = driver(llm, timeout_s=1).complete('sys', 'p', schema=SCHEMA)
        assert r.action is None and r.error and ('time budget' in r.error or 'transport error' in r.error or 'timed out' in r.error.lower())
        assert r.wall_seconds < 2.5


def test_a_malformed_payload_and_missing_usage_are_charged_a_bound():
    with FakeLLM(lambda req: Reply(200, {'foo': 1})) as llm:
        r = driver(llm).complete('sys', 'p', schema=SCHEMA)
        assert r.action is None and r.error and not r.accounting_complete and r.llm_tokens > 0
    with FakeLLM(lambda req: Reply(200, {'model': 'fake-model', 'choices': [{'message': {'content': json.dumps(OK)}, 'finish_reason': 'stop'}]})) as llm:
        r = driver(llm).complete('sys', 'p', schema=SCHEMA)
        assert r.error is None and not r.accounting_complete and r.llm_tokens > 0


def test_headers_extra_body_and_api_key(monkeypatch):
    monkeypatch.setenv('MY_LLM_KEY', 'secret-1')
    with FakeLLM() as llm:
        d = driver(llm, headers={'User-Agent': 'infrarsi-test'}, extra_body={'chat_template_kwargs': {'enable_thinking': False}},
                   api_key_env='MY_LLM_KEY')
        d.complete('sys', 'p', schema={'type': 'object'})
        req = llm.requests[0]
        assert req['headers']['User-Agent'] == 'infrarsi-test' and req['headers']['Authorization'] == 'Bearer secret-1'
        assert req['body']['chat_template_kwargs'] == {'enable_thinking': False}


def test_responses_api_round_trip_and_cut():
    with FakeLLM(lambda req: Cut(OK) if req['n'] == 0 else OK) as llm:
        r = driver(llm, api='responses', max_tokens=16000).complete('sys', 'p', schema=SCHEMA, max_output_tokens=500)
        assert r.error is None and r.attempts == 2
        assert all(q['path'].endswith('/responses') for q in llm.requests)
        b = llm.requests[0]['body']
        assert b['instructions'] == 'sys' and b['max_output_tokens'] == 500 and 'temperature' not in b and b['store'] is False


def test_base_url_variables_and_bad_urls(monkeypatch):
    with FakeLLM() as llm:
        monkeypatch.setenv('FAKE_LLM_URL', llm.base_url)
        d = make_driver({'backend': 'openai-compat', 'model': llm.model, 'base_url': '$FAKE_LLM_URL', 'api_key_env': None})
        assert d.base_url == llm.base_url and preflight(d)['ok']
    monkeypatch.delenv('FAKE_LLM_URL')
    with pytest.raises(ValueError, match='FAKE_LLM_URL is not set'):
        make_driver({'backend': 'openai-compat', 'model': 'm', 'base_url': '$FAKE_LLM_URL'})
    for bad in ('ftp://x/v1', 'localhost:8000/v1', 'http:///v1', 'http://h:notaport/v1'):
        with pytest.raises(ValueError, match='not an http'):
            make_driver({'backend': 'openai-compat', 'model': 'm', 'base_url': bad})
    with pytest.raises(ValueError, match='needs a model name'):
        make_driver('openai-compat')
    with pytest.raises(ValueError, match='unknown llm backend'):
        make_driver('bogus:m')
    assert make_driver('none') is None and make_driver(None) is None
    assert make_driver('vllm:m').base_url == 'http://127.0.0.1:8000/v1'


def test_context_and_thinking_budget_refusals_are_not_sent():
    with FakeLLM() as llm:
        r = driver(llm, context_tokens=100).complete('sys', 'p' * 2000, schema=SCHEMA)
        assert 'context too long' in r.error and len(llm.requests) == 0 and r.llm_tokens == 0
        r = driver(llm, max_tokens=1500, extra_body={'reasoning': {'max_tokens': 2000}}).complete('sys', 'p', schema=SCHEMA)
        assert 'thinking budget' in r.error and len(llm.requests) == 0


def test_price_and_unpriced_accounting():
    with FakeLLM() as llm:
        d = driver(llm, price={'input_miss': 1.0, 'input_hit': 1.0, 'output': 2.0})
        r = d.complete('sys', 'p', schema={'type': 'object'})
        assert r.llm_usd > 0 and r.accounting_complete and not d.unpriced
        assert driver(llm).unpriced                                   # a paid API without llm.price: USD not known
        # `llm.price: null` is stripped by the spec parser (None values are dropped), so only the vllm backend means
        # "self-hosted, USD 0": a note for the README, not a defect
        selfhost = make_driver({'backend': 'vllm', 'model': llm.model, 'base_url': llm.base_url})
        assert selfhost.unpriced is False and selfhost.price is None


def test_llm_spec_parsing_and_override():
    assert parse('vllm:Qwen3-30B') == {'backend': 'vllm', 'model': 'Qwen3-30B'} and parse('none') == {} and parse(None) == {}
    m = {'backend': 'vllm', 'model': 'a', 'base_url': 'http://h:1/v1', 'capability': 0.5}
    assert override(m, 'vllm:b') == {'backend': 'vllm', 'model': 'b', 'base_url': 'http://h:1/v1'}   # capability only for the same model
    assert override(m, 'vllm:a')['capability'] == 0.5
    assert override(m, 'anthropic') == {'backend': 'anthropic'} and override(m, 'none') == {'backend': 'none'}


# ---------------------------------------------------------------- doctor and run with a model
def _with_llm(root, llm, **extra):
    doc = manifest(root)
    doc['llm'] = {'backend': 'openai-compat', 'model': llm.model, 'base_url': llm.base_url, 'api_key_env': None,
                  'timeout_s': 10, **extra}
    write_manifest(root, doc)


def test_doctor_reports_the_preflight(toy):
    with FakeLLM() as llm:
        _with_llm(toy, llm)
        res = cli.run('doctor', toy)
        assert res.code == 0 and f'llm: openai-compat {llm.model} answers' in res.out
        assert len(llm.requests) == 1 and role_of(llm.requests[0]) == 'preflight'


def test_run_stops_before_any_gpu_time_when_the_model_refuses(toy):
    with FakeLLM(lambda req: Reply(403, {'error': {'message': 'provider terms'}})) as llm:
        _with_llm(toy, llm)
        res = cli.run('run', toy, '--strategy', 'hls')
        assert res.code == 2 and 'refused a test call' in res.err and '--llm none' in res.err
        assert runs(toy) == [], 'a run directory was created although the model refused before any GPU time'


def test_run_stops_before_any_gpu_time_when_the_endpoint_is_unreachable(toy):
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()                                                            # a closed port: connection refused
    doc = manifest(toy)
    doc['llm'] = {'backend': 'openai-compat', 'model': 'm', 'base_url': f'http://127.0.0.1:{port}/v1', 'api_key_env': None, 'timeout_s': 5}
    write_manifest(toy, doc)
    res = cli.run('run', toy, '--strategy', 'hls')
    assert res.code == 2 and 'endpoint unreachable' in res.err and '--llm none' in res.err, res.text
    assert runs(toy) == []
    res = cli.run('doctor', toy)                                         # doctor says so too, and still runs the reference
    assert res.code == 0 and 'got no answer from' in res.out and 'reference run: OK' in res.out   # (U-3-12: not a refusal)


def test_run_goes_on_when_the_model_fails_after_the_preflight_and_says_so(toy):
    """A model that answers the preflight and then fails every call (an outage mid-run): the search goes on without its
    proposals and the report says how many calls failed."""
    with FakeLLM(lambda req: {'ok': True} if role_of(req) == 'preflight' else Reply(500, {'error': {'message': 'overloaded'}})) as llm:
        _with_llm(toy, llm, timeout_s=3)
        doc = manifest(toy)
        doc['search']['budget']['screens'] = 4
        write_manifest(toy, doc)
        res = cli.run('run', toy, '--strategy', 'hls')
        assert res.code == 0, res.text
        r = check_run(latest_run(toy), budget_screens=4)
        assert r['outcome'] in ('improved', 'no_change')
        assert r['search']['llm_calls'] >= 1 and r['search']['llm_errors'] and 'HTTP 500' in r['search']['llm_errors'][0]
        assert 'model calls failed' in report(latest_run(toy))


def test_a_scripted_model_drives_hls_to_the_gain_and_every_call_is_saved(toy):
    hs = [[hypothesis({'precision': 'bf16'}, 40, mechanism='precision', change_class='numerics'),
           hypothesis({'fused_optimizer': True}, 25, mechanism='optimizer'),
           hypothesis({'workers': 4}, 20, mechanism='data_workers')], []]
    with FakeLLM(by_role(hypotheses=hs)) as llm:
        _with_llm(toy, llm)
        res = cli.run('run', toy, '--strategy', 'hls', '--screens', '8')
        assert res.code == 0, res.text
        r = check_run(latest_run(toy), budget_screens=8)
        assert r['outcome'] == 'improved', r['promotion'].get('why')
        cfg = r['promotion']['config']
        assert cfg['precision'] == 'bf16' or cfg['fused_optimizer'] or cfg['workers'] == 4
        # the gain came from the model's hypotheses, not from the algorithmic neighbourhood (origin hls-explore)
        promoted = next(c for c in r['candidates'] if c['id'] == r['promotion']['promoted'])
        assert promoted.get('origin') in ('hls', 'hls-bundle'), promoted.get('origin')
        scripted = {'precision': 'bf16', 'fused_optimizer': True, 'workers': 4}
        changed = {k: v[1] for k, v in (promoted.get('change') or {}).items()}
        assert changed and all(scripted.get(k) == v for k, v in changed.items()), changed
        roles = [role_of(q) for q in llm.requests]
        assert roles[0] == 'preflight' and 'hypotheses' in roles
        assert r['search']['llm_calls'] >= 1 and r['cost']['llm_tokens'] > 0 and r['cost']['llm_usd_known'] is False
        calls = sorted((latest_run(toy) / 'llm_calls').glob('*/request.json'))
        assert len(calls) == r['search']['llm_calls'] and (calls[0].parent / 'receipt.json').is_file()
        ledger = r['search']['ledger'] or []
        assert any(h.get('mechanism') == 'precision' and h.get('status') in ('confirmed', 'bundled', 'refuted', 'unresolved', 'invalid')
                   for h in ledger), ledger


def test_proposals_that_break_the_contract_cost_no_run(toy, tmp_path):
    outside = tmp_path / 'evil.py'
    hs = [[hypothesis({'nope': 1}, 30),                                       # an unknown knob
           hypothesis({'grad_accum': 4}, 30),                                 # breaks the global-batch invariant
           hypothesis({'precision': 'int4'}, 30),                             # not a choice
           hypothesis({}, 30, patch={'loss.py': '# frozen file rewritten\n'}),      # a frozen file
           hypothesis({}, 30, patch={'../../../../../evil.py': 'x = 1\n'}),         # outside the tree
           hypothesis({}, 30, patch={'kernels.py/../loss.py': '# via dot-dot\n'}),
           hypothesis({'workers': 4}, 20)], []]                               # one honest proposal
    with FakeLLM(by_role(hypotheses=hs)) as llm:
        _with_llm(toy, llm)
        res = cli.run('run', toy, '--strategy', 'hls', '--screens', '6')
        assert res.code == 0, res.text
        r = check_run(latest_run(toy), budget_screens=6)
        changes = [c.get('change') or {} for c in r['candidates']]
        assert not any('nope' in c for c in changes) and not any(c.get('grad_accum') == [8, 4] for c in changes)
        assert not any(c.get('precision') == ['fp32', 'int4'] for c in changes)
        files = {f for c in r['candidates'] for f in c.get('overlay_files') or []}
        assert 'loss.py' not in files and not any('..' in f for f in files), files
        assert not outside.exists() and not list(tmp_path.glob('**/evil.py')), 'a patch escaped the candidate tree'
        assert (toy / 'loss.py').read_text(encoding='utf-8').startswith('"""Frozen'), 'the frozen file was rewritten'
        inv = r['search']['invalid_proposals'] or {}
        assert sum(inv.values()) >= 5, inv


def test_a_code_patch_from_the_review_is_measured_promoted_and_applied(toy, tmp_path):
    """The whole code-patch journey: the model's code review names a kernels.py edit, the patch is screened, it is the
    recommendation, `apply --write` writes it, and the applied job is faster when measured outside the search."""
    import yaml
    cfg = yaml.safe_load((toy / 'config.yaml').read_text(encoding='utf-8'))
    cfg['train']['fused_optimizer'] = True                           # knob moves alone cannot beat the kernel patch now
    (toy / 'config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
    doc = manifest(toy)
    doc['knobs']['workers']['default'] = 4
    write_manifest(toy, doc)
    old, new = 'return 0.02 * chunks * chunks / 4', 'return 0.02 * chunks / 4'
    review = {'verdict': 'patch', 'summary': 'the attention cost is quadratic in the chunk count',
              'findings': [{'file': 'kernels.py', 'issue': 'quadratic chunk cost', 'knob': None, 'patch': None,
                            'edits': [{'file': 'kernels.py', 'old': old, 'new': new}], 'predicted_gain_pct': 40,
                            'uncertainty_pct': 10, 'p_feasible': 0.95, 'change_class': 'execution'}]}
    with FakeLLM(by_role(review=review, hypotheses=[[]])) as llm:
        _with_llm(toy, llm)
        res = cli.run('run', toy, '--strategy', 'hls', '--screens', '6')
        assert res.code == 0, res.text
        r = check_run(latest_run(toy), budget_screens=6)
        assert (r['search'].get('code_review') or {}).get('verdict') == 'patch'
        patched = [c for c in r['candidates'] if 'kernels.py' in (c.get('overlay_files') or [])]
        assert patched and any(c.get('valid') for c in patched), [(c['id'], c.get('status'), c.get('reasons')) for c in patched]
        assert r['outcome'] == 'improved', r['promotion'].get('why')
        overlay = r['promotion'].get('overlay') or {}
        assert 'kernels.py' in overlay and new in overlay['kernels.py'], 'the recommendation is not the measured patch'
        assert r['promotion']['base'] == {'kernels.py': (toy / 'kernels.py').read_text(encoding='utf-8')}
        wr = cli.run('apply', latest_run(toy), '--write')
        assert wr.code == 0 and 'written:' in wr.out, wr.text
        assert new in (toy / 'kernels.py').read_text(encoding='utf-8')
        # the applied job, measured outside the search against the pre-patch kernel
        runner = ProjectRunner(Project.load(toy), out_dir=toy / '.infrarsi' / 'indep')
        p = Project.load(toy)
        task, v = p.task(), make_verifier(p)
        a = runner.run(task, p.reference_config(), {'kernels.py': overlay['kernels.py'].replace(new, old)}, 'rescore', 'i', candidate_id='before')
        b = runner.run(task, p.reference_config(), None, 'rescore', 'i', candidate_id='after')
        va, vb = v(a, None, a), v(b, a, a)
        assert va.valid and vb.valid and va.step_time / vb.step_time > 1.1, (va.step_time, vb.step_time)


def test_a_patch_path_with_dot_dot_cannot_escape_a_glob(toy, tmp_path):
    """With a glob such as `**/*.py` (the README's `src/kernels/*.py` behaves the same) a model's patch path that climbs
    out of the candidate tree must be rejected before any file is written. It is: the hooks' canonical-path check
    (guards.path_problem) runs before the glob match; the glob match and executor.own_file alone would not confine it
    (F-SYS-2, latent)."""
    doc = manifest(toy)
    doc['code']['mutable'] = ['**/*.py']
    write_manifest(toy, doc)
    marker = tmp_path / 'escaped.py'
    up = '../' * 12                                                    # far above the toy's run directories
    hs = [[hypothesis({}, 30, patch={f'{up}escaped.py': 'x = 1\n'}), hypothesis({'workers': 4}, 20)], []]
    with FakeLLM(by_role(hypotheses=hs)) as llm:
        _with_llm(toy, llm)
        res = cli.run('run', toy, '--strategy', 'hls', '--screens', '4')
        assert res.code == 0, res.text
        escaped = [p for p in tmp_path.rglob('escaped.py')] + [p for p in tmp_path.parent.glob('escaped.py')]
        r = result(latest_run(toy))
        files = {f for c in r['candidates'] for f in c.get('overlay_files') or []}
        assert not escaped and not any('..' in f for f in files), (escaped, files)
        assert not marker.exists()
