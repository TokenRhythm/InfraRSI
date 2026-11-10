import json
import math
import urllib.request

import pytest

from infrarsi.agent import Budget, FakeScreenRunner, VirtualClock
from infrarsi.agent.backends import ScriptedBackend
from infrarsi.agent.judge import ReceiptJudge
from infrarsi.baselines.claude_openai_shim import ShimServer, collapse_messages
from infrarsi.baselines.common import new_session, screen_start, space_for
from infrarsi.baselines.openevolve_adapter import EvalServer, EvalService, RemoteEvaluator, build_program, parse_program
from infrarsi.baselines.optico_style import optico as oc
from infrarsi.baselines.random_l0 import run_random
from infrarsi.baselines.space import config_to_trial_params, decode_layout, sample_config
from agent_helpers import DENSE_4B, FR_DENSE, FR_MOE, MOE_30B


def _run(fn, task, mf, start, **kw):
    clock = VirtualClock()
    return fn(task, Budget.from_task(task), start, FakeScreenRunner(task, mf, clock=clock), model_facts=mf, clock=clock,
              judge=ReceiptJudge(), **kw).to_dict()


def test_layouts_and_conditional_sampler_respect_constraints(tasks):
    from infrarsi.actions import validate_l0
    import random
    space = space_for(tasks['D4'], MOE_30B)
    assert space['__layout__'] and all(validate_l0(decode_layout(c), tasks['D4'], MOE_30B).ok for c in space['__layout__'])
    rng = random.Random(0)
    bad = 0
    for _ in range(300):
        _, cfg = sample_config(lambda n, ch: rng.choice(ch), space, MOE_30B)
        bad += not validate_l0(cfg, tasks['D4'], MOE_30B).ok
    assert bad == 0
    params, choices = config_to_trial_params(FR_MOE, space, MOE_30B)
    assert params['layout'] == 'tp1.pp1.cp1.ep8.etp1.mbs1' and all(params[k] in choices[k] for k in params)


def test_random_is_seeded_and_budgeted(tasks):
    a = _run(run_random, tasks['S2'], DENSE_4B, FR_DENSE)
    b = _run(run_random, tasks['S2'], DENSE_4B, FR_DENSE)
    assert [c['config_digest'] for c in a['candidates']] == [c['config_digest'] for c in b['candidates']]
    assert a['used']['screens'] <= 16 and a['arm'] == 'random_L0' and a['extra']['seed'] == 42
    assert a['stop_reason'].startswith(('budget:', 'same_failure'))


def test_tpe_runs_on_the_same_space(tasks):
    pytest.importorskip('optuna')
    from infrarsi.baselines.tpe_l0 import run_tpe
    tr = _run(run_tpe, tasks['S2'], DENSE_4B, FR_DENSE)
    assert tr['arm'] == 'tpe_L0' and tr['extra']['sampler'] == 'optuna.TPESampler' and tr['used']['screens'] <= 16
    assert tr['selected_step_time'] <= tr['start_step_time']


@pytest.mark.skipif(not oc.OPTICO_ROOT.exists(), reason='OptiCo-public not fetched')
def test_optico_materials_come_from_the_fetched_repo():
    m = oc.load_materials()
    assert set(m['wrappers']) == {'give_sugg', 'change_para', 'loc_error'}
    assert '{TABLE}' in m['texts']['give_sugg'] and 'Step 0' in m['texts']['give_sugg_cot']
    text = oc.compose(m, 'give_sugg', 'QUESTION-BODY')
    assert text.startswith(m['wrappers']['give_sugg']['instruction']) and 'QUESTION-BODY' in text and 'Step 0' in text
    assert all(len(v) == 64 for v in m['provenance'].values())


def test_optico_codec_roundtrip(tasks):
    codec = oc.ParamCodec(tasks['D4'], MOE_30B)
    lines = codec.to_lines(FR_MOE)
    assert 'EP: 8' in lines and 'AC: full' in lines and 'AC_LAYERS: 1' in lines
    d = dict(x.split(': ', 1) for x in lines.splitlines())
    from infrarsi.actions import validate_l0
    assert validate_l0(codec.to_config(d)).resolved == validate_l0(FR_MOE).resolved
    d2 = oc.parse_change_para('<think>..</think>\nThe modified parameters are:\nAC: none, TP:2\nBOGUS: 1', d)
    assert d2['AC'] == 'none' and d2['TP'] == '2' and 'BOGUS' not in d2


@pytest.mark.skipif(not oc.OPTICO_ROOT.exists(), reason='OptiCo-public not fetched')
def test_optico_style_loop_with_scripted_llm(tasks):
    def respond(role, system, prompt):
        assert system == oc.SYSTEM
        if role == 'optico_give_sugg':
            assert '|Result|StepTime_s|' in prompt
            return '<think>x</think>\nThe suggestions are:\nChange OVERLAP_GRAD_REDUCE from false to true. '
        if role == 'optico_change_para':
            cur = prompt.split('Current Configuration:')[1].split('Modification Suggestions:')[0]
            return '<think>y</think>\nThe modified parameters are:\n' + cur.replace('OVERLAP_GRAD_REDUCE: false', 'OVERLAP_GRAD_REDUCE: true')
        return 'The conclusion is:\nOOM (Confidence: 80)'
    clock = VirtualClock()
    t = tasks['S2']
    tr = oc.run_optico_style(t, Budget.from_task(t), FR_DENSE, FakeScreenRunner(t, DENSE_4B, clock=clock),
                             llm_backend=ScriptedBackend(respond, clock=clock), model_facts=DENSE_4B, clock=clock,
                             judge=ReceiptJudge()).to_dict()
    assert tr['arm'] == 'optico_style_L0' and tr['used']['screens'] >= 2 and tr['extra']['deviations'] == oc.DEVIATIONS
    assert tr['stop_reason'] == 'no_new_config' and tr['invalid_proposals'].get('duplicate', 0) >= 6


def test_openevolve_program_roundtrip_and_eval_service(tasks, pristine_overlays):
    t = tasks['S2']
    code = build_program('S2', {'tensor_model_parallel_size': 1, 'sequence_parallel': False, 'env': {'NCCL_ALGO': 'Ring'}}, {})
    cfg, ov = parse_program(code)
    assert cfg['env'] == {'NCCL_ALGO': 'Ring'} and ov == {}
    with pytest.raises(ValueError):
        parse_program('X = 1\n')
    clock = VirtualClock()
    session = new_session(task=t, budget=Budget.from_task(t, screens=4), runner=FakeScreenRunner(t, DENSE_4B, clock=clock), judge=ReceiptJudge(),
                          run_id='oe', arm='openevolve_L0L1', out_dir=None, start_config=FR_DENSE, start_receipt=None,
                          reference_receipt=None, model_facts=DENSE_4B, clock=clock, layers=('L0', 'L1'))
    screen_start(session)
    svc = EvalService(session)
    ok = svc.evaluate(build_program('S2', dict(FR_DENSE, overlap_grad_reduce=True), {}))
    assert ok['valid'] == 1.0 and math.isfinite(ok['combined_score'])
    assert svc.evaluate(build_program('S2', dict(FR_DENSE, sequence_parallel=True), {}))['combined_score'] == -1.0
    assert svc.evaluate('L0_CONFIG = {\n')['combined_score'] == -1.0
    evil = {'host_sync.py': 'import time\n\n\ndef on_train_start(ctx):\n    time.perf_counter = lambda: 0\n'}
    assert svc.evaluate(build_program('S2', FR_DENSE, evil))['combined_score'] == -1.0
    good_ov = {'host_sync.py': 'import gc\n\n\ndef on_train_start(ctx):\n    gc.disable()\n'}
    assert svc.evaluate(build_program('S2', FR_DENSE, good_ov))['valid'] in (0.0, 1.0)
    server = EvalServer(svc)
    try:
        import tempfile, pathlib
        p = pathlib.Path(tempfile.mkdtemp()) / 'prog.py'
        p.write_text(build_program('S2', dict(FR_DENSE, bucket_size=20000000), {}), encoding='utf-8')
        out = RemoteEvaluator(server.url)(str(p))
        assert 'combined_score' in out
        while not session.ledger.exhausted():
            session.ledger.charge_screen(0.0, 0.0)
        assert svc.evaluate(build_program('S2', dict(FR_DENSE, bucket_size=40000000), {}))['budget_exhausted'] == 1.0
    finally:
        server.stop()


def test_shim_serves_openai_format_with_accounting():
    seen, allow = [], {'ok': True}
    backend = ScriptedBackend(lambda role, s, p: 'diff text', model='claude-opus-5-5', tokens_per_call=1234, usd_per_call=0.5)
    shim = ShimServer(backend, usage_cb=seen.append, budget_check=lambda text: None if allow['ok'] else 'llm_tokens').start()
    try:
        body = {'model': 'claude-opus-5-5', 'temperature': 0.7, 'messages': [{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'U'}]}
        req = urllib.request.Request(shim.url + '/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        resp = json.loads(urllib.request.urlopen(req).read())
        assert resp['choices'][0]['message']['content'] == 'diff text' and resp['usage']['total_tokens'] == 1234
        assert resp['model'] == 'claude-opus-5-5' and len(seen) == 1 and seen[0].llm_usd == 0.5
        assert backend.calls[0]['system'] == 'S' and backend.calls[0]['prompt'] == 'U'
        models = json.loads(urllib.request.urlopen(shim.url + '/models').read())
        assert models['data'][0]['id'] == 'claude-opus-5-5'
        allow['ok'] = False
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(urllib.request.Request(shim.url + '/chat/completions', data=json.dumps(body).encode(),
                                                          headers={'Content-Type': 'application/json'}))
        assert e.value.code == 429 and len(seen) == 1
    finally:
        shim.stop()
    s, p = collapse_messages([{'role': 'system', 'content': 'A'}, {'role': 'user', 'content': 'B'}, {'role': 'assistant', 'content': 'C'},
                              {'role': 'user', 'content': 'D'}])
    assert s == 'A' and p.startswith('[user]\nB') and p.endswith('[user]\nD')
