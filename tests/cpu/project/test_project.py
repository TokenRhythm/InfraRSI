"""infrarsi.project on CPU: manifest, signals, runner (real subprocesses of the toy trainer), hooks, the one flow,
apply, init detection, and the Anthropic API backend with a fake client."""
import json
import shutil
import types
from pathlib import Path

import pytest
import yaml

from infrarsi.project import Project, ProjectRunner
from infrarsi.project import detect, signals
from infrarsi.project.apply import plan
from infrarsi.project.flow import make_verifier, run_project
from infrarsi.project.hooks import ProjectHooks
from infrarsi.project.manifest import ManifestError, evaluate

TOY = Path(__file__).resolve().parents[3] / 'examples' / 'toy_cpu'


@pytest.fixture
def toy(tmp_path, monkeypatch):
    monkeypatch.setenv('TOY_TIME_SCALE', '0.004')
    d = tmp_path / 'toy'
    shutil.copytree(TOY, d, ignore=shutil.ignore_patterns('.infrarsi', '__pycache__'))
    doc = yaml.safe_load((d / 'infrarsi.yaml').read_text(encoding='utf-8'))
    doc['workload'] = {'steps': 12, 'window': [5, 12], 'invariants': {'global_batch': 'micro_batch_size * grad_accum'}}
    doc['checkpoint']['save_at'] = 6
    (d / 'infrarsi.yaml').write_text(yaml.safe_dump(doc), encoding='utf-8')
    return Project.load(d)


# ---------------------------------------------------------------- manifest
def test_manifest_defaults_and_problems(tmp_path):
    with pytest.raises(ManifestError, match='launch.command'):
        Project({'name': 'x'}, tmp_path / 'infrarsi.yaml')
    with pytest.raises(ManifestError, match='exactly one target'):
        Project({'name': 'x', 'launch': {'command': 'true'}, 'knobs': {'a': {'type': 'int', 'choices': [1]}}},
                tmp_path / 'infrarsi.yaml')
    p = Project({'name': 'x', 'launch': {'command': 'true'}}, tmp_path / 'infrarsi.yaml')
    assert p.window == [11, 30] and p.tolerance == 0.03
    with pytest.raises(ManifestError, match='nothing to search'):
        p.task()


def test_expressions():
    assert evaluate('gpus % (tp * pp) == 0', {'gpus': 8, 'tp': 2, 'pp': 2}) is True
    assert evaluate('mbs * ga * gpus / (tp * pp)', {'mbs': 2, 'ga': 4, 'gpus': 8, 'tp': 2, 'pp': 1}) == 32
    for bad in ('__import__("os")', 'open("x")', 'a.b'):
        with pytest.raises(ManifestError):
            evaluate(bad, {'a': 1})


def test_reference_config_schema_invariants(toy):
    ref = toy.reference_config()
    assert ref == {'micro_batch_size': 1, 'grad_accum': 8, 'fused_optimizer': False, 'workers': 1, 'precision': 'fp32',
                   'compile': False}
    assert toy.invariants(ref) == {'global_batch': 8}
    t = toy.task()
    assert t.action_space == 'knobs+patch' and list(t.frozen_files) == ['loss.py'] and t.mutable_globs == ('kernels.py',)
    hooks = ProjectHooks(toy, t)
    cfg, errs, _ = hooks.validate_config({'micro_batch_size': 2, 'grad_accum': 4})
    assert not errs and cfg['workers'] == 1
    _, errs, _ = hooks.validate_config({'grad_accum': 4})
    assert errs and 'workload invariant changed' in errs[0]
    _, errs, _ = hooks.validate_config({'nope': 1})
    assert 'unknown knobs' in errs[0]
    assert 'loss must stay within 3.0%' in hooks.contract_text()


# ---------------------------------------------------------------- signals
def test_log_scanner_formats():
    lines = [(1.0, ' iteration        1/      30 | elapsed time per iteration (ms): 1234.5 | learning rate: 1.000000E-05 | '
                   'global batch size:    64 | lm loss: 2.345678E+00 | loss scale: 1.0 | grad norm: 1.234 |'),
             (2.0, 'step:  2  loss:  2.3100  memory: 12.34GiB(15.6%)  tps: 1234'),
             (3.0, 'iter 3: loss 2.2801, time 123.45ms'),
             (4.0, 'step=4 loss=nan lr=1e-4 tokens_per_step=8192 tokens/s=1234 data_digest=deadbeefcafe'),
             (5.0, 'validation loss at iteration 4 | lm loss value: 9.9')]
    rows = signals.LogParser(ignore=r'(?i)validation').parse(lines)
    assert rows[0] == {'step': 1, 't': 1.0, 'loss': 2.345678, 'grad_norm': 1.234, 'lr': 1e-05, 'samples': 64, 'step_time': 1.2345}
    assert rows[1]['mem_gib'] == 12.34 and rows[2]['loss'] == 2.2801
    assert rows[3]['loss'] == 'nan' and rows[3]['tokens'] == 8192 and rows[3]['data'] == 'deadbeefcafe'
    hf = signals.LogParser(count_steps=True).parse([(1.0, "{'loss': 2.3, 'grad_norm': 1.1}"), (2.0, "{'loss': 2.2}")])
    assert [r['step'] for r in hf] == [1, 2] and hf[0]['grad_norm'] == 1.1


def test_clock_and_buffering():
    rows = [{'step': s, 't': 10.0 + 0.5 * s} for s in range(1, 11)]
    t, rsd, note = signals.clock_time(rows, [4, 10])
    assert t == pytest.approx(0.5) and note is None
    burst = [{'step': s, 't': 100.0 + (5.0 if s == 10 else 1e-5 * s)} for s in range(1, 11)]
    t, _, note = signals.clock_time(burst, [4, 10])
    assert t is None and 'bursts' in note
    ev = signals.evidence([{'step': 0, 't': 1.0, 'loss': 2.0}, {'step': 1, 't': 2.0, 'loss': 1.9}], steps=2, window=[1, 2],
                          exit_code=0, timed_out=False, events={}, frozen={})
    assert ev['steps_completed'] == 2 and ev['step_base'] == 0


# ---------------------------------------------------------------- runner + verifier on the toy trainer
def test_runner_and_verifier(toy, tmp_path):
    t = toy.task()
    r = ProjectRunner(toy, out_dir=tmp_path / 'runs')
    v = make_verifier(toy)
    ref = r.run(t, toy.reference_config(), None, 'reference', 'r', candidate_id='c0')
    assert v(ref, None, ref).valid and ref['evidence']['steps_completed'] == 12
    fast = r.run(t, dict(toy.reference_config(), fused_optimizer=True, workers=4), None, 'screen', 'r', candidate_id='c1')
    vf = v(fast, ref, ref)
    assert vf.valid and vf.step_time < v(ref, None, ref).step_time
    lossy = r.run(t, dict(toy.reference_config(), precision='fp8'), None, 'screen', 'r', candidate_id='c2')
    vl = v(lossy, ref, ref)
    assert not vl.valid and 'loss_within_tolerance' in vl.feedback
    assert make_verifier(toy, tolerance=0.05)(lossy, ref, ref).valid
    oom = r.run(t, dict(toy.reference_config(), micro_batch_size=8, grad_accum=1), None, 'screen', 'r', candidate_id='c3')
    vo = v(oom, ref, ref)
    assert not vo.valid and {c['id'] for c in vo.checks if c['status'] == 'fail'} >= {'completes', 'no_errors'}
    patched = (toy.root / 'kernels.py').read_text(encoding='utf-8').replace('chunks * chunks / 4', 'chunks / 2')
    p = r.run(t, toy.reference_config(), {'kernels.py': patched}, 'screen', 'r', candidate_id='c4')
    assert v(p, ref, ref).valid and p['overlay'] == {'kernels.py': patched}


def test_runtime_write_into_frozen_file_is_caught(toy, tmp_path):
    t = toy.task()
    r = ProjectRunner(toy, out_dir=tmp_path / 'runs')
    ref = r.run(t, toy.reference_config(), None, 'reference', 'r', candidate_id='c0')
    evil = (toy.root / 'kernels.py').read_text(encoding='utf-8') + \
        "\nopen(__file__.replace('kernels.py', 'loss.py'), 'a').write('\\n# touched\\n')\n"
    bad = r.run(t, toy.reference_config(), {'kernels.py': evil}, 'screen', 'r', candidate_id='c1')
    v = make_verifier(toy)(bad, ref, ref)
    assert not v.valid and 'frozen_files' in v.feedback


# ---------------------------------------------------------------- the flow, report, apply
def test_run_project_random_then_apply(toy, tmp_path):
    res = run_project(toy, strategy='random', budget={'screens': 6}, out_root=tmp_path / 'out', log=lambda m: None)
    assert res['outcome'] in ('improved', 'no_change') and res['search']['used']['screens'] == 6
    report = (Path(res['run_dir']) / 'REPORT.md').read_text(encoding='utf-8')
    assert '## Recommendation' in report and '## Search' in report
    if res['outcome'] == 'improved':
        items = {i['id']: i['status'] for i in res['promotion']['tried'][-1]['items']}
        assert items == {'speed_reproduces': 'pass', 'resume_continues': 'pass', 'eval_matches': 'pass', 'user_tests': 'pass'}
        p = plan(res['run_dir'])
        assert p['files'] or p['manual']
    saved = json.loads((Path(res['run_dir']) / 'result.json').read_text(encoding='utf-8'))
    assert saved['run_id'] == res['run_id']


def test_hls_without_a_job_size_spends_the_budget_and_says_so(toy, tmp_path):
    # no `search.job_gpu_hours`: no value/cost stop, the budget ends the search (tests/cpu/search/test_early_stop.py)
    assert not toy.raw['search'].get('job_gpu_hours')
    res = run_project(toy, strategy='hls', budget={'screens': 2}, out_root=tmp_path / 'out', log=lambda m: None)
    s = res['search']
    assert s['job_gpu_hours'] is None and s['stop_rule'] == 'off' and s['stop_reason'] == 'budget:screens'
    report = (Path(res['run_dir']) / 'REPORT.md').read_text(encoding='utf-8')
    assert 'no value/cost stop: `search.job_gpu_hours`' in report


def test_reference_failure_stops_before_search(toy, tmp_path):
    raw = dict(toy.raw)
    raw['launch'] = dict(raw['launch'], command='{python} -c "import sys; sys.exit(3)"')
    p = Project(raw, toy.path)
    res = run_project(p, strategy='random', out_root=tmp_path / 'out', log=lambda m: None)
    assert res['outcome'] == 'reference_failed' and 'completes' in res['feedback']


# ---------------------------------------------------------------- init
def test_detect_draft(tmp_path):
    (tmp_path / 'train.py').write_text(
        'import argparse\nimport torch\nap = argparse.ArgumentParser()\n'
        "ap.add_argument('--config')\nap.add_argument('--max-steps', type=int, default=100)\n"
        "ap.add_argument('--micro-batch-size', type=int, default=2)\nap.add_argument('--grad-accum', type=int, default=4)\n"
        "ap.add_argument('--compile', action='store_true')\nap.add_argument('--num-workers', type=int, default=2)\n",
        encoding='utf-8')
    (tmp_path / 'cfg.yaml').write_text('model: {gradient_checkpointing: true, dim: 512}\n', encoding='utf-8')
    m, notes = detect.draft(tmp_path, config='cfg.yaml')
    assert m['framework'] == 'pytorch'
    assert {'micro_batch_size', 'grad_accum', 'compile', 'num_workers', 'gradient_checkpointing'} <= set(m['knobs'])
    assert m['knobs']['gradient_checkpointing'] == {'type': 'bool', 'path': 'model.gradient_checkpointing'}
    assert '--max-steps {steps}' in m['launch']['command'] and '--config {config}' in m['launch']['command']
    assert m['workload']['invariants']['global_batch'] == 'micro_batch_size * grad_accum * gpus'
    out = detect.write(tmp_path, m, notes)
    assert Project.load(out).knobs.keys() == m['knobs'].keys()
    with pytest.raises(FileExistsError):
        detect.write(tmp_path, m, notes)


# ---------------------------------------------------------------- Anthropic API backend (fake client)
def test_anthropic_backend_parses_and_prices():
    from infrarsi.agent.backends import AnthropicBackend
    calls = []

    def create(**kw):
        calls.append(kw)
        usage = types.SimpleNamespace(input_tokens=1000, output_tokens=200, cache_read_input_tokens=0,
                                      cache_creation_input_tokens=0, iterations=None)
        return types.SimpleNamespace(content=[types.SimpleNamespace(type='text', text='{"action": "finish"}')],
                                     stop_reason='end_turn', model='claude-opus-5-5', usage=usage)

    client = types.SimpleNamespace(beta=types.SimpleNamespace(messages=types.SimpleNamespace(create=create)),
                                   with_options=lambda **k: client)
    b = AnthropicBackend(client=client)
    res = b.complete('sys', 'prompt', schema={'type': 'object', 'required': ['action']})
    assert res.action == {'action': 'finish'} and res.error is None and res.accounting_complete
    assert res.llm_tokens == 1200 and res.llm_usd == pytest.approx((1000 * 4 + 200 * 20) / 1e6)
    kw = calls[0]
    assert kw['fallbacks'] == 'default' and kw['betas'] == ['server-side-fallback-2026-07-01']
    assert kw['output_config'] == {'effort': 'high'} and kw['model'] == 'claude-opus-5-5' and 'temperature' not in kw

    def refuse(**kw):
        return types.SimpleNamespace(content=[], stop_reason='refusal', model='claude-opus-5-5', usage=None,
                                     stop_details=types.SimpleNamespace(category='cyber'))
    client.beta.messages.create = refuse
    res = b.complete('sys', 'prompt')
    assert res.action is None and res.error.startswith('refusal')
