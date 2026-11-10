"""Strategy optico in a project run: OptiCo's four agents (baselines.optico_style) on the manifest's knobs through
ProjectHooks, on the flow's session - the reference run reused as its start and first pool row, configurations proposed
through the knob codec and screened under the run's budget, the promotion's locked pair, REPORT.md and the experience
write-back; the knob codec on the knobs of the recorded Qwen3.5-9B HF FSDP2 SFT job (Q9U), its empty-string env value
included; the fixed-parameter block on that manifest; the errors before any run (no model, a model that only answers
JSON, OptiCo's prompts missing, an unknown strategy); the wall budget near its end (NM27: no request, a refusal's
re-send included, starts without the time left as its timeout or with less than MIN_CALL_WALL_S left).

OptiCo-public has no license: the prompt files here are synthetic stand-ins written by the test (as in
tests/cpu/agent/test_optico_think_format.py), read through INFRARSI_OPTICO_ROOT."""
import copy
import dataclasses
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from infrarsi import search
from infrarsi.agent import VirtualClock
from infrarsi.agent.backends import OpenAICompatBackend
from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.agent.optimize import MIN_CALL_WALL_S
from infrarsi.baselines.optico_style import optico as oc
from infrarsi.cli.main import main
from infrarsi.project import Project, ProjectRunner, llm
from infrarsi.project.flow import make_verifier, patch_status, run_project
from infrarsi.project.hooks import ProjectHooks
from infrarsi.project.manifest import env_for
from infrarsi.project.report import _stop_words, _truncated
from infrarsi.search import optico

TOY = Path(__file__).resolve().parents[3] / 'examples' / 'toy_cpu'
GS, CP, LE = 'optico_give_sugg', 'optico_change_para', 'optico_loc_error'
THINK = re.compile(r'</?think>|think tags', re.I)
ANALYZER = '''
class ChangePrompt:
    def give_sugg(self, cot_path=None):
        instruction = "Synthetic give_sugg instruction.\\n"
        user_prompt = f"Question:\\n{self.input_text}"

    def change_para(self, cot=None):
        instruction = "Synthetic change_para instruction.\\n"
        user_prompt = f"Question:\\n{self.input_text}"

    def loc_error(self, cot=None):
        instruction = "Synthetic loc_error instruction.\\n"
        user_prompt = f"Question:\\n{self.input_text}"
'''
PROMPTS = {
    'give_sugg': ('Fixed parameters:\n{FIXED_PARA}\nPerformance table:\n{TABLE}\nCurrent parameters:\n{PARAMETERS}\n'
                  'Reason step by step inside <think> tags before answering.\n'
                  "End with 'The suggestions are:' and the suggestions."),
    'give_sugg_cot': 'Step 0: compare the rows of the table.\n',
    'change_para': ('Table:\n{TABLE}\nCurrent Configuration:\n{PARAMETERS}\nModification Suggestions:\n{SUGGESTIONS}\n'
                    'Parameter names: {PARAMETER_NAME}\n'
                    "End with 'The modified parameters are:' and KEY: VALUE lines."),
    'loc_error': "Log:\n{LOG}\nParameters:\n{PARAMETERS}\nEnd with 'The conclusion is:' and the cause.",
}
# the toy's reference: micro_batch_size 1, grad_accum 8, fused_optimizer false, workers 1 (global batch 8). A larger
# micro-batch alone changes the global batch (refused before any run); then two faster configurations are screened.
EDITS = [{'micro_batch_size': '2'},
         {'micro_batch_size': '1', 'fused_optimizer': 'true'},
         {'workers': '4'}]
# the knobs and facts of the recorded Q9U job (scripts pc/nm/Q9U-opus2.yaml), its paths and hosts replaced
Q9U = yaml.safe_load(r'''
schema: infrarsi-project-v1
name: q9-like
framework: hf_trainer
launch:
  command: '{python} -m torch.distributed.run --nproc_per_node {gpus} --master_port 29640 train_hf.py --output_dir out
    --save_strategy no --max_steps {steps} --logging_steps 1 --disable_tqdm true --report_to none --learning_rate 1e-5
    --bf16 true --fsdp "full_shard auto_wrap" --model_path /models/Qwen3.5-9B --packed /data/packed/sft-8k-s4096
    --seq_len 4096 {args}'
  timeout_s: 900
config: {file: null}
knobs:
  optim: {type: str, flag: --optim, choices: [adamw_torch, adamw_torch_fused], default: adamw_torch}
  use_liger_kernel: {type: bool, flag: --use_liger_kernel, style: value, default: false}
  torch_compile: {type: bool, flag: --torch_compile, style: value, default: false}
  gradient_checkpointing: {type: bool, flag: --gradient_checkpointing, style: value, default: true}
  tf32: {type: bool, flag: --tf32, style: value, default: false}
  dataloader_num_workers: {type: int, flag: --dataloader_num_workers, choices: [0, 2, 4, 8], default: 2}
  dataloader_pin_memory: {type: bool, flag: --dataloader_pin_memory, style: value, default: true}
  per_device_train_batch_size: {type: int, flag: --per_device_train_batch_size, choices: [1, 2], default: 1}
  gradient_accumulation_steps: {type: int, flag: --gradient_accumulation_steps, choices: [2, 1], default: 2}
  fsdp_config: {type: str, flag: --fsdp_config, choices: [fsdp_v2.json, fsdp_v2_noreshard.json], default: fsdp_v2.json}
  PYTORCH_CUDA_ALLOC_CONF: {type: str, env: PYTORCH_CUDA_ALLOC_CONF, choices: ['', 'expandable_segments:True'], default: ''}
  NCCL_NVLS_ENABLE: {type: str, env: NCCL_NVLS_ENABLE, choices: ['2', '0'], default: '2'}
constraints: []
code: {mutable: [], frozen: [train_hf.py, fsdp_v2.json, fsdp_v2_noreshard.json], readable: []}
workload:
  steps: 30
  window: [11, 30]
  invariants: {global_batch: per_device_train_batch_size * gradient_accumulation_steps * gpus}
  seq_len: 4096
model: {params_b: 8.95, layers: 32, hidden: 4096, ffn: 12288, heads: 16, kv_heads: 4, vocab: 248320}
hardware: {gpus: 8, memory_gib: 141, gpu_type: NVIDIA L20X (H200-class)}
verify: {tolerance: 0.03}
search: {strategy: hls, budget: {screens: 10, wall_hours: 3}}
llm: {backend: claude-cli, model: opus}
execution: {where: ssh, hosts: [node0], python: /opt/envs/hf/bin/python, remote_dir: /opt/remote}
''')


def write_root(path: Path) -> Path:
    (path / 'prompt').mkdir(parents=True)
    for name, text in PROMPTS.items():
        (path / 'prompt' / f'{name}.txt').write_text(text, encoding='utf-8')
    (path / 'script' / 'analyzer').mkdir(parents=True)
    (path / 'script' / 'analyzer' / 'analyzer.py').write_text(ANALYZER, encoding='utf-8')
    return path


def responder(edits: list[dict]):
    """give_sugg: a suggestion; change_para: the current parameters with the k-th edit applied (unchanged after the
    last: a duplicate); loc_error: a conclusion."""
    n = {'cp': 0}

    def respond(role, system, prompt):
        if role == GS:
            return 'The suggestions are:\nChange the parameters as discussed.'
        if role == CP:
            n['cp'] += 1
            cur = dict(x.split(': ', 1) for x in
                       prompt.split('Current Configuration:\n')[1].split('\nModification Suggestions:')[0].splitlines())
            cur.update(edits[n['cp'] - 1] if n['cp'] <= len(edits) else {})
            return 'The modified parameters are:\n' + '\n'.join(f'{k}: {v}' for k, v in cur.items())
        return 'The conclusion is:\nmicro_batch_size 2 changed the global batch (Confidence: 90)'
    return respond


def table(prompt: str) -> list[dict]:
    """The rows of the performance table (OptiCo's markdown) in a prompt."""
    rows = [x.strip('|').split('|') for x in prompt.splitlines() if x.startswith('|')]
    return [dict(zip(rows[0], r)) for r in rows[2:]]


@pytest.fixture
def toy(tmp_path, monkeypatch):
    monkeypatch.setenv('TOY_TIME_SCALE', '0.004')
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))     # this run's ledger rows and experience pack
    monkeypatch.setenv(optico.ROOT_ENV, str(write_root(tmp_path / 'OptiCo-synthetic')))
    d = tmp_path / 'toy'
    shutil.copytree(TOY, d, ignore=shutil.ignore_patterns('.infrarsi', '__pycache__'))
    doc = yaml.safe_load((d / 'infrarsi.yaml').read_text(encoding='utf-8'))
    doc['workload'] = {'steps': 12, 'window': [5, 12], 'invariants': {'global_batch': 'micro_batch_size * grad_accum'},
                       'seq_len': 512}
    doc['model'] = {'params_b': 0.05}                     # the experience write-back needs the size, the length and
    doc['hardware']['gpu_type'] = 'toy-cpu'               # the GPU type
    doc['checkpoint']['save_at'] = 6
    (d / 'infrarsi.yaml').write_text(yaml.safe_dump(doc), encoding='utf-8')
    return d


def _manifest(d: Path, **changes) -> None:
    doc = yaml.safe_load((d / 'infrarsi.yaml').read_text(encoding='utf-8'))
    for path, value in changes.items():
        a, b = path.split('__')
        doc[a][b] = value
    (d / 'infrarsi.yaml').write_text(yaml.safe_dump(doc), encoding='utf-8')


@pytest.mark.parametrize('how', ['flag', 'manifest'])
def test_a_project_runs_optico(toy, tmp_path, how):
    if how == 'flag':
        _manifest(toy, workload__global_batch_tokens=8 * 512)     # tokens per update known: TokensPerSec is filled
    else:
        _manifest(toy, search__strategy='optico')
    drv, logs = ScriptedBackend(responder(EDITS)), []
    res = run_project(Project.load(toy), strategy='optico' if how == 'flag' else None, driver=drv, budget={'screens': 2},
                      out_root=tmp_path / 'out', log=logs.append)
    run = Path(res['run_dir'])
    trace = json.loads((run / 'search' / 'trace.json').read_text(encoding='utf-8'))
    assert res['strategy'] == 'optico' and trace['arm'] == 'optico' and trace['stop_reason'] == 'budget:screens'
    # OptiCo's agents: loc_error after the refused proposal, nothing sent with its think-format lines (D-OC-15)
    assert [c['role'] for c in drv.calls] == [GS, CP, LE, GS, CP, GS, CP]
    assert not any(THINK.search(c['prompt']) for c in drv.calls)
    assert [r['role'] for r in trace['llm_calls']] == [c['role'] for c in drv.calls]
    # the reference run is the start and the first pool row: never screened again, both screens are proposals
    c0 = trace['candidates'][0]
    assert c0['id'] == 'c0' and c0['attempt_id'] == res['reference']['attempt_id'] and c0['step_time'] == res['reference']['step_time']
    assert res['search']['used']['screens'] == 2 and [s['candidate_id'] for s in trace['screens']] == ['c1', 'c2']
    assert all(s['fidelity'] == 'full' and s['valid'] for s in trace['screens'])
    # proposed through the knob codec, on the reference; the refused one is a failed pool row and no candidate
    assert [(c['id'], c['parent'], c['origin'], c['change']) for c in trace['candidates'][1:]] == [
        ('c1', 'c0', 'optico', {'fused_optimizer': [False, True]}),
        ('c2', 'c0', 'optico', {'fused_optimizer': [False, True], 'workers': [1, 4]})]
    assert trace['invalid_proposals'] == {'l0_invalid': 1}
    gs = [c['prompt'] for c in drv.calls if c['role'] == GS]
    fixed = gs[0].split('Fixed parameters:\n')[1].split('\nPerformance table:')[0].splitlines()
    for want in ('Model Size: 0.05B', 'Global Batch Size: 8', 'Sequence Length: 512', 'Number of Nodes: 1', 'Number of GPUs: 1',
                 'GPU Type: toy-cpu', 'GPU memory limit per card: 16384 MiB (16 GiB)',
                 'Updates per run: 12 (the step time is the mean over updates 5-12)',
                 'Must not change: global_batch = micro_batch_size * grad_accum = 8',
                 'micro_batch_size: 1 | 2 | 4 | 8', 'fused_optimizer: true | false', 'precision: fp32 | bf16 | fp8'):
        assert want in fixed, want
    assert not any(x.startswith('Precison') for x in fixed)       # the toy searches its precision: no fixed one
    assert ('Tokens per Update: 4096' in fixed) == (how == 'flag')
    rows = table(gs[2])
    assert [r['Result'] for r in rows] == ['success', 'fail', 'success'] and rows[1]['Log Director'] == '2'
    assert rows[1]['Reasons for failure'].startswith('micro_batch_size 2 changed the global batch')   # loc_error's cause
    for r, cid in ((rows[0], 'c0'), (rows[2], 'c1')):
        t = next(c['step_time'] for c in trace['candidates'] if c['id'] == cid)
        assert float(r['StepTime_s']) == round(t, 4)
        assert r['TokensPerSec'] == (str(round(4096 / t, 1)) if how == 'flag' else 'NULL')
        assert r['MFU'] == r['GpuUtilAve'] == 'NULL'
        if r['MemMaxGiB'] != 'NULL':
            assert r['MemUsage%'] == str(round(100 * float(r['MemMaxGiB']) / 16, 1))
    ex = trace['extra']
    assert ex['deviations'] == oc.DEVIATIONS_PROJECT and ex['pool_rows'] == 4 and ex['n_refusals'] == 0
    assert ex['pool_header'][:6] == list(Project.load(toy).knobs)
    assert set(ex['optico_provenance']) == {'prompt/give_sugg.txt', 'prompt/give_sugg_cot.txt', 'prompt/change_para.txt',
                                            'prompt/loc_error.txt', 'script/analyzer/analyzer.py'}
    # the locked pair: the fastest candidate re-run back to back with the reference, then promoted
    tried = res['promotion']['tried']
    assert tried[0]['candidate'] == 'c2' and len(tried[0]['rescore']) == 2
    assert res['outcome'] == 'improved' and res['promotion']['promoted'] == 'c2'
    # the run's log, report and experience pack
    assert any(m.startswith('[infrarsi] run 1 of 2: c1: valid') for m in logs)
    assert any(m.startswith("[infrarsi] optico: OptiCo's prompts from") for m in logs)
    assert any('code patches off: strategy optico changes knobs only' in m for m in logs)
    report = (run / 'REPORT.md').read_text(encoding='utf-8')
    assert 'Strategy `optico`: 2 full runs (2 valid' in report and 'LLM: 7 calls' in report
    assert res['experience']['written'] and Path(res['experience']['written']).is_file()


@pytest.mark.skipif(not oc.OPTICO_ROOT.exists(), reason='OptiCo-public not fetched')
def test_the_fetched_prompts_run_on_a_project(toy, tmp_path, monkeypatch):
    """OptiCo's own prompt files (read where the package keeps them, INFRARSI_OPTICO_ROOT unset): the fixed-parameter
    block and the knob lines go into the real give_sugg / change_para texts, D-OC-15 drops their think-format lines."""
    monkeypatch.delenv(optico.ROOT_ENV)

    def respond(role, system, prompt):
        if role == GS:
            return 'The suggestions are:\nChange fused_optimizer from false to true.'
        if role == CP:
            block = prompt.split('Current Configuration:')[1].split('Modification Suggestions:')[0]
            cur = dict(x.split(': ', 1) for x in block.splitlines() if ': ' in x)
            return 'The modified parameters are:\n' + '\n'.join(f'{k}: {v}' for k, v in {**cur, 'fused_optimizer': 'true'}.items())
        return 'The conclusion is:\nOOM (Confidence: 80)'
    drv = ScriptedBackend(respond)
    res = run_project(Project.load(toy), strategy='optico', driver=drv, budget={'screens': 1}, out_root=tmp_path / 'out',
                      log=lambda m: None)
    trace = json.loads((Path(res['run_dir']) / 'search' / 'trace.json').read_text(encoding='utf-8'))
    assert [(c['id'], c['change']) for c in trace['candidates'][1:]] == [('c1', {'fused_optimizer': [False, True]})]
    assert trace['screens'][0]['candidate_id'] == 'c1' and trace['screens'][0]['valid']
    gs = next(c['prompt'] for c in drv.calls if c['role'] == GS)
    assert 'Global Batch Size: 8' in gs and 'micro_batch_size: 1 | 2 | 4 | 8' in gs and 'fused_optimizer: false' in gs
    assert not any(THINK.search(c['prompt']) for c in drv.calls)
    assert all(r['lines_dropped'] >= 1 for r in trace['llm_calls'] if r['role'] == GS)
    assert set(trace['extra']['optico_provenance']) >= {'prompt/give_sugg.txt', 'script/analyzer/analyzer.py'}


class Clocked(ScriptedBackend):
    """ScriptedBackend on the search session's virtual clock: keeps each call's keywords, then spends the wall time
    `spend` gives the k-th call (1-based); the calls in `refuse` come back in the Claude CLI's refusal shape."""

    def __init__(self, script, clock, spend, refuse=()):
        super().__init__(script)
        self.vclock, self.spend, self.refuse, self.kw = clock, dict(spend), set(refuse), []

    def complete(self, system, prompt, **kw):
        self.kw.append(kw)
        res = super().complete(system, prompt, **kw)
        self.vclock.advance(self.spend.get(len(self.kw), 0.0))
        if len(self.kw) in self.refuse:
            return dataclasses.replace(res, text='refused', error='CLI failed or incomplete result',
                                       meta={**res.meta, 'stop_reason': 'refusal'})
        return res


@pytest.mark.parametrize('case', ['request', 'refusal re-send'])
def test_no_request_starts_near_the_end_of_the_wall_budget(toy, tmp_path, monkeypatch, case):
    """NM27 (Codex P2): with seconds of the wall budget left a request still went out, with the backend's own 900 s
    timeout. Now every request - a refusal's re-send too - gets the wall time left as its timeout, none starts with less
    than MIN_CALL_WALL_S left, and the search stops on the wall budget."""
    wall, clock = 600.0, VirtualClock()
    monkeypatch.setattr('infrarsi.agent.search.time', SimpleNamespace(monotonic=clock))   # the search session's clock
    monkeypatch.setattr(oc, '_sleep', clock.advance)                                        # the wait before a re-send
    left = 50.0 if case == 'request' else 35.0          # wall time left when the third request (give_sugg) starts
    drv = Clocked(responder([{'fused_optimizer': 'true'}]), clock, {2: wall - left, 3: 25.0 if case == 'request' else 0.0},
                  refuse={3} if case == 'refusal re-send' else ())
    res = run_project(Project.load(toy), strategy='optico', driver=drv, budget={'screens': 3, 'wall_seconds': wall},
                      out_root=tmp_path / 'out', log=lambda m: None)
    run = Path(res['run_dir'])
    trace = json.loads((run / 'search' / 'trace.json').read_text(encoding='utf-8'))
    assert [s['candidate_id'] for s in trace['screens']] == ['c1']
    # 25 s left after the give_sugg (or after the refused one and the 10 s wait): the next request is not sent
    assert [c['role'] for c in drv.calls] == [GS, CP, GS] and 25.0 < MIN_CALL_WALL_S
    assert [k['timeout'] for k in drv.kw] == [wall, wall, left]          # the wall time left, not the backend's default
    assert all(k['max_output_tokens'] > 0 and 'token_allowance' in k and 'max_usd' not in k for k in drv.kw)
    assert trace['stop_reason'] == res['search']['stop_reason'] == 'budget:wall_seconds'
    assert trace['used']['wall_seconds'] == wall - 25.0
    stops = [e['payload'] for e in map(json.loads, (run / 'search' / 'events.jsonl').read_text(encoding='utf-8').splitlines())
             if e['kind'] == 'stop']
    assert stops == [{'reason': 'budget:wall_seconds', 'request': CP if case == 'request' else GS,
                      'attempt': 1 if case == 'request' else 2}]
    assert trace['extra']['n_refusals'] == (1 if case == 'refusal re-send' else 0)
    assert 'stop: the wall-clock budget was used (budget:wall_seconds)' in (run / 'REPORT.md').read_text(encoding='utf-8')


def test_without_a_session_the_requests_keep_the_recorded_admission(toy, tmp_path):
    """run_optico_style without a caller's session (as the recorded L0 and F-OPTICO arms run): the output cap stays the
    only budget keyword of a request - no timeout, no token allowance - on a project's hooks too."""
    project = Project.load(toy)
    task, verifier = project.task(), make_verifier(project)
    runner = ProjectRunner(project, out_dir=tmp_path / 'attempts', log=lambda m: None)
    ref = runner.run(task, project.reference_config(), None, 'reference', 'r', candidate_id='c0')
    drv = Clocked(responder([{'fused_optimizer': 'true'}]), VirtualClock(), {})
    tr = oc.run_optico_style(task, task.search_budget(screens=1), project.reference_config(), runner, llm_backend=drv,
                             judge=verifier, run_id='r', start_receipt=ref, reference_receipt=ref,
                             hooks=ProjectHooks(project, task), optico_root=optico.root(), clock=VirtualClock())
    assert [c['role'] for c in drv.calls] == [GS, CP] and tr.stop_reason == 'budget:screens'
    assert {tuple(sorted(k)) for k in drv.kw} == {('call_dir', 'expect_json', 'max_output_tokens', 'role')}
    assert tr.extra['deviations'] == oc.DEVIATIONS_HOOKS


def test_the_knob_codec_round_trips_the_q9u_knobs(tmp_path):
    project = Project(copy.deepcopy(Q9U), tmp_path / 'infrarsi.yaml')
    task = project.task()
    codec, hooks, ref = oc.KnobCodec(task), ProjectHooks(project, task), project.reference_config()
    text = codec.to_lines(ref)
    assert codec.names() == list(project.knobs) and len(codec.names()) == 12
    assert "PYTORCH_CUDA_ALLOC_CONF: ''" in text.splitlines() and 'NCCL_NVLS_ENABLE: 2' in text.splitlines()
    lines = dict(x.split(': ', 1) for x in text.splitlines())
    assert codec.to_config(lines) == ref
    for name, knob in project.knobs.items():                # every choice, written and read back with its type
        for v in knob.get('choices') or [True, False]:
            back = codec.to_config(dict(x.split(': ', 1) for x in codec.to_lines({**ref, name: v}).splitlines()))[name]
            assert back == v and type(back) is type(v), (name, v, back)
    # what a model may write for the empty value, through OptiCo's own reply parser: the reference again (a duplicate
    # of c0 for the session), never None (which would re-run the reference as a new configuration)
    for written in ('', "''", '""', 'None', 'null'):
        reply = 'The modified parameters are:\n' + text.replace("PYTORCH_CUDA_ALLOC_CONF: ''", f'PYTORCH_CUDA_ALLOC_CONF: {written}')
        cfg = codec.to_config(oc.parse_change_para(reply, lines))
        assert cfg == ref and hooks.validate_config(cfg) == hooks.validate_config(ref), written
    reply = ('The modified parameters are:\n' + text.replace("PYTORCH_CUDA_ALLOC_CONF: ''", 'PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True')
             .replace('NCCL_NVLS_ENABLE: 2', "NCCL_NVLS_ENABLE: '0'").replace('dataloader_num_workers: 2', 'dataloader_num_workers: 4'))
    cfg = codec.to_config(oc.parse_change_para(reply, lines))
    merged, errors, _ = hooks.validate_config(cfg)
    assert not errors and {k: v for k, v in merged.items() if v != ref[k]} == {
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'NCCL_NVLS_ENABLE': '0', 'dataloader_num_workers': 4}
    env = env_for(project, merged)
    assert env['PYTORCH_CUDA_ALLOC_CONF'] == 'expandable_segments:True' and env['NCCL_NVLS_ENABLE'] == '0'
    assert env_for(project, codec.to_config(lines))['PYTORCH_CUDA_ALLOC_CONF'] == ''      # the reference's value
    # a value outside the choices is kept as written, for the validator to refuse
    assert codec.parse('fsdp_config', 'fsdp_v3.json') == 'fsdp_v3.json' and hooks.validate_config({'fsdp_config': 'fsdp_v3.json'})[1]


def test_json_and_comma_values_survive_optico_s_reply_parser(tmp_path):
    """G4U's gradient_checkpointing_kwargs choices are JSON; a choice with a comma is written with / (change_para splits
    replies on commas)."""
    gck = ['{"every_n_layers": 1}', '{"every_n_layers": 2}']
    alloc = ['', 'expandable_segments:True,max_split_size_mb:512']
    doc = {'name': 'g', 'launch': {'command': 'python t.py {args}'},
           'knobs': {'gck': {'type': 'str', 'flag': '--gck', 'choices': gck, 'default': gck[0]},
                     'alloc': {'type': 'str', 'env': 'PYTORCH_CUDA_ALLOC_CONF', 'choices': alloc, 'default': ''}}}
    task = Project(doc, tmp_path / 'infrarsi.yaml').task()
    codec = oc.KnobCodec(task)
    text = codec.to_lines({'gck': gck[1], 'alloc': alloc[1]})
    assert text.splitlines() == ['gck: {"every_n_layers": 2}', 'alloc: expandable_segments:True/max_split_size_mb:512']
    assert codec.to_config(oc.parse_change_para('The modified parameters are:\n' + text, {'gck': '', 'alloc': ''})) == \
        {'gck': gck[1], 'alloc': alloc[1]}
    assert codec.domain_text().splitlines() == ['gck: {"every_n_layers": 1} | {"every_n_layers": 2}',
                                                "alloc: '' | expandable_segments:True/max_split_size_mb:512",
                                                '(a comma inside a value is written /)']


def test_fixed_parameters_text_on_the_q9u_manifest(tmp_path):
    project = Project(copy.deepcopy(Q9U), tmp_path / 'infrarsi.yaml')
    hooks = ProjectHooks(project, project.task())
    text = hooks.fixed_parameters_text()
    lines = text.splitlines()
    for want in ('Model Size: 8.95B', 'NUM_LAYERS: 32', 'Hidden Size: 4096', 'FFN Hidden Size: 12288', 'Attention Heads: 16',
                 'KV Heads: 4', 'Vocabulary Size: 248320', 'Precison: bf16', 'Global Batch Size: 16', 'Sequence Length: 4096',
                 'Number of Nodes: 1', 'Number of GPUs: 8', 'GPU Type: NVIDIA L20X (H200-class)',
                 'GPU memory limit per card: 144384 MiB (141 GiB)', 'Framework: hf_trainer',
                 'Updates per run: 30 (the step time is the mean over updates 11-30)',
                 'Must not change: global_batch = per_device_train_batch_size * gradient_accumulation_steps * gpus = 16',
                 'Validity: a run counts only if it completes its 30 updates of the same workload with the loss within '
                 '3.0% of the reference run (the current configuration)',
                 'Parameters you may change and their allowed values:',
                 'optim: adamw_torch | adamw_torch_fused', 'use_liger_kernel: true | false', 'dataloader_num_workers: 0 | 2 | 4 | 8',
                 'gradient_accumulation_steps: 2 | 1', 'fsdp_config: fsdp_v2.json | fsdp_v2_noreshard.json',
                 "PYTORCH_CUDA_ALLOC_CONF: '' | expandable_segments:True", 'NCCL_NVLS_ENABLE: 2 | 0'):
        assert want in lines, want
    assert lines.index('Parameters you may change and their allowed values:') == len(lines) - 13
    # facts only: no model name it does not give, no tokens per update it does not give, no word on which value is good
    assert not any(x.startswith(('Model Type', 'Tokens per Update', 'Constraints')) for x in lines)
    assert not re.search(r'(?i)default|recommend|faster|slower|speed|best|measured|experience|gain', text)
    assert hooks.tokens_per_update() is None and hooks.gpu_memory_gib() == 141.0
    doc = copy.deepcopy(Q9U)
    doc['workload']['global_batch_tokens'] = 65536
    doc['states'] = {'precision': 'bf16'}
    doc['hardware'].pop('memory_gib')
    doc['execution']['hosts'] = ['node0', 'node1']
    hooks = ProjectHooks(Project(doc, tmp_path / 'infrarsi.yaml'), Project(doc, tmp_path / 'infrarsi.yaml').task())
    lines = hooks.fixed_parameters_text().splitlines()
    assert 'Tokens per Update: 65536' in lines and hooks.tokens_per_update() == 65536 and hooks.gpu_memory_gib() is None
    assert 'Number of Nodes: 2' in lines and 'Number of GPUs: 16 (8 per node)' in lines
    assert 'Global Batch Size: 16' in lines and not any(x.startswith('GPU memory') for x in lines)   # the invariant as
    assert 'Precison: bf16' in lines                                                    # written: `gpus` is per node


def test_errors_before_any_run(tmp_path, monkeypatch, capsys):
    with pytest.raises(ValueError, match="unknown search strategy 'optco'.*'optico'"):
        search.resolve('optco', has_llm=True)
    assert search.resolve('optico', has_llm=True) == 'optico' and 'optico' in search.LLM_STRATEGIES
    assert search.resolve(None, has_llm=True, knobs=Q9U['knobs']) == 'doe+hls'       # auto never picks the baseline
    drv = ScriptedBackend(lambda role, system, prompt: 'x')
    # the model: none, one that reads every reply as JSON, an OpenAI-compatible one in JSON mode
    assert 'needs an LLM backend' in optico.problem(None)
    assert 'anthropic backend' in optico.backend_problem(SimpleNamespace(kind='anthropic_api'))
    kw = dict(base_url='http://127.0.0.1:9/v1', model='m', api_key_env=None, price=None)
    assert 'llm.structured: none' in optico.backend_problem(OpenAICompatBackend(**kw))
    assert optico.backend_problem(OpenAICompatBackend(**kw, structured='none')) is None and optico.backend_problem(drv) is None
    # OptiCo's prompts: missing at INFRARSI_OPTICO_ROOT (the strategy itself refuses before it opens a session)
    monkeypatch.setenv(optico.ROOT_ENV, str(tmp_path / 'nowhere'))
    assert optico.root() == tmp_path / 'nowhere' and optico.ROOT_ENV in optico.problem(drv)
    ctx = SimpleNamespace(task=SimpleNamespace(knob_schema={'properties': {'k': {'type': 'boolean'}}}), driver=drv, options={})
    with pytest.raises(ValueError, match="cannot read them from .*nowhere.*INFRARSI_OPTICO_ROOT"):
        search.run('optico', ctx)
    with pytest.raises(ValueError, match='changes knobs only'):
        search.run('optico', SimpleNamespace(task=SimpleNamespace(knob_schema={}), driver=drv, options={}))
    monkeypatch.setenv(optico.ROOT_ENV, str(write_root(tmp_path / 'OptiCo-synthetic')))
    assert optico.problem(drv) is None
    monkeypatch.delenv(optico.ROOT_ENV)
    assert optico.root() == oc.OPTICO_ROOT
    # the CLI says so before the reference run costs its GPU time
    toy = tmp_path / 'toy'
    shutil.copytree(TOY, toy, ignore=shutil.ignore_patterns('.infrarsi', '__pycache__'))
    ran = []
    monkeypatch.setattr('infrarsi.project.flow.run_project', lambda *a, **k: ran.append(1))
    with pytest.raises(SystemExit) as e:
        main(['run', str(toy), '--strategy', 'optico'])
    assert e.value.code == 2 and 'the optico strategy needs an LLM backend' in capsys.readouterr().err
    monkeypatch.setenv(optico.ROOT_ENV, str(tmp_path / 'nowhere'))
    monkeypatch.setattr(llm, 'make_driver', lambda spec, **k: drv)
    with pytest.raises(SystemExit) as e:
        main(['run', str(toy), '--strategy', 'optico'])
    assert e.value.code == 2 and optico.ROOT_ENV in capsys.readouterr().err and not ran and not drv.calls


def test_the_report_and_the_patch_status_name_optico_s_stops(tmp_path):
    assert _stop_words('no_new_config', 10).startswith('OptiCo proposed nothing new') and _stop_words('refusals', 10)
    note = _truncated({'search': {'used': {'screens': 3}, 'budget': {'screens': 10}, 'stop_reason': 'no_new_config'}})
    assert note and note[0].startswith('Only 3 of the 10 budgeted full runs were made - OptiCo proposed nothing new')
    project = Project.load(TOY)
    assert patch_status(project, 'optico', 0.9, True) == 'off: strategy optico changes knobs only'
    assert patch_status(project, 'hls', 0.9, True).startswith('on: the search may edit kernels.py')     # as before


def test_live_note_reaches_optico_model_requests(toy, tmp_path, monkeypatch):
    from infrarsi.project.live import LiveConsole
    def enter(console):
        console.submit('/note Prefer reducing communication before compilation')
        return console
    monkeypatch.setattr(LiveConsole, '__enter__', enter)
    drv = ScriptedBackend(responder([{'workers': '4'}]))
    result = run_project(Project.load(toy), strategy='optico', driver=drv, budget={'screens': 1},
                         out_root=tmp_path / 'out', experience='off', log=lambda _: None, live=True)
    calls = [c for c in drv.calls if c['role'] in (GS, CP)]
    assert calls and all('Prefer reducing communication before compilation' in c['prompt'] for c in calls)
    events = (Path(result['run_dir']) / 'search/events.jsonl').read_text()
    assert 'live_message' in events
