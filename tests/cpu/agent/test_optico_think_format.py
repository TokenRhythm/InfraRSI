"""F-OPTICO (optico_style_L0) under D-OC-15 (DECISIONS_20261021 D7, AMENDMENT_EXT_001 Q5; builder X4).

Every request goes out through ext.optico.prompt_variants.strip_think_format, the drop-only rule of the B-NAT shim;
each call row records the lines dropped and the SHA-256 of the original and of the sent prompt. A refusal (Claude
stop_reason 'refusal', the shape the G-B smoke recorded) is charged, counted and re-sent unchanged, as OptiCo's
analyzer.api_infer re-sends a failed request; the third refusal in a run stops it (stop_reason 'refusals') and the
dispatcher arm keeps the evidence and fails loudly (escalation), without scoring.

OptiCo-public has no LICENSE: the prompt files here are synthetic stand-ins written by the test. One test runs the
fetched prompts exactly as optico_style reads them at run time and checks only counts and hashes.
"""
import dataclasses
import hashlib
import json
import re
import sys

import pytest

import f1_world as F1W
from agent_helpers import DENSE_4B, FIXTURES, FR_DENSE
from infrarsi.agent import Budget, FakeScreenRunner, VirtualClock
from infrarsi.agent.backends import ScriptedBackend, claude_cli, is_refusal
from infrarsi.agent.judge import ReceiptJudge
from infrarsi.baselines.optico_style import optico as oc
from infrarsi.ext.optico import prompt_variants as pv

THINK = re.compile(r'</?think>|think tags|reasoning goes here', re.I)        # the declared D-OC-15 line rule
GS, CP, LE = 'optico_give_sugg', 'optico_change_para', 'optico_loc_error'
DROPPED = {GS: 3, CP: 1, LE: 0}                    # think-demand lines in the synthetic prompts below, per request
REFUSE = 'REFUSE'
REFUSAL_TEXT = ("API Error: Opus 5.5's safeguards flagged this message (https://www.anthropic.com/legal/aup). "
                "Claude Code can't respond to this message with Opus 5.5.\n\nDetails: `[reasoning_extraction]`")
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
                  'Reason step by step inside <think> tags before answering.\n<think>your reasoning goes here</think>\n'
                  "End with 'The suggestions are:' and the suggestions."),
    'give_sugg_cot': 'Step 0: compare the rows of the table.\nStep 1: write all of it between think tags.\n',
    'change_para': ('Table:\n{TABLE}\nCurrent Configuration:\n{PARAMETERS}\nModification Suggestions:\n{SUGGESTIONS}\n'
                    'Parameter names: {PARAMETER_NAME}\nWrap your reasoning in <think></think> first.\n'
                    "End with 'The modified parameters are:' and KEY: VALUE lines."),
    'loc_error': "Log:\n{LOG}\nParameters:\n{PARAMETERS}\nEnd with 'The conclusion is:' and the cause.",
}


@pytest.fixture
def synthetic_root(tmp_path):
    root = tmp_path / 'OptiCo-synthetic'
    (root / 'prompt').mkdir(parents=True)
    for name, text in PROMPTS.items():
        (root / 'prompt' / f'{name}.txt').write_text(text, encoding='utf-8')
    (root / 'script' / 'analyzer').mkdir(parents=True)
    (root / 'script' / 'analyzer' / 'analyzer.py').write_text(ANALYZER, encoding='utf-8')
    return root


class RefusingBackend(ScriptedBackend):
    """ScriptedBackend; a 'REFUSE' reply comes back in the Claude CLI's refusal shape: error set, stop_reason
    'refusal' in meta, usage charged (as recorded in x3_gb_smoke_20261021a)."""

    def complete(self, system, prompt, **kw):
        res = super().complete(system, prompt, **kw)
        if res.text == REFUSE:
            return dataclasses.replace(res, text=REFUSAL_TEXT, error='CLI failed or incomplete result',
                                       meta={**res.meta, 'stop_reason': 'refusal', 'terminal_reason': 'api_error'})
        return res


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def _params(prompt: str) -> dict:
    cur = prompt.split('Current Configuration:\n')[1].split('\nModification Suggestions:')[0]
    return dict(line.split(': ', 1) for line in cur.strip().splitlines())


def responder(edits: list[dict], refuse=()):
    """give_sugg: a suggestion; change_para: the current parameters with edits[k] applied (k-th answered change_para
    call; afterwards unchanged = a duplicate); loc_error: a conclusion. (role, k) in refuse, k = the role's k-th call
    (1-based, re-sends included) -> REFUSE."""
    calls, answered = {}, {}

    def respond(role, system, prompt):
        assert system == oc.SYSTEM
        calls[role] = calls.get(role, 0) + 1
        if (role, calls[role]) in refuse:
            return REFUSE
        answered[role] = answered.get(role, 0) + 1
        if role == GS:
            return 'The suggestions are:\nChange the parameters as discussed.'
        if role == CP:
            p = _params(prompt)
            p.update(edits[answered[CP] - 1] if answered[CP] <= len(edits) else {})
            return 'The modified parameters are:\n' + '\n'.join(f'{k}: {v}' for k, v in p.items())
        return 'The conclusion is:\nthe configuration was rejected (Confidence: 90)'
    return respond


def _run(tasks, backend, root, monkeypatch=None, out_dir=None, budget=None, max_unproductive=2):
    clock = VirtualClock()
    t = tasks['S2']
    backend.clock = clock
    return oc.run_optico_style(t, budget or Budget(screens=6, llm_tokens=10 ** 6, wall_seconds=36000, gpu_hours=100), FR_DENSE,
                               FakeScreenRunner(t, DENSE_4B, clock=clock), llm_backend=backend, model_facts=DENSE_4B,
                               clock=clock, judge=ReceiptJudge(), optico_root=root, out_dir=out_dir,
                               max_unproductive=max_unproductive).to_dict()


def _spy(monkeypatch):
    seen = []

    def spy(text):
        out = pv.strip_think_format(text)
        seen.append((text, out[0]))
        return out
    monkeypatch.setattr(oc, 'strip_think_format', spy)
    return seen


def test_the_transform_is_b_nats_and_declared():
    assert oc.strip_think_format is pv.strip_think_format and pv.DEVIATION['strip_think_format'] == 'D-OC-15'
    entry = [d for d in oc.DEVIATIONS if d.startswith('D-OC-15 ')]
    assert len(entry) == 1 and 'strip_think_format' in entry[0] and 'n_refusals' in entry[0]


def test_every_request_is_sent_stripped_and_recorded(tasks, synthetic_root, monkeypatch):
    c0 = dict(x.split(': ', 1) for x in oc.ParamCodec(tasks['S2'], DENSE_4B).to_lines(FR_DENSE).splitlines())
    edits = [{'TP': '7'},                                                  # rejected before launch -> failed pool row
             {'TP': c0['TP'], 'OVERLAP_GRAD_REDUCE': 'true'}]              # a new configuration, screened
    seen = _spy(monkeypatch)
    backend = ScriptedBackend(responder(edits))
    tr = _run(tasks, backend, synthetic_root)
    roles = [c['role'] for c in backend.calls]
    assert roles == [GS, CP, LE, GS, CP, GS, CP, LE, GS, CP]            # loc_error after each failed row (rejected, duplicate)
    assert tr['stop_reason'] == 'no_new_config' and tr['used']['screens'] == 2
    rows = tr['llm_calls']
    assert len(rows) == len(seen) == len(backend.calls)
    for row, call, (original, sent) in zip(rows, backend.calls, seen):
        assert call['prompt'] == sent == '\n'.join(x for x in original.split('\n') if not THINK.search(x))
        assert not any(THINK.search(x) for x in call['prompt'].split('\n'))
        assert row['prompt_transform'] == 'strip_think_format' and row['lines_dropped'] == DROPPED[row['role']]
        assert len(original.split('\n')) - len(sent.split('\n')) == DROPPED[row['role']]
        assert row['prompt_sha256_original'] == _sha(original) and row['prompt_sha256_sent'] == _sha(call['prompt'])
        assert row['refusal'] is False and row['attempt'] == 1
    assert 'Question:' in backend.calls[0]['prompt'] and "The suggestions are:'" in backend.calls[0]['prompt']
    assert tr['extra']['n_refusals'] == 0 and tr['extra']['prompt_transform'] == {
        'name': 'strip_think_format', 'deviation': 'D-OC-15', 'requests': 10, 'lines_dropped': 4 * 3 + 4 * 1 + 2 * 0}
    assert any(d.startswith('D-OC-15 ') for d in tr['extra']['deviations'])


def test_a_refusal_is_charged_counted_and_resent_unchanged(tasks, synthetic_root, tmp_path):
    backend = RefusingBackend(responder([{'OVERLAP_GRAD_REDUCE': 'true'}], refuse={(GS, 1)}))
    tr = _run(tasks, backend, synthetic_root, out_dir=tmp_path / 'run', max_unproductive=1)
    roles = [c['role'] for c in backend.calls]
    assert roles == [GS, GS, CP, GS, CP] and backend.calls[0]['prompt'] == backend.calls[1]['prompt']
    walls = [e['wall'] for e in F1W.events(tmp_path / 'run') if e['kind'] == 'llm_call']
    assert walls[1] - walls[0] == pytest.approx(oc.RETRY_WAIT_S + backend.seconds_per_call)   # api_infer's 10 s wait
    assert walls[2] - walls[1] == pytest.approx(backend.seconds_per_call)                     # no wait otherwise
    rows = tr['llm_calls']
    assert (rows[0]['refusal'], rows[0]['attempt'], rows[1]['refusal'], rows[1]['attempt']) == (True, 1, False, 2)
    assert rows[0]['error'] and rows[1]['error'] is None
    assert {k: rows[0][k] for k in ('prompt_sha256_original', 'prompt_sha256_sent', 'lines_dropped')} == \
           {k: rows[1][k] for k in ('prompt_sha256_original', 'prompt_sha256_sent', 'lines_dropped')}
    assert sum(r['attempt'] > 1 for r in rows) == 1                         # the one re-send, nothing else retried
    assert tr['extra']['n_refusals'] == 1 and tr['extra']['prompt_transform']['requests'] == 4
    assert tr['used']['llm_tokens'] == 5 * backend.tokens_per_call          # the refused call is paid for
    assert tr['stop_reason'] == 'no_new_config' and tr['used']['screens'] == 2


@pytest.mark.parametrize('refuse, roles, screens', [
    ({(GS, 1), (GS, 2), (GS, 3)}, [GS, GS, GS], 1),                        # the same request three times
    ({(GS, 1), (CP, 1), (GS, 3)}, [GS, GS, CP, CP, GS], 2),                # spread over agents and iterations
])
def test_the_third_refusal_stops_the_run(tasks, synthetic_root, tmp_path, refuse, roles, screens):
    backend = RefusingBackend(responder([{'OVERLAP_GRAD_REDUCE': 'true'}], refuse=refuse))
    tr = _run(tasks, backend, synthetic_root, out_dir=tmp_path / 'run')
    assert [c['role'] for c in backend.calls] == roles
    assert tr['stop_reason'] == 'refusals' and tr['extra']['n_refusals'] == 3 and tr['used']['screens'] == screens
    assert sum(r['refusal'] for r in tr['llm_calls']) == 3
    stops = [e['payload'] for e in F1W.events(tmp_path / 'run') if e['kind'] == 'stop']
    assert stops == [{'reason': 'refusals', 'n_refusals': 3, 'escalate': True}]


def _fake_popen(stream: str, exit_code: int):
    class FakePopen:
        def __init__(self, argv, stdin=None, stdout=None, stderr=None, cwd=None, env=None, **kw):
            self.pid, self.returncode, self._out = 4242, None, stdout

        def communicate(self, data=None, timeout=None):
            self._out.write(stream.encode('utf-8'))
            self.returncode = exit_code
            return b'', b''
    return FakePopen


def test_the_cli_backend_reports_a_refusal(tmp_path, monkeypatch):
    """The stream shape of the refused give_sugg calls (x3_gb_smoke_20261021a): is_error, stop_reason 'refusal',
    exit 1, usage and cost present. The call fails, is charged in full, and is_refusal() sees it."""
    model = 'claude-opus-5-5'
    usage = {'input_tokens': 4, 'cache_creation_input_tokens': 12475, 'cache_read_input_tokens': 0, 'output_tokens': 0}
    events = [{'type': 'system', 'subtype': 'init', 'model': model, 'tools': [], 'mcp_servers': [], 'skills': [],
               'slash_commands': [], 'agents': [], 'plugins': []},
              {'type': 'assistant', 'message': {'model': model, 'stop_reason': 'refusal',
                                                'content': [{'type': 'text', 'text': REFUSAL_TEXT}]}},
              {'type': 'result', 'subtype': 'success', 'is_error': True, 'stop_reason': 'refusal', 'terminal_reason': 'api_error',
               'num_turns': 2, 'result': REFUSAL_TEXT, 'total_cost_usd': 0.099816, 'usage': usage,
               'modelUsage': {model: {'inputTokens': 4}}, 'permission_denials': []}]
    b = claude_cli.ClaudeCLIBackend(executable=sys.executable, max_output_tokens=1000, timeout_seconds=60)
    b2 = claude_cli.ClaudeCLIBackend(executable=sys.executable, model='claude-haiku-5-5', timeout_seconds=60)
    monkeypatch.setattr(claude_cli.subprocess, 'Popen', _fake_popen('\n'.join(json.dumps(e) for e in events) + '\n', 1))
    r = b.complete('system', 'prompt', call_dir=tmp_path / 'c1', expect_json=False)
    assert r.error == 'CLI failed or incomplete result' and is_refusal(r) and '[reasoning_extraction]' in r.text
    assert (r.llm_tokens, r.llm_usd, r.accounting_complete) == (12479, 0.099816, True)
    assert json.loads((tmp_path / 'c1' / 'receipt.json').read_text())['stop_reason'] == 'refusal'
    haiku = (FIXTURES / 'stream_haiku_5_5_real.jsonl').read_text(encoding='utf-8')
    monkeypatch.setattr(claude_cli.subprocess, 'Popen', _fake_popen(haiku, 0))
    ok = b2.complete('system', 'prompt', call_dir=tmp_path / 'c2')
    assert ok.error is None and not is_refusal(ok)


# ---------------------------------------------------------------- the dispatcher arm (F-OPTICO on the F1 simulation)

RUN = {'run_id': 'XB-F1-F-OPTICO-sim', 'task': 'S1', 'arm': 'optico_style_L0', 'screens': 12, 'wall_hours_cap': 12,
       'llm_tokens_cap': 1500000, 'seed': 42, 'priority': 'P1', 'promotion_lifecycle': False}


@pytest.fixture
def world(tmp_path, monkeypatch, synthetic_root):
    logs, slept = [], []
    real = oc.load_materials
    monkeypatch.setattr(oc, 'load_materials', lambda root=None: real(synthetic_root))
    monkeypatch.setattr(oc, '_sleep', slept.append)                  # a real run sleeps RETRY_WAIT_S before a re-send
    for w in F1W.world_setup(tmp_path, monkeypatch, log=logs.append):
        w.update(logs=logs, slept=slept)
        yield w


def test_arm_summary_carries_n_refusals(world, monkeypatch):
    backend = ScriptedBackend(responder([{'MICRO_BATCH_SIZE': '2'}]))
    out = F1W.dispatch(world, backend, monkeypatch, RUN)
    res = out['result']
    assert world['slept'] == []
    rows = json.loads((world['ctx'].search_dir(RUN['run_id']) / 'trace.json').read_text(encoding='utf-8'))['llm_calls']
    assert res['optico']['n_refusals'] == 0 and res['optico']['llm_calls'] == len(rows) == len(backend.calls)
    assert res['optico']['prompt_transform'] == {'name': 'strip_think_format', 'deviation': 'D-OC-15', 'requests': len(rows),
                                                 'lines_dropped': sum(DROPPED[r['role']] for r in rows)}
    assert res['locked_pair']['promoted'] and res['promotion_lifecycle']['ran'] is False      # MBS 2 is faster here
    assert json.loads((world['ctx'].search_dir(RUN['run_id']) / 'result.json').read_text())['optico'] == res['optico']


def test_three_refusals_fail_the_arm_loudly_with_the_evidence_kept(world, monkeypatch):
    backend = RefusingBackend(lambda role, system, prompt: REFUSE)
    with pytest.raises(RuntimeError, match='3 LLM refusals despite D-OC-15'):
        F1W.dispatch(world, backend, monkeypatch, RUN)
    sdir = world['ctx'].search_dir(RUN['run_id'])
    res = json.loads((sdir / 'result.json').read_text(encoding='utf-8'))
    assert res['trace']['stop_reason'] == 'refusals' and res['optico']['n_refusals'] == 3 and len(backend.calls) == 3
    assert world['slept'] == [oc.RETRY_WAIT_S] * 2                                         # before each re-send
    assert 'locked_pair' not in res and F1W.requests(world, RUN['run_id']) == []           # nothing screened or scored
    assert any(m.startswith('ESCALATE ') for m in world['logs'])


# ---------------------------------------------------------------- the fetched prompts, as optico_style reads them

@pytest.mark.skipif(not oc.OPTICO_ROOT.exists(), reason='OptiCo-public not fetched')
def test_the_fetched_give_sugg_prompt_loses_its_think_lines(tasks, monkeypatch):
    seen = _spy(monkeypatch)

    def respond(role, system, prompt):
        if role == GS:
            return 'The suggestions are:\nChange OVERLAP_GRAD_REDUCE from false to true.'
        if role == CP:
            cur = prompt.split('Current Configuration:')[1].split('Modification Suggestions:')[0]
            return 'The modified parameters are:\n' + cur.replace('OVERLAP_GRAD_REDUCE: false', 'OVERLAP_GRAD_REDUCE: true')
        return 'The conclusion is:\nOOM (Confidence: 80)'
    backend = ScriptedBackend(respond)
    tr = _run(tasks, backend, oc.OPTICO_ROOT, max_unproductive=1)
    gs = [r for r in tr['llm_calls'] if r['role'] == GS]
    assert gs and all(r['lines_dropped'] >= 1 for r in gs)                   # D-OC-15 acts on the real give_sugg request
    for row, call, (original, sent) in zip(tr['llm_calls'], backend.calls, seen):
        assert call['prompt'] == sent and not any(THINK.search(x) for x in sent.split('\n'))
        assert row['prompt_sha256_original'] == _sha(original) and row['prompt_sha256_sent'] == _sha(sent)
    assert tr['extra']['n_refusals'] == 0 and tr['used']['screens'] >= 2
