"""The code review before the first hypotheses (10-23 code trials: the model looks at the code and decides whether the
search patches it at all) and the patch shapes models send (trial C1: valid edits were rejected because the same
hypothesis carried a malformed `patch`)."""
import json

from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.ext.common.session_hooks import SessionHooks
from infrarsi.project.flow import _code_review
from infrarsi.project.report import _code_review as review_lines
from infrarsi.search.proposers import LLMProposer, patch_of

from test_search import TOGGLE, setup

FILES = {'step.py': 'x = 1\ntorch.cuda.synchronize()\ny = 2\n', 'model.py': "attn = 'eager'\n"}
SYNC = {'file': 'step.py', 'old': 'torch.cuda.synchronize()\n', 'new': ''}
FIXED = {'step.py': 'x = 1\ny = 2\n'}


def _review(verdict, findings, summary='s'):
    return {'verdict': verdict, 'summary': summary, 'findings': findings}


def test_patch_shapes_models_send():
    # C1: correct edits next to a malformed whole-file `patch`: the edits are the patch
    assert patch_of({'edits': [SYNC], 'patch': {'file': 'step.py', 'patch': 'x = 1\n'}}, FILES) == (FIXED, True)
    assert patch_of({'patch': {'file': 'step.py', 'old': SYNC['old'], 'new': ''}}, FILES) == (FIXED, True)
    assert patch_of({'patch': {'file': 'model.py', 'new': "attn = 'sdpa'\n"}}, FILES) == ({'model.py': "attn = 'sdpa'\n"}, True)
    assert patch_of({'patch': {'model.py': "attn = 'sdpa'\n"}}, FILES) == ({'model.py': "attn = 'sdpa'\n"}, True)
    assert patch_of({'edits': [{'file': 'step.py', 'old': 'missing', 'new': ''}]}, FILES) == (None, False)
    assert patch_of({'patch': {}, 'edits': []}, FILES) == (None, True)


def test_review_queues_patches_and_keeps_the_code():
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    logs = []
    eng.log = logs.append
    review = _review('patch', [
        {'file': 'step.py', 'issue': 'host-device sync after every micro-batch', 'knob': 'None', 'edits': [SYNC],
         'predicted_gain_pct': 8},                                          # C1b: Qwen3's "no knob"
        {'file': 'model.py', 'issue': 'attention chosen in code', 'knob': 'flash_attn'},
        {'file': 'step.py', 'issue': 'stale', 'knob': None, 'edits': [{'file': 'step.py', 'old': 'zzz', 'new': ''}]}])
    hyp = {'hypotheses': [{'mechanism': 'fused', 'change': {'fused_adam': True}, 'predicted_gain_pct': 3}]}
    drv = ScriptedBackend(lambda role, system, prompt: review if role == 'code_review' else hyp)
    prop = LLMProposer(drv, SessionHooks(s.task, base_files=dict(FILES)))
    hyps = prop.hypotheses(eng, 'c0')
    assert [c['role'] for c in drv.calls] == ['code_review', 'hypotheses']
    assert json.loads(drv.calls[0]['prompt'])['code']['mutable'] == FILES
    assert hyps[0]['mechanism'] == 'review: step.py: host-device sync after every micro-batch'
    assert hyps[0]['patch'] == FIXED and hyps[0]['change'] == {} and hyps[1]['mechanism'] == 'fused'
    second = json.loads(drv.calls[1]['prompt'])
    assert second['code']['mutable'] == FILES and second['code_review']['verdict'] == 'patch'
    cr = prop.describe()['code_review']
    assert [f['patch'] for f in cr['findings']] == ['queued', None, 'edit did not apply'] and prop.unusable_edits == 1
    assert logs[0].startswith('[search] code review: patch, 3 findings, 1 patches queued (step.py)')
    prop.hypotheses(eng, 'c0')
    assert [c['role'] for c in drv.calls].count('code_review') == 1          # once per search


def test_review_no_patch_leaves_the_search_to_knobs():
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    hyp = {'hypotheses': [{'mechanism': 'both', 'change': {'fused_adam': True}, 'predicted_gain_pct': 3,
                           'edits': [{'file': 'step.py', 'old': 'x = 1\n', 'new': 'x = 2\n'}]}]}
    drv = ScriptedBackend(lambda role, system, prompt: _review('no_patch', []) if role == 'code_review' else hyp)
    prop = LLMProposer(drv, SessionHooks(s.task, base_files=dict(FILES)))
    hyps = prop.hypotheses(eng, 'c0') + prop.hypotheses(eng, 'c0')
    assert [c['role'] for c in drv.calls] == ['code_review', 'hypotheses', 'hypotheses']
    assert all('code' not in json.loads(c['prompt']) and 'CODE:' not in c['system'] for c in drv.calls[1:])
    assert [h['patch'] for h in hyps] == [None, None] and hyps[0]['change'] == {'fused_adam': True}


def test_review_without_a_usable_answer_keeps_the_code_visible():
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    drv = ScriptedBackend(lambda role, system, prompt: 'not json' if role == 'code_review' else {'hypotheses': []})
    prop = LLMProposer(drv, SessionHooks(s.task, base_files=dict(FILES)))
    prop.hypotheses(eng, 'c0')
    assert prop.describe()['code_review']['verdict'] is None and 'code' in json.loads(drv.calls[1]['prompt'])


def test_patch_of_keeps_other_files_and_leaves_out_unchanged_ones():
    files = {'kernel.py': 'from helper import slow\nslow()\n', 'helper.py': 'def slow():\n    pass\n'}
    edit = {'file': 'kernel.py', 'old': 'from helper import slow\nslow()\n', 'new': 'from helper import fast\nfast()\n'}
    both = {'kernel.py': 'from helper import fast\nfast()\n', 'helper.py': 'def fast():\n    pass\n'}
    assert patch_of({'edits': [edit], 'patch': {'helper.py': 'def fast():\n    pass\n'}}, files) == (both, True)  # QA13 F4
    assert patch_of({'edits': [edit], 'patch': {'file': 'helper.py', 'old': 'slow', 'new': 'fast'}}, files) == (both, True)
    assert patch_of({'edits': [{'file': 'kernel.py', 'old': 'slow()', 'new': 'slow()'}]}, files) == (None, True)   # C4b


def test_a_no_patch_verdict_queues_nothing_and_unchanged_findings_are_not_queued():
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    finding = {'file': 'step.py', 'issue': 'sync', 'knob': None, 'edits': [SYNC], 'predicted_gain_pct': 10}
    drv = ScriptedBackend(lambda role, system, prompt: _review('no_patch', [finding]) if role == 'code_review'
                          else {'hypotheses': []})
    prop = LLMProposer(drv, SessionHooks(s.task, base_files=dict(FILES)))
    assert prop.hypotheses(eng, 'c0') == [] and prop.describe()['code_review']['findings'][0]['patch'] is None  # QA13 F2
    same = {'file': 'step.py', 'old': 'y = 2\n', 'new': 'y = 2\n'}
    review = _review('patch', [{'file': 'step.py', 'issue': 'copies', 'knob': None, 'edits': [same]}])
    hyp = {'hypotheses': [{'mechanism': 'noop', 'change': {}, 'edits': [same], 'predicted_gain_pct': 4}]}
    drv = ScriptedBackend(lambda role, system, prompt: review if role == 'code_review' else hyp)
    prop = LLMProposer(drv, SessionHooks(s.task, base_files=dict(FILES)))
    assert prop.hypotheses(eng, 'c0') == [] and prop.describe()['code_review']['findings'][0]['patch'] == 'no change'


def test_untested_review_patches_follow_the_incumbent():
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    review = _review('patch', [{'file': 'step.py', 'issue': 'sync', 'knob': None, 'edits': [SYNC], 'predicted_gain_pct': 8}])
    drv = ScriptedBackend(lambda role, system, prompt: review if role == 'code_review' else {'hypotheses': []})
    prop = LLMProposer(drv, SessionHooks(s.task, base_files=dict(FILES)))
    first = prop.hypotheses(eng, 'c0')
    c0 = s.candidates['c0']
    s.candidates['c9'] = type('C', (), {'config': dict(c0.config), 'overlay': {'model.py': "attn = 'sdpa'\n"},
                                       'receipt': c0.receipt, 'verdict': c0.verdict})()
    mech = first[0]['mechanism']
    again = prop.hypotheses(eng, 'c9', ledger=[{'mechanism': mech, 'status': 'open'}])     # QA13 F1: not tested yet
    assert [h['patch'] for h in first] == [h['patch'] for h in again] == [FIXED]
    assert prop.hypotheses(eng, 'c9', ledger=[{'mechanism': mech, 'status': 'refuted'}]) == []
    assert [c['role'] for c in drv.calls].count('code_review') == 1


def test_a_review_that_spends_the_budget_ends_the_call():
    sim, s, eng = setup(TOGGLE, tokens=10 ** 5)                 # the scripted review is charged the whole budget
    review = _review('patch', [{'file': 'step.py', 'issue': 'sync', 'knob': None, 'edits': [SYNC]}])
    drv = ScriptedBackend(lambda role, system, prompt: review if role == 'code_review' else {'hypotheses': []},
                          tokens_per_call=10 ** 5)               # (QA13 F3)
    hyps = LLMProposer(drv, SessionHooks(s.task, base_files=dict(FILES))).hypotheses(eng, 'c0')
    assert [c['role'] for c in drv.calls] == ['code_review'] and [h['patch'] for h in hyps] == [FIXED]


def test_calls_with_code_get_room_for_edits():
    # C3c: Opus's hypotheses with edits overran 8000 tokens; C1b: the vLLM backend's own 4096 cut answers short
    from infrarsi.project.llm import make_driver
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    caps = []

    class Capped(ScriptedBackend):
        def complete(self, system, prompt, **kw):
            caps.append((kw.get('role'), kw.get('max_output_tokens')))
            return super().complete(system, prompt, **kw)
    for verdict in ('patch', 'no_patch'):
        drv = Capped(lambda role, system, prompt: _review(verdict, []) if role == 'code_review' else {'hypotheses': []},
                     tokens_per_call=32000)                 # a backend whose own cap leaves the room
        LLMProposer(drv, SessionHooks(s.task, base_files=dict(FILES))).hypotheses(eng, 'c0')
    assert caps == [('code_review', 16000), ('hypotheses', 16000), ('code_review', 16000), ('hypotheses', 8000)]
    assert make_driver({'backend': 'vllm', 'model': 'Qwen3-30B-A3B'}).max_tokens == 16384
    assert make_driver({'backend': 'vllm', 'model': 'Qwen3-30B-A3B', 'max_tokens': 2048}).max_tokens == 2048


def test_review_reaches_the_result_and_the_report():
    cr = _review('patch', [{'file': 'step.py', 'issue': 'sync | per micro-batch', 'knob': None, 'patch': 'queued'}], 'one sync')
    trace = {'extra': {'strategy': {'name': 'chain', 'parts': [{'name': 'doe'}, {'name': 'hls', 'proposer': {'code_review': cr}}]}}}
    assert _code_review(trace) == cr and _code_review({'extra': {'strategy': 'tpe'}}) is None
    text = '\n'.join(review_lines(cr))
    assert '## Code review' in text and 'code changes worth a test run. one sync' in text
    assert '| step.py | sync / per micro-batch | code only | queued |' in text
