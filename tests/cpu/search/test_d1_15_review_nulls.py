"""Trial D1-15 (Opus): the code review gave null predictions for findings a knob already reaches, as its prompt asks
("with a change give ..."); the schema wanted numbers and the whole review was dropped."""
from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.ext.common.session_hooks import SessionHooks
from infrarsi.search.proposers import LLMProposer

from test_search import TOGGLE, setup

FILES = {'step.py': 'x = 1\ntorch.cuda.empty_cache()\ny = 2\n'}
NULLS = {'predicted_gain_pct': None, 'uncertainty_pct': None, 'p_feasible': None}


def test_a_review_with_null_predictions_is_used():
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    review = {'verdict': 'patch', 'summary': 's', 'findings': [
        {'file': 'step.py', 'issue': 'empty_cache every update', 'knob': None, 'change_class': 'execution',
         'edits': [{'file': 'step.py', 'old': 'torch.cuda.empty_cache()\n', 'new': ''}], 'patch': None,
         'predicted_gain_pct': 5, 'uncertainty_pct': None, 'p_feasible': None},
        {'file': 'train.py', 'issue': 'fp32 matmuls', 'knob': 'fused_adam', 'edits': None, 'patch': None, **NULLS}]}
    hyp = {'hypotheses': [{'mechanism': 'fused', 'change': {'fused_adam': True}, 'predicted_gain_pct': 3}]}
    drv = ScriptedBackend(lambda role, system, prompt: review if role == 'code_review' else hyp)
    prop = LLMProposer(drv, SessionHooks(s.task, base_files=dict(FILES)))
    hyps = prop.hypotheses(eng, 'c0')
    assert prop.describe()['code_review']['verdict'] == 'patch'
    assert hyps[0]['patch'] == {'step.py': 'x = 1\ny = 2\n'} and hyps[0]['p_feasible'] == 0.8
    assert [f['patch'] for f in prop.describe()['code_review']['findings']] == ['queued', None]
