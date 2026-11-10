"""Whole-codebase review fixes in the search proposer (2026-10-24, C2 #7 #8 / C4 #5): an LLM call is made only when
it fits what the search budget has left (tokens, USD, wall time are passed on), and a failed call is said in the
progress output."""
from infrarsi.agent.backends.base import LLMResult
from infrarsi.ext.common.session_hooks import SessionHooks
from infrarsi.search.proposers import LLMProposer
from test_search import TOGGLE, setup

HYPS = {'hypotheses': [{'mechanism': 'fused adam', 'change': {'fused_adam': True}, 'predicted_gain_pct': 4}]}


class Recorder:
    model = 'fake'

    def __init__(self, action=None, error=None):
        self.calls, self.action, self.error = [], action, error

    def complete(self, system, prompt, **kw):
        self.calls.append(kw)
        return LLMResult(action=self.action, text=None, llm_tokens=100, llm_usd=0.0, wall_seconds=0.1,
                         accounting_complete=True, error=self.error, model='fake', observed_models=['fake'])


def test_the_proposer_spends_only_the_budget_it_has_and_says_when_a_call_fails():
    sim, s, eng = setup(TOGGLE, tokens=0)
    d = Recorder(action=HYPS)
    assert LLMProposer(d, SessionHooks(s.task, base_files={})).hypotheses(eng, 'c0') == [] and not d.calls
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    d = Recorder(action=HYPS)
    assert LLMProposer(d, SessionHooks(s.task, base_files={})).hypotheses(eng, 'c0')
    assert d.calls[0]['token_allowance'] == 10 ** 6 and d.calls[0]['timeout'] > 0 and d.calls[0]['max_output_tokens'] <= 8000
    logs = []
    eng.log = logs.append
    LLMProposer(Recorder(error='HTTP 401: invalid API key'), SessionHooks(s.task, base_files={})).hypotheses(eng, 'c0')
    assert any('hypothesis call failed: HTTP 401' in m for m in logs)
