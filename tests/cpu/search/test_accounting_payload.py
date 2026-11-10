"""The accounting field (textbook design P0-1, 2026-10-24): the proposer sees the closed-form estimate for the incumbent
at its measured update time, with the rule that frames it; without an accounting (ablated, or no model shape) neither
appears."""
from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.ext.common.session_hooks import SessionHooks
from infrarsi.search.proposers import ACCOUNTING_RULE, LLMProposer

from test_search import TOGGLE, setup


def _ask(accounting):
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    backend = ScriptedBackend(lambda role, system, prompt: {'hypotheses': []})
    LLMProposer(backend, SessionHooks(s.task, base_files={}), accounting=accounting).hypotheses(eng, 'c0')
    return backend.calls[-1], s.candidates['c0']


def test_the_proposer_sees_the_incumbents_estimate():
    asked = []

    def view(config, step_time):
        asked.append((dict(config), step_time))
        return {'memory_gib': 61.0, 'capacity_gib': 80, 'margin_gib': 19.0, 'mfu': 0.41}
    call, c0 = _ask(view)
    assert ACCOUNTING_RULE in call['system'] and '"accounting": {"memory_gib": 61.0' in call['prompt']
    assert asked == [(c0.config, c0.verdict['step_time'])] and asked[0][1] > 0
    call, _ = _ask(None)
    assert ACCOUNTING_RULE not in call['system'] and '"accounting"' not in call['prompt']
