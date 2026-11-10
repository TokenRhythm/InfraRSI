"""The capability knob: endpoints reproduce the old weak / strong protocol, values in between interpolate, the search
scaffolding follows it, and the LLM proposer's menu mode only accepts enumerated moves."""
import json

from infrarsi.agent import capability as cap
from infrarsi.agent import policy as policy_mod
from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.ext.common.session_hooks import SessionHooks
from infrarsi.search.engine import Engine
from infrarsi.search.proposers import LLMProposer

from test_search import TOGGLE, setup


def test_endpoints_are_the_old_protocols():
    weak, strong = policy_mod.DEFAULT_POLICY['modes']['weak'], policy_mod.DEFAULT_POLICY['modes']['strong']
    for c, ref in ((0.0, weak), (1.0, strong)):
        over = cap.policy_overrides(c)
        for k, v in over.items():
            assert v == ref[k], (c, k)
    assert cap.mode_for(0.49) == 'weak' and cap.mode_for(0.5) == 'strong'


def test_interpolation_is_monotone_and_loads():
    prev = None
    for c in (0.0, 0.25, 0.5, 0.75, 1.0):
        o = cap.policy_overrides(c)
        pol = policy_mod.load(policy_mod.default_yaml(), cap.mode_for(c), **o)
        assert pol.patch_line_limit == o['patch_line_limit']
        if prev:
            assert o['patch_line_limit'] >= prev['patch_line_limit'] and o['max_retries'] <= prev['max_retries']
        prev = o
    lo, hi = cap.search_params(0.1), cap.search_params(0.9)
    assert lo['menu'] and not hi['menu'] and lo['kappa0'] < hi['kappa0'] and lo['probe_below'] > hi['probe_below']


def test_infer_from_model_names():
    assert cap.infer('claude-fable-5-1') > cap.infer('claude-opus-5-5') > cap.infer('claude-sonnet-5-5') > cap.infer('claude-haiku-5-5')
    assert cap.infer('qwen2.5-7b-instruct') < 0.3 and cap.infer(None) == 0.6


def test_menu_mode_accepts_only_enumerated_moves():
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    replies = [{'hypotheses': [
        {'mechanism': 'flash', 'change': {'flash_attn': True}, 'predicted_gain_pct': 8, 'p_feasible': 0.9},
        {'mechanism': 'invented', 'change': {'flash_attn': True, 'compile': True}, 'predicted_gain_pct': 20},
        {'mechanism': 'unknown knob', 'change': {'turbo': True}, 'predicted_gain_pct': 50}]}]
    drv = ScriptedBackend(lambda role, system, prompt: replies[0])
    hooks = SessionHooks(s.task, base_files={'patches.txt': ''})
    menu = LLMProposer(drv, hooks, menu=True, max_hypotheses=4).hypotheses(eng, 'c0')
    assert [h['mechanism'] for h in menu] == ['flash'] and 'MENU MODE' in drv.calls[0]['system']
    assert json.loads(drv.calls[0]['prompt'])['menu']
    free = LLMProposer(drv, hooks).hypotheses(eng, 'c0')
    assert [c['role'] for c in drv.calls] == ['hypotheses', 'code_review', 'hypotheses']   # the review sees the code
    assert len(free) == 3 and s.ledger.used['llm_tokens'] == 3 * drv.tokens_per_call
