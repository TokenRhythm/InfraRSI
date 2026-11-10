"""HLS asks the model again when the search runs dry (FABLE_HLS_REASK_DECISION, 2026-10-24). E-C-IS2b: one failed
hypothesis call on the patch incumbent, two slower knob values, and the search ended `strategy_done` with 0.4 of its
2.85 GPU-hours spent. Now, when the incumbent's ledger is measured, its single-knob neighbourhood tried and no bundle is
open, the model is asked again: at most `reask` times per incumbent, not after an answer that added nothing new; a
failed call is retried within the cap."""
import json

from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.ext.common.session_hooks import SessionHooks
from infrarsi.search.engine import Outcome
from infrarsi.search.proposers import LLMProposer
from infrarsi.search.strategies import HLS

from test_composition import FakeEngine, hyp
from test_search import TOGGLE, setup


class Script:
    """A proposer that answers each call from a list; None is a failed call (no answer: LLMProposer counts `failed`)."""

    def __init__(self, answers):
        self.answers, self.calls, self.failed = list(answers), 0, 0

    def describe(self):
        return {}

    def hypotheses(self, engine, base, ledger=None):
        self.calls += 1
        a = self.answers.pop(0) if self.answers else []
        if a is None:
            self.failed += 1
            return []
        return [dict(h) for h in a]


def refuted(cid):
    return Outcome(cid, 'screened', True, 1.0, -0.01, 'full', 0.1, [], [])


def test_dry_search_asks_again_and_stops_after_an_answer_with_nothing_new():
    eng, prop = FakeEngine(), Script([[], [hyp('a')], []])
    hls = HLS(prop, explore=False)
    props = hls.propose(eng, 16)                                    # the first call gave nothing: asked again at once
    assert [p.mechanism for p in props] == ['a'] and prop.calls == 2 and hls.reasks == {'c0': ['new']}
    hls.observe(eng, props[0], refuted('c1'))
    assert hls.propose(eng, 16) == [] and prop.calls == 3 and hls.reasks == {'c0': ['new', 'nothing']}
    assert hls.propose(eng, 16) == [] and prop.calls == 3           # nothing new: no more calls on this incumbent
    eng._inc = 'c9'                                                 # a new incumbent: its own call and re-asks
    assert hls.propose(eng, 16) == [] and prop.calls == 5 and hls.reasks['c9'] == ['nothing']
    assert hls.describe()['reasks'] == hls.reasks


def test_reasks_are_capped_per_incumbent():
    eng, prop = FakeEngine(), Script([[hyp(n)] for n in 'abcdef'])   # a model that always has one more idea
    hls = HLS(prop, explore=False, reask=2)
    for cid in ('c1', 'c2', 'c3'):
        props = hls.propose(eng, 16)
        assert len(props) == 1
        hls.observe(eng, props[0], refuted(cid))
    assert hls.propose(eng, 16) == [] and prop.calls == 3 and hls.reasks == {'c0': ['new', 'new']}
    assert hls.propose(eng, 16) == [] and prop.calls == 3
    old = HLS(Script([[], [hyp('a')]]), explore=False, reask=0)       # reask=0: the old behaviour, one call per incumbent
    assert old.propose(eng, 16) == [] and old.proposer.calls == 1


def test_a_failed_call_is_retried_within_the_cap():
    eng, prop = FakeEngine(), Script([None, None, [hyp('a')]])
    hls = HLS(prop, explore=False)
    props = hls.propose(eng, 16)
    assert [p.mechanism for p in props] == ['a'] and prop.calls == 3 and hls.reasks == {'c0': ['failed', 'new']}
    hls.observe(eng, props[0], refuted('c1'))
    assert hls.propose(eng, 16) == [] and prop.calls == 3           # the cap: no fourth call
    eng, prop = FakeEngine(), Script([None, None, None, [hyp('a')]])
    hls = HLS(prop, explore=False)
    assert hls.propose(eng, 16) == [] and prop.calls == 3 and hls.reasks == {'c0': ['failed', 'failed']}


def test_a_repeat_under_another_name_is_nothing_new():
    eng, prop = FakeEngine(), Script([[hyp('a')], [dict(hyp('a'), mechanism='a, again')], [hyp('b')]])
    hls = HLS(prop, explore=False)
    hls.observe(eng, hls.propose(eng, 16)[0], refuted('c1'))
    again = hls.propose(eng, 16)                                    # the same change under another name: proposed once,
    assert [p.mechanism for p in again] == ['a, again'] and hls.reasks == {'c0': ['nothing']}
    hls.observe(eng, again[0], Outcome(None, 'duplicate', False, None, None, 'none', 0.0, ['already tried'], []))
    assert hls.propose(eng, 16) == [] and prop.calls == 2           # ... the engine ran nothing, and no third call


def test_another_change_under_a_used_name_is_new():
    # DF7_r1 #2: a different change (or patch) under a name already used on this base is a new entry, not dropped
    eng, prop = FakeEngine(), Script([[hyp('a')], [dict(hyp('b'), mechanism='a')], []])
    hls = HLS(prop, explore=False)
    hls.observe(eng, hls.propose(eng, 16)[0], refuted('c1'))
    again = hls.propose(eng, 16)
    assert [(p.mechanism, p.change) for p in again] == [('a', {'b': 1})] and hls.reasks == {'c0': ['new']}
    assert len([h for h in hls.ledger.values() if h['mechanism'] == 'a']) == 2       # the first measurement stays


def test_same_name_variants_are_carried_apart():
    # DF7_r2 #2: two changes under one name, both unresolved on c0, both reach the new incumbent
    tune = dict(hyp('a'), mechanism='tune')
    eng, prop = FakeEngine(), Script([[tune], [dict(tune, change={'a': 2})], []])
    hls = HLS(prop, explore=False, bundle=False)
    hls.observe(eng, hls.propose(eng, 16)[0], Outcome('c1', 'screened', True, 1.0, 0.01, 'full', 0.1, [], []))
    p = hls.propose(eng, 16)[0]                                      # the re-ask's variant under the same name
    assert p.change == {'a': 2}
    hls.observe(eng, p, Outcome('c2', 'screened', True, 1.0, 0.02, 'full', 0.1, [], []))   # both within noise
    eng._inc = 'c9'
    hls.propose(eng, 16)
    carried = sorted(h['change']['a'] for h in hls.ledger.values() if h['base'] == 'c9' and h['mechanism'] == 'tune')
    assert carried == [1, 2]


def test_toy_search_goes_on_after_a_failed_call_instead_of_ending_strategy_done():
    lines = []
    sim, s, eng = setup(TOGGLE, budget=8, tokens=10 ** 6, job_gpu_hours=None, log=lines.append)
    trio = {'mechanism': 'fused trio', 'change': {'fused_adam': True, 'flash_attn': True, 'overlap_grad': True},
            'predicted_gain_pct': 30, 'p_feasible': 0.95}
    # the first call and the first re-ask fail (not JSON), the second re-ask proposes the trio (it moves the incumbent);
    # on the new incumbent the model has nothing, twice
    replies = iter(['not json', 'not json', {'hypotheses': [trio]}])
    drv = ScriptedBackend(lambda role, system, prompt: next(replies, {'hypotheses': []}))
    prop = LLMProposer(drv, SessionHooks(s.task, base_files={'patches.txt': ''}), code_tokens=0)
    hls = HLS(prop, explore=False)
    tr = eng.run(hls).to_dict()
    inc = eng.incumbent()
    assert tr['stop_reason'] == 'strategy_done' and inc != 'c0' and s.ledger.used['screens'] == 1
    assert prop.failed == 2 and len(drv.calls) == 5 and hls.reasks == {'c0': ['failed', 'new'], inc: ['nothing']}
    assert tr['extra']['strategy']['reasks'] == hls.reasks and tr['extra']['strategy']['proposer']['failed_calls'] == 2
    seen = json.loads(drv.calls[-1]['prompt'])['ledger']           # the model asked again sees what was measured since
    assert [e['mechanism'] for e in seen] == ['fused trio'] and seen[0]['status'] == 'confirmed'
    assert seen[0]['measured_gain_pct'] > 0 and seen[0]['candidate'] == inc
    assert [m for m in lines if 'asking the model again' in m] == [
        '[search] hls: nothing left to test on c0: asking the model again (1 of 2)',
        '[search] hls: nothing left to test on c0: asking the model again (2 of 2)',
        f'[search] hls: nothing left to test on {inc}: asking the model again (1 of 2)']
    # before this change (reask=0) the failed call ended the search at once, with nothing measured
    sim, s, eng = setup(TOGGLE, budget=8, tokens=10 ** 6, job_gpu_hours=None)
    drv = ScriptedBackend(lambda role, system, prompt: 'not json')
    hls = HLS(LLMProposer(drv, SessionHooks(s.task, base_files={'patches.txt': ''}), code_tokens=0), explore=False, reask=0)
    tr = eng.run(hls).to_dict()
    assert tr['stop_reason'] == 'strategy_done' and s.ledger.used['screens'] == 0 and len(drv.calls) == 1
