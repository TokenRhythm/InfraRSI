import dataclasses
import json
from types import SimpleNamespace

import pytest

from infrarsi.agent.budget import Budget, SearchLedger
from infrarsi.search.bcomp import BComp
from infrarsi.search.proposers import LLMProposer, _allowances
from infrarsi.search.token_governor import TokenGovernor, parse_duration_hours, parse_tokens


@pytest.mark.parametrize('value,expected', [('500k', 500000), ('1.5M', 1500000), ('250000', 250000), (0, 0), ('2m', 2000000)])
def test_token_units(value, expected):
    assert parse_tokens(value) == expected


@pytest.mark.parametrize('value', ['-1', '1.2', 'NaN', '1G', True, 1.5, ''])
def test_bad_tokens(value):
    with pytest.raises(ValueError):
        parse_tokens(value)


@pytest.mark.parametrize('value,expected', [('90m', 1.5), ('2h', 2.0), ('3600s', 1.0), (1.25, 1.25), ('0', 0),
                                            ('.5', .5), ('1e-2', .01)])
def test_duration_units(value, expected):
    assert parse_duration_hours(value) == expected


@pytest.mark.parametrize('value', ['-2h', '1d', 'inf', float('nan'), True, ''])
def test_bad_duration(value):
    with pytest.raises(ValueError):
        parse_duration_hours(value)


def ledger(tokens=100000, used=0):
    led = SearchLedger(Budget(10, tokens, 7200, 100))
    led.used['llm_tokens'] = used
    return led


@pytest.mark.parametrize('total,used,zone,code,scale,revise,count', [
    (100000, 64999, 'green', 12000, 1, True, 12),
    (100000, 65000, 'economy', 4000, .5, False, 4),
    (100000, 87999, 'economy', 4000, .5, False, 4),
    (100000, 88000, 'wind_down', 1500, .25, False, 2),
    (31999, 0, 'economy', 4000, .5, False, 4),
    (32000, 0, 'green', 12000, 1, True, 12),
    (12000, 0, 'economy', 4000, .5, False, 4),
    (11999, 0, 'wind_down', 1500, .25, False, 2),
    (100000, 100001, 'wind_down', 1500, .25, False, 2),
    (0, 0, 'green', 12000, 1, True, 12),
])
def test_zones(total, used, zone, code, scale, revise, count):
    g = TokenGovernor(ledger(total, used))
    assert (g.zone, g.code_tokens, g.history_scale, g.allow_revise, g.max_hypotheses) == (zone, code, scale, revise, count)


def test_remaining_tokens_admit_a_smaller_answer_and_keep_time_limits():
    from infrarsi.agent.optimize import input_reserve
    driver = SimpleNamespace(max_output_tokens=16000, call_overhead_tokens=64)
    s = SimpleNamespace(ledger=ledger(10000))
    text = 'prompt' * 100
    kw = _allowances(s, driver, text, 16000)
    assert kw['max_output_tokens'] == 10000 - input_reserve(text) - 64
    s.ledger.used['llm_tokens'] = 10000 - input_reserve(text) - 64 - 1023
    assert _allowances(s, driver, text, 16000) is None
    s.ledger.budget = dataclasses.replace(s.ledger.budget, wall_seconds=0)
    assert _allowances(s, driver, text, 16000) is None


def test_revise_is_skipped_in_economy_without_consuming_its_due_reason():
    st = BComp(SimpleNamespace(driver=None))
    st.bc_revise_due = 'a failed move'
    st._bc_revise(SimpleNamespace(s=SimpleNamespace(ledger=ledger(100000, 70000))))
    assert st.bc_revise_due == 'a failed move' and not st.bc_revisions


def test_proposer_compacts_history_and_hypothesis_count():
    from infrarsi.agent.backends import ScriptedBackend
    from test_search import TOGGLE, setup
    _, s, eng = setup(TOGGLE, tokens=100000)
    seen = []
    backend = ScriptedBackend(lambda role, system, prompt: seen.append(json.loads(prompt)) or {'hypotheses': []})
    prop = LLMProposer(backend, s.hooks, code_tokens=0, max_hypotheses=12)
    s.ledger.used['llm_tokens'] = 70000
    prop.hypotheses(eng, 'c0', ledger=[{'id': i} for i in range(40)])
    assert seen[0]['max_hypotheses'] == 4 and len(seen[0]['ledger']) == 20
    s.ledger.used['llm_tokens'] = 90000
    prop.hypotheses(eng, 'c0', ledger=[{'id': i} for i in range(40)])
    assert seen[1]['max_hypotheses'] == 2 and len(seen[1]['ledger']) == 10


def test_review_usage_compacts_the_next_proposal_in_the_same_round():
    from test_df16_large_file_patch import Driver, REVIEW, proposer
    driver = Driver([REVIEW, {'hypotheses': []}])
    complete = driver.complete

    def charged_review(*a, **kw):
        result = complete(*a, **kw)
        return dataclasses.replace(result, llm_tokens=70000) if kw.get('role') == 'code_review' else result

    driver.complete = charged_review
    prop, eng, _ = proposer(driver)
    eng.s.budget = eng.s.ledger.budget = dataclasses.replace(eng.s.budget, llm_tokens=100000)
    prop.hypotheses(eng, 'c0', ledger=[{'id': i} for i in range(40)])
    review, hypotheses = driver.calls
    assert review['role'] == 'code_review' and hypotheses['role'] == 'hypotheses'
    assert hypotheses['prompt']['max_hypotheses'] == 4 and len(hypotheses['prompt']['ledger']) == 20

