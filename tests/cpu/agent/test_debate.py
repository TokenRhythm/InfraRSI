"""The dual-model debate (agent/debate.py) and the patch check (search/patch_check.py): two proposals, a blind judgement, at
most two checks of a patch; off by default."""
import json
from types import SimpleNamespace

import pytest

from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.agent.debate import AUTHOR_KEY, DEBATE_ROLES, DebateDriver, identity_words, logged, model_name
from infrarsi.search.patch_check import check_patches
from infrarsi.search.proposers import LLMProposer


def backend(replies, model):
    return ScriptedBackend(list(replies), model=model, tokens_per_call=1000, usd_per_call=0.01)


def hyp(mech, gain=5):
    return {'mechanism': mech, 'change': {'k': 1}, 'predicted_gain_pct': gain, 'rationale': f'{mech}: because of the timers'}


SCHEMA = {'type': 'object', 'required': ['hypotheses'], 'properties': {'hypotheses': {'type': 'array'}}}


def test_a_debate_makes_three_calls_and_charges_them_all():
    main = backend([{'hypotheses': [hyp('m1')]}, {'winner': 'A', 'selected': ['A1', 'B1'], 'analysis': 'both'}], 'claude-opus-5-5')
    second = backend([{'hypotheses': [hyp('s1')]}], 'deepseek-chat')
    d = DebateDriver(main, second, seed=1)
    res = d.complete('SYSTEM', 'PROMPT', schema=SCHEMA, role='hypotheses')
    assert [c['role'] for c in main.calls] == ['hypotheses', 'debate_judge'] and len(second.calls) == 1
    assert res.llm_tokens == 3000 and res.llm_usd == pytest.approx(0.03) and res.accounting_complete
    assert res.attempts == 1                             # a debated call is one ask of the strategy's call allowance
    assert [h['mechanism'] for h in res.action['hypotheses']] == ['m1', 's1']
    assert second.calls[0]['system'] == 'SYSTEM' and second.calls[0]['prompt'] == 'PROMPT'   # the same request, no main answer


def test_the_second_model_does_not_see_the_main_answer():
    main = backend([{'hypotheses': [hyp('main-secret-idea')]}, {'winner': 'A'}], 'm')
    second = backend([{'hypotheses': [hyp('other')]}], 's')
    DebateDriver(main, second).complete('S', 'P', schema=SCHEMA, role='hypotheses')
    assert 'main-secret-idea' not in second.calls[0]['prompt'] and 'main-secret-idea' not in second.calls[0]['system']


def _judge_prompt(main):
    return next(c for c in main.calls if c['role'] == 'debate_judge')


def test_the_judge_cannot_tell_who_wrote_what():
    main = backend([{'hypotheses': [{**hyp('m1'), 'rationale': 'As Claude (claude-opus-5-5) I think'}]}, {'winner': 'A'}], 'claude-opus-5-5')
    main.kind = 'claude_cli'
    second = backend([{'hypotheses': [{**hyp('s1'), 'rationale': 'deepseek-chat: DeepSeek says so'}]}], 'deepseek-chat')
    DebateDriver(main, second).complete('S', 'P', schema=SCHEMA, role='hypotheses')
    text = _judge_prompt(main)['prompt'] + _judge_prompt(main)['system']
    for word in ('claude', 'opus', 'deepseek', 'anthropic', AUTHOR_KEY, 'main_model', 'second_model'):
        assert word.lower() not in text.lower(), word
    view = json.loads(_judge_prompt(main)['prompt'])
    assert set(view) == {'request', 'proposal_A', 'proposal_B'}
    assert '[model]' in json.dumps(view['proposal_A']) and '[model]' in json.dumps(view['proposal_B'])


def test_code_and_knob_values_are_left_as_they_are():
    edits = [{'file': 'train.py', 'old': 'from deepseek_model import X', 'new': 'from deepseek_model import Y'}]
    main = backend([{'hypotheses': [{**hyp('m'), 'edits': edits}]}, {'winner': 'A'}], 'deepseek-chat')
    second = backend([{'hypotheses': [hyp('s')]}], 'other')
    DebateDriver(main, second).complete('S', 'P', schema=SCHEMA, role='hypotheses')
    assert 'deepseek_model' in _judge_prompt(main)['prompt']


def _labels(seed):
    main = backend([{'hypotheses': [hyp('from-main')]}, {'winner': 'A'}], 'm')
    second = backend([{'hypotheses': [hyp('from-second')]}], 's')
    DebateDriver(main, second, seed=seed).complete('S', 'P', schema=SCHEMA, role='hypotheses')
    view = json.loads(_judge_prompt(main)['prompt'])
    return 'main' if 'from-main' in json.dumps(view['proposal_A']) else 'second'


def test_the_labels_are_random():
    assert {_labels(s) for s in range(8)} == {'main', 'second'}


def test_the_winner_whole_when_no_items_are_selected():
    main = backend([{'hypotheses': [hyp('m1'), hyp('m2')]}, {'winner': 'B'}], 'm')
    second = backend([{'hypotheses': [hyp('s1')]}], 's')
    res = DebateDriver(main, second, seed=3).complete('S', 'P', schema=SCHEMA, role='hypotheses')
    assert len(res.action['hypotheses']) in (1, 2)       # B is whichever the random label gave
    assert all(AUTHOR_KEY in h for h in res.action['hypotheses'])


def test_selected_items_keep_the_judges_order_and_their_author():
    main = backend([{'hypotheses': [hyp('a'), hyp('b')]}, {'winner': 'A', 'selected': ['B1', 'A2', 'nonsense', 'A9', 'B1']}], 'm')
    second = backend([{'hypotheses': [hyp('c')]}], 's')
    d = DebateDriver(main, second, seed=3)               # (this seed labels the main model's proposal A)
    res = d.complete('S', 'P', schema=SCHEMA, role='hypotheses')
    view = json.loads(_judge_prompt(main)['prompt'])
    b1, a2 = view['proposal_B']['hypotheses'][0], view['proposal_A']['hypotheses'][1]
    assert [h['mechanism'] for h in res.action['hypotheses']] == [b1['mechanism'], a2['mechanism']]
    assert {h[AUTHOR_KEY] for h in res.action['hypotheses']} <= {'main', 'second'}


def test_a_failed_main_answer_comes_back_as_it_was():
    main = backend(['not json'], 'm')
    second = backend([{'hypotheses': []}], 's')
    res = DebateDriver(main, second).complete('S', 'P', schema=SCHEMA, role='hypotheses')
    assert res.action is None and res.error and not second.calls and res.llm_tokens == 1000


def test_a_failed_second_answer_leaves_the_main_one():
    main = backend([{'hypotheses': [hyp('m1')]}], 'm')
    second = backend(['not json'], 's')
    d = DebateDriver(main, second)
    res = d.complete('S', 'P', schema=SCHEMA, role='hypotheses')
    assert [h['mechanism'] for h in res.action['hypotheses']] == ['m1'] and res.llm_tokens == 2000
    assert d.summary()['outcomes'] == {'second_failed': 1} and len(main.calls) == 1


def test_a_failed_judgement_leaves_the_main_answer():
    main = backend([{'hypotheses': [hyp('m1')]}, 'not json'], 'm')
    second = backend([{'hypotheses': [hyp('s1')]}], 's')
    d = DebateDriver(main, second)
    res = d.complete('S', 'P', schema=SCHEMA, role='hypotheses')
    assert [h['mechanism'] for h in res.action['hypotheses']] == ['m1'] and res.llm_tokens == 3000
    assert d.summary()['outcomes'] == {'judge_failed': 1}


def test_two_identical_answers_need_no_judgement():
    main = backend([{'hypotheses': [hyp('same')]}], 'm')
    second = backend([{'hypotheses': [hyp('same')]}], 's')
    d = DebateDriver(main, second)
    res = d.complete('S', 'P', schema=SCHEMA, role='hypotheses')
    assert len(main.calls) == 1 and res.llm_tokens == 2000 and d.summary()['outcomes'] == {'same_answer': 1}


def _spied(driver):
    seen, real = [], driver.complete

    def spy(*a, **kw):
        seen.append(kw)
        return real(*a, **kw)
    driver.complete = spy
    return seen


def test_the_allowances_left_shrink_call_by_call():
    main = backend([{'hypotheses': [hyp('m')]}, {'winner': 'A'}], 'm')
    sec = ScriptedBackend(lambda *a: {'hypotheses': [hyp('s')]}, model='s', tokens_per_call=500, usd_per_call=0.2)
    second_kw, main_kw = _spied(sec), _spied(main)
    DebateDriver(main, sec).complete('S', 'P', schema=SCHEMA, role='hypotheses', token_allowance=100000, max_usd=1.0, timeout=100.0,
                                     max_output_tokens=5000)
    assert second_kw[0]['token_allowance'] == 99000 and second_kw[0]['max_usd'] == pytest.approx(0.99) and second_kw[0]['timeout'] == 99.0
    judge = main_kw[1]
    assert judge['token_allowance'] == 98500 and judge['max_usd'] == pytest.approx(0.79) and judge['timeout'] == 98.0
    assert judge['max_output_tokens'] <= 2048            # the judgement is short: the answer's cap is not its


def test_a_call_that_does_not_fit_the_tokens_left_is_not_sent():
    main = backend([{'hypotheses': [hyp('m')]}, {'winner': 'A'}], 'm')
    second = backend([{'hypotheses': [hyp('s')]}], 's')
    d = DebateDriver(main, second)
    res = d.complete('S', 'P' * 40000, schema=SCHEMA, role='hypotheses', token_allowance=3000)
    assert not second.calls and len(main.calls) == 1 and d.summary()['outcomes'] == {'no_budget': 1} and res.action


def test_the_judge_is_not_sent_when_it_does_not_fit():
    main = backend([{'hypotheses': [hyp('m')]}, {'winner': 'A'}], 'm')
    second = ScriptedBackend(lambda *a: {'hypotheses': [{**hyp('s'), 'rationale': 'x' * 60000}]}, model='s', tokens_per_call=1000)
    d = DebateDriver(main, second)
    res = d.complete('S', 'P', schema=SCHEMA, role='hypotheses', token_allowance=20000)
    assert len(main.calls) == 1 and len(second.calls) == 1 and d.summary()['outcomes'] == {'judge_failed': 1}
    assert [h['mechanism'] for h in res.action['hypotheses']] == ['m']


def test_nulls_are_kept_and_two_answers_that_differ_by_one_are_judged():
    a = {'configs': [{'change': {'fa': True, 'nw': None}, 'gain_pct': 3}]}
    b = {'configs': [{'change': {'fa': True}, 'gain_pct': 3}]}
    main = backend([a, {'winner': 'A'}], 'm')
    second = backend([b], 's')
    d = DebateDriver(main, second, seed=3)
    d.complete('S', 'P', role='configs')
    assert d.summary()['outcomes'] == {'judged': 1}
    assert '"nw": null' in _judge_prompt(main)['prompt']


def test_every_text_field_is_blinded_but_not_code_or_values():
    reply = {'configs': [{'name': "Claude's proposed configuration", 'change': {'model_name': 'claude-opus-5-5'}, 'why': 'by Anthropic'}],
             'hypotheses': [{'mechanism': 'As ChatGPT from OpenAI', 'edits': [{'file': 'a.py', 'old': 'claude', 'new': 'gpt'}]}]}
    from infrarsi.agent.debate import _identity_pattern, blinded
    pat = _identity_pattern(SimpleNamespace(model='claude-opus-5-5', kind='claude_cli'), SimpleNamespace(model='gpt-6.1-sol', kind='openai_compat'))
    out = blinded(reply, pat)
    assert 'Claude' not in out['configs'][0]['name'] and 'Anthropic' not in out['configs'][0]['why']
    assert 'ChatGPT' not in out['hypotheses'][0]['mechanism'] and 'OpenAI' not in out['hypotheses'][0]['mechanism']
    assert out['configs'][0]['change'] == {'model_name': 'claude-opus-5-5'}
    assert out['hypotheses'][0]['edits'] == reply['hypotheses'][0]['edits']


def test_a_patch_written_as_one_edit_is_blinded_around_its_code():
    from infrarsi.agent.debate import _identity_pattern, blinded
    pat = _identity_pattern(SimpleNamespace(model='claude-opus-5-5', kind='claude_cli'), SimpleNamespace(model='deepseek-chat', kind='openai_compat'))
    edit = {'file': 'train.py', 'old': 'a = 1', 'new': 'a = "deepseek"', 'reason': 'Claude from Anthropic proposes this edit', AUTHOR_KEY: 'main'}
    out = blinded({'hypotheses': [{'patch': edit, 'text_edit': 1}]}, pat)['hypotheses'][0]['patch']
    assert out['new'] == edit['new'] and 'Claude' not in out['reason'] and AUTHOR_KEY not in out
    whole = {'patch': {'train.py': 'claude = 1'}}
    assert blinded({'hypotheses': [whole]}, pat)['hypotheses'][0]['patch'] == whole['patch']   # a {path: text} map is code


def test_executable_edit_bodies_and_transfer_payloads_are_kept():
    from infrarsi.agent.debate import _identity_pattern, blinded
    pat = _identity_pattern(SimpleNamespace(model='gpt-6.1-sol', kind='openai_compat'), SimpleNamespace(model='deepseek-chat', kind='openai_compat'))
    reply = {'hypotheses': [{'edits': [{'file': 'a.py', 'op': 'append', 'text': 'if m == "deepseek": x()', 'code': 'if m == "gpt": y()'}]}],
             'start_config': {'backend': 'deepseek'}, 'mem_terms': [{'gib_expr': '8 if b == "deepseek" else 2', 'when': 'b == "gpt"', 'name': 'DeepSeek term'}]}
    out = blinded(reply, pat)
    assert out['hypotheses'][0]['edits'][0] == {**reply['hypotheses'][0]['edits'][0]}
    assert out['start_config'] == reply['start_config'] and out['mem_terms'][0]['gib_expr'] == reply['mem_terms'][0]['gib_expr']
    assert out['mem_terms'][0]['when'] == reply['mem_terms'][0]['when'] and 'DeepSeek' not in out['mem_terms'][0]['name']


def test_a_text_field_outside_an_edit_is_blinded():
    from infrarsi.agent.debate import _identity_pattern, blinded
    pat = _identity_pattern(SimpleNamespace(model='claude-opus-5-5', kind='claude_cli'), SimpleNamespace(model='deepseek-chat', kind='openai_compat'))
    out = blinded({'hypotheses': [{'text': 'Claude from Anthropic proposes this', 'edits': [{'file': 'a', 'text': 'claude = 1'}]}]}, pat)
    assert 'Claude' not in out['hypotheses'][0]['text'] and out['hypotheses'][0]['edits'][0]['text'] == 'claude = 1'


def test_an_empty_selection_means_none_is_run():
    main = backend([{'hypotheses': [hyp('m1')]}, {'winner': 'A', 'selected': [], 'analysis': 'neither is worth a run'}], 'm')
    second = backend([{'hypotheses': [hyp('s1')]}], 's')
    res = DebateDriver(main, second, seed=3).complete('S', 'P', schema=SCHEMA, role='hypotheses')
    assert res.action['hypotheses'] == []


def test_the_agents_memory_records_stay_as_written():
    from infrarsi.agent.debate import _identity_pattern, blinded
    pat = _identity_pattern(SimpleNamespace(model='deepseek-chat', kind='openai_compat'), SimpleNamespace(model='gpt-6', kind='openai_compat'))
    entry = {'tags': {'model': ['deepseek']}, 'slow': {'b': 'gpt'}, 'fast': {'b': 'deepseek'}, 'lesson': 'DeepSeek backend is fine'}
    out = blinded({'action': 'write_memory', 'entry': entry}, pat)['entry']
    assert out['tags'] == entry['tags'] and out['slow'] == entry['slow'] and out['fast'] == entry['fast'] and 'DeepSeek' not in out['lesson']


def test_text_next_to_code_is_blinded_and_the_agents_payloads_are_kept():
    from infrarsi.agent.debate import _identity_pattern, blinded
    pat = _identity_pattern(SimpleNamespace(model='claude-opus-5-5', kind='claude_cli'), SimpleNamespace(model='deepseek-chat', kind='openai_compat'))
    reply = {'hypotheses': [{'edits': [{'file': 'a.py', 'old': 'x', 'new': 'y', 'reason': 'Claude from Anthropic proposes this edit', '_author': 'main'}]}],
             'action': 'propose_patch', 'content': 'if m == "deepseek": pass', 'files': {'b.py': 'claude = 1'}, 'config': {'name': 'deepseek'}}
    out = blinded(reply, pat)
    assert 'Claude' not in str(out['hypotheses']) and 'Anthropic' not in str(out['hypotheses']) and '_author' not in str(out['hypotheses'])
    assert out['content'] == reply['content'] and out['files'] == reply['files'] and out['config'] == reply['config']


@pytest.mark.parametrize('rounds', [3, -1, 'x'])
def test_the_driver_refuses_more_than_two_rounds(rounds):
    with pytest.raises(ValueError, match='two rounds'):
        DebateDriver(backend([], 'm'), backend([], 's'), rounds=rounds)


def test_the_patch_check_respects_the_strategys_dollar_cap():
    main = backend([], 'm')
    second = backend([{'verdict': 'pass'}], 's')
    d = DebateDriver(main, second)
    engine, _ = _engine()

    class Budget:                                        # (as BudgetDriver: the strategy's cap around the model)
        def __init__(self, inner, left):
            self.inner, self.left = inner, left

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def usd_left(self):
            return self.left
    prop = _prop(Budget(d, 0.01))
    assert check_patches(prop, engine, 'c1', [_patch()]) == [_patch()] and not second.calls
    assert d.summary()['patch_checks'] == {'unchecked': 1}
    prop = _prop(Budget(d, 0.5))
    seen = _spied(second)
    check_patches(prop, engine, 'c1', [{**_patch(), 'patch': {'train.py': 'a = 5\nb = 2\n'}}])
    assert seen[0]['max_usd'] == 0.5


def test_no_dollars_left_means_no_second_opinion():
    main = backend([{'hypotheses': [hyp('m')]}], 'm')
    second = backend([{'hypotheses': [hyp('s')]}], 's')
    d = DebateDriver(main, second)
    res = d.complete('S', 'P', schema=SCHEMA, role='hypotheses', max_usd=0.05)
    assert not second.calls and [h['mechanism'] for h in res.action['hypotheses']] == ['m']
    assert d.summary()['outcomes'] == {'no_budget': 1}


@pytest.mark.parametrize('role', sorted(set(DEBATE_ROLES) - {'hypotheses', 'configs'}))
def test_tables_are_picked_whole(role):
    main = backend([{'levels': [1]}, {'winner': 'A'}], 'm')
    second = backend([{'levels': [2]}], 's')
    res = DebateDriver(main, second, seed=5).complete('S', 'P', role=role)
    assert res.action['levels'] in ([1], [2]) and len(main.calls) == 2


@pytest.mark.parametrize('role', ['preflight', 'patch_check', 'summarizer', 'debate_judge'])
def test_other_calls_are_the_main_models(role):
    main = backend([{'ok': True}], 'm')
    second = backend([], 's')
    res = DebateDriver(main, second).complete('S', 'P', role=role)
    assert res.action == {'ok': True} and not second.calls


def test_attributes_read_and_write_through_to_the_main_model():
    main = backend([], 'main-model')
    d = DebateDriver(main, backend([], 's'))
    assert d.model == 'main-model' and d.tokens_per_call == 1000
    d.max_retries = 3
    assert main.max_retries == 3
    assert d.describe()['debate']['second'] == 's'


def test_identity_words():
    d = SimpleNamespace(model='gpt-6.1-sol', kind='codex_cli')
    assert {'gpt', 'codex', 'openai'} <= set(identity_words(d))
    assert 'chat' not in identity_words(SimpleNamespace(model='deepseek-chat', kind='openai_compat'))


def test_the_log_is_written_and_counted_again_on_resume(tmp_path):
    main = backend([{'hypotheses': [hyp('m')]}, {'winner': 'A'}], 'm')
    second = backend([{'hypotheses': [hyp('s')]}], 's')
    d = DebateDriver(main, second, seed=1)
    d.attach(tmp_path)
    d.complete('S', 'P', schema=SCHEMA, role='hypotheses')
    rows = [json.loads(x) for x in (tmp_path / 'debates.jsonl').read_text(encoding='utf-8').splitlines()]
    assert rows[0]['outcome'] == 'judged' and set(rows[0]['labels'].values()) == {'main', 'second'}
    d2 = DebateDriver(backend([], 'm'), backend([], 's'))
    d2.attach(tmp_path)
    assert d2.summary()['debates'] == 1 and d2.seq == 1


# ---------------------------------------------------------------- the patch check
class Hooks:
    def __init__(self):
        self.files = {'train.py': 'a = 1\nb = 2\n'}

    def candidate_files(self, overlay):
        return dict(self.files)

    def contract_text(self):
        return 'the contract'

    def verifier_checklist(self):
        return ['same workload']


def _engine():
    charged = []
    ledger = SimpleNamespace(used={'llm_tokens': 0, 'llm_usd': 0.0}, budget=SimpleNamespace(llm_tokens=10 ** 7, llm_usd=None, wall_seconds=10 ** 6),
                             wall=lambda: 0.0)
    session = SimpleNamespace(candidates={'c1': SimpleNamespace(overlay={})}, ledger=ledger, charge_llm=lambda r, role: charged.append(role))
    return SimpleNamespace(s=session, log=None), charged


def _prop(driver):
    return LLMProposer(driver, Hooks())


def _patch():
    return {'mechanism': 'faster a', 'change': {}, 'patch': {'train.py': 'a = 2\nb = 2\n'}, 'rationale': 'why'}


def test_a_passed_patch_goes_on_and_the_other_model_checks_it():
    main = backend([], 'm')
    second = backend([{'verdict': 'pass', 'issues': []}], 's')
    d = DebateDriver(main, second)
    engine, charged = _engine()
    out = check_patches(_prop(d), engine, 'c1', [_patch()])
    assert out == [_patch()] and charged == ['patch_check'] and len(second.calls) == 1 and not main.calls
    assert '-a = 1' in second.calls[0]['prompt'] and 'the contract' in second.calls[0]['prompt']


def test_the_author_is_not_its_own_verifier():
    main = backend([{'verdict': 'pass'}], 'm')
    second = backend([], 's')
    d = DebateDriver(main, second)
    engine, _ = _engine()
    check_patches(_prop(d), engine, 'c1', [{**_patch(), 'author': 'second'}])
    assert len(main.calls) == 1 and not second.calls


def test_an_explicit_verifier_checks_every_patch():
    main, second = backend([], 'm'), backend([], 's')
    ver = backend([{'verdict': 'pass'}, {'verdict': 'pass'}], 'v')
    d = DebateDriver(main, second, verifier=ver)
    engine, _ = _engine()
    p2 = {**_patch(), 'author': 'second', 'patch': {'train.py': 'a = 3\nb = 2\n'}}
    check_patches(_prop(d), engine, 'c1', [_patch(), p2])
    assert len(ver.calls) == 2 and not main.calls and not second.calls


def test_a_failed_check_gets_one_revision_and_a_second_check():
    main = backend([{'edits': [{'file': 'train.py', 'old': 'a = 1', 'new': 'a = 9'}]}], 'm')
    second = backend([{'verdict': 'fail', 'issues': ['wrong value']}, {'verdict': 'pass'}], 's')
    d = DebateDriver(main, second)
    engine, charged = _engine()
    out = check_patches(_prop(d), engine, 'c1', [_patch()])
    assert out[0]['patch'] == {'train.py': 'a = 9\nb = 2\n'} and charged == ['patch_check', 'patch_revise', 'patch_check']
    assert 'wrong value' in main.calls[0]['prompt']
    assert d.summary()['patch_checks'] == {'revised_passed': 1}


def test_at_most_two_checks_then_the_patch_is_dropped():
    main = backend([{'edits': [{'file': 'train.py', 'old': 'a = 1', 'new': 'a = 9'}]}], 'm')
    second = backend([{'verdict': 'fail', 'issues': ['one']}, {'verdict': 'fail', 'issues': ['two']}, {'verdict': 'pass'}], 's')
    d = DebateDriver(main, second)
    engine, _ = _engine()
    prop = _prop(d)
    assert check_patches(prop, engine, 'c1', [_patch()]) == []
    assert len(second.calls) == 2 and d.summary()['patch_checks'] == {'rejected': 1}
    assert 'two' in prop.dropped[-1]['problems'][0]      # the model is told why


def test_one_round_only_checks_once():
    main = backend([], 'm')
    second = backend([{'verdict': 'fail', 'issues': ['x']}], 's')
    d = DebateDriver(main, second, rounds=1)
    engine, _ = _engine()
    assert check_patches(_prop(d), engine, 'c1', [_patch()]) == [] and len(second.calls) == 1 and not main.calls


def test_an_unusable_revision_stands_as_the_failed_check():
    main = backend([{'edits': [{'file': 'train.py', 'old': 'not in the file', 'new': 'x'}]}], 'm')
    second = backend([{'verdict': 'fail', 'issues': ['x']}], 's')
    d = DebateDriver(main, second)
    engine, _ = _engine()
    assert check_patches(_prop(d), engine, 'c1', [_patch()]) == [] and len(second.calls) == 1


def test_a_check_without_an_answer_leaves_the_patch():
    main = backend([], 'm')
    second = backend(['not json'], 's')
    d = DebateDriver(main, second)
    engine, _ = _engine()
    assert check_patches(_prop(d), engine, 'c1', [_patch()]) == [_patch()]
    assert d.summary()['patch_checks'] == {'unchecked': 1}


def test_a_repeated_patch_is_not_checked_again():
    main = backend([], 'm')
    second = backend([{'verdict': 'pass'}], 's')
    d = DebateDriver(main, second)
    engine, _ = _engine()
    prop = _prop(d)
    check_patches(prop, engine, 'c1', [_patch()])
    assert check_patches(prop, engine, 'c1', [_patch()]) == [_patch()] and len(second.calls) == 1


def test_knob_hypotheses_are_not_checked():
    main, second = backend([], 'm'), backend([], 's')
    d = DebateDriver(main, second)
    engine, _ = _engine()
    item = {'mechanism': 'k', 'change': {'k': 1}, 'patch': None}
    assert check_patches(_prop(d), engine, 'c1', [item]) == [item] and not second.calls


def test_without_a_debate_nothing_is_checked():
    main = backend([], 'm')
    engine, _ = _engine()
    assert check_patches(_prop(main), engine, 'c1', [_patch()]) == [_patch()] and not main.calls


def test_rounds_zero_is_no_check():
    main, second = backend([], 'm'), backend([], 's')
    d = DebateDriver(main, second, rounds=0)
    engine, _ = _engine()
    assert check_patches(_prop(d), engine, 'c1', [_patch()]) == [_patch()] and not second.calls


def test_a_resumed_run_goes_on_after_a_line_a_killed_process_did_not_finish(tmp_path):
    row = {'event': 'debate', 'seq': 0, 'role': 'prior', 'outcome': 'judged', 'winner': 'second'}
    path = tmp_path / 'debates.jsonl'
    path.write_bytes((json.dumps(row) + '\n{"analysis": "\u5012').encode('utf-8')[:-1])      # cut inside a character
    d = DebateDriver(backend([], 'm'), backend([], 's'))
    d.attach(tmp_path)
    assert d.seq == 1
    d.record_check({'outcome': 'passed'})
    assert [r['event'] for r in logged(path)] == ['debate', 'patch_check']


def test_a_model_left_on_its_default_is_shown_by_its_backend():
    assert model_name(backend([], 'default')) == 'scripted' and model_name(backend([], 'gpt-x')) == 'gpt-x' and model_name(None) is None
