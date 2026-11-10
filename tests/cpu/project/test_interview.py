"""The startup questions: Enter keeps each value; answers land on the run's flags or in infrarsi.yaml."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from infrarsi.cli.main import build, main
from infrarsi.project import Project, interview
from infrarsi.project import llm as llm_mod
from infrarsi.project.interview import Interview, Quit, RequirementsConfirm, questions

from test_project import toy  # noqa: F401 - the toy project fixture


def namespace(**kw):
    base = dict(tolerance=None, strategy=None, screens=None, wall_hours=None, budget_tokens=None, budget_usd=None, target_speedup=None,
                target_mfu=None, llm=None, api_base=None, api_key_env=None, on_interrupt=None, debate=None, second_llm=None, verifier_llm=None,
                requirements=None, interactive_requirements=None, interactive=None, yes=False)
    return SimpleNamespace(**{**base, **kw})


class Script:
    """Answers the questions in order; records every prompt."""

    def __init__(self, *answers):
        self.answers, self.prompts, self.said = list(answers), [], []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError(f'one question too many: {prompt}')
        return self.answers.pop(0)

    def say(self, text):
        self.said.append(text)


def qa(project, a, script):
    from functools import partial
    q = Interview(project, a, read=script, write=script.say)
    q.run = partial(q.run, walk=True)
    return q


def _index(label) -> int:
    return next(i for i, item in enumerate(questions()) if item.label.startswith(label))


def enter_through(project, a) -> int:
    """How many items the project has (one Enter each)."""
    ctx = Interview(project, a)
    return sum(1 for item in questions() if item.when(ctx))


def answers_at(project, a, wanted: dict) -> list:
    """One answer per item in order: Enter, except the items of `wanted` {label prefix: answer}."""
    ctx = Interview(project, a)
    out = []
    for item in questions():
        if item.when(ctx):
            out.append(next((v for k, v in wanted.items() if item.label.startswith(k)), ''))
    return out


# ---------------------------------------------------------------- Enter is the default configuration
def test_enter_keeps_everything(toy):
    a = namespace()
    before = (toy.root / 'infrarsi.yaml').read_text(encoding='utf-8')
    script = Script(*[''] * enter_through(toy, a), '')
    q = qa(toy, a, script)
    q.run()
    assert not script.answers and not q.changes and q.save() is None
    assert a == namespace() and (toy.root / 'infrarsi.yaml').read_text(encoding='utf-8') == before
    assert not (toy.root / 'infrarsi.yaml.bak').exists()


def test_every_question_shows_its_value(toy):
    a = namespace()
    script = Script(*[''] * enter_through(toy, a), '')
    qa(toy, a, script).run()
    assert all(p.rstrip().endswith(']:') for p in script.prompts[:-1])
    assert any('Loss tolerance [0.03]' in p for p in script.prompts) and any('Dual-model debate at proposals [off]' in p for p in script.prompts)


def test_a_accepts_the_rest(toy):
    a = namespace()
    script = Script('', 'a', '')                         # one Enter, then "the rest", then the final Enter
    qa(toy, a, script).run()
    assert not script.answers and len(script.prompts) == 3


def test_q_quits(toy):
    with pytest.raises(Quit):
        qa(toy, namespace(), Script('q')).run()


def test_a_closed_stdin_runs_nothing(toy):
    def closed(prompt):
        raise EOFError
    with pytest.raises(Quit):
        Interview(toy, namespace(), read=closed, write=lambda m: None).run()


def test_help_and_a_wrong_answer_ask_again(toy):
    a = namespace()
    answers = answers_at(toy, a, {'Loss tolerance': '?'})
    i = _index('Loss tolerance')
    n_before = sum(1 for item in questions()[:i] if item.when(Interview(toy, a)))
    answers[n_before:n_before] = ['nonsense']
    answers[n_before + 1:n_before + 1] = ['5']           # (out of range: asked again)
    answers.insert(n_before + 3, '')
    script = Script(*answers, '')
    qa(toy, a, script).run()
    assert a.tolerance is None and any('a number' in x for x in script.said) and any('fraction' in x for x in script.said)


def test_the_final_go_ahead_can_change_an_item(toy):
    a = namespace()
    answers = [''] * enter_through(toy, a)
    n = _index('Loss tolerance') + 1
    script = Script(*answers, str(n), '2%', '')
    qa(toy, a, script).run()
    assert a.tolerance == 0.02


def test_the_final_go_ahead_can_quit(toy):
    a = namespace()
    with pytest.raises(Quit):
        qa(toy, a, Script(*[''] * enter_through(toy, a), 'q')).run()


# ---------------------------------------------------------------- the answers
@pytest.mark.parametrize('label,answer,attr,value', [
    ('Loss tolerance', '0.05', 'tolerance', 0.05), ('Loss tolerance', '5%', 'tolerance', 0.05),
    ('Target speed-up', '1.5', 'target_speedup', 1.5), ('Target MFU', '45', 'target_mfu', 0.45),
    ('Search budget: full runs', '8', 'screens', 8), ('Search budget: wall clock', '90m', 'wall_hours', 1.5),
    ('Search budget: LLM tokens', '500k', 'budget_tokens', 500000), ('Search budget: LLM dollars', '$3', 'budget_usd', 3.0),
    ('Search strategy', 'tpe', 'strategy', 'tpe')])
def test_a_setting_lands_on_the_run(toy, label, answer, attr, value):
    a = namespace()
    script = Script(*answers_at(toy, a, {label: answer}), '')
    qa(toy, a, script).run()
    assert getattr(a, attr) == value


def test_none_clears_an_optional_setting(toy):
    a = namespace(target_speedup=1.5)
    script = Script(*answers_at(toy, a, {'Target speed-up': 'none'}), '')
    qa(toy, a, script).run()
    assert a.target_speedup is None


@pytest.mark.parametrize('label,answer,path,value', [
    ('Updates per run', '20', ('workload', 'steps'), 20), ('GPUs per node', '4', ('hardware', 'gpus'), 4),
    ('Timed updates', '6-12', ('workload', 'window'), [6, 12]), ('Code files', 'none', ('code', 'mutable'), [])])
def test_a_manifest_field_is_written_back(toy, label, answer, path, value):
    a = namespace()
    script = Script(*answers_at(toy, a, {label: answer}), '')
    q = qa(toy, a, script)
    q.run()
    saved = q.save()
    doc = yaml.safe_load(saved.read_text(encoding='utf-8'))
    cur = doc
    for key in path:
        cur = cur[key]
    assert cur == value
    assert Project.load(saved)                           # it still loads
    assert (toy.root / 'infrarsi.yaml.bak').is_file()


def test_a_field_that_would_break_the_manifest_is_asked_again(toy):
    a = namespace()
    answers = answers_at(toy, a, {'Timed updates': '6-999'})
    i = next(n for n, x in enumerate(answers) if x == '6-999')
    answers.insert(i + 1, '6-12')
    script = Script(*answers, '')
    q = qa(toy, a, script)
    q.run()
    assert q.changes[('workload', 'window')] == [6, 12] and any('window' in x for x in script.said)


def test_the_comments_at_the_top_of_the_manifest_stay(toy):
    path = toy.root / 'infrarsi.yaml'
    path.write_text('# NOTE: read me\n' + path.read_text(encoding='utf-8'), encoding='utf-8')
    toy = Project.load(path)
    a = namespace()
    q = qa(toy, a, Script(*answers_at(toy, a, {'Updates per run': '20'}), ''))
    q.run()
    assert q.save().read_text(encoding='utf-8').startswith('# NOTE: read me\n')


def test_knobs_can_be_held(toy):
    a = namespace()
    q = qa(toy, a, Script(*answers_at(toy, a, {'Knobs the search': '-precision,compile'}), ''))
    q.run()
    assert q.project.knobs['precision']['choices'] == ['fp32'] and q.project.knobs['compile']['choices'] == [False]
    assert len(q.project.knobs['workers']['choices']) > 1


def test_only_lets_only_these_knobs_change(toy):
    a = namespace()
    q = qa(toy, a, Script(*answers_at(toy, a, {'Knobs the search': 'only workers'}), ''))
    q.run()
    assert all(len(v['choices']) == 1 for k, v in q.project.knobs.items() if k != 'workers')


def test_an_unknown_knob_is_named(toy):
    a = namespace()
    answers = answers_at(toy, a, {'Knobs the search': '-nope'})
    i = answers.index('-nope')
    answers.insert(i + 1, '')
    script = Script(*answers, '')
    qa(toy, a, script).run()
    assert any('not a knob' in x for x in script.said)


def test_an_invariant_can_be_added(toy):
    a = namespace()
    q = qa(toy, a, Script(*answers_at(toy, a, {'What must stay': 'total=micro_batch_size*2'}), ''))
    q.run()
    assert q.project.raw['workload']['invariants']['total'] == 'micro_batch_size*2'


def test_the_api_questions_come_only_with_an_api_model(toy):
    a = namespace()
    script = Script(*[''] * enter_through(toy, a), '')
    qa(toy, a, script).run()
    assert not any('API base' in p for p in script.prompts)
    a = namespace(llm='deepseek:deepseek-chat')
    n = enter_through(toy, a)
    script = Script(*[''] * n, '')
    qa(toy, a, script).run()
    assert any('API base' in p for p in script.prompts) and any('API key' in p for p in script.prompts)


def test_a_key_is_never_asked_for_only_its_variable(toy):
    a = namespace(llm='deepseek:deepseek-chat')
    script = Script(*answers_at(toy, a, {'API key variable': 'MY_KEY'}), '')
    qa(toy, a, script).run()
    assert a.api_key_env == 'MY_KEY' and any('not set in this shell' in x for x in script.said)


def test_the_debate_questions_open_when_it_is_turned_on(toy, monkeypatch):
    monkeypatch.setattr(llm_mod, 'make_driver', lambda s, **k: object())
    a = namespace()
    answers = answers_at(toy, a, {'Dual-model debate': 'y'})
    i = answers.index('y')
    answers[i + 1:i + 1] = ['codex-cli', 'none']         # the two questions the "y" opens
    script = Script(*answers, '')
    qa(toy, a, script).run()
    assert a.debate is True and a.second_llm == 'codex-cli' and a.verifier_llm == 'none'


def test_the_debate_is_off_unless_asked_for(toy):
    a = namespace()
    script = Script(*[''] * enter_through(toy, a), '')
    qa(toy, a, script).run()
    assert a.debate is None and not any('Second model' in p for p in script.prompts)


def test_an_unknown_backend_is_asked_again(toy):
    a = namespace()
    answers = answers_at(toy, a, {'Model': 'bogus'})
    i = answers.index('bogus')
    answers.insert(i + 1, '')
    script = Script(*answers, '')
    qa(toy, a, script).run()
    assert a.llm is None and any('unknown backend' in x for x in script.said)


def test_a_custom_endpoint_can_be_given_after_the_model(toy, monkeypatch):
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    monkeypatch.delenv('OPENROUTER_BASE_URL', raising=False)
    a = namespace()
    answers = answers_at(toy, a, {'Model': 'openai-compat:local'})
    i = answers.index('openai-compat:local')
    answers[i + 1:i + 1] = ['http://127.0.0.1:9/v1', 'MY_KEY']   # the two questions the model opens
    script = Script(*answers, '')
    qa(toy, a, script).run()
    assert a.llm == 'openai-compat:local' and a.api_base == 'http://127.0.0.1:9/v1' and a.api_key_env == 'MY_KEY'


def test_a_model_that_needs_a_name_is_asked_again(toy):
    a = namespace()
    answers = answers_at(toy, a, {'Model': 'openai-compat'})
    i = answers.index('openai-compat')
    answers.insert(i + 1, '')
    script = Script(*answers, '')
    qa(toy, a, script).run()
    assert a.llm is None and any('needs a model name' in x for x in script.said)


def test_none_clears_a_value_the_manifest_set(toy):
    toy.raw['search']['budget'].update(target_speedup=1.5, llm_usd=3.0)
    toy.raw['llm'] = {'backend': 'openai-compat', 'model': 'm', 'base_url': 'http://x/v1'}
    a = namespace()
    q = qa(toy, a, Script(*answers_at(toy, a, {'Target speed-up': 'none', 'Search budget: LLM dollars': 'none', 'API base URL': 'none'}), ''))
    q.run()
    assert q.project.raw['search']['budget'].get('target_speedup') is None
    assert q.project.raw['search']['budget'].get('llm_usd') is None and q.project.raw['llm'].get('base_url') is None


def test_none_clears_a_target_set_under_search_stop_too(toy):
    toy.raw['search']['stop'] = {'target_speedup': 1.5, 'target_mfu': 0.45}
    toy.raw['search']['budget'].update(target_speedup=None, target_mfu=None)
    from infrarsi.project import Project as P
    p = P(toy.raw, toy.path)
    assert p.raw['search']['budget']['target_speedup'] == 1.5
    a = namespace()
    q = qa(p, a, Script(*answers_at(p, a, {'Target speed-up': 'none', 'Target MFU': 'none'}), ''))
    q.run()
    assert q.project.raw['search']['budget'].get('target_speedup') is None and q.project.raw['search']['budget'].get('target_mfu') is None


def test_clearing_an_api_field_shows_in_the_summary(toy):
    toy.raw['llm'] = {'backend': 'openai', 'model': 'test', 'base_url': 'http://old.invalid/v1', 'api_key_env': 'OLD_KEY'}
    a = namespace()
    q = Interview(toy, a, read=Script(*answers_at(toy, a, {'API base URL': 'none', 'API key variable': 'none'}), ''), write=lambda m: None,
                  llm_spec={'backend': 'openai', 'model': 'test', 'base_url': 'http://old.invalid/v1', 'api_key_env': 'OLD_KEY'})
    q.run(walk=True)
    assert 'base_url' not in q.llm_spec and 'api_key_env' not in q.llm_spec


def test_a_detected_url_that_was_cleared_is_not_used(toy, tmp_path, monkeypatch, capsys):
    from infrarsi.project import autopilot
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    for name in ('INFRARSI_LLM', 'DEEPSEEK_API_KEY', 'ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'OPENROUTER_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('VLLM_URL', 'http://vllm.remote.invalid:8000/v1')
    made = []

    def make(spec, **kw):
        made.append(llm_mod.parse(spec))
        return None
    monkeypatch.setattr(llm_mod, 'make_driver', make)
    a = namespace()
    ctx = Interview(toy, a, llm_spec=autopilot.detect_llm_spec(toy, None))
    answers = [{'API base URL': 'none', 'Search strategy': 'random', 'Search budget: full runs': '1'}.get(i.label, '') for i in questions() if i.when(ctx)]
    _terminal(monkeypatch, answers + [''])
    main(['optimize', str(toy.root), '--no-doctor', '--out', str(tmp_path / 'runs'), '--dry-run'])
    assert 'the backend\'s default' in capsys.readouterr().out
    assert made[-1]['backend'] == 'vllm' and 'base_url' not in made[-1]


def test_the_knob_question_lists_the_knobs(toy):
    s = Script('?', '')
    qa(toy, namespace(), s).ask(questions()[_index('Knobs the search may change')], 1, 1)
    text = ' '.join(s.said)
    assert all(f'{k}=' in text for k in toy.knobs) and "compile=False of true/false" in text and "workers=1 of [1, 2, 4]" in text


def test_auto_stays_auto_for_the_run_to_decide(toy):
    a = namespace()
    qa(toy, a, Script(*answers_at(toy, a, {'Search strategy': 'auto'}), '')).run()
    assert a.strategy == 'auto'


def test_a_second_answer_to_the_knob_question_replaces_the_first(toy):
    a = namespace()
    q = qa(toy, a, Script(*answers_at(toy, a, {'Knobs the search': 'only workers'}), str(_index('Knobs the search') + 1), 'only compile', ''))
    q.run()
    assert len(q.project.knobs['compile'].get('choices') or [False, True]) > 1
    assert q.project.knobs['workers']['choices'] == [1] and q.project.knobs['precision']['choices'] == ['fp32']


def test_the_second_answer_can_free_every_knob(toy):
    a = namespace()
    before = toy.knobs['workers']['choices']
    q = qa(toy, a, Script(*answers_at(toy, a, {'Knobs the search': 'only compile'}), str(_index('Knobs the search') + 1), '-precision', ''))
    q.run()
    assert q.project.knobs['workers']['choices'] == before and 'choices' not in q.project.knobs['compile']
    doc = yaml.safe_load(q.save().read_text(encoding='utf-8'))
    assert 'choices' not in doc['knobs']['compile']


def test_q_at_the_requirements_prompts_quits(toy):
    with pytest.raises(Quit):
        _confirm(toy, 'q').new_text(['x'])
    c = _confirm(toy, '', 'q')
    with pytest.raises(Quit):
        c(_rec(source={'interactive': ['keep this']}))


def test_q_while_typing_requirements_quits(toy):
    a = namespace()
    answers = answers_at(toy, a, {'Requirements in your own words': 'x'})
    i = answers.index('x')
    answers.insert(i + 1, 'q')
    with pytest.raises(Quit):
        qa(toy, a, Script(*answers, '')).run()


def test_quitting_at_the_compiled_requirements_runs_nothing(toy, tmp_path, monkeypatch, capsys):
    from infrarsi.cli.main import _apply_requirements
    a = namespace(requirements=['x'], requirements_file=None, no_requirements=False, requirements_partial=False,
                  confirm_requirements=None)
    from infrarsi.agent.backends.fake import ScriptedBackend
    drv = ScriptedBackend(lambda *x: {'items': [{'text': 'x', 'kind': 'advisory', 'advice': 'careful'}]}, tokens_per_call=10)
    confirm = RequirementsConfirm(qa(toy, a, Script('q')))
    a.confirm_requirements = confirm
    with pytest.raises(Exception) as e:
        _apply_requirements(a, toy, drv)
    assert 'stopped at the requirements' in str(e.value)


@pytest.mark.parametrize('flag,on_manifest', [(True, False), (False, True)])
def test_enter_keeps_a_configured_policy_for_unsupported_requirements(toy, flag, on_manifest):
    from infrarsi.cli.main import _apply_requirements
    if on_manifest:
        toy.raw['requirements'] = {'text': ['x'], 'on_unsupported': 'continue'}
    a = namespace(requirements=None if on_manifest else ['x'], requirements_file=None, no_requirements=False, requirements_partial=flag)
    from infrarsi.agent.backends.fake import ScriptedBackend
    drv = ScriptedBackend(lambda *x: {'items': [{'text': 'x', 'kind': 'objective'}]}, tokens_per_call=10)
    confirm = RequirementsConfirm(qa(toy, a, Script('')))
    a.confirm_requirements = confirm
    _, record = _apply_requirements(a, toy, drv)
    assert record['accepted_partial'] is True


def test_requirements_are_typed_line_by_line(toy):
    a = namespace()
    answers = answers_at(toy, a, {'Requirements in your own words': '不要动 loss'})
    i = answers.index('不要动 loss')
    answers[i + 1:i + 1] = ['batch 不能超过 4', '']
    script = Script(*answers, '')
    qa(toy, a, script).run()
    assert a.interactive_requirements == ['不要动 loss', 'batch 不能超过 4']


# ---------------------------------------------------------------- when it asks
@pytest.mark.parametrize('kw,tty,expect', [({}, True, True), ({}, False, False), ({'interactive': True}, False, True), ({'interactive': False}, True, False),
                                           ({'yes': True}, True, False)])
def test_when_it_asks(kw, tty, expect):
    term = SimpleNamespace(isatty=lambda: tty)
    assert interview.wanted(namespace(**kw), stdin=term, stdout=term) is expect


def test_a_suite_or_a_resume_is_not_asked():
    term = SimpleNamespace(isatty=lambda: True)
    assert not interview.wanted(namespace(), suite=True, stdin=term, stdout=term)
    assert not interview.wanted(namespace(), resume=True, stdin=term, stdout=term)


def test_a_pipe_asks_nothing(toy, tmp_path, monkeypatch, capsys):
    # the existing behaviour of scripts and CI: no question, no change
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    monkeypatch.setattr('builtins.input', lambda *a: pytest.fail('asked a question in a pipe'))
    main(['run', str(toy.root), '--llm', 'none', '--strategy', 'random', '--screens', '1', '--out', str(tmp_path / 'runs')])


def test_the_flags_parse():
    p = build()
    assert p.parse_args(['run', '-y']).yes is True
    assert p.parse_args(['run', '--no-interactive']).interactive is False
    assert p.parse_args(['optimize', '--interactive']).interactive is True
    assert p.parse_args(['run']).interactive is None


# ---------------------------------------------------------------- the command line, on a terminal
def _terminal(monkeypatch, answers):
    term = SimpleNamespace(isatty=lambda: True)
    monkeypatch.setattr(interview.sys, 'stdin', term)
    monkeypatch.setattr(interview.sys, 'stdout', SimpleNamespace(isatty=lambda: True, write=__import__('sys').stdout.write, flush=lambda: None))
    feed = iter(['walk', *answers])
    monkeypatch.setattr('builtins.input', lambda *a: next(feed))


def test_run_asks_on_a_terminal_and_uses_the_answers(toy, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    a = namespace(llm='none')
    answers = answers_at(toy, a, {'Search budget: full runs': '1', 'Search strategy': 'random', 'Model': 'none', 'Updates per run': '12'})
    _terminal(monkeypatch, answers + [''])
    main(['run', str(toy.root), '--out', str(tmp_path / 'runs')])
    r = json.loads((next((tmp_path / 'runs').iterdir()) / 'result.json').read_text(encoding='utf-8'))
    assert r['strategy'] == 'random' and r['search']['used']['screens'] == 1
    assert (toy.root / 'infrarsi.yaml.bak').is_file() or yaml.safe_load((toy.root / 'infrarsi.yaml').read_text(encoding='utf-8'))['workload']['steps'] == 12


def test_quitting_runs_nothing(toy, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    _terminal(monkeypatch, ['q'])
    with pytest.raises(SystemExit) as e:
        main(['run', str(toy.root), '--llm', 'none', '--out', str(tmp_path / 'runs')])
    assert e.value.code == 2 and 'stopped at the questions' in capsys.readouterr().err and not (tmp_path / 'runs').exists()


def test_optimize_asks_too(toy, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    a = namespace()
    answers = answers_at(toy, a, {'Search budget: full runs': '1', 'Search strategy': 'random', 'Model': 'none', 'Updates per run': '12'})
    _terminal(monkeypatch, answers + [''])
    main(['optimize', str(toy.root), '--no-doctor', '--out', str(tmp_path / 'runs'), '--dry-run'])
    assert 'stage 3/5 (run)' in capsys.readouterr().out


# ---------------------------------------------------------------- confirming the requirements
class Rec(dict):
    pass


def _rec(blocking=False, source=None):
    return {'items': [{'i': 1, 'status': 'objective' if blocking else 'enforced', 'text': 'x', 'words': 'y'}], 'counts': {}, 'source': source or {}}


def _confirm(toy, *answers):
    return RequirementsConfirm(qa(toy, namespace(), Script(*answers)))


def test_enter_goes_on_when_all_can_be_honoured(toy):
    assert _confirm(toy, '')(_rec()) == 'go'


def test_enter_quits_when_something_cannot_be_honoured(toy):
    assert _confirm(toy, '')(_rec(blocking=True)) == 'quit'
    assert _confirm(toy, 's')(_rec(blocking=True)) == 'skip'


def test_edit_collects_new_text(toy):
    c = _confirm(toy, 'e', 'new one', '')
    assert c(_rec()) == 'edit' and c.new_text(['old']) == ['new one']


def test_typed_requirements_can_be_kept_in_the_manifest(toy):
    c = _confirm(toy, '', 'y')
    assert c(_rec(source={'interactive': ['keep this']})) == 'go'
    assert yaml.safe_load((toy.root / 'infrarsi.yaml').read_text(encoding='utf-8'))['requirements']['text'] == ['keep this']
