"""The dual-model debate through a whole project run (bcomp on the toy), its settings, and its being off by default."""
import json
from pathlib import Path

import pytest
import yaml

from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.agent.debate import DebateDriver
from infrarsi.cli.main import main
from infrarsi.project import Project
from infrarsi.project import llm as llm_mod
from infrarsi.project.flow import run_project
from infrarsi.project.llm import debate_settings, make_debate
from infrarsi.project.manifest import ManifestError

from test_bcomp_project import scripted
from test_project import toy  # noqa: F401 - the toy project fixture


def debaters(toy, *, reject_silent=False):
    """(main, second): the scripted model of test_bcomp_project as both, the second with an extra hypothesis; the judge takes
    every item of the proposal it calls A and the first of B."""
    main_drv, calls, _ = scripted(toy.root)
    plain = main_drv.script

    def checks(role, prompt):
        if role == 'patch_check':
            return {'verdict': 'fail', 'issues': ['a tweak with no effect']} if reject_silent and '_TWEAK' in prompt else {'verdict': 'pass'}
        return None

    def main_script(role, system, prompt):
        if role == 'debate_judge':
            view = json.loads(prompt)
            ids = [h['id'] for h in view['proposal_A'].get('hypotheses') or []] + [h['id'] for h in (view['proposal_B'].get('hypotheses') or [])[:1]]
            return {'winner': 'A', 'analysis': 'scripted', **({'selected': ids} if ids else {})}
        return checks(role, prompt) or plain(role, system, prompt)

    def second_script(role, system, prompt):
        made = checks(role, prompt)
        if made:
            return made
        got = plain(role, system, prompt)
        if role == 'hypotheses':
            got = {'hypotheses': got['hypotheses'][::-1]}     # the same ideas, its own order: not the same answer
        elif role == 'prior':
            got = {**got, 'notes': 'the second model'}
        return got
    main_drv.script = main_script
    return main_drv, ScriptedBackend(second_script, model='second-model', tokens_per_call=2000), calls


def test_a_whole_search_with_the_debate_on(toy, tmp_path, monkeypatch):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    m, s, _ = debaters(toy)
    d = DebateDriver(m, s, seed=7)
    res = run_project(Project.load(toy.root), strategy='bcomp', driver=d, budget={'screens': 8, 'llm_tokens': 10 ** 7},
                      out_root=tmp_path / 'out', log=lambda x: None)
    run = Path(res['run_dir'])
    rows = [json.loads(x) for x in (run / 'debates.jsonl').read_text(encoding='utf-8').splitlines()]
    debates = [r for r in rows if r['event'] == 'debate']
    assert {r['role'] for r in debates} >= {'prior', 'hypotheses'} and res['debate']['debates'] == len(debates)
    assert any(r['outcome'] == 'judged' for r in debates)
    assert any(r['event'] == 'patch_check' for r in rows)
    judge_calls = [c for c in m.calls if c['role'] == 'debate_judge']
    assert judge_calls and all('second-model' not in c['prompt'] and 'scripted-model' not in c['prompt'] for c in judge_calls)
    assert res['cost']['llm_tokens'] >= 2000 * (len(m.calls) + len(s.calls) - 1)   # every call charged the run's budget
    text = (run / 'REPORT.md').read_text(encoding='utf-8')
    assert '## Dual-model debate' in text and 'without knowing which was which' in text


def test_a_rejected_patch_costs_no_run(toy, tmp_path, monkeypatch):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    m, s, _ = debaters(toy, reject_silent=True)
    res = run_project(Project.load(toy.root), strategy='bcomp', driver=DebateDriver(m, s, seed=7), budget={'screens': 8, 'llm_tokens': 10 ** 7},
                      out_root=tmp_path / 'out', log=lambda x: None)
    assert res['debate']['patch_checks'].get('rejected', 0) >= 1
    assert not any('silent tweak' in str(c.get('rationale')) for c in res['candidates'])


def test_without_the_debate_there_is_no_trace_of_it(toy, tmp_path, monkeypatch):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    drv, calls, _ = scripted(toy.root)
    res = run_project(Project.load(toy.root), strategy='bcomp', driver=drv, budget={'screens': 8, 'llm_tokens': 10 ** 7},
                      out_root=tmp_path / 'out', log=lambda x: None)
    assert 'debate' not in res and not (Path(res['run_dir']) / 'debates.jsonl').exists()
    assert not {'debate_judge', 'patch_check', 'patch_revise'} & set(calls)
    assert '## Dual-model debate' not in (Path(res['run_dir']) / 'REPORT.md').read_text(encoding='utf-8')


def test_the_debate_is_off_by_default(toy):
    assert toy.raw['debate'] == {'enabled': False, 'second': None, 'verifier': None, 'verify_rounds': 2}
    assert debate_settings(toy.raw['debate'])['enabled'] is False


@pytest.mark.parametrize('kw,enabled', [({}, False), ({'enabled': True}, True), ({'second': 'x'}, True),
                                        ({'second': 'x', 'enabled': False}, False)])
def test_the_command_line_turns_it_on(kw, enabled):
    assert debate_settings({'enabled': False}, **kw)['enabled'] is enabled


def test_the_manifest_can_turn_it_on_and_the_flag_off():
    assert debate_settings({'enabled': True, 'second': 'x'})['enabled'] is True
    assert debate_settings({'enabled': True, 'second': 'x'}, enabled=False)['enabled'] is False


def test_at_most_two_rounds(toy):
    doc = yaml.safe_load((toy.root / 'infrarsi.yaml').read_text(encoding='utf-8'))
    doc['debate'] = {'enabled': True, 'second': 'x', 'verify_rounds': 3}
    (toy.root / 'infrarsi.yaml').write_text(yaml.safe_dump(doc), encoding='utf-8')
    with pytest.raises(ManifestError, match='verify_rounds'):
        Project.load(toy.root)


def test_make_debate(monkeypatch):
    made = []
    real = llm_mod.make_driver

    def fake(spec, **kw):
        made.append(spec)
        return None if spec in (None, 'none') else ScriptedBackend([], model=str(spec))
    monkeypatch.setattr(llm_mod, 'make_driver', fake)
    d = make_debate(ScriptedBackend([], model='main'), {'second': 'two', 'verifier': None, 'verify_rounds': 2})
    assert d.model == 'main' and d.second.model == 'two' and d.verifier is None and d.rounds == 2
    d = make_debate(ScriptedBackend([], model='main'), {'second': 'two', 'verifier': 'three'})
    assert d.verifier.model == 'three'
    d = make_debate(ScriptedBackend([], model='main'), {'second': 'two', 'verifier': 'none'})
    assert d.verifier is None and d.rounds == 0
    with pytest.raises(ValueError, match='second model'):
        make_debate(ScriptedBackend([], model='main'), {'second': None})
    assert real is not fake


def _fake_llms(monkeypatch, refuse=()):
    from infrarsi.agent.backends.fake import ScriptedBackend as SB

    def make(spec, **kw):
        s = llm_mod.parse(spec)
        if not s or s.get('backend') == 'none':
            return None
        if s.get('backend') in refuse:
            b = SB(lambda *a: 'refused', model=s.get('model') or s['backend'])
            real = b.complete
            b.complete = lambda *a, **k: __import__('dataclasses').replace(real(*a, **k), error='HTTP 403 forbidden', action=None)
            return b
        return SB(lambda *a: {'ok': True}, model=s.get('model') or s['backend'])
    monkeypatch.setattr(llm_mod, 'make_driver', make)


def test_a_debate_without_a_second_model_stops_before_any_run(toy, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    _fake_llms(monkeypatch)
    with pytest.raises(SystemExit) as e:
        main(['run', str(toy.root), '--llm', 'main-backend:m', '--strategy', 'bcomp', '--debate', '--screens', '1'])
    assert e.value.code == 2 and 'second model' in capsys.readouterr().err
    assert not list((toy.root / '.infrarsi').glob('runs/*'))


def test_a_refused_second_model_stops_before_any_run(toy, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    _fake_llms(monkeypatch, refuse=('badbackend',))
    with pytest.raises(SystemExit) as e:
        main(['run', str(toy.root), '--llm', 'main-backend:m', '--strategy', 'bcomp', '--second-llm', 'badbackend:x', '--screens', '1'])
    assert e.value.code == 2 and 'refused a test call' in capsys.readouterr().err
    assert not list((toy.root / '.infrarsi').glob('runs/*'))


def test_optico_with_the_debate_stops_before_any_run(toy, tmp_path, monkeypatch, capsys):
    from infrarsi.search import optico
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    _fake_llms(monkeypatch)
    monkeypatch.setattr(optico, 'problem', lambda driver: None)   # (OptiCo's fetched prompts are not what this is about)
    with pytest.raises(SystemExit) as e:
        main(['run', str(toy.root), '--llm', 'main-backend:m', '--strategy', 'optico', '--second-llm', 'two:x', '--screens', '1'])
    assert e.value.code == 2 and 'optico' in capsys.readouterr().err
    assert not list((toy.root / '.infrarsi').glob('runs/*'))


# ---------------------------------------------------------------- the native agent: who reviews a candidate
def agent_model(proposals, judge=None):
    """For the agent strategy: proposes `proposals` in turn, runs each, finishes; `judge` answers the debate's judgement."""
    steps = iter([x for i, p in enumerate(proposals) for x in ({'action': 'propose_config', 'changes': p, 'rationale': f'proposal {i}'},
                                                               {'action': 'run_screen', 'candidate_id': f'c{i + 1}'})]
                 + [{'action': 'finish', 'candidate_id': f'c{len(proposals)}'}])

    def reply(role, system, prompt):
        if role == 'proposer':
            return next(steps)
        if role == 'verifier':
            return {'verdict': 'pass', 'reasons': []}
        return judge(json.loads(prompt)) if role == 'debate_judge' else {}
    return reply


def _winner(view, knob):
    return {'winner': next(x for x in 'AB' if knob in view[f'proposal_{x}']['changes']), 'analysis': 'scripted'}


def _agent_debate(tmp_path, monkeypatch, toy, verifier=None):
    """The second model's proposal wins the first judgement, the main model's the second."""
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    wins = iter(['compile', 'fused_optimizer'])
    main_model = ScriptedBackend(agent_model([{'fused_optimizer': True}, {'fused_optimizer': True, 'workers': 2}],
                                             lambda view: _winner(view, next(wins))), model='main-m', tokens_per_call=500)
    second = ScriptedBackend(agent_model([{'compile': True}, {'compile': True, 'workers': 2}]), model='second-m', tokens_per_call=500)
    run_project(Project.load(toy.root), strategy='agent', driver=DebateDriver(main_model, second, verifier=verifier, seed=7),
                budget={'screens': 5, 'llm_tokens': 10 ** 7}, out_root=tmp_path / 'out', log=lambda x: None)
    return main_model, second


def _reviewed(model) -> list:
    return [json.loads(c['prompt'].split('\n', 1)[1])['candidate']['id'] for c in model.calls if c['role'] == 'verifier']


def test_an_agent_candidate_is_reviewed_by_the_model_that_did_not_write_it(toy, tmp_path, monkeypatch):
    m, s = _agent_debate(tmp_path, monkeypatch, toy)
    assert _reviewed(m) == ['c1'] and _reviewed(s) == ['c2']      # c1 is the second model's, c2 the main model's


def test_an_explicit_verifier_reviews_every_agent_candidate(toy, tmp_path, monkeypatch):
    v = ScriptedBackend(agent_model([]), model='third-m', tokens_per_call=500)
    m, s = _agent_debate(tmp_path, monkeypatch, toy, verifier=v)
    assert _reviewed(v) == ['c1', 'c2'] and not _reviewed(m) and not _reviewed(s)
