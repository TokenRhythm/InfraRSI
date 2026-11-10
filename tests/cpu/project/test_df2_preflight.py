"""Base-model trial D1-16b-gpt56luna (2026-10-24): the provider refused the model from this network (HTTP 403); it was
found at the first call after the reference run and the search ran without the model. doctor and run now make one
small call first; run stops on a refusal."""
from types import SimpleNamespace

import pytest

from infrarsi.cli.main import _print_llm, main
from infrarsi.project import llm


class Driver:
    model = 'openai/gpt-5.6-luna'

    def __init__(self, error=None, meta=None):
        self.error, self.meta, self.calls = error, meta or {}, 0

    def complete(self, system, prompt, **kw):
        self.calls += 1
        return SimpleNamespace(action=None if self.error else {'ok': True}, error=self.error, llm_tokens=30, meta=self.meta)


def test_preflight_tells_a_refusal_from_a_failure():
    assert llm.preflight(Driver())['ok'] is True
    p = llm.preflight(Driver('HTTP 403: {"message": "The request is prohibited due to a violation of provider Terms"}'))
    assert p['ok'] is False and p['refused'] is True
    p = llm.preflight(Driver('timeout: 300s call budget used'))
    assert p['ok'] is False and p['refused'] is False
    # the Anthropic backend's errors name the SDK exception; its meta has the status code / unreachable (DF2_r1 #4)
    assert llm.preflight(Driver('AuthenticationError: Error code: 401', {'status_code': 401}))['refused'] is True
    assert llm.preflight(Driver('APIConnectionError: Connection error.', {'unreachable': True}))['refused'] is True


def _project(tmp_path):
    (tmp_path / 'infrarsi.yaml').write_text(
        'schema: infrarsi-project-v1\nname: t\nlaunch: {command: "bash run.sh --steps {steps} {args}"}\n'
        'knobs: {mbs: {type: int, choices: [1, 2], flag: --mbs, default: 1}}\n'
        'llm: {backend: openai-compat, model: openai/gpt-5.6-luna, base_url: "https://x/v1"}\n', encoding='utf-8')
    return tmp_path / 'infrarsi.yaml'


def test_run_stops_before_the_reference_when_the_model_is_refused(tmp_path, monkeypatch, capsys):
    path = _project(tmp_path)
    monkeypatch.setattr(llm, 'make_driver', lambda spec, **kw: Driver('HTTP 403: refused'))
    ran = []
    monkeypatch.setattr('infrarsi.project.flow.run_project', lambda *a, **k: ran.append(1))
    with pytest.raises(SystemExit):
        main(['run', str(path)])
    assert not ran and 'refused a test call: HTTP 403' in capsys.readouterr().err


def test_doctor_shows_whether_the_model_answers(tmp_path, monkeypatch, capsys):
    proj = SimpleNamespace(raw={'llm': {'backend': 'openai-compat', 'model': 'openai/gpt-5.6-luna'}})
    monkeypatch.setattr(llm, 'make_driver', lambda spec, **kw: Driver())
    _print_llm(proj)
    monkeypatch.setattr(llm, 'make_driver', lambda spec, **kw: Driver('HTTP 403: refused'))
    _print_llm(proj)
    out = capsys.readouterr().out
    assert 'llm: openai-compat openai/gpt-5.6-luna answers' in out and 'refused a test call: HTTP 403' in out
    assert '--llm none' in out


def test_strategies_that_never_call_the_model_are_not_stopped(tmp_path, monkeypatch):
    # DF2_r1 #2: a suite's tpe / random baselines keep running when the project's model is refused
    path = _project(tmp_path)
    monkeypatch.setattr(llm, 'make_driver', lambda spec, **kw: Driver('HTTP 403: refused'))
    ran = []
    monkeypatch.setattr('infrarsi.project.flow.run_project',
                        lambda *a, **k: ran.append(k['strategy']) or {'outcome': 'no_change', 'run_dir': str(tmp_path)})
    main(['run', str(path), '--strategy', 'tpe'])
    assert ran == ['tpe']


def test_doctor_takes_the_string_form_of_llm(monkeypatch, capsys):
    # DF2_r1 #3: `llm: vllm:local-model` (and `llm: anthropic`) are valid manifest values
    monkeypatch.setattr(llm, 'make_driver', lambda spec, **kw: Driver())
    _print_llm(SimpleNamespace(raw={'llm': 'vllm:local-model'}))
    assert 'llm: vllm openai/gpt-5.6-luna answers' in capsys.readouterr().out
