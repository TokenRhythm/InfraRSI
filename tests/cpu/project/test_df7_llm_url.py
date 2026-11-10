"""The F phase (2026-10-24): manifests with `llm.base_url: $VLLM_URL` sent the text as written; every call failed with
"unknown url type" and the searches went on without the model. The variable is expanded; a value that is still not an
http(s) URL is refused before any GPU time (make_driver's ValueError: `infrarsi run` stops, the doctor says it)."""
import pytest

from infrarsi.project.llm import make_driver


def test_base_url_may_name_an_environment_variable(monkeypatch):
    monkeypatch.setenv('VLLM_URL', 'http://10.1.2.3:8000/v1')
    assert make_driver({'backend': 'vllm', 'model': 'Q', 'base_url': '$VLLM_URL'}).base_url == 'http://10.1.2.3:8000/v1'
    monkeypatch.setenv('HOST_A', '10.1.2.4')
    assert make_driver({'backend': 'openai-compat', 'model': 'Q',
                        'base_url': 'http://${HOST_A}:8001/v1'}).base_url == 'http://10.1.2.4:8001/v1'


def test_a_base_url_that_is_not_a_url_stops_before_the_run(monkeypatch):
    monkeypatch.delenv('VLLM_URL', raising=False)
    with pytest.raises(ValueError, match='the environment variable VLLM_URL is not set'):
        make_driver({'backend': 'vllm', 'model': 'Q', 'base_url': '$VLLM_URL'})
    with pytest.raises(ValueError, match='not an http'):
        make_driver({'backend': 'openai-compat', 'model': 'Q', 'base_url': 'localhost:8000/v1'})
    monkeypatch.delenv('VLLM_PORT', raising=False)                  # DF7_r1 #3: an unset variable inside the URL
    with pytest.raises(ValueError, match='the environment variable VLLM_PORT is not set'):
        make_driver({'backend': 'vllm', 'model': 'Q', 'base_url': 'http://127.0.0.1:${VLLM_PORT}/v1'})
    with pytest.raises(ValueError, match='a valid port'):
        make_driver({'backend': 'vllm', 'model': 'Q', 'base_url': 'http://127.0.0.1:80x/v1'})
    assert make_driver({'backend': 'vllm', 'model': 'Q'}).base_url == 'http://127.0.0.1:8000/v1'   # the default
