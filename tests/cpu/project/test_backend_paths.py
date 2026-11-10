from types import SimpleNamespace

import pytest

from infrarsi.agent import backends
from infrarsi.cli.main import _api_overrides, _prepare_job, build
from infrarsi.project import autopilot
from infrarsi.project.llm import make_driver


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    for key in ('DEEPSEEK_API_KEY', 'DEEPSEEK_BASE_URL', 'INFRARSI_DEEPSEEK_MODEL', 'OPENAI_API_KEY', 'OPENAI_BASE_URL',
                'OPENROUTER_API_KEY', 'OPENROUTER_BASE_URL', 'ANTHROPIC_API_KEY', 'VLLM_URL', 'INFRARSI_LLM',
                'INFRARSI_CLAUDE_EXE', 'INFRARSI_CODEX_EXE', 'CLAUDE_CONFIG_DIR', 'CODEX_HOME'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(autopilot.Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(autopilot.shutil, 'which', lambda name: None)


def test_deepseek_defaults_overrides_and_price_table(monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'fake-deepseek-key')
    driver = make_driver('deepseek')
    assert (driver.model, driver.base_url, driver.api_key) == ('deepseek-chat', 'https://api.deepseek.com', 'fake-deepseek-key')
    assert driver.price == {'input_miss': .27, 'input_hit': .07, 'output': 1.10}
    assert make_driver('deepseek:deepseek-reasoner').price['output'] == 2.19
    monkeypatch.setenv('DEEPSEEK_BASE_URL', 'https://test.example/v1')
    monkeypatch.setenv('INFRARSI_DEEPSEEK_MODEL', 'deepseek-reasoner')
    assert make_driver('deepseek').model == 'deepseek-reasoner'
    assert make_driver('deepseek').base_url == 'https://test.example/v1'
    monkeypatch.setenv('MY_KEY', 'fake-explicit-key')
    assert make_driver({'backend': 'deepseek', 'base_url': 'https://explicit.example', 'model': 'deepseek-chat',
                        'api_key_env': 'MY_KEY'}).api_key == 'fake-explicit-key'


def test_compatible_api_environment_and_explicit_precedence(monkeypatch):
    monkeypatch.setenv('OPENROUTER_BASE_URL', 'https://router.example/v1')
    monkeypatch.setenv('OPENROUTER_API_KEY', 'fake-router-key')
    assert make_driver('openai-compat:model').api_key == 'fake-router-key'
    assert make_driver('openai-compat:model').base_url == 'https://router.example/v1'
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://openai.example/v1')
    monkeypatch.setenv('OPENAI_API_KEY', 'fake-openai-key')
    monkeypatch.setenv('KEY', 'fake-explicit-key')
    for kind in ('openai', 'openai-compat'):
        driver = make_driver(f'{kind}:model')
        assert driver.base_url == 'https://openai.example/v1' and driver.api_key == 'fake-openai-key'
        driver = make_driver({'backend': kind, 'model': 'model', 'base_url': 'https://explicit.example', 'api_key_env': 'KEY'})
        assert driver.base_url == 'https://explicit.example' and driver.api_key == 'fake-explicit-key'


@pytest.mark.parametrize('name,cls', [('claude', 'ClaudeCLIBackend'), ('claude-cli', 'ClaudeCLIBackend'),
                                    ('codex', 'CodexCLIBackend'), ('codex-cli', 'CodexCLIBackend')])
def test_cli_aliases(name, cls, monkeypatch):
    monkeypatch.setattr(backends, cls, lambda **kw: kw)
    opts = make_driver(name)
    assert opts['max_retries'] == 1
    if cls == 'ClaudeCLIBackend':
        assert opts['strict_audit'] is False
    else:
        assert opts['model'] == 'default'


def test_detection_precedence_and_local_cli_config(monkeypatch, tmp_path):
    project = SimpleNamespace(raw={})
    assert autopilot.detect_llm_spec(project) == {'backend': 'none'}
    (tmp_path / '.codex').mkdir()
    (tmp_path / '.codex/config.toml').write_text('model = "local-model"\n', encoding='utf-8')
    monkeypatch.setattr(autopilot.shutil, 'which', lambda name: '/installed/' + name)
    assert autopilot.detect_llm_spec(project) == {'backend': 'codex-cli'}
    (tmp_path / '.claude.json').write_text('{}', encoding='utf-8')
    assert autopilot.detect_llm_spec(project) == {'backend': 'claude-cli'}
    for key, backend in [('VLLM_URL', 'vllm'), ('OPENROUTER_API_KEY', 'openai-compat'), ('OPENAI_API_KEY', 'openai'),
                         ('ANTHROPIC_API_KEY', 'anthropic'), ('DEEPSEEK_API_KEY', 'deepseek')]:
        monkeypatch.setenv(key, 'fake-test-value')
        assert autopilot.detect_llm_spec(project)['backend'] == backend
    assert autopilot.detect_llm_spec(project, 'none') == {'backend': 'none'}
    assert autopilot.detect_llm_spec(SimpleNamespace(raw={'llm': 'codex'})) == {'backend': 'codex'}


@pytest.mark.parametrize('command', ['run', 'optimize'])
def test_cli_endpoint_overrides_manifest(command):
    a = build().parse_args([command, '--api-base', 'https://custom.example/v1', '--api-key-env', 'CUSTOM_KEY'])
    project = SimpleNamespace(raw={'llm': {'backend': 'openai-compat', 'model': 'model', 'base_url': 'https://old.example'}})
    spec = _api_overrides(project, a)
    assert spec['base_url'] == 'https://custom.example/v1' and spec['api_key_env'] == 'CUSTOM_KEY'
    assert make_driver(spec).base_url == spec['base_url']


def test_run_preparation_applies_cli_endpoint_and_key_after_model_override(tmp_path, monkeypatch):
    manifest = tmp_path / 'infrarsi.yaml'
    manifest.write_text('launch: {command: "python train.py"}\n'
                        'llm: {backend: openai-compat, model: old, base_url: "https://old.example/v1"}\n', encoding='utf-8')
    monkeypatch.setenv('CUSTOM_KEY', 'fake-key')
    args = build().parse_args(['run', str(manifest), '--llm', 'openai-compat:new', '--api-base', 'https://new.example/v1',
                              '--api-key-env', 'CUSTOM_KEY'])
    project, driver = _prepare_job(args, manifest, None)
    assert project.raw['llm'] == {'backend': 'openai-compat', 'model': 'new', 'base_url': 'https://new.example/v1',
                                  'api_key_env': 'CUSTOM_KEY'}
    assert driver.model == 'new' and driver.base_url == 'https://new.example/v1' and driver.api_key == 'fake-key'


def test_anthropic_custom_endpoint_and_key_reach_sdk(monkeypatch):
    import sys
    monkeypatch.setenv('CUSTOM_ANTHROPIC_KEY', 'test-key')
    seen = {}
    monkeypatch.setitem(sys.modules, 'anthropic', SimpleNamespace(Anthropic=lambda **kw: seen.update(kw) or object()))
    b = make_driver({'backend': 'anthropic', 'base_url': 'https://custom.example', 'api_key_env': 'CUSTOM_ANTHROPIC_KEY'})
    b.client()
    assert seen['base_url'] == 'https://custom.example' and seen['api_key'] == 'test-key'

