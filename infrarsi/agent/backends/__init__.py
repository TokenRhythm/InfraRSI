"""LLM backends: Anthropic API (official SDK), Claude Code CLI, OpenAI-compatible endpoints, scripted (tests)."""
from .anthropic_api import AnthropicBackend
from .base import Backend, BackendError, LLMResult, is_refusal, parse_json_object
from .claude_cli import ClaudeCLIBackend, parse_stream
from .codex_cli import CodexCLIBackend, parse_codex_stream
from .fake import ScriptedBackend
from .openai_compat import OpenAICompatBackend

DEEPSEEK_URL = 'https://api.deepseek.com'
SELFHOST_URL = 'http://127.0.0.1:18000/v1'       # WIN0 vLLM through `ssh -L 18000:127.0.0.1:8000` (design E5)


def make_backend(kind: str, **kw):
    """'claude' (strong, claude-opus-5-5 CLI), 'deepseek' (R1 API, json_object), 'vllm' (R2/R4 self-hosted; JSON mode via
    response_format json_object, the setting verified on WIN0 with vLLM 0.26 - see agent.arms.weak_driver)."""
    if kind in ('claude', 'claude-cli'):
        return ClaudeCLIBackend(**kw)
    if kind in ('codex', 'codex-cli'):
        return CodexCLIBackend(**kw)
    if kind == 'anthropic':
        return AnthropicBackend(**kw)
    if kind == 'deepseek':
        kw.setdefault('base_url', DEEPSEEK_URL)
        kw.setdefault('model', 'deepseek-flash')
        return OpenAICompatBackend(**kw)
    if kind == 'vllm':
        kw.setdefault('base_url', SELFHOST_URL)
        kw.setdefault('api_key_env', None)
        kw.setdefault('price', None)
        kw.setdefault('structured', 'json_object')
        return OpenAICompatBackend(**kw)
    raise ValueError(f'unknown backend kind {kind!r}')


__all__ = ['AnthropicBackend', 'Backend', 'BackendError', 'ClaudeCLIBackend', 'CodexCLIBackend', 'LLMResult',
           'OpenAICompatBackend', 'ScriptedBackend', 'is_refusal', 'make_backend', 'parse_json_object', 'parse_stream',
           'parse_codex_stream']
