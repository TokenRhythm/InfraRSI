import json
import sys
from types import SimpleNamespace

import pytest

from agent_helpers import FIXTURES
from infrarsi.agent.backends import claude_cli as cc


def test_subscription_alias_and_user_configuration_are_accepted_only_when_relaxed():
    events = [json.loads(s) for s in (FIXTURES / 'stream_haiku_5_5_real.jsonl').read_text(encoding='utf-8').splitlines() if s]
    events[0].update(skills=['user-skill'], mcp_servers=[{'name': 'local'}], plugins=[{'name': 'custom', 'path': '/user'}],
                     tools=['mcp__configured_but_unused'])
    events[-1]['total_cost_usd'] = None
    raw = '\n'.join(json.dumps(e) for e in events)
    action, fields = cc.parse_stream(raw, model='opus', exit_code=0, status='RETURNED', strict_audit=False)
    assert action and fields['accounting_complete'] and fields['llm_usd'] == 0.0 and fields['llm_tokens'] == 2862
    assert cc.parse_stream(raw, model='opus', exit_code=0, status='RETURNED')[0] is None
    events[-1]['usage'].pop('output_tokens')
    assert cc.parse_stream('\n'.join(map(json.dumps, events)), model='opus', exit_code=0, status='RETURNED',
                           strict_audit=False)[0] is None


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows executable discovery')
def test_relaxed_windows_shims_and_native_preference(tmp_path, monkeypatch):
    shim = tmp_path / 'claude.cmd'
    shim.write_bytes(b'@echo off')
    monkeypatch.setattr(cc.Path, 'home', lambda: tmp_path)
    monkeypatch.delenv('INFRARSI_CLAUDE_EXE', raising=False)
    monkeypatch.setattr(cc.shutil, 'which', lambda name: str(shim) if name == 'claude' else None)
    monkeypatch.setattr(cc.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='version'))
    assert cc.ClaudeCLIBackend(strict_audit=False).executable == shim
    with pytest.raises(ValueError, match='native'):
        cc.ClaudeCLIBackend()
    native = tmp_path / '.local/bin/claude.exe'
    native.parent.mkdir(parents=True)
    native.write_bytes(b'exe')
    assert cc.ClaudeCLIBackend(strict_audit=False).executable == native

