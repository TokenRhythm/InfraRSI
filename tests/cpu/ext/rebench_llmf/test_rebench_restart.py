"""AMENDMENT_EXT_006 (A-NAT restart): sampling flags and request seed, the stop-only loop detector inside the real basic
agent loop, D-RB-16 on the restart's receipts, the abandoned record of the aborted run, the wall rule, the CLI dry run.
CPU only; no endpoint is contacted."""
import datetime
import json
import tempfile
from pathlib import Path

import pytest

from infrarsi.agent.backends import ScriptedBackend
from infrarsi.agent.budget import VirtualClock
from infrarsi.ext.rebench_llmf import arms, cli, native
from infrarsi.ext.rebench_llmf.driver import RequestSeed

import rebench_helpers as H


# ---------------------------------------------------------------- the loop detector
def test_twelve_identical_turns_in_a_row_trip_it():
    d = native.LoopDetector()
    act, out = {'tool': 'bash', 'cmd': 'cat finetune.yaml', 'timeout': 30}, {'exit_code': 0, 'stdout': 'x', 'stderr': ''}
    trips = [d.observe(dict(act, note=f'thinking {i}'), dict(out, seconds=0.1 * i), turn=i) for i in range(1, 13)]
    assert trips == [False] * 11 + [True]                                   # the note and the wall time are ignored
    assert d.tripped['turn'] == 12 and d.tripped['consecutive'] == 12 and d.n_identical_max == 12


def test_eleven_then_a_different_turn_does_not_trip_and_outputs_matter():
    d = native.LoopDetector()
    act = {'tool': 'bash', 'cmd': 'ls'}
    for i in range(11):
        assert not d.observe(act, {'stdout': 'a'}, turn=i)
    assert not d.observe({'tool': 'bash', 'cmd': 'pwd'}, {'stdout': 'a'})
    for i in range(11):
        assert not d.observe(act, {'stdout': f'changing {i}'})              # same command, different output
    assert d.tripped is None


def test_thirty_in_the_last_sixty_trip_it_even_when_interleaved():
    d = native.LoopDetector()
    a, b = ({'tool': 'bash', 'cmd': 'cat a'}, {'stdout': '1'}), ({'tool': 'bash', 'cmd': 'cat b'}, {'stdout': '2'})
    tripped_at = None
    for i in range(60):
        act, out = a if i % 2 == 0 else b
        if d.observe(act, out, turn=i + 1):
            tripped_at = i + 1
            break
    assert tripped_at == 59 and d.tripped['in_window'] == 30 and 'last 60' in d.tripped['rule']


class FakeShell:
    def __init__(self, clock):
        self.clock, self.calls = clock, []

    def bash(self, cmd, timeout):
        self.calls.append(cmd)
        self.clock.advance(2)
        return {'exit_code': 0, 'timed_out': False, 'seconds': 2.0 + 0.01 * len(self.calls), 'stdout': 'same\n', 'stderr': ''}

    def python(self, code, timeout):
        return self.bash(code, timeout)


def test_the_basic_agent_stops_with_degenerate_loop_and_the_model_sees_nothing(tmp_path):
    clock = VirtualClock()
    prompts = []

    def script(role, system, prompt):
        prompts.append(prompt)
        return {'tool': 'bash', 'cmd': 'cat finetune.yaml', 'timeout': 30, 'note': 'look again'}
    drv = ScriptedBackend(script, clock=clock, tokens_per_call=1000)
    task = H.task('A-NAT')
    agent = native.LoopStopAgent(driver=drv, executor=FakeShell(clock), task_text='TASK', env_tools=[],
                                 budget=task.search_budget(), run_id='anat_b', env='rebench_llmf', task_id=task.id,
                                 arm='A-NAT', out_dir=tmp_path / 'agent', clock=clock)
    trace = agent.run()
    assert trace['stop_reason'] == 'degenerate_loop' and trace['turns'] == 12 and len(prompts) == 12
    assert not any('loop' in p.lower() or 'degenerate' in p.lower() for p in prompts)
    assert agent.loop.tripped['turn'] == 12
    events = [json.loads(x) for x in (tmp_path / 'agent' / 'events.jsonl').read_text().splitlines()]
    det = [e for e in events if e['kind'] == 'loop_detector']
    assert len(det) == 1 and det[0]['payload']['tripped']['rule'] == '>= 12 in a row'
    assert [e['kind'] for e in events][-2:] == ['loop_detector', 'turn']        # logged, never shown to the model


# ---------------------------------------------------------------- driver flags
class FakeBackend:
    def __init__(self):
        self.temperature, self.extra_body, self.bodies = 0.0, {'chat_template_kwargs': {'thinking': False}}, []
        self.timeout_seconds, self.context_tokens = 300, 32768

    def describe(self):
        return {'kind': 'fake', 'temperature': self.temperature, 'extra_body': dict(self.extra_body)}

    def complete(self, *a, **k):
        self.bodies.append({**self.extra_body, 'temperature': self.temperature})
        return type('R', (), {'llm_tokens': 10, 'wall_seconds': 1.0, 'meta': {}})()


def test_temperature_and_per_call_request_seed(monkeypatch):
    import infrarsi.agent.arms as agent_arms
    fb = FakeBackend()
    monkeypatch.setattr(agent_arms, 'weak_driver', lambda structured=True, selfhost=False: ('selfhost', fb))
    drv, route, meter = arms.weak_driver(temperature=1.0, request_seed=42)
    for _ in range(3):
        drv.complete('s', 'p')
    assert [b['seed'] for b in fb.bodies] == [42, 43, 44] and all(b['temperature'] == 1.0 for b in fb.bodies)
    assert all(b['chat_template_kwargs'] == {'thinking': False} for b in fb.bodies)      # thinking stays off
    assert drv.describe()['request_seed'] == 42 and drv.describe()['request_seed_mode'] == 'base + call index'
    assert fb.timeout_seconds == 600 and fb.context_tokens == 98_304 and meter.calls == 3
    fb2 = FakeBackend()
    monkeypatch.setattr(agent_arms, 'weak_driver', lambda structured=True, selfhost=False: ('selfhost', fb2))
    drv2, _, _ = arms.weak_driver(temperature=1.0, request_seed=42, seed_per_call=False)
    drv2.complete('s', 'p'); drv2.complete('s', 'p')
    assert [b['seed'] for b in fb2.bodies] == [42, 42]
    fb3 = FakeBackend()
    monkeypatch.setattr(agent_arms, 'weak_driver', lambda structured=True, selfhost=False: ('selfhost', fb3))
    drv3, _, _ = arms.weak_driver()                                       # unchanged default: greedy, no seed
    drv3.complete('s', 'p')
    assert fb3.bodies[0]['temperature'] == 0.0 and 'seed' not in fb3.bodies[0] and not isinstance(drv3._inner, RequestSeed)


# ---------------------------------------------------------------- D-RB-16, the abandoned record, the wall rule, the CLI
def test_d_rb_16_on_every_receipt_of_the_restart():
    r, t = H.runner(H.node_result(H.task('A-NAT')), arm='A-NAT')
    r.deviations = tuple(r.deviations) + ('D-RB-16',)                     # what cmd_a_nat does for the restart
    rec = r.run(t, {}, None, 'screen', 'run')
    assert 'D-RB-16' in rec['deviations'] and 'D-RB-10' in rec['deviations'] and rec['valid']


def test_abandon_record_counts_the_loop(tmp_path):
    d = tmp_path / 'anat_a'
    d.mkdir()
    rows = [{'utc': 'u0', 'kind': 'llm_call', 'payload': {'llm_tokens': 100, 'wall_seconds': 2.0}}]
    rows += [{'utc': f'u{i}', 'kind': 'turn', 'payload': {'turn': i, 'tool': 'bash', 'action': {'tool': 'bash', 'cmd': 'ls', 'note': str(i)},
                                                           'result': {'stdout': 'x', 'seconds': i}}} for i in range(1, 15)]
    (d / 'events.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    rec = cli.abandon_record(d, restart_run_id='anat_b', reason='loop')
    assert rec['status'] == 'abandoned' and rec['scorings'] == 0 and rec['turns'] == 14 and rec['llm_tokens'] == 100
    assert rec['most_repeated_signature'] == {'count': 14, 'first_turn': 1, 'max_consecutive': 14}
    assert json.loads((d / 'ABANDONED.json').read_text())['restarted_as'] == 'anat_b'


def test_wall_rule_shortens_only_when_the_hard_stop_requires_it():
    a = type('A', (), {'hard_stop': '2026-10-22T04:00:00+08:00', 'wall_s': None})()
    early = cli._wall_rule(a, now=datetime.datetime(2026, 10, 21, 2, 50, tzinfo=datetime.timezone.utc))   # 10:50 SGT
    assert early['shortened'] is False and early['wall_seconds'] == 57_600
    late = cli._wall_rule(a, now=datetime.datetime(2026, 10, 21, 4, 0, tzinfo=datetime.timezone.utc))     # 12:00 SGT
    assert late['shortened'] and late['wall_seconds'] == pytest.approx((16 * 3600) - 55 * 60, abs=1)
    assert cli._wall_rule(type('A', (), {'hard_stop': None, 'wall_s': None})()) is None


def test_cli_dry_run_of_the_restart(capsys):
    rc = cli.main(['a-nat', '--node', 'WINH1', '--run-id', 'anat_b', '--dry-run', '--local', '--temperature', '1.0',
                   '--request-seed', '42', '--restart-of', 'anat_a', '--hard-stop', '2099-01-01T00:00:00+08:00'])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out['temperature'] == 1.0 and out['request_seed'] == 42 and out['restart_of'] == 'anat_a'
    assert 'D-RB-16' in out['deviations'] and out['loop_detector'] == {'consecutive': 12, 'window': 60, 'window_count': 30}
    assert out['driver']['temperature'] == 1.0 and out['driver']['structured'] == 'json_object'
    assert out['driver']['extra_body']['chat_template_kwargs'] == {'thinking': False} and out['context_tokens'] == 72_000
    assert out['budget']['screens'] == 60 and out['budget']['gpu_hours'] == 34.0 and out['budget']['wall_seconds'] == 57_600
    p = cli.build_parser().parse_args(['a-nat', '--run-id', 'x'])
    assert p.temperature is None and p.request_seed is None and p.restart_of is None      # defaults unchanged
