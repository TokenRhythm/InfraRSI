"""Fixes from the 2026-10-23 Q&A on the project flow (reports/infrarsi_qa_20261023):
A3 `--experience off` (or a pack selection) also keeps earlier runs' results out of the prompt;
A4 transport failures are retried, then end the run as `infrastructure_failure` (before: the reference run or
promotion crashed, H2b arm C);
A5 a run keeps a log file and survives a console that went away."""
import json
import time
import types
from pathlib import Path

import pytest

from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.contracts import InfrastructureError
from infrarsi.project import ProjectRunner
from infrarsi.project import executor as ex
from infrarsi.project import runner as runner_mod
from infrarsi.project import flow
from infrarsi.project.flow import ledger_path, run_project

from test_project import toy  # noqa: F401 - the toy project fixture


class Flaky(ex.LocalExecutor):
    """Fails to place the tree on the calls listed in `fail` (1-based), like a copy over a dropped ssh connection."""

    def __init__(self, fail):
        super().__init__()
        self.fail, self.places = set(fail), 0

    def place(self, tree, rel):
        self.places += 1
        if self.places in self.fail:
            raise InfrastructureError('copy to node0 failed: Connection reset by peer')
        return super().place(tree, rel)


@pytest.fixture
def no_wait(monkeypatch):
    monkeypatch.setattr(ProjectRunner, 'infra_retry_waits', (0.0, 0.0))


def test_runner_retries_a_transport_failure_then_raises(toy, tmp_path, no_wait):
    r = ProjectRunner(toy, out_dir=tmp_path / 'a', executor=Flaky({1, 2}))
    rec = r.run(toy.task(), toy.reference_config(), None, 'reference', 'r', candidate_id='c0')
    assert rec['valid'] and r.executor.places == 3
    r = ProjectRunner(toy, out_dir=tmp_path / 'b', executor=Flaky({1, 2, 3}))
    with pytest.raises(InfrastructureError):
        r.run(toy.task(), toy.reference_config(), None, 'reference', 'r', candidate_id='c0')


def test_unreachable_hosts_end_the_run_cleanly(toy, tmp_path, no_wait, monkeypatch):
    monkeypatch.setattr(runner_mod.ex, 'make', lambda execution: Flaky(range(1, 10 ** 6)))
    res = run_project(toy, strategy='random', budget={'screens': 2}, out_root=tmp_path / 'out', log=lambda m: None)
    assert res['outcome'] == 'infrastructure_failure' and 'Connection reset' in res['infrastructure']
    assert 'could not be reached' in (Path(res['run_dir']) / 'REPORT.md').read_text(encoding='utf-8')


def test_losing_the_hosts_mid_search_skips_promotion(toy, tmp_path, no_wait, monkeypatch):
    monkeypatch.setattr(runner_mod.ex, 'make', lambda execution: Flaky(range(2, 10 ** 6)))     # the reference runs
    before = ledger_path().read_text(encoding='utf-8') if ledger_path().is_file() else ''
    res = run_project(toy, strategy='random', budget={'screens': 3}, out_root=tmp_path / 'out', log=lambda m: None)
    assert res['outcome'] == 'infrastructure_failure' and res['search']['stop_reason'].startswith('infrastructure')
    assert res['promotion']['promoted'] is None and not res['promotion']['tried']
    assert res['search']['used']['screens'] == 0
    assert 'Stopped early' in (Path(res['run_dir']) / 'REPORT.md').read_text(encoding='utf-8')
    after = ledger_path().read_text(encoding='utf-8') if ledger_path().is_file() else ''
    assert 'invalid' not in after[len(before):]                  # nothing about the candidate reaches the ledger


def test_experience_off_keeps_earlier_runs_out_of_the_prompt(toy, tmp_path):
    ledger_path().parent.mkdir(parents=True, exist_ok=True)
    with ledger_path().open('a', encoding='utf-8') as f:
        f.write(json.dumps({'project': 'other', 'gpu_type': toy.raw['hardware'].get('gpu_type'), 'mechanism': 'earlier',
                            'change': {'workers': 4}, 'measured_gain_pct': 9.0, 'status': 'confirmed'}) + '\n')
    seen = {}
    for mode in ('auto', 'off'):
        drv = ScriptedBackend(lambda role, system, prompt: {'hypotheses': []})
        res = run_project(toy, strategy='hls', driver=drv, budget={'screens': 1}, out_root=tmp_path / mode,
                          log=lambda m: None, experience=mode)
        prompts = [json.loads(c['prompt']) for c in drv.calls if c['role'] == 'hypotheses']
        seen[mode] = [r['mechanism'] for p in prompts for r in p.get('earlier_runs') or []]
    assert 'earlier' in seen['auto'] and seen['off'] == []
    assert '[search] hls: ' in (Path(res['run_dir']) / 'run.log').read_text(encoding='utf-8')   # the engine's decisions


def test_run_log_and_a_dead_console(toy, tmp_path):
    def dead(msg):
        raise OSError(22, 'Invalid argument')

    res = run_project(toy, strategy='random', budget={'screens': 1}, out_root=tmp_path / 'out', log=dead)
    assert res['outcome'] in ('improved', 'no_change')
    text = (Path(res['run_dir']) / 'run.log').read_text(encoding='utf-8')
    assert '[infrarsi]' in text and 'reference -> ' in text


class CutOff(ex.LocalExecutor):
    """The first execution runs for a while, leaves probe rows of its own and loses the connection; the retry runs."""

    def __init__(self):
        super().__init__()
        self.probes = []

    def run(self, cmd, *, env, **kw):
        self.probes.append(env.get('INFRARSI_PROBE'))
        if len(self.probes) == 1:
            with open(env['INFRARSI_PROBE'], 'a', encoding='utf-8') as f:
                for step in (1, 2, 3):
                    f.write(json.dumps({'step': step, 't': 1000.0 + 50 * step, 'loss': 9.0}) + chr(10))
            time.sleep(0.2)
            raise InfrastructureError('ssh to node0 failed: Connection to node0 closed by remote host.')
        return super().run(cmd, env=env, **kw)


def test_a_retry_gets_its_own_probe_file_and_the_lost_time_is_charged(toy, tmp_path, no_wait):
    raw = dict(toy.raw, signals=dict(toy.raw['signals'], source='probe'))
    p = type(toy)(raw, toy.path)
    r = ProjectRunner(p, out_dir=tmp_path / 'r', executor=CutOff())
    rec = r.run(p.task(), p.reference_config(), None, 'reference', 'r', candidate_id='c0')
    first, second = r.executor.probes
    assert first != second and second.endswith('probe-retry1.jsonl')
    assert rec['valid'] and rec['evidence']['step_time'] < 10            # not mixed with the cut-off run's rows
    assert rec['cost']['accounting_complete'] is False
    assert rec['cost']['gpu_seconds'] >= p.world * 0.2


def test_promotion_keeps_the_cost_of_runs_before_the_hosts_went_away(toy, tmp_path):
    receipts = tmp_path / 'receipts'
    receipts.mkdir()
    (receipts / 'c1.json').write_text(json.dumps({'config': dict(toy.reference_config(), workers=4), 'overlay': {}}),
                                      encoding='utf-8')
    calls = []

    class Runner:
        def run(self, *a, **kw):
            calls.append(a[3])
            if len(calls) == 2:
                raise InfrastructureError('copy to node0 failed: Connection reset by peer')
            return {'cost': {'gpu_seconds': 3600.0}}

    verdict = types.SimpleNamespace(step_time=1.0, valid=True)
    trace = {'candidates': [{'id': 'c1', 'valid': True, 'step_time': 0.5}], 'extra': {}}
    promo = flow.promote(toy, toy.task(), Runner(), lambda *a, **kw: verdict, trace, {}, 'r', receipts)
    assert promo['promoted'] is None and 'infrastructure' in promo and promo['gpu_hours'] == 1.0


def test_saved_calls_hold_every_answer_and_a_failed_api_call(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
    from test_openai_compat import backend, OK, U, WEAK
    b, _ = backend([(200, 'not json', 'deepseek-flash', U), (200, OK, 'deepseek-flash', U)], out_root=tmp_path / 'calls')
    b.complete('sys', 'prompt', schema=WEAK, role='hypotheses')
    call = next((tmp_path / 'calls').iterdir())
    assert (call / 'answer_0.txt').read_text(encoding='utf-8') == 'not json' and 'propose_config' in (call / 'answer_1.txt').read_text(encoding='utf-8')

    from infrarsi.agent.backends import AnthropicBackend

    def boom(**kw):
        raise TimeoutError('read timed out')

    client = types.SimpleNamespace(beta=types.SimpleNamespace(messages=types.SimpleNamespace(create=boom)),
                                   with_options=lambda **k: client)
    AnthropicBackend(client=client, out_root=tmp_path / 'api').complete('sys', 'the prompt', role='hypotheses')
    saved = json.loads(next((tmp_path / 'api').iterdir()).joinpath('call.json').read_text(encoding='utf-8'))
    assert saved['prompt'] == 'the prompt' and 'read timed out' in saved['error']

