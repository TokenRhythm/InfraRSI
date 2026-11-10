import json
from pathlib import Path

import pytest

from infrarsi.agent.backends.base import LLMResult
from infrarsi.agent.budget import VirtualClock
from infrarsi.ext.common.env_judge import EnvJudge
from infrarsi.ext.common.remote_exec import LocalNodeOps
from infrarsi.ext.optico import native
from infrarsi.ext.optico import params as pm
from infrarsi.ext.optico import task as taskmod
from infrarsi.ext.optico import workloads as wl
from infrarsi.harness import signing
from optico_helpers import frozen_json, megatron_log, report_csv, report_row, write

W3 = wl.get('W3')
SEQ = [W3.defaults, W3.knobs(MICRO_BATCH_SIZE=2), W3.knobs(MICRO_BATCH_SIZE=2, FUSED_CROSS_ENTROPY=True),
       W3.knobs(MICRO_BATCH_SIZE=4, FUSED_CROSS_ENTROPY=True), W3.knobs(MICRO_BATCH_SIZE=4, OVERLAP_GRAD_REDUCE=True),
       W3.knobs(MICRO_BATCH_SIZE=8), W3.knobs(MICRO_BATCH_SIZE=8, AC='sel')]


class LoopSim:
    """Plays loop.sh + run_circle.sh on the fake node: one report row per driver poll (trial + analysis)."""

    def __init__(self, ops_ref, clock, n_rows, *, fail_retry_at=None, trial_s=150.0, copy_mismatch=(), bad_row_at=None):
        self.ops_ref, self.clock, self.n_rows, self.trial_s = ops_ref, clock, n_rows, trial_s
        self.copy_mismatch, self.bad_row_at = copy_mismatch, bad_row_at
        self.fail_retry_at = fail_retry_at
        self.dir = None
        self.rows, self.t_node = [], 1000.0
        self.token = None

    def launch(self, ops, script, log):
        self.dir = Path(script).parent
        spec = dict(line.split('=', 1) for line in (self.dir / 'spec.sh').read_text().splitlines())
        self.token = spec['TOKEN'].strip("'")
        self.out = self.dir / 'out' / 'output_qwen2.5_0.5B_n1'
        write(self.dir, 'frozen_before.json', frozen_json(taskmod.frozen_files()))
        write(self.dir, 'copy_check_pre.json', json.dumps({'checked': 38, 'mismatch': list(self.copy_mismatch)}))
        self.stamp(kind='watch_start')
        return {'pid': 777}

    def stamp(self, **kw):
        kw['t'] = self.t_node
        with open(self.dir / 'stamps.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(kw) + '\n')

    def killed(self):
        return self.token in self.ops_ref[0].killed_tokens

    def step(self, seconds):
        self.clock.advance(seconds)
        if self.dir is None or self.killed() or (self.dir / 'status.json').exists():
            return
        i = len(self.rows)
        if i >= self.n_rows:
            write(self.dir, 'frozen_after.json', frozen_json(taskmod.frozen_files()))
            write(self.dir, 'copy_check_post.json', json.dumps({'checked': 38, 'mismatch': []}))
            write(self.dir, 'status.json', json.dumps({'rc': 0, 't_start': 1.0, 't_materialized': 2.0, 't_loop_end': 3.0,
                                                       't_end': 4.0, 'host': 'fake'}))
            self.stamp(kind='watch_end', rows=i)
            return
        label = str(i + 1)
        retry = self.fail_retry_at is not None and i == self.fail_retry_at + 1
        p = self.rows[-1]['params'] if retry else SEQ[(i - (1 if self.fail_retry_at is not None and i > self.fail_retry_at else 0)) % len(SEQ)]
        failed = self.fail_retry_at == i
        self.stamp(kind='iter_dir', name=label, rows=i)
        self.t_node += self.trial_s
        d = self.out / label
        logged = W3.knobs(MICRO_BATCH_SIZE=8) if self.bad_row_at == i else p      # row and log of different runs
        write(d, 'megatron_log/0.txt', megatron_log('W3', logged, n_iters=1 if failed else 5))
        write(d, 'memory.txt', '40C 300W 16727MiB 81559MiB 100%\n')
        write(d, 'parameters.txt', pm.encode_params(p))
        reason = 'NCCL watchdog timeout. No parameter issue. RETRY (Confidence: 80)' if failed else 'NULL'
        self.rows.append({'params': p, 'row': report_row(p, result='fail' if failed else 'success', log_dir=label, reason=reason)})
        write(self.out, 'report.csv', report_csv([r['row'] for r in self.rows]))
        self.stamp(kind='rows', rows=i + 1)
        if failed:
            write(d, 'loc_error.txt', reason)
            write(d, 'analyzer_log_2026.10.21-10.00.00_loc_error.log', '[x] start to run loc_error\n[x] response is: \n' + reason)
        elif not retry or True:
            write(d, 'analyzer_log_2026.10.21-10.00.10_auto.log',
                  '[x] start to run give_sugg\n[x] response is: \n<think>..</think>The suggestions are: Change MBS\n'
                  '[x] start to run change_para\n[x] response is: \nThe modified parameters are:\nMICRO_BATCH_SIZE:2\n')
            write(d, 'analyze_result.txt', 'success\n')
        self.t_node += 30.0


class FakeTunnel:
    def __init__(self, die_after=None):
        self.starts, self.stopped, self.die_after, self.checks = 0, 0, die_after, 0
        self._alive = False

    def start(self):
        self.starts += 1
        self._alive = True
        return self

    def alive(self):
        self.checks += 1
        if self.die_after is not None and self.checks == self.die_after:
            self._alive = False
        return self._alive

    def restart(self):
        self.stop()
        return self.start()

    def stop(self):
        self.stopped += 1
        self._alive = False


class FakeShim:
    def __init__(self, exhaust_after_polls=None):
        self.exhausted, self.port, self.polls, self.limit = None, 54321, 0, exhaust_after_polls
        self.stopped = False

    def start(self):
        return self

    def stop(self):
        self.stopped = True

    def snapshot(self):
        self.polls += 1
        if self.limit is not None and self.polls >= self.limit:
            self.exhausted = 'llm_usd'
        return {'calls': 6, 'llm_tokens': 60000, 'llm_usd': 0.42, 'refused': 0, 'accounting_complete': True,
                'exhausted': self.exhausted}


def _driver(tmp_path, n_rows, *, cycles=3, max_trials=26, tunnel=None, shim=None, wall_cap_s=5400.0, **sim_kw):
    clock = VirtualClock()
    ref = []
    sim = LoopSim(ref, clock, n_rows, **sim_kw)
    ops = LocalNodeOps(tmp_path / 'node', node='WINH4', launcher=sim.launch)
    ref.append(ops)
    tunnel = tunnel or FakeTunnel()
    shim = shim or FakeShim()
    d = native.NativeDriver('W3', ops, run_id='bnat-cpu', cycles=cycles, max_trials=max_trials, wall_cap_s=wall_cap_s,
                            out_dir=tmp_path / 'pc', poll_s=10.0, health_s=30.0, clock=clock, sleep=sim.step,
                            tunnel_factory=lambda node, rp, lp, path: tunnel, shim_factory=lambda: shim,
                            port_finder=lambda ops: 18123, endpoint_check=lambda ops, port: True, log=lambda m: None)
    return d, ops, sim, tunnel, shim


def test_smoke_three_iterations(tmp_path):
    d, ops, sim, tunnel, shim = _driver(tmp_path, 3)
    res = d.run()
    assert res['status'] == 'done' and res['stop_reason'] == 'loop_finished' and res['n_rows'] == 3
    assert res['n_new'] == 2 and res['n_retries'] == 0 and res['n_success'] == 3 and res['n_valid'] == 3
    assert [c['analyze_result'] for c in res['iteration_checks']] == ['success'] * 3
    assert all(c['give_sugg_response'] and c['change_para_response'] for c in res['iteration_checks'])
    recs = res['receipt_objects']
    assert len(recs) == 3 and all(signing.verify(r) and EnvJudge(r, None, None).valid for r in recs)
    assert [r['native']['kind'] for r in recs] == ['default', 'new', 'new']
    assert recs[0]['params'] == W3.defaults and recs[1]['params']['MICRO_BATCH_SIZE'] == 2
    assert all(r['cost']['gpu_seconds'] == pytest.approx(8 * 150.0) and r['cost']['accounting_complete'] for r in recs)
    assert {'D-OC-4', 'D-OC-8', 'D-OC-11', 'D-OC-12', 'D-OC-14'} <= set(recs[0]['deviations'])
    assert 'D-OC-13' not in recs[0]['deviations']
    assert recs[0]['g_b2_policy']['reference_loss1'] == 11.92976 and recs[0]['g_b2_policy']['hard'] is False
    assert sim.token in ops.killed_tokens and tunnel.stopped and shim.stopped
    assert list((tmp_path / 'node' / 'data' / 'infrarsi' / 'leases').rglob('*.json')) == []
    out = tmp_path / 'pc'
    assert json.loads((out / 'result.json').read_text())['n_rows'] == 3 and len(list((out / 'receipts').glob('*.json'))) == 3
    assert (out / 'node' / 'iterations' / '2' / 'megatron_0.txt').exists()
    assert res['best_success_row']['tflops_per_gpu'] == pytest.approx(228.833, abs=1e-3)


def test_trial_cap_kills_the_loop(tmp_path):
    d, ops, sim, _, _ = _driver(tmp_path, 50, cycles=21, max_trials=4)
    res = d.run()
    assert res['stop_reason'] == 'max_trials' and res['n_rows'] == 4 and sim.token in ops.killed_tokens
    assert res['status'] == 'stopped'


def test_wall_and_llm_caps(tmp_path):
    d, ops, sim, _, _ = _driver(tmp_path, 50, cycles=21, wall_cap_s=35.0)
    res = d.run()
    assert res['stop_reason'] == 'max_wall' and sim.token in ops.killed_tokens
    d, ops, sim, _, shim = _driver(tmp_path / 'b', 50, cycles=21, shim=FakeShim(exhaust_after_polls=2))
    res = d.run()
    assert res['stop_reason'] == 'llm_budget:llm_usd'


def test_tunnel_restart_when_the_process_dies(tmp_path):
    tunnel = FakeTunnel(die_after=2)
    d, _, _, tunnel, _ = _driver(tmp_path, 3, tunnel=tunnel)
    res = d.run()
    assert res['stop_reason'] == 'loop_finished' and tunnel.starts == 2
    assert any(e['kind'] == 'tunnel_restart' for e in d.events)


def test_retry_rows_are_classified(tmp_path):
    d, _, _, _, _ = _driver(tmp_path, 4, fail_retry_at=1)
    res = d.run()
    kinds = [r['native']['kind'] for r in res['receipt_objects']]
    assert kinds == ['default', 'new', 'retry', 'new'] and res['n_retries'] == 1 and res['n_new'] == 2
    failed = res['receipt_objects'][1]
    assert not failed['judge_valid'] and failed['guard_rejections'] == [] and 'RETRY' in failed['invalid_reasons'][0]
    assert res['receipt_objects'][2]['native']['retry_of'] == 1


def test_occupied_node_is_never_used(tmp_path):
    busy = {'hostname': 'fake', 'utc': 'x', 'error': None, 'gpus': [{'index': g, 'mem_used_mib': 0} for g in range(8)],
            'apps': [{'pid': 5, 'gpu_index': 0, 'used_mib': 9000, 'cwd': '/home/other', 'cmdline': 'python x.py'}]}
    ops = LocalNodeOps(tmp_path / 'node', node='WINH4', launcher=lambda *a: pytest.fail('launched on a busy node'),
                       probe_fn=lambda: busy)
    d = native.NativeDriver('W3', ops, run_id='x', out_dir=tmp_path / 'pc', clock=VirtualClock(), sleep=lambda s: None,
                            tunnel_factory=lambda *a: pytest.fail('tunnel on a busy node'),
                            shim_factory=lambda: pytest.fail('shim on a busy node'), log=lambda m: None)
    res = d.run()
    assert res['status'] == 'occupied' and res['n_rows'] == 0


def test_budgeted_shim_caps_calls():
    class Backend:
        model, call_overhead_tokens = 'claude-opus-5-5', 100
        calls = []

        def complete(self, system, prompt, **kw):
            self.calls.append({**kw, 'prompt': prompt})
            return LLMResult(action=None, text='The modified parameters are:\nMICRO_BATCH_SIZE:2', llm_tokens=5000,
                             llm_usd=min(1.25, kw['max_usd']), wall_seconds=0.1, accounting_complete=True, error=None,
                             model=self.model, observed_models=[self.model], meta={'token_breakdown': {'output_tokens': 50}})

    import tempfile
    tmp = Path(tempfile.mkdtemp())
    b = Backend()
    shim = native.BudgetedShim(b, tokens_cap=1_000_000, usd_cap=3.0, log_path=tmp / 'calls.jsonl', calls_dir=tmp / 'calls')
    body = {'model': 'gpt-5.2', 'messages': [{'role': 'user', 'content': 'hi'}], 'temperature': 0.7, 'max_tokens': 8192}
    codes = [shim.server.handle(body)[0] for _ in range(4)]
    assert codes == [200, 200, 200, 429] and shim.exhausted == 'llm_usd' and shim.used['calls'] == 3
    assert shim.used['llm_usd'] <= 3.0 and b.calls[2]['max_usd'] == pytest.approx(0.5)   # never above the cap
    assert b.calls[0]['max_output_tokens'] == 8192                     # OptiCo's own max_tokens kept as the cap
    # the worst case of the next call is reserved before it runs
    near = native.BudgetedShim(b, tokens_cap=1_000_000, usd_cap=3.0, log_path=tmp / 'c3.jsonl', calls_dir=tmp / 'c3')
    near.used['llm_usd'] = 3.0 - near.worst_usd('hi', 8192) + 1e-6
    assert near.server.handle(body)[0] == 429 and near.exhausted == 'llm_usd'
    row = json.loads((tmp / 'calls.jsonl').read_text().splitlines()[0])
    assert row['requested_model'] == 'gpt-5.2' and row['ignored_params']['temperature'] == 0.7
    small = native.BudgetedShim(b, tokens_cap=5000, usd_cap=30.0, log_path=tmp / 'c2.jsonl', calls_dir=tmp / 'c2')
    assert small.server.handle(body)[0] == 200 and small.server.handle(body)[0] == 429 and small.exhausted == 'llm_tokens'


def test_classify_rows_retry_limit():
    fail = lambda p, i: report_row(p, result='fail', log_dir=str(i), reason='hang. No parameter issue. RETRY (Confidence: 80)')  # noqa: E731
    p = W3.defaults
    rows = [report_row(p, log_dir='1'), fail(SEQ[1], 2), fail(SEQ[1], 3), fail(SEQ[1], 4), fail(SEQ[1], 5)]
    kinds = [k['kind'] for k in native.classify_rows(rows)]
    assert kinds == ['default', 'new', 'retry', 'retry', 'new']          # at most 2 retries in a row (run_circle.sh)


def test_protocol_maxima_are_enforced(tmp_path):
    ops = LocalNodeOps(tmp_path / 'node', node='WINH4')
    for kw, needle in [({'cycles': 22}, 'cycles 22 exceeds'), ({'max_trials': 27}, 'max_trials 27'),
                       ({'wall_cap_s': 5401.0}, 'wall_cap_s 5401.0'), ({'llm_usd_cap': 31.0}, 'llm_usd_cap 31.0'),
                       ({'llm_tokens_cap': 1_500_001}, 'llm_tokens_cap'), ({'wall_cap_s': 0}, 'positive'),
                       ({'prompt_transform': 'rewrite_all'}, 'unknown prompt transform')]:
        with pytest.raises(ValueError, match=needle):
            native.NativeDriver('W3', ops, run_id='x', out_dir=tmp_path / 'pc', **kw)
    d = native.NativeDriver('W3', ops, run_id='x', out_dir=tmp_path / 'pc', cycles=3, llm_usd_cap=3.0, wall_cap_s=2700)
    assert d.wall_cap_s == 2700 and d.max_trials == 26 and {'D-OC-15', 'D-OC-16'} <= set(d.deviations)
    raw = native.NativeDriver('W3', ops, run_id='x', out_dir=tmp_path / 'pc2', cycles=3, prompt_transform=None)
    assert 'D-OC-15' not in raw.deviations and 'D-OC-16' in raw.deviations


def test_row_log_mismatch_and_copy_mismatch_are_rejected(tmp_path):
    d, _, _, _, _ = _driver(tmp_path, 3, bad_row_at=1)
    recs = d.run()['receipt_objects']
    bad = recs[1]
    assert bad['judge_valid'] and not bad['valid'] and 'G-B1' in {g['id'] for g in bad['guard_rejections']}
    assert 'requested knobs' in bad['guard_rejections'][0]['detail']
    assert recs[0]['valid'] and recs[2]['valid']
    d, _, _, _, _ = _driver(tmp_path / 'b', 2, copy_mismatch=('script/run_circle.sh',))
    recs = d.run()['receipt_objects']
    assert recs and all(not r['valid'] and 'frozen_files' in [g['id'] for g in r['guard_rejections']] for r in recs)


def test_prompt_transform_is_opt_in_and_recorded(tmp_path):
    from infrarsi.ext.optico import prompt_variants as pv
    text = ('Analyze.\nYou MUST put reasoning inside `<think>` tags.\n`<think>`\n[Your step-by-step reasoning goes here]\n'
            '`</think>`\nThe suggestions are:\n')
    new, n = pv.strip_think_format(text)
    assert n == 4 and new == 'Analyze.\nThe suggestions are:\n'
    keep = 'Strictly provide: The modified parameters are:\n'
    assert pv.strip_think_format(keep) == (keep, 0)

    class Backend:
        model, call_overhead_tokens, prompts = 'claude-opus-5-5', 100, []

        def complete(self, system, prompt, **kw):
            self.prompts.append(prompt)
            return LLMResult(action=None, text='The suggestions are: x', llm_tokens=10, llm_usd=0.01, wall_seconds=0.1,
                             accounting_complete=True, error=None, model=self.model, observed_models=[self.model], meta={})
    b = Backend()
    shim = native.BudgetedShim(b, tokens_cap=10_000, usd_cap=3.0, log_path=tmp_path / 'l.jsonl', calls_dir=tmp_path / 'c',
                               transform='strip_think_format')
    assert shim.server.handle({'messages': [{'role': 'user', 'content': text}]})[0] == 200
    assert b.prompts == ['Analyze.\nThe suggestions are:\n'] and shim.transform_log[0]['lines_dropped'] == 4
    plain = native.BudgetedShim(b, tokens_cap=10_000, usd_cap=3.0, log_path=tmp_path / 'l2.jsonl', calls_dir=tmp_path / 'c2')
    plain.server.handle({'messages': [{'role': 'user', 'content': text}]})
    assert b.prompts[-1] == text and plain.snapshot()['transform'] is None
    d = native.NativeDriver('W3', LocalNodeOps(tmp_path / 'node'), run_id='x', out_dir=tmp_path / 'pc', cycles=3,
                            prompt_transform='strip_think_format')
    assert 'D-OC-15' in d.deviations and 'D-OC-15' in d.runner.deviations
