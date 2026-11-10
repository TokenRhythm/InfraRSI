import json
import math
from pathlib import Path

import pytest

from infrarsi.agent.budget import VirtualClock
from infrarsi.ext.common.env_judge import EnvJudge
from infrarsi.ext.common.remote_exec import LocalNodeOps
from infrarsi.ext.optico import guards as gb
from infrarsi.ext.optico import params as pm
from infrarsi.ext.optico import task as taskmod
from infrarsi.ext.optico import workloads as wl
from infrarsi.ext.optico.runner import OptiCoRunner
from infrarsi.harness import signing
from optico_helpers import R3_W3_MS, frozen_json, megatron_log, report_csv, report_row, write

W3 = wl.get('W3')


class FakeTrial:
    """Plays trial.sh: writes what run_monitor.sh and the script leave in the attempt directory."""

    def __init__(self, scenario='success', clock=None, seconds=144.0):
        self.scenario, self.clock, self.seconds, self.launched = scenario, clock, seconds, []

    def __call__(self, ops, script, log):
        a = Path(script).parent
        spec = dict(line.split('=', 1) for line in (a / 'spec.sh').read_text().splitlines())
        label = spec['ITER_LABEL'].strip("'")
        params = pm.typed(pm.decode_params((a / 'parameters.txt').read_text()))      # native sets may be outside the domain
        self.launched.append({'dir': a, 'params': params, 'token': spec['TOKEN'].strip("'"),
                              'spec': {k: v.strip("'") for k, v in spec.items()}})
        files = taskmod.frozen_files()
        write(a, 'frozen_before.json', frozen_json(files))
        if self.scenario == 'hang':
            return {'pid': 4242}
        if self.clock is not None:
            self.clock.advance(self.seconds)                               # the trial takes time on the node
        after = dict(files)
        if self.scenario == 'tree_changed':
            after['script/monitor/run_monitor.sh'] = '0' * 64
        write(a, 'frozen_after.json', frozen_json(after))
        write(a, 'copy_check.json', json.dumps({'checked': 38, 'mismatch': ['script/monitor/run_monitor.sh']
                                                if self.scenario == 'copy_mismatch' else []}))
        ok = self.scenario != 'fail'
        log_kw = {'precision': 'bf16'} if self.scenario == 'bf16' else {}
        n_iters = 4 if self.scenario == 'short' else (1 if self.scenario == 'fail' else 5)
        write(a, f'out/{label}/megatron_log/0.txt', megatron_log('W3', params, n_iters=n_iters, **log_kw))
        write(a, f'out/{label}/memory.txt', 'Temp Power Memory-Usage Memory-ALL Util% \n40C 300W 16727MiB 81559MiB 100%\n')
        tflops = {'short': (233.6 + 208.1) / 2, 'row_tflops_mismatch': 250.0}.get(self.scenario)
        row = report_row(params, result='success' if ok else 'fail', log_dir=label, tflops=tflops)
        write(a, 'out/report.csv', report_csv([row]))
        if not ok:
            write(a, f'out/{label}/monitor_error.txt', 'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate X UNIT\n'
                                                       '--- Tracebacks ---\nTraceback ...\n')
        write(a, 'status.json', json.dumps({'rc': 0, 'timed_out': False, 't_start': 100.0, 't_materialized': 112.5,
                                            't_monitor_end': 242.5, 't_end': 244.0, 'host': 'fake'}))
        return {'pid': 4242}


def _runner(tmp_path, scenario='success', clock=None, **kw):
    clock = clock or VirtualClock()
    fake = FakeTrial(scenario, clock=clock)
    ops = LocalNodeOps(tmp_path / 'node', node='WINH4', launcher=fake)
    r = OptiCoRunner(ops, 'W3', out_dir=tmp_path / 'pc', clock=clock, sleep=clock.advance, poll_s=10, **kw)
    return r, ops, fake


def test_success_trial_receipt(tmp_path):
    task = taskmod.make_task('W3', 'B-IS')
    runner, ops, fake = _runner(tmp_path)
    p = W3.knobs(MICRO_BATCH_SIZE=2, PP=2, OVERLAP_GRAD_REDUCE=True, OVERLAP_PARAM_GATHER=True, AC='sel', FUSED_CROSS_ENTROPY=True)
    r = runner.run(task, p, None, 'screen', 'run1', candidate_id='c1')
    assert signing.verify(r) and r['schema'] == 'ext-receipt-v1' and r['env'] == 'optico' and r['task_id'] == 'ENV-B-W3'
    assert r['valid'] and r['judge_valid'] and r['guard_rejections'] == [] and r['invalid_reasons'] == []
    want = sum(R3_W3_MS[2:]) / 3000.0
    assert math.isclose(r['metric']['value'], want) and r['step_time']['mean_window_max_rank_s'] == r['metric']['value']
    raw = r['metric']['raw']
    assert math.isclose(raw['tflops_per_gpu'], 228.833, abs_tol=1e-3) and math.isclose(raw['mfu_h100'], 228.833 / 989, abs_tol=1e-5)
    assert math.isclose(raw['mfu_optico'], 228.833 / 312, abs_tol=1e-4)
    assert r['params'] == p and r['params_applied'] == p and r['candidate_digest'] == pm.params_key(p)
    assert set(['D-OC-1', 'D-OC-3', 'D-OC-5', 'D-OC-6', 'D-OC-12', 'D-OC-13']) <= set(r['deviations'])
    assert r['memory']['max_peak_reserved_gib'] == round(12668 / 1024, 3) and r['loss']['iter1'] == 11.92976
    assert r['cost']['accounting_complete'] and r['cost']['gpu_seconds'] == 8 * 144.0 and r['phases']['monitor_s'] == 130.0
    assert r['summary']['params'].startswith('MBS2 TP1 PP2 AC sel')
    # the parameter file the monitor exported is OptiCo's own format; leftovers killed by token; bulky copy removed
    launched = fake.launched[0]
    assert launched['params'] == p and launched['token'] in ops.killed_tokens
    assert [c for c in ops.calls if c[0] == 'remove_tree']
    assert EnvJudge(r, None, None).valid                                    # X1's judge accepts it
    lease_files = list((tmp_path / 'node' / 'data' / 'infrarsi' / 'leases').rglob('*.json'))
    assert lease_files == []                                                # lease and GPU claims released
    assert (tmp_path / 'pc' / 'receipts' / f'{r["attempt_id"]}.json').exists()
    assert runner.history == [r]


def test_invalid_proposal_never_takes_the_node(tmp_path):
    task = taskmod.make_task('W3', 'B-IS')
    runner, ops, fake = _runner(tmp_path)
    r = runner.run(task, W3.knobs(TP=4), None, 'screen', 'run1', candidate_id='c2')
    assert not r['valid'] and r['not_launched'] == 'invalid_proposal' and signing.verify(r)
    assert r['cost'] == {'gpu_seconds': 0.0, 'wall_seconds': 0.0, 'accounting_complete': True,
                         'basis': 'rejected by validate_optico_params before any launch (D-OC-13)'}
    assert fake.launched == [] and not [c for c in ops.calls if c[0] in ('gpu_probe', 'acquire')]
    assert 'invalid_proposal' in r['invalid_reasons'][0] and not EnvJudge(r, None, None).valid


def test_judge_failure_keeps_guards_as_reports(tmp_path):
    task = taskmod.make_task('W3', 'B-IS')
    runner, ops, fake = _runner(tmp_path, 'fail')
    r = runner.run(task, W3.knobs(MICRO_BATCH_SIZE=8), None, 'screen', 'run1', candidate_id='c3')
    assert not r['valid'] and not r['judge_valid'] and r['guard_rejections'] == []
    assert r['invalid_reasons'][0].startswith('judge: OptiCo Result=fail')
    assert any(x.get('would_reject') for x in r['guard_reports'])          # G-B3 would reject: diagnosis only
    assert r['failure_log_head'][0].startswith('torch.OutOfMemoryError') and r['metric']['value'] is None


@pytest.mark.parametrize('scenario,gid', [('bf16', 'G-B1'), ('short', 'G-B3'), ('tree_changed', 'frozen_files'),
                                          ('copy_mismatch', 'frozen_files'), ('row_tflops_mismatch', 'output_integrity')])
def test_guards_veto_judge_valid_trials(tmp_path, scenario, gid):
    task = taskmod.make_task('W3', 'B-IS')
    runner, _, _ = _runner(tmp_path, scenario)
    r = runner.run(task, W3.defaults, None, 'screen', 'run1', candidate_id='c4')
    assert r['judge_valid'] and not r['valid'] and gid in [g['id'] for g in r['guard_rejections']]
    assert r['metric']['value'] is None and any(x.startswith(f'guard:{gid}') for x in r['invalid_reasons'])
    assert not EnvJudge(r, None, None).valid


def test_g_b2_hard_veto_through_the_runner(tmp_path):
    task = taskmod.make_task('W3', 'B-IS')
    runner, _, _ = _runner(tmp_path, gb2=gb.GB2Policy(reference_loss1=10.0, hard=True))
    r = runner.run(task, W3.defaults, None, 'screen', 'run1', candidate_id='c5')
    assert r['judge_valid'] and not r['valid'] and [g['id'] for g in r['guard_rejections']] == ['G-B2']
    runner2, _, _ = _runner(tmp_path / 'b', gb2=gb.GB2Policy(reference_loss1=10.0, hard=False))
    r2 = runner2.run(task, W3.defaults, None, 'screen', 'run1', candidate_id='c5')
    assert r2['valid'] and any(x['id'] == 'G-B2' and x['exceeded'] for x in r2['guard_reports'])


def test_hung_trial_is_killed_and_charged(tmp_path):
    task = taskmod.make_task('W3', 'B-IS')
    runner, ops, fake = _runner(tmp_path, 'hang', cap_s=900)
    r = runner.run(task, W3.defaults, None, 'screen', 'run1', candidate_id='c6')
    assert not r['valid'] and fake.launched[0]['token'] in ops.killed_tokens
    assert r['runner']['overdue'] and r['cost']['gpu_seconds'] >= 8 * 900       # held for the whole cap
    assert any('no report.csv row' in x for x in r['invalid_reasons'])


def test_occupied_node_is_refused(tmp_path):
    task = taskmod.make_task('W3', 'B-IS')
    fake = FakeTrial()
    busy = {'hostname': 'fake', 'utc': 'x', 'error': None, 'gpus': [{'index': g, 'mem_used_mib': 0} for g in range(8)],
            'apps': [{'pid': 99, 'gpu_index': 3, 'used_mib': 30000, 'cwd': '/data/tyc/job', 'cmdline': 'python train.py'}]}
    ops = LocalNodeOps(tmp_path / 'node', node='WINH4', launcher=fake, probe_fn=lambda: busy)
    r = OptiCoRunner(ops, 'W3', clock=VirtualClock()).run(task, W3.defaults, None, 'screen', 'run1', candidate_id='c7')
    assert not r['valid'] and r['invalid_reasons'][0].startswith('occupied') and fake.launched == []
    assert r['cost']['accounting_complete'] is False and r['cost']['gpu_seconds'] == 0.0


def test_transport_error_after_launch_still_kills_before_release(tmp_path):
    """An exception while polling must not hand the node back with our processes on it."""
    task = taskmod.make_task('W3', 'B-IS')
    clock = VirtualClock()
    fake = FakeTrial('hang', clock=clock)

    class Flaky(LocalNodeOps):
        order = []

        def call(self, func, _timeout=300, **kw):
            self.order.append(func)
            if func == 'read_json' and str(kw.get('path', '')).endswith('status.json'):
                raise RuntimeError('ssh transport dropped')
            return super().call(func, _timeout, **kw)
    ops = Flaky(tmp_path / 'node', node='WINH4', launcher=fake)
    r = OptiCoRunner(ops, 'W3', clock=clock, sleep=clock.advance).run(task, W3.defaults, None, 'screen', 'run1', candidate_id='c8')
    assert not r['valid'] and r['invalid_reasons'][0].startswith('runner_error')
    assert fake.launched[0]['token'] in ops.killed_tokens
    releases = [i for i, f in enumerate(ops.order) if f in ('release', 'release_status')]   # ext.common's lease release
    assert releases and ops.order.index('kill_token') < max(releases)


def test_metric_is_the_exact_window_and_extra_records_are_rejected():
    from infrarsi.ext.optico import trial
    from optico_helpers import iteration_line
    p = W3.defaults
    log = megatron_log('W3', p) + iteration_line(6, 6, 99999.0, 1.0, 9.0) + '\n'
    run = trial.evaluate('W3', candidate_id='c', row=report_row(p, tflops=(233.6 + 208.1 + 244.8 + 1.0) / 4), log_text=log,
                         requested=p, exit_code=0)
    assert run.metric_value == pytest.approx(sum(R3_W3_MS[2:]) / 3000.0)          # iterations 3..5 by number
    assert 'G-B3' in [g['id'] for g in run.guards.rejections]


def test_success_row_without_tflops_is_an_integrity_rejection():
    from infrarsi.ext.optico import trial
    p = W3.defaults
    row = {**report_row(p), 'TFLOPS': 'NULL'}
    run = trial.evaluate('W3', candidate_id='c', row=row, log_text=megatron_log('W3', p), requested=p, exit_code=0)
    assert run.judge_valid and 'output_integrity' in [g['id'] for g in run.guards.rejections]
