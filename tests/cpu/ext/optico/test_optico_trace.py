"""H-04 (Codex W9 R1): B-NAT emits a SearchTrace that ext.common.trajectory reads -- one screen per report row
(default, new, RETRY, failures, sanitizer rejections) with cumulative screens / wall / exact shim tokens+USD / GPU-hours,
a final `used` and the tail; the published-protocol best success row stays separate."""
import json
import math
from pathlib import Path

import pytest

from infrarsi.agent.budget import VirtualClock
from infrarsi.ext.common import trajectory
from infrarsi.ext.common.remote_exec import LocalNodeOps
from infrarsi.ext.optico import arms
from infrarsi.ext.optico import native
from infrarsi.ext.optico import task as taskmod
from infrarsi.ext.optico import workloads as wl
from optico_helpers import frozen_json, write
from test_optico_native import FakeTunnel, LoopSim

W3 = wl.get('W3')
OFFSET = 2.0                                                # node clock = PC clock + 2 s


def _rec(k, kind, *, valid, step=None, tflops=None, start=None, end=None, retry_of=None, sanitizer=False):
    return {'candidate_id': f'it{k + 1}', 'attempt_id': f'bnat-x-r{k:02d}', 'valid': valid, 'judge_valid': valid,
            'metric': {'value': step if valid else None, 'raw': {'tflops_per_gpu': tflops}},
            'invalid_reasons': [] if valid else ['judge: Result fail'], 'judge': {'result': 'success' if valid else 'fail'},
            'cost': {'gpu_seconds': 8.0 * (end - start) if end else 0.0, 'wall_seconds': (end - start) if end else None},
            'native': {'kind': kind, 'retry_of': retry_of, 'iteration_label': str(k + 1),
                       'node_window': [start + OFFSET if start else None, end + OFFSET if end else None],
                       'sanitizer_events': [{'key': 'PP', 'reason': 'unsafe characters'}] if sanitizer else []},
            'params': dict(W3.defaults), 'guard_rejections': []}


# PC times. Lease 980, launch 1000; the analyzer's calls sit between a row and the next trial (run_circle.sh order).
RECEIPTS = [_rec(0, 'default', valid=True, step=9.6, tflops=190.0, start=1030, end=1160),
            _rec(1, 'new', valid=False, start=1195, end=1330),                                   # OOM, loc_error: RETRY
            _rec(2, 'retry', valid=True, step=9.0, tflops=202.7, start=1360, end=1490, retry_of=1),
            _rec(3, 'new', valid=False, start=1515, end=1516, sanitizer=True),                  # D-OC-16 rejection
            _rec(4, 'new', valid=True, step=8.0, tflops=228.0, start=1555, end=1690)]
CALLS = [(1180, 1000, 0.01), (1190, 500, 0.005), (1350, 3000, 0.03), (1500, 1000, 0.01), (1508, 500, 0.005),
         (1530, 3000, 0.03), (1540, 1000, 0.01), (1550, 500, 0.005), (1700, 1000, 0.01), (1710, 500, 0.005)]
BEST = {'attempt_id': 'bnat-x-r04', 'tflops_per_gpu': 228.0, 'valid': True, 'iter_time_s': 8.0}


def _trace(**kw):
    task = taskmod.make_task('W3', 'B-NAT')
    calls = [{'utc': float(t), 'call': i + 1, 'llm_tokens': tok, 'llm_usd': usd, 'accounting_complete': True}
             for i, (t, tok, usd) in enumerate(CALLS)]
    args = dict(receipts=RECEIPTS, calls=calls, task=task, run_id='bnat-w3', t_lease_pc=980.0, t_launch_pc=1000.0,
                t_end_pc=1760.0, held_s=820.0, offset_s=OFFSET, offset_info={'uncertainty_s': 0.5, 'measured': True},
                shim_used={'llm_tokens': 12000, 'llm_usd': 0.12, 'accounting_complete': True}, best=BEST,
                selected_config=dict(W3.defaults))
    args.update(kw)
    return task, native.native_trace(**args)


def tr_ok():
    return _trace()[1]


def test_trace_rows_cumulative_cost_and_tail():
    task, tr = _trace()
    assert tr['schema'] == 'infrarsi-search-trace-v1' and tr['arm'] == 'B-NAT' and tr['mode'] == 'native'
    s = tr['screens']
    assert [x['kind'] for x in s] == ['default', 'new', 'retry', 'new', 'new'] and s[3]['sanitizer_rejected']
    assert [x['cum']['screens'] for x in s] == [1, 2, 3, 4, 5]
    assert [x['cum']['llm_tokens'] for x in s] == [0, 1500, 4500, 6000, 10500]
    assert [x['cum']['llm_usd'] for x in s] == pytest.approx([0.0, 0.015, 0.045, 0.06, 0.105])
    assert [x['cum']['wall_seconds'] for x in s] == [160.0, 330.0, 490.0, 516.0, 690.0]         # since the launch
    assert [x['cum']['gpu_hours'] for x in s] == pytest.approx([8 * t / 3600 for t in (180, 350, 510, 536, 710)])
    assert [x['valid'] for x in s] == [True, False, True, False, True] and s[1]['step_time'] is None
    assert s[2]['gpu_hours'] == pytest.approx(8 * 130 / 3600) and s[2]['wall_seconds'] == 130
    assert {x['time_source'] for x in s} == {'node_stamp'}
    u = tr['used']
    assert u == {'screens': 5, 'llm_tokens': 12000, 'llm_usd': 0.12, 'gpu_hours': pytest.approx(8 * 820 / 3600),
                 'wall_seconds': 760.0, 'accounting_complete': True}
    tail = tr['extra']['tail']
    assert tail['llm_tokens'] == 1500 and tail['llm_usd'] == pytest.approx(0.015) and tail['wall_seconds'] == 70.0
    assert tail['gpu_hours'] == pytest.approx(8 * 110 / 3600, abs=1e-6)                      # costs kept to 6 decimals
    assert [c['after_screen'] for c in tr['llm_calls']] == [0, 0, 1, 2, 2, 3, 3, 3, 4, 4]
    assert tr['extra']['calls_near_row_boundary'] == 0 and tr['extra']['llm_log']['matches_shim']
    # the published-protocol result stays separate; selected = that row (the configuration B-NAT locks)
    assert tr['extra']['published_best_success_row'] == BEST and tr['selected'] == 'it5' and tr['selected_step_time'] == 8.0
    assert tr['selected_config'] == W3.defaults and tr['start'] == 'it1' and tr['start_step_time'] == 9.6


def test_arm_row_screen_count_auc_inputs_and_ctg():
    task, tr = _trace()
    ref = 9.6                                                # the default's iteration time (y = ln(ref / t))
    evs = trajectory.events(tr, task, ref)
    assert [e.kind for e in evs] == ['screen'] * 5 + ['tail']
    assert [e.tokens for e in evs] == [0, 1500, 3000, 1500, 4500, 1500]
    assert [e.wall_s for e in evs] == pytest.approx([160, 170, 160, 26, 174, 70])
    assert [e.usd for e in evs] == pytest.approx([0.0, 0.015, 0.03, 0.015, 0.045, 0.015])
    assert sum(e.gpuh for e in evs) == pytest.approx(8 * 820 / 3600)
    row = trajectory.arm_row(tr, task, ref)
    assert row['n_screens'] == 5 and row['n_valid'] == 3 and row['arm'] == 'B-NAT'
    assert row['best_search'] == 8.0 and row['best_search_candidate'] == 'it5'
    assert row['y_best_search'] == pytest.approx(math.log(9.6 / 8.0))
    # CTG(tau = ln 1.05): reached at the RETRY row (ln(9.6/9.0) = 0.0645), with the cost up to and including it
    c = row['CTG']
    assert c['reached'] and c['gpuh'] == pytest.approx(8 * 510 / 3600) and c['usd'] == pytest.approx(0.045)
    assert c['wall_s'] == pytest.approx(490.0)
    # AUC_budget: x = max(screens / 21, tokens / 1.5M, wall / 5400); step function held to x = 1
    b = task.budget
    assert (b['screens'], b['llm_tokens'], b['wall_seconds']) == (21, 1_500_000, 5400.0)
    xs = [max(k / 21, s['cum']['llm_tokens'] / 1.5e6, s['cum']['wall_seconds'] / 5400)
          for k, s in enumerate(tr['screens'], start=1)]
    ys = [0.0, 0.0, math.log(9.6 / 9.0), math.log(9.6 / 9.0), math.log(9.6 / 8.0)]
    area, x_prev, y = 0.0, 0.0, 0.0
    for x, y_after in zip(xs, ys):
        area += y * (x - x_prev)
        x_prev, y = x, y_after
    area += y * (1.0 - x_prev)
    assert row['AUC_budget'] == pytest.approx(area) and row['AUC_final_y'] == pytest.approx(math.log(9.6 / 8.0))
    # design §0.3's y on TFLOPS: values from the trace, reference = the default's TFLOPS
    row2 = trajectory.arm_row(tr, task, ref, values=native.trace_values(tr), ref_raw={'tflops': 190.0})
    assert row2['AUC_final_y'] == pytest.approx(math.log(228.0 / 190.0)) and row2['CTG']['wall_s'] == pytest.approx(490.0)


def test_row_time_fallbacks_and_clock_offset():
    assert tr_ok()['used']['accounting_complete'] and all(x['cum']['accounting_complete'] for x in tr_ok()['screens'])
    recs = [dict(r) for r in RECEIPTS[:3]]
    recs[1] = {**recs[1], 'native': {**recs[1]['native'], 'node_window': [None, None]}}
    _, tr = _trace(receipts=recs, row_seen_pc={1: 1335.0})
    s = tr['screens']
    assert [x['time_source'] for x in s] == ['node_stamp', 'driver_poll', 'node_stamp']
    assert s[1]['cum']['wall_seconds'] == 335.0 and s[1]['cum']['llm_tokens'] == 1500
    # a poll time is only an upper bound: from that row on the accounting is incomplete, and the trace says why
    assert [x['cum']['accounting_complete'] for x in s] == [True, False, False] and not tr['used']['accounting_complete']
    assert any('not the node watcher stamp' in n for n in tr['extra']['accounting_notes'])
    # with the trial-start stamp the LLM attribution stays exact (no analysis call while a trial runs) ...
    recs[1] = {**RECEIPTS[1], 'native': {**RECEIPTS[1]['native'], 'node_window': [1195 + OFFSET, None]}}
    _, tr = _trace(receipts=recs, row_seen_pc={1: 1351.0})            # the poll came after the loc_error call (1350)
    s = tr['screens']
    assert s[1]['attribution_source'] == 'trial_start_stamp' and s[1]['cum']['llm_tokens'] == 1500
    assert s[2]['cum']['llm_tokens'] == 4500 and s[1]['cum']['wall_seconds'] == 351.0     # ... the wall is the bound
    assert not s[1]['cum']['accounting_complete']
    # an unmeasured offset of 0 on a node 2 s ahead moves rows 2 s later: attribution happens to be unchanged here,
    # but node stamps compared with PC times under an unknown offset are not exact
    _, tr0 = _trace(offset_s=0.0, offset_info={'measured': False})
    assert [x['cum']['llm_tokens'] for x in tr0['screens']] == [0, 1500, 4500, 6000, 10500]
    assert not tr0['used']['accounting_complete'] and not tr0['screens'][0]['cum']['accounting_complete']
    assert any('offset was not measured' in n for n in tr0['extra']['accounting_notes'])
    # a call ending within 5 s of a row time cannot be attributed with certainty: incomplete from that row on
    near = [{'utc': 1163.0, 'call': 99, 'llm_tokens': 10, 'llm_usd': 0.0, 'accounting_complete': True}]
    calls = [{'utc': float(t), 'call': i + 1, 'llm_tokens': tok, 'llm_usd': usd, 'accounting_complete': True}
             for i, (t, tok, usd) in enumerate(CALLS)]
    _, tr2 = _trace(calls=calls + near, shim_used={'llm_tokens': 12010, 'llm_usd': 0.12, 'accounting_complete': True})
    assert [x['cum']['accounting_complete'] for x in tr2['screens']] == [False] * 5
    assert tr2['extra']['calls_near_row_boundary'] == 1 and not tr2['used']['accounting_complete']


class LoggingShim:
    """FakeShim with a call log: one call per snapshot (the driver takes one per new row and one at the end), ending 5 s
    before the simulated node's current time, i.e. after the newest row as run_circle.sh's analysis would."""

    def __init__(self, sim, out_dir, offset):
        self.sim, self.out, self.offset = sim, Path(out_dir), offset
        self.exhausted, self.port, self.n = None, 54321, 0

    def start(self):
        return self

    def stop(self):
        pass

    def snapshot(self):
        self.n += 1
        row = {'utc': self.sim.t_node - 5.0 - self.offset, 'call': self.n, 'llm_tokens': 1000, 'llm_usd': 0.01,
               'accounting_complete': True}
        self.out.mkdir(parents=True, exist_ok=True)
        with (self.out / 'shim_calls.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
        return {'calls': self.n, 'llm_tokens': 1000 * self.n, 'llm_usd': round(0.01 * self.n, 6), 'refused': 0,
                'accounting_complete': True, 'exhausted': None}


def test_driver_writes_the_trace_next_to_the_published_result(tmp_path):
    clock, ref = VirtualClock(), []
    sim = LoopSim(ref, clock, 3)
    ops = LocalNodeOps(tmp_path / 'node', node='WINH4', launcher=sim.launch)
    ref.append(ops)
    shim = LoggingShim(sim, tmp_path / 'pc', offset=-4000.0)        # node clock = PC clock - 4000 s
    d = native.NativeDriver('W3', ops, run_id='bnat-cpu', cycles=3, out_dir=tmp_path / 'pc', poll_s=10.0, clock=clock,
                            sleep=sim.step, tunnel_factory=lambda node, rp, lp, path: FakeTunnel(), shim_factory=lambda: shim,
                            port_finder=lambda ops: 18123, endpoint_check=lambda ops, port: True, log=lambda m: None,
                            wallclock=lambda: 5000.0 + clock(), clock_probe=lambda ops: 1000.0 + clock())
    res = d.run()
    out = tmp_path / 'pc'
    tr = json.loads((out / 'trace.json').read_text(encoding='utf-8'))
    assert res['trace'] == str(out / 'trace.json') and tr['extra']['clock']['offset_node_minus_pc_s'] == -4000.0
    assert len(tr['screens']) == res['n_rows'] == 3 and [s['kind'] for s in tr['screens']] == ['default', 'new', 'new']
    assert [s['cum']['wall_seconds'] for s in tr['screens']] == [150.0, 330.0, 510.0]
    assert [s['cum']['llm_tokens'] for s in tr['screens']] == [1000, 2000, 3000]          # calls before each row
    assert tr['used']['llm_tokens'] == 5000 and tr['extra']['tail']['llm_tokens'] == 2000   # last row's analysis
    assert tr['extra']['llm_log']['matches_shim'] and tr['extra']['calls_near_row_boundary'] == 0
    row = trajectory.arm_row(tr, d.task, tr['screens'][0]['step_time'])
    assert row['n_screens'] == 3 and row['used'] == tr['used']
    # the published-protocol best success row is unchanged in result.json, and both files select the same configuration
    result = json.loads((out / 'result.json').read_text(encoding='utf-8'))
    assert result['best_success_row'] == tr['extra']['published_best_success_row']
    assert arms.selection_from(out / 'trace.json') == arms.selection_from(out / 'result.json')


class TimedOps(LocalNodeOps):
    """Lease acquisition, workspace cleanup and lease release take time on the test clock."""
    delays = {'acquire': 2.0, 'remove_tree': 5.0, 'release_status': 1.0}

    def call(self, func, _timeout=300, **kw):
        self.clock.advance(self.delays.get(func, 0.0))
        return super().call(func, _timeout, **kw)


def test_lease_interval_and_arm_end_bound_the_trace(tmp_path):
    clock, ref = VirtualClock(), []
    sim = LoopSim(ref, clock, 3)                       # poll_s 180 = trial 150 + analysis 30: node time keeps pace
    ops = TimedOps(tmp_path / 'node', node='WINH4', launcher=sim.launch)
    ops.clock = clock
    ref.append(ops)
    d = native.NativeDriver('W3', ops, run_id='bnat-t', cycles=3, out_dir=tmp_path / 'pc', poll_s=180.0, health_s=1e9,
                            clock=clock, sleep=sim.step, tunnel_factory=lambda node, rp, lp, path: FakeTunnel(),
                            shim_factory=lambda: LoggingShim(sim, tmp_path / 'pc', offset=-4018.0),
                            port_finder=lambda ops: 18123, endpoint_check=lambda ops, port: True, log=lambda m: None,
                            wallclock=lambda: 5000.0 + clock(), clock_probe=lambda ops: 982.0 + clock())
    res = d.run()
    tr = d.trace
    c = tr['extra']['clock']
    assert c['t_lease_pc'] == 5018.0 and c['t_launch_pc'] == 5018.0           # lease held once 9 acquire calls (18 s) returned
    assert [s['cum']['wall_seconds'] for s in tr['screens']] == [150.0, 330.0, 510.0]
    assert [s['cum']['gpu_hours'] for s in tr['screens']] == pytest.approx([8 * t / 3600 for t in (150, 330, 510)], abs=1e-6)
    # the arm ends after collection, cleanup (3 x 5 s) and the release (9 x 1 s): 720 s loop + 15 + 9
    assert c['t_loop_end_pc'] == 5738.0 and c['t_end_pc'] == 5762.0
    assert res['held_s'] == 744.0 and tr['used']['wall_seconds'] == 744.0
    assert tr['used']['gpu_hours'] == pytest.approx(8 * 744 / 3600, abs=1e-6)
    assert tr['extra']['tail']['wall_seconds'] == 234.0 and tr['used']['accounting_complete']


class NoKillOps(LocalNodeOps):
    def call(self, func, _timeout=300, **kw):
        if func == 'kill_token':
            raise TimeoutError('kill_token: transport timeout')
        return super().call(func, _timeout, **kw)


def test_an_unconfirmed_kill_keeps_the_node_leased(tmp_path):
    clock, ref = VirtualClock(), []
    sim = LoopSim(ref, clock, 3)
    ops = NoKillOps(tmp_path / 'node', node='WINH4', launcher=sim.launch)
    ref.append(ops)
    d = native.NativeDriver('W3', ops, run_id='bnat-k', cycles=3, out_dir=tmp_path / 'pc', poll_s=10.0, clock=clock,
                            sleep=sim.step, tunnel_factory=lambda node, rp, lp, path: FakeTunnel(),
                            shim_factory=lambda: LoggingShim(sim, tmp_path / 'pc', offset=-4000.0),
                            port_finder=lambda ops: 18123, endpoint_check=lambda ops, port: True, log=lambda m: None,
                            wallclock=lambda: 5000.0 + clock(), clock_probe=lambda ops: 1000.0 + clock())
    res = d.run()
    assert d.kill_unconfirmed and res['lease_kept'] and any(e['kind'] == 'lease_kept' for e in d.events)
    assert list((tmp_path / 'node' / 'data' / 'infrarsi' / 'leases').rglob('*.json'))
    assert not d.trace['used']['accounting_complete']
    assert any('lease was not released' in n for n in d.trace['extra']['accounting_notes'])


class EnvOps(LocalNodeOps):
    envs = []

    def call(self, func, _timeout=300, **kw):
        if func == 'launch':
            self.envs.append(dict(kw.get('env') or {}))
        return super().call(func, _timeout, **kw)


def test_loop_carries_the_token_and_a_failing_logger_cannot_strand_the_lease(tmp_path):
    def boom(msg):
        raise OSError('console gone')
    clock, ref = VirtualClock(), []
    sim = LoopSim(ref, clock, 3)
    EnvOps.envs = []
    ops = EnvOps(tmp_path / 'node', node='WINH4', launcher=sim.launch)
    ref.append(ops)
    d = native.NativeDriver('W3', ops, run_id='bnat-e', cycles=3, out_dir=tmp_path / 'pc', poll_s=10.0, clock=clock,
                            sleep=sim.step, tunnel_factory=lambda node, rp, lp, path: FakeTunnel(),
                            shim_factory=lambda: LoggingShim(sim, tmp_path / 'pc', offset=-4000.0),
                            port_finder=lambda ops: 18123, endpoint_check=lambda ops, port: True, log=boom,
                            wallclock=lambda: 5000.0 + clock(), clock_probe=lambda ops: 1000.0 + clock())
    res = d.run()
    assert EnvOps.envs == [{'INFRARSI_TOKEN': sim.token}] and res['stop_reason'] == 'loop_finished'
    assert res['event_errors'] and not res['lease_kept']
    assert list((tmp_path / 'node' / 'data' / 'infrarsi' / 'leases').rglob('*.json')) == []


class CopyCheckOps(LocalNodeOps):
    """Plays the node for the driver's cleanup copy check (the sh command is not run on this PC)."""
    write_post, sh_cmds = True, []

    def call(self, func, _timeout=300, **kw):
        cmd = kw.get('cmd', '') if func == 'sh' else ''
        if 'optico_node.py hash' in cmd:                              # the driver's after-run base hash
            path = next(t for t in cmd.split() if t.endswith('/frozen_after.json'))
            write(self.local(path).parent, 'frozen_after.json', frozen_json(taskmod.frozen_files()))
            return {'code': 0, 'stdout': '', 'stderr': ''}
        if 'copycheck' in cmd:
            self.sh_cmds.append(cmd)
            if self.write_post:
                path = next(t for t in cmd.split() if t.endswith('/copy_check_post.json'))
                write(self.local(path).parent, 'copy_check_post.json', json.dumps({'checked': 38, 'mismatch': []}))
            return {'code': 0, 'stdout': '', 'stderr': ''}
        return super().call(func, _timeout, **kw)


@pytest.mark.parametrize('write_post', [True, False])
def test_a_stopped_loop_still_gets_its_post_run_copy_check(tmp_path, write_post):
    clock, ref = VirtualClock(), []
    sim = LoopSim(ref, clock, 50)                     # the driver stops the loop at 4 rows: loop.sh never writes post
    CopyCheckOps.write_post, CopyCheckOps.sh_cmds = write_post, []
    ops = CopyCheckOps(tmp_path / 'node', node='WINH4', launcher=sim.launch)
    ref.append(ops)
    d = native.NativeDriver('W3', ops, run_id='bnat-c', cycles=21, max_trials=4, out_dir=tmp_path / 'pc', poll_s=10.0,
                            clock=clock, sleep=sim.step, tunnel_factory=lambda node, rp, lp, path: FakeTunnel(),
                            shim_factory=lambda: LoggingShim(sim, tmp_path / 'pc', offset=-4000.0),
                            port_finder=lambda ops: 18123, endpoint_check=lambda ops, port: True, log=lambda m: None,
                            wallclock=lambda: 5000.0 + clock(), clock_probe=lambda ops: 1000.0 + clock())
    res = d.run()
    recs = res['receipt_objects']
    assert res['stop_reason'] == 'max_trials' and len(recs) == 4 and len(CopyCheckOps.sh_cmds) == 1
    assert '/workspace/code/OptiCo' in CopyCheckOps.sh_cmds[0] and ' loop ' in CopyCheckOps.sh_cmds[0]
    if write_post:
        assert all(r['valid'] for r in recs)
    else:                                             # no evidence the copy stayed intact: every row fails closed
        assert not any(r['valid'] for r in recs)
        assert all(any('copy_check_post.json missing' in str(g) for g in r['guard_rejections']) for r in recs)


def test_occupied_node_writes_no_trace(tmp_path):
    clock = VirtualClock()
    busy = {'hostname': 'fake', 'utc': 'x', 'error': None, 'gpus': [{'index': g, 'mem_used_mib': 0} for g in range(8)],
            'apps': [{'pid': 99, 'gpu_index': 3, 'used_mib': 30000, 'cwd': '/data/tyc/job', 'cmdline': 'python train.py'}]}
    ops = LocalNodeOps(tmp_path / 'node', node='WINH4', probe_fn=lambda: busy)
    d = native.NativeDriver('W3', ops, run_id='bnat-busy', cycles=3, out_dir=tmp_path / 'pc', clock=clock, sleep=clock.advance,
                            log=lambda m: None)
    res = d.run()
    assert res['status'] == 'occupied' and not (tmp_path / 'pc' / 'trace.json').exists()
