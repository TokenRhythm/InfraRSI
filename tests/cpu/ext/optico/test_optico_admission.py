"""H-05 (Codex W9 R1): OptiCoRunner's bounded admission -- the worst case per attempt is the sum of phase deadlines that
DeadlineOps enforces on every remote call (the common lease's included), the admitted wall reaches the node
(DEADLINE_S, monitor timeout = min(cap, W - elapsed)), and SearchSession refuses a trial that cannot fit."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from infrarsi.agent.budget import VirtualClock
from infrarsi.ext.common.remote_exec import LocalNodeOps
from infrarsi.ext.optico import arms
from infrarsi.ext.optico import runner as rn
from infrarsi.ext.optico import task as taskmod
from infrarsi.ext.optico import workloads as wl
from infrarsi.ext.optico.runner import AttemptDeadline, DeadlineOps, OptiCoRunner
from test_optico_runner import FakeTrial

W3 = wl.get('W3')
OVER = rn.PRE_LAUNCH_S + rn.KILL_AFTER_S + 120.0 + 10.0 + rn.MIN_CALL_S + rn.POST_S          # 565 s (grace 120, poll 10)


def _runner(tmp_path, seconds=144.0, scenario='success', ops_cls=LocalNodeOps, **kw):
    clock = VirtualClock()
    fake = FakeTrial(scenario, clock=clock, seconds=seconds)
    ops = ops_cls(tmp_path / 'node', node='WINH4', launcher=fake)
    ops.clock = clock
    return OptiCoRunner(ops, 'W3', out_dir=tmp_path / 'pc', clock=clock, sleep=clock.advance, poll_s=10, **kw), fake, clock


def test_worst_case_and_bounded_cap(tmp_path):
    r, _, _ = _runner(tmp_path)
    assert r.attempt_overhead_s(bounded=True) == OVER == 565.0 and rn.PRE_LAUNCH_S == 180.0
    assert r.attempt_overhead_s(bounded=False) == rn.PRE_LAUNCH_S + 120.0 + rn.MATERIALIZE_SLACK_S + 10.0 + rn.MIN_CALL_S + rn.POST_S
    assert r.cap_s == 900.0 and r.worst_case_wall_s(400) == 965.0 and r.worst_case_wall_s() == r.worst_case() == 900.0 + 835.0
    assert r.worst_case_gpu_hours(8, 400) == pytest.approx(8 * 965 / 3600) and r.worst_case_gpu_hours(8, 0) == 0.0
    assert r.pair_wall_cap(200.0) == 0 and r.stop_grace_s == 0.0               # ENV-B: no locked-pair reserve in the arm
    # plenty left: W = cap + materialization slack; little wall left: W shrinks so that W + overhead fits exactly
    assert r.bounded_cap(10.0, 8, 5400.0, 200.0) == 1200
    assert r.bounded_cap(10.0, 8, 810.0, 200.0) == 245 and 245 + OVER == 810.0
    assert r.bounded_cap(10.0, 8, 710.0, 200.0) is None                       # 145 s < the expected 200 s
    assert r.bounded_cap(0.1, 8, 5400.0, 200.0) is None                       # 45 GPU-s per GPU left: nothing fits
    # cheap rejected screens lower the session's mean: the measured default trial (130 s) + 30 s is the floor
    assert r.bounded_cap(10.0, 8, 724.0, 10.0) is None and r.bounded_cap(10.0, 8, 725.0, 10.0) == 160


class Recorder:
    node = 'WINH4'

    def __init__(self):
        self.calls = []

    def call(self, func, _timeout=300, **kw):
        self.calls.append((func, _timeout, kw))
        return {'ok': True}


def test_deadline_ops_cuts_timeouts_refuses_late_calls_and_keeps_essential_ones():
    clock, inner = VirtualClock(40.0), Recorder()
    ops = DeadlineOps(inner, clock)
    ops.read_json('/x')                                                       # no deadline yet: untouched
    ops.set_phase('trial', 100.0, essential_deadline=200.0)
    ops.read_json('/x')
    ops.sh('true', timeout=120)
    assert [c[1] for c in inner.calls] == [300, 60.0, 60.0] and inner.calls[2][2]['timeout'] == 58
    clock.advance(55)                                                         # 95: 5 s left -> 5 s (no floor)
    ops.read_json('/x')
    assert inner.calls[-1][1] == 5.0
    clock.advance(25)                                                         # 120: past the phase deadline
    with pytest.raises(AttemptDeadline):
        ops.read_files(['/x'])
    ops.kill_token('tok')                                                     # essential: until 200
    assert inner.calls[-1][0] == 'kill_token' and inner.calls[-1][1] == 80.0 and ops.overruns == []
    clock.advance(100)                                                        # 220: past the end deadline
    ops.call('release_status', path='/l', attempt_id='a')
    assert inner.calls[-1][1] == rn.MIN_CALL_S and ops.overruns[0]['func'] == 'release_status'
    assert ops.refused[0]['func'] == 'read_files' and ops.node == 'WINH4'


def test_admitted_wall_reaches_the_node(tmp_path):
    task = taskmod.make_task('W3', 'B-RND')
    r, fake, _ = _runner(tmp_path)
    rec = r.run(task, W3.defaults, None, 'screen', 'run1', candidate_id='c1', wall_cap=400)
    spec = fake.launched[0]['spec']
    assert spec['DEADLINE_S'] == '400' and spec['CAP_S'] == '400' and spec['WATCHDOG_S'] == str(400 + rn.WATCHDOG_EXTRA_S)
    assert rec['valid'] and rec['runner']['admitted_wall_s'] == 400.0 and rec['runner']['cap_s_effective'] == 400.0
    assert rec['runner']['worst_case_wall_s'] == 965.0
    assert rec['deadline']['worst_case_s'] == 965.0 and rec['deadline']['overrun_s'] == 0.0 and not rec['deadline']['refused_calls']
    rec = r.run(task, W3.defaults, None, 'screen', 'run1', candidate_id='c2', wall_cap=5000)
    spec = fake.launched[1]['spec']
    assert spec['DEADLINE_S'] == '5000' and spec['CAP_S'] == '900'           # the per-trial cap still bounds the monitor
    rec = r.run(task, W3.defaults, None, 'b_def', 'run1', candidate_id='def')  # no admission: as before
    spec = fake.launched[2]['spec']
    assert 'DEADLINE_S' not in spec and spec['CAP_S'] == '900' and spec['WATCHDOG_S'] == str(900 + rn.WATCHDOG_EXTRA_S)
    assert 'deadline' not in rec and r.ops is not None and not isinstance(r.ops, DeadlineOps)


def test_hung_bounded_trial_ends_within_its_worst_case(tmp_path):
    task = taskmod.make_task('W3', 'B-RND')
    r, fake, clock = _runner(tmp_path, scenario='hang')
    t0 = clock()
    rec = r.run(task, W3.defaults, None, 'screen', 'run1', candidate_id='c3', wall_cap=300)
    assert not rec['valid'] and rec['runner']['overdue'] and fake.launched[0]['token'] in r.ops.killed_tokens
    assert clock() - t0 <= r.worst_case_wall_s(300) and rec['deadline']['overrun_s'] == 0.0
    assert list((tmp_path / 'node' / 'data' / 'infrarsi' / 'leases').rglob('*.json')) == []


class SlowOps(LocalNodeOps):
    """Every remote call takes `delay` seconds of the test clock; a call longer than its transport timeout times out."""
    delay = 8.0

    def call(self, func, _timeout=300, **kw):
        if self.delay > _timeout:
            self.clock.advance(_timeout)
            raise TimeoutError(f'{func}: transport timeout {_timeout}s')
        self.clock.advance(self.delay)
        return super().call(func, _timeout, **kw)


def test_slow_transport_cannot_push_an_attempt_past_its_deadlines(tmp_path):
    task = taskmod.make_task('W3', 'B-RND')
    SlowOps.delay = 14.0                          # 14 s per call: lease + mkdirs take 168 s, the upload gets the last 12 s
    try:
        r, fake, clock = _runner(tmp_path, ops_cls=SlowOps)
        t0 = clock()
        rec = r.run(task, W3.defaults, None, 'screen', 'run1', candidate_id='c4', wall_cap=400)
        assert fake.launched == [] and not rec['valid'] and rec['not_launched'] == 'pre_launch'
        assert rec['invalid_reasons'][0].startswith('infra_refusal:') and 'transport timeout 12.0s' in rec['invalid_reasons'][1]
        assert clock() - t0 <= rn.PRE_LAUNCH_S + 9 * SlowOps.delay + 1e-9  # the deadline, then the partial lease's release
        assert list((tmp_path / 'node' / 'data' / 'infrarsi' / 'leases').rglob('*.json')) == []
        SlowOps.delay = 8.0                       # 8 s per call (above the 6.3 s seen on WINH4): the attempt runs normally
        r2, fake2, clock2 = _runner(tmp_path / 'b', ops_cls=SlowOps)
        t0 = clock2()
        rec = r2.run(task, W3.defaults, None, 'screen', 'run1', candidate_id='c5', wall_cap=400)
        assert rec['valid'] and fake2.launched and clock2() - t0 <= r2.worst_case_wall_s(400)
        assert rec['deadline']['overrun_s'] == 0.0 and not rec['deadline']['refused_calls']
    finally:
        SlowOps.delay = 8.0


class StallOps(LocalNodeOps):
    """One pre-launch call stalls past the whole pre-launch phase; the next call is refused (AttemptDeadline)."""

    def call(self, func, _timeout=300, **kw):
        if func == 'mkdirs':
            self.clock.advance(rn.PRE_LAUNCH_S + 5)
        return super().call(func, _timeout, **kw)


def test_a_pre_launch_refusal_is_an_infra_refusal_and_occupied_keeps_its_token(tmp_path):
    task = taskmod.make_task('W3', 'B-RND')
    r, fake, _ = _runner(tmp_path, ops_cls=StallOps)
    rec = r.run(task, W3.defaults, None, 'screen', 'run1', candidate_id='c6', wall_cap=400)
    assert fake.launched == [] and rec['not_launched'] == 'pre_launch'
    assert rec['invalid_reasons'][0].startswith('infra_refusal:') and 'AttemptDeadline' in rec['invalid_reasons'][1]
    assert rec['deadline']['refused_calls'][0]['func'] == 'write_files'
    busy = {'hostname': 'fake', 'utc': 'x', 'error': None, 'gpus': [{'index': g, 'mem_used_mib': 0} for g in range(8)],
            'apps': [{'pid': 99, 'gpu_index': 3, 'used_mib': 30000, 'cwd': '/data/tyc/job', 'cmdline': 'python train.py'}]}
    clock = VirtualClock()
    ops = LocalNodeOps(tmp_path / 'busy', node='WINH4', launcher=FakeTrial(clock=clock), probe_fn=lambda: busy)
    r2 = OptiCoRunner(ops, 'W3', clock=clock, sleep=clock.advance)
    rec = r2.run(task, W3.defaults, None, 'screen', 'run1', candidate_id='c7', wall_cap=400)
    assert rec['invalid_reasons'][0].startswith('occupied') and 'not_launched' not in rec


class StallOnceOps(StallOps):
    stalled = False

    def call(self, func, _timeout=300, **kw):
        if func == 'mkdirs' and not StallOnceOps.stalled:
            StallOnceOps.stalled = True
            return StallOps.call(self, func, _timeout, **kw)
        return LocalNodeOps.call(self, func, _timeout, **kw)


def test_a_session_records_the_refused_start_as_infra_refusal_and_continues(tmp_path):
    StallOnceOps.stalled = False
    r, fake, clock = _runner(tmp_path, ops_cls=StallOnceOps)
    tr = arms.run_b_rnd('W3', None, run_id='brnd', out_dir=tmp_path / 'trace', screens=3, runner=r, clock=clock)
    s = tr['screens']
    assert s[0]['candidate_id'] == 'c0' and s[0]['reasons'][0].startswith('infra_refusal') and not s[0]['valid']
    assert len(s) == 3 and any(x['valid'] for x in s[1:]) and len(fake.launched) == 2


def test_infra_refusals_do_not_count_toward_the_same_failure_stop(tmp_path):
    r, fake, clock = _runner(tmp_path, ops_cls=StallOps)            # every attempt refused before launch
    tr = arms.run_b_rnd('W3', None, run_id='brnd', out_dir=tmp_path / 'trace', screens=8, runner=r, clock=clock)
    assert fake.launched == [] and len(tr['screens']) == 8
    assert all(x['reasons'][0].startswith('infra_refusal:') for x in tr['screens'])
    assert tr['stop_reason'] == 'budget:screens'                    # not same_failure_x6 (agent/search.py exemption)


class LostAcquireOps(LocalNodeOps):
    """The third GPU claim is created on the node, but its answer is lost."""

    def call(self, func, _timeout=300, **kw):
        res = super().call(func, _timeout, **kw)
        if func == 'acquire' and str(kw.get('path', '')).endswith('2.json'):
            raise TimeoutError('acquire: answer lost')
        return res


def test_a_claim_whose_answer_was_lost_is_released(tmp_path):
    task = taskmod.make_task('W3', 'B-RND')
    r, fake, _ = _runner(tmp_path, ops_cls=LostAcquireOps)
    rec = r.run(task, W3.defaults, None, 'screen', 'run1', candidate_id='c7', wall_cap=400)
    assert fake.launched == [] and not rec['valid']
    assert [a['state'] for a in rec['deadline']['ambiguous_acquires']] == ['removed']
    assert list((tmp_path / 'node' / 'data' / 'infrarsi' / 'leases').rglob('*.json')) == []


class NoKillOps(LocalNodeOps):
    """The token kill never gets through (transport)."""

    def call(self, func, _timeout=300, **kw):
        if func == 'kill_token':
            raise TimeoutError('kill_token: transport timeout')
        return super().call(func, _timeout, **kw)


@pytest.mark.parametrize('wall_cap', [400, None])
def test_an_unconfirmed_kill_keeps_the_gpus_leased(tmp_path, wall_cap):
    task = taskmod.make_task('W3', 'B-RND')
    r, fake, _ = _runner(tmp_path, ops_cls=NoKillOps)
    kw = {'wall_cap': wall_cap} if wall_cap else {}
    rec = r.run(task, W3.defaults, None, 'screen', 'run1', candidate_id='c8', **kw)
    assert fake.launched and not rec['valid'] and rec['containment']['ok'] is False
    assert 'token kill is unconfirmed' in rec['containment']['error'] and rec['lease_release']['kept']
    assert list((tmp_path / 'node' / 'data' / 'infrarsi' / 'leases').rglob('*.json'))          # left for a human


class EnvOps(LocalNodeOps):
    envs = []

    def call(self, func, _timeout=300, **kw):
        if func == 'launch':
            self.envs.append(dict(kw.get('env') or {}))
        return super().call(func, _timeout, **kw)


def test_trial_sh_carries_the_token_from_exec(tmp_path):
    EnvOps.envs = []
    r, fake, _ = _runner(tmp_path, ops_cls=EnvOps)
    r.run(taskmod.make_task('W3', 'B-RND'), W3.defaults, None, 'screen', 'run1', candidate_id='c9', wall_cap=400)
    assert EnvOps.envs == [{'INFRARSI_TOKEN': fake.launched[0]['token']}]


def test_kill_confirmation_is_pessimistic_and_logging_cannot_break_it(tmp_path):
    def boom(msg):
        raise OSError('log volume gone')
    r, _, _ = _runner(tmp_path, ops_cls=NoKillOps, log=boom)
    assert r._kill_token('isxb-x', 0.0) == ['unconfirmed: the lease is kept'] and r._kill_unconfirmed
    with pytest.raises(Exception, match='unconfirmed'):
        r.contain()
    r2, _, _ = _runner(tmp_path / 'b', log=boom)
    assert r2._kill_token('isxb-y', 0.0) == [] and not r2._kill_unconfirmed and r2.contain() == {}


class LateLaunchOps(LocalNodeOps):
    """The launch starts trial.sh on the node, but its answer is lost (transport timeout)."""
    order = []

    def call(self, func, _timeout=300, **kw):
        self.order.append(func)
        res = super().call(func, _timeout, **kw)
        if func == 'launch':
            raise TimeoutError('launch: transport timeout after the script started')
        return res


@pytest.mark.parametrize('wall_cap', [400, None])
def test_a_launch_that_times_out_is_still_killed_before_the_release(tmp_path, wall_cap):
    task = taskmod.make_task('W3', 'B-RND')
    LateLaunchOps.order = []
    r, fake, _ = _runner(tmp_path, scenario='hang', ops_cls=LateLaunchOps)
    kw = {'wall_cap': wall_cap} if wall_cap else {}
    rec = r.run(task, W3.defaults, None, 'screen', 'run1', candidate_id='c6', **kw)
    order = LateLaunchOps.order
    assert not rec['valid'] and fake.launched and fake.launched[0]['token'] in r.ops.killed_tokens
    assert order.index('kill_token') > order.index('launch')
    assert order.index('kill_token') < max(i for i, f in enumerate(order) if f in ('release', 'release_status'))


def test_search_session_never_admits_past_the_wall_budget(tmp_path):
    r, fake, clock = _runner(tmp_path)
    tr = arms.run_b_rnd('W3', None, run_id='brnd', out_dir=tmp_path / 'trace', screens=21, wall_s=1500.0, runner=r, clock=clock)
    assert tr['stop_reason'] == 'budget:wall_seconds' and len(tr['screens']) == 6
    for s in tr['screens']:                       # every admitted attempt's worst case fit the wall left at admission
        rec = json.loads((tmp_path / 'trace' / 'receipts' / f'{s["candidate_id"]}.json').read_text(encoding='utf-8'))
        before = s['cum']['wall_seconds'] - s['wall_seconds']
        assert rec['runner']['admitted_wall_s'] is not None and rec['deadline']['overrun_s'] == 0.0
        assert before + rec['runner']['worst_case_wall_s'] <= 1500.0 + 1e-6
    assert tr['used']['wall_seconds'] <= 1500.0
    assert [x['spec']['DEADLINE_S'] for x in fake.launched][-1] == '215'    # the last one ran with what was left


@pytest.mark.skipif(shutil.which('bash') is None or 'system32' in (shutil.which('bash') or '').lower(),
                    reason='needs a POSIX bash (not the WSL launcher)')
def test_trial_sh_monitor_timeout_honours_the_deadline():
    text = (rn.NODE_DIR / 'trial.sh').read_text(encoding='utf-8')
    block = re.search(r'MON_CAP=\$CAP_S\n.*?\nfi\n', text, re.S).group(0)
    script = 't_start=$(( $(date +%s) - 100 )).5\n' + block + 'echo $MON_CAP\n'
    run = lambda env: subprocess.run(['bash', '-c', script], capture_output=True, text=True, env=env, timeout=60)  # noqa: E731
    base = {'PATH': __import__('os').environ['PATH']}
    assert run({**base, 'CAP_S': '900', 'DEADLINE_S': '400'}).stdout.split()[-1] in ('300', '299')
    assert run({**base, 'CAP_S': '200', 'DEADLINE_S': '5000'}).stdout.split()[-1] == '200'
    assert run({**base, 'CAP_S': '900', 'DEADLINE_S': ''}).stdout.split()[-1] == '900'
    assert int(run({**base, 'CAP_S': '900', 'DEADLINE_S': '50'}).stdout.split()[-1]) < 1    # nothing left: not launched
    assert 'if [ "$MON_CAP" -ge 1 ]' in text and 'DEADLINE_S=${DEADLINE_S:-}' in text
