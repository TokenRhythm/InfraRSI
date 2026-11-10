"""X1 follow-up 2 (Codex W9 round 1): uid trust domains and containment (B-02), worst-case admission and wall caps
(H-05), the USD cap before every LLM call (H-06), lease release retries (M-02) and canonical patch paths."""
import json
import os
import subprocess
import sys

import pytest

import importlib
from infrarsi.agent.backends import ScriptedBackend
from infrarsi.agent.budget import Budget, VirtualClock
from infrarsi.agent.search import SearchSession
from infrarsi.ext.common import node_ext
from infrarsi.ext.common.basic_agent import BasicAgent, EnvTool, ShellExecutor
from infrarsi.ext.common.env_judge import EnvJudge
from infrarsi.ext.common.guards import path_problem
from infrarsi.ext.common.remote_exec import (ContainmentError, GpuLease, LocalNodeOps, NodeError, assert_uid_unused,
                                               gpu_claim_path, node_lease_path, uid_kill)
from infrarsi.ext.common.session_hooks import SessionHooks
from infrarsi.rsi import build_a0
from x1_common_helpers import BASE_FILES, ToyRunner, local_ops, toy_task

PASS = {'verdict': 'pass', 'reasons': []}
opt = importlib.import_module('infrarsi.agent.optimize')


@pytest.fixture(scope='module')
def a0(tmp_path_factory):
    return build_a0(tmp_path_factory.mktemp('pkg') / 'A0')


class UsdBackend(ScriptedBackend):
    """Records the max_usd every call was given (None: the keyword was not passed)."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.max_usd = []

    def complete(self, system, prompt, **kw):
        self.max_usd.append(kw.get('max_usd'))
        return super().complete(system, prompt, **kw)


class BoundedToy(ToyRunner):
    """The toy environment with a declared worst case and a trust-domain uid."""
    worst_case_s = 300.0

    def domain_uids(self):
        return [64010]

    def execute(self, task, config, overlay, purpose, run_id, attempt_id, candidate_id, **kw):
        self.seen_caps = getattr(self, 'seen_caps', []) + [kw.get('wall_cap')]
        return super().execute(task, config, overlay, purpose, run_id, attempt_id, candidate_id, **kw)


# ---------------------------------------------------------------- B-02: uid trust domains
def test_uid_kill_and_unused_checks(tmp_path):
    ops = LocalNodeOps(tmp_path)
    for bad in (65534, 0, 1000, 59999):
        with pytest.raises(ValueError):
            uid_kill(ops, bad)
        with pytest.raises(ValueError):
            assert_uid_unused(ops, bad)
        with pytest.raises(ValueError):
            node_ext._check_domain_uid(bad)
    assert uid_kill(ops, 64001)['remaining'] == [] and ops.uid_kills == [64001]
    ops.uid_survivors[64001] = [99]
    with pytest.raises(ContainmentError, match='survived'):
        uid_kill(ops, 64001)
    assert assert_uid_unused(ops, 64002)['pids'] == []
    ops.uid_procs[64002] = [5]
    with pytest.raises(ContainmentError, match='in use'):
        assert_uid_unused(ops, 64002)
    ops.uid_passwd[64003] = 'svc'
    with pytest.raises(ContainmentError, match='svc'):
        assert_uid_unused(ops, 64003)


@pytest.mark.skipif(not os.path.isdir('/proc'), reason='needs /proc (the node side runs on Linux)')
def test_node_ext_uid_scan_on_linux():
    me = os.getuid()
    assert os.getpid() not in node_ext.uid_pids(me)                     # the scanning process itself is excluded
    with pytest.raises(ValueError):
        node_ext.uid_kill(65534)


def test_nobody_cmd_sets_no_new_privs():
    from infrarsi.ext.common.remote_exec import nobody_cmd
    cmd = nobody_cmd('id', home='/h', uid=64001)
    assert '--reuid=64001 --regid=64001 --clear-groups --no-new-privs env -i ' in cmd


def _shell(tmp_path, launcher, **kw):
    clock = VirtualClock()
    ops = LocalNodeOps(tmp_path, launcher=launcher)
    sh = ShellExecutor(ops, home='/h', cwd='/h', work_dir='/data/infrarsi/w', token='isx-cnat-0123456789', clock=clock,
                       sleep=clock.advance, **kw)
    return sh, ops


def test_shell_executor_cleanup_is_in_finally_and_by_uid(tmp_path):
    sh, ops = _shell(tmp_path, lambda *a: {'pid': 1}, uid=64002)
    real = ops.call

    def flaky(func, _timeout=300, **kw):                                 # SSH drops while polling for the exit code
        if func == 'read_files' and any(str(p).endswith('.rc') for p in kw.get('paths', [])):
            raise NodeError('ssh: connection reset')
        return real(func, _timeout, **kw)
    ops.call = flaky
    with pytest.raises(NodeError):
        sh.bash('setsid sleep 1000 &', 30)
    assert ops.uid_kills == [64002] and ops.killed_tokens == ['isx-cnat-0123456789']     # cleaned up anyway
    ops.call = real
    sh.bash('true', 5)
    assert ops.uid_kills == [64002, 64002]                              # after every call
    ops.uid_survivors[64002] = [31337]
    with pytest.raises(ContainmentError):                               # a survivor is never reported as success
        sh.bash('true', 5)
    legacy, lops = _shell(tmp_path / 'l', lambda *a: {'pid': 1})        # uid 65534: token label only, never kill-by-uid
    legacy.bash('true', 5)
    assert lops.uid_kills == [] and lops.killed_tokens == ['isx-cnat-0123456789']


def test_env_runner_contains_before_release_and_keeps_the_lease_on_survivors(tmp_path):
    task = toy_task()
    ops = local_ops(tmp_path)
    r = BoundedToy(ops)
    rec = r.run(task, {'workers': 2}, None, 'screen', 'run1', candidate_id='c1')
    assert rec['valid'] and rec['containment']['ok'] and ops.uid_kills == [64010]
    assert not ops.local(node_lease_path('WINH0')).exists()
    ops.uid_survivors[64010] = [7]
    rec = r.run(task, {'workers': 2}, None, 'screen', 'run1', candidate_id='c2')
    assert not rec['valid'] and rec['metric']['value'] is None and not rec['containment']['ok']
    assert any(x.startswith('containment:') for x in rec['invalid_reasons']) and EnvJudge('toyenv')(rec).valid is False
    assert ops.local(node_lease_path('WINH0')).exists() and rec['lease_release']['kept']
    for g in r.gpus:
        assert ops.local(gpu_claim_path('WINH0', g)).exists()            # the GPUs stay claimed


# ---------------------------------------------------------------- M-02: lease release
def test_gpu_lease_release_retries_and_never_forgets(tmp_path):
    ops = local_ops(tmp_path)
    lease = GpuLease(ops, env='toyenv', gpus=[0, 1])
    lease.acquire('r', 'a1')
    real, fails = ops.call, {'n': 2}

    def flaky(func, _timeout=300, **kw):
        if func == 'release_status' and fails['n'] > 0:
            fails['n'] -= 1
            raise NodeError('ssh: timeout')
        return real(func, _timeout, **kw)
    ops.call = flaky
    waits = []
    done = lease.release(sleep=waits.append)
    assert len(done) == 3 and lease.acquired == [] and lease.release_failures == [] and waits == [1.0, 2.0]
    assert lease.release(sleep=waits.append) == []                       # idempotent
    lease = GpuLease(ops, env='toyenv', gpus=[2])
    lease.acquire('r', 'a2')
    fails['n'] = 99
    done = lease.release(sleep=waits.append, attempts=3)
    assert done == [] and len(lease.acquired) == 2 and len(lease.release_failures) == 2            # kept, recorded
    assert all('ssh: timeout' in f['error'] for f in lease.release_failures)
    fails['n'] = 0
    assert len(lease.release(sleep=waits.append)) == 2 and lease.acquired == []                    # a later retry works
    lease = GpuLease(ops, env='toyenv', gpus=[3])
    lease.acquire('r', 'a3')
    path = gpu_claim_path('WINH0', 3)
    ops.local(path).write_text(json.dumps({'attempt_id': 'someone-else', 'owner': 'x'}), encoding='utf-8')
    done = lease.release(sleep=waits.append)
    assert path not in done and ops.local(path).exists() and lease.release_failures[-1]['error'] == 'held by another attempt'


# ---------------------------------------------------------------- H-05: worst-case admission, wall cap
def test_search_session_admits_only_when_the_worst_case_fits(tmp_path):
    task = toy_task()
    hooks = SessionHooks(task, base_files=BASE_FILES)
    r = BoundedToy(local_ops(tmp_path))
    budget = task.search_budget(wall_seconds=500.0, gpu_hours=100.0, screens=50)
    s = SearchSession(task=task, budget=budget, runner=r, judge=EnvJudge(), run_id='r', arm='T-RND', start_config={},
                      layers=('L0',), clock=r.vclock, hooks=hooks)
    s.screen('c0')
    for w in (2, 4, 8, 1, 2, 4):
        fb = s.propose_config(config={'workers': w, 'flash': w == 1})
        if 'candidate_id' in fb:
            out = s.screen(fb['candidate_id'])
            if 'error' in out:
                break
    assert s.stop_flag == 'budget:wall_seconds' and r.seen_caps and all(c == 300 for c in r.seen_caps)
    assert budget.wall_seconds - s.ledger.wall() < 300 <= budget.wall_seconds - (s.ledger.wall() - s.screens[-1]['wall_seconds'])
    r2 = BoundedToy(local_ops(tmp_path / 'g'))                           # GPU-hours: world 4 x 300 s per attempt
    b2 = task.search_budget(wall_seconds=1e6, gpu_hours=0.5, screens=50)
    s2 = SearchSession(task=task, budget=b2, runner=r2, judge=EnvJudge(), run_id='r', arm='T-RND', start_config={},
                       layers=('L0',), clock=r2.vclock, hooks=SessionHooks(task, base_files=BASE_FILES))
    s2.screen('c0')                                                      # 4 x 120 s: 0.133 h used, 330 s of GPU left each
    assert 'error' not in s2.screen(s2.propose_config(config={'workers': 2})['candidate_id'])
    fb = s2.screen(s2.propose_config(config={'workers': 4})['candidate_id'])          # 260 s left < the 300 s worst case
    assert 'error' in fb and s2.stop_flag == 'budget:gpu_hours' and len(r2.seen_caps) == 2
    assert ToyRunner(local_ops(tmp_path / 'n')).bounded_cap is None      # no worst case declared: mean-cost admission


def test_basic_agent_experiment_admission_and_wall_cap(tmp_path):
    task = toy_task()
    runner = BoundedToy(local_ops(tmp_path))
    calls = []

    def score(args, agent):
        calls.append(agent.wall_left())
        rec = runner.run(task, {'workers': 2}, None, 'screen', 'nat', candidate_id=f'e{len(calls)}', wall_cap=agent.wall_left())
        agent.record_experiment(rec, EnvJudge('toyenv')(rec), candidate_id=f'e{len(calls)}')
        return {'valid': rec['valid']}
    tool = EnvTool('score', '{"tool":"score"}', {}, score, counts_as_experiment=True,
                   admit=lambda agent: runner.admits(agent.ledger, agent.budget))
    drv = ScriptedBackend(lambda role, s, p: {'tool': 'score'}, clock=runner.vclock, tokens_per_call=10)
    b = Budget(screens=50, llm_tokens=10_000_000, wall_seconds=650.0, gpu_hours=1e6, scoring_reserve_screens=0.0)
    agent = BasicAgent(driver=drv, executor=None, task_text='T', env_tools=[tool], budget=b, run_id='n', env='toyenv',
                       task_id=task.id, arm='T-NAT', clock=runner.vclock)
    tr = agent.run()
    assert tr['stop_reason'] == 'budget:wall_seconds' and len(calls) == len(tr['screens']) >= 1
    assert b.wall_seconds - agent.ledger.wall() < 300 and all(c >= 300 for c in calls)           # never started short
    assert runner.seen_caps == [pytest.approx(c, abs=1.0) for c in calls]                         # the cap reached execute


# ---------------------------------------------------------------- H-06: USD cap before every call
def test_usd_allowance_rules():
    class Led:
        used = {'llm_usd': 0.0}
    led = Led()
    assert opt.usd_allowance(Budget(screens=1, llm_tokens=1, wall_seconds=1, gpu_hours=1), led) == (True, None)
    b = Budget(screens=1, llm_tokens=1, wall_seconds=1, gpu_hours=1, llm_usd=1.0)
    assert opt.usd_allowance(b, led) == (True, 1.0)
    led.used['llm_usd'] = 0.96
    ok, left = opt.usd_allowance(b, led)
    assert not ok and left == pytest.approx(0.04)                       # below MIN_CALL_USD (0.05)

    class Priced:                                                       # a backend that prices tokens itself
        price = {'input_miss': 0.30, 'output': 1.20}
        max_tokens, max_retries = 4096, 3
    led.used['llm_usd'] = 0.0
    assert opt.usd_allowance(b, led, Priced(), 'x' * 1000)[0]
    assert not opt.usd_allowance(b, led, Priced(), 'x' * 900_000)[0]    # 4 x (0.9M bytes x $0.3/M) > $1
    assert opt.usd_kw(None) == {} and opt.usd_kw(0.5) == {'max_usd': 0.5}


def test_basic_agent_passes_and_enforces_the_usd_cap(tmp_path):
    task = toy_task()
    runner = ToyRunner(local_ops(tmp_path))
    drv = UsdBackend(lambda role, s, p: {'tool': 'bash', 'cmd': 'true'}, clock=runner.vclock, tokens_per_call=10, usd_per_call=0.02)

    class Shell:
        def bash(self, cmd, timeout):
            return {'exit_code': 0, 'timed_out': False, 'seconds': 0.1, 'stdout': '', 'stderr': ''}
    b = Budget(screens=5, llm_tokens=10_000_000, wall_seconds=1e6, gpu_hours=1e6, llm_usd=0.1, scoring_reserve_screens=0.0)
    agent = BasicAgent(driver=drv, executor=Shell(), task_text='T', env_tools=[], budget=b, run_id='n', env='toyenv',
                       task_id=task.id, arm='T-NAT', clock=runner.vclock)
    tr = agent.run()
    assert tr['stop_reason'] == 'budget:llm_usd' and drv.max_usd == pytest.approx([0.1, 0.08, 0.06])
    assert tr['used']['llm_usd'] == pytest.approx(0.06) and tr['used']['llm_usd'] <= b.llm_usd
    drv2 = UsdBackend([{'tool': 'submit'}], clock=runner.vclock)
    BasicAgent(driver=drv2, executor=Shell(), task_text='T', env_tools=[], run_id='n', env='toyenv', task_id=task.id,
               arm='T-NAT', clock=runner.vclock,
               budget=Budget(screens=5, llm_tokens=10_000_000, wall_seconds=1e6, gpu_hours=1e6)).run()
    assert drv2.max_usd == [None]                                       # no cap: the call is unchanged


def test_optimize_roles_get_the_remaining_usd(tmp_path, a0):
    budget = {'screens': 6, 'llm_tokens': 50_000, 'wall_seconds': 36000.0, 'gpu_hours': 10.0, 'llm_usd': 0.1}
    task = toy_task(budget=budget)
    runner = ToyRunner(local_ops(tmp_path))
    drv = UsdBackend(lambda role, s, p: PASS if role == 'verifier' else {'action': 'read_env'}, clock=runner.vclock,
                     tokens_per_call=100, usd_per_call=0.02)
    tr = opt.optimize(task, a0, drv, task.search_budget(), {}, runner, judge=EnvJudge('toyenv'), run_id='usd', arm='T-IS',
                      out_dir=tmp_path / 'run', clock=runner.vclock, hooks=SessionHooks(task, base_files=BASE_FILES)).to_dict()
    assert tr['stop_reason'] == 'budget:llm_usd' and drv.max_usd == pytest.approx([0.1, 0.08, 0.06])
    assert tr['used']['llm_usd'] <= 0.1


# ---------------------------------------------------------------- canonical patch paths
@pytest.mark.parametrize('path', ['/etc/passwd', '../x.py', 'solution/../score.py', 'solution//run.py', './solution/run.py',
                                  'solution\\run.py', 'C:/x.py', 'solution/run.py/', 'solution/run\n.py', ''])
def test_validate_patch_rejects_non_canonical_paths(path):
    hooks = SessionHooks(toy_task(), base_files=BASE_FILES)
    errors, _ = hooks.validate_patch({path: 'x = 1\n'})
    assert errors and errors[0].startswith('non-canonical paths') and path_problem(path)


def test_validate_patch_still_accepts_canonical_paths():
    hooks = SessionHooks(toy_task(), base_files=BASE_FILES)
    errors, _ = hooks.validate_patch({'solution/run.py': BASE_FILES['solution/run.py'].replace('1.0', '0.5')})
    assert errors == [] and path_problem('solution/run.py') is None and path_problem('a/b/c.py') is None


def test_code_hash_is_stable_across_hash_seeds():
    """Set constants are hashed in sorted order: the live (candidate process) and expected (root process) sides agree."""
    prog = ('import sys; sys.path.insert(0, sys.argv[1]); from infrarsi.ext.autoresearch.node import code_attest as c\n'
            'src = "def f(x):\\n    return x in {\\"alpha\\", \\"beta\\", \\"gamma\\", \\"delta\\"}\\n"\n'
            'm = compile(src, "t.py", "exec", dont_inherit=True)\nprint(c.code_hash(m.co_consts[0]))')
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    outs = set()
    for seed in ('1', '2', '3'):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outs.add(subprocess.run([sys.executable, '-c', prog, root], capture_output=True, text=True, env=env, check=True).stdout)
    assert len(outs) == 1
