import json
import re
from pathlib import Path

from infrarsi.agent.backends import ScriptedBackend
from infrarsi.agent.budget import Budget, VirtualClock
from infrarsi.ext.common.basic_agent import BasicAgent, EnvTool, ShellExecutor, best_screen, truncate
from infrarsi.ext.common.env_judge import EnvJudge
from infrarsi.ext.common.remote_exec import LocalNodeOps
from infrarsi.ext.common import trajectory as tj
from x1_common_helpers import ToyRunner, local_ops, toy_task


class FakeShell:
    def __init__(self, clock):
        self.clock, self.calls = clock, []

    def bash(self, cmd, timeout):
        self.calls.append(('bash', cmd, timeout))
        self.clock.advance(2)
        return {'exit_code': 0, 'timed_out': False, 'seconds': 2.0, 'stdout': 'x' * 20000 if 'big' in cmd else 'ok\n', 'stderr': ''}

    def python(self, code, timeout):
        self.calls.append(('python', code, timeout))
        return {'exit_code': 1, 'timed_out': False, 'seconds': 0.1, 'stdout': '', 'stderr': 'Traceback'}


def score_tool(tmp_path, runner, task):
    state = {'n': 0}

    def score(args, agent):
        state['n'] += 1
        cid = f'e{state["n"]}'
        rec = runner.run(task, {'workers': args.get('workers', 1)}, None, 'screen', 'nat', candidate_id=cid)
        agent.record_experiment(rec, EnvJudge('toyenv')(rec), candidate_id=cid)
        return {'score': rec['metric']['value'], 'valid': rec['valid']}
    return EnvTool('score', '{"tool":"score","workers":2}  scores the solution', {'workers': {'type': 'integer'}}, score,
                   counts_as_experiment=True)


def _agent(tmp_path, script, *, budget=None, context_tokens=150_000, **kw):
    task = toy_task()
    runner = ToyRunner(local_ops(tmp_path))
    clock = runner.vclock
    drv = ScriptedBackend(script, clock=clock, tokens_per_call=1000)
    agent = BasicAgent(driver=drv, executor=FakeShell(clock), task_text='TASK TEXT: speed up run.py', env_tools=[score_tool(tmp_path, runner, task)],
                       budget=budget or task.search_budget(), run_id='nat1', env='toyenv', task_id=task.id, arm='T-NAT',
                       out_dir=tmp_path / 'agent', clock=clock, context_tokens=context_tokens, **kw)
    return agent, drv, task


def test_tool_loop_records_experiments_and_transcript(tmp_path):
    script = [{'tool': 'bash', 'cmd': 'ls', 'timeout': 30, 'note': 'look around'},
              {'tool': 'python', 'code': 'print(1)'},
              {'tool': 'score', 'workers': 2},
              'not json',
              {'tool': 'bash'},                                           # missing cmd
              {'tool': 'score', 'workers': 4},
              {'tool': 'bash', 'cmd': 'big output', 'timeout': 5000},
              {'tool': 'submit', 'note': 'done'}]
    agent, drv, task = _agent(tmp_path, script)
    tr = agent.run()
    assert tr['stop_reason'] == 'submit' and tr['turns'] == 8 and len(tr['screens']) == 2
    assert [s['step_time'] for s in tr['screens']] == [50.0, 25.0] and tr['screens'][1]['best_valid'] == 'e2'
    assert tr['used']['screens'] == 2 and tr['used']['llm_tokens'] == 8000 and tr['accounting_complete']
    assert agent.exec.calls[-1][2] == 1800                             # bash timeout capped at 1800 s
    prompts = [c['prompt'] for c in drv.calls]
    assert 'TASK TEXT' in prompts[0] and '# BUDGET' in prompts[0] and 'rejected: ' in prompts[4]
    assert '[... ' in prompts[-1] and 'characters omitted' in prompts[-1]          # long outputs are cut head/tail
    out = Path(tmp_path / 'agent')
    kinds = [json.loads(x)['kind'] for x in (out / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert kinds.count('model_error') == 2 and kinds.count('screen') == 2 and (out / 'trace.json').exists()
    assert len(list((out / 'receipts').glob('e*.json'))) == 2 and (out / 'outputs' / 'turn_00001.json').exists()
    assert best_screen(tr)['candidate_id'] == 'e2'
    row = tj.arm_row(tr, task, 100.0)
    assert row['n_screens'] == 2 and row['best_search'] == 25.0


def test_budgets_stop_the_agent(tmp_path):
    loop = lambda role, s, p: {'tool': 'score', 'workers': 1}          # noqa: E731
    b = Budget(screens=3, llm_tokens=10_000_000, wall_seconds=1e9, gpu_hours=1e6, scoring_reserve_screens=0.0)
    agent, *_ = _agent(tmp_path, loop, budget=b)
    tr = agent.run()
    assert tr['stop_reason'] == 'budget:screens' and len(tr['screens']) == 3
    agent, *_ = _agent(tmp_path, lambda role, s, p: {'tool': 'bash', 'cmd': 'true'},
                       budget=Budget(screens=3, llm_tokens=5_000, wall_seconds=1e9, gpu_hours=1e6, scoring_reserve_screens=0.0))
    assert agent.run()['stop_reason'] == 'budget:llm_tokens'
    agent, *_ = _agent(tmp_path, lambda role, s, p: 'garbage', max_model_errors=3)
    assert agent.run()['stop_reason'] == 'model_errors'


def test_history_is_cut_from_the_middle(tmp_path):
    agent, *_ = _agent(tmp_path, [], context_tokens=4500, keep_last=5)
    for i in range(60):
        agent.turns.append({'turn': i + 1, 'action': {'tool': 'bash', 'cmd': f'echo {i}'}, 'result': {'stdout': 'y' * 300}})
    text = agent.render()
    assert text.startswith('# TASK\nTASK TEXT') and 'earlier turns omitted' in text
    kept = [int(n) for n in re.findall(r'## turn (\d+)', text)]
    assert kept and kept[-1] == 60 and len(kept) <= 5


def test_shell_executor_detached_runs(tmp_path):
    clock = VirtualClock()

    def launcher(ops, script, log):
        text = Path(script).read_text(encoding='utf-8')
        base = re.search(r"mkdir (\S+)\.started", text).group(1).strip("'")
        Path(ops.local(base + '.out')).write_bytes(b'hello\n')
        Path(ops.local(base + '.err')).write_bytes(b'')
        Path(ops.local(base + '.rc')).write_bytes(b'0\n')
        return {'pid': 1}
    ops = LocalNodeOps(tmp_path, launcher=launcher)
    sh = ShellExecutor(ops, home='/data/infrarsi/ext/runs/x/agent/home', cwd='/data/infrarsi/ext/runs/x/agent/home/repo',
                       work_dir='/data/infrarsi/ext/runs/x/agent/tool', token='isx-cnat-0123456789', path_dirs=('/env/bin',),
                       gpus=(4,), clock=clock, sleep=clock.advance)
    r = sh.bash('git status', 60)
    assert r['exit_code'] == 0 and r['stdout'] == 'hello\n' and ops.killed_tokens == ['isx-cnat-0123456789']
    launches = [kw for f, kw in ops.calls if f == 'launch']                    # PROPOSED_CHANGES_X3 #4: the wrapper carries the token
    assert launches[0]['env'] == {'INFRARSI_TOKEN': 'isx-cnat-0123456789'}
    script = (tmp_path / 'data/infrarsi/ext/runs/x/agent/tool/call_0001.sh').read_text(encoding='utf-8')
    assert 'setpriv --reuid=65534' in script and 'CUDA_VISIBLE_DEVICES=4' in script and 'timeout --kill-after=10 60' in script
    assert 'INFRARSI_TOKEN=isx-cnat-0123456789' in script and '.rc.tmp' in script
    r = sh.python('print(2)', 30)
    assert (tmp_path / 'data/infrarsi/ext/runs/x/agent/tool_code/py_0002.py').read_text() == 'print(2)'   # root-owned, not in the home
    lost = ShellExecutor(LocalNodeOps(tmp_path, launcher=lambda *a: {'pid': 2}), home='/h', cwd='/h', work_dir='/data/infrarsi/w',
                         token='isx-cnat-zzzzzzzzzz', clock=clock, sleep=clock.advance)
    r = lost.bash('sleep 100', 10)                                      # no exit code ever appears: reported as a timeout
    assert r['exit_code'] is None and r['timed_out'] is True


def test_truncate():
    assert truncate('abc') == 'abc' and len(truncate('z' * 50000)) < 17000


def test_shell_executor_runs_under_the_arms_uid(tmp_path):
    """O follow-up: co-located native arms get distinct numeric uids; python() code is readable by that arm's group only."""
    clock = VirtualClock()

    def launcher(ops, script, log):
        base = re.search(r"mkdir (\S+)\.started", Path(script).read_text(encoding='utf-8')).group(1).strip("'")
        Path(ops.local(base + '.rc')).write_bytes(b'0\n')
        return {'pid': 1}
    ops = LocalNodeOps(tmp_path, launcher=launcher)
    sh = ShellExecutor(ops, home='/data/infrarsi/ext/runs/y/agent/home', cwd='/data/infrarsi/ext/runs/y/agent/home/repo',
                       work_dir='/data/infrarsi/ext/runs/y/agent/tool', code_dir='/data/infrarsi/ext/runs/y/agent/code',
                       token='isx-cnat-uid64002xx', uid=64002, clock=clock, sleep=clock.advance)
    sh.python('print(1)', 30)
    script = (tmp_path / 'data/infrarsi/ext/runs/y/agent/tool/call_0001.sh').read_text(encoding='utf-8')
    assert '--reuid=64002 --regid=64002' in script and 'USER=isx64002' in script
    mk = [kw for f, kw in ops.calls if f == 'mkdirs' and kw['paths'] == ['/data/infrarsi/ext/runs/y/agent/code']]
    wr = [kw for f, kw in ops.calls if f == 'write_files' and '/data/infrarsi/ext/runs/y/agent/code/py_0001.py' in kw['files']]
    assert mk[0]['mode'] == 0o750 and mk[0]['gid'] == 64002 and mk[0].get('uid') is None
    assert wr[0]['mode'] == 0o640 and wr[0]['gid'] == 64002
