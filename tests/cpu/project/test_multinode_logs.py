"""Multi-node fixes from the 2-node WINH doctor (2026-10-24): `{master_addr}` can name node 0 the way the other nodes
reach it (an ssh alias does not resolve there), and the update lines may come from another node than 0 (Megatron
prints them on the last rank, which lives on the last node)."""
import subprocess
import sys

from infrarsi.project import executor as ex

STEP = '1700000000.000001' + chr(9) + ' [x] iteration        {i}/       3 | elapsed time per iteration (ms): 1374.5 | lm loss: 1.7'


def _two_nodes(monkeypatch, outputs):
    seen, progs = [], iter(outputs)

    def fake_popen(argv, **kw):                         # node i's ssh client: its stamped lines, then the end marker
        seen.append(argv)
        lines = next(progs)
        prog = ''.join(f'print({x!r}); ' for x in lines) + f'print("{ex.END_MARK} 0")'
        return subprocess.Popen([sys.executable, '-c', prog], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    monkeypatch.setattr(ex, '_popen', fake_popen)
    return seen


def test_master_addr_names_node_0_for_the_other_nodes(tmp_path, monkeypatch):
    seen = _two_nodes(monkeypatch, [[], []])
    ex.SSHExecutor(['winh2', 'winh3'], '/r', master_addr='10.0.0.2').run(
        'torchrun --master_addr {master_addr} --node_rank {node_rank}', cwd='/r/t', env={}, timeout_s=60,
        log_path=tmp_path / 'a.log')
    assert all('--master_addr 10.0.0.2' in a[-1] for a in seen) and '--node_rank 1' in seen[1][-1]
    seen = _two_nodes(monkeypatch, [[], []])
    ex.SSHExecutor(['u@winh2:22', 'winh3'], '/r').run('x --master_addr {master_addr}', cwd='/r/t', env={}, timeout_s=60,
                                                       log_path=tmp_path / 'b.log')
    assert '--master_addr winh2' in seen[0][-1]                                  # unset: host 0's ssh name
    assert ex.make({'where': 'ssh', 'hosts': ['a', 'b'], 'master_addr': '10.0.0.9'}).master_addr == '10.0.0.9'


def test_the_update_lines_of_the_last_node_are_read(tmp_path, monkeypatch):
    from infrarsi.project import ProjectRunner
    from test_gpu_link_checks import SmiExecutor, _toy

    class Ssh(SmiExecutor):
        kind = 'ssh'

        def probe(self, cmd, *, cwd, env, timeout_s=30):
            return None

        def run(self, cmd, *, cwd, env, timeout_s, log_path, stop=None, python='python3'):
            return {'exit_code': 0, 'timed_out': False, 'stopped': False, 'wall_s': 5.0, 'lines': [(1.0, 'node 0: setup')],
                    'node_lines': [[(10.0 + 1.4 * i, f'step {i} | loss {2.0 - 0.01 * i}') for i in range(1, 7)]]}
    proj = _toy(tmp_path, monkeypatch)
    proj.raw['hardware']['nodes'] = 2
    rec = ProjectRunner(proj, out_dir=tmp_path / 'att', executor=Ssh([]), log=lambda m: None).run(
        None, proj.reference_config(), None, 'screen', 'r', candidate_id='c1')
    assert rec['evidence']['steps_completed'] == 6 and rec['step_log'].endswith('node1.log')
    assert rec['log'].endswith('train.log')                 # node 0 keeps the launch output (the argument dump)
    _two_nodes(monkeypatch, [['node 0 line'], [STEP.format(i=1)]])
    out = ex.SSHExecutor(['h1', 'h2'], '/r').run('train', cwd='/r/t', env={}, timeout_s=60, log_path=tmp_path / 'c.log')
    assert [t for _, t in out['node_lines'][0]] == [STEP.format(i=1).split(chr(9), 1)[1]]   # node 1, remote-stamped


def test_a_failed_reference_report_names_the_update_log_too():
    from infrarsi.project.report import markdown
    r = {'run_id': 'x', 'project': {'name': 'p', 'gpus': 8, 'steps': 30, 'window': [11, 30]}, 'tolerance': 0.03,
         'outcome': 'reference_failed', 'feedback': 'completes: 0/30', 'reference': {'log': '/r/train.log',
                                                                                    'step_log': '/r/node1.log'}}
    text = markdown(r)
    assert '`/r/train.log`' in text and '`/r/node1.log`' in text
