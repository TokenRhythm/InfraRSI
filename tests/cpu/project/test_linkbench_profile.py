"""The runner's two diagnostics (textbook designs A10 / A11, 2026-10-24): a multi-node job's all-reduce bandwidth after
its reference run (launched like the job, in its environment; not charged to it), and the one-off phase profile (the
hook staged with the tree, first on PYTHONPATH; rank 0's phases.json read back into the evidence)."""
import json
from pathlib import Path

from infrarsi.project import ProjectRunner
from infrarsi.project import executor as ex
from infrarsi.project import linkbench
from infrarsi.project.runner import PROFHOOK

from test_gpu_link_checks import FREE, SmiExecutor, _toy



class SshLike(SmiExecutor):
    """An ssh-shaped executor that records each run; the link benchmark answers with its line."""
    kind = 'ssh'

    def __init__(self):
        super().__init__([])
        self.runs, self.places = [], []

    def place(self, tree, rel, timeout_s=None):
        self.places.append((rel, timeout_s))
        return super().place(tree, rel)

    def probe(self, cmd, *, cwd, env, timeout_s=30):
        return None

    def run(self, cmd, *, cwd, env, timeout_s, log_path, stop=None, python='python3'):
        self.runs.append((cmd, cwd, dict(env)))
        line = f'{linkbench.MARK} ' + json.dumps({'busbw_gbs': 41.5, 'algbw_gbs': 22.1, 'world': 16})
        if getattr(self, 'busy', False):            # a rank found another process on its GPU and left
            line = f'{linkbench.BUSY} rank 9: GPU-abc, 4242'
        return {'exit_code': 0, 'timed_out': False, 'stopped': False, 'wall_s': 0.1,
                'lines': [(1.0, line)] if 'torch.distributed.run' in cmd else []}


def test_linkbench_parses_its_line():
    assert linkbench.parse(['noise', f'{linkbench.MARK} {{"busbw_gbs": 12.3, "world": 16}}']) == {'busbw_gbs': 12.3, 'world': 16}
    assert linkbench.parse([f'{linkbench.MARK} {{"busbw_gbs": 0}}', 'x']) is None and linkbench.parse([]) is None


def test_a_multi_node_reference_measures_its_all_reduce_bandwidth(tmp_path, monkeypatch):
    proj = _toy(tmp_path, monkeypatch)
    proj.raw['hardware']['nodes'] = 2
    proj.raw['launch']['setup'] = 'export NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1'
    proj.raw['launch']['env'] = {'NCCL_DEBUG': 'WARN'}
    sh = SshLike()
    runner = ProjectRunner(proj, out_dir=tmp_path / 'att', executor=sh, log=lambda m: None)
    runner.run(None, proj.reference_config(), None, 'screen', 'r', candidate_id='c1')
    assert len(sh.runs) == 1 and 'bandwidth_gbs' not in runner.link_facts()       # only a reference / doctor run
    rec = runner.run(None, proj.reference_config(), None, 'reference', 'r', candidate_id='c0')
    cmd, cwd, env = sh.runs[-1]
    assert '--nnodes {nnodes} --node_rank {node_rank} --master_addr {master_addr}' in cmd and Path(cwd).name == 'linkbench'
    assert env['NCCL_SOCKET_IFNAME'] == 'eth0' and env['NCCL_IB_DISABLE'] == '1' and env['NCCL_DEBUG'] == 'WARN'
    assert runner.link_facts()['bandwidth_gbs'] == 41.5 and 'link_bench' in rec['phases']
    assert rec['cost']['gpu_seconds'] < proj.world * 0.2                       # the job's own run only
    assert sh.places[-1] == ('r/' + Path(rec['log']).parent.name + '/linkbench', 120)    # a bounded copy
    runner.run(None, proj.reference_config(), None, 'reference', 'r', candidate_id='c0', bench=False)
    assert 'torch.distributed.run' not in sh.runs[-1][0]


def test_the_benchmark_takes_the_jobs_gpus_or_none(tmp_path, monkeypatch):
    proj = _toy(tmp_path, monkeypatch)
    proj.raw['hardware']['nodes'] = 2
    proj.raw['launch']['setup'] = 'export CUDA_VISIBLE_DEVICES=6,7'
    sh = SshLike()
    ProjectRunner(proj, out_dir=tmp_path / 'att', executor=sh, log=lambda m: None).run(
        None, proj.reference_config(), None, 'reference', 'r', candidate_id='c0')
    assert sh.runs[-1][2]['CUDA_VISIBLE_DEVICES'] == '6,7' and 'torch.distributed.run' in sh.runs[-1][0]
    proj.raw['launch']['setup'] = 'export CUDA_VISIBLE_DEVICES=$FREE_GPUS'
    logs, sh = [], SshLike()
    ProjectRunner(proj, out_dir=tmp_path / 'att2', executor=sh, log=logs.append).run(
        None, proj.reference_config(), None, 'reference', 'r', candidate_id='c0')
    assert not any('torch.distributed.run' in c for c, _, _ in sh.runs) and any('benchmark skipped' in m for m in logs)


def test_the_profile_run_stages_its_hook_and_reads_rank_0s_phases(tmp_path, monkeypatch):
    proj = _toy(tmp_path, monkeypatch)
    seen = {}

    class Local(SmiExecutor):
        def run(self, cmd, *, cwd, env, timeout_s, log_path, stop=None):
            seen.update(cmd=cmd, cwd=cwd, env=dict(env), hook=(Path(cwd) / PROFHOOK / 'sitecustomize.py').is_file())
            out = Path(env['INFRARSI_PROFILE'])
            out.mkdir(parents=True)
            (out / 'phases.json').write_text(json.dumps({'unit': 'ms', 'mean_ms': {'compute': 20.0, 'update': 22.0}}))
            return {'exit_code': 0, 'timed_out': False, 'stopped': False, 'wall_s': 0.1, 'lines': []}

    runner = ProjectRunner(proj, out_dir=tmp_path / 'att', executor=Local([FREE]), log=lambda m: None, keep_trees=True)
    rec = runner.run(None, proj.reference_config(), None, 'profile', 'r', candidate_id='c0', steps=10, profile=True)
    assert seen['hook'] and seen['cmd'].startswith(f'export PYTHONPATH={ex.LocalExecutor.quote(seen["cwd"] + "/" + PROFHOOK)}')
    assert '"${PYTHONPATH:+:$PYTHONPATH}" && ' in seen['cmd'] and seen['env']['INFRARSI_PROFILE_WARMUP'] == '5'
    assert rec['evidence']['profile']['mean_ms']['compute'] == 20.0
    plain = runner.run(None, proj.reference_config(), None, 'screen', 'r', candidate_id='c1')
    assert 'profile' not in plain['evidence']


def test_the_benchmark_reads_the_last_assignment_and_a_busy_rank_gives_no_number(tmp_path, monkeypatch):
    assert ex.setup_devices('export CUDA_VISIBLE_DEVICES=0,1; export CUDA_VISIBLE_DEVICES=$FREE_GPUS') is False
    assert ex.setup_devices('export CUDA_VISIBLE_DEVICES=$X; export CUDA_VISIBLE_DEVICES=6,7') == {'CUDA_VISIBLE_DEVICES': '6,7'}
    assert ex.setup_devices('export CUDA_VISIBLE_DEVICES=6,7; unset CUDA_VISIBLE_DEVICES') is False
    proj = _toy(tmp_path, monkeypatch)
    proj.raw['hardware']['nodes'] = 2
    sh = SshLike()
    sh.busy = True
    logs = []
    runner = ProjectRunner(proj, out_dir=tmp_path / 'att', executor=sh, log=logs.append)
    runner.run(None, proj.reference_config(), None, 'reference', 'r', candidate_id='c0')
    assert any('torch.distributed.run' in c for c, _, _ in sh.runs) and 'bandwidth_gbs' not in runner.link_facts()
    assert any('link benchmark stopped' in m for m in logs)
