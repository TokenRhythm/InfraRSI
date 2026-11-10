"""GPU co-tenancy snapshots and link detection (textbook designs P1-3a / P1-3b1, 2026-10-24): nvidia-smi on the job's
GPUs right before each attempt and after it on every host (another job on the GPUs contaminated H2b arm B), ECC errors
kept apart, and the job's interconnect from node 0 (/dev/infiniband plus the NCCL settings of the job's own shell:
unicom's containers have mlx5 NICs but no /dev/infiniband = TCP)."""
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import yaml

from infrarsi.contracts import InfrastructureError
from infrarsi.project import Project, ProjectRunner
from infrarsi.project import executor as ex
from infrarsi.project.flow import apply_links, gpu_contention

TOY = Path(__file__).resolve().parents[3] / 'examples' / 'toy_cpu'
FREE = '0, GPU-a, 3, 0, Enabled, 0\n1, GPU-b, 3, 0, Enabled, 0\n__apps__\n'
BUSY = '0, GPU-a, 40960, 77, Enabled, 0\n1, GPU-b, 3, 0, Enabled, 0\n__apps__\nGPU-a, 4242, 40950\n'
OTHER = '0, GPU-a, 3, 0, Enabled, 0\n1, GPU-b, 40960, 90, Enabled, 0\n__apps__\nGPU-b, 77, 40950\n'
ECC = '0, GPU-a, 3, 0, Enabled, 2\n1, GPU-b, 3, 0, Disabled, 0\n__apps__\n'
AFTERGLOW = '0, GPU-a, 3, 64, Enabled, 0\n1, GPU-b, 3, 0, Enabled, 0\n__apps__\n'   # our last second, no process


def test_parse_smi_contention_and_ecc():
    assert ex.parse_smi('') is None and ex.parse_smi('__apps__\n') is None
    busy = ex.parse_smi(BUSY)
    assert busy['apps'] == 1 and busy['apps_mib'] == 40950 and busy['gpus'][0]['util'] == 77
    assert ex.gpu_contention([busy, ex.parse_smi(FREE), None], 'before') == [
        'node0: 1 GPU(s) in use before the run (40.0 GiB, 1 process(es))']
    assert ex.gpu_contention([ex.parse_smi(OTHER, {0})], 'before') == []            # GPU 1 is not the job's
    assert ex.parse_smi(OTHER, {0})['apps'] == 0 and ex.parse_smi(OTHER, {1})['apps'] == 1
    assert ex.ecc_errors([ex.parse_smi(FREE)], [ex.parse_smi(ECC)]) == ['node0 GPU 0: 2 uncorrected ECC error(s) during the run']
    assert ex.gpu_contention([ex.parse_smi(ECC)], 'after') == []                   # an ECC count is not another job
    na = ex.parse_smi('0, GPU-a, 3, [N/A], Enabled, [N/A]\n__apps__\n')
    assert na['gpus'][0]['ecc'] is None and ex.ecc_errors([na], [na]) == [] and ex.gpu_contention([na], 'after') == []


def test_parse_link():
    topo = '\tGPU0\tGPU1\tNIC0\nGPU0\t X \tNV18\tPXB\nGPU1\tNV18\t X \tPXB\n'
    ib = f'__uverbs__\n8\n__ll__\nInfiniBand\n__topo__\n{topo}__env__\n'
    assert ex.parse_link(ib, nodes=2) == {'intra': 'nvlink', 'inter': 'ib'}
    assert ex.parse_link(ib + 'NCCL_IB_DISABLE=1\n', nodes=2)['inter'] == 'tcp'      # the job's own environment
    assert ex.parse_link(ib + 'NCCL_NET=Socket\n', nodes=2)['inter'] == 'tcp'
    assert ex.parse_link('Welcome to the GPU cluster\n' + ib, nodes=2)['inter'] == 'ib'   # a profile banner first
    assert ex.parse_link(ib, nodes=1) == {'intra': 'nvlink'}
    no_dev = f'__uverbs__\n0\n__ll__\nEthernet\n__topo__\n{topo}__env__\n'   # mlx5 NICs listed, no /dev/infiniband
    assert ex.parse_link(no_dev, nodes=2)['inter'] == 'tcp'
    roce = '__uverbs__\n8\n__ll__\nEthernet\n__topo__\nGPU0\t X \tSYS\n__env__\n'
    assert ex.parse_link(roce, nodes=2) == {'intra': 'pcie', 'inter': 'roce'}
    assert ex.parse_link('__uverbs__\n8\n__ll__\n__topo__\n  Legend: NV#  = Connection traversing a bonded set of # NVLinks\n',
                         nodes=1) == {}                                       # the legend line is not a link
    assert ex.parse_link(None, nodes=2) == {} and ex.parse_link('', nodes=2) == {}


class Shell:
    """An executor stand-in for the command text (ssh-style quoting)."""
    kind, prefix = 'ssh', ''
    quote = staticmethod(lambda s: s)

    @staticmethod
    def python(configured):
        return 'python3'


def test_the_job_command_is_left_as_it_is(tmp_path, monkeypatch):
    proj = _toy(tmp_path, monkeypatch)
    proj.raw['launch']['setup'] = 'cd sub && source {config}'
    r = ProjectRunner(proj, out_dir=tmp_path / 'att', executor=Shell(), log=lambda m: None)
    cmd = r.command(proj.reference_config(), cfg_rel='c.yaml', steps=6, run_dir='/r')
    assert cmd.startswith('cd sub && source c.yaml && ') and '__uverbs__' not in cmd
    assert r.setup_line(cfg_rel='c.yaml', steps=6, run_dir='/r') == 'cd sub && source c.yaml'


class SmiExecutor(ex.LocalExecutor):
    """The local executor with canned nvidia-smi answers (popped in order; FREE when they run out)."""

    def __init__(self, answers):
        super().__init__()
        self.answers = list(answers)

    def on_hosts(self, cmd, timeout_s=20, login=False):
        return [self.answers.pop(0) if self.answers else FREE]


def _toy(tmp_path, monkeypatch):
    monkeypatch.setenv('TOY_TIME_SCALE', '0.004')
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)   # the job's GPUs: the default first ones
    d = tmp_path / 'toy'
    shutil.copytree(TOY, d, ignore=shutil.ignore_patterns('.infrarsi', '__pycache__'))
    doc = yaml.safe_load((d / 'infrarsi.yaml').read_text(encoding='utf-8'))
    doc['workload'] = {'steps': 6, 'window': [3, 6], 'invariants': {'global_batch': 'micro_batch_size * grad_accum'}}
    doc['checkpoint']['save_at'] = 3                              # inside the shortened run (the manifest checks it)
    (d / 'infrarsi.yaml').write_text(yaml.safe_dump(doc), encoding='utf-8')
    return Project.load(d)


def _run(proj, tmp_path, answers, logs=None, cid='c0'):
    runner = ProjectRunner(proj, out_dir=tmp_path / 'att', executor=SmiExecutor(answers),
                           log=(logs.append if logs is not None else lambda m: None))
    return runner, runner.run(None, proj.reference_config(), None, 'reference', 'r', candidate_id=cid)


def test_runner_records_other_jobs_on_the_gpus(tmp_path, monkeypatch):
    proj = _toy(tmp_path, monkeypatch)
    logs = []
    runner, rec = _run(proj, tmp_path, [BUSY, FREE], logs)
    gc = rec['evidence']['gpu_check']
    assert gc['contention'] == ['node0: 1 GPU(s) in use before the run (40.0 GiB, 1 process(es))'] and gc['ecc'] == []
    assert any('GPUs not free' in m for m in logs)
    assert gpu_contention(runner) == [{'candidate': 'c0', 'purpose': 'reference', 'warnings': gc['contention'], 'ecc': []}]
    clean = runner.run(None, proj.reference_config(), None, 'screen', 'r', candidate_id='c1')
    assert clean['evidence']['gpu_check']['contention'] == [] and len(gpu_contention(runner)) == 1


def test_after_only_busy_reading_is_looked_at_again(tmp_path, monkeypatch):
    proj = _toy(tmp_path, monkeypatch)
    monkeypatch.setattr('infrarsi.project.runner.time.sleep', lambda s: None)
    _, rec = _run(proj, tmp_path, [FREE, AFTERGLOW, FREE])          # the second look is free: our own afterglow
    assert rec['evidence']['gpu_check']['contention'] == []
    _, rec = _run(proj, tmp_path, [FREE, BUSY, BUSY], cid='c2')       # still busy: another job arrived
    assert rec['evidence']['gpu_check']['contention'] == ['node0: 1 GPU(s) in use after the run (40.0 GiB, 1 process(es))']


def test_ecc_is_not_reported_as_another_job_and_a_missing_first_look_keeps_the_last(tmp_path, monkeypatch):
    proj = _toy(tmp_path, monkeypatch)
    monkeypatch.setattr('infrarsi.project.runner.time.sleep', lambda s: None)
    runner, rec = _run(proj, tmp_path, [FREE, ECC])
    assert rec['evidence']['gpu_check']['ecc'] and not rec['evidence']['gpu_check']['contention']
    assert gpu_contention(runner)[0]['warnings'] == [] and gpu_contention(runner)[0]['ecc']
    _, rec = _run(proj, tmp_path, ['', BUSY, BUSY], cid='c3')        # nvidia-smi did not answer before: after still counts
    assert rec['evidence']['gpu_check']['contention'] == ['node0: 1 GPU(s) in use after the run (40.0 GiB, 1 process(es))']
    glow_ecc = AFTERGLOW.replace('Enabled, 0\n1', 'Enabled, 2\n1', 1)
    _, rec = _run(proj, tmp_path, [FREE, glow_ecc, ''], cid='c4')    # the second look fails: the ECC count stays
    assert rec['evidence']['gpu_check']['ecc'] == ['node0 GPU 0: 2 uncorrected ECC error(s) during the run']


class StopShell(SmiExecutor):
    """Each run ends stopped by the runner (an abort, a probe's planned stop)."""

    def run(self, cmd, *, cwd, env, timeout_s, log_path, stop=None):
        return {'exit_code': -15, 'timed_out': False, 'stopped': True, 'wall_s': 0.1, 'lines': []}


def test_a_stopped_runs_own_processes_are_waited_for_up_to_a_bound(tmp_path, monkeypatch):
    proj = _toy(tmp_path, monkeypatch)
    monkeypatch.setattr('infrarsi.project.runner.time.sleep', lambda s: None)
    logs = []
    runner = ProjectRunner(proj, out_dir=tmp_path / 'att', executor=StopShell([FREE, BUSY, BUSY, FREE]), log=logs.append)
    rec = runner.run(None, proj.reference_config(), None, 'screen', 'r', candidate_id='c1')
    assert rec['evidence']['gpu_check']['contention'] == [] and not runner.executor.answers   # two more looks
    assert any('the stopped run: its GPUs are not free yet' in m for m in logs) and any('were free after' in m for m in logs)
    runner = ProjectRunner(proj, out_dir=tmp_path / 'att', executor=StopShell([FREE, BUSY, BUSY]), log=lambda m: None)
    runner.own_wait_s = 0.0                                          # still held at the bound: recorded as before
    rec = runner.run(None, proj.reference_config(), None, 'screen', 'r', candidate_id='c2')
    assert rec['evidence']['gpu_check']['contention'] == ['node0: 1 GPU(s) in use after the run (40.0 GiB, 1 process(es))']


class TwoHosts(StopShell):
    """StopShell on two hosts: each answer is a pair of nvidia-smi outputs ('' = the host did not answer)."""

    def on_hosts(self, cmd, timeout_s=20, login=False):
        return list(self.answers.pop(0)) if self.answers else [FREE, FREE]


def test_a_look_that_fails_is_no_free_gpu(tmp_path, monkeypatch):
    proj = _toy(tmp_path, monkeypatch)
    clock = {'t': 0.0}
    monkeypatch.setattr('infrarsi.project.runner.time.sleep', lambda s: clock.__setitem__('t', clock['t'] + s))
    monkeypatch.setattr('infrarsi.project.runner.time.monotonic', lambda: clock['t'])
    runner = ProjectRunner(proj, out_dir=tmp_path / 'att', executor=StopShell([FREE, BUSY, '', FREE]), log=lambda m: None)
    rec = runner.run(None, proj.reference_config(), None, 'screen', 'r', candidate_id='c1')
    assert rec['evidence']['gpu_check']['contention'] == [] and not runner.executor.answers   # looked again after the miss
    runner = ProjectRunner(proj, out_dir=tmp_path / 'att', executor=StopShell([FREE, BUSY, '', '']), log=lambda m: None)
    runner.own_wait_s = 15.0                                         # the bound passes on failed looks: still held
    rec = runner.run(None, proj.reference_config(), None, 'screen', 'r', candidate_id='c2')
    assert rec['evidence']['gpu_check']['contention'] == ['node0: 1 GPU(s) in use after the run (40.0 GiB, 1 process(es))']
    runner = ProjectRunner(proj, out_dir=tmp_path / 'att', log=lambda m: None,
                           executor=TwoHosts([(FREE, FREE), (BUSY, FREE), ('', FREE), (FREE, FREE)]))
    rec = runner.run(None, proj.reference_config(), None, 'screen', 'r', candidate_id='c3')
    assert rec['evidence']['gpu_check']['contention'] == [] and not runner.executor.answers   # node 0's miss kept it busy


class CopyFails(SmiExecutor):
    """The first copy of the tree fails (the transport), the run itself never started."""

    def __init__(self, answers):
        super().__init__(answers)
        self.fail = 1

    def place(self, tree, rel):
        if self.fail:
            self.fail -= 1
            raise InfrastructureError('copy failed')
        return super().place(tree, rel)


def test_a_retry_waits_for_the_attempt_the_transport_cut_off(tmp_path, monkeypatch):
    proj = _toy(tmp_path, monkeypatch)
    monkeypatch.setattr('infrarsi.project.runner.time.sleep', lambda s: None)
    logs = []
    runner = ProjectRunner(proj, out_dir=tmp_path / 'att', executor=LinkShell([FREE, BUSY, BUSY, FREE, FREE], fail=1),
                           log=logs.append)
    runner.infra_retry_waits = (0.0,)
    rec = runner.run(None, proj.reference_config(), None, 'screen', 'r', candidate_id='c3')
    assert rec['evidence']['gpu_check']['contention'] == [] and not runner.executor.answers
    assert any('the cut-off attempt: its GPUs are not free yet' in m for m in logs)
    _, rec = _run(proj, tmp_path, [BUSY, FREE], cid='c4')           # a first attempt does not wait: another job's
    assert rec['evidence']['gpu_check']['contention'] == ['node0: 1 GPU(s) in use before the run (40.0 GiB, 1 process(es))']
    logs = []
    runner = ProjectRunner(proj, out_dir=tmp_path / 'att', executor=CopyFails([BUSY, FREE]), log=logs.append)
    runner.infra_retry_waits = (0.0,)                               # a failed copy ran nothing: no wait either
    rec = runner.run(None, proj.reference_config(), None, 'screen', 'r', candidate_id='c5')
    assert rec['evidence']['gpu_check']['contention'] == ['node0: 1 GPU(s) in use before the run (40.0 GiB, 1 process(es))']
    assert not any('not free yet' in m for m in logs)


def test_no_snapshot_without_nvidia_smi(tmp_path, monkeypatch):
    proj = _toy(tmp_path, monkeypatch)
    _, rec = _run(proj, tmp_path, ['', ''])
    assert 'gpu_check' not in rec['evidence']


def test_devices_follow_cuda_visible_devices(tmp_path, monkeypatch):
    proj = _toy(tmp_path, monkeypatch)
    r = ProjectRunner(proj, out_dir=tmp_path / 'att', executor=SmiExecutor([]), log=lambda m: None)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    assert r.devices() == set(range(proj.gpus))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4,5')
    assert r.devices() == {4, 5}
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-b')                    # a uuid (prefix) selects that GPU
    assert r.devices() == {'GPU-b'} and ex.parse_smi(OTHER, r.devices())['apps'] == 1
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'MIG-1234')                 # a MIG slice: no judgement of the parent GPU
    assert r.devices() is None and r.gpu_state() is None
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')                         # no GPU at all
    assert r.devices() == set() and r.gpu_state() is None


def test_apply_links_fills_an_unset_interconnect_and_keeps_a_set_one(tmp_path):
    p = Project({'name': 'x', 'launch': {'command': 'true'}}, tmp_path / 'infrarsi.yaml')
    logs = []
    found = apply_links(p, SimpleNamespace(link_facts=lambda: {'intra': 'nvlink', 'inter': 'tcp'}), logs.append)
    assert found == {'intra': 'nvlink', 'inter': 'tcp'} and p.raw['hardware']['interconnect'] == found
    q = Project({'name': 'x', 'launch': {'command': 'true'}, 'hardware': {'interconnect': 'ib'}}, tmp_path / 'infrarsi.yaml')
    apply_links(q, SimpleNamespace(link_facts=lambda: {'inter': 'tcp'}), logs.append)
    assert q.raw['hardware']['interconnect'] == 'ib' and 'the manifest is used' in logs[-1]


class LinkShell(SmiExecutor):
    """Runs nothing (the first `fail` runs: a cut-off connection); answers the fabric probe as node 0's shell would (an
    IB device, the NCCL variables of its environment), after `delay` seconds, and records how it was asked."""

    def __init__(self, answers, fail=0, delay=0.0, error=None):
        super().__init__(answers)
        self.fail, self.delay, self.error = fail, delay, error

    def run(self, cmd, *, cwd, env, timeout_s, log_path, stop=None):
        if self.fail:
            self.fail -= 1
            raise InfrastructureError('connection closed')
        if self.error:
            raise self.error
        return {'exit_code': 0, 'timed_out': False, 'stopped': False, 'wall_s': 0.1, 'lines': []}

    def probe(self, cmd, *, cwd, env, timeout_s=30):
        self.probes = getattr(self, 'probes', []) + [(cmd, cwd, dict(env))]
        time.sleep(self.delay)
        nccl = ''.join(f'{k}={v}\n' for k, v in env.items() if k.startswith('NCCL_'))
        return f'Welcome\n__uverbs__\n2\n__ll__\nLink layer: InfiniBand\n__topo__\nGPU0\t X \tNV18\n__env__\n{nccl}'


def test_links_come_from_a_probe_in_the_reference_runs_context(tmp_path, monkeypatch):
    proj = _toy(tmp_path, monkeypatch)
    proj.raw['hardware']['nodes'] = 2
    proj.raw['launch']['setup'] = 'source {config}'
    proj.raw['launch']['env'] = {'NCCL_SOCKET_IFNAME': 'eth0'}
    sh = LinkShell([])
    runner = ProjectRunner(proj, out_dir=tmp_path / 'att', executor=sh, log=lambda m: None)
    assert runner.link_facts() == {}
    runner.run(None, proj.reference_config(), None, 'screen', 'r', candidate_id='c1')
    assert runner.link_facts() == {} and not getattr(sh, 'probes', [])   # only a reference / doctor run probes
    runner.run(None, proj.reference_config(), None, 'reference', 'r', candidate_id='c0')
    assert runner.link_facts() == {'intra': 'nvlink', 'inter': 'ib'}
    cmd, cwd, env = sh.probes[-1]
    assert cmd == f'( {ex.LINK_QUERY} )'                     # the setup is not replayed: it may change the tree
    assert env['NCCL_SOCKET_IFNAME'] == 'eth0' and Path(cwd).name == 'tree'


def test_nccl_variables_the_setup_assigns_count_over_the_env(tmp_path, monkeypatch):
    assert ex.setup_nccl('source a.sh && export NCCL_IB_DISABLE="1"; NCCL_NET=Socket python x') == \
        {'NCCL_IB_DISABLE': '1', 'NCCL_NET': 'Socket'}
    assert ex.setup_nccl('export NCCL_IB_DISABLE=1 && export NCCL_IB_DISABLE=0') == {'NCCL_IB_DISABLE': '0'}
    assert ex.setup_nccl(None) == {} and ex.setup_nccl('export NCCL_DEBUG=INFO') == {}
    proj = _toy(tmp_path, monkeypatch)
    proj.raw['hardware']['nodes'] = 2
    proj.raw['launch']['setup'] = 'export NCCL_IB_DISABLE=1'
    proj.raw['launch']['env'] = {'NCCL_IB_DISABLE': '0'}
    runner = ProjectRunner(proj, out_dir=tmp_path / 'att', executor=LinkShell([]), log=lambda m: None)
    runner.run(None, proj.reference_config(), None, 'reference', 'r', candidate_id='c0')
    assert runner.link_facts()['inter'] == 'tcp'


def test_the_probe_is_not_charged_to_the_run(tmp_path, monkeypatch):
    proj = _toy(tmp_path, monkeypatch)
    runner = ProjectRunner(proj, out_dir=tmp_path / 'att', executor=LinkShell([], fail=1, delay=0.4), log=lambda m: None)
    runner.infra_retry_waits = (0.0,)
    rec = runner.run(None, proj.reference_config(), None, 'reference', 'r', candidate_id='c0')
    assert rec['phases']['link_probe'] >= 0.8                 # two attempts, one probe each
    assert rec['cost']['gpu_seconds'] < proj.world * (0.1 + 0.3) and not rec['cost']['accounting_complete']
    runner = ProjectRunner(proj, out_dir=tmp_path / 'att', log=lambda m: None,
                           executor=LinkShell([], delay=0.4, error=OSError(28, 'No space left on device')))
    rec = runner.run(None, proj.reference_config(), None, 'reference', 'r', candidate_id='c0')
    assert not rec['valid'] and rec['cost']['gpu_seconds'] < proj.world * 0.3   # a runner error: the probe is not its
