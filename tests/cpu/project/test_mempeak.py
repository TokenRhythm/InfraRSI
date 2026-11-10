"""D3 (the user's decision 2026-10-31): every run of every strategy records its peak GPU memory - the maximum over every
process of every node - as an extra receipt field, `memory_peak_measured_gib`; an observation only (the verifier, the
judge and the digest read `memory_peak_gib`, the log's, unchanged). The run hook (profhook/sitecustomize.py, first on the
job's PYTHONPATH, chaining the job's own sitecustomize) reads, in each process (PeakWatch), the peak reserved bytes of
every device from the allocator's nested statistics (host-side counters, no device synchronisation): a full run does no
periodic work during the job - no thread, one read at interpreter exit (the counter is the whole run's high-water mark;
a full run stopped by a signal has no peak: unknown, logged so); a probe or short run (INFRARSI_MEMPEAK_FAST, stopped
by a signal after a few updates) starts a sampler thread that reads every 2 s once CUDA is up, plus the exit read. The
runner takes the maximum of the files after the run. D7: rank 0 of the reference run also writes the modules it imported,
once, at exit (the library files D7 may read)."""
import ast
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from infrarsi.contracts import InfrastructureError
from infrarsi.project import Project, ProjectRunner
from infrarsi.project import executor as ex
from infrarsi.project.flow import make_verifier
from infrarsi.project.profhook import sitecustomize as hook
from infrarsi.project.runner import MEMPEAK, MODULES

TOY = Path(__file__).resolve().parents[3] / 'examples' / 'toy_cpu'
GIB = 2 ** 30


STAT_NAMES = ('allocated_bytes', 'reserved_bytes', 'active_bytes', 'inactive_split_bytes', 'requested_bytes',
              'allocation', 'segment', 'active', 'inactive_split', 'oversize_allocations', 'oversize_segments')


class FakeCuda:
    """The allocator statistics of a process: per device (current, peak) reserved bytes; the nested dict the C binding
    returns is built on every call (as the binding builds it), and the calls are counted per device."""

    def __init__(self, devices=8, used=(0,), peak_gib=6.0, initialized=True, fail=False):
        self.n, self.used, self.initialized, self.fail = devices, set(used), initialized, fail
        self.peak = {d: (peak_gib if d in self.used else 0.0) for d in range(devices)}
        self.calls: dict[int, int] = {}

    def stats(self, d):
        self.calls[d] = self.calls.get(d, 0) + 1
        if self.fail:
            raise RuntimeError('CUDA error: an illegal memory access was encountered')
        b = int(self.peak[d] * GIB)
        out = {name: {pool: {'current': 0, 'peak': 0, 'allocated': 0, 'freed': 0}
                      for pool in ('all', 'small_pool', 'large_pool')} for name in STAT_NAMES}
        out['reserved_bytes']['all'] = {'current': b, 'peak': b, 'allocated': b, 'freed': 0}
        out.update(num_alloc_retries=0, num_ooms=0, num_sync_all_streams=0, num_device_alloc=0, num_device_free=0,
                   max_split_size=-1)
        return out

    def torch(self):
        t = types.ModuleType('torch')
        t._C = SimpleNamespace(_cuda_memoryStats=self.stats)
        t.cuda = SimpleNamespace(is_initialized=lambda: self.initialized, device_count=lambda: self.n)
        return t


def sample(w: 'hook.PeakWatch', seconds: float) -> list[float]:
    """A sampler thread's wake-ups over `seconds` (the delays step() returns): the times of its reads."""
    t, times = 0.0, []
    while t < seconds:
        before = w.reads
        delay = w.step()
        if w.reads > before:
            times.append(t)
        t += delay
    return times


def test_d3_a_full_run_reads_every_device_once_at_exit_and_nothing_before(tmp_path):
    cuda = FakeCuda(devices=8, used=(3,), peak_gib=6.0)
    path, mods = tmp_path / 'h-1.txt', tmp_path / 'm.json'
    w = hook.PeakWatch(str(path), str(mods), torch_module=cuda.torch())
    assert w.fast is False and w.reads == 0 and w.module_dumps == 0 and not path.exists() and not cuda.calls
    w.final()                                                         # at exit: every device once, the module list once
    assert path.read_text(encoding='utf-8') == 'mempeak 6.0000\n' and w.reads == 1 and w.module_dumps == 1
    assert cuda.calls == {d: 1 for d in range(8)}
    assert json.loads(mods.read_text(encoding='utf-8'))[__name__].endswith('test_mempeak.py')
    w.final()                                                         # (a second exit hook: nothing more)
    assert w.reads == 1 and w.module_dumps == 1 and cuda.calls == {d: 1 for d in range(8)}


def test_d3_a_full_run_process_starts_no_thread_and_a_probe_process_the_sampler(tmp_path):
    """The hook as a job loads it (first on PYTHONPATH, in a child process; a stand-in torch on four GPUs): a full run's
    process starts no thread and writes its peak (every device's) and its modules at exit; a probe's or short run's
    process (INFRARSI_MEMPEAK_FAST=1) starts the sampler."""
    job = tmp_path / 'job.py'
    job.write_text('import sys, threading, types\n'
                   't = types.ModuleType("torch")\n'
                   'r = lambda d: {"reserved_bytes": {"all": {"current": 0, "peak": (d + 1) * 2 ** 30}}}\n'
                   't._C = types.SimpleNamespace(_cuda_memoryStats=r)\n'
                   't.cuda = types.SimpleNamespace(is_initialized=lambda: True, device_count=lambda: 4)\n'
                   'sys.modules["torch"] = t\n'
                   'print(sorted(x.name for x in threading.enumerate()))\n', encoding='utf-8')
    hookdir = str(Path(hook.__file__).resolve().parent)
    for fast in (False, True):
        peaks, mods = tmp_path / f'peaks-{fast}', tmp_path / f'mods-{fast}'
        env = {k: v for k, v in os.environ.items() if k not in ('INFRARSI_MEMPEAK_FAST', 'RANK', 'LOCAL_RANK',
                                                                'INFRARSI_PROFILE', 'PYTHONPATH')}
        env.update(PYTHONPATH=hookdir, INFRARSI_MEMPEAK=str(peaks), INFRARSI_MODULES=str(mods),
                   **({'INFRARSI_MEMPEAK_FAST': '1'} if fast else {}))
        out = subprocess.run([sys.executable, str(job)], env=env, capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        names = ast.literal_eval(out.stdout.strip().splitlines()[-1])
        assert ('infrarsi-mempeak' in names) is fast and (names == ['MainThread']) is not fast, (fast, names)
        files = list(peaks.glob('*.txt'))
        assert len(files) == 1 and files[0].read_text(encoding='utf-8') == 'mempeak 4.0000\n'   # the exit read
        written = [json.loads(f.read_text(encoding='utf-8')) for f in mods.glob('*.json')]
        assert len(written) == 1 and 'threading' in written[0]


def test_d3_a_probe_samples_every_two_seconds_and_keeps_its_peak_when_killed(tmp_path):
    cuda = FakeCuda(devices=8, used=(0,), peak_gib=2.0)
    path = tmp_path / 'p.txt'
    w = hook.PeakWatch(str(path), fast=True, torch_module=cuda.torch())
    reads = sample(w, 3)                                               # loading
    cuda.peak[0] = 41.5                                                # the first update's forward and backward
    reads += sample(w, 5)                                              # ... and the process is killed (no exit hook)
    assert path.read_text(encoding='utf-8') == 'mempeak 41.5000\n'
    assert len(reads) == 5 and all(cuda.calls[d] == 5 for d in range(8))   # every 2 s, every device each time


def test_r4_a_probe_after_a_long_startup_keeps_its_training_peak(tmp_path):
    """Codex HLS5 r4: CUDA up at once, a long load (118 s), the first updates raise the peak from 2 to 41.5 GiB at 120 s,
    the probe stopped by a signal at 124 s (no exit hook): a probe reads every PEAK_FAST seconds throughout."""
    cuda = FakeCuda(devices=8, used=(0,), peak_gib=2.0)
    path = tmp_path / 'p.txt'
    w = hook.PeakWatch(str(path), fast=True, torch_module=cuda.torch())
    sample(w, 120)
    cuda.peak[0] = 41.5
    sample(w, 4)                                                       # killed here: final() never runs
    assert path.read_text(encoding='utf-8') == 'mempeak 41.5000\n'
    full = hook.PeakWatch(str(tmp_path / 'f.txt'), torch_module=FakeCuda(peak_gib=2.0).torch())
    assert full.fast is False                                        # (a full run: no sampler, its exit read)


def test_r4_a_device_that_gets_memory_later_is_read(tmp_path):
    cuda = FakeCuda(devices=8, used=(0,), peak_gib=2.0)
    path = tmp_path / 'd.txt'
    w = hook.PeakWatch(str(path), fast=True, torch_module=cuda.torch())
    sample(w, 10)
    cuda.peak[1] = 41.5                                                # a library allocates on another GPU
    sample(w, hook.PEAK_FAST)                                          # a probe: the next read
    assert path.read_text(encoding='utf-8') == 'mempeak 41.5000\n'
    cuda = FakeCuda(devices=8, used=(0,), peak_gib=2.0)
    full = hook.PeakWatch(str(tmp_path / 'f.txt'), torch_module=cuda.torch())
    cuda.peak[5] = 30.0
    full.final()                                                       # a full run: its exit read sees every device
    assert (tmp_path / 'f.txt').read_text(encoding='utf-8') == 'mempeak 30.0000\n'


def test_d3_nothing_read_before_cuda_nor_from_a_failing_read(tmp_path):
    for cuda in (FakeCuda(initialized=False), FakeCuda(fail=True)):
        for fast in (True, False):
            path = tmp_path / f'x-{id(cuda)}-{fast}.txt'
            w = hook.PeakWatch(str(path), fast=fast, torch_module=cuda.torch())
            if fast:
                sample(w, 30)
            w.final()
            assert not path.exists()
    w = hook.PeakWatch(str(tmp_path / 'none.txt'), fast=True)          # no torch imported: the hook never imports it
    saved = sys.modules.pop('torch', None)
    try:
        assert w.step() == hook.PEAK_FAST and 'torch' not in sys.modules
        w.final()
        assert 'torch' not in sys.modules and not (tmp_path / 'none.txt').exists()
    finally:
        if saved is not None:
            sys.modules['torch'] = saved


def test_d3_the_cost_of_a_read_cpu_micro_benchmark():
    """The per-read cost of the read (every device, the nested statistics) against the first version's (every device,
    flattened - what torch.cuda.max_memory_reserved does), on this CPU with the statistics built as the binding builds
    them; and the reads a process makes during the job. The GPU's numbers differ; the ratio is the point."""
    import time

    def pairs(stats, prefix=''):
        out = []
        for k, v in stats.items():
            if isinstance(v, dict):
                out += pairs(v, f'{prefix}{k}.')
            else:
                out.append((f'{prefix}{k}', v))
        return out

    def flatten(stats):                                # torch.cuda.memory_stats' flattening, as it does it
        return dict(sorted(pairs(stats)))
    cuda = FakeCuda(devices=8, used=(5,))
    t = cuda.torch()
    w = hook.PeakWatch('', torch_module=t)
    n = 250
    a = time.perf_counter()
    for _ in range(n):
        w.read(t)
    new = (time.perf_counter() - a) / n
    a = time.perf_counter()
    for _ in range(n):
        max(flatten(cuda.stats(d))['reserved_bytes.all.peak'] for d in range(8))
    old = (time.perf_counter() - a) / n
    per_min_probe = 60 / hook.PEAK_FAST
    print(f'\nD3 per read of 8 devices: {1e6 * new:.0f} us now (nested) vs {1e6 * old:.0f} us before (flattened); a full '
          f'run: no thread, 0 reads during the job and 1 at exit; a probe or short run: {per_min_probe:.0f} reads/min '
          f'({1e3 * new * per_min_probe:.2f} ms of interpreter time per minute)')
    assert new < old / 4 and new < 2e-3


def test_r4_the_runner_marks_probes_and_short_runs(toy, tmp_path):
    class Rec(ex.LocalExecutor):
        def __init__(self):
            super().__init__()
            self.envs = []

        def run(self, cmd, *, cwd, env, timeout_s, log_path, stop=None):
            self.envs.append(dict(env))
            return super().run(cmd, cwd=cwd, env=env, timeout_s=timeout_s, log_path=log_path, stop=stop)
    r = ProjectRunner(toy, out_dir=tmp_path / 'runs')
    r.executor = Rec()
    t = toy.task()
    r.run(t, toy.reference_config(), None, 'reference', 'r', candidate_id='c0')
    r.run(t, toy.reference_config(), None, 'probe', 'r', candidate_id='c1', stop_after=3)
    r.run(t, toy.reference_config(), None, 'short', 'r', candidate_id='c2', stop_after=6)
    fast = [e.get('INFRARSI_MEMPEAK_FAST') for e in r.executor.envs]
    assert fast == [None, '1', '1']


def test_d3_the_runner_takes_the_maximum_over_every_process_of_every_host(tmp_path):
    r = ProjectRunner.__new__(ProjectRunner)
    r.executor = ex.LocalExecutor()
    d = tmp_path / MEMPEAK / 'exec-0'
    d.mkdir(parents=True)
    (d / 'node0-11.txt').write_text('mempeak 31.5000\n', encoding='utf-8')
    (d / 'node0-12.txt').write_text('mempeak 70.2500\n', encoding='utf-8')
    assert r.memory_peak(str(d)) == 70.25
    assert r.memory_peak(str(tmp_path / 'nothing')) is None

    class Hosts:                                                      # the ssh executor: one read per host
        def __init__(self, texts):
            self.texts, self.cmds = texts, []

        def quote(self, s):
            return s

        def on_hosts(self, cmd, timeout_s=20):
            self.cmds.append(cmd)
            if isinstance(self.texts, Exception):
                raise self.texts
            return self.texts
    r.executor = Hosts(['mempeak 40.0\nmempeak 41.5\n', None, 'mempeak 77.125\n'])
    assert r.memory_peak('/remote/run/mempeak/exec-0') == 77.125
    assert r.executor.cmds == ['cat /remote/run/mempeak/exec-0/*.txt 2>/dev/null']
    r.executor = Hosts(TimeoutError('ssh'))
    assert r.memory_peak('/remote/run/mempeak/exec-0') is None


@pytest.fixture
def toy(tmp_path, monkeypatch):
    """The CPU toy job whose kernels module stands in for torch on two GPUs (3 and 6 GiB reserved) and lives long enough
    for one read."""
    monkeypatch.setenv('TOY_TIME_SCALE', '0.004')
    d = tmp_path / 'toy'
    shutil.copytree(TOY, d, ignore=shutil.ignore_patterns('.infrarsi', '__pycache__'))
    doc = yaml.safe_load((d / 'infrarsi.yaml').read_text(encoding='utf-8'))
    doc['workload'] = {'steps': 12, 'window': [5, 12], 'invariants': {'global_batch': 'micro_batch_size * grad_accum'}}
    doc['checkpoint']['save_at'] = 6
    (d / 'infrarsi.yaml').write_text(yaml.safe_dump(doc), encoding='utf-8')
    k = d / 'kernels.py'
    k.write_text('import sys as _s, time as _t, types as _y\n'
                 '_m = _y.ModuleType("torch")\n'
                 '_r = lambda d: {"reserved_bytes": {"all": {"current": (d + 1) * 3 * 2 ** 30, "peak": (d + 1) * 3 * 2 ** 30}}}\n'
                 '_m.cuda = _y.SimpleNamespace(is_initialized=lambda: True, device_count=lambda: 2,\n'
                 '                             memory_stats_as_nested_dict=_r)\n'
                 '_s.modules["torch"] = _m\n'
                 f'_t.sleep({hook.PEAK_FAST + 0.8})\n' + k.read_text(encoding='utf-8'), encoding='utf-8')
    return Project.load(d)


def test_d3_a_run_records_its_measured_peak_and_nothing_else_changes(toy, tmp_path, monkeypatch):
    t = toy.task()
    r = ProjectRunner(toy, out_dir=tmp_path / 'runs')
    v = make_verifier(toy)
    ref = r.run(t, toy.reference_config(), None, 'reference', 'r', candidate_id='c0')
    ev = ref['evidence']
    assert ev['memory_peak_measured_gib'] == 6.0                       # the largest of the process's devices
    mods = {}
    for f in Path(ev['modules_dir']).glob('*.json'):                   # D7: the reference run's imports (rank 0)
        mods.update(json.loads(f.read_text(encoding='utf-8')))
    assert mods['yaml'].replace('\\', '/').split('/')[-2] == 'yaml' and Path(mods['kernels']).name == 'kernels.py'
    scr = r.run(t, toy.reference_config(), None, 'screen', 'r', candidate_id='c1')
    assert scr['evidence']['memory_peak_measured_gib'] == 6.0 and 'modules_dir' not in scr['evidence']
    monkeypatch.setenv('INFRARSI_NO_MEMPEAK', '1')                   # the hook staged for no run: as before
    off = r.run(t, toy.reference_config(), None, 'reference', 'r', candidate_id='c2')
    assert 'memory_peak_measured_gib' not in off['evidence'] and 'modules_dir' not in off['evidence']
    for a, b in ((ref, off), (scr, off)):                              # the verdict reads what it read before
        va, vb = v(a, None, a), v(b, None, b)
        assert va.valid == vb.valid and [c['id'] for c in va.checks] == [c['id'] for c in vb.checks]
        assert a['evidence'].get('memory_peak_gib') == b['evidence'].get('memory_peak_gib')
        assert a['evidence']['steps_completed'] == b['evidence']['steps_completed'] == 12


class CutOnce(ex.LocalExecutor):
    """A transport that cuts its first execution off after the job's processes wrote their peaks (Codex HLS5 r1: the
    runner tries again in the same attempt directory)."""

    def __init__(self, stale: float):
        super().__init__()
        self.stale, self.envs = stale, []

    def run(self, cmd, *, cwd, env, timeout_s, log_path, stop=None):
        self.envs.append(dict(env))
        if len(self.envs) == 1:
            for key in ('INFRARSI_MEMPEAK', 'INFRARSI_MODULES'):
                d = Path(env[key])
                d.mkdir(parents=True, exist_ok=True)
            peaks = Path(env['INFRARSI_MEMPEAK'])
            (peaks / 'node0-1.txt').write_text(f'mempeak {self.stale:.4f}\n', encoding='utf-8')
            (Path(env['INFRARSI_MODULES']) / '1.json').write_text(json.dumps({'cut_off_only': '/x.py'}),
                                                                     encoding='utf-8')
            raise InfrastructureError('ssh: connection reset by peer')
        return super().run(cmd, cwd=cwd, env=env, timeout_s=timeout_s, log_path=log_path, stop=stop)


def test_r1_a_transport_retry_reports_the_peak_of_the_execution_it_describes(toy, tmp_path):
    r = ProjectRunner(toy, out_dir=tmp_path / 'runs')
    r.executor, r.infra_retry_waits = CutOnce(76.0), (0.0,)
    ref = r.run(toy.task(), toy.reference_config(), None, 'reference', 'r', candidate_id='c0')
    ev = ref['evidence']
    assert len(r.executor.envs) == 2 and ev['steps_completed'] == 12
    assert ev['memory_peak_measured_gib'] == 6.0                       # the final execution's, not the cut-off one's 76
    first, last = (e['INFRARSI_MEMPEAK'] for e in r.executor.envs)
    assert first != last and Path(last).name == 'exec-1'
    mods = {}
    for f in Path(ev['modules_dir']).glob('*.json'):                   # D7: its own module list too
        mods.update(json.loads(f.read_text(encoding='utf-8')))
    assert 'cut_off_only' not in mods and 'kernels' in mods


def test_d3_a_full_run_stopped_by_a_signal_has_no_peak_and_the_log_says_so(toy, tmp_path, monkeypatch):
    """A full run reads its peak at its exit only: one the runner stops (an early abort for speed) has none - unknown,
    not zero, and the run's log says so; a probe stopped as planned keeps its sampler's value."""
    monkeypatch.setenv('TOY_TIME_SCALE', '0.3')                       # (updates slow enough to be stopped)
    logs = []
    r = ProjectRunner(toy, out_dir=tmp_path / 'runs', log=logs.append)
    t = toy.task()
    scr = r.run(t, toy.reference_config(), None, 'screen', 'r', candidate_id='c1',
                stop=lambda rows: bool(rows) and rows[-1]['step'] >= 2)
    assert scr['evidence']['stopped_early'] and 'memory_peak_measured_gib' not in scr['evidence']
    assert any('c1: screen: peak memory unknown' in m for m in logs)
    probe = r.run(t, toy.reference_config(), None, 'probe', 'r', candidate_id='c2', stop_after=3)
    assert probe['evidence']['planned_stop'] and probe['evidence']['memory_peak_measured_gib'] == 6.0
    assert not any('c2: probe: peak memory unknown' in m for m in logs)


class NoPeakWrites(ex.LocalExecutor):
    """A job whose processes write no peak although the run was neither stopped nor timed out: one that ends with
    os._exit, or one killed from outside after its updates (the exit read never ran)."""

    def run(self, cmd, *, cwd, env, timeout_s, log_path, stop=None):
        env = {k: v for k, v in env.items() if k != 'INFRARSI_MEMPEAK'}
        return super().run(cmd, cwd=cwd, env=env, timeout_s=timeout_s, log_path=log_path, stop=stop)


def test_r6_a_full_run_that_wrote_no_peak_says_so_even_when_not_stopped(toy, tmp_path):
    """Codex HLS5 r6: the unknown-peak line does not depend on the runner having stopped the run."""
    logs = []
    r = ProjectRunner(toy, out_dir=tmp_path / 'runs', log=logs.append)
    r.executor = NoPeakWrites()
    scr = r.run(toy.task(), toy.reference_config(), None, 'screen', 'r', candidate_id='c3')
    ev = scr['evidence']
    assert not ev.get('stopped_early') and not ev.get('timed_out') and 'memory_peak_measured_gib' not in ev
    assert any('c3: screen: peak memory unknown' in m for m in logs)
