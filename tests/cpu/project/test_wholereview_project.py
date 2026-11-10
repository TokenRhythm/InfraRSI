"""Whole-codebase review fixes in the project layer (2026-10-24, C1 / C2 / C4): staging never writes through a link;
apply merges onto later edits, renders every constant as the runs did and prints env values as the runs set them; the
HF Trainer summary and checkpoint lines are not updates; the update log is the one with the losses; an output directory
inside the project is not copied and a relative one is made absolute; string `llm` forms; eval / test commands after
launch.setup; NaN in a probe file; srun runs the whole line; the GPU snapshots look at the job's devices; a failed run
exits nonzero; --llm over the manifest's endpoint; integer ranges from 0; the probe keeps the schedule horizon on the
real toy job (Fable 5.1's design)."""
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

from infrarsi.project import Project, ProjectRunner
from infrarsi.project import executor as ex
from infrarsi.project import signals
from infrarsi.project.apply import apply, plan
from infrarsi.project.flow import capability_of, make_verifier, run_project
from infrarsi.project.llm import override
from infrarsi.search.knob_space import choices
from test_project import TOY, toy  # noqa: F401 - the fixture


def _status(v, cid):
    return {c['id']: c['status'] for c in v.checks}[cid]


def test_staging_does_not_write_through_links(toy, tmp_path):
    shared = tmp_path / 'shared'
    (shared / 'pkg').mkdir(parents=True)
    for n in ('config.yaml', 'kernels.py'):
        shutil.copyfile(toy.root / n, shared / n)
    (shared / 'pkg' / 'mod.py').write_text('x = 1\n', encoding='utf-8')
    orig = {p: p.read_text(encoding='utf-8') for p in shared.rglob('*') if p.is_file()}
    try:
        for n in ('config.yaml', 'kernels.py'):
            (toy.root / n).unlink()
            os.symlink(shared / n, toy.root / n)
        os.symlink(shared / 'pkg', toy.root / 'pkg', target_is_directory=True)
    except OSError:
        pytest.skip('symbolic links are not available here')
    proj = Project.load(toy.root)
    tree = tmp_path / 'tree'
    ProjectRunner(proj, out_dir=tmp_path / 'runs').materialize(
        tree, dict(proj.reference_config(), fused_optimizer=True), {'kernels.py': '# patched\n', 'pkg/mod.py': 'x = 2\n'})
    assert {p: p.read_text(encoding='utf-8') for p in orig} == orig           # the shared files are untouched
    assert (tree / 'kernels.py').read_text(encoding='utf-8') == '# patched\n' and not (tree / 'kernels.py').is_symlink()
    assert 'fused_optimizer: true' in (tree / 'config.yaml').read_text(encoding='utf-8')


def _toy_const(tmp_path):
    d = tmp_path / 'toyc'
    shutil.copytree(TOY, d, ignore=shutil.ignore_patterns('.infrarsi', '__pycache__'))
    (d / 'kernels.py').write_text((d / 'kernels.py').read_text(encoding='utf-8') + '\n\nSCALE = 2\n', encoding='utf-8')
    doc = yaml.safe_load((d / 'infrarsi.yaml').read_text(encoding='utf-8'))
    doc['knobs']['scale'] = {'type': 'int', 'choices': [2, 4], 'const': 'SCALE', 'file': 'kernels.py'}
    doc['knobs']['p2p_off'] = {'type': 'bool', 'env': 'NCCL_P2P_DISABLE'}
    (d / 'infrarsi.yaml').write_text(yaml.safe_dump(doc), encoding='utf-8')
    return Project.load(d)


def test_apply_merges_later_edits_and_renders_every_constant(tmp_path):
    proj = _toy_const(tmp_path)
    ref = proj.reference_config()
    base = (proj.root / 'kernels.py').read_text(encoding='utf-8')
    found = base.replace('chunks * chunks / 4', 'chunks / 2').replace('SCALE = 2', 'SCALE = 4')   # the overlay moved SCALE
    run = tmp_path / 'run'
    run.mkdir()
    promo = {'promoted': 'c3', 'config': dict(ref, p2p_off=True), 'overlay': {'kernels.py': found}, 'base': {'kernels.py': base}}
    (run / 'result.json').write_text(json.dumps({'run_id': 'r1', 'manifest': str(proj.path), 'reference': {'config': ref},
                                                 'promotion': promo}), encoding='utf-8')
    p = plan(run)
    assert 'chunks / 2' in p['files'][0]['new'] and 'SCALE = 2' in p['files'][0]['new']   # the constant the runs had
    assert any('NCCL_P2P_DISABLE=1 ' in m for m in p['manual'])
    (proj.root / 'kernels.py').write_text('# edited after the search\n' + base, encoding='utf-8')
    p = plan(run)                                                     # an edit elsewhere: merged
    assert not p['conflicts'] and p['files'][0]['new'].startswith('# edited after the search\n')
    assert 'chunks / 2' in p['files'][0]['new']
    (proj.root / 'kernels.py').write_text(base.replace('chunks * chunks / 4', 'chunks * chunks / 3'), encoding='utf-8')
    with pytest.raises(RuntimeError, match='does not merge'):        # an edit of the same line: nothing written
        apply(run, write=True)
    assert 'chunks * chunks / 3' in (proj.root / 'kernels.py').read_text(encoding='utf-8')


def test_hf_summary_and_checkpoint_lines_are_not_updates():
    hf = [(float(i), "{'loss': %.3f, 'grad_norm': 1.0, 'learning_rate': 1e-05, 'epoch': %.2f}" % (2 - 0.01 * i, i / 30))
          for i in range(1, 31)]
    hf.append((31.0, "{'train_runtime': 12.3, 'train_samples_per_second': 9.9, 'train_steps_per_second': 2.4, "
                     "'train_loss': 1.9, 'epoch': 1.0}"))
    assert len(signals.LogParser({}, None, True).parse(hf)) == 30
    meg = [(0.5, ' successfully loaded checkpoint from /ckpt [ t 1/1, p 1/1 ] at iteration 0')] + \
        [(1.0 + i, f' iteration {i:8d}/      30 | lm loss: {2 - 0.01 * i:.6E} |') for i in range(1, 31)]
    rows = signals.LogParser({}, None, False).parse(meg)
    assert [r['step'] for r in rows[:2]] == [1, 2] and len(rows) == 30


def test_the_log_with_the_losses_is_read(tmp_path, monkeypatch):
    from test_gpu_link_checks import SmiExecutor, _toy

    class Ssh(SmiExecutor):
        kind = 'ssh'

        def probe(self, cmd, *, cwd, env, timeout_s=30):
            return None

        def run(self, cmd, *, cwd, env, timeout_s, log_path, stop=None, python='python3'):
            return {'exit_code': 0, 'timed_out': False, 'stopped': False, 'wall_s': 5.0, 'lines': [(1.0, 'resuming at step 0')],
                    'node_lines': [[(10.0 + 1.4 * i, f'step {i} | loss {2.0 - 0.01 * i}') for i in range(1, 7)]]}
    proj = _toy(tmp_path, monkeypatch)
    proj.raw['hardware']['nodes'] = 2
    rec = ProjectRunner(proj, out_dir=tmp_path / 'att', executor=Ssh([]), log=lambda m: None).run(
        None, proj.reference_config(), None, 'screen', 'r', candidate_id='c1')
    assert rec['evidence']['steps_completed'] == 6 and rec['step_log'].endswith('node1.log')


def test_a_relative_output_and_one_inside_the_project(toy, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = run_project(toy, strategy='random', budget={'screens': 1}, out_root='rel_runs', log=lambda m: None)
    assert Path(res['run_dir']).is_absolute() and res['outcome'] in ('improved', 'no_change')
    assert 'same_data' in res['reference']['coverage']['unchecked']   # what a candidate's signals cannot show
    res = run_project(toy, strategy='random', budget={'screens': 1}, out_root=toy.root / 'runs', log=lambda m: None)
    assert res['outcome'] in ('improved', 'no_change')                # not copied into its own trees
    assert toy.describe()['world'] == toy.gpus * toy.nodes


def test_string_llm_forms_and_the_cli_override(toy):
    for spec in ('none', 'claude-cli', 'vllm:local-model'):
        toy.raw['llm'] = spec
        assert 0 <= capability_of(toy, None) <= 1
    m = {'backend': 'openai-compat', 'model': 'a', 'base_url': 'http://h:18000/v1', 'context_tokens': 40960, 'capability': 0.4}
    assert override(m, 'openai-compat:b') == {'backend': 'openai-compat', 'model': 'b', 'base_url': 'http://h:18000/v1',
                                              'context_tokens': 40960}
    assert override(m, 'openai-compat:a')['capability'] == 0.4      # the same model keeps its capability
    assert override(m, 'none') == {'backend': 'none'} and override(m, 'claude-cli') == {'backend': 'claude-cli'}


def test_tests_run_after_the_setup(toy, tmp_path):
    toy.raw['launch']['setup'] = 'echo set up'
    out = ProjectRunner(toy, out_dir=tmp_path / 'runs').run_tests(toy.reference_config(), None, 'r', 'c0')
    assert out[0]['cmd'].startswith('echo set up && ') and out[0]['exit_code'] == 0


def test_nan_in_a_probe_file():
    rows, _ = signals.read_probe('{"step": 1, "t": 1.0, "loss": NaN, "grad_norm": Infinity}\n')
    assert (rows[0]['loss'], rows[0]['grad_norm']) == ('nan', 'inf')


def test_srun_runs_the_whole_line_in_the_allocation(tmp_path, monkeypatch):
    seen = {}

    def fake_popen(cmd, **kw):
        seen['cmd'] = cmd
        return subprocess.Popen([sys.executable, '-c', 'print(1)'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    monkeypatch.setattr(ex, '_popen', fake_popen)
    ex.make({'where': 'slurm', 'slurm_args': '--nodes 1'}).run('export A=1 && python t.py', cwd=str(tmp_path), env={},
                                                              timeout_s=30, log_path=tmp_path / 'l.log')
    assert seen['cmd'] == "srun --nodes 1 bash -c 'export A=1 && python t.py'"


def test_gpu_snapshots_look_at_the_jobs_devices(tmp_path, monkeypatch):
    from test_gpu_link_checks import SmiExecutor, _toy
    proj = _toy(tmp_path, monkeypatch)
    for setup, want in (('export CUDA_VISIBLE_DEVICES=4,5', ['4,5', '4,5']), ('export CUDA_VISIBLE_DEVICES=$GPUS', [])):
        proj.raw['launch']['setup'] = setup
        runner = ProjectRunner(proj, out_dir=tmp_path / 'att', executor=SmiExecutor([]), log=lambda m: None)
        seen = []
        monkeypatch.setattr(runner, 'gpu_state', lambda vis=None: seen.append(vis))
        runner.run(None, proj.reference_config(), None, 'screen', 'r', candidate_id='c1')
        assert seen == want, setup


def test_a_failed_run_exits_nonzero(toy, tmp_path):
    from infrarsi.cli.main import main
    doc = yaml.safe_load((toy.root / 'infrarsi.yaml').read_text(encoding='utf-8'))
    doc['launch']['command'] = '{python} -c "import sys; sys.exit(3)"'
    (toy.root / 'infrarsi.yaml').write_text(yaml.safe_dump(doc), encoding='utf-8')
    with pytest.raises(SystemExit) as e:
        main(['run', str(toy.root / 'infrarsi.yaml'), '--llm', 'none', '--out', str(tmp_path / 'o')])
    assert e.value.code == 1


def test_integer_ranges_from_zero():
    task = types.SimpleNamespace(knob_schema={'properties': {'w': {'type': 'integer', 'minimum': 0, 'maximum': 128}}})
    c = choices(task)['w']
    assert c[0] == 0 and c[-1] == 128 and len(c) > 8


def test_a_probe_keeps_the_schedule_horizon_on_the_toy(toy, tmp_path):
    t, r, v = toy.task(), ProjectRunner(toy, out_dir=tmp_path / 'runs'), make_verifier(toy)
    ref_cfg = toy.reference_config()
    ref = r.run(t, ref_cfg, None, 'reference', 'r', candidate_id='c0')
    pr = r.run(t, dict(ref_cfg, fused_optimizer=True), None, 'probe', 'r', candidate_id='c1', steps=12, stop_after=2)
    ev = pr['evidence']
    assert (ev['steps'], ev['stop_after']) == (12, 2) and ev['steps_completed'] >= 2 and not ev['stopped_early']
    vp = v(pr, ref, ref)
    assert vp.valid and _status(vp, 'optimizer_steps') == 'unchecked', vp.reasons   # the toy logs lr, no step counter
    old = r.run(t, dict(ref_cfg, fused_optimizer=True), None, 'probe', 'r', candidate_id='c2', steps=2)
    assert _status(v(old, ref, ref), 'optimizer_steps') == 'fail'        # the defect: a 2-update schedule
    bad = r.run(t, dict(ref_cfg, precision='fp8'), None, 'probe', 'r', candidate_id='c3', steps=12, stop_after=2)
    vb = v(bad, ref, ref)
    assert not vb.valid and 'loss_within_tolerance' in vb.feedback


# ---------------------------------------------------------------- round 2 (Astra CF_r1): the planned stop
class _Canned(ex.LocalExecutor):
    """A local executor that returns a canned run (stdout lines, exit code, whether our stop fired and where)."""

    def __init__(self, res):
        super().__init__()
        self.res = res

    def on_hosts(self, cmd, timeout_s=20, login=False):
        return []

    def run(self, cmd, *, cwd, env, timeout_s, log_path, stop=None):
        return dict(self.res)


def _updates(n):
    return [(1.0 + i, f'step {i} | loss {2.0 - 0.01 * i:.4f} | lr 1e-05') for i in range(1, n + 1)]


def test_a_stop_is_planned_only_when_ours_and_its_shutdown_is_not_an_error(toy, tmp_path):
    t, v = toy.task(), make_verifier(toy)
    tb = [(9.0, 'Traceback (most recent call last):'), (9.1, 'torch.distributed.elastic.multiprocessing.api.SignalException')]

    def run(res, cid, purpose='probe', **kw):
        r = ProjectRunner(toy, out_dir=tmp_path / cid, executor=_Canned(res), log=lambda m: None)
        return r.run(t, toy.reference_config(), None, purpose, 'r', candidate_id=cid, **kw)
    ref = run({'exit_code': 0, 'timed_out': False, 'stopped': False, 'stop_index': None, 'wall_s': 9.0,
               'lines': _updates(12)}, 'c0', 'reference', steps=12)
    ours = run({'exit_code': -15, 'timed_out': False, 'stopped': True, 'stop_index': 2, 'wall_s': 3.0,
                'lines': _updates(2) + tb}, 'c1', steps=12, stop_after=2)
    assert ours['evidence']['planned_stop'] and v(ours, ref, ref).valid           # torchrun's SIGTERM traceback is ours
    failed = run({'exit_code': 1, 'timed_out': False, 'stopped': False, 'stop_index': None, 'wall_s': 3.0,
                  'lines': _updates(2)}, 'c2', steps=12, stop_after=2)
    assert not failed['evidence']['planned_stop'] and not v(failed, ref, ref).valid    # it ended by itself, failing
    before = run({'exit_code': -15, 'timed_out': False, 'stopped': True, 'stop_index': 4, 'wall_s': 3.0,
                  'lines': _updates(1) + tb + _updates(2)[1:]}, 'c3', steps=12, stop_after=2)
    assert not v(before, ref, ref).valid                                          # a traceback before our stop counts


def test_the_stop_is_polled_for_a_silent_job_and_reads_every_node(tmp_path, monkeypatch):
    import time
    proc = ex._popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    t0 = time.monotonic()
    res = ex.stream(proc, timeout_s=60, log_path=tmp_path / 'a.log', stop=lambda lines: True, poll_s=0.2)
    assert res['stopped'] and time.monotonic() - t0 < 20                          # not the 30 s the job wanted
    from test_multinode_logs import STEP
    seen = []

    def fake_popen(argv, **kw):                         # node 0 says nothing for a while, node 1 prints the updates
        seen.append(argv)
        lines = [] if len(seen) == 1 else [STEP.format(i=i) for i in (1, 2, 3)]
        prog = ('import time; ' + ''.join(f'print({x!r}, flush=True); ' for x in lines)
                + ('time.sleep(8); ' if len(seen) == 1 else '') + f'print("{ex.END_MARK} 0")')
        return subprocess.Popen([sys.executable, '-c', prog], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                start_new_session=os.name != 'nt')   # as ssh clients: their own session, so killpg reaches them
    monkeypatch.setattr(ex, '_popen', fake_popen)
    monkeypatch.setattr(ex.SSHExecutor, '_reap', lambda self, i, cwd, grace_s=30: None)
    out = ex.SSHExecutor(['h1', 'h2'], '/r').run('train', cwd='/r/t', env={}, timeout_s=60, log_path=tmp_path / 'b.log',
                                                 stop=lambda lines: any('iteration' in x for _, x in lines))
    assert out['stopped']


def test_out_dot_skips_this_runs_directory(toy):
    r = ProjectRunner(toy, out_dir=toy.root / 'toy-cpu-run' / 'attempts', skip=toy.root)
    assert r.skip == 'toy-cpu-run'


# ---------------------------------------------------------------- round 3 (Astra CF_r2): the stop path
def _nodes(monkeypatch, progs):
    seen = []

    def fake_popen(argv, **kw):
        prog = progs[len(seen)]
        seen.append(argv)
        return subprocess.Popen([sys.executable, '-c', prog], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                start_new_session=os.name != 'nt')   # as ssh clients: their own session, so killpg reaches them
    monkeypatch.setattr(ex, '_popen', fake_popen)
    monkeypatch.setattr(ex.SSHExecutor, '_reap', lambda self, i, cwd, grace_s=30: None)


def test_a_worker_that_failed_before_our_stop_is_no_planned_stop(tmp_path, monkeypatch, toy):
    from test_multinode_logs import STEP
    worker = (''.join(f'print({STEP.format(i=i)!r}, flush=True); ' for i in (1, 2))
              + "print('1700000000.000003\ttorch.OutOfMemoryError: CUDA out of memory.', flush=True); "
              + f'print("{ex.END_MARK} 1")')
    _nodes(monkeypatch, ['import time; time.sleep(8); print("x")', worker])
    out = ex.SSHExecutor(['h1', 'h2'], '/r').run('train', cwd='/r/t', env={}, timeout_s=60, log_path=tmp_path / 'a.log',
                                                 stop=lambda lines: sum('iteration' in x for _, x in lines) >= 2)
    assert out['stopped'] and out['failed_before_stop'] == [1] and out['node_stop_index'] == [3]
    canned = dict(out, lines=[], wall_s=3.0)
    r = ProjectRunner(toy, out_dir=tmp_path / 'r', executor=_Canned(canned), log=lambda m: None)
    rec = r.run(toy.task(), toy.reference_config(), None, 'probe', 'r', candidate_id='c1', steps=12, stop_after=2)
    assert not rec['evidence']['planned_stop'] and rec['evidence']['events']['oom'] == 1   # the worker's OOM is the job's


def test_the_cutoff_is_the_live_buffer_when_the_stop_is_accepted(tmp_path):
    lines, calls = [], []

    def stop(snapshot):                                  # a slow predicate: the reader appends while it runs
        calls.append(len(snapshot))
        if len(calls) == 1:
            return False
        lines.append((0.0, 'Traceback (most recent call last):'))
        return True
    proc = ex._popen([sys.executable, '-c', 'import time; print("step 1", flush=True); time.sleep(30)'])
    res = ex.stream(proc, timeout_s=60, log_path=tmp_path / 'a.log', stop=stop, poll_s=0.2, lines=lines)
    assert res['stopped'] and res['stop_index'] == 2      # 'step 1' and the traceback read before the stop


def test_a_silent_ssh_job_is_stopped_by_its_probe_file(tmp_path, monkeypatch):
    import time
    _nodes(monkeypatch, ['import time; time.sleep(30)', 'import time; time.sleep(30)'])
    t0 = time.monotonic()
    out = ex.SSHExecutor(['h1', 'h2'], '/r').run('train', cwd='/r/t', env={}, timeout_s=60, log_path=tmp_path / 'a.log',
                                                 stop=lambda lines: True)       # the probe file says k updates
    assert out['stopped'] and time.monotonic() - t0 < 20

def test_node_0_failing_before_our_stop_is_no_planned_stop(tmp_path, monkeypatch):
    from test_multinode_logs import STEP
    node0 = f'import time; print("{ex.END_MARK} 1", flush=True); time.sleep(8)'      # failed; its ssh client lingers
    node1 = 'import time; ' + ''.join(f'print({STEP.format(i=i)!r}, flush=True); ' for i in (1, 2)) + 'time.sleep(8)'
    _nodes(monkeypatch, [node0, node1])
    out = ex.SSHExecutor(['h1', 'h2'], '/r').run('train', cwd='/r/t', env={}, timeout_s=60, log_path=tmp_path / 'a.log',
                                                 stop=lambda lines: sum('iteration' in x for _, x in lines) >= 2)
    assert out['stopped'] and out['failed_before_stop'] == [0]
