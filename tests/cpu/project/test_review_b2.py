"""Review B2 (gpt-6.1-sol): override reference values, invariant typos, quoted path placeholders, SSH root and
multi-node world size, optional optuna, library modules as entry points, the ledger after promotion, process-group kill."""
import json
import os
import subprocess
import sys
import time

import pytest
import yaml

from infrarsi import search
from infrarsi.project import Project, ProjectRunner, detect, executor as ex
from infrarsi.project.flow import prior_results, remember
from infrarsi.project.manifest import ManifestError, args_for


def _project(tmp_path, cfg: dict, knobs: dict, **extra):
    (tmp_path / 'train.py').write_text('print("step 1 loss 1.0")\n', encoding='utf-8')
    (tmp_path / 'conf.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
    raw = {'name': 'b2', 'launch': {'command': '{python} train.py {config} {args}'}, 'config': {'file': 'conf.yaml'},
           'knobs': knobs, 'workload': {'steps': 4, 'window': [2, 4]}, **extra}
    return Project(raw, tmp_path / 'infrarsi.yaml')


def test_override_reference_reads_the_config(tmp_path):
    p = _project(tmp_path, {'precision': 'amp_bf16', 'fsdp': {'ac': True}},
                 {'precision': {'type': 'str', 'choices': ['amp_bf16', 'amp_fp8'], 'override': 'precision'},
                  'ac': {'type': 'bool', 'override': 'fsdp.ac'},
                  'offload': {'type': 'bool', 'override': 'fsdp.offload'}})
    ref = p.reference_config()
    assert ref == {'precision': 'amp_bf16', 'ac': True, 'offload': None}       # absent override stays unset
    assert args_for(p, ref) == ['precision=amp_bf16', 'fsdp.ac=true']           # nothing forced to false


def test_invariant_typo_is_rejected_at_load(tmp_path):
    with pytest.raises(ManifestError, match='workload.invariants.global_batch'):
        _project(tmp_path, {'mbs': 2}, {'mbs': {'type': 'int', 'choices': [1, 2], 'path': 'mbs'}},
                 workload={'steps': 4, 'window': [2, 4], 'invariants': {'global_batch': 'unknown * mbs'}})


def test_path_placeholders_are_quoted(tmp_path):
    p = _project(tmp_path, {'mbs': 2}, {'mbs': {'type': 'int', 'choices': [1, 2], 'path': 'mbs'}},
                 checkpoint={'save': '--save {ckpt_dir}', 'resume': '--load {ckpt_dir}'})
    r = ProjectRunner(p, out_dir=tmp_path / 'runs')
    cmd = r.command({'mbs': 2}, cfg_rel='my conf.yaml', steps=4, run_dir='/tmp/a b', extra_args='--save {ckpt_dir}')
    q = r.executor.quote
    assert q('my conf.yaml') in cmd and q('/tmp/a b/ckpt') in cmd and q('my conf.yaml') != 'my conf.yaml'


def test_ssh_root_and_world_size(tmp_path, monkeypatch):
    p = _project(tmp_path, {'mbs': 2}, {'mbs': {'type': 'int', 'choices': [1, 2], 'path': 'mbs'}},
                 hardware={'gpus': 8}, execution={'where': 'ssh', 'hosts': ['u@h1:22', 'u@h2:22'], 'remote_dir': '~/is'})
    assert (p.nodes, p.world, p.task().raw.get('world_size', p.world)) == (2, 16, 16)
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout='/home/u', stderr='')

    monkeypatch.setattr(ex.subprocess, 'run', fake_run)
    r = ProjectRunner(p, out_dir=tmp_path / 'runs')
    assert r.shared_dir('run1', 'ckpt-c3') == '/home/u/is/run1/ckpt-c3'
    assert r.executor.root() == '/home/u/is' and len(calls) == 1                 # expanded once


def test_auto_strategy_without_optuna(monkeypatch):
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name, *a: None if name == 'optuna' else real(name, *a))
    assert search.resolve('auto', has_llm=False) == 'random'
    assert search.resolve(None, has_llm=True) == 'hls'                      # an LLM and no two-level knob schema


def test_library_module_is_not_an_entry_point(tmp_path):
    (tmp_path / 'megatron_patch').mkdir()
    (tmp_path / 'megatron_patch' / 'training.py').write_text(
        'import os\nfrom typing import TYPE_CHECKING\nX = 1\n\ndef pretrain(a):\n    return a\n', encoding='utf-8')
    assert detect.entry_script(tmp_path) is None
    (tmp_path / 'pretrain_gpt.py').write_text('def main():\n    pass\n\nif __name__ == "__main__":\n    main()\n',
                                              encoding='utf-8')
    assert detect.entry_script(tmp_path).name == 'pretrain_gpt.py'


def test_ledger_rows_carry_the_promotion_outcome(tmp_path, monkeypatch):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    p = _project(tmp_path, {'mbs': 2}, {'mbs': {'type': 'int', 'choices': [1, 2], 'path': 'mbs'}})
    trace = {'extra': {'ledger': [
        {'mechanism': 'a', 'status': 'confirmed', 'measured_gain_pct': 9.0, 'candidate': 'c2'},
        {'mechanism': 'b', 'status': 'confirmed', 'measured_gain_pct': 5.0, 'candidate': 'c4'},
        {'mechanism': 'c', 'status': 'open', 'candidate': None}]}}
    promo = {'tried': [{'candidate': 'c2', 'passed': False, 'items': [{'id': 'speed_reproduces', 'status': 'fail'}]},
                       {'candidate': 'c4', 'passed': True, 'items': []}]}
    assert remember(p, trace, promo) == 2
    rows = {r['mechanism']: r for r in prior_results(p)}
    assert rows['a']['promotion'] == 'failed: speed_reproduces' and rows['b']['promotion'] == 'passed'


@pytest.mark.skipif(os.name == 'nt', reason='process groups are POSIX')
def test_kill_tree_reaches_an_orphaned_child(tmp_path):
    marker = tmp_path / 'alive'
    child = f'import time, pathlib; time.sleep(3); pathlib.Path({str(marker)!r}).write_text("x")'
    proc = ex._popen(f'{sys.executable} -c {ex.LocalExecutor.quote(child)} & exit 0', shell=True)
    time.sleep(0.5)                                        # the launcher has exited; the child holds the pipe
    ex._kill_tree(proc, grace_s=2.0)
    time.sleep(3.5)
    assert not marker.exists()


# ---------------------------------------------------------------- review R3a (gpt-6.1-sol)
def test_override_set_to_null_stays_null(tmp_path):
    p = _project(tmp_path, {'fsdp': {'ac': None}}, {'ac': {'type': 'bool', 'override': 'fsdp.ac', 'default': True}})
    assert p.reference_config() == {'ac': None} and args_for(p, p.reference_config()) == []


def test_quoted_path_placeholder_is_not_quoted_twice(tmp_path):
    p = _project(tmp_path, {'mbs': 2}, {'mbs': {'type': 'int', 'choices': [1, 2], 'path': 'mbs'}})
    p.raw['launch']['command'] = '{python} train.py --config "{config}" --out \'{run_dir}\' --save {ckpt_dir} {args}'
    r = ProjectRunner(p, out_dir=tmp_path / 'runs')
    cmd = r.command({'mbs': 2}, cfg_rel='my conf.yaml', steps=4, run_dir='/tmp/a b')
    assert '--config "my conf.yaml" --out \'/tmp/a b\'' in cmd            # the template's quotes, the raw value
    assert f'--save {r.executor.quote("/tmp/a b/ckpt")}' in cmd                # a bare placeholder: quoted for the shell


def test_template_quotes_inside_an_embedded_program_stay(tmp_path):
    from infrarsi.project.runner import fill
    t = '{python} -c \'import pathlib; assert pathlib.Path("{config}").is_file()\' --out "{run_dir}"'
    got = fill(t, {'python': 'py', 'config': "'conf.yaml'", 'run_dir': 'C:\\R&D\\runs'},
               {'config': 'conf.yaml', 'run_dir': 'C:\\R&D\\runs'})
    assert got == 'py -c \'import pathlib; assert pathlib.Path("conf.yaml").is_file()\' --out "C:\\R&D\\runs"'


def test_the_end_marker_ends_a_client_that_outlives_its_job(tmp_path, monkeypatch):
    # trial C3d: a Windows `ssh -tt` client kept waiting 30 minutes after the remote job had ended
    line = '1700000000.000001' + chr(9) + 'step 1 | loss 2.0'
    prog = f'import time; print({line!r}, flush=True); print("{ex.END_MARK} 3", flush=True); time.sleep(60)'
    t0 = time.monotonic()
    res = ex.stream(ex._popen([sys.executable, '-c', prog]), timeout_s=120, log_path=tmp_path / 'a.log', end=ex.END_MARK,
                    end_grace_s=1.0)
    assert res['exit_code'] == 3 and [t for _, t in res['lines']] == [line] and not res['timed_out']
    assert time.monotonic() - t0 < 30
    seen = []

    def fake_popen(argv, **kw):                         # the ssh client: the job's stamped line, then the marker
        seen.append(argv)
        return ex.subprocess.Popen([sys.executable, '-c', f'print({line!r}); print("{ex.END_MARK} 7")'],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    monkeypatch.setattr(ex, '_popen', fake_popen)
    res = ex.SSHExecutor(['u@h1:22'], '/r').run('train', cwd='/r/t', env={}, timeout_s=60, log_path=tmp_path / 'b.log')
    assert res['exit_code'] == 7 and res['node_exit_codes'] == [7] and ex.END_MARK in seen[0][-1]
    assert [t for _, t in res['lines']] == ['step 1 | loss 2.0']


def test_the_end_marker_after_an_unterminated_line_and_a_lingering_client(tmp_path):
    # QA18: a job whose last line has no newline still reports its status; a client lingering after the marker is
    # not a timeout of the job
    line = '1700000000.000001' + chr(9) + 'final'
    prog = (f'import sys, time; sys.stdout.write({line!r}); sys.stdout.write("\\n{ex.END_MARK} 7\\n"); '
            'sys.stdout.flush(); time.sleep(3)')
    res = ex.stream(ex._popen([sys.executable, '-c', prog]), timeout_s=1, log_path=tmp_path / 'u.log', end=ex.END_MARK,
                    end_grace_s=2.0)
    assert res['exit_code'] == 7 and not res['timed_out'] and [t for _, t in res['lines']] == [line]


def test_a_stopped_ssh_run_waits_for_the_job_to_leave_the_node(tmp_path, monkeypatch):
    # trial C3d: after an early abort torchrun still held port 29500; the next two candidates failed with EADDRINUSE
    line = '1700000000.000001' + chr(9) + 'step 1 | loss 2.0'
    real = ex._popen                                   # a client in its own process group, as the executor starts it
    monkeypatch.setattr(ex, '_popen', lambda argv, **kw: real(
        [sys.executable, '-c', f'import time; print({line!r}, flush=True); time.sleep(60)']))
    reaped, real_run = [], subprocess.run

    def fake_run(argv, **kw):                          # the reap over ssh is recorded; taskkill (Windows) still runs
        if argv[0] != 'ssh':
            return real_run(argv, **kw)
        reaped.append(argv)
        return subprocess.CompletedProcess(argv, 0, '', '')
    monkeypatch.setattr(ex.subprocess, 'run', fake_run)
    res = ex.SSHExecutor(['u@h1:22', 'u@h2:22'], '/r').run('train', cwd='/r/run/c3/tree', env={}, timeout_s=60,
                                                            log_path=tmp_path / 'c.log', stop=lambda lines: True)
    assert res['stopped'] and [a[-2] for a in reaped] == ['u@h1', 'u@h2']
    assert all("readlink -f /r/run/c3/tree" in a[-1] and 'kill -9' in a[-1] for a in reaped)
    assert all('INFRARSI_RUN_TREE=$t' in a[-1] for a in reaped)        # QA19: a job that left its tree


def test_a_side_node_timeout_also_reaps(tmp_path, monkeypatch):
    # QA19: rank 0 finished, rank 1 timed out; its processes must leave the node before the next candidate starts
    real, seen = ex._popen, []
    progs = iter([f'print("{ex.END_MARK} 0")', 'import time; time.sleep(30)'])
    monkeypatch.setattr(ex, '_popen', lambda argv, **kw: seen.append(argv) or real([sys.executable, '-c', next(progs)]))
    reaped, real_run = [], subprocess.run

    def fake_run(argv, **kw):
        if argv[0] != 'ssh':
            return real_run(argv, **kw)
        reaped.append(argv)
        return subprocess.CompletedProcess(argv, 0, '', '')
    monkeypatch.setattr(ex.subprocess, 'run', fake_run)
    res = ex.SSHExecutor(['u@h1:22', 'u@h2:22'], '/r').run('train', cwd='/r/c1/tree', env={}, timeout_s=1,
                                                            log_path=tmp_path / 'd.log')
    assert res['node_exit_codes'][0] == 0 and res['node_exit_codes'][1] != 0 and [a[-2] for a in reaped] == ['u@h1', 'u@h2']
    assert all('INFRARSI_RUN_TREE="$(pwd -P)"' in a[-1] for a in seen)


def test_a_dropped_connection_reaps_the_job_before_it_is_raised(tmp_path, monkeypatch):
    # trial D1-17: c8's ssh session to winh7 was reset mid-run; the job ran on and held the GPU and the rendezvous
    # port, and the next four runs failed at start
    real = ex._popen
    prog = ('import sys; print("client_loop: send disconnect: Connection reset by peer", file=sys.stderr, flush=True); '
            'sys.exit(255)')
    monkeypatch.setattr(ex, '_popen', lambda argv, **kw: real([sys.executable, '-c', prog]))
    reaped, real_run = [], subprocess.run

    def fake_run(argv, **kw):
        if argv[0] != 'ssh':
            return real_run(argv, **kw)
        reaped.append(argv)
        return subprocess.CompletedProcess(argv, 0, '', '')
    monkeypatch.setattr(ex.subprocess, 'run', fake_run)
    with pytest.raises(ex.InfrastructureError, match='Connection reset by peer'):
        ex.SSHExecutor(['u@h1:22', 'u@h2:22'], '/r').run('train', cwd='/r/run/c8/tree', env={}, timeout_s=60,
                                                         log_path=tmp_path / 'e.log')
    assert [a[-2] for a in reaped] == ['u@h1', 'u@h2'] and all('readlink -f /r/run/c8/tree' in a[-1] for a in reaped)


def test_the_remote_stamper_reads_carriage_return_redraws_as_lines(tmp_path):
    # F-E-C-1: autoresearch's `\rstep k ...` line (and tqdm's bars) arrived as one line over ssh: one stamp for all
    stamper = tmp_path / ex.STAMPER
    stamper.write_bytes(ex.STAMPER_SRC)
    out = subprocess.run([sys.executable, str(stamper)], input=b'\rstep 00000 a\rstep 00001 b\n', capture_output=True,
                         check=True).stdout.decode('utf-8')
    assert [ln.split(chr(9), 1)[1] for ln in out.splitlines()] == ['', 'step 00000 a', 'step 00001 b']


def test_a_connection_that_died_silently_is_a_transport_failure():
    # R2-10 set-up: ssh's keepalive ends a dead connection with "Timeout, server <host> not responding." and exit 255
    assert ex.transport_failure(255, [(0.0, 'Timeout, server winh6 not responding.')]) == 'Timeout, server winh6 not responding.'
    assert ex.transport_failure(255, [(0.0, '1700000000.000001' + chr(9) + 'Timeout, server x not responding.')]) is None


def test_ssh_eval_and_tests_use_the_configured_python(tmp_path, monkeypatch):
    (tmp_path / 'proj').mkdir()
    p = _project(tmp_path / 'proj', {'mbs': 2}, {'mbs': {'type': 'int', 'choices': [1, 2], 'path': 'mbs'}},
                 execution={'where': 'ssh', 'hosts': ['u@h1:22'], 'remote_dir': '/r', 'python': '/opt/venv/bin/python'},
                 verify={'eval': '{python} eval.py {ckpt_dir}', 'tests': ['{python} -m pytest -q']})
    r = ProjectRunner(p, out_dir=tmp_path / 'out')
    seen = []
    monkeypatch.setattr(r.executor, 'place', lambda tree, rel: '/r/' + rel)
    monkeypatch.setattr(r.executor, 'run', lambda cmd, **kw: seen.append(kw.get('python')) or
                        {'exit_code': 0, 'lines': [(0.0, 'eval_loss 1.0')]})
    assert r.run_eval({'mbs': 2}, None, 'run', 'c1', '/r/ckpt')['value'] == 1.0
    r.run_tests({'mbs': 2}, None, 'run', 'c1')
    assert seen == ['/opt/venv/bin/python', '/opt/venv/bin/python']


def test_unpromoted_and_bundled_rows_are_marked(tmp_path, monkeypatch):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    p = _project(tmp_path, {'mbs': 2}, {'mbs': {'type': 'int', 'choices': [1, 2], 'path': 'mbs'}})
    trace = {'extra': {'ledger': [
        {'mechanism': 'a', 'status': 'confirmed', 'measured_gain_pct': 9.0, 'candidate': 'c2'},
        {'mechanism': 'b', 'status': 'refuted', 'measured_gain_pct': -1.0, 'candidate': 'c3'},
        {'mechanism': 'c', 'status': 'bundled', 'measured_gain_pct': 1.0, 'candidate': 'c5', 'bundle_candidate': 'c9'}]}}
    remember(p, trace, {'tried': [{'candidate': 'c9', 'passed': True, 'items': []}]})
    rows = {r['mechanism']: r for r in prior_results(p)}
    assert rows['a']['promotion'] == 'not tried' and rows['b']['promotion'] is None and rows['c']['promotion'] == 'passed'
