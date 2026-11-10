"""G1-G4 - a real PyTorch job on one GPU (opt-in, INFRARSI_SYSTEST_GPU=1 on a node with a free GPU): init drafts the
knobs from the script, doctor runs the reference with the profile hook, a model-less search finds a real GPU gain
(bf16 / compile / fused AdamW) and never promotes the traps (skipped optimizer steps, a shorter sequence), and the
claim reproduces outside the search."""
import os
import shutil
from pathlib import Path

import pytest
import yaml

from _support import cli
from _support.consistency import check_run
from _support.toy import latest_run, report
from infrarsi.project import Project, ProjectRunner
from infrarsi.project.flow import make_verifier

pytestmark = pytest.mark.gpu
HERE = Path(__file__).resolve().parent / '_support' / 'gpujob'


def make_job(dst: Path, *, traps=(), steps: int = 20) -> Path:
    dst.mkdir(parents=True)
    for n in ('train.py', 'config.json'):
        shutil.copyfile(HERE / n, dst / n)
    knobs = {'bf16': {'type': 'bool', 'flag': '--bf16'}, 'compile': {'type': 'bool', 'flag': '--compile'},
             'fused': {'type': 'bool', 'flag': '--fused'},
             'bs': {'type': 'int', 'choices': [1, 2, 4], 'path': 'train.bs'}, 'accum': {'type': 'int', 'choices': [1, 2, 4], 'path': 'train.accum'}}
    if 'half_steps' in traps:
        knobs['half_steps'] = {'type': 'bool', 'flag': '--half-steps'}
    if 'seq' in traps:
        knobs['seq'] = {'type': 'int', 'choices': [512, 1024], 'flag': '--seq', 'default': 1024}
    doc = {'schema': 'infrarsi-project-v1', 'name': 'gpu-mlp',
           'launch': {'command': '{python} train.py --config {config} --steps {steps} {args}', 'timeout_s': 600,
                      'env': {'CUDA_VISIBLE_DEVICES': os.environ.get('INFRARSI_SYSTEST_GPU_ID', '0')}},
           'config': {'file': 'config.json'}, 'knobs': knobs,
           'workload': {'steps': steps, 'window': [8, steps], 'invariants': {'global_batch': 'bs * accum'}},
           'checkpoint': {'save': '--save {ckpt_dir} --save-at {save_at}', 'resume': '--load {ckpt_dir}', 'save_at': 10},
           'hardware': {'gpus': 1, 'memory_gib': 79, 'gpu_type': 'NVIDIA H100 80GB HBM3'},
           'verify': {'tolerance': 0.05}, 'search': {'budget': {'screens': 8, 'wall_hours': 1}}}
    (dst / 'infrarsi.yaml').write_text(yaml.safe_dump(doc, sort_keys=False), encoding='utf-8')
    return dst


def test_init_reads_the_scripts_knobs(tmp_path, home):
    root = tmp_path / 'job'
    root.mkdir()
    for n in ('train.py', 'config.json'):
        shutil.copyfile(HERE / n, root / n)
    res = cli.run('init', root)
    assert res.code == 0, res.text
    p = Project.load(root)
    assert {'bf16', 'compile', 'fused'} <= set(p.knobs), sorted(p.knobs)
    assert 'bs' in p.knobs or 'train_bs' in ''.join(p.knobs)


def test_doctor_runs_the_reference_with_phases(tmp_path, home):
    root = make_job(tmp_path / 'job')
    res = cli.run('doctor', root)
    assert res.code == 0, res.text
    assert 'reference run: OK' in res.out and '20 updates logged' in res.out
    assert 'where the time goes' in res.out, 'no phases from the profile hook on a real torch job'


def test_a_model_less_search_finds_a_gpu_gain_and_never_promotes_the_token_trap(tmp_path, home):
    root = make_job(tmp_path / 'job', traps=('seq',))
    res = cli.run('run', root, '--llm', 'none', '--strategy', 'tpe', '--screens', '8')
    assert res.code == 0, res.text
    r = check_run(latest_run(root), budget_screens=8)
    for c in r['candidates']:
        ch = c.get('change') or {}
        if ch.get('seq') == [1024, 512]:
            assert c['valid'] is False and 'same_workload' in (c.get('failed_checks') or []), c
    if r['outcome'] == 'improved':
        cfg = r['promotion']['config']
        assert cfg.get('seq', 1024) == 1024
        assert cfg.get('bf16') or cfg.get('compile') or cfg.get('fused') or cfg['bs'] != 2
        p = Project.load(root)
        runner = ProjectRunner(p, out_dir=root / '.infrarsi' / 'indep')
        task, v = p.task(), make_verifier(p)
        a = runner.run(task, p.reference_config(), None, 'rescore', 'i', candidate_id='ref')
        b = runner.run(task, cfg, None, 'rescore', 'i', candidate_id='cand')
        va, vb = v(a, None, a), v(b, None, b)
        assert va.valid and vb.valid and va.step_time / vb.step_time > 1.02
        print(f'\nclaimed x{r["promotion"]["speedup"]:.3f}, independent x{va.step_time / vb.step_time:.3f}, config {cfg}')
    print('\n' + report(latest_run(root)))


def test_skipped_optimizer_steps_are_rejected_by_the_step_counter(tmp_path, home):
    """F-SYS-14: a candidate that runs an optimizer step only every second update. The job logs `optim_step N`, so the
    counter differs from the reference's and the candidate fails optimizer_steps (the loss alone would not tell: on this
    job it barely moves)."""
    root = make_job(tmp_path / 'job', traps=('half_steps',))
    p = Project.load(root)
    runner = ProjectRunner(p, out_dir=root / '.infrarsi' / 't')
    task, v = p.task(), make_verifier(p)
    ref = runner.run(task, p.reference_config(), None, 'reference', 'r', candidate_id='c0')
    half = runner.run(task, dict(p.reference_config(), half_steps=True), None, 'screen', 'r', candidate_id='c1')
    verdict = v(half, ref, ref)
    items = {c['id']: c for c in verdict.checks}
    print('\n' + items['optimizer_steps']['detail'] + '\n' + items['loss_within_tolerance']['detail'])
    assert not verdict.valid and items['optimizer_steps']['status'] == 'fail' and 'optim_step' in items['optimizer_steps']['detail']


def test_without_a_step_counter_the_item_is_unchecked_and_doctor_says_so(tmp_path, home):
    """F-SYS-14 (fixed): with the learning rate alone the item is `unchecked`, never `pass`, and `doctor` warns; the user
    can make it a required item or expose the counter."""
    root = make_job(tmp_path / 'job', traps=('half_steps',))
    doc = yaml.safe_load((root / 'infrarsi.yaml').read_text(encoding='utf-8'))
    doc['launch']['command'] += ' --no-step-counter'
    (root / 'infrarsi.yaml').write_text(yaml.safe_dump(doc, sort_keys=False), encoding='utf-8')
    res = cli.run('doctor', root)
    assert res.code == 0 and 'no optimizer-step counter' in res.out, res.text
    p = Project.load(root)
    runner = ProjectRunner(p, out_dir=root / '.infrarsi' / 't')
    task, v = p.task(), make_verifier(p)
    ref = runner.run(task, p.reference_config(), None, 'reference', 'r', candidate_id='c0')
    half = runner.run(task, dict(p.reference_config(), half_steps=True), None, 'screen', 'r', candidate_id='c1')
    items = {c['id']: c for c in v(half, ref, ref).checks}
    assert items['optimizer_steps']['status'] == 'unchecked' and 'proxy' in items['optimizer_steps']['detail']
    strict = make_verifier(p)
    strict.required = (*strict.required, 'optimizer_steps')
    assert not strict(half, ref, ref).valid                        # required: the unchecked item vetoes
