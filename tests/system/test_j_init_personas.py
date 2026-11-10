"""J4 - `infrarsi init` on the repository shapes users bring, as a journey: the draft loads, `doctor` runs on it (where
a stub job can run on this machine), the notes say what to fill in, and a repository it cannot read is one line."""
import json

import pytest
import yaml

from _support import cli
from infrarsi.project import Project, detect

H100 = {'gpus': 8, 'memory_gib': 77.3, 'gpu_type': 'NVIDIA H100 80GB HBM3'}


@pytest.fixture(autouse=True)
def eight_gpus(monkeypatch):
    monkeypatch.setattr(detect, 'gpu_facts', lambda: dict(H100))


def _w(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


# a stub "HF Trainer" script: TrainingArguments in the text (what init looks for), HF-style dict logs, any flag accepted
HF = '''import argparse, math, time
from transformers import TrainingArguments, Trainer   # noqa: F401 (not imported at run time below)
def hf_entry():
    """The shape of a transformers example: HfArgumentParser over TrainingArguments (not run by the stub)."""
    from transformers import HfArgumentParser
    parser = HfArgumentParser((TrainingArguments,))
    training_args, = parser.parse_args_into_dataclasses()
    return Trainer(args=training_args)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_steps', type=int, default=30)
    ap.add_argument('--output_dir', default='out')
    a, rest = ap.parse_known_args()
    for s in range(1, a.max_steps + 1):
        time.sleep(0.01)
        print({'loss': round(2.5 * math.exp(-0.02 * s), 4), 'grad_norm': 1.1, 'learning_rate': 1e-5, 'epoch': s / 30}, flush=True)
    print({'train_runtime': 1.0, 'train_loss': 2.0}, flush=True)
main()
'''
HF_RUNNABLE = HF.replace('from transformers import TrainingArguments, Trainer   # noqa: F401 (not imported at run time below)',
                         'try:\n    from transformers import TrainingArguments, Trainer\nexcept ImportError:\n    TrainingArguments = Trainer = None')


def test_hf_trainer_shape_drafts_count_steps_and_doctor_runs(tmp_path, home):
    root = tmp_path / 'hf'
    _w(root / 'train_hf.py', HF_RUNNABLE)
    res = cli.run('init', root)
    assert res.code == 0, res.text
    assert 'HF Trainer (TrainingArguments output_dir, save_steps, resume_from_checkpoint): no checkpoint' in res.out, res.out
    p = Project.load(root)
    assert p.raw['framework'] in ('hf_trainer', 'pytorch'), p.raw['framework']
    doc = yaml.safe_load((root / 'infrarsi.yaml').read_text(encoding='utf-8'))
    doc['workload'] = {'steps': 12, 'window': [5, 12], 'invariants': doc.get('workload', {}).get('invariants', {})}
    doc['hardware'] = {'gpus': 1}
    doc['launch']['command'] = doc['launch']['command'].replace('torchrun --nproc_per_node {gpus}', '{python}')
    (root / 'infrarsi.yaml').write_text(yaml.safe_dump(doc, sort_keys=False), encoding='utf-8')
    res = cli.run('doctor', root)
    assert 'Traceback' not in res.text, res.text
    assert p.raw['signals'].get('count_steps') is True, 'init did not draft count_steps for HF Trainer dict logs'
    assert res.out.startswith('reference run: OK') and 'loss 12' in res.out, res.out


def test_launch_script_shape(tmp_path, home):
    root = tmp_path / 'sh'
    _w(root / 'train.py', 'import argparse\nap = argparse.ArgumentParser()\nap.add_argument("--micro-batch-size", type=int, default=2)\n'
                          'ap.add_argument("--bf16", action="store_true")\nap.add_argument("--max-steps", type=int, default=30)\nprint(ap.parse_args())\n')
    _w(root / 'run.sh', '#!/bin/bash\nMBS=${MBS:-4}\ntorchrun --nproc_per_node 8 train.py --micro-batch-size $MBS --bf16 --max-steps 100 "$@"\n')
    res = cli.run('init', root, '--entry', 'run.sh')
    assert res.code == 0, res.text
    p = Project.load(root)
    assert 'run.sh' in p.raw['launch']['command'] and '{args}' in p.raw['launch']['command']
    assert 'micro_batch_size' in p.knobs or 'bf16' in p.knobs, sorted(p.knobs)
    notes = [ln for ln in res.out.splitlines() if ln.startswith('  - ')]
    assert notes, 'init printed no notes for a launch-script repository'


def test_torchtitan_toml_and_deepspeed_json_shapes_load(tmp_path, home):
    tt = tmp_path / 'titan'
    _w(tt / 'train.py', 'import argparse, torch\nfrom torchtitan.config_manager import JobConfig\nJobConfig()\n')
    _w(tt / 'train_configs' / 'llama3_8b.toml', '[training]\nbatch_size = 2\nseq_len = 8192\ncompile = false\n[activation_checkpoint]\nmode = "selective"\n')
    _w(tt / 'run_train.sh', '#!/bin/bash\nCONFIG_FILE=${CONFIG_FILE:-"./train_configs/llama3_8b.toml"}\n'
                            'torchrun --nproc_per_node 8 train.py --job.config_file ${CONFIG_FILE} "$@"\n')
    res = cli.run('init', tt, '--entry', 'run_train.sh')
    assert res.code == 0, res.text
    p = Project.load(tt)
    assert any('--' in str(k.get('flag', '')) and '.' in str(k.get('flag', '')) for k in p.knobs.values()) or p.knobs, p.knobs
    ds = tmp_path / 'ds'
    _w(ds / 'train.py', 'import argparse, deepspeed\nap = argparse.ArgumentParser()\nap.add_argument("--deepspeed_config", default="ds_config.json")\n'
                        'ap.add_argument("--max-steps", type=int, default=30)\nprint(ap.parse_args())\n')
    _w(ds / 'ds_config.json', json.dumps({'train_micro_batch_size_per_gpu': 2, 'gradient_accumulation_steps': 4, 'bf16': {'enabled': False},
                                          'zero_optimization': {'stage': 2, 'overlap_comm': False}}))
    res = cli.run('init', ds)
    assert res.code == 0, res.text
    p = Project.load(ds)
    assert p.config_file and p.config_file.suffix == '.json'
    assert any('path' in k for k in p.knobs.values()), p.knobs


def test_a_repository_init_cannot_read_is_one_line(tmp_path, home):
    empty = tmp_path / 'empty'
    empty.mkdir()
    res = cli.run('init', empty)
    assert res.code == 2 and res.err.startswith('infrarsi init:') and 'Traceback' not in res.err, res.text
    assert res.err.count('\n') == 1
    docs = tmp_path / 'docs'
    _w(docs / 'README.md', '# nothing to train\n')
    res = cli.run('init', docs)
    assert res.code == 2 and 'Traceback' not in res.err


def test_init_on_a_script_with_a_syntax_error_names_it(tmp_path, home):
    root = tmp_path / 'broken'
    _w(root / 'train.py', 'import torch\ndef main(:\n    pass\n')
    res = cli.run('init', root)
    assert 'Traceback' not in res.text, res.text
    assert res.code in (0, 2)
    if res.code == 0:
        assert 'syntax' in res.out.lower() or 'could not' in res.out.lower(), res.out


def test_drafted_manifest_has_the_documented_sections(tmp_path, home):
    from _support.toy import make_toy
    root = make_toy(tmp_path / 'toy', manifest=False)
    assert cli.run('init', root).code == 0
    doc = yaml.safe_load((root / 'infrarsi.yaml').read_text(encoding='utf-8'))
    assert doc['schema'] == 'infrarsi-project-v1'
    for key in ('name', 'launch', 'knobs', 'workload', 'hardware', 'code'):
        assert key in doc, key
    assert doc['hardware']['gpu_type'] == H100['gpu_type'] and doc['hardware']['gpus'] >= 1    # a plain script: 1 GPU
    text = (root / 'infrarsi.yaml').read_text(encoding='utf-8')
    assert text.lstrip().startswith('#'), 'the draft has no notes at the top'
    assert 'mutable' in doc['code'] and isinstance(doc['code']['mutable'], list)


def test_toy_draft_checks_resume_eval_and_tests_at_promotion(tmp_path, home):
    """U-4-5: on the toy without its manifest, init drafts the hand-written checkpoint, verify.eval and verify.tests,
    doctor accepts the draft, and the promotion checks all three items (none stays unchecked)."""
    from _support.consistency import check_run
    from _support.toy import TOY, latest_run, make_toy, result
    root = make_toy(tmp_path / 'toy', manifest=False)
    assert cli.run('init', root).code == 0
    doc = yaml.safe_load((root / 'infrarsi.yaml').read_text(encoding='utf-8'))
    hand = yaml.safe_load((TOY / 'infrarsi.yaml').read_text(encoding='utf-8'))
    assert doc['checkpoint'] == {**hand['checkpoint'], 'save_at': doc['workload']['steps'] // 2}
    assert (doc['verify']['eval'], doc['verify']['tests']) == (hand['verify']['eval'], hand['verify']['tests'])
    res = cli.run('doctor', root)
    assert res.code == 0 and 'reference run: OK' in res.out, res.text
    # one knob whose change is a gain (the fused optimizer step), so that the search has a candidate to promote
    doc['knobs'] = {'fused_optimizer': doc['knobs']['fused_optimizer']}
    doc['workload']['invariants'] = {}
    (root / 'infrarsi.yaml').write_text(yaml.safe_dump(doc, sort_keys=False), encoding='utf-8')
    res = cli.run('run', root, '--llm', 'none', '--strategy', 'random', '--screens', '2')
    assert res.code == 0, res.text
    run = latest_run(root)
    check_run(run, budget_screens=2)
    tried = result(run)['promotion']['tried']
    items = {i['id']: i['status'] for i in tried[0]['items']}
    assert items['resume_continues'] == items['eval_matches'] == items['user_tests'] == 'pass', tried[0]['items']
