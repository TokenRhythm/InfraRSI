"""Phase D round 1 fixes in the project layer (2026-10-24): a constraint naming no knob is a manifest error (F-D1-12-3);
a failure names the log's error line (F-D1-09-1); CRLF launch scripts are pointed at (F-D1-12-1); a progress bar's
0/N line is no update and init drafts the diffusers training bar (F-D1-05-1); init says when it ran without GPUs or
without Megatron's sources (F-D1-12-2, F-D1-16-1) and takes the job's command line (F-D1-02-1)."""
import pytest
import yaml

from infrarsi.cli.main import main
from infrarsi.harness import signing
from infrarsi.project import Project, detect, signals
from infrarsi.project.manifest import ManifestError
from infrarsi.verify import SCHEMA, Verifier

BASE = {'schema': 'infrarsi-project-v1', 'name': 't', 'launch': {'command': 'bash run.sh --steps {steps} {args}'},
        'knobs': {'mbs': {'type': 'int', 'choices': [1, 2], 'flag': '--mbs', 'default': 1}}}


def test_a_constraint_or_invariant_naming_no_knob_is_a_manifest_error(tmp_path):
    (tmp_path / 'run.sh').write_text('echo\n', encoding='utf-8')
    with pytest.raises(ManifestError, match="unknown name 'tensor_model_parallel_size'.*remove the constraint"):
        Project(dict(BASE, constraints=['gpus % (tensor_model_parallel_size * mbs) == 0']), tmp_path / 'infrarsi.yaml')
    with pytest.raises(ManifestError, match="workload.invariants.gb: unknown name 'grad_accum'"):
        Project(dict(BASE, workload={'steps': 30, 'invariants': {'gb': 'mbs * grad_accum * gpus'}}), tmp_path / 'infrarsi.yaml')
    Project(dict(BASE, constraints=['gpus % mbs == 0', 'max(mbs, 1) <= 8']), tmp_path / 'infrarsi.yaml')   # fine


def _receipt(lines, exit_code):
    ev = {'steps': 30, 'steps_completed': 0, 'exit_code': exit_code, 'per_step': [], 'window': [11, 30],
          'events': signals.scan_events([(0.0, x) for x in lines]), 'frozen': {'checked': False}}
    return signing.sign({'schema': SCHEMA, 'attempt_id': 'a', 'evidence': ev})


def test_a_failure_names_the_logs_error_line():
    lines = ['[rank1]: Traceback (most recent call last):', '[rank1]:   File "x.py", line 3, in <module>',
             '[rank1]: torch.distributed.DistNetworkError: The server socket has failed to listen: address already in use',
             'torch.distributed.elastic.multiprocessing.errors.ChildFailedError: ']
    v = Verifier()(_receipt(lines, 1), None, None)
    detail = {c['id']: c['detail'] for c in v.checks}['completes']
    assert 'the log: torch.distributed.DistNetworkError: The server socket' in detail
    shell = ['run.sh: line 18: /venv/bin/python\r: No such file or directory']
    assert signals.scan_events([(0.0, x) for x in shell])['error'].startswith('run.sh: line 18:')


def test_crlf_launch_scripts_are_pointed_at(tmp_path):
    (tmp_path / 'run.sh').write_bytes(b'#!/bin/bash\r\nexport A=1\r\npython train.py "$@"\r\n')
    p = Project(dict(BASE, execution={'where': 'ssh', 'hosts': ['h']}), tmp_path / 'infrarsi.yaml')
    assert p.crlf_scripts() == ['run.sh'] and p.posix_hosts()
    (tmp_path / 'run.sh').write_bytes(b'#!/bin/bash\npython train.py "$@"\n')
    assert p.crlf_scripts() == []


def _bar(n, loss=None, total=3):
    return f'Steps:  {100 * n // total}%|#  | {n}/{total} [00:0{n}<00:01,  1.0s/it' + (f', lr=1e-05, step_loss={loss}]' if loss else ']')


def test_a_progress_bar_start_is_no_update_and_init_drafts_the_training_bar(tmp_path, monkeypatch):
    # D1-05d's log: the bar advances before the new postfix is set, so each count is drawn with the last loss first
    lines = ['Resolving data files: 100%|##########| 197/197 [00:00<00:00, 1.2it/s]', 'Steps:   0%|          | 0/3 [00:00<?, ?it/s]',
             _bar(1), _bar(1, 0.4), _bar(2, 0.4), _bar(2, 0.3), _bar(3, 0.3), _bar(3, 10.0), _bar(3, 10.0)]
    rows = signals.LogParser(detect.TQDM_STEPS, None, False).parse([(float(i), x) for i, x in enumerate(lines)])
    assert [(r['step'], r['loss'], r['t']) for r in rows] == [(1, 0.4, 2.0), (2, 0.3, 4.0), (3, 10.0, 6.0)]
    # under accumulation the postfix is set after each micro-batch: the 0/N line has values and is still no update
    acc = ['Steps:   0%|          | 0/3 [00:00<?, ?it/s, lr=1e-05, step_loss=0.5]', _bar(1, 0.5), _bar(1, 0.4), _bar(2, 0.4),
           _bar(2, 0.3), _bar(3, 0.3), _bar(3, 0.2)]
    rows = signals.LogParser(detect.TQDM_STEPS, None, False).parse([(float(i), x) for i, x in enumerate(acc)])
    assert [r['step'] for r in rows] == [1, 2, 3]
    # a last loss that is not finite replaces the redraw's stale finite one (the verifier then rejects the run)
    rows = signals.LogParser(detect.TQDM_STEPS, None, False).parse([(float(i), x) for i, x in enumerate(
        [_bar(1, 1.0), _bar(2, 1.0), _bar(3, 1.0), _bar(3, 'nan')])])
    assert [r['loss'] for r in rows] == [1.0, 1.0, 'nan']
    # a 0-based trainer whose step 0 carries no scalar keeps it (the rebase counts 4 updates)
    plain = signals.LogParser(None, None, False).parse([(0.0, 'step 0'), (1.0, 'step 1 loss 2.0'), (2.0, 'step 2 loss 1.9'),
                                                        (3.0, 'step 3 loss 1.8')])
    assert [r['step'] for r in signals._rebase(plain)[0]] == [1, 2, 3, 4]
    monkeypatch.setattr(detect, 'gpu_facts', lambda: {'gpus': 8, 'memory_gib': 77.3, 'gpu_type': 'NVIDIA H100 80GB HBM3'})
    (tmp_path / 'train_t2i.py').write_text(
        'import argparse\nfrom accelerate import Accelerator\nfrom tqdm.auto import tqdm\n'
        'def main():\n    ap = argparse.ArgumentParser()\n    ap.add_argument("--max_train_steps", type=int)\n'
        '    ap.add_argument("--train_batch_size", type=int, default=16)\n    a = ap.parse_args()\n'
        '    progress_bar = tqdm(range(a.max_train_steps), desc="Steps")\n    logs = {"step_loss": 0.1, "lr": 1e-5}\n'
        '    progress_bar.set_postfix(**logs)\nif __name__ == "__main__":\n    main()\n', encoding='utf-8')
    m, notes = detect.draft(tmp_path)
    assert m['signals']['patterns'] == detect.TQDM_STEPS and any('tqdm bar' in n for n in notes)


def test_init_says_when_it_ran_without_gpus_or_megatron_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(detect, 'gpu_facts', lambda: {'gpus': 1, 'memory_gib': None, 'gpu_type': None})
    (tmp_path / 'run.sh').write_text('#!/bin/bash\nMEGATRON=/nonexistent/Megatron-LM\ntorchrun --nproc-per-node 8 '
                                     '$MEGATRON/pretrain_gpt.py --micro-batch-size 1 --train-iters 30 "$@"\n', encoding='utf-8')
    m, notes = detect.draft(tmp_path, entry='run.sh')
    assert any(n.startswith('No GPU on this machine') for n in notes)
    assert any("Megatron's argument definitions" in n for n in notes)


def test_init_takes_the_jobs_command(tmp_path, monkeypatch):
    monkeypatch.setattr(detect, 'gpu_facts', lambda: {'gpus': 8, 'memory_gib': 77.3, 'gpu_type': 'NVIDIA H100 80GB HBM3'})
    (tmp_path / 'train_hf.py').write_text(
        'import torch.distributed\nfrom transformers import HfArgumentParser, Trainer, TrainingArguments\n'
        'def main():\n    targs, = HfArgumentParser((TrainingArguments,)).parse_args_into_dataclasses()\n'
        'if __name__ == "__main__":\n    main()\n', encoding='utf-8')
    main(['init', str(tmp_path), '--command', 'torchrun --nproc_per_node 8 train_hf.py --output_dir out --max_steps 30 '
          '--per_device_train_batch_size 1 --gradient_accumulation_steps 4 --bf16'])
    assert (tmp_path / 'infrarsi_launch.sh').read_text(encoding='utf-8').rstrip().endswith('--bf16 "$@"')
    knobs = yaml.safe_load((tmp_path / 'infrarsi.yaml').read_text(encoding='utf-8'))['knobs']
    assert (knobs['per_device_train_batch_size']['default'], knobs['gradient_accumulation_steps']['default'],
            knobs['bf16']['default']) == (1, 4, True)
    with pytest.raises(SystemExit):                                     # the launch script is not replaced silently
        main(['init', str(tmp_path), '--command', 'torchrun train_hf.py'])


@pytest.mark.parametrize('command', ['torchrun train_hf.py --max_steps 30 | tee train.log', 'cd x && python train_hf.py',
                                     'bash run.sh --lr 1e-5', 'CUDA_VISIBLE_DEVICES=0 ./run.sh', '/bin/bash run.sh',
                                     'CUDA_VISIBLE_DEVICES=0 /bin/sh run.sh'])
def test_init_refuses_commands_the_search_cannot_append_to(tmp_path, command, capsys):
    with pytest.raises(SystemExit):
        main(['init', str(tmp_path), '--command', command])
    assert not (tmp_path / 'infrarsi_launch.sh').exists() and '--entry' in capsys.readouterr().err


@pytest.mark.parametrize('command', ['python train.py --separator "|"', "python train.py --separator ';'",
                                     'python train.py --a="x;y" > train.log 2>&1', r'python train.py --sep \|'])
def test_init_takes_quoted_operators_and_redirections(tmp_path, command, monkeypatch):
    monkeypatch.setattr(detect, 'gpu_facts', lambda: {'gpus': 1, 'memory_gib': 77.3, 'gpu_type': 'NVIDIA H100 80GB HBM3'})
    (tmp_path / 'train.py').write_text('print(1)\n', encoding='utf-8')
    main(['init', str(tmp_path), '--command', command])
    assert (tmp_path / 'infrarsi_launch.sh').read_text(encoding='utf-8').rstrip().endswith(command + ' "$@"')


def test_init_keeps_the_commands_optimizer(tmp_path, monkeypatch):
    monkeypatch.setattr(detect, 'gpu_facts', lambda: {'gpus': 8, 'memory_gib': 77.3, 'gpu_type': 'NVIDIA H100 80GB HBM3'})
    monkeypatch.setattr(detect, 'hf_defaults', lambda: {'optim': 'adamw_torch', 'bf16': False})
    (tmp_path / 'train_hf.py').write_text(
        'from transformers import HfArgumentParser, Trainer, TrainingArguments\n'
        'targs, = HfArgumentParser((TrainingArguments,)).parse_args_into_dataclasses()\n', encoding='utf-8')
    main(['init', str(tmp_path), '--command', 'python train_hf.py --output_dir out --optim adamw_torch_fused'])
    knobs = yaml.safe_load((tmp_path / 'infrarsi.yaml').read_text(encoding='utf-8'))['knobs']
    assert knobs['optim']['default'] == 'adamw_torch_fused'
