"""Round-2 trials R2-05 / R2-04 (2026-10-24): a launcher's own flags were taken for the training script's. Its
accelerate config became the job's config (`--config_file {config}` in the launch, a zero_stage knob) and the reference
died on `unrecognized arguments`; an option loop was read as positional arguments and its argument array as no "$@".
The DreamBooth scripts' training bar says `loss=` (text-to-image: `step_loss=`)."""
from infrarsi.project import detect, signals
from infrarsi.project import report as report_mod
from infrarsi.project.flow import run_project

from test_project import toy  # noqa: F401 - the toy project fixture

LAUNCHER = '''#!/bin/bash
ACCEL=zero3
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --accel) ACCEL=$2; shift 2;;
    *) ARGS+=("$1"); shift;;
  esac
done
exec accelerate launch --config_file accel/$ACCEL.yaml --num_processes 8 --main_process_port 29651 \\
  train_db.py --pretrained_model_name_or_path /models/flux --train_batch_size 1 --mixed_precision bf16 "${ARGS[@]}"
'''
SCRIPT = '''import argparse
from tqdm.auto import tqdm
from accelerate import Accelerator
ap = argparse.ArgumentParser()
ap.add_argument("--pretrained_model_name_or_path")
ap.add_argument("--train_batch_size", type=int, default=4)
ap.add_argument("--mixed_precision", default="no")
ap.add_argument("--max_train_steps", type=int)
args = ap.parse_args()
progress_bar = tqdm(range(args.max_train_steps), desc="Steps")
logs = {"loss": 0.1, "lr": 1e-5}
progress_bar.set_postfix(**logs)
'''


def test_a_launchers_own_config_is_not_the_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(detect, 'gpu_facts', lambda: {'gpus': 8, 'memory_gib': 77.3, 'gpu_type': 'NVIDIA H100 80GB HBM3'})
    (tmp_path / 'accel').mkdir()
    (tmp_path / 'accel' / 'zero3.yaml').write_text('distributed_type: DEEPSPEED\ndeepspeed_config:\n  zero_stage: 3\n', encoding='utf-8')
    (tmp_path / 'run.sh').write_text(LAUNCHER, encoding='utf-8')
    (tmp_path / 'train_db.py').write_text(SCRIPT, encoding='utf-8')
    m, notes = detect.draft(tmp_path, entry='run.sh')
    cmd = m['launch']['command']
    assert '--config_file' not in cmd and '{config}' not in cmd and cmd.startswith('bash run.sh --max_train_steps {steps}')
    assert 'zero_stage' not in m['knobs'] and m['knobs']['train_batch_size']['default'] == 1
    assert not any('"$@"' in n or 'positional' in n for n in notes)   # its option loop forwards the rest
    assert m['hardware']['gpus'] == 8                                      # the launcher's --num_processes still counts
    assert m['signals']['patterns'] == detect.TQDM_STEPS                   # the DreamBooth bar: `loss=`


def test_both_bar_postfixes_are_read():
    p = signals.LogParser(detect.TQDM_STEPS, None, False)
    rows = p.parse([(0.0, 'Steps:  33%|#  | 1/3 [00:01<00:02,  1.0s/it, loss=0.5, lr=1e-05]'),
                    (1.0, 'Steps:  67%|#  | 2/3 [00:02<00:01,  1.0s/it, lr=1e-05, step_loss=0.3]')])
    assert [r['loss'] for r in rows] == [0.5, 0.3]


def test_an_unpriced_apis_usd_is_not_reported_as_zero(toy, tmp_path):
    # D2-16-gpt56terra (a gateway without llm.price): "USD 0" read as free
    res = run_project(toy, strategy='random', budget={'screens': 1}, out_root=tmp_path / 'out', log=lambda m: None)
    assert res['cost']['llm_usd_known'] is True and 'USD not known' not in report_mod.markdown(res)
    res['cost']['llm_usd_known'] = False
    assert 'USD not known (no llm.price for this model)' in report_mod.markdown(res)


WAN = '''import torch, os, argparse, accelerate
from diffsynth.core import UnifiedDataset
from diffsynth.diffusion import *


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--max_steps", type=int, default=None)
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    print(parser)
'''


def test_a_parser_built_by_library_helpers_is_named(tmp_path, monkeypatch):
    # R2-04 (DiffSynth): "No known performance knob" while the parser takes some 60 arguments from the library
    monkeypatch.setattr(detect, 'gpu_facts', lambda: {'gpus': 8, 'memory_gib': 77.3, 'gpu_type': 'NVIDIA H100 80GB HBM3'})
    (tmp_path / 'train_wan.py').write_text(WAN, encoding='utf-8')
    assert detect.outside_parser_args(tmp_path / 'train_wan.py') == ['add_general_config (diffsynth.diffusion)',
                                                                      'add_video_size_config (diffsynth.diffusion)']
    m, notes = detect.draft(tmp_path, entry='train_wan.py')
    assert any('hands its argument parser to add_general_config (diffsynth.diffusion), add_video_size_config' in n for n in notes)


def test_the_launchers_config_at_the_root_is_not_the_jobs(tmp_path, monkeypatch):
    # DF4_r1 #1: accelerate.yaml next to the launcher, named by its --config_file, is skipped by the fallback scan too
    monkeypatch.setattr(detect, 'gpu_facts', lambda: {'gpus': 8, 'memory_gib': 77.3, 'gpu_type': 'NVIDIA H100 80GB HBM3'})
    (tmp_path / 'accelerate.yaml').write_text('distributed_type: DEEPSPEED\ndeepspeed_config:\n  zero_stage: 3\n', encoding='utf-8')
    (tmp_path / 'run.sh').write_text('#!/bin/bash\naccelerate launch --config_file accelerate.yaml --num_processes 8 train_db.py '
                                     '--train_batch_size 1 "$@"\n', encoding='utf-8')
    (tmp_path / 'train_db.py').write_text(SCRIPT, encoding='utf-8')
    m, _ = detect.draft(tmp_path, entry='run.sh')
    assert '{config}' not in m['launch']['command'] and 'zero_stage' not in m['knobs']
    assert not (m.get('config') or {}).get('file')


def test_a_flag_the_launcher_sets_keeps_the_scripts_choices(tmp_path, monkeypatch):
    # F-D2-R205-3: `--mixed_precision bf16` from the launcher, choices no / fp16 / bf16 in the script (not bf16 / fp32)
    monkeypatch.setattr(detect, 'gpu_facts', lambda: {'gpus': 8, 'memory_gib': 77.3, 'gpu_type': 'NVIDIA H100 80GB HBM3'})
    (tmp_path / 'accel').mkdir()
    (tmp_path / 'accel' / 'zero3.yaml').write_text('distributed_type: DEEPSPEED\n', encoding='utf-8')
    (tmp_path / 'run.sh').write_text(LAUNCHER, encoding='utf-8')
    (tmp_path / 'train_db.py').write_text(SCRIPT.replace('default="no")', 'default=None, choices=["no", "fp16", "bf16"])'),
                                          encoding='utf-8')
    m, _ = detect.draft(tmp_path, entry='run.sh')
    assert m['knobs']['mixed_precision']['choices'] == ['no', 'fp16', 'bf16'] and m['knobs']['mixed_precision']['default'] == 'bf16'
