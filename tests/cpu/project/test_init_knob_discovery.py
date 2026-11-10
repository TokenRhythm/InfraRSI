"""`infrarsi init` finds the knobs a job exposes outside its entry script's own parser (2026-10-26): arguments a library
or a module of the repository adds (DiffSynth's add_general_config: D2-R204, Wan), HfArgumentParser's dataclasses, the
options a launch script parses itself (D2-R205b, FLUX's --accel), options of unknown name with speed-setting values, the
registry's env vars a library reads, Megatron flags of the levers the shipped packs measured (M8 had no bf16 main grads),
and it names the measured levers no knob of the draft reaches."""
import importlib

import pytest

from infrarsi.experience.store import measured, unreached
from infrarsi.project import Project, detect
from infrarsi.project.manifest import args_for, env_for

H100 = {'gpus': 8, 'memory_gib': 77.3, 'gpu_type': 'NVIDIA H100 80GB HBM3'}


@pytest.fixture(autouse=True)
def eight_gpus(monkeypatch):
    monkeypatch.setattr(detect, 'gpu_facts', lambda: dict(H100))


def _w(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def _load(root, m, notes):
    return Project.load(detect.write(root, m, notes, force=True))


WAN = ("import torch, os, argparse, accelerate\nfrom diffsynth.diffusion import *\n\n\ndef wan_parser():\n"
       "    parser = argparse.ArgumentParser()\n    parser = add_general_config(parser)\n"
       "    parser = add_video_size_config(parser)\n    parser.add_argument('--max_steps', type=int, default=None)\n"
       "    return parser\n\n\nif __name__ == '__main__':\n    args = wan_parser().parse_args()\n"
       "    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)\n")
PARSERS = ("def add_general_config(parser):\n"
           "    parser.add_argument('--dataset_num_workers', type=int, default=0)\n"
           "    parser.add_argument('--use_gradient_checkpointing', default=False, action='store_true')\n"
           "    parser.add_argument('--use_gradient_checkpointing_offload', default=False, action='store_true')\n"
           "    parser.add_argument('--learning_rate', type=float, default=1e-4)\n"
           "    parser = add_output_config(parser)\n    return parser\n"
           "def add_output_config(parser):\n    parser.add_argument('--output_path', type=str, default='./models')\n"
           "    return parser\n"
           "def add_video_size_config(parser):\n    parser.add_argument('--height', type=int, default=480)\n    return parser\n")


def test_a_librarys_parser_helpers_are_read_from_its_installed_source(tmp_path, monkeypatch):
    # D2-R204 (Wan, DiffSynth): "No known performance knob"; the parser's arguments come from library helpers, and the
    # library reads DIFFSYNTH_ATTENTION_IMPLEMENTATION. The library is read, not imported (its __init__ would fail here).
    site = tmp_path / 'site'
    _w(site / 'diffsynth' / '__init__.py', 'raise ImportError("never imported")\n'
                                            'ATTN = os.environ.get("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")\n')
    _w(site / 'diffsynth' / 'diffusion' / '__init__.py', 'from .parsers import *\n')
    _w(site / 'diffsynth' / 'diffusion' / 'parsers.py', PARSERS)
    _w(tmp_path / 'repo' / 'train_wan.py', WAN)
    monkeypatch.syspath_prepend(str(site))
    importlib.invalidate_caches()
    m, notes = detect.draft(tmp_path / 'repo', entry='train_wan.py')
    k = m['knobs']
    assert k['dataset_num_workers'] == {'type': 'int', 'flag': '--dataset_num_workers', 'choices': [0, 2, 4, 8], 'default': 0}
    assert k['use_gradient_checkpointing']['type'] == 'bool' and k['use_gradient_checkpointing_offload']['type'] == 'bool'
    assert not {'learning_rate', 'output_path', 'height'} & set(k)          # not speed settings
    # no default: unset, DiffSynth picks the first importable of FA4 > FA3 > FA2 > ... > torch; `torch` as the reference
    # would make restoring the library's own choice look like a gain
    assert k['DIFFSYNTH_ATTENTION_IMPLEMENTATION'] == {'type': 'str', 'env': 'DIFFSYNTH_ATTENTION_IMPLEMENTATION', 'choices': [
        'torch', 'flash_attention_2', 'flash_attention_4', 'xformers']}
    assert not any('hands its argument parser' in n for n in notes)
    # a measured lever whose registry flag this job's parser does not define: a patch, not "add --allow_tf32"
    assert any(n.startswith('Not reached by any knob: tf32 off -> on') and "the job's parser has no --allow_tf32" in n
               for n in notes)
    p = _load(tmp_path / 'repo', m, notes)
    assert 'DIFFSYNTH_ATTENTION_IMPLEMENTATION' not in env_for(p, p.reference_config())
    assert env_for(p, {**p.reference_config(), 'DIFFSYNTH_ATTENTION_IMPLEMENTATION': 'xformers'})[
        'DIFFSYNTH_ATTENTION_IMPLEMENTATION'] == 'xformers'


def test_a_local_argument_module_and_options_of_unknown_name(tmp_path):
    # `args = parse_args()` from a module of the repository; `--strategy ddp|fsdp` is a speed setting, `--optim_name
    # adamw|sgd` is not; a switch named like one (`--channels_last`) is a knob
    _w(tmp_path / 'opts.py', "import argparse\ndef parse_args():\n    p = argparse.ArgumentParser()\n"
                             "    p.add_argument('--num_workers', type=int, default=2)\n"
                             "    p.add_argument('--strategy', default='ddp', choices=['ddp', 'fsdp'])\n"
                             "    p.add_argument('--optim_name', default='adamw', choices=['adamw', 'sgd'])\n"
                             "    p.add_argument('--lr_scheduler', default='cosine', choices=['cosine', 'linear'])\n"
                             "    p.add_argument('--channels_last', action='store_true')\n"
                             "    p.add_argument('--cross_entropy_loss_fusion', action='store_true')\n"
                             # model and data settings that share a word with speed settings
                             "    p.add_argument('--zero_init_residual', action='store_true')\n"
                             "    p.add_argument('--use_bucketing', action='store_true')\n"
                             "    p.add_argument('--attention_type', default='full', choices=['full', 'linear'])\n"
                             "    p.add_argument('--fusion_method', default='concat', choices=['concat', 'attention'])\n"
                             "    p.add_argument('--padding', default='zero', choices=['zero', 'reflect'])\n"
                             # another device's kernel; work outside the training step
                             "    p.add_argument('--enable_npu_flash_attention', action='store_true')\n"
                             "    p.add_argument('--prior_generation_precision', default=None, choices=['fp32', 'bf16'])\n"
                             "    p.add_argument('--eval_compile', action='store_true')\n"
                             "    return p.parse_args()\n")
    _w(tmp_path / 'train.py', 'import torch\nfrom opts import parse_args\nargs = parse_args()\n')
    m, notes = detect.draft(tmp_path, entry='train.py')
    k = m['knobs']
    assert k['num_workers']['choices'] == [0, 2, 4, 8]
    assert k['strategy'] == {'type': 'str', 'flag': '--strategy', 'choices': ['ddp', 'fsdp'], 'default': 'ddp'}
    assert k['channels_last'] == {'type': 'bool', 'flag': '--channels_last', 'default': False}
    assert k['cross_entropy_loss_fusion']['type'] == 'bool'
    assert not {'optim_name', 'lr_scheduler', 'zero_init_residual', 'use_bucketing', 'attention_type', 'fusion_method',
                'padding', 'enable_npu_flash_attention', 'prior_generation_precision', 'eval_compile'} & set(k)


def test_hf_argument_parser_dataclasses_of_the_script(tmp_path):
    _w(tmp_path / 'sft.py', "from dataclasses import dataclass, field\n"
                            "from transformers import HfArgumentParser, TrainingArguments, Trainer\n"
                            "@dataclass\nclass ScriptArguments:\n    use_liger: bool = False\n"
                            "    attn_implementation: str = field(default='sdpa', metadata={'choices': ['eager', 'sdpa', "
                            "'flash_attention_2']})\n    dataset_name: str = 'x'\n"
                            "    compile_mode: Literal['default', 'max-autotune'] = 'default'\n\n"
                            "if __name__ == '__main__':\n    script_args, training_args = HfArgumentParser("
                            "(ScriptArguments, TrainingArguments)).parse_args_into_dataclasses()\n")
    m, notes = detect.draft(tmp_path, entry='sft.py')
    k = m['knobs']
    assert k['use_liger'] == {'type': 'bool', 'flag': '--use_liger', 'default': False, 'style': 'value'}
    assert k['attn_implementation']['choices'] == ['eager', 'sdpa', 'flash_attention_2'] and 'dataset_name' not in k
    assert k['compile_mode'] == {'type': 'str', 'flag': '--compile_mode', 'choices': ['default', 'max-autotune'],
                                 'default': 'default'}
    p = _load(tmp_path, m, notes)
    assert '--use_liger' in args_for(p, {**p.reference_config(), 'use_liger': True})


FLUX = """#!/bin/bash
# --accel picks the accelerate config (accel/<name>.yaml)
ACCEL=zero3
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --accel) ACCEL=$2; shift 2;;
    *) ARGS+=("$1"); shift;;
  esac
done
exec accelerate launch --config_file accel/$ACCEL.yaml --num_processes 8 train_dreambooth_flux.py --resolution 1024 \\
  --mixed_precision bf16 --dataloader_num_workers 2 "${ARGS[@]}"
"""


def test_an_option_the_launch_script_parses_itself(tmp_path):
    # D2-R205b (FLUX): the sharding lever was a wrapper option the user added by hand, the last knob, never tried
    for name in ('zero2', 'zero3', 'fsdp'):
        _w(tmp_path / 'accel' / f'{name}.yaml', 'distributed_type: DEEPSPEED\n')
    _w(tmp_path / 'train_dreambooth_flux.py', "import argparse\nfrom accelerate import Accelerator\n"
                                              "p = argparse.ArgumentParser()\np.add_argument('--max_train_steps', type=int)\n"
                                              "p.add_argument('--mixed_precision', default=None, choices=['no', 'fp16', 'bf16'])\n"
                                              "p.add_argument('--dataloader_num_workers', type=int, default=0)\nargs = p.parse_args()\n")
    _w(tmp_path / 'run_flux.sh', FLUX)
    m, notes = detect.draft(tmp_path, entry='run_flux.sh')
    assert m['knobs']['accel'] == {'type': 'str', 'flag': '--accel', 'choices': ['fsdp', 'zero2', 'zero3'], 'default': 'zero3'}
    p = _load(tmp_path, m, notes)
    assert args_for(p, {**p.reference_config(), 'accel': 'fsdp'})[-2:] == ['--accel', 'fsdp']
    # the values of a `case` block over the option's variable
    _w(tmp_path / 'b' / 'train.py', "import argparse\np = argparse.ArgumentParser()\np.add_argument('--max-steps', type=int)\n"
                                    "args = p.parse_args()\n")
    _w(tmp_path / 'b' / 'go.sh', '#!/bin/bash\nMODE=eager\nwhile [ $# -gt 0 ]; do\n  case "$1" in\n'
                                 '    --mode) MODE=$2; shift 2;;\n    *) shift;;\n  esac\ndone\n'
                                 'case "$MODE" in\n  eager) C=0;;\n  compiled|inductor) C=1;;\nesac\n'
                                 'COMPILE=$C torchrun --nproc_per_node 8 train.py "$@"\n')
    m, _ = detect.draft(tmp_path / 'b', entry='go.sh')
    assert m['knobs']['mode']['choices'] == ['compiled', 'eager', 'inductor'] and m['knobs']['mode']['default'] == 'eager'
    # `--accel|-a) shift; ACCEL=$1` and a default given where the value is used
    flux = FLUX.replace('ACCEL=zero3\n', '').replace('--accel) ACCEL=$2; shift 2;;', '--accel|-a) shift; ACCEL=$1; shift;;')
    _w(tmp_path / 'run_flux.sh', flux.replace('accel/$ACCEL.yaml', 'accel/${ACCEL:-zero3}.yaml'))
    m, _ = detect.draft(tmp_path, entry='run_flux.sh')
    assert m['knobs']['accel'] == {'type': 'str', 'flag': '--accel', 'choices': ['fsdp', 'zero2', 'zero3'], 'default': 'zero3'}


ARGS = ("import enum\nclass AttnBackend(enum.Enum):\n    flash = 1\n    fused = 2\n    unfused = 3\n    auto = 5\n"
        "def _add(parser):\n    group = parser.add_argument_group('t')\n"
        "    group.add_argument('--micro-batch-size', type=int, default=None)\n"
        "    group.add_argument('--use-distributed-optimizer', action='store_true')\n"
        "    group.add_argument('--main-grads-dtype', default='fp32', choices=['fp32', 'bf16'])\n"
        "    group.add_argument('--exp-avg-dtype', default='fp32', choices=['fp32', 'bf16'])\n"
        "    group.add_argument('--use-precision-aware-optimizer', action='store_true')\n"
        "    group.add_argument('--no-check-for-nan-in-loss-and-grad', action='store_false', "
        "dest='check_for_nan_in_loss_and_grad')\n"
        "    group.add_argument('--manual-gc', action='store_true')\n"
        "    group.add_argument('--attention-backend', type=lambda a: AttnBackend[a], default=AttnBackend.auto, "
        "choices=list(AttnBackend))\n"
        "    group.add_argument('--moe-grouped-gemm', action='store_true')\n    return parser\n")


def _megatron(root, args):
    _w(root / 'megatron' / 'training' / 'arguments.py', args)
    _w(root / 'pretrain_gpt.py', 'from megatron.training import pretrain\nif __name__ == "__main__":\n    pretrain()\n')
    return detect.draft(root, entry='pretrain_gpt.py')


def test_megatron_drafts_the_flags_of_the_levers_the_packs_measured(tmp_path):
    # M8 searched 15 knobs without bf16 main grads (+6.6 % on Megatron in the H200 packs) or the NaN check: the flags
    # drafted from Megatron's argument module follow the measurements, with what their states require
    assert {'bf16_main_grads', 'nan_check', 'attention_backend'} <= measured('megatron')
    assert not {'manual_gc', 'bf16_moments'} & measured('megatron')           # x1.001, x0.994: nothing to draft
    m, notes = _megatron(tmp_path / 'a', ARGS)
    k, c = m['knobs'], m['constraints']
    assert k['main_grads_dtype'] == {'type': 'str', 'flag': '--main-grads-dtype', 'choices': ['fp32', 'bf16'], 'default': 'fp32'}
    assert k['use_precision_aware_optimizer'] == {'type': 'bool', 'flag': '--use-precision-aware-optimizer', 'default': False}
    assert k['no_check_for_nan_in_loss_and_grad']['type'] == 'bool'
    # an enum default the source does not spell as a literal: no default, the reference keeps Megatron's own
    assert k['attention_backend'] == {'type': 'str', 'flag': '--attention-backend', 'choices': ['auto', 'fused', 'flash',
                                                                                                'unfused']}
    assert not {'manual_gc', 'exp_avg_dtype', 'moe_grouped_gemm'} & set(k)     # not measured faster; a dense job
    assert "main_grads_dtype != 'bf16' or use_precision_aware_optimizer" in c
    raw = m
    raw['knobs']['micro_batch_size']['default'] = 1
    p = _load(tmp_path / 'a', raw, notes)
    assert p.constraint_errors({**p.reference_config(), 'main_grads_dtype': 'bf16'})
    assert not p.constraint_errors({**p.reference_config(), 'main_grads_dtype': 'bf16', 'use_precision_aware_optimizer': True})
    # an older Megatron without the precision-aware optimizer: bf16 main grads cannot be set, and the draft says so
    m, notes = _megatron(tmp_path / 'b', ARGS.replace("    group.add_argument('--use-precision-aware-optimizer', "
                                                      "action='store_true')\n", ''))
    assert 'main_grads_dtype' not in m['knobs']
    assert any(n.startswith('--main-grads-dtype bf16 needs precision_aware_optimizer on') for n in notes)
    assert any(n.startswith('Not reached by any knob: bf16_main_grads fp32 -> bf16') and 'add --main-grads-dtype bf16' in n
               for n in notes)


def test_unreached_levers_name_the_flag_or_a_patch():
    out = {r['mechanism']: r for r in unreached('megatron', {})}
    assert out and all(r['speedup'] >= 1.03 for r in out.values())
    assert 'moe_dispatcher' not in out                                         # MoE levers only for a job with experts
    assert not {'precision', 'fp8_attention'} & set(out)                       # FP8 needs --fp8-format as well
    assert out['bf16_main_grads']['alias'] == '--main-grads-dtype bf16 with --use-precision-aware-optimizer'
    # Wan's measured attention lever (cuDNN first in SDPA's priority) is no flag or env var of the stack: a patch
    wan = {r['mechanism']: r for r in unreached('accelerate', {}, diffusion=True, limit=20)}
    assert wan['attention_backend']['move'][1] == 'cudnn_prio' and wan['attention_backend']['alias'] is None
    # an unmapped project switch that may be the mechanism is named (F-F-4: `bf16` is a state of precision)
    sw = {r['mechanism']: r for r in unreached('megatron', {'bf16': {'type': 'bool', 'flag': '--bf16'}}, limit=20)}
    assert 'bf16' in sw['precision']['maybe']
    # a knob of the mechanism that does not give the measured state leaves the lever named (Wan's cuDNN-first SDPA)
    env = {'DIFFSYNTH_ATTENTION_IMPLEMENTATION': {'type': 'str', 'env': 'DIFFSYNTH_ATTENTION_IMPLEMENTATION',
                                                  'choices': ['torch', 'flash_attention_2', 'flash_attention_4', 'xformers']}}
    assert 'attention_backend' in {r['mechanism'] for r in unreached('accelerate', env, diffusion=True, limit=20)}


# Codex review 2026-10-26 (gpt-6.1-sol): the cases it found, each fixed

def test_a_registry_value_never_replaces_the_jobs_default(tmp_path):
    # --mixed_precision default None inherits Accelerate's config ("no"): the reference must not pass bf16
    _w(tmp_path / 'train.py', "import argparse\nfrom accelerate import Accelerator\np = argparse.ArgumentParser()\n"
                              "p.add_argument('--mixed_precision', default=None)\nargs = p.parse_args()\n"
                              "accelerator = Accelerator(mixed_precision=args.mixed_precision)\n")
    m, notes = detect.draft(tmp_path, entry='train.py')
    k = m['knobs']['mixed_precision']
    assert 'default' not in k and set(k['choices']) == {'no', 'fp16', 'bf16'}
    p = _load(tmp_path, m, notes)
    assert '--mixed_precision' not in args_for(p, p.reference_config())


def test_init_false_dataclass_fields_and_wrapped_parser_builders(tmp_path):
    _w(tmp_path / 'sft.py', "from dataclasses import dataclass, field\nfrom transformers import HfArgumentParser, "
                            "TrainingArguments, Trainer\n@dataclass\nclass ScriptArguments:\n    use_liger: bool = "
                            "field(default=False, init=False)\n    channels_last: bool = False\n\n"
                            "if __name__ == '__main__':\n    s, t = HfArgumentParser((ScriptArguments, TrainingArguments))"
                            ".parse_args_into_dataclasses()\n")
    m, _ = detect.draft(tmp_path, entry='sft.py')
    assert 'use_liger' not in m['knobs'] and 'channels_last' in m['knobs']    # HfArgumentParser skips init=False
    # parse_args() returning build_parser().parse_args()
    _w(tmp_path / 'b' / 'opts.py', "import argparse\ndef build_parser():\n    p = argparse.ArgumentParser()\n"
                                   "    p.add_argument('--num_workers', type=int, default=2)\n    return p\n"
                                   "def parse_args():\n    return build_parser().parse_args()\n")
    _w(tmp_path / 'b' / 'train.py', 'import torch\nfrom opts import parse_args\nargs = parse_args()\n')
    m, _ = detect.draft(tmp_path / 'b', entry='train.py')
    assert m['knobs']['num_workers']['choices'] == [0, 2, 4, 8]


def test_requirements_hold_for_a_flag_the_launch_passes(tmp_path):
    _w(tmp_path / 'megatron' / 'training' / 'arguments.py', ARGS)
    _w(tmp_path / 'pretrain_gpt.py', 'from megatron.training import pretrain\nif __name__ == "__main__":\n    pretrain()\n')
    _w(tmp_path / 'run.sh', '#!/bin/bash\ntorchrun --nproc_per_node 8 pretrain_gpt.py --micro-batch-size 1 '
                            '--main-grads-dtype fp32 "$@"\n')
    m, _ = detect.draft(tmp_path, entry='run.sh')
    assert "main_grads_dtype != 'bf16' or use_precision_aware_optimizer" in m['constraints']


def test_a_value_selected_option_keeps_its_implementations():
    a = {'flag': '--optim_name', 'name': 'optim_name', 'kind': 'str', 'default': 'adamw',
         'choices': ['adamw', 'fused_adamw', 'sgd']}
    assert detect.option_role(a, {}) == 'choice' and detect.perf_values(a) == ['adamw', 'fused_adamw']
