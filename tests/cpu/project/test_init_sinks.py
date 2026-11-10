"""`infrarsi init` finds a job's knobs by what its code does, not by what they are called (sinks.py, 2026-10-26): the
settings that control torch.compile, DataLoader workers, fused AdamW, TF32, autocast, DDP's sync and buckets, gradient
checkpointing, ... - an option of any name, a config key, an env var, a constant, passed through locals, parameters,
`**kwargs` and library functions - are drafted and mapped to the registry; a setting that also decides what is trained
is not; a speed call set in code is named with its line."""
import importlib

import pytest

from infrarsi.experience.registry import label
from infrarsi.project import Project, detect
from infrarsi.project.manifest import args_for, env_for


@pytest.fixture(autouse=True)
def eight_gpus(monkeypatch):
    monkeypatch.setattr(detect, 'gpu_facts', lambda: {'gpus': 8, 'memory_gib': 77.3, 'gpu_type': 'NVIDIA H100 80GB HBM3'})


def _w(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def _states(k, fw='pytorch', name='x'):
    lab = label(name, k, fw)
    vals = k.get('choices') or [False, True]
    return lab.mechanism, {v: lab.state(v) for v in vals}


def test_a_setting_of_any_name_that_gates_a_speed_call(tmp_path):
    # --fast is in no catalogue: it is a knob because it gates torch.compile; --train_text_encoder gates a text
    # encoder's checkpointing too, but it also decides what is trained: not a knob
    _w(tmp_path / 'train.py', '''import argparse, torch
p = argparse.ArgumentParser()
p.add_argument('--fast', action='store_true')
p.add_argument('--train_text_encoder', action='store_true')
p.add_argument('--steps', type=int, default=10)
args = p.parse_args()
model = torch.nn.Linear(4, 4)
te = torch.nn.Linear(4, 4)
if args.fast:
    model = torch.compile(model)
params = list(model.parameters())
if args.train_text_encoder:
    te.gradient_checkpointing_enable()
    params += list(te.parameters())
print(f'fast={args.fast}')
''')
    m, notes = detect.draft(tmp_path, entry='train.py')
    k = m['knobs']
    assert k['fast'] == {'type': 'bool', 'flag': '--fast', 'default': False, 'mechanism': 'compile',
                         'states': {False: 'off', True: 'on'}}
    assert 'train_text_encoder' not in k
    assert any(n.startswith('Mapped to the registry') and 'fast -> compile (train.py:10 compile())' in n for n in notes)


def test_settings_reach_speed_calls_through_config_env_constants_and_parameters(tmp_path):
    _w(tmp_path / 'config.yaml', 'loader:\n  procs: 2\noptim:\n  lr: 0.001\n')
    _w(tmp_path / 'engine.py', '''import torch

def run(model, batches, *, half: bool):
    for x in batches:
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=half):
            model(x).sum().backward()
''')
    _w(tmp_path / 'train.py', '''import argparse, os, yaml, torch
from engine import run
COMPILE_MODEL = False
p = argparse.ArgumentParser()
p.add_argument('--config', default='config.yaml')
p.add_argument('--half-precision', action='store_true')
a = p.parse_args()
cfg = yaml.safe_load(open(a.config))
use_fused = os.environ.get('MY_FUSED', '0') == '1'
model = torch.nn.Linear(4, 4)
if COMPILE_MODEL:
    model = torch.compile(model)
opt = torch.optim.AdamW(model.parameters(), lr=cfg['optim']['lr'], fused=use_fused)
loader = torch.utils.data.DataLoader(list(range(8)), num_workers=cfg['loader']['procs'])
run(model, loader, half=a.half_precision)
''')
    m, notes = detect.draft(tmp_path, entry='train.py')
    k = m['knobs']
    assert k['procs']['path'] == 'loader.procs' and k['procs']['choices'] == [0, 2, 4, 8]       # a config key
    assert label('procs', k['procs'], 'pytorch').mechanism == 'data_workers'
    assert k['MY_FUSED'] == {'type': 'str', 'env': 'MY_FUSED', 'choices': ['0', '1'], 'mechanism': 'optimizer',
                             'states': {'0': 'adamw', '1': 'adamw_fused'}}                   # unset in the reference
    assert k['compile_model'] == {'type': 'bool', 'const': 'COMPILE_MODEL', 'file': 'train.py', 'mechanism': 'compile',
                                  'states': {False: 'off', True: 'on'}}
    assert _states(k['half_precision'], name='half_precision') == ('precision', {False: 'fp32', True: 'bf16'})
    p = Project.load(detect.write(tmp_path, m, notes, force=True))
    assert 'MY_FUSED' not in env_for(p, p.reference_config())


def test_integer_switches_and_ddp_keyword_dicts(tmp_path):
    # D8's shape under other names: type=int 0/1 switches, DDP options passed through a dict
    _w(tmp_path / 'train.py', '''import argparse, torch
from torch.nn.parallel import DistributedDataParallel
p = argparse.ArgumentParser()
p.add_argument('--matmul-tf32', type=int, default=1)
p.add_argument('--sg', type=int, default=0)
p.add_argument('--skip-sync', type=int, default=0)
p.add_argument('--accum', type=int, default=2)
args = p.parse_args()
torch.backends.cuda.matmul.allow_tf32 = bool(args.matmul_tf32)
kw = dict(device_ids=[0], static_graph=bool(args.sg))
ddp = DistributedDataParallel(torch.nn.Linear(4, 4), **kw)
for i in range(args.accum):
    boundary = i == args.accum - 1
    with (ddp.no_sync() if (args.skip_sync and not boundary) else torch.enable_grad()):
        pass
''')
    m, _ = detect.draft(tmp_path, entry='train.py')
    k = m['knobs']
    assert k['matmul_tf32']['choices'] == [0, 1] and k['matmul_tf32']['default'] == 1
    assert _states(k['matmul_tf32'], name='matmul_tf32') == ('tf32', {0: 'off', 1: 'on'})
    assert _states(k['sg'], name='sg') == ('ddp_static_graph', {0: 'off', 1: 'on'})
    assert _states(k['skip_sync'], name='skip_sync') == ('grad_sync_every_micro', {0: 'on', 1: 'off'})


def test_an_optimizer_selector_keeps_only_its_implementations(tmp_path):
    _w(tmp_path / 'train.py', '''import argparse, torch
p = argparse.ArgumentParser()
p.add_argument('--optim', default='adamw', choices=['adamw', 'fused_adamw', 'sgd'])
args = p.parse_args()
params = torch.nn.Linear(4, 4).parameters()
if args.optim == 'sgd':
    opt = torch.optim.SGD(params, lr=0.1)
else:
    opt = torch.optim.AdamW(params, lr=1e-3, fused=args.optim == 'fused_adamw')
''')
    m, _ = detect.draft(tmp_path, entry='train.py')
    k = m['knobs']['optim']
    assert k['choices'] == ['adamw', 'fused_adamw'] and k['default'] == 'adamw'          # not sgd: another algorithm
    assert label('optim', k, 'pytorch').mechanism == 'optimizer'
    assert {v: label('optim', k, 'pytorch').state(v) for v in k['choices']} == {'adamw': 'adamw', 'fused_adamw': 'adamw_fused'}


def test_a_library_handed_the_settings_object(tmp_path, monkeypatch):
    # DiffSynth's shape: the library builds the parser and runs the loop with `args`; its loader reads a setting of a
    # name no catalogue knows
    site = tmp_path / 'site'
    _w(site / 'mylib' / '__init__.py', 'from .runner import *\nfrom .parsers import *\n')
    _w(site / 'mylib' / 'parsers.py', "def add_config(parser):\n    parser.add_argument('--loader_procs', type=int, default=1)\n"
                                       "    return parser\n")
    _w(site / 'mylib' / 'runner.py', '''import torch

def launch(model, dataset, args=None):
    procs = 1
    if args is not None:
        procs = args.loader_procs
    loader = torch.utils.data.DataLoader(dataset, num_workers=procs)
    return loader
''')
    _w(tmp_path / 'repo' / 'train.py', '''import argparse, torch
from mylib import *
parser = argparse.ArgumentParser()
parser = add_config(parser)
args = parser.parse_args()
launch(torch.nn.Linear(4, 4), list(range(8)), args=args)
''')
    monkeypatch.syspath_prepend(str(site))
    importlib.invalidate_caches()
    m, _ = detect.draft(tmp_path / 'repo', entry='train.py')
    k = m['knobs']['loader_procs']
    assert k['flag'] == '--loader_procs' and k['choices'] == [0, 1, 2, 4, 8]
    assert label('loader_procs', k, 'pytorch').mechanism == 'data_workers'


def test_speed_calls_set_in_code(tmp_path):
    # a DiT-like job that compiles and fuses AdamW in code: those measured levers are not suggested; its hard-coded
    # 4 loader workers are named with their line (DiT 4 -> 16 workers measured x1.04)
    _w(tmp_path / 'train.py', '''import torch
from diffusers.models import AutoencoderKL
model = torch.compile(torch.nn.Linear(4, 4))
opt = torch.optim.AdamW(model.parameters(), fused=True)
loader = torch.utils.data.DataLoader(list(range(8)), num_workers=4)
''')
    m, notes = detect.draft(tmp_path, entry='train.py')
    reach = [n for n in notes if n.startswith('Not reached by any knob')]
    assert not any(n.startswith(('Not reached by any knob: compile', 'Not reached by any knob: optimizer')) for n in reach)
    assert any(n.startswith('Not reached by any knob: data_workers 4 -> 16') and 'train.py:5 sets it in code' in n
               for n in reach)


def test_launcher_env_switches_for_groups_of_flags(tmp_path):
    # M8's shape: the launch script turns env vars into groups of Megatron flags; mapped through the flags' aliases,
    # what a value requires is implied (no constraint, nothing removed)
    _w(tmp_path / 'pretrain_gpt.py', 'from megatron.training import pretrain\nif __name__ == "__main__":\n    pretrain()\n')
    _w(tmp_path / 'run.sh', '''#!/bin/bash
EXTRA=()
[ "${MG_BF16:-0}" = "1" ] && EXTRA+=(--use-precision-aware-optimizer --main-grads-dtype bf16)
if [ "$USE_FP8" = "1" ]; then
  EXTRA+=(--fp8-format hybrid --fp8-recipe delayed)
fi
LOCAL=1
[ "$LOCAL" = "1" ] && EXTRA+=(--overlap-grad-reduce)
torchrun --nproc_per_node 8 pretrain_gpt.py --micro-batch-size 1 --bf16 "${EXTRA[@]}" "$@"
''')
    m, notes = detect.draft(tmp_path, entry='run.sh')
    k = m['knobs']
    assert k['MG_BF16']['choices'] == ['0', '1'] and k['MG_BF16']['default'] == '0'
    lab = label('MG_BF16', k['MG_BF16'], 'megatron')
    assert (lab.mechanism, lab.state('1'), lab.implies['1']) == ('bf16_main_grads', 'bf16', {'precision_aware_optimizer': 'on'})
    assert label('USE_FP8', k['USE_FP8'], 'megatron').state('1') == 'fp8_delayed' and 'default' not in k['USE_FP8']
    assert 'LOCAL' not in k                                                  # set in the script: not from outside
    assert not any('needs precision_aware_optimizer' in n for n in notes)
    p = Project.load(detect.write(tmp_path, m, notes, force=True))
    assert env_for(p, {**p.reference_config(), 'MG_BF16': '1'})['MG_BF16'] == '1'


def test_a_launch_option_over_config_files_maps_to_what_they_set(tmp_path):
    # FLUX R205b: --accel picks accel/<name>.yaml; the files say ZeRO 2, ZeRO 3, FSDP full shard - the knob is the
    # registry's zero_stage (the x1.58 zero3 -> zero2 lever is then reached), FSDP implied
    _w(tmp_path / 'accel' / 'zero2.yaml', 'distributed_type: DEEPSPEED\ndeepspeed_config:\n  zero_stage: 2\n')
    _w(tmp_path / 'accel' / 'zero3.yaml', 'distributed_type: DEEPSPEED\ndeepspeed_config:\n  zero_stage: 3\n')
    _w(tmp_path / 'accel' / 'fsdp.yaml', 'distributed_type: FSDP\nfsdp_config:\n  fsdp_sharding_strategy: FULL_SHARD\n')
    _w(tmp_path / 'train.py', "import argparse\nfrom accelerate import Accelerator\np = argparse.ArgumentParser()\n"
                              "p.add_argument('--max_train_steps', type=int)\nargs = p.parse_args()\n")
    _w(tmp_path / 'run.sh', '#!/bin/bash\nACCEL=zero3\nwhile [ $# -gt 0 ]; do\n  case "$1" in\n'
                            '    --accel) ACCEL=$2; shift 2;;\n    *) ARGS+=("$1"); shift;;\n  esac\ndone\n'
                            'accelerate launch --config_file accel/$ACCEL.yaml train.py "${ARGS[@]}"\n')
    m, _ = detect.draft(tmp_path, entry='run.sh')
    k = m['knobs']['accel']
    assert k['mechanism'] == 'zero_stage' and k['states'] == {'zero2': '2', 'zero3': '3'}
    assert k['implies'] == {'fsdp': {'fsdp_sharding': 'full_shard'}}


@pytest.mark.parametrize('source, expect', [
    # click: options are decorators, the function's parameters receive them
    ("""import click, torch
@click.command()
@click.option('--workers', type=int, default=4)
@click.option('--turbo', is_flag=True)
def main(workers, turbo):
    model = torch.nn.Linear(4, 4)
    if turbo:
        model = torch.compile(model)
    torch.utils.data.DataLoader(list(range(8)), num_workers=workers)
if __name__ == '__main__':
    main()
""", {'turbo': ('--turbo', 'compile'), 'workers': ('--workers', 'data_workers')}),
    # typer: the parameters are the options, spelled with dashes
    ("""import typer, torch
app = typer.Typer()
@app.command()
def main(loader_procs: int = 2, use_compile: bool = False):
    model = torch.nn.Linear(4, 4)
    if use_compile:
        model = torch.compile(model)
    torch.utils.data.DataLoader(list(range(8)), num_workers=loader_procs)
if __name__ == '__main__':
    app()
""", {'use_compile': ('--use-compile', 'compile'), 'loader_procs': ('--loader-procs', 'data_workers')}),
    # tyro: a dataclass parsed into dashed flags
    ("""from dataclasses import dataclass
import tyro, torch
@dataclass
class Args:
    loader_workers: int = 2
    fused: bool = False
args = tyro.cli(Args)
opt = torch.optim.AdamW(torch.nn.Linear(4, 4).parameters(), fused=args.fused)
torch.utils.data.DataLoader(list(range(8)), num_workers=args.loader_workers)
""", {'fused': ('--fused', 'optimizer'), 'loader_workers': ('--loader-workers', 'data_workers')}),
    # pyrallis: a wrapped main whose parameter's annotation is the config dataclass
    ("""from dataclasses import dataclass
import pyrallis, torch
@dataclass
class TrainConfig:
    compile_model: bool = False
@pyrallis.wrap()
def main(cfg: TrainConfig):
    model = torch.nn.Linear(4, 4)
    if cfg.compile_model:
        model = torch.compile(model)
main()
""", {'compile_model': ('--compile_model', 'compile')}),
    # env switches cast from strings, with the code's own default
    ("""import os, torch
opt = torch.optim.AdamW(torch.nn.Linear(4, 4).parameters(), fused=bool(int(os.environ.get('FUSED', '0'))))
torch.backends.cudnn.benchmark = os.environ.get('CUDNN_BENCH', 'false').lower() == 'true'
""", {'FUSED': ('FUSED', 'optimizer'), 'CUDNN_BENCH': ('CUDNN_BENCH', 'cudnn_benchmark')}),
])
def test_command_line_libraries_and_env_switches(tmp_path, source, expect):
    _w(tmp_path / 'train.py', source)
    m, _ = detect.draft(tmp_path, entry='train.py')
    k = m['knobs']
    for name, (target, mech) in expect.items():
        assert (k[name].get('flag') or k[name].get('env')) == target
        assert label(name, k[name], 'pytorch').mechanism == mech
    if 'FUSED' in expect:
        assert k['FUSED']['choices'] == ['0', '1'] and k['CUDNN_BENCH']['choices'] == ['false', 'true']


def test_an_argparse_dest_and_hydra_overrides(tmp_path):
    # the code reads args.<dest>, not the flag's name; Hydra's settings are config keys set by `key=value` overrides
    _w(tmp_path / 'a' / 'train.py', "import argparse, torch\np = argparse.ArgumentParser()\n"
                                    "p.add_argument('--use-fast-kernels', dest='fast', action='store_true')\n"
                                    "args = p.parse_args()\nm = torch.nn.Linear(4, 4)\nif args.fast:\n    m = torch.compile(m)\n")
    m, _ = detect.draft(tmp_path / 'a', entry='train.py')
    assert _states(m['knobs']['use_fast_kernels'], name='use_fast_kernels') == ('compile', {False: 'off', True: 'on'})
    _w(tmp_path / 'h' / 'conf' / 'config.yaml', 'data:\n  loader_workers: 2\ntrain:\n  use_compile: false\n')
    _w(tmp_path / 'h' / 'train.py', "import hydra, torch\nfrom omegaconf import DictConfig\n"
                                    "@hydra.main(config_path='conf', config_name='config', version_base=None)\n"
                                    "def main(cfg: DictConfig):\n"
                                    "    torch.utils.data.DataLoader(list(range(8)), num_workers=cfg.data.loader_workers)\n"
                                    "    if cfg.train.use_compile:\n        torch.compile(torch.nn.Linear(4, 4))\n"
                                    "if __name__ == '__main__':\n    main()\n")
    m, _ = detect.draft(tmp_path / 'h', entry='train.py')
    k = m['knobs']
    assert k['loader_workers'] == {'type': 'int', 'override': 'data.loader_workers', 'choices': [0, 2, 4, 8],
                                   'mechanism': 'data_workers'} or label('loader_workers', k['loader_workers'],
                                                                         'pytorch').mechanism == 'data_workers'
    assert 'default' not in k['loader_workers'] and k['use_compile']['override'] == 'train.use_compile'
    assert label('use_compile', k['use_compile'], 'pytorch').mechanism == 'compile'


def test_a_config_file_option_over_its_siblings(tmp_path):
    # HF Trainer's --deepspeed ds/zero2.json next to ds/zero3.json: a choice of files, mapped by what they set
    _w(tmp_path / 'ds' / 'zero2.json', '{"zero_optimization": {"stage": 2}, "optimizer": {"type": "AdamW", "params": '
                                       '{"lr": 0.001}}, "train_micro_batch_size_per_gpu": "auto"}')
    _w(tmp_path / 'ds' / 'zero3.json', '{"zero_optimization": {"stage": 3, "stage3_prefetch_bucket_size": "auto"}, '
                                       '"optimizer": {"type": "AdamW", "params": {"lr": 0.001}}, '
                                       '"train_micro_batch_size_per_gpu": "auto"}')
    _w(tmp_path / 'ds' / 'zero3_sgd.json', '{"zero_optimization": {"stage": 3}, "optimizer": {"type": "SGD", "params": '
                                           '{"lr": 0.1}}, "train_micro_batch_size_per_gpu": "auto"}')   # also another optimizer
    _w(tmp_path / 'ds' / 'notes.json', '{"comment": "not a DeepSpeed config"}')
    _w(tmp_path / 'train.py', "from transformers import HfArgumentParser, TrainingArguments, Trainer\n"
                              "args, = HfArgumentParser((TrainingArguments,)).parse_args_into_dataclasses()\n")
    _w(tmp_path / 'run.sh', '#!/bin/bash\ntorchrun --nproc_per_node 8 train.py --deepspeed ds/zero2.json "$@"\n')
    m, _ = detect.draft(tmp_path, entry='run.sh')
    k = m['knobs']['deepspeed']
    assert k == {'type': 'str', 'flag': '--deepspeed', 'choices': ['ds/zero2.json', 'ds/zero3.json'], 'default': 'ds/zero2.json',
                 'mechanism': 'zero_stage', 'states': {'ds/zero2.json': '2', 'ds/zero3.json': '3'}}


def test_lightning_precision_from_a_config_key(tmp_path):
    # the knob's name `precision` is a mechanism, but its values are Lightning's: mapped by the Trainer call
    _w(tmp_path / 'conf.yaml', 'trainer:\n  precision: 32-true\n  accum: 1\n')
    _w(tmp_path / 'train.py', """import lightning as L
from omegaconf import OmegaConf
cfg = OmegaConf.load('conf.yaml')
trainer = L.Trainer(precision=cfg.trainer.precision, accumulate_grad_batches=cfg.trainer.accum)
""")
    m, _ = detect.draft(tmp_path, entry='train.py', config='conf.yaml')
    k = m['knobs']['precision']
    assert k['mechanism'] == 'precision' and k['states']['32-true'] == 'fp32' and k['states']['bf16-mixed'] == 'bf16'


# Codex review round 2 (2026-10-26, gpt-6.1-sol): the cases it found, each fixed

def test_a_semantic_use_inside_another_argument_of_a_speed_call(tmp_path):
    _w(tmp_path / 'train.py', "import argparse, torch\np = argparse.ArgumentParser()\n"
                              "p.add_argument('--train_text_encoder', action='store_true')\nargs = p.parse_args()\n"
                              "unet, te = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)\n"
                              "if args.train_text_encoder:\n    te.gradient_checkpointing_enable()\n"
                              "opt = torch.optim.AdamW(list(unet.parameters()) + (list(te.parameters()) if "
                              "args.train_text_encoder else []), lr=1e-4)\n")
    m, _ = detect.draft(tmp_path, entry='train.py')
    assert 'train_text_encoder' not in m['knobs']                  # it also picks the parameters the optimizer trains


def test_an_optimizer_selector_never_changes_the_algorithm(tmp_path):
    assert detect.perf_values({'name': 'optim', 'default': 'sgd', 'choices': ['sgd', 'fused_adamw']}) is None
    assert detect.perf_values({'name': 'optim', 'default': 'adamw_torch', 'choices': ['adamw_torch', 'adamw_torch_fused',
                                                                                     'sgd']}) == ['adamw_torch', 'adamw_torch_fused']
    _w(tmp_path / 'train.py', "import argparse, torch\np = argparse.ArgumentParser()\n"
                              "p.add_argument('--optim', default='sgd', choices=['sgd', 'fused_adamw'])\nargs = p.parse_args()\n"
                              "params = torch.nn.Linear(4, 4).parameters()\nif args.optim == 'sgd':\n"
                              "    opt = torch.optim.SGD(params, lr=0.1)\nelse:\n"
                              "    opt = torch.optim.AdamW(params, fused=True, lr=0.001)\n")
    m, _ = detect.draft(tmp_path, entry='train.py')
    assert 'optim' not in m['knobs']


def test_an_optional_bool_dataclass_field_is_not_false(tmp_path):
    _w(tmp_path / 'sft.py', "from dataclasses import dataclass\nfrom typing import Optional\nimport torch\n"
                            "from transformers import HfArgumentParser, TrainingArguments, Trainer\n"
                            "@dataclass\nclass Extra:\n    tf32_override: Optional[bool] = None\n"
                            "extra, t = HfArgumentParser((Extra, TrainingArguments)).parse_args_into_dataclasses()\n"
                            "torch.backends.cuda.matmul.allow_tf32 = True\nif extra.tf32_override is not None:\n"
                            "    torch.backends.cuda.matmul.allow_tf32 = extra.tf32_override\n")
    m, _ = detect.draft(tmp_path, entry='sft.py')
    assert 'tf32_override' not in m['knobs']                       # unset is a third state: --x false would turn it off


def test_a_conjunction_of_settings_maps_a_value_only_when_it_decides(tmp_path):
    _w(tmp_path / 'train.py', "import argparse, torch\np = argparse.ArgumentParser()\n"
                              "p.add_argument('--fast', action='store_true')\np.add_argument('--use_inductor', action='store_true')\n"
                              "args = p.parse_args()\nm = torch.nn.Linear(4, 4)\nif args.fast and args.use_inductor:\n"
                              "    m = torch.compile(m)\n")
    m, _ = detect.draft(tmp_path, entry='train.py')
    k = m['knobs']['fast']
    assert k.get('states', {}).get(True) is None                   # fast alone does not compile
    assert k.get('states', {}).get(False) in ('off', None)


def test_autocast_dtype_counts_only_when_enabled_and_compile_disable(tmp_path):
    _w(tmp_path / 'train.py', "import argparse, torch\np = argparse.ArgumentParser()\n"
                              "p.add_argument('--half', default='bfloat16', choices=['bfloat16', 'float16'])\n"
                              "p.add_argument('--fast', action='store_true')\nargs = p.parse_args()\nm = torch.nn.Linear(4, 4)\n"
                              "with torch.autocast('cuda', dtype=getattr(torch, args.half), enabled=False):\n    pass\n"
                              "m = torch.compile(m, disable=not args.fast)\n")
    m, notes = detect.draft(tmp_path, entry='train.py')
    k = m['knobs']
    assert 'half' not in k or label('half', k['half'], 'pytorch') is None or k['half'].get('mechanism') != 'precision'
    assert _states(k['fast'], name='fast') == ('compile', {False: 'off', True: 'on'})   # disable=not fast: eager by default


def test_a_numeric_env_setting_gets_numbers(tmp_path):
    _w(tmp_path / 'train.py', "import os, torch\nloader = torch.utils.data.DataLoader(list(range(8)), "
                              "num_workers=int(os.getenv('LOADER_WORKERS', '4')))\n")
    m, _ = detect.draft(tmp_path, entry='train.py')
    assert m['knobs']['LOADER_WORKERS']['choices'] == ['0', '2', '4', '8']


# Codex review round 3 (2026-10-26): the cases it found, each fixed

@pytest.mark.parametrize('source', [
    # an f-string that picks the data, not a log line
    "import argparse, torch\np = argparse.ArgumentParser()\np.add_argument('--variant', action='store_true')\n"
    "args = p.parse_args()\nm = torch.nn.Linear(4, 4)\nif args.variant:\n    m = torch.compile(m)\n"
    "data = load_dataset(f'corpus_{args.variant}')\n",
    # one branch a lever, the other another model (LoRA)
    "import argparse, torch\np = argparse.ArgumentParser()\np.add_argument('--variant', action='store_true')\n"
    "args = p.parse_args()\nbase = torch.nn.Linear(4, 4)\n"
    "model = get_peft_model(base, cfg) if args.variant else torch.compile(base)\n",
])
def test_settings_that_also_pick_data_or_the_model_are_not_knobs(tmp_path, source):
    _w(tmp_path / 'train.py', source)
    m, _ = detect.draft(tmp_path, entry='train.py')
    assert 'variant' not in m['knobs']


def test_a_bare_implementation_word_does_not_replace_another_algorithm():
    assert detect.perf_values({'name': 'optim', 'default': 'sgd', 'choices': ['sgd', 'fused']}) is None
    assert detect.perf_values({'name': 'strategy', 'default': 'ddp', 'choices': ['ddp', 'fsdp']}) == ['ddp', 'fsdp']
    assert detect.perf_values({'name': 'mode', 'default': 'eager', 'choices': ['eager', 'compiled']}) == ['eager', 'compiled']


@pytest.mark.parametrize('source, absent, present', [
    # click: an inverted flag (passing it turns TF32 off) is not drafted; a bool value option takes `--x true|false`
    ("import click, torch\n@click.command()\n@click.option('--no-tf32', 'tf32', is_flag=True, flag_value=False, default=True)\n"
     "@click.option('--cudnn-bench', type=bool, default=False)\ndef main(tf32, cudnn_bench):\n"
     "    torch.backends.cuda.matmul.allow_tf32 = tf32\n    torch.backends.cudnn.benchmark = cudnn_bench\n"
     "if __name__ == '__main__':\n    main()\n", 'no_tf32', ('cudnn_bench', '--cudnn-bench')),
    # typer: an Annotated option name, and a positional Argument (no flag)
    ("from typing import Annotated\nimport typer, torch\napp = typer.Typer()\n@app.command()\n"
     "def main(loader_procs: Annotated[int, typer.Option('--workers')] = 4, shards: int = typer.Argument(2)):\n"
     "    torch.utils.data.DataLoader(list(range(8)), num_workers=loader_procs)\n    torch.utils.data.DataLoader(\n"
     "        list(range(8)), num_workers=shards)\nif __name__ == '__main__':\n    app()\n", 'shards', ('loader_procs', '--workers')),
])
def test_command_line_library_spellings(tmp_path, source, absent, present):
    _w(tmp_path / 'train.py', source)
    m, notes = detect.draft(tmp_path, entry='train.py')
    k = m['knobs']
    assert absent not in k and k[present[0]]['flag'] == present[1]
    if present[0] == 'cudnn_bench':
        assert k['cudnn_bench']['style'] == 'value'
        p = Project.load(detect.write(tmp_path, m, notes, force=True))
        assert args_for(p, p.reference_config())[-2:] == ['--cudnn-bench', 'false']


@pytest.mark.parametrize('option', ["@click.option('--tf32', type=bool, default=None)",
                                    "@click.option('--tf32', type=bool)"])
def test_a_click_bool_value_option_without_a_default_is_not_drafted(tmp_path, option):
    # Codex review round 4: drafted with default False, the reference run would pass `--tf32 false` and turn TF32 off
    _w(tmp_path / 'train.py', f"import click, torch\n@click.command()\n{option}\ndef main(tf32):\n"
                              "    torch.backends.cuda.matmul.allow_tf32 = True\n    if tf32 is not None:\n"
                              "        torch.backends.cuda.matmul.allow_tf32 = tf32\nif __name__ == '__main__':\n    main()\n")
    m, _ = detect.draft(tmp_path, entry='train.py')
    assert 'tf32' not in m['knobs']


def test_a_setting_inside_a_call_another_setting_gates_has_no_state(tmp_path):
    _w(tmp_path / 'train.py', "import argparse, torch\np = argparse.ArgumentParser()\n"
                              "p.add_argument('--enable', action='store_true')\np.add_argument('--fast', action='store_true')\n"
                              "args = p.parse_args()\nm = torch.nn.Linear(4, 4)\nif args.enable:\n"
                              "    m = torch.compile(m, disable=not args.fast)\n")
    m, _ = detect.draft(tmp_path, entry='train.py')
    assert 'states' not in m['knobs'].get('fast', {})               # fast alone does not compile: enable decides too


def test_finite_choices_of_an_integer_mechanism_do_not_reach_every_state():
    from infrarsi.experience.store import unreached
    knob = {'num_workers': {'type': 'int', 'flag': '--num-workers', 'choices': [0, 2, 4, 8], 'default': 4}}
    got = {r['mechanism']: r for r in unreached('pytorch', knob, diffusion=True, limit=20)}
    assert got['data_workers']['move'] == ['4', '16']               # no choice gives 16 workers
