"""`infrarsi init` on the repository shapes of the 10-23 usability trials (I3-I13, D1): launch scripts, Megatron's
argument module, HF TrainingArguments, DeepSpeed / Lightning / TorchTitan configs, and the request errors."""
import json
from pathlib import Path

import pytest
import yaml

from infrarsi.cli.main import main
from infrarsi.project import Project, detect
from infrarsi.project.manifest import ManifestError, args_for

H100 = {'gpus': 8, 'memory_gib': 77.3, 'gpu_type': 'NVIDIA H100 80GB HBM3'}


@pytest.fixture(autouse=True)
def eight_gpus(monkeypatch):
    monkeypatch.setattr(detect, 'gpu_facts', lambda: dict(H100))


def _w(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


SFT = ("import argparse\nimport os\nimport torch\nimport torch.distributed as dist\n"
       "def main():\n    ap = argparse.ArgumentParser()\n    ap.add_argument('--config', default='config.yaml')\n"
       "    ap.add_argument('--max-steps', type=int, default=30)\n    ap.add_argument('--micro-batch-size', type=int)\n"
       "    ap.add_argument('--grad-accum', type=int)\n    ap.add_argument('--workers', type=int)\n"
       "    ap.add_argument('--bf16', action='store_true')\n    ap.add_argument('--tf32', action='store_true')\n"
       "    ap.add_argument('--grad-ckpt', action='store_true')\n"
       "    ap.add_argument('--attn', default='sdpa', choices=['sdpa', 'eager', 'flash_attention_2'])\n"
       "    if int(os.environ.get('LOCAL_RANK', '0')) >= 0:\n        dist.init_process_group('nccl')\n"
       "if __name__ == '__main__':\n    main()\n")
SFT_CFG = 'train:\n  micro_batch_size: 1\n  grad_accum: 4\n  workers: 2\nmodel:\n  path: /m\n'


def test_flag_overriding_a_config_key_drafts_the_key(tmp_path):
    # D1: `--micro-batch-size` (no default) overrides train.micro_batch_size; the draft must load and give the invariant
    _w(tmp_path / 'train.py', SFT)
    _w(tmp_path / 'config.yaml', SFT_CFG)
    m, notes = detect.draft(tmp_path)
    k = m['knobs']
    assert k['micro_batch_size'] == {'type': 'int', 'path': 'train.micro_batch_size', 'choices': [1, 2, 4, 8, 16]}
    assert k['grad_accum']['path'] == 'train.grad_accum' and k['workers']['path'] == 'train.workers'
    assert {'bf16', 'tf32', 'grad_ckpt'} <= set(k) and k['bf16'] == {'type': 'bool', 'flag': '--bf16', 'default': False}
    assert k['attn'] == {'type': 'str', 'flag': '--attn', 'choices': ['sdpa', 'eager', 'flash_attention_2'], 'default': 'sdpa'}
    assert m['launch']['command'] == 'torchrun --nproc_per_node {gpus} train.py --config {config} --max-steps {steps} {args}'
    out = detect.write(tmp_path, m, notes)
    p = Project.load(out)
    assert p.invariants(p.reference_config()) == {'global_batch': 32}


def test_request_errors_print_one_line(tmp_path, capsys):
    with pytest.raises(detect.InitError, match='not a directory'):
        detect.draft(tmp_path / 'missing')
    with pytest.raises(detect.InitError, match='no training entry point'):
        detect.draft(tmp_path)                                       # an empty directory: nothing to draft
    _w(tmp_path / 'train.py', SFT)
    with pytest.raises(detect.InitError, match='nope.py not found'):
        detect.draft(tmp_path, entry='nope.py')
    with pytest.raises(detect.InitError, match='x.yaml not found'):
        detect.draft(tmp_path, config='x.yaml')
    with pytest.raises(SystemExit) as e:
        main(['init', str(tmp_path / 'missing')])
    assert e.value.code == 2
    main(['init', str(tmp_path)])
    with pytest.raises(SystemExit) as e:
        main(['init', str(tmp_path)])                                # twice without --force
    assert e.value.code == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert err[0].startswith('infrarsi init: ') and 'not a directory' in err[0]
    assert err[1].startswith('infrarsi init: ') and 'use --force' in err[1] and len(err) == 2


def test_unknown_default_in_an_invariant_is_a_clear_error(tmp_path, capsys):
    _w(tmp_path / 'train.py', 'print(1)\n')
    raw = {'schema': 'infrarsi-project-v1', 'name': 't', 'launch': {'command': '{python} train.py {args}'},
           'knobs': {'mbs': {'type': 'int', 'flag': '--mbs', 'choices': [1, 2]},
                     'ga': {'type': 'int', 'flag': '--ga', 'choices': [1, 2], 'default': 2}},
           'workload': {'steps': 30, 'window': [11, 30], 'invariants': {'global_batch': 'mbs * ga * gpus'}}}
    _w(tmp_path / 'infrarsi.yaml', yaml.safe_dump(raw))
    with pytest.raises(ManifestError, match='uses mbs, whose current value is unknown'):
        Project.load(tmp_path / 'infrarsi.yaml')
    with pytest.raises(SystemExit) as e:
        main(['doctor', str(tmp_path / 'infrarsi.yaml')])
    assert e.value.code == 2 and 'give it a default' in capsys.readouterr().err
    # an expression that handles the unset value itself is fine
    raw['knobs']['mbs']['default'] = 1
    raw['knobs']['rc'] = {'type': 'str', 'flag': '--rc', 'choices': ['full', 'selective']}
    raw['constraints'] = ['rc == None or rc == "selective"']
    _w(tmp_path / 'infrarsi.yaml', yaml.safe_dump(raw))
    p = Project.load(tmp_path / 'infrarsi.yaml')
    assert p.constraint_errors(p.reference_config()) == [] and p.constraint_errors({**p.reference_config(), 'rc': 'full'})


def test_cli_one_line_only_for_request_errors(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        main(['report', str(tmp_path)])
    assert e.value.code == 2 and 'is not a run directory' in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        main(['doctor', str(tmp_path / 'nope')])
    assert e.value.code == 2 and 'does not exist' in capsys.readouterr().err
    _w(tmp_path / 'bad' / 'infrarsi.yaml', 'knobs: {a: 1\n')
    with pytest.raises(SystemExit) as e:
        main(['doctor', str(tmp_path / 'bad')])
    assert e.value.code == 2 and 'is not valid YAML' in capsys.readouterr().err
    toy = Path(__file__).resolve().parents[3] / 'examples' / 'toy_cpu'
    with pytest.raises(SystemExit) as e:                # an LLM the request names that cannot be built (trial C3b)
        main(['run', str(toy), '--llm', 'vllm'])
    assert e.value.code == 2 and 'infrarsi run: llm: llm backend vllm needs a model name' in capsys.readouterr().err
    import infrarsi.project.doctor as doctor_module

    def broken(*a, **k):
        raise FileNotFoundError('internal receipt.json missing')
    monkeypatch.setattr(doctor_module, 'doctor', broken)
    with pytest.raises(FileNotFoundError, match='internal receipt'):    # a failure of the run itself keeps its traceback
        main(['doctor', str(toy)])


MEGATRON_ARGS = ("def _add(parser):\n    group = parser.add_argument_group('t')\n"
                 "    group.add_argument('--micro-batch-size', type=int, default=None)\n"
                 "    group.add_argument('--tensor-model-parallel-size', type=int, default=1)\n"
                 "    group.add_argument('--pipeline-model-parallel-size', type=int, default=1)\n"
                 "    group.add_argument('--sequence-parallel', action='store_true')\n"
                 "    group.add_argument('--use-distributed-optimizer', action='store_true')\n"
                 "    group.add_argument('--overlap-grad-reduce', action='store_true', default=False)\n"
                 "    group.add_argument('--overlap-param-gather', action='store_true', default=False)\n"
                 "    group.add_argument('--tp-comm-overlap', action='store_true')\n"
                 "    group.add_argument('--recompute-granularity', type=str, default=None, choices=['full', 'selective'])\n"
                 "    group.add_argument('--num-workers', type=int, default=2)\n"
                 "    group.add_argument('--fp16-lm-cross-entropy', action='store_true')\n    return parser\n")
RUN_SH = """#!/bin/bash
# Megatron on one node
MEGATRON=$(dirname $0)/mlm
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1} NCCL_NVLS_ENABLE=0
EXTRA=()
[ "${OVERLAP:-0}" = "1" ] && EXTRA+=(--overlap-grad-reduce --overlap-param-gather)
if [ -n "$FP8" ]; then
  EXTRA+=(--fp8-format hybrid)
fi
torchrun --nproc-per-node 8 mlm/pretrain_gpt.py \\
  --num-layers ${LAYERS:-32} --num-attention-heads 32 --num-query-groups 8 --global-batch-size 64 \\
  --tensor-model-parallel-size 2 --use-distributed-optimizer --bf16 --train-iters 500 \\
  "${EXTRA[@]}" "$@"
"""


def test_megatron_launch_script(tmp_path):
    _w(tmp_path / 'mlm' / 'megatron' / 'training' / 'arguments.py', MEGATRON_ARGS)
    _w(tmp_path / 'mlm' / 'pretrain_gpt.py', 'from megatron.training import pretrain\nif __name__ == "__main__":\n    pretrain()\n')
    _w(tmp_path / 'run.sh', RUN_SH)
    m, notes = detect.draft(tmp_path, entry='run.sh')
    k, c = m['knobs'], m['constraints']
    assert m['framework'] == 'megatron' and m['hardware']['gpus'] == 8
    assert m['launch']['command'] == 'bash run.sh --train-iters {steps} --log-interval 1 --timing-log-level 1 {args}'
    assert k['tensor_model_parallel_size'] == {'type': 'int', 'flag': '--tensor-model-parallel-size', 'choices': [1, 2, 4, 8],
                                               'default': 2}
    assert 'use_distributed_optimizer' not in k and 'bf16' not in k                # switches the script sets stay on
    assert k['overlap_grad_reduce']['default'] is False and k['overlap_param_gather']['default'] is False  # conditional lines
    assert 'fp16_lm_cross_entropy' not in k                                     # not one of the searched Megatron flags
    assert 'NCCL_NVLS_ENABLE' not in k and k['CUDA_DEVICE_MAX_CONNECTIONS']['env'] == 'CUDA_DEVICE_MAX_CONNECTIONS'
    assert 'not overlap_param_gather or (overlap_grad_reduce)' in c
    assert "CUDA_DEVICE_MAX_CONNECTIONS == '1' or (tensor_model_parallel_size == 1)" in c       # TP > 1 before Blackwell
    assert k['recompute_granularity']['choices'] == ['selective']               # full also needs --recompute-method
    assert 'not sequence_parallel or (tensor_model_parallel_size > 1)' in c
    assert 'not tp_comm_overlap or (tensor_model_parallel_size > 1 and sequence_parallel)' in c
    assert '32 % tensor_model_parallel_size == 0' in c and '8 % tensor_model_parallel_size == 0' in c
    assert '64 % (micro_batch_size * ((gpus * nodes) // (tensor_model_parallel_size * pipeline_model_parallel_size))) == 0' in c
    assert notes[0].startswith('Set the default of micro_batch_size')
    assert m['code'] == {'mutable': ['mlm/pretrain_gpt.py'], 'frozen': [], 'readable': ['run.sh']}
    raw = yaml.safe_load(detect.write(tmp_path, m, notes).read_text(encoding='utf-8'))
    raw['knobs']['micro_batch_size']['default'] = 1
    _w(tmp_path / 'infrarsi.yaml', yaml.safe_dump(raw))
    p = Project.load(tmp_path / 'infrarsi.yaml')
    assert p.constraint_errors(p.reference_config()) == []
    assert p.constraint_errors({**p.reference_config(), 'overlap_param_gather': True})
    assert p.constraint_errors({**p.reference_config(), 'CUDA_DEVICE_MAX_CONNECTIONS': '32'})          # TP 2 needs '1'
    assert not p.constraint_errors({**p.reference_config(), 'CUDA_DEVICE_MAX_CONNECTIONS': '32',
                                    'tensor_model_parallel_size': 1})


def test_launch_line_after_cd_and_one_line_blocks(tmp_path):
    _w(tmp_path / 'train.py', SFT)
    _w(tmp_path / 'config.yaml', SFT_CFG)
    _w(tmp_path / 'go.sh', 'set -e\nif [ -n "$DEBUG" ]; then EXTRA="--compile"; fi\n'
                          'cd "$(dirname "$0")" && torchrun --nproc_per_node 4 train.py --config config.yaml --bf16 "$@"\n')
    m, notes = detect.draft(tmp_path, entry='go.sh')
    assert m['hardware']['gpus'] == 4 and 'bf16' not in m['knobs']           # the switch the script sets stays on
    assert m['launch']['command'] == 'bash go.sh --config {config} --max-steps {steps} {args}'
    assert m['knobs']['micro_batch_size']['path'] == 'train.micro_batch_size' and m['config']['file'] == 'config.yaml'


def test_megatron_python_entry_without_a_launch_script(tmp_path):
    _w(tmp_path / 'megatron' / 'training' / 'arguments.py', MEGATRON_ARGS)
    _w(tmp_path / 'pretrain_gpt.py', 'from megatron.training import pretrain\nif __name__ == "__main__":\n    pretrain()\n')
    _w(tmp_path / '.pre-commit-config.yaml', 'repos: []\n')
    m, notes = detect.draft(tmp_path, entry='pretrain_gpt.py')
    assert m['config']['file'] is None                                           # tooling yaml is not a training config
    assert m['launch']['command'] == ('torchrun --nproc_per_node {gpus} pretrain_gpt.py --train-iters {steps} '
                                      '--log-interval 1 --timing-log-level 1 {args}')
    assert {'micro_batch_size', 'tensor_model_parallel_size', 'overlap_grad_reduce', 'recompute_granularity'} <= set(m['knobs'])
    assert notes[0].startswith('Megatron requires --micro-batch-size') and any('model, data and tokenizer' in n for n in notes)


def test_hf_trainer_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(detect, 'hf_defaults', lambda: {**detect.HF_FIELDS, 'optim': 'adamw_torch'})
    _w(tmp_path / 'train_hf.py', 'from transformers import HfArgumentParser, Trainer, TrainingArguments\n'
                                 'args, = HfArgumentParser((TrainingArguments,)).parse_args_into_dataclasses()\n'
                                 'Trainer(args=args).train()\n')
    m, notes = detect.draft(tmp_path)
    k = m['knobs']
    assert m['framework'] == 'hf_trainer' and m['signals']['count_steps'] is True
    assert k['per_device_train_batch_size'] == {'type': 'int', 'flag': '--per_device_train_batch_size',
                                                'choices': [1, 2, 4, 8, 16], 'default': 8}
    assert k['tf32'] == {'type': 'bool', 'flag': '--tf32', 'style': 'value', 'default': False}
    p = Project(m, tmp_path / 'infrarsi.yaml')
    assert '--tf32 true' in ' '.join(args_for(p, {**p.reference_config(), 'tf32': True}))    # Optional[bool] needs a value
    assert k['optim']['choices'] == ['adamw_torch', 'adamw_torch_fused']
    assert m['workload']['invariants']['global_batch'] == 'per_device_train_batch_size * gradient_accumulation_steps * gpus'
    assert m['launch']['command'] == ('torchrun --nproc_per_node {gpus} train_hf.py --max_steps {steps} --logging_steps 1 '
                                      '--disable_tqdm true --report_to none {args}')
    assert any(n.startswith('HF Trainer: add the arguments') for n in notes)


def test_deepspeed_picks_the_config_with_the_knobs(tmp_path):
    _w(tmp_path / 'train.py', SFT)
    _w(tmp_path / 'config.yaml', SFT_CFG)
    _w(tmp_path / 'train_ds.py', "import argparse\nimport deepspeed\nap = argparse.ArgumentParser()\n"
                                 "ap.add_argument('--config', default='config.yaml')\nap.add_argument('--ds-config', default='ds_config.json')\n"
                                 "ap.add_argument('--max-steps', type=int, default=20)\nap.add_argument('--local_rank', type=int, default=-1)\n"
                                 "a = ap.parse_args()\nengine = deepspeed.initialize(config=a.ds_config)\n")
    _w(tmp_path / 'ds_config.json', json.dumps({
        'train_batch_size': 16, 'train_micro_batch_size_per_gpu': 1, 'gradient_accumulation_steps': 2, 'bf16': {'enabled': True},
        'zero_optimization': {'stage': 2, 'overlap_comm': False, 'reduce_bucket_size': 50000000}}))
    m, notes = detect.draft(tmp_path)
    assert m['launch']['command'].startswith('torchrun') and any('other candidates: train_ds.py' in n for n in notes)
    m, notes = detect.draft(tmp_path, entry='train_ds.py')
    k = m['knobs']
    assert m['config']['file'] == 'ds_config.json' and 'train_batch_size' not in k
    assert m['launch']['command'] == 'deepspeed --num_gpus {gpus} train_ds.py --ds-config {config} --max-steps {steps} {args}'
    assert k['bf16'] == {'type': 'bool', 'path': 'bf16.enabled'} and k['stage']['path'] == 'zero_optimization.stage'
    assert m['workload']['invariants']['global_batch'] == 'train_micro_batch_size_per_gpu * gradient_accumulation_steps * gpus'
    p = Project.load(detect.write(tmp_path, m, notes))
    assert p.invariants(p.reference_config()) == {'global_batch': 16}


def test_lightning_cli(tmp_path):
    _w(tmp_path / 'train_lit.py', 'import lightning as L\nfrom lightning.pytorch.cli import LightningCLI\n'
                                  'if __name__ == "__main__":\n    LightningCLI(L.LightningModule)\n')
    _w(tmp_path / 'config.yaml', 'trainer:\n  max_steps: 30\n  precision: 32-true\n  accumulate_grad_batches: 1\n'
                                 'data:\n  batch_size: 8\n  num_workers: 2\n')
    m, notes = detect.draft(tmp_path)
    k = m['knobs']
    assert m['launch']['command'] == '{python} train_lit.py fit --config {config} --trainer.max_steps {steps} {args}'
    assert k['precision']['choices'] == ['32-true', 'bf16-mixed'] and k['batch_size']['path'] == 'data.batch_size'
    assert m['workload']['invariants']['global_batch'] == 'batch_size * accumulate_grad_batches * gpus'
    assert any(n.startswith('batch_size was taken as the per-device micro-batch') for n in notes)
    assert not any('config-file argument' in n for n in notes)


def test_torchtitan_toml_config(tmp_path):
    _w(tmp_path / 'torchtitan' / '__init__.py', '')
    _w(tmp_path / 'torchtitan' / 'train.py', 'import torch\nfrom torchtitan.config import JobConfig\n'
                                             'if __name__ == "__main__":\n    JobConfig()\n')
    _w(tmp_path / 'cfg' / 'debug.toml', '[training]\nlocal_batch_size = 8\nsteps = 10\n[parallelism]\ntensor_parallel_degree = 1\n'
                                        '[activation_checkpoint]\nmode = "selective"\n[compile]\nenable = false\n')
    m, notes = detect.draft(tmp_path, config='cfg/debug.toml')
    k = m['knobs']
    assert m['config']['file'] is None
    assert m['launch']['command'] == ('torchrun --nproc_per_node {gpus} -m torchtitan.train --job.config_file cfg/debug.toml '
                                      '--training.steps {steps} {args}')
    assert k['local_batch_size'] == {'type': 'int', 'flag': '--training.local_batch_size', 'choices': [1, 2, 4, 8, 16], 'default': 8}
    assert k['activation_checkpoint_mode']['choices'] == ['none', 'selective', 'full']
    assert k['compile'] == {'type': 'bool', 'flag': '--compile.enable', 'default': False}
    Project.load(detect.write(tmp_path, m, notes))


def test_entry_ranking_and_single_gpu_scripts(tmp_path):
    for name in ('train_text_to_image_lora.py', 'train_text_to_image.py'):
        _w(tmp_path / name, "import argparse\nimport torch\nap = argparse.ArgumentParser()\n"
                            "ap.add_argument('--train_batch_size', type=int, default=16)\n"
                            "ap.add_argument('--max_train_steps', type=int, default=None)\n"
                            "ap.add_argument('--allow_tf32', action='store_true')\nap.parse_args()\n")
    m, notes = detect.draft(tmp_path)
    assert m['code'] == {'mutable': ['train_text_to_image.py'], 'frozen': [], 'readable': []}
    assert notes[0] == ('Entry point: train_text_to_image.py (other candidates: train_text_to_image_lora.py); pass --entry to '
                        'use another.')
    assert m['launch']['command'] == '{python} train_text_to_image.py --max_train_steps {steps} {args}'
    assert m['hardware']['gpus'] == 1 and 'NCCL_NVLS_ENABLE' not in m['knobs']
    assert m['knobs']['train_batch_size']['default'] == 16 and 'allow_tf32' in m['knobs']
    _w(tmp_path / 'old.py', 'print "hello"\n')
    m, notes = detect.draft(tmp_path, entry='old.py')
    assert any(n.startswith('old.py is not valid Python 3 (line 1') for n in notes)


def test_cli_strategy_replaces_the_suite_strategies(tmp_path, monkeypatch, capsys):
    import shutil

    import infrarsi.project.flow as flow
    toy = Path(__file__).resolve().parents[3] / 'examples' / 'toy_cpu'
    for name in ('a', 'b'):
        shutil.copytree(toy, tmp_path / name, ignore=shutil.ignore_patterns('.infrarsi'))
    suite = _w(tmp_path / 'suite.yaml', yaml.safe_dump({'schema': 'infrarsi-suite-v1', 'projects': ['a/infrarsi.yaml', 'b/infrarsi.yaml'],
                                                        'strategies': ['tpe', 'random']}))
    calls = []

    def fake_run(project, *, strategy, **kw):
        calls.append((project.root.name, strategy))
        return {'outcome': 'no_change', 'run_dir': str(tmp_path), 'strategy': strategy, 'promotion': {}, 'search': {}}
    monkeypatch.setattr(flow, 'run_project', fake_run)
    main(['run', str(suite), '--llm', 'none'])
    assert calls == [('a', 'tpe'), ('a', 'random'), ('b', 'tpe'), ('b', 'random')]
    calls.clear()
    main(['run', str(suite), '--llm', 'none', '--strategy', 'random'])
    assert calls == [('a', 'random'), ('b', 'random')]                       # not each project twice


def _megatron_repo(tmp_path, flags, env_line=''):
    _w(tmp_path / 'mlm' / 'megatron' / 'training' / 'arguments.py', MEGATRON_ARGS)
    _w(tmp_path / 'mlm' / 'pretrain_gpt.py', 'from megatron.training import pretrain\nif __name__ == "__main__":\n    pretrain()\n')
    _w(tmp_path / 'run.sh', f'#!/bin/bash\n{env_line}\ntorchrun --nproc-per-node 8 mlm/pretrain_gpt.py --micro-batch-size 1 '
                            f'--tensor-model-parallel-size 1 {flags} "$@"\n')
    return detect.draft(tmp_path, entry='run.sh')


def test_megatron_recompute_full_needs_its_companions(tmp_path):
    m, _ = _megatron_repo(tmp_path / 'a', '--recompute-granularity selective')
    assert 'recompute_granularity' not in m['knobs']           # the launch's own value; full would need method and layers
    m, _ = _megatron_repo(tmp_path / 'b', '--recompute-granularity selective --recompute-method uniform --recompute-num-layers 1')
    assert m['knobs']['recompute_granularity']['choices'] == ['full', 'selective']
    m, _ = _megatron_repo(tmp_path / 'c', '--recompute-granularity selective --recompute-method uniform')
    assert 'recompute_granularity' not in m['knobs']


def test_megatron_fsdp_keeps_its_connections(tmp_path):
    m, _ = _megatron_repo(tmp_path / 'a', '--use-custom-fsdp', 'export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-32}')
    assert 'CUDA_DEVICE_MAX_CONNECTIONS' not in m['knobs']      # FSDP needs a value other than 1: nothing to search
    assert not any('CUDA_DEVICE_MAX_CONNECTIONS' in c for c in m['constraints'])
    m, _ = _megatron_repo(tmp_path / 'b', '', 'export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}')
    assert m['knobs']['NCCL_NVLS_ENABLE']['default'] == '0'     # the launch script's own default is the current value
    assert m['knobs']['CUDA_DEVICE_MAX_CONNECTIONS']['default'] == '1'


def test_broken_training_config_is_a_request_error(tmp_path, capsys):
    _w(tmp_path / 'train.py', 'print(1)\n')
    _w(tmp_path / 'config.yaml', 'train:\n  micro_batch_size: [1,\n')
    _w(tmp_path / 'infrarsi.yaml', yaml.safe_dump({
        'schema': 'infrarsi-project-v1', 'launch': {'command': '{python} train.py --config {config} {args}'},
        'config': {'file': 'config.yaml'}, 'knobs': {'mbs': {'type': 'int', 'path': 'train.micro_batch_size', 'choices': [1, 2]}}}))
    with pytest.raises(ManifestError, match='config.yaml cannot be parsed'):
        Project.load(tmp_path)
    with pytest.raises(SystemExit) as e:
        main(['doctor', str(tmp_path)])
    assert e.value.code == 2 and 'cannot be parsed' in capsys.readouterr().err


def test_recompute_keeps_the_current_full_from_a_config_or_constant(tmp_path):
    cfg = {'model': {'num_layers': 32, 'num_attention_heads': 8, 'recompute_granularity': 'full', 'recompute_method': 'uniform',
                     'recompute_num_layers': 1}}
    _w(tmp_path / 'a' / 'train.py', "import argparse\nfrom megatron.core.transformer import TransformerConfig\n"
                                    "ap = argparse.ArgumentParser()\nap.add_argument('--config', default='config.yaml')\nap.parse_args()\n")
    _w(tmp_path / 'a' / 'config.yaml', yaml.safe_dump(cfg))
    m, notes = detect.draft(tmp_path / 'a')
    assert m['framework'] == 'megatron' and m['knobs']['recompute_granularity']['choices'] == ['full', 'selective']
    p = Project.load(detect.write(tmp_path / 'a', m, notes))
    assert p.reference_config()['recompute_granularity'] == 'full' and p.constraint_errors(p.reference_config()) == []
    _w(tmp_path / 'b' / 'train.py', "import torch\nfrom megatron.core import parallel_state\nRECOMPUTE_GRANULARITY = 'full'\n"
                                    "for _ in range(1):\n    pass\n")
    m, _ = detect.draft(tmp_path / 'b')
    assert m['knobs']['recompute_granularity']['choices'] == ['full', 'selective']      # the current value stays reachable


def test_nvls_current_value_is_nccl_auto(tmp_path):
    _w(tmp_path / 'train.py', SFT)
    _w(tmp_path / 'config.yaml', SFT_CFG)
    m, _ = detect.draft(tmp_path)
    assert m['knobs']['NCCL_NVLS_ENABLE'] == {'type': 'str', 'env': 'NCCL_NVLS_ENABLE', 'choices': ['2', '0'], 'default': '2'}


def test_recompute_current_value_from_entry_defaults(tmp_path):
    head = "import argparse\nfrom megatron.core.transformer import TransformerConfig\nap = argparse.ArgumentParser()\n"
    tail = ("ap.add_argument('--recompute-method', default='uniform')\n"
            "ap.add_argument('--recompute-num-layers', type=int, default=1)\nap.parse_args()\n")
    for name, current in (('a', 'full'), ('b', 'selective')):
        _w(tmp_path / name / 'train.py', head + f"ap.add_argument('--recompute-granularity', default='{current}', "
                                                "choices=['full', 'selective'])\n" + tail)
        m, notes = detect.draft(tmp_path / name)
        assert m['knobs']['recompute_granularity']['choices'] == ['full', 'selective']   # the defaults give method and layers
        p = Project.load(detect.write(tmp_path / name, m, notes))
        assert p.constraint_errors(p.reference_config()) == []


def test_default_draft_lets_the_search_patch_code(tmp_path):
    # the default flow (init -> doctor -> run) never reached a patch: the draft's code.mutable was empty
    from infrarsi.project.flow import configured_patch_status, patch_status
    from infrarsi.project.hooks import ProjectHooks
    from infrarsi.search.proposers import code_view
    _w(tmp_path / 'train.py', "import argparse\nimport torch\nimport model\nfrom loss import lm_loss\nfrom data import batches\n"
                              "from eval_utils import evaluate\nimport infrarsi_probe\nap = argparse.ArgumentParser()\n"
                              "ap.add_argument('--micro-batch-size', type=int, default=2)\n"
                              "ap.add_argument('--max-steps', type=int, default=30)\nap.parse_args()\n")
    for name in ('model', 'loss', 'data', 'eval_utils', 'infrarsi_probe', 'unused'):
        _w(tmp_path / f'{name}.py', f'# {name}\nX = 1\n')
    m, notes = detect.draft(tmp_path)
    assert m['code'] == {'mutable': ['train.py', 'model.py'], 'frozen': ['eval_utils.py', 'infrarsi_probe.py', 'loss.py'],
                         'readable': ['data.py']}
    assert any(n.startswith('Code patches: the search may edit train.py, model.py (code.mutable); eval_utils.py, '
                            'infrarsi_probe.py, loss.py frozen; data.py shown, not edited.') for n in notes)
    p = Project.load(detect.write(tmp_path, m, notes))
    task = p.task()
    assert task.action_space == 'knobs+patch'
    view = code_view(ProjectHooks(p, task), None)                  # what the proposer sends: the files' text
    assert sorted(view['mutable']) == ['model.py', 'train.py'] and list(view['readable']) == ['data.py']
    assert patch_status(p) == ('on: the search may edit model.py, train.py; frozen: eval_utils.py, '
                                'infrarsi_probe.py, loss.py')
    assert patch_status(p, 'hls', 0.9).startswith('on: ')
    assert patch_status(p, 'tpe') == 'off: strategy tpe changes knobs only'
    assert patch_status(p, 'hls', 0.35) == 'off: a model of capability 0.35 (below 0.4) only rates knob moves'
    assert patch_status(p, 'hls', 0.9, has_llm=False) == 'off: no LLM (only an LLM proposes patches)'
    # doctor reads the manifest: no llm backend there -> `run --llm` decides
    assert configured_patch_status(p).startswith('on once the run has an LLM (--llm or llm.backend): the search may edit')
    for llm, status in (('none', 'off: no LLM'), ({'backend': 'none'}, 'off: no LLM'),
                        ({'backend': 'vllm', 'model': 'Qwen3-8B'}, 'off: a model of capability 0.20 (below 0.4)'),
                        ({'backend': 'claude-cli', 'model': 'opus', 'capability': None}, 'on: the search may edit')):
        p.raw['llm'] = llm
        assert configured_patch_status(p).startswith(status), llm
    p.raw['search']['strategy'] = 'random'
    assert configured_patch_status(p) == 'off: strategy random changes knobs only'
    p.raw['code']['mutable'] = []
    assert patch_status(p, 'hls', 0.9) == 'off: code.mutable lists no file, so only knobs are searched'
    # a launch script that starts a Python file outside the workdir: nothing to patch, and the notes say so
    _w(tmp_path / 'b' / 'run.sh', '#!/bin/bash\ntorchrun --nproc_per_node 8 /opt/job/train.py --micro-batch-size 2 "$@"\n')
    m, notes = detect.draft(tmp_path / 'b', entry='run.sh')
    assert m['code'] == {'mutable': [], 'frozen': [], 'readable': ['run.sh']}
    assert any(n.startswith('No Python file of the job is inside the workdir') for n in notes)


def test_megatron_drafts_log_every_update_with_its_timers(tmp_path):
    m, notes = _megatron_repo(tmp_path / 'a', '--log-interval 10')
    cmd = m['launch']['command']
    assert cmd.index('--log-interval 1') > cmd.index('run.sh') and '--timing-log-level 1' in cmd
    assert any('--timing-log-level 1' in n and 'synchronize' in n for n in notes)
    m, _ = _megatron_repo(tmp_path / 'b', '--log-interval 1 --timing-log-level 1')
    assert '--log-interval' not in m['launch']['command'] and '--timing-log-level' not in m['launch']['command']


def test_accumulation_names_take_the_ga_role():
    # U-2-1: hand-written scripts name the accumulation count in many ways; switches, accounts and fusion toggles are not it
    for name in ('accum', '--accum', 'train.accum', 'ACCUM_STEPS', 'accum_steps', 'acc_steps', 'grad_acc', 'gradient_accumulation',
                 'n_accum', 'num_accumulation_steps', 'accum_iter', 'accum_freq', 'grad_accum_every', 'accum_grad',
                 'gradient_accumulation_steps', 'grad_accum', 'grad_accum_steps', 'accumulate_grad_batches'):
        assert detect.role_of(name) == 'ga', name
    for name in ('accumulate', 'account', 'accuracy', 'acc', 'accum_dtype', 'gradient_accumulation_fusion',
                 '--no-gradient-accumulation-fusion', '--accumulate-allreduce-grads-in-fp32',
                 '--account-for-embedding-in-pipeline-split', 'fuse_wgrad_accumulation', 'device_train_grad_accum'):
        assert detect.role_of(name) is None, name


def test_an_accum_flag_or_key_is_the_accumulation_knob(tmp_path):
    _w(tmp_path / 'a' / 'train.py', "import argparse\nap = argparse.ArgumentParser()\n"
                                    "ap.add_argument('--micro-batch-size', type=int, default=2)\n"
                                    "ap.add_argument('--accum', type=int, default=4)\nap.add_argument('--accumulate', action='store_true')\n"
                                    "ap.add_argument('--max-steps', type=int, default=30)\nap.parse_args()\n")
    m, notes = detect.draft(tmp_path / 'a')
    assert m['knobs']['accum'] == {'type': 'int', 'flag': '--accum', 'choices': [1, 2, 4, 8, 16], 'default': 4}
    assert 'accumulate' not in m['knobs'] and not any(n.startswith('No gradient-accumulation knob') for n in notes)
    assert m['workload']['invariants'] == {'global_batch': 'micro_batch_size * accum * gpus'}
    # the usability study's config key; a switch or a string named like accumulation is not the count
    _w(tmp_path / 'b' / 'train.py', "import argparse\nap = argparse.ArgumentParser()\nap.add_argument('--config', default='config.json')\n"
                                    "ap.add_argument('--n-accum', action='store_true')\nap.parse_args()\n")
    _w(tmp_path / 'b' / 'config.json', json.dumps({'train': {'bs': 2, 'accum': 2}, 'precision': {'accumulation': 'fp32'}}))
    m, notes = detect.draft(tmp_path / 'b')
    assert m['knobs']['accum'] == {'type': 'int', 'path': 'train.accum', 'choices': [1, 2, 4, 8, 16]}
    assert 'n_accum' not in m['knobs'] and 'accumulation' not in m['knobs']
    assert m['workload']['invariants'] == {'global_batch': 'bs * accum * gpus'}
    # DeepSpeed's 'auto' is no count: no knob (init crashed on it)
    _w(tmp_path / 'c' / 'train.py', "import argparse\nimport deepspeed\nap = argparse.ArgumentParser()\n"
                                    "ap.add_argument('--deepspeed_config', default='ds.json')\nap.parse_args()\n")
    _w(tmp_path / 'c' / 'ds.json', json.dumps({'train_micro_batch_size_per_gpu': 'auto', 'gradient_accumulation_steps': 'auto'}))
    m, _ = detect.draft(tmp_path / 'c')
    assert not {'train_micro_batch_size_per_gpu', 'gradient_accumulation_steps'} & set(m['knobs'])


TRAIN_CKPT = ("import argparse\nimport model\nap = argparse.ArgumentParser()\nap.add_argument('--max-steps', type=int, default=30)\n"
              "ap.add_argument('--save')\nap.add_argument('--save-at', type=int)\nap.add_argument('--load')\nap.parse_args()\n")
EVAL_CKPT = ("import argparse\nap = argparse.ArgumentParser()\nap.add_argument('--load', required=True)\nap.add_argument('--bs', default=8)\n"
             "a = ap.parse_args()\nprint(f'eval_loss {1.25:.4f}', flush=True)\n")
PYTEST_FILE = 'import model\n\n\ndef test_x():\n    assert model.X == 1\n'


def _ckpt_repo(d, eval_src=EVAL_CKPT, tests=(('test_model.py', PYTEST_FILE),)):
    _w(d / 'train.py', TRAIN_CKPT)
    _w(d / 'model.py', 'X = 1\n')
    if eval_src:
        _w(d / 'eval.py', eval_src)
    for name, text in tests:
        _w(d / 'tests' / name, text)
    return detect.draft(d)


def test_checkpoint_eval_and_tests_are_drafted_when_unambiguous(tmp_path):
    # U-4-5: the toy's shape (--save / --save-at / --load, an eval.py that takes --load and prints eval_loss, the job's tests)
    m, notes = _ckpt_repo(tmp_path / 'a')
    assert m['checkpoint'] == {'save': '--save {ckpt_dir} --save-at {save_at}', 'resume': '--load {ckpt_dir}', 'save_at': 15}
    assert m['verify'] == {'tolerance': 0.03, 'eval': '{python} eval.py --load {ckpt_dir}',
                           'tests': ['{python} -m pytest tests/test_model.py -q']}
    assert any(n.startswith('Promotion checks drafted from the repository (checkpoint, verify.eval, verify.tests)') for n in notes)
    p = Project.load(detect.write(tmp_path / 'a', m, notes))
    assert p.raw['checkpoint']['save_at'] < p.steps
    # an eval whose output or arguments are not the ones run_eval needs: a note with the line to write
    for name, src, why in (('b', EVAL_CKPT.replace("f'eval_loss {", "f'loss {"), 'it prints no eval_loss or val_loss value'),
                           ('c', EVAL_CKPT.replace("'--load'", "'--checkpoint'"), 'it does not take --load'),
                           ('d', EVAL_CKPT.replace("'--bs', default=8", "'--data', required=True"), 'it requires --data too')):
        m, notes = _ckpt_repo(tmp_path / name, eval_src=src)
        assert 'eval' not in m['verify'] and 'checkpoint' in m, name
        assert any(n.startswith(f'eval.py: {why}') and 'verify.eval: "{python} eval.py' in n for n in notes), (name, notes)
    # the job's own tests run as scripts: a note; a test suite of other code (a framework checkout's): nothing
    m, notes = _ckpt_repo(tmp_path / 'e', tests=(('test_model.py', 'import model\nassert model.X == 1\n'),))
    assert 'tests' not in m['verify']
    assert any(n.startswith('tests/test_model.py: no pytest test functions') and '["{python} tests/test_model.py"]' in n
               for n in notes)
    m, notes = _ckpt_repo(tmp_path / 'f', tests=(('test_core.py', 'import torch\n\n\ndef test_y():\n    pass\n'),))
    assert 'tests' not in m['verify'] and not any(n.startswith('tests/') for n in notes)
    # check.py, the toy's test command
    _w(tmp_path / 'g' / 'check.py', 'import sys\nimport model\nsys.exit(0 if model.X == 1 else 1)\n')
    m, _ = _ckpt_repo(tmp_path / 'g', tests=())
    assert m['verify']['tests'] == ['{python} check.py']


def test_pytest_tests_without_pytest_get_a_note(tmp_path, monkeypatch):
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name, *a, **k: None if name == 'pytest' else real(name, *a, **k))
    m, notes = _ckpt_repo(tmp_path)
    assert 'tests' not in m['verify']
    assert any(n.startswith('tests/test_model.py: pytest tests, but pytest is not installed') for n in notes)


def test_save_without_a_save_at_argument_gets_a_note_not_a_block(tmp_path):
    head = "import argparse\nap = argparse.ArgumentParser()\nap.add_argument('--max-steps', type=int, default=30)\n"
    _w(tmp_path / 'a' / 'train.py', head + "ap.add_argument('--save')\nap.add_argument('--load')\nap.parse_args()\n")
    _w(tmp_path / 'a' / 'eval.py', EVAL_CKPT)
    m, notes = detect.draft(tmp_path / 'a')
    assert 'checkpoint' not in m and 'eval' not in m['verify']
    note = next(n for n in notes if n.startswith('train.py has --save, --load but not one save-directory'))
    assert ('add `checkpoint: {save: "--save {ckpt_dir} <argument that saves after update N> {save_at}", resume: "--load '
            '{ckpt_dir}", save_at: 15}`') in note
    assert any(n.startswith('eval.py: no checkpoint was drafted') for n in notes)
    Project.load(detect.write(tmp_path / 'a', m, notes))
    # an interval argument, a store_true resume and an untyped save-at are named, not drafted
    _w(tmp_path / 'b' / 'train.py', head + "ap.add_argument('--save-dir')\nap.add_argument('--save-interval', type=int)\n"
                                           "ap.add_argument('--save-at')\nap.add_argument('--resume', action='store_true')\nap.parse_args()\n")
    m, notes = detect.draft(tmp_path / 'b')
    note = next(n for n in notes if n.startswith('train.py has --save-dir, --save-interval, --save-at, --resume but'))
    assert 'checkpoint' not in m and '{save: "--save-dir {ckpt_dir} --save-at {save_at}", resume: "<load argument> {ckpt_dir}"' in note
    # the checkpoint flags a launch script passes (Megatron's)
    m, notes = _megatron_repo(tmp_path / 'c', '--save /ckpt --load /ckpt --save-interval 1000')
    assert 'checkpoint' not in m and any(n.startswith('run.sh has --save, --load, --save-interval but') for n in notes)


def _run(cwd, *argv):
    import subprocess
    import sys
    return subprocess.run([sys.executable, *argv], cwd=cwd, capture_output=True, text=True)


def test_required_arguments_keep_eval_and_check_py_undrafted(tmp_path):
    # review r1 P1; the oracle is the command itself: drafted exactly when it runs as drafted (argparse exits 2 when a
    # required positional or short-only option is missing)
    for name, extra in (('positional', "ap.add_argument('dataset')\n"), ('short', "ap.add_argument('-d', required=True)\n"),
                        ('optional', "ap.add_argument('files', nargs='*')\nap.add_argument('-v', required=False)\n")):
        d = tmp_path / name
        head = 'import argparse\nap = argparse.ArgumentParser()\n' + extra
        _w(d / 'eval.py', head + "ap.add_argument('--load', required=True)\nap.parse_args()\nprint(f'eval_loss {1.0:.4f}')\n")
        _w(d / 'check.py', head + 'ap.parse_args()\n')
        m, notes = _ckpt_repo(d, eval_src=None, tests=())
        runs = _run(d, 'eval.py', '--load', str(d)).returncode == 0, _run(d, 'check.py').returncode == 0
        assert runs == ('eval' in m['verify'], 'tests' in m['verify']) == (name == 'optional',) * 2, (name, notes)
        if name != 'optional':
            assert any(n.startswith('eval.py: it requires ') for n in notes), notes
            assert any(n.startswith('check.py: it requires ') for n in notes), notes


def test_the_pytest_command_names_only_the_jobs_test_files(tmp_path):
    # review r1 P1: the job's test next to a file that cannot even be collected here; the oracle runs the commands
    m, _ = _ckpt_repo(tmp_path, tests=(('test_model.py', PYTEST_FILE),
                                       ('test_gpu.py', 'import missing_cuda_ext\n\n\ndef test_z():\n    pass\n')))
    assert m['verify']['tests'] == ['{python} -m pytest tests/test_model.py -q']
    assert _run(tmp_path, *m['verify']['tests'][0].split()[1:]).returncode == 0
    assert _run(tmp_path, '-m', 'pytest', 'tests', '-q').returncode != 0          # the whole directory: a collection error


EVAL_PRINTS = (("print(f'eval_loss {x:.4f}')", True), ("print('val_loss:', x, flush=True)", True),
               ("print({'eval_loss': x})", True), ("print(json.dumps({'eval_loss': x}))", True),
               ("print(m['eval_loss'])", False), ("print('eval_loss')", False), ("print(f'eval_loss is {x}')", False),
               ("print(f'loss {x}')", False), ("print(f'eval_loss {x}', file=open('eval.log', 'w'))", False))


def test_eval_detection_agrees_with_what_the_runner_reads(tmp_path):
    # review r1 P1; the oracle is the script's real output read with run_eval's own pattern (a labelled value)
    from infrarsi.project.runner import EVAL_RE
    for i, (stmt, drafted) in enumerate(EVAL_PRINTS):
        f = _w(tmp_path / f'e{i}.py', f"import json\nx = 1.25\nm = {{'eval_loss': x}}\n{stmt}\n")
        out = _run(tmp_path, f.name).stdout
        read = [r.group('v') for r in EVAL_RE.finditer(out) if r.group('k').lower() != 'loss']
        assert detect.prints_eval_loss(f) is bool(read) is drafted, (stmt, out)


def test_a_resume_switch_is_not_drafted_as_the_checkpoint_path(tmp_path):
    # review r1 P2: BooleanOptionalAction takes no value; the oracle is the script's own argparse
    head = ("import argparse\nap = argparse.ArgumentParser()\nap.add_argument('--max-steps', type=int, default=30)\n"
            "ap.add_argument('--save')\nap.add_argument('--save-at', type=int)\n")
    for name, resume in (('switch', "ap.add_argument('--resume', action=argparse.BooleanOptionalAction)\n"),
                         ('path', "ap.add_argument('--resume')\n")):
        d = tmp_path / name
        _w(d / 'train.py', head + resume + 'ap.parse_args()\n')
        m, notes = detect.draft(d)
        if name == 'switch':
            assert 'checkpoint' not in m and _run(d, 'train.py', '--resume', str(d)).returncode == 2
            assert any(n.startswith('train.py has --save, --save-at, --resume but')
                       and 'resume: "<load argument> {ckpt_dir}"' in n for n in notes), notes
        else:
            for extra in (m['checkpoint']['save'], m['checkpoint']['resume']):        # both parse
                argv = extra.replace('{ckpt_dir}', str(d)).replace('{save_at}', '3').split()
                assert _run(d, 'train.py', *argv).returncode == 0, extra


def test_non_integer_accumulation_choices_stay_out_of_the_invariant(tmp_path):
    # review r1 P2: explicit choices bypassed the integer rule; the oracle is the invariant's value (a number of samples)
    head = "import argparse\nap = argparse.ArgumentParser()\nap.add_argument('--micro-batch-size', type=int, default=2)\n"
    for name, accum, ga in (('str', "ap.add_argument('--accum', choices=['fp32', 'bf16'], default='fp32')\n", 1),
                            ('float', "ap.add_argument('--accum', type=float, choices=[0.5, 1.0], default=1.0)\n", 1),
                            ('int', "ap.add_argument('--accum', type=int, choices=[1, 2, 4], default=2)\n", 2)):
        _w(tmp_path / name / 'train.py', head + accum + "ap.add_argument('--max-steps', type=int, default=30)\nap.parse_args()\n")
        m, notes = detect.draft(tmp_path / name)
        p = Project.load(detect.write(tmp_path / name, m, notes))
        gb = p.invariants(p.reference_config())['global_batch']
        assert ('accum' in m['knobs']) is (name == 'int') and type(gb) is int and gb == 2 * ga, (name, gb)


def test_hf_trainer_gets_a_checkpoint_note_not_a_block(tmp_path, monkeypatch):
    # review r1 P2: the fields seen, why no block, and the block to fill in (it loads as the manifest's checkpoint)
    import re
    monkeypatch.setattr(detect, 'hf_defaults', lambda: dict(detect.HF_FIELDS))
    _w(tmp_path / 'train_hf.py', 'from transformers import HfArgumentParser, Trainer, TrainingArguments\n'
                                 'args, = HfArgumentParser((TrainingArguments,)).parse_args_into_dataclasses()\n'
                                 'Trainer(args=args).train()\n')
    m, notes = detect.draft(tmp_path)
    note = next(n for n in notes if n.startswith('HF Trainer (TrainingArguments output_dir, save_steps, resume_from_checkpoint)'))
    assert 'checkpoint' not in m and 'carry no update number' in note and 'trainer.train()' in note
    p = Project({**m, **yaml.safe_load(re.search(r'`(checkpoint: .*)`', note).group(1))}, tmp_path / 'infrarsi.yaml')
    assert p.raw['checkpoint']['resume'] == '--resume_from_checkpoint {ckpt_dir}/checkpoint-{save_at}'
