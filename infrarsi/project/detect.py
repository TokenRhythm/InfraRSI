"""`infrarsi init`: draft an infrarsi.yaml for a training repository (the user reviews it, `doctor` checks it).

Finds the entry point - a Python script, or a launch script such as a Megatron-LM run script - the framework
(Megatron-LM / Megatron-Core, DeepSpeed, HF Trainer / Accelerate, Lightning, TorchTitan, Composer, FSDP, plain
PyTorch), the config file (yaml or json; a TorchTitan toml), and the performance knobs the job exposes: argparse
definitions of the script, of the helpers that build its parser outside it (a module of the repository or the
installed library, read from source and never imported) and of Megatron's argument module, HfArgumentParser's
dataclasses, the flags a launch script passes and the options it parses itself, HF TrainingArguments, keys of the
config file and module constants, matched against a catalogue of known throughput knobs, the experience registry's
flags and env vars, and speed-setting values of options the catalogue does not name. Whatever the names, what the code
does decides too (sinks.py): every setting that controls a framework speed call - torch.compile, DataLoader workers,
fused AdamW, TF32, autocast, DDP sync and buckets, checkpointing, Accelerate's precision, ... - traced through locals,
parameters, `self` fields and the library functions the job hands its settings to, is drafted when it does nothing
else, and mapped to the call's mechanism; so are the env switches a launch script turns into groups of flags. Batch
knobs come with the global-batch invariant, parallel sizes, Megatron's flag dependencies and the registry's
requirements with constraints.
The levers the shipped experience measured on jobs like this one that no drafted knob reaches are named in a note. The
checkpoint, eval and test commands of the promotion checks are drafted when the repository shows them unambiguously.
Anything it cannot determine is left as a note in the draft.
"""
from __future__ import annotations

import ast
import builtins
import dataclasses
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from . import sinks
from .manifest import expr_names
from .signals import VAL

try:
    import tomllib
except ModuleNotFoundError:                                  # Python 3.10
    import tomli as tomllib  # the declared Python 3.10 compatibility dependency


class InitError(ValueError):
    """Nothing can be drafted: the directory, the entry point or the config file is missing."""


FRAMEWORK_IMPORTS = (('megatron', 'megatron'), ('torchtitan', 'torchtitan'), ('composer', 'composer'),
                     ('deepspeed', 'deepspeed'), ('accelerate', 'accelerate'), ('transformers', 'hf_trainer'),
                     ('lightning', 'lightning'), ('torch.distributed.fsdp', 'fsdp'), ('torch', 'pytorch'))
SHELL_FRAMEWORKS = ((r'pretrain_\w+\.py|\bmegatron\b', 'megatron'), (r'torchtitan', 'torchtitan'),
                    (r'(^|\s)deepspeed\s', 'deepspeed'), (r'accelerate\s+launch', 'accelerate'))
SELF_DISTRIBUTED = ('megatron', 'torchtitan', 'deepspeed', 'accelerate', 'hf_trainer', 'lightning', 'composer')
DIST = re.compile(r'init_process_group|\bLOCAL_RANK\b|\blocal_rank\b|DistributedDataParallel|FullyShardedDataParallel|'
                  r'\bfully_shard\(|\bAccelerator\(|deepspeed\.initialize|init_distributed|torch\.distributed\b')
DOTLIST = re.compile(r'from_cli\(|from_dotlist\(|@hydra\.main|hydra\.main\(|OmegaConf\.from_cli|om\.from_cli')
HF_ARGS = re.compile(r'\b(TrainingArguments|SFTConfig|DPOConfig|GRPOConfig|ORPOConfig|KTOConfig)\b')
HF_PARSER = re.compile(r'HfArgumentParser|TrlParser|parse_args_into_dataclasses')
LAUNCHER = re.compile(r'torchrun|torch\.distributed\.(run|launch)|deepspeed |accelerate launch|pretrain_\w+\.py|python3? \S+\.py')
ENTRY_NAMES = ('train', 'pretrain', 'finetune', 'sft', 'main', 'run')
STEP_ARGS = ('max_steps', 'max-steps', 'train_iters', 'train-iters', 'max_train_steps', 'total_steps', 'num_steps',
             'max_iters', 'iters', 'steps')
STEP_KEYS = ('max_steps', 'train_iters', 'total_steps', 'max_iters', 'num_steps', 'max_duration')
DRAFT_STEPS = 30                                             # workload.steps of a draft; its checkpoint is saved half-way
CONFIG_SUFFIX = re.compile(r'\.(ya?ml|json|toml)$')
CONFIG_ARG = re.compile(r'(^|_)(config|cfg|conf|config_file|config_path|deepspeed|deepspeed_config|ds_config)$')
TOOLING = re.compile(r'^(\.|mkdocs|environment|docker-compose|compose|codecov|action|chart|conda|renovate|dependabot|'
                     r'readthedocs|pre-commit|citation|tsconfig|package|infrarsi)', re.I)
# (regex on the normalized knob name, role); role decides choices, invariant and constraint handling. 'bs' is a batch
# size that may be per device or global (resolved in draft()).
KNOWN = (
    (r'^(tensor_model_parallel_size|tp|tp_size|tensor_parallel_degree)$', 'tp'),
    (r'^(pipeline_model_parallel_size|pp|pp_size|pipeline_parallel_degree)$', 'pp'),
    (r'^(context_parallel_size|cp|cp_size|context_parallel_degree)$', 'cp'),
    (r'^(expert_model_parallel_size|ep|ep_size|expert_parallel_degree)$', 'ep'),
    (r'^(micro_batch_size|per_device_train_batch_size|train_micro_batch_size_per_gpu|device_train_microbatch_size|'
     r'device_batch_size|micro_bsz|mbs|local_batch_size)$', 'mbs'),
    (r'^(batch_size|train_batch_size|bs)$', 'bs'),
    (r'^(sharding_strategy|fsdp_sharding_strategy)$', 'shard'),
    # accum, n_accum, grad_acc, gradient_accumulation, accum_iter, ...; not accumulate (a switch), account, *_fusion
    (r'^(((n|num)_)?(grad(ient)?_)?(accum|accumulation)(_(steps?|iters?|every|freq|grad|batches))?|(grad(ient)?_)?acc_steps?|'
     r'grad(ient)?_acc|accumulate_grad_batches)$', 'ga'),
    (r'^(recompute_granularity|activation_checkpoint_mode|ac_mode)$', 'recompute'),
    (r'^(attn|attn_impl|attn_implementation|attention_impl|attention_implementation|attention_backend|attn_backend)$',
     'attn'),
    (r'^(bf16|fp16|tf32|allow_tf32|fp8|amp|use_amp|use_bf16|use_fp16|use_tf32)$', 'switch'),
    (r'(recompute|gradient_checkpoint|grad_checkpoint|grad_ckpt|activation_checkpoint|activation_ckpt|act_ckpt|'
     r'checkpoint_activations)', 'switch'),
    (r'(sequence_parallel|use_distributed_optimizer|overlap_grad_reduce|overlap_param_gather|tp_comm_overlap|'
     r'overlap_comm|contiguous_gradients|use_flash_attn|flash_attn|fused|compile|torch_compile|pin_memory|'
     r'cross_entropy_loss_fusion|bias_gelu_fusion|liger|xformers|memory_efficient_attention|foreach|'
     r'reshard_after_forward|cpu_offload|offload)', 'switch'),
    (r'(num_workers|dataloader_num_workers|data_workers|^workers$)', 'workers'),
    (r'^(precision|dtype|mixed_precision|compute_dtype|param_dtype)$', 'precision'),
    (r'(zero_stage|^stage$)', 'zero'),
    (r'(bucket_cap_mb|bucket_size|reduce_bucket_size|allgather_bucket_size)', 'bucket'),
)
GENERIC_LEAVES = ('enabled', 'enable', 'mode')               # bf16.enabled -> bf16, activation_checkpoint.mode -> ..._mode
ATTN_IMPLS = ('eager', 'sdpa', 'flash_attention_2', 'flash_attention_3', 'flex_attention')
LIGHTNING_PRECISION = ('32-true', '32', 'bf16-mixed', '16-mixed', 'bf16-true', '16-true', 'bf16', '16')
# first = the current value when the launch does not set one (NVLS 2: NCCL's automatic mode, its default)
ENV_KNOBS = {'PYTORCH_CUDA_ALLOC_CONF': ['', 'expandable_segments:True'], 'NCCL_NVLS_ENABLE': ['2', '0']}
MEGATRON_ENV = {'CUDA_DEVICE_MAX_CONNECTIONS': ['1', '32']}
# Megatron flags worth searching; drafted when this Megatron's argument module defines them (without it: the switches)
MEGATRON_ARGS = ('megatron/training/arguments.py', 'megatron/arguments.py')
MEGATRON_KNOBS = ('--micro-batch-size', '--tensor-model-parallel-size', '--pipeline-model-parallel-size',
                  '--context-parallel-size', '--sequence-parallel', '--use-distributed-optimizer', '--overlap-grad-reduce',
                  '--overlap-param-gather', '--tp-comm-overlap', '--recompute-granularity', '--recompute-activations',
                  '--cross-entropy-loss-fusion', '--num-workers')
MEGATRON_SWITCHES = ('--sequence-parallel', '--use-distributed-optimizer', '--overlap-grad-reduce', '--overlap-param-gather',
                     '--tp-comm-overlap', '--recompute-activations', '--cross-entropy-loss-fusion')
# TrainingArguments fields drafted for an HfArgumentParser script (defaults read from the installed transformers)
HF_FIELDS = {'per_device_train_batch_size': 8, 'gradient_accumulation_steps': 1, 'gradient_checkpointing': False,
             'bf16': False, 'tf32': None, 'dataloader_num_workers': 0, 'torch_compile': False, 'optim': None}
HF_OPTIMS = ('adamw_torch', 'adamw_torch_fused')
# where else a job's knobs live, read from source (never imported or run): argument builders by name, HfArgumentParser's
# own dataclasses, the speed words that make an option of unknown role a knob, env var reads, image / video jobs
ARG_FUNC = re.compile(r'(?i)^(?:(?:parse|get|build|make|add|create|setup|init)_?(?:args|arguments|argparser|parser|opts|options|'
                      r'config|cfg|cli|flags|hparams)|args|arguments|argparser)$')
HF_ARG_CLASSES = {'TrainingArguments', 'Seq2SeqTrainingArguments', 'SFTConfig', 'DPOConfig', 'GRPOConfig', 'ORPOConfig',
                  'KTOConfig'}
PERF_NAME = re.compile(r'(?i)(?<![a-z])(?:zero_?(?:stage|[0-3])|fsdp2?|ddp|deepspeed|offload(?:ing)?|sharding|sharded|compiled?|'
                       r'inductor|cuda_?graphs?|bf16|fp16|fp8|tf32|amp|mixed_precision|flash(?:_?attn|_?attention)?\d?|sdpa|'
                       r'xformers|cudnn|(?:attn|attention)_?(?:impl(?:ementation)?|backend|kernel)|fused|(?<=_)fusion|liger|'
                       r'triton|channels_last|prefetch(?:ing)?|pin_memory|recompute|overlap(?:ping)?)(?![a-z])')
PERF_VALUE = re.compile(r'(?i)(?<![a-z])(?:zero_?[0-3]|fsdp2?|ddp|deepspeed|offload(?:ing)?|compiled?|eager|inductor|cuda_?graphs?|'
                        r'bf16|fp16|fp8|tf32|amp|flash(?:_?attn|_?attention)?(?:_?\d)?|sdpa|xformers|cudnn|fused|liger|'
                        r'triton|channels_last)(?![a-z])')     # not a model's or the data's words: zero_init, attention_type
NOT_STEP = re.compile(r'(?i)(?<![a-z])(?:npu|tpu|xla|mps|hpu|xpu|ipex|save|export|eval|evaluation|val|validation|inference|'
                      r'infer|test|prior|sample|sampling|log|logging)(?![a-z])')  # another device; not the training step
ENV_READ = re.compile(r'''(?:environ(?:\.get)?\s*[\[(]|getenv\s*\()\s*['"]([A-Z][A-Z0-9_]{2,})['"]''')
DIFFUSION_JOB = re.compile(r'(?i)\b(?:diffusers|diffsynth|diffusion|unet|dit)\b')


def norm(name: str) -> str:
    return name.lstrip('-').replace('-', '_').lower()


def role_of(name: str) -> str | None:
    n = norm(name).split('.')[-1]
    if NOT_STEP.search(n):                               # --eval_compile, --save_fp16: not the training step
        return None
    for rx, role in KNOWN:
        if re.search(rx, n):
            return role
    return None


def leaf_role(path: str) -> tuple[str, str | None]:
    """(knob name, role) of a config leaf; a generic leaf takes its parent's name."""
    parts = path.split('.')
    name = norm(parts[-1])
    if name in GENERIC_LEAVES and len(parts) > 1:
        name = norm(parts[-2]) if name in ('enabled', 'enable') else f'{norm(parts[-2])}_{name}'
    return name, role_of(name)


def choices_for(role: str, current, kind: str, name: str = '') -> list | None:
    if kind == 'bool':
        return None
    if role in ('tp', 'pp', 'cp', 'ep'):
        return [1, 2, 4, 8]
    if role in ('mbs', 'ga', 'bs'):                         # not for DeepSpeed's 'auto' or a dtype named like them
        return [1, 2, 4, 8, 16] if current is None or type(current) is int else None
    if role == 'workers':
        return [0, 2, 4, 8]
    if role == 'zero':
        return [1, 2, 3]
    if role == 'precision' and str(current).lower() in ('fp32', 'float32', 'bf16', 'bfloat16', 'fp16', 'float16', 'fp8'):
        return sorted({str(current), 'bf16', 'fp32'})          # lossy changes are governed by the loss tolerance
    if role == 'precision' and str(current).lower().startswith('amp_'):
        return sorted({str(current), 'amp_bf16', 'amp_fp8'})  # composer / llm-foundry precision names
    if role == 'precision' and str(current) in LIGHTNING_PRECISION:
        return sorted({str(current), 'bf16-mixed', '32-true'})
    if role == 'recompute' and str(current).lower() in ('none', 'selective', 'full'):
        return ['none', 'selective', 'full'] if name.endswith('mode') else ['full', 'selective']
    if role == 'attn' and str(current) in ATTN_IMPLS:
        return sorted({str(current), 'sdpa', 'flash_attention_2'})
    if role == 'shard' and str(current).upper() in ('FULL_SHARD', 'SHARD_GRAD_OP', 'HYBRID_SHARD', 'NO_SHARD', '_HYBRID_SHARD_ZERO2'):
        return ['FULL_SHARD', 'SHARD_GRAD_OP', 'HYBRID_SHARD']
    if role == 'bucket' and isinstance(current, (int, float)) and current > 0:
        return sorted({type(current)(current / 4), type(current)(current / 2), current, type(current)(current * 2),
                       type(current)(current * 4)})
    return None


def merged(choices, current) -> list:
    """The choices with the current value added (it must stay reachable)."""
    if current is None or current in choices:
        return list(choices)
    try:
        return sorted(set(choices) | {current})
    except TypeError:
        return [*choices, current]


def kind_of(v) -> str:
    return 'bool' if isinstance(v, bool) else ('int' if isinstance(v, int) else ('float' if isinstance(v, float) else 'str'))


def typed(v: str | None):
    if v is None:
        return None
    if re.fullmatch(r'[+-]?\d+', v):
        return int(v)
    if re.fullmatch(r'[+-]?(\d+\.\d*|\.\d+|\d+(\.\d*)?[eE][+-]?\d+)', v):
        return float(v)
    return {'true': True, 'false': False}.get(v.lower(), v)


def framework(root: Path, files: list[Path]) -> str:
    return framework_of([f.read_text(encoding='utf-8', errors='replace') for f in files[:50]]) or 'pytorch'


def framework_of(texts: list[str]) -> str | None:
    text = '\n'.join(t[:200_000] for t in texts)
    for mod, fw in FRAMEWORK_IMPORTS:                   # also `import torch, os, argparse, accelerate` (DiffSynth)
        if re.search(rf'^\s*(?:from\s+{re.escape(mod)}\b|import\s+(?:[\w.]+(?:\s+as\s+\w+)?\s*,\s*)*{re.escape(mod)}\b)',
                     text, re.M):
            if fw == 'hf_trainer' and not re.search(r'\bTrainer\b|\bSFTTrainer\b', text):
                continue
            return fw
    return None


def module_constants(script: Path) -> list[dict]:
    """Module-level UPPER_CASE assignments of a script (nanoGPT / autoresearch-style configuration); value is None
    when it is computed (e.g. 2**19) or not a scalar."""
    try:
        tree = ast.parse(script.read_text(encoding='utf-8', errors='replace'))
    except SyntaxError:
        return []
    out = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if isinstance(target, ast.Name) and target.id.isupper():
            try:
                v = ast.literal_eval(value)
            except (ValueError, SyntaxError, TypeError):
                v = None
            out.append({'name': target.id, 'value': v if isinstance(v, (bool, int, float, str)) else None})
    return out


def load_config(p: Path | None):
    """The parsed yaml / json / toml file, or None."""
    if p is None or not p.is_file():
        return None
    try:
        text = p.read_text(encoding='utf-8')
        if p.suffix == '.json':
            return json.loads(text)
        if p.suffix == '.toml':
            return tomllib.loads(text) if tomllib else None
        return yaml.safe_load(text)
    except (ValueError, yaml.YAMLError, UnicodeDecodeError):
        return None


def perf_keys(doc) -> int:
    return sum(1 for p, _ in config_leaves(doc) if leaf_role(p)[1]) if isinstance(doc, dict) else 0


def find_config(root: Path, named: list[Path] = (), exclude=()) -> Path | None:
    """The job's config: the file with the most known performance keys among those the entry point names (argument
    defaults, launch-script flags), else among yaml/json files near it (top level, then configs/yamls/conf trees,
    depth <= 4; tooling files and `exclude`, the launcher's own configs, skipped); a named file without such keys is
    still the config."""
    named = [p for p in named if p.is_file()]
    skip = {p.resolve() for p in exclude}
    cands = named or [p for p in sorted(root.glob('*.yaml')) + sorted(root.glob('*.yml')) + sorted(root.glob('*.json'))
                      if not TOOLING.match(p.name) and p.resolve() not in skip]
    for d in ('configs', 'config', 'yamls', 'conf', 'recipes'):
        cands += sorted(p for p in (root / d).rglob('*.y*ml') if len(p.relative_to(root).parts) <= 5
                        and p.resolve() not in skip) if (root / d).is_dir() and not named else []
    best, best_n = None, 0
    for p in cands:
        n = perf_keys(load_config(p))
        if n > best_n:
            best, best_n = p, n
    return best or (named[0] if named else None)


def runs_something(script: Path) -> bool:
    """True for a script that does work when executed (top-level calls or loops, or a __main__ guard); False for a
    library module (imports, definitions, assignments only)."""
    try:
        tree = ast.parse(script.read_text(encoding='utf-8', errors='replace'))
    except SyntaxError:
        return False
    for n in tree.body:
        if isinstance(n, (ast.For, ast.While, ast.With)) or (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)):
            return True
        if isinstance(n, ast.If) and '__main__' in ast.unparse(n.test):
            return True
    return False


def _rank(paths: list[tuple[int, Path]]) -> list[Path]:
    """Name priority (train, pretrain, ...), then the location group, then the shorter (less specialized) name."""
    def key(item):
        group, p = item
        s = p.stem.lower()
        return next((i for i, n in enumerate(ENTRY_NAMES) if s.startswith(n)), len(ENTRY_NAMES)), group, len(s), s
    return [p for _, p in sorted(paths, key=key)]


def entry_candidates(root: Path) -> list[Path]:
    """Python entry points under root, best first."""
    seen, found = set(), []
    for group, paths in enumerate((sorted(root.glob('*.py')), sorted(root.glob('scripts/*.py')), sorted(root.glob('*/train*.py')))):
        for p in paths:
            if p not in seen and runs_something(p):
                seen.add(p)
                found.append((group, p))
    return _rank(found)


def entry_script(root: Path) -> Path | None:
    ranked = entry_candidates(root)
    return ranked[0] if ranked else None


def shell_candidates(root: Path) -> list[Path]:
    """Launch scripts (top level and scripts/) that start a training command, best first."""
    found = [(g, p) for g, paths in enumerate((sorted(root.glob('*.sh')), sorted(root.glob('scripts/*.sh')))) for p in paths
             if LAUNCHER.search(p.read_text(encoding='utf-8', errors='replace'))]
    return _rank(found)


def is_shell(script: Path) -> bool:
    if script.suffix in ('.sh', '.bash'):
        return True
    if script.suffix == '.py':
        return False
    head = script.read_text(encoding='utf-8', errors='replace')[:200]
    return head.startswith('#!') and 'sh' in head.split('\n', 1)[0]


def syntax_error(script: Path) -> str | None:
    try:
        ast.parse(script.read_text(encoding='utf-8', errors='replace'))
    except SyntaxError as exc:
        return f'line {exc.lineno}: {exc.msg}'
    return None


def imported_modules(script: Path) -> set[str]:
    """Top-level names of the modules a script imports (`from .mod import x` counts as mod)."""
    try:
        tree = ast.parse(script.read_text(encoding='utf-8', errors='replace'))
    except SyntaxError:
        return set()
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names |= {a.name.split('.')[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level <= 1:
            names.add(n.module.split('.')[0])
    return names


def local_modules(script: Path) -> list[Path]:
    """Python files next to the script that it imports (one level)."""
    return [script.parent / f'{m}.py' for m in sorted(imported_modules(script)) if (script.parent / f'{m}.py').is_file()]


FROZEN_STEM = re.compile(r'(?i)(?:^|_)(?:loss(?:es)?|criterion|eval(?:uate|uation|uator)?|metrics?|scor(?:e|ing)|'
                         r'valid(?:ate|ation)?|tests?)(?:_|$)')
DATA_STEM = re.compile(r'(?i)(?:^|_)(?:data(?:sets?|loaders?|modules?)?|tokeni[sz](?:e|er|ation)|preprocess(?:ing)?|'
                       r'collat(?:e|or|ion)|prepare)(?:_|$)')


def code_section(root: Path, py: Path | None) -> dict:
    """What the search may patch: the Python file the job runs and the local modules it imports (one level). Modules
    named like a loss, metric or evaluation, and a copy of InfraRSI's probe (the measurement), are frozen; data
    modules are shown, not patched (they decide what the job trains on)."""
    out = {'mutable': [], 'frozen': [], 'readable': []}
    if py is None or not py.is_relative_to(root):
        return out
    out['mutable'].append(py.relative_to(root).as_posix())
    for m in local_modules(py):
        frozen = FROZEN_STEM.search(m.stem) or m.stem.lower().startswith('infrarsi')
        kind = 'frozen' if frozen else 'readable' if DATA_STEM.search(m.stem) else 'mutable'
        out[kind].append(m.relative_to(root).as_posix())
    return out


def module_name(root: Path, script: Path) -> str | None:
    """`pkg.mod` for a script inside a package under root (run with -m so that its absolute imports resolve)."""
    if not script.is_relative_to(root):
        return None
    parts = list(script.relative_to(root).with_suffix('').parts)
    if len(parts) < 2 or not all((root.joinpath(*parts[:i]) / '__init__.py').is_file() for i in range(1, len(parts))):
        return None
    return '.'.join(parts)


def argparse_args(script: Path) -> list[dict]:
    """add_argument('--x', type=int, default=.., choices=[..], action='store_true') definitions of a script (--long
    options); one_value: the option takes exactly one value (not a switch, a list or a custom action)."""
    tree = _tree(script)
    return node_args(tree) if tree is not None else []


_TREES: dict = {}


def _tree(path: Path):
    """The parsed module at path (None: unreadable or not Python 3), one tree per file version: the tracer of
    sinks.py keys what a function's parameters receive by its node."""
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _TREES:
        if len(_TREES) > 2000:
            _TREES.clear()
        try:
            _TREES[key] = ast.parse(path.read_text(encoding='utf-8', errors='replace'))
        except (SyntaxError, ValueError, OSError):
            _TREES[key] = None
    return _TREES[key]


def node_args(tree) -> list[dict]:
    """The add_argument calls under an AST node (a module, a function), as argparse_args gives them."""
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'):
            continue
        flags = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith('--')]
        if not flags:
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        lit = lambda n: ast.literal_eval(n) if n is not None else None          # noqa: E731
        try:
            default, choices = lit(kw.get('default')), lit(kw.get('choices'))
        except (ValueError, SyntaxError):
            default, choices = None, None
        action = kw.get('action').value if isinstance(kw.get('action'), ast.Constant) else None
        typ = kw.get('type').id if isinstance(kw.get('type'), ast.Name) else None
        kind = 'bool' if action in ('store_true', 'store_false') else {'int': 'int', 'float': 'float'}.get(typ, 'str')
        nargs = kw.get('nargs')                 # an action given as an object (BooleanOptionalAction) is not a store
        one = (kw.get('action') is None or action == 'store') and (nargs is None or getattr(nargs, 'value', None) == '?')
        dest = kw['dest'].value if isinstance(kw.get('dest'), ast.Constant) else None    # the code reads args.<dest>
        out.append({'flag': flags[0], 'name': norm(flags[0]), 'kind': kind, 'default': default, 'choices': choices,
                    'action': action, 'one_value': one, **({'dest': dest} if dest else {})})
    return out


def required_args(script: Path) -> list[str]:
    """The arguments a script cannot run without: options declared required (short-only ones too; an expression counts
    as required) and positionals other than nargs '?' / '*' / REMAINDER."""
    try:
        tree = ast.parse(script.read_text(encoding='utf-8', errors='replace'))
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'):
            continue
        names = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        req, nargs = kw.get('required'), kw.get('nargs')
        if not names:
            continue
        if names[0].startswith('-'):
            if req is not None and not (isinstance(req, ast.Constant) and not req.value):
                out.append(next((n for n in names if n.startswith('--')), names[0]))
        elif not (getattr(nargs, 'value', None) in ('?', '*', '...') or getattr(nargs, 'attr', None) in
                  ('OPTIONAL', 'ZERO_OR_MORE', 'REMAINDER')):
            out.append(names[0])
    return out


def outside_parser_args(script: Path) -> list[str]:
    """`add_general_config (diffsynth.diffusion)` for each function the script hands its ArgumentParser to that it does
    not define (`parser = add_general_config(parser)`): the arguments those add are not read (DiffSynth, R2-04)."""
    try:
        tree = ast.parse(script.read_text(encoding='utf-8', errors='replace'))
    except SyntaxError:
        return []
    nodes = list(ast.walk(tree))
    defined = {n.name for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    parsers = {t.id for n in nodes if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
               and getattr(n.value.func, 'attr', getattr(n.value.func, 'id', None)) == 'ArgumentParser'
               for t in n.targets if isinstance(t, ast.Name)}
    origin = {a.asname or a.name: n.module for n in nodes if isinstance(n, ast.ImportFrom) and n.module for a in n.names}
    star = ' or '.join(sorted({n.module for n in nodes if isinstance(n, ast.ImportFrom) and n.module
                               and any(a.name == '*' for a in n.names)}))
    out = []
    for n in sorted((n for n in nodes if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)),
                    key=lambda n: (n.lineno, n.col_offset)):
        f = n.func.id
        if f in defined or hasattr(builtins, f) or not any(isinstance(a, ast.Name) and a.id in parsers for a in n.args):
            continue
        where = origin.get(f) or star
        if (item := f'{f} ({where})' if where else f) not in out:
            out.append(item)
    return out


def module_file(name: str, here: Path, root: Path, level: int = 0) -> Path | None:
    """The source of module `name` as the file `here` imports it, found without importing anything: in its package for
    a relative import (level > 0), else a module of the repository (next to `here`, then under root), else the installed
    one (the top-level package's spec - finding a top-level name runs no code - and its submodules' files). None: not
    here, part of the standard library, or not Python source."""
    parts = [p for p in name.split('.') if p]
    base = here.parent
    for _ in range(level - 1):
        base = base.parent
    for b in [base] if level else [here.parent, root]:
        p = b.joinpath(*parts)
        for c in ([p.with_suffix('.py')] if parts else []) + [p / '__init__.py']:
            if c.is_file():
                return c
    if level or not parts or parts[0] in sys.stdlib_module_names:
        return None
    try:
        spec = importlib.util.find_spec(parts[0])
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    if len(parts) == 1:
        return Path(spec.origin) if spec.origin and spec.origin.endswith('.py') else None
    for loc in spec.submodule_search_locations or []:
        p = Path(loc).joinpath(*parts[1:])
        for c in (p.with_suffix('.py'), p / '__init__.py'):
            if c.is_file():
                return c
    return None


def _imports(tree) -> dict:
    """A module's imports: name -> (module, imported name or None for a module, level); '*' -> [(module, level)]."""
    out = {'*': []}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                if a.name == '*':
                    out['*'].append((n.module or '', n.level))
                else:
                    out[a.asname or a.name] = (n.module or '', a.name, n.level)
        elif isinstance(n, ast.Import):
            for a in n.names:
                out[a.asname or a.name.split('.')[0]] = (a.name if a.asname else a.name.split('.')[0], None, 0)
    return out


def find_def(path: Path, name: str, root: Path, depth: int = 0):
    """(node, file, tree) of the function or class `name` in the module at `path`, through the module's re-exports
    (`from .x import name`, `from .x import *`) a few levels; None when its source does not define it."""
    tree = _tree(path) if depth <= 4 else None
    if tree is None:
        return None
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name:
            return n, path, tree
    for n in tree.body:
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                if a.name == '*' or (a.asname or a.name) == name:
                    src = module_file(n.module or '', path, root, n.level)
                    hit = src is not None and src != path and find_def(src, name if a.name == '*' else a.name, root, depth + 1)
                    if hit:
                        return hit
    return None


def resolve_call(func, path: Path, tree, root: Path):
    """((node, file, tree) or None, missing) of what a call in this module reaches through its own definitions and its
    imports; `missing` names a callee whose module has no source here (`add_general_config (diffsynth.diffusion)`: a
    library not installed where init runs). The standard library is not followed."""
    imp = _imports(tree)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in imp:
        mod, sub, level = imp[func.value.id]              # mod.f(...): `import mod`, or `from pkg import mod`
        mod = f'{mod}.{sub}' if sub and mod else sub or mod
        if not level and mod.split('.')[0] in sys.stdlib_module_names:
            return None, None
        src = module_file(mod, path, root, level)
        return (find_def(src, func.attr, root) if src else None), (None if src else f'{func.attr} ({mod})')
    if not isinstance(func, ast.Name) or hasattr(builtins, func.id):
        return None, None
    local = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                  and n.name == func.id), None)
    if local is not None:
        return (local, path, tree), None
    if func.id in imp and imp[func.id][1] is not None:
        mod, orig, level = imp[func.id]
        if not level and mod.split('.')[0] in sys.stdlib_module_names:
            return None, None
        src = module_file(mod, path, root, level)
        return (find_def(src, orig, root) if src else None), (None if src else f'{func.id} ({mod})')
    lost = []
    for mod, level in imp['*']:
        src = module_file(mod, path, root, level)
        if src is None:
            lost.append(mod)
        elif hit := find_def(src, func.id, root):
            return hit, None
    return None, (f'{func.id} ({" or ".join(sorted(lost))})' if lost else None)


def _parser_names(node, params: bool = True) -> set[str]:
    """Names that hold an argument parser in a module or function: assigned from ArgumentParser, a subcommand's parser
    or an argument group, and (params) a function's parameters - a helper handed the parser."""
    fn = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    names = {a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs} if params and fn else set()
    for n in ast.walk(node):
        if isinstance(n, (ast.Assign, ast.AnnAssign)) and isinstance(n.value, ast.Call) and (getattr(n.value.func, 'attr', None) or getattr(
                n.value.func, 'id', None)) in ('ArgumentParser', 'ArgParser', 'add_parser', 'add_argument_group',
                                               'add_mutually_exclusive_group'):
            names |= {t.id for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) if isinstance(t, ast.Name)}
    return names


def dataclass_args(cls: ast.ClassDef, dash: bool = False) -> list[dict]:
    """The command-line arguments HfArgumentParser (and pyrallis, draccus, simple_parsing; tyro with dashes: dash)
    makes of a dataclass: `--field` for each annotated field (a bool takes `--field true|false`); choices from
    `field(metadata={'choices': [...]})` or a `Literal[...]` annotation."""
    out = []
    for n in cls.body:
        if not (isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)):
            continue
        ann, v, choices = ast.unparse(n.annotation), n.value, None
        if isinstance(n.annotation, ast.Subscript) and ast.unparse(n.annotation.value).split('.')[-1] == 'Literal':
            try:
                lit = ast.literal_eval(n.annotation.slice)
                choices = list(lit) if isinstance(lit, tuple) else [lit]
            except (ValueError, SyntaxError):
                pass
        if isinstance(v, ast.Call) and (getattr(v.func, 'id', None) or getattr(v.func, 'attr', None)) == 'field':
            kw = {k.arg: k.value for k in v.keywords if k.arg}
            if isinstance(kw.get('init'), ast.Constant) and kw['init'].value is False:
                continue                                 # not an argument: HfArgumentParser skips init=False fields
            v = kw.get('default')
            try:
                meta = ast.literal_eval(kw['metadata']) if 'metadata' in kw else {}
            except (ValueError, SyntaxError):
                meta = {}
            choices = (meta.get('choices') if isinstance(meta, dict) else None) or choices
        try:
            default = ast.literal_eval(v) if v is not None else None
        except (ValueError, SyntaxError):
            default = None
        kind = 'bool' if re.search(r'\bbool\b', ann) else 'int' if re.search(r'\bint\b', ann) else 'float' if 'float' in ann else 'str'
        if dash and kind == 'bool' and default:
            continue                                     # tyro: on by default, turned off by --no-x: not drafted
        if kind == 'bool' and default is None:
            continue                                     # Optional[bool] = None: unset is a third state, not False
        out.append({'flag': '--' + (n.target.id.replace('_', '-') if dash else n.target.id), 'name': norm(n.target.id),
                    'kind': kind, 'default': default,
                    'choices': list(choices) if choices else None, 'action': 'store_true' if dash and kind == 'bool' else None,
                    'one_value': True, **({'style': 'value'} if kind == 'bool' and not dash else {})})
    return out


# dataclass CLIs: the call that parses a dataclass into flags -> its flags spell field names with dashes (tyro)
DATACLASS_CLI = {'tyro.cli': True, 'pyrallis.parse': False, 'draccus.parse': False, 'simple_parsing.parse': False,
                 'jsonargparse.CLI': False, 'jsonargparse.auto_cli': False}
DATACLASS_WRAP = ('pyrallis.wrap', 'draccus.wrap')     # @pyrallis.wrap() def main(cfg: Config)


def _arg(flag: str, dest: str, kind: str, default, choices=None, switch=False) -> dict:
    return {'flag': flag, 'name': norm(dest), 'kind': kind, 'default': default, 'choices': list(choices) if choices else None,
            'action': 'store_true' if switch else None, 'one_value': not switch,
            **({'style': 'value'} if kind == 'bool' and not switch else {})}      # click type=bool: --x true|false


def function_cli(tree) -> dict:
    """{function name: [option]} of CLIs built from a function: click's @click.option decorators, typer's parameters
    (@app.command(), typer.run(f)), fire.Fire(f)'s - each option with the flag its library parses and the parameter
    it fills (the tracer of sinks.py binds the parameter to the option)."""
    imp, out = _imports(tree), {}
    runs = {a.id for n in ast.walk(tree) if isinstance(n, ast.Call) and (sinks.qualname(n.func, imp) or '') in
            ('typer.run', 'fire.Fire') for a in n.args[:1] if isinstance(a, ast.Name)}
    fire = {a.id for n in ast.walk(tree) if isinstance(n, ast.Call) and (sinks.qualname(n.func, imp) or '') == 'fire.Fire'
            for a in n.args[:1] if isinstance(a, ast.Name)}
    for fn in (n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
        opts, typer = [], fn.name in runs or any(isinstance(d, ast.Call) and getattr(d.func, 'attr', None) == 'command' and
                                                 not (sinks.qualname(d.func, imp) or '').startswith('click') for d in fn.decorator_list)
        for d in fn.decorator_list:                      # click
            if isinstance(d, ast.Call) and (getattr(d.func, 'attr', None) or getattr(d.func, 'id', None)) == 'option':
                names = [a.value for a in d.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                longs = [n.split('/')[0] for n in names if n.startswith('--')]
                if not longs:
                    continue
                kw = {k.arg: k.value for k in d.keywords if k.arg}
                dest = next((n for n in names if not n.startswith('-')), longs[0].lstrip('-').replace('-', '_'))
                t, default = kw.get('type'), sinks.lit(kw['default']) if 'default' in kw else None
                choices = sinks.lit(t.args[0]) if isinstance(t, ast.Call) and getattr(t.func, 'attr', None) == 'Choice' and t.args else None
                switch = (isinstance(kw.get('is_flag'), ast.Constant) and kw['is_flag'].value is True) or any('/' in n for n in names)
                if switch and (default is True or (isinstance(kw.get('flag_value'), ast.Constant) and kw['flag_value'].value is False)):
                    continue                             # --no-tf32 / on by default: the flag turns it off, not drafted
                kind = 'bool' if switch else {'int': 'int', 'float': 'float', 'bool': 'bool'}.get(getattr(t, 'id', None), 'str')
                if kind == 'bool' and not switch and (default is None or default is sinks.UNKNOWN):
                    continue                             # type=bool without a default: unset (None) is a third state, not False
                opts.append(_arg(longs[0], dest, kind, None if default is sinks.UNKNOWN else default,
                                 None if choices is sinks.UNKNOWN else choices, switch))
        if typer or fn.name in fire:                     # typer / fire: the parameters
            args = fn.args.args + fn.args.kwonlyargs
            defaults = [None] * (len(fn.args.args) - len(fn.args.defaults)) + fn.args.defaults + fn.args.kw_defaults
            for a, d in zip(args, defaults):
                meta, ann = None, a.annotation          # Annotated[int, typer.Option('--workers')] = 4
                if isinstance(ann, ast.Subscript) and ast.unparse(ann.value).split('.')[-1] == 'Annotated' and \
                        isinstance(ann.slice, ast.Tuple):
                    meta = next((x for x in ann.slice.elts[1:] if isinstance(x, ast.Call)), None)
                if isinstance(d, ast.Call) and (getattr(d.func, 'attr', None) or getattr(d.func, 'id', None)) in ('Option', 'Argument'):
                    meta, d = d, (d.args[0] if d.args else next((k.value for k in d.keywords if k.arg == 'default'), None))
                if meta is not None and (getattr(meta.func, 'attr', None) or getattr(meta.func, 'id', None)) == 'Argument':
                    continue                             # a positional argument: no flag to draft
                if d is None:
                    continue
                flags = [x.value.split('/')[0] for x in (meta.args if meta is not None else [])
                         if isinstance(x, ast.Constant) and str(x.value).startswith('--')]
                v = sinks.lit(d) if d is not None else None
                v = None if v is sinks.UNKNOWN else v
                flag = flags[0] if flags else '--' + (a.arg if fn.name in fire else a.arg.replace('_', '-'))
                kind = kind_of(v) if v is not None else 'str'
                if kind == 'bool' and v:
                    continue                             # on by default: turned off by --no-x, not drafted
                opts.append(_arg(flag, a.arg, kind, v, switch=kind == 'bool'))
        if opts:
            out[fn.name] = opts
    return out


def helper_args(script: Path, root: Path) -> tuple[list[dict], list[str]]:
    """The arguments the job's parser gets outside the entry script, read from source: the functions the script hands
    its parser to (`parser = add_general_config(parser)`: DiffSynth, R2-04, some 60 arguments), the argument builders
    it imports (`args = parse_args()` from a module of the repository), the dataclasses it gives HfArgumentParser, and
    the helpers those hand the parser on to (three levels) - in the repository or in the installed library. Returns
    (arguments, the parser helpers whose source is not here)."""
    tree = _tree(script)
    if tree is None:
        return [], []
    out, missing, seen = [], [], set()
    own = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}

    def walk(node, path, mtree, depth, handed, dash=False):
        key = (str(path), node.name, node.lineno)
        if key in seen or len(seen) > 40 or path.as_posix().endswith(MEGATRON_ARGS):
            return                                       # Megatron's argument module is read with its filter (step 4)
        seen.add(key)
        out.extend(dataclass_args(node, dash) if isinstance(node, ast.ClassDef) and any(
            'dataclass' in ast.unparse(d) for d in node.decorator_list) else node_args(node))
        names = _parser_names(node, params=handed)       # a builder found by its name is not handed a parser
        for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)) if depth < 3 else ():
            given = any(isinstance(a, ast.Name) and a.id in names for a in call.args)
            if given or ARG_FUNC.match(getattr(call.func, 'id', None) or getattr(call.func, 'attr', None) or ''):
                hit, lost = resolve_call(call.func, path, mtree, root)   # parse_args() -> build_parser() as well
                if hit:
                    walk(*hit, depth + 1, given)
                elif lost and given and lost not in missing:
                    missing.append(lost)

    parsers, imp = _parser_names(tree), _imports(tree)
    for fn in (n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
        if any((sinks.qualname(d.func if isinstance(d, ast.Call) else d, imp) or '') in DATACLASS_WRAP for d in fn.decorator_list):
            ann = fn.args.args[0].annotation if fn.args.args else None        # @pyrallis.wrap() def main(cfg: Config)
            hit, _ = resolve_call(ann, script, tree, root) if isinstance(ann, ast.Name) else (None, None)
            if hit and isinstance(hit[0], ast.ClassDef):
                walk(*hit, 1, False)
    for call in sorted((n for n in ast.walk(tree) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset)):
        fname = getattr(call.func, 'id', None) or getattr(call.func, 'attr', None) or ''
        qual = sinks.qualname(call.func, imp) or ''
        if qual in DATACLASS_CLI:                        # tyro.cli(Config), pyrallis.parse(config_class=Config), ...
            c = call.args[0] if call.args else next((k.value for k in call.keywords if k.arg == 'config_class'), None)
            hit, _ = resolve_call(c, script, tree, root) if isinstance(c, ast.Name) else (None, None)
            if hit and isinstance(hit[0], ast.ClassDef):
                walk(*hit, 1, False, DATACLASS_CLI[qual])
            continue
        if fname == 'HfArgumentParser':
            arg = call.args[0] if call.args else None
            for c in (arg.elts if isinstance(arg, (ast.Tuple, ast.List)) else [arg] if arg is not None else []):
                if isinstance(c, ast.Name) and c.id not in HF_ARG_CLASSES:
                    hit, _ = resolve_call(c, script, tree, root)
                    if hit and isinstance(hit[0], ast.ClassDef):
                        walk(*hit, 1, False)
            continue
        handed = any(isinstance(a, ast.Name) and a.id in parsers for a in call.args)
        if not (handed or ARG_FUNC.match(fname)) or (isinstance(call.func, ast.Name) and call.func.id in own):
            continue                                     # the script's own functions are read with the script
        hit, lost = resolve_call(call.func, script, tree, root)
        if hit:
            walk(*hit, 1, handed)
        elif lost and handed and lost not in missing:
            missing.append(lost)
    return out, missing


def wrapper_options(text: str, here: Path, root: Path) -> list[dict]:
    """Options a launch script parses itself (`--accel) ACCEL=$2; shift 2;;` in its option loop) that take one of a few
    values: the labels of a `case "$ACCEL" in` block, or the files a path built from the value names (`accel/$ACCEL.yaml`
    -> zero2, zero3, fsdp: D2-R205b's sharding lever, added by hand and never reached). Kept when the name or a value is a
    speed setting. [{flag, name, choices, default}]"""
    out = []
    loops = [(m.group(1), m.group(2) or m.group(3)) for m in re.finditer(    # --x|-x) VAR=$2 / --x) shift; VAR=$1
        r'(--[A-Za-z][\w-]*)(?:\|-{1,2}[\w-]+)*\)\s*(?:shift;?\s*([A-Za-z_]\w*)="?\$\{?1\}?"?|([A-Za-z_]\w*)="?\$\{?2\}?"?)', text)]
    loops += re.findall(r'(--[A-Za-z][\w-]*)=\*\)\s*([A-Za-z_]\w*)="?\$\{1#\*=\}"?', text)
    for flag, var in loops:
        ref = r'\$(?:\{' + var + r'(?::?[-=][^}]*)?\}|' + var + r'(?!\w))'          # $VAR, ${VAR}, ${VAR:-default}
        values, paths = set(), {}
        for block in re.findall(r'case\s+"?\$\{?' + var + r'\}?"?\s+in\b(.*?)\besac\b', text, re.S):
            for label in re.findall(r'(?:^|;;)\s*\(?([^()\n;]+)\)', block, re.M):
                values |= {v.strip().strip('"\'') for v in label.split('|') if re.fullmatch(r'["\']?[\w.+-]+["\']?', v.strip())}
        for pre, post in re.findall(r'([\w./-]*)' + ref + r'([\w.-]*)', text):
            if '/' not in pre and not post.startswith('.'):
                continue                                 # a value used in a path, not as a plain word
            head, _, stem = pre.rpartition('/')
            for base in {here.parent, root}:
                for f in (base / head).glob(f'{stem}*{post}') if (base / head).is_dir() else ():
                    v = f.name[len(stem):len(f.name) - len(post) if post else None]
                    if v and re.fullmatch(r'[\w.+-]+', v):
                        values.add(v)
                        paths.setdefault(v, f)
        m = re.search(r'(?m)^\s*(?:export\s+)?' + var + r'=["\']?(?:\$\{' + var + r':?-)?([\w.+-]+)', text) or \
            re.search(r'\$\{' + var + r':?[-=]([\w.+-]+)\}', text)
        default, name = m.group(1) if m else None, norm(flag)
        if paths:                                        # files to choose between: only those that change speed alone
            same = speed_siblings(paths, default)
            values, paths = {v for v in values if v not in paths or v in same}, same
        every = sorted(values | ({default} if default else set()))
        keep = every if role_of(flag) else None if NOT_STEP.search(name) else perf_values(
            {'name': name, 'choices': every, 'default': default})
        if len(values) >= 2 and keep:
            out.append({'flag': flag, 'name': name, 'choices': keep, 'default': default,
                        'paths': {v: p for v, p in paths.items() if v in keep}})
    return out


# what may differ between config files a speed knob chooses between: sharding, offload, communication; a whole engine
# section when the engine changes (zero2.yaml -> fsdp.yaml); not the optimizer, the learning rate or the batch
SPEED_KEYS = re.compile(r'(^|\.)(zero_optimization|zero_stage|zero3_\w+|stage3_\w+|offload_\w+|fsdp_\w+|distributed_type|'
                        r'overlap_comm|contiguous_gradients|\w*bucket_size|round_robin_gradients|sub_group_size)(\.|$)')
ENGINE_SECTIONS = ('deepspeed_config', 'fsdp_config', 'megatron_lm_config')


def flat(doc, prefix: str = '') -> dict:
    if not isinstance(doc, dict):
        return {prefix[:-1]: doc}
    return {k: v for key, val in doc.items() for k, v in flat(val, f'{prefix}{key}.').items()}


def speed_siblings(paths: dict, default) -> dict:
    """The config files (value -> path) that differ from the job's own (`default`) only in speed settings: a ZeRO
    stage, an offload device, FSDP's sharding - not a sibling that also changes the optimizer or the learning rate."""
    base = load_config(paths[default]) if default in paths else None
    if not isinstance(base, dict):
        return {}
    fb, out = flat(base), {}
    for v, p in paths.items():
        doc = load_config(p)
        if not isinstance(doc, dict):
            continue
        fd = flat(doc)
        engine = fd.get('distributed_type') != fb.get('distributed_type')
        if all(SPEED_KEYS.search(k) or (engine and k.split('.')[0] in ENGINE_SECTIONS)
               for k in set(fd) | set(fb) if fd.get(k) != fb.get(k)):
            out[v] = p
    return out


def config_states(paths: dict) -> dict:
    """{value: {mechanism: state}} of the config files a launch option picks between (accel/zero2.yaml: DeepSpeed ZeRO
    stage 2; accel/fsdp.yaml: FSDP full shard), read from their content: the option is the registry's mechanism."""
    out = {}
    for v, p in paths.items():
        doc = load_config(p)
        if not isinstance(doc, dict):
            continue
        ds = doc['deepspeed_config'] if isinstance(doc.get('deepspeed_config'), dict) else doc
        zo = ds.get('zero_optimization') if isinstance(ds.get('zero_optimization'), dict) else {}
        stage = ds.get('zero_stage', zo.get('stage'))
        fs = doc.get('fsdp_config') if isinstance(doc.get('fsdp_config'), dict) else {}
        shard = str(fs.get('fsdp_sharding_strategy', '')).split('.')[-1].upper()
        shard = {'1': 'FULL_SHARD', '2': 'SHARD_GRAD_OP', '4': 'HYBRID_SHARD', '3': 'NO_SHARD'}.get(shard, shard)
        got = {**({'zero_stage': str(stage)} if stage is not None else {}),
               **({'fsdp_sharding': dict(sinks.SHARDING).get(shard)} if dict(sinks.SHARDING).get(shard) else {})}
        if got:
            out[v] = got
    return out


def config_choices(flag: str, value: str, root: Path, here: Path) -> dict | None:
    """A config-file option over its siblings (`--deepspeed ds/zero2.json` next to ds/zero3.json): the files of the same
    kind whose content sets a registry mechanism (config_states), as a choice knob mapped to it, the job's own file the
    default; None when fewer than two or when the job's own file sets none."""
    p = next((b / value for b in (root, here.parent) if (b / value).is_file()), None)
    if p is None:
        return None
    head = value.rpartition('/')[0]
    states = config_states(speed_siblings({(f'{head}/{f.name}' if head else f.name): f
                                           for f in sorted(p.parent.glob('*' + p.suffix))}, value))
    if len(states) < 2 or value not in states:
        return None
    per = {}
    for v, got in states.items():
        for m, s in got.items():
            per.setdefault(m, {})[v] = s
    vals = sorted(states)
    return {'type': 'str', 'flag': flag, 'choices': vals, 'default': value, **mapping(per, vals)}


def shell_switches(text: str) -> list[dict]:
    """Settings a launch script turns into a group of flags: an env var it tests (`[ "${BF16_MAIN_GRADS:-0}" = "1" ] &&
    EXTRA+=(--use-precision-aware-optimizer --main-grads-dtype bf16)`: M8's measured lever, out of init's sight), in an
    `if ...; then ... fi` block or a `${FP8:+--fp8-format hybrid}` expansion, or an option of its own that sets the
    variable (`--fp8) FP8=1;;`). [{var, option, on, off, default, flags: [(flag, value)]}]"""
    cond = (r'(?:\[\[?|\btest)\s*(?:-n\s+)?"?\$\{?(?P<var>[A-Za-z_]\w*)(?::?[-=](?P<d>[^}"\s]*))?\}?"?'
            r'(?:\s*(?:==?|-eq)\s*"?(?P<val>[\w.+-]+)"?)?\s*\]?\]?')
    found = [(m.group('var'), m.group('d'), m.group('val'), m.group('body').split(' #')[0])
             for m in re.finditer(cond + r'\s*&&\s*(?P<body>[^\n]*)', text)]
    found += [(m.group('var'), m.group('d'), m.group('val'), m.group('body'))
              for m in re.finditer(r'\bif\s+' + cond + r'\s*;?\s*then\b(?P<body>.*?)\b(?:elif|else|fi)\b', text, re.S)]
    found += [(m.group(1), None, None, m.group(2)) for m in re.finditer(r'\$\{([A-Za-z_]\w*):?\+([^}]*)\}', text)]
    options = {var: (flag, val) for flag, var, val in re.findall(        # --fp8) FP8=1; shift;;
        r'(--[A-Za-z][\w-]*)(?:\|-{1,2}[\w-]+)*\)\s*([A-Za-z_]\w*)=["\']?([\w.+-]+)["\']?\s*;', text)}
    out = {}
    for var, d, val, body in found:
        flags = re.findall(r'(--[A-Za-z][\w-]*)(?:[ =]+(?!-)([^\s)"\'\\;]+))?', body)
        set_here = re.search(r'(?m)^\s*(?:export\s+)?' + var + r'=(?!"?\$\{' + var + r'\b)', text)
        if not flags or var in out or (set_here and var not in options):
            continue                                     # set in the script itself: not a setting from outside
        on = val if val is not None else '1'
        if d is not None and d == on:                    # on unless set otherwise: the other value turns it off
            if on not in ('0', '1'):
                continue
            off = '1' if on == '0' else '0'
        else:
            off = d if d not in (None, on) else ('0' if on == '1' else '')
        opt = options.get(var)
        out[var] = {'var': var, 'option': opt[0] if opt and opt[1] == on else None, 'on': on, 'off': off,
                    'default': d if d is not None else None, 'flags': flags}
    return list(out.values())


def flag_states(flags: list, reg: dict) -> dict:
    """{mechanism: state} a group of flags sets through the registry's aliases (--main-grads-dtype bf16: bf16_main_grads
    bf16; --use-precision-aware-optimizer: precision_aware_optimizer on)."""
    out = {}
    for f, v in flags:
        if f not in reg:
            continue
        m, a, _ = reg[f]
        st = a['present'] if 'present' in a else a['map'].get(v) if isinstance(a.get('map'), dict) else (v or None)
        if st is not None:
            out[m] = str(st)
    return out


def mapping(per: dict, values: list) -> dict:
    """mechanism, states and implies of a knob from {mechanism: {value: state}}: the mechanism most values set is the
    knob's; what a value also sets on the others is implied (BF16_MAIN_GRADS=1: bf16 main grads, implying the
    precision-aware optimizer); the states of a mechanism a value requires are implied, not its own."""
    from ..experience.registry import mechanisms
    reg = mechanisms()
    per = {m: st for m, st in per.items() if m in reg and st}
    if not per:
        return {}
    needed = {m2 for m, st in per.items() for s in set(st.values())
              for m2 in ((reg[m].get('requires') or {}).get(s) or {})}
    main = max((m for m in per if m not in needed) or per, key=lambda m: len(per[m]))
    out = {'mechanism': main}
    if isinstance(reg[main].get('states'), list) or any(str(v) != s for v, s in per[main].items()):
        out['states'] = {v: per[main][v] for v in values if v in per[main]}
    implies = {}
    for m, st in per.items():
        for v, s in st.items():
            if m != main and v in values:
                implies.setdefault(v, {})[m] = s
    if implies:
        out['implies'] = implies
    return out


LAUNCHER_ENV = sinks.LAUNCHER_ENV                   # set by the launcher: the topology, not a setting


def leaf_path(keys: tuple, leaf_vals: dict) -> str | None:
    """The config key a chain names (cfg['train']['workers'], self.cfg.train.workers -> train.workers): the one leaf
    whose path and the chain's last keys agree."""
    hits = [p for p in leaf_vals if keys and (p.split('.') == list(keys[-len(p.split('.')):]) or
                                              list(keys) == p.split('.')[-len(keys):])]
    return hits[0] if len(hits) == 1 else None


def place(s: tuple, opts: dict, leaf_vals: dict, by: dict, dotlist: bool, toml: bool, root: Path):
    """(knob name, source) of a setting the sink trace found - the option its chain names (whatever parser defined
    it), the config key, the env var, a constant of the job's own files; None when nothing here sets it."""
    if s[0] == 'chain':
        a, p = opts.get(norm(s[1][-1])), leaf_path(s[1][1:], leaf_vals)
        if a is not None and (p is None or a['kind'] == 'bool' or a['default'] is not None):
            return by.get(('flag', a['flag']), a['name']), ('opt', a)
        if p is not None:
            key = ('flag', f'--{p}') if toml else ('override' if dotlist else 'path', p)
            return by.get(key, leaf_role(p)[0]), ('leaf', p)
        return None
    if s[0] == 'env' and s[1] not in LAUNCHER_ENV:
        return by.get(('env', s[1]), s[1]), s
    if s[0] == 'const' and Path(s[2]).resolve().is_relative_to(root.resolve()):
        return by.get(('const', s[1]), s[1].lower()), s
    return None


def values_for(kind: str, d, truthy: bool, role: str | None, lits: list, name: str) -> list | None:
    """The values to search of a setting whose definition lists none: 0 / 1 for an integer that only switches a call
    (D8's --fused-adam), the role's (workers 0, 2, 4, 8), else the values the code compares it with or maps."""
    if kind == 'int' and d in (0, 1) and truthy:
        return [0, 1]
    ch = choices_for(role, d, kind, name) if role else None
    if ch:
        return ch
    if kind == 'str':
        vals = sorted({x for x in lits if isinstance(x, str)} | ({d} if isinstance(d, str) else set()))
        return vals if len(vals) >= 2 else None
    return None


def new_knob(src: tuple, uses: list, *, leaf_vals: dict, set_by_shell: set, shell, rel, toml: bool, dotlist: bool):
    """The knob of a setting that controls a speed call, from its definition: an option's flag, type and choices, a
    config key, an env var (unset in the reference unless the launch script gives it a default), a constant."""
    truthy = all(u.mode in ('occurs', 'truth') for u in uses)
    role = next((u.spec.role for u in uses if u.spec.role != 'switch'), None)
    lits = [x for u in uses for x in u.fn.lits] + \
        [key for u in uses if u.mode == 'value' and isinstance(u.spec.smap, tuple) for key, _ in u.spec.smap]
    if src[0] == 'opt':
        a = src[1]
        if a['action'] == 'store_false' or a['name'] in set_by_shell:
            return None                                  # inverted, or set by the launch (step 1 drafts its values)
        k, d = {'type': a['kind'], 'flag': a['flag'], **({'style': a['style']} if a.get('style') else {})}, a['default']
        if a['kind'] == 'bool':
            return {**k, 'default': bool(d)}
        ch = list(a['choices']) if a['choices'] else values_for(a['kind'], d, truthy, role, lits, a['name'])
    elif src[0] == 'leaf':
        d = leaf_vals[src[1]]
        kind = kind_of(d)
        k = {'type': kind, 'flag': f'--{src[1]}'} if toml else {'type': kind, ('override' if dotlist else 'path'): src[1]}
        if kind == 'bool':
            return None if toml and d else {**k, 'default': False} if toml else k
        ch = values_for(kind, d, truthy, role, lits, leaf_role(src[1])[0])
        if ch and not toml:                              # the reference reads the config file (as step 5's keys do)
            return {**k, 'choices': merged(ch, d)} if len(ch) >= 2 else None
    elif src[0] == 'env':
        if shell and src[1] in shell['literal_env']:
            return None
        d = shell['env_defaults'].get(src[1]) if shell else None
        k = {'type': 'str', 'env': src[1]}
        ch = sorted({str(x) for x in [*lits, src[2]] if x is not None and not isinstance(x, bool)})   # with the code's default
        numeric = any(s in (int, float) for u in uses for s in u.fn.steps)   # int(os.getenv('LOADER_WORKERS', '4'))
        if truthy and set(ch) <= {'0', '1'}:
            ch = ['0', '1']                              # bool(int(os.environ.get('FUSED', '0')))
        elif numeric:                                    # numbers only: '' would stop int('')
            base = int(src[2]) if isinstance(src[2], str) and src[2].isdigit() else None
            ch = [str(x) for x in choices_for(role, base, 'int', src[1].lower()) or []] if role and base is not None else []
        ch = ch if len(ch) >= 2 else ['', ch[0]] if ch and not numeric else None
    else:                                                # a constant of the job's files
        d = src[3]
        kind = kind_of(d)
        k = {'type': kind, 'const': src[1], 'file': rel(Path(src[2]))}
        if kind == 'bool':
            return k
        ch = values_for(kind, d, truthy, role, lits, src[1].lower())
    if not ch or len(ch) < 2:
        return None
    return {**k, 'choices': merged(ch, d) if d is not None else list(ch), **({'default': d} if d is not None else {})}


def sink_knobs(tr, *, args, leaves, knobs, put, fw, set_by_shell, shell, dotlist, toml, rel, root, notes):
    """Knobs from what the code does (sinks.py): each setting that controls a speed call - a flag, a config key, an env
    var, a constant, whatever it is called (`--fast` gating torch.compile, D8's type=int --fused-adam) - drafted when no
    step drafted it, and mapped to the call's mechanism when no registry alias maps it, keeping the values that give a
    known state (S1's --bf16 searched x1.52 unmapped and x3.52 once mapped to precision by hand). A setting the trace
    reaches but cannot place is named in a note."""
    from ..experience.registry import label, mechanisms, state_owners
    opts = {}
    for a in args:
        opts.setdefault(norm(a['dest']) if a.get('dest') else a['name'], a)   # --use-fast-kernels dest=fast: args.fast
    leaf_vals = dict(leaves)
    by = {(t, k[t]): n for n, k in knobs.items() for t in ('flag', 'env', 'path', 'override', 'const') if k.get(t)}
    groups, unplaced, used = {}, {}, {}
    for u in tr.uses:
        got = place(u.source, opts, leaf_vals, by, dotlist, toml, root)
        if got is not None:
            groups.setdefault(got[0], (got[1], []))[1].append(u)
        elif u.source[0] == 'chain' and u.mechanism and any(sinks.SETTINGS_ROOT.match(c) for c in u.source[1][:-1]):
            unplaced.setdefault((u.source[1], u.mechanism), u)
    for s, ctx in tr.contexts.items():                   # every use of the setting, whichever chain reaches it
        got = place(s, opts, leaf_vals, by, dotlist, toml, root)
        if got is not None:
            used.setdefault(got[0], set()).update(ctx)
    mapped = []
    for name, (src, uses) in groups.items():
        k, made = knobs.get(name), name not in knobs
        if made:
            ctx = used.get(name, set())
            named = role_of(name) or (PERF_NAME.search(name) and not NOT_STEP.search(name)) or name in mechanisms() or \
                state_owners(name)
            if not named and not ctx <= {'speed', 'log', 'pass'}:
                continue                                 # it also decides what is trained (--train_text_encoder)
            k = new_knob(src, uses, leaf_vals=leaf_vals, set_by_shell=set_by_shell, shell=shell, rel=rel, toml=toml,
                         dotlist=dotlist)
            if k is None:
                continue
        values = k.get('choices') or ([False, True] if k['type'] == 'bool' else [])
        lab = label(name, k, fw)                         # a registry alias, or a name that knows every value, stays
        if not k.get('mechanism') and (lab is None or not lab.own and any(lab.state(v) is None for v in values)):
            per = {}
            for u in uses:
                for v in values if u.mechanism else ():
                    if (s := u.state(v)) is not None:
                        per.setdefault(u.mechanism, {}).setdefault(v, s)
            mp = mapping(per, values)
            if mp and made and k['type'] == 'str' and 'states' in mp:   # not the values that do something else (sgd)
                keep = [v for v in values if v in mp['states'] or v == k.get('default')]
                if len(keep) < 2:
                    continue
                k['choices'] = keep
            if mp:
                k.update(mp)
                u = next(u for u in uses if u.mechanism == mp['mechanism'])
                mapped.append(f'{name} -> {mp["mechanism"]} ({rel(Path(u.where[0]))}:{u.where[1]} {u.what})')
        if made:
            put(name, next((u.spec.role for u in uses if u.spec.role != 'switch'), None) or role_of(name) or 'switch', k)
    if mapped:
        notes.append('Mapped to the registry by what the code does with them (check them): ' + '; '.join(mapped[:8]) +
                     (f'; {len(mapped) - 8} more' if len(mapped) > 8 else '') + '.')
    for (ch, mech), u in list(unplaced.items())[:4]:
        notes.append(f'{rel(Path(u.where[0]))}:{u.where[1]}: {u.what} depends on `{".".join(ch)}`, which no flag, config '
                     f'key or env var of this draft sets: add it as a knob with mechanism: {mech}.')


def registry_flags(framework: str | None) -> dict:
    """flag -> (mechanism, alias, registry entry): the experience registry's command-line aliases of the framework."""
    from ..experience.registry import mechanisms
    out = {}
    for m, d in mechanisms().items():
        for a in ((d.get('aliases') or {}).get(framework) or []) if framework else []:
            if a.get('flag'):
                out.setdefault(a['flag'], (m, a, d))
    return out


def alias_values(alias: dict, entry: dict) -> list | None:
    """The values of a registry alias with finite values - a map's keys, or a value alias's states; None for a switch
    or an integer mechanism. Never a default: the reference run keeps the job's own (a registry default of bf16 would
    replace the "no" an Accelerate job inherits from its config)."""
    if isinstance(alias.get('map'), dict):
        return [str(k) for k in alias['map']]
    if 'present' not in alias and isinstance(entry.get('states'), list):
        return [str(s) for s in entry['states']]
    return None


def option_role(a: dict, reg: dict) -> str | None:
    """The role of an option the catalogue does not name: a framework flag the registry maps, an option whose finite
    values or name are a speed setting (`--accel zero2|zero3|fsdp`), or a switch named like one (`--channels_last`)."""
    if a['flag'] in reg:
        return 'registry'
    if NOT_STEP.search(a['name']):                       # --enable_npu_flash_attention, --prior_generation_precision
        return None
    if a['kind'] == 'bool' or (a['kind'] == 'int' and a.get('default') in (0, 1) and not a.get('choices')):
        return 'switch' if PERF_NAME.search(a['name']) else None
    return 'choice' if perf_values(a) else None


def perf_values(a: dict) -> list | None:
    """The choices of an option that is a speed setting: all of them when its name says so (--compile_mode); else the
    values that are speed settings and the job's own (--optim_name adamw|fused_adamw|sgd: not sgd, another algorithm).
    None below two."""
    vals, d = list(a.get('choices') or []), a.get('default')
    family = lambda v: PERF_VALUE.sub('', str(v)).strip('_-. ')       # noqa: E731 - fused_adamw: adamw; zero2: ''
    keep = vals if PERF_NAME.search(a['name']) else [v for v in vals if v == d or (     # not sgd -> fused(_adamw):
        PERF_VALUE.search(str(v)) and family(v) == ('' if d is None else family(d)))]   # another algorithm
    return keep if len(keep) >= 2 else None


def measured_flags(framework: str | None, fixed: dict) -> dict:
    """flag -> (mechanism, alias, entry): the framework's registry flags whose mechanism the shipped packs measured faster
    (M8 searched without --main-grads-dtype, +6.6 % on Megatron), and the partners their states require
    (--use-precision-aware-optimizer); not FP8 (it also needs --fp8-format), Megatron FSDP or CPU offload (another
    memory regime), nor MoE mechanisms for a job without experts."""
    from ..experience.store import measured
    flags, good = registry_flags(framework), measured(framework)
    moe = isinstance(fixed.get('num_experts'), int) and fixed['num_experts'] > 1
    out = {f: v for f, v in flags.items() if v[0] in good and v[0] != 'precision' and not v[0].startswith(
        ('fp8', 'megatron_fsdp', 'cpu_')) and (moe or not (v[0].startswith('moe_') or v[0] in ('ep', 'etp')))}
    need = {m for _, _, d in out.values() for req in (d.get('requires') or {}).values() for m in req}
    out.update({f: v for f, v in flags.items() if v[0] in need})
    return out


def mechanisms_default(m: str):
    from ..experience.registry import mechanisms
    return (mechanisms().get(m) or {}).get('default')


def registry_env(texts: list[str], libraries: set[str], framework: str | None, distributed: bool = True) -> dict:
    """env var -> choices: the registry's env aliases this job's code reads, that an alias of the job's framework
    names after a library the job imports (DIFFSYNTH_ATTENTION_IMPLEMENTATION for diffsynth, D2-R204; not
    DIFFUSERS_ATTN_BACKEND for DiT, a plain PyTorch job that takes only diffusers' VAE), or that set a framework-wide
    lever the shipped packs measured faster on this framework (NCCL ones for a job of several processes), with the
    values the registry maps. No default: unset, the library chooses (DiffSynth 2.1.12 the first importable of
    FlashAttention-4 > 3 > 2 > SageAttention > xformers > torch), and the reference run must be the job as it runs."""
    from ..experience.registry import mechanisms
    from ..experience.store import measured
    reads, libs, out, good = set(ENV_READ.findall('\n'.join(texts))), {m.lower() for m in libraries}, {}, measured(framework)
    for mech, d in mechanisms().items():
        for fw, aliases in (d.get('aliases') or {}).items():
            for a in aliases:
                env = a.get('env')
                lever = fw == 'any' and mech in good and (distributed or 'NCCL' not in (env or ''))
                if not env or env in out or (env not in reads and not lever and (
                        fw != framework or env.split('_')[0].lower() not in libs)):
                    continue
                choices = alias_values(a, d)
                if choices and len(choices) >= 2:
                    out[env] = choices
    return out


def requires_constraints(knobs: dict, fixed: dict, framework: str | None, names: list[str], notes: list[str]) -> list[str]:
    """The registry's `requires` for knobs drafted from it (bf16 main grads need the precision-aware optimizer): a
    constraint over the knob and its partner's knob; nothing when the launch turns the partner on; otherwise the state
    leaves the knob's choices (and the knob, when nothing else is left to try)."""
    from ..experience.registry import label, mechanisms
    out, labs = [], {n: label(n, k, framework) for n, k in knobs.items()}
    for name in names:
        lab, k = labs.get(name), knobs.get(name)
        if lab is None or k is None:
            continue
        for state, req in ((mechanisms().get(lab.mechanism) or {}).get('requires') or {}).items():
            mine = [v for v in (k.get('choices') or [False, True]) if lab.state(v) == str(state)]
            if not mine or name not in knobs:
                continue
            for m2, s2 in req.items():
                if all(lab.implies.get(str(v).lower() if isinstance(v, bool) else str(v), {}).get(m2) == str(s2) for v in mine):
                    continue                             # the value sets the partner too (BF16_MAIN_GRADS=1 adds both flags)
                partner = next((n for n, x in labs.items() if x is not None and x.mechanism == m2 and n in knobs), None)
                on = any(fixed.get(norm(f)) is True for f, (m, a, _) in registry_flags(framework).items()
                         if m == m2 and a.get('present') == s2)
                if partner is not None:
                    pk = knobs[partner]
                    theirs = [v for v in (pk.get('choices') or [False, True]) if labs[partner].state(v) == str(s2)]
                    left = ((f'not {name}' if mine == [True] else name) if k.get('type') == 'bool' else
                            f'{name} != {mine[0]!r}' if len(mine) == 1 else f'{name} not in {mine!r}')
                    right = partner if pk.get('type') == 'bool' and theirs == [True] else f'{partner} in {theirs!r}'
                    out.append(f'{left} or {right}')
                elif not on:
                    rest = [v for v in (k.get('choices') or []) if v not in mine]
                    if len(rest) >= 2:
                        k['choices'] = rest
                    else:
                        del knobs[name]
                    notes.append(f'{k.get("flag", name)} {mine[0]} needs {m2} {s2}, which this job cannot set: not drafted.')
                    break                                # that state is gone: its other requirements do not matter
    return out


def megatron_args(dirs: list[Path], extra=()) -> list[dict]:
    """The MEGATRON_KNOBS and `extra` flags this Megatron defines (from its argument module under one of dirs), else the
    switches."""
    for d in dirs:
        for rel in MEGATRON_ARGS:
            if (d / rel).is_file():
                return [a for a in argparse_args(d / rel) if a['flag'] in MEGATRON_KNOBS or a['flag'] in extra]
    return [{'flag': f, 'name': norm(f), 'kind': 'bool', 'default': False, 'choices': None, 'action': 'store_true'}
            for f in MEGATRON_SWITCHES]


def hf_defaults() -> dict:
    """Defaults of the HF_FIELDS in the installed transformers (the static table without it)."""
    if importlib.util.find_spec('transformers') is None:
        return dict(HF_FIELDS)
    try:
        from transformers import TrainingArguments
    except Exception:                                        # a broken install must not stop init
        return dict(HF_FIELDS)
    fields = getattr(TrainingArguments, '__dataclass_fields__', {})
    out = {}
    for name, fallback in HF_FIELDS.items():
        f = fields.get(name)
        d = fallback if f is None else (None if f.default is dataclasses.MISSING else f.default)
        out[name] = getattr(d, 'value', d)                   # enum members (OptimizerNames)
    return out


def shell_launch(script: Path, root: Path) -> dict:
    """What a launch script runs: the flags it passes ({flag: (text, resolved)}; text None for a bare switch),
    resolving variables from its own assignments and ${X:-default}; the Python script it starts; processes per node;
    the variables it assigns literally; whether it forwards extra arguments ("$@", or an array its option loop fills)
    and reads positional ones ($1, without `shift`). Flags on conditional lines (`[ .. ] && X+=(..)`, inside if/fi) are
    not counted as set; flags before the Python script on its line are the launcher's (`launcher_flags`: accelerate's
    --config_file is not the job's config), counted only for the processes per node."""
    text = re.sub(r'\\\r?\n', ' ', script.read_text(encoding='utf-8', errors='replace'))
    lines = [re.sub(r'(^|\s)#.*$', '', ln) for ln in text.splitlines()]
    env = {}
    for ln in lines:
        if re.match(r'^\s*(export\s+)?[A-Za-z_]\w*=', ln):
            try:
                toks = shlex.split(ln)
            except ValueError:
                continue
            for t in toks:
                m = re.fullmatch(r'([A-Za-z_]\w*)=(.*)', t)
                if m:
                    env.setdefault(m.group(1), m.group(2))

    def resolve(s: str | None, depth: int = 0) -> str | None:
        if s is None or depth > 6:
            return None

        def sub(m):
            name, default = m.group(1) or m.group(3) or m.group(4), m.group(2)
            v = env.get(name)
            if v is not None and v != m.group(0):
                r = resolve(v, depth + 1)
                if r is not None:
                    return r
            return default if default is not None else '\0'
        out = re.sub(r'\$\{(\w+):?-([^}]*)\}|\$\{(\w+)\}|\$(\w+)', sub, s)
        return None if '\0' in out or '$' in out or '`' in out else out

    flags, launcher, py, depth = {}, {}, None, 0
    for ln in lines:
        s = ln.strip()
        opens = re.match(r'(if|case|while|until|for)\b', s)
        closes = re.search(r'(^|[;\s])(fi|esac|done)\s*(;.*)?$', s)
        if opens or closes:                         # a block's lines are conditional (a one-line block too)
            depth = depth + 1 if opens and not closes else (max(0, depth - 1) if closes and not opens else depth)
            continue
        if depth or re.match(r'(\[\[?|test\b|elif\b|else\b)', s):       # `[ .. ] && X+=(..)` is conditional too
            continue
        try:
            toks = shlex.split(s)
        except ValueError:
            toks = s.split()
        at = next((i for i, t in enumerate(toks) if t.endswith('.py')), None)     # the script this line starts
        for i, t in enumerate(toks):
            m = re.fullmatch(r'(--[A-Za-z][\w.-]*)(?:=(.*))?', t)
            if m:
                val = m.group(2)
                nxt = toks[i + 1] if i + 1 < len(toks) else None
                if val is None and nxt is not None and not nxt.startswith('--') and not re.fullmatch(r'[|;&<>)]+', nxt) \
                        and not nxt.endswith('.py') and '$@' not in nxt and '[@]' not in nxt:
                    val = nxt
                (launcher if at is not None and i < at else flags)[m.group(1)] = \
                    (val, resolve(val)) if val is not None else (None, None)
            elif t.endswith('.py') and py is None:
                py = t
    py_path = None
    if py:
        r = resolve(py)
        for c in ([Path(r)] if r else []) + [root / py, root / Path(py).name]:
            if c.is_file():
                py_path = c.resolve()
                break
    both = {**flags, **launcher}
    nproc = next((typed(both[f][1]) for f in ('--nproc_per_node', '--nproc-per-node', '--num_gpus', '--num-gpus',
                                               '--num_processes', '--num-processes') if f in both), None)
    return {'flags': flags, 'launcher_flags': launcher, 'py': py_path, 'nproc': nproc if isinstance(nproc, int) else None,
            'text': text,
            'literal_env': {k for k, v in env.items() if '$' not in v},
            'env_defaults': {k: m.group(1) for k, v in env.items() if (m := re.fullmatch(r'\$\{' + k + r':?-([^}$]*)\}', v))},
            'forwards': bool(re.search(r'\$@|\$\{@\}|\$\*|\w\+=\(\s*"?\$\{?1\}?"?\s*\)', text)),
            'positional': bool(re.search(r'\$\{?[1-9]\b', text)) and not re.search(r'\bshift\b', text)}


def config_leaves(doc, prefix: str = '') -> list[tuple[str, object]]:
    out = []
    if isinstance(doc, dict):
        for k, v in doc.items():
            out += config_leaves(v, f'{prefix}{k}.') if isinstance(v, dict) else [(f'{prefix}{k}', v)]
    return out


def gpu_facts() -> dict:
    if not shutil.which('nvidia-smi'):
        return {'gpus': 1, 'memory_gib': None, 'gpu_type': None}
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=20)
        rows = [x.split(',') for x in r.stdout.strip().splitlines() if x.strip()]
        return {'gpus': len(rows) or 1, 'gpu_type': rows[0][0].strip() if rows else None,
                'memory_gib': round(float(rows[0][1]) / 1024 * 0.97, 1) if rows else None}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {'gpus': 1, 'memory_gib': None, 'gpu_type': None}


MODEL_KEYS = ('model_name_or_path', 'pretrained_model_name_or_path', 'model_path', 'model_dir', 'model_name', 'base_model',
              'model_id', 'hf_model', 'model')
SEQ_KEYS = ('max_seq_length', 'max_seq_len', 'seq_len', 'seq_length', 'sequence_length', 'block_size', 'max_length',
            'context_length', 'model_max_length', 'cutoff_len')
RES_KEYS = ('resolution', 'image_size', 'img_size', 'height')
HF_SHAPE = {'layers': ('num_hidden_layers', 'n_layer', 'num_layers'), 'hidden': ('hidden_size', 'n_embd', 'd_model'),
            'ffn': ('moe_intermediate_size', 'intermediate_size', 'n_inner', 'ffn_dim'),
            'heads': ('num_attention_heads', 'n_head'), 'kv_heads': ('num_key_value_heads',), 'head_dim': ('head_dim',),
            'vocab': ('vocab_size',), 'experts': ('num_experts', 'num_local_experts', 'n_routed_experts'),
            'topk': ('num_experts_per_tok', 'moe_topk')}
# decoders with a gated (three-matrix) MLP: the only ones the shape formula fits (GPT-NeoX / GPT-2: two matrices)
GATED_MLP = {'llama', 'mistral', 'mixtral', 'qwen2', 'qwen2_moe', 'qwen3', 'qwen3_moe', 'gemma', 'gemma2', 'gemma3_text',
             'phi3', 'deepseek_v2', 'deepseek_v3', 'internlm2', 'olmo2', 'granite', 'granitemoe', 'glm4', 'cohere'}
# diffusers transformers: (VAE downsampling, the pipeline's 2x2 packing - None: the model's own patch_size)
LATENT_TOKENS = {'DiTTransformer2DModel': (8, None), 'PixArtTransformer2DModel': (8, None),
                 'SD3Transformer2DModel': (8, None), 'FluxTransformer2DModel': (8, 2)}


def _model_dir(value: str, root: Path, revision: str | None = None) -> Path | None:
    """The local directory of the model a job loads: the path itself (absolute or under the workdir), or a hub id in
    the Hugging Face (at the job's revision: a branch / tag ref or a commit, else main) or ModelScope cache; None when
    it is not on this machine (a requested revision that is not cached is not replaced by another one)."""
    has = lambda d: (d / 'config.json').is_file() or (d / 'model_index.json').is_file()     # noqa: E731
    p = Path(value).expanduser()
    for d in ([p] if p.is_absolute() else [root / p, p]):
        if has(d):
            return d
    if re.fullmatch(r'[\w.-]+/[\w.-]+', value):
        hub = Path(os.environ.get('HF_HUB_CACHE')
                   or Path(os.environ.get('HF_HOME') or Path.home() / '.cache' / 'huggingface') / 'hub')
        repo = hub / f'models--{value.replace("/", "--")}'
        ref = repo / 'refs' / (revision or 'main')      # hashes have no order: the ref names the snapshot
        if revision:
            d = repo / 'snapshots' / (ref.read_text(encoding='utf-8').strip() if ref.is_file() else revision)
            return d if has(d) else None
        snaps = [repo / 'snapshots' / ref.read_text(encoding='utf-8').strip()] if ref.is_file() else []
        snaps += sorted((repo / 'snapshots').glob('*/'), key=lambda p: p.stat().st_mtime, reverse=True)
        ms = Path(os.environ.get('MODELSCOPE_CACHE') or Path.home() / '.cache' / 'modelscope' / 'hub')
        for d in snaps + [ms / value, ms / 'models' / value]:
            if has(d):
                return d
    return None


SHARD = re.compile(r'^(?P<base>.+?)(?P<variant>\.[a-z0-9_]+)?(?:-(?P<i>\d{5})-of-(?P<n>\d{5}))?(?P<variant2>\.[a-z0-9_]+)?'
                   r'\.safetensors$')


def _header_params(f: Path) -> int | None:
    try:
        with open(f, 'rb') as fh:
            n = int.from_bytes(fh.read(8), 'little')
            head = json.loads(fh.read(n)) if 0 < n < 1 << 28 else None
    except (OSError, ValueError):
        return None
    if not isinstance(head, dict):
        return None
    return sum(math.prod(v['shape']) for k, v in head.items()
               if k != '__metadata__' and isinstance(v, dict) and isinstance(v.get('shape'), list))


def safetensors_params(d: Path) -> int | None:
    """Exact parameter count of one checkpoint in a directory, from its safetensors headers (no weight is read): the
    shards an index names (all of them, or no count), else one variant (the plain one before .fp16 / .bf16 copies of the
    same network) with all of its numbered shards; None when that set is incomplete or unreadable."""
    idx = sorted(d.glob('*.safetensors.index*.json'), key=lambda p: (not p.name.endswith('.safetensors.index.json'), p.name))
    if idx:
        try:
            files = sorted(set((json.loads(idx[0].read_text(encoding='utf-8')).get('weight_map') or {}).values()))
        except (OSError, ValueError):
            return None
    else:
        groups: dict[tuple, list] = {}
        for f in sorted(d.glob('*.safetensors')):
            m = SHARD.match(f.name)
            if m:
                groups.setdefault((m['base'], m['variant'] or m['variant2'] or ''), []).append((f.name, m['n']))
        if not groups:
            return None
        key = min(groups, key=lambda k: (k[1] != '', k[0] not in ('model', 'diffusion_pytorch_model'),
                                         -sum((d / n).stat().st_size for n, _ in groups[k])))
        files = [n for n, _ in groups[key]]
        n = groups[key][0][1]
        if n is not None and len(files) != int(n):
            return None                                 # a numbered shard is missing
    counts = [_header_params(d / f) if (d / f).is_file() else None for f in files]
    return sum(counts) if counts and None not in counts else None


def model_facts(root: Path, fixed: dict, leaves=()) -> tuple[dict, int | None, str | None]:
    """The manifest's `model:` (params_b and, for a Transformer LM, its shape) and `workload.seq_len`, from the model
    directory the job loads (Hugging Face or diffusers, local or in the hub caches; named by a model argument or a
    `model.path`-like config key) and its sequence-length or resolution argument - what the experience write-back and
    the memory look-up need. ({}, seq, None) when the model is not on this machine; the third item says where the
    numbers came from."""
    seq = next((int(fixed[k]) for k in SEQ_KEYS if type(fixed.get(k)) is int and fixed[k] > 0), None)
    refs = [str(fixed[k]).strip() for k in MODEL_KEYS if isinstance(fixed.get(k), str) and str(fixed[k]).strip()]
    refs += [v.strip() for p, v in leaves if isinstance(v, str) and v.strip() and 'model' in p.lower().split('.')[0]
             and p.split('.')[-1].lower() in ('path', 'name', 'name_or_path', 'pretrained_model_name_or_path', 'id', 'dir')]
    rev = next((str(fixed[k]).strip() for k in ('revision', 'model_revision') if isinstance(fixed.get(k), str)
                and str(fixed[k]).strip()), None)
    d = next((x for x in (_model_dir(r, root, rev) for r in refs) if x is not None), None)
    if d is None:
        return {}, seq, None
    if (d / 'model_index.json').is_file():               # a diffusers pipeline: the network that trains
        d = next((d / s for s in ('transformer', 'unet', 'prior') if (d / s / 'config.json').is_file()), d)
    try:
        cfg = json.loads((d / 'config.json').read_text(encoding='utf-8')) if (d / 'config.json').is_file() else {}
    except (OSError, ValueError):
        cfg = {}
    lm = cfg.get('text_config') if isinstance(cfg.get('text_config'), dict) else cfg    # VLM wrappers: the LM's shape
    shape = {f: next((lm[k] for k in keys if type(lm.get(k)) is int), None) for f, keys in HF_SHAPE.items()}
    shape = {k: v for k, v in shape.items() if v is not None} if shape.get('layers') and shape.get('hidden') else {}
    if shape and cfg.get('tie_word_embeddings') is False:
        shape['untie'] = True
    # a pre-quantized checkpoint stores packed elements (4-bit: two per byte, GPTQ / AWQ: eight per int32) plus scales:
    # its header count is not the parameter count
    n = None if cfg.get('quantization_config') or lm.get('quantization_config') else safetensors_params(d)
    mtype = lm.get('model_type') or cfg.get('model_type')
    if n is None and shape.get('ffn') and shape.get('heads') and mtype in GATED_MLP:
        from ..experience.store import params_b      # its formula is a gated (SwiGLU / GeGLU) MLP decoder
        pb, how = params_b({'untie': False, **shape}), 'config.json shape (gated-MLP estimate)'
    else:
        pb, how = (round(n / 1e9, 3), 'safetensors headers') if n else (None, None)
    cls = cfg.get('_class_name')
    if seq is None and not shape and cls in LATENT_TOKENS and type(cfg.get('patch_size')) is int:
        res = next((int(fixed[k]) for k in RES_KEYS if type(fixed.get(k)) is int and fixed[k] > 0), None)
        if res:                                          # a diffusion transformer: latent tokens per image
            vae, pack = LATENT_TOKENS[cls]
            seq = (res // vae // (pack or cfg['patch_size'])) ** 2
    model = {**({'params_b': pb} if pb else {}), **shape, **({'family': 'moe'} if shape.get('experts') else {})}
    return model, seq, f'{d} ({how})' if pb else None


# the training bar of diffusers' example scripts: `Steps:  10%|#  | 10/100 [00:12<01:48, 1.21s/it, lr=1e-05, step_loss=0.08]`
# (text-to-image: step_loss; DreamBooth: loss)
TQDM_STEPS = {'step': r'Steps:\s+\d+%\|[^|]*\|\s*(?P<v>\d+)/\d+ \[', 'loss': r'\b(?:step_)?loss=' + VAL,
              'lr': r'\blr=' + VAL}         # VAL: a nan / inf loss is read too (the verifier rejects it)
# checkpoint arguments of a training script by role: save (the directory a run saves into), at (the update after which it
# saves, the toy's --save-at), every (every N updates, Megatron's --save-interval: layouts vary, a note only), load
CKPT_ARGS = ((r'^(save|save_dir|ckpt_dir|checkpoint_dir)$', 'save'), (r'^(save|ckpt|checkpoint)_at(_step)?$', 'at'),
             (r'^((save|ckpt|checkpoint)_(interval|steps|every|freq)|checkpointing_steps)$', 'every'),
             (r'^(load|load_dir|resume|resume_from|resume_from_checkpoint)$', 'load'))
EVAL_SCRIPTS = ('eval.py', 'evaluate.py')                       # and eval/*.py
EVAL_LABEL = r'(?:eval|val(?:idation)?)[_ /]?loss'
# a printed string that ends where runner.EVAL_RE expects the value: `eval_loss `, `val_loss=`
EVAL_KEY = re.compile(r'''(?i)(?<![\w])''' + EVAL_LABEL + r'''['"]?\s*[:=]?\s*$''')
PYTEST_TEST = re.compile(r'^\s*(?:async\s+)?def\s+test|^class\s+Test', re.M)


def ckpt_role(name: str) -> str | None:
    n = norm(name)
    return next((role for rx, role in CKPT_ARGS if re.match(rx, n)), None)


def prints_eval_loss(script: Path) -> bool:
    """True when a print() to the log writes an eval / validation loss label followed by its value, the way run_eval
    reads it: print(f'eval_loss {x}'), print('eval_loss', x), print({'eval_loss': x}), print(json.dumps({...})); not a
    label alone or a lookup (print(m['eval_loss']) prints only the number)."""
    try:
        tree = ast.parse(script.read_text(encoding='utf-8', errors='replace'))
    except SyntaxError:
        return False

    def label(n):
        return isinstance(n, ast.Constant) and isinstance(n.value, str) and bool(EVAL_KEY.search(n.value))

    def labelled(a, more: bool) -> bool:                      # one argument of print(); more: another one follows it
        if isinstance(a, ast.Call) and getattr(a.func, 'attr', getattr(a.func, 'id', None)) == 'dumps' and a.args:
            a = a.args[0]
        if isinstance(a, ast.Dict):
            return any(isinstance(k, ast.Constant) and isinstance(k.value, str) and bool(re.fullmatch(EVAL_LABEL, k.value, re.I))
                       for k in a.keys)
        if isinstance(a, ast.JoinedStr):                      # the label segment, then a formatted value
            return any(label(s) and isinstance(v, ast.FormattedValue) for s, v in zip(a.values, a.values[1:]))
        return label(a) and more

    return any(isinstance(c, ast.Call) and getattr(c.func, 'id', None) == 'print'
               and not any(k.arg == 'file' and ast.unparse(k.value) not in ('sys.stdout', 'sys.stderr') for k in c.keywords)
               and any(labelled(a, i + 1 < len(c.args)) for i, a in enumerate(c.args)) for c in ast.walk(tree))


def promotion_checks(root: Path, args: list[dict], launch: dict, names: tuple[str, str], code: dict, save_at: int,
                     hf: bool) -> tuple[dict | None, dict, list[str]]:
    """(checkpoint, verify.eval / verify.tests, notes): the commands of the resume, eval and test promotion items,
    drafted only when the repository shows them unambiguously (a wrong one fails every promotion): one save-directory,
    one save-at-update (an int) and one load argument of the training script (`args`), each taking one value; one
    eval.py / evaluate.py / eval/*.py that takes the same load argument, requires no other and prints an eval loss;
    check.py, and the pytest files of the job's own modules under tests/. Otherwise a note says what was seen and what
    to write. launch: the launch script's flags; names: the training script's and the launch script's (for the notes);
    hf: an HF Trainer script, whose log numbers no update (a resumed run would be counted from 1): a note only."""
    notes, verify = [], {}
    roles = [(a, ckpt_role(a['name'])) for a in args]
    pick = {r: [a['flag'] for a, rr in roles if rr == r and a['one_value'] and a['kind'] == ('int' if r == 'at' else 'str')]
            for r in ('save', 'at', 'load')}                  # {save_at} an int, {ckpt_dir} a path
    load = pick['load'][0] if len(pick['load']) == 1 and not hf else None
    ck = None
    if hf:
        notes.append('HF Trainer (TrainingArguments output_dir, save_steps, resume_from_checkpoint): no checkpoint was '
                     'drafted, because its log lines carry no update number and a resumed run would be counted from update '
                     '1 again (resume_continues fails); once each logged update carries its number and the script passes '
                     'training_args.resume_from_checkpoint to trainer.train(), add `checkpoint: {save: "--output_dir '
                     '{ckpt_dir} --save_strategy steps --save_steps {save_at}", resume: "--resume_from_checkpoint '
                     f'{{ckpt_dir}}/checkpoint-{{save_at}}", save_at: {save_at}}}`.')
    elif all(len(v) == 1 for v in pick.values()):
        ck = {'save': f'{pick["save"][0]} {{ckpt_dir}} {pick["at"][0]} {{save_at}}', 'resume': f'{load} {{ckpt_dir}}',
              'save_at': save_at}
    else:
        defined = {a['flag'] for a in args}
        seen = [a['flag'] for a, r in roles if r] + [f for f in launch if ckpt_role(f) and f not in defined]
        if seen:                                          # the block to write, with the value arguments seen filled in
            value = {a['flag'] for a, _ in roles if a['one_value']} | {f for f, (t, _) in launch.items() if t is not None}

            def first(role, missing):
                return next((f for f in seen if ckpt_role(f) == role and f in value), missing)
            notes.append(f'{names[0] if any(r for _, r in roles) else names[1]} has {", ".join(seen)} but not one '
                         'save-directory, one save-at-update (an integer) and one load argument that each take one value, '
                         'so no checkpoint was drafted: to check that a resumed run continues the loss (resume_continues), '
                         'add `checkpoint: '
                         f'{{save: "{first("save", "<save-dir argument>")} {{ckpt_dir}} '
                         f'{first("at", "<argument that saves after update N>")} {{save_at}}", resume: '
                         f'"{first("load", "<load argument>")} {{ckpt_dir}}", save_at: {save_at}}}` with the arguments '
                         'that save after update {save_at} into {ckpt_dir} and resume from it.')
    evals = [p for p in [root / n for n in EVAL_SCRIPTS] + sorted((root / 'eval').glob('*.py'))
             if p.is_file() and runs_something(p)]
    if evals:
        takes = load is not None and any(a['flag'] == load and a['one_value'] and a['kind'] == 'str'
                                         for a in argparse_args(evals[0]))
        other = [n for n in required_args(evals[0]) if n != load]
        why = next((w for bad, w in ((len(evals) > 1, 'several evaluation scripts'),
                                     (ck is None, 'no checkpoint was drafted (the eval runs on the saved checkpoints)'),
                                     (not takes, f'it does not take {load} (the training script\'s load argument)'),
                                     (other, f'it requires {", ".join(other)} too'),
                                     (not prints_eval_loss(evals[0]), 'it prints no eval_loss or val_loss value')) if bad),
                   None)
        name = evals[0].relative_to(root).as_posix()
        if why is None:
            mod = module_name(root, evals[0])                     # in a package: run with -m, as the launch command
            verify['eval'] = f'{{python}} {f"-m {mod}" if mod else name} {load} {{ckpt_dir}}'
        else:
            notes.append(f'{", ".join(p.relative_to(root).as_posix() for p in evals)}: {why}, so no verify.eval was '
                         'drafted; to compare the evaluation of the two checkpoints (eval_matches), set `verify.eval: '
                         f'"{{python}} {name} {load if takes else "<its checkpoint argument>"} {{ckpt_dir}}"`, a command '
                         'that loads {ckpt_dir} and prints `eval_loss <value>`.')
    tests = []
    check = root / 'check.py'
    need = required_args(check) if check.is_file() and runs_something(check) else None
    if need == []:
        tests.append('{python} check.py')
    elif need:
        notes.append(f'check.py: it requires {", ".join(need)}, so no test command was drafted from it; to run it on the '
                     'recommended change (user_tests), add "{python} check.py <its arguments>" to verify.tests.')
    rels = [p.relative_to(root).as_posix() for p in sorted((root / 'tests').rglob('*.py'))[:200]]
    files = [r for r in rels if re.fullmatch(r'[\w./-]*/(test_\w*|\w+_test)\.py', r)]       # (paths a shell keeps whole)
    own = {Path(f).stem for f in code['mutable'] + code['frozen'] + code['readable'] if f.endswith('.py')}
    mine = [r for r in files if imported_modules(root / r) & own]    # the job's own tests, not a framework's suite
    unit = [r for r in mine if PYTEST_TEST.search((root / r).read_text(encoding='utf-8', errors='replace'))]
    run_unit = f'{{python}} -m pytest {" ".join(unit)} -q'           # those files only: another one may not even import
    if unit and importlib.util.find_spec('pytest'):
        tests.append(run_unit)
    elif mine:
        cmd = run_unit if unit else f'{{python}} {mine[0]}'
        notes.append(f'{", ".join(mine[:3])}: '
                     + ('pytest tests, but pytest is not installed for this Python' if unit else 'no pytest test functions')
                     + ', so no verify.tests was drafted; to run the project\'s tests on the recommended change '
                     f'(user_tests), set `verify.tests: ["{cmd}"]` (they must pass on the current code).')
    if tests:
        verify['tests'] = tests
    drafted = [k for k, v in (('checkpoint', ck), ('verify.eval', verify.get('eval')), ('verify.tests', tests)) if v]
    if drafted:
        notes.insert(0, f'Promotion checks drafted from the repository ({", ".join(drafted)}): each promotion runs them, '
                        'and one that fails blocks the recommendation, so check that they work on the current code'
                        + ('; keep checkpoint.save_at below workload.steps.' if ck else '.'))
    return ck, verify, notes


def draft(root: str | Path, *, config: str | None = None, entry: str | None = None) -> tuple[dict, list[str]]:
    """(manifest dict, notes). The manifest is a draft: knobs and invariants are guesses the user should confirm."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise InitError(f'{root} is not a directory')
    notes = []
    rel = lambda p: p.relative_to(root).as_posix() if p.is_relative_to(root) else str(p)     # noqa: E731
    if entry:
        script = root / entry
        if not script.is_file():
            raise InitError(f'entry point {entry} not found in {root}')
    else:
        ranked = entry_candidates(root) or shell_candidates(root)
        if not ranked:
            raise InitError(f'no training entry point found in {root} (a train / pretrain / finetune / sft / main / run '
                            'Python script with top-level code, or a launch script): pass --entry')
        script = ranked[0]
        others = [p for p in ranked[1:] if p.stem.lower().startswith(ENTRY_NAMES)]
        if others:
            notes.append(f'Entry point: {rel(script)} (other candidates: {", ".join(rel(p) for p in others[:3])}); '
                         'pass --entry to use another.')
    if config and not (root / config).is_file():
        raise InitError(f'config file {config} not found in {root}')
    shell = shell_launch(script, root) if is_shell(script) else None
    py = shell['py'] if shell else script
    py_text = py.read_text(encoding='utf-8', errors='replace') if py else ''
    if py is not None and (err := syntax_error(py)):
        notes.append(f'{rel(py)} is not valid Python 3 ({err}): no arguments were read from it.')
    texts = [py_text] + [m.read_text(encoding='utf-8', errors='replace') for m in (local_modules(py) if py else [])]
    fw = next((f for rx, f in SHELL_FRAMEWORKS if re.search(rx, shell['text'])), None) if shell else None
    fw = fw or framework_of(texts) or framework(root, sorted(root.glob('*.py')) + sorted(root.glob('*/*.py'))[:40])
    distributed = fw in SELF_DISTRIBUTED or any(DIST.search(t) for t in texts) or bool(
        shell and re.search(r'torchrun|torch\.distributed\.(run|launch)|deepspeed|accelerate launch|mpirun|srun', shell['text']))
    args = argparse_args(py) if py else []
    more, unread = helper_args(py, root) if py else ([], [])     # the parser's arguments added outside the script
    cli = function_cli(_tree(py)) if py and _tree(py) is not None else {}   # click / typer / fire: a function's
    more += [a for opts in cli.values() for a in opts]                       # parameters are the options
    args += [a for a in more if a['flag'] not in {b['flag'] for b in args}]
    libs = set().union(*(imported_modules(f) for f in [py, *local_modules(py)])) if py else set()
    reg = registry_flags(fw)
    hf = bool(HF_ARGS.search(py_text) and HF_PARSER.search(py_text))
    lightning_cli = 'LightningCLI(' in py_text
    sflags = shell['flags'] if shell else {}
    set_by_shell = {norm(f) for f in sflags}
    fixed = {norm(f): (True if t is None else typed(v)) for f, (t, v) in sflags.items()}   # values the job runs with

    # the config file: given, else the one the entry point names or the best candidate
    named = [root / a['default'] for a in args if CONFIG_ARG.search(a['name']) and isinstance(a['default'], str)
             and CONFIG_SUFFIX.search(a['default'])]
    named += [root / v for t, v in sflags.values() if v and CONFIG_SUFFIX.search(v)]
    own = [root / v for t, v in (shell['launcher_flags'] if shell else {}).values() if v and CONFIG_SUFFIX.search(v)]
    cfg_path = (root / config) if config else find_config(root, named, exclude=own)    # not accelerate's own file
    doc = load_config(cfg_path)
    if cfg_path and doc is None:
        notes.append(f'{rel(cfg_path)} could not be read as yaml, json or toml: no knobs were taken from it.')
    toml = bool(cfg_path and cfg_path.suffix == '.toml' and doc is not None)
    if toml and fw != 'torchtitan':
        notes.append(f'{rel(cfg_path)} is toml: its keys can only be searched as --section.key flags (TorchTitan); '
                     'add knobs by hand.')
        doc = None
    dotlist = bool(DOTLIST.search(py_text))
    leaves = config_leaves(doc or {})
    leaf_names = {leaf_role(p)[0] for p, _ in leaves}
    for p, v in leaves:                     # current values: the launch's flags, the config, constants, argparse defaults
        fixed.setdefault(leaf_role(p)[0], v)
    for c in (module_constants(py) if py and not shell else []):
        fixed.setdefault(c['name'].lower(), c['value'])
    for a in args:
        if a['default'] is not None:
            fixed.setdefault(a['name'], a['default'])
    knobs, roles = {}, {}

    def put(name, role, k):
        values = [*(k.get('choices') or []), *([k['default']] if k.get('default') is not None else [])]
        if role == 'ga' and (k['type'] == 'bool' or not all(type(v) is int for v in values)):
            return              # a switch, a dtype or a fraction named like accumulation is not the step count
        knobs[name], roles[name] = k, role

    # 1. value flags the launch script passes (a switch it sets cannot be turned off by appended flags); the script's
    # own argparse choices when it defines them (D2-R205b: mixed_precision no/fp16/bf16, not bf16/fp32)
    defined = {a['flag']: a['choices'] for a in args if a['choices']}
    for flag, (t, v) in sflags.items():
        name, role, v = norm(flag), role_of(flag), typed(v)
        if t is None or isinstance(v, bool):
            continue
        if isinstance(v, str) and CONFIG_SUFFIX.search(v) and (k := config_choices(flag, v, root, script)):
            put(name, 'choice', k)                       # --deepspeed ds/zero2.json next to ds/zero3.json
            continue
        role = role or option_role({'flag': flag, 'name': name, 'kind': kind_of(v) if v is not None else 'str', 'default': v,
                                    'choices': defined.get(flag)}, reg)
        if role is None:
            continue
        kind = kind_of(v) if v is not None else ('int' if role in ('tp', 'pp', 'cp', 'ep', 'mbs', 'ga', 'bs', 'workers', 'zero') else None)
        ch = ((perf_values({'name': name, 'choices': defined[flag], 'default': v}) if role == 'choice' else list(defined[flag]))
              if flag in defined else choices_for(role, v, kind, name)) if kind else None
        if not ch and flag in reg:                       # the registry's values (Megatron's --attention-backend)
            ch = alias_values(*reg[flag][1:])
        if not ch and kind == 'int' and role == 'switch' and v in (0, 1):
            ch = [0, 1]                                   # --compile 1: a switch declared as an integer
        if ch:
            put(name, role, {'type': kind, 'flag': flag, 'choices': merged(ch, v), **({'default': v} if v is not None else {})})
    # 1b. options the launch script parses itself and dispatches on (D2-R205b: --accel zero2|zero3|fsdp)
    for w in wrapper_options(shell['text'], script, root) if shell else []:
        if w['name'] not in knobs:
            per = {}
            for v, got in config_states(w['paths']).items():        # accel/zero2.yaml: ZeRO stage 2
                for m, s in got.items():
                    per.setdefault(m, {})[v] = s
            put(w['name'], 'choice', {'type': 'str', 'flag': w['flag'], 'choices': w['choices'],
                                      **({'default': w['default']} if w['default'] is not None else {}),
                                      **mapping(per, w['choices'])})
    # 2. argparse definitions of the Python script and of the helpers that build its parser (helper_args)
    for a in args:
        role = role_of(a['name']) or option_role(a, reg)
        if role is None or a['name'] in knobs or a['name'] in set_by_shell:
            continue
        if a['kind'] != 'bool' and a['default'] is None and a['name'] in leaf_names:
            continue                                     # `--x` overrides the config's x when given: the config leaf is the knob
        k = {'type': a['kind'], 'flag': a['flag']}
        if a['kind'] != 'bool':
            ch = (perf_values(a) if role == 'choice' else a['choices']) or choices_for(role, a['default'], a['kind'], a['name'])
            if not ch and a['flag'] in reg:                # the registry's values; the default stays the job's own
                ch = alias_values(*reg[a['flag']][1:])
            if not ch and a['kind'] == 'int' and role == 'switch' and a['default'] in (0, 1):
                ch = [0, 1]                               # D8's --compile / --fused-adam / --tf32: type=int switches
            if not ch:
                continue
            k['choices'] = list(ch) if a['choices'] else merged(ch, a['default'])
            if a['default'] is not None:
                k['default'] = a['default']
        else:
            if a['action'] == 'store_false':
                continue                                  # inverted switches are rare and confusing to search: skip
            k['default'] = bool(a['default'])
            if a.get('style'):                            # HfArgumentParser: `--x true|false`
                k['style'] = a['style']
        put(a['name'], role, k)
    on = [a['flag'] for a in args if a['action'] == 'store_true' and a['name'] in set_by_shell and role_of(a['name'])]
    if on:                                               # U-4-2: say why a switch of the script is not a knob
        notes.append(f'{rel(script)} turns on {", ".join(on)}: a switch the launch sets cannot be turned off by the flags '
                     'the search appends, so it is not a knob (take it out of the command to let the search try both).')
    # 3. HF TrainingArguments of an HfArgumentParser script
    if hf:
        for name, d in hf_defaults().items():
            if name in knobs:
                continue
            if name == 'optim':
                v = fixed.get('optim', d)                  # the launch's optimizer (another one: not a knob)
                if v in HF_OPTIMS:
                    put(name, 'optim', {'type': 'str', 'flag': '--optim', 'choices': list(HF_OPTIMS), 'default': v})
            elif d is None or isinstance(d, bool):         # HfArgumentParser takes `--x true|false` (Optional[bool]: only that)
                # the value the launch passes (a bare --bf16 is true; an appended --bf16 false still turns it off)
                put(name, role_of(name), {'type': 'bool', 'flag': f'--{name}', 'style': 'value',
                                          'default': bool(fixed[name]) if name in fixed else bool(d)})
            else:
                put(name, role_of(name), {'type': 'int', 'flag': f'--{name}', 'choices': merged(choices_for(role_of(name), d, 'int'), d),
                                          'default': d})
    # 4. Megatron flags this Megatron defines
    if fw == 'megatron':
        mdirs = [root] + ([py.parent] if py else [])
        extra = measured_flags(fw, fixed)                    # flags of the levers the shipped packs measured faster
        margs = megatron_args(mdirs, extra)                  # (without the sources: the plain switches only)
        if not any((d / r).is_file() for d in mdirs for r in MEGATRON_ARGS):   # e.g. init on a login machine
            notes.append(f'Megatron\'s argument definitions ({" or ".join(MEGATRON_ARGS)}) were not found next to '
                         f'{rel(py) if py else "the workdir"}: knobs whose defaults come from Megatron (micro_batch_size, '
                         'recompute_granularity, context_parallel_size, num_workers, ...) are not drafted; run init where '
                         'the Megatron sources are, or add those knobs by hand.')
        for a in margs:
            name = a['name']
            if name in knobs or name in set_by_shell:
                continue
            fixed.setdefault(name, False if a['kind'] == 'bool' else a['default'])
            role = role_of(name)
            if a['kind'] == 'bool':
                put(name, role or 'switch', {'type': 'bool', 'flag': a['flag'], 'default': False})
                continue
            ch, d = a['choices'] or choices_for(role, a['default'], a['kind'], name), a['default']
            if not ch and a['flag'] in extra:                # --attention-backend's choices are an enum: the registry's
                ch = alias_values(*extra[a['flag']][1:])     # (no default: Megatron's own applies in the reference)
            if ch:
                put(name, role, {'type': a['kind'], 'flag': a['flag'], 'choices': list(ch) if a['choices'] else merged(ch, d),
                                 **({'default': d} if d is not None else {})})
    # 5. keys of the config file
    for path, v in leaves:
        name, role = leaf_role(path)
        if role is None or name in knobs:
            continue
        kind = kind_of(v)
        if toml:                                        # TorchTitan takes --section.key overrides of its toml
            if kind == 'bool':
                if not v:                               # a switch that is on cannot be turned off by a flag
                    put(name, role, {'type': 'bool', 'flag': f'--{path}', 'default': False})
                continue
            ch = choices_for(role, v, kind, name)
            if ch:
                put(name, role, {'type': kind, 'flag': f'--{path}', 'choices': merged(ch, v), 'default': v})
            continue
        k = {'type': kind, ('override' if dotlist else 'path'): path}
        if kind != 'bool':
            ch = choices_for(role, v, kind, name)
            if not ch:
                continue
            k['choices'] = sorted(set(ch) | ({v} if v is not None else set()))
        put(name, role, k)
    # 6. module constants of a script without arguments
    rel_py = rel(py) if py else None
    for c in (module_constants(py) if py and not args and not shell else []):
        role = role_of(c['name'].lower())
        if role is None or c['value'] is None or c['name'].lower() in knobs:
            continue
        kind = kind_of(c['value'])
        k = {'type': kind, 'const': c['name'], 'file': rel_py}
        if kind != 'bool':
            ch = choices_for(role, c['value'], kind, c['name'].lower())
            if not ch:
                continue
            if role in ('mbs', 'bs'):                          # per-device batches are usually large powers of two here
                ch = sorted({max(1, c['value'] // 2), c['value'], c['value'] * 2})
            k['choices'] = sorted(set(ch) | {c['value']})
        put(c['name'].lower(), role, k)
    # a batch size of unknown scope is the per-device micro-batch only when nothing else is
    bs = [n for n, r in roles.items() if r == 'bs']
    for n in (bs if 'mbs' in roles.values() else bs[1:]):
        del knobs[n], roles[n]
    if bs and bs[0] in roles:
        roles[bs[0]] = 'mbs'
        notes.append(f'{bs[0]} was taken as the per-device micro-batch: if it is the global batch, remove it from the '
                     'invariant.')
    # 7. environment variables (not ones the launch script sets itself)
    hw = gpu_facts()
    if hw['gpu_type'] is None:                              # a machine without GPUs (a laptop driving ssh hosts)
        notes.append('No GPU on this machine (nvidia-smi not found or not answering): hardware.gpus, memory_gib and gpu_type '
                     'are not the job\'s; set them for the hosts it runs on, and `execution` (ssh hosts, master_addr, or '
                     'srun) when it runs elsewhere (a job that runs here on the CPU can leave them).')   # (U-1-3)
    gpus = (shell['nproc'] if shell else None) or (hw['gpus'] if distributed else 1)
    if distributed and not (shell and shell['nproc']) and gpus > 1:   # U-4-1: every GPU of a shared machine, silently
        one = shell is not None and not re.search(r'torchrun|torch\.distributed\.(?:run|launch)|deepspeed|accelerate\s+launch'
                                                  r'|srun|mpirun', shell['text'])
        notes.append(f'Drafted as a {gpus}-GPU job (hardware.gpus: every GPU of this machine)'
                     + (f'; {rel(script)} starts a single process (no torchrun or deepspeed): set hardware.gpus: 1 unless '
                        'the job uses more.' if one else ': set hardware.gpus to the number of GPUs the job uses.'))
    env = {k: v for k, v in ENV_KNOBS.items() if k != 'NCCL_NVLS_ENABLE' or gpus > 1}
    env.update(MEGATRON_ENV if fw == 'megatron' else {})
    for name, ch in env.items():
        if shell and name in shell['literal_env']:
            notes.append(f'{rel(script)} sets {name} itself: not drafted as a knob (write it as ${{{name}:-...}} to make it one).')
            continue
        d = (shell['env_defaults'].get(name) if shell else None) or ch[0]
        knobs[name] = {'type': 'str', 'env': name, 'choices': merged(ch, d), 'default': d}
    # the registry's env vars the job's code or its libraries read (D2-R204: DIFFSYNTH_ATTENTION_IMPLEMENTATION)
    for name, ch in registry_env(texts, libs, fw, gpus > 1).items():
        if name in knobs or (shell and name in shell['literal_env']):
            continue
        d = shell['env_defaults'].get(name) if shell else None          # else unset in the reference run
        knobs[name] = {'type': 'str', 'env': name, 'choices': merged(ch, d), **({'default': d} if d is not None else {})}
    # the launch script's own switches for groups of flags (M8: BF16_MAIN_GRADS=1 adds the precision-aware optimizer and
    # bf16 main grads), mapped through the registry's aliases of those flags; not when every flag is a knob already
    have = {k.get('flag') for k in knobs.values()}
    for sw in shell_switches(shell['text']) if shell else []:
        perf = [f for f, _ in sw['flags'] if f in reg or role_of(f) or (PERF_NAME.search(norm(f)) and not NOT_STEP.search(norm(f)))]
        name = norm(sw['option']) if sw['option'] else sw['var']
        if not perf or all(f in have for f in perf) or name in knobs:
            continue
        on = flag_states(sw['flags'], reg)
        off = {m: str(mechanisms_default(m)) for m in on if mechanisms_default(m) is not None}
        if sw['option']:
            k = {'type': 'bool', 'flag': sw['option'], 'default': False, **mapping(
                {m: {True: s, **({False: off[m]} if m in off else {})} for m, s in on.items()}, [False, True])}
        else:
            vals = [sw['off'], sw['on']]
            k = {'type': 'str', 'env': sw['var'], 'choices': vals, **({'default': sw['default']} if sw['default'] is not None else {}),
                 **mapping({m: {sw['on']: s, **({sw['off']: off[m]} if m in off else {})} for m, s in on.items()}, vals)}
        knobs[name] = k
    # 8. what the code does with its settings (sinks.py): the settings that control torch.compile, DataLoader workers,
    # fused AdamW, TF32, autocast, DDP sync and buckets, checkpointing, ... whatever they are called and wherever they
    # are defined - drafted when missing, mapped to the call's mechanism
    trace = sinks.trace(py, root, set(cli)) if py else sinks.Trace()
    sink_knobs(trace, args=args, leaves=leaves, knobs=knobs, put=put, fw=fw, set_by_shell=set_by_shell, shell=shell,
               dotlist=dotlist, toml=toml, rel=rel, root=root, notes=notes)

    # invariants and constraints
    by_role = {}
    for n, r in roles.items():
        by_role.setdefault(r, n)
    constraints, invariants = [], {}
    par = [by_role[r] for r in ('tp', 'pp', 'cp') if r in by_role]
    if par:
        constraints.append(f'gpus % ({" * ".join(par)}) == 0')
    gbs_names = {'global_train_batch_size', 'global_batch_size', 'total_batch_size'} | ({'train_batch_size'} if fw == 'deepspeed' else set())
    gbs = next((n for n in [leaf_role(p)[0] for p, _ in leaves] + [c['name'].lower() for c in (module_constants(py) if py else [])]
                + [a['name'] for a in args] + list(set_by_shell) if n in gbs_names), None)
    if 'mbs' in by_role:
        dp = f'gpus / ({" * ".join(par)})' if par else 'gpus'
        ga = by_role.get('ga')
        if gbs and not ga:
            notes.append(f'The global batch is fixed by {gbs} (accumulation is derived): micro-batch changes keep the '
                         'workload; the verifier checks tokens per update at run time.')
        elif fw == 'megatron' and not ga:
            notes.append('Megatron derives gradient accumulation from --global-batch-size: keep that fixed (not a knob).')
        else:
            invariants['global_batch'] = f'{by_role["mbs"]} * {ga or 1} * {dp}'
            if not ga:                                   # U-2-1: how to add it, in the file's own terms
                notes.append('No gradient-accumulation knob found: changing the micro-batch would change the global batch, '
                             'so the invariant forbids it; add the accumulation knob to let the search trade them, e.g. '
                             '`accum: {type: int, path: <its key in the config>, choices: [1, 2, 4]}` (or `flag:`), and '
                             f'put its name in place of the 1 in global_batch ({by_role["mbs"]} * accum * {dp}).')
    if fw == 'megatron':
        constraints += megatron_constraints(knobs, fixed, notes, hw['gpu_type'])
    constraints += requires_constraints(knobs, fixed, fw, list(knobs), notes)   # whichever step drafted the knob
    for n in [n for n in roles if n not in knobs]:          # knobs the dependency rules took out
        del roles[n]
    # the reference run must give every knob an invariant or a constraint uses a value
    used = set().union(*(expr_names(e) for e in [*invariants.values(), *constraints])) if invariants or constraints else set()
    unknown = sorted(n for n in used if n in knobs and knobs[n].get('type') != 'bool' and ('flag' in knobs[n] or 'env' in knobs[n])
                     and knobs[n].get('default') is None)
    if unknown:
        notes.insert(0, f'Set the default of {", ".join(unknown)} to the value the job runs with now: the invariants or '
                        f'constraints use {"them" if len(unknown) > 1 else "it"}.')

    # the launch command
    rel_script = rel(script)
    if shell:
        parts = [f'bash {rel_script}']
        if not shell['forwards']:
            notes.insert(0, f'{rel_script} does not pass extra arguments on (no "$@" in its launch line): add "$@" so the '
                            'search can pass knob flags.')
        if shell['positional']:
            notes.append(f'{rel_script} reads positional arguments ($1 ...): add their values after the script name.')
        launch_flags = {**sflags, **shell['launcher_flags']}           # torchrun's --nnodes is the launcher's
        nnodes = next((typed(launch_flags[f][1]) for f in ('--nnodes', '--num_nodes', '--num-nodes') if f in launch_flags), None)
        if isinstance(nnodes, int) and nnodes > 1:
            notes.append(f'{rel_script} runs on {nnodes} nodes: set hardware.nodes and execution (ssh hosts or slurm).')
    else:
        mod = module_name(root, script)
        target = f'-m {mod}' if mod else rel_script
        if fw == 'deepspeed' and re.search(r'deepspeed\.initialize|local_rank', py_text):
            parts = [f'deepspeed --num_gpus {{gpus}} {rel_script}']
        elif fw == 'accelerate':
            parts = [f'accelerate launch --num_processes {{gpus}} {rel_script}']
        elif fw == 'lightning':
            parts = [f'{{python}} {target}' + (' fit' if lightning_cli else '')]
        elif distributed:
            parts = [f'torchrun --nproc_per_node {{gpus}} {target}']
        else:
            parts = [f'{{python}} {target}']
            notes.append(f'{rel_script} does not set up torch.distributed: drafted as a one-GPU job.')
    cfg_flag = None
    if cfg_path and doc is not None:
        target_rel = rel(cfg_path)
        cfg_flag = next((a['flag'] for a in args if isinstance(a['default'], str) and a['default'] == target_rel), None) \
            or next((f for f, (t, v) in sflags.items() if v == target_rel), None) \
            or next((a['flag'] for a in args if CONFIG_ARG.search(a['name'])), None)
    if toml:
        parts.append(f'--job.config_file {rel(cfg_path)}')                   # not rewritten: knobs are flags
    elif cfg_path and doc is not None and dotlist:
        parts.append('{config}')                                # positional config + key=value overrides
    elif cfg_path and doc is not None:
        parts.append(f'{cfg_flag or "--config"} {{config}}')
        if not cfg_flag and not lightning_cli:
            notes.append('Could not find the config-file argument of the entry point: check "--config {config}".')
    step_arg = next((f for f in sflags if norm(f) in {norm(s) for s in STEP_ARGS}), None) \
        or next((a['flag'] for a in args if a['name'] in {norm(s) for s in STEP_ARGS}), None)
    step_key = next((p for p, _ in leaves if p.split('.')[-1] in STEP_KEYS), None) if dotlist else None
    if step_key:
        unit = 'ba' if step_key.endswith('max_duration') else ''
        parts.append(f'{step_key}={{steps}}{unit}')
    elif step_arg:
        parts.append(f'{step_arg} {{steps}}')
    elif fw == 'megatron':
        parts.append('--train-iters {steps}')
    elif fw == 'torchtitan':
        parts.append('--training.steps {steps}')
    elif lightning_cli:
        parts.append('--trainer.max_steps {steps}')
    elif hf:
        parts.append('--max_steps {steps}')
    elif dotlist:
        parts.append('max_steps={steps}')
        notes.append('No step key found in the config: check that max_steps={steps} stops the job after {steps} updates.')
    else:
        parts.append('--max-steps {steps}')
        notes.append('No max-steps argument found: replace --max-steps with the argument that stops the job after {steps} '
                     'updates.')
    if fw == 'megatron':                                # every update and where its time goes, in the log
        for flag in ('--log-interval', '--timing-log-level'):
            if (sflags.get(flag) or (None, None))[1] != '1':
                parts.append(f'{flag} 1')
        notes.append('Megatron: --log-interval 1 --timing-log-level 1 log every update and its phase timers '
                     '(forward-backward, all-grads-sync, optimizer), which the search reads; the timers synchronize the '
                     'GPU at each phase boundary, a small cost on updates of a second or more: drop --timing-log-level '
                     'for very short updates.')
    if hf:
        parts.append('--logging_steps 1 --disable_tqdm true --report_to none')
        if not shell:                                       # a launch script's flags are the job's values already
            notes.append('HF Trainer: add the arguments your run passes (--output_dir, model and data) to launch.command, and '
                         'set each knob default to the value you pass today (the drafted defaults are TrainingArguments\'; '
                         '`infrarsi init --command "<your command line>"` takes them from it).')
    if fw == 'megatron' and not shell:
        notes.append('Megatron needs the model, data and tokenizer arguments of your run: add them to launch.command, or '
                     'pass your launch script as --entry.')
    if fw == 'megatron' and 'micro_batch_size' in knobs and knobs['micro_batch_size'].get('default') is None \
            and 'micro_batch_size' not in unknown:
        notes.insert(0, 'Megatron requires --micro-batch-size: set knobs.micro_batch_size.default to the value of your run.')
    parts.append('{args}')
    code = code_section(root, py)
    if shell:
        code['readable'].insert(0, rel_script)
    # the resume, eval and test promotion items
    ck, checks, more = promotion_checks(root, args, sflags, (rel(py) if py else rel_script, rel_script), code,
                                        DRAFT_STEPS // 2, hf)
    notes += more
    # the model's size and the sequence length: what a run needs to write its measurements back to the experience
    # store (Megatron's come from its argument dump at run time)
    model, seq, src = ({}, None, None) if fw == 'megatron' else model_facts(root, fixed, leaves)
    if fw != 'megatron':
        if model.get('params_b'):
            notes.append(f'model: {model["params_b"]} B parameters from {src}' + (f', sequence length {seq}' if seq else '')
                         + ': check them (the experience write-back and the memory look-up use them).')
        missing = [x for x, v in (('model.params_b', model.get('params_b')), ('workload.seq_len', seq)) if not v]
        if missing:                                      # U-1-4: each missing key with its meaning and its place
            what = {'model.params_b': '`model: {params_b: <the model size in billions of parameters>}`',
                    'workload.seq_len': '`seq_len: <tokens per sample>` under workload'}
            notes.append(f'Add {" and ".join(what[m] for m in missing)}: without '
                         f'{"them" if len(missing) > 1 else "it"} a run does not write its measurements back to the '
                         'experience store.')
    signals = {'source': 'log', **({'count_steps': True} if hf else {})}
    probe = [f for f in code['frozen'] if Path(f).stem.lower().startswith('infrarsi')
             and 'INFRARSI_PROBE' in (root / f).read_text(encoding='utf-8', errors='replace')]
    if probe:                                            # U-4-3: the probe is only read with signals.source: probe
        notes.append(f'{rel(py)} imports a copy of InfraRSI\'s probe ({probe[0]}): if the training loop calls it, set '
                     '`signals: {source: probe}` so the runs read its exact signals (data digest, samples, model size, '
                     'optimizer step) instead of the log.')
    if 'progress_bar.set_postfix' in py_text and re.search(r'''["'](?:step_)?loss["']''', py_text):   # diffusers' bar
        signals['patterns'] = dict(TQDM_STEPS)
        notes.append('The script reports each update on a tqdm bar (`Steps: n/N [... loss=..., lr=...]`): '
                     'signals.patterns read it; other bars in the log (data loading) are not updates.')
    manifest = {
        'schema': 'infrarsi-project-v1', 'name': root.name, 'framework': fw,
        'launch': {'command': ' '.join(parts), 'timeout_s': 3600},
        'config': {'file': rel(cfg_path)} if cfg_path and doc is not None and not toml else {'file': None},
        'knobs': knobs, 'constraints': constraints,
        'code': code,
        'signals': signals,
        'workload': {'steps': DRAFT_STEPS, 'window': [11, DRAFT_STEPS], 'invariants': invariants,
                     **({'seq_len': seq} if seq else {})},
        **({'model': model} if model else {}),
        **({'checkpoint': ck} if ck else {}),
        'hardware': {'gpus': gpus, 'memory_gib': hw['memory_gib'], 'gpu_type': hw['gpu_type']},
        'verify': {'tolerance': 0.03, **checks},
        'search': {'strategy': 'auto', 'budget': {'screens': 16}},
    }
    try:                                                    # the draft checks itself: what doctor would reject first
        from .manifest import ManifestError, Project
        proj = Project(manifest, root / 'infrarsi.yaml')
        for e in proj.constraint_errors(proj.reference_config()):
            notes.insert(0, f'The current values break a constraint ({e}): check the knob defaults and hardware.gpus.')
    except ManifestError as exc:
        if not unknown:
            notes.insert(0, f'The draft does not load yet: {exc}')
    shells = [p.relative_to(root).as_posix() for p in sorted(root.rglob('*.sh'))[:200]
              if re.search(r'torchrun|deepspeed |accelerate launch|pretrain_\w+\.py', p.read_text(encoding='utf-8', errors='replace'))]
    if shells and not args and not shell:
        notes.append(f'Launch scripts found ({", ".join(shells[:3])}): if the job is started by one of them, pass it as '
                     '--entry.')
    if not [k for k in knobs if k not in ENV_KNOBS and k not in MEGATRON_ENV]:
        notes.append('No known performance knob was found: add knobs by hand (flag, override, config path, const or env).')
    if unread:
        notes.append(f'{rel(py)} hands its argument parser to {", ".join(unread)}, defined outside the script in source that '
                     'is not here: the arguments they add were not read; run init where the job\'s libraries are installed, '
                     'or add the ones that change speed (data workers, checkpointing, offload, precision) as knobs by hand.')
    # the levers measured on jobs like this one that no knob of the draft reaches (F-D2-R205-4, M8): named, not silent
    from ..experience.store import unreached
    every = '\n'.join(texts + ([shell['text']] if shell else []))
    parsed = {a['flag'] for a in args} | set(sflags) if args and not unread and fw != 'megatron' else None
    current = {m: s for m, (s, *_) in trace.fixed.items()}          # what the code sets without a setting
    for r in unreached(fw, knobs, diffusion=bool(DIFFUSION_JOB.search(' '.join(sorted(libs)) + '\n' + every)),
                       moe=bool(re.search(r'(?i)\bmoe\b|num.experts', every)), current=current):
        flag = r['alias'].split()[0] if r['alias'] and r['alias'].startswith('--') else None
        at = trace.fixed.get(r['mechanism'])
        how = (f'{rel(Path(at[1]))}:{at[2]} sets it in code: a patch can make it a setting' if at else
               'no flag or env var of this framework sets it: a patch of the code can' if not r['alias'] else
               f'the job\'s parser has no {flag} (other {fw} scripts take {r["alias"]}): a patch of the code can'
               if flag and parsed is not None and flag not in parsed else f'add {r["alias"]}')
        maybe = f'; {", ".join(r["maybe"])} may be it (knobs.<name>.mechanism: {r["mechanism"]})' if r['maybe'] else ''
        notes.append(f'Not reached by any knob: {r["mechanism"]} {r["move"][0]} -> {r["move"][1]} measured x{r["speedup"]} on '
                     f'{r["model"]} ({r["gpu"]}, {r["evidence"]}); {how}{maybe}.')
    shown = [f for f in code['readable'] if f != rel_script]
    if code['mutable']:
        notes.append(f'Code patches: the search may edit {", ".join(code["mutable"])} (code.mutable)'
                     + (f'; {", ".join(code["frozen"])} frozen' if code['frozen'] else '')
                     + (f'; {", ".join(shown)} shown, not edited' if shown else '')
                     + '. Move files between code.mutable, code.frozen and code.readable to change what it may touch '
                       '(the loss, the data and the evaluation must not change); an empty code.mutable searches knobs only.')
    else:
        notes.append('No Python file of the job is inside the workdir, so code.mutable is empty and the search changes knobs '
                     'only: list the files it may patch under code.mutable.')
    notes.append('Log one line per update with the step, loss, grad norm, learning rate and tokens per update (or use the probe); '
                 'run `infrarsi doctor` to see which checklist items the signals cover.')
    return manifest, notes


def megatron_constraints(knobs: dict, fixed: dict, notes: list[str], gpu_type: str | None = None) -> list[str]:
    """Megatron's flag dependencies and divisibility rules over the drafted knobs; a switch whose requirement the
    launch fixes off is dropped."""
    out = []
    tp, pp, cp = 'tensor_model_parallel_size', 'pipeline_model_parallel_size', 'context_parallel_size'

    def ref(n):                                     # the knob, or the fixed value as a literal, or None (unknown)
        return n if n in knobs else (repr(fixed[n]) if fixed.get(n) is not None else None)

    def needs(name, *reqs, tp_above_one=False):
        if name not in knobs:
            return
        terms = []
        if tp_above_one:                            # sequence parallelism and its overlap act only with TP > 1
            if tp in knobs:
                terms.append(f'{tp} > 1')
            elif not (isinstance(fixed.get(tp), int) and fixed[tp] > 1):
                del knobs[name]
                return
        for r in reqs:
            if r in knobs:
                terms.append(r)
            elif fixed.get(r) is not True:
                del knobs[name]
                notes.append(f'--{name.replace("_", "-")} needs --{r.replace("_", "-")}, which the launch does not set: not drafted.')
                return
        if terms:
            out.append(f'not {name} or ({" and ".join(terms)})')
    needs('sequence_parallel', tp_above_one=True)
    needs('tp_comm_overlap', 'sequence_parallel', tp_above_one=True)
    needs('overlap_param_gather', 'overlap_grad_reduce', 'use_distributed_optimizer')
    for count, size in (('num_attention_heads', tp), ('num_query_groups', tp), ('num_layers', pp)):
        if size in knobs and isinstance(fixed.get(count), int):
            out.append(f'{fixed[count]} % {size} == 0')
    rg = knobs.get('recompute_granularity')
    cur = fixed.get('recompute_granularity', (rg or {}).get('default'))
    if rg and not (fixed.get('recompute_method') and fixed.get('recompute_num_layers') is not None):
        keep = [c for c in rg['choices'] if c != 'full' or c == cur]   # full needs --recompute-method and --recompute-num-layers
        if [c for c in keep if c != cur]:
            rg['choices'] = keep
        else:
            del knobs['recompute_granularity']
    conn = 'CUDA_DEVICE_MAX_CONNECTIONS'
    if conn in knobs and (fixed.get('use_custom_fsdp') is True or fixed.get('use_torch_fsdp2') is True):
        del knobs[conn]                                     # Megatron FSDP requires a value other than '1': nothing to search
    elif conn in knobs and not re.search(r'B[123]00|Blackwell', gpu_type or '', re.I):  # validate_args, before Blackwell
        if any(isinstance(fixed.get(n), int) and fixed[n] > 1 and n not in knobs for n in (tp, cp)):
            del knobs[conn]                                 # TP or CP > 1 in the launch: only '1' is allowed
        elif terms := [f'{n} == 1' for n in (tp, cp) if n in knobs]:
            out.append(f"{conn} == '1' or ({' and '.join(terms)})")
    gbs = fixed.get('global_batch_size')
    if 'micro_batch_size' in knobs and isinstance(gbs, int):
        factors = [r for r in (ref(tp), ref(pp), ref(cp)) if r not in (None, '1')]
        dp = f'(gpus * nodes) // ({" * ".join(factors)})' if factors else 'gpus * nodes'
        out.append(f'{gbs} % (micro_batch_size * ({dp})) == 0')
    return out


def write(root: str | Path, manifest: dict, notes: list[str], *, force: bool = False) -> Path:
    out = Path(root) / 'infrarsi.yaml'
    if out.exists() and not force:
        raise FileExistsError(f'{out} exists (use --force to overwrite)')
    head = '# Drafted by `infrarsi init`; review every knob, the invariants and the launch command.\n' + \
           ''.join(f'# NOTE: {n}\n' for n in notes)
    out.write_text(head + yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding='utf-8')
    return out
