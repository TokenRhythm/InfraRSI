"""ENV-A declarative knob space (design §2.4) and the finetune.yaml / finetune.sh templater.

Driver-side only (yaml + text; never imports torch). `KNOBS` is the exact space of design §2.4 (values and the
files they touch). `render(yaml_text, sh_text, knobs)` returns the edited (finetune.yaml, finetune.sh) texts; it only
ever touches the declared keys, so the fixed semantics (precision amp_bf16, max_seq_len 4096, global_train_batch_size
16, optimizer / scheduler / seed, --output_precision bf16) are preserved by construction. `validate(knobs)` is the
hooks' validate_config: it normalises, applies the pre-declared repair (device_train_microbatch_size='auto' wins over
activation_checkpointing) and lists errors/warnings. `sample(rng)` is the A-RND sampler (seed 42, conditional repair).
"""
from __future__ import annotations

import io
import re

import yaml

# name -> (allowed values, where it is applied). 'auto'/None/bools are kept as Python values.
KNOBS: dict[str, dict] = {
    'device_train_microbatch_size': {'values': [4, 8, 16, 'auto'], 'where': 'finetune.yaml'},
    'fsdp_config.sharding_strategy': {'values': ['FULL_SHARD', 'SHARD_GRAD_OP', 'NO_SHARD'], 'where': 'finetune.yaml'},
    'fsdp_config.activation_checkpointing': {'values': [True, False], 'where': 'finetune.yaml'},
    'fsdp_config.limit_all_gathers': {'values': [True, False], 'where': 'finetune.yaml'},
    'fsdp_config.forward_prefetch': {'values': [True, False], 'where': 'finetune.yaml'},
    'model.use_flash_attention_2': {'values': [False, True], 'where': 'finetune.yaml'},
    'train_loader.num_workers': {'values': [0, 1, 4, 8], 'where': 'finetune.yaml'},
    'tokenize.num_workers': {'values': [None, 1, 8, 32], 'where': 'finetune.sh'},
    'skip_valid_copy': {'values': [False, True], 'where': 'finetune.sh+finetune.yaml'},
}

# Semantics that must never change (reported in the task view; the knob space cannot reach them).
FIXED_SEMANTICS = {'precision': 'amp_bf16', 'max_seq_len': 4096, 'global_train_batch_size': 16,
                   'output_precision': 'bf16', 'optimizer': 'decoupled_adamw (lr 1e-5, betas .9/.95)',
                   'scheduler': 'cosine_with_warmup', 'seed': 17, 'global_seed': 17}
SPACE_SIZE = 6144


def knob_schema() -> dict:
    """A JSON-schema object for EnvTask.knob_schema (values are enums; None -> null)."""
    props = {}
    for name, spec in KNOBS.items():
        enum = [None if v is None else v for v in spec['values']]
        props[name] = {'enum': enum, 'applied_in': spec['where']}
    return {'type': 'object', 'properties': props, 'additionalProperties': False,
            'x-fixed-semantics': FIXED_SEMANTICS, 'x-space-size': SPACE_SIZE}


def _normalize_value(name: str, v):
    allowed = KNOBS[name]['values']
    # tolerate the string spellings an LLM / yaml round-trip may produce
    if isinstance(v, str):
        low = v.strip()
        if low.lower() == 'none':
            v = None
        elif low.lower() == 'auto':
            v = 'auto'
        elif low.lower() in ('true', 'false'):
            v = low.lower() == 'true'
        elif re.fullmatch(r'-?\d+', low):
            v = int(low)
        else:
            v = low
    return v


def validate(knobs: dict) -> tuple[dict, list[str], list[str]]:
    """Return (normalized_knobs, errors, warnings). Unknown keys and out-of-domain values are errors; the
    pre-declared repair (auto microbatch wins over activation checkpointing) is applied and warned."""
    errors, warnings, out = [], [], {}
    if not isinstance(knobs, dict):
        return {}, ['knobs must be an object'], []
    for name, v in knobs.items():
        if name not in KNOBS:
            errors.append(f'unknown knob {name!r}')
            continue
        nv = _normalize_value(name, v)
        if nv not in KNOBS[name]['values']:
            errors.append(f'{name}={v!r} not in {KNOBS[name]["values"]}')
        else:
            out[name] = nv
    if errors:
        return out, errors, warnings
    if out.get('device_train_microbatch_size') == 'auto' and out.get('fsdp_config.activation_checkpointing') is True:
        out['fsdp_config.activation_checkpointing'] = False      # pre-declared repair: auto wins (design §2.4)
        warnings.append('repair: device_train_microbatch_size=auto is mutually exclusive with '
                        'activation_checkpointing; forced activation_checkpointing=false')
    return out, errors, warnings


def _set(d: dict, dotted: str, value) -> None:
    keys = dotted.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def render(yaml_text: str, sh_text: str, knobs: dict) -> tuple[str, str, dict]:
    """Edit the base finetune.yaml / finetune.sh texts for `knobs` (already validated). Returns
    (yaml_text, sh_text, applied) where `applied` is the effective knob dict."""
    applied, errs, warns = validate(knobs)
    cfg = yaml.safe_load(yaml_text)
    skip_valid = bool(applied.get('skip_valid_copy'))

    for name, v in applied.items():
        if name == 'skip_valid_copy' or KNOBS[name]['where'] == 'finetune.sh':
            continue
        _set(cfg, name, v)
    if skip_valid:
        cfg.pop('eval_loader', None)                 # no valid split is produced; drop the eval dataloader
    new_yaml = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, allow_unicode=True)

    new_sh = sh_text
    tw = applied.get('tokenize.num_workers')
    if tw is not None and 'tokenize.num_workers' in applied:
        if '--num_workers' in new_sh:
            new_sh = re.sub(r'--num_workers[= ]\S+', f'--num_workers {tw}', new_sh)
        else:
            new_sh = new_sh.replace('--tokenizer ${BASE_MODEL}',
                                    f'--tokenizer ${{BASE_MODEL}} \\\n  --num_workers {tw}')
    if skip_valid:
        new_sh = re.sub(r'^\s*cp -r finetuning_tokens/train finetuning_tokens/valid\s*$', '',
                        new_sh, flags=re.MULTILINE)
        new_sh = re.sub(r'\n\s*eval_loader\.dataset\.split=valid \\', '', new_sh)
        new_sh = re.sub(r'\n\s*eval_interval=0 \\', '', new_sh)
    return new_yaml, new_sh, applied


def sample(rng) -> dict:
    """One A-RND point: an independent choice per knob, then the pre-declared conditional repair. `rng` is a
    random.Random (seed 42 upstream)."""
    raw = {name: rng.choice(spec['values']) for name, spec in KNOBS.items()}
    out, _errs, _warns = validate(raw)
    return out


def sample_order(seed: int = 42, n: int = 60):
    """A-RND's deterministic sequence: up to `n` distinct knob points from a seed-42 RNG (duplicates skipped). The
    orchestrator screens them in order under the budget."""
    import json as _json
    import random as _random
    rng = _random.Random(seed)
    seen, out = set(), []
    for _ in range(n * 40):
        if len(out) >= n:
            break
        point = sample(rng)
        key = _json.dumps(point, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(point)
    return out


def check_fixed_semantics(yaml_text: str, sh_text: str) -> list[str]:
    """Defence-in-depth: confirm a rendered pair still carries the fixed semantics (used by tests and the runner)."""
    problems = []
    cfg = yaml.safe_load(yaml_text)
    if cfg.get('precision') != 'amp_bf16':
        problems.append(f'precision changed to {cfg.get("precision")!r}')
    if int(cfg.get('max_seq_len', 0)) != 4096:
        problems.append(f'max_seq_len changed to {cfg.get("max_seq_len")!r}')
    if int(cfg.get('global_train_batch_size', 0)) != 16:
        problems.append(f'global_train_batch_size changed to {cfg.get("global_train_batch_size")!r}')
    def _int_or_interp(v, ref):     # the base yaml uses seed: ${global_seed}
        if isinstance(v, str) and v.strip() == '${global_seed}':
            return ref
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    if int(cfg.get('global_seed', -1)) != 17 or _int_or_interp(cfg.get('seed'), 17) != 17:
        problems.append('seed / global_seed changed')
    opt = cfg.get('optimizer') or {}
    if opt.get('name') != 'decoupled_adamw' or float(opt.get('lr', 0)) != 1.0e-5:
        problems.append('optimizer changed')
    if '--output_precision bf16' not in sh_text:
        problems.append('--output_precision bf16 missing from finetune.sh')
    return problems
