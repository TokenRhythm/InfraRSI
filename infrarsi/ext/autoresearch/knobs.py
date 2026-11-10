"""C-RND infra knob space of ENV-C (design §4.3, AMENDMENT_EXT_007) and its rendering into train.py.

  DEVICE_BATCH_SIZE    {32, 64, 128}                              gradient-accumulation split (default 128)
  compile_mode         {default, max-autotune-no-cudagraphs}      torch.compile(model, ...)
  compile_dynamic      {False, None}                              torch.compile(model, dynamic=...)
  expandable_segments  {True, False}                              PYTORCH_ALLOC_CONF
  optimizer_foreach    {default, True}                            AdamW groups as torch._foreach_* ops
3 x 2 x 2 x 2 x 2 = 48 points; the all-default point is the baseline. AMENDMENT_EXT_007 (feasibility rule, applied
once before any arm, C-RND's sampler only) removed `compile_mode = reduce-overhead` (CUDA-graph capture assertion in
2 / 2 pre-arm calibration attempts), `compile_mode = max-autotune` (the same assertion in its one pre-arm attempt,
envc_calib_b_mxat acce795c621e1) and `DEVICE_BATCH_SIZE = 256` (CUDA out of memory at launch). The frozen space, the
removed values with their evidence and the seed-42 draw order are knob_space_C.json (its sha256 is in every C-RND
receipt); the 128-point order of the first build is void. The mode and dynamic knobs apply to the model's
torch.compile call (the optimizer's compiled step functions keep their decorators). `optimizer_foreach=True` replaces
MuonAdamW._step_adamw by the multi-tensor form of adamw_step_fused (same update math, not bitwise). Rendering edits
exactly the anchored lines and fails loudly when an anchor is missing (a knob never silently does nothing).
"""
from __future__ import annotations

import ast
import itertools
import random

SPACE = {
    'DEVICE_BATCH_SIZE': [32, 64, 128],
    'compile_mode': ['default', 'max-autotune-no-cudagraphs'],
    'compile_dynamic': [False, None],
    'expandable_segments': [True, False],
    'optimizer_foreach': ['default', True],
}
DEFAULTS = {'DEVICE_BATCH_SIZE': 128, 'compile_mode': 'default', 'compile_dynamic': False, 'expandable_segments': True,
            'optimizer_foreach': 'default'}
TYPES = {'DEVICE_BATCH_SIZE': 'integer', 'compile_mode': 'string', 'compile_dynamic': ['boolean', 'null'],
         'expandable_segments': 'boolean', 'optimizer_foreach': ['string', 'boolean']}
KNOB_SCHEMA = {'type': 'object', 'additionalProperties': False,
               'properties': {k: {'type': TYPES[k], 'enum': list(v), 'default': DEFAULTS[k]} for k, v in SPACE.items()}}

DBS_LINE = 'DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)'
COMPILE_LINE = 'model = torch.compile(model, dynamic=False)'
ALLOC_LINE = 'os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"'
FOREACH_ADAMW = '''    def _step_adamw(self, group):
        # optimizer_foreach=True (InfraRSI C-RND knob): adamw_step_fused as torch._foreach_* ops over the group
        params = [p for p in group['params'] if p.grad is not None]
        if not params:
            return
        for p in params:
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
        grads = [p.grad for p in params]
        exp_avgs = [self.state[p]['exp_avg'] for p in params]
        exp_avg_sqs = [self.state[p]['exp_avg_sq'] for p in params]
        step = self.state[params[0]]['step']
        lr, (beta1, beta2), eps, wd = group['lr'], group['betas'], group['eps'], group['weight_decay']
        torch._foreach_mul_(params, 1 - lr * wd)
        torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
        torch._foreach_lerp_(exp_avg_sqs, torch._foreach_mul(grads, grads), 1 - beta2)
        bias1 = 1 - beta1 ** step
        bias2 = 1 - beta2 ** step
        denom = torch._foreach_add(torch._foreach_sqrt(torch._foreach_div(exp_avg_sqs, bias2)), eps)
        torch._foreach_addcdiv_(params, exp_avgs, denom, value=-lr / bias1)'''


REMOVED = [                               # AMENDMENT_EXT_007: values a pre-arm calibration showed cannot run here
    {'knob': 'compile_mode', 'value': 'reduce-overhead', 'error_class': 'cuda_graph',
     'evidence': ['envc_calib_a/p2 (train exit 1 after the capture assertion)', 'envc_calib_b/p2/a3d6753d59d7d (train exit 1 after 21 s)'],
     'mechanism': 'CUDA-graph capture conflicts with the loader buffer reuse whatever DBS, allocator or optimizer form'},
    {'knob': 'DEVICE_BATCH_SIZE', 'value': 256, 'error_class': 'oom',
     'evidence': ['envc_calib_b/p3/a5b5710f69616 (CUDA out of memory at 79 GiB after 15 s)'],
     'mechanism': 'about 2 x 45 GB of activations; no knob in the space reduces activation memory'},
    {'knob': 'compile_mode', 'value': 'max-autotune', 'error_class': 'cuda_graph',
     'evidence': ['envc_calib_b_mxat/mxat/acce795c621e1 (train exit 1)'],
     'mechanism': 'torch.compile mode max-autotune captures CUDA graphs like reduce-overhead'},
]
KNOB_SPACE_FILE = 'knob_space_C.json'


class KnobError(ValueError):
    pass


def points() -> list[dict]:
    """All points of the space in a fixed order."""
    keys = list(SPACE)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(SPACE[k] for k in keys))]


def sample_order(seed: int = 42) -> list[dict]:
    """The C-RND draw order: a seeded permutation of the space (sampling without replacement), baseline excluded."""
    rest = [p for p in points() if p != DEFAULTS]
    random.Random(seed).shuffle(rest)
    return rest


def frozen_space(seed: int = 42) -> dict:
    """The content of knob_space_C.json (AMENDMENT_EXT_007): values, removed values with their evidence, the size
    and the seed-42 draw order."""
    order = sample_order(seed)
    return {'schema': 'ext-autoresearch-knob-space-v1', 'amendment': 'AMENDMENT_EXT_007_FABLE51', 'space': SPACE,
            'defaults': DEFAULTS, 'removed': REMOVED, 'size': len(points()), 'size_without_baseline': len(order),
            'seed': seed, 'draw_order': order}


def _space_path():
    from pathlib import Path
    return Path(__file__).resolve().parent / KNOB_SPACE_FILE


def frozen_reference() -> dict:
    """{file, sha256, size} of the frozen knob_space_C.json; refuses a file that does not match this module."""
    import hashlib
    import json
    raw = _space_path().read_bytes()
    if json.loads(raw.decode('utf-8')) != json.loads(json.dumps(frozen_space())):
        raise KnobError(f'{KNOB_SPACE_FILE} does not match knobs.SPACE / the seed-42 order (re-freeze before any arm)')
    return {'file': f'ext/autoresearch/{KNOB_SPACE_FILE}', 'sha256': hashlib.sha256(raw).hexdigest(),
            'size': len(points()), 'size_without_baseline': len(points()) - 1}


def _replace_once(text: str, old: str, new: str, knob: str) -> str:
    if text.count(old) != 1:
        raise KnobError(f'{knob}: anchor {old!r} found {text.count(old)} times')
    return text.replace(old, new)


def _replace_method(text: str, cls: str, name: str, new_src: str) -> str:
    tree, lines = ast.parse(text), text.splitlines()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == name:
                    start = min([f.lineno] + [d.lineno for d in f.decorator_list])
                    out = lines[:start - 1] + new_src.splitlines() + lines[f.end_lineno:]
                    return '\n'.join(out) + ('\n' if text.endswith('\n') else '')
    raise KnobError(f'optimizer_foreach: {cls}.{name} not found')


def apply(text: str, knobs: dict) -> str:
    """train.py with the knob values rendered in; default values leave their lines untouched."""
    k = {**DEFAULTS, **(knobs or {})}
    unknown = set(k) - set(SPACE)
    if unknown:
        raise KnobError(f'unknown knobs {sorted(unknown)}')
    for name, v in k.items():
        if v not in SPACE[name] or type(v) is not type(next(x for x in SPACE[name] if x == v)):
            raise KnobError(f'{name}={v!r} not in {SPACE[name]}')
    out = text
    if k['DEVICE_BATCH_SIZE'] != DEFAULTS['DEVICE_BATCH_SIZE']:
        out = _replace_once(out, DBS_LINE, DBS_LINE.replace('= 128', f'= {k["DEVICE_BATCH_SIZE"]}'), 'DEVICE_BATCH_SIZE')
    if k['compile_mode'] != 'default' or k['compile_dynamic'] is not False:
        mode = '' if k['compile_mode'] == 'default' else f', mode={k["compile_mode"]!r}'
        out = _replace_once(out, COMPILE_LINE, f'model = torch.compile(model, dynamic={k["compile_dynamic"]!r}{mode})', 'compile')
    if k['expandable_segments'] is False:
        out = _replace_once(out, ALLOC_LINE, ALLOC_LINE.replace('expandable_segments:True', 'expandable_segments:False'),
                            'expandable_segments')
    if k['optimizer_foreach'] is True:
        out = _replace_method(out, 'MuonAdamW', '_step_adamw', FOREACH_ADAMW)
    ast.parse(out)
    return out
