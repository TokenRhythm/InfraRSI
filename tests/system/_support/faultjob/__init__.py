"""The stand-in fault job as a project: train.py (modes), config.json, frozen.py, check.py, eval.py and a manifest whose
knobs cover every target kind (path, flag, override, const, env)."""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
FILES = ('train.py', 'config.json', 'frozen.py', 'check.py', 'eval.py')


def manifest_doc(*, modes=('ok', 'fast'), default_mode: str = 'ok', only_mode: bool = False, steps: int = 12,
                 window=(5, 12), save_at: int = 6, timeout_s: float = 4, screens: int = 8, tolerance: float = 0.03,
                 checkpoint: bool = True, tests: bool = True, eval_cmd: bool = True, knobs: dict | None = None) -> dict:
    """modes: the mode knob's choices; default_mode: the reference's mode (a faulty reference for `doctor`); only_mode:
    the mode knob alone (a search over the modes); knobs: a knob set of your own (no invariant then)."""
    mode_knob = {'type': 'str', 'choices': list(dict.fromkeys([default_mode, *modes])), 'flag': '--mode', 'default': default_mode}
    if knobs is None:
        knobs = {'mode': mode_knob} if only_mode else {
            'mode': mode_knob,
            'workers': {'type': 'int', 'choices': [1, 2], 'path': 'train.workers'},
            'bs': {'type': 'int', 'choices': [2, 4], 'path': 'train.bs'},
            'accum': {'type': 'int', 'choices': [1, 2], 'path': 'train.accum'},
            'SLOW': {'type': 'bool', 'env': 'FAULT_SLOW'},
            'opt': {'type': 'str', 'choices': ['plain', 'fused'], 'override': 'train.opt'},
            'STEP_SCALE': {'type': 'float', 'choices': [1.0, 0.8], 'const': 'STEP_SCALE', 'file': 'train.py'}}
    invariants = {'global_batch': 'bs * accum'} if {'bs', 'accum'} <= set(knobs) else {}
    doc = {
        'schema': 'infrarsi-project-v1', 'name': 'fault-job',
        'launch': {'command': '{python} train.py --config {config} --steps {steps} {args}', 'timeout_s': timeout_s},
        'config': {'file': 'config.json'},
        'knobs': knobs,
        'code': {'frozen': ['frozen.py'], 'readable': ['train.py']},
        'workload': {'steps': steps, 'window': list(window), 'invariants': invariants},
        'hardware': {'gpus': 1, 'memory_gib': 16},
        'verify': {'tolerance': tolerance, **({'tests': ['{python} check.py']} if tests else {}),
                   **({'eval': '{python} eval.py --load {ckpt_dir}'} if eval_cmd else {})},
        'search': {'budget': {'screens': screens, 'wall_hours': 0.2}}}
    if checkpoint:
        doc['checkpoint'] = {'save': '--save {ckpt_dir} --save-at {save_at}', 'resume': '--load {ckpt_dir}', 'save_at': save_at}
    return doc


def make_project(dst: Path, **kw) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copyfile(HERE / name, dst / name)
    (dst / 'infrarsi.yaml').write_text(yaml.safe_dump(manifest_doc(**kw), sort_keys=False), encoding='utf-8')
    return dst
