"""ENV-A model-artifact sanitizer (Codex W9 B-01). Runs on the node as an UNPRIVILEGED arm uid, never as root.

    python sanitize.py <src trained_model dir> <dst staging dir>

Reads the weights the candidate wrote -- in the order transformers 4.40 would pick them (model.safetensors >
model.safetensors.index.json > pytorch_model.bin > pytorch_model.bin.index.json), so the judge sees exactly the
tensors score.py would have loaded -- with `torch.load(weights_only=True)` / safetensors, requires a flat
{str: Tensor} mapping (any non-tensor object is rejected), and writes tensors-only `model.safetensors` plus
re-serialised `config.json` / `generation_config.json` (JSON only; custom-code / quantisation keys rejected). Root
then validates the staged files without deserialising anything and installs them read-only. A crash or exploit here
only reaches the unprivileged uid, whose processes are killed afterwards. Prints one JSON line; exit 0 = ok, 2 = rejected.
"""
from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

BANNED_CONFIG_KEYS = ('auto_map', 'trust_remote_code', 'custom_pipelines', 'quantization_config')
WEIGHT_ORDER = (('model.safetensors', 'st', False), ('model.safetensors.index.json', 'st', True),
                ('pytorch_model.bin', 'pt', False), ('pytorch_model.bin.index.json', 'pt', True))


class Rejected(Exception):
    pass


def _regular(path: Path) -> None:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        raise Rejected(f'missing {path.name}')
    if not stat.S_ISREG(st.st_mode):
        raise Rejected(f'{path.name} is not a regular file (symlink / dir / special)')


def _json_dict(path: Path) -> dict:
    _regular(path)
    try:
        obj = json.loads(path.read_text(encoding='utf-8'),
                         parse_constant=lambda c: (_ for _ in ()).throw(Rejected(f'non-finite JSON constant {c}')))
    except (ValueError, UnicodeDecodeError) as exc:
        raise Rejected(f'{path.name} is not valid JSON: {exc}')
    if not isinstance(obj, dict):
        raise Rejected(f'{path.name} is not a JSON object')
    return obj


def check_config(cfg: dict) -> dict:
    bad = [k for k in BANNED_CONFIG_KEYS if k in cfg]
    if bad:
        raise Rejected(f'config.json has custom-code / quantisation keys {bad}')
    return cfg


def select_weights(src: Path) -> tuple[str, list]:
    """(kind, [shard file names]) per transformers 4.40's from_pretrained preference; shard names must be plain
    basenames of regular files inside src."""
    for name, kind, indexed in WEIGHT_ORDER:
        p = src / name
        if not os.path.lexists(p):
            continue
        if not indexed:
            _regular(p)
            return kind, [name]
        index = _json_dict(p)
        wm = index.get('weight_map')
        if not isinstance(wm, dict) or not wm:
            raise Rejected(f'{name} has no weight_map')
        shards = sorted(set(wm.values()))
        for s in shards:
            if not isinstance(s, str) or not s or '/' in s or '\\' in s or s in ('.', '..'):
                raise Rejected(f'{name} lists a non-basename shard {s!r}')
            _regular(src / s)
        return kind, shards
    raise Rejected('no weights (model.safetensors / pytorch_model.bin) in trained_model')


def load_tensors(src: Path, kind: str, shards: list) -> dict:
    import torch
    out = {}
    for s in shards:
        path = str(src / s)
        if kind == 'st':
            from safetensors.torch import load_file
            part = load_file(path, device='cpu')
        else:
            try:
                part = torch.load(path, map_location='cpu', weights_only=True)
            except Exception as exc:                                  # noqa: BLE001 - any refusal is a rejection
                raise Rejected(f'{s}: refused by torch.load(weights_only=True): {type(exc).__name__}: {str(exc)[:200]}')
        if not isinstance(part, dict):
            raise Rejected(f'{s}: weights are a {type(part).__name__}, not a mapping')
        for k, v in part.items():
            if not isinstance(k, str):
                raise Rejected(f'{s}: non-string key {k!r}')
            if not isinstance(v, torch.Tensor):
                raise Rejected(f'{s}: {k!r} is a {type(v).__name__}, not a tensor')
            if k in out:
                raise Rejected(f'duplicate tensor {k!r} across shards')
            out[k] = v
    if not out:
        raise Rejected('no tensors')
    return out


def sanitize(src: Path, dst: Path) -> dict:
    if os.path.islink(src) or not src.is_dir():
        raise Rejected('trained_model is not a real directory')
    kind, shards = select_weights(src)
    cfg = check_config(_json_dict(src / 'config.json'))
    gen = _json_dict(src / 'generation_config.json') if os.path.lexists(src / 'generation_config.json') else None
    tensors = load_tensors(src, kind, shards)
    from safetensors.torch import save_file
    dst.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().clone().contiguous() for k, v in tensors.items()}, str(dst / 'model.safetensors'),
              metadata={'format': 'pt'})
    (dst / 'config.json').write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding='utf-8')
    if gen is not None:
        (dst / 'generation_config.json').write_text(json.dumps(gen, indent=2, sort_keys=True), encoding='utf-8')
    return {'ok': True, 'kind': kind, 'source_files': shards, 'n_tensors': len(tensors)}


if __name__ == '__main__':
    try:
        res = sanitize(Path(sys.argv[1]), Path(sys.argv[2]))
        code = 0
    except Rejected as exc:
        res, code = {'ok': False, 'reason': str(exc)[:500]}, 2
    except Exception as exc:                                          # noqa: BLE001 - unexpected = rejection
        res, code = {'ok': False, 'reason': f'{type(exc).__name__}: {str(exc)[:400]}'}, 2
    print('@@X2_SANITIZE@@' + json.dumps(res))
    sys.exit(code)
