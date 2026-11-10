"""Experience packs (infrarsi-exp/1): one JSON-lines file per contribution - a `pack` header, `env` lines (defined once,
referenced by id), `observation`, `effect`, `constraint`, `rule` (and InfraRSI's own `judgment`) lines. Speed-ups are
ratios t_base / t_new (> 1 faster), times in seconds per update, memory in GiB, noise in percent. `validate` checks a pack
without extra dependencies; `derive_effects` pairs the observations of one environment that differ in exactly one
mechanism, so a contributor never has to compute an effect."""
from __future__ import annotations

import gzip
import json
import math
import re
from pathlib import Path

from .registry import known, mechanisms

SCHEMA = 'infrarsi-exp/1'
KINDS = ('pack', 'env', 'observation', 'effect', 'constraint', 'rule', 'judgment')
FRAMEWORKS = ('megatron', 'fsdp', 'deepspeed', 'torchtitan', 'hf_trainer', 'accelerate', 'lightning', 'pytorch', 'other')
MECH_ID = re.compile(r'^(x-[a-z0-9-]+|[a-z][a-z0-9_]*)$')
PRIVATE = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b|(?:/home/|/Users/|C:\\Users\\)\w+|\b(?:sk|ghp|hf)_[A-Za-z0-9]{12,}')


def read_lines(path) -> list[dict]:
    p = Path(path)
    raw = gzip.open(p, 'rt', encoding='utf-8') if p.suffix == '.gz' else p.open(encoding='utf-8')
    with raw as f:
        return [migrate(json.loads(x)) for x in f if x.strip()]


def migrate(x):
    """A line of a harvest pack with an x- id that mechanisms.yaml registers since (x-fp8-attention: fp8_attention)
    under the registered id when its states are that mechanism's and the line does not name it too, so its
    measurements and constraints reach the projects that map it; other x- ids stay as they are."""
    if not isinstance(x, dict):
        return x
    for f in ('config', 'base', 'when', 'change', 'with'):
        v = x.get(f)
        if isinstance(v, dict) and any(str(k).startswith('x-') for k in v):
            x[f] = {_registered(k, s, v): s for k, s in v.items()}
    return x


def _registered(k, s, cfg: dict):
    m = str(k)[2:].replace('-', '_') if str(k).startswith('x-') else None
    reg = mechanisms()
    if m not in reg or m in cfg:
        return k
    st, vals = reg[m].get('states'), [str(v) for v in (s if isinstance(s, list) else [s])]   # a change: both states
    ok = all(v in [str(x) for x in st] for v in vals) if isinstance(st, list) else st != 'int' or all(v.isdigit() for v in vals)
    return m if ok else k


def write_lines(path, lines) -> None:
    with Path(path).open('w', encoding='utf-8', newline='\n') as f:
        for x in lines:
            f.write(json.dumps(x, ensure_ascii=False) + '\n')


def _need(line, *keys) -> list[str]:
    return [f'{line.get("id")}: missing {k}' for k in keys if line.get(k) in (None, '', {})]


def _num_errors(line, *keys) -> list[str]:
    """Measured values are positive finite numbers (a CSV import's "1.0" is a string)."""
    return [f'{line.get("id")}: {k} must be a positive number, not {line[k]!r}' for k in keys if line.get(k) is not None
            and not (type(line[k]) in (int, float) and math.isfinite(line[k]) and line[k] > 0)]


def _config_errors(line, field) -> list[str]:
    cfg = line.get(field) or {}
    if not isinstance(cfg, dict):
        return [f'{line.get("id")}: {field} must be a dict mechanism -> state']
    return [f'{line.get("id")}: {field} key {k!r} is not a mechanism id' for k in cfg if not MECH_ID.match(str(k))]


def validate(lines: list[dict]) -> tuple[list[str], list[str]]:
    """(errors, warnings) of a pack; errors make it unusable, warnings are shown to the contributor."""
    errors, warnings, ids, envs = [], [], set(), set()
    if not lines or lines[0].get('kind') != 'pack':
        errors.append('the first line must be the pack header (kind "pack")')
    for x in lines:
        k, i = x.get('kind'), x.get('id')
        if k not in KINDS:
            errors.append(f'{i}: unknown kind {k!r}')
            continue
        if not i or i in ids:
            errors.append(f'{i}: missing or duplicate id')
        ids.add(i)
        if k == 'env':
            envs.add(i)
    reg = mechanisms()
    for x in lines:
        k = x.get('kind')
        if k == 'pack':
            errors += _need(x, 'schema', 'pack_id', 'title', 'license', 'tier')
            if x.get('schema') != SCHEMA:
                errors.append(f'schema must be {SCHEMA!r}')
        elif k == 'env':
            errors += _need(x, 'gpu', 'gpus_per_node', 'nodes', 'framework')
            if x.get('framework') not in FRAMEWORKS:
                errors.append(f'{x.get("id")}: framework must be one of {FRAMEWORKS}')
            if not (x.get('model') or {}).get('params_b') or not (x.get('workload') or {}).get('seq_len'):
                errors.append(f'{x.get("id")}: model.params_b and workload.seq_len are required')
        elif k in ('observation', 'effect'):
            if x.get('env') not in envs:
                errors.append(f'{x.get("id")}: env {x.get("env")!r} is not defined in this pack')
            if k == 'observation':
                errors += _need(x, 'config', 'status') + _config_errors(x, 'config') + \
                    _num_errors(x, 'step_time_s', 'tokens_per_s_per_gpu', 'mem_peak_gib')
                if x.get('status') not in ('ok', 'oom', 'crash', 'invalid'):
                    errors.append(f'{x.get("id")}: status must be ok | oom | crash | invalid')
                if x.get('status') == 'ok' and not (x.get('step_time_s') or x.get('tokens_per_s_per_gpu')):
                    errors.append(f'{x.get("id")}: an ok observation needs step_time_s or tokens_per_s_per_gpu')
            else:
                errors += _need(x, 'change', 'speedup') + _config_errors(x, 'base') + _config_errors(x, 'change') + \
                    _num_errors(x, 'speedup')
                for m, ab in (x.get('change') or {}).items():
                    if not (isinstance(ab, list) and len(ab) == 2):
                        errors.append(f'{x.get("id")}: change[{m!r}] must be [from, to]')
                s = x.get('speedup')
                if isinstance(s, (int, float)) and not 0.2 <= s <= 5:
                    warnings.append(f'{x.get("id")}: speed-up {s} is outside 0.2-5 (a ratio t_base / t_new, not a percent?)')
        elif k == 'constraint':
            errors += _need(x, 'when', 'symptom') + _config_errors(x, 'when')
        elif k == 'rule':
            errors += _need(x, 'text')
        elif k == 'judgment' and not isinstance(x.get('assessment'), list):   # a run's transfer judgment (runs.py)
            errors.append(f'{x.get("id")}: a judgment needs its assessment (a list of judged moves)')
        for field in ('config', 'base', 'when', 'change'):
            for m in (x.get(field) or {}) if isinstance(x.get(field), dict) else ():
                if MECH_ID.match(str(m)) and not known(m):
                    warnings.append(f'{x.get("id")}: mechanism {m!r} is not registered (use x-{m} or add it to mechanisms.yaml)')
                elif m in reg and isinstance(reg[m].get('states'), list) and field != 'change':
                    st = (x.get(field) or {}).get(m)
                    if st is not None and str(st) not in reg[m]['states'] and not str(st).startswith('x-'):
                        warnings.append(f'{x.get("id")}: state {st!r} of {m} is not one of {reg[m]["states"]}')
        text = json.dumps(x, ensure_ascii=False)
        if PRIVATE.search(text):
            warnings.append(f'{x.get("id")}: looks like an IP address, a user path or a key - remove private data')
    return sorted(set(errors), key=errors.index), sorted(set(warnings), key=warnings.index)


def defaults() -> dict:
    return {m: str(d['default']) for m, d in mechanisms().items() if d.get('default') is not None}


def full_config(cfg: dict) -> dict:
    """A configuration with the registry defaults filled in for the mechanisms it does not mention."""
    return {**defaults(), **{m: str(v) for m, v in (cfg or {}).items()}}


def primary(fa: dict, fb: dict, diff: list) -> str | None:
    """The one mechanism a difference changes: the single differing mechanism, or the one whose new (or old) state
    requires every other difference (registry `requires`: bf16 main grads with the precision-aware optimizer)."""
    if len(diff) == 1:
        return diff[0]
    reg = mechanisms()
    for m in diff:
        for side in (fb, fa):
            req = ((reg.get(m) or {}).get('requires') or {}).get(side.get(m)) or {}
            if set(diff) - {m} <= set(req) and all(side.get(k) == str(req[k]) for k in set(diff) - {m}):
                return m
    return None


def derive_effects(lines: list[dict]) -> list[dict]:
    """Effects from pairs of ok observations of one environment whose configurations differ in one mechanism (or in
    one and what it requires); method independent_runs unless both say paired_runs; the base is the shared configuration."""
    obs = [x for x in lines if x.get('kind') == 'observation' and x.get('status') == 'ok']
    out, seen = [], set()
    for i, a in enumerate(obs):
        for b in obs[i + 1:]:
            if a.get('env') != b.get('env'):
                continue
            fa, fb = full_config(a['config']), full_config(b['config'])
            diff = sorted(m for m in set(fa) | set(fb) if fa.get(m) != fb.get(m))
            m = primary(fa, fb, diff)
            if m is None:
                continue
            if a.get('step_time_s') and b.get('step_time_s'):     # one unit for both: seconds per update, else
                ta, tb = a['step_time_s'], b['step_time_s']      # tokens per second per GPU
            elif a.get('tokens_per_s_per_gpu') and b.get('tokens_per_s_per_gpu'):
                ta, tb = 1 / a['tokens_per_s_per_gpu'], 1 / b['tokens_per_s_per_gpu']
            else:
                continue
            key = (a['env'], m, fa.get(m), fb.get(m), tuple(sorted((k, v) for k, v in fa.items() if k not in diff)))
            if key in seen:
                continue
            seen.add(key)
            out.append({'kind': 'effect', 'id': f'derived:{a["id"]}~{b["id"]}', 'env': a['env'],
                        'base': {k: v for k, v in (a['config'] or {}).items() if k not in diff}, 'change': {m: [fa.get(m), fb.get(m)]},
                        **({'with': {k: [fa.get(k), fb.get(k)] for k in diff if k != m}} if len(diff) > 1 else {}),
                        'speedup': ta / tb, 'n': 1, 'derived': True, 'evidence': [a['id'], b['id']],
                        'method': 'paired_runs' if a.get('method') == b.get('method') == 'paired_runs' else 'independent_runs'})
    return out


def gain(effect: dict) -> float | None:
    s = effect.get('speedup')
    return math.log(s) if isinstance(s, (int, float)) and s > 0 else None
