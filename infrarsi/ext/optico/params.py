"""OptiCo's dense parameter set: knob schema, encode/decode (parameters.txt, report.csv rows) and the pre-launch
validator `validate_optico_params` shared by every non-native arm (design §3.4, deviation D-OC-13).

Knobs keep OptiCo's own names and values (the parameter file run_monitor.sh exports into run_<model>.sh):
MICRO_BATCH_SIZE, TP, PP, OVERLAP_GRAD_REDUCE, OVERLAP_PARAM_GATHER, AC (none|sel|full|offload), AC_LAYERS,
FUSED_CROSS_ENTROPY, FIRST_PP_NUM_LAYERS, LAST_PP_NUM_LAYERS (None or an integer). GBS, seq, precision, LR, iterations
and DO/SP/TE/flash are script constants outside the space.

Validator rules, each tagged with its origin:
  [domain]    value sets of design §3.4;
  [layout]    TP*PP divides 8 (world 8), GBS 1024 % (MBS*DP) == 0;
  [script]    run_<model>.sh's own checks, computed exactly as bash does: AC=full needs (NUM_LAYERS / PP) % AC_LAYERS == 0
              (integer division); PP=1 needs FIRST/LAST none; PP=2 with a custom split needs FIRST+LAST == NUM_LAYERS;
              PP>2 needs NUM_LAYERS-FIRST-LAST > 0 and divisible by PP-2 (an unset side counts 0);
  [megatron]  Megatron core 0.15 start-up assertions that make the launch fail with certainty: heads, KV groups and
              FFN width divisible by TP; NUM_LAYERS % PP == 0 without a custom split; with one, every set side >= 1 and
              the remaining layers divisible by the remaining stages; uniform full recompute needs AC_LAYERS to divide
              every stage; CPU offloading needs PP=1 and AC_LAYERS < NUM_LAYERS; --overlap-param-gather needs
              --overlap-grad-reduce.
Normalizations (warnings, never errors): AC_LAYERS -> 1 when AC is none/sel (the script ignores it); overlap flags ->
false when AC=offload (the script drops them), so the parameter file, the report row and the dedup key agree.
"""
from __future__ import annotations

import re

from ...contracts import digest
from . import workloads as wl

KNOBS = ('MICRO_BATCH_SIZE', 'TP', 'PP', 'OVERLAP_GRAD_REDUCE', 'OVERLAP_PARAM_GATHER', 'AC', 'AC_LAYERS',
         'FUSED_CROSS_ENTROPY', 'FIRST_PP_NUM_LAYERS', 'LAST_PP_NUM_LAYERS')
MBS_CHOICES = (1, 2, 4, 8)
TP_CHOICES = (1, 2, 4, 8)
PP_CHOICES = (1, 2, 4, 8)
AC_CHOICES = ('none', 'sel', 'full', 'offload')
INT_KNOBS = ('MICRO_BATCH_SIZE', 'TP', 'PP', 'AC_LAYERS')
BOOL_KNOBS = ('OVERLAP_GRAD_REDUCE', 'OVERLAP_PARAM_GATHER', 'FUSED_CROSS_ENTROPY')
OPT_INT_KNOBS = ('FIRST_PP_NUM_LAYERS', 'LAST_PP_NUM_LAYERS')
NONE_WORDS = ('none', 'null', '')


def knob_schema(workload) -> dict:
    w = wl.get(workload)
    d = w.defaults

    def opt_int(side):
        return {'oneOf': [{'type': 'null'}, {'type': 'integer', 'minimum': 1, 'maximum': w.num_layers - 1}],
                'default': d[f'{side}_PP_NUM_LAYERS'],
                'description': f'layers of the {side.lower()} pipeline stage: None (even split) or an integer in '
                               f'[1, {w.num_layers - 1}] (PP > 1 only; script and Megatron split rules)'}
    return {'type': 'object', 'additionalProperties': False, 'required': list(KNOBS),
            'description': f'OptiCo dense parameter set for run_circle.sh {w.model_type} {w.model_size} '
                           f'(NUM_LAYERS {w.num_layers}, 8 GPUs, GBS {wl.GBS}, seq {wl.SEQ}, fp16, {wl.TRAIN_ITERS} iterations)',
            'properties': {
                'MICRO_BATCH_SIZE': {'type': 'integer', 'enum': list(MBS_CHOICES), 'default': d['MICRO_BATCH_SIZE']},
                'TP': {'type': 'integer', 'enum': list(TP_CHOICES), 'default': d['TP'],
                       'description': 'tensor parallel size (sequence parallel and --tp-comm-overlap follow TP > 1)'},
                'PP': {'type': 'integer', 'enum': list(PP_CHOICES), 'default': d['PP'], 'description': 'pipeline parallel size'},
                'OVERLAP_GRAD_REDUCE': {'type': 'boolean', 'default': d['OVERLAP_GRAD_REDUCE']},
                'OVERLAP_PARAM_GATHER': {'type': 'boolean', 'default': d['OVERLAP_PARAM_GATHER'],
                                         'description': 'needs OVERLAP_GRAD_REDUCE=true'},
                'AC': {'type': 'string', 'enum': list(AC_CHOICES), 'default': d['AC'],
                       'description': 'activation checkpointing: none, sel (selective), full (uniform, AC_LAYERS per '
                                      'group), offload (CPU offloading of AC_LAYERS layers, PP=1 only)'},
                'AC_LAYERS': {'type': 'integer', 'minimum': 1, 'maximum': w.num_layers, 'default': d['AC_LAYERS'],
                              'description': 'full: layers per recompute group (divides every stage); offload: layers '
                                             'offloaded; ignored for none/sel'},
                'FUSED_CROSS_ENTROPY': {'type': 'boolean', 'default': d['FUSED_CROSS_ENTROPY'],
                                        'description': '--cross-entropy-loss-fusion'},
                'FIRST_PP_NUM_LAYERS': opt_int('FIRST'),
                'LAST_PP_NUM_LAYERS': opt_int('LAST')}}


# ---------------------------------------------------------------- value coercion
def coerce(key: str, value):
    """OptiCo-style text or JSON value -> typed knob value; raises ValueError on a value of the wrong kind."""
    if key in BOOL_KNOBS:
        if type(value) is bool:
            return value
        s = str(value).strip().strip('"\'').lower()
        if s in ('true', 'false'):
            return s == 'true'
        raise ValueError(f'{key} must be true or false, got {value!r}')
    if key in INT_KNOBS or key in OPT_INT_KNOBS:
        if value is None or (isinstance(value, str) and value.strip().strip('"\'').lower() in NONE_WORDS):
            if key in OPT_INT_KNOBS:
                return None
            raise ValueError(f'{key} must be an integer, got {value!r}')
        if type(value) is bool:
            raise ValueError(f'{key} must be an integer, got {value!r}')
        if type(value) is int:
            return value
        if type(value) is float and value.is_integer():
            return int(value)
        s = str(value).strip().strip('"\'')
        if re.fullmatch(r'[+-]?\d+', s):
            return int(s)
        if re.fullmatch(r'[+-]?\d+\.0*', s):
            return int(float(s))
        raise ValueError(f'{key} must be an integer{" or None" if key in OPT_INT_KNOBS else ""}, got {value!r}')
    if key == 'AC':
        s = str(value).strip().strip('"\'').lower()
        if s in AC_CHOICES:
            return s
        raise ValueError(f'AC must be one of {", ".join(AC_CHOICES)}, got {value!r}')
    raise ValueError(f'unknown parameter {key!r}')


# ---------------------------------------------------------------- pipeline split
def stage_layers(n: int, pp: int, first: int | None, last: int | None) -> tuple[list[int] | None, list[str]]:
    """Layers per pipeline stage under the script's and Megatron's rules; (None, errors) when invalid."""
    errs = []
    if pp == 1:
        if first is not None or last is not None:
            errs.append('[script] When PP=1, FIRST_PP_NUM_LAYERS and LAST_PP_NUM_LAYERS should be none')
            return None, errs
        return [n], errs
    if first is None and last is None:
        if n % pp:
            return None, [f'[megatron] NUM_LAYERS {n} is not divisible by PP {pp} (uneven layouts need FIRST/LAST_PP_NUM_LAYERS)']
        return [n // pp] * pp, errs
    f, l = first or 0, last or 0
    if pp == 2:
        if f + l != n:
            errs.append(f'[script] When PP=2, FIRST_PP_NUM_LAYERS + LAST_PP_NUM_LAYERS must equal NUM_LAYERS ({f} + {l} != {n})')
    else:
        middle = n - f - l
        if middle <= 0:
            errs.append(f'[script] Middle layers({middle}) must be greater than 0')
        elif middle % (pp - 2):
            errs.append(f'[script] Middle layers({middle}) must be divisible by (PP-2)={pp - 2}')
    for name, v in (('FIRST_PP_NUM_LAYERS', first), ('LAST_PP_NUM_LAYERS', last)):
        if v is not None and v <= 0:
            errs.append(f'[megatron] {name} must be larger than 0')
    rest_layers = n - f - l
    rest_stages = pp - (first is not None) - (last is not None)
    if rest_stages > 0 and (rest_layers <= 0 or rest_layers % rest_stages):
        errs.append(f'[megatron] number of layers at middle stage {rest_layers} must be divisible by the middle pipeline '
                    f'size {rest_stages} and positive')
    elif rest_stages == 0 and rest_layers != 0:
        errs.append(f'[megatron] FIRST + LAST = {f + l} layers != NUM_LAYERS {n}')
    if errs:
        return None, errs
    per = rest_layers // rest_stages if rest_stages else 0
    stages = ([first] if first is not None else []) + [per] * rest_stages + ([last] if last is not None else [])
    return stages, errs


# ---------------------------------------------------------------- validator
def validate_optico_params(params: dict, workload) -> tuple[dict, list[str], list[str]]:
    """(normalized params, errors, warnings). Missing keys take the workload's script defaults; unknown keys are errors."""
    w = wl.get(workload)
    errors, warnings = [], []
    if not isinstance(params, dict):
        return dict(w.defaults), ['parameters must be an object of OptiCo knobs'], []
    out = dict(w.defaults)
    unknown = sorted(k for k in params if k not in KNOBS)
    if unknown:
        errors.append(f'[domain] unknown parameter(s) {unknown}; allowed: {", ".join(KNOBS)}')
    for k in KNOBS:
        if k in params:
            try:
                out[k] = coerce(k, params[k])
            except ValueError as exc:
                errors.append(f'[domain] {exc}')
    if errors:
        return out, errors, warnings
    mbs, tp, pp, ac, k = out['MICRO_BATCH_SIZE'], out['TP'], out['PP'], out['AC'], out['AC_LAYERS']
    for key, v, choices in (('MICRO_BATCH_SIZE', mbs, MBS_CHOICES), ('TP', tp, TP_CHOICES), ('PP', pp, PP_CHOICES)):
        if v not in choices:
            errors.append(f'[domain] {key} {v} not in {list(choices)}')
    if k < 1:
        errors.append(f'[domain] AC_LAYERS {k} must be >= 1')
    for key in OPT_INT_KNOBS:
        if out[key] is not None and not 1 <= out[key] < w.num_layers:
            errors.append(f'[domain] {key} {out[key]} must be None or an integer in [1, {w.num_layers - 1}]')
    if errors:
        return out, errors, warnings
    # normalizations of values the script ignores (keeps encode/decode/report rows consistent)
    if ac in ('none', 'sel') and k != 1:
        warnings.append(f'AC_LAYERS {k} has no effect with AC={ac}; normalized to 1')
        out['AC_LAYERS'] = k = 1
    if ac == 'offload' and (out['OVERLAP_GRAD_REDUCE'] or out['OVERLAP_PARAM_GATHER']):
        warnings.append('run_<model>.sh disables --overlap-grad-reduce/--overlap-param-gather when AC=offload; normalized to false')
        out['OVERLAP_GRAD_REDUCE'] = out['OVERLAP_PARAM_GATHER'] = False
    # layout
    if wl.GPUS % (tp * pp):
        errors.append(f'[layout] TP*PP = {tp * pp} does not divide the {wl.GPUS} GPUs')
    else:
        dp = wl.GPUS // (tp * pp)
        if wl.GBS % (mbs * dp):
            errors.append(f'[layout] GBS {wl.GBS} % (MBS {mbs} * DP {dp}) != 0')
    # model shape vs TP (Megatron TransformerConfig / ColumnParallelLinear)
    for name, size in (('num_attention_heads', w.heads), ('num_query_groups', w.kv_heads), ('ffn_hidden_size', w.ffn)):
        if size % tp:
            errors.append(f'[megatron] {name} {size} is not divisible by TP {tp}')
    # pipeline split (script + Megatron)
    stages, split_errs = stage_layers(w.num_layers, pp, out['FIRST_PP_NUM_LAYERS'], out['LAST_PP_NUM_LAYERS'])
    errors += split_errs
    # activation checkpointing
    if ac == 'full':
        if (w.num_layers // pp) % k:
            errors.append('[script] Error: the num layers per pp rank must be a multiple of the recompute layers. '
                          f'(({w.num_layers} / {pp}) % {k} != 0)')
        if stages and any(s % k for s in stages):
            errors.append(f'[megatron] AC_LAYERS {k} must divide the layers of every pipeline stage {stages} (uniform recompute)')
    elif ac == 'offload':
        if pp > 1:
            errors.append('[megatron] Currently there is no support for Pipeline parallelism with CPU offloading (AC=offload needs PP=1)')
        if k >= w.num_layers:
            errors.append(f'[megatron] CPU offloading can be done only for layers less than {w.num_layers} (AC_LAYERS {k})')
    if ac != 'offload' and out['OVERLAP_PARAM_GATHER'] and not out['OVERLAP_GRAD_REDUCE']:
        errors.append('[megatron] Must use --overlap-param-gather with --overlap-grad-reduce '
                      '(OVERLAP_PARAM_GATHER=true needs OVERLAP_GRAD_REDUCE=true)')
    return out, errors, warnings


def layout(params: dict) -> dict:
    tp, pp = int(params['TP']), int(params['PP'])
    dp = max(1, wl.GPUS // (tp * pp))
    return {'tp': tp, 'pp': pp, 'dp': dp, 'mbs': int(params['MICRO_BATCH_SIZE']),
            'microbatches': wl.GBS // max(1, dp * int(params['MICRO_BATCH_SIZE']))}


def is_default(params: dict, workload) -> bool:
    return params_key(params) == params_key(wl.get(workload).defaults)


# ---------------------------------------------------------------- encode / decode
def _fmt(key: str, v) -> str:
    if key in OPT_INT_KNOBS and v is None:
        return 'None'
    if type(v) is bool:
        return 'true' if v else 'false'
    return str(v)


def encode_params(params: dict) -> str:
    """parameters.txt in run_monitor.sh's own format ('KEY: value', bools lower-case, unset splits 'None')."""
    return ''.join(f'{k}: {_fmt(k, params[k])}\n' for k in KNOBS)


def decode_params(text: str) -> dict:
    """Parse a parameter file ('KEY: VALUE' or analyzer.py's 'KEY:VALUE'); values stay raw strings (see coerce)."""
    out = {}
    for line in (text or '').splitlines():
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        key = key.strip()
        if key:
            out[key] = value.strip()
    return out


def typed(raw: dict, *, strict: bool = False) -> dict:
    """Raw knob strings -> typed values; values that do not parse stay as given unless strict (then ValueError)."""
    out = {}
    for k, v in raw.items():
        if k not in KNOBS:
            continue
        try:
            out[k] = coerce(k, v)
        except ValueError:
            if strict:
                raise
            out[k] = v
    return out


def params_from_row(row: dict) -> dict:
    """The knob columns of an OptiCo report.csv row (typed where they parse)."""
    return typed({k: row.get(k) for k in KNOBS if k in row})


def params_key(params: dict) -> str:
    """Dedup key over the ten knobs (typed, canonical)."""
    return digest({k: params.get(k) for k in KNOBS})


def format_params(params: dict) -> str:
    """One-line summary in the R3 table style: MBS1 TP1 PP4 AC full/1 ..."""
    p = params
    ac = p['AC'] + (f'/{p["AC_LAYERS"]}' if p['AC'] in ('full', 'offload') else '')
    split = '' if p['FIRST_PP_NUM_LAYERS'] is None and p['LAST_PP_NUM_LAYERS'] is None else \
        f' split {p["FIRST_PP_NUM_LAYERS"]}/{p["LAST_PP_NUM_LAYERS"]}'
    flags = ''.join(f' {n}' for n, key in (('ogr', 'OVERLAP_GRAD_REDUCE'), ('opg', 'OVERLAP_PARAM_GATHER'),
                                           ('fce', 'FUSED_CROSS_ENTROPY')) if p.get(key))
    return f'MBS{p["MICRO_BATCH_SIZE"]} TP{p["TP"]} PP{p["PP"]} AC {ac}{split}{flags}'
