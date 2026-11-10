"""Knob space of the flagship production path (the formal-54 trainer), as its frozen code honours it.

The trainer fixes the parallel layout: TP8 (asserted), CP1/EP32/ETP1, SP, alltoall + DeepEP, full/uniform/1 recompute,
DP4 x 2 micro-batches (train_flash_strategy_20260925.py:11,67,78-82,178; flash_strategy_signals_20260925.py:63-71).
What its registered recipes vary is the implementation stack (`family`) and, in the systems family, one policy row of
systems_policy.POLICIES. Env knobs are read at run time by the QSA kernels and by the trainer.

Frozen code: reports/rsi_infra_execution_20261018/code/uc_native_sync_ast_fix_v1/runtime (CODE_FREEZE d6a8f3b6...).
File:line citations below refer to that directory unless they name the trainer.

    canonical(knobs) -> (knob dict with defaults filled, errors, warnings)
    variant_name(knobs) -> registered recipe name, or a custom systems name 'SX_a<0|1>e<0|1>q<0|1>m<0|1>'
    systems_policy(knobs, name) -> the POLICIES row built field by field from the knobs (never looked up)
    job_env(knobs) -> job-env entries the knobs set beyond the base job
    space() -> every canonical knob dict that passes the static rules (E1 only unless lossy=True)
"""
from __future__ import annotations

import itertools

NATIVE, SCALED, HOST = 'native_fp32_v_offload', 'scaled_fp16_dual_cuda', 'scaled_fp16_v_offload'
PREFIX = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]          # systems_policy.py:4
EXPERT_PAIR = [0, 23]                                        # boundary_contract.py: candidate_selected_layers=[0,23]
FAMILIES = ('U_plus_C', 'B', 'W', 'systems')
SYSTEMS_KNOBS = ('retain_attention_prefix', 'retain_expert_layers', 'qkv_subgroup', 'shared_master_guard', 'moment_policy')
LAUNCH_KNOBS = ('qsa_fwd_launch', 'qsa_dq_launch', 'qsa_dkdv_launch')
LAUNCH_ENV = {'qsa_fwd_launch': 'FLASH_QSA_FWD_LAUNCH', 'qsa_dq_launch': 'FLASH_QSA_DQ_LAUNCH',
              'qsa_dkdv_launch': 'FLASH_QSA_DKDV_LAUNCH'}
LAUNCH_REGISTERED = {'qsa_fwd_launch': '4,1', 'qsa_dq_launch': '4,2', 'qsa_dkdv_launch': '4,2'}   # flash_qsa_tuned_launch_v1.py:26-28
LAUNCH_VENDORED = {'qsa_fwd_launch': '8,2', 'qsa_dq_launch': '8,1', 'qsa_dkdv_launch': '8,1'}     # flash_qsa_tuned_launch_v1.py:3-6
KNOBS = ('family',) + SYSTEMS_KNOBS + LAUNCH_KNOBS + ('qsa_dkdv_fastmask', 'aux_readback_mode')

_SYSTEMS_ROWS = ('S0', 'S_match', 'S_match_full', 'Q', 'Q_full', 'Qo', 'Qo_full')
_ALL = FAMILIES

# One entry per knob. applies_to: families whose code reads it (elsewhere it must be None). sources: value -> registered
# recipes that use it. lossy: values outside E1. new: values no registered recipe ran (GPU-unverified on this path).
SCHEMA = {
    'family': dict(
        values=list(FAMILIES), applies_to=_ALL, lossy=[], new=[],
        sources={'U_plus_C': ['U_plus_C'], 'B': ['B'], 'W': ['W'], 'systems': list(_SYSTEMS_ROWS)},
        honoured_by='family_transform.py:85-99 (one transform per family); uc_transform.py:56-81; '
                    'systems_transform.py:6-25; w_family_transform.py:4-17',
        description='implementation stack. U_plus_C = B with six rollbacks to native kernels (uc_contract.py:4-15); '
                    'B = production recipe; W = B + QSA fastmask v1, host-sync memo, combine segsum (w_family_contract.py); '
                    'systems = B + the systems policy row below'),
    'retain_attention_prefix': dict(
        values=[True, False], applies_to=('systems',), lossy=[], new=[],
        sources={True: ['S0', 'S_match', 'Q', 'Qo'], False: ['S_match_full', 'Q_full', 'Qo_full']},
        honoured_by='systems_transform.py:12-20 (block/0 recompute + PrefixRetention, else NativeFullAccounting); '
                    'systems_policy.py:21 (recompute), :50 (QSA forward count 15 vs 26)',
        description='keep the attention prefix of the 11 QSA layers 3,7,...,43 instead of recomputing it (only all 11 or '
                    'none: counts() hardcodes 15 vs 26 QSA forwards)'),
    'retain_expert_layers': dict(
        values=[True, False], applies_to=('systems',), lossy=[], new=[],
        sources={True: ['S_match', 'Q', 'Qo'], False: ['S0', 'S_match_full', 'Q_full', 'Qo_full']},
        honoured_by='build_unified_bundle.py:135 (g2_mode candidate iff experts retained); boundary_contract.py:30-36; '
                    'replay_training.py:33-47,96-101',
        description='retain the expert suffix of layers 0 and 23 (G2 candidate) instead of replaying them'),
    'qkv_subgroup': dict(
        values=[True, False], applies_to=('systems',), lossy=[], new=[],
        sources={True: ['S_match', 'S_match_full', 'Q', 'Q_full', 'Qo', 'Qo_full'], False: ['S0']},
        honoured_by='systems_runtime.py:58-62; family_runtime.py:159; systems_collection.py:49',
        description='QKV subgroup training path'),
    'shared_master_guard': dict(
        values=[True, False], applies_to=('systems',), lossy=[], new=[],
        sources={True: ['S_match', 'S_match_full', 'Q', 'Q_full', 'Qo', 'Qo_full'], False: ['S0']},
        honoured_by='systems_runtime.py:31-33; family_runtime.py:92; systems_collection.py:55',
        description='shared-master pack guard (MasterPackTraining)'),
    'moment_policy': dict(
        values=[NATIVE, SCALED, HOST], applies_to=('systems',), lossy=[SCALED, HOST], new=[],
        sources={NATIVE: ['S0', 'S_match', 'S_match_full'], SCALED: ['Q', 'Q_full'], HOST: ['Qo', 'Qo_full']},
        honoured_by='systems_transform.py:21-23; systems_runtime.py:16; systems_collection.py:20',
        description='Adam moment storage; the two scaled-fp16 encodings are lossy (Q/Qo: CE up to 6.9e-3 off U+C)'),
    'qsa_fwd_launch': dict(
        values=['4,1', '4,2', '8,2'], applies_to=('systems', 'W'), lossy=[], new=['4,2', '8,2'],
        sources={'4,1': ['S0', 'S_match', 'S_match_full', 'W (default)']},
        honoured_by='flash_qsa_tuned_launch_v1.py:20-28 (num_warps,num_stages; loaded by flash_qsa_fastmask_v1/v2.py:31-40)',
        description='QSA forward Triton launch (num_warps,num_stages); 8,2 is the vendored launch, 4,1 the tuned one '
                    '(bitwise identical on four dumped calls, flash_qsa_tuned_launch_v1.py:3-6)'),
    'qsa_dq_launch': dict(
        values=['4,2', '4,1', '8,1'], applies_to=('systems', 'W'), lossy=[], new=['4,1', '8,1'],
        sources={'4,2': ['S0', 'S_match', 'S_match_full', 'W (default)']},
        honoured_by='flash_qsa_tuned_launch_v1.py:27', description='QSA dQ Triton launch; 8,1 vendored'),
    'qsa_dkdv_launch': dict(
        values=['4,2', '4,1', '8,1'], applies_to=('systems', 'W'), lossy=[], new=['4,1', '8,1'],
        sources={'4,2': ['S0', 'S_match', 'S_match_full', 'W (default)']},
        honoured_by='flash_qsa_tuned_launch_v1.py:28', description='QSA dKdV Triton launch; 8,1 vendored'),
    'qsa_dkdv_fastmask': dict(
        values=['0', '1'], applies_to=('systems', 'W'), lossy=[], new=['1'],
        sources={'0': ['S0', 'S_match', 'S_match_full', 'W']},
        honoured_by='flash_qsa_fastmask_v2.py:44-46 (systems), flash_qsa_fastmask_v1.py:38 (W; but w_family_runtime.py:93 '
                    'asserts the W flags, which pin it to 0)',
        description="dKdV word-load fast mask: ~1.6x faster dKdV, NOT bitwise (0.02 % of dk/dv elements ~1 ulp; "
                    "flash_qsa_fastmask_v2.py:44-45); E1 gate decides"),
    'aux_readback_mode': dict(
        values=['control', 'deferred'], applies_to=_ALL, lossy=[], new=['deferred'],
        sources={'control': ['U_plus_C', 'B', 'W', 'S0', 'S_match', 'S_match_full', 'Q', 'Q_full', 'Qo', 'Qo_full']},
        honoured_by='train_flash_strategy_20260925.py:165-166,300 (trainer, every family); no family runtime pins it',
        description='aux-loss readback: control = per micro-batch, deferred = batched at the update end'),
}


def _systems(attn, experts, qkv, master, moment):
    return dict(family='systems', retain_attention_prefix=attn, retain_expert_layers=experts, qkv_subgroup=qkv,
                shared_master_guard=master, moment_policy=moment, **LAUNCH_REGISTERED, qsa_dkdv_fastmask='0',
                aux_readback_mode='control')


def _fixed(family):
    k = {name: None for name in KNOBS}
    k.update(family=family, aux_readback_mode='control')
    if family == 'W':
        k.update(LAUNCH_REGISTERED, qsa_dkdv_fastmask='0')
    return k


# Registered recipes as knob dicts (build_unified_bundle.py:132-144, systems_policy.py:9-23, prepare_precision_jobs.py:61-68).
REGISTERED = {
    'U_plus_C': _fixed('U_plus_C'), 'B': _fixed('B'), 'W': _fixed('W'),
    'S0': _systems(True, False, False, False, NATIVE),
    'S_match': _systems(True, True, True, True, NATIVE),
    'S_match_full': _systems(False, False, True, True, NATIVE),
    'Q': _systems(True, True, True, True, SCALED),
    'Q_full': _systems(False, False, True, True, SCALED),
    'Qo': _systems(True, True, True, True, HOST),
    'Qo_full': _systems(False, False, True, True, HOST),
}
E1_REGISTERED = ('U_plus_C', 'B', 'W', 'S0', 'S_match', 'S_match_full')


def _bit(v) -> str:
    return '1' if v else '0'


def custom_name(attn: bool, experts: bool, qkv: bool, master: bool) -> str:
    return f'SX_a{_bit(attn)}e{_bit(experts)}q{_bit(qkv)}m{_bit(master)}'


def canonical(knobs: dict, *, lossy: bool = False) -> tuple[dict, list, list]:
    """Defaults filled (registered launch/env values; family knobs it does not read set to None). Errors are static
    rejections citing the frozen code; warnings mark values no registered recipe ran (GPU-unverified)."""
    errors, warnings = [], []
    if type(knobs) is not dict:
        return {}, ['knobs must be an object'], []
    unknown = sorted(set(knobs) - set(KNOBS))
    if unknown:
        errors.append(f'unknown knobs: {unknown}')
    family = knobs.get('family')
    if family not in FAMILIES:
        return {}, errors + [f'family {family!r} not in {list(FAMILIES)}'], warnings
    out = {name: None for name in KNOBS}
    out['family'] = family
    for name in KNOBS[1:]:
        spec, value = SCHEMA[name], knobs.get(name)
        if family not in spec['applies_to']:
            if value is not None:
                errors.append(f'{name}: not read by family {family} ({spec["honoured_by"]}); leave it unset')
            continue
        if value is None:
            if name in SYSTEMS_KNOBS:
                errors.append(f'{name}: required for family systems (no default: the start is U+C)')
                continue
            value = LAUNCH_REGISTERED.get(name, {'qsa_dkdv_fastmask': '0', 'aux_readback_mode': 'control'}.get(name))
        if value not in spec['values']:
            errors.append(f'{name}={value!r} not in {spec["values"]}')
            continue
        if value in spec['lossy'] and not lossy:
            errors.append(f'{name}={value!r} is lossy (E2), excluded under E1 ({spec["honoured_by"]})')
        if value in spec['new']:
            warnings.append(f'{name}={value!r}: no registered recipe ran it (GPU-unverified on this path)')
        out[name] = value
    if family == 'systems' and not errors:
        if out['retain_expert_layers'] and not out['retain_attention_prefix']:
            errors.append('retain_expert_layers needs retain_attention_prefix: without the prefix the native '
                          'accounting asserts both retention lists are empty (native_full_accounting.py:15)')
    if family == 'W' and out.get('qsa_dkdv_fastmask') not in (None, '0'):
        errors.append('qsa_dkdv_fastmask must stay 0 in W: w_family_runtime.py:93 asserts the W flags '
                      '(w_family_contract.py:129-130)')
    if out.get('qsa_dkdv_fastmask') == '1' and not errors:
        warnings.append('qsa_dkdv_fastmask=1 is not bitwise (flash_qsa_fastmask_v2.py:44-45): E1 at best, never E0')
    return out, errors, warnings


def registered_name(knobs: dict) -> str | None:
    """The registered recipe whose knob dict equals the canonical knobs exactly (env knobs included), else None."""
    c, errors, _ = canonical(knobs, lossy=True)
    if errors:
        return None
    return next((name for name, k in REGISTERED.items() if k == c), None)


def variant_name(knobs: dict, *, lossy: bool = False) -> str:
    """The `infra_variant` a bundle for these knobs carries: the registered name of the family / systems row (env knobs
    do not change it), or a custom systems name for a row no recipe registered."""
    c, errors, _ = canonical(knobs, lossy=lossy)
    if errors:
        raise ValueError('; '.join(errors))
    if c['family'] != 'systems':
        return c['family']
    row = {k: c[k] for k in SYSTEMS_KNOBS}
    for name in _SYSTEMS_ROWS:
        if {k: REGISTERED[name][k] for k in SYSTEMS_KNOBS} == row:
            return name
    if c['moment_policy'] != NATIVE:
        raise ValueError('custom systems rows are native-moment only (lossy rows stay the registered Q family)')
    return custom_name(c['retain_attention_prefix'], c['retain_expert_layers'], c['qkv_subgroup'], c['shared_master_guard'])


def systems_policy(knobs: dict, name: str) -> dict:
    """The systems_policy.POLICIES row for these knobs, built field by field (systems_policy.py:9-23 layout)."""
    c, errors, _ = canonical(knobs, lossy=True)
    if errors or c['family'] != 'systems':
        raise ValueError(f'not a valid systems knob dict: {errors or c["family"]}')
    prefix = list(PREFIX) if c['retain_attention_prefix'] else []
    return dict(variant=name, retained_attention_layers=prefix,
                retained_expert_layers=list(EXPERT_PAIR) if c['retain_expert_layers'] else [],
                replay_enabled=True, qsa_fastmask_v2=True, segsum=True, qkv_subgroup=bool(c['qkv_subgroup']),
                shared_master_guard=bool(c['shared_master_guard']), moment_policy=c['moment_policy'],
                recompute=['full', 'block', 0] if prefix else ['full', 'uniform', 1],
                extra_gc_freeze=False, old_host_memo=False)


def job_env(knobs: dict) -> dict:
    """Env entries for job.json beyond the base job (prepare_precision_jobs.py:61-68). Systems bundles always carry the
    QSA launch and G2 entries (build_unified_bundle.py:136-137); W carries a launch entry only when it differs from the
    default, so the registered W bundle stays byte-identical; W's own flags are added by the builder."""
    c, errors, _ = canonical(knobs, lossy=True)
    if errors:
        raise ValueError('; '.join(errors))
    env = {}
    if c['family'] == 'systems':
        env.update({LAUNCH_ENV[k]: c[k] for k in LAUNCH_KNOBS})
        env['FLASH_QSA_DKDV_FASTMASK'] = c['qsa_dkdv_fastmask']
        env['FLASH_G2_MODE'] = 'candidate' if c['retain_expert_layers'] else 'control'
    elif c['family'] == 'W':
        env.update({LAUNCH_ENV[k]: c[k] for k in LAUNCH_KNOBS if c[k] != LAUNCH_REGISTERED[k]})
    if c['aux_readback_mode'] != 'control':
        env['FLASH_AUX_READBACK_MODE'] = c['aux_readback_mode']
    return env


def custom_rows() -> list[dict]:
    """Every E1 systems row that passes the static rules and no recipe registered (the custom-variant registry)."""
    rows = []
    for attn, experts, qkv, master in itertools.product((True, False), repeat=4):
        k = dict(_systems(attn, experts, qkv, master, NATIVE))
        if canonical(k)[1]:
            continue
        name = variant_name(k)
        if name.startswith('SX_'):
            rows.append(dict(name=name, knobs=k, policy=systems_policy(k, name)))
    return rows


def rejected_rows() -> list[dict]:
    """E1 systems rows the static rules reject, with the reason."""
    out = []
    for attn, experts, qkv, master in itertools.product((True, False), repeat=4):
        k = _systems(attn, experts, qkv, master, NATIVE)
        errors = canonical(k)[1]
        if errors:
            out.append(dict(name=custom_name(attn, experts, qkv, master), knobs=k, errors=errors))
    return out


def space(*, lossy: bool = False):
    """Every canonical knob dict that passes the static rules."""
    for family in FAMILIES:
        names = [n for n in KNOBS[1:] if family in SCHEMA[n]['applies_to']]
        for values in itertools.product(*[SCHEMA[n]['values'] for n in names]):
            k = dict(zip(names, values), family=family)
            c, errors, _ = canonical(k, lossy=lossy)
            if not errors:
                yield c
