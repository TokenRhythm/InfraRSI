"""S_F1: the frozen switch space of the flagship production path (AMENDMENT_EXT_003 Q1), on top of `knobs`.

`switch_space.json` (next to this file) is the frozen artifact: `generate()` rebuilds it and the tests require byte
equality; its sha256 goes into every receipt. One space for every F1 arm, start = U+C.

Switches (neutral names and values). Task views show `task_view()` only: names, values, applicability, one neutral
sentence per switch and per value; never `provenance` (which recipe used a value) and never stored timings.
  implementation_stack        native_rollbacks | production | production_qsa_memo_segsum | systems_policy
                              (the three families of the registered recipes are mutually exclusive in the frozen code:
                              one transform per `infra_variant`, family_transform.py:85-99; the rollback block and the W
                              block only build whole, uc_transform.py:57, w_family_runtime.py:93; so they are one switch)
  retain_attention_prefix, retain_expert_layers, qkv_subgroup, shared_master_guard        systems_policy only
  qsa_fwd_launch, qsa_dq_launch, qsa_dkdv_launch     systems_policy and production_qsa_memo_segsum
Fixed, not proposable: moment_policy = native_fp32_v_offload; FLASH_AUX_READBACK_MODE = control and every other FLASH_* /
NCCL variable outside the three families; the layout; seed 42; run_limit 30; FLASH_STOP 1024; qsa_dkdv_fastmask = '0'
(AMENDMENT_EXT_008 Q1: the frozen code asserts the dK/dV fast mask off before the first update, qsa_fastmask_training.py:17
and w_family_runtime.py:123, so '1' was never an option of the path). qsa_dkdv_fastmask stays a key of every canonical
point ('0' where the stack reads it, i.e. systems_policy; unset elsewhere), so recorded points keep their canonical dicts;
it is not in the switch table, task views list it under `fixed`, and any value but '0' is an invalid_proposal.

    validate(point) -> (canonical point, errors, warnings)      errors => invalid_proposal (zero GPU cost)
    to_knobs(point) / from_knobs(knobs)                         bridge to knobs / variant.build_bundle
    points()                                                    every admissible canonical point (353)
    RandomSwitchSampler(seed=42) / sample_order(seed=42, n)     F-RND, the ext/optico/b_rnd pattern
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from . import knobs as K

PATH = Path(__file__).resolve().parent / 'switch_space.json'
SCHEMA = 'flashnext-switch-space-v1'
SPACE_ID = 'S_F1'
SEED = 42
MAX_DRAWS = 5000

STACKS = {'native_rollbacks': 'U_plus_C', 'production': 'B', 'production_qsa_memo_segsum': 'W',
          'systems_policy': 'systems'}
FAMILY_STACK = {v: k for k, v in STACKS.items()}
START_STACK = 'native_rollbacks'
SYSTEMS_SWITCHES = ('retain_attention_prefix', 'retain_expert_layers', 'qkv_subgroup', 'shared_master_guard')
LAUNCH_SWITCHES = K.LAUNCH_KNOBS
SWITCHES = ('implementation_stack',) + SYSTEMS_SWITCHES + LAUNCH_SWITCHES + ('qsa_dkdv_fastmask',)   # point keys
# AMENDMENT_EXT_008 Q1: keys with one admissible value in every stack; not in the switch table, never drawn.
FIXED_KEYS = {'qsa_dkdv_fastmask': '0'}
PROPOSABLE = tuple(s for s in SWITCHES if s not in FIXED_KEYS)                                  # the switch table
SUPERSEDED_SHA256 = ['ea48b5226ab558341f35c7942aa8d787cbe1adb99900c91ef9a48f4bdff94b77']      # 677-point v1
APPLIES = {**{s: ('systems_policy',) for s in SYSTEMS_SWITCHES},
           **{s: ('systems_policy', 'production_qsa_memo_segsum') for s in LAUNCH_SWITCHES},
           'qsa_dkdv_fastmask': ('systems_policy',),
           'implementation_stack': tuple(STACKS)}
# Value the code uses when a switch its stack reads is not given (flash_qsa_tuned_launch_v1.py:26-28 defaults,
# flash_qsa_fastmask_v2.py:46 default '0'); systems switches have none and must be given.
CODE_DEFAULT = {**K.LAUNCH_REGISTERED, 'qsa_dkdv_fastmask': '0'}

DESCRIPTION = {
    'implementation_stack': 'Which implementation stack runs the fixed layout; the other switches apply only where '
                            'their stack reads them.',
    'retain_attention_prefix': 'Whether the activations of the attention prefix of the 11 QSA layers (3, 7, ..., 43) '
                               'are kept from the forward pass instead of being recomputed in the backward pass.',
    'retain_expert_layers': 'Whether the expert part of decoder layers 0 and 23 is kept from the forward pass instead '
                            'of being replayed in the backward pass (requires retain_attention_prefix).',
    'qkv_subgroup': 'Whether the QKV projections run through the subgroup training path.',
    'shared_master_guard': 'Whether the master parameters are packed through the shared-master guard.',
    'qsa_fwd_launch': 'Triton launch configuration "num_warps,num_stages" of the QSA block-sparse attention forward '
                      'kernel.',
    'qsa_dq_launch': 'Triton launch configuration "num_warps,num_stages" of the QSA block-sparse attention dQ kernel.',
    'qsa_dkdv_launch': 'Triton launch configuration "num_warps,num_stages" of the QSA block-sparse attention dK/dV '
                       'kernel.',
    'qsa_dkdv_fastmask': 'Whether the QSA dK/dV kernel reads its selection mask as packed words instead of per-byte '
                         'gathers.',
}
VALUE_DESCRIPTION = {
    'implementation_stack': {
        'native_rollbacks': 'native implementations for the token selector, sort, integer frontend, SwiGLU, fc1 '
                            'weight-gradient and parameter sync',
        'production': 'the production implementations of those six components',
        'production_qsa_memo_segsum': 'the production implementations plus QSA fast-mask v1, a host-sync memo and the '
                                      'segmented-sum MoE combine',
        'systems_policy': 'the production implementations plus the replay / segmented-sum / QSA fast-mask v2 path, '
                          'configured by the systems switches'},
}
DOMAIN = {
    'implementation_stack': 'family_transform.py:85-99; uc_contract.py; uc_transform.py:56-81; '
                            'w_family_contract.py; w_family_runtime.py:93,102; systems_transform.py:6-25',
    'retain_attention_prefix': 'systems_policy.py:4,17-23 (11-layer prefix or none), :47-57 (counts)',
    'retain_expert_layers': 'boundary_contract.py:30-36 (layers 0,23); native_full_accounting.py:15',
    'qkv_subgroup': 'systems_runtime.py:58-62; systems_collection.py:49-54',
    'shared_master_guard': 'systems_runtime.py:31-33; systems_collection.py:55',
    'qsa_fwd_launch': 'flash_qsa_tuned_launch_v1.py:20-28 (any int pair); vendored launch :3-6',
    'qsa_dq_launch': 'flash_qsa_tuned_launch_v1.py:20-28; vendored :3-6',
    'qsa_dkdv_launch': 'flash_qsa_tuned_launch_v1.py:20-28; vendored :3-6',
    'qsa_dkdv_fastmask': 'flash_qsa_fastmask_v2.py:44-46 (not bitwise, ~1 ulp on 0.02 % of dk/dv)',
}


def _values(switch: str) -> list:
    """Admissible values (a fixed key has one)."""
    if switch == 'implementation_stack':
        return list(STACKS)
    if switch in FIXED_KEYS:
        return [FIXED_KEYS[switch]]
    return list(K.SCHEMA[switch]['values'])


def start_point() -> dict:
    return {s: (START_STACK if s == 'implementation_stack' else None) for s in SWITCHES}


def to_knobs(point: dict) -> dict:
    """Canonical S_F1 point -> knob dict for knobs / variant (moment native, aux control: the fixed values)."""
    stack = point['implementation_stack']
    family = STACKS[stack]
    k = {name: None for name in K.KNOBS}
    k.update(family=family, aux_readback_mode='control')
    if family == 'systems':
        k.update({s: point[s] for s in SYSTEMS_SWITCHES}, moment_policy=K.NATIVE)
    if family in ('systems', 'W'):
        k.update({s: point[s] for s in LAUNCH_SWITCHES})
        k['qsa_dkdv_fastmask'] = point['qsa_dkdv_fastmask'] if family == 'systems' else '0'
    return k


def from_knobs(knob_dict: dict) -> dict:
    c, errors, _ = K.canonical(knob_dict)
    if errors:
        raise ValueError('; '.join(errors))
    if c['aux_readback_mode'] != 'control' or (c['family'] == 'systems' and c['moment_policy'] != K.NATIVE):
        raise ValueError('outside S_F1: aux readback and moment policy are fixed')
    p = start_point()
    p['implementation_stack'] = FAMILY_STACK[c['family']]
    for s in SWITCHES[1:]:
        if p['implementation_stack'] in APPLIES[s]:
            p[s] = c[s]
    return p


def validate(point) -> tuple[dict, list, list]:
    """Membership in S_F1. Switches a stack does not read must be unset; systems switches must be given; launch / fast
    mask switches default to the code's own default when not given. errors -> invalid_proposal."""
    errors, warnings = [], []
    if type(point) is not dict:
        return {}, ['a point is an object of switch -> value'], []
    unknown = sorted(set(point) - set(SWITCHES))
    if unknown:
        errors.append(f'not switches of {SPACE_ID}: {unknown} (fixed or excluded: layout, moment policy, aux readback, '
                      'seed, run length, other env)')
    stack = point.get('implementation_stack')
    if stack not in STACKS:
        return {}, errors + [f'implementation_stack {stack!r} not in {list(STACKS)}'], []
    out = start_point()
    out['implementation_stack'] = stack
    for s in SWITCHES[1:]:
        v = point.get(s)
        if s in FIXED_KEYS:                     # '0' or unset -> '0' where the stack reads it (else unset); else refused
            if v is not None and v != FIXED_KEYS[s]:
                errors.append(f'{s}={v!r} not in {_values(s)}: fixed')
            elif stack in APPLIES[s]:
                out[s] = FIXED_KEYS[s]
            continue
        if stack not in APPLIES[s]:
            if v is not None:
                errors.append(f'{s}: not read by implementation_stack {stack}; leave it unset')
            continue
        if v is None:
            if s in CODE_DEFAULT:
                v = CODE_DEFAULT[s]
            else:
                errors.append(f'{s}: required with implementation_stack {stack}')
                continue
        if v not in _values(s):
            errors.append(f'{s}={v!r} not in {_values(s)}')
            continue
        out[s] = v
    if errors:
        return {}, errors, warnings
    _, kerr, kwarn = K.canonical(to_knobs(out))
    if kerr:
        return {}, kerr, warnings
    warnings += kwarn
    return out, [], warnings


def key(point: dict) -> str:
    return json.dumps({s: point.get(s) for s in SWITCHES}, sort_keys=True, separators=(',', ':'))


def points() -> list[dict]:
    """Every admissible canonical point, in a fixed order."""
    out = []
    for k in K.space():
        if k['aux_readback_mode'] != 'control':
            continue
        if k['family'] == 'systems' and k['moment_policy'] != K.NATIVE:
            continue
        if k['family'] in ('W', 'systems') and k['qsa_dkdv_fastmask'] != FIXED_KEYS['qsa_dkdv_fastmask']:
            continue                                                       # EXT_008 Q1 (W: w_family_contract FLAGS)
        out.append(from_knobs(k))
    seen, uniq = set(), []
    for p in out:
        if key(p) not in seen:
            seen.add(key(p))
            uniq.append(p)
    return uniq


def registered_points() -> dict:
    return {name: from_knobs(K.REGISTERED[name]) for name in K.E1_REGISTERED}


def task_view() -> dict:
    """What every arm's task view may show: no recipe names, no provenance, no timings, no recommended value."""
    return {'space_id': SPACE_ID, 'start': start_point(),
            'switches': [{'name': s, 'values': _values(s), 'applies_to_stacks': list(APPLIES[s]),
                          'default_when_unset': CODE_DEFAULT.get(s), 'description': DESCRIPTION[s],
                          **({'value_descriptions': VALUE_DESCRIPTION[s]} if s in VALUE_DESCRIPTION else {})}
                         for s in PROPOSABLE],
            'constraints': ['retain_expert_layers=true requires retain_attention_prefix=true'],
            'fixed': 'layout TP8/EP32/CP1/ETP1/DP4, 2 micro-batches, seq 32768, full recompute unless the attention prefix '
                     'is retained, fp32 Adam moments, seed 42, 30 updates, QSA dK/dV fast mask off (qsa_dkdv_fastmask = 0)'}


def _encode(obj) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=False, ensure_ascii=False) + '\n').encode('utf-8')


def generate() -> bytes:
    """The bytes of switch_space.json (deterministic)."""
    pts = points()
    by_stack = {}
    for p in pts:
        by_stack[p['implementation_stack']] = by_stack.get(p['implementation_stack'], 0) + 1
    reg = registered_points()
    switches = []
    for s in PROPOSABLE:
        prov = {}
        if s == 'implementation_stack':
            prov = {v: K.SCHEMA['family']['sources'][STACKS[v]] for v in STACKS}
        else:
            prov = {str(v).lower() if isinstance(v, bool) else v: srcs for v, srcs in K.SCHEMA[s]['sources'].items()}
        status = {}
        for v in _values(s):
            label = str(v).lower() if isinstance(v, bool) else v
            used = any(p.get(s) == v for p in reg.values())
            status[label] = 'registered recipe' if used else 'admitted by the E1 spike (bundle builds and CPU-materializes); GPU run unverified'
        switches.append({'name': s, 'values': _values(s), 'start': start_point()[s], 'applies_to_stacks': list(APPLIES[s]),
                         'default_when_unset': CODE_DEFAULT.get(s), 'description': DESCRIPTION[s],
                         **({'value_descriptions': VALUE_DESCRIPTION[s]} if s in VALUE_DESCRIPTION else {}),
                         'domain': DOMAIN[s], 'provenance_not_for_task_views': prov, 'status': status})
    doc = {
        'schema': SCHEMA, 'space_id': SPACE_ID,
        'ruling': 'AMENDMENT_EXT_003 Q1 (D-F1-2); corrected once by AMENDMENT_EXT_008 Q1 (qsa_dkdv_fastmask fixed at 0)',
        'superseded_sha256': list(SUPERSEDED_SHA256),
        'code_freeze_sha256': 'd6a8f3b6b6c95e9013ffedcc58ddd62fb37708056ad75f013a72bcaf0c3fc826',
        'start': start_point(), 'switches': switches,
        'constraints': [{'id': 'experts_need_prefix', 'rule': 'retain_expert_layers=true requires retain_attention_prefix=true',
                         'source': 'native_full_accounting.py:15 (asserted at construction, before the first update)'}],
        'fixed': {'moment_policy': K.NATIVE, 'aux_readback_mode': 'control', 'qsa_dkdv_fastmask': FIXED_KEYS['qsa_dkdv_fastmask'],
                  'seed': 42, 'run_limit': 30, 'flash_stop': 1024,
                  'layout': 'TP8 / EP32 / CP1 / ETP1 / DP4, 2 micro-batches, seq 32768 (uc_contract.CONTRACT)',
                  'precision_arm': 'bf16', 'data_order': 'frozen_luna_q50_schedule_and_pack_plan_absolute_update_no_seed_reshuffle'},
        'excluded': ['moment_policy scaled_fp16_dual_cuda / scaled_fp16_v_offload (lossy, E2)',
                     'FLASH_AUX_READBACK_MODE and every FLASH_* / NCCL variable outside the three families',
                     'layout, seed 43, run_limit, FLASH_STOP, data order, precision arm, allocator policy, LAUNCH_MATH',
                     'experts retained without the attention prefix (constraint)',
                     "qsa_dkdv_fastmask = '1' (AMENDMENT_EXT_008 Q1; 324 points of v1 removed, every systems_policy point "
                     "with '1'): the frozen code asserts the dK/dV fast mask off before the first update - "
                     "qsa_fastmask_training.py:17 'assert not tuned.DKDV_FAST' (systems_policy) and "
                     "w_family_runtime.py:123 'assert not self.tuned.DKDV_FAST' (production_qsa_memo_segsum; also "
                     "w_family_contract.py:21-22 FLAGS FLASH_QSA_DKDV_FASTMASK '0'); DKDV_FAST is read from "
                     "FLASH_QSA_DKDV_FASTMASK at import, flash_qsa_fastmask_v2.py:44-46"],
        'registered_points_not_for_task_views': reg,
        'size': {'admissible_points': len(pts), 'by_stack': by_stack,
                 'systems_rows': len({json.dumps({s: p[s] for s in SYSTEMS_SWITCHES}) for p in pts
                                      if p['implementation_stack'] == 'systems_policy'})},
        'task_view_fields': ['name', 'values', 'applies_to_stacks', 'default_when_unset', 'description', 'value_descriptions'],
    }
    return _encode(doc)


def load(path: Path = PATH) -> dict:
    """The frozen file; refuses a file that is not what generate() makes (the space cannot drift silently)."""
    raw = Path(path).read_bytes()
    if raw != generate():
        raise ValueError(f'{path} differs from the generated {SPACE_ID}: regenerate and re-freeze deliberately')
    return json.loads(raw)


def sha256(path: Path = PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RandomSwitchSampler:
    """F-RND (AMENDMENT_EXT_003 Q5): seed 42; the stack uniform over its values, then every switch that stack reads
    uniform over its admissible values (a switch is drawn only where it can matter, as ext/optico/b_rnd); inadmissible,
    duplicate and start (U+C) draws cost nothing and are redrawn; gives up after MAX_DRAWS draws."""

    def __init__(self, seed: int = SEED):
        self.seed = seed
        self.rng = random.Random(seed)
        self.draws = 0

    def draw(self) -> dict:
        self.draws += 1
        stack = self.rng.choice(list(STACKS))
        p = {'implementation_stack': stack}
        for s in PROPOSABLE[1:]:                      # fixed keys are not drawn (validate sets them)
            if stack in APPLIES[s]:
                p[s] = self.rng.choice(_values(s))
        return p

    def next(self, seen=()) -> dict | None:
        seen = set(seen) | {key(start_point())}
        for _ in range(MAX_DRAWS):
            c, errors, _ = validate(self.draw())
            if errors or key(c) in seen:
                continue
            return c
        return None

    def sequence(self, n: int, seen=()) -> list[dict]:
        out, keys = [], set(seen)
        for _ in range(n):
            p = self.next(keys)
            if p is None:
                break
            keys.add(key(p))
            out.append(p)
        return out


def sample_order(seed: int = SEED, n: int = 12, seen=()) -> list[dict]:
    """The first n points F-RND proposes with this seed (deterministic; `seen` = keys already screened)."""
    return RandomSwitchSampler(seed).sequence(n, seen)


if __name__ == '__main__':                       # python -m infrarsi.ext.flashnext.switch_space  -> (re)write the file
    PATH.write_bytes(generate())
    print(PATH, sha256())
