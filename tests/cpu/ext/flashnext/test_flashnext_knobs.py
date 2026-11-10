"""Knob schema and static acceptance rules (ext/flashnext/knobs.py). Pure Python except the table comparison."""
import json
import subprocess
import sys
from collections import Counter

import pytest

from infrarsi.ext.flashnext import knobs


def test_registered_recipes_are_canonical_and_named():
    for name, k in knobs.REGISTERED.items():
        c, errors, warnings = knobs.canonical(k, lossy=True)
        assert not errors and c == k, (name, errors)
        assert knobs.variant_name(k, lossy=True) == name
        assert knobs.registered_name(k) == name
        assert not [w for w in warnings if 'GPU-unverified' in w], (name, warnings)


def test_lossy_rows_are_rejected_under_e1():
    for name in ('Q', 'Q_full', 'Qo', 'Qo_full'):
        errors = knobs.canonical(knobs.REGISTERED[name])[1]
        assert any('lossy (E2)' in e for e in errors), name
    for name in knobs.E1_REGISTERED:
        assert not knobs.canonical(knobs.REGISTERED[name])[1], name


def test_experts_without_the_attention_prefix_are_rejected():
    k = dict(knobs.REGISTERED['S_match_full'], retain_expert_layers=True)
    errors = knobs.canonical(k)[1]
    assert errors and 'native_full_accounting.py:15' in errors[0]
    assert [r['name'] for r in knobs.rejected_rows()] == ['SX_a0e1q1m1', 'SX_a0e1q1m0', 'SX_a0e1q0m1', 'SX_a0e1q0m0']


def test_knobs_a_family_does_not_read_must_stay_unset():
    k = dict(knobs.REGISTERED['U_plus_C'], qsa_fwd_launch='8,2')
    assert any('not read by family U_plus_C' in e for e in knobs.canonical(k)[1])
    k = dict(knobs.REGISTERED['B'], qkv_subgroup=True)
    assert any('not read by family B' in e for e in knobs.canonical(k)[1])
    k = dict(knobs.REGISTERED['W'], qsa_dkdv_fastmask='1')
    assert any('w_family_runtime.py:93' in e for e in knobs.canonical(k)[1])
    assert any('unknown knobs' in e for e in knobs.canonical(dict(knobs.REGISTERED['B'], tp=4))[1])


def test_systems_rows_need_every_policy_knob():
    k = {'family': 'systems', 'retain_attention_prefix': True}
    errors = knobs.canonical(k)[1]
    assert {e.split(':')[0] for e in errors} == {'retain_expert_layers', 'qkv_subgroup', 'shared_master_guard', 'moment_policy'}


def test_new_values_are_flagged_unverified_and_fastmask_non_bitwise():
    k = dict(knobs.REGISTERED['S_match'], qsa_fwd_launch='8,2', qsa_dkdv_fastmask='1', aux_readback_mode='deferred')
    c, errors, warnings = knobs.canonical(k)
    assert not errors
    assert sum('GPU-unverified' in w for w in warnings) == 3
    assert any('not bitwise' in w for w in warnings)
    assert knobs.variant_name(c) == 'S_match'                     # env knobs never change the variant
    env = knobs.job_env(c)
    assert env['FLASH_QSA_FWD_LAUNCH'] == '8,2' and env['FLASH_QSA_DKDV_FASTMASK'] == '1'
    assert env['FLASH_AUX_READBACK_MODE'] == 'deferred' and env['FLASH_G2_MODE'] == 'candidate'


def test_job_env_of_registered_recipes_matches_the_frozen_builder():
    """build_unified_bundle.py:136-137,144: systems rows always carry the launch and G2 entries; W only its flags."""
    assert knobs.job_env(knobs.REGISTERED['S0']) == {'FLASH_QSA_FWD_LAUNCH': '4,1', 'FLASH_QSA_DQ_LAUNCH': '4,2',
                                                      'FLASH_QSA_DKDV_LAUNCH': '4,2', 'FLASH_QSA_DKDV_FASTMASK': '0',
                                                      'FLASH_G2_MODE': 'control'}
    assert knobs.job_env(knobs.REGISTERED['S_match'])['FLASH_G2_MODE'] == 'candidate'
    for name in ('U_plus_C', 'B', 'W'):
        assert knobs.job_env(knobs.REGISTERED[name]) == {}, name
    assert knobs.job_env(dict(knobs.REGISTERED['W'], qsa_dq_launch='8,1')) == {'FLASH_QSA_DQ_LAUNCH': '8,1'}


def test_space_size():
    e1 = list(knobs.space())
    assert Counter(k['family'] for k in e1) == {'systems': 12 * 27 * 2 * 2, 'W': 27 * 2, 'U_plus_C': 2, 'B': 2}
    rows = {knobs.variant_name(k) for k in e1 if k['family'] == 'systems'}
    assert len(rows) == 12 and {'S0', 'S_match', 'S_match_full'} <= rows
    assert len(knobs.custom_rows()) == 9
    assert len({json.dumps(k, sort_keys=True) for k in e1}) == len(e1)          # canonical form: no duplicates


def test_knob_policies_equal_the_frozen_table(frozen):
    """systems_policy(knobs) builds each registered row field by field; it must equal POLICIES[name] exactly."""
    code = ('import json,sys;sys.path.insert(0,sys.argv[1]);import systems_policy as s;'
            'print(json.dumps(s.POLICIES))')
    table = json.loads(subprocess.run([sys.executable, '-B', '-c', code, str(frozen.FROZEN_RUNTIME)],
                                      capture_output=True, text=True, check=True).stdout)
    assert set(table) == {'Q', 'S0', 'S_match', 'S_match_full', 'Q_full', 'Qo', 'Qo_full'}
    for name, row in table.items():
        assert knobs.systems_policy(knobs.REGISTERED[name], name) == row, name


@pytest.mark.parametrize('bad', [{'family': 'X'}, {'family': 'systems', 'retain_attention_prefix': 'yes',
                                                    'retain_expert_layers': False, 'qkv_subgroup': True,
                                                    'shared_master_guard': True, 'moment_policy': knobs.NATIVE}])
def test_bad_values_are_errors(bad):
    assert knobs.canonical(bad)[1]
