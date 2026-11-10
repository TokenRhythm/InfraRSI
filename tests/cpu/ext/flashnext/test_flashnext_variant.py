"""Bundles from knob dicts (ext/flashnext/variant.py) against the frozen formal-54 builder and bundles. CPU only.

1. Every E1 registered recipe built from its knob dict is byte-identical to the formal-54 bundle on disk (after the wave
   controller's bind edits; only the local build directory differs, and materialize_result.json is the node's product).
2. Locally materialized (the frozen worker's --materialize-only path), U+C and S_match give exactly the checks and static
   block the formal-54 node materialization recorded.
3. The custom runtime only registers names: registered rows unchanged, registered recipes keep their effective trainer.
4. Every custom E1 systems row builds and materializes; the statically rejected row trips the runtime's own assert.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from infrarsi.ext.flashnext import knobs

ENTRIES = {'U_plus_C': 'A1_03_U_plus_C', 'B': 'A1_08_B', 'W': 'A1_06_W', 'S0': 'A1_01_S0', 'S_match': 'A1_04_S_match',
           'S_match_full': 'A1_09_S_match_full'}
PREFIX_TRAINER = 'df3d91b3bd148247fd8394ff844042647fdd565745865150702385aa57c13bad'      # S0 / S_match (formal-54 receipts)
FULL_TRAINER = '91ae17f0669ef6fc507bcf26bf701fb1feefb3a3117f4b192246aef482cffaef'        # S_match_full


def _rebuild(V, name, out, **kw):
    reg = V.registered_bundle(ENTRIES[name])
    ra, rj = V.read(reg / 'attempt.json'), V.read(reg / 'job.json')
    res = V.build_bundle(knobs.REGISTERED[name], group=rj['group'], port=rj['master_port'], output=out,
                         run_limit=ra['run_limit'], seed=ra['training_seed'], ids=V.ids_of(reg), **kw)
    return reg, ra, rj, res


@pytest.mark.parametrize('name', list(ENTRIES))
def test_registered_recipe_from_knobs_is_byte_identical(frozen, registered, tmp_path_factory, name):
    V = frozen
    reg, ra, rj, res = _rebuild(V, name, tmp_path_factory.mktemp('b'))
    ours = Path(res['bundle'])
    assert ours.name == reg.name and res['spec']['policy'] == ra.get('systems_policy')
    V.bind_like(ours, ra['phase_gate'], mode=rj['diagnostic_kind'].removeprefix('rsi_'))
    cmp = V.compare_bundles(ours, reg)
    assert cmp['identical'], cmp
    assert cmp['same_files'] == 93 and cmp['skipped'] == ['materialize_result.json']


@pytest.mark.parametrize('name', ['U_plus_C', 'S_match'])
def test_local_materialization_matches_the_formal54_node(frozen, registered, tmp_path_factory, name):
    V = frozen
    base = tmp_path_factory.mktemp('m')
    reg, ra, rj, res = _rebuild(V, name, base / 'x')
    V.bind_like(res['bundle'], ra['phase_gate'], mode=rj['diagnostic_kind'].removeprefix('rsi_'))
    mat = V.materialize_local(res['bundle'], base / 'mat')
    assert mat['ok'], mat.get('error')
    remote = V.remote_receipt(reg)
    assert mat['receipt']['checks'] == remote['checks']               # incl. effective_sha256 of the trainer source
    assert mat['receipt']['static'] == remote['static']
    assert mat['file_sha256_lf'] == mat['effective_sha256']
    assert mat['receipt']['status'] == 'PASS_LOCAL_CPU_SOURCE_ONLY' and remote['status'] == 'PASS_MATERIALIZED_SOURCE'


def test_custom_runtime_only_registers_names(frozen, custom_rt):
    V = frozen
    assert custom_rt['changed_files'] == ['collect_family.py', 'family_policy.py', 'family_runtime.py',
                                          'family_transform.py', 'family_worker.py', 'systems_policy.py']
    assert custom_rt['edits'] == dict(V.NAME_TUPLES, **{'systems_policy.py': 9})
    code = ('import json,sys;sys.path.insert(0,sys.argv[1]);import systems_policy as s,family_policy as f;'
            'print(json.dumps(dict(P=s.POLICIES,V=list(f.VARIANTS))))')
    dump = lambda rt: json.loads(subprocess.run([sys.executable, '-B', '-c', code, str(rt)], capture_output=True,  # noqa: E731
                                                text=True, check=True).stdout)
    frozen_t, patched = dump(V.FROZEN_RUNTIME), dump(custom_rt['runtime'])
    assert all(patched['P'][n] == frozen_t['P'][n] for n in frozen_t['P'])
    assert [v for v in patched['V'] if not v.startswith('SX_')] == frozen_t['V']      # inserted after 'S_match_full'
    for row in knobs.custom_rows():
        assert patched['P'][row['name']] == row['policy']
        assert row['name'] in patched['V']


def test_registered_recipe_on_the_custom_runtime_keeps_its_trainer(frozen, registered, custom_rt, tmp_path_factory):
    V = frozen
    base = tmp_path_factory.mktemp('p')
    res = V.build_bundle(knobs.REGISTERED['S_match'], group='A', port=48270, output=base / 'x',
                         runtime_dir=custom_rt['runtime'], force_runtime=True)
    mat = V.materialize_local(res['bundle'], base / 'mat')
    assert mat['ok'], mat.get('error')
    remote = V.remote_receipt(V.registered_bundle(ENTRIES['S_match']))
    assert mat['receipt']['checks'] == remote['checks']
    diff = {k for k in set(mat['receipt']['static']) | set(remote['static'])
            if mat['receipt']['static'].get(k) != remote['static'].get(k)}
    assert diff == {'bundle_code_sha256'}


def test_every_custom_row_builds_and_materializes(frozen, custom_rt, tmp_path_factory):
    V = frozen
    seen = {}
    for row in knobs.custom_rows():
        base = tmp_path_factory.mktemp('c')
        res = V.build_bundle(row['knobs'], group='A', port=48270, output=base / 'x', runtime_dir=custom_rt['runtime'])
        attempt = V.read(os.path.join(res['bundle'], 'attempt.json'))
        assert attempt['infra_variant'] == row['name'] and attempt['systems_policy'] == row['policy']
        assert attempt['g2_mode'] == ('candidate' if row['policy']['retained_expert_layers'] else 'control')
        mat = V.materialize_local(res['bundle'], base / 'mat')
        assert mat['ok'], (row['name'], mat.get('error'))
        seen[row['name']] = mat['effective_sha256']
    # The source transform depends only on prefix retention and the moment policy (systems_transform.py:12-23):
    # every custom row runs the same trainer source as a registered recipe; the rest is run-time policy.
    for name, effective in seen.items():
        assert effective == (PREFIX_TRAINER if name.startswith('SX_a1') else FULL_TRAINER), name


def test_rejected_row_trips_the_runtime_assert(frozen):
    """CPU materialization would not catch experts-without-prefix (the transform ignores it); the native accounting
    asserts at construction, before the first update (native_full_accounting.py:15)."""
    code = ('import os,sys,traceback;sys.path.insert(0,sys.argv[1]);import systems_policy as s\n'
            "s._POLICY=dict(s.POLICIES['S_match_full'],retained_expert_layers=[0,23],variant='probe')\n"
            'from native_full_accounting import NativeFullAccounting\n'
            'try:\n NativeFullAccounting(*([None]*8));print("accepted")\n'
            'except AssertionError as e:\n t=traceback.extract_tb(e.__traceback__)[-1];print(os.path.basename(t.filename),t.lineno)\n')
    out = subprocess.run([sys.executable, '-B', '-c', code, str(frozen.FROZEN_RUNTIME)], capture_output=True,
                         text=True, check=True).stdout.split()
    assert out == ['native_full_accounting.py', '15']


def test_custom_rows_need_the_custom_runtime(frozen, tmp_path):
    row = knobs.custom_rows()[0]
    with pytest.raises(ValueError, match='runtime_dir'):
        frozen.bundle_spec(row['knobs'], group='A', port=48270, output=tmp_path)
    with pytest.raises(ValueError, match='lossy'):
        frozen.bundle_spec(knobs.REGISTERED['Qo'], group='A', port=48270, output=tmp_path)


@pytest.mark.parametrize('name', ['U_plus_C', 'S_match'])
def test_rerooted_bundle_trains_exactly_what_formal54_trained(frozen, registered, tmp_path_factory, name):
    """Every write re-rooted under /data/infrarsi/ext (family_worker.py:40 edited in the custom copy, helper roots
    re-pointed): the effective trainer source and every materialization check equal the formal-54 node receipt; only
    the bundle's code hash differs."""
    V = frozen
    base = tmp_path_factory.mktemp('rr')
    rt = V.custom_runtime(base / 'rt', root=V.FN_ROOT)
    res = V.build_bundle(knobs.REGISTERED[name], group='A', port=48270, output=base / 'x', runtime_dir=rt['runtime'],
                         root=V.FN_ROOT)
    bundle = Path(res['bundle'])
    V.ext_bind(bundle, cap_s=1800)
    lay = V.remote_layout(bundle)
    assert lay['metadata'].startswith(V.FN_ROOT + '/metadata/') and lay['output_root'].startswith(V.FN_ROOT + '/runs/')
    assert all(r.startswith(V.FN_ROOT + '/metadata/') for _, r in lay['files'])
    mat = V.materialize_local(bundle, base / 'mat')
    assert mat['ok'], mat.get('error')
    remote = V.remote_receipt(V.registered_bundle(ENTRIES[name]))
    assert mat['receipt']['checks'] == remote['checks']
    diff = {k for k in set(mat['receipt']['static']) | set(remote['static'])
            if mat['receipt']['static'].get(k) != remote['static'].get(k)}
    assert diff == {'bundle_code_sha256'}
    with pytest.raises(ValueError, match='run root'):
        V.bundle_spec(knobs.REGISTERED[name], group='A', port=48270, output=base / 'y', runtime_dir=V.FROZEN_RUNTIME,
                      force_runtime=True, root=V.FN_ROOT)
