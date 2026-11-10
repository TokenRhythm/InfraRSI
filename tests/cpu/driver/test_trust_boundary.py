"""Codex R2 trust-boundary findings: HMAC over the whole validated receipt (1), launch ledger and request echoes (2),
bound resume signals (3, see test_harness_validate), overlay identity (4); E1 completeness (5) is in test_scoring."""
import copy
import hashlib
import json
import os
import subprocess

import pytest

from driver_helpers import (DUMMY_MOD, attempt_dir, indexed_remote, make_request, pin_overlay, rewrite_receipt,
                            sim_launcher, simulated, validate_dir)

from infrarsi.contracts import digest
from infrarsi.harness import codesync, ledger, run_screen, signing, validate as V
from infrarsi.harness.remote import LocalRemote
from infrarsi.harness.runner import observe, recheck_against
from infrarsi.reward import evaluate


def _names(rv):
    return {v.split(':')[0] for v in rv['violations']}


# ---------------------------------------------------------------- 1. the signature covers the harness block

CONSUMED = [('harness', 'numerics', 'compared'), ('harness', 'numerics', 'class'), ('harness', 'numerics', 'is_reference'),
            ('harness', 'numerics', 'reference_attempt_id'), ('harness', 'numerics', 'violations'),
            ('harness', 'numerics', 'grad_norm', 'log_bias_window'), ('harness', 'numerics', 'self_check', 'repeated_pairs'),
            ('harness', 'bindings', 'signals_bound'), ('harness', 'bindings', 'layout_digest'), ('harness', 'failure_kind'),
            ('harness', 'gpu_hours'), ('harness', 'stored'), ('harness', 'warnings'), ('harness', 'microbatches'),
            ('harness', 'derived', 'tail_step_ratio'), ('valid',), ('numerics_role',), ('step_time', 'mean_window_max_rank_s')]


def _mutate(rec, path):
    d = rec
    for k in path[:-1]:
        d = d.setdefault(k, {})
    old = d.get(path[-1])
    d[path[-1]] = (not old) if isinstance(old, bool) else ([['forged']] if isinstance(old, list) else
                                                           (old + 1.0 if isinstance(old, float) else 'forged'))


@pytest.fixture(scope='module')
def signed_pair(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('sig')
    ref = simulated(tmp, make_request('D2', plan='B', purpose='score'))
    cand = simulated(tmp, make_request('D2', plan='B', purpose='score', numerics_role='candidate',
                                       config={'tensor_model_parallel_size': 2}), reference=ref)
    assert ref['valid'] and cand['valid'] and signing.verify(ref) and signing.verify(cand)
    return ref, cand


@pytest.mark.parametrize('path', CONSUMED, ids=['.'.join(p) for p in CONSUMED])
def test_any_mutation_without_resigning_is_rejected(signed_pair, path):
    ref, cand = signed_pair
    bad = copy.deepcopy(cand)
    _mutate(bad, path)
    assert not signing.verify(bad) and not V.verify_receipt_digest(bad)
    assert 'receipt_signature' in _names(evaluate(bad, ref))
    req = make_request('D2', plan='B', purpose='score')
    forged_ref = copy.deepcopy(ref)
    _mutate(forged_ref, path)
    assert any('signature' in p for p in V.reference_problems(forged_ref, task=req.task, plan='B', steps=30, window=req.window))
    with pytest.raises(ValueError):
        recheck_against(bad, ref)
    assert 'parent_receipt' in _names(evaluate(cand, forged_ref))


def test_review_probe_compared_flag_and_reference_id(signed_pair):
    """The R2 probe: edit only harness.numerics.compared / reference_attempt_id - no longer verifies."""
    ref, cand = signed_pair
    bad = copy.deepcopy(cand)
    bad['harness']['numerics'].update(compared=True, reference_attempt_id='expected')
    assert not signing.verify(bad)
    old_style = copy.deepcopy(cand)                       # a recomputable digest is no signature
    old_style.pop(signing.SIGNATURE_FIELD)
    old_style['harness']['receipt_digest'] = digest({k: v for k, v in old_style.items() if k != 'harness'})
    assert not signing.verify(old_style)
    other_key = copy.deepcopy(cand)
    other_key[signing.SIGNATURE_FIELD]['mac'] = hashlib.sha256(b'other key').hexdigest()
    assert not signing.verify(other_key)


def test_signature_is_final_after_storage(tmp_path, fsroot):
    """run_screen signs after the lease release and the storage path are recorded; the remote and local copies verify."""
    r = indexed_remote(fsroot, launcher=sim_launcher())
    req = make_request('D2', nodes=['WINH2'])
    rec = run_screen(req, remote=r, poll_s=0.01, local_dir=tmp_path / 'rc', log=lambda m: None)
    assert rec['valid'] and signing.verify(rec) and rec['harness']['stored'] and 'leases_released' in rec['harness']
    remote_copy = json.loads(r.local(rec['harness']['stored']).read_text())
    local_copy = json.loads((tmp_path / 'rc' / f'{req.attempt_id}.json').read_text())
    assert signing.verify(remote_copy) and signing.verify(local_copy)
    remote_copy['harness']['numerics']['class'] = 'E0'                  # WINH-side edit of the stored copy
    assert not signing.verify(remote_copy)


def test_key_file_is_local_restricted_and_never_in_receipts(tmp_path, monkeypatch, signed_pair):
    key = tmp_path / 'k' / 'receipt_hmac.key'
    monkeypatch.setenv(signing.KEY_ENV, str(key))
    rec = signing.sign({'schema': 'x', 'n': 1})
    raw = key.read_bytes()
    assert len(raw) == 32 and signing.verify(rec) and rec[signing.SIGNATURE_FIELD]['key_id'] == signing.key_id(raw)
    assert raw.hex() not in json.dumps(rec) and raw.hex() not in json.dumps(signed_pair[1])
    if os.name == 'nt':
        acl = subprocess.run(['icacls', str(key)], capture_output=True, text=True).stdout
        user = os.environ.get('USERNAME', '')
        assert user.lower() in acl.lower() and '(I)' not in acl and 'Everyone' not in acl and 'Users:' not in acl
    else:
        assert (key.stat().st_mode & 0o077) == 0
    monkeypatch.setenv(signing.KEY_ENV, str(tmp_path / 'other.key'))      # another PC's key does not verify it
    assert not signing.verify(rec)


# ---------------------------------------------------------------- 2. launch ledger and trusted request echoes

def test_launch_ledger_and_observe_refuse_a_flipped_role(tmp_path, fsroot):
    r = indexed_remote(fsroot, launcher=sim_launcher())
    req = make_request('D2', nodes=['WINH2'], numerics_role='candidate')
    ref = run_screen(make_request('D2', nodes=['WINH2']), remote=r, poll_s=0.01, local_dir=None, log=lambda m: None)
    req.reference_receipt = ref['harness']['stored']
    first = run_screen(req, remote=r, poll_s=0.01, local_dir=None, log=lambda m: None)
    entry = ledger.lookup(req.attempt_id)
    remote_req = json.loads(r.local(f'{req.out_dir}/request.json').read_text())
    assert entry and entry['request_digest'] == digest(remote_req) == first['request_digest']
    again = observe(req.run_id, req.attempt_id, remote=r, poll_s=0.01, local_dir=None, log=lambda m: None)
    assert again['valid'] and again['numerics_role'] == 'candidate'
    flipped = dict(remote_req, numerics_role='reference', reference_receipt=None)       # the R2 restart attack
    r.local(f'{req.out_dir}/request.json').write_text(json.dumps(flipped))
    with pytest.raises(ledger.LedgerError):
        observe(req.run_id, req.attempt_id, remote=r, poll_s=0.01, local_dir=None, log=lambda m: None)
    planted = dict(remote_req, attempt_id='a000000000000', numerics_role='reference', reference_receipt=None)
    r.write_json('WINH0', f'/data/infrarsi/runs/{req.run_id}/a000000000000/request.json', planted)
    with pytest.raises(ledger.LedgerError):                                             # never launched from here
        observe(req.run_id, 'a000000000000', remote=r, poll_s=0.01, local_dir=None, log=lambda m: None)


@pytest.mark.parametrize('field,value', [('request_digest', '0' * 64), ('numerics_role', 'candidate'),
                                         ('data_manifest', '/data/other/manifest.json'),
                                         ('effective_overlay_digest', '1' * 64), ('overlay_digest', '2' * 64),
                                         ('request_digest', None)])
def test_receipt_must_echo_the_trusted_request(tmp_path, field, value):
    req = make_request('D2')
    out = attempt_dir(tmp_path, req)
    rewrite_receipt(out, lambda rec: rec.update({field: value}) if value is not None else rec.pop(field))
    rec = validate_dir(out, req)
    assert not rec['valid'] and any(x.startswith(f'harness: identity.{field}') for x in rec['invalid_reasons'])


def test_role_flip_in_the_raw_receipt_is_caught_without_the_ledger(tmp_path):
    """Even when validate() is handed a request whose role was flipped, the trainer's echo of the launched request
    (request_digest, numerics_role) disagrees."""
    req = make_request('D2', numerics_role='candidate')
    out = attempt_dir(tmp_path, req)
    flipped = copy.deepcopy(req)
    flipped.numerics_role = 'reference'
    rec = validate_dir(out, flipped)
    assert not rec['valid'] and {'identity.request_digest', 'identity.numerics_role'} <= {
        c['name'] for c in rec['harness']['checks'] if not c['ok']}


# ---------------------------------------------------------------- 4. overlay identity

def _frozen(fsroot):
    import shutil
    snap = fsroot / 'pkg'
    shutil.copytree(codesync.PACKAGE_DIR, snap, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.md'))
    return snap


def test_code_cache_identity_includes_default_overlays(fsroot):
    snap = _frozen(fsroot)
    r = LocalRemote(fsroot)
    a = codesync.ensure_code(r, 'WINH0', package_dir=snap)
    assert a['code_root'].endswith(f'{a["code_digest"]}-ov{a["overlay_defaults_digest"][:16]}')
    sched = snap / 'runtime' / 'overlays' / 'schedule.py'
    sched.write_text(sched.read_text(encoding='utf-8') + '\n# release B: a different default hook file\n', encoding='utf-8')
    b = codesync.ensure_code(r, 'WINH0', package_dir=snap)
    assert b['code_digest'] == a['code_digest'] and b['code_root'] != a['code_root'] and b['uploaded']
    assert b['overlay_defaults']['schedule.py'] != a['overlay_defaults']['schedule.py']
    # a tree in B's slot whose manifest differs only in a default overlay is refused (no warning-only path)
    man = json.loads(r.local(b['code_root'] + '/MANIFEST.json').read_text())
    man['files']['runtime/overlays/host_sync.py'] = '0' * 64
    r.local(b['code_root'] + '/MANIFEST.json').write_text(json.dumps(man))
    with pytest.raises(RuntimeError, match='differs from this release'):
        codesync.ensure_code(r, 'WINH0', package_dir=snap)


def test_effective_overlay_digest_matches_the_runtime_loader(tmp_path):
    """The driver's merge rule agrees with runtime/overlay_loader: defaults replaced by request files of the same name."""
    from infrarsi.runtime import overlay_loader
    defaults = codesync.overlay_defaults(codesync.tree_manifest()['files'])
    assert set(defaults) == {f'{m}.py' for m in overlay_loader.HOOK_MODULES}
    assert codesync.effective_overlay_digest(defaults, None) == overlay_loader.load(None).digest
    src = (codesync.PACKAGE_DIR / 'runtime' / 'overlays' / 'host_sync.py').read_bytes() + b'\n# replaced by a request\n'
    (tmp_path / 'host_sync.py').write_bytes(src)
    spec = {'dir': str(tmp_path), 'files': {'host_sync.py': hashlib.sha256(src).hexdigest()}}
    assert codesync.effective_overlay_digest(defaults, spec['files']) == overlay_loader.load(spec).digest
    with pytest.raises(ValueError):
        codesync.effective_overlay_digest(defaults, {'sitecustomize.py': '0' * 64})


def test_run_screen_pins_the_effective_overlay(tmp_path, fsroot):
    r = indexed_remote(fsroot, launcher=sim_launcher())
    files = {'host_sync.py': (codesync.PACKAGE_DIR / 'runtime' / 'overlays' / 'host_sync.py').read_bytes() + b'\n# req\n'}
    fr = run_screen(make_request('D2', nodes=['WINH2']), remote=r, poll_s=0.01, local_dir=None, log=lambda m: None)
    req = make_request('D2', nodes=['WINH2'], numerics_role='candidate', reference_receipt=fr['harness']['stored'])
    rec = run_screen(req, remote=r, overlay_files=files, poll_s=0.01, local_dir=None, log=lambda m: None)
    uploaded = json.loads(r.local(f'{req.out_dir}/request.json').read_text())
    assert rec['valid'], rec['invalid_reasons']
    assert uploaded['effective_overlay_digest'] == req.effective_overlay_digest == rec['overlay_digest'] != fr['overlay_digest']
    assert uploaded['request_overlay_digest'] == digest([['host_sync.py', hashlib.sha256(files['host_sync.py']).hexdigest()]])
    calls = len(r.calls)
    with pytest.raises(ValueError):                                     # not an L1 hook module: refused before any lease
        run_screen(make_request('D2', nodes=['WINH2']), remote=r, overlay_files={'evil.py': b'x = 1\n'}, log=lambda m: None)
    assert not [c for c in r.calls[calls:] if c[1] in ('acquire_leases', 'launch_wrapper')]


def test_runtime_running_other_overlay_bytes_is_invalid(tmp_path):
    """A run that executed default-hook bytes other than the ones pinned for the request (stale cache, swapped file)."""
    req = make_request('D2')
    out = attempt_dir(tmp_path, req)
    stale = DUMMY_MOD.sha([['schedule.py', '0' * 64]])
    rewrite_receipt(out, lambda rec: rec.update(overlay_digest=stale))
    rec = validate_dir(out, req)
    assert not rec['valid'] and any(x.startswith('harness: identity.overlay_digest') for x in rec['invalid_reasons'])
    assert pin_overlay(req) == req.effective_overlay_digest
