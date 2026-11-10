"""Shared driver test helpers (conftest re-exports the fixtures): synthetic attempts (via the torch-free half of
tests/gpu/driver/dummy_entry.py), a LocalRemote whose launcher simulates the node wrappers, a fake packed-data
INDEX.json + manifests shaped like adapters/data/build_index.py output (run_screen fails closed without them), and
requests for frozen task specs.

Other builders' tests may load this module by path: a LocalRemote used with run_screen needs `fake_index(remote)`
once (the simulated attempts then read their data facts from that file system automatically).

Importing this module points the receipt HMAC key (INFRARSI_RECEIPT_KEY) and the launch ledger
(INFRARSI_LAUNCH_LEDGER) at session temp files unless they are already set, so tests never touch the PC's real key
or execution/ledger/launches.jsonl."""
from __future__ import annotations

import atexit
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import pytest

_SESSION = Path(tempfile.mkdtemp(prefix='iss'))
atexit.register(shutil.rmtree, _SESSION, True)
os.environ.setdefault('INFRARSI_RECEIPT_KEY', str(_SESSION / 'receipt_hmac.key'))
os.environ.setdefault('INFRARSI_LAUNCH_LEDGER', str(_SESSION / 'launches.jsonl'))

from infrarsi.contracts import RunRequest, TaskSpec, load_all_tasks
from infrarsi.harness import codesync, dataindex, remote_side
from infrarsi.harness.dataindex import INDEX_PATH, manifest_order_digest
from infrarsi.harness.remote import LocalRemote
from infrarsi.harness.validate import validate

REPO = Path(__file__).resolve().parents[3]
HF = Path(__file__).parent / 'fixtures' / 'hf_configs'
DUMMY = REPO / 'tests' / 'gpu' / 'driver' / 'dummy_entry.py'
PLAN_UPDATES = 48


def load_dummy():
    spec = importlib.util.spec_from_file_location('dummy_entry_sim', DUMMY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DUMMY_MOD = load_dummy()


# ---------------------------------------------------------------- packed-data INDEX in a LocalRemote file system

def fake_index(remote, task_ids=None) -> dict:
    """Write INDEX.json and one manifest per task into `remote` (a LocalRemote); returns {task: {plan: facts}} with
    the dummy's data facts {sup, order, ppu, manifest, manifest_sha256, order_digest}."""
    tasks = load_all_tasks()
    ids = list(task_ids) if task_ids else [t for t in tasks if t != 'F1']      # F1 is not in the real INDEX either
    index = {'schema': 'infrarsi-packed-index-v1', 'generated_utc': 'test', 'tasks': {}, 'pending': {}}
    facts = {}
    for tid in ids:
        t = tasks[tid]
        wl = t.raw['workload']
        ppu = int(wl['packs_per_update'])
        total = ppu * PLAN_UPDATES
        orders = {'A': list(reversed(range(total))), 'B': [(i * 7) % total for i in range(total)]}
        manifest = {'files': {'tokens.bin': {'sha256': 'ab' * 32, 'bytes': 1}},
                    'plans': {p: {'order': o} for p, o in orders.items()}}
        mpath = f'/data/infrarsi/datasets/packed/FAKE-{tid}__s{wl["seq_len"]}__p{total}/manifest.json'
        raw = json.dumps(manifest).encode()
        remote.put_bytes('WINH0', raw, mpath)
        phys = int(wl['physical_tokens_per_update'])
        sup = {p: [int(phys * (0.30 + 0.01 * ((u * 7 + (3 if p == 'B' else 0)) % 11))) for u in range(PLAN_UPDATES)]
               for p in orders}
        msha = hashlib.sha256(raw).hexdigest()
        od = {p: manifest_order_digest(manifest, p) for p in orders}
        index['tasks'][tid] = {'manifest': mpath, 'manifest_sha256': msha, 'task_digest': t.digest,
                               'physical_tokens_per_update': phys, 'packs_per_update': ppu,
                               'supervised_tokens_per_update': sup, 'data_order_digest': od}
        facts[tid] = {p: {'sup': sup[p], 'order': orders[p], 'ppu': ppu, 'manifest': mpath, 'manifest_sha256': msha,
                          'order_digest': od[p]} for p in orders}
    remote.put_bytes('WINH0', json.dumps(index).encode(), INDEX_PATH)
    return facts


_FAKE: dict = {}


def fake_fs() -> dict:
    """A module-wide LocalRemote holding the fake INDEX for simulated() (validation without run_screen)."""
    if not _FAKE:
        root = Path(tempfile.mkdtemp(prefix='isf'))
        atexit.register(shutil.rmtree, root, True)
        r = LocalRemote(root)
        _FAKE.update(remote=r, root=root, facts=fake_index(r), expected={})
    return _FAKE


def expected_data(request: RunRequest) -> dict:
    """What run_screen's resolve_data returns for `request` (pins request.data_manifest like it)."""
    if dataindex.is_synthetic(request.task):
        return {'skipped': 'synthetic data'}
    fs = fake_fs()
    key = (TaskSpec(raw=request.task).digest, request.input_plan)
    if key not in fs['expected']:
        fs['expected'][key] = dataindex.load(fs['remote'], ['WINH0'], request.task, request.input_plan)
    exp = fs['expected'][key]
    request.data_manifest = exp['manifest']
    return exp


def data_facts(request: RunRequest) -> dict | None:
    if dataindex.is_synthetic(request.task):
        return None
    return fake_fs()['facts'][request.task['id']][request.input_plan]


# ---------------------------------------------------------------- requests and simulated attempts

def make_request(task_id='D2', purpose='screen', plan='A', config=None, nodes=None, **kw) -> RunRequest:
    """Fixture runs model the task's numerics reference (FR-like) unless a test passes numerics_role='candidate'
    (the production default, which makes a run without reference_receipt unverified)."""
    kw.setdefault('numerics_role', 'reference')
    t = TaskSpec.load(task_id)
    gpn = t.gpus if t.nodes == 1 else t.gpus // t.nodes
    nodes = nodes or [f'WINH{i}' for i in range(t.nodes)]
    r = RunRequest(run_id=kw.pop('run_id', f'T-{task_id}'), purpose=purpose, task=t.raw,
                   config=config if config is not None else {'tensor_model_parallel_size': 1}, nodes=nodes,
                   gpus_per_node=gpn, input_plan=plan, **kw)
    r.code_digest = 'c' * 64
    r.code_root = '/data/infrarsi/code/' + 'c' * 64
    return r


_OVERLAY_DEFAULTS: dict = {}


def pin_overlay(request: RunRequest) -> str:
    """What run_screen pins: the effective overlay digest of this repository's default hook files and the request's
    overlay files (the dummy reports the same set as its runtime overlay_digest)."""
    if not _OVERLAY_DEFAULTS:
        _OVERLAY_DEFAULTS.update(codesync.overlay_defaults(codesync.tree_manifest()['files']))
    if request.effective_overlay_digest is None:
        request.effective_overlay_digest = codesync.effective_overlay_digest(_OVERLAY_DEFAULTS,
                                                                             (request.overlay or {}).get('files'))
    return request.effective_overlay_digest


def attempt_dir(tmp_path: Path, request: RunRequest, **sim) -> Path:
    """Simulate an attempt of `request` into tmp_path/<attempt_id> with the fake INDEX's data facts."""
    expected_data(request)
    pin_overlay(request)
    sim.setdefault('data', data_facts(request))
    out = Path(tmp_path) / request.attempt_id
    DUMMY_MOD.simulate_attempt(request.to_dict(), str(out), **sim)
    return out


def validate_dir(out: Path, request: RunRequest, reference=None, stop=None) -> dict:
    return validate(remote_side.collect_run(str(out), len(request.nodes)), request, reference, stop=stop,
                    data=expected_data(request))


def simulated(tmp_path: Path, request: RunRequest, reference=None, stop=None, **sim) -> dict:
    """Validated receipt for a simulated attempt of `request` (no remote involved)."""
    return validate_dir(attempt_dir(tmp_path, request, **sim), request, reference, stop=stop)


def rewrite_receipt(out: Path, fn) -> None:
    p = Path(out) / 'receipt.json'
    r = json.loads(p.read_text(encoding='utf-8'))
    fn(r)
    p.write_text(json.dumps(r), encoding='utf-8')


def rewrite_signals(out: Path, rank: int, fn, world: int) -> None:
    """Rewrite one rank's signal rows and re-digest them into the receipt (the trainer digesting forged rows)."""
    p = Path(out) / 'signals' / f'rank{rank}.jsonl'
    rows = [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
    p.write_text(''.join(json.dumps(fn(r), sort_keys=True) + '\n' for r in rows), encoding='utf-8', newline='\n')
    rewrite_receipt(out, lambda r: r.update(signals_digest=DUMMY_MOD.signals_digest(str(out), world)))


def sim_launcher(behaviour=None, threaded=False):
    """LocalRemote launcher: once the last node wrapper is 'launched', write the attempt's outputs (the default
    behaviour reads the data facts from the LocalRemote's INDEX.json, written by fake_index)."""
    def launcher(k, kw, remote):
        spec = json.loads(remote.local(kw['spec_path']).read_text(encoding='utf-8'))
        if spec['node_index'] == spec['nnodes'] - 1:
            out = remote.local(spec['out_dir'])
            req = json.loads((out / 'request.json').read_text(encoding='utf-8'))
            fn = behaviour or (lambda r, o: DUMMY_MOD.simulate_attempt(r, o))
            if threaded:
                threading.Thread(target=fn, args=(req, str(out)), daemon=True).start()
            else:
                fn(req, str(out))
        return {'pid': 4000 + k, 'hostname': f'fake-winh{k}', 'utc': remote_side.utc()}
    return launcher


def wait_for_stop_then(stop_at=5, timeout=20):
    def behaviour(req, out):
        p = Path(out) / 'STOP.json'
        end = time.time() + timeout
        while not p.exists() and time.time() < end:
            time.sleep(0.02)
        DUMMY_MOD.simulate_attempt(req, out, stop_at=stop_at)
    return behaviour


def indexed_remote(root, **kw) -> LocalRemote:
    """LocalRemote with the fake packed-data INDEX already written."""
    r = LocalRemote(root, **kw)
    fake_index(r)
    return r


@pytest.fixture
def fsroot():
    """Short root for LocalRemote (Windows MAX_PATH: code trees live under a 64-hex digest directory)."""
    d = Path(tempfile.mkdtemp(prefix='ist'))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def local_remote(fsroot):
    return indexed_remote(fsroot, launcher=sim_launcher())


@pytest.fixture
def dummy():
    return DUMMY_MOD
