"""run_screen(request) -> validated receipt: sync code, check occupancy, lease, launch 1-4 nodes, watch, collect.

Blocking. Stops the attempt (STOP.json, then the node wrappers' grace/kill) on: a local stop request
(execution/queue/stop/<attempt_id|run_id>.json), a caller stop_check(), the wall cap, a stale heartbeat, a startup
hang or a lost node wrapper. External STOP.json files are honoured as well. Leases are released only when every
node reported its exit and its process group is gone.

Fail closed before any lease: the numerics reference (path or dict) must pass validate.reference_problems, and every
non-synthetic task must resolve its INDEX.json entry and manifest bytes (sha256 and order digest verified) and pin
request.data_manifest; otherwise run_screen raises and nothing is leased or launched.

Trust boundary (Codex R2): the request is final before upload - data_manifest, code identity and
effective_overlay_digest (the default hook files of the verified code tree merged with the request's overlay files)
are pinned - and its digest goes into the local launch ledger (harness/ledger.py) before request.json is written;
observe() refuses a remote request.json that differs from the ledger. Validated receipts are HMAC-signed
(harness/signing.py) after every harness field is final (leases released, storage path), and copies on /data are
trusted only through that signature.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

from infrarsi.contracts import EXECUTION_DIR, RemotePaths, RunRequest, TaskSpec, WINH_NODES, digest, utc_now

from . import codesync, dataindex, launch as L, leases, ledger, signing, validate as V
from .remote import Remote, RemoteError, default_remote, node_name

STOP_DIR = EXECUTION_DIR / 'queue' / 'stop'
LOCAL_RECEIPTS = EXECUTION_DIR / 'receipts'
PARTIAL_LAUNCH_SETTLE_S = 20.0


def _log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def request_stop(out_dir: str, reason: str, *, by: str = 'driver', kind: str = 'request', remote: Remote | None = None,
                 node: str | None = None) -> dict:
    """Write runs/<run>/<attempt>/STOP.json; the training process exits at its next update boundary."""
    remote = remote or default_remote()
    payload = {'schema': 'infrarsi-stop-v1', 'reason': reason, 'kind': kind, 'requested_by': by, 'utc': utc_now()}
    return remote.call(node or 'WINH0', 'write_stop', out_dir=out_dir, payload=payload)


def local_stop_request(request: RunRequest, stop_dir: Path = STOP_DIR) -> dict | None:
    for name in (request.attempt_id, request.run_id):
        p = Path(stop_dir) / f'{name}.json'
        if p.exists():
            try:
                return json.loads(p.read_text(encoding='utf-8'))
            except ValueError:
                return {'reason': f'unreadable stop request {p.name}'}
    return None


class ReferenceUnusable(ValueError):
    """The numerics reference fails validate.reference_problems (nothing is launched)."""


def check_reference(reference: dict, request: RunRequest, where: str = 'reference') -> dict:
    """The one fail-closed reference loader, applied to a reference for `request` (probes ignore references)."""
    if request.purpose == 'probe':
        return reference
    problems = V.reference_problems(reference, task=request.task, plan=request.input_plan, steps=request.steps,
                                    window=request.window)
    h = reference.get('harness') if isinstance(reference, dict) else None
    stored = h.get('stored') if isinstance(h, dict) else None
    if stored and request.reference_receipt and stored != request.reference_receipt:
        problems.append(f'reference dict was stored at {stored}, the request names {request.reference_receipt}')
    if problems:
        raise ReferenceUnusable(f'{where} unusable for {request.run_id}/{request.attempt_id}: ' + '; '.join(problems[:8]))
    return reference


def load_reference(remote: Remote, node, path: str | None, request: RunRequest | None = None) -> dict | None:
    if not path:
        return None
    ref = remote.read_json(node, path)
    if ref is None:
        raise FileNotFoundError(f'reference receipt {path} not found')
    return check_reference(ref, request, where=path) if request is not None else ref


def resolve_data(remote, nodes, request: RunRequest) -> dict:
    """INDEX.json facts for the task and plan; pins request.data_manifest to the manifest the driver validated so the
    trainer reads exactly that file. Fails closed: any missing or inconsistent fact raises (nothing is leased)."""
    if dataindex.is_synthetic(request.task):
        if request.data_manifest:
            raise ValueError(f'synthetic task {request.task.get("id")} cannot pin a data manifest')
        return {'skipped': 'synthetic data'}
    data = dataindex.load(remote, nodes, request.task, request.input_plan)      # raises when anything is missing
    if data.get('manifest_sha_ok') is not True:
        raise RuntimeError(f'{data["manifest"]}: sha256 differs from INDEX.json; not launching')
    if data.get('manifest_order_digest_ok') is not True:
        raise RuntimeError(f'{data["manifest"]}: plan {request.input_plan} order digest differs from INDEX.json; '
                           'not launching')
    if data.get('task_digest_ok') is not True:
        raise RuntimeError(f'INDEX.json entry of {request.task.get("id")} was built from another task spec; not launching')
    if request.data_manifest and request.data_manifest != data['manifest']:
        raise ValueError(f'request.data_manifest {request.data_manifest} is not the INDEX.json manifest {data["manifest"]}')
    request.data_manifest = data['manifest']
    return data


def _check_request(request: RunRequest) -> TaskSpec:
    task = TaskSpec(raw=request.task)
    if not request.nodes or len(set(request.nodes)) != len(request.nodes):
        raise ValueError(f'bad node list {request.nodes}')
    if any(n not in WINH_NODES for n in request.nodes):
        raise ValueError(f'nodes must be WINH0-7: {request.nodes}')
    if len(request.nodes) > 8:
        raise ValueError('at most the eight WINH nodes')
    if not 1 <= request.gpus_per_node <= 8:
        raise ValueError('gpus_per_node must be 1..8')
    if request.gpus_per_node < 8 and len(request.nodes) > 1:
        raise ValueError('partial nodes only for single-node tasks')
    if request.world_size != task.gpus:
        raise ValueError(f'world size {request.world_size} != task {task.id} gpus {task.gpus}')
    L.check_env((request.config or {}).get('env'))
    return task


def run_screen(request: RunRequest, *, remote: Remote | None = None, entry: tuple[str, str] | None = None,
               reference: dict | None = None, overlay_files: dict[str, bytes] | None = None, poll_s: float = 15.0,
               heartbeat_timeout_s: float = 1800.0, first_heartbeat_timeout_s: float = 1200.0,
               stop_grace_s: float = 300.0, node_lost_s: float = 300.0, stop_check: Callable[[], str | None] | None = None,
               stop_dir: Path = STOP_DIR, local_dir: Path | None = LOCAL_RECEIPTS, log: Callable[[str], None] = _log,
               extra_files: dict[str, bytes] | None = None) -> dict:
    """Launch one attempt and block until it ends; returns the validated receipt (also stored remotely/locally).

    entry: ('module', name) or ('script', local_path) - default the generic training entry. A local script is
    uploaded into <out_dir>/entry/. Raises leases.OccupiedError (nothing launched) if a node is busy.
    """
    remote = remote or default_remote()
    _check_request(request)
    nodes, node0 = list(request.nodes), request.nodes[0]
    if reference is None and request.reference_receipt:
        reference = load_reference(remote, node0, request.reference_receipt, request)
    elif reference is not None:
        check_reference(reference, request)
    data = resolve_data(remote, nodes, request)                       # before request.json is written
    cs = codesync.ensure_code(remote, node0)
    request.code_root, request.code_digest = cs['code_root'], cs['code_digest']
    overlay_shas = ({name: hashlib.sha256(b).hexdigest() for name, b in sorted(overlay_files.items())} if overlay_files
                    else ((request.overlay or {}).get('files') or None))
    effective = codesync.effective_overlay_digest(cs['overlay_defaults'], overlay_shas)    # raises on a non-hook file
    if request.effective_overlay_digest and request.effective_overlay_digest != effective:
        raise ValueError(f'request.effective_overlay_digest {request.effective_overlay_digest} is not the effective '
                         f'overlay set {effective} of this code tree and overlay')
    request.effective_overlay_digest = effective
    log(f'{request.run_id}/{request.attempt_id}: code {cs["code_digest"][:12]} uploaded={cs["uploaded"]}')
    occ = leases.check_free(remote, nodes, run_id=request.run_id)
    held = all((occ[node_name(n)]['lease'] or {}).get('scope') == 'run' for n in nodes)   # dispatcher's run lease
    if not held:
        leases.acquire(remote, nodes, leases.make_lease(request.run_id, request.attempt_id, nodes, request.purpose))
    log(f'leases held on {nodes} ({"run" if held else "attempt"} scope)')
    launched = False
    try:
        out = request.out_dir
        remote.call(node0, 'makedirs', paths=[out, f'{out}/logs', f'{out}/signals', f'{out}/launch'])
        if overlay_files:
            remote.call(node0, 'makedirs', paths=[f'{out}/overlay'])
            for name, blob in sorted(overlay_files.items()):
                remote.put_bytes(node0, blob, f'{out}/overlay/{name}')
            request.overlay = {'dir': f'{out}/overlay', 'files': overlay_shas}
        for name, data in (extra_files or {}).items():
            remote.put_bytes(node0, data, f'{out}/{name}')
        if entry and entry[0] == 'script':
            local = Path(entry[1])
            remote.call(node0, 'makedirs', paths=[f'{out}/entry'])
            remote.put_bytes(node0, local.read_bytes(), f'{out}/entry/{local.name}')
            entry = ('script', f'{out}/entry/{local.name}')
        ledger.record_launch(request)                                  # trusted local record before the upload
        remote.write_json(node0, f'{out}/request.json', request.to_dict(), exclusive=True)
        master_addr = leases.internal_ip({'ips': occ[node_name(node0)]['ips'], 'hostname': node0})
        port = remote.call(node0, 'first_free_port', candidates=L.pick_ports(request.attempt_id))
        if port is None:
            raise RuntimeError('no free master port on ' + node0)
        leases.check_free(remote, nodes, attempt_id=request.attempt_id, run_id=request.run_id)   # re-check before launch
        deadline = float(request.wall_seconds_cap) + stop_grace_s + 600.0         # wrapper backstop after driver STOP
        launches, initial_stop = [], None
        for i, node in enumerate(nodes):
            spec = L.node_spec(request=request, node_rank=i, master_addr=master_addr, master_port=port,
                               code_root=request.code_root, entry=entry, deadline_s=deadline, stop_grace_s=stop_grace_s)
            spec_path = f'{out}/launch/node{i}.json'
            remote.write_json(node0, spec_path, spec)
            try:
                res = remote.call(node, 'launch_wrapper', python=L.WRAPPER_PYTHON,
                                  wrapper=f'{request.code_root}/infrarsi/harness/node_wrapper.py', spec_path=spec_path,
                                  log_path=f'{out}/logs/wrapper{i}.log')
            except Exception as e:                                         # noqa: BLE001
                if not launched:
                    raise
                initial_stop = _abort_partial_launch(request, remote, i, e, log)
                break
            launched = True
            launches.append(dict(res, node=node, node_index=i))
        log(f'launched {len(launches)}/{len(nodes)} node(s), master {master_addr}:{port}, out {out}')
        info = {'code_sync': cs, 'master_addr': master_addr, 'master_port': port, 'launches': launches,
                'entry': entry or ('module', L.ENTRY_MODULE), 'launched_utc': utc_now(), 'run_scope_lease': held}
        return _watch_and_finish(request, remote, reference, info, poll_s=poll_s, heartbeat_timeout_s=heartbeat_timeout_s,
                                 first_heartbeat_timeout_s=first_heartbeat_timeout_s, stop_grace_s=stop_grace_s,
                                 node_lost_s=node_lost_s, stop_check=stop_check, stop_dir=stop_dir,
                                 local_dir=local_dir, log=log, stop=initial_stop, data=data)
    finally:
        if not launched and not held:
            leases.release(remote, nodes, request.attempt_id)


def _abort_partial_launch(request, remote, failed_index, error, log) -> dict:
    """Some wrappers run, node `failed_index` did not start: STOP the attempt and give the missing nodes an exit
    record (only where no wrapper pid file appeared) so peers and the driver finish instead of waiting on rendezvous."""
    out = request.out_dir
    reason = f'partial launch failure on {request.nodes[failed_index]}: {error!r}'[:500]
    log('STOP launch_failed: ' + reason)
    request_stop(out, reason, by='harness.run_screen', kind='launch_failed', remote=remote, node=request.nodes[0])
    time.sleep(PARTIAL_LAUNCH_SETTLE_S)              # a wrapper whose launch call timed out may still be starting
    snap = _call_any(remote, request.nodes, 'run_snapshot', out_dir=out, nnodes=len(request.nodes), log_tail_bytes=0)
    for k in range(failed_index, len(request.nodes)):
        n = snap['nodes'][k]
        if n['exit'] is None and n['pid'] is None:
            try:
                remote.write_json(request.nodes[0], f'{out}/exit_node{k}.json', {
                    'schema': 'infrarsi-node-exit-v1', 'node_index': k, 'exit_code': 98, 'reason': 'launch_failed',
                    'error': {'message': reason}, 'started_utc': None, 'finished_utc': utc_now(), 'wall_s': 0.0,
                    'killed_by_wrapper': False, 'synthesized_by_driver': True,
                    'release': {'gpu_apps_after': [], 'process_group_gone': True}}, exclusive=True)
            except RemoteError:
                pass                                 # the wrapper wrote its own exit record meanwhile
    return {'kind': 'launch_failed', 'reason': reason, 'utc': utc_now(), 't': 0.0}


def _call_any(remote, nodes, func, **kw):
    """Read-only calls on the shared GPFS: try the run's nodes in order, then the rest of WINH."""
    last = None
    for n in list(nodes) + [x for x in WINH_NODES if x not in nodes]:
        try:
            return remote.call(n, func, **kw)
        except RemoteError as e:
            last = e
    raise last


def observe(run_id: str, attempt_id: str, *, remote: Remote | None = None, reference: dict | None = None,
            **kw) -> dict:
    """Re-attach to an attempt launched earlier (e.g. after a driver restart): watch, collect, validate, release."""
    remote = remote or default_remote()
    d = remote.read_json('WINH0', f'{RemotePaths.runs}/{run_id}/{attempt_id}/request.json')
    if d is None:
        raise FileNotFoundError(f'no request.json for {run_id}/{attempt_id}')
    ledger.check_remote_request(attempt_id, d)          # refuses a request.json edited after launch (e.g. a role flip)
    for k in ('world_size', 'out_dir', 'config_digest', 'task_digest', 'request_overlay_digest'):
        d.pop(k, None)
    request = RunRequest.from_dict(d)
    if ledger.request_digest(request) != ledger.lookup(attempt_id)['request_digest']:
        raise ledger.LedgerError(f'{attempt_id}: the reconstructed request does not reproduce the launched digest')
    if reference is None and request.reference_receipt:
        reference = load_reference(remote, request.nodes[0], request.reference_receipt, request)
    elif reference is not None:
        check_reference(reference, request)
    return _watch_and_finish(request, remote, reference, {'observed_utc': utc_now()}, **kw)


def _watch_and_finish(request, remote, reference, info, *, poll_s=15.0, heartbeat_timeout_s=1800.0,
                      first_heartbeat_timeout_s=1200.0, stop_grace_s=300.0, node_lost_s=300.0, stop_check=None,
                      stop_dir=STOP_DIR, local_dir=LOCAL_RECEIPTS, log=_log, stop=None, data=None) -> dict:
    nodes, node0, out = list(request.nodes), request.nodes[0], request.out_dir
    t0 = time.monotonic()
    last_step, errors = None, 0
    while True:
        try:
            snap = _call_any(remote, nodes, 'run_snapshot', out_dir=out, nnodes=len(nodes), log_tail_bytes=0)
            errors = 0
        except RemoteError as e:
            errors += 1
            log(f'snapshot failed ({errors}): {e}')
            if errors >= 20:
                raise
            time.sleep(poll_s)
            continue
        if all(n['exit'] is not None for n in snap['nodes']):
            break
        el = time.monotonic() - t0
        hb = snap.get('heartbeat') or {}
        if hb.get('step') != last_step and hb.get('step') is not None:
            last_step = hb.get('step')
            log(f'{request.run_id}/{request.attempt_id}: phase {hb.get("phase")} step {last_step}')
        if stop is None:
            why = None
            if snap.get('stop'):
                why = ('external', (snap['stop'] or {}).get('reason', 'STOP.json present'))
            elif (req := local_stop_request(request, stop_dir)) is not None:
                why = ('request', req.get('reason', 'local stop request'))
            elif stop_check is not None and (r := stop_check()):
                why = ('caller', r)
            elif el > float(request.wall_seconds_cap):
                why = ('wall_cap', f'wall cap {request.wall_seconds_cap}s exceeded')
            elif snap.get('heartbeat_age_s') is None and el > first_heartbeat_timeout_s:
                why = ('startup', f'no heartbeat after {el:.0f}s')
            elif (snap.get('heartbeat_age_s') or 0) > heartbeat_timeout_s:
                why = ('heartbeat', f'heartbeat stale {snap["heartbeat_age_s"]:.0f}s')
            elif el > 2 * node_lost_s and any(n['exit'] is None and (n['alive_age_s'] is None or n['alive_age_s'] > node_lost_s)
                                              for n in snap['nodes']):
                why = ('node_lost', 'a node wrapper stopped reporting')
            if why:
                stop = {'kind': why[0], 'reason': why[1], 'utc': utc_now(), 't': el}
                if why[0] != 'external':
                    request_stop(out, why[1], by='harness.run_screen', kind=why[0], remote=remote, node=node0)
                log(f'STOP {why[0]}: {why[1]}')
        elif el - stop['t'] > stop_grace_s + 900:
            log('nodes did not report exit after STOP + grace; collecting what exists')
            stop['unresponsive'] = True
            break
        time.sleep(poll_s)
    coll = _call_any(remote, nodes, 'collect_run', out_dir=out, nnodes=len(nodes), timeout=600)
    info = dict(info, harness_code_digest=codesync.code_digest(), polled_s=round(time.monotonic() - t0, 1))
    if data is None:                                                   # observe(): re-read, but stay on the run's manifest
        try:
            data = dataindex.load(remote, nodes, request.task, request.input_plan)
            if not dataindex.is_synthetic(request.task) and data.get('manifest') != request.data_manifest:
                data = {'error': f'INDEX.json now points to {data.get("manifest")}; the run read {request.data_manifest}'}
        except Exception as e:                                         # noqa: BLE001 - recorded; data checks fail closed
            data = {'error': repr(e)[:300]}
    validated = V.validate(coll, request, reference, stop=stop, harness_info=info, data=data)
    exits = [n.get('exit') for n in coll.get('nodes', [])]
    if info.get('run_scope_lease'):
        validated['harness']['leases_released'] = []                    # the dispatcher releases its run lease
    elif all(e is not None and (e.get('release') or {}).get('process_group_gone') for e in exits):
        rel = leases.release(remote, nodes, request.attempt_id)
        validated['harness']['leases_released'] = rel['released']
    else:
        validated['harness']['leases_released'] = []
        log(f'LEASES KEPT on {nodes}: not every node confirmed its process group is gone')
    store(validated, remote=remote, node=node0, local_dir=local_dir)
    log(f'{request.run_id}/{request.attempt_id}: valid={validated["valid"]} failure={validated["harness"]["failure_kind"]} '
        f'step_time={(validated.get("step_time") or {}).get("mean_window_max_rank_s")}')
    return validated


def store(validated: dict, *, remote: Remote, node, local_dir: Path | None = LOCAL_RECEIPTS) -> str:
    """Finalize (storage path inside the harness block), sign, then write the remote copy and the local copy."""
    att = validated['attempt_id']
    path = f'{RemotePaths.receipts}/{att}.json'
    for attempt in range(2):
        validated['harness']['stored'] = path
        signing.sign(validated)
        try:
            remote.write_json(node, path, validated, exclusive=True)
            break
        except RemoteError as e:
            if 'FileExistsError' not in str(e) or attempt:
                raise
            path = f'{RemotePaths.receipts}/{att}.{int(time.time())}.json'
    if local_dir is not None:
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / f'{att}.json').write_text(json.dumps(validated, indent=1, sort_keys=True), encoding='utf-8')
    return path


def recheck_against(validated: dict, reference: dict) -> dict:
    """Post-hoc numerics/invariant comparison against a reference produced later (e.g. FR plan-B scoring run).

    Fail closed: the receipt being re-checked must carry a valid harness signature (it is re-signed afterwards), the
    task spec must still hash to its task_digest, and the reference must pass validate.reference_problems for the
    receipt's task, plan, steps and window (ValueError otherwise). The new reference checks replace every earlier
    numerics/reference check; all other harness checks keep their result."""
    if not signing.verify(validated):
        raise ValueError('the receipt to re-check is not harness-signed or was altered')
    task = TaskSpec.load(validated['task_id'])
    if digest(task.raw) != validated.get('task_digest'):
        raise ValueError(f'task spec {validated["task_id"]} changed since the receipt was validated')
    problems = V.reference_problems(reference, task=task.raw, plan=validated.get('input_plan'), steps=validated.get('steps'),
                                    window=validated.get('window'))
    if problems:
        raise ReferenceUnusable('reference unusable for recheck: ' + '; '.join(problems[:8]))
    num = task.raw.get('numerics') or {}
    out = json.loads(json.dumps(validated))
    role = out.get('numerics_role') or 'candidate'
    c = V.Checks()
    tier = V.numerics_tier_check(task.raw)                          # AMENDMENT-002: the tier's frozen gates
    tol = (tier['table'] or {}).get('tolerance') or num.get('tolerance') or {}
    numerics = V.compare_numerics(out['per_step'], V.ref_steps(reference), tol, out.get('window'),
                                  num.get('numerics_class_allowed', 'E1'), role=role, steps=out.get('steps'))
    numerics.update(reference_attempt_id=reference.get('attempt_id'), reference_plan=reference.get('input_plan'), role=role,
                    self_check=V.self_check(out['per_step']), tier=tier['tier'],  # self_check.* check itself is kept
                    tolerance=tol)
    if tier['problems']:
        numerics['tier_problems'] = tier['problems'][:8]
    c.add('ref.usable', True, '')
    c.add('numerics.tier', not tier['problems'], f'tier {tier["tier"]}: ' + '; '.join(tier['problems'][:6]))
    c.add('numerics.class', numerics.get('allowed_ok', False), f'{numerics.get("class")}: {numerics.get("violations")}')
    V.reference_invariants(c, out.get('invariants') or {}, out['per_step'], reference, out.get('purpose'),
                           deterministic=V.deterministic_lifecycle(out, out.get('purpose')))
    out['harness']['numerics'] = numerics
    out['harness']['checks'] = [x for x in out['harness']['checks'] if not x['name'].startswith(('ref.', 'numerics.'))] + c.items
    reasons = [r for r in out['invalid_reasons'] if not r.startswith(('harness: ref.', 'harness: numerics.'))]
    out['invalid_reasons'] = reasons + [f'harness: {x}' for x in c.invalid]
    base_invalid = [x for x in out['harness']['checks'] if not x['ok'] and x['severity'] == 'invalid']
    out['valid'] = bool(out['harness']['reported'].get('valid')) and not base_invalid
    out['harness']['rechecked_utc'] = utc_now()
    return signing.sign(out)
