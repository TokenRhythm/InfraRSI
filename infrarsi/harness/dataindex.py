"""Expected data invariants of an attempt from /data/infrarsi/datasets/packed/INDEX.json (T-data's build_index).

Fail closed for every non-synthetic task: load() raises unless INDEX.json has a complete entry for the task and plan
(manifest path and sha256, physical tokens per update, packs per update, the plan's supervised-token vector and its
data_order_digest) and the manifest bytes could be read from a node; resolve_data (runner) calls it before any lease.
The manifest carries the plan permutations, from which the harness recomputes:
  - the INDEX order digest (adapters/data PackedTHDDataset.order_digest) - the only accepted
    invariants.data_order_digest (INTERFACES "Data invariants");
  - the consumed-pack digests of the completed prefix: invariants.data_consumed_digest = digest of the sorted global
    pack set of every completed update (runtime/invariants.data_order_digest), and
    invariants.data_consumed_per_update[k] = digest(sorted pack ids of update k+1). Both are mandatory for every
    non-lifecycle run and checked when present on a lifecycle run;
  - per-update supervised tokens: the INDEX vector must cover the whole plan and match every completed update, and the
    receipt's own vector invariants.supervised_tokens_per_update[plan] must list exactly the completed updates;
  - invariants.supervised_tokens_total = the INDEX sum over the completed updates (also the lifecycle's own expected
    total: a lifecycle attempt runs save_at + resume_steps updates, more than its plan-B score reference).
Synthetic tasks (data.synthetic or a D-SYN key) skip INDEX checks.
"""
from __future__ import annotations

import hashlib
import json
import re

from infrarsi.contracts import RemotePaths, WINH_NODES, digest

INDEX_PATH = f'{RemotePaths.datasets_packed}/INDEX.json'
_HEX64 = re.compile(r'[0-9a-f]{64}')


def is_synthetic(task: dict) -> bool:
    data = task.get('data') or {}
    return bool(data.get('synthetic')) or str(data.get('key', '')).startswith('D-SYN')


def update_packs(order: list[int], ppu: int, updates: int) -> list[list[int]]:
    return [sorted(int(order[i]) for i in range(u * ppu, (u + 1) * ppu)) for u in range(updates)]


def update_packs_digest(order: list[int], ppu: int, updates: int) -> str:
    """invariants.data_consumed_digest of the first `updates` updates of a plan order."""
    return digest(update_packs(order, ppu, updates))


def per_update_digests(order: list[int], ppu: int, updates: int) -> list[str]:
    """invariants.data_consumed_per_update of the first `updates` updates of a plan order."""
    return [digest(u) for u in update_packs(order, ppu, updates)]


def manifest_order_digest(manifest: dict, plan: str) -> str:
    return digest({'files': {n: f['sha256'] for n, f in sorted(manifest['files'].items())}, 'plan': plan,
                   'order': [int(x) for x in manifest['plans'][plan]['order']]})


def _need(cond, msg):
    if not cond:
        raise ValueError(f'INDEX.json: {msg}')


def load(remote, nodes, task: dict, plan: str, call_any=None) -> dict:
    """Expected data facts for (task, plan); {'skipped': reason} for synthetic data; raises if anything is missing."""
    if is_synthetic(task):
        return {'skipped': 'synthetic data'}
    from .runner import _call_any
    call_any = call_any or _call_any
    index = call_any(remote, nodes, 'read_json', path=INDEX_PATH)
    if not isinstance(index, dict):
        raise FileNotFoundError(f'{INDEX_PATH} missing or unreadable')
    e = (index.get('tasks') or {}).get(task['id'])
    if not isinstance(e, dict):
        raise LookupError(f'{task["id"]} not in INDEX.json ({(index.get("pending") or {}).get(task["id"])})')
    sup = (e.get('supervised_tokens_per_update') or {}).get(plan)
    order_digest = (e.get('data_order_digest') or {}).get(plan)
    _need(isinstance(e.get('manifest'), str) and e['manifest'].startswith('/'), f'{task["id"]}: manifest path missing')
    _need(_HEX64.fullmatch(str(e.get('manifest_sha256') or '')), f'{task["id"]}: manifest_sha256 missing')
    _need(isinstance(e.get('physical_tokens_per_update'), int) and isinstance(e.get('packs_per_update'), int)
          and e['packs_per_update'] > 0, f'{task["id"]}: physical_tokens_per_update / packs_per_update missing')
    _need(isinstance(sup, list) and sup and all(isinstance(x, int) and not isinstance(x, bool) for x in sup),
          f'{task["id"]}: supervised_tokens_per_update[{plan}] missing or malformed')
    _need(_HEX64.fullmatch(str(order_digest or '')), f'{task["id"]}: data_order_digest[{plan}] missing')
    out = {'index_generated_utc': index.get('generated_utc'), 'manifest': e['manifest'], 'plan': plan,
           'manifest_sha256': e['manifest_sha256'], 'task_digest_ok': e.get('task_digest') == digest(task),
           'physical_tokens_per_update': e['physical_tokens_per_update'], 'supervised_tokens_per_update': sup,
           'index_order_digest': order_digest, 'packs_per_update': e['packs_per_update']}
    raw = None
    for n in list(nodes) + [x for x in WINH_NODES if x not in nodes]:   # shared GPFS: any node can serve the bytes
        try:
            raw = remote.get_bytes(n, e['manifest'])
        except Exception:                                       # noqa: BLE001 - try the next node
            raw = None
        if raw is not None:
            break
    if raw is None:
        raise FileNotFoundError(f'manifest {e["manifest"]} could not be read from any node')
    out['manifest_sha_ok'] = hashlib.sha256(raw).hexdigest() == e['manifest_sha256']
    m = json.loads(raw)
    order = [int(x) for x in m['plans'][plan]['order']]
    _need(order and len(order) % e['packs_per_update'] == 0, f'{task["id"]}: plan {plan} order is not whole updates')
    out['order'] = order
    out['plan_updates'] = len(order) // e['packs_per_update']
    out['manifest_order_digest_ok'] = manifest_order_digest(m, plan) == order_digest
    return out


def check(c, receipt: dict, derived: dict, request, expected: dict | None) -> None:
    """Add the data checks to a validate.Checks; every check is mandatory for a non-synthetic task."""
    if is_synthetic(request.task):
        return
    if not expected or 'error' in expected or expected.get('skipped'):
        why = (expected or {}).get('error') or (expected or {}).get('skipped') or 'not read'
        c.add('data.index', False, f'INDEX.json facts unavailable ({why}): data invariants unverifiable')
        return
    inv = receipt.get('invariants') or {}
    per = derived['per_step']
    done = derived['steps_completed']
    plan = request.input_plan
    c.add('data.index_task', expected.get('task_digest_ok') is True, 'INDEX.json entry was built from another task spec')
    c.add('data.manifest_sha256', expected.get('manifest_sha_ok') is True, f'{expected["manifest"]} differs from INDEX.json')
    c.add('data.index_order_digest', expected.get('manifest_order_digest_ok') is True,
          'plan order digest in INDEX.json does not match the manifest')
    c.add('data.request_manifest', request.data_manifest == expected['manifest'],
          f'request pinned {request.data_manifest!r}, INDEX.json names {expected["manifest"]}')
    used = inv.get('data') if isinstance(inv.get('data'), dict) else {}
    c.add('data.manifest_used', used.get('source') == expected['manifest'],
          f'trainer read {used.get("source")!r}, the driver validated {expected["manifest"]}')
    c.add('data.manifest_digest', used.get('manifest_digest') == expected['manifest_sha256'],
          f'trainer manifest digest {used.get("manifest_digest")!r} != INDEX.json {expected["manifest_sha256"]}')
    phys = expected['physical_tokens_per_update']
    bad = [p['update'] for p in per if p.get('physical_tokens') != phys]
    c.add('data.physical_tokens', not bad, f'updates {bad[:8]} differ from INDEX {phys}')
    sup = expected.get('supervised_tokens_per_update')
    plan_updates = expected.get('plan_updates')
    if not isinstance(sup, list) or len(sup) < done or (plan_updates is not None and len(sup) != plan_updates):
        c.add('data.supervised_vector', False, f'INDEX supervised-token vector for plan {plan} has '
              f'{len(sup) if isinstance(sup, list) else sup!r} entries; {done} completed updates, plan of {plan_updates}')
    else:
        bad = [p['update'] for p in per if p.get('supervised_tokens') != sup[p['update'] - 1]]
        c.add('data.supervised_tokens', not bad, f'updates {bad[:8]} differ from INDEX plan {plan}')
        total = inv.get('supervised_tokens_total')
        c.add('data.supervised_tokens_total', total == sum(sup[:done]),
              f'receipt total {total!r} != INDEX sum {sum(sup[:done])} over the {done} completed updates')
    vec = (inv.get('supervised_tokens_per_update') or {}).get(plan) if isinstance(inv.get('supervised_tokens_per_update'), dict) else None
    sig = [p.get('supervised_tokens') for p in per]
    c.add('data.receipt_supervised_vector', isinstance(vec, list) and vec == sig,
          f'receipt supervised_tokens_per_update[{plan}] has {len(vec) if isinstance(vec, list) else vec!r} entries '
          f'(must equal the signals of the {done} completed updates)')
    got = inv.get('data_order_digest')
    c.add('data.order_digest', got == expected['index_order_digest'],
          f'receipt data_order_digest {got!r} != INDEX.json data_order_digest[{plan}] {expected["index_order_digest"]}')
    order, ppu = expected.get('order'), expected['packs_per_update']
    required = request.purpose != 'lifecycle'
    if not order or done > len(order) // ppu:
        c.add('data.consumed_digest', False, f'plan order unavailable or shorter than {done} updates')
        return
    consumed = inv.get('data_consumed_digest')
    if consumed is not None or required:
        want = update_packs_digest(order, ppu, done)
        c.add('data.consumed_digest', consumed == want,
              f'receipt data_consumed_digest {consumed!r} != digest of the plan-{plan} packs of updates 1..{done} {want}')
    per_u = inv.get('data_consumed_per_update')
    if per_u is not None or required:
        want_u = per_update_digests(order, ppu, done)
        bad = [k + 1 for k in range(done) if not isinstance(per_u, list) or k >= len(per_u) or per_u[k] != want_u[k]]
        extra = len(per_u) - done if isinstance(per_u, list) and len(per_u) > done else 0
        c.add('data.consumed_per_update', not bad and not extra,
              f'consumed-pack digests missing or different at updates {bad[:8]}'
              + (f'; {extra} extra entries' if extra else ''))
