"""Read-only preflight of the F1 node set (AMENDMENT_EXT_003 G-F1 (4)). Writes nothing on any node.

Per node: hostname vs the node map (runner.HOSTNAMES), GPU occupancy (any compute process, unexplained memory), the
node lease (leases/WINH<k>.json), the two legacy host lock files the launcher flocks (present and writable; no formal-
queue launcher running), the node's wall clock against the others (the screen-wide deadline and the wall_cap guard
compare clocks across nodes), the /mnt/base/qwen-sft link the source tree's venv resolves through, the frozen hashes
of every read-only file a screen executes or reads (deploy.REMOTE_READ_ONLY), the deployed variant layer under
FN_SRC against frozen_manifest.json, and whether the ext compile-cache copy is populated.

    python -m infrarsi.ext.flashnext.preflight WINH0 WINH1 WINH3 WINH4 WINH6   -> JSON report on stdout
"""
from __future__ import annotations

import json
import sys
import time

from . import deploy as D
from . import runner as R
from . import variant as V
from ..common.remote_exec import IDLE_MEM_MIB, node_lease_path, q


def deployed_paths(manifest: dict) -> dict:
    files = {f'{V.FN_SRC}/runtime/{n}': h for n, h in manifest['variant_runtime']['inventory'].items()}
    files.update({f'{V.FN_SRC}/node/{n}': h for n, h in manifest['node_files'].items()})
    return files


CLOCK_SPREAD_MAX_S = 5.0     # the launchers' screen-wide deadline and the wall_cap guard compare wall clocks across nodes


def clock_offset(ops) -> dict:
    """Node wall clock minus this machine's, bracketed by the round trip (offset +- uncertainty)."""
    t0 = time.time()
    res = ops.sh('date +%s.%N', timeout=30)
    t1 = time.time()
    try:
        node_t = float((res.get('stdout') or '').split()[0])
    except (ValueError, IndexError):
        return {'offset_s': None, 'uncertainty_s': None}
    return {'offset_s': round(node_t - (t0 + t1) / 2, 3), 'uncertainty_s': round((t1 - t0) / 2, 3)}


def clock_spread(report: dict) -> None:
    """Flag a node whose clock is more than CLOCK_SPREAD_MAX_S (+ measurement uncertainty) off the nodes' median."""
    offs = {n: r['clock'] for n, r in report['nodes'].items() if (r.get('clock') or {}).get('offset_s') is not None}
    if len(offs) < 2:
        return
    values = sorted(c['offset_s'] for c in offs.values())
    median = values[len(values) // 2]
    report['clock_spread_s'] = round(values[-1] - values[0], 3)
    for n, c in offs.items():
        if abs(c['offset_s'] - median) > CLOCK_SPREAD_MAX_S + c['uncertainty_s']:
            r = report['nodes'][n]
            r['problems'].append(f'clock {c["offset_s"] - median:+.1f} s off the other nodes (> {CLOCK_SPREAD_MAX_S:.0f} s)')
            r['ok'] = False


def check_node(ops, node: str, manifest: dict | None = None) -> dict:
    out = {'node': node, 'problems': [], 'notes': []}
    out['clock'] = clock_offset(ops)
    if out['clock']['offset_s'] is None:
        out['problems'].append('wall clock not measured (date failed): the cross-node deadline cannot be checked')
    probe = ops.probe()
    out['hostname'] = probe.get('hostname')
    expected = R.HOSTNAMES[R.node_index(node)]
    if probe.get('hostname') != expected:
        out['problems'].append(f'hostname {probe.get("hostname")} != {expected}')
    if probe.get('error'):
        out['problems'].append(f'probe error {probe["error"]}')
    apps = probe.get('apps') or []
    out['compute_apps'] = [{k: a.get(k) for k in ('pid', 'gpu_index', 'used_mib', 'cwd')} for a in apps][:16]
    if apps:
        out['problems'].append(f'{len(apps)} compute processes on the GPUs')
    busy = [g['index'] for g in probe.get('gpus') or [] if (g.get('mem_used_mib') or 0) > IDLE_MEM_MIB]
    if busy and not apps:
        out['problems'].append(f'GPU memory in use without a visible process on {busy}')
    out['gpus'] = len(probe.get('gpus') or [])
    if out['gpus'] != 8:
        out['problems'].append(f'{out["gpus"]} GPUs visible')
    lease = ops.read_json(node_lease_path(node))
    out['lease'] = lease and {k: lease.get(k) for k in ('owner', 'run_id', 'attempt_id', 'purpose', 'since')}
    if lease:
        out['problems'].append(f'node lease held by {lease.get("owner")}:{lease.get("run_id")}')
    link = ops.sh('readlink -f /mnt/base/qwen-sft; test -x /mnt/base/qwen-sft/.venv-megatron-qwen35/bin/python && echo VENV_OK',
                  timeout=60)
    lines = (link.get('stdout') or '').split()
    out['mnt_base_qwen_sft'] = lines[0] if lines else None
    if 'VENV_OK' not in lines:
        out['problems'].append('/mnt/base/qwen-sft venv python not reachable')
    host = R.node_index(node) + 1                    # the launcher flocks these two (never creates or writes them)
    legacy = [f'{d}/host{host:02d}.lock' for d in R.LEGACY_LOCK_DIRS]
    res = ops.sh('for f in ' + ' '.join(q(p) for p in legacy) + '; do if test -f "$f" && test -w "$f"; then echo "OK $f";'
                 ' else echo "MISSING $f"; fi; done; pgrep -f "[l]aunch_lp_node[.]py" >/dev/null && echo LEGACY_LAUNCHER_RUNNING; true',
                 timeout=60)
    lines = (res.get('stdout') or '').splitlines()
    out['legacy_locks_ok'] = all(f'OK {p}' in lines for p in legacy)
    if not out['legacy_locks_ok']:
        out['problems'].append(f'legacy host lock files missing or not writable: {[x for x in lines if x.startswith("MISSING")]}')
    if 'LEGACY_LAUNCHER_RUNNING' in lines:
        out['problems'].append('a formal-queue launcher (launch_lp_node.py) is running on this node')
    got = ops.hash_files(sorted(D.REMOTE_READ_ONLY))
    bad = sorted(p for p, h in D.REMOTE_READ_ONLY.items() if got.get(p) != h)
    out['read_only_hashes_ok'] = not bad
    if bad:
        out['problems'].append(f'read-only file hashes differ: {bad[:4]}')
    if manifest is not None:
        want = deployed_paths(manifest)
        got = ops.hash_files(sorted(want))
        missing = sorted(p for p in want if got.get(p) is None)
        wrong = sorted(p for p in want if got.get(p) not in (None, want[p]))
        out['variant_layer'] = 'deployed' if not missing and not wrong else ('absent' if len(missing) == len(want) else 'partial')
        if wrong or (missing and len(missing) != len(want)):
            out['problems'].append(f'variant layer differs from frozen_manifest: missing {missing[:3]}, wrong {wrong[:3]}')
        if out['variant_layer'] == 'absent':
            out['notes'].append('variant layer not deployed yet (deploy.py)')
    cache = ops.sh(f'ls -A {q(V.FN_CACHE)} 2>/dev/null | head -3 | tr "\\n" " "', timeout=60)
    out['ext_cache_entries'] = (cache.get('stdout') or '').split()
    if not out['ext_cache_entries']:
        out['notes'].append('ext compile cache empty (first screen compiles cold unless deploy copies the warm cache)')
    out['ok'] = not out['problems']
    return out


def preflight(ops_by_node: dict, nodes) -> dict:
    try:
        manifest = D.load_manifest()
    except Exception as exc:                          # noqa: BLE001 - reported
        manifest = None
        err = repr(exc)
    else:
        err = None
    report = {'schema': 'flashnext-preflight-v1', 'read_only': True, 'manifest_error': err, 'nodes': {}}
    for n in nodes:
        try:
            report['nodes'][n] = check_node(ops_by_node[n], n, manifest)
        except Exception as exc:                      # noqa: BLE001 - an unreachable node is a finding
            report['nodes'][n] = {'node': n, 'ok': False, 'problems': [f'unreachable: {exc!r}'[:300]]}
    clock_spread(report)
    report['ok_nodes'] = sorted(n for n, r in report['nodes'].items() if r.get('ok'))
    return report


if __name__ == '__main__':
    from ..common.remote_exec import WinhNodeOps
    nodes = [n.upper() for n in sys.argv[1:]] or list(R.DEFAULT_NODES)
    print(json.dumps(preflight({n: WinhNodeOps(n) for n in nodes}, nodes), indent=1, default=str))
