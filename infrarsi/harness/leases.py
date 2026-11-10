"""Leases (/data/infrarsi/leases/WINH<k>.json) and occupancy checks; refuse when anyone else is on a node."""
from __future__ import annotations

import os
import socket

from infrarsi.contracts import SCHEMA_LEASE, RemotePaths, utc_now

from .remote import node_index, node_name

IDLE_MEM_MIB = 2048          # a GPU with more memory in use but no visible process belongs to another container


class OccupiedError(RuntimeError):
    """Raised before any launch when a target node is busy; the run must not start."""

    def __init__(self, report: dict):
        self.report = report
        super().__init__('target nodes occupied: ' + '; '.join(f'{n}: {r}' for n, r in report.items()))


def occupancy(remote, nodes, own_sessions: dict | None = None) -> dict:
    """Per node: foreign compute processes, busy GPU memory, current lease. own_sessions: {node: {sid,...}}."""
    own_sessions = own_sessions or {}
    leases = remote.call(nodes[0], 'read_leases', lease_dir=RemotePaths.leases)
    report = {}
    for node in nodes:
        name = node_name(node)
        st = remote.call(node, 'gpu_status')
        mine = set(own_sessions.get(name, ()))
        foreign = [a for a in st['compute_apps'] if a.get('sid') not in mine or a.get('sid') is None]
        busy = [g for g in st['gpus'] if g['mem_used_mib'] > IDLE_MEM_MIB] if not st['compute_apps'] else []
        report[name] = {'foreign_apps': foreign, 'busy_gpus_no_process': busy, 'lease': leases.get(name),
                        'error': st.get('error'), 'ips': st.get('ips', []), 'hostname': st.get('hostname')}
    return report


def lease_is_ours(lease: dict | None, attempt_id: str | None, run_id: str | None) -> bool:
    """Our attempt's lease, or a run-scoped lease the dispatcher holds for this run across its attempts."""
    if not lease:
        return False
    if attempt_id is not None and lease.get('attempt_id') == attempt_id:
        return True
    return run_id is not None and lease.get('scope') == 'run' and lease.get('run_id') == run_id


def check_free(remote, nodes, attempt_id: str | None = None, run_id: str | None = None) -> dict:
    """Raise OccupiedError if any node has foreign processes, unexplained memory use, or a lease that is not ours."""
    rep = occupancy(remote, nodes)
    bad = {}
    for name, r in rep.items():
        why = []
        if r['error']:
            why.append(f'probe error {r["error"]}')
        if r['foreign_apps']:
            why.append(f'{len(r["foreign_apps"])} compute processes {[a["pid"] for a in r["foreign_apps"]]}')
        if r['busy_gpus_no_process']:
            why.append(f'GPU memory in use without visible process on {[g["index"] for g in r["busy_gpus_no_process"]]}')
        lease = r['lease']
        if lease and not lease_is_ours(lease, attempt_id, run_id):
            why.append(f'leased by {lease.get("owner")}:{lease.get("run_id")}/{lease.get("attempt_id")}')
        if why:
            bad[name] = ', '.join(why)
    if bad:
        raise OccupiedError(bad)
    return rep


def make_lease(run_id: str, attempt_id: str, nodes: list[str], purpose: str = '', scope: str = 'attempt') -> dict:
    return {'schema': SCHEMA_LEASE, 'run_id': run_id, 'attempt_id': attempt_id, 'since': utc_now(),
            'owner': 'infrarsi', 'nodes': [node_name(n) for n in nodes], 'purpose': purpose, 'scope': scope,
            'driver_host': socket.gethostname(), 'driver_pid': os.getpid()}


def acquire(remote, nodes, lease: dict) -> None:
    res = remote.call(nodes[0], 'acquire_leases', lease_dir=RemotePaths.leases,
                      nodes=[node_name(n) for n in nodes], lease=lease)
    if not res['ok']:
        raise OccupiedError({n: f'leased by {(c or {}).get("owner")}:{(c or {}).get("run_id")}/{(c or {}).get("attempt_id")}'
                             for n, c in res['conflicts'].items()})


def release(remote, nodes, attempt_id: str) -> dict:
    return remote.call(nodes[0], 'release_leases', lease_dir=RemotePaths.leases,
                       nodes=[node_name(n) for n in nodes], attempt_id=attempt_id)


def internal_ip(status: dict) -> str:
    ips = [ip for ip in status.get('ips', []) if ip.startswith('172.')] or status.get('ips', [])
    if not ips:
        raise RuntimeError(f'no internal IP for {status.get("hostname")}')
    return ips[0]


__all__ = ['OccupiedError', 'occupancy', 'check_free', 'lease_is_ours', 'make_lease', 'acquire', 'release', 'internal_ip',
           'node_index', 'node_name']
