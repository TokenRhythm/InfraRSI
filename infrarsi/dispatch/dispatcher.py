"""Dispatcher: one scheduling cycle = plan -> (paused?) -> per assignment: occupancy re-check, run-scoped lease,
handler in a thread. Never launches on a node with a foreign process or a foreign lease; a refused node is skipped
for the cycle and the run stays pending (lane re-assignment happens on the next cycle).
"""
from __future__ import annotations

import threading
import time
import traceback

from infrarsi.contracts import utc_now

from . import queue as Q
from .handlers import Context, Deferred, handlers as all_handlers, is_deferred, terminal_state


class Dispatcher:
    def __init__(self, *, remote=None, store: Q.QueueStore | None = None, matrix: dict | None = None, include_p2=False,
                 enabled: set | None = None, live: bool = True, hw: dict | None = None, runner=None, handler_map=None,
                 log=print):
        self.store = store or Q.QueueStore()
        self.matrix = matrix or Q.load_matrix()
        self.remote, self.live, self.log = remote, live, log
        self.include_p2, self.enabled = include_p2, set(enabled or ())
        self.handlers = handler_map if handler_map is not None else all_handlers()
        self.ctx = Context(store=self.store, remote=remote, runner=runner, hw=hw, log=log)
        self.threads: dict[str, threading.Thread] = {}

    def _remote(self):
        if self.remote is None:
            from infrarsi.harness.remote import default_remote
            self.remote = self.ctx.remote = default_remote()
        return self.remote

    def live_free(self) -> tuple[set[int], dict]:
        """Nodes with no lease and no GPU process (read-only probe of all eight nodes)."""
        from infrarsi.harness import leases as Lz
        rep = Lz.occupancy(self._remote(), [f'WINH{k}' for k in range(8)])
        free = {int(n[4:]) for n, r in rep.items() if not (r['foreign_apps'] or r['busy_gpus_no_process'] or r['lease'] or r['error'])}
        return free, rep

    def plan(self, live: bool | None = None, check_downloads: bool = False) -> dict:
        live = self.live if live is None else live
        free, rep, dl = None, None, None
        if live:
            free, rep = self.live_free()
        if check_downloads:
            dl = Q.downloads_status(self._remote())
        p = Q.plan(self.matrix, self.store, include_p2=self.include_p2, enabled_conditional=self.enabled, downloads=dl,
                   live_free=free, handlers=self.handlers)
        if rep is not None:
            p['occupancy'] = {n: {'lease': (r['lease'] or {}).get('run_id'), 'foreign_pids': [a['pid'] for a in r['foreign_apps']],
                                  'busy_gpus': [g['index'] for g in r['busy_gpus_no_process']], 'error': r['error']}
                              for n, r in rep.items()}
        if dl is not None:
            p['downloads'] = dl
        return p

    def step(self, check_downloads: bool = False) -> list[str]:
        """Launch whatever is runnable now; returns the run ids started in this cycle."""
        self._reap()
        p = self.plan(check_downloads=check_downloads)
        blocked = Q.paused_nodes(p['paused'])
        if blocked is not None and not blocked:
            self.log(f'queue paused: {p["paused"]["reasons"][-1]["reason"]}')
            return []
        started = []
        for a in p['assign']:
            h = self.handlers.get(self.matrix['by_id'][a['run_id']]['arm'])
            if h is None or is_deferred(h) or not a.get('launchable') or a['run_id'] in self.threads:
                continue
            if blocked and {int(n[4:]) for n in a['nodes'] or []} & blocked:
                continue                                  # lane-scoped pause
            if a['nodes'] and not self._claim(a):
                continue
            self.store.update_run(a['run_id'], status='running', nodes=a['nodes'],
                                  node_indices=[int(n[4:]) for n in a['nodes']], started_utc=utc_now(), error=None)
            t = threading.Thread(target=self._run, args=(a['run_id'], a['nodes'], h), daemon=True, name=a['run_id'])
            self.threads[a['run_id']] = t
            t.start()
            started.append(a['run_id'])
            self.log(f'started {a["run_id"]} on {a["nodes"] or "driver"}')
        return started

    def _claim(self, a: dict) -> bool:
        from infrarsi.harness import leases as Lz
        try:
            Lz.check_free(self._remote(), a['nodes'])
            Lz.acquire(self._remote(), a['nodes'], Lz.make_lease(a['run_id'], 'run:' + a['run_id'], a['nodes'],
                                                                 purpose='dispatch', scope='run'))
            return True
        except Lz.OccupiedError as e:
            self.store.update_run(a['run_id'], last_refusal={'utc': utc_now(), 'nodes': a['nodes'], 'report': e.report})
            self.log(f'refused {a["run_id"]} on {a["nodes"]}: {e}')
            return False

    def _run(self, run_id, nodes, handler):
        """Terminal state from the handler's result (handlers.terminal_state): only 'done' (and an FR / ED
        'infeasible') satisfies dependants; an invalid receipt or failed run never does."""
        run = self.matrix['by_id'][run_id]
        try:
            res = handler(run, nodes, self.ctx)
            status, why = terminal_state(run['arm'], res)
            self.store.update_run(run_id, status=status, result=res, error=why, finished_utc=utc_now())
        except Exception as e:                                         # noqa: BLE001
            from infrarsi.harness.leases import OccupiedError
            status = 'pending' if isinstance(e, OccupiedError) else ('deferred' if isinstance(e, Deferred) else 'failed')
            self.store.update_run(run_id, status=status, error=repr(e), traceback=traceback.format_exc()[-4000:],
                                  finished_utc=utc_now())
        finally:
            if nodes:
                self._release(run_id, nodes)

    def _release(self, run_id, nodes):
        from infrarsi.harness import leases as Lz
        try:
            rep = Lz.occupancy(self._remote(), nodes)
            if any(r['foreign_apps'] for r in rep.values()):
                self.store.update_run(run_id, lease_kept='GPU processes still present after the run; inspect before releasing')
                return
            Lz.release(self._remote(), nodes, 'run:' + run_id)
        except Exception as e:                                         # noqa: BLE001
            self.store.update_run(run_id, lease_release_error=repr(e))

    def _reap(self):
        for rid in [r for r, t in self.threads.items() if not t.is_alive()]:
            self.threads.pop(rid)

    def loop(self, interval_s: float = 60.0, max_cycles: int | None = None, check_downloads: bool = True):
        n = 0
        while max_cycles is None or n < max_cycles:
            try:
                self.step(check_downloads=check_downloads)
            except Exception as e:                                     # noqa: BLE001
                self.log(f'dispatch cycle error: {e!r}')
            n += 1
            time.sleep(interval_s)


def format_plan(p: dict, matrix: dict) -> str:
    lines = [f'plan @ {p["utc"]}', f'paused: {p["paused"]["reasons"][-1] if p["paused"] else "no"}',
             f'free nodes: {["WINH%d" % k for k in p["free_nodes"]]}  reserved: {p["reserved_nodes"]}',
             f'running: {p["running"] or "-"}', 'next runnable (priority order):']
    for a in p['assign']:
        r = matrix['by_id'][a['run_id']]
        lines.append(f'  {a["run_id"]:<34} {r["priority"][:2]} {r["gpus"]:>2}g/{r["nodes"]}n lane={r["node_lane"]:<12} '
                     f'-> {a["nodes"] if a["nodes"] is not None else a["where"]}  handler={"yes" if a["handler"] else "MISSING"}'
                     f'{"" if a.get("launchable") or not a["handler"] else " (not now)"}')
    if p.get('deferred'):
        lines.append(f'deferred (P2, not in v0.1): {", ".join(p["deferred"])}')
    miss = {}
    for b in p['blocked']:
        for m in b['missing']:
            miss[m] = miss.get(m, 0) + 1
    lines.append(f'blocked: {len(p["blocked"])} runs; top blockers:')
    for m, c in sorted(miss.items(), key=lambda x: -x[1])[:12]:
        lines.append(f'  {c:>3} x {m}')
    if p.get('occupancy'):
        def occ(n, o):
            bits = [f'lease {o["lease"]}' if o['lease'] else '', f'pids {o["foreign_pids"]}' if o['foreign_pids'] else '',
                    f'busy GPUs {o["busy_gpus"]}' if o['busy_gpus'] else '', 'probe error' if o['error'] else '']
            return f'{n} ' + (', '.join(b for b in bits if b) or 'free')
        lines.append('occupancy: ' + '; '.join(occ(n, o) for n, o in p['occupancy'].items()))
    return '\n'.join(lines)
