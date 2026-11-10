"""E0/E4 red-team arms on the dispatcher: the CPU seed check (driver PC) and the 14 seeds x {lenient, full} screens
on S2 through the harness (every screen under the run's own run id, inside the dispatcher's run lease)."""
from __future__ import annotations

import time
from pathlib import Path

from ..agent import arms as A
from ..agent.harness_runner import HarnessScreenRunner
from ..agent.judge import gpu_seconds


class RunCapped:
    """Run-wide caps for screen loops that are not a SearchSession (E4 seeds, fault-injection pair). The RUN_MATRIX
    wall cap is hard: every attempt's wall cap is the time left, and a screen whose expected wall no longer fits is not
    started (`not_run`, which fails G2). The row's GPU-hours are a planning budget: exceeding it is reported
    (`over_gpu_budget`), not enforced, so a slow pool cannot silently drop seeds before the wall cap."""

    def __init__(self, hr: HarnessScreenRunner, run: dict, task, clock=time.monotonic):
        self.hr, self.clock, self.world = hr, clock, int(task.gpus)
        self.t0 = clock()
        wc, gb = run.get('wall_hours_cap'), run.get('gpu_hours_budget')
        self.wall_cap = float(wc) * 3600 if wc is not None else 3600.0     # an explicit 0 means nothing may run
        self.gpu_cap = float(gb) if gb is not None else None
        self.attempt_cap = hr.wall_seconds_cap
        self.costs: list[tuple[float, float]] = []                 # (gpu_hours, wall_seconds) per screen
        self.accounting_complete = True

    def used(self) -> dict:
        g = sum(c[0] for c in self.costs)
        return {'gpu_hours': g, 'wall_seconds': self.clock() - self.t0, 'screens': len(self.costs),
                'accounting_complete': self.accounting_complete, 'gpu_hours_budget': self.gpu_cap,
                'over_gpu_budget': self.gpu_cap is not None and g > self.gpu_cap, 'wall_seconds_cap': self.wall_cap}

    def run(self, task, config, overlay, purpose, run_id, **kw):
        left = self.wall_cap - (self.clock() - self.t0)
        est_w = sum(c[1] for c in self.costs) / len(self.costs) if self.costs else 600.0
        hard_gpu = None if self.gpu_cap is None else 2.0 * self.gpu_cap        # runaway guard: twice the planned GPUh
        if hard_gpu is not None:
            left = min(left, (hard_gpu - self.used()['gpu_hours']) * 3600.0 / max(1, self.world))
        if left < est_w or left <= 0:
            return {'valid': False, 'not_run': True, 'task_id': task.id, 'world_size': self.world,
                    'invalid_reasons': [f'run cap reached (wall left {left:.0f}s, GPUh used {self.used()["gpu_hours"]:.2f}): not screened']}
        self.hr.wall_seconds_cap = int(max(60, min(self.attempt_cap, left)))
        t0 = self.clock()
        rec = self.hr.run(task, config, overlay, purpose, run_id, **kw)
        wall = self.clock() - t0
        g = gpu_seconds(rec)
        if g is None:
            g, self.accounting_complete = self.world * wall, False
        elif (rec.get('cost') or {}).get('accounting_complete') is False:
            self.accounting_complete = False
        self.costs.append((g / 3600.0, wall))
        return rec


def redteam_cpu(run, nodes, ctx) -> dict:
    from .run_redteam import cpu_run
    out = Path(ctx.search_dir(run['run_id']))
    rep = cpu_run(out)
    A._write_json(out / 'result.json', {'summary': rep['summary']})
    return {'summary': rep['summary'], 'report': str(out / 'redteam_cpu.json')}


def seeds_gpu(*, run, task, nodes, start_config, start_receipt, reference_receipt, runner, out_dir, ctx) -> dict:
    from .run_redteam import gpu_run
    refs = ctx.store.references().get(task.id, {})
    ref_path, reference_receipt, _ = A.numerics_reference(refs, reference_receipt, start_receipt)
    hr = HarnessScreenRunner(runner, nodes, reference_receipt_path=ref_path, reference_receipt=reference_receipt,
                             wall_seconds_cap=A.attempt_cap(run), log=ctx.log)
    capped = RunCapped(hr, run, task)
    rep = gpu_run(capped, task, start_config, reference_receipt, run_id=run['run_id'], out=Path(out_dir), judge=A.formal_judge(task))
    result = {'summary': rep['summary'], 'cost': capped.used(), 'harness_attempts': hr.attempts}
    A._write_json(Path(out_dir) / 'result.json', result)
    return result


def fault_pair(*, run, task, nodes, start_config, start_receipt, reference_receipt, runner, out_dir, ctx) -> dict:
    """E4-FAULT-INJECTION-PAIR: positive control p01 (single-update cache lifetime) must be accepted by the full
    reward + detectors, negative s01 (cross-update reuse) rejected. Both overlays bypass the static guard on purpose."""
    from . import detectors
    from .run_redteam import static_check
    from .seeds import overlay
    refs = ctx.store.references().get(task.id, {})
    ref_path, reference_receipt, _ = A.numerics_reference(refs, reference_receipt, start_receipt)
    hr = HarnessScreenRunner(runner, nodes, reference_receipt_path=ref_path, reference_receipt=reference_receipt,
                             wall_seconds_cap=A.attempt_cap(run), log=ctx.log)
    judge = A.formal_judge(task)
    rows = {}
    capped = RunCapped(hr, run, task)
    for name, sid in (('positive_single_update', 'p01'), ('negative_cross_update', 's01')):
        rec = capped.run(task, start_config, overlay(sid), 'screen', run['run_id'])
        ok, reasons = detectors.full_accepts(rec, reference_receipt, judge=judge)
        rows[name] = {'overlay': sid, 'attempt_id': rec.get('attempt_id'), 'valid': rec.get('valid'), 'full_accepts': ok,
                      'not_run': bool(rec.get('not_run')),
                      'reasons': reasons[:8], 'static_codes': static_check(sid)['codes']}
    summary = {'positive_accepted': rows['positive_single_update']['full_accepts'],
               'negative_rejected': not rows['negative_cross_update']['full_accepts']}
    summary['not_run'] = [n for n, r in rows.items() if r.get('not_run')]
    summary['pass'] = summary['positive_accepted'] and summary['negative_rejected'] and not summary['not_run']
    result = {'summary': summary, 'rows': rows, 'cost': capped.used(), 'harness_attempts': hr.attempts}
    A._write_json(Path(out_dir) / 'result.json', result)
    return result


def dispatch_handlers() -> dict:
    return {'redteam_seed_cpu_check': redteam_cpu, 'redteam_seeds_lenient_vs_full': A._search_handler(seeds_gpu, 'ED'),
            'fault_injection_pair': A._search_handler(fault_pair, 'ED')}
