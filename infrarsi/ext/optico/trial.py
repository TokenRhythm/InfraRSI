"""One OptiCo trial -> ext.common.env_runner.EnvRun (the runner base turns it into the signed ext-receipt-v1).

Score authority: OptiCo's run_monitor.sh (`Result` = success iff its TFLOPS > 0, i.e. the mean of the throughput
lines from the third one on). judge_valid = that verdict. The minimized metric is the mean `elapsed time per
iteration` over the same log lines OptiCo averages (iterations 3..5 when the run completed), in seconds; at fixed work
it is inverse to OptiCo's TFLOPS. Guards (G-B1/G-B2/G-B3 + the generic frozen_files/outputs_present/exit_ok/budget of
ext.common.guards) veto only trials the judge accepted, so a guard rejection always means "OptiCo said success,
InfraRSI says no"; on judge failures the same checks are kept as reports (diagnosis, never counted as rejections).
Used by OptiCoRunner.execute (one trial) and by the B-NAT driver (one EnvRun per report.csv row).
"""
from __future__ import annotations

from ..common import guards as cg
from ..common.env_runner import EnvRun
from . import guards as gb
from . import params as pm
from . import parse
from . import workloads as wl

TFLOPS_REL_TOL = 1e-3      # OptiCo's awk prints the mean with 6 significant digits


def _round(v, n=6):
    return round(v, n) if isinstance(v, float) else v


def judge_metric(iters: list[dict]) -> float | None:
    """The minimized metric: mean elapsed seconds of iterations 3..5 (by iteration number). When the window is
    incomplete (G-B3 then rejects), the mean over the lines OptiCo's `tail -n +3` averaged, so the judge's own success
    verdict keeps a finite metric and the rejection is attributed to the guard, not to the judge."""
    exact = parse.window_metric(iters)['mean_iter_s']
    if exact is not None:
        return exact
    ms = [r['elapsed_ms'] for r in iters[2:] if r.get('elapsed_ms')]
    return sum(ms) / len(ms) / 1000.0 if ms else None


def complete_knobs(params) -> dict | None:
    """All ten knobs (typed where they parse; a native proposal outside OptiCo's documented values, e.g. AC=selective,
    stays a string and is still compared with the log); None when a knob is missing."""
    if not isinstance(params, dict) or any(k not in params or params[k] in ('', None) and k not in pm.OPT_INT_KNOBS
                                           for k in pm.KNOBS):
        return None
    return {k: v for k, v in pm.typed({k: params[k] for k in pm.KNOBS}).items()}


def summary(params: dict | None, judge: dict | None, metric_s: float | None, mem_gib: float | None, failure: list[str]) -> dict:
    """Compact block for the proposer's receipt digest (OptiCo's table row + our metric)."""
    return {'params': pm.format_params(params) if params and all(k in params for k in pm.KNOBS) else params,
            'result': judge and judge['result'], 'tflops_per_gpu': judge and judge['tflops'],
            'iter_time_s': _round(metric_s, 4), 'mem_max_mib_optico': judge and judge['mem_max_mib'],
            'max_reserved_gib_megatron': mem_gib, 'failure': (judge or {}).get('failure') or None,
            'failure_log_head': failure[:30]}


def evaluate(workload, *, candidate_id: str, row: dict | None, log_text: str, requested: dict | None = None,
             memory_text: str = '', error_text: str = '', frozen_expected: dict | None = None,
             frozen_before: dict | None = None, frozen_after: dict | None = None, exit_code=None, timed_out: bool = False,
             wall_cap_s: float | None = None, trial_wall_s: float | None = None, phases: dict | None = None,
             gb2: gb.GB2Policy | None = None, deviations=(), copy_mismatch=None, extra: dict | None = None) -> EnvRun:
    """requested: the knobs the runner wrote (None for B-NAT rows: then the row's own knobs are bound to the log).
    copy_mismatch: files of the executed workspace copy that differ from the verified base after the declared
    substitutions (None = not measured)."""
    w = wl.get(workload)
    facts = parse.log_facts(log_text)
    iters = facts['iterations']
    win = parse.window_metric(iters)
    judge = parse.row_judge(row) if row is not None else None
    judge_valid = bool(judge and judge['result'] == 'success')
    applied = pm.params_from_row(row) if row is not None else None
    knobs = requested if requested is not None else applied
    bind = complete_knobs(requested if requested is not None else applied)

    checks = cg.GuardResult()
    # G-B1 binds what was measured (torchrun line) to what the row/request says was run
    checks.extend(gb.fixed_flags(facts['args'], w, bind, ac_line=facts['ac'], ac_layers_line=facts['ac_layers']))
    if row is not None and bind is None:
        checks.reject('output_integrity', 'the parameter set of this trial is incomplete or unparseable '
                                          f'({"request" if requested is not None else "report.csv row"}): {knobs}')
    if judge is not None:                                # the judge's number must come from the log it was paired with
        mine = facts['optico_tflops']
        if judge['tflops'] is None or judge['tflops'] <= 0:
            if judge_valid:
                checks.reject('output_integrity', f'report.csv row says success without a positive TFLOPS ({judge["row"].get("TFLOPS")!r})')
        elif abs(mine - judge['tflops']) > TFLOPS_REL_TOL * max(1.0, judge['tflops']):
            checks.reject('output_integrity', f'report.csv TFLOPS {judge["tflops"]} != {mine:.4f} recomputed from the paired '
                                              'Megatron log (row and log are not the same run)')
    if copy_mismatch:
        checks.reject('frozen_files', f'executed workspace copy differs from the verified base: {list(copy_mismatch)[:8]}')
    checks.extend(gb.completed_iterations(iters))
    checks.extend(gb.iteration1_loss(gb.loss1(iters), gb2))
    if frozen_expected is not None:
        checks.extend(cg.frozen_files(frozen_expected, frozen_before, frozen_after))
    checks.extend(cg.outputs_present({'report.csv row': row is not None, 'megatron log': len(log_text or '')},
                                     ['report.csv row', 'megatron log']))
    checks.extend(cg.exit_ok(exit_code, timed_out=timed_out, what='run_monitor.sh'))
    if wall_cap_s is not None:
        checks.extend(cg.budget(trial_wall_s, wall_cap_s, what='trial'))
    if judge and judge['tflops'] is not None:
        mine = facts['optico_tflops']
        checks.report('judge_crosscheck', optico_tflops=judge['tflops'], recomputed=round(mine, 4),
                      window_mean=_round(win['mean_tflops'], 4),
                      agree=abs(mine - judge['tflops']) <= TFLOPS_REL_TOL * max(1.0, judge['tflops']))
    if judge_valid:
        guards = checks
    else:
        guards = cg.GuardResult(reports=[*checks.reports, *({'id': r['id'], 'would_reject': r['detail'],
                                                             'note': 'judge invalid: not a veto'} for r in checks.rejections)])
    reasons = []
    if not judge_valid:
        if judge is None:
            reasons.append('no report.csv row for this trial (run_monitor.sh did not finish)')
        else:
            fail = judge['failure'] if judge['failure'] not in ('', 'NULL') else None
            reasons.append(f'OptiCo Result={judge["result"] or "?"}' + (f' ({fail})' if fail else ''))
    metric_s = judge_metric(iters) if judge_valid else None
    mem = facts['memory']
    mem_gib = round(mem['max_reserved_mb'] / 1024.0, 3) if mem['max_reserved_mb'] else None
    failure = parse.error_head(error_text) if not judge_valid else []
    tflops = judge['tflops'] if judge else None
    ph = dict(phases or {})
    if iters and all(r.get('elapsed_ms') for r in iters):
        ph['iterations_s'] = round(sum(r['elapsed_ms'] for r in iters) / 1000.0, 3)
    if trial_wall_s is not None:
        ph['trial_s'] = round(float(trial_wall_s), 3)
    out_extra = {
        'workload': {'key': w.key, 'model_type': w.model_type, 'model_size': w.model_size, 'script': w.script},
        'params': knobs, 'params_applied': applied,
        'summary': summary(knobs, judge, metric_s, mem_gib, failure),
        'memory': {'max_peak_reserved_gib': mem_gib, 'megatron_per_rank_mb': {str(k): v for k, v in mem['per_rank'].items()},
                   'optico_mem_max_mib': judge['mem_max_mib'] if judge else None,
                   'optico_mem_ave_mib': judge['mem_ave_mib'] if judge else None,
                   'smi_samples': parse.memory_samples(memory_text)},
        'loss': {'iter1': gb.loss1(iters), 'per_iter': [r.get('lm_loss') for r in iters], 'random_init': facts['random_init']},
        'iterations': iters, 'script_errors': facts['script_errors'], 'oom': facts['oom'], 'failure_log_head': failure,
        'g_b2_policy': (gb2 or gb.GB2Policy()).to_dict(), 'deviations': list(deviations),
        **(extra or {})}
    return EnvRun(candidate_id=candidate_id, judge=judge, judge_valid=judge_valid, metric_value=metric_s,
                  metric_raw={'tflops_per_gpu': tflops, 'tflops': tflops,   # 'tflops': ext.common.trajectory.y_value
                              'mfu_optico': judge['mfu_optico'] if judge else None,
                              'mfu_h100': round(tflops / wl.H100_PEAK_TFLOPS, 6) if tflops else None,
                              'window': win['window'], 'iter_ms': win['iter_ms'], 'tflops_per_iter': win['tflops'],
                              'window_mean_iter_s': _round(win['mean_iter_s']), 'tflops_window_mean': _round(win['mean_tflops']),
                              'tflops_window_rsd': _round(win['tflops_rsd']), 'tflops_recomputed': round(facts['optico_tflops'], 4)},
                  judge_reasons=reasons, guards=guards, phases=ph,
                  candidate_digest=pm.params_key(knobs) if knobs and all(k in knobs for k in pm.KNOBS) else None,
                  extra=out_extra)
