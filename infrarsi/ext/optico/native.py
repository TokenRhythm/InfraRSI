"""B-NAT driver (design §3.3): OptiCo's own `run_circle.sh <model> <size> <cycles>` loop on one leased 8-GPU node, its
prompts and loop logic unchanged; the analyzer's OpenAI client reaches InfraRSI's shim on this PC through a reverse
SSH tunnel. Output: one signed EnvReceipt per report.csv row, a run summary (result.json, with the published-protocol
best success row) and a SearchTrace (trace.json: one screen per row with cumulative cost, for CTG / AUC_budget).

  lease + occupancy (ext.common GpuLease, owner infrarsi-ext)
  -> shim (baselines.claude_openai_shim over the Claude CLI backend; LLM caps enforced per call before it runs:
     tokens with the full output allowance reserved, USD with a worst-case cost reserved, and the CLI's own per-call
     budget set to what is left)
  -> reverse tunnel (reports/external_bench_20261021/scripts/rtunnel.py as a subprocess: it reconnects by itself; the
     driver restarts it when the process dies or the node cannot reach the endpoint twice in a row)
  -> node/loop.sh (fresh workspace copy checked against the verified base, D-OC deviations, node-side watcher
     enforcing the trial and wall caps even without the driver)
  -> poll stamps / report.csv / status, enforce the caps from the driver side too (trials incl. RETRY, wall, LLM),
     kill the loop by its INFRARSI_TOKEN when a cap is hit
  -> collect report.csv and every iteration's logs, delete the bulky copy, release the lease (outermost finally)
  -> receipts: trial.evaluate per row (judge = OptiCo's row, bound to the iteration's log; guards recorded
     separately) assembled and signed by the OptiCoRunner base, LLM usage from the shim's own call log.
Protocol maxima (design §3.5, D-OC-11): cycles <= 21 (20 new + default), trials incl. RETRY <= 26, wall <= the workload's
arm wall (3 h, W3 1.5 h), 1.5M tokens, $30; smaller values are allowed (smoke tests), larger ones are refused.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
import uuid
from pathlib import Path

from ...contracts import PROJECT_ROOT, utc_now
from ...harness import signing
from ...harness.leases import OccupiedError
from ..common.remote_exec import GpuLease, new_token, q
from . import deviations as dv
from . import guards as gb
from . import params as pm
from . import parse
from . import prompt_variants as pv
from . import trial
from . import workloads as wl
from .runner import EXT_ROOT, RUNS, SRC, VENV, OptiCoRunner, node_files, read_events, read_log, spec_sh
from .task import ARMS, LLM_TOKENS, LLM_USD, make_task

RTUNNEL = PROJECT_ROOT / 'reports' / 'external_bench_20261021' / 'scripts' / 'rtunnel.py'
OPTICO_MAX_TOKENS = 8192            # analyzer.py's own max_tokens (kept as the output cap, D-OC-8)
CALL_TIMEOUT_S = 540.0              # below the OpenAI client's 600 s default, so the analyzer never re-sends a live call
PORTS = (18100, 18200)
MAX_CYCLES = 21                     # run_circle.sh <model> <size> 21: the default + 20 new trials
MAX_TRIALS = ARMS['B-NAT']['max_trials']
# Admission bounds per token (above Opus 5.5 list prices: input $4/M, 1 h cache writes $8/M, output $20/M); the cost
# actually charged is the CLI's total_cost_usd.
USD_IN_BOUND, USD_OUT_BOUND = 10e-6, 25e-6
MIN_CALL_USD = 0.02


# ---------------------------------------------------------------- LLM side
class _CappedBackend:
    """Passes the USD left in the arm to the backend as the call's own budget (ClaudeCLIBackend: --max-budget-usd)."""

    def __init__(self, backend, shim: 'BudgetedShim'):
        self._backend, self._shim = backend, shim

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def complete(self, system, prompt, **kw):
        kw['max_usd'] = max(MIN_CALL_USD, self._shim.usd_cap - self._shim.used['llm_usd'])
        return self._backend.complete(system, prompt, **kw)


class BudgetedShim:
    """baselines.claude_openai_shim.ShimServer with the arm's caps. Admission runs under the shim's lock before every
    call and reserves the worst case: input estimate + per-call overhead + the full output allowance (tokens) and the
    same priced at upper-bound rates (USD), so neither cap can be overrun by an admitted call. transform: rewrites the
    user messages of every request (prompt_variants; D-OC-15 strip_think_format is the B-NAT default)."""

    def __init__(self, backend, *, tokens_cap: int, usd_cap: float, log_path: Path, calls_dir: Path,
                 call_timeout: float = CALL_TIMEOUT_S, max_tokens: int = OPTICO_MAX_TOKENS, port: int = 0,
                 transform: str | None = None):
        from ...baselines.claude_openai_shim import ShimServer
        self.tokens_cap, self.usd_cap, self.max_tokens = int(tokens_cap), float(usd_cap), int(max_tokens)
        self.used = {'calls': 0, 'llm_tokens': 0, 'llm_usd': 0.0, 'refused': 0}
        self.accounting_complete = True
        self.exhausted: str | None = None
        self.overhead = int(getattr(backend, 'call_overhead_tokens', 4000))
        self.transform_name = transform
        self.transform = pv.TRANSFORMS[transform] if transform else None
        self.transform_log: list[dict] = []
        self.server = ShimServer(_CappedBackend(backend, self), port=port, usage_cb=self._usage, budget_check=self._check,
                                 log_path=log_path, calls_dir=calls_dir, call_timeout=lambda: call_timeout,
                                 max_output=self._max_output, served_name=getattr(backend, 'model', None))
        if self.transform:
            handle = self.server.handle

            def transformed(body: dict):
                body = dict(body)
                msgs, dropped = [], 0
                for m in body.get('messages') or []:
                    if isinstance(m, dict) and m.get('role') == 'user' and isinstance(m.get('content'), str):
                        text, n = self.transform(m['content'])
                        m, dropped = {**m, 'content': text}, dropped + n
                    msgs.append(m)
                body['messages'] = msgs
                self.transform_log.append({'utc': time.time(), 'transform': transform, 'lines_dropped': dropped})
                return handle(body)
            self.server.handle = transformed

    @staticmethod
    def _input_estimate(text: str) -> int:
        return int(len(text.encode('utf-8')) / 3.5) + 1

    def _usage(self, res):
        self.used['calls'] += 1
        self.used['llm_tokens'] += int(res.llm_tokens or 0)
        self.used['llm_usd'] += float(res.llm_usd or 0.0)
        self.accounting_complete &= bool(res.accounting_complete)

    def _left(self, text: str) -> int:
        return self.tokens_cap - self.used['llm_tokens'] - self._input_estimate(text) - self.overhead

    def worst_usd(self, text: str, out_tokens: int) -> float:
        return (self._input_estimate(text) + self.overhead) * USD_IN_BOUND + out_tokens * USD_OUT_BOUND

    def _check(self, text: str):
        why = None
        left = self._left(text)
        if left < 256:
            why = 'llm_tokens'
        elif self.used['llm_usd'] + self.worst_usd(text, min(self.max_tokens, left)) > self.usd_cap:
            why = 'llm_usd'
        if why:
            self.exhausted = self.exhausted or why
            self.used['refused'] += 1
        return why

    def _max_output(self, text: str):
        left = self._left(text)
        return min(self.max_tokens, left) if left >= 256 else None

    @property
    def port(self) -> int:
        return self.server.httpd.server_address[1]

    def start(self):
        self.server.start()
        return self

    def stop(self):
        self.server.stop()

    def snapshot(self) -> dict:
        return {**self.used, 'llm_usd': round(self.used['llm_usd'], 6), 'accounting_complete': self.accounting_complete,
                'exhausted': self.exhausted, 'tokens_cap': self.tokens_cap, 'usd_cap': self.usd_cap,
                'transform': self.transform_name, 'transformed_calls': len(self.transform_log)}


class TunnelProcess:
    """rtunnel.py <node> <remote_port> <local_port> as a child process (it reconnects with backoff on its own)."""

    def __init__(self, node: str, remote_port: int, local_port: int, log_path: Path, *, script: Path = RTUNNEL,
                 python: str = sys.executable, popen=subprocess.Popen):
        self.args = [python, str(script), str(int(node.replace('WINH', ''))), str(remote_port), str(local_port)]
        self.log_path, self.popen = Path(log_path), popen
        self.proc = None
        self.starts = 0

    def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, 'ab')
        flags = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
        self.proc = self.popen(self.args, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags)
        log.close()
        self.starts += 1
        return self

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except Exception:                  # noqa: BLE001
                self.proc.kill()
        self.proc = None

    def restart(self):
        self.stop()
        return self.start()


def default_backend():
    from ...agent.backends.claude_cli import ClaudeCLIBackend
    return ClaudeCLIBackend(model='claude-opus-5-5', effort='high')


# ---------------------------------------------------------------- driver helpers
def find_free_port(ops, lo: int = PORTS[0], hi: int = PORTS[1]) -> int | None:
    code = ('import socket\nfor p in range(%d, %d):\n    s = socket.socket()\n    try:\n        s.bind(("127.0.0.1", p))\n'
            '        print(p)\n        break\n    except OSError:\n        pass\n    finally:\n        s.close()\n') % (lo, hi)
    out = (ops.sh(f'python3 -c {q(code)}', timeout=60).get('stdout') or '').strip().splitlines()
    return int(out[-1]) if out and out[-1].isdigit() else None


def endpoint_ok(ops, port: int) -> bool:
    try:
        r = ops.sh(f"curl -s -m 8 -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{int(port)}/v1/models", timeout=30)
    except Exception:                           # noqa: BLE001
        return False
    return (r.get('stdout') or '').strip() == '200'


def read_stamps(text: str | None) -> list[dict]:
    out = []
    for line in (text or '').splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def trial_windows(stamps: list[dict], n_rows: int) -> dict:
    """{row index: (t_iter_dir_seen, t_row_seen)} from the node watcher's stamps (node clock)."""
    dirs = [s for s in stamps if s.get('kind') == 'iter_dir']
    rows = [s for s in stamps if s.get('kind') == 'rows']
    out = {}
    for i in range(n_rows):
        t_row = next((s['t'] for s in rows if s.get('rows', 0) >= i + 1), None)
        start = dirs[i]['t'] if i < len(dirs) else None
        out[i] = (start, t_row)
    return out


def classify_rows(rows: list[dict]) -> list[dict]:
    """OptiCo's RETRY rule (run_circle.sh): after a failed row whose loc_error verdict contains RETRY, the same
    parameters run again (at most 2 retries in a row) without counting as an iteration."""
    out, streak = [], 0
    for i, r in enumerate(rows):
        retry_of = None
        if i > 0:
            prev = rows[i - 1]
            if parse.row_result(prev) == 'fail' and 'RETRY' in str(prev.get('Reasons for failure', '')) and streak < 2 \
                    and pm.params_key(pm.params_from_row(prev)) == pm.params_key(pm.params_from_row(r)):
                retry_of = next((x['index'] for x in reversed(out) if x['retry_of'] is None), i - 1)
        streak = streak + 1 if retry_of is not None else 0
        out.append({'index': i, 'label': str(r.get('Log Director', '')).strip(), 'retry_of': retry_of,
                    'kind': 'default' if i == 0 else ('retry' if retry_of is not None else 'new')})
    return out


# ---------------------------------------------------------------- SearchTrace of a B-NAT loop (Codex W9 H-04)
def read_calls(path) -> list[dict]:
    """The shim's own call log (claude_openai_shim: one JSON row per backend call, 'utc' = PC epoch when it ended)."""
    try:
        return read_stamps(Path(path).read_text(encoding='utf-8'))
    except OSError:
        return []


def node_clock(ops) -> float:
    """The node's epoch clock (the driver brackets this call with its own clock to get the offset)."""
    return float((ops.sh('date +%s.%N', timeout=60).get('stdout') or '').split()[-1])


def _num(v) -> bool:
    return type(v) in (int, float) and math.isfinite(v)


def native_trace(*, receipts: list[dict], calls: list[dict], task, run_id: str, t_lease_pc: float, t_launch_pc: float,
                 t_end_pc: float, held_s: float, offset_s: float = 0.0, offset_info: dict | None = None,
                 shim_used: dict | None = None, row_seen_pc: dict | None = None, stop_reason: str = 'loop_finished',
                 best: dict | None = None, selected_config: dict | None = None, started_utc: str | None = None,
                 out_dir=None, driver: dict | None = None, world: int = wl.GPUS, extra: dict | None = None,
                 lease_released: bool = True) -> dict:
    """agent.search.SearchTrace (as a dict) of one B-NAT loop, for ext.common.trajectory (events, CTG, AUC_budget,
    arm_row). screens: one row per report.csv row in order -- default, new and RETRY trials, failed runs and D-OC-16
    sanitizer rejections included -- whose `cum` is the arm's cumulative cost when that row appeared on the node:
      screens        rows so far
      wall_seconds   since the loop launch (OptiCo's analysis and LLM time included)
      llm_tokens/usd the shim's exact usage of every call that ended before the row (its own call log)
      gpu_hours      the node lease: 8 GPUs x time since the lease (the node stays reserved during OptiCo's analysis)
    A row's own gpu_hours / wall_seconds are its trial window. `used` holds the arm's totals (LLM: the shim's ledger;
    GPU: 8 x lease held; wall: launch to loop end), so the cost after the last row (its analysis, a killed trial,
    collection) is the tail. Times are on the PC clock; node stamps move onto it with offset_s = node - PC, measured by
    the driver. Exact row times come from the node watcher's row stamp; without it the driver's first poll that saw the
    row (an upper bound) or the previous row (a lower bound) is used and the accounting is marked incomplete. LLM calls
    are attributed by the row stamp or, failing that, the trial's start stamp -- equally exact, because run_circle.sh
    makes no analysis call while a trial runs. The published-protocol result (best Result=success row) stays separate:
    extra.published_best_success_row and result.json; `selected` = that row, the configuration B-NAT's locked re-test
    runs (design §3.3)."""
    from ...agent.search import TRACE_SCHEMA, SearchTrace
    b = dict(task.budget)
    calls = sorted((c for c in calls if _num(c.get('utc'))), key=lambda c: float(c['utc']))
    times, t_prev, a_prev = [], float(t_launch_pc), float(t_launch_pc)
    for k, r in enumerate(receipts):
        win = list((r.get('native') or {}).get('node_window') or []) + [None, None]
        seen = (row_seen_pc or {}).get(k)
        if _num(win[1]):
            t, src = float(win[1]) - offset_s, 'node_stamp'
        elif _num(seen):
            t, src = float(seen), 'driver_poll'                     # upper bound: the first poll that saw the row
        else:
            t, src = t_prev, 'previous_row'                         # lower bound
        if _num(win[1]):
            ta, asrc = float(win[1]) - offset_s, 'row_stamp'
        elif _num(win[0]):
            ta, asrc = float(win[0]) - offset_s, 'trial_start_stamp'
        else:
            ta, asrc = t, src
        t_prev, a_prev = max(t_prev, t), max(a_prev, ta)
        times.append((t_prev, src, a_prev, asrc))

    def frac(v, cap):
        return v / cap if cap else 0.0

    offset_ok = bool((offset_info or {}).get('measured'))          # node stamps vs PC times need the measured offset
    unc = (offset_info or {}).get('uncertainty_s')
    near = max(5.0, 2.0 * unc) if _num(unc) else 5.0
    first_near = min((k for k, (_, _, ta, asrc) in enumerate(times) if asrc in ('row_stamp', 'trial_start_stamp')
                      for c in calls if abs(float(c['utc']) - ta) < near), default=None)

    screens, best_valid, n = [], None, 0
    tok, usd, complete, exact = 0, 0.0, True, True
    for k, (r, (t, src, ta, asrc)) in enumerate(zip(receipts, times)):
        while n < len(calls) and float(calls[n]['utc']) <= ta:
            tok += int(calls[n].get('llm_tokens') or 0)
            usd += float(calls[n].get('llm_usd') or 0.0)
            complete &= calls[n].get('accounting_complete') is True
            n += 1
        exact &= (src == 'node_stamp' and asrc in ('row_stamp', 'trial_start_stamp') and offset_ok
                  and (first_near is None or k < first_near))
        nat, metric, cost = r.get('native') or {}, r.get('metric') or {}, r.get('cost') or {}
        valid = r.get('valid') is True
        step = metric.get('value') if valid and _num(metric.get('value')) else None
        if step is not None and (best_valid is None or step < best_valid[1]):
            best_valid = (r.get('candidate_id'), step)
        cum = {'screens': k + 1, 'llm_tokens': tok, 'llm_usd': round(usd, 6),
               'gpu_hours': round(world * max(0.0, t - t_lease_pc) / 3600.0, 6),
               'wall_seconds': round(max(0.0, t - t_launch_pc), 3), 'accounting_complete': complete and exact}
        screens.append({
            'index': k, 'candidate_id': r.get('candidate_id'), 'kind': nat.get('kind'), 'attempt_id': r.get('attempt_id'),
            'valid': valid, 'step_time': step, 'reasons': (r.get('invalid_reasons') or [])[:4],
            'gpu_hours': round(float(cost.get('gpu_seconds') or 0.0) / 3600.0, 6), 'wall_seconds': cost.get('wall_seconds'),
            'cum': cum, 'fractions': {'screens': frac(k + 1, b.get('screens')), 'llm_tokens': frac(tok, b.get('llm_tokens')),
                                      'gpu_hours': frac(cum['gpu_hours'], b.get('gpu_hours')),
                                      'wall_seconds': frac(cum['wall_seconds'], b.get('wall_seconds'))},
            'best_valid': best_valid[0] if best_valid else None, 'best_step_time': best_valid[1] if best_valid else None,
            'judge_valid': r.get('judge_valid') is True, 'result': (r.get('judge') or {}).get('result'),
            'tflops_per_gpu': (metric.get('raw') or {}).get('tflops_per_gpu'), 'retry_of': nat.get('retry_of'),
            'iteration_label': nat.get('iteration_label'), 'sanitizer_rejected': bool(nat.get('sanitizer_events')),
            'guard_rejections': [g.get('id') for g in r.get('guard_rejections') or []],
            't_pc': round(t, 3), 'time_source': src, 'attribution_t_pc': round(ta, 3), 'attribution_source': asrc})
    log_tok = sum(int(c.get('llm_tokens') or 0) for c in calls)
    log_usd = sum(float(c.get('llm_usd') or 0.0) for c in calls)
    log_ok = all(c.get('accounting_complete') is True for c in calls)
    if shim_used and _num(shim_used.get('llm_tokens')):
        u_tok, u_usd = int(shim_used['llm_tokens']), float(shim_used.get('llm_usd') or 0.0)
        u_ok = shim_used.get('accounting_complete') is not False and log_ok
    else:
        u_tok, u_usd, u_ok = log_tok, log_usd, log_ok
    log_matches = (u_tok, round(u_usd, 6)) == (log_tok, round(log_usd, 6))
    last = screens[-1]['cum'] if screens else {'llm_tokens': 0, 'llm_usd': 0.0, 'gpu_hours': 0.0, 'wall_seconds': 0.0}
    used = {'screens': len(screens), 'llm_tokens': max(u_tok, last['llm_tokens']),
            'llm_usd': round(max(u_usd, last['llm_usd']), 6),
            'gpu_hours': round(max(world * float(held_s) / 3600.0, last['gpu_hours']), 6),
            'wall_seconds': round(max(float(t_end_pc) - float(t_launch_pc), last['wall_seconds'], 0.0), 3),
            'accounting_complete': bool(u_ok and exact and log_matches and lease_released)}
    tail = {k: round(used[k] - last[k], 6) for k in ('llm_tokens', 'llm_usd', 'gpu_hours', 'wall_seconds')}
    row_t = [t for t, src, _, _ in times if src == 'node_stamp']
    llm_calls = []
    for c in calls:
        tc = float(c['utc'])
        after = [s['index'] for s in screens if s['t_pc'] < tc]
        llm_calls.append({'call': c.get('call'), 't_pc': round(tc, 3), 'after_screen': after[-1] if after else None,
                          'llm_tokens': c.get('llm_tokens'), 'llm_usd': c.get('llm_usd'),
                          'accounting_complete': c.get('accounting_complete'), 'error': c.get('error'),
                          'requested_model': c.get('requested_model'), 'served': c.get('served'),
                          'near_row_boundary': any(abs(tc - t) < near for t in row_t)})
    sel = next((r for r in receipts if best and r.get('attempt_id') == best.get('attempt_id')), None)
    first = receipts[0] if receipts else None
    fmt = lambda p: pm.format_params(p) if isinstance(p, dict) and all(x in p for x in pm.KNOBS) else p  # noqa: E731
    trace = SearchTrace(
        schema=TRACE_SCHEMA, run_id=run_id, arm='B-NAT', task_id=task.id, mode='native', package_digest=None,
        driver=driver, judge='EnvJudge(optico)', budget=b, used=used, accounting_complete=used['accounting_complete'],
        stop_reason=stop_reason, selected=sel.get('candidate_id') if sel else None, selected_config=selected_config,
        selected_overlay={}, selected_step_time=(sel.get('metric') or {}).get('value') if sel else None,
        start=first.get('candidate_id') if first else None,
        start_step_time=(first.get('metric') or {}).get('value') if first else None,
        candidates=[{'id': x['candidate_id'], 'kind': x['kind'], 'parent': None, 'status': 'screened', 'valid': x['valid'],
                     'step_time': x['step_time'], 'attempt_id': x['attempt_id'], 'params': fmt(r.get('params'))}
                    for x, r in zip(screens, receipts)],
        screens=screens, llm_calls=llm_calls, skill_hits=[], memory_hits=[], memory_writes=[], tool_calls=[],
        invalid_proposals={}, verifier={}, started_utc=started_utc, ended_utc=utc_now(),
        out_dir=str(out_dir) if out_dir is not None else None,
        extra={'published_best_success_row': best,
               'selected_note': 'selected = the published-protocol best success row (OptiCo: highest TFLOPS among '
                                'Result=success rows), the configuration the B-NAT locked re-test runs; the best valid '
                                'screen is trajectory.best_search over `screens`',
               'cost_basis': {'gpu_hours': 'node lease: 8 GPUs x time since the lease; a screen row: its trial window',
                              'wall_seconds': 'since the loop launch (driver), OptiCo analysis and LLM time included',
                              'llm': 'shim call log; a call counts for a row when it ended before the row appeared'},
               'tail': tail, 'clock': {**(offset_info or {}), 'offset_node_minus_pc_s': offset_s, 't_lease_pc': t_lease_pc,
                                       't_launch_pc': t_launch_pc, 't_end_pc': t_end_pc},
               'row_time_sources': {x: sum(1 for _, y, _, _ in times if y == x) for x in sorted({y for _, y, _, _ in times})},
               'attribution_sources': {x: sum(1 for *_, y in times if y == x) for x in sorted({y for *_, y in times})},
               'accounting_notes': [m for m, bad in (
                   ('a row time is not the node watcher stamp (upper/lower bound)',
                    any(src != 'node_stamp' for _, src, _, _ in times)),
                   ('the node - PC clock offset was not measured', bool(times) and not offset_ok),
                   (f'an LLM call ended within {near:.0f} s of a row (attribution ambiguous)', first_near is not None),
                   ('the shim call log does not match the shim ledger', not log_matches),
                   ('the node lease was not released (GPU-hours still running)', not lease_released),
                   ('the shim reported incomplete accounting', not u_ok)) if bad],
               'calls_near_row_boundary': sum(1 for c in llm_calls if c['near_row_boundary']),
               'llm_log': {'calls': len(calls), 'llm_tokens': log_tok, 'llm_usd': round(log_usd, 6),
                           'matches_shim': log_matches},
               **(extra or {})})
    return trace.to_dict()


def trace_values(trace: dict) -> dict:
    """{row position: (iteration time, {'tflops': TFLOPS/GPU})} of the valid screens: pass as trajectory.events /
    arm_row(values=..., ref_raw={'tflops': the default's TFLOPS}) for y = ln(TFLOPS / TFLOPS_def) (design §0.3)."""
    return {k: (x['step_time'], {'tflops': x.get('tflops_per_gpu')}) for k, x in enumerate(trace.get('screens') or [])
            if x.get('valid') and _num(x.get('step_time'))}


def _positive(name: str, v, cap):
    if not (type(v) in (int, float) and math.isfinite(v) and v > 0):
        raise ValueError(f'{name} must be a positive finite number, got {v!r}')
    if v > cap:
        raise ValueError(f'{name} {v} exceeds the protocol maximum {cap} (design §3.5, D-OC-11)')
    return v


# ---------------------------------------------------------------- driver
class NativeDriver:
    def __init__(self, workload, ops, *, run_id: str, cycles: int = MAX_CYCLES, max_trials: int = MAX_TRIALS,
                 wall_cap_s: float | None = None, llm_tokens_cap: int = LLM_TOKENS, llm_usd_cap: float = LLM_USD,
                 backend=None, out_dir: str | Path, poll_s: float = 15.0, health_s: float = 30.0,
                 gb2: gb.GB2Policy | None = None, purpose: str = 'native', tunnel_factory=None, shim_factory=None,
                 clock=time.monotonic, sleep=time.sleep, log=None,
                 driver_label: str = 'claude-opus-5-5 via claude_openai_shim (requested model recorded per call)',
                 lost_node_s: float = 1800.0, port_finder=None, endpoint_check=None,
                 prompt_transform: str | None = 'strip_think_format', wallclock=time.time, clock_probe=None):
        self.w = wl.get(workload)
        self.ops, self.node = ops, ops.node
        self.run_id = run_id
        self.cycles = int(_positive('cycles', cycles, MAX_CYCLES))
        self.max_trials = int(_positive('max_trials', max_trials, MAX_TRIALS))
        self.wall_cap_s = float(_positive('wall_cap_s', self.w.wall_budget_s if wall_cap_s is None else wall_cap_s,
                                          self.w.wall_budget_s))
        self.llm_tokens_cap = int(_positive('llm_tokens_cap', llm_tokens_cap, LLM_TOKENS))
        self.llm_usd_cap = float(_positive('llm_usd_cap', llm_usd_cap, LLM_USD))
        if prompt_transform is not None and prompt_transform not in pv.TRANSFORMS:
            raise ValueError(f'unknown prompt transform {prompt_transform!r} (known: {sorted(pv.TRANSFORMS)})')
        self.prompt_transform = prompt_transform
        self.backend = backend
        self.out_dir = Path(out_dir)
        self.poll_s, self.health_s, self.lost_node_s = poll_s, health_s, lost_node_s
        self.gb2, self.purpose, self.driver_label = gb2, purpose, driver_label
        self.tunnel_factory = tunnel_factory or (lambda node, rport, lport, path: TunnelProcess(node, rport, lport, path))
        self.shim_factory = shim_factory or (lambda: BudgetedShim(self.backend or default_backend(),
                                                                  tokens_cap=self.llm_tokens_cap, usd_cap=self.llm_usd_cap,
                                                                  log_path=self.out_dir / 'shim_calls.jsonl',
                                                                  calls_dir=self.out_dir / 'calls',
                                                                  transform=self.prompt_transform))
        self.clock, self.sleep = clock, sleep
        self.wallclock, self.clock_probe = wallclock, clock_probe or node_clock     # PC epoch / node epoch (trace times)
        self.offsets: list[dict] = []                    # node - PC clock, measured at launch and at the loop end
        self.row_seen_pc: dict = {}                      # row index -> PC time the driver first saw it (fallback)
        self.trace: dict | None = None
        self.kill_unconfirmed = False
        self.event_errors: list[str] = []
        self.port_finder = port_finder or find_free_port
        self.endpoint_check = endpoint_check or endpoint_ok
        self.events: list[dict] = []
        self._log = log or (lambda msg: print(msg, flush=True))
        self.task = make_task(self.w, 'B-NAT')
        self.runner = OptiCoRunner(ops, self.w, out_dir=self.out_dir)
        self.deviations = dv.for_native(self.w, think_transform=prompt_transform == 'strip_think_format')
        self.runner.deviations = tuple(self.deviations)

    def event(self, kind: str, **kw):
        """Record an event (memory, events.jsonl, log). Diagnostics only: a failing disk or logger never raises here,
        so it can never interrupt the lease release or the kill confirmation."""
        e = {'utc': utc_now(), 't': round(self.clock(), 3), 'kind': kind, **kw}
        self.events.append(e)
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            with (self.out_dir / 'events.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(e, default=str) + '\n')
            self._log(f'[{e["utc"][11:19]}] {kind} ' + ' '.join(f'{k}={v}' for k, v in kw.items() if k != 'detail')[:300])
        except Exception as exc:                        # noqa: BLE001
            self.event_errors.append(f'{kind}: {type(exc).__name__}: {str(exc)[:200]}')

    def _try(self, what: str, fn):
        """Run one cleanup step; failures are recorded and never stop the later steps (the lease release last)."""
        try:
            return fn()
        except Exception as exc:                    # noqa: BLE001
            self.event(f'{what}_failed', detail=repr(exc))
            return None

    # ------------------------------------------------------------ run
    def run(self) -> dict:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        attempt_id = 'bnat-' + uuid.uuid4().hex[:10]
        rdir = f'{RUNS}/{self.run_id}/{attempt_id}'
        token = new_token('isxn')
        lease = GpuLease(self.ops, env='optico', gpus=list(range(wl.GPUS)), mode='own')
        utc_start, t0 = utc_now(), self.clock()
        summary = {'schema': 'infrarsi-ext-optico-native-v1', 'run_id': self.run_id, 'attempt_id': attempt_id,
                   'workload': self.w.key, 'task_id': self.w.task_id, 'node': self.node, 'run_dir': rdir,
                   'cycles': self.cycles, 'caps': {'max_trials_incl_retries': self.max_trials, 'wall_s': self.wall_cap_s,
                                                   'llm_tokens': self.llm_tokens_cap, 'llm_usd': self.llm_usd_cap},
                   'prompt_transform': self.prompt_transform, 'deviations': self.deviations, 'utc_start': utc_start}
        if self.prompt_transform is None:
            summary['note'] = 'verbatim OptiCo prompts (no D-OC-15): diagnosis only, Opus 5.5 refuses give_sugg'
        try:
            lease.acquire(self.run_id, attempt_id)
        except OccupiedError as exc:
            self.event('occupied', detail=str(exc))
            summary.update(status='occupied', detail=str(exc), occupancy=lease.report)
            return self._finish(summary, [])
        t_held, t_lease_pc = self.clock(), self.wallclock()           # the lease interval starts once it is held
        shim = tunnel = None
        stop_reason, loop_status, collected = None, None, {}
        t_launch = t_end = t_launch_pc = t_end_pc = None
        try:                                            # from here on the lease is released (or kept) in the finally
            try:
                self.event('lease', node=self.node, attempt_id=attempt_id)
                shim = self.shim_factory().start()
                rport = self.port_finder(self.ops)
                if rport is None:
                    raise RuntimeError(f'no free loopback port in {PORTS} on {self.node}')
                tunnel = self.tunnel_factory(self.node, rport, shim.port, self.out_dir / 'rtunnel.log').start()
                self.event('tunnel_start', remote_port=rport, local_port=shim.port)
                ok = False
                for _ in range(30):
                    if self.endpoint_check(self.ops, rport):
                        ok = True
                        break
                    self.sleep(3)
                if not ok:
                    raise RuntimeError(f'endpoint http://127.0.0.1:{rport}/v1 not reachable from {self.node} through the tunnel')
                self.event('endpoint_ok', port=rport)
                spec = {'MT': self.w.model_type, 'MS': self.w.model_size, 'CYCLES': self.cycles, 'PORT': rport,
                        'TOKEN': token, 'SRC': SRC, 'E': EXT_ROOT, 'V': VENV, 'MAX_ROWS': self.max_trials,
                        'MAX_WALL_S': int(self.wall_cap_s), 'DRIVER': self.driver_label}
                files = {f'{rdir}/{n}': t for n, t in node_files().items() if n != 'trial.sh'}
                files[f'{rdir}/spec.sh'] = spec_sh(spec)
                self.ops.mkdirs([rdir])
                self.ops.write_files(files)
                self.offsets.append(self._measure_offset('launch'))
                launch = self.ops.launch(f'{rdir}/loop.sh', f'{rdir}/loop.log', token=token)   # token from exec
                t_launch, t_launch_pc = self.clock(), self.wallclock()
                self.event('loop_launch', pid=launch.get('pid'), cycles=self.cycles, transform=self.prompt_transform)
                stop_reason, loop_status = self._poll(rdir, token, tunnel, shim, rport, t_launch)
                t_end, t_end_pc = self.clock(), self.wallclock()
                self.offsets.append(self._measure_offset('end'))
            except Exception as exc:                # noqa: BLE001 - recorded; cleanup below
                stop_reason = f'driver_error: {type(exc).__name__}: {exc}'
                self.event('driver_error', detail=repr(exc))
            finally:
                killed = self._kill_confirmed(token)
                if killed:
                    self.event('cleanup_kill', pids=killed[:20], n=len(killed))
                self._try('frozen_after', lambda: self.ops.sh(            # after-run hashes also when the loop was killed
                    f'[ -f {q(rdir)}/frozen_after.json ] || python3 {q(rdir)}/optico_node.py hash {q(SRC)} '
                    f'{q(rdir)}/frozen_after.json', timeout=120))
                self._try('copy_check_post', lambda: self.ops.sh(         # and the post-run copy check (a killed loop.sh
                    f'[ -f {q(rdir)}/copy_check_post.json ] || python3 {q(rdir)}/optico_node.py copycheck {q(SRC)} '
                    f'{q(rdir)}/workspace/code/OptiCo {q(rdir)}/workspace {q(rdir)}/copy_check_post.json loop '
                    f'{q(rdir)}/sanitize_monitor.sh', timeout=300))         # never wrote it), before the copy is deleted
                collected = self._try('collect', lambda: self._collect(rdir)) or {}
                for p in ('workspace', 'hf_home', 'bin'):
                    self._try('remove_tree', lambda p=p: self.ops.remove_tree(f'{rdir}/{p}'))
                if tunnel is not None:
                    self._try('tunnel_stop', tunnel.stop)
                if shim is not None:
                    self._try('shim_stop', shim.stop)
        finally:
            if self.kill_unconfirmed:                    # never hand back GPUs our loop may still be using
                released = []
                self.event('lease_kept', reason='token kill unconfirmed', paths=list(lease.acquired))
            else:
                released = self._try('lease_release', lease.release) or []
            held = max(0.0, self.clock() - t_held)                     # acquire returned -> release returned
            t_release_pc = self.wallclock()
            self.event('lease_released', paths=len(released), kept=list(lease.acquired),
                       failures=list(getattr(lease, 'release_failures', []) or []))
        summary.update(event_errors=self.event_errors[:20])
        summary.update(status='done' if stop_reason in (None, 'loop_finished') else 'stopped', stop_reason=stop_reason,
                       loop_status=loop_status, tunnel_starts=getattr(tunnel, 'starts', None),
                       llm=shim.snapshot() if shim else None, transform_log=getattr(shim, 'transform_log', None),
                       loop_wall_s=round(t_end - t_launch, 3) if t_launch and t_end else None,
                       held_s=round(held, 3), occupancy_gpu_seconds=round(wl.GPUS * held, 3),
                       driver_total_s=round(self.clock() - t0, 3), lease_kept=list(lease.acquired),
                       occupancy=lease.report and
                       {'problems': lease.report.get('problems'), 'lease': lease.report.get('lease')})
        receipts = self._receipts(collected, summary, utc_start)
        times = {'t_lease_pc': t_lease_pc, 't_launch_pc': t_launch_pc if t_launch_pc is not None else t_lease_pc,
                 't_loop_end_pc': t_end_pc, 't_end_pc': t_release_pc, 'held_s': held,
                 'lease_released': not lease.acquired}
        return self._finish(summary, receipts, collected, times=times)

    def _kill_confirmed(self, token: str) -> list | None:
        """kill_token with retries; when no call succeeds the kill is unconfirmed and the lease is kept."""
        self.kill_unconfirmed = True                    # pessimistic: cleared only by a kill call that got through
        for wait in (0.0, 5.0, 15.0, 45.0):
            if wait:
                self.sleep(wait)
            try:
                killed = self.ops.kill_token(token)
            except Exception as exc:                    # noqa: BLE001 - retried, then the lease is kept
                self.event('cleanup_kill_failed', detail=repr(exc)[:300])
                continue
            self.kill_unconfirmed = False
            return killed
        return None

    def _measure_offset(self, at: str) -> dict:
        """node clock - PC clock from one node call bracketed by the PC clock (uncertainty: half the round trip)."""
        try:
            a = self.wallclock()
            node_t = float(self.clock_probe(self.ops))
            b = self.wallclock()
            if not math.isfinite(node_t):
                raise ValueError(f'node clock {node_t!r}')
            return {'at': at, 'offset_s': round(node_t - (a + b) / 2.0, 3), 'uncertainty_s': round((b - a) / 2.0, 3),
                    'measured': True}
        except Exception as exc:                    # noqa: BLE001 - the trace then assumes synchronized clocks
            self.event('clock_offset_unmeasured', at=at, detail=repr(exc)[:300])
            return {'at': at, 'offset_s': None, 'uncertainty_s': None, 'measured': False}

    def _write_trace(self, summary: dict, receipts: list[dict], times: dict) -> dict:
        ok = [o for o in self.offsets if o.get('measured')]
        offset = sum(o['offset_s'] for o in ok) / len(ok) if ok else 0.0
        info = {'measurements': self.offsets, 'measured': bool(ok),
                'uncertainty_s': max(o['uncertainty_s'] for o in ok) if ok else None,
                'drift_s': round(ok[-1]['offset_s'] - ok[0]['offset_s'], 3) if len(ok) > 1 else None}
        best = summary.get('best_success_row')
        selected, error = None, None
        if best:
            rec = next((r for r in receipts if r.get('attempt_id') == best.get('attempt_id')), None)
            try:
                selected = pm.typed(rec['params'], strict=True)     # = arms.selection_from(result.json)
            except (ValueError, TypeError, KeyError) as exc:
                error = repr(exc)
        trace = native_trace(
            receipts=receipts, calls=read_calls(self.out_dir / 'shim_calls.jsonl'), task=self.task, run_id=self.run_id,
            t_lease_pc=times['t_lease_pc'], t_launch_pc=times['t_launch_pc'], t_end_pc=times['t_end_pc'],
            held_s=times['held_s'], offset_s=offset, offset_info={**info, 't_loop_end_pc': times.get('t_loop_end_pc')},
            shim_used=summary.get('llm'), lease_released=times.get('lease_released', True),
            row_seen_pc=self.row_seen_pc, stop_reason=summary.get('stop_reason') or 'loop_finished', best=best,
            selected_config=selected, started_utc=summary.get('utc_start'), out_dir=self.out_dir,
            driver={'kind': 'OptiCo run_circle.sh (native loop)', 'llm': self.driver_label,
                    'prompt_transform': self.prompt_transform, 'cycles': self.cycles},
            extra={'native_attempt': summary.get('attempt_id'), 'caps': summary.get('caps'), 'deviations': self.deviations,
                   'selected_config_error': error})
        path = self.out_dir / 'trace.json'
        path.write_text(json.dumps(trace, indent=1, default=str), encoding='utf-8')
        summary.update(trace=str(path), trace_used=trace['used'], trace_tail=trace['extra']['tail'],
                       calls_near_row_boundary=trace['extra']['calls_near_row_boundary'])
        self.trace = trace
        return trace

    def _poll(self, rdir: str, token: str, tunnel, shim, rport: int, t_launch: float):
        out = f'{rdir}/out/output_{self.w.model_type}_{self.w.model_size}_n1'
        last_health, bad_health, last_rows, last_ok = self.clock(), 0, -1, self.clock()
        while True:
            now = self.clock()
            try:
                files = self.ops.read_files([f'{rdir}/status.json', f'{out}/report.csv', f'{rdir}/stamps.jsonl'],
                                            max_bytes=1 << 20)
                last_ok = now
            except Exception as exc:                # noqa: BLE001 - transport trouble: the node watcher still enforces caps
                self.event('poll_error', detail=repr(exc)[:300])
                if now - last_ok > self.lost_node_s:
                    return 'driver_lost_node', None
                self.sleep(self.poll_s)
                continue
            dec = {k: (v.decode('utf-8', 'replace') if v is not None else None) for k, v in files.items()}
            status = None
            if dec[f'{rdir}/status.json']:
                try:
                    status = json.loads(dec[f'{rdir}/status.json'])
                except ValueError:
                    status = None
            _, rows = parse.read_report(dec[f'{out}/report.csv'] or '')
            stamps = read_stamps(dec[f'{rdir}/stamps.jsonl'])
            if len(rows) != last_rows:
                for k in range(max(0, last_rows), len(rows)):
                    self.row_seen_pc.setdefault(k, self.wallclock())
                last_rows = len(rows)
                r = rows[-1] if rows else {}
                self.event('rows', n=len(rows), last_result=r.get('Result'), last_tflops=r.get('TFLOPS'),
                           last_params=pm.format_params(pm.params_from_row(r)) if r else None, llm=shim.snapshot()['llm_usd'])
            node_stop = next((s for s in reversed(stamps) if s.get('kind') == 'stop'), None)
            if node_stop is not None:                      # the node watcher hit a cap and killed the loop (loop.sh
                return f'node_watch:{node_stop.get("reason")}', status   # itself survives and may still write status)
            if status is not None:
                return 'loop_finished', status
            if any(s.get('kind') == 'watch_end' for s in stamps):
                return 'loop_died', None                   # loop.sh gone without its status file
            reason = None
            if len(rows) >= self.max_trials:
                reason = 'max_trials'
            elif now - t_launch > self.wall_cap_s:
                reason = 'max_wall'
            elif shim.exhausted:
                reason = f'llm_budget:{shim.exhausted}'
            if reason:
                killed = self.ops.kill_token(token)
                self.event('stop', reason=reason, rows=len(rows), killed=len(killed))
                return reason, None
            if not tunnel.alive():
                tunnel.restart()
                self.event('tunnel_restart', why='process exited', starts=tunnel.starts)
            elif now - last_health >= self.health_s:
                last_health = now
                if self.endpoint_check(self.ops, rport):
                    bad_health = 0
                else:
                    bad_health += 1
                    self.event('endpoint_check_failed', n=bad_health)
                    if bad_health >= 2:
                        tunnel.restart()
                        bad_health = 0
                        self.event('tunnel_restart', why='endpoint unreachable twice', starts=tunnel.starts)
            self.sleep(self.poll_s)

    # ------------------------------------------------------------ collection
    def _collect(self, rdir: str) -> dict:
        out = f'{rdir}/out/output_{self.w.model_type}_{self.w.model_size}_n1'
        top = [f'{rdir}/{n}' for n in ('status.json', 'stamps.jsonl', 'run_circle.log', 'watch.log', 'loop.log', 'env.txt',
                                       'frozen_before.json', 'frozen_after.json', 'copy_check_pre.json',
                                       'copy_check_post.json', 'driver.txt', 'sanitizer.jsonl')]
        top += [f'{out}/{n}' for n in ('report.csv', 'parameters.txt', 'fixed_parameters.txt', 'suggestion.txt', 'loc_error.txt',
                                       'monitor_result.txt', 'monitor_error.txt')]
        texts = {k: (v.decode('utf-8', 'replace') if v is not None else None)
                 for k, v in self.ops.read_files(top, max_bytes=4 << 20).items()}
        iters = {}
        for name in self.ops.call('listdir', path=out) or []:
            d = f'{out}/{name}'
            if name == 'cache' or '.' in name:            # iteration directories are ITER_CUR labels (1, 2, ...)
                continue
            try:
                names = self.ops.call('listdir', path=d)
            except Exception:                           # noqa: BLE001 - a file, not an iteration directory
                continue
            if names is None:
                continue
            want = [f'{d}/{n}' for n in ('memory.txt', 'monitor_error.txt', 'parameters.txt', 'monitor_result.txt',
                                         'analyze_result.txt', 'loc_error.txt')]
            want += [f'{d}/{n}' for n in names if n.startswith('analyzer_log_')]
            files = {Path(k).name: (v.decode('utf-8', 'replace') if v is not None else None)
                     for k, v in self.ops.read_files(want, max_bytes=2 << 20).items()}
            files['megatron_0.txt'] = read_log(self.ops, f'{d}/megatron_log/0.txt')
            iters[name] = files
        local = self.out_dir / 'node'
        for k, v in texts.items():
            if v is not None:
                (local / Path(k).name).parent.mkdir(parents=True, exist_ok=True)
                (local / Path(k).name).write_text(v, encoding='utf-8')
        for name, files in iters.items():
            for fn, v in files.items():
                if v is not None:
                    (local / 'iterations' / name).mkdir(parents=True, exist_ok=True)
                    (local / 'iterations' / name / fn).write_text(v, encoding='utf-8')
        return {'top': {Path(k).name: v for k, v in texts.items()}, 'iterations': iters}

    # ------------------------------------------------------------ receipts
    @staticmethod
    def _json(text):
        try:
            return json.loads(text) if text else None
        except ValueError:
            return None

    def _receipts(self, collected: dict, summary: dict, utc_start: str) -> list[dict]:
        top = collected.get('top') or {}
        iters = collected.get('iterations') or {}
        _, rows = parse.read_report(top.get('report.csv') or '')
        kinds = classify_rows(rows)
        stamps = read_stamps(top.get('stamps.jsonl'))
        windows = trial_windows(stamps, len(rows))
        fb = self._json(top.get('frozen_before.json')) or {}
        fa = self._json(top.get('frozen_after.json')) or {}
        pre, post = self._json(top.get('copy_check_pre.json')), self._json(top.get('copy_check_post.json'))
        mismatch = sorted(set((pre or {}).get('mismatch') or []) | set((post or {}).get('mismatch') or []))
        if pre is None and rows:
            mismatch = ['<copy_check_pre.json missing>']
        elif post is None and rows:                    # fail closed: no evidence the copy stayed intact during the loop
            mismatch = mismatch + ['<copy_check_post.json missing>']
        tree_changed = bool(fb.get('tree_digest') and fa.get('tree_digest') and fb['tree_digest'] != fa['tree_digest'])
        sanitized = {}
        for e in read_events(top.get('sanitizer.jsonl')):
            sanitized.setdefault(str(e.get('iter')), []).append(e)
        policy = self.gb2
        if policy is None and rows:
            ref = gb.loss1(parse.iterations((iters.get(kinds[0]['label']) or {}).get('megatron_0.txt') or ''))
            policy = gb.GB2Policy(reference_loss1=ref, hard=False,
                                  rule='B-NAT: report only; reference = iteration-1 loss of this loop\'s default trial')
        receipts = []
        for k, row in zip(kinds, rows):
            files = iters.get(k['label']) or {}
            start, end = windows.get(k['index'], (None, None))
            wall = (end - start) if start is not None and end is not None and end >= start else None
            run = trial.evaluate(
                self.w, candidate_id=f'it{k["label"]}', row=row, log_text=files.get('megatron_0.txt') or '', requested=None,
                memory_text=files.get('memory.txt') or '', error_text=files.get('monitor_error.txt') or '',
                frozen_expected=dict(self.task.frozen_files), frozen_before=fb.get('files'), frozen_after=fa.get('files'),
                exit_code=0, timed_out=False, trial_wall_s=wall, phases={'trial_s': wall}, gb2=policy,
                copy_mismatch=mismatch or None,
                extra={'native': {'row_index': k['index'], 'kind': k['kind'], 'retry_of': k['retry_of'],
                                  'iteration_label': k['label'], 'analyze_result': (files.get('analyze_result.txt') or '').strip()
                                  or None, 'loc_error': files.get('loc_error.txt'), 'parameters_txt': files.get('parameters.txt'),
                                  'native_attempt': summary['attempt_id'], 'run_dir': summary['run_dir'],
                                  'node_window': [start, end], 'prompt_transform': self.prompt_transform,
                                  'sanitizer_events': sanitized.get(k['label'], [])},
                       'base_tree': {'before': fb.get('tree_digest'), 'after': fa.get('tree_digest'),
                                     'n_files': fb.get('n_files')},
                       'copy_check': {'pre': pre, 'post': post}})
            if tree_changed:
                detail = f'OptiCo base tree digest changed during the loop: {fb["tree_digest"][:16]} -> {fa["tree_digest"][:16]}'
                if run.judge_valid:
                    run.guards.reject('frozen_files', detail)
                else:
                    run.guards.report('frozen_files', would_reject=detail, note='judge invalid: not a veto')
            receipt = self.runner.receipt(self.task, run.extra.get('params'), None, self.purpose, self.run_id,
                                          f'{summary["attempt_id"]}-r{k["index"]:02d}', run.candidate_id, run, None,
                                          utc_start=utc_start, wall=wall or 0.0, held=wall or 0.0, occupancy=None)
            if wall is None:
                receipt['cost']['accounting_complete'] = False
            signing.sign(receipt)
            receipts.append(receipt)
        return receipts

    def _finish(self, summary: dict, receipts: list[dict], collected: dict | None = None, times: dict | None = None) -> dict:
        top = (collected or {}).get('top') or {}
        iters = (collected or {}).get('iterations') or {}
        kinds = classify_rows(parse.read_report(top.get('report.csv') or '')[1]) if top else []
        checks = []
        for k in kinds:
            files = iters.get(k['label']) or {}
            logs = [v for n, v in files.items() if n.startswith('analyzer_log_') and v]
            checks.append({'label': k['label'], 'kind': k['kind'],
                           'analyze_result': (files.get('analyze_result.txt') or '').strip() or None,
                           'give_sugg_response': any('run give_sugg' in t and 'response is:' in t for t in logs),
                           'change_para_response': any('run change_para' in t and 'response is:' in t for t in logs),
                           'loc_error_response': any('run loc_error' in t and 'response is:' in t for t in logs)})
        valid = [r for r in receipts if r.get('judge_valid')]
        best = max(valid, key=lambda r: (r['metric']['raw'] or {}).get('tflops_per_gpu') or 0, default=None)
        sanitizer = read_events(top.get('sanitizer.jsonl'))
        fmt = lambda p: pm.format_params(p) if isinstance(p, dict) and all(x in p for x in pm.KNOBS) else p  # noqa: E731
        summary.update(utc_end=utc_now(), n_rows=len(receipts),
                       n_new=sum(1 for k in kinds if k['kind'] == 'new'), n_retries=sum(1 for k in kinds if k['kind'] == 'retry'),
                       n_success=len(valid), n_valid=sum(1 for r in receipts if r.get('valid')),
                       n_guard_rejections=sum(1 for r in receipts if r.get('guard_rejections')),
                       iteration_checks=checks, sanitizer_fired=len(sanitizer), sanitizer_events=sanitizer[:10],
                       best_success_row=best and {'attempt_id': best['attempt_id'], 'params': fmt(best.get('params')),
                                                  'tflops_per_gpu': best['metric']['raw'].get('tflops_per_gpu'),
                                                  'valid': best['valid'], 'iter_time_s': best['metric']['value']},
                       receipts=[{'attempt_id': r['attempt_id'], 'candidate_id': r['candidate_id'], 'kind': r['native']['kind'],
                                  'params': fmt(r.get('params')), 'result': (r.get('judge') or {}).get('result'),
                                  'tflops_per_gpu': (r['metric'].get('raw') or {}).get('tflops_per_gpu'),
                                  'iter_time_s': r['metric']['value'], 'valid': r['valid'],
                                  'guard_rejections': [g['id'] for g in r.get('guard_rejections') or []],
                                  'gpu_seconds': r['cost']['gpu_seconds']} for r in receipts])
        if times is not None:                       # a loop ran (not an occupied node): SearchTrace for CTG / AUC
            self._write_trace(summary, receipts, times)
        (self.out_dir / 'receipts').mkdir(parents=True, exist_ok=True)
        for r in receipts:
            (self.out_dir / 'receipts' / f'{r["attempt_id"]}.json').write_text(json.dumps(r, indent=1), encoding='utf-8')
        (self.out_dir / 'result.json').write_text(json.dumps(summary, indent=1, default=str), encoding='utf-8')
        summary['receipt_objects'] = receipts
        return summary
