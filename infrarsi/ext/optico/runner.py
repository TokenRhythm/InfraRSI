"""OptiCoRunner: the ENV-B EnvRunner (EXT_INTERFACES §2) - one trial of OptiCo's own run_monitor.sh for one parameter
set on a leased 8-GPU node, turned into an ext-receipt-v1 by the ext.common.env_runner base.

Per attempt (directory /data/infrarsi/ext/runs/optico/<run_id>/<candidate_id>-<attempt_id>/):
  1. run(): validate_optico_params first; an invalid set never takes the node (signed receipt, zero GPU cost, D-OC-13);
  2. base EnvRunner.run: occupancy check (nvidia-smi compute apps + leases) and a node lease (owner infrarsi-ext);
  3. execute(): upload node/{trial.sh,optico_node.py}, spec.sh and parameters.txt (run_monitor.sh's own format), launch
     trial.sh in a new session (every process carries INFRARSI_TOKEN), poll status.json until done or the trial cap;
     kill whatever still carries the token, collect report.csv, the last node's Megatron log, memory.txt and the
     error file, delete the bulky workspace copy (logs, parameter files and hashes stay), evaluate (trial.evaluate);
  4. the base signs the receipt on this PC and stores it under out_dir/receipts/.
Driver-side code: no torch import.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ...contracts import new_attempt_id, utc_now
from ...harness import signing
from ..common.env_runner import EnvRunner
from ..common.remote_exec import EXT_RUNS, ContainmentError, NodeOps, new_token, q
from ..common.session_hooks import sha_text
from . import deviations as dv
from . import guards as gb
from . import params as pm
from . import parse
from . import trial
from . import workloads as wl

NODE_DIR = Path(__file__).resolve().parent / 'node'
EXT_ROOT = '/data/infrarsi/ext'
SRC = EXT_ROOT + '/src/optico'
VENV = '/mnt/base/qwen-sft/.venv-megatron-qwen35'
RUNS = EXT_RUNS + '/optico'
ITER_LABEL = 'trial'
WATCHDOG_EXTRA_S = 180              # trial.sh's own watchdog kills the token CAP_S + this after the start
KILL_AFTER_S = 30.0                 # trial.sh: timeout --kill-after
MATERIALIZE_SLACK_S = 300.0         # driver deadline allowance for materialization + base hashes (attempts without a deadline)
# Phase budgets of a bounded attempt, enforced by DeadlineOps, sized as remote calls x DESIGN_CALL_S. One remote call
# (an SSH exec of node_ext) took ~1-2 s on WINH6/WINH3 and up to 6.3 s on WINH4 on 2026-10-21 (bis_w3_a: upload of 2
# calls 12.6 s; the former 60 s pre-launch phase was missed twice there); DESIGN_CALL_S is about 2x that worst case.
DESIGN_CALL_S = 12.0
PRE_LAUNCH_S = 180.0                # 14 calls (lease 11, mkdirs, upload, launch) x 12 s = 168 s, rounded up
POST_S = 210.0                      # 17 calls (kill, 3-4 reads, 3 cleanups, 9 lease releases) x 12 s = 204 s
RELEASE_RESERVE_S = 110.0           # the end of POST_S kept for the 9 essential lease releases (9 x 12 s = 108 s)
INFRA_REFUSAL = 'infra_refusal'     # reason token of an attempt that ended before trial.sh was launched
MIN_CALL_S = 15.0                   # transport timeout floor of an essential call (a lease must never be dropped)
ESSENTIAL_CALLS = frozenset({'kill_token', 'release', 'release_status', 'uid_kill', 'uid_status'})
MIN_CAP_S = 60.0
MATERIALIZE_TYPICAL_S = 30.0        # measured 3-10 s: a W shorter than default trial + this would cut a default trial
LOG_HEAD_BYTES = 262144
LOG_FULL_MAX = 8 << 20


def node_files() -> dict:
    return {n: (NODE_DIR / n).read_text(encoding='utf-8') for n in ('trial.sh', 'loop.sh', 'optico_node.py',
                                                                    'sanitize_monitor.sh')}


def read_events(text: str | None) -> list[dict]:
    out = []
    for line in (text or '').splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def spec_sh(values: dict) -> str:
    return ''.join(f'{k}={q(v)}\n' for k, v in values.items())


def trial_cap_s(workload) -> float:
    """Per-trial wall cap: 4 x the measured default trial, at least 15 min (a hung NCCL job cannot eat the arm)."""
    return max(900.0, 4.0 * wl.get(workload).default_trial_s)


def read_log(ops, path: str) -> str:
    """The whole Megatron log, or head + tail when it is very large (the torchrun line is at the head)."""
    size = (ops.sizes([path]) or {}).get(path)
    if size is None:
        return ''
    if size <= LOG_FULL_MAX:
        return (ops.read_text(path) or '')
    head = ops.sh(f'head -c {LOG_HEAD_BYTES} {q(path)}', timeout=60).get('stdout') or ''
    tail = ops.read_text(path, max_bytes=LOG_FULL_MAX // 2) or ''
    return head + '\n...[middle of a large log not read]...\n' + tail


class AttemptDeadline(RuntimeError):
    """A bounded attempt reached a phase deadline; the remote call was not made."""


class DeadlineOps(NodeOps):
    """NodeOps proxy for one bounded attempt (Codex W9 H-05). Every remote call of the attempt -- the common GpuLease's
    acquire/release included, since the base runner builds its lease on self.ops -- gets its transport timeout cut to
    the time left before the current phase deadline. Past that deadline a non-essential call is refused
    (AttemptDeadline). An essential call (token kill, lease release, uid containment) is never refused and never gets
    less than MIN_CALL_S: a GPU lease or a live process must not be left behind to keep a time bound. So the bound is
    hard while the essential calls of the release reserve answer in RELEASE_RESERVE_S (typical: 1-2 s each); late
    essential calls are recorded (`overruns`), as are harness.remote's own transport retries inside one call (up to 3
    tries), through the receipt's actual wall against its worst case. An `acquire` whose answer is lost may have
    created the claim on the node: the proxy releases that path for this attempt (owner-checked) before re-raising."""

    def __init__(self, inner: NodeOps, clock):
        super().__init__(inner.node)
        self.inner, self.clock = inner, clock
        self.phase, self.deadline, self.essential_deadline = None, None, None
        self.overruns: list[dict] = []
        self.refused: list[dict] = []
        self.ambiguous_acquires: list[dict] = []
        self.launch_attempted = False                # once a launch call was issued, trial.sh may have run

    def __getattr__(self, name):                 # anything else the wrapped ops carries (remote, test hooks)
        return getattr(self.inner, name)

    def set_phase(self, phase: str, deadline: float, essential_deadline: float | None = None) -> None:
        self.phase, self.deadline = phase, float(deadline)
        self.essential_deadline = float(essential_deadline) if essential_deadline is not None else self.deadline

    def call(self, func, _timeout: float = 300, **kw):
        if self.deadline is not None:
            essential = func in ESSENTIAL_CALLS
            left = (self.essential_deadline if essential else self.deadline) - self.clock()
            if left <= 0:
                late = {'phase': self.phase, 'func': func, 'late_s': round(-left, 3)}
                if not essential:
                    self.refused.append(late)
                    raise AttemptDeadline(f'{self.phase} deadline passed before {func} ({-left:.0f} s late)')
                self.overruns.append(late)
            t = max(MIN_CALL_S, min(float(_timeout), left)) if essential else max(1.0, min(float(_timeout), left))
            if kw.get('timeout') is not None:                         # sh: the remote command's own timeout
                kw['timeout'] = max(1, int(min(float(kw['timeout']), t - 2)))
            _timeout = t
        if func == 'launch':
            self.launch_attempted = True
        try:
            return self.inner.call(func, _timeout=_timeout, **kw)
        except BaseException:
            if func == 'acquire' and isinstance(kw.get('lease'), dict):
                self._release_ambiguous(kw.get('path'), kw['lease'].get('attempt_id'))
            raise

    def _release_ambiguous(self, path, attempt_id) -> None:
        rec = {'path': path, 'attempt_id': attempt_id}
        try:
            rec['state'] = (self.inner.call('release_status', _timeout=MIN_CALL_S, path=path, attempt_id=attempt_id)
                            or {}).get('state')
        except Exception as exc:                                     # noqa: BLE001 - recorded: a possible stale claim
            rec['error'] = f'{type(exc).__name__}: {str(exc)[:200]}'
        self.ambiguous_acquires.append(rec)


class OptiCoRunner(EnvRunner):
    env = 'optico'

    def __init__(self, ops, workload, *, gpus=tuple(range(wl.GPUS)), gb2: gb.GB2Policy | None = None,
                 cap_s: float | None = None, poll_s: float = 10.0, grace_s: float = 120.0, keep_workspace: bool = False,
                 frozen_expected: dict | None = None, sleep=time.sleep, **kw):
        super().__init__(ops, gpus=gpus, **kw)
        self.workload = wl.get(workload)
        self.deviations = tuple(dv.for_runner(self.workload))
        self.gb2 = gb2
        self.cap_s = float(cap_s) if cap_s else trial_cap_s(self.workload)
        self.poll_s, self.grace_s, self.keep_workspace = poll_s, grace_s, keep_workspace
        self.frozen_expected = frozen_expected
        self.sleep = sleep
        self.history: list[dict] = []          # receipts of this runner (task view's performance table)
        self._attempt_cap: float | None = None  # this attempt's cap from SearchSession's bounded admission
        self._deadline: dict | None = None      # bounded attempt: start and end deadline (runner clock)
        self._kill_unconfirmed = False          # this attempt's token kill could not be confirmed

    def describe(self) -> dict:
        return {**super().describe(), 'workload': self.workload.key, 'cap_s': self.cap_s,
                'attempt_overhead_s': {'bounded': self.attempt_overhead_s(True), 'unbounded': self.attempt_overhead_s(False)},
                'gb2': (self.gb2 or gb.GB2Policy()).to_dict()}

    # ------------------------------------------------------------ bounded admission (agent.search.SearchSession)
    # The HarnessScreenRunner protocol: SearchSession admits a screen only if its worst case fits the GPU-hours and wall
    # left, and passes the admitted wall W back as `wall_cap`. Such an attempt runs on DeadlineOps with absolute phase
    # deadlines from its start t0: pre-launch (lease, upload, launch) by t0 + PRE_LAUNCH_S, else it is not launched;
    # trial.sh gets DEADLINE_S = W (the monitor ends W after trial.sh started, materialization included; monitor timeout
    # min(cap, W - elapsed)); the status wait ends W + kill-after + grace after the upload (one poll and one cut call
    # late at most); then kill, collection, cleanup and lease release by the end deadline t0 + W + overhead, the last
    # RELEASE_RESERVE_S kept for the essential calls. ENV-B's locked pairs are outside the arm budgets (design §3.5;
    # EnvTask.search_budget keeps no scoring reserve), so the pair reserve is zero: pair_wall_cap() = 0,
    # stop_grace_s = 0 and worst_case_gpu_hours(world, 0) = 0.
    stop_grace_s = 0.0

    def attempt_overhead_s(self, bounded: bool = True) -> float:
        """Wall of an attempt beyond its node budget (the admitted W; else the monitor cap, unbounded attempts): the sum
        of the phase deadlines DeadlineOps enforces for bounded attempts."""
        node = KILL_AFTER_S + self.grace_s if bounded else self.grace_s + MATERIALIZE_SLACK_S
        return PRE_LAUNCH_S + node + self.poll_s + MIN_CALL_S + POST_S

    def worst_case(self) -> float:
        """ext.common EnvRunner API: conservative wall of one attempt without an admitted cap (B-DEF, B-ED, locked)."""
        return self.worst_case_wall_s(None)

    def worst_case_wall_s(self, wall_cap: float | None = None) -> float:
        """Whole-attempt worst case: admitted wall W + overhead, or (no W) the monitor cap + the unbounded overhead."""
        if wall_cap is None:
            return self.cap_s + self.attempt_overhead_s(bounded=False)
        return float(wall_cap) + self.attempt_overhead_s(bounded=True)

    def worst_case_gpu_hours(self, world: int, wall_cap: float | None = None) -> float:
        if wall_cap is not None and float(wall_cap) <= 0:
            return 0.0                                                  # the (empty) locked-pair reserve
        return int(world) * self.worst_case_wall_s(wall_cap) / 3600.0

    def bounded_cap(self, gpu_hours_left: float, world: int, wall_left: float, expected_s: float) -> int | None:
        """Largest node wall W (materialization + monitor) whose worst case fits the GPU-hours and wall left. None when
        W would be shorter than the attempt needs: the session's expected screen wall, at least the workload's measured
        default trial + materialization (cheap rejected screens lower the session's mean) and MIN_CAP_S; a trial cut by
        its cap is wasted."""
        over = self.attempt_overhead_s(bounded=True)
        cap = min(self.cap_s + MATERIALIZE_SLACK_S, float(gpu_hours_left) * 3600.0 / max(1, int(world)) - over,
                  float(wall_left) - over)
        floor = self.workload.default_trial_s + MATERIALIZE_TYPICAL_S
        need = max(MIN_CAP_S, min(self.cap_s, max(float(expected_s or 0.0), floor)))
        return int(cap) if cap >= need else None

    def pair_wall_cap(self, expected_s: float) -> int:
        return 0

    # ------------------------------------------------------------ ScreenRunner
    def run(self, task, config: dict, overlay: dict | None, purpose: str, run_id: str, *, native: bool = False, **kw) -> dict:
        """native=True: a parameter set OptiCo's own loop produced (B-NAT locked re-test) runs as OptiCo ran it, without
        the non-native validator (D-OC-13 does not apply); the node-side D-OC-16 sanitizer still applies."""
        wall_cap = kw.pop('wall_cap', None)                       # SearchSession's admitted wall W (bounded attempt)
        self._attempt_cap = float(wall_cap) if wall_cap is not None else None
        self._deadline = None
        if native:
            norm, warnings = trial.complete_knobs(config), []
            errors = [] if norm is not None else [f'incomplete parameter set: {config}']
        else:
            norm, errors, warnings = pm.validate_optico_params(config, self.workload)
        if overlay:
            errors = errors + ['ENV-B has no file actions (knobs only)']
        if errors:
            receipt = self.not_launched(task, config, purpose, run_id, kw.get('candidate_id'), errors)
        else:
            receipt = self._attempt(task, norm, purpose, run_id, **kw)
            if warnings:
                receipt.setdefault('validator_warnings', warnings)
                signing.sign(receipt)
                self._store(receipt)
        self.history.append(receipt)
        return receipt

    def _attempt(self, task, config: dict, purpose: str, run_id: str, **kw) -> dict:
        """One launched attempt; a bounded one (admitted wall W) runs all its remote calls under DeadlineOps."""
        if self._attempt_cap is None:
            return super().run(task, config, None, purpose, run_id, **kw)
        inner, t0 = self.ops, self.clock()
        worst = self.worst_case_wall_s(self._attempt_cap)
        proxy = DeadlineOps(inner, self.clock)
        proxy.set_phase('pre_launch', t0 + PRE_LAUNCH_S, essential_deadline=t0 + worst)
        self._deadline = {'t0': t0, 'end': t0 + worst, 'proxy': proxy}
        self.ops = proxy
        try:
            receipt = super().run(task, config, None, purpose, run_id, **kw)
        finally:
            self.ops = inner
        actual = self.clock() - t0
        first = str((receipt.get('invalid_reasons') or [''])[0])
        if receipt.get('valid') is False and first.startswith('runner_error') and proxy.phase == 'pre_launch' \
                and not proxy.launch_attempted:
            # nothing was launched (pre-launch deadline or transport trouble in lease/upload): an infrastructure
            # refusal, not a failure of the candidate; its own reason token (occupied / occupied_sibling keep theirs)
            receipt['invalid_reasons'] = [f'{INFRA_REFUSAL}: the pre-launch phase ({PRE_LAUNCH_S:.0f} s: lease, upload, '
                                          f'launch) ended before trial.sh was launched; nothing ran on the GPUs'] + \
                list(receipt.get('invalid_reasons') or [])
            receipt['not_launched'] = 'pre_launch'
        receipt['deadline'] = {'admitted_wall_s': self._attempt_cap, 'worst_case_s': worst, 'actual_s': round(actual, 3),
                               'overrun_s': round(max(0.0, actual - worst), 3), 'late_essential_calls': proxy.overruns,
                               'refused_calls': proxy.refused, 'ambiguous_acquires': proxy.ambiguous_acquires}
        signing.sign(receipt)
        self._store(receipt)
        return receipt

    def contain(self) -> dict:
        """The base's uid containment (none for ENV-B) plus the attempt's token: an unconfirmed kill raises, so the base
        keeps the GPUs leased and invalidates the receipt instead of handing back a node our trial may still use."""
        out = super().contain()
        if self._kill_unconfirmed:
            raise ContainmentError(f'{self.node}: the attempt token kill is unconfirmed (transport); the GPUs stay '
                                   f'leased until someone checks the node')
        return out

    def not_launched(self, task, config, purpose, run_id, candidate_id, errors) -> dict:
        attempt_id = new_attempt_id()
        now = utc_now()
        r = {'schema': 'ext-receipt-v1', 'env': self.env, 'task_id': task.id, 'arm': getattr(task, 'arm', None),
             'run_id': run_id, 'attempt_id': attempt_id, 'candidate_id': candidate_id or attempt_id, 'purpose': purpose,
             'candidate_digest': None, 'frozen_digest': task.frozen_digest, 'task_digest': task.digest, 'node': None,
             'gpus': [], 'world_size': int(task.world_size), 'deviations': list(self.deviations),
             'valid': False, 'invalid_reasons': [f'invalid_proposal: {e}' for e in errors],
             'step_time': {'mean_window_max_rank_s': None},
             'metric': {'name': task.metric_name, 'value': None, 'direction': 'min', 'raw': {}},
             'judge': None, 'judge_valid': False, 'guard_rejections': [], 'guard_reports': [], 'phases': {'total': 0.0},
             'params': config if isinstance(config, dict) else None, 'not_launched': 'invalid_proposal',
             'utc_start': now, 'utc_end': now, 'wall_s': 0.0,
             'cost': {'gpu_seconds': 0.0, 'wall_seconds': 0.0, 'accounting_complete': True,
                      'basis': 'rejected by validate_optico_params before any launch (D-OC-13)'}}
        signing.sign(r)
        self._store(r)
        return r

    # ------------------------------------------------------------ one trial
    def execute(self, task, config: dict, overlay: dict, purpose: str, run_id: str, attempt_id: str, candidate_id: str,
                **kw):
        w, ops = self.workload, self.ops
        adir = f'{RUNS}/{run_id}/{candidate_id}-{attempt_id}'
        token = new_token('isxb')
        self._kill_unconfirmed = False
        # bounded attempt (SearchSession admission): node wall W = the admitted cap, the monitor timeout on the node is
        # min(per-trial cap, W - time already spent in trial.sh); otherwise the per-trial cap, as before
        bounded = self._attempt_cap is not None
        node_wall = float(self._attempt_cap) if bounded else None
        cap = min(self.cap_s, node_wall) if bounded else self.cap_s
        spec = {'MT': w.model_type, 'MS': w.model_size, 'TOKEN': token, 'CAP_S': int(cap), 'SRC': SRC,
                'E': EXT_ROOT, 'V': VENV, 'ITER_LABEL': ITER_LABEL,
                'WATCHDOG_S': int(node_wall if bounded else cap) + WATCHDOG_EXTRA_S}
        if bounded:
            spec['DEADLINE_S'] = int(node_wall)
        files = {f'{adir}/{n}': t for n, t in node_files().items() if n != 'loop.sh'}
        files[f'{adir}/spec.sh'] = spec_sh(spec)
        files[f'{adir}/parameters.txt'] = pm.encode_params(config)
        t0 = self.clock()
        ops.mkdirs([adir])
        ops.write_files(files)
        t_up = self.clock()
        deadline = t_up + (node_wall + KILL_AFTER_S + self.grace_s if bounded      # materialization inside W
                           else cap + self.grace_s + MATERIALIZE_SLACK_S)
        dl = self._deadline if bounded else None
        status, overdue, launch = None, False, {}
        try:                                     # the launch inside: one that timed out may still have started
            launch = ops.launch(f'{adir}/trial.sh', f'{adir}/trial.log', token=token)   # token from exec
            self.log(f'{run_id}/{candidate_id}: launched {pm.format_params(config)} on {self.node} pid {launch.get("pid")}')
            if dl:                               # the status reads may run one poll and one cut call past `deadline`
                dl['proxy'].set_phase('trial', deadline + self.poll_s + MIN_CALL_S, essential_deadline=dl['end'])
            while True:
                status = ops.read_json(f'{adir}/status.json')
                if isinstance(status, dict) and 'rc' in status:
                    break
                if self.clock() > deadline:
                    overdue = True
                    break
                self.sleep(self.poll_s)
        finally:                                                     # never hand back the node with our processes on it
            t_done = self.clock()
            if dl:                               # kill, collect, clean up; the last RELEASE_RESERVE_S for the release
                dl['proxy'].set_phase('post', dl['end'] - RELEASE_RESERVE_S, essential_deadline=dl['end'])
            killed = self._kill_token(token, t_up + (node_wall if bounded else cap) + WATCHDOG_EXTRA_S + 60.0)
        out = f'{adir}/out'
        log_path = f'{out}/{ITER_LABEL}/megatron_log/0.txt'          # last node of WORLD_SIZE=1 (run_monitor.sh)
        texts = {k: (b.decode('utf-8', 'replace') if b is not None else None) for k, b in ops.read_files([
            f'{out}/report.csv', f'{out}/{ITER_LABEL}/memory.txt', f'{out}/{ITER_LABEL}/monitor_error.txt',
            f'{out}/{ITER_LABEL}/parameters.txt', f'{out}/fixed_parameters.txt', f'{out}/monitor_result.txt',
            f'{adir}/frozen_before.json', f'{adir}/frozen_after.json', f'{adir}/copy_check.json', f'{adir}/env.txt',
            f'{adir}/trial.log', f'{adir}/monitor_stdout.log', f'{adir}/sanitizer.jsonl'], max_bytes=1 << 20).items()}
        log_text = read_log(ops, log_path)
        t_collect = self.clock()
        if not self.keep_workspace:
            for p in ('workspace', 'hf_home', 'bin'):
                try:
                    ops.remove_tree(f'{adir}/{p}')
                except Exception as exc:                             # noqa: BLE001 - reported, not fatal
                    self.log(f'{adir}/{p}: cleanup failed: {exc}')
        _, rows = parse.read_report(texts[f'{out}/report.csv'] or '')
        row = rows[-1] if rows else None

        def js(path):
            try:
                return json.loads(texts[path]) if texts.get(path) else None
            except ValueError:
                return None
        before, after, copy = js(f'{adir}/frozen_before.json'), js(f'{adir}/frozen_after.json'), js(f'{adir}/copy_check.json')
        expected = self.frozen_expected if self.frozen_expected is not None else dict(task.frozen_files)
        st = status if isinstance(status, dict) else {}
        monitor_s = (st['t_monitor_end'] - st['t_materialized']) if {'t_monitor_end', 't_materialized'} <= set(st) else None
        phases = {'upload_s': round(t_up - t0, 3), 'materialize_s': round(st['t_materialized'] - st['t_start'], 3)
                  if {'t_materialized', 't_start'} <= set(st) else None,
                  'monitor_s': round(monitor_s, 3) if monitor_s is not None else None,
                  'wait_s': round(t_done - t_up, 3), 'collect_s': round(t_collect - t_done, 3)}
        run = trial.evaluate(
            w, candidate_id=candidate_id, row=row, log_text=log_text, requested=config,
            memory_text=texts[f'{out}/{ITER_LABEL}/memory.txt'] or '', error_text=texts[f'{out}/{ITER_LABEL}/monitor_error.txt'] or '',
            frozen_expected=expected, frozen_before=(before or {}).get('files'), frozen_after=(after or {}).get('files'),
            exit_code=st.get('rc') if not overdue else None, timed_out=bool(st.get('timed_out')) or overdue,
            wall_cap_s=cap, trial_wall_s=monitor_s, phases=phases, gb2=self.gb2,
            copy_mismatch=(copy or {}).get('mismatch') if copy else (['<copy_check.json missing>'] if status else None),
            extra={'runner': {'attempt_dir': adir, 'token': token, 'launch': launch, 'killed_after': killed,
                              'status': status, 'overdue': overdue, 'cap_s': self.cap_s, 'cap_s_effective': cap,
                              'admitted_wall_s': node_wall,
                              'worst_case_wall_s': self.worst_case_wall_s(node_wall),
                              'trial_sh_sha256': sha_text(files[f'{adir}/trial.sh']),
                              'parameters_txt': files[f'{adir}/parameters.txt'],
                              'parameters_txt_after': texts[f'{out}/{ITER_LABEL}/parameters.txt'],
                              'fixed_parameters': texts[f'{out}/fixed_parameters.txt'],
                              'env_txt': texts[f'{adir}/env.txt']},
                   'base_tree': {'before': (before or {}).get('tree_digest'), 'after': (after or {}).get('tree_digest'),
                                 'n_files': (before or {}).get('n_files')},
                   'copy_check': copy,
                   'sanitizer': {'fired': len(read_events(texts[f'{adir}/sanitizer.jsonl'])),
                                 'events': read_events(texts[f'{adir}/sanitizer.jsonl'])[:5]}})
        if before and after and before.get('tree_digest') != after.get('tree_digest'):
            detail = (f'OptiCo base tree digest changed during the run: {str(before.get("tree_digest"))[:16]} -> '
                      f'{str(after.get("tree_digest"))[:16]}')
            if run.judge_valid:
                run.guards.reject('frozen_files', detail)
            else:
                run.guards.report('frozen_files', would_reject=detail, note='judge invalid: not a veto')
        if self.out_dir:
            self._save_texts(candidate_id, attempt_id, {**{Path(k).name: v for k, v in texts.items()}, 'megatron_0.txt': log_text})
        return run

    def _kill_token(self, token: str, watchdog_deadline: float) -> list:
        """Kill what still carries the attempt's token (monitor_gpu.sh, orphaned workers). Transport trouble: retry with
        backoff, then wait for the node-side watchdog of trial.sh (CAP_S + WATCHDOG_EXTRA_S after the start) and try
        once more, so the base never releases the lease while our processes may still hold the GPUs."""
        self._kill_unconfirmed = True                    # pessimistic: cleared only by a kill call that got through
        for wait in (0.0, 5.0, 15.0, 45.0):
            if wait:
                self.sleep(wait)
            try:
                killed = self.ops.kill_token(token)
            except Exception as exc:                                 # noqa: BLE001
                self._note(f'kill_token {token} failed: {exc!r}')
                continue
            self._kill_unconfirmed = False
            return killed
        while self.clock() < watchdog_deadline:
            self.sleep(min(60.0, max(1.0, watchdog_deadline - self.clock())))
        try:
            killed = self.ops.kill_token(token)
        except Exception as exc:                                     # noqa: BLE001 - containment then keeps the lease
            self._note(f'kill_token {token} failed after the watchdog deadline: {exc!r}')
            return ['unconfirmed: the lease is kept']
        self._kill_unconfirmed = False
        return killed

    def _note(self, msg: str) -> None:
        """Best-effort log line: a failing logger must never interrupt cleanup."""
        try:
            self.log(msg)
        except Exception:                                            # noqa: BLE001
            pass

    def _save_texts(self, candidate_id: str, attempt_id: str, texts: dict) -> None:
        d = self.out_dir / 'attempts' / f'{candidate_id}-{attempt_id}'
        d.mkdir(parents=True, exist_ok=True)
        for name, text in texts.items():
            if text is not None:
                (d / name).write_text(text, encoding='utf-8')
