"""Verifier: the judge of project runs (revamp design A). Objective = speed only; one human knob, `tolerance`; the
correctness checklist decides validity and its failures go back to the search as feedback.

    Verifier(tolerance)(receipt, parent, reference) -> agent.judge.Verdict          (screen items)
    Verifier(tolerance).promotion(candidate, reference, rescore=..., resume=..., tests=...) -> list of item results

Receipts are `infrarsi-project-receipt-v1` (project/runner.py); the verifier reads only their `evidence` block,
compares it with the reference run's, and never trusts a receipt whose HMAC (harness.signing) does not verify.
`tolerance` bounds the relative loss deviation from the reference: per update when both runs provably saw the same
global batches, else on the scoring-window mean; the first-update gradient norm is held to the same bound. A probe or
short run keeps the job's {steps} (its schedule horizon) and is stopped after `stop_after` updates: `completes` accepts
such a run, and every comparison is at the updates it ran. The manifest tolerance is for the window mean; a run stopped
before the window's end compares a mean of a few updates, which carries the loss's own per-update noise (a diffusion
loss swings with the sampled timestep), so its loss is held to the tolerance or, if larger, LOSS_NOISE_Z times the
noise of such a mean read from the reference's per-update losses (`loss_noise`); a full run keeps the manifest tolerance
unless the job's draws differ between runs: when the reference's own repeat (`Verifier.repeat`, the doctor's run of the
same configuration) is off by more than the tolerance per update (Wan through DiffSynth: 40 % per update and 23 % on the
window mean between two runs of one configuration, D2-R204d), two full runs compare means of different draws too, and a
full run's window mean is held to the same LOSS_NOISE_Z x noise of a mean of its updates. Seeded jobs (every other trial,
two diffusion losses among them) repeat to 0.2 % per update and keep the manifest tolerance.
"""
from __future__ import annotations

import math
import statistics

from ..agent.judge import Verdict
from ..harness import signing
from . import checklist as C

SCHEMA = 'infrarsi-project-receipt-v1'
LR_REL = 1e-6                   # learning rates are equal when they agree to this relative precision
FROZEN_SCALARS_RUN = 2          # this many adjacent updates with bitwise-equal (loss, grad norm) means a skipped update
LOSS_NOISE_Z = 3.0              # a short run's loss mean is held to this many sd of its noise when that exceeds the tolerance


def _num(x):
    return float(x) if type(x) in (int, float) and math.isfinite(x) else None


def _pos(x):
    """A usable time: finite and positive (a zero or negative step time is no measurement)."""
    v = _num(x)
    return v if v is not None and v > 0 else None


def _signed(a: float, b: float) -> float:
    """a's relative deviation from b with its sign, a / b - 1 (U-1-21: a report says +1.8 %, not 1.8 %)."""
    if a == b:
        return 0.0
    return a / b - 1.0 if b else math.copysign(math.inf, a)


def _rel(a: float, b: float) -> float:
    return abs(_signed(a, b))


def _steps(ev: dict, key: str) -> dict:
    """{update: value} of one per-step signal (finite numbers or strings), first occurrence wins."""
    out = {}
    for row in ev.get('per_step') or []:
        s, v = row.get('step'), row.get(key)
        if type(s) is int and s not in out and v is not None and (type(v) is str or _num(v) is not None):
            out[s] = v
    return out


def evidence(receipt) -> dict:
    ev = receipt.get('evidence') if isinstance(receipt, dict) else None
    return ev if isinstance(ev, dict) else {}


def _expected(ref: dict, key: str, ev: dict) -> list[int]:
    """Updates where the reference reports `key` and the candidate ran (the candidate must report it there too)."""
    done = ev.get('steps_completed') if type(ev.get('steps_completed')) is int else 0
    return sorted(s for s in _steps(ref, key) if s <= done)


def margin_of(ev: dict, floor: float = 0.03, k: float = 3.0) -> float:
    """Noise margin of a speed comparison: max(floor, k x RSD of the reference's per-update times in the window)."""
    rsd = _num(ev.get('rsd_window'))
    return max(floor, k * rsd) if rsd is not None else floor


def loss_noise(losses: dict, lo: int, hi: int) -> float | None:
    """Relative per-update noise of a run's loss over updates lo..hi: the sample sd of the differences between
    consecutive updates / sqrt 2, over the mean. A smooth trend has near-constant differences and adds almost nothing
    (the toy's exponential decay: 0.002; sft_loop's cross-entropy on 8K batches: 0.10), a loss that swings with the
    sampled batch or timestep shows in every difference (Wan 2.1 through DiffSynth, D2-R204c: 0.37, per-update losses
    0.03..0.10 around 0.057, while two full runs agreed to 1.1 % on the window mean and 42 % on their first two updates).
    None with fewer than three consecutive updates or a non-positive mean."""
    xs = [losses[s] for s in range(lo, hi + 1) if s in losses]
    d = [losses[s] - losses[s - 1] for s in range(lo + 1, hi + 1) if s in losses and s - 1 in losses]
    m = sum(xs) / len(xs) if xs else 0.0
    return statistics.stdev(d) / math.sqrt(2) / m if len(d) >= 2 and m > 0 else None


def loss_repeat_note(rep: dict | None, tolerance: float, doctor: bool = False) -> str | None:
    """One line on the unpaired loss check of a job whose per-update loss noise would move a window mean past the
    manifest tolerance (`Verifier.loss_repeat`): what full runs are held to and why, or what is missing to decide.
    None for a quiet loss or a job that compares per update (the manifest tolerance holds either way)."""
    if not rep or rep.get('paired') or rep['tolerance'] <= tolerance:
        return None
    noise, n, tol, off = rep['noise'], rep['updates'], rep['tolerance'], rep['repeat']
    more = '; more updates per run (workload.steps, window) tighten it'
    # U-2-4: "draws" said once in plain words (the batches and the noise a run sees), "loss changes" not "breaks"
    if doctor or rep['draws_vary'] is None:
        who = 'this run' if doctor else '`infrarsi doctor` within a day'
        return (f'the loss varies {noise:.1%} between updates and the data order is not verified: if runs draw differently '
                f'(two runs of one configuration see other batches or noise), a {n}-update window mean resolves only loss '
                f'changes above {tol:.1%}; `run` compares its reference with {who} per update to tell'
                + ('' if doctor else f' - without it full runs are held to the manifest {tolerance:.1%}') + more)
    if rep['draws_vary']:
        return (f'this job\'s draws differ between runs (two runs of one configuration see other batches or noise: the '
                f'reference\'s own repeat is {off:.1%} off per update): an unpaired {n}-update window mean is held to '
                f'{tol:.1%} ({LOSS_NOISE_Z:g} x the reference\'s {noise:.1%} per-update loss noise), not the manifest '
                f'{tolerance:.1%}' + more)
    return (f'the loss varies {noise:.1%} between updates, but runs repeat their draws (a repeat of the reference agrees '
            f'within tolerance per update, {off:.2%} off), so the manifest {tolerance:.1%} holds for the window mean')


class Verifier:
    def __init__(self, tolerance: float = 0.03, *, required=C.DEFAULT_REQUIRED, disabled=(), memory_cap_gib=None,
                 timing_tolerance: float = 0.03, margin_floor: float = 0.03, require_signature: bool = True,
                 repeat: dict | None = None):
        """repeat: the evidence of another run of the reference configuration (`infrarsi doctor`'s, set by the flow once
        the reference ran): the loss check reads from it whether the job's draws differ between runs."""
        if not (type(tolerance) in (int, float) and 0 <= tolerance < 1):
            raise ValueError(f'tolerance must be in [0, 1), got {tolerance!r}')
        self.tolerance, self.required, self.disabled = float(tolerance), tuple(required), set(disabled)
        self.memory_cap_gib, self.timing_tolerance = memory_cap_gib, timing_tolerance
        self.margin_floor, self.require_signature = margin_floor, require_signature
        self.repeat = repeat

    def describe(self) -> dict:
        return {'kind': 'Verifier', 'tolerance': self.tolerance, 'required': list(self.required),
                'disabled': sorted(self.disabled), 'memory_cap_gib': self.memory_cap_gib}

    # ------------------------------------------------------------ screen items
    def screen_checks(self, ev: dict, ref: dict | None) -> list[dict]:
        """ref None: this run is the reference (comparisons pass trivially)."""
        out = [self._completes(ev), self._no_errors(ev), self._memory(ev)]
        data = self._same_data(ev, ref)
        out += [self._same_workload(ev, ref), data, self._optimizer_steps(ev, ref), self._same_model(ev, ref),
                self._frozen(ev)]
        paired = data['status'] == C.PASS
        out += [self._loss(ev, ref, paired), self._grad_norm(ev, ref, data['status']), self._timing(ev, ref),
                self._precision(ev, ref)]
        return [C.result(r['id'], C.OFF, 'disabled in the manifest') if r['id'] in self.disabled else r for r in out]

    def workload_unverifiable(self, ev: dict) -> bool:
        """No candidate could pass the workload check against this reference (D2-R211, D2-R204b): same_workload is
        required and, judged on the reference's own evidence (screen_checks of the run against itself, so a disabled
        check is off), unchecked."""
        r = next(x for x in self.screen_checks(ev, ev) if x['id'] == 'same_workload')
        return r['status'] == C.UNCHECKED and r['id'] in self.required

    def repeat_disagreement(self, ref: dict, lo: int | None = None, hi: int | None = None) -> float | None:
        """How far the reference's own repeat (`self.repeat`) is from it per update: the median relative deviation over
        the window. Beyond the tolerance, the two runs of one configuration did not see the same draws (Wan through
        DiffSynth: 40-52 % at every update, D2-R204b/c/d); seeded jobs repeat to 0.2 % (sft_loop, Megatron, DeepSpeed,
        HF Trainer, diffusers and FLUX - a diffusion loss repeats too). None without a repeat or fewer than 3 common
        finite updates."""
        if not isinstance(self.repeat, dict):
            return None
        a, b = _steps(self.repeat, 'loss'), _steps(ref, 'loss')
        if not a or not b:
            return None
        if lo is None or hi is None:
            lo, hi = ref.get('window') or [min(b), max(b)]
        devs = [_rel(float(a[s]), float(b[s])) for s in range(lo, hi + 1)
                if s in a and s in b and type(a[s]) is not str and type(b[s]) is not str and float(b[s]) > 0]
        return statistics.median(devs) if len(devs) >= 3 else None

    def loss_repeat(self, ref: dict) -> dict:
        """What the unpaired loss check knows about a reference: `noise` (its per-update loss noise, `loss_noise`),
        `updates` (the window's), `tolerance` (what a full run's window mean is held to when runs draw differently: the
        manifest tolerance or LOSS_NOISE_Z x the noise of such a mean, the larger), `repeat` (its own repeat's median
        per-update deviation; None without one), `draws_vary` (that repeat beyond the tolerance; None when unknown) and
        `paired` (the reference logs a per-update data digest: candidates compare per update instead)."""
        b = {s: float(v) for s, v in _steps(ref, 'loss').items() if type(v) is not str}
        lo, hi = ref.get('window') or ([min(b), max(b)] if b else [0, 0])
        n = sum(1 for s in b if lo <= s <= hi)
        noise = loss_noise(b, lo, hi) if n else None
        tol = max(self.tolerance, LOSS_NOISE_Z * noise * math.sqrt(2 / n)) if noise else self.tolerance
        rep = self.repeat_disagreement(ref, lo, hi)
        return {'noise': noise, 'updates': n, 'tolerance': tol, 'repeat': rep,
                'draws_vary': None if rep is None else rep > self.tolerance, 'paired': bool(_steps(ref, 'data'))}

    def _completes(self, ev):
        n, done, code = ev.get('steps'), ev.get('steps_completed'), ev.get('exit_code')
        if ev.get('timed_out'):
            return C.result('completes', C.FAIL, f'killed at the timeout after {done}/{n} updates')
        k = ev.get('stop_after')
        if ev.get('planned_stop') and type(k) is int and type(done) is int and done >= k:   # the exit code is the stop's
            return C.result('completes', C.PASS, f'{done} updates, stopped as planned after {k} (schedule horizon {n})')
        err = (ev.get('events') or {}).get('error')     # the log's own first error line, when there is one
        fatal = (ev.get('events') or {}).get('fatal')   # the first line after which the job did not go on
        fs = ev.get('fatal_stop')
        if fs:                                          # stopped by the runner's fatal-line watch (no update since)
            return C.result('completes', C.FAIL, f'stopped {fs.get("grace_s", 0):.0f} s after a fatal line with no update '
                            f'since, {done}/{n} updates; the log: {fs.get("line")}', hint=C.fault_hint(fs.get('line')))
        if code != 0:
            more = f'; first fatal line: {fatal}' if fatal and not (err and err in fatal) else ''
            return C.result('completes', C.FAIL, f'exit code {code}, {done}/{n} updates' + (f'; the log: {err}' if err else '')
                            + more, hint=C.fault_hint(err) or C.fault_hint(fatal))
        if type(n) is not int or done != n:
            return C.result('completes', C.FAIL, f'{done}/{n} updates logged')
        return C.result('completes', C.PASS, f'{done}/{n} updates, exit 0')

    def _no_errors(self, ev):
        e = ev.get('events') or {}
        bad = []
        loss, gn = _steps(ev, 'loss'), _steps(ev, 'grad_norm')
        nonfinite = sorted(set(e.get('nan_inf_steps') or []) | {s for v in (loss, gn) for s, x in v.items() if type(x) is str})
        if nonfinite:
            bad.append(f'NaN/Inf loss or gradient norm at updates {nonfinite[:6]}')
        # collective lines in a run that logged every update by itself and exited 0, and nothing else wrong: the job's
        # race at its end (D2-R204f, Wan through accelerate: 3 of 5 promotion re-runs - two of the reference
        # configuration - and the doctor run; a pending all-reduce raised as the ranks finished), reported, not a
        # failure: a collective error that breaks a run stops or stalls it (completes), or shows in its loss
        n, done = ev.get('steps'), ev.get('steps_completed')
        ran_out = (ev.get('exit_code') == 0 and type(n) is int and type(done) is int and done == n
                   and not ev.get('stop_after') and not nonfinite and not e.get('oom') and not e.get('traceback'))
        # only lines that arrived after the last update are the end's (a line in the middle of a run that went on is a
        # failure the job swallowed); evidence without the position (older receipts) is read as the end's
        after_last = e.get('collective_at_end', e.get('collective'))
        at_end = e.get('collective') if ran_out and e.get('collective') and after_last == e.get('collective') else None
        for key, what in (('oom', 'out-of-memory'), ('collective', 'collective/NCCL error'), ('traceback', 'traceback')):
            if e.get(key) and not (key == 'collective' and at_end):
                bad.append(f'{e[key]} {what} line(s)')
        same = [s for s in sorted(loss) if s - 1 in loss and s in gn and s - 1 in gn
                and loss[s] == loss[s - 1] and gn[s] == gn[s - 1]]
        if len(same) >= FROZEN_SCALARS_RUN - 1 and same:
            bad.append(f'loss and grad norm bitwise identical to the previous update at {same[:6]}')
        if bad:                                # (a failed exit names the error line under completes)
            err = e.get('error') if ev.get('exit_code') == 0 else None
            return C.result('no_errors', C.FAIL, '; '.join(bad) + (f'; the log: {err}' if err else ''),
                            samples=(e.get('samples') or [])[:3])
        if at_end:
            return C.result('no_errors', C.PASS, f'{at_end} collective/NCCL error line(s) in a run that logged every '
                            'update and exited 0 (the job\'s own end, not a failure); finite loss and gradient norm',
                            samples=(e.get('samples') or [])[:3])
        return C.result('no_errors', C.PASS, 'no error lines, finite loss and gradient norm')

    def _memory(self, ev):
        peak, cap = _num(ev.get('memory_peak_gib')), _num(self.memory_cap_gib) or _num(ev.get('memory_cap_gib'))
        if peak is None or cap is None:
            return C.result('memory_cap', C.UNCHECKED, 'peak memory or device cap not known')
        ok = peak <= cap
        return C.result('memory_cap', C.PASS if ok else C.FAIL, f'peak {peak:.1f} GiB vs cap {cap:.1f} GiB')

    def _same_workload(self, ev, ref):
        if ref is None:
            return C.result('same_workload', C.PASS, 'reference run')
        checked, bad = [], []
        for key in ('tokens', 'samples', 'seq_len'):
            want = _expected(ref, key, ev)
            if not want:
                continue
            a, b = _steps(ev, key), _steps(ref, key)
            missing = [s for s in want if s not in a]
            diff = [s for s in want if s in a and a[s] != b[s]]
            if missing:
                bad.append(f'{key} not reported at updates {missing[:6]}')
            elif diff:
                bad.append(f'{key} per update {a[diff[0]]} != reference {b[diff[0]]} at updates {diff[:6]}')
            checked.append(f'{key} ({len(want)} updates)')
        sa, sb = ev.get('static_workload') or {}, ref.get('static_workload') or {}
        keys = sorted(set(sa) & set(sb))
        diff = [k for k in keys if sa[k] != sb[k]]
        if diff:
            bad.append('; '.join(f'{k} {sa[k]} != reference {sb[k]}' for k in diff[:4]) + ' (from the configuration)')
        if bad:
            return C.result('same_workload', C.FAIL, '; '.join(bad))
        if checked:
            return C.result('same_workload', C.PASS, 'equal per update: ' + ', '.join(checked)
                            + (f'; {", ".join(keys)} equal (configuration)' if keys else ''))
        if keys:
            return C.result('same_workload', C.PASS, f'{", ".join(keys)} equal (from the configuration; not measured)')
        return C.result('same_workload', C.UNCHECKED, 'no tokens/samples per update in the signals and no workload '
                        'invariant in the manifest (expose them in the log or through the probe)')

    def _same_data(self, ev, ref):
        if ref is None:
            return C.result('same_data', C.PASS, 'reference run')
        a, b = _steps(ev, 'data'), _steps(ref, 'data')
        want = _expected(ref, 'data', ev)
        if not want:
            return C.result('same_data', C.UNCHECKED, 'no per-update data digest in the signals')
        if not a:
            return C.result('same_data', C.FAIL, f'the reference reports a data digest at {len(want)} updates, this run at '
                            'none (the data path or its logging changed)')
        missing = [s for s in want if s not in a]
        diff = [s for s in want if s in a and a[s] != b[s]]
        if missing or diff:
            return C.result('same_data', C.FAIL, f'global batch digest missing at {missing[:6]}' if missing
                            else f'global batch digest differs at updates {diff[:6]}')
        return C.result('same_data', C.PASS, f'same global batches at {len(want)} updates')

    def _optimizer_steps(self, ev, ref):
        if ref is None:
            return C.result('optimizer_steps', C.PASS, 'reference run')
        same = []                                       # both when both are logged: equal counters, another schedule
        for key, eq, what in (('optim_step', lambda x, y: x == y, 'optimizer-step counter'),
                              ('lr', lambda x, y: _rel(float(x), float(y)) <= LR_REL, 'learning rate')):
            want = _expected(ref, key, ev)
            a, b = _steps(ev, key), _steps(ref, key)
            if want and not a and key == 'optim_step':  # the reference counts optimizer steps: so must the candidate
                return C.result('optimizer_steps', C.FAIL, f'optim_step not reported (the reference reports it at {len(want)} updates)')
            if not want or not a:
                continue
            missing = [s for s in want if s not in a]
            diff = [s for s in want if s in a and not eq(a[s], b[s])]
            if missing:
                return C.result('optimizer_steps', C.FAIL, f'{key} not reported at updates {missing[:6]}')
            if diff:
                s = diff[0]
                return C.result('optimizer_steps', C.FAIL, f'{key} {a[s]} != reference {b[s]} at updates {diff[:6]}')
            same.append(f'{what} equal at {len(want)} updates')
        if same and any('counter' in s for s in same):
            return C.result('optimizer_steps', C.PASS, '; '.join(same))
        if same:                                        # the learning rate alone: its schedule advances whether or not
            return C.result('optimizer_steps', C.UNCHECKED, same[0] + ' (a proxy: no optimizer-step counter is exposed, '
                            'so a skipped or merged optimizer step would not show; log one per update, e.g. `optim_step N`, '
                            'or use the probe)')       # the optimizer stepped (a job that skips steps passed, F-SYS-14)
        return C.result('optimizer_steps', C.UNCHECKED, 'no per-update learning rate or optimizer-step counter in the signals')

    def _same_model(self, ev, ref):
        if ref is None:
            return C.result('same_model', C.PASS, 'reference run')
        ma, mb = ev.get('meta') or {}, ref.get('meta') or {}
        dropped = [k for k in ('trainable_params', 'model_digest') if mb.get(k) is not None and ma.get(k) is None]
        if dropped:                                     # the reference reports it: so must the candidate (as same_data)
            return C.result('same_model', C.FAIL, f'{", ".join(dropped)} not reported (the reference reports '
                            f'{"them" if len(dropped) > 1 else "it"}): the model signals or their logging changed')
        keys = [k for k in ('trainable_params', 'model_digest') if ma.get(k) is not None and mb.get(k) is not None]
        if not keys:
            return C.result('same_model', C.UNCHECKED, 'no trainable-parameter count or model digest in the signals')
        diff = [k for k in keys if ma[k] != mb[k]]
        if diff:
            return C.result('same_model', C.FAIL, '; '.join(f'{k} {ma[k]} != reference {mb[k]}' for k in diff))
        return C.result('same_model', C.PASS, f'{", ".join(keys)} equal')

    def _frozen(self, ev):
        fz = ev.get('frozen') or {}
        if not fz.get('checked'):
            return C.result('frozen_files', C.UNCHECKED, 'no frozen files declared')
        bad = sorted(set(fz.get('changed') or []) | set(fz.get('missing') or []))
        if bad:
            return C.result('frozen_files', C.FAIL, f'frozen files changed or missing: {bad[:6]}')
        return C.result('frozen_files', C.PASS, f'{fz.get("n", 0)} frozen files unchanged')

    def _loss(self, ev, ref, paired: bool):
        tol = self.tolerance
        if ref is None:
            return C.result('loss_within_tolerance', C.PASS if _steps(ev, 'loss') else C.UNCHECKED, 'reference run')
        a, b = _steps(ev, 'loss'), _steps(ref, 'loss')
        if not a or not b:
            return C.result('loss_within_tolerance', C.UNCHECKED, 'no loss values in the signals of this run or the reference')
        want = [s for s in _expected(ref, 'loss', ev) if type(b[s]) is not str]
        missing = [s for s in want if s not in a or type(a[s]) is str]
        if missing:
            return C.result('loss_within_tolerance', C.FAIL, f'loss missing or non-finite at updates {missing[:6]}')
        a = {s: float(a[s]) for s in want}
        b, b_all = {s: float(b[s]) for s in want}, {s: float(v) for s, v in b.items() if type(v) is not str}
        common = want
        if not common:
            return C.result('loss_within_tolerance', C.UNCHECKED, 'no comparable loss values')
        widened = ''
        if paired:
            dev = max(_rel(a[s], b[s]) for s in common)
            worst = max(common, key=lambda s: _rel(a[s], b[s]))
            sdev = _signed(a[worst], b[worst])
            how = f'max per-update deviation {dev:.4%} (update {worst}, {a[worst]:.6g} vs {b[worst]:.6g})'
        else:
            lo, hi = ref.get('window') or [common[0], common[-1]]
            win = [s for s in common if lo <= s <= hi] or common
            ma, mb = sum(a[s] for s in win) / len(win), sum(b[s] for s in win) / len(win)
            dev, sdev = _rel(ma, mb), _signed(ma, mb)
            how = f'window-mean deviation {dev:.4%} ({ma:.6g} vs {mb:.6g} over {len(win)} updates; data order not verified)'
            # stopped before the window's end (a probe or short run): the tolerance is for the window mean, and a mean of
            # this few updates carries the loss's per-update noise (D2-R204c: 22-53 % over 2 updates of a diffusion loss
            # for changes that cannot touch the numerics, 42 % for the reference's own repeat)
            done = ev.get('steps_completed')            # the run itself stopped early, not a missing last loss sample
            short = type(done) is int and done < hi
            # a full run on a job whose draws differ between runs compares means of different draws too: the reference's
            # own repeat says so (D2-R204d: the doctor's run of the same configuration 40 % off per update, 23 % on the
            # window mean - rejected at 3 %); a repeat within the tolerance per update (every seeded job) keeps the manifest
            rep = None if short else self.repeat_disagreement(ref, lo, hi)
            vary = rep is not None and rep > self.tolerance
            noise = loss_noise(b_all, lo, hi) if short or vary else None
            if noise and LOSS_NOISE_Z * noise * math.sqrt(2 / len(win)) > tol:
                tol = LOSS_NOISE_Z * noise * math.sqrt(2 / len(win))
                widened = (f' (the manifest {self.tolerance:.2%} is for the window mean; the reference\'s loss varies '
                           f'{noise:.1%} between updates, so a {len(win)}-update mean is held to {LOSS_NOISE_Z:g} x its noise'
                           + (f'; the draws differ between runs: the reference\'s own repeat is {rep:.1%} off per update' if vary
                              else '') + ')')
        ok = dev <= tol
        return C.result('loss_within_tolerance', C.PASS if ok else C.FAIL, f'{how}; tolerance {tol:.2%}{widened}',
                        deviation=dev, signed_deviation=sdev, paired=paired, tolerance=tol)

    def _grad_norm(self, ev, ref, data_status: str):
        if ref is None:
            return C.result('grad_norm_within_tolerance', C.PASS if _steps(ev, 'grad_norm') else C.UNCHECKED, 'reference run')
        a, b = _steps(ev, 'grad_norm'), _steps(ref, 'grad_norm')
        if not a or not b:
            return C.result('grad_norm_within_tolerance', C.UNCHECKED, 'no gradient norm in the signals')
        s = min(b)
        if type(b[s]) is str:
            return C.result('grad_norm_within_tolerance', C.UNCHECKED, f'reference gradient norm at update {s} is not finite')
        if s not in a or type(a[s]) is str:
            return C.result('grad_norm_within_tolerance', C.FAIL, f'gradient norm of the first update ({s}) missing or not finite')
        dev = _rel(float(a[s]), float(b[s]))
        detail = f'update {s}: {a[s]:.6g} vs {b[s]:.6g} ({dev:.4%}); tolerance {self.tolerance:.2%}'
        if data_status == C.UNCHECKED:          # without proof of the same first batch it is a report, not a veto
            return C.result('grad_norm_within_tolerance', C.UNCHECKED, detail + ' (data order not verified: report only)')
        return C.result('grad_norm_within_tolerance', C.PASS if dev <= self.tolerance else C.FAIL, detail,
                        deviation=dev, signed_deviation=_signed(float(a[s]), float(b[s])))

    def _timing(self, ev, ref):
        clock, rep = _pos(ev.get('clock_step_time')), _pos(ev.get('reported_step_time'))
        if clock is None:
            if ref is not None and _pos(ref.get('clock_step_time')) is not None:
                return C.result('timing_consistent', C.FAIL, 'the reference was timed on InfraRSI\'s clock, this run '
                                f'could not be ({ev.get("clock_note") or "no reliable timestamps"}): its speed would rest '
                                'on the code\'s own report')
            return C.result('timing_consistent', C.UNCHECKED, ev.get('clock_note') or 'no reliable step-completion timestamps')
        if rep is None:
            return C.result('timing_consistent', C.PASS, f'measured on InfraRSI\'s clock ({clock:.4g} s/update); the '
                            'codebase reports no step time to cross-check')
        if ref is None or _pos(ref.get('clock_step_time')) is None or _pos(ref.get('reported_step_time')) is None:
            gap = abs(math.log(clock / rep))
            ok = gap <= max(self.timing_tolerance, 0.10)    # absolute agreement is loose: logging/eval steps differ
            if ref is None and not ok:                  # the reference run: the job's own timer covers part of the update
                return C.result('timing_consistent', C.INFO, f'the code\'s step time ({rep:.4g} s) covers {rep / clock:.0%} of '
                                f'the update on InfraRSI\'s clock ({clock:.4g} s): candidates are compared on the clock, '
                                'the reported time is a cross-check (data loading or logging outside the timed region)')
            return C.result('timing_consistent', C.PASS if ok else C.FAIL,
                            f'clock {clock:.4g} s vs reported {rep:.4g} s per update (|log ratio| {gap:.3f})')
        g_clock = math.log(ref['clock_step_time'] / clock)
        g_rep = math.log(ref['reported_step_time'] / rep)
        gap = abs(g_clock - g_rep)
        ok = gap <= self.timing_tolerance
        return C.result('timing_consistent', C.PASS if ok else C.FAIL,
                        f'gain on the clock {g_clock:+.3f} vs gain in the reported step time {g_rep:+.3f} (log; |diff| {gap:.3f}, '
                        f'limit {self.timing_tolerance})')

    def _precision(self, ev, ref):
        pa = (ev.get('meta') or {}).get('precision')
        pb = ((ref or {}).get('meta') or {}).get('precision') if ref is not None else pa
        if not isinstance(pa, dict):
            return C.result('precision', C.INFO, 'not reported')
        diff = {k: [pb.get(k) if isinstance(pb, dict) else None, v] for k, v in pa.items()
                if not isinstance(pb, dict) or pb.get(k) != v}
        return C.result('precision', C.INFO, 'unchanged' if not diff else f'changed vs reference: {diff}')

    # ------------------------------------------------------------ the judge
    def __call__(self, receipt, parent=None, reference=None) -> Verdict:
        if not isinstance(receipt, dict):
            return Verdict(valid=False, reasons=['receipt is not an object'], source='verifier')
        if self.require_signature and not signing.verify(receipt):
            return Verdict(valid=False, reasons=['receipt signature missing or invalid'], source='verifier')
        if receipt.get('schema') != SCHEMA:
            return Verdict(valid=False, reasons=[f'schema {receipt.get("schema")!r} != {SCHEMA}'], source='verifier')
        ev = evidence(receipt)
        is_ref = reference is None or (isinstance(reference, dict) and reference.get('attempt_id') == receipt.get('attempt_id'))
        ref = None if is_ref else evidence(reference)
        results = self.screen_checks(ev, ref)
        block = C.blocking(results, self.required)
        t = _pos(ev.get('step_time'))
        if t is None and not block:
            block = [C.result('completes', C.FAIL, 'no step time could be measured')]
            results.append(block[0])
        valid = not block
        pt = _pos(evidence(parent).get('step_time')) if isinstance(parent, dict) else None
        reward = math.log(pt / t) if valid and pt and t else None
        return Verdict(valid=valid, reasons=[f'{r["id"]}: {r["detail"]}' for r in block], step_time=t,
                       peak_reserved_gib=_num(ev.get('memory_peak_gib')), reward=reward,
                       numerics_class='reference' if is_ref else None, source='verifier', checks=results,
                       feedback=C.feedback(results, self.required), coverage=C.coverage(results))

    # ------------------------------------------------------------ promotion items
    def promotion(self, candidate: dict, reference: dict, *, rescore: tuple | None = None, resume: dict | None = None,
                  save_at: int | None = None, tests: list | None = None, evals: tuple | None = None,
                  margin: float | None = None) -> list[dict]:
        """rescore: (reference, candidate) receipts of the fresh back-to-back pair, judged against `margin` (2 sigma_pair
        of the measured noise; without it the reference's per-update RSD rule); resume: the receipt of the run resumed
        from the checkpoint saved after update `save_at` (its updates must continue the candidate's uninterrupted run);
        tests: [{'cmd', 'exit_code', 'tail'}]; evals: (reference, candidate) results of the eval command on the two
        checkpoints saved after the same update, {'exit_code', 'value', 'cmd', 'tail'}."""
        out = []
        if rescore is None:
            out.append(C.result('speed_reproduces', C.UNCHECKED, 'no locked re-run'))
        else:
            r_ref, r_cand = rescore
            vr, vc = self(r_ref, None, None), self(r_cand, None, r_ref)
            if not (vr.valid and vc.valid):
                bad = vr.reasons if not vr.valid else vc.reasons
                out.append(C.result('speed_reproduces', C.FAIL, f're-run not valid: {"; ".join(bad)[:300]}'))
            else:
                m = margin if margin is not None else margin_of(evidence(r_ref), self.margin_floor)
                gain = math.log(vr.step_time / vc.step_time)
                ok = gain > math.log1p(m)
                out.append(C.result('speed_reproduces', C.PASS if ok else C.FAIL,
                                    f're-run speedup x{math.exp(gain):.3f} vs required > x{1 + m:.3f} (noise margin)',
                                    speedup=math.exp(gain), margin=m))
        if resume is None:
            out.append(C.result('resume_continues', C.UNCHECKED, 'no checkpoint commands configured'))
        else:
            ra, rc = evidence(resume), evidence(candidate)
            a, b = _steps(ra, 'loss'), _steps(rc, 'loss')
            first = min(a) if a else None
            k = int(save_at) if save_at is not None else (first - 1 if first else 0)
            want = sorted(s for s in b if s > k and type(b[s]) is not str)    # the uninterrupted run after the checkpoint
            missing = [s for s in want if s not in a or type(a[s]) is str]
            common = [s for s in want if s not in missing]
            if first is not None and first <= k:
                out.append(C.result('resume_continues', C.FAIL, f'the resumed run started again at update {first}, not after '
                                    f'the checkpoint at update {k} (the checkpoint was not loaded)'))
            elif not (ra.get('exit_code') == 0 and want) or missing:
                out.append(C.result('resume_continues', C.FAIL, f'resumed run exit {ra.get("exit_code")}; loss at '
                                    f'{len(common)} of the {len(want)} updates after update {k}'
                                    + (f' (missing {missing[:6]})' if missing else '')))
            else:
                dev = max(_rel(float(a[s]), float(b[s])) for s in common)
                out.append(C.result('resume_continues', C.PASS if dev <= self.tolerance else C.FAIL,
                                    f'resumed loss vs uninterrupted: max deviation {dev:.4%} over updates {common[0]}..{common[-1]}'))
        if evals is None:
            out.append(C.result('eval_matches', C.UNCHECKED, 'no eval command (or no checkpoint commands) configured'))
        else:
            er, ec = evals
            bad = [n for n, e in (('reference', er), ('candidate', ec)) if e.get('exit_code') != 0 or _num(e.get('value')) is None]
            if bad:
                out.append(C.result('eval_matches', C.FAIL, f'eval command failed or printed no eval loss for {bad}'))
            else:
                dev = _rel(float(ec['value']), float(er['value']))
                out.append(C.result('eval_matches', C.PASS if dev <= self.tolerance else C.FAIL,
                                    f'eval loss {ec["value"]:.6g} vs reference {er["value"]:.6g} ({dev:.4%}); tolerance '
                                    f'{self.tolerance:.2%}', deviation=dev))
        if not tests:
            out.append(C.result('user_tests', C.UNCHECKED, 'no test commands in the manifest'))
        else:
            bad = [t for t in tests if t.get('exit_code') != 0]
            out.append(C.result('user_tests', C.FAIL if bad else C.PASS,
                                f'{len(tests) - len(bad)}/{len(tests)} test commands passed'
                                + (f'; failed: {bad[0].get("cmd")} -> {str(bad[0].get("tail"))[-300:]}' if bad else '')))
        return [C.result(r['id'], C.OFF, 'disabled in the manifest') if r['id'] in self.disabled else r for r in out]
