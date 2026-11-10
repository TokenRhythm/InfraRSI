"""InfraRSI's one-off phase profile of a training job (textbook design P2, generic phases; FABLE_DESIGN_CHECK 5.2).

Loaded by every Python process that has this directory first on PYTHONPATH (the sitecustomize it shadows is imported
first, as Python would have); active only when INFRARSI_PROFILE names an output directory, and only in the training
process of rank 0. A global torch.optim step post-hook (DeepSpeed: the engine's step at an accumulation boundary)
finds the end of an update - with several optimizers, the step of the one that comes last in an update - and marks it
on the GPU timeline (`infrarsi::update_end`, one tiny kernel). After INFRARSI_PROFILE_WARMUP updates it records the
next INFRARSI_PROFILE_UPDATES + 1 with torch.profiler, then writes `phases.json`: per update, the GPU timeline split
into mutually exclusive parts - optimizer (kernels inside an optimizer step), compute (the other non-communication
kernels), comm (communication kernels while no other kernel runs: the exposed part) and idle (no kernel: host-side
work, input, launch gaps). It stands aside inside a CUDA graph capture and in compiled code. Any failure ends in
{"error": ...} or in nothing; the job itself never fails because of this file.

The run's peak memory (D3, every run of every strategy; observation only): with INFRARSI_MEMPEAK naming a directory,
every process of the job - every rank on every node - reads the peak reserved bytes of every device from its caching
allocator's statistics as the C binding returns them (reserved_bytes.all.peak: no flattening, no device
synchronisation, no CUDA call; a device it holds nothing on reads 0) and never imports torch itself (PeakWatch). A full
run (the reference, the screens, the promotion's re-runs) does no periodic work during the job: one read of every
device at interpreter exit - the counter is a high-water mark, so that is the whole run's peak. A full run stopped by a
signal (an early abort, a wall cap) leaves no value: its peak is unknown (the runner logs it so); a job that resets the
peak statistics itself leaves the peak since its last reset. A probe or short run (the runner sets
INFRARSI_MEMPEAK_FAST: it is stopped by a signal after a few updates, so its exit may never come) starts a daemon
thread that reads every device every PEAK_FAST seconds once the process has initialised CUDA, plus the read at exit. The
largest value goes to <dir>/<host>-<pid>.txt (`mempeak <GiB>`) when it grew; the runner takes the maximum over every file
of every host after the run. (The first version read every device's flattened statistics every 2 s in every process:
+1.7 % per update and twice the RSD on 16 GPUs, the stalls of 16 ranks adding up through the collectives; the second, a
sampler backing off to every 30 s, still left the full runs' in-run RSD about 1 point higher - measured 2026-10-31.)

The modules the job imports (D7, the reference run only): with INFRARSI_MODULES naming a directory, rank 0 (and its
launcher) writes {module: source file} of what it imported in <dir>/<pid>.json once, at interpreter exit; the
library-source excerpts (search/library.py) read only files listed there.
"""
import json
import os

_DIR = os.environ.get('INFRARSI_PROFILE')
_ENV_RANK = os.environ.get('RANK', os.environ.get('LOCAL_RANK'))
MARK = 'infrarsi::update_end'
COMM = ('nccl', 'allreduce', 'all_reduce', 'allgather', 'all_gather', 'reducescatter', 'reduce_scatter', 'alltoall',
        'sendrecv', 'broadcast')


def _union(iv):
    """Merged [start, end) intervals."""
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _length(iv):
    return sum(b - a for a, b in iv)


def _clip(iv, lo, hi):
    return [(max(a, lo), min(b, hi)) for a, b in iv if b > lo and a < hi]


def _minus(iv, cut):
    """The parts of the merged intervals `iv` not covered by the merged intervals `cut`."""
    out, j = [], 0
    for a, b in iv:
        cur = a
        while j < len(cut) and cut[j][1] <= cur:
            j += 1
        k = j
        while k < len(cut) and cut[k][0] < b:
            if cut[k][0] > cur:
                out.append((cur, cut[k][0]))
            cur = max(cur, cut[k][1])
            k += 1
        if cur < b:
            out.append((cur, b))
    return out


def split(events, updates: int) -> dict:
    """phases.json from a chrome trace's events: the last `updates` complete updates between consecutive update ends
    (the hook's `infrarsi::update_end` marks; a trace without them: the ends of the optimizer's GPU ranges, the one
    that steps last), each split into optimizer / compute / comm (exposed) / idle ms."""
    kernels = [(e['ts'], e['ts'] + e['dur'], str(e.get('name', '')).lower()) for e in events
               if e.get('ph') == 'X' and e.get('cat') == 'kernel' and e.get('dur') is not None]
    ann = sorted((e['ts'], e['ts'] + e['dur'], str(e.get('name', ''))) for e in events if e.get('ph') == 'X'
                 and e.get('cat') == 'gpu_user_annotation' and e.get('dur') is not None)
    steps = [x for x in ann if x[2].startswith('Optimizer.step')]
    marks = [b for _, b, n in ann if n == MARK]
    ends = (marks or [b for _, b, n in steps if n == steps[-1][2]] if steps or marks else [])[-(updates + 1):]
    if not kernels or len(ends) < 2:
        return {'error': f'{len(kernels)} kernels, {len(ends)} update ends on the GPU timeline: no update boundaries'}
    opt = _union([(a, b) for a, b, _ in steps])
    rows = []
    for lo, hi in zip(ends, ends[1:]):
        ks = [(max(a, lo), min(b, hi), n) for a, b, n in kernels if b > lo and a < hi]
        comm = _union([(a, b) for a, b, n in ks if any(w in n for w in COMM)])
        other = _union([(a, b) for a, b, n in ks if not any(w in n for w in COMM)])
        o = _union(_clip(opt, lo, hi))
        optimizer = _minus(other, _minus(other, o))          # non-comm kernel time inside the optimizer's ranges
        compute = _minus(other, o)
        exposed = _minus(comm, other)
        busy = _union(comm + other)
        rows.append({'optimizer': _length(optimizer) / 1e3, 'compute': _length(compute) / 1e3,
                     'comm': _length(exposed) / 1e3, 'idle': (hi - lo - _length(busy)) / 1e3, 'update': (hi - lo) / 1e3})
    keys = ('compute', 'comm', 'optimizer', 'idle', 'update')
    return {'unit': 'ms', 'updates': len(rows), 'per_update': rows,
            'mean_ms': {k: round(sum(r[k] for r in rows) / len(rows), 3) for k in keys}}


PEAK_FAST = 2.0                                         # a probe's or short run's sampler: seconds between two reads
PEAK_RETRY = 30.0                                       # ... after a wake-up that failed


def _modules_now() -> dict:
    """D7: {module name: its source file} of what this process imported (files only)."""
    import sys
    out = {}
    for name, m in list(sys.modules.items()):
        f = getattr(m, '__file__', None)
        if isinstance(f, str) and f.endswith('.py'):
            out[name] = f
    return out


def _reserved(t, d: int) -> tuple | None:
    """(current, peak) reserved bytes of the caching allocator on device d, from the nested statistics the C binding
    returns (no flattening: torch.cuda.max_memory_reserved flattens every statistic into a new dict on every call); None
    when they cannot be read. Host-side counters: no device synchronisation, no CUDA call."""
    try:
        f = getattr(getattr(t, '_C', None), '_cuda_memoryStats', None)
        stats = f(d) if f is not None else t.cuda.memory_stats_as_nested_dict(d)
        r = stats['reserved_bytes']['all']
        return int(r['current']), int(r['peak'])
    except Exception:                                   # noqa: BLE001 - a diagnostic never breaks the job
        return None


class PeakWatch:
    """D3 / D7 of one process (the module docstring): `final()` at interpreter exit (every process); `step()`, one
    wake-up of a probe's or short run's sampler thread, returns the seconds to the next one. `torch_module` is for the
    tests."""

    def __init__(self, path: str, modules: str | None = None, *, fast: bool = False, torch_module=None):
        self.path, self.modules = path, modules
        self.fast = bool(fast)                          # a probe or short run (INFRARSI_MEMPEAK_FAST): the sampler
        self.torch_module = torch_module
        self.best = 0                                   # bytes
        self.reads = 0                                  # (tests: reads made, module lists written)
        self.module_dumps = 0
        self._exit_registered = False                   # (the sampler's own exit hook, after torch's)
        self._final_done = False

    def _torch(self):
        import sys
        return self.torch_module if self.torch_module is not None else sys.modules.get('torch')

    def _write_peak(self, peak: int) -> None:
        try:
            with open(self.path + '.tmp', 'w', encoding='utf-8') as f:
                f.write('mempeak %.4f\n' % (peak / 2 ** 30))
            os.replace(self.path + '.tmp', self.path)
        except OSError:
            pass

    def _dump_modules(self) -> None:
        try:
            mods = _modules_now()
            with open(self.modules + '.tmp', 'w', encoding='utf-8') as f:
                json.dump(mods, f)
            os.replace(self.modules + '.tmp', self.modules)
            self.module_dumps += 1
        except (OSError, ValueError, RuntimeError):     # (the dict changed while read: the next time)
            pass

    def read(self, t) -> bool:
        """One read of every device's peak reserved bytes (this process's allocator: 0 where it holds nothing); the file
        rewritten when the peak grew. True when it grew."""
        peaks = [r[1] for r in (_reserved(t, d) for d in range(int(t.cuda.device_count()))) if r]
        self.reads += 1
        peak = max(peaks, default=0)
        if peak <= self.best:
            return False
        self.best = peak
        self._write_peak(peak)
        return True

    def step(self) -> float:
        """One wake-up of the sampler (a probe or short run only): a read once the process has initialised CUDA."""
        t = self._torch()
        try:
            if t is None or not t.cuda.is_initialized():
                return PEAK_FAST                        # (a dictionary look-up and one flag until CUDA is up)
        except Exception:                               # noqa: BLE001
            return PEAK_RETRY
        if not self._exit_registered:                   # registered after torch's own exit hooks: it runs before them
            import atexit
            atexit.register(self.final)
            self._exit_registered = True
        self.read(t)
        return PEAK_FAST

    def final(self) -> None:
        """At interpreter exit, once: one read of every device (a full run's only read - the whole run's high-water
        mark), the module list."""
        if self._final_done:
            return
        self._final_done = True
        if self.path:
            t = self._torch()
            try:
                if t is not None and t.cuda.is_initialized():
                    self.read(t)
            except Exception:                           # noqa: BLE001
                pass
        if self.modules:
            self._dump_modules()


def _watch(w: PeakWatch) -> None:
    """A probe's or short run's sampler thread."""
    import time
    delay = PEAK_FAST
    while True:
        time.sleep(max(0.05, delay))
        try:
            delay = w.step()
        except Exception:                               # noqa: BLE001 - a diagnostic never breaks the job
            delay = PEAK_RETRY


def _write(obj) -> None:
    """phases.json, best effort (a full disk must not reach the job)."""
    try:
        os.makedirs(_DIR, exist_ok=True)
        with open(os.path.join(_DIR, 'phases.json'), 'w', encoding='utf-8') as f:
            json.dump(obj, f)
    except OSError:
        pass


def _chain() -> None:
    """Import the sitecustomize this one shadows the way Python would have (its own module, in sys.modules)."""
    import importlib
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    me = sys.modules.pop('sitecustomize', None)
    ours = [(i, p) for i, p in enumerate(sys.path) if os.path.abspath(p or '.') == here]
    try:
        sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != here]
        try:
            importlib.import_module('sitecustomize')
        except ImportError as exc:
            if exc.name != 'sitecustomize':
                raise
        except Exception as exc:                            # noqa: BLE001 - as Python's site reports a failing one
            print(f'Error in sitecustomize; set PYTHONVERBOSE for traceback:\n{type(exc).__name__}: {exc}',
                  file=sys.stderr)
    finally:
        for i, p in ours:                               # back where it was; the imported one's own changes stay
            sys.path.insert(min(i, len(sys.path)), p)
        if 'sitecustomize' not in sys.modules and me is not None:
            sys.modules['sitecustomize'] = me


_LIVE = __name__ == 'sitecustomize'                     # imported as a module (tests): definitions only
if _LIVE:
    _chain()

_PEAK_DIR = os.environ.get('INFRARSI_MEMPEAK')
_MOD_DIR = os.environ.get('INFRARSI_MODULES') if _ENV_RANK in (None, '0') else None   # D7: rank 0 (and its launcher)
if _LIVE and (_PEAK_DIR or _MOD_DIR):                   # D3: every process of the job (any rank, any node)
    try:
        import atexit
        import socket
        for d in (_PEAK_DIR, _MOD_DIR):
            if d:
                os.makedirs(d, exist_ok=True)
        _w = PeakWatch(os.path.join(_PEAK_DIR, f'{socket.gethostname()}-{os.getpid()}.txt') if _PEAK_DIR else '',
                       os.path.join(_MOD_DIR, f'{os.getpid()}.json') if _MOD_DIR else None,
                       fast=bool(_PEAK_DIR) and os.environ.get('INFRARSI_MEMPEAK_FAST') == '1')
        atexit.register(_w.final)                       # every process: the peak and the module list at its exit
        if _w.fast:                                     # a probe or short run only: the sampler (a full run starts no
            import threading                            # thread and does nothing during the job)
            threading.Thread(target=_watch, args=(_w,), daemon=True, name='infrarsi-mempeak').start()
    except Exception:                                   # noqa: BLE001 - a diagnostic never breaks the job
        pass

if _LIVE and _DIR and _ENV_RANK in (None, '0'):
    try:
        import atexit
        import time

        import torch
        from torch.optim.optimizer import register_optimizer_step_post_hook

        _nocompile = getattr(getattr(torch, 'compiler', None), 'disable', None) or (lambda f: f)
        _warm = int(os.environ.get('INFRARSI_PROFILE_WARMUP', '5'))
        _n_prof = int(os.environ.get('INFRARSI_PROFILE_UPDATES', '3'))
        _st = {'n': 0, 'prof': None, 'done': False, 'in_ds': False, 'seen': [], 'last': None, 'cur': [], 'tick': None}

        def _stop(error=None):
            """End the profile: stop the profiler if it runs (outside a capture only), and record why when it failed."""
            p, _st['prof'], _st['done'] = _st['prof'], None, True
            if p is not None:
                try:
                    p.__exit__(None, None, None)
                except Exception:                           # noqa: BLE001
                    pass
            if error:
                _write({'error': error[:500]})

        def _stop_late():
            """A profiler left running by a capture: stopped once outside it (or at exit)."""
            if _st['prof'] is not None and not torch.cuda.is_current_stream_capturing():
                _stop()

        atexit.register(_stop_late)

        @_nocompile
        def _update():
            """One finished training update."""
            if _st['done']:
                _stop_late()
                return
            if torch.cuda.is_current_stream_capturing():   # a CUDA graph capture: no sync, no profiler call now;
                _st['done'] = True                         # a running profiler stops at the next call outside one
                _write({'error': 'the update is captured in a CUDA graph: no phases'})
                return
            if torch.distributed.is_available() and torch.distributed.is_initialized() \
                    and torch.distributed.get_rank() != 0:
                _st['done'] = True                         # not rank 0 (e.g. mp.spawn without RANK in the env)
                return
            _st['n'] += 1
            try:
                if _st['prof'] is not None:                 # mark the update's end on the GPU timeline
                    with torch.profiler.record_function(MARK):
                        _st['tick'].zero_()
                if _st['n'] == _warm:
                    _st['tick'] = torch.zeros(1, device='cuda')
                    p = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA])
                    p.__enter__()
                    _st['prof'] = p
                    with torch.profiler.record_function(MARK):
                        _st['tick'].zero_()
                elif _st['prof'] is not None and _st['n'] >= _warm + _n_prof + 1:
                    torch.cuda.synchronize()
                    p = _st['prof']
                    _stop()
                    path = os.path.join(_DIR, f'trace_{os.getpid()}.json')
                    os.makedirs(_DIR, exist_ok=True)
                    p.export_chrome_trace(path)
                    with open(path, encoding='utf-8') as f:
                        events = json.load(f).get('traceEvents') or []
                    os.remove(path)
                    out = split(events, _n_prof)
                    out.update(warmup=_warm, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
                    _write(out)
            except Exception as exc:                        # noqa: BLE001 - a diagnostic never breaks the job
                _stop(f'{type(exc).__name__}: {exc}')

        @_nocompile
        def _post_step(opt, args, kwargs):
            """Several optimizers: the update ends with the step of the one that comes last in the first round of
            steps (the last new one before any repeats); DeepSpeed's inner steps are counted per engine step."""
            if _st['in_ds'] or _st['done']:
                if _st['done']:
                    _stop_late()
                return
            if _st['last'] is None:
                if id(opt) not in _st['seen']:
                    _st['seen'].append(id(opt))
                    return
                _st['last'] = _st['seen'][-1]               # a repeat: the round is complete
            _st['cur'].append(id(opt))
            if id(opt) == _st['last']:
                if _st['cur'] != _st['seen']:               # steps that do not repeat the round: no update boundary
                    _st['done'] = True
                    _write({'error': 'the optimizers do not step in one repeating order per update: no phases'})
                    _stop_late()                            # stops a running profiler only outside a capture
                    return
                _st['cur'] = []
                _update()

        register_optimizer_step_post_hook(_post_step)
        try:
            from deepspeed.runtime.engine import DeepSpeedEngine
        except Exception:                                   # noqa: BLE001 - no DeepSpeed in this job
            DeepSpeedEngine = None
        if DeepSpeedEngine is not None:
            _ds_step = DeepSpeedEngine.step

            def _engine_step(self, *a, **k):
                boundary = self.is_gradient_accumulation_boundary()
                _st['in_ds'] = True
                try:
                    out = _ds_step(self, *a, **k)
                finally:
                    _st['in_ds'] = False
                if boundary:
                    _update()
                return out

            DeepSpeedEngine.step = _engine_step
    except Exception as exc:                                # noqa: BLE001 - e.g. no torch: nothing to profile
        _write({'error': f'hook: {type(exc).__name__}: {exc}'[:500]})
