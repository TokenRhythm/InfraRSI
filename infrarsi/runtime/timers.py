"""Frozen timers.

Whole-update time per rank: ``cuda.synchronize`` + barrier, perf_counter at the aligned start, then
``cuda.synchronize`` + perf_counter at the end, so every byte of GPU work issued by the update is inside.
The fwd_bwd / optim splits use CUDA events on the current stream (no extra host sync inside the update);
data_s is the host time spent fetching and staging micro-batches. inter_step_s is the distance between
consecutive aligned starts and exposes work moved outside the timer.

Tamper resistance: the host clocks and the CUDA synchronize / event entry points are bound when this module
is imported, which the trainer does before any L1 overlay runs, so monkeypatching ``time.perf_counter`` or
``torch.cuda.synchronize`` later cannot scale or truncate a measurement. Each update also records the POSIX
wall clock of the same aligned interval (``utc_start_s``/``utc_end_s``) and a CUDA-event span over the whole
update (``cuda_whole_s``) for the driver's cross-checks.
"""
from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter as _clock, time as _utc

try:            # bound at import; absent on the driver PC, where only PhaseClock is used
    import torch as _torch
    _sync = _torch.cuda.synchronize
    _Event = _torch.cuda.Event
    _record = _torch.cuda.Event.record
    _elapsed_ms = _torch.cuda.Event.elapsed_time
    _reset_peak = _torch.cuda.reset_peak_memory_stats
    _current_device = _torch.cuda.current_device
    _current_stream = _torch.cuda.current_stream
except ImportError:
    _torch = None


class PhaseClock:
    """Accumulated wall seconds per job phase (startup, load, build, compile_first_step, train, save, export)."""

    def __init__(self, t0: float | None = None):
        self.t0 = _clock() if t0 is None else t0
        self.phases: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str):
        start = _clock()
        try:
            yield
        finally:
            self.add(name, _clock() - start)

    def add(self, name: str, seconds: float) -> None:
        self.phases[name] = self.phases.get(name, 0.0) + float(seconds)

    def snapshot(self) -> dict:
        out = dict(self.phases)
        out['total'] = _clock() - self.t0
        return out


class UpdateTimer:
    def __init__(self, barrier, device: int):
        if _torch is None:
            raise RuntimeError('UpdateTimer needs torch')
        self._barrier = barrier
        self._device = int(device)      # every sync / event / peak reset names this rank's GPU explicitly
        self._prev_start = None
        self._events = {}
        self.data_s = 0.0

    def _on_device(self) -> None:
        if _current_device() != self._device:
            raise RuntimeError(f'current CUDA device {_current_device()} is not this rank\'s device {self._device}')

    def begin(self) -> None:
        self._on_device()
        _sync(self._device)
        self._barrier()
        self.t0 = _clock()
        self.utc0 = _utc()
        self.inter_step_s = None if self._prev_start is None else self.t0 - self._prev_start
        self._prev_start = self.t0
        self.data_s = 0.0
        _reset_peak(self._device)
        self._events = {}
        self.mark('start')

    @contextmanager
    def data(self):
        start = _clock()
        try:
            yield
        finally:
            self.data_s += _clock() - start

    def mark(self, name: str) -> None:
        event = _Event(enable_timing=True)
        _record(event, _current_stream(self._device))
        self._events[name] = event

    def _span(self, a: str, b: str) -> float | None:
        if a in self._events and b in self._events:
            return _elapsed_ms(self._events[a], self._events[b]) / 1000.0
        return None

    def end(self) -> dict:
        self.mark('end')
        _sync(self._device)
        t1, u1 = _clock(), _utc()
        self._on_device()
        return {'whole_update_s': t1 - self.t0, 'utc_start_s': self.utc0, 'utc_end_s': u1,
                'cuda_whole_s': self._span('start', 'end'), 'fwd_bwd_s': self._span('start', 'fwd_bwd'),
                'optim_s': self._span('fwd_bwd', 'optim'), 'data_s': self.data_s, 'inter_step_s': self.inter_step_s}
