"""Rank-0 heartbeat (INTERFACES section 3.4): every update, and every 60 s while loading or compiling."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from infrarsi.contracts import SCHEMA_HEARTBEAT, utc_now

PHASES = ('load', 'build', 'train', 'save', 'resume', 'export', 'compare', 'done', 'failed')


class Heartbeat:
    def __init__(self, out_dir: str | Path, enabled: bool, interval: float = 60.0, name: str = 'heartbeat.json'):
        self.path = Path(out_dir) / name
        self.enabled = enabled
        self.interval = interval
        self.phase, self.step = 'load', None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def beat(self, phase: str | None = None, step: int | None = None, **extra) -> None:
        if phase is not None:
            if phase not in PHASES:
                raise ValueError(f'heartbeat phase {phase!r}')
            self.phase = phase
        if step is not None:
            self.step = step
        if not self.enabled:
            return
        row = dict(schema=SCHEMA_HEARTBEAT, phase=self.phase, step=self.step, utc=utc_now(), pid=os.getpid(), **extra)
        with self._lock:
            tmp = self.path.with_name(self.path.name + f'.{os.getpid()}.tmp')
            tmp.write_text(json.dumps(row) + '\n', encoding='utf-8')
            os.replace(tmp, self.path)

    def start(self) -> 'Heartbeat':
        """Background ticker for long phases (load, build, first-step compile)."""
        if self.enabled and self._thread is None:
            self._thread = threading.Thread(target=self._run, name='infrarsi-heartbeat', daemon=True)
            self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.beat()
            except OSError:
                pass

    def close(self, phase: str | None = None) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if phase:
            self.beat(phase)
