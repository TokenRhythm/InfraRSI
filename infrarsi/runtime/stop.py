"""STOP handling at update boundaries.

The driver (or Opus, on a monitor request) writes ``<out_dir>/STOP.json``. Rank 0 polls it once per update
and the flag travels in the per-step statistics all-gather, so every rank leaves at the same boundary.
The wall-clock cap of the request is enforced the same way.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import monotonic as _monotonic

STOP_FILE = 'STOP.json'
NONE, STOP_REQUESTED, WALL_CAP = 0, 1, 2
REASONS = {NONE: None, STOP_REQUESTED: 'stopped by STOP.json', WALL_CAP: 'wall_seconds_cap exceeded'}


def read_stop(out_dir: str | Path) -> dict | None:
    """Parsed STOP.json if present ({} when unreadable), else None."""
    path = Path(out_dir) / STOP_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}


class StopMonitor:
    def __init__(self, out_dir: str | Path, wall_seconds_cap: float | None, started: float | None = None):
        self.out_dir = Path(out_dir)
        self.cap = float(wall_seconds_cap) if wall_seconds_cap else None
        self.started = _monotonic() if started is None else started

    def flag(self, poll_file: bool = True) -> int:
        if poll_file and read_stop(self.out_dir) is not None:
            return STOP_REQUESTED
        if self.cap is not None and _monotonic() - self.started > self.cap:
            return WALL_CAP
        return NONE

    @staticmethod
    def reason(flag: int) -> str | None:
        return REASONS.get(int(flag), f'stop flag {flag}')
