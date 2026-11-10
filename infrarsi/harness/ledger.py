"""Trusted local launch ledger (Codex R2 finding 2): execution/ledger/launches.jsonl on the driver PC.

run_screen appends {attempt_id, run_id, request_digest, numerics_role, purpose, utc} BEFORE request.json is uploaded;
observe() (driver restart) refuses a remote request.json whose digest differs from the recorded one, or an attempt the
ledger never saw, so restart-time remote state (e.g. a flipped numerics_role) can never become authoritative.
request_digest = contracts.digest(RunRequest.to_dict()) = digest of request.json as written.
Override the path with env INFRARSI_LAUNCH_LEDGER (CPU tests).
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from infrarsi.contracts import EXECUTION_DIR, digest, utc_now

LEDGER_ENV = 'INFRARSI_LAUNCH_LEDGER'
DEFAULT_PATH = EXECUTION_DIR / 'ledger' / 'launches.jsonl'
_tlock = threading.Lock()


class LedgerError(RuntimeError):
    pass


def ledger_path() -> Path:
    p = os.environ.get(LEDGER_ENV)
    return Path(p) if p else DEFAULT_PATH


def request_digest(request) -> str:
    return digest(request.to_dict() if hasattr(request, 'to_dict') else request)


class _FileLock:
    def __init__(self, path: Path, stale_s: float = 60.0):
        self.path, self.stale_s = path, stale_s

    def __enter__(self):
        end = time.monotonic() + 30
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f'{os.getpid()}'.encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > self.stale_s:
                        self.path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() > end:
                    raise LedgerError(f'launch ledger lock {self.path} held')
                time.sleep(0.05)

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def record_launch(request) -> dict:
    """Append the launch record (fsync'd) before anything is uploaded; returns it."""
    entry = {'schema': 'infrarsi-launch-v1', 'attempt_id': request.attempt_id, 'run_id': request.run_id,
             'request_digest': request_digest(request), 'numerics_role': getattr(request, 'numerics_role', None),
             'purpose': request.purpose, 'utc': utc_now()}
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, sort_keys=True) + '\n'
    with _tlock, _FileLock(path.with_name(path.name + '.lock')):
        if lookup(request.attempt_id, path=path) is not None:
            raise LedgerError(f'attempt {request.attempt_id} is already in the launch ledger')
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
    return entry


def lookup(attempt_id: str, path: Path | None = None) -> dict | None:
    path = path or ledger_path()
    try:
        text = path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return None
    found = None
    for line in text.splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if isinstance(e, dict) and e.get('attempt_id') == attempt_id:
            found = e
    return found


def check_remote_request(attempt_id: str, remote_request: dict) -> dict:
    """The ledger entry of attempt_id if the remote request.json is byte-for-byte (by digest) what was launched."""
    entry = lookup(attempt_id)
    if entry is None:
        raise LedgerError(f'attempt {attempt_id} is not in the launch ledger {ledger_path()}; refusing to trust its '
                          'remote request.json')
    got = digest(remote_request)
    if got != entry.get('request_digest'):
        raise LedgerError(f'remote request.json of {attempt_id} (digest {got[:16]}) differs from the launch ledger '
                          f'({str(entry.get("request_digest"))[:16]}); refusing to observe it')
    return entry
