"""Search-state checkpoints (not the training job's checkpoints): what an object says about its state, and the file that
keeps it. `infrarsi run --resume` (project/resume.py) continues a run from the last one.

A class whose instances belong to a search's state - the session and its ledger, the engine, the strategies, the
proposer, the budget driver - is `Stateful`: `state_dict()` is every attribute of the instance except those the class
lists in `CHECKPOINT_LIVE`, `load_state_dict()` puts them back. A name in CHECKPOINT_LIVE is deliberately not saved:

    handles       the runner, the hooks, the judge, the task, the clock, the LLM backend, the loggers, the progress line,
                  the engine a strategy was bound to - the resuming run builds its own and the objects hold them again
    configuration what the resuming invocation supplies (the budget, the output directory, the steps of a screen, the
                  caps a manifest sets), so that a budget changed on resume is the new one

Everything else is saved - an attribute added to a class later is saved without a line anywhere - and a value that cannot
be pickled (a lambda, a lock, a file) stops the save with the attribute named (`unsavable`): it is either made a plain
value, or listed here as live. An object that has live parts and state of its own (the budget driver and its call
counters; the ledger and its clock) names the parts in `checkpoint_parts()`: their state goes along and is loaded in
place.

The serialisation is pickle with persistent ids: the live objects of the saving run (the engine, the session, the strategy,
the runner ...) stay out of the pickle wherever they are referenced (a bound method of the strategy in a proposal's gate, the
engine in a ladder's memo) and are replaced by the resuming run's own on load. Files, in <run>/search/state/:

    checkpoint.pkl        the state (written to a temporary file, flushed, renamed over the last one)
    checkpoint.prev.pkl   the one before it
    checkpoint.json       what a person wants to know: schema, time, screens done, budget used, incumbent, package hash
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import os
import pickle
import random
import sys
from pathlib import Path

SCHEMA = 'infrarsi-search-state-v1'
STATE_DIR = 'state'
CHECKPOINT, PREVIOUS, SUMMARY = 'checkpoint.pkl', 'checkpoint.prev.pkl', 'checkpoint.json'
PARTS = '__parts__'
LIVE_TAG = 'infrarsi-live'


class CheckpointError(RuntimeError):
    """A checkpoint that cannot be written, read or used."""


def live_names(cls) -> frozenset:
    """The attribute names a class (with its bases) declares live in CHECKPOINT_LIVE."""
    out: set = set()
    for k in cls.__mro__:
        out |= set(k.__dict__.get('CHECKPOINT_LIVE', ()))
    return frozenset(out)


class Stateful:
    """The protocol (module docstring). A subclass overrides `checkpoint_parts()` to return `{name: stateful object}` for
    the parts of it that are bound to live handles but have state of their own."""
    CHECKPOINT_LIVE: tuple = ()

    def checkpoint_parts(self) -> dict:
        return {}

    def state_dict(self) -> dict:
        live = live_names(type(self))
        out = {k: v for k, v in vars(self).items() if k not in live}
        parts = {name: part.state_dict() for name, part in self.checkpoint_parts().items() if part is not None}
        if parts:
            out[PARTS] = parts
        return out

    def load_state_dict(self, state: dict) -> None:
        """Puts the saved attributes back (a name the saved state lacks keeps the constructor's value; one the class
        lists live is not touched); the parts are loaded in place."""
        state = dict(state)
        saved = state.pop(PARTS, None) or {}
        now = self.checkpoint_parts()
        for name, part_state in saved.items():
            if now.get(name) is not None:
                now[name].load_state_dict(part_state)
        live = live_names(type(self))
        vars(self).update({k: v for k, v in state.items() if k not in live})


# ---------------------------------------------------------------- the pickle
class _Pickler(pickle.Pickler):
    def __init__(self, file, live_ids: dict, protocol=None):
        super().__init__(file, protocol)
        self._live_ids = live_ids

    def persistent_id(self, obj):
        name = self._live_ids.get(id(obj))
        return (LIVE_TAG, name) if name is not None else None


class _Unpickler(pickle.Unpickler):
    def __init__(self, file, live: dict):
        super().__init__(file)
        self._live = live

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and len(pid) == 2 and pid[0] == LIVE_TAG):
            raise pickle.UnpicklingError(f'unknown persistent id {pid!r}')
        if pid[1] not in self._live:
            raise CheckpointError(f'the checkpoint refers to the live object {pid[1]!r}, which this run does not provide')
        return self._live[pid[1]]


def dumps(obj, live: dict | None = None) -> bytes:
    """`obj` pickled; the objects of `live` ({name: object}) are written as references to their names."""
    buf = io.BytesIO()
    _Pickler(buf, {id(o): n for n, o in (live or {}).items() if o is not None}, pickle.HIGHEST_PROTOCOL).dump(obj)
    return buf.getvalue()


def loads(data: bytes, live: dict | None = None):
    return _Unpickler(io.BytesIO(data), live or {}).load()


def unsavable(state: dict, live: dict | None = None, prefix: str = '') -> list[str]:
    """The keys of `state` (one level down, and the keys of a dict value) whose values do not pickle, with the error - the
    answer to 'which attribute did that': `prefix.key: TypeError: ...`."""
    ids = {id(o): n for n, o in (live or {}).items() if o is not None}
    bad = []
    for k, v in state.items():
        buf = io.BytesIO()
        try:
            _Pickler(buf, ids, pickle.HIGHEST_PROTOCOL).dump(v)
        except Exception as exc:                          # noqa: BLE001 - the diagnosis names whatever it was
            deeper = unsavable(v, live, f'{prefix}{k}.') if isinstance(v, dict) and len(v) > 1 else []
            bad += deeper or [f'{prefix}{k}: {type(exc).__name__}: {str(exc)[:160]}']
    return bad


# ---------------------------------------------------------------- randomness
def rng_state() -> dict:
    """The global random generators' state (python's `random`, numpy's legacy generator when numpy is loaded); a strategy's
    own generators are attributes (`random.Random` pickles with its state)."""
    out = {'python': random.getstate()}
    np = sys.modules.get('numpy')
    if np is not None:
        try:
            out['numpy'] = np.random.get_state()
        except Exception:                                 # noqa: BLE001 - numpy without a legacy generator: nothing to save
            pass
    return out


def set_rng_state(state: dict | None) -> None:
    state = state or {}
    if state.get('python') is not None:
        random.setstate(state['python'])
    np = sys.modules.get('numpy')
    if np is not None and state.get('numpy') is not None:
        np.random.set_state(state['numpy'])


# ---------------------------------------------------------------- the code version
_HASH: dict = {}


def package_hash(root: str | Path | None = None) -> str:
    """sha256 over the files of the infrarsi package (relative path and bytes, `__pycache__` left out): the code version a
    checkpoint was made by."""
    base = Path(root) if root else Path(__file__).resolve().parents[1]
    key = str(base)
    if key not in _HASH:
        h = hashlib.sha256()
        for p in sorted(base.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
                h.update(p.relative_to(base).as_posix().encode('utf-8') + b'\0')
                h.update(hashlib.sha256(p.read_bytes()).digest())
        _HASH[key] = h.hexdigest()
    return _HASH[key]


# ---------------------------------------------------------------- the files
def utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def write(directory: Path, header: dict, body: bytes, flow: dict | None = None) -> Path:
    """checkpoint.pkl atomically: a temporary file, flushed to disk, renamed over the last checkpoint (kept as
    checkpoint.prev.pkl); then checkpoint.json (the header) the same way. The file is a pickle of
    {'schema', 'header', 'flow', 'body'}: the header and the flow's own state are plain data (what a person and the
    resume's checks read without any live object), the body is the search's state pickled with the live objects as
    references (dumps), read with the live objects of the resuming run."""
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / (CHECKPOINT + '.tmp')
    with open(tmp, 'wb') as f:
        f.write(pickle.dumps({'schema': SCHEMA, 'header': header, 'flow': flow, 'body': body}, pickle.HIGHEST_PROTOCOL))
        f.flush()
        os.fsync(f.fileno())
    cur = directory / CHECKPOINT
    if cur.exists():
        os.replace(cur, directory / PREVIOUS)
    os.replace(tmp, cur)
    tmp_json = directory / (SUMMARY + '.tmp')
    tmp_json.write_text(json.dumps({'schema': SCHEMA, **header}, indent=1, default=str), encoding='utf-8')
    os.replace(tmp_json, directory / SUMMARY)
    return cur


def read(directory: Path) -> tuple[dict, Path]:
    """({'schema', 'header', 'flow', 'body'}, the file it came from): checkpoint.pkl, else - missing or unreadable -
    checkpoint.prev.pkl, else a complete temporary file left by a save the run did not finish."""
    errors = []
    for name in (CHECKPOINT, PREVIOUS, CHECKPOINT + '.tmp'):
        p = directory / name
        if not p.is_file():
            continue
        try:
            data = pickle.loads(p.read_bytes())
            if isinstance(data, dict) and data.get('schema') == SCHEMA and isinstance(data.get('body'), bytes):
                return data, p
            errors.append(f'{name}: not a {SCHEMA} checkpoint')
        except Exception as exc:                          # noqa: BLE001 - try the previous one
            errors.append(f'{name}: {type(exc).__name__}: {str(exc)[:120]}')
    raise CheckpointError(f'no readable checkpoint in {directory}' + (': ' + '; '.join(errors) if errors else ''))


def state_of_file(data: dict, live: dict | None = None):
    """The state in a checkpoint read by `read`, its live references resolved to the objects of `live`."""
    return loads(data['body'], live)
