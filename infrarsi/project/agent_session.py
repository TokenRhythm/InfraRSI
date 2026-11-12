"""`infrarsi agent ...`: a run driven from a coding-agent session (Claude Code, Codex, ...).

The session's agent adapts the project and talks to the user; the engine does everything a run does (the task contract,
the reference run, the search, the checks, the promotion, the report) as a process of its own, and every model call of
the search becomes a request the session answers (agent/backends/session.py). The commands print one JSON object each.

    infrarsi agent start [TARGET] [--model NAME] [optimize's options]   the engine in the background: `optimize`'s
                                              stages with `--llm session`, without writing the result (apply stays the user's)
    infrarsi agent next [--wait SECONDS]      wait until the run needs the session - a request to answer, the end, a
                                              failure - or SECONDS passed
    infrarsi agent answer ID [--file F] [--tokens N]   hand in the answer to request ID (default: its answer.json)
    infrarsi agent status                     where the run stands, without waiting
    infrarsi agent note TEXT                  guidance for later proposals (the live console's /note)
    infrarsi agent stop [--now]               stop the search at its next boundary and re-verify the best candidate so far
    infrarsi agent resume [run's options]     continue a stopped or dead run from its checkpoint (`run --resume`)

A session is a directory, <project root>/.infrarsi/sessions/<UTC time>/ (`current` beside them names the newest):
session.json (its state), engine.log (the engine's console), heartbeat, and the request and control files of the session
backend. The engine is detached from the command that started it: a session that disconnects, compacts its context or
has a waiting command killed loses nothing - `infrarsi agent status` finds the run again.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from ..agent.backends import session as S

SESSION_SCHEMA = 'infrarsi-session/1'
HEARTBEAT_S = 5.0                                        # the engine touches <session>/heartbeat this often
DEAD_AFTER_S = 45.0                                      # no heartbeat for this long (and no such process here): it is gone
STARTING_S = 60.0                                        # a new engine has this long to write its first heartbeat
DEFAULT_WAIT_S = 540.0                                   # `agent next`: under the ten minutes a foreground command may take

# optimize's options a session may give `agent start`, and run's for `agent resume`: passed to the engine as they are
# (the model, the questions and the write of the result are not the session's to set: --llm, --yes, --dry-run)
START_OPTIONS = (('--config', 1), ('--entry', 1), ('--command', 1), ('--no-doctor', 0), ('--strategy', 1), ('--tolerance', 1),
                 ('--screens', 1), ('--capability', 1), ('--budget-usd', 1), ('--budget-tokens', 1), ('--wall-hours', 1),
                 ('--target-speedup', 1), ('--target-mfu', 1), ('--out', 1), ('--experience', 1), ('--ablate', 1),
                 ('--on-interrupt', 1), ('--requirements', '*'), ('--requirements-file', 1), ('--requirements-partial', 0),
                 ('--no-requirements', 0), ('--second-llm', 1), ('--verifier-llm', 1), ('--verbose', 0))
RESUME_OPTIONS = (('--tolerance', 1), ('--screens', 1), ('--capability', 1), ('--budget-usd', 1), ('--budget-tokens', 1),
                  ('--wall-hours', 1), ('--target-speedup', 1), ('--target-mfu', 1), ('--experience', 1), ('--ablate', 1),
                  ('--on-interrupt', 1), ('--second-llm', 1), ('--verifier-llm', 1), ('--remeasure-reference', 0),
                  ('--allow-code-change', 0), ('--verbose', 0))

DO = {
    'request': ('Give request {id} to a fresh sub-agent that has not seen this conversation: it reads README.md in the '
                'request directory and writes {answer} there. Do not answer it yourself and do not edit its answer. '
                'Then run: infrarsi agent answer {id}'),
    'running': 'The engine is working (a training run can take minutes). Run `infrarsi agent next` again.',
    'finished': ('The run is over. Read the report (infrarsi report {run_dir}) and tell the user what it says, with its '
                 'numbers. Write the change only when the user agrees: infrarsi apply {run_dir} --write'),
    'failed': ('The engine stopped with an error: read log_tail (the whole console is in {log}). Fix the cause, then '
               '`infrarsi agent start` again{resume}.'),
    'dead': ('The engine process is gone (a restart of the machine, or it was killed). `infrarsi agent resume` continues '
             'the search from its last checkpoint{fresh}.'),
}


class AgentError(RuntimeError):
    """A request to `infrarsi agent` that cannot be done; the message is the one line the session reads."""


def utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')


def sessions_root(root: Path) -> Path:
    return Path(root) / '.infrarsi' / 'sessions'


def project_root(target) -> Path:
    """The project a target belongs to: the directory itself, the manifest's, or the one `optimize` resolves a launch
    script to."""
    p = Path(target or '.').resolve()
    if p.is_dir():
        return p
    if not p.is_file():
        raise AgentError(f'{target}: no such file or directory')
    if p.name != 'infrarsi.yaml':
        from .autopilot import is_script_path, resolve_script_target
        if is_script_path(p):
            return resolve_script_target(p)[0]
    return p.parent


def current(start=None) -> Path | None:
    """The newest session of the project that `start` (default: the current directory) is in; None without one."""
    here = Path(start or '.').resolve()
    for d in (here, *here.parents):
        try:
            name = (sessions_root(d) / 'current').read_text(encoding='utf-8').strip()
        except OSError:
            continue
        if name and (sessions_root(d) / name / 'session.json').is_file():
            return sessions_root(d) / name
    return None


def locate(session=None, project=None) -> Path:
    """The session a command means: --session, the engine's own (INFRARSI_SESSION_DIR), else the project's newest."""
    d = session or os.environ.get(S.ENV_DIR)
    if d:
        d = Path(d).resolve()
        if not (d / 'session.json').is_file():
            raise AgentError(f'{d} is not a session directory (no session.json)')
        return d
    d = current(project)
    if d is None:
        raise AgentError(f'no session under {Path(project or ".").resolve()}: start one with `infrarsi agent start`')
    return d


def load(d: Path) -> dict:
    return S.read_json(Path(d) / 'session.json') or {}


def save(d: Path, **changes) -> dict:
    state = {**load(d), **changes}
    S.write_json(Path(d) / 'session.json', state)
    return state


def pid_alive(pid) -> bool:
    if type(pid) is not int or pid <= 0:
        return False
    if os.name == 'nt':                                  # (os.kill would end the process there)
        import ctypes
        k = ctypes.windll.kernel32
        k.OpenProcess.restype = ctypes.c_void_p
        handle = k.OpenProcess(0x1000, False, pid)       # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(k.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))) and code.value == 259
        finally:
            k.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    try:                                                 # Linux: an ended process its parent has not collected yet
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except (OSError, IndexError):
        return True


def engine_alive(d: Path, state: dict | None = None) -> bool:
    """Whether the session's engine still runs. Its state must say so; then, on this machine, its process must be there
    (a killed engine is gone at once, whatever its last heartbeat) and its heartbeat not stale (the process number of a
    dead engine can be another process's by now); from another machine that shares the files, the heartbeat alone."""
    state = load(d) if state is None else state
    if state.get('state') not in ('starting', 'running'):
        return False
    if state.get('state') == 'starting':                 # not up yet: the time a start takes, counted from its request
        try:
            t0 = _dt.datetime.fromisoformat(state.get('spawned_utc') or state.get('created_utc')).timestamp()
        except (TypeError, ValueError):
            return False
        return time.time() - t0 < STARTING_S
    try:
        age = time.time() - (Path(d) / 'heartbeat').stat().st_mtime
    except OSError:
        age = None                                       # (none written yet)
    fresh = age is None or age < DEAD_AFTER_S
    eng = state.get('engine') or {}
    if eng.get('host') == socket.gethostname():
        return fresh and pid_alive(eng.get('pid'))
    return age is not None and fresh


def run_dir_of(d: Path) -> Path | None:
    got = S.read_json(Path(d) / 'run.json') or {}
    return Path(got['run_dir']) if got.get('run_dir') else None


def summary(d: Path) -> dict | None:
    """What the run produced, from its result.json: None before there is one."""
    run = run_dir_of(d)
    res = S.read_json(run / 'result.json') if run else None
    if not isinstance(res, dict):
        return {'run_dir': str(run)} if run else None
    promo, search = res.get('promotion') or {}, res.get('search') or {}
    return {'run_dir': str(run), 'outcome': res.get('outcome'), 'strategy': res.get('strategy'),
            'promoted': bool(promo.get('promoted')), 'speedup': promo.get('speedup'),
            'screens': (search.get('used') or {}).get('screens'), 'stop_reason': search.get('stop_reason'),
            'report': str(run / 'REPORT.md') if (run / 'REPORT.md').is_file() else None}


def log_tail(d: Path, lines: int = 12, width: int = 300) -> list[str]:
    path = Path(d) / 'engine.log'
    try:
        with path.open('rb') as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 65536))
            text = f.read().decode('utf-8', errors='replace')
    except OSError:
        return []
    return [ln[:width] for ln in text.splitlines() if ln.strip()][-lines:]


def _request_view(r: S.Request) -> dict:
    meta, n = r.meta, r.rejected()
    out = {'id': r.seq, 'role': r.role, 'dir': str(r.path), 'answer': str(r.path / (meta.get('answer_file') or S.ANSWER)),
           'created_utc': meta.get('created_utc'), 'bytes': meta.get('bytes'),
           'attempts_left': max(0, int(meta.get('max_attempts') or S.MAX_ATTEMPTS) - n)}
    if n:                                                # a reply was handed in and could not be used
        try:
            out['feedback'] = r.text(S.FEEDBACK).strip()
        except OSError:
            pass
    return out


def status(d: Path) -> dict:
    """Where the session's run stands, as `agent status` and `agent next` print it."""
    d = Path(d)
    state = load(d)
    st, alive = state.get('state'), engine_alive(d, state)
    run = run_dir_of(d)
    out = {'session': str(d), 'state': st, 'run_dir': str(run) if run else None}
    waiting = S.pending(d) if alive else []
    if st in ('starting', 'running') and not alive:
        out['state'] = 'dead'
    elif waiting:
        out['state'] = 'request'
        out['request'] = _request_view(waiting[0])
    elif st == 'starting':
        out['state'] = 'running'
    progress = S.read_json(d / 'progress.json')
    if progress:
        out['progress'] = {k: progress.get(k) for k in ('screens_done', 'screens_budget', 'best', 'best_speedup', 'llm_calls', 'utc')}
    if out['state'] in ('finished', 'failed', 'dead'):
        out['result'] = summary(d)
        if state.get('exit_code') is not None:
            out['exit_code'] = state['exit_code']
    out['log_tail'] = log_tail(d, 6 if out['state'] in ('request', 'running') else 14)
    resumable = bool(run and (run / 'search' / 'state' / 'checkpoint.json').is_file())
    fmt = {'run_dir': str(run) if run else '<run dir>', 'log': str(d / 'engine.log'),
           'resume': ' (or `infrarsi agent resume`: the search has a checkpoint)' if resumable else '',
           'fresh': '' if resumable else '; this run has no checkpoint yet, so start it again: `infrarsi agent start`'}
    if out['state'] == 'request':
        fmt.update(id=out['request']['id'], answer=Path(out['request']['answer']).name)
    out['do'] = DO[out['state']].format(**fmt) if out['state'] in DO else None
    return out


def wait_next(d: Path, wait: float = DEFAULT_WAIT_S, *, poll: float = 1.0, sleep=time.sleep) -> dict:
    """`agent next`: the status once it needs the session (a request, the end, a failure, a dead engine), or after `wait`
    seconds of a run that needs nothing."""
    deadline = time.monotonic() + max(0.0, float(wait))
    while True:
        out = status(d)
        if out['state'] != 'running' or time.monotonic() >= deadline:
            return out
        sleep(min(poll, max(0.0, deadline - time.monotonic())))


ENGINE_LOCK, LAUNCH_LOCK = 'engine.lock', 'launch.lock'  # in <project>/.infrarsi/sessions: one engine, one launch at a time
BUSY = 3                                                 # serve's exit status when another engine has the project


hold = S.hold                                            # the operating system's file lock: gone with its holder


@contextlib.contextmanager
def launching(base: Path, patience: float = 90.0):
    """One `start` / `resume` of a project at a time, from its check that no engine runs to the moment its own engine is
    up: a second one (a command sent twice, two tool calls at once) waits here and then finds the run active."""
    base.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    while True:
        lock = hold(base / LAUNCH_LOCK)
        if lock is not None:
            break
        if time.monotonic() - t0 > patience:
            raise AgentError('another `infrarsi agent start` or `resume` of this project is still starting: try again')
        time.sleep(0.2)
    try:
        yield
    finally:
        lock.close()


def _spawn(d: Path, cwd=None):
    """The engine as a process of its own: its own session (POSIX) or process group without a window (Windows), so that
    the end of the command that started it - or of the tool call around that - does not end the run."""
    env = os.environ.copy()
    env[S.ENV_DIR] = str(d)
    env['PYTHONUNBUFFERED'] = '1'
    parent = str(Path(__file__).resolve().parents[2])    # this tree, whether installed or shipped as a directory
    env['PYTHONPATH'] = parent + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    # (a launcher that keeps Python's bytecode cache out of this tree - the skill's - keeps the engine's out as well)
    cache = ['-X', f'pycache_prefix={sys.pycache_prefix}'] if sys.pycache_prefix else []
    cmd = [sys.executable, *cache, '-m', 'infrarsi', 'agent', 'serve', '--session', str(d)]
    with (d / 'engine.log').open('ab') as log:
        kw = {'stdin': subprocess.DEVNULL, 'stdout': log, 'stderr': subprocess.STDOUT, 'env': env, 'cwd': cwd}
        if os.name != 'nt':
            return subprocess.Popen(cmd, start_new_session=True, **kw)
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        try:                                             # out of a job object that ends its processes with the caller
            return subprocess.Popen(cmd, creationflags=flags | 0x01000000, **kw)   # CREATE_BREAKAWAY_FROM_JOB
        except OSError:
            return subprocess.Popen(cmd, creationflags=flags, **kw)


def _launch(d: Path, cwd=None, settle: float = 20.0) -> dict:
    """Start the engine and wait until it is up (or gone again): the status then. An engine that found the project
    taken by another one (BUSY) changed nothing, and nothing is changed for it here: the caller is told the run is active."""
    proc = _spawn(d, cwd)
    t0 = time.monotonic()
    while time.monotonic() - t0 < settle:
        if load(d).get('state') != 'starting' or proc.poll() is not None:
            break
        time.sleep(0.2)
    if proc.poll() == BUSY:
        raise AgentError('a run of this project is active (another engine has it): `infrarsi agent next` goes on with it')
    if proc.poll() is not None and load(d).get('state') == 'starting':   # it died before it could say why
        save(d, state='failed', exit_code=proc.returncode, ended_utc=utc())
    return status(d)


def _engine_model(model) -> str:
    return 'session' + (f':{model}' if model else '')


def preflight(command: str = 'start') -> None:
    """(`command`: the launching command in progress, `start` or `resume`: the one the user is asked to let through.)
    What the engine needs outside the project, tried from this process (the engine it starts has the same limits): the
    ledger in its home directory can be appended to, as every run does at its end, and its receipt-signing key can be
    used. A session inside a sandbox fails here, at once and in words for the user, not after the search at the
    write-back; and the agent is told that the key is not its to make (the engine signs every run's receipt with it: a key
    the agent wrote is a key the agent knows). Nothing more is asked of the home directory than a run asks: one that is
    read-only but holds a ledger that can be written passes."""
    from ..harness import signing
    from .flow import home, ledger_path
    h = home()
    try:                                                 # what every run does there at its end: append to the ledger
        h.mkdir(parents=True, exist_ok=True)             # (opened for appending, nothing written; made empty when new)
        with ledger_path().open('a', encoding='utf-8'):
            pass
    except OSError as exc:
        raise AgentError(
            f'the engine cannot write its home directory {h} from this session ({type(exc).__name__} at its ledger); it '
            'keeps the experience library and its ledger there. If this session runs in a sandbox, ask the user to allow '
            f'writing that directory or to let `infrarsi agent {command}` run outside the sandbox (INFRARSI_HOME names '
            'another directory). Do not create files there yourself.') from None
    try:
        signing.key_id()
    except Exception as exc:                             # noqa: BLE001 - whatever stops the key: said in one line
        raise AgentError(
            f'the engine cannot use its receipt-signing key {signing.key_path()} from this session '
            f'({type(exc).__name__}: {str(exc)[:160]}). The engine makes and protects this key itself: do not create, copy '
            f'or edit it. If this session runs in a sandbox, ask the user to let `infrarsi agent {command}` run outside the '
            'sandbox (the engine then runs outside it too).') from None


def start(target='.', *, model=None, options=(), cwd=None) -> dict:
    root = project_root(target)
    base = sessions_root(root)
    preflight()
    with launching(base):
        prev = current(root)
        if prev is not None and engine_alive(prev):
            raise AgentError(f'a run of this project is active (session {prev}): `infrarsi agent next` goes on with it, '
                             '`infrarsi agent stop` stops it')
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%d-%H%M%S')
        for k in range(1, 100):
            d = base / (stamp + (f'-{k}' if k > 1 else ''))
            try:
                d.mkdir(exist_ok=False)
                break
            except FileExistsError:
                continue
        else:
            raise AgentError(f'{base}: a hundred sessions in one second')
        argv = ['optimize', str(Path(target or '.').resolve()), '--llm', _engine_model(model), '--yes', '--dry-run', *options]
        S.write_json(d / 'session.json', {'schema': SESSION_SCHEMA, 'id': d.name, 'root': str(root), 'created_utc': utc(),
                                          'spawned_utc': utc(), 'state': 'starting', 'mode': 'start', 'model': model,
                                          'options': list(options), 'argv': argv})
        try:
            out = _launch(d, cwd)
        except AgentError:                               # another engine has the project: this session never began
            shutil.rmtree(d, ignore_errors=True)
            raise
        S.write_text(base / 'current', d.name + '\n')
    from ..agent.capability import infer
    out['model'] = model
    if not model:                                        # (the search scales its trust in the answers by the model)
        out['note'] = (f'No --model was given: the search treats the answers as a middle model\'s (capability '
                       f'{infer(None):.2f}). If this session\'s model is a stronger one, `infrarsi agent stop --now` and '
                       'start again with --model <its name>.')
    return out


def _carried(started: list, given: list) -> list[str]:
    """The options of a resume: those the run started with that `run --resume` takes too (its budget, its targets, its
    tolerance ...: the run goes on as it was asked), unless the resume gives the option itself."""
    arity, keep = dict(START_OPTIONS), {f for f, _ in RESUME_OPTIONS}
    out, i = [], 0
    while i < len(started):
        flag = started[i]
        n = 1 if arity.get(flag) in (1, '*') else 0
        if flag in keep and flag not in given:
            out += started[i:i + 1 + n]
        i += 1 + n
    return out + list(given)


def resume(d: Path, *, model=None, options=(), cwd=None) -> dict:
    d = Path(d)
    if not d.is_dir():
        raise AgentError(f'nothing to resume: {d} is not a session directory')
    preflight('resume')
    with launching(d.parent):
        state = load(d)
        if engine_alive(d, state):
            raise AgentError('the run is active: `infrarsi agent next` goes on with it')
        run = run_dir_of(d)
        if run is None or not (run / 'search' / 'state' / 'checkpoint.json').is_file():
            raise AgentError('nothing to resume: this run has no search checkpoint (it stopped before or at the start of '
                             'its search); start again with `infrarsi agent start`')
        meta = S.read_json(run / 'run_meta.json') or {}
        manifest = meta.get('manifest') or str(Path(state.get('root') or '.') / 'infrarsi.yaml')
        stop = d / S.CONTROL / 'stop'
        if stop.is_file():                               # the stop that ended the last part is not this part's
            os.replace(stop, stop.with_name('stop.' + _dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%d-%H%M%S')))
        for r in S.pending(d):                           # asked by the engine that is gone: the new one asks again, and
            S.write_text(r.path / S.ABANDONED, utc() + '\n')   # the session must not be shown the old request meanwhile
        model = model or state.get('model')
        options = _carried(list(state.get('options') or []), list(options))
        argv = ['run', str(manifest), '--resume', str(run), '--llm', _engine_model(model), '--yes', *options]
        save(d, state='starting', mode='resume', model=model, argv=argv, options=options, spawned_utc=utc(),
             exit_code=None, ended_utc=None, error=None, resumes=int(state.get('resumes') or 0) + 1)
        try:
            return _launch(d, cwd)
        except AgentError:                               # the project's engine is another session's: this one stays as it was
            S.write_json(d / 'session.json', state)
            raise


def serve(d: Path) -> int:
    """The engine's process. It first becomes the project's one engine (the lock is its for as long as it lives); when
    another engine has the project it ends at once with BUSY and has touched nothing of the session."""
    d = Path(d).resolve()
    lock = hold(d.parent / ENGINE_LOCK)
    if lock is None:
        print('infrarsi agent serve: another engine runs for this project; this one does not start', file=sys.stderr)
        return BUSY
    try:
        # the project's engine names its session as the current one itself: a launcher that is killed while it waits for
        # the engine to come up (a tool's time limit) then leaves no engine that `status` and `next` cannot find
        S.write_text(d.parent / 'current', d.name + '\n')
        return _serve(d)
    finally:
        lock.close()


def _serve(d: Path) -> int:
    """The command of session.json run as the CLI runs it, a heartbeat beside it, the session's state kept (running, then
    finished or failed with the exit status)."""
    state = load(d)
    if not state.get('argv'):
        print(f'infrarsi agent serve: {d} is not a session (no argv in session.json)', file=sys.stderr)
        return 2
    os.environ[S.ENV_DIR] = str(d)
    beat, done = d / 'heartbeat', threading.Event()

    def heartbeat():
        while True:
            try:
                beat.touch()
            except OSError:
                pass
            if done.wait(HEARTBEAT_S):
                return
    threading.Thread(target=heartbeat, name='infrarsi-heartbeat', daemon=True).start()
    save(d, state='running', engine={'pid': os.getpid(), 'host': socket.gethostname(), 'started_utc': utc()})
    code, error = 1, None
    try:
        from ..cli.main import main as cli_main
        try:
            rc = cli_main(list(state['argv']))
            code = rc if type(rc) is int else 0
        except SystemExit as exc:                        # the CLI's own way of ending with a status (or a message)
            if exc.code is None or type(exc.code) is int:
                code = exc.code or 0
            else:
                print(exc.code, file=sys.stderr)
                code, error = 1, str(exc.code)[:500]
        except KeyboardInterrupt:
            code, error = 130, 'interrupted'
        except Exception as exc:                         # noqa: BLE001 - the traceback goes to engine.log, the line to the state
            traceback.print_exc()
            code, error = 1, f'{type(exc).__name__}: {str(exc)[:500]}'
    finally:
        done.set()
        for r in S.pending(d):                           # asked and never answered: no engine will read an answer now
            S.close(r, 'the engine ended before this call was answered', status='ENDED')
        save(d, state='finished' if code == 0 else 'failed', exit_code=code, error=error, ended_utc=utc(), result=summary(d))
    return code


def answer(d: Path, seq: int, *, file=None, tokens=None, by=None) -> dict:
    req = S.find(d, seq)
    if req is None:
        raise AgentError(f'no request {seq} in session {d}')
    path = Path(file) if file else req.path / (req.meta.get('answer_file') or S.ANSWER)
    try:
        text = path.read_text(encoding='utf-8-sig')      # (a byte-order mark some editors and shells write is not JSON)
    except OSError as exc:
        raise AgentError(f'no answer to hand in: {path} cannot be read ({type(exc).__name__}); the answerer writes it there')
    except UnicodeDecodeError:
        raise AgentError(f'{path} is not UTF-8 text: write the answer as UTF-8')
    try:
        got = S.submit(req, text, tokens=tokens, by=by)
    except TimeoutError as exc:                          # another `agent answer` of this request has not finished
        raise AgentError(f'{exc}: try again') from None
    out = {'ok': got['ok'], 'request': req.seq, 'error': got['error'], 'closed': got['closed'], 'attempts_left': got['left']}
    if got['ok']:
        out['do'] = 'Handed in. Run `infrarsi agent next`.'
    elif not got['closed']:
        out['do'] = (f'The answer was not usable. Have the same kind of fresh sub-agent fix {path.name} in {req.path} (it '
                     f'reads {S.FEEDBACK} there), then run `infrarsi agent answer {req.seq}` again.')
    else:
        out['do'] = 'This request is closed; the engine goes on. Run `infrarsi agent next`.'
    return out


def note(d: Path, text: str) -> dict:
    text = str(text).strip()
    if not text:
        raise AgentError('an empty note')
    path = Path(d) / S.CONTROL / 'notes.jsonl'
    path.parent.mkdir(exist_ok=True)
    with path.open('a', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps({'utc': utc(), 'text': text[:2000]}, ensure_ascii=False) + '\n')
    return {'ok': True, 'do': 'Saved: later proposals see it as guidance (the checks stay as they are). A requirement the '
                              'run must obey belongs in --requirements of a new run.'}


def stop(d: Path, *, now: bool = False) -> dict:
    """The stop a session asks for: at the search's next boundary, with the re-verification of the best candidate (the
    live console's /stop). `now`: the engine's own interrupt, as Ctrl-C twice - the run under way is cut and so is a
    re-verification that has begun; one that had already finished stands, and the status returned says which it was."""
    d = Path(d)
    state = load(d)
    if not engine_alive(d, state):
        return {'ok': False, **status(d)}
    (d / S.CONTROL).mkdir(exist_ok=True)
    S.write_text(d / S.CONTROL / 'stop', utc() + '\n')
    if not now:
        return {'ok': True, 'do': 'Stop requested: the engine finishes the run under way, re-verifies the best candidate so '
                                  'far and writes its report. Keep running `infrarsi agent next` until it is finished.'}
    # no start or resume of this project while its engine is ended and its state set right: the state written below is
    # then this engine's, not that of one a resume started meanwhile
    with launching(d.parent):
        state = load(d)
        pid = (state.get('engine') or {}).get('pid')
        if (state.get('engine') or {}).get('host') != socket.gethostname() or not pid_alive(pid):
            raise AgentError('the engine is not up yet, or does not run on this machine: `infrarsi agent status` says where '
                             'it stands (the stop at its next boundary is requested)')
        if os.name == 'nt':                              # no signal reaches a process without a console: end its tree
            subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], capture_output=True)
        else:
            for pause in (0.3, 0.0):                     # the first ends the search, the second its re-verification
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    break
                time.sleep(pause)
        t0 = time.monotonic()
        while pid_alive(pid) and time.monotonic() - t0 < 30:
            time.sleep(0.2)
        now_state = load(d)
        if not pid_alive(pid) and now_state.get('state') in ('starting', 'running') \
                and (now_state.get('engine') or {}).get('pid') == pid:   # it ended without writing its own end
            for r in S.pending(d):                       # (nor closing what it had asked: no engine reads an answer now)
                S.close(r, 'the engine was ended before this call was answered (agent stop --now)', status='ENDED')
            save(d, state='failed', exit_code=130, error='stopped at once (agent stop --now)', ended_utc=utc(),
                 result=summary(d))
    out = status(d)
    promoted = bool((out.get('result') or {}).get('promoted'))
    out['do'] = ('The engine was told to end at once. ' + (
        'Its re-verification had already finished, so the report\'s recommendation stands: read it before telling the user.'
        if promoted else 'Nothing was promoted.')
        + (f' `infrarsi finish {out["run_dir"]} --dry-run` re-verifies what the search measured, if the user wants that '
           '(without --dry-run it would also write the winner into the code).'
           if out.get('run_dir') and not promoted else ''))
    return {'ok': True, **out}


def _options(parser, table) -> None:
    for flag, arity in table:
        if arity == 0:
            parser.add_argument(flag, action='store_true', dest='opt' + flag, help=argparse.SUPPRESS)
        elif arity == '*':
            parser.add_argument(flag, action='append', dest='opt' + flag, help=argparse.SUPPRESS)
        else:
            parser.add_argument(flag, dest='opt' + flag, help=argparse.SUPPRESS)


def _passed(a, table) -> list[str]:
    out = []
    for flag, arity in table:
        v = getattr(a, 'opt' + flag, None)
        if arity == 0:
            out += [flag] if v else []
        elif arity == '*':
            for item in v or []:
                out += [flag, str(item)]
        elif v is not None:
            out += [flag, str(v)]
    return out


def build() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog='infrarsi agent', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='action', required=True, metavar='{start,next,answer,status,note,stop,resume}')
    where = argparse.ArgumentParser(add_help=False)
    where.add_argument('--session', help='the session directory (default: the newest session of the project)')
    where.add_argument('--project', help='the project directory (default: the current directory)')
    p = sub.add_parser('start', help='start a run in the background',
                       epilog='optimize\'s options are passed on: ' + ' '.join(f for f, _ in START_OPTIONS))
    p.add_argument('target', nargs='?', default='.', help='a launch script, a training script, a project directory or infrarsi.yaml')
    p.add_argument('--model', help='the model of this session\'s agent, by name (e.g. claude-opus-5-5): the search scales '
                                   'its trust in the answers by it, as llm.capability; unnamed, a middle value is used')
    _options(p, START_OPTIONS)
    p = sub.add_parser('next', parents=[where], help='wait until the run needs the session')
    p.add_argument('--wait', type=float, default=DEFAULT_WAIT_S, help=f'seconds to wait at most (default: {DEFAULT_WAIT_S:.0f})')
    p = sub.add_parser('answer', parents=[where], help='hand in the answer to a request')
    p.add_argument('id', type=int, help='the request\'s number')
    p.add_argument('--file', help='the answer (default: answer.json in the request\'s directory)')
    p.add_argument('--tokens', type=int, help='the tokens the answer cost, when the session knows them (else estimated)')
    p.add_argument('--by', help='who answered (a model name), for the record')
    sub.add_parser('status', parents=[where], help='where the run stands')
    p = sub.add_parser('note', parents=[where], help='guidance for later proposals')
    p.add_argument('text')
    p = sub.add_parser('stop', parents=[where], help='stop the search and re-verify the best candidate so far')
    p.add_argument('--now', action='store_true',
                   help='end at once: the run under way is cut, and so is a re-verification that has begun (one that had '
                        'already finished stands; the status printed says which)')
    p = sub.add_parser('resume', parents=[where], help='continue a stopped or dead run from its checkpoint',
                       epilog='run\'s options are passed on: ' + ' '.join(f for f, _ in RESUME_OPTIONS))
    p.add_argument('--model', help='as for start (default: the one the run started with)')
    _options(p, RESUME_OPTIONS)
    p = sub.add_parser('serve', help=argparse.SUPPRESS)
    p.add_argument('--session', required=True)
    return ap


def cli(argv) -> int:
    """`infrarsi agent <argv>`: the exit status (0; 1 for an answer that was not taken or a run that is not there to act
    on). Every action but serve prints one JSON object."""
    a = build().parse_args(list(argv))
    if a.action == 'serve':
        return serve(Path(a.session))
    if a.action == 'start':
        out = start(a.target, model=a.model, options=_passed(a, START_OPTIONS))
    else:
        d = locate(a.session, a.project)
        if a.action == 'next':
            out = wait_next(d, a.wait)
        elif a.action == 'status':
            out = status(d)
        elif a.action == 'answer':
            out = answer(d, a.id, file=a.file, tokens=a.tokens, by=a.by)
        elif a.action == 'note':
            out = note(d, a.text)
        elif a.action == 'stop':
            out = stop(d, now=a.now)
        else:
            out = resume(d, model=a.model, options=_passed(a, RESUME_OPTIONS))
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if a.action in ('answer', 'note', 'stop'):           # done or not done
        return 0 if out.get('ok') else 1
    return 0 if out.get('state') not in ('failed', 'dead') else 1
