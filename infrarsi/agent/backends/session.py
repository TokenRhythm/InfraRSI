"""Session backend: a run's model calls answered by the coding-agent session that started it (Claude Code, Codex, ...).

`infrarsi agent start` (project/agent_session.py) runs the engine as its own process with `--llm session`. Every call the
search makes is written to the session directory as a request and the engine waits for its answer; the session's agent
sees the request with `infrarsi agent next`, has a fresh sub-agent answer it, and hands the answer in with
`infrarsi agent answer`. The request is the stateless call every other backend sends - the system prompt, the prompt and
the answer's schema exactly as the search built them - so the search is the one `--llm claude-cli` runs; who answers is
the only difference.

    <session dir>/requests/0007-hypotheses/
        request.json     the call: sequence number, role, digest, limits (written once, last, by the engine)
        system.txt       its system prompt
        prompt.txt       its prompt
        schema.json      the JSON schema of the answer (when the call has one)
        README.md        what the answerer does
        answer.json      <- the answerer writes the reply here
        rejected-1.json  a reply that could not be used, with why (feedback.txt: the latest reason)
        accepted.json    the reply the engine takes (written atomically by `agent answer`, or by the engine when it closes
                         the call itself: no answer in time, the run stopped)
        abandoned        the engine that asked is gone; a resumed run asks again
    <session dir>/control/stop, control/notes.jsonl      `agent stop`, `agent note`: read between runs (SessionControl)
    <session dir>/progress.json                            where the search stood at the last boundary
    <session dir>/run.json                                 the run directory the engine writes to

A reply is checked where it is handed in and again where it is read (one JSON object, the call's schema: `check_reply`,
the CLI backends' rule). A reply that fails is returned to the answerer with the reason; the `max_attempts`-th failure
(2: one re-ask, as the CLI backends make), or an earlier one after which the run's tokens leave no room for another,
closes the call with that error, as a CLI backend's retries end. Tokens: every reply handed in is charged, a returned
one too - the count the session reports for it when it gives one, else an estimate from the sizes of the request and
the reply (the accounting is then marked incomplete); dollars are 0, as for codex-cli: the session's own plan pays.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import json
import math
import os
import re
import socket
import time
from pathlib import Path

from ..jsonschema_lite import validate
from .base import LLMResult, parse_json_object

ENV_DIR = 'INFRARSI_SESSION_DIR'                         # the session directory (`infrarsi agent start` sets it)
ENV_MODEL = 'INFRARSI_SESSION_MODEL'                     # the model of the session's agent, by name (for llm.capability)
REQUEST_SCHEMA = 'infrarsi-session-request/1'
ANSWER_SCHEMA = 'infrarsi-session-answer/1'
REQUESTS, CONTROL = 'requests', 'control'
ANSWER, ACCEPTED, ABANDONED, FEEDBACK = 'answer.json', 'accepted.json', 'abandoned', 'feedback.txt'
MAX_ATTEMPTS = 2                                         # replies to one call before it is closed with the last error:
#                                                          one re-ask, as the CLI backends make (project/llm.make_driver)
NOTES_KEPT = 12                                          # as the live console keeps them (project/live.py)

README = """# InfraRSI request {seq:04d} ({role})

One question from the InfraRSI search engine, complete in this directory. The engine measures what you answer; it does
not see anything else you say.

1. Read `system.txt` (the instructions for this request) and `prompt.txt` (the request).{schema_line}
2. Write your answer to `{answer}` in this directory: exactly {shape}, as `system.txt` asks.{cap_line}
3. Stop. Run nothing, start no training run, edit no other file, and read no file outside this directory: the engine
   applies and measures your answer itself, and a question it did not ask cannot be answered here.

Answer from the request alone. Earlier requests of this run are not context: each one already holds what the engine
wants you to know.
"""


def utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')


def write_text(path: Path, text: str) -> None:
    """The whole file or none of it: a reader on the other side of the protocol never sees half a write. (Windows
    refuses to replace a file another process has open - the reader of a moment ago: tried again for a second.)"""
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.{os.urandom(3).hex()}.tmp')   # (its own name: two writers, two files)
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    for left in range(20, -1, -1):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if not left:
                raise
            time.sleep(0.05)


def write_json(path: Path, obj) -> None:
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=1, default=str) + '\n')


def read_json(path: Path):
    """The object in a JSON file; None when it is not there or not JSON. (A file being replaced at this moment cannot
    be opened on Windows: tried again for a moment, so that a status is not read as missing.)"""
    for left in range(10, -1, -1):
        try:
            return json.loads(Path(path).read_text(encoding='utf-8'))
        except FileNotFoundError:
            return None
        except ValueError:
            return None
        except OSError:
            if not left:
                return None
            time.sleep(0.02)


def request_digest(role: str, system: str, prompt: str, schema) -> str:
    body = json.dumps([role, system, prompt, schema], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(body.encode('utf-8')).hexdigest()


def size_estimate(request_bytes: int, reply: str | None, overhead: int) -> int:
    """One reply's tokens from the sizes when the session gave no count: 3.5 bytes a token, as the other backends bound
    an unknown usage, plus the overhead they reserve around a call."""
    return math.ceil(int(request_bytes) / 3.5) + int(overhead) + math.ceil(len((reply or '').encode('utf-8')) / 3.5)


def estimate_tokens(system: str, prompt: str, reply: str | None, overhead: int) -> int:
    return size_estimate(len(system.encode('utf-8')) + len(prompt.encode('utf-8')), reply, overhead)


def check_reply(text, schema, expect_json: bool = True) -> tuple[dict | None, str | None, str]:
    """(action, error, parse mode) of a reply, by the rule of the CLI backends: one JSON object (alone, or the one fenced
    block), then the call's schema. A free-text call (expect_json False) has no action and no error."""
    if not expect_json:
        return None, None, 'not_parsed'
    action, mode = parse_json_object(text)
    if action is None:
        return None, 'invalid structured JSON action', mode
    problems = validate(action, schema) if schema else []
    if problems:
        return None, 'schema: ' + '; '.join(problems[:6]), mode
    return action, None, mode


class Request:
    """One request directory. What it holds decides its state: `accepted.json` answered (or closed), `abandoned`
    abandoned, else pending. request.json is written once, last, so a directory without it is not a request yet."""

    def __init__(self, path):
        self.path = Path(path)
        self._meta = None

    @property
    def meta(self) -> dict:
        if self._meta is None:
            self._meta = read_json(self.path / 'request.json') or {}
        return self._meta

    @property
    def seq(self) -> int:
        return int(self.meta.get('seq') or 0)

    @property
    def role(self) -> str:
        return str(self.meta.get('role') or '')

    @property
    def state(self) -> str:
        if (self.path / ACCEPTED).is_file():
            return 'answered'
        return 'abandoned' if (self.path / ABANDONED).exists() else 'pending'

    def rejected(self) -> int:
        return len(list(self.path.glob('rejected-*.json')))

    def schema(self):
        return read_json(self.path / 'schema.json')

    def accepted(self) -> dict | None:
        return read_json(self.path / ACCEPTED)

    def text(self, name: str) -> str:
        return (self.path / name).read_text(encoding='utf-8')


def requests(directory) -> list[Request]:
    """The session's requests by sequence number."""
    root = Path(directory) / REQUESTS
    out = [Request(d) for d in (root.iterdir() if root.is_dir() else ()) if (d / 'request.json').is_file()]
    return sorted((r for r in out if r.meta.get('schema') == REQUEST_SCHEMA), key=lambda r: r.seq)


def pending(directory) -> list[Request]:
    return [r for r in requests(directory) if r.state == 'pending']


def find(directory, seq: int) -> Request | None:
    return next((r for r in requests(directory) if r.seq == int(seq)), None)


def usage(req: Request, final: dict | None = None) -> tuple[int, bool]:
    """(tokens, whether every count was reported) of all the replies handed in for one request: the ones that were
    returned to the answerer, and `final` when it is an answer that passed. (The failure that ends a call is already
    among the returned ones.) A reply that came without a count is estimated from the sizes."""
    meta = req.meta
    request_bytes, overhead = sum((meta.get('bytes') or {}).values()), int(meta.get('overhead_tokens') or 0)
    rows = [read_json(p) or {} for p in req.path.glob('rejected-*.json')]
    if final is not None and isinstance(final.get('text'), str) and not final.get('error'):
        rows.append(final)
    total, reported = 0, True
    for row in rows:
        count = row.get('llm_tokens')
        if type(count) is int and count >= 0:
            total += count
        else:
            total += size_estimate(request_bytes, row.get('text'), overhead)
            reported = False
    return total, reported


def hold(path):
    """An exclusive lock on `path` for as long as the returned file stays open; None when another process (or another
    open file of this one) has it. The operating system drops the lock when its holder ends, however that happens:
    there is never a stale one to judge."""
    f = open(path, 'a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


@contextlib.contextmanager
def changing(req: Request, patience: float = 60.0):
    """One writer at a time changes a request - hands in a reply, closes it: the number of a returned reply, its record
    and the accepted record are then one step for everyone else. The lock is the operating system's (`hold`), so a
    writer that dies half way leaves the request as it was and no lock behind."""
    t0 = time.monotonic()
    while True:
        lock = hold(req.path / '.lock')
        if lock is not None:
            break
        if time.monotonic() - t0 > patience:
            raise TimeoutError(f'request {req.seq} has been held by another process for {patience:.0f} s')
        time.sleep(0.005)
    try:
        yield
    finally:
        lock.close()


def close(req: Request, error: str, *, status: str) -> bool:
    """Close a pending request without an answer (no answer in time, the run stopped): a reply handed in later is refused.
    False when an answer got there first."""
    with changing(req):
        return _accept(req, {'schema': ANSWER_SCHEMA, 'seq': req.seq, 'digest': req.meta.get('digest'), 'text': None,
                             'error': error, 'status': status, 'attempts': req.rejected(), 'llm_tokens': None,
                             'answered_utc': utc()})


def _accept(req: Request, record: dict) -> bool:
    """accepted.json once (the caller is `changing` the request): False when the request has one already."""
    if (req.path / ACCEPTED).exists():
        return False
    write_json(req.path / ACCEPTED, record)
    return True


def submit(req: Request, text: str, *, tokens: int | None = None, by: str | None = None) -> dict:
    """`infrarsi agent answer`: check a reply and hand it to the engine. {'ok', 'error', 'closed', 'attempts', 'left'}:
    ok - the engine takes it; not ok and not closed - the answerer may fix it and hand it in again (`left` more times);
    closed - the call is over (answered before, abandoned, or this was its last failure: the engine gets the error)."""
    with changing(req):
        return _submit(req, text, tokens, by)


def exhausted(req: Request) -> bool:
    """(Called while `changing` the request.) Whether the request is over because its replies are used up - the limit of
    attempts, or the tokens for another one - the last of them recorded as unusable; the record that ends the call is
    written here when it is not there yet. Recording a reply as unusable is the step that counts; an answer command that
    dies right after it, before the call is closed, leaves this for the next writer - another answer, or the engine that
    waits - to finish."""
    if (req.path / ACCEPTED).exists():
        return True
    n = req.rejected()
    limit = max(1, int(req.meta.get('max_attempts') or MAX_ATTEMPTS))
    if not n or (n < limit and room_for_another(req)):
        return False
    last = read_json(req.path / f'rejected-{n}.json') or {}
    _accept(req, {'schema': ANSWER_SCHEMA, 'seq': req.seq, 'digest': req.meta.get('digest'), 'text': last.get('text'),
                  'error': last.get('error') or 'invalid structured JSON action', 'status': 'RETURNED', 'attempts': n,
                  'parse_mode': last.get('parse_mode'), 'llm_tokens': last.get('llm_tokens'), 'answered_utc': utc(),
                  **({'by': last['by']} if last.get('by') else {})})
    return True


def _submit(req: Request, text: str, tokens, by) -> dict:
    exhausted(req)                                       # (a close an earlier answer command left half made)
    state = req.state
    if state != 'pending':
        was = req.accepted() or {}
        why = (f'was closed without an answer ({was.get("error")})' if state == 'answered' and was.get('text') is None
               else f'is already {state}')
        return {'ok': False, 'closed': True, 'attempts': req.rejected(), 'left': 0,
                'error': f'request {req.seq} {why}: nothing was handed in'}
    limit = max(1, int(req.meta.get('max_attempts') or MAX_ATTEMPTS))
    action, error, mode = check_reply(text, req.schema(), bool(req.meta.get('expect_json', True)))
    n = req.rejected() + 1
    if tokens is not None and (type(tokens) is not int or tokens < 0):
        tokens = None
    record = {'schema': ANSWER_SCHEMA, 'seq': req.seq, 'digest': req.meta.get('digest'), 'text': text, 'error': error,
              'status': 'RETURNED', 'attempts': n, 'parse_mode': mode, 'llm_tokens': tokens, 'answered_utc': utc(),
              **({'by': str(by)[:120]} if by else {})}
    if error is None:
        if not _accept(req, record):
            return {'ok': False, 'closed': True, 'attempts': n - 1, 'left': 0,
                    'error': f'request {req.seq} was closed while this answer was handed in'}
        return {'ok': True, 'error': None, 'closed': True, 'attempts': n, 'left': 0}
    write_json(req.path / f'rejected-{n}.json', {'text': text, 'error': error, 'llm_tokens': tokens, 'utc': utc(),
                                                 'parse_mode': mode, **({'by': str(by)[:120]} if by else {})})
    last = exhausted(req)                                # the last reply the call allows: it is closed with this error
    write_text(req.path / FEEDBACK, f'Your answer could not be used: {error}\n'
               + ('This was the last attempt: the engine goes on without an answer to this request.\n' if last else
                  f'Fix {ANSWER} and hand it in again ({limit - n} more attempt(s)). It must be exactly one JSON object, '
                  'as system.txt asks, with no text around it.\n'))
    return {'ok': False, 'error': error, 'closed': last, 'attempts': n, 'left': 0 if last else limit - n}


def room_for_another(req: Request) -> bool:
    """Whether the run's token budget allows one more reply to a request, by the CLI backends' rule for a re-ask: what
    the call was given, less what its replies cost so far, still covers the request with its reserve and a short answer."""
    allowance = req.meta.get('token_allowance')
    if type(allowance) is not int:
        return True
    meta = req.meta
    request_bytes, overhead = sum((meta.get('bytes') or {}).values()), int(meta.get('overhead_tokens') or 0)
    need = math.ceil(math.ceil(request_bytes / 3.5) * 1.25) + overhead + 256
    return allowance - usage(req)[0] >= need


class SessionControl:
    """`agent stop` and `agent note` for a search without a terminal: the file form of project/live.LiveConsole. Only the
    search thread calls it, between runs, and it writes where the search stands for `agent status` / `agent next`."""

    def __init__(self, directory):
        self.directory = Path(directory)

    def stop_requested(self) -> bool:
        return (self.directory / CONTROL / 'stop').is_file()

    def _new_notes(self, session) -> list[str]:
        """The notes this search has not read yet. How many it has read is kept on the session (`session_notes_read`),
        next to the notes themselves, so a checkpoint holds both or neither: a resumed run reads on from where its
        restored state stands - no note twice, and none lost to an engine that ended between reading and saving."""
        try:
            lines = (self.directory / CONTROL / 'notes.jsonl').read_text(encoding='utf-8').splitlines()
        except OSError:
            return []
        done = int(getattr(session, 'session_notes_read', 0) or 0)
        out = []
        for line in lines[done:]:
            try:
                text = json.loads(line).get('text')
            except (ValueError, AttributeError):
                continue
            if isinstance(text, str) and text.strip():
                out.append(text.strip()[:2000])
        if len(lines) > done:
            session.session_notes_read = len(lines)
        return out

    def _progress(self, session) -> None:
        try:
            used = session.ledger.snapshot()
            best = session.best_valid() or 'c0'
            ref, top = session.candidates.get('c0'), session.candidates.get(best)
            t_ref = ((ref.verdict or {}).get('step_time') if ref else None)
            t_best = ((top.verdict or {}).get('step_time') if top else None)
            write_json(self.directory / 'progress.json', {
                'utc': utc(), 'screens_done': used.get('screens'), 'screens_budget': session.budget.screens,
                'best': best, 'best_speedup': round(t_ref / t_best, 4) if t_ref and t_best else None,
                'reference_step_time': t_ref, 'llm_calls': len(session.llm_calls), 'used': used})
        except Exception:                                # noqa: BLE001 - a status file never stops the search
            pass

    def poll(self, session, checkpoint=None):
        self._progress(session)
        notes = list(getattr(session, 'live_notes', []))
        new = self._new_notes(session)
        for text in new:
            session.event('live_message', {'text': '/note ' + text})
            notes = (notes + [text])[-NOTES_KEPT:]
        if new:
            session.live_notes = notes
        if session.hooks is not None:
            session.hooks.live_notes = notes
        if new and checkpoint is not None:
            checkpoint()
        if self.stop_requested():
            from ..search import USER_STOP
            session.stop_flag = USER_STOP
            if checkpoint is not None:
                checkpoint()

    def before_trial(self, session):
        if self.stop_requested():
            from ..search import USER_STOP
            session.stop_flag = USER_STOP


class SessionBackend:
    kind = 'session'
    call_overhead_tokens = 4000                          # an agent's own context around a request, as the CLI backends reserve

    def __init__(self, *, directory=None, model: str | None = None, max_output_tokens: int = 32000,
                 answer_timeout: float | None = None, poll_seconds: float = 0.5, max_attempts: int = MAX_ATTEMPTS,
                 out_root=None):
        d = directory or os.environ.get(ENV_DIR)
        if not d:
            raise ValueError('the session backend answers a run started from a coding-agent session: start it with '
                             f'`infrarsi agent start` (it sets {ENV_DIR}), or use another --llm backend')
        self.directory = Path(d).resolve()
        if not self.directory.is_dir():
            raise FileNotFoundError(f'the session directory {self.directory} does not exist')
        self.model = str(model or os.environ.get(ENV_MODEL) or 'session')
        self.max_output_tokens = int(max_output_tokens)
        self.answer_timeout = None if answer_timeout is None else float(answer_timeout)
        self.poll_seconds = max(0.01, float(poll_seconds))
        self.max_attempts = max(1, int(max_attempts))
        self.session_control = SessionControl(self.directory)   # the flow polls it between runs (project/flow.py)
        self._out_root = None
        (self.directory / REQUESTS).mkdir(exist_ok=True)
        (self.directory / CONTROL).mkdir(exist_ok=True)
        for r in pending(self.directory):                # asked by an engine that is gone: this one asks again
            write_text(r.path / ABANDONED, utc() + '\n')
        if out_root:
            self.out_root = out_root

    @property
    def out_root(self):
        return self._out_root

    @out_root.setter
    def out_root(self, value) -> None:
        """The flow sets the run's model to <run dir>/llm_calls (flow.run_project): that names the run directory for the
        session. A session backend that is the debate's second model or its patch checker is given a directory below
        that one (agent/debate.DebateDriver.attach: llm_calls/second, llm_calls/verifier): it names nothing."""
        self._out_root = Path(value) if value else None
        if self._out_root is not None and self._out_root.name == 'llm_calls':
            write_json(self.directory / 'run.json', {'run_dir': str(self._out_root.parent), 'utc': utc()})

    def describe(self) -> dict:
        return {'kind': self.kind, 'model': self.model, 'directory': str(self.directory), 'max_attempts': self.max_attempts}

    def _result(self, *, action=None, text=None, tokens=0, wall=0.0, complete=True, error=None, attempts=1, meta=None):
        return LLMResult(action=action, text=text, llm_tokens=tokens, llm_usd=0.0, wall_seconds=wall,
                         accounting_complete=complete, error=error, model=self.model, observed_models=[],
                         attempts=attempts, meta=meta or {})

    def _ask(self, system, prompt, schema, role, expect_json, cap, allowance) -> Request:
        root = self.directory / REQUESTS
        # the next number after every directory there, a half-made one too (an engine that ended between making its
        # directory and its request.json: that directory is no request, and its number is not free)
        taken = [int(m.group(1)) for m in (re.match(r'(\d+)-', p.name) for p in root.iterdir()) if m]
        seq = max(taken, default=0) + 1
        path = root / f'{seq:04d}-{re.sub(r"[^A-Za-z0-9_.-]+", "-", str(role))[:40] or "call"}'
        path.mkdir(parents=True, exist_ok=False)
        write_text(path / 'system.txt', system)
        write_text(path / 'prompt.txt', prompt)
        if schema:
            write_json(path / 'schema.json', schema)
        answer = ANSWER if expect_json else 'answer.txt'
        write_text(path / 'README.md', README.format(
            seq=seq, role=role, answer=answer,
            schema_line=' `schema.json` is the JSON schema your answer must satisfy.' if schema else '',
            shape='one JSON object, with no Markdown fence and no text before or after it' if expect_json else 'your reply as plain text',
            cap_line=f' The engine allows about {int(cap)} tokens for it.' if cap else ''))
        write_json(path / 'request.json', {
            'schema': REQUEST_SCHEMA, 'seq': seq, 'role': role, 'digest': request_digest(role, system, prompt, schema),
            'created_utc': utc(), 'expect_json': bool(expect_json), 'answer_file': answer, 'max_attempts': self.max_attempts,
            'max_output_tokens': cap, 'token_allowance': allowance, 'overhead_tokens': self.call_overhead_tokens,
            'bytes': {'system': len(system.encode('utf-8')), 'prompt': len(prompt.encode('utf-8'))},
            'engine': {'pid': os.getpid(), 'host': socket.gethostname()}})
        return Request(path)

    def _wait(self, req: Request, limit: float | None) -> dict:
        """The accepted record of a request: the answer handed in, or the close this engine writes when the time it has
        for the call is over or the run is being stopped (an answer that got there first still counts)."""
        t0 = time.monotonic()
        while True:
            got = req.accepted()
            if got is not None:
                return got
            if (req.path / ABANDONED).exists():          # another engine took the session over (a resume): not this one's
                return {'text': None, 'status': 'ABANDONED', 'attempts': req.rejected(),
                        'error': 'the request was abandoned: another engine took over this session'}
            try:
                if self.session_control.stop_requested():
                    close(req, 'the run was stopped while this call waited for its answer', status='STOPPED')
                    continue
                if limit is not None and time.monotonic() - t0 >= limit:
                    close(req, f'no answer from the session within {limit:.0f} s', status='TIMEOUT')
                    continue
                if req.rejected():                       # replies were returned: the last one the call allows ends it,
                    with changing(req):                  # also when the command that recorded it died before closing
                        if exhausted(req):
                            continue
            except TimeoutError:                         # an answer is being handed in for a long time: look again
                pass
            time.sleep(self.poll_seconds)

    def complete(self, system: str, prompt: str, *, schema: dict | None = None, call_dir=None, role: str = 'proposer',
                 expect_json: bool = True, timeout: float | None = None, max_output_tokens: int | None = None,
                 token_allowance: int | None = None, **_ignored) -> LLMResult:
        """One call: a request written, its answer waited for. The wait ends with the answer, with the time the caller
        gives the call (`timeout`: what the search's wall-clock budget has left) or `answer_timeout`, or when the run is
        stopped; a call that ends without an answer is an error the search handles as any failed call. Every reply that
        was handed in is charged (`usage`), also the ones returned as unusable before the call ended."""
        if role == 'preflight':                          # nothing to ask an agent: the session is attached (__init__)
            return self._result(action={'ok': True}, text='{"ok": true}', meta={'status': 'LOCAL'})
        if self.session_control.stop_requested():        # the search is on its way to its next boundary, where it stops:
            return self._result(error='the run is being stopped: the call was not made',   # no request for the session
                                meta={'status': 'STOPPED'})
        cap = int(min(self.max_output_tokens, max_output_tokens or self.max_output_tokens))
        limits = [float(x) for x in (self.answer_timeout, timeout) if x is not None]
        started = time.monotonic()
        req = self._ask(system, prompt, schema, role, expect_json, cap, token_allowance)
        got = self._wait(req, min(limits) if limits else None)
        wall = time.monotonic() - started
        meta = {'call_id': f'{req.seq:04d}', 'call_dir': str(req.path), 'status': got.get('status') or 'RETURNED',
                'answered_by': got.get('by')}
        text = got.get('text') if isinstance(got.get('text'), str) else None
        tokens, known = usage(req, got)                  # every reply handed in, the returned ones too
        meta['usage_estimated'] = not known
        if text is None:                                 # closed without an answer
            res = self._result(tokens=tokens, wall=wall, complete=known, error=str(got.get('error') or 'no answer'),
                               attempts=max(1, int(got.get('attempts') or 1)), meta=meta)
        else:
            action, error, mode = check_reply(text, schema, expect_json)
            res = self._result(action=action, text=text, tokens=tokens, wall=wall, complete=known, error=error,
                               attempts=max(1, int(got.get('attempts') or 1)), meta={**meta, 'parse_mode': mode})
        self._record(req, system, prompt, schema, res, call_dir)
        return res

    def _record(self, req: Request, system, prompt, schema, res: LLMResult, call_dir) -> None:
        """The call in the run's own llm_calls/ directory, as the CLI backends leave one (request.json, result.txt,
        receipt.json): the run directory then holds every prompt and answer without the session's."""
        base = Path(call_dir) if call_dir else (self.out_root / req.path.name if self.out_root else None)
        if base is None:
            return
        try:
            base.mkdir(parents=True, exist_ok=True)
            write_json(base / 'request.json', {'system': system, 'prompt': prompt, 'schema': schema})
            if res.text is not None:
                write_text(base / 'result.txt', res.text)
            write_json(base / 'receipt.json', {'schema': 'infrarsi-session-call-v1', 'role': req.role, 'model': self.model,
                                              'request_dir': str(req.path), 'digest': req.meta.get('digest'),
                                              'llm_tokens': res.llm_tokens, 'llm_usd': res.llm_usd, 'error': res.error,
                                              'attempts': res.attempts, 'wall_seconds': round(res.wall_seconds, 3),
                                              'accounting_complete': res.accounting_complete, **res.meta})
        except OSError:                                  # a record, not a part of the call
            pass
