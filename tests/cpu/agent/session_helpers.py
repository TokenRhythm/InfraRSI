"""Stand-ins for a coding-agent session in tests of the session backend (agent/backends/session.py) and of `infrarsi agent`."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from infrarsi.agent.backends import session as S
from infrarsi.project import agent_session as A


class Answerer(threading.Thread):
    """The session's side of a run: every pending request is answered with `reply(role, system, prompt)` (a dict, or the
    text to hand in), written to the request's answer file and handed in as `infrarsi agent answer` does. `tokens`: the
    count reported with each answer (None: none, the engine estimates). `seen`: (seq, role, system, prompt) in order;
    `handed`: what each hand-in returned."""

    def __init__(self, directory, reply, *, tokens=None, pause: float = 0.005):
        super().__init__(name='test-answerer', daemon=True)
        self.directory, self.reply, self.tokens, self.pause = Path(directory), reply, tokens, pause
        self.seen, self.handed, self.errors, self.done = [], [], [], threading.Event()

    def run(self):
        while not self.done.is_set():
            try:
                for r in S.pending(self.directory):
                    if any(seq == r.seq for seq, *_ in self.seen) and r.rejected() == 0:
                        continue                         # (handed in; the engine has not read it yet)
                    system, prompt = r.text('system.txt'), r.text('prompt.txt')
                    self.seen.append((r.seq, r.role, system, prompt))
                    item = self.reply(r.role, system, prompt)
                    text = item if isinstance(item, str) else json.dumps(item)
                    (r.path / (r.meta.get('answer_file') or S.ANSWER)).write_text(text, encoding='utf-8')
                    self.handed.append(A.answer(self.directory, r.seq, tokens=self.tokens))
            except Exception as exc:                     # noqa: BLE001 - the test reads it
                self.errors.append(exc)
                return
            time.sleep(self.pause)

    def stop(self):
        self.done.set()
        self.join(timeout=10)
        assert not self.errors, self.errors

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


def session_dir(root: Path) -> Path:
    """An empty session directory, as `infrarsi agent start` makes one, whose engine is this process (so it is alive)."""
    import os
    import socket
    d = Path(root) / '.infrarsi' / 'sessions' / 'test'
    d.mkdir(parents=True)
    S.write_json(d / 'session.json', {'schema': A.SESSION_SCHEMA, 'id': 'test', 'root': str(root), 'state': 'running',
                                      'engine': {'pid': os.getpid(), 'host': socket.gethostname()}})
    return d


def wait_for(cond, timeout: float = 20.0, pause: float = 0.01):
    """`cond()` once it is true (its value); an AssertionError after `timeout` seconds."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        got = cond()
        if got:
            return got
        time.sleep(pause)
    raise AssertionError('timed out waiting')
