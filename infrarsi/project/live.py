"""Ctrl-G conversation panel. Only the search thread calls the model or changes search state."""
from __future__ import annotations

import codecs
import collections
import json
import os
import queue
import select
import sys
import threading

from ..agent.search import USER_STOP


class LiveConsole:
    def __init__(self, driver=None, *, enabled=True, stdin=None, stdout=None, log=print):
        self.driver, self.enabled = driver, enabled
        self.stdin, self.stdout = stdin or sys.stdin, stdout or sys.stdout
        self.log = log
        self.messages = queue.Queue(maxsize=8)
        self.closed = threading.Event()
        self.paused = threading.Event()
        self.stopping = threading.Event()
        self.lock = threading.RLock()
        self.panel, self.buffer = False, ''
        self.logs = collections.deque(maxlen=100)
        self.snapshot = 'Completed 0 full runs; search is starting.'
        self.thread, self.saved = None, None

    def __enter__(self):
        if not self.enabled or os.name != 'posix' or not all(getattr(s, 'isatty', lambda: False)() for s in (self.stdin, self.stdout)):
            return self
        import termios
        try:
            self.fd = self.stdin.fileno()
            self.saved = termios.tcgetattr(self.fd)
            attrs = termios.tcgetattr(self.fd)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)  # leave ISIG enabled: Ctrl-C still interrupts
            attrs[6][termios.VMIN], attrs[6][termios.VTIME] = 1, 0
            termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
            self.thread = threading.Thread(target=self._listen, name='infrarsi-chat', daemon=True)
            self.thread.start()
            self.log('[infrarsi] 💬 Ctrl-G: open chat. /status /pause /resume /stop; Esc closes the panel.')
        except (OSError, ValueError, AttributeError, RuntimeError, termios.error):
            self._shutdown()
            self.thread = None
            return self
        except BaseException:
            self._shutdown()
            raise
        return self

    def _restore(self):
        if self.saved is not None:
            import termios
            try:
                termios.tcsetattr(self.fd, termios.TCSANOW, self.saved)
            except (OSError, termios.error):
                pass
            self.saved = None

    def _shutdown(self):
        self.closed.set()
        self.paused.clear()
        try:
            if self.thread is not None and self.thread.ident is not None:
                self.thread.join(timeout=1)
        finally:
            self._restore()
            self.close_panel()
    def __exit__(self, *_):
        self._shutdown()
        if not self.messages.empty():
            self.log('[infrarsi] Search ended with unread chat messages; no further model calls were made.')

    def _out(self, text):
        self.stdout.write(text)
        self.stdout.flush()

    def emit(self, text):
        with self.lock:
            if self.panel:
                self._out('\r\x1b[2K' + str(text) + '\nchat> ' + self.buffer)
            else:
                self.log('[infrarsi] ' + str(text))

    def write_log(self, text):
        with self.lock:
            if self.panel:
                self.logs.append(text)
            else:
                self.log(text)

    def close_panel(self):
        with self.lock:
            if self.panel:
                self._out('\r\x1b[2K[chat closed; search continues]\n')
                self.panel = False
                self.buffer = ''
                for text in self.logs:
                    self.log(text)
                self.logs.clear()

    def submit(self, text):
        text = text.strip()
        if not text:
            return
        if text == '/status':
            self.emit(self.snapshot)
        elif text == '/pause':
            self.paused.set()
            self.emit('Pause requested: the current trial or confirmation pair finishes first. /resume continues; wall-clock budget keeps running.')
        elif text == '/resume':
            self.paused.clear()
            self.emit('Continuing the search.')
        elif text == '/stop':
            self.stopping.set()
            self.paused.clear()
            self.emit('Stop requested: finish this trial or confirmation pair, save state, then verify the best candidate.')
        elif text in ('/close', '/quit'):
            self.close_panel()
        elif text.startswith('/note '):
            try:
                self.messages.put_nowait(text[:2000])
                self.emit('Guidance queued for future proposals; existing checks remain enforced.')
            except queue.Full:
                self.emit('Message queue is full.')
        elif text.startswith('/'):
            self.emit('Commands: /status /pause /resume /stop /close. Plain text asks the model; /note TEXT adds search guidance.')
        else:
            try:
                self.messages.put_nowait(text[:2000])
                self.emit('Message queued; the model replies at the next trial boundary.')
            except queue.Full:
                self.emit('Eight messages are pending. Wait for the next trial boundary before sending more.')

    def _listen(self):
        decoder = codecs.getincrementaldecoder('utf-8')('replace')
        try:
            while not self.closed.is_set():
                if not select.select([self.fd], [], [], .1)[0]:
                    continue
                raw = os.read(self.fd, 256)
                if not raw:
                    break
                for ch in decoder.decode(raw):
                    with self.lock:
                        if ch == '\x07':
                            if self.panel:
                                self.close_panel()
                            else:
                                self.panel = True
                                self._out('\n┌─ InfraRSI chat ──────────────────────────\n'
                                          '│ /status /pause /resume /stop /close\n'
                                          '│ /note TEXT: guidance, not an enforced requirement\n'
                                          '│ Replies arrive after the current trial. Esc closes.\n└─\nchat> ')
                        elif ch == '\x04':
                            self.closed.set()
                            break
                        elif not self.panel:
                            continue
                        elif ch == '\x1b':
                            self.close_panel()
                        elif ch in ('\r', '\n'):
                            text, self.buffer = self.buffer, ''
                            self.submit(text)
                            if self.panel:
                                self._out('\r\x1b[2Kchat> ')
                        elif ch in ('\x7f', '\b'):
                            self.buffer = self.buffer[:-1]
                            self._out('\r\x1b[2Kchat> ' + self.buffer)
                        elif ch.isprintable() and len(self.buffer) < 2000:
                            self.buffer += ch
                            self._out(ch)
        except (OSError, ValueError):
            pass
        finally:
            self.closed.set()
            self.paused.clear()

    def poll(self, session, checkpoint=None):
        """Called by the search thread, between trials. Conversation and charges enter the checkpoint."""
        best = session.candidates.get(session.best_valid() or 'c0')
        self.snapshot = (f'Completed {session.ledger.used["screens"]} full runs; '
                         f'best={best.id if best else "none"}; used={session.ledger.snapshot()}')
        notes = getattr(session, 'live_notes', [])
        if session.hooks is not None:
            session.hooks.live_notes = notes
        if self.paused.is_set() and checkpoint is not None:
            checkpoint()
        while True:
            while not self.messages.empty():
                try:
                    text = self.messages.get_nowait()
                except queue.Empty:
                    break
                session.event('live_message', {'text': text})
                if text.startswith('/note '):
                    notes = (notes + [text[6:]])[-12:]
                    session.live_notes = notes
                    if session.hooks is not None:
                        session.hooks.live_notes = notes
                    self.emit('Guidance saved for future model proposals. Hard constraints require a new run with --requirements.')
                else:
                    self._reply(session, text)
                if checkpoint is not None:
                    checkpoint()
            if self.stopping.is_set():
                session.stop_flag = USER_STOP
                if checkpoint is not None:
                    checkpoint()
                return
            if not self.paused.is_set() or self.closed.is_set() or session.ledger.exhausted():
                return
            try:
                self.closed.wait(.1)
            except KeyboardInterrupt:
                if checkpoint is not None:
                    checkpoint()
                raise

    def before_trial(self, session):
        """Honor controls between fidelities; model messages wait for a checkpointable proposal boundary."""
        while self.paused.is_set() and not self.closed.is_set() and not session.ledger.exhausted():
            if self.stopping.is_set():
                break
            self.closed.wait(.1)
        if self.stopping.is_set():
            session.stop_flag = USER_STOP

    def _reply(self, session, text):
        if self.driver is None:
            self.emit(self.snapshot + ' No model configured; /note is saved, but this strategy cannot interpret it.')
            return
        remaining = session.budget.llm_tokens - session.ledger.used['llm_tokens']
        history = getattr(session, 'live_chat_history', [])
        prompt = json.dumps({'conversation': history[-8:], 'question': text, 'status': self.snapshot, 'task': session.hooks.task_view() if session.hooks else {},
                             'guidance': getattr(session, 'live_notes', []), 'recent_trials': session.screens[-4:]},
                            ensure_ascii=False, default=str)
        # Leave room for the bounded reply and the backend's own framing.
        if session.ledger.exhausted() or remaining < len(prompt.encode()) + 6000:
            self.emit('Chat skipped: insufficient remaining search budget.')
            return
        system = ('Answer the user about this live optimization run, in their language. Use measured evidence. '
                  'You cannot execute commands or change constraints. Explain /note for advisory guidance; '
                  'hard requirements need a new run. Never claim a change was applied. Reply as JSON {"reply": "..."}.')
        schema = {'type': 'object', 'properties': {'reply': {'type': 'string'}}, 'required': ['reply'], 'additionalProperties': False}
        try:
            result = self.driver.complete(system, prompt, schema=schema, role='live_chat', max_output_tokens=1000,
                                          timeout=min(120, max(0, session.budget.wall_seconds - session.ledger.wall())),
                                          token_allowance=remaining,
                                          max_usd=None if session.budget.llm_usd is None else max(0, session.budget.llm_usd - session.ledger.used['llm_usd']))
            session.charge_llm(result, 'live_chat')
            reply = str(result.error or (result.action or {}).get('reply') or 'The model returned no reply.')
        except Exception as exc:
            reply = f'Chat unavailable: {type(exc).__name__}: {exc}'
        session.live_chat_history = (history + [{'user': text, 'assistant': reply}])[-8:]
        session.event('live_reply', {'text': reply})
        self.emit(reply)
