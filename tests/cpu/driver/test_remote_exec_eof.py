"""harness.remote.WinhRemote._exec on a stand-in SSH channel (F1 incident 2026-10-21 19:48, f1_prompts_20261021b c1).

sshd sends the exit status when the child exits, while the end of a large stdout can still wait behind the channel
window (paramiko: 2 MiB). The pre-fix loop left as soon as the exit status was ready and the buffers were momentarily
empty, so the tail was lost; the fixed loop reads until both streams hit EOF. Arrival over time is simulated by the
stand-in channel: one chunk arrives per poll (the time.sleep between iterations of _exec's loop)."""
import time as _time
import types

from infrarsi.harness import remote as RM


class StandInChannel:
    """paramiko.Channel as _exec uses it. The exit status is ready from the start (it overtook the data); stdout and
    stderr chunks arrive one per poll; then EOF (or, with eof=False, a close without EOF)."""

    def __init__(self, out_chunks, err_chunks=(), eof=True, first=1):
        self.pending = [('o', c) for c in out_chunks[first:]] + [('e', c) for c in err_chunks]
        self.out, self.err = bytearray(b''.join(out_chunks[:first])), bytearray()
        self.eof_received, self.closed, self._eof = 0, False, eof

    def tick(self):                                       # the next chunk arrives
        if self.pending:
            kind, chunk = self.pending.pop(0)
            (self.out if kind == 'o' else self.err).extend(chunk)
        if not self.pending:
            if self._eof:
                self.eof_received = True
            else:
                self.closed = True

    def exec_command(self, cmd):
        self.command = cmd

    def sendall(self, data):
        pass

    def shutdown_write(self):
        pass

    def recv_ready(self):
        return bool(self.out)

    def recv_stderr_ready(self):
        return bool(self.err)

    def recv(self, n):
        b = bytes(self.out[:n])
        del self.out[:n]
        return b

    def recv_stderr(self, n):
        b = bytes(self.err[:n])
        del self.err[:n]
        return b

    def exit_status_ready(self):
        return True

    def recv_exit_status(self):
        return 0

    def close(self):
        pass


def _remote(monkeypatch, chan):
    transport = types.SimpleNamespace(is_active=lambda: True, set_keepalive=lambda s: None,
                                      open_session=lambda timeout=20: chan)
    client = types.SimpleNamespace(get_transport=lambda: transport, close=lambda: None)
    monkeypatch.setattr(RM, 'time', types.SimpleNamespace(sleep=lambda s: chan.tick(), monotonic=_time.monotonic,
                                                          time=_time.time))
    return RM.WinhRemote(retries=1, connect=lambda k: client)


def _pre_fix_exec(chan):
    """_exec's read loop before the fix (harness/remote.py:118 as of 2026-10-20), on the same stand-in channel."""
    out = bytearray()
    while True:
        while chan.recv_ready():
            out.extend(chan.recv(1 << 16))
        while chan.recv_stderr_ready():
            chan.recv_stderr(1 << 16)
        if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
            break
        chan.tick()
    return bytes(out)


CHUNKS = [bytes([65 + i % 26]) * (64 << 10) for i in range(48)] + [b'\n@@END@@\n']      # ~3 MB, past the 2 MiB window


def test_the_pre_fix_loop_loses_the_tail_that_arrives_after_the_exit_status():
    got = _pre_fix_exec(StandInChannel(CHUNKS))
    assert got == CHUNKS[0] and not got.endswith(b'@@END@@\n')                 # all but the first chunk lost


def test_exec_reads_until_eof_after_the_exit_status(monkeypatch):
    chan = StandInChannel(CHUNKS)
    res = _remote(monkeypatch, chan).sh('WINH0', 'python3 - big', timeout=30)
    assert res['stdout'].encode() == b''.join(CHUNKS) and res['code'] == 0


def test_both_streams_are_read_to_eof(monkeypatch):
    chan = StandInChannel(CHUNKS[:8], err_chunks=[b'warning %d\n' % i for i in range(5)])
    res = _remote(monkeypatch, chan).sh('WINH0', 'cmd', timeout=30)
    assert res['stdout'].encode() == b''.join(CHUNKS[:8])
    assert res['stderr'] == ''.join('warning %d\n' % i for i in range(5))


def test_a_close_without_eof_still_ends_the_read(monkeypatch):
    chan = StandInChannel(CHUNKS[:4], eof=False)
    res = _remote(monkeypatch, chan).sh('WINH0', 'cmd', timeout=30)
    assert res['stdout'].encode() == b''.join(CHUNKS[:4])


def test_output_that_arrived_before_the_exit_status_is_unchanged(monkeypatch):
    chan = StandInChannel([b'small reply\n'], first=1)
    res = _remote(monkeypatch, chan).sh('WINH0', 'cmd', timeout=30)
    assert res['stdout'] == 'small reply\n'
