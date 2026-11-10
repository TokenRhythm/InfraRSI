"""Remote access to WINH0-7 (paramiko) and LocalRemote, an in-process stand-in used by CPU tests.

Both expose the same small API: call(node, func, **kw) runs a function of harness.remote_side on that node and
returns its JSON result; put_bytes/get_bytes move files (any node sees the shared GPFS /data); sh runs a shell
command. Calls marked non-idempotent are never retried after the request may have reached the node.
"""
from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from infrarsi.contracts import PROJECT_ROOT

from . import remote_side

SENTINEL = '@@INFRARSI_RESULT@@'
NON_IDEMPOTENT = {'launch_wrapper', 'install_code', 'write_stop'}
_SIDE_SOURCE = Path(remote_side.__file__).read_text(encoding='utf-8')


class RemoteError(RuntimeError):
    pass


def node_index(node: str | int) -> int:
    if isinstance(node, int):
        return node
    if not node.startswith('WINH'):
        raise ValueError(f'not a WINH node: {node!r}')
    return int(node[4:])


def node_name(node: str | int) -> str:
    return f'WINH{node_index(node)}'


def _load_winh_remote():
    """The workspace's existing login helper (ports + credential source); imported, never modified."""
    path = PROJECT_ROOT / 'winh8_20260925' / 'remote.py'
    spec = importlib.util.spec_from_file_location('winh8_remote', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Remote:
    def call(self, node, func: str, timeout: float = 120, **kw) -> Any:
        raise NotImplementedError

    def put_bytes(self, node, data: bytes, path: str) -> None:
        raise NotImplementedError

    def get_bytes(self, node, path: str) -> bytes | None:
        raise NotImplementedError

    def sh(self, node, cmd: str, timeout: float = 120) -> dict:
        raise NotImplementedError

    def close(self) -> None:
        pass

    # convenience wrappers
    def read_json(self, node, path, default=None):
        return self.call(node, 'read_json', path=path, default=default)

    def write_json(self, node, path, obj, exclusive=False):
        return self.call(node, 'write_json', path=path, obj=obj, exclusive=exclusive)


class WinhRemote(Remote):
    """Cached paramiko clients per node; SFTP per call under a per-node lock."""

    def __init__(self, retries: int = 3, connect: Callable | None = None):
        self._connect = connect or _load_winh_remote().connect
        self._clients: dict[int, Any] = {}
        self._locks: dict[int, threading.Lock] = {}
        self._guard = threading.Lock()
        self.retries = retries

    def _lock(self, k):
        with self._guard:
            return self._locks.setdefault(k, threading.Lock())

    def _client(self, k, fresh=False):
        with self._lock(k):
            c = self._clients.get(k)
            if fresh or c is None or c.get_transport() is None or not c.get_transport().is_active():
                if c is not None:
                    try:
                        c.close()
                    except Exception:      # noqa: BLE001
                        pass
                c = self._connect(k)
                c.get_transport().set_keepalive(30)
                self._clients[k] = c
            return c

    def _exec(self, k, command: str, stdin: bytes | None, timeout: float) -> dict:
        chan = self._client(k).get_transport().open_session(timeout=20)
        try:
            chan.exec_command(command)
            if stdin is not None:
                chan.sendall(stdin)
            chan.shutdown_write()
            out, err = bytearray(), bytearray()
            end = time.monotonic() + timeout
            while True:
                while chan.recv_ready():
                    out.extend(chan.recv(1 << 16))
                while chan.recv_stderr_ready():
                    err.extend(chan.recv_stderr(1 << 16))
                # The exit status can arrive while the end of a large stdout still waits behind the channel window
                # (2 MiB): leave only once both streams have hit EOF (SSH sends one EOF for stdout and stderr; a close
                # without EOF ends them too) and both buffers are drained (F1 incident 2026-10-21 19:48).
                if (chan.exit_status_ready() and (chan.eof_received or chan.closed)
                        and not chan.recv_ready() and not chan.recv_stderr_ready()):
                    break
                if time.monotonic() > end:
                    raise TimeoutError(f'WINH{k}: remote command exceeded {timeout}s')
                time.sleep(0.03)
            return {'code': chan.recv_exit_status(), 'stdout': out.decode(errors='replace'),
                    'stderr': err.decode(errors='replace')}
        finally:
            chan.close()

    def _retrying(self, k, fn, idempotent=True):
        last = None
        for attempt in range(self.retries if idempotent else 1):
            try:
                return fn()
            except (OSError, EOFError, socket.timeout, TimeoutError) as e:     # paramiko errors subclass these
                last = e
                time.sleep(2 * (attempt + 1))
                self._client(k, fresh=True)
            except Exception as e:                                             # noqa: BLE001
                if type(e).__module__.startswith('paramiko'):
                    last = e
                    time.sleep(2 * (attempt + 1))
                    self._client(k, fresh=True)
                    continue
                raise
        raise RemoteError(f'WINH{k}: transport failed: {last!r}')

    def sh(self, node, cmd, timeout=120):
        k = node_index(node)
        return self._retrying(k, lambda: self._exec(k, cmd, None, timeout))

    def call(self, node, func, timeout=120, **kw):
        k = node_index(node)
        script = (_SIDE_SOURCE + '\nimport json as _j, traceback as _tb\n'
                  f'_kw = _j.loads({json.dumps(json.dumps(kw))})\n'
                  'try:\n'
                  f'    _r = {{"ok": True, "value": {func}(**_kw)}}\n'
                  'except BaseException as _e:\n'
                  '    _r = {"ok": False, "error": repr(_e), "traceback": _tb.format_exc()}\n'
                  f'print("\\n{SENTINEL}" + _j.dumps(_r))\n')
        res = self._retrying(k, lambda: self._exec(k, 'python3 -', script.encode(), timeout),
                             idempotent=func not in NON_IDEMPOTENT)
        if SENTINEL not in res['stdout']:
            raise RemoteError(f'WINH{k} {func}: no result (code {res["code"]}): {res["stderr"][-2000:]}')
        payload = json.loads(res['stdout'].rsplit(SENTINEL, 1)[1].strip().splitlines()[0])
        if not payload['ok']:
            raise RemoteError(f'WINH{k} {func}: {payload["error"]}\n{payload["traceback"][-3000:]}')
        return payload['value']

    def put_bytes(self, node, data, path):
        k = node_index(node)

        def go():
            sftp = self._client(k).open_sftp()
            try:
                tmp = f'{path}.part{int(time.time() * 1000) % 100000}'
                with sftp.open(tmp, 'wb') as f:
                    f.set_pipelined(True)
                    f.write(data)
                sftp.posix_rename(tmp, path)
            finally:
                sftp.close()
        self._retrying(k, go)

    def get_bytes(self, node, path):
        k = node_index(node)

        def go():
            sftp = self._client(k).open_sftp()
            try:
                with sftp.open(path, 'rb') as f:
                    f.prefetch()
                    return f.read()
            except FileNotFoundError:
                return None
            finally:
                sftp.close()
        return self._retrying(k, go)

    def close(self):
        for c in self._clients.values():
            try:
                c.close()
            except Exception:              # noqa: BLE001
                pass
        self._clients.clear()


class LocalRemote(Remote):
    """In-process fake: '/data/...' maps to <root>/data/...; GPU state and launches are programmable.

    gpu_apps: {node_index: [ {pid, name, used_mib, sid}, ... ]} foreign or own processes reported by gpu_status.
    launcher: callable(node_index, kwargs, remote) invoked for launch_wrapper (simulates the node wrapper).
    """

    def __init__(self, root: str | Path, gpu_apps: dict | None = None, launcher: Callable | None = None,
                 ips: dict | None = None, busy_ports: set | None = None):
        self.root = Path(root)
        self.gpu_apps = gpu_apps or {}
        self.gpu_mem = {}
        self.launcher = launcher
        self.ips = ips or {k: [f'172.27.0.{10 + k}'] for k in range(8)}
        self.busy_ports = busy_ports or set()
        self.calls: list[tuple[int, str]] = []

    def local(self, path: str) -> Path:
        if not path.startswith('/'):
            raise ValueError(path)
        return self.root / path.lstrip('/')

    def _map(self, v):
        if isinstance(v, str) and v.startswith('/data/'):
            return str(self.local(v))
        if isinstance(v, list):
            return [self._map(x) for x in v]
        return v

    def call(self, node, func, timeout=120, **kw):
        k = node_index(node)
        self.calls.append((k, func))
        if func == 'gpu_status':
            gpus = [{'index': g, 'mem_used_mib': self.gpu_mem.get(k, {}).get(g, 0), 'mem_total_mib': 81559, 'util_pct': 0}
                    for g in range(8)]
            return {'hostname': f'fake-winh{k}', 'ips': self.ips[k], 'utc': remote_side.utc(), 'error': None,
                    'compute_apps': list(self.gpu_apps.get(k, [])), 'gpus': gpus}
        if func == 'port_free':
            return kw['port'] not in self.busy_ports
        if func == 'first_free_port':
            return next((p for p in kw['candidates'] if p not in self.busy_ports), None)
        if func == 'launch_wrapper':
            if self.launcher is None:
                raise RemoteError('LocalRemote has no launcher')
            return self.launcher(k, kw, self)
        if func == 'disk_free':
            return {'path': kw.get('path', '/data'), 'free_bytes': 68 * 2 ** 40, 'total_bytes': 140 * 2 ** 40}
        mapped = {key: self._map(v) for key, v in kw.items()}
        fn = getattr(remote_side, func)
        try:
            return json.loads(json.dumps(fn(**mapped)))          # same JSON round trip as the real transport
        except Exception as e:                                    # noqa: BLE001
            raise RemoteError(f'WINH{k} {func}: {e!r}') from e

    def put_bytes(self, node, data, path):
        p = self.local(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get_bytes(self, node, path):
        p = self.local(path)
        return p.read_bytes() if p.exists() else None

    def sh(self, node, cmd, timeout=120):
        return {'code': 0, 'stdout': '', 'stderr': ''}


_default: Remote | None = None


def default_remote() -> Remote:
    global _default
    if _default is None:
        _default = WinhRemote()
    return _default


def set_default_remote(remote: Remote | None) -> None:
    global _default
    _default = remote


if __name__ == '__main__':                     # python -m infrarsi.harness.remote 2 gpu_status
    r = WinhRemote()
    print(json.dumps(r.call(int(sys.argv[1]), sys.argv[2], **json.loads(sys.argv[3] if len(sys.argv) > 3 else '{}')), indent=2))
