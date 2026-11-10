"""Remote execution for ext runners and the native-method agent (EXT_INTERFACES §5, §6; decision D3).

NodeOps is the node interface the runners use: probe GPUs, leases and per-GPU claims, file transfer, hashing,
detached launches, token-scoped cleanup. WinhNodeOps ships node_ext.py over harness.remote.WinhRemote.sh (paramiko,
same login helper as winh8_20260925/remote.py); LocalNodeOps runs the same functions in-process on a local directory
(CPU tests) with a programmable probe and launcher.

Untrusted commands (native agents, candidate training) run as nobody (uid 65534) with a cleared environment:
  setpriv --reuid=65534 --regid=65534 --clear-groups env -i HOME=<home> PATH=<env bin>:/usr/bin:/bin
      CUDA_VISIBLE_DEVICES=<gpus> HF_HUB_OFFLINE=1 ... INFRARSI_TOKEN=<token> bash -c '<cmd>'
A non-login shell by default (login=True gives EXT_INTERFACES' `bash -lc`): the WINH login profile only prints a MOTD
banner (/etc/profile.d/banner.sh) into every output, and PATH is set explicitly (exported again inside the shell).
INFRARSI_TOKEN marks every process a run starts, so `kill_token` can end exactly those and nothing else.

Occupancy (before every launch): no foreign compute process anywhere on the node, nothing on the target GPUs, no
unexplained memory on them, and the node lease is free or ours. Processes count as ours when their cwd or command
line is under /data/infrarsi/ext/runs/ (co-located ext arms on other GPUs of the same node).
"""
from __future__ import annotations

import base64
import json
import os
import shlex
import socket
import uuid
from pathlib import Path

from ...contracts import RemotePaths, utc_now
from ...harness.leases import OccupiedError, make_lease
from . import node_ext

EXT_ROOT = '/data/infrarsi/ext'
REQUEST_DIR = f'{EXT_ROOT}/.requests'           # root-only: node_ext requests too large for the command line
ARGV_BUDGET = 96_000                            # bytes of a shipped command (Linux MAX_ARG_STRLEN is 131072 per argument)
EXT_RUNS = EXT_ROOT + '/runs'
LEASE_DIR = RemotePaths.leases
GPU_CLAIM_DIR = LEASE_DIR + '/ext'           # per-GPU claims of co-located ext arms (the node lease stays the gate)
OWNER = 'infrarsi-ext'
NOBODY = 65534
IDLE_MEM_MIB = 2048
OFFLINE_ENV = {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'PYTHONUNBUFFERED': '1',
               'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8'}
_NODE_SRC = Path(node_ext.__file__).read_text(encoding='utf-8')


def q(s) -> str:
    return shlex.quote(str(s))


def new_token(prefix: str = 'isx') -> str:
    return f'{prefix}-{uuid.uuid4().hex[:20]}'


# ---------------------------------------------------------------- commands
def user_name(uid: int) -> str:
    return 'nobody' if int(uid) == NOBODY else f'isx{int(uid)}'


def nobody_env(*, home: str, path_dirs=(), gpus=(), token: str | None = None, extra: dict | None = None,
               uid: int = NOBODY) -> dict:
    env = {'HOME': home, 'USER': user_name(uid), 'LOGNAME': user_name(uid), 'PATH': ':'.join([*path_dirs, '/usr/bin', '/bin']),
           'CUDA_VISIBLE_DEVICES': ','.join(str(int(g)) for g in gpus), 'TMPDIR': f'{home}/tmp', **OFFLINE_ENV}
    if token:
        env[node_ext.TOKEN_VAR] = token
    env.update({k: str(v) for k, v in (extra or {}).items()})
    return env


def nobody_cmd(cmd: str, *, home: str, cwd: str | None = None, path_dirs=(), gpus=(), token: str | None = None,
               extra: dict | None = None, timeout: float | None = None, kill_after: int = 10, login: bool = False,
               uid: int = NOBODY, gid: int | None = None) -> str:
    """The shell command that runs `cmd` as an unprivileged numeric uid (default nobody, EXT_INTERFACES §5; O's
    follow-up: one uid per co-located native arm, no user is created); timeout -> TERM, then KILL after kill_after."""
    uid = int(uid)
    gid = uid if gid is None else int(gid)
    if uid <= 0 or gid <= 0:
        raise ValueError('refusing to run an untrusted command as root')
    env = nobody_env(home=home, path_dirs=path_dirs, gpus=gpus, token=token, extra=extra, uid=uid)
    inner = f'export PATH={q(env["PATH"])}; ' + (f'cd {q(cwd)} && ' if cwd else '') + cmd
    run = (f'setpriv --reuid={uid} --regid={gid} --clear-groups --no-new-privs env -i '
           + ' '.join(f'{k}={q(v)}' for k, v in env.items()) + f' bash {"-lc" if login else "-c"} {q(inner)}')
    if timeout:
        run = f'timeout --kill-after={int(kill_after)} {int(max(1, timeout))} {run}'
    return run


# ---------------------------------------------------------------- node operations
class NodeError(RuntimeError):
    pass


class NodeOps:
    """Driver-side handle of one node; every method maps onto a node_ext function."""

    def __init__(self, node: str):
        self.node = node

    def call(self, func: str, _timeout: float = 300, **kw):
        raise NotImplementedError

    # thin wrappers -------------------------------------------------
    def probe(self) -> dict:
        return self.call('gpu_probe')

    def mkdirs(self, paths, mode=None, uid=None, gid=None):
        return self.call('mkdirs', paths=list(paths), mode=mode, uid=uid, gid=gid)

    def mkdir_domain(self, paths, mode, uid=None, gid=None):
        """Create directories of one trust domain; an existing one with another owner or group is an error."""
        return self.call('mkdir_domain', paths=list(paths), mode=mode, uid=uid, gid=gid)

    def write_files(self, files: dict, mode=None, uid=None, gid=None):
        enc = {p: base64.b64encode(t.encode('utf-8') if isinstance(t, str) else t).decode('ascii') for p, t in files.items()}
        return self.call('write_files', files=enc, mode=mode, uid=uid, gid=gid)

    def read_files(self, paths, max_bytes=None) -> dict:
        raw = self.call('read_files', paths=list(paths), max_bytes=max_bytes)
        return {p: (None if v is None else base64.b64decode(v)) for p, v in raw.items()}

    def read_text(self, path, max_bytes=None) -> str | None:
        b = self.read_files([path], max_bytes=max_bytes)[path]
        return None if b is None else b.decode('utf-8', 'replace')

    def read_json(self, path, default=None):
        return self.call('read_json', path=path, default=default)

    def hash_files(self, paths) -> dict:
        return self.call('hash_files', paths=list(paths))

    def symlinks(self, links: dict):
        return self.call('symlinks', links=dict(links))

    # inside trees an untrusted user owns: never follow a planted symlink
    def read_user_file(self, path, root, max_bytes=None) -> bytes | None:
        r = self.call('read_user_file', path=path, root=root, max_bytes=max_bytes)
        return base64.b64decode(r['b64']) if r.get('ok') else None

    def read_user_text(self, path, root, max_bytes=None) -> str | None:
        b = self.read_user_file(path, root, max_bytes)
        return None if b is None else b.decode('utf-8', 'replace')

    def write_user_file(self, path, root, data, uid=None, gid=None, mode=0o644):
        b64 = base64.b64encode(data.encode('utf-8') if isinstance(data, str) else data).decode('ascii')
        return self.call('write_user_file', path=path, root=root, b64=b64, uid=uid, gid=gid, mode=mode)

    def regular_file_info(self, paths) -> dict:
        return self.call('regular_file_info', paths=list(paths))

    def sizes(self, paths) -> dict:
        return self.call('sizes', paths=list(paths))

    def acquire(self, path, lease) -> dict:
        return self.call('acquire', path=path, lease=lease)

    def release(self, path, attempt_id) -> bool:
        return self.call('release', path=path, attempt_id=attempt_id)

    def launch(self, script, log, env: dict | None = None, token: str | None = None) -> dict:
        """Start a node script detached. token: INFRARSI_TOKEN of the script's own exec environment, so that
        kill_token(token) also ends the script itself, not only the children that set it later."""
        if token:
            env = {**(env or {}), node_ext.TOKEN_VAR: token}
        return self.call('launch', script=script, log=log, env=env)

    def kill_token(self, token) -> list:
        """Audit-label cleanup (legacy, uid 65534 users); containment of a trust domain is uid_kill."""
        return self.call('kill_token', token=token)

    def uid_kill(self, uid, rounds: int = 20) -> dict:
        return self.call('uid_kill', uid=int(uid), rounds=rounds)

    def uid_status(self, uid) -> dict:
        return self.call('uid_status', uid=int(uid))

    def remove_tree(self, path, root=EXT_RUNS):
        return self.call('remove_tree', path=path, root=root)

    def sh(self, cmd, timeout=120) -> dict:
        return self.call('sh', cmd=cmd, timeout=timeout, _timeout=timeout + 60)


class WinhNodeOps(NodeOps):
    """node_ext over harness.remote (WinhRemote by default). Calls are idempotent, so transport retries are safe. A
    request whose command would exceed ARGV_BUDGET (file contents: candidate code, run logs) is uploaded to a root-only
    request file first (remote.put_bytes) and named on the command line as @<path>; it is removed after the call."""

    def __init__(self, node: str, remote=None, request_dir: str = REQUEST_DIR):
        super().__init__(node)
        if remote is None:
            from ...harness.remote import default_remote
            remote = default_remote()
        self.remote = remote
        self.request_dir, self._request_dir_ready = request_dir.rstrip('/'), False

    def call(self, func, _timeout: float = 300, **kw):
        body = json.dumps({'func': func, 'kw': kw}).encode('utf-8')
        args = base64.b64encode(body).decode('ascii')
        src = base64.b64encode(_NODE_SRC.encode('utf-8')).decode('ascii')
        req_path = None
        if len(src) + len(args) + 64 > ARGV_BUDGET:
            req_path = f'{self.request_dir}/{uuid.uuid4().hex}.json'
            if not self._request_dir_ready:
                self.remote.sh(self.node, f'mkdir -p {q(self.request_dir)} && chmod 700 {q(self.request_dir)}', timeout=60)
                self._request_dir_ready = True
            self.remote.put_bytes(self.node, body, req_path)
            args = q('@' + req_path)
        try:
            res = self.remote.sh(self.node, f'echo {src} | base64 -d | python3 - {args}', timeout=_timeout)
        finally:
            if req_path:
                try:
                    self.remote.sh(self.node, f'rm -f {q(req_path)}', timeout=60)
                except Exception:                       # noqa: BLE001 - a leftover request file is harmless
                    pass
        out = res.get('stdout') or ''
        if node_ext.SENTINEL not in out:
            raise NodeError(f'{self.node} {func}: no result (code {res.get("code")}): {(res.get("stderr") or "")[-1500:]}')
        payload = json.loads(out.rsplit(node_ext.SENTINEL, 1)[1].strip().splitlines()[0])
        if not payload['ok']:
            raise NodeError(f'{self.node} {func}: {payload["error"]}\n{payload.get("traceback", "")[-1500:]}')
        return payload['value']


class LocalNodeOps(NodeOps):
    """In-process stand-in: '/data/...' maps to <root>/data/...; probe and launch are programmable.

    probe_fn() -> gpu_probe dict; launcher(ops, script_path_local, log_path_local) simulates a launched script."""

    def __init__(self, root, node: str = 'WINH0', *, probe_fn=None, launcher=None):
        super().__init__(node)
        self.root = Path(root)
        self.probe_fn = probe_fn or (lambda: {'hostname': 'fake', 'utc': utc_now(), 'error': None,
                                              'gpus': [{'index': g, 'uuid': f'GPU-{g}', 'mem_used_mib': 0} for g in range(8)],
                                              'apps': []})
        self.launcher = launcher
        self.calls: list[tuple[str, dict]] = []
        self.killed_tokens: list[str] = []
        self.links: dict[str, str] = {}
        self.uid_kills: list[int] = []
        self.owners: dict[str, tuple] = {}             # mkdir_domain ownership as the node would have it
        self.uid_survivors: dict[int, list] = {}
        self.uid_passwd: dict[int, str] = {}
        self.uid_procs: dict[int, list] = {}

    def local(self, path: str) -> Path:
        if not str(path).startswith('/'):
            raise ValueError(path)
        return self.root / str(path).lstrip('/')

    def _map(self, v):
        if isinstance(v, str) and v.startswith('/data/'):
            return str(self.local(v))
        if isinstance(v, list):
            return [self._map(x) for x in v]
        if isinstance(v, dict):
            return {self._map(k): x for k, x in v.items()}
        return v

    def _local_user_file(self, func, kw):
        """Windows stand-ins of the dir_fd/O_NOFOLLOW helpers (the POSIX versions run on the nodes)."""
        import hashlib
        if func == 'regular_file_info':
            out = {}
            for p in kw['paths']:
                lp = self.local(p)
                out[p] = {'size': lp.stat().st_size, 'sha256': hashlib.sha256(lp.read_bytes()).hexdigest()} if lp.is_file() else None
            return out
        lp = self.local(kw['path'])
        if not str(lp).startswith(str(self.local(kw['root']))):
            raise NodeError(f'{kw["path"]} is not inside {kw["root"]}')
        if func == 'read_user_file':
            if not lp.is_file():
                return {'ok': False, 'reason': 'missing'}
            data = lp.read_bytes()
            if kw.get('max_bytes') is not None and len(data) > kw['max_bytes']:
                return {'ok': False, 'reason': 'too large'}
            return {'ok': True, 'b64': base64.b64encode(data).decode('ascii'), 'size': len(data)}
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_bytes(base64.b64decode(kw['b64']))
        return {'ok': True, 'path': kw['path']}

    def _unmap(self, v):
        base = str(self.root)
        if isinstance(v, str) and v.startswith(base):
            return '/' + Path(v).relative_to(self.root).as_posix()
        if isinstance(v, dict):
            return {self._unmap(k): x for k, x in v.items()}
        if isinstance(v, list):
            return [self._unmap(x) for x in v]
        return v

    def call(self, func, _timeout: float = 300, **kw):
        self.calls.append((func, kw))
        if func == 'gpu_probe':
            return self.probe_fn()
        if func == 'launch':
            if self.launcher is None:
                raise NodeError('LocalNodeOps has no launcher')
            return self.launcher(self, self._map(kw['script']), self._map(kw['log']))
        if func == 'kill_token':
            self.killed_tokens.append(kw['token'])
            return []
        if func == 'uid_kill':                                  # programmable: survivors per uid (default none)
            node_ext._check_domain_uid(kw['uid'])
            self.uid_kills.append(kw['uid'])
            return {'uid': kw['uid'], 'killed': [], 'remaining': list(self.uid_survivors.get(kw['uid'], [])), 'rounds': 1}
        if func == 'uid_status':
            return {'uid': kw['uid'], 'passwd': self.uid_passwd.get(kw['uid']), 'pids': list(self.uid_procs.get(kw['uid'], []))}
        if func == 'symlinks':                                  # recorded, not created (Windows needs privileges)
            self.links.update(kw['links'])
            return {'linked': sorted(kw['links'])}
        if os.name == 'nt' and func in ('read_user_file', 'write_user_file', 'regular_file_info'):
            return self._local_user_file(func, kw)
        if func == 'sh':
            return {'code': 0, 'stdout': '', 'stderr': ''}
        if func == 'mkdir_domain':                              # chown is not available here: ownership is recorded
            want = (kw.get('uid') or 0, kw.get('gid') or 0)
            made, kept = [], []
            for p in kw['paths']:
                lp = self.local(p)
                if lp.is_symlink() or (lp.exists() and not lp.is_dir()):
                    raise NodeError(f'{self.node} mkdir_domain: {p} is not a plain directory')
                if lp.exists():
                    have = self.owners.get(p, (0, 0))
                    if have != want:
                        raise NodeError(f'{self.node} mkdir_domain: refusing to re-own {p} ({have[0]}:{have[1]}, wanted '
                                        f'{want[0]}:{want[1]}): a directory of another trust domain')
                    kept.append(p)
                else:
                    lp.mkdir(parents=True)
                    self.owners[p] = want
                    made.append(p)
            return {'made': made, 'kept': kept}
        mapped = {k: self._map(v) for k, v in kw.items()}
        if func in ('mkdirs', 'write_files'):
            mapped['uid'] = mapped['gid'] = None                # no chown on the test machine
            if os.name == 'nt':
                mapped['mode'] = None
        try:
            res = getattr(node_ext, func)(**mapped)
        except Exception as e:                                  # noqa: BLE001
            raise NodeError(f'{self.node} {func}: {e!r}') from e
        return json.loads(json.dumps(self._unmap(res)))


# ---------------------------------------------------------------- lease failures (Codex W9 M-02, round 2)
class LeaseError(RuntimeError):
    """A lease operation left claims behind - an acquire whose rollback was incomplete, or a release that failed. Fail
    closed: `kept` are the paths still held, `failures` the recorded release failures."""

    def __init__(self, message: str, *, kept=(), failures=()):
        super().__init__(message)
        self.kept, self.failures = list(kept), list(failures)


# ---------------------------------------------------------------- uid trust domains (Codex W9 B-02, B-03)
class ContainmentError(RuntimeError):
    """Processes of a trust-domain uid survived uid_kill: the GPUs must stay leased."""


def uid_kill(ops: NodeOps, uid: int) -> dict:
    """Kill every process of trust-domain `uid` on the node and verify none remain (retries inside node_ext); raises
    ContainmentError otherwise. Refuses 65534 and uids below node_ext.MIN_UID."""
    node_ext._check_domain_uid(uid)
    res = ops.uid_kill(uid)
    if res.get('remaining'):
        raise ContainmentError(f'{ops.node}: processes of uid {uid} survived: {res["remaining"][:10]}')
    return res


def assert_uid_unused(ops: NodeOps, uid: int) -> dict:
    """A trust-domain uid must have no passwd entry and no process on the node before an arm starts."""
    node_ext._check_domain_uid(uid)
    st = ops.uid_status(uid)
    if st.get('passwd') or st.get('pids'):
        raise ContainmentError(f'{ops.node}: uid {uid} is in use (passwd {st.get("passwd")!r}, pids {st.get("pids", [])[:10]})')
    return st


# ---------------------------------------------------------------- leases and occupancy
def node_lease_path(node: str) -> str:
    return f'{LEASE_DIR}/{node}.json'


def gpu_claim_path(node: str, gpu: int) -> str:
    return f'{GPU_CLAIM_DIR}/{node}.gpu{int(gpu)}.json'


def ext_lease(run_id: str, attempt_id: str, node: str, env: str, gpus, scope: str = 'attempt') -> dict:
    """harness.leases.make_lease with owner infrarsi-ext and purpose ext:<env> (EXT_INTERFACES §2)."""
    lease = make_lease(run_id, attempt_id, [node], purpose=f'ext:{env}', scope=scope)
    lease.update(owner=OWNER, gpus=[int(g) for g in gpus], driver_host=socket.gethostname())
    return lease


def is_ours(app: dict, runs_root: str = EXT_RUNS) -> bool:
    root = runs_root.rstrip('/') + '/'
    return any(isinstance(x, str) and root in (x + '/') for x in (app.get('cwd'), app.get('cmdline')))


def occupancy_problems(probe: dict, gpus, *, lease: dict | None, lease_ok, runs_root: str = EXT_RUNS,
                       own_gpus_only: bool = False, sibling_uids=()) -> list[str]:
    """Reasons a launch on `gpus` must not start (empty list = free). own_gpus_only (co-located arms under a group
    lease): a compute process on another GPU never blocks; sibling_uids (the co-located arms' trust domains): their
    processes on other GPUs are never foreign. Any process on the attempt's own GPUs always blocks."""
    why = []
    if probe.get('error'):
        why.append(f'probe error {probe["error"]}')
    apps = probe.get('apps') or []
    targets, siblings = {int(g) for g in gpus}, {int(u) for u in sibling_uids}

    def elsewhere_ok(a) -> bool:
        gi = a.get('gpu_index')
        return gi is not None and gi not in targets and (own_gpus_only or a.get('uid') in siblings)
    foreign = [a for a in apps if not is_ours(a, runs_root) and not elsewhere_ok(a)]
    if foreign:
        why.append(f'{len(foreign)} foreign compute processes {[a.get("pid") for a in foreign][:8]}')
    on_target = [a for a in apps if a.get('gpu_index') in set(int(g) for g in gpus)]
    if on_target:
        why.append(f'compute processes on target GPUs: {[(a.get("pid"), a.get("gpu_index")) for a in on_target][:8]}')
    busy = [g['index'] for g in probe.get('gpus') or [] if g.get('index') in set(int(x) for x in gpus)
            and (g.get('mem_used_mib') or 0) > IDLE_MEM_MIB and not on_target]
    if busy:
        why.append(f'GPU memory in use without a visible process on {busy}')
    known = {g.get('index') for g in probe.get('gpus') or []}
    missing = [int(g) for g in gpus if int(g) not in known]
    if missing and not probe.get('error'):
        why.append(f'target GPUs {missing} not present on the node')
    if lease and not lease_ok(lease):
        why.append(f'leased by {lease.get("owner")}:{lease.get("run_id")}/{lease.get("attempt_id")}')
    return why


class GpuLease:
    """Holds a node's GPUs. mode 'own': exclusive node lease plus per-GPU claims, released by release(). mode 'held':
    a run-scoped node lease (owner infrarsi-ext, run_id = held_run_id) already exists - the orchestrator's for
    co-located arms, or the arm's own - and only the per-GPU claims under leases/ext/ are taken; a claim that the same
    run already holds (an arm-scoped claim) counts as held and is left to its owner."""

    def __init__(self, ops: NodeOps, *, env: str, gpus, mode: str = 'own', held_run_id: str | None = None,
                 runs_root: str = EXT_RUNS, sibling_uids=()):
        if mode not in ('own', 'held'):
            raise ValueError(f'lease mode {mode!r}')
        if mode == 'held' and not held_run_id:
            raise ValueError("lease mode 'held' needs held_run_id")
        self.ops, self.env, self.gpus, self.mode = ops, env, [int(g) for g in gpus], mode
        self.held_run_id, self.runs_root = held_run_id, runs_root
        self.sibling_uids = tuple(int(u) for u in sibling_uids)    # co-located arms' uids (never foreign elsewhere)
        self.attempt_id: str | None = None
        self.acquired: list[str] = []
        self.ambiguous: set[str] = set()       # claims whose acquire answer was lost: maybe ours, checked on release
        self.release_failures: list[dict] = []
        self.rollback: dict | None = None      # the state a failed acquire left (M-02 round 2)
        self.taken: list[str] = []             # claims this acquire created (also those rolled back again)
        self.sleep = None                      # release backoff sleep (None: time.sleep); tests set a stub
        self.report: dict | None = None

    def state(self) -> dict:
        """The final release state for results and events: complete = nothing of this lease is held any more."""
        return {'complete': not self.acquired, 'kept': list(self.acquired), 'failures': list(self.release_failures)}

    def _lease_ok(self, lease: dict) -> bool:
        if self.mode == 'held':
            return lease.get('owner') == OWNER and lease.get('run_id') == self.held_run_id
        return lease.get('attempt_id') == self.attempt_id

    def acquire(self, run_id: str, attempt_id: str, scope: str = 'attempt') -> dict:
        self.attempt_id = attempt_id
        lease_path = node_lease_path(self.ops.node)
        current = self.ops.read_json(lease_path)
        probe = self.ops.probe()
        kw = {'lease': current, 'lease_ok': self._lease_ok, 'runs_root': self.runs_root,
              'own_gpus_only': self.mode == 'held', 'sibling_uids': self.sibling_uids}
        why = occupancy_problems(probe, self.gpus, **kw)
        if self.mode == 'held' and current is None:
            why.append(f'no node lease held for {self.held_run_id}')
        # A refusal caused only by a sibling arm's process on this attempt's GPU is `occupied_sibling`, not a failure.
        sib, apps = set(self.sibling_uids), probe.get('apps') or []
        sib_gpus = {a.get('gpu_index') for a in apps if a.get('uid') in sib and a.get('gpu_index') in set(self.gpus)}
        rest = {**probe, 'apps': [a for a in apps if a.get('uid') not in sib],
                'gpus': [{**g, 'mem_used_mib': 0} if g.get('index') in sib_gpus else g for g in probe.get('gpus') or []]}
        sibling_only = (bool(why) and bool(sib) and self.mode == 'held' and current is not None
                        and not occupancy_problems(rest, self.gpus, **kw))
        self.report = {'probe': probe, 'lease': current, 'problems': why, 'sibling_only': sibling_only}
        if why:
            raise OccupiedError({self.ops.node: '; '.join(why)})
        lease = ext_lease(run_id, attempt_id, self.ops.node, self.env, self.gpus, scope=scope)
        self.taken = []
        pending = None                                         # a claim whose acquire RPC has not answered yet
        try:
            if self.mode == 'own':
                pending = lease_path
                res = self.ops.acquire(lease_path, lease)
                pending = None
                if not res['ok']:
                    raise OccupiedError({self.ops.node: f'lease taken: {res.get("existing")}'})
                self.acquired.append(lease_path)
                self.taken.append(lease_path)
            for g in self.gpus:
                path = gpu_claim_path(self.ops.node, g)
                pending = path
                res = self.ops.acquire(path, lease)
                pending = None
                cur = res.get('existing') if isinstance(res.get('existing'), dict) else {}
                if not res['ok'] and cur.get('owner') == OWNER and cur.get('run_id') == run_id and cur.get('scope') == 'run':
                    continue                                    # the arm holds this GPU for the whole run
                if not res['ok']:
                    raise OccupiedError({self.ops.node: f'GPU {g} claimed: {res.get("existing")}'})
                self.acquired.append(path)
                self.taken.append(path)
        except BaseException as exc:
            if pending is not None:
                # The claim may exist although its answer was lost (X3 #8): released owner-checked like the others; if it
                # is absent or another attempt's, it was never ours and is dropped without a failure record.
                self.ambiguous.add(pending)
                self.acquired.append(pending)
            self.release()
            self.rollback = {'error': f'{type(exc).__name__}: {str(exc)[:500]}', 'taken': list(self.taken),
                             'ambiguous': pending is not None, **self.state()}
            if self.acquired and isinstance(exc, Exception):          # fail closed with the paths still held
                raise LeaseError(f'{self.ops.node}: acquire failed ({type(exc).__name__}: {str(exc)[:300]}) and the '
                                 f'rollback left {self.acquired} held', kept=self.acquired,
                                 failures=self.release_failures) from exc
            raise
        return lease

    def release(self, *, attempts: int = 4, backoff_s: float = 1.0, sleep=None) -> list[str]:
        """Release every held path idempotently (a path already gone counts as released), retrying transport failures
        with backoff. A path that could not be released stays in `acquired` and is recorded in `release_failures`; a
        lease that now belongs to another attempt is never touched (recorded, dropped). Returns the released paths."""
        import time as _time
        sleep = sleep or self.sleep or _time.sleep
        done = []
        for path in list(reversed(self.acquired)):
            err = None
            for i in range(max(1, int(attempts))):
                try:
                    st = self.ops.call('release_status', path=path, attempt_id=self.attempt_id)
                except Exception as exc:                        # noqa: BLE001 - transport failure: retried, then recorded
                    err = f'{type(exc).__name__}: {str(exc)[:300]}'
                    sleep(backoff_s * (2 ** i))
                    continue
                err = None
                if path in self.ambiguous:                      # never confirmed as ours: removed if it was, else dropped
                    self.ambiguous.discard(path)
                    if st.get('state') == 'removed':
                        done.append(path)
                elif st.get('state') in ('removed', 'absent'):
                    done.append(path)
                else:
                    self.release_failures.append({'path': path, 'error': 'held by another attempt', 'holder': st.get('holder'),
                                                  'utc': utc_now()})
                self.acquired.remove(path)
                break
            if err is not None:
                self.release_failures.append({'path': path, 'error': err, 'utc': utc_now()})
        return done
