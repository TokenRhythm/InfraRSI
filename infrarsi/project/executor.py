"""Where a project run executes: this machine (optionally under an `srun` prefix for Slurm) or SSH hosts.

Every executor streams the training output line by line and stamps each line on arrival (InfraRSI's clock),
enforces the wall timeout by killing the whole process tree, and can stop a run early (`stop(lines) -> bool`).
No leases or uid containment here: on a user's own machines the run is the user's job (revamp design B).
"""
from __future__ import annotations

import inspect
import io
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from pathlib import Path

from ..contracts import InfrastructureError

# ssh's own messages when the connection, not the remote command, failed (ssh then exits with 255)
SSH_TRANSPORT = re.compile(r'(?i)ssh: connect to host|connection (refused|reset|timed out)|connection closed by|'
                           r'closed by remote host|kex_exchange_identification|banner exchange|could not resolve hostname|'
                           r'permission denied \(publickey|host key verification failed|no route to host|'
                           r'network is unreachable|broken pipe|'
                           r'timeout, server .* not responding')   # a connection that died silently (ServerAlive*)
STAMPED = re.compile(r'^\d+\.\d+\t')            # a line the remote stamper wrote: the job's output, never ssh's
# the remote shell's last line, with the job's exit status: reading ends there (trial C3d: an `ssh -tt` client on
# Windows outlived its finished job by 30 minutes); the job's own lines are stamped, so they never match it
END_MARK = '__infrarsi_exit__'
# a host's GPUs before / after a run (other jobs on them contaminate the measurement: H2b arm B) and its links
SMI_QUERY = ('nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu,persistence_mode,'
             'ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits 2>/dev/null; echo __apps__; '
             'nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null')
# the job's links, read on node 0 in the job's own context (its tree, launch.env and the knobs' env, launch.setup) by
# a short probe right before the reference run - outside the run, its timeout and its accounting
LINK_QUERY = ('echo __uverbs__; ls /dev/infiniband 2>/dev/null | grep -c uverbs; echo __ll__; '
              'cat /sys/class/infiniband/*/ports/*/link_layer 2>/dev/null | sort -u; echo __topo__; '
              "nvidia-smi topo -m 2>/dev/null | head -24; echo __env__; env | grep -E '^NCCL_(IB_DISABLE|NET)='")


def _num(s: str):
    return float(s) if re.fullmatch(r'\d+(\.\d+)?', s.strip()) else None


def parse_smi(text: str | None, devices=None) -> dict | None:
    """One host's SMI_QUERY output, restricted to the job's GPUs (`devices`: a set of indices or of GPU-uuid prefixes;
    None: all) -> {'gpus': [{index, uuid, mem_used_mib, util, persistence, ecc}], 'apps': n, 'apps_mib'} (the processes
    on those GPUs); None when nvidia-smi printed nothing (no NVIDIA GPU, no driver)."""
    head, _, apps = (text or '').partition('__apps__')
    every = []
    for line in head.strip().splitlines():
        f = [x.strip() for x in line.split(',')]
        if len(f) >= 6 and f[0].isdigit():
            every.append({'index': int(f[0]), 'uuid': f[1], 'mem_used_mib': _num(f[2]), 'util': _num(f[3]),
                          'persistence': f[4], 'ecc': _num(f[5])})
    if not every:
        return None
    gpus = [g for g in every if devices is None or g['index'] in devices
            or any(isinstance(d, str) and g['uuid'].startswith(d) for d in devices)]
    mine = {g['uuid'] for g in gpus}
    rows = [[x.strip() for x in r.split(',')] for r in apps.strip().splitlines() if r.strip()]
    rows = [r for r in rows if len(r) >= 3 and r[0] in mine]
    return {'gpus': gpus, 'apps': len(rows), 'apps_mib': sum(_num(r[2]) or 0 for r in rows)}


def gpu_contention(snap: list | None, when: str) -> list[str]:
    """The job's GPUs another process holds or computes on (> 1 GiB used, > 10 % busy, or a compute process) at a
    moment when the job itself is not running (before it starts, after it ended)."""
    out = []
    for i, h in enumerate(snap or []):
        if h is None:
            continue
        busy = [g for g in h['gpus'] if (g['mem_used_mib'] or 0) > 1024 or (g['util'] or 0) > 10]
        if busy or h['apps']:
            gib = sum(g['mem_used_mib'] or 0 for g in busy) / 1024
            out.append(f'node{i}: {len(busy)} GPU(s) in use {when} the run ({gib:.1f} GiB'
                       + (f', {h["apps"]} process(es)' if h['apps'] else '') + ')')
    return out


def ecc_errors(before: list | None, after: list | None) -> list[str]:
    """Uncorrected ECC errors that appeared on the job's GPUs during the run (a GPU going bad, not another job)."""
    out = []
    for i, (b, a) in enumerate(zip(before or [], after or [])):
        for gb, ga in zip((b or {}).get('gpus') or [], (a or {}).get('gpus') or []):
            if gb['ecc'] is not None and ga['ecc'] is not None and ga['ecc'] > gb['ecc']:
                out.append(f'node{i} GPU {ga["index"]}: {int(ga["ecc"] - gb["ecc"])} uncorrected ECC error(s) during the run')
    return out


SETUP_NCCL = re.compile(r'\b(NCCL_IB_DISABLE|NCCL_NET)=(["\']?)([^\s"\';&|)]*)\2')
SETUP_NCCL_ALL = re.compile(r'\b(NCCL_[A-Z0-9_]+)=(["\']?)([^\s"\';&|)]*)\2')


SETUP_DEVICES = re.compile(r'\bCUDA_VISIBLE_DEVICES=(["\']?)([0-9A-Za-z,:-]*)\1(?=[\s;&|)]|$)')


def setup_devices(setup: str | None):
    """The CUDA_VISIBLE_DEVICES launch.setup leaves the job with: {} when it does not name it,
    {'CUDA_VISIBLE_DEVICES': v} when its last assignment is a literal, False when that last one is not (`=$GPUS`) or the
    variable is named some other way (`unset` after it included). A sourced file that selects GPUs is out of the
    scan's reach (as for the link probe); each benchmark rank checks the GPU it was given before NCCL starts."""
    text = setup or ''
    at = [m.start() for m in re.finditer(r'\bCUDA_VISIBLE_DEVICES=', text)]
    if not at or re.search(r'\bunset\b[^;&|]*\bCUDA_VISIBLE_DEVICES\b', text[at[-1]:]):
        return False if 'CUDA_VISIBLE_DEVICES' in text else {}      # named otherwise, or unset after the last one
    m = SETUP_DEVICES.match(text, at[-1])
    return {'CUDA_VISIBLE_DEVICES': m.group(2)} if m else False


def setup_nccl(setup: str | None, *, every: bool = False) -> dict:
    """The NCCL variables launch.setup assigns literally (`export NCCL_IB_DISABLE=1`), the last one winning: the fabric
    ones (NCCL_IB_DISABLE, NCCL_NET), or `every` NCCL_ variable (the link benchmark runs with the job's NCCL
    settings). The link probe and the benchmark do not replay the setup, which may change the staged tree."""
    return {k: v for k, _, v in (SETUP_NCCL_ALL if every else SETUP_NCCL).findall(setup or '')}


def parse_link(text: str | None, *, nodes: int) -> dict:
    """Node 0's LINK_QUERY output (the reference run's probe) -> {'intra': nvlink | pcie, 'inter': ib | roce |
    tcp} (inter only for a multi-node job): NCCL can use RDMA only where the container has /dev/infiniband (unicom's
    hosts have mlx5 NICs but their containers do not: TCP), and NCCL_IB_DISABLE=1 or NCCL_NET=Socket in the job's own
    environment (the query's __env__ part) force sockets."""
    text = (text or '').split('__uverbs__', 1)[-1]  # anything a profile printed before the query is not part of it
    uverbs, _, rest = text.partition('__ll__')
    ll, _, rest = rest.partition('__topo__')
    topo, _, shell_env = rest.partition('__env__')
    nccl = dict(line.strip().split('=', 1) for line in shell_env.splitlines() if '=' in line)
    out = {}
    if re.search(r'\bNV\d+\b', topo):
        out['intra'] = 'nvlink'
    elif re.search(r'\b(PIX|PXB|PHB|NODE|SYS)\b', topo):
        out['intra'] = 'pcie'
    if nodes > 1 and (text or '').strip():
        first = uverbs.strip().splitlines()[:1]
        n = int(first[0]) if first and first[0].strip().isdigit() else 0
        forced = nccl.get('NCCL_IB_DISABLE', '').lower() in ('1', 'true') or nccl.get('NCCL_NET', '').lower() == 'socket'
        out['inter'] = 'tcp' if n == 0 or forced else ('roce' if 'Ethernet' in ll and 'InfiniBand' not in ll else 'ib')
    return out


def transport_failure(code, lines, *, silent_counts: bool = True) -> str | None:
    """ssh's own diagnostic when the connection, not the job, failed: exit 255 and a transport message in a line the
    remote stamper did not stamp (the job's output always passes the stamper), or no output at all (only when
    `silent_counts`: not for a process the executor itself killed); else None."""
    if code != 255:
        return None
    if not lines:
        return 'no output' if silent_counts else None
    hits = [t for _, t in lines if not STAMPED.match(t) and SSH_TRANSPORT.search(t)]
    return hits[-1][-300:] if hits else None


def _kill_tree(proc: subprocess.Popen, grace_s: float = 10.0) -> None:
    """Kill the run's whole process group (POSIX: its session), also when the launcher already exited and a child
    still holds the GPUs and the output pipe. Windows: taskkill of the tree (best effort)."""
    if os.name == 'nt':
        subprocess.run(['taskkill', '/T', '/F', '/PID', str(proc.pid)], capture_output=True)
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        proc.poll()                                     # reap the leader, or the group never looks empty
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _reap_local(launch: str, grace_s: float = 10.0) -> None:
    """The processes of one local launch that left its session - a launcher that starts its workers in sessions of
    their own (accelerate / torchrun): after D2-R204d's timeout kill, orphans of the job held its output pipe and a GPU
    for 80 min and the run never ended - found by the INFRARSI_LAUNCH tag they all inherit (one per launch: a late
    reap never reaches the next command in the same tree, nor a shell the user opened there): SIGTERM, SIGKILL what is
    left after grace_s, at once when interrupted. Linux (/proc) only."""
    if not os.path.isdir('/proc'):
        return
    tag = f'INFRARSI_LAUNCH={launch}'.encode()

    def left() -> list[int]:
        out = []
        for d in os.listdir('/proc'):
            if not d.isdigit() or int(d) == os.getpid():
                continue
            try:
                with open(f'/proc/{d}/environ', 'rb') as f:
                    if tag in f.read().split(b'\0'):
                        out.append(int(d))
            except OSError:
                continue
        return out

    def send(pids, sig):
        for p in pids:
            try:
                os.kill(p, sig)
            except OSError:
                pass
    pids = left()
    if not pids:
        return
    send(pids, signal.SIGTERM)
    try:
        deadline = time.monotonic() + grace_s
        while pids and time.monotonic() < deadline:
            time.sleep(0.2)
            pids = left()
    except BaseException:                               # Ctrl-C / SIGTERM of InfraRSI in the grace: no grace left
        send(left(), signal.SIGKILL)
        raise
    send(pids, signal.SIGKILL)


def _stop_tree(proc: subprocess.Popen) -> None:
    """_kill_tree on an interrupt; a second Ctrl-C during its grace period kills at once instead of leaving the job."""
    try:
        _kill_tree(proc)
    except BaseException:
        _kill_tree(proc, grace_s=0.0)
        raise


def _popen(argv_or_cmd, *, cwd=None, env=None, shell=False) -> subprocess.Popen:
    kw = dict(cwd=cwd, env=env, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
              text=True, encoding='utf-8', errors='replace', bufsize=1)
    if os.name == 'nt':
        kw['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw['start_new_session'] = True
    return subprocess.Popen(argv_or_cmd, **kw)


def takes_node(fn) -> bool:
    """Whether a stop predicate takes the node index (`stop(lines, node=i)`); a plain `stop(lines)` is asked
    without it."""
    try:
        return 'node' in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def stream(proc: subprocess.Popen, *, timeout_s: float, log_path: Path, stop=None, end: str | None = None,
           end_grace_s: float = 10.0, lines: list | None = None, poll_s: float = 2.0, reap=None) -> dict:
    """Read the process output to the end: stamped lines, timeout and early stop kill the process tree. `end`: the
    marker line a remote shell prints last with the job's exit status; reading stops there, the exit status is the
    marker's, and a client still running `end_grace_s` later is killed. `stop(lines)` is asked on every line and
    every `poll_s` (a job whose updates reach a probe file or another node's output); `stop_index`: the lines read
    when it fired (what follows is the shutdown's). `lines`: the caller's list to collect into. `reap()`: after each
    kill, the job's processes outside its session (they may hold the output pipe: the read would never end)."""
    lines = [] if lines is None else lines
    state = {'timed_out': False, 'stopped': False, 'stop_index': None, 'stop_t': None, 'end_t': None}
    ended = None
    t0 = time.perf_counter()
    lock, done = threading.Lock(), threading.Event()

    def kill():
        _kill_tree(proc)
        if reap is not None:
            reap()

    def on_timeout():
        state['timed_out'] = True
        kill()

    def check(snapshot):
        with lock:
            if not state['stopped'] and stop(snapshot):
                # the cutoff is the live buffer's now: a line read while the predicate ran came before the stop
                state['stopped'], state['stop_index'], state['stop_t'] = True, len(lines), time.perf_counter()
                kill()

    def poll():
        while not done.wait(poll_s) and not state['stopped']:
            try:
                check(list(lines))
            except Exception:                           # noqa: BLE001 - the per-line check reports it
                pass

    timer = threading.Timer(timeout_s, on_timeout)
    timer.daemon = True
    timer.start()
    if stop is not None and poll_s:
        threading.Thread(target=poll, daemon=True).start()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, 'w', encoding='utf-8') as log:
            for raw in proc.stdout:
                t = time.perf_counter()
                text = raw.rstrip('\r\n')
                if end and text.startswith(end):
                    ended, state['end_t'] = text[len(end):].strip(), t
                    timer.cancel()                       # the job finished in time; only the client lingers
                    break
                lines.append((t, text))
                log.write(text + '\n')
                if stop is not None and not state['stopped']:
                    check(lines)
        if ended is not None:
            try:
                proc.wait(timeout=end_grace_s)
            except subprocess.TimeoutExpired:
                _kill_tree(proc)
        proc.wait()
    except BaseException:                               # Ctrl-C / SIGTERM of InfraRSI: the job runs in its own session,
        _stop_tree(proc)                                # so the terminal's signal never reaches it (D1-18: left running)
        if reap is not None:
            reap()
        raise
    finally:
        timer.cancel()
        done.set()
    code = proc.returncode
    if ended is not None and ended.lstrip('-').isdigit():
        code = int(ended)
    return {'exit_code': code, 'timed_out': state['timed_out'], 'stopped': state['stopped'],
            'stop_index': state['stop_index'], 'stop_t': state['stop_t'], 'end_t': state['end_t'],
            'wall_s': time.perf_counter() - t0, 'lines': lines}


class LocalExecutor:
    kind = 'local'

    def __init__(self, prefix: str = ''):
        self.prefix = prefix.strip()          # e.g. 'srun --nodes 1 --gpus-per-node 8' (execution.where: slurm)

    def describe(self) -> dict:
        return {'kind': self.kind, 'prefix': self.prefix or None}

    @staticmethod
    def quote(s: str) -> str:
        return subprocess.list2cmdline([s]) if os.name == 'nt' else shlex.quote(s)

    def python(self, configured: str | None) -> str:
        return configured or sys.executable

    def place(self, tree: Path, rel: str) -> str:
        return str(tree)

    def run(self, cmd: str, *, cwd: str, env: dict, timeout_s: float, log_path: Path, stop=None) -> dict:
        full = f'{self.prefix} bash -c {shlex.quote(cmd)}' if self.prefix else cmd   # the whole line in the allocation
        launch = uuid.uuid4().hex                      # the tag its processes inherit, also those in other sessions
        proc = _popen(full, cwd=cwd, env={**os.environ, **env, 'PYTHONUNBUFFERED': '1', 'INFRARSI_LAUNCH': launch},
                      shell=True)
        try:
            return stream(proc, timeout_s=timeout_s, log_path=log_path, stop=stop, reap=lambda: _reap_local(launch))
        finally:
            _reap_local(launch)                        # what outlived the run without its output pipe (a GPU held)

    def read(self, path: str) -> str | None:
        p = Path(path)
        return p.read_text(encoding='utf-8', errors='replace') if p.is_file() else None

    def on_hosts(self, cmd: str, timeout_s: float = 20, login: bool = False) -> list[str | None]:
        """A short shell command on every host of the job (here: this machine, in this process's environment, as the
        job inherits it); [] under srun, whose GPUs are the allocation's, and on Windows (no POSIX shell)."""
        if self.prefix or os.name == 'nt':
            return []
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, errors='replace', timeout=timeout_s)
            return [r.stdout]
        except (OSError, subprocess.SubprocessError):
            return [None]

    def probe(self, cmd: str, *, cwd: str, env: dict, timeout_s: float = 30) -> str | None:
        """A short command where and as the job runs (its directory, its environment), apart from the job: its output
        whatever its exit status; None under srun / on Windows or when it could not run."""
        if self.prefix or os.name == 'nt':
            return None
        try:
            return subprocess.run(cmd, shell=True, cwd=cwd, env={**os.environ, **env}, capture_output=True, text=True,
                                  errors='replace', timeout=timeout_s).stdout
        except (OSError, subprocess.SubprocessError):
            return None


STAMPER = '_infrarsi_stamp.py'
PLACED = '.infrarsi_placed'   # the copy's token: a host that reads it already sees the tree (shared storage)
WORKER_GRACE_S = 60.0           # after node 0's job failed: the other nodes' time to end before they are stopped
# universal newlines: a job that redraws its progress with \r (tqdm, autoresearch) gives one line per update, as the
# local executor reads it (POSIX sys.stdin splits at \n only: E-C-IS2 had all 30 updates stamped at once)
STAMPER_SRC = '\n'.join(['import io, sys, time',
                         "for line in io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace', newline=None):",
                         '    sys.stdout.write("%.6f\\t%s" % (time.time(), line))', '    sys.stdout.flush()', '']).encode()


def remote_stamped(lines: list) -> list:
    """Lines prefixed by the remote stamper (`<epoch seconds>TAB<text>`) take the remote time: network and ssh
    buffering deliver them in bursts, but each was stamped when the training process wrote it."""
    out = []
    for t, text in lines:
        head, sep, rest = text.partition('\t')
        try:
            out.append((float(head), rest) if sep else (t, text))
        except ValueError:
            out.append((t, text))
    return out


class SSHExecutor:
    """execution: {where: ssh, hosts: ['user@host:port', ...], remote_dir: /scratch/infrarsi}. The candidate tree is
    copied to every host (tar over ssh); node 0's output is the parsed log, the others go to node<i>.log. Commands may
    use {node_rank}, {nnodes}, {master_addr}. `ssh -tt` ties the remote job to the connection, so killing the local
    ssh (timeout, early stop) hangs up and ends the remote process group. Each output line is stamped on the remote
    host by a tiny stamper process (outside the training code), so step times do not suffer from ssh buffering."""
    kind = 'ssh'

    def __init__(self, hosts, remote_dir: str, ssh: str = 'ssh', options=('-o', 'BatchMode=yes'),
                 master_addr: str | None = None):
        self.hosts = [self._parse(h) for h in hosts]
        self.remote_dir = remote_dir.rstrip('/')
        self.ssh, self.options = ssh, tuple(options)
        self.master_addr = master_addr          # {master_addr}: given, else host 0's ssh name (an alias may not resolve
                                                # on the nodes: then execution.master_addr names node 0 for them)

    @staticmethod
    def _parse(h: str) -> tuple[str, str | None]:
        if h.count(':') == 1 and h.rsplit(':', 1)[1].isdigit():
            target, port = h.rsplit(':', 1)
            return target, port
        return h, None

    def describe(self) -> dict:
        return {'kind': self.kind, 'hosts': [f'{h}:{p}' if p else h for h, p in self.hosts], 'remote_dir': self.remote_dir}

    quote = staticmethod(shlex.quote)

    def python(self, configured: str | None) -> str:
        return configured or 'python3'

    def _argv(self, i: int, tty: bool = False) -> list[str]:
        target, port = self.hosts[i]
        return [self.ssh, *self.options, *(['-p', port] if port else []), *(['-tt'] if tty else []), target]

    def _reap(self, i: int, cwd: str, grace_s: int = 30) -> None:
        """After a killed run (timeout, early stop): wait until no process of the job is left on host i (a process
        whose working directory is in the candidate tree), SIGKILL what remains after `grace_s`. Killing the local
        client only hangs up the remote session: in trial C3d torchrun still held port 29500 and the GPUs 20 s after an
        early abort, and the next two candidates failed at start. The job's processes are also found by the
        INFRARSI_RUN_TREE they inherit, when one has left the tree."""
        script = (f't=$(readlink -f {shlex.quote(cwd)}); for k in $(seq {grace_s}); do left=; '
                  'for p in /proc/[0-9]*; do c=$(readlink $p/cwd 2>/dev/null); '
                  'case "$c" in "$t"|"$t"/*) left="$left ${p#/proc/}"; continue;; esac; '
                  'while IFS= read -r -d "" e; do '                  # a job that left its tree keeps the tag
                  '[ "$e" = "INFRARSI_RUN_TREE=$t" ] && { left="$left ${p#/proc/}"; break; }; '
                  'done 2>/dev/null < "$p/environ"; done; '
                  f'[ -z "$left" ] && exit 0; [ $k = {grace_s} ] && kill -9 $left; sleep 1; done')
        try:
            subprocess.run(self._argv(i) + [f'bash -c {shlex.quote(script)}'], capture_output=True, timeout=grace_s + 60)
        except (OSError, subprocess.SubprocessError):
            pass                                       # a node we cannot reach: the next run reports it

    def root(self) -> str:
        """remote_dir as an absolute path (a leading ~ is expanded once, on host 0)."""
        if self.remote_dir.startswith('~'):
            r = subprocess.run(self._argv(0) + ['printf %s "$HOME"'], capture_output=True, text=True, errors='replace')
            home = r.stdout.strip()
            if r.returncode != 0 or not home.startswith('/'):
                raise InfrastructureError(f'cannot resolve {self.remote_dir} on {self.hosts[0][0]}: '
                                          f'{(r.stdout + r.stderr)[-300:]}')
            self.remote_dir = home + self.remote_dir[1:]
        return self.remote_dir

    def place(self, tree: Path, rel: str, timeout_s: float | None = None) -> str:
        """Copy `tree` to every host (`timeout_s`: each copy is killed after it, a stalled connection included). A
        host that already sees the copy - a remote_dir on storage the hosts share - gets none: a second extraction
        replaces the files under the first host (D2-R201: node 0's launcher 'Stale file handle', exit 126)."""
        remote = f'{self.root()}/{rel}'
        token = uuid.uuid4().hex
        for i in range(len(self.hosts)):
            if i and self._sees(i, f'{remote}/{PLACED}', token):
                continue
            try:
                proc = subprocess.Popen(self._argv(i) + [f'mkdir -p {shlex.quote(remote)} && tar -xzf - -C {shlex.quote(remote)}'],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            except OSError as exc:
                raise InfrastructureError(f'cannot start ssh to {self.hosts[i][0]}: {exc}') from exc
            watchdog = threading.Timer(timeout_s, proc.kill) if timeout_s else None
            if watchdog is not None:
                watchdog.start()
            broken = None
            try:
                with tarfile.open(fileobj=proc.stdin, mode='w|gz') as tar:
                    tar.add(str(tree), arcname='.')
                    info = tarfile.TarInfo(STAMPER)
                    info.size = len(STAMPER_SRC)
                    tar.addfile(info, io.BytesIO(STAMPER_SRC))
                    info = tarfile.TarInfo(PLACED)
                    info.size = len(token)
                    tar.addfile(info, io.BytesIO(token.encode()))
                proc.stdin.close()
            except OSError as exc:                     # ssh died under the copy: a broken pipe (EINVAL on Windows)
                broken = exc
            out = proc.stdout.read().decode('utf-8', 'replace')
            code = proc.wait()
            if watchdog is not None:
                watchdog.cancel()
            if code != 0 or broken is not None:
                raise InfrastructureError(f'copy to {self.hosts[i][0]} failed: {(out.strip() or repr(broken))[-500:]}')
        return remote

    def _sees(self, i: int, path: str, token: str) -> bool:
        """Host i reads this copy's token at `path` (False when it does not answer: its own copy then fails loudly)."""
        try:
            r = subprocess.run(self._argv(i) + [f'cat {shlex.quote(path)} 2>/dev/null'], capture_output=True, text=True,
                               errors='replace', timeout=30)
        except (OSError, subprocess.SubprocessError):
            return False
        return r.stdout.strip() == token

    def run(self, cmd: str, *, cwd: str, env: dict, timeout_s: float, log_path: Path, stop=None, python: str = 'python3') -> dict:
        exports = ' '.join(f'{k}={shlex.quote(str(v))}' for k, v in {**env, 'PYTHONUNBUFFERED': '1'}.items())
        master = self.master_addr or self.hosts[0][0].split('@')[-1]
        procs, side = [], []
        try:
            return self._run(cmd, cwd, exports, master, python, timeout_s, log_path, stop, procs, side)
        except Exception:                              # node 0's stream failed (a full disk): the other clients too
            try:
                for p in procs:
                    _kill_tree(p)
            except (KeyboardInterrupt, SystemExit):    # Ctrl-C during that: the whole cancellation
                self._cancel(procs, cwd)
                raise
            raise
        except BaseException:                          # Ctrl-C / SIGTERM anywhere in the run: what was launched stops
            self._cancel(procs, cwd)
            raise

    def _cancel(self, procs: list, cwd: str) -> None:
        """Stop every client launched, then leave no job on its hosts (the reap, as after a kill). A further Ctrl-C
        makes the remaining steps immediate (no grace); it skips no client and no host."""
        hurry = False
        steps = [lambda h, p=p: _kill_tree(p, grace_s=0.0 if h else 10.0) for p in procs]
        steps += [lambda h, i=i: self._reap(i, cwd, grace_s=1 if h else 30) for i in range(len(procs))]
        for fn in steps:
            while True:
                try:
                    fn(hurry)
                    break
                except (KeyboardInterrupt, SystemExit):
                    hurry = True

    def _run(self, cmd, cwd, exports, master, python, timeout_s, log_path, stop, procs, side) -> dict:
        for i in range(len(self.hosts)):
            c = cmd.replace('{node_rank}', str(i)).replace('{nnodes}', str(len(self.hosts))).replace('{master_addr}', master)
            stamp = f' 2>&1 | {python} -u {STAMPER}'     # every node: the job's lines are stamped, ssh's own are not
            script = (f'set -o pipefail; cd {shlex.quote(cwd)} && export {exports} && '
                      f'export INFRARSI_RUN_TREE="$(pwd -P)" && {{ {c} ; }}{stamp}; '
                      f'printf "\\n{END_MARK} %s\\n" "$?"')    # on a line of its own after an unterminated last line
            try:
                procs.append(_popen(self._argv(i, tty=True) + [f'bash -lc {shlex.quote(script)}']))
            except OSError as exc:
                for p in procs:
                    _kill_tree(p)
                raise InfrastructureError(f'cannot start ssh to {self.hosts[i][0]}: {exc}') from exc
        side_res = [None] * len(procs)
        buf = [[] for _ in procs]                      # every node's lines so far: the stop reads them all (Megatron
                                                       # logs its updates on the last node)

        def drain(i, p):                               # other nodes: drained to their own logs
            side_res[i] = stream(p, timeout_s=timeout_s, log_path=log_path.with_name(f'node{i}.log'), end=END_MARK,
                                 lines=buf[i])

        for i, p in enumerate(procs[1:], 1):
            th = threading.Thread(target=drain, args=(i, p), daemon=True)
            th.start()
            side.append(th)
        ask = None if stop is None else (
            (lambda lines, i: stop(lines, node=i)) if takes_node(stop) else (lambda lines, i: stop(lines)))
        res = stream(procs[0], timeout_s=timeout_s, log_path=log_path, end=END_MARK, lines=buf[0],
                     stop=(lambda _: any(ask(remote_stamped(list(n)), i) for i, n in enumerate(buf))) if stop else None)   # (a probe file:
                                                                                     # asked with no lines too)
        raw0 = res['lines']
        res['lines'] = remote_stamped(raw0)
        if res['timed_out'] or res['stopped']:
            for p in procs[1:]:
                _kill_tree(p)
        workers_stopped = set()                         # the nodes whose clients this run stopped after node 0 failed
        if not (res['timed_out'] or res['stopped']) and res.get('exit_code') not in (0, None) and side:
            # node 0's job failed: a multi-node job cannot finish without it (D2-R201: node 1 waited 30 min at the
            # rendezvous for a node 0 that had exited at once); the others end on their own or are stopped and reaped
            deadline = time.monotonic() + WORKER_GRACE_S
            for th in side:
                th.join(max(0.0, deadline - time.monotonic()))
            workers_stopped = {i for i, th in enumerate(side, 1) if th.is_alive()}
            for i in sorted(workers_stopped):
                _kill_tree(procs[i])
        for th in side:
            th.join()                                  # each drain ends at its marker, timeout or the client's exit
        if workers_stopped or any(r and (r['timed_out'] or r['stopped']) for r in [res, *side_res[1:]]):
            for i in range(len(self.hosts)):           # any node killed: wait until the job left every node
                self._reap(i, cwd)
        waited = [p.wait() for p in procs]             # each node's exit status: its end marker's, else ssh's
        codes = [res['exit_code']] + [(side_res[i] or {}).get('exit_code', waited[i]) for i in range(1, len(procs))]
        res['exit_code'] = next((c for c in codes if c != 0), 0)
        res['node_exit_codes'] = codes
        res['node_lines'] = [remote_stamped((side_res[i] or {}).get('lines') or []) for i in range(1, len(procs))]
        if res['stopped']:                             # our stop: each worker's lines before it, and the nodes whose
            st = res['stop_t']                         # job had already ended failing (their exit is no stop of ours)
            res['node_stop_index'] = [sum(1 for t, _ in (side_res[i] or {}).get('lines') or [] if t <= st)
                                      for i in range(1, len(procs))]
            ends = [(res.get('end_t'), codes[0])] + [((side_res[i] or {}).get('end_t'), codes[i]) for i in range(1, len(procs))]
            res['failed_before_stop'] = [i for i, (t, c) in enumerate(ends) if t is not None and t <= st and c != 0]
        killed = res['timed_out'] or res['stopped']    # node 0 waiting for an unreachable worker can time out
        broken = [(i, why) for i, code in enumerate(codes) if (why := transport_failure(   # a worker we stopped is no
            code, raw0 if i == 0 else (side_res[i] or {}).get('lines') or [],              # lost connection
            silent_counts=not (killed or i in workers_stopped))) is not None]
        if broken:                                     # any node's connection: the job never had a fair run, and it
            for i in range(len(self.hosts)):           # may run on without its session (trial D1-17: the orphan held
                self._reap(i, cwd, grace_s=5)          # the GPU and the rendezvous port while the next runs started)
            i, why = broken[0]
            raise InfrastructureError(f'ssh to {self.hosts[i][0]} failed: {why}')
        return res

    def read(self, path: str) -> str | None:
        r = subprocess.run(self._argv(0) + [f'cat {shlex.quote(path)}'], capture_output=True, text=True, errors='replace')
        return r.stdout if r.returncode == 0 else None

    def on_hosts(self, cmd: str, timeout_s: float = 30, login: bool = False) -> list[str | None]:
        """A short shell command on every host, concurrently (`login`: in a login shell, as the job runs); a host
        that does not answer gives None."""
        out: list[str | None] = [None] * len(self.hosts)

        def one(i):
            try:
                r = subprocess.run(self._argv(i) + [f'bash {"-lc" if login else "-c"} {shlex.quote(cmd)}'], capture_output=True,
                                   text=True, errors='replace', timeout=timeout_s)
                out[i] = r.stdout if r.returncode in (0, 1) else None
            except (OSError, subprocess.SubprocessError):
                pass
        threads = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(len(self.hosts))]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        return out

    def probe(self, cmd: str, *, cwd: str, env: dict, timeout_s: float = 30) -> str | None:
        """A short command on host 0 where and as the job runs there (login shell, its tree, its exports), apart from
        the job: its output whatever its exit status; None when the host did not answer."""
        exports = ' '.join(f'{k}={shlex.quote(str(v))}' for k, v in env.items())
        script = f'cd {shlex.quote(cwd)} && ' + (f'export {exports} && ' if exports else '') + cmd
        try:
            return subprocess.run(self._argv(0) + [f'bash -lc {shlex.quote(script)}'], capture_output=True, text=True,
                                  errors='replace', timeout=timeout_s).stdout
        except (OSError, subprocess.SubprocessError):
            return None


def make(execution: dict):
    where = execution.get('where', 'local')
    if where == 'ssh':
        return SSHExecutor(execution['hosts'], execution.get('remote_dir') or '~/.infrarsi/runs',
                           master_addr=execution.get('master_addr'))
    if where == 'slurm':
        return LocalExecutor(prefix=f'srun {execution.get("slurm_args") or ""}')
    return LocalExecutor()


def copy_tree(src: Path, dst: Path, excluded) -> None:
    """Copy the project tree (excluded(rel) -> True skips a file or directory)."""
    def ignore(d, names):
        rel = Path(d).relative_to(src).as_posix()
        return {n for n in names if excluded(f'{rel}/{n}' if rel != '.' else n)}
    shutil.copytree(src, dst, ignore=ignore, symlinks=True, dirs_exist_ok=True)


def own_file(tree: Path, rel: str) -> Path:
    """tree/rel as the tree's own file, to be written: a link the copy kept (the file, or a directory on its way, linked
    to a shared config or source) becomes a copy of its target, so the write cannot reach the original."""
    d = tree
    for part in Path(rel).parts[:-1]:
        d = d / part
        if d.is_symlink():
            target = d.resolve()
            d.unlink()
            shutil.copytree(target, d, symlinks=True)
    f = d / Path(rel).name
    if f.is_symlink():
        target = f.resolve()
        f.unlink()
        if target.is_file():
            shutil.copyfile(target, f)
    return f

