"""ext/flashnext per-node launcher: one node of a four-node screen. Stdlib only; runs as `python3` on the WINH node.

Copy of the formal-54 launcher /data/low_precision_winh_20261014/launch_lp_node.py (sha256 1cd303cb..., local copy
reports/low_precision_winh_20261014/launch_lp_node.py) with its safety logic kept:
  - refuse to start when any compute process is on the node's GPUs;
  - exclusive host flocks taken before the GPU check: the original's two (this queue's
    /data/low_precision_winh_20261014/locks/hostNN.lock and the historical launcher's <source run dir>/locks/hostNN.lock,
    so that neither of those launchers can start on this host meanwhile) plus the ext lock under /data/infrarsi/ext;
  - the job runs in its own session (process group); only that group is ever signalled: SIGTERM, 20 s, SIGKILL;
  - stop on a peer's non-zero node<i>.exit.json, on STOP.json, or at the wall cap;
  - node<i>.exit.json and node<i>.release.json (idle check) are written in every case.
  Strengthened: the whole job process group (not only its leader) is drained on every path - also after a zero exit
  of the leader - inside the lock scope, so the host locks are released only after the group's last live (non-zombie)
  member is gone: SIGTERM, up to 20 s, SIGKILL, up to 30 s (inside the 60 s stop reserve). A member that survives
  SIGKILL that long cannot be stopped from here; the launcher then unlocks and records group_gone_before_unlock false
  (the runner's sweep finds it and keeps the node leased). The original waited for the leader only and stopped the
  group only on a non-zero exit.
Changes (AMENDMENT_EXT_003 D-F1-1):
  - the node map, leader IP, run root, ext lock directory, cap and stop reserve come from fn_spec.json (written by
    FlashNextRunner next to the bundle); every file this launcher writes is under /data/infrarsi/ext;
  - the two legacy lock files are only locked, never created or written: they must exist (host01..08 do), and they are
    opened write-only without O_CREAT (the original's 'a' mode minus creation; an exclusive flock may need a writable
    descriptor), so the formal trees stay byte- and mtime-identical; a missing one refuses the launch;
  - the cap is one screen-wide deadline, first launcher start (the earliest started_unix among the four
    node<i>.pid.json) + timeout_s - stop_reserve_s, so that first launcher start -> last node exit stays within
    timeout_s including the TERM/KILL sequence (<= 40 s); the original counted timeout_seconds from each node's own
    start. Each node also stops at its own start + timeout_s - stop_reserve_s;
  - cwd = the run's output directory, which marks every process of the screen as ours (ext.common.remote_exec.is_ours);
    start/finish times are also recorded as unix seconds;
  - dropped: the original's optional `required_runtime_receipt` check (launch_lp_node.py:44-48) - no formal-54 job and
    no ext job sets that key, so the branch never ran.

usage: python3 fn_launch_node.py <job.json> <node_rank> <fn_spec.json>
"""
import contextlib
import datetime
import fcntl
import json
import os
import pathlib
import re
import signal
import socket
import subprocess
import sys
import time
import traceback

EXT_ROOT = '/data/infrarsi/ext/'
LEGACY_LOCK_DIRS = ('/data/low_precision_winh_20261014/locks',
                    '/data/zehua_flashnext_replay_20261013/source_tree/qwen-sft/runs/h100_scale_20260923_v1/locks')
POLL_S = 2.0
TERM_WAIT_S, KILL_WAIT_S = 20.0, 30.0   # + poll: inside the runner's 60 s stop reserve


def atomic(path, value):
    temp = path.with_name(path.name + '.%d.tmp' % os.getpid())
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def gpu_processes():
    p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True,
                       capture_output=True, check=True, timeout=20)
    return p.stdout.strip()


def live(pgid):
    """Live (non-zombie) members of process group `pgid` (a zombie holds no memory, GPU context or file)."""
    out = []
    for name in os.listdir('/proc'):
        if name.isdigit():
            try:
                with open('/proc/%s/stat' % name) as f:
                    fields = f.read().rsplit(')', 1)[1].split()
                if int(fields[2]) == pgid and fields[0] not in ('Z', 'X'):
                    out.append(int(name))
            except (OSError, ValueError, IndexError):
                pass
    return out


def gone(child, seconds):
    """Wait up to `seconds` for the job's process group to have no live member (the leader is reaped meanwhile)."""
    end = time.monotonic() + seconds
    while True:
        child.poll()
        if not live(child.pid):
            return True
        if time.monotonic() >= end:
            return False
        time.sleep(0.5)


def drain(child):
    """Stop whatever is left of the job's process group: SIGTERM, up to 20 s, SIGKILL, up to 30 s. Called inside the
    lock scope; True when no live member is left."""
    for sig, wait in ((signal.SIGTERM, TERM_WAIT_S), (signal.SIGKILL, KILL_WAIT_S)):
        if gone(child, 0):
            return True
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            return gone(child, 0)
        if gone(child, wait):
            return True
    return False


def legacy_lock(stack, directory, host):
    """Exclusive flock on an existing legacy host lock file (never created, never written)."""
    path = os.path.join(directory, 'host%02d.lock' % host)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CLOEXEC)       # FileNotFoundError refuses the launch
    stack.callback(os.close, fd)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return path


def first_start(out, own):
    """The screen's first launcher start: the earliest started_unix among the node<i>.pid.json present (own included)."""
    starts = [own]
    for peer in range(4):
        try:
            value = json.loads((out / ('node%d.pid.json' % peer)).read_text()).get('started_unix')
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(value, (int, float)) and value > 0:
            starts.append(float(value))
    return min(starts)


def main():
    job_path = pathlib.Path(sys.argv[1]).resolve(strict=True)
    node = int(sys.argv[2])
    spec = json.loads(pathlib.Path(sys.argv[3]).read_text())
    job = json.loads(job_path.read_text())
    name = job['name']
    if len(spec['nodes']) != 4 or node not in range(4):
        raise ValueError('Expected four nodes and node rank 0..3')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,220}', name):
        raise ValueError('Invalid run name')
    if socket.gethostname() != spec['hostnames'][node]:
        raise RuntimeError('Wrong physical node')
    out = pathlib.Path(spec['out_root']) / name
    if str(out) != job['output_root'] or not str(out).startswith(EXT_ROOT):
        raise ValueError('output root outside the ext run root')
    cap = float(spec['timeout_s'])
    reserve = float(spec.get('stop_reserve_s', 60.0))
    if not 0 < reserve < cap:
        raise ValueError('need 0 < stop_reserve_s < timeout_s')
    host = int(spec['host_numbers'][node])
    out.mkdir(parents=True, exist_ok=True)
    child = None
    rc = 99
    error = None
    started_unix = None
    locks = []
    group_gone = None
    drain_error = None
    try:
        with contextlib.ExitStack() as stack:
            lockpath = pathlib.Path(spec['locks']) / ('host%02d.lock' % host)
            if not str(lockpath).startswith(EXT_ROOT):
                raise ValueError('lock outside the ext root')
            lockpath.parent.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context(lockpath.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(str(lockpath))
            for directory in LEGACY_LOCK_DIRS:
                locks.append(legacy_lock(stack, directory, host))
            try:
                with (out / ('node%d.log' % node)).open('x') as log:
                    if gpu_processes():
                        raise RuntimeError('GPU is occupied')
                    replacements = {'{node_rank}': str(node), '{master_addr}': spec['master_addr'],
                                    '{master_port}': str(job['master_port']), '{output_dir}': str(out),
                                    '{host}': str(host)}

                    def expand(value):
                        value = str(value)
                        for token, replacement in replacements.items():
                            value = value.replace(token, replacement)
                        return value
                    command = [expand(v) for v in job['command']]
                    env = dict(os.environ, **{str(k): expand(v) for k, v in job.get('env', {}).items()})
                    env.update(SCALE_PROBE_OUT=str(out), SCALE_HOST_NUMBER=str(host),
                               TORCH_NCCL_ASYNC_ERROR_HANDLING='1', LP_RUN_NAME=name, LP_GROUP=job['group'])
                    child = subprocess.Popen(command, env=env, cwd=str(out), stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                    started_unix = time.time()
                    own_deadline = time.monotonic() + cap - reserve
                    atomic(out / ('node%d.pid.json' % node), {'pid': child.pid, 'hostname': socket.gethostname(),
                                                              'node_rank': node, 'node': spec['nodes'][node],
                                                              'host': host, 'command': command,
                                                              'started_utc': utc(), 'started_unix': started_unix,
                                                              'timeout_s': cap, 'stop_reserve_s': reserve,
                                                              'locks': locks})
                    while child.poll() is None:
                        failed = []
                        for peer in range(4):
                            p = out / ('node%d.exit.json' % peer)
                            if p.exists() and json.loads(p.read_text())['exit_code'] != 0:
                                failed.append(peer)
                        if failed or (out / 'STOP.json').exists():
                            error = {'peer_failures': failed, 'explicit_stop': (out / 'STOP.json').exists()}
                            rc = 125
                            break
                        first = first_start(out, started_unix)
                        if time.time() >= first + cap - reserve or time.monotonic() >= own_deadline:
                            error = {'type': 'RunTimeout', 'timeout_seconds': cap, 'stop_reserve_s': reserve,
                                     'first_start_unix': first}
                            rc = 124
                            break
                        time.sleep(POLL_S)
                    else:
                        rc = child.returncode
            finally:
                if child is not None:                  # every path, a zero exit too: the group goes before the locks
                    try:
                        group_gone = drain(child)
                    except BaseException as exc:       # noqa: BLE001 - still inside the lock scope
                        drain_error = repr(exc)
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except OSError:
                            pass
                        group_gone = not live(child.pid)
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    finally:
        atomic(out / ('node%d.exit.json' % node), {'exit_code': rc, 'error': error, 'finished_utc': utc(),
                                                   'finished_unix': time.time(), 'started_unix': started_unix,
                                                   'hostname': socket.gethostname(), 'node': spec['nodes'][node],
                                                   'locks': locks, 'group_gone_before_unlock': group_gone,
                                                   'drain_error': drain_error})
        try:
            processes = gpu_processes()
        except Exception as exc:
            processes = 'unavailable: ' + str(exc)
        atomic(out / ('node%d.release.json' % node), {'idle': not processes, 'processes': processes,
                                                      'process_group_gone': child is None or not live(child.pid),
                                                      'hostname': socket.gethostname(), 'finished_utc': utc()})
    return rc


if __name__ == '__main__':
    sys.exit(main())
