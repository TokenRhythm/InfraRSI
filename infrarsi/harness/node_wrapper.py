"""Per-node launcher (stdlib only): one torchrun in its own session; STOP / peer failure / deadline; exit_node<i>.json.

Started by the driver with setsid + nohup (remote_side.launch_wrapper). It only ever signals the process group it
created itself, refuses to start when the node has foreign GPU processes, and never restarts anything. The exit record
carries this process's own clock anchors (UTC: wrapper start, child start, first heartbeat seen, child exit, wrapper
end; monotonic wall_s) for the harness's external-clock check against the training process's phases.
Usage: python3 node_wrapper.py <spec.json>
"""
import datetime
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback

ALIVE_EVERY_S = 30


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic(path, obj):
    tmp = '%s.%d.tmp' % (path, os.getpid())
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def read(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return None


def gpu_apps():
    p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, timeout=30)
    if p.returncode:
        raise RuntimeError('nvidia-smi failed: ' + p.stderr[-300:])
    return [x.strip() for x in p.stdout.splitlines() if x.strip()]


def gpu_mem_used(limit_gpus):
    p = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, timeout=30)
    used = {}
    for line in p.stdout.splitlines():
        if line.strip():
            i, m = [x.strip() for x in line.split(',')]
            if limit_gpus is None or int(i) in limit_gpus:
                used[int(i)] = int(m)
    return used


def alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def terminate(child, grace_s=20):
    """SIGTERM then SIGKILL to the child's own process group (pgid == child pid via start_new_session)."""
    for sig, wait in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 20)):
        if not alive(child.pid):
            break
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            break
        try:
            child.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            pass
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def main(spec_path):
    spec = read(spec_path)
    out, i, n = spec['out_dir'], int(spec['node_index']), int(spec['nnodes'])
    exit_path = os.path.join(out, 'exit_node%d.json' % i)
    alive_path = os.path.join(out, 'wrapper_node%d.json' % i)
    stop_path = os.path.join(out, 'STOP.json')
    hb_path = os.path.join(out, 'heartbeat.json')
    started = utc()
    t0 = time.monotonic()
    clock = {'wrapper_start_utc_s': time.time(), 'child_start_utc_s': None, 'first_heartbeat_utc_s': None,
             'child_exit_utc_s': None}
    child, rc, reason, error, killed = None, None, None, None, False
    try:
        if os.path.exists(exit_path):
            raise RuntimeError('exit file already present; refusing to relaunch an attempt')
        if spec.get('require_idle', True):
            apps = gpu_apps()
            if apps:
                raise RuntimeError('GPU occupied before launch: %s' % apps)
            busy = {k: v for k, v in gpu_mem_used(spec.get('gpu_indices')).items() if v > spec.get('idle_mem_mib', 2048)}
            if busy:
                raise RuntimeError('GPU memory in use before launch (foreign container?): %s' % busy)
        env = dict(os.environ)
        env.update({k: str(v) for k, v in spec['env'].items()})
        os.makedirs(os.path.join(out, 'logs'), exist_ok=True)
        with open(os.path.join(out, 'logs', 'node%d.log' % i), 'ab') as log:
            child = subprocess.Popen(spec['argv'], env=env, cwd=spec.get('cwd') or out, stdout=log,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        clock['child_start_utc_s'] = time.time()
        atomic(os.path.join(out, 'pid_node%d.json' % i), {'pid': child.pid, 'pgid': child.pid, 'wrapper_pid': os.getpid(),
               'hostname': socket.gethostname(), 'started_utc': started, 'argv': spec['argv']})
        deadline = t0 + float(spec['deadline_s'])
        stop_seen = peer_seen = None
        last_alive = 0.0
        while child.poll() is None:
            now = time.monotonic()
            if clock['first_heartbeat_utc_s'] is None and os.path.exists(hb_path):
                clock['first_heartbeat_utc_s'] = time.time()
            if now - last_alive >= ALIVE_EVERY_S:
                atomic(alive_path, {'utc': utc(), 'wrapper_pid': os.getpid(), 'child_pid': child.pid, 'node_index': i})
                last_alive = now
            if os.path.exists(stop_path):
                stop_seen = stop_seen or now
                reason = reason or 'stop'
                if now - stop_seen > float(spec.get('stop_grace_s', 300)):
                    terminate(child)
                    killed = True
                    break
            failed = [j for j in range(n) if j != i and (read(os.path.join(out, 'exit_node%d.json' % j)) or {}).get('exit_code') not in (None, 0)]
            if failed:
                peer_seen = peer_seen or now
                reason = reason or 'peer_failure:%s' % failed
                if now - peer_seen > float(spec.get('peer_grace_s', 60)):
                    terminate(child)
                    killed = True
                    break
            if now > deadline:
                reason = 'deadline'
                terminate(child)
                killed = True
                break
            time.sleep(2)
        rc = child.returncode if child.returncode is not None else child.wait(timeout=30)
        clock['child_exit_utc_s'] = time.time()
        reason = reason or ('completed' if rc == 0 else 'child_failed')
    except BaseException as exc:                  # noqa: BLE001 - recorded in the exit file
        error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        reason = reason or 'wrapper_error'
        if child is not None and child.poll() is None:
            terminate(child)
            killed = True
        rc = child.returncode if child is not None and child.returncode is not None else 99
        if child is not None and clock['child_exit_utc_s'] is None:
            clock['child_exit_utc_s'] = time.time()
    finally:
        try:
            apps = gpu_apps()
        except Exception as exc:                  # noqa: BLE001
            apps = ['unavailable: %r' % (exc,)]
        clock['wrapper_end_utc_s'] = time.time()
        atomic(exit_path, dict(clock, **{
            'schema': 'infrarsi-node-exit-v1', 'node_index': i, 'hostname': socket.gethostname(), 'exit_code': rc,
            'reason': reason, 'killed_by_wrapper': killed, 'error': error, 'started_utc': started, 'finished_utc': utc(),
            'wall_s': round(time.monotonic() - t0, 3),
            'release': {'gpu_apps_after': apps, 'process_group_gone': child is None or not alive(child.pid)}}))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
