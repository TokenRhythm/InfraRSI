"""Functions executed on a WINH node (stdlib only, Python >= 3.8).

The driver ships this file's source with every call (harness.remote.WinhRemote.call), so the code that
touches /data/infrarsi on the nodes is versioned with the driver. LocalRemote runs the same functions
in-process against a local directory (CPU tests). Never kill or signal a process from here.
"""
import datetime
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import tarfile
import time


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except ValueError as e:                       # partially written or corrupt file
        return {'__unreadable__': repr(e)}


def write_json(path, obj, exclusive=False):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    data = json.dumps(obj, indent=2, sort_keys=True)
    if exclusive:
        with open(path, 'x', encoding='utf-8') as f:
            f.write(data)
        return {'path': path, 'created': True}
    tmp = '%s.%d.tmp' % (path, os.getpid())
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(data)
    os.replace(tmp, path)
    return {'path': path, 'created': True}


def write_text(path, text):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = '%s.%d.tmp' % (path, os.getpid())
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text)
    os.replace(tmp, path)
    return {'path': path}


def read_text(path, max_bytes=None):
    try:
        with open(path, 'rb') as f:
            data = f.read() if max_bytes is None else f.read(max_bytes)
        return data.decode('utf-8', errors='replace')
    except FileNotFoundError:
        return None


def makedirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)
    return {'made': list(paths)}


def file_age(path):
    try:
        return time.time() - os.stat(path).st_mtime
    except FileNotFoundError:
        return None


def tail(path, nbytes=4096):
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as f:
            f.seek(max(0, size - nbytes))
            return f.read().decode('utf-8', errors='replace')
    except FileNotFoundError:
        return None


def exists(paths):
    return {p: os.path.exists(p) for p in paths}


def listdir(path):
    try:
        return sorted(os.listdir(path))
    except FileNotFoundError:
        return None


def disk_free(path='/data'):
    s = os.statvfs(path)
    return {'path': path, 'free_bytes': s.f_bavail * s.f_frsize, 'total_bytes': s.f_blocks * s.f_frsize}


def du_bytes(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


# ---------------------------------------------------------------- GPUs, ports, processes

def _smi(args, timeout=30):
    p = subprocess.run(['nvidia-smi'] + args, capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError('nvidia-smi %s failed: %s' % (' '.join(args), p.stderr.strip()[-500:]))
    return [line.strip() for line in p.stdout.splitlines() if line.strip()]


def gpu_status():
    """Compute processes and per-GPU memory/utilisation; processes in other containers show only as memory."""
    out = {'hostname': socket.gethostname(), 'utc': utc(), 'error': None, 'compute_apps': [], 'gpus': []}
    try:
        out['ips'] = subprocess.run(['hostname', '-I'], capture_output=True, text=True, timeout=10).stdout.split()
    except Exception as e:                        # noqa: BLE001 - report, never raise from a probe
        out['ips'] = []
        out['error'] = 'hostname -I: %r' % (e,)
    try:
        for line in _smi(['--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits']):
            pid, name, mem = [x.strip() for x in line.split(',', 2)]
            app = {'pid': int(pid), 'name': name, 'used_mib': int(mem) if mem.isdigit() else None}
            try:
                app['sid'] = os.getsid(int(pid))
                app['pgid'] = os.getpgid(int(pid))
            except (OSError, ValueError):
                app['sid'] = app['pgid'] = None
            try:
                with open('/proc/%d/cmdline' % int(pid), 'rb') as f:
                    app['cmdline'] = f.read(600).replace(bytes(1), b' ').decode('utf-8', 'replace').strip()
            except (OSError, ValueError):
                app['cmdline'] = None
            out['compute_apps'].append(app)
        for line in _smi(['--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits']):
            idx, used, total, util = [x.strip() for x in line.split(',')]
            out['gpus'].append({'index': int(idx), 'mem_used_mib': int(used), 'mem_total_mib': int(total),
                                'util_pct': int(util) if util.isdigit() else None})
    except Exception as e:                        # noqa: BLE001
        out['error'] = repr(e)
    return out


def port_free(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(('0.0.0.0', int(port)))
        return True
    except OSError:
        return False
    finally:
        s.close()


def first_free_port(candidates):
    for p in candidates:
        if port_free(p):
            return p
    return None


def session_alive(pgid):
    try:
        os.killpg(int(pgid), 0)                   # signal 0: existence check only
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


# ---------------------------------------------------------------- leases

def read_leases(lease_dir):
    out = {}
    for name in listdir(lease_dir) or []:
        if re.fullmatch(r'WINH\d+\.json', name):
            out[name[:-5]] = read_json(os.path.join(lease_dir, name))
    return out


def acquire_leases(lease_dir, nodes, lease):
    """All-or-nothing exclusive creation of leases/<node>.json; an existing lease of the same attempt counts as held."""
    os.makedirs(lease_dir, exist_ok=True)
    created, conflicts = [], {}
    for node in nodes:
        path = os.path.join(lease_dir, node + '.json')
        try:
            write_json(path, dict(lease, node=node), exclusive=True)
            created.append(node)
        except FileExistsError:
            cur = read_json(path)
            if isinstance(cur, dict) and cur.get('attempt_id') == lease.get('attempt_id'):
                continue
            conflicts[node] = cur
    if conflicts:
        for node in created:
            try:
                os.remove(os.path.join(lease_dir, node + '.json'))
            except FileNotFoundError:
                pass
        return {'ok': False, 'created': [], 'conflicts': conflicts}
    return {'ok': True, 'created': created, 'conflicts': {}}


def release_leases(lease_dir, nodes, attempt_id):
    released, kept = [], {}
    for node in nodes:
        path = os.path.join(lease_dir, node + '.json')
        cur = read_json(path)
        if cur is None:
            continue
        if isinstance(cur, dict) and cur.get('attempt_id') == attempt_id:
            os.remove(path)
            released.append(node)
        else:
            kept[node] = cur
    return {'released': released, 'kept': kept}


# ---------------------------------------------------------------- code trees

def code_status(code_dir):
    return {'exists': os.path.isdir(code_dir), 'manifest': read_json(os.path.join(code_dir, 'MANIFEST.json'))}


def verify_code(code_dir, files):
    """Re-hash a cached code tree before it is reused: every manifest file present with its sha256, nothing else under
    infrarsi/ (no extra module, no __pycache__, no symlink) and MANIFEST.json readable. Read-only."""
    root = os.path.join(code_dir, 'infrarsi')
    mismatched, missing, extra = [], [], []
    for rel, sha in sorted(files.items()):
        p = os.path.join(root, rel)
        if os.path.islink(p) or not os.path.isfile(p):
            missing.append(rel)
            continue
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        if h.hexdigest() != sha:
            mismatched.append(rel)
    for base, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            p = os.path.join(base, name)
            rel = os.path.relpath(p, root).replace(os.sep, '/')
            if os.path.islink(p):
                extra.append(rel + ' (symlink)')
            elif name in names and rel not in files:
                extra.append(rel)
    for name in os.listdir(code_dir):
        if name not in ('infrarsi', 'MANIFEST.json'):
            extra.append('../' + name)
    manifest = read_json(os.path.join(code_dir, 'MANIFEST.json'))
    return {'ok': not (mismatched or missing or extra) and isinstance(manifest, dict), 'mismatched': mismatched[:50],
            'missing': missing[:50], 'extra': extra[:50], 'manifest_readable': isinstance(manifest, dict),
            'files': len(files)}


def install_code(tar_path, code_dir, manifest):
    """Extract a staged tar, verify every file against the manifest, then rename into place (atomic)."""
    if os.path.isdir(code_dir):
        os.remove(tar_path)
        return {'installed': False, 'existing': read_json(os.path.join(code_dir, 'MANIFEST.json'))}
    tmp = '%s.incoming.%d' % (code_dir, os.getpid())
    os.makedirs(tmp)
    try:
        with tarfile.open(tar_path) as tf:
            for m in tf.getmembers():
                if not m.isfile() or m.name.startswith('/') or '..' in m.name.split('/'):
                    raise ValueError('unsafe tar member %r' % m.name)
            try:
                tf.extractall(tmp, filter='data')
            except TypeError:                     # Python without extraction filters
                tf.extractall(tmp)
        for rel, sha in manifest['files'].items():
            with open(os.path.join(tmp, 'infrarsi', rel), 'rb') as f:
                got = hashlib.sha256(f.read()).hexdigest()
            if got != sha:
                raise ValueError('sha mismatch for %s' % rel)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)    # never leave a partial tree or the staged tar behind
        try:
            os.remove(tar_path)
        except FileNotFoundError:
            pass
        raise
    write_json(os.path.join(tmp, 'MANIFEST.json'), manifest)
    try:
        os.rename(tmp, code_dir)
    except OSError:                               # lost a race with another installer of the same digest
        shutil.rmtree(tmp, ignore_errors=True)
    os.remove(tar_path)
    return {'installed': True, 'existing': None}


# ---------------------------------------------------------------- launch and run state

def launch_wrapper(python, wrapper, spec_path, log_path):
    """Start the node wrapper in a new session (setsid + nohup); returns at once with its pid."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'ab') as log:
        p = subprocess.Popen(['nohup', python, wrapper, spec_path], stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)
    return {'pid': p.pid, 'hostname': socket.gethostname(), 'utc': utc()}


def _signals_sizes(out_dir):
    sig = os.path.join(out_dir, 'signals')
    res = {}
    for name in listdir(sig) or []:
        if name.endswith('.jsonl'):
            res[name] = os.path.getsize(os.path.join(sig, name))
    return res


def _last_json_line(path, nbytes=8192):
    t = tail(path, nbytes)
    if not t:
        return None
    for line in reversed(t.strip().splitlines()):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return None


def run_snapshot(out_dir, nnodes, log_tail_bytes=3000):
    """Cheap per-poll view of one attempt; ages use this node's clock against GPFS mtimes."""
    snap = {'utc': utc(), 'out_dir': out_dir, 'heartbeat': read_json(os.path.join(out_dir, 'heartbeat.json')),
            'heartbeat_age_s': file_age(os.path.join(out_dir, 'heartbeat.json')),
            'stop': read_json(os.path.join(out_dir, 'STOP.json')),
            'receipt_exists': os.path.exists(os.path.join(out_dir, 'receipt.json')),
            'signals': _signals_sizes(out_dir),
            'rank0_last': _last_json_line(os.path.join(out_dir, 'signals', 'rank0.jsonl')),
            'nodes': []}
    for i in range(nnodes):
        snap['nodes'].append({
            'node_index': i,
            'exit': read_json(os.path.join(out_dir, 'exit_node%d.json' % i)),
            'pid': read_json(os.path.join(out_dir, 'pid_node%d.json' % i)),
            'alive_age_s': file_age(os.path.join(out_dir, 'wrapper_node%d.json' % i)),
            'log_tail': tail(os.path.join(out_dir, 'logs', 'node%d.log' % i), log_tail_bytes),
        })
    return snap


def collect_run(out_dir, nnodes, log_tail_bytes=20000):
    """Everything the driver needs after the attempt ended (signals are small: one line per update per rank)."""
    sig_dir = os.path.join(out_dir, 'signals')
    signals = {}
    for name in listdir(sig_dir) or []:
        if name.endswith('.jsonl'):
            signals[name] = read_text(os.path.join(sig_dir, name))
    res = run_snapshot(out_dir, nnodes, log_tail_bytes)
    res['receipt'] = read_json(os.path.join(out_dir, 'receipt.json'))
    try:
        res['receipt_mtime'] = os.stat(os.path.join(out_dir, 'receipt.json')).st_mtime   # kernel clock, not the process's
    except FileNotFoundError:
        res['receipt_mtime'] = None
    res['signals_text'] = signals
    resume_dir = os.path.join(out_dir, 'signals_resume')
    res['signals_resume_text'] = {n: read_text(os.path.join(resume_dir, n)) for n in (listdir(resume_dir) or [])
                                  if n.endswith('.jsonl')}
    res['request'] = read_json(os.path.join(out_dir, 'request.json'))
    return res


def write_stop(out_dir, payload):
    path = os.path.join(out_dir, 'STOP.json')
    if os.path.exists(path):
        return {'path': path, 'written': False, 'existing': read_json(path)}
    write_json(path, payload)
    return {'path': path, 'written': True, 'existing': None}


# ---------------------------------------------------------------- monitor scans (read-only)

def _receipt_oom(rec):
    if not isinstance(rec, dict):
        return None
    ev = rec.get('events') or {}
    text = ' '.join(str(x) for x in (rec.get('invalid_reasons') or []) + list(ev.get('exceptions') or []))
    return bool((ev.get('alloc_ooms') or 0) > 0 or 'out of memory' in text.lower())


def scan_runs(runs_root, recent_hours=24.0, log_tail_bytes=2500, max_attempts=400):
    """Attempts whose directory changed recently or that have not finished; newest first."""
    now = time.time()
    found = []
    for run_id in listdir(runs_root) or []:
        rdir = os.path.join(runs_root, run_id)
        for att in listdir(rdir) or []:
            adir = os.path.join(rdir, att)
            req = os.path.join(adir, 'request.json')
            if not os.path.isfile(req):
                continue
            mtimes = [os.stat(adir).st_mtime]
            for name in ('heartbeat.json', 'receipt.json', 'STOP.json'):
                try:
                    mtimes.append(os.stat(os.path.join(adir, name)).st_mtime)
                except FileNotFoundError:
                    pass
            found.append((max(mtimes), run_id, att, adir))
    found.sort(reverse=True)
    out = []
    for mtime, run_id, att, adir in found[:max_attempts]:
        request = read_json(os.path.join(adir, 'request.json')) or {}
        nnodes = len(request.get('nodes') or [1])
        exits = [read_json(os.path.join(adir, 'exit_node%d.json' % i)) for i in range(nnodes)]
        finished = all(e is not None for e in exits)
        if finished and now - mtime > recent_hours * 3600:
            continue
        snap = run_snapshot(adir, nnodes, log_tail_bytes)
        snap.update(run_id=run_id, attempt_id=att, finished=finished, mtime_age_s=now - mtime,
                    launched_by_harness=os.path.exists(os.path.join(adir, 'launch', 'node0.json')),
                    request={k: request.get(k) for k in ('purpose', 'nodes', 'gpus_per_node', 'steps', 'window',
                                                          'wall_seconds_cap', 'input_plan', 'created_utc', 'world_size')},
                    task_id=(request.get('task') or {}).get('id'),
                    task_gpus=((request.get('task') or {}).get('hardware') or {}).get('gpus'),
                    task_family=((request.get('task') or {}).get('model') or {}).get('family'),
                    receipt_valid=(read_json(os.path.join(adir, 'receipt.json')) or {}).get('valid'),
                    receipt_oom=_receipt_oom(read_json(os.path.join(adir, 'receipt.json'))))
        out.append(snap)
    return out


def download_status(models_root, datasets_root, log_dir, log_tail_bytes=600):
    res = {'utc': utc(), 'entries': {}, 'logs': {}}
    for root, kind in ((models_root, 'model'), (datasets_root, 'dataset')):
        for name in listdir(root) or []:
            d = os.path.join(root, name)
            if os.path.isdir(d):
                res['entries'][name] = {'kind': kind, 'bytes': du_bytes(d),
                                        'done': read_json(os.path.join(d, 'DOWNLOAD_DONE.json'))}
    for name in listdir(log_dir) or []:
        p = os.path.join(log_dir, name)
        t = tail(p, log_tail_bytes) or ''
        lines = [x for x in re.split(r'[\r\n]+', t) if x.strip()]
        res['logs'][name] = {'age_s': file_age(p), 'last': lines[-1][-300:] if lines else '',
                             'failed': 'FAILED' in t, 'size': os.path.getsize(p)}
    return res


def ledger_status(ledger_root, tail_lines=5):
    out = {}
    for name in listdir(ledger_root) or []:
        p = os.path.join(ledger_root, name)
        if os.path.isfile(p):
            t = tail(p, 4000) or ''
            out[name] = {'size': os.path.getsize(p), 'age_s': file_age(p),
                         'tail': t.strip().splitlines()[-tail_lines:]}
    return out


def hf_model_facts(model_dir):
    """config.json plus only the metadata of model.safetensors.index.json (the weight map can be megabytes)."""
    cfg = read_json(os.path.join(model_dir, 'config.json'))
    idx = read_json(os.path.join(model_dir, 'model.safetensors.index.json'))
    files = listdir(model_dir) or []
    return {'model_dir': model_dir, 'config': cfg, 'index_metadata': (idx or {}).get('metadata') if isinstance(idx, dict) else None,
            'safetensors': sorted(f for f in files if f.endswith('.safetensors')),
            'download_done': read_json(os.path.join(model_dir, 'DOWNLOAD_DONE.json'))}


def tree_digest(root, suffixes=('.md', '.py', '.json', '.yaml', '.db', '.txt')):
    """sha256 of (relative path, file sha256) pairs; used by the monitor to watch frozen packages."""
    pairs = []
    for base, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(suffixes):
                p = os.path.join(base, name)
                with open(p, 'rb') as f:
                    pairs.append([os.path.relpath(p, root).replace(os.sep, '/'), hashlib.sha256(f.read()).hexdigest()])
    pairs.sort()
    return hashlib.sha256(json.dumps(pairs, separators=(',', ':')).encode()).hexdigest() if pairs else None
