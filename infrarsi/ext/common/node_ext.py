"""Node-side helpers of the ext runners (stdlib only, Python >= 3.8, run as root on a WINH node).

remote_exec.WinhNodeOps ships this file's source with every call (`base64 -d | python3 - <args>`), so the code that
touches /data/infrarsi on the nodes is versioned with the driver; remote_exec.LocalNodeOps calls the same functions
in-process (CPU tests). Every function is idempotent except `launch`, whose scripts guard themselves with an atomic
mkdir. Processes are signalled only by `kill_token`, and only those carrying this run's INFRARSI_TOKEN.
"""
import base64
import datetime
import hashlib
import json
import os
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time

SENTINEL = '@@INFRARSI_EXT@@'
TOKEN_VAR = 'INFRARSI_TOKEN'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _smi(args):
    p = subprocess.run(['nvidia-smi'] + args, capture_output=True, text=True, timeout=60)
    if p.returncode:
        raise RuntimeError('nvidia-smi %s: %s' % (' '.join(args), p.stderr.strip()[-300:]))
    return [line.strip() for line in p.stdout.splitlines() if line.strip()]


def gpu_probe():
    """GPUs (index, uuid, memory used) and compute processes with their GPU index, cwd and command line."""
    out = {'hostname': socket.gethostname(), 'utc': utc(), 'error': None, 'gpus': [], 'apps': []}
    try:
        uuid_to_index = {}
        for line in _smi(['--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits']):
            idx, uuid, used = [x.strip() for x in line.split(',')]
            uuid_to_index[uuid] = int(idx)
            out['gpus'].append({'index': int(idx), 'uuid': uuid, 'mem_used_mib': int(used) if used.isdigit() else None})
        for line in _smi(['--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader,nounits']):
            pid, uuid, used = [x.strip() for x in line.split(',', 2)]
            app = {'pid': int(pid), 'gpu_index': uuid_to_index.get(uuid), 'used_mib': int(used) if used.isdigit() else None}
            try:
                app['cwd'] = os.readlink('/proc/%d/cwd' % int(pid))
            except OSError:
                app['cwd'] = None
            try:
                with open('/proc/%d/cmdline' % int(pid), 'rb') as f:
                    app['cmdline'] = f.read(600).replace(b'\0', b' ').decode('utf-8', 'replace').strip()
            except OSError:
                app['cmdline'] = None
            app['token'] = _token_of(int(pid))
            app['uid'] = _uid_of(int(pid))         # /proc/<pid>/status is readable for any uid (environ is not)
            out['apps'].append(app)
    except Exception as e:                          # noqa: BLE001 - a probe reports, it never raises
        out['error'] = repr(e)
    return out


def _uid_of(pid):
    try:
        with open('/proc/%d/status' % pid, 'r') as f:
            for line in f:
                if line.startswith('Uid:'):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _token_of(pid):
    try:
        with open('/proc/%d/environ' % pid, 'rb') as f:
            for item in f.read().split(b'\0'):
                if item.startswith(TOKEN_VAR.encode() + b'='):
                    return item.split(b'=', 1)[1].decode('utf-8', 'replace')
    except OSError:
        return None
    return None


def kill_token(token):
    """SIGKILL every process whose environment carries INFRARSI_TOKEN=<token> (processes this run started); returns
    their pids. Nothing else is ever signalled."""
    if not token or len(token) < 8:
        raise ValueError('refusing an empty or short token')
    killed = []
    for name in os.listdir('/proc'):
        if not name.isdigit() or int(name) == os.getpid():
            continue
        if _token_of(int(name)) == token:
            try:
                os.kill(int(name), signal.SIGKILL)
                killed.append(int(name))
            except OSError:
                pass
    return killed


def token_pids(token):
    return [int(n) for n in os.listdir('/proc') if n.isdigit() and _token_of(int(n)) == token]


# ---------------------------------------------------------------- uid trust domains (Codex W9 B-02)
# Containment is by numeric uid: every untrusted process of an arm runs as that arm's uid, which nothing else on the
# node uses (assert_uid_unused); a process cannot leave its uid without privileges (setpriv --no-new-privs blocks setuid
# binaries), so killing the uid kills everything the arm started, setsid/double-forked children included. The token
# above is only an audit label. Never used for 65534 (system daemons may run as nobody) or below MIN_UID.
MIN_UID = 60000


def uid_pids(uid, zombies=False):
    """Pids whose real, effective, saved or fs uid is `uid` (from /proc/<pid>/status). Zombies (exited, waiting for
    their parent to reap them; they hold no memory, GPU or files) are left out unless zombies=True."""
    uid, out = str(int(uid)), []
    for name in os.listdir('/proc'):
        if not name.isdigit() or int(name) == os.getpid():
            continue
        try:
            state, mine = None, False
            with open('/proc/%s/status' % name, 'r') as f:
                for line in f:
                    if line.startswith('State:'):
                        state = line.split()[1]
                    elif line.startswith('Uid:'):
                        mine = uid in line.split()[1:5]
                        break
            if mine and (zombies or state != 'Z'):
                out.append(int(name))
        except OSError:
            pass
    return out


def _check_domain_uid(uid):
    uid = int(uid)
    if uid < MIN_UID or uid == 65534:
        raise ValueError('refusing kill-by-uid for uid %d (trust-domain uids are >= %d and never 65534)' % (uid, MIN_UID))
    return uid


def uid_kill(uid, rounds=40):
    """SIGKILL every process of trust-domain `uid`: `kill -9 -1` issued as that uid (one syscall reaches all of its
    processes, also ones forked between two scans), then root kills whatever a /proc scan still finds, until none
    remain or the rounds (about 30 s; CUDA teardown can take seconds) run out. Zombies are reported, not waited for."""
    uid = _check_domain_uid(uid)
    killed, r = [], 0
    for r in range(int(rounds)):
        pids = uid_pids(uid)
        if not pids:
            break
        if r % 5 == 0:
            subprocess.run(['setpriv', '--reuid=%d' % uid, '--regid=%d' % uid, '--clear-groups', 'kill', '-9', '-1'],
                           capture_output=True, timeout=30)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        killed += pids
        time.sleep(0.1 if r < 10 else 1.0)
    return {'uid': uid, 'killed': sorted(set(killed)), 'remaining': uid_pids(uid), 'rounds': r,
            'zombies': sorted(set(uid_pids(uid, zombies=True)) - set(uid_pids(uid)))}


def uid_status(uid):
    """{'uid', 'passwd': name or None, 'pids': [...]} - a trust-domain uid must have neither."""
    import pwd
    try:
        name = pwd.getpwuid(int(uid)).pw_name
    except KeyError:
        name = None
    return {'uid': int(uid), 'passwd': name, 'pids': uid_pids(uid)}


def read_json(path, default=None):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except ValueError as e:
        return {'__unreadable__': repr(e)}


def _owner(path, uid, gid):
    if uid is not None:
        os.chown(path, int(uid), int(gid if gid is not None else uid))
    elif gid is not None:                          # root-owned, readable by one group (e.g. an arm's code dir)
        os.chown(path, -1, int(gid))


def mkdirs(paths, mode=None, uid=None, gid=None):
    for p in paths:
        os.makedirs(p, exist_ok=True)
        if mode is not None:
            os.chmod(p, int(mode))
        _owner(p, uid, gid)
    return {'made': list(paths)}


def mkdir_domain(paths, mode, uid=None, gid=None):
    """Directories of one trust domain: a missing one is created (parents root-owned) with owner uid (root if None),
    group gid (root if None) and mode; an existing one must already have exactly that owner and group and is only
    re-moded. A directory is never re-owned or re-grouped (a parent shared by several uids would lock the others out)
    and never a symlink. Returns {'made': [...], 'kept': [...]}."""
    want = (0 if uid is None else int(uid), 0 if gid is None else int(gid))
    made, kept = [], []
    for p in paths:
        parent = os.path.dirname(p.rstrip('/'))
        if parent:
            os.makedirs(parent, exist_ok=True)
        try:
            os.mkdir(p)                        # exclusive: of two concurrent creators exactly one owns it
            os.chmod(p, int(mode))
            os.chown(p, want[0], want[1])
            made.append(p)
            continue
        except FileExistsError:
            pass
        st = os.lstat(p)
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
            raise ValueError('%s is not a plain directory' % p)
        if (st.st_uid, st.st_gid) != want:
            raise ValueError('refusing to re-own %s (%d:%d, wanted %d:%d): a directory of another trust domain'
                             % (p, st.st_uid, st.st_gid, want[0], want[1]))
        os.chmod(p, int(mode))
        kept.append(p)
    return {'made': made, 'kept': kept}


def write_files(files, mode=None, uid=None, gid=None):
    """files {path: base64 bytes}; atomic replace, parents created (root-owned unless uid)."""
    for path, b64 in files.items():
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        tmp = '%s.%d.tmp' % (path, os.getpid())
        with open(tmp, 'wb') as f:
            f.write(base64.b64decode(b64))
        if mode is not None:
            os.chmod(tmp, int(mode))
        _owner(tmp, uid, gid)
        os.replace(tmp, path)
    return {'written': sorted(files)}


def read_files(paths, max_bytes=None):
    """{path: base64 bytes | None}; max_bytes keeps the tail of larger files."""
    out = {}
    for p in paths:
        try:
            size = os.path.getsize(p)
            with open(p, 'rb') as f:
                if max_bytes is not None and size > int(max_bytes):
                    f.seek(size - int(max_bytes))
                out[p] = base64.b64encode(f.read()).decode('ascii')
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            out[p] = None
    return out


def hash_files(paths):
    """{path: sha256 | None}; symlinks are followed (frozen data is linked into candidate homes)."""
    out = {}
    for p in paths:
        try:
            h = hashlib.sha256()
            with open(p, 'rb') as f:
                for chunk in iter(lambda: f.read(1 << 22), b''):
                    h.update(chunk)
            out[p] = h.hexdigest()
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError):
            out[p] = None
    return out


def sizes(paths):
    out = {}
    for p in paths:
        try:
            out[p] = os.path.getsize(p)
        except OSError:
            out[p] = None
    return out


def listdir(path):
    """Sorted entries of a directory; None when the path is missing or not a directory."""
    try:
        return sorted(os.listdir(path))
    except (FileNotFoundError, NotADirectoryError):
        return None


def symlinks(links):
    """links {link path: target}; parents created; an existing link to another target is replaced."""
    for link, target in links.items():
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if os.path.islink(link):
            if os.readlink(link) == target:
                continue
            os.remove(link)
        os.symlink(target, link)
    return {'linked': sorted(links)}


def acquire(path, lease):
    """Exclusive create of a lease/claim file; an existing one with the same attempt_id counts as held."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, 'x', encoding='utf-8') as f:
            json.dump(lease, f, indent=2, sort_keys=True)
        return {'ok': True, 'existing': None}
    except FileExistsError:
        cur = read_json(path)
        if isinstance(cur, dict) and cur.get('attempt_id') == lease.get('attempt_id'):
            return {'ok': True, 'existing': cur}
        return {'ok': False, 'existing': cur}


def release(path, attempt_id):
    cur = read_json(path)
    if isinstance(cur, dict) and cur.get('attempt_id') == attempt_id:
        os.remove(path)
        return True
    return False


def release_status(path, attempt_id):
    """Idempotent release: {'state': 'removed' | 'absent' | 'foreign', 'holder'}; a foreign lease is never touched."""
    cur = read_json(path)
    if cur is None:
        return {'state': 'absent', 'holder': None}
    if isinstance(cur, dict) and cur.get('attempt_id') == attempt_id:
        os.remove(path)
        return {'state': 'removed', 'holder': None}
    return {'state': 'foreign', 'holder': cur}


def remove_tree(path, root):
    """rm -rf of a path strictly inside root (refuses anything else). A symlink is removed itself, never followed."""
    base = os.path.realpath(root)
    parent = os.path.realpath(os.path.dirname(os.path.abspath(path)))
    target = os.path.join(parent, os.path.basename(os.path.abspath(path)))
    if target == base or os.path.commonpath([target, base]) != base:
        raise ValueError('refusing to remove %s outside %s' % (path, root))
    if os.path.islink(target):
        os.remove(target)
    elif os.path.isdir(target):
        shutil.rmtree(target, ignore_errors=True)
    elif os.path.lexists(target):
        os.remove(target)
    return {'removed': path}


# ---------------------------------------------------------------- files inside trees an untrusted user owns
# Root reads and writes inside nobody-owned trees (agent homes, candidate out/ dirs) never follow a symlink below the
# tree's root: each directory component is opened with O_NOFOLLOW|O_DIRECTORY relative to its parent, the file itself
# with O_NOFOLLOW (and created with O_EXCL), so a planted link cannot redirect root to another file.

def _walk(root, path):
    base = os.path.realpath(root)
    # the path below the root as the caller names both (a root reached through a symlinked mount - /data -> /mnt/data
    # on WINH2: every file looked outside its resolved root), else against the resolved root
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    if rel.split(os.sep)[0] == '..':
        rel = os.path.relpath(os.path.abspath(path), base)
    parts = rel.split(os.sep)
    if rel in ('.', '') or parts[0] == '..' or os.path.isabs(rel):
        raise ValueError('%s is not inside %s' % (path, root))
    fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for comp in parts[:-1]:
            nfd = os.open(comp, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nfd
    except BaseException:
        os.close(fd)
        raise
    return fd, parts[-1]


def read_user_file(path, root, max_bytes=None):
    """{'ok', 'b64', 'size', 'reason'}: a regular file under root, read without following symlinks."""
    try:
        dfd, name = _walk(root, path)
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
        finally:
            os.close(dfd)
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                return {'ok': False, 'reason': 'not a regular file'}
            if max_bytes is not None and st.st_size > int(max_bytes):
                return {'ok': False, 'reason': 'larger than %d bytes' % int(max_bytes), 'size': st.st_size}
            chunks = []
            while True:
                b = os.read(fd, 1 << 20)
                if not b:
                    break
                chunks.append(b)
            data = b''.join(chunks)
        finally:
            os.close(fd)
        return {'ok': True, 'b64': base64.b64encode(data).decode('ascii'), 'size': len(data),
                'sha256': hashlib.sha256(data).hexdigest()}
    except FileNotFoundError:
        return {'ok': False, 'reason': 'missing'}
    except OSError as e:                           # ELOOP: a symlink where a file or directory was expected
        return {'ok': False, 'reason': 'refused: %s' % e.strerror}


def write_user_file(path, root, b64, uid=None, gid=None, mode=0o644):
    """Replace a file under root by a new regular file (the old entry - even a symlink - is unlinked, never followed;
    O_EXCL|O_NOFOLLOW on create), then fchown/fchmod it."""
    dfd, name = _walk(root, path)
    try:
        try:
            os.unlink(name, dir_fd=dfd)
        except FileNotFoundError:
            pass
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, int(mode), dir_fd=dfd)
        try:
            os.write(fd, base64.b64decode(b64))
            if uid is not None:
                os.fchown(fd, int(uid), int(gid if gid is not None else uid))
            os.fchmod(fd, int(mode))
        finally:
            os.close(fd)
    finally:
        os.close(dfd)
    return {'ok': True, 'path': path}


def regular_file_info(paths):
    """{path: {'size', 'sha256'} | None}: regular files only, opened with O_NOFOLLOW (a symlink counts as missing)."""
    out = {}
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError:
            out[p] = None
            continue
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                out[p] = None
                continue
            h, n = hashlib.sha256(), 0
            while True:
                b = os.read(fd, 1 << 22)
                if not b:
                    break
                h.update(b)
                n += len(b)
            out[p] = {'size': n, 'sha256': h.hexdigest()}
        finally:
            os.close(fd)
    return out


def launch(script, log, env=None):
    """Start `bash script` in a new session (setsid + nohup); returns at once. The script's own mkdir guard makes a
    repeated launch a no-op. env: extra variables (e.g. the script's own INFRARSI_TOKEN, so kill_token can stop it)."""
    os.makedirs(os.path.dirname(log), exist_ok=True)
    full = dict(os.environ, **{str(k): str(v) for k, v in (env or {}).items()})
    with open(log, 'ab') as f:
        p = subprocess.Popen(['nohup', 'bash', script], stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True, close_fds=True, env=full)
    return {'pid': p.pid, 'hostname': socket.gethostname(), 'utc': utc()}


def sh(cmd, timeout=120):
    p = subprocess.run(['bash', '-c', cmd], capture_output=True, timeout=timeout)
    return {'code': p.returncode, 'stdout': p.stdout.decode('utf-8', 'replace'), 'stderr': p.stderr.decode('utf-8', 'replace')}


def _main():
    arg = sys.argv[1]
    if arg.startswith('@'):                        # a large request comes as a root-only file (one argv string <= 128 KiB)
        with open(arg[1:], 'rb') as f:
            req = json.loads(f.read().decode('utf-8'))
    else:
        req = json.loads(base64.b64decode(arg).decode('utf-8'))
    try:
        res = {'ok': True, 'value': globals()[req['func']](**req['kw'])}
    except BaseException as e:                     # noqa: BLE001 - reported to the driver
        import traceback
        res = {'ok': False, 'error': repr(e), 'traceback': traceback.format_exc()[-3000:]}
    sys.stdout.write('\n' + SENTINEL + json.dumps(res) + '\n')


if __name__ == '__main__':
    _main()
