"""ENV-A node-side scorer (design §2.3; Follow-ups 1-2 / Codex W9 R1). Runs on the leased WINH node as root under the
env's Python; never imported by driver code.

Trust split
  * root (this process) orchestrates and judges. It runs the task's own score.py as the authority, with three seams
    injected into the loaded module (score.py's bytes are untouched, G2):
      score.subprocess     -> UidSubprocess: score.py's one `check_call(["./finetune.sh"])` runs as the arm uid
      score.compare_models -> safe compare: after score.py has measured execution_time and before any model load, it
                              kills the arm uid, records G8 on the raw output, has the UNPRIVILEGED sanitizer convert it
                              to tensors-only safetensors + validated JSON, installs that root-owned read-only (also as
                              solution/trained_model), and compares the installed copy (sanitizer failure -> G9, and
                              score.py's own model_not_found path)
      score.load_model     -> safe loader: only {model.safetensors, config.json, generation_config.json}, regular
                              files, `use_safetensors=True` -- root never deserialises a pickle (B-01)
  * candidate code (finetune.sh, the sanitizer of its outputs) runs as the arm's dedicated numeric uid (A-NAT 64101,
    A-ISW 64102, A-RND 64103) and the arm's trusted unprivileged work (G4's ref'(D'), the base-model conversion) as
    the arm's own judge uid (64111 / 64112 / 64113; Codex W9 R2 N-01: never shared between arms), each with a clean
    environment and no-new-privs. After every training / sanitize every process of that uid is killed (pkill -u/-U plus `kill -9 -1`
    as the uid, verified on /proc, retried) and its /dev/shm leftovers are removed; uid 65534 is never used (B-02).
    INFRARSI_TOKEN is only an audit label.
  * root never writes through a path the arm uid controls: homes and HF caches are created fresh and then handed over;
    the sanitizer's staging dir is read component by component with O_NOFOLLOW, file by file (regular, owned by the
    uid) into a fresh root-owned dir which is then validated. The reference, the base-model copy, ref'(D') and the
    installed candidate models are root-owned under 0700 dirs; ref'(D')'s tree digest is re-verified before and after
    every comparison that uses it (H-01). env.sh / train_data.jsonl are root-owned 0644 in a root-owned home (M-01).
    Overlay paths are canonical-relative and contained by path components (H-03).
  * B-05: a reuse-home scoring (A-NAT) snapshots the solution tree before scoring (tar named by its content digest);
    a locked re-score restores the selected snapshot into a fresh home at the canonical path (the final working tree
    is moved aside, root-only, never scored) and requires the restored digest to equal the selection's. A locked
    re-score ('clean_slate') also gets a fresh HOME and starts with the arm uid's /tmp, /var/tmp, /dev/shm leftovers
    removed, so no cache carried over from the search can shorten it.

Job (JSON file in argv[1]): mode 'score' (default) or 'prepare'; run_id, attempt_id, purpose, uid, judge_uid, gpus, env_dir,
task_dir, foundry_dir, model_dir, model_id, model_rev, run_dir, sanitize_py, finetune_sh, finetune_yaml,
finetune_sh_base, finetune_yaml_base, overlay, reference_src, run_g4, g4_share, reuse_home, restore_snapshot,
expected_digest, clean_slate, score_timeout, timeout_s, out. Writes {'ok', 'result'|'error'} to job['out'].
"""
from __future__ import annotations

import contextlib
import datetime
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import uuid

SENTINEL = '@@X2_NODE_RESULT@@'
SANITIZE_SENTINEL = '@@X2_SANITIZE@@'
MODEL_FILES = ('config.json', 'generation_config.json', 'pytorch_model.bin', 'special_tokens_map.json',
               'tokenizer.json', 'tokenizer_config.json', 'LICENSE', 'README.md')
REAL_RE = re.compile(r'real\s+(\d+)m([\d.]+)s')
TOKEN_VAR = 'INFRARSI_TOKEN'
ARM_UIDS = (64101, 64102, 64103, 64104, 64105)
JUDGE_UIDS = (64111, 64112, 64113, 64114, 64115)
ARM_JUDGE = dict(zip(ARM_UIDS, JUDGE_UIDS))     # each arm's judge uid; a job may touch only its own pair
OUR_UIDS = ARM_UIDS + JUDGE_UIDS
FORBIDDEN_UID = 65534                     # never used, never killed by uid: system daemons may run as nobody
ALLOWED_MODEL_FILES = ('model.safetensors', 'config.json', 'generation_config.json')
BANNED_CONFIG_KEYS = ('auto_map', 'trust_remote_code', 'custom_pipelines', 'quantization_config')
DTYPE_BYTES = {'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2, 'I64': 8, 'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1,
               'F8_E4M3': 1, 'F8_E5M2': 1}
MAX_MODEL_BYTES = 16 << 30
MAX_JSON_BYTES = 1 << 20
MAX_SOLUTION_BYTES = 2 << 30              # a solution tree (outputs excluded) larger than this is not snapshot / scored
OUTPUT_DIRS = ('finetuning_tokens', 'ftout', 'trained_model')
LEFTOVER_DIRS = ('/dev/shm', '/tmp', '/var/tmp')
_ACTIVE: dict = {'uids': set(), 'tmpdirs': set()}
# open(2) flags; the getattr defaults only let the pure helpers run in the PC's CPU tests (the node is Linux)
_NOFOLLOW, _NONBLOCK, _BINARY = (getattr(os, n, 0) for n in ('O_NOFOLLOW', 'O_NONBLOCK', 'O_BINARY'))


# ------------------------------------------------------------------ paths (H-03)
def is_within(child, parent) -> bool:
    """Component containment of the resolved paths (never a string-prefix test)."""
    c, p = pathlib.Path(os.path.realpath(child)), pathlib.Path(os.path.realpath(parent))
    return c == p or c.is_relative_to(p)


def canonical_rel(rel) -> bool:
    """A relative POSIX path: no leading '/' or '~', no backslash / NUL / drive, no empty, '.' or '..' component."""
    if not isinstance(rel, str) or not rel or rel.startswith(('/', '~')) or '\\' in rel or '\x00' in rel or ':' in rel:
        return False
    return all(part not in ('', '.', '..') for part in rel.split('/'))


def private_dir(path, uid: int | None = None, mode: int = 0o700) -> pathlib.Path:
    """mkdir (parents root-created) and set owner/mode, refusing a symlink or a dir owned by anyone but root / `uid`."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(FileExistsError):
        os.mkdir(p, mode)
    st = os.lstat(p)
    owner = 0 if uid is None else int(uid)
    if not stat.S_ISDIR(st.st_mode):
        raise ValueError(f'{p} is not a real directory')
    if st.st_uid not in (0, owner):
        raise ValueError(f'{p} is owned by uid {st.st_uid}, not {owner}')
    os.chown(p, owner, owner)
    os.chmod(p, mode)
    return p


def remove_tree(path) -> None:
    """Remove a path without following symlinks (root needs no write permission on read-only dirs)."""
    p = pathlib.Path(path)
    if os.path.islink(p) or (os.path.lexists(p) and not os.path.isdir(p)):
        os.unlink(p)
    elif os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)


# ------------------------------------------------------------------ hashing
def sha256_file(path) -> str | None:
    path = pathlib.Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def tree_digest_files(d) -> str:
    """Digest of a flat root-owned model dir: sorted (name, sha256) of its files (H-01, installed copies)."""
    d = pathlib.Path(d)
    rows = sorted((n, sha256_file(d / n)) for n in os.listdir(d))
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def tree_digest_solution(sol, max_bytes: int = MAX_SOLUTION_BYTES) -> str:
    """Content identity of a solution tree (B-05): every dir, regular file (sha256 + exec bit) and symlink (target),
    without score.py's three outputs at the top level and without __pycache__. Run only while no process of the
    arm uid is alive. Raises ValueError above max_bytes."""
    sol = pathlib.Path(sol)
    rows, total = [], 0
    for dp, dns, fns in os.walk(sol, followlinks=False):
        rel_dp = os.path.relpath(dp, sol)
        dns[:] = sorted(d for d in dns if d != '__pycache__' and not (rel_dp == '.' and d in OUTPUT_DIRS))
        for d in dns:
            p = os.path.join(dp, d)
            rel = os.path.relpath(p, sol).replace(os.sep, '/')
            rows.append([rel, 'l', os.readlink(p)] if os.path.islink(p) else [rel, 'd'])
        for fn in sorted(fns):
            if rel_dp == '.' and fn in OUTPUT_DIRS:
                continue
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, sol).replace(os.sep, '/')
            st = os.lstat(p)
            if stat.S_ISLNK(st.st_mode):
                rows.append([rel, 'l', os.readlink(p)])
            elif stat.S_ISREG(st.st_mode):
                total += st.st_size
                if total > max_bytes:
                    raise ValueError(f'solution tree exceeds {max_bytes} bytes (outputs excluded)')
                rows.append([rel, 'f', sha256_file(p), bool(st.st_mode & 0o111)])
            else:
                rows.append([rel, 'x', stat.S_IFMT(st.st_mode)])
    rows.sort()
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def env_sh_text(home, model_id: str) -> str:
    """The exact env.sh the task expects for a home (M-01; the driver derives the same text for G2)."""
    return f'export BASE_MODEL="{model_id}"\nexport PYTHONPATH="{home}/solution${{PYTHONPATH:+:$PYTHONPATH}}"\n'


def frozen_hashes(score_py, home: pathlib.Path, model_dir: pathlib.Path) -> dict:
    return {'score.py': sha256_file(score_py), 'train_data.jsonl': sha256_file(home / 'train_data.jsonl'),
            'env.sh': sha256_file(home / 'env.sh'),
            'base_model.pytorch_model.bin': sha256_file(model_dir / 'pytorch_model.bin'),
            'base_model.config.json': sha256_file(model_dir / 'config.json')}


def owner_mode(path) -> dict:
    try:
        st = os.lstat(path)
        return {'uid': st.st_uid, 'mode': oct(stat.S_IMODE(st.st_mode))[2:], 'regular': stat.S_ISREG(st.st_mode)}
    except FileNotFoundError:
        return {'uid': None, 'mode': None, 'regular': False}


# ------------------------------------------------------------------ uids (B-02)
def check_uid(uid) -> int:
    uid = int(uid)
    if uid == FORBIDDEN_UID or uid not in OUR_UIDS:
        raise ValueError(f'uid {uid} is not an ENV-A arm / judge uid {OUR_UIDS}')
    return uid


def judge_for(uid, judge_uid) -> int:
    """The job's judge uid, which must be the judge of the job's arm uid (Codex W9 R2 N-01)."""
    uid, judge_uid = check_uid(uid), check_uid(judge_uid)
    if ARM_JUDGE.get(uid) != judge_uid:
        raise ValueError(f'judge uid {judge_uid} is not the judge of arm uid {uid} ({ARM_JUDGE})')
    return judge_uid


def uid_tmpdir(token: str, uid: int) -> str:
    """Short private TMPDIR per (job token, uid): fits the 108-byte AF_UNIX limit, and streaming then uses
    <TMPDIR>/streaming instead of the shared /tmp/streaming."""
    return f'/tmp/{token}-u{int(uid)}'


def ensure_uid_tmp(token: str, uid: int) -> str:
    d = uid_tmpdir(token, uid)
    private_dir(d, uid=check_uid(uid))
    _ACTIVE['tmpdirs'].add(d)
    return d


def uid_env(env_bin: str, gpus: str, home_root, token: str, uid: int) -> dict:
    """Clean environment for code run as an arm / judge uid (ensure_uid_tmp first)."""
    uid = check_uid(uid)
    return {'PATH': f'{env_bin}:/usr/local/cuda-12.8/bin:/usr/bin:/bin', 'CUDA_VISIBLE_DEVICES': gpus,
            'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'TOKENIZERS_PARALLELISM': 'false', 'HOME': str(home_root), 'XDG_CACHE_HOME': f'{home_root}/.cache',
            'TMPDIR': uid_tmpdir(token, uid), 'USER': f'isx{uid}', 'LOGNAME': f'isx{uid}', 'PYTHONUNBUFFERED': '1',
            'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', TOKEN_VAR: token}


def uid_command(inner: str, env: dict, uid: int) -> str:
    """`inner` as the numeric uid/gid, groups cleared, no-new-privs, cleared environment, non-login shell (the env is
    fully specified; a login profile may query the passwd entry these uids deliberately lack)."""
    uid = check_uid(uid)
    exported = f'export PATH={shlex.quote(env["PATH"])}; ' + inner
    assignments = ' '.join(f'{k}={shlex.quote(str(v))}' for k, v in env.items())
    return (f'setpriv --reuid={uid} --regid={uid} --clear-groups --no-new-privs env -i {assignments} '
            f'bash -c {shlex.quote(exported)}')


def uid_pids(uid: int) -> list:
    """Live (non-zombie) pids whose real, effective, saved or filesystem uid is `uid`."""
    uid, out = str(check_uid(uid)), []
    for name in os.listdir('/proc'):
        if not name.isdigit() or int(name) == os.getpid():
            continue
        try:
            with open(f'/proc/{name}/status', encoding='utf-8', errors='replace') as f:
                status = f.read()
        except OSError:
            continue
        m_uid = re.search(r'^Uid:\s+(.*)$', status, re.M)
        m_state = re.search(r'^State:\s+(\S)', status, re.M)
        if m_uid and uid in m_uid.group(1).split()[:4] and not (m_state and m_state.group(1) == 'Z'):
            out.append(int(name))
    return out


def kill_uid(uid: int, rounds: int = 40) -> dict:
    """SIGKILL every process of the uid -- pkill -u/-U plus `kill -9 -1` issued as the uid (reaches processes forked
    between scans) -- until a /proc scan finds none alive (zombies hold nothing and are ignored)."""
    uid = check_uid(uid)
    for i in range(rounds):
        if not uid_pids(uid):
            return {'uid': uid, 'clean': True, 'rounds': i}
        for flag in ('-u', '-U'):
            subprocess.run(['pkill', '-9', flag, str(uid)], capture_output=True)
        if i % 4 == 0:
            subprocess.run(['setpriv', f'--reuid={uid}', f'--regid={uid}', '--clear-groups', 'kill', '-9', '-1'],
                           capture_output=True, timeout=30)
        time.sleep(0.25 if i < 8 else 1.0)
    left = uid_pids(uid)
    return {'uid': uid, 'clean': not left, 'rounds': rounds, 'remaining': left[:20]}


def clean_uid_leftovers(uid: int, *, tmp: bool = False) -> list:
    """Remove the uid's top-level entries in /dev/shm (stale streaming / NCCL segments break the next run) and, with
    tmp, in /tmp and /var/tmp (clean slate: no cache from earlier runs). Call after kill_uid."""
    uid = check_uid(uid)
    gone = []
    for base in (LEFTOVER_DIRS if tmp else LEFTOVER_DIRS[:1]):
        with contextlib.suppress(OSError):
            for e in os.scandir(base):
                with contextlib.suppress(OSError):
                    if e.stat(follow_symlinks=False).st_uid == uid:
                        remove_tree(e.path)
                        gone.append(e.path)
    return gone


def chown_tree(path, uid: int) -> None:
    """lchown a tree (never follows a symlink; os.walk does not descend into symlinked dirs)."""
    path = pathlib.Path(path)
    if not os.path.lexists(path):
        return
    os.lchown(path, uid, uid)
    for dp, dns, fns in os.walk(path, followlinks=False):
        for n in dns + fns:
            with contextlib.suppress(OSError):
                os.lchown(os.path.join(dp, n), uid, uid)


# ------------------------------------------------------------------ model dirs (B-01): root-side checks, no deserialising
def read_safetensors_header(path, max_header: int = 100 << 20) -> dict:
    """Parse and bounds-check a safetensors header by hand (no library, no tensor loaded)."""
    fd = os.open(str(path), os.O_RDONLY | _NOFOLLOW | _BINARY)
    with os.fdopen(fd, 'rb') as f:
        size = os.fstat(f.fileno()).st_size
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError('safetensors: truncated header length')
        n = int.from_bytes(raw, 'little')
        if n <= 0 or n > max_header or 8 + n > size:
            raise ValueError(f'safetensors: bad header length {n}')
        hdr = json.loads(f.read(n))
    if not isinstance(hdr, dict):
        raise ValueError('safetensors: header is not an object')
    data_len, count = size - 8 - n, 0
    for name, info in hdr.items():
        if name == '__metadata__':
            continue
        if not isinstance(info, dict) or info.get('dtype') not in DTYPE_BYTES:
            raise ValueError(f'safetensors: tensor {name!r} has an unsupported dtype')
        shape, off = info.get('shape'), info.get('data_offsets')
        if not (isinstance(shape, list) and all(type(x) is int and x >= 0 for x in shape)):
            raise ValueError(f'safetensors: tensor {name!r} has a bad shape')
        if not (isinstance(off, list) and len(off) == 2 and all(type(x) is int for x in off) and 0 <= off[0] <= off[1] <= data_len):
            raise ValueError(f'safetensors: tensor {name!r} has bad offsets')
        numel = 1
        for x in shape:
            numel *= x
        if numel * DTYPE_BYTES[info['dtype']] != off[1] - off[0]:
            raise ValueError(f'safetensors: tensor {name!r} size does not match its shape / dtype')
        count += 1
    if not count:
        raise ValueError('safetensors: no tensors')
    return {'n_tensors': count, 'bytes': size}


def _json_file(path) -> dict:
    st = os.lstat(path)
    if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_JSON_BYTES:
        raise ValueError(f'{pathlib.Path(path).name}: not a regular JSON file <= {MAX_JSON_BYTES} bytes')
    obj = json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
    if not isinstance(obj, dict):
        raise ValueError(f'{pathlib.Path(path).name} is not a JSON object')
    return obj


def check_config_dict(cfg: dict, expected_model_type: str | None) -> None:
    bad = [k for k in BANNED_CONFIG_KEYS if k in cfg]
    if bad:
        raise ValueError(f'config.json has custom-code / quantisation keys {bad}')
    if expected_model_type is not None and cfg.get('model_type') != expected_model_type:
        raise ValueError(f'config.json model_type {cfg.get("model_type")!r} != {expected_model_type!r}')


def validate_model_dir(d, expected_model_type: str | None) -> dict:
    """A root-owned installed model dir: only {model.safetensors, config.json[, generation_config.json]}, regular
    files within size limits, a well-formed safetensors header, a config without custom-code keys. Raises ValueError."""
    d = pathlib.Path(d)
    if os.path.islink(d) or not d.is_dir():
        raise ValueError('model dir is not a real directory')
    names = sorted(os.listdir(d))
    if set(names) - set(ALLOWED_MODEL_FILES) or 'model.safetensors' not in names or 'config.json' not in names:
        raise ValueError(f'model dir must hold only {ALLOWED_MODEL_FILES} (got {names[:8]})')
    total = 0
    for n in names:
        st = os.lstat(d / n)
        if not stat.S_ISREG(st.st_mode):
            raise ValueError(f'{n} is not a regular file')
        total += st.st_size
    if total > MAX_MODEL_BYTES:
        raise ValueError(f'model dir is {total} bytes > {MAX_MODEL_BYTES}')
    check_config_dict(_json_file(d / 'config.json'), expected_model_type)
    if 'generation_config.json' in names:
        _json_file(d / 'generation_config.json')
    return read_safetensors_header(d / 'model.safetensors')


def check_safe_model_dir(d) -> None:
    """What the root-side loader accepts: a real dir holding only the allowed names, all regular files (no pickle)."""
    d = pathlib.Path(d)
    if os.path.islink(d) or not d.is_dir():
        raise ValueError(f'{d}: not a real directory')
    names = os.listdir(d)
    if set(names) - set(ALLOWED_MODEL_FILES) or 'model.safetensors' not in names:
        raise ValueError(f'{d}: refusing to load a model dir with {sorted(names)[:8]} (safetensors only)')
    for n in names:
        if not stat.S_ISREG(os.lstat(d / n).st_mode):
            raise ValueError(f'{d}/{n}: not a regular file')


def _fresh_dir(dst: pathlib.Path) -> None:
    remove_tree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(dst, 0o700)


def _seal(dst: pathlib.Path) -> None:
    for n in os.listdir(dst):
        os.chown(dst / n, 0, 0)
        os.chmod(dst / n, 0o444)
    os.chown(dst, 0, 0)
    os.chmod(dst, 0o555)


def _copy_fd(fd_in: int, dst_file: pathlib.Path) -> None:
    fd_out = os.open(str(dst_file), os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _BINARY, 0o600)
    with os.fdopen(fd_in, 'rb', closefd=False) as fi, os.fdopen(fd_out, 'wb') as fo:
        shutil.copyfileobj(fi, fo, 1 << 22)


def install_copy(src, dst, names) -> pathlib.Path:
    """Fresh root-owned read-only copy of trusted (root-side) files; O_NOFOLLOW reads, no inode shared with src."""
    src, dst = pathlib.Path(src), pathlib.Path(dst)
    _fresh_dir(dst)
    for n in names:
        fd_in = os.open(str(src / n), os.O_RDONLY | _NOFOLLOW | _NONBLOCK | _BINARY)
        try:
            if not stat.S_ISREG(os.fstat(fd_in).st_mode):
                raise ValueError(f'{n}: not a regular file')
            _copy_fd(fd_in, dst / n)
        finally:
            os.close(fd_in)
    _seal(dst)
    return dst


def open_dir_nofollow(path) -> int:
    """fd of an absolute directory path whose every component is opened with O_NOFOLLOW|O_DIRECTORY from '/'."""
    parts = pathlib.PurePosixPath(os.path.abspath(path)).parts
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for comp in parts[1:]:
            nfd = os.open(comp, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nfd
    except BaseException:
        os.close(fd)
        raise
    return fd


def install_staged(stage, dst, uid: int) -> None:
    """Copy the sanitizer's output (a dir the arm uid owns; its processes are dead) into a fresh root-owned read-only
    dst: no symlinked path component, only the allowed names, each a regular file owned by the uid, opened
    O_NOFOLLOW|O_NONBLOCK (a planted link, FIFO or root file cannot be read through it)."""
    sfd = open_dir_nofollow(stage)
    try:
        if os.fstat(sfd).st_uid != uid:
            raise ValueError('staging dir is not owned by the arm uid')
        names = sorted(os.listdir(sfd))
        if set(names) - set(ALLOWED_MODEL_FILES) or 'model.safetensors' not in names or 'config.json' not in names:
            raise ValueError(f'staged output must hold only {ALLOWED_MODEL_FILES} (got {names[:8]})')
        dst = pathlib.Path(dst)
        _fresh_dir(dst)
        total = 0
        for n in names:
            fd_in = os.open(n, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=sfd)
            try:
                st = os.fstat(fd_in)
                if not stat.S_ISREG(st.st_mode) or st.st_uid != uid:
                    raise ValueError(f'{n}: not a regular file owned by uid {uid}')
                total += st.st_size
                if total > MAX_MODEL_BYTES:
                    raise ValueError(f'staged output exceeds {MAX_MODEL_BYTES} bytes')
                _copy_fd(fd_in, dst / n)
            finally:
                os.close(fd_in)
        _seal(dst)
    finally:
        os.close(sfd)


def sanitize_to_judge(src, dst, *, uid: int, env_bin: str, token: str, sanitize_py: str,
                      expected_model_type: str | None, timeout: int = 900) -> dict:
    """Unprivileged conversion of `src` (as `uid`), then a root-owned read-only install at `dst`, validated there.
    Never raises for an unsafe artifact: returns {'ok': False, 'reason'} (G9)."""
    uid = check_uid(uid)
    tmp = ensure_uid_tmp(token, uid)
    stage = pathlib.Path(tmp) / f'stage_{uuid.uuid4().hex[:8]}'
    env = uid_env(env_bin, '', tmp, token, uid)                 # HOME = its private tmpdir; no GPU
    cmd = uid_command(f'{shlex.quote(env_bin + "/python")} -I {shlex.quote(sanitize_py)} '
                      f'{shlex.quote(str(src))} {shlex.quote(str(stage))}', env, uid)
    out = ''
    try:
        p = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True, timeout=timeout)
        out = p.stdout
    except subprocess.TimeoutExpired:
        out = ''
    finally:
        kill_uid(uid)                                           # nothing of the sanitizer outlives it
    payload = {'ok': False, 'reason': 'sanitizer produced no result'}
    if SANITIZE_SENTINEL in out:
        with contextlib.suppress(ValueError, IndexError):
            payload = json.loads(out.rsplit(SANITIZE_SENTINEL, 1)[1].strip().splitlines()[0])
    try:
        if not payload.get('ok'):
            return {'ok': False, 'reason': str(payload.get('reason'))[:500]}
        try:
            install_staged(stage, dst, uid)
            info = validate_model_dir(dst, expected_model_type)
        except (ValueError, OSError) as exc:
            remove_tree(dst)
            return {'ok': False, 'reason': f'staged output rejected: {exc}'[:500]}
    finally:
        remove_tree(stage)
    return {'ok': True, 'path': str(dst), 'tree_digest': tree_digest_files(dst), 'kind': payload.get('kind'),
            'n_tensors': info['n_tensors'], 'source_files': payload.get('source_files')}


# ------------------------------------------------------------------ homes
def setup_hf_cache(home_root: pathlib.Path, model_dir: pathlib.Path, model_rev: str) -> None:
    cache = home_root / '.cache' / 'huggingface' / 'hub' / 'models--deepseek-ai--deepseek-coder-1.3b-base'
    (cache / 'refs').mkdir(parents=True, exist_ok=True)
    snap = cache / 'snapshots' / model_rev
    snap.mkdir(parents=True, exist_ok=True)
    (cache / 'refs' / 'main').write_text(model_rev)
    for fn in MODEL_FILES:
        src, dst = model_dir / fn, snap / fn
        if src.exists() and not os.path.lexists(dst):
            dst.symlink_to(src)


def fresh_home_root(path: pathlib.Path, model_dir: pathlib.Path, model_rev: str, uid: int) -> pathlib.Path:
    """A new HOME (offline HF cache) built by root in a root-created dir, then handed to the uid."""
    remove_tree(path)
    os.mkdir(path, 0o700)
    setup_hf_cache(path, model_dir, model_rev)
    chown_tree(path, uid)
    return path


def home_root_for(path: pathlib.Path, model_dir: pathlib.Path, model_rev: str, uid: int) -> pathlib.Path:
    """The run's persistent HOME: created once; afterwards root never writes inside it (the uid controls it)."""
    if not os.path.lexists(path):
        return fresh_home_root(path, model_dir, model_rev, uid)
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != uid:
        raise ValueError(f'{path} is not a directory of uid {uid}')
    return path


def write_frozen(home: pathlib.Path, train_data_src: str, model_id: str) -> None:
    """train_data.jsonl and env.sh: root-owned 0644 in a root-owned 0755 home (M-01: the arm cannot replace them)."""
    os.chown(home, 0, 0)
    os.chmod(home, 0o755)
    for n in ('train_data.jsonl', 'env.sh'):
        remove_tree(home / n)
    shutil.copyfile(train_data_src, home / 'train_data.jsonl')
    (home / 'env.sh').write_text(env_sh_text(home, model_id), encoding='utf-8')
    for n in ('train_data.jsonl', 'env.sh'):
        os.chown(home / n, 0, 0)
        os.chmod(home / n, 0o644)


def write_home(home: pathlib.Path, foundry_dir: str, train_data_src: str, model_id: str, finetune_sh: str,
               finetune_yaml: str, overlay: dict, uid: int) -> None:
    """A fresh home: the base llm-foundry tree + the rendered finetune.sh / .yaml + overlay files, solution handed to
    the uid only after root has finished writing into it."""
    remove_tree(home)
    home.mkdir(parents=True)
    sol = home / 'solution'
    sol.mkdir()
    cp = subprocess.run(['cp', '-a', f'{foundry_dir}/.', str(sol)], capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f'cp -a base solution failed: {cp.stderr[-400:]}')
    (sol / 'finetune.sh').write_text(finetune_sh, encoding='utf-8')
    (sol / 'finetune.yaml').write_text(finetune_yaml, encoding='utf-8')
    for rel, text in (overlay or {}).items():
        if not canonical_rel(rel):
            raise ValueError(f'overlay path is not a canonical relative path: {rel!r}')
        target = sol / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not is_within(target.parent, sol) or os.path.islink(target):
            raise ValueError(f'overlay path escapes the solution: {rel!r}')
        target.write_text(text, encoding='utf-8')
    os.chmod(sol / 'finetune.sh', 0o755)
    write_frozen(home, train_data_src, model_id)
    chown_tree(sol, uid)
    os.chmod(sol, 0o700)


def snapshot_solution(home: pathlib.Path, snaps: pathlib.Path, digest: str) -> dict:
    """tar of solution/ (without score.py's three outputs) named by its tree digest; identical trees share one tar."""
    tar = snaps / f'{digest}.tar'
    reused = tar.exists()
    if not reused:
        tmp = snaps / f'.{digest}.{os.getpid()}.tmp'
        excl = [f'--exclude=solution/{d}' for d in OUTPUT_DIRS]
        subprocess.run(['tar', '-cf', str(tmp), '--anchored', *excl, '-C', str(home), 'solution'], check=True,
                       capture_output=True)
        os.chmod(tmp, 0o600)
        os.replace(tmp, tar)
    return {'path': str(tar), 'tree_digest': digest, 'sha256': sha256_file(tar), 'bytes': tar.stat().st_size,
            'reused': reused}


def extract_solution(snapshot: str, home: pathlib.Path) -> None:
    with tarfile.open(snapshot) as tf:
        tf.extractall(home, filter='tar', numeric_owner=True)   # members resolving outside home are refused


def restore_home(home: pathlib.Path, snapshot: str, expected_digest: str, train_data_src: str, model_id: str,
                 uid: int, aside: pathlib.Path) -> str:
    """B-05: the selected snapshot in a fresh home at the canonical path (absolute paths in the agent's edits keep their
    meaning); the final working tree is moved aside, root-only, never scored. Raises on a digest mismatch."""
    if os.path.lexists(home):
        os.replace(home, aside)
        chown_tree(aside, 0)
        os.chmod(aside, 0o700)
    home.mkdir(parents=True)
    extract_solution(snapshot, home)
    sol = home / 'solution'
    got = tree_digest_solution(sol)
    if got != expected_digest:
        raise ValueError(f'restored tree digest {got[:16]} != selected {str(expected_digest)[:16]}')
    write_frozen(home, train_data_src, model_id)
    chown_tree(sol, uid)
    return got


def install_reference(run_dir: pathlib.Path, reference_src: str | None, expected_model_type: str) -> dict:
    """G1: the reference lives outside the agent tree, root-owned, parent mode 700, safetensors + JSON only (root
    copies bytes and parses headers; nothing is deserialised)."""
    parent = run_dir / 'ref'
    ref_dir = parent / 'trained_model_reference'
    info = {'path': str(ref_dir)}
    try:
        private_dir(parent)
        if not ref_dir.exists():
            src = pathlib.Path(reference_src or '')
            if not (src / 'model.safetensors').is_file():
                raise ValueError(f'reference source {src} has no model.safetensors')
            stage = parent / '.stage'
            install_copy(src, stage, [n for n in ALLOWED_MODEL_FILES if (src / n).is_file()])
            validate_model_dir(stage, expected_model_type)
            os.replace(stage, ref_dir)
        validate_model_dir(ref_dir, expected_model_type)
        info['tree_digest'] = tree_digest_files(ref_dir)
    except (ValueError, OSError) as exc:
        info['error'] = str(exc)[:300]
    st = os.lstat(parent) if os.path.lexists(parent) else None
    info.update(outside_agent_tree=not is_within(ref_dir, run_dir / 'home_agent'),
                mode=oct(stat.S_IMODE(st.st_mode))[2:] if st else None, owner=st.st_uid if st else None)
    return info


def ensure_base_safetensors(run_dir, model_dir, *, judge_uid, env_bin, token, sanitize_py, expected_model_type) -> dict:
    """The base model as a root-owned safetensors copy (converted once per run dir by the sanitizer as the arm's judge
    uid), so G3's root-side loads are safetensors too."""
    dst = run_dir / 'judge' / 'base' / 'trained_model'
    if dst.exists():
        with contextlib.suppress(ValueError, OSError):
            validate_model_dir(dst, expected_model_type)
            return {'ok': True, 'path': str(dst), 'tree_digest': tree_digest_files(dst), 'cached': True}
    private_dir(run_dir / 'judge' / 'base')
    return sanitize_to_judge(model_dir, dst, uid=judge_uid, env_bin=env_bin, token=token, sanitize_py=sanitize_py,
                             expected_model_type=expected_model_type)


# ------------------------------------------------------------------ score.py seams
@contextlib.contextmanager
def capture_fds(log_path: pathlib.Path):
    """fd 1/2 -> log_path for the duration (score.py's finetune.sh output, incl. the three `time` lines)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    f = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.dup2(f, 1)
        os.dup2(f, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        for fd in (f, saved_out, saved_err):
            os.close(fd)


class UidSubprocess:
    """Injected as `score.subprocess`: score.py's check_call (./finetune.sh) runs as the arm uid. A success returns at
    once (score.py takes its timing first; the safe compare then kills the uid); failure / timeout kill the uid."""
    CalledProcessError = subprocess.CalledProcessError
    TimeoutExpired = subprocess.TimeoutExpired

    def __init__(self, env: dict, uid: int):
        self.env, self.uid = env, check_uid(uid)

    def check_call(self, args, cwd=None, timeout=None, **_):
        inner = (f'cd {shlex.quote(str(cwd))} && ' if cwd else '') + ' '.join(shlex.quote(str(a)) for a in args)
        try:
            p = subprocess.run(['bash', '-c', uid_command(inner, self.env, self.uid)], timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_uid(self.uid)
            raise
        if p.returncode != 0:
            kill_uid(self.uid)
            raise subprocess.CalledProcessError(p.returncode, args)
        return 0


def make_safe_loader(score_mod):
    auto_model = score_mod.AutoModel

    def safe_load_model(folder):
        check_safe_model_dir(folder)
        return auto_model.from_pretrained(str(folder), use_safetensors=True)
    return safe_load_model


class Judge:
    """Per-scoring context shared by the score.py seams and main()."""

    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.g8 = self.g9 = self.trained = self.original_compare = None
        self.compared = False
        self.kills = []


def make_safe_compare(judge: Judge, original):
    judge.original_compare = original

    def safe_compare(model_dir, reference_dir):
        judge.compared = True
        judge.kills.append(kill_uid(judge.uid))                 # nothing of the candidate survives into the judging
        clean_uid_leftovers(judge.uid)
        raw = pathlib.Path(model_dir)
        judge.g8 = symlink_escape(raw, judge.solution)
        res = sanitize_to_judge(raw, judge.models / f'{judge.attempt}_cand' / 'trained_model', uid=judge.uid,
                                env_bin=judge.env_bin, token=judge.token, sanitize_py=judge.sanitize_py,
                                expected_model_type=judge.model_type)
        judge.g9 = res
        if not res['ok']:
            # score.py's own path for an unusable model (FileNotFoundError -> model_not_found, no norm, no score)
            return original(model_dir=str(judge.models / 'missing'), reference_dir=reference_dir)
        judge.trained = pathlib.Path(res['path'])
        remove_tree(raw)                                        # solution/trained_model := root-owned sanitized copy
        install_copy(judge.trained, raw, sorted(os.listdir(judge.trained)))
        return original(model_dir=str(judge.trained), reference_dir=reference_dir)
    return safe_compare


def load_score_module(score_py: str, home: pathlib.Path, ref_dir: pathlib.Path):
    spec = importlib.util.spec_from_file_location('rebench_llmf_score_trusted', score_py)
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)
    score.SOLUTION_DIR = home / 'solution'
    score.DEFAULT_MODEL_DIR = score.SOLUTION_DIR / 'trained_model'
    score.DEFAULT_REFERENCE_DIR = ref_dir
    score.load_model = make_safe_loader(score)                  # root loads safetensors only (B-01)
    return score


def norm_vs(score, a, b) -> float:
    ma, mb = score.load_model(a), score.load_model(b)
    nd, _ = score.compute_norm_and_max_diff(ma, mb)
    del ma, mb
    import gc
    gc.collect()
    return float(nd)


# ------------------------------------------------------------------ guards' raw measurements
def symlink_escape(trained: pathlib.Path, solution: pathlib.Path) -> dict:
    """G8 no_symlink_escape: trained_model and everything under it are real files inside solution/."""
    out = {'trained_present': os.path.lexists(trained), 'escapes': []}
    if not out['trained_present']:
        return {**out, 'ok': True}
    if os.path.islink(trained):
        out['escapes'].append('trained_model is a symlink')
    elif not is_within(trained, solution):
        out['escapes'].append('trained_model resolves outside solution')
    else:
        for dp, dns, fns in os.walk(trained, followlinks=False):
            for name in dns + fns:
                p = pathlib.Path(dp) / name
                if os.path.islink(p):
                    out['escapes'].append(f'symlink: {p.relative_to(trained)}')
                elif not is_within(p, solution):
                    out['escapes'].append(f'escapes solution: {p.relative_to(trained)}')
    out['ok'] = not out['escapes']
    out['escapes'] = out['escapes'][:8]
    return out


def composer_work(ckpt: pathlib.Path) -> dict:
    """G5 (report only): batches from the checkpoint *name* (ep1-ba62-rank0.pt); the checkpoint is never loaded."""
    out = {'checkpoint_present': os.path.lexists(ckpt)}
    if out['checkpoint_present']:
        target = os.readlink(ckpt) if os.path.islink(ckpt) else ckpt.name
        m = re.search(r'ep(\d+)-ba(\d+)', str(target))
        if m:
            out.update(epoch=int(m.group(1)), batches=int(m.group(2)), samples=int(m.group(2)) * 16)
    return out


def phases_from_log(path: pathlib.Path) -> dict:
    reals = REAL_RE.findall(path.read_text(errors='replace')) if path.exists() else []
    if len(reals) < 3:
        return {}
    secs = [int(m) * 60 + float(s) for m, s in reals[:3]]
    return {'data_prep': secs[0], 'train': secs[1], 'convert': secs[2]}


def run_finetune(env_bin, home: pathlib.Path, gpus: str, home_root, timeout_s: int, log_path: pathlib.Path,
                 token: str, uid: int) -> dict:
    """./finetune.sh as `uid` from a clean slate (G4's D' trainings); the uid is killed afterwards whatever happens."""
    kill_uid(uid)
    clean_uid_leftovers(uid, tmp=True)
    ensure_uid_tmp(token, uid)
    cmd = uid_command(f'cd {shlex.quote(str(home / "solution"))} && ./finetune.sh',
                      uid_env(env_bin, gpus, home_root, token, uid), uid)
    t0, timed_out = time.time(), False
    try:
        with open(log_path, 'wb') as log:
            try:
                code = subprocess.run(['bash', '-c', cmd], stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s).returncode
            except subprocess.TimeoutExpired:
                code, timed_out = 124, True
    finally:
        kill_uid(uid)
        clean_uid_leftovers(uid)
    return {'ok': code == 0 and not timed_out, 'exit_code': code, 'timed_out': timed_out,
            'wall_s': time.time() - t0, 'phases': phases_from_log(log_path)}


def run_g4(job, judge: Judge, *, env_bin, run_dir, home, ref_dir, train_data_src, model_dir, snap, gpus,
           timeout_s) -> dict:
    """G4 data dependence. ref'(D') = the starting solution trained on D' once per run dir as the arm's judge uid, sanitised,
    installed root-owned read-only with its tree digest recorded (H-01). cand(D') = the scored candidate re-materialised
    in the canonical home (a tree candidate from its snapshot, a knob/overlay candidate re-rendered) with
    train_data.jsonl := D', trained as the arm uid from a clean slate and sanitised (an unsafe output is G9). Every
    comparison that uses ref'(D') is bracketed by a digest re-check (fails closed)."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import permute
    pdir = private_dir(run_dir / 'ref_permuted')
    dprime = pdir / 'train_data_dprime.jsonl'
    if not dprime.exists():
        permute.build_dprime(train_data_src, str(dprime), seed=42)
    refprime = pdir / 'trained_model'
    digest_file = pdir / 'TREE_DIGEST.json'
    cached = refprime.exists() and digest_file.exists()
    if not cached:
        hr = run_dir / 'home_dprime_ref'
        write_home(hr, job['foundry_dir'], str(dprime), job['model_id'], job['finetune_sh_base'],
                   job['finetune_yaml_base'], {}, judge.judge_uid)       # ref'(D') = the STARTING script
        hroot = fresh_home_root(run_dir / 'home_root_ref', model_dir, job['model_rev'], judge.judge_uid)
        r = run_finetune(env_bin, hr, gpus, hroot, timeout_s, run_dir / 'results' / 'g4_refprime.log',
                         judge.token, judge.judge_uid)
        if not r['ok']:
            remove_tree(hr)
            return {'computed': False, 'error': f"ref'(D') training failed: {r}"}
        res = sanitize_to_judge(hr / 'solution' / 'trained_model', refprime, uid=judge.judge_uid, env_bin=env_bin,
                                token=judge.token, sanitize_py=judge.sanitize_py, expected_model_type=judge.model_type)
        remove_tree(hr)
        if not res['ok']:
            return {'computed': False, 'error': f"ref'(D') sanitize failed: {res['reason']}"}
        digest_file.write_text(json.dumps({'tree_digest': res['tree_digest']}))
        os.chmod(digest_file, 0o600)
    expected = json.loads(digest_file.read_text())['tree_digest']

    def verify(when):
        got = tree_digest_files(refprime)
        if got != expected:
            raise ValueError(f"ref'(D') tree digest changed ({when}): {got[:16]} != {expected[:16]}")

    if job.get('g4_share'):
        cand_dprime = refprime                                  # unchanged-start smoke: cand script == start script
    else:
        try:
            if snap:                                            # a tree candidate (A-NAT): its exact scored tree
                remove_tree(home)
                home.mkdir(parents=True)
                extract_solution(snap['path'], home)
                write_frozen(home, str(dprime), job['model_id'])
                chown_tree(home / 'solution', judge.uid)
            else:
                write_home(home, job['foundry_dir'], str(dprime), job['model_id'], job['finetune_sh'],
                           job['finetune_yaml'], job.get('overlay') or {}, judge.uid)
            hroot = fresh_home_root(run_dir / 'home_root_dprime', model_dir, job['model_rev'], judge.uid)
            r = run_finetune(env_bin, home, gpus, hroot, timeout_s, run_dir / 'results' / f'{judge.attempt}_g4_cand.log',
                             judge.token, judge.uid)
            if not r['ok']:
                return {'computed': False, 'error': f"cand(D') training failed: {r}"}
            res = sanitize_to_judge(home / 'solution' / 'trained_model', judge.models / f'{judge.attempt}_dprime' / 'trained_model',
                                    uid=judge.uid, env_bin=env_bin, token=judge.token, sanitize_py=judge.sanitize_py,
                                    expected_model_type=judge.model_type)
            if not res['ok']:
                return {'computed': False, 'unsafe_artifact': True, 'error': f"cand(D') unsafe artifact: {res['reason']}"}
            cand_dprime = pathlib.Path(res['path'])
        finally:
            write_frozen(home, train_data_src, job['model_id'])  # the home holds D again
    try:
        verify('before')
        cmp = judge.original_compare(model_dir=str(cand_dprime), reference_dir=str(refprime))   # safe loader inside
        moves = {'cand_dprime_vs_cand_d': norm_vs(judge.score, cand_dprime, judge.trained),
                 'refprime_vs_ref_d': norm_vs(judge.score, refprime, ref_dir)}
        verify('after')
    except Exception as exc:                                    # noqa: BLE001 - fails closed
        return {'computed': False, 'error': repr(exc)[:300]}
    return {'computed': True, 'score_py_valid': float(cmp['norm_diff']) <= float(judge.score.MAX_NORM_DIFF),
            'norm_cand_dprime_vs_refprime': float(cmp['norm_diff']), **moves, 'refprime_cached': bool(cached),
            'refprime_tree_digest': expected, 'refprime_owner': owner_mode(refprime), 'g4_shared': bool(job.get('g4_share'))}


# ------------------------------------------------------------------ main
def uid_report() -> dict:
    """Record that the ENV-A uids are unused by anyone else (no passwd / group entry, no live process)."""
    rows = {}
    for u in OUR_UIDS:
        pw = subprocess.run(['getent', 'passwd', str(u)], capture_output=True, text=True).stdout.strip()
        gr = subprocess.run(['getent', 'group', str(u)], capture_output=True, text=True).stdout.strip()
        rows[str(u)] = {'passwd': pw or None, 'group': gr or None, 'live_processes': len(uid_pids(u))}
    return rows


def _token(prefix: str) -> str:
    return f'{prefix}-{uuid.uuid4().hex[:12]}'


def prepare(job: dict) -> dict:
    """CPU-only, before the arm's GPU work: the reference (safetensors only) and the base-model safetensors copy."""
    run_dir = private_dir(job['run_dir'], mode=0o711)
    env_bin, model_dir = job['env_dir'] + '/bin', pathlib.Path(job['model_dir'])
    token = _token('isx2p')
    judge_uid = judge_for(job['uid'], job['judge_uid'])
    _ACTIVE['uids'].add(judge_uid)
    private_dir(run_dir / 'judge')
    model_type = _json_file(model_dir / 'config.json')['model_type']
    ref = install_reference(run_dir, job.get('reference_src'), model_type)
    base = ensure_base_safetensors(run_dir, model_dir, judge_uid=judge_uid, env_bin=env_bin, token=token,
                                   sanitize_py=job['sanitize_py'], expected_model_type=model_type)
    rvb = None
    if base.get('ok') and not ref.get('error'):                 # G3's constant term, off the GPU lease
        score = load_score_module(job['task_dir'] + '/assets/score.py', run_dir / 'home_agent', pathlib.Path(ref['path']))
        rvb = ref_vs_base(run_dir, score, pathlib.Path(ref['path']), ref, base)
    return {'mode': 'prepare', 'reference': ref, 'base': {k: base.get(k) for k in ('ok', 'tree_digest', 'cached', 'reason', 'n_tensors')},
            'ref_vs_base': rvb, 'uids': uid_report(), 'model_type': model_type, 'judge_uid': judge_uid}


def main(job: dict) -> dict:
    if job.get('mode') == 'prepare':
        return prepare(job)
    run_dir = private_dir(job['run_dir'], mode=0o711)
    home = run_dir / 'home_agent'
    env_bin, model_dir = job['env_dir'] + '/bin', pathlib.Path(job['model_dir'])
    score_py = job['task_dir'] + '/assets/score.py'
    train_data_src = job['task_dir'] + '/assets/train_data.jsonl'
    gpus, uid = job.get('gpus', '0,1,2,3'), check_uid(job['uid'])
    judge_uid = judge_for(uid, job['judge_uid'])
    score_timeout = int(job.get('score_timeout', 480))
    clean_slate = bool(job.get('clean_slate'))
    token = _token('isx2')
    _ACTIVE['uids'].update({uid, judge_uid})                  # this arm's pair only: never another arm's
    utc_start, t0 = datetime.datetime.now(datetime.timezone.utc).isoformat(), time.time()

    kills = [kill_uid(uid)]                                     # start from a clean uid
    clean_uid_leftovers(uid, tmp=clean_slate)
    for d in ('judge', 'judge/models', 'snapshots', 'results'):
        private_dir(run_dir / d)
    ensure_uid_tmp(token, uid)
    home_root = (fresh_home_root(run_dir / 'home_root_locked', model_dir, job['model_rev'], uid) if clean_slate
                 else home_root_for(run_dir / 'home_root', model_dir, job['model_rev'], uid))
    model_type = _json_file(model_dir / 'config.json')['model_type']
    reference = install_reference(run_dir, job.get('reference_src'), model_type)
    ref_dir = pathlib.Path(reference['path'])
    base = ensure_base_safetensors(run_dir, model_dir, judge_uid=judge_uid, env_bin=env_bin, token=token,
                                   sanitize_py=job['sanitize_py'], expected_model_type=model_type)

    tree_candidate = bool(job.get('reuse_home') or job.get('restore_snapshot'))
    if job.get('restore_snapshot'):
        restore_home(home, job['restore_snapshot'], job['expected_digest'], train_data_src, job['model_id'], uid,
                     run_dir / f'home_agent_final_{job["attempt_id"]}')
    elif job.get('reuse_home') and os.path.isdir(home / 'solution') and not os.path.islink(home / 'solution'):
        write_frozen(home, train_data_src, job['model_id'])     # re-assert root-owned env.sh / train_data (M-01)
        chown_tree(home / 'solution', uid)
    else:
        write_home(home, job['foundry_dir'], train_data_src, job['model_id'], job['finetune_sh'], job['finetune_yaml'],
                   job.get('overlay') or {}, uid)
    sol = home / 'solution'
    tree_digest = tree_digest_solution(sol)
    if job.get('expected_digest') and tree_digest != job['expected_digest']:
        raise ValueError(f'candidate tree {tree_digest[:16]} != expected {job["expected_digest"][:16]}')
    snap = snapshot_solution(home, run_dir / 'snapshots', tree_digest) if tree_candidate else None

    frozen_before = frozen_hashes(score_py, home, model_dir)
    env_sh_state = owner_mode(home / 'env.sh')
    score = load_score_module(score_py, home, ref_dir)
    judge = Judge(uid=uid, judge_uid=judge_uid, env_bin=env_bin, token=token, solution=sol, models=run_dir / 'judge' / 'models',
                  attempt=job['attempt_id'], model_type=model_type, sanitize_py=job['sanitize_py'], score=score)
    score.subprocess = UidSubprocess(uid_env(env_bin, gpus, home_root, token, uid), uid)
    score.compare_models = make_safe_compare(judge, score.compare_models)
    log_main = run_dir / 'results' / f'{job["attempt_id"]}_finetune.log'
    try:
        with capture_fds(log_main):
            judge_out = score.score(strip_score_information=False, reference_dir=str(ref_dir), timeout=score_timeout)
    finally:
        kills.append(kill_uid(uid))
        clean_uid_leftovers(uid)
    judge_out = {k: (None if isinstance(v, float) and v != v else v) for k, v in judge_out.items()}
    judge_out['message'] = {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in (judge_out.get('message') or {}).items()}
    frozen_after = frozen_hashes(score_py, home, model_dir)
    phases = {**phases_from_log(log_main), 'total': judge_out['message'].get('execution_time')}

    trained_raw = sol / 'trained_model'
    if not judge.compared:                                      # finetune failed: nothing was sanitised or loaded
        judge.g8 = symlink_escape(trained_raw, sol)
        judge.g9 = {'ok': True, 'skipped': 'no scored output (finetune did not complete)'}
    valid_judge = isinstance(judge_out.get('score'), (int, float)) and judge_out['score'] == judge_out['score']
    g3 = None
    if valid_judge and judge.trained and base.get('ok'):
        try:
            g3 = {'trained_vs_base': norm_vs(score, judge.trained, base['path']),
                  'ref_vs_base': ref_vs_base(run_dir, score, ref_dir, reference, base)}
        except Exception as exc:                                # noqa: BLE001
            g3 = {'error': repr(exc)[:300]}
    g5 = composer_work(sol / 'ftout' / 'latest-rank0.pt')
    g5['param_shapes_match'] = None
    g4 = None
    if job.get('run_g4') and valid_judge and judge.trained:
        g4 = run_g4(job, judge, env_bin=env_bin, run_dir=run_dir, home=home, ref_dir=ref_dir,
                    train_data_src=train_data_src, model_dir=model_dir, snap=snap, gpus=gpus, timeout_s=score_timeout)
        sol = home / 'solution'
        trained_raw = sol / 'trained_model'
    for d in judge.models.glob(f'{judge.attempt}_*'):
        remove_tree(d)                                          # installed copies are per attempt
    if (os.path.isdir(trained_raw) and not os.path.islink(trained_raw) and os.lstat(trained_raw).st_uid == 0):
        for n in os.listdir(trained_raw):                       # our flat sealed copy: hand it back to the arm
            if not os.path.islink(trained_raw / n):
                os.chmod(trained_raw / n, 0o644)
        os.chmod(trained_raw, 0o755)
        chown_tree(trained_raw, uid)
    wall = time.time() - t0
    world = len(gpus.split(','))
    result = {
        'node': job.get('node'), 'gpus': gpus, 'uid': uid, 'judge': judge_out, 'exit_code': 0,
        'timed_out': 'TimeoutExpired' in str(judge_out['message'].get('error', '')),
        'outputs': {'trained_model': judge.trained is not None},
        'reference': reference, 'base': {k: base.get(k) for k in ('ok', 'tree_digest', 'cached', 'reason')},
        'frozen_before': frozen_before, 'frozen_after': frozen_after, 'env_sh': env_sh_state,
        'g3': g3, 'g4': g4, 'g5': g5, 'g8': judge.g8, 'g9': judge.g9, 'phases': phases, 'snapshot': snap,
        'tree_digest': tree_digest, 'clean_slate': clean_slate, 'kills': kills + judge.kills,
        'judge_uid': judge_uid, 'remaining_uid_processes': {str(u): len(uid_pids(u)) for u in sorted({uid, judge_uid})},
        'utc_start': utc_start, 'utc_end': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'wall_s': wall, 'cost': {'gpu_seconds': round(world * wall, 1), 'wall_seconds': round(wall, 1)},
        'env_versions': env_versions(),
    }
    append_scores_jsonl(run_dir, job, result)
    return result


def ref_vs_base(run_dir: pathlib.Path, score, ref_dir, reference: dict, base: dict) -> float:
    """||ref - base|| once per (reference, base) digest pair (cached in the judge dir)."""
    key = {'ref': reference.get('tree_digest'), 'base': base.get('tree_digest')}
    cache = run_dir / 'judge' / 'ref_vs_base.json'
    with contextlib.suppress(OSError, ValueError, KeyError):
        row = json.loads(cache.read_text())
        if row['key'] == key and None not in key.values():
            return float(row['value'])
    value = norm_vs(score, ref_dir, base['path'])
    cache.write_text(json.dumps({'key': key, 'value': value}))
    os.chmod(cache, 0o600)
    return value


def append_scores_jsonl(run_dir: pathlib.Path, job: dict, result: dict) -> None:
    m = result.get('judge') or {}
    msg = m.get('message') or {}
    row = {'utc': result['utc_end'], 'attempt_id': job['attempt_id'], 'purpose': job.get('purpose'),
           'execution_time': msg.get('execution_time'), 'norm_diff': msg.get('norm_diff'),
           'max_scalar_diff': msg.get('max_scalar_diff'), 'score': m.get('score'), 'tree_digest': result.get('tree_digest'),
           'snapshot': (result.get('snapshot') or {}).get('path'), 'g9_ok': (result.get('g9') or {}).get('ok')}
    with open(run_dir / 'scores.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    os.chmod(run_dir / 'scores.jsonl', 0o600)


def env_versions() -> dict:
    try:
        import torch
        import transformers
        return {'python': sys.version.split()[0], 'torch': torch.__version__, 'transformers': transformers.__version__}
    except Exception:                                           # noqa: BLE001
        return {'python': sys.version.split()[0]}


def _cleanup_active(*_args) -> None:
    """Kill every uid this job used, remove their /dev/shm leftovers and this job's tmpdirs (exit or SIGTERM)."""
    for u in sorted(_ACTIVE['uids']):
        with contextlib.suppress(Exception):
            kill_uid(u)
            clean_uid_leftovers(u)
    for d in sorted(_ACTIVE['tmpdirs']):
        with contextlib.suppress(Exception):
            remove_tree(d)


if __name__ == '__main__':
    job = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))

    def _on_term(signum, frame):
        _cleanup_active()
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, _on_term)
    try:
        res = {'ok': True, 'result': main(job)}
    except BaseException as exc:                                # noqa: BLE001
        import traceback
        res = {'ok': False, 'error': repr(exc), 'traceback': traceback.format_exc()}
    finally:
        _cleanup_active()
    out = job.get('out')
    if out:
        pathlib.Path(out).write_text(json.dumps(res, default=str), encoding='utf-8')
    print(SENTINEL + json.dumps(res, default=str)[:2000])
