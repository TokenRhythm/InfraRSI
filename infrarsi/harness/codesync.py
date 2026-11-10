"""Content-addressed code trees on WINH (INTERFACES section 3.2).

code_digest = sha256(canonical_json(sorted [[relpath, file_sha256], ...])) over every *.py/*.json/*.yaml under
infrarsi/infrarsi/ (relpath relative to that package directory, '/' separators), excluding runtime/overlays/
and __pycache__. The tree is uploaded once per digest to /data/infrarsi/code/<digest>/infrarsi/...; the
default overlay files travel with it but are digested separately (MANIFEST.json overlay_defaults_digest).
Each file is read once into a snapshot that feeds both the manifest and the tar, so edits made by other builders
while a sync runs cannot produce a tree that disagrees with its digest (the node re-verifies every file anyway).
A tree that already exists is never trusted from its MANIFEST.json alone: before every reuse the node re-hashes each
file (remote_side.verify_code) and the sync refuses a tree with a changed, missing or extra file (including any
__pycache__ entry or symlink) - such a tree is reported, never repaired or overwritten in place.

Cache identity (Codex R2 finding 4a) is (code_digest, overlay_defaults_digest): the directory is code/<code_digest> or,
when the tree carries default overlays, code/<code_digest>-ov<first 16 hex of overlay_defaults_digest>; any manifest
difference from this release, default overlays included, refuses the cached tree (no warning-only path).
effective_overlay_digest() gives the L1 overlay set the runtime will run (runtime/overlay_loader: the six hook
modules' default files, replaced by the request's files of the same name), which run_screen pins in the request.
"""
from __future__ import annotations

import hashlib
import io
import tarfile
import time
from pathlib import Path

from infrarsi.contracts import CODE_SUFFIXES, OVERLAY_PREFIX, RemotePaths, overlay_digest, tree_digests, utc_now
from infrarsi.runtime.overlay_loader import HOOK_MODULES

PACKAGE_DIR = Path(__file__).resolve().parents[1]           # .../infrarsi/infrarsi
SUFFIXES, OVERLAYS = CODE_SUFFIXES, OVERLAY_PREFIX


def snapshot(package_dir: Path = PACKAGE_DIR) -> dict[str, bytes]:
    out = {}
    for p in sorted(Path(package_dir).rglob('*')):
        if p.is_file() and p.suffix in SUFFIXES and '__pycache__' not in p.parts:
            out[p.relative_to(package_dir).as_posix()] = p.read_bytes()
    return out


def tree_manifest(package_dir: Path = PACKAGE_DIR, snap: dict[str, bytes] | None = None) -> dict:
    snap = snapshot(package_dir) if snap is None else snap
    files = {k: hashlib.sha256(v).hexdigest() for k, v in sorted(snap.items())}
    code, overlays = tree_digests(files)
    return {'schema': 'infrarsi-code-manifest-v1', 'code_digest': code, 'overlay_defaults_digest': overlays,
            'files': files, 'created_utc': utc_now()}


def code_digest(package_dir: Path = PACKAGE_DIR) -> str:
    return tree_manifest(package_dir)['code_digest']


def build_tar(manifest: dict, package_dir: Path = PACKAGE_DIR, snap: dict[str, bytes] | None = None) -> bytes:
    """Deterministic tar (sorted members, fixed mtime/owner) with members infrarsi/<relpath>."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w', format=tarfile.PAX_FORMAT) as tf:
        for rel in sorted(manifest['files']):
            data = snap[rel] if snap is not None else (Path(package_dir) / rel).read_bytes()
            if hashlib.sha256(data).hexdigest() != manifest['files'][rel]:
                raise RuntimeError(f'{rel} changed while building the code tree; retry the sync')
            info = tarfile.TarInfo('infrarsi/' + rel)
            info.size, info.mtime, info.mode, info.uid, info.gid = len(data), 0, 0o644, 0, 0
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def code_dir_name(manifest: dict) -> str:
    ov = manifest.get('overlay_defaults_digest')
    return manifest['code_digest'] if not ov else f'{manifest["code_digest"]}-ov{ov[:16]}'


def overlay_defaults(files: dict[str, str]) -> dict[str, str]:
    """{'<hook module>.py': sha256} of the tree's default L1 overlay files (runtime/overlay_loader.HOOK_MODULES)."""
    out = {}
    for m in HOOK_MODULES:
        sha = files.get(f'{OVERLAYS}{m}.py')
        if not sha:
            raise RuntimeError(f'code tree has no default overlay {OVERLAYS}{m}.py')
        out[f'{m}.py'] = sha
    return out


def effective_overlay_files(defaults: dict[str, str], request_files: dict[str, str] | None) -> dict[str, str]:
    """The runtime's effective overlay set: the default hook files, each replaced by a request file of the same name."""
    out = dict(defaults)
    for name, sha in (request_files or {}).items():
        if name not in defaults:
            raise ValueError(f'overlay file {name!r} is not one of the L1 hook modules {sorted(defaults)}')
        out[name] = sha
    return out


def effective_overlay_digest(defaults: dict[str, str], request_files: dict[str, str] | None) -> str:
    """contracts.overlay_digest of the effective set (= runtime Overlays.digest of the bytes that will run)."""
    return overlay_digest(effective_overlay_files(defaults, request_files))


def ensure_code(remote, node, package_dir: Path = PACKAGE_DIR) -> dict:
    """Upload this release unless its cache directory already holds exactly it (every file re-hashed); returns
    code_root, code_digest and the default overlay files ({'<module>.py': sha256})."""
    snap = snapshot(package_dir)
    man = tree_manifest(package_dir, snap)
    root = f'{RemotePaths.code}/{code_dir_name(man)}'
    defaults = overlay_defaults(man['files'])
    base = {'code_root': root, 'code_digest': man['code_digest'], 'overlay_defaults_digest': man['overlay_defaults_digest'],
            'overlay_defaults': defaults}
    st = remote.call(node, 'code_status', code_dir=root)
    if st['exists']:
        have = (st.get('manifest') or {}).get('files') or {}
        if not have:
            raise RuntimeError(f'code tree {root} has no readable MANIFEST.json; refusing to reuse it')
        diff = sorted(k for k in set(have) | set(man['files']) if have.get(k) != man['files'].get(k))
        if diff:                      # a foreign or corrupted tree (default overlays included): never reuse, never overwrite
            raise RuntimeError(f'code tree {root} differs from this release in {diff[:5]}; refusing to reuse it')
        ver = remote.call(node, 'verify_code', code_dir=root, files=have, timeout=300)
        if not ver.get('ok'):
            raise RuntimeError(f'cached code tree {root} failed re-hashing (changed {ver.get("mismatched")}, missing '
                               f'{ver.get("missing")}, extra {ver.get("extra")}); refusing to launch from it')
        return dict(base, uploaded=False, rehashed_files=ver.get('files'))
    staging = f'{RemotePaths.code}/.incoming'
    remote.call(node, 'makedirs', paths=[staging])
    tar_path = f'{staging}/{man["code_digest"]}.{int(time.time() * 1000)}.tar'
    remote.put_bytes(node, build_tar(man, package_dir, snap), tar_path)
    res = remote.call(node, 'install_code', tar_path=tar_path, code_dir=root, manifest=man, timeout=300)
    return dict(base, uploaded=bool(res.get('installed')))
