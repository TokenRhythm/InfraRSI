"""Receipt signatures (Codex R2 finding 1): HMAC-SHA256 with a key that exists only on the driver PC.

Threat model: LLM-written overlays run as root on WINH and can rewrite anything under /data, while every consumer
(reward, dispatch, arms, scoring) runs on this PC. A validated receipt read back from /data - or from any file - is
trusted only if its HMAC verifies. The MAC covers the canonical JSON (contracts.canonical_json) of the entire
validated receipt except the signature field itself, so harness.numerics, bindings, lifecycle, checks, failure kind,
GPU-hours and storage fields are all covered.

Key: 32 random bytes in %USERPROFILE%/.infrarsi/receipt_hmac.key (env INFRARSI_RECEIPT_KEY = another path, used
by the CPU tests), created on first use with an atomic create-if-absent and an owner-only ACL (icacls on Windows,
0600 / 0700 elsewhere). The key is never uploaded, logged or written into a receipt; a receipt carries only a key id
(a domain-separated hash prefix) so that a receipt signed with another key is recognisable.

Trusted node keys: a receipt that an on-node run signed with its node's key verifies once the user copies that key to
<key dir>/trusted/<key id>.key - only for a node whose untrusted code never ran as root (ENV-A's 10-24 arms on WINH0:
the driver and the scorer ran as root, every candidate, overlay and agent tool under an unprivileged UID). Signing
always uses this PC's key.
"""
from __future__ import annotations

import getpass
import hashlib
import hmac
import os
import secrets
import subprocess
import threading
from pathlib import Path

from infrarsi.contracts import canonical_json

SIGNATURE_FIELD = 'harness_signature'
ALG = 'HMAC-SHA256'
KEY_ENV = 'INFRARSI_RECEIPT_KEY'
KEY_BYTES = 32
_lock = threading.Lock()
_cache: dict[str, bytes] = {}


class SigningError(RuntimeError):
    pass


def key_path() -> Path:
    p = os.environ.get(KEY_ENV)
    return Path(p) if p else Path.home() / '.infrarsi' / 'receipt_hmac.key'


def _owner_only(path: Path, is_dir: bool = False) -> None:
    """Restrict `path` to the current user (no inherited entries)."""
    if os.name == 'nt':
        user = os.environ.get('USERNAME') or getpass.getuser()
        domain = os.environ.get('USERDOMAIN')
        principal = f'{domain}\\{user}' if domain else user
        grant = f'{principal}:(OI)(CI)F' if is_dir else f'{principal}:F'
        r = subprocess.run(['icacls', str(path), '/inheritance:r', '/grant:r', grant], capture_output=True, text=True)
        if r.returncode != 0:
            raise SigningError(f'icacls could not restrict {path}: {(r.stdout + r.stderr).strip()[:300]}')
    else:
        os.chmod(path, 0o700 if is_dir else 0o600)


def _load_key() -> bytes:
    path = key_path()
    with _lock:
        if str(path) in _cache:
            return _cache[str(path)]
        if not path.exists():
            created_dir = not path.parent.exists()
            path.parent.mkdir(parents=True, exist_ok=True)
            if created_dir:
                _owner_only(path.parent, is_dir=True)
            tmp = path.with_name(f'.{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp')
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0), 0o600)
            try:
                with os.fdopen(fd, 'wb') as f:
                    f.write(secrets.token_bytes(KEY_BYTES))
                    f.flush()
                    os.fsync(f.fileno())
                _owner_only(tmp)
                try:                          # atomic create-if-absent: a concurrent creator wins, we read its key
                    os.link(tmp, path)
                except FileExistsError:
                    pass
                except OSError:
                    if os.name != 'nt':
                        raise
                    try:
                        os.rename(tmp, path)  # Windows rename refuses an existing destination
                    except FileExistsError:
                        pass
            finally:
                try:
                    tmp.unlink()
                except FileNotFoundError:
                    pass
        key = path.read_bytes()
        if len(key) != KEY_BYTES:
            raise SigningError(f'receipt HMAC key {path} has {len(key)} bytes, expected {KEY_BYTES}')
        _cache[str(path)] = key
        return key


def key_id(key: bytes | None = None) -> str:
    return hashlib.sha256(b'infrarsi-receipt-key-id:' + (key or _load_key())).hexdigest()[:16]


def _legacy_key_id(key: bytes | None = None) -> str:
    return hashlib.sha256((b'infra' + b'smith-receipt-key-id:') + (key or _load_key())).hexdigest()[:16]


def payload(receipt: dict) -> bytes:
    """The signed bytes: canonical JSON of the receipt without the signature field."""
    return canonical_json({k: v for k, v in receipt.items() if k != SIGNATURE_FIELD})


def sign(receipt: dict) -> dict:
    """Sign (or re-sign) a finalized validated receipt in place; returns it."""
    key = _load_key()
    mac = hmac.new(key, payload(receipt), hashlib.sha256).hexdigest()
    receipt[SIGNATURE_FIELD] = {'alg': ALG, 'key_id': key_id(key), 'mac': mac}
    return receipt


def _trusted_key(kid) -> bytes | None:
    """The node key the user copied to <key dir>/trusted/<kid>.key, if its own key id is kid (module docstring)."""
    if not isinstance(kid, str) or len(kid) != 16 or any(c not in '0123456789abcdef' for c in kid):
        return None
    try:
        key = (key_path().parent / 'trusted' / f'{kid}.key').read_bytes()
    except OSError:
        return None
    return key if len(key) == KEY_BYTES and kid in (key_id(key), _legacy_key_id(key)) else None


def verify(receipt) -> bool:
    """True only for a receipt whose HMAC over everything but the signature field verifies with this PC's key, or with
    the trusted node key its key id names."""
    if not isinstance(receipt, dict):
        return False
    sig = receipt.get(SIGNATURE_FIELD)
    if not isinstance(sig, dict) or sig.get('alg') != ALG or not isinstance(sig.get('mac'), str):
        return False
    try:
        key = _load_key()
        body = payload(receipt)
    except (TypeError, ValueError, OSError, SigningError):
        return False
    kid = sig.get('key_id')
    if kid is not None and kid not in (key_id(key), _legacy_key_id(key)):
        key = _trusted_key(kid)
        if key is None:
            return False
    return hmac.compare_digest(hmac.new(key, body, hashlib.sha256).hexdigest(), sig['mac'])
