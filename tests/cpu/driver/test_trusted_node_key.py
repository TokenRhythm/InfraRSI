"""DF19 (2026-10-25): ENV-A's 10-24 arms ran on WINH0 and signed their receipts with WINH0's key, so the PC failed
them and the phase-E labels stayed pending. A node key the user copies to <key dir>/trusted/<key id>.key verifies
the receipts that name it; nothing else changes."""
import hashlib
import hmac
import secrets

import driver_helpers  # noqa: F401  (points the receipt key at the session's test key)
from infrarsi.harness import signing


def node_signed(receipt, key):
    receipt = dict(receipt)
    receipt[signing.SIGNATURE_FIELD] = {'alg': signing.ALG, 'key_id': signing.key_id(key),
                                        'mac': hmac.new(key, signing.payload(receipt), hashlib.sha256).hexdigest()}
    return receipt


def test_a_trusted_node_key_verifies_its_receipts_only(tmp_path, monkeypatch):
    monkeypatch.setenv(signing.KEY_ENV, str(tmp_path / 'keys' / 'receipt_hmac.key'))
    own = signing.sign({'attempt_id': 'pc', 'x': 1})
    node = secrets.token_bytes(signing.KEY_BYTES)
    r = node_signed({'attempt_id': 'n', 'x': 2}, node)
    assert signing.verify(own) and not signing.verify(r)                    # not trusted yet
    trusted = tmp_path / 'keys' / 'trusted'
    trusted.mkdir()
    (trusted / f'{signing.key_id(node)}.key').write_bytes(node)
    assert signing.verify(r) and signing.verify(own)
    assert not signing.verify({**r, 'x': 3})                                 # the MAC still covers the body
    other = secrets.token_bytes(signing.KEY_BYTES)
    assert not signing.verify(node_signed({'attempt_id': 'o'}, other))      # an unknown key id
    (trusted / f'{signing.key_id(other)}.key').write_bytes(node)            # a file not holding the key it names
    assert not signing.verify(node_signed({'attempt_id': 'o'}, other))
    bad = node_signed({'attempt_id': 'b'}, node)
    bad[signing.SIGNATURE_FIELD]['key_id'] = '../receipt_hmac'              # a key id that is no key id
    assert not signing.verify(bad)
    assert signing.sign({'attempt_id': 's'})[signing.SIGNATURE_FIELD]['key_id'] == signing.key_id()   # signs with the PC key
