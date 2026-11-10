import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Receipts are HMAC-signed with a key that lives only on the driver PC (harness/signing.py) and every launch is
# recorded in a local ledger: CPU tests use throwaway session files for both, never the PC's real key or
# execution/ledger/launches.jsonl (same variables and rule as tests/cpu/driver/driver_helpers.py).
_SESSION = Path(tempfile.mkdtemp(prefix='isagent'))
atexit.register(shutil.rmtree, _SESSION, ignore_errors=True)
os.environ.setdefault('INFRARSI_RECEIPT_KEY', str(_SESSION / 'receipt_hmac.key'))
os.environ.setdefault('INFRARSI_LAUNCH_LEDGER', str(_SESSION / 'launches.jsonl'))
os.environ.setdefault('INFRARSI_HOME', str(_SESSION / 'home'))   # ledger, noise rows, experience write-back

from infrarsi.actions import patch_guard  # noqa: E402
from infrarsi.contracts import TaskSpec  # noqa: E402

from agent_helpers import FIXTURES, PRISTINE, DENSE_8B, DENSE_4B, MOE_30B, FR_DENSE, FR_MOE  # noqa: E402,F401


@pytest.fixture
def pristine_overlays(tmp_path, monkeypatch):
    """A pristine overlay set (the runtime's no-op hook points) for patch-guard and L1 tests."""
    d = tmp_path / 'pristine_overlays'
    d.mkdir()
    for name, text in PRISTINE.items():
        (d / name).write_text(text, encoding='utf-8')
    original = patch_guard.base_overlays
    monkeypatch.setattr(patch_guard, 'base_overlays', lambda base_dir=None: original(base_dir or d))
    return d


@pytest.fixture(scope='session')
def tasks():
    return {k: TaskSpec.load(k) for k in ('S1', 'S2', 'D1', 'D2', 'D4', 'D5', 'D6', 'H1', 'H3')}


@pytest.fixture(scope='session')
def a0_package(tmp_path_factory):
    from infrarsi.rsi import build_a0
    return build_a0(tmp_path_factory.mktemp('pkg') / 'A0')
