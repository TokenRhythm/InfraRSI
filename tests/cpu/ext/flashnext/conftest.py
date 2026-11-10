"""ext/flashnext CPU tests (X5 E1 spike).

The bundle tests need, on this PC: the frozen formal-54 builder tree (reports/rsi_infra_execution_20261018/...), the local
LF source mirror (deliverables/WINH_ACCEL_CODEBASE_20261016/...) and the registered bundles (E:/rsi/f54v1/final/bundles,
or FLASHNEXT_REGISTERED_BUNDLES). They skip when those are absent. No GPU, no network, no remote access.
"""
import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Receipts are signed with a throwaway key, never the PC's real one (as the other ext suites).
_SESSION = Path(tempfile.mkdtemp(prefix='isflashnext'))
atexit.register(shutil.rmtree, _SESSION, ignore_errors=True)
os.environ.setdefault('INFRARSI_RECEIPT_KEY', str(_SESSION / 'receipt_hmac.key'))
os.environ.setdefault('INFRARSI_LAUNCH_LEDGER', str(_SESSION / 'launches.jsonl'))


@pytest.fixture(scope='session')
def frozen():
    from infrarsi.ext.flashnext import variant
    if not variant.FROZEN_RUNTIME.exists() or not variant.SOURCE_MIRROR.exists():
        pytest.skip('frozen formal-54 builder tree or local source mirror not on this machine')
    check = variant.frozen_check()
    assert check['ok'], check['problems']
    return variant


@pytest.fixture(scope='session')
def registered(frozen):
    if not frozen.REGISTERED_BUNDLES.exists():
        pytest.skip('registered formal-54 bundles not on this machine')
    return frozen.REGISTERED_BUNDLES


@pytest.fixture(scope='session')
def custom_rt(frozen, tmp_path_factory):
    """One custom runtime for the session (short path: bundle names come close to MAX_PATH on Windows)."""
    return frozen.custom_runtime(tmp_path_factory.mktemp('rt'))
