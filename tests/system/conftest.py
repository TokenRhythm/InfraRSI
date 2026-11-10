"""System tests of InfraRSI (2026-10-25): user journeys through the CLI, fault injection on a stand-in job, component
contracts (manifest, signals, verifier, apply, LLM endpoint, experience, report), search oracles, and POSIX / GPU legs.

    .venv/Scripts/python -m pytest tests/system -q                 # the CPU set (seconds to a few minutes)
    INFRARSI_SYSTEST_SLOW=1 ... -m slow                          # overhead and packaging checks (minutes)
    (on a Linux node) python -m pytest tests/system -q             # the posix-marked tests run there

Every test gets its own INFRARSI_HOME (ledger, experience write-back) and the receipts are signed with a throwaway
key, never the PC's real one. Nothing here touches the main tree's own .infrarsi directories.
"""
import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

_SESSION = Path(tempfile.mkdtemp(prefix='issys'))
atexit.register(shutil.rmtree, _SESSION, ignore_errors=True)
os.environ.setdefault('INFRARSI_RECEIPT_KEY', str(_SESSION / 'receipt_hmac.key'))
os.environ.setdefault('INFRARSI_LAUNCH_LEDGER', str(_SESSION / 'launches.jsonl'))
os.environ.setdefault('INFRARSI_HOME', str(_SESSION / 'home'))
os.environ.setdefault('TOY_TIME_SCALE', '0.004')
os.environ.setdefault('FAULT_STEP_S', '0.05')

SLOW = bool(os.environ.get('INFRARSI_SYSTEST_SLOW'))
POSIX = os.name != 'nt'


def pytest_configure(config):
    config.addinivalue_line('markers', 'posix: POSIX process semantics (sessions, signals, /proc): run on a Linux node')
    config.addinivalue_line('markers', 'slow: minutes; needs INFRARSI_SYSTEST_SLOW=1')
    config.addinivalue_line('markers', 'gpu: needs torch with a free GPU; needs INFRARSI_SYSTEST_GPU=1')
    config.addinivalue_line('markers', 'packaging: builds a fresh venv; needs INFRARSI_SYSTEST_PACKAGING=1')


def pytest_collection_modifyitems(config, items):
    for item in items:
        if 'posix' in item.keywords and not POSIX:
            item.add_marker(pytest.mark.skip(reason='POSIX only (run on a Linux node)'))
        if 'slow' in item.keywords and not SLOW:
            item.add_marker(pytest.mark.skip(reason='slow: set INFRARSI_SYSTEST_SLOW=1'))
        if 'gpu' in item.keywords and not os.environ.get('INFRARSI_SYSTEST_GPU'):
            item.add_marker(pytest.mark.skip(reason='gpu: set INFRARSI_SYSTEST_GPU=1 on a node with a free GPU'))
        if 'packaging' in item.keywords and not os.environ.get('INFRARSI_SYSTEST_PACKAGING'):
            item.add_marker(pytest.mark.skip(reason='packaging: set INFRARSI_SYSTEST_PACKAGING=1'))


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A private INFRARSI_HOME for the test: an empty ledger, no written-back experience, no package."""
    h = tmp_path / 'home'
    h.mkdir()
    monkeypatch.setenv('INFRARSI_HOME', str(h))
    return h


@pytest.fixture
def toy(tmp_path, home):
    """A fresh copy of examples/toy_cpu with 12 updates (window 5..12) and a checkpoint at update 6."""
    from _support.toy import make_toy
    return make_toy(tmp_path / 'toy')


@pytest.fixture
def fault_project(tmp_path, home):
    """A fresh copy of the stand-in fault job (its manifest: knobs of all five target kinds)."""
    from _support.faultjob import make_project
    return make_project(tmp_path / 'fault')
