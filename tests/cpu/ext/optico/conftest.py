"""ENV-B (ext/optico) CPU tests: receipts are signed with a throwaway key, never the PC's real one."""
import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

_SESSION = Path(tempfile.mkdtemp(prefix='isoptico'))
atexit.register(shutil.rmtree, _SESSION, ignore_errors=True)
os.environ.setdefault('INFRARSI_RECEIPT_KEY', str(_SESSION / 'receipt_hmac.key'))
os.environ.setdefault('INFRARSI_LAUNCH_LEDGER', str(_SESSION / 'launches.jsonl'))
