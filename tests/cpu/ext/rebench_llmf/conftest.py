"""ENV-A (ext/rebench_llmf) CPU tests: no GPU, no network, no torch. Receipts are HMAC-signed with a throwaway key,
never the PC's real one (same rule as the other suites). Synthetic score.py outputs come from rebench_helpers."""
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

_SESSION = Path(tempfile.mkdtemp(prefix='isrebench'))
atexit.register(shutil.rmtree, _SESSION, ignore_errors=True)
# Pre-create a throwaway test key: exercise real signing/verification without changing Windows ACLs
# (owner-only ACL creation removes the sandbox principal's access). Never use the operator's real key.
(_SESSION / 'receipt_hmac.key').write_bytes(os.urandom(32))
os.environ['INFRARSI_RECEIPT_KEY'] = str(_SESSION / 'receipt_hmac.key')
os.environ.setdefault('INFRARSI_LAUNCH_LEDGER', str(_SESSION / 'launches.jsonl'))
