"""CPU tests of the Claude Code / Codex skill (agent_skill/). Receipts are signed with a throwaway key, never the PC's real one."""
import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(HERE), str(HERE.parent / 'agent'), str(HERE.parent / 'project'), str(REPO / 'agent_skill')):
    if p not in sys.path:
        sys.path.insert(0, p)

_SESSION = Path(tempfile.mkdtemp(prefix='isskill'))
atexit.register(shutil.rmtree, _SESSION, ignore_errors=True)
os.environ['INFRARSI_RECEIPT_KEY'] = str(_SESSION / 'receipt_hmac.key')   # (set, not defaulted: a key or a home named
os.environ['INFRARSI_LAUNCH_LEDGER'] = str(_SESSION / 'launches.jsonl')   #  in the caller's environment must not be used)
os.environ['INFRARSI_HOME'] = str(_SESSION / 'home')
