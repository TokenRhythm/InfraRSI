"""CPU tests of infrarsi.experience: the shipped seed packs, a Megatron-Core job shaped like the H2b project (Llama-3-8B,
TP1, 8 GPUs), and synthetic packs. INFRARSI_HOME points at a throwaway directory."""
import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

_SESSION = Path(tempfile.mkdtemp(prefix='isexp'))
atexit.register(shutil.rmtree, _SESSION, ignore_errors=True)
os.environ.setdefault('INFRARSI_HOME', str(_SESSION / 'home'))
