"""A job's own Python environment for the D7 tests: a real virtual environment (no pip), whose interpreter reports its
own site-packages root - the only place project/libsource reads library source from (Codex HLS5 r1)."""
import json
import os
import subprocess
import sys
from pathlib import Path


def make_venv(root: Path) -> tuple[str, Path]:
    """(the environment's python, its site-packages directory)."""
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(root)], check=True, capture_output=True,
                   timeout=300)
    py = root / ('Scripts' if os.name == 'nt' else 'bin') / ('python.exe' if os.name == 'nt' else 'python')
    out = subprocess.run([str(py), '-c', 'import json, sysconfig; print(json.dumps(sysconfig.get_paths()["purelib"]))'],
                         check=True, capture_output=True, text=True, timeout=120).stdout
    site = Path(json.loads(out.strip().splitlines()[-1]))
    site.mkdir(parents=True, exist_ok=True)
    return str(py), site
