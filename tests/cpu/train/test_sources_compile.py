"""Every T-train module must at least compile on the driver PC (torch-only modules are not imported here)."""
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[3] / 'infrarsi'
FILES = sorted(p for d in ('runtime', 'adapters/megatron_core') for p in (PKG / d).rglob('*.py'))


@pytest.mark.parametrize('path', FILES, ids=lambda p: p.relative_to(PKG).as_posix())
def test_compiles(path):
    compile(path.read_text(encoding='utf-8'), str(path), 'exec')
