import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]          # repo root holding the `infrarsi` package
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datafakes import FakeTokenizer, write_raw     # noqa: E402


@pytest.fixture(scope='session')
def raw(tmp_path_factory):
    """(raw_root, expected) with the three raw formats in their RemotePaths.datasets_raw layout."""
    root = tmp_path_factory.mktemp('raw')
    return root, write_raw(root)


@pytest.fixture(params=[False, True], ids=['nobos', 'bos'])
def tok(request):
    return FakeTokenizer(bos=request.param)
