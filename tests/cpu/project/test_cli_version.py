"""`infrarsi --version` prints the installed distribution's version and exits 0, before any subcommand; a checkout
run without installing says so."""
from importlib.metadata import PackageNotFoundError, version

import pytest

from infrarsi.cli.main import main


def test_version_prints_the_installed_version(capsys):
    try:
        want = version('infrarsi')
    except PackageNotFoundError:
        want = 'unknown (not installed)'
    with pytest.raises(SystemExit) as e:
        main(['--version'])
    assert e.value.code == 0
    assert capsys.readouterr().out.strip() == f'infrarsi {want}'
