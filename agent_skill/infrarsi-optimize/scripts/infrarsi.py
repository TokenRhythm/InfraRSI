#!/usr/bin/env python3
"""The InfraRSI command of this skill: `python infrarsi.py <command> ...` is `infrarsi <command> ...`.

It runs the engine shipped with the skill (scripts/engine/infrarsi, a copy of the `infrarsi` package made when the
skill was built); without one, an installed `infrarsi` that has the `agent` command. Python's bytecode cache goes to the
user's InfraRSI directory, so that using the skill does not write into it.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    if sys.version_info < (3, 10):
        print(f'infrarsi needs Python 3.10 or newer (this is {sys.version.split()[0]}): run this script with the Python of '
              'your training environment', file=sys.stderr)
        return 2
    # this file is named as the package is: its own directory must not be where `import infrarsi` looks (Python puts
    # a script's directory first), or an installed package could never be found behind it
    sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != HERE]
    engine = HERE / 'engine'
    if (engine / 'infrarsi' / '__init__.py').is_file():
        sys.path.insert(0, str(engine))
        if not sys.pycache_prefix:
            sys.pycache_prefix = str(Path(os.environ.get('INFRARSI_HOME') or Path.home() / '.infrarsi') / 'pycache')
    try:
        import yaml  # noqa: F401
    except ImportError:
        print('infrarsi needs PyYAML in the Python that runs this script: pip install pyyaml', file=sys.stderr)
        return 2
    if sys.version_info < (3, 11):                       # (3.11 has the TOML reader built in)
        try:
            import tomli  # noqa: F401
        except ImportError:
            print('on Python 3.10 infrarsi also needs tomli: pip install tomli (or use Python 3.11 or newer)', file=sys.stderr)
            return 2
    try:
        from infrarsi.cli.main import main as cli
        from infrarsi.project import agent_session  # noqa: F401  (an engine old enough to lack `infrarsi agent`)
    except ImportError as exc:
        print(f'no usable InfraRSI engine ({exc}). The skill ships one in scripts/engine: reinstall the skill, or install '
              'an `infrarsi` package that has the `agent` command.', file=sys.stderr)
        return 2
    sys.argv[0] = 'infrarsi'
    cli()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
