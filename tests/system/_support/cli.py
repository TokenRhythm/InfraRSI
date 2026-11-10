"""Run `infrarsi <args>` as a user would: in this process (fast; stdout, stderr and the exit status captured) or as a
separate process (signals, a clean environment)."""
from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


@dataclass
class Result:
    code: int
    out: str
    err: str

    @property
    def text(self) -> str:
        return self.out + self.err

    def __contains__(self, s: str) -> bool:
        return s in self.text

    def lines(self, prefix: str) -> list[str]:
        return [ln for ln in self.text.splitlines() if ln.startswith(prefix)]


def run(*argv, cwd=None) -> Result:
    """`infrarsi <argv>` in this process. The exit status is the one a shell would see: 0 when the command returned,
    the SystemExit code otherwise (a message given to SystemExit goes to stderr with status 1, as the interpreter does)."""
    from infrarsi.cli.main import main
    out, err = io.StringIO(), io.StringIO()
    old = os.getcwd()
    if cwd is not None:
        os.chdir(cwd)
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rc = main([str(a) for a in argv])
                code = int(rc) if isinstance(rc, int) else 0
            except SystemExit as exc:
                if exc.code is None:
                    code = 0
                elif isinstance(exc.code, int):
                    code = exc.code
                else:
                    err.write(str(exc.code) + '\n')
                    code = 1
    finally:
        os.chdir(old)
    return Result(code, out.getvalue(), err.getvalue())


def run_proc(*argv, cwd=None, env=None, timeout: float = 900, stdin_text: str | None = None) -> Result:
    """`python -m infrarsi <argv>` as a separate process, the test's environment plus `env`."""
    p = subprocess.run([sys.executable, '-m', 'infrarsi', *map(str, argv)], cwd=cwd, env={**os.environ, **(env or {})},
                       capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout, input=stdin_text)
    return Result(p.returncode, p.stdout, p.stderr)
