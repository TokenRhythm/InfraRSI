"""Guards (EXT_INTERFACES §4; design §8 risk 14): they only ever add rejections to the environment judge's verdict.

Every rejection is {'id', 'detail'}; checks that never veto (reports) are {'id', ...data}. Generic guards here:
frozen_files (sha256 before and after every run), outputs_present, exit_ok, budget. Environment guards (ENV-A
G1-G7, ENV-B G-B1..B3, ENV-C infra_only_check) return the same GuardResult and are merged with `combine`.
"""
from __future__ import annotations

import dataclasses
import fnmatch
import math


@dataclasses.dataclass
class GuardResult:
    rejections: list = dataclasses.field(default_factory=list)       # [{'id', 'detail'}]
    reports: list = dataclasses.field(default_factory=list)          # [{'id', ...}] non-vetoing

    def reject(self, gid: str, detail: str) -> 'GuardResult':
        self.rejections.append({'id': str(gid), 'detail': str(detail)[:2000]})
        return self

    def report(self, gid: str, **data) -> 'GuardResult':
        self.reports.append({'id': str(gid), **data})
        return self

    def extend(self, other: 'GuardResult | None') -> 'GuardResult':
        if other is not None:
            self.rejections += list(other.rejections)
            self.reports += list(other.reports)
        return self

    @property
    def ok(self) -> bool:
        return not self.rejections

    def ids(self) -> list[str]:
        return [r['id'] for r in self.rejections]

    def to_dict(self) -> dict:
        return {'rejections': list(self.rejections), 'reports': list(self.reports)}


def combine(*results: GuardResult | None) -> GuardResult:
    out = GuardResult()
    for r in results:
        out.extend(r)
    return out


def frozen_files(expected: dict, before: dict | None, after: dict | None, *, gid: str = 'frozen_files') -> GuardResult:
    """expected {path: sha256} (the task's verified base tree) vs the node's hashes before and after the run
    ({path: sha256 | None when missing}). A missing measurement is itself a rejection: the run cannot prove it."""
    g = GuardResult()
    for when, seen in (('before', before), ('after', after)):
        if not isinstance(seen, dict):
            g.reject(gid, f'no {when}-run hashes')
            continue
        for path, sha in sorted(expected.items()):
            got = seen.get(path)
            if got is None:
                g.reject(gid, f'{path} missing {when} the run')
            elif got != sha:
                g.reject(gid, f'{path} changed {when} the run: {str(got)[:16]} != {sha[:16]}')
    return g


def outputs_present(present: dict | None, required, *, gid: str = 'outputs_present') -> GuardResult:
    """present {name: size in bytes | bool | None}; every required output must exist and be non-empty."""
    g = GuardResult()
    present = present if isinstance(present, dict) else {}
    for name in required:
        v = present.get(name)
        if v is None or v is False or (type(v) in (int, float) and v <= 0):
            g.reject(gid, f'{name} missing or empty')
    return g


def exit_ok(exit_code, *, timed_out: bool = False, what: str = 'run', gid: str = 'exit_ok') -> GuardResult:
    g = GuardResult()
    if timed_out:
        g.reject(gid, f'{what} exceeded its timeout and was killed')
    elif type(exit_code) is not int:
        g.reject(gid, f'{what} exit code unknown ({exit_code!r})')
    elif exit_code != 0:
        g.reject(gid, f'{what} exited with {exit_code}')
    return g


def budget(wall_s, cap_s, *, what: str = 'run', gid: str = 'budget') -> GuardResult:
    """A run must stay within its per-run wall cap (e.g. ENV-C: a run over 10 minutes is a failure)."""
    g = GuardResult()
    if not (type(wall_s) in (int, float) and math.isfinite(wall_s) and wall_s >= 0):
        g.reject(gid, f'{what} wall time unknown')
    elif cap_s is not None and wall_s > cap_s:
        g.reject(gid, f'{what} took {wall_s:.0f}s > cap {cap_s:.0f}s')
    return g


def path_problem(path) -> str | None:
    """Why a candidate's file path is not a canonical relative POSIX path, or None (Codex W9: every environment
    rejects absolute paths, '..' / '.' / empty components and non-canonical separators before any glob is applied; an
    fnmatch '*' also matches '/')."""
    if type(path) is not str or not path:
        return 'empty or non-string path'
    if any(ord(c) < 32 or ord(c) == 127 for c in path):
        return 'control character in path'
    if chr(92) in path:
        return 'backslash separator (use /)'
    if path.startswith('/') or (len(path) > 1 and path[1] == ':'):
        return 'absolute path'
    if any(part in ('', '.', '..') for part in path.split('/')):
        return "empty, '.' or '..' path component"
    return None


def touched_outside(files, mutable_globs) -> list[str]:
    """Paths of a candidate's file set that no mutable glob admits (the patch guard's first check)."""
    return sorted(p for p in files if not any(fnmatch.fnmatchcase(p, g) for g in mutable_globs))


def tolerance_report(gid: str, value, limit, *, veto: bool, detail: str = '') -> GuardResult:
    """A numeric check that rejects when veto is set and reports otherwise (pre-declared report-only checks)."""
    g = GuardResult()
    bad = not (type(value) in (int, float) and math.isfinite(value)) or (limit is not None and value > limit)
    if bad and veto:
        g.reject(gid, detail or f'{value!r} > {limit!r}')
    g.report(gid, value=value, limit=limit, exceeded=bool(bad), veto=veto, detail=detail)
    return g
