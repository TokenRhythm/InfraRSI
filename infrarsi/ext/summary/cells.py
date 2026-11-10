"""Table cells: every number carries its sources; missing values are `pending` or `n/a`, never invented.

A cell is a dict: {'v': value, 'src': [source ids]} (+ 'note'), or {'v': None, 'status': 'pending' | 'n/a', 'why': ...}.
Source ids: '<run_id>:<attempt_id>' for a receipt, '<run_id>/<file>' for another record, or a ledger / amendment name.
"""
from __future__ import annotations

import math

PENDING, NA = 'pending', 'n/a'


def finite(x) -> bool:
    return type(x) in (int, float) and math.isfinite(x)


def cell(v, src=None, *, note: str | None = None) -> dict:
    """A value with its sources; a value the record does not carry becomes an explicit n/a (never a silent None)."""
    if isinstance(v, float) and not math.isfinite(v):
        v = None                                   # JSON has no inf: CTG "not reached" is said by its own field
    out = {'v': v}
    if src:
        out['src'] = sorted({str(s) for s in (src if isinstance(src, (list, tuple, set)) else [src]) if s})
    if note:
        out['note'] = note
    if v is None:
        out.update(status=NA, why=note or 'no value in the record')
    return out


def pending(why: str) -> dict:
    return {'v': None, 'status': PENDING, 'why': why}


def na(why: str) -> dict:
    return {'v': None, 'status': NA, 'why': why}


def status(c) -> str:
    if isinstance(c, dict) and c.get('status') in (PENDING, NA):
        return c['status']
    return 'ok'


def val(c):
    return c.get('v') if isinstance(c, dict) else c


def ok(c) -> bool:
    return isinstance(c, dict) and status(c) == 'ok' and c.get('v') is not None


def fmt(c, digits: int | None = None) -> str:
    """Markdown text of a cell (or a plain value)."""
    if isinstance(c, dict) and 'v' in c:
        if status(c) != 'ok':
            return c['status']
        v = c['v']
    else:
        v = c
    if v is None:
        return 'n/a'
    if isinstance(v, bool):
        return 'yes' if v else 'no'
    if isinstance(v, float):
        if digits is None:
            digits = 4 if abs(v) < 10 else 2
        return f'{v:.{digits}f}'
    if isinstance(v, (list, tuple)):
        return ', '.join(str(x) for x in v) if v else '—'
    if isinstance(v, dict):
        return ', '.join(f'{k} {x}' for k, x in v.items()) if v else '—'
    return str(v).replace('|', '\\|').replace('\n', ' ')


def missing(status: str, why: str | None = None) -> dict:
    """The cell of a row without the value: n/a for a sanctioned `not run` arm, pending otherwise."""
    return na(f'not run: {why}' if why else 'not run') if status == 'not run' else pending(status)
