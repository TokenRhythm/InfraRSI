"""Markdown helpers shared by the section renderers (cells -> text; one table per call)."""
from __future__ import annotations

from .cells import fmt


def table(headers: list[str], rows: list[list], digits: dict | None = None) -> str:
    """rows: lists of cells / values; digits: {column index: decimals}."""
    digits = digits or {}
    out = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join('---' for _ in headers) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(fmt(c, digits.get(i)) for i, c in enumerate(r)) + ' |')
    return '\n'.join(out)


def bullets(items) -> str:
    return '\n'.join(f'- {x}' for x in items) if items else '- none'


def why_list(cells: dict) -> list[str]:
    """'name: why' for every pending / n/a cell of a flat {name: cell} dict."""
    return [f'{k}: {c.get("status")} ({c.get("why")})' for k, c in cells.items()
            if isinstance(c, dict) and c.get('status') in ('pending', 'n/a') and c.get('why')]


def label_text(res: dict | None) -> str:
    if not res:
        return 'pending'
    lab = res.get('label') or 'pending'
    extra = res.get('suffix') or ''
    return f'{lab}{extra}'
