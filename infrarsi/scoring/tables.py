"""Result tables: CSV and Markdown renderings of score rows and the RSI verdict."""
from __future__ import annotations

import csv
import io
import math
from pathlib import Path

E1_COLUMNS = ['task', 'arm', 't_FR', 't_ED', 't_cand', 'Speedup_FR', 'Speedup_ED', 'margin', 'win', 'noisy', 'CTG_gpuh',
              'CTG_usd', 'CTG_wall_s', 'AUC_budget', 'invalid_accept', 'NumericsClass', 'LifecyclePass', 'ref_kind', 'red']


def fmt(v) -> str:
    if v is None:
        return ''
    if isinstance(v, bool):
        return 'yes' if v else 'no'
    if isinstance(v, float):
        if math.isinf(v):
            return 'inf'
        return f'{v:.4g}'
    if isinstance(v, dict):
        return ';'.join(f'{k}={fmt(x)}' for k, x in v.items() if not isinstance(x, (dict, list)))
    return str(v)


def to_csv(rows: list[dict], columns: list[str] | None = None, path: str | Path | None = None) -> str:
    cols = columns or sorted({k for r in rows for k in r})
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(cols)
    for r in rows:
        w.writerow([fmt(r.get(c)) for c in cols])
    text = buf.getvalue()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding='utf-8')
    return text


def markdown(rows: list[dict], columns: list[str] | None = None) -> str:
    cols = columns or sorted({k for r in rows for k in r})
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in rows:
        lines.append('| ' + ' | '.join(fmt(r.get(c)) for c in cols) + ' |')
    return '\n'.join(lines)


def e1_table(rows: list[dict]) -> str:
    return markdown(rows, E1_COLUMNS)


def rsi_table(verdict: dict) -> str:
    head = [f'**Verdict: {verdict["label"]}** (n = {verdict["n_tasks"]})', '']
    crit = [{'criterion': k, 'value': fmt(v['value']) if not isinstance(v['value'], list) else ','.join(v['value']),
             'need': fmt(v.get('need')), 'ok': v['ok']} for k, v in verdict['criteria'].items()]
    pairs = markdown(verdict['pairs'], ['task', 'gain_x', 'gain_y', 'd', 'margin', 'within', 'beyond', 'auc_x', 'auc_y',
                                        'd_auc', 'inheritance_calls_x'])
    return '\n'.join(head + [markdown(crit, ['criterion', 'value', 'need', 'ok']), '', pairs])
