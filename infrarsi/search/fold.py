"""v4.1 bounded asks (the user's rule of 11-03: the past a model call shows grew with the run, and late calls cost several
times an early one; what does not fit is kept as a short summary that still carries its information): what a model call
shows of the search's past is bounded. The rows that matter most are shown whole while they fit a character budget; the
others are folded into one summary row that keeps what the model reads them for - what was tried, how it did, what failed
and how.

    fold_history(entries)    the measured history (finalist_base._history: the patch call's `measured_history`, the
                             configuration call's history): whole rows - the HISTORY_TOP fastest measured configurations, the
                             HISTORY_NEWEST newest runs of any outcome, the NOT_RUN_NEWEST newest proposals not run, then the
                             other measured ones by speed and the rest newest first - while they fit HISTORY_CHARS and the
                             strategy's row cap; the others as a last row {'folded': ...}: how many ended how, the speed
                             range of the measured ones, the knob values and code patches every one of them held, and per
                             other value and patch how many held it, the best speed among them and how the others ended
    fold_records(rows, ...)  a list oldest first (the ledger of the patch call, the inert patches, the failed runs): the
                             newest whole within the budget, the older ones one line each (newest first) within LINES_CHARS,
                             the oldest counted by class; the summary as the first row {'folded': ...}

A list that fits is returned as it is: an early call is the call it was before.
"""
from __future__ import annotations

import json
import math
from collections import Counter

HISTORY_CHARS = 12000           # the history's whole rows at most (characters of their JSON)
HISTORY_TOP = 8                 # the fastest measured configurations are kept whole first,
HISTORY_NEWEST = 8              # then the newest runs of any outcome,
NOT_RUN_NEWEST = 4              # then the newest proposals that were not run
LEDGER_CHARS = 8000             # the ledger's whole entries
LEDGER_ROWS = 40                # ... and at most this many (wcs's view showed the last 40)
INERT_CHARS = 6000              # the inert patches' whole records (one carries its proof lines: two or three fit)
FAILURE_CHARS = 2500            # the failed runs' whole records
LINES_CHARS = 2400              # the one-line digests of a folded list
LINE = 160                      # one digest line at most
SUMMARY_ITEMS = 24              # knob values, and code patches, with a line of their own in the history's summary

HISTORY_NOTE = ('the %d rows not shown whole, folded: how many ended how, the speed range of those measured, the knob values '
                'and code patches every one of them held, and per other value and patch how many of them held it, the best '
                'speed among those and how the others ended')
RECORDS_NOTE = '%d older %s, not shown whole: one line each, the newest first%s'


def size(row) -> int:
    return len(json.dumps(row, ensure_ascii=False, default=str))


def word(v) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)


# ---------------------------------------------------------------- the measured history
def fold_history(entries: list[dict], *, chars: int = HISTORY_CHARS, max_rows: int | None = None,
                 limits: dict | None = None) -> list[dict]:
    """`entries`: {'row': the row as shown, 'group': 'measured' / 'probe' / 'failed' / 'not_run', 'order': its place in
    time within its kind (runs, proposals not run), 'y': the measured log speed (measured rows), 'label': how it ended in a
    few words}, in the order the rows are shown when nothing is folded; `limits`: {group: whole rows of it at most}. The
    rows kept whole in that order, then the summary row of the others when there are any."""
    cap = len(entries) if max_rows is None else max(1, int(max_rows))
    limits = dict(limits or {})
    per = Counter(e['group'] for e in entries)
    if len(entries) <= cap and all(per[g] <= k for g, k in limits.items()) and sum(size(e['row']) for e in entries) <= chars:
        return [e['row'] for e in entries]
    measured = sorted((e for e in entries if e['group'] == 'measured'), key=lambda e: -e['y'])
    runs = sorted((e for e in entries if e['group'] != 'not_run'), key=lambda e: -e['order'])
    not_run = sorted((e for e in entries if e['group'] == 'not_run'), key=lambda e: -e['order'])
    keep, used, per = set(), 0, Counter()
    for e in (measured[:HISTORY_TOP] + runs[:HISTORY_NEWEST] + not_run[:NOT_RUN_NEWEST] + measured[HISTORY_TOP:]
              + runs[HISTORY_NEWEST:] + not_run[NOT_RUN_NEWEST:]):
        if id(e) in keep or per[e['group']] >= limits.get(e['group'], cap):
            continue
        n = size(e['row'])
        if len(keep) >= cap:
            break
        if used + n > chars:                             # (a row over what is left is folded; a smaller one may still fit)
            continue
        keep.add(id(e))
        used += n
        per[e['group']] += 1
    rest = [e for e in entries if id(e) not in keep]
    return [e['row'] for e in entries if id(e) in keep] + ([{'folded': history_summary(rest)}] if rest else [])


def history_summary(rest: list[dict]) -> dict:
    n = len(rest)
    out = {'note': HISTORY_NOTE % n, 'rows': n, 'ended': dict(Counter(e['label'] for e in rest).most_common())}
    ys = [e['y'] for e in rest if e['group'] == 'measured']
    if ys:
        out['measured_speed'] = f'x{math.exp(min(ys)):.3f} to x{math.exp(max(ys)):.3f} the reference speed'
    values: dict = {}
    patches: dict = {}
    for e in rest:
        for k, v in (e['row'].get('change') or {}).items():
            values.setdefault(f'{k}={word(v)}', []).append(e)
        for m in dict.fromkeys(str(x)[:60] for x in e['row'].get('code_patches') or []):
            patches.setdefault(m, []).append(e)
    for name, groups in (('values', values), ('code_patches', patches)):
        every = sorted(k for k, es in groups.items() if len(es) == n) if n > 1 else []
        if every:
            out[f'{name}_every_row_held'] = every
        lines = [_item_line(k, es) for k, es in sorted(groups.items(), key=lambda t: (-len(t[1]), t[0]))
                 if k not in every]
        if lines:
            out[name] = lines[:SUMMARY_ITEMS]
        if len(lines) > SUMMARY_ITEMS:
            out[f'{name}_not_listed'] = len(lines) - SUMMARY_ITEMS
    return out


def _item_line(key: str, es: list[dict]) -> str:
    ys = [e['y'] for e in es if e['group'] == 'measured']
    other = Counter(e['label'] for e in es if e['group'] != 'measured')
    s = f'{key}: {len(es)} row' + ('s' if len(es) > 1 else '')
    if ys:
        s += f', {len(ys)} measured, best x{math.exp(max(ys)):.3f}'
    if other:
        s += ', ' + ', '.join(f'{c} {lab}' for lab, c in other.most_common(3))
    return s[:LINE]


# ---------------------------------------------------------------- lists of records
def fold_records(rows: list, *, chars: int, line, label, what: str, max_rows: int | None = None,
                 lines: int = LINES_CHARS) -> list:
    """`rows` oldest first; `line(row)`: its digest line, `label(row)`: its class for the count; `what`: the records'
    name in the summary's note."""
    rows = list(rows or [])
    cap = len(rows) if max_rows is None else max(1, int(max_rows))
    if len(rows) <= cap and sum(size(r) for r in rows) <= chars:
        return rows
    keep, used = 0, 0
    for r in reversed(rows):                             # (the newest whole while they fit; one alone over the budget: none)
        n = size(r)
        if keep >= cap or used + n > chars:
            break
        keep, used = keep + 1, used + n
    older = rows[:len(rows) - keep]
    digest, used = [], 0
    for r in reversed(older):
        s = str(line(r))[:LINE]
        if digest and used + len(s) > lines:
            break
        digest.append(s)
        used += len(s)
    rest = older[:len(older) - len(digest)]
    out = {'note': RECORDS_NOTE % (len(older), what, ', the oldest only counted by how they ended' if rest else ''),
           'lines': digest}
    if rest:
        out['oldest_counted'] = dict(Counter(str(label(r)) for r in rest).most_common())
    return [{'folded': out}] + rows[len(rows) - keep:]


def fold_ledger(rows: list, *, chars: int = LEDGER_CHARS) -> list:
    """The ledger of the patch call (wcs.model_view, every entry): the newest whole, at most LEDGER_ROWS of them."""
    return fold_records(rows, chars=chars, line=ledger_line, label=ledger_label, what='ledger entries',
                        max_rows=LEDGER_ROWS)


def ledger_line(r: dict) -> str:
    s = str(r.get('mechanism') or '')[:90]
    ch = r.get('change') or {}
    if ch:
        s += ' [' + ', '.join(f'{k}={word(v)}' for k, v in ch.items())[:60] + ']'
    s += ' -> ' + str(r.get('status') or 'open')
    g = r.get('measured_gain_pct')
    if isinstance(g, (int, float)):
        s += f' {g:+.2f} %'
    elif r.get('why'):
        s += ': ' + str(r['why'])[:70]
    return s


def ledger_label(r: dict) -> str:
    return str(r.get('status') or 'open')


def inert_line(r: dict) -> str:
    s = f'{r.get("candidate")}: {str(r.get("mechanism") or "")[:70]}'
    if r.get('at') == 'probe':
        return s + (f' @ {str(r["site"])[:40]}' if r.get('site') else '') + f' - inert at its probe: {str(r.get("reason") or "")[:80]}'
    m, p = r.get('measured_pct'), r.get('predicted_pct')
    return s + ' - ran in full' + (f', measured {m:+.2f} % against a predicted {p:+.2f} %'
                                   if isinstance(m, (int, float)) and isinstance(p, (int, float)) else '')


def inert_label(r: dict) -> str:
    return f'inert at its {r.get("at") or "run"}'


def failure_line(r: dict) -> str:
    return (f'{r.get("candidate")} {r.get("class")}: {str(r.get("mechanism") or "")[:70]}'
            + (f' - {str(r["detail"])[:70]}' if r.get('detail') else ''))


def failure_label(r: dict) -> str:
    return str(r.get('class') or 'failed')
