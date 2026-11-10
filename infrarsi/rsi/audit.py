"""Answer-isolation audit (G3): a package may contain no held-out task ids, held-out model/data identifiers or
combination labels, and no receipts. Scans every package file, including memory.db rows (logically, not bytes).
"""
from __future__ import annotations

import dataclasses
import json
import re
import sqlite3
from pathlib import Path

from ..contracts import load_all_tasks

SCAN_SUFFIXES = ('.md', '.yaml', '.yml', '.json', '.analysis', '.txt')
RECEIPT_RE = re.compile(r'''["']?schema["']?\s*[:=]\s*["']infrarsi-receipt-v1["']''')


@dataclasses.dataclass
class AuditReport:
    passed: bool
    findings: list
    scanned: list

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _aliases(model: dict) -> set[str]:
    out = set()
    for m in (model, model.get('fallback') or {}):
        rid = m.get('repo_id')
        if rid:
            out |= {rid, rid.split('/')[-1]}
        for k in ('key', 'bridge_type', 'family'):
            if m.get(k):
                out.add(m[k])
    return {a for a in out if a and not a.startswith('local:')}


def isolation_terms(tasks: dict | None = None) -> dict:
    """Regexes derived from the frozen task specs: what only held-out tasks contain."""
    tasks = tasks or load_all_tasks()
    held = {k: t.raw for k, t in tasks.items() if t.raw.get('split') == 'heldout'}
    other = {k: t.raw for k, t in tasks.items() if t.raw.get('split') != 'heldout'}
    seen_models = set().union(*(_aliases(r['model']) for r in other.values())) if other else set()
    seen_data = {r['data']['key'] for r in other.values()}
    seen_combos = {(r['model']['repo_id'], r['data']['key'], int(r['hardware']['gpus'])) for r in other.values()}
    ids = sorted(held)
    terms = {'heldout_ids': ids, 'task_id': [rf'(?<![A-Za-z0-9_]){re.escape(i)}(?![A-Za-z0-9_])' for i in ids],
             'model': [], 'data': [], 'combination': []}
    for r in held.values():
        for a in sorted(_aliases(r['model']) - {x for x in seen_models}):
            terms['model'].append(rf'(?i)(?<![A-Za-z0-9]){re.escape(a)}(?![A-Za-z0-9])')
        if r['data']['key'] not in seen_data:
            terms['data'].append(rf'(?i){re.escape(r["data"]["key"])}')
        combo = (r['model']['repo_id'], r['data']['key'], int(r['hardware']['gpus']))
        if combo not in seen_combos and _aliases(r['model']) & seen_models:     # unseen models are flagged by name
            terms['combination'].append({'models': sorted(_aliases(r['model']) & seen_models),
                                         'data': r['data']['key'], 'gpus': combo[2], 'nodes': int(r['hardware']['nodes'])})
    for k in ('model', 'data'):
        terms[k] = sorted(set(terms[k]))
    return terms


COMBO_WINDOW_CHARS = 400


def _gpu_marker(n: int, nodes: int) -> re.Pattern:
    return re.compile(rf'(?i)(\b{n}\s*(x|×)?\s*(h100|gpus?|cards?|ranks?)\b|\bgpus{n}\b|\b{n}-gpu\b|\bnodes{nodes}\b|\b{nodes}\s+nodes\b)')


def _scan_text(text: str, where: str, terms: dict, findings: list):
    for kind in ('task_id', 'model', 'data'):
        for pat in terms[kind]:
            for m in re.finditer(pat, text):
                findings.append({'file': where, 'kind': kind, 'term': m.group(0), 'excerpt': text[max(0, m.start() - 60): m.end() + 60]})
    low_text = text.lower()
    for combo in terms['combination']:
        # Model, data and GPU count within a bounded window around each model mention (not only one line), so
        # a combination split over several lines or table cells is still found.
        marker = _gpu_marker(combo['gpus'], combo['nodes'])
        hit = None
        for a in combo['models']:
            for m in re.finditer(re.escape(a.lower()), low_text):
                lo, hi = max(0, m.start() - COMBO_WINDOW_CHARS), m.end() + COMBO_WINDOW_CHARS
                win = text[lo:hi]
                if combo['data'].lower() in win.lower() and marker.search(win):
                    hit = win
                    break
            if hit is not None:
                break
        if hit is not None:
            findings.append({'file': where, 'kind': 'combination', 'term': f'{combo["models"][:1]}+{combo["data"]}+{combo["gpus"]}gpu',
                             'excerpt': ' '.join(hit.split())[:240]})
    if RECEIPT_RE.search(text):
        findings.append({'file': where, 'kind': 'receipt', 'term': 'infrarsi-receipt-v1', 'excerpt': ''})


def audit_package(pkg_dir: str | Path, *, tasks: dict | None = None, heldout_receipts=(), extra_terms=()) -> AuditReport:
    """heldout_receipts: receipts (dicts) of held-out runs; their attempt/config/signals digests must not appear."""
    root = Path(pkg_dir)
    terms = isolation_terms(tasks)
    if extra_terms:
        terms['task_id'] = terms['task_id'] + [re.escape(t) for t in extra_terms]
    secrets = set()
    for r in heldout_receipts:
        for k in ('attempt_id', 'config_digest', 'signals_digest', 'run_id'):
            if r.get(k):
                secrets.add(str(r[k]))
    findings, scanned = [], []
    for p in sorted(root.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts:
            continue
        rel = p.relative_to(root).as_posix()
        if rel == 'memory.db':
            text = _memory_text(p, findings, terms)
        elif p.suffix in SCAN_SUFFIXES:
            text = p.read_text(encoding='utf-8', errors='replace')
        else:
            findings.append({'file': rel, 'kind': 'unexpected_file', 'term': p.suffix, 'excerpt': ''})
            continue
        scanned.append(rel)
        _scan_text(text, rel, terms, findings)
        for s in secrets:
            if s in text:
                findings.append({'file': rel, 'kind': 'heldout_receipt_value', 'term': s, 'excerpt': ''})
    return AuditReport(passed=not findings, findings=findings, scanned=scanned)


def _memory_text(path: Path, findings: list, terms: dict) -> str:
    try:
        con = sqlite3.connect(f'file:{path.as_posix()}?mode=ro', uri=True)
        rows = [dict(zip([c[0] for c in cur.description], row)) for cur in [con.execute('SELECT * FROM entries')] for row in cur.fetchall()]
        con.close()
    except sqlite3.DatabaseError as exc:
        findings.append({'file': 'memory.db', 'kind': 'unreadable', 'term': str(exc), 'excerpt': ''})
        return ''
    held_ids = set(terms['heldout_ids'])
    for r in rows:
        if r.get('task_id') and r['task_id'] in held_ids:
            findings.append({'file': 'memory.db', 'kind': 'task_id', 'term': r['task_id'], 'excerpt': f'entry {r.get("id")}'})
    return '\n'.join(json.dumps(r, ensure_ascii=False, default=str) for r in rows)
