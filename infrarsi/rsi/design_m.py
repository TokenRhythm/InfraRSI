"""E6 M arm (RUN_MATRIX E6-M-DESIGN-A2M): the base optimizer A0 acts as designer. It reads the development experience
pack (digests of every development search trace) and A1's package source as text -- it never executes A1 -- and
writes A2_M as one edit of A1 in the reflect edit grammar (skills, policy, prompts, tools, memory). The edit must pass
the package self-check and the answer-isolation audit; otherwise A2_M = A1 (invalid_fallback). A2_M is snapshotted
and frozen in the lineage like A_R and is then run on held-out tasks (A2_M_heldout) by digest.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..agent.tools import REFLECT_SCHEMA
from .audit import audit_package
from .heldout import ReadWhitelist
from .loop import history_receipts
from .package import OptimizerPackage, compute_digest, copy_package, write_manifest
from .reflect import apply_edit, package_view, trace_digest
from .selfcheck import selfcheck

DESIGN_NOTE = ('\n\nDESIGN MODE (M arm). You are not reflecting on one task: the user message holds the complete development '
               'experience pack (one digest per development search) and the current package (A1) as text. Design the next '
               'optimizer package as ONE edit object of the same schema. Do not mention or target any task outside the pack.')


def design_m(trace_dirs, a1: OptimizerPackage, designer: OptimizerPackage, driver, out_dir: str | Path, *, lineage,
             ledger=None, run_id: str = 'E6-M-DESIGN-A2M', label: str = 'A2_M') -> dict:
    dev = ReadWhitelist.dev()
    traces = []
    for d in trace_dirs:
        p = Path(d) / 'trace.json'
        if p.exists():
            t = json.loads(p.read_text(encoding='utf-8'))
            dev.check_task(t.get('task_id', ''))                     # development experience only (fails closed)
            traces.append(t)
    if not traces:
        raise RuntimeError('M design needs at least one finished development search trace')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    prompt = json.dumps({'mode': 'design', 'experience_pack': [trace_digest(t, max_candidates=16) for t in traces],
                         'package': package_view(a1)}, ensure_ascii=False, default=str)
    res = driver.complete(designer.prompts['reflect.md'] + DESIGN_NOTE, prompt, schema=REFLECT_SCHEMA, call_dir=out / 'call',
                          role='design')
    llm = {'llm_tokens': res.llm_tokens, 'llm_usd': res.llm_usd, 'accounting_complete': res.accounting_complete,
           'error': res.error, 'model': res.model, 'wall_seconds': res.wall_seconds, 'call_id': res.meta.get('call_id')}
    if ledger is not None:
        ledger.add(book='llm', run_id=run_id, arm='M', task_id=None, phase='design', llm_tokens=res.llm_tokens or 0,
                   llm_usd=res.llm_usd or 0.0, model=res.model, wall_seconds=res.wall_seconds,
                   accounting_complete=res.accounting_complete, source='design_m')
    errors, candidate = [], out / 'candidate'
    if res.action is None:
        errors.append(f'no usable design: {res.error}')
    else:
        errors += apply_edit(a1.root, candidate, res.action, task_id='M', search_run_id=run_id)
        if not errors:
            write_manifest(candidate, label=label, parent_digest=a1.digest,
                           producer={'type': 'design_m', 'designer_digest': designer.digest, 'call_id': res.meta.get('call_id'),
                                     'model': res.model, 'rationale': (res.action.get('rationale') or '')[:500],
                                     'experience_tasks': [t['task_id'] for t in traces]})
            check = selfcheck(candidate, history_receipts=history_receipts(trace_dirs))
            errors += check.errors
            audit = audit_package(candidate)
            errors += [f'audit: {f["kind"]} {f["term"]} in {f["file"]}' for f in audit.findings[:8]]
    final = out / 'package'
    if errors:
        copy_package(a1.root, final)
        status = 'invalid_fallback'
    else:
        candidate.rename(final)
        status = 'unchanged' if compute_digest(final) == compute_digest(a1.root) else 'changed'
    pkg = OptimizerPackage.load(final)
    snap = lineage.snapshot(label, pkg)
    frozen = lineage.freeze(label)
    result = {'status': status, 'label': label, 'digest': snap.digest, 'path': str(snap.root), 'parent_digest': a1.digest,
              'designer_digest': designer.digest, 'errors': errors[:40], 'llm': llm, 'frozen': frozen,
              'experience_tasks': [t['task_id'] for t in traces]}
    (out / 'design_result.json').write_text(json.dumps(result, indent=1, default=str), encoding='utf-8')
    return result
