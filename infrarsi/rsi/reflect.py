"""reflect(): after each development task the LLM writes the next package version from that task's trace.

The proposal is applied to a copy of A_k, re-manifested and self-checked. Pass -> A_{k+1} (changed/unchanged);
fail (or no usable reply) -> A_{k+1} = A_k with status invalid_fallback. Reflection cost goes to the R arm's
development books, never to a search budget.
"""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import yaml

from ..agent.memory import MemoryDB
from ..agent.tools import REFLECT_SCHEMA
from .lineage import Lineage
from .package import OptimizerPackage, REQUIRED_PROMPTS, compute_digest, copy_package, write_manifest
from .selfcheck import selfcheck

SKILL_NAME = re.compile(r'^[a-z0-9][a-z0-9-]{2,63}$')
TOOL_NAME = re.compile(r'^[a-z0-9][a-z0-9_]{1,47}$')


@dataclasses.dataclass
class ReflectResult:
    status: str                   # changed | unchanged | invalid_fallback
    package_dir: str
    digest: str
    parent_digest: str
    errors: list
    llm: dict
    edit_summary: dict

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def trace_digest(trace: dict, max_candidates: int = 40) -> dict:
    """What reflection sees of a finished search (its own task only)."""
    return {
        'task_id': trace['task_id'], 'arm': trace['arm'], 'mode': trace['mode'], 'stop_reason': trace['stop_reason'],
        'start': {'id': trace['start'], 'step_time': trace['start_step_time']},
        'selected': {'id': trace['selected'], 'step_time': trace['selected_step_time'], 'config': trace['selected_config'],
                     'overlay_files': sorted(trace.get('selected_overlay') or {})},
        'candidates': [{k: c.get(k) for k in ('id', 'parent', 'kind', 'status', 'change', 'template', 'valid', 'step_time',
                                                'peak_reserved_gib', 'reasons', 'rationale')} for c in trace['candidates'][:max_candidates]],
        'screens': [{k: s.get(k) for k in ('index', 'candidate_id', 'valid', 'step_time', 'reasons', 'gpu_hours', 'fractions')} for s in trace['screens']],
        'used': trace['used'], 'budget': trace['budget'], 'invalid_proposals': trace['invalid_proposals'], 'verifier': trace['verifier'],
        'skill_hits': trace['skill_hits'][-40:], 'tool_calls': trace.get('tool_calls', [])[-20:], 'memory_writes': trace['memory_writes'],
    }


def package_view(pkg: OptimizerPackage, max_memory_rows: int = 60) -> dict:
    skills = {p.parent.name: p.read_text(encoding='utf-8') for p in sorted(pkg.skills_dir.glob('*/SKILL.md'))}
    db = MemoryDB(pkg.memory_path, read_only=True)
    try:
        rows = db.rows()[-max_memory_rows:]
    finally:
        db.close()
    return {'label': pkg.label, 'digest': pkg.digest, 'skills': skills, 'policy': pkg.policy_text, 'prompts': pkg.prompts,
            'tools': pkg.tools, 'memory': [{k: r[k] for k in ('task_id', 'tags', 'slow', 'fast', 'outcome', 'failure_reason', 'lesson')} for r in rows]}


def apply_edit(src: Path, dst: Path, edit: dict, *, task_id: str, search_memory=(), search_run_id: str = '') -> list[str]:
    """Apply a reflect edit to a copy of the package. Returns structural errors (selfcheck does the rest).

    search_memory: entries the agent wrote during the task's search (trace memory_writes); they enter A_{k+1} here.
    """
    errors = []
    copy_package(src, dst)
    if search_memory:
        db = MemoryDB(dst / 'memory.db')
        try:
            for e in search_memory:
                try:
                    db.add(e, source=search_run_id or f'search:{task_id}', task_id=task_id)
                except (KeyError, ValueError) as exc:
                    errors.append(f'search memory entry rejected: {exc}')
        finally:
            db.close()
    for name, text in (edit.get('skills') or {}).items():
        if not SKILL_NAME.match(name):
            errors.append(f'bad skill name {name!r}')
            continue
        d = dst / 'skills' / name
        if text is None:
            if d.exists():
                for p in d.iterdir():
                    p.unlink()
                d.rmdir()
            continue
        d.mkdir(parents=True, exist_ok=True)
        (d / 'SKILL.md').write_text(text, encoding='utf-8', newline='\n')
    if edit.get('policy') is not None:
        try:
            yaml.safe_load(edit['policy'])
        except yaml.YAMLError as exc:
            errors.append(f'policy yaml: {exc}')
        (dst / 'policy.yaml').write_text(edit['policy'], encoding='utf-8', newline='\n')
    for name, text in (edit.get('prompts') or {}).items():
        if name not in REQUIRED_PROMPTS:
            errors.append(f'unknown prompt {name!r}')
            continue
        (dst / 'prompts' / name).write_text(text, encoding='utf-8', newline='\n')
    for name, src_text in (edit.get('tools') or {}).items():
        if not TOOL_NAME.match(name):
            errors.append(f'bad tool name {name!r}')
            continue
        p = dst / 'tools' / f'{name}.analysis'
        if src_text is None:
            p.unlink(missing_ok=True)
        else:
            p.write_text(src_text, encoding='utf-8', newline='\n')
    entries = edit.get('memory') or []
    if entries:
        db = MemoryDB(dst / 'memory.db')
        try:
            for e in entries:
                try:
                    db.add(e, source=f'reflect:{task_id}', task_id=task_id)
                except (KeyError, ValueError) as exc:
                    errors.append(f'memory entry rejected: {exc}')
        finally:
            db.close()
    return errors


def reflect(task_id: str, trace: dict, package: OptimizerPackage | str | Path, driver, out_dir: str | Path, *,
            label: str, history_receipts=(), lineage: Lineage | None = None, ledger=None, run_id: str | None = None,
            tasks=None) -> ReflectResult:
    from .heldout import ReadWhitelist
    ReadWhitelist.dev().check_task(task_id)          # reflection is development-only (G3); fails closed
    pkg = package if isinstance(package, OptimizerPackage) else OptimizerPackage.load(package)
    if trace.get('task_id') != task_id:
        raise ValueError('trace belongs to another task')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    prompt = json.dumps({'package': package_view(pkg), 'trace': trace_digest(trace)}, ensure_ascii=False, default=str)
    res = driver.complete(pkg.prompts['reflect.md'], prompt, schema=REFLECT_SCHEMA, call_dir=out / 'call', role='reflect')
    llm = {'llm_tokens': res.llm_tokens, 'llm_usd': res.llm_usd, 'accounting_complete': res.accounting_complete,
           'error': res.error, 'model': res.model, 'call_id': res.meta.get('call_id')}
    if ledger is not None:
        ledger.add(book='llm', run_id=run_id or f'E6-R-{task_id}-reflect', arm='R', task_id=task_id, phase='reflect',
                   llm_tokens=res.llm_tokens or 0, llm_usd=res.llm_usd or 0.0, model=res.model,
                   wall_seconds=res.wall_seconds, accounting_complete=res.accounting_complete, source='reflect')
    errors, summary = [], {}
    candidate = out / 'candidate'
    search_memory = [w['entry'] for w in trace.get('memory_writes', []) if not w.get('noop') and isinstance(w.get('entry'), dict)]
    if res.action is None:
        errors.append(f'no usable reflection: {res.error}')
    else:
        edit = res.action
        summary = {'skills': sorted(edit.get('skills') or {}), 'policy': edit.get('policy') is not None,
                   'prompts': sorted(edit.get('prompts') or {}), 'tools': sorted(edit.get('tools') or {}),
                   'memory': len(edit.get('memory') or []), 'search_memory': len(search_memory),
                   'rationale': (edit.get('rationale') or '')[:2000]}
        errors += apply_edit(pkg.root, candidate, edit, task_id=task_id, search_memory=search_memory,
                             search_run_id=trace.get('run_id', ''))
        if not errors:
            write_manifest(candidate, label=label, parent_digest=pkg.digest,
                           producer={'type': 'reflect', 'task_id': task_id, 'call_id': res.meta.get('call_id'),
                                     'model': res.model, 'rationale': summary['rationale'][:500]})
            report = selfcheck(candidate, history_receipts=history_receipts, tasks=tasks)
            (out / 'selfcheck.json').write_text(json.dumps(report.to_dict(), indent=1), encoding='utf-8')
            errors += report.errors
    if errors:
        final = out / 'package'
        copy_package(pkg.root, final)                 # A_{k+1} = A_k, same content digest
        status, new = 'invalid_fallback', OptimizerPackage.load(final)
    else:
        final = out / 'package'
        candidate.rename(final)
        new = OptimizerPackage.load(final)
        status = 'unchanged' if compute_digest(final) == compute_digest(pkg.root) else 'changed'
    result = ReflectResult(status=status, package_dir=str(final), digest=new.digest, parent_digest=pkg.digest,
                           errors=errors[:40], llm=llm, edit_summary=summary)
    (out / 'reflect_result.json').write_text(json.dumps(result.to_dict(), indent=1, default=str), encoding='utf-8')
    if lineage is not None:
        lineage.register(new, status=status, task_id=task_id, parent_digest=pkg.digest,
                         reflect={'errors': errors[:10], 'call_id': res.meta.get('call_id'), 'out': str(out)})
    return result
