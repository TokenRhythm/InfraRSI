"""RSI dispatcher arms (E6/E3/E7/E8): R-arm development searches and reflections, the M-arm design, and the frozen
packages on held-out / flagship / cross-GPU tasks. Search arms reuse infrarsi.agent.arms (locked scoring pair,
budgets, formal judge); every non-development task runs under a HeldoutSession pinned to the frozen package digest.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..agent import arms as A
from ..contracts import EXECUTION_DIR
from .loop import DEV_ORDER


def _searches_root(ctx) -> Path:
    return Path(ctx.searches_root or EXECUTION_DIR / 'searches')


def _search_dir(ctx, task_id: str) -> Path:
    return _searches_root(ctx) / f'E6-R-{task_id}-search'


def _events(out: Path, role: str, llm: dict) -> None:
    class _R:                                                         # LLMResult-shaped view of a recorded call
        model, llm_tokens, llm_usd = llm.get('model'), llm.get('llm_tokens'), llm.get('llm_usd')
        accounting_complete, error, wall_seconds = llm.get('accounting_complete'), llm.get('error'), llm.get('wall_seconds')
    A.llm_event(out, role, _R)


def _cost_ledger(run: dict):
    from .ledger import CostLedger
    led = CostLedger(A.COST_LEDGER)
    stop = led.check_caps(str(run.get('priority') or 'P0'))
    if stop:
        raise RuntimeError(f'LLM cost cap: {stop}')
    return led


def r_reflect(run, nodes, ctx) -> dict:
    """E6-R-<task>-reflect: A_k + the task's R-arm search trace -> A_{k+1} (snapshot A1 after D2, A_R after D6 + G3)."""
    from .loop import advance
    from .package import OptimizerPackage
    task_id = run['task']
    trace_path = _search_dir(ctx, task_id) / 'trace.json'
    if not trace_path.exists():
        raise RuntimeError(f'{trace_path} missing: run E6-R-{task_id}-search first')
    trace = json.loads(trace_path.read_text(encoding='utf-8'))
    lin = A.lineage()
    pkg = OptimizerPackage.load(lin.current()['path'])
    done = [r['task_id'] for r in lin.rows() if r['status'] in ('changed', 'unchanged', 'invalid_fallback')]
    dirs = [_search_dir(ctx, t) for t in DEV_ORDER if t in done or t == task_id]
    out = Path(ctx.search_dir(run['run_id']))
    stale = out / f'reflect_{len(done) + 1:02d}_{task_id}'
    if stale.exists():                                  # an earlier attempt died before registering A_{k+1}
        stale.rename(stale.with_name(stale.name + f'.failed-{len(list(out.glob(stale.name + ".failed-*")))}'))
    nxt, res = advance(lin, task_id, trace, pkg, A.DRIVERS['strong'](), out, ledger=_cost_ledger(run), trace_dirs=dirs)
    _events(out, 'reflect', res.llm)
    result = {'task': task_id, 'status': res.status, 'parent_digest': res.parent_digest, 'digest': res.digest,
              'package_dir': res.package_dir, 'next_package': {'label': nxt.label, 'digest': nxt.digest, 'path': str(nxt.root)},
              'errors': res.errors[:20], 'llm': res.llm, 'edit_summary': res.edit_summary,
              'frozen': {k: lin.frozen(k) for k in ('A_R',) if lin.frozen(k)}}
    A._write_json(out / 'result.json', result)
    return result


def m_design(run, nodes, ctx) -> dict:
    """E6-M-DESIGN-A2M: designer A0 + dev experience pack + A1 (as text) -> A2_M, self-checked, audited, frozen."""
    from .design_m import design_m
    from .package import OptimizerPackage
    lin = A.lineage()
    if lin.frozen('A2_M'):                              # A2_M is immutable once frozen; a re-run never replaces it
        rec = lin.frozen('A2_M')
        return {'status': 'already_frozen', 'label': 'A2_M', 'digest': rec['digest'], 'path': rec['path'], 'frozen': rec}
    snaps = [r for r in lin.rows() if r['status'] == 'snapshot' and r['label'] == 'A1']
    if not snaps:
        raise RuntimeError('snapshot A1 missing (R arm after D2)')
    a1 = OptimizerPackage.load(snaps[-1]['path'])
    if a1.digest != snaps[-1]['digest']:
        raise RuntimeError('snapshot A1 changed')
    out = Path(ctx.search_dir(run['run_id']))
    dirs = [_search_dir(ctx, t) for t in DEV_ORDER if (_search_dir(ctx, t) / 'trace.json').exists()]
    work = out / f'design_{len(list(out.glob("design_*")))}'           # a fresh directory per attempt
    res = design_m(dirs, a1, A.a0_package(), A.DRIVERS['strong'](), work, lineage=lin, ledger=_cost_ledger(run),
                   run_id=run['run_id'])
    _events(out, 'design', res['llm'])
    A._write_json(out / 'result.json', res)
    return res


RAW_SKILL = 'raw-dev-logs'
RAW_SKILL_BYTES = 60_000
RAW_SKILL_HEAD = """---
name: raw-dev-logs
description: Raw logs of the development searches (candidates, changes, verdicts, step times), unedited.
license: Apache-2.0
when_to_use: any task; look for a development task with similar hardware, model family and sequence length.
metadata:
  author: InfraRSI E6 RAWMEM ablation
  source: raw development logs (not distilled)
  version: 1
  tags: [raw, logs, development]
---

# Raw development logs

## Trigger conditions
- Always available; these are unedited logs.

## Required observations
- The current task facts and the parent receipt.

## Allowed actions
- Any L0/L1 action of the menu.

## Failure signals
- Logged failures in similar setups.

## Verification
- Screens decide; logs are hints only.

## Success case
- See the logs.

## Failure case
- See the logs.
"""


def _raw_skill_text(traces: list[dict]) -> str:
    """A0-format skill whose rule sections are the raw development logs (one section per task, not distilled)."""
    from .reflect import trace_digest
    budget = (RAW_SKILL_BYTES - len(RAW_SKILL_HEAD.encode('utf-8'))) // max(1, len(traces))
    parts = [RAW_SKILL_HEAD]
    for t in traces:
        d = trace_digest(t, max_candidates=40)
        rows = [json.dumps({k: c.get(k) for k in ('id', 'parent', 'kind', 'change', 'template', 'valid', 'step_time',
                                                  'peak_reserved_gib', 'reasons')}, ensure_ascii=False, default=str)
                for c in d['candidates']]
        head = {'task_id': d['task_id'], 'stop_reason': d['stop_reason'], 'start': d['start'], 'selected': d['selected']}
        text = '\n'.join([f'\n## Rules: raw development log {d["task_id"]}', json.dumps(head, ensure_ascii=False, default=str)]
                         + rows) + '\n'
        raw = text.encode('utf-8')
        parts.append(raw[:budget].decode('utf-8', errors='ignore') + ('\n' if len(raw) > budget else ''))
    return ''.join(parts)


def rawmem_package(ctx=None):
    """E6 RAWMEM: A0 plus the raw development logs (every R-arm development search, not distilled) as one extra skill.
    Built once after the R arm finished, gated by self-check and the isolation audit, then reused by digest."""
    import shutil
    from .audit import audit_package
    from .heldout import ReadWhitelist
    from .package import OptimizerPackage, copy_package, write_manifest
    from .selfcheck import selfcheck
    dst = A.PACKAGES_DIR / 'A0_raw_dev_memory'
    if (dst / 'manifest.json').exists():
        return OptimizerPackage.load(dst)
    root = _searches_root(ctx) if ctx is not None else EXECUTION_DIR / 'searches'
    dev, traces = ReadWhitelist.dev(), []
    for t in DEV_ORDER:
        p = root / f'E6-R-{t}-search' / 'trace.json'
        if p.exists():
            tr = json.loads(p.read_text(encoding='utf-8'))
            dev.check_task(tr.get('task_id', ''))
            traces.append(tr)
    if not traces:
        raise RuntimeError('no development search traces yet (RAWMEM depends on the R arm)')
    a0 = A.a0_package()
    tmp = dst.with_name(dst.name + '.building')
    shutil.rmtree(tmp, ignore_errors=True)
    copy_package(a0.root, tmp)
    (tmp / 'skills' / RAW_SKILL).mkdir(parents=True)
    (tmp / 'skills' / RAW_SKILL / 'SKILL.md').write_text(_raw_skill_text(traces), encoding='utf-8', newline='\n')
    write_manifest(tmp, label='A0_raw_dev_memory', parent_digest=a0.digest,
                   producer={'type': 'raw_dev_memory', 'tasks': [t['task_id'] for t in traces]})
    check, audit = selfcheck(tmp), audit_package(tmp)
    if not (check.ok and audit.passed):
        raise RuntimeError(f'raw-memory package rejected: selfcheck {check.errors[:3]} audit {audit.findings[:3]}')
    tmp.rename(dst)
    return OptimizerPackage.load(dst)


def dispatch_handlers() -> dict:
    out = A.rsi_search_handlers()
    out.update({'R_arm_reflect': r_reflect, 'M_arm_design': m_design})
    return out
