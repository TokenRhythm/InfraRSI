"""A verifier model checks the code patches of a search before they cost a run (the debate's patch check, off by default).

`check_patches` is called on what the proposer is about to return. For each code patch, at most `rounds` checks with one
revision between them: the verifier (an explicit one, else the model that did not write the patch) reads the diff against the
job's rules; a failed check goes back to the patch's author for one revision, and the revised patch is checked again; a patch
that still fails is dropped and the model is told why. A check that gets no usable answer (or that the budget cannot pay)
leaves the patch in place: every run is still judged by the verifier of the search, which is what decides validity.
"""
from __future__ import annotations

import difflib
import hashlib
import json

from ..agent.optimize import MIN_CALL_USD
from .proposers import _allowances, patch_of

CHECK_SCHEMA = {'type': 'object', 'required': ['verdict'], 'properties': {
    'verdict': {'type': 'string', 'enum': ['pass', 'fail']}, 'issues': {'type': ['array', 'null'], 'items': {'type': 'string'}}}}
REVISE_SCHEMA = {'type': 'object', 'required': ['edits'], 'properties': {'edits': {'type': 'array', 'items': {
    'type': 'object', 'required': ['file', 'old', 'new'],
    'properties': {'file': {'type': 'string'}, 'old': {'type': 'string'}, 'new': {'type': 'string'}}}}}}
CHECK_SYSTEM = """You check one code change proposed for a training job, before it is run. The job must stay the same job and
only get faster. `contract` and `checklist` are the rules; `diff` is the change; `mechanism` and `rationale` are what its
author says it does. Report a problem only when it is real: the change does not do what it says; a bug (a wrong variable, a
crash on this job, a broken import, an edit that cannot work); or it breaks a rule (work skipped or moved outside the timed
update, a different workload or data, changed numerics beyond its declared `change_class`, a frozen file or a forbidden
thing touched). Do not judge whether it will be faster: a measured run decides that. Do not ask for style or more safeguards.
Reply with exactly one JSON object and nothing else: {"verdict": "pass" or "fail", "issues": ["<what is wrong and where>"]}."""
REVISE_SYSTEM = """A reviewer found problems in a code change you proposed for a training job. Fix them and keep the change's
purpose. `files` has the current text of the files (the change is NOT applied to them); `diff` is your change; `issues` is what
the reviewer found. Reply with exactly one JSON object and nothing else: {"edits": [{"file": "<path>", "old": "<text that
occurs exactly once in the current file>", "new": "<its replacement>"}]}, the complete change as edits of the current files."""


def _diff(files: dict, patch: dict, context: int = 12) -> str:
    lines = []
    for rel in sorted(patch):
        lines += difflib.unified_diff((files.get(rel) or '').splitlines(), str(patch[rel]).splitlines(), f'a/{rel}', f'b/{rel}',
                                      lineterm='', n=context)
    return '\n'.join(lines)


def _say(engine, msg: str) -> None:
    (getattr(engine, 'log', None) or (lambda m: None))(msg)


def _ask(prop, engine, driver, system: str, payload: dict, schema: dict, role: str, out_cap: int):
    """One call to `driver`, charged to the run; its answer, or None when it cannot be paid or gives no usable answer."""
    s = engine.s
    body = json.dumps(payload, ensure_ascii=False, default=str)
    kw = _allowances(s, driver, system + body, out_cap)
    if kw is None:
        return None
    usd_left = getattr(prop.driver, 'usd_left', None)  # the strategy's own dollar cap (BudgetDriver), as its other calls
    if callable(usd_left):
        left = usd_left()
        if left < MIN_CALL_USD:
            return None
        kw['max_usd'] = min(float(kw.get('max_usd', left)), left)
    res = driver.complete(system, body, schema=schema, role=role, **kw)
    s.charge_llm(res, role)
    return res.action if isinstance(res.action, dict) else None


def _contract(hooks) -> tuple:
    return (hooks.contract_text() if hasattr(hooks, 'contract_text') else '',
            hooks.verifier_checklist() if hasattr(hooks, 'verifier_checklist') else [])


def _check_one(prop, engine, h: dict, files: dict, rounds: int) -> tuple:
    """(the patch that goes on, None when it is dropped; the record): the check / revise rounds of one patch."""
    drv = prop.driver
    author = h.get('author')
    check, revise = drv.checker(author)
    contract, checklist = _contract(prop.hooks)
    current, issues, n = dict(h['patch']), [], 0
    head = {'contract': contract, 'checklist': checklist, 'mechanism': h.get('mechanism'), 'rationale': h.get('rationale'),
            'change_class': h.get('change_class')}
    while n < rounds:
        n += 1
        ans = _ask(prop, engine, check, CHECK_SYSTEM, {**head, 'diff': _diff(files, current)}, CHECK_SCHEMA, 'patch_check', 2048)
        if ans is None:
            return current, {'outcome': 'unchecked', 'rounds': n}
        if ans.get('verdict') == 'pass':
            return current, {'outcome': 'revised_passed' if n > 1 else 'passed', 'rounds': n}
        issues = [str(x)[:400] for x in ans.get('issues') or []] or ['the check failed without saying why']
        if n >= rounds:
            break
        back = _ask(prop, engine, revise, REVISE_SYSTEM, {'mechanism': h.get('mechanism'), 'issues': issues, 'diff': _diff(files, current),
                                                          'files': {rel: files.get(rel, '') for rel in current}},
                    REVISE_SCHEMA, 'patch_revise', prop.code_max_tokens)
        patch, ok = patch_of({'edits': back.get('edits')}, files) if back else (None, False)
        if not (ok and patch):                         # no revision to check: the failed check stands
            break
        current = patch
    return None, {'outcome': 'rejected', 'rounds': n, 'issues': issues}


def check_patches(prop, engine, base: str, items: list) -> list:
    """`items` without the code patches the verifier rejected, the others as checked (a revised patch replaces its original).
    Without the debate's patch check (or with `rounds` 0) `items` come back as they are; one decision per patch and base
    (the review's patches are proposed again until they are tested)."""
    drv = prop.driver
    checker, rounds = getattr(drv, 'checker', None), getattr(drv, 'rounds', 0)
    if not (callable(checker) and rounds and isinstance(items, list)) or not any(isinstance(h, dict) and h.get('patch') for h in items):
        return items
    files = prop.hooks.candidate_files(engine.s.candidates[base].overlay) if hasattr(prop.hooks, 'candidate_files') else {}
    out = []
    for h in items:
        if not (isinstance(h, dict) and h.get('patch')):
            out.append(h)
            continue
        key = hashlib.sha256(json.dumps([base, h['patch']], sort_keys=True, default=str).encode('utf-8')).hexdigest()
        if key not in prop.patch_checked:
            kept, rec = _check_one(prop, engine, h, files, int(rounds))
            prop.patch_checked[key] = kept
            drv.record_check({'mechanism': str(h.get('mechanism'))[:200], 'author': h.get('author') or 'main', **rec})
            _say(engine, f'[search] patch check: {str(h.get("mechanism"))[:100]}: {rec["outcome"]}, {rec["rounds"]} round(s)'
                 + (f': {rec["issues"][0][:200]}' if rec.get('issues') else ''))
            if kept is None:
                prop.dropped.append({'call': prop.calls, 'index': len(prop.dropped), 'mechanism': str(h.get('mechanism'))[:160],
                                     'problems': ['rejected by the patch check: ' + '; '.join(rec['issues'])[:600]]})
        kept = prop.patch_checked[key]
        if kept is not None:
            out.append({**h, 'patch': kept})
    return out
