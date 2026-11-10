"""Requirements in the user's own words: compiled once by the model into what the manifest can already enforce.

    requirements: {text: ["不能改优化器", "不要动 src/loss.py"], on_unsupported: stop}      # in infrarsi.yaml, optional
    --requirements TEXT / --requirements-file F / the startup questions                   # for one run
    infrarsi requirements [MANIFEST]                                                      # compile and show, run nothing

The text goes to the model once (cached under .infrarsi/requirements/) and comes back as items. Each item is checked by the
harness against the manifest, never trusted, and is one of: enforced (a narrower knob, a constraint, a frozen file, a required
check: the existing fields), already (the manifest guarantees it), advisory (told to the model, not checked), or a status that
blocks the run unless the user says otherwise (conflict, setting, widening, objective, unclear, invalid). Requirements can only
narrow the search or add checks: `apply` refuses an overlay that would widen it or touch the verifier. Without text nothing in
this module runs and nothing else changes.
"""
from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import re
from pathlib import Path

from ..ext.common.guards import path_problem
from ..verify import CHECKS, DEFAULT_REQUIRED, PROMOTION_IDS, SCREEN_IDS
from .manifest import ManifestError, Project, evaluate, expr_names

SCHEMA = 'infrarsi-requirements-v2'
BLOCKING = ('conflict', 'setting', 'widening', 'objective', 'unclear', 'invalid')
TEXT_MAX, ITEM_MAX, MAX_ITEMS = 4000, 1000, 16
CACHE_DIR = '.infrarsi/requirements'
HOW_TO_WRITE = ('write them as infrarsi.yaml fields: knob values -> knobs.<k>.choices; relations -> constraints; files -> '
                'code.frozen / code.mutable; checks -> verify.required')
SETTINGS_WITH_FLAGS = {'loss tolerance': '--tolerance / verify.tolerance', 'runs budget': '--screens / search.budget.screens',
                       'wall clock': '--wall-hours', 'LLM dollars': '--budget-usd', 'LLM tokens': '--budget-tokens',
                       'target speed-up': '--target-speedup', 'target MFU': '--target-mfu', 'strategy': '--strategy',
                       'model': '--llm', 'test commands': 'verify.tests', 'eval command': 'verify.eval',
                       'checks to turn off': 'verify.disabled', 'GPUs / hosts': 'hardware / execution'}
COMPILE_SYSTEM = """You translate a user's requirements for InfraRSI, which makes one existing training job faster without changing the
job, into the harness's structured form. InfraRSI's objective is fixed: the time per update of the same job; a verifier
rejects any change that alters the job or moves the loss beyond the tolerance. Requirements can only NARROW what the
search may try, or add guidance. They cannot widen the search, weaken the verifier, change the objective, or set a run
setting that has its own flag (`settings_with_flags`).
The input gives the job: its knobs (current value, allowed values, the mechanism and the update phases each acts on:
comm, compute, optimizer, data, all), the files the search may patch (`code.mutable`), may read, and must never touch
(`code.frozen`), the constraints and workload invariants, the verifier's checks, the launch command, framework and
hardware; and the user's text, in any language, as numbered pieces.
Split the text into separate requirements, one per distinct demand (one piece may hold several; one demand may need two
items, e.g. a knob and a file), and give each exactly one kind:
- pin_knob {knob}: the knob stays at its current value.
- restrict_values {knob, allowed}: the knob may take only these of its allowed values; the current value must be among
  them. For a knob with min/max give {knob, min, max} instead.
- only_knobs {knobs}: only these knobs may change; every other knob stays at its current value. Use the mechanism and
  phases to pick them (e.g. "communication parameters" = the knobs whose phases include "comm").
- constraint {expr}: a relation between knobs as a Python expression over knob names, `gpus` and `nodes`, e.g.
  "tp * pp <= 4" (only ==, !=, <, <=, >, >=, and, or, not, + - * / // % **, min, max, abs, int).
- freeze_file {path}: a file of `code.mutable` must not be patched.
- only_files {paths}: only these files of `code.mutable` may be patched; [] means no code patch at all.
- max_patch_lines {n}: at most n changed lines per patch (lower than now).
- require_check {check}: one of the verifier's checks (by id) becomes required for every candidate.
- advisory {advice}: guidance the model should follow that no field above can enforce (e.g. "do not use bf16" when no
  knob sets the precision; "prefer changes that save memory"). Write `advice` in English, one or two sentences.
- setting {flag}: a run setting with its own flag or field: name it from `settings_with_flags`.
- widening: a demand to try more than the manifest allows (a value not listed, a file not mutable, a check turned off).
- objective: a demand that changes what is optimized (cost, energy, accuracy, "slower is fine").
- unclear: you cannot tell what is meant, or the knob, value, file or check named does not exist: say in `why` what you
  looked for.
Use only knob names, values, paths and check ids that appear in the input; never invent one. Put the user's own words
of each requirement in `text` (verbatim, shortened to 400 characters if longer) and a one-line reason in `why`, in the
user's language. Reply with exactly one JSON object: {"items": [{"text", "kind", "knob", "allowed", "min", "max",
"knobs", "expr", "path", "paths", "n", "check", "advice", "flag", "why"}]} with the fields a kind needs and null for
the others, at most 16 items."""
KINDS = ('pin_knob', 'restrict_values', 'only_knobs', 'constraint', 'freeze_file', 'only_files', 'max_patch_lines', 'require_check',
         'advisory', 'setting', 'widening', 'objective', 'unclear')
COMPILE_SCHEMA = {'type': 'object', 'required': ['items'], 'properties': {'items': {'type': 'array', 'items': {
    'type': 'object', 'required': ['text', 'kind'], 'properties': {
        'text': {'type': 'string'}, 'kind': {'type': 'string', 'enum': list(KINDS)},
        'knob': {'type': ['string', 'null']}, 'allowed': {'type': ['array', 'null']},
        'min': {'type': ['number', 'null']}, 'max': {'type': ['number', 'null']},
        'knobs': {'type': ['array', 'null'], 'items': {'type': 'string'}}, 'expr': {'type': ['string', 'null']},
        'path': {'type': ['string', 'null']}, 'paths': {'type': ['array', 'null'], 'items': {'type': 'string'}},
        'n': {'type': ['integer', 'null']}, 'check': {'type': ['string', 'null']}, 'advice': {'type': ['string', 'null']},
        'flag': {'type': ['string', 'null']}, 'why': {'type': ['string', 'null']}}}}}}
GUIDANCE_HEAD = ('User requirements for this run (the enforced ones are already applied to the knobs, files and constraints you '
                 'see; follow their intent too; the guidance ones are not checked by the verifier):')


class RequirementsError(RuntimeError):
    """The requirements could not be compiled or honoured; the run does not start."""


# ---------------------------------------------------------------- the text
def manifest_texts(project: Project) -> list[str]:
    block = project.raw.get('requirements') or {}
    text = block.get('text') if isinstance(block, dict) else None
    return [text] if isinstance(text, str) else list(text or [])


def normalize(texts) -> list[str]:
    """The pieces of text, stripped, spaces collapsed, empty ones left out; ValueError when they are too long."""
    out = [re.sub(r'\s+', ' ', str(t)).strip() for t in texts or []]
    out = [t for t in out if t]
    if any(len(t) > ITEM_MAX for t in out) or sum(len(t) for t in out) > TEXT_MAX:
        raise ValueError(f'requirements: at most {TEXT_MAX} characters in total ({ITEM_MAX} per item)')
    return out


def digest(project: Project, texts) -> str:
    """What the compiled result depends on (not the model that compiled it: its answer is checked by this module)."""
    key = {'schema': SCHEMA, 'payload': _payload(project, normalize(texts))}      # everything the compile reads
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode('utf-8')).hexdigest()[:16]


# ---------------------------------------------------------------- the model call
def _knob_table(project: Project) -> dict:
    from ..experience.registry import mechanism_phases
    from ..experience.store import labels
    ref, marks = project.reference_config(), labels(project.knobs, project.raw.get('framework'))
    table = {}
    for name, k in project.knobs.items():
        target = next((t for t in ('path', 'flag', 'override', 'const', 'env') if t in k), None)
        mech = getattr(marks.get(name), 'mechanism', None)
        table[name] = {'type': k.get('type', 'str'), 'target': target, 'current': ref.get(name), 'choices': _choices(k),
                       'description': k.get('description'), 'mechanism': mech, 'phases': mechanism_phases(mech)}
        if 'min' in k or 'max' in k:
            table[name].update(min=k.get('min'), max=k.get('max'))
    return table


def _payload(project: Project, texts: list[str]) -> dict:
    raw, code = project.raw, project.raw['code']
    return {'project': {'name': project.name, 'framework': raw.get('framework'), 'gpus': project.gpus, 'nodes': project.nodes,
                        'launch_command': raw['launch']['command']},
            'knobs': _knob_table(project), 'constraints': raw.get('constraints') or [],
            'workload_invariants': raw['workload'].get('invariants') or {},
            'code': {'mutable': project.files(code['mutable']), 'frozen': project.files(code['frozen']),
                     'readable': project.files(code['readable']), 'max_patch_lines': code['max_patch_lines']},
            'verify': {'tolerance': raw['verify']['tolerance'], 'required': list(raw['verify'].get('required') or DEFAULT_REQUIRED),
                       'disabled': list(raw['verify'].get('disabled') or []),
                       'checks': {c.id: c.title for c in CHECKS if c.family != 'info'}},
            'settings_with_flags': SETTINGS_WITH_FLAGS, 'text': [{'i': i + 1, 'text': t} for i, t in enumerate(texts)]}


def _call_dir(root: Path, name: str, n: int) -> Path | None:
    d = root / name / f'{_dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S-%f")}-{n}'
    try:
        d.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return d


def compile(project: Project, texts, *, driver, log=None, refresh: bool = False, cached_only: bool = False, call_dir=None) -> dict | None:
    """The record of `texts` (see the module docstring and `_record`), from the cache when it has it; None without text, or
    with `cached_only` and no cached compile. RequirementsError when the model is needed and cannot answer."""
    texts = normalize(texts)
    if not texts:
        return None
    say = log or (lambda m: None)
    key = digest(project, texts)
    cache = project.root / CACHE_DIR
    path = cache / f'{key}.json'
    if not refresh:
        try:
            hit = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(hit, dict) and hit.get('schema') == SCHEMA and hit.get('digest') == key:
                hit['compile'] = {**(hit.get('compile') or {}), 'cached': True}
                return hit
            say('[infrarsi] requirements: the cached compile could not be read; compiling again')
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            say('[infrarsi] requirements: the cached compile could not be read; compiling again')
    if cached_only:
        return None
    if driver is None:
        raise RequirementsError('requirements need a model once to be read (one call, cached under .infrarsi/requirements/): run '
                                'with --llm <backend>, or `infrarsi requirements --llm <backend>` first, or ' + HOW_TO_WRITE)
    payload = _payload(project, texts)
    tokens, usd, calls, error = 0, 0.0, 0, None
    res = None
    for n in (1, 2):
        body = {**payload, **({'previous_reply_error': error} if error else {})}
        res = driver.complete(COMPILE_SYSTEM, json.dumps(body, ensure_ascii=False, default=str), schema=COMPILE_SCHEMA, role='requirements',
                              call_dir=call_dir or _call_dir(cache, key, n), max_output_tokens=4096)
        calls += 1
        tokens += res.llm_tokens or 0
        usd += res.llm_usd or 0.0
        if isinstance(res.action, dict) and len(res.action.get('items') or []) <= MAX_ITEMS:
            break
        error = str(res.error or (f'more than {MAX_ITEMS} items' if isinstance(res.action, dict) else 'not one JSON object'))[:500]
        res = None
    if res is None or not isinstance(res.action, dict):
        raise RequirementsError(f'the model could not read the requirements ({error}); run again, or ' + HOW_TO_WRITE)
    record = _record(project, texts, key, res.action.get('items') or [])
    record['compile'] = {'cached': False, 'utc': _dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                         'backend': str(getattr(driver, 'kind', '') or ''), 'model': str(getattr(res, 'model', '') or ''),
                         'calls': calls, 'llm_tokens': tokens, 'llm_usd': round(usd, 6), 'call_dir': f'{CACHE_DIR}/{key}'}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    except OSError:
        pass                                           # a read-only project: compiled again next time
    return record


# ---------------------------------------------------------------- the model's answer, checked
def _choices(k: dict) -> list | None:
    return list(k['choices']) if k.get('choices') else ([False, True] if k.get('type') == 'bool' else None)


def _matching(allowed, choices: list) -> list:
    """The values of `choices` named in `allowed` (a value in another spelling counts: "4" for 4, "True" for True)."""
    names = {str(a).strip().lower() for a in allowed}
    return [c for c in choices if str(c).strip().lower() in names]


def _knobs_change(narrow: dict, name: str, effect: dict) -> None:
    narrow.setdefault(name, []).append(effect)


def _pin(project: Project, name: str, ref: dict) -> dict:
    return {'choices': [ref.get(name)]}


def _already_pinned(k: dict, cur) -> bool:
    """The knob's one allowed value is the value the job runs with (an unset knob with one value could still be set to it)."""
    return bool(k.get('choices')) and len(k['choices']) == 1 and k['choices'][0] == cur


def _validate(project: Project, it: dict, ref: dict, M: list, F: list) -> dict:
    """One item of the model's answer as (status, effect, words); anything it names that is not in the job makes it invalid."""
    kind, K = it.get('kind'), project.knobs

    def bad(words: str) -> dict:
        return {'status': 'invalid', 'effect': None, 'words': words}

    def knob_named(name):
        return None if name in K else f'{name!r} is not a knob of this job'
    if kind == 'pin_knob':
        name = it.get('knob')
        if knob_named(name):
            return bad(knob_named(name))
        if _already_pinned(K[name], ref.get(name)):
            return {'status': 'already', 'effect': {'hold': [name]}, 'words': f'knob {name} has only one allowed value already'}
        unset = ' (never set: its current value is the framework\'s default)' if ref.get(name) is None else ''
        return {'status': 'enforced', 'effect': {'knobs': {name: _pin(project, name, ref)}},
                'words': f'knob {name} stays {ref.get(name)!r}{unset} (its only allowed value now); a proposal that moves it is refused before any run'}
    if kind == 'restrict_values':
        name = it.get('knob')
        if knob_named(name):
            return bad(knob_named(name))
        k, cur = K[name], ref.get(name)
        if it.get('allowed') is not None:
            opts = _choices(k)
            if opts is None:
                return bad(f'knob {name} has a range, not a list of values: give min and max')
            kept = _matching(it['allowed'], opts)
            if not kept:
                return bad(f'none of {it["allowed"]!r} is a value of knob {name} ({opts!r})')
            if set(map(str, it['allowed'])) - set(map(str, kept)):
                missing = [a for a in it['allowed'] if str(a).strip().lower() not in {str(c).lower() for c in kept}]
                if missing:
                    return bad(f'{missing!r} is not a value of knob {name} ({opts!r})')
            if cur not in kept:
                return {'status': 'conflict', 'effect': None,
                        'words': f'your job runs with {name}={cur!r} now; a requirement cannot change the reference: change it where the job is configured and run again'}
            if len(kept) == len(opts):                  # (a plain on/off knob is not held: asking for both values says nothing)
                return {'status': 'already', 'effect': {'hold': [name]} if k.get('choices') else None,
                        'words': f'knob {name} takes only those values already'}
            return {'status': 'enforced', 'effect': {'knobs': {name: {'choices': kept}}}, 'words': f'knob {name} may take only {kept!r}'}
        lo, hi, cmin, cmax = it.get('min'), it.get('max'), k.get('min'), k.get('max')
        if lo is None or hi is None or cmin is None or cmax is None or not cmin <= lo <= hi <= cmax:
            return bad(f'knob {name}: give a range within {cmin}..{cmax}')
        if cur is None:
            return {'status': 'conflict', 'effect': None, 'words': f'your job leaves {name} unset; a requirement cannot give it a range'}
        if not lo <= cur <= hi:
            return {'status': 'conflict', 'effect': None, 'words': f'your job runs with {name}={cur!r} now, outside {lo}..{hi}; a requirement cannot change the reference'}
        if lo == cmin and hi == cmax:
            return {'status': 'already', 'effect': {'hold': [name]}, 'words': f'knob {name} is in that range already'}
        return {'status': 'enforced', 'effect': {'knobs': {name: {'min': lo, 'max': hi}}}, 'words': f'knob {name} may take only {lo}..{hi}'}
    if kind == 'only_knobs':
        names = list(it.get('knobs') or [])
        wrong = [n for n in names if n not in K]
        if wrong or not names:
            return bad(f'{wrong or "no knob"} is not a knob of this job' if wrong else 'no knob named')
        pins = {n: _pin(project, n, ref) for n, k in K.items() if n not in names and not _already_pinned(k, ref.get(n))}
        hold = [n for n, k in K.items() if n not in names and _already_pinned(k, ref.get(n))]
        if not pins:
            return {'status': 'already', 'effect': {'hold': hold}, 'words': f'only {", ".join(names)} can change already'}
        held = ', '.join(f'{n}={ref.get(n)!r}' for n in pins)
        return {'status': 'enforced', 'effect': {'knobs': pins, 'hold': hold},
                'words': f'only {", ".join(names)} may change; {len(pins)} other knob(s) stay at their current values: {held}'}
    if kind == 'constraint':
        expr = it.get('expr')
        if not isinstance(expr, str) or not expr.strip():
            return bad('no expression given')
        unknown = expr_names(expr) - set(K) - {'gpus', 'nodes'}
        if unknown:
            return bad(f'{sorted(unknown)} is not a knob of this job')
        try:
            ok = evaluate(expr, project.names(ref))
        except (ManifestError, ArithmeticError, TypeError, ValueError) as exc:
            return bad(f'the expression {expr!r} cannot be evaluated: {exc}')
        if not ok:
            return {'status': 'conflict', 'effect': None, 'words': f'your current configuration breaks {expr}: a requirement cannot change the reference'}
        return {'status': 'enforced', 'effect': {'constraints': [expr]}, 'words': f'candidates must satisfy {expr}'}
    if kind == 'freeze_file':
        path = it.get('path')
        why = path_problem(path) or ('wildcards are not paths' if set(str(path)) & set('*?[]') else None)
        if why:
            return bad(f'{path!r}: {why}')
        if not (project.root / path).is_file() or project.excluded(path):
            return bad(f'{path}: no such file under the project (or it is excluded)')
        if path in F or path not in M:
            return {'status': 'already', 'effect': None, 'words': f'{path} cannot be patched already: no patch could touch it'}
        return {'status': 'enforced', 'effect': {'freeze': [path]}, 'words': f'{path} is frozen (its hash is checked on every run) and no longer shown as editable'}
    if kind == 'only_files':
        paths = it.get('paths')
        if not isinstance(paths, list):
            return bad('no list of files given')
        wrong = [p for p in paths if path_problem(p) or set(str(p)) & set('*?[]')]
        if wrong:
            return bad(f'{wrong!r}: not plain relative paths')
        kept = [m for m in M if m in paths]
        ignored = [p for p in paths if p not in M]
        note = f'; ignored, not in code.mutable (add it there yourself): {", ".join(ignored)}' if ignored else ''
        if kept == M and sorted(project.raw['code']['mutable'] or []) == M:    # (a glob would still let new files in)
            return {'status': 'already', 'effect': None, 'words': 'those are the only files the search may patch already' + note}
        return {'status': 'enforced', 'effect': {'mutable': kept},
                'words': (f'only {", ".join(kept)} may be patched' if kept else 'no code patch (knobs only)') + note}
    if kind == 'max_patch_lines':
        n, cur = it.get('n'), int(project.raw['code'].get('max_patch_lines') or 300)
        if type(n) is not int or n < 1:
            return bad('give a positive number of lines')
        if n >= cur:
            return {'status': 'already', 'effect': None, 'words': f'patches are limited to {cur} changed lines already'}
        return {'status': 'enforced', 'effect': {'max_patch_lines': n}, 'words': f'at most {n} changed lines per patch'}
    if kind == 'require_check':
        check, v = it.get('check'), project.raw['verify']
        if check not in SCREEN_IDS + PROMOTION_IDS or next(c for c in CHECKS if c.id == check).family == 'info':
            return bad(f'{check!r} is not a check of the verifier')
        if check in (v.get('disabled') or []):
            return {'status': 'widening', 'effect': None, 'words': f'the check {check} is turned off in verify.disabled: a requirement does not turn it on, edit infrarsi.yaml'}
        if check in (v.get('required') or DEFAULT_REQUIRED):
            return {'status': 'already', 'effect': None, 'words': f'{check} is required already'}
        return {'status': 'enforced', 'effect': {'required': [check]}, 'words': f'{check} is required: a candidate whose logs cannot show it is rejected'}
    if kind == 'advisory':
        advice = str(it.get('advice') or '').strip()
        if not advice:
            return bad('no advice given')
        return {'status': 'advisory', 'effect': None, 'words': f'the model is told "{advice}"; the verifier does NOT check it', 'advice': advice}
    if kind == 'setting':
        flag = SETTINGS_WITH_FLAGS.get(str(it.get('flag')), it.get('flag')) or 'the setting named above'
        return {'status': 'setting', 'effect': None, 'words': f'this is a run setting with its own flag: use {flag}'}
    if kind == 'widening':
        return {'status': 'widening', 'effect': None, 'words': 'a requirement can only narrow the search; to widen it edit infrarsi.yaml'}
    if kind == 'objective':
        return {'status': 'objective', 'effect': None, 'words': 'InfraRSI optimizes one objective, the time per update of the same job; a requirement cannot change it'}
    return {'status': 'unclear', 'effect': None, 'words': f'{it.get("why") or "not understood"}; name the knob or file as infrarsi.yaml names it, or write the field yourself'}


def _record(project: Project, texts: list[str], key: str, answer) -> dict:
    ref = project.reference_config()
    code = project.raw['code']
    M, F = project.files(code['mutable']), project.files(code['frozen'])
    items = []
    for n, raw in enumerate(a for a in answer if isinstance(a, dict)):
        items.append({'i': n + 1, 'text': str(raw.get('text') or '')[:400], 'kind': raw.get('kind'),
                      'why': str(raw.get('why') or '')[:300], **_validate(project, raw, ref, M, F)})
    if not items:
        items.append({'i': 1, 'text': ' '.join(texts)[:400], 'kind': 'unclear', 'why': '', 'status': 'unclear', 'effect': None,
                      'words': 'the model found no requirement in this text; name the knob or file as infrarsi.yaml names it'})
    _intersect(project, items, ref)
    overlay = _overlay(project, items, M)
    counts = {'enforced': sum(i['status'] == 'enforced' for i in items), 'already': sum(i['status'] == 'already' for i in items),
              'advisory': sum(i['status'] == 'advisory' for i in items), 'blocking': sum(i['status'] in BLOCKING for i in items)}
    return {'schema': SCHEMA, 'texts': texts, 'digest': key, 'items': items, 'overlay': overlay, 'guidance': _guidance(items),
            'counts': counts, 'accepted_partial': False}


def _intersect(project: Project, items: list, ref: dict) -> None:
    """Items that narrow one knob all hold: their allowed values are intersected; when nothing or not the current value is left,
    the items contradict each other and are conflicts."""
    for name in project.knobs:
        hits = [i for i in items if i['status'] == 'enforced' and ((i['effect'] or {}).get('knobs') or {}).get(name) is not None]
        if len(hits) < 2:
            continue
        k, cur = project.knobs[name], ref.get(name)
        opts = None
        lo, hi = k.get('min'), k.get('max')
        for i in hits:
            e = i['effect']['knobs'][name]
            if 'choices' in e:
                opts = [c for c in (opts if opts is not None else e['choices']) if c in e['choices']]
            else:
                lo, hi = max(lo, e['min']), min(hi, e['max'])
        ok = (cur in opts) if opts is not None else (cur is not None and lo <= cur <= hi)
        for i in hits:
            if not ok:
                i.update(status='conflict', effect=None, words=f'it contradicts another requirement on knob {name}: together they leave no value that keeps your job\'s current {name}={cur!r}')
            else:
                merged = {'choices': opts} if opts is not None else {'min': lo, 'max': hi}
                i['effect'] = {**i['effect'], 'knobs': {**i['effect']['knobs'], name: merged}}


def _overlay(project: Project, items: list, M: list) -> dict:
    """The effects of the enforced items, merged; None for a field no item touches ([] for `code.mutable` means no patches)."""
    knobs, constraints, frozen, mutable, lines, required = {}, [], [], None, None, None
    hold = sorted({n for i in items if i['status'] in ('enforced', 'already') for n in (i['effect'] or {}).get('hold') or ()})
    for i in items:
        if i['status'] != 'enforced':
            continue
        e = i['effect']
        for name, spec in (e.get('knobs') or {}).items():
            knobs[name] = {**knobs.get(name, {}), **spec}
        constraints += e.get('constraints') or []
        frozen += e.get('freeze') or []
        if 'mutable' in e:
            mutable = e['mutable'] if mutable is None else [m for m in mutable if m in e['mutable']]
        if e.get('max_patch_lines') is not None:
            lines = min(e['max_patch_lines'], lines) if lines is not None else e['max_patch_lines']
        if e.get('required'):
            required = sorted(set(required or []) | set(e['required']))
    if frozen:
        mutable = [m for m in (M if mutable is None else mutable) if m not in frozen]
    return {'knobs': knobs, 'hold': hold, 'constraints': constraints,
            'code': {'mutable': mutable, 'frozen': sorted(set(frozen)), 'max_patch_lines': lines}, 'verify': {'required': required}}


def _guidance(items: list) -> list[str]:
    """What the model is told with every call: the items that were accepted, never the ones that were not (it must not read
    them as enforced)."""
    lines = []
    for i in items:
        if i['status'] in ('enforced', 'already'):
            lines.append(f'- "{i["text"]}": {i["words"]}')
        elif i['status'] == 'advisory':
            lines.append(f'- "{i["text"]}" (guidance only): {i.get("advice")}')
    return ([GUIDANCE_HEAD] + lines) if lines else []


# ---------------------------------------------------------------- what the record does
def blocking(record: dict) -> list[dict]:
    return [i for i in record['items'] if i['status'] in BLOCKING]


def guidance(record: dict | None) -> list[str]:
    return list((record or {}).get('guidance') or [])


def explain(record: dict) -> list[str]:
    """The console's account of a record: one line per item with what it meant."""
    c, n = record['counts'], len(record['items'])
    src = record.get('source') or {}
    where = ', '.join(f'{name} ({len(v)})' for name, v in (('infrarsi.yaml', src.get('manifest')), ('--requirements', src.get('cli')),
                                                         ('the questions', src.get('interactive'))) if v)
    lines = [f'[infrarsi] requirements: {n} item(s)' + (f' from {where}' if where else '') + ': ' + (
        f'{c["enforced"]} enforced, {c["already"]} already guaranteed, {c["advisory"]} guidance only, {c["blocking"]} cannot be honoured')]
    words = {'enforced': 'enforced', 'already': 'already guaranteed', 'advisory': 'guidance only', 'conflict': 'conflict',
             'setting': 'not applied', 'widening': 'not applied', 'objective': 'not supported', 'unclear': 'not understood',
             'invalid': 'not understood'}
    for i in record['items']:
        lines.append(f'  {i["i"]}. "{i["text"]}" -> {words.get(i["status"], i["status"])}: {i["words"]}')
    return lines


def decide(record: dict, *, interactive=None, partial: bool = False) -> dict | None:
    """The record to run with; None when the user wants to edit the text. `interactive(record)` -> 'go' | 'skip' | 'edit' | 'quit'
    (Enter never drops a requirement: with items that cannot be honoured it quits). Without it: items that cannot be honoured
    stop the run unless `partial`."""
    block = blocking(record)
    if interactive is not None:
        choice = interactive(record)
        if choice == 'edit':
            return None
        if choice == 'quit' or (block and choice == 'go'):
            raise RequirementsError('stopped at the requirements: nothing was run')
        return {**record, 'accepted_partial': bool(block)}
    ids = ', '.join(str(i['i']) for i in block)
    if block and not partial:
        raise RequirementsError(f'requirement(s) {ids} cannot be honoured; nothing was run. Fix the text or the manifest, or run with '
                                '--requirements-partial to search without them (the report will say so)')
    return {**record, 'accepted_partial': bool(block)}


def _monotone(project: Project, overlay: dict) -> list[str]:
    """Why the overlay would widen the search or touch the verifier; empty when it only narrows (whatever the model answered)."""
    wrong, code = [], project.raw['code']
    ref = project.reference_config()
    for name, spec in (overlay.get('knobs') or {}).items():
        k = project.knobs.get(name)
        if k is None:
            wrong.append(f'unknown knob {name}')
            continue
        if 'choices' in spec:
            old = _choices(k)
            if old is not None and any(v not in old and v != ref.get(name) for v in spec['choices']):
                wrong.append(f'knob {name}: a value outside its choices')
            if any(v is not None and ((k.get('min') is not None and v < k['min']) or (k.get('max') is not None and v > k['max']))
                   for v in spec['choices']):
                wrong.append(f'knob {name}: a value outside its range')
            if ref.get(name) not in spec['choices']:
                wrong.append(f'knob {name}: the current value is left out')
        if 'min' in spec and (spec['min'] < k.get('min', spec['min']) or spec['max'] > k.get('max', spec['max'])):
            wrong.append(f'knob {name}: a wider range')
    off = set((overlay.get('verify') or {}).get('required') or ()) & set(project.raw['verify'].get('disabled') or ())
    if off:
        wrong.append(f'verify: {", ".join(sorted(off))} is turned off in verify.disabled now, so it cannot be required')
    c = overlay.get('code') or {}
    M, F = project.files(code['mutable']), set(project.files(code['frozen']))
    if c.get('mutable') is not None and set(c['mutable']) - set(M):
        wrong.append('code.mutable: files it did not have')
    if c.get('max_patch_lines') is not None and c['max_patch_lines'] > int(code.get('max_patch_lines') or 300):
        wrong.append('code.max_patch_lines: more lines')
    if not F <= (F | set(c.get('frozen') or ())):
        wrong.append('code.frozen: files taken out')
    return wrong


def apply(project: Project, record: dict | None) -> Project:
    """The project narrowed by the record's overlay (a new Project; `project` is left as it is)."""
    if record is None:
        return project
    overlay = record.get('overlay') or {}
    wrong = _monotone(project, overlay)
    if wrong:
        _drop_cache(project, record)
        raise RequirementsError(f'the compiled requirements would widen the search ({"; ".join(wrong)}); this is a bug, the cached compile was discarded')
    raw = copy.deepcopy(project.raw)
    for name in overlay.get('hold') or ():              # the values asked for are all it has: still, it cannot be left unset
        raw['knobs'][name]['narrowed'] = True
    for name, spec in (overlay.get('knobs') or {}).items():
        k = raw['knobs'][name]
        k['narrowed'] = True                            # (the strategies and the null rule read it; only a requirement sets it)
        if 'choices' in spec:
            k['choices'] = list(spec['choices'])
            k.pop('min', None)
            k.pop('max', None)
        else:
            k['min'], k['max'] = spec['min'], spec['max']
    raw['constraints'] = list(raw.get('constraints') or []) + list(overlay.get('constraints') or [])
    c = overlay.get('code') or {}
    if c.get('mutable') is not None:
        raw['code']['mutable'] = list(c['mutable'])
    raw['code']['frozen'] = list(raw['code'].get('frozen') or []) + list(c.get('frozen') or [])
    if c.get('max_patch_lines') is not None:
        raw['code']['max_patch_lines'] = int(c['max_patch_lines'])
    v = overlay.get('verify') or {}
    if v.get('required'):
        raw['verify']['required'] = sorted(set(raw['verify'].get('required') or DEFAULT_REQUIRED) | set(v['required']))
    try:
        return Project(raw, project.path)
    except ManifestError as exc:
        _drop_cache(project, record)
        raise RequirementsError(f'the compiled requirements make the manifest invalid: {exc}') from exc


def _drop_cache(project: Project, record: dict) -> None:
    try:
        (project.root / CACHE_DIR / f'{record.get("digest")}.json').unlink()
    except OSError:
        pass


def append_texts(manifest_path, texts) -> None:
    """Keep `texts` in infrarsi.yaml: appended at its end (a block already there is the user's: its YAML goes in the error)."""
    import yaml
    path = Path(manifest_path)
    text = path.read_text(encoding='utf-8')
    block = yaml.safe_dump({'requirements': {'text': list(texts)}}, allow_unicode=True, sort_keys=False)
    if 'requirements' in (yaml.safe_load(text) or {}):
        raise RequirementsError(f'{path} has a requirements block already; add these to it:\n{block}')
    path.write_text(text + ('' if text.endswith('\n') else '\n') + block, encoding='utf-8')
