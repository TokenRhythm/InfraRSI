"""A patch's self-proof, read from its probe (FABLE_HLS3_FIXES_R2 F10, R3 F13 / F21) - for any strategy that probes a patch
before its full run and opts in (its proposer's `self_proof` / `deferred` features: search/proposers.FEATURES).

A patch written to the proof rule (proposers.PROOF_RULE, PROOF_DEFER_RULE) prints, per code site, `[infrarsi-patch]
<site>: found X -> set Y` when it installs, `<site>: active` once the new path has run, and `<site>: deferred (<why>)` once
when its replacement hands a call back to the original. project.signals reads them per site (add_proof): the display
lines are cut at 300 characters, the evidence reads each line whole (`patch_proof_whole`, when that says more: F13). The
engine gives a Proposal's `gate` the probe's detailed trial row (agent.search.trial(detail=True)): its notes, its proof
lines and their evidence.

    check(engine, base, row, patch, h)   for a Proposal gate: None when the patch proved it took effect, else why
    proof_reason(row, base_events, notes, claim, h)   the same from its parts: (why, the note it rests on)
    proof_of(ev, whole) / proof_verdict(proof, inherited) / note_reason(cur, old, note) / parsed / overlong
    patch_claim(files, patch, h)         what a patch says it fixes (its site, evidence note, the names its code sets)
    claimed_notes / targeted_notes       the job-log notes a patch names explicitly / by a code name

hls3 (search/hls3.py) uses them behind its components hls3_self_proof (F10), hls3_scan_whole (F13), hls3_deferred (F21).
"""
from __future__ import annotations

import re

from ..project.signals import PROOF_LINE_MAX, proof_evidence

PROBE_STEPS_MAX = 8             # a probe's updates at most (`active_from_update` is supported up to it)
QUOTE_MIN = 20                  # F10a: a note-removal claim quotes at least this many characters of the note
CODE_NAME = re.compile(r'`([^`\s]{3,80})`|\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+|[A-Za-z0-9]*_[A-Za-z0-9_]+)\b')
SET_NAME = re.compile(r'([A-Za-z_][\w.]*(?:\[[^\]]*\])*)\s*=(?!=)|\bdef\s+([A-Za-z_]\w*)|setattr\(\s*[^,]+,\s*[\'"]([A-Za-z_]\w*)')


# ---------------------------------------------------------------- a run's evidence in the search
def events_of(engine, cid) -> dict:
    """The events a candidate's run left in its receipt (signals.scan_events), {} without one."""
    c = engine.s.candidates.get(cid) if cid else None
    rec = getattr(c, 'receipt', None) or {}
    ev = rec.get('evidence') if isinstance(rec.get('evidence'), dict) else {}
    return ev.get('events') if isinstance(ev.get('events'), dict) else {}


def notes_of(engine, cid) -> list:
    """The job-log notes of a candidate's run (its receipt), else the reference's."""
    for x in (cid, 'c0'):
        notes = events_of(engine, x).get('perf_notes')
        if notes:
            return list(notes)
    return []


def files_of(engine, cid) -> dict:
    """The job's files as a candidate runs them (its overlay over the project's), for a patch's claim."""
    c = engine.s.candidates[cid]
    hooks = engine.s.hooks
    return hooks.candidate_files(c.overlay) if hooks is not None and hasattr(hooks, 'candidate_files') else \
        dict(c.overlay or {})


# ---------------------------------------------------------------- what a patch claims
def code_names(text) -> set:
    """F10a: the code names a text gives - backticked words, dotted names, names with an underscore (6+ characters)."""
    out = set()
    for m in CODE_NAME.finditer(str(text or '')):
        w = (m.group(1) or m.group(2) or '').strip('`.()')
        if len(w) >= 6 and ('_' in w or '.' in w or m.group(1)):
            out.add(w.lower())
            out.update(x.lower() for x in w.split('.') if len(x) >= 6 and '_' in x)   # (mod.name_x: name_x too)
    return out


def set_names(lines) -> str:
    """F10a: the names a patch's code sets - assigned, passed as a keyword, set as an attribute, defined (comments out)."""
    out = []
    for ln in lines or []:
        for m in SET_NAME.finditer(str(ln).split('#', 1)[0]):
            out.append(next(g for g in m.groups() if g))
    return ' '.join(out)


def patch_claim(files: dict, patch: dict | None, h: dict | None) -> str:
    """F10a: what a patch says it fixes - its site and evidence note - and the names the code lines it adds set (a name
    it only calls is no claim). `files`: its base's files; `h`: its hypothesis (`site`, `evidence_note`)."""
    words = ' '.join(str((h or {}).get(k) or '') for k in ('site', 'evidence_note'))
    added = []
    for f, text in (patch or {}).items():
        have = set(str(files.get(f) or '').splitlines())
        added += [ln for ln in str(text or '').splitlines() if ln not in have]
    return words + '\n' + set_names(added[:400])


def targeted_notes(claim: str, notes) -> list:
    """F10a: the job-log notes a patch's claim (its site, evidence note and the names its code sets) names by a code
    name."""
    names = code_names(claim)
    return [n for n in notes or [] if names & code_names(n)]


def _quote_key(text) -> str:
    return ' '.join(str(text or '').lower().split())


def claimed_notes(evidence_note, notes) -> list:
    """F10a: the job-log note the proposer named as the one its patch removes: the one note `evidence_note` quotes - a
    passage of QUOTE_MIN characters or more of it, or the whole note - and no other. A paraphrase or a shared code name
    names none (Codex HLS4R2 r1 / r2: only an explicitly identified note can veto; the others are feedback)."""
    if not isinstance(evidence_note, str):
        return []
    e = _quote_key(evidence_note)
    if len(e) < QUOTE_MIN:
        return []
    quoted = [n for n in notes or [] if e in _quote_key(n) or len(_quote_key(n)) >= QUOTE_MIN and _quote_key(n) in e]
    return quoted if len(quoted) == 1 else []


def _note_key(note) -> str:
    return ' '.join(str(note or '').lower().split())[:160]


def supported_active(h) -> int | None:
    """A declared `active_from_update` honoured - 1..PROBE_STEPS_MAX - else None: a value out of range is absent, never
    an exemption (Codex HLS4R2 r2)."""
    v = (h or {}).get('active_from_update')
    return v if isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= PROBE_STEPS_MAX else None


# ---------------------------------------------------------------- what its run printed
def _evidence(x) -> dict:
    """Per-site proof evidence (signals.add_proof), from the evidence itself or from raw lines."""
    return x if isinstance(x, dict) else proof_evidence(x or [])


SHOW = 120                      # a found / set value as reasons and feedback show it (compared whole: canonical_pair)


def _show(x) -> str:
    x = str(x or '')
    return x if len(x) <= SHOW else x[:SHOW] + '...'


def proof_of(ev: dict, whole: bool = True) -> dict:
    """A run's per-site proof evidence (its events, or a probe's detailed trial row): its own - with `whole` (F13) its
    lines read whole when that says more than the display lines - else from its display lines."""
    pw = ev.get('patch_proof_whole') if whole else None
    pp = pw if isinstance(pw, dict) and pw else ev.get('patch_proof')
    return pp if isinstance(pp, dict) and pp else proof_evidence(ev.get('patch_lines') or [], whole=whole)


def new_installs(proof, inherited=None) -> list:
    """F10b: the installation records (site, found, set, at) of this run that its base's run did not print - the new
    patch's; an installation only an inherited patch printed is not."""
    old = {(s.lower(), f.lower(), v.lower()) for s, r in _evidence(inherited).items() for f, v, *_ in r.get('installs') or []}
    return [(s, f, v, (rest or [None])[0]) for s, r in _evidence(proof).items() for f, v, *rest in r.get('installs') or []
            if (s.lower(), f.lower(), v.lower()) not in old]


def proof_verdict(proof, inherited=None, *, unseen_active: bool = False, deferred: bool = False) -> str | None:
    """F10b: why a patch's proof does not show that it took effect - no installation record of its own (a `found -> set`
    line its base's run did not print), found equal to set (it set what was there), or a site of a new installation with
    no `active` line in this run (the patched code did not run; not held against it when its first run is declared
    beyond the probe: `unseen_active`); None when it does. A site's `active` line counts whatever the base printed (Codex
    HLS4R2 r1: the same text at the same site). `proof`, `inherited`: per-site evidence or raw lines. `deferred` (F21): a
    site without `active` that printed `deferred (<why>)` is reported with that reason."""
    cur = _evidence(proof)
    new = new_installs(cur, inherited)
    if not new:
        return ('it printed no `[infrarsi-patch] <site>: found ... -> set ...` line of its own'
                + (' (its installation lines are its base\'s)' if any(r.get('installs') for r in cur.values()) else ''))
    for s, f, v, _ in new:
        if f.lower() == v.lower():                     # (whole canonical values: a display cuts them, never this)
            return f'{s}: found {_show(f)} -> set {_show(v)}: it set what was already there'
    for s in dict.fromkeys(s for s, *_ in new):
        if cur[s].get('active_at') is None and not unseen_active:
            d = cur[s].get('deferred') if deferred else None
            if d:
                return (f'{s}: `deferred ({d[0] or "no reason given"})` and no `active` line - the replacement handed '
                        'every call back to the original')
            return f'{s}: no `active` line - the patched code never ran within the probe'
    return None


def note_reason(cur: dict, old: dict, note: str, *, deferred: bool = True) -> str:
    """F10: why a patch is inert when the job-log note it names is still reported after it installed - with `deferred`
    (F21) either of the two explanations, the deferral when its replacement ran at least once, a `deferred` line's
    reason; without, round 2's sentence."""
    head = f'the job-log note it names is still reported after it installed: "{note[:200]}" - '
    if not deferred:
        return head + 'the code it replaced is not the code the job runs'
    sites = list(dict.fromkeys(s for s, *_ in new_installs(cur, old)))
    d = next((cur[s]['deferred'] for s in sites if (cur.get(s) or {}).get('deferred')), None)
    said = f' (it printed `deferred ({d[0] or "no reason given"})`)' if d else ''
    if any((cur.get(s) or {}).get('active_at') is not None for s in sites):
        return head + 'the replacement ran at least once and the note was still reported: it defers on some calls' + said
    return head + 'either the replaced object is not the one called, or the replacement defers to the original' + said


def overlong(proof: dict) -> str:
    """F13: a site whose line was longer than a display line, and the found / set pair it was read as."""
    for s, r in proof.items():
        if (r.get('longest') or 0) > PROOF_LINE_MAX:
            pairs = '; '.join(f'found {f[:60]} -> set {v[:60]}' for f, v, *_ in r.get('installs') or [])
            return (f' (the line at {s} was {r["longest"]} characters: the found / set pair was read as '
                    + (pairs or 'none - no `found ... -> set ...` in it') + ')')
    return ''


def parsed(cur: dict, old: dict) -> dict:
    """F13: what the scanner read of a probe's proof lines, per site (at most 8): its new installation (found, set),
    whether its `active` line came, a `deferred` line's reason, a line longer than a display line."""
    new = {}
    for s, f, v, _ in new_installs(cur, old):
        new.setdefault(s, (f, v))
    out = {}
    for s, r in list(cur.items())[:8]:
        f, v = new.get(s, (None, None))
        out[s] = {'found': None if f is None else _show(f), 'set': None if v is None else _show(v),
                  'active': r.get('active_at') is not None,
                  **({'deferred': r['deferred'][0]} if r.get('deferred') else {}),
                  **({'line_chars': r['longest']} if r.get('longest') else {})}
    return out


# ---------------------------------------------------------------- the verdict
def proof_reason(row: dict, base_events: dict, notes: list, claim: str | None, h: dict | None = None,
                 feedback: list | None = None, *, whole: bool = True, deferred: bool = True
                 ) -> tuple[str | None, str | None]:
    """F10: (why the patch did not prove at its probe that it took effect, the note it targets) - `row` its probe's
    detailed trial row, `base_events` its base's run's events, `notes` the job-log notes it may target, `claim` its
    patch_claim, `h` its hypothesis. The note it named (`evidence_note`) reported after it installed: that hook is not
    the one the job calls. Else its proof (installed found != set, `active` within the probe - unless its first run is
    declared beyond it). A note its claim shares only a code name with, or one without the order of the lines, is no
    veto: it goes to `feedback`. `whole` (F13): the lines read whole; `deferred` (F21): a `deferred` line, and a note
    still reported names both of its explanations."""
    cur, old = proof_of(row, whole), proof_of(base_events, whole)
    named = claimed_notes((h or {}).get('evidence_note'), notes)
    incidental = [n for n in targeted_notes(claim or '', notes) if n not in named]
    now = {_note_key(n) for n in row.get('perf_notes') or []}
    last = {_note_key(k): v for k, v in (row.get('perf_note_last') or {}).items() if isinstance(v, (int, float))}
    at = [x[3] for x in new_installs(cur, old) if isinstance(x[3], (int, float))]
    first = min(at) if at else None

    def after(n):                                       # True, False, or None when the order is not known
        k = _note_key(n)
        if k not in now:
            return False
        return None if first is None or k not in last else last[k] > first
    for n in named:
        if after(n):
            return note_reason(cur, old, n, deferred=deferred), n
    for n in named + incidental:
        if after(n) is not False and feedback is not None:
            feedback.append({'note': n[:200], 'named': n in named, 'after_install': after(n)})
    a = supported_active(h)                             # (supported values only: no exemption)
    steps = row.get('steps')
    unseen = bool(a) and isinstance(steps, int) and a > steps   # (a probe shorter than the declared first run)
    why = proof_verdict(cur, old, unseen_active=unseen, deferred=deferred)
    if why and whole and why.startswith('it printed no'):
        why += overlong(cur)                            # F13: a line too long to display, as it was read
    return why, (named or incidental or [None])[0]


def check(engine, base: str, row: dict, patch: dict | None, h: dict | None = None, *, whole: bool = True,
          deferred: bool = True, feedback: list | None = None) -> dict | None:
    """For a strategy's Proposal gate after a patch's probe (`row`: the engine's detailed trial row): None when the patch
    proved it took effect, else {'reason', 'note', 'lines', 'parsed'} - its gate returns (reason, 'patch_inert'), and the
    record is what the proposer is told of the inert patch (its lines, what the scanner read)."""
    bev = events_of(engine, base)
    reason, note = proof_reason(row, bev, notes_of(engine, base), patch_claim(files_of(engine, base), patch, h), h,
                                feedback, whole=whole, deferred=deferred)
    if not reason:
        return None
    return {'reason': reason, 'note': note, 'lines': list(row.get('patch_lines') or [])[:6],
            **({'parsed': parsed(proof_of(row, True), proof_of(bev, True))} if whole else {})}
