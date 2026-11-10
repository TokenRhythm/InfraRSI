"""What a search arm learned of its starts that a later arm can start from (F9b; the user's decision D4, 2026-10-31) - for
any strategy: the sets of changes that failed to start together (with what each did alone), the pairs two or more failed
starts held, the values that failed to start alone. A strategy puts them in its describe() as the `learned` section
(section(...)); runs.write_back writes them into the run's experience pack as `constraint` lines (source "learned",
symptom "crash", scoped to the framework and the software versions) and a later arm's store reads them back:

    write   runs.write_back adds constraints(learned, ...) - `learned`: the strategy's describe()['learned'] (or a part's)
    read    store.plan's `start_failures`: the learned constraints in scope, in this project's knob values

In a later arm the transfer judgment reads them with every other constraint in scope (store.payload), the plan's check
warns (a matching proposal is probed first), and a strategy may start from the plan's `start_failures` (hls3: their pairs
suspect, F5a, and their sets composites that died at their start, F16-A). Knob values travel as the store's mechanism
states (store.states_of: a knob without a mechanism is its own x- id), so another project with the same knobs reads them
back; a value with no state is not written. Random and the other strategies read the store as before (a constraint warns
them as any constraint does).

    python -m infrarsi.experience.learned <arm dir> <out .jsonl>     # an hls3 arm written before F9b: its result.json
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

from .pack import SCHEMA, validate, write_lines
from .store import UNKNOWN, labels, payload_constraints, states_of, x_id

SOURCE = 'learned'
KINDS = (('failed_together', 'failed to start together'), ('pairs', 'failed to start together, more than once'),
         ('alone', 'failed to start alone'))


def section(failed_together=(), pairs=(), alone=()) -> dict:
    """The `learned` section of a strategy's describe() (runs.write_back reads it), each item {'members': [{knob: value},
    ...] (a set's members; one member for `alone`), 'kind': the failure class, 'run': the candidate, 'separately': {member
    words: what it did alone} (sets), 'strikes': the failed starts that held it (pairs)}; empty kinds left out."""
    return {k: [dict(x) for x in v] for k, v in (('failed_together', failed_together), ('pairs', pairs), ('alone', alone))
            if v}


def scope_of(fp: dict) -> dict:
    """Where such a start failure holds: the framework and the software versions of the run (its fingerprint)."""
    return {'framework': fp.get('framework') or 'other',
            **{f'versions.{k}': f'=={v}' for k, v in sorted((fp.get('versions') or {}).items()) if v}}


def constraints(learned: dict | None, fp: dict, labs: dict, run_id: str) -> list[dict]:
    """The `constraint` lines of what a run learned (a `learned` section): one per failed set, recurrent pair and value
    that failed alone, `when` the members' mechanism states (a member with no state: the item is not written)."""
    out = []
    for key, words in KINDS:
        for it in (learned or {}).get(key) or []:
            cfg = {}
            for m in it.get('members') or []:
                cfg.update(m)
            own = {x_id(n) if labs.get(n) is None else labs[n].mechanism for n in cfg}
            when = {k: v for k, v in states_of({}, labs, cfg).items() if k in own and v != UNKNOWN}
            if not cfg or len(when) < len(own):
                continue                                    # (a value the registry gives no state)
            alone = it.get('separately') or {}
            msg = (f'{", ".join(f"{k}={v}" for k, v in cfg.items())}: {words} in run {run_id}'
                   + (f' ({it.get("kind")})' if it.get('kind') else '')
                   + (f'; alone: {", ".join(f"{k} {v}" for k, v in alone.items())}' if alone else '')
                   + (f'; {it["strikes"]} failed starts held it' if it.get('strikes') else ''))
            out.append({'kind': 'constraint', 'id': f'learned-{len(out) + 1}', 'when': when, 'symptom': 'crash',
                        'message': msg[:300], 'scope': scope_of(fp), 'source': SOURCE, 'learned': key,
                        **({'evidence': [f'run-{run_id}/{it["run"]}']} if it.get('run') else {})})
    return out


def _choices(knob: dict) -> list:
    return list(knob.get('choices') or ([False, True] if knob.get('type') == 'bool' else []))


def start_failures(store, fp: dict, knobs: dict, labs: dict) -> list[dict]:
    """The learned constraints in scope, in this project's knob values: [{'members': [{knob: value}, ...], 'id',
    'learned', 'message'}]; one whose states no knob of this project takes is left out."""
    out = []
    for c in payload_constraints(store, fp):
        if c.get('source') != SOURCE:
            continue
        members = []
        for m, s in (c.get('when') or {}).items():
            hit = None
            for n, k in (knobs or {}).items():
                lab = labs.get(n)
                if (x_id(n) if lab is None else lab.mechanism) != m:
                    continue
                for v in _choices(k if isinstance(k, dict) else {}):
                    st = (str(v).lower() if isinstance(v, bool) else str(v)) if lab is None else lab.state(v)
                    if st is not None and str(st) == str(s):
                        hit = {n: v}
                        break
                if hit:
                    break
            if hit is None:
                members = None
                break
            members.append(hit)
        if members:
            out.append({'members': members, 'id': f'{c.get("pack")}/{c.get("id")}', 'learned': c.get('learned'),
                        'message': c.get('message')})
    return out


def _value(text: str):
    """A knob value as hls3's report words print it (`k=v`: Python's repr of a JSON value)."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def learned_from_report(report: dict) -> dict:
    """An hls3 arm written before F9b: what its report (result.json search.hls3) shows it learned, as a `learned` section
    - its failed-together sets (members `knob=value`)."""
    sets = []
    for t in report.get('failed_together') or []:
        members = []
        for w in t.get('members') or []:
            parts = [p.split('=', 1) for p in str(w).split(', ')]
            if any(len(p) != 2 or p[0].startswith('patch:') for p in parts):
                members = None
                break
            members.append({k: _value(v) for k, v in parts})
        if members and len(members) >= 2:
            sets.append({'members': members, 'kind': t.get('kind'), 'run': t.get('candidate'),
                         'separately': dict(t.get('separately') or {})})
    return section(failed_together=sets)


def backfill(arm: Path, out: Path) -> Path | None:
    """A pack of an earlier arm's learned start failures (its result.json, its manifest's knobs and framework)."""
    import yaml
    res = json.loads(next(arm.glob('*__result.json')).read_text(encoding='utf-8'))
    man = yaml.safe_load((arm / 'repo' / 'infrarsi.yaml').read_text(encoding='utf-8'))
    learned = learned_from_report(((res.get('search') or {}).get('hls3')) or {})
    fp = {'framework': man.get('framework'), 'versions': {}}
    run_id = res.get('run_id') or arm.name
    lines = [{'kind': 'pack', 'id': 'pack', 'schema': SCHEMA, 'pack_id': f'learned-{run_id}',
              'title': f'InfraRSI run {run_id}: what the search learned of its starts', 'contributors': ['infrarsi'],
              'license': 'unspecified', 'created': str(res.get('utc_end') or '')[:10] or 'unknown', 'tier': 'verified',
              'summary': ''}]
    lines += constraints(learned, fp, labels(man.get('knobs') or {}, fp['framework']), run_id)
    lines[0]['summary'] = f'{len(lines) - 1} start failures learned by the search (from its report)'
    errors, _ = validate(lines)
    if errors or len(lines) < 2:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    write_lines(out, lines)
    return out


if __name__ == '__main__':
    p = backfill(Path(sys.argv[1]), Path(sys.argv[2]))
    print('wrote', p) if p else print('nothing to write')
