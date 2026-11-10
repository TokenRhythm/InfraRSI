"""Read-only library source next to the code (D7, the user's decision 2026-10-31) - for any strategy whose proposer opts in
(its `library_source` feature: search/proposers.FEATURES).

The code prompts may show excerpts of the library source the job imports (its reference run's module list, the runner's
`modules_dir`), at the code names its job-log notes and the patch sites give, read where the job runs with its own python
(the executor's probe; project/libsource: at most 400 lines, the named functions and regions, never whole files). It is
static environment text, not a measurement of the job: what the code a patch replaces calls, its fallbacks, the arguments
its callers pass.

    names(engine, base, sites)           the code names to read: the notes' (the base's run, the reference's), the sites'
    source(engine, base, sites, cache)   the proposer's `library_source` field, or None
"""
from __future__ import annotations

from .proof import notes_of


def names(engine, base: str, sites=()) -> list[str]:
    """The code names the job-log notes (the base's run, the reference's) and the patch sites (`sites`: the proposer's
    patch entries, the sites inert patches printed - the last 12) give - notes first, at most eight."""
    from ..project.libsource import names_in
    texts = notes_of(engine, base) + (notes_of(engine, 'c0') if base != 'c0' else [])
    return names_in(texts + [str(s) for s in sites][-12:])[:8]


def _evidence(c) -> dict:
    rec = getattr(c, 'receipt', None) or {}
    return rec.get('evidence') if isinstance(rec.get('evidence'), dict) else {}


def source(engine, base: str, sites=(), cache: dict | None = None) -> dict | None:
    """Read-only excerpts of the library source the job imported (its reference run's `modules_dir`) at `names`, read
    where the job runs with its own python (the executor's probe); once per set of names (`cache`). None without an
    executor that probes, the module list, or a name."""
    runner = getattr(engine.s, 'runner', None)
    exe, proj = getattr(runner, 'executor', None), getattr(runner, 'project', None)
    mdir = _evidence(engine.s.candidates.get('c0')).get('modules_dir')
    if exe is None or proj is None or not mdir or not hasattr(exe, 'probe'):
        return None
    got_names = names(engine, base, sites)
    if not got_names:
        return None
    cache = {} if cache is None else cache
    key = tuple(got_names)
    if key not in cache:
        from ..project.libsource import excerpts
        py = exe.python(((getattr(proj, 'raw', None) or {}).get('execution') or {}).get('python'))
        cache[key] = excerpts(exe, py, mdir, got_names)
    got = cache[key]
    if not got:
        return None
    from ..project.libsource import NOTE
    return {'note': NOTE, 'names': got_names, 'excerpts': got}
