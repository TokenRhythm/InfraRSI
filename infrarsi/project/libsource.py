"""Read-only excerpts of the library source a job imports (D7, the user's decision 2026-10-31; for any strategy whose
proposer opts in: search/library.py): the regions of the library files that the job's job-log notes or a patch's site
name, read where the job runs - with the job's own python, over the executor (`probe`: host 0, in the job's
environment) - and only from files the job imported (its reference run's module list, written by the run hook: runner
MODULES) that resolve - links followed - inside one of that interpreter's own site-packages / dist-packages roots
(Codex HLS5 r1: not a path that merely names one). Static environment text, not a measurement of the job.

    excerpts(executor, python, modules_dir, names) -> [{'name', 'module', 'qualname', 'file', 'lines', 'text'}, ...]

A dotted name whose prefix is an imported module is looked up there (`pkg.mod.Class.method`); another name is searched
as a definition (def, class, assignment) in the imported library files, a `Class.member` also as a member of a class
whose name ends with it. At most MAX_LINES lines in all, PER_NAME per region (its head, the rest counted)."""
from __future__ import annotations

import json
import re

MAX_LINES, PER_NAME = 400, 120
NOTE = ('read-only environment text: regions of the library source this job imports, at the names its job-log notes and '
        'patch sites give - what the code you patch calls or replaces; not this job\'s files, not patchable')
NAME = re.compile(r'`([A-Za-z_][\w.]{2,80})`|\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+|[A-Za-z0-9]*_[A-Za-z0-9_]+)\b')

# run by the job's python where the job runs; standard library only; argv: modules dir, names (JSON), budget, per name
SCRIPT = r'''
import ast, glob, json, os, re, sys
mods = {}
for p in glob.glob(os.path.join(sys.argv[1], '*.json')):
    try:
        with open(p, encoding='utf-8') as f:
            mods.update(json.load(f))
    except Exception:
        pass
names, budget, per = json.loads(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
import site, sysconfig
cands = []
for get in (lambda: list(site.getsitepackages()), lambda: [site.getusersitepackages()],
            lambda: [sysconfig.get_paths()[k] for k in ('purelib', 'platlib')]):
    try:
        cands += get()
    except Exception:
        pass
nc = os.path.normcase
roots = sorted({nc(os.path.realpath(r)) for r in cands if isinstance(r, str) and os.path.isdir(r)
                and os.path.basename(os.path.normpath(r)) in ('site-packages', 'dist-packages')})

def inside(f):
    try:
        rf = os.path.realpath(f)
        return rf if any(os.path.commonpath([nc(rf), r]) == r for r in roots) else None
    except (ValueError, OSError):
        return None

lib = {}
for m, f in mods.items():
    rf = inside(f) if isinstance(f, str) else None
    if rf and os.path.isfile(rf):
        lib[m] = rf
trees = {}

def tree(f):
    if f not in trees:
        try:
            with open(f, encoding='utf-8', errors='replace') as h:
                text = h.read()
            trees[f] = (text.splitlines(), ast.parse(text))
        except Exception:
            trees[f] = None
    return trees[f]

def member(node, part):
    for n in getattr(node, 'body', []):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == part:
            return n
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            ts = n.targets if isinstance(n, ast.Assign) else [n.target]
            if any(isinstance(t, ast.Name) and t.id == part for t in ts):
                return n
    return None

def walk(root, qual):
    cur = root
    for part in qual.split('.'):
        cur = member(cur, part)
        if cur is None:
            return None
    return cur

def find(name):
    parts = name.split('.')
    for i in range(len(parts), 0, -1):
        m = '.'.join(parts[:i])
        if m in lib and i < len(parts):
            t = tree(lib[m])
            node = walk(t[1], '.'.join(parts[i:])) if t else None
            return [(m, lib[m], '.'.join(parts[i:]), node)] if node is not None else []
    last, owner = parts[-1], (parts[-2] if len(parts) > 1 else None)
    pat = re.compile(r'^\s*(?:async\s+def|def|class)\s+%s\b|^%s\s*[:=]' % (re.escape(last), re.escape(last)), re.M)
    hits = []
    for m, f in sorted(lib.items()):
        try:
            with open(f, encoding='utf-8', errors='replace') as h:
                text = h.read()
        except OSError:
            continue
        if last not in text or not pat.search(text):
            continue
        t = tree(f)
        if not t:
            continue
        if owner:
            for c in [n for n in t[1].body if isinstance(n, ast.ClassDef) and n.name.endswith(owner)]:
                node = member(c, last)
                if node is not None:
                    hits.append((m, f, c.name + '.' + last, node))
        node = member(t[1], last)
        if node is not None:
            hits.append((m, f, last, node))
        if len(hits) >= 3:
            break
    return hits[:3]

out, used, seen = [], 0, set()
for name in names:
    for m, f, qual, node in find(name):
        if (f, qual) in seen or used >= budget:
            continue
        seen.add((f, qual))
        lines = trees[f][0]
        a = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
        b = getattr(node, 'end_lineno', a) or a
        n = min(b - a + 1, per, budget - used)
        text = '\n'.join(lines[a - 1:a - 1 + n])
        if b - a + 1 > n:
            text += '\n# ... %d more lines' % (b - a + 1 - n)
        used += n
        out.append({'name': name, 'module': m, 'qualname': qual, 'file': f, 'lines': '%d-%d' % (a, b), 'text': text})
print('__libsource__' + json.dumps(out))
'''


def names_in(texts) -> list[str]:
    """The code names texts give, in their case and order (backticked words, dotted names, names with an underscore)."""
    out = []
    for t in texts or []:
        for m in NAME.finditer(str(t or '')):
            w = (m.group(1) or m.group(2) or '').strip('.')
            if len(w) >= 6 and ('_' in w or '.' in w or m.group(1)) and w not in out:
                out.append(w)
    return out


def excerpts(executor, python: str, modules_dir: str, names: list[str], *, cwd: str | None = None,
             max_lines: int = MAX_LINES, per_name: int = PER_NAME) -> list[dict] | None:
    """The excerpts (SCRIPT, run with the job's python over the executor); None when it could not run."""
    q = executor.quote
    cmd = f'{python} -c {q(SCRIPT)} {q(modules_dir)} {q(json.dumps(list(names)))} {int(max_lines)} {int(per_name)}'
    try:
        text = executor.probe(cmd, cwd=cwd or modules_dir, env={}, timeout_s=120)
    except Exception:                                   # noqa: BLE001 - read-only context: never the search's failure
        return None
    for line in reversed((text or '').splitlines()):
        if line.startswith('__libsource__'):
            try:
                got = json.loads(line[len('__libsource__'):])
            except ValueError:
                return None
            return got if isinstance(got, list) else None
    return None
