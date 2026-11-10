"""`infrarsi apply <run dir>`: show (default) or write the recommended change into the project.

Config-file knobs go to a new file next to the user's config (`<name>.infrarsi<ext>`, the original stays untouched);
patched mutable files and script constants (const knobs) are written in place - a file edited since the search gets
the change merged onto the edits, and when the two do not merge nothing is written; flag, override and environment
knobs are printed as instructions, because the launch command belongs to the user. With --branch in a git repository
the files are written on a new branch.
"""
from __future__ import annotations

import difflib
import json
import subprocess
from pathlib import Path

import yaml

from ..search.strategies import merge3
from .manifest import Project, env_for, set_path, write_const


def plan(run_dir: str | Path) -> dict:
    r = json.loads((Path(run_dir) / 'result.json').read_text(encoding='utf-8'))
    promo = r.get('promotion') or {}
    if not promo.get('promoted'):
        return {'run_id': r.get('run_id'), 'files': [], 'manual': [], 'note': 'nothing to apply: ' + str(promo.get('why') or r.get('outcome'))}
    project = Project.load(r['manifest'])
    ref, cfg = r['reference']['config'], promo.get('config') or {}
    changed = {k: v for k, v in cfg.items() if ref.get(k) != v}
    files, manual, config_copy = [], [], None
    path_knobs = {k for k in changed if 'path' in project.knobs.get(k, {})}
    cf = project.config_file
    if path_knobs and cf is not None:
        doc = project.base_config() or {}
        for k in sorted(path_knobs):
            set_path(doc, project.knobs[k]['path'], changed[k])
        text = json.dumps(doc, indent=1) if cf.suffix.lower() == '.json' else yaml.safe_dump(doc, sort_keys=False)
        new = cf.with_name(f'{cf.stem}.infrarsi{cf.suffix}')
        files.append({'path': str(new), 'old': new.read_text(encoding='utf-8') if new.exists() else '', 'new': text})
        manual.append(f'point your launch command at {new.name} instead of {cf.name}')
        config_copy = Path(project.raw['config']['file']).with_name(new.name).as_posix()   # as config.file must name it
    for k in sorted(changed):
        spec = project.knobs.get(k, {})
        if 'flag' in spec:
            manual.append(flag_step(spec, ref.get(k), changed[k]))
        elif 'override' in spec:
            v = changed[k]
            manual.append(f'override {spec["override"]}={str(v).lower() if isinstance(v, bool) else v} (was {ref.get(k)!r})')
        elif 'env' in spec:
            env = env_for(project, cfg)                 # as the runs set it (a bool as 1 / 0)
            manual.append(f'environment {spec["env"]}={env.get(str(spec["env"]), "(unset)")} (was {ref.get(k)!r}): set it '
                          'where the job is launched')
    texts = dict(promo.get('overlay') or {})          # patched files, then constants rewritten on top (as the runner:
    for k, spec in sorted(project.knobs.items()):    # every constant of a patched file, the changed ones elsewhere)
        if 'const' in spec and cfg.get(k) is not None and (spec['file'] in texts or k in changed):
            rel = spec['file']
            text = texts[rel] if rel in texts else (project.root / rel).read_text(encoding='utf-8')
            texts[rel] = write_const(text, spec['const'], cfg[k])
    base, conflicts = promo.get('base') or {}, []
    for rel, text in sorted(texts.items()):
        f = project.root / rel
        old = f.read_text(encoding='utf-8') if f.exists() else ''
        if base.get(rel) is not None and old != base[rel]:   # edited since the search: the change goes on top of it
            merged = merge3(base[rel], text, old)
            if merged is None:
                conflicts.append(rel)
            text = text if merged is None else merged
        files.append({'path': str(f), 'old': old, 'new': text})
    return {'run_id': r['run_id'], 'root': str(project.root), 'files': files, 'manual': manual, 'candidate': promo['promoted'],
            'conflicts': conflicts, 'changed': {k: [ref.get(k), v] for k, v in changed.items()},   # (U-1-23: printed first)
            'config_copy': config_copy}                   # the config's copy as config.file must name it (U-1-24)


def flag_step(spec: dict, old, new) -> str:
    """A flag knob's change as the user makes it on their command line (U-2-8): a switch is added or removed, not
    'False -> True'."""
    if isinstance(new, bool) and spec.get('style', 'switch') == 'switch':
        off = spec.get('off_flag')                      # a pair of switches: the one in the command gives way to the other
        if new:
            return f'replace {off} with {spec["flag"]} in your launch command' if off else f'add {spec["flag"]} to your launch command'
        return f'replace {spec["flag"]} with {off} in your launch command' if off else f'remove {spec["flag"]} from your launch command'
    return f'flag {spec["flag"]}: {old!r} -> {new!r}'


def diff_text(p: dict) -> str:
    out = []
    for f in p['files']:
        out += difflib.unified_diff(f['old'].splitlines(), f['new'].splitlines(), f'a/{Path(f["path"]).name}',
                                    f'b/{Path(f["path"]).name}', lineterm='')
    return '\n'.join(out)


def apply(run_dir: str | Path, *, write: bool = False, branch: str | None = None) -> dict:
    p = plan(run_dir)
    p['diff'] = diff_text(p)
    if write and p['files']:
        if p['conflicts']:
            raise RuntimeError(f'{", ".join(p["conflicts"])} changed since the search and the recommended change does not '
                               'merge with the edits: nothing is written (merge the diff by hand or run the search again)')
        if branch:
            r = subprocess.run(['git', '-C', p['root'], 'checkout', '-b', branch], capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f'git checkout -b {branch} failed: {(r.stdout + r.stderr).strip()[:300]}')
        for f in p['files']:
            Path(f['path']).parent.mkdir(parents=True, exist_ok=True)
            Path(f['path']).write_text(f['new'], encoding='utf-8')
        p['written'] = [f['path'] for f in p['files']]
    return p
