#!/usr/bin/env python3
"""Build the Claude Code / Codex skill `infrarsi-optimize`.

    python agent_skill/build_skill.py [--out DIR]        ->  DIR/infrarsi-optimize/   (default DIR: agent_skill/dist)

The built skill is the skill's own files (agent_skill/infrarsi-optimize: SKILL.md, references/, scripts/infrarsi.py,
agents/) plus a copy of the engine, the `infrarsi` package of this tree, under scripts/engine/. The engine is copied
whole and unchanged, so the skill runs the code `infrarsi` of this tree runs, on a machine without network access and
with nothing to install but PyYAML. scripts/engine/VERSION.json records what was copied (version, commit, a hash of the
package: the one a search checkpoint carries).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
NAME = 'infrarsi-optimize'
SKIP = shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache')


def package_hash(base: Path) -> str:
    """As infrarsi.agent.state.package_hash: sha256 over the package's files (relative path and bytes)."""
    h = hashlib.sha256()
    for p in sorted(base.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
            h.update(p.relative_to(base).as_posix().encode('utf-8') + b'\0')
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def _commit(repo: Path) -> str | None:
    """The commit of the tree the engine was copied from, `+changes` when the package differs from it; None outside git."""
    try:
        head = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=30)
        if head.returncode != 0:
            return None
        dirty = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain', '--', 'infrarsi'], capture_output=True,
                               text=True, timeout=30).stdout.strip()
        return head.stdout.strip() + ('+changes' if dirty else '')
    except (OSError, subprocess.SubprocessError):
        return None


def build(out: Path, repo: Path = REPO) -> Path:
    """The skill built under `out`; the directory it is in."""
    src, engine = repo / 'agent_skill' / NAME, repo / 'infrarsi'
    if not (src / 'SKILL.md').is_file() or not (engine / '__init__.py').is_file():
        raise SystemExit(f'{repo} is not the InfraRSI tree (agent_skill/{NAME}/SKILL.md and infrarsi/ are expected in it)')
    dst = Path(out).resolve() / NAME
    for part in (src.resolve(), engine.resolve()):       # (--out agent_skill would delete the skill's own source)
        if dst == part or part in dst.parents or dst in part.parents:
            raise SystemExit(f'--out {out}: the build would go to {dst}, which is inside, or holds, {part}; choose another '
                             'directory (the default is agent_skill/dist)')
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=SKIP)
    shutil.copytree(engine, dst / 'scripts' / 'engine' / 'infrarsi', ignore=SKIP)
    version = re.search(r'^version\s*=\s*"([^"]+)"', (repo / 'pyproject.toml').read_text(encoding='utf-8'), re.M)
    files = [p for p in (dst / 'scripts' / 'engine' / 'infrarsi').rglob('*') if p.is_file()]
    (dst / 'scripts' / 'engine' / 'VERSION.json').write_text(json.dumps({
        'package': 'infrarsi', 'version': version.group(1) if version else None, 'commit': _commit(repo),
        'package_hash': package_hash(dst / 'scripts' / 'engine' / 'infrarsi'), 'files': len(files),
        'bytes': sum(p.stat().st_size for p in files),
        'built_utc': _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')}, indent=1) + '\n', encoding='utf-8')
    return dst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(HERE / 'dist'), help='where to build (default: agent_skill/dist)')
    a = ap.parse_args(argv)
    dst = build(Path(a.out))
    info = json.loads((dst / 'scripts' / 'engine' / 'VERSION.json').read_text(encoding='utf-8'))
    total = sum(p.stat().st_size for p in dst.rglob('*') if p.is_file())
    print(f'built {dst}: {total / 1e6:.1f} MB, engine infrarsi {info["version"]} ({info["files"]} files, commit '
          f'{str(info["commit"])[:12]})')
    print(f'install: python {Path(__file__).with_name("install_skill.py")} --host both')
    return 0


if __name__ == '__main__':
    sys.exit(main())
