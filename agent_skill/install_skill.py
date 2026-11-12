#!/usr/bin/env python3
"""Install the skill `infrarsi-optimize` for Claude Code and / or Codex.

    python agent_skill/install_skill.py [--host claude|codex|both] [--project DIR] [--from DIR]
    python agent_skill/install_skill.py --check        what is installed where
    python agent_skill/install_skill.py --uninstall    remove it

For each host it puts in place the skill (built from this tree first, unless --from names a built one) and the
definition of the sub-agent that answers the engine's requests:

    Claude Code   ~/.claude/skills/infrarsi-optimize/      ~/.claude/agents/infrarsi-answerer.md
    Codex         ~/.codex/skills/infrarsi-optimize/       ~/.codex/agents/infrarsi-answerer.toml

With --project DIR it goes under DIR for that project only: .claude/skills and .claude/agents for Claude Code,
.agents/skills and .codex/agents for Codex. Prefer the home directory: a skill inside the training project is copied
into every candidate tree the search makes, unless `.claude`, `.agents` and `.codex` are in the manifest's `code.exclude`.
Nothing else is written; an installed copy is replaced as a whole. In a session, the skill is then `/infrarsi-optimize`
(Claude Code) or `$infrarsi-optimize` (Codex), or it is picked up when the user asks to make a training job faster.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME, AGENT = 'infrarsi-optimize', 'infrarsi-answerer'


def places(host: str, base: Path, project: bool = False) -> tuple[Path, Path]:
    """(the skill's directory, the sub-agent's file) of a host under `base` (the home directory, or a project). Codex
    reads a repository's skills from .agents/skills and a user's from ~/.codex/skills; its agents from .codex/agents."""
    if host == 'claude':
        return base / '.claude' / 'skills' / NAME, base / '.claude' / 'agents' / f'{AGENT}.md'
    skills = base / '.agents' / 'skills' if project else base / '.codex' / 'skills'
    return skills / NAME, base / '.codex' / 'agents' / f'{AGENT}.toml'


def _remove(path: Path) -> bool:
    """Remove what is at `path` - a directory, a file, or a link to either (the link, not what it points to)."""
    if path.is_symlink() or path.is_file():
        path.unlink()
        return True
    if path.exists():
        shutil.rmtree(path)
        return True
    return False


def install(host: str, built: Path, base: Path, project: bool = False) -> list[Path]:
    skill, agent = places(host, base, project)
    _remove(skill)
    skill.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(built, skill)
    agent.parent.mkdir(parents=True, exist_ok=True)
    _remove(agent)
    shutil.copyfile(built / 'agents' / host / agent.name, agent)
    return [skill, agent]


def uninstall(host: str, base: Path, project: bool = False) -> list[Path]:
    return [p for p in places(host, base, project) if _remove(p)]


def check(host: str, base: Path, project: bool = False) -> dict:
    skill, agent = places(host, base, project)
    info = None
    try:
        info = json.loads((skill / 'scripts' / 'engine' / 'VERSION.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        pass
    return {'host': host, 'skill': str(skill) if (skill / 'SKILL.md').is_file() else None,
            'agent': str(agent) if agent.is_file() else None,
            'engine': {k: info.get(k) for k in ('version', 'commit', 'built_utc')} if info else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host', choices=['claude', 'codex', 'both'], default='both')
    ap.add_argument('--project', help='install into this project directory instead of the home directory')
    ap.add_argument('--from', dest='built', help='a built skill directory (default: build from this tree)')
    ap.add_argument('--home', help=argparse.SUPPRESS)       # (tests: another home directory)
    ap.add_argument('--check', action='store_true', help='show what is installed')
    ap.add_argument('--uninstall', action='store_true', help='remove the skill and the sub-agent definition')
    a = ap.parse_args(argv)
    hosts = ['claude', 'codex'] if a.host == 'both' else [a.host]
    project = bool(a.project)
    base = Path(a.project).resolve() if a.project else Path(a.home).resolve() if a.home else Path.home()
    if a.check:
        for h in hosts:
            print(json.dumps(check(h, base, project)))
        return 0
    if a.uninstall:
        for h in hosts:
            for p in uninstall(h, base, project):
                print(f'removed {p}')
        return 0
    with tempfile.TemporaryDirectory(prefix='infrarsi_skill_') as tmp:
        if a.built:
            given = Path(a.built).resolve()
            if not (given / 'SKILL.md').is_file() or not (given / 'scripts' / 'engine' / 'infrarsi').is_dir():
                raise SystemExit(f'{given} is not a built skill (SKILL.md and scripts/engine/infrarsi are expected in it): '
                                 'build one with build_skill.py')
            built = Path(tmp) / NAME                     # a copy first: --from may name the installed skill itself,
            shutil.copytree(given, built)                # which the install is about to replace
        else:
            sys.path.insert(0, str(HERE))
            from build_skill import build
            built = build(Path(tmp))
        for h in hosts:
            for p in install(h, built, base, project):
                print(f'installed {p}')
    print('Start a new session (or reload skills) to use it: /infrarsi-optimize in Claude Code, $infrarsi-optimize in Codex.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
