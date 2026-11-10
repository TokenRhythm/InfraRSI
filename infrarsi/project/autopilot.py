"""One-command autopilot shell around the 5-stage InfraRSI macro pipeline (`init` -> `doctor` -> `run` -> `report` -> `apply`).

Drop a launch script (`run.sh`, `scripts/train.sh`), a Python training script (`train.py`), a `--command "..."` line,
or a codebase directory directly into `infrarsi <target>` (or `infrarsi optimize <target>`):

1. Stage 1 (`init`): Resolves the repository root and entry script, wraps shell scripts that do not forward `"$@"`,
   drafts `infrarsi.yaml`, and heals missing reference defaults so zero manual YAML editing is required.
2. Stage 2 (`doctor`): Runs one reference verification pass to check signals, checklist coverage, and calibrate the
   noise / repeat floor before searching.
3. Stage 3 (`run`): Executes the micro search algorithm (`bcomp` / B-Comp v4.1 by default when an LLM backend is
   configured or auto-detected in the environment; `tpe` / `random` when running without an LLM) and runs the
   promotion verifier.
4. Stage 4 (`report`): Emits the run report (`REPORT.md`).
5. Stage 5 (`apply`): Applies verified code patches and config changes (`--dry-run` previews without writing) and
   writes a ready-to-run optimized launch script when flag, environment, or config knobs changed.
"""
from __future__ import annotations

import os
import posixpath
import re
import shlex
import shutil
from pathlib import Path

import yaml

from . import detect
from .llm import override as llm_override, parse as parse_llm
from .manifest import ManifestError, Project, args_for, env_for, expr_names

LAUNCH_WRAPPER = 'infrarsi_launch.sh'
OPTIMIZED_SCRIPT = 'infrarsi_optimized.sh'
SCRIPT_DIRS = ('scripts', 'script', 'bin', 'tools', 'examples', 'slurm')


def is_script_path(path: Path) -> bool:
    """True when `path` is a shell launch script or Python entry script (rather than a manifest or suite YAML)."""
    if not path.is_file():
        return False
    if path.suffix.lower() in ('.sh', '.bash', '.py'):
        return True
    if path.suffix.lower() in ('.yaml', '.yml', '.json', '.toml', '.md', '.txt'):
        return False
    return detect.is_shell(path)


def resolve_script_target(script: Path) -> tuple[Path, str]:
    """Resolve `(project_root, relative_entry_posix)` from a launch script or Python training script path."""
    script = script.resolve()
    parent = script.parent
    if (parent / 'infrarsi.yaml').is_file():
        return parent, script.relative_to(parent).as_posix()
    if parent.name.lower() in SCRIPT_DIRS and parent.parent.is_dir():
        gp = parent.parent
        if (gp / 'infrarsi.yaml').is_file() or (gp / '.git').exists() or (gp / 'pyproject.toml').is_file() \
                or list(gp.glob('*.py')) or (gp / 'configs').is_dir() or (gp / 'src').is_dir():
            return gp, script.relative_to(gp).as_posix()
    return parent, script.relative_to(parent).as_posix()


def ensure_shell_forwards(root: Path, entry: str, *, force: bool = False) -> tuple[str, str | None]:
    """If `entry` is a shell script that does not forward `"$@"`, create `infrarsi_launch.sh` with `"$@"` appended
    to its training launch line so search flags and step overrides reach the training command automatically."""
    script = root / entry
    if not script.is_file() or not detect.is_shell(script):
        return entry, None
    info = detect.shell_launch(script, root)
    if info['forwards'] or info['positional']:
        return entry, None
    raw_lines = script.read_text(encoding='utf-8', errors='replace').splitlines()
    idx = None
    for i in range(len(raw_lines) - 1, -1, -1):
        stripped = re.sub(r'(^|\s)#.*$', '', raw_lines[i]).strip()
        if not stripped:
            continue
        if detect.LAUNCHER.search(stripped) or '.py' in stripped or stripped.startswith(('python', 'torchrun', 'deepspeed', 'accelerate')):
            idx = i
            break
    if idx is None:
        for i in range(len(raw_lines) - 1, -1, -1):
            stripped = re.sub(r'(^|\s)#.*$', '', raw_lines[i]).strip()
            if stripped and not stripped.startswith(('export ', 'set ', 'cd ', 'source ', '. ', 'echo ', 'mkdir ')):
                idx = i
                break
    if idx is None:
        return entry, None
    # If the matched line continues onto following lines via trailing backslash, advance to the end of the continuation
    while idx + 1 < len(raw_lines) and raw_lines[idx].rstrip().endswith('\\'):
        idx += 1
    line = raw_lines[idx]
    comment_match = re.search(r'(\s+#.*)$', line)
    if comment_match:
        body = line[:comment_match.start()].rstrip()
        tail = comment_match.group(1)
        raw_lines[idx] = f'{body} "$@"{tail}'
    else:
        raw_lines[idx] = f'{line.rstrip()} "$@"'
    wrapper = root / LAUNCH_WRAPPER
    if not wrapper.exists() or force or entry != LAUNCH_WRAPPER:
        text = '\n'.join(raw_lines) + '\n'
        wrapper.write_text(text, encoding='utf-8', newline='\n')
    note = f'Wrapped {entry} -> {LAUNCH_WRAPPER} with "$@" so search flags reach the training command automatically.'
    return LAUNCH_WRAPPER, note


def auto_heal_manifest(manifest: dict, root: Path) -> list[str]:
    """Fill missing reference defaults on knobs used in invariants/constraints and drop broken auto-drafted constraints
    so a drafted manifest loads and runs without manual YAML editing."""
    healed: list[str] = []
    knobs = manifest.get('knobs') or {}
    invariants = (manifest.get('workload') or {}).get('invariants') or {}
    constraints = list(manifest.get('constraints') or [])
    used = set().union(*(expr_names(e) for e in [*invariants.values(), *constraints])) if invariants or constraints else set()
    for name in sorted(used):
        k = knobs.get(name)
        if not isinstance(k, dict) or k.get('default') is not None:
            continue
        if 'path' in k or 'const' in k:
            continue
        choices = k.get('choices') or []
        kind = k.get('type', 'str')
        if kind == 'bool':
            k['default'] = False
        elif choices:
            k['default'] = 1 if 1 in choices else choices[0]
        elif kind == 'int':
            k['default'] = 1
        elif kind == 'float':
            k['default'] = 1.0
        else:
            k['default'] = ''
        healed.append(f'auto-filled default {name}={k["default"]!r} for invariant/constraint evaluation')
    # Ensure Megatron micro_batch_size has a sensible default if unset
    if manifest.get('framework') == 'megatron' and 'micro_batch_size' in knobs and knobs['micro_batch_size'].get('default') is None:
        choices = knobs['micro_batch_size'].get('choices') or [1]
        knobs['micro_batch_size']['default'] = 1 if 1 in choices else choices[0]
        healed.append(f'auto-filled default micro_batch_size={knobs["micro_batch_size"]["default"]!r}')
    # Verify the manifest loads and filter out any auto-drafted constraint that fails on the reference config
    try:
        proj = Project(manifest, root / 'infrarsi.yaml')
        ref = proj.reference_config()
        if constraints:
            valid_constraints = []
            for expr in constraints:
                errs = proj.constraint_errors(ref)
                if any(expr in e for e in errs):
                    healed.append(f'dropped constraint {expr!r} that failed on initial defaults')
                else:
                    valid_constraints.append(expr)
            manifest['constraints'] = valid_constraints
    except ManifestError:
        # If an invariant cannot be evaluated even after filling defaults, clear broken invariants so the project loads
        if invariants:
            manifest['workload']['invariants'] = {}
            healed.append('cleared unverifiable auto-drafted workload.invariants')
    return healed


def ensure_manifest(target: str | Path, *, entry: str | None = None, config: str | None = None,
                    command: str | None = None, force: bool = False) -> tuple[Project, bool, list[str]]:
    """Resolve `target` (a launch script, Python training script, directory, or manifest path), drafting and healing
    `infrarsi.yaml` when needed. Returns `(project, drafted_now, notes)`."""
    p = Path(target or '.')
    notes: list[str] = []
    if p.is_file() and p.name == 'infrarsi.yaml' and not force and not entry and not command and not config:
        return Project.load(p), False, notes
    if is_script_path(p):
        if entry or command:
            raise ValueError('give a script path or --entry / --command, not both')
        root, entry = resolve_script_target(p)
    elif p.is_file() and p.name == 'infrarsi.yaml':
        root = p.parent.resolve()
    else:
        root = p.resolve()
    manifest_path = root / 'infrarsi.yaml'
    if manifest_path.is_file() and not force and not entry and not command and not config:
        return Project.load(manifest_path), False, notes

    if command:
        if entry:
            raise ValueError('give --command or --entry, not both')
        try:
            words = shlex.split(command)
        except ValueError as exc:
            raise ValueError(f'--command cannot be read: {exc}') from None
        from ..cli.main import shell_ops
        ops = sorted(set(shell_ops(command)))
        prog = [w for w in words if not ('=' in w and w.split('=')[0].isidentifier())]
        if ops:
            raise ValueError(f'--command takes one command, not a pipeline or a list ({" ".join(ops)})')
        if prog and (posixpath.basename(prog[0]) in ('bash', 'sh') or prog[0].endswith('.sh')):
            raise ValueError('the command starts a shell script: pass the script directly or with --entry')
        root.mkdir(parents=True, exist_ok=True)
        launch = root / LAUNCH_WRAPPER
        launch.write_text(
            f'#!/bin/bash\n# drafted by infrarsi autopilot (--command)\n{command.strip()} "$@"\n',
            encoding='utf-8',
            newline='\n',
        )
        entry = LAUNCH_WRAPPER
        notes.append(f'{LAUNCH_WRAPPER} wraps your command and forwards search flags via "$@".')
    elif entry:
        entry, wrap_note = ensure_shell_forwards(root, entry, force=force)
        if wrap_note:
            notes.append(wrap_note)
    else:
        # Check if auto-discovered entry is a shell script needing "$@" forwarding
        py_cands = detect.entry_candidates(root)
        sh_cands = detect.shell_candidates(root)
        # When a repository has a top-level launch script (e.g. run.sh) and no entry was specified, check if py_cands
        # vs sh_cands exists
        chosen = (py_cands or sh_cands or [None])[0]
        if chosen is not None and detect.is_shell(chosen):
            rel_chosen = chosen.relative_to(root).as_posix()
            entry, wrap_note = ensure_shell_forwards(root, rel_chosen, force=force)
            if wrap_note:
                notes.append(wrap_note)

    manifest, draft_notes = detect.draft(root, config=config, entry=entry)
    healed = auto_heal_manifest(manifest, root)
    all_notes = notes + healed + draft_notes
    detect.write(root, manifest, all_notes, force=True)
    return Project.load(manifest_path), True, all_notes


def _logged_in_cli() -> dict | None:
    """Detect installed CLIs with local login/config files without making a model call."""
    home = Path.home()
    claude_home = Path(os.environ.get('CLAUDE_CONFIG_DIR') or home / '.claude')
    codex_home = Path(os.environ.get('CODEX_HOME') or home / '.codex')
    claude = (os.environ.get('INFRARSI_CLAUDE_EXE') or shutil.which('claude')
              or next((str(p) for p in (home / '.local/bin/claude.exe', home / '.local/bin/claude') if p.is_file()), None))
    if claude and any(p.is_file() for p in (claude_home / '.credentials.json', home / '.claude.json')):
        return {'backend': 'claude-cli'}
    codex = os.environ.get('INFRARSI_CODEX_EXE') or shutil.which('codex')
    if codex and any((codex_home / name).is_file() for name in ('auth.json', 'config.toml')):
        return {'backend': 'codex-cli'}
    return None


def detect_llm_spec(project: Project, cli_llm: str | None = None) -> dict:
    """Resolve the LLM spec from `--llm`, the manifest, environment variables (`INFRARSI_LLM`, `DEEPSEEK_API_KEY`,
    `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `VLLM_URL`), then installed Claude/Codex logins.
    Falls back to `{'backend': 'none'}`."""
    if cli_llm is not None:
        return llm_override(project.raw.get('llm'), cli_llm)
    manifest_spec = parse_llm(project.raw.get('llm'))
    if manifest_spec.get('backend') and manifest_spec.get('backend') != 'none':
        return manifest_spec
    env_llm = os.environ.get('INFRARSI_LLM')
    if env_llm:
        return llm_override(manifest_spec, env_llm)
    if os.environ.get('DEEPSEEK_API_KEY'):
        return {'backend': 'deepseek', 'model': os.environ.get('INFRARSI_DEEPSEEK_MODEL', 'deepseek-chat')}
    if os.environ.get('ANTHROPIC_API_KEY'):
        return {'backend': 'anthropic', 'model': os.environ.get('INFRARSI_ANTHROPIC_MODEL', 'claude-opus-5-5')}
    if os.environ.get('OPENAI_API_KEY'):
        return {'backend': 'openai', 'model': os.environ.get('INFRARSI_OPENAI_MODEL', 'gpt-4o')}
    if os.environ.get('OPENROUTER_API_KEY'):
        return {
            'backend': 'openai-compat',
            'base_url': os.environ.get('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
            'api_key_env': 'OPENROUTER_API_KEY',
            'model': os.environ.get('INFRARSI_OPENROUTER_MODEL', 'anthropic/claude-3.7-sonnet'),
        }
    if os.environ.get('VLLM_URL'):
        return {
            'backend': 'vllm',
            'base_url': os.environ['VLLM_URL'],
            'model': os.environ.get('VLLM_MODEL', 'default'),
        }
    return _logged_in_cli() or {'backend': 'none'}


def default_strategy(cli_strategy: str | None, llm_spec: dict | None, knobs: dict | None = None) -> str:
    """Resolve the search strategy for `optimize`: defaults to `'bcomp'` (B-Comp v4.1) when an LLM is active,
    or `'tpe'`/`'random'` via `search.resolve('auto', has_llm=False)` when no LLM is active."""
    from .. import search
    has_llm = bool(llm_spec and llm_spec.get('backend') and llm_spec.get('backend') != 'none')
    if cli_strategy in (None, 'auto'):
        req = 'bcomp' if has_llm else 'auto'
    else:
        req = cli_strategy
    return search.resolve(req, has_llm=has_llm, knobs=knobs)


def write_optimized_launch(project: Project, run_dir: Path, applied: dict) -> Path | None:
    """Write `<root>/infrarsi_optimized.sh` containing the promoted environment variables, config file, and CLI flags
    so the user can launch the optimized job with a single script."""
    res_file = Path(run_dir) / 'result.json'
    if not res_file.is_file():
        return None
    import json
    res = json.loads(res_file.read_text(encoding='utf-8'))
    promo = res.get('promotion') or {}
    if not promo.get('promoted'):
        return None
    cfg = promo.get('config') or {}
    env = env_for(project, cfg)
    extra_args = args_for(project, cfg)
    cfg_rel = applied.get('config_copy') or (project.raw.get('config') or {}).get('file') or ''
    cmd = str(project.raw['launch']['command'])
    cmd = cmd.replace('{python}', 'python')
    cmd = cmd.replace('{gpus}', str(project.gpus))
    cmd = cmd.replace('{nodes}', str(project.nodes))
    cmd = cmd.replace('{world}', str(project.world))
    if '{config}' in cmd and cfg_rel:
        cmd = cmd.replace('{config}', cfg_rel)
    # Remove step cap template (`--max-steps {steps}`, `--steps {steps}`, `max_steps={steps}`) for full production runs
    cmd = re.sub(r'\s+(?:--[\w.-]+\s+\{steps\}|[\w.]+\s*=\s*\{steps\}\w*)', '', cmd)
    arg_str = ' '.join(shlex.quote(str(x)) for x in extra_args)
    if '{args}' in cmd:
        cmd = cmd.replace('{args}', arg_str).strip()
    elif arg_str:
        cmd = f'{cmd.strip()} {arg_str}'
    lines = [
        '#!/bin/bash',
        f'# Auto-generated by infrarsi from verified run {res.get("run_id")} (speedup x{promo.get("speedup", 1.0):.3f})',
        'set -e',
    ]
    for k, v in sorted(env.items()):
        lines.append(f'export {k}={shlex.quote(str(v))}')
    lines.append(f'{cmd} "$@"')
    out_path = project.root / OPTIMIZED_SCRIPT
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    try:
        out_path.chmod(0o755)
    except OSError:
        pass
    return out_path
