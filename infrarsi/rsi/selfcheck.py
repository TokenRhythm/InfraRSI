"""Package self-check (FINAL_PLAN 3.6): schema and identity, skills parse with all required sections, prompts keep
their placeholders and fit the argv budget, policy within caps, tools run on historical receipts, memory schema,
answer-isolation audit. A reflected package that fails is not used (invalid_fallback).
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import yaml

from ..agent import policy as policy_mod
from ..agent.backends.claude_cli import MAX_SYSTEM_ARG_UTF16_UNITS, system_arg_units
from ..agent.context import check_skill, receipt_view
from ..agent.memory import MemoryDB
from ..agent.roles import DEFAULT_CHECKLIST, fill
from ..agent.tools import STRONG_MENU, WEAK_MENU
from . import pure_analysis
from .audit import audit_package
from .package import A0_SKILLS, MANIFEST_SCHEMA, REQUIRED_PROMPTS, compute_digest, content_manifest

LIMITS = {'total_bytes': 4 << 20, 'skill_bytes': 64 << 10, 'prompt_bytes': 32 << 10, 'tool_bytes': 16000, 'max_skills': 32,
          'max_tools': 32, 'catalogue_bytes': 128 << 10}
PROMPT_PLACEHOLDERS = {'proposer.md': ('{ACTION_MENU}',), 'weak_proposer.md': ('{ACTION_MENU}',), 'verifier.md': ('{CHECKLIST}',)}
TOP_LEVEL = {'skills', 'prompts', 'tools', 'policy.yaml', 'memory.db', 'manifest.json'}


@dataclasses.dataclass
class SelfcheckReport:
    ok: bool
    errors: list
    warnings: list
    checked: dict

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def minimal_view() -> dict:
    """Synthetic receipt view so tools are exercised even before any history exists."""
    return {'valid': True, 'task': {'gpus': 8, 'seq_len': 8192, 'packs_per_update': 32, 'tokens_per_update': 262144, 'hbm_cap_gib': 78.0},
            'step_time': {'mean_window_max_rank_s': 5.0, 'median_window_max_rank_s': 5.0, 'rsd_window': 0.01, 'inter_step_mean_s': 5.02},
            'window_steps': [5.0, 5.02, 4.98], 'slowest_ranks': [0, 3, 3], 'memory': {'max_peak_reserved_gib': 60.0, 'max_peak_alloc_gib': 55.0, 'reserved_per_rank': [60.0, 59.5]},
            'events': {'alloc_retries': 0, 'alloc_ooms': 0, 'nccl_errors': 0, 'nan_inf_steps': 0}, 'phases': {'train': 150.0},
            'profile': None, 'parent': None}


def selfcheck(pkg_dir: str | Path, *, history_receipts=(), tasks=None, require_core_skills: bool = True) -> SelfcheckReport:
    root = Path(pkg_dir)
    errors, warnings, checked = [], [], {}
    mpath = root / 'manifest.json'
    if not mpath.is_file():
        return SelfcheckReport(False, ['manifest.json missing'], [], {})
    try:
        manifest = json.loads(mpath.read_text(encoding='utf-8'))
    except ValueError as exc:
        return SelfcheckReport(False, [f'manifest.json unparseable: {exc}'], [], {})
    if manifest.get('schema') != MANIFEST_SCHEMA:
        errors.append('manifest schema')
    # identity
    try:
        files = content_manifest(root)
        if files != manifest.get('files'):
            errors.append('manifest file list/hashes differ from content')
        if compute_digest(root) != manifest.get('digest'):
            errors.append('manifest digest differs from content digest')
    except Exception as exc:                                   # e.g. corrupt memory.db
        errors.append(f'cannot digest package: {type(exc).__name__}: {exc}')
    extra = sorted({p.name for p in root.iterdir()} - TOP_LEVEL)
    if extra:
        errors.append(f'unexpected top-level entries {extra}')
    total = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if total > LIMITS['total_bytes']:
        errors.append(f'package {total} bytes > {LIMITS["total_bytes"]}')
    # policy
    try:
        doc = yaml.safe_load((root / 'policy.yaml').read_text(encoding='utf-8'))
        errors += [f'policy: {e}' for e in policy_mod.check_policy(doc)]
        checked['policy'] = True
    except (OSError, yaml.YAMLError) as exc:
        errors.append(f'policy.yaml unreadable: {exc}')
    # skills
    skills = sorted(p for p in (root / 'skills').glob('*/SKILL.md')) if (root / 'skills').is_dir() else []
    names = [p.parent.name for p in skills]
    if len(skills) > LIMITS['max_skills']:
        errors.append(f'{len(skills)} skills > {LIMITS["max_skills"]}')
    for p in skills:
        if p.stat().st_size > LIMITS['skill_bytes']:
            errors.append(f'skill {p.parent.name} too large')
        errors += [f'skill {p.parent.name}: {e}' for e in check_skill(p.read_text(encoding='utf-8'), p.parent.name)]
    for p in (root / 'skills').rglob('*') if (root / 'skills').is_dir() else []:
        rel = p.relative_to(root / 'skills').as_posix()
        if p.is_file() and not (rel.endswith('/SKILL.md') and rel.count('/') == 1) and not (rel.startswith('_catalogue/') and rel.endswith('.md')):
            errors.append(f'unexpected file in skills/: {rel}')
    if require_core_skills:
        missing = sorted(set(A0_SKILLS) - set(names))
        if missing:
            errors.append(f'core skills removed: {missing}')
    checked['skills'] = names
    # prompts
    prompts = {p.name: p.read_text(encoding='utf-8') for p in (root / 'prompts').glob('*')} if (root / 'prompts').is_dir() else {}
    for name in REQUIRED_PROMPTS:
        if name not in prompts:
            errors.append(f'prompt {name} missing')
    for name, text in prompts.items():
        if name not in REQUIRED_PROMPTS:
            errors.append(f'unexpected prompt file {name}')
            continue
        if len(text.encode('utf-8')) > LIMITS['prompt_bytes']:
            errors.append(f'prompt {name} too large')
        for ph in PROMPT_PLACEHOLDERS.get(name, ()):
            if ph not in text:
                errors.append(f'prompt {name} lost placeholder {ph}')
        rendered = fill(text, ACTION_MENU=WEAK_MENU if name.startswith('weak') else STRONG_MENU, MODE='strong', LAYERS='L0,L1',
                        PATCH_LINE_LIMIT=300, TEMPLATES='[]', CHECKLIST='\n'.join(DEFAULT_CHECKLIST))
        if system_arg_units(rendered) > MAX_SYSTEM_ARG_UTF16_UNITS:
            errors.append(f'prompt {name} exceeds the CLI argv budget when rendered')
    # tools on historical receipts
    tools = {p.stem: p.read_text(encoding='utf-8') for p in (root / 'tools').glob('*.analysis')} if (root / 'tools').is_dir() else {}
    stray = [p.name for p in (root / 'tools').iterdir() if p.suffix != '.analysis'] if (root / 'tools').is_dir() else []
    if stray:
        errors.append(f'unexpected files in tools/: {stray}')
    if len(tools) > LIMITS['max_tools']:
        errors.append(f'{len(tools)} tools > {LIMITS["max_tools"]}')
    views = [minimal_view()]
    for item in history_receipts:
        rec, parent, task_raw = (item if isinstance(item, tuple) else (item, None, None))
        v = receipt_view(rec, parent, task_raw)
        if v:
            views.append(v)
    runs = 0
    for name, src in tools.items():
        try:
            pure_analysis.validate_source(src)
        except pure_analysis.DriverRejection as exc:
            errors.append(f'tool {name}: invalid source: {exc}')
            continue
        for i, v in enumerate(views):
            try:
                pure_analysis.run(src, v)
                runs += 1
            except pure_analysis.DriverRejection as exc:
                errors.append(f'tool {name}: fails on receipt view {i}: {exc}')
                break
    checked['tool_runs'] = runs
    checked['history_receipts'] = len(views) - 1
    # memory
    mem = root / 'memory.db'
    try:
        db = MemoryDB(mem, read_only=True)
        if not db.schema_ok():
            errors.append('memory.db schema mismatch')
        checked['memory_entries'] = db.count()
        db.rows()
        db.close()
    except Exception as exc:
        errors.append(f'memory.db unusable: {type(exc).__name__}: {exc}')
    # answer isolation
    audit = audit_package(root, tasks=tasks)
    if not audit.passed:
        errors += [f'isolation: {f["kind"]} {f["term"]!r} in {f["file"]}' for f in audit.findings[:20]]
    checked['audit_scanned'] = len(audit.scanned)
    return SelfcheckReport(ok=not errors, errors=errors, warnings=warnings, checked=checked)
