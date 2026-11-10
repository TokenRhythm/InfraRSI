"""OptimizerPackage directories (FINAL_PLAN 3.2): skills/, memory.db, policy.yaml, prompts/, tools/, manifest.json.

The package digest covers every file except manifest.json; memory.db enters through its logical (row) digest
so sqlite page layout never changes identity. Only rsi.reflect edits a package, and only during development;
G3 freezes A_R by digest (rsi.lineage).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import shutil
from pathlib import Path

from ..agent import memory as memory_mod
from ..agent.policy import default_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_SRC = REPO_ROOT / 'skills'
A0_DIR = Path(__file__).resolve().parent / 'a0'
MANIFEST_SCHEMA = 'infrarsi-optimizer-package-v1'
PACKAGE_DIRS = ('skills', 'prompts', 'tools')
PACKAGE_FILES = ('policy.yaml', 'memory.db', 'manifest.json')
REQUIRED_PROMPTS = ('proposer.md', 'weak_proposer.md', 'verifier.md', 'summarizer.md', 'reflect.md')
A0_SKILLS = ('discover-training-system', 'choose-parallel-layout', 'profile-whole-update', 'manage-memory-lifetimes',
             'recompute-policy', 'moe-ep-communication', 'optimize-data-path', 'optimize-training-kernels',
             'verify-training-semantics', 'validate-training-lifecycle', 'weak-model-protocol', 'learn-optimization-skills')


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_manifest(root: str | Path) -> dict[str, str]:
    """{relative path: sha256} for every package file except manifest.json; memory.db by logical digest."""
    root = Path(root)
    out = {}
    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel == 'manifest.json' or '__pycache__' in rel or rel.endswith(('-journal', '-wal', '-shm')):
            continue
        out[rel] = ('logical:' + memory_mod.logical_digest(p)) if rel == 'memory.db' else _sha(p)
    return out


def compute_digest(root: str | Path) -> str:
    rows = sorted(content_manifest(root).items())
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode('utf-8')).hexdigest()


def write_manifest(root: str | Path, *, label: str, parent_digest: str | None, producer: dict) -> dict:
    root = Path(root)
    files = content_manifest(root)
    manifest = {'schema': MANIFEST_SCHEMA, 'label': label, 'parent_digest': parent_digest, 'producer': producer,
                'created_utc': _dt.datetime.now(_dt.timezone.utc).isoformat(), 'files': files,
                'digest': hashlib.sha256(json.dumps(sorted(files.items()), separators=(',', ':')).encode('utf-8')).hexdigest()}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding='utf-8')
    return manifest


class OptimizerPackage:
    def __init__(self, root: Path, manifest: dict):
        self.root, self.manifest = root, manifest

    @classmethod
    def load(cls, root: str | Path, *, verify: bool = True) -> 'OptimizerPackage':
        root = Path(root)
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('schema') != MANIFEST_SCHEMA:
            raise ValueError(f'{root}: not an optimizer package')
        if verify:
            actual = compute_digest(root)
            if actual != manifest['digest']:
                raise ValueError(f'{root}: package content digest {actual[:12]} != manifest {manifest["digest"][:12]}')
        return cls(root, manifest)

    @property
    def digest(self) -> str:
        return self.manifest['digest']

    @property
    def label(self) -> str:
        return self.manifest.get('label', '')

    @property
    def skills_dir(self) -> Path:
        return self.root / 'skills'

    @property
    def memory_path(self) -> Path:
        return self.root / 'memory.db'

    @property
    def policy_text(self) -> str:
        return (self.root / 'policy.yaml').read_text(encoding='utf-8')

    @property
    def prompts(self) -> dict[str, str]:
        return {p.name: p.read_text(encoding='utf-8') for p in sorted((self.root / 'prompts').glob('*.md'))}

    @property
    def tools(self) -> dict[str, str]:
        return {p.stem: p.read_text(encoding='utf-8') for p in sorted((self.root / 'tools').glob('*.analysis'))}

    def skill_names(self) -> list[str]:
        return sorted(p.parent.name for p in self.skills_dir.glob('*/SKILL.md'))


def copy_package(src: str | Path, dst: str | Path) -> Path:
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise FileExistsError(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*-journal', '*-wal', '*-shm'))
    return dst


def without_memory(pkg: OptimizerPackage, dst: str | Path) -> OptimizerPackage:
    """E3 no_cross_task_memory: same skills/policy/prompts/tools, empty memory.db, new identity."""
    dst = copy_package(pkg.root, dst)
    (dst / 'memory.db').unlink()
    memory_mod.create(dst / 'memory.db')
    write_manifest(dst, label=f'{pkg.label}-nomem', parent_digest=pkg.digest,
                   producer={'type': 'ablation', 'name': 'no_cross_task_memory', 'from': pkg.digest})
    return OptimizerPackage.load(dst)


def build_a0(out_dir: str | Path, *, skills_src: str | Path = SKILLS_SRC, label: str = 'A0') -> OptimizerPackage:
    """A0: the 12 human-authored skills (+ the negative-results catalogue), empty memory, default policy, prompts, tools."""
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty')
    out.mkdir(parents=True, exist_ok=True)
    src = Path(skills_src)
    missing = [s for s in A0_SKILLS if not (src / s / 'SKILL.md').is_file()]
    if missing:
        raise FileNotFoundError(f'A0 skills missing in {src}: {missing}')
    for name in A0_SKILLS:
        (out / 'skills' / name).mkdir(parents=True)
        shutil.copyfile(src / name / 'SKILL.md', out / 'skills' / name / 'SKILL.md')
    if (src / '_catalogue').is_dir():
        shutil.copytree(src / '_catalogue', out / 'skills' / '_catalogue')
    (out / 'prompts').mkdir()
    for name in REQUIRED_PROMPTS:
        shutil.copyfile(A0_DIR / 'prompts' / name, out / 'prompts' / name)
    (out / 'tools').mkdir()
    for p in sorted((A0_DIR / 'tools').glob('*.analysis')):
        shutil.copyfile(p, out / 'tools' / p.name)
    (out / 'policy.yaml').write_text(default_yaml(), encoding='utf-8')
    memory_mod.create(out / 'memory.db')
    producer = {'type': 'human-authored (A0)', 'skills_source': str(src), 'skills': list(A0_SKILLS)}
    write_manifest(out, label=label, parent_digest=None, producer=producer)
    return OptimizerPackage.load(out)
