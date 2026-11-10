"""The toy project (examples/toy_cpu) as the system tests use it: a fresh copy per test, short runs, run-directory helpers."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
TOY = REPO / 'examples' / 'toy_cpu'


def make_toy(dst: Path, *, steps: int = 12, window=(5, 12), save_at: int = 6, screens: int | None = None, edit=None,
             manifest: bool = True) -> Path:
    """A copy of the toy with `steps` updates per run; `edit(doc)` changes the manifest document before it is written;
    manifest=False leaves the repository without infrarsi.yaml (for `init`)."""
    shutil.copytree(TOY, dst, ignore=shutil.ignore_patterns('.infrarsi', '__pycache__'))
    doc = yaml.safe_load((dst / 'infrarsi.yaml').read_text(encoding='utf-8'))
    doc['workload'] = {'steps': steps, 'window': list(window), 'invariants': {'global_batch': 'micro_batch_size * grad_accum'}}
    doc['checkpoint']['save_at'] = save_at
    if screens:
        doc['search']['budget']['screens'] = screens
    if edit:
        edit(doc)
    if manifest:
        write_manifest(dst, doc)
    else:
        (dst / 'infrarsi.yaml').unlink()
    return dst


def manifest(dst: Path) -> dict:
    return yaml.safe_load((dst / 'infrarsi.yaml').read_text(encoding='utf-8'))


def write_manifest(dst: Path, doc: dict) -> None:
    (dst / 'infrarsi.yaml').write_text(yaml.safe_dump(doc, sort_keys=False), encoding='utf-8')


def runs(dst: Path) -> list[Path]:
    d = dst / '.infrarsi' / 'runs'
    return sorted(p for p in d.iterdir() if p.is_dir()) if d.is_dir() else []


def latest_run(dst: Path) -> Path:
    rs = runs(dst)
    assert rs, f'no run under {dst}'
    return rs[-1]


def result(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / 'result.json').read_text(encoding='utf-8'))


def report(run_dir: Path) -> str:
    return (Path(run_dir) / 'REPORT.md').read_text(encoding='utf-8')


def receipts(run_dir: Path) -> dict:
    """{candidate id: receipt} of the search's receipts."""
    d = Path(run_dir) / 'search' / 'receipts'
    return {p.stem: json.loads(p.read_text(encoding='utf-8')) for p in sorted(d.glob('*.json'))} if d.is_dir() else {}
