"""Record SHA-256 of the frozen design deliverables into execution/DESIGN_FREEZE.json."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'reports' / 'infra_rsi_system_20261020'

CORRECTIONS = [
    'WINH has direct ModelScope egress (2026-10-20 05:30 +08:00: WINH0 downloaded 2.16 GB in 60 s, '
    'about 36 MB/s, no proxy environment); downloads run directly on WINH instead of the WIN0 relay '
    'described in design section 7.1. The design text itself is not edited.',
]


def main():
    files = [ROOT / 'design' / 'EXPERIMENT_DESIGN_FABLE51.md', ROOT / 'design' / 'RUN_MATRIX.yaml',
             ROOT / 'design' / 'reward.yaml', ROOT / 'review' / 'FINAL_PLAN.md',
             ROOT / 'review' / 'FINAL_REVIEW_FABLE51.md']
    files += sorted((ROOT / 'design' / 'task_specs').glob('*.yaml'))
    freeze = {
        'schema': 'infrarsi-design-freeze-v1',
        'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'authority': ['review/FINAL_PLAN.md', 'review/FINAL_REVIEW_FABLE51.md'],
        'files': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'execution_corrections': CORRECTIONS,
    }
    out = ROOT / 'execution' / 'DESIGN_FREEZE.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f'{out} already exists; refusing to overwrite a freeze record')
    out.write_text(json.dumps(freeze, indent=2), encoding='utf-8')
    print(f'froze {len(freeze["files"])} files -> {out}')


if __name__ == '__main__':
    main()
