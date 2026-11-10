"""G0 download check (run on WINH0, which reaches ModelScope): compare every downloaded file with the
ModelScope file list (size and, where published, sha256). Read-only on /data/infrarsi.

Usage: $VENV/bin/python verify_downloads.py --specs <task_specs dir> --out /data/infrarsi/logs/download/VERIFY.json
(token, if needed, from a chmod-600 env file: `set -a; . ~/.ms_token.env; set +a`)
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOTS = ('/data/infrarsi/models', '/data/infrarsi/datasets/raw')
EXCLUDED = {'LLM-Research/Meta-Llama-3.1-8B': ['original/*']}      # mirrors the ms_fetch --exclude flags


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(16 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def remote_files(api, kind: str, repo: str) -> dict:
    """{relpath: (size, sha256 or None)} from the hub."""
    if kind == 'model':
        rows = api.get_model_files(repo, recursive=True)
    else:
        rows = api.get_dataset_files(repo, recursive=True)
    out = {}
    for r in rows:
        if r.get('Type') == 'tree':
            continue
        out[r['Path']] = (r.get('Size'), r.get('Sha256') or None)
    return out


def check_repo(api, d: Path, workers: int) -> dict:
    done = json.loads((d / 'DOWNLOAD_DONE.json').read_text())
    repo, kind = done['repo'], done['kind']
    res = {'dir': str(d), 'repo': repo, 'kind': kind, 'problems': []}
    try:
        files = remote_files(api, kind, repo)
    except Exception as e:                                            # noqa: BLE001
        res['problems'].append(f'listing failed: {e!r}')
        return res
    skip = EXCLUDED.get(repo, [])
    files = {k: v for k, v in files.items()                      # 'original/*' also covers the 'original' entry
             if not any(fnmatch.fnmatch(k, pat) or fnmatch.fnmatch(k + '/_', pat) for pat in skip)}
    res['remote_files'], res['remote_bytes'] = len(files), sum(int(v[0] or 0) for v in files.values())

    def one(item):
        rel, (size, sha) = item
        p = d / rel
        if not p.is_file():
            return rel, 'missing'
        if size is not None and p.stat().st_size != int(size):
            return rel, f'size {p.stat().st_size} != {size}'
        if sha and sha256(p) != sha:
            return rel, 'sha256 mismatch'
        return rel, 'ok' if sha else 'ok(size only)'

    with ThreadPoolExecutor(workers) as ex:
        results = dict(ex.map(one, sorted(files.items())))
    res['status'] = results
    res['ok_sha'] = sum(v == 'ok' for v in results.values())
    res['ok_size_only'] = sum(v == 'ok(size only)' for v in results.values())
    res['problems'] += [f'{k}: {v}' for k, v in results.items() if not v.startswith('ok')]
    return res


def expected_repos(specs_dir: str) -> dict:
    """{repo_id: files} the frozen task specs need: None = every hub file of a model repo (minus EXCLUDED),
    else the dataset files named in `data.file`. Local (non-hub) models and data are not checked here."""
    import yaml
    out = {}
    for p in sorted(Path(specs_dir).glob('*.yaml')):
        t = yaml.safe_load(p.read_text())
        repo = t['model']['repo_id']
        if not repo.startswith('local:'):
            out[repo] = None
        d = t.get('data') or {}
        words = [w for w in (d.get('source') or '').split() if w.count('/') == 1 and not w.startswith(('local:', '/'))]
        if words and d.get('file'):
            out.setdefault(words[-1], set()).add(d['file'])
    return out


def repo_ok(r: dict, files) -> bool:
    if files is None:                       # model: every listed (non-excluded) file present with the hub size
        return r.get('remote_files', 0) > 0 and not r['problems']
    st = r.get('status') or {}
    return all(st.get(f) == 'ok' for f in files)           # dataset: each named file present with the hub sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--specs', required=True, help='frozen task_specs dir; every repo they name must verify')
    a = ap.parse_args()
    from modelscope.hub.api import HubApi
    api = HubApi()
    if os.environ.get('MODELSCOPE_API_TOKEN'):
        api.login(os.environ['MODELSCOPE_API_TOKEN'])
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    repos = [check_repo(api, d, a.workers) for root in ROOTS for d in sorted(Path(root).iterdir())
             if (d / 'DOWNLOAD_DONE.json').is_file()]
    want = expected_repos(a.specs)
    by_repo = {r['repo']: r for r in repos}
    verdict = {repo: (repo in by_repo and repo_ok(by_repo[repo], files)) for repo, files in want.items()}
    out = {'schema': 'infrarsi-download-verify-v1', 'started_utc': started,
           'finished_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'repos': repos,
           'expected': {k: (sorted(v) if v else None) for k, v in want.items()}, 'verdict': verdict,
           'all_ok': bool(want) and all(verdict.values())}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for r in repos:
        print(f"{r['repo']:45s} files={r.get('remote_files')} sha_ok={r.get('ok_sha')} size_only={r.get('ok_size_only')} "
              f"problems={len(r['problems'])} {r['problems'][:3]}")
    print('ALL_OK' if out['all_ok'] else 'PROBLEMS')


if __name__ == '__main__':
    main()
