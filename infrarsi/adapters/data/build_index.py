"""Prepare every packed dataset the public tasks need and write <out_root>/INDEX.json (task id -> manifest).

  cd <code root> && $VENV/bin/python -m infrarsi.adapters.data.build_index --specs <dir of task yaml>
Re-running skips finished outputs (sha256-verified), picks up models whose download completed and rewrites
INDEX.json; --wait-hours H keeps polling every 5 minutes until every task is prepared or H hours pass.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path

from infrarsi.contracts import PLAN_SEEDS, RemotePaths, TaskSpec, file_digest, utc_now
from .dataset import PackedTHDDataset
from .packing import PACKER_VERSION, dumps, output_name, prepare
from .sources import DATA_KEYS
from .tokenizer import HFTokenizer, tokenizer_files

INDEX_SCHEMA = 'infrarsi-packed-index-v1'
TASK_IDS = ('S1', 'S2', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'X1', 'X2', 'X3')
UPDATES = 48            # packs_total = packs_per_update x 48 (design §1.2)


def log(*args):
    print(time.strftime('%H:%M:%S'), *args, flush=True)


def tokenizer_sources(models_root, repo_ids) -> dict:
    """repo -> (tokenizer dir or None, note). A model counts once DOWNLOAD_DONE.json exists; one still downloading is
    accepted when its tokenizer files are byte-identical to a finished model's, whose dir is then used."""
    dirs = {r: Path(models_root) / r.replace('/', '__') for r in sorted(set(repo_ids))}
    done = {r: tokenizer_files(d) for r, d in dirs.items()
            if (d / 'DOWNLOAD_DONE.json').is_file() and (d / 'tokenizer.json').is_file()}
    out = {}
    for r, d in dirs.items():
        if r in done:
            out[r] = (d, 'download complete')
            continue
        files = tokenizer_files(d) if d.is_dir() else {}
        twin = next((x for x in sorted(done) if 'tokenizer.json' in files and done[x] == files), None)
        out[r] = ((dirs[twin], f'download in progress; tokenizer files byte-identical to {twin}') if twin
                  else (None, 'model download not complete (no DOWNLOAD_DONE.json)'))
    return out


def raw_problems(raw_root, data_keys) -> dict:
    out = {}
    for key in sorted(data_keys):
        dk = DATA_KEYS[key]
        p = dk.raw_path(raw_root)
        done = (Path(raw_root) / dk.repo.replace('/', '__') / 'DOWNLOAD_DONE.json').is_file()
        ok = done and p.is_file() and p.stat().st_size == dk.bytes
        out[key] = None if ok else f'raw file {p} missing, still downloading, or not {dk.bytes} bytes'
    return out


def task_entry(spec, manifest_path, tok_dir, note, models_root, _cache={}) -> dict:
    m = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    ppu = int(spec.raw['workload']['packs_per_update'])
    if (manifest_path, ppu) not in _cache:
        sup, order_digest = {}, {}
        for plan in sorted(PLAN_SEEDS):
            with PackedTHDDataset(manifest_path, plan) as ds:
                sup[plan] = [ds.update_token_counts(u, ppu)['supervised_tokens'] for u in range(UPDATES)]
                order_digest[plan] = ds.order_digest
        _cache[manifest_path, ppu] = sup, order_digest
    sup, order_digest = _cache[manifest_path, ppu]
    c = m['counts']
    repo = spec.raw['model']['repo_id']
    return {'manifest': manifest_path, 'manifest_sha256': file_digest(manifest_path), 'task_digest': spec.digest,
            'model': repo, 'model_dir': str(Path(models_root) / repo.replace('/', '__')),
            'tokenizer_dir': str(tok_dir), 'tokenizer_note': note,
            'tokenizer_digest': m['tokenizer']['digest'], 'data_key': m['data_key'], 'kind': m['kind'],
            'seq_len': m['seq_len'], 'packs_per_update': ppu, 'packs_total': m['packs_total'], 'updates': UPDATES,
            'physical_tokens': c['physical_tokens'], 'supervised_tokens': c['supervised_tokens'],
            'real_tokens': c['real_tokens'], 'dropped_samples': c['dropped_samples'],
            'invalid_samples': c['invalid_samples'], 'physical_tokens_per_update': ppu * m['seq_len'],
            'supervised_tokens_per_update': sup, 'data_order_digest': order_digest}


def build(specs: dict, a) -> dict:
    """One pass: prepare what is ready, rewrite INDEX.json, return the pending tasks."""
    t_all = time.time()
    raw = raw_problems(a.raw_root, {s.raw['data']['key'] for s in specs.values()})
    srcs = tokenizer_sources(a.models_root, [s.raw['model']['repo_id'] for s in specs.values()])
    toks = {str(d): HFTokenizer(d) for d, _ in srcs.values() if d is not None}
    canon = {}                                   # digest -> first tokenizer dir (sorted) that has it
    for d in sorted(toks):
        canon.setdefault(toks[d].digest, d)
    tasks, pending, built = {}, {}, {}
    for tid, spec in specs.items():
        w, data, repo = spec.raw['workload'], spec.raw['data'], spec.raw['model']['repo_id']
        key, seq, ppu, total = data['key'], spec.seq_len, int(w['packs_per_update']), int(w['packs_total'])
        if total != ppu * UPDATES or int(data['seq_len']) != seq:
            raise ValueError(f'{tid}: packs_total {total} != {ppu} x {UPDATES} or data/workload seq_len differ')
        d, note = srcs[repo]
        if raw[key] or d is None:
            pending[tid] = {'model': repo, 'reason': raw[key] or note, 'data_key': key, 'seq_len': seq,
                            'packs_per_update': ppu, 'packs_total': total, 'command': a.command}
            continue
        tok_dir = canon[toks[str(d)].digest]
        tok = toks[tok_dir]
        combo = (key, tok.digest, seq, total)
        if combo not in built:
            t0 = time.time()
            existed = (Path(a.out_root) / output_name(*combo)).exists() and not a.force
            built[combo] = prepare(key, tok_dir, a.out_root, seq, total, raw_root=a.raw_root, tokenizer=tok,
                                   force=a.force)
            c = json.loads(Path(built[combo]).read_text(encoding='utf-8'))['counts']
            log(f'{"reused" if existed else "built "} {Path(built[combo]).parent.name} in {time.time() - t0:.1f} s:',
                f'physical {c["physical_tokens"]} supervised {c["supervised_tokens"]} dropped {c["dropped_samples"]}',
                f'examined {c.get("samples_examined", c.get("documents_used"))}')
        tasks[tid] = task_entry(spec, built[combo], tok_dir, note, a.models_root)
    index = {'schema': INDEX_SCHEMA, 'generated_utc': utc_now(), 'builder': PACKER_VERSION, 'out_root': a.out_root,
             'tasks': tasks, 'pending': pending,
             'datasets': {Path(p).parent.name: sorted(t for t, e in tasks.items() if e['manifest'] == p)
                          for p in built.values()}}
    path = Path(a.out_root) / 'INDEX.json'
    path.parent.mkdir(parents=True, exist_ok=True)         # a first pass may have built nothing yet
    tmp = path.with_name(f'.INDEX.json.tmp-{os.getpid()}')
    tmp.write_text(dumps(index) + '\n', encoding='utf-8')
    os.replace(tmp, path)
    log(f'INDEX.json: {len(tasks)} tasks, {len(built)} datasets, pending {sorted(pending)}; '
        f'pass took {time.time() - t_all:.1f} s')
    return pending


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--specs', required=True, help='directory with <task>.yaml (design/task_specs copy)')
    ap.add_argument('--tasks', default=','.join(TASK_IDS))
    ap.add_argument('--models-root', default=RemotePaths.models)
    ap.add_argument('--raw-root', default=RemotePaths.datasets_raw)
    ap.add_argument('--out-root', default=RemotePaths.datasets_packed)
    ap.add_argument('--threads', type=int, default=0, help='tokenizer threads (RAYON_NUM_THREADS); 0 = all cores')
    ap.add_argument('--wait-hours', type=float, default=0.0)
    ap.add_argument('--force', action='store_true', help='rebuild outputs that already exist (first pass only)')
    a = ap.parse_args(argv)
    if a.threads:
        os.environ['RAYON_NUM_THREADS'] = str(a.threads)
    code_root = str(Path(__file__).resolve().parents[3])
    real_data = os.path.realpath('/data')         # WINH: /data -> /mnt/data; keep the /data spelling
    if real_data != '/data' and code_root.startswith(real_data + '/'):
        code_root = '/data' + code_root[len(real_data):]
    a.command = (f'cd {shlex.quote(code_root)} && {shlex.quote(sys.executable)} -m '
                 f'infrarsi.adapters.data.build_index ' + ' '.join(shlex.quote(x) for x in (argv or sys.argv[1:])))
    specs = {t: TaskSpec.load(Path(a.specs) / f'{t}.yaml') for t in a.tasks.split(',')}
    deadline = time.time() + a.wait_hours * 3600
    while True:
        pending = build(specs, a)
        a.force = False
        if not pending or time.time() >= deadline:
            return 0 if not pending else 3
        time.sleep(300)


if __name__ == '__main__':
    sys.exit(main())
