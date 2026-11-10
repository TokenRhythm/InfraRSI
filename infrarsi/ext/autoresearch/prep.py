"""One-time ENV-C preparation on WINH (no GPU): the frozen tokenizer directory and the asset hashes of assets.json.

  python -m infrarsi.ext.autoresearch.prep assets [--node WINH0] [--determinism]
  python -m infrarsi.ext.autoresearch.prep reference --receipt <signed c0 receipt.json>   (the ENV-C anchor)

assets: copies R2's tokenizer (the one the reproduced baseline used; the original program trains it once in
prepare.py) into datasets/climbmix_tokenizer/ (root, read-only), optionally re-trains it with the original prepare.py
in a staging HOME to record whether the training is deterministic, and hashes the data shards, the tokenizer and the
FA3 kernel sources into assets.json. reference: records a verified solo baseline c0 (base train.py, D-AR-7 loader, the
arms' node) as the ENV-C anchor (AMENDMENT_EXT_002): val_bpb and steps for tau, AUC_budget and the calibration gate,
its first 10 printed losses for check (4), its optimizer parameter groups for the runtime check.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys

from ...harness import signing
from ..common.remote_exec import WinhNodeOps, q
from . import source
from .task import ASSETS_PATH

R2_TOKENIZER = f'{source.NODE_RUNS}/r2_20261021a/home/.cache/autoresearch/tokenizer'
STAGING = f'{source.NODE_RUNS}/_prep'


def freeze_tokenizer(ops) -> dict:
    have = ops.hash_files([f'{source.NODE_TOKENIZER}/{f}' for f in source.TOKENIZER_FILES])
    want = source.R2['tokenizer_sha256']
    if all(have[f'{source.NODE_TOKENIZER}/{f}'] == want[f] for f in source.TOKENIZER_FILES):
        return {'tokenizer': 'present', 'sha256': want}
    r2 = ops.hash_files([f'{R2_TOKENIZER}/{f}' for f in source.TOKENIZER_FILES])
    if any(r2[f'{R2_TOKENIZER}/{f}'] != want[f] for f in source.TOKENIZER_FILES):
        raise RuntimeError(f'R2 tokenizer differs from its recorded sha256: {r2}')
    res = ops.sh(f'mkdir -p {q(source.NODE_TOKENIZER)} && cp {q(R2_TOKENIZER)}/tokenizer.pkl {q(R2_TOKENIZER)}/token_bytes.pt '
                 f'{q(source.NODE_TOKENIZER)}/ && chmod 755 {q(source.NODE_TOKENIZER)} && chmod 444 {q(source.NODE_TOKENIZER)}/*', timeout=120)
    if res['code']:
        raise RuntimeError(res['stderr'][-500:])
    return {'tokenizer': 'copied from R2', 'sha256': want}


def tokenizer_determinism(ops) -> dict:
    """Re-train the tokenizer with the original prepare.py (CPU, ~1 min) and compare with R2's."""
    home = f'{STAGING}/home'
    cmd = (f'rm -rf {q(STAGING)} && mkdir -p {q(home)}/.cache/autoresearch {q(STAGING)}/repo && '
           f'ln -s {q(source.NODE_DATA)} {q(home)}/.cache/autoresearch/data && cp {q(source.NODE_SRC)}/prepare.py {q(STAGING)}/repo/ && '
           f'cd {q(STAGING)}/repo && env HOME={q(home)} CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 {q(source.NODE_PYTHON)} prepare.py --num-shards 10 '
           f'> {q(STAGING)}/prepare.log 2>&1; echo rc=$?')
    res = ops.sh(cmd, timeout=900)
    got = ops.hash_files([f'{home}/.cache/autoresearch/tokenizer/{f}' for f in source.TOKENIZER_FILES])
    same = {f: got[f'{home}/.cache/autoresearch/tokenizer/{f}'] == source.R2['tokenizer_sha256'][f] for f in source.TOKENIZER_FILES}
    return {'rc': res['stdout'].strip()[-20:], 'deterministic': all(same.values()), 'same': same,
            'retrained_sha256': {f: got[f'{home}/.cache/autoresearch/tokenizer/{f}'] for f in source.TOKENIZER_FILES}}


def measure(ops) -> dict:
    data = ops.hash_files([f'{source.NODE_DATA}/{s}' for s in source.DATA_SHARDS])
    tok = ops.hash_files([f'{source.NODE_TOKENIZER}/{f}' for f in source.TOKENIZER_FILES])
    ker = ops.hash_files([f'{source.NODE_KERNEL}/{f}' for f in source.KERNEL_FILES])
    so = [n for n in (ops.call('listdir', path=source.NODE_KERNEL) or []) if n.endswith('.so')]
    so_sha = ops.hash_files([f'{source.NODE_KERNEL}/{n}' for n in so])
    out = {'data': {s: data[f'{source.NODE_DATA}/{s}'] for s in source.DATA_SHARDS},
           'tokenizer': {f: tok[f'{source.NODE_TOKENIZER}/{f}'] for f in source.TOKENIZER_FILES},
           'kernel': {f: ker[f'{source.NODE_KERNEL}/{f}'] for f in source.KERNEL_FILES},
           'kernel_so': {n: so_sha[f'{source.NODE_KERNEL}/{n}'] for n in so}}
    missing = [f'{k}/{n}' for k, v in out.items() for n, s in v.items() if s is None]
    if missing:
        raise RuntimeError(f'assets missing on {ops.node}: {missing}')
    return out


def cmd_assets(node: str, determinism: bool) -> dict:
    ops = WinhNodeOps(node)
    info = {'tokenizer_freeze': freeze_tokenizer(ops)}
    if determinism:
        info['tokenizer_determinism'] = tokenizer_determinism(ops)
    assets = {'schema': 'ext-autoresearch-assets-v1', 'node': node, 'measured_utc': _dt.datetime.now(_dt.timezone.utc).isoformat(),
              **measure(ops), 'prep': info}
    old = json.loads(ASSETS_PATH.read_text(encoding='utf-8')) if ASSETS_PATH.exists() else {}
    assets['reference'] = old.get('reference')
    ASSETS_PATH.write_text(json.dumps(assets, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    return assets


def cmd_reference(receipt_path: str) -> dict:
    """Record the unmodified baseline's run as the runtime check's reference (signed, valid, base train.py only)."""
    rec = json.loads(open(receipt_path, encoding='utf-8').read())
    if not signing.verify(rec) or rec.get('valid') is not True or rec.get('env') != 'autoresearch':
        raise ValueError('reference must be a valid, signed ENV-C receipt')
    if (rec.get('files') or {}).get('train.py') != source.BASE_TRAIN_SHA256:
        raise ValueError('reference must be the unmodified baseline train.py')
    meta, summ = rec.get('run_meta') or {}, rec.get('summary') or {}
    losses = {int(k): v for k, v in (summ.get('first_losses') or {}).items()}
    if sorted(losses) != list(range(10)):
        raise ValueError('reference must have printed the losses of steps 0-9')
    assets = json.loads(ASSETS_PATH.read_text(encoding='utf-8'))
    assets['reference'] = {'receipt': receipt_path.replace('\\', '/'), 'attempt_id': rec['attempt_id'], 'node': rec['node'],
                           'gpu': rec['gpus'], 'val_bpb': rec['metric']['value'], 'num_steps': summ.get('num_steps'),
                           'prepare_variant': 'h2d_sync' if 'D-AR-7' in (rec.get('deviations') or []) else 'upstream',
                           'optimizer_groups': [{k: g.get(k) for k in ('kind', 'n_tensors', 'numel', 'initial_lr', 'betas', 'eps', 'ns_steps', 'beta2')}
                                                for g in meta.get('optimizer_groups') or []],
                           'first_losses': [losses[i] for i in range(10)]}
    ASSETS_PATH.write_text(json.dumps(assets, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    return assets['reference']


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('assets')
    a.add_argument('--node', default='WINH0')
    a.add_argument('--determinism', action='store_true')
    r = sub.add_parser('reference')
    r.add_argument('--receipt', required=True)
    args = ap.parse_args(argv)
    out = cmd_assets(args.node, args.determinism) if args.cmd == 'assets' else cmd_reference(args.receipt)
    json.dump(out, sys.stdout, indent=1, default=str)
    print()


if __name__ == '__main__':
    main()
