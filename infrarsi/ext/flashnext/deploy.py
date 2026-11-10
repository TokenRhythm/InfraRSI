"""E2 of the F1 variant layer (driver side): the frozen manifest and the WINH deployment. Writes only under
/data/infrarsi/ext; the formal-54 trees, the source tree and the formal compile cache are read only.

    custom_runtime_local(dst) -> the patched + re-rooted runtime copy (variant.custom_runtime(root=FN_ROOT))
    manifest() -> frozen_manifest.json content (deterministic); `python -m infrarsi.ext.flashnext.deploy manifest`
    deploy(ops) -> receipt: FN_SRC/{runtime,node}/ + switch_space.json + frozen_manifest.json uploaded and verified;
                   FN_ROOT/{metadata,runs,phase_gate}, FN_LOCKS, FN_CACHE created; the warm compile cache copied
                   (cp -a from the formal cache, read-only source) unless already present.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

from . import switch_space as S
from . import variant as V

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / 'frozen_manifest.json'
NODE_FILES = ('fn_launch_node.py',)
FORMAL_CACHE = V.ROOT + '/cache'
# Read-only remote files a screen executes or reads (sha256 read on WINH0, 2026-10-21; checked by preflight).
REMOTE_READ_ONLY = {
    V.ROOT + '/deployment/flash_mcore_env_portable.sh': '850725ac62b7281bb5b3aec2d364ad0448e2c351ad374322988109af1c24afb3',
    V.ROOT + '/deployment/sitecustomize.py': '4f28901af2d64c700b3756c17c270b00b1a942fb49af8e426c498b45f469562e',
    V.ROOT + '/deployment/portable_path_bootstrap.py': 'f5bf54f3521ba92d102edda446856e570b2116f1e88c515d1292a56df421079f',
    V.ROOT + '/deployment/portable_path_manifest.json': 'ff276d1bac5c3ea9465b37321ce4c5c70d77266bf5144c65c38dd99f3ccf3f66',
    V.ROOT + '/deployment/native_swiglu_binary_adapter.py': '88659d5a263c6d970da57e81d67f8d1251b780fb3f508ff9864f214e7eafcfdd',
    '/data/zehua_flashnext_replay_20261013/source_tree/qwen-sft/runs/qwen38_flash_mcore_ad_20260908_v1/train_flash_strategy_20260925.py':
        '4ab14ced3243dfe0398fbbd0630eb3ec758e59102541b7d186bad6af52b8d79f',
    '/data/zehua_flashnext_replay_20261013/source_tree/qwen-sft/artifacts/flash_pool_budget_views_20261006_v2/luna_q50/manifest.json':
        'de1fc04b3469ee704ce3e8e05bff3fc66f18082082ad0022218a176d6c625910',
    '/data/zehua_flashnext_replay_20261013/source_tree/qwen-sft/runs/qwen38_flash_mcore_ad_20260908_v1/tp8_p1_pool_luna_q50_h1024_mtp01_20261006_v1.sources.json':
        '635ad3683fe2df7585bd9a25faab844be092b2c140351c1163707d7f94576426',
    '/data/zehua_flashnext_replay_20261013/source_tree/qwen-sft/runs/h100_scale_20260923_v1/pool_exploration_20261006_v1/arms/luna_q50_h1024/UPDATE_SUPERVISION.json':
        '95afc499235a69baae5846e9a8df57a91178a21ab9b216119c9773ff7fc9b541',
    '/data/zehua_flashnext_replay_20261013/source_tree/qwen-sft/runs/h100_scale_20260923_v1/flash_import_snapshot_parent_20260928.py':
        '8c0b643b147b2412181bfba3574e880d5bf1f317947afeac8e6040733043e72b',
    '/data/zehua_flashnext_replay_20261013/source_tree/qwen-sft/runs/h100_scale_20260923_v1/flash_import_snapshot_worker_20260928.py':
        'fb90967ce25d65dee14ac4ae5b3df97c825705e4c63aee24703727ba2f78dc87',
}


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def custom_runtime_local(dst) -> dict:
    return V.custom_runtime(dst, root=V.FN_ROOT)


def manifest() -> dict:
    with tempfile.TemporaryDirectory(prefix='fnrt') as tmp:
        rt = custom_runtime_local(Path(tmp) / 'rt')
    freeze = V.read(V.FROZEN_DIR / 'CODE_FREEZE.json')
    return {
        'schema': 'flashnext-frozen-manifest-v1',
        'ruling': 'AMENDMENT_EXT_003 Q3 (frozen hashes) and Q4 D-F1-1',
        'formal54': {'code_freeze_sha256': V.CODE_FREEZE_SHA256,
                     'runtime_inventory_sha256': hashlib.sha256(V.encode(V.inventory(V.FROZEN_RUNTIME))).hexdigest(),
                     'builder_sha256': freeze['builder_sha256'], 'builder_input_sha256': freeze['builder_input_sha256']},
        'source_pins': {'training_source_sha256': '4ab14ced3243dfe0398fbbd0630eb3ec758e59102541b7d186bad6af52b8d79f',
                        'data_manifest_sha256': 'de1fc04b3469ee704ce3e8e05bff3fc66f18082082ad0022218a176d6c625910',
                        'sources_sha256': '635ad3683fe2df7585bd9a25faab844be092b2c140351c1163707d7f94576426',
                        'update_supervision_sha256': '95afc499235a69baae5846e9a8df57a91178a21ab9b216119c9773ff7fc9b541'},
        'variant_runtime': {k: rt[k] for k in ('root', 'names', 'edits', 'changed_files', 'inventory', 'inventory_sha256')},
        'switch_space_sha256': S.sha256(),
        'node_files': {n: sha(HERE / 'node' / n) for n in NODE_FILES},
        'remote_read_only': dict(REMOTE_READ_ONLY),
        'remote_layout': {'src': V.FN_SRC, 'runs': V.FN_ROOT, 'locks': V.FN_LOCKS, 'cache': V.FN_CACHE,
                          'phase_gate': V.FN_PHASE, 'cache_copied_from': FORMAL_CACHE},
    }


def manifest_bytes() -> bytes:
    return (json.dumps(manifest(), indent=1, sort_keys=True) + '\n').encode('utf-8')


def load_manifest(path: Path = MANIFEST) -> dict:
    raw = Path(path).read_bytes()
    if raw != manifest_bytes():
        raise ValueError(f'{path} differs from the generated manifest: the variant layer drifted')
    return json.loads(raw)


def deploy(ops, *, copy_cache: bool = True, log=print) -> dict:
    """Upload the variant layer to FN_SRC and prepare the ext directories on WINH (shared /data: one node suffices)."""
    m = load_manifest()
    with tempfile.TemporaryDirectory(prefix='fnrt') as tmp:
        rt = custom_runtime_local(Path(tmp) / 'rt')
        assert rt['inventory'] == m['variant_runtime']['inventory']
        files = {f'{V.FN_SRC}/runtime/{n}': (Path(rt['runtime']) / n).read_bytes() for n in sorted(rt['inventory'])}
    files.update({f'{V.FN_SRC}/node/{n}': (HERE / 'node' / n).read_bytes() for n in NODE_FILES})
    files[f'{V.FN_SRC}/switch_space.json'] = S.PATH.read_bytes()
    files[f'{V.FN_SRC}/frozen_manifest.json'] = MANIFEST.read_bytes()
    dirs = [f'{V.FN_SRC}/runtime', f'{V.FN_SRC}/node', V.FN_ROOT + '/metadata', V.FN_ROOT + '/runs', V.FN_PHASE,
            V.FN_LOCKS, V.FN_CACHE]
    assert all(p.startswith('/data/infrarsi/ext/') for p in list(files) + dirs)
    ops.mkdirs(dirs)
    names = sorted(files)
    for i in range(0, len(names), 24):
        ops.write_files({p: files[p] for p in names[i:i + 24]})
    got = ops.hash_files(names)
    bad = [p for p in names if got.get(p) != hashlib.sha256(files[p]).hexdigest()]
    if bad:
        raise RuntimeError(f'deployment verification failed: {bad[:4]}')
    log(f'uploaded and verified {len(names)} files under {V.FN_SRC}')
    cache = {'copied': False}
    if copy_cache:
        probe = ops.sh(f'ls -A {V.FN_CACHE} | head -1; du -sk {FORMAL_CACHE} | cut -f1', timeout=300)
        lines = (probe.get('stdout') or '').split()
        already = len(lines) == 2
        if not already:
            res = ops.sh(f'cp -a {FORMAL_CACHE}/. {V.FN_CACHE}/ && du -sk {V.FN_CACHE} | cut -f1', timeout=3600)
            if res.get('code'):
                raise RuntimeError(f'cache copy failed: {res.get("stderr", "")[-800:]}')
            cache = {'copied': True, 'kib': (res.get('stdout') or '').strip()}
        else:
            cache = {'copied': False, 'note': 'cache directory already populated', 'formal_kib': lines[-1]}
        log(f'cache: {cache}')
    return {'status': 'DEPLOYED', 'files': len(names), 'manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            'switch_space_sha256': S.sha256(), 'cache': cache, 'runtime_inventory_sha256': m['variant_runtime']['inventory_sha256']}


if __name__ == '__main__':
    if sys.argv[1:] == ['manifest']:
        MANIFEST.write_bytes(manifest_bytes())
        print(MANIFEST, sha(MANIFEST))
    elif sys.argv[1:2] == ['deploy']:
        from ..common.remote_exec import WinhNodeOps
        print(json.dumps(deploy(WinhNodeOps(sys.argv[2] if len(sys.argv) > 2 else 'WINH0')), indent=1))
    else:
        raise SystemExit('usage: python -m infrarsi.ext.flashnext.deploy manifest | deploy [WINHk]')
