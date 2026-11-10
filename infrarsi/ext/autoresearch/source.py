"""ENV-C base tree and reference facts: autoresearch @228791f (karpathy/autoresearch, MIT) as fetched on this PC and
staged on WINH (sha256 identical, verified 2026-10-21), plus the R2 reproduction it is compared against.

The candidates' base train.py is the upstream file with the one-line D-AR-2 change (FA3 from kernels-community, the
ModelScope mirror served offline); everything else is upstream bytes. Node paths follow EXT_INTERFACES §7.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from ...contracts import REPO_ROOT

UPSTREAM = {'repo': 'karpathy/autoresearch', 'commit': '228791fb499afffb54b46200aca536f79142f117'}
LOCAL_SRC = REPO_ROOT / 'third_party_src' / 'autoresearch'
UPSTREAM_SHA256 = {
    'train.py': '2954175f4ac42ad65164aef40910ef953789abcd05a5cc886ac9ba5a00814414',
    'prepare.py': '4f2ba9cbb8ba8c4a3d35be405a913e2f3be3af9aea103ed52ef7b2a662058150',
    'program.md': '86cf987a5c381e46eefe0d0a82765223fd766d8d7acdc2afacfbbce15ecacece',
    'README.md': '3958fd4195ac2f98ed35c4eaa4f3028a335165ee24945e96426c408d25793a41',
    'pyproject.toml': '675c150a9e0769f0e39a43eb7d836934266fa348d6e2403521f1ff99f9b9f1af',
}
D_AR_2_OLD = 'repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"'
D_AR_2_NEW = 'repo = "kernels-community/flash-attn3"  # D-AR-2'
BASE_TRAIN_SHA256 = 'f67e371e7d7678cd2c3af3f9c869e302a24d4fa678293419f11a6fd50ed96a0a'   # = R2's repo/train.py

E = '/data/infrarsi/ext'
NODE_SRC = f'{E}/src/autoresearch'
NODE_ENV = f'{E}/envs/autoresearch'
NODE_PYTHON = f'{NODE_ENV}/bin/python'
NODE_DATA = f'{E}/datasets/climbmix'
NODE_TOKENIZER = f'{E}/datasets/climbmix_tokenizer'          # built once from the original prepare.py (= R2's)
NODE_KERNEL = f'{E}/kernels/kernels-community__flash-attn3/build/torch29-cxx11-cu128-x86_64-linux'
NODE_RUNS = f'{E}/runs/autoresearch'
KERNEL_REPO = 'kernels-community/flash-attn3'
KERNEL_VARIANT = 'torch29-cxx11-cu128-x86_64-linux'
KERNEL_REV = hashlib.sha1(b'modelscope:kernels-community/flash-attn3@master').hexdigest()   # R2 recipe
DATA_SHARDS = tuple(f'shard_{i:05d}.parquet' for i in range(10)) + ('shard_06542.parquet',)
TOKENIZER_FILES = ('tokenizer.pkl', 'token_bytes.pt')
KERNEL_FILES = ('__init__.py', '_ops.py', 'flash_attn_interface.py', 'flash_attn_config.py', 'metadata.json',
                'flash_attn3/__init__.py')
NODE_CUDA_ENV = {'LD_LIBRARY_PATH': '/usr/local/cuda/lib64', 'CUDA_HOME': '/usr/local/cuda'}   # the container env R2 ran with
DEVIATIONS = ('D-AR-1', 'D-AR-2', 'D-AR-3')       # data mirror, FA3 mirror, pip venv (REPRO_ISSUES #7, #14)

# R2 (r2_20261021a, WINH3, 1 H100): the reproduced baseline (REPRO_RESULTS.md) and its first printed losses.
R2 = {'run_id': 'r2_20261021a', 'node': 'WINH3', 'val_bpb': 0.9977, 'training_seconds': 300.1, 'total_seconds': 354.8,
      'peak_vram_mb': 45060.2, 'mfu_percent': 39.59, 'total_tokens_M': 497.0, 'num_steps': 948, 'num_params': 50332176,
      'num_params_M': 50.3, 'depth': 8, 'tok_per_sec': 1.63e6, 'num_flops_per_token': 2.390784e8, 'grad_accum_steps': 2,
      'model_config': {'sequence_len': 2048, 'vocab_size': 8192, 'n_layer': 8, 'n_head': 4, 'n_kv_head': 4, 'n_embd': 512,
                       'window_pattern': 'SSSL'},
      'first_losses': [9.011198, 8.854702, 8.442315, 7.971718, 7.578771, 7.248855, 6.998092, 6.787086, 6.606999, 6.455156],
      'tokenizer_sha256': {'tokenizer.pkl': '888c9fa865c72c25f9c5e9467136a9cc8655253db459fb97481deb326681c629',
                           'token_bytes.pt': '23517e0590d6aa9389c3263fab1f6062b2810b7c398a7b380291857639ac8d98'}}
PUBLISHED = {'val_bpb': 0.9979, 'num_steps': 953, 'mfu_percent': 39.8, 'num_params_M': 50.3, 'peak_vram_mb': 45060.2}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


@lru_cache(maxsize=None)
def upstream(name: str) -> str:
    """An upstream file as text, verified against its recorded sha256 (exact bytes, LF line ends)."""
    data = (LOCAL_SRC / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != UPSTREAM_SHA256[name]:
        raise ValueError(f'{LOCAL_SRC / name}: sha256 differs from autoresearch @{UPSTREAM["commit"][:7]}')
    return data.decode('utf-8')


# prepare.make_dataloader refills ONE pinned host buffer while the previous non_blocking H2D copy may still be queued
# behind the GPU's work: which batch a micro-step trains on then depends on CPU-vs-GPU timing (node and GPU dependent;
# X1 found it on WINH0, R2 on WINH3 happened to be race-free). 'h2d_sync' waits for that copy before returning the
# batch: deviation D-AR-7, adopted by O for every ENV-C arm (2026-10-21; REPRO_ISSUES #20); the judge always loads the
# original file. 'upstream' remains available for diagnosis.
PREPARE_VARIANTS = ('upstream', 'h2d_sync')
DEFAULT_PREPARE = 'h2d_sync'
H2D_ANCHOR = '        gpu_buffer.copy_(cpu_buffer, non_blocking=True)\n'
H2D_SYNC = '        torch.cuda.current_stream().synchronize()  # D-AR-7: the pinned buffer is reused by the next refill\n'


@lru_cache(maxsize=None)
def candidate_prepare(variant: str = DEFAULT_PREPARE) -> str:
    """prepare.py as the candidates import it (the trusted judge always loads the upstream file)."""
    text = upstream('prepare.py')
    if variant == 'upstream':
        return text
    if variant != 'h2d_sync':
        raise ValueError(f'prepare variant {variant!r} not in {PREPARE_VARIANTS}')
    if text.count(H2D_ANCHOR) != 1:
        raise ValueError('H2D copy anchor not found exactly once in prepare.py')
    return text.replace(H2D_ANCHOR, H2D_ANCHOR + H2D_SYNC)


@lru_cache(maxsize=None)
def base_train() -> str:
    """Upstream train.py with D-AR-2 applied (the start candidate c0 of every ENV-C arm)."""
    text = upstream('train.py')
    if text.count(D_AR_2_OLD) != 1:
        raise ValueError('D-AR-2 anchor not found exactly once in train.py')
    out = text.replace(D_AR_2_OLD, D_AR_2_NEW)
    if sha256_text(out) != BASE_TRAIN_SHA256:
        raise ValueError('base train.py differs from the R2 train.py')
    return out


def node_path(rel: str, cand_dir: str) -> str:
    """Node path of a frozen-file key: repo/<f> (candidate tree), data/<shard>, tokenizer/<f>, kernel/<f>,
    judge/<f> and tools/<f> (the trusted files written into the candidate directory)."""
    head, _, rest = rel.partition('/')
    roots = {'repo': f'{cand_dir}/repo', 'data': NODE_DATA, 'tokenizer': NODE_TOKENIZER, 'kernel': NODE_KERNEL,
             'judge': f'{cand_dir}/judge', 'tools': f'{cand_dir}/tools', 'judge2': f'{cand_dir}/judge_h2d'}
    if head not in roots or not rest:
        raise ValueError(f'unknown frozen-file key {rel!r}')
    return f'{roots[head]}/{rest}'
