"""Trusted ENV-C files written into every candidate directory (design §4.1 D-AR-5), built on this PC.

judge/model_ref.py   the model of the baseline train.py, extracted verbatim (GPTConfig, norm, has_ve,
                     apply_rotary_emb, CausalSelfAttention, MLP, Block, GPT, build_model_config and the architecture
                     constants), with FA3 loaded as in the candidate runs (D-AR-2)
judge/judge_bpb.py   rebuilds the model from final_model.pt and recomputes val_bpb with the ORIGINAL prepare.py
judge/prepare.py     the original prepare.py (upstream bytes)
judge/base_train.py  the baseline train.py (the trusted code expectation compiles its protected callables)
tools/run_train.py   the candidate entry: runs train.py unchanged, then saves final_model.pt + run_meta.json
tools/code_attest.py code-object attestation of the protected callables (run_train.py; the root expectation step)
The driver never imports these (they import torch); their sha256 enter the task's frozen files.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

from .source import KERNEL_REPO, UPSTREAM, base_train, candidate_prepare, sha256_text, upstream

NODE_DIR = Path(__file__).resolve().parent / 'node'
MODEL_NAMES = ('GPTConfig', 'norm', 'has_ve', 'apply_rotary_emb', 'CausalSelfAttention', 'MLP', 'Block', 'GPT')
ARCH_CONSTANTS = ('ASPECT_RATIO', 'HEAD_DIM', 'WINDOW_PATTERN', 'DEPTH')


def _segment(lines: list[str], node: ast.AST) -> str:
    start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
    return '\n'.join(lines[start - 1:node.end_lineno])


def model_ref_source(train_text: str | None = None) -> str:
    """judge/model_ref.py from the baseline train.py (verbatim definitions, in source order)."""
    text = train_text if train_text is not None else base_train()
    tree, lines = ast.parse(text), text.splitlines()
    defs, consts, found = [], [], set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in MODEL_NAMES + ('build_model_config',):
            defs.append((node.name, _segment(lines, node)))
            found.add(node.name)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ARCH_CONSTANTS:
            consts.append(_segment(lines, node))
            found.add(node.targets[0].id)
    missing = set(MODEL_NAMES + ARCH_CONSTANTS + ('build_model_config',)) - found
    if missing:
        raise ValueError(f'baseline train.py lacks {sorted(missing)}')
    head = (f'"""Trusted reference model for ENV-C judging: definitions extracted verbatim from the baseline train.py\n'
            f'(autoresearch @{UPSTREAM["commit"][:7]}, D-AR-2) by infrarsi.ext.autoresearch.judge_files. Do not edit;\n'
            f'its sha256 is one of the task\'s frozen files."""\n'
            'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom dataclasses import dataclass, asdict\n\n'
            'from kernels import get_kernel\n'
            f'fa3 = get_kernel("{KERNEL_REPO}").flash_attn_interface      # D-AR-2, as in the candidate runs\n\n'
            'from prepare import MAX_SEQ_LEN\n\n')
    body = '\n\n\n'.join(seg for name, seg in defs if name != 'build_model_config')
    build = next(seg for name, seg in defs if name == 'build_model_config')
    tail = ('\n\n\n# architecture constants of the baseline (frozen)\n' + '\n'.join(consts) + '\nvocab_size = None\n\n\n' + build +
            '\n\n\ndef expected_config(vocab):\n'
            '    """The baseline GPTConfig for the tokenizer\'s vocabulary (build_model_config(DEPTH))."""\n'
            '    global vocab_size\n'
            '    vocab_size = vocab\n'
            '    return build_model_config(DEPTH)\n')
    return head + body + tail


@lru_cache(maxsize=None)
def node_script(name: str) -> str:
    return (NODE_DIR / name).read_text(encoding='utf-8')


def trusted_files(prepare_variant: str = 'h2d_sync') -> dict[str, str]:
    """{frozen-file key: text} of the files the runner writes into each candidate directory; repo/prepare.py follows
    the task's prepare variant (source.PREPARE_VARIANTS), judge/prepare.py is always the original."""
    return {'judge/model_ref.py': model_ref_source(), 'judge/judge_bpb.py': node_script('judge_bpb.py'),
            'judge/prepare.py': upstream('prepare.py'), 'judge/base_train.py': base_train(),
            'tools/run_train.py': node_script('run_train.py'), 'tools/code_attest.py': node_script('code_attest.py'),
            'repo/prepare.py': candidate_prepare(prepare_variant)}


def trusted_sha256(prepare_variant: str = 'h2d_sync') -> dict[str, str]:
    return {k: sha256_text(v) for k, v in trusted_files(prepare_variant).items()}
