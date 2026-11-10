"""Tokenizer adapter over the `tokenizers` library (the fast tokenizer transformers wraps; no transformers import).

Anything with `encode_batch(texts) -> list[list[int]]`, `bos_id`, `eos_id`, `pad_id`, `digest` and `info` can stand
in for it (CPU tests use a fake).
"""
from __future__ import annotations

import json
from pathlib import Path

from infrarsi.contracts import digest, file_digest

TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json')


def tokenizer_files(d) -> dict:
    d = Path(d)
    return {n: file_digest(d / n) for n in TOKENIZER_FILES if (d / n).is_file()}


def _token_text(v):
    if isinstance(v, dict):          # serialized AddedToken
        v = v.get('content')
    return v if isinstance(v, str) and v else None


class HFTokenizer:
    """<dir>/tokenizer.json with bos/eos/pad from tokenizer_config.json (special_tokens_map.json fills keys the config
    lacks). Special-token strings inside raw text are encoded as plain text; bos/eos only ever enter as ids.

    `digest` identifies the effective tokenization: tokenizer.json bytes + the bos/eos/pad ids used, so models that
    ship the same tokenizer (Qwen3 dense/MoE) share one prepared dataset.
    """

    def __init__(self, tokenizer_dir):
        import tokenizers
        d = Path(tokenizer_dir)
        tok = tokenizers.Tokenizer.from_file(str(d / 'tokenizer.json'))
        tok.no_truncation()
        tok.no_padding()
        tok.encode_special_tokens = True
        self._tok = tok
        files = tokenizer_files(d)
        cfg = {}
        for name in ('special_tokens_map.json', 'tokenizer_config.json'):    # later file wins
            if name in files:
                cfg.update(json.loads((d / name).read_text(encoding='utf-8')))
        special = {}
        for role in ('bos', 'eos', 'pad'):
            text = _token_text(cfg.get(f'{role}_token'))
            tid = tok.token_to_id(text) if text else None
            if text and tid is None:
                raise ValueError(f'{d}: {role}_token {text!r} is not in the vocabulary')
            special[role] = [text, tid]
        if special['eos'][1] is None:
            raise ValueError(f'{d}: tokenizer defines no eos_token')
        if special['pad'][1] is None:
            special['pad'] = list(special['eos'])     # padding is never supervised; eos is a safe filler
        self.bos_id, self.eos_id, self.pad_id = (special[r][1] for r in ('bos', 'eos', 'pad'))
        self.digest = digest({'tokenizer.json': files['tokenizer.json'], 'bos_id': self.bos_id,
                              'eos_id': self.eos_id, 'pad_id': self.pad_id, 'special_tokens_in_text': 'plain'})
        self.info = {'dir': str(d), 'files_sha256': files, 'bos': special['bos'], 'eos': special['eos'],
                     'pad': special['pad'], 'special_tokens_in_text': 'plain',
                     'vocab_size': tok.get_vocab_size(with_added_tokens=True),
                     'library': f'tokenizers {tokenizers.__version__}'}

    def encode_batch(self, texts):
        encode = getattr(self._tok, 'encode_batch_fast', self._tok.encode_batch)
        return [e.ids for e in encode(texts, add_special_tokens=False)]
