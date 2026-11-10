"""Raw sources and templates of the UpdateBench v0 data keys (design §1.2, task specs `data` blocks).

SFT readers expose `len()` and `turns(i) -> [(human, assistant), ...] | None` addressed by the sample's row id in
the raw file (None = malformed row); the PT reader yields '{title}\n\n{text}' documents in file order.
"""
from __future__ import annotations

import csv
import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataKey:
    key: str
    kind: str        # sft_packed | pt_stream
    repo: str        # ModelScope dataset; raw files live in <raw_root>/<owner>__<name>/
    file: str
    bytes: int       # raw file size recorded in the design
    template: str
    seq_len: int     # nominal sequence length of the key

    def raw_path(self, raw_root) -> Path:
        return Path(raw_root) / self.repo.replace('/', '__') / self.file


_SHAREGPT = ('swift/sharegpt', 'common_en_70k.jsonl', 426474586)
_LONGALPACA = ('AI-ModelScope/LongAlpaca-12k', 'LongAlpaca-12k_.csv', 489749483)
_WIKIPEDIA = ('swift/wikipedia', 'data/20220301.en/train-00001-of-00041.parquet', 704741684)

DATA_KEYS = {k.key: k for k in (
    DataKey('D-SFT-8K', 'sft_packed', *_SHAREGPT, 'plain_v1', 8192),
    DataKey('D-SFT-32K', 'sft_packed', *_LONGALPACA, 'plain_v1', 32768),
    DataKey('D-SFT-64K', 'sft_packed', *_LONGALPACA, 'plain_v1', 65536),
    DataKey('D-PT-4K', 'pt_stream', *_WIKIPEDIA, 'pt_v1', 4096),
    DataKey('D-PT-8K', 'pt_stream', *_WIKIPEDIA, 'pt_v1', 8192),
)}

# Recorded verbatim in every manifest.
TEMPLATES = {
    'plain_v1': {
        'turn': 'User: {human}\n\nAssistant: {assistant}{eos}\n\n',
        'bos': 'one bos id at the start of each sample, only if the tokenizer defines bos_token',
        'loss': "assistant text (including the template's separating space) + eos; everything else masked",
        'pieces': "per turn: 'User: {human}\\n\\nAssistant:' masked | ' {assistant}' supervised | eos id supervised | "
                  "'\\n\\n' masked; each text piece is tokenized alone (split at a pre-token boundary)",
        'longalpaca_human': "instruction, or instruction + '\\n\\n' + input when input is not blank",
    },
    'pt_v1': {
        'document': '{title}\n\n{text}{eos}',
        'stream': 'documents concatenated in file order; eos inserted as an id',
        'windows': 'non-overlapping windows of seq_len+1 tokens: tokens = w[:-1], labels = w[1:]; every label is '
                   'supervised, the first token of a window is never a target',
    },
}


class ShareGPT:
    """jsonl, one conversation per line: {"conversation": [{"human": .., "assistant": ..}, ..]}; id = line index."""

    def __init__(self, path):
        lines = Path(path).read_bytes().split(b'\n')
        if lines and not lines[-1]:
            lines.pop()
        self.lines = lines

    def __len__(self):
        return len(self.lines)

    def turns(self, i):
        try:
            turns = [(t['human'], t['assistant']) for t in json.loads(self.lines[i])['conversation']]
        except (ValueError, KeyError, TypeError):
            return None
        ok = turns and all(isinstance(h, str) and isinstance(a, str) for h, a in turns)
        return turns if ok else None


class LongAlpaca:
    """csv with columns instruction,output,file,input; id = data row index (header excluded)."""

    def __init__(self, path):
        csv.field_size_limit(2 ** 31 - 1)
        with open(path, newline='', encoding='utf-8') as f:
            rows = list(csv.reader(f))
        head = rows[0]
        self.cols = [head.index(c) for c in ('instruction', 'input', 'output')]
        self.width = len(head)
        self.rows = rows[1:]

    def __len__(self):
        return len(self.rows)

    def turns(self, i):
        row = self.rows[i]
        if len(row) != self.width:
            return None
        ins, inp, out = (row[c] for c in self.cols)
        return [(f'{ins}\n\n{inp}' if inp.strip() else ins, out)]


@functools.lru_cache(maxsize=2)
def _sft_source(path, size, mtime_ns):
    return {'.jsonl': ShareGPT, '.csv': LongAlpaca}[Path(path).suffix](path)


def sft_source(path):
    """Parsed SFT source, cached per (path, size, mtime) so several prepares in one process parse it once."""
    st = os.stat(path)
    return _sft_source(str(path), st.st_size, st.st_mtime_ns)


def wiki_num_rows(path) -> int:
    import pyarrow.parquet as pq
    with pq.ParquetFile(path) as pf:
        return pf.metadata.num_rows


def wiki_documents(path, batch_rows=512):
    """Batches of '{title}\n\n{text}' in file order."""
    import pyarrow.parquet as pq
    with pq.ParquetFile(path) as pf:
        for b in pf.iter_batches(batch_size=batch_rows, columns=['title', 'text']):
            titles, texts = b.column('title').to_pylist(), b.column('text').to_pylist()
            yield [f'{t or ""}\n\n{x or ""}' for t, x in zip(titles, texts)]
