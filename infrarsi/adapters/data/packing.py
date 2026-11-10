"""prepare(): tokenize a raw source with the task model's tokenizer and write THD packs + manifest (INTERFACES §4b).

SFT (plain_v1): samples are visited in a fixed seed-42 permutation of their row ids and placed by bounded-space
first fit (<= 8 open packs); one sample = one segment, padded to a multiple of 8, positions restart at 0 per segment,
samples longer than seq_len are dropped and counted. PT (pt_v1): the file-order token stream is cut into
non-overlapping seq_len+1 windows, one segment per pack. Inside a segment labels are already shifted
(labels[j] is the target of tokens[j], as in flash-next packed_sft_mmap_dataset); -100 = unsupervised.
Plans A/B are permutations (seeds 42/1042) of the same packs. Output bytes depend only on the inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

import numpy as np

from infrarsi.contracts import PLAN_SEEDS, RemotePaths, digest, file_digest
from .sources import DATA_KEYS, TEMPLATES, sft_source, wiki_documents, wiki_num_rows
from .tokenizer import HFTokenizer

SCHEMA_PACKED = 'infrarsi-packed-v1'
PACKER_VERSION = 'infrarsi-packer-v2'     # bump whenever output bytes (binaries or manifest) could change
SEG_MULTIPLE = 8
MAX_OPEN_PACKS = 8
IGNORE_INDEX = -100
SAMPLE_ORDER_SEED = 42

LAYOUT = {
    'tokens.bin': 'int32 [packs_total, seq_len]; pad_id at padding',
    'labels.bin': 'int32 [packs_total, seq_len]; labels[j] is the target of tokens[j] (shifted inside each segment); '
                  '-100 = unsupervised (prompt, padding, last position of a segment)',
    'position_ids.bin': 'int32 [packs_total, seq_len]; 0..width-1 per segment; trailing pack padding continues the '
                        'last segment',
    'cu_seqlens.bin': 'int32, pack i = [cu_offsets[i], cu_offsets[i+1]); starts at 0, ends at seq_len, multiples of 8',
    'cu_offsets.bin': 'int64 [packs_total + 1]',
    'segments.bin': 'int64 [n_segments, 3] = (pack, source_id, valid_len); pack i owns rows '
                    '[cu_offsets[i] - i, cu_offsets[i+1] - i - 1); positions [cu[k] + valid_len, cu[k+1]) are padding; '
                    'source_id = raw row id (sft) or index of the document holding the window start (pt)',
}


def permutation(n: int, seed: int) -> np.ndarray:
    """Legacy MT19937 stream: numpy keeps RandomState streams frozen across versions."""
    return np.random.RandomState(seed).permutation(n).astype(np.int64)


def _ceil(n: int, q: int) -> int:
    return -(-n // q) * q


def dumps(obj, ind=0) -> str:
    """Deterministic, readable JSON: sorted keys, one key per line, lists kept on one line."""
    if isinstance(obj, dict) and obj:
        pad = ' ' * (ind + 1)
        body = ',\n'.join(f'{pad}{json.dumps(str(k), ensure_ascii=False)}: {dumps(v, ind + 1)}'
                          for k, v in sorted(obj.items()))
        return '{\n' + body + '\n' + ' ' * ind + '}'
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'), allow_nan=False, sort_keys=True)


class FirstFitPacker:
    """Bounded-space first fit (Next-k-Fit): an item goes to the oldest open pack with room; if none has room and
    `max_open` packs are open, the oldest closes first. A pack also closes once it cannot take another `quantum`.
    `closed` holds member lists in closing order, which is the pack id order."""

    def __init__(self, capacity, max_open=MAX_OPEN_PACKS, quantum=SEG_MULTIPLE):
        self.capacity, self.max_open, self.quantum = capacity, max_open, quantum
        self.open = []          # [free, members]
        self.closed = []

    def add(self, item, size):
        if not 0 < size <= self.capacity:
            raise ValueError(f'item size {size} outside (0, {self.capacity}]')
        k = next((k for k, (free, _) in enumerate(self.open) if free >= size), None)
        if k is None:
            if len(self.open) == self.max_open:
                self.closed.append(self.open.pop(0)[1])
            self.open.append([self.capacity, []])
            k = len(self.open) - 1
        pack = self.open[k]
        pack[0] -= size
        pack[1].append(item)
        if pack[0] < self.quantum:
            self.closed.append(self.open.pop(k)[1])

    def flush(self):
        self.closed += [members for _, members in self.open]
        self.open = []


def encode_plain_v1(tok, samples, nl):
    """[(ids int32, supervised bool)] for samples given as [(human, assistant), ...] turn lists."""
    texts = [t for turns in samples for h, a in turns for t in (f'User: {h}\n\nAssistant:', f' {a}')]
    enc = iter(tok.encode_batch(texts) if texts else ())
    out = []
    for turns in samples:
        ids, sup = [], []
        if tok.bos_id is not None:
            ids.append(tok.bos_id)
            sup.append(False)
        for _ in turns:
            prompt, reply = next(enc), next(enc)
            ids += prompt
            ids += reply
            ids.append(tok.eos_id)
            ids += nl
            sup += [False] * len(prompt)
            sup += [True] * (len(reply) + 1)
            sup += [False] * len(nl)
        out.append((np.asarray(ids, dtype=np.int32), np.asarray(sup, dtype=np.bool_)))
    return out


def _build_sft(tok, path, seq_len, packs_total, chunk):
    src = sft_source(path)
    n = len(src)
    perm = permutation(n, SAMPLE_ORDER_SEED)
    nl = tok.encode_batch(['\n\n'])[0]
    packer = FirstFitPacker(seq_len)
    kept, examined, invalid, dropped, pos = {}, 0, 0, 0, 0
    while len(packer.closed) < packs_total and pos < n:
        sids = perm[pos:pos + chunk].tolist()
        pos += len(sids)
        turns = {s: src.turns(s) for s in sids}
        valid = [s for s in sids if turns[s] is not None]
        enc = dict(zip(valid, encode_plain_v1(tok, [turns[s] for s in valid], nl)))
        for s in sids:
            if len(packer.closed) >= packs_total:
                break
            examined += 1
            if s not in enc:
                invalid += 1
                continue
            width = len(enc[s][0]) - 1
            if width > seq_len:
                dropped += 1
                continue
            kept[s] = enc[s]
            packer.add(s, _ceil(width, SEG_MULTIPLE))
    if len(packer.closed) < packs_total:
        packer.flush()
    if len(packer.closed) < packs_total:
        raise ValueError(f'{path}: only {len(packer.closed)} packs of {seq_len} from {n} samples; need {packs_total}')
    packs = packer.closed[:packs_total]

    P, S = packs_total, seq_len
    tokens = np.full((P, S), tok.pad_id, dtype=np.int32)
    labels = np.full((P, S), IGNORE_INDEX, dtype=np.int32)
    positions = np.zeros((P, S), dtype=np.int32)
    cu_all, offsets, segs = [], [0], []
    for p, members in enumerate(packs):
        cur, cu = 0, [0]
        for s in members:
            ids, sup = kept[s]
            m = len(ids) - 1
            tokens[p, cur:cur + m] = ids[:-1]
            labels[p, cur:cur + m] = np.where(sup[1:], ids[1:], IGNORE_INDEX)
            width = _ceil(m, SEG_MULTIPLE)
            positions[p, cur:cur + width] = np.arange(width, dtype=np.int32)
            segs.append((p, s, m))
            cur += width
            cu.append(cur)
        if cur < S:     # trailing padding joins the last segment so cu_seqlens[-1] == seq_len
            start = cu[-2]
            positions[p, cur:] = np.arange(cur - start, S - start, dtype=np.int32)
            cu[-1] = S
        cu_all += cu
        offsets.append(len(cu_all))
    per_pack = [len(m) for m in packs]
    real = sum(m for _, _, m in segs)
    counts = {
        'samples_total': n, 'samples_examined': examined, 'invalid_samples': invalid, 'dropped_samples': dropped,
        'samples_packed': len(segs), 'samples_unplaced': examined - invalid - dropped - len(segs),
        'segments': len(segs), 'segments_per_pack': {'min': min(per_pack), 'max': max(per_pack),
                                                     'mean': round(len(segs) / P, 4)},
        'longest_sample_tokens': max(m for _, _, m in segs),
        'physical_tokens': P * S, 'real_tokens': real, 'padding_tokens': P * S - real,
        'supervised_tokens': int((labels != IGNORE_INDEX).sum()), 'utilization': round(real / (P * S), 6),
    }
    packing = {
        'algorithm': 'bounded-space first fit: oldest open pack with room; if none fits and 8 are open the oldest '
                     'closes; a pack closes once it cannot take 8 more tokens; pack id = closing order',
        'max_open_packs': MAX_OPEN_PACKS, 'segment_multiple': SEG_MULTIPLE,
        'sample_order': {'seed': SAMPLE_ORDER_SEED, 'generator': 'numpy.random.RandomState(seed).permutation(n)',
                         'sha256': hashlib.sha256(perm.astype('<i8').tobytes()).hexdigest()},
        'drop_rule': 'samples with more than seq_len input tokens (len(ids) - 1 > seq_len) are dropped and counted',
        'trailing_padding': 'part of the last segment (cu_seqlens[-1] == seq_len; its positions continue)',
    }
    return (tokens, labels, positions, cu_all, offsets, segs), counts, packing


def _build_pt(tok, path, seq_len, packs_total, batch_rows):
    P, S = packs_total, seq_len
    need = P * (S + 1)
    stream = np.empty(need, dtype=np.int32)
    doc_starts, filled = [], 0
    for texts in wiki_documents(path, batch_rows):
        for ids in tok.encode_batch(texts):
            if filled >= need:
                break
            doc_starts.append(filled)
            ids = list(ids) + [tok.eos_id]
            k = min(len(ids), need - filled)
            stream[filled:filled + k] = ids[:k]
            filled += k
        if filled >= need:
            break
    if filled < need:
        raise ValueError(f'{path}: stream has {filled} tokens; need {need}')
    windows = stream.reshape(P, S + 1)
    first_doc = np.searchsorted(np.asarray(doc_starts), np.arange(P) * (S + 1), side='right') - 1
    arrays = (np.ascontiguousarray(windows[:, :-1]), np.ascontiguousarray(windows[:, 1:]),
              np.tile(np.arange(S, dtype=np.int32), (P, 1)), [0, S] * P, list(range(0, 2 * P + 1, 2)),
              [(p, int(first_doc[p]), S) for p in range(P)])
    counts = {
        'documents_total': wiki_num_rows(path), 'documents_used': len(doc_starts), 'stream_tokens': need,
        'invalid_samples': 0, 'dropped_samples': 0, 'segments': P,
        'physical_tokens': P * S, 'real_tokens': P * S, 'padding_tokens': 0, 'supervised_tokens': P * S,
        'utilization': 1.0,
    }
    packing = {'algorithm': 'contiguous stream windows', 'window_tokens': S + 1, 'stride_tokens': S + 1,
               'segments_per_pack': 1}
    return arrays, counts, packing


def _source_sha(path, _cache={}):
    st = os.stat(path)
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _cache:
        _cache[key] = file_digest(path)
    return _cache[key]


def verify_dir(root) -> dict:
    """Manifest of a prepared dir after checking every binary's size and sha256."""
    root = Path(root)
    m = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    for name, f in m['files'].items():
        if (root / name).stat().st_size != f['bytes'] or file_digest(root / name) != f['sha256']:
            raise RuntimeError(f'{root / name}: size or sha256 differs from manifest')
    return m


def output_name(data_key, tokenizer_digest, seq_len, packs_total) -> str:
    return f'{data_key}__{tokenizer_digest[:12]}__s{int(seq_len)}__p{int(packs_total)}'


def prepare(data_key, tokenizer_dir, out_root=RemotePaths.datasets_packed, seq_len=None, packs_total=None, *,
            raw_root=RemotePaths.datasets_raw, tokenizer=None, force=False, chunk=1024) -> str:
    """Write <out_root>/<data_key>__<tok12>__s<seq_len>__p<packs_total>/ and return its manifest path.

    An existing output is verified (sha256) and reused unless force. `tokenizer` overrides loading tokenizer_dir.
    `chunk` (samples / documents per tokenizer batch) does not affect the output.
    """
    dk = DATA_KEYS[data_key]
    seq_len = int(dk.seq_len if seq_len is None else seq_len)
    if packs_total is None:
        raise ValueError('packs_total is required (packs_per_update x 48 from the task spec)')
    packs_total, chunk = int(packs_total), int(chunk)
    if seq_len <= 0 or seq_len % SEG_MULTIPLE or packs_total < 1 or chunk < 1:
        raise ValueError(f'seq_len {seq_len} must be a positive multiple of {SEG_MULTIPLE}; '
                         f'packs_total {packs_total} and chunk {chunk} must be >= 1')
    tok = tokenizer if tokenizer is not None else HFTokenizer(tokenizer_dir)
    final = Path(out_root) / output_name(data_key, tok.digest, seq_len, packs_total)
    if final.exists() and not force:
        if verify_dir(final).get('builder') != PACKER_VERSION:
            raise RuntimeError(f'{final} was written by another packer version; rebuild with force=True')
        return str(final / 'manifest.json')

    src = dk.raw_path(raw_root)
    build = _build_sft if dk.kind == 'sft_packed' else _build_pt
    (tokens, labels, positions, cu, offsets, segs), counts, packing = build(tok, src, seq_len, packs_total, chunk)
    packing['pad_id'] = tok.pad_id
    arrays = {'tokens.bin': tokens, 'labels.bin': labels, 'position_ids.bin': positions,
              'cu_seqlens.bin': np.asarray(cu, dtype='<i4'), 'cu_offsets.bin': np.asarray(offsets, dtype='<i8'),
              'segments.bin': np.asarray(segs, dtype='<i8').reshape(-1, 3)}
    tmp = final.parent / f'.{final.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}'
    tmp.mkdir(parents=True)
    try:
        _write(tmp, final, arrays, dk, src, tok, seq_len, packs_total, packing, counts, force)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)       # gone after a successful rename; debris after a failure
    return str(final / 'manifest.json')


def _write(tmp, final, arrays, dk, src, tok, seq_len, packs_total, packing, counts, force):
    files = {}
    for name, arr in arrays.items():
        data = np.ascontiguousarray(arr).astype(arr.dtype.newbyteorder('<'), copy=False).tobytes()
        (tmp / name).write_bytes(data)
        files[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                       'dtype': np.dtype(arr.dtype).newbyteorder('<').str, 'shape': list(arr.shape)}
    plans = {}
    for plan, seed in sorted(PLAN_SEEDS.items()):
        order = permutation(packs_total, seed).tolist()
        plans[plan] = {'seed': seed, 'order': order, 'sha256': digest(order)}
    manifest = {
        'schema': SCHEMA_PACKED, 'builder': PACKER_VERSION, 'data_key': dk.key, 'kind': dk.kind,
        'template': {'id': dk.template, **TEMPLATES[dk.template]},
        'source': {'repo': dk.repo, 'file': dk.file, 'path': str(src), 'bytes': os.stat(src).st_size,
                   'sha256': _source_sha(src), 'design_bytes': dk.bytes},
        'tokenizer': {'digest': tok.digest, **tok.info},
        'seq_len': seq_len, 'packs_total': packs_total, 'packing': packing, 'counts': counts,
        # flat copies under the names INTERFACES §4b uses
        'source_sha256': _source_sha(src), 'tokenizer_digest': tok.digest, 'template_id': dk.template,
        'dropped_samples': counts['dropped_samples'], 'physical_tokens': counts['physical_tokens'],
        'supervised_tokens': counts['supervised_tokens'],
        'layout': LAYOUT, 'labels_shifted': True, 'ignore_index': IGNORE_INDEX, 'files': files,
        'plans': plans, 'plan_generator': 'numpy.random.RandomState(seed).permutation(packs_total)',
    }
    body = (dumps(manifest) + '\n').encode('utf-8')
    (tmp / 'manifest.json').write_bytes(body)
    _publish(tmp, final, body, force)


def _publish(tmp: Path, final: Path, manifest_bytes: bytes, force: bool):
    """Atomic directory rename. A concurrent identical build is accepted, a different one is an error. With force
    the previous output is moved aside, restored if the swap fails, and deleted only once the new one is in place."""
    old = None
    if force and final.exists():
        old = final.parent / f'.{final.name}.old-{os.getpid()}-{uuid.uuid4().hex[:8]}'
        os.rename(final, old)
    try:
        os.rename(tmp, final)
    except OSError:
        if old is not None and not final.exists():
            os.rename(old, final)
            old = None
        if not _same_output(final, manifest_bytes):
            raise
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)


def _same_output(root: Path, manifest_bytes: bytes) -> bool:
    """True if `root` holds exactly the build described by manifest_bytes (manifest and every binary's sha256)."""
    try:                     # any read failure (e.g. a concurrent move) means "not the same"; the caller re-raises
        if (root / 'manifest.json').read_bytes() != manifest_bytes:
            return False
        verify_dir(root)
    except (OSError, RuntimeError):
        return False
    return True
