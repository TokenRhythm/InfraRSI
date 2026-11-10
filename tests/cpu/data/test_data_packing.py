import json
from pathlib import Path

import numpy as np
import pytest

from datafakes import EOS, OFFSET, PAD, FakeTokenizer, expected_sample
from infrarsi.adapters.data import (IGNORE_INDEX, FirstFitPacker, PackedTHDDataset, permutation, prepare,
                                      verify_dir)

SFT_CASES = [('D-SFT-8K', 'sharegpt', 128, 24), ('D-SFT-32K', 'longalpaca', 128, 10),
             ('D-SFT-64K', 'longalpaca', 256, 6)]


def _arrays(manifest_path):
    m = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    root = Path(manifest_path).parent
    P, S = m['packs_total'], m['seq_len']
    a = {n: np.fromfile(root / n, dtype=f['dtype']) for n, f in m['files'].items()}
    for n in ('tokens.bin', 'labels.bin', 'position_ids.bin'):
        a[n] = a[n].reshape(P, S)
    a['segments.bin'] = a['segments.bin'].reshape(-1, 3)
    return m, a


@pytest.mark.parametrize('key,source,seq,packs', SFT_CASES)
def test_sft_pack_invariants(tmp_path, raw, tok, key, source, seq, packs):
    raw_root, expected = raw
    mp = prepare(key, None, tmp_path, seq_len=seq, packs_total=packs, raw_root=raw_root, tokenizer=tok, chunk=7)
    m, a = _arrays(mp)
    tokens, labels, pos = a['tokens.bin'], a['labels.bin'], a['position_ids.bin']
    cu_all, off, segs = a['cu_seqlens.bin'], a['cu_offsets.bin'], a['segments.bin']
    assert len(off) == packs + 1 and off[0] == 0 and off[-1] == len(cu_all)
    seen = set()
    for p in range(packs):
        cu = cu_all[off[p]:off[p + 1]]
        rows = segs[off[p] - p:off[p + 1] - p - 1]
        assert cu[0] == 0 and cu[-1] == seq and np.all(np.diff(cu) > 0) and np.all(cu % 8 == 0)
        assert len(rows) == len(cu) - 1 and np.all(rows[:, 0] == p)
        for k, (_, sid, valid) in enumerate(rows.tolist()):
            lo, hi = int(cu[k]), int(cu[k + 1])
            assert np.array_equal(pos[p, lo:hi], np.arange(hi - lo))           # positions restart per segment
            if k < len(rows) - 1:
                assert hi - lo == -(-valid // 8) * 8                         # padded to a multiple of 8 only
            assert sid not in seen                                          # a sample lives in exactly one pack
            seen.add(sid)
            ids, sup = expected_sample(tok, expected[source][sid])
            assert valid == len(ids) - 1 <= seq
            assert tokens[p, lo:lo + valid].tolist() == ids[:-1]
            want = [t if s else IGNORE_INDEX for t, s in zip(ids[1:], sup[1:])]
            assert labels[p, lo:lo + valid].tolist() == want                  # shifted, masked to reply + eos
            assert np.all(tokens[p, lo + valid:hi] == PAD) and np.all(labels[p, lo + valid:hi] == IGNORE_INDEX)
    c = m['counts']
    assert c['physical_tokens'] == packs * seq and c['supervised_tokens'] == int((labels != IGNORE_INDEX).sum())
    assert c['real_tokens'] == int(segs[:, 2].sum()) and c['samples_packed'] == len(segs) == len(seen)
    examined = permutation(len(expected[source]), 42)[:c['samples_examined']].tolist()
    invalid = [s for s in examined if expected[source][s] is None]
    dropped = [s for s in examined if s not in invalid and len(expected_sample(tok, expected[source][s])[0]) - 1 > seq]
    assert c['invalid_samples'] == len(invalid) and c['dropped_samples'] == len(dropped)
    assert seen <= set(examined) - set(invalid) - set(dropped)
    if key == 'D-SFT-8K':
        assert c['dropped_samples'] > 0 and c['invalid_samples'] > 0       # the fixtures exercise both paths


def test_reply_mask_is_reply_plus_eos(tmp_path, raw):
    """Read the supervised targets back as text: exactly ' {assistant}<eos>' per turn, in order."""
    raw_root, expected = raw
    tok = FakeTokenizer()
    with PackedTHDDataset(prepare('D-SFT-8K', None, tmp_path, 128, 24, raw_root=raw_root, tokenizer=tok)) as ds:
        for i in range(len(ds)):
            b = ds.pack(i)
            for k, sid in enumerate(b['source_ids']):
                lo, hi = b['cu_seqlens'][k], b['cu_seqlens'][k + 1]
                got = ''.join('<eos>' if t == EOS else chr(t - OFFSET) for t in b['labels'][lo:hi] if t != IGNORE_INDEX)
                assert got == ''.join(f' {a}<eos>' for _, a in expected['sharegpt'][sid])


def test_pt_windows(tmp_path, raw, tok):
    raw_root, expected = raw
    S, P = 64, 12
    mp = prepare('D-PT-4K', None, tmp_path, seq_len=S, packs_total=P, raw_root=raw_root, tokenizer=tok, chunk=5)
    m, a = _arrays(mp)
    tokens, labels = a['tokens.bin'], a['labels.bin']
    stream, starts = [], []
    for title, text in expected['wiki']:                                   # pt_v1, no bos
        starts.append(len(stream))
        stream += [OFFSET + ord(c) for c in f'{title}\n\n{text}'] + [tok.eos_id]
    for p in range(P):
        w = stream[p * (S + 1):(p + 1) * (S + 1)]                          # non-overlapping seq+1 windows
        assert tokens[p].tolist() == w[:-1] and labels[p].tolist() == w[1:]
        assert a['segments.bin'][p].tolist() == [p, int(np.searchsorted(starts, p * (S + 1), side='right') - 1), S]
    assert np.all(labels != IGNORE_INDEX) and np.array_equal(a['position_ids.bin'], np.tile(np.arange(S), (P, 1)))
    assert a['cu_seqlens.bin'].tolist() == [0, S] * P
    c = m['counts']
    assert c['supervised_tokens'] == c['physical_tokens'] == P * S and c['dropped_samples'] == 0
    assert c['documents_total'] == 70 and m['kind'] == 'pt_stream'


def test_first_fit_rules():
    pk = FirstFitPacker(64, max_open=8, quantum=8)
    for item, size in enumerate([48, 48, 16, 16]):
        pk.add(item, size)
    assert pk.closed == [[0, 2], [1, 3]] and pk.open == []                # first fit; full packs close at once
    pk = FirstFitPacker(64, max_open=8, quantum=8)
    for item in range(9):
        pk.add(item, 40)
    assert pk.closed == [[0]] and [m for _, m in pk.open] == [[k] for k in range(1, 9)]   # oldest evicted
    pk.add(9, 24)
    assert pk.closed == [[0], [1, 9]]                                      # oldest pack with room wins
    pk = FirstFitPacker(64, max_open=8, quantum=8)
    pk.add(0, 64)
    assert pk.closed == [[0]]
    with pytest.raises(ValueError):
        pk.add(1, 72)


def test_first_fit_bounded_and_capacity():
    rng = np.random.RandomState(3)
    pk = FirstFitPacker(256, max_open=8, quantum=8)
    sizes = {}
    for item in range(2000):
        sizes[item] = int(rng.randint(1, 33)) * 8
        pk.add(item, sizes[item])
        assert len(pk.open) <= 8
        assert all(free >= 8 for free, _ in pk.open)
    pk.flush()
    assert sorted(i for m in pk.closed for i in m) == list(range(2000))
    assert all(sum(sizes[i] for i in m) <= 256 for m in pk.closed)


def test_rerun_is_byte_identical_and_chunk_free(tmp_path, raw):
    raw_root, _ = raw
    for key, seq, packs in (('D-SFT-8K', 128, 24), ('D-PT-8K', 64, 12)):
        a = prepare(key, None, tmp_path / 'a', seq, packs, raw_root=raw_root, tokenizer=FakeTokenizer(), chunk=3)
        b = prepare(key, None, tmp_path / 'b', seq, packs, raw_root=raw_root, tokenizer=FakeTokenizer(), chunk=500)
        ma = json.loads(Path(a).read_text(encoding='utf-8'))
        for name in list(ma['files']) + ['manifest.json']:
            assert (Path(a).parent / name).read_bytes() == (Path(b).parent / name).read_bytes()
        assert prepare(key, None, tmp_path / 'a', seq, packs, raw_root=raw_root, tokenizer=FakeTokenizer()) == a


def test_plans_are_permutations_of_the_same_packs(tmp_path, raw):
    raw_root, _ = raw
    mp = prepare('D-SFT-32K', None, tmp_path, 128, 10, raw_root=raw_root, tokenizer=FakeTokenizer())
    with PackedTHDDataset(mp, 'A') as A, PackedTHDDataset(mp, 'B') as B:
        oa, ob = A.order(), B.order()
        assert sorted(oa) == sorted(ob) == list(range(10)) and oa != ob
        assert oa == np.random.RandomState(42).permutation(10).tolist()
        assert ob == np.random.RandomState(1042).permutation(10).tolist()
        for i in range(10):                                                # same pack ids -> same content
            pa, pb = A.pack(i), B.pack(i)
            assert all(np.array_equal(pa[k], pb[k]) for k in ('tokens', 'labels', 'position_ids', 'cu_seqlens'))
        assert A.order_digest != B.order_digest


def test_integrity_checks(tmp_path, raw):
    raw_root, _ = raw
    mp = prepare('D-SFT-8K', None, tmp_path, 128, 24, raw_root=raw_root, tokenizer=FakeTokenizer())
    m = verify_dir(Path(mp).parent)
    assert set(m['files']) == {'tokens.bin', 'labels.bin', 'position_ids.bin', 'cu_seqlens.bin', 'cu_offsets.bin',
                               'segments.bin'}
    assert m['schema'] == 'infrarsi-packed-v1' and Path(mp).parent.name.endswith('__s128__p24')
    for k in ('data_key', 'source_sha256', 'tokenizer_digest', 'template_id', 'seq_len', 'packs_total',
              'dropped_samples', 'physical_tokens', 'supervised_tokens', 'plans', 'files'):   # §4b names, top level
        assert k in m, k
    assert m['supervised_tokens'] == m['counts']['supervised_tokens'] and m['template_id'] == 'plain_v1'
    PackedTHDDataset(mp, verify=True).close()
    path = Path(mp).parent / 'labels.bin'
    data = bytearray(path.read_bytes())
    data[100] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError):
        PackedTHDDataset(mp, verify=True)
    with pytest.raises(RuntimeError):
        prepare('D-SFT-8K', None, tmp_path, 128, 24, raw_root=raw_root, tokenizer=FakeTokenizer())
    fixed = prepare('D-SFT-8K', None, tmp_path, 128, 24, raw_root=raw_root, tokenizer=FakeTokenizer(), force=True)
    PackedTHDDataset(fixed, verify=True).close()


def test_force_swap_failure_restores_previous_output(tmp_path, raw, monkeypatch):
    from infrarsi.adapters.data import packing
    raw_root, _ = raw
    mp = Path(prepare('D-SFT-8K', None, tmp_path, 128, 24, raw_root=raw_root, tokenizer=FakeTokenizer()))
    before = mp.read_bytes()
    real_rename = packing.os.rename

    def failing_rename(src, dst):
        if '.tmp-' in Path(src).name:
            raise OSError('simulated publish failure')
        return real_rename(src, dst)

    monkeypatch.setattr(packing.os, 'rename', failing_rename)
    # identical rebuild: the restored previous output already equals it, so the call succeeds
    prepare('D-SFT-8K', None, tmp_path, 128, 24, raw_root=raw_root, tokenizer=FakeTokenizer(), force=True, chunk=5)
    assert mp.read_bytes() == before
    # different previous output: the failed swap restores it and raises
    altered = before.replace(b'"builder"', b'"builder" ', 1)
    mp.write_bytes(altered)
    with pytest.raises(OSError):
        prepare('D-SFT-8K', None, tmp_path, 128, 24, raw_root=raw_root, tokenizer=FakeTokenizer(), force=True)
    # corrupt binary under an intact manifest: a failed repair must not report success
    mp.write_bytes(before)
    labels = mp.parent / 'labels.bin'
    data = bytearray(labels.read_bytes())
    data[64] ^= 1
    labels.write_bytes(bytes(data))
    with pytest.raises(OSError):
        prepare('D-SFT-8K', None, tmp_path, 128, 24, raw_root=raw_root, tokenizer=FakeTokenizer(), force=True)
    monkeypatch.undo()
    assert sorted(p.name for p in tmp_path.iterdir()) == [mp.parent.name]
    prepare('D-SFT-8K', None, tmp_path, 128, 24, raw_root=raw_root, tokenizer=FakeTokenizer(), force=True)
    assert mp.read_bytes() == before
    PackedTHDDataset(str(mp), verify=True).close()


def test_other_packer_version_is_not_reused(tmp_path, raw):
    raw_root, _ = raw
    mp = Path(prepare('D-SFT-8K', None, tmp_path, 128, 24, raw_root=raw_root, tokenizer=FakeTokenizer()))
    m = json.loads(mp.read_text(encoding='utf-8'))
    mp.write_text(mp.read_text(encoding='utf-8').replace(m['builder'], 'infrarsi-packer-v0'), encoding='utf-8')
    with pytest.raises(RuntimeError, match='packer version'):
        prepare('D-SFT-8K', None, tmp_path, 128, 24, raw_root=raw_root, tokenizer=FakeTokenizer())
    assert prepare('D-SFT-8K', None, tmp_path, 128, 24, raw_root=raw_root, tokenizer=FakeTokenizer(), force=True)
    assert json.loads(mp.read_text(encoding='utf-8'))['builder'] == m['builder']


def test_bad_requests(tmp_path, raw):
    raw_root, _ = raw
    for chunk in (0, -3):                                                  # would never advance
        with pytest.raises(ValueError):
            prepare('D-SFT-8K', None, tmp_path, 128, 4, raw_root=raw_root, tokenizer=FakeTokenizer(), chunk=chunk)
    with pytest.raises(ValueError):                                        # not enough data
        prepare('D-SFT-32K', None, tmp_path, 128, 500, raw_root=raw_root, tokenizer=FakeTokenizer())
    with pytest.raises(ValueError):                                        # seq_len not a multiple of 8
        prepare('D-SFT-8K', None, tmp_path, 100, 4, raw_root=raw_root, tokenizer=FakeTokenizer())
    with pytest.raises(ValueError):
        prepare('D-SFT-8K', None, tmp_path, 128, None, raw_root=raw_root, tokenizer=FakeTokenizer())
    with pytest.raises(KeyError):
        prepare('D-SFT-16K', None, tmp_path, 128, 4, raw_root=raw_root, tokenizer=FakeTokenizer())
