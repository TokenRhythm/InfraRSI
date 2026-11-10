import copy

import numpy as np
import pytest

from infrarsi.adapters.megatron_core import synthetic_data as syn
from infrarsi.adapters.megatron_core.data import DataSource, concat_packs, local_arrays, zigzag_index


@pytest.mark.parametrize('kind', ['pt_stream', 'sft_packed'])
def test_synthetic_pack_contract(kind):
    ds = syn.SyntheticPackedTHDDataset(None, 'A', seq_len=4096, packs_total=64, vocab_size=1000, kind=kind)
    p = ds.pack(5)
    assert set(p) >= {'tokens', 'labels', 'position_ids', 'cu_seqlens', 'supervised_tokens', 'physical_tokens'}
    assert p['tokens'].shape == p['labels'].shape == p['position_ids'].shape == (4096,)
    cu = p['cu_seqlens']
    assert cu.dtype == np.int32 and cu[0] == 0 and cu[-1] == 4096 and np.all(np.diff(cu) % 8 == 0)
    for lo, hi in zip(cu[:-1], cu[1:]):
        assert np.array_equal(p['position_ids'][lo:hi], np.arange(hi - lo))
        lab, tok = p['labels'][lo:hi], p['tokens'][lo:hi]
        sup = lab != -100
        assert np.array_equal(lab[:-1][sup[:-1]], tok[1:][sup[:-1]])        # aligned next-token targets
        if kind == 'sft_packed':
            assert lab[-1] == -100
    assert p['supervised_tokens'] == int((p['labels'] != -100).sum()) and p['physical_tokens'] == 4096
    q = syn.SyntheticPackedTHDDataset(None, 'B', seq_len=4096, packs_total=64, vocab_size=1000, kind=kind).pack(5)
    assert all(np.array_equal(p[k], q[k]) for k in ('tokens', 'labels', 'cu_seqlens'))   # plans share packs
    if kind == 'pt_stream':
        assert p['supervised_tokens'] == 4096 and len(cu) == 2


def test_plans_are_fixed_permutations():
    a = syn.SyntheticPackedTHDDataset(None, 'A', seq_len=64, packs_total=96, vocab_size=10).order()
    b = syn.SyntheticPackedTHDDataset(None, 'B', seq_len=64, packs_total=96, vocab_size=10).order()
    assert sorted(a) == sorted(b) == list(range(96)) and not np.array_equal(a, b)
    assert np.array_equal(a, np.random.RandomState(42).permutation(96))


@pytest.mark.parametrize('dp,mbs', [(1, 1), (2, 1), (4, 2), (8, 1), (8, 2), (16, 1)])
def test_update_assignment_layout_invariant(dp, mbs):
    ppu = 32
    for u in range(3):
        mbs_lists = [syn.update_assignment(u, ppu, d, dp, mbs) for d in range(dp)]
        flat = [x for rank in mbs_lists for mb in rank for x in mb]
        assert sorted(flat) == list(range(u * ppu, (u + 1) * ppu))
        assert all(len(rank) == ppu // (dp * mbs) and all(len(mb) == mbs for mb in rank) for rank in mbs_lists)
        if dp > 1:   # interleaved like Megatron's pretraining sampler: rank 1 starts right after rank 0
            assert mbs_lists[1][0][0] == u * ppu + mbs


def test_datasource_same_global_packs_for_any_layout(tasks):
    task = copy.deepcopy(tasks['D2'].raw)
    task['data']['synthetic'] = True
    src = DataSource(task, 'A', vocab_size=1000)
    assert src.max_updates == 48 and src.source == 'synthetic'
    ref = [sorted(src.update_packs(u, 1, 1)) for u in range(4)]
    for dp, mbs in ((2, 1), (4, 2), (8, 4), (32, 1)):
        assert [sorted(src.update_packs(u, dp, mbs)) for u in range(4)] == ref
    assert sorted(x for u in ref for x in u) == sorted(int(i) for i in src.order[:4 * 32])
    with pytest.raises(ValueError):
        src.microbatches(48, 0, 8, 1)


def test_concat_and_context_parallel_split():
    ds = syn.SyntheticPackedTHDDataset(None, 'A', seq_len=512, packs_total=8, vocab_size=500, kind='sft_packed')
    packs = [ds.pack(0), ds.pack(1)]
    row = concat_packs(packs)
    assert row['cu_seqlens'][-1] == 1024 and len(row['cu_seqlens']) == len(packs[0]['cu_seqlens']) + len(packs[1]['cu_seqlens']) - 1
    full = local_arrays(packs)
    assert full['tokens'].shape == (1, 1024) and full['tokens'].dtype == np.int64
    assert np.array_equal(full['loss_mask'][0], row['labels'] != -100)
    assert (full['labels'][0][~full['loss_mask'][0]] == 0).all()
    for cp in (2, 4):
        shards = [local_arrays(packs, cp, r) for r in range(cp)]
        idx = [zigzag_index(row['cu_seqlens'], cp, r) for r in range(cp)]
        assert sorted(np.concatenate(idx).tolist()) == list(range(1024))
        rebuilt = np.empty(1024, dtype=np.int64)
        for r in range(cp):
            rebuilt[idx[r]] = shards[r]['tokens'][0]
            assert shards[r]['tokens'].shape == (1, 1024 // cp)
            assert np.array_equal(shards[r]['cu_seqlens'], row['cu_seqlens'])
        assert np.array_equal(rebuilt, row['tokens'])
    seg = np.array([0, 16], dtype=np.int32)
    assert zigzag_index(seg, 2, 0).tolist() == [0, 1, 2, 3, 12, 13, 14, 15]
    assert zigzag_index(seg, 2, 1).tolist() == [4, 5, 6, 7, 8, 9, 10, 11]
    with pytest.raises(ValueError):
        zigzag_index(np.array([0, 12], dtype=np.int32), 4, 0)


def test_resolve_manifest_index_is_authoritative(tmp_path, tasks):
    import json
    from infrarsi.adapters.megatron_core.data import DataUnavailable, resolve_manifest
    from infrarsi.contracts import file_digest
    task = copy.deepcopy(tasks['D2'].raw)
    w = task['workload']
    d = tmp_path / f"{task['data']['key']}__abcdef012345__s{w['seq_len']}__p{w['packs_total']}"
    d.mkdir()
    (d / 'manifest.json').write_text('{"schema": "test"}')
    assert resolve_manifest(task, str(tmp_path)) == str(d / 'manifest.json')            # no index: glob
    entry = {'manifest': str(d / 'manifest.json'), 'manifest_sha256': file_digest(d / 'manifest.json'),
             'model': task['model']['repo_id'], 'data_key': task['data']['key'], 'seq_len': int(w['seq_len']),
             'packs_per_update': int(w['packs_per_update']), 'packs_total': int(w['packs_total'])}

    def write_index(e):
        (tmp_path / 'INDEX.json').write_text(json.dumps({'tasks': {'D2': e}, 'pending': {}}))

    write_index(entry)
    assert resolve_manifest(task, str(tmp_path)) == entry['manifest']
    write_index(dict(entry, packs_total=999))          # mismatch: fail, no glob fallback
    with pytest.raises(DataUnavailable, match='does not match'):
        resolve_manifest(task, str(tmp_path))
    write_index(dict(entry, manifest_sha256='0' * 64))
    with pytest.raises(DataUnavailable, match='sha256'):
        resolve_manifest(task, str(tmp_path))
    other = copy.deepcopy(tasks['D1'].raw)
    with pytest.raises(DataUnavailable, match='D1'):
        resolve_manifest(other, str(tmp_path))


def test_synthetic_order_digest_identifies_plan():
    a = syn.SyntheticPackedTHDDataset(None, 'A', seq_len=64, packs_total=96, vocab_size=10)
    a2 = syn.SyntheticPackedTHDDataset(None, 'A', seq_len=64, packs_total=96, vocab_size=10)
    b = syn.SyntheticPackedTHDDataset(None, 'B', seq_len=64, packs_total=96, vocab_size=10)
    assert a.order_digest == a2.order_digest != b.order_digest


def test_explicit_data_manifest_must_match_index(tmp_path, tasks):
    import json
    from infrarsi.adapters.megatron_core.data import DataUnavailable, resolve_manifest
    from infrarsi.contracts import file_digest
    task = copy.deepcopy(tasks['D2'].raw)
    good, other = tmp_path / 'good.json', tmp_path / 'other.json'
    good.write_text('{"schema": "good"}')
    other.write_text('{"schema": "other"}')
    assert resolve_manifest(task, str(tmp_path), data_manifest=str(other)) == str(other)      # no index
    (tmp_path / 'INDEX.json').write_text(json.dumps({'tasks': {'D2': {'manifest': str(good),
                                                                        'manifest_sha256': file_digest(good)}}}))
    assert resolve_manifest(task, str(tmp_path), data_manifest=str(good)) == str(good)
    with pytest.raises(DataUnavailable, match='manifest_sha256'):
        resolve_manifest(task, str(tmp_path), data_manifest=str(other))
