import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from datafakes import FakeTokenizer
from infrarsi.adapters.data import (IGNORE_INDEX, PackedTHDDataset, manifest_for_task, prepare,
                                      update_assignment, update_slots)

PPU, UPDATES = 8, 3          # 24 packs = packs_per_update x updates (the real tasks use x 48)
COMBOS = [(1, 1), (1, 2), (2, 1), (2, 2), (4, 1), (4, 2), (8, 1), (1, 8), (2, 4)]


@pytest.fixture(scope='module')
def manifest(raw, tmp_path_factory):
    raw_root, _ = raw
    return prepare('D-SFT-8K', None, tmp_path_factory.mktemp('packed'), 128, PPU * UPDATES, raw_root=raw_root,
                   tokenizer=FakeTokenizer())


def _task(manifest_path=None):
    t = {'id': 'T9', 'model': {'repo_id': 'fake/char'}, 'data': {'key': 'D-SFT-8K'},
         'workload': {'seq_len': 128, 'packs_per_update': PPU, 'packs_total': PPU * UPDATES}}
    if manifest_path:
        t['data']['manifest'] = str(manifest_path)
    return t


def test_sampler_layout():
    assert update_assignment(0, 8, 0, 2, 2) == [[0, 1], [4, 5]]
    assert update_assignment(0, 8, 1, 2, 2) == [[2, 3], [6, 7]] == update_slots(0, 8, 1, 2, 2)
    assert update_assignment(2, 4, 0, 1, 1) == [[8], [9], [10], [11]]
    order = list(range(23, -1, -1))
    assert update_assignment(1, 8, 1, 2, 1, order=order) == [[order[9]], [order[11]], [order[13]], [order[15]]]
    for bad in [(0, 8, 0, 3, 1), (0, 8, 2, 2, 1), (-1, 8, 0, 1, 1), (0, 8, 0, 0, 1), (0, 8, 0, 1, 0)]:
        with pytest.raises(ValueError):
            update_assignment(*bad)
    with pytest.raises(TypeError):
        update_assignment(0, 8.0, 0, 1, 1)
    with pytest.raises(IndexError):
        update_assignment(3, 8, 0, 1, 1, order=order)


def test_layout_matches_the_synthetic_twin():
    syn = pytest.importorskip('infrarsi.adapters.megatron_core.synthetic_data')
    for dp, mbs in COMBOS + [(16, 2), (32, 1)]:
        for u in range(3):
            for r in range(dp):
                assert update_assignment(u, 32, r, dp, mbs) == syn.update_assignment(u, 32, r, dp, mbs)


@pytest.mark.parametrize('plan', ['A', 'B'])
@pytest.mark.parametrize('dp,mbs', COMBOS)
def test_update_assignment_partitions_each_update(manifest, plan, dp, mbs):
    with PackedTHDDataset(manifest, plan) as ds:
        order, used = ds.order(), []
        for u in range(UPDATES):
            got = []
            for r in range(dp):
                mbs_r = ds.update_assignment(u, PPU, r, dp, mbs)
                assert mbs_r == update_assignment(u, PPU, r, dp, mbs, order=order)
                assert len(mbs_r) == PPU // (dp * mbs) and all(len(mb) == mbs for mb in mbs_r)
                got += [p for mb in mbs_r for p in mb]
            assert sorted(got) == sorted(order[u * PPU:(u + 1) * PPU])        # exact partition of the update
            used += got
        assert sorted(used) == list(range(len(ds)))                           # every pack exactly once
        with pytest.raises(IndexError):
            ds.update_assignment(UPDATES, PPU, 0, dp, mbs)


def test_through_the_trainer_datasource(manifest):
    """The real consumer (megatron_core.data.DataSource) on a prepared dataset, every layout, CP splits."""
    dsrc = pytest.importorskip('infrarsi.adapters.megatron_core.data')
    root = str(Path(manifest).parent.parent)
    if 'root' in dsrc.resolve_manifest.__code__.co_varnames:          # discovery without an explicit manifest
        assert Path(dsrc.resolve_manifest(_task(), root)) == Path(manifest)          # <key>__*__s<seq>__p<total>
        index = Path(root) / 'INDEX.json'
        entry = {'manifest': str(manifest), 'manifest_sha256': hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                 'model': 'fake/char', 'data_key': 'D-SFT-8K', 'seq_len': 128, 'packs_per_update': PPU,
                 'packs_total': PPU * UPDATES}                 # resolver validates the entry (manifest_for_task)
        index.write_text(json.dumps({'tasks': {'T9': entry}}), encoding='utf-8')
        assert Path(dsrc.resolve_manifest(_task(), root)) == Path(manifest)
        index.unlink()
    for plan in ('A', 'B'):
        src = dsrc.DataSource(_task(manifest), plan, vocab_size=200000)
        with PackedTHDDataset(manifest, plan) as ds:
            ref = [sorted(src.update_packs(u, 1, 1)) for u in range(UPDATES)]
            for dp, mbs in COMBOS:
                assert [sorted(src.update_packs(u, dp, mbs)) for u in range(UPDATES)] == ref
                assert src.microbatches(1, dp - 1, dp, mbs) == ds.update_assignment(1, PPU, dp - 1, dp, mbs)
            assert sorted(p for u in ref for p in u) == list(range(PPU * UPDATES))
        packs = [src.pack(p) for p in src.microbatches(0, 0, 2, 2)[0]]
        full = dsrc.local_arrays(packs)
        assert np.array_equal(full['loss_mask'][0], np.concatenate([p['labels'] for p in packs]) != IGNORE_INDEX)
        for cp in (2, 4):                                      # segments are multiples of 8 -> CP <= 4
            assert sum(dsrc.local_arrays(packs, cp, r)['tokens'].shape[1] for r in range(cp)) == 2 * 128


def test_manifest_for_task(manifest, tmp_path):
    entry = {'manifest': str(manifest), 'manifest_sha256': hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
             'model': 'fake/char', 'data_key': 'D-SFT-8K', 'seq_len': 128, 'packs_per_update': PPU,
             'packs_total': PPU * UPDATES}
    index = tmp_path / 'INDEX.json'
    index.write_text(json.dumps({'tasks': {'T9': entry}, 'pending': {'T8': {'reason': 'model download not complete'}}}))
    assert manifest_for_task(_task(), index) == str(manifest)
    with pytest.raises(LookupError, match='download'):
        manifest_for_task(dict(_task(), id='T8'), index)
    bad = _task()
    bad['workload']['packs_total'] = 48
    with pytest.raises(ValueError):
        manifest_for_task(bad, index)
    index.write_text(json.dumps({'tasks': {'T9': dict(entry, manifest_sha256='0' * 64)}}))
    with pytest.raises(ValueError):
        manifest_for_task(_task(), index)


def test_same_global_set_across_layouts(manifest):
    with PackedTHDDataset(manifest, 'A') as ds:
        for u in range(UPDATES):
            sets = {frozenset(p for r in range(dp) for mb in ds.update_assignment(u, PPU, r, dp, mbs) for p in mb)
                    for dp, mbs in COMBOS}
            assert len(sets) == 1


def test_pack_fields_and_token_counts(manifest):
    with PackedTHDDataset(manifest, 'B') as ds:
        assert len(ds) == PPU * UPDATES
        sup_total = 0
        for i in range(len(ds)):
            b = ds.pack(i)
            for k in ('tokens', 'labels', 'position_ids'):
                assert b[k].dtype == np.int32 and b[k].shape == (128,) and b[k].flags.writeable
            cu = b['cu_seqlens']
            assert cu.dtype == np.int32 and cu[0] == 0 and cu[-1] == 128
            assert b['physical_tokens'] == 128 and b['supervised_tokens'] == int((b['labels'] != IGNORE_INDEX).sum())
            assert b['max_seqlen'] == int(np.diff(cu).max()) and len(b['source_ids']) == len(cu) - 1
            assert not np.any(b['padding_mask'] & (b['labels'] != IGNORE_INDEX))
            assert int((~b['padding_mask']).sum()) == b['real_tokens']
            sup_total += b['supervised_tokens']
        per_update = [ds.update_token_counts(u, PPU) for u in range(UPDATES)]
        assert sum(c['supervised_tokens'] for c in per_update) == sup_total == ds.manifest['counts']['supervised_tokens']
        assert all(c['physical_tokens'] == PPU * 128 for c in per_update)
        with pytest.raises(IndexError):
            ds.pack(len(ds))
        clone = pickle.loads(pickle.dumps(ds))                                 # DataLoader-worker safe
        assert np.array_equal(clone.pack(3)['tokens'], ds.pack(3)['tokens'])


def test_bad_plan(manifest):
    with pytest.raises(ValueError):
        PackedTHDDataset(manifest, 'C')
