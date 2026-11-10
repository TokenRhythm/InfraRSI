import json
import math
import random
import statistics

import pytest

from infrarsi.contracts import check_receipt_shape
from infrarsi.runtime import invariants as inv, receipt_writer as rw
from infrarsi.runtime.signals import SignalWriter, read_signals, signals_digest

from train_helpers import QWEN3_8B, fr_config


def write_signals(out, world, steps, seed=0, sup=200_000, phys=262_144, skip=None):
    rng = random.Random(seed)
    expected_sup = {}
    for r in range(world):
        w = SignalWriter(out, r)
        for s in range(1, steps + 1):
            if skip and (r, s) in skip:
                continue
            whole = 10.0 + 0.1 * rng.random() + (0.5 if (s % 7 == 0 and r == 3) else 0.0)
            row = {'step': s, 'whole_update_s': whole, 'utc_start_s': 1.7e9 + 10.2 * s,
                   'utc_end_s': 1.7e9 + 10.2 * s + whole + 1e-4, 'cuda_whole_s': whole - 1e-4,
                   'fwd_bwd_s': 8.0, 'optim_s': 1.0, 'data_s': 0.01, 'inter_step_s': None if s == 1 else 10.2,
                   'peak_alloc_gib': 40.0 + r, 'peak_reserved_gib': 50.0 + r, 'alloc_retries': 0, 'alloc_ooms': 0,
                   'microbatches': 4}
            if r == 0:
                row.update(main_ce=2.0 - 0.01 * s, aux_loss=None, grad_norm=1.5, lr=inv.lr_at(s),
                           physical_tokens=phys, supervised_tokens=sup + s, optimizer_steps=s, clip_calls=1)
                expected_sup[s] = sup + s
            w.write(row)
        w.close()
    return expected_sup


def parts_for(steps, expected_sup, phys=262_144):
    return {'final': True, 'overlay_digest': 'x', 'invariants': {'gbs_packs': 32}, 'phases': {'train': 1.0},
            'env': {'torch': 'test'}, 'events': {'exceptions': [], 'nccl_errors': 0},
            'expected': {'lr': {str(s): inv.lr_at(s) for s in range(1, steps + 1)},
                         'supervised_tokens': {str(k): v for k, v in expected_sup.items()}, 'physical_tokens': phys}}


@pytest.fixture
def req(make_request, tasks):
    from infrarsi.adapters.megatron_core.config_map import model_facts
    return make_request('D2', fr_config(tasks['D2'].raw, model_facts(QWEN3_8B)), gpus=4)


def test_per_step_max_rank_and_window(tmp_path, req):
    sup = write_signals(tmp_path, 4, 30)
    signals = read_signals(tmp_path / 'signals')
    table = rw.per_step_table(signals)
    assert len(table) == 30 and all(r['ranks_reporting'] == 4 for r in table)
    for row in table:
        times = {r: rows[row['step'] - 1]['whole_update_s'] for r, rows in signals.items()}
        assert row['max_rank_whole_s'] == max(times.values())
        assert row['slowest_rank'] == min(r for r, t in times.items() if t == max(times.values()))
    stats = rw.window_stats(table, (17, 30), 4)
    window = [r['max_rank_whole_s'] for r in table if 17 <= r['step'] <= 30]
    assert stats['mean_window_max_rank_s'] == pytest.approx(sum(window) / 14, rel=1e-15)
    assert stats['median_window_max_rank_s'] == statistics.median(window)
    assert stats['rsd_window'] == pytest.approx(statistics.stdev(window) / statistics.mean(window))
    assert stats['inter_step_mean_s'] == pytest.approx(10.2) and stats['inter_step_steps'] == [18, 30]
    receipt = rw.assemble(req, tmp_path / 'signals', parts_for(30, sup))
    assert receipt['valid'], receipt['invalid_reasons']
    assert check_receipt_shape(receipt) == []
    assert receipt['steps_completed'] == 30 and receipt['memory']['max_peak_reserved_gib'] == 53.0
    assert receipt['signals_digest'] == signals_digest(tmp_path / 'signals')
    json.dumps(rw.sanitize(receipt), allow_nan=False)


def test_signals_digest_is_rank_ordered_concatenation(tmp_path):
    write_signals(tmp_path, 3, 2)
    import hashlib
    h = hashlib.sha256()
    for r in range(3):
        h.update((tmp_path / 'signals' / f'rank{r}.jsonl').read_bytes())
    assert signals_digest(tmp_path / 'signals') == h.hexdigest()


def test_invalid_receipts(tmp_path, req):
    sup = write_signals(tmp_path, 4, 30, skip={(2, 30)})
    r = rw.assemble(req, tmp_path / 'signals', parts_for(30, sup))
    assert not r['valid'] and r['steps_completed'] == 29
    assert any('incomplete' in x for x in r['invalid_reasons'])
    assert any('window' in x for x in r['invalid_reasons'])


@pytest.mark.parametrize('mutate,needle', [
    (lambda row: row.update(main_ce=float('nan')), 'non-finite main_ce'),
    (lambda row: row.update(lr=row['lr'] * 1.01), 'lr'),
    (lambda row: row.update(supervised_tokens=row['supervised_tokens'] - 1), 'supervised tokens'),
    (lambda row: row.update(physical_tokens=row['physical_tokens'] - 8192), 'physical tokens'),
    (lambda row: row.update(alloc_ooms=1), 'OOM'),
])
def test_row_level_violations(tmp_path, req, mutate, needle):
    sup = write_signals(tmp_path, 4, 30)
    path = tmp_path / 'signals' / 'rank0.jsonl'
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    mutate(rows[-1])
    path.write_text(''.join(json.dumps(x) + '\n' for x in rows))
    r = rw.assemble(req, tmp_path / 'signals', parts_for(30, sup))
    assert not r['valid'] and any(needle in x for x in r['invalid_reasons']), r['invalid_reasons']
    json.dumps(rw.sanitize(r), allow_nan=False)


def test_fallback_and_parts_problems(tmp_path, req):
    sup = write_signals(tmp_path, 4, 12)
    rw.write_json_atomic(tmp_path / rw.PARTS_FILE, parts_for(30, sup))
    r = rw.fallback(req, tmp_path, 'trainer exited with code 1')
    assert not r['valid'] and r['steps_completed'] == 12
    assert 'trainer exited with code 1' in r['invalid_reasons']
    assert check_receipt_shape(r) == []
    parts = parts_for(30, write_signals(tmp_path / 'b', 4, 30))
    parts['problems'] = ['dtype census: exp_avg expected torch.float32, found {"torch.bfloat16": 4}']
    r2 = rw.assemble(req, tmp_path / 'b' / 'signals', parts)
    assert not r2['valid'] and 'dtype census' in r2['invalid_reasons'][0]


def test_probe_has_no_window(tmp_path, make_request):
    req = make_request('D2', {}, purpose='probe')
    sup = write_signals(tmp_path, 8, 5)
    r = rw.assemble(req, tmp_path / 'signals', parts_for(5, sup))
    assert r['window'] is None and r['step_time'] is None and r['valid'], r['invalid_reasons']


def test_lifecycle_target_steps(tmp_path, make_request):
    req = make_request('D2', {}, purpose='lifecycle', lifecycle={'save_at': 30, 'resume_steps': 10}, input_plan='B')
    sup = write_signals(tmp_path, 8, 40)
    parts = parts_for(40, sup)
    parts['target_steps'] = 40
    parts['lifecycle'] = {'saved_at': 30}
    r = rw.assemble(req, tmp_path / 'signals', parts)
    assert r['steps'] == 30 and r['steps_completed'] == 40 and r['valid'] and r['lifecycle']['saved_at'] == 30


def test_oom_before_first_row_counts_as_allocator_oom(tmp_path, req):
    (tmp_path / 'signals').mkdir()
    parts = {'events': {'exceptions': [{'rank': 3, 'phase': 'train', 'kind': 'oom', 'error': 'OOM', 'step': None}]}}
    r = rw.assemble(req, tmp_path / 'signals', parts)
    assert r['events']['alloc_ooms'] == 1 and not r['valid'] and r['steps_completed'] == 0


def test_hole_in_the_middle_is_invalid(tmp_path, req):
    sup = write_signals(tmp_path, 4, 30, skip={(r, 5) for r in range(4)})
    r = rw.assemble(req, tmp_path / 'signals', parts_for(30, sup))
    assert not r['valid'] and r['steps_completed'] == 4
    assert any('incomplete: 4/30' in x for x in r['invalid_reasons'])


def test_duplicate_rows_are_invalid(tmp_path, req):
    sup = write_signals(tmp_path, 4, 30)
    path = tmp_path / 'signals' / 'rank2.jsonl'
    lines = path.read_text().splitlines()
    path.write_text('\n'.join(lines + [lines[3]]) + '\n')
    r = rw.assemble(req, tmp_path / 'signals', parts_for(30, sup))
    assert not r['valid'] and any('duplicate signal rows on ranks [2]' in x for x in r['invalid_reasons'])


def test_scaled_clock_is_detected(tmp_path, req):
    """An overlay that scales perf_counter shrinks whole_update_s but not the import-bound UTC interval."""
    sup = write_signals(tmp_path, 4, 30)
    path = tmp_path / 'signals' / 'rank1.jsonl'
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    for row in rows:
        row['whole_update_s'] *= 0.8
    path.write_text(''.join(json.dumps(x) + '\n' for x in rows))
    r = rw.assemble(req, tmp_path / 'signals', parts_for(30, sup))
    assert not r['valid'] and any(x.startswith('clock mismatch: rank 1') for x in r['invalid_reasons'])


def test_signal_rows_require_utc_interval(tmp_path):
    w = SignalWriter(tmp_path, 0)
    with pytest.raises(KeyError, match='utc_start_s'):
        w.write({'step': 1, 'whole_update_s': 1.0, 'fwd_bwd_s': 1.0, 'optim_s': 0.1, 'data_s': 0.0,
                 'peak_alloc_gib': 1.0, 'peak_reserved_gib': 1.0, 'alloc_retries': 0, 'alloc_ooms': 0})
    w.close()


def test_remove_artifacts_handles_files_and_directories(tmp_path):
    from infrarsi.runtime.lifecycle import remove_artifacts
    (tmp_path / 'ckpt' / '6').mkdir(parents=True)
    (tmp_path / 'ckpt' / '6' / 'x.distcp').write_bytes(b'0' * 8)
    (tmp_path / 'lifecycle').mkdir()
    (tmp_path / 'lifecycle' / 'mcore_logits.pt').write_bytes(b'0' * 8)
    (tmp_path / 'lifecycle' / 'forward_compare.json').write_text('{}')
    removed = remove_artifacts([tmp_path / 'ckpt', tmp_path / 'hf_export', tmp_path / 'lifecycle' / 'mcore_logits.pt'])
    assert removed == [str(tmp_path / 'ckpt'), str(tmp_path / 'lifecycle' / 'mcore_logits.pt')]
    assert not (tmp_path / 'ckpt').exists() and (tmp_path / 'lifecycle' / 'forward_compare.json').exists()


def test_repeated_adjacent_scalars_and_microbatches_are_invalid(tmp_path, req):
    sup = write_signals(tmp_path, 4, 30)
    parts = parts_for(30, sup)
    parts['expected']['microbatches'] = 4
    r = rw.assemble(req, tmp_path / 'signals', parts)
    assert r['valid'] and r['invariants']['self_check'] == {'repeated_pairs': []}
    rows = read_signals(tmp_path / 'signals')
    rows[0][10]['main_ce'], rows[0][10]['grad_norm'] = rows[0][9]['main_ce'], rows[0][9]['grad_norm']
    rows[3][4]['microbatches'] = 3                  # one micro-batch of rank 3 skipped in update 5
    for rank, rr in rows.items():
        (tmp_path / 'signals' / f'rank{rank}.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rr))
    r = rw.assemble(req, tmp_path / 'signals', parts)
    assert r['invariants']['self_check'] == {'repeated_pairs': [[10, 11]]} and not r['valid']
    assert any('repeated adjacent scalars' in x for x in r['invalid_reasons'])
    assert any(x.startswith('microbatches != 4 at (rank, update) [(3, 5)]') for x in r['invalid_reasons'])
