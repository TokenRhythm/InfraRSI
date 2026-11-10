"""Rank-0 receipt assembly (INTERFACES section 4). Pure Python so the supervisor and CPU tests share it.

The per-step table is the max over ranks of the native whole-update time; the window statistics are
recomputed from it. The driver re-derives the same numbers from ``signals/*.jsonl`` and never trusts
this summary, so everything here must be a deterministic function of the signal rows.
"""
from __future__ import annotations

import json
import math
import os
import statistics
from json import dumps as _json_dumps
from pathlib import Path

from infrarsi.contracts import SCHEMA_RECEIPT, check_receipt_shape, digest, utc_now
from infrarsi.runtime import invariants as inv
from infrarsi.runtime.signals import read_signals, signals_digest

PER_STEP_SCALARS = ('main_ce', 'aux_loss', 'grad_norm', 'lr', 'physical_tokens', 'supervised_tokens')
PARTS_FILE = 'receipt_parts.json'    # rank 0 keeps it current so a crash still yields a rich invalid receipt


def per_step_table(signals: dict[int, list[dict]]) -> list[dict]:
    """One row per step: slowest rank's whole-update time and rank 0's global scalars."""
    by_step: dict[int, dict[int, dict]] = {}
    for rank, rows in signals.items():
        for row in rows:
            by_step.setdefault(int(row['step']), {})[int(rank)] = row
    table = []
    for step in sorted(by_step):
        ranks = by_step[step]
        order = sorted(ranks)
        slowest = max(order, key=lambda r: float(ranks[r]['whole_update_s']))   # ties -> lowest rank
        inter = [float(ranks[r]['inter_step_s']) for r in order if ranks[r].get('inter_step_s') is not None]
        g = ranks.get(0, {})
        row = {'step': step, 'max_rank_whole_s': float(ranks[slowest]['whole_update_s']), 'slowest_rank': slowest,
               'ranks_reporting': len(order), 'max_rank_inter_step_s': max(inter) if inter else None}
        row.update({k: g.get(k) for k in PER_STEP_SCALARS})
        table.append(row)
    return table


def window_stats(per_step: list[dict], window, world_size: int | None = None) -> dict | None:
    """StepTime = mean over the window of max-rank whole-update time; RSD uses the sample std (n-1)."""
    if not window:
        return None
    a, b = int(window[0]), int(window[1])
    rows = [r for r in per_step if a <= r['step'] <= b and (world_size is None or r['ranks_reporting'] == world_size)]
    times = [r['max_rank_whole_s'] for r in rows]
    if len(times) != b - a + 1:
        return None
    mean = math.fsum(times) / len(times)
    rsd = statistics.stdev(times) / mean if len(times) > 1 and mean > 0 else 0.0
    inter = [r['max_rank_inter_step_s'] for r in per_step
             if a + 1 <= r['step'] <= b and r['max_rank_inter_step_s'] is not None]
    return {'mean_window_max_rank_s': mean, 'median_window_max_rank_s': statistics.median(times),
            'rsd_window': rsd, 'inter_step_mean_s': math.fsum(inter) / len(inter) if inter else None,
            'n_window': len(times), 'rsd_ddof': 1, 'inter_step_steps': [a + 1, b]}


def memory_summary(signals: dict[int, list[dict]]) -> dict:
    per_rank = {}
    for rank, rows in signals.items():
        if rows:
            per_rank[str(rank)] = {'peak_alloc_gib': max(float(r['peak_alloc_gib']) for r in rows),
                                   'peak_reserved_gib': max(float(r['peak_reserved_gib']) for r in rows)}
    vals = list(per_rank.values())
    return {'per_rank': per_rank,
            'max_peak_alloc_gib': max((v['peak_alloc_gib'] for v in vals), default=None),
            'max_peak_reserved_gib': max((v['peak_reserved_gib'] for v in vals), default=None)}


def allocator_events(signals: dict[int, list[dict]]) -> dict:
    """Allocator counters are cumulative per process; the last row of each rank holds the total."""
    retries = sum(int(rows[-1]['alloc_retries']) for rows in signals.values() if rows)
    ooms = sum(int(rows[-1]['alloc_ooms']) for rows in signals.values() if rows)
    return {'alloc_retries': retries, 'alloc_ooms': ooms}


UTC_TOLERANCE_S = 0.05          # absolute slack for the wall-clock interval vs perf_counter, plus 1 %


def clock_problems(signals: dict[int, list[dict]]) -> list[str]:
    """utc_end_s - utc_start_s must match whole_update_s on every rank and step (clock tampering, frozen timers)."""
    bad = []
    for rank, rows in signals.items():
        for r in rows:
            if 'utc_start_s' not in r or 'utc_end_s' not in r:
                bad.append(f'rank {rank} step {r.get("step")}: no UTC interval')
                continue
            span, whole = float(r['utc_end_s']) - float(r['utc_start_s']), float(r['whole_update_s'])
            if abs(span - whole) > UTC_TOLERANCE_S + 0.01 * whole:
                bad.append(f'rank {rank} step {r["step"]}: UTC span {span:.3f}s vs whole_update_s {whole:.3f}s')
    return [f'clock mismatch: {x}' for x in bad[:8]]


def row_problems(per_step: list[dict], expected: dict) -> list[str]:
    """Frozen-zone checks on the per-step scalars (LR schedule, token counts, finiteness)."""
    problems = []
    lr_expected = {int(k): float(v) for k, v in (expected.get('lr') or {}).items()}
    tokens = {int(k): int(v) for k, v in (expected.get('supervised_tokens') or {}).items()}
    physical = expected.get('physical_tokens')
    for r in per_step:
        s = r['step']
        for k in ('main_ce', 'grad_norm'):
            v = r.get(k)
            if v is None or not math.isfinite(float(v)):
                problems.append(f'step {s}: non-finite {k}')
        if s in lr_expected and (r.get('lr') is None or abs(float(r['lr']) - lr_expected[s]) > 1e-12 * lr_expected[s]):
            problems.append(f'step {s}: lr {r.get("lr")} != schedule {lr_expected[s]}')
        if s in tokens and r.get('supervised_tokens') != tokens[s]:
            problems.append(f'step {s}: supervised tokens {r.get("supervised_tokens")} != data {tokens[s]}')
        if physical is not None and r.get('physical_tokens') != physical:
            problems.append(f'step {s}: physical tokens {r.get("physical_tokens")} != {physical}')
    return problems


def repeated_pairs(per_step: list[dict]) -> list[list[int]]:
    """AMENDMENT-001 self-check: adjacent updates whose (main_ce, grad_norm) are bitwise identical."""
    rows = sorted(per_step, key=lambda r: r['step'])
    return [[a['step'], b['step']] for a, b in zip(rows, rows[1:])
            if b['step'] == a['step'] + 1 and a.get('main_ce') is not None and a.get('grad_norm') is not None
            and a.get('main_ce') == b.get('main_ce') and a.get('grad_norm') == b.get('grad_norm')]


def sanitize(obj):
    """Non-finite floats become None so the receipt is strict JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


def write_json_atomic(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    tmp.write_text(_json_dumps(sanitize(obj), indent=1, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def identity(request) -> dict:
    """Receipt identity block from a RunRequest (contracts) or its to_dict() form."""
    d = request.to_dict() if hasattr(request, 'to_dict') else dict(request)
    return {'schema': SCHEMA_RECEIPT, 'run_id': d['run_id'], 'attempt_id': d['attempt_id'], 'purpose': d['purpose'],
            'task_id': d['task'].get('id'), 'task_digest': digest(d['task']), 'config_digest': digest(d['config']),
            'code_digest': d.get('code_digest', ''), 'input_plan': d['input_plan'],
            'world_size': len(d['nodes']) * int(d['gpus_per_node']), 'nodes': list(d['nodes']),
            'steps': d['steps'], 'window': list(d['window']) if d.get('window') else None}


def assemble(request, signals_dir: str | Path, parts: dict, extra_problems: list[str] | None = None) -> dict:
    """Build the receipt from the signal files plus the trainer's parts (invariants, env, phases, events...)."""
    ident = identity(request)
    signals = read_signals(signals_dir)
    table = per_step_table(signals)
    world = ident['world_size']
    complete = [r for r in table if r['ranks_reporting'] == world]
    done = {r['step'] for r in complete}
    steps_completed = 0                       # longest complete prefix 1..k, not the largest step seen
    while steps_completed + 1 in done:
        steps_completed += 1
    receipt = dict(ident)
    receipt.update({
        'overlay_digest': parts.get('overlay_digest'),
        'steps_completed': steps_completed,
        'step_time': None if ident['purpose'] == 'probe' else window_stats(table, ident['window'], world),
        'per_step': table,
        'invariants': parts.get('invariants', {}),
        'memory': memory_summary(signals),
        'events': dict(parts.get('events', {}), **allocator_events(signals)),
        'phases': parts.get('phases', {}),
        'env': parts.get('env', {}),
        'effective_config': parts.get('effective_config'),
        'signals_digest': signals_digest(signals_dir),
        'created_utc': utc_now(),
    })
    receipt['events'].setdefault('nccl_errors', 0)
    receipt['events'].setdefault('exceptions', [])
    oom_records = sum(1 for e in receipt['events']['exceptions'] if isinstance(e, dict) and e.get('kind') == 'oom')
    receipt['events']['alloc_ooms'] = max(receipt['events']['alloc_ooms'], oom_records)   # OOM before any row
    receipt['events']['nan_inf_steps'] = sum(
        1 for r in table if any(r.get(k) is None or not math.isfinite(float(r[k])) for k in ('main_ce', 'grad_norm')))
    if ident['purpose'] == 'lifecycle' or 'lifecycle' in parts:
        receipt['lifecycle'] = parts.get('lifecycle', {})
    problems = list(extra_problems or [])
    problems += parts.get('problems', [])
    target = int(parts.get('target_steps', ident['steps']))   # lifecycle: save_at + resume_steps
    if steps_completed != target:
        problems.append(f'incomplete: {steps_completed}/{target} updates')
    missing = [s for s in range(1, target + 1) if s not in done]
    if missing and steps_completed == target:
        problems.append(f'missing or partial updates: {missing[:10]}')
    unexpected = sorted({r['step'] for r in table} - set(range(1, target + 1)))
    if unexpected:
        problems.append(f'unexpected update rows: {unexpected[:10]}')
    duplicated = sorted(r for r, rows in signals.items() if len(rows) != len({int(x['step']) for x in rows}))
    if duplicated:
        problems.append(f'duplicate signal rows on ranks {duplicated[:8]}')
    if ident['purpose'] != 'probe' and receipt['step_time'] is None:
        problems.append('scoring window incomplete')
    problems += row_problems(complete, parts.get('expected', {}))
    problems += clock_problems(signals)
    pairs = repeated_pairs(complete)
    receipt['invariants'] = dict(receipt['invariants'] or {}, self_check={'repeated_pairs': pairs})
    if pairs:
        problems.append(f'repeated adjacent scalars (main_ce, grad_norm) at updates {pairs[:4]}')
    want_mb = (parts.get('expected') or {}).get('microbatches')
    if want_mb is not None:
        bad_mb = sorted({(r, row.get('step')) for r, rows in signals.items() for row in rows
                         if row.get('microbatches') != want_mb})
        if bad_mb:
            problems.append(f'microbatches != {want_mb} at (rank, update) {bad_mb[:6]}')
    if receipt['events']['alloc_ooms']:
        problems.append(f'allocator OOMs: {receipt["events"]["alloc_ooms"]}')
    if receipt['events']['exceptions']:
        problems.append('exceptions recorded')
    receipt['invalid_reasons'] = sorted(set(problems), key=problems.index)
    receipt['valid'] = not receipt['invalid_reasons']
    shape = check_receipt_shape(receipt)
    if shape:
        receipt['invalid_reasons'] += [f'shape: {p}' for p in shape]
        receipt['valid'] = False
    return receipt


def fallback(request, out_dir: str | Path, reason: str) -> dict:
    """Invalid receipt written by the supervisor when the trainer could not write its own."""
    out_dir = Path(out_dir)
    parts_path = out_dir / PARTS_FILE
    try:
        parts = json.loads(parts_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        parts = {}
    events = parts.setdefault('events', {})
    events.setdefault('exceptions', []).append(reason)
    receipt = assemble(request, out_dir / 'signals', parts, extra_problems=[reason])
    receipt['valid'] = False
    return receipt


def expected_lr_table(steps: list[int], schedule: dict) -> dict[int, float]:
    return {s: inv.lr_at(s, schedule) for s in steps}
