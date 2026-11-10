"""Re-derive per-step and window statistics from signals/rank<r>.jsonl (the receipt summary is never trusted).

StepTime = mean over the scoring window of max over ranks of whole_update_s (design 1.6). RSD uses the sample
standard deviation. Window [a, b] is in 1-based update numbering; signals may number steps from 0 or 1 and the
base is detected from rank 0. Every whole_update_s must be finite and positive.

InterStep comes only from the signals: per-rank inter_step_s (start(k) - start(k-1), runtime/signals.py) with every
rank reporting a finite positive value at every update a+1..b; InterStep is the mean of the max-rank values
(receipt_writer convention). Otherwise it is not derivable (a problem) - there is no fallback to any summary the
training process reports.

The profile block gives, per timer (whole/fwd_bwd/optim/data), the window mean of the max-rank and of the min-rank
value. The microbatches block collects every rank's per-update backward micro-batch count (AMENDMENT-001 invariant). The clock block compares each row's UTC span (utc_start_s/utc_end_s, time.time at the same points as
whole_update_s) with the perf-counter time and records rows without UTC stamps and rows whose stamps are not
strictly positive and ordered.

signals_digest has a single convention (INTERFACES): sha256 of the rank files rank<r>.jsonl concatenated in
increasing rank order, exactly as runtime/signals.signals_digest computes it.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics

GLOBAL_KEYS = ('main_ce', 'aux_loss', 'grad_norm', 'lr', 'physical_tokens', 'supervised_tokens', 'optimizer_steps',
               'clip_calls')
PROFILE_KEYS = ('whole_update_s', 'fwd_bwd_s', 'optim_s', 'data_s')
CLOCK_REL, CLOCK_ABS = 0.02, 0.005      # |(utc_end_s - utc_start_s) - whole_update_s| <= 2 % + 5 ms per rank and step
RANK_RE = re.compile(r'rank(\d+)\.jsonl')            # the whole file name, as runtime/signals.signal_files globs it


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def rank_of(name: str) -> int | None:
    m = RANK_RE.fullmatch(str(name).replace('\\', '/').rsplit('/', 1)[-1])
    return int(m.group(1)) if m else None


def signals_digest(texts: dict[str, str]) -> str:
    """sha256 of the concatenated rank files in increasing rank order (the only accepted convention)."""
    h = hashlib.sha256()
    for _, name in sorted((r, n) for n in texts if (r := rank_of(n)) is not None):
        h.update((texts[name] or '').encode('utf-8'))
    return h.hexdigest()


def parse(texts: dict[str, str]) -> tuple[dict[int, dict[int, dict]], list[str]]:
    rows: dict[int, dict[int, dict]] = {}
    problems = []
    for name, text in texts.items():
        r = rank_of(name)
        if r is None:
            continue
        rows[r] = {}
        for ln, line in enumerate((text or '').splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                problems.append(f'rank{r} line {ln}: unparsable')
                continue
            s = obj.get('step') if isinstance(obj, dict) else None
            if not isinstance(s, int) or isinstance(s, bool):
                problems.append(f'rank{r} line {ln}: step {s!r}')
                continue
            if s in rows[r]:
                problems.append(f'rank{r}: duplicate step {s}')
            rows[r][s] = obj
    return rows, problems


def derive(texts: dict[str, str], *, world_size: int, steps: int, window, start_step: int | None = None,
           profile_window=None) -> dict:
    rows, problems = parse(texts)
    missing = [r for r in range(world_size) if r not in rows]
    if missing:
        problems.append(f'signals missing for ranks {missing[:16]}')
    extra = [r for r in rows if r >= world_size]
    if extra:
        problems.append(f'signals for unexpected ranks {extra}')
    r0 = rows.get(0, {})
    base = start_step if start_step is not None else (0 if (r0 and min(r0) == 0) else 1)
    ranks = [r for r in range(world_size) if r in rows]
    # steps_completed: longest prefix present on every expected rank
    done = 0
    while ranks and not missing and all((base + done) in rows[r] for r in ranks):
        done += 1
    if done > steps:
        problems.append(f'{done} updates in signals > requested {steps}')
    per_step = []
    for u in range(1, done + 1):
        s = base + u - 1
        times = [(rows[r][s].get('whole_update_s'), r) for r in ranks]
        bad = [r for t, r in times if not (finite(t) and t > 0)]
        if bad:
            problems.append(f'update {u}: whole_update_s missing, non-finite or not positive on ranks {bad[:8]}')
            times = [(t, r) for t, r in times if finite(t) and t > 0]
        tmax, slow = max(times) if times else (None, None)
        g = r0.get(s, {})
        row = {'step': s, 'update': u, 'max_rank_whole_s': tmax, 'slowest_rank': slow}
        for k in GLOBAL_KEYS:
            row[k] = g.get(k)
        inter = [rows[r][s].get('inter_step_s') for r in ranks]
        ok = [x for x in inter if finite(x) and x > 0]
        row['max_rank_inter_step_s'] = max(ok) if ok and len(ok) == len(ranks) else None   # every rank, or none
        per_step.append(row)
    out_mb = _microbatches(rows, ranks, base, done)
    nan_inf = [p['update'] for p in per_step if not (finite(p['main_ce']) and finite(p['grad_norm']))]
    out = {'ranks_present': ranks, 'step_base': base, 'steps_completed': done, 'per_step': per_step,
           'nan_inf_updates': nan_inf, 'problems': problems, 'signals_digest': signals_digest(texts)}
    out['microbatches'] = out_mb
    out['memory'] = _memory(rows, ranks)
    out['events'] = _events(rows, ranks)
    out['step_time'], out['window_complete'], out['tail_step_ratio'], win_problems = _window(per_step, window)
    problems += win_problems
    if window and not out['window_complete']:
        problems.append(f'window {list(window)} incomplete: {done} updates')
    out['profile'] = profile(rows, ranks, base, profile_window, world_size) if profile_window else None
    out['clock'] = clock(rows, ranks, base, done)
    return out


def profile(rows, ranks, base, window, world_size) -> dict:
    """Per timer: window mean over updates of the max-rank and of the min-rank value (all ranks must report)."""
    a, b = int(window[0]), int(window[1])
    out = {}
    for k in PROFILE_KEYS:
        hi, lo = [], []
        for u in range(a, b + 1):
            vals = [rows[r][base + u - 1].get(k) for r in ranks if (base + u - 1) in rows[r]]
            vals = [v for v in vals if finite(v)]
            if len(vals) == world_size:
                hi.append(max(vals))
                lo.append(min(vals))
        full = len(hi) == b - a + 1
        out[k] = {'window_mean_max_rank_s': statistics.fmean(hi) if full else None,
                  'window_mean_min_rank_s': statistics.fmean(lo) if full else None}
    return out


def clock(rows, ranks, base, done) -> dict:
    """Independent clock, per rank and step: UTC span (utc_end_s - utc_start_s) vs the perf-counter whole_update_s.

    missing: rows without finite UTC stamps; disordered: rows whose stamps are not strictly positive and ordered
    (0 < utc_start_s < utc_end_s, and utc_start_s >= the previous row's utc_end_s on the same rank); outliers: rows
    whose UTC span and whole_update_s differ by more than CLOCK_REL + CLOCK_ABS. All three map rank -> [step].
    train_utc_s = max over ranks of utc_end_s at the last update - min over ranks of utc_start_s at the first."""
    steps = [base + u for u in range(done)]
    res = {'present': False, 'complete': False, 'rows': len(ranks) * len(steps), 'rows_with_utc': 0, 'missing': {},
           'disordered': {}, 'outliers': {}, 'max_abs_dev_s': None, 'train_utc_s': None}
    worst, n_utc = 0.0, 0
    for r in ranks:
        miss, dis, bad = [], [], []
        prev_end = None
        for s in steps:
            row = rows[r][s]
            us, ue, w = row.get('utc_start_s'), row.get('utc_end_s'), row.get('whole_update_s')
            if not (finite(us) and finite(ue)):
                miss.append(s)
                prev_end = None
                continue
            n_utc += 1
            if us <= 0 or ue <= us or (prev_end is not None and us < prev_end):
                dis.append(s)
            prev_end = ue
            if not finite(w):
                bad.append(s)
                continue
            dev = abs((ue - us) - w)
            worst = max(worst, dev)
            if dev > CLOCK_REL * w + CLOCK_ABS:
                bad.append(s)
        for key, val in (('missing', miss), ('disordered', dis), ('outliers', bad)):
            if val:
                res[key][str(r)] = val
    res.update(present=n_utc > 0, complete=bool(steps) and n_utc == res['rows'], rows_with_utc=n_utc)
    if not n_utc:
        return res
    res['max_abs_dev_s'] = worst
    first, last = steps[0], steps[-1]
    starts = [rows[r][first].get('utc_start_s') for r in ranks]
    ends = [rows[r][last].get('utc_end_s') for r in ranks]
    if all(finite(x) for x in starts + ends):
        res['train_utc_s'] = max(ends) - min(starts)
    return res


def _window(per_step, window):
    """(step_time, window_complete, tail_step_ratio, problems)."""
    if not window:
        return None, True, None, []
    a, b = int(window[0]), int(window[1])
    sel = [p for p in per_step if a <= p['update'] <= b]
    if len(sel) != b - a + 1 or any(not finite(p['max_rank_whole_s']) for p in sel):
        return None, False, None, []
    t = [p['max_rank_whole_s'] for p in sel]
    mean = statistics.fmean(t)
    rsd = statistics.stdev(t) / mean if len(t) > 1 and mean > 0 else 0.0
    problems = []
    inter = [p['max_rank_inter_step_s'] for p in sel if p['update'] > a]
    if inter and all(finite(x) for x in inter):
        inter_mean, source = statistics.fmean(inter), 'signals.inter_step_s'
    else:
        gaps = [p['update'] for p in sel if p['update'] > a and not finite(p['max_rank_inter_step_s'])]
        inter_mean, source = None, None
        problems.append(f'InterStep not derivable from the signals: inter_step_s missing, non-finite or not positive '
                        f'on some rank at updates {gaps[:8] or [a]} of window {[a, b]}')
    st = {'mean_window_max_rank_s': mean, 'median_window_max_rank_s': statistics.median(t), 'rsd_window': rsd,
          'inter_step_mean_s': inter_mean, 'inter_step_source': source, 'window': [a, b], 'n': len(t)}
    tail = statistics.fmean(t[-3:]) / mean if mean > 0 else None
    return st, True, tail, problems


def _microbatches(rows, ranks, base, done) -> dict:
    """Per-rank backward micro-batch counts of every completed update (signals 'microbatches', AMENDMENT-001):
    {values: {count: rows}, missing: [[rank, update], ...], n_missing}."""
    values, missing = {}, []
    for u in range(1, done + 1):
        for r in ranks:
            v = rows[r][base + u - 1].get('microbatches')
            if isinstance(v, int) and not isinstance(v, bool):
                values.setdefault(v, []).append([r, u])
            else:
                missing.append([r, u])
    return {'values': {str(v): len(x) for v, x in sorted(values.items())},
            'examples': {str(v): x[:4] for v, x in sorted(values.items())}, 'missing': missing[:16],
            'n_missing': len(missing)}


def _memory(rows, ranks):
    per_rank = {}
    for r in ranks:
        alloc = [x.get('peak_alloc_gib') for x in rows[r].values() if finite(x.get('peak_alloc_gib'))]
        res = [x.get('peak_reserved_gib') for x in rows[r].values() if finite(x.get('peak_reserved_gib'))]
        per_rank[str(r)] = {'peak_alloc_gib': max(alloc) if alloc else None, 'peak_reserved_gib': max(res) if res else None}
    va = [v['peak_alloc_gib'] for v in per_rank.values() if v['peak_alloc_gib'] is not None]
    vr = [v['peak_reserved_gib'] for v in per_rank.values() if v['peak_reserved_gib'] is not None]
    return {'per_rank': per_rank, 'max_peak_alloc_gib': max(va) if va else None,
            'max_peak_reserved_gib': max(vr) if vr else None}


def _events(rows, ranks):
    """alloc_retries/alloc_ooms read as cumulative counters: per-rank max, summed over ranks."""
    out = {}
    for k in ('alloc_retries', 'alloc_ooms'):
        tot = 0
        for r in ranks:
            vals = [x.get(k) for x in rows[r].values() if finite(x.get(k))]
            tot += int(max(vals)) if vals else 0
        out[k] = tot
    return out
