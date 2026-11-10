"""Parsers for the autoresearch train.py output: the final summary block and the per-step progress lines.

The progress line is printed with a carriage return and no newline:
  step 00012 (0.1%) | loss: 6.120718 | lrm: 1.00 | dt: 317ms | tok/sec: 1,651,836 | mfu: 39.9% | epoch: 1 | remaining: 299s
Lines whose frequency the candidate changed are simply fewer; the format itself is frozen (design §4.2).
"""
from __future__ import annotations

import math
import re

SUMMARY_KEYS = ('val_bpb', 'training_seconds', 'total_seconds', 'peak_vram_mb', 'mfu_percent', 'total_tokens_M', 'num_steps',
                'num_params_M', 'depth')
_SUMMARY = re.compile(r'^(%s):\s+(\S+)\s*$' % '|'.join(SUMMARY_KEYS), re.M)
_STEP = re.compile(r'step (\d+) \(([\d.]+)%\) \| loss: ([-+\d.eEnainf]+) \| lrm: ([\d.]+) \| dt: (\d+)ms \| tok/sec: ([\d,]+) \| '
                   r'mfu: ([\d.]+)% \| epoch: (\d+) \| remaining: (\d+)s')


def _num(s: str):
    try:
        v = float(s)
    except ValueError:
        return None
    return int(v) if v.is_integer() and 'e' not in s.lower() and '.' not in s else v


def summary(text: str) -> dict:
    """The '---' block printed at the end (last occurrence of each key)."""
    out = {}
    for m in _SUMMARY.finditer(text or ''):
        out[m.group(1)] = _num(m.group(2))
    return out


def steps(text: str) -> list[dict]:
    """Every parsed progress line in order (the carriage-return stream split into records)."""
    rows = []
    for m in _STEP.finditer((text or '').replace('\r', '\n')):
        loss = float(m.group(3)) if re.fullmatch(r'[-+\d.eE]+', m.group(3)) else math.nan
        rows.append({'step': int(m.group(1)), 'loss': loss, 'lrm': float(m.group(4)), 'dt_ms': int(m.group(5)),
                     'tok_per_sec': int(m.group(6).replace(',', '')), 'mfu': float(m.group(7)), 'epoch': int(m.group(8))})
    return rows


def first_losses(text: str, n: int = 10) -> dict[int, float]:
    """{step: printed loss} for steps 0..n-1 that were printed."""
    return {r['step']: r['loss'] for r in steps(text) if r['step'] < n}


def steady_stats(rows: list[dict], skip: int = 10) -> dict:
    """Median dt and tok/s over the timed steps (step > skip); report numbers, not a metric."""
    xs = sorted(r['dt_ms'] for r in rows if r['step'] > skip)
    ts = sorted(r['tok_per_sec'] for r in rows if r['step'] > skip)
    if not xs:
        return {'steps_parsed': len(rows)}
    return {'steps_parsed': len(rows), 'median_dt_ms': xs[len(xs) // 2], 'median_tok_per_sec': ts[len(ts) // 2],
            'last_step': rows[-1]['step']}


# AMENDMENT_EXT_007 §4.4: n_failed by class per arm (CUDA-graph, OOM, other) - from the attempt's own log
_CUDA_GRAPH = re.compile(r'CUDAGraphs|cudagraph|CUDA graph', re.I)
_OOM = re.compile(r'OutOfMemoryError|CUDA out of memory')


def failure_class(log_text: str | None, train_exit, timed_out: bool = False) -> str | None:
    """'cuda_graph' | 'oom' | 'timeout' | 'other' for a failed training run, None when it exited 0."""
    if train_exit == 0 and not timed_out:
        return None
    tail = (log_text or '')[-20000:]
    if _OOM.search(tail):
        return 'oom'
    if _CUDA_GRAPH.search(tail) and ('Error' in tail or 'error' in tail):
        return 'cuda_graph'
    return 'timeout' if timed_out else 'other'
