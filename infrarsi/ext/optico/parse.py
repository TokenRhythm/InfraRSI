"""Parsers for OptiCo's outputs: report.csv (the judge's table), the last node's Megatron log (megatron_log/<WORLD_SIZE-1>.txt),
memory.txt and the parameter file. Pure functions over text; the runner and the native driver share them.

The minimized metric is the mean `elapsed time per iteration` over iterations 3..5 of the last-rank log (Megatron
prints the iteration line on the last rank only). At fixed work (GBS 1024 x seq 4096, 5 iterations) it is inverse to
the TFLOPS/GPU that OptiCo's run_monitor.sh averages over the same iterations (`tail -n +3` of the throughput lines).
"""
from __future__ import annotations

import csv
import io
import math
import re

from . import workloads as wl

DENSE_HEADER = ('MICRO_BATCH_SIZE', 'TP', 'PP', 'OVERLAP_GRAD_REDUCE', 'OVERLAP_PARAM_GATHER', 'AC', 'AC_LAYERS',
                'FUSED_CROSS_ENTROPY', 'FIRST_PP_NUM_LAYERS', 'LAST_PP_NUM_LAYERS', 'Result', 'TFLOPS', 'MFU', 'MemAve',
                'MemMax', 'GpuUtilAve', 'GpuUtilMax', 'Log Director', 'Reasons for failure')
_ITER = re.compile(r'\biteration\s+(\d+)\s*/\s*(\d+)\s*\|(.*)$')
_TFLOPS_OPTICO = re.compile(r'throughput per GPU \(TFLOP/s/GPU\): (\d+\.\d+)')       # run_monitor.sh's grep -oP
_MEM = re.compile(r'\[Rank (\d+)\] \(after (\d+) iterations\) memory \(MB\) \| (.*)$')
_NUM = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')


def _num(s) -> float | None:
    if s is None:
        return None
    m = _NUM.search(str(s))
    if not m:
        return None
    try:
        v = float(m.group(0))
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def cell_number(v) -> float | None:
    """A numeric report.csv cell ('NULL', '', 'None' -> None)."""
    if v is None:
        return None
    s = str(v).strip()
    if s.upper() in ('NULL', 'NONE', 'NAN', ''):
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


# ---------------------------------------------------------------- report.csv
def read_report(text: str) -> tuple[list[str], list[dict]]:
    """(header, rows) of report.csv. Like run_circle.sh's row count, '#' comment lines and blank lines are skipped;
    quoted cells (loc_error rewrites the file through pandas) are handled by the csv module."""
    lines = [ln for ln in (text or '').splitlines() if ln.strip() and not ln.startswith('#')]
    if not lines:
        return [], []
    reader = csv.reader(io.StringIO('\n'.join(lines)))
    rows = list(reader)
    header = [h.strip() for h in rows[0]]
    out = []
    for r in rows[1:]:
        if not any(c.strip() for c in r):
            continue
        r = r + [''] * (len(header) - len(r)) if len(r) < len(header) else r
        row = {h: r[i] for i, h in enumerate(header)}
        if len(r) > len(header):                      # an unquoted comma in the last (free-text) column
            row[header[-1]] = ','.join(r[len(header) - 1:])
        out.append(row)
    return header, out


def row_result(row: dict) -> str:
    return str(row.get('Result', '')).strip().lower()


def row_judge(row: dict) -> dict:
    """The judge's own row, verbatim (strings), plus its parsed numbers."""
    return {'row': dict(row), 'result': row_result(row), 'tflops': cell_number(row.get('TFLOPS')),
            'mfu_optico': cell_number(row.get('MFU')), 'mem_ave_mib': cell_number(row.get('MemAve')),
            'mem_max_mib': cell_number(row.get('MemMax')), 'gpu_util_ave': cell_number(row.get('GpuUtilAve')),
            'gpu_util_max': cell_number(row.get('GpuUtilMax')), 'log_dir': str(row.get('Log Director', '')).strip(),
            'failure': str(row.get('Reasons for failure', '')).strip()}


# ---------------------------------------------------------------- Megatron log
def torchrun_line(text: str) -> str | None:
    """The launch command run_<model>.sh echoes into the log (first line starting with 'torchrun ')."""
    for line in (text or '').splitlines():
        s = line.strip()
        if s.startswith('torchrun '):
            return s
    return None


def torchrun_args(line: str | None) -> dict:
    """--flag -> value (True for switches; the last occurrence wins)."""
    out = {}
    if not line:
        return out
    toks = line.split()
    i = 1
    while i < len(toks):
        t = toks[i]
        if t.startswith('--'):
            if i + 1 < len(toks) and not toks[i + 1].startswith('--'):
                out[t] = toks[i + 1]
                i += 2
                continue
            out[t] = True
        i += 1
    return out


def _fields(rest: str) -> dict:
    out = {}
    for part in rest.split('|'):
        if ':' in part:
            k, v = part.split(':', 1)
            out[k.strip()] = v.strip()
    return out


def iterations(text: str) -> list[dict]:
    """Megatron training_log lines, first occurrence per iteration, in iteration order."""
    seen = {}
    for line in (text or '').splitlines():
        m = _ITER.search(line)
        if not m:
            continue
        it, of = int(m.group(1)), int(m.group(2))
        if it in seen:
            continue
        f = _fields(m.group(3))
        seen[it] = {'iteration': it, 'of': of,
                    'elapsed_ms': _num(f.get('elapsed time per iteration (ms)')),
                    'tflops': _num(f.get('throughput per GPU (TFLOP/s/GPU)')),
                    'lm_loss': _num(f.get('lm loss')), 'loss_scale': _num(f.get('loss scale')),
                    'grad_norm': _num(f.get('grad norm')),
                    'skipped': int(_num(f.get('number of skipped iterations')) or 0),
                    'nan': int(_num(f.get('number of nan iterations')) or 0)}
    return [seen[k] for k in sorted(seen)]


def memory_report(text: str) -> dict:
    """Megatron's '[Rank r] (after n iterations) memory (MB) | ...' lines (data-parallel rank 0 of each TP x PP slot)."""
    ranks = {}
    for line in (text or '').splitlines():
        m = _MEM.search(line)
        if not m:
            continue
        ranks[int(m.group(1))] = {k: _num(v) for k, v in _fields('|' + m.group(3)).items()}
    peak = [r.get('max reserved') for r in ranks.values() if r.get('max reserved') is not None]
    return {'per_rank': ranks, 'max_reserved_mb': max(peak) if peak else None}


def optico_tflops(text: str) -> float:
    """run_monitor.sh's TFLOPS exactly: throughput values matched by its regex, skip the first two, mean, else 0."""
    vals = [float(x) for x in _TFLOPS_OPTICO.findall(text or '')][2:]
    return sum(vals) / len(vals) if vals else 0.0


def log_facts(text: str) -> dict:
    line = torchrun_line(text)
    ac = re.search(r'^AC is ([a-z]*)\s*$', text or '', re.M)
    acl = re.search(r'^AC_LAYERS is (\d*)\s*$', text or '', re.M)
    script_errors = [ln.strip() for ln in (text or '').splitlines() if ln.startswith('Error:')][:10]
    low = (text or '').lower()
    return {'torchrun': line, 'args': torchrun_args(line), 'ac': ac.group(1) if ac else None,
            'ac_layers': int(acl.group(1)) if acl and acl.group(1) else None, 'iterations': iterations(text),
            'memory': memory_report(text), 'optico_tflops': optico_tflops(text), 'script_errors': script_errors,
            'oom': 'out of memory' in low or 'outofmemoryerror' in low,
            'random_init': 'will not load any checkpoints and will start from random' in (text or ''),
            'bytes': len(text or '')}


def window_metric(iters: list[dict], window=wl.WINDOW) -> dict:
    """Mean elapsed seconds and TFLOPS over the window iterations; values None when any window iteration is missing."""
    a, b = window
    rows = {r['iteration']: r for r in iters}
    sel = [rows.get(i) for i in range(a, b + 1)]
    ms = [r['elapsed_ms'] if r else None for r in sel]
    tf = [r['tflops'] if r else None for r in sel]
    ok_t = all(v is not None and v > 0 for v in ms)
    ok_f = all(v is not None and v > 0 for v in tf)
    mean_s = sum(ms) / len(ms) / 1000.0 if ok_t else None
    mean_tf = sum(tf) / len(tf) if ok_f else None
    rsd = None
    if ok_f and len(tf) > 1 and mean_tf:
        sd = math.sqrt(sum((x - mean_tf) ** 2 for x in tf) / (len(tf) - 1))
        rsd = sd / mean_tf
    return {'window': [a, b], 'iter_ms': ms, 'tflops': tf, 'mean_iter_s': mean_s, 'mean_tflops': mean_tf,
            'tflops_rsd': rsd}


# ---------------------------------------------------------------- memory.txt (monitor_gpu.sh samples)
def memory_samples(text: str) -> dict:
    """run_monitor.sh's own aggregation of memory.txt (MiB values of the 'Memory-Usage' column), for cross-checks."""
    used, limits = [], []
    for line in (text or '').splitlines():
        if 'MiB' not in line:
            continue
        parts = line.split()
        if len(parts) >= 4:
            u, lim = _num(parts[2]), _num(parts[3])
            if u is not None:
                used.append(u)
            if lim is not None:
                limits.append(lim)
    return {'samples': len(used), 'mem_max_mib': max(used) if used else None,
            'mem_ave_mib': sum(used) / len(used) if used else None, 'mem_limit_mib': min(limits) if limits else None}


def error_head(text: str, n: int = 30) -> list[str]:
    """First n lines of OptiCo's extracted error file (monitor_error.txt), as the proposer sees failures."""
    return [ln.rstrip() for ln in (text or '').splitlines() if ln.strip()][:n]
