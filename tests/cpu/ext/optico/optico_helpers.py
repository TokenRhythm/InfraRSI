"""Synthetic OptiCo outputs for the ENV-B CPU tests: Megatron last-rank logs in the format observed on WINH4 (R3,
r3_20261021a qwen2.5-0.5B) and report.csv rows as run_monitor.sh writes them. No GPU, no network."""
from __future__ import annotations

import json
from pathlib import Path

from infrarsi.ext.optico import params as pm
from infrarsi.ext.optico import parse
from infrarsi.ext.optico import workloads as wl

R3_W3_TFLOPS = [94.7, 223.6, 233.6, 208.1, 244.8]          # the real R3 qwen2.5-0.5B default run
R3_W3_MS = [19341.4, 8187.8, 7837.2, 8796.6, 7480.4]
WS = '/data/infrarsi/ext/runs/optico/r/c1-a1/workspace'


def torchrun_line(workload, params: dict, *, gbs=wl.GBS, seq=wl.SEQ, iters=wl.TRAIN_ITERS, precision='fp16',
                  drop=(), extra: str = '') -> str:
    """The launch command run_<model>.sh echoes, built with the script's own flag logic."""
    w, p = wl.get(workload), params
    tp = p['TP']
    parts = [f'torchrun --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 9987 '
             f'{WS}/code/OptiCo/Pai-Megatron-Patch-v0-12-3/examples/qwen2/pretrain_qwen.py',
             '--lr 1e-5 --min-lr 1e-6 --lr-decay-style cosine --weight-decay 0.1',
             f'--lr-decay-iters {iters} --lr-warmup-iters 0 --train-iters {iters} --micro-batch-size {p["MICRO_BATCH_SIZE"]} '
             f'--global-batch-size {gbs} --num-layers {w.num_layers} --hidden-size {w.hidden} --num-attention-heads {w.heads}']
    if w.ffn_flag:
        parts.append(f'--ffn-hidden-size {w.ffn}')
    parts.append(f'--seq-length {seq} --max-position-embeddings 32768 --log-interval 1 --log-throughput '
                 f'--tensor-model-parallel-size {tp} --pipeline-model-parallel-size {p["PP"]}')
    if w.cp_flag:
        parts.append('--context-parallel-size 1')
    parts.append('--no-load-optim --no-load-rng --num-workers 1 --seed 1234 --swiglu')
    parts.append({'fp16': '--fp16 --apply-query-key-layer-scaling --initial-loss-scale 65536', 'bf16': '--bf16'}[precision])
    parts.append('--transformer-impl transformer_engine')
    if p['AC'] == 'full':
        parts.append(f'--recompute-method uniform --recompute-num-layers {p["AC_LAYERS"]} --recompute-granularity full')
    elif p['AC'] == 'sel':
        parts.append('--recompute-activations')
    elif p['AC'] == 'offload':
        parts.append(f'--cpu-offloading --cpu-offloading-num-layers {p["AC_LAYERS"]}')
    parts.append('--use-distributed-optimizer --use-flash-attn')
    if tp > 1:
        parts.append('--sequence-parallel')
    if w.gqa_flag:
        parts.append(f'--group-query-attention --num-query-groups {w.kv_heads}')
    if p['AC'] != 'offload':
        if p['OVERLAP_GRAD_REDUCE']:
            parts.append('--overlap-grad-reduce')
        if p['OVERLAP_PARAM_GATHER']:
            parts.append('--overlap-param-gather')
    if tp > 1:
        parts.append('--tp-comm-overlap')
    if p['FUSED_CROSS_ENTROPY']:
        parts.append('--cross-entropy-loss-fusion')
    if p['FIRST_PP_NUM_LAYERS'] is not None:
        parts.append(f'--decoder-first-pipeline-num-layers {p["FIRST_PP_NUM_LAYERS"]}')
    if p['LAST_PP_NUM_LAYERS'] is not None:
        parts.append(f'--decoder-last-pipeline-num-layers {p["LAST_PP_NUM_LAYERS"]}')
    if extra:
        parts.append(extra)
    line = ' '.join(parts)
    for flag in drop:
        line = line.replace(flag, '--dropped-flag')
    return line


def iteration_line(i, n, ms, tflops, loss, skipped=0):
    t = '' if tflops is None else f' throughput per GPU (TFLOP/s/GPU): {tflops:.1f} |'
    lm = '' if loss is None else f' lm loss: {loss:.6E} |'
    return (f' [2026-10-21 04:13:{i:02d}] iteration {i:>8d}/{n:>8d} | consumed samples: {1024 * i:>12d} | '
            f'elapsed time per iteration (ms): {ms:.1f} |{t} learning rate: 1.000000E-06 | global batch size:  1024 |'
            f'{lm} loss scale: 65536.0 | grad norm: 4.753 | number of skipped iterations: {skipped:>3d} | '
            'number of nan iterations:   0 |')


def megatron_log(workload, params: dict, *, ms=R3_W3_MS, tflops=R3_W3_TFLOPS, loss1=11.92976, n_iters=5,
                 total=wl.TRAIN_ITERS, max_reserved_mb=12668.0, ac_line=None, ac_layers_line=None, **line_kw) -> str:
    lines = [torchrun_line(workload, params, **line_kw), f'AC is {ac_line or params["AC"]}',
             f'AC_LAYERS is {params["AC_LAYERS"] if ac_layers_line is None else ac_layers_line}',
             'INFO:megatron.core.optimizer:Setting up optimizer ...',
             'WARNING: could not find the metadata file /x/latest_checkpointed_iteration.txt',
             '    will not load any checkpoints and will start from random']
    for i in range(1, n_iters + 1):
        lines.append(iteration_line(i, total, ms[i - 1], tflops[i - 1] if tflops else None,
                                    loss1 - 0.1 * (i - 1) if loss1 else None))
        if i == 1:
            lines.append(f'[Rank 0] (after 1 iterations) memory (MB) | allocated: 3225.5 | max allocated: 9895.0 | '
                         f'reserved: {max_reserved_mb} | max reserved: {max_reserved_mb}')
    lines.append('[after training is done] datetime: 2026-10-21 04:13:32')
    return '\n'.join(lines) + '\n'


def report_row(params: dict, *, result='success', tflops=None, log_dir='trial', reason='NULL', mem_max=16727) -> dict:
    p = params
    ok = result == 'success'
    tf = (sum(R3_W3_TFLOPS[2:]) / 3) if tflops is None else tflops
    fmt = lambda v: 'true' if v is True else ('false' if v is False else ('None' if v is None else str(v)))  # noqa: E731
    row = {k: fmt(p[k]) for k in pm.KNOBS}
    row.update({'Result': result, 'TFLOPS': f'{tf:g}' if ok else 'NULL', 'MFU': f'{tf / 312:g}' if ok else 'NULL',
                'MemAve': '7261.2' if ok else 'NULL', 'MemMax': str(mem_max) if ok else 'NULL', 'GpuUtilAve': '27.825' if ok else 'NULL',
                'GpuUtilMax': '100' if ok else 'NULL', 'Log Director': log_dir, 'Reasons for failure': reason})
    return row


def report_csv(rows: list[dict]) -> str:
    import csv
    import io
    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator='\n')
    wr.writerow(parse.DENSE_HEADER)
    for r in rows:
        wr.writerow([r[h] for h in parse.DENSE_HEADER])
    return buf.getvalue()


def frozen_json(files: dict, tree='t' * 64) -> str:
    return json.dumps({'files': files, 'n_files': len(files) + 100, 'tree_digest': tree})


def write(root: Path, rel: str, text: str) -> Path:
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding='utf-8')
    return p
