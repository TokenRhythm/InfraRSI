"""ENV-B guards (design §3.4). They only add rejections to OptiCo's verdict; reports never veto.

G-B1 fixed_flags: the torchrun line in the log keeps --global-batch-size 1024 --seq-length 4096 --train-iters 5 and
      the fp16 precision flags (design list), plus the other script constants that are outside the space (model shape,
      8 GPUs on 1 node, LR, seed, DO/TE/flash, SP iff TP>1, CP 1) and, when the runner knows what it asked for, the
      requested knobs (MBS/TP/PP, overlap, AC/AC_LAYERS, fused CE, first/last split).
G-B3 completed_iterations: 5 of 5 iterations logged and iterations 3..5 have throughput (and elapsed-time) values.
G-B2 iteration1_loss: |loss_1 - loss_1(default)| / loss_1(default) <= 2 %; a report unless the enabling rule holds:
      default and B-ED (different layout) differ by <= 2 % at iteration 1 -> G-B2 becomes a hard veto.
"""
from __future__ import annotations

import dataclasses
import math

from ..common.guards import GuardResult
from . import workloads as wl

G_B1, G_B2, G_B3 = 'G-B1', 'G-B2', 'G-B3'
LOSS_TOL = 0.02
PRECISION = {'--fp16': True, '--apply-query-key-layer-scaling': True, '--initial-loss-scale': '65536'}
FORBIDDEN_PRECISION = ('--bf16', '--fp8-format', '--fp8-amax-compute-algo')


def _same_num(a, b) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def fixed_flags(args: dict, workload, requested: dict | None = None, *, ac_line: str | None = None,
                ac_layers_line: int | None = None) -> GuardResult:
    """G-B1 over the parsed torchrun line (parse.torchrun_args); requested = the normalized knobs the runner wrote."""
    w = wl.get(workload)
    g = GuardResult()
    if not args:
        return g.reject(G_B1, 'no torchrun line in the log (training never launched)')
    bad = []

    def want(flag, value, label=None):
        got = args.get(flag)
        if value is True:                                  # a switch
            if got is None:
                bad.append(f'{label or flag} missing')
        elif got is None or got is True or (not _same_num(got, value) and str(got) != str(value)):
            bad.append(f'{label or flag} {got!r} != {value!r}')

    # design list
    want('--global-batch-size', wl.GBS)
    want('--seq-length', wl.SEQ)
    want('--train-iters', wl.TRAIN_ITERS)
    for flag, value in PRECISION.items():
        want(flag, value)
    for flag in FORBIDDEN_PRECISION:
        if flag in args:
            bad.append(f'{flag} present (precision is fixed fp16)')
    core = list(bad)
    # other script constants
    want('--nproc_per_node', wl.GPUS)
    want('--nnodes', 1)
    want('--num-layers', w.num_layers)
    want('--hidden-size', w.hidden)
    want('--num-attention-heads', w.heads)
    if w.ffn_flag:
        want('--ffn-hidden-size', w.ffn)
    if w.gqa_flag:
        want('--num-query-groups', w.kv_heads)
    want('--lr', '1e-5')
    want('--min-lr', '1e-6')
    want('--seed', 1234)
    want('--use-distributed-optimizer', True)
    want('--use-flash-attn', True)
    want('--transformer-impl', 'transformer_engine')
    if w.cp_flag:
        want('--context-parallel-size', 1)
    tp = args.get('--tensor-model-parallel-size')
    try:
        tp_i = int(tp)
    except (TypeError, ValueError):
        tp_i = None
        bad.append(f'--tensor-model-parallel-size {tp!r} unreadable')
    if tp_i is not None:
        for flag in ('--sequence-parallel', '--tp-comm-overlap'):      # the script adds both iff TP > 1
            if (tp_i > 1) != (flag in args):
                bad.append(f'{flag} {"missing" if tp_i > 1 else "present"} with TP {tp_i} (script: iff TP > 1)')
    want('--swiglu', True)
    if w.gqa_flag:
        want('--group-query-attention', True)
    elif '--group-query-attention' in args:
        bad.append('--group-query-attention present but this script has no GQA')
    script = bad[len(core):]
    applied = []
    if requested:
        r = requested
        for flag, key in (('--micro-batch-size', 'MICRO_BATCH_SIZE'), ('--tensor-model-parallel-size', 'TP'),
                          ('--pipeline-model-parallel-size', 'PP')):
            if not _same_num(args.get(flag), r[key]):
                applied.append(f'{flag} {args.get(flag)!r} != requested {r[key]}')
        for flag, key in (('--overlap-grad-reduce', 'OVERLAP_GRAD_REDUCE'), ('--overlap-param-gather', 'OVERLAP_PARAM_GATHER'),
                          ('--cross-entropy-loss-fusion', 'FUSED_CROSS_ENTROPY')):
            if (flag in args) != bool(r[key]):
                applied.append(f'{flag} {"present" if flag in args else "absent"} but requested {key}={r[key]}')
        for flag, key in (('--decoder-first-pipeline-num-layers', 'FIRST_PP_NUM_LAYERS'),
                          ('--decoder-last-pipeline-num-layers', 'LAST_PP_NUM_LAYERS')):
            got = args.get(flag)
            if (r[key] is None) != (got is None) or (got is not None and not _same_num(got, r[key])):
                applied.append(f'{flag} {got!r} != requested {r[key]}')
        ac = r['AC']
        if ac_line is not None and ac_line != ac:
            applied.append(f'log "AC is {ac_line}" != requested {ac}')
        if ac_layers_line is not None and ac_layers_line != r['AC_LAYERS']:
            applied.append(f'log "AC_LAYERS is {ac_layers_line}" != requested {r["AC_LAYERS"]}')
        if ac == 'full':
            if args.get('--recompute-granularity') != 'full' or args.get('--recompute-method') != 'uniform' \
                    or not _same_num(args.get('--recompute-num-layers'), r['AC_LAYERS']):
                applied.append('full recompute flags differ from AC=full/AC_LAYERS')
        elif ac == 'sel':
            if '--recompute-activations' not in args:
                applied.append('--recompute-activations missing for AC=sel')
        elif ac == 'offload':
            if '--cpu-offloading' not in args or not _same_num(args.get('--cpu-offloading-num-layers'), r['AC_LAYERS']):
                applied.append('CPU offloading flags differ from AC=offload/AC_LAYERS')
        if ac != 'full' and '--recompute-granularity' in args:
            applied.append(f'--recompute-granularity present with AC={ac}')
        if ac != 'sel' and '--recompute-activations' in args:
            applied.append(f'--recompute-activations present with AC={ac}')
        if ac != 'offload' and '--cpu-offloading' in args:
            applied.append(f'--cpu-offloading present with AC={ac}')
    for group, items in (('design fixed flags', core), ('script constants', script), ('requested knobs', applied)):
        if items:
            g.reject(G_B1, f'{group}: ' + '; '.join(items))
    return g


def completed_iterations(iters: list[dict], window=wl.WINDOW) -> GuardResult:
    """G-B3: iterations 1..5 logged (of 5) and the window iterations carry throughput and elapsed time."""
    g = GuardResult()
    got = {r['iteration']: r for r in iters}
    missing = [i for i in range(1, wl.TRAIN_ITERS + 1) if i not in got]
    wrong_total = sorted({r['of'] for r in iters if r['of'] != wl.TRAIN_ITERS})
    if missing:
        g.reject(G_B3, f'iterations {missing} of {wl.TRAIN_ITERS} not in the log ({len(got)} logged)')
    if wrong_total:
        g.reject(G_B3, f'log reports {wrong_total} total iterations, expected {wl.TRAIN_ITERS}')
    a, b = window
    no_tp = [i for i in range(a, b + 1) if i in got and not got[i].get('tflops')]
    no_t = [i for i in range(a, b + 1) if i in got and not got[i].get('elapsed_ms')]
    if no_tp:
        g.reject(G_B3, f'no throughput value for iterations {no_tp}')
    if no_t:
        g.reject(G_B3, f'no elapsed time for iterations {no_t}')
    skipped = [r['iteration'] for r in iters if r.get('skipped')]
    nan = [r['iteration'] for r in iters if r.get('nan')]
    if skipped or nan:
        g.report(G_B3, skipped_iterations=skipped, nan_iterations=nan,
                 detail='fp16 loss-scale skips / NaN iterations (reported, not a veto)')
    return g


@dataclasses.dataclass(frozen=True)
class GB2Policy:
    """Reference iteration-1 loss (the default run of the workload) and whether G-B2 vetoes (enabling rule)."""
    reference_loss1: float | None = None
    hard: bool = False
    tol: float = LOSS_TOL
    reference_attempt: str | None = None
    rule: str | None = None             # how `hard` was decided (enabling_rule output)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def loss1(iters: list[dict]) -> float | None:
    for r in iters:
        if r['iteration'] == 1:
            v = r.get('lm_loss')
            return v if v is not None and v > 0 else None
    return None


def receipt_loss1(receipt: dict | None) -> float | None:
    try:
        v = receipt['loss']['iter1']
    except (KeyError, TypeError):
        return None
    return float(v) if type(v) in (int, float) and math.isfinite(v) and v > 0 else None


def enabling_rule(default_loss1: float | None, bed_loss1: float | None, *, tol: float = LOSS_TOL) -> tuple[bool, str]:
    """Design §3.4: G-B2 becomes a hard veto iff default and B-ED differ by <= tol at iteration 1."""
    if default_loss1 is None or bed_loss1 is None:
        return False, 'enabling rule undecided (iteration-1 loss of the default or of B-ED unavailable): G-B2 stays a report'
    rel = abs(bed_loss1 - default_loss1) / default_loss1
    if rel <= tol:
        return True, f'default {default_loss1:.6g} vs B-ED {bed_loss1:.6g}: rel {rel:.4f} <= {tol} -> G-B2 is a hard veto'
    return False, f'default {default_loss1:.6g} vs B-ED {bed_loss1:.6g}: rel {rel:.4f} > {tol} -> G-B2 stays a report'


def policy_from_receipts(default_receipt: dict | None, bed_receipt: dict | None, *, tol: float = LOSS_TOL) -> GB2Policy:
    ref = receipt_loss1(default_receipt)
    hard, rule = enabling_rule(ref, receipt_loss1(bed_receipt), tol=tol)
    return GB2Policy(reference_loss1=ref, hard=hard, tol=tol, rule=rule,
                     reference_attempt=(default_receipt or {}).get('attempt_id'))


def iteration1_loss(value: float | None, policy: GB2Policy | None) -> GuardResult:
    g = GuardResult()
    p = policy or GB2Policy()
    if p.reference_loss1 is None or value is None:
        return g.report(G_B2, value=value, reference=p.reference_loss1, rel=None, limit=p.tol, veto=p.hard, exceeded=None,
                        detail='iteration-1 lm loss or the default reference unavailable (no check)')
    rel = abs(value - p.reference_loss1) / p.reference_loss1
    exceeded = rel > p.tol
    detail = f'iteration-1 lm loss {value:.6g} vs default {p.reference_loss1:.6g}: rel {rel:.4f} (limit {p.tol})'
    if exceeded and p.hard:
        g.reject(G_B2, detail)
    return g.report(G_B2, value=value, reference=p.reference_loss1, rel=round(rel, 6), limit=p.tol, veto=p.hard,
                    exceeded=exceeded, detail=detail, rule=p.rule)
