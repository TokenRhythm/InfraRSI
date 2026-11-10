"""Closed-form accounting of one training job (textbook workflow W1; FABLE_DESIGN_CHECK section 2): the per-GPU memory
nvidia-smi will show (parameter, gradient and optimizer-state bytes for the framework's sharding, activations, logits,
and the CUDA context / NCCL buffers and allocator slack the H20 harvest measured per stack, rule h20-R11), and the ideal
time of one update (model FLOPs over the GPUs' measured matrix throughput) -> MFU. An anchor for the model and the
doctor and a probe-first prior where the memory look-up has no match: never a verdict (expect +-25 %).

    estimate(fingerprint, states) -> {mem_gib, mem_terms, capacity_gib, fit_margin_gib, flops_per_update, ...} | None
    mfu(est, step_time_s) -> float | None
"""
from __future__ import annotations

from .store import gpus

GIB = 2 ** 30
# GiB per GPU outside the allocator (CUDA context, NCCL buffers): the H20 fit (h20-memory-model-2 non_torch_gib)
NON_TORCH = {'megatron': 9.3, 'megatron_nvls_off': 5.1, 'megatron_tp4': 10.6, 'megatron_tp8': 11.2, 'megatron_cp': 11.5,
             'megatron_fp8': 12.5, 'megatron_moe': 16.5, 'megatron_moe_nvls_off': 9.4, 'fsdp': 1.3, 'deepspeed': 1.5,
             'hf_trainer': 1.5, 'torchtitan': 1.3, 'lightning': 1.3}
# measured / textbook activations (the Llama-style count below) on the H20 + H200 harvests' activation-heavy runs (the
# median per stack: kept intermediates, fp32 norms, the backward's activation gradients at the peak); and the GiB a full
# recompute run needs beyond its stored layer inputs (the recomputed layer, its gradients)
ACT_SCALE = {'megatron': 1.66, 'torchtitan': 1.31, 'fsdp': 1.74, 'deepspeed': 1.84, 'hf_trainer': 2.34, 'lightning': 2.5}
RECOMPUTE_EXTRA = {'megatron': 4.8, 'fsdp': 8.6, 'torchtitan': 3.1, 'hf_trainer': 7.3, 'deepspeed': 4.0}
# allocator slack beyond the live tensors (same fit, allocator_slack_gib)
SLACK = {'megatron': 3.0, 'megatron_tp4': 4.0, 'megatron_moe': 14.0, 'fsdp': 10.2, 'deepspeed_0': 9.0, 'deepspeed_1': 9.0,
         'deepspeed_2': 17.1, 'deepspeed_3': 20.5, 'hf_trainer': 5.0, 'torchtitan': 6.0, 'lightning': 6.0,
         'megatron_fsdp': 22.5}     # Megatron custom FSDP: its buffers (mechanisms.yaml megatron_fsdp, H20 M63)
FSDP_ALWAYS = ('fsdp', 'torchtitan')                   # frameworks that shard with FSDP whatever the states say
# the mechanism states estimate() reads (an unknown one would silently become its default); a registry mechanism and
# the harvest's x- name of the same thing (the registry's wins)
READS = ('tp', 'pp', 'cp', 'ep', 'etp', 'x-etp', 'micro_batch', 'precision', 'recompute', 'distributed_optimizer',
         'bf16_main_grads', 'bf16_moments', 'sequence_parallel', 'zero_stage', 'cpu_offload', 'fsdp_sharding',
         'fsdp_version', 'x-fsdp-version', 'hsdp', 'x-hsdp', 'reshard_after_forward', 'x-reshard-after-forward', 'qlora',
         'x-qlora', 'x-hf-backend', 'x-gc', 'loss_impl', 'x-loss-impl', 'fused_cross_entropy', 'nvls', 'megatron_fsdp',
         'x-megatron-fsdp')


def _int(v, default=1) -> int:
    try:
        return max(1, int(str(v)))
    except (TypeError, ValueError):
        return default


def _zero(st: dict) -> int:
    z = str(st.get('zero_stage', '0'))
    return int(z) if z.isdigit() else 0


def _on(v) -> bool:
    return str(v).lower() in ('on', 'true', '1', 'yes')


def _get(st: dict, *names):
    """The first state given among a mechanism's names (the registry's, then the harvest's x- alias)."""
    return next((st[n] for n in names if st.get(n) not in (None, '')), None)


def split(m: dict, ep: int = 1) -> tuple[float, float, float | None]:
    """(dense, expert, active) parameters of a decoder: the routed experts' MLPs from the shape when the expert count is
    known (active None without the router's top-k); else, for a job with expert parallelism or a size far beyond its
    dense shape, what the dense shape does not explain (active None: the routing is unknown)."""
    n = float(m['params_b']) * 1e9
    e = _int(m.get('experts'), 0) if m.get('experts') else 0
    if e and m.get('ffn'):
        exp = min(m['layers'] * e * 3 * m['hidden'] * m['ffn'], 0.98 * n)
        return n - exp, exp, (n - exp + exp * _int(m['topk']) / e) if m.get('topk') else None
    h, heads = m['hidden'], _int(m.get('heads'), 0) or max(1, m['hidden'] // 128)
    hd = _int(m.get('head_dim'), 0) or h // heads
    shape = m['layers'] * (2 * h * heads * hd + 2 * h * _int(m.get('kv_heads'), heads) * hd + 3 * h * (m.get('ffn') or 4 * h)) \
        + (m.get('vocab') or 0) * h * 2
    if (ep > 1 or e) and n > 1.3 * shape or n > 2.0 * shape:
        return shape, n - shape, None
    return n, 0.0, n


def estimate(fp: dict, states: dict) -> dict | None:
    """Per-GPU memory and ideal update time of the job `fp` (an experience fingerprint: model shape, workload, GPUs,
    framework) at mechanism `states` (tp, pp, cp, ep, micro_batch, precision, recompute, distributed_optimizer,
    bf16_main_grads, bf16_moments, sequence_parallel, zero_stage, cpu_offload, ...); None without the shape."""
    m, wl = fp.get('model') or {}, fp.get('workload') or {}
    if not (m.get('params_b') and m.get('layers') and m.get('hidden') and wl.get('seq_len')):
        return None
    st = {k: str(v) for k, v in (states or {}).items()}
    if st.get('precision') == 'fp32':                  # activations and the throughput basis are bf16 / fp8 here
        return None
    fw = fp.get('framework') or 'pytorch'
    tp, pp, cp, ep = (_int(st.get(k)) for k in ('tp', 'pp', 'cp', 'ep'))
    world = _int(fp.get('gpus_per_node')) * _int(fp.get('nodes'))
    dp = max(1, world // (tp * pp * cp))
    mbs = _int(st.get('micro_batch'))
    seq, h, layers = int(wl['seq_len']), int(m['hidden']), int(m['layers'])
    heads = _int(m.get('heads'), 0) or max(1, h // 128)
    kv = _int(m.get('kv_heads'), heads)
    ffn = int(m.get('ffn') or 4 * h)
    vocab = int(m.get('vocab') or 0)
    dense, expert, active = split(m, ep)
    tune = ' '.join(str(x) for x in (wl.get('trainable'), wl.get('x-finetune')) if x).lower()
    qlora = _on(_get(st, 'qlora', 'x-qlora')) or 'qlora' in tune
    lora = qlora or 'lora' in tune                      # the frozen base: no gradients, no optimizer states
    moe = expert > 0
    # Megatron's expert tensor parallelism (default: TP) splits each expert; the other stacks keep an expert whole
    etp = (_int(_get(st, 'etp', 'x-etp')) if _get(st, 'etp', 'x-etp') else tp) if fw == 'megatron' else 1
    w = dense / (tp * pp) + expert / (etp * ep * pp)    # parameters held by one GPU before any data-parallel sharding
    notes = []
    backend = st.get('x-hf-backend')                     # HF Trainer's own sharding backend
    sharding = None if _get(st, 'fsdp_sharding') in (None, 'none') else st['fsdp_sharding']
    fsdp = backend == 'fsdp' or sharding is not None or _get(st, 'fsdp_version', 'x-fsdp-version') is not None
    zero = st.get('zero_stage') not in (None, 'none')   # Accelerate / Lightning / HF: the states name the backend
    mfsdp = fw == 'megatron' and _get(st, 'megatron_fsdp', 'x-megatron-fsdp') not in (None, 'none')
    style = ('megatron' if fw == 'megatron' else 'fsdp' if fw in FSDP_ALWAYS else
             'deepspeed' if fw == 'deepspeed' or backend == 'deepspeed' or zero and not fsdp else 'fsdp' if fsdp else 'ddp')

    # parameters, gradients, optimizer states (bytes)
    offload = st.get('cpu_offload') not in (None, 'none', 'off')
    opt = 0.0 if offload else 8.0                       # Adam's two fp32 moments
    params_off = st.get('cpu_offload') in ('params', 'all')   # FSDP / ZeRO-3 parameter offload: no sharded states here
    if style == 'megatron':
        grads = 2.0 if st.get('bf16_main_grads') == 'bf16' else 4.0
        mopt = 0.0 if offload else 4.0 + (4.0 if st.get('bf16_moments') == 'bf16' else 8.0)
        dist = _on(st.get('distributed_optimizer'))
        wd, we = dense / (tp * pp), expert / (etp * ep * pp)   # experts' states shard over the expert-data-parallel ranks
        edp = max(1, world // (etp * ep * pp))
        if mfsdp:                                       # Megatron FSDP (optim_grads_params): every state sharded over
            per = 2.0 + grads + mopt                    # the data-parallel ranks, two layers' gathered params in flight
            states_b = wd * per / (dp * cp) + we * per / edp + 2 * 2.0 * w / layers
        else:
            states_b = w * (2.0 + grads) + wd * mopt / (dp * cp if dist else 1) + we * mopt / (edp if dist else 1)
        if st.get('precision', 'bf16').startswith('fp8'):
            states_b += w * 1.4                          # TE's FP8 weight copies (h20-R11)
    elif style == 'deepspeed':
        z = _zero(st)
        if lora:
            states_b = w * (2.0 / dp if z == 3 else 2.0)
        elif offload:                                   # ZeRO-Offload: fp32 master weights, moments and the gradient
            # partition on the CPU; the GPU keeps the bf16 weights (a shard at stage 3, none with parameter offload);
            # the reduce and gather buffers are in the slack
            states_b = w * ((0.0 if params_off else 2.0 / dp) if z == 3 else 2.0)
        else:                                           # weights, grads, fp32 master + moments: what each stage shards
            states_b = w * {0: 16.0, 1: 4.0 + 12.0 / dp, 2: 2.0 + 14.0 / dp}.get(z, 16.0 / dp)
    elif style == 'fsdp':                               # FSDP shards over every rank of a model replica (CP included)
        shard = max(1, world // (tp * pp))
        hs = _get(st, 'hsdp', 'x-hsdp')
        if hs and hs != 'none' and 'x' in hs:
            shard = _int(hs.split('x')[-1], shard)     # replicate x shard
        elif sharding == 'hybrid_shard':               # shards within a node, replicas across nodes
            shard = min(shard, max(1, _int(fp.get('gpus_per_node')) // tp))
        reshard = _get(st, 'reshard_after_forward', 'x-reshard-after-forward')
        keep = 2.0 * w if reshard == 'off' or sharding == 'shard_grad_op' else 0.0   # bf16 params kept fwd -> bwd
        layer = 2 * 2.0 * w / layers                                         # two layers' gathered params in flight
        per = 0.0 if params_off else (0.5 if qlora else 2.0) if lora else 4.0 + 4.0 + opt
        wd, we = dense / (tp * pp), expert / (ep * pp)   # experts: EP-sharded, then FSDP over the ranks left per expert
        states_b = wd * per / shard + we * per / max(1, shard // ep) + layer + keep
    else:                                               # DDP: fp32 master weights, fp32 grads, Adam
        states_b = w * ((0.5 if qlora else 2.0) if lora else 4.0 + 4.0 + opt)
    if lora:
        notes.append('LoRA: the frozen base weights only (adapters and their optimizer ignored)')

    # activations (bytes): Llama-style decoder, flash attention, bf16; Korthikanti et al. with a gated MLP
    s, b = seq // cp, mbs
    sp = _on(st.get('sequence_parallel')) or fw in ('torchtitan',) and tp > 1
    ffn_tok = (_int(m.get('topk'), 1) if moe else 1) * ffn
    rest = s * b * (h * (2 + 2 * kv / heads) + 2 * h + 6 * ffn_tok)    # Q K V O, MLP in/outs: split by TP
    shared = s * b * 8 * h                                              # norms' and GEMMs' inputs: split by SP only
    per_layer = rest / tp + shared / (tp if sp else 1)
    rc = st.get('recompute', 'none')
    if lora and rc == 'none' and st.get('x-gc') == 'on':
        rc = 'full'
    scale = ACT_SCALE.get(style if style in ('fsdp', 'deepspeed') and fw not in ACT_SCALE else fw, 1.7)
    if rc == 'full':
        act = s * b * 2 * h / (tp if sp else 1) * layers / pp + per_layer * scale \
            + RECOMPUTE_EXTRA.get(fw, 5.0) * GIB
    else:
        act = per_layer * scale * (0.85 if rc == 'selective' else 1.0) * layers / pp
    gbs = int(wl.get('global_batch_tokens') or 0) // seq if wl.get('global_batch_tokens') else dp * b
    inflight = min(pp, max(1, gbs // max(1, dp * b))) if pp > 1 else 1
    act *= inflight
    # logits: bf16 + an fp32 copy for the loss; a fused or chunked loss (Megatron's, Liger, chunked CE) keeps ~1/3 of it
    fused_loss = fw == 'megatron' and _on(st.get('fused_cross_entropy')) or \
        _get(st, 'loss_impl', 'x-loss-impl') in ('liger', 'chunked')
    logits = s * b * vocab / tp * (2 if fused_loss else 6) if vocab else 0.0
    torch_b = states_b + act + logits

    # outside the allocator and slack (GiB), per stack
    if fw == 'megatron':
        key = (('megatron_moe_nvls_off' if st.get('nvls') == 'off' else 'megatron_moe') if moe else
               'megatron_fp8' if st.get('precision', '').startswith('fp8') else
               'megatron_cp' if cp > 1 else 'megatron_tp8' if tp >= 8 else 'megatron_tp4' if tp >= 4 else
               'megatron_nvls_off' if st.get('nvls') == 'off' else 'megatron')
        non_torch = NON_TORCH[key]
        slack = SLACK['megatron_fsdp' if mfsdp else 'megatron_moe' if moe else 'megatron_tp4' if tp >= 4 else 'megatron']
    elif fw == 'deepspeed':
        non_torch = NON_TORCH['deepspeed']
        slack = SLACK.get(f'deepspeed_{_zero(st)}', 9.0)
    else:
        non_torch = NON_TORCH.get(fw, 1.5)
        slack = SLACK.get(fw, 5.0)
    mem = torch_b / GIB + non_torch + slack
    g = gpus().get(fp.get('gpu')) or {}
    cap = fp.get('memory_gib') or g.get('memory_gib')

    # model FLOPs per update: 6 N_active + 6 L h s per token (causal attention: flash attention skips the masked half of
    # PaLM's 12 L h s; on the harvests the non-causal count put long-sequence MFU above 1)
    tokens = int(wl['global_batch_tokens']) if wl.get('global_batch_tokens') else None
    flops = (6 * active + 6 * layers * h * seq) * tokens if active is not None and tokens else None
    if active is None:
        notes.append('MoE with an unknown expert count / top-k (model.experts, model.topk): no FLOPs, no MFU')
    elif not tokens:
        notes.append('tokens per update unknown (workload.global_batch_tokens): no FLOPs, no MFU')
    fp8 = st.get('precision', '').startswith('fp8') or st.get('precision') == 'float8'
    tf, basis = (g.get('fp8_tflops_measured'), 'fp8 measured') if fp8 else (None, None)
    if not tf:
        tf, basis = g.get('bf16_tflops_measured'), 'bf16 measured'
        if fp8:
            notes.append('FP8 run measured against bf16 throughput (no FP8 figure for this GPU): MFU can exceed 1')
    if not tf:
        tf, basis = g.get('bf16_tflops_peak'), 'bf16 peak'
    ideal = flops / (world * tf * 1e12) if tf and flops else None

    # the data-parallel gradient sync of one update at the job's measured all-reduce bus bandwidth (multi-node only, a
    # lower bound before overlap): reduce-scatter of the grads + all-gather of the bf16 weights (sharded optimizers:
    # Megatron's distributed one, ZeRO >= 1; FSDP gathers the weights twice), else an all-reduce of the grads
    bw = (fp.get('interconnect') or {}).get('bandwidth_gbs') if _int(fp.get('nodes')) > 1 else None
    sync = None
    if isinstance(bw, (int, float)) and bw > 0 and dp > 1 and not lora:
        gb = 2.0 if st.get('bf16_main_grads') == 'bf16' or style in ('fsdp', 'deepspeed') else 4.0
        frac = (dp - 1) / dp
        hybrid = sharding == 'hybrid_shard' or (_get(st, 'hsdp', 'x-hsdp') or 'none') != 'none'
        if hybrid:                                   # shard and replica groups apart: not modelled, no number
            wire = None
        elif style == 'fsdp' or mfsdp:              # weights gathered for forward, again for backward unless kept
            kept = _get(st, 'reshard_after_forward', 'x-reshard-after-forward') == 'off' or sharding == 'shard_grad_op'
            wire = w * ((1 if kept else 2) * 2.0 + gb) * frac
        elif style == 'megatron' and _on(st.get('distributed_optimizer')) or style == 'deepspeed' and _zero(st) >= 1:
            wire = w * (gb + 2.0) * frac
        else:
            wire = w * gb * 2 * frac
        sync = wire / (bw * 1e9) if wire else None
    return {'mem_gib': round(mem, 2), 'mem_torch_gib': round(torch_b / GIB, 2),
            'mem_terms': {'states_gib': round(states_b / GIB, 2), 'activations_gib': round(act / GIB, 2),
                          'logits_gib': round(logits / GIB, 2), 'non_torch_gib': non_torch, 'slack_gib': slack},
            'capacity_gib': cap, 'fit_margin_gib': round(cap - mem, 2) if cap else None,
            'params_local_b': round(w / 1e9, 3), 'dp': dp, 'flops_per_update': flops,
            'ideal_step_s': round(ideal, 4) if ideal else None, 'tflops_basis': basis if tf else None,
            'dp_sync_s': round(sync, 4) if sync else None, 'bandwidth_gbs': bw if sync else None, 'notes': notes}


def mfu(est: dict | None, step_time_s: float | None) -> float | None:
    """Model FLOPs utilisation of a measured update against the GPUs' (measured, else nominal) bf16 throughput."""
    if not est or not est.get('ideal_step_s') or not step_time_s:
        return None
    return round(est['ideal_step_s'] / step_time_s, 4)
