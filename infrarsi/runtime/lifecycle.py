"""Lifecycle checks: train -> save -> NEW process resume -> continuity -> HF export -> forward compare.

Checkpoints use Megatron-Core dist_checkpointing exactly as Megatron-LM's torch_dist format does
(model + DistributedOptimizer 'dp_reshardable' state + scheduler + update index). The HF export goes through
the bridge's ``save_weights``; the forward comparison loads the exported directory with transformers in a
separate process and compares full-vocabulary logits of a fixed token window against the Megatron model.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import shutil
import struct
from pathlib import Path

HF_AUX_FILES = ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json',
                'merges.txt', 'tokenizer.model', 'special_tokens_map.json', 'added_tokens.json', 'chat_template.jinja',
                'configuration.json')
# bf16 kernel differences (TE vs. transformers sdpa) flip argmax on near-ties (~3% of positions observed on
# Qwen3-1.7B); rel_l2 carries the equivalence, a broken export gives rel_l2 ~ 1 and top-1 near chance.
DEFAULT_FORWARD_TOLERANCE = {'rel_l2': 0.05, 'top1_agreement': 0.90}
FIXED_BATCH_TOKENS = 256


# ---------------------------------------------------------------- checkpoint (torch)

@contextlib.contextmanager
def synced_params(ddp_models):
    """Full model parameters for save/export/eval right after an optimizer step.

    With the distributed optimizer and overlap_param_gather, each rank only holds its updated shard until the
    next forward's pre-hooks gather the rest; like Megatron-LM's save path, drop the hooks with a synchronous
    all-gather and restore them afterwards. A no-op for every other configuration.
    """
    hooked = [m for m in ddp_models if getattr(m, 'use_forward_hook', False)]
    for m in hooked:
        m.disable_forward_pre_hook(param_sync=True)
    try:
        yield
    finally:
        for m in hooked:
            m.enable_forward_pre_hook()


def _metadata():
    from megatron.core import parallel_state as ps
    return {'distrib_optim_sharding_type': 'dp_reshardable', 'singleton_local_shards': False,
            'chained_optim_avoid_prefix': True, 'dp_cp_group': ps.get_data_parallel_group(with_context_parallel=True)}


def _sharded_state_dict(ddp_models, optimizer, scheduler, step: int, is_loading: bool):
    meta = _metadata()
    sd = {}
    for i, model in enumerate(ddp_models):       # unwrapped as in Megatron-LM checkpointing (no 'module.' prefix)
        sd['model' if len(ddp_models) == 1 else f'model{i}'] = getattr(model, 'module', model).sharded_state_dict(
            metadata=meta)
    sd['optimizer'] = optimizer.sharded_state_dict(sd, is_loading=is_loading, metadata=meta)
    sd['opt_param_scheduler'] = scheduler.state_dict()
    sd['iteration'] = step
    return sd, {k: v for k, v in meta.items() if k != 'dp_cp_group'}


def _step_slots(optimizer):
    """Every optimizer step counter in canonical order: ('g', group dict) for TE FusedAdam (per param group) and
    ('s', state dict) for torch optimizers (per parameter, e.g. the CPU side of the hybrid offload optimizer)."""
    from infrarsi.runtime.gates import leaf_optimizers
    slots = []
    for leaf in leaf_optimizers(optimizer):
        inner = getattr(leaf, 'optimizer', None)
        for opt in getattr(inner, 'sub_optimizers', None) or ([inner] if inner is not None else []):
            state = getattr(opt, 'state', {})
            for group in getattr(opt, 'param_groups', None) or []:
                slots.append(('g', group))
                for p in group['params']:
                    st = state.get(p) or {}
                    if 'step' in st or 'exp_avg' in st:      # Adam state with or without its counter
                        slots.append(('s', st))
    return slots


def step_counter_problems(optimizer, applied: int) -> list[str]:
    """Counters that must exist after ``applied`` successful steps and equal it: TE FusedAdam keeps one per
    non-empty param group (none for an empty group), torch Adam / AdamW one per parameter state (every shard
    parameter has a gradient on the distributed-optimizer path, so every one has a state)."""
    import torch
    from infrarsi.runtime.gates import leaf_optimizers
    if applied < 1:
        return []
    bad = []
    for li, leaf in enumerate(leaf_optimizers(optimizer)):
        if getattr(leaf, 'is_stub_optimizer', False) is True:
            continue
        inner = getattr(leaf, 'optimizer', None)
        for si, opt in enumerate(getattr(inner, 'sub_optimizers', None) or ([inner] if inner is not None else [])):
            per_group = type(opt).__name__ == 'FusedAdam'
            state = getattr(opt, 'state', {})
            for gi, group in enumerate(getattr(opt, 'param_groups', None) or []):
                values = [group.get('step')] if per_group and group['params'] else \
                    [] if per_group else [(state.get(p) or {}).get('step') for p in group['params']]
                for v in values:
                    got = None if v is None else float(v.item() if torch.is_tensor(v) else v)
                    if got != float(applied):
                        bad.append(f'leaf{li}.opt{si}.group{gi}: step {got} != {applied}')
                        break
    return bad


def optimizer_steps(optimizer) -> list:
    """JSON view of every step counter (None where a group has never stepped, e.g. empty on this rank)."""
    import torch
    out = []
    for kind, d in _step_slots(optimizer):
        v = d.get('step')
        out.append([kind, None if v is None else float(v.item() if torch.is_tensor(v) else v)])
    return out


def restore_optimizer_steps(optimizer, saved: list) -> int:
    """Put this rank's saved step counters back after the Megatron load; returns how many changed."""
    import torch
    slots = _step_slots(optimizer)
    if [k for k, _ in slots] != [k for k, _ in saved]:
        raise RuntimeError('optimizer step layout differs from the checkpoint sidecar')
    changed = 0
    for (kind, d), (_, value) in zip(slots, saved):
        cur = d.get('step')
        cur_f = None if cur is None else float(cur.item() if torch.is_tensor(cur) else cur)
        if cur_f == value:
            continue
        changed += 1
        if value is None:
            d.pop('step', None)
        elif torch.is_tensor(cur):
            cur.fill_(value)
        elif kind == 's':
            d['step'] = torch.tensor(value, dtype=torch.float32)
        else:
            d['step'] = int(value) if float(value).is_integer() else value
    return changed


def save_checkpoint(ckpt_dir: str | Path, ddp_models, optimizer, scheduler, step: int) -> dict:
    import torch.distributed as dist
    from megatron.core import dist_checkpointing
    ckpt_dir = Path(ckpt_dir)
    if dist.get_rank() == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    sd, content = _sharded_state_dict(ddp_models, optimizer, scheduler, step, is_loading=False)
    dist_checkpointing.save(sd, str(ckpt_dir), content_metadata=content)
    sidecar = ckpt_dir / 'infrarsi_steps' / f'rank{dist.get_rank()}.json'
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(optimizer_steps(optimizer)), encoding='utf-8')
    dist.barrier()
    files = sorted(p for p in ckpt_dir.rglob('*') if p.is_file())
    return {'dir': str(ckpt_dir), 'step': step, 'files': len(files), 'bytes': sum(p.stat().st_size for p in files)}


def load_checkpoint(ckpt_dir: str | Path, ddp_models, optimizer, scheduler) -> tuple[int, dict]:
    """Megatron dist-checkpoint load, then this rank's optimizer step counters from the sidecar."""
    import torch.distributed as dist
    from megatron.core import dist_checkpointing
    sd, _ = _sharded_state_dict(ddp_models, optimizer, scheduler, 0, is_loading=True)
    state = dist_checkpointing.load(sd, str(ckpt_dir))
    for i, model in enumerate(ddp_models):
        model.load_state_dict(state['model' if len(ddp_models) == 1 else f'model{i}'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['opt_param_scheduler'])
    saved = json.loads((Path(ckpt_dir) / 'infrarsi_steps' / f'rank{dist.get_rank()}.json').read_text(encoding='utf-8'))
    before = optimizer_steps(optimizer)
    changed = restore_optimizer_steps(optimizer, saved)
    return int(state['iteration']), {'steps_restored': changed, 'steps_after_megatron_load': before,
                                     'steps_saved': saved}


_CHUNK = 1 << 28        # bytes per device-to-host copy


def tensor_checksum(t) -> str:
    """sha256 of the tensor's exact bytes in element order (dtype and shape are recorded next to it).

    Copied to the host in bounded chunks; used only around checkpoint save/load, never inside a timed update.
    """
    import torch
    raw = t.detach().contiguous().view(-1).view(torch.uint8)
    h = hashlib.sha256()
    for lo in range(0, raw.numel(), _CHUNK):
        h.update(raw[lo:lo + _CHUNK].cpu().numpy())
    return h.hexdigest()


def state_digest(ddp_models, optimizer) -> dict:
    """Per-rank digest of params, fp32 masters, Adam moments and Adam steps, by category and traversal order."""
    import torch
    from infrarsi.contracts import digest
    from infrarsi.runtime.gates import leaf_optimizers

    cats = {'params': [], 'masters': [], 'exp_avg': [], 'exp_avg_sq': [], 'steps': []}

    def add(cat, t):
        cats[cat].append([str(t.dtype), list(t.shape), tensor_checksum(t)])

    for m in ddp_models:
        for p in m.parameters():
            add('params', p)
    for li, leaf in enumerate(leaf_optimizers(optimizer)):
        for group in list(getattr(leaf, 'shard_fp32_from_float16_groups', None) or []):
            for t in group:
                if t is not None:
                    add('masters', t)
        inner = getattr(leaf, 'optimizer', None)
        # Canonical order: param groups, then their params (dict order of optimizer.state changes on load).
        for si, opt in enumerate(getattr(inner, 'sub_optimizers', None) or ([inner] if inner is not None else [])):
            state = getattr(opt, 'state', {})
            for gi, group in enumerate(getattr(opt, 'param_groups', None) or []):
                if 'step' in group:
                    cats['steps'].append([li, si, gi, float(group['step'])])
                for p in group['params']:
                    st = state.get(p) or {}
                    for key, cat in (('exp_avg', 'exp_avg'), ('exp_avg_sq', 'exp_avg_sq'), ('master_param', 'masters')):
                        if torch.is_tensor(st.get(key)):
                            add(cat, st[key])
                    if 'step' in st:
                        cats['steps'].append([li, si, gi, float(st['step'])])
    return {k: {'n': len(v), 'digest': digest(v)} for k, v in cats.items()}


# ---------------------------------------------------------------- continuity (pure)

# AMENDMENT-001 (design/AMENDMENT_001_NUMERICS_20261020.md); task specs carry it as numerics.lifecycle_tolerance.
# AMENDMENT-002: the spec carries its numerics tier's values (Gated DeltaNet tasks: first resumed update CE 1e-3 /
# grad norm 0.03, deterministic 'unavailable'); the driver's harness rejects a spec that differs from its tier's
# frozen table, so reading the spec here reads the tier.
LIFECYCLE_TOLERANCE = {'state_digest_exact': True,
                       'first_resumed_update': {'main_ce_rel_max': 1.0e-4, 'grad_norm_rel_max': 5.0e-3},
                       'later_updates': {'main_ce': 'task_per_step_tolerance', 'grad_norm': 'report_only',
                                         'lr_tokens_clip_calls': 'exact'},
                       'forward': {'rel_l2_max': 0.05, 'top1_agreement_min': 0.90}}
# request.lifecycle.deterministic (AMENDMENT-002: the driver sets it on every lifecycle): both lifecycle training
# processes use deterministic kernels, so an exact restore reproduces the uninterrupted run bit for bit. Without them
# identical runs differ by up to ~15 % in a single update's grad norm (atomics in attention / embedding / MoE
# backward, amplified by Adam and clipping). NVTE_*: deterministic TE attention backward; CUBLAS_*: required by
# torch.use_deterministic_algorithms; NCCL_ALGO=Ring: no NVLS in-switch reductions. The trainer also sets
# TransformerConfig.deterministic_mode and flash attention (except MLA). Gated DeltaNet has no deterministic path.
DETERMINISTIC_ENV = {'NVTE_ALLOW_NONDETERMINISTIC_ALGO': '0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                     'NCCL_ALGO': 'Ring'}


def deterministic_overrides(effective: dict | None, env_set: dict | None, attention_ran) -> dict:
    """{setting: {'requested', 'ran'}} for what a deterministic lifecycle replaced (AMENDMENT-002 receipt field
    lifecycle.deterministic_overrides): the receipt's effective_config keeps reporting the requested layout, this
    records what actually ran. effective: the plan's L0 view; env_set: the env the request asked for (whitelisted
    request values plus FR defaults, before the deterministic overrides); attention_ran: the backend the model built."""
    out = {'torch.use_deterministic_algorithms': {'requested': False, 'ran': True},
           'TransformerConfig.deterministic_mode': {'requested': False, 'ran': True}}
    want = (effective or {}).get('attention_backend')
    if attention_ran is not None and str(want) != str(attention_ran):
        out['attention_backend'] = {'requested': want, 'ran': str(attention_ran)}
    for k, v in DETERMINISTIC_ENV.items():
        req = (env_set or {}).get(k)
        if req is None or str(req) != v:
            out[f'env.{k}'] = {'requested': req, 'ran': v}
    return out


def lifecycle_tolerance(task: dict) -> dict:
    tol = dict(LIFECYCLE_TOLERANCE)
    tol.update((task.get('numerics') or {}).get('lifecycle_tolerance') or {})
    return tol


def forward_tolerance(task: dict, life: dict | None) -> dict:
    """compare_logits tolerance: the request's lifecycle.forward_tolerance, else the task's lifecycle_tolerance."""
    if (life or {}).get('forward_tolerance'):
        return dict(life['forward_tolerance'])
    f = lifecycle_tolerance(task).get('forward') or {}
    return {'rel_l2': f.get('rel_l2_max', DEFAULT_FORWARD_TOLERANCE['rel_l2']),
            'top1_agreement': f.get('top1_agreement_min', DEFAULT_FORWARD_TOLERANCE['top1_agreement'])}

def _rel(a, b) -> float:
    if a is None or b is None:
        return math.inf
    a, b = float(a), float(b)
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf
    return abs(a - b) / max(abs(b), 1e-12)


def compare_continuation(reference: list[dict], resumed: list[dict], tolerance: dict,
                         life_tolerance: dict | None = None) -> dict:
    """Resumed rank-0 rows vs the uninterrupted rows of the same updates (AMENDMENT-001).

    First resumed update: CE and grad norm within ``first_resumed_update``. Later updates: CE within the task's
    per-step tolerance; LR, supervised tokens and clip calls exact; grad norm reported only (identical runs differ
    by up to ~15 % in single-update grad norms). ``bitwise`` when every compared scalar is identical;
    ``bitwise_updates`` counts the resumed updates that are (AMENDMENT-002 report, lifecycle.resumed_bitwise_updates).
    """
    lt = dict(LIFECYCLE_TOLERANCE, **(life_tolerance or {}))
    first_tol = dict(LIFECYCLE_TOLERANCE['first_resumed_update'], **(lt.get('first_resumed_update') or {}))
    ce_tol = float(tolerance.get('main_ce_rel_max_per_step', 0.005))
    ref = {int(r['step']): r for r in reference}
    rows, ok = [], bool(resumed)
    for r in resumed:
        s = int(r['step'])
        base = ref.get(s)
        if base is None:
            rows.append({'step': s, 'missing_reference': True})
            ok = False
            continue
        row = {'step': s, 'main_ce_rel': _rel(r.get('main_ce'), base.get('main_ce')),
               'grad_norm_rel': _rel(r.get('grad_norm'), base.get('grad_norm')),
               'lr_equal': r.get('lr') == base.get('lr'),
               'tokens_equal': r.get('supervised_tokens') == base.get('supervised_tokens'),
               'clip_calls_equal': r.get('clip_calls') == base.get('clip_calls'),
               'bitwise': all(r.get(k) == base.get(k) for k in ('main_ce', 'grad_norm', 'aux_loss', 'lr'))}
        first = s == min(int(x['step']) for x in resumed)
        if first:
            good = (row['main_ce_rel'] <= first_tol['main_ce_rel_max']
                    and row['grad_norm_rel'] <= first_tol['grad_norm_rel_max'])
        else:
            good = row['main_ce_rel'] <= ce_tol
        row['ok'] = good and row['lr_equal'] and row['tokens_equal'] and row['clip_calls_equal']
        ok = ok and row['ok']
        rows.append(row)
    return {'match': ok, 'bitwise': ok and all(r.get('bitwise') for r in rows), 'steps': rows,
            'bitwise_updates': sum(1 for r in rows if r.get('bitwise') is True), 'updates': len(rows),
            'max_main_ce_rel': max((r.get('main_ce_rel', math.inf) for r in rows), default=None),
            'max_grad_norm_rel': max((r.get('grad_norm_rel', math.inf) for r in rows), default=None)}


# ---------------------------------------------------------------- HF export

def safetensors_names(directory: str | Path) -> dict[str, list[int]]:
    """Tensor name -> shape from the safetensors headers of a directory (no tensor reads)."""
    out = {}
    for path in sorted(Path(directory).glob('*.safetensors')):
        with open(path, 'rb') as f:
            header = json.loads(f.read(struct.unpack('<Q', f.read(8))[0]))
        out.update({k: v['shape'] for k, v in header.items() if k != '__metadata__'})
    return out


def export_hf(cfg, models, source_dir: str | Path, export_dir: str | Path, rank: int, tied: bool) -> dict | None:
    """Bridge save_weights (collective; rank 0 writes) + tokenizer/config copies + key/shape check."""
    import torch.distributed as dist
    export_dir = Path(export_dir)
    cfg.bridge.save_weights(models, str(export_dir))
    dist.barrier()
    if rank != 0:
        return None
    for name in HF_AUX_FILES:
        if (Path(source_dir) / name).exists():
            shutil.copy2(Path(source_dir) / name, export_dir / name)
    src, dst = safetensors_names(source_dir), safetensors_names(export_dir)
    missing = sorted(set(src) - set(dst) - ({'lm_head.weight'} if tied else set()))
    extra = sorted(set(dst) - set(src))
    shape_mismatch = sorted(k for k in set(src) & set(dst) if list(src[k]) != list(dst[k]))
    files = sorted(p for p in export_dir.iterdir() if p.is_file())
    return {'dir': str(export_dir), 'tensors': len(dst), 'source_tensors': len(src), 'missing_keys': missing[:50],
            'extra_keys': extra[:50], 'shape_mismatch': shape_mismatch[:50], 'files': len(files),
            'bytes': sum(p.stat().st_size for p in files),
            'export_ok': not missing and not extra and not shape_mismatch and len(dst) > 0}


# ---------------------------------------------------------------- forward comparison

def mcore_logits(ddp_models, plan, tokens, forward_backward_func):
    """Full-vocab fp32 logits [L, V] of one segment, returned on (last stage, tp0, cp0, dp0); None elsewhere."""
    import numpy as np
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.packed_seq_params import PackedSeqParams
    from infrarsi.adapters.megatron_core.data import zigzag_index
    n = len(tokens)
    cu = np.asarray([0, n], dtype=np.int32)
    arrays = {'tokens': np.asarray(tokens, dtype=np.int64), 'position_ids': np.arange(n, dtype=np.int64)}
    cp_rank = ps.get_context_parallel_rank()
    idx = zigzag_index(cu, plan.cp, cp_rank) if plan.cp > 1 else None
    if idx is not None:
        arrays = {k: v[idx] for k, v in arrays.items()}
    batch = {k: torch.from_numpy(np.ascontiguousarray(v[None])).cuda() for k, v in arrays.items()}
    cu_t = torch.from_numpy(cu).cuda()
    padded = cu_t if plan.cp > 1 else None
    psp = PackedSeqParams(qkv_format='thd', cu_seqlens_q=cu_t, cu_seqlens_kv=cu_t, max_seqlen_q=n, max_seqlen_kv=n,
                          cu_seqlens_q_padded=padded, cu_seqlens_kv_padded=padded)

    def step(it, model):
        b = next(it)
        out = model(input_ids=b['tokens'], position_ids=b['position_ids'], attention_mask=None,
                    packed_seq_params=psp, runtime_gather_output=True)
        return out, lambda o, non_loss_data=False: o.detach().float()

    for m in ddp_models:
        m.eval()
    with torch.no_grad():
        store = forward_backward_func(forward_step_func=step, data_iterator=iter([batch]), model=ddp_models,
                                      num_microbatches=1, seq_length=n, micro_batch_size=1, forward_only=True,
                                      collect_non_loss_data=True)
    for m in ddp_models:
        m.train()
    if not ps.is_pipeline_last_stage():
        return None
    logits = store[0][0]                           # [L / CP, V]
    if plan.cp > 1:
        parts = [torch.empty_like(logits) for _ in range(plan.cp)]
        dist.all_gather(parts, logits.contiguous(), group=ps.get_context_parallel_group())
        full = torch.empty((n, logits.shape[-1]), dtype=logits.dtype, device=logits.device)
        for r, part in enumerate(parts):
            full[torch.from_numpy(zigzag_index(cu, plan.cp, r)).cuda()] = part
        logits = full
    owner = (ps.get_tensor_model_parallel_rank() == 0 and cp_rank == 0
             and ps.get_data_parallel_rank() == 0)
    return logits.cpu() if owner else None


def compare_logits(mcore, hf, tolerance: dict | None = None) -> dict:
    import torch
    tol = dict(DEFAULT_FORWARD_TOLERANCE, **(tolerance or {}))
    v = min(mcore.shape[-1], hf.shape[-1])
    a, b = mcore[..., :v].double(), hf[..., :v].double()
    diff = (a - b).abs()
    rel_l2 = float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b))
    top1 = float((a.argmax(-1) == b.argmax(-1)).double().mean())
    out = {'tokens': int(a.shape[0]), 'vocab_compared': v, 'vocab_mcore': int(mcore.shape[-1]),
           'vocab_hf': int(hf.shape[-1]), 'max_abs_logit_diff': float(diff.max()),
           'mean_abs_logit_diff': float(diff.mean()), 'rel_l2': rel_l2, 'top1_agreement': top1, 'tolerance': tol}
    out['forward_ok'] = rel_l2 <= tol['rel_l2'] and top1 >= tol['top1_agreement']
    return out


def hf_forward(export_dir: str | Path, tokens) -> 'torch.Tensor':
    """transformers forward of the exported checkpoint (bf16, all visible GPUs), fp32 logits [L, V] on CPU."""
    import torch
    from transformers import AutoModelForCausalLM
    multi = torch.cuda.device_count() > 1
    model = AutoModelForCausalLM.from_pretrained(str(export_dir), dtype=torch.bfloat16,
                                                 device_map='auto' if multi else 'cuda', attn_implementation='sdpa')
    model.eval()
    device = next(model.parameters()).device
    ids = torch.as_tensor(tokens, dtype=torch.long, device=device)[None]
    with torch.no_grad():
        out = model(input_ids=ids, position_ids=torch.arange(ids.shape[1], device=device)[None], use_cache=False)
    return out.logits[0].float().cpu()


def remove_artifacts(paths) -> list[str]:
    """Delete retention-managed artifacts (directories or single files); returns what was removed."""
    removed = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()
        else:
            continue
        removed.append(str(p))
    return removed
