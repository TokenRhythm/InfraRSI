"""Dev check: bridge build + HF weight load + one forward pass (no optimizer) for models too large to train on
one node (e.g. Qwen3-Next-80B on 8 GPUs). Prints FORWARD_CHECK with the supervised CE of the first plan pack.

  torchrun --nproc_per_node 8 tests/gpu/train/forward_check.py --request <request.json>
(PYTHONPATH as in launch_dev.sh). Not part of the frozen package.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from infrarsi.adapters.megatron_core import config_map, model_build as mb
from infrarsi.adapters.megatron_core.data import DataSource, local_arrays
from infrarsi.contracts import RunRequest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request', required=True)
    p.add_argument('--tokens', type=int, default=0, help='truncate the pack to this many tokens (0 = full pack)')
    a = p.parse_args()
    req = RunRequest.from_dict(json.loads(Path(a.request).read_text(encoding='utf-8')))
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30))
    import mcore_bridge  # noqa: F401
    from megatron.core import parallel_state as ps
    from megatron.core.packed_seq_params import PackedSeqParams
    model_dir = mb.model_dir_for(req.task)
    facts = config_map.model_facts(model_dir)
    plan = config_map.make_plan(req, facts)
    mb.initialize_parallel(plan)
    os.environ.setdefault('USE_MCORE_GDN', '1')
    t0 = time.perf_counter()
    cfg, derived = mb.model_config(plan, model_dir)
    models = mb.build_models(cfg)
    t1 = time.perf_counter()
    mb.load_hf_weights(cfg, models, model_dir)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    source = DataSource(req.task, req.input_plan, facts['vocab_size'], data_manifest=req.data_manifest)
    arrays = local_arrays([source.pack(int(source.order[0]))], plan.cp, ps.get_context_parallel_rank())
    cu = torch.from_numpy(arrays['cu_seqlens']).cuda()
    seg = int((cu[1:] - cu[:-1]).max())
    psp = PackedSeqParams(qkv_format='thd', cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=seg, max_seqlen_kv=seg)
    batch = {k: torch.from_numpy(v).cuda() for k, v in arrays.items() if k != 'cu_seqlens'}
    for m in models:
        m.eval()
    with torch.no_grad():
        loss = models[0](input_ids=batch['tokens'], position_ids=batch['position_ids'], attention_mask=None,
                         labels=batch['labels'], packed_seq_params=psp)
        mask = batch['loss_mask'].view(-1).float()
        ce = float((loss.float().view(-1) * mask).sum() / mask.sum())
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    params = sum(p.numel() for m in models for p in m.parameters())
    counts = [None] * dist.get_world_size()
    dist.all_gather_object(counts, params)
    if rank == 0:
        print('FORWARD_CHECK', json.dumps({
            'model': req.task['model']['repo_id'], 'plan': plan.summary(), 'ce': ce, 'finite': ce == ce,
            'supervised_tokens': int(mask.sum()), 'build_s': t1 - t0, 'load_s': t2 - t1, 'forward_s': t3 - t2,
            'peak_alloc_gib': torch.cuda.max_memory_allocated() / 2**30, 'local_params_rank_sum': sum(counts),
            'experimental_attention_variant': str(cfg.experimental_attention_variant),
            'num_moe_experts': cfg.num_moe_experts, 'moe_router_topk': cfg.moe_router_topk,
            'bridge_derived': {k: str(v) for k, v in derived.items()}}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
