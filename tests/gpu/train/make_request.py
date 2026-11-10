"""Build a dev RunRequest (FR defaults of design section 1.4) for T-train GPU checks.

Runs on the driver PC (needs the design task specs); upload the JSON to the printed remote path with put.py.
  python tests/gpu/train/make_request.py --task S1 --gpus 1 --steps 10 --synthetic --out req.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from infrarsi.contracts import RemotePaths, RunRequest, TaskSpec  # noqa: E402

NUM_EXPERTS = {'Qwen/Qwen3-30B-A3B': 128, 'deepseek-ai/DeepSeek-V2-Lite': 64,
               'Qwen/Qwen3-Next-80B-A3B-Instruct': 512}


def fr_config(task: dict, tp: int, ep: int | None = None) -> dict:
    experts = NUM_EXPERTS.get(task['model']['repo_id'])
    cfg = {'tensor_model_parallel_size': tp, 'pipeline_model_parallel_size': 1, 'context_parallel_size': 1,
           'sequence_parallel': tp > 1, 'micro_batch_size': 1, 'recompute_granularity': 'full',
           'recompute_method': 'uniform', 'recompute_num_layers': 1, 'overlap_grad_reduce': False,
           'overlap_param_gather': False, 'apply_rope_fusion': False, 'bias_activation_fusion': False,
           'masked_softmax_fusion': False, 'gradient_accumulation_fusion': False, 'cross_entropy_loss_fusion': False,
           'attention_backend': 'auto', 'env': {'CUDA_DEVICE_MAX_CONNECTIONS': '1'}}
    if experts:
        cfg.update(expert_model_parallel_size=ep or min(experts, 8), expert_tensor_parallel_size=1,
                   moe_token_dispatcher_type='alltoall', moe_grouped_gemm=True, moe_permute_fusion=False,
                   moe_enable_deepep=False)
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True)
    p.add_argument('--purpose', default='screen', choices=('probe', 'screen', 'score', 'lifecycle'))
    p.add_argument('--nodes', default='WINH1')
    p.add_argument('--gpus', type=int, required=True, help='GPUs per node')
    p.add_argument('--steps', type=int)
    p.add_argument('--window', type=int, nargs=2)
    p.add_argument('--tp', type=int, default=1)
    p.add_argument('--ep', type=int)
    p.add_argument('--set', default='{}', help='JSON dict of extra L0 keys')
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--plan', default='A')
    p.add_argument('--save-at', type=int)
    p.add_argument('--resume-steps', type=int)
    p.add_argument('--keep-artifacts', action='store_true')
    p.add_argument('--run-id')
    p.add_argument('--wall', type=int, default=1800)
    p.add_argument('--code-root', default='/data/infrarsi/code/dev_train')
    p.add_argument('--code-digest', default='', help="contracts.code_digest of the remote package dir (empty skips the check)")
    p.add_argument('--data-manifest', help='pin RunRequest.data_manifest (the INDEX.json manifest path), as the driver does')
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    task = copy.deepcopy(TaskSpec.load(a.task).raw)
    if a.synthetic:
        task['data']['synthetic'] = True
    cfg = fr_config(task, a.tp, a.ep)
    cfg.update(json.loads(a.set))
    window = tuple(a.window) if a.window else None
    if a.steps and not window and a.purpose != 'probe':
        lo, hi = task['workload']['scoring_window']
        window = (a.steps // 2 + 1, a.steps) if a.steps < hi else (lo, hi)
    life = None
    if a.purpose == 'lifecycle':
        life = {'save_at': a.save_at or a.steps or task['workload']['screen_steps'],
                'resume_steps': a.resume_steps or task['workload']['resume_continuation_steps'],
                'export': True, 'forward_compare': True, 'keep_artifacts': a.keep_artifacts}
    req = RunRequest(run_id=a.run_id or f'dev-T-train-{a.task}', purpose=a.purpose, task=task, config=cfg,
                     nodes=a.nodes.split(','), gpus_per_node=a.gpus, input_plan=a.plan, steps=a.steps, window=window,
                     lifecycle=life, code_root=a.code_root, code_digest=a.code_digest, wall_seconds_cap=a.wall,
                     data_manifest=a.data_manifest)
    Path(a.out).write_text(json.dumps(req.to_dict(), indent=1), encoding='utf-8')
    print(json.dumps({'out_dir': req.out_dir, 'request': f'{req.out_dir}/request.json',
                      'runs_root': RemotePaths.runs, 'world_size': req.world_size}))


if __name__ == '__main__':
    main()
