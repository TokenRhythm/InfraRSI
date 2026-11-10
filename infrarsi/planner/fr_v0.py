"""fr_v0: the deterministic first-runnable reference (design 1.4), no LLM, no estimate - probes decide.

PP=1, CP=1, MBS=1, GA=packs/DP, full uniform recompute (1 layer), smallest structurally valid power-of-two TP in
probe order 1->2->4->8 with SP=(TP>1); MoE: EP=min(experts, 8) inside a node, ETP=1, alltoall + grouped GEMM
(Megatron-LM MoE README quick start, recorded as FD), no permute fusion, no DeepEP; distributed optimizer; no
overlap; all fusions off; attention_backend auto; default NCCL/allocator env with CUDA_DEVICE_MAX_CONNECTIONS=1.
OOM escalation: next TP; past the largest valid TP: optimizer_cpu_offload (fraction 1.0); still OOM: infeasible.
"""
from __future__ import annotations

import copy

from infrarsi.contracts import TaskSpec, digest

from .feasibility import layout

VERSION = 'fr_v0'
FD_MOE = 'Megatron-LM MoE README quick start (alltoall dispatcher + grouped GEMM)'
FD_FLAGSHIP = 'U+C: flagship upstream + listed necessary adaptations (design 1.1, F1)'
FR_ENV = {'CUDA_DEVICE_MAX_CONNECTIONS': '1'}


def base_config(task: TaskSpec, facts: dict, tp: int) -> dict:
    cfg = {'tensor_model_parallel_size': tp, 'pipeline_model_parallel_size': 1, 'context_parallel_size': 1,
           'sequence_parallel': tp > 1, 'micro_batch_size': 1, 'recompute_granularity': 'full',
           'recompute_method': 'uniform', 'recompute_num_layers': 1, 'use_distributed_optimizer': True,
           'overlap_grad_reduce': False, 'overlap_param_gather': False, 'apply_rope_fusion': False,
           'bias_activation_fusion': False, 'masked_softmax_fusion': False, 'gradient_accumulation_fusion': False,
           'cross_entropy_loss_fusion': False, 'attention_backend': 'auto', 'optimizer_cpu_offload': False,
           'env': dict(FR_ENV)}
    if facts.get('moe'):
        cfg.update(expert_model_parallel_size=min(facts['moe']['num_experts'], 8), expert_tensor_parallel_size=1,
                   moe_token_dispatcher_type='alltoall', moe_grouped_gemm=True, moe_permute_fusion=False,
                   moe_enable_deepep=False)
    return cfg


def states(task: TaskSpec, facts: dict, gpus_per_node: int = 8) -> tuple[list[dict], list[dict]]:
    """The full escalation ladder (configs) and the TP values skipped as structurally invalid (with reasons)."""
    ladder, skipped = [], []
    gpn = min(gpus_per_node, task.gpus)
    for tp in (1, 2, 4, 8):
        if tp > gpn:
            break
        cfg = base_config(task, facts, tp)
        probs = layout(task, cfg, facts, gpn).problems
        if probs:
            skipped.append({'tp': tp, 'reasons': probs})
            continue
        ladder.append(cfg)
    if ladder:
        off = copy.deepcopy(ladder[-1])
        off.update(optimizer_cpu_offload=True, optimizer_offload_fraction=1.0)
        ladder.append(off)
    return ladder, skipped


class FRChain:
    """OOM-escalation state machine; advance('ok'|'oom'|'error', receipt). Deterministic for equal inputs."""

    def __init__(self, task: TaskSpec, facts: dict, hw: dict | None = None):
        self.task, self.facts = task, facts
        gpn = ((hw or {}).get('summary') or {}).get('gpus_per_node', 8)
        self.flagship = task.raw['model'].get('bridge_type') == 'qwen4_exp'
        self.ladder, self.skipped = ([], []) if self.flagship else states(task, facts, gpn)
        self.i = 0
        self.history: list[dict] = []
        self.status = 'not_applicable' if self.flagship else ('probing' if self.ladder else 'infeasible')
        self.config: dict | None = None

    def current(self) -> dict | None:
        return copy.deepcopy(self.ladder[self.i]) if self.status == 'probing' else None

    def advance(self, outcome: str, receipt: dict | None = None) -> None:
        if self.status != 'probing':
            raise RuntimeError(f'chain is {self.status}')
        cfg = self.ladder[self.i]
        self.history.append({'config_digest': digest(cfg), 'tp': cfg['tensor_model_parallel_size'],
                             'offload': cfg.get('optimizer_cpu_offload', False), 'outcome': outcome,
                             'attempt_id': (receipt or {}).get('attempt_id')})
        if outcome == 'ok':
            self.status, self.config = 'feasible', copy.deepcopy(cfg)
        elif outcome == 'oom':
            self.i += 1
            if self.i >= len(self.ladder):
                self.status = 'infeasible'
        else:
            self.status = 'error'            # non-OOM failure: a pipeline problem for the intervention ledger

    def result(self) -> dict:
        return {'version': VERSION, 'task': self.task.id, 'status': self.status, 'config': self.config,
                'ladder': self.ladder, 'skipped_tp': self.skipped, 'history': self.history,
                'fd_sources': [FD_FLAGSHIP] if self.flagship else ([FD_MOE] if self.facts.get('moe') else []),
                'config_digest': digest(self.config) if self.config else None}


def fr_v0(task: TaskSpec | str, hw: dict | None = None, probe=None, facts: dict | None = None) -> dict:
    """Without probe: the plan (ladder). With probe(config) -> (outcome, receipt): run the chain to its end."""
    from infrarsi.discover import workload_facts
    task = TaskSpec.load(task) if isinstance(task, str) else task
    chain = FRChain(task, facts or workload_facts(task), hw)
    if probe is not None:
        while (cfg := chain.current()) is not None:
            outcome, receipt = probe(cfg)
            chain.advance(outcome, receipt)
    return chain.result()
