"""ENV-B task contracts (EnvTask per workload x arm), the InfraRSI task view and the SessionHooks of design §3.4.

The proposer sees (design §3.4): OptiCo's fixed_parameters.txt, the performance table in OptiCo's compressed markdown
format, the current parameters, receipt digests (phases, memory peak, first 30 failure-log lines) and the top-2
retrieved skill sections; the verifier checks the script constraints and duplicates; one action per turn.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from ..common.env_task import SCHEMA_TASK, EnvTask
from ..common.session_hooks import SessionHooks
from . import deviations as dv
from . import params as pm
from . import parse
from . import workloads as wl

FROZEN_MANIFEST = Path(__file__).resolve().parent / 'frozen_manifest.json'
OPTICO_ROOT = Path(__file__).resolve().parents[3] / 'third_party_src' / 'OptiCo-public'
METRIC = 'iter_time_s'
TAU = math.log(1.05)               # CTG(+5 %): TFLOPS >= 1.05 x default (design §1.4)
MARGIN = {'kind': 'rsd', 'floor': 0.03, 'k': 3}     # max(3 %, 3 x RSD of the default's iterations 3..5 TFLOPS)
LLM_TOKENS, LLM_USD = 1_500_000, 30.0               # per LLM arm and workload (design §3.5, §6.2)
# design §0.2 / §3.5: 20 new trials per search arm. SearchSession arms (B-IS, B-ISW, B-RND) first re-run the default
# start (c0), as OptiCo's loop does with its iteration 1, and SearchSession counts that screen: 21 = start + 20 new.
# B-NAT: run_circle.sh NUMBER_OF_CYCLES 21 (= default + 20 new), <= 26 trials incl. RETRY (driver-enforced).
ARMS = {
    'B-DEF': {'screens': 1, 'llm': False},
    'B-ED': {'screens': 1, 'llm': False},
    'B-NAT': {'screens': 21, 'llm': True, 'max_trials': 26},
    'B-IS': {'screens': 21, 'llm': True},
    'B-IS2': {'screens': 21, 'llm': True},          # the current search (search.run, HLS), phase E of 10-24
    'B-IS2-noexp': {'screens': 21, 'llm': True},    # the same without the experience packs
    'B-ISW': {'screens': 21, 'llm': True, 'wall_factor': 2.0},
    'B-RND': {'screens': 21, 'llm': False},
}


def frozen_files() -> dict:
    return dict(json.loads(FROZEN_MANIFEST.read_text(encoding='utf-8'))['files'])


def facts(workload) -> dict:
    w = wl.get(workload)
    return {'model': {'name': f'{w.model_type}-{w.model_size}', 'model_type': w.model_type, 'model_size': w.model_size,
                      'script': f'script/train/{w.script}', 'num_layers': w.num_layers, 'hidden_size': w.hidden,
                      'num_attention_heads': w.heads, 'num_query_groups': w.kv_heads, 'ffn_hidden_size': w.ffn,
                      'vocab': w.vocab, 'tied_embeddings': w.tied_embeddings,
                      'weights': 'random init (the script\'s --load is not a Megatron checkpoint; Megatron starts from random)'},
            'data': {'name': 'optico-mmap', 'kind': 'Megatron MMAP (substitute: swift/wikipedia shard, D-OC-1)',
                     'seq_len': wl.SEQ, 'global_batch_size': wl.GBS, 'tokens_per_iteration': wl.GBS * wl.SEQ},
            'phases': {'iterations': wl.TRAIN_ITERS, 'judge_window': list(wl.WINDOW),
                       'judge': 'OptiCo run_monitor.sh: TFLOPS/GPU = mean of the throughput lines of iterations 3..5 '
                                '(last node log); success iff > 0'},
            'precision': 'fp16 (--apply-query-key-layer-scaling, initial loss scale 65536), distributed optimizer, TE, '
                         'flash attention, sequence parallel iff TP > 1, --tp-comm-overlap iff TP > 1 (script constants)',
            'hardware': {'gpu_type': 'H100-80GB', 'gpus': wl.GPUS, 'nodes': 1, 'nvlink_intra': True,
                         'gpu_mem_limit_mib': wl.GPU_MEM_LIMIT_MIB, 'peak_tflops_dense': wl.H100_PEAK_TFLOPS},
            'default': {'params': dict(w.defaults), 'tflops_per_gpu_r3': w.default_tflops, 'trial_wall_s_r3': w.default_trial_s},
            'fixed_parameters_txt': w.fixed_parameters_text()}


def make_task(workload, arm: str = 'B-IS', *, frozen: dict | None = None) -> EnvTask:
    w = wl.get(workload)
    if arm not in ARMS:
        raise ValueError(f'unknown ENV-B arm {arm!r} (known: {sorted(ARMS)})')
    a = ARMS[arm]
    wall = w.wall_budget_s * a.get('wall_factor', 1.0)
    budget = {'screens': a['screens'], 'llm_tokens': LLM_TOKENS if a['llm'] else 0, 'wall_seconds': wall,
              'gpu_hours': w.gpu_hours_budget * a.get('wall_factor', 1.0) if a['screens'] > 1 else
              round(wl.GPUS * w.default_trial_s * 3 / 3600.0, 3)}
    if a['llm']:
        budget['llm_usd'] = LLM_USD                  # no USD key for non-LLM arms (a 0 cap would block screens)
    if a.get('max_trials'):
        budget['max_trials_incl_retries'] = a['max_trials']
    spec = {'schema': SCHEMA_TASK, 'id': w.task_id, 'env': 'optico', 'split': 'external', 'arm': arm,
            'metric_name': METRIC, 'action_space': 'knobs', 'knob_schema': pm.knob_schema(w),
            'frozen_files': dict(frozen if frozen is not None else frozen_files()), 'mutable_globs': [],
            'max_patch_lines': 0, 'budget': budget, 'margin': dict(MARGIN), 'tau': TAU, 'world_size': wl.GPUS,
            'facts': facts(w), 'hardware': {'gpus': wl.GPUS, 'nodes': 1, 'gpu_type': 'H100-80GB'},
            'workload': {'key': w.key, 'run_circle_args': [w.model_type, w.model_size], 'seq_len': wl.SEQ,
                         'global_batch_size': wl.GBS, 'iterations': wl.TRAIN_ITERS},
            'start_config': dict(w.defaults), 'seed': 42,
            'deviations': dv.for_native(w) if arm == 'B-NAT' else dv.for_runner(w)}
    return EnvTask.from_dict(spec)


# ---------------------------------------------------------------- OptiCo's performance table
def performance_table(receipts) -> str:
    """OptiCo's convert_table2md output for the trials so far: the report.csv rows (judge rows of the receipts), MICRO_BATCH_SIZE
    as the index column, cells compacted the way analyzer.py does ('|' without padding, '-' rules)."""
    header = list(parse.DENSE_HEADER)
    lines = ['|' + '|'.join(header) + '|', '|' + '|'.join('-' for _ in header) + '|']
    for r in receipts or []:
        row = ((r or {}).get('judge') or {}).get('row')
        if row:
            lines.append('|' + '|'.join(str(row.get(h, '')) for h in header) + '|')
    text = '\n'.join(lines)
    text = re.sub(r' *\| *', '|', text)
    return re.sub('-+', '-', text).replace('-:', '-').replace(':-', '-')


def parameter_rules(workload) -> list[str]:
    w = wl.get(workload)
    return [f'MICRO_BATCH_SIZE in {list(pm.MBS_CHOICES)}, TP in {list(pm.TP_CHOICES)}, PP in {list(pm.PP_CHOICES)}; '
            f'TP*PP must divide {wl.GPUS}; DP = {wl.GPUS}/(TP*PP); {wl.GBS} % (MBS*DP) == 0',
            f'TP must divide num_attention_heads {w.heads}, num_query_groups {w.kv_heads} and ffn_hidden_size {w.ffn}',
            f'NUM_LAYERS {w.num_layers}: without FIRST/LAST_PP_NUM_LAYERS, NUM_LAYERS % PP == 0',
            'PP=1: FIRST_PP_NUM_LAYERS and LAST_PP_NUM_LAYERS must be None; PP=2 with a split: FIRST + LAST == NUM_LAYERS; '
            'PP>2 with a split: NUM_LAYERS-FIRST-LAST > 0 and divisible by PP-2 (unset side = 0), and the layers left '
            'after the set sides divisible by the stages left',
            'AC in none|sel|full|offload; AC=full: (NUM_LAYERS / PP) % AC_LAYERS == 0 and AC_LAYERS divides every stage; '
            'AC=offload: PP=1 and AC_LAYERS < NUM_LAYERS (overlap flags are dropped by the script); AC_LAYERS is ignored '
            'for none/sel',
            'OVERLAP_PARAM_GATHER=true needs OVERLAP_GRAD_REDUCE=true']


class _Live:
    """A task-view entry recomputed whenever the view is serialized: agent.optimize builds the view once and renders it
    every turn with json.dumps(..., default=str), so OptiCo's performance table stays current (design §3.4)."""

    def __init__(self, fn):
        self.fn = fn

    def __str__(self) -> str:
        v = self.fn()
        return v if isinstance(v, str) else json.dumps(v, default=str)

    __repr__ = __str__


class OptiCoHooks(SessionHooks):
    """SessionHooks for ENV-B: OptiCo knobs (L0 only), validate_optico_params, OptiCo's table in the task view."""

    def __init__(self, task: EnvTask, *, runner=None, receipts=None, **kw):
        kw.setdefault('layers', ('L0',))
        super().__init__(task, **kw)
        self.workload = wl.get(task.id)
        self.runner = runner                # an OptiCoRunner: its history feeds the performance table
        self.receipts = receipts            # or an explicit receipt list (B-NAT rows, tests)

    def _history(self) -> list:
        if self.receipts is not None:
            return list(self.receipts)
        return list(getattr(self.runner, 'history', None) or [])

    def _launched(self) -> list:
        return [r for r in self._history() if (r or {}).get('judge')]

    def _best(self):
        best = min((r for r in self._launched() if r.get('valid')), key=lambda r: r['metric']['value'], default=None)
        return best and {'attempt_id': best.get('attempt_id'), 'params': best.get('params'),
                         'iter_time_s': best['metric']['value'],
                         'tflops_per_gpu': (best['metric'].get('raw') or {}).get('tflops_per_gpu')}

    def task_view(self) -> dict:
        view = super().task_view()
        view.update(
            fixed_parameters=self.workload.fixed_parameters_text(),
            start_parameters=dict(self.workload.defaults),
            parameter_rules=parameter_rules(self.workload),
            performance_table=_Live(lambda: performance_table(self._launched())),
            current_best=_Live(self._best),
            trials_so_far=_Live(lambda: len(self._launched())))
        return view

    def validate_config(self, config) -> tuple[dict, list, list]:
        if type(config) is not dict:
            return {}, ['config must be an object of OptiCo knobs'], []
        norm, errors, warnings = pm.validate_optico_params(config, self.workload)
        return (norm if not errors else {}), errors, warnings

    def validate_patch(self, files) -> tuple[list, list]:
        return ['ENV-B has no file actions: change the OptiCo parameter set with propose_config'], []

    def contract_text(self) -> str:
        w = self.workload
        return (f'ENV-B is OptiCo-public\'s training workload `run_circle.sh {w.model_type} {w.model_size}` '
                f'(Pai-Megatron-Patch v0.12.3, Megatron core 0.15.0) on one node of {wl.GPUS} H100 80GB GPUs. One trial runs '
                f'OptiCo\'s own run_monitor.sh: {wl.TRAIN_ITERS} iterations, global batch {wl.GBS}, seq {wl.SEQ}, fp16; '
                'GBS, seq, precision, LR, iterations, distributed optimizer, TE, flash attention and SP are fixed by the script. '
                'You change only OptiCo\'s parameter file: ' + ', '.join(pm.KNOBS) + '. The judge is OptiCo\'s report.csv '
                '(success iff TFLOPS/GPU > 0, averaged over iterations 3-5); InfraRSI minimizes iter_time_s, the mean '
                'elapsed time of iterations 3-5 (inverse to TFLOPS at fixed work). Guards reject runs whose fixed flags '
                'changed (G-B1), that did not complete 5 iterations (G-B3), and, once enabled, whose iteration-1 loss is '
                '> 2 % from the default (G-B2). Proposals that break the rules are rejected before launch at no GPU cost: '
                + ' | '.join(parameter_rules(w)))

    def verifier_checklist(self) -> list[str]:
        return ['The parameter set satisfies the script rules (TP*PP | 8, GBS % (MBS*DP), AC_LAYERS per stage, '
                'first/last pipeline split) and Megatron start-up rules (TP divides heads/KV groups).',
                'The set is not a duplicate of an earlier trial and avoids configurations that already failed '
                '(OOM, crashes) in the performance table.',
                'The change has a concrete expected benefit on iter_time_s (pipeline bubble, recompute cost, '
                'communication overlap, kernel fusion, memory headroom) given the parent receipt.']

    def skill_tags(self) -> list[str]:
        return sorted({'optico', 'external', 'megatron', 'h100-80gb', f'gpus{wl.GPUS}', 'single-node', 'dense', 'fp16'})

    def skill_query(self) -> str:
        return ('megatron parallel layout tensor pipeline parallel micro batch size recompute activation checkpointing '
                'memory headroom fused cross entropy overlap grad reduce param gather pipeline bubble throughput TFLOPS')

    def memory_tags(self) -> dict:
        return {'hardware': ['h100-80gb', f'gpus{wl.GPUS}', 'nodes1'],
                'model': ['optico', self.workload.model_type, 'dense'], 'data': ['pt_stream', 'seq4k']}
