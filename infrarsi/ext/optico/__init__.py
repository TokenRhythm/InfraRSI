"""ENV-B adapter (design §3): OptiCo-public's dense Megatron workloads (run_circle.sh / run_monitor.sh, Pai-Megatron-Patch
v0.12.3, Megatron core 0.15.0) on one 8-H100 WINH node.

  workloads   W1 qwen2.5-7B, W3 qwen2.5-0.5B, W2 llama3-8B (+ optional W4 llama2-7B, W5 gpt3-7B): shapes, defaults, R3
  params      knob schema, parameters.txt / report-row codec, validate_optico_params (D-OC-13)
  parse       report.csv, last-rank Megatron log, memory samples
  guards      G-B1 fixed flags, G-B2 iteration-1 loss (+ the §3.4 enabling rule), G-B3 completed iterations
  trial       one trial -> ext.common EnvRun (judge = OptiCo's row; guards veto only judge-valid trials)
  runner      OptiCoRunner (ext.common.EnvRunner): one trial of run_monitor.sh on a leased node
  native      B-NAT driver: OptiCo's own run_circle.sh loop via reverse tunnel + shim, receipts per row
  b_ed, b_rnd B-ED planner projection; B-RND sampler (seed 42, same validator)
  task        EnvTask per workload x arm, task view (OptiCo's table), OptiCoHooks (SessionHooks)
  deviations  D-OC-* registry recorded in receipts
OptiCo-public has no LICENSE: it is run and cited, never redistributed (only hashes and numbers are kept here).
"""
from .b_ed import pick as b_ed_pick
from .b_rnd import OptiCoRandomSampler
from .guards import GB2Policy, enabling_rule, policy_from_receipts
from .params import KNOBS, decode_params, encode_params, knob_schema, validate_optico_params
from .runner import OptiCoRunner
from .task import OptiCoHooks, make_task, performance_table
from .workloads import WORKLOADS, get as workload

__all__ = ['GB2Policy', 'KNOBS', 'OptiCoHooks', 'OptiCoRandomSampler', 'OptiCoRunner', 'WORKLOADS', 'b_ed_pick', 'decode_params',
           'enabling_rule', 'encode_params', 'knob_schema', 'make_task', 'performance_table', 'policy_from_receipts',
           'validate_optico_params', 'workload']
