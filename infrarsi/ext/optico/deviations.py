"""Deviation ids of ENV-B (design §3.1 and Appendix A; recorded in every receipt's `deviations`).

Numbering follows the design (the ledger REPRO_ISSUES.md fixes D-OC-1/2/3/9; the design adds 4-8 and 10-14). The
header comments of reports/external_bench_20261021/scripts/run_optico_{r3,loop}.sh use D-OC-4..8 for other items
(path substitution, sysctl, torchrun wrapper, LLM driver, MODEL_PATH); those items are folded into the design's ids
below (D-OC-5 and D-OC-12) and the conflict is reported to the orchestrator (DEVIATION_IDS.md is O's file).
D-OC-15 and D-OC-16 were declared by the orchestrator on 2026-10-21 (DECISIONS_20261021.md D7; follow-up 1).
"""
from __future__ import annotations

DEVIATIONS = {
    'D-OC-1': 'data: PAI OSS WuDao MMAP returns 403; substitute MMAP built from one swift/wikipedia shard with Pai\'s '
              'own preprocessing tool (Qwen2.5 and Llama-2 tokenizers); throughput-only, loss values not comparable',
    'D-OC-2': 'data build: Pai v0.12.3 preprocess_data_megatron.py lacks padded_vocab_size; wrapper sets 0 (= tokenizer size)',
    'D-OC-3': 'transformers 4.51.3 / tokenizers 0.21.1 / huggingface_hub 0.30.2 overlay first on PYTHONPATH (venv has 5.8.1)',
    'D-OC-4': 'LLM: analyzer.py\'s hard-coded model "gpt-5.2" is served by claude-opus-5-5 through the InfraRSI shim '
              '(tools-disabled Claude Code CLI, one fresh session per call, default system prompt added); the requested '
              'name is recorded per call',
    'D-OC-5': 'no NGC 24.11 container: frozen H100 venv (torch 2.10, TE 2.16, apex) with the same Pai/Megatron sources on '
              'PYTHONPATH; torchrun = wrapper around `python -m torch.distributed.run` (stale console-script shebang)',
    'D-OC-6': 'hardware: 1 node x 8 H100 80GB (paper: A100); receipts carry TFLOPS/GPU (judge value), OptiCo\'s MFU '
              'column (/312, copied as is) and H100-MFU (/989)',
    'D-OC-7': 'workload set: Qwen3-MoE-A0.6B dropped (model unavailable); dense workloads only',
    'D-OC-8': 'LLM sampling: temperature 0.7 cannot be set through the CLI (recorded, ignored); max_tokens 8192 kept as '
              'the output cap',
    'D-OC-9': 'gpt3: Megatron core 0.15 GPT2BPETokenizer resolves "gpt2" on the HF hub; offline HF cache at the true '
              'revision 607a30d7 from the ModelScope files',
    'D-OC-10': 'weak driver: endpoint chain PC -> WIN0 vLLM local forward (B-ISW only)',
    'D-OC-11': 'loop budget: 20 new trials (run_circle.sh NUMBER_OF_CYCLES 21 incl. the default), RETRY kept but total '
               'trials incl. retries <= 26, wall cap 3 h (W3 1.5 h), LLM cap; enforced by the driver (loop killed)',
    'D-OC-12': 'paths and host: /workspace -> per-run workspace under /data/infrarsi/ext (path substitution only); '
               'run_circle.sh `sysctl -w` lines not executed (shared nodes); MODEL_PATH (vLLM path) unused by the API path',
    'D-OC-13': 'non-native arms: proposals failing validate_optico_params (script checks, TP*PP | 8, 1024 % (MBS*DP), '
               'per-stage AC_LAYERS, first/last PP rules, Megatron start-up assertions) are not launched (0 GPU cost)',
    'D-OC-14': 'analyzer endpoint reachable only through a reverse SSH tunnel (rtunnel.py): BASE_URL=http://127.0.0.1:<port>/v1',
    'D-OC-15': 'strip_think_format (decision D7, issue #19): the shim drops only the prompt lines that demand <think>-tagged '
               'reasoning (claude-opus-5-5 refuses them: [reasoning_extraction]); prompts, parsers and loop otherwise unchanged',
    'D-OC-16': 'parameter-file sanitizer in front of run_monitor.sh (workspace copy only; OptiCo source untouched): keys '
               '^[A-Z][A-Z0-9_]*$ outside a shell/loader/path/OptiCo-internal denylist, values ^[A-Za-z0-9_.,+-]*$ after '
               'trimming; otherwise the trial is not launched and is recorded as a failed run (the values reach export and '
               'an eval as root)',
}

# one-trial EnvRunner (B-DEF, B-ED, B-IS, B-ISW, B-RND, locked re-runs): OptiCo's run_monitor.sh only, no LLM
RUNNER = ('D-OC-1', 'D-OC-2', 'D-OC-3', 'D-OC-5', 'D-OC-6', 'D-OC-12', 'D-OC-13', 'D-OC-16')
# B-NAT: OptiCo's run_circle.sh loop with its analyzer (D-OC-15 only when the driver applies strip_think_format)
NATIVE = ('D-OC-1', 'D-OC-2', 'D-OC-3', 'D-OC-4', 'D-OC-5', 'D-OC-6', 'D-OC-8', 'D-OC-11', 'D-OC-12', 'D-OC-14', 'D-OC-15',
          'D-OC-16')


def for_runner(workload) -> list[str]:
    return sorted(set(RUNNER) | set(workload.extra_deviations), key=_order)


def for_native(workload, *, weak: bool = False, think_transform: bool = True) -> list[str]:
    ids = set(NATIVE) | set(workload.extra_deviations) | ({'D-OC-10'} if weak else set())
    if not think_transform:
        ids.discard('D-OC-15')                   # verbatim prompts (diagnosis only: refused by Opus 5.5)
    return sorted(ids, key=_order)


def _order(i: str) -> int:
    return int(i.rsplit('-', 1)[1])


def describe(ids) -> dict:
    return {i: DEVIATIONS[i] for i in ids}
