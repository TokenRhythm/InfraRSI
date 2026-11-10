"""ENV-B workloads: OptiCo-public dense training scripts (Pai-Megatron-Patch v0.12.3 + Megatron core 0.15.0) on WINH.

Model shapes are the values hard-coded in OptiCo's script/train/run_<model>.sh (tests/cpu/ext/optico cross-check them
against the local third_party_src copy); `defaults` are the scripts' own `${X:-default}` values (= OptiCo's starting
point); default TFLOPS and trial walls are R3 (REPRO_RESULTS.md, WINH4, OptiCo's own monitor). Budgets are design
§3.5 (per arm and workload: 20 new trials, 3 h wall, W3 1.5 h; LLM 1.5M tokens / $30, design §6.2).

OptiCo-public has no LICENSE: only numbers (shapes, defaults) are recorded here, never its text.
"""
from __future__ import annotations

import dataclasses

GBS = 1024                      # GLOBAL_BATCH_SIZE, hard-coded in every run_<model>.sh
SEQ = 4096                      # SEQ_LENGTH
TRAIN_ITERS = 5
GPUS = 8                        # one node, CUDA_VISIBLE_DEVICES=0..7 (run_circle.sh)
WINDOW = (3, 5)                 # OptiCo's judge: TFLOPS of iterations 3..5 (`tail -n +3`) of the last node's log
H100_PEAK_TFLOPS = 989.0        # dense bf16/fp16 peak, H100 SXM (H100-MFU = TFLOPS / 989)
OPTICO_MFU_PEAK = 312.0         # OptiCo's MFU column divides by the A100 constant (D-OC-6: copied as is)
GPU_MEM_LIMIT_MIB = 81559       # nvidia-smi memory.total on WINH H100 80GB HBM3 (OptiCo's fixed_parameters line)

DEFAULT_KNOBS = {'MICRO_BATCH_SIZE': 1, 'TP': 1, 'PP': 1, 'OVERLAP_GRAD_REDUCE': False, 'OVERLAP_PARAM_GATHER': False,
                 'AC': 'none', 'AC_LAYERS': 1, 'FUSED_CROSS_ENTROPY': False, 'FIRST_PP_NUM_LAYERS': None,
                 'LAST_PP_NUM_LAYERS': None}


@dataclasses.dataclass(frozen=True)
class Workload:
    key: str                        # 'W1'
    task_id: str                    # 'ENV-B-W1'
    model_type: str                 # run_circle.sh argument 1 (MODEL_TYPE)
    model_size: str                 # argument 2 (MODEL_SIZE)
    script: str                     # script/train/<file>
    num_layers: int
    hidden: int
    heads: int
    kv_heads: int                   # --num-query-groups (= heads without GQA)
    ffn: int                        # --ffn-hidden-size (gpt3: Megatron's swiglu default, the flag is not passed)
    vocab: int                      # embedding rows (tokenizer + extra/padded vocab)
    tied_embeddings: bool
    qkv_bias: bool
    hf_model_type: str              # count_params family: gated MLP (--swiglu) for every OptiCo dense script
    defaults: dict                  # OptiCo knobs of the script defaults (B-DEF = OptiCo's own start)
    default_tflops: float           # R3, TFLOPS/GPU (mean of iterations 3..5, last rank)
    default_trial_s: float          # R3 wall of one monitor run
    wall_budget_s: float            # per arm (design §3.5)
    gpu_hours_budget: float         # per arm (design §3.5, 8-GPU occupancy) [estimate]
    order: int                      # design §3.2 order
    ffn_flag: bool = True           # the script passes --ffn-hidden-size
    gqa_flag: bool = True           # the script passes --num-query-groups
    cp_flag: bool = True            # the script passes --context-parallel-size
    extra_deviations: tuple = ()    # workload-specific deviation ids

    def knobs(self, **changes) -> dict:
        return {**self.defaults, **changes}

    def fixed_parameters_text(self) -> str:
        """OptiCo's fixed_parameters.txt as run_monitor.sh writes it for this workload on one WINH node."""
        return '\n'.join([f'Model Type: {self.model_type}', f'Model Size: {self.model_size}', f'NUM_LAYERS: {self.num_layers}',
                          'Precison: fp16', f'Global Batch Size: {GBS}', f'Sequence Length: {SEQ}', 'Number of Nodes: 1',
                          f'Number of GPUs: {GPUS}', f'GPU memory limit per card: {GPU_MEM_LIMIT_MIB}']) + '\n'

    def hf_like_config(self) -> dict:
        """The script's model as an HF-style config for discover.workload.count_params (planner facts)."""
        return {'model_type': self.hf_model_type, 'hidden_size': self.hidden, 'num_hidden_layers': self.num_layers,
                'vocab_size': self.vocab, 'num_attention_heads': self.heads, 'num_key_value_heads': self.kv_heads,
                'intermediate_size': self.ffn, 'tie_word_embeddings': self.tied_embeddings, 'attention_bias': self.qkv_bias}


def _w(key, task_id, mt, ms, script, layers, hidden, heads, kv, ffn, vocab, tied, bias, defaults, tflops, trial_s, wall_h,
       gpuh, order, **kw) -> Workload:
    return Workload(key=key, task_id=task_id, model_type=mt, model_size=ms, script=script, num_layers=layers, hidden=hidden,
                    heads=heads, kv_heads=kv, ffn=ffn, vocab=vocab, tied_embeddings=tied, qkv_bias=bias,
                    hf_model_type='qwen2' if mt == 'qwen2.5' else 'llama', defaults={**DEFAULT_KNOBS, **defaults},
                    default_tflops=tflops, default_trial_s=trial_s, wall_budget_s=wall_h * 3600.0, gpu_hours_budget=gpuh,
                    order=order, **kw)


_FULL_PP = lambda pp: {'PP': pp, 'AC': 'full', 'AC_LAYERS': 1}          # noqa: E731

WORKLOADS = {
    # design §3.2: W1 qwen2.5 7B (README quick start), W3 qwen2.5 0.5B, W2 llama3 8B; W4/W5 optional
    'W1': _w('W1', 'ENV-B-W1', 'qwen2.5', '7B', 'run_qwen.sh', 28, 3584, 28, 4, 18944, 152064, False, True,
             _FULL_PP(4), 345.4, 394.0, 3.0, 24.0, 1),
    'W3': _w('W3', 'ENV-B-W3', 'qwen2.5', '0.5B', 'run_qwen.sh', 24, 896, 14, 2, 4864, 151936, True, True,
             {}, 228.8, 130.0, 1.5, 8.0, 2),
    # llama3 8B runs with the Llama-2 tokenizer and dataset (run_llama3.sh's own DATASET_PATH/TOKENIZER_MODEL_PATH)
    'W2': _w('W2', 'ENV-B-W2', 'llama3', '8B', 'run_llama3.sh', 32, 4096, 32, 8, 14336, 32000, False, False,
             _FULL_PP(8), 384.6, 376.0, 3.0, 24.0, 3),
    'W4': _w('W4', 'ENV-B-W4', 'llama2', '7B', 'run_llama2.sh', 32, 4096, 32, 32, 11008, 32000, False, False,
             _FULL_PP(8), 376.5, 362.0, 3.0, 24.0, 4, gqa_flag=False),
    # gpt3: --swiglu without --ffn-hidden-size -> Megatron's int(4*h*2/3/64)*64; GPT2BPETokenizer 50257 padded to 50304
    'W5': _w('W5', 'ENV-B-W5', 'gpt3', '7B', 'run_gpt3.sh', 32, 4096, 32, 32, 10880, 50304, False, True,
             _FULL_PP(8), 331.8, 415.0, 3.0, 24.0, 5, ffn_flag=False, gqa_flag=False, cp_flag=False,
             extra_deviations=('D-OC-9',)),
}
BY_TASK = {w.task_id: w for w in WORKLOADS.values()}
BY_ARGS = {(w.model_type, w.model_size): w for w in WORKLOADS.values()}


def get(key_or_task: str | Workload) -> Workload:
    if isinstance(key_or_task, Workload):
        return key_or_task
    if key_or_task in WORKLOADS:
        return WORKLOADS[key_or_task]
    if key_or_task in BY_TASK:
        return BY_TASK[key_or_task]
    raise KeyError(f'unknown ENV-B workload {key_or_task!r} (known: {sorted(WORKLOADS)} / {sorted(BY_TASK)})')
