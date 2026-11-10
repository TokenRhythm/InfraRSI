## Appendix for this run: infrastructure-only track

This appendix overrides "What you CAN do" above for this run. You may change only *how* `train.py` computes, not
*what* it computes. The goal is still the lowest val_bpb within the fixed 5-minute training budget, reached by making
training faster (more optimizer steps in the 300 s) without changing the model, the data, the objective or the
optimizer math. After every experiment a trusted checker re-evaluates the saved weights with the original
`prepare.py` and applies the checks below; an experiment that breaks a rule is recorded as invalid and does not count,
whatever val_bpb it printed.

Frozen (do not change):
- the model architecture and its constants: `GPTConfig` fields and defaults, `DEPTH`, `ASPECT_RATIO`, `HEAD_DIM`,
  `WINDOW_PATTERN`, value embeddings, weight initialisation, `build_model_config`;
- `TOTAL_BATCH_SIZE` and every learning-rate, weight-decay, beta, warmup, warmdown and final-LR constant; the
  Muon/AdamW update math and the LR, momentum and weight-decay schedules (`setup_optimizer` and its parameter groups);
- the data and its order: `make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")` called once, seed 42;
- the loss definition; `prepare.py` (checked by hash); the 300 s budget rule with the timer starting after step 10;
  bf16 autocast and `torch.set_float32_matmul_precision("high")`; the fast-fail rule; the format of the per-step log
  line (you may print it less often);
- the bindings of the protected definitions: `GPTConfig`, `GPT` (with `GPT.setup_optimizer`), `build_model_config`, the
  schedule functions, `MuonAdamW`, `adamw_step_fused`, `muon_step_fused` and the names imported from `prepare` are
  defined once (you may change the optimizer's implementation in its definition) and never re-assigned, deleted or
  monkey-patched, also not through another name (`X = GPT; X.attr = ...`); no object gets `step`, `zero_grad`,
  `param_groups`, `state`, `defaults`, `forward`, `__call__`, `__init__` or a protected name assigned as an
  attribute (e.g. `optimizer.step = ...`, `model.forward = ...`), no attribute is assigned on a computed object
  (`type(x).attr = ...`), and there are no star imports. `setattr`, `delattr`, `exec`, `eval`, `compile`,
  `globals()`, `locals()`, `vars()`, `getattr`/`hasattr` with a computed name, frame and code-object attributes
  (`__code__`, `__dict__`, `f_globals`, `sys.modules`, ...), torch's optimizer hook registries and imports of
  `importlib`, `inspect`, `types`, `ctypes`, `pickle`, `marshal` or `runpy` are not allowed. The code the protected
  names, the model and the optimizer are bound to is checked at the first optimizer step and at the end of the run.

Allowed (infrastructure):
- `DEVICE_BATCH_SIZE` (the gradient-accumulation split; numerically equivalent, not bitwise);
- `torch.compile` mode and options, CUDA graphs;
- attention call details (FA3 arguments, window handling) and fused kernels for norm, rotary and MLP with unchanged math;
- dataloader prefetching, pinned memory and threads; allocator settings; garbage collection;
- optimizer implementation efficiency (fused or foreach ops; the Newton-Schulz implementation with the same number of
  iterations and the same coefficients);
- logging synchronisation points; host-to-device overlap.

Checks after every experiment:
1. a static comparison of `train.py` with the baseline (constant assignments, `GPTConfig` defaults,
   `build_model_config`, `setup_optimizer` parameter groups, the other frozen statements and the binding rules above);
2. the saved model's config and parameter count, and `run_meta.json` (`TOTAL_BATCH_SIZE`, learning-rate constants,
   steps, total tokens) against the baseline (steps and tokens may differ: they are what you improve), and the code
   the protected definitions are bound to at the end of the run;
3. val_bpb recomputed by the trusted judge from the saved weights (this is the recorded score);
4. the printed loss of the first 10 steps within 1% (relative) of the baseline's loss at the same step.
