"""InfraRSI's view of ENV-C (design §4.3 C-IS / C-RND): task view, knob validation, the train.py patch guard.

C-IS sees read_env (hardware, the R2 baseline summary, prepare.py constants), read_file on train.py (base or a
candidate's version) and the documented read-only files (prepare.py, README.md), proposes {'train.py': text} patches
(<= 300 changed lines) and screens them; validate_patch applies the static infra-only check before any launch, so a
statically invalid candidate costs no GPU time. C-RND proposes knob points (knobs.KNOB_SCHEMA). The verifier uses the
§4.2 checklist. Native arms use `native=True`: only the mutable file set is enforced before a run (their checks run
after every experiment, as the design prescribes).
"""
from __future__ import annotations

from pathlib import Path

from ..common.session_hooks import SessionHooks
from . import infra_only_check, knobs, source

APPENDIX = (Path(__file__).resolve().parent / 'program_infra_appendix.md').read_text(encoding='utf-8')


class AutoresearchHooks(SessionHooks):
    banned_imports = ('subprocess', 'socket', 'requests', 'urllib', 'urllib3', 'http', 'ctypes', 'importlib', 'runpy')

    def __init__(self, task, *, native: bool = False, frozen_package_digest: str | None = None):
        super().__init__(task, base_files={'train.py': source.base_train()},
                         read_only_files={'prepare.py': source.candidate_prepare(task.raw.get('prepare_variant', source.DEFAULT_PREPARE)),
                                          'README.md': source.upstream('README.md')},
                         frozen_package_digest=frozen_package_digest)
        self.track = task.raw.get('track', 'infra')
        self.native = native

    def contract_text(self) -> str:
        note = ('Instructions above that name Megatron-Core, StepTime, L0 schema keys or runtime/overlays do not apply '
                'here; this contract does. ')
        if self.track == 'free':
            return note + ('autoresearch (karpathy), free track: train.py may change in any way (prepare.py, the data and '
                           'the evaluation are fixed); the score is the printed val_bpb after the fixed 300 s of training.')
        return note + ('autoresearch (karpathy) on 1 H100: train.py trains a 50M-parameter GPT for a fixed 300 s (timer '
                       'starts after step 10) and is scored by val_bpb (lower is better), recomputed by a trusted judge from '
                       'the saved weights with the original prepare.py. Infra-only track: you may change only HOW train.py '
                       'computes - more optimizer steps in the 300 s - never WHAT it computes. A candidate is the complete '
                       'new train.py.\n' + APPENDIX)

    def verifier_checklist(self) -> list[str]:
        if self.track == 'free':
            return super().verifier_checklist()
        return ['Model architecture, its constants and initialisation unchanged (GPTConfig, DEPTH, ASPECT_RATIO, HEAD_DIM, '
                'WINDOW_PATTERN, value embeddings, build_model_config).',
                'TOTAL_BATCH_SIZE, every LR / WD / beta / warmup / warmdown constant, the Muon/AdamW math and the schedules unchanged.',
                'Data and order unchanged: make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train") once, seed 42.',
                'Loss, prepare.py, the 300 s rule (timer after step 10), bf16 autocast, matmul precision "high", fast-fail and '
                'the progress-line format unchanged.',
                'The change is infrastructure only (batch split, compile options, fused kernels with the same math, data '
                'prefetch, allocator, GC, optimizer implementation efficiency, sync points).',
                'The change plausibly yields more optimizer steps in 300 s (higher tok/s) given the parent receipt.']

    def validate_config(self, config):
        resolved, errors, warnings = super().validate_config(config)
        if not errors and resolved and resolved != knobs.DEFAULTS:
            try:
                knobs.apply(source.base_train(), resolved)
            except knobs.KnobError as exc:
                errors = [str(exc)]
        return (resolved if not errors else {}), errors, warnings

    def validate_patch(self, files):
        if self.native:                       # native arms: only the mutable file set before a run
            outside = sorted(set(files or {}) - set(self.task.mutable_globs)) if isinstance(files, dict) else ['?']
            return ([f'not mutable: {outside}'] if outside else []), []
        errors, warnings = super().validate_patch(files)
        text = files.get('train.py') if isinstance(files, dict) else None
        if self.track == 'infra' and isinstance(text, str) and not any('syntax error' in e for e in errors):
            errors += [f'infra-only: {r["detail"]}' for r in infra_only_check.static(source.base_train(), text).rejections]
        return errors, warnings

    def task_view(self) -> dict:
        view = super().task_view()
        view['start'] = 'c0 = the unmodified baseline train.py (D-AR-2 applied); read it with read_file("train.py")'
        return view

    def skill_tags(self) -> list[str]:
        return sorted(set(super().skill_tags()) | {'single-node', 'gpus1', 'h100', 'pretraining', 'torch-compile'})

    def skill_query(self) -> str:
        return ('single GPU pretraining throughput torch.compile kernels data path prefetch memory allocator optimizer '
                'step time tokens per second profile whole update')
