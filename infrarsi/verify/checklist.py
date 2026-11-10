"""The correctness checklist: generic test cases an optimized training job must pass (revamp design A).

The items are deliberately codebase-independent; an adapter only says where the evidence comes from. Each result is
{'id', 'status': pass|fail|unchecked|info|off, 'detail', 'hint'} and a failed item carries the hint (likely cause,
what to change) back to the search. Only `fail` vetoes, and `unchecked` vetoes only for the required items.
"""
from __future__ import annotations

import dataclasses
import re

PASS, FAIL, UNCHECKED, INFO, OFF = 'pass', 'fail', 'unchecked', 'info', 'off'


@dataclasses.dataclass(frozen=True)
class CheckDef:
    id: str
    family: str
    when: str           # 'screen' (every run) | 'promotion' (the candidate that is handed to the user)
    title: str
    hint: str


CHECKS = (
    CheckDef('completes', 'run health', 'screen', 'Every requested update ran and the process exited cleanly.',
             'The run crashed, hung or stopped early: read the log tail (out of memory, launch or import error, timeout).'),
    CheckDef('no_errors', 'run health', 'screen',
             'No NaN/Inf loss, out-of-memory, collective (NCCL) error or traceback; loss and gradient norm change between updates.',
             'NaN/Inf: an unstable numeric change (precision, fused kernel); OOM: save memory (recompute, smaller micro-batch, '
             'offload, more model parallelism); NCCL: a process-group or env change; frozen scalars: updates are skipped or cached.'),
    CheckDef('memory_cap', 'run health', 'screen', 'Peak device memory stays within the device cap.',
             'Too close to the memory limit for production: recompute, offload or a different parallel layout.'),
    CheckDef('same_workload', 'same job', 'screen',
             'Tokens (or samples) per update and sequence length equal the reference at every update.',
             'The global batch changed: keep micro-batch x gradient accumulation x data-parallel size constant and the sequence '
             'length unchanged.'),
    CheckDef('same_data', 'same job', 'screen', 'Each update consumes the same global batch as the reference.',
             'The data order changed: keep the sampler, its seed and the global batch composition when the layout changes.'),
    CheckDef('optimizer_steps', 'same job', 'screen',
             'One optimizer step per update, with the same learning rate as the reference at every update.',
             'Do not skip, defer or merge optimizer steps; keep the learning-rate schedule and its step counting.'),
    CheckDef('same_model', 'same job', 'screen', 'Same trainable parameters and model configuration as the reference.',
             'Do not freeze or drop parameters, or change the architecture, router top-k or vocabulary.'),
    CheckDef('frozen_files', 'same job', 'screen', 'Files declared frozen (loss, data, evaluation, measurement) are unchanged.',
             'Patch only the mutable files and never write into frozen ones at run time.'),
    CheckDef('loss_within_tolerance', 'error', 'screen', 'Training loss within the accepted tolerance of the reference run.',
             'The change alters the numerics beyond the tolerance (precision, approximation, wrong reduction or loss scaling): '
             'revert it or make it exact.'),
    CheckDef('grad_norm_within_tolerance', 'error', 'screen',
             'Gradient norm of the first update (identical state) within the tolerance of the reference.',
             'Gradient or loss scaling, clipping or the gradient reduction changed.'),
    CheckDef('timing_consistent', 'honest speed', 'screen',
             "InfraRSI's own clock agrees with the codebase's step time; no work moved out of the measured loop.",
             'The logged step time does not cover the whole update (work moved outside the timed region) or is mis-reported.'),
    CheckDef('speed_reproduces', 'deliverable', 'promotion',
             'A fresh back-to-back re-run (reference, candidate) keeps the speedup beyond the noise margin.',
             'The gain was noise or node state; prefer larger, mechanistic gains.'),
    CheckDef('resume_continues', 'deliverable', 'promotion',
             'Saving a checkpoint and resuming continues the loss within the tolerance.',
             'Checkpoint save/load breaks under the change (layout-dependent or missing state).'),
    CheckDef('eval_matches', 'deliverable', 'promotion',
             "The project's eval command on the checkpoint of the optimized run agrees with the reference's (within the tolerance).",
             'The optimized run trained a different model (checkpoint layout, precision or state placement changed the weights).'),
    CheckDef('user_tests', 'deliverable', 'promotion', "The codebase's own test commands pass on the optimized tree.",
             'A code patch broke behaviour the project tests cover.'),
    CheckDef('precision', 'info', 'screen', 'Numeric precision of parameters, gradients and optimizer state (reported only).', ''),
)
BY_ID = {c.id: c for c in CHECKS}
DEFAULT_REQUIRED = ('completes', 'same_workload', 'loss_within_tolerance')
SCREEN_IDS = tuple(c.id for c in CHECKS if c.when == 'screen')
PROMOTION_IDS = tuple(c.id for c in CHECKS if c.when == 'promotion')


def result(cid: str, status: str, detail: str = '', hint: str | None = None, **evidence) -> dict:
    """hint: a failed item's own hint over the check's general one."""
    r = {'id': cid, 'status': status, 'detail': str(detail)[:600]}
    if evidence:
        r['evidence'] = evidence
    if status == FAIL and (hint or (BY_ID.get(cid) and BY_ID[cid].hint)):
        r['hint'] = hint or BY_ID[cid].hint
    return r


# the log's error line of a failed run that names a cause outside the configuration (D2-R204c: an NVLink fault at
# update 29 of the Wan reference; the F phase: two 1-GPU torchrun jobs on one node and port 29500)
FAULT_HINTS = (
    (re.compile(r'hardware error|\bXid\b|uncorrectable ECC|fell off the bus|GPU is lost', re.I),
     'The GPU driver reported a hardware or interconnect fault, not a configuration problem: run again (on other GPUs '
     'if it repeats); nvidia-smi -q -d ECC and nvidia-smi nvlink -e show the counters.'),
    (re.compile(r'EADDRINUSE|address already in use', re.I),
     'Another process holds the port the job listens on (often torchrun\'s rendezvous, 29500): give each job its own '
     '(torchrun --master-port, MASTER_PORT / PET_MASTER_PORT) or use torchrun --standalone.'),
)


def fault_hint(error_line: str | None) -> str | None:
    return next((h for p, h in FAULT_HINTS if error_line and p.search(error_line)), None)


UNVERIFIABLE_WORKLOAD = ('every candidate would be rejected: same_workload is required, and the log has no tokens, '
                         'samples or sequence length per update nor the manifest a workload invariant. Log the tokens or '
                         'samples of each update, or declare workload.invariants for what the knobs must not change (e.g. '
                         'global_batch_size: 512, or an expression over the knobs)')


def blocking(results: list[dict], required=DEFAULT_REQUIRED) -> list[dict]:
    """The results that make a verdict invalid: every fail, and unchecked required items."""
    return [r for r in results if r['status'] == FAIL or (r['status'] == UNCHECKED and r['id'] in required)]


def coverage(results: list[dict]) -> dict:
    applicable = [r for r in results if r['status'] in (PASS, FAIL, UNCHECKED)]
    checked = [r for r in applicable if r['status'] != UNCHECKED]
    return {'checked': len(checked), 'applicable': len(applicable),
            'unchecked': [r['id'] for r in applicable if r['status'] == UNCHECKED]}


def feedback(results: list[dict], required=DEFAULT_REQUIRED, limit: int = 1200) -> str:
    """Compact text for the search: failed items with evidence and hint, then required items that could not be checked."""
    lines = []
    for r in results:
        if r['status'] == FAIL:
            lines.append(f'FAILED {r["id"]}: {r["detail"]}' + (f' -> {r["hint"]}' if r.get('hint') else ''))
    for r in results:
        if r['status'] == UNCHECKED and r['id'] in required:
            lines.append(f'NOT CHECKABLE (required) {r["id"]}: {r["detail"]}')
    text = '\n'.join(lines)
    return text if len(text) <= limit else text[:limit] + ' ...'
