"""Candidate verdicts. Speed is the only objective: a valid verdict's reward is log(T_parent / T). Formal core runs use
the reward engine (`infrarsi.reward`); project runs use `infrarsi.verify.Verifier`; `ReceiptJudge` is the
receipt-only fallback that applies the hard constraints visible in a harness-validated receipt.

A judge is callable(receipt, parent_receipt | None, reference_receipt | None) -> Verdict; verdicts carry the
correctness checklist (`checks`) and its `feedback` for the search. The LLM reviewer role and the agent's own
claims never decide validity.
"""
from __future__ import annotations

import dataclasses
import math

from ..contracts import check_receipt_shape


@dataclasses.dataclass
class Verdict:
    valid: bool
    reasons: list
    step_time: float | None = None
    inter_step: float | None = None
    peak_reserved_gib: float | None = None
    reward: float | None = None                  # speed only: log(T_parent / T)
    numerics_class: str | None = None
    flags: list = dataclasses.field(default_factory=list)
    source: str = 'receipt'
    checks: list = dataclasses.field(default_factory=list)      # correctness checklist results (verify.checklist)
    feedback: str = ''                                          # failed items with evidence and hints, for the search
    coverage: dict | None = None                                # checked / applicable checklist items

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def step_time(receipt: dict | None) -> float | None:
    try:
        v = receipt['step_time']['mean_window_max_rank_s']
    except (KeyError, TypeError):
        return None
    return float(v) if type(v) in (int, float) and math.isfinite(v) and v > 0 else None


def inter_step(receipt: dict | None) -> float | None:
    try:
        v = receipt['step_time']['inter_step_mean_s']
    except (KeyError, TypeError):
        return None
    return float(v) if type(v) in (int, float) and math.isfinite(v) and v > 0 else None


def numerics_block(receipt: dict | None) -> dict:
    """Harness numerics comparison vs the FR plan-A reference (receipt['harness']['numerics'] in validated receipts)."""
    r = receipt or {}
    return r.get('numerics') or (r.get('harness') or {}).get('numerics') or {}


def _nonneg(v) -> float | None:
    return float(v) if type(v) in (int, float) and math.isfinite(v) and v >= 0 else None


def gpu_seconds(receipt: dict | None) -> float | None:
    """Driver-measured occupancy: harness.gpu_hours of a validated receipt first (the training process cannot write
    it), else cost.gpu_seconds (synthetic receipts and the runner's own failure records). None when missing,
    negative or non-finite: the caller then charges world x elapsed and marks accounting incomplete."""
    if not isinstance(receipt, dict):
        return None
    h = (receipt.get('harness') or {}).get('gpu_hours') if isinstance(receipt.get('harness'), dict) else None
    if h is not None:
        h = _nonneg(h)
        return None if h is None else h * 3600.0
    c = receipt.get('cost')
    return _nonneg(c.get('gpu_seconds')) if isinstance(c, dict) else None


def wall_seconds(receipt: dict | None) -> float | None:
    if not isinstance(receipt, dict):
        return None
    h = receipt.get('harness')
    if isinstance(h, dict) and h.get('wall_s') is not None:
        return _nonneg(h.get('wall_s'))
    c = receipt.get('cost')
    return _nonneg(c.get('wall_seconds')) if isinstance(c, dict) else None


def peak_reserved(receipt: dict | None) -> float | None:
    mem = (receipt or {}).get('memory') or {}
    v = mem.get('max_peak_reserved_gib')
    if v is None and isinstance(mem.get('per_rank'), list):
        vals = [r.get('peak_reserved_gib') for r in mem['per_rank'] if isinstance(r, dict)]
        vals = [x for x in vals if type(x) in (int, float)]
        v = max(vals) if vals else None
    return float(v) if type(v) in (int, float) else None


class ReceiptJudge:
    """Hard constraints from reward.yaml that a validated receipt exposes, plus the speed gain vs parent."""

    def __init__(self, *, peak_reserved_gib_max: float = 78.0, inter_step_consistency: float = 0.02,
                 numerics_allowed=('E0', 'E1'), extra_checks=()):
        self.peak_max, self.inter_tol = peak_reserved_gib_max, inter_step_consistency
        self.numerics_allowed = numerics_allowed
        self.extra_checks = list(extra_checks)       # callables(receipt, parent, reference) -> list[str]

    def __call__(self, receipt: dict, parent: dict | None = None, reference: dict | None = None) -> Verdict:
        reasons = [f'shape: {p}' for p in check_receipt_shape(receipt)]
        if receipt.get('valid') is not True:
            reasons += [f'receipt invalid: {r}' for r in (receipt.get('invalid_reasons') or ['valid != true'])]
        if receipt.get('steps') is not None and receipt.get('steps_completed') != receipt.get('steps'):
            reasons.append(f'incomplete updates {receipt.get("steps_completed")}/{receipt.get("steps")}')
        ev = receipt.get('events') or {}
        for k in ('alloc_ooms', 'nccl_errors', 'nan_inf_steps'):
            if (ev.get(k) or 0) > 0:
                reasons.append(f'events.{k}={ev.get(k)}')
        t, ti, mem = step_time(receipt), inter_step(receipt), peak_reserved(receipt)
        if t is None:
            reasons.append('no StepTime')
        if mem is not None and mem > self.peak_max:
            reasons.append(f'peak reserved {mem:.1f} GiB > {self.peak_max}')
        num = numerics_block(receipt)
        ncls = num.get('class')
        if ncls is not None and ncls not in self.numerics_allowed:
            reasons.append(f'numerics class {ncls} not allowed')
        if num.get('passed') is False or (num.get('compared') and num.get('allowed_ok') is False):
            reasons.append('numerics outside tolerance: ' + ', '.join(map(str, (num.get('violations') or num.get('reasons') or [])[:3])))
        pt, pti = step_time(parent), inter_step(parent)
        if t and ti and pt and pti:
            gap = abs(math.log(pti / ti) - math.log(pt / t))
            if gap > self.inter_tol:
                reasons.append(f'InterStep inconsistent with StepTime improvement (|delta log|={gap:.3f} > {self.inter_tol})')
        for check in self.extra_checks:
            reasons += check(receipt, parent, reference)
        reward = math.log(pt / t) if not reasons and t and pt else None
        return Verdict(valid=not reasons, reasons=reasons, step_time=t, inter_step=ti, peak_reserved_gib=mem,
                       reward=reward, numerics_class=ncls, source='receipt',
                       feedback='\n'.join(f'FAILED {r}' for r in reasons)[:1200])


class RewardEngineJudge:
    """The formal core judge: infrarsi.reward.evaluate on harness-validated, signed receipts.

    evaluate(receipt, parent_receipt, reward_cfg, ...) -> {'valid', 'R', 'violations', 'metrics', 'checks', ...}.
    The parent is the candidate's direct parent (InterStep consistency, frozen-zone digest, log-ratio reward).
    """

    def __init__(self, reward_cfg: dict | None = None, **kwargs):
        from ..reward import evaluate                  # ImportError where the reward engine is not shipped
        self._evaluate, self.reward_cfg, self.kwargs = evaluate, reward_cfg, kwargs

    def __call__(self, receipt: dict, parent: dict | None = None, reference: dict | None = None) -> Verdict:
        rv = self._evaluate(receipt, parent, self.reward_cfg, **self.kwargs)
        m = rv.get('metrics') or {}
        valid = bool(rv.get('valid'))
        r = rv.get('R')
        return Verdict(valid=valid, reasons=list(rv.get('violations') or ([] if valid else ['reward engine: invalid'])),
                       step_time=m.get('StepTime') or step_time(receipt), inter_step=m.get('InterStep') or inter_step(receipt),
                       peak_reserved_gib=(m.get('PeakMemory') or {}).get('max_reserved_gib') or peak_reserved(receipt),
                       reward=r if valid and type(r) in (int, float) and math.isfinite(r) else None,
                       numerics_class=numerics_block(receipt).get('class'), source='reward_engine',
                       checks=list(rv.get('checks') or []), feedback=rv.get('feedback') or '', coverage=rv.get('coverage'))


class LenientJudge:
    """E4 'lenient reward' (StepTime and completion only); never for formal tables. Accepted candidates must be
    re-reviewed offline with the full reward (redteam.detectors.full_accepts / RewardEngineJudge)."""

    def __call__(self, receipt: dict, parent: dict | None = None, reference: dict | None = None) -> Verdict:
        t, pt = step_time(receipt), step_time(parent)
        ok = receipt.get('steps') is not None and receipt.get('steps_completed') == receipt.get('steps') and t is not None
        return Verdict(valid=ok, reasons=[] if ok else ['incomplete or no StepTime'], step_time=t, inter_step=inter_step(receipt),
                       peak_reserved_gib=peak_reserved(receipt), reward=math.log(pt / t) if ok and pt else None, source='lenient')


def default_judge(**kw):
    """The reward engine when it is importable (formal runs), else the receipt-only fallback."""
    try:
        return RewardEngineJudge(kw.pop('reward_cfg', None))
    except ImportError:
        return ReceiptJudge(**kw)
