"""EnvJudge: the agent.judge verdict for signed EnvReceipts (EXT_INTERFACES §2).

A receipt that fails `harness.signing.verify` is not read further. Otherwise valid = receipt['valid'], which the
runner computed as judge_valid and no guard rejection; the judge re-checks that conjunction and the metric, so it can
only keep or remove validity, never grant it. step_time = the minimized metric, numerics_class 'env',
source 'env:<env>'. `EnvJudge(receipt, parent, reference)` evaluates at once (the spelling of EXT_INTERFACES);
`EnvJudge(env='autoresearch')` builds a reusable judge object for SearchSession / optimize (judge=...).
"""
from __future__ import annotations

import math

from ...agent.judge import Verdict
from ...harness import signing

SCHEMA_RECEIPT = 'ext-receipt-v1'


def metric_value(receipt: dict | None) -> float | None:
    try:
        v = receipt['metric']['value']
    except (KeyError, TypeError):
        return None
    return float(v) if type(v) in (int, float) and math.isfinite(v) and v > 0 else None


class EnvJudge:
    def __new__(cls, *args, **kw):
        if args and (isinstance(args[0], dict) or (args[0] is None and len(args) > 1)):   # EnvJudge(receipt, parent, reference)
            judge = object.__new__(cls)._setup(kw.pop('env', None), require_signature=kw.pop('require_signature', True))
            return judge(*args, **kw)
        return object.__new__(cls)

    def __init__(self, env: str | None = None, *, require_signature: bool = True):
        self._setup(env, require_signature=require_signature)

    def _setup(self, env: str | None = None, *, require_signature: bool = True) -> 'EnvJudge':
        self.env, self.require_signature = env, require_signature
        return self

    def __call__(self, receipt, parent: dict | None = None, reference: dict | None = None) -> Verdict:
        source = f'env:{self.env or (receipt.get("env") if isinstance(receipt, dict) else None)}'
        if not isinstance(receipt, dict):
            return Verdict(valid=False, reasons=['receipt is not an object'], numerics_class='env', source=source)
        if self.require_signature and not signing.verify(receipt):
            return Verdict(valid=False, reasons=['harness signature missing or invalid'], numerics_class='env', source=source)
        reasons = []
        if receipt.get('schema') != SCHEMA_RECEIPT:
            reasons.append(f'schema {receipt.get("schema")!r} != {SCHEMA_RECEIPT}')
        if self.env is not None and receipt.get('env') != self.env:
            reasons.append(f'receipt of env {receipt.get("env")!r}, judge for {self.env!r}')
        rejections = receipt.get('guard_rejections')
        if not isinstance(rejections, list):
            reasons.append('guard_rejections missing')
            rejections = []
        if receipt.get('valid') is not True:
            reasons += [str(r) for r in (receipt.get('invalid_reasons') or ['valid != true'])]
        elif receipt.get('judge_valid') is not True or rejections:
            # A receipt claiming validity against its own judge or guards is inconsistent: never accepted.
            reasons.append('receipt valid=true contradicts judge_valid/guard_rejections')
        t = metric_value(receipt)
        st = (receipt.get('step_time') or {}).get('mean_window_max_rank_s') if isinstance(receipt.get('step_time'), dict) else None
        if not reasons:
            if t is None:
                reasons.append('no metric value')
            elif st is None or abs(float(st) - t) > 1e-12 * max(1.0, abs(t)):
                reasons.append('step_time.mean_window_max_rank_s != metric.value')
            if (receipt.get('metric') or {}).get('direction') != 'min':
                reasons.append('metric direction must be min')
        valid = not reasons
        pt = metric_value(parent) if isinstance(parent, dict) and parent.get('valid') is True else None
        reward = math.log(pt / t) if valid and pt and t else None
        mem = receipt.get('memory') if isinstance(receipt.get('memory'), dict) else {}
        return Verdict(valid=valid, reasons=reasons, step_time=t, reward=reward,
                       peak_reserved_gib=mem.get('max_peak_reserved_gib'), numerics_class='env',
                       flags=[g.get('id') for g in rejections if isinstance(g, dict)], source=source)
