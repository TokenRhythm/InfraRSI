"""Phase D round 1 fixes in the search (2026-10-24): the same-failure stop compares the log's error too (F-D1-06-1),
and a change's loss deviation is also measured from its base, the run it was applied to (F-D1-14-1)."""
import dataclasses

import pytest

from infrarsi.harness import signing
from infrarsi.search.engine import Proposal
from test_search import TOGGLE, setup


def _with_errors(monkeypatch, s, errors):
    run, it = s.runner.run, iter(errors)

    def failing(*a, **kw):
        r = run(*a, **kw)
        if not r['valid']:
            r['evidence'].setdefault('events', {})['error'] = next(it)
            signing.sign(r)
        return r
    monkeypatch.setattr(s.runner, 'run', failing)


def test_the_same_failure_needs_the_same_error(monkeypatch):
    sim, s, eng = setup(TOGGLE)
    s.stop_after_same_failures = 3
    _with_errors(monkeypatch, s, ['torch.OutOfMemoryError: CUDA out of memory',
                                  'torch.distributed.DistNetworkError: address already in use'] * 3)
    for t in ('fused_adam', 'flash_attn', 'pin_memory', 'compile'):     # tp_overlap without sp crashes the toy
        eng.evaluate(Proposal(change={'tp_overlap': True, t: True}))
    assert s.stop_flag is None                                           # two causes, alternating: no stop
    sim, s, eng = setup(TOGGLE)
    s.stop_after_same_failures = 3
    _with_errors(monkeypatch, s, ['torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB'] * 3)
    for t in ('fused_adam', 'flash_attn', 'pin_memory'):
        eng.evaluate(Proposal(change={'tp_overlap': True, t: True}))
    assert s.stop_flag == 'same_failure_x3'                               # the same cause three times


def test_a_change_is_also_measured_from_its_base():
    sim, s, eng = setup(dataclasses.replace(TOGGLE, antagonism=0.0))     # T4: fused_ce + compile moves the loss 6 %
    trap = eng.evaluate(Proposal(change={'fused_ce': True, 'compile': True}, base='c0'))
    assert trap.loss_deviation == pytest.approx(0.06, rel=0.05) and trap.loss_deviation_base == trap.loss_deviation
    child = eng.evaluate(Proposal(change={'fused_adam': True}, base=trap.cid))
    assert child.loss_deviation == pytest.approx(0.06, rel=0.05)         # from the reference: the base's effect
    assert child.loss_deviation_base == pytest.approx(0.0, abs=1e-9)     # from its base: its own effect is none
