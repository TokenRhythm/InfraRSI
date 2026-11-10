"""Lower-fidelity runs keep the job's schedule horizon (C1 #5 / C2 #4, Fable 5.1 design 2026-10-24): a probe or short run
gets the full {steps} and is stopped after its updates, so a decaying learning rate is compared like with like; a
numerics break is still caught at the probe; a short run is timed against the same updates of its base. Plus the two
engine fixes of the same review: a configuration measured short only still gets its full screen (C2 #5), and a patch
on an already screened configuration is not a duplicate (C2 #6)."""
import dataclasses
import math

import pytest

from infrarsi.search.engine import Proposal
from infrarsi.search.strategies import DOE
from test_doe_probe import FixedFactors
from test_search import TOGGLE, setup


def test_probe_keeps_the_schedule_horizon():
    sim, s, eng = setup(TOGGLE, runner_kw={'lr_decay': True})
    out = eng.evaluate(Proposal(change={'fused_adam': True}, p_feasible=0.3, fidelity=('probe', 'full')))
    assert out.status == 'screened'
    probe = next(r for r in s.screens if r['fidelity'] == 'probe')
    assert probe['valid'] and probe['steps'] == 2
    row = next(r for r in s.runner.log if r.get('stop_after') == 2)
    assert (row['steps'], row['done']) == (30, 2)
    # the defect, reachable only without the horizon: a 2-update schedule has another learning rate
    cid = s.propose_config(changes={'flash_attn': True}, base='c0', origin='t')['candidate_id']
    old = s.trial(cid, steps=2, fidelity='probe')
    assert not old['valid'] and 'optimizer_steps' in s.candidates[cid].results['probe']['failed_checks']


def test_numerics_break_is_still_caught_at_the_probe():
    sim, s, eng = setup(dataclasses.replace(TOGGLE, antagonism=0.0), runner_kw={'lr_decay': True})
    out = eng.evaluate(Proposal(change={'fused_ce': True, 'compile': True}, p_feasible=0.3, fidelity=('probe', 'full')))
    assert (out.status, out.fidelity) == ('infeasible', 'probe')
    assert 'loss_within_tolerance' in out.failed_checks and s.ledger.used['screens'] == 0


def test_short_runs_compare_with_the_same_updates_of_the_base():
    sim, s, eng = setup(TOGGLE)
    out = eng.evaluate(Proposal(change={'overlap_grad': True}, fidelity=('short',)))
    base = eng.t_window('c0', [eng.warmup + 1, eng.short_steps])
    assert base != eng.t('c0')                                   # the early updates are slower than the full window
    assert out.gain == pytest.approx(math.log(base / out.step_time))
    doe = DOE(FixedFactors(['fused_adam', 'flash_attn', 'pin_memory', 'compile']), probe_first=False)
    eng.run(doe)
    assert doe.y[len(doe.design) - 1] == base                    # the all-minus row: the base over the rows' updates


def test_a_config_measured_short_still_gets_its_full_screen():
    sim, s, eng = setup(TOGGLE)
    short = eng.evaluate(Proposal(change={'flash_attn': True}, fidelity=('short',)))
    assert short.status == 'short'
    p = Proposal(change={'flash_attn': True}, fidelity=('full',))
    assert eng.key(p) not in eng.tried                           # not taken for the short run's duplicate
    full = eng.evaluate(p)
    assert full.status == 'screened' and full.cid == short.cid


def test_a_patch_on_a_screened_config_is_not_a_duplicate():
    sim, s, eng = setup(TOGGLE)
    knob = eng.evaluate(Proposal(change={'fused_adam': True}, base='c0'))
    assert knob.status == 'screened'
    both = eng.evaluate(Proposal(change={'fused_adam': True}, patch={'patches.txt': 'prefetch_data\n'}, base='c0'))
    assert both.status == 'screened' and both.cid != knob.cid
    assert s.candidates[both.cid].parent == knob.cid
