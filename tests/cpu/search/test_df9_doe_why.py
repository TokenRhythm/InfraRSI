"""D2-R201b (2026-10-24): `--strategy doe` on a 16-GPU job with a 12 GPU-h budget - the factor probes and the smallest
screening design did not fit the 35 % of the budget a design may use, DOE set itself done, and the run ended no_change
after 0 runs with nothing said. DOE now records why no design ran; the log and the report say it."""
from infrarsi.search.strategies import DOE
from test_doe_probe import FACTORS, FixedFactors
from test_search import TOGGLE, setup


def test_a_design_that_does_not_fit_the_budget_says_so():
    lines = []
    sim, s, eng = setup(TOGGLE, budget=2, log=lines.append)
    doe = DOE(FixedFactors(FACTORS))
    eng.run(doe)
    assert doe.state == 'done' and doe.design is None and not s.screens
    assert 'raise search.budget.gpu_hours' in doe.describe()['why']
    assert any(m.startswith('[search] doe: no screening design: ') for m in lines)



def test_too_few_factors_say_so():
    sim, s, eng = setup(TOGGLE, budget=30)
    doe = DOE(FixedFactors(['fused_adam', 'flash_attn']))
    eng.run(doe)
    assert doe.describe()['why'] == 'the knobs give 2 two-level factors (a design needs 4)'
