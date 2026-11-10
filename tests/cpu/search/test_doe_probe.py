"""DOE probe-first (textbook design P1-3c, 2026-10-24): a probe-only plan ends as `probed` and leaves the config open
for a later run; DOE drops the factors a known constraint warns about (no GPU) and the ones whose own 2-update probe
fails, then designs over the rest; `probe_first=False` (the ablation) keeps the old behaviour; `search.ablate` names
are validated."""
import pytest

from infrarsi.project import Project
from infrarsi.project.manifest import ManifestError
from infrarsi.search.engine import Proposal
from infrarsi.search.strategies import DOE
from test_search import TOGGLE, setup


class FixedFactors:
    """A proposer stub: the named boolean knobs as DOE factors, plus level = the other value."""

    def __init__(self, names):
        self.names = names

    def factors(self, engine, base):
        cfg = engine.s.candidates[base].config
        return [{'name': n, 'plus': {n: not cfg.get(n)}} for n in self.names]


FACTORS = ['fused_adam', 'flash_attn', 'tp_overlap', 'compile', 'pin_memory']   # tp_overlap alone crashes (no SP)


def test_probe_only_plan_is_probed_and_the_config_stays_open():
    sim, s, eng = setup(TOGGLE)
    ok = eng.evaluate(Proposal(change={'compile': True}, fidelity=('probe',)))
    assert (ok.status, ok.valid, ok.fidelity) == ('probed', True, 'probe')
    bad = eng.evaluate(Proposal(change={'tp_overlap': True}, fidelity=('probe',)))
    assert bad.status == 'infeasible' and not bad.valid
    again = eng.evaluate(Proposal(change={'compile': True}, fidelity=('short',)))
    assert again.status == 'short' and again.cid == ok.cid          # the probed candidate is reused, not a duplicate
    assert not s.invalid.get('duplicate')
    assert eng.evaluate(Proposal(change={'tp_overlap': True}, fidelity=('short',))).status == 'duplicate'


def test_doe_probes_each_factor_and_drops_the_one_that_crashes():
    sim, s, eng = setup(TOGGLE, budget=30)
    doe = DOE(FixedFactors(FACTORS))
    eng.run(doe)
    d = doe.describe()
    assert d['probed'] == FACTORS
    assert [x['name'] for x in d['dropped']] == ['tp_overlap'] and d['dropped'][0]['why'].startswith('probe:')
    assert d['factors'] == [f for f in FACTORS if f != 'tp_overlap']
    rows = [r for r in s.screens if r['fidelity'] == 'short']
    assert rows and all(r['valid'] for r in rows)                    # no row of the design crashes
    assert d['effects'] and all(v != float('inf') for v in d['effects'].values())


def test_doe_without_probe_first_runs_the_crashing_factor_in_the_design():
    sim, s, eng = setup(TOGGLE, budget=30)
    doe = DOE(FixedFactors(FACTORS), probe_first=False)
    eng.run(doe)
    assert doe.describe()['probed'] == [] and not doe.describe()['dropped']
    assert any(not r['valid'] for r in s.screens if r['fidelity'] == 'short')


def test_doe_drops_warned_factors_without_running_them_and_skips_measured_ones():
    sim, s, eng = setup(TOGGLE, budget=30)
    exp = {'check': lambda cfg: ['c1 (error): compile breaks on this stack'] if cfg.get('compile') else [],
           'coverage': {'fused_adam': {'level': 'covered', 'value': True}}}
    doe = DOE(FixedFactors(['fused_adam', 'flash_attn', 'compile', 'pin_memory', 'overlap_grad']), experience=exp)
    eng.run(doe)
    d = doe.describe()
    assert d['dropped'] == [{'name': 'compile', 'why': 'experience: c1 (error): compile breaks on this stack'}]
    assert d['probed'] == ['flash_attn', 'pin_memory', 'overlap_grad']   # fused_adam: the store ran it here
    assert not any(r.get('candidate_id') and s.candidates[r['candidate_id']].config.get('compile') for r in s.screens)


def test_a_memory_estimate_makes_its_factor_probe_first_never_dropped():
    sim, s, eng = setup(TOGGLE, budget=30)
    est = 'memory (estimate): about 130 GiB of 110 (closed-form accounting, +-25 %)'
    exp = {'check': lambda cfg: [est] if cfg.get('fused_adam') else [], 'soft': lambda w: w.startswith('memory (estimate)'),
           'coverage': {'fused_adam': {'level': 'covered', 'value': True}}}
    doe = DOE(FixedFactors(['fused_adam', 'flash_attn', 'compile', 'pin_memory', 'overlap_grad']), experience=exp)
    eng.run(doe)
    d = doe.describe()
    assert not d['dropped'] and 'fused_adam' in d['factors']
    assert 'fused_adam' in d['probed']                                  # known to run, yet the estimate asks a probe


def test_doe_gives_up_when_too_few_factors_survive():
    sim, s, eng = setup(TOGGLE, budget=30)
    doe = DOE(FixedFactors(['fused_adam', 'tp_overlap', 'flash_attn', 'pin_memory']))   # min_factors 4, one crashes
    eng.run(doe)
    assert doe.state == 'done' and doe.design is None and not [r for r in s.screens if r['fidelity'] == 'short']


def test_search_ablate_is_validated(tmp_path):
    with pytest.raises(ManifestError, match='search.ablate'):
        Project({'name': 'x', 'launch': {'command': 'true'}, 'search': {'ablate': ['bogus']}}, tmp_path / 'infrarsi.yaml')
    p = Project({'name': 'x', 'launch': {'command': 'true'}, 'search': {'ablate': ['probe_first']}},
                tmp_path / 'infrarsi.yaml')
    assert p.raw['search']['ablate'] == ['probe_first']
