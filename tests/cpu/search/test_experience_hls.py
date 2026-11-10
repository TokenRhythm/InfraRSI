"""Experience in HLS (experience decision 2.1-2.2, 6.6): seeded entries are measurements, not predictions (no kappa, not
in kappa's pairs; an LLM twin is merged and still calibrated); the start composite goes through the bundle path (probe
first, confirmed, split when it fails, its members measured alone); open seeds follow the incumbent; known-to-run moves
skip the probe, warned ones are probe-first."""
from types import SimpleNamespace

import pytest

from infrarsi.search.engine import Outcome
from infrarsi.search.strategies import HLS
from test_composition import FakeEngine, FakeProposer, hyp


def row(knob, gain, level='covered', **kw):
    return {'knob': knob, 'mechanism': f'm{knob}', 'change': {knob: 1}, 'level': level, 'gain': gain,
            'evidence_ids': [f'pack/{knob}'], 'warnings': [], **kw}


def out(cid, gain, status='screened', fidelity='full', failed=()):
    return Outcome(cid, status, status == 'screened', 1.0 if status == 'screened' else None, gain, fidelity, 0.1, [], list(failed))


def by(props, hypothesis):
    return next(p for p in props if p.hypothesis == hypothesis)


def test_seeds_are_not_scaled_by_kappa_nor_calibrate_it():
    eng = FakeEngine()
    hls = HLS(FakeProposer([hyp('b', 0.04)]), explore=False, experience={'seeds': [row('a', 0.05)]})
    hls.pairs = [(0.04, 0.0)] * 10                                  # a badly calibrated model: kappa near its floor
    props = hls.propose(eng, 16)
    seed, llm = by(props, 'c0|experience:ma'), by(props, 'c0|b')
    assert seed.mu == pytest.approx(0.05) and seed.origin == 'hls-experience' and seed.fidelity == ('full',)
    assert llm.mu == pytest.approx(hls.kappa() * 0.04) and llm.mu < 0.01
    hls.observe(eng, seed, out('c1', 0.06))
    assert hls.ledger['c0|experience:ma']['status'] == 'confirmed' and len(hls.pairs) == 10


def test_an_llm_twin_is_merged_and_still_calibrated():
    eng = FakeEngine()
    hls = HLS(FakeProposer([hyp('a', 0.03)]), explore=False, experience={'seeds': [row('a', 0.05)]})
    props = hls.propose(eng, 16)
    assert [p.hypothesis for p in props] == ['c0|experience:ma']     # no second entry for the same change
    hls.observe(eng, props[0], out('c1', 0.05))
    assert hls.pairs == [(pytest.approx(0.03), 0.05)]


def composite_hls():
    seeds = [row(k, g) for k, g in zip('abcd', (0.04, 0.03, 0.02, 0.01))]
    comp = {'members': list('abcd'), 'change': {k: 1 for k in 'abcd'}, 'gain': 0.1, 'warnings': []}
    return HLS(FakeProposer([]), explore=False, experience={'seeds': seeds, 'composite': comp})


def test_the_start_composite_is_one_probed_confirmed_bundle():
    eng, hls = FakeEngine(), composite_hls()
    b = by(hls.propose(eng, 16), 'bundle|exp')
    assert b.change == {k: 1 for k in 'abcd'} and b.origin == 'experience-bundle' and b.confirm
    assert b.fidelity == ('probe', 'full') and b.p_feasible == 0.7 and b.mu == pytest.approx(0.10)
    assert b.sigma == pytest.approx(2 * max(eng.sigma_pair(), 0.02))


def test_a_failed_composite_is_split_and_its_members_stay_open():
    eng, hls = FakeEngine(), composite_hls()
    b = by(hls.propose(eng, 16), 'bundle|exp')
    hls.observe(eng, b, out(None, None, 'invalid', failed=['loss_within_tolerance']))
    assert all(hls.ledger[f'c0|experience:m{k}']['status'] == 'open' for k in 'abcd')
    assert sorted(sorted(p.change) for p in hls.pending.values()) == [['a', 'c'], ['b', 'd']]   # by prior, alternating
    assert hls.exp_fate['status'] == 'invalid' and not hls.exp_fate['moved']
    singles = {p.hypothesis for p in hls.propose(eng, 16) if p.origin == 'hls-experience'}
    assert singles == {f'c0|experience:m{k}' for k in 'abcd'}       # each can still be measured alone


def test_a_moved_composite_retires_its_members():
    eng, hls = FakeEngine(), composite_hls()
    b = by(hls.propose(eng, 16), 'bundle|exp')
    eng.s.candidates['c7'] = SimpleNamespace(config={'a': 1, 'b': 1, 'c': 1, 'd': 1, 'x': 0}, overlay={})
    eng._inc = 'c7'
    hls.observe(eng, b, out('c7', 0.09))
    assert {hls.ledger[f'c0|experience:m{k}']['status'] for k in 'abcd'} == {'bundled'}
    assert hls.describe()['experience']['bundled'] == 4 and hls.exp_fate['moved']
    assert not [p for p in hls.propose(eng, 16) if p.origin == 'hls-experience']   # all four are in the incumbent


def test_open_seeds_follow_the_incumbent():
    eng = FakeEngine()
    eng.s.candidates['c1'] = SimpleNamespace(config={'a': 1, 'b': 0, 'c': 0, 'd': 0, 'x': 0}, overlay={})
    hls = HLS(FakeProposer([]), explore=False, experience={'seeds': [row('a', 0.05), row('b', 0.01), row('c', -0.02)]})
    props = hls.propose(eng, 16)
    hls.observe(eng, by(props, 'c0|experience:mc'), out('c2', -0.03))  # measured: not seeded again
    hls.observe(eng, by(props, 'c0|experience:ma'), out('c1', 0.06))
    eng._inc = 'c1'
    assert {p.hypothesis for p in hls.propose(eng, 16)} == {'c1|experience:mb'}


def test_partial_rows_take_the_judgments_sign_and_feasibility():
    eng = FakeEngine()
    seeds = [row('a', 0.04, 'partial'), row('b', 0.04, 'partial', sign='-', p_feasible=0.95),
             row('c', 0.04, 'partial', sign='unknown'), row('d', 0.04, 'partial', sign='infeasible', p_feasible=0.6)]
    hls = HLS(FakeProposer([]), explore=False, experience={'seeds': seeds})
    p = {x.hypothesis[3:]: x for x in hls.propose(eng, 16)}
    assert p['experience:ma'].mu == pytest.approx(0.02) and p['experience:ma'].p_feasible == 0.7
    assert p['experience:mb'].mu == pytest.approx(-0.02) and p['experience:mb'].p_feasible == 0.9
    assert p['experience:mc'].mu == 0.0
    assert p['experience:md'].p_feasible == 0.2 and p['experience:md'].fidelity == ('probe', 'full')


def test_known_moves_skip_the_probe_and_warned_ones_are_probed():
    eng = FakeEngine()
    eng.s.task = SimpleNamespace(knob_schema={'properties': {'a': {'type': 'integer', 'minimum': 0, 'maximum': 1},
                                                             'b': {'type': 'integer', 'minimum': 0, 'maximum': 1}}})
    exp = {'known_ok': [{'a': 1}], 'check': lambda cfg: ['memory: does not fit'] if cfg.get('b') == 1 else []}
    hls = HLS(FakeProposer([]), experience=exp)
    p = {x.hypothesis: x for x in hls.propose(eng, 16)}
    assert p['c0|explore:a=1'].p_feasible == 0.9 and p['c0|explore:a=1'].fidelity == ('full',)
    assert p['c0|explore:b=1'].p_feasible == 0.2 and p['c0|explore:b=1'].fidelity == ('probe', 'full')
    hls2 = HLS(FakeProposer([dict(hyp('b'), change={'b': 1})]), explore=False, experience=exp)
    h = by(hls2.propose(eng, 16), 'c0|b')
    assert h.p_feasible == 0.2 and hls2.ledger['c0|b']['warnings'] == ['memory: does not fit']


def test_the_transfer_hook_runs_once_before_seeding():
    eng, calls = FakeEngine(), []
    exp = {'seeds': []}

    def transfer(session):
        calls.append(session)
        exp['seeds'].append(row('a', 0.03, 'partial', sign='+'))
    exp['transfer'] = transfer
    hls = HLS(FakeProposer([]), explore=False, experience=exp)
    assert [p.hypothesis for p in hls.propose(eng, 16)] == ['c0|experience:ma']
    eng._inc = 'c9'
    hls.propose(eng, 16)
    assert calls == [eng.s]


def test_r9_4_a_seed_is_not_carried_to_another_starting_point():
    eng = FakeEngine()                                              # c9 changes x, the "layout" knob of seed b
    hls = HLS(FakeProposer([]), explore=False, experience={'seeds': [row('a', 0.03, valid_on={'a': 0}),
                                                                     row('b', 0.03, valid_on={'b': 0, 'x': 0})]})
    hls.propose(eng, 16)
    eng._inc = 'c9'
    assert {p.hypothesis for p in hls.propose(eng, 16)} == {'c9|experience:ma'}


def test_r9_9_neighbourhood_probing_is_explicit():
    eng = FakeEngine()
    eng.s.task = SimpleNamespace(knob_schema={'properties': {'a': {'type': 'integer', 'minimum': 0, 'maximum': 1}}})
    plain = {p.hypothesis: p for p in HLS(FakeProposer([]), probe_below=0.5).propose(eng, 16)}
    assert plain['c0|explore:a=1'].fidelity == ('probe', 'full')    # as before experience, whatever probe_below is
    known = {p.hypothesis: p for p in HLS(FakeProposer([]), probe_below=0.95, experience={'known_ok': [{'a': 1}]}).propose(eng, 16)}
    assert known['c0|explore:a=1'].fidelity == ('full',)


def test_r10_3_a_known_move_skips_the_probe_when_proposed_again():
    eng = FakeEngine()
    eng.s.task = SimpleNamespace(knob_schema={'properties': {k: {'type': 'integer', 'minimum': 0, 'maximum': 1} for k in 'ab'}})
    hls = HLS(FakeProposer([]), probe_below=0.95, experience={'known_ok': [{'a': 1}]})
    first = {p.hypothesis: p for p in hls.propose(eng, 16)}
    hls.observe(eng, first['c0|explore:b=1'], out('c2', -0.01))     # measured, the incumbent stays
    again = {p.hypothesis: p for p in hls.propose(eng, 16)}
    assert again['c0|explore:a=1'].fidelity == ('full',)


@pytest.mark.parametrize('mode,mu', [('partial', 0.015), ('full', 0.03), ('none', None)])
def test_carried_seeds_are_partial_by_default(mode, mu):
    eng = FakeEngine()
    exp = {'seeds': [row('a', 0.05), row('b', 0.03)], **({} if mode == 'partial' else {'reseed': mode})}
    hls = HLS(FakeProposer([]), explore=False, experience=exp)
    p = by(hls.propose(eng, 16), 'c0|experience:ma')
    assert p.mu == pytest.approx(0.05)                               # on the first base: the source gain
    eng._inc = 'c9'
    carried = {x.hypothesis: x for x in hls.propose(eng, 16)}.get('c9|experience:mb')
    if mu is None:
        assert carried is None
    else:
        assert carried.mu == pytest.approx(mu)
        assert hls.ledger['c9|experience:mb']['transfer'] == ('carried' if mode == 'partial' else 'covered')


def test_r11_a_first_base_moved_by_the_design_is_later():
    eng = FakeEngine()
    eng._inc = 'c9'                                                 # doe+hls: HLS starts on the design's incumbent
    hls = HLS(FakeProposer([]), explore=False, experience={'seeds': [row('a', 0.04)]})
    p = by(hls.propose(eng, 16), 'c9|experience:ma')
    assert p.mu == pytest.approx(0.02) and hls.ledger['c9|experience:ma']['transfer'] == 'carried'
