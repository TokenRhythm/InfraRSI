"""FABLE_HLS3_FIXES_R1 (reports/infrarsi_newmodels_20261028/design/FABLE_HLS3_FIXES_R1.md): a unit test per fix, each
component switched off by its ablate name gives hls3 as it was (r4). F1 the probe's start cap, speed gate and kept probe;
F2 the coverage bundle; F3 what a spender needs; F4 coupled knobs completed; F5 suspect pairs, the ambiguous death and the
swap; F6 the rest of a partly taken entry; F7 memory warnings; F8 dead elements and inert patches."""
import dataclasses
import json
import math
from types import SimpleNamespace

import pytest

from infrarsi.search import HLS3_COMPONENTS, HLS3_R1
from infrarsi.search.engine import Outcome, Proposal
from infrarsi.search.hls3 import HLS3, SELF_CHECK, proposer_flags, site_keys
from test_composition import FakeProposer
from test_hls3 import G4U_OOM, KNOBS, START, engine, h, observe, run

R1 = list(HLS3_R1)
CRASH = 'completes: exit code 1, 0/30 updates; the log: AssertionError: Node view_8 was invalid'


def stamped(eng, cid: str, *, wall: float, first: float, gaps=(1.0,) * 8, st: float = 1.0) -> None:
    """A receipt with its wall and stamped updates (time to the first update = `first`)."""
    ts = [0.0]
    for g in gaps:
        ts.append(ts[-1] + g)
    ts.append(wall - first)
    eng.s.candidates[cid].receipt = {'evidence': {'wall_s': wall, 'step_time': st, 'per_step': [{'t': t} for t in ts]},
                                     'cost': {'wall_seconds': wall, 'gpu_seconds': 8 * wall}}


def by_origin(props) -> dict:
    out = {}
    for q in props:
        out.setdefault(str(q.origin), []).append(q)
    return out


# ---------------------------------------------------------------- registration
def test_r1_components_registered():
    assert len(R1) == 9 and set(R1) <= set(HLS3_COMPONENTS)
    assert {HLS3_COMPONENTS[c][0] for c in R1} == {f'F{i}' for i in range(1, 9)}
    f = proposer_flags()
    assert f['inert'] and f['coverage'] and not proposer_flags(['hls3_dead_patch', 'hls3_coverage'])['inert']
    assert all(not v for k, v in HLS3(FakeProposer([]), off=R1).on.items() if k in R1)


# ---------------------------------------------------------------- F1
def test_r1_f1a_probe_cap_from_the_base_receipt():
    eng = engine(KNOBS, START)
    stamped(eng, 'c0', wall=200.0, first=100.0)
    hls = HLS3(FakeProposer([h('n2', {'n': 2}, mu=0.05, p=0.5)]), explore=False)
    assert hls._probe_cap(eng, 'c0') == pytest.approx(400.0)        # 4 x 100 s (more than 100 + 2 updates + 60 s)
    q = next(q for q in hls.propose(eng, 16) if q.change == {'n': 2})
    assert 'probe' in q.fidelity and q.probe_cap == pytest.approx(400.0) and q.keep_probe   # start-risky: kept
    off = HLS3(FakeProposer([h('n2', {'n': 2}, mu=0.05, p=0.5)]), explore=False, off=['hls3_start_cap'])
    assert next(q for q in off.propose(eng, 16) if q.change == {'n': 2}).probe_cap is None


def test_r1_f1a_start_slow_is_not_a_start_failure():
    from test_search import TOGGLE, setup
    _, s, _ = setup(TOGGLE)
    s.start_failure_limit = 1
    k1, k2 = [k for k, v in s.candidates['c0'].config.items() if isinstance(v, bool)][:2]
    hang = {'valid': False, 'evidence': {'steps_completed': 0, 'timed_out': True, 'events': {'traceback': '0'},
                                         'per_step': [], 'wall_s': 60.0}, 'cost': {'gpu_seconds': 480.0}}
    s.runner.run = lambda *a, **kw: dict(hang, kw=kw)
    ref = s.candidates['c0'].config
    a = s.propose_config(changes={k1: not ref[k1]}, base='c0')['candidate_id']
    row = s.trial(a, steps=2, fidelity='probe', wall_cap=60.0, detail=True)
    assert row['start_slow'] and s.candidates[a].results['probe']['start_slow'] and not s.unrunnable_values
    b = s.propose_config(changes={k2: not ref[k2]}, base='c0')['candidate_id']
    assert 'start_slow' not in s.trial(b, steps=2, fidelity='probe')   # the job's own timeout: a start failure (C5)
    assert s.value_unrunnable(k2, not ref[k2]) and s.value_unrunnable(k1, not ref[k1]) is None


def test_r1_f1a_a_stopped_probe_is_screened_in_full_only():
    eng = engine(KNOBS, START)
    hls = HLS3(FakeProposer([h('n2', {'n': 2}, mu=0.05, p=0.5)]), explore=False)
    q = next(q for q in hls.propose(eng, 16) if q.change == {'n': 2})
    out = run(eng, 'c1', {'n': 2}, status='infeasible', fidelity='probe',
              reasons=['completes: killed at the timeout after 0/30 updates'])
    eng.s.candidates['c1'].results = {'probe': {'start_slow': True, 'wall_s': 400.0}}
    assert hls.classify(eng, q, out)['kind'] == 'start_slow'
    hls.observe(eng, q, out)
    e = hls.ledger[q.hypothesis]
    assert e['start_slow'] and hls.slow_starts and not hls.together
    off = HLS3(FakeProposer([]), explore=False, off=['hls3_start_cap'])
    assert off.classify(eng, q, out)['kind'] == 'start_slow'        # (classified alike; only hls3_start_cap acts on it)


def test_r1_f1b_gate_stops_a_slow_probe_that_spends():
    eng = engine(KNOBS, START)
    stamped(eng, 'c0', wall=200.0, first=100.0)
    hls = HLS3(FakeProposer([h('n4', {'n': 4}, mu=0.1, mem_dir='spends', p=0.5), h('a', {'a': 1}, mu=0.1)]),
               explore=False, off=['hls3_gate_low'])         # (the round-1 gate, 2.0; F12's 1.6: test_hls3_r2)
    hls.propose(eng, 16)
    run(eng, 'c1', {'n': 4}, status='probed', fidelity='probe')
    run(eng, 'c2', {'a': 1}, status='probed', fidelity='probe')
    assert 'slow start' in hls._gate(eng, 'c1', 'c0', {'intervals': [2.5]})
    assert hls._gate(eng, 'c2', 'c0', {'intervals': [2.5]}) is None and hls.gated[-1]['full_run']   # no spend: noted
    assert hls._gate(eng, 'c1', 'c0', {'intervals': [1.9]}) is None                                  # below the gate
    out = Outcome('c1', 'infeasible', False, None, None, 'probe', 0.1, ['slow start'], ['probe_speed'])
    k = hls.classify(eng, Proposal(change={'n': 4}, base='c0'), out)
    assert k['kind'] == 'slow_probe'                                # no measured evidence of the memory edge (P1-1)
    q = next(q for q in hls.propose(eng, 16) if 'probe' in q.fidelity and tuple(q.fidelity) != ('probe',))
    assert q.gate is not None
    off = HLS3(FakeProposer([h('n4', {'n': 4}, mu=0.1, mem_dir='spends', p=0.5)]), explore=False, off=['hls3_probe_gate'])
    assert all(q.gate is None for q in off.propose(eng, 16))


def test_r1_f1c_kept_probe_priced_at_its_cap_and_the_wall():
    eng = engine(KNOBS, START)
    stamped(eng, 'c0', wall=100.0, first=50.0)
    led = eng.s.ledger
    led.used.update(gpu_hours=0.0)
    led.wall = lambda: 0.0
    led.mean_screen = lambda: (0.3, 100.0)
    hls = HLS3(FakeProposer([]), explore=False, facts={'world': 8})
    eng.s.budget = SimpleNamespace(screens=10, gpu_hours=100.0, wall_seconds=10 * 100.0 + 300.0 + 400.0,
                                   scoring_reserve_screens=2)
    assert hls._screens_scarce(eng, 200.0)                          # 1700 - 200 >= 9 x 100 + 100 x 3
    assert not hls._screens_scarce(eng, 600.0)                      # 1700 - 600 < 1200: the wall cannot pay it
    off = HLS3(FakeProposer([]), explore=False, off=['hls3_start_cap'])
    assert off._screens_scarce(eng, 600.0)                          # r4: the GPU-hours only


def test_r1_f1c_a_start_risky_probe_kept_when_its_expected_saving_pays():
    eng = engine(KNOBS, START)                                      # a full run 0.3 GPU-h, a probe 0.105
    hls = HLS3(FakeProposer([]), explore=False, facts={'world': 8})
    risky = Proposal(change={'n': 2}, base='c0', p_feasible=0.3)
    assert hls.probe_below == 0.8 and hls._probe_pays(eng, risky)   # 0.7 x 0.3 >= 0.105
    assert not hls._probe_pays(eng, dataclasses.replace(risky, p_feasible=0.75))   # 0.25 x 0.3 < 0.105
    assert not hls._probe_pays(eng, dataclasses.replace(risky, p_feasible=0.9))    # not start-risky
    assert not hls._probe_pays(eng, risky, cap=100.0)               # at its cap: 8 x 100 s = 0.22 GPU-h > 0.21
    assert not HLS3(FakeProposer([]), explore=False, off=['hls3_start_cap'])._probe_pays(eng, risky)


# ---------------------------------------------------------------- Codex HLS4 r1 (reviews/HLS4_r1_out.md)
def test_codex_p1_1_a_slow_probe_is_no_memory_evidence():
    """P1-1: a 2.5x probe of a declared spender with a measured 20 GiB peak on an 80 GiB card stops its full run but
    is no thrash: no memory boundary, no measured relation, the same configuration not predicted out of memory. With a
    measured peak near the card it is thrash (memory evidence)."""
    eng = engine(KNOBS, START)
    stamped(eng, 'c0', wall=200.0, first=100.0)
    hls = HLS3(FakeProposer([h('n4', {'n': 4}, mu=0.1, mem_dir='spends', p=0.5)]), explore=False, facts={'cap_gib': 80.0},
               off=['hls3_coverage'])
    q = next(q for q in hls.propose(eng, 16) if q.change == {'n': 4})
    run(eng, 'c1', {'n': 4}, status='infeasible', fidelity='probe', peak=20.0)
    why = hls._gate(eng, 'c1', 'c0', {'intervals': [2.5]})
    out = Outcome('c1', 'infeasible', False, None, None, 'probe', 0.1, [why], ['probe_speed'])
    assert hls.classify(eng, q, out)['kind'] == 'slow_probe'
    hls.observe(eng, q, out)
    assert hls.ledger[q.hypothesis]['status'] == 'invalid' and hls.ledger[q.hypothesis]['failure'] == 'slow_probe'
    assert not hls.mem.measured and 'c1' not in hls.mem.runs
    assert hls._predict(eng, hls._mcfg(eng, 'c0', {'n': 4}))[0] != 'infeasible'
    run(eng, 'c2', {'n': 4, 'z': 1}, status='infeasible', fidelity='probe', peak=75.0)   # 75 of 80 GiB: the edge
    why = hls._gate(eng, 'c2', 'c0', {'intervals': [2.5]})
    out = Outcome('c2', 'infeasible', False, None, None, 'probe', 0.1, [why], ['probe_speed'])
    assert hls.classify(eng, Proposal(change={'n': 4, 'z': 1}, base='c0'), out)['kind'] == 'thrash'


def test_codex_hls4_r2_a_passed_probes_peak_counts_from_its_probe_results():
    """HLS4 r2: SearchSession.trial keeps a passed probe's measured peak in results['probe'] (no full receipt or verdict
    yet), so the speed gate's class reads it there: 20 GiB of 80 stays slow_probe, 75 GiB of 80 is thrash."""
    eng = engine(KNOBS, START)
    stamped(eng, 'c0', wall=200.0, first=100.0)
    hls = HLS3(FakeProposer([h('n4', {'n': 4}, mu=0.1, mem_dir='spends', p=0.5)]), explore=False, facts={'cap_gib': 80.0},
               off=['hls3_coverage'])
    hls.propose(eng, 16)
    for cid, change, gib, kind in (('c1', {'n': 4}, 20.0, 'slow_probe'), ('c2', {'n': 4, 'z': 1}, 75.0, 'thrash')):
        run(eng, cid, change, status='infeasible', fidelity='probe')
        eng.s.candidates[cid].results = {'probe': {'valid': True, 'memory_peak_gib': gib}}   # as SearchSession.trial
        assert 'peak_reserved_gib' not in eng.s.candidates[cid].verdict and not eng.s.candidates[cid].receipt
        why = hls._gate(eng, cid, 'c0', {'intervals': [2.5]})
        out = Outcome(cid, 'infeasible', False, None, None, 'probe', 0.1, [why], ['probe_speed'])
        assert hls.classify(eng, Proposal(change=change, base='c0'), out)['kind'] == kind


def test_codex_p1_2_a_cap_stopped_probe_gets_one_full_retry():
    """P1-2: a probe its own start cap stopped (start_slow) keeps its entry for one full run at p <= 0.3, marked a
    retry: the engine does not drop it as tried, the session reopens its candidate, and only once."""
    eng = engine(KNOBS, START)
    hls = HLS3(FakeProposer([h('n2', {'n': 2}, mu=0.05, p=0.5)]), explore=False, off=['hls3_coverage'])
    q = next(q for q in hls.propose(eng, 16) if q.change == {'n': 2})
    out = run(eng, 'c1', {'n': 2}, status='infeasible', fidelity='probe',
              reasons=['completes: killed at the timeout after 0/30 updates'])
    eng.s.candidates['c1'].results = {'probe': {'start_slow': True, 'wall_s': 400.0}}
    hls.observe(eng, q, out)
    e = hls.ledger[q.hypothesis]
    assert e['status'] == 'open' and e['retry_full'] and e['p_feasible'] <= 0.3
    r = next(x for x in hls.propose(eng, 16) if x.hypothesis == q.hypothesis)
    assert r.fidelity == ('full',) and r.retry and r.p_feasible <= 0.3
    hls.observe(eng, r, run(eng, 'c1', {'n': 2}, reasons=['completes: killed at the timeout after 0/30 updates']))
    assert not e['retry_full'] and e['status'] == 'invalid'         # one retry: it failed in full, it is done
    from test_search import TOGGLE, setup
    _, s, real = setup(TOGGLE)
    k1 = next(k for k, v in s.candidates['c0'].config.items() if isinstance(v, bool))
    hang = {'valid': False, 'evidence': {'steps_completed': 0, 'timed_out': True, 'events': {'traceback': '0'},
                                         'per_step': [], 'wall_s': 60.0}, 'cost': {'gpu_seconds': 480.0}}
    toy = s.runner.run
    s.runner.run = lambda task, cfg, ov, purpose, rid, **kw: dict(hang) if purpose == 'probe' else toy(task, cfg, ov, purpose,
                                                                                                         rid, **kw)
    p = Proposal(change={k1: not s.candidates['c0'].config[k1]}, base='c0', fidelity=('probe', 'full'), keep_probe=True,
                 probe_cap=60.0, mechanism='slow to start')
    first = real.evaluate(p)
    assert first.status == 'infeasible' and s.candidates[first.cid].results['probe']['start_slow']
    retry = dataclasses.replace(p, fidelity=('full',), keep_probe=False, probe_cap=None, retry=True)
    assert real._fresh(retry) and not real._fresh(dataclasses.replace(retry, retry=False))
    second = real.evaluate(retry)
    assert second.cid == first.cid and second.status not in ('duplicate', 'rejected', 'infeasible')
    assert not real._fresh(retry)                                   # once


def test_codex_p1_3_a_passed_third_probe_keeps_its_full_run():
    """P1-3: spender alone out of memory, a freeing-enabler probe out of memory, an unknown-enabler probe passes: three
    tries; its full continuation completes that try and is proposed; a new try of the spender still waits."""
    eng = engine({**KNOBS, 'w': [0, 1]}, {**START, 'w': 0})
    hyps = [h('n4', {'n': 4}, mu=0.12, mem_dir='spends'), h('a frees', {'a': 1}, mu=0.0, mem_dir='frees'),
            h('z', {'z': 1}, mu=0.0)]
    hls = HLS3(FakeProposer(hyps), explore=False, off=['hls3_explore', 'hls3_bundles', 'hls3_coverage'])
    hls.propose(eng, 16)
    observe(hls, eng, 'c1', {'n': 4}, hypothesis='c0|n4', reasons=[G4U_OOM])
    hls.idle = 3
    probes = [q for q in hls.propose(eng, 16) if q.origin == 'hls3-stall-probe']
    assert [q.change for q in probes] == [{'n': 4, 'a': 1}, {'n': 4, 'z': 1}]
    hls.observe(eng, probes[0], run(eng, 'c2', probes[0].change, reasons=[G4U_OOM], fidelity='probe', status='infeasible'))
    hls.observe(eng, probes[1], run(eng, 'c3', probes[1].change, status='probed', failed=(), fidelity='probe'))
    n = next(iter(hls.need.values()))
    assert n['probes'] == 3
    bid = probes[1].hypothesis[len('probe|'):]
    full = next((q for q in hls.propose(eng, 16) if q.hypothesis == bid), None)
    assert full is not None and full.fidelity == ('full',)
    assert hls._capped(eng, 'c0', {'n': 4, 'w': 1})                 # another try of the spender still waits


def test_codex_p1_4_expected_savings_never_override_affordability():
    """P1-4, the accepted round-4 admission scenario: 8 of 10 screens, 2.72 GPU-h cap, 1.60 used, admission mean 0.20,
    a full run 0.30, a probe 0.20, two-screen reserve, p 0.3: keeping the probe would cost screen 10 - not kept. With
    room for it (a 3.0 GPU-h cap) the expected saving keeps it."""
    from hls2_helpers import SimEngine, SimSession
    s = SimSession(START, knobs=KNOBS, screens=10)
    eng = SimEngine(s, None, sigma_runs=[0.01 / math.sqrt(2)], full_costs=[0.3], probe_cost=0.2)
    s.screens += [{'fidelity': 'full', 'gpu_hours': 0.3, 'valid': True}, {'fidelity': 'probe', 'gpu_hours': 0.2, 'valid': True}]
    s.ledger.used.update(screens=8, gpu_hours=1.60)
    s.ledger.screen_costs, s.ledger.prior = [(0.2, 100.0)] * 8, None
    s.ledger.mean_screen, s.ledger.wall = (lambda: (0.2, 100.0)), (lambda: 0.0)
    s.budget = SimpleNamespace(screens=10, gpu_hours=2.72, wall_seconds=None, scoring_reserve_screens=2)
    hls = HLS3(FakeProposer([]), explore=False, off=['hls3_gate_low'])   # (round 1's two-update probe: F12's gated one
    risky = Proposal(change={'n': 2}, base='c0', p_feasible=0.3, fidelity=('probe', 'full'), mechanism='risky')   # costs
    #                                                                   one update more, test_hls3_r2)
    assert hls._probe_pays(eng, risky) and not hls._screens_scarce(eng)
    assert not hls._screen(eng, 'c0', risky).keep_probe            # (the expected saving alone kept it)
    assert hls._admits(eng, 1.60, 0.0) == 2 and hls._admits(eng, 1.80, 0.0) == 1   # the probe would cost screen 10
    s.budget.gpu_hours = 3.0
    assert hls._screen(eng, 'c0', risky).keep_probe


def test_codex_p2_5_an_ambiguous_invariant_group_is_probed_as_one():
    """P2-5: reference {mbs: 1, ga: 2} under mbs x ga = 2; both values of the move {mbs: 2, ga: 1} ambiguous (each
    failure also changed z): one diagnostic probe {mbs: 2, ga: 1}, valid - not {mbs: 2} and {ga: 1}."""
    knobs, start = {'mbs': [1, 2], 'ga': [1, 2], 'z': [0, 1]}, {'mbs': 1, 'ga': 2, 'z': 0}
    eng = engine(knobs, start)
    eng.s.hooks.validate_config = lambda cfg: ({}, [] if cfg['mbs'] * cfg['ga'] == 2 else ['global batch changed'], [])
    hls = HLS3(FakeProposer([]), explore=False, facts={'coupled': [['ga', 'mbs']]})
    run(eng, 'c1', {'mbs': 2, 'ga': 1, 'z': 1}, reasons=[CRASH])
    c, p0 = eng.s.candidates['c1'], eng.s.candidates['c0']
    assert hls._veto('mbs', 2, c, p0) and hls._veto('ga', 1, c, p0)
    probes = hls._alone_probes(eng)
    assert [q.change for q in probes] == [{'mbs': 2, 'ga': 1}]
    assert not eng.s.hooks.validate_config({**start, **probes[0].change})[1]
    hls.observe(eng, probes[0], run(eng, 'c2', probes[0].change, status='infeasible', fidelity='probe', reasons=[CRASH]))
    assert {('mbs', '2'), ('ga', '1')} <= hls.alone_fails             # it failed alone: C5 may rule the values out
    assert not hls._veto('mbs', 2, c, p0)


# ---------------------------------------------------------------- F2
def test_r1_f2_coverage_bundle_of_the_moves_below_the_noise():
    hyps = [h('a', {'a': 1}, mu=0.005), h('z', {'z': 1}, mu=0.004), h('mb', {'mb': 2}, mu=0.006),
            h('n4', {'n': 4}, mu=0.08)]
    eng = engine(KNOBS, START)
    hls = HLS3(FakeProposer(hyps), explore=False, off=['hls3_cov_unresolved'])   # (F2's rule: R3's F17 off)
    o = by_origin(hls.propose(eng, 16))
    cov = o['hls3-coverage'][0]
    assert cov.change == {'a': 1, 'z': 1, 'mb': 2} and cov.mu == pytest.approx(0.02)   # max(sum, 2 sigma_pair)
    assert cov.fidelity == ('probe', 'full') and cov.keep_probe     # probed first: a pair of members may not start
    assert [q.change for q in o['hls'] if len(q.change) == 1] == [{'n': 4}]   # the members withheld
    eng.s.candidates['c0'].verdict = {'valid': True, 'step_time': 1.0}
    hls.observe(eng, cov, run(eng, 'c1', cov.change, status='screened', gain=-0.001, st=1.001, failed=()))
    assert {hls.ledger[k]['status'] for k in hls.members[cov.hypothesis]} == {'open'}   # back to their single screens
    assert all(hls.ledger[k]['coverage']['candidate'] == 'c1' for k in hls.members[cov.hypothesis])
    assert hls.coverage[0]['status'] == 'screened' and 'coverage' in hls.context(eng, 'c0')
    assert 'hls3-coverage' not in by_origin(hls.propose(eng, 16))     # (not bundled again on this incumbent)
    for odd in (h('a', {'a': 1}, mu=0.005, mem_dir='spends'), h('a', {'a': 1}, mu=0.005, p=0.5),
                h('a', {}, mu=0.005, patch={'f.py': 'x\n'}), h('a', {'a': 1}, mu=-0.005)):
        two = HLS3(FakeProposer([odd, h('z', {'z': 1}, mu=0.004)]), explore=False)   # a spender, a start risk, a patch,
        assert 'hls3-coverage' not in by_origin(two.propose(engine(KNOBS, START), 16))   # a predicted loss: no member
    off = HLS3(FakeProposer(hyps), explore=False, off=['hls3_coverage'])
    o = by_origin(off.propose(engine(KNOBS, START), 16))
    assert 'hls3-coverage' not in o and len(o['hls']) == 4


def test_r1_f2_a_coverage_bundle_that_ran_is_not_split():
    knobs = {**KNOBS, 'w': [0, 1]}
    eng = engine(knobs, {**START, 'w': 0})
    eng.s.candidates['c0'].verdict = {'valid': True, 'step_time': 1.0}
    hyps = [h('a', {'a': 1}, mu=0.005), h('z', {'z': 1}, mu=0.004), h('mb', {'mb': 2}, mu=0.006), h('w', {'w': 1}, mu=0.003)]
    hls = HLS3(FakeProposer(hyps), explore=False)
    cov = by_origin(hls.propose(eng, 16))['hls3-coverage'][0]
    inside = set(hls.members[cov.hypothesis])
    assert len(inside) == 4
    hls.observe(eng, cov, run(eng, 'c1', cov.change, status='screened', gain=-0.004, st=1.004, failed=()))
    assert not [k for k in hls.pending if set(hls.members.get(k) or ()) <= inside]   # (hls would test the halves)


# ---------------------------------------------------------------- F3
def test_r1_f3_a_spender_bundle_must_free_more_than_it_failed_by():
    hyps = [h('n4', {'n': 4}, mu=0.12, mem_dir='spends'), h('a frees', {'a': 1}, mu=0.0, mem_dir='frees', mem_gib=1.0)]

    def setup(**kw):
        eng = engine(KNOBS, START)
        hls = HLS3(FakeProposer(hyps), explore=False, **{'off': ['hls3_coverage', 'hls3_stall_probes', 'hls3_explore'] +
                                                        kw.get('off', [])})
        hls.propose(eng, 16)
        observe(hls, eng, 'c1', {'n': 4}, hypothesis='c0|n4', reasons=[G4U_OOM])
        return eng, hls, hls.propose(eng, 16)
    eng, hls, props = setup()
    n = next(iter(hls.need.values()))
    assert n['probes'] == 1 and n['need_lb'] == pytest.approx(4.53)   # its own run: a try and a bound
    assert not [q for q in props if str(q.origin) == 'hls3-bundle']  # 1 GiB freed <= 4.53 short: no bundle
    _, _, props = setup(off=['hls3_need'])
    assert [q for q in props if str(q.origin) == 'hls3-bundle']      # r4: the enabler "covers" it
    n['probes'] = 3                                                 # tried three times: a holder of n=4 waits
    assert hls._capped(eng, 'c0', {'n': 4, 'z': 1}) and hls._capped(eng, 'c0', {'z': 1}) is None
    hls.mem.relate(eng.s.candidates['c0'].config, {**START, 'a': 1}, 'frees')
    run(eng, 'c2', {'a': 1}, status='screened', gain=0.01, st=0.99, failed=())
    n['last'] = 'c0'
    assert hls._capped(eng, 'c2', {'n': 4}) is None                 # a measured freeing move since: tried again


def test_r1_f3_a_change_in_a_declared_spending_combination_is_no_unknown_enabler():
    eng = engine({**KNOBS, 'w': [0, 1]}, {**START, 'w': 0})
    hyps = [h('n4', {'n': 4}, mu=0.12, mem_dir='spends'), h('w and a', {'w': 1, 'a': 1}, mu=0.1, mem_dir='spends'),
            h('a frees', {'a': 1}, mu=0.0, mem_dir='frees'), h('w', {'w': 1}, mu=0.0), h('z', {'z': 1}, mu=0.0)]
    hls = HLS3(FakeProposer(hyps), explore=False)
    hls.propose(eng, 16)
    observe(hls, eng, 'c1', {'n': 4}, hypothesis='c0|n4', reasons=[G4U_OOM])
    unknown = [e['change'] for e, _ in hls._enablers(eng, 'c0', hls.ledger['c0|n4'], unknown=True) if not e.get('mem_dir')]
    assert {'w': 1} not in unknown and {'z': 1} in unknown           # w spends with a, which frees: w is the spender
    hls._kn = None
    assert hls._known(eng).get(frozenset({('w', '0', '1')})) == 'spends'   # (merged by move: derived)
    off = HLS3(FakeProposer(hyps), explore=False, off=['hls3_need'])
    off.propose(eng, 16)
    assert {'w': 1} in [e['change'] for e, _ in off._enablers(eng, 'c0', off.ledger['c0|n4'], unknown=True)]


# ---------------------------------------------------------------- F4
def test_r1_f4_coupled_knobs_completed():
    knobs, start = {'mbs': [1, 2], 'ga': [1, 2], 'a': [0, 1]}, {'mbs': 1, 'ga': 2, 'a': 0}
    eng = engine(knobs, start)
    eng.s.hooks.validate_config = lambda cfg: ({}, [] if cfg['mbs'] * cfg['ga'] == 2 else ['global batch changed'], [])
    facts = {'coupled': [['ga', 'mbs']]}
    hyps = [h('mbs2', {'mbs': 2}, mu=0.05), h('mbs2 ga1', {'mbs': 2, 'ga': 1}, mu=0.04)]
    hls = HLS3(FakeProposer(hyps[:1]), explore=False, facts=facts)
    assert [q.change for q in hls.propose(eng, 16)] == [{'mbs': 2, 'ga': 1}]
    dup = HLS3(FakeProposer(hyps), explore=False, facts=facts)       # completed, it is another entry's move
    assert [q.change for q in dup.propose(eng, 16)] == [{'mbs': 2, 'ga': 1}]
    assert dup.ledger['c0|mbs2']['status'] == 'superseded'
    off = HLS3(FakeProposer(hyps[:1]), explore=False, facts=facts, off=['hls3_complete'])
    assert [q.change for q in off.propose(eng, 16)] == [{'mbs': 2}]


# ---------------------------------------------------------------- F5
def test_r1_f5_suspect_pairs_by_configuration():
    def setup(**kw):
        eng = engine(KNOBS, START)
        hls = HLS3(FakeProposer([h('a and z', {'a': 1, 'z': 1}, mu=0.08), h('a z mb', {'a': 1, 'z': 1, 'mb': 2}, mu=0.06),
                                 h('n2', {'n': 2}, mu=0.01)]), explore=False,
                   **{'off': ['hls3_antagonism', 'hls3_coverage', 'hls3_dead_patch'] + kw.get('off', [])})
        hls.propose(eng, 16)
        observe(hls, eng, 'c1', {'a': 1, 'z': 1, 'n': 2}, reasons=[CRASH])   # three changes, none an entry
        return eng, hls
    eng, hls = setup()
    az = frozenset({frozenset({('a', '1')}), frozenset({('z', '1')})})
    assert len(hls.suspects) == 3 and all(v['strikes'] == 1 for v in hls.suspects.values())   # every pair of the three
    q = next(q for q in hls.propose(eng, 16) if q.hypothesis == 'c0|a and z')
    assert 'probe' in q.fidelity and q.keep_probe                   # one strike: probed first (the start cap bounds it)
    observe(hls, eng, 'c2', {'a': 1, 'z': 1}, hypothesis='c0|a and z', reasons=[CRASH], fidelity='probe',
            status='infeasible')
    assert hls.suspects[az]['strikes'] == 2                         # a second failed start that held the pair
    assert 'c0|a z mb' not in {q.hypothesis for q in hls.propose(eng, 16)}
    assert hls.ledger['c0|a z mb']['deferred'].startswith('it holds')
    eng.s.candidates['c0'].verdict = {'valid': True, 'step_time': 1.0}
    observe(hls, eng, 'c3', {'a': 1, 'n': 2}, status='screened', gain=0.0, st=1.0, failed=())   # (a, n2) started together
    assert frozenset({frozenset({('a', '1')}), frozenset({('n', '2')})}) not in hls.suspects and az in hls.suspects
    _, off = setup(off=['hls3_pairs'])
    assert not off.suspects


def test_r1_f5_a_value_that_failed_only_with_a_partner_is_probed_alone():
    eng = engine(KNOBS, START)
    eng.s.start_failure_limit = 2
    hls = HLS3(FakeProposer([h('a and z', {'a': 1, 'z': 1}, mu=0.08)]), explore=False)
    hls.propose(eng, 16)
    observe(hls, eng, 'c1', {'a': 1, 'z': 1}, reasons=[CRASH])
    ev = {'steps_completed': 0, 'timed_out': False, 'events': {'traceback': '1'}}
    verdict = SimpleNamespace(reasons=[CRASH])
    for cid in ('c1', 'c1'):
        eng.s._note_start_failure(eng.s.candidates[cid], eng.s.candidates['c0'], ev, 'AssertionError', verdict)
    assert eng.s.value_unrunnable('a', 1) is None and ('a', '1') in hls.ambiguous   # the veto kept it
    alone = by_origin(hls.propose(eng, 16)).get('hls3-alone-probe')
    assert alone and alone[0].change == {'a': 1} and alone[0].fidelity == ('probe',) and alone[0].base == 'c0'
    out = run(eng, 'c2', {'a': 1}, status='infeasible', fidelity='probe', reasons=[CRASH])
    hls.observe(eng, alone[0], out)
    assert ('a', '1') in hls.alone_fails
    assert not hls._veto('a', 1, eng.s.candidates['c2'], eng.s.candidates['c0'])    # a failure alone: C5 rules it out
    off = HLS3(FakeProposer([]), explore=False, off=['hls3_pairs'])
    assert not off._veto('a', 1, eng.s.candidates['c1'], eng.s.candidates['c0'])


def test_r1_f5_swap_out_the_incumbent_member():
    eng = engine(KNOBS, START)
    hls = HLS3(FakeProposer([h('a', {'a': 1}, mu=0.05), h('z', {'z': 1}, mu=0.1)]), explore=False,
               off=['hls3_coverage'])
    hls.propose(eng, 16)
    eng.s.candidates['c0'].verdict = {'valid': True, 'step_time': 1.0}
    observe(hls, eng, 'c1', {'a': 1}, hypothesis='c0|a', status='screened', gain=0.03, st=0.97, failed=())
    eng._inc = 'c1'
    hls.ledger['c0|a'].update(status='confirmed', measured=0.03)
    hls.suspects[frozenset({frozenset({('a', '1')}), frozenset({('z', '1')})})] = {'run': 'cX', 'kind': 'crash'}
    hls.ledger['c1|z'] = {**h('z', {'z': 1}, mu=0.1), 'base': 'c1', 'status': 'open', 'key': 'c1|z'}
    swap = by_origin(hls.propose(eng, 16)).get('hls3-swap')
    assert swap and swap[0].change == {'z': 1, 'a': 0} and swap[0].mu == pytest.approx(0.1 * hls.kappa() - 0.03)
    assert hls.swaps[-1]['proposed'] and hls.swaps[-1]['without'] == 'a=1'


# ---------------------------------------------------------------- F6, F7
def test_r1_f6_the_rest_of_a_partly_taken_entry():
    def setup(**kw):
        eng = engine(KNOBS, START)
        hls = HLS3(FakeProposer([h('a and z', {'a': 1, 'z': 1}, mu=0.08), h('a', {'a': 1}, mu=0.06)]), explore=False,
                   **{'off': ['hls3_coverage'] + kw.get('off', [])})
        hls.propose(eng, 16)
        eng.s.candidates['c0'].verdict = {'valid': True, 'step_time': 1.0}
        observe(hls, eng, 'c1', {'a': 1}, hypothesis='c0|a', status='screened', gain=0.03, st=0.97, failed=())
        eng._inc = 'c1'
        hls.propose(eng, 16)
        return hls
    hls = setup()
    rest = [e for e in hls.ledger.values() if e.get('remainder_of') == 'c0|a and z']
    assert rest and rest[0]['change'] == {'z': 1} and rest[0]['base'] == 'c1'
    assert rest[0]['mu'] == pytest.approx(0.08 - math.log(1 / 0.97))   # less what the taken part measured
    assert not [e for e in setup(off=['hls3_remainder']).ledger.values() if e.get('remainder_of')]


def test_r1_f7_a_memory_warning_without_out_of_memory_evidence():
    eng = engine(KNOBS, START)
    hls = HLS3(FakeProposer([]), explore=False)
    cfg = eng.s.candidates['c0'].config
    assert hls._warn_p(eng, cfg, {'n': 4}, 0.9, ['memory (estimate): 120 GiB of 80']) == 0.9 and hls.warn_kept == 1
    assert hls._warn_p(eng, cfg, {'n': 4}, 0.9, ['oom_limit (out of memory)']) == 0.2        # a constraint: as before
    hls.mem.record('c1', {**cfg, 'n': 4}, 'oom')
    hls.mem.relate(cfg, {**cfg, 'n': 4}, 'spends')
    assert hls._warn_p(eng, cfg, {'n': 4}, 0.9, ['memory (estimate): 120 GiB of 80']) == 0.2  # hls3's runs bound it
    off = HLS3(FakeProposer([]), explore=False, off=['hls3_warn'])
    assert off._warn_p(eng, cfg, {'n': 3}, 0.9, ['memory (estimate): 120 GiB of 80']) == 0.2


# ---------------------------------------------------------------- F8
HOOK = 'PreTrainedModel.gradient_checkpointing_enable = _wrapped_enable'


def test_r1_f8_the_same_fatal_line_twice_rules_out_the_code_they_shared():
    line = "completes: exit code 1, 0/30 updates; the log: TypeError: enable() got an unexpected keyword argument 'k'"

    def setup(**kw):
        eng = engine(KNOBS, START)
        p1 = {'train_patches.py': f'def _wrapped_enable(self, **kw):\n    return 1\n{HOOK}\nfirst_idea_body_line = 1\n'}
        p2 = {'train_patches.py': f'def _wrapped_enable(self, **kw):\n    return 2\n{HOOK}\nsecond_idea_body_line = 2\n'}
        p3 = {'train_patches.py': f'{HOOK}\nthird_idea_body_line_x = 3\n'}
        p4 = {'train_patches.py': 'unrelated_patch_body_line = 4\n'}
        hls = HLS3(FakeProposer([h('one', {}, patch=p1), h('two', {}, patch=p2), h('three', {}, patch=p3, mu=0.1),
                                 h('four', {}, patch=p4, mu=0.1)]), explore=False, **kw)
        hls.propose(eng, 16)
        for cid, key, p in (('c1', 'c0|one', p1), ('c2', 'c0|two', p2)):
            out = run(eng, cid, {}, status='infeasible', fidelity='probe', reasons=[line])
            hls.observe(eng, Proposal(change={}, patch=p, base='c0', hypothesis=key), out)
        return hls, {q.hypothesis for q in hls.propose(eng, 16)}
    hls, props = setup()
    assert ('line', ' '.join(HOOK.split())) in hls.dead_elements
    assert 'c0|three' not in props and hls.ledger['c0|three']['why'].startswith('unrunnable') and 'c0|four' in props
    _, props = setup(off=['hls3_dead_patch'])
    assert 'c0|three' in props


def test_r1_f8_an_inert_patch_demotes_its_site():
    site = 'the Experts.forward eager loop of every layer'
    eng = engine(KNOBS, START)
    eng.s.candidates['c0'].verdict = {'valid': True, 'step_time': 1.0}
    hyps = [h('first', {}, patch={'f.py': 'x = 1\n'}, mu=0.05, site=site),
            h('second', {}, patch={'f.py': 'y = 2\n'}, mu=0.05, site='same Experts.forward loop, another way'),
            h('checked', {}, patch={'f.py': "z = 3\nprint('hook ran')\n"}, mu=0.05, site='Experts.forward again')]
    hls = HLS3(FakeProposer(hyps), explore=False, off=['hls3_coverage'])
    first = next(q for q in hls.propose(eng, 16) if q.hypothesis == 'c0|first')
    hls.observe(eng, first, run(eng, 'c1', {}, status='screened', gain=0.001, st=0.999, failed=()))
    assert hls.inert and hls.inert[0]['candidate'] == 'c1' and 'inert_patches' in hls.context(eng, 'c0')
    mus = {q.hypothesis: q.mu for q in hls.propose(eng, 16)}
    assert mus['c0|second'] < 0 and mus['c0|checked'] > 0           # a self-check line keeps its place
    assert site_keys(site) == {'experts.forward'} and SELF_CHECK.search("print('hook ran')")
    off = HLS3(FakeProposer(hyps), explore=False, off=['hls3_coverage', 'hls3_dead_patch'])
    first = next(q for q in off.propose(eng, 16) if q.hypothesis == 'c0|first')
    off.observe(eng, first, run(eng, 'c2', {}, status='screened', gain=0.001, st=0.999, failed=()))
    assert all(q.mu > 0 for q in off.propose(eng, 16))
