"""hls3 (search/hls3.py; FABLE_TAKEAWAYS_20261030 section 4): a unit test per component - S-1 the memory feasibility
model and the kept probe, the proposer's memory fields, S-2 spender + enabler bundles, S-3 failure classes, thrash and the
antagonism table, S-4 stall probes, S-5 re-basing, S-6 the whole-configuration step, S-7 the phase re-rank, S-8 the code
sites, S-9 the engine's robust mode and stall removers, S-10 card-size pruning and the sharded accounting - each switched
off by its ablate name; registration (strategy, ablations, CLI help); every component off is hls2; and a regression test
for each finding of the Codex review round 1 (reviews/HLS3_r1_out.md: P1-1 .. P2-6 and the declarations' design)."""
import json
import math
from types import SimpleNamespace

import pytest

from hls2_helpers import SCENARIOS, SimEngine, SimSession, run_scenario
from infrarsi.search import ABLATIONS, HLS3_COMPONENTS, HLS3_R1, LLM_STRATEGIES, STRATEGIES, resolve
from infrarsi.search.engine import Engine, Outcome, Proposal, median_update
from infrarsi.search.hls3 import HLS3, MemoryModel, parse_oom, sharded_accounting, sharded_floor, sig
from infrarsi.search.strategies import HLS2
from test_composition import FakeProposer
from test_hls_reask import Script

ALL = sorted(HLS3_COMPONENTS)
R1 = list(HLS3_R1)                                                  # FABLE_HLS3_FIXES_R1's components (off: hls3 as of r4)
G4U_OOM = ('completes: exit code 1, 0/30 updates; the log: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate '
           '8.00 GiB. GPU 1 has a total capacity of 139.81 GiB of which 3.47 GiB is free. Including non-PyTorch memory, this '
           'process has 136.33 GiB memory in use. Of the allocated memory 134.37 GiB is allocated by PyTorch, and 436.24 MiB '
           'is reserved by PyTorch but unallocated.')          # G4U-code-env-opus-h2 c4 (recompute off at micro-batch 2)


def h(name, change, mu=0.05, p=0.9, patch=None, **kw):
    return {'mechanism': name, 'change': change, 'patch': patch, 'mu': mu, 'sigma': 0.02, 'p_feasible': p, **kw}


def engine(knobs: dict, start: dict, *, screens: int = 10, sp: float = 0.01) -> SimEngine:
    s = SimSession(start, knobs=knobs, screens=screens)
    eng = SimEngine(s, None, sigma_runs=[sp / math.sqrt(2)])
    return eng


def run(eng, cid, change, *, base='c0', status='invalid', reasons=(), gain=None, failed=('completes',), fidelity='full',
        peak=None, st=None):
    """A recorded outcome on `eng`: the candidate (its config = base + change) and the Outcome for observe()."""
    b = eng.s.candidates[base]
    cfg = {**b.config, **change}
    verdict = {'valid': status == 'screened', 'step_time': st} if status == 'screened' else {'valid': False}
    if peak is not None:
        verdict['peak_reserved_gib'] = peak
    eng.s.candidates[cid] = SimpleNamespace(id=cid, parent=base, config=cfg, overlay=dict(b.overlay), verdict=verdict,
                                            status='screened', results={}, receipt=None)
    if status == 'screened':
        eng.s.ledger.used['screens'] += 1
    return Outcome(cid, status, status == 'screened', st, gain, fidelity, 0.3, list(reasons), list(failed))


def observe(hls, eng, cid, change, *, hypothesis=None, base='c0', **kw):
    out = run(eng, cid, change, base=base, **kw)
    hls.observe(eng, Proposal(change=change, base=base, hypothesis=hypothesis), out)
    return out


KNOBS = {'n': [1, 2, 3, 4], 'mb': [1, 2], 'a': [0, 1], 'z': [0, 1]}
START = {'n': 1, 'mb': 1, 'a': 0, 'z': 0}


# ---------------------------------------------------------------- S-3: failure classes
def test_parse_oom_reads_the_failed_allocation():
    o = parse_oom(G4U_OOM)
    assert o['tried_gib'] == 8.0 and o['capacity_gib'] == 139.81 and o['free_gib'] == 3.47
    assert o['short_gib'] == pytest.approx(4.53) and o['need_gib'] == pytest.approx(144.33)
    assert o['reserved_gib'] == pytest.approx(436.24 / 1024, abs=1e-3)
    mib = parse_oom('Tried to allocate 968.00 MiB. GPU 3 has a total capacity of 139.81 GiB of which 613.69 MiB is free.')
    assert mib['short_gib'] == pytest.approx((968 - 613.69) / 1024, abs=1e-3)
    assert parse_oom('RuntimeError: Triton Error [CUDA]: out of memory') is None and parse_oom(None) is None


def test_classify_every_failure_class():
    eng = engine(KNOBS, START)
    hls = HLS3(FakeProposer([]), facts={'cap_gib': 139.81})

    def kind(reasons, status='invalid', failed=('completes',), **kw):
        out = run(eng, f'c{len(eng.s.candidates)}', {'a': 1}, status=status, reasons=reasons, failed=failed, **kw)
        return hls.classify(eng, Proposal(change={'a': 1}, base='c0'), out)
    k = kind([G4U_OOM])
    assert k['kind'] == 'oom' and k['oom']['short_gib'] == pytest.approx(4.53)
    assert kind(['completes: exit code 1, 0/30 updates; the log: RuntimeError: Triton Error [CUDA]: out of memory'])['kind'] == 'oom'
    autotune = 'torch.OutOfMemoryError: out of resource: shared memory, Required: 131072, Hardware limit: 101376.'
    assert kind(['completes: killed at the timeout after 0/30 updates', autotune])['kind'] == 'start_cap'   # not an OOM
    assert kind(['completes: killed at the timeout after 15/30 updates'])['kind'] == 'hang'
    assert kind(['completes: exit code 1, 0/30 updates; the log: torch._inductor.exc.InductorError: RuntimeError: '
                 'PyTorch is checking whether allow_tf32_new is enabled'])['kind'] == 'compiler'
    assert kind(['completes: exit code 1, 3/30 updates; the log: torch.distributed.DistBackendError: NCCL error in: '
                 'ProcessGroupNCCL.cpp'])['kind'] == 'collective'
    assert kind(['loss_within_tolerance: window-mean deviation 4.1 %'], failed=('loss_within_tolerance',))['kind'] == 'numerics'
    assert kind(['completes: exit code 1, 0/30 updates; the log: AssertionError: Node view_828 was invalid, but is '
                 'output'])['kind'] == 'crash'
    eng.s.candidates['c0'].verdict = {'valid': True, 'step_time': 1.0}
    thrash = kind([], status='screened', failed=(), gain=-0.6, st=1.82, peak=131.0)
    assert thrash['kind'] == 'thrash' and thrash['slowdown'] == pytest.approx(1.822, abs=1e-3)
    assert kind([], status='screened', failed=(), gain=-0.6, st=1.82, peak=60.0)['kind'] == 'ok'   # slow, far from the cap
    assert kind([], status='screened', failed=(), gain=-0.1, st=1.1, peak=131.0)['kind'] == 'ok'   # near the cap, not slow


# ---------------------------------------------------------------- S-1: the memory model
def test_memory_model_dominance_scale_and_reverts():
    ref, oom = {'n': 1, 'mb': 1, 'a': 0}, {'n': 4, 'mb': 1, 'a': 0}
    m = MemoryModel(80.0)
    m.record('c0', ref, 'ok')
    m.record('c1', oom, 'oom')
    m.relate(ref, oom, 'spends')                                    # n 1 -> 4 ran out of memory: the scale grows
    known = dict(m.measured)
    known[sig(ref, {**ref, 'mb': 2})] = 'spends'                    # the proposer: micro-batch 2 spends
    known[sig(ref, {**ref, 'a': 1})] = 'frees'
    assert m.predict({'n': 4, 'mb': 2, 'a': 0}, known, ref)[0] == 'infeasible'       # needs at least what c1 needed
    assert m.predict({'n': 3, 'mb': 2, 'a': 0}, known, ref)[0] == 'unsure'           # n 4 -> 3 frees (the scale)
    assert m.predict({'n': 4, 'mb': 1, 'a': 1}, known, ref)[0] == 'unsure'           # c1 plus an enabler
    assert m.predict({'n': 1, 'mb': 1, 'a': 1}, known, ref)[0] == 'feasible'         # the reference minus memory
    assert m.predict({'n': 1, 'mb': 1, 'a': 0, 'z': 1}, known, ref)[0] is None       # unknown, and not near c1:
    assert m._knob('n', '2', '3', known) == 'spends' and m._knob('n', '4', '2', known) == 'frees'   # it reverts n
    two = {**ref, 'mb': 2, 'g': 1}
    known[sig(ref, two)] = 'spends'                                 # a two-knob move matched whole (micro-batch + accum.)
    assert m.split(ref, two, known)['spends'] == [sig(ref, two)]
    assert m.split(two, ref, known)['frees'] == [sig(two, ref)]


def test_feasibility_defers_probes_first_and_keeps_the_probe():
    def setup(**kw):
        eng = engine(KNOBS, START)
        eng.s.screens = [{'fidelity': 'probe', 'gpu_hours': 0.2}, {'fidelity': 'full', 'gpu_hours': 0.3, 'valid': True}]
        hyps = [h('n4', {'n': 4}, mu=0.1, mem_dir='spends'), h('n4 mb2', {'n': 4, 'mb': 2}, mu=0.15),
                h('n3', {'n': 3}, mu=0.07, p=0.85), h('mb2', {'mb': 2}, mu=0.04, mem_dir='spends'),
                h('a', {'a': 1}, mu=0.01, mem_dir='frees', mem_gib=3.0)]
        hls = HLS3(FakeProposer(hyps), explore=False, **kw)
        hls.propose(eng, 16)
        observe(hls, eng, 'c1', {'n': 4}, hypothesis='c0|n4', reasons=[G4U_OOM])
        return eng, hls, {p.hypothesis: p for p in hls.propose(eng, 16)}
    eng, hls, props = setup()
    nm = props['c0|n4 mb2']             # c1 plus micro-batch 2, which only the proposer says spends: probed, not deferred
    assert nm.fidelity == ('probe', 'full') and nm.p_feasible == 0.4 and not hls.ledger['c0|n4 mb2'].get('deferred')
    assert hls.ledger['c0|n4 mb2']['memory'].startswith('unsure: by the declared moves') and hls.mem.cap == 139.81
    n3 = props['c0|n3']                                             # nearly bounded by c1: probed first, the probe kept
    assert n3.fidelity == ('probe', 'full') and n3.keep_probe and n3.p_feasible == 0.4
    assert eng.costs()['probe'] >= 0.5 * eng.costs()['full'] and eng.plan(n3) == ('probe', 'full')
    assert eng.plan(Proposal(fidelity=('probe', 'full'))) == ('full',)   # the engine's rule for any other proposal
    assert props['c0|a'].fidelity == ('full',) and not props['c0|a'].keep_probe          # fits: the reference minus memory
    observe(hls, eng, 'c2', {'mb': 2}, hypothesis='c0|mb2', reasons=[G4U_OOM])   # micro-batch 2 ran out of memory too:
    props = {p.hypothesis: p for p in hls.propose(eng, 16)}        # now measured, c1 + it is ruled out
    assert 'c0|n4 mb2' not in props and hls.ledger['c0|n4 mb2']['deferred'].startswith('predicted out of memory')
    assert hls.ledger['c0|n4 mb2']['short_gib'] == pytest.approx(4.53)
    d = hls.describe()['hls3']
    assert d['counts']['deferred'] >= 1 and d['failures'][0]['class'] == 'oom' and 'short by at least 4.53' in \
        d['failures'][0]['detail']
    view = hls.context(eng, 'c0')['memory_view']
    assert view['capacity_gib'] == 139.81 and {'change': {'n': 4, 'mb': 2}, 'why': hls.ledger['c0|n4 mb2']['deferred']} \
        in view['deferred']
    eng, hls, props = setup(off=['hls3_feasibility'])               # off: proposed as hls2 does, the probe as the engine
    assert 'c0|n4 mb2' in props and not props['c0|n3'].keep_probe and props['c0|n3'].fidelity == ('full',)


def test_a_declaration_never_rules_a_configuration_out():
    """Codex r1 design answer: the declarations order and build, at most ask for a probe. A configuration that holds an
    out-of-memory run's change plus moves only the proposer calls spending is probed first; the same configuration with
    those moves measured is deferred; a run that fitted with it makes it feasible whatever is declared."""
    ref, oom = {'n': 1, 'mb': 1, 'a': 0}, {'n': 4, 'mb': 1, 'a': 0}
    m = MemoryModel(80.0)
    m.record('c0', ref, 'ok')
    m.record('c1', oom, 'oom')
    m.relate(ref, oom, 'spends')
    known = {**m.measured, sig(ref, {**ref, 'mb': 2}): 'spends'}   # declared: micro-batch 2 spends
    cfg = {'n': 4, 'mb': 2, 'a': 0}
    assert m.predict(cfg, known, ref, hard=m.measured)[0] == 'unsure'
    assert m.predict(cfg, known, ref)[0] == 'infeasible'           # (taken as measured: ruled out)
    m.record('c2', cfg, 'ok')                                      # it ran and fitted: no estimate overrides that
    assert m.predict(cfg, known, ref)[0] == 'feasible'


def test_a_probe_is_kept_while_the_gpu_hours_left_pay_every_screen_left():
    """The kept probe's rule: GPU-hours left (less the probe) >= screens left x a full run. A comparison of the shares
    left switched it off after the first probe, before any screen (the memory toy's out-of-memory screens came from such
    dropped probes); and it is off when the GPU-hours would run out first."""
    eng = engine(KNOBS, START)
    eng.s.screens = [{'fidelity': 'probe', 'gpu_hours': 0.2}, {'fidelity': 'full', 'gpu_hours': 0.3, 'valid': True}]
    eng.s.ledger.used.update(screens=0, gpu_hours=0.2)              # one probe, no screen yet
    hls = HLS3(FakeProposer([]), explore=False)
    eng.s.budget.gpu_hours = 12.0                                   # 40 full runs for 10 screens: the screens bind
    assert hls._screens_scarce(eng)                                 # (shares: 1.0 screens left > 0.983 GPU-hours)
    eng.s.budget.gpu_hours = 3.2                                    # 2.8 left after this probe: not 10 x 0.3
    assert not hls._screens_scarce(eng)
    del eng.s.budget.gpu_hours                                      # no GPU-hour cap: the screens are the limit
    assert hls._screens_scarce(eng)


def test_the_kept_probe_leaves_the_scoring_reserve_for_the_last_screen():
    """Codex HLS3 r2: admission keeps scoring_reserve_screens screen costs for the scoring pair, so the kept probe's
    rule prices the last screen the same way. 9 of 10 screens, 4.00 GPU-hours, 3.05 used, a full run 0.30, a probe
    0.20: the last screen is admissible before the probe (3.05 + 0.30 x 3 = 3.95) and not after it (3.25 + 0.90)."""
    from infrarsi.agent.budget import Budget, SearchLedger
    eng = engine(KNOBS, START)
    eng.s.screens = [{'fidelity': 'probe', 'gpu_hours': 0.2}] + [{'fidelity': 'full', 'gpu_hours': 0.3, 'valid': True}] * 9
    hls = HLS3(FakeProposer([]), explore=False)

    def session(gpu_hours):
        eng.s.budget = Budget(screens=10, llm_tokens=10 ** 6, wall_seconds=10 ** 6, gpu_hours=gpu_hours)
        eng.s.ledger = led = SearchLedger(eng.s.budget)
        for _ in range(9):
            led.charge_screen(0.3, 10.0)
        led.charge_screen(0.35, 1.0, count=False)                   # earlier probes: 3.05 used in all
        return led
    led = session(4.0)
    assert led.admit_screen() is None                               # the last screen fits now ...
    assert not hls._screens_scarce(eng)                             # ... so this probe is not kept
    led.charge_screen(0.2, 1.0, count=False)
    assert led.admit_screen() == 'gpu_hours'                        # (a kept probe would have cost that screen)
    led = session(4.2)
    assert hls._screens_scarce(eng)                                 # 0.95 left after the probe pays 0.30 x (1 + 2)
    led.charge_screen(0.2, 1.0, count=False)
    assert led.admit_screen() is None


def test_the_kept_probe_prices_the_last_screen_at_the_full_cost_when_failed_screens_were_cheap():
    """Codex HLS3 r3: admission averages every counted screen (cheap failed ones too) while the engine's full cost
    averages valid full screens, and the full runs still to come raise the admission mean. 8 of 10 screens (four
    failed at 0.10, four valid at 0.30), 2.72 GPU-hours, 1.60 used, a probe 0.20: priced at the mean (0.20) the probe
    looked affordable, yet after it and screen 9 the mean is 0.211 and screen 10 needs 0.633 with 0.62 left."""
    from infrarsi.agent.budget import Budget, SearchLedger
    eng = engine(KNOBS, START)
    eng.s.screens = ([{'fidelity': 'probe', 'gpu_hours': 0.2}] + [{'fidelity': 'full', 'gpu_hours': 0.1}] * 4
                     + [{'fidelity': 'full', 'gpu_hours': 0.3, 'valid': True}] * 4)
    eng.s.budget = Budget(screens=10, llm_tokens=10 ** 6, wall_seconds=10 ** 6, gpu_hours=2.72)
    eng.s.ledger = led = SearchLedger(eng.s.budget)
    for cost in [0.1] * 4 + [0.3] * 4:
        led.charge_screen(cost, 10.0)
    hls = HLS3(FakeProposer([]), explore=False)
    assert led.admit_screen() is None and abs(led.used['gpu_hours'] - 1.6) < 1e-9
    assert not hls._screens_scarce(eng)                             # 0.92 left after the probe < 0.30 + 0.30 x 3
    led.charge_screen(0.2, 1.0, count=False)                        # what keeping it would have done:
    led.charge_screen(0.3, 10.0)                                    # screen 9 at the predicted full cost ...
    assert led.admit_screen() == 'gpu_hours'                        # ... and screen 10 is refused


def test_the_proposers_declarations_switch_off():
    eng = engine(KNOBS, START)
    hls = HLS3(FakeProposer([h('mb2', {'mb': 2}, mem_dir='spends'), h('n4', {'n': 4})]), explore=False,
               off=['hls3_memory_schema'])
    hls.propose(eng, 16)
    observe(hls, eng, 'c1', {'mb': 2, 'n': 2}, reasons=[G4U_OOM])
    assert hls._known(eng) == hls.mem.measured                      # the runs' only: the declaration is ignored
    on = HLS3(FakeProposer([h('mb2', {'mb': 2}, mem_dir='spends')]), explore=False)
    on.propose(eng, 16)
    assert sig({**START}, {**START, 'mb': 2}) in on._known(eng)


# ---------------------------------------------------------------- S-2: bundles
def test_a_blocked_spender_is_bundled_with_the_best_enabler():
    def setup(**kw):
        eng = engine(KNOBS, START)
        hyps = [h('n4', {'n': 4}, mu=0.12, p=0.9, mem_dir='spends', mem_gib=10.0),
                h('a frees', {'a': 1}, mu=0.0, mem_dir='frees', mem_gib=6.0),
                h('z frees', {'z': 1}, mu=0.0, mem_dir='frees', mem_gib=2.0)]
        hls = HLS3(FakeProposer(hyps), explore=False, **kw)
        hls.propose(eng, 16)
        observe(hls, eng, 'c1', {'n': 4}, hypothesis='c0|n4', reasons=[G4U_OOM])
        return eng, hls, hls.propose(eng, 16)
    eng, hls, props = setup()
    assert hls.ledger['c0|n4']['status'] == 'invalid' and hls.ledger['c0|n4']['failure'] == 'oom'
    b = next(p for p in props if p.origin == 'hls3-bundle')
    assert b.change == {'n': 4, 'a': 1} and b.fidelity == ('probe', 'full') and b.keep_probe   # 6 GiB covers the 4.53
    row = hls.describe()['hls3']['bundles'][0]
    assert row['spender_mechanism'] == 'n4' and row['enablers'] == ['a frees'] and row['short_gib'] == pytest.approx(4.53)
    eng.s.candidates['c0'].verdict = {'valid': True, 'step_time': 1.0}
    eng._inc = 'c2'
    out = run(eng, 'c2', {'n': 4, 'a': 1}, status='screened', gain=0.1, st=0.905, failed=())
    hls.observe(eng, b, out)
    assert hls.ledger['c0|n4']['status'] == 'bundled' and hls.ledger['c0|a frees']['status'] == 'bundled'
    assert hls.describe()['hls3']['bundles'][0]['moved'] is True
    _, off, props = setup(off=['hls3_bundles'])
    assert not [p for p in props if p.origin == 'hls3-bundle'] and not off.describe()['hls3']['bundles']


# ---------------------------------------------------------------- S-3: thrash, the summary, the antagonism table
def test_thrash_is_the_memory_edge_and_the_proposer_sees_the_failures():
    def setup(**kw):
        eng = engine(KNOBS, START)
        eng.s.candidates['c0'].verdict = {'valid': True, 'step_time': 1.0}
        hls = HLS3(FakeProposer([h('n3', {'n': 3}, mu=0.1, mem_dir='spends')]), explore=False,
                   facts={'cap_gib': 140.0}, **kw)
        hls.propose(eng, 16)
        observe(hls, eng, 'c1', {'n': 3}, hypothesis='c0|n3', status='screened', gain=-0.7, st=2.01, failed=(), peak=133.0)
        return eng, hls
    eng, hls = setup()
    e = hls.ledger['c0|n3']
    assert e['status'] == 'invalid' and e['failure'] == 'thrash' and not hls.pairs     # no calibration pair
    assert hls.mem.runs['c1']['status'] == 'thrash'
    f = hls.context(eng, 'c0')['failures'][0]
    assert f['class'] == 'thrash' and f['change'] == {'n': 3} and 'x2.014 slower' in f['detail']
    eng, hls = setup(off=['hls3_failures'])
    assert hls.ledger['c0|n3']['status'] == 'refuted' and 'failures' not in hls.context(eng, 'c0')


def test_changes_that_failed_together_run_separately_first():
    def setup(**kw):
        eng = engine(KNOBS, START)
        hyps = [h('a and z', {'a': 1, 'z': 1}, mu=0.08), h('a z n2', {'a': 1, 'z': 1, 'n': 2}, mu=0.1),
                h('a alone', {'a': 1}, mu=0.03)]
        hls = HLS3(FakeProposer(hyps), explore=False, **{'off': R1, **kw})   # (F5 pairs: test_r1_f5_*)
        hls.propose(eng, 16)
        observe(hls, eng, 'c1', {'a': 1, 'z': 1}, hypothesis='c0|a and z',
                reasons=['completes: exit code 1, 0/30 updates; the log: AssertionError: Node view_8 was invalid'])
        return eng, hls, {p.hypothesis: p for p in hls.propose(eng, 16)}
    eng, hls, props = setup()
    t = hls.together[0]
    assert t['kind'] == 'crash' and sorted(map(sorted, t['members'])) == [[('a', '1')], [('z', '1')]]
    assert 'c0|a z n2' not in props and hls.ledger['c0|a z n2']['deferred'].startswith('failed together')
    sep = {k: v for k, v in hls.ledger.items() if v.get('separately_of')}
    assert sorted(json.dumps(v['change']) for v in sep.values()) == ['{"a": 1}', '{"z": 1}']
    assert 'failed_together' in hls.context(eng, 'c0')
    eng.s.candidates['c0'].verdict = {'valid': True, 'step_time': 1.0}
    observe(hls, eng, 'c2', {'a': 1}, status='screened', gain=0.01, st=0.99, failed=())
    observe(hls, eng, 'c3', {'z': 1}, status='screened', gain=0.0, st=1.0, failed=())
    assert set(t['separately']) == set(t['members'])                # both ran alone: they may be combined again
    assert 'c0|a z n2' in {p.hypothesis for p in hls.propose(eng, 16)}
    _, off, props = setup(off=['hls3_antagonism'] + R1)
    assert not off.together and 'c0|a z n2' in props


def test_a_failed_bundle_keeps_the_knobs_an_invariant_ties_in_one_member():
    """Codex r1 P1-2: an experience composite with micro-batch and accumulation as separate seeds crashes; the knobs the
    global-batch invariant ties stay one member - its separate run sets both (a legal run), and once each member ran
    without the others the set may be combined again (two members would have blocked it for good)."""
    knobs, start = {'mb': [1, 2], 'ga': [2, 1], 'z': [0, 1]}, {'mb': 1, 'ga': 2, 'z': 0}
    rows = [{'knob': k, 'mechanism': k, 'change': {k: v}, 'level': 'partial', 'gain': 0.0, 'sign': '+', 'p_feasible': 0.8,
             'evidence_ids': [], 'warnings': []} for k, v in (('mb', 2), ('ga', 1), ('z', 1))]
    exp = {'seeds': rows, 'composite': {'members': ['mb', 'ga', 'z'], 'change': {'mb': 2, 'ga': 1, 'z': 1}, 'gain': None,
                                        'warnings': []}}
    eng = engine(knobs, start)
    eng.s.candidates['c0'].verdict = {'valid': True, 'step_time': 1.0}
    hls = HLS3(FakeProposer([]), explore=False, experience=exp, facts={'coupled': [['ga', 'mb']]})
    hls.propose(eng, 16)
    comp = hls.pending['bundle|exp']
    hls.observe(eng, comp, run(eng, 'c1', comp.change, reasons=[
        'completes: exit code 1, 0/30 updates; the log: AssertionError: Node view_8 was invalid']))
    t = hls.together[0]
    assert sorted(map(sorted, t['members'])) == [[('ga', '1'), ('mb', '2')], [('z', '1')]]
    hls._separately(eng, 'c0', t)
    sep = sorted(json.dumps(v['change'], sort_keys=True) for v in hls.ledger.values() if v.get('separately_of'))
    assert sep == ['{"ga": 1, "mb": 2}', '{"z": 1}']                # micro-batch and accumulation run together
    observe(hls, eng, 'c2', {'mb': 2, 'ga': 1}, status='screened', gain=0.01, st=0.99, failed=())
    observe(hls, eng, 'c3', {'z': 1}, status='screened', gain=0.0, st=1.0, failed=())
    assert hls._blocked(hls._mcfg(eng, 'c0', comp.change)) is None  # each ran alone: no longer blocked


def test_a_slowdown_is_a_memory_boundary_only_with_evidence():
    """Codex r1 P1-4: without a measured peak, a run 2x slower than its base is thrash only when an out-of-memory run
    bounds it through measured moves; a declared spend is no evidence - the slow run stays a (noted) slowdown, fits in
    the memory model, and defers nothing (the repro: compile + a memory-neutral flag was 'predicted out of memory')."""
    eng = engine(KNOBS, START)
    eng.s.candidates['c0'].verdict = {'valid': True, 'step_time': 1.0}
    hls = HLS3(FakeProposer([h('compile', {'a': 1}, mu=0.1, mem_dir='spends'),
                             h('compile and z', {'a': 1, 'z': 1}, mu=0.12, mem_dir='spends')]), explore=False,
               facts={'cap_gib': 140.0})
    hls.propose(eng, 16)
    observe(hls, eng, 'c1', {'a': 1}, hypothesis='c0|compile', status='screened', gain=-0.7, st=2.01, failed=())
    e = hls.ledger['c0|compile']
    assert e.get('failure') is None and e['memory'].startswith('possible thrash: x2.014')
    assert hls.mem.runs['c1']['status'] == 'ok' and hls.counts['possible_thrash'] == 1 and not hls.failures
    props = {p.hypothesis: p for p in hls.propose(eng, 16)}
    assert 'c0|compile and z' in props and not hls.ledger['c0|compile and z'].get('deferred')
    observe(hls, eng, 'c2', {'n': 4}, reasons=[G4U_OOM])           # n 1 -> 4 ran out of memory (measured): n 3 is close
    observe(hls, eng, 'c3', {'n': 3}, status='screened', gain=-0.7, st=2.01, failed=())
    assert hls.failures[-1]['class'] == 'thrash' and hls.mem.runs['c3']['status'] == 'thrash'


def test_bundle_enablers_go_together_and_a_retry_is_larger():
    """Codex r1 P2-6: two values of one knob are not counted together - the bundle takes the next enabler that goes with
    the ones it holds, records only what it kept, and after it runs out of memory the retry adds one more."""
    def setup(short):
        eng = engine({**KNOBS, 'w': [0, 1, 2]}, {**START, 'w': 0})
        hyps = [h('n4', {'n': 4}, mu=0.12, mem_dir='spends', mem_gib=10.0),
                h('w1', {'w': 1}, mu=0.0, mem_dir='frees', mem_gib=8.0), h('w2', {'w': 2}, mu=0.0, mem_dir='frees', mem_gib=6.0),
                h('a', {'a': 1}, mu=0.0, mem_dir='frees', mem_gib=5.0)]
        hls = HLS3(FakeProposer(hyps), explore=False, off=['hls3_stall_probes', 'hls3_explore'])
        hls.propose(eng, 16)
        observe(hls, eng, 'c1', {'n': 4}, hypothesis='c0|n4', reasons=[
            f'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate {short + 2:.2f} GiB. GPU 0 has a total capacity '
            'of 139.81 GiB of which 2.00 GiB is free.'])
        return eng, hls, [p for p in hls.propose(eng, 16) if p.origin == 'hls3-bundle']
    eng, hls, (b,) = setup(10.0)                                    # 8 + 5 cover the 10 GiB short (not 8 + 6 of one knob)
    row = hls.describe()['hls3']['bundles'][0]
    assert b.change == {'n': 4, 'w': 1, 'a': 1} and row['enablers'] == ['w1', 'a'] and row['freed_gib_declared'] == 13.0
    eng, hls, (b,) = setup(7.0)                                     # one enabler covers 7 GiB
    assert b.change == {'n': 4, 'w': 1}
    hls.observe(eng, b, run(eng, 'c2', b.change, reasons=[G4U_OOM]))
    retry = [p for p in hls.propose(eng, 16) if p.origin == 'hls3-bundle']
    assert [p.change for p in retry] == [{'n': 4, 'w': 1, 'a': 1}]    # larger: w2 (another value of w) passed over
    assert hls.describe()['hls3']['bundles'][1]['enablers'] == ['w1', 'a']


# ---------------------------------------------------------------- S-4: stall probes
def test_a_stall_probes_the_spender_with_each_enabler():
    def setup(**kw):
        eng = engine({**KNOBS, 'w': [0, 1]}, {**START, 'w': 0})
        hyps = [h('n4', {'n': 4}, mu=0.12, mem_dir='spends'), h('a frees', {'a': 1}, mu=0.0, mem_dir='frees'),
                h('z', {'z': 1}, mu=0.01), h('w', {'w': 1}, mu=0.0)]
        hls = HLS3(FakeProposer(hyps), explore=False, **{'off': ['hls3_explore', 'hls3_bundles', 'hls3_pair_probes'] + R1,
                                                          **kw})        # (F16 off: it acts at the out of memory, before S-4)
        hls.propose(eng, 16)
        observe(hls, eng, 'c1', {'n': 4}, hypothesis='c0|n4', reasons=[G4U_OOM])
        hls.idle = 3                                                # three ideas tested since the incumbent changed
        return eng, hls, hls.propose(eng, 16)
    eng, hls, props = setup()
    assert [p.origin for p in props] == ['hls3-stall-probe'] * 3 and all(p.fidelity == ('probe',) for p in props)
    assert [p.change for p in props][0] == {'n': 4, 'a': 1}         # the declared enabler first, then the unknown ones
    assert sorted(json.dumps(p.change, sort_keys=True) for p in props[1:]) == ['{"n": 4, "w": 1}', '{"n": 4, "z": 1}']
    assert hls.stalls[0]['probes'][0] == 'a frees' and hls.idle == 0
    first = props[0]
    out = run(eng, 'c2', first.change, status='probed', failed=(), fidelity='probe')
    hls.observe(eng, first, out)                                    # it fits: one full screen, no second probe
    bid = first.hypothesis[len('probe|'):]
    assert hls.pending[bid].fidelity == ('full',) and hls.pending[bid].p_feasible == 0.95
    second = props[1]
    hls.observe(eng, second, run(eng, 'c3', second.change, reasons=[G4U_OOM], fidelity='probe', status='infeasible'))
    assert frozenset(hls.members[second.hypothesis[len('probe|'):]]) in hls.evaluated
    _, off, props = setup(off=['hls3_explore', 'hls3_bundles', 'hls3_stall_probes', 'hls3_pair_probes'] + R1)
    assert not [p for p in props if p.origin == 'hls3-stall-probe'] and not off.stalls


# ---------------------------------------------------------------- S-5: re-basing
def test_rejected_moves_are_queued_again_on_a_new_incumbent():
    def setup(**kw):
        eng = engine(KNOBS, START)
        eng.s.candidates['c0'].verdict = {'valid': True, 'step_time': 1.0}
        hyps = [h('n2', {'n': 2}, mu=0.06), h('mb2', {'mb': 2}, mu=0.04), h('z', {'z': 1}, mu=0.01), h('a', {'a': 1}, mu=0.1)]
        hls = HLS3(Script([hyps, []]), explore=False, **kw)     # the model does not propose them again on c4
        hls.propose(eng, 16)
        for cid, (key, ch) in zip(('c1', 'c2', 'c3'), (('n2', {'n': 2}), ('mb2', {'mb': 2}), ('z', {'z': 1}))):
            observe(hls, eng, cid, ch, hypothesis=f'c0|{key}', status='screened', gain=-0.02, st=1.02, failed=())
        observe(hls, eng, 'c4', {'a': 1}, hypothesis='c0|a', status='screened', gain=0.1, st=0.905, failed=())
        eng._inc = 'c4'
        hls.propose(eng, 16)
        return hls
    hls = setup()
    re = {k: v for k, v in hls.ledger.items() if v.get('rebased_from')}
    assert sorted(json.dumps(v['change']) for v in re.values()) == ['{"mb": 2}', '{"n": 2}']   # the two strongest
    assert re['c4|rebase:n2']['mu'] == pytest.approx(0.03) and re['c4|rebase:n2']['base'] == 'c4'
    assert [r['move'] for r in hls.describe()['hls3']['rebased']] == ['n2', 'mb2']
    assert not [v for v in setup(off=['hls3_rebase']).ledger.values() if v.get('rebased_from')]


# ---------------------------------------------------------------- S-6: one whole-configuration step
def test_a_stall_runs_the_composite_without_the_pair_that_failed_together():
    rows = [{'knob': k, 'mechanism': k, 'change': {k: v}, 'level': 'partial', 'gain': 0.0, 'sign': '+', 'p_feasible': 0.8,
             'evidence_ids': [], 'warnings': []} for k, v in (('a', 1), ('z', 1), ('mb', 2))]
    exp = {'seeds': rows, 'composite': {'members': ['a', 'z', 'mb'], 'change': {'a': 1, 'z': 1, 'mb': 2}, 'gain': None,
                                        'warnings': []}}

    def setup(**kw):
        eng = engine(KNOBS, START)
        hls = HLS3(FakeProposer([h('az', {'a': 1, 'z': 1}, mu=0.05)]), explore=False, experience=exp,
                   **{'off': ['hls3_stall_probes'], **kw})
        hls.propose(eng, 16)
        hls.ledger['c0|experience:a']['mu'] = 0.02                  # z's prior is the larger: a is left out
        hls.ledger['c0|experience:z']['mu'] = 0.04
        observe(hls, eng, 'c1', {'a': 1, 'z': 1}, hypothesis='c0|az',
                reasons=['completes: killed at the timeout after 0/30 updates'])
        hls.idle = 3
        return hls, hls.propose(eng, 16)
    hls, props = setup()
    assert len(props) == 1 and props[0].origin == 'hls3-whole-experience'     # the composite, run at last
    whole = props[0]
    assert whole.change == {'z': 1, 'mb': 2} and whole.fidelity == ('probe', 'full') and whole.keep_probe
    assert hls.stalls[0]['whole'] == whole.mechanism
    hls, props = setup(off=['hls3_stall_probes', 'hls3_explore'])
    assert not [p for p in props if str(p.origin).startswith('hls3-whole')]


# ---------------------------------------------------------------- S-7: phase re-rank
def test_the_incumbents_own_phases_cap_its_moves():
    def setup(**kw):
        eng = engine(KNOBS, START)
        eng.s.hooks.knob_phases = {'z': ['comm']}
        eng.s.hooks.class_shares = lambda receipt: {'compute': 0.95, 'comm': 0.05}
        eng.s.candidates['c0'].verdict = {'valid': True, 'step_time': 1.0}
        hls = HLS3(FakeProposer([h('a', {'a': 1}, mu=0.1), h('z', {'z': 1}, mu=0.2)]), explore=False, **kw)
        hls.propose(eng, 16)
        observe(hls, eng, 'c1', {'a': 1}, hypothesis='c0|a', status='screened', gain=0.1, st=0.905, failed=())
        eng.s.candidates['c1'].receipt = {'evidence': {'phases': {'share': {'compute': 0.95, 'comm': 0.05}}}}
        eng._inc = 'c1'
        hls.propose(eng, 16)
        return hls
    hls = setup()
    carried = next(v for k, v in hls.ledger.items() if k.startswith('c1|') and v.get('change') == {'z': 1})
    assert carried['mu'] == pytest.approx(-math.log(0.95)) and carried['mu_raw'] == 0.2
    assert hls.describe()['hls3']['phase_ranked'][0]['capped'] == 1
    off = setup(off=['hls3_phases'])
    assert next(v for k, v in off.ledger.items() if k.startswith('c1|') and v.get('change') == {'z': 1})['mu'] == 0.2


# ---------------------------------------------------------------- the proposer: memory fields, S-8 sites, the context
class Recorder:
    """A ScriptedBackend that also keeps the schema each call was sent."""

    def __init__(self, script):
        from infrarsi.agent.backends.fake import ScriptedBackend
        self.inner, self.schemas = ScriptedBackend(script), []

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def complete(self, system, prompt, *, schema=None, role=None, **kw):
        if role == 'hypotheses':
            self.schemas.append(schema)
        return self.inner.complete(system, prompt, schema=schema, role=role, **kw)


def test_proposer_fields_rules_and_context():
    from infrarsi.search import proposers as P
    from test_search import TOGGLE, setup
    answer = {'hypotheses': [{'mechanism': 'x', 'change': {'fused_adam': True}, 'predicted_gain_pct': 3,
                              'memory': {'direction': 'spends', 'gib': -12}, 'variance': True, 'site': 'opt.step'}]}
    drv = Recorder(lambda role, system, prompt: {'verdict': 'no_patch', 'findings': []} if role == 'code_review' else answer)
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    used = dict(s.ledger.used)
    outs = []
    for flags in ({}, {'memory': True, 'memory_view': True, 'failures': True, 'variance': True, 'sites': True, 'rebase': True}):
        s.ledger.used.update(used)
        prop = P.LLMProposer(drv, s.hooks, search_calibrates=True, ledger_states=True, hls3=flags)
        if flags:
            prop.context = lambda engine, base: {'failures': [{'candidate': 'c1', 'class': 'oom'}]}
        outs.append(prop.hypotheses(eng, 'c0', ledger=[]))
    hyp = [c for c in drv.inner.calls if c['role'] == 'hypotheses']
    plain, rich = hyp
    assert drv.schemas[0] is P.HYPOTHESIS_SCHEMA and 'memory' not in json.dumps(drv.schemas[0])
    assert plain['system'].endswith(P.LEDGER_STATES_RULE) and 'failures' not in json.loads(plain['prompt'])
    for rule in (P.MEMORY_DECLARE_RULE, P.MEMORY_VIEW_RULE, P.FAILURES_RULE, P.VARIANCE_RULE, P.REBASE_RULE):
        assert rule in rich['system'] and rule not in plain['system']
    assert P.SITE_RULE not in rich['system']                        # no code in this prompt: no site rule
    assert json.loads(rich['prompt'])['failures'] == [{'candidate': 'c1', 'class': 'oom'}]
    items = drv.schemas[1]['properties']['hypotheses']['items']['properties']
    assert {'memory', 'variance', 'site'} <= set(items)
    assert 'mem_dir' not in outs[0][0] and outs[1][0]['mem_dir'] == 'spends' and outs[1][0]['mem_gib'] == 12.0
    assert outs[1][0]['variance'] is True and outs[1][0]['site'] == 'opt.step'


def test_sites_in_the_code_prompts():
    from infrarsi.search import proposers as P
    assert P.hls3_schema(P.REVIEW_SCHEMA, {'sites': True}, review=True)['properties']['findings']['items'][
        'properties']['site'] == {'type': ['string', 'null']}
    assert P.hls3_schema(P.REVIEW_SCHEMA, {}, review=True) is P.REVIEW_SCHEMA
    prop = P.LLMProposer(None, SimpleNamespace(), hls3={'sites': True})
    assert prop._hls3_rules(True) == P.SITE_RULE and prop._hls3_rules(False) == ''
    assert P.hls3_fields({'site': ' attention mask path '}, {'sites': True}) == {'site': 'attention mask path'}
    assert P.hls3_fields({'site': 'x'}, {}) == {}


# ---------------------------------------------------------------- S-9: the engine's robust mode
def receipt(times, window=(4, 12)):
    t, rows = 0.0, []
    for i, dt in enumerate(times, start=1):
        t += dt
        rows.append({'step': i, 't': t})
    return {'evidence': {'per_step': rows, 'window': list(window), 'rsd_window': 0.02, 'rsd_robust': 0.01}}


def test_robust_mode_reads_medians():
    stall = [1.0] * 3 + [1.0, 1.0, 3.5, 1.0, 1.0, 1.0, 1.0, 3.5, 1.0]    # two stalls in the window
    ev = receipt(stall)['evidence']
    assert median_update(ev) == pytest.approx(1.0) and median_update(ev, [4, 12]) == pytest.approx(1.0)
    s = SimpleNamespace(candidates={'c0': SimpleNamespace(verdict={'valid': True, 'step_time': 1.555}, receipt=receipt(stall))},
                        budget=SimpleNamespace(screens=10), ledger=SimpleNamespace(used={'screens': 0}))
    eng = Engine(s, full_steps=12, warmup=3)
    assert eng.robust is None and eng.t('c0') == 1.555 and eng.sigma_step() == 0.02
    eng.robust = True
    assert eng.t('c0') == pytest.approx(1.0) and eng.sigma_step() == 0.01 and eng.t_mean('c0') == 1.555
    rows = receipt([1.0] * 3 + [1.02, 1.02, 4.0, 1.02, 1.02, 1.02, 1.02])['evidence']['per_step']
    eng.robust = False
    assert eng.abort_rule(1.0)(rows) is True                        # the means: one stall kills a run as fast as 1.0
    eng.robust = True
    assert eng.abort_rule(1.0)(rows) is False                       # the medians: it does not
    slow = receipt([1.0] * 3 + [1.2] * 7)['evidence']['per_step']
    assert eng.abort_rule(1.0)(slow) is True                        # 20 % slower in every update does


def test_the_strategy_switches_the_engine_and_a_stall_remover_is_judged_by_the_mean():
    class Once:
        robust_stats = True
        name, uses_engine_stop = 'once', False

        def propose(self, engine, k):
            return []

        def observe(self, *a):
            pass

        def residual(self, engine):
            return {}

        def describe(self):
            return {}
    s = SimSession(START, knobs=KNOBS)
    eng = Engine(s, full_steps=30)
    eng.run(Once())
    assert eng.robust is True
    eng2 = Engine(SimSession(START, knobs=KNOBS), full_steps=30)
    eng2.run(HLS2(FakeProposer([]), explore=False))
    assert eng2.robust is False                                     # every other strategy: the means, as before
    eng3 = engine(KNOBS, START)
    hls = HLS3(FakeProposer([h('gc', {'a': 1}, mu=0.0, variance=True)]), explore=False)
    hls.variance['reference'] = {'outliers': True, 'prior': 0.03}
    props = hls.propose(eng3, 16)
    assert props[0].stat == 'mean' and hls.ledger['c0|gc']['mu'] == 0.03     # early: half the stalls' share
    hls.variance['reference'].update(mean_over_median_pct=6.0, stalls=3, updates=20, stall_factor=1.5)
    assert hls.context(eng3, 'c0')['variance']['stalls'] == 3
    off = HLS3(FakeProposer([h('gc', {'a': 1}, variance=True)]), explore=False, off=['hls3_variance'])
    assert off.robust_stats is False and off.propose(engine(KNOBS, START), 16)[0].stat is None


class Updates:
    """A session whose runs have per-update times (by the value of knob 'a'; `again` for re-runs): Engine.evaluate's
    full screen with the early-abort rule watching each update, and re-runs for confirm()."""

    def __init__(self, times: dict, again: dict | None = None, window=(4, 12)):
        self.times, self.again, self.window = times, again or times, list(window)
        self.budget = SimpleNamespace(screens=10, gpu_hours=10.0)
        self.ledger = SimpleNamespace(used={'screens': 0, 'gpu_hours': 0.0})
        self.screens, self.invalid = [], {}
        self.candidates = {'c0': SimpleNamespace(id='c0', parent=None, config={'a': 0}, overlay={}, results={},
                                                 status='proposed', verdict=None, receipt=None)}
        self.screen('c0')

    def _run(self, times, stop=None):
        t, rows = 0.0, []
        for i, dt in enumerate(times, start=1):
            t += dt
            rows.append({'step': i, 't': t})
            if stop is not None and stop(rows):
                return rows, True
        return rows, False

    def _result(self, rows):
        from infrarsi.project.signals import clock_time
        ev = {'per_step': rows, 'window': self.window, 'rsd_window': 0.05, 'rsd_robust': 0.01, 'steps_completed': len(rows)}
        return {'valid': True, 'step_time': clock_time(rows, self.window)[0], 'evidence': ev}

    def propose_config(self, changes, base, origin, rationale):
        cid = f'c{len(self.candidates)}'
        self.candidates[cid] = SimpleNamespace(id=cid, parent=base, config={**self.candidates[base].config, **changes},
                                               overlay={}, results={}, status='proposed', verdict=None, receipt=None)
        return {'candidate_id': cid}

    def screen(self, cid, stop=None):
        c = self.candidates[cid]
        rows, aborted = self._run(self.times[c.config['a']], stop)
        self.screens.append({'fidelity': 'full', 'gpu_hours': 0.1, 'valid': not aborted})
        self.ledger.used['screens'] += 1
        r = self._result(rows)
        c.receipt = {'evidence': r['evidence']}
        if aborted:
            c.status, c.verdict = 'aborted', {'valid': False}
            return {'valid': False, 'reasons': ['aborted']}
        c.status, c.verdict = 'screened', {'valid': True, 'step_time': r['step_time']}
        return {'valid': True, 'step_time': r['step_time']}

    def rescore(self, cid):
        self.screens.append({'fidelity': 'rescore', 'gpu_hours': 0.1, 'valid': True})
        return self._result(self._run(self.again[self.candidates[cid].config['a']])[0])


STALLS = [1.0] * 3 + [1.0, 1.0, 3.5, 1.0, 1.0, 1.0, 1.0, 3.5, 1.0]     # median 1.0, window mean 1.556
FEWER = [0.9] * 3 + [0.9, 0.9, 3.0, 0.9, 0.9, 0.9, 0.9, 3.0, 0.9]       # median 0.9, window mean 1.367


def test_robust_confirmation_keeps_the_means_pairs_for_the_promotion():
    """Codex r1 P1-1: robust mode decides a confirmation on the median updates, but the pairs of runs of one
    configuration it exports (noise['pairs'], noise['sigma_pair']: the promotion's margin, the project's noise ledger)
    stay the window means' offsets - the ones every other strategy exports; the median pairs stay in the search."""
    means = [math.log(1.0 / (14 / 9)), math.log(0.9 / (12.3 / 9))]   # the stall-free re-runs against the screens' means
    for robust in (True, False):
        s = Updates({0: STALLS, 1: FEWER}, again={0: [1.0] * 12, 1: [0.9] * 12})
        eng = Engine(s, full_steps=12, warmup=3, run_noise=None, robust=robust)
        s.propose_config({'a': 1}, 'c0', 'test', 'fewer stalls')
        s.screen('c1')
        eng.confirm('c1', 0.1)
        assert eng.pairs == pytest.approx(means, abs=1e-6)          # the same pairs, whatever the search compared
        n = eng.noise()
        assert n['pairs'] == pytest.approx(means, abs=1e-5) and n['sigma_pair'] >= 0.4
        if robust:
            assert eng.pairs_robust == pytest.approx([0.0, 0.0], abs=1e-9) and n['robust']['sigma_pair'] < 0.05
            assert eng.confirmations[-1]['statistic'] == 'median update'
            assert eng.confirmations[-1]['d2'] == pytest.approx(math.log(1.0 / 0.9), abs=1e-5)
        else:
            assert 'robust' not in n and not eng.pairs_robust and 'statistic' not in eng.confirmations[-1]
    for stat, d2 in (('mean', math.log((14 / 9) / 0.9)), (None, math.log(1.0 / 0.9))):   # the incumbent's re-run stalls
        s = Updates({0: STALLS, 1: FEWER}, again={0: STALLS, 1: [0.9] * 12})
        eng = Engine(s, full_steps=12, warmup=3, run_noise=None, robust=True)
        s.propose_config({'a': 1}, 'c0', 'test', 'fewer stalls')
        s.screen('c1')
        eng.confirm('c1', 0.1, stat=stat)                           # a stall remover (stat 'mean'): on the means
        assert eng.confirmations[-1]['statistic'] == ('mean' if stat else 'median update')
        assert eng.confirmations[-1]['d2'] == pytest.approx(d2, abs=1e-5)


def test_a_stall_remover_is_judged_by_the_mean_from_its_first_update():
    """Codex r1 P2-5: a move that removes the stalls but makes the typical update 4 % slower is not aborted by the
    medians' rule - its early abort compares means, as its gain does; any other move is aborted on the same updates."""
    remover = [1.04] * 12                                           # no stalls: mean 1.04 against the incumbent's 1.556
    for stat, aborted in (('mean', False), (None, True)):
        s = Updates({0: STALLS, 1: remover})
        eng = Engine(s, full_steps=12, warmup=3, run_noise=None, robust=True)
        out = eng.evaluate(Proposal(change={'a': 1}, mu=0.1, stat=stat, mechanism='gc stalls'))
        assert (out.status == 'aborted') is aborted, stat
        if not aborted:
            assert out.gain == pytest.approx(math.log((14 / 9) / 1.04), abs=1e-4) and eng.incumbent() == 'c1'


# ---------------------------------------------------------------- S-10: pruning and the sharded accounting
def test_the_card_size_arithmetic_prunes_only_from_a_configurations_own_state():
    """S-10 (Codex r1 P1-3): a configuration is pruned only when the shared accounting puts its training state alone
    above the card and every run that fitted agrees with that accounting; a declared spend over the sharded-state floor
    is probed first, never pruned; a run that fitted with the configuration overrides the estimate."""
    facts = {'params_b': 35.0, 'world': 16, 'cap_gib': 77.3}
    fl = sharded_floor(facts)
    assert fl['floor_gib'] == pytest.approx(35e9 * 16 / 16 / 2 ** 30, abs=0.01) and fl['shard'] == 16
    assert sharded_floor({**facts, 'trainable': 'lora-r16'}) is None and sharded_floor({}) is None

    def accounting(states):                                         # the shared accounting: a configuration's state
        return lambda cfg, t=None: {'memory_gib': states(cfg) + 30.0, 'capacity_gib': 77.3,
                                    'terms_gib': {'weights_grads_optimizer': states(cfg), 'activations': 30.0}}
    kept = accounting(lambda cfg: 32.6 + (65.2 if cfg.get('keep') else 0.0))   # parameters kept unsharded: +2 B each

    def setup(**kw):
        eng = engine({**KNOBS, 'keep': [False, True]}, {**START, 'keep': False})
        hyps = [h('keep params', {'keep': True}, mu=0.04, mem_dir='spends', mem_gib=65.2),
                h('mb2', {'mb': 2}, mu=0.05, mem_dir='spends', mem_gib=20.0)]
        hls = HLS3(FakeProposer(hyps), facts=facts, **kw)
        return eng, hls, {p.hypothesis: p for p in hls.propose(eng, 16)}
    eng, hls, props = setup()                                       # the declaration alone: probed first
    k = props['c0|keep params']
    assert k.fidelity == ('probe', 'full') and k.p_feasible == 0.4 and hls.ledger['c0|keep params']['status'] == 'open'
    assert 'keep=True: the sharded state alone is 33 GiB' in hls.ledger['c0|keep params']['memory']
    assert props['c0|mb2'].fidelity == ('full',) and not hls.describe()['hls3']['pruned']
    eng, hls, props = setup(accounting=kept)                        # the configuration's own state is above the card
    assert 'c0|keep params' not in props and hls.ledger['c0|keep params']['why'].startswith('unrunnable on this card')
    assert 'c0|mb2' in props and list(hls.describe()['hls3']['pruned']) == ['keep=True']
    assert 'c0|explore:keep=True' not in props                      # the neighbourhood leaves it out too
    eng.s.candidates['c9'] = SimpleNamespace(id='c9', parent='c0', config={**START, 'keep': True}, overlay={},
                                             verdict={'valid': True, 'step_time': 1.0}, status='screened', results={},
                                             receipt=None)          # ... until a run with it fits (at 70 GiB, say)
    assert hls._prune_why(eng, 'c0', {'keep': True}) is None and hls._accounting_contradicted(eng, 77.3) == 'c9'
    eng, hls, props = setup(accounting=accounting(lambda cfg: 402.3))   # the reference's state "above" the card it ran on:
    assert 'c0|keep params' in props                                # the accounting is wrong for this job, prunes nothing
    eng, hls, props = setup(off=['hls3_prune'], accounting=kept)
    assert 'c0|keep params' in props and props['c0|keep params'].fidelity == ('full',) and hls.mem.floor is None


def test_the_sharded_accounting_only_corrects_an_estimate_the_reference_contradicts():
    def view(config, step_time=None):
        return {'memory_gib': 510.26, 'capacity_gib': 77.3, 'margin_gib': -432.96,
                'terms_gib': {'weights_grads_optimizer': 402.33, 'activations': 95.75, 'logits': 5.68,
                              'cuda_nccl_outside_allocator': 1.5, 'allocator_slack': 5.0}, 'notes': []}
    wrapped = sharded_accounting(view, {'params_b': 27.0, 'world': 16, 'cap_gib': 77.3})
    a = wrapped({})
    assert a['terms_gib']['weights_grads_optimizer'] == pytest.approx(25.15, abs=0.01)
    assert a['memory_gib'] == pytest.approx(510.26 - 402.33 + 25.15, abs=0.02) and 'sharded' in a['notes'][-1]
    fits = lambda c, t=None: {**view(c), 'memory_gib': 60.0}                       # noqa: E731
    assert sharded_accounting(fits, {'params_b': 27.0, 'world': 16, 'cap_gib': 77.3})({})['memory_gib'] == 60.0
    assert sharded_accounting(None, {'params_b': 1}) is None and sharded_accounting(view, {}) is view


# ---------------------------------------------------------------- registration
def test_hls3_is_a_strategy_with_ablatable_components():
    assert 'hls3' in STRATEGIES and 'hls3' in LLM_STRATEGIES and resolve('hls3', has_llm=True) == 'hls3'
    assert set(HLS3_COMPONENTS) <= set(ABLATIONS) and len(HLS3_COMPONENTS) == 37 and len(HLS3_R1) == 9   # (R3: +13)
    assert {s for s, _ in HLS3_COMPONENTS.values()} == {f'S-{i}' for i in range(1, 11)} | {f'F{i}' for i in range(1, 9)} | {'F10', 'F11', 'F12'} | {'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F21', 'F22', 'F23', 'F9a', 'F9b', 'D7'}
    st = HLS3(FakeProposer([]))
    assert isinstance(st, HLS2) and st.name == 'hls3' and all(st.on.values()) and st.robust_stats
    assert not any(HLS3(FakeProposer([]), off=ALL).on.values())


def test_cli_help_names_hls2_and_hls3():
    from infrarsi.cli.main import build
    run_ = next(a for a in build()._subparsers._group_actions[0].choices['run']._actions if '--strategy' in a.option_strings)
    assert 'hls2' in run_.help and 'hls3' in run_.help


def test_manifest_accepts_the_component_names():
    from infrarsi.project.manifest import Project
    raw = {'name': 'p', 'launch': {'command': '{python} t.py {args}'}, 'knobs': {'k': {'type': 'bool', 'flag': '--k'}},
           'workload': {'steps': 4, 'window': [2, 4]}, 'search': {'strategy': 'hls3', 'ablate': ['hls3_bundles', 'hls3_prune']}}
    p = Project(raw)
    assert p.raw['search']['ablate'] == ['hls3_bundles', 'hls3_prune']


def test_engine_strategy_builds_hls3_from_the_ablate_list(monkeypatch):
    import infrarsi.search as search
    from infrarsi.search import engine as engine_mod
    from test_search import TOGGLE, setup
    built = {}

    class Stop(Exception):
        pass

    def fake_run(self, strategy, **kw):
        built['strategy'] = strategy
        raise Stop
    monkeypatch.setattr(engine_mod.Engine, 'run', fake_run)
    sim, s, _ = setup(TOGGLE)
    s.hooks.memory_facts = lambda: {'params_b': 7.0, 'world': 8, 'cap_gib': 80.0}
    for ablate in ([], ['hls3_bundles', 'hls3_memory_schema', 'hls3_sites']):
        ctx = search.Context(task=s.task, budget=s.budget, start_config=s.candidates['c0'].config, runner=s.runner,
                             judge=s.judge, hooks=s.hooks, run_id='t', out_dir=None, start_receipt={'valid': True},
                             options={'full_steps': 30, 'job_gpu_hours': None, 'capability': 0.9, 'ablate': ablate,
                                      'accounting': lambda c, t=None: None})
        with pytest.raises(Stop):
            search.run('hls3', ctx)
        st = built['strategy']
        assert type(st) is HLS3 and st.proposer.ledger_states and st.proposer.search_calibrates
        assert {k for k, v in st.on.items() if not v} == set(ablate) and st.facts['cap_gib'] == 80.0
        assert st.proposer.hls3.get('memory', False) == ('hls3_memory_schema' not in ablate)
        assert st.proposer.hls3.get('sites', False) == ('hls3_sites' not in ablate)
        assert st.proposer.context is not None


def test_project_hooks_give_the_memory_facts(tmp_path):
    from infrarsi.project.hooks import ProjectHooks
    from infrarsi.project.manifest import Project
    raw = {'name': 'p', 'launch': {'command': '{python} t.py {args}'},
           'knobs': {'mbs': {'type': 'int', 'flag': '--mbs', 'choices': [1, 2], 'default': 1},
                     'ga': {'type': 'int', 'flag': '--ga', 'choices': [2, 1], 'default': 2},
                     'k': {'type': 'bool', 'flag': '--k', 'default': False}},
           'workload': {'steps': 4, 'window': [2, 4], 'invariants': {'global_batch': 'mbs * ga * gpus'}},
           'model': {'params_b': 25.81}, 'hardware': {'gpus': 8, 'memory_gib': 141}}
    p = Project(raw, tmp_path / 'infrarsi.yaml')
    facts = ProjectHooks(p, p.task()).memory_facts()
    assert facts == {'cap_gib': 141.0, 'world': 8, 'params_b': 25.81, 'trainable': None, 'coupled': [['ga', 'mbs']]}


# ---------------------------------------------------------------- every component off is hls2
@pytest.mark.parametrize('name', sorted(SCENARIOS))
def test_every_component_off_is_hls2(name):
    class Off(HLS3):
        def __init__(self, prop, **kw):
            super().__init__(prop, off=ALL, **kw)
    a, b = run_scenario(HLS2, name), run_scenario(Off, name)
    assert sorted(a) == sorted(b)
    for key in a:
        if key == 'describe':
            assert {**a[key], 'name': None} == {**b[key], 'name': None}
        else:
            assert a[key] == b[key], key


def test_a_declared_spend_is_raised_to_what_the_run_showed():
    eng = engine(KNOBS, START)
    eng.s.candidates['c0'].verdict = {'valid': True, 'step_time': 1.0, 'peak_reserved_gib': 120.0}
    hls = HLS3(FakeProposer([h('n4', {'n': 4}, mem_dir='spends', mem_gib=5.0)]), explore=False)
    hls.propose(eng, 16)
    observe(hls, eng, 'c1', {'n': 4}, hypothesis='c0|n4', reasons=[G4U_OOM])
    assert hls.ledger['c0|n4']['mem_gib_bound'] == pytest.approx(139.81 - 120.0 + 4.53)    # cap - base peak + short
    assert hls._declared('n', 4) == ('spends', pytest.approx(24.34))


def test_a_patch_records_its_size_and_gain_per_line():
    s = SimSession(START, knobs=KNOBS, base_files={'f.py': 'a\nb\n'})
    eng = SimEngine(s, None, sigma_runs=[0.007])
    hls = HLS3(FakeProposer([h('p', {}, mu=0.04, patch={'f.py': 'a\nc\nd\n'})]), explore=False)
    hls.propose(eng, 16)
    e = hls.ledger['c0|p']
    assert e['patch_lines'] == 3 and e['gain_pct_per_line'] == pytest.approx(100 * 0.04 / 3, abs=1e-4)
    off = HLS3(FakeProposer([h('p', {}, mu=0.04, patch={'f.py': 'a\nc\nd\n'})]), explore=False, off=['hls3_sites'])
    off.propose(eng, 16)
    assert 'patch_lines' not in off.ledger['c0|p']
