"""FABLE_HLS3_FIXES_R2_Q9H_M35H2 (reports/infrarsi_newmodels_20261028/design/FABLE_HLS3_FIXES_R2_Q9H_M35H2.md): a unit
test per fix, each component switched off by its ablate name gives hls3 as it was (the approved round-1 tree). F10 a patch
proves at its probe that it took effect (its two lines, the job-log note it targets gone), else it is inert at start; F10c
the post-hoc inert floor; F11 the ranking prior of a declared memory spender no run bounds; F12 the speed gate at 1.6.
And the shared pieces, inert for every other strategy: the log scanner, the detailed trial, the gate's check name, the
proposer's rule and field, the toy's printing patches and inert mode."""
import dataclasses
from types import SimpleNamespace

import pytest

from infrarsi.project.signals import proof_evidence, scan_events
from infrarsi.search import HLS3_COMPONENTS, HLS3_R1, HLS3_R2
from infrarsi.search import proposers as P
from infrarsi.search.engine import Outcome, Proposal
from infrarsi.search.hls3 import (GATE, GATE_LOW, HLS3, PROBE_STEPS_MAX, WARMUP_MAX, claimed_notes, code_names,
                                    proof_verdict, proposer_flags, set_names, targeted_notes)
from test_composition import FakeProposer
from test_hls3 import G4U_OOM, KNOBS, START, engine, h, run
from test_hls3_r1 import stamped

R2 = list(HLS3_R2)
NOTE = '`fast_scan_fn` is falling back to its reference PyTorch implementation, which is much slower.'
OTHER = '`other_kernel` is not installed: falling back to the slow path.'
SITE = 'the `fast_scan_fn` call in every mixer layer'
PATCH = {'train_patches.py': 'import mod\nmod.fast_scan_fn = _kernel\n'}
GOOD = ['[infrarsi-patch] mixer.fast_scan_fn: found reference -> set kernel',
        '[infrarsi-patch] mixer.fast_scan_fn: active']


def with_events(eng, cid: str, **events) -> None:
    """The candidate's receipt events (its job-log notes, its patch lines)."""
    c = eng.s.candidates[cid]
    c.receipt = dict(c.receipt or {})
    c.receipt['evidence'] = {**(c.receipt.get('evidence') or {}), 'events': dict(events)}


def patch_q(hls, eng):
    return next(q for q in hls.propose(eng, 16) if q.patch)


def probe_row(lines=(), notes=(), *, note_after=True, steps=3, **kw) -> dict:
    """A probe's detail row as the session gives it: its notes, its proof lines and their evidence per site, each note's
    last place in the log (after the patch's installation, or before it)."""
    proof = proof_evidence(list(lines))
    at = [x[2] for r in proof.values() for x in r['installs']]
    place = (max(at) + 1.0) if note_after and at else -1.0
    return {'perf_notes': list(notes), 'patch_lines': list(lines), 'patch_proof': proof,
            'perf_note_last': {n: place for n in notes}, 'steps': steps, **kw}


def scan(**kw):
    return h('kernel scan', {}, patch=PATCH, mu=0.06, site=SITE, **kw)


# ---------------------------------------------------------------- registration, the proposer's rule and field
def test_r2_components_registered():
    assert len(R2) == 3 and {HLS3_COMPONENTS[c][0] for c in R2} == {'F10', 'F11', 'F12'} and not set(R2) & set(HLS3_R1)
    f = proposer_flags()
    assert f['self_proof'] and not proposer_flags(['hls3_self_proof'])['self_proof']
    assert f['inert'] and proposer_flags(['hls3_self_proof'])['inert'] and not proposer_flags(['hls3_dead_patch'])['inert']
    assert all(not v for k, v in HLS3(FakeProposer([]), off=R2).on.items() if k in R2)
    sch = P.hls3_schema(P.HYPOTHESIS_SCHEMA, {'self_proof': True})
    assert sch['properties']['hypotheses']['items']['properties']['evidence_note'] == {'type': ['string', 'null']}
    assert P.hls3_fields({'evidence_note': f' {NOTE} '}, {'self_proof': True}) == {'evidence_note': NOTE}
    assert P.hls3_fields({'evidence_note': NOTE}, {}) == {}
    prop = P.LLMProposer(None, SimpleNamespace(), hls3={'sites': True, 'self_proof': True})
    assert prop._hls3_rules(True) == P.SITE_RULE + P.PROOF_RULE and prop._hls3_rules(False) == ''
    assert '[infrarsi-patch] <site>: found' in P.PROOF_RULE and '`inert_patches`' in P.PROOF_RULE
    assert P.LLMProposer(None, SimpleNamespace(), hls3={'sites': True})._hls3_rules(True) == P.SITE_RULE


def test_r2_f10_the_code_review_asks_for_the_proof_too():
    from test_hls3 import Recorder
    from test_search import TOGGLE, setup
    for flags, has in (({'sites': True, 'self_proof': True}, True), ({'sites': True}, False)):
        drv = Recorder(lambda role, system, prompt: {'verdict': 'no_patch', 'findings': []} if role == 'code_review'
                       else {'hypotheses': []})
        _, s, eng = setup(TOGGLE, tokens=10 ** 6)
        P.LLMProposer(drv, s.hooks, hls3=flags).hypotheses(eng, 'c0', ledger=[])
        review = [c for c in drv.inner.calls if c['role'] == 'code_review']
        assert review and P.SITE_RULE in review[0]['system'] and (P.PROOF_RULE in review[0]['system']) is has


# ---------------------------------------------------------------- F10
def test_r2_f10_the_proof_lines_read_three_ways():
    assert proof_verdict(GOOD) is None
    assert 'already there' in proof_verdict(['[infrarsi-patch] experts: found grouped -> set grouped',
                                             '[infrarsi-patch] experts: active'])           # found == set
    assert 'never ran' in proof_verdict(GOOD[:1])                                              # no `active`
    assert 'printed no' in proof_verdict([]) and 'printed no' in proof_verdict(GOOD[1:])       # no `found -> set`
    assert 'never ran' in proof_verdict(GOOD + ['[infrarsi-patch] loader.fetch: found sync -> set async'])   # per site
    old = ['[infrarsi-patch] opt.step: found eager -> set fused', '[infrarsi-patch] opt.step: active']
    assert 'printed no' in proof_verdict(old, inherited=old)        # the base's patch proves nothing about this one
    assert proof_verdict(old + GOOD, inherited=old) is None
    assert proof_verdict(['[rank3]: [infrarsi-patch] x.y: found 1 -> set 2', 'log [infrarsi-patch] x.y: active']) is None


def test_r2_f10_the_notes_a_claim_targets():
    assert code_names(NOTE) == {'fast_scan_fn'} and 'fast_scan_fn' in code_names('mod.fast_scan_fn = _kernel')
    assert targeted_notes('mod.fast_scan_fn = _kernel', [NOTE, OTHER]) == [NOTE]
    assert targeted_notes('experts_implementation = "grouped"', [NOTE]) == []
    assert targeted_notes('', [NOTE]) == [] and targeted_notes('fast_scan_fn', []) == []
    assert set_names(['mod.fast_scan_fn = _kernel', 'if a == b: pass', 'm = load(p, attn_impl="sdpa")  # x = 1',
                      "setattr(mod, 'fast_scan_fn', k)", 'def fast_scan_fn(x):', 'y = mod.other_kernel(x)']) == \
        'mod.fast_scan_fn m attn_impl fast_scan_fn fast_scan_fn y'     # set, not called: other_kernel is no claim


def test_r2_f10_a_patch_that_only_calls_a_noted_function_does_not_claim_its_note():
    eng = engine(KNOBS, START)
    with_events(eng, 'c0', perf_notes=[NOTE])
    calls = {'train_patches.py': 'y = mod.fast_scan_fn(x) * gate\n'}
    hls = HLS3(FakeProposer([h('fuse mixer', {}, patch=calls, mu=0.06, site='the mixer output gate')]), explore=False)
    q = patch_q(hls, eng)
    run(eng, 'c1', {}, status='probed', fidelity='probe')
    assert q.gate(eng, 'c1', 'c0', probe_row(GOOD, [NOTE])) is None   # proven; the note is not its claim


def test_r2_f10_a_patch_is_probed_and_its_gate_reads_its_proof():
    """A patch (p 0.9: hls2 screens it in full) is probed - a kept probe, the screens being the scarce budget - and
    after its probe: the note it targets still reported, found == set, no `active`, no lines at all (D1) -> inert; its
    two lines and the note gone -> its full run goes on; a note it does not target changes nothing."""
    eng = engine(KNOBS, START)
    with_events(eng, 'c0', perf_notes=[NOTE, OTHER])
    hls = HLS3(FakeProposer([scan(evidence_note=NOTE)]), explore=False)
    q = patch_q(hls, eng)
    assert q.fidelity == ('probe', 'full') and q.keep_probe and q.gate is not None
    run(eng, 'c1', {}, status='probed', fidelity='probe')
    why, check = q.gate(eng, 'c1', 'c0', probe_row(GOOD, [NOTE]))
    assert check == 'patch_inert' and 'still reported' in why and 'fast_scan_fn' in why                # F10a
    same = ['[infrarsi-patch] mixer.fast_scan_fn: found kernel -> set kernel', GOOD[1]]
    assert 'already there' in q.gate(eng, 'c1', 'c0', probe_row(same))[0]                              # F10b
    assert 'never ran' in q.gate(eng, 'c1', 'c0', probe_row(GOOD[:1], [OTHER]))[0]
    assert 'printed no' in q.gate(eng, 'c1', 'c0', probe_row())[0]                                    # D1
    assert q.gate(eng, 'c1', 'c0', probe_row(GOOD, [OTHER])) is None                                   # proven
    assert hls.proof_fail['c1']['note'] == NOTE


def test_r2_f10_inert_at_start_goes_back_to_the_proposer():
    eng = engine(KNOBS, START)
    with_events(eng, 'c0', perf_notes=[NOTE])
    hls = HLS3(FakeProposer([scan(evidence_note=NOTE)]), explore=False)
    q = patch_q(hls, eng)
    run(eng, 'c1', {}, status='probed', fidelity='probe')
    why, check = q.gate(eng, 'c1', 'c0', probe_row(GOOD, [NOTE]))
    out = Outcome('c1', 'infeasible', False, None, None, 'probe', 0.1, [why], [check])
    assert hls.classify(eng, q, out) == {'kind': 'inert_at_start', 'line': why[:300]}
    hls.observe(eng, q, out)
    e = hls.ledger[q.hypothesis]
    assert e['status'] == 'invalid' and e['inert'] and e['why'].startswith('inert at its probe: the job-log note it names')
    row = hls.context(eng, 'c0')['inert_patches'][-1]
    assert row['at'] == 'probe' and row['note'] == NOTE and row['printed'] == GOOD
    assert row['patch'] == ['import mod', 'mod.fast_scan_fn = _kernel'] and 'still reported' in row['reason']
    assert 'c1' not in hls.mem.runs and not hls.together and not hls.suspects    # no memory record, no start failure
    assert hls.describe()['hls3']['r2']['inert_at_start'][0]['candidate'] == 'c1'
    assert all(x.hypothesis != q.hypothesis for x in hls.propose(eng, 16))       # its entry is done


def test_r2_f10_off_is_round_one():
    eng = engine(KNOBS, START)
    with_events(eng, 'c0', perf_notes=[NOTE])
    off = HLS3(FakeProposer([scan(evidence_note=NOTE)]), explore=False, off=['hls3_self_proof'])
    q = patch_q(off, eng)
    assert q.fidelity == ('full',) and not q.keep_probe and q.gate is None
    low = HLS3(FakeProposer([scan(p=0.5)]), explore=False, off=['hls3_self_proof'])
    q = patch_q(low, eng)
    run(eng, 'c1', {}, status='probed', fidelity='probe')
    assert 'probe' in q.fidelity and q.gate(eng, 'c1', 'c0', probe_row((), [NOTE])) is None
    assert 'r2' not in (low.describe()['hls3'] or {}) or 'inert_at_start' not in low.describe()['hls3']['r2']
    assert not getattr(low, 'proof_fail')


def test_r2_f10_a_patch_its_start_cap_sent_to_a_full_run_stays_full():
    """F1a: a patch whose probe its start cap stopped gets one full run - F10 does not probe it again."""
    eng = engine(KNOBS, START)
    hls = HLS3(FakeProposer([scan()]), explore=False, off=['hls3_coverage'])
    q = patch_q(hls, eng)
    assert 'probe' in q.fidelity
    out = run(eng, 'c1', {}, status='infeasible', fidelity='probe',
              reasons=['completes: killed at the timeout after 0/30 updates'])
    eng.s.candidates['c1'].results = {'probe': {'start_slow': True, 'wall_s': 400.0}}
    hls.observe(eng, q, out)
    r = next(x for x in hls.propose(eng, 16) if x.hypothesis == q.hypothesis)
    assert r.fidelity == ('full',) and r.retry and not r.keep_probe and r.gate is None


def test_r2_f10c_the_inert_floor():
    """F10c (M35H2-h3 c3): -0.15 % measured against a predicted +6 % on a 0.1 % sigma_pair - beyond sigma_pair, so F8b
    alone does not call it inert; within kappa x mu / 4 it is, and the reason its run's lines give goes along."""
    for off, inert in (((), True), (['hls3_self_proof'], False)):
        eng = engine(KNOBS, START, sp=0.001)
        hls = HLS3(FakeProposer([scan()]), explore=False, off=['hls3_coverage', *off])
        q = patch_q(hls, eng)
        hls.observe(eng, q, run(eng, 'c1', {}, status='screened', gain=-0.0015, st=1.0015, failed=()))
        assert bool(hls.inert) is inert, off
        if inert:
            assert 'printed no' in hls.inert[-1]['reason']          # (its full run printed no line: said along)


# ---------------------------------------------------------------- F11
def test_r2_f11_pi_counts_the_runs_of_feared_spenders():
    """pi = (fitted + 1) / (tried + 2) over the runs of feared moves - declared to spend memory and declared below the
    probe rule (the moves F11 lifts): once per candidate; a confident spender or a move without a declared spend is not
    counted."""
    eng = engine(KNOBS, START)
    hyps = [h('n4', {'n': 4}, mu=0.1, p=0.15, mem_dir='spends', mem_gib=40.0), h('mb2', {'mb': 2}, mu=0.05, p=0.3,
                                                                                  mem_dir='spends'),
            h('a', {'a': 1}, mu=0.02), h('z', {'z': 1}, mu=0.02, p=0.9, mem_dir='spends')]
    hls = HLS3(FakeProposer(hyps), explore=False, off=['hls3_coverage'])
    props = {q.hypothesis: q for q in hls.propose(eng, 16)}
    assert hls._pi() == 0.5                                             # Beta(1, 1): nothing tried
    hls.observe(eng, props['c0|mb2'], run(eng, 'c1', {'mb': 2}, status='probed', fidelity='probe', failed=()))
    assert (hls.fear['tried'], hls.fear['fitted']) == (1, 1) and hls._pi() == pytest.approx(2 / 3)
    hls.observe(eng, props['c0|mb2'], run(eng, 'c1', {'mb': 2}, status='screened', gain=0.01, st=0.99, failed=()))
    assert hls.fear['tried'] == 1                                       # its full run: the same try
    hls.observe(eng, props['c0|a'], run(eng, 'c2', {'a': 1}, status='screened', gain=0.01, st=0.99, failed=()))
    hls.observe(eng, props['c0|z'], run(eng, 'c4', {'z': 1}, status='screened', gain=0.01, st=0.99, failed=()))
    assert hls.fear['tried'] == 1                                       # no declared spend / a confident one: not counted
    hls.observe(eng, props['c0|n4'], run(eng, 'c3', {'n': 4}, status='infeasible', fidelity='probe',
                                         reasons=[G4U_OOM]))
    assert (hls.fear['tried'], hls.fear['fitted']) == (2, 1) and hls._pi() == pytest.approx(0.5)


def test_r2_f11_a_feared_spender_is_ranked_by_this_runs_evidence():
    """Q9H's micro-batch 2: declared at p 0.15 (40 GiB), the transfer judgment's 0.55, no run bounds it -> ranked at
    max(0.15, 0.55, pi), probed first, the entry keeping 0.15 (D2); two feared spenders that fitted lift pi to 3/4;
    an out-of-memory run that bounds it (the memory model unsure) overrides; switched off: 0.15."""
    exp = {'judgment': {'assessment': [{'change': {'mb': 2}, 'sign': '+', 'p_feasible': 0.55}]}}
    hyps = [h('mb2', {'mb': 2}, mu=0.03, p=0.15, mem_dir='spends', mem_gib=40.0),
            h('n2', {'n': 2}, mu=0.002, p=0.5, mem_dir='spends'), h('n3', {'n': 3}, mu=0.002, p=0.5, mem_dir='spends')]
    eng = engine(KNOBS, START)
    hls = HLS3(FakeProposer(hyps), explore=False, experience=exp, off=['hls3_coverage'])
    props = {q.hypothesis: q for q in hls.propose(eng, 16)}
    q = props['c0|mb2']
    assert q.p_feasible == pytest.approx(0.55) and q.fidelity == ('probe', 'full') and q.keep_probe
    assert hls.ledger['c0|mb2']['p_feasible'] == 0.15 and hls.ranked['c0|mb2'] == 0.55
    for cid, key, ch in (('c1', 'c0|n2', {'n': 2}), ('c2', 'c0|n3', {'n': 3})):
        hls.observe(eng, props[key], run(eng, cid, ch, status='screened', gain=0.0, st=1.0, failed=()))
    assert hls._pi() == pytest.approx(0.75)
    q = next(x for x in hls.propose(eng, 16) if x.hypothesis == 'c0|mb2')
    assert q.p_feasible == pytest.approx(0.75) and hls.ledger['c0|mb2']['p_feasible'] == 0.15
    assert hls.describe()['hls3']['r2']['fear_prior']['ranked']['c0|mb2'] == 0.75
    hls.ranked.clear()
    real = hls._predict
    hls._predict = lambda e, cfg: ('unsure', 'an out-of-memory run bounds it', 'c3')   # (a measured bound, unsure)
    q = next(x for x in hls.propose(eng, 16) if x.hypothesis == 'c0|mb2')
    assert q.p_feasible <= 0.4 and not hls.ranked                       # no prior lifts it
    hls._predict = real
    hls.observe(eng, Proposal(change={'mb': 2}, base='c0'),             # its own configuration ran out of memory
                run(eng, 'c3', {'mb': 2}, status='infeasible', fidelity='probe', reasons=[G4U_OOM]))
    assert hls._predict(eng, hls._mcfg(eng, 'c0', {'mb': 2}))[0] == 'infeasible'
    assert all(x.hypothesis != 'c0|mb2' for x in hls.propose(eng, 16)) and hls.ledger['c0|mb2'].get('deferred')
    eng = engine(KNOBS, START)
    off = HLS3(FakeProposer(hyps), explore=False, experience=exp, off=['hls3_coverage', 'hls3_fear_prior'])
    assert next(x for x in off.propose(eng, 16) if x.hypothesis == 'c0|mb2').p_feasible == 0.15 and not off.ranked


# ---------------------------------------------------------------- F12
def test_r2_f12_the_speed_gate_at_1_6():
    eng = engine(KNOBS, START)
    stamped(eng, 'c0', wall=200.0, first=100.0)
    hyps = [h('n4', {'n': 4}, mu=0.1, mem_dir='spends', p=0.5),
            h('no recompute', {}, patch={'train_patches.py': 'RECOMPUTE = False\n'}, mu=0.1, p=0.5, mem_dir='spends',
              mem_gib=20.0)]
    hls = HLS3(FakeProposer(hyps), explore=False, off=['hls3_self_proof', 'hls3_coverage'])
    props = {q.hypothesis: q for q in hls.propose(eng, 16)}
    run(eng, 'c1', {'n': 4}, status='probed', fidelity='probe')
    run(eng, 'c2', {}, status='probed', fidelity='probe')
    assert (GATE, GATE_LOW, hls.gate_ratio) == (2.0, 1.6, 1.6)
    assert 'slow start' in props['c0|n4'].gate(eng, 'c1', 'c0', {'intervals': [1.0, 1.9]})       # x1.9 >= 1.6
    assert props['c0|n4'].gate(eng, 'c1', 'c0', {'intervals': [1.0, 1.5]}) is None
    assert 'slow start' in props['c0|no recompute'].gate(eng, 'c2', 'c0', {'intervals': [1.0, 1.9]})   # its patch's spend
    off = HLS3(FakeProposer(hyps), explore=False, off=['hls3_self_proof', 'hls3_coverage', 'hls3_gate_low'])
    props = {q.hypothesis: q for q in off.propose(eng, 16)}
    assert off.gate_ratio == GATE and props['c0|n4'].gate(eng, 'c1', 'c0', {'intervals': [1.9]}) is None
    assert props['c0|no recompute'].gate(eng, 'c2', 'c0', {'intervals': [2.5]}) is None   # round 1: its patch not seen


# ---------------------------------------------------------------- the coordinator's round-2 decisions
def test_r2_decision_f11_pi_waits_for_a_feared_spenders_run():
    """Decision 1 (variant 1 as measured): before a feared move of this run - a declared spender below the probe rule -
    has run, a feared spender is ranked at the most this run declared for it (here 0.15: no lift to the prior's 0.5;
    with the judgment's exact match, 0.55); a confident spender's run opens nothing; once a feared one has run (fitted or
    failed) pi joins (one fit: 2/3)."""
    hyps = [h('mb2', {'mb': 2}, mu=0.03, p=0.15, mem_dir='spends', mem_gib=40.0),
            h('n2', {'n': 2}, mu=0.002, p=0.5, mem_dir='spends'), h('z', {'z': 1}, mu=0.002, p=0.9, mem_dir='spends')]
    eng = engine(KNOBS, START)
    hls = HLS3(FakeProposer(hyps), explore=False, off=['hls3_coverage'])
    props = {q.hypothesis: q for q in hls.propose(eng, 16)}
    assert props['c0|mb2'].p_feasible == 0.15 and 'c0|mb2' not in hls.ranked and hls._pi() == 0.5
    hls.observe(eng, props['c0|z'], run(eng, 'c2', {'z': 1}, status='screened', gain=0.0, st=1.0, failed=()))
    assert next(x for x in hls.propose(eng, 16) if x.hypothesis == 'c0|mb2').p_feasible == 0.15
    hls.observe(eng, props['c0|n2'], run(eng, 'c1', {'n': 2}, status='screened', gain=0.0, st=1.0, failed=()))
    q = next(x for x in hls.propose(eng, 16) if x.hypothesis == 'c0|mb2')
    assert q.p_feasible == pytest.approx(2 / 3) and hls.ledger['c0|mb2']['p_feasible'] == 0.15
    exp = {'judgment': {'assessment': [{'change': {'mb': 2}, 'sign': '+', 'p_feasible': 0.55}]}}
    eng = engine(KNOBS, START)
    judged = HLS3(FakeProposer(hyps), explore=False, experience=exp, off=['hls3_coverage'])
    assert next(x for x in judged.propose(eng, 16) if x.hypothesis == 'c0|mb2').p_feasible == pytest.approx(0.55)


def test_r2_decision_f12_an_unknown_memory_effect_counts_as_a_possible_spend():
    """Decision 2: a slow probe (x1.8) whose change has an unknown memory direction (a patch that declared none, as Q9H
    c5's 'unknown') is stopped - a possible spend; a declared freeing or neutral change is only noted; F12 off: round 1
    (a declared or measured spend only, at 2.0)."""
    eng = engine(KNOBS, START)
    stamped(eng, 'c0', wall=200.0, first=100.0)
    hyps = [h('no recompute', {}, patch={'train_patches.py': 'RECOMPUTE = False\n'}, mu=0.1, p=0.7),
            h('a frees', {'a': 1}, mu=0.05, p=0.5, mem_dir='frees', mem_gib=2.0),
            h('z neutral', {'z': 1}, mu=0.05, p=0.5, mem_dir='neutral')]
    hls = HLS3(FakeProposer(hyps), explore=False, off=['hls3_self_proof', 'hls3_coverage'])
    props = {q.hypothesis: q for q in hls.propose(eng, 16)}
    run(eng, 'c1', {}, status='probed', fidelity='probe')
    run(eng, 'c2', {'a': 1}, status='probed', fidelity='probe')
    run(eng, 'c3', {'z': 1}, status='probed', fidelity='probe')
    why = props['c0|no recompute'].gate(eng, 'c1', 'c0', {'intervals': [1.0, 1.8]})
    assert 'may spend memory' in why and hls.gated[-1]['unknown'] and not hls.gated[-1]['full_run']
    out = Outcome('c1', 'infeasible', False, None, None, 'probe', 0.1, [why], ['probe_speed'])
    assert hls.classify(eng, props['c0|no recompute'], out)['kind'] == 'slow_probe'   # no memory evidence: no thrash
    assert props['c0|a frees'].gate(eng, 'c2', 'c0', {'intervals': [1.0, 1.8]}) is None and hls.gated[-1]['full_run']
    assert props['c0|z neutral'].gate(eng, 'c3', 'c0', {'intervals': [1.0, 1.8]}) is None
    off = HLS3(FakeProposer(hyps), explore=False, off=['hls3_self_proof', 'hls3_coverage', 'hls3_gate_low'])
    props = {q.hypothesis: q for q in off.propose(eng, 16)}
    assert props['c0|no recompute'].gate(eng, 'c1', 'c0', {'intervals': [2.5]}) is None   # round 1: noted only


# ---------------------------------------------------------------- Codex HLS4R2 r1
def test_codex_r2_1_the_gate_reads_the_updates_after_the_warm_up():
    """P1: a recompilation in the second update made a legal compile probe look slow. A gated probe runs three updates;
    only the intervals after the second update are compared with the base's same ones; none -> no veto; an observed
    recompilation -> no veto; a declared longer warm-up is skipped (beyond WARMUP_MAX: not supported, the default
    warm-up - Codex HLS4R2 r2); the probe's cap and price count its updates."""
    eng = engine(KNOBS, START)
    stamped(eng, 'c0', wall=200.0, first=100.0)
    hyps = [h('n4', {'n': 4}, mu=0.1, mem_dir='spends', p=0.5),
            h('compiled', {'z': 1}, mu=0.1, p=0.5, mem_dir='spends', warmup_updates=4),
            h('long', {'a': 1}, mu=0.1, p=0.5, mem_dir='spends', warmup_updates=9)]
    hls = HLS3(FakeProposer(hyps), explore=False, off=['hls3_self_proof', 'hls3_coverage'])
    props = {q.hypothesis: q for q in hls.propose(eng, 16)}
    for c, ch in (('c1', {'n': 4}), ('c2', {'z': 1}), ('c3', {'a': 1})):
        run(eng, c, ch, status='probed', fidelity='probe')
    n4 = props['c0|n4']
    assert n4.gate(eng, 'c1', 'c0', {'intervals': [2.5, 1.0]}) is None            # the second update recompiled: legal
    assert n4.probe_steps == 3 and props['c0|compiled'].probe_steps == 5 and props['c0|long'].probe_steps == 3
    assert n4.gate(eng, 'c1', 'c0', {'intervals': [2.5]}) is None                 # (two updates: nothing to read)
    assert 'slow start' in n4.gate(eng, 'c1', 'c0', {'intervals': [1.0, 2.0]})
    assert n4.gate(eng, 'c1', 'c0', {'intervals': [1.0, 2.0], 'recompiles': 1}) is None
    assert props['c0|compiled'].gate(eng, 'c2', 'c0', {'intervals': [2.0, 2.0, 2.0, 1.0]}) is None
    assert 'slow start' in props['c0|compiled'].gate(eng, 'c2', 'c0', {'intervals': [1.0, 1.0, 1.0, 2.0]})
    assert 'slow start' in props['c0|long'].gate(eng, 'c3', 'c0', {'intervals': [1.0, 2.0]})   # (9: no exemption)
    plain = dataclasses.replace(n4, probe_steps=None)
    assert eng.expected_cost(n4) > eng.expected_cost(plain)
    assert hls._probe_cap(eng, 'c0', 3) >= hls._probe_cap(eng, 'c0') and n4.probe_cap == hls._probe_cap(eng, 'c0', 3)
    off = HLS3(FakeProposer(hyps), explore=False, off=['hls3_self_proof', 'hls3_coverage', 'hls3_gate_low'])
    assert next(q for q in off.propose(eng, 16) if q.hypothesis == 'c0|n4').probe_steps is None   # round 1: two updates


def test_codex_r2_2_a_new_installation_at_an_inherited_site_is_proven_by_its_own_active_line():
    """P1: the base printed `found eager -> set v1` and `active` at a site; a new patch there prints `found v1 -> set v2`
    and the same `active` text - proven (its installation is new; its site ran in this run). An installation only the
    inherited patch printed proves nothing."""
    base = ['[infrarsi-patch] mixer.fast_scan_fn: found eager -> set v1', '[infrarsi-patch] mixer.fast_scan_fn: active']
    new = ['[infrarsi-patch] mixer.fast_scan_fn: found v1 -> set v2', '[infrarsi-patch] mixer.fast_scan_fn: active']
    assert proof_verdict(new, inherited=base) is None
    assert 'of its own' in proof_verdict(base, inherited=base)
    eng = engine(KNOBS, START)
    with_events(eng, 'c0', patch_lines=base)
    hls = HLS3(FakeProposer([scan()]), explore=False)
    q = patch_q(hls, eng)
    run(eng, 'c1', {}, status='probed', fidelity='probe')
    assert q.gate(eng, 'c1', 'c0', probe_row(new)) is None


def test_codex_r2_3_only_a_named_note_reported_after_installation_vetoes():
    """P2: a fallback warning logged at import, before the patch installed, is no veto; the note the patch names
    (`evidence_note`) reported after it installed is; a note the claim only shares a code name with, or one without the
    order of the lines, is feedback for the proposer, no veto."""
    eng = engine(KNOBS, START)
    with_events(eng, 'c0', perf_notes=[NOTE])
    named = HLS3(FakeProposer([scan(evidence_note=NOTE)]), explore=False)
    q = patch_q(named, eng)
    run(eng, 'c1', {}, status='probed', fidelity='probe')
    assert q.gate(eng, 'c1', 'c0', probe_row(GOOD, [NOTE], note_after=False)) is None    # logged before it installed
    assert q.gate(eng, 'c1', 'c0', probe_row(GOOD, [NOTE]))[1] == 'patch_inert'          # after it installed
    unordered = {k: v for k, v in probe_row(GOOD, [NOTE]).items() if k != 'perf_note_last'}
    assert q.gate(eng, 'c1', 'c0', unordered) is None and named.note_feedback[-1]['after_install'] is None
    eng = engine(KNOBS, START)
    with_events(eng, 'c0', perf_notes=[NOTE])
    incidental = HLS3(FakeProposer([scan()]), explore=False)                    # site and code name it, no evidence_note
    q = patch_q(incidental, eng)
    run(eng, 'c1', {}, status='probed', fidelity='probe')
    assert q.gate(eng, 'c1', 'c0', probe_row(GOOD, [NOTE])) is None
    fb = incidental.context(eng, 'c0')['patch_notes'][-1]
    assert fb['candidate'] == 'c1' and fb['note'] == NOTE and not fb['named'] and fb['after_install']


def test_codex_r2_4_a_feared_spenders_stall_probe_counts_in_pi():
    """P2: a stall probe (`probe|bundle|...`) holding a feared spender is a run of it: counted (once per candidate),
    passed or out of memory."""
    eng = engine(KNOBS, START)
    hls = HLS3(FakeProposer([h('mb2', {'mb': 2}, mu=0.03, p=0.15, mem_dir='spends', mem_gib=40.0)]), explore=False,
               off=['hls3_coverage'])
    hls.propose(eng, 16)
    hls.members['bundle|x'] = ['c0|mb2']
    hls.members['bundle|y'] = ['c0|mb2']
    stall = Proposal(change={'mb': 2, 'a': 1}, base='c0', hypothesis='probe|bundle|x', fidelity=('probe',))
    hls.observe(eng, stall, run(eng, 'c1', {'mb': 2, 'a': 1}, status='probed', fidelity='probe', failed=()))
    assert (hls.fear['tried'], hls.fear['fitted']) == (1, 1)
    hls.observe(eng, stall, run(eng, 'c1', {'mb': 2, 'a': 1}, status='probed', fidelity='probe', failed=()))
    assert hls.fear['tried'] == 1                                       # the same candidate: once
    oom = Proposal(change={'mb': 2, 'z': 1}, base='c0', hypothesis='probe|bundle|y', fidelity=('probe',))
    hls.observe(eng, oom, run(eng, 'c2', {'mb': 2, 'z': 1}, status='infeasible', fidelity='probe', reasons=[G4U_OOM]))
    assert (hls.fear['tried'], hls.fear['fitted']) == (2, 1)


def test_codex_r2_5_the_proof_evidence_covers_the_whole_log():
    """P2: eight ranks install three sites, each line with its own object address (24 distinct lines), then each site
    runs: the evidence per site holds every site's `active` - the 20 display lines do not."""
    sites = ('mixer.scan', 'mlp.act', 'opt.step')
    lines = [(0.0, f'[rank{r}]: [infrarsi-patch] {s}: found <fn at 0x{r}{i}a> -> set <fn at 0x{r}{i}b>')
             for r in range(8) for i, s in enumerate(sites)]
    lines += [(0.0, f'[rank0]: [infrarsi-patch] {s}: active') for s in sites]
    ev = scan_events(lines)
    assert len(ev['patch_lines']) == 20 and all(ev['patch_proof'][s]['active_at'] is not None for s in sites)
    assert proof_verdict(ev['patch_proof']) is None
    assert 'never ran' in proof_verdict(ev['patch_lines'])            # (the display lines alone lose the `active` lines)


def test_codex_r2_note_the_proof_horizon_is_the_probe():
    """A patch whose code first runs at a declared later update: its probe runs that many updates (at most
    PROBE_STEPS_MAX) and its `active` line is expected there; declared beyond the bound, the declaration is not
    supported: the default probe, and a missing `active` is inert (Codex HLS4R2 r2)."""
    eng = engine(KNOBS, START)
    hyps = [scan(evidence_note=NOTE, active_from_update=5),
            h('late hook', {}, patch={'train_patches.py': 'mod.late_fn = _late\n'}, mu=0.06, active_from_update=50)]
    hls = HLS3(FakeProposer(hyps), explore=False)
    props = {q.hypothesis: q for q in hls.propose(eng, 16) if q.patch}
    q5, q50 = props['c0|kernel scan'], props['c0|late hook']
    run(eng, 'c1', {}, status='probed', fidelity='probe')
    assert 'never ran' in q50.gate(eng, 'c1', 'c0', probe_row(['[infrarsi-patch] late: found off -> set on']))[0]
    assert 'never ran' in q5.gate(eng, 'c1', 'c0', probe_row(GOOD[:1], steps=5))[0]
    assert q5.probe_steps == 5 and q50.probe_steps == 3


def test_codex_r2r2_1_only_the_quoted_note_can_veto():
    """Codex HLS4R2 r2 P2: two warnings name the same function; the proposer's `evidence_note` paraphrases the first
    (no quote); the first disappears, the second stays after the patch installed - no veto (the second is feedback).
    Quoting the first: a veto only while the first stays. A passage both notes hold names neither."""
    nc = '`fast_scan_fn` falls back to the reference path for noncontiguous input; this is much slower.'
    dt = '`fast_scan_fn` falls back to the reference path: unsupported dtype bfloat16; this is much slower.'
    para = 'Remove the noncontiguous-input fast_scan_fn fallback warning.'
    for note, ok_rows, bad_rows in ((para, ([dt],), ()), (nc, ([dt],), ([nc],)),
                                    ('falls back to the reference path', ([dt], [nc]), ())):
        eng = engine(KNOBS, START)
        with_events(eng, 'c0', perf_notes=[nc, dt])
        hls = HLS3(FakeProposer([scan(evidence_note=note)]), explore=False)
        q = patch_q(hls, eng)
        run(eng, 'c1', {}, status='probed', fidelity='probe')
        for notes in ok_rows:
            assert q.gate(eng, 'c1', 'c0', probe_row(GOOD, notes)) is None, (note, notes)
        for notes in bad_rows:
            assert q.gate(eng, 'c1', 'c0', probe_row(GOOD, notes))[1] == 'patch_inert', (note, notes)
        if note == para:
            fb = hls.context(eng, 'c0')['patch_notes'][-1]
            assert fb['note'] == dt and not fb['named'] and fb['after_install']
    assert claimed_notes(para, [nc, dt]) == [] and claimed_notes(nc, [nc, dt]) == [nc]
    assert claimed_notes('falls back to the reference path', [nc, dt]) == []     # (both hold it: no one note)
    assert claimed_notes('noncontiguous input', [nc, dt]) == []                  # (shorter than QUOTE_MIN)


def test_codex_r2r2_2_an_out_of_range_declaration_exempts_from_nothing():
    """Codex HLS4R2 r2 P2: `warmup_updates` is supported 1..WARMUP_MAX and `active_from_update` 1..PROBE_STEPS_MAX - in
    the schemas, in the parsing (an unsupported value ignored, with feedback) and in hls3 itself (out of range is absent).
    A spender declaring a warm-up of 7 gets the gated three-update probe and its 20x interval stops it; a patch declaring
    its first run at update 1000 gets no exemption from its `active` line."""
    flags = {'self_proof': True, 'warmup': True}
    eng = engine(KNOBS, START)
    stamped(eng, 'c0', wall=200.0, first=100.0)
    hyps = [h('n4', {'n': 4}, mu=0.1, mem_dir='spends', p=0.5, **P.hls3_fields({'warmup_updates': 7}, flags)),
            h('z4', {'z': 1}, mu=0.1, mem_dir='spends', p=0.5, warmup_updates=7),      # (not through the parsing)
            h('late hook', {}, patch={'train_patches.py': 'mod.late_fn = _late\n'}, mu=0.06,
              **P.hls3_fields({'active_from_update': 1000}, flags)),
            h('late too', {}, patch={'train_patches.py': 'mod.other_fn = _other\n'}, mu=0.06, active_from_update=1000)]
    hls = HLS3(FakeProposer(hyps), explore=False, off=['hls3_coverage'])
    props = {q.hypothesis: q for q in hls.propose(eng, 16)}
    run(eng, 'c1', {'n': 4}, status='probed', fidelity='probe')
    run(eng, 'c2', {'z': 1}, status='probed', fidelity='probe')
    for key, cid in (('c0|n4', 'c1'), ('c0|z4', 'c2')):
        assert props[key].probe_steps == 3 and 'slow start' in props[key].gate(eng, cid, 'c0', {'intervals': [1.0, 20.0]})
    run(eng, 'c3', {}, status='probed', fidelity='probe')
    for key in ('c0|late hook', 'c0|late too'):
        q = props[key]
        assert q.probe_steps == 3
        assert 'never ran' in q.gate(eng, 'c3', 'c0', probe_row(['[infrarsi-patch] late: found off -> set on']))[0]
    ignored = hls.context(eng, 'c0')['declarations_ignored']
    assert {x['mechanism'] for x in ignored} == {'n4', 'late hook'}
    assert any('warmup_updates=7' in s for x in ignored for s in x['ignored'])
    sch = P.hls3_schema(P.HYPOTHESIS_SCHEMA, flags)['properties']['hypotheses']['items']['properties']
    assert (sch['warmup_updates']['minimum'], sch['warmup_updates']['maximum']) == (1, WARMUP_MAX)
    assert (sch['active_from_update']['minimum'], sch['active_from_update']['maximum']) == (1, PROBE_STEPS_MAX)
    assert P.hls3_fields({'warmup_updates': 4, 'active_from_update': 5}, flags) == {'warmup_updates': 4,
                                                                                    'active_from_update': 5}
    bad = P.hls3_fields({'warmup_updates': 7, 'active_from_update': 1000}, flags)
    assert 'warmup_updates' not in bad and 'active_from_update' not in bad and len(bad['rejected']) == 2
    assert P.hls3_fields({'warmup_updates': True, 'active_from_update': 2.5}, flags)['rejected']


# ---------------------------------------------------------------- shared code
def test_r2_shared_the_log_scanner_keeps_a_patchs_lines():
    lines = [(0, '[rank0]: [infrarsi-patch] mixer.scan: found reference -> set kernel'),
             (1, '[rank1]: [infrarsi-patch] mixer.scan: found reference -> set kernel'), (2, 'step 1 loss 2.31'),
             (3, '[infrarsi-patch] mixer.scan: active')]
    ev = scan_events(lines)
    assert ev['patch_lines'] == ['[infrarsi-patch] mixer.scan: found reference -> set kernel',
                                 '[infrarsi-patch] mixer.scan: active']
    assert ev['patch_proof']['mixer.scan']['installs'][0][:2] == ['reference', 'kernel']
    assert ev['patch_proof']['mixer.scan']['active_at'] is not None
    plain = scan_events([(0, 'step 1 loss 2.31'), (1, 'torch._dynamo: Recompiling function forward')])
    assert 'patch_lines' not in plain and 'patch_proof' not in plain and plain['recompiles'] == 1
    assert 'recompiles' not in scan_events([(0, 'step 1 loss 2.31')])


def test_r2_shared_a_detailed_trial_hands_over_the_notes_and_lines():
    from test_search import TOGGLE, setup
    _, s, _ = setup(TOGGLE)
    toy = s.runner.run

    def noted(task, cfg, ov, purpose, rid, **kw):
        r = toy(task, cfg, ov, purpose, rid, **kw)
        r['evidence']['events'].update(perf_notes=[NOTE], patch_lines=GOOD)
        return r
    s.runner.run = noted
    ref = s.candidates['c0'].config
    k1, k2 = [k for k, v in ref.items() if isinstance(v, bool)][:2]
    a = s.propose_config(changes={k1: not ref[k1]}, base='c0')['candidate_id']
    row = s.trial(a, steps=2, fidelity='probe', detail=True)
    assert row['perf_notes'] == [NOTE] and row['patch_lines'] == GOOD
    b = s.propose_config(changes={k2: not ref[k2]}, base='c0')['candidate_id']
    assert 'patch_lines' not in s.trial(b, steps=2, fidelity='probe')    # without detail: as in round 1


def test_r2_shared_the_engine_names_the_gates_check():
    """The toy job: a patch that prints its lines hands them to the gate; a gate's (why, check) is the outcome's failed
    check, a plain why the speed gate's (probe_speed), as in round 1."""
    from test_search import TOGGLE, setup
    sim, s, eng = setup(TOGGLE)
    good = next(p for p, ok in sorted(sim.patch_ok.items()) if ok)
    seen = []

    def gate(engine, cid, base, row):
        seen.append(row)
        return 'it did not prove it took effect', 'patch_inert'
    p = Proposal(change={}, base='c0', patch={'patches.txt': f'{good}\nproof:{good}\n'}, fidelity=('probe', 'full'),
                 keep_probe=True, gate=gate, mechanism='a patch')
    out = eng.evaluate(p)
    assert out.status == 'infeasible' and out.fidelity == 'probe' and out.failed_checks == ['patch_inert']
    assert seen[-1]['patch_lines'] == [f'[infrarsi-patch] {good}: found off -> set on', f'[infrarsi-patch] {good}: active']
    q = dataclasses.replace(p, patch={'patches.txt': f'{good}\n'}, gate=lambda *a: seen.append(a[3]) or 'slow start')
    out = eng.evaluate(q)
    assert out.failed_checks == ['probe_speed'] and seen[-1]['patch_lines'] == []        # no proof marker: no lines


def test_r2_toy_patches_print_when_asked_and_an_inert_one_never_runs():
    from infrarsi.search.toytrain import ToySim, tasks
    from test_search import TOGGLE, setup
    sim = ToySim(tasks(36)[0])
    sp = P.SimProposer(sim)
    assert not sp.ask_proof
    HLS3(sp, off=['hls3_deferred'])                                 # (F10's two lines: R3's third, F21, off)
    assert sp.ask_proof                                             # hls3 with F10 asks for the two lines
    other = P.SimProposer(sim)
    HLS3(other, off=['hls3_self_proof'])
    assert not other.ask_proof
    sp.proof_rate = 1.0
    assert sp._hyp({}, 'prefetch_data', (), 'patch:prefetch_data', 0.05, 0.01, 0.9)['patch'] == {
        'patches.txt': 'prefetch_data\nproof:prefetch_data\n'}
    assert other._hyp({}, 'prefetch_data', (), 'patch:prefetch_data', 0.05, 0.01, 0.9)['patch'] == {
        'patches.txt': 'prefetch_data\n'}
    sim, s, _ = setup(TOGGLE)
    good = next(p for p, ok in sorted(sim.patch_ok.items()) if ok)
    cfg = sim.reference()
    t_with, t_without = sim.truth(cfg, (good,))[0], sim.truth(cfg, ())[0]
    assert t_with < t_without
    sim.inert = frozenset({good})                                   # (toybench --inert)
    assert sim.truth(cfg, (good,))[0] == t_without and sim.truth(cfg, (good,), intended=True)[0] == t_with
    r = s.runner.run(s.task, cfg, {'patches.txt': f'{good}\nproof:{good}\n'}, 'probe', 'r', candidate_id='c9', steps=30,
                     stop_after=2)
    assert r['evidence']['events']['patch_lines'] == [f'[infrarsi-patch] {good}: found off -> set on']
