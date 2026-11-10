"""Within-run structure is not between-run noise (Fable noise decision, D2-05 / D1-20): split halves and the white floor
are robust to a stall that recurs at the same update in every run; an incumbent whose screen two agreeing aborts
contradict is re-run once and the later runs are compared with the re-run."""
import math
import statistics

import pytest

from infrarsi.project.signals import clock_time, robust_rsd, split_half, intervals
from infrarsi.search.engine import Proposal, pooled_sigma_pair, white_floor
from test_search import TOGGLE, setup

# D2-05, c1's screen (dataloader_num_workers 8): the per-update intervals of updates 2..30 on InfraRSI's clock; the
# epoch ends at update 15 and the 8 workers per GPU re-spawn (one stall of 1.3 s in every run of the configuration)
C1 = [0.398, 0.381, 0.383, 0.386, 0.384, 0.382, 0.393, 0.385, 0.384, 0.381, 0.384, 0.383, 0.383, 0.382, 0.676, 1.34, 0.391,
      0.384, 0.386, 0.381, 0.383, 0.383, 0.385, 0.388, 0.383, 0.382, 0.382, 0.38, 0.381]
REF = [0.746, 0.702, 0.678, 0.697, 0.693, 0.689, 0.7, 0.682, 0.694, 0.694, 0.692, 0.705, 0.678, 0.702, 0.379, 0.996, 0.71,
       0.712, 0.718, 0.69, 0.711, 0.68, 0.72, 0.729, 0.706, 0.678, 0.709, 0.683, 0.716]


def rows_of(dt):
    rows, t = [{'step': 1, 't': 0.0}], 0.0
    for i, d in enumerate(dt):
        t += d
        rows.append({'step': i + 2, 't': t})
    return rows


def test_split_half_and_robust_rsd_ignore_a_repeatable_stall():
    rows = rows_of(C1)
    mean, rsd, _ = clock_time(rows, [11, 30])
    assert mean == pytest.approx(0.4459, abs=5e-4) and rsd > 0.45                 # the stall is in the metric and the RSD
    assert split_half(rows, [11, 30]) < -0.25                                     # the plain half carries it (one run alone)
    assert abs(split_half(rows, [11, 30], robust=True)) < 0.01                    # the robust half does not
    assert robust_rsd(intervals(rows, [11, 30])[1]) < 0.01                        # nor in the robust dispersion
    ev = {'rsd_window': rsd, 'rsd_robust': 0.0058, 'window': [11, 30]}
    assert white_floor(ev, robust=True) == pytest.approx(math.sqrt(2) * 0.0058 / math.sqrt(20))   # with a pair in the pool
    assert white_floor(ev) == pytest.approx(math.sqrt(2) * rsd / math.sqrt(20))                   # one run alone: the plain RSD
    assert white_floor({'rsd_window': rsd, 'window': [11, 30]}, robust=True) == pytest.approx(math.sqrt(2) * rsd / math.sqrt(20))
    quiet = C1[:14] + C1[16:]                                                     # the stall (updates 16-17) taken out ...
    late = rows_of(quiet[:22] + [0.676, 1.34] + quiet[22:])                     # ... and put into the second half
    assert abs(split_half(late, [11, 30], robust=True)) < 0.01 and split_half(late, [11, 30]) > 0.25
    drift = [{'step': s, 't': 1.0 * s if s <= 20 else 20 + 1.02 * (s - 20)} for s in range(1, 31)]
    assert split_half(drift, [11, 30]) == pytest.approx(math.log(1.02))         # drift is seen by both
    assert split_half(drift, [11, 30], robust=True) == pytest.approx(math.log(1.02))
    reported = [{'step': s, 'step_time': 0.385 if s not in (15, 16) else 1.0} for s in range(1, 31)]
    assert split_half(reported, [11, 30], robust=True) == 0.0 and split_half(reported, [11, 30]) < -0.2  # the fallback too


def test_case_d2_05_margin_follows_the_pairs_not_the_stall():
    """The data of D2-05: pairs [doctor 2.0 %, promotion re-run 1.2 %], seven candidate screens with the stall and the
    reference; the old pooled estimate was 25 %, the margin x1.657 against a re-run speed-up of x1.587."""
    pairs = [0.020137, 0.01178]
    halves = [split_half(rows_of(REF), [11, 30], robust=True)] + [split_half(rows_of(C1), [11, 30], robust=True)] * 7
    sp = max(pooled_sigma_pair(pairs, halves), white_floor({'rsd_robust': robust_rsd(intervals(rows_of(REF), [11, 30])[1]),
                                                             'rsd_window': 0.155, 'window': [11, 30]}, robust=True))
    assert 0.01 < sp < 0.021 and math.expm1(2 * sp) < 0.05                        # margin about x1.034: x1.587 passes
    old = [0.004682, -0.285229, -0.286483, -0.299291, -0.288915, -0.256205, -0.284515, -0.281658]
    assert pooled_sigma_pair(pairs, old) > 0.09                                    # the mean-based halves, centered: tenfold


def test_halves_are_robust_only_with_a_pair():
    sim, s, eng = setup(TOGGLE, run_noise=None)
    ev = s.candidates['c0'].receipt['evidence']
    ev['split_half'], ev['split_half_robust'] = -0.28, -0.003           # a stall in one half: the plain and the robust value
    assert eng.pairs == [] and eng.split_halves() == [-0.28]             # one run alone: the plain value, as at HEAD
    eng.pairs.append(0.002)                                              # a pair of runs of one configuration licenses the robust
    assert eng.split_halves() == [-0.003] and eng.split_halves(robust=False) == [-0.28]
    n = eng.noise()
    assert n['split_halves'] == [-0.003] and n['split_halves_robust'] == [-0.003]
    r = s.rescore('c0')                                                  # a re-run carries both values too
    assert type(r['split_half']) is float and type(r['split_half_robust']) is float and len(eng.split_halves()) == 2
    del ev['split_half_robust']                                          # a receipt written before the robust value existed
    assert eng.split_halves()[0] == -0.28


FLUX = [0.0482, 0.0483, 0.0512, 0.0529, 0.0443, 0.0656, 0.0520, 0.0476]   # D2-R205b: the second half 5 % slower in all 8 runs


def test_a_trend_common_to_every_run_is_not_noise():
    assert pooled_sigma_pair([0.0092], FLUX) == pytest.approx(0.0092)        # the halves' spread (0.64 %) is below the doctor pair
    assert math.expm1(2 * pooled_sigma_pair([0.0092], FLUX)) < 0.02          # a margin of x1.019, not x1.109
    assert pooled_sigma_pair([], FLUX) == pytest.approx(statistics.stdev(FLUX))
    assert pooled_sigma_pair([], FLUX[:2]) == pytest.approx(math.sqrt(sum(x * x for x in FLUX[:2]) / 2))   # below three: RMS
    assert pooled_sigma_pair([], [0.095, 0.0, 0.0, 0.0]) == pytest.approx(0.0475)   # one random stall among four: the same weight
    assert pooled_sigma_pair([], [0.0, 0.0, 0.0]) == 0.0 and pooled_sigma_pair([0.01], []) == pytest.approx(0.01)


def test_partial_mean_of_an_aborted_run():
    sim, s, eng = setup(TOGGLE)
    good = eng.evaluate(Proposal(change={'fused_adam': True, 'flash_attn': True, 'overlap_grad': True}, mu=0.1))
    slow = eng.evaluate(Proposal(change={'deterministic': True, 'nccl_tree': True}, mu=0.0))      # x1.071: aborted
    truth = sim.truth(dict(sim.reference(), fused_adam=True, flash_attn=True, overlap_grad=True, deterministic=True, nccl_tree=True))[0]
    assert slow.status == 'aborted' and abs(eng.t_partial(slow.cid) / truth - 1) < 0.05
    assert eng.t_partial(good.cid) == pytest.approx(eng.t_window(good.cid, [eng.warmup + 1, 30]))   # a full run: its post-warm-up mean
    assert eng.t_partial('c0') is not None and eng.t_partial('nope') is None


def test_incumbent_is_rerun_after_two_agreeing_aborts(monkeypatch):
    sim, s, eng = setup(TOGGLE)
    good = eng.evaluate(Proposal(change={'fused_adam': True, 'flash_attn': True, 'overlap_grad': True}, mu=0.1))
    assert eng.incumbent() == good.cid
    honest = eng.t(good.cid)
    s.candidates[good.cid].verdict['step_time'] = honest * 0.94        # a screen the configuration does not reproduce
    monkeypatch.setattr(eng, 't_partial', lambda cid: honest)          # neutral children: aborted at the honest speed
    a = eng.evaluate(Proposal(change={'expandable_segments': True}, mu=0.0))     # no effect at low memory: neutral
    assert a.status == 'aborted' and eng.rechecks == []                 # one abort alone is a slower candidate
    b = eng.evaluate(Proposal(change={'seq_parallel': True}, mu=0.0))
    assert b.status == 'aborted' and len(eng.rechecks) == 1
    rec = eng.rechecks[0]
    assert rec['incumbent'] == good.cid and rec['after_aborts'] == [a.cid, b.cid] and rec['moved']
    assert rec['t_rerun'] == eng.t(good.cid) and rec['d'] > rec['threshold'] == pytest.approx(2 * eng.sigma_pair(), abs=1e-5)
    assert rec['d'] == pytest.approx(math.log(rec['t_rerun'] / (honest * 0.94)), abs=1e-6)
    assert eng.pairs == [] and s.screens[-1]['fidelity'] == 'rescore'   # the re-run's pair stays out of the noise pool
    c = eng.evaluate(Proposal(change={'cuda_graphs': True}, mu=0.0))
    assert c.status == 'screened' and eng.incumbent() == good.cid       # compared with the re-run now: not aborted
    for change in ({'nccl_tree': True}, {'nccl_tree': True, 'seq_parallel': True}):
        eng.evaluate(Proposal(change=change, mu=0.0))
    assert len(eng.rechecks) == 1                                       # not again once a re-check moved the time


def test_slower_candidates_that_do_not_agree_do_not_trigger_a_recheck(monkeypatch):
    sim, s, eng = setup(TOGGLE)
    good = eng.evaluate(Proposal(change={'fused_adam': True, 'flash_attn': True, 'overlap_grad': True}, mu=0.1))
    t0 = eng.t(good.cid)
    seen = iter([1.05 * t0, 1.10 * t0])                                 # two slower candidates with their own costs
    monkeypatch.setattr(eng, 't_partial', lambda cid: next(seen))
    slow1 = eng.evaluate(Proposal(change={'deterministic': True}, mu=0.0))                       # x1.05
    slow2 = eng.evaluate(Proposal(change={'deterministic': True, 'nccl_tree': True}, mu=0.0))    # x1.071
    assert slow1.status == 'aborted' and slow2.status == 'aborted' and eng.rechecks == [] and eng.t(good.cid) == t0


def test_an_agreeing_recheck_averages_the_incumbent_and_adds_no_pair(monkeypatch):
    sim, s, eng = setup(TOGGLE)
    good = eng.evaluate(Proposal(change={'fused_adam': True, 'flash_attn': True, 'overlap_grad': True}, mu=0.1))
    t0, screens = eng.t(good.cid), s.ledger.used['screens']
    monkeypatch.setattr(eng, 't_partial', lambda cid: 1.05 * t0)       # two slower candidates of the same cost
    s1 = eng.evaluate(Proposal(change={'deterministic': True}, mu=0.0))
    s2 = eng.evaluate(Proposal(change={'deterministic': True, 'nccl_tree': True}, mu=0.0))
    assert s1.status == 'aborted' and s2.status == 'aborted' and len(eng.rechecks) == 1
    rec = eng.rechecks[0]
    assert not rec['moved'] and eng.t(good.cid) == pytest.approx(math.sqrt(t0 * rec['t_rerun'])) and eng.pairs == []
    assert s.screens[-1]['fidelity'] == 'rescore' and s.ledger.used['screens'] == screens + 2   # a trial, not a screen
    s3 = eng.evaluate(Proposal(change={'deterministic': True, 'seq_parallel': True}, mu=0.0))
    s4 = eng.evaluate(Proposal(change={'deterministic': True, 'nccl_tree': True, 'seq_parallel': True}, mu=0.0))
    assert s3.status == s4.status == 'aborted' and len(eng.rechecks) == 2       # a second one is allowed ...
    s5 = eng.evaluate(Proposal(change={'deterministic': True, 'cuda_graphs': True}, mu=0.0))
    s6 = eng.evaluate(Proposal(change={'deterministic': True, 'nccl_tree': True, 'cuda_graphs': True}, mu=0.0))
    assert len(eng.rechecks) == 2                                               # ... a third is not


def test_the_engine_reads_the_robust_floor_only_with_a_pair(monkeypatch):
    # DF4_r1 #2: one run cannot tell a stall that recurs in every run from a random one (which the window mean carries)
    from infrarsi.search import engine as engine_mod
    sim, s, eng = setup(TOGGLE)
    eng.evaluate(Proposal(change={'fused_adam': True}, mu=0.0))
    eng.run_noise = None                                                # measured noise (the toy's is fixed)
    seen = []
    monkeypatch.setattr(engine_mod, 'white_floor', lambda ev, robust=False: seen.append(robust) or 0.0)
    eng.sigma_pair()
    eng.pairs.append(0.002)
    eng.sigma_pair()
    assert seen == [False, True]


def test_a_corrected_incumbent_is_compared_with_its_rerun(monkeypatch):
    # DF4_r1 #3: after a re-check moved the incumbent's time, a confirmation's pair and a short run's base come from the
    # re-run, not from the screen it replaced
    sim, s, eng = setup(TOGGLE)
    good = eng.evaluate(Proposal(change={'fused_adam': True, 'flash_attn': True, 'overlap_grad': True}, mu=0.1))
    honest = eng.t(good.cid)
    s.candidates[good.cid].verdict['step_time'] = honest * 0.94
    monkeypatch.setattr(eng, 't_partial', lambda cid: honest)
    eng.evaluate(Proposal(change={'expandable_segments': True}, mu=0.0))
    eng.evaluate(Proposal(change={'seq_parallel': True}, mu=0.0))
    assert len(eng.rechecks) == 1 and eng.rechecks[0]['moved']
    t_rerun, ev = eng._rerun[good.cid]
    assert t_rerun == eng.t(good.cid) and ev.get('per_step')
    window = [eng.warmup + 1, 30]
    assert eng.t_window(good.cid, window) == pytest.approx(clock_time(ev['per_step'], window)[0])
    c = eng.evaluate(Proposal(change={'cuda_graphs': True}, mu=0.0))
    assert c.status == 'screened'
    n = len(eng.pairs)
    eng.confirm(c.cid, 0.005)
    r_inc = s.candidates[good.cid].results['rescore'][-1]['step_time']
    assert eng.pairs[n] == pytest.approx(math.log(r_inc / t_rerun))     # not against the 6 % lucky screen


def test_an_environment_arm_counts_its_reruns_as_trials():
    # DF5_r1 #2: an environment's budget counts every full run; the re-run's row names its receipt
    sim, s, eng = setup(TOGGLE)
    s.count_rescores = True
    used = s.ledger.used['screens']
    r = s.rescore('c0')
    assert r.get('valid') and s.ledger.used['screens'] == used + 1
    assert s.screens[-1]['fidelity'] == 'rescore' and s.screens[-1]['attempt_id']


def test_a_rerun_row_is_a_boundary_of_the_summarys_event_sequence(tmp_path):
    # DF5_r2 #1: every trace row, a confirmation re-run too, has its boundary in events.jsonl (the summary's accounting)
    import json
    from infrarsi.ext.summary.records import Store
    sim, s, eng = setup(TOGGLE)
    s.count_rescores = True
    s.screen('c0')                                           # row 0 (the start), then its re-run: row 1
    s.rescore('c0')
    p = tmp_path / 'events.jsonl'
    p.write_text(''.join(json.dumps(e) + '\n' for e in s.events), encoding='utf-8')
    rows = [k for kind, k in Store(tmp_path).event_sequence(p) if kind == 'screen']
    assert rows == list(range(len(s.screens))) and s.screens[-1]['fidelity'] == 'rescore'
