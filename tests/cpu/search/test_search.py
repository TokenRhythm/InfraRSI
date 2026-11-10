"""infrarsi.search on ToyTrain: Plackett-Burman designs, the engine (fidelities, early abort, noise-aware incumbent,
numeric stop), the strategies (greedy finishes early, HLS calibrates, DOE finds the helpful toggles) and toybench."""
import math

import pytest

from infrarsi.agent.budget import VirtualClock
from infrarsi.agent.search import SearchSession
from infrarsi.ext.common.env_task import EnvTask
from infrarsi.ext.common.session_hooks import SessionHooks
from infrarsi.search.engine import Engine, Proposal, Strategy, expected_improvement
from infrarsi.search.proposers import SimProposer
from infrarsi.search.strategies import DOE, HLS, Greedy, pb_design
from infrarsi.search.toybench import run_one
from infrarsi.search.toytrain import ToyRunner, ToySim, ToyTask, toy_task_spec
from infrarsi.verify import Verifier


def setup(task: ToyTask, budget: float = 16, tokens: int = 0, runner_kw=None, **engine_kw):
    sim = ToySim(task)
    clock = VirtualClock()
    runner = ToyRunner(sim, clock, **(runner_kw or {}))
    ref_cfg = sim.reference()
    probe = EnvTask.from_dict(toy_task_spec(sim, {'screens': 1, 'llm_tokens': 0, 'wall_seconds': 1e12, 'gpu_hours': 1e9}))
    ref = runner.run(probe, ref_cfg, None, 'reference', 't', candidate_id='c0', steps=30)
    c_full = ref['cost']['gpu_seconds'] / 3600
    t = EnvTask.from_dict(toy_task_spec(sim, {'screens': int(4 * budget), 'llm_tokens': tokens, 'wall_seconds': 1e12,
                                             'gpu_hours': budget * c_full}))
    s = SearchSession(task=t, budget=t.search_budget(), runner=runner, judge=Verifier(0.03, memory_cap_gib=sim.hw['mem']),
                      run_id='t', arm='t', start_config=ref_cfg, start_receipt=ref, reference_receipt=ref,
                      hooks=SessionHooks(t, base_files={'patches.txt': ''}), clock=clock, stop_after_same_failures=10 ** 6)
    return sim, s, Engine(s, full_steps=30, short_steps=10, probe_steps=2, **{'job_gpu_hours': 300 * c_full, **engine_kw})


TOGGLE = ToyTask('7b', 'h100', 'toggle', 'none', 'low')
BOTH = ToyTask('7b', 'h100', 'both', 'hostsync', 'low')


def test_pb_design_is_balanced_and_orthogonal():
    for f in (3, 7, 11, 15, 19):
        d = pb_design(f)
        n = len(d)
        assert n % 4 == 0 and n - 1 >= f
        for j in range(f):
            assert sum(r[j] for r in d) == 0                     # each factor half +, half -
            for k in range(j + 1, f):
                assert sum(r[j] * r[k] for r in d) == 0          # main effects orthogonal


def test_expected_improvement():
    assert expected_improvement(0.0, 0.0) == 0.0
    assert expected_improvement(0.1, 0.0) == pytest.approx(0.1)
    assert expected_improvement(0.0, 0.1) == pytest.approx(0.1 / math.sqrt(2 * math.pi))
    assert expected_improvement(0.05, 0.02, q=0.5) < expected_improvement(0.05, 0.02)


def test_full_screen_incumbent_and_abort():
    sim, s, eng = setup(TOGGLE)
    t0 = eng.t('c0')
    good = eng.evaluate(Proposal(change={'fused_adam': True, 'flash_attn': True, 'overlap_grad': True}, mu=0.1))
    assert good.status == 'screened' and good.gain > 2 * eng.sigma_pair() and eng.incumbent() == good.cid
    small = eng.evaluate(Proposal(change={'expandable_segments': True}, mu=0.01))   # no true effect at low memory
    assert small.status in ('screened', 'aborted') and eng.incumbent() == good.cid             # never a switch within noise
    slow = eng.evaluate(Proposal(change={'deterministic': True, 'nccl_tree': True}, mu=0.0))
    assert slow.status == 'aborted' and s.candidates[slow.cid].status == 'aborted'
    assert s.screens[-1]['aborted'] and s.stop_flag is None and t0


def test_probe_catches_infeasible_and_is_skipped_when_expensive():
    sim, s, eng = setup(BOTH)
    out = eng.evaluate(Proposal(change={'tp_overlap': True, 'seq_parallel': False}, p_feasible=0.3, fidelity=('probe', 'full')))
    assert out.status == 'infeasible' and out.fidelity == 'probe'
    assert s.ledger.used.get('trials') == 1 and s.ledger.used['screens'] == 0
    s.screens.append({'fidelity': 'probe', 'gpu_hours': 10 * eng.costs()['full'], 'valid': False})
    assert eng.plan(Proposal(fidelity=('probe', 'full'))) == ('full',)


def test_stop_rule_after_floor():
    sim, s, eng = setup(TOGGLE, floor_screens=0)

    class Nothing(Strategy):
        def propose(self, engine, k):
            return [Proposal(change={'nccl_tree': True}, mu=-0.05, sigma=0.001)]
    tr = eng.run(Nothing()).to_dict()
    assert tr['stop_reason'] == 'stop_rule' and tr['extra']['stop_numbers']['best_value_per_cost'] < 1


def test_greedy_finishes_early_hls_keeps_going():
    tasks = [ToyTask(m, g, sch, 'none', 'low') for m, g, sch in (('7b', 'h100', 'toggle'), ('1b', 'l20x', 'both'),
                                                                  ('30b-moe', 'h20', 'toggle'), ('7b', 'l20x', 'layout'))]
    g = [run_one(t, 'greedy', 16) for t in tasks]
    h = [run_one(t, 'hls', 16) for t in tasks]
    assert any(r['stop'] == 'strategy_done' and r['gpu_h_share'] < 0.9 for r in g)       # the agent-like early finish
    assert sum(r['frac'] for r in h) >= sum(r['frac'] for r in g)


def test_hls_calibration_shrinks_overprediction():
    sim, s, eng = setup(BOTH, budget=12)
    hls = HLS(SimProposer(sim, q=0.9, b=3.0, rho=0.2))
    eng.run(hls)
    assert hls.pairs and hls.kappa() < 0.9                      # predictions were 3x too high


def test_doe_finds_helpful_toggles():
    sim, s, eng = setup(TOGGLE, budget=30)
    doe = DOE(SimProposer(sim, q=1.0))
    eng.run(doe)
    eff = doe.effects
    assert eff['flash_attn'] < 0 and eff['fused_adam'] < 0 and eff['deterministic'] > 0


def test_toybench_row_shape():
    r = run_one(TOGGLE, 'tpe', 8)
    assert {'frac', 'gpu_h_share', 'invalid_share', 'aborted_share', 'curve', 'stop'} <= set(r)
    assert 0 <= r['frac'] <= 1.0 + 1e-9 and r['curve']['1.0'] >= r['curve']['0.25'] - 1e-9


def test_greedy_threshold_is_the_noise():
    sim, s, eng = setup(TOGGLE)
    gr = Greedy(SimProposer(sim, q=1.0, b=0.05, rho=0.0))          # pessimistic model: every prediction below noise
    assert gr.propose(eng, 8) == []
