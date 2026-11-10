"""Q1-Q5 - search oracles on the toy: the budget is never exceeded, every candidate keeps the contract, the fp8 trap is
never promoted at 3 %, the claimed gain is re-measured independently, the fraction of the attainable gain is recorded
per strategy, and a knob-space strategy is deterministic for a seed."""
import json
import math

import pytest

from _support.consistency import check_run
from _support.toy import make_toy, receipts, report
from infrarsi.harness import signing
from infrarsi.project import Project, ProjectRunner
from infrarsi.project.flow import make_verifier, run_project

STRATEGIES = ['random', 'tpe', 'hls', 'doe']


def toy_cost(cfg: dict, seq: int = 512) -> float:
    """The toy trainer's modelled time per update (examples/toy_cpu/train.py), in its time units: the oracle."""
    mbs, ga, prec = int(cfg['micro_batch_size']), int(cfg['grad_accum']), cfg.get('precision', 'fp32')
    workers, compile_ = int(cfg.get('workers') or 1), bool(cfg.get('compile'))
    mem = 3.0 + 2.5 * mbs * (0.6 if prec != 'fp32' else 1.0)
    if mem > 16.0 or prec == 'fp8':                              # OOM, or the loss check (fp8 is 3.5 % off)
        return math.inf
    chunks = max(1, seq // 128)
    per_sample = {'fp32': 1.0, 'bf16': 0.6, 'fp8': 0.45}[prec] * (0.02 * chunks * chunks / 4) * (0.85 if compile_ else 1.0)
    compute = per_sample * mbs * ga / (0.7 + 0.3 * min(mbs, 4))
    data = max(0.0, 0.6 - 0.15 * workers)
    optim = 0.3 if cfg.get('fused_optimizer') else 0.9
    return compute + data + optim


def toy_optimum() -> tuple[dict, float]:
    best, best_t = None, math.inf
    for mbs in (1, 2, 4, 8):
        for ga in (1, 2, 4, 8):
            if mbs * ga != 8:
                continue
            for prec in ('fp32', 'bf16', 'fp8'):
                for fused in (False, True):
                    for workers in (1, 2, 4):
                        for comp in (False, True):
                            cfg = {'micro_batch_size': mbs, 'grad_accum': ga, 'precision': prec, 'fused_optimizer': fused, 'workers': workers, 'compile': comp}
                            t = toy_cost(cfg)
                            if t < best_t:
                                best, best_t = cfg, t
    return best, best_t


@pytest.mark.parametrize('strategy', STRATEGIES)
@pytest.mark.parametrize('screens', [2, 6])
def test_budget_and_contract_hold(tmp_path, home, strategy, screens):
    root = make_toy(tmp_path / f'{strategy}{screens}')
    p = Project.load(root)
    r = run_project(p, strategy=strategy, budget={'screens': screens}, log=lambda m: None, seed=7)
    check_run(r['run_dir'], budget_screens=screens)
    assert r['search']['used']['screens'] <= screens
    # the configurations that ran, as the receipts record them (a change is relative to its parent, not the reference)
    receipts = [json.loads(f.read_text(encoding='utf-8')) for f in (root / '.infrarsi' / 'runs').glob('*/search/receipts/*.json')]
    assert receipts and all(signing.verify(rec) for rec in receipts)
    for rec in receipts:
        full = rec['config']
        assert full['micro_batch_size'] * full['grad_accum'] == 8, f'{rec["candidate_id"]} ran outside the global-batch invariant'
        for k, v in full.items():
            choices = p.knobs[k].get('choices')
            assert choices is None or v in choices or (p.knobs[k]['type'] == 'bool' and isinstance(v, bool)), (k, v)
    ran = {rec['candidate_id'] for rec in receipts}
    for c in r['candidates']:
        if c.get('status') in ('screened', 'aborted'):
            assert c['id'] in ran, f'{c["id"]} is in the table but never ran'


@pytest.mark.parametrize('strategy', STRATEGIES)
def test_the_fp8_trap_is_never_promoted_and_the_claim_reproduces(tmp_path, home, strategy):
    root = make_toy(tmp_path / strategy)
    p = Project.load(root)
    r = run_project(p, strategy=strategy, budget={'screens': 8}, log=lambda m: None, seed=3)
    check_run(r['run_dir'], budget_screens=8)
    best, t_opt = toy_optimum()
    t_ref = toy_cost(r['reference']['config'])
    fp8_runs = [c for c in r['candidates'] if (c.get('change') or {}).get('precision', [None, None])[1] == 'fp8']
    for c in fp8_runs:
        assert c['valid'] is False and 'loss_within_tolerance' in (c.get('failed_checks') or []), c
    if strategy in ('tpe', 'hls'):                                  # reachable on this toy in 8 screens: a gain is required
        assert r['outcome'] == 'improved', r['promotion'].get('why')
    if r['outcome'] == 'improved':
        cfg = r['promotion']['config']
        assert cfg['precision'] != 'fp8'
        t_cand = toy_cost(cfg)
        frac = math.log(t_ref / t_cand) / math.log(t_ref / t_opt)
        print(f'\n[{strategy}] promoted {cfg}: modelled x{t_ref / t_cand:.3f} of attainable x{t_ref / t_opt:.3f} (frac {frac:.2f}); '
              f'claimed x{r["promotion"]["speedup"]:.3f}; fp8 screened {len(fp8_runs)} time(s), all rejected')
        assert frac > 0, 'the promoted configuration is not faster in the toy\'s own cost model'
        if strategy in ('tpe', 'hls'):
            assert frac >= 0.3, f'{strategy} reached only {frac:.2f} of the attainable gain in 8 screens'
        # an independent pair outside the search: the candidate judged against its own fresh reference
        runner = ProjectRunner(p, out_dir=root / '.infrarsi' / 'indep')
        task, v = p.task(), make_verifier(p)
        a = runner.run(task, r['reference']['config'], None, 'rescore', 'i', candidate_id='ref')
        b = runner.run(task, cfg, None, 'rescore', 'i', candidate_id='cand')
        va, vb = v(a, None, a), v(b, a, a)
        assert va.valid and vb.valid, vb.reasons
        assert va.step_time / vb.step_time > 1.0
    else:
        print(f'\n[{strategy}] no_change: {r["promotion"].get("why")}')


def test_fp8_passes_at_a_5_percent_tolerance(tmp_path, home):
    root = make_toy(tmp_path / 'tol')
    p = Project.load(root)
    runner = ProjectRunner(p, out_dir=root / '.infrarsi' / 't')
    task = p.task()
    ref = runner.run(task, p.reference_config(), None, 'reference', 'r', candidate_id='c0')
    fp8 = runner.run(task, dict(p.reference_config(), precision='fp8'), None, 'screen', 'r', candidate_id='c1')
    assert not make_verifier(p, tolerance=0.03)(fp8, ref, ref).valid
    assert make_verifier(p, tolerance=0.05)(fp8, ref, ref).valid
    assert make_verifier(p, tolerance=0.0)(ref, None, ref).valid              # a zero tolerance accepts the reference itself


def test_random_search_is_deterministic_for_a_seed(tmp_path, home):
    seqs = []
    for i in range(2):
        root = make_toy(tmp_path / f'd{i}')
        r = run_project(Project.load(root), strategy='random', budget={'screens': 4}, log=lambda m: None, seed=11)
        seqs.append([c.get('change') for c in r['candidates'] if c['id'] != 'c0'])
    assert seqs[0] == seqs[1], seqs


def test_a_knob_space_that_is_exhausted_ends_the_search_and_says_so(tmp_path, home):
    def small(doc):
        doc['knobs'] = {'workers': {'type': 'int', 'choices': [1, 2], 'flag': '--workers', 'default': 1}}
        doc['workload']['invariants'] = {}
        doc['code']['mutable'] = []
    root = make_toy(tmp_path / 'small', edit=small)
    r = run_project(Project.load(root), strategy='random', budget={'screens': 6}, log=lambda m: None)
    check_run(r['run_dir'], budget_screens=6)
    assert r['search']['used']['screens'] == 1, r['search']['used']       # one configuration besides the reference
    assert r['search']['stop_reason'] == 'strategy_done'
    assert 'stop: the strategy had nothing left to test (strategy_done)' in report(r['run_dir'])


def tight(doc):
    """The toy's batch knobs, workers and fused optimizer (96 configurations) under two invariants, the global batch and
    grad_accum x workers: 6 configurations keep both, the reference among them."""
    doc['knobs'] = {k: doc['knobs'][k] for k in ('micro_batch_size', 'grad_accum', 'workers', 'fused_optimizer')}
    doc['workload']['invariants']['loader'] = 'grad_accum * workers'


@pytest.mark.parametrize('strategy', ['random', 'tpe'])
def test_a_tight_invariant_is_sampled_inside_until_nothing_feasible_is_left(tmp_path, home, strategy):
    """U-1-20: random used to spend about 2000 refused draws (l0_invalid) after the feasible ones and stop
    no_valid_sample, TPE stopped no_new_valid_sample after 3 of the 5 feasible configurations; now each feasible
    configuration runs once, nothing refused is proposed, and the search says it had nothing left to test."""
    root = make_toy(tmp_path / strategy, edit=tight)
    r = run_project(Project.load(root), strategy=strategy, budget={'screens': 8}, log=lambda m: None, seed=4)
    check_run(r['run_dir'], budget_screens=8)
    knobs = ('micro_batch_size', 'grad_accum', 'workers', 'fused_optimizer')
    feasible = {(m, 8 // m, m, f) for m in (1, 2, 4) for f in (False, True)}         # mbs * ga == 8 == ga * workers
    ran = [tuple(rec['config'][k] for k in knobs) for cid, rec in receipts(r['run_dir']).items() if cid != 'c0']
    assert sorted(ran) == sorted(feasible - {tuple(r['reference']['config'][k] for k in knobs)}), ran
    s = r['search']
    assert s['stop_reason'] == 'strategy_done' and s['used']['screens'] == len(feasible) - 1, s
    assert not s['invalid_proposals'], s['invalid_proposals']
    assert 'stop: the strategy had nothing left to test (strategy_done); proposals refused before any run: none' in report(r['run_dir'])
