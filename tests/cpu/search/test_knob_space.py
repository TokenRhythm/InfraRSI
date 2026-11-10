"""search.knob_space (U-1-20): random and TPE sample inside the manifest's constraints and invariants. A project whose
two invariants leave 6 of its 96 configurations feasible (the reference among them) and a deterministic fake runner:
every feasible configuration is measured once and the search ends strategy_done, refused draws are never proposed (no
l0_invalid or duplicate counts), TPE starts from feasible configurations and is told FAIL for refused ones, a seed
fixes the proposals, the pre-check accepts exactly what propose_config does, a space of 200 000 is examined to its last
configuration and a larger one up to CHECK_CAP."""
import itertools
import math
import types

import pytest

from infrarsi.agent.budget import VirtualClock
from infrarsi.agent.judge import Verdict
from infrarsi.agent.search import SearchSession
from infrarsi.project import Project
from infrarsi.project.hooks import ProjectHooks
from infrarsi.search import knob_space

KNOBS = {'micro_batch_size': {'type': 'int', 'choices': [1, 2, 4, 8], 'flag': '--mbs', 'default': 1},
         'grad_accum': {'type': 'int', 'choices': [1, 2, 4, 8], 'flag': '--ga', 'default': 8},
         'workers': {'type': 'int', 'choices': [1, 2, 4], 'flag': '--workers', 'default': 1},
         'fused': {'type': 'bool', 'flag': '--fused'}}
INVARIANTS = {'global_batch': 'micro_batch_size * grad_accum', 'loader': 'grad_accum * workers'}   # both stay 8
REF = {'micro_batch_size': 1, 'grad_accum': 8, 'workers': 1, 'fused': False}
FEASIBLE = [{'micro_batch_size': m, 'grad_accum': 8 // m, 'workers': m, 'fused': f} for m in (1, 2, 4) for f in (False, True)]


def cost(cfg: dict) -> float:                       # 1.0 for the knobs of LARGE
    return (1.0 + 0.1 * cfg.get('grad_accum', 0) / cfg.get('micro_batch_size', 1) - 0.05 * cfg.get('workers', 0)
            - 0.2 * cfg.get('fused', False))


class Runner:
    def run(self, task, config, overlay, purpose, run_id, **kw):
        return {'valid': True, 'config': dict(config), 'cost': {'gpu_seconds': 1.0, 'wall_seconds': 1.0}}


def judge(receipt, parent=None, reference=None):
    return Verdict(valid=True, reasons=[], step_time=cost(receipt['config']))


def session(tmp_path, screens: int = 8, invariants=INVARIANTS, knobs=KNOBS, layers=('L0', 'L1'), clock=None,
            **budget) -> SearchSession:
    p = Project({'name': 'tight', 'launch': {'command': 'true'}, 'knobs': knobs, 'workload': {'invariants': invariants}},
                tmp_path / 'infrarsi.yaml')
    task, ref = p.task(), p.reference_config()
    return SearchSession(task=task, budget=task.search_budget(screens=screens, **budget), runner=Runner(), judge=judge,
                         run_id='t', arm='knobs', start_config=ref,
                         start_receipt=Runner().run(task, ref, None, 'reference', 't'), hooks=ProjectHooks(p, task),
                         layers=layers, clock=clock)


def proposed(s: SearchSession) -> list[dict]:
    return [c.config for c in s.candidates.values() if c.id != 'c0']


def search(strategy: str, s: SearchSession, seed: int) -> str:
    if strategy == 'tpe':
        pytest.importorskip('optuna')
    return {'random': knob_space.random_search, 'tpe': knob_space.tpe_search}[strategy](s, seed=seed)


def test_check_config_is_the_rule_propose_config_applies_and_counts_nothing(tmp_path):
    s = session(tmp_path)
    c0 = s.candidates['c0']
    new, bad = dict(REF, fused=True), dict(REF, grad_accum=4)
    assert s.check_config(new, c0)[0] not in s.by_key and s.check_config(REF, c0)[0] in s.by_key   # new; the reference
    key, _, errors, _, _ = s.check_config(bad, c0)
    assert key is None and 'workload invariant changed' in errors[0]
    assert s.invalid == {} and list(s.candidates) == ['c0']                    # asked three times: nothing counted
    assert 'candidate_id' in s.propose_config(config=new, base='c0')
    assert 'errors' in s.propose_config(config=bad, base='c0') and 'duplicate_of' in s.propose_config(config=REF, base='c0')
    assert s.invalid == {'l0_invalid': 1, 'duplicate': 1}


@pytest.mark.parametrize('layers', [('L0', 'L1'), ('L1',)])
def test_check_config_accepts_exactly_what_propose_config_accepts(tmp_path, layers):
    """Also with the session's knob actions closed (U2 review r1: check_config accepted what propose_config refused
    there, and both strategies raised KeyError 'candidate_id'); propose_config's own accounting is unchanged."""
    s = session(tmp_path, layers=layers)
    c0, space = s.candidates['c0'], knob_space.choices(s.task)
    for cfg in [REF] + [dict(zip(space, vals)) for vals in itertools.product(*space.values())]:
        key = s.check_config(dict(cfg), c0)[0]
        fresh = key is not None and key not in s.by_key
        assert ('candidate_id' in s.propose_config(config=cfg, base='c0')) == fresh, cfg
    assert s.invalid == ({'duplicate': 2, 'l0_invalid': 90} if 'L0' in layers else {'layer_closed': 97})


@pytest.mark.parametrize('strategy', ['random', 'tpe'])
@pytest.mark.parametrize('seed', [1, 4, 7])          # each makes TPE stall: its forced feasible draws finish the space
def test_every_feasible_configuration_is_measured_once_then_strategy_done(tmp_path, strategy, seed):
    s = session(tmp_path, screens=8)
    assert search(strategy, s, seed) == 'strategy_done'
    got = proposed(s)
    assert sorted(map(str, got)) == sorted(str(c) for c in FEASIBLE if c != REF)          # each once, nothing else
    assert s.ledger.used['screens'] == len(FEASIBLE) - 1
    assert s.invalid == {}       # before: random ~2000 l0_invalid, ~130 duplicates; TPE stopped after 3 of the 5


def test_tpe_starts_feasible_and_is_told_fail_for_refused_suggestions(tmp_path, monkeypatch):
    optuna = pytest.importorskip('optuna')
    from optuna.trial import TrialState
    studies, create = [], optuna.create_study
    monkeypatch.setattr(optuna, 'create_study', lambda **kw: studies.append(create(**kw)) or studies[-1])
    s = session(tmp_path)
    assert knob_space.tpe_search(s, seed=4) == 'strategy_done'
    trials = studies[0].trials
    assert trials[0].params == REF                                             # the reference, then 3 startup trials
    assert all(t.params in FEASIBLE and t.value == cost(t.params) for t in trials[1:4])
    refused = [t for t in trials if t.params not in FEASIBLE]
    assert refused and all(t.state == TrialState.FAIL for t in refused)       # never run, never counted
    assert all(t.params not in FEASIBLE for t in trials if t.state == TrialState.FAIL)    # the measured keep values


@pytest.mark.parametrize('strategy', ['random', 'tpe'])
def test_a_seed_fixes_the_proposals(tmp_path, strategy):
    runs = []
    for _ in range(2):
        s = session(tmp_path, screens=5)
        search(strategy, s, 11)
        runs.append(proposed(s))
    assert runs[0] == runs[1] and len(runs[0]) == 5


@pytest.mark.parametrize('strategy', ['random', 'tpe'])
@pytest.mark.parametrize('case', [{'invariants': {k: k for k in KNOBS}}, {'layers': ('L1',)}],
                         ids=['only-the-reference-feasible', 'knob-actions-closed'])
def test_nothing_to_run_ends_strategy_done(tmp_path, strategy, case):
    s = session(tmp_path, **case)
    assert search(strategy, s, 1) == 'strategy_done'                           # before: 2000 refused proposals
    assert s.ledger.used['screens'] == 0 and s.invalid == {}


LARGE = {**{f'x{i}': {'type': 'int', 'choices': list(range(10)), 'flag': f'--x{i}', 'default': 0} for i in range(5)},
         'b': {'type': 'bool', 'flag': '--b'}}


@pytest.mark.parametrize('strategy', ['random', 'tpe'])
def test_a_large_space_is_examined_to_its_last_configuration(tmp_path, strategy):
    """U2 review r1: 200 000 configurations, one feasible besides the reference (the switch flipped); round 1's cap of
    100 000 refused draws in a row gave up on it with seed 3 (no_valid_sample / no_new_valid_sample)."""
    s = session(tmp_path, invariants={'sum': 'x0 + x1 + x2 + x3 + x4'}, knobs=LARGE)
    assert math.prod(len(v) for v in knob_space.choices(s.task).values()) == 200_000
    assert search(strategy, s, 3) == 'strategy_done' and s.ledger.used['screens'] == 1 and s.invalid == {}
    assert proposed(s) == [{'x0': 0, 'x1': 0, 'x2': 0, 'x3': 0, 'x4': 0, 'b': True}]


@pytest.mark.parametrize('strategy', ['random', 'tpe'])
def test_a_wall_budget_used_up_by_the_last_scan_is_the_stop_reason(tmp_path, monkeypatch, strategy):
    """U2 review r2: the stream checks configurations inside next(), after the loop's budget check; when the wall budget
    ran out during the final scan the search said strategy_done (the parent said budget:wall_seconds)."""
    clock, scan = VirtualClock(), knob_space._feasible

    def slow(session, space, rng):                 # one simulated second per check in the sampling stream
        def check(*a):
            clock.advance(1.0)
            return session.check_config(*a)
        return scan(types.SimpleNamespace(candidates=session.candidates, by_key=session.by_key, check_config=check),
                    space, rng)
    monkeypatch.setattr(knob_space, '_feasible', slow)
    s = session(tmp_path, clock=clock, wall_seconds=95)                         # the 96 checks end at 96 s
    assert search(strategy, s, 4) == 'budget:wall_seconds' and s.ledger.used['screens'] == len(FEASIBLE) - 1
    assert clock() == 96 and s.ledger.exhausted() == ['wall_seconds']


@pytest.mark.parametrize('strategy,stop', [('random', 'no_valid_sample'), ('tpe', 'no_new_valid_sample')])
def test_a_space_above_the_check_cap_ends_with_the_fallback_reason(tmp_path, monkeypatch, strategy, stop):
    monkeypatch.setattr(knob_space, 'CHECK_CAP', 50)                            # 50 of the 96 configurations examined
    s = session(tmp_path, screens=16)
    assert search(strategy, s, 2) == stop and s.invalid == {}                   # it cannot know that nothing is left
    assert proposed(s) and all(c in FEASIBLE for c in proposed(s))
