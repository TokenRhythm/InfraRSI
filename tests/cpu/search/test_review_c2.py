"""Review C2 (gpt-6.1-sol): trial admission like a screen, the between-run noise floor, the capability's session
settings for engine strategies, and the noise floor learned from promotion re-runs."""
import json
import math

import pytest

from infrarsi import search
from infrarsi.search.engine import Engine
from test_search import TOGGLE, setup


def test_trial_is_admitted_like_a_screen():
    sim, s, eng = setup(TOGGLE, budget=1.2)
    cid = s.propose_config(changes={'fused_adam': True}, base='c0', origin='t', rationale='t')['candidate_id']
    s.ledger.used['gpu_hours'] = s.budget.gpu_hours * 0.9          # a full screen (plus reserve) no longer fits
    row = s.trial(cid, steps=2, fidelity='probe')
    assert 'budget exhausted (gpu_hours)' in row['error'] and s.stop_flag == 'budget:gpu_hours'
    assert s.ledger.used.get('trials', 0) == 0                      # nothing was run


def test_run_noise_floors_sigma_run():
    sim, s, eng = setup(TOGGLE)
    assert eng.sigma_run() == pytest.approx(max(0.015, eng.sigma_step()))
    sim, s, eng = setup(TOGGLE, run_noise=0.05)
    assert eng.sigma_run() == pytest.approx(0.05) and eng.sigma_pair() == pytest.approx(0.05 * 2 ** 0.5)


def test_engine_session_gets_the_capability(monkeypatch):
    seen = {}

    class Stop(Exception):
        pass

    def fake_init(self, session, **kw):
        seen.update(mode=session.mode, limit=session.patch_line_limit, layers=session.layers, run_noise=kw.get('run_noise'))
        raise Stop

    sim, s, _ = setup(TOGGLE)
    monkeypatch.setattr(Engine, '__init__', fake_init)
    hooks = s.hooks
    ctx = search.Context(task=s.task, budget=s.budget, start_config=s.start_config if hasattr(s, 'start_config') else {},
                         runner=s.runner, judge=s.judge, hooks=hooks, run_id='t', out_dir=None,
                         start_receipt=s.reference, reference_receipt=s.reference,
                         options={'capability': 0.0, 'full_steps': 30, 'job_gpu_hours': 1.0, 'run_noise': 0.03})
    with pytest.raises(Stop):
        search.run('hls', ctx)
    assert seen['mode'] == 'weak' and seen['limit'] == 80 and seen['run_noise'] == 0.03
    assert seen['layers'] == tuple(getattr(hooks, 'layers', None) or ('L0', 'L1'))


def test_run_noise_learned_from_promotion_reruns(tmp_path, monkeypatch):
    from infrarsi.project.flow import noise_pairs, remember_noise, run_noise
    from infrarsi.project.manifest import Project
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    raw = {'name': 'p', 'launch': {'command': '{python} train.py {args}'}, 'hardware': {'gpu_type': 'H100'},
           'knobs': {'k': {'type': 'bool', 'flag': '--k'}}, 'workload': {'steps': 4, 'window': [2, 4]}}
    p = Project(raw, tmp_path / 'infrarsi.yaml')
    assert run_noise(p) is None and noise_pairs(p) == []           # measured by the engine, nothing recorded yet
    assert remember_noise(p, [0.05, 0.0004]) == 2
    assert noise_pairs(p) == [0.05, 0.0004]                         # no floor: a quiet node's pairs are kept as measured
    rows = [json.loads(x) for x in (tmp_path / 'home' / 'ledger.jsonl').read_text(encoding='utf-8').splitlines()]
    assert {r['kind'] for r in rows} == {'noise'}
    p.raw['search']['run_noise'] = 0.01                             # the manifest wins
    assert run_noise(p) == 0.01


def test_default_strategy_follows_the_knob_schema():
    toggles = {f't{i}': {'type': 'bool'} for i in range(5)}
    mixed = {**{f't{i}': {'type': 'bool'} for i in range(3)}, 'tp': {'type': 'int', 'choices': [1, 2, 4]},
             'pp': {'type': 'int', 'choices': [1, 2, 4]}}
    assert search.resolve('auto', has_llm=True, knobs=toggles) == 'doe+hls'     # mostly two-level: a design pays
    assert search.resolve('auto', has_llm=True, knobs=mixed) == 'hls'
    assert search.resolve('auto', has_llm=True) == 'hls' and search.resolve('greedy', has_llm=True) == 'greedy'


def test_t5_regime_defaults_leave_the_simulator_unchanged():
    import dataclasses
    from infrarsi.search.toytrain import ToySim, tasks
    t = next(x for x in tasks(36) if x.schema == 'both')
    a = ToySim(t)
    b = ToySim(dataclasses.replace(t, gain_scale=1.0, run_sd=a.run_sd))
    assert a.optimum()[0] == b.optimum()[0] and a.truth(a.reference()) == b.truth(b.reference())
    c = ToySim(dataclasses.replace(t, gain_scale=2.0))
    on = {**c.reference(), 'fused_adam': True}
    g = lambda sim: math.log(sim.truth(sim.reference())[0] / sim.truth(on)[0])          # noqa: E731
    assert g(c) > g(a) > 0                                                          # larger effects at s = 2


def test_doe_factors_include_two_choice_knobs():
    from types import SimpleNamespace
    from infrarsi.search.proposers import LLMProposer
    props = {'fused': {'type': 'boolean'}, 'mbs': {'type': 'integer', 'enum': [1, 2]},
             'nvls': {'type': 'string', 'enum': ['1', '0']}, 'tp': {'type': 'integer', 'enum': [1, 2, 4]},
             'ac': {'type': 'boolean', 'enum': [False, True]}, 'unset': {'type': 'string', 'enum': ['a', 'b']}}
    cfg = {'fused': False, 'mbs': 1, 'nvls': '1', 'tp': 1, 'ac': True, 'unset': None}
    eng = SimpleNamespace(s=SimpleNamespace(candidates={'c0': SimpleNamespace(config=cfg)},
                                            task=SimpleNamespace(knob_schema={'properties': props})))
    f = LLMProposer.factors(None, eng, 'c0')
    assert {x['name']: x['plus'] for x in f} == {'fused': {'fused': True}, 'mbs': {'mbs': 2}, 'nvls': {'nvls': '0'},
                                                 'ac': {'ac': False}}
