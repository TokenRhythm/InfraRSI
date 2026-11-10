"""Shared set-up for the candidates' tests (test_rpp.py, test_esls.py): the toybench's engine on a ToyTrain task without
its run, and a run that keeps the strategy object."""
from infrarsi.search import toybench as tb


def toy(task, budget: int = 10, *, memory: bool = True):
    """run_one's set-up with the screens bound (B screens, 4B screens' GPU-hours), not run: (engine, sim, a factory of
    the simulated model, the memory facts)."""
    sim = tb.make_sim(task)
    clock = tb.VirtualClock()
    runner = tb.ToyRunner(sim, clock)
    ref_cfg = sim.reference()
    probe_task = tb.EnvTask.from_dict(tb.toy_task_spec(sim, {'screens': 1, 'llm_tokens': 0, 'wall_seconds': 1e12,
                                                             'gpu_hours': 1e9}))
    ref = runner.run(probe_task, ref_cfg, None, 'reference', 'toy', candidate_id='c0', steps=tb.FULL)
    c_full = ref['cost']['gpu_seconds'] / 3600
    env = tb.EnvTask.from_dict(tb.toy_task_spec(sim, {'screens': budget, 'llm_tokens': 0, 'wall_seconds': 1e12,
                                                      'gpu_hours': 4 * budget * c_full}))
    facts = getattr(sim, 'memory_facts', None)
    hooks = tb.MemoryHooks(env, facts, base_files={'patches.txt': ''}) if facts else \
        tb.SessionHooks(env, base_files={'patches.txt': ''})
    session = tb.SearchSession(task=env, budget=env.search_budget(), runner=runner,
                               judge=tb.Verifier(0.03, memory_cap_gib=sim.hw['mem']), run_id='toy-t', arm='t',
                               start_config=ref_cfg, start_receipt=ref, reference_receipt=ref, hooks=hooks, clock=clock,
                               stop_after_same_failures=10 ** 6)
    engine = tb.Engine(session, full_steps=tb.FULL, short_steps=tb.SHORT, probe_steps=tb.PROBE, job_gpu_hours=None,
                       multi_fidelity=True)

    def sp():
        return tb.SimProposer(sim, q=0.7, b=1.5, rho=0.5, memory=memory)
    return engine, sim, sp, facts


def run_kept(monkeypatch, task, make, budget: int = 10):
    """toybench.run_one of the strategy `make(proposer, facts, sim)` (the memory toy's settings); (row, strategy)."""
    box = {}
    orig = tb.candidate_arms

    def arms(sim, sp, facts):
        out = orig(sim, sp, facts)

        def factory():
            box['st'] = make(sp(), facts, sim)
            return box['st']
        out['t'] = (factory, {'multi_fidelity': True})
        return out
    monkeypatch.setattr(tb, 'candidate_arms', arms)
    row = tb.run_one(task, 't', budget, job_factor=None, proposer={'memory': True}, bind='screens')
    return row, box['st']
