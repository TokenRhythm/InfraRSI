"""Shared set-up for the tests of the finalists' shared parts (search/finalist_base.py): a minimal finalist registered only
in the tests (`Dummy`: the pending proposals, the code track as S1 gates it, a configuration round, the tail, the draw),
a scripted model for the simulated engine (`CodeProposer`), the dummy on the simulated engine (`fin`) and on the memory
toy with an LLMProposer over a priced scripted backend (`llm_toy`)."""
import dataclasses
import json
from types import SimpleNamespace

from infrarsi.search.finalist_base import FB_PARTS, FinalistBase
from infrarsi.search.wcs import proposer_flags as wcs_flags
from test_wcs import PriorProposer, engine

DUMMY_COMPONENTS = {**FB_PARTS, 'fbd_code': "wcs's code track (patches only, the self-proof, S1's asks)",
                    'fbd_ledger': 'the memory ledger (S4); off: wcs feasibility',
                    'fbd_rounds': 'a configuration call when nothing else is proposed'}


class Dummy(FinalistBase):
    """The smallest finalist: pending proposals, then the code track (S1 decides whether to ask), then one configuration
    round per full screen (its best open item through the ladder), then the tail, then the draw."""
    name = 'fbdummy'

    @classmethod
    def proposer_features(cls, off=()) -> dict:
        return wcs_flags(['wcs_code'] if 'fbd_code' in set(off or ()) else [])

    def __init__(self, proposer, *, off=(), protected=(), **kw):
        off = set(off or ())
        super().__init__(proposer, fb_off=[x for x in off if x in FB_PARTS],
                         off_wcs=['wcs_code'] if 'fbd_code' in off else [], ledger='fbd_ledger' not in off, **kw)
        self.rounds_on = 'fbd_rounds' not in off
        if protected:
            self.protected_screens = frozenset(protected)
        self.round_at = -1
        self.trail: list[str] = []                       # each step's pick

    def propose(self, engine, k):
        if self.prior is None:
            self._init(engine)
        self.engine = engine
        left = self._left(engine)
        if left <= 0:
            return []
        inc = engine.incumbent()
        while self.pending:
            q = self.pending.pop(0)
            if engine._fresh(q):
                return [q]
        if self.on['wcs_code']:
            q = self._code_step(engine, inc, left)
            if q is not None:
                self.trail.append('code')
                return [q]
        if self.rounds_on:
            items = self._open()
            if not items and self.round_at != engine.full_screens():
                self.round_at = engine.full_screens()
                self._ask(engine, inc)
                items = self._open()
            for item in sorted(items, key=lambda a: -(a['p'] * self._stated(a))):
                q = self._run_item(engine, inc, item, kind='round')
                if q is not None:
                    self.trail.append('round')
                    return [q]
        q = self._tail_step(engine, inc, left)
        if q is not None:
            self.trail.append('tail')
            return [q]
        q = self._fallback(engine, inc)
        if q is not None:
            self.trail.append('draw')
        return [q] if q is not None else []


class CodeProposer(PriorProposer):
    """A scripted model for the simulated engine: the prior (test_wcs.answer or None), patch calls that read the
    strategy's extra fields (`context`, recorded in `seen`) and answer `answers(ctx, n)`, configuration calls that answer
    `configs(n)`; every call counted."""
    context = None

    def __init__(self, prior=None, answers=None, configs=None):
        super().__init__(prior)
        self.answers, self.configs, self.seen = answers, configs, []

    def hypotheses(self, engine, base, ledger=None):
        self.calls += 1
        ctx = self.context(engine, base) if self.context is not None else {}
        self.seen.append(json.loads(json.dumps(ctx, default=str)))
        return self.answers(ctx, len(self.seen)) if self.answers else []

    def config_batch(self, engine, base, *, k=5, history=None, purpose='round'):
        self.calls += 1
        return {'configs': self.configs(self.calls) if self.configs else []}


def fin(prior=None, *, answers=None, configs=None, screens=10, effect=None, cap=100.0, ref_peak=None, **kw):
    """The dummy on test_wcs's simulated engine (its landscape unless `effect`), a card of `cap` GiB and the reference's
    measured peak `ref_peak` (None: none measured)."""
    eng = engine(screens=screens, **({'effect': effect} if effect else {}))
    if ref_peak is not None:
        eng.s.candidates['c0'].receipt['evidence'] = {'memory_peak_measured_gib': float(ref_peak)}
    st = Dummy(CodeProposer(prior, answers, configs), facts={'cap_gib': cap}, seed=3, **kw)
    return st, eng


def llm_toy(script, **kw):
    """The dummy with an LLMProposer over a priced scripted backend (`script(role, system, payload)` -> the answer; each
    call costs `usd_per_call`, capped by what the strategy's budget gives it) on a memory toy, run for `screens` full
    screens: (strategy, session, backend, calls seen as (role, payload, screens done, calls before), trace, proposer)."""
    r = llm_toy_setup(script, **kw)
    r.trace = r.eng.run(r.st)
    return r


def llm_toy_setup(script, *, task=None, screens=10, usd_per_call=1.0, calls_max=None, protected=(), off=(), retries=0,
                  **kw):
    """llm_toy's set-up, not run; `retries`: the backend's own re-asks per call, every one it is granted used (an attempt
    each, as BudgetDriver counts them)."""
    from infrarsi.agent.backends.fake import ScriptedBackend
    from infrarsi.agent.budget import VirtualClock
    from infrarsi.search import toybench as tb
    from infrarsi.search.proposers import LLMProposer
    from infrarsi.search.toytrain import mem_tasks
    task = task or mem_tasks()[1]
    sim = tb.make_sim(task)
    clock = VirtualClock()
    runner = tb.ToyRunner(sim, clock)
    ref_cfg = sim.reference()
    probe_task = tb.EnvTask.from_dict(tb.toy_task_spec(sim, {'screens': 1, 'llm_tokens': 0, 'wall_seconds': 1e12,
                                                             'gpu_hours': 1e9}))
    ref = runner.run(probe_task, ref_cfg, None, 'reference', 'toy', candidate_id='c0', steps=tb.FULL)
    c_full = ref['cost']['gpu_seconds'] / 3600
    env = tb.EnvTask.from_dict(tb.toy_task_spec(sim, {'screens': screens, 'llm_tokens': 10 ** 7, 'wall_seconds': 1e12,
                                                      'gpu_hours': 4 * screens * c_full}))
    hooks = tb.MemoryHooks(env, sim.memory_facts, base_files={'patches.txt': ''})
    s = tb.SearchSession(task=env, budget=env.search_budget(), runner=runner,
                         judge=tb.Verifier(0.03, memory_cap_gib=sim.hw['mem']), run_id='toy-f', arm='f', start_config=ref_cfg,
                         start_receipt=ref, reference_receipt=ref, hooks=hooks, clock=clock, stop_after_same_failures=10 ** 6)
    eng = tb.Engine(s, full_steps=tb.FULL, short_steps=tb.SHORT, probe_steps=tb.PROBE, job_gpu_hours=None,
                    multi_fidelity=True)
    seen = []

    def wrapped(role, system, prompt):
        payload = json.loads(prompt)
        seen.append((role, payload, s.ledger.used['screens'], len(seen)))
        return script(role, system, payload)

    class Priced(ScriptedBackend):
        def __init__(self):
            super().__init__(wrapped, tokens_per_call=2000)
            self.max_retries, self.kw, self.attempts_made, self.granted = retries, [], 0, []

        def complete(self, system, prompt, **kwargs):
            self.kw.append(dict(kwargs))
            res = super().complete(system, prompt, **{k: v for k, v in kwargs.items() if k in ('schema', 'call_dir', 'role',
                                                                                               'expect_json')})
            n = 1 + int(self.max_retries or 0)           # (every re-ask it was granted, used)
            self.granted.append((kwargs.get('role'), int(self.max_retries or 0)))
            self.attempts_made += n
            cap = kwargs.get('max_usd')
            return dataclasses.replace(res, llm_usd=usd_per_call if cap is None else min(usd_per_call, float(cap)),
                                       attempts=n)
    be = Priced()
    prop = LLMProposer(be, hooks, search_calibrates=True, features=Dummy.proposer_features(off))
    st = Dummy(prop, facts=sim.memory_facts, seed=1, protected=protected, off=off,
               **({'calls_max': calls_max} if calls_max else {}), **kw)
    return SimpleNamespace(st=st, s=s, be=be, seen=seen, trace=None, prop=prop, sim=sim, eng=eng)
