"""The session backend changes who answers a call, not what the search does with the answer: bcomp on the memory toy (a
simulated job on a virtual clock, so a run is exactly repeatable) with the scripted backend, and again with the same
answers handed in through a session (agent/backends/session.py) - the prompts the model is sent, in order, the candidates,
every run with its verdict and its time, the ledger and the selection are the same, call for call."""
import contextlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))

from infrarsi.agent.backends import session as S  # noqa: E402
from infrarsi.agent.backends.fake import ScriptedBackend  # noqa: E402
from infrarsi.search.bcomp import BComp  # noqa: E402
from infrarsi.search.toytrain import mem_tasks  # noqa: E402
from session_helpers import Answerer, session_dir  # noqa: E402
from test_bcomp import _toy_script  # noqa: E402


def bcomp_on_the_toy(make_backend, *, screens=10):
    """test_bcomp.bcomp_llm_toy with the backend given: `make_backend(script)` -> (backend, a context to run under)."""
    from infrarsi.agent.budget import VirtualClock
    from infrarsi.search import toybench as tb
    from infrarsi.search.proposers import LLMProposer
    task = mem_tasks()[1]
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
    s = tb.SearchSession(task=env, budget=env.search_budget(), runner=runner, judge=tb.Verifier(0.03, memory_cap_gib=sim.hw['mem']),
                         run_id='toy-b', arm='b', start_config=ref_cfg, start_receipt=ref, reference_receipt=ref, hooks=hooks,
                         clock=clock, stop_after_same_failures=10 ** 6)
    eng = tb.Engine(s, full_steps=tb.FULL, short_steps=tb.SHORT, probe_steps=tb.PROBE, job_gpu_hours=None, multi_fidelity=True)
    seen, inner = [], _toy_script()

    def script(role, system, prompt):
        seen.append((role, system, prompt, s.ledger.used['screens']))
        return inner(role, system, json.loads(prompt))
    be, running = make_backend(script)
    prop = LLMProposer(be, hooks, search_calibrates=True, features=BComp.proposer_features(()))
    st = BComp(prop, facts=sim.memory_facts, seed=1, packs=tb.toy_packs(task, 2), env=tb.toy_env(sim))
    with running:
        trace = eng.run(st)
    return SimpleNamespace(st=st, s=s, be=be, seen=seen, trace=trace.to_dict())


def same(x):
    """`x` (text, or plain data) with every run's attempt id replaced: the runner draws them at random, so two runs of
    one search differ in them and in nothing else."""
    text = x if isinstance(x, str) else json.dumps(x, sort_keys=True, default=str)
    return re.sub(r'"attempt_id": "[0-9a-f]+"', '"attempt_id": "-"', text)


def prompts(r) -> list:
    return [(role, system, same(prompt), screens) for role, system, prompt, screens in r.seen]


def view(r) -> dict:
    """What a run did, without what names the backend (the model, a call's id) and the real seconds an answer took."""
    t = r.trace
    calls = [{k: c.get(k) for k in ('role', 'llm_tokens', 'llm_usd', 'error', 'attempts', 'accounting_complete')}
             for c in t['llm_calls']]
    events = [{k: v for k, v in e.items() if k != 'utc'} for e in r.s.events if e['kind'] != 'llm_call']
    out = {'candidates': t['candidates'], 'screens': t['screens'], 'selected': t['selected'], 'stop_reason': t['stop_reason'],
           'used': t['used'], 'calls': calls, 'events': events, 'steps': r.st.describe()['bcomp']['bcomp']['steps']}
    return {k: same(v) for k, v in out.items()}


def scripted_run():
    return bcomp_on_the_toy(lambda script: (ScriptedBackend(script, tokens_per_call=2000, usd_per_call=0.0),
                                            contextlib.nullcontext()))


def test_bcomp_decides_the_same_whoever_hands_in_the_answers(tmp_path):
    scripted, again = scripted_run(), scripted_run()
    assert prompts(scripted) == prompts(again) and view(scripted) == view(again)   # the yardstick: the toy repeats itself
    d = session_dir(tmp_path)
    # one hand-in per call, as the scripted backend makes one attempt (the toy's script answers `revise` with a reply
    # of another shape: with more hand-ins the stand-in session would be asked again and its script would move on)
    session = bcomp_on_the_toy(lambda script: (S.SessionBackend(directory=d, poll_seconds=0.002, max_attempts=1),
                                               Answerer(d, script, tokens=2000, pause=0.002)))
    roles = [x[0] for x in scripted.seen]
    assert roles[0] == 'prior' and {'revise', 'code_review', 'hypotheses'} <= set(roles) and len(roles) >= 6
    assert prompts(session) == prompts(scripted)         # the same prompts, system and all, at the same screens
    a, b = view(scripted), view(session)
    for key in a:
        assert a[key] == b[key], key
    assert session.trace['used']['screens'] == 10 and any(c['error'] for c in session.trace['llm_calls'])   # (a failed call too)
    reqs = S.requests(d)
    assert [r.role for r in reqs] == roles and all(r.state == 'answered' for r in reqs)
