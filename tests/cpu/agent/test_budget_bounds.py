"""Worst-case budget admission (driver-side Codex review R1, finding 15): indivisible GPU attempts are admitted with
world x (wall cap + stop grace) while both locked-pair members stay reserved; LLM calls reserve their maximum output
and get the remaining allowance as their output cap."""
import json

import pytest

from infrarsi.agent.backends import ScriptedBackend
from infrarsi.agent.backends.openai_compat import OpenAICompatBackend
from infrarsi.agent.budget import Budget, VirtualClock
from infrarsi.agent.harness_runner import HarnessScreenRunner
from infrarsi.agent.judge import ReceiptJudge
from infrarsi.agent.runner import FakeScreenRunner, synthetic_receipt
from infrarsi.agent.search import SearchSession
from infrarsi.agent.tools import action_schema
from agent_helpers import DENSE_8B, FR_DENSE


class BoundedFake(FakeScreenRunner):
    """FakeScreenRunner plus the HarnessScreenRunner admission interface; records the wall cap of every attempt."""
    stop_grace_s = 300.0
    wall_seconds_cap = 3600

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.caps, self.used_before, self.session = [], [], None

    worst_case_gpu_hours = HarnessScreenRunner.worst_case_gpu_hours
    bounded_cap = HarnessScreenRunner.bounded_cap
    pair_wall_cap = HarnessScreenRunner.pair_wall_cap

    def run(self, task, config, overlay, purpose, run_id, wall_cap=None):
        self.caps.append(wall_cap)
        self.used_before.append((self.session.ledger.used['gpu_hours'], self.session.pair_reserve['gpu_hours'],
                                 self.session._expected_wall()) if self.session else (0.0, 0.0, 0.0))
        return super().run(task, config, overlay, purpose, run_id)


def _session(tasks, runner, clock, gpu_hours, wall_s=10 * 3600):
    t = tasks['D2']
    start = synthetic_receipt(t.raw, {}, step_time_s=5.0, peak_reserved_gib=60, run_id='r', attempt_id='ed')
    start['cost'] = {'gpu_seconds': 8 * 600.0, 'wall_seconds': 600.0}        # 10-minute screens on 8 GPUs
    return SearchSession(task=t, budget=Budget(screens=16, llm_tokens=10 ** 6, wall_seconds=wall_s, gpu_hours=gpu_hours),
                         runner=runner, judge=ReceiptJudge(), run_id='r', arm='A0', start_receipt=start,
                         start_config=dict(FR_DENSE, tensor_model_parallel_size=2, sequence_parallel=True),
                         model_facts=DENSE_8B, clock=clock)


def test_attempt_caps_shrink_to_the_worst_case_that_fits(tasks):
    clock = VirtualClock()
    r = BoundedFake(tasks['D2'], DENSE_8B, clock=clock)
    s = _session(tasks, r, clock, gpu_hours=10.0)
    r.session = s
    fb = s.propose_config(changes={'micro_batch_size': 2}, base='c0')
    s.screen(fb['candidate_id'])
    pair_cap = s.pair_wall_cap
    assert pair_cap == int(min(3600, max(1.5 * 600.0, 600.0 + 300.0)))      # 900 s per locked-pair member
    reserve = 2 * 8 * (pair_cap + 300.0) / 3600.0
    first = r.caps[-1]
    assert 8 * (first + 300.0) / 3600.0 <= 10.0 - reserve + 1e-9              # worst case fits beside the reserve
    # Spend until the next screen's worst case cannot fit any more: the search stops instead of overrunning.
    while not s.stop_flag:
        fb = s.propose_config(changes={'micro_batch_size': 2, 'bucket_size': 10 ** 7 * (len(r.caps) + 1)}, base='c0')
        if 'candidate_id' not in fb:
            break
        s.screen(fb['candidate_id'])
    # Every attempt was admitted with its own worst case beside the pair reserve, given what had actually been used.
    for (used, pair_reserve, expected), cap in zip(r.used_before, r.caps):
        assert used + 8 * (cap + 300.0) / 3600.0 + pair_reserve <= 10.0 + 1e-9
        assert cap >= expected                                                  # never shorter than an expected screen
    assert s.ledger.used['gpu_hours'] + s.pair_reserve['gpu_hours'] <= 10.0 + 1e-9 and s.stop_flag == 'budget:gpu_hours'


def test_no_attempt_when_even_one_screen_cannot_fit(tasks):
    clock = VirtualClock()
    r = BoundedFake(tasks['D2'], DENSE_8B, clock=clock)
    s = _session(tasks, r, clock, gpu_hours=4.0)        # reserve 2 x 8 x 1200 s = 5.3 GPUh > 4.0
    fb = s.propose_config(changes={'micro_batch_size': 2}, base='c0')
    assert 'budget exhausted' in s.screen(fb['candidate_id'])['error'] and r.caps == []


def test_llm_calls_reserve_their_output_and_get_the_allowance(tasks, a0_package):
    from infrarsi.agent.optimize import optimize
    seen = []

    class Drv(ScriptedBackend):
        def complete(self, system, prompt, **kw):
            seen.append(kw.get('max_output_tokens'))
            return super().complete(system, prompt, **kw)
    drv = Drv(lambda role, s, p: {'action': 'read_env'}, tokens_per_call=3000)
    t = tasks['D2']
    start = synthetic_receipt(t.raw, {}, step_time_s=5.0, peak_reserved_gib=60, run_id='r', attempt_id='ed')
    tr = optimize(t, a0_package, drv, Budget(screens=4, llm_tokens=40000, wall_seconds=36000, gpu_hours=100),
                  dict(FR_DENSE, tensor_model_parallel_size=2, sequence_parallel=True), FakeScreenRunner(t, DENSE_8B),
                  judge=ReceiptJudge(), model_facts=DENSE_8B, start_receipt=start).to_dict()
    assert tr['stop_reason'] == 'budget:llm_tokens' and tr['used']['llm_tokens'] <= 40000
    assert seen and all(m is not None and m <= 3000 for m in seen)


def test_openai_re_asks_stop_at_the_token_allowance():
    replies = [(200, {'model': 'm', 'usage': {'prompt_tokens': 900, 'completion_tokens': 100},
                      'choices': [{'message': {'content': '{"action": "bogus"}'}}]})] * 4
    calls = []

    def transport(url, headers, body, timeout):
        calls.append(body['max_tokens'])
        return replies.pop(0)
    b = OpenAICompatBackend(base_url='http://x/v1', model='m', api_key_env=None, price=None, transport=transport, max_tokens=500)
    r = b.complete('sys', 'x' * 3150, schema=action_schema('weak'), token_allowance=2500, max_output_tokens=400)
    assert len(calls) == 2 and r.llm_tokens <= 2500 and all(c <= 400 for c in calls)    # a third attempt would not fit
