"""Regression tests for the Codex review round 4 findings (guard aliasing, endpoint loss on every LLM path, terminal
transport errors, fixed self-host request fields, conservative locked-pair GPU fallback, red-team caps, OptiCo
token admission)."""
import json

import pytest

from infrarsi.actions import patch_guard
from infrarsi.agent import arms
from infrarsi.agent.backends.base import LLMResult
from infrarsi.agent.backends.openai_compat import OpenAICompatBackend
from infrarsi.agent.budget import Budget, VirtualClock
from infrarsi.agent.judge import ReceiptJudge, Verdict
from infrarsi.agent.runner import FakeScreenRunner, synthetic_receipt
from infrarsi.agent.tools import action_schema
from infrarsi.contracts import TaskSpec
from agent_helpers import DENSE_8B, FR_DENSE

ALIASED_HELPER = 'import torch\n\n\ndef helper():\n    sync = torch.cuda.synchronize\n    sync()\n\n\nh = helper\n\n\ndef after_update(ctx, step):\n    h()\n'


@pytest.mark.parametrize('src', [
    ALIASED_HELPER,
    'import torch\nimport functools\n\n\ndef helper():\n    torch.cuda.synchronize()\n\n\nH = functools.partial(helper)\n\n\ndef after_update(ctx, step):\n    H()\n',
    'import torch\n\n\nTABLE = {"s": torch.cuda.synchronize}\n\n\ndef before_update(ctx, step):\n    TABLE["s"]()\n',
    'import torch.distributed as dist\n\n\ndef on_train_start(ctx):\n    return None\n',
])
def test_host_sync_module_may_not_reference_sync_or_collectives(pristine_overlays, src):
    r = patch_guard.apply({'host_sync.py': src})
    assert not r.accepted and 'G19' in r.codes(), r.findings


def test_aliased_hooks_are_checked_too(pristine_overlays):
    src = 'def _fb(ctx, run, **kwargs):\n    return []\n\n\nforward_backward = _fb\n'
    r = patch_guard.apply({'schedule.py': src})
    assert not r.accepted and 'G19' in r.codes()
    r = patch_guard.apply({'schedule.py': 'forward_backward = lambda ctx, run, **kw: []\n'})
    assert not r.accepted


class _Script:
    """Backend stand-in: replies by role from a list of LLMResults."""

    def __init__(self, replies):
        self.replies, self.roles = list(replies), []

    def describe(self):
        return {'kind': 'test'}

    def complete(self, system, prompt, *, role='proposer', **kw):
        self.roles.append(role)
        return self.replies.pop(0)


def _res(action=None, error=None, unreachable=False):
    return LLMResult(action=action, text=json.dumps(action) if action else None, llm_tokens=10, llm_usd=0.0, wall_seconds=0.1,
                     accounting_complete=True, error=error, model='m', observed_models=['m'], attempts=1,
                     meta={'unreachable': unreachable})


def _search(tasks, a0_package, drv, **kw):
    from infrarsi.agent.optimize import optimize
    t = tasks['D2']
    start = synthetic_receipt(t.raw, {}, step_time_s=5.0, peak_reserved_gib=60, run_id='r', attempt_id='ed')
    return optimize(t, a0_package, drv, Budget(screens=8, llm_tokens=10 ** 6, wall_seconds=36000, gpu_hours=100),
                    dict(FR_DENSE, tensor_model_parallel_size=2, sequence_parallel=True), FakeScreenRunner(t, DENSE_8B),
                    judge=ReceiptJudge(), model_facts=DENSE_8B, start_receipt=start, **kw).to_dict()


def test_endpoint_lost_during_a_retry_stops_the_search(tasks, a0_package):
    drv = _Script([_res(error='schema: bad'), _res(error='endpoint unreachable: refused', unreachable=True),
                   _res(action={'action': 'finish'})])
    tr = _search(tasks, a0_package, drv, mode='weak')
    assert tr['stop_reason'] == 'endpoint_unreachable' and len(drv.roles) == 2


def test_transport_errors_are_not_re_asked(tasks, a0_package):
    drv = _Script([_res(error='transport error: TimeoutError: read timed out')] * 6 + [_res(action={'action': 'finish'})])
    tr = _search(tasks, a0_package, drv, mode='weak')
    assert drv.roles.count('proposer') == len(drv.roles)            # no proposer_retry re-asks for transport errors


def test_endpoint_lost_in_the_summarizer_stops_before_the_next_proposal(tasks, a0_package, monkeypatch):
    import importlib
    opt = importlib.import_module('infrarsi.agent.optimize')
    from infrarsi.agent import policy as policy_mod
    real = policy_mod.load

    def load(text, mode, **kw):
        p = real(text, mode, **kw)
        p.summarizer, p.max_context_tokens, p.verifier = 'llm', 4000, False
        return p
    monkeypatch.setattr(opt.policy_mod, 'load', load)
    drv = _Script([])

    def complete(system, prompt, *, role='proposer', **kw):
        drv.roles.append(role)
        if role == 'summarizer':
            return _res(error='endpoint unreachable: refused', unreachable=True)
        return _res(action={'action': 'read_env'}) if len(drv.roles) < 40 else _res(action={'action': 'finish'})
    drv.complete = complete
    tr = _search(tasks, a0_package, drv)
    assert 'summarizer' in drv.roles and drv.roles[-1] == 'summarizer'       # nothing is sent after the loss
    assert tr['stop_reason'] == 'endpoint_unreachable'


def test_read_timeout_is_charged_once_and_never_resent():
    calls = []

    def transport(url, headers, body, timeout):
        calls.append(1)
        raise TimeoutError('read timed out')
    b = OpenAICompatBackend(base_url='http://127.0.0.1:18000/v1', model='m', api_key_env=None, price=None, transport=transport)
    r = b.complete('sys', 'prompt', schema=action_schema('weak'))
    assert len(calls) == 1 and r.error.startswith('transport error') and r.accounting_complete is False
    assert r.meta['transport_error'] and not r.meta['unreachable']


def test_status_json_cannot_change_fixed_request_fields(tmp_path, monkeypatch):
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    (tmp_path / 'STATUS.json').write_text(json.dumps({'extra_request_params': {
        'chat_template_kwargs': {'thinking': True, 'other': 1}, 'temperature': 1.0, 'model': 'x', 'max_tokens': 99999,
        'stream': True}}))
    monkeypatch.setattr(arms, 'WEAK_STATUS', tmp_path / 'STATUS.json')
    _, b = arms.weak_driver()
    seen = []
    b.transport = lambda url, h, body, t: (seen.append(body), (200, {'model': b.model, 'usage': {'prompt_tokens': 1, 'completion_tokens': 1},
                                                                    'choices': [{'message': {'content': '{"action": "finish"}'}}]}))[1]
    b.complete('s', 'p', schema=action_schema('weak'))
    body = seen[0]
    assert body['chat_template_kwargs'] == {'thinking': False, 'other': 1} and body['temperature'] == 0.0
    assert body['model'] == b.model and body['stream'] is False and body['max_tokens'] <= 4096


def test_locked_pair_without_gpu_timing_is_charged_conservatively(tmp_path):
    class PR:
        score_reference, reference = None, None

        def run(self, task, config, ov, purpose, run_id, role='candidate', recheck=False):
            if config.get('who') == 'FR':
                return {'valid': True, 'world_size': 8, 'cost': {'wall_seconds': 60},
                        'step_time': {'mean_window_max_rank_s': 1.0, 'rsd_window': 0.01},
                        'harness': {'numerics': {'compared': False, 'is_reference': True}}}
            return {'valid': True, 'world_size': 8, 'cost': {'gpu_seconds': 72, 'wall_seconds': 9},
                    'step_time': {'mean_window_max_rank_s': 0.9}, 'harness': {'numerics': {'compared': True, 'class': 'E0'}}}
    refs = {'fr_status': 'feasible', 'fr_screen': '/data/infrarsi/receipts/fr.json', 'fr_config': {'who': 'FR'}}
    pair = arms.locked_pair(TaskSpec.load('D2'), {'run_id': 'r'}, PR(), refs, {'who': 'c'}, None,
                            selected='c1', out_dir=tmp_path, judge=lambda c, p, r: Verdict(valid=True, reasons=[]))
    assert pair['accounting_complete'] is False and pair['gpu_hours'] >= (8 * 60 + 72) / 3600 - 1e-9


def test_red_team_caps_parse_zero_and_bound_gpu():
    from infrarsi.redteam.arms import RunCapped
    clock = VirtualClock()

    class HR:
        wall_seconds_cap = 3600

        def run(self, *a):
            clock.advance(3600.0)
            return {'valid': False, 'harness': {'gpu_hours': 2.0}}
    t = TaskSpec.load('S2')
    rc = RunCapped(HR(), {'wall_hours_cap': 0, 'gpu_hours_budget': 0}, t, clock=clock)
    assert rc.run(t, {}, None, 'screen', 'r').get('not_run')
    rc = RunCapped(HR(), {'wall_hours_cap': 10, 'gpu_hours_budget': 0.01}, t, clock=clock)
    assert rc.run(t, {}, None, 'screen', 'r').get('not_run')                  # 2 x 0.01 GPUh cannot fit a 600 s screen


def test_optico_respects_a_zero_token_budget(tasks, monkeypatch):
    from infrarsi.agent.backends import ScriptedBackend
    from infrarsi.baselines.optico_style import run_optico_style
    t = tasks['D2']
    start = synthetic_receipt(t.raw, {}, step_time_s=5.0, peak_reserved_gib=60, run_id='r', attempt_id='ed')
    drv = ScriptedBackend(['x'])
    tr = run_optico_style(t, Budget(screens=4, llm_tokens=0, wall_seconds=3600, gpu_hours=10),
                          dict(FR_DENSE, tensor_model_parallel_size=2, sequence_parallel=True), FakeScreenRunner(t, DENSE_8B),
                          llm_backend=drv, judge=ReceiptJudge(), model_facts=DENSE_8B, start_receipt=start).to_dict()
    assert drv.calls == [] and tr['used']['llm_tokens'] == 0 and tr['stop_reason'] == 'budget:llm_tokens'
