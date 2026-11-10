"""Fail-closed accounting, phase isolation and admission (Codex review round 2: H3, H4, M1-M4), and the M-arm design."""
import json
import math

import pytest

from infrarsi.agent.backends import ScriptedBackend
from infrarsi.agent.backends.openai_compat import OpenAICompatBackend
from infrarsi.agent.budget import Budget, SearchLedger, VirtualClock
from infrarsi.agent.judge import ReceiptJudge, gpu_seconds, wall_seconds
from infrarsi.agent.runner import FakeScreenRunner, synthetic_receipt
from infrarsi.agent.search import SearchSession
from infrarsi.agent.tools import action_schema
from infrarsi.contracts import TaskSpec
from infrarsi.rsi import OptimizerPackage, copy_package
from infrarsi.rsi.lineage import Lineage
from agent_helpers import DENSE_8B, FR_DENSE

WEAK = action_schema('weak')
OK = json.dumps({'action': 'finish'})


def _transport(replies, seen=None):
    def t(url, headers, body, timeout):
        if seen is not None:
            seen.append(timeout)
        item = replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    return t


def _backend(replies, seen=None, **kw):
    return OpenAICompatBackend(base_url='https://api.deepseek.com', model='deepseek-flash', api_key='k',
                               transport=_transport(replies, seen), max_tokens=1000, **kw)


def _ok(usage):
    return 200, {'model': 'deepseek-flash', 'choices': [{'message': {'content': OK}}], 'usage': usage}


# ---------------------------------------------------------------- H3: unknown / negative usage never reads as complete
def test_http_error_without_usage_charges_an_upper_bound():
    r = _backend([(500, {'error': {'message': 'boom'}}), _ok({'prompt_tokens': 10, 'completion_tokens': 5})]).complete(
        'sys', 'prompt', schema=WEAK)
    assert r.action == {'action': 'finish'} and r.accounting_complete is False
    assert r.llm_tokens >= 1000 + 15 and r.llm_usd > (10 * 0.30 + 5 * 1.20) / 1e6


@pytest.mark.parametrize('usage', [{'prompt_tokens': -20, 'completion_tokens': 0}, {'prompt_tokens': 1.5, 'completion_tokens': 1},
                                   {'prompt_tokens': 10, 'completion_tokens': 1, 'prompt_cache_hit_tokens': 50}, None])
def test_invalid_usage_is_unknown_not_negative(usage):
    r = _backend([_ok(usage)]).complete('sys', 'prompt', schema=WEAK)
    assert r.accounting_complete is False and r.llm_tokens > 0 and r.llm_usd > 0


def test_transport_failure_is_charged_and_the_timeout_is_honoured():
    seen = []
    r = _backend([TimeoutError('read timed out'), _ok({'prompt_tokens': 10, 'completion_tokens': 5})], seen,
                 timeout_seconds=300).complete('sys', 'prompt', schema=WEAK, timeout=0.25)
    assert r.accounting_complete is False and r.llm_tokens >= 1000 and seen and max(seen) <= 1.0


def test_cost_fields_must_be_finite_and_non_negative():
    assert gpu_seconds({'harness': {'gpu_hours': -3}}) is None and gpu_seconds({'cost': {'gpu_seconds': math.inf}}) is None
    assert gpu_seconds({'harness': {'gpu_hours': 0.5}, 'cost': {'gpu_seconds': 1.0}}) == 1800.0   # driver-measured first
    assert wall_seconds({'harness': {'wall_s': -1}}) is None and gpu_seconds(None) is None
    led = SearchLedger(Budget(screens=4, llm_tokens=100, wall_seconds=10, gpu_hours=1), VirtualClock())
    led.charge_llm(-20, -1.0, True)
    led.charge_screen(-3.0, 5.0, True)
    assert led.used['llm_tokens'] == 0 and led.used['gpu_hours'] == 0.0 and not led.accounting_complete


@pytest.mark.parametrize('bad', [None, 'not a receipt', 'raise'])
def test_screen_is_charged_on_every_path(tasks, bad):
    clock = VirtualClock()

    class Runner:
        def run(self, *a):
            clock.advance(100.0)
            if bad == 'raise':
                raise RuntimeError('lost the node')
            return bad
    s = SearchSession(task=tasks['D2'], budget=Budget(screens=4, llm_tokens=1000, wall_seconds=36000, gpu_hours=100),
                      runner=Runner(), judge=ReceiptJudge(), run_id='r', arm='A0', start_config=dict(FR_DENSE, tensor_model_parallel_size=2,
                                                                                                  sequence_parallel=True),
                      model_facts=DENSE_8B, clock=clock)
    fb = s.screen('c0')
    assert fb['valid'] is False and s.ledger.used['screens'] == 1
    assert abs(s.ledger.used['gpu_hours'] - 8 * 100.0 / 3600) < 1e-9 and not s.ledger.accounting_complete


# ---------------------------------------------------------------- M2: scoring reserve from the first screen
def test_first_screen_keeps_the_scoring_reserve(tasks):
    clock = VirtualClock()
    t = tasks['D2']
    start = synthetic_receipt(t.raw, {}, step_time_s=5.0, peak_reserved_gib=60, run_id='r', attempt_id='ed')
    start['cost'] = {'gpu_seconds': 0.6 * 3600, 'wall_seconds': 600}
    s = SearchSession(task=t, budget=Budget(screens=8, llm_tokens=1000, wall_seconds=36000, gpu_hours=1.0),
                      runner=FakeScreenRunner(t, DENSE_8B, clock=clock), judge=ReceiptJudge(), run_id='r', arm='A0',
                      start_config=dict(FR_DENSE, tensor_model_parallel_size=2, sequence_parallel=True), start_receipt=start,
                      model_facts=DENSE_8B, clock=clock)
    assert s.ledger.prior == (0.6, 600)
    assert s.ledger.admit_screen() == 'gpu_hours'                 # 0.6 + 2 x 0.6 reserve > 1.0 before any screen
    s2 = SearchSession(task=t, budget=Budget(screens=8, llm_tokens=1000, wall_seconds=36000, gpu_hours=1.0),
                       runner=FakeScreenRunner(t, DENSE_8B, clock=clock), judge=ReceiptJudge(), run_id='r', arm='A0',
                       start_config=dict(FR_DENSE, tensor_model_parallel_size=2, sequence_parallel=True), model_facts=DENSE_8B,
                       clock=clock)
    plan = t.raw['gpu_hours_plan']['search_incl_scoring']
    assert s2.ledger.prior[0] == pytest.approx(plan / (8 + 2))         # task plan when no start receipt


# ---------------------------------------------------------------- H4 / M3: phase isolation fails closed
def test_heldout_task_requires_a_heldout_session(tasks, a0_package):
    from infrarsi.agent.optimize import optimize
    drv = ScriptedBackend([{'action': 'finish'}])
    with pytest.raises(PermissionError):
        optimize(tasks['H3'], a0_package, drv, Budget(screens=2, llm_tokens=10 ** 6, wall_seconds=3600, gpu_hours=100),
                 dict(FR_DENSE, tensor_model_parallel_size=2, sequence_parallel=True), FakeScreenRunner(tasks['H3'], DENSE_8B),
                 judge=ReceiptJudge(), model_facts=DENSE_8B)


def test_reflect_and_history_are_development_only(tmp_path, a0_package):
    from infrarsi.rsi.loop import history_receipts
    from infrarsi.rsi.reflect import reflect
    trace = {'task_id': 'H3', 'arm': 'R', 'candidates': []}
    with pytest.raises(PermissionError):
        reflect('H3', trace, a0_package, ScriptedBackend(['x']), tmp_path / 'r', label='A_1')
    d = tmp_path / 'heldout_search'
    (d / 'receipts').mkdir(parents=True)
    (d / 'trace.json').write_text(json.dumps({'task_id': 'H1', 'candidates': []}))
    with pytest.raises(PermissionError):
        history_receipts([d])
    d2 = tmp_path / 'dev_search'
    (d2 / 'receipts').mkdir(parents=True)
    (d2 / 'trace.json').write_text(json.dumps({'task_id': 'D2', 'candidates': [{'id': 'c0', 'parent': None}]}))
    (d2 / 'receipts' / 'c0.json').write_text(json.dumps({'task_id': 'H5', 'attempt_id': 'x'}))
    with pytest.raises(PermissionError):
        history_receipts([d2])


def test_audit_finds_combinations_split_over_lines(tmp_path, a0_package):
    from infrarsi.rsi.audit import audit_package
    copy_package(a0_package.root, tmp_path / 'p')
    note = tmp_path / 'p' / 'skills' / 'recompute-policy' / 'notes.md'
    note.write_text('Lesson table:\n\nQwen3-8B\nD-PT-8K\n16 GPUs\nworked with TP2.\n', encoding='utf-8')
    rep = audit_package(tmp_path / 'p')
    assert not rep.passed and any(f['kind'] == 'combination' for f in rep.findings)


# ---------------------------------------------------------------- M1: every model call is admitted and bounded
def test_no_model_call_with_too_little_wall_left(tasks, a0_package):
    from infrarsi.agent.optimize import MIN_CALL_WALL_S, optimize
    clock = VirtualClock()
    drv = ScriptedBackend(lambda role, s, p: {'action': 'read_env'}, clock=clock, seconds_per_call=50.0)
    budget = Budget(screens=4, llm_tokens=10 ** 7, wall_seconds=MIN_CALL_WALL_S + 120, gpu_hours=100)
    start = synthetic_receipt(tasks['D2'].raw, {}, step_time_s=5.0, peak_reserved_gib=60, run_id='r', attempt_id='ed')
    tr = optimize(tasks['D2'], a0_package, drv, budget, dict(FR_DENSE, tensor_model_parallel_size=2, sequence_parallel=True),
                  FakeScreenRunner(tasks['D2'], DENSE_8B, clock=clock), judge=ReceiptJudge(), model_facts=DENSE_8B,
                  start_receipt=start, clock=clock).to_dict()
    assert tr['stop_reason'] == 'budget:wall_seconds' and len(drv.calls) == 3
    assert tr['used']['wall_seconds'] <= budget.wall_seconds + 50.0


# ---------------------------------------------------------------- M4: a receipt without any independent clock
def test_missing_independent_clock_is_a_finding():
    from infrarsi.redteam import detect, synthetic
    ctx = synthetic.context('S2')
    c = json.loads(json.dumps(ctx['clean']))
    c.pop('wallclock', None)
    c.pop('cost', None)
    c.pop('harness', None)
    kinds = {f['detector'] for f in detect(c, ctx['reference'])}
    assert 'timer_wallclock' in kinds


# ---------------------------------------------------------------- E6 M arm: design A2_M from A1 + dev experience
def test_m_design_freezes_a2m(tmp_path, a0_package):
    from infrarsi.rsi.design_m import design_m
    lin = Lineage(tmp_path / 'lin')
    lin.register(a0_package, status='initial')
    copy_package(a0_package.root, tmp_path / 'A1src')
    a1 = lin.snapshot('A1', OptimizerPackage.load(tmp_path / 'A1src'))
    d = tmp_path / 'E6-R-D2-search'
    (d / 'receipts').mkdir(parents=True)
    trace = {'task_id': 'D2', 'run_id': 'E6-R-D2-search', 'arm': 'R', 'mode': 'strong', 'stop_reason': 'finish', 'start': 'c0',
             'start_step_time': 5.0, 'selected': 'c0', 'selected_step_time': 5.0, 'selected_config': {}, 'selected_overlay': {},
             'candidates': [{'id': 'c0', 'parent': None}], 'screens': [], 'used': {}, 'budget': {}, 'invalid_proposals': {},
             'verifier': {}, 'skill_hits': [], 'tool_calls': [], 'memory_writes': []}
    (d / 'trace.json').write_text(json.dumps(trace))
    t = TaskSpec.load('D2').raw
    (d / 'receipts' / 'c0.json').write_text(json.dumps(synthetic_receipt(t, {}, step_time_s=5.0, peak_reserved_gib=60,
                                                                           run_id='r', attempt_id='c0')))
    skill = (a1.skills_dir / 'recompute-policy' / 'SKILL.md').read_text(encoding='utf-8')
    edit = {'skills': {'recompute-policy': skill + '\n- designed from the development pack\n'}, 'rationale': 'M design'}
    drv = ScriptedBackend([edit])
    res = design_m([d], a1, a0_package, drv, tmp_path / 'design', lineage=lin)
    assert res['status'] == 'changed' and lin.frozen('A2_M')['digest'] == res['digest'] and lin.verify_frozen('A2_M')
    assert 'DESIGN MODE' in drv.calls[0]['system'] and 'experience_pack' in drv.calls[0]['prompt']
    bad = design_m([d], a1, a0_package, ScriptedBackend(['not json']), tmp_path / 'design2', lineage=lin, label='A2_M_retry')
    assert bad['status'] == 'invalid_fallback' and bad['digest'] == a1.digest


def test_rawmem_package_is_a0_plus_raw_dev_logs(tmp_path, monkeypatch, a0_package):
    from types import SimpleNamespace
    from infrarsi.agent import arms
    from infrarsi.rsi.arms import RAW_SKILL, rawmem_package
    from infrarsi.rsi.package import copy_package as cp
    monkeypatch.setattr(arms, 'PACKAGES_DIR', tmp_path / 'packages')
    (tmp_path / 'packages').mkdir()
    cp(a0_package.root, tmp_path / 'packages' / 'A0')
    root = tmp_path / 'searches'
    for tid, st in (('S1', 1.0), ('D2', 5.0)):
        d = root / f'E6-R-{tid}-search'
        d.mkdir(parents=True)
        (d / 'trace.json').write_text(json.dumps({
            'task_id': tid, 'run_id': d.name, 'arm': 'R', 'mode': 'strong', 'stop_reason': 'finish', 'start': 'c0',
            'start_step_time': st, 'selected': 'c1', 'selected_step_time': st * 0.9, 'selected_config': {}, 'selected_overlay': {},
            'candidates': [{'id': 'c1', 'parent': 'c0', 'kind': 'L0', 'change': {'micro_batch_size': [1, 2]}, 'valid': True,
                            'step_time': st * 0.9}],
            'screens': [], 'used': {}, 'budget': {}, 'invalid_proposals': {}, 'verifier': {}, 'skill_hits': [], 'tool_calls': [],
            'memory_writes': []}))
    pkg = rawmem_package(SimpleNamespace(searches_root=root))
    assert RAW_SKILL in pkg.skill_names() and len(pkg.skill_names()) == len(a0_package.skill_names()) + 1
    text = (pkg.skills_dir / RAW_SKILL / 'SKILL.md').read_text(encoding='utf-8')
    assert 'raw development log S1' in text and 'raw development log D2' in text and len(text.encode()) <= 60_000
    again = rawmem_package(SimpleNamespace(searches_root=root))
    assert again.digest == pkg.digest


def test_m_design_handler_freezes_a2m_in_the_lineage(tmp_path, monkeypatch, a0_package):
    from infrarsi.agent import arms
    from infrarsi.dispatch.handlers import Context
    from infrarsi.dispatch import QueueStore
    from infrarsi.rsi.arms import m_design
    monkeypatch.setattr(arms, 'PACKAGES_DIR', tmp_path / 'packages')
    monkeypatch.setattr(arms, 'LINEAGE_DIR', tmp_path / 'packages' / 'lineage')
    monkeypatch.setattr(arms, 'COST_LEDGER', tmp_path / 'ledger.jsonl')
    copy_package(a0_package.root, tmp_path / 'packages' / 'A0')
    lin = arms.lineage()
    lin.snapshot('A1', arms.a0_package())
    d = tmp_path / 'searches' / 'E6-R-S1-search'
    (d / 'receipts').mkdir(parents=True)
    (d / 'trace.json').write_text(json.dumps({
        'task_id': 'S1', 'run_id': d.name, 'arm': 'R', 'mode': 'strong', 'stop_reason': 'finish', 'start': 'c0',
        'start_step_time': 1.0, 'selected': 'c0', 'selected_step_time': 1.0, 'selected_config': {}, 'selected_overlay': {},
        'candidates': [], 'screens': [], 'used': {}, 'budget': {}, 'invalid_proposals': {}, 'verifier': {}, 'skill_hits': [],
        'tool_calls': [], 'memory_writes': []}))
    monkeypatch.setitem(arms.DRIVERS, 'strong', lambda: ScriptedBackend(
        [{'memory': [{'tags': {'model': ['qwen3-dense']}, 'lesson': 'designed'}], 'rationale': 'M'}]))
    ctx = Context(store=QueueStore(tmp_path / 'q'), searches_root=tmp_path / 'searches', log=lambda m: None)
    res = m_design({'run_id': 'E6-M-DESIGN-A2M', 'task': 'none', 'arm': 'M_arm_design', 'priority': 'P1'}, [], ctx)
    assert res['status'] == 'changed' and arms.frozen_package('A2_M').digest == res['digest']
    assert (tmp_path / 'searches' / 'E6-M-DESIGN-A2M' / 'events.jsonl').exists()


# ---------------------------------------------------------------- Codex review round 3: accounting and caps
def test_usd_none_and_inconsistent_cache_counts_are_unknown():
    led = SearchLedger(Budget(screens=4, llm_tokens=100, wall_seconds=10, gpu_hours=1), VirtualClock())
    led.charge_llm(10, None, True)
    assert led.used['llm_tokens'] == 10 and not led.accounting_complete
    ok = OpenAICompatBackend._usage_ok
    assert not ok({'prompt_tokens': 100, 'completion_tokens': 1, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 1})
    assert not ok({'prompt_tokens': 100, 'completion_tokens': 1, 'prompt_cache_miss_tokens': 101})
    assert ok({'prompt_tokens': 100, 'completion_tokens': 1, 'prompt_cache_hit_tokens': 40, 'prompt_cache_miss_tokens': 60})
    b = _backend([_ok({'prompt_tokens': 100, 'completion_tokens': 10, 'prompt_cache_hit_tokens': 40})])
    r = b.complete('sys', 'prompt', schema=WEAK)
    assert r.accounting_complete and abs(r.llm_usd - (60 * 0.30 + 40 * 0.30 + 10 * 1.20) / 1e6) < 1e-12


def test_zero_token_budget_forbids_llm_calls_not_screens():
    led = SearchLedger(Budget(screens=4, llm_tokens=0, wall_seconds=100, gpu_hours=1), VirtualClock())
    assert led.exhausted() == [] and led.admit_screen() is None and led.fractions()['llm_tokens'] == 0.0
    led.charge_llm(5, 0.0, True)
    assert 'llm_tokens' in led.exhausted()


def test_redteam_loop_respects_the_run_caps():
    from infrarsi.redteam.arms import RunCapped
    from infrarsi.redteam.run_redteam import gpu_run
    clock = VirtualClock()

    class HR:
        wall_seconds_cap, attempts = 3600, []

        def run(self, task, config, ov, purpose, run_id):
            clock.advance(1500.0)
            return {'valid': False, 'attempt_id': 'a', 'invalid_reasons': ['timeout'], 'harness': {'gpu_hours': 0.8}}
    task = TaskSpec.load('S2')
    capped = RunCapped(HR(), {'wall_hours_cap': 1, 'gpu_hours_budget': 5.0}, task, clock=clock)
    rep = gpu_run(capped, task, {}, None, run_id='E4-S2-SEEDS-14x2', judge=ReceiptJudge())
    used = capped.used()
    assert used['screens'] == 2 and used['wall_seconds'] <= 3600 and rep['summary']['not_run']
    assert rep['summary']['G2_pass'] is False                                 # unscreened seeds can never pass G2
