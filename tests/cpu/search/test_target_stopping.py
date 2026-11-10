import dataclasses

import pytest

from infrarsi.search.bcomp import BComp
from infrarsi.search.engine import Proposal, Strategy
from infrarsi.search.proposers import SimProposer
from infrarsi.harness import signing
from test_search import TOGGLE, setup


GOOD = {'fused_adam': True, 'flash_attn': True, 'overlap_grad': True}


def search(**budget):
    sim, s, eng = setup(TOGGLE, tokens=100000, job_gpu_hours=None, early_abort=False)
    s.budget = s.ledger.budget = dataclasses.replace(s.budget, **budget)
    return sim, s, eng


class OneMove(Strategy):
    uses_engine_stop = False

    def propose(self, engine, k):
        return [Proposal(change=GOOD)]


def test_target_speedup_stops_after_a_valid_full_screen_with_a_milestone():
    _, s, eng = search(target_speedup=1.1)
    log = []
    eng.log = log.append
    trace = eng.run(OneMove())
    assert trace.stop_reason.startswith('target_speedup:') and '>=1.1' in trace.stop_reason
    assert s.ledger.used['screens'] == 1 and eng.incumbent() != 'c0'
    assert any('milestone:' in line for line in log)


@pytest.mark.parametrize('source', ['receipt', 'accounting', 'reference'])
def test_mfu_target_uses_measurement_then_accounting_then_reference_scaling(source):
    _, s, eng = search(target_mfu=45)
    s.candidates['c0'].receipt.setdefault('evidence', {})['mfu'] = .44
    s.candidates['c0'].receipt = s.reference = signing.sign(s.candidates['c0'].receipt)
    original = s.runner.run

    def run(*a, **kw):
        receipt = original(*a, **kw)
        if source == 'receipt':
            receipt['evidence']['mfu'] = .5
        return signing.sign(receipt)

    s.runner.run = run
    calls = []
    if source != 'reference':
        eng.accounting = lambda cfg, step: calls.append((cfg, step)) or {'mfu': .49}
    trace = eng.run(OneMove())
    assert trace.stop_reason.startswith('target_mfu:') and '>=0.45' in trace.stop_reason
    assert bool(calls) == (source == 'accounting')
    if source == 'receipt':
        assert trace.stop_reason == 'target_mfu:0.500>=0.45'


def test_short_or_invalid_runs_never_satisfy_targets():
    _, s, eng = search(target_speedup=1.01, target_mfu=.01)
    eng.accounting = lambda *a: {'mfu': 1.0}
    out = eng.evaluate(Proposal(change=GOOD, fidelity=('short',)))
    assert out.status == 'short' and not s.stop_flag
    _, s, eng = search(target_speedup=1.01, target_mfu=.01)
    original = s.runner.run

    def invalid(*a, **kw):
        receipt = original(*a, **kw)
        receipt['valid'] = False
        receipt['invalid_reasons'] = ['test failure']
        receipt['evidence']['exit_code'] = 1
        receipt['evidence']['mfu'] = .9
        return signing.sign(receipt)

    s.runner.run = invalid
    out = eng.evaluate(Proposal(change=GOOD))
    assert not out.valid and s.stop_flag is None


def test_session_strategies_also_stop_at_targets():
    _, s, _ = search(target_speedup=1.1)
    cid = s.propose_config(changes=GOOD)['candidate_id']
    row = s.screen(cid)
    assert row['valid'] and s.stop_flag.startswith('target_speedup:')
    assert 'search stopped' in s.screen('c0')['error']


def test_bcomp_drains_pending_proposals_after_tokens_are_spent(monkeypatch):
    sim, s, eng = search()
    prop = SimProposer(sim)
    st = BComp(prop)
    st._init(eng)
    st.pending = [Proposal(change=GOOD, hypothesis='queued')]
    monkeypatch.setattr(st, 'complete', lambda *a: None)
    monkeypatch.setattr(st, '_bc_contrarian', lambda *a: None)
    monkeypatch.setattr(st, '_bc_escape', lambda *a: None)
    monkeypatch.setattr(st, '_code_step', lambda *a, **kw: None)
    monkeypatch.setattr(prop, 'hypotheses', lambda *a, **kw: pytest.fail('no model call may be made'))
    monkeypatch.setattr(prop, 'knob_prior', lambda *a, **kw: pytest.fail('no new prior may be requested'))
    s.ledger.used['llm_tokens'] = s.budget.llm_tokens
    trace = eng.run(st)
    assert s.ledger.used['screens'] == 1 and trace.stop_reason == 'budget:llm_tokens'
    assert s.ledger.used['llm_tokens'] == s.budget.llm_tokens
    assert not s.ledger.exhausted() and s.ledger.exhausted(allow_token_free=False) == ['llm_tokens']


@pytest.mark.parametrize('move', ['_code_step', 'complete', '_bc_contrarian', '_bc_escape'])
def test_bcomp_uses_ready_code_and_memory_moves_without_new_model_calls(monkeypatch, move):
    sim, s, eng = search()
    prop = SimProposer(sim)
    strategy = BComp(prop)
    strategy._init(eng)
    strategy.pending = []
    ready = [Proposal(change=GOOD)]
    for name in ('_code_step', 'complete', '_bc_contrarian', '_bc_escape'):
        monkeypatch.setattr(strategy, name, lambda *a, **kw: None)

    def next_move(*args, **kw):
        if move == '_code_step':
            assert kw['ask'] is False
        return ready.pop() if ready else None

    monkeypatch.setattr(strategy, move, next_move)
    monkeypatch.setattr(prop, 'hypotheses', lambda *a, **kw: pytest.fail('model call after tokens exhausted'))
    monkeypatch.setattr(prop, 'knob_prior', lambda *a, **kw: pytest.fail('model call after tokens exhausted'))
    s.ledger.used['llm_tokens'] = s.budget.llm_tokens
    trace = eng.run(strategy)
    assert s.ledger.used['screens'] == 1 and trace.stop_reason == 'budget:llm_tokens'


@pytest.mark.parametrize('limit', [{'screens': 0}, {'wall_seconds': 0}, {'gpu_hours': 0}])
def test_zero_token_moves_still_obey_other_limits(limit):
    sim, s, eng = search(**limit)
    st = BComp(SimProposer(sim))
    st.pending = [Proposal(change=GOOD)]
    s.ledger.used['llm_tokens'] = s.budget.llm_tokens
    trace = eng.run(st)
    assert s.ledger.used['screens'] == 0 and trace.stop_reason == 'budget:' + next(iter(limit))

