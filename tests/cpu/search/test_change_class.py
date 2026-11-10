"""Change classes (textbook design P1-1, 2026-10-24): the proposer labels each hypothesis execution / numerics /
algorithmic, the engine reports the loss deviation of every full run, and the strategies' ledgers carry both - a
report-only diagnostic (the tolerance stays the one acceptance rule)."""
from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.ext.common.session_hooks import SessionHooks
from infrarsi.search.engine import Outcome, Proposal, loss_deviation
from infrarsi.search.proposers import LLMProposer, change_class
from infrarsi.search.strategies import HLS, Greedy

from test_search import TOGGLE, setup

HYPS = {'hypotheses': [
    {'mechanism': 'fused adam', 'change': {'fused_adam': True}, 'predicted_gain_pct': 4, 'change_class': 'Execution'},
    {'mechanism': 'fused ce', 'change': {'fused_ce': True}, 'predicted_gain_pct': 3, 'change_class': 'numerics'},
    {'mechanism': 'flash', 'change': {'flash_attn': True}, 'predicted_gain_pct': 6, 'change_class': 'bogus'}]}


def _prop(s):
    return LLMProposer(ScriptedBackend(lambda role, system, prompt: HYPS), SessionHooks(s.task, base_files={}))


def test_labels_are_normalised():
    assert [change_class(x) for x in ('Execution', ' numerics ', 'algorithmic', 'bogus', None)] == [
        'execution', 'numerics', 'algorithmic', None, None]
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    hyps = _prop(s).hypotheses(eng, 'c0')
    assert [h['change_class'] for h in hyps] == ['execution', 'numerics', None]


def test_a_full_run_reports_its_loss_deviation():
    sim, s, eng = setup(TOGGLE)
    out = eng.evaluate(Proposal(change={'fused_adam': True}))
    assert out.status == 'screened' and isinstance(out.loss_deviation, float) and out.loss_deviation >= 0
    assert out.loss_deviation == loss_deviation(s.candidates[out.cid].verdict)
    assert loss_deviation(None) is None and loss_deviation({'checks': [{'id': 'completes'}]}) is None


def test_hls_ledger_carries_class_and_deviation():
    sim, s, eng = setup(TOGGLE, budget=12, tokens=10 ** 6)
    hls = HLS(_prop(s), explore=False, bundle=False)
    eng.run(hls)
    rows = {r['mechanism']: r for r in hls.ledger_view()}
    assert rows['fused adam']['change_class'] == 'execution' and rows['fused ce']['change_class'] == 'numerics'
    assert 'change_class' not in rows['flash']
    assert any('loss_deviation_pct' in r for r in rows.values() if r['status'] != 'open')


def test_greedy_history_carries_class_and_deviation():
    sim, s, eng = setup(TOGGLE, budget=8, tokens=10 ** 6)
    gr = Greedy(_prop(s), finish_below=-1.0)
    eng.run(gr)
    assert gr.history and any(h.get('change_class') for h in gr.history)
    assert all('loss_deviation_pct' in h for h in gr.history if h['status'] in ('adopted', 'not adopted'))


def test_a_repeat_of_the_reference_gives_the_floor():
    from infrarsi.agent.judge import Verdict
    from infrarsi.project.flow import repeat_deviation
    ok = Verdict(valid=True, reasons=[], checks=[{'id': 'loss_within_tolerance', 'evidence': {'deviation': 0.001}}])
    assert repeat_deviation(ok) == 0.001
    assert repeat_deviation(Verdict(valid=False, reasons=['completes: no step time'])) is None and repeat_deviation(None) is None


def test_the_model_does_not_see_the_diagnostic():
    sim, s, eng = setup(TOGGLE, budget=12, tokens=10 ** 6)
    hls = HLS(_prop(s), explore=False, bundle=False)
    eng.run(hls)
    assert any('change_class' in r for r in hls.ledger_view())
    assert not any(k in r for r in hls.model_view() for k in ('change_class', 'loss_deviation_pct'))
    seen = []
    sim, s, eng = setup(TOGGLE, budget=8, tokens=10 ** 6)
    prop = _prop(s)
    hyps = prop.hypotheses

    def spy(engine, base, ledger=None):
        seen.append(list(ledger or []))
        return hyps(engine, base, ledger=ledger)
    prop.hypotheses = spy
    gr = Greedy(prop, finish_below=-1.0)
    eng.run(gr)
    assert any('change_class' in h for h in gr.history) and any(seen)
    assert not any(k in h for rows in seen for h in rows for k in ('change_class', 'loss_deviation_pct'))


def test_a_bundle_keeps_its_own_deviation():
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    hls = HLS(_prop(s))
    hls.ledger = {'a': {'mechanism': 'compile', 'change_class': 'execution', 'mu': 0.05, 'status': 'unresolved'},
                  'b': {'mechanism': 'fused ce', 'change_class': 'execution', 'mu': 0.03, 'status': 'unresolved'}}
    hls.members['bundle|1'] = ['a', 'b']
    out = Outcome(cid='c9', status='invalid', valid=False, step_time=1.0, gain=None, fidelity='full', gpu_hours=0.1,
                  reasons=['loss_within_tolerance: 6 %'], failed_checks=['loss_within_tolerance'], loss_deviation=0.06)
    hls._observe_bundle(eng, Proposal(change={}, hypothesis='bundle|1'), out)
    assert hls.describe()['bundle_numerics'] == [{'candidate': 'c9', 'members': ['compile', 'fused ce'], 'status': 'invalid',
                                                  'classes': ['execution', 'execution'], 'loss_deviation_pct': 6.0}]


def test_the_report_reads_each_deviation_against_the_repeat_floor():
    from infrarsi.project.report import _numerics
    r = {'tolerance': 0.03, 'search': {'doctor_repeat_loss_deviation': 0.0002, 'ledger': [
        {'mechanism': 'fused adam', 'change_class': 'execution', 'loss_deviation_pct': 0.01},
        {'mechanism': 'te cross entropy', 'change_class': 'execution', 'loss_deviation_pct': 1.9},
        {'mechanism': 'fp8', 'change_class': 'numerics', 'loss_deviation_pct': 0.8},
        {'mechanism': 'workers on bf16', 'change_class': 'execution', 'loss_deviation_pct': 0.2, 'loss_deviation_base_pct': 0.0},
        {'mechanism': 'flash', 'status': 'open', 'change_class': 'execution'}],
        'bundle_numerics': [{'members': ['compile', 'fused ce'], 'classes': ['execution', 'execution'],
                             'loss_deviation_pct': 6.0}]},
         'promotion': {'tried': [{'reference_repeat_loss_deviation': 0.0003}, {'reference_repeat_loss_deviation': None}]}}
    text = '\n'.join(_numerics(r))
    assert 'Repeat floor: 0.0300 %, the largest of 2 repeat(s) of one configuration' in text
    assert '| fused adam | execution | 0.0100 % | - | within |' in text
    assert '| te cross entropy | execution | 1.9000 % | - | above: declared execution, numerics changed |' in text
    assert '| fp8 | numerics | 0.8000 % | - | above |' in text and 'flash' not in text
    assert '| workers on bf16 | execution | 0.2000 % | 0.0000 % | within |' in text   # its own effect: from its base
    assert '| bundle: compile + fused ce | execution, execution | 6.0000 % | - | above: declared execution, numerics' in text
    tiny = {'tolerance': 0.03, 'search': {'doctor_repeat_loss_deviation': 6e-7, 'ledger': [
        {'mechanism': 'x', 'change_class': 'execution', 'loss_deviation_pct': 100 * 6e-7}]}}
    assert '| x | execution | 0.0001 % | - | within |' in '\n'.join(_numerics(tiny))  # equal deviations: compared unrounded
    r['search']['doctor_repeat_loss_deviation'], r['promotion'] = None, {}
    assert 'unknown (no re-run' in '\n'.join(_numerics(r)) and _numerics({'search': {}}) == []
