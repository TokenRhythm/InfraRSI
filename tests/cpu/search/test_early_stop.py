"""Item 1 after H2b: the search stopped early. Arm A (hls, Opus) ended after 6 of 24 runs because (1) the calibration was
applied twice - the model, shown its misses, discounted its next predictions and kappa shrank them again - and (2)
without `search.job_gpu_hours` the engine assumed a production job of 300 reference runs. Now HLS's model sees
measurements, not the predictions made for them (this run's ledger and earlier runs), and is asked for undiscounted
predictions; Greedy, which has no kappa, keeps its prompt; an undeclared job size turns the value/cost stop off."""
import json
import math

import pytest

from infrarsi import search
from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.ext.common.session_hooks import SessionHooks
from infrarsi.search.engine import Engine, Proposal, Strategy
from infrarsi.search.proposers import LLMProposer
from infrarsi.search.strategies import HLS

from test_search import TOGGLE, setup

EARLIER = [{'mechanism': 'grad overlap', 'change': {}, 'predicted_gain_pct': 6.77, 'measured_gain_pct': 0.42, 'status': 'confirmed'},
           {'mechanism': 'fp8', 'change': {}, 'predicted_gain_pct': 19.9, 'measured_gain_pct': None, 'status': 'invalid'}]


class Poor(Strategy):
    """The proposal that the declared-horizon rule stops on at once (test_search.test_stop_rule_after_floor)."""
    def propose(self, engine, k):
        return [Proposal(change={'nccl_tree': True}, mu=-0.05, sigma=0.001)]


def test_no_value_stop_without_a_declared_job_size():
    sim, s, eng = setup(TOGGLE, floor_screens=0, job_gpu_hours=None)
    tr = eng.run(Poor()).to_dict()
    assert tr['stop_reason'] == 'strategy_done' and tr['extra']['stop_numbers'] is None
    assert tr['extra']['stop_rule'] == 'off' and [d['stop_rule'] for d in tr['extra']['decisions']] == ['off']
    assert s.ledger.used['screens'] == 1                    # evaluated; the declared rule would not have run it
    sim, s, eng = setup(TOGGLE, floor_screens=0)
    tr = eng.run(Poor()).to_dict()
    assert tr['stop_reason'] == 'stop_rule' and tr['extra']['stop_rule'] == 'on' and s.ledger.used['screens'] == 0


def test_hls_model_sees_measurements_not_the_predictions_made_for_them():
    sim, s, eng = setup(TOGGLE, budget=4, tokens=10 ** 6)
    reply = {'hypotheses': [{'mechanism': 'fused trio', 'change': {'fused_adam': True, 'flash_attn': True, 'overlap_grad': True},
                             'predicted_gain_pct': 30, 'p_feasible': 0.95}]}
    drv = ScriptedBackend(lambda role, system, prompt: reply)
    hls = HLS(LLMProposer(drv, SessionHooks(s.task, base_files={'patches.txt': ''}), prior=EARLIER, search_calibrates=True))
    eng.run(hls)
    calls = [c for c in drv.calls if c['role'] == 'hypotheses']     # the code review is the first call
    assert eng.incumbent() != 'c0' and len(calls) >= 2          # the move made the model look again
    seen = [e for c in calls[1:] for e in json.loads(c['prompt'])['ledger']]
    measured = [e for e in seen if e.get('measured_gain_pct') is not None]
    assert measured and all('predicted_gain_pct' not in e for e in measured)
    kept = [e for e in hls.ledger_view() if e['mechanism'] == 'fused trio']    # the trace keeps the prediction
    assert kept[0]['measured_gain_pct'] is not None and kept[0]['predicted_gain_pct'] == round(100 * math.log1p(0.3), 2)
    earlier = {e['mechanism']: e for e in json.loads(calls[0]['prompt'])['earlier_runs']}
    assert 'predicted_gain_pct' not in earlier['grad overlap'] and earlier['grad overlap']['measured_gain_pct'] == 0.42
    assert earlier['fp8']['predicted_gain_pct'] == 19.9          # nothing measured to set it against
    system = calls[0]['system']
    assert 'do not\ndiscount predictions' in system and 'Be calibrated' not in system


def test_greedy_keeps_its_prompt_and_earlier_pairs():
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    drv = ScriptedBackend(lambda role, system, prompt: {'hypotheses': []})
    LLMProposer(drv, SessionHooks(s.task, base_files={'patches.txt': ''}), prior=EARLIER).hypotheses(eng, 'c0')
    calls = [c for c in drv.calls if c['role'] == 'hypotheses']     # the code review is the first call
    assert 'Be calibrated: the ledger shows' in calls[0]['system'] and 'do not\ndiscount' not in calls[0]['system']
    assert json.loads(calls[0]['prompt'])['earlier_runs'] == EARLIER


@pytest.mark.parametrize('name, calibrates', [('greedy', False), ('hls', True), ('doe+hls', True)])
def test_only_kappa_strategies_ask_for_undiscounted_predictions(monkeypatch, name, calibrates):
    seen = {}

    class Stop(Exception):
        pass

    def fake_run(self, strategy):
        hls = strategy.strategies[-1] if hasattr(strategy, 'strategies') else strategy
        seen['calibrates'] = hls.proposer.search_calibrates
        raise Stop

    sim, s, _ = setup(TOGGLE)
    monkeypatch.setattr(Engine, 'run', fake_run)
    ctx = search.Context(task=s.task, budget=s.budget, start_config={}, runner=s.runner, judge=s.judge, hooks=s.hooks,
                         run_id='t', out_dir=None, start_receipt=s.reference, reference_receipt=s.reference,
                         driver=ScriptedBackend(lambda role, system, prompt: {'hypotheses': []}),
                         options={'capability': 0.9, 'full_steps': 30, 'job_gpu_hours': None})
    with pytest.raises(Stop):
        search.run(name, ctx)
    assert seen['calibrates'] is calibrates


def test_model_view_keeps_what_is_not_the_models_prediction():
    hls = HLS(None)
    base = {'change': {}, 'base': 'c0', 'failed_checks': None}
    hls.ledger = {'c0|a': dict(base, mechanism='a', mu=0.05, status='confirmed', measured=0.01),
                  'c0|b': dict(base, mechanism='b', mu=0.02, status='open'),
                  'c0|c': dict(base, mechanism='c', mu=0.10, status='invalid'),
                  'c0|x': dict(base, mechanism='experience:x', mu=0.04, status='refuted', measured=-0.01, source='experience')}
    view = {e['mechanism']: e for e in hls.model_view()}
    assert 'predicted_gain_pct' not in view['a'] and view['a']['measured_gain_pct'] == 1.0
    assert view['b']['predicted_gain_pct'] == 2.0 and view['c']['predicted_gain_pct'] == 10.0   # nothing measured to compare
    assert view['experience:x']['predicted_gain_pct'] == 4.0                                     # data, not a prediction
    assert all('predicted_gain_pct' in e for e in hls.ledger_view())
