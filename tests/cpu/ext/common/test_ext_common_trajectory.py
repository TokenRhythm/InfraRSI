import math

import pytest

from infrarsi.ext.common import trajectory as tj
from infrarsi.scoring import score
from x1_common_helpers import toy_task


def _trace(rows, used):
    """A SearchTrace-shaped dict: screens with cumulative snapshots."""
    screens = []
    for i, (cid, valid, value, cum) in enumerate(rows):
        screens.append({'index': i, 'candidate_id': cid, 'valid': valid, 'step_time': value, 'cum': cum})
    return {'arm': 'X', 'screens': screens, 'used': used, 'stop_reason': 'budget:screens'}


def cum(screens, tokens, wall, gpuh, usd=0.0):
    return {'screens': screens, 'llm_tokens': tokens, 'wall_seconds': wall, 'gpu_hours': gpuh, 'llm_usd': usd}


def test_y_per_environment():
    assert tj.y_value('autoresearch', 0.9950, 0.9977) == pytest.approx(0.0027)
    assert tj.y_value('rebench_llmf', 108.05, 216.1) == pytest.approx(math.log(2))
    assert tj.y_value('optico', 1.0, 1.0, raw={'tflops': 400.0}, ref_raw={'tflops': 345.4}) == pytest.approx(math.log(400 / 345.4))
    assert tj.y_value('optico', 0.9, 1.0) == pytest.approx(math.log(1 / 0.9))
    assert tj.y_value('autoresearch', None, 0.99) is None and tj.y_value('x', 1.0, 0) is None


def test_margins_on_y():
    assert tj.margin_y(toy_task(env='autoresearch', margin={'kind': 'fixed', 'value': 0.002})) == 0.002
    assert tj.margin_y(toy_task(env='rebench_llmf')) == pytest.approx(-math.log(0.95))
    t = toy_task(env='optico', margin={'kind': 'rsd', 'floor': 0.03, 'k': 3})
    assert tj.margin_y(t, rsd=0.005) == pytest.approx(math.log(1.03)) and tj.margin_y(t, rsd=0.02) == pytest.approx(math.log(1.06))


def test_events_ctg_auc_env_c():
    task = toy_task(env='autoresearch', metric_name='val_bpb', tau=0.002, margin={'kind': 'fixed', 'value': 0.002},
                    budget={'screens': 4, 'llm_tokens': 1000, 'wall_seconds': 1000.0, 'gpu_hours': 4.0})
    tr = _trace([('c0', True, 0.9977, cum(1, 100, 100, 0.1)), ('c1', False, None, cum(2, 300, 300, 0.2)),
                 ('c2', True, 0.9940, cum(3, 400, 500, 0.3))], used=cum(3, 600, 700, 0.3, 1.0))
    ev = tj.events(tr, task, 0.9977)
    assert [e.kind for e in ev] == ['screen', 'screen', 'screen', 'tail']
    assert [round(e.tokens) for e in ev] == [100, 200, 100, 200] and ev[1].step_time is None
    assert -math.log(ev[2].step_time) == pytest.approx(0.0037)          # pseudo step time encodes y
    c = tj.ctg(ev, task)
    assert c['reached'] and c['gpuh'] == pytest.approx(0.3) and c['wall_s'] == pytest.approx(500)
    a = tj.auc(ev, task)
    # x = max(screens/4, tokens/1000, wall/1000): 0.25, 0.5, 0.75 at the screens, 0.75 at the tail; y = 0.0037 after c2
    assert a['final_y'] == pytest.approx(0.0037) and a['auc'] == pytest.approx(0.0037 * 0.25)
    row = tj.arm_row(tr, task, 0.9977, locked=0.9945, locked_valid=True)
    assert row['best_search'] == 0.9940 and row['y_locked'] == pytest.approx(0.0032) and row['n_valid'] == 2


def test_compare_labels():
    task = toy_task(env='autoresearch', metric_name='val_bpb', tau=0.002, margin={'kind': 'fixed', 'value': 0.002})
    is_row = {'locked_valid': True, 'y_locked': 0.005, 'AUC_budget': 0.002, 'invalid_accept_locked': 0}
    assert tj.compare(task, is_row, {'locked_valid': True, 'y_locked': 0.002, 'AUC_budget': 0.001})['label'] == 'advantage'
    assert tj.compare(task, is_row, {'locked_valid': True, 'y_locked': 0.0045, 'AUC_budget': 0.001})['label'] == 'parity'
    assert tj.compare(task, is_row, {'locked_valid': True, 'y_locked': 0.008, 'AUC_budget': 0.001})['label'] == 'disadvantage'
    assert tj.compare(task, is_row, {'locked_valid': True, 'y_locked': 0.002, 'AUC_budget': 0.009})['label'] == 'no_advantage'
    bad = tj.compare(task, {**is_row, 'locked_valid': False}, {'locked_valid': True, 'y_locked': 0.0})
    assert bad['label'] == 'invalid' and bad['invalid_accept'] == 1
    assert tj.compare(task, is_row, {'locked_valid': False, 'y_locked': None, 'AUC_budget': 0.0})['label'] == 'advantage'


def test_scoring_score_is_used_unchanged():
    ev = [score.Event(gpuh=1, tokens=0, wall_s=10, step_time=tj.pseudo_step_time(0.5))]
    assert score.ctg(ev, 1.0, 0.4)['reached'] and not score.ctg(ev, 1.0, 0.6)['reached']
