"""The report explains a promoted candidate whose fresh re-run did not reproduce its screen (D1-20) and an incumbent
re-run after two agreeing aborts; the promotion row carries the two numbers it needs."""
import json
import math
import sys
from pathlib import Path

import pytest

from infrarsi.project.report import markdown
from infrarsi.verify import Verifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'verify'))   # test_verifier's receipt helper


def _result(promoted=True, beyond=True, rechecks=None):
    items = [{'id': 'speed_reproduces', 'status': 'pass', 'detail': '', 'evidence': {'speedup': 1.048, 'margin': 0.007}}]
    tried = [{'candidate': 'c1', 'step_time': 3.0624, 'items': items, 'passed': True, 'rescore': ['a', 'b'],
              'rerun_vs_screen': {'screen': 3.0624, 'rerun': 3.2445, 'log_ratio': round(math.log(3.2445 / 3.0624), 5), 'beyond_margin': beyond},
              'reference_repeat': -0.0033}]
    promo = {'promoted': 'c1' if promoted else None, 'config': {'micro_batch_size': 2}, 'tried': tried, 'speedup': 1.048,
             'why': 'no candidate passed promotion'}
    return {'run_id': 'x', 'run_dir': '/r', 'project': {'name': 'p', 'gpus': 8, 'steps': 30, 'window': [11, 30]}, 'tolerance': 0.03,
            'outcome': 'improved' if promoted else 'no_change', 'reference': {'step_time': 3.4114, 'config': {'micro_batch_size': 1}},
            'candidates': [{'id': 'c0', 'step_time': 3.4114, 'valid': True}, {'id': 'c1', 'step_time': 3.0624, 'valid': True, 'change': {'micro_batch_size': [1, 2]}}],
            'promotion': promo, 'search': {'used': {'screens': 9}, 'stop_reason': 'budget:screens', 'rechecks': rechecks,
                                           'noise': {'mode': 'measured', 'sigma_pair': 0.0035, 'pairs': [-0.0037, -0.0033], 'split_halves': [0.0] * 5}},
            'screens': [], 'cost': {}}


def test_promoted_candidate_whose_rerun_disagrees_gets_the_two_sentences():
    text = markdown(_result())
    assert 'Its fresh re-run was 5.9 % slower than its screen (3.244 s vs 3.062 s; two runs of one configuration agree within 0.7 % ' \
           'on this machine, and the reference repeated within 0.3 %): the configuration ran at two speeds during this run, and ' \
           'x1.048 is the one the slower run gives.' in text
    assert 'run the recommended configuration twice: a repeat near 3.062 s means the faster speed is available, one near 3.244 s means it is not.' in text
    assert 'fresh re-run' not in markdown(_result(beyond=False)).split('## Recommendation')[1].split('Apply with')[0].replace('fresh re-run x1.048', '')


def test_turned_down_candidate_whose_rerun_disagrees_is_explained():
    text = markdown(_result(promoted=False))
    assert "c1's screen did not reproduce: its fresh re-run was 5.9 % slower than its screen" in text and 'run it twice by hand' in text


def test_recheck_line_in_the_search_section():
    rc = [{'incumbent': 'c1', 'after_aborts': ['c5', 'c6'], 'aborted_step_times': [3.2461, 3.2386], 't_screen': 3.0624,
           't_rerun': 3.2445, 'd': 0.0578, 'threshold': 0.0073, 'moved': True, 'gpu_hours': 0.4}]
    text = markdown(_result(rechecks=rc))
    assert ('Incumbent re-run: after c5 and c6 were aborted at the same speed (3.246 s, 3.239 s) against c1\'s screen (3.062 s), c1 was '
            'run again: 3.244 s (+5.8 %, beyond the noise), so later candidates were compared with 3.244 s; c5 and c6 may be as fast '
            'as c1 rather than slower.') in text
    rc[0].update(t_rerun=3.07, d=0.0025, moved=False)
    assert '3.07 s (+0.2 %, within the noise), so its screen stands.' in markdown(_result(rechecks=rc))


def test_promotion_row_carries_the_rerun_against_the_screen(tmp_path):
    from infrarsi.project import Project
    from infrarsi.project.flow import promote
    from test_verifier import receipt                  # the verify suite's helper
    (tmp_path / 'train.py').write_text('print(1)\n', encoding='utf-8')
    p = Project({'name': 'n', 'launch': {'command': '{python} train.py {args}'}, 'knobs': {},
                 'workload': {'steps': 10, 'window': [3, 10]}}, tmp_path / 'infrarsi.yaml')
    (tmp_path / 'rc').mkdir()
    (tmp_path / 'rc' / 'c1.json').write_text(json.dumps({'config': {}, 'overlay': {}}), encoding='utf-8')
    ref = receipt('ref', t=1.0)
    runs = iter([receipt('rerun', t=1.0, split_half=0.001, rsd_window=0.0001), receipt('c1', t=1 / 1.048, split_half=0.001, rsd_window=0.0001)])

    class Runner:
        def run(self, *a, **kw):
            return next(runs)

        def run_tests(self, *a, **kw):
            return []
    noise = {'mode': 'measured', 'sigma_pair': 0.0035, 'pairs': [-0.0037], 'split_halves': [0.001] * 3}
    trace = {'candidates': [{'id': 'c1', 'valid': True, 'step_time': 1 / 1.114}], 'extra': {'noise': noise}}   # the screen: x1.114
    promo = promote(p, None, Runner(), Verifier(), trace, ref, 'r', tmp_path / 'rc')
    row = promo['tried'][0]
    assert promo['promoted'] == 'c1' and row['rerun_vs_screen']['beyond_margin'] and row['reference_repeat'] == 0.0
    assert row['rerun_vs_screen']['log_ratio'] == pytest.approx(math.log(1.114 / 1.048), abs=1e-4)


def test_a_faster_rerun_is_worded_by_the_numbers():
    # DF4_r1 #4: a screen of 3.244 s and a fresh re-run of 3.062 s
    r = _result()
    t = r['promotion']['tried'][0]
    t['rerun_vs_screen'] = {'screen': 3.2445, 'rerun': 3.0624, 'log_ratio': round(math.log(3.0624 / 3.2445), 5),
                            'beyond_margin': True}
    text = markdown(r)
    assert 'Its fresh re-run was 5.6 % faster than its screen (3.062 s vs 3.244 s' in text
    assert ('x1.048 is the one the faster run gives. Before relying on x1.048, run the recommended configuration twice: a '
            'repeat near 3.062 s means the faster speed is available, one near 3.244 s means it is not.') in text
    rc = [{'incumbent': 'c1', 'after_aborts': ['c5', 'c6'], 'aborted_step_times': [3.4, 3.41], 't_screen': 3.2445,
           't_rerun': 3.0624, 'd': -0.05776, 'threshold': 0.0073, 'moved': False, 'gpu_hours': 0.4}]
    assert ('3.062 s (-5.8 %, faster than its screen beyond the noise), so later candidates were compared with the mean '
            'of the two runs.') in markdown(_result(rechecks=rc))


def test_robust_halves_need_a_valid_reference_pair_at_promotion(tmp_path):
    # DF4_r2 #1: no pair in the search and a reference re-run invalid against the original reference (its loss 5 % off
    # at a 3 % tolerance) while the fresh runs agree with each other: the plain halves stay in use (HEAD: > x1.144)
    from infrarsi.project import Project
    from infrarsi.project.flow import promote
    from test_verifier import receipt                  # the verify suite's helper
    (tmp_path / 'train.py').write_text('print(1)\n', encoding='utf-8')
    p = Project({'name': 'n', 'launch': {'command': '{python} train.py {args}'}, 'knobs': {},
                 'workload': {'steps': 10, 'window': [3, 10]}}, tmp_path / 'infrarsi.yaml')
    (tmp_path / 'rc').mkdir()
    (tmp_path / 'rc' / 'c1.json').write_text(json.dumps({'config': {}, 'overlay': {}}), encoding='utf-8')
    ref = receipt('ref', t=1.0)
    off = lambda s: 1.05 * (2.0 - 0.01 * s)            # noqa: E731 - both fresh runs 5 % above the original reference
    runs = iter([receipt('rerun', t=1.0, loss=off, split_half=math.log(1.1), split_half_robust=0.0),
                 receipt('c1', t=0.99, loss=off, split_half=0.0, split_half_robust=0.0)])

    class Runner:
        def run(self, *a, **kw):
            return next(runs)

        def run_tests(self, *a, **kw):
            return []
    noise = {'mode': 'measured', 'sigma_pair': 0.0, 'pairs': [], 'split_halves': [math.log(1.1), 0.0],
             'split_halves_plain': [math.log(1.1), 0.0], 'split_halves_robust': [0.0, 0.0]}
    trace = {'candidates': [{'id': 'c1', 'valid': True, 'step_time': 0.99}], 'extra': {'noise': noise}}
    promo = promote(p, None, Runner(), Verifier(), trace, ref, 'r', tmp_path / 'rc')
    m = next(i['evidence']['margin'] for i in promo['tried'][0]['items'] if i['id'] == 'speed_reproduces' and i.get('evidence'))
    assert promo['promoted'] is None and m > 0.14       # x1.010 is inside the plain halves' noise
