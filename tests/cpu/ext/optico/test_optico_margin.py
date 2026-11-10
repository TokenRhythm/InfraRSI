"""H-07 / AMENDMENT_EXT_002 Q1 (D11): margin_w = max(3 %, 3 x run-to-run RSD) over the valid protocol default runs,
pooled fallback (with the W3 calibration group) while n < 4; margin_<W>.json from the receipts."""
import json
import math
import statistics
import uuid

import pytest

from infrarsi.ext.common import trajectory
from infrarsi.ext.optico import margin as mg
from infrarsi.ext.optico import task as taskmod
from infrarsi.ext.optico import workloads as wl
from infrarsi.harness import signing

W1, W3 = wl.get('W1'), wl.get('W3')
CAL = [228.83, 233.67, 226.20, 223.50]                     # the amendment's W3 calibration set (R3 + G-B smokes)


def rec(T, *, w=W3, arm='B-DEF', purpose='b_def', cid='def', valid=True, params=None, native=None, sanitizer=0,
        run_id='run', sign=True):
    r = {'schema': 'ext-receipt-v1', 'env': 'optico', 'task_id': w.task_id, 'arm': arm, 'purpose': purpose,
         'candidate_id': cid, 'run_id': run_id, 'attempt_id': uuid.uuid4().hex[:12], 'node': 'WINH6', 'valid': valid,
         'judge_valid': valid, 'metric': {'value': 9.6 if valid else None, 'raw': {'tflops_per_gpu': T}},
         'params': dict(params or w.defaults), 'sanitizer': {'fired': sanitizer, 'events': []}}
    if native is not None:
        r['native'] = native
    if sign:
        signing.sign(r)
    return r


def nat(T, kind, row, loop, *, retry_of=None, valid=True, purpose='native'):
    return rec(T, arm='B-NAT', purpose=purpose, cid=f'it{row + 1}', valid=valid, run_id=loop,
               native={'kind': kind, 'row_index': row, 'retry_of': retry_of, 'native_attempt': loop, 'sanitizer_events': []})


def test_w3_provisional_margin_is_5_7_percent():
    out = mg.compute('W3', [rec(229.0)])
    assert out['estimator'] == 'pooled_fallback' and out['n'] == 1 and out['pooled_df'] == 3
    assert out['rsd_used'] == pytest.approx(statistics.stdev(CAL) / statistics.mean(CAL))
    assert round(out['rsd_used'] * 100, 2) == 1.90 and round(out['margin'] * 100, 1) == 5.7
    assert [x['T'] for x in out['S_w']] == [229.0]                          # calibration reported beside S_w, not in it
    assert [x['T'] for x in out['calibration']['W3']] == CAL
    assert mg.compute('W3', [])['margin'] == pytest.approx(out['margin'])   # before any protocol run: same 5.7 %


def test_run_to_run_once_four_protocol_defaults_exist():
    rs = [rec(230.0), rec(226.0, arm='B-IS', purpose='screen', cid='c0'), rec(228.0, arm='B-RND', purpose='screen', cid='c0'),
          nat(232.0, 'default', 0, 'loopA'), rec(224.0, arm='B-IS', purpose='locked', cid='locked-default'),
          rec(231.0, arm='B-ISW', purpose='screen', cid='c0')]                # B-ISW runs on W1 only: not a W3 kind
    out = mg.compute('W3', rs)
    vals = [230.0, 226.0, 228.0, 232.0, 224.0]
    rsd = statistics.stdev(vals) / statistics.mean(vals)
    assert out['estimator'] == 'run_to_run' and out['n'] == 5 and out['rsd'] == pytest.approx(rsd)
    assert out['margin'] == pytest.approx(max(0.03, 3 * rsd)) and out['mean'] == pytest.approx(228.0)
    assert sorted(x['kind'] for x in out['S_w']) == sorted(['b_def', 'search_start', 'search_start', 'b_nat_cycle0',
                                                            'locked_default'])
    low = mg.compute('W3', [rec(228.0), rec(228.1, cid='c0', arm='B-IS', purpose='screen'), nat(228.2, 'default', 0, 'L'),
                            rec(228.1, purpose='locked', arm='B-RND', cid='locked-default')])
    assert low['estimator'] == 'run_to_run' and low['margin'] == 0.03                      # the 3 % floor


def test_membership_by_run_kind_validity_and_provenance():
    bad_sig = rec(240.0)
    bad_sig['metric']['raw']['tflops_per_gpu'] = 260.0                     # edited after signing
    rs = [rec(229.0),
          rec(250.0, arm='B-ED', purpose='b_ed', cid='bed'),                # not a default run kind
          rec(229.5, arm='B-IS', purpose='screen', cid='c3'),               # a later screen, even with default params
          rec(229.6, arm='B-IS', purpose='locked', cid='locked-candidate'), # the candidate half of a pair
          nat(229.7, 'new', 1, 'loopA'),
          nat(230.0, 'default', 0, 'smoke', purpose='calibration'),         # G-B smoke: calibration group, never S_w
          rec(191.6, arm='B-IS', purpose='functional_check', cid='fc1'),
          rec(0.0, valid=False),                                            # failed B-DEF: excluded, not replaced
          rec(231.0, params=W3.knobs(MICRO_BATCH_SIZE=2)),                  # not the defaults
          bad_sig, rec(232.0, sanitizer=1), rec(233.0, sign=False),
          nat(0.0, 'default', 0, 'loopB', valid=False), nat(227.0, 'retry', 1, 'loopB', retry_of=0),   # retried cycle 0
          nat(226.0, 'default', 0, 'loopC'), nat(226.5, 'retry', 1, 'loopC', retry_of=0),             # second value ignored
          rec(345.0, w=W1),                                                 # another workload
          rec(229.8, arm='B-DEF', purpose='locked', cid='locked-default'),  # B-DEF has no locked pair
          rec(229.9, arm='B-ISW', purpose='screen', cid='c0'),              # B-ISW is a W1 arm
          rec(229.95, arm='B-ISW', purpose='locked', cid='locked-default')]
    members, excluded = mg.sample(rs, 'W3')
    assert sorted((m['kind'], m['receipt']['metric']['raw']['tflops_per_gpu']) for m in members) == \
        [('b_def', 229.0), ('b_nat_cycle0', 226.0), ('b_nat_cycle0', 227.0)]
    why = sorted(x['why'] for x in excluded)
    assert why.count('harness signature missing or invalid') == 2 and 'sanitizer event' in why
    assert 'parameters are not the workload defaults' in why and why.count('not valid (judge or guards)') == 2
    assert 'not the first valid cycle-0 row of its loop' in why and len(excluded) == 7


def test_b_isw_kinds_count_on_w1_only():
    rs = [rec(345.0, w=W1), rec(346.0, w=W1, arm='B-ISW', purpose='screen', cid='c0'),
          rec(347.0, w=W1, arm='B-ISW', purpose='locked', cid='locked-default'),
          rec(348.0, w=W1, arm='B-NAT', purpose='locked', cid='locked-default')]
    members, excluded = mg.sample(rs, 'W1')
    assert sorted(m['kind'] for m in members) == ['b_def', 'locked_default', 'locked_default', 'search_start']
    assert excluded == [] and mg.sample([rec(229.0, arm='B-ISW', purpose='screen', cid='c0')], 'W3') == ([], [])


def test_pooled_fallback_spans_workloads_and_calibration():
    rs = [rec(345.0, w=W1), rec(350.0, w=W1), rec(229.0)]
    out = mg.compute('W3', rs)
    m1 = 347.5
    num = sum((t / m1 - 1) ** 2 for t in (345.0, 350.0)) + sum((t / statistics.mean(CAL) - 1) ** 2 for t in CAL)
    assert out['estimator'] == 'pooled_fallback' and out['pooled_df'] == 1 + 3
    assert out['rsd_used'] == pytest.approx(math.sqrt(num / 4)) and set(out['pooled_groups']) == {'W1', 'calibration_W3'}
    assert mg.compute('W1', rs)['pooled_df'] == 4                           # W1 has n = 2 < 4: the same pool


def test_writes_margin_json_for_trajectory_compare(tmp_path):
    rs = [rec(229.0), rec(226.0, arm='B-RND', purpose='screen', cid='c0')]
    runs = tmp_path / 'runs'
    for i, r in enumerate(rs):
        d = runs / f'run{i}'
        d.mkdir(parents=True)
        (d / 'receipt_b_def.json').write_text(json.dumps(r), encoding='utf-8')
    (runs / 'run0' / 'b_ed_pick.json').write_text('{"params": {}}', encoding='utf-8')            # not a receipt
    (runs / 'run1' / 'receipts').mkdir()
    (runs / 'run1' / 'receipts' / 'c0.json').write_text(json.dumps(rs[1]), encoding='utf-8')      # same attempt twice
    loaded = mg.load_receipts([runs])
    assert len(loaded) == 2
    path = mg.write('W3', loaded, tmp_path / 'out')
    out = json.loads(path.read_text(encoding='utf-8'))
    assert path.name == 'margin_W3.json' and out['n'] == 2 and out['estimator'] == 'pooled_fallback'
    assert {x['attempt_id'] for x in out['S_w']} == {r['attempt_id'] for r in rs} and all(x['path'] for x in out['S_w'])
    task = taskmod.make_task('W3', 'B-IS')
    assert trajectory.margin_y(task, out['for_trajectory_compare']['rsd']) == pytest.approx(math.log1p(out['margin']))
