"""ENV-B tables, margin_w and labels (design §3.6 / §0.3, AMENDMENT_EXT_002 Q1) on synthetic records."""
import json
import math

import pytest
from summary_helpers import World, write

from infrarsi.ext.optico import margin as OM
from infrarsi.ext.summary import env_b
from infrarsi.ext.summary import records as RC


def build(w):
    store = RC.Store(w.root)
    return env_b.build(store, w.root, w.ledger, abandoned=RC.abandoned(w.ledger)), store


def test_workload_advantage_margin_and_environment_rule(tmp_path):
    w = World(tmp_path)
    for wk in ('W1', 'W3', 'W2'):
        w.env_b_workload(wk, is_T=330.0 if wk != 'W2' else 302.0, nat_T=300.0)
    sec, store = build(w)
    W3 = sec['workloads']['W3']
    m = W3['margin_record']
    assert m['estimator'] == 'run_to_run' and m['n'] == 8 and W3['margin_w']['v'] == pytest.approx(max(0.03, 3 * m['rsd']))
    lab = W3['labels']['B-IS vs B-NAT']
    assert lab['label'] == 'advantage'
    assert lab['delta_y'] == pytest.approx(math.log(330.0 / 300.0))
    assert lab['margin_y'] == pytest.approx(math.log1p(W3['margin_w']['v']))
    assert lab['drift_default_halves'] == pytest.approx((228.0 - 2.0) / (228.0 + 1.0))
    assert sec['workloads']['W2']['labels']['B-IS vs B-NAT']['label'] == 'parity'
    assert sec['env_label']['label'] == 'advantage'                    # W1, W3 advantage, W2 parity, 0 disadvantage
    a = W3['arms']['B-IS']
    assert a['locked']['tflops_locked']['v'] == 330.0 and a['locked']['speedup_vs_default']['v'] == pytest.approx(330.0 / 226.0)
    s = a['search']
    assert s['n_trials']['v'] == 3 and s['n_failed_runs']['v'] == 1 and s['n_failed_runs']['note'] == 'oom 1'
    assert s['tflops_best_search']['v'] == 330.0 and s['trials_to_first_improvement']['v'] == 2
    assert s['AUC_budget']['v'] > 0 and s['CTG_reached']['v'] is True
    assert sec['method_behaviour_reproduced']['value'] == 'yes'
    assert sec['problems'] == [] and store.problems == []


def test_invalid_counts_as_a_disadvantage_for_the_environment_rule(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W1', is_T=330.0)
    w.env_b_workload('W3', is_T=330.0)
    w.env_b_workload('W2', is_T=330.0, is_valid=False)
    sec, _ = build(w)
    assert sec['workloads']['W2']['labels']['B-IS vs B-NAT']['label'] == 'invalid'
    assert sec['env_label']['label'].startswith('no advantage')
    assert sec['workloads']['W2']['arms']['B-IS']['locked']['invalid_accept_locked']['v'] == 1


def test_disadvantage_and_llm_vs_random_sentence(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3', is_T=250.0, nat_T=300.0, is_auc_T=250.0)
    sec, _ = build(w)
    W = sec['workloads']['W3']
    assert W['labels']['B-IS vs B-NAT']['label'] == 'disadvantage'
    assert W['labels']['B-IS vs B-RND']['label'] == 'disadvantage'
    assert any('no evidence that LLM search (B-IS) beats random' in s for s in W['random_sentences'])


def test_pooled_fallback_and_abandoned_runs_never_enter_s_w(tmp_path):
    w = World(tmp_path)
    w.b_single('W1', 'B-DEF', 'bdef_W1', 344.5)
    w.b_search('W1', 'B-NAT', 'bnat_old', [250.0, 400.0])          # abandoned: an outlier default row
    w.ledger['ENV-B']['W1']['B-NAT'] = 'bnat_new'
    w.ledger['ENV-B']['W1']['abandoned'] = [{'run_id': 'bnat_old', 'reason': 'plan mode (#29)'}]
    sec, _ = build(w)
    m = sec['workloads']['W1']['margin_record']
    assert m['estimator'] == 'pooled_fallback' and [x['run_id'] for x in m['S_w']] == ['bdef_W1']
    assert m['margin'] == pytest.approx(max(0.03, 3 * OM.pooled_rsd({'c': [t for _, t in OM.CALIBRATION['W3']]})[0]))
    assert sec['workloads']['W1']['arms']['B-NAT']['status'] == 'not started'
    assert [x['run_id'] for x in sec['abandoned']] == ['bnat_old'] and sec['abandoned'][0]['cost']['gpu_hours'] == 2.0
    assert any('margin_w: pooled_fallback' in p for p in sec['pending'])


def test_a_records_margin_file_that_disagrees_is_a_problem(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    write(w.root / 'optico' / 'margin_W3.json', {'S_w': [{'attempt_id': 'other'}], 'estimator': 'run_to_run', 'margin': 0.03})
    sec, _ = build(w)
    assert any('margin_W3.json differs' in p for p in sec['problems'])


def test_locked_candidate_must_be_the_final_selection(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    w.b_locked('W3', 'B-IS', 'lk_bis_W3', 226.0, 330.0, params={'MICRO_BATCH_SIZE': 4, 'TP': 1, 'PP': 1, 'AC': 'none'})
    sec, _ = build(w)
    lk = sec['workloads']['W3']['arms']['B-IS']['locked']
    assert lk['locked_valid']['v'] is False and any('not the arm\'s final selection' in p for p in sec['problems'])


def test_incomplete_accounting_and_not_run_arms(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W1')
    sel = w.b_search('W1', 'B-IS', 'bis_W1', [230.0, None, 330.0], accounting=False)
    assert sel['MICRO_BATCH_SIZE'] == 2
    w.ledger['ENV-B']['W1']['B-ISW'] = None                         # null = not started yet: pending
    sec, _ = build(w)
    W = sec['workloads']['W1']
    assert W['labels']['B-IS vs B-NAT']['label'] == 'incomplete'
    assert W['labels']['B-ISW vs B-NAT']['label'] == 'pending' and any('B-ISW' in p for p in sec['pending'])
    w.ledger['ENV-B']['W1']['B-ISW'] = 'not run: cut by §6.4 (order 2)'   # a sanctioned final state
    sec, _ = build(w)
    assert sec['workloads']['W1']['labels']['B-ISW vs B-NAT']['label'] == 'not run'
    assert not any('B-ISW' in p for p in sec['pending'])


def test_runner_errors_and_stop_rules_are_findings(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    d = w.root / 'optico' / 'brnd_W3'
    tr = json.loads((d / 'trace.json').read_text(encoding='utf-8'))
    att = 'brnd_W3-c9'
    write(d / 'receipts' / f'{att}.json', w.b_receipt('W3', 'brnd_W3', att, 'c9', arm='B-RND', purpose='screen', T=1.0,
                                                      valid=False, runner_error={'kind': 'occupied'}))
    tr['screens'].append({'index': 3, 'candidate_id': 'c9', 'attempt_id': att, 'valid': False, 'step_time': None,
                          'cum': tr['screens'][-1]['cum']})
    tr['stop_reason'] = 'same_failure_x6'
    write(d / 'trace.json', tr)
    sec, _ = build(w)
    s = sec['workloads']['W3']['arms']['B-RND']['search']
    assert s['n_trials']['v'] == 3 and s['n_runner_errors']['v'] == {'occupied': 1} and s['n_failed_runs']['v'] == 1
    assert any('same_failure_x6' in f for f in sec['findings'])


def test_markdown(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    text = env_b.markdown(build(w)[0])
    assert '## W3' in text and 'margin_w' in text and 'iteration 4' in text and 'B-IS vs B-NAT' in text
