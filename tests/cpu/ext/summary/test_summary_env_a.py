"""ENV-A table and labels (design §2.7 / §0.3; AMENDMENT_EXT_001 / 004 / 005 / 006) on synthetic records."""
import math

import pytest
from summary_helpers import World, iso, write

from infrarsi.ext.summary import env_a
from infrarsi.ext.summary import records as RC

CLAUSE_I = ("G4", "score.py judged (cand(D'), ref'(D')) invalid on the permuted dataset")
CLAUSE_II = ("G4", "||cand(D')-cand(D)|| 1.000 < 0.5 * ||ref'(D')-ref(D)|| 29.000 (ratio 0.034): output barely depends on the data")


def build(w, g4=None):
    store = RC.Store(w.root)
    return env_a.build(store, w.root, w.ledger, abandoned=RC.abandoned(w.ledger), g4_inputs=g4), store


def lab(sec, pair='A-ISW vs A-NAT'):
    return sec['labels'][pair]


def test_advantage_from_block_values_and_node_referenced_auc(tmp_path):
    w = World(tmp_path)
    w.env_a_complete(isw_t=100.0, nat_t=150.0, rnd_t=140.0)
    sec, store = build(w)
    res = lab(sec)
    assert res['label'] == 'advantage' and res['predeclared']['label'] == 'advantage'
    assert res['delta_y'] == pytest.approx(math.log(150.0 / 100.0))
    assert res['margin_y'] == pytest.approx(-math.log(0.95))
    isw = sec['rows']['A-ISW']
    assert isw['y_B']['v'] == pytest.approx(math.log(216.0 / 100.0))
    assert isw['locked']['locked_valid_amended']['v'] is True
    assert isw['locked']['t_locked']['v'] == 101.0 and isw['locked_vs_block']['v'] == pytest.approx(101.0 / 100.0)
    assert isw['y_best_search']['v'] == pytest.approx(math.log(216.0 / 110.0))      # own node reference (WINH3)
    assert isw['AUC_budget']['src'] and all(':' in s or '/' in s for s in isw['AUC_budget']['src'])
    assert sec['node_factor']['WINH3']['v'] == pytest.approx(1.0)
    assert sec['problems'] == [] and not any('block' in p for p in sec['pending'])


@pytest.mark.parametrize('isw_t,nat_t,want', [(100.0, 102.0, 'parity'), (150.0, 100.0, 'disadvantage')])
def test_parity_and_disadvantage(tmp_path, isw_t, nat_t, want):
    w = World(tmp_path)
    w.env_a_complete(isw_t=isw_t, nat_t=nat_t)
    sec, _ = build(w)
    assert lab(sec)['label'] == want


def test_no_advantage_when_auc_condition_fails(tmp_path):
    w = World(tmp_path)
    w.env_a_complete(isw_t=100.0, nat_t=150.0, isw_screen=210.0, nat_screen=120.0)
    sec, _ = build(w)
    res = lab(sec)
    assert res['label'] == 'no_advantage' and res['conditions']['auc_not_worse'] is False
    assert res['conditions']['beyond_margin'] is True


def test_invalid_when_the_infrarsi_locked_rescore_is_vetoed(tmp_path):
    w = World(tmp_path)
    w.env_a_complete(isw_locked={'t': 101.0, 'rejections': [('G3', '||trained-base|| too small')]})
    sec, _ = build(w)
    assert lab(sec)['label'] == 'invalid' and lab(sec)['invalid_accept'] == 1
    assert sec['rows']['A-ISW']['locked']['invalid_accept_locked']['v'] == 1


def test_native_invalid_falls_back_to_the_reference(tmp_path):
    w = World(tmp_path)
    w.env_a_complete(isw_t=100.0, nat_locked={'t': 151.0, 'judge_valid': False, 'nd': 17.8})
    sec, _ = build(w)
    res = lab(sec)
    assert res['label'] == 'advantage' and res['delta_y'] == pytest.approx(math.log(216.0 / 100.0))   # y_native = 0


def test_incomplete_accounting_labels_incomplete(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    w.a_arm('A-ISW', 'aisw_x', node='WINH3', screens=[('c0', 216.0, True), ('c1', 110.0, True)], locked={'t': 101.0},
            accounting=False)
    sec, _ = build(w)
    assert lab(sec)['label'] == 'incomplete' and lab(sec)['label_if_complete'] == 'advantage'
    assert any('accounting incomplete' in f for f in sec['findings'])


def test_degenerate_native_arm_gives_no_comparison(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    w.a_arm('A-NAT', 'anat_x', node='WINH1', screens=[('score01', 211.0, False)], locked=None, stop='degenerate_loop')
    sec, _ = build(w)
    assert lab(sec)['label'] == 'no comparison (native arm degenerate under the D1 driver)'
    assert lab(sec, 'A-NAT vs A-RND')['label'].startswith('no comparison')
    assert lab(sec, 'A-ISW vs A-RND')['label'] in ('advantage', 'parity', 'disadvantage', 'no_advantage')


def test_pending_without_block_and_while_an_arm_runs(tmp_path):
    w = World(tmp_path)
    w.node_ref('WINH3', 216.0)
    w.a_arm('A-ISW', 'aisw_x', node='WINH3', screens=[('c0', 216.0, True), ('c1', 110.0, True)], locked={'t': 101.0})
    (w.root / 'rebench_llmf' / 'anat_x').mkdir(parents=True)
    write(w.root / 'rebench_llmf' / 'anat_x' / 'events.jsonl', {'utc': iso(0), 'kind': 'llm_call'})
    w.ledger['ENV-A']['arms']['A-NAT'] = {'run_id': 'anat_x', 'node': 'WINH1'}
    w.ledger['ENV-A']['arms']['A-RND'] = {'run_id': None}
    sec, _ = build(w)
    assert lab(sec)['label'] == 'pending' and 'A-NAT running' in lab(sec)['why']
    assert sec['rows']['A-NAT']['status'] == 'running'
    assert any('block' in p for p in sec['pending'])
    assert sec['rows']['A-ISW']['AUC_budget']['v'] is not None          # its node reference exists


def test_g4_clause_i_predeclared_invalid_amended_by_the_calibrated_limit(tmp_path):
    w = World(tmp_path)
    w.env_a_complete(isw_locked={'t': 101.0, 'nd': 4.44, 'rejections': [CLAUSE_I],
                                 'g4': {'cand_move': 29.12, 'ref_move': 29.11, 'ratio': 1.0004}})
    sec, _ = build(w)                                                   # no inputs: the D' norm is not in the receipt
    lk = sec['rows']['A-ISW']['locked']
    assert lk['locked_valid_predeclared']['v'] is False
    assert lk['locked_valid_amended']['status'] == 'pending' and lk['g4_dprime_norm']['status'] == 'pending'
    assert lab(sec)['label'] == 'pending' and lab(sec)['predeclared']['label'] == 'invalid'
    assert lk['g4_limit_dprime']['v'] == 6.17
    sec, _ = build(w, g4={'locked_dprime_norms': {'A-ISW': {'norm_cand_dprime_vs_refprime': 5.41, 'source': 'node file'}}})
    lk = sec['rows']['A-ISW']['locked']
    assert lk['locked_valid_amended']['v'] is True and lk['g4_dprime_norm']['v'] == 5.41
    assert lab(sec)['label'] == 'advantage' and lab(sec)['predeclared']['label'] == 'invalid'
    assert sec['g4_calibration']['estimator'].startswith('provisional') and sec['g4_calibration']['final'] is False


def test_g4_limit_computed_from_pairwise_norms_can_go_either_way(tmp_path):
    w = World(tmp_path)
    w.env_a_complete(isw_locked={'t': 101.0, 'rejections': [CLAUSE_I], 'g4': {'cand_move': 29.1, 'ref_move': 29.1, 'ratio': 1.0}})
    low = {'final': True, 'locked_dprime_norms': {'A-ISW': 5.41},
           'refprime_pairwise_norms': [{'pair': list(p), 'norm_diff': 3.0, 'source': 't'}
                                       for p in (('aisw_x', 'anat_x'), ('aisw_x', 'arnd_x'), ('anat_x', 'arnd_x'))]}
    sec, _ = build(w, g4=low)
    cal = sec['g4_calibration']
    d = list(env_a.PREARM_S0_D) + [4.2, 4.2, 4.0]                    # + node_ref WINH3 / WINH4 and the block S0 norms
    assert cal['final'] is True
    assert cal['m'] == pytest.approx(sum(d) / len(d)) and cal['m_prime'] == pytest.approx((4.79 + 3 * 3.0) / 4)
    assert cal['limit'] == pytest.approx(5.3 + cal['m_prime'] - cal['m']) and cal['limit'] < 5.41
    assert sec['rows']['A-ISW']['locked']['locked_valid_amended']['v'] is False
    assert lab(sec)['label'] == 'invalid'
    # an arm that passed clause (i) at 5.3 needs its D' norm when the final limit is below 5.3
    assert sec['rows']['A-NAT']['locked']['locked_valid_amended']['status'] == 'pending'


def test_g4_clause_ii_and_g6_identity_invalidate_under_both_rules(tmp_path):
    w = World(tmp_path)
    w.env_a_complete(isw_locked={'t': 101.0, 'rejections': [CLAUSE_II], 'g4': {'cand_move': 1.0, 'ref_move': 29.0, 'ratio': 0.034}},
                     nat_locked={'t': 151.0, 'digest_ok': False})
    sec, _ = build(w, g4={'locked_dprime_norms': {'A-ISW': 5.0}})
    isw, nat = sec['rows']['A-ISW']['locked'], sec['rows']['A-NAT']['locked']
    assert isw['locked_valid_amended']['v'] is False and isw['locked_valid_predeclared']['v'] is False
    assert nat['identity_ok']['v'] is False and nat['locked_valid_amended']['v'] is False
    assert any('G6 identity' in p for p in sec['problems'])


def test_g7_is_reported_not_applied_at_the_locked_stage(tmp_path):
    w = World(tmp_path)
    w.env_a_complete(isw_locked={'t': 101.0, 'nd': 5.2})
    sec, _ = build(w)
    lk = sec['rows']['A-ISW']['locked']
    assert lk['g7_locked']['v'].startswith('would veto') and lk['locked_valid_amended']['v'] is True
    assert lab(sec)['label'] == 'advantage'


def test_block_fallback_suffix(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    w.block(216.0, {'A-NAT': 150.0, 'A-ISW': 100.0, 'A-RND': 140.0}, fallback=('A-ISW',))
    sec, _ = build(w)
    assert lab(sec)['label'] == 'advantage' and lab(sec)['suffix'] == ' (fallback)'


def test_llm_vs_random_sentence(tmp_path):
    w = World(tmp_path)
    w.env_a_complete(isw_t=100.0, rnd_t=101.0)
    sec, _ = build(w)
    assert lab(sec, 'A-ISW vs A-RND')['label'] == 'parity'
    assert any('no evidence that LLM search (A-ISW) beats random' in s for s in sec['random_sentences'])


def test_search_columns_t_best_8h_first_cold_and_llm_aggregates(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    sec, _ = build(w)
    s = sec['rows']['A-ISW']['search']
    assert s['t_best_search']['v'] == 110.0 and s['t_best_search_8h']['v'] == 110.0
    assert s['t_first']['v'] == 216.0 and s['first_cold']['v'] is True        # data_prep 24 s > 13.4 s
    assert s['llm_tokens_uncached']['v'] == 40_000 and s['llm_gpu_hours_h20']['v'] == 0.4
    assert s['early_finish']['v'] is True and s['n_scored']['v'] == 2
    assert sec['rows']['A-NAT']['search']['restart_of']['v'] == 'anat_a'


def test_solo_wall_fraction_and_abandoned_runs_stay_off_the_result_rows(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    w.a_arm('A-NAT', 'anat_x', node='WINH1', screens=[('score01', 216.0, True), ('score02', 160.0, True)],
            locked={'t': 151.0}, started=300.0)
    d = w.root / 'rebench_llmf' / 'anat_old'
    write(d / 'ABANDONED.json', {'status': 'abandoned', 'run_id': 'anat_old', 'turns': 50, 'scorings': 0, 'llm_calls': 51,
                                 'llm_tokens': 123, 'llm_wall_seconds': 360.0, 'first_event_utc': iso(-3600),
                                 'last_event_utc': iso(-1800), 'most_repeated_signature': {'count': 40, 'first_turn': 9,
                                                                                            'max_consecutive': 40}})
    w.ledger['ENV-A']['arms']['A-NAT']['abandoned'] = [{'run_id': 'anat_old', 'reason': 'loop'}]
    sec, _ = build(w)
    isw = sec['rows']['A-ISW']['solo_wall_fraction']['v']           # A-ISW [0, 660 s], A-NAT [300, 960 s]
    assert isw == pytest.approx(300.0 / 660.0)
    assert sec['rows']['A-NAT']['solo_wall_fraction']['v'] == pytest.approx(1 - 360.0 / 660.0)
    assert [x['run_id'] for x in sec['abandoned']] == ['anat_old']
    ab = sec['abandoned'][0]
    assert ab['cost']['wall_hours'] == pytest.approx(0.5) and ab['cost']['gpu_hours'] == pytest.approx(2.0)
    assert ab['cost']['llm_gpu_hours_h20'] == pytest.approx(0.4) and ab['most_repeated']['count'] == 40
    assert all(r['run_id'] != 'anat_old' for r in sec['rows'].values())


def test_matched_budget_prefix_after_an_infrastructure_stop(tmp_path):
    w = World(tmp_path)
    w.env_a_complete(isw_t=100.0, nat_t=150.0)
    w.a_arm('A-NAT', 'anat_x', node='WINH1', screens=[('score01', 216.0, True), ('score02', 160.0, True)],
            locked={'t': 151.0}, stop='endpoint_unreachable')
    sec, _ = build(w)
    mp = lab(sec).get('matched_prefix')
    assert mp and mp['x_matched'] == pytest.approx(min(sec['rows'][a]['x_end']['v'] for a in ('A-ISW', 'A-NAT')))
    pts = [(0.0, 0.0), (0.5, 1.0)]
    assert env_a.auc_prefix(pts, 0.75) == pytest.approx(0.25) and env_a.auc_prefix(pts, 0.25) == pytest.approx(0.0)


def test_locked_rescore_run_by_the_executor_is_found_through_the_ledger(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    res = w.a_arm('A-RND', 'arnd_x', node='WINH4', screens=[('c0', 216.0, True), ('c1', 150.0, True)], locked=None)
    assert res['locked']['attempt_id'] is None
    sec, _ = build(w)
    assert sec['rows']['A-RND']['locked']['status']['status'] == 'pending'
    d = w.root / 'rebench_llmf' / 'arnd_lock_x'
    write(d / 'receipts' / 'exec-locked.json', w.a_receipt('arnd_lock_x', 'exec-locked', 'locked', arm='A-RND', purpose='score',
                                                           t=141.0, node='WINH4', digest='dg-c1',
                                                           g4={'cand_move': 29.0, 'ref_move': 29.0, 'ratio': 1.0}))
    w.ledger['ENV-A']['arms']['A-RND']['locked'] = 'arnd_lock_x'
    sec, _ = build(w)
    lk = sec['rows']['A-RND']['locked']
    assert lk['t_locked']['v'] == 141.0 and lk['locked_valid_amended']['v'] is True
    assert lk['attempt_id']['src'] == ['arnd_lock_x:exec-locked']


def test_runner_errors_are_not_scorings(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    d = w.root / 'rebench_llmf' / 'aisw_x'
    import json
    res = json.loads((d / 'result.json').read_text(encoding='utf-8'))
    att = 'aisw_x-c2'
    write(d / 'receipts' / f'{att}.json', w.a_receipt('aisw_x', att, 'c2', arm='A-ISW', t=0.0, judge_valid=False,
                                                      runner_error={'kind': 'occupied', 'message': 'x'}))
    res['trace']['screens'].append({'index': 2, 'candidate_id': 'c2', 'attempt_id': att, 'valid': False, 'step_time': None,
                                    'cum': res['trace']['screens'][-1]['cum']})
    write(d / 'result.json', res)
    sec, _ = build(w)
    s = sec['rows']['A-ISW']['search']
    assert s['n_scored']['v'] == 2 and s['n_runner_errors']['v'] == {'occupied': 1}
    assert any('refused or lost by our harness' in f for f in sec['findings'])


def test_a_tampered_receipt_is_a_problem(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    import json
    p = w.root / 'rebench_llmf' / 'aisw_x' / 'receipts' / 'aisw_x-locked.json'
    r = json.loads(p.read_text(encoding='utf-8'))
    r['judge']['message']['execution_time'] = 50.0
    write(p, r)
    sec, store = build(w)
    assert any('signature does not verify' in x for x in store.problems)


def test_markdown_renders_every_section(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    text = env_a.markdown(build(w)[0])
    for h in ('## Labels', '## Locked re-score per arm', '## Block', '## Search', '## Costs', '## Abandoned runs',
              'Token dimension not applied'):
        assert h in text
    assert 'advantage' in text and 'locked_valid_amended' in text
