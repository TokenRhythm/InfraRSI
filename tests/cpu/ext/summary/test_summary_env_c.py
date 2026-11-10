"""ENV-C table and labels (design §4.4 / §0.3, AMENDMENT_EXT_002 Q2, AMENDMENT_EXT_007) on synthetic records."""
import json

import pytest
from summary_helpers import World, write

from infrarsi.ext.summary import env_c
from infrarsi.ext.summary import records as RC


def build(w):
    store = RC.Store(w.root)
    return env_c.build(store, w.root, w.ledger, abandoned=RC.abandoned(w.ledger)), store


def test_infra_track_advantage_and_columns(tmp_path):
    w = World(tmp_path)
    w.env_c_complete(is_locked=0.990, nat_locked=0.994)
    sec, store = build(w)
    lab = sec['labels']['C-IS vs C-NAT-infra']
    assert lab['label'] == 'advantage' and lab['delta_y'] == pytest.approx(0.004) and lab['margin_y'] == 0.002
    a = sec['arms']['C-IS']
    assert a['locked']['val_bpb_locked']['v'] == 0.990 and a['locked']['steps']['v'] == 936
    assert a['search']['val_bpb_best_search']['v'] == 0.991
    assert a['search']['AUC_budget']['v'] > 0
    nat = sec['arms']['C-NAT-infra']['search']
    assert nat['n_crash']['v'] == 1 and nat['n_failed_by_class']['v'] == {'cuda_graph': 1}
    assert sec['arms']['C-RND']['search']['n_failed_by_class']['v'] == {'oom': 1}
    assert sec['calibration']['gate']['v'] is True and sec['anchor']['val_bpb']['v'] == pytest.approx(0.99636466)
    assert 'C-NAT-free' not in ' '.join(sec['labels'])                # the free track is never compared
    assert sec['problems'] == [] and store.problems == []


@pytest.mark.parametrize('is_b,nat_b,want', [(0.9930, 0.9935, 'parity'), (0.9960, 0.9930, 'disadvantage')])
def test_parity_and_disadvantage(tmp_path, is_b, nat_b, want):
    w = World(tmp_path)
    w.env_c_complete(is_locked=is_b, nat_locked=nat_b)
    sec, _ = build(w)
    assert sec['labels']['C-IS vs C-NAT-infra']['label'] == want


def test_invalid_locked_candidate(tmp_path):
    w = World(tmp_path)
    w.env_c_complete(is_valid=False)
    sec, _ = build(w)
    assert sec['labels']['C-IS vs C-NAT-infra']['label'] == 'invalid'


def test_llm_vs_random_sentence_carries_the_coverage(tmp_path):
    w = World(tmp_path)
    w.env_c_complete(is_locked=0.9945, nat_locked=0.9950, rnd_locked=0.9950)
    sec, _ = build(w)
    assert sec['labels']['C-IS vs C-RND']['label'] == 'parity'
    s = ' '.join(sec['random_sentences'])
    assert 'no evidence that LLM search (C-IS) beats random' in s and 'covered 0.043' in s


def test_occupied_screens_are_runner_errors_and_make_the_arm_incomplete(tmp_path):
    w = World(tmp_path)
    w.env_c_complete()
    d = w.root / 'autoresearch' / 'crnd_x'
    tr = json.loads((d / 'trace.json').read_text(encoding='utf-8'))
    for i in range(3, 6):
        att = f'crnd_x-occ{i}'
        write(d / 'runner' / 'receipts' / f'{att}.json',
              w.c_receipt('crnd_x', att, f'c{i}', arm='C-RND', valid=False, runner_error={'kind': 'occupied'}))
        cum = dict(tr['screens'][-1]['cum'], accounting_complete=False)
        tr['screens'].append({'index': i, 'candidate_id': f'c{i}', 'attempt_id': att, 'valid': False, 'step_time': None,
                              'cum': cum})
    tr['stop_reason'] = 'same_failure_x6'
    write(d / 'trace.json', tr)
    sec, _ = build(w)
    s = sec['arms']['C-RND']['search']
    assert s['n_experiments']['v'] == 3 and s['n_runner_errors']['v'] == {'occupied': 3}
    assert sec['labels']['C-IS vs C-RND']['label'] == 'incomplete'
    assert any('occupied' in f for f in sec['findings']) and any('same_failure_x6' in f for f in sec['findings'])


def test_pending_while_running_and_abandoned_runs(tmp_path):
    w = World(tmp_path)
    w.c_anchor()
    (w.root / 'autoresearch' / 'cis_y').mkdir(parents=True)
    w.ledger['ENV-C']['arms'] = {'C-IS': 'cis_y', 'C-NAT-infra': None, 'C-NAT-free': None, 'C-RND': None}
    w.c_arm('C-IS', 'cis_old', [0.996], stop='budget:llm_tokens')
    w.ledger['ENV-C']['arms']['C-IS'] = 'cis_y'
    w.ledger['ENV-C']['abandoned'] = [{'run_id': 'cis_* (old)', 'reason': 'plan mode (#29)'}]
    sec, _ = build(w)
    assert sec['arms']['C-IS']['status'] == 'running' and sec['labels']['C-IS vs C-NAT-infra']['label'] == 'pending'
    assert [x['run_id'] for x in sec['abandoned']] == ['cis_old'] and sec['abandoned'][0]['cost']['llm_usd'] == 1.0


def test_markdown(tmp_path):
    w = World(tmp_path)
    w.env_c_complete()
    text = env_c.markdown(build(w)[0])
    assert 'Infra-track label' in text and 'C-NAT-free — published-protocol reproduction' in text
    assert 'D-AR-7' in text and 'G-C calibration' in text
