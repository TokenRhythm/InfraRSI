"""Regression tests for the Codex review round 6 of the W5 summary: R6-N1 (an explicitly incomplete receipt is never
accepted on a producer-complete trace; only D26-recognised refusals are allowlisted), R6-N2 (producer-asserted rows and
the producer's basis come from the arm, a mismatching schema / mode is a record error), R6-N3 (an arm without an LLM
budget shows no LLM use and explicit zero tokens and USD), R6-N4 (declared non-screen charges must exist and not be
screens; F1's start copy and locked halves are validated), R6-N5 (R1 #34 agrees with D26)."""
import json
import math

import pytest
from summary_helpers import World, write
from test_summary_d26 import _with_refusal
from test_summary_d27 import _f1, _f1_world, build_a, build_c

from infrarsi.ext.summary import env_b, repro
from infrarsi.ext.summary import records as RC


def build_b(w):
    return env_b.build(RC.Store(w.root), w.root, w.ledger, abandoned=RC.abandoned(w.ledger, w.root))


def resign(path, **changes):
    r = json.loads(path.read_text(encoding='utf-8'))
    r.pop('harness_signature', None)
    r.update(changes)
    write(path, RC.signing.sign(r))


def edit_json(p, fn):
    d = json.loads(p.read_text(encoding='utf-8'))
    fn(d)
    write(p, d)


def b_acc(sec, arm, w='W3'):
    return sec['workloads'][w]['arms'][arm]['search']['accounting']


# ---------------------------------------------------------------- R6-N1
def test_r6n1_a_receipt_saying_incomplete_fails_on_a_producer_complete_trace(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    f = w.root / 'optico' / 'bis_W3' / 'receipts' / 'bis_W3-c1.json'
    resign(f, cost={**json.loads(f.read_text(encoding='utf-8'))['cost'], 'accounting_complete': False})   # numbers intact
    ok, why = b_acc(build_b(w), 'B-IS')
    assert ok is False and any('accounting_complete false' in x for x in why), why


def test_r6n1_a_producer_complete_trace_never_gets_the_d26_allowlist(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    p, _ = _with_refusal(w)                                            # a charged D24 refusal, reconciled totals ...
    def complete(tr):                                                  # ... but the producer says complete
        tr['accounting_complete'] = True
        tr['used']['accounting_complete'] = True
        for s in tr['screens']:
            s['cum']['accounting_complete'] = True
    edit_json(p, complete)
    sec = build_b(w)
    ok, why = b_acc(sec, 'B-IS')
    assert ok is False and sec['workloads']['W3']['arms']['B-IS']['search']['accounting_d26'] is None
    assert any('accounting_complete false' in x for x in why)


def test_r6n1_env_a_a_receipt_saying_incomplete_fails(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    f = w.root / 'rebench_llmf' / 'arnd_x' / 'receipts' / 'arnd_x-c1.json'
    resign(f, cost={**json.loads(f.read_text(encoding='utf-8'))['cost'], 'accounting_complete': False})
    ok, why = build_a(w)['rows']['A-RND']['search']['accounting']
    assert ok is False and any('accounting_complete false' in x for x in why)


# ---------------------------------------------------------------- R6-N2
@pytest.mark.parametrize('env, arm, run, change', [
    ('A', 'A-ISW', 'aisw_x', {'schema': 'ext-basic-agent-trace-v1'}),     # would claim the producer-asserted exception
    ('A', 'A-NAT', 'anat_x', {'schema': 'infrarsi-search-trace-v1'}),
    ('C', 'C-IS', 'cis_x', {'schema': 'ext-basic-agent-trace-v1'}),
    ('B', 'B-IS', 'bis_W3', {'mode': 'native'}),                           # would claim the lease basis (>=)
    ('B', 'B-NAT', 'bnat_W3', {'mode': 'strong'}),
])
def test_r6n2_the_producer_comes_from_the_arm_and_a_mismatch_is_a_record_error(tmp_path, env, arm, run, change):
    w = World(tmp_path)
    if env == 'A':
        w.env_a_complete()
        p = w.root / 'rebench_llmf' / run / 'result.json'
        edit_json(p, lambda r: (r['trace'] if isinstance(r.get('trace'), dict) else r).update(change))
        if arm == 'A-ISW':
            (w.root / 'rebench_llmf' / run / 'events.jsonl').unlink()
        ok, why = build_a(w)['rows'][arm]['search']['accounting']
    elif env == 'C':
        w.env_c_complete()
        edit_json(w.root / 'autoresearch' / run / 'trace.json', lambda t: t.update(change))
        (w.root / 'autoresearch' / run / 'events.jsonl').unlink()
        ok, why = build_c(w)['arms'][arm]['search']['accounting']
    else:
        w.env_b_workload('W3')
        edit_json(w.root / 'optico' / run / 'trace.json', lambda t: t.update(change))
        ok, why = b_acc(build_b(w), arm)
    assert ok is False and any('is not the arm\'s producer' in x for x in why), why


# ---------------------------------------------------------------- R6-N3
@pytest.mark.parametrize('case', ['used_usd', 'row_usd_nan', 'used_usd_missing', 'a_call'])
def test_r6n3_an_arm_without_an_llm_budget_shows_explicit_zero_and_no_call(tmp_path, case):
    w = World(tmp_path)
    w.env_b_workload('W3')
    p = w.root / 'optico' / 'brnd_W3' / 'trace.json'
    if case == 'used_usd':
        edit_json(p, lambda t: t['used'].update(llm_usd=1.0))
    elif case == 'row_usd_nan':
        edit_json(p, lambda t: t['screens'][1]['cum'].update(llm_usd=float('nan')))
    elif case == 'used_usd_missing':
        edit_json(p, lambda t: t['used'].pop('llm_usd'))
    else:
        edit_json(p, lambda t: t.update(llm_calls=[{'call_id': 'x', 'llm_tokens': 0, 'llm_usd': 0.0,
                                                    'accounting_complete': True}]))
    ok, why = b_acc(build_b(w), 'B-RND')
    assert ok is False and any('without an LLM budget' in x for x in why), why


@pytest.mark.parametrize('bad', [math.inf, -1.0])
def test_r6n3_an_llm_arm_with_a_non_finite_or_negative_usd_fails(tmp_path, bad):
    w = World(tmp_path)
    w.env_b_workload('W3')
    edit_json(w.root / 'optico' / 'bis_W3' / 'trace.json', lambda t: t['used'].update(llm_usd=bad))
    ok, why = b_acc(build_b(w), 'B-IS')
    assert ok is False and any('non-finite or negative' in x for x in why), why


# ---------------------------------------------------------------- R6-N4
def _consistent(w):
    p = w.root / 'optico' / 'bis_W3'
    return json.loads((p / 'trace.json').read_text(encoding='utf-8')), RC.Store(w.root).receipts(p), \
        RC.Store(w.root).event_sequence(p / 'events.jsonl')


def test_r6n4_declared_non_screen_ids_must_exist_and_not_be_screens(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    tr, recs, ev = _consistent(w)
    assert RC.accounting_checks(tr, recs, ev)[0] == []
    fails, _ = RC.accounting_checks(tr, recs, ev, non_screen={'ghost': 'locked pair uc half'})
    assert any('without a receipt' in x for x in fails)
    fails, _ = RC.accounting_checks(tr, recs, ev, non_screen={tr['screens'][0]['attempt_id']: 'start reference'})
    assert any('that are screen receipts' in x for x in fails)


def test_r6n4_env_a_a_missing_locked_receipt_is_a_coverage_failure(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    (w.root / 'rebench_llmf' / 'aisw_x' / 'receipts' / 'aisw_x-locked.json').unlink()
    ok, why = build_a(w)['rows']['A-ISW']['search']['accounting']
    assert ok is False and any('without a receipt' in x for x in why)


@pytest.mark.parametrize('case', ['c0_not_the_copy', 'uc_half_names_the_candidate', 'locked_half_missing'])
def test_r6n4_f1_the_start_copy_and_the_locked_halves_are_validated(tmp_path, case):
    w = _f1_world(tmp_path)
    d = w.root / 'flashnext' / 'fa0_x'
    if case == 'c0_not_the_copy':
        resign(d / 'receipts' / 'c0.json', valid=False)                  # same attempt id, different content
    elif case == 'uc_half_names_the_candidate':
        write(d / 'locked.json', {'uc_half': {'attempt_id': 'lkcand'}, 'candidate_half': {'attempt_id': 'lkuc'}})
    else:
        write(d / 'locked.json', {'uc_half': {'attempt_id': 'lkuc'}, 'candidate_half': {'attempt_id': 'lkcand'},
                                  'extra': 1})
        (d / 'receipts' / 'lkcand.json').unlink()
    sec = _f1(w, tmp_path)
    acc = sec['accounting']['F-A0']
    assert acc['complete'] is False, acc
    want = {'c0_not_the_copy': 'not the copy of the U+C reference', 'uc_half_names_the_candidate': 'not the locked-uc half',
            'locked_half_missing': 'without a receipt'}[case]
    assert any(want in x for x in acc['reasons']), acc['reasons']


# ---------------------------------------------------------------- R6-N5
def test_r6n5_r1_issue_34_agrees_with_d26():
    m = json.loads(repro.MAPPING.read_text(encoding='utf-8'))
    note = m['issues']['34']['fidelity_note']
    assert 'does not make the labels incomplete' in note and 'D26' in note and 'labels `incomplete`' not in note
