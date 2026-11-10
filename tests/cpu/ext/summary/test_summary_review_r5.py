"""Regression tests for the Codex review round 5 of the W5 summary: R5-N1 (the ENV-B accounting checks run for every
finished search, also one whose producer says complete) and R5-N2 (every receipt of the run directory used by exactly
one screen; per-row tokens and USD reconciled against an explicit call-to-row boundary for every LLM arm)."""
import json

import pytest
from summary_helpers import World, write
from test_summary_d26 import D24, _near

from infrarsi.ext.summary import env_b
from infrarsi.ext.summary import records as RC


def build_b(w):
    return env_b.build(RC.Store(w.root), w.root, w.ledger, abandoned=RC.abandoned(w.ledger, w.root))


def acc(sec, arm, w='W3'):
    s = sec['workloads'][w]['arms'][arm]['search']
    return s['accounting'], s.get('accounting_d26'), s.get('accounting_reconciliation')


def resign(path, drop=(), **changes):
    r = json.loads(path.read_text(encoding='utf-8'))
    r.pop('harness_signature', None)
    r.update(changes)
    for k in drop:
        r.pop(k, None)
    write(path, RC.signing.sign(r))


def edit_trace(w, rid, fn):
    p = w.root / 'optico' / rid / 'trace.json'
    tr = json.loads(p.read_text(encoding='utf-8'))
    fn(tr)
    write(p, tr)


def test_producer_complete_searches_pass_the_checks_with_their_declared_basis(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    sec = build_b(w)
    for arm in ('B-NAT', 'B-IS', 'B-RND'):
        (ok, why), d26, rec = acc(sec, arm)
        assert ok is True and why == [] and d26 is None and rec['mismatches'] == []
    assert acc(sec, 'B-IS')[2]['row_boundary'].startswith('events.jsonl')
    assert acc(sec, 'B-NAT')[2]['row_boundary'].startswith('after_screen')
    assert acc(sec, 'B-NAT')[2]['gpu_basis'].startswith('lease') and acc(sec, 'B-IS')[2]['gpu_basis'].startswith('sum')


# ---------------------------------------------------------------- R5-N1
@pytest.mark.parametrize('case', ['receipt_without_cost', 'used_gpu_zero', 'call_without_completeness', 'nan_total',
                                  'negative_usd', 'brnd_llm_use'])
def test_r5n1_a_producer_complete_trace_that_fails_the_checks_is_incomplete(tmp_path, case):
    w = World(tmp_path)
    w.env_b_workload('W3')
    arm, rid = ('B-RND', 'brnd_W3') if case == 'brnd_llm_use' else ('B-IS', 'bis_W3')
    if case == 'receipt_without_cost':
        resign(w.root / 'optico' / rid / 'receipts' / f'{rid}-c1.json', drop=('cost',))
    elif case == 'used_gpu_zero':
        edit_trace(w, rid, lambda tr: tr['used'].update(gpu_hours=0.0))
    elif case == 'call_without_completeness':
        edit_trace(w, rid, lambda tr: tr['llm_calls'][1].pop('accounting_complete'))
    elif case == 'nan_total':
        edit_trace(w, rid, lambda tr: tr['screens'][1]['cum'].update(wall_seconds=float('nan')))
    elif case == 'negative_usd':
        edit_trace(w, rid, lambda tr: tr['llm_calls'][0].update(llm_usd=-1.0))
    else:
        edit_trace(w, rid, lambda tr: tr['used'].update(llm_tokens=10))
    sec = build_b(w)
    (ok, why), d26, rec = acc(sec, arm)
    assert ok is False and d26 is None and any(x.startswith('accounting checks failed') for x in why), why
    assert sec['workloads']['W3']['labels'][f'{arm} vs B-NAT' if arm == 'B-IS' else 'B-IS vs B-RND']['label'] == 'incomplete'
    assert any(f'{rid}' in f and 'accounting incomplete' in f for f in sec['findings'])


# ---------------------------------------------------------------- R5-N2 (1): receipt coverage
def test_r5n2_an_unreferenced_charged_receipt_is_incomplete(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    write(w.root / 'optico' / 'bis_W3' / 'receipts' / 'extra.json',
          w.b_receipt('W3', 'bis_W3', 'extra', 'c99', arm='B-IS', purpose='screen', T=300.0, gpu_s=7200.0))
    (ok, why), _, rec = acc(build_b(w), 'B-IS')
    assert ok is False and any('no screen references' in x and '2.000000 GPUh' in x for x in why)
    assert rec['n_receipts'] == rec['n_screens'] + 1


def test_r5n2_an_unreferenced_receipt_blocks_d26(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    _near(w)
    assert acc(build_b(w), 'B-NAT')[0][0] is True                          # D26(b) granted without it
    f = write(w.root / 'optico' / 'bnat_W3' / 'receipts' / 'auxcharge.json',
              w.b_receipt('W3', 'bnat_W3', 'auxcharge', 'it99', arm='B-NAT', purpose='native', T=1.0, valid=False,
                          runner_error=D24, gpu_s=7200.0))
    resign(f, judge=None, phases={'total': 80.0},
           deadline={'refused_calls': [{'phase': 'pre_launch', 'func': 'launch', 'late_s': 3.0}]},
           cost={'gpu_seconds': 7200.0, 'wall_seconds': 900.0, 'accounting_complete': False, 'basis': 'held'})
    (ok, why), d26, rec = acc(build_b(w), 'B-NAT')
    assert ok is False and d26 is None and any('no screen references' in x for x in why)


# ---------------------------------------------------------------- R5-N2 (2): per-row LLM totals, explicit boundary
def _usd(tr, per_call=0.1):
    for k, c in enumerate(tr['llm_calls']):
        c['llm_usd'] = per_call
    for i, s in enumerate(tr['screens']):
        s['cum']['llm_usd'] = per_call * (i + 1)
    tr['used']['llm_usd'] = per_call * len(tr['screens'])


@pytest.mark.parametrize('case', ['row_tokens', 'row_usd', 'row_usd_above_final', 'no_events', 'event_ids_differ'])
def test_r5n2_per_row_llm_totals_of_an_infrarsi_arm_need_the_events_boundary(tmp_path, case):
    w = World(tmp_path)
    w.env_b_workload('W3')
    edit_trace(w, 'bis_W3', _usd)                                          # nonzero USD, consistent everywhere
    assert acc(build_b(w), 'B-IS')[0][0] is True
    if case == 'row_tokens':
        edit_trace(w, 'bis_W3', lambda tr: tr['screens'][1]['cum'].update(llm_tokens=1500))      # finals unchanged
    elif case == 'row_usd':
        edit_trace(w, 'bis_W3', lambda tr: tr['screens'][1]['cum'].update(llm_usd=0.25))
    elif case == 'row_usd_above_final':
        edit_trace(w, 'bis_W3', lambda tr: tr['screens'][2]['cum'].update(llm_usd=5.0))
    elif case == 'no_events':
        (w.root / 'optico' / 'bis_W3' / 'events.jsonl').unlink()
    else:
        p = w.root / 'optico' / 'bis_W3' / 'events.jsonl'
        p.write_text(p.read_text(encoding='utf-8').replace('bis_W3-call1', 'other-call'), encoding='utf-8')
    (ok, why), _, rec = acc(build_b(w), 'B-IS')
    assert ok is False and any(x.startswith('accounting checks failed') for x in why), why
    if case in ('no_events', 'event_ids_differ'):
        assert any('no explicit call-to-row boundary' in x for x in why)


def test_r5n2_a_native_call_without_an_explicit_after_screen_has_no_boundary(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    edit_trace(w, 'bnat_W3', lambda tr: tr['llm_calls'][1].pop('after_screen'))
    (ok, why), _, _ = acc(build_b(w), 'B-NAT')
    assert ok is False and any('no explicit call-to-row boundary' in x for x in why)
