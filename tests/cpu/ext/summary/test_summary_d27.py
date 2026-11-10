"""D27 (O's ruling 10-21): the accounting checks for every finished search arm in ENV-A, ENV-C and F1 - well-formed
receipts and call records, finite non-negative totals, every receipt used by exactly one screen or a declared non-screen
charge (ENV-A's locked re-score; F1's start reference and locked halves), GPU = the receipts on the declared basis,
tokens / USD = the call log; per-row LLM attribution verified where the producer records a boundary (InfraRSI: the
events.jsonl order), producer-asserted for the basic-agent arms (A-NAT, C-NAT-free, C-NAT-infra), never incomplete
for that reason alone."""
import json

import pytest
from summary_helpers import World, signed, write

from infrarsi.ext.summary import env_a, env_c, f1
from infrarsi.ext.summary import records as RC


def build_a(w):
    return env_a.build(RC.Store(w.root), w.root, w.ledger, abandoned=RC.abandoned(w.ledger, w.root),
                       g4_inputs={'final': True, 'refprime_pairwise_norms': [
                           {'pair': list(p), 'norm_diff': 4.9, 'source': 't'}
                           for p in (('aisw_x', 'anat_x'), ('aisw_x', 'arnd_x'), ('anat_x', 'arnd_x'))]})


def build_c(w):
    return env_c.build(RC.Store(w.root), w.root, w.ledger, abandoned=RC.abandoned(w.ledger, w.root))


def edit_json(p, fn):
    d = json.loads(p.read_text(encoding='utf-8'))
    fn(d)
    write(p, d)


# ---------------------------------------------------------------- ENV-A
def test_d27_env_a_every_search_arm_is_checked(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    sec = build_a(w)
    for arm in ('A-NAT', 'A-ISW', 'A-RND'):
        s = sec['rows'][arm]['search']
        assert s['accounting'] == (True, []), (arm, s['accounting'])
        assert any(k.startswith('locked re-score') for k in s['accounting_reconciliation']['non_screen_charges'])
    assert sec['rows']['A-NAT']['search']['accounting_reconciliation']['row_boundary'].startswith('producer-asserted')
    assert sec['rows']['A-ISW']['search']['accounting_reconciliation']['row_boundary'].startswith('events.jsonl')
    assert 'per-row LLM attribution producer-asserted (D27)' in env_a.markdown(sec)
    assert sec['labels']['A-ISW vs A-NAT']['label'] == 'advantage'


@pytest.mark.parametrize('case', ['unreferenced_receipt', 'gpu_total', 'anat_tokens', 'anat_call_flag'])
def test_d27_env_a_a_failed_check_makes_the_arm_incomplete(tmp_path, case):
    w = World(tmp_path)
    w.env_a_complete()
    d = w.root / 'rebench_llmf'
    arm = {'unreferenced_receipt': 'A-ISW', 'gpu_total': 'A-RND'}.get(case, 'A-NAT')
    if case == 'unreferenced_receipt':
        write(d / 'aisw_x' / 'receipts' / 'stray.json', w.a_receipt('aisw_x', 'stray', 'c9', arm='A-ISW', t=150.0))
    elif case == 'gpu_total':
        edit_json(d / 'arnd_x' / 'result.json', lambda r: r['trace']['used'].update(gpu_hours=r['trace']['used']['gpu_hours'] + 1))
    elif case == 'anat_tokens':                          # A-NAT's rows are producer-asserted, its totals are not
        edit_json(d / 'anat_x' / 'result.json', lambda r: r['used'].update(llm_tokens=r['used']['llm_tokens'] + 5))
    else:
        edit_json(d / 'anat_x' / 'result.json', lambda r: r['llm_calls'][0].pop('accounting_complete'))
    sec = build_a(w)
    ok, why = sec['rows'][arm]['search']['accounting']
    assert ok is False and any(x.startswith('accounting checks failed') for x in why), why
    pair = 'A-ISW vs A-NAT' if arm in ('A-ISW', 'A-NAT') else 'A-NAT vs A-RND'
    assert sec['labels'][pair]['label'] == 'incomplete'


# ---------------------------------------------------------------- ENV-C
def test_d27_env_c_every_search_arm_is_checked(tmp_path):
    w = World(tmp_path)
    w.env_c_complete()
    sec = build_c(w)
    for arm in ('C-IS', 'C-NAT-infra', 'C-NAT-free', 'C-RND'):
        assert sec['arms'][arm]['search']['accounting'] == (True, []), arm
    for arm in ('C-NAT-infra', 'C-NAT-free'):
        assert sec['arms'][arm]['search']['accounting_reconciliation']['row_boundary'].startswith('producer-asserted')
    assert sec['arms']['C-IS']['search']['accounting_reconciliation']['row_boundary'].startswith('events.jsonl')
    assert 'producer-asserted (D27)' in env_c.markdown(sec)


@pytest.mark.parametrize('case', ['nat_usd', 'cis_no_events', 'crnd_unreferenced'])
def test_d27_env_c_a_failed_check_makes_the_arm_incomplete(tmp_path, case):
    w = World(tmp_path)
    w.env_c_complete()
    d = w.root / 'autoresearch'
    if case == 'nat_usd':
        arm = 'C-NAT-infra'
        edit_json(d / 'cnatinfra_x' / 'agent' / 'trace.json', lambda t: t['used'].update(llm_usd=2.0))
    elif case == 'cis_no_events':                        # an InfraRSI arm has a boundary to verify: none is a failure
        arm = 'C-IS'
        (d / 'cis_x' / 'events.jsonl').unlink()
    else:
        arm = 'C-RND'
        write(d / 'crnd_x' / 'runner' / 'receipts' / 'stray.json', w.c_receipt('crnd_x', 'stray', 'c9', arm='C-RND'))
    ok, why = build_c(w)['arms'][arm]['search']['accounting']
    assert ok is False and any(x.startswith('accounting checks failed') for x in why), why


# ---------------------------------------------------------------- F1 (synthetic; real records at G-FIN)
def _receipt(att, rid, arm, gpu_s, cid=None, purpose='screen'):
    return signed({'schema': 'ext-receipt-v1', 'env': 'flashnext', 'task_id': 'F1', 'arm': arm, 'run_id': rid,
                   'attempt_id': att, 'candidate_id': cid or att, 'purpose': purpose, 'valid': True,
                   'cost': {'gpu_seconds': gpu_s, 'wall_seconds': gpu_s / 32, 'accounting_complete': True}})


def _f1_world(tmp_path, *, locked=True, start_declared=True, used_gpu=4.0):
    w = World(tmp_path)
    ex = w.root / 'flashnext'
    uc = _receipt('ucatt', 'ucref_x', 'UC-REF', 7200.0)
    write(ex / 'ucref_x' / 'uc_ref.json', {'kind': 'UC-REF', 'receipt_attempt': 'ucatt' if start_declared else 'other',
                                           'receipt': uc})
    d = ex / 'fa0_x'
    write(d / 'receipts' / 'c0.json', uc)                 # SearchSession's copy of the start reference (not charged)
    for att in ('s1', 's2'):
        write(d / 'receipts' / f'{att}.json', _receipt(att, 'fa0_x', 'F-A0', 7200.0))
    for att, cid in (('lkuc', 'locked-uc'), ('lkcand', 'locked-candidate')):          # X6's half_id names
        write(d / 'receipts' / f'{att}.json', _receipt(att, 'fa0_x', 'F-A0', 7200.0, cid=cid, purpose='locked'))
    if locked:
        write(d / 'locked.json', {'uc_half': {'attempt_id': 'lkuc'}, 'candidate_half': {'attempt_id': 'lkcand'}})
    rows = [{'index': i, 'candidate_id': f'c{i + 1}', 'attempt_id': f's{i + 1}',
             'cum': {'gpu_hours': 2.0 * (i + 1), 'wall_seconds': 100.0 * (i + 1), 'llm_tokens': 1000 * (i + 1),
                     'llm_usd': 0.0, 'accounting_complete': True}} for i in range(2)]
    write(d / 'trace.json', {'arm': 'F-A0', 'schema': 'infrarsi-search-trace-v1', 'mode': 'strong', 'screens': rows,
                             'budget': {'llm_tokens': 1_000_000},
                             'used': {'screens': 2, 'gpu_hours': used_gpu, 'wall_seconds': 250.0, 'llm_tokens': 2000,
                                      'llm_usd': 0.0, 'accounting_complete': True},
                             'llm_calls': [{'call_id': f'k{i}', 'llm_tokens': 1000, 'llm_usd': 0.0,
                                            'accounting_complete': True} for i in range(2)],
                             'stop_reason': 'finish', 'ended_utc': '2026-10-21T10:00:00+00:00'})
    (d / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in (
        {'kind': 'llm_call', 'payload': {'call_id': 'k0'}}, {'kind': 'screen', 'payload': {'index': 0}},
        {'kind': 'llm_call', 'payload': {'call_id': 'k1'}}, {'kind': 'screen', 'payload': {'index': 1}})), encoding='utf-8')
    w.ledger['F1']['references'] = {'UC-REF': 'ucref_x', 'F-HT': None}
    w.ledger['F1']['arms']['F-A0'] = 'fa0_x'
    return w


class _NoX6:
    @staticmethod
    def main(args):
        return 3                                         # the stub refuses: only the accounting is under test here


def _f1(w, tmp_path):
    return f1.build(RC.Store(w.root), w.root, w.ledger, tmp_path / 'f1out', allow_partial=True, runner=_NoX6)


def test_d27_f1_an_arm_with_declared_start_reference_and_locked_halves_is_complete(tmp_path):
    w = _f1_world(tmp_path)
    sec = _f1(w, tmp_path)
    acc = sec['accounting']['F-A0']
    assert acc['complete'] is True, acc
    assert set(acc['reconciliation']['non_screen_charges']) == {
        'start reference (U+C, charged to UC-REF) (ucatt)', 'locked pair uc half (locked.json) (lkuc)',
        'locked pair candidate half (locked.json) (lkcand)'}
    assert acc['reconciliation']['row_boundary'].startswith('events.jsonl')
    assert not any('F-A0' in p and 'accounting' in p for p in sec['pending'])
    assert f1.rows(sec)['F-A0']['accounting']['v'] == 'complete'


@pytest.mark.parametrize('kw', [{'locked': False}, {'start_declared': False}, {'used_gpu': 5.0}])
def test_d27_f1_an_undeclared_receipt_or_a_wrong_total_blocks_the_arm(tmp_path, kw):
    w = _f1_world(tmp_path, **kw)
    sec = _f1(w, tmp_path)
    assert sec['accounting']['F-A0']['complete'] is False
    assert any(p.startswith('F1 F-A0 (fa0_x): accounting incomplete') for p in sec['pending'])
