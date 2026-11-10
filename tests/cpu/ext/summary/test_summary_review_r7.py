"""Regression tests for the Codex review round 7 of the W5 summary: R7-N1 (an F1 arm with a U+C reference must keep
its receipts/c0.json copy; a locked run declares both halves; the no-improvement record stays valid) and R7-N2 (the G4
limit is final only with the complete EXT_005 D inventory: the five facts, a valid block-S0 norm and one valid norm per
applicable node-reference run; a restatement never replaces the facts; norms are finite and non-negative)."""
import json

import pytest
from summary_helpers import World, write
from test_summary_d27 import _f1, _f1_world

from infrarsi.ext.summary import env_a
from infrarsi.ext.summary import records as RC

PAIRS = (('aisw_x', 'anat_x'), ('aisw_x', 'arnd_x'), ('anat_x', 'arnd_x'))


def g4(final=True, value=4.9, **extra):
    return {'final': final, 'refprime_pairwise_norms': [{'pair': list(p), 'norm_diff': value, 'source': 't'} for p in PAIRS],
            **extra}


def build_a(w, inputs):
    return env_a.build(RC.Store(w.root), w.root, w.ledger, abandoned=RC.abandoned(w.ledger, w.root), g4_inputs=inputs)


def resign(path, **changes):
    r = json.loads(path.read_text(encoding='utf-8'))
    r.pop('harness_signature', None)
    r.update(changes)
    write(path, RC.signing.sign(r))


# ---------------------------------------------------------------- R7-N1
def test_r7n1_a_missing_c0_copy_fails(tmp_path):
    w = _f1_world(tmp_path)
    (w.root / 'flashnext' / 'fa0_x' / 'receipts' / 'c0.json').unlink()
    acc = _f1(w, tmp_path)['accounting']['F-A0']
    assert acc['complete'] is False and any('c0.json' in x and 'is missing' in x for x in acc['reasons'])


def test_r7n1_a_locked_run_without_its_candidate_half_fails(tmp_path):
    w = _f1_world(tmp_path)
    d = w.root / 'flashnext' / 'fa0_x'
    write(d / 'locked.json', {'locked_run': True, 'uc_half': {'attempt_id': 'lkuc'}})
    (d / 'receipts' / 'lkcand.json').unlink()                          # the half and its receipt dropped together
    acc = _f1(w, tmp_path)['accounting']['F-A0']
    assert acc['complete'] is False and any('without its candidate_half attempt id' in x for x in acc['reasons'])


def test_r7n1_the_no_improvement_record_has_no_pair(tmp_path):
    w = _f1_world(tmp_path)
    d = w.root / 'flashnext' / 'fa0_x'
    write(d / 'locked.json', {'locked_run': False, 'status': 'no_improvement', 'order': []})
    for att in ('lkuc', 'lkcand'):
        (d / 'receipts' / f'{att}.json').unlink()
    assert _f1(w, tmp_path)['accounting']['F-A0']['complete'] is True


# ---------------------------------------------------------------- R7-N2
def test_r7n2_exact_pairs_without_the_d_inventory_are_not_final():
    cal = env_a.g4_calibration(g4(), [], {'A-NAT': 'anat_x', 'A-ISW': 'aisw_x', 'A-RND': 'arnd_x'})
    assert cal['final'] is False and any('block S0' in x for x in cal['d_inventory_gaps'])


def test_r7n2_the_complete_inventory_is_final(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    cal = build_a(w, g4())['g4_calibration']
    assert cal['final'] is True and cal['d_inventory_gaps'] == [] and len(cal['samples_d']) == 5 + 1 + 2


@pytest.mark.parametrize('case', ['block_not_run', 'one_row_override', 'block_s0_invalid', 'node_ref_negative',
                                  'pair_negative', 'node_ref_not_run'])
def test_r7n2_an_incomplete_d_inventory_is_not_final(tmp_path, case):
    w = World(tmp_path)
    w.env_a_complete()
    inputs = g4()
    if case == 'block_not_run':
        w.ledger['ENV-A']['block'] = {'run_id': None, 'node': 'WINH1'}
    elif case == 'one_row_override':
        inputs = g4(s0_d_norms=[{'source': 'only-one', 'norm_diff': 1.0}])
    elif case == 'block_s0_invalid':
        resign(w.root / 'rebench_llmf' / 'blk_x' / 'receipts' / 'blk-S0.json', valid=False)
    elif case == 'node_ref_negative':
        f = w.root / 'rebench_llmf' / 'blk_x' / 'receipts' / 'nr-WINH3.json'
        r = json.loads(f.read_text(encoding='utf-8'))
        resign(f, judge={**r['judge'], 'message': {**r['judge']['message'], 'norm_diff': -0.5}})
    elif case == 'pair_negative':
        inputs['refprime_pairwise_norms'][0]['norm_diff'] = -1.0
    else:
        w.ledger['ENV-A']['node_refs']['WINH4'] = None
    sec = build_a(w, inputs)
    cal = sec['g4_calibration']
    assert cal['final'] is False, cal['estimator']
    assert any("limit_D'" in p and 'not final' in p for p in sec['pending'])
    if case == 'one_row_override':                                       # the facts stay; m is not replaced
        assert [x['norm_diff'] for x in cal['samples_d'][:5]] == list(env_a.PREARM_S0_D)
        assert all(x['source'] != 'only-one' for x in cal['samples_d'])
        assert any('s0_d_norms' in x for x in cal['d_inventory_gaps'])
