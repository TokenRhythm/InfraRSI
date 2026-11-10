"""Regression tests for the Codex review round 2 of the W5 summary (partials of findings 1-5, 7, 9; new N1, N2)."""
import json
from pathlib import Path

import pytest
from summary_helpers import World, write

from infrarsi.ext.summary import cli, env_a, env_b, env_c, f1, repro
from infrarsi.ext.summary import records as RC

PAIRS3 = (('aisw_x', 'anat_x'), ('aisw_x', 'arnd_x'), ('anat_x', 'arnd_x'))


def build_a(w, g4=None):
    return env_a.build(RC.Store(w.root), w.root, w.ledger, abandoned=RC.abandoned(w.ledger), g4_inputs=g4)


def build_c(w):
    return env_c.build(RC.Store(w.root), w.root, w.ledger, abandoned=RC.abandoned(w.ledger))


def g4(pairs, final=True, value=4.9):
    return {'final': final, 'refprime_pairwise_norms': [{'pair': list(p), 'norm_diff': value, 'source': 't'} for p in pairs]}


# ---------------------------------------------------------------- 1: G4 calibration
def test_a_started_but_unfinished_arm_keeps_its_pairs_expected(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    w.ledger['ENV-A']['arms']['A-RND'] = {'run_id': None}                 # not started: its cache will come
    cal = build_a(w, g4(PAIRS3[:1]))['g4_calibration']
    assert cal['final'] is False and ['A-RND', 'aisw_x'] in cal['pairs_expected']


def test_a_sanctioned_cut_arm_has_no_pairs_and_the_rest_needs_the_declaration(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    w.ledger['ENV-A']['arms']['A-RND'] = 'not run: cut (§6.4)'
    assert build_a(w, g4(PAIRS3[:1]))['g4_calibration']['final'] is True
    assert build_a(w, g4(PAIRS3[:1], final=False))['g4_calibration']['final'] is False
    assert build_a(w, g4(PAIRS3))['g4_calibration']['final'] is False     # pairs with a cut arm are unusable


@pytest.mark.parametrize('bad', [None, 'x', float('nan')])
def test_non_numeric_pairwise_norms_do_not_crash_and_are_not_final(tmp_path, bad):
    w = World(tmp_path)
    w.env_a_complete()
    rows = g4(PAIRS3)
    rows['refprime_pairwise_norms'][0]['norm_diff'] = bad
    cal = build_a(w, rows)['g4_calibration']
    assert cal['final'] is False and 'unusable' in cal['estimator']


# ---------------------------------------------------------------- 2: formal gate is generic
def test_pending_labels_and_accounting_of_uncompared_arms_enter_the_gate(tmp_path):
    w = World(tmp_path)
    w.env_c_complete()
    w.c_arm('C-NAT-free', 'cnatfree_x', [0.993, 0.985], accounting=False)
    w.ledger['ENV-C']['anchor'] = 'a0c0ffee00099 (x1check_x, missing)'
    c = build_c(w)
    a = {'labels': {}, 'rows': {}}
    b = {'workloads': {}, 'env_label': {'label': 'advantage'}}
    pend = cli.pending_labels(a, b, c) + cli.accounting_pending(a, b, c) + c['pending']
    assert any('C-IS vs C-NAT-infra: pending' in p for p in pend)
    assert any('ENV-C C-NAT-free' in p and 'accounting incomplete' in p for p in pend)
    assert any(p.startswith('ENV-C anchor') for p in pend)


def test_missing_calibration_is_pending(tmp_path):
    w = World(tmp_path)
    w.env_c_complete()
    w.ledger['ENV-C']['calibration'] = None
    assert any('G-C calibration' in p for p in build_c(w)['pending'])


# ---------------------------------------------------------------- 3: ENV-A unknown identity is always pending
def test_unknown_identity_with_an_invalid_judge_is_pending_not_invalid(tmp_path):
    w = World(tmp_path)
    w.env_a_complete(isw_locked={'t': 101.0, 'judge_valid': False, 'nd': 17.0})
    (w.root / 'rebench_llmf' / 'aisw_x' / 'receipts' / 'aisw_x-c1.json').unlink()
    sec = build_a(w)
    lk = sec['rows']['A-ISW']['locked']
    assert lk['locked_valid_amended']['status'] == 'pending' and lk['locked_valid_predeclared']['status'] == 'pending'
    assert sec['labels']['A-ISW vs A-NAT']['label'] == 'pending'


# ---------------------------------------------------------------- 4: ENV-C identity holes
def test_unknown_selected_experiment_is_unverifiable_not_the_baseline(tmp_path):
    w = World(tmp_path)
    w.env_c_complete()
    p = w.root / 'autoresearch' / 'cis_x' / 'trace.json'
    tr = json.loads(p.read_text(encoding='utf-8'))
    tr['selected'] = 'c7'                                               # no such screen
    write(p, tr)
    w.c_selected['C-IS'] = 'sha-base-train'
    w.c_locked('C-IS', 'lk_cis_x', 0.9965, 0.990)                       # candidate trains the baseline train.py
    sec = build_c(w)
    assert sec['arms']['C-IS']['locked']['locked_valid']['status'] == 'pending'
    w.c_locked('C-IS', 'lk_cis_x', 0.9965, 0.990, cand_valid=False)
    assert build_c(w)['arms']['C-IS']['locked']['locked_valid']['status'] == 'pending'


def test_the_anchor_needs_the_ledger_node(tmp_path):
    w = World(tmp_path)
    w.env_c_complete()
    w.ledger['ENV-C'].pop('node')
    sec = build_c(w)
    assert sec['anchor']['val_bpb']['status'] == 'pending' and any('ENV-C.node' in p for p in sec['problems'])


# ---------------------------------------------------------------- 5: R2 ENV-C needs every line; receipt identity
def test_r2_env_c_status_needs_all_three_lines_and_bound_receipts(tmp_path):
    w = World(tmp_path)
    w.c_r2_receipts()
    src = json.loads(repro.SOURCES.read_text(encoding='utf-8'))
    (w.root / 'autoresearch' / 'x1check_20261021' / 'receipts' / 'a7d6d5a710811.json').unlink()
    t = repro.r2(RC.Store(w.root), w.root, src, '', '', None)
    assert t['rows']['ENV-C']['status'] == 'pending'
    w.c_r2_receipts()
    write(w.root / 'autoresearch' / 'x1check_20261021' / 'receipts' / 'a21767e9dbd4f.json',
          w.c_receipt('elsewhere', 'a21767e9dbd4f', 'c0', arm='C-BASE', bpb=0.9963646606577251))
    t = repro.r2(RC.Store(w.root), w.root, src, '', '', None)
    assert t['rows']['ENV-C']['status'] == 'pending' and any("'run_id': 'elsewhere'" in p for p in t['problems'])


# ---------------------------------------------------------------- 7: `not run` everywhere
def test_env_a_accepts_the_string_not_run_form_and_renders_na(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    w.ledger['ENV-A']['arms']['A-RND'] = 'not run: cut by §6.4'
    sec = build_a(w)
    assert sec['labels']['A-ISW vs A-RND']['label'] == 'not run' and sec['rows']['A-RND']['status'] == 'not run'
    text = env_a.markdown(sec)
    row = next(line for line in text.splitlines() if line.startswith('| A-RND | n/a | not run'))
    assert 'pending' not in row


def test_a_sanctioned_f1_arm_cut_is_accepted_from_x6s_nonformal_reason(tmp_path):
    calls = []

    class X6:
        @staticmethod
        def main(args):
            calls.append(args)
            out = Path(args[args.index('--out') + 1])
            write(out / 'summary_F1.json', {'formal': False, 'rows': {a: {} for a in f1.ARMS if a != 'F-RND'},
                                            'nonformal_reasons': ["arms missing from the window: ['F-RND']"]})
            return 0
    w = World(tmp_path)
    ex = w.root / 'flashnext'
    write(ex / 'uc' / 'uc_ref.json', {})
    w.ledger['F1']['references'] = {'UC-REF': 'uc', 'F-HT': None}
    for a in f1.ARMS:
        if a != 'F-RND':
            write(ex / a.lower() / 'trace.json', {})
            w.ledger['F1']['arms'][a] = a.lower()
    w.ledger['F1']['arms']['F-RND'] = 'not run: cut (§6.4 order)'
    sec = f1.build(RC.Store(w.root), w.root, w.ledger, tmp_path / 'f', allow_partial=False, runner=X6)
    assert '--allow-nonformal' in calls[0]
    assert sec['cut_reason_accepted'] and not any('not formal' in p for p in sec['pending'])
    assert f1.status(sec, 'F-RND').startswith('not run') and f1.rows(sec)['F-RND']['speedup_vs_UC']['status'] == 'n/a'


# ---------------------------------------------------------------- 9: abandoned slots never used
def test_abandoned_executor_locked_block_and_node_ref_slots(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    w.a_arm('A-RND', 'arnd_x', node='WINH4', screens=[('c0', 216.0, True), ('c1', 150.0, True)], locked=None)
    w.ledger['ENV-A']['arms']['A-RND']['locked'] = 'lk_old'
    w.ledger['ENV-A']['arms']['A-RND']['abandoned'] = [{'run_id': 'lk_old', 'reason': 'x'},
                                                       {'run_id': 'nref_winh3', 'reason': 'x'},
                                                       {'run_id': 'blk_x', 'reason': 'x'}]
    sec = build_a(w)
    assert sec['rows']['A-RND']['locked']['status']['status'] == 'pending'
    assert sec['block'] is None and sec['node_refs']['WINH3']['status'] == 'pending'
    probs = ' | '.join(sec['problems'])
    assert 'lk_old' in probs and 'nref_winh3' in probs and 'blk_x' in probs


def test_f1_abandoned_reference_slot_and_cost_line(tmp_path):
    w = World(tmp_path)
    ex = w.root / 'flashnext'
    write(ex / 'uc_old' / 'uc_ref.json', {})
    write(ex / 'uc_old' / 'trace.json', {'used': {'gpu_hours': 5.0, 'llm_usd': 0.0, 'wall_seconds': 900}})
    w.ledger['F1']['references']['UC-REF'] = 'uc_old'
    w.ledger['F1']['abandoned'] = [{'run_id': 'uc_old', 'reason': 'cold cache'}]
    sec = f1.build(RC.Store(w.root), w.root, w.ledger, tmp_path / 'f', allow_partial=True)
    assert any('abandoned' in p for p in sec['problems']) and sec['abandoned'][0]['cost']['gpu_hours'] == 5.0
    assert 'Abandoned runs' in f1.markdown(sec)


# ---------------------------------------------------------------- N1: the pooled fallback can be final
def test_pooled_fallback_is_final_once_every_env_b_slot_is(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    store = RC.Store(w.root)
    sec = env_b.build(store, w.root, w.ledger, abandoned=[])
    assert sec['workloads']['W3']['margin_record']['estimator'] == 'run_to_run'
    for rid in ('bnat_W3', 'bis_W3', 'brnd_W3', 'lk_bed_W3', 'lk_bnat_W3', 'lk_bis_W3', 'lk_brnd_W3'):
        for f in (w.root / 'optico' / rid / 'receipts').glob('*.json'):
            r = json.loads(f.read_text(encoding='utf-8'))
            if r.get('candidate_id') in ('c0', 'it1', 'locked-default'):
                r.pop('harness_signature')
                r['valid'] = r['judge_valid'] = False                    # invalid defaults: n_w < 4 at G-FIN
                write(f, RC.signing.sign(r))
    sec = env_b.build(RC.Store(w.root), w.root, w.ledger, abandoned=[])
    m = sec['workloads']['W3']['margin_record']
    assert m['estimator'] == 'pooled_fallback' and not any('margin_w' in p for p in sec['pending'])


# ---------------------------------------------------------------- N2: (env, run_id) identity
def test_the_same_run_id_in_two_environments_is_no_conflict(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    w.ledger['ENV-B']['W3'] = {'abandoned': [{'run_id': 'aisw_x', 'reason': 'an ENV-B run of the same name'}]}
    assert RC.ledger_conflicts(w.ledger) == []
    assert build_a(w)['rows']['A-ISW']['status'] == 'finished'
