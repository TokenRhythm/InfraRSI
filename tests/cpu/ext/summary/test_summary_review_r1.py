"""Regression tests for the Codex review round 1 of the W5 summary (2 BLOCKER, 5 HIGH, 3 MEDIUM findings)."""
import json

import pytest
from summary_helpers import f1_arm, f1_uc_ref, LEDGER_REL, reports_fixture, World, write

from infrarsi.ext.flashnext import cli as fcli
from infrarsi.ext.summary import cells, cli, env_a, env_b, env_c, f1, repro
from infrarsi.ext.summary import records as RC

PAIRS = (('aisw_x', 'anat_x'), ('aisw_x', 'arnd_x'), ('anat_x', 'arnd_x'))


def build_a(w, g4=None):
    store = RC.Store(w.root)
    return env_a.build(store, w.root, w.ledger, abandoned=RC.abandoned(w.ledger), g4_inputs=g4), store


def build_c(w):
    store = RC.Store(w.root)
    return env_c.build(store, w.root, w.ledger, abandoned=RC.abandoned(w.ledger)), store


# ---------------------------------------------------------------- 1 BLOCKER: a partial pairwise set is never final
@pytest.mark.parametrize('pairs,final', [(PAIRS[:1], False), (PAIRS + PAIRS[:1], False), (PAIRS, True),
                                         ((('A-ISW', 'A-NAT'), ('A-ISW', 'A-RND'), ('A-NAT', 'A-RND')), True),
                                         ((('aisw_x', 'other_run'),) + PAIRS[1:], False)])
def test_g4_pairwise_set_must_be_complete(tmp_path, pairs, final):
    w = World(tmp_path)
    w.env_a_complete()
    g4 = {'final': True, 'refprime_pairwise_norms': [{'pair': list(p), 'norm_diff': 4.9, 'source': 't'} for p in pairs]}
    sec, _ = build_a(w, g4)
    cal = sec['g4_calibration']
    assert cal['final'] is final and (cal['estimator'] == 'computed') is final
    assert any("limit_D'" in p for p in sec['pending']) is (not final)
    assert sorted(map(tuple, cal['pairs_expected'])) == sorted(tuple(sorted(p)) for p in PAIRS)


def test_no_pairwise_sample_is_final_only_when_declared(tmp_path):
    # round 3 (R3-N3): with >= 2 non-cut arms the pairs are expected, so no pairwise sample is never final; with the
    # other arms cut nothing is expected and the declaration decides
    w = World(tmp_path)
    w.env_a_complete()
    assert build_a(w, {'final': True})[0]['g4_calibration']['final'] is False
    w.ledger['ENV-A']['arms']['A-NAT'] = 'not run: cut'
    w.ledger['ENV-A']['arms']['A-RND'] = 'not run: cut'
    assert build_a(w, {'final': True})[0]['g4_calibration']['final'] is True
    assert build_a(w, {})[0]['g4_calibration']['final'] is False


# ---------------------------------------------------------------- 2 BLOCKER: G-FIN means semantically complete
def _world(tmp_path, monkeypatch):
    def fake(args):
        out = args[args.index('--out') + 1]
        write(__import__('pathlib').Path(out) / 'summary_F1.json',
              {'formal': True, 'rows': {a: {'speedup_vs_UC': 1.1} for a in f1.ARMS}, 'labels': {}, 'nonformal_reasons': []})
        return 0
    monkeypatch.setattr(fcli, 'main', fake)
    w = World(tmp_path)
    w.env_a_complete()
    for wk in ('W1', 'W3', 'W2'):
        w.env_b_workload(wk)
    w.env_c_complete()
    w.c_r2_receipts()
    reports_fixture(w)
    ex = w.root / 'flashnext'
    uc = f1_uc_ref(ex / 'uc')
    write(ex / 'ht' / 'ht.json', {})
    w.ledger['F1']['references'] = {'UC-REF': 'uc', 'F-HT': 'ht'}
    for a in f1.ARMS:
        f1_arm(ex / a.lower(), a, uc)
        w.ledger['F1']['arms'][a] = a.lower()
    write(tmp_path / 'g4.json', {'final': True, 'refprime_pairwise_norms': [{'pair': list(p), 'norm_diff': 4.9, 'source': 't'}
                                                                       for p in PAIRS]})
    return w


def _run(w, out, *extra):
    w.save_ledger()
    return cli.main(['--ledger', str(w.tmp / LEDGER_REL), '--out', str(out), '--records', str(w.root), '--reports',
                     str(w.reports), '--g4-inputs', str(w.tmp / 'g4.json'), *extra])


def test_an_incomplete_label_keeps_the_summary_non_formal(tmp_path, monkeypatch, capsys):
    w = _world(tmp_path, monkeypatch)
    assert _run(w, tmp_path / 'ok') == 0
    w.c_arm('C-IS', 'cis_x', [0.9964, 0.991], accounting=False)
    assert _run(w, tmp_path / 'o') == 4
    assert 'incomplete' in capsys.readouterr().out


def test_runner_errors_missing_design_and_ledger_conflicts_are_problems(tmp_path, monkeypatch, capsys):
    w = _world(tmp_path, monkeypatch)
    (w.reports / 'design' / 'AMENDMENT_EXT_004_FABLE51.md').unlink()
    d = w.root / 'autoresearch' / 'crnd_x'
    tr = json.loads((d / 'trace.json').read_text(encoding='utf-8'))
    write(d / 'runner' / 'receipts' / 'occ.json', w.c_receipt('crnd_x', 'occ', 'c9', arm='C-RND', valid=False,
                                                              runner_error={'kind': 'occupied'}))
    tr['screens'].append({'index': 9, 'candidate_id': 'c9', 'attempt_id': 'occ', 'valid': False, 'step_time': None,
                          'cum': tr['screens'][-1]['cum']})
    write(d / 'trace.json', tr)
    w.ledger['ENV-C']['abandoned'] = [{'run_id': 'lk_crnd_x', 'reason': 'test'}]
    assert _run(w, tmp_path / 'o', '--allow-partial') == 0
    s = json.loads((tmp_path / 'o' / 'summary.json').read_text(encoding='utf-8'))
    probs = ' | '.join(s['problems'])
    assert 'design file missing: AMENDMENT_EXT_004_FABLE51.md' in probs
    assert 'refused or lost by our harness' in probs and 'abandoned run lk_crnd_x' in probs
    assert s['ENV-C']['arms']['C-RND']['locked']['status']['status'] == 'pending'
    fr = json.loads((tmp_path / 'o' / 'FREEZE.json').read_text(encoding='utf-8'))
    tools = ' '.join(t['path'].replace('\\', '/') for t in fr['tool'])
    for m in ('common/trajectory.py', 'scoring/score.py', 'optico/margin.py', 'rebench_llmf/guards.py', 'flashnext/arms.py',
              'autoresearch/knob_space_C.json'):
        assert m in tools, m


def test_a_none_value_is_an_explicit_na_cell():
    c = cells.cell(None, 'run:a')
    assert c['status'] == 'n/a' and c['why'] and c['src'] == ['run:a'] and not cells.ok(c)


# ---------------------------------------------------------------- 3 HIGH: ENV-A locked receipts bound to the selection
def test_env_a_locked_receipt_must_identify_as_the_arms_locked_rescore(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    p = w.root / 'rebench_llmf' / 'aisw_x' / 'receipts' / 'aisw_x-locked.json'
    r = json.loads(p.read_text(encoding='utf-8'))
    r.pop('harness_signature')
    r['candidate_id'] = 'block'
    write(p, RC.signing.sign(r))
    sec, _ = build_a(w)
    assert sec['rows']['A-ISW']['locked']['status']['status'] == 'pending'
    assert any('is not this arm' in x for x in sec['problems']) and sec['labels']['A-ISW vs A-NAT']['label'] == 'pending'


def test_env_a_native_identity_uses_the_snapshot_digest(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    p = w.root / 'rebench_llmf' / 'anat_x' / 'result.json'
    res = json.loads(p.read_text(encoding='utf-8'))
    res['extra']['locked']['selected']['tree_digest'] = 'dg-not-the-locked-one'
    write(p, res)
    sec, _ = build_a(w)
    lk = sec['rows']['A-NAT']['locked']
    assert lk['identity_ok']['v'] is False and lk['locked_valid_amended']['v'] is False


def test_env_a_missing_selection_evidence_is_pending_not_assumed(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    (w.root / 'rebench_llmf' / 'arnd_x' / 'receipts' / 'arnd_x-c1.json').unlink()
    sec, _ = build_a(w)
    lk = sec['rows']['A-RND']['locked']
    assert lk['identity_ok']['status'] == 'pending' and lk['locked_valid_amended']['status'] == 'pending'
    assert any('cannot be checked' in x for x in sec['problems'])


# ---------------------------------------------------------------- 4 HIGH: ENV-C anchor and locked identity
def test_env_c_anchor_must_be_the_dar7_c_base_run(tmp_path):
    w = World(tmp_path)
    w.env_c_complete()
    att, run = 'a0c0ffee00001', 'x1check_x'
    write(w.root / 'autoresearch' / run / 'receipts' / f'{att}.json',
          w.c_receipt(run, att, 'c0', arm='C-BASE', deviations=('D-AR-1', 'D-AR-2')))
    sec, _ = build_c(w)
    assert sec['anchor']['val_bpb']['status'] == 'pending' and any('not the D-AR-7 C-BASE' in p for p in sec['problems'])
    assert sec['labels']['C-IS vs C-NAT-infra']['label'] == 'pending'


def test_env_c_locked_candidate_must_train_the_final_train_py(tmp_path):
    w = World(tmp_path)
    w.env_c_complete()
    w.c_selected['C-IS'] = 'sha-something-else'
    w.c_locked('C-IS', 'lk_cis_x', 0.9965, 0.990)
    sec, _ = build_c(w)
    assert sec['arms']['C-IS']['locked']['locked_valid']['v'] is False
    assert sec['labels']['C-IS vs C-NAT-infra']['label'] == 'invalid'
    assert any('does not train the arm' in p for p in sec['problems'])


# ---------------------------------------------------------------- 5 HIGH: R2 ENV-C status from its receipts
def test_r2_env_c_status_follows_its_receipts(tmp_path):
    w = World(tmp_path)
    rd = reports_fixture(w)
    src = json.loads(repro.SOURCES.read_text(encoding='utf-8'))
    res = (rd / 'REPRO_RESULTS.md').read_text(encoding='utf-8')
    t = repro.r2(RC.Store(w.root), w.root, src, res, '', None)
    assert t['rows']['ENV-C']['status'] == 'pending' and len([p for p in t['pending'] if 'ENV-C' in p]) == 2
    w.c_r2_receipts()
    t = repro.r2(RC.Store(w.root), w.root, src, res, '', None)
    assert t['rows']['ENV-C']['status'].startswith('reproduced with D-AR-7')
    write(w.root / 'autoresearch' / 'x1check_20261021' / 'receipts' / 'a21767e9dbd4f.json',
          w.c_receipt('x1check_20261021', 'a21767e9dbd4f', 'c0', arm='C-BASE', valid=False))
    t = repro.r2(RC.Store(w.root), w.root, src, res, '', None)
    assert t['rows']['ENV-C']['status'] == 'pending' and any('line (c)' in p for p in t['problems'])


# ---------------------------------------------------------------- 7 HIGH: sanctioned `not run` vs pending
def test_sanctioned_not_run_arms_are_final_not_pending(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    w.ledger['ENV-A']['arms']['A-RND'] = {'run_id': None, 'status': 'not run (cut §6.4)'}
    sec, _ = build_a(w)
    assert sec['labels']['A-ISW vs A-RND']['label'] == 'not run' and not any('A-RND' in p for p in sec['pending'])
    w.ledger['ENV-A']['arms']['A-RND'] = {'run_id': None}
    sec, _ = build_a(w)
    assert sec['labels']['A-ISW vs A-RND']['label'] == 'pending' and any('A-RND' in p for p in sec['pending'])
    w2 = World(tmp_path / 'c')
    w2.env_c_complete()
    w2.ledger['ENV-C']['arms']['C-NAT-free'] = 'not run: $450 stop (§6.4)'
    sec, _ = build_c(w2)
    assert sec['arms']['C-NAT-free']['status'] == 'not run' and not any('C-NAT-free' in p for p in sec['pending'])


def test_f1_dropped_by_g_f1_is_not_run_everywhere(tmp_path):
    w = World(tmp_path)
    w.ledger['F1']['status'] = 'not_run (G-F1 failed: switch space unavailable)'
    sec = f1.build(RC.Store(w.root), w.root, w.ledger, tmp_path / 'f', allow_partial=True)
    assert sec['pending'] == [] and sec['not_run']
    rows = f1.rows(sec)
    assert all(rows[a]['speedup_vs_UC']['status'] == 'n/a' for a in f1.ARMS)
    r3 = repro.r3(json.loads(repro.SOURCES.read_text(encoding='utf-8')), rows, sec)
    assert all(r['status'].startswith('not_run (G-F1 failed') for r in r3 if r.get('arm') in f1.ARMS)
    assert 'not_run' in f1.markdown(sec)


# ---------------------------------------------------------------- 8 MEDIUM: C-RND space verified, coverage recomputed
def test_c_rnd_space_digest_and_recomputed_coverage(tmp_path):
    w = World(tmp_path)
    w.env_c_complete()
    sec, _ = build_c(w)
    s = sec['arms']['C-RND']['search']
    assert s['space_coverage_recomputed']['v'] == pytest.approx(2 / 47) and sec['problems'] == []
    d = w.root / 'autoresearch' / 'crnd_x'
    tr = json.loads((d / 'trace.json').read_text(encoding='utf-8'))
    tr['extra']['knob_space_sha256'] = 'f' * 64
    write(d / 'trace.json', tr)
    sec, _ = build_c(w)
    assert any('not the frozen knob_space_C.json' in p for p in sec['problems'])


# ---------------------------------------------------------------- 9 MEDIUM: abandoned ids never in a result slot
def test_an_abandoned_run_in_a_result_slot_is_a_conflict(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    w.ledger['ENV-A']['arms']['A-ISW']['abandoned'] = [{'run_id': 'aisw_x', 'reason': 'oops'}]
    assert any('aisw_x' in c for c in RC.ledger_conflicts(w.ledger))
    sec, _ = build_a(w)
    assert sec['rows']['A-ISW']['status'] == 'abandoned (ledger conflict)'
    assert sec['labels']['A-ISW vs A-NAT']['label'] == 'pending'


# ---------------------------------------------------------------- 10 MEDIUM: no key -> no numeric margin
def test_without_a_receipt_key_the_env_b_margin_is_pending(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    store = RC.Store(w.root)
    store.key_present = False
    sec = env_b.build(store, w.root, w.ledger, abandoned=[])
    W = sec['workloads']['W3']
    assert W['margin_w']['status'] == 'pending' and W['labels']['B-IS vs B-NAT']['label'] == 'pending'
