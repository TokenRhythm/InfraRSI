"""Regression tests for the Codex review round 3 of the W5 summary (R3-N3 .. R3-N7) and the 10-21 afternoon additions:
D20 skipped locked pairs, D23 same-node y, validator-normalized ENV-B identity, D24 pre-launch refusals, the ENV-B
raw-ratio / paired sensitivity, the trials column and glob patterns among abandoned runs."""
import json
import math

import pytest
from summary_helpers import World, write
from test_summary_cli import fake_x6, run, world

from infrarsi.ext.flashnext import cli as fcli
from infrarsi.ext.optico import task as OT
from infrarsi.ext.optico import workloads as OW
from infrarsi.ext.summary import cli, env_a, env_b, env_c, f1
from infrarsi.ext.summary import records as RC

PAIRS3 = (('aisw_x', 'anat_x'), ('aisw_x', 'arnd_x'), ('anat_x', 'arnd_x'))
PRE = {'kind': 'runner_error', 'type': 'AttemptDeadline', 'message': 'pre_launch deadline passed before launch (3 s late)'}


def build_a(w, g4=None):
    return env_a.build(RC.Store(w.root), w.root, w.ledger, abandoned=RC.abandoned(w.ledger, w.root), g4_inputs=g4)


def build_b(w):
    return env_b.build(RC.Store(w.root), w.root, w.ledger, abandoned=RC.abandoned(w.ledger, w.root))


def build_c(w):
    return env_c.build(RC.Store(w.root), w.root, w.ledger, abandoned=RC.abandoned(w.ledger, w.root))


def resign(path, **changes):
    r = json.loads(path.read_text(encoding='utf-8'))
    r.pop('harness_signature', None)
    r.update(changes)
    write(path, RC.signing.sign(r))


# ---------------------------------------------------------------- R3-N3: zero-pair G4
def test_zero_pair_g4_with_two_or_more_arms_is_never_final(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    cal = build_a(w, {'final': True})['g4_calibration']
    assert cal['final'] is False and cal['limit'] == pytest.approx(6.17)
    assert sorted(map(tuple, cal['pairs_expected'])) == sorted(tuple(sorted(p)) for p in PAIRS3)
    assert 'missing' in cal['estimator'] and 'aisw_x' in cal['estimator']
    w.ledger['ENV-A']['arms']['A-RND'] = 'not run: cut'                    # two arms left: one pair still expected
    sec = build_a(w, {'final': True})
    assert sec['g4_calibration']['final'] is False and sec['g4_calibration']['pairs_expected'] == [['aisw_x', 'anat_x']]
    assert any("limit_D'" in p for p in sec['pending'])


def test_zero_pair_g4_cannot_reach_formal(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fcli, 'main', fake_x6([]))
    w = world(tmp_path)
    write(tmp_path / 'g4.json', {'final': True})
    assert run(w, tmp_path / 'out') == 4
    assert "limit_D' 6.170 not final" in capsys.readouterr().out


# ---------------------------------------------------------------- R3-N4: sanctioned cuts never manufacture pending
def test_a_cut_env_a_llm_arm_has_no_pending_cell(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    w.ledger['ENV-A']['arms']['A-NAT'] = 'not run: cut (§6.4)'
    sec = build_a(w, {'final': True, 'refprime_pairwise_norms': [{'pair': ['aisw_x', 'arnd_x'], 'norm_diff': 4.9,
                                                                   'source': 't'}]})
    assert sec['rows']['A-NAT']['solo_wall_fraction']['status'] == 'n/a'
    assert sec['g4_calibration']['final'] is True and sec['pending'] == []
    assert sec['labels']['A-ISW vs A-NAT']['label'] == 'not run'
    assert cli.pending_cells({'rows': sec['rows'], 'labels': sec['labels']}) == []
    assert '| pending |' not in env_a.markdown(sec)


def test_not_run_env_b_references_are_n_a(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    w.ledger['ENV-B']['W3']['B-DEF'] = 'not run: cut'
    w.ledger['ENV-B']['W3']['B-ED'] = 'not run: cut'
    sec = build_b(w)
    W = sec['workloads']['W3']
    assert W['arms']['B-DEF']['T']['status'] == 'n/a' and W['arms']['B-ED']['T']['status'] == 'n/a'
    rows = [x for x in env_b.markdown(sec).splitlines() if x.startswith(('| B-DEF |', '| B-ED |'))]
    assert len(rows) == 2 and not any('pending' in x for x in rows)


def test_not_run_c_nat_free_narrative_is_n_a(tmp_path):
    w = World(tmp_path)
    w.env_c_complete()
    w.ledger['ENV-C']['arms']['C-NAT-free'] = 'not run: cut'
    w.ledger['ENV-C']['locked'].pop('C-NAT-free')
    sec = build_c(w)
    para = next(x for x in env_c.markdown(sec).splitlines() if x.startswith('Status not run'))
    assert 'pending' not in para and 'n/a' in para
    assert not any('C-NAT-free' in p for p in sec['pending'])


def test_a_stray_pending_cell_blocks_a_formal_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fcli, 'main', fake_x6([]))
    w = world(tmp_path)
    real = env_a.solo_fractions
    monkeypatch.setattr(env_a, 'solo_fractions',
                        lambda arms, iv: {**real(arms, iv), 'A-NAT': {'v': None, 'status': 'pending', 'why': 'injected'}})
    assert run(w, tmp_path / 'out') == 4
    assert 'pending value(s) under an otherwise formal summary' in capsys.readouterr().out


# ---------------------------------------------------------------- R3-N5: abandoned ENV-C anchor / calibration
def test_abandoned_env_c_anchor_and_calibration_are_never_used(tmp_path):
    w = World(tmp_path)
    w.env_c_complete()
    w.ledger['ENV-C']['abandoned'] = [{'run_id': 'x1check_x', 'reason': 'x'}, {'run_id': 'calib_x', 'reason': 'y'}]
    sec = build_c(w)
    assert sec['anchor']['val_bpb']['status'] == 'pending'
    assert sec['calibration']['gate']['status'] == 'pending' and 'c_base_steps' not in sec['calibration']
    probs = ' | '.join(sec['problems'])
    assert 'x1check_x' in probs and 'calib_x' in probs
    assert sec['labels']['C-IS vs C-NAT-infra']['label'] == 'pending'
    conf = ' | '.join(RC.ledger_conflicts(w.ledger))
    assert 'ENV-C anchor' in conf and 'ENV-C calibration' in conf
    assert 'Calibration calib_x' not in env_c.markdown(sec)


# ---------------------------------------------------------------- R3-N6: the pooled-fallback note follows the gate
def _invalid_defaults(w):
    for rid in ('bnat_W3', 'bis_W3', 'brnd_W3', 'lk_bed_W3', 'lk_bnat_W3', 'lk_bis_W3', 'lk_brnd_W3'):
        for f in (w.root / 'optico' / rid / 'receipts').glob('*.json'):
            r = json.loads(f.read_text(encoding='utf-8'))
            if r.get('candidate_id') in ('c0', 'it1', 'locked-default'):
                resign(f, valid=False, judge_valid=False)


def test_pooled_fallback_note_follows_the_gate(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    _invalid_defaults(w)
    sec = build_b(w)
    m = sec['workloads']['W3']
    assert m['margin_record']['estimator'] == 'pooled_fallback'
    assert m['margin_w']['note'] == 'pooled_fallback (final pooled fallback under AMENDMENT_EXT_002)'
    assert m['labels']['B-IS vs B-NAT']['margin_final'] is True
    w.ledger['ENV-B']['W3']['locked'].pop('B-RND')                        # one slot no longer final
    sec = build_b(w)
    m = sec['workloads']['W3']
    assert m['margin_w']['note'] == 'pooled_fallback (provisional until every ENV-B slot is final)'
    assert m['labels']['B-IS vs B-NAT']['margin_final'] is False
    assert '(margin_w provisional)' in env_b.markdown(sec)


# ---------------------------------------------------------------- R3-N7: F1 global not-run keeps the abandoned cost line
def test_f1_global_not_run_keeps_abandoned_costs(tmp_path):
    w = World(tmp_path)
    write(w.root / 'flashnext' / 'uc_old' / 'trace.json', {'used': {'gpu_hours': 5.0, 'llm_usd': 0.0, 'wall_seconds': 900}})
    w.ledger['F1']['status'] = 'not_run (G-F1 failed: test)'
    w.ledger['F1']['abandoned'] = [{'run_id': 'uc_old', 'reason': 'cold cache'}]
    sec = f1.build(RC.Store(w.root), w.root, w.ledger, tmp_path / 'f', allow_partial=True)
    assert sec['not_run'] and sec['abandoned'][0]['cost']['gpu_hours'] == 5.0
    text = f1.markdown(sec)
    assert 'no accepted F1 result runs' in text and 'Abandoned runs' in text and 'no GPU spent' not in text


# ---------------------------------------------------------------- D20: a skipped locked pair (selection = the start)
def test_d20_skipped_env_b_locked_pair(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    wl = OW.get('W3')
    assert w.b_search('W3', 'B-RND', 'brnd_W3', [228.5, None, 200.0]) == dict(wl.defaults)   # the start stays best
    d = w.root / 'optico' / 'lk_brnd_W3'
    for f in (d / 'receipts').glob('*.json'):
        f.unlink()
    write(d / 'locked.json', {'arm': 'B-RND', 'skipped': 'selection_is_default', 'y': 0.0, 'speedup_vs_default': 1.0,
                              'invalid_accept_locked': 0, 'default': None, 'candidate': None, 'order': []})
    sec = build_b(w)
    W = sec['workloads']['W3']
    lk, T = W['arms']['B-RND']['locked'], W['T_def']['v']
    assert lk['skipped'] and lk['tflops_locked']['v'] == T and lk['invalid_accept_locked']['v'] == 0
    assert W['labels']['B-IS vs B-RND']['delta_y'] == pytest.approx(math.log(330.0 / T))
    assert not any('B-RND locked' in p for p in sec['pending'])
    w.b_search('W3', 'B-RND', 'brnd_W3', [228.5, None, 300.0])           # now the selection is not the default
    sec = build_b(w)
    assert sec['workloads']['W3']['arms']['B-RND']['locked']['status']['status'] == 'pending'
    assert any('D20 not confirmed' in p for p in sec['problems'])


def test_d20_skipped_env_c_locked_pair(tmp_path):
    w = World(tmp_path)
    w.env_c_complete()
    w.c_arm('C-RND', 'crnd_x', [0.9960, 0.9990, None], failures=('', '', 'oom'),
            extra={'space_coverage': 0.0426, 'space_size': 47})              # c0 (the baseline) stays best
    d = w.root / 'autoresearch' / 'lk_crnd_x'
    for f in (d / 'runner' / 'receipts').glob('*.json'):
        f.unlink()
    skip = {'skipped': 'selection_is_baseline', 'y': 0.0, 'invalid_accept_locked': 0, 'baseline': None, 'candidate': None,
            'order': None, 'selection_sha256': 'sha-base-train', 'arm': 'C-RND'}
    write(d / 'result.json', skip)
    sec = build_c(w)
    lk = sec['arms']['C-RND']['locked']
    assert lk['skipped'] and lk['val_bpb_locked']['v'] == pytest.approx(0.9963646606577251)
    assert lk['invalid_accept_locked']['v'] == 0 and sec['labels']['C-IS vs C-RND']['label'] != 'pending'
    write(d / 'result.json', {**skip, 'selection_sha256': 'sha-other'})
    sec = build_c(w)
    assert sec['arms']['C-RND']['locked']['status']['status'] == 'pending'
    assert any('D20 not confirmed' in p for p in sec['problems'])


# ---------------------------------------------------------------- D23, normalized identity, sensitivity
def test_d23_cross_node_arm_uses_a_same_node_ratio(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W1')
    p2 = {**OW.get('W1').defaults, 'MICRO_BATCH_SIZE': 2}
    sel = w.b_search('W1', 'B-ISW', 'bisw_W1', [240.0, None, 330.0], best_params=p2)
    w.b_locked('W1', 'B-ISW', 'lk_bisw_W1', 240.0, 336.0, params=sel)
    for rid in ('bisw_W1', 'lk_bisw_W1'):
        for f in (w.root / 'optico' / rid / 'receipts').glob('*.json'):
            resign(f, node='WINH1')
    sec = build_b(w)
    isw = sec['workloads']['W1']['arms']['B-ISW']
    assert isw['node'] == 'WINH1' and isw['y_ref']['v'] == 240.0 and 'D23' in isw['y_ref_basis']
    assert isw['node_factor']['v'] == pytest.approx(240.0 / 228.0)
    lab = sec['workloads']['W1']['labels']['B-ISW vs B-NAT']
    assert lab['delta_y'] == pytest.approx(math.log(336.0 / 240.0) - math.log(300.0 / 228.0))
    assert lab['raw_ratio'] == pytest.approx(336.0 / 300.0) and lab['label_if_raw'] is not None
    assert 'B-ISW ran on WINH1' in env_b.markdown(sec)


def test_identity_uses_the_validator_normalization(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    sel = {**OW.get('W3').defaults, 'MICRO_BATCH_SIZE': 2, 'AC': 'none', 'AC_LAYERS': 8}   # AC_LAYERS inert under AC none
    p = w.root / 'optico' / 'bnat_W3' / 'trace.json'
    tr = json.loads(p.read_text(encoding='utf-8'))
    tr['selected_config'] = sel
    write(p, tr)
    w.b_locked('W3', 'B-NAT', 'lk_bnat_W3', 229.0, 300.0, params={**sel, 'AC_LAYERS': 1})
    sec = build_b(w)
    assert not any('final selection' in x for x in sec['problems'])
    assert sec['workloads']['W3']['arms']['B-NAT']['locked']['locked_valid']['v'] is True
    w.b_locked('W3', 'B-NAT', 'lk_bnat_W3', 229.0, 300.0, params={**sel, 'MICRO_BATCH_SIZE': 4})
    assert any('final selection' in x for x in build_b(w)['problems'])


def test_paired_reading_flags_a_label_the_default_drift_would_change(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    p2 = {**OW.get('W3').defaults, 'MICRO_BATCH_SIZE': 2}
    w.b_locked('W3', 'B-IS', 'lk_bis_W3', 220.0, 310.0, params=p2)
    w.b_locked('W3', 'B-NAT', 'lk_bnat_W3', 236.0, 305.0, params=p2)
    lab = build_b(w)['workloads']['W3']['labels']['B-IS vs B-NAT']
    assert lab['raw_ratio'] == pytest.approx(310.0 / 305.0) and lab['label'] == 'parity' == lab['label_if_raw']
    assert lab['paired_ratio'] == pytest.approx((310.0 / 220.0) / (305.0 / 236.0))
    assert lab['label_if_paired'] in ('advantage', 'no_advantage') and lab['sensitivity_flag']


# ---------------------------------------------------------------- trials used / budget (stop reason)
def test_trials_used_column_in_every_environment(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    w.env_b_workload('W3')
    w.env_c_complete()
    a, b, c = build_a(w), build_b(w), build_c(w)
    assert a['rows']['A-ISW']['search']['trials_used']['v'] == '2 / 60 (finish)'
    cap = OT.make_task('W3', 'B-IS').budget['screens']
    assert b['workloads']['W3']['arms']['B-IS']['search']['trials_used']['v'] == f'3 / {cap} (budget:screens)'
    assert c['arms']['C-IS']['search']['trials_used']['v'] == '2 / 36 (budget:wall_seconds)'
    for text in (env_a.markdown(a), env_b.markdown(b), env_c.markdown(c)):
        assert 'trials used / budget (stop reason)' in text


# ---------------------------------------------------------------- D24: pre-launch refusals
def _refuse(w, stop):
    d = w.root / 'optico' / 'bis_W3'
    tr = json.loads((d / 'trace.json').read_text(encoding='utf-8'))
    att = 'bis_W3-c9'
    write(d / 'receipts' / f'{att}.json', w.b_receipt('W3', 'bis_W3', att, 'c9', arm='B-IS', purpose='screen', T=1.0,
                                                      valid=False, runner_error=PRE))
    tr['screens'] = [s for s in tr['screens'] if s['candidate_id'] != 'c9']
    tr['screens'].append({'index': 3, 'candidate_id': 'c9', 'attempt_id': att, 'valid': False, 'step_time': None,
                          'cum': tr['screens'][-1]['cum']})
    tr['stop_reason'] = stop
    write(d / 'trace.json', tr)


def test_d24_prelaunch_refusals_are_accepted_unless_the_screen_budget_bound(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    _refuse(w, 'finish')
    b = build_b(w)
    assert b['workloads']['W3']['arms']['B-IS']['search']['n_runner_errors']['v'] == {'prelaunch': 1}
    a, c = {'rows': {}}, {'arms': {}}
    assert cli.runner_error_problems(a, b, c) == []
    assert any('prelaunch' in f and 'D24' in f for f in b['findings'])
    _refuse(w, 'budget:screens')
    assert any('D24 does not cover' in p for p in cli.runner_error_problems(a, build_b(w), c))


def test_post_1405_prelaunch_refusal_form_is_a_harness_refusal_not_an_invalid_proposal(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    _refuse(w, 'finish')
    f = w.root / 'optico' / 'bis_W3' / 'receipts' / 'bis_W3-c9.json'
    reasons = ['infra_refusal: the pre-launch phase (180 s: lease, upload, launch) ended before trial.sh was launched',
               'runner_error: AttemptDeadline: x']
    resign(f, not_launched='pre_launch', invalid_reasons=reasons)
    s = build_b(w)['workloads']['W3']['arms']['B-IS']['search']
    assert s['n_runner_errors']['v'] == {'prelaunch': 1} and s['n_invalid_prelaunch']['v'] == 0
    resign(f, not_launched='pre_launch', invalid_reasons=reasons, runner_error=None)       # without a runner_error dict
    s = build_b(w)['workloads']['W3']['arms']['B-IS']['search']
    assert s['n_runner_errors']['v'] == {'prelaunch': 1} and s['n_invalid_prelaunch']['v'] == 0
    resign(f, not_launched='invalid_proposal', invalid_reasons=['invalid_proposal: x'], runner_error=None)
    s = build_b(w)['workloads']['W3']['arms']['B-IS']['search']
    assert s['n_runner_errors']['v'] == {} and s['n_invalid_prelaunch']['v'] == 1


# ---------------------------------------------------------------- glob patterns among abandoned runs
def test_glob_abandoned_runs_expand_to_run_dirs_and_conflicts(tmp_path):
    w = World(tmp_path)
    for rid in ('envc_cis_d', 'envc_crnd_d', 'envc_lock_crnd_d', 'envc_cis_e'):
        write(w.root / 'autoresearch' / rid / 'plan.json', {})
    w.ledger['ENV-C']['abandoned'] = [{'run_id': 'envc_*_d', 'reason': '#31'}]
    ab = RC.abandoned(w.ledger, w.root)
    assert sorted(x['run_id'] for x in ab) == ['envc_cis_d', 'envc_crnd_d', 'envc_lock_crnd_d']
    assert all(x['pattern'] == 'envc_*_d' for x in ab)
    assert [x['run_id'] for x in RC.abandoned(w.ledger)] == ['envc_*_d']            # no root: the pattern itself
    w.ledger['ENV-C']['arms']['C-IS'] = 'envc_cis_d'
    assert any('envc_cis_d' in x for x in RC.ledger_conflicts(w.ledger))           # fnmatch without a root
    assert any('envc_cis_d' in x for x in RC.ledger_conflicts(w.ledger, w.root))
