"""D-RB-20: synthetic receipts, protocol gates, numeric-only serialization."""
import json
import math
from pathlib import Path

import pytest
from summary_helpers import World, _events, signed, write

from infrarsi.harness import signing
from infrarsi.ext.summary import env_a_phase_e as E
from infrarsi.ext.rebench_llmf import phase_e_block as PB
from infrarsi.ext.rebench_llmf.task import START_TREE_DIGEST


PRIVATE = 'PRIVATE_TASK_EDIT_COMMAND_SENTINEL'
CLAUSE_I = ('G4', "score.py judged (cand(D'), ref'(D')) invalid on the permuted dataset")


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


@pytest.fixture
def inputs(tmp_path):
    w = World(tmp_path)
    w.env_a_complete(isw_t=109.0, nat_t=217.6, rnd_t=108.2,
                     isw_screen=109.0, nat_screen=214.0, rnd_screen=110.0,
                     isw_locked={'t': 109.1, 'rejections': [CLAUSE_I]})
    g4 = {'final': True, 'refprime_pairwise_norms': [
        {'pair': [a, b], 'norm_diff': 5.0} for a, b in
        [('A-NAT', 'A-ISW'), ('A-NAT', 'A-RND'), ('A-ISW', 'A-RND')]],
        'locked_dprime_norms': {'A-ISW': 5.41, 'A-NAT': 4.4, 'A-RND': 4.6}}
    gp = write(w.root / 'rebench_llmf/g4_inputs.json', g4)
    dirs = {}
    for arm, run in [('A-IS2', 'is2'), ('A-NAT2', 'nat2')]:
        res = w.a_arm(arm, run, node='WINH0', screens=[('c0', 220.0, True), ('c1', 110.0 if arm == 'A-IS2' else 219.0, True)],
                      locked={'t': 111.0} if arm == 'A-IS2' else None, ledger=False,
                      stop='strategy_done' if arm == 'A-IS2' else 'submit')
        d = w.root / 'rebench_llmf' / run
        dirs[arm] = d
        res['arm_lease'] = {'run_id': run, 'node': 'WINH0', 'gpus': [0, 1, 2, 3], 'complete': True}
        tr = res['trace']
        tr['selected_overlay'] = {'private.py': PRIVATE}
        tr['extra'] = {'private': PRIVATE}
        p = d / 'receipts' / f'{run}-c0.json'
        first = read(p)
        first['phases']['data_prep'] = 9.0
        first['candidate_digest'] = PB.candidate_digest({}, None) if arm == 'A-IS2' else START_TREE_DIGEST
        write(p, signed(first))
        if arm == 'A-IS2':
            for suffix in ('c1', 'locked'):
                rp = d / 'receipts' / f'{run}-{suffix}.json'
                rr = read(rp)
                rr['candidate_digest'] = PB.candidate_digest(tr.get('selected_config') or {}, tr['selected_overlay'])
                write(rp, signed(rr))
            _events(d, run, 2)
        else:
            p = d / 'receipts' / f'{run}-c1.json'
            selected = read(p)
            selected['candidate_digest'] = first['candidate_digest']
            write(p, signed(selected))
            lk = {'locked': None, 'skipped': 'selection_is_start', 'rule': 'D20', 'invalid_accept': 0,
                  'selected': {'candidate_id': 'c1', 'tree_digest': first['candidate_digest']}}
            # Native records use the basic-agent trace as the outer result.
            res = {**tr, 'schema': 'ext-basic-agent-trace-v1', 'arm_lease': res['arm_lease'],
                   'extra': {'locked': lk, 'private': PRIVATE}}
        write(d / 'result.json', res)
    report = tmp_path / 'FINAL_REPORT.md'
    report.write_text(PRIVATE, encoding='utf-8')
    return dict(records=w.root, ledger=w.save_ledger(), g4_inputs=gp,
                is2=dirs['A-IS2'], nat2=dirs['A-NAT2'], published_report=report)


def comparison(report, pair, reading='corrected'):
    return next(c for c in report['comparisons'] if c['pair'] == pair and c['G4_reading'] == reading)


def test_build_preserves_history_and_distinguishes_unmeasured_blocks(inputs):
    paths = [p for p in inputs['records'].rglob('*') if p.is_file()]
    before = {p: p.read_bytes() for p in paths}
    report = E.build(**inputs)
    assert report['historical']['labels'] == E.PUBLISHED
    assert report['historical']['arms']['A-RND']['published_AUC_budget'] == 0.620
    assert report['historical']['G4_corrected_limit'] > 5.41
    assert all(p.read_bytes() == data for p, data in before.items())
    assert report['arms']['A-IS2']['y_descriptive'] == pytest.approx(math.log(220 / 111))
    assert report['arms']['A-IS2']['y'] is None
    assert report['arms']['A-NAT2']['y'] == 0
    assert report['arms']['A-NAT2']['t_locked'] is None
    assert report['arms']['A-NAT2']['G4']['corrected'] == 'skipped_selection_is_start'
    assert len(report['comparisons']) == 10
    assert {c['label'] for c in report['comparisons']} == {'pending_EXT_004'}
    assert comparison(report, 'A-IS2 vs A-NAT2')['descriptive_label'] == 'advantage'
    assert comparison(report, 'A-IS2 vs A-ISW')['descriptive_label'] == 'parity'
    assert comparison(report, 'A-NAT2 vs A-NAT')['descriptive_label'] == 'parity'
    assert all(r['accounting'] == 'complete' for r in report['arms'].values())
    assert report['receipt_verification']['failed'] == 0
    assert report['blocks']['WINH0_20261024']['measured_historical_pairings'] == []
    for c in report['comparisons']:
        assert c['margin'] == pytest.approx(-math.log(.95))
        assert 'D-RB-20' in c['deviations']
        assert ('D-RB-18' in c['deviations']) == ('A-NAT2' in c['pair'])
        assert ('D-RB-19' in c['deviations']) == ('A-IS2' in c['pair'])


def test_g4_bound_is_not_ordinary_judge_norm(inputs):
    report = E.build(**inputs)
    g = report['arms']['A-IS2']['G4']
    assert g['norm_D'] == 4.4
    assert g['norm_Dprime'] is None
    assert g['norm_Dprime_upper_bound'] == 5.3
    assert g['norm_Dprime_evidence'] == 'le_predeclared_limit'
    assert g['ratio'] == 1.0
    assert g['predeclared'] == g['corrected'] == 'pass'
    assert report['arms']['A-NAT2']['InvalidAccept'] == {'predeclared': 0, 'corrected': 0}


@pytest.mark.parametrize('change,expected', [
    ('identity', 'fail'), ('missing_G4', 'fail'), ('movement', 'fail'), ('nonfinite', 'fail'),
    ('judge', 'fail'), ('hard_guard', 'fail'), ('clause_i', 'pending'), ('clause_ii', 'fail'), ('G7', 'pass')])
def test_locked_guard_failures_cannot_be_promoted(inputs, change, expected):
    p = inputs['is2'] / 'receipts/is2-locked.json'
    r = read(p)
    g = next(x for x in r['guard_reports'] if x['id'] == 'G4')
    if change == 'identity':
        r['candidate_digest'] = 'different'
    elif change == 'missing_G4':
        r['guard_reports'].remove(g)
    elif change == 'movement':
        g['cand_move'] = 1.0
    elif change == 'nonfinite':
        g['cand_move'] = float('nan')
    elif change == 'judge':
        r['judge_valid'] = False
    elif change == 'hard_guard':
        r['guard_rejections'] = [{'id': 'G6', 'detail': PRIVATE}]
    elif change == 'clause_i':
        r['guard_rejections'] = [{'id': 'G4', 'detail': CLAUSE_I[1]}]
    elif change == 'clause_ii':
        r['guard_rejections'] = [{'id': 'G4', 'detail': '||cand 1 < 14'}]
    elif change == 'G7':
        r['guard_rejections'] = [{'id': 'G7', 'detail': PRIVATE}]
    # Nonfinite data tests the reader directly; canonical JSON refuses signing it.
    sel = read(inputs['is2'] / 'receipts/is2-c1.json')
    g4 = E._g4(r, sel, 6.22571955243687)
    assert g4['corrected'] == expected
    if expected == 'fail':
        assert g4['predeclared'] == 'fail'
    assert PRIVATE not in json.dumps(g4)


@pytest.mark.parametrize('change', ['digest', 'selected', 'rule', 'attempt'])
def test_d20_requires_start_identity(inputs, change):
    p = inputs['nat2'] / 'result.json'
    d = read(p)
    lk = d['extra']['locked']
    if change == 'digest':
        lk['selected']['tree_digest'] = 'changed'
    elif change == 'selected':
        lk['selected']['candidate_id'] = 'c0'
    elif change == 'rule':
        lk['rule'] = 'other'
    else:
        lk['attempt_id'] = 'not-skipped'
    write(p, d)
    with pytest.raises(E.RecordError, match='D20'):
        E.build(**inputs)


def test_all_four_conditions_and_invalid_opponent_reference_fallback():
    a = {'arm': 'A-IS2', 'G4': {'predeclared': 'pass', 'corrected': 'pass'},
         'y_descriptive': .69, 'AUC_budget_descriptive': .62, 'accounting': 'complete'}
    b = {'arm': 'A-ISW', 'G4': {'predeclared': 'fail', 'corrected': 'pass'},
         'y_descriptive': .69, 'AUC_budget_descriptive': .67, 'accounting': 'complete'}
    pre = E._comparison(a, b, 'predeclared')
    assert pre['label'] == 'no_advantage'
    assert pre['delta_y'] == .69
    assert pre['conditions']['beyond_margin'] is True
    assert pre['conditions']['auc_not_worse'] is False
    assert E._comparison(a, b, 'corrected')['label'] == 'parity'
    b['y_descriptive'], b['AUC_budget_descriptive'] = 0, .01
    assert E._comparison(a, b, 'corrected')['label'] == 'advantage'
    a['accounting'] = 'incomplete'
    assert E._comparison(a, b, 'corrected')['label'] == 'incomplete'


def test_no_transcripts_read_and_numeric_output_projection(inputs, monkeypatch, tmp_path):
    original = Path.read_bytes
    def guarded(path):
        assert not any(part in ('calls', 'llm_calls', 'outputs', 'snapshots') for part in path.parts)
        assert not (path.name == 'events.jsonl' and path.parent == inputs['nat2'])
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', guarded)
    report = E.build(**inputs)
    out = tmp_path / 'out'
    json_path, md_path = E.write_outputs(report, out)
    assert read(json_path) == report
    for p in (json_path, md_path):
        text = p.read_text(encoding='utf-8')
        assert PRIVATE not in text
        assert 'selected_overlay' not in text
        assert 'guard_rejections' not in text
    md_path.write_bytes(md_path.read_bytes().replace(b'\n', b'\r\n'))
    E.write_outputs(report, out)
    assert b'\r\n' in md_path.read_bytes()
    assert b'\r\r\n' not in md_path.read_bytes()


def test_cli_writes_both_reports(inputs, tmp_path, capsys):
    out = tmp_path / 'cli'
    args = [item for k, v in inputs.items() for item in ('--' + k.replace('_', '-'), str(v))]
    assert E.main(args + ['--out', str(out)]) == 0
    assert (out / 'ENV-A_PHASE_E.md').is_file()
    assert (out / 'ENV-A_PHASE_E.json').is_file()
    assert PRIVATE not in capsys.readouterr().out


def test_metrics_are_recomputed_without_recalibrating_history(inputs):
    before = E.build(**inputs)
    p = inputs['is2'] / 'receipts/is2-locked.json'
    r = read(p)
    r['judge']['message']['execution_time'] = 140.0
    r['judge']['message']['norm_diff'] = 5.2
    write(p, signed(r))
    after = E.build(**inputs)
    assert after['arms']['A-IS2']['y_descriptive'] == pytest.approx(math.log(220 / 140))
    assert after['arms']['A-IS2']['y_descriptive'] != before['arms']['A-IS2']['y_descriptive']
    assert after['historical'] == before['historical']


def test_conflicting_receipt_copies_are_rejected(inputs):
    p = inputs['is2'] / 'receipts/is2-locked.json'
    r = read(p)
    r['candidate_digest'] = 'contradictory'
    write(p.with_name('locked-copy.json'), signed(r))
    with pytest.raises(E.RecordError, match='record_integrity'):
        E.build(**inputs)


def test_actual_record_regression():
    root = E.PROJECT_ROOT
    paths = {'records': E.EXECUTION_DIR / 'ext_bench',
             'ledger': root / 'reports/external_bench_20261021/results/RUN_LEDGER.json',
             'g4_inputs': E.EXECUTION_DIR / 'ext_bench/rebench_llmf/g4_inputs.json',
             'is2': root / 'rebench_private/a_is2_20261024',
             'nat2': root / 'rebench_private/a_nat2_20261024',
             'published_report': root / 'reports/external_bench_20261021/FINAL_REPORT_20261021.md'}
    if not all(p.exists() for p in paths.values()):
        pytest.skip('phase_E_records_unavailable')
    report = E.build(**paths)
    a, n = report['arms']['A-IS2'], report['arms']['A-NAT2']
    assert (a['scorings'], n['scorings']) == (38, 30)
    assert a['t_locked'] == 110.044333
    assert a['t_start_screen'] == 218.667722
    assert a['AUC_budget_descriptive'] == pytest.approx(.6240240072733231)
    assert n['AUC_budget_descriptive'] == pytest.approx(.010475665044209742)
    assert a['G4']['ratio'] == pytest.approx(1.0002598205371946)
    assert comparison(report, 'A-IS2 vs A-ISW', 'predeclared')['descriptive_label'] == 'no_advantage'
    assert comparison(report, 'A-IS2 vs A-ISW')['descriptive_label'] == 'parity'
    assert comparison(report, 'A-IS2 vs A-NAT')['descriptive_label'] == 'advantage'


@pytest.fixture
def measured_block(inputs, tmp_path):
    from infrarsi.ext.common.remote_exec import LocalNodeOps
    from infrarsi.ext.rebench_llmf.block_cli import synthetic_node_runner
    out = tmp_path / 'measured'
    PB.run(LocalNodeOps(tmp_path / 'node', node='WINH0'), is2=inputs['is2'], nat2=inputs['nat2'],
           out_dir=out, run_id='phase_e', deadline='2099-01-01T00:00:00+08:00', do_prime=False,
           node_runner=synthetic_node_runner({'S0': 222., 'A-IS2': 110.}))
    # These signed fake-node receipts exercise the measured-input path. Real
    # --local CLI outputs are explicitly synthetic and tested separately.
    return out / PB.OUTPUT


def test_measured_pair_recomputes_auc_and_keeps_cross_date_labels_pending(inputs, measured_block, tmp_path):
    before = E.build(**inputs)
    after = E.build(**inputs, block=measured_block)
    assert after['historical'] == before['historical']
    assert after['status'] == 'phase_E_pair_complete_historical_pending'
    a, n = after['arms']['A-IS2'], after['arms']['A-NAT2']
    assert a['t_B'] == 110 and n['t_B'] == 222
    assert a['y'] == pytest.approx(math.log(222 / 110)) and n['y'] == 0
    assert a['t_S0_node'] == 222 and a['node_factor'] == 1
    tr = read(inputs['is2'] / 'result.json')['trace']
    assert a['AUC_budget'] == E.TJ.auc(E.TJ.events(tr, E.A.score_task('A-IS2'), 222), E.A.score_task('A-IS2'))['auc']
    assert a['AUC_budget'] != a['AUC_budget_descriptive']
    for reading in E.READINGS:
        c = comparison(after, 'A-IS2 vs A-NAT2', reading)
        assert c['label'] == 'advantage' and c['pending_reason'] is None
        assert all(c['conditions'].values())
    for c in after['comparisons']:
        if c['pair'] != 'A-IS2 vs A-NAT2':
            assert c['label'] == 'pending_EXT_004'
            assert c['pending_reason'] == 'historical_comparator_not_in_same_EXT_004_block'
    for p in E.write_outputs(after, tmp_path / 'report'):
        assert PRIVATE not in p.read_text()
        assert 'historical_comparator_not_in_same_EXT_004_block' in p.read_text()


@pytest.mark.parametrize('source', ['selected', 'locked', 'warm', 'native_start', 'native_selected',
                                  'block_s0', 'block_candidate'])
@pytest.mark.parametrize('damage', ['unsigned', 'altered'])
def test_unverified_phase_e_receipt_keeps_formal_labels_pending(inputs, measured_block, source, damage):
    d = read(measured_block)
    paths = {'selected': inputs['is2'] / 'receipts/is2-c1.json',
             'locked': inputs['is2'] / 'receipts/is2-locked.json',
             'warm': inputs['is2'] / 'receipts/is2-c0.json',
             'native_start': inputs['nat2'] / 'receipts/nat2-c0.json',
             'native_selected': inputs['nat2'] / 'receipts/nat2-c1.json',
             'block_s0': measured_block.parent / 'receipts' / (d['runs'][0]['attempt_id'] + '.json'),
             'block_candidate': measured_block.parent / 'receipts' / (d['runs'][1]['attempt_id'] + '.json')}
    p = paths[source]
    r = read(p)
    if damage == 'unsigned':
        r.pop(signing.SIGNATURE_FIELD)
    else:
        r['wall_s'] = r.get('wall_s', 0) + 1
    write(p, r)
    report = E.build(**inputs, block=measured_block)
    for reading in E.READINGS:
        c = comparison(report, 'A-IS2 vs A-NAT2', reading)
        assert c['label'] == 'pending_EXT_004'
        assert c['pending_reason'] == 'phase_E_receipt_signature_failed'
    assert report['receipt_verification']['failed'] == 1
    descriptive = E.build(**inputs)
    assert comparison(descriptive, 'A-IS2 vs A-NAT2')['descriptive_label'] == 'advantage'


@pytest.mark.parametrize('key_state', ['no_key', 'failed'])
def test_phase_e_verification_key_required(inputs, measured_block, monkeypatch, tmp_path, key_state):
    key = tmp_path / 'other_receipt.key'
    if key_state == 'failed':
        key.write_bytes(b'\x01' * 32)
    monkeypatch.setenv(signing.KEY_ENV, str(key))
    report = E.build(**inputs, block=measured_block)
    for reading in E.READINGS:
        c = comparison(report, 'A-IS2 vs A-NAT2', reading)
        assert c['label'] == 'pending_EXT_004'
        assert c['pending_reason'] == 'phase_E_receipt_signature_' + key_state
    assert report['receipt_verification'][key_state] > 0
    assert key.exists() is (key_state == 'failed')


def test_unsigned_measurement_cannot_be_authenticated_by_matching_manifest(inputs, measured_block):
    d = read(measured_block)
    p = measured_block.parent / 'receipts' / (d['runs'][1]['attempt_id'] + '.json')
    r = read(p)
    r['judge']['message']['execution_time'] = 222.
    r.pop(signing.SIGNATURE_FIELD)
    write(p, r)
    d['runs'][1]['t'] = 222.
    v = d['values']['arms']['A-IS2']
    v['t_B'], v['y_B'], v['locked_vs_block'] = 222., 0., 111. / 222.
    d['values']['magnitude'][0].update(ratio=1., magnitude='within 5 %')
    write(measured_block, d)
    report = E.build(**inputs, block=measured_block)
    for reading in E.READINGS:
        c = comparison(report, 'A-IS2 vs A-NAT2', reading)
        assert c['label'] == 'pending_EXT_004'
        assert c['pending_reason'] == 'phase_E_receipt_signature_failed'


@pytest.mark.parametrize('change,reason', [
    ('synthetic', 'phase_E_block_not_measured'), ('running', 'phase_E_block_incomplete'),
    ('stopped', 'phase_E_block_stopped'), ('node', 'phase_E_WINH0_pairing'),
    ('binding', 'phase_E_block_selection_binding'), ('band', 'phase_E_warm_band'),
    ('values', 'phase_E_decision_values_mismatch'), ('item', 'phase_E_item_receipt_mismatch'),
    ('order', 'phase_E_attempt_order'), ('repeat', 'phase_E_attempt_order'),
    ('missing', 'receipt_missing'), ('purpose', 'phase_E_scoring_role'),
    ('G4', 'phase_E_scoring_role'), ('run_id', 'receipt_identity'),
    ('incomplete', 'phase_E_required_attempt_missing'),
    ('prepare', 'phase_E_reference_prepare_failed'),
])
def test_bad_measured_evidence_has_explicit_pending_reason(inputs, measured_block, change, reason):
    d = read(measured_block)
    if change == 'synthetic':
        d['synthetic'] = True
    elif change == 'running':
        d['status'] = 'running'
    elif change == 'stopped':
        d['stopped'] = PRIVATE
    elif change == 'prepare':
        d['prepare']['reference']['error'] = PRIVATE
    elif change == 'node':
        d['node'] = 'WINH1'
    elif change == 'binding':
        d['bindings']['A-IS2']['locked_attempt_id'] = 'wrong'
    elif change == 'band':
        d['band'][1] += 10
    elif change == 'values':
        d['values']['arms']['A-IS2']['t_B'] = 1
    elif change == 'item':
        d['runs'][1]['t'] = 1
    elif change == 'order':
        d['runs'].reverse()
    elif change == 'repeat':
        d['runs'].append(d['runs'][1])
    elif change == 'incomplete':
        d['runs'].pop()
    else:
        rp = measured_block.parent / 'receipts' / (d['runs'][1]['attempt_id'] + '.json')
        if change == 'missing':
            rp.unlink()
        else:
            r = read(rp)
            if change == 'G4':
                r['guard_reports'].append({'id': 'G4'})
            else:
                r[change] = 'wrong'
            write(rp, signed(r))
    write(measured_block, d)
    report = E.build(**inputs, block=measured_block)
    assert comparison(report, 'A-IS2 vs A-NAT2')['pending_reason'] == reason
    assert all(c['label'] == 'pending_EXT_004' for c in report['comparisons'])
    assert PRIVATE not in json.dumps(report)


def test_measured_magnitude_does_not_override_locked_g4(inputs, measured_block):
    p = inputs['is2'] / 'receipts/is2-locked.json'
    r = read(p)
    r['guard_rejections'] = [{'id': 'G4', 'detail': CLAUSE_I[1]}]
    write(p, signed(r))
    report = E.build(**inputs, block=measured_block)
    assert comparison(report, 'A-IS2 vs A-NAT2', 'predeclared')['label'] == 'invalid'
    c = comparison(report, 'A-IS2 vs A-NAT2')
    assert c['label'] == 'pending_G4'
    assert c['pending_reason'] == 'locked_G4_Dprime_evidence_missing'


def test_measured_cli(inputs, measured_block, tmp_path):
    out = tmp_path / 'cli_measured'
    args = [v for k, p in inputs.items() for v in ('--' + k.replace('_', '-'), str(p))]
    assert E.main(args + ['--block', str(measured_block), '--out', str(out)]) == 0
    assert comparison(read(out / 'ENV-A_PHASE_E.json'), 'A-IS2 vs A-NAT2')['label'] == 'advantage'


def test_measured_void_fallback_keeps_locked_validity(inputs, tmp_path):
    from infrarsi.ext.common.remote_exec import LocalNodeOps
    from infrarsi.ext.rebench_llmf.block_cli import synthetic_node_runner
    base = synthetic_node_runner({'S0': 222., 'A-IS2': 110.})
    def node(ops, job):
        r = base(ops, job)
        if job.get('mode') != 'prepare' and job['uid'] == 64105:
            r['judge']['score'] = None
            r['judge']['message']['norm_diff_too_high'] = True
        return r
    out = tmp_path / 'fallback'
    log = PB.run(LocalNodeOps(tmp_path / 'node'), is2=inputs['is2'], nat2=inputs['nat2'], out_dir=out,
                 run_id='phase_e', deadline='2099-01-01T00:00:00+08:00', do_prime=False, node_runner=node)
    assert len(log['runs']) == 3
    report = E.build(**inputs, block=out / PB.OUTPUT)
    c = comparison(report, 'A-IS2 vs A-NAT2')
    assert c['label'] == 'advantage (fallback)' and c['fallback'] is True
    assert report['arms']['A-IS2']['t_B'] == 111.
    assert report['arms']['A-IS2']['InvalidAccept']['corrected'] == 0


def test_module_entrypoint_preserves_pending_reason(inputs, tmp_path):
    import os
    import subprocess
    import sys
    out = tmp_path / 'module_cli'
    args = [v for k, p in inputs.items() for v in ('--' + k.replace('_', '-'), str(p))]
    result = subprocess.run([sys.executable, '-B', '-m', E.__name__, *args,
                             '--block', str(tmp_path / 'missing.json'), '--out', str(out)],
                            env=os.environ.copy(), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    report = read(out / 'ENV-A_PHASE_E.json')
    assert comparison(report, 'A-IS2 vs A-NAT2')['pending_reason'] == 'phase_E_block_missing'
