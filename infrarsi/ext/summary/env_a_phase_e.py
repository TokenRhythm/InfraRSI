"""D-RB-20: read-only phase-E ENV-A integration and numeric-only reports.

The 10-21 builder remains unchanged. Its block and calibration are frozen inputs,
not recalibrated with the new arms. With no phase-E EXT_004 block/node reference,
the recorded warm start supports descriptive comparisons only. Never silently
turn that screen into a protocol node reference or a measured block.

Run as a module; --help lists source and destination arguments. No remote jobs,
agent transcripts, task files, snapshots, or candidate sources are accessed.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from ...contracts import EXECUTION_DIR, PROJECT_ROOT
from ..common import trajectory as TJ
from ..rebench_llmf.phase_e_block import RecordError
from . import env_a as A
from .cells import val
from .records import Store, accounting_verified, finite


ARMS = ('A-IS2', 'A-NAT2')
PAIRS = (('A-IS2', 'A-NAT2'), ('A-IS2', 'A-ISW'), ('A-IS2', 'A-RND'),
         ('A-IS2', 'A-NAT'), ('A-NAT2', 'A-NAT'))
READINGS = ('predeclared', 'corrected')
DEVIATIONS = {'A-IS2': ['D-RB-17', 'D-RB-19', 'D-RB-20'],
              'A-NAT2': ['D-RB-17', 'D-RB-18', 'D-RB-20']}
RULES = ['ENV-A/0.3', 'AMENDMENT_EXT_001', 'AMENDMENT_EXT_004',
         'AMENDMENT_EXT_005', 'AMENDMENT_EXT_006', 'D20', 'D27', 'G4', 'G6', 'D-RB-20']
PUBLISHED = {
    'A-ISW vs A-NAT': {'corrected': 'advantage', 'predeclared': 'invalid', 'dy': 0.692},
    'A-ISW vs A-RND': {'corrected': 'parity', 'predeclared': 'invalid', 'dy': -0.007},
    'A-NAT vs A-RND': {'corrected': 'disadvantage', 'predeclared': 'disadvantage', 'dy': -0.699},
}
PUBLISHED_ARMS = {'A-NAT': (217.61, -0.002, 0.015), 'A-ISW': (108.94, 0.690, 0.667),
                  'A-RND': (108.22, 0.696, 0.620)}


def _require(condition, rule: str) -> None:
    if not condition:
        raise RecordError(rule)


def _id(value) -> str:
    _require(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.-]+', value), 'record_identity')
    return value


def _positive(value) -> float:
    _require(finite(value) and value > 0, 'finite_positive_metric')
    return float(value)


def _time(receipt: dict) -> float:
    return _positive(A._judge_msg(receipt).get('execution_time'))


def _status(value) -> str:
    return 'pass' if value is True else 'fail' if value is False else 'pending'


def _path(path: Path) -> str:
    return path.resolve().as_posix()


def _receipt(receipts: dict, attempt, *, arm: str, run: str, candidate: str, node: str) -> dict:
    r = receipts.get(attempt)
    _require(isinstance(r, dict), 'receipt_missing')
    _require(all(r.get(k) == v for k, v in {'arm': arm, 'run_id': run, 'candidate_id': candidate,
                                          'node': node, 'env': 'rebench_llmf', 'task_id': 'ENV-A'}.items()),
             'receipt_identity')
    _require(r.get('gpus') == [0, 1, 2, 3], 'four_gpu_pairing')
    return r


def _g4(receipt: dict, selected: dict, corrected_limit: float) -> dict:
    """Clause (i) comes from the G4 verdict, never the ordinary judge D norm.

    Receipts retain movement norms but omit the exact D-prime norm. An unrejected
    clause (i) proves its predeclared bound; it also proves a larger frozen limit.
    Missing/nonfinite movement measurements cannot be treated as a G4 pass.
    """
    reports = [x for x in receipt.get('guard_reports', []) if x.get('id') == 'G4']
    rep = reports[0] if len(reports) == 1 else {}
    rejects = receipt.get('guard_rejections') or []
    clauses = {A.g4_clause(x.get('detail', '')) for x in rejects if x.get('id') == 'G4'}
    cand, ref = rep.get('cand_move'), rep.get('ref_move')
    measured = all(finite(x) and x >= 0 for x in (cand, ref))
    clause_ii = bool(measured and cand >= A.G4_RATIO * ref)
    clause_i = bool(rep and not (clauses - {'clause_ii'}))
    identity = bool(selected.get('candidate_digest') and
                    selected['candidate_digest'] == receipt.get('candidate_digest'))
    hard = any(x.get('id') not in ('G4', 'G7') for x in rejects)
    ordinary_norm = A._judge_msg(receipt).get('norm_diff')
    judge = bool(receipt.get('judge_valid') and finite(ordinary_norm) and 0 <= ordinary_norm <= A.MAX_NORM)
    valid = bool(identity and judge and not hard and not clauses and clause_i and clause_ii)
    # A failed clause (i) cannot be rescued without its exact D-prime evidence.
    corrected = valid if valid or 'clause_i' not in clauses else None
    if not identity or not judge or hard or not clause_ii or clauses - {'clause_i'}:
        corrected = False
    if corrected_limit < A.MAX_NORM:
        corrected = None if valid else corrected
    return {'norm_D': ordinary_norm if finite(ordinary_norm) else None, 'norm_Dprime': None,
            'norm_Dprime_evidence': 'le_predeclared_limit' if clause_i else 'unavailable',
            'norm_Dprime_upper_bound': A.MAX_NORM if clause_i else None,
            'cand_move': cand if finite(cand) else None, 'ref_move': ref if finite(ref) else None,
            'ratio': cand / ref if measured and ref else None,
            'clause_i': _status(clause_i), 'clause_ii': _status(clause_ii),
            'identity_G6': _status(identity), 'judge': _status(judge),
            'predeclared': _status(valid), 'corrected': _status(corrected),
            'predeclared_limit': A.MAX_NORM, 'corrected_limit': corrected_limit,
            'rule': 'G4', 'G7': 'reported_only'}


def load_arm(store: Store, directory: Path, arm: str, limit: float) -> tuple[dict, dict, dict]:
    res = store.json(directory / 'result.json')
    _require(isinstance(res, dict) and res.get('arm') == arm, 'arm_result')
    tr = res.get('trace', res)
    run = _id(tr.get('run_id'))
    lease = res.get('arm_lease') or {}
    node = _id(lease.get('node'))
    _require(node == 'WINH0' and lease.get('gpus') == [0, 1, 2, 3], 'phase_e_WINH0_pairing')
    _require(lease.get('run_id') == run and lease.get('complete') is True, 'arm_lease')
    recs = store.receipts(directory)
    screens = tr.get('screens') or []
    _require(bool(screens), 'search_screens')
    for i, s in enumerate(screens):
        r = _receipt(recs, s.get('attempt_id'), arm=arm, run=run,
                     candidate=_id(s.get('candidate_id')), node=node)
        _require(s.get('index') == i and bool(s.get('valid')) == bool(r.get('valid')), 'screen_receipt')
        if s.get('valid'):
            _require(s.get('step_time') == _time(r), 'screen_metric')
    _require(len(screens) == (tr.get('used') or {}).get('screens'), 'charged_scorings')
    selected = _id(tr.get('selected'))
    matches = [s for s in screens if s['candidate_id'] == selected]
    _require(bool(matches), 'selected_screen')
    # Rechecks may repeat a candidate id; selection identity must be stable.
    selected_recs = [recs[s['attempt_id']] for s in matches]
    _require(len({r.get('candidate_digest') for r in selected_recs}) == 1, 'selected_identity')
    sel = selected_recs[0]
    first = recs[screens[0]['attempt_id']]
    lk = res.get('locked', (res.get('extra') or {}).get('locked', {}))
    locked_attempt = lk.get('attempt_id')
    if lk.get('skipped') == 'selection_is_start':
        _require(arm == 'A-NAT2' and lk.get('rule') == 'D20' and lk.get('invalid_accept') == 0, 'D20')
        digests = {recs[s['attempt_id']].get('candidate_digest') for s in screens}
        _require(len(digests) == 1 and first.get('candidate_digest') and
                 (lk.get('selected') or {}).get('tree_digest') == first['candidate_digest'] and
                 (lk.get('selected') or {}).get('candidate_id') == selected and not locked_attempt and
                 all(r.get('valid') for r in selected_recs), 'D20_start_identity')
        g4 = {'rule': 'D20', 'predeclared': 'skipped_selection_is_start',
              'corrected': 'skipped_selection_is_start', 'identity_G6': 'pass',
              'norm_D': None, 'norm_Dprime': None, 'ratio': None,
              'predeclared_limit': A.MAX_NORM, 'corrected_limit': limit}
        locked_t = None
        lock_state = 'D20_reference'
    else:
        r = _receipt(recs, locked_attempt, arm=arm, run=run, candidate='locked', node=node)
        _require(r.get('purpose') == 'score' and lk.get('selected') == selected, 'locked_selection')
        locked_t = _time(r)
        g4 = _g4(r, sel, limit)
        lock_state = 'locked'
    ev = store.event_sequence(directory / 'events.jsonl') if arm == 'A-IS2' else None
    acc = accounting_verified(tr, recs, ev,
                              non_screen={locked_attempt: 'locked re-score'} if locked_attempt else {},
                              producer='infrarsi' if arm == 'A-IS2' else 'agent')
    good = [s['step_time'] for s in screens if s.get('valid')]
    invalid_accept = {reading: 0 if g4[reading] in ('pass', 'skipped_selection_is_start')
                      else 1 if g4[reading] == 'fail' else None for reading in READINGS}
    row = {'arm': arm, 'run': run, 'node': node, 'gpus': [0, 1, 2, 3], 'selected': selected,
           'scorings': len(screens), 'scoring_cap': A.CAPS['screens'],
           'stop': tr.get('stop_reason') if tr.get('stop_reason') in ('strategy_done', 'submit', 'finish', 'budget')
                   else 'other',
           't_start_screen': _time(first), 't_selected_search': _time(sel), 't_best_search': min(good) if good else None,
           't_locked': locked_t, 'locked_state': lock_state, 'G4': g4, 'InvalidAccept': invalid_accept,
           'accounting': 'complete' if acc[0] else 'incomplete',
           'accounting_boundary': 'verified' if arm == 'A-IS2' else 'producer_asserted',
           'deviations': DEVIATIONS[arm], 'source': _path(directory / 'result.json'),
           'locked_source': _path(directory / 'receipts' / f'{locked_attempt}.json') if locked_attempt else None,
           'block': 'missing_EXT_004', 'node_reference': 'missing_EXT_004'}
    return row, tr, first


def _comparison(a: dict, b: dict, reading: str, *, formal=False) -> dict:
    """Apply all four existing conditions; use the existing invalid-opponent fallback."""
    def compare_row(row):
        state = row['G4'][reading]
        return {'locked_valid': state in ('pass', 'skipped_selection_is_start'),
                'invalid_accept_locked': 0 if state in ('pass', 'skipped_selection_is_start') else 1,
                'y_locked': row['y'] if formal else row['y_descriptive'],
                'AUC_budget': row['AUC_budget'] if formal else row['AUC_budget_descriptive'],
                'accounting_complete': row['accounting'] == 'complete'}
    if any(r['G4'][reading] == 'pending' for r in (a, b)):
        return {'label': 'pending_G4', 'delta_y': None, 'conditions': {}}
    result = TJ.compare(A.score_task(a['arm']), compare_row(a), compare_row(b))
    return {k: result[k] for k in ('label', 'delta_y', 'conditions', 'label_if_complete') if k in result}


def build(*, records: Path, ledger: Path, g4_inputs: Path, is2: Path, nat2: Path,
          published_report: Path, block: Path | None = None) -> dict:
    """Add phase-E records without modifying the historical builder or its inputs."""
    store = Store(records)
    history_rows = A.build(store, records, store.json(ledger), abandoned=[], g4_inputs=store.json(g4_inputs))
    _require(not history_rows['problems'] and not history_rows['pending'], 'historical_ENV_A_complete')
    _require(history_rows['g4_calibration']['final'], 'EXT_005_final')
    for pair, expected in PUBLISHED.items():
        actual = history_rows['labels'][pair]
        _require(actual['label'] == expected['corrected'] and
                 actual['predeclared']['label'] == expected['predeclared'], 'published_labels_unchanged')
    limit = history_rows['g4_calibration']['limit']
    block_id = _id(history_rows['block']['run_id'])
    rows, traces, starts = {}, {}, {}
    for arm, directory in zip(ARMS, (is2, nat2)):
        rows[arm], traces[arm], starts[arm] = load_arm(store, directory, arm, limit)
    # Signature failures are reported as evidence status, not suppressed or
    # re-signed. Contradictory copies / malformed records cannot be summarized.
    _require(all(p.endswith(': receipt signature does not verify') for p in store.problems), 'record_integrity')
    # This is an explicitly descriptive proxy. EXT_004(b) requires a dedicated
    # post-arm node reference; a warm c0 is not retroactively one of those runs.
    start = starts['A-IS2']
    _require(start.get('valid') and finite((start.get('phases') or {}).get('data_prep')) and
             start['phases']['data_prep'] <= A.RB.COLD_DATA_PREP_S, 'warm_start_proxy')
    ref = _time(start)
    historical_s0 = val(history_rows['block']['t_S0_B'])
    for arm in ARMS:
        r, tr = rows[arm], traces[arm]
        t = ref if r['locked_state'] == 'D20_reference' else r['t_locked']
        r.update(t_reference_descriptive=ref, y_descriptive=math.log(ref / t),
                 t_WINH1_equivalent_descriptive=t * historical_s0 / ref,
                 node_factor_descriptive=ref / historical_s0,
                 y=0.0 if r['locked_state'] == 'D20_reference' else None,
                 AUC_budget_descriptive=TJ.auc(TJ.events(tr, A.score_task(arm), ref), A.score_task(arm))['auc'])
    historical_rows = {}
    for arm, r in history_rows['rows'].items():
        lk = r['locked']
        historical_rows[arm] = {
            'arm': arm, 'node': r['node'], 'block_node': 'WINH1',
            't_B': val(r['t_B']), 'y_B': val(r['y_B']), 'AUC_budget': val(r['AUC_budget']),
            't_locked': val(lk['t_locked']), 't_S0_node': val(r['y_ref_node']),
            'node_factor': val(history_rows['node_factor'][r['node']]),
            'G4': {'predeclared': _status(val(lk['locked_valid_predeclared'])),
                   'corrected': _status(val(lk['locked_valid_amended'])),
                   'norm_Dprime': val(lk['g4_dprime_norm']), 'ratio': val(lk['g4_ratio'])},
            'published_t_B': PUBLISHED_ARMS[arm][0], 'published_y_B': PUBLISHED_ARMS[arm][1],
            'published_AUC_budget': PUBLISHED_ARMS[arm][2]}
        rows[arm] = {**historical_rows[arm], 'y_descriptive': val(r['y_B']),
                     'AUC_budget_descriptive': val(r['AUC_budget']),
                     'accounting': 'complete' if r['search']['accounting'][0] else 'incomplete'}
    comparisons = []
    for aa, bb in PAIRS:
        a, b = rows[aa], rows[bb]
        deviations = sorted(set(DEVIATIONS.get(aa, []) + DEVIATIONS.get(bb, [])))
        for reading in READINGS:
            result = _comparison(a, b, reading)
            comparisons.append({'pair': f'{aa} vs {bb}', 'G4_reading': reading,
                                'G4_limit': A.MAX_NORM if reading == 'predeclared' else limit,
                                'dy_descriptive': a['y_descriptive'] - b['y_descriptive'],
                                'dy_rule_descriptive': result.get('delta_y'),
                                'dy_rule': result.get('delta_y'), 'margin': TJ.margin_y(A.score_task(aa)),
                                'label': 'pending_EXT_004', 'pending_reason': 'phase_E_block_missing',
                                'descriptive_label': result['label'],
                                'conditions_descriptive': result.get('conditions', {}),
                                'pairing': 'WINH0_start_proxy' if bb in ARMS else 'cross_date_node_normalized_proxy',
                                'rules': ['ENV-A/0.3', 'AMENDMENT_EXT_004', 'D-RB-20'] +
                                         (['D20'] if 'A-NAT2' in (aa, bb) else []) +
                                         (['AMENDMENT_EXT_005'] if reading == 'corrected' else []) +
                                         (['invalid_opponent_reference_fallback'] if b['G4'][reading] == 'fail' else []),
                                'deviations': deviations})
    counts = {s: sum(f.get('signature') == s for f in store.files.values()) for s in ('verified', 'failed', 'no_key')}
    report = {'schema': 'ENV-A_PHASE_E-v1', 'status': 'pending_EXT_004', 'rules': RULES,
            'deviation_labels': {'D-RB-17': 'driver_substitution', 'D-RB-18': 'mandatory_charged_S0',
                                 'D-RB-19': 'product_HLS_experience',
                                 'D-RB-20': ['phase_E_integration', 'missing_EXT_004_block', 'missing_WINH0_node_ref',
                                             'descriptive_start_proxy', 'frozen_EXT_005_20261021']},
            'historical': {'status': 'published_unchanged', 'source': _path(published_report),
                           'block': block_id, 'block_node': 'WINH1', 't_S0_B': historical_s0,
                           'G4_corrected_limit': limit, 'arms': historical_rows, 'labels': PUBLISHED},
            'arms': {arm: rows[arm] for arm in ARMS}, 'comparisons': comparisons,
            'blocks': {'WINH0_20261024': {'status': 'missing_EXT_004', 'node_ref': 'missing_EXT_004',
                                        'descriptive_reference': ref, 'reference_role': 'search_start',
                                        'reference_source': _path(is2 / 'receipts' / f'{start["attempt_id"]}.json'),
                                        'comparators': ['A-NAT2', 'A-IS2'], 'measured_historical_pairings': []},
                       'WINH1_20261021': {'status': 'published_unchanged', 'block': block_id,
                                         'comparators': ['A-NAT', 'A-ISW', 'A-RND'],
                                         'phase_E_pairing': 'descriptive_node_normalization_only'}},
            'receipt_verification': counts,
            'evidence_status': 'receipt_signatures_unverified' if counts['failed'] or counts['no_key'] else 'verified',
            'sources': sorted({_path(Path(p)) for p in store.files} | {_path(published_report)})}
    if block is not None:
        _apply_measured(report, store, block, is2=is2, nat2=nat2, traces=traces)
    report['receipt_verification'] = {s: sum(f.get('signature') == s for f in store.files.values())
                                      for s in ('verified', 'failed', 'no_key')}
    counts = report['receipt_verification']
    report['evidence_status'] = 'receipt_signatures_unverified' if counts['failed'] or counts['no_key'] else 'verified'
    report['sources'] = sorted({_path(Path(p)) for p in store.files} | {_path(published_report)})
    return report


def _apply_measured(report, store, block, *, is2, nat2, traces):
    from ..rebench_llmf.phase_e_block import measured
    try:
        values, evidence = measured(store, block, is2=is2, nat2=nat2)
    except (RecordError, OSError, ValueError, KeyError, TypeError, A.RB.BlockError) as exc:
        # RecordError messages are fixed rule identifiers, not private evidence.
        why = str(exc) if isinstance(exc, RecordError) else 'phase_E_block_malformed'
        report['blocks']['WINH0_20261024'].update(status='pending_EXT_004', pending_reason=why,
                                               source=_path(block))
        for c in report['comparisons']:
            c['pending_reason'] = why
        return
    report['blocks']['WINH0_20261024'] = evidence
    report['deviation_labels']['D-RB-20'] = ['phase_E_integration', 'measured_EXT_004_block',
                                           'block_S0_is_WINH0_node_ref', 'frozen_EXT_005_20261021',
                                           'historical_pairings_not_in_phase_E_block']
    ref = values['node_refs']['WINH0']
    for arm, row in report['arms'].items():
        v = values['arms'][arm]
        row.update(t_B=v['t_B'], y=v.get('y_B'), y_B=v.get('y_B'), t_S0_node=ref,
                   t_S0_B=values['t_S0_B'], node_factor=ref / values['t_S0_B'],
                   locked_vs_block=v.get('locked_vs_block'), fallback=v.get('fallback', False),
                   block_attempt=v.get('block_attempt'), block='EXT_004',
                   node_reference=evidence['node_ref'],
                   AUC_budget=TJ.auc(TJ.events(traces[arm], A.score_task(arm), ref), A.score_task(arm))['auc'])
    for c in report['comparisons']:
        aa, bb = c['pair'].split(' vs ')
        if bb not in ARMS:
            c.update(pending_reason='historical_comparator_not_in_same_EXT_004_block',
                     pairing='cross_date_descriptive_only')
            continue
        a, b = report['arms'][aa], report['arms'][bb]
        if any(r['y'] is None for r in (a, b)):
            c['pending_reason'] = 'phase_E_decision_value_unavailable'
            continue
        formal = _comparison(a, b, c['G4_reading'], formal=True)
        fallback = any(r['fallback'] for r in (a, b)) or evidence['node_ref'] != 'measured'
        label = formal['label']
        c.update(label=label + (' (fallback)' if fallback and label not in ('invalid', 'pending_G4') else ''),
                 label_base=label, fallback=fallback, dy=formal.get('delta_y'), dy_rule=formal.get('delta_y'),
                 conditions=formal.get('conditions', {}), pairing='WINH0_same_EXT_004_block',
                 pending_reason='locked_G4_Dprime_evidence_missing' if label == 'pending_G4' else
                                'search_accounting_incomplete' if label == 'incomplete' else None)
        if 'label_if_complete' in formal:
            c['label_if_complete'] = formal['label_if_complete']
    own = [c for c in report['comparisons'] if c['pair'] == 'A-IS2 vs A-NAT2']
    report['status'] = ('phase_E_pair_complete_historical_pending' if all(c['pending_reason'] is None for c in own)
                        else 'phase_E_pair_pending')


def markdown(report: dict) -> str:
    """Only fixed labels, rule names, numeric values and source paths are rendered."""
    def table(headers, rows):
        def value(v):
            return 'NA' if v is None else f'{v:.9f}' if isinstance(v, float) else str(v)
        return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + [
            '| ' + ' | '.join(value(v) for v in row) + ' |' for row in rows]
    lines = ['# ENV-A_PHASE_E', '', report['status'], '', report['evidence_status'], '',
             ' / '.join(report['rules']), '']
    lines += table(['arm', 'node', 'scorings', 'selected', 't_start_screen', 't_locked', 'y', 'y_descriptive',
                    'AUC_budget_descriptive', 'deviations'],
                   [[a, r['node'], f"{r['scorings']}/{r['scoring_cap']}", r['selected'], r['t_start_screen'],
                     r['t_locked'], r['y'], r['y_descriptive'], r['AUC_budget_descriptive'],
                     ', '.join(r['deviations'])] for a, r in report['arms'].items()])
    lines += ['', 'EXT_004 / formal decision values', ''] + table(
        ['arm', 't_B', 't_S0_B', 't_S0_node', 'y_B', 'AUC_budget', 'fallback'],
        [[a] + [r.get(k) for k in ('t_B', 't_S0_B', 't_S0_node', 'y_B', 'AUC_budget', 'fallback')]
         for a, r in report['arms'].items()])
    lines += ['', 'ENV-A/0.3 / formal labels', ''] + table(
        ['pair', 'G4_reading', 'label', 'dy', 'pending_reason'],
        [[c.get(k) for k in ('pair', 'G4_reading', 'label', 'dy', 'pending_reason')] for c in report['comparisons']])
    lines += ['', 'ENV-A/0.3 / formal conditions', ''] + table(
        ['pair', 'G4_reading', 'valid_locked', 'invalid_accept_zero', 'beyond_margin', 'auc_not_worse'],
        [[c['pair'], c['G4_reading']] + [_status(c.get('conditions', {}).get(k)) for k in
          ('valid_locked', 'invalid_accept_zero', 'beyond_margin', 'auc_not_worse')] for c in report['comparisons']])
    lines += ['', 'G4', ''] + table(
        ['arm', 'norm_D', 'norm_Dprime', 'norm_Dprime_upper_bound', 'cand_move', 'ref_move', 'ratio',
         'predeclared', 'corrected', 'rule'],
        [[a, g.get('norm_D'), g.get('norm_Dprime'), g.get('norm_Dprime_upper_bound'),
          g.get('cand_move'), g.get('ref_move'), g.get('ratio'),
          g['predeclared'], g['corrected'], g['rule']] for a, r in report['arms'].items() for g in [r['G4']]])
    lines += ['', 'ENV-A/0.3 / D-RB-20', ''] + table(
        ['pair', 'dy_descriptive', 'dy_rule', 'margin', 'label', 'descriptive_label', 'G4_reading', 'G4_limit', 'deviations'],
        [[c[k] for k in ('pair', 'dy_descriptive', 'dy_rule', 'margin', 'label', 'descriptive_label', 'G4_reading', 'G4_limit')]
         + [', '.join(c['deviations'])] for c in report['comparisons']])
    lines += ['', 'ENV-A/0.3 / conditions_descriptive', ''] + table(
        ['pair', 'G4_reading', 'valid_locked', 'invalid_accept_zero', 'beyond_margin', 'auc_not_worse', 'deviations'],
        [[c['pair'], c['G4_reading']] + [_status(c['conditions_descriptive'].get(k)) for k in
          ('valid_locked', 'invalid_accept_zero', 'beyond_margin', 'auc_not_worse')] + [', '.join(c['deviations'])]
         for c in report['comparisons']])
    lines += ['', 'AMENDMENT_EXT_004', ''] + table(['block', 'status', 'pairing', 'comparators'], [
        [b, r['status'], r.get('reference_role', r.get('phase_E_pairing', 'WINH0_start_proxy')), ', '.join(r['comparators'])]
        for b, r in report['blocks'].items()])
    lines += ['', 'AMENDMENT_EXT_004 / descriptive_node_normalization_only', ''] + table(
        ['arm', 't_reference_descriptive', 't_WINH1_equivalent_descriptive', 'node_factor_descriptive'],
        [[a, r['t_reference_descriptive'], r['t_WINH1_equivalent_descriptive'], r['node_factor_descriptive']]
         for a, r in report['arms'].items()])
    lines += ['', '20261021 / published_unchanged', ''] + table(
        ['arm', 't_B_published', 'y_B_published', 'AUC_budget_published'],
        [[a, r['published_t_B'], r['published_y_B'], r['published_AUC_budget']]
         for a, r in report['historical']['arms'].items()])
    lines += ['', '20261021 / published_unchanged', ''] + table(
        ['pair', 'dy_published', 'corrected', 'predeclared'],
        [[p, r['dy'], r['corrected'], r['predeclared']] for p, r in report['historical']['labels'].items()])
    lines += ['', 'D-RB-17 / D-RB-18 / D-RB-19 / D-RB-20', '']
    for d, labels in report['deviation_labels'].items():
        lines += [d + ' / ' + (' / '.join(labels) if isinstance(labels, list) else labels)]
    lines += ['', 'receipt_verification', '', json.dumps(report['receipt_verification'], sort_keys=True), '', 'sources', '']
    lines += [f'- {p}' for p in report['sources']]
    return '\n'.join(lines) + '\n'


def write_outputs(report: dict, out: Path) -> tuple[Path, Path]:
    # Serialization is an explicit projection in build(), never a copy of a
    # result, trace, guard detail, model response, candidate edit or command.
    payloads = {'ENV-A_PHASE_E.json': json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False) + '\n',
                'ENV-A_PHASE_E.md': markdown(report)}
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, content in payloads.items():
        path = out / name
        newline = '\r\n' if path.exists() and b'\r\n' in path.read_bytes() else '\n'
        path.write_bytes(content.replace('\n', newline).encode('utf-8'))
        paths.append(path)
    return tuple(paths)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', type=Path, default=EXECUTION_DIR / 'ext_bench')
    p.add_argument('--ledger', type=Path, default=PROJECT_ROOT / 'reports/external_bench_20261021/results/RUN_LEDGER.json')
    p.add_argument('--g4-inputs', type=Path)
    p.add_argument('--is2', type=Path, default=PROJECT_ROOT / 'rebench_private/a_is2_20261024')
    p.add_argument('--nat2', type=Path, default=PROJECT_ROOT / 'rebench_private/a_nat2_20261024')
    p.add_argument('--published-report', type=Path,
                   default=PROJECT_ROOT / 'reports/external_bench_20261021/FINAL_REPORT_20261021.md')
    p.add_argument('--block', type=Path, help='phase-e block_PHASE_E.json; its S0 is also the WINH0 node reference')
    p.add_argument('--out', type=Path, default=PROJECT_ROOT / 'reports/infrarsi_textbook_goal_20261024/E')
    args = p.parse_args(argv)
    report = build(records=args.records, ledger=args.ledger,
                   g4_inputs=args.g4_inputs or args.records / 'rebench_llmf/g4_inputs.json',
                   is2=args.is2, nat2=args.nat2, published_report=args.published_report, block=args.block)
    for path in write_outputs(report, args.out):
        print(path.resolve().as_posix())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
