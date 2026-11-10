"""EXT_004 phase-E inputs and receipt-backed replay; no additional scorings."""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

from ...contracts import digest
from . import block as B
from . import knobs
from . import task
from .guards import g1_reference

ORDER = ('S0', 'A-NAT2', 'A-IS2')
DEADLINE = '2026-10-26T06:00:00+08:00'
OUTPUT = 'block_PHASE_E.json'


class RecordError(ValueError):
    """Numeric/identity evidence is absent or contradictory."""


def preflight(node='WINH0', gpus=(0, 1, 2, 3), *, remote=None):
    """Read-only SSH: no request uploads, prepare, priming, leases or GPU jobs."""
    import base64
    from ...harness.remote import default_remote
    from ..common import remote_exec as R
    want = B._local_node_hashes()
    paths = [R.node_lease_path(node)] + [R.gpu_claim_path(node, g) for g in gpus]
    # Execute the existing read-only probe helpers in memory. WinhNodeOps.call
    # may upload a request file, so use the repository transport directly here.
    source = base64.b64encode(R._NODE_SRC.encode()).decode()
    script = f'''import base64, json, os
ns = {{"__name__": "preflight"}}
exec(base64.b64decode({source!r}), ns)
result = {{"hashes": ns["hash_files"]({list(want)!r}), "probe": ns["gpu_probe"](),
          "leases": {{p: ns["read_json"](p) for p in {paths!r}}},
          "uids": [ns["uid_status"](u) for u in [64103, 64113, 64104, 64114, 64105, 64115]],
          "python_executable": os.access({(task.NODE_ENV + '/bin/python')!r}, os.X_OK)}}
print(json.dumps(result))
'''
    res = (remote or default_remote()).sh(node, "python3 -B - <<'PREFLIGHT'\n" + script + 'PREFLIGHT\n', timeout=120)
    if res.get('code') != 0:
        return {'node': node, 'ready': False, 'problems': ['read_only_probe_failed'], 'exit_code': 2}
    data = json.loads(res['stdout'])
    have = data['hashes']
    differ = [p for p in want if have.get(p) not in (None, want[p])]
    missing = [p for p in want if have.get(p) is None]
    problems = R.occupancy_problems(data['probe'], gpus, lease=data['leases'][paths[0]], lease_ok=lambda _: False)
    problems += [f'GPU claim present: {p}' for p in paths[1:] if data['leases'].get(p) is not None]
    problems += [f'uid {r["uid"]} in use' for r in data['uids'] if r['passwd'] or r['pids']]
    problems += [f'scorer mismatch: {p}' for p in differ]
    if not data['python_executable']:
        problems.append('scoring environment Python unavailable')
    return {'node': node, 'gpus': list(gpus), 'ready': not problems, 'problems': problems,
            'scorer_compatible': not differ, 'scorer_missing_install_on_run': missing,
            'python_executable': data['python_executable'], 'uids': data['uids'],
            'gpu_memory_mib': {str(g['index']): g.get('mem_used_mib') for g in data['probe']['gpus']},
            'compute_processes': [{'pid': a['pid'], 'gpu_index': a.get('gpu_index')} for a in data['probe']['apps']],
            'exit_code': 0 if not problems else 3}


def candidate_digest(config, overlay):
    cfg, errors, _ = knobs.validate(config) if config else ({}, [], [])
    if errors:
        raise B.BlockError('invalid recorded selected_config')
    ov = {k: hashlib.sha256(v.encode('utf-8', 'replace')).hexdigest() for k, v in (overlay or {}).items()}
    return digest({'knobs': cfg, 'overlay': ov})


def inputs(is2: Path, nat2: Path, *, store=None):
    # Share the summary's identity / D20 checks. Reading records does not build
    # a task, open candidate source files, or access a node.
    from ..summary import env_a_phase_e as E
    from ..summary.records import Store
    store = store or Store(is2.parent)
    selections, bindings, warm, first = {}, {}, [], {}
    for arm, directory in (('A-IS2', is2), ('A-NAT2', nat2)):
        row, tr, start = E.load_arm(store, directory, arm, E.A.MAX_NORM)
        recs = store.receipts(directory)
        selected = next(recs[s['attempt_id']] for s in tr['screens'] if s['candidate_id'] == row['selected'])
        lk = store.json(directory / 'result.json')
        lk = lk.get('locked', (lk.get('extra') or {}).get('locked', {}))
        att = lk.get('attempt_id')
        binding = {'arm': arm, 'run_id': row['run'], 'selected': row['selected'],
                   'candidate_digest': selected['candidate_digest'], 'locked_attempt_id': att,
                   'is_start': row['locked_state'] == 'D20_reference'}
        if arm == 'A-IS2':
            config, overlay = tr.get('selected_config') or {}, tr.get('selected_overlay') or None
            E._require(not binding['is_start'] and recs[att]['candidate_digest'] == binding['candidate_digest']
                       == candidate_digest(config, overlay), 'phase_E_locked_config_identity')
            selections[arm] = {'arm': arm, 'run_id': row['run'], 'selected': row['selected'],
                               'candidate_id': 'block', 'config': config, 'overlay': overlay,
                               'bind_digest': binding['candidate_digest'], 'locked_attempt_id': att,
                               't_locked': row['t_locked'], 'node': row['node']}
            E._require(start['candidate_digest'] == candidate_digest({}, None), 'phase_E_start_identity')
            samples = [start]
        else:
            E._require(binding['is_start'] and binding['candidate_digest'] == task.START_TREE_DIGEST,
                       'phase_E_D20_start_identity')
            selections[arm] = {'arm': arm, 'run_id': row['run'], 'selected': row['selected'],
                               'is_start': True, 'node': row['node'], 't_locked': None}
            samples = [recs[s['attempt_id']] for s in tr['screens']]
        bindings[arm] = binding
        first[arm] = {'arm': arm, 'node': row['node'], 'attempt_id': start['attempt_id'], 't_first': E._time(start),
                      'phases': {k: start.get('phases', {}).get(k) for k in ('data_prep', 'train', 'convert')},
                      'valid': start.get('valid'),
                      'first_cold': (start.get('phases', {}).get('data_prep') or 0) > B.COLD_DATA_PREP_S}
        for r in samples:
            dp = r.get('phases', {}).get('data_prep')
            if r.get('valid') and E.finite(dp) and dp <= B.COLD_DATA_PREP_S:
                warm.append({'arm': arm, 'attempt_id': r['attempt_id'], 't': E._time(r)})
    E._require(bool(warm), 'WINH0_warm_history_missing')
    E._require(all(p.endswith(': receipt signature does not verify') for p in store.problems), 'record_integrity')
    center = sum(x['t'] for x in warm) / len(warm)
    history = {'rule': 'EXT_004 pre-existing warm S0 mean +/- 3%', 'node': 'WINH0',
               'center': center, 'band': [center * .97, center * 1.03], 'samples': warm}
    return selections, bindings, history, first


def run(ops, *, is2, nat2, out_dir, run_id, gpus=(0, 1, 2, 3), reference_src=B.DEFAULT_REF,
        deadline=DEADLINE, do_prime=True, node_runner=None, synthetic=False, now=None):
    from ..summary.env_a_phase_e import _require, _id
    _require(ops.node == 'WINH0' and tuple(gpus) == (0, 1, 2, 3), 'phase_E_WINH0_pairing')
    _id(run_id)
    deadline = datetime.datetime.fromisoformat(deadline) if isinstance(deadline, str) else deadline
    _require(deadline is not None and deadline.utcoffset() is not None, 'phase_E_deadline_offset')
    selections, bindings, history, first = inputs(Path(is2), Path(nat2))
    out_dir = Path(out_dir)
    # A second invocation must not silently make another block or lose receipts.
    _require(not (out_dir / OUTPUT).exists() and not list((out_dir / 'receipts').glob('*.json')),
             'phase_E_output_already_used')
    metadata = {'phase': 'E', 'run_id': run_id, 'synthetic': bool(synthetic), 'bindings': bindings,
                'warm_history': history, 'reference_role': 'block_S0_is_node_reference',
                'fallback_anchor_s': B.WARM_ANCHOR_S}
    # Write the manifest before the first remote operation, including on interruption.
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUTPUT).write_text(json.dumps({**metadata, 'status': 'running'}, indent=1), encoding='utf-8')
    return B.run_block(ops, selections, out_dir=out_dir, node_refs={}, gpus=gpus,
                       reference_src=reference_src, deadline=deadline, node_runner=node_runner, now=now,
                       s0_run_id=run_id, order=ORDER, band=tuple(history['band']), output_name=OUTPUT,
                       metadata=metadata, prepare_s0=True, do_prime=do_prime, search_first=first)


def measured(store, path: Path, *, is2: Path, nat2: Path):
    """Recompute voids and decision values from receipts, never trust cached y/t_B."""
    from ..summary import env_a_phase_e as E
    selections, bindings, history, _ = inputs(is2, nat2, store=store)
    data = store.json(path)
    E._require(isinstance(data, dict), 'phase_E_block_missing')
    E._require(data.get('synthetic') is False, 'phase_E_block_not_measured')
    E._require(data.get('status') != 'running', 'phase_E_block_incomplete')
    E._require(not data.get('stopped'), 'phase_E_block_stopped')
    if 'prepare' in data:
        E._require(not g1_reference(data['prepare']).rejections, 'phase_E_reference_prepare_failed')
    E._require(data.get('kind') == 'block' and data.get('version') == 'AMENDMENT_EXT_004'
               and data.get('phase') == 'E', 'phase_E_block_protocol')
    E._require(data.get('node') == 'WINH0' and data.get('gpus') == [0, 1, 2, 3], 'phase_E_WINH0_pairing')
    E._require(data.get('order') == list(ORDER), 'phase_E_block_order')
    E._require(data.get('bindings') == bindings, 'phase_E_block_selection_binding')
    E._require(data.get('band') == history['band'] and data.get('warm_history') == history, 'phase_E_warm_band')
    E._require(data.get('reference_role') == 'block_S0_is_node_reference', 'phase_E_reference_role')
    run_id = E._id(data.get('run_id'))
    recs = store.receipts(path.parent)
    final, retry, seen = {}, [], set()
    plan = [('S0', False), ('A-IS2', False)]
    runs = data.get('runs') or []
    for i, it in enumerate(runs):
        if i == 2:
            plan.extend((name, True) for name in retry)
        E._require(i < len(plan) and (it.get('name'), it.get('rerun')) == plan[i], 'phase_E_attempt_order')
        name = it['name']
        if it.get('abandoned'):
            E._require(it['abandoned'] == 'hard stop' and i == len(runs) - 1, 'phase_E_hard_stop')
            deadline = datetime.datetime.fromisoformat(data['deadline'])
            when = datetime.datetime.fromisoformat(it['utc'])
            E._require(not B._deadline_ok(when, deadline, B.RebenchLlmfRunner.worst_case_s), 'phase_E_hard_stop_time')
            break
        att = it.get('attempt_id')
        E._require(att not in seen, 'phase_E_duplicate_attempt')
        seen.add(att)
        arm = B.REF_ARM if name == 'S0' else name
        r = E._receipt(recs, att, arm=arm, run=run_id if name == 'S0' else bindings[name]['run_id'],
                       candidate='block', node='WINH0')
        E._require(r.get('purpose') == 'score' and not any(g.get('id') in ('G4', 'G7')
                   for g in r.get('guard_reports', []) + r.get('guard_rejections', [])), 'phase_E_scoring_role')
        if name == 'S0':
            expected_digest = B._expected_digest(r, candidate_digest({}, None))
            E._require(r.get('candidate_digest') == expected_digest, 'phase_E_S0_identity')
        selection = selections.get(name, {})
        computed = B.item_record(r, name=name, band=tuple(history['band']) if name == 'S0' else None,
                                 bind_digest=selection.get('bind_digest'),
                                 config=selection.get('config'), overlay=selection.get('overlay'))
        E._require(not computed.get('fatal'), 'phase_E_containment_or_lease_retained')
        E._require(all(it.get(k) == computed[k] for k in ('t', 'void', 'rerun_allowed', 'candidate_digest')),
                   'phase_E_item_receipt_mismatch')
        if not it['rerun']:
            final[name] = computed
            if computed['void'] and computed['rerun_allowed']:
                retry.append(name)
        elif not computed['void']:
            final[name] = computed
        else:
            final[name] = {**final[name], 'void_twice': True, 'rerun_record': att}
    expected = 2 + len(retry)
    E._require(len(runs) == expected or bool(runs and runs[-1].get('abandoned')), 'phase_E_required_attempt_missing')
    E._require(data.get('block_void_reruns') == sum(bool(x.get('rerun')) and not x.get('abandoned')
                                                  for x in runs), 'phase_E_rerun_count')
    values = B._values(final, selections, {}, 'WINH0', {}, arms=ORDER[1:])
    E._require(values == data.get('values'), 'phase_E_decision_values_mismatch')
    E._require(all(p.endswith(': receipt signature does not verify') for p in store.problems), 'record_integrity')
    # Formal evidence includes the selection/locked receipts, warm history,
    # search measurements used by AUC, and every block attempt (including voids).
    for directory in (is2, nat2, path.parent):
        for r in store.receipts(directory).values():
            signature = store.signature(r)
            E._require(signature == 'verified', 'phase_E_receipt_signature_' + signature)
    return values, {'status': 'measured' if values['t_S0_B_note'] == 'measured' else 'fallback',
                    'source': E._path(path), 'node_ref': values['t_S0_B_note'],
                    'reference_role': data['reference_role'], 't_S0_B': values['t_S0_B'],
                    'band': history['band'], 'warm_center': history['center'],
                    'warm_samples': len(history['samples']), 'block_void_reruns': data['block_void_reruns'],
                    'comparators': ['A-NAT2', 'A-IS2'], 'measured_historical_pairings': []}
