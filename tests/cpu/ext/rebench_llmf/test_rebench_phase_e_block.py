"""One phase-E block: exact identities, state checks, retry/fallback and preflight."""
import datetime
import json
from pathlib import Path

import pytest

from infrarsi.harness import signing
from infrarsi.ext.common.env_runner import candidate_digest as runner_candidate_digest
from infrarsi.ext.common.remote_exec import LocalNodeOps
from infrarsi.ext.rebench_llmf import block, block_cli, phase_e_block as E, task
from infrarsi.ext.summary.records import Store


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def records(tmp_path):
    dirs = {}
    for arm, run in [('A-IS2', 'is2'), ('A-NAT2', 'nat2')]:
        d = tmp_path / run
        dirs[run] = d
        cfg = {'skip_valid_copy': True} if arm == 'A-IS2' else {}
        selected = E.candidate_digest(cfg, None) if arm == 'A-IS2' else task.START_TREE_DIGEST
        screens = []
        for i, cid in enumerate(('c0', 'c1', 'locked') if arm == 'A-IS2' else ('c0', 'c1')):
            t = 220. if i == 0 or arm == 'A-NAT2' else 110.
            r = {'schema': 'ext-receipt-v1', 'env': 'rebench_llmf', 'task_id': 'ENV-A', 'arm': arm,
                 'run_id': run, 'candidate_id': cid, 'attempt_id': f'{run}-{cid}', 'node': 'WINH0',
                 'gpus': [0, 1, 2, 3], 'purpose': 'score', 'valid': True, 'judge_valid': True,
                 'judge': {'message': {'execution_time': t, 'norm_diff': 4.4}},
                 'candidate_digest': E.candidate_digest({}, None) if i == 0 and arm == 'A-IS2' else selected,
                 'phases': {'data_prep': 9.}, 'guard_rejections': [],
                 'guard_reports': [{'id': 'G4', 'cand_move': 20., 'ref_move': 20.}] if cid == 'locked' else [],
                 'cost': {'gpu_seconds': 100., 'wall_seconds': 25., 'accounting_complete': True}}
            signing.sign(r)
            write(d / 'receipts' / (r['attempt_id'] + '.json'), r)
            if cid != 'locked':
                screens.append({'index': i, 'candidate_id': cid, 'attempt_id': r['attempt_id'],
                                'valid': True, 'step_time': t, 'cum': {}})
        tr = {'arm': arm, 'run_id': run, 'selected': 'c1', 'selected_config': cfg,
              'screens': screens, 'used': {'screens': 2}, 'budget': {}, 'stop_reason': 'submit'}
        lease = {'run_id': run, 'node': 'WINH0', 'gpus': [0, 1, 2, 3], 'complete': True}
        if arm == 'A-IS2':
            res = {'arm': arm, 'trace': tr, 'arm_lease': lease,
                   'locked': {'attempt_id': 'is2-locked', 'selected': 'c1'}}
        else:
            res = {**tr, 'arm_lease': lease, 'extra': {'locked': {
                'skipped': 'selection_is_start', 'rule': 'D20', 'invalid_accept': 0,
                'selected': {'candidate_id': 'c1', 'tree_digest': selected}}}}
        write(d / 'result.json', res)
    return dirs


class Node:
    def __init__(self, plan=None, *, prepare_error=None):
        self.plan = {k: list(v) for k, v in (plan or {}).items()}
        self.prepare_error = prepare_error
        self.jobs = []
        self.base = block_cli.synthetic_node_runner({'S0': 220., 'A-IS2': 110.})

    def __call__(self, ops, job):
        r = self.base(ops, job)
        if job.get('mode') == 'prepare':
            if self.prepare_error:
                r['reference']['error'] = self.prepare_error
            return r
        self.jobs.append(job)
        name = 'S0' if job['uid'] == 64103 else 'A-IS2'
        spec = self.plan[name].pop(0) if self.plan.get(name) else {}
        if spec.get('exception'):
            raise RuntimeError('transient node failure')
        r['phases']['data_prep'] = spec.get('dp', 9.)
        if spec.get('invalid'):
            r['judge']['score'] = None
            r['judge']['message']['norm_diff_too_high'] = True
            r['judge']['message']['norm_diff'] = 6.
        if spec.get('guard'):
            r['frozen_after']['score.py'] = 'wrong'
        if spec.get('reference_error'):
            r['reference']['error'] = spec['reference_error']
            r['judge'] = {'message': {'execution_time': r['phases']['total'], 'model_not_found': True}}
        if 't' in spec:
            r['judge']['message']['execution_time'] = spec['t']
        return r


def run(records, tmp_path, plan=None, **kw):
    node = Node(plan)
    out = tmp_path / 'out'
    ops = LocalNodeOps(tmp_path / 'node', node='WINH0')
    log = E.run(ops, **records, out_dir=out, run_id='phase_e', do_prime=False, node_runner=node,
                deadline=kw.pop('deadline', '2099-01-01T00:00:00+08:00'), **kw)
    return log, node, ops, out


def replay(records, out):
    return E.measured(Store(out), out / E.OUTPUT, **records)[0]


def test_one_block_uses_recorded_arms_and_shared_s0(records, tmp_path):
    log, node, _, out = run(records, tmp_path)
    assert log['order'] == ['S0', 'A-NAT2', 'A-IS2']
    assert [r['name'] for r in log['runs']] == ['S0', 'A-IS2']
    assert [j['uid'] for j in node.jobs] == [64103, 64105]
    assert [j['run_id'] for j in node.jobs] == ['phase_e', 'is2']
    assert all(j['purpose'] == 'score' and not j['run_g4'] and j['clean_slate'] for j in node.jobs)
    assert log['values']['arms']['A-NAT2']['t_B'] == log['values']['t_S0_B'] == 220.
    assert replay(records, out) == log['values']
    with pytest.raises(ValueError, match='output_already_used'):
        E.run(LocalNodeOps(tmp_path / 'other'), **records, out_dir=out, run_id='phase_e')


@pytest.mark.parametrize('spec', [{'dp': 24.}, {'invalid': True}, {'t': 240.}])
def test_void_s0_retries_once_after_candidate(records, tmp_path, spec):
    log, node, _, out = run(records, tmp_path, {'S0': [spec, {'t': 219.}]})
    assert [r['name'] for r in log['runs']] == ['S0', 'A-IS2', 'S0']
    assert log['block_void_reruns'] == 1 and len(node.jobs) == 3
    assert replay(records, out)['t_S0_B'] == 219.


def test_s0_runner_exception_retries_with_canonical_scored_identity(records, tmp_path):
    log, node, _, out = run(records, tmp_path, {'S0': [{'exception': True}, {'t': 219.}]})
    assert [r['name'] for r in log['runs']] == ['S0', 'A-IS2', 'S0']
    assert log['block_void_reruns'] == 1 and len(node.jobs) == 3
    failed, _, retry = log['runs']
    recs = Store(out).receipts(out)
    failure = recs[failed['attempt_id']]
    assert failure['runner_error']['type'] == 'RuntimeError'
    assert failure['judge'] is None and not failure['valid']
    assert signing.verify(failure)
    assert failed['candidate_digest'] == runner_candidate_digest({}, None)
    assert failed['void'] and failed['rerun_allowed'] and failed['t'] is None
    assert retry['candidate_digest'] == E.candidate_digest({}, None)
    assert retry['rerun'] and not retry['void']
    values = replay(records, out)
    assert values == log['values']
    assert values['t_S0_B'] == values['arms']['A-NAT2']['t_B'] == 219.
    assert values['t_S0_B_note'] == 'measured'

    # A successful retry must still identify the canonical ENV-A start.
    scored = recs[retry['attempt_id']]
    scored['candidate_digest'] = retry['candidate_digest'] = runner_candidate_digest({}, None)
    write(out / 'receipts' / (scored['attempt_id'] + '.json'), signing.sign(scored))
    write(out / E.OUTPUT, log)
    with pytest.raises(E.RecordError, match='phase_E_S0_identity'):
        replay(records, out)


@pytest.mark.parametrize('overlay', [None, {'scripts/retry_probe.py': '# regression fixture\n'}])
def test_is2_runner_exception_retries_with_locked_scored_identity(records, tmp_path, overlay):
    result = json.loads((records['is2'] / 'result.json').read_text())
    config = result['trace']['selected_config']
    result['trace']['selected_overlay'] = overlay
    write(records['is2'] / 'result.json', result)
    locked_digest = E.candidate_digest(config, overlay)
    for cid in ('c1', 'locked'):
        p = records['is2'] / 'receipts' / f'is2-{cid}.json'
        r = json.loads(p.read_text())
        r['candidate_digest'] = locked_digest
        write(p, signing.sign(r))

    log, node, _, out = run(records, tmp_path, {'A-IS2': [{'exception': True}, {'t': 105.}]})
    assert [r['name'] for r in log['runs']] == ['S0', 'A-IS2', 'A-IS2']
    assert log['block_void_reruns'] == 1 and len(node.jobs) == 3
    _, failed, retry = log['runs']
    recs = Store(out).receipts(out)
    failure = recs[failed['attempt_id']]
    assert failure['runner_error']['type'] == 'RuntimeError'
    assert failure['judge'] is None and not failure['valid'] and not failure['judge_valid']
    assert signing.verify(failure)
    failure_digest = runner_candidate_digest(config, overlay)
    assert failed['candidate_digest'] == failure_digest != locked_digest
    assert failed['void'] and failed['rerun_allowed'] and failed['t'] is None
    assert 'identity' not in failed['guard_rejections']
    assert retry['candidate_digest'] == locked_digest and retry['rerun'] and not retry['void']
    values = replay(records, out)
    assert values == log['values']
    assert values['arms']['A-IS2']['t_B'] == 105.
    assert values['arms']['A-IS2']['source'] == 'block' and not values['arms']['A-IS2']['fallback']

    wrong_failure = {**failure, 'candidate_digest': runner_candidate_digest({}, None)}
    rejected = block.item_record(wrong_failure, name='A-IS2', bind_digest=locked_digest,
                                 config=config, overlay=overlay)
    assert rejected['void'] and not rejected['rerun_allowed'] and 'identity' in rejected['guard_rejections']

    # A scored retry cannot use the generic failure digest, even with a matching manifest.
    scored = recs[retry['attempt_id']]
    scored['candidate_digest'] = retry['candidate_digest'] = failure_digest
    rejected = block.item_record(scored, name='A-IS2', bind_digest=locked_digest,
                                 config=config, overlay=overlay)
    assert rejected['void'] and not rejected['rerun_allowed'] and 'identity' in rejected['guard_rejections']
    write(out / 'receipts' / (scored['attempt_id'] + '.json'), signing.sign(scored))
    write(out / E.OUTPUT, log)
    with pytest.raises(E.RecordError, match='phase_E_item_receipt_mismatch'):
        replay(records, out)


def test_both_voids_retry_in_original_order_then_fallback(records, tmp_path):
    bad = {'invalid': True}
    log, node, _, out = run(records, tmp_path, {'S0': [bad, bad], 'A-IS2': [bad, bad]})
    assert [r['name'] for r in log['runs']] == ['S0', 'A-IS2', 'S0', 'A-IS2']
    assert len(node.jobs) == 4
    v = replay(records, out)
    assert v['t_S0_B'] == 216.1 and v['arms']['A-IS2']['t_B'] == 110.
    assert v['arms']['A-IS2']['label'] == '(fallback)'


def test_guard_rejection_is_not_retried(records, tmp_path):
    log, node, _, out = run(records, tmp_path, {'A-IS2': [{'guard': True}]})
    assert len(node.jobs) == 2 and log['block_void_reruns'] == 0
    assert replay(records, out)['arms']['A-IS2']['fallback']


REFERENCE_ERROR = 'reference source C:/Program Files/Git/data/reference has no model.safetensors'


@pytest.mark.parametrize('name', ['S0', 'A-IS2'])
def test_judge_invalid_with_g1_is_not_retried_in_execution_or_replay(records, tmp_path, name):
    # EXT_004(c): G1 remains a no-retry veto even when the judge also failed.
    log, node, _, out = run(records, tmp_path, {name: [{'reference_error': REFERENCE_ERROR}]})
    failed = next(r for r in log['runs'] if r['name'] == name)
    assert failed['guard_rejections'] == ['G1'] and not failed['judge_valid']
    assert failed['void'] and not failed['rerun_allowed']
    assert [r['name'] for r in log['runs']] == ['S0', 'A-IS2']
    assert log['block_void_reruns'] == 0 and len(node.jobs) == 2
    assert replay(records, out) == log['values']


def test_failed_reference_prepare_stops_before_priming_or_scoring(records, tmp_path, monkeypatch):
    node = Node(prepare_error=REFERENCE_ERROR)
    monkeypatch.setattr(block, 'prime', lambda *_: pytest.fail('primed after failed preparation'))
    out = tmp_path / 'out'
    log = E.run(LocalNodeOps(tmp_path / 'node'), **records, out_dir=out, run_id='phase_e',
                node_runner=node, deadline='2099-01-01T00:00:00+08:00')
    assert 'S0 reference preparation failed' in log['stopped']
    assert log['prepare']['reference']['error'] == REFERENCE_ERROR
    assert not log['runs'] and not node.jobs and log['block_void_reruns'] == 0
    assert not list((out / 'receipts').glob('*.json'))
    assert json.loads((out / E.OUTPUT).read_text()) == log
    with pytest.raises(E.RecordError, match='phase_E_block_stopped'):
        replay(records, out)


def test_replay_rejects_completed_block_with_failed_reference_prepare(records, tmp_path):
    # Reproduce the old manifest: prepare failed, S0 had G1 + model_not_found,
    # A-IS2 succeeded, no retry, and the block was written as completed.
    log, _, _, out = run(records, tmp_path, {'S0': [{'reference_error': REFERENCE_ERROR}]})
    log['prepare']['reference']['error'] = REFERENCE_ERROR
    write(out / E.OUTPUT, log)
    with pytest.raises(E.RecordError, match='phase_E_reference_prepare_failed'):
        replay(records, out)


def test_locked_identity_change_fails_before_any_remote_call(records, tmp_path):
    p = records['is2'] / 'result.json'
    r = json.loads(p.read_text())
    r['trace']['selected_config'] = {}
    write(p, r)
    ops = LocalNodeOps(tmp_path / 'node')
    with pytest.raises(ValueError, match='locked_config_identity'):
        E.run(ops, **records, out_dir=tmp_path / 'out', run_id='phase_e')
    assert not ops.calls


def test_d20_identity_change_is_not_scored(records, tmp_path):
    p = records['nat2'] / 'result.json'
    r = json.loads(p.read_text())
    r['extra']['locked']['selected']['tree_digest'] = 'changed'
    write(p, r)
    with pytest.raises(ValueError, match='D20'):
        run(records, tmp_path)


def test_deadline_abandons_without_scoring_and_replays_fallback(records, tmp_path):
    log, node, _, out = run(records, tmp_path, deadline='2000-01-01T00:00:00+08:00')
    assert not node.jobs and log['runs'][0]['abandoned'] == 'hard stop'
    assert replay(records, out)['arms']['A-IS2']['fallback']


def test_uid_in_use_stops_before_scoring(records, tmp_path):
    ops = LocalNodeOps(tmp_path / 'node')
    ops.uid_survivors[64103] = [123]
    node = Node()
    log = E.run(ops, **records, out_dir=tmp_path / 'out', run_id='phase_e',
                node_runner=node, deadline='2099-01-01T00:00:00+08:00', do_prime=False)
    assert log.get('stopped') and not node.jobs
    with pytest.raises(ValueError, match='block_stopped'):
        replay(records, tmp_path / 'out')


def test_cli_dry_run_persists_synthetic_and_new_deadline(records, tmp_path):
    out = tmp_path / 'cli'
    args = ['phase-e', '--local', '--no-prime', '--out', str(out), '--run-id', 'phase_e']
    for k, p in records.items():
        args += ['--' + k, str(p)]
    parsed = block_cli.build_parser().parse_args(args)
    assert parsed.deadline == E.DEADLINE
    assert block_cli.main(args + ['--deadline', '2099-01-01T00:00:00+08:00']) == 0
    assert json.loads((out / E.OUTPUT).read_text())['synthetic'] is True
    with pytest.raises(ValueError, match='not_measured'):
        replay(records, out)


@pytest.mark.parametrize('problem', [None, 'scorer', 'gpu', 'claim', 'uid', 'python'])
def test_preflight_is_read_only_and_reports_admission(problem):
    from infrarsi.ext.common import remote_exec as R
    want = block._local_node_hashes()
    data = {'hashes': dict(want), 'probe': {'error': None, 'apps': [],
            'gpus': [{'index': g, 'mem_used_mib': 0} for g in range(8)]},
            'leases': {R.node_lease_path('WINH0'): None},
            'uids': [{'uid': 64103, 'passwd': None, 'pids': []}], 'python_executable': True}
    if problem == 'scorer':
        data['hashes'][next(iter(want))] = 'wrong'
    elif problem == 'gpu':
        data['probe']['gpus'][0]['mem_used_mib'] = 80000
    elif problem == 'claim':
        data['leases'][R.gpu_claim_path('WINH0', 0)] = {'attempt_id': 'busy'}
    elif problem == 'uid':
        data['uids'][0]['pids'] = [123]
    elif problem == 'python':
        data['python_executable'] = False

    class Remote:
        def sh(self, node, command, timeout):
            assert node == 'WINH0' and command.startswith("python3 -B - <<'PREFLIGHT'")
            assert 'ns["hash_files"]' in command and 'ns["gpu_probe"]' in command
            assert 'ns["acquire"]' not in command and 'ns["write_files"]' not in command
            return {'code': 0, 'stdout': json.dumps(data)}
    result = E.preflight(remote=Remote())
    assert result['ready'] is (problem is None)
    assert result['exit_code'] == (0 if problem is None else 3)
