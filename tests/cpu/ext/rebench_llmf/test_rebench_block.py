"""AMENDMENT_EXT_004 post-arm steps (block.py / block_cli.py): void classification, node_ref, the WINH1 block (order,
identity binding, re-runs at the end, fallbacks, hard stop, fail closed), selections from the arm records, the
checked node runner that never overwrites the shipped scorer. CPU only, fake node runners."""
import datetime
import json
import math
import tempfile
from pathlib import Path

import pytest

from infrarsi.ext.common.remote_exec import LocalNodeOps
from infrarsi.ext.rebench_llmf import block, block_cli
from infrarsi.ext.rebench_llmf import knobs as knobmod
from infrarsi.ext.rebench_llmf import task as taskmod
from infrarsi.ext.rebench_llmf.runner import RebenchLlmfRunner

import rebench_helpers as H

ARM_OF = {64101: 'A-NAT', 64102: 'A-ISW', 64103: 'A-RND'}
ISW_CFG = {'device_train_microbatch_size': 16, 'fsdp_config.sharding_strategy': 'NO_SHARD', 'skip_valid_copy': True}
RND_CFG = {'device_train_microbatch_size': 8, 'train_loader.num_workers': 8}


def _ops():
    return LocalNodeOps(tempfile.mkdtemp(prefix='isblock'), node='WINH1')


def _cdig(arm, config):
    r = RebenchLlmfRunner(_ops(), task=H.task(arm), node_runner=lambda o, j: None)
    return r._candidate_digest(knobmod.validate(config)[0] if config else {}, None)


class BlockNode:
    """Fake node: per item name ('S0', 'A-NAT', 'A-ISW', 'A-RND') a queue of outcomes {'t', 'dp', 'valid', 'g2bad',
    'kept'}; records every job."""

    def __init__(self, plan):
        self.plan = {k: list(v) for k, v in plan.items()}
        self.jobs = []

    def name_of(self, job):
        arm = ARM_OF[job['uid']]
        if (arm == 'A-RND' and not job.get('restore_snapshot') and job['finetune_sh'] == job['finetune_sh_base']
                and job['finetune_yaml'] == job['finetune_yaml_base']):
            return 'S0'
        return arm

    def __call__(self, ops, job):
        if job.get('mode') == 'prepare':
            return {'mode': 'prepare', 'reference': {'outside_agent_tree': True, 'mode': '700', 'owner': 0},
                    'base': {'ok': True}, 'ref_vs_base': 23.78}
        self.jobs.append(job)
        name = self.name_of(job)
        spec = self.plan[name].pop(0) if self.plan.get(name) else {'t': 200.0}
        if spec.get('exception'):
            raise RuntimeError('transient node failure')
        arm = ARM_OF[job['uid']]
        t = spec.get('t', 200.0)
        node = H.node_result(H.task(arm), run_id=job['run_id'], execution_time=t, valid_judge=spec.get('valid', True),
                             phases={'data_prep': spec.get('dp', 8.9), 'train': t - 35.5, 'convert': 26.6, 'total': t},
                             tree_digest=job.get('expected_digest') or ('7' * 64))
        if spec.get('g2bad'):
            node['frozen_after'] = {**node['frozen_after'], 'score.py': 'z' * 64}
        return node


def _sels(nat_node='WINH1'):
    return {'A-NAT': {'arm': 'A-NAT', 'run_id': 'anat_a', 'candidate_id': 'block', 'config': {},
                      'restore_snapshot': '/s/abc.tar', 'expected_digest': 'a' * 64, 'bind_digest': 'a' * 64,
                      't_locked': 150.0, 'node': nat_node, 'selected': 'score07'},
            'A-ISW': {'arm': 'A-ISW', 'run_id': 'aisw_a', 'candidate_id': 'block', 'config': dict(ISW_CFG), 'overlay': None,
                      'bind_digest': _cdig('A-ISW', ISW_CFG), 't_locked': 109.0, 'node': 'WINH3', 'selected': 'c2'},
            'A-RND': {'arm': 'A-RND', 'run_id': 'arnd_a', 'candidate_id': 'block', 'config': dict(RND_CFG), 'overlay': None,
                      'bind_digest': _cdig('A-RND', RND_CFG), 't_locked': 170.0, 'node': 'WINH4', 'selected': 'c9'}}


REFS = {'WINH3': 220.0, 'WINH4': 225.0}


def _block(plan, *, deadline=None, now=None, refs=REFS, sels=None):
    node = BlockNode(plan)
    out = Path(tempfile.mkdtemp(prefix='isblockout'))
    log = block.run_block(_ops(), sels or _sels(), out_dir=out, node_refs=refs, node_runner=node, deadline=deadline, now=now)
    return log, node, out


# ---------------------------------------------------------------- void classification
def _rec(**kw):
    t = kw.pop('t', 215.0)
    node = H.node_result(H.task('A-RND'), run_id='r', execution_time=t,
                         phases={'data_prep': kw.pop('dp', 8.9), 'train': 180.0, 'convert': 26.0, 'total': t}, **kw)
    r, tk = H.runner(node, arm='A-RND', enforce_g7=False)
    return r.run(tk, {}, None, 'score', 'r', run_g4=False)


def test_item_classification():
    ok = block.item_record(_rec(), name='S0', band=block.BAND)
    assert not ok['void'] and ok['t'] == 215.0 and not ok['first_cold']
    out_of_band = block.item_record(_rec(t=230.0), name='S0', band=block.BAND)
    assert out_of_band['void'] and out_of_band['rerun_allowed'] and 'band' in out_of_band['void_reasons'][0]
    cold = block.item_record(_rec(dp=23.8), name='S0', band=block.BAND)
    assert cold['first_cold'] and cold['void'] and cold['rerun_allowed']
    bad = block.item_record(_rec(valid_judge=False, trained=False), name='A-RND')
    assert bad['void'] and bad['rerun_allowed'] and bad['void_reasons'][0].startswith('judge_invalid')
    g2 = H.node_result(H.task('A-RND'), run_id='r')
    g2['frozen_after'] = {**g2['frozen_after'], 'score.py': 'z' * 64}
    r, tk = H.runner(g2, arm='A-RND', enforce_g7=False)
    guard = block.item_record(r.run(tk, {}, None, 'score', 'r', run_g4=False), name='A-RND')
    assert guard['void'] and not guard['rerun_allowed'] and 'block_guard_rejection' in guard['void_reasons'][0]
    ident = block.item_record(_rec(), name='A-RND', bind_digest='f' * 64)
    assert ident['void'] and not ident['rerun_allowed'] and 'identity' in ident['guard_rejections']
    r2, tk2 = H.runner(H.node_result(H.task('A-RND'), run_id='r'), arm='A-RND', enforce_g7=False)
    r2.ops.uid_survivors[64103] = [5]
    fatal = block.item_record(r2.run(tk2, {}, None, 'score', 'r', run_g4=False), name='S0', band=block.BAND)
    assert fatal.get('fatal')


# ---------------------------------------------------------------- node_ref
def test_node_ref_measures_one_s0_through_the_a_rnd_settings():
    node = BlockNode({'S0': [{'t': 219.5}]})
    out = Path(tempfile.mkdtemp(prefix='isnref'))
    ops = LocalNodeOps(tempfile.mkdtemp(prefix='isnrefops'), node='WINH3')
    rec = block.node_ref(ops, run_id='nref_winh3_a', out_dir=out, node_runner=node)
    assert rec['t_S0_node'] == 219.5 and rec['measured'] and rec['uids'] == [64103, 64113] and rec['runner_arm'] == 'A-RND'
    job = node.jobs[0]
    assert job['purpose'] == 'score' and job['run_g4'] is False and job['clean_slate'] is True and job['uid'] == 64103
    assert job['finetune_sh'] == job['finetune_sh_base']                     # the unchanged start
    assert ops.uid_kills.count(64103) >= 2 and json.loads((out / 'node_ref_WINH3.json').read_text())['t_S0_node'] == 219.5


def test_node_ref_void_twice_falls_back_and_void_once_asks_for_the_rerun():
    ops = LocalNodeOps(tempfile.mkdtemp(prefix='isnref2'), node='WINH4')
    rec = block.node_ref(ops, run_id='n', out_dir=tempfile.mkdtemp(), node_runner=BlockNode({'S0': [{'dp': 24.0}, {'valid': False}]}))
    assert rec['t_S0_node'] == 216.1 and rec['measured'] is False and len(rec['attempts']) == 2
    once = block.node_ref(LocalNodeOps(tempfile.mkdtemp(), node='WINH4'), run_id='n', out_dir=tempfile.mkdtemp(),
                          node_runner=BlockNode({'S0': [{'dp': 24.0}]}), max_attempts=1)
    assert once['t_S0_node'] is None and 're-run' in once['note']
    retried = block.node_ref(LocalNodeOps(tempfile.mkdtemp(), node='WINH4'), run_id='n', out_dir=tempfile.mkdtemp(),
                             node_runner=BlockNode({'S0': [{'dp': 24.0}, {'t': 221.0}]}))
    assert retried['t_S0_node'] == 221.0 and retried['measured'] and retried['attempts'][0]['first_cold']


def test_node_ref_priming_reads_the_listed_trees():
    ops = LocalNodeOps(tempfile.mkdtemp(prefix='isnref3'), node='WINH4')
    rec = block.node_ref(ops, run_id='n', out_dir=tempfile.mkdtemp(), node_runner=BlockNode({}), do_prime=True)
    cmds = [c[1]['cmd'] for c in ops.calls if c[0] == 'sh']
    assert any('cat > /dev/null' in c and taskmod.NODE_MODEL in c and taskmod.NODE_FOUNDRY in c for c in cmds)
    assert rec['prime']['dirs'][0] == taskmod.NODE_ENV


# ---------------------------------------------------------------- the block
def test_block_runs_the_fixed_order_with_each_arms_runner_and_locked_item():
    log, node, out = _block({'S0': [{'t': 215.0}], 'A-NAT': [{'t': 140.0}], 'A-ISW': [{'t': 100.0}], 'A-RND': [{'t': 160.0}]})
    assert [r['name'] for r in log['runs']] == ['S0', 'A-NAT', 'A-ISW', 'A-RND']
    assert [j['uid'] for j in node.jobs] == [64103, 64101, 64102, 64103] and all(j['purpose'] == 'score' for j in node.jobs)
    assert all(j['run_g4'] is False and j['clean_slate'] for j in node.jobs)
    nat = node.jobs[1]
    assert nat['restore_snapshot'] == '/s/abc.tar' and nat['expected_digest'] == 'a' * 64 and nat['run_id'] == 'anat_a'
    assert node.jobs[0]['run_id'] == 'arnd_a' and node.jobs[2]['run_id'] == 'aisw_a'
    v = log['values']
    assert v['t_S0_B'] == 215.0 and v['arms']['A-ISW']['t_B'] == 100.0 and v['arms']['A-ISW']['label'] == ''
    assert abs(v['arms']['A-NAT']['y_B'] - math.log(215.0 / 140.0)) < 1e-12
    assert abs(v['node_factor']['WINH3'] - 220.0 / 215.0) < 1e-12
    assert v['magnitude'][0]['magnitude'] == 'A-ISW faster by >= 5 %'
    assert json.loads((out / 'block_A.json').read_text())['values']['t_S0_B'] == 215.0


def test_void_items_are_rerun_once_at_the_end_in_the_original_order():
    plan = {'S0': [{'dp': 23.9}, {'t': 216.0}], 'A-NAT': [{'t': 140.0}], 'A-ISW': [{'valid': False}, {'t': 101.0}],
            'A-RND': [{'t': 160.0}]}
    log, node, _ = _block(plan)
    assert [(r['name'], r['rerun']) for r in log['runs']] == [('S0', False), ('A-NAT', False), ('A-ISW', False),
                                                              ('A-RND', False), ('S0', True), ('A-ISW', True)]
    v = log['values']
    assert v['t_S0_B'] == 216.0 and v['arms']['A-ISW']['t_B'] == 101.0 and v['arms']['A-ISW']['source'] == 'block'


@pytest.mark.parametrize('arm', ['A-NAT', 'A-ISW', 'A-RND'])
def test_selected_runner_exception_retries_at_end_with_locked_identity(arm):
    sels = _sels()
    plan = {'S0': [{'t': 215.}], arm: [{'exception': True}, {'t': 101.}]}
    log, node, out = _block(plan, sels=sels)
    assert [r['name'] for r in log['runs']] == [*block.BLOCK_ORDER, arm]
    assert log['block_void_reruns'] == 1 and len(node.jobs) == 5
    failed, retry = [r for r in log['runs'] if r['name'] == arm]
    receipt = json.loads((out / 'receipts' / (failed['attempt_id'] + '.json')).read_text())
    assert receipt['runner_error']['type'] == 'RuntimeError' and receipt['judge'] is None
    assert failed['void'] and failed['rerun_allowed'] and failed['t'] is None
    assert 'identity' not in failed['guard_rejections']
    assert retry['rerun'] and not retry['void'] and retry['candidate_digest'] == sels[arm]['bind_digest']
    assert log['values']['arms'][arm]['t_B'] == 101.
    assert log['values']['arms'][arm]['source'] == 'block'


def test_void_twice_and_guard_rejections_fall_back_with_the_node_normalised_locked_time():
    plan = {'S0': [{'t': 215.0}], 'A-NAT': [{'g2bad': True}], 'A-ISW': [{'valid': False}, {'valid': False}],
            'A-RND': [{'t': 160.0}]}
    log, node, _ = _block(plan)
    names = [r['name'] for r in log['runs']]
    assert names == ['S0', 'A-NAT', 'A-ISW', 'A-RND', 'A-ISW']           # the guard rejection is not re-run
    v = log['values']['arms']
    assert v['A-ISW']['label'] == '(fallback)' and abs(v['A-ISW']['t_B'] - 109.0 * 215.0 / 220.0) < 1e-9
    assert v['A-NAT']['label'] == '(fallback)' and abs(v['A-NAT']['t_B'] - 150.0) < 1e-9   # WINH1: own block S0
    assert 'block_guard_rejection' in v['A-NAT']['block_void_reasons'][0]
    assert all(c['magnitude'].endswith('(fallback)') for c in log['values']['magnitude'][:2])


def test_s0_void_twice_uses_the_warm_anchor():
    log, _, _ = _block({'S0': [{'t': 230.0}, {'t': 231.0}], 'A-NAT': [{'t': 140.0}], 'A-ISW': [{'t': 100.0}], 'A-RND': [{'t': 160.0}]})
    v = log['values']
    assert v['t_S0_B'] == 216.1 and 'void' in v['t_S0_B_note'] and v['arms']['A-NAT']['t_B'] == 140.0


def test_hard_stop_abandons_the_remaining_items_into_fallbacks():
    t0 = datetime.datetime(2026, 10, 21, 18, 0, tzinfo=datetime.timezone.utc)
    clock = iter([t0, t0, t0 + datetime.timedelta(seconds=3000), t0 + datetime.timedelta(seconds=3000)] * 3)
    deadline = t0 + datetime.timedelta(seconds=2600)
    log, node, _ = _block({'S0': [{'t': 215.0}], 'A-NAT': [{'t': 140.0}]}, deadline=deadline, now=lambda: next(clock))
    assert [r['name'] for r in log['runs']] == ['S0', 'A-NAT', 'A-ISW']
    assert log['runs'][-1]['abandoned'] == 'hard stop' and len(node.jobs) == 2
    v = log['values']['arms']
    assert v['A-ISW']['label'] == '(fallback)' and v['A-RND']['label'] == '(fallback)'


def test_a_retained_lease_or_uid_process_stops_the_block():
    node = BlockNode({'S0': [{'t': 215.0}], 'A-NAT': [{'t': 140.0}]})
    ops = _ops()
    ops.uid_survivors[64101] = [9]
    log = block.run_block(ops, _sels(), out_dir=tempfile.mkdtemp(), node_refs=REFS, node_runner=node)
    assert 'A-NAT' in log['stopped'] and [r['name'] for r in log['runs']] == ['S0', 'A-NAT'] and len(node.jobs) == 2


def test_a_selection_that_is_not_the_locked_item_is_void_without_rerun():
    sels = _sels()
    sels['A-RND']['bind_digest'] = '0' * 64                               # the locked receipt's digest differs
    log, _, _ = _block({'S0': [{'t': 215.0}], 'A-NAT': [{'t': 140.0}], 'A-ISW': [{'t': 100.0}], 'A-RND': [{'t': 160.0}]}, sels=sels)
    rnd = [r for r in log['runs'] if r['name'] == 'A-RND']
    assert len(rnd) == 1 and 'identity' in rnd[0]['guard_rejections'] and log['values']['arms']['A-RND']['label'] == '(fallback)'


# ---------------------------------------------------------------- selections from the records
def _records(tmp, arm, *, locked_rec, result):
    d = Path(tmp) / arm
    (d / 'receipts').mkdir(parents=True)
    (d / 'result.json').write_text(json.dumps(result))
    if locked_rec is not None:
        (d / 'receipts' / f'{locked_rec["attempt_id"]}.json').write_text(json.dumps(locked_rec))
    return d


def test_load_selection_and_locked_validity_from_the_arm_records(tmp_path):
    nat = _records(tmp_path, 'anat_a', locked_rec={'attempt_id': 'aL', 'judge_valid': True, 'guard_rejections': [], 'node': 'WINH1',
                                                   'phases': {'data_prep': 9.0}},
                   result={'run_id': 'anat_a', 'arm_lease': {'node': 'WINH1'},
                           'extra': {'locked': {'attempt_id': 'aL', 'locked': {'execution_time': 150.0, 'valid': True},
                                                'selected': {'candidate_id': 'score07', 'snapshot_path': '/s/x.tar',
                                                             'tree_digest': 'd' * 64}}}})
    s = block.load_selection('A-NAT', nat)
    assert s['restore_snapshot'] == '/s/x.tar' and s['expected_digest'] == s['bind_digest'] == 'd' * 64 and s['node'] == 'WINH1'
    isw = _records(tmp_path, 'aisw_a', locked_rec={'attempt_id': 'aI', 'run_id': 'aisw_a', 'candidate_digest': 'c' * 64,
                                                   'judge_valid': True, 'node': 'WINH3', 'phases': {'data_prep': 7.6},
                                                   'judge': {'message': {'execution_time': 109.0}},
                                                   'guard_rejections': [{'id': 'G4', 'detail': 'x'}, {'id': 'G7', 'detail': 'y'}]},
                   result={'locked': {'attempt_id': 'aI'}, 'trace': {'selected': 'c2', 'selected_config': {**ISW_CFG, 'note': 'x'}}})
    s2 = block.load_selection('A-ISW', isw)
    assert s2['config'] == ISW_CFG and s2['bind_digest'] == 'c' * 64 and s2['t_locked'] == 109.0 and s2['node'] == 'WINH3'
    v = block.locked_validity('A-ISW', isw)
    assert v['locked_valid'] is False and v['g7_reported_only'] and 'G4' in v['rejections']
    assert block.locked_validity('A-NAT', nat)['locked_valid'] is True
    with pytest.raises(block.BlockError):
        block.load_selection('A-RND', _records(tmp_path, 'arnd_a', locked_rec=None, result={'locked': {'attempt_id': 'zz'}, 'trace': {}}))


# ---------------------------------------------------------------- the shipped scorer is never overwritten
class HashOps:
    def __init__(self, remote):
        self.remote, self.written, self.cmds = remote, {}, []

    def hash_files(self, paths):
        return {p: self.remote.get(p) for p in paths}

    def mkdirs(self, paths, **kw):
        return {}

    def write_files(self, files, **kw):
        self.written.update(files)

    def sh(self, cmd, timeout=120):
        self.cmds.append(cmd)
        return {'code': 0, 'stdout': '', 'stderr': ''}

    def read_json(self, path, default=None):
        return {'ok': True, 'result': {'r': 1}}


def test_checked_node_runner_never_overwrites_the_shipped_scorer():
    want = block._local_node_hashes()
    job = {'run_dir': '/data/x/r', 'attempt_id': 'a1', 'out': '/data/x/r/results/a1.json', 'score_timeout': 480}
    same = HashOps(dict(want))
    assert block.checked_node_runner(same, job) == {'r': 1}
    assert not any(p.endswith('.py') for p in same.written)                  # identical: nothing shipped
    missing = HashOps({p: h for p, h in want.items() if not p.endswith('sanitize.py')})
    block.checked_node_runner(missing, job)
    assert [p for p in missing.written if p.endswith('.py')] == [p for p in want if p.endswith('sanitize.py')]
    differ = HashOps({**want, next(iter(want)): '0' * 64})
    with pytest.raises(block.BlockError):
        block.checked_node_runner(differ, job)
    assert differ.cmds == []                                                  # nothing ran


# ---------------------------------------------------------------- CLI dry runs
@pytest.fixture
def failed_reference_prepare(monkeypatch):
    jobs = []
    base = block_cli.synthetic_node_runner()

    def node(ops, job):
        jobs.append(job)
        if job.get('mode') != 'prepare':
            pytest.fail('scored after failed preparation')
        result = base(ops, job)
        result['reference']['error'] = 'reference weights missing'
        return result

    monkeypatch.setattr(block_cli, 'synthetic_node_runner', lambda *args: node)
    monkeypatch.setattr(block, 'prime', lambda *_: pytest.fail('primed after failed preparation'))
    return jobs


@pytest.mark.parametrize('prime', [False, True])
def test_node_ref_cli_stops_on_failed_reference_prepare(tmp_path, failed_reference_prepare, prime):
    out = tmp_path / 'out'
    rc = block_cli.main(['node-ref', '--local', '--node', 'WINH3', '--run-id', 'n', '--out', str(out)]
                        + (['--prime'] if prime else []))
    rec = json.loads((out / 'node_ref_WINH3.json').read_text())
    assert rc == 3 and 'reference preparation failed' in rec['fatal']
    assert rec['prepare']['reference']['error'] == 'reference weights missing'
    assert rec['t_S0_node'] is None and rec['measured'] is False
    assert rec['attempts'] == [] and rec['gpu_seconds'] == 0 and rec['utc_end']
    assert [job['mode'] for job in failed_reference_prepare] == ['prepare']
    assert not list((out / 'receipts').glob('*.json'))


def test_block_cli_stops_on_failed_reference_prepare(tmp_path, failed_reference_prepare):
    nat = _records(tmp_path, 'anat_a', locked_rec=None,
                   result={'arm_lease': {'node': 'WINH1'},
                           'extra': {'locked': {'skipped': 'selection_is_start', 'selected': {}}}})
    isw, rnd = [_records(tmp_path, run, locked_rec=None,
                         result={'locked': {'node': 'WINH1'}, 'trace': {'selected': 'c0', 'start': 'c0'}})
                for run in ('aisw_a', 'arnd_a')]
    out = tmp_path / 'out'
    rc = block_cli.main(['block', '--local', '--out', str(out), '--anat', str(nat), '--aisw', str(isw),
                        '--arnd', str(rnd), '--deadline', '2099-01-01T00:00:00+08:00'])
    log = json.loads((out / 'block_A.json').read_text())
    assert rc == 3 and 'S0 reference preparation failed' in log['stopped']
    assert log['prepare']['reference']['error'] == 'reference weights missing'
    assert log['runs'] == [] and log['block_void_reruns'] == 0
    assert [job['mode'] for job in failed_reference_prepare] == ['prepare']
    assert not list((out / 'receipts').glob('*.json'))


def test_block_cli_dry_runs(tmp_path):
    nat = _records(tmp_path, 'anat_a', locked_rec={'attempt_id': 'aL', 'judge_valid': True, 'guard_rejections': [], 'node': 'WINH1'},
                   result={'run_id': 'anat_a', 'arm_lease': {'node': 'WINH1'},
                           'extra': {'locked': {'attempt_id': 'aL', 'locked': {'execution_time': 150.0},
                                                'selected': {'snapshot_path': '/s/x.tar', 'tree_digest': 'd' * 64}}}})
    isw = _records(tmp_path, 'aisw_a', locked_rec={'attempt_id': 'aI', 'run_id': 'aisw_a', 'judge_valid': True, 'node': 'WINH3',
                                                   'candidate_digest': _cdig('A-ISW', ISW_CFG), 'guard_rejections': [],
                                                   'judge': {'message': {'execution_time': 109.0}}},
                   result={'locked': {'attempt_id': 'aI'}, 'trace': {'selected_config': ISW_CFG}})
    rnd = _records(tmp_path, 'arnd_a', locked_rec={'attempt_id': 'aR', 'run_id': 'arnd_a', 'judge_valid': True, 'node': 'WINH4',
                                                   'candidate_digest': _cdig('A-RND', RND_CFG), 'guard_rejections': [],
                                                   'judge': {'message': {'execution_time': 170.0}}},
                   result={'locked': {'attempt_id': 'aR'}, 'trace': {'selected_config': RND_CFG}})
    out = tmp_path / 'out'
    rc = block_cli.main(['block', '--local', '--out', str(out), '--anat', str(nat), '--aisw', str(isw), '--arnd', str(rnd),
                         '--node-ref', 'WINH3=220.0', '--node-ref', 'WINH4=225.0', '--deadline', '2099-01-01T00:00:00+08:00'])
    blk = json.loads((out / 'block_A.json').read_text())
    assert rc == 0 and [r['name'] for r in blk['runs']] == ['S0', 'A-NAT', 'A-ISW', 'A-RND']
    assert all(not r['void'] for r in blk['runs']) and blk['values']['t_S0_B'] == 216.0
    assert block_cli.main(['node-ref', '--local', '--node', 'WINH3', '--run-id', 'n', '--out', str(tmp_path / 'nr')]) == 0
    assert block_cli.main(['block', '--local', '--anat', str(tmp_path / 'missing'), '--aisw', str(isw), '--arnd', str(rnd)]) == 2


def test_band_flag_and_first_scorings_and_rerun_count(tmp_path):
    log, _, _ = _block({'S0': [{'t': 230.0}, {'t': 215.0}], 'A-NAT': [{'t': 140.0}], 'A-ISW': [{'t': 100.0}], 'A-RND': [{'t': 160.0}]})
    s0 = [r for r in log['runs'] if r['name'] == 'S0']
    assert [r['in_band'] for r in s0] == [False, True] and log['block_void_reruns'] == 1
    rnd = _records(tmp_path, 'arnd_a', locked_rec={'attempt_id': 'c0', 'node': 'WINH4', 'valid': True,
                                                   'judge': {'message': {'execution_time': 262.22}},
                                                   'phases': {'data_prep': 24.0, 'train': 204.1, 'convert': 30.3}},
                   result={})
    f = block.first_scoring('A-RND', rnd)
    assert f['t_first'] == 262.22 and f['first_cold'] is True and f['phases']['data_prep'] == 24.0
    assert block.first_scoring('A-NAT', tmp_path / 'none')['t_first'] is None


def test_block_without_a_degenerate_native_arm():
    sels = _sels()
    sels['A-NAT'] = None                                                 # EXT_006: restarted A-NAT with no valid score
    log, node, _ = _block({'S0': [{'t': 215.0}], 'A-ISW': [{'t': 100.0}], 'A-RND': [{'t': 160.0}]}, sels=sels)
    assert [r['name'] for r in log['runs']] == ['S0', 'A-ISW', 'A-RND'] and [j['uid'] for j in node.jobs] == [64103, 64102, 64103]
    v = log['values']
    assert v['arms']['A-NAT']['label'] == 'degenerate (no valid score)'
    mags = {m['pair']: m['magnitude'] for m in v['magnitude']}
    assert mags['A-ISW vs A-NAT'].startswith('no comparison') and mags['A-ISW vs A-RND'] == 'A-ISW faster by >= 5 %'


# ---------------------------------------------------------------- decision D20: the selection is the unchanged start
def test_d20_finish_paths_skip_the_locked_rescore_of_the_start():
    from infrarsi.ext.rebench_llmf import arms, tools
    calls = []
    runner = type('R', (), {'node': 'WINH4', 'base': None, 'task': None,
                            'run': lambda self, *a, **k: calls.append((a, k)) or {}})()
    out = arms.locked_rescore_selection(runner, None, {'selected': 'c0', 'start': 'c0', 'selected_config': {}}, run_id='r')
    assert out['skipped'] == 'selection_is_start' and out['invalid_accept'] == 0 and out['locked'] is None and calls == []
    assert arms.selection_is_start({'selected': 'c7', 'start': 'c0', 'selected_config': {}})        # renders the start
    assert not arms.selection_is_start({'selected': 'c7', 'start': 'c0', 'selected_config': dict(RND_CFG)})
    st = tools.ScoreState('anat')
    st.best = {'candidate_id': 'score01', 'n': 1, 'execution_time': 216.0, 'score': 5.4,
               'tree_digest': taskmod.START_TREE_DIGEST, 'snapshot_path': '/s/start.tar'}
    nat = tools.locked_rescore(runner, 'anat', st)
    assert nat['skipped'] == 'selection_is_start' and nat['selected']['candidate_id'] == 'score01' and calls == []


def test_d20_block_uses_the_s0_for_a_start_selection(tmp_path):
    # A-RND's old-code records: a locked re-score ran, but its selection is the start (c0) -> D20 applies anyway
    rnd = _records(tmp_path, 'arnd_a', locked_rec={'attempt_id': 'aR', 'run_id': 'arnd_a', 'judge_valid': True, 'node': 'WINH4',
                                                   'candidate_digest': 'x' * 64, 'guard_rejections': [],
                                                   'judge': {'message': {'execution_time': 214.0}}},
                   result={'locked': {'attempt_id': 'aR'}, 'trace': {'selected': 'c0', 'start': 'c0', 'selected_config': {}}})
    sel = block.load_selection('A-RND', rnd)
    assert sel['is_start'] and sel['node'] == 'WINH4' and sel['skipped'] == 'selection_is_start'
    v = block.locked_validity('A-RND', rnd)
    assert v['locked_valid'] is True and v['invalid_accept'] == 0
    sels = _sels()
    sels['A-RND'] = sel
    log, node, _ = _block({'S0': [{'t': 215.0}], 'A-NAT': [{'t': 140.0}], 'A-ISW': [{'t': 100.0}]}, sels=sels)
    assert [r['name'] for r in log['runs']] == ['S0', 'A-NAT', 'A-ISW'] and 64103 == node.jobs[0]['uid'] and len(node.jobs) == 3
    row = log['values']['arms']['A-RND']
    assert row['t_B'] == 215.0 and row['label'] == 'selection_is_start (S0)' and row['t_locked'] == 225.0   # N_WINH4
    assert abs(row['y_B']) < 1e-12 and not row['fallback']
    mags = {m['pair']: m['magnitude'] for m in log['values']['magnitude']}
    assert mags['A-ISW vs A-RND'] == 'A-ISW faster by >= 5 %'                  # no '(fallback)' suffix for D20
    nat_rec = _records(tmp_path, 'anat_b', locked_rec=None,
                       result={'run_id': 'anat_b', 'arm_lease': {'node': 'WINH1'},
                               'extra': {'locked': {'locked': None, 'skipped': 'selection_is_start',
                                                    'selected': {'candidate_id': 'score01', 'tree_digest': taskmod.START_TREE_DIGEST}}}})
    assert block.load_selection('A-NAT', nat_rec)['is_start'] and block.locked_validity('A-NAT', nat_rec)['invalid_accept'] == 0


# ---------------------------------------------------------------- AMENDMENT_EXT_005 G4 inputs (g4-inputs)
def _g4_records(tmp_path):
    for run, locked in (('anat_b', {'extra': {'locked': {'attempt_id': 'aN'}}}), ('aisw_a', {'locked': {'attempt_id': 'aI'}}),
                        ('arnd_a', {'locked': {'attempt_id': 'aR'}})):
        (tmp_path / run).mkdir(parents=True)
        (tmp_path / run / 'result.json').write_text(json.dumps(locked))
    (tmp_path / 'block_a').mkdir()
    (tmp_path / 'block_a' / 'block_A.json').write_text(json.dumps({'runs': [{'name': 'S0', 'void': False, 'norm_diff': 4.0, 'attempt_id': 'b0'}]}))
    (tmp_path / 'block_a' / 'node_ref_WINH3.json').write_text(json.dumps({'attempts': [{'void': False, 'norm_diff': 4.2, 'attempt_id': 'n3'}]}))
    return tmp_path


def test_g4_inputs_pairs_locked_norms_lease_and_preview(tmp_path):
    rec = _g4_records(tmp_path / 'records')
    ops = LocalNodeOps(tempfile.mkdtemp(prefix='isg4'), node='WINH3')
    for run, att, v in (('anat_b', 'aN', 4.9), ('aisw_a', 'aI', 5.41), ('arnd_a', 'aR', 5.0)):
        ops.write_files({f'{taskmod.NODE_RUNS}/{run}/results/{att}_node_result.json':
                         json.dumps({'ok': True, 'result': {'g4': {'computed': True, 'norm_cand_dprime_vs_refprime': v}}})})
    seen = []

    def pairs(o, a, b):
        seen.append((a, b))
        return {'ok': True, 'norm_diff': 5.0, 'max_scalar_diff': 0.002, 'models': {a: {'recorded': 'x'}, b: {'recorded': 'y'}}}
    doc = block.g4_inputs(ops, records=rec, out_path=tmp_path / 'g4_inputs.json', pair_runner=pairs)
    assert doc['final'] is True and len(seen) == 3 and all(p.endswith('/ref_permuted/trained_model') for ab in seen for p in ab)
    assert [r['pair'] for r in doc['refprime_pairwise_norms']] == [['anat_b', 'aisw_a'], ['anat_b', 'arnd_a'], ['aisw_a', 'arnd_a']]
    assert doc['locked_dprime_norms']['aI']['norm_cand_dprime_vs_refprime'] == 5.41 and len(doc['locked_dprime_norms']['aI']['sha256']) == 64
    assert doc['lease']['kept'] == [] and len(doc['lease']['released']) == 5 and ops.read_json('/data/infrarsi/leases/WINH3.json') is None
    pv = doc['preview']
    assert pv['m_samples'] == [3.41, 4.27, 3.43, 4.22, 4.28, 4.0, 4.2] and pv['m_prime_samples'] == [4.79, 5.0, 5.0, 5.0]
    assert abs(pv['limit_dprime'] - (5.3 + (sum(pv['m_prime_samples']) / 4 - sum(pv['m_samples']) / 7))) < 1e-12
    assert json.loads((tmp_path / 'g4_inputs.json').read_text())['final'] is True


def test_g4_inputs_not_final_when_a_pair_failed(tmp_path):
    rec = _g4_records(tmp_path / 'records')
    ops = LocalNodeOps(tempfile.mkdtemp(prefix='isg4b'), node='WINH3')
    calls = iter([{'ok': True, 'norm_diff': 5.0}, {'ok': False, 'error': 'digest changed'}, {'ok': True, 'norm_diff': 5.1}])
    doc = block.g4_inputs(ops, records=rec, out_path=tmp_path / 'g.json', pair_runner=lambda o, a, b: next(calls))
    assert doc['final'] is False and doc['refprime_pairwise_norms'][1]['error'] == 'digest changed'
    assert doc['locked_dprime_norms']['aN']['norm_cand_dprime_vs_refprime'] is None      # node file absent: recorded as None


def test_g4_pair_script_checks_the_cache_digests(tmp_path):
    """The node-side pair script, run here against a stub scorer module: digests are checked before and after, the norm
    comes from the scorer's own compute_norm_and_max_diff, and a changed cache fails the pair."""
    import subprocess
    import sys
    tools = tmp_path / 'tools'
    tools.mkdir()
    (tools / 'score_runner.py').write_text(
        "import json, os\n"
        "STATE = {'n': 0}\n"
        "def check_safe_model_dir(p): pass\n"
        "def tree_digest_files(p):\n"
        "    STATE['n'] += 1\n"
        "    return 'changed' if os.environ.get('MUTATE') and STATE['n'] > 2 else 'dig'\n"
        "class S:\n"
        "    def load_model(self, p): return p\n"
        "    def compute_norm_and_max_diff(self, a, b): return 4.97, 0.001\n"
        "def load_score_module(score_py, home, ref): return S()\n")
    for m in ('a', 'b'):
        (tmp_path / m / 'trained_model').mkdir(parents=True)
        (tmp_path / m / 'TREE_DIGEST.json').write_text(json.dumps({'tree_digest': 'dig'}))
    script = tmp_path / 'g4_pair.py'
    script.write_text(block.G4_PAIR_SCRIPT % {'tools': str(tools), 'sentinel': block.G4_SENTINEL})
    args = [sys.executable, str(script), str(tmp_path / 'a' / 'trained_model'), str(tmp_path / 'b' / 'trained_model'), 'score.py']
    out = subprocess.run(args, capture_output=True, text=True).stdout
    res = json.loads(out.rsplit(block.G4_SENTINEL, 1)[1])
    assert res['ok'] and res['norm_diff'] == 4.97 and all(v['before'] == v['after'] == 'dig' for v in res['models'].values())
    import os
    bad = subprocess.run(args, capture_output=True, text=True, env={**os.environ, 'MUTATE': '1'}).stdout
    res2 = json.loads(bad.rsplit(block.G4_SENTINEL, 1)[1])
    assert res2['ok'] is False and 'changed' in res2['error']


def test_g4_inputs_cli_local(tmp_path):
    rec = _g4_records(tmp_path / 'records')
    out = tmp_path / 'g4.json'
    rc = block_cli.main(['g4-inputs', '--local', '--node', 'WINH3', '--records', str(rec), '--out', str(out)])
    doc = json.loads(out.read_text())
    assert rc == 0 and doc['final'] is True and set(doc['locked_dprime_norms']) == {'aN', 'aI', 'aR'}
    assert block_cli.main(['g4-inputs', '--local', '--records', str(tmp_path / 'none'), '--out', str(out)]) == 2
