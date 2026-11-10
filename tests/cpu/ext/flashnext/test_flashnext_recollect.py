"""The harness's supersede step (AMENDMENT_EXT_009 Q1, recollect.supersede / check) and the collector proof (Q4), on a
simulated node: the integrity conditions (i)-(iv), (vi), the attempt binding as an identity veto (v), the reference
checks, value-blindness, X6's record format and acceptance (supersede.record_problems / index). The judge's verdict is
injected here; the unchanged judge ran on the real c1 files (X5_F1_BUILD.md §11.3)."""
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from infrarsi.ext.common.remote_exec import WinhNodeOps
from infrarsi.ext.flashnext import guards as G
from infrarsi.ext.flashnext import judge as J
from infrarsi.ext.flashnext import recollect as RC
from infrarsi.ext.flashnext import runner as R
from infrarsi.ext.flashnext import supersede as SU
from infrarsi.ext.flashnext import switch_space as S
from infrarsi.harness import signing
from test_flashnext_collect import RacyRemote

ATT = 'abbd3dc252602'
NAME = 'flash32_joint_tp8_p1_lp_rsi_unified_s_match_a_00000000_atest'
META = f'/data/infrarsi/ext/runs/flashnext/metadata/{NAME}'
OUT = f'/data/infrarsi/ext/runs/flashnext/runs/{NAME}'
NODES = ['WINH0', 'WINH1', 'WINH4', 'WINH6']
T0 = 1_792_550_000.0                                        # node start (unix)
UTC_START = datetime.datetime.fromtimestamp(T0 - 120, datetime.timezone.utc).isoformat()
UTC_END = datetime.datetime.fromtimestamp(T0 + 900, datetime.timezone.utc).isoformat()
TIMEOUT_MSG = "WINH0: transport failed: TimeoutError('WINH0: remote command exceeded 300s')"
LAY = R.layout_from_records({'run': NAME, 'attempt_id': 'abundle', 'lifecycle_run_root': OUT + '/training'},
                            {'name': NAME, 'metadata_directory': META, 'output_root': OUT})


def _iso(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).isoformat()


def _original(**over):
    r = {'schema': 'ext-receipt-v1', 'env': 'flashnext', 'task_id': 'ENV-F1', 'arm': 'F-PROMPTS', 'run_id': 'f1_prompts_t',
         'attempt_id': ATT, 'candidate_id': 'c1', 'purpose': 'screen', 'point': S.registered_points()['S_match'],
         'nodes': NODES, 'utc_start': UTC_START, 'utc_end': UTC_END, 'valid': False, 'judge': None, 'judge_valid': False,
         'invalid_reasons': ['runner_error: JSONDecodeError: Unterminated string starting at: line 1 column 41868003'],
         'runner_error': {'kind': 'runner_error', 'type': 'JSONDecodeError', 'message': 'Unterminated string'},
         'cost': {'gpu_seconds': 16309.008, 'wall_seconds': 1286.671, 'accounting_complete': True},
         'metric': {'name': 'step_time_s', 'value': None, 'direction': 'min', 'raw': {}}, 'cap_s': 1800.0,
         'numerics_role': 'candidate', 'release_by_node': {n: {'state': 'removed'} for n in NODES},
         'switch_space_sha256': S.sha256(), 'wall_s': 1286.671, 'phases': {'total': 1286.671}}
    r.update(over)
    return signing.sign(r)


def _uc(**over):
    r = {'schema': 'ext-receipt-v1', 'env': 'flashnext', 'task_id': 'ENV-F1', 'arm': 'UC-REF', 'run_id': 'f1_ucref_t',
         'attempt_id': 'aucref', 'candidate_id': 'uc-ref', 'purpose': 'reference', 'point': S.start_point(),
         'nodes': NODES, 'valid': True}
    r.update(over)
    return signing.sign(r)


def _judged(dest, lay, original, reference, task, extra_rejections=()):
    """A judge verdict as the runner's receipt carries it (valid, E0, the frozen gates, guards.check's reports), the
    caller's identity vetoes applied."""
    rej = [{'id': g, 'detail': d} for g, d in extra_rejections]
    valid = not rej
    return {**{k: original[k] for k in SU.IDENTITY}, 'valid': valid, 'judge_valid': True,
            'invalid_reasons': [f'guard:{r["id"]}: {r["detail"]}' for r in rej],
            'metric': {'name': 'step_time_s', 'value': 8.341 if valid else None, 'direction': 'min', 'raw': {}},
            'step_time': {'mean_window_max_rank_s': 8.341 if valid else None},
            'judge': {'judge_valid': True, 'numerics_class': 'E0', 'reasons': [], 'metric_s': 8.341, 'details': {}},
            'numerics_gates_sha256': J.gates_sha256(), 'guard_rejections': rej,
            'guard_reports': [{'id': i} for i in SU.REQUIRED_REPORTS], 'cost': {'gpu_seconds': 0.0}}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A simulated node holding one screen's 88 outputs (mtimes inside their windows); the run directory with the
    original runner_error receipt, the run's own U+C reference as receipts/c0.json and the attempt directory the failed
    old collector left (the first 7 batches of 8 of its sorted order); a reader over the node."""
    root = tmp_path.resolve() / 'node'
    if sys.platform == 'win32':
        root = Path('\\\\?\\' + str(root))
    wanted, logs = R.FlashNextRunner.output_files(LAY)
    files = {}
    for r, p in enumerate(LAY['signals']):
        files[p] = b''.join(json.dumps({'kind': 'train_step', 'rank': r, 'step': s, 'v': 'x' * 2000}).encode() + b'\n'
                            for s in range(1, 31))
    for r, p in enumerate(LAY['rank_receipts']):
        files[p] = json.dumps({'rank': r}).encode()
    for n in range(4):
        files[f'{OUT}/node{n}.exit.json'] = json.dumps({'exit_code': 0, 'started_unix': T0 + n, 'finished_unix': T0 + 480}).encode()
        files[f'{OUT}/node{n}.pid.json'] = json.dumps({'pid': 4000 + n, 'started_unix': T0 + n}).encode()
        files[f'{OUT}/node{n}.release.json'] = json.dumps({'idle': True, 'process_group_gone': True,
                                                           'finished_utc': _iso(T0 + 482)}).encode()
        files[f'{OUT}/node{n}.log'] = b'training\n' * 100
        files[f'{META}/fn_node{n}.launch.log'] = b'nohup: ignoring input\n'
    files.update({f'{META}/attempt.json': json.dumps({'run': NAME, 'attempt_id': 'abundle',
                                                      'lifecycle_run_root': OUT + '/training'}).encode(),
                  f'{META}/job.json': json.dumps({'name': NAME, 'metadata_directory': META, 'output_root': OUT,
                                                  'master_port': R.master_port(ATT)}).encode(),
                  f'{META}/fn_spec.json': json.dumps({'nodes': NODES}).encode(),
                  f'{OUT}/family_materialization/materialization.json': b'{"ok": true}'})
    for path, data in files.items():
        p = root / path.lstrip('/')
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        name = wanted.get(path) or logs[path]
        # uploads well before the start; launch logs just before their node's start (as measured: -0.2..-0.8 s)
        t = T0 - 60 if name in RC.PRE_LAUNCH else (T0 + int(name[13]) - 0.5 if name.endswith('.launch.log') else T0 + 300)
        os.utime(p, (t, t))
    run_dir = tmp_path / 'exec' / 'f1_prompts_t'
    (run_dir / 'receipts').mkdir(parents=True)
    original, uc = _original(), _uc()
    (run_dir / 'receipts' / f'{ATT}.json').write_text(json.dumps(original), encoding='utf-8')
    (run_dir / 'receipts' / 'c0.json').write_text(json.dumps(uc), encoding='utf-8')
    part = run_dir / 'attempts' / f'c1-{ATT}'
    by_name = {v: k for k, v in wanted.items()}
    for name in RC.old_collector_order(LAY)[:7 * RC.OLD_BATCH]:            # what a failure at batch 8 leaves
        dst = part / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(files[by_name[name]])
    remote = RacyRemote(root)
    rdr = RC.reader('WINH3', {n: WinhNodeOps(n, remote=remote) for n in ('WINH3', 'WINH0', 'WINH1', 'WINH4')})
    monkeypatch.setattr(rdr, 'judge_recollected', _judged)
    monkeypatch.setattr(J, 'reference_from_receipt', lambda r: r)
    monkeypatch.setattr(G, 'uc_reference_problems', lambda r: [])          # X6's identity checks: tested by X6
    return {'root': root, 'run_dir': run_dir, 'rdr': rdr, 'original': original, 'uc': uc, 'files': files, 'part': part}


def _supersede(w, uc=None):
    return RC.supersede(w['run_dir'], ATT, uc or w['uc'], rdr=w['rdr'])


def _node_file(w, remote: str) -> Path:
    return w['root'] / remote.lstrip('/')


def test_a_37_class_screen_gets_an_admitted_signed_record(world):
    res = _supersede(world)
    rec = res['record']
    assert Path(res['path']).name == 'c1.recollected.json' and signing.verify(json.loads(Path(res['path']).read_text()))
    assert rec['collection'] == 'recollected_readonly' and rec['supersedes'] == ATT and rec['valid'] is True
    assert all(rec[k] == world['original'][k] for k in SU.IDENTITY) and rec['cost'] == world['original']['cost']
    rc = rec['recollection']
    assert all(rc['conditions'].values()) and len(rc['files']) == SU.LAYOUT_FILES and rc['error_class'] == 'truncated_reply'
    pc = rc['pulled_check']
    assert (pc['files'], pc['identical'], pc['old_collector_prefix'], pc['batches']) == (56, 56, True, 7)
    assert len(rc['pulled']) == 56 and rc['binding_problems'] == [] and rc['method'] == 'fixed_collector'
    assert set(rc['node_records']) == set(NODES) and rc['stage'] == 'collect'
    log = next(f for f in rc['files'] if f['path'] == 'nodes/node0.log')
    assert log['size'] == 900 and log['source_bytes'] == 900          # a short log: the tail is the whole file
    originals, admitted, problems = SU.index([world['original'], rec])
    assert problems == [] and admitted[ATT]['valid'] is True
    assert (world['run_dir'].parent / '_recollected' / 'f1_prompts_t' / f'c1-{ATT}' / 'signals/rank31.jsonl').is_file()
    with pytest.raises(RC.Refused, match=r'\(vi\)'):        # one supersede per attempt
        _supersede(world)


@pytest.mark.parametrize('over,needle', [
    ({'runner_error': {'kind': 'runner_error', 'type': 'RuntimeError', 'message': 'collection: ...'}}, 'collection class'),
    ({'runner_error': {'kind': 'runner_error', 'type': 'RemoteError', 'message': TIMEOUT_MSG}}, 'collection class'),
    ({'runner_error': {'kind': 'runner_error', 'type': 'NodeError', 'message': TIMEOUT_MSG}}, 'collection class'),
    ({'runner_error': {'kind': 'occupied', 'type': 'OccupiedError', 'message': 'x'}}, 'runner_error with complete'),
    ({'cost': {'gpu_seconds': 1.0, 'accounting_complete': False}}, 'complete accounting'),
    ({'release_by_node': {**{n: {'state': 'removed'} for n in NODES}, 'WINH4': {'state': 'kept'}}}, 'not after the job'),
    ({'supersedes': 'a0'}, '(vi)'),
    ({'purpose': 'locked'}, 'only a screen')])
def test_an_original_outside_the_rule_is_refused_and_nothing_is_written(world, over, needle):
    (world['run_dir'] / 'receipts' / f'{ATT}.json').write_text(json.dumps(_original(**over)), encoding='utf-8')
    with pytest.raises(RC.Refused, match=needle.replace('(', r'\(').replace(')', r'\)')):
        _supersede(world)
    assert not list((world['run_dir'] / 'receipts').glob('*.supersede.json'))


def test_the_ruled_c3_timeout_is_the_collect_transport_timeout_class(world, monkeypatch):
    """O's ruling under EXT_009 Q3 (c3): only the ruled attempt, only RemoteError in the exact timeout form."""
    monkeypatch.setitem(RC.RULED_TRANSPORT_TIMEOUTS, ('f1_prompts_t', ATT), 'test ruling')
    err = {'kind': 'runner_error', 'type': 'RemoteError', 'message': TIMEOUT_MSG}
    (world['run_dir'] / 'receipts' / f'{ATT}.json').write_text(json.dumps(_original(runner_error=err)), encoding='utf-8')
    rec = _supersede(world)['record']
    assert rec['collection'] == 'recollected_readonly' and rec['recollection']['error_class'] == 'collect_transport_timeout'
    assert rec['recollection']['error_class_ruling'] == 'test ruling' and rec['recollection']['original_error'] == err


@pytest.mark.parametrize('uc_over,c0_over,needle', [
    ({'nodes': ['WINH0', 'WINH1', 'WINH3', 'WINH4']}, None, 'the reference ran on'),
    ({'attempt_id': 'aother'}, None, "own U\\+C reference"),
    (None, {'attempt_id': 'aother'}, "own U\\+C reference")])
def test_a_reference_other_than_the_runs_own_is_refused(world, uc_over, c0_over, needle):
    uc = _uc(**uc_over) if uc_over else world['uc']
    if c0_over:
        (world['run_dir'] / 'receipts' / 'c0.json').write_text(json.dumps(_uc(**c0_over)), encoding='utf-8')
    with pytest.raises(RC.Refused, match=needle):
        _supersede(world, uc=uc)
    assert not list((world['run_dir'] / 'receipts').glob('*.supersede.json'))


def test_a_node_record_that_is_not_a_completed_job_is_refused(world):
    p = _node_file(world, f'{OUT}/node2.exit.json')
    p.write_text(json.dumps({'exit_code': 1, 'started_unix': T0, 'finished_unix': T0 + 100}))
    os.utime(p, (T0 + 300, T0 + 300))
    with pytest.raises(RC.Refused, match='completed job'):
        _supersede(world)


@pytest.mark.parametrize('damage', ['late_mtime', 'pulled_differs', 'pulled_file_removed', 'missing_file'])
def test_a_failed_integrity_condition_gives_the_unrecoverable_flag_record(world, damage):
    def sig(r):
        return _node_file(world, f'{OUT}/training/{NAME}.signals.rank{r}/signals.jsonl')
    if damage == 'late_mtime':                              # (iii): written after the release + 60 s
        os.utime(sig(9), (T0 + 482 + 61, T0 + 482 + 61))
        cond = 'iii_mtimes_within_node_windows'
    elif damage == 'pulled_differs':                        # (iv): the node copy of a pulled file changed afterwards
        p = sig(10)
        p.write_bytes(p.read_bytes().replace(b'"rank": 10', b'"rank": 11', 1))
        os.utime(p, (T0 + 300, T0 + 300))
        cond = 'iv_pulled_files_identical'
    elif damage == 'pulled_file_removed':                   # (iv): a pulled file was removed from the attempt directory
        (world['part'] / 'signals' / 'rank11.jsonl').unlink()
        cond = 'iv_pulled_files_identical'
    else:                                                   # (ii): a layout file is gone
        sig(20).unlink()
        cond = 'ii_readonly_complete_layout'
    res = _supersede(world)
    rec = res['record']
    assert Path(res['path']).name == 'c1.unrecoverable.json'
    assert rec['collection'] == 'unrecoverable' and rec['valid'] is False and rec['recollection']['conditions'][cond] is False
    assert rec['recollection']['conditions']['i_collection_runner_error'] is True
    originals, admitted, problems = SU.index([world['original'], rec])
    assert problems == [] and admitted[ATT]['collection'] == 'unrecoverable'


def test_an_attempt_binding_mismatch_is_an_identity_veto_in_the_verdict(world):
    """Q1 (v): the files are judged; the binding mismatch vetoes the screen (it stays invalid with that reason)."""
    p = _node_file(world, f'{META}/job.json')
    job = json.loads(p.read_text())
    job['master_port'] = R.master_port(ATT) + 1
    p.write_text(json.dumps(job))
    os.utime(p, (T0 - 60, T0 - 60))
    shutil.copy(p, world['part'] / 'job.json')              # what the failed collection had pulled, too
    rec = _supersede(world)['record']
    assert rec['collection'] == 'recollected_readonly' and rec['valid'] is False
    assert [g['id'] for g in rec['guard_rejections']] == ['attempt_binding']
    assert SU.index([world['original'], rec])[2] == []


def test_check_is_value_blind_and_leaves_nothing(world):
    out = RC.check(world['run_dir'], ATT, rdr=world['rdr'])
    assert all(out['conditions'].values()) and 'valid' not in out and 'metric_s' not in out
    assert not (world['run_dir'].parent / '_recollected').exists()
    assert not list((world['run_dir'] / 'receipts').glob('*.supersede.json'))


def test_the_proof_compares_every_file_of_a_recorded_collection(world, tmp_path):
    """Q4 on the simulated node: a complete recorded collection matches sha256 for sha256; one changed file is named."""
    rdr = world['rdr']
    full = tmp_path / 'recorded'
    rdr.fetch_outputs(LAY, full)
    res = RC.proof(full, rdr=rdr, dest=tmp_path / 'again')
    assert res['all_equal'] and res['files'] == res['equal'] == SU.LAYOUT_FILES
    (full / 'signals/rank05.jsonl').write_bytes(b'tampered')
    res = RC.proof(full, rdr=rdr, dest=tmp_path / 'again2')
    assert not res['all_equal'] and [r['name'] for r in res['rows'] if not r['equal']] == ['signals/rank05.jsonl']
