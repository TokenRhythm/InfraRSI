"""Dispatcher: dependency and gate resolution, priorities, lanes, reservations, pause, occupancy refusal, run leases,
terminal states, and the E0/E7/deferred handlers."""
import json
import time

import pytest

from infrarsi.dispatch import Dispatcher, QueueStore, lane_nodes, load_matrix, parse_priority, plan
from infrarsi.harness.remote import LocalRemote

G0 = 'G0:design_frozen+cpu_tests+downloads_verified+occupancy_check'


@pytest.fixture
def store(tmp_path):
    return QueueStore(tmp_path / 'queue')


@pytest.fixture(scope='module')
def matrix():
    return load_matrix()


def ids(p):
    return [a['run_id'] for a in p['assign']]


def test_parsers():
    assert lane_nodes('WINH4+5') == [4, 5] and lane_nodes('WINH0-3') == [0, 1, 2, 3] and lane_nodes('WINH2') == [2]
    assert lane_nodes('free-lane') is None and lane_nodes('driver-PC') == 'driver' and lane_nodes('WIN0 8xH20') == 'external'
    assert lane_nodes('free-lane (WINH3 D3-4)') == [3]
    assert parse_priority('P0 (defer to P1 if no key)')['level'] == 'P0'
    assert parse_priority('P2 (not executed by default)')['default_enabled'] is False


def test_nothing_runs_before_g0(store, matrix):
    p = plan(matrix, store)
    assert p['assign'] == [] and len(p['blocked']) == len(matrix['runs'])


def test_gate_then_run_dependencies(store, matrix):
    store.set_marker(G0)
    p = plan(matrix, store)
    assert set(ids(p)) == {'E0-CPU-HARNESS-SELFTEST', 'E0-SMOKE-S1-SYN', 'E0-CAL-S1-FR', 'E0-REDTEAM-CPU', 'E0-WEAK-ENDPOINT-CHECK'}
    store.update_run('E0-SMOKE-S1-SYN', status='done')
    p = plan(matrix, store)
    assert {'E0-CAL-S2-FR', 'E0-CAL-D2-FR', 'E0-CAL-D4-FR', 'E0-CAL-D5-FR'} <= set(ids(p))
    assert 'E0-LIFECYCLE-DENSE8' not in ids(p)                      # needs E0-CAL-D2-FR done
    store.set_marker('G1')                                          # short gate name satisfies 'G1:E0_pass'
    store.set_marker('DL:qwen3-8b')
    assert {'E1-H3-FR', 'E1-H4-FR'} <= set(ids(plan(matrix, store)))


def test_downloads_resolve_dl_markers(store, matrix):
    store.set_marker('G1')
    p = plan(matrix, store, downloads={'qwen3-4b': True, 'qwen3-next-80b': False})
    assert 'E1-H6-FR' in ids(p) and 'E1-D6-FR' not in ids(p)


def test_priority_order_and_lane_assignment(store, matrix):
    store.set_marker(G0)
    store.update_run('E0-SMOKE-S1-SYN', status='done')
    p = plan(matrix, store)
    order = ids(p)
    assert order.index('E0-CAL-S1-FR') < order.index('E0-CAL-D5-FR')     # same priority: planned_day, then matrix order
    by = {a['run_id']: a for a in p['assign']}
    assert by['E0-CAL-D2-FR']['nodes'] == ['WINH2'] and by['E0-CAL-D4-FR']['nodes'] == ['WINH1']
    assert by['E0-CAL-D5-FR']['nodes'] == ['WINH4', 'WINH5'] and by['E0-CPU-HARNESS-SELFTEST']['where'] == 'driver'
    used = [n for a in p['assign'] if a['nodes'] for n in a['nodes']]
    assert len(used) == len(set(used))                               # whole-node exclusivity


def test_multinode_reservation_blocks_backfill(store, matrix):
    store.set_marker(G0)
    for rid in ('E0-SMOKE-S1-SYN', 'E0-CAL-S1-FR', 'E0-CAL-S2-FR', 'E0-CAL-D2-FR', 'E0-CAL-D1-FR', 'E0-CAL-D4-FR', 'E0-CAL-D3-FR'):
        store.update_run(rid, status='done')
    p = plan(matrix, store, live_free={5})                          # one free node: the 2-node D5 waits and holds it
    by = {a['run_id']: a for a in p['assign']}
    assert by['E0-CAL-D5-FR']['nodes'] is None and p['reserved_nodes'] == [5]
    assert by['E0-LIFECYCLE-DENSE8']['nodes'] is None                # lower priority cannot backfill the reserved node
    p = plan(matrix, store, live_free={0, 1, 5})                     # two free nodes (not the lane): D5 still starts
    assert {a['run_id']: a for a in p['assign']}['E0-CAL-D5-FR']['nodes'] is not None


def test_p2_and_not_by_default(store, matrix):
    for m in ('P0_CORE_COMPLETE', 'USER_APPROVAL'):
        store.set_marker(m)
    assert 'P2-T3-PLANNER-64' not in ids(plan(matrix, store))
    assert 'P2-T3-PLANNER-64' in ids(plan(matrix, store, include_p2=True))
    assert 'E7-F1-SMATCH-1024' not in ids(plan(matrix, store, include_p2=True))
    assert 'E7-F1-SMATCH-1024' in ids(plan(matrix, store, enabled_conditional={'E7-F1-SMATCH-1024'}))


def _disp(store, matrix, remote, calls):
    def fake(run, nodes, ctx):
        calls.append((run['run_id'], nodes))
        return {'ok': True}
    return Dispatcher(remote=remote, store=store, matrix=matrix, handler_map={'FR': fake, 'harness_selftest': fake},
                      log=lambda m: None)


def _join(d):
    for t in list(d.threads.values()):
        t.join(5)


def test_paused_queue_launches_nothing(store, matrix, tmp_path):
    store.set_marker(G0)
    store.pause('CRIT: test')                                          # no lanes: global pause
    calls = []
    d = _disp(store, matrix, LocalRemote(tmp_path / 'fs'), calls)
    assert d.step() == [] and calls == []
    store.unpause()
    store.pause('CRIT: lane test', lanes=['WINH0'])                    # lane-scoped: only WINH0 assignments wait
    assert set(d.step()) == {'E0-CPU-HARNESS-SELFTEST'}
    _join(d)
    store.unpause()
    assert set(d.step()) == {'E0-CAL-S1-FR'}
    _join(d)
    assert store.state()['runs']['E0-CAL-S1-FR']['status'] == 'done'


def test_occupied_node_is_never_used(store, matrix, tmp_path):
    store.set_marker(G0)
    r = LocalRemote(tmp_path / 'fs', gpu_apps={0: [{'pid': 99, 'name': 'python', 'used_mib': 1000, 'sid': 99}]})
    calls = []
    d = _disp(store, matrix, r, calls)
    started = d.step()
    _join(d)
    assert 'E0-CAL-S1-FR' in started
    node = store.state()['runs']['E0-CAL-S1-FR']['nodes']
    assert node != ['WINH0'] and ('E0-CAL-S1-FR', node) in calls        # lane WINH0 busy -> re-assigned
    assert not list((tmp_path / 'fs' / 'data' / 'infrarsi' / 'leases').glob('*.json'))   # run lease released


def test_claim_refusal_keeps_run_pending(store, matrix, tmp_path):
    store.set_marker(G0)
    r = LocalRemote(tmp_path / 'fs')
    calls = []
    d = _disp(store, matrix, r, calls)
    d.live = False                                                      # stale plan says WINH0 is free
    r.gpu_apps = {0: [{'pid': 5, 'name': 'x', 'used_mib': 1, 'sid': 5}]}
    started = d.step()
    _join(d)
    assert 'E0-CAL-S1-FR' not in started and not any(c[0] == 'E0-CAL-S1-FR' for c in calls)
    rec = store.state()['runs']['E0-CAL-S1-FR']
    assert rec.get('status', 'pending') == 'pending' and 'WINH0' in rec['last_refusal']['report']


def test_search_handler_adapter(store, tmp_path):
    from infrarsi.dispatch.handlers import Context, search_handler
    r = LocalRemote(tmp_path / 'fs')
    r.write_json('WINH0', '/data/infrarsi/receipts/fr.json', {'attempt_id': 'fr'})
    r.write_json('WINH0', '/data/infrarsi/receipts/ed.json', {'attempt_id': 'ed'})
    store.set_reference('D2', ed_config={'tensor_model_parallel_size': 2}, ed_screen='/data/infrarsi/receipts/ed.json',
                        fr_status='feasible', fr_screen='/data/infrarsi/receipts/fr.json',
                        fr_score='/data/infrarsi/receipts/fr_b.json')
    seen = {}

    def fake_search(**kw):
        seen.update(kw)
        return {'ok': True}
    ctx = Context(store=store, remote=r, searches_root=tmp_path / 'searches')
    res = search_handler(fake_search)({'run_id': 'E1-D2-A0', 'task': 'D2', 'arm': 'A0'}, ['WINH2'], ctx)
    assert res['search_dir'] == str(tmp_path / 'searches' / 'E1-D2-A0') and (tmp_path / 'searches' / 'E1-D2-A0').is_dir()
    assert seen['start_config'] == {'tensor_model_parallel_size': 2} and seen['start_receipt']['attempt_id'] == 'ed'
    assert seen['reference_receipt']['attempt_id'] == 'fr' and seen['nodes'] == ['WINH2'] and callable(seen['runner'])
    with pytest.raises(RuntimeError):
        search_handler(fake_search)({'run_id': 'E1-D4-A0', 'task': 'D4', 'arm': 'A0'}, ['WINH2'], ctx)


def test_failed_handler_marks_failed_and_releases(store, matrix, tmp_path):
    store.set_marker(G0)

    def boom(run, nodes, ctx):
        raise RuntimeError('pipeline bug')
    d = Dispatcher(remote=LocalRemote(tmp_path / 'fs'), store=store, matrix=matrix, handler_map={'FR': boom}, log=lambda m: None)
    d.step()
    _join(d)
    time.sleep(0.05)
    rec = store.state()['runs']['E0-CAL-S1-FR']
    assert rec['status'] == 'failed' and 'pipeline bug' in rec['error']
    assert not list((tmp_path / 'fs' / 'data' / 'infrarsi' / 'leases').glob('*.json'))


# ---------------------------------------------------------------- Codex R1 1 and 10: handlers and terminal states

def test_every_matrix_arm_has_a_handler(matrix):
    from infrarsi.dispatch.handlers import handlers, is_deferred
    h = handlers()
    arms = {r['arm'] for r in matrix['runs']}
    assert not [a for a in arms if a not in h]
    assert {a for a in arms if is_deferred(h[a])} == {'FR+ED_at_64_GPUs', 'S_match_1024_quality', 'rebench_llm_foundry'}
    for arm in ('harness_selftest', 'FR_smoke_synthetic', 'lifecycle_promoted'):
        assert not is_deferred(h[arm])


def test_deferred_arms_show_as_deferred_and_never_launch(store, matrix, tmp_path):
    from infrarsi.dispatch.dispatcher import format_plan
    from infrarsi.dispatch.handlers import handlers
    for m in ('P0_CORE_COMPLETE', 'USER_APPROVAL'):
        store.set_marker(m)
    p = plan(matrix, store, include_p2=True, enabled_conditional={'E7-F1-SMATCH-1024'}, handlers=handlers())
    assert set(p['deferred']) == {'P2-T3-PLANNER-64', 'P2-H7-REBENCH', 'E7-F1-SMATCH-1024'}
    assert not set(p['deferred']) & set(ids(p))
    blocked = {b['run_id']: b for b in p['blocked']}
    assert blocked['P2-T3-PLANNER-64']['missing'] == ['deferred (P2, not in v0.1)']
    assert 'deferred (P2, not in v0.1): ' in format_plan(p, matrix) and 'MISSING' not in format_plan(p, matrix)


def test_terminal_states_from_handler_results():
    from infrarsi.dispatch.handlers import terminal_state
    assert terminal_state('FR', {'status': 'feasible'}) == ('done', None)
    assert terminal_state('FR', {'status': 'invalid'})[0] == 'invalid'
    assert terminal_state('FR', {'status': 'infeasible'})[0] == 'infeasible'
    assert terminal_state('ED', {'status': 'infeasible'})[0] == 'infeasible'
    assert terminal_state('FR', {'status': 'error'})[0] == 'failed'
    assert terminal_state('FR', {'status': 'probing'})[0] == 'failed'
    assert terminal_state('FR_lifecycle', {'status': 'fail'})[0] == 'invalid'
    assert terminal_state('FR_lifecycle', {'status': 'pass'})[0] == 'done'
    assert terminal_state('FR_smoke_synthetic', {'status': 'infeasible'})[0] == 'failed'   # only FR / ED may be infeasible
    assert terminal_state('A0', {'search_dir': 'x', 'result': {}})[0] == 'done'
    assert terminal_state('A0', None)[0] == 'failed'


def _run_with(store, matrix, tmp_path, result):
    store.set_marker(G0)

    def fr(run, nodes, ctx):
        return result
    d = Dispatcher(remote=LocalRemote(tmp_path / 'fs'), store=store, matrix=matrix, handler_map={'FR': fr}, log=lambda m: None)
    d.step()
    _join(d)
    return store.state()['runs']['E0-CAL-S1-FR']


def test_invalid_fr_never_satisfies_dependants(store, matrix, tmp_path):
    """Codex R1 10: an FR returning status 'invalid' is recorded as invalid and E1-S1-ED stays blocked."""
    rec = _run_with(store, matrix, tmp_path, {'status': 'invalid', 'fr': {}})
    assert rec['status'] == 'invalid' and 'invalid' in rec['error']
    p = plan(matrix, store)
    assert 'E1-S1-ED' not in ids(p) and 'E0-AGENT-DRYRUN' not in ids(p)
    assert any(b['run_id'] == 'E1-S1-ED' and b['missing'] == ['run E0-CAL-S1-FR is invalid'] for b in p['blocked'])


def test_infeasible_fr_lets_ed_become_the_reference(store, matrix, tmp_path):
    rec = _run_with(store, matrix, tmp_path, {'status': 'infeasible', 'fr': {}})
    assert rec['status'] == 'infeasible'
    assert 'E1-S1-ED' in ids(plan(matrix, store))                    # design 1.6 fallback: ED may run
    store.update_run('E1-S1-ED', status='infeasible')
    from infrarsi.dispatch.queue import dep_ok
    st = store.state()
    assert dep_ok('E1-S1-ED', st, {}, matrix)[0] and dep_ok('E0-CAL-S1-FR', st, {}, matrix)[0]
    store.update_run('E0-SMOKE-S1-SYN', status='infeasible')                       # not an FR / ED run
    assert not dep_ok('E0-SMOKE-S1-SYN', store.state(), {}, matrix)[0]


def test_failed_lifecycle_dependency(store, matrix):
    store.set_marker(G0)
    store.update_run('E0-SMOKE-S1-SYN', status='failed')
    assert 'E0-CAL-S2-FR' not in ids(plan(matrix, store))
    store.update_run('E0-SMOKE-S1-SYN', status='invalid')
    assert 'E0-CAL-S2-FR' not in ids(plan(matrix, store))
    store.update_run('E0-SMOKE-S1-SYN', status='done')
    assert 'E0-CAL-S2-FR' in ids(plan(matrix, store))


def _fake_pytest(failures=0, errors=0, tests=12, rc=None):
    class P:
        def __init__(self, cmd):
            self.returncode = rc if rc is not None else (1 if failures or errors else 0)
            self.stdout = f'{tests - failures - errors} passed, {failures} failed in 1.0s\n'
            self.stderr = ''

    def run_fn(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        junit = cmd[-1].split('=', 1)[1]
        cases = ''.join(f'<testcase classname="t" name="f{i}"><failure message="x"/></testcase>' for i in range(failures))
        with open(junit, 'w', encoding='utf-8') as f:
            f.write(f'<testsuites><testsuite tests="{tests}" failures="{failures}" errors="{errors}" skipped="1">'
                    f'{cases}</testsuite></testsuites>')
        assert cmd[1:6] == ['-m', 'pytest', '-q', '-p', 'no:cacheprovider'] and 'tests/cpu' in cmd
        return P(cmd)
    return run_fn


def test_harness_selftest_records_counts_and_fails_on_any_failure(store, tmp_path):
    from infrarsi.dispatch.handlers import Context, run_selftest, terminal_state
    ctx = Context(store=store)
    out = tmp_path / 'E0' / 'selftest.json'
    run = {'run_id': 'E0-CPU-HARNESS-SELFTEST', 'arm': 'harness_selftest', 'wall_hours_cap': 2}
    res = run_selftest(run, [], ctx, out_path=out, run_fn=_fake_pytest())
    rec = json.loads(out.read_text())
    assert terminal_state('harness_selftest', res) == ('done', None)
    assert rec['counts']['passed'] == 11 and rec['counts']['skipped'] == 1 and rec['returncode'] == 0
    with pytest.raises(RuntimeError, match='selftest failed'):
        run_selftest(run, [], ctx, out_path=out, run_fn=_fake_pytest(failures=2))
    rec = json.loads(out.read_text())
    assert rec['counts']['failed'] == 2 and rec['counts']['failures'] == ['t::f0', 't::f1']
    with pytest.raises(RuntimeError):
        run_selftest(run, [], ctx, out_path=out, run_fn=_fake_pytest(tests=0))          # no test at all
    with pytest.raises(RuntimeError):
        run_selftest(run, [], ctx, out_path=out, run_fn=_fake_pytest(rc=2))             # pytest itself failed


def test_smoke_synthetic_uses_a_distinct_synthetic_task(store, tmp_path, monkeypatch):
    import importlib
    from driver_helpers import HF
    from infrarsi.contracts import TaskSpec
    from infrarsi.discover import workload_facts
    from infrarsi.harness.dataindex import is_synthetic
    H = importlib.import_module('infrarsi.dispatch.handlers')
    monkeypatch.setattr(H, '_facts', lambda task, ctx: workload_facts(task, cache_dir=HF))
    seen = []

    def runner(req, **kw):
        seen.append(req)
        return {'attempt_id': req.attempt_id, 'valid': True, 'harness': {'failure_kind': 'ok'},
                'per_step': [{'update': u, 'max_rank_whole_s': 1.0} for u in range(1, 6)],
                'step_time': {'mean_window_max_rank_s': 1.0, 'rsd_window': 0.0}}
    ctx = H.Context(store=store, runner=runner)
    res = H.run_smoke_synthetic({'run_id': 'E0-SMOKE-S1-SYN', 'task': 'S1', 'wall_hours_cap': 1}, ['WINH0'], ctx)
    assert H.terminal_state('FR_smoke_synthetic', res) == ('done', None)
    real = TaskSpec.load('S1')
    assert res['task_digest'] != real.digest and res['synthetic_data_key'] == 'D-SYN-PT-4K'
    assert all(is_synthetic(r.task) and r.task['id'] == 'S1' and r.run_id == 'E0-SMOKE-S1-SYN' for r in seen)
    assert [r.purpose for r in seen] == ['probe', 'screen'] and seen[-1].input_plan == 'A'
    assert seen[-1].numerics_role == 'reference' and store.references() == {}        # never a task reference


def _promoted_world(tmp_path, store, promoted=True, overlay=True, flag=None, same_as_ref=False):
    import hashlib
    from driver_helpers import make_request, simulated
    from infrarsi.dispatch.handlers import Context
    r = LocalRemote(tmp_path / 'fs')
    cfg = {'tensor_model_parallel_size': 1} if same_as_ref else {'tensor_model_parallel_size': 2, 'micro_batch_size': 2}
    overlay = overlay and not same_as_ref
    fr_b = simulated(tmp_path, make_request('D2', plan='B', purpose='score', run_id='E7-TEST-SEARCH'))
    req = make_request('D2', plan='B', purpose='score', run_id='E7-TEST-SEARCH', numerics_role='candidate', config=cfg)
    files = {'schedule.py': b'# overlay\n'}
    if overlay:
        odir = f'{req.out_dir}/overlay'
        for n, b in files.items():
            r.put_bytes('WINH0', b, f'{odir}/{n}')
        req.overlay = {'dir': odir, 'files': {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}}
    cand_b = simulated(tmp_path, req, reference=fr_b)
    assert cand_b['valid'] and fr_b['valid'], cand_b['invalid_reasons']
    r.write_json('WINH0', f'{req.out_dir}/request.json', req.to_dict())
    sdir = tmp_path / 'searches' / 'E7-TEST-SEARCH'
    (sdir / 'locked').mkdir(parents=True)
    (sdir / 'locked' / 'candidate_plan_b.json').write_text(json.dumps(cand_b))
    (sdir / 'locked' / 'fr_plan_b.json').write_text(json.dumps(fr_b))
    (sdir / 'result.json').write_text(json.dumps({'locked_pair': {
        'promoted': promoted if flag is None else flag, 't_cand_locked': 0.8 if promoted else None, 'reference_kind': 'FR', 'selected': 'c3',
        'candidate': {'attempt_id': cand_b['attempt_id']}, 'reference': {'attempt_id': fr_b['attempt_id']}}}))
    seen = []

    def runner(q, **kw):
        seen.append((q, kw))
        return {'attempt_id': q.attempt_id, 'valid': True, 'harness': {'failure_kind': 'ok'}, 'lifecycle': {'export_ok': True},
                'config_digest': q.config_digest, 'task_digest': cand_b['task_digest'], 'code_digest': cand_b['code_digest'],
                'overlay_digest': cand_b['overlay_digest'] if kw.get('overlay_files') == (files if overlay else None) else 'x'}
    ctx = Context(store=store, remote=r, runner=runner, searches_root=tmp_path / 'searches')
    run = {'run_id': 'E7-TEST-LIFECYCLE', 'task': 'D2', 'arm': 'lifecycle_promoted', 'wall_hours_cap': 4,
           'depends_on': ['E7-TEST-SEARCH']}
    return ctx, run, seen, cand_b, fr_b, files, r, req


def test_lifecycle_promoted_runs_the_exact_promoted_candidate(store, tmp_path):
    from infrarsi.dispatch.handlers import run_lifecycle_promoted, terminal_state
    ctx, run, seen, cand_b, fr_b, files, r, req = _promoted_world(tmp_path, store)
    res = run_lifecycle_promoted(run, ['WINH2'], ctx)
    ((q, kw),) = seen
    assert terminal_state('lifecycle_promoted', res) == ('done', None) and res['promoted'] and not res['unbound']
    assert q.purpose == 'lifecycle' and q.input_plan == 'B' and q.numerics_role == 'candidate'
    assert q.config_digest == cand_b['config_digest'] and kw['overlay_files'] == files
    assert kw['reference'] is not None and kw['reference']['attempt_id'] == fr_b['attempt_id']
    assert q.reference_receipt.endswith(f'{fr_b["attempt_id"]}.json') and q.lifecycle['save_at'] == 30


def test_lifecycle_promoted_refuses_unbound_candidates(store, tmp_path):
    from infrarsi.dispatch.handlers import run_lifecycle_promoted
    ctx, run, seen, cand_b, fr_b, files, r, req = _promoted_world(tmp_path, store)
    r.put_bytes('WINH0', b'# changed\n', f'{req.out_dir}/overlay/schedule.py')        # overlay edited after the locked run
    with pytest.raises(RuntimeError, match='overlay file schedule.py'):
        run_lifecycle_promoted(run, ['WINH2'], ctx)
    ctx2, run2, seen2, cand2, *_ = _promoted_world(tmp_path / 'b', store)
    bad = dict(cand2, valid=False)                                                  # unsigned / invalid locked receipt
    (tmp_path / 'b' / 'searches' / 'E7-TEST-SEARCH' / 'locked' / 'candidate_plan_b.json').write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match='cannot be bound'):
        run_lifecycle_promoted(run2, ['WINH2'], ctx2)
    assert seen == [] and seen2 == []


def test_lifecycle_promoted_without_promotion_is_a_no_op(store, tmp_path):
    from infrarsi.dispatch.handlers import run_lifecycle_promoted, terminal_state
    ctx, run, seen, *_ = _promoted_world(tmp_path, store, promoted=False)
    res = run_lifecycle_promoted(run, ['WINH2'], ctx)
    assert res['promoted'] is False and terminal_state('lifecycle_promoted', res) == ('done', None) and seen == []


def test_lifecycle_promoted_follows_the_receipts_not_the_flag(store, tmp_path):
    """Codex R2 (c0): a verified locked candidate that differs from the reference is a promotion even when the search
    recorded promoted=False; a candidate identical to the reference needs no lifecycle."""
    from infrarsi.dispatch.handlers import run_lifecycle_promoted
    ctx, run, seen, cand_b, fr_b, files, r, req = _promoted_world(tmp_path, store, flag=False)
    res = run_lifecycle_promoted(run, ['WINH2'], ctx)
    assert res['promoted'] and res['search_promoted_flag'] is False and len(seen) == 1
    ctx2, run2, seen2, *_ = _promoted_world(tmp_path / 'same', store, same_as_ref=True, flag=True)
    res2 = run_lifecycle_promoted(run2, ['WINH2'], ctx2)
    assert res2['promoted'] is False and 'reference itself' in res2['note'] and seen2 == []
