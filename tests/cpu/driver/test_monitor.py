"""Monitor rules (design 5.2) on synthetic states, escalation side effects, and poll() against LocalRemote."""
import datetime as dt
import json

from driver_helpers import DUMMY_MOD, make_request

from infrarsi.dispatch import QueueStore, load_matrix
from infrarsi.harness.remote import LocalRemote
from infrarsi.monitor import escalate, evaluate, poll

NOW = '2026-10-21T04:00:00+00:00'


def run(**kw):
    r = {'run_id': 'E1-D2-A0', 'attempt_id': 'a1', 'purpose': 'screen', 'task_id': 'D2', 'task_gpus': 8,
         'task_family': 'qwen3-dense', 'nodes': ['WINH2'], 'world_size': 8, 'steps': 30, 'wall_seconds_cap': 3600,
         'phase': 'train', 'step': 12, 'heartbeat_age_s': 30.0, 'elapsed_s': 900.0, 'finished': False,
         'last_step_time_s': 60.0, 'log_tails': [''], 'pids': [], 'receipt_valid': None}
    r.update(kw)
    return r


def state(runs=(), nodes=None, disk=68e12, **kw):
    s = {'polled_utc': NOW, 'runs': list(runs), 'errors': [], 'disk': {'free_bytes': disk},
         'nodes': nodes or {f'WINH{k}': {'reachable': True, 'compute_apps': [], 'foreign_apps': [], 'lease': None,
                                         'hostname': f'h{k}'} for k in range(8)},
         'queue': {'runnable': [], 'statuses': {}}, 'downloads': {}, 'download_logs': {}, 'ledger': {}}
    s.update(kw)
    return s


def codes(alerts, level=None):
    return {a['code'] for a in alerts if level is None or a['level'] == level}


def test_heartbeat_thresholds_and_stop_proposal():
    a = evaluate(state([run(heartbeat_age_s=700.0, last_step_time_s=10.0)]))
    assert 'heartbeat_stale' in codes(a, 'WARN')
    a = evaluate(state([run(heartbeat_age_s=400.0, last_step_time_s=100.0)]))   # 5 x step = 500 s > 400 s: fine
    assert 'heartbeat_stale' not in codes(a)
    a = evaluate(state([run(heartbeat_age_s=1900.0)]))
    crit = [x for x in a if x['code'] == 'heartbeat_lost']
    assert crit and crit[0]['level'] == 'CRIT' and crit[0]['stop_proposal']['attempt_id'] == 'a1'


def test_first_step_limits():
    a = evaluate(state([run(step=0, phase='load', elapsed_s=26 * 60)]))
    assert 'first_step_slow' in codes(a, 'WARN')
    a = evaluate(state([run(step=0, phase='build', elapsed_s=26 * 60, task_family='qwen3-moe')]))
    assert 'first_step_slow' not in codes(a)                                   # MoE limit is 40 min
    a = evaluate(state([run(step=0, phase='load', elapsed_s=125 * 60, task_gpus=32)]))
    assert 'first_step_missing' in codes(a, 'CRIT')


def test_log_keywords():
    oom = 'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB'
    a = evaluate(state([run(purpose='probe', log_tails=[oom])]))
    assert {x['level'] for x in a if x['code'] == 'log_oom'} == {'INFO'}
    a = evaluate(state([run(log_tails=[oom])]))
    assert {x['level'] for x in a if x['code'] == 'log_oom'} == {'CRIT'}
    a = evaluate(state([run(log_tails=['[rank3]: Watchdog caught collective operation timeout: WorkNCCL(...)'])]))
    assert 'log_nccl' in codes(a, 'CRIT')
    a = evaluate(state([run(log_tails=['NCCL INFO Channel 00/32 : 0[0] -> 1[1] via P2P/CUMEM'])]))
    assert 'log_nccl' not in codes(a)                                          # normal NCCL chatter is not an error
    a = evaluate(state([run(log_tails=['Traceback (most recent call last):\n  File "x"'])]))
    assert 'log_traceback' in codes(a, 'CRIT')
    a = evaluate(state([run(purpose='probe', log_tails=['Traceback (most recent call last):\n' + oom])]))
    assert {x['level'] for x in a if x['code'] in ('log_oom', 'log_traceback')} == {'INFO'}


def test_probe_oom_from_receipt_downgrades_launcher_traceback():
    tb = 'Traceback (most recent call last): ... ChildFailedError: failed (exitcode: 3)'
    a = evaluate(state([run(purpose='probe', finished=True, receipt_oom=True, log_tails=[tb])]))
    assert {x['level'] for x in a if x['code'] == 'log_traceback'} == {'INFO'}
    a = evaluate(state([run(purpose='screen', finished=True, receipt_oom=True, log_tails=[tb])]))
    assert {x['level'] for x in a if x['code'] == 'log_traceback'} == {'CRIT'}


def test_unmanaged_attempt_skips_liveness():
    a = evaluate(state([run(heartbeat_age_s=5000.0, launched_by_harness=False)]))
    assert codes(a) == {'unmanaged_attempt'} and not any(x.get('stop_proposal') for x in a)


def test_foreign_process_on_leased_node():
    nodes = {f'WINH{k}': {'reachable': True, 'compute_apps': [], 'foreign_apps': [], 'lease': None, 'hostname': f'h{k}'} for k in range(8)}
    nodes['WINH3'].update(foreign_apps=[{'pid': 1234}], compute_apps=[{'pid': 1234}], lease={'run_id': 'E1-D3-A0'})
    nodes['WINH6'].update(foreign_apps=[{'pid': 77}], compute_apps=[{'pid': 77}])
    a = evaluate(state(nodes=nodes))
    lvl = {x['key']: x['level'] for x in a}
    assert lvl['foreign:WINH3'] == 'CRIT' and lvl['foreign:WINH6'] == 'INFO'


def test_disk_and_budget():
    assert 'disk_low' in codes(evaluate(state(disk=2.5e12)), 'WARN')
    assert 'disk_low' in codes(evaluate(state(disk=0.5e12)), 'CRIT')
    m = load_matrix()                                                          # E1-D2-A0 budget 20 GPUh
    a = evaluate(state([run(elapsed_s=2.4 * 3600, world_size=8)]), matrix=m)   # 19.2 GPUh = 96 %
    assert 'budget_gpuh' in codes(a, 'WARN')
    a = evaluate(state([run(elapsed_s=3.0 * 3600, world_size=8)]), matrix=m)   # 24 GPUh = 120 %
    assert 'budget_gpuh' in codes(a, 'CRIT')
    a = evaluate(state([run(elapsed_s=4500, wall_seconds_cap=3600)]))
    assert any(x['code'] == 'attempt_over_wall' and x.get('stop_proposal') for x in a)


def test_download_stall():
    prev = state(polled_utc='2026-10-21T03:30:00+00:00', downloads={'Qwen__X': {'bytes': 100, 'done': None}})
    cur = state(downloads={'Qwen__X': {'bytes': 100, 'done': None}})
    assert 'download_stalled' in codes(evaluate(cur, prev), 'WARN')
    cur = state(downloads={'Qwen__X': {'bytes': 200, 'done': None}})
    assert 'download_stalled' not in codes(evaluate(cur, prev))


def test_validity_rules():
    led = {'ledger.jsonl': {'tail': [json.dumps({'accepted': True, 'attempt_id': 'a1'})]}}
    a = evaluate(state([run(finished=True, receipt_valid=False)], ledger=led))
    assert 'invalid_accepted' in codes(a, 'CRIT')
    a = evaluate(state(packages={'/data/infrarsi/packages/A_R': {'frozen': 'x', 'current': 'y'}}))
    assert 'package_changed' in codes(a, 'CRIT')


def test_lane_behind_schedule():
    m = load_matrix()
    late = dt.datetime(2026, 10, 22, 5, 0, tzinfo=dt.timezone(dt.timedelta(hours=8)))
    a = evaluate(state(), matrix=m, now=late)
    assert any(x['code'] == 'lane_behind' and 'WINH2' in x['key'] for x in a)
    a = evaluate(state(), matrix=m, now=dt.datetime(2026, 10, 21, 12, 0, tzinfo=dt.timezone(dt.timedelta(hours=8))))
    assert 'lane_behind' not in codes(a)
    # terminal outcomes that need no intervention keep a lane on schedule; invalid / failed runs do not
    due = [r['run_id'] for r in m['runs'] if str(r['node_lane']) == 'WINH2' and r['priority'][:2] == 'P0'
           and str(r['planned_day']) == 'D1']
    states = {rid: s for rid, s in zip(due, ['done', 'infeasible', 'deferred'] * len(due))}
    a = evaluate(state(queue={'runnable': [], 'statuses': states}), matrix=m, now=late)
    assert not any(x['code'] == 'lane_behind' and x['key'] == 'behind:WINH2' for x in a)
    states[due[0]] = 'invalid'
    a = evaluate(state(queue={'runnable': [], 'statuses': states}), matrix=m, now=late)
    assert any(x['code'] == 'lane_behind' and x['key'] == 'behind:WINH2' for x in a)


def fin(att, valid, created, **kw):
    return run(attempt_id=att, finished=True, receipt_valid=valid, created_utc=created, **kw)


def test_invalid_finishes_warn_and_streak_crit():
    a = evaluate(state([fin('a1', False, '2026-10-21T01:00:00+00:00')]))
    assert [x['level'] for x in a if x['code'] == 'invalid_finish'] == ['WARN'] and 'invalid_streak' not in codes(a)
    two = [fin('a1', False, '2026-10-21T01:00:00+00:00'), fin('a2', False, '2026-10-21T02:00:00+00:00')]
    a = evaluate(state(two))
    crit = [x for x in a if x['code'] == 'invalid_streak']
    assert crit and crit[0]['level'] == 'CRIT' and crit[0]['lanes'] == ['WINH2']
    healed = two + [fin('a3', True, '2026-10-21T03:00:00+00:00')]
    assert 'invalid_streak' not in codes(evaluate(state(healed)))
    probes = [fin('p1', False, '2026-10-21T01:00:00+00:00', purpose='probe', receipt_oom=True),
              fin('p2', False, '2026-10-21T02:00:00+00:00', purpose='probe', receipt_oom=True)]
    a = evaluate(state(probes))
    assert 'invalid_streak' not in codes(a) and {x['level'] for x in a if x['code'] == 'invalid_finish'} == {'INFO'}
    red = [fin('r1', False, 't1', run_id='E4-S2-SEEDS-14x2'), fin('r2', False, 't2', run_id='E4-S2-SEEDS-14x2')]
    assert 'invalid_streak' not in codes(evaluate(state(red)))


def test_no_progress_with_fresh_heartbeat():
    prev = state([run(step=12, heartbeat_age_s=20.0, last_step_time_s=60.0)], polled_utc='2026-10-21T03:30:00+00:00')
    prev['runs'][0]['step_since_utc'] = '2026-10-21T03:30:00+00:00'
    cur = state([run(step=12, heartbeat_age_s=20.0, last_step_time_s=60.0)])         # 30 min later, same step
    a = evaluate(cur, prev)
    assert 'no_progress' in codes(a, 'WARN')                    # limit max(2*60 s + 10 min, 15 min) = 15 min
    slow = state([run(step=12, heartbeat_age_s=20.0, last_step_time_s=600.0)])      # limit 2*10 + 10 = 30 min
    prev['runs'][0]['step_since_utc'] = '2026-10-21T03:35:00+00:00'
    assert 'no_progress' not in codes(evaluate(slow, prev))
    moved = state([run(step=13, heartbeat_age_s=20.0)])
    assert 'no_progress' not in codes(evaluate(moved, prev)) and moved['runs'][0]['step_since_utc'] == NOW


def test_llm_spend_rules(tmp_path):
    from infrarsi.monitor.poll import llm_spend
    root = tmp_path / 'searches'
    (root / 'E1-D2-A0').mkdir(parents=True)
    (root / 'E1-D2-A0' / 'trace.json').write_text(json.dumps({'used': {'llm_usd': 700.0, 'llm_tokens': 10},
                                                              'accounting_complete': True}))
    (root / 'E1-D1-A0').mkdir()
    ev = [{'kind': 'llm_call', 'utc': 'x', 'payload': {'llm_usd': 300.0, 'llm_tokens': 5, 'accounting_complete': True}},
          {'kind': 'llm_call', 'utc': 'y', 'payload': {'llm_usd': 250.0, 'llm_tokens': 5, 'accounting_complete': False}}]
    (root / 'E1-D1-A0' / 'events.jsonl').write_text('\n'.join(json.dumps(e) for e in ev))
    s = llm_spend(root)
    assert s['total_usd'] == 1250.0 and s['incomplete'] == ['E1-D1-A0']
    a = evaluate(state(llm_spend=s), matrix=load_matrix())
    assert 'llm_spend_p1_stop' in codes(a, 'WARN') and 'llm_accounting_incomplete' in codes(a, 'WARN')
    s['total_usd'] = 1500.0
    assert 'llm_spend_cap' in codes(evaluate(state(llm_spend=s), matrix=load_matrix()), 'CRIT')


def test_lane_scoped_pause(tmp_path):
    from infrarsi.dispatch.queue import paused_nodes
    store = QueueStore(tmp_path / 'queue')
    two = [fin('a1', False, '2026-10-21T01:00:00+00:00'), fin('a2', False, '2026-10-21T02:00:00+00:00')]
    escalate(evaluate(state(two)), {}, alerts_dir=tmp_path / 'alerts', store=store, now=NOW)
    assert paused_nodes(store.paused()) == {2}                  # only the failing run's lane
    escalate(evaluate(state(disk=0.5e12)), {}, alerts_dir=tmp_path / 'alerts', store=store, now=NOW)
    assert paused_nodes(store.paused()) == set()                # a CRIT without lanes pauses everything


def test_escalate_files_pause_dedup(tmp_path):
    store = QueueStore(tmp_path / 'queue')
    alerts = evaluate(state([run(heartbeat_age_s=1900.0)], disk=2e12))
    res = escalate(alerts, {}, alerts_dir=tmp_path / 'alerts', store=store, now=NOW)
    assert any(f.endswith('.md') and 'CRIT_' in f for f in res['files'])
    assert store.paused() and 'heartbeat' in store.paused()['reasons'][0]['reason']
    prop = json.loads((tmp_path / 'alerts' / 'STOP_REQUEST_a1.json').read_text())
    assert prop['status'] == 'proposed' and 'confirm-stop' in prop['confirm_with']
    assert (tmp_path / 'alerts' / 'SUMMARY.md').read_text().count('CRIT') >= 1
    again = escalate(alerts, {}, alerts_dir=tmp_path / 'alerts', store=store, now='2026-10-21T04:05:00+00:00')
    assert again['emitted'] == []                                              # within cooldown: nothing re-emitted
    later = escalate(alerts, {}, alerts_dir=tmp_path / 'alerts', store=store, now='2026-10-21T04:20:00+00:00')
    assert {a['code'] for a in later['emitted']} == {'heartbeat_lost'}          # CRIT cooldown 15 min, WARN 30 min
    resolved = escalate([], {}, alerts_dir=tmp_path / 'alerts', store=store, now='2026-10-21T04:25:00+00:00')
    assert resolved['open'] == []


def test_poll_on_local_remote(tmp_path):
    r = LocalRemote(tmp_path / 'fs')
    req = make_request('D2', nodes=['WINH2'])
    out = r.local(req.out_dir)
    out.mkdir(parents=True)
    (out / 'request.json').write_text(json.dumps(req.to_dict()))
    DUMMY_MOD.simulate_attempt(req.to_dict(), str(out))
    (out / 'exit_node0.json').unlink()                                         # still running
    (out / 'pid_node0.json').write_text(json.dumps({'pid': 4242, 'pgid': 4242, 'hostname': 'fake-winh2'}))
    (out / 'launch').mkdir()
    (out / 'launch' / 'node0.json').write_text('{}')
    r.gpu_apps = {2: [{'pid': 4300, 'name': 'python', 'used_mib': 1000, 'sid': 4242}],
                  5: [{'pid': 999, 'name': 'python', 'used_mib': 1000, 'sid': 999}],
                  6: [{'pid': 55, 'name': 'python', 'used_mib': 1000, 'sid': 55, 'cmdline': 'python -m infrarsi.adapters.x'}]}
    lease = r.local('/data/infrarsi/leases/WINH6.json')
    lease.parent.mkdir(parents=True, exist_ok=True)
    lease.write_text(json.dumps({'owner': 'infrarsi', 'run_id': 'dev-run', 'attempt_id': 'x'}))
    st = poll(r, store=QueueStore(tmp_path / 'queue'), stagger_s=0)
    assert st['nodes']['WINH6']['foreign_apps'] == [] and st['runs'][0]['launched_by_harness']
    assert [x['attempt_id'] for x in st['runs']] == [req.attempt_id] and st['runs'][0]['finished'] is False
    assert st['nodes']['WINH2']['foreign_apps'] == [] and st['nodes']['WINH2']['compute_apps'][0]['owner']
    assert [a['pid'] for a in st['nodes']['WINH5']['foreign_apps']] == [999]
    assert st['queue']['matrix_total'] == 134 and st['disk']['free_bytes'] > 0
    json.dumps(st)                                                              # machine-readable
