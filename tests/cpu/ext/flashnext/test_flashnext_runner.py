"""FlashNextRunner on four simulated nodes (LocalNodeOps sharing one /data): all-or-nothing lease, peer failure, kill
on every node, wall cap / STOP, surviving processes, zero-cost refusals, admission. AMENDMENT_EXT_003 G-F1 (3)."""
import json

import pytest

from fn_helpers import Cluster, judge_ok, runner, task
from infrarsi.ext.common.remote_exec import NodeError
from infrarsi.ext.flashnext import runner as R
from infrarsi.ext.flashnext import switch_space as S
from infrarsi.harness import signing

START = S.start_point()
SMATCH = S.registered_points()['S_match']


def _run(r, point=SMATCH, **kw):
    return r.run(task(), point, None, 'screen', 'fntest', candidate_id='c1', **kw)


def test_success_kills_on_every_node_and_releases_all_leases(tmp_path):
    c = Cluster(tmp_path / 'data')
    rec = _run(runner(c, tmp_path))
    assert rec['valid'] is True and rec['metric']['value'] == 12.5, rec['invalid_reasons']
    assert signing.verify(rec) and rec['switch_space_sha256'] == S.sha256() and rec['point'] == SMATCH
    assert sorted(n for n, _ in c.launched) == sorted(c.nodes) and sorted(r for _, r in c.launched) == [0, 1, 2, 3]
    assert {n for n, _ in c.kills} == set(c.nodes)                        # token killed on all four nodes
    assert c.leases() == []                                               # every node lease released
    assert rec['world_size'] == 32 and rec['cost']['accounting_complete'] is True
    assert rec['cost']['gpu_seconds'] == pytest.approx(32 * rec['phases']['held_s'], rel=1e-6)
    assert rec['cost']['open_ended_nodes'] == [] and set(rec['cost']['by_node']) == set(c.nodes)
    assert {s['state'] for s in rec['release_by_node'].values()} == {'removed'} and rec['held_nodes'] == []
    assert rec['deviations'] == list(R.DEVIATIONS) and rec['nodes'] == list(R.DEFAULT_NODES)
    local = tmp_path / 'out' / 'attempts'
    attempt = next(local.iterdir())
    assert len(list((attempt / 'signals').glob('rank*.jsonl'))) == 32
    assert len(list((attempt / 'nodes').glob('node*.exit.json'))) == 4
    spec = json.loads((attempt / 'fn_spec.json').read_text())
    assert spec['nodes'] == list(R.DEFAULT_NODES) and spec['master_addr'] == R.IPS[0] and spec['timeout_s'] == 1800.0
    assert spec['stop_reserve_s'] == R.STOP_RESERVE_S
    # every node swept once (clean at the first round) with the launcher's session and the job's pid file
    assert sorted((n, p.rsplit('/', 1)[1], g) for n, _, p, g in c.sweeps) == sorted(
        (n, f'node{r}.pid.json', [3000 + r]) for r, n in enumerate(c.nodes))


@pytest.mark.parametrize('blocker', ['foreign_process', 'foreign_lease'])
def test_lease_is_all_or_nothing(tmp_path, blocker):
    c = Cluster(tmp_path / 'data')
    if blocker == 'foreign_process':
        c.apps['WINH3'] = [{'pid': 77, 'gpu_index': 5, 'cwd': '/data/tyc/other', 'cmdline': 'python other.py', 'token': None}]
    else:
        c.write('/data/infrarsi/leases/WINH4.json', {'owner': 'someone', 'run_id': 'x', 'attempt_id': 'y'})
    rec = _run(runner(c, tmp_path))
    assert rec['valid'] is False and rec['invalid_reasons'][0].startswith('occupied')
    assert c.launched == []                                               # nothing started anywhere
    expected = ['WINH4.json'] if blocker == 'foreign_lease' else []
    assert c.leases() == expected                                         # no lease of ours left behind
    assert rec['cost']['gpu_seconds'] == 0.0


def test_lease_rollback_when_a_later_lease_is_taken_concurrently(tmp_path):
    c = Cluster(tmp_path / 'data')
    r = runner(c, tmp_path)
    original = c.ops['WINH3'].call

    def racing(func, _timeout=300, **kw):                                 # someone takes WINH3 between probe and acquire
        if func == 'acquire':
            c.write(kw['path'], {'owner': 'racer', 'run_id': 'r', 'attempt_id': 'z'})
        return original(func, _timeout, **kw)
    c.ops['WINH3'].call = racing
    rec = _run(r)
    assert rec['valid'] is False and 'occupied' in rec['invalid_reasons'][0]
    assert c.leases() == ['WINH3.json'] and c.launched == []             # WINH0/WINH1 leases rolled back
    assert {n: s['state'] for n, s in rec['release_by_node'].items()} == {'WINH0': 'removed', 'WINH1': 'removed'}
    assert json.loads(c.local('/data/infrarsi/leases/WINH3.json').read_text())['owner'] == 'racer'   # never touched
    assert rec['cost']['accounting_complete'] is True and rec['held_nodes'] == []


def test_peer_failure_is_invalid_and_still_cleans_up(tmp_path):
    c = Cluster(tmp_path / 'data')
    c.exit_codes.update({1: 1, 0: 125, 2: 125, 3: 125})
    rec = _run(runner(c, tmp_path))
    assert rec['valid'] is False
    assert any(g['id'] == 'complete_verified' for g in rec['guard_rejections'])
    assert {n for n, _ in c.kills} == set(c.nodes) and c.leases() == []


def test_cap_writes_stop_and_kills_everywhere(tmp_path):
    c = Cluster(tmp_path / 'data')
    c.hang = True                                                         # no launcher ever exits
    r = runner(c, tmp_path, poll_s=30.0)
    rec = _run(r, wall_cap=900)
    assert rec['cap_s'] == 900.0
    out = next(p for p in (tmp_path / 'data/data/infrarsi/ext/runs/flashnext/runs').iterdir())
    assert (out / 'STOP.json').exists()
    assert rec['valid'] is False and {n for n, _ in c.kills} == set(c.nodes) and c.leases() == []
    assert any(g['id'] == 'runner_timeout' for g in rec['guard_rejections'] + [{'id': x.get('id')} for x in rec['guard_reports']])
    held = rec['phases']['held_s']
    assert 900 + r.stop_margin_s <= held <= 900 + r.stop_margin_s + r.screen_grace_s + r.poll_s + 1
    assert rec['cost']['gpu_seconds'] == pytest.approx(32 * held)


def _held(c, r, rec, node):
    """A node that could not be proven clean: it stays leased, its cost is open-ended, and the runner refuses more."""
    assert rec['valid'] is False and any(g['id'] == 'process_cleanup' for g in rec['guard_rejections'])
    assert c.leases() == [f'{node}.json'] and r.held_nodes == [node] and rec['held_nodes'] == [node]
    assert rec['release_by_node'][node]['state'] == 'kept'
    assert {s['state'] for n, s in rec['release_by_node'].items() if n != node} == {'removed'}
    assert rec['cost']['accounting_complete'] is False and rec['cost']['open_ended_nodes'] == [node]
    assert sum(1 for n, *_ in c.sweeps if n == node) == len(R.FlashNextRunner.KILL_WAITS)   # every round tried
    nxt = _run(r)
    assert nxt['not_launched'] == 'nodes_held' and nxt['cost']['gpu_seconds'] == 0.0 and len(c.launched) == 4


def test_a_surviving_process_keeps_its_node_leased(tmp_path):
    c = Cluster(tmp_path / 'data')
    c.survivors['WINH1'] = [{'pid': 99, 'gpu_index': 2, 'cwd': R.V.FN_ROOT + '/runs/whatever', 'token': None}]
    r = runner(c, tmp_path)
    _held(c, r, _run(r), 'WINH1')


@pytest.mark.parametrize('kind', ['token_not_on_gpu', 'group_member_without_token', 'sweep_cannot_run', 'probe_error'])
def test_unproven_cleanup_keeps_the_node_leased(tmp_path, kind):
    """Codex HIGH: GPU-invisible processes with the token, live members of our process groups, and a check that cannot
    run all keep the node leased (fail closed)."""
    c = Cluster(tmp_path / 'data')
    if kind == 'token_not_on_gpu':
        c.token_survivors['WINH3'] = [5555]
    elif kind == 'group_member_without_token':
        c.group_survivors['WINH3'] = {4002: [4100]}                 # WINH3 is rank 2: job group 4002
    elif kind == 'sweep_cannot_run':
        c.sweep_fail.add('WINH3')
    else:
        c.probe_error_after_launch['WINH3'] = 'nvidia-smi: NVML driver/library version mismatch'
    r = runner(c, tmp_path)
    _held(c, r, _run(r), 'WINH3')


def test_release_is_retried_and_an_unproven_release_stays_held(tmp_path):
    """Codex HIGH: release_status retried with backoff; a release that cannot be proven keeps the node in held_nodes,
    vetoes the screen (lease_release) and leaves its cost open-ended; cost is per node, acquisition -> release."""
    c = Cluster(tmp_path / 'data')
    c.release_failures.update({'WINH1': 2, 'WINH4': 99})
    r = runner(c, tmp_path)
    rec = _run(r)
    rel = rec['release_by_node']
    assert rel['WINH1']['state'] == 'removed' and rel['WINH4']['state'] == 'unresolved' and 'transport' in rel['WINH4']['error']
    assert c.leases() == ['WINH4.json'] and rec['held_nodes'] == ['WINH4']
    assert rec['valid'] is False and any(g['id'] == 'lease_release' for g in rec['guard_rejections'])
    cost = rec['cost']
    assert cost['accounting_complete'] is False and cost['open_ended_nodes'] == ['WINH4']
    assert cost['gpu_seconds'] == pytest.approx(8 * sum(v['held_s'] for v in cost['by_node'].values()), abs=0.05)
    # released in parallel (fake sleeps are instant, so per-node times are not compared); WINH4 is charged to the end
    held = {n: v['held_s'] for n, v in cost['by_node'].items()}
    assert held['WINH4'] >= max(held[n] for n in ('WINH0', 'WINH1', 'WINH3'))


def test_acquire_transport_failure_is_rolled_back(tmp_path):
    """A lease created on the node whose acquire call then failed in transport is still released (release_status)."""
    c = Cluster(tmp_path / 'data')
    original = c.ops['WINH3'].call

    def lost_reply(func, _timeout=300, **kw):
        res = original(func, _timeout, **kw)
        if func == 'acquire':
            raise NodeError('WINH3 acquire: simulated reply lost after the lease was created')
        return res
    c.ops['WINH3'].call = lost_reply
    rec = _run(runner(c, tmp_path))
    assert rec['valid'] is False and c.launched == [] and c.leases() == []
    assert {n: s['state'] for n, s in rec['release_by_node'].items()} == {'WINH0': 'removed', 'WINH1': 'removed',
                                                                          'WINH3': 'removed'}
    assert rec['cost']['accounting_complete'] is True and rec['held_nodes'] == []


@pytest.mark.parametrize('purpose,recipe,asked,want', [
    ('screen', 'U_plus_C', 3600, 1800),        # a search screen never exceeds 1,800 s, whatever the caller asks
    ('locked', 'S_match', None, 1800),
    ('reference', 'U_plus_C', 3600, 3600),     # the U+C reference screen, cold cache
    ('reference', 'S_match', 3600, 3600),      # the F-HT (= S_match) reference screen, cold cache
    ('reference', 'B', 3600, 1800),            # any other point is no reference screen
    ('reference', 'U_plus_C', None, 1800)])    # 3,600 s only when asked for
def test_cap_is_1800_except_the_two_cold_reference_screens(tmp_path, purpose, recipe, asked, want):
    c = Cluster(tmp_path / 'data')
    r = runner(c, tmp_path)
    kw = {'cap_s': asked} if asked else {}
    rec = r.run(task(), S.registered_points()[recipe], None, purpose, 'fntest', candidate_id='c1', **kw)
    assert rec['valid'] is True and rec['cap_s'] == want and rec['cap']['limit_s'] in (1800.0, 3600.0)
    spec = json.loads(next((tmp_path / 'out' / 'attempts').iterdir()).joinpath('fn_spec.json').read_text())
    assert spec['timeout_s'] == want and spec['stop_reserve_s'] == R.STOP_RESERVE_S
    assert r.run(task(), S.registered_points()[recipe], None, purpose, 'fntest', candidate_id='c2', wall_cap=700,
                 **kw)['cap_s'] == 700.0                                # never more than the admitted wall cap


def test_cap_below_the_minimum_is_not_launched(tmp_path):
    c = Cluster(tmp_path / 'data')
    rec = _run(runner(c, tmp_path), wall_cap=R.MIN_CAP_S - 1)
    assert rec['not_launched'] == 'cap_too_small' and rec['cost']['gpu_seconds'] == 0.0 and c.launched == []


def test_sweep_command_roundtrip_and_script_compiles():
    cmd = R.sweep_command('isxf-0123456789abcdef', R.V.FN_ROOT + '/runs/x/node2.pid.json', 3002)
    assert R.sweep_args(cmd) == ('isxf-0123456789abcdef', R.V.FN_ROOT + '/runs/x/node2.pid.json', 3002)
    assert R.sweep_args(R.sweep_command('isxf-0123456789abcdef', None, 1)) == ('isxf-0123456789abcdef', '-', 0)
    with pytest.raises(ValueError):
        R.sweep_command('isxf-0123456789abcdef', '/etc/passwd')
    compile(R.SWEEP_PY, 'sweep', 'exec')


def test_node_script_is_idempotent_and_runs_the_deployed_launcher():
    s = R.node_script(R.V.FN_SRC, '/data/infrarsi/ext/runs/flashnext/metadata/x/job.json', 2,
                      '/data/infrarsi/ext/runs/flashnext/metadata/x/fn_spec.json')
    lines = s.strip().splitlines()
    assert lines[2].startswith('mkdir /data/infrarsi/ext/runs/flashnext/metadata/x/.launched2 2>/dev/null ||')
    assert 'exit 0' in lines[2] and lines[3].startswith(f'exec /usr/bin/python3 -B {R.V.FN_SRC}/node/{R.LAUNCHER} ')


def test_screen_grace_is_the_sum_of_the_enforced_budgets():
    assert R.FlashNextRunner.screen_grace_s == 60.0 + R.CLEANUP_BUDGET_S + R.RELEASE_BUDGET_S + R.LAUNCH_ALLOWANCE_S == 240.0
    assert R.FlashNextRunner.stop_grace_s == 0.0                          # SearchSession's pair reserve: none (Q5)


@pytest.mark.parametrize('what', ['sweep', 'gpu_probe', 'release_status'])
def test_a_hanging_node_call_is_bounded_by_the_deadlines(tmp_path, monkeypatch, what):
    """Codex rounds 2-3: a hanging node call ends with its own timeout, capped by the time left before the cleanup /
    release deadline (no new round or attempt after it, no abandoned call); the other nodes are verified and released
    in parallel; the hanging node stays leased and its cost open-ended."""
    import threading
    import time as _time
    monkeypatch.setattr(R, 'CLEANUP_BUDGET_S', 0.5)
    monkeypatch.setattr(R, 'RELEASE_BUDGET_S', 0.5)
    c = Cluster(tmp_path / 'data')
    gate = threading.Event()
    c.block[('WINH4', what)] = gate
    r = runner(c, tmp_path)
    t0 = _time.monotonic()
    try:
        rec = _run(r)
    finally:
        gate.set()
    assert _time.monotonic() - t0 < 30
    assert rec['valid'] is False and r.held_nodes == ['WINH4'] and c.leases() == ['WINH4.json']
    state = rec['release_by_node']['WINH4']
    want = 'unresolved' if what == 'release_status' else 'kept'
    assert state['state'] == want and 'simulated hang' in str(state.get('reason') or state.get('error'))
    assert {s['state'] for n, s in rec['release_by_node'].items() if n != 'WINH4'} == {'removed'}
    assert rec['cost']['open_ended_nodes'] == ['WINH4'] and rec['cost']['accounting_complete'] is False


def test_points_outside_the_space_cost_nothing(tmp_path):
    c = Cluster(tmp_path / 'data')
    built = []
    r = runner(c, tmp_path, build_fn=lambda *a, **k: built.append(1))
    for bad in ({'implementation_stack': 'systems_policy'}, dict(START, aux_readback_mode='deferred'),
                dict(SMATCH, retain_attention_prefix=False)):
        rec = _run(r, point=bad)
        assert rec['valid'] is False and rec['not_launched'] == 'invalid_proposal'
        assert rec['cost'] == {'gpu_seconds': 0.0, 'wall_seconds': 0.0, 'accounting_complete': True,
                               'basis': 'refused by the S_F1 validator before any build or lease'}
    assert built == [] and c.launched == [] and c.leases() == []


def test_admission_interface():
    r = object.__new__(R.FlashNextRunner)
    r.cap_s = 1800.0
    assert r.screen_grace_s == 240.0 and r.stop_grace_s == 0.0
    assert r.worst_case_gpu_hours(32) == pytest.approx(32 * 2040 / 3600)
    assert r.bounded_cap(192.0, 32, 18000.0, 700.0) == 1800
    assert r.bounded_cap(5.0, 32, 18000.0, 700.0) is None              # 5 GPUh -> 562 s - grace < expected
    assert r.bounded_cap(192.0, 32, 900.0, 600.0) == 660
    assert r.pair_wall_cap(700.0) == 0 and r.pair_wall_cap(1500.0) == 0 and r.worst_case_gpu_hours(32, 0) == 0.0


def test_the_locked_pair_is_not_reserved_inside_the_arm_budget():
    """AMENDMENT_EXT_003 Q5: each arm's locked pair runs right after its search, outside its budget. Through the real
    SearchSession._admission: a budget with exactly one screen's worst case left admits that screen at the full cap
    (a pair reserve of 2 members would refuse it) and the recorded pair reserve is zero."""
    from infrarsi.agent.budget import Budget, SearchLedger, VirtualClock
    from infrarsi.agent.search import SearchSession
    r = object.__new__(R.FlashNextRunner)
    r.cap_s = 1800.0
    one = r.worst_case_gpu_hours(32)                                     # 18.13 GPUh
    s = object.__new__(SearchSession)
    s.runner, s.hooks, s.task, s.task_raw = r, object(), None, {'world_size': 32}
    s.budget = Budget(screens=12, llm_tokens=10 ** 6, wall_seconds=1800 + r.screen_grace_s + 1, gpu_hours=one + 1e-6)
    s.ledger = SearchLedger(s.budget, VirtualClock())
    s.ledger.prior = (one, 700.0)
    assert s._admission() == (None, 1800)
    assert s.pair_reserve == {'gpu_hours': 0.0, 'wall_seconds': 0.0, 'member_wall_cap': 0}


def test_node_map():
    assert R.node_spec(R.ALT_NODES, master_port=44000, timeout_s=1800)['host_numbers'] == [2, 4, 5, 7]
    for bad in (['WINH0', 'WINH0', 'WINH1', 'WINH3'], ['WINH0', 'WINH1', 'WINH3'], ['WINH0', 'WINH1', 'WINH3', 'WINH9']):
        with pytest.raises(ValueError):
            R.node_spec(bad, master_port=44000, timeout_s=1800)
    assert 43000 <= R.master_port('a' * 33) < 47000


def test_real_bundle_upload_and_launch_scripts(frozen, tmp_path_factory):
    """The real re-rooted build: uploaded bundle carries the single-group phase gate, the cap, the ext cache and the
    formal portable shell (read only); every node script runs the deployed launcher with the right rank."""
    base = tmp_path_factory.mktemp('rr')
    rt = frozen.custom_runtime(base / 'rt', root=frozen.FN_ROOT)
    c = Cluster(base / 'data')
    r = R.FlashNextRunner(c.ops, nodes=R.DEFAULT_NODES, runtime_dir=rt['runtime'],
                          out_dir=base / 'out', clock=c.clock, sleep=c.clock.sleep,
                          judge_fn=judge_ok)
    rec = r.run(task(), START, None, 'reference', 'fntest', candidate_id='uc-ref', role='reference', cap_s=3600)
    assert rec['valid'] is True, rec['invalid_reasons']
    meta = next((base / 'data/data/infrarsi/ext/runs/flashnext/metadata').iterdir())
    attempt, job = json.loads((meta / 'attempt.json').read_text()), json.loads((meta / 'job.json').read_text())
    assert attempt['infra_variant'] == 'U_plus_C' and attempt['lifecycle_run_root'].startswith(frozen.FN_ROOT + '/runs/')
    gate = attempt['phase_gate']
    assert gate['members'] == {'A': attempt['attempt_id']} and gate['synchronized_start'] is False
    assert gate['root'] == frozen.FN_PHASE and gate['timeout_seconds'] == 3600
    assert job['command'][1] == frozen.FORMAL_SHELL and job['command'][3] == '--max-restarts=0'
    assert job['env']['LP_CACHE_ROOT'] == frozen.FN_CACHE and job['timeout_seconds'] == 3600
    assert job['master_port'] == R.master_port(rec['attempt_id']) or 43000 <= job['master_port'] < 47000
    for k in range(4):
        script = (meta / f'fn_node{k}.sh').read_text()
        assert f'{frozen.FN_SRC}/node/{R.LAUNCHER}' in script and f' {k} ' in script
    assert len(list(meta.joinpath('code').glob('*.py'))) == 86


def test_default_judge_is_x6s_pipeline(tmp_path):
    """judge_fn=None: parse.collect -> judge.evaluate(role=...) -> guards.check -> judge.receipt_extra. The simulator's
    signal rows are not real train_step records, so X6's parser must reject the screen as incomplete - through the
    judge (a reason), never as a runner error."""
    c = Cluster(tmp_path / 'data')
    rec = _run(runner(c, tmp_path, judge_fn=None), role='candidate', reference_parsed=None)
    assert rec['valid'] is False and not any(r.startswith('runner_error') for r in rec['invalid_reasons'])
    assert rec['judge_valid'] is False and rec['judge']['details']['role'] == 'candidate'
    assert 'numerics_gates_sha256' in rec and 'summary' in rec
    assert {n for n, _ in c.kills} == set(c.nodes) and c.leases() == []


def test_launcher_holds_the_same_legacy_locks_as_the_preflight_checks():
    import ast
    tree = ast.parse((R.NODE_DIR / R.LAUNCHER).read_text(encoding='utf-8'))
    found = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
             and any(getattr(t, 'id', None) == 'LEGACY_LOCK_DIRS' for t in n.targets)]
    assert found == [R.LEGACY_LOCK_DIRS]
    assert all(d.startswith(('/data/low_precision_winh_20261014/', '/data/zehua_flashnext_replay_20261013/'))
               for d in R.LEGACY_LOCK_DIRS)
