"""ENV-C CLI (O follow-up 1): dry-run plans, the arms through synthetic nodes, the calibration and its gates."""
import json

import pytest

from infrarsi.agent.backends import ScriptedBackend
from infrarsi.ext.autoresearch import cli, knobs, source
from infrarsi.ext.autoresearch.runner import AutoresearchRunner
from infrarsi.ext.common.remote_exec import nobody_cmd
from infrarsi.rsi import build_a0
from x1_ar_helpers import Node, make_assets

PASS = {'verdict': 'pass', 'reasons': []}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A synthetic WINH0 with frozen assets, the CLI's records under tmp, tasks built on the synthetic assets."""
    assets = make_assets(tmp_path)
    assets['reference'] = {**assets['reference'], 'val_bpb': 0.996365, 'num_steps': 936, 'attempt_id': 'a21767e9dbd4f',
                           'node': 'WINH0', 'gpu': [0], 'prepare_variant': 'h2d_sync', 'first_losses': source.R2['first_losses']}
    from infrarsi.ext.autoresearch import task as taskmod
    real = taskmod.env_c_task
    monkeypatch.setattr(cli, 'env_c_task', lambda arm, **kw: real(arm, assets=assets, **kw))
    monkeypatch.setattr(cli, 'EXEC_ROOT', tmp_path / 'exec')
    node = Node(tmp_path)
    orig_init = AutoresearchRunner.__init__

    def fast_init(self, *a, **kw):                     # synthetic clock: no real sleeping while polling
        kw.update(clock=node.clock, sleep=node.clock.advance, poll_s=5)
        orig_init(self, *a, **kw)
    monkeypatch.setattr(AutoresearchRunner, '__init__', fast_init)
    return node, tmp_path


def _out(tmp_path, run_id):
    return json.loads((tmp_path / 'exec' / run_id / 'result.json').read_text(encoding='utf-8'))


def test_dry_run_plans_touch_nothing(env, capsys):
    node, tmp = env
    for cmd, arm, gpu, uid in (('c-base', 'C-BASE', 0, 64000), ('c-rnd', 'C-RND', 0, 64004), ('c-nat-free', 'C-NAT-free', 2, 64001),
                               ('c-nat-infra', 'C-NAT-infra', 4, 64002), ('c-is', 'C-IS', 6, 64003)):
        assert cli.main([cmd, '--run-id', f'dry-{cmd}', '--dry-run'], ops_factory=lambda n: pytest.fail('node touched')) == 0
        p = json.loads(capsys.readouterr().out)
        assert (p['arm'], p['gpus'], p['node'], p['prepare_variant']) == (arm, [gpu], 'WINH0', 'h2d_sync') and 'D-AR-7' in p['deviations']
        assert p['uids'] == [uid]                                          # Codex W9 B-03: one trust domain per arm
        with pytest.raises(SystemExit):                                    # B-04: no other loader for an arm command
            cli.main([cmd, '--run-id', 'x', '--dry-run', '--prepare-variant', 'upstream'])
        capsys.readouterr()
        assert p['anchor']['num_steps'] == 936 and p['budget']['screens'] == 36 and p['budget']['wall_seconds'] == 14400
        assert (p['driver'] is not None) == (arm in cli.LLM_ARMS) and p['lease'].startswith('node lease')
    cli.main(['c-nat-infra', '--run-id', 'x', '--dry-run', '--group-run-id', 'grp', '--screens', '10', '--llm-usd', '20'])
    p = json.loads(capsys.readouterr().out)
    assert p['uid'] == 64002 and 'group node lease grp' in p['lease'] and p['budget']['screens'] == 10 and p['budget']['llm_usd'] == 20
    cli.main(['calib', '--run-id', 'c', '--dry-run'])
    p = json.loads(capsys.readouterr().out)
    assert p['gpus'] == [0, 2, 4, 6] and p['points'] == knobs.sample_order(42)[:3] and p['gate_reference_steps'] == 936
    assert p['uids'] == [64000, 64005, 64006, 64007]                       # one uid per concurrent calibration run
    cli.main(['locked', '--run-id', 'l', '--arm', 'C-NAT-free', '--from', 'x.py', '--arm-index', '1', '--dry-run'])
    assert json.loads(capsys.readouterr().out)['uids'] == [64001]
    assert not (tmp / 'exec').exists() and node.ops.calls == []


def test_c_base_with_loader_check(env):
    node, tmp = env
    node.behaviours = [{'judged': 0.9963, 'judged2': 0.9963}]
    cli.main(['c-base', '--run-id', 'cb', '--loader-check'], ops_factory=lambda n: node.ops)
    r = _out(tmp, 'cb')
    assert r['valid'] and r['val_bpb'] == 0.9963 and 'D-AR-7' in r['deviations']
    kinds = [e['kind'] for e in r['lease_events']]
    assert kinds == ['contain', 'release'] and r['lease_events'][0]['uid'] == 64000          # killed before the claims went
    assert ('uid_status', {'uid': 64000}) in node.ops.calls                                    # checked unused at the start
    assert '--reuid=64000' in node.scripts[0]
    eq = next(g for g in r['reports'] if g['id'] == 'judge_loader_equivalence')
    assert eq['ok'] is True and eq['abs_dev'] == 0.0 and eq['tolerance'] == 1e-6
    s = node.scripts[0]
    assert s.count('judge_bpb.py') == 2 and '/judge_h2d/judge_bpb.py' in s
    assert (tmp / 'exec' / 'cb' / 'plan.json').exists() and list((tmp / 'exec' / 'cb' / 'runner' / 'receipts').glob('*.json'))
    h2d = node.ops.local(f'{source.NODE_RUNS}/cb/c0').glob('*/judge_h2d/prepare.py')
    assert 'D-AR-7' in next(h2d).read_text(encoding='utf-8')                  # the second judge loads the D-AR-7 loader
    node.behaviours = [{'judged': 0.9963, 'judged2': 0.99631}]
    cli.main(['c-base', '--run-id', 'cb2', '--loader-check'], ops_factory=lambda n: node.ops)
    eq = next(g for g in _out(tmp, 'cb2')['reports'] if g['id'] == 'judge_loader_equivalence')
    assert eq['ok'] is False                                                 # a report, never a veto


def test_calibration_runs_concurrently_and_gates_on_the_anchor(env):
    node, tmp = env
    cli.main(['calib', '--run-id', 'cal'], ops_factory=lambda n: node.ops)
    r = _out(tmp, 'cal')
    assert set(r['runs']) == {'c0', 'p1', 'p2', 'p3'} and [r['runs'][k]['gpus'] for k in ('c0', 'p1', 'p2', 'p3')] == [[0], [2], [4], [6]]
    assert r['gate'] == {**r['gate'], 'reference_steps': 936, 'c_base_steps': 948, 'limit': 0.02, 'passed': True}
    assert r['gate']['rel_dev'] == pytest.approx(12 / 936) and r['loader_equivalence']['ok'] is True
    assert all('D-AR-7' in v['deviations'] for v in r['runs'].values())
    reuid = sorted(int(s.split('--reuid=')[1].split()[0]) for s in node.scripts[-4:])
    assert reuid == [64000, 64005, 64006, 64007]                           # ending one run never kills another
    run = f'{source.NODE_RUNS}/cal'                                        # the shared run dir: traverse-only, root:root
    assert all(kw['mode'] == 0o711 and kw.get('gid') is None for f, kw in node.ops.calls
               if f in ('mkdirs', 'mkdir_domain') and run in kw['paths'])
    assert sorted(node.ops.owners[f'{run}/{c}'][1] for c in ('c0', 'p1', 'p2', 'p3')) == [64000, 64005, 64006, 64007]
    assert sorted(e['uid'] for e in r['lease_events'] if e['kind'] == 'contain') == [64000, 64005, 64006, 64007]
    node.behaviours = []
    cli.main(['calib', '--run-id', 'cal2', '--gate', '0.01'], ops_factory=lambda n: node.ops)
    assert _out(tmp, 'cal2')['gate']['passed'] is False                      # 1.3 % > 1 %


def test_search_arms_and_locked(env):
    node, tmp = env
    cli.main(['c-rnd', '--run-id', 'rnd', '--screens', '3'], ops_factory=lambda n: node.ops)
    r = _out(tmp, 'rnd')
    assert r['stop_reason'] == 'budget:screens' and (tmp / 'exec' / 'rnd' / 'trace.json').exists()
    a0 = build_a0(tmp / 'pkg' / 'A0')
    base = source.base_train()
    script = [{'action': 'propose_patch', 'base': 'c0', 'files': {'train.py': base.replace('DEVICE_BATCH_SIZE = 128', 'DEVICE_BATCH_SIZE = 64')}},
              {'action': 'run_screen', 'candidate_id': 'c1'}, {'action': 'finish', 'candidate_id': 'c1'}]
    drv = ScriptedBackend(lambda role, s, p: PASS if role == 'verifier' else script.pop(0), clock=node.clock)
    cli.main(['c-is', '--run-id', 'is1', '--screens', '5'], ops_factory=lambda n: node.ops, driver_factory=lambda: drv,
             package_factory=lambda name: a0)
    assert _out(tmp, 'is1')['selected'] == 'c1'
    text = cli.candidate_text(str(tmp / 'exec' / 'is1' / 'trace.json'))
    assert 'DEVICE_BATCH_SIZE = 64' in text
    rnd_tr = json.loads((tmp / 'exec' / 'rnd' / 'trace.json').read_text(encoding='utf-8'))
    assert cli.candidate_text(str(tmp / 'exec' / 'rnd' / 'trace.json')) == knobs.apply(source.base_train(), rnd_tr['selected_config'])
    n0 = len(node.scripts)
    cli.main(['locked', '--run-id', 'lk', '--arm', 'C-IS', '--from', str(tmp / 'exec' / 'is1' / 'trace.json'), '--arm-index', '3'],
             ops_factory=lambda n: node.ops)
    r = _out(tmp, 'lk')
    assert r['order'] == ['candidate', 'baseline'] and r['candidate']['valid'] and r['baseline']['valid'] and len(node.scripts) == n0 + 2
    assert r['candidate']['purpose'] == 'score' and 'D-AR-7' in r['candidate']['deviations']


def test_native_arm_through_the_cli(env):
    node, tmp = env
    drv = ScriptedBackend([{'tool': 'run_experiment'}, {'tool': 'submit'}], clock=node.clock)
    cli.main(['c-nat-free', '--run-id', 'nf'], ops_factory=lambda n: node.ops, driver_factory=lambda: drv)
    r = _out(tmp, 'nf')
    assert r['stop_reason'] == 'submit' and r['selected'] == 'e01'
    agent_dir = tmp / 'exec' / 'nf' / 'agent'
    assert (agent_dir / 'experiments' / 'e01.train.py').read_text(encoding='utf-8') == source.base_train()
    assert cli.candidate_text(str(agent_dir / 'trace.json')) == source.base_train()
    homes = [kw for f, kw in node.ops.calls if f == 'mkdirs' and kw.get('uid') == 64001]
    assert homes and any(p.endswith('/agent/home') for p in homes[0]['paths'])


def test_uid_plumbing():
    cmd = nobody_cmd('id', home='/h', uid=64001)
    assert '--reuid=64001 --regid=64001 --clear-groups' in cmd and 'USER=isx64001' in cmd
    with pytest.raises(ValueError):
        nobody_cmd('id', home='/h', uid=0)
    assert '--reuid=65534 --regid=65534' in nobody_cmd('id', home='/h') and 'USER=nobody' in nobody_cmd('id', home='/h')


def test_group_lease_for_colocated_arms(env, capsys):
    node, tmp = env
    cli.main(['group-lease', '--run-id', 'grp'], ops_factory=lambda n: node.ops)
    lease = json.loads(capsys.readouterr().out)['lease']
    assert lease['owner'] == 'infrarsi-ext' and lease['scope'] == 'run' and lease['run_id'] == 'grp' and lease['gpus'] == [0, 2, 4, 6]
    drv = ScriptedBackend([{'tool': 'submit'}], clock=node.clock)
    cli.main(['c-nat-infra', '--run-id', 'ni', '--group-run-id', 'grp'], ops_factory=lambda n: node.ops, driver_factory=lambda: drv)
    capsys.readouterr()
    assert _out(tmp, 'ni')['stop_reason'] == 'submit'
    with pytest.raises(Exception):                                         # a second group cannot take the node
        cli.main(['group-lease', '--run-id', 'other'], ops_factory=lambda n: node.ops)
    node.ops.uid_survivors[64003] = [777]                                  # a C-IS process that cannot be killed
    from infrarsi.ext.common.remote_exec import ContainmentError, node_lease_path
    with pytest.raises(ContainmentError):
        cli.main(['group-release', '--run-id', 'grp'], ops_factory=lambda n: node.ops)
    assert node.ops.local(node_lease_path('WINH0')).exists()               # the node stays leased
    node.ops.uid_survivors.clear()
    n0 = len(node.ops.uid_kills)
    cli.main(['group-release', '--run-id', 'grp'], ops_factory=lambda n: node.ops)
    out = json.loads(capsys.readouterr().out)
    assert out['released'] is True and sorted(node.ops.uid_kills[n0:]) == list(cli.ALL_UIDS)
    node.ops.uid_procs[64002] = [55]                                       # a uid in use: no group lease
    with pytest.raises(ContainmentError):
        cli.main(['group-lease', '--run-id', 'g2'], ops_factory=lambda n: node.ops)
    with pytest.raises(ContainmentError):                                  # nor an arm on it
        cli.main(['c-nat-infra', '--run-id', 'ni2'], ops_factory=lambda n: node.ops, driver_factory=lambda: drv)
    node.ops.uid_procs.clear()
    node.ops.uid_passwd[64000] = 'someone'                                 # a passwd entry also disqualifies a uid
    with pytest.raises(ContainmentError):
        cli.main(['c-base', '--run-id', 'cb9'], ops_factory=lambda n: node.ops)
