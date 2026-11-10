import json

import pytest

from infrarsi.agent.backends import ScriptedBackend
from infrarsi.ext.autoresearch import arms, knobs, native, source
from infrarsi.ext.common import trajectory as tj
from infrarsi.ext.common.remote_exec import gpu_claim_path, node_lease_path
from infrarsi.harness.leases import OccupiedError
from infrarsi.rsi import build_a0
from x1_ar_helpers import Node, ar_task

PASS = {'verdict': 'pass', 'reasons': []}


@pytest.fixture(scope='module')
def a0(tmp_path_factory):
    return build_a0(tmp_path_factory.mktemp('pkg') / 'A0')


def _runner(node, task, run_id, **kw):
    r = arms.make_runner(node.ops, task, gpu=0, run_id=run_id, clock=node.clock, sleep=node.clock.advance, poll_s=5, **kw)
    return r


def test_arm_lease_holds_the_gpu_for_the_run(tmp_path):
    node = Node(tmp_path)
    task = ar_task(tmp_path, 'C-RND')
    with arms.arm_lease(node.ops, 'rnd1', [0]):
        lease = json.loads(node.ops.local(node_lease_path('WINH0')).read_text(encoding='utf-8'))
        assert lease['owner'] == 'infrarsi-ext' and lease['scope'] == 'run' and lease['run_id'] == 'rnd1'
        rec = _runner(node, task, 'rnd1').run(task, {}, None, 'screen', 'rnd1', candidate_id='c0')
        assert rec['valid'] and node.ops.local(gpu_claim_path('WINH0', 0)).exists()
        with pytest.raises(OccupiedError):
            with arms.arm_lease(node.ops, 'other', [0]):
                pass
    assert not node.ops.local(node_lease_path('WINH0')).exists() and not node.ops.local(gpu_claim_path('WINH0', 0)).exists()
    # without the arm's lease the held-mode runner refuses to start
    assert not _runner(node, task, 'rnd1').run(task, {}, None, 'screen', 'rnd1')['valid']


def test_c_rnd_draws_the_seeded_permutation(tmp_path):
    node = Node(tmp_path)
    task = ar_task(tmp_path, 'C-RND')
    with arms.arm_lease(node.ops, 'rnd', [0]):
        r = _runner(node, task, 'rnd')
        tr = arms.run_c_rnd(task, r, run_id='rnd', clock=node.clock).to_dict()
    assert tr['arm'] == 'C-RND' and tr['stop_reason'] in ('budget:screens', 'budget:wall_seconds', 'budget:gpu_hours')
    # Codex W9 H-05: a screen is admitted only if its worst case fits; the arm stops with less than one worst case left
    left = min(task.budget['gpu_hours'] * 3600 - tr['used']['gpu_hours'] * 3600, task.budget['wall_seconds'] - tr['used']['wall_seconds'])
    assert tr['used']['screens'] < 36 and left < r.worst_case() and tr['used']['gpu_hours'] <= task.budget['gpu_hours']
    assert tr['candidates'][0]['id'] == 'c0' and tr['screens'][0]['candidate_id'] == 'c0'
    order = knobs.sample_order(42)
    screened = [c for c in tr['candidates'] if c['id'] != 'c0'][:5]
    assert [c['change'] for c in screened][0] == {k: [knobs.DEFAULTS[k], v] for k, v in order[0].items() if v != knobs.DEFAULTS[k]}
    assert tr['used']['screens'] <= 36 and tr['used']['llm_tokens'] == 0
    row = tj.arm_row(tr, task, 0.9977)
    assert row['n_screens'] == len(tr['screens']) and row['best_search'] == 0.9977


def test_c_is_through_optimize(tmp_path, a0):
    node = Node(tmp_path)
    task = ar_task(tmp_path, 'C-IS')
    base = source.base_train()
    script = [{'action': 'read_env'}, {'action': 'read_file', 'path': 'train.py', 'start': 430, 'end': 455},
              {'action': 'propose_patch', 'base': 'c0', 'files': {'train.py': base.replace('MATRIX_LR = 0.04', 'MATRIX_LR = 0.05')}},
              {'action': 'propose_patch', 'base': 'c0', 'files': {'train.py': base.replace('DEVICE_BATCH_SIZE = 128', 'DEVICE_BATCH_SIZE = 64')}},
              {'action': 'run_screen', 'candidate_id': 'c1'}, {'action': 'read_receipt', 'candidate_id': 'c1'},
              {'action': 'finish', 'candidate_id': 'c1'}]
    drv = ScriptedBackend(lambda role, s, p: PASS if role == 'verifier' else script.pop(0), clock=node.clock)
    with arms.arm_lease(node.ops, 'is1', [0]):
        tr = arms.run_c_is(task, a0, drv, _runner(node, task, 'is1'), run_id='is1', out_dir=tmp_path / 'is', clock=node.clock).to_dict()
    assert tr['invalid_proposals'] == {'patch_rejected': 1} and len(node.scripts) == 2         # c0 and c1 ran; the LR change never launched
    assert tr['selected'] == 'c1' and [s['valid'] for s in tr['screens']] == [True, True]
    events = [json.loads(x) for x in (tmp_path / 'is' / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    fb = {e['payload']['turn']: e['payload']['feedback'] for e in events if e['kind'] == 'turn'}
    assert 'infra-only: constant MATRIX_LR changed' in fb[3]['findings'][0]
    assert 'DEVICE_BATCH_SIZE = 128' in fb[2]['text'] and fb[6]['summary']['checks']['C-static'] == 'ok'
    assert 'Infra-only track' in drv.calls[0]['system'] and 'TOTAL_BATCH_SIZE' in next(c for c in drv.calls if c['role'] == 'verifier')['system']


def test_c_nat_runs_program_md_with_the_experiment_tool(tmp_path):
    node = Node(tmp_path)
    task = ar_task(tmp_path, 'C-NAT-infra')
    L = native.layout('nat1')

    def tool_hook(script, ops):                              # the agent edits train.py with bash
        if 'EDIT' in script:
            p = ops.local(L['repo'] + '/train.py')
            p.write_bytes(p.read_bytes().replace(b'DEVICE_BATCH_SIZE = 128', b'DEVICE_BATCH_SIZE = 64'))
        return 'ok\n'
    node.tool_hook = tool_hook
    script = [{'tool': 'bash', 'cmd': 'cat program.md'}, {'tool': 'run_experiment', 'note': 'baseline'},
              {'tool': 'bash', 'cmd': 'echo EDIT && git commit -am dbs64'}, {'tool': 'run_experiment'},
              {'tool': 'submit', 'note': 'done'}]
    drv = ScriptedBackend(script, clock=node.clock)
    with arms.arm_lease(node.ops, 'nat1', [0]):
        tr = arms.run_c_nat(task, drv, _runner(node, task, 'nat1'), node.ops, run_id='nat1', out_dir=tmp_path / 'nat',
                            tag='oct21', clock=node.clock)
    assert tr['stop_reason'] == 'submit' and [s['candidate_id'] for s in tr['screens']] == ['e01', 'e02']
    assert tr['screens'][0]['kind'] == 'start' and tr['screens'][1]['kind'] == 'L1' and tr['selected'] == 'e01'
    assert 'DEVICE_BATCH_SIZE = 64' in node.ops.local(f'{source.NODE_RUNS}/nat1/e02').joinpath(
        next(p.name for p in node.ops.local(f'{source.NODE_RUNS}/nat1/e02').iterdir()), 'repo', 'train.py').read_text(encoding='utf-8')
    prompt = drv.calls[0]['prompt']
    assert 'autoresearch/oct21' in prompt and 'Appendix for this run' in prompt
    git = [kw['cmd'] for f, kw in node.ops.calls if f == 'sh' and ' init -q' in kw.get('cmd', '')]
    assert git and "checkout -q -b autoresearch/oct21" in git[0] and '--reuid=64002' in git[0]   # created by the arm's own uid
    assert node.ops.local(L['repo'] + '/run.log').read_bytes().startswith(b'Vocab size')            # program.md reads run.log
    assert any(f == 'read_user_file' and kw['path'].endswith('/repo/train.py') for f, kw in node.ops.calls)
    with pytest.raises(ValueError):
        arms.run_c_nat(ar_task(tmp_path, 'C-IS'), drv, None, node.ops, run_id='x')


def test_locked_pair_order(tmp_path):
    node = Node(tmp_path)
    task = ar_task(tmp_path, 'C-IS')
    cand = source.base_train().replace('DEVICE_BATCH_SIZE = 128', 'DEVICE_BATCH_SIZE = 64')
    with arms.arm_lease(node.ops, 'lk', [0]):
        r = _runner(node, task, 'lk')
        odd = arms.locked_pair(r, task, cand, run_id='lk', arm_index=1)
        even = arms.locked_pair(r, task, cand, run_id='lk', arm_index=2)
    assert odd['order'] == ['candidate', 'baseline'] and even['order'] == ['baseline', 'candidate']
    assert odd['candidate']['purpose'] == 'score' and odd['candidate']['valid'] and odd['baseline']['files']['train.py'] == source.BASE_TRAIN_SHA256


def test_c_nat_copy_keep_revert(tmp_path):
    node = Node(tmp_path)
    task = ar_task(tmp_path, 'C-NAT-free')
    L = native.layout('natc')

    def tool_hook(script_text, ops):                           # the agent edits train.py between keep and revert
        if 'EDIT' in script_text:
            ops.local(L['repo'] + '/train.py').write_bytes(b'broken edit')
        return 'ok\n'
    node.tool_hook = tool_hook
    script = [{'tool': 'keep'}, {'tool': 'bash', 'cmd': 'echo EDIT'}, {'tool': 'revert', 'note': 'discard'}, {'tool': 'submit'}]
    drv = ScriptedBackend(script, clock=node.clock)
    with arms.arm_lease(node.ops, 'natc', [0]):
        tr = arms.run_c_nat(task, drv, _runner(node, task, 'natc'), node.ops, run_id='natc', tag='t', clock=node.clock,
                            keep_revert='copy')
    assert tr['extra']['keep_revert'] == 'copy' and tr['stop_reason'] == 'submit'
    cmds = [kw['cmd'] for f, kw in node.ops.calls if f == 'sh']
    assert not any(' init -q' in c for c in cmds)                                 # no git repository in copy mode
    assert node.ops.local(L['repo'] + '/train.py').read_text(encoding='utf-8') == source.base_train()   # revert restored the head
    assert node.ops.local(L['agent'] + '/head/train.py').read_text(encoding='utf-8') == source.base_train()
    writes = [kw for f, kw in node.ops.calls if f == 'write_user_file']
    assert writes and writes[-1]['uid'] == 64001 and writes[-1]['root'] == L['home']        # C-NAT-free's own uid
    assert '{"tool":"keep",' in drv.calls[0]['system'] and 'git is not available' in drv.calls[0]['prompt']


def test_arms_are_trust_domains(tmp_path):
    """Codex W9 B-02/B-03/B-04: arm uid unused at the start, killed before the claims go, D-AR-7 arm tasks only."""
    from infrarsi.ext.autoresearch.task import ARM_UIDS, env_c_task
    from infrarsi.ext.common.remote_exec import ContainmentError
    from x1_ar_helpers import make_assets
    node = Node(tmp_path)
    task = ar_task(tmp_path, 'C-RND')
    events = []
    with arms.arm_lease(node.ops, 'rnd2', [0], uids=[ARM_UIDS['C-RND']], events=events) as lease:
        assert lease.events is events and ('uid_status', {'uid': 64004}) in node.ops.calls
        r = _runner(node, task, 'rnd2')
        assert r.train_uid == 64004 and r.domain_uids() == [64004]
    assert [e['kind'] for e in events] == ['contain', 'release'] and not node.ops.local(gpu_claim_path('WINH0', 0)).exists()
    node.ops.uid_survivors[64004] = [1234]
    with pytest.raises(ContainmentError):
        with arms.arm_lease(node.ops, 'rnd3', [0], uids=[64004], events=events):
            pass
    assert events[-1]['kind'] == 'claims_kept' and node.ops.local(gpu_claim_path('WINH0', 0)).exists()     # GPU stays claimed
    node2 = Node(tmp_path / 'n2')
    node2.ops.uid_procs[64004] = [9]
    with pytest.raises(ContainmentError):                               # the uid is in use: the arm never starts
        with arms.arm_lease(node2.ops, 'rnd4', [0], uids=[64004]):
            pytest.fail('started')
    assert not node2.ops.local(node_lease_path('WINH0')).exists()
    diag = env_c_task('C-BASE', assets=make_assets(tmp_path / 'd'), prepare_variant='upstream', diagnostic=True)
    with pytest.raises(ValueError):
        arms.make_runner(node.ops, diag, gpu=0, run_id='x')
    other = arms.make_runner(node.ops, ar_task(tmp_path, 'C-IS'), gpu=0, run_id='x')        # C-IS uid 64003
    with pytest.raises(ValueError, match='runner uid'):
        arms.run_c_rnd(task, other, run_id='x')
    with pytest.raises(ValueError):
        arms.make_runner(node.ops, task, gpu=0, run_id='x', calib_uid=65534)
