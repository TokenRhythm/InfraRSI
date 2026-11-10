import shlex

import pytest

from infrarsi.ext.autoresearch import source
from infrarsi.ext.autoresearch.runner import AutoresearchRunner, parse_times
from infrarsi.ext.common.env_judge import EnvJudge
from infrarsi.harness import signing
from x1_ar_helpers import Node, ar_task, run_meta, script_c, train_log


def _runner(node, task, **kw):
    kw.setdefault('train_uid', task.raw['uid'])
    return AutoresearchRunner(node.ops, gpus=(0,), track=task.raw['track'], expected=task.raw['expected_runtime'],
                              clock=node.clock, sleep=node.clock.advance, poll_s=5, **kw)


def test_baseline_c0_infra_receipt(tmp_path):
    task = ar_task(tmp_path, 'C-BASE')
    node = Node(tmp_path)
    rec = _runner(node, task).run(task, {}, None, 'screen', 'x1check', candidate_id='c0')
    assert signing.verify(rec) and rec['valid'], rec['invalid_reasons']
    assert rec['metric']['value'] == 0.9977 and rec['metric']['raw']['val_bpb_printed'] == 0.9977 and rec['metric']['raw']['abs_dev'] == 0.0
    assert rec['deviations'] == ['D-AR-1', 'D-AR-2', 'D-AR-3', 'D-AR-5', 'D-AR-7'] and rec['files']['train.py'] == source.BASE_TRAIN_SHA256
    assert rec['judge']['ok'] and rec['judge_valid'] and rec['guard_rejections'] == []
    assert {r['id'] for r in rec['guard_reports']} >= {'C-runtime', 'C-loss10', 'printed_vs_judge'}
    assert rec['summary']['checks'] == {k: 'ok' for k in ('frozen_files', 'exit_ok', 'outputs_present', 'C-static', 'C-runtime', 'C-loss10')}
    assert rec['summary']['num_steps'] == 948 and rec['phases']['train'] == 345.0 and rec['phases']['judge'] == 34.0
    assert rec['candidate_dir'] == f'{source.NODE_RUNS}/x1check/c0/{rec["attempt_id"]}' and rec['candidate_id'] == 'c0'
    assert rec['outputs']['final_model.pt']['bytes'] == 700 and rec['outputs']['final_model.pt']['sha256']
    C = node.ops.local(rec['candidate_dir'])
    assert not (C / 'out' / 'final_model.pt').exists() and (C / 'train.log').exists()          # bulky output removed
    assert (C / 'repo' / 'train.py').read_text(encoding='utf-8') == source.base_train()
    assert (C / 'judge' / 'judge_bpb.py').exists() and (C / 'tools' / 'run_train.py').exists()
    assert EnvJudge('autoresearch')(rec).valid and EnvJudge('autoresearch')(rec).step_time == 0.9977
    links = node.ops.links
    assert links[f'{rec["candidate_dir"]}/home/.cache/autoresearch/data'] == source.NODE_DATA
    assert links[f'{rec["candidate_dir"]}/judge/home/.cache/autoresearch/tokenizer'] == source.NODE_TOKENIZER
    assert any(k.endswith(f'snapshots/{source.KERNEL_REV}/build/{source.KERNEL_VARIANT}') for k in links)


def test_script_runs_training_as_the_arm_uid_and_judge_as_root(tmp_path):
    task = ar_task(tmp_path, 'C-IS')
    node = Node(tmp_path)
    rec = _runner(node, task).run(task, {}, None, 'screen', 'r', candidate_id='c0')
    s = node.scripts[0]
    C = script_c(s)
    train = next(l for l in s.splitlines() if 'run_train.py' in l)
    assert task.raw['uid'] == 64003 and rec['train_uid'] == 64003
    assert train.startswith('timeout --kill-after=10 600 setpriv --reuid=64003 --regid=64003 --clear-groups --no-new-privs env -i ')
    for part in (f'HOME={C}/home', 'CUDA_VISIBLE_DEVICES=0', f'INFRARSI_OUT={C}/out', 'INFRARSI_SAVE=1', 'HF_HUB_OFFLINE=1',
                 f'TORCHINDUCTOR_CACHE_DIR={C}/home/caches/inductor', f'> {C}/train.log 2>&1'):
        assert part in train, part
    judge = next(l for l in s.splitlines() if 'judge_bpb.py' in l).strip()
    assert judge.startswith('timeout --kill-after=10 300 env -i ') and 'setpriv' not in judge and '--batch-size 128' in judge
    assert f'TORCHINDUCTOR_CACHE_DIR={source.NODE_RUNS}/r/.judge_cache/inductor' in judge
    assert s.count('INFRARSI_TOKEN=isx-c') == 3                       # training and judge env (labels), judge kill
    # Containment by uid (Codex W9 B-02): the judge starts only when no process of the arm uid survived the training.
    lines = s.splitlines()
    contain = next(i for i, l in enumerate(lines) if l.startswith('if contain; then'))
    assert next(i for i, l in enumerate(lines) if 'run_train.py' in l) < contain < next(i for i, l in enumerate(lines) if 'judge_bpb.py' in l)
    assert 'setpriv --reuid=64003 --regid=64003 --clear-groups kill -9 -1' in s and 'pkill -9 -U 64003' in s
    assert f'grep -q "^contain_ok" {C}/times.txt' in s
    attest = next(l for l in lines if 'code_attest.py expect' in l)        # trusted code expectation, root, before training
    assert attest.startswith('timeout --kill-after=10 60 env -i ') and ' -I ' in attest and 'setpriv' not in attest
    assert lines.index(attest) < next(i for i, l in enumerate(lines) if 'run_train.py' in l)
    assert s.rstrip().endswith(f'mv {C}/times.txt {C}/DONE.txt')
    launch = next(kw for f, kw in node.ops.calls if f == 'launch')
    assert launch['env']['INFRARSI_TOKEN'].startswith('isx-cscript')       # run.sh itself is killable by its token
    assert node.ops.uid_kills.count(64003) == 2                         # execute's finally + EnvRunner before release
    assert rec['containment']['ok'] and rec['lease_release'] == {'kept': [], 'failures': []}
    shlex.split(train)                                                   # well-formed quoting
    mk = [kw for f, kw in node.ops.calls if f == 'mkdirs']               # run dir 0711 root:root; tree root:<gid> 0750 / uid 0700
    assert any(kw['paths'] == [f'{source.NODE_RUNS}/r'] and kw['mode'] == 0o711 and kw.get('gid') is None and kw.get('uid') is None
               for kw in mk)
    dom = [kw for f, kw in node.ops.calls if f == 'mkdir_domain']
    assert any(kw['mode'] == 0o750 and kw.get('gid') == 64003 and kw.get('uid') is None and f'{source.NODE_RUNS}/r/c0' in kw['paths']
               for kw in dom)
    assert not any(f'{source.NODE_RUNS}/r' in kw['paths'] for kw in dom)                       # never re-grouped
    assert any(kw['mode'] == 0o700 and kw.get('uid') == 64003 and f'{C}/out' in kw['paths'] for kw in mk)


def test_several_uids_share_one_run_dir(tmp_path):
    """O 09:20: the calibration's four uids share <run_id>/; it stays traverse-only root:root and each candidate
    subtree belongs to its own uid; a candidate dir of another uid is never re-owned."""
    from x1_ar_helpers import ar_task
    node = Node(tmp_path)
    recs = {}
    for cid, uid in (('c0', 64000), ('p1', 64005), ('p2', 64006), ('p3', 64007)):
        task = ar_task(tmp_path, 'C-BASE' if cid == 'c0' else 'C-RND')
        recs[cid] = _runner(node, task, train_uid=uid).run(task, {}, None, 'screen', 'calib', candidate_id=cid)
        assert recs[cid]['valid'], recs[cid]['invalid_reasons']
    run = f'{source.NODE_RUNS}/calib'
    run_calls = [kw for f, kw in node.ops.calls if f in ('mkdirs', 'mkdir_domain') and run in kw['paths']]
    assert run_calls and all(kw['mode'] == 0o711 and kw.get('gid') is None and kw.get('uid') is None for kw in run_calls)
    assert {p: node.ops.owners[p] for p in (f'{run}/c0', f'{run}/p1', f'{run}/p2', f'{run}/p3')} == {
        f'{run}/c0': (0, 64000), f'{run}/p1': (0, 64005), f'{run}/p2': (0, 64006), f'{run}/p3': (0, 64007)}
    for cid, uid in (('c0', 64000), ('p1', 64005)):                      # every attempt tree carries its own gid
        C = recs[cid]['candidate_dir']
        assert node.ops.owners[C] == (0, uid) and node.ops.owners[f'{C}/repo'] == (0, uid)
    task = ar_task(tmp_path, 'C-RND')                                     # another uid on p1's candidate dir: refused
    clash = _runner(node, task, train_uid=64004).run(task, {}, None, 'screen', 'calib', candidate_id='p1')
    assert not clash['valid'] and any('refusing to re-own' in r for r in clash['invalid_reasons'])
    assert node.ops.owners[f'{run}/p1'] == (0, 64005)                    # p1 untouched


def test_cleanup_is_in_finally_and_containment_failure_keeps_the_lease(tmp_path):
    from infrarsi.ext.common.remote_exec import NodeError
    task = ar_task(tmp_path, 'C-IS')
    node = Node(tmp_path, [{'no_done': True}])
    real = node.ops.call

    def flaky(func, _timeout=300, **kw):                                 # SSH drops while polling DONE.txt
        if func == 'read_files' and any(str(p).endswith('DONE.txt') for p in kw.get('paths', [])):
            raise NodeError('ssh: connection reset')
        return real(func, _timeout, **kw)
    node.ops.call = flaky
    rec = _runner(node, task).run(task, {}, None, 'screen', 'r', candidate_id='c1')
    assert not rec['valid'] and any('runner_error' in r or 'connection reset' in r for r in rec['invalid_reasons']), rec['invalid_reasons']
    assert any(t.startswith('isx-cscript') for t in node.ops.killed_tokens)        # the root script was stopped
    assert node.ops.uid_kills.count(64003) == 2 and rec['containment']['ok']
    lease = node.ops.local('/data/infrarsi/leases/WINH0.json')
    assert not lease.exists()                                           # contained: released
    node = Node(tmp_path)
    node.ops.uid_survivors[64003] = [4242]                              # a process of the arm uid cannot be killed
    rec = _runner(node, task).run(task, {}, None, 'screen', 'r', candidate_id='c2')
    assert not rec['valid'] and rec['metric']['value'] is None and not rec['containment']['ok']
    assert any(r.startswith('containment:') for r in rec['invalid_reasons'])
    assert rec['lease_release']['kept'] and node.ops.local('/data/infrarsi/leases/WINH0.json').exists()      # GPU stays leased
    assert signing.verify(rec)


def test_wall_cap_bounds_the_attempt(tmp_path):
    task = ar_task(tmp_path, 'C-IS')
    node = Node(tmp_path)
    r = _runner(node, task)
    worst = r.worst_case()
    assert worst == 600 + 120 + 60 + 10 + 40 + 310 + 10 + 120 and r.bounded_cap is not None
    assert r.bounded_cap(4.0, 1, worst - 1, 400) is None and r.bounded_cap(4.0, 1, worst, 400) == worst
    assert r.bounded_cap(worst / 3600 - 0.01, 1, 1e6, 400) is None                 # GPU-hours left decide too
    rec = r.run(task, {}, None, 'screen', 'r', candidate_id='c1', wall_cap=worst - 100)
    assert 'timeout --kill-after=10 500 setpriv' in node.scripts[0] and rec['train_timeout_s'] == 500
    rec = r.run(task, {}, None, 'screen', 'r', candidate_id='c2', wall_cap=300)
    assert len(node.scripts) == 1 and not rec['valid'] and any('wall cap 300' in x for x in rec['invalid_reasons'])
    free = _runner(Node(tmp_path), ar_task(tmp_path, 'C-NAT-free'))
    assert free.worst_case() == 600 + 120 + 60 + 10 + 40 + 10 + 120                 # no judge on the free track


@pytest.mark.parametrize('attest,why', [
    ({'train:GPT.setup_optimizer': 'f' * 64}, 'protected callables not bound to their checked definitions: train:GPT.setup_optimizer'),
    ({'train:get_lr_multiplier': 'foreign code: f in /x/train.py'}, 'train:get_lr_multiplier'),
    ({'bound_to_prepare': {'MAX_SEQ_LEN': True, 'TIME_BUDGET': True, 'Tokenizer': True, 'make_dataloader': False,
                           'evaluate_bpb': True}}, "not bound to the prepare module objects at run end: ['make_dataloader']"),
    ({'instance_overrides': ['setup_optimizer']}, 'overrides protected methods'),
    ({'model_class_is_GPT': False, 'model_class': 'OptimizedModule'}, 'not the GPT class'),
    ({'python': '3.0.0'}, 'interpreters differ'),
    ({'optimizer_steps': 0}, 'no optimizer step was observed'),
    ({'optimizer_steps': 900}, '900 optimizer steps went through the class step method, the run made 948 training steps'),
    ({'override_steps': 3}, '3 hooked steps ran while the optimizer instance overrode a step method'),
    ({'first_step': None, 'step_hook': 'not registered'}, 'no attestation at the first optimizer step'),
    ({'optimizers_stepped': 2}, '2 different optimizers took steps'),
    ({'same_optimizer': False}, 'not the one that took the first step'),
    ({'optimizer': {'present': True, 'class': 'MuonAdamW', 'class_is_train_def': True, 'instance_overrides': ['step']}},
     "the optimizer instance overrides ['step'] at run end"),
])
def test_code_attestation_rejects_rebinding(tmp_path, attest, why):
    task = ar_task(tmp_path, 'C-IS')
    node = Node(tmp_path, [{'attest': attest}])
    rec = _runner(node, task).run(task, {}, None, 'screen', 'r', candidate_id='c1')
    assert not rec['valid'] and rec['judge_valid'], rec['invalid_reasons']
    assert any(r.startswith('guard:C-runtime') and why in r for r in rec['invalid_reasons']), rec['invalid_reasons']


def test_code_attestation_passes_for_the_baseline_and_needs_run_meta(tmp_path):
    task = ar_task(tmp_path, 'C-IS')
    node = Node(tmp_path, [{}, {'meta': run_meta(code_attestation=None)}])
    ok = _runner(node, task).run(task, {}, None, 'screen', 'r', candidate_id='c0')
    assert ok['valid'] and any(g['id'] == 'C-attest' for g in ok['guard_reports'])
    C = node.ops.local(ok['candidate_dir'])
    exp = __import__('json').loads((C / 'judge' / 'code_expected.json').read_text(encoding='utf-8'))
    assert exp['candidate'] == {k: v for k, v in exp['baseline'].items()} | {k: v for k, v in exp['candidate'].items() if k.startswith('prepare:')}
    missing = _runner(node, task).run(task, {}, None, 'screen', 'r', candidate_id='c1')
    assert not missing['valid'] and any('no code attestation' in r for r in missing['invalid_reasons'])


def test_contain_fail_marker_rejects_and_skips_the_judge(tmp_path):
    task = ar_task(tmp_path, 'C-IS')
    node = Node(tmp_path, [{'contain_fail': True}])
    rec = _runner(node, task).run(task, {}, None, 'screen', 'r', candidate_id='c1')
    assert not rec['valid'] and any(r.startswith('guard:containment') for r in rec['invalid_reasons'])
    assert rec['judge'] is None or not rec['judge_valid']


def test_runner_refuses_nobody_and_non_arm_variants(tmp_path):
    task = ar_task(tmp_path, 'C-IS')
    for bad in (65534, 0, 1000):
        with pytest.raises(ValueError):
            _runner(Node(tmp_path), task, train_uid=bad)


@pytest.mark.parametrize('behaviour,expect', [
    ({'crash': True}, ('guard:exit_ok', 'judge: trusted judge did not run')),
    ({'judge_fail': True}, ('judge: state_dict names',)),
    ({'losses': [v * 1.05 for v in source.R2['first_losses']]}, ('guard:C-loss10',)),
    ({'tamper': True}, ('guard:frozen_files',)),
    ({'meta': run_meta(float32_matmul_precision='medium')}, ('guard:C-runtime',)),
    ({'no_done': True}, ('guard:exit_ok', 'judge:')),
])
def test_invalid_attempts(tmp_path, behaviour, expect):
    task = ar_task(tmp_path, 'C-IS')
    node = Node(tmp_path, [behaviour])
    rec = _runner(node, task).run(task, {}, None, 'screen', 'r', candidate_id='c1')
    assert not rec['valid'] and rec['metric']['value'] is None and signing.verify(rec)
    for e in expect:
        assert any(r.startswith(e) for r in rec['invalid_reasons']), (e, rec['invalid_reasons'])
    assert not EnvJudge('autoresearch')(rec).valid
    if behaviour.get('crash'):
        assert 'CUDA out of memory' in rec['summary']['log_tail']


def test_static_violation_runs_and_is_rejected_after(tmp_path):
    task = ar_task(tmp_path, 'C-NAT-infra')
    node = Node(tmp_path)
    text = source.base_train().replace('MATRIX_LR = 0.04', 'MATRIX_LR = 0.06')
    rec = _runner(node, task).run(task, {}, {'train.py': text}, 'screen', 'r', candidate_id='e3')
    assert len(node.scripts) == 1 and not rec['valid'] and rec['judge_valid']        # it ran; the check rejects it after
    assert any(r.startswith('guard:C-static: constant MATRIX_LR') for r in rec['invalid_reasons'])
    assert rec['metric']['raw']['judge_value'] == 0.9977


def test_knob_candidates_and_free_track(tmp_path):
    task = ar_task(tmp_path, 'C-RND')
    node = Node(tmp_path, [{'printed': 0.9962, 'judged': 0.99615}])
    rec = _runner(node, task).run(task, {'DEVICE_BATCH_SIZE': 64, 'compile_mode': 'default', 'compile_dynamic': False,
                                         'expandable_segments': True, 'optimizer_foreach': 'default'}, None, 'screen', 'r', candidate_id='c4')
    assert rec['valid'] and rec['metric']['value'] == 0.99615 and rec['metric']['raw']['abs_dev'] == pytest.approx(5e-5)
    red = next(r for r in rec['guard_reports'] if r['id'] == 'printed_vs_judge')
    assert red['red'] is True
    written = node.ops.local(rec['candidate_dir']) / 'repo' / 'train.py'
    assert 'DEVICE_BATCH_SIZE = 64' in written.read_text(encoding='utf-8')
    free = ar_task(tmp_path, 'C-NAT-free')
    node = Node(tmp_path, [{'printed': 0.9901}])
    text = source.base_train().replace('DEPTH = 8 ', 'DEPTH = 10 ')                    # free track: semantics may change
    rec = _runner(node, free).run(free, {}, {'train.py': text}, 'screen', 'r', candidate_id='e1')
    assert rec['valid'] and rec['metric']['value'] == 0.9901 and rec['judge'] is None and rec['deviations'] == ['D-AR-1', 'D-AR-2', 'D-AR-3', 'D-AR-7']
    assert 'INFRARSI_SAVE=0' in node.scripts[0] and 'judge_bpb.py' in node.scripts[0]
    assert set(rec['summary']['checks']) == {'frozen_files', 'exit_ok'}


def test_parse_times():
    t = parse_times('launch 1.5\nhost h\ngpu 0, H100\ntrain_exit 124\njunk x\n')
    assert t == {'launch': 1.5, 'host': 'h', 'gpu': '0, H100', 'train_exit': 124, 'junk': 'x'}
    assert parse_times(None) == {}


def test_d_ar_7_is_the_default_and_recorded(tmp_path):
    from infrarsi.ext.autoresearch.task import env_c_task, require_arm_task
    from x1_ar_helpers import make_assets
    assets = make_assets(tmp_path)
    v = env_c_task('C-IS', assets=assets)
    for arm in ('C-BASE', 'C-NAT-free', 'C-NAT-infra', 'C-IS', 'C-RND'):    # Codex W9 B-04: arm tasks refuse other variants
        with pytest.raises(ValueError, match='D-AR-7'):
            env_c_task(arm, assets=assets, prepare_variant='upstream')
    with pytest.raises(ValueError):
        env_c_task('C-IS', assets=assets, prepare_variant='upstream', diagnostic=True)       # the diagnostic is C-BASE only
    up = env_c_task('C-BASE', assets=assets, prepare_variant='upstream', diagnostic=True)   # the named diagnostic path
    assert up.id == 'ENV-C-DIAG' and up.raw['split'] == 'diagnostic' and up.raw['uid'] == 64009 and up.raw['diagnostic']
    with pytest.raises(ValueError, match='diagnostic'):
        require_arm_task(up)
    require_arm_task(v)
    assert v.raw['prepare_variant'] == 'h2d_sync' and 'D-AR-7' in v.raw['deviations'] and 'D-AR-7' not in up.raw['deviations']
    assert all('D-AR-7' in env_c_task(a, assets=assets).raw['deviations'] for a in ('C-BASE', 'C-NAT-free', 'C-NAT-infra', 'C-RND'))
    assert up.frozen_files['repo/prepare.py'] == source.UPSTREAM_SHA256['prepare.py'] != v.frozen_files['repo/prepare.py']
    assert up.frozen_files['judge/prepare.py'] == v.frozen_files['judge/prepare.py'] == source.UPSTREAM_SHA256['prepare.py']
    text = source.candidate_prepare('h2d_sync')
    assert text.count('torch.cuda.current_stream().synchronize()  # D-AR-7') == 1 and text.replace(source.H2D_SYNC, '') == source.upstream('prepare.py')
    with pytest.raises(ValueError):
        source.candidate_prepare('other')
    node = Node(tmp_path)
    rec = _runner(node, v).run(v, {}, None, 'screen', 'r', candidate_id='c0')
    assert rec['valid'] and 'D-AR-7' in rec['deviations'] and rec['files']['prepare.py'] == v.frozen_files['repo/prepare.py']
    written = node.ops.local(rec['candidate_dir']) / 'repo' / 'prepare.py'
    assert 'D-AR-7' in written.read_text(encoding='utf-8')
    assert 'D-AR-7' not in (node.ops.local(rec['candidate_dir']) / 'judge' / 'prepare.py').read_text(encoding='utf-8')
