"""RebenchLlmfRunner (subclass of common.env_runner.EnvRunner): ext-receipt-v1 assembly over the shared base,
judge_valid vs guard_rejections kept separate, EnvJudge acceptance, per-arm uids and their containment before the
lease is released (B-02), G6 identity of a restored candidate (B-05), G9 (B-01), exact-content G2 (M-01), bounded
admission (H-05), and the remote command bound."""
import json
import math
import tempfile
from pathlib import Path

import pytest

from infrarsi.ext.common.env_judge import EnvJudge
from infrarsi.ext.common.remote_exec import LocalNodeOps
from infrarsi.ext.rebench_llmf import SCHEMA_RECEIPT, build_env_a_task
from infrarsi.ext.rebench_llmf import task as taskmod
from infrarsi.ext.rebench_llmf.runner import RebenchLlmfRunner, node_bound, winh_scorer
from infrarsi.harness import signing

import rebench_helpers as H

HERE = Path(__file__).resolve().parent


def _ids(rec):
    return {x['id'] for x in rec['guard_rejections']}


def test_valid_receipt_is_signed_and_accepted_by_envjudge():
    r, t = H.runner(H.node_result(H.task()))
    rec = r.run(t, {'device_train_microbatch_size': 8}, None, 'screen', 'run')
    assert rec['schema'] == SCHEMA_RECEIPT and rec['env'] == 'rebench_llmf'
    assert rec['valid'] is True and rec['judge_valid'] is True and rec['guard_rejections'] == []
    assert rec['metric']['value'] == 216.1 and rec['metric']['direction'] == 'min'
    assert rec['step_time']['mean_window_max_rank_s'] == rec['metric']['value']
    assert abs(rec['metric']['raw']['norm_diff'] - 3.41) < 1e-9
    assert rec['metric']['raw']['ln_seconds_score'] == math.log(216.1)
    assert rec['phases']['train'] == 177.6 and rec['phases']['finetune_total'] == 216.1
    assert {g['id'] for g in rec['guard_reports']} >= {'G1', 'G3', 'G5', 'G7', 'G8', 'G9'}
    assert signing.verify(rec)
    v = EnvJudge('rebench_llmf')(rec)
    assert v.valid and abs(v.step_time - 216.1) < 1e-9 and v.numerics_class == 'env'


def test_judge_valid_and_guards_are_recorded_separately():
    r, t = H.runner(H.node_result(H.task(), execution_time=221.0, norm_diff=5.1), enforce_g7=True)
    rec = r.run(t, {}, None, 'screen', 'run')
    assert rec['judge_valid'] is True       # score.py accepts (5.1 <= 5.3)
    assert rec['valid'] is False            # but InfraRSI G7 (5.0) rejects
    assert [x['id'] for x in rec['guard_rejections']] == ['G7']
    assert not EnvJudge('rebench_llmf')(rec).valid


def test_native_arm_does_not_apply_g7():
    r, t = H.runner(H.node_result(H.task('A-NAT'), execution_time=221.0, norm_diff=5.1), arm='A-NAT')
    rec = r.run(t, {}, None, 'screen', 'run')
    assert rec['valid'] is True and rec['guard_rejections'] == []
    assert 'D-RB-10' in rec['deviations']


# ---------------------------------------------------------------- B-02: per-arm uids, containment before release
@pytest.mark.parametrize('arm,uid,judge', [('A-NAT', 64101, 64111), ('A-ISW', 64102, 64112), ('A-RND', 64103, 64113),
                                           ('A-NAT2', 64104, 64114), ('A-IS2', 64105, 64115)])
def test_each_arm_runs_as_its_own_uid_pair_and_both_are_killed_before_release(arm, uid, judge):
    r, t = H.runner(H.node_result(H.task(arm)), arm=arm)
    rec = r.run(t, {}, None, 'screen', 'run')
    assert r.fake.jobs[-1]['uid'] == uid and r.fake.jobs[-1]['judge_uid'] == judge
    assert r.domain_uids() == [uid, judge] and r.describe()['judge_uid'] == judge
    assert r.ops.uid_kills == [uid, judge]                              # in the base's finally
    assert rec['containment']['ok'] is True and rec['lease_release']['kept'] == []
    assert rec['valid'] is True


def test_no_uid_is_nobody_or_shared_between_arms_and_unknown_arms_are_refused():
    uids = list(taskmod.ARM_UIDS.values()) + list(taskmod.JUDGE_UIDS.values())
    assert taskmod.FORBIDDEN_UID not in uids and len(set(uids)) == len(uids) == 10       # N-01: no judge uid is shared
    assert set(taskmod.ARM_UIDS) == set(taskmod.JUDGE_UIDS)
    with pytest.raises(ValueError):
        taskmod.arm_uid('A-IS')
    with pytest.raises(ValueError):
        taskmod.judge_uid('A-IS')


def test_surviving_uid_process_keeps_the_lease_and_invalidates():
    r, t = H.runner(H.node_result(H.task()))
    r.ops.uid_survivors[64102] = [4242]
    rec = r.run(t, {}, None, 'screen', 'run')
    assert rec['valid'] is False and 'containment' in _ids(rec)
    assert rec['lease_release']['kept']                                 # GPUs stay leased: nothing is re-admitted
    assert not EnvJudge('rebench_llmf')(rec).valid


def test_node_failure_still_kills_the_uids():
    t = build_env_a_task(arm='A-RND')

    def boom(ops, job):
        raise RuntimeError('node down')
    ops = LocalNodeOps(tempfile.mkdtemp(prefix='isboom'), node='WINH1')
    r = RebenchLlmfRunner(ops, task=t, node_runner=boom)
    rec = r.run(t, {}, None, 'screen', 'run')
    assert rec['valid'] is False and rec['cost']['accounting_complete'] is False
    assert any('runner_error' in reason for reason in rec['invalid_reasons'])
    assert ops.uid_kills == [64103, 64113]
    assert signing.verify(rec)


def test_prepare_is_a_cpu_job_followed_by_containment():
    r, t = H.runner(H.node_result(H.task()))
    out = r.prepare('run')
    assert out['mode'] == 'prepare' and r.fake.jobs[-1]['mode'] == 'prepare'
    assert set(out['uids_unused_before']) == {'64102', '64112'} and r.fake.jobs[-1]['judge_uid'] == 64112
    assert r.ops.uid_kills == [64102, 64112]
    assert not any(c[0] == 'acquire' for c in r.ops.calls)              # no GPU lease for the CPU-only prepare


def test_prepare_refuses_a_uid_in_use():
    from infrarsi.ext.common.remote_exec import ContainmentError
    r, _ = H.runner(H.node_result(H.task()))
    r.ops.uid_procs[64102] = [1234]
    with pytest.raises(ContainmentError):
        r.prepare('run')
    assert r.fake.jobs == []
    r2, _ = H.runner(H.node_result(H.task()))
    r2.ops.uid_passwd[64112] = 'someone'
    with pytest.raises(ContainmentError):
        r2.prepare('run')


# ---------------------------------------------------------------- G1 / G2 (M-01) / G3 / G8 / G9 (B-01)
def test_g2_tamper_invalidates():
    t = H.task()
    r, _ = H.runner(H.node_result(t, frozen_after={**H.frozen(t), 'score.py': 'z' * 64}))
    rec = r.run(t, {}, None, 'screen', 'run')
    assert rec['valid'] is False and 'G2' in _ids(rec)


def test_g2_env_sh_edited_before_scoring_is_caught_by_its_exact_expected_content():
    """M-01: an env.sh changed before the scoring matches itself before and after; only the exact expected content
    (derived from the run layout + model id) catches it."""
    t = H.task('A-NAT')
    edited = {**H.frozen(t), 'env.sh': 'f' * 64}
    r, _ = H.runner(H.node_result(t, frozen_before=edited, frozen_after=edited), arm='A-NAT')
    rec = r.run(t, {}, None, 'screen', 'run')
    assert rec['valid'] is False and 'G2' in _ids(rec)
    # the expected env.sh is the run's own: another run id's env.sh does not pass
    other = H.frozen(t, run_id='other')
    r2, _ = H.runner(H.node_result(t, frozen_before=other, frozen_after=other), arm='A-NAT')
    assert 'G2' in _ids(r2.run(t, {}, None, 'screen', 'run'))


def test_g2_env_sh_must_be_root_owned_read_only():
    t = H.task('A-NAT')
    r, _ = H.runner(H.node_result(t, env_sh={'uid': 64101, 'mode': '644'}), arm='A-NAT')
    rec = r.run(t, {}, None, 'screen', 'run')
    assert rec['valid'] is False and 'G2' in _ids(rec)


def test_g1_rejects_a_reference_that_is_not_safetensors_only():
    t = H.task()
    ref = {'outside_agent_tree': True, 'mode': '700', 'owner': 0, 'path': '/r', 'error': 'no model.safetensors'}
    r, _ = H.runner(H.node_result(t, reference=ref))
    assert 'G1' in _ids(r.run(t, {}, None, 'screen', 'run'))


def test_g3_barely_trained_invalidates():
    t = H.task()
    r, _ = H.runner(H.node_result(t, g3={'trained_vs_base': 1.0, 'ref_vs_base': 3.4}))
    rec = r.run(t, {}, None, 'screen', 'run')
    assert rec['valid'] is False and 'G3' in _ids(rec)


def test_g8_symlink_escape_invalidates():
    t = H.task()
    r, _ = H.runner(H.node_result(t, g8=H.g8_escape()))
    rec = r.run(t, {}, None, 'screen', 'run')
    assert rec['valid'] is False and 'G8' in _ids(rec)


def test_g9_unsafe_artifact_invalidates():
    t = H.task()
    g9 = {'ok': False, 'reason': "pytorch_model.bin: refused by torch.load(weights_only=True): UnpicklingError"}
    r, _ = H.runner(H.node_result(t, valid_judge=False, g9=g9))       # score.py: model_not_found -> no score
    rec = r.run(t, {}, None, 'screen', 'run')
    assert rec['valid'] is False and 'G9' in _ids(rec)


def test_g9_missing_sanitizer_record_is_a_rejection():
    t = H.task()
    node = H.node_result(t)
    node.pop('g9')
    r, _ = H.runner(node)
    assert 'G9' in _ids(r.run(t, {}, None, 'screen', 'run'))


def test_g9_covers_the_g4_dprime_candidate():
    t = H.task()
    bad_g4 = {'computed': False, 'unsafe_artifact': True, 'error': "cand(D') unsafe artifact: x is a dict"}
    r, _ = H.runner(H.node_result(t), g4=bad_g4)
    rec = r.run(t, {}, None, 'score', 'run')
    assert rec['valid'] is False and {'G9', 'G4'} <= _ids(rec)


def test_score_purpose_runs_g4_on_a_clean_slate_and_copy_is_caught():
    t = H.task()
    ok, _ = H.runner(H.node_result(t), g4=H.g4_ok())
    rec = ok.run(t, {}, None, 'score', 'run')
    assert rec['valid'] is True and any(x['id'] == 'G4' for x in rec['guard_reports'])
    assert ok.fake.jobs[-1]['clean_slate'] is True and ok.fake.jobs[-1]['run_g4'] is True
    copy, _ = H.runner(H.node_result(t), g4=H.g4_copy())
    rec2 = copy.run(t, {}, None, 'score', 'run')
    assert rec2['valid'] is False and 'G4' in _ids(rec2)
    screen, _ = H.runner(H.node_result(t))
    screen.run(t, {}, None, 'screen', 'run')
    assert screen.fake.jobs[-1]['clean_slate'] is False and screen.fake.jobs[-1]['run_g4'] is False


# ---------------------------------------------------------------- B-05: G6 identity of a restored candidate
def test_restore_binds_the_receipt_to_the_selected_snapshot_digest():
    t = H.task('A-NAT')
    sel = 'd' * 64
    r, _ = H.runner(H.node_result(t), arm='A-NAT', g4=H.g4_ok())
    rec = r.run(t, None, None, 'score', 'run', run_g4=True, restore_snapshot=f'/s/{sel}.tar', expected_digest=sel)
    job = r.fake.jobs[-1]
    assert job['restore_snapshot'] == f'/s/{sel}.tar' and job['expected_digest'] == sel and job['clean_slate']
    assert rec['candidate_digest'] == sel and rec['valid'] is True


def test_restored_tree_that_is_not_the_selection_is_rejected_by_g6():
    t = H.task('A-NAT')
    r, _ = H.runner(H.node_result(t), arm='A-NAT', g4=H.g4_ok(), restored_tree='e' * 64)
    rec = r.run(t, None, None, 'score', 'run', run_g4=True, restore_snapshot='/s/x.tar', expected_digest='d' * 64)
    assert rec['valid'] is False and 'G6' in _ids(rec) and rec['candidate_digest'] == 'e' * 64


def test_screen_of_the_working_tree_is_identified_by_its_tree_digest():
    t = H.task('A-NAT')
    r, _ = H.runner(H.node_result(t, tree_digest='a' * 64), arm='A-NAT')
    rec = r.run(t, None, None, 'screen', 'run', reuse_home=True)
    assert rec['candidate_digest'] == 'a' * 64 and rec['snapshot']['tree_digest'] == 'a' * 64


# ---------------------------------------------------------------- driver-side rejections (never reach the node)
def test_invalid_knobs_never_reach_the_node():
    t = H.task()
    r, _ = H.runner(H.node_result(t))
    rec = r.run(t, {'device_train_microbatch_size': 99}, None, 'screen', 'run')
    assert rec['valid'] is False and r.fake.jobs == []
    assert any('invalid knobs' in reason for reason in rec['invalid_reasons'])


def test_non_canonical_overlay_path_never_reaches_the_node():
    t = H.task()
    r, _ = H.runner(H.node_result(t))
    rec = r.run(t, {}, {'scripts/../composer/trainer/trainer.py': 'x = 1\n'}, 'screen', 'run')
    assert rec['valid'] is False and r.fake.jobs == []
    assert any('canonical' in reason for reason in rec['invalid_reasons'])


# ---------------------------------------------------------------- H-05: bounded admission
def test_runner_declares_its_real_worst_case_and_bounded_cap_uses_it():
    """H-05 (R2): the declared worst case is the real maximum of an uncapped screen: the node job's hard bound
    (480 + 600) + the node timeout's kill grace (60) + the transport waits (70) + lease / shipping / containment (60)."""
    from infrarsi.ext.rebench_llmf import runner as rmod
    r, _ = H.runner(H.node_result(H.task()))
    assert r.worst_case() == 480 + 600 + 60 + 70 + 60 == 1270
    assert node_bound({'run_g4': False}) + rmod.DRIVER_RESERVE_S == r.worst_case()
    assert r.bounded_cap(34.0, 4, 57_600, 300) == 1270
    assert r.bounded_cap(34.0, 4, 1269, 300) is None                   # the worst case no longer fits: not started
    assert r.bounded_cap(0.3, 4, 57_600, 300) is None                  # 4 GPUs x 1270 s > 0.3 GPU-h


def test_wall_cap_shortens_the_finetune_timeout_and_the_node_bound():
    t = H.task()
    r, _ = H.runner(H.node_result(t))
    r.run(t, {}, None, 'screen', 'run', wall_cap=1000)
    job = r.fake.jobs[-1]
    assert job['bound_s'] == 1000 - 190 and job['score_timeout'] == 1000 - 190 - 600
    capped, _ = H.runner(H.node_result(t))
    capped.run(t, {}, None, 'screen', 'run', wall_cap=capped.worst_case())   # the admitted cap = the worst case
    assert capped.fake.jobs[-1]['bound_s'] == 1080 and capped.fake.jobs[-1]['score_timeout'] == 480
    full, _ = H.runner(H.node_result(t))
    full.run(t, {}, None, 'screen', 'run')
    assert full.fake.jobs[-1]['score_timeout'] == 480 and full.fake.jobs[-1]['bound_s'] is None
    tiny, _ = H.runner(H.node_result(t))
    rec = tiny.run(t, {}, None, 'screen', 'run', wall_cap=700)
    assert rec['valid'] is False and tiny.fake.jobs == []


def test_job_carries_score_py_default_timeout_and_the_sanitizer():
    t = H.task()
    r, _ = H.runner(H.node_result(t))
    r.run(t, {}, None, 'screen', 'run')
    job = r.fake.jobs[-1]
    assert job['score_timeout'] == 480                       # score.py's own default, not a driver-chosen limit
    assert job['sanitize_py'].endswith('/sanitize.py') and job['uid'] == 64102


def test_failed_finetune_is_invalid_with_null_metric():
    t = H.task()
    r, _ = H.runner(H.node_result(t, valid_judge=False, trained=False))
    rec = r.run(t, {}, None, 'screen', 'run')
    assert rec['valid'] is False and rec['judge_valid'] is False
    assert rec['metric']['value'] is None and rec['step_time']['mean_window_max_rank_s'] is None
    assert 'G9' not in _ids(rec) and 'G3' not in _ids(rec)  # nothing to sanitise or measure: only score.py's verdict
    assert not EnvJudge('rebench_llmf')(rec).valid


def test_deviations_recorded():
    r, t = H.runner(H.node_result(H.task()))
    rec = r.run(t, {}, None, 'screen', 'run')
    for dev in ('D-RB-1', 'D-RB-2', 'D-RB-8', 'D-RB-9', 'D-RB-11', 'D-RB-13', 'D-RB-15'):
        assert dev in rec['deviations']


def test_equivalence_with_the_follow_up_2_gpu_node_result():
    """The real WINH1 node result of the Follow-up 2 path (A-ISW uid 64102, unprivileged sanitizer, root loads
    safetensors only; run x2_fu2_verify_20261020_234021, run_g4=False) through the integrated runner: valid, exec
    215.61 s (within 3 % of 215.97), norm_diff 4.22, G2 exact (env.sh of that run's home), G1/G3/G5/G7/G8/G9 reported,
    no rejection, no uid process left, EnvJudge valid."""
    node = json.loads((HERE / 'gpu_node_result_fu2.json').read_text(encoding='utf-8'))
    t = build_env_a_task(arm='A-ISW', action_space='knobs+patch')
    ops = LocalNodeOps(tempfile.mkdtemp(prefix='isfu2'), node='WINH1')
    r = RebenchLlmfRunner(ops, task=t, enforce_g7=True, node_runner=lambda ops, job: node)
    rec = r.run(t, {}, None, 'screen', 'x2_fu2_verify_20261020_234021')
    assert rec['valid'] is True and rec['guard_rejections'] == [], rec['invalid_reasons']
    assert abs(rec['metric']['value'] - 215.611807) < 1e-6 and abs(rec['metric']['value'] - 215.97) / 215.97 <= 0.03
    assert abs(rec['metric']['raw']['norm_diff'] - 4.2190143208893005) < 1e-9
    assert {g['id'] for g in rec['guard_reports']} >= {'G1', 'G3', 'G5', 'G7', 'G8', 'G9'}
    assert node['uid'] == 64102 and node['g9']['ok'] and node['env_sh'] == {'uid': 0, 'mode': '644', 'regular': True}
    assert set(node['remaining_uid_processes'].values()) == {0}
    assert EnvJudge('rebench_llmf')(rec).valid
    # the same node result under another run id fails G2: env.sh is bound to its run's home
    other = r.run(t, {}, None, 'screen', 'another_run')
    assert other['valid'] is False and 'G2' in _ids(other)


def test_equivalence_with_the_follow_up_2_g4_gpu_node_result():
    """The real WINH1 locked-style scoring with G4 (run x2_fu2_g4_20261021_000620: purpose score / clean slate,
    run_g4=True, g4_share=False; D as 64102, ref'(D') as the then-shared judge uid 64100 (since R2 N-01 A-ISW's own
    judge 64112; same code path), cand(D') as 64102, all sanitised): valid, exec 215.45 s,
    G4 computed with ratio 1.00 and score.py valid on D', ref' cache root-owned read-only, every guard passing."""
    node = json.loads((HERE / 'gpu_node_result_fu2_g4.json').read_text(encoding='utf-8'))
    t = build_env_a_task(arm='A-ISW', action_space='knobs+patch')
    ops = LocalNodeOps(tempfile.mkdtemp(prefix='isfu2g4'), node='WINH1')
    r = RebenchLlmfRunner(ops, task=t, enforce_g7=True, node_runner=lambda ops, job: node)
    rec = r.run(t, {}, None, 'score', 'x2_fu2_g4_20261021_000620', run_g4=True)
    assert rec['valid'] is True and rec['guard_rejections'] == [], rec['invalid_reasons']
    assert abs(rec['metric']['value'] - 215.453556) < 1e-6
    g4 = next(g for g in rec['guard_reports'] if g['id'] == 'G4')
    assert abs(g4['ratio'] - 1.0) < 1e-3 and g4['refprime_cached'] is False
    assert {g['id'] for g in rec['guard_reports']} >= {'G1', 'G3', 'G4', 'G5', 'G7', 'G8', 'G9'}
    assert node['g4']['score_py_valid'] is True and node['g4']['g4_shared'] is False
    assert node['g4']['refprime_owner'] == {'uid': 0, 'mode': '555', 'regular': False} and node['clean_slate'] is True
    assert set(node['remaining_uid_processes'].values()) == {0} and EnvJudge('rebench_llmf')(rec).valid


def test_follow_up_1_node_results_no_longer_pass():
    """The Follow-up 1 GPU node result (nobody path, raw candidate loaded by root, env.sh of that run's home) is
    rejected under the Follow-up 2 rules: no sanitizer record (G9) and env.sh not the expected content (G2)."""
    node = json.loads((HERE / 'gpu_node_result_nobody.json').read_text(encoding='utf-8'))
    t = build_env_a_task(arm='A-ISW', action_space='knobs+patch')
    ops = LocalNodeOps(tempfile.mkdtemp(prefix='isfu1'), node='WINH1')
    r = RebenchLlmfRunner(ops, task=t, enforce_g7=True, node_runner=lambda ops, job: node)
    rec = r.run(t, {}, None, 'screen', 'run')
    assert rec['judge_valid'] is True and rec['valid'] is False and {'G9', 'G2'} <= _ids(rec)


# ---------------------------------------------------------------- the remote command
class _RecordingOps:
    """Records the remote commands winh_scorer issues; returns a canned scorer payload."""

    def __init__(self, result):
        self.result, self.cmds, self.written, self.modes = result, [], {}, []

    def mkdirs(self, paths, **kw):
        return {}

    def write_files(self, files, **kw):
        self.written.update(files)
        self.modes.append(kw.get('mode'))
        return {}

    def sh(self, cmd, timeout=120):
        self.cmds.append((cmd, timeout))
        return {'code': 0, 'stdout': '', 'stderr': ''}

    def read_json(self, path, default=None):
        return {'ok': True, 'result': self.result}


def _job(**kw):
    base = {'run_id': 'r', 'attempt_id': 'a1', 'run_dir': '/data/infrarsi/ext/runs/rebench_llmf/r',
            'out': '/data/infrarsi/ext/runs/rebench_llmf/r/results/a1_node_result.json', 'timeout_s': 0,
            'score_timeout': 480}
    base.update(kw)
    return base


def test_winh_scorer_bounds_the_remote_command_and_ships_the_sanitizer():
    for kw, trainings in (({'run_g4': False}, 1), ({'run_g4': True, 'g4_share': True}, 2),
                          ({'run_g4': True, 'g4_share': False}, 3)):
        ops = _RecordingOps({'judge': None})
        assert winh_scorer(ops, _job(**kw)) == {'judge': None}
        cmd, _ = ops.cmds[-1]
        assert cmd.startswith('mkdir ') and f' || exit 75; timeout --signal=TERM --kill-after=60 {trainings * 480 + 600} ' in cmd
        assert ops.cmds[-1][1] == trainings * 480 + 600 + 70                # our wait ends 10 s after the hard kill
        assert 'score_runner.py' in cmd
        shipped = {p.rsplit('/', 1)[1] for p in ops.written}
        assert {'score_runner.py', 'sanitize.py', 'permute.py'} <= shipped     # shipped every time
        assert 0o600 in ops.modes                                           # the job file is root-only
    ops = _RecordingOps({'judge': None})
    winh_scorer(ops, _job(bound_s=880))
    assert ' || exit 75; timeout --signal=TERM --kill-after=60 880 ' in ops.cmds[-1][0]
    assert node_bound(_job(bound_s=880)) == 880


# ---------------------------------------------------------------- Codex W9 R2 N-01: two co-located arms
def test_two_arms_on_disjoint_gpus_never_touch_each_others_uids():
    """A-NAT on GPUs 0-3 and A-ISW on GPUs 4-7 of one node under the orchestrator's group lease: each attempt's
    containment (and each node job) names only its own arm's pair, so A-NAT's cleanup cannot reach A-ISW's judge."""
    import infrarsi.ext.common.remote_exec as rx
    ops = LocalNodeOps(tempfile.mkdtemp(prefix='istwoarms'), node='WINH1')
    group = rx.ext_lease('group-1', 'run:group-1', 'WINH1', 'rebench_llmf', range(8), scope='run')
    assert ops.acquire(rx.node_lease_path('WINH1'), group)['ok']
    runs = {}
    for arm, gpus in (('A-NAT', (0, 1, 2, 3)), ('A-ISW', (4, 5, 6, 7))):
        t = H.task(arm)
        fake = H.FakeNodeRunner(H.node_result(t, run_id=f'{arm}-run'))
        r = RebenchLlmfRunner(ops, task=t, gpus=gpus, enforce_g7=(arm == 'A-ISW'), node_runner=fake, lease_mode='held',
                              held_run_id='group-1')
        before = len(ops.uid_kills)
        rec = r.run(t, {}, None, 'screen', f'{arm}-run')
        runs[arm] = {'kills': ops.uid_kills[before:], 'job': fake.jobs[-1], 'rec': rec}
    nat, isw = runs['A-NAT'], runs['A-ISW']
    assert nat['rec']['valid'] and isw['rec']['valid']
    assert nat['kills'] == [64101, 64111] and isw['kills'] == [64102, 64112]
    assert {nat['job']['uid'], nat['job']['judge_uid']}.isdisjoint({isw['job']['uid'], isw['job']['judge_uid']})
    assert nat['job']['gpus'] == '0,1,2,3' and isw['job']['gpus'] == '4,5,6,7'
