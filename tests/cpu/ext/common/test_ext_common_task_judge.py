import copy
import json
import math

import pytest
import yaml

from infrarsi.agent.judge import Verdict
from infrarsi.ext.common import guards
from infrarsi.ext.common.env_judge import EnvJudge
from infrarsi.ext.common.env_task import EnvTask, check_spec
from infrarsi.harness import signing
from x1_common_helpers import SHA, ToyRunner, local_ops, toy_spec, toy_task


# ---------------------------------------------------------------- EnvTask
def test_env_task_fields_and_budget():
    t = toy_task()
    assert t.split == 'external' and t.mutable_globs == ('solution/*',) and t.gpus == t.world_size == 4 and t.arm == 'T-IS'
    b = t.search_budget()
    assert (b.screens, b.llm_tokens, b.wall_seconds, b.gpu_hours, b.scoring_reserve_screens) == (6, 50_000, 36000.0, 10.0, 0.0)
    assert t.raw['id'] == 'ENV-T' and t.digest == toy_task().digest and t.frozen_digest == toy_task().frozen_digest
    assert toy_task(tau=1.0).digest != t.digest
    with pytest.raises(TypeError):
        EnvTask('ENV-T', 'toyenv')                       # keyword-only: the field order of EXT_INTERFACES with a split default


def test_env_task_split_default_and_loading(tmp_path):
    spec = toy_spec()
    spec.pop('split')
    assert EnvTask.from_dict(spec).split == 'external'
    p = tmp_path / 't.yaml'
    p.write_text(yaml.safe_dump(toy_spec()), encoding='utf-8')
    q = tmp_path / 't.json'
    q.write_text(json.dumps(toy_spec()), encoding='utf-8')
    assert EnvTask.load(p).to_dict() == EnvTask.load(q).to_dict()
    (tmp_path / 'bad.json').write_text(json.dumps({**toy_spec(), 'schema': 'other'}), encoding='utf-8')
    with pytest.raises(ValueError):
        EnvTask.load(tmp_path / 'bad.json')


@pytest.mark.parametrize('field,value', [
    ('split', 'dev'), ('action_space', 'free'), ('frozen_files', {'/abs': SHA}), ('frozen_files', {'a': 'nothex'}),
    ('mutable_globs', ['../x']), ('margin', {'kind': 'rsd', 'floor': 0.03}), ('margin', {'kind': 'fixed', 'value': 0}),
    ('budget', {'screens': 1}), ('world_size', 0), ('tau', float('nan')), ('knob_schema', {}), ('env', 'Bad-Env')])
def test_env_task_rejects_bad_specs(field, value):
    spec = toy_spec(**{field: value})
    assert check_spec(spec)
    with pytest.raises(ValueError):
        EnvTask.from_dict(spec)


def test_patch_space_needs_globs():
    assert any('mutable_globs' in e for e in check_spec(toy_spec(action_space='patch', mutable_globs=[])))


# ---------------------------------------------------------------- guards
def test_frozen_files_guard():
    exp = {'a': SHA, 'b': SHA}
    assert guards.frozen_files(exp, {'a': SHA, 'b': SHA}, {'a': SHA, 'b': SHA}).ok
    g = guards.frozen_files(exp, {'a': SHA, 'b': SHA}, {'a': SHA, 'b': '0' * 64})
    assert g.ids() == ['frozen_files'] and 'changed after' in g.rejections[0]['detail']
    g = guards.frozen_files(exp, {'a': SHA}, None)
    assert len(g.rejections) == 2 and {'b missing before the run', 'no after-run hashes'} == {r['detail'] for r in g.rejections}


def test_generic_guards():
    assert guards.outputs_present({'x': 10, 'y': 0}, ['x', 'y', 'z']).ids() == ['outputs_present'] * 2
    assert guards.exit_ok(0).ok and not guards.exit_ok(1).ok and not guards.exit_ok(None).ok
    assert 'timeout' in guards.exit_ok(0, timed_out=True).rejections[0]['detail']
    assert guards.budget(100, 600).ok and not guards.budget(700, 600).ok and not guards.budget(float('nan'), 600).ok
    assert guards.touched_outside({'solution/a.py': '', 'score.py': ''}, ('solution/*',)) == ['score.py']
    r = guards.tolerance_report('X', 0.5, 0.1, veto=False)
    assert r.ok and r.reports[0]['exceeded'] is True
    assert not guards.tolerance_report('X', 0.5, 0.1, veto=True).ok
    c = guards.combine(guards.exit_ok(1), None, r)
    assert c.ids() == ['exit_ok'] and len(c.reports) == 1


# ---------------------------------------------------------------- EnvJudge
def _receipt(tmp_path, overlay=None, config=None):
    task = toy_task()
    runner = ToyRunner(local_ops(tmp_path))
    return runner.run(task, config or {'workers': 2}, overlay, 'screen', 'r1', candidate_id='c1')


def test_env_judge_accepts_signed_valid_receipts(tmp_path):
    rec = _receipt(tmp_path)
    v = EnvJudge('toyenv')(rec, None, None)
    assert isinstance(v, Verdict) and v.valid and v.step_time == 50.0 and v.numerics_class == 'env' and v.source == 'env:toyenv'
    direct = EnvJudge(rec, None, None)                   # the EXT_INTERFACES spelling evaluates at once
    assert isinstance(direct, Verdict) and direct.valid
    assert EnvJudge(rec, rec, None).reward == 0.0


def test_env_judge_never_turns_invalid_into_valid(tmp_path):
    rec = _receipt(tmp_path)
    forged = copy.deepcopy(rec)
    forged['metric']['value'] = forged['step_time']['mean_window_max_rank_s'] = 1.0
    assert not EnvJudge()(forged).valid and 'signature' in EnvJudge()(forged).reasons[0]
    unsigned = {k: v for k, v in rec.items() if k != signing.SIGNATURE_FIELD}
    assert not EnvJudge()(unsigned).valid
    assert EnvJudge(require_signature=False)(unsigned).valid
    for mutate in (lambda r: r.update(judge_valid=False), lambda r: r['guard_rejections'].append({'id': 'G2', 'detail': 'x'}),
                   lambda r: r.update(schema='other'), lambda r: r['metric'].update(direction='max'),
                   lambda r: r['step_time'].update(mean_window_max_rank_s=7.0)):
        bad = copy.deepcopy(unsigned)
        mutate(bad)
        signing.sign(bad)
        assert not EnvJudge()(bad).valid
    assert not EnvJudge('optico')(rec).valid
    assert not EnvJudge()(None).valid and not EnvJudge()('x').valid


def test_guard_rejection_and_judge_failure_receipts(tmp_path):
    cheat = _receipt(tmp_path, overlay={'solution/run.py': 'cheat'})
    assert cheat['judge_valid'] is True and cheat['valid'] is False and cheat['metric']['value'] is None
    assert cheat['step_time']['mean_window_max_rank_s'] is None and cheat['metric']['raw']['judge_value'] == 50.0
    assert cheat['invalid_reasons'] == ['guard:G2: score.py changed'] and not EnvJudge()(cheat).valid
    crash = _receipt(tmp_path, overlay={'solution/run.py': 'crash'})
    assert crash['judge_valid'] is False and crash['invalid_reasons'] == ['judge: solution crashed']
    assert crash['guard_reports'] == [{'id': 'G5', 'checkpoint': 'ok'}]
    assert math.isclose(crash["cost"]["gpu_seconds"], 4 * (50.0 + 20.0)) and crash["cost"]["accounting_complete"] is True


def test_judge_success_without_metric_is_an_integrity_guard(tmp_path):
    """O / PROPOSED_CHANGES_X3 #2: judge_valid stays the environment judge's verdict; the missing metric is a guard."""
    class NoMetric(ToyRunner):
        def execute(self, task, config, overlay, purpose, run_id, attempt_id, candidate_id, **kw):
            run = super().execute(task, config, overlay, purpose, run_id, attempt_id, candidate_id, **kw)
            run.metric_value = float('nan')
            return run
    rec = NoMetric(local_ops(tmp_path)).run(toy_task(), {}, None, 'screen', 'r1')
    assert rec['judge_valid'] is True and rec['valid'] is False and rec['metric']['value'] is None
    assert [g['id'] for g in rec['guard_rejections']] == ['output_integrity']
    assert rec['invalid_reasons'] == [f'guard:output_integrity: {rec["guard_rejections"][0]["detail"]}']
    assert rec['metric']['raw']['judge_value'] == 'nan' and not EnvJudge()(rec).valid
