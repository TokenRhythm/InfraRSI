"""DF16 (F-D2-R204d-2): `run` reads the doctor's run of the same configuration as the reference's repeat
(flow.doctor_run) before it judges the repeat or any candidate, so the loss check learns whether the job's draws differ
between runs; the result and the report record what the user was told (reference.loss_repeat, the repeat floor)."""
from infrarsi.project.doctor import doctor
from infrarsi.project.flow import run_project
from infrarsi.project.report import markdown
from infrarsi.verify.verifier import loss_repeat_note

from test_project import toy  # noqa: F401 - the toy project fixture


def test_the_doctor_run_is_the_reference_repeat_and_the_result_records_it(toy, tmp_path):
    rep = doctor(toy, log=lambda m: None)
    assert rep['valid'] and rep['loss_repeat']['repeat'] is None           # one run: no repeat yet
    assert rep['loss_repeat']['tolerance'] == 0.03 and not any('window mean' in w for w in rep['warnings'])   # a quiet loss
    logs = []
    res = run_project(toy, strategy='random', budget={'screens': 2}, out_root=tmp_path / 'out', log=logs.append)
    lr = res['reference']['loss_repeat']
    assert lr['repeat'] == 0.0 and lr['draws_vary'] is False and lr['tolerance'] == 0.03   # the toy repeats exactly
    assert lr['updates'] == 8 and lr['noise'] is not None and not lr['paired']
    assert res['search']['doctor_repeat_loss_deviation'] == 0.0              # the repeat floor, from the doctor's run
    assert not any(m.startswith('[infrarsi] loss check') for m in logs)    # nothing to say about a quiet loss
    assert loss_repeat_note(lr, res['tolerance']) is None and 'Loss check:' not in markdown(res)
    # the same result with D2-R204d's numbers (Wan): the report carries the line the run log printed
    res['reference']['loss_repeat'] = {'noise': 0.3152, 'updates': 20, 'tolerance': 0.2991, 'repeat': 0.3975,
                                       'draws_vary': True, 'paired': False}
    assert ("Loss check: this job's draws differ between runs (two runs of one configuration see other batches or noise: "
            "the reference's own repeat is 39.8% off per update): an unpaired 20-update window mean is held to 29.9% (3 x "
            "the reference's 31.5% per-update loss noise), not the manifest 3.0%; more updates per run (workload.steps, "
            'window) tighten it.') in markdown(res)                         # (U-2-4: what "draws" are)
