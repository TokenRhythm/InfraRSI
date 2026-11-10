"""The job's own slow-path notes reach the proposer (2026-10-28): every Qwen3.5-9B run logged "`causal_conv1d_fn` is
falling back to its reference PyTorch implementation ... This is correct but much slower" - the one bottleneck only code
reaches - and no part of the search saw it. scan_events collects such lines (de-duplicated, prefixes stripped, at most
five) and the project digest shows them next to the verdict, the run's own or else the reference run's."""
import yaml

from infrarsi.project import Project
from infrarsi.project import signals as sig
from infrarsi.project.hooks import ProjectHooks

NOTE = ('`causal_conv1d_fn` is falling back to its reference PyTorch implementation because `causal_conv1d` is not '
        'installed. This is correct but much slower; install `causal_conv1d` for the optimized kernel.')


def test_scan_events_collects_distinct_slow_path_notes():
    lines = [(0, f'[transformers] {NOTE}'), (1, f'[rank3]: [transformers] {NOTE}'), (2, 'step 1 loss 2.0'),
             (3, '[rank0]:W1028 09:31:31.963000 3463726 torch/x.py:987] Flash attention is not available, using math'),
             (4, 'an ordinary line')]
    notes = sig.scan_events(lines)['perf_notes']
    assert notes == [NOTE, 'Flash attention is not available, using math']
    assert 'perf_notes' not in sig.scan_events([(0, 'step 1 loss 2.0')])
    many = [(i, f'kernel {i} is not installed, using the slow path') for i in range(9)]
    assert len(sig.scan_events(many)['perf_notes']) == sig.PERF_NOTES_MAX
    assert 'perf_notes' not in sig.scan_events([(0, f'[transformers] {NOTE}')], ignore='causal_conv1d')


def project(tmp_path, mutable=True):
    d = tmp_path / ('notes-code' if mutable else 'notes-knobs')
    d.mkdir()
    (d / 'job.py').write_text('print("step 1 loss 2.0")\n', encoding='utf-8')
    doc = {'schema': 'infrarsi-project-v1', 'name': 'notes-toy',
           'launch': {'command': '{python} job.py --steps {steps} {args}', 'timeout_s': 60},
           'knobs': {'mode': {'type': 'str', 'flag': '--mode', 'choices': ['a', 'b'], 'default': 'a'}},
           'workload': {'steps': 4, 'window': [2, 4], 'invariants': {'global_batch': '8'}}, 'hardware': {'gpus': 1}}
    if mutable:
        doc['code'] = {'mutable': ['job.py']}
    (d / 'infrarsi.yaml').write_text(yaml.safe_dump(doc), encoding='utf-8')
    return Project.load(d)


def test_python_warnings_are_one_note_and_an_autotune_skip_is_no_oom():
    lines = [(0, '/x/apex/norm.py:12: UserWarning: Apex is not installed. Falling back to Torch Norm'),
             (1, '  warnings.warn("Apex is not installed. Falling back to Torch Norm")'),
             (2, '/y/other.py:40: UserWarning: Apex is not installed. Falling back to Torch Norm')]
    assert sig.scan_events(lines)['perf_notes'] == ['Apex is not installed. Falling back to Torch Norm']
    ev = sig.scan_events([(0, 'No valid triton configs. OutOfMemoryError: out of resource: triton_mm Required: 262144')])
    assert ev['oom'] == 0 and 'fatal' not in ev


def test_a_knob_only_arm_sees_no_notes(tmp_path):
    p = project(tmp_path, mutable=False)
    hooks = ProjectHooks(p, p.task(), tolerance=0.03)
    hooks.reference_notes = [NOTE]
    assert 'job_log_notes' not in hooks.receipt_digest({'evidence': {'events': {'perf_notes': [NOTE]}}}, None)


def test_the_digest_shows_the_notes_next_to_the_verdict(tmp_path):
    p = project(tmp_path)
    hooks = ProjectHooks(p, p.task(), tolerance=0.03)
    d = hooks.receipt_digest({'evidence': {'events': {'perf_notes': [NOTE]}}}, None)
    assert d['job_log_notes'] == [NOTE] and list(d).index('job_log_notes') <= 2
    hooks.reference_notes = ['the reference run said so']
    assert hooks.receipt_digest({'evidence': {'events': {}}}, None)['job_log_notes'] == ['the reference run said so']
    hooks.reference_notes = None
    assert 'job_log_notes' not in hooks.receipt_digest({'evidence': {'events': {}}}, None)


def test_the_digest_keeps_its_cap_and_its_order(tmp_path):
    from infrarsi.agent.context import estimate_tokens
    import json
    p = project(tmp_path)
    hooks = ProjectHooks(p, p.task(), tolerance=0.03)
    hooks.reference_phases = {'unit': 'ms', 'mean_ms': {'compute': 900.0, 'comm': 60.0}, 'step_ms': 1000.0,
                              'share': {'compute': 0.9, 'comm': 0.06}, 'top': ['compute', 'comm']}
    notes = [f'note {i}: ' + 'x' * 290 + ' is not installed' for i in range(5)]
    rec = {'evidence': {'events': {'perf_notes': notes}}, 'summary': 'y' * 4000}
    d = hooks.receipt_digest(rec, None, max_tokens=1500)
    assert estimate_tokens(json.dumps(d, default=str)) <= 1500
    keys = list(d)
    assert keys.index('job_log_notes') < keys.index('step_phases') and d['job_log_notes'][0] == notes[0]
