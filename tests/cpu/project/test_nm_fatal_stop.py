"""The fatal-line stop (2026-10-28, new-model campaign): one rank's OOM or failed collective left the other ranks
waiting in a collective until the job's own timeout - 2,163-3,646 s per failed candidate of an 8-GPU Qwen3.5-9B search
(23.1 of the arm's 24.0 search GPU-h) and the 2,400 s launch timeout on 64 GPUs - while a run whose ranks all fail at
once ends in about 100 s. The runner now stops a run that logged a fatal line and made no update for
INFRARSI_FATAL_GRACE_S (120 s) and records it as a failed run (evidence.fatal_stop), not as an abort for speed; the
reported cause is the fatal line, and the manifest's signals.ignore keeps a known-harmless start-up traceback out of it."""
import time

import yaml

from infrarsi.project import Project, ProjectRunner
from infrarsi.project import signals as sig
from infrarsi.project.flow import make_verifier
from infrarsi.project.runner import FatalWatch, _either

DSA = "AttributeError: module 'cudnn' has no attribute 'DSA'"
OOM = '[rank5]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.95 GiB.'


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def lines(*texts, t0=0.0):
    return [(t0 + i, x) for i, x in enumerate(texts)]


def watch(clock, ignore=None, grace=120):
    return FatalWatch(sig.LogParser(None, ignore, False), grace, clock=clock, ignore=ignore)


def test_scan_events_names_the_fatal_line_and_skips_ignored_ones():
    log = lines(DSA, 'step 1 | loss 2.0', OOM, 'RuntimeError: DeepEP error: CPU recv timeout')
    ev = sig.scan_events(log)
    assert ev['error'] == DSA and ev['fatal'] == OOM and ev['oom'] == 1
    ev = sig.scan_events(log, ignore="has no attribute 'DSA'")
    assert ev['error'].startswith('torch.OutOfMemoryError') and ev['fatal'] == OOM
    assert 'fatal' not in sig.scan_events(lines(DSA, 'step 1 | loss 2.0'))


def test_a_fatal_line_without_updates_after_it_stops_the_run_after_the_grace():
    c = Clock()
    w = watch(c)
    buf = lines('step 1 loss 2.0', 'step 2 loss 1.9')
    c.t = 10
    assert not w(buf)
    buf += lines(OOM, t0=2)
    c.t = 20
    assert not w(buf)
    c.t = 139
    assert not w(buf)
    c.t = 140
    assert w(buf) and w.line == OOM
    assert w([]) is True                                   # stays fired


def test_updates_after_a_fatal_looking_line_keep_the_run_going():
    c = Clock()
    w = watch(c)
    buf = lines('NCCL error: unhandled system error (a warning this job survives)')
    c.t = 1
    w(buf)
    for k in range(1, 30):                                 # an update every 60 s: never 120 s without one
        c.t = 1 + 60 * k
        buf = buf + [(c.t, f'step {k} loss 1.5')]
        assert not w(buf)
    c.t += 121
    assert w(buf)                                          # then it hangs: stopped 120 s after its last update


def test_an_ignored_line_is_not_fatal_and_two_nodes_are_read_separately():
    c = Clock()
    w = watch(c, ignore="has no attribute 'DSA'")
    node0 = lines('rank0 start', 'step 1 loss 2.0')
    node1 = lines('rank8 start', DSA)
    c.t = 5
    w(node0)
    w(node1)
    c.t = 500
    assert not w(node0) and not w(node1)                   # the harmless line: no stop, however long the wait
    node1 = node1 + [(9.0, OOM)]
    c.t = 510
    w(node0)
    w(node1)
    node0 = node0 + [(9.5, 'step 2 loss 1.9')]            # rank 0 logs an update in flight after the OOM elsewhere
    c.t = 515
    w(node0)
    w(node1)
    c.t = 634
    assert not w(node0)
    c.t = 635
    assert w(node0)                                        # 120 s after the last update


def test_either_asks_every_predicate():
    seen = []

    def a(_):
        seen.append('a')
        return True

    def b(_):
        seen.append('b')
        return False
    assert _either(a, b)([]) is True and seen == ['a', 'b']
    assert _either(None, b) is b and _either(None, None) is None


JOB = r'''
import sys, time
n = int(sys.argv[sys.argv.index("--steps") + 1])
mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "ok"
print("AttributeError: module 'cudnn' has no attribute 'DSA'", flush=True)
for s in range(1, n + 1):
    if mode == "hang" and s == 3:
        print("[rank1]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.", flush=True)
        time.sleep(60)
    time.sleep(0.05)
    print(f"step {s} loss {2.0 - 0.01 * s:.4f}", flush=True)
'''


def project(tmp_path, ignore=None):
    d = tmp_path / 'fatal'
    d.mkdir()
    (d / 'job.py').write_text(JOB, encoding='utf-8')
    doc = {'schema': 'infrarsi-project-v1', 'name': 'fatal-toy',
           'launch': {'command': '{python} job.py --steps {steps} {args}', 'timeout_s': 90},
           'knobs': {'mode': {'type': 'str', 'flag': '--mode', 'choices': ['ok', 'hang'], 'default': 'ok'}},
           'workload': {'steps': 6, 'window': [2, 6], 'invariants': {'global_batch': '8'}}, 'hardware': {'gpus': 1}}
    if ignore:
        doc['signals'] = {'ignore': ignore}
    (d / 'infrarsi.yaml').write_text(yaml.safe_dump(doc), encoding='utf-8')
    return Project.load(d)


def test_the_runner_stops_a_hung_failed_job_and_records_a_failure(tmp_path, monkeypatch):
    monkeypatch.setenv('INFRARSI_FATAL_GRACE_S', '2')
    p = project(tmp_path, ignore="has no attribute 'DSA'")
    r, v, t = ProjectRunner(p, out_dir=tmp_path / 'runs'), make_verifier(p), p.task()
    ref = r.run(t, p.reference_config(), None, 'reference', 'r', candidate_id='c0')
    assert v(ref, None, ref).valid and 'fatal_stop' not in ref['evidence']
    t0 = time.perf_counter()
    bad = r.run(t, dict(p.reference_config(), mode='hang'), None, 'screen', 'r', candidate_id='c1')
    assert time.perf_counter() - t0 < 40                   # not the job's 60 s sleep, nor the 90 s timeout
    ev = bad['evidence']
    assert ev['fatal_stop']['line'].startswith('[rank1]: torch.OutOfMemoryError') and ev['stopped_early'] is False
    assert ev['events']['error'].startswith('torch.OutOfMemoryError')       # the ignored start-up line is not the cause
    vb = v(bad, ref, ref)
    done = {c['id']: c for c in vb.checks}['completes']
    assert not vb.valid and done['status'] == 'fail' and done['detail'].startswith('stopped 2 s after a fatal line')


def test_the_completes_text_names_the_fatal_line_once():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'verify'))   # the verifier tests' receipt helper
    from infrarsi.verify import Verifier
    from test_verifier import receipt
    ref = receipt('ref', steps=10)
    oom = '[rank3]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.50 GiB.'
    ev = {'error': 'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.50 GiB.', 'fatal': oom, 'oom': 1}
    v = Verifier(0.03)(receipt('c', steps=10, exit_code=1, completed=0, events=ev), None, ref)
    detail = {c['id']: c['detail'] for c in v.checks}['completes']
    assert 'first fatal line' not in detail and detail.count('OutOfMemoryError') == 1
    ev = {'error': "AttributeError: module 'cudnn' has no attribute 'DSA'", 'fatal': oom, 'oom': 1}
    v = Verifier(0.03)(receipt('c', steps=10, exit_code=1, completed=0, events=ev), None, ref)
    assert 'first fatal line: ' + oom in {c['id']: c['detail'] for c in v.checks}['completes']


def test_deepep_failures_are_fatal_and_its_status_lines_are_not():
    for line in ('DeepEP combine RDMA receiver timeout, channel: 3, RDMA: 2, nvl: 4',
                 'DeepEP timeout check failed: rank = 3, thread = 1, value = 0)',
                 '[rank3]: RuntimeError: DeepEP error: CPU recv timeout'):
        assert sig.FATAL_RE.search(line), line
    for line in ('            (experts): GroupedExpertsDeepEP()', '    dispatcher: deepep'):
        assert not sig.FATAL_RE.search(line), line


def test_two_nodes_with_the_same_first_line_are_read_apart():
    c = Clock()
    w = watch(c)
    node0 = [(1.0, 'start'), (2.0, 'step 1 loss 2.0')]
    node1 = [(1.0, 'start'), (3.0, 'CUDA error: unspecified launch failure')]
    c.t = 10
    w(node0, node=0)
    w(node1, node=1)
    c.t = 131
    assert w(node0, node=0) and w.line.startswith('CUDA error')


def test_either_passes_the_node_on():
    seen = []

    def a(lines, node=None):
        seen.append(('a', node))
        return False

    def b(lines, node=None):
        seen.append(('b', node))
        return True
    assert _either(a, b)([], node=3) is True and seen == [('a', 3), ('b', 3)]


def test_the_node_index_goes_only_to_a_predicate_that_takes_it():
    from infrarsi.project import executor as ex
    assert ex.takes_node(watch(Clock())) and ex.takes_node(lambda lines, node=None: 0)
    assert not ex.takes_node(lambda lines: 0) and not ex.takes_node(len)
    seen = []

    def plain(lines):
        seen.append('plain')
        return False
    assert _either(plain, watch(Clock()))([], node=1) is False and seen == ['plain']
