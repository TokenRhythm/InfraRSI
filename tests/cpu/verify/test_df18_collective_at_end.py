"""D2-R204f (2026-10-25, Wan 2.1 through accelerate, 8 x H100): 3 of 5 promotion re-runs - two of them of the reference
configuration - and the doctor run logged an NCCL error ("unhandled system error" of a pending all-reduce) as the ranks
finished, with every update done and exit 0; `no_errors` failed them, so no candidate could pass promotion. Collective
lines in a run that logged every update by itself and exited 0, with nothing else wrong, are reported, not a failure;
anywhere else they still fail the check."""
from infrarsi.harness import signing
from infrarsi.verify import Verifier

from test_verifier import receipt, status


def detail(v, cid):
    return {c['id']: c['detail'] for c in v.checks}[cid]


def test_collective_lines_at_the_end_of_a_complete_run_are_reported_not_a_failure():
    ref = receipt('ref', steps=10)
    v = Verifier(0.03)(receipt('c1', steps=10, events={'collective': 3}), None, ref)
    assert status(v, 'no_errors') == 'pass' and v.valid
    assert detail(v, 'no_errors').startswith('3 collective/NCCL error line(s) in a run that logged every update')


def test_anywhere_else_they_still_fail():
    ref, vf = receipt('ref', steps=10), Verifier(0.03)
    for kw in ({'completed': 8}, {'exit_code': 1}, {'events': {'collective': 3, 'traceback': 1}},
               {'events': {'collective': 3, 'oom': 1}}):
        r = receipt('c', steps=10, **{'events': {'collective': 3}, **kw})
        v = vf(r, None, ref)
        assert status(v, 'no_errors') == 'fail' and not v.valid, kw
    probe = receipt('p', steps=10, events={'collective': 3}, completed=2, stop_after=2, planned_stop=True, sign=False)
    probe['evidence']['per_step'] = probe['evidence']['per_step'][:2]
    probe = signing.sign(probe)
    assert status(vf(probe, None, ref), 'no_errors') == 'fail'            # a planned stop: its own lines, strict
    nan = receipt('c', steps=10, events={'collective': 3}, loss=lambda s: 'nan' if s == 5 else 2.0 - 0.01 * s)
    assert status(vf(nan, None, ref), 'no_errors') == 'fail'


def test_a_rank_prefixed_traceback_after_the_last_update_still_fails():
    # DF18_r1: torchrun / accelerate prefix every line with its rank; the traceback guard must see such a traceback
    from infrarsi.project import signals as sig
    lines = ['[rank7]: Traceback (most recent call last):', '[rank7]:   File "train.py", line 150, in save_checkpoint',
             '[rank7]: torch.distributed.DistBackendError: NCCL error: unhandled system error',
             '[rank7]: ncclSystemError: System call failed']
    ev = sig.scan_events([(0.0, x) for x in lines])
    assert ev['traceback'] == 1 and ev['collective'] == 2
    v = Verifier(0.03)(receipt('c', steps=10, events=ev), None, receipt('ref', steps=10))
    assert status(v, 'no_errors') == 'fail' and not v.valid
