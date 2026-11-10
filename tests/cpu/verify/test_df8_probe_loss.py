"""The loss check of a run stopped before the window's end (FABLE_DF8_DECISION A, F-D2-R204c-2): the manifest tolerance
is for the window mean; a probe's 2-update mean is held to the tolerance or, if larger, 3 x the noise of such a mean,
read from the reference's own per-update losses (consecutive differences, so a smooth trend adds nothing). D2-R204c
(Wan 2.1 through DiffSynth): the reference's loss swings 0.03..0.10 per update, the probes of changes that cannot touch
the numerics deviated 22-53 % over 2 updates and the reference's own doctor repeat 42 %, while two full runs agreed to
1.1 % on the window mean. The sft_loop bf16 + compile break (3.07 vs 1.00 over 2 updates, F1-S1) must still fail at the
probe, a noiseless loss widens nothing (the toy's 6 % trap), and a full run keeps the manifest tolerance."""
import math

import pytest

from infrarsi.harness import signing
from infrarsi.verify import Verifier
from infrarsi.verify.verifier import LOSS_NOISE_Z, loss_noise

from test_verifier import receipt, status

# per-update losses of the D2-R204c reference (Wan, window [11, 30]) and of the F1-S1 reference (sft_loop, Qwen2.5-0.5B)
WAN = [0.101, 0.0955, 0.1016, 0.0957, 0.0945, 0.0458, 0.0492, 0.0943, 0.0502, 0.0457, 0.0417, 0.0478, 0.0468, 0.0845,
       0.0468, 0.0425, 0.0904, 0.0601, 0.0493, 0.0768, 0.1033, 0.054, 0.0427, 0.049, 0.0395, 0.0907, 0.0452, 0.0462,
       0.0593, 0.0327]
WAN_DOCTOR_FIRST_TWO = [0.0649, 0.0494]            # the reference's own repeat: 0.0572 vs 0.0982 over 2 updates (42 %)
SFT = [0.9583, 1.0483, 1.0296, 1.0814, 1.0135, 0.9667, 0.9931, 0.9384, 1.0495, 1.1929, 0.9974, 0.8293, 0.944, 1.0066,
       1.1295, 0.9726, 1.0275, 0.9769, 1.1768, 1.0089, 1.0702, 1.0268, 1.1969, 1.1588, 0.8918, 1.0914, 0.9279, 1.0924,
       1.0271, 1.1286]


def series(values):
    return {s + 1: v for s, v in enumerate(values)}


def run(attempt, losses, *, k=None, t=1.0):
    """A 30-update run with window [11, 30] and these per-update losses, unpaired (no data digest); with k, the same run
    stopped as planned after k updates (the engine's probe or short run: the job's horizon, `stop_after` k)."""
    r = receipt(attempt, steps=30, loss=lambda s: losses[(s - 1) % len(losses)], t=t, data=False, sign=False,
                window=[11, 30], **({'completed': k, 'stop_after': k, 'planned_stop': True} if k else {}))
    if k:
        r['evidence']['per_step'] = r['evidence']['per_step'][:k]
    return signing.sign(r)


def check(v):
    return next(c for c in v.checks if c['id'] == 'loss_within_tolerance')


def test_a_full_run_without_a_last_loss_sample_keeps_the_manifest_tolerance():
    # DF8_r1 #1: both runs completed all 30 updates, the log has no loss at update 30: not a run stopped early
    def full(attempt, losses):
        r = receipt(attempt, steps=30, loss=lambda s: losses[(s - 1) % len(losses)], data=False, sign=False,
                    window=[11, 30])
        r['evidence']['per_step'][-1].pop('loss')
        return signing.sign(r)
    ref = full('ref', WAN)
    c = check(Verifier(0.03)(full('c1', [1.2 * x for x in WAN]), None, ref))
    assert c['status'] == 'fail' and c['evidence']['tolerance'] == pytest.approx(0.03)


def test_loss_noise_of_the_two_references_and_of_a_trend():
    assert 0.3 < loss_noise(series(WAN), 11, 30) < 0.45              # a diffusion loss: the sampled timestep
    assert 0.08 < loss_noise(series(SFT), 11, 30) < 0.12             # cross-entropy on 8K batches
    smooth = {s: 2.5 * math.exp(-0.02 * s) for s in range(1, 31)}    # the toy's decay: a trend is not noise
    assert loss_noise(smooth, 11, 30) < 0.003
    assert loss_noise({s: 2.0 - 0.01 * s for s in range(1, 31)}, 11, 30) == pytest.approx(0.0, abs=1e-12)
    assert loss_noise({1: 1.0, 2: 1.1}, 1, 2) is None and loss_noise({s: 0.0 for s in range(1, 6)}, 1, 5) is None
    assert loss_noise(series(WAN), 11, 30) == pytest.approx(loss_noise({**series(WAN), 5: 'nan'}, 11, 30))   # outside the window


def test_probe_of_a_swinging_loss_is_held_to_its_noise_not_the_manifest_tolerance():
    ref, vf = run('ref', WAN), Verifier(0.03)
    noise = loss_noise(series(WAN), 11, 30)
    # the D2-R204c probes (2-update means 0.0463, 0.0762, 0.0667, 0.0586 against the reference's 0.0982) and the doctor's
    # own first two updates: all judged infeasible at 3 %, all within 3 x the per-update noise of this loss
    for name, pair in (('alloc', [0.0463, 0.0463]), ('nvls', [0.0762, 0.0762]), ('workers2', [0.0667, 0.0667]),
                       ('workers8', [0.0586, 0.0586]), ('doctor', WAN_DOCTOR_FIRST_TWO)):
        v = vf(run(name, pair, k=2), ref, ref)
        r = check(v)
        assert v.valid and r['status'] == 'pass', (name, r['detail'])
        assert r['evidence']['tolerance'] == pytest.approx(LOSS_NOISE_Z * noise)       # k = 2: sqrt(2 / 2) = 1
        assert r['evidence']['deviation'] > 0.03 and 'is for the window mean' in r['detail']
    assert 1.0 < LOSS_NOISE_Z * noise < 1.4
    v = vf(run('broken', [3 * x for x in WAN[:2]], k=2), ref, ref)                      # a loss three times the reference's
    assert not v.valid and status(v, 'loss_within_tolerance') == 'fail'


def test_a_short_run_tightens_with_its_updates_and_a_full_run_keeps_the_manifest_tolerance():
    ref, vf = run('ref', WAN), Verifier(0.03)
    noise = loss_noise(series(WAN), 11, 30)
    short = check(vf(run('short', [1.2 * x for x in WAN], k=10), ref, ref))             # 20 % over 10 updates
    assert short['status'] == 'pass'
    assert short['evidence']['tolerance'] == pytest.approx(LOSS_NOISE_Z * noise * math.sqrt(2 / 10))
    assert check(vf(run('short2', [1.6 * x for x in WAN], k=10), ref, ref))['status'] == 'fail'    # 60 %: beyond it
    full = check(vf(run('full', [1.05 * x for x in WAN]), ref, ref))                      # the whole window: 3 %
    assert full['status'] == 'fail' and full['evidence']['tolerance'] == 0.03 and 'window mean;' not in full['detail']
    assert full['detail'].endswith('tolerance 3.00%')
    assert vf(run('full_ok', [1.01 * x for x in WAN]), ref, ref).valid


def test_a_smooth_loss_still_fails_a_break_at_the_probe_and_a_noiseless_one_widens_nothing():
    ref, vf = run('ref', SFT), Verifier(0.03)
    broken = vf(run('bf16+compile', [3.07258, 3.07258], k=2), ref, ref)                 # F1-S1 c9: 3.07 vs 1.00
    assert not broken.valid and status(broken, 'loss_within_tolerance') == 'fail'
    assert 0.25 < check(broken)['evidence']['tolerance'] < 0.35                         # 3 x 0.10: a 2-update mean
    assert vf(run('bf16', [1.005 * x for x in SFT], k=2), ref, ref).valid              # bf16 alone: within 0.5 %
    # the test receipt's linear loss, the toy's decay: no per-update noise, so the manifest 3 % holds at the probe too
    smooth = receipt('ref2', steps=30, data=False, window=[11, 30])
    trap = receipt('trap', steps=30, data=False, window=[11, 30], loss=lambda s: (2.0 - 0.01 * s) * 1.06, completed=2,
                   stop_after=2, planned_stop=True, sign=False)
    trap['evidence']['per_step'] = trap['evidence']['per_step'][:2]
    signing.sign(trap)
    r = check(vf(trap, smooth, smooth))
    assert r['status'] == 'fail' and r['evidence']['tolerance'] == 0.03 and 'window mean;' not in r['detail']
    assert Verifier(0)(run('exact', SFT, k=2), ref, ref).valid                          # tolerance 0 widens too
