"""The loss check of a full run on a job whose draws differ between runs (FABLE_DF16_DECISION A, F-D2-R204d-2): the
manifest tolerance is for the window mean of runs that see the same draws; the reference's own repeat (`infrarsi
doctor`'s run of the same configuration, `Verifier.repeat`) says whether this job's runs do. D2-R204d (Wan 2.1 through
DiffSynth, 8 x H100): the doctor's run and the search's reference differ 39.7 % per update (median) and 23.4 % on the
20-update window mean - the reference's own repeat failed the 3 % check and the report said "Repeat floor: unknown" -
while the per-update losses of every seeded job (sft_loop, Megatron, DeepSpeed, HF Trainer, diffusers SD1.5 and FLUX,
two diffusion losses among them) repeat to 0.2 %. With the repeat off per update, a full run's window mean is held to
3 x the noise of such a mean (cv 0.315: 29.9 %), as DF8 holds a probe's; with a repeat that agrees, or without one,
the manifest tolerance stays (today's verdicts; sft_loop's bf16 + compile 77 % and a 5 % break still fail)."""
import math

import pytest

from infrarsi.harness import signing
from infrarsi.verify import Verifier
from infrarsi.verify.verifier import LOSS_NOISE_Z, loss_noise, loss_repeat_note

from test_verifier import receipt, status

# D2-R204d (wan-t2v-20261024-135157, window [11, 30]): the search's reference, the doctor's run of the same configuration
# (/data/infrarsi_tb/trials/D2-R204d-wan-q30-df8/repo/.infrarsi/doctor/doctor/c0-doctor-a6350b21) and the full runs
WAN_REF = [0.051042, 0.058135, 0.097253, 0.043414, 0.053247, 0.07221, 0.053994, 0.049297, 0.023225, 0.050689, 0.046243,
           0.095212, 0.046152, 0.046837, 0.055542, 0.057375, 0.042945, 0.070252, 0.092035, 0.039805, 0.043765, 0.063467,
           0.041345, 0.043649, 0.056273, 0.045706, 0.042114, 0.048255, 0.045588, 0.049816]
WAN_DOCTOR = [0.067367, 0.079709, 0.043271, 0.048703, 0.014598, 0.079734, 0.088207, 0.097857, 0.070384, 0.056191, 0.081727,
              0.08373, 0.097135, 0.039156, 0.059394, 0.055911, 0.059751, 0.086993, 0.042512, 0.099488, 0.07631, 0.050031,
              0.092521, 0.048189, 0.078983, 0.044226, 0.011901, 0.045742, 0.081492, 0.088097]
C1_PATCH = [0.013796, 0.069573, 0.052113, 0.032385, 0.099426, 0.086781, 0.063684, 0.047711, 0.090171, 0.086128, 0.098411,
            0.060512, 0.095283, 0.053602, 0.045392, 0.056297, 0.089382, 0.051713, 0.087582, 0.041011, 0.047849, 0.045777,
            0.079826, 0.060384, 0.09932, 0.098948, 0.051997, 0.046736, 0.046026, 0.097758]
C2_WORKERS4 = [0.099445, 0.048352, 0.095651, 0.096889, 0.093561, 0.060167, 0.046884, 0.053671, 0.094925, 0.100954, 0.057036,
               0.098251, 0.080882, 0.011905, 0.078869, 0.035287, 0.05157, 0.045892, 0.049092, 0.079431, 0.060486, 0.078421,
               0.094358, 0.098538, 0.097734, 0.069462, 0.055002, 0.078534, 0.073842, 0.041639]
C4_ALLOC = [0.052518, 0.095662, 0.050591, 0.039845, 0.050203, 0.098532, 0.046201, 0.097277, 0.089257, 0.104836, 0.075947,
            0.03278, 0.048029, 0.051898, 0.050146, 0.012579, 0.087104, 0.095662, 0.044636, 0.045165, 0.09516, 0.060148,
            0.023597, 0.09438, 0.091798, 0.04256, 0.097637, 0.082889, 0.022729, 0.04599]
C6_WORKERS2 = [0.041509, 0.048124, 0.085219, 0.052752, 0.026128, 0.091853, 0.099155, 0.092845, 0.048333, 0.042518, 0.043309,
               0.043679, 0.047444, 0.062308, 0.057006, 0.072507, 0.0, 0.0567, 0.096723, 0.043561, 0.022778, 0.097342,
               0.072313, 0.053486, 0.064352, 0.043314, 0.056672, 0.043815, 0.0, 0.031252]
# D2-R204c (wan-t2v-20261024-120523): the pair whose window means agreed to 1.08 % (DF8) while 52 % apart per update
R204C_REF = [0.101015, 0.095462, 0.101636, 0.095712, 0.094532, 0.045789, 0.04917, 0.094257, 0.050213, 0.045677, 0.041712,
             0.047836, 0.046777, 0.084533, 0.046843, 0.042454, 0.090371, 0.060126, 0.049295, 0.076825, 0.103312, 0.054034,
             0.042735, 0.049015, 0.039499, 0.090708, 0.045246, 0.046242, 0.059325, 0.032671]
R204C_DOCTOR = [0.064915, 0.049434, 0.09433, 0.023028, 0.053642, 0.045921, 0.048905, 0.051081, 0.046683, 0.061394, 0.095329,
                0.023034, 0.08403, 0.040849, 0.097212, 0.099072, 0.039033, 0.041114, 0.0528, 0.059426, 0.0, 0.051503,
                0.095129, 0.045561, 0.01425, 0.04701, 0.048182, 0.06573, 0.076534, 0.086214]
# the F1-S1 reference (sft_loop, Qwen2.5-0.5B): cross-entropy on 8K batches, a seeded job (its doctor repeat: 0.0 %)
SFT = [0.9583, 1.0483, 1.0296, 1.0814, 1.0135, 0.9667, 0.9931, 0.9384, 1.0495, 1.1929, 0.9974, 0.8293, 0.944, 1.0066,
       1.1295, 0.9726, 1.0275, 0.9769, 1.1768, 1.0089, 1.0702, 1.0268, 1.1969, 1.1588, 0.8918, 1.0914, 0.9279, 1.0924,
       1.0271, 1.1286]


def series(values):
    return {s + 1: v for s, v in enumerate(values)}


def run(attempt, losses, *, k=None, t=1.0):
    """A 30-update run with window [11, 30] and these per-update losses, unpaired (no data digest); with k, the same run
    stopped as planned after k updates (a probe)."""
    r = receipt(attempt, steps=30, loss=lambda s: losses[(s - 1) % len(losses)], t=t, data=False, sign=False,
                window=[11, 30], **({'completed': k, 'stop_after': k, 'planned_stop': True} if k else {}))
    if k:
        r['evidence']['per_step'] = r['evidence']['per_step'][:k]
    return signing.sign(r)


def check(v):
    return next(c for c in v.checks if c['id'] == 'loss_within_tolerance')


def test_a_doctor_run_of_another_job_is_no_repeat():
    # DF16_r1 #1: the doctor run under the same knobs at half the tokens per update fails same_workload - it must not
    # widen a seeded job's tolerance; Wan's repeat, which only the loss sets apart, is one; a foreign receipt is not
    from infrarsi.project.flow import repeat_admissible
    strict = Verifier(0.03)
    ref, half = run('ref', SFT), receipt('doc', steps=30, loss=lambda s: 1.1 * SFT[(s - 1) % 30], tokens=2048,
                                         data=False, window=[11, 30])
    v = strict(half, None, ref)
    assert not v.valid and not repeat_admissible(v)
    assert repeat_admissible(strict(run('doctor', WAN_DOCTOR), None, run('ref', WAN_REF)))
    assert repeat_admissible(strict(run('doctor', SFT), None, ref))                       # a seeded repeat: valid
    assert not repeat_admissible(strict(dict(run('doctor', WAN_DOCTOR), schema='x'), None, run('ref', WAN_REF)))
    # DF17 / DF18 (D2-R204e): the doctor run completed all 30 updates and rank 7's pending all-reduce raised an NCCL error
    # at its end: the same job (no_errors reports such lines, DF18); a traceback or a NaN loss is not
    wan_ref = run('ref', WAN_REF)
    nccl = receipt('doc2', steps=30, loss=lambda s: WAN_DOCTOR[s - 1], t=1.0, data=False, window=[11, 30],
                   events={'collective': 3})
    v = strict(nccl, None, wan_ref)
    assert {r.split(':', 1)[0] for r in v.reasons} == {'loss_within_tolerance'} and repeat_admissible(v)
    tb = receipt('doc3', steps=30, loss=lambda s: WAN_DOCTOR[s - 1], t=1.0, data=False, window=[11, 30],
                 events={'collective': 3, 'traceback': 1})
    assert not repeat_admissible(strict(tb, None, wan_ref))
    nan = receipt('doc4', steps=30, loss=lambda s: 'nan' if s == 20 else WAN_DOCTOR[s - 1], t=1.0, data=False,
                  window=[11, 30], events={'collective': 3})
    assert not repeat_admissible(strict(nan, None, wan_ref))


def test_an_admitted_repeat_supplies_the_speed_pair_and_the_floor(monkeypatch):
    # DF17_r1: the doctor run admitted with collective lines at its exit, judged with the repeat installed, gives the
    # speed-noise pair and the repeat floor; a doctor run whose loss still fails gives neither
    from infrarsi.project import flow
    wan_ref = run('ref', WAN_REF)
    nccl = receipt('doc2', steps=30, loss=lambda s: WAN_DOCTOR[s - 1], t=1.03, data=False, window=[11, 30],
                   events={'collective': 3})
    vf = Verifier(0.03, repeat=nccl['evidence'])
    v = vf(nccl, None, wan_ref)
    assert v.valid and flow.repeat_valid(v)
    floor = flow.doctor_floor(vf, nccl, wan_ref)
    assert floor is not None and 0.2 < floor < 0.3                       # the doctor's 23.4 % window-mean deviation
    monkeypatch.setattr(flow, 'doctor_run', lambda project, ref: nccl)
    pair = flow.doctor_pair(None, vf, wan_ref, 1.0)
    assert pair is not None and abs(pair - math.log(1.03)) < 1e-9
    broken = receipt('doc5', steps=30, loss=lambda s: 3 * WAN_DOCTOR[s - 1], t=1.0, data=False, window=[11, 30],
                     events={'collective': 3})                         # the loss 3x the reference's: no repeat
    assert flow.doctor_floor(vf, broken, wan_ref) is None and not flow.repeat_valid(vf(broken, None, wan_ref))


def test_the_reference_repeat_tells_jobs_whose_draws_differ_from_seeded_ones():
    ref = run('ref', WAN_REF)['evidence']
    vf = Verifier(0.03, repeat=run('doctor', WAN_DOCTOR)['evidence'])
    assert 0.35 < vf.repeat_disagreement(ref) < 0.45                  # D2-R204d: 39.7 % per update, 95 % of them beyond 3 %
    c = Verifier(0.03, repeat=run('doc_c', R204C_DOCTOR)['evidence'])
    assert c.repeat_disagreement(run('ref_c', R204C_REF)['evidence']) > 0.5     # D2-R204c: 51.8 % (window means 1.08 % apart)
    seeded = Verifier(0.03, repeat=run('again', [1.0002 * x for x in SFT])['evidence'])
    assert seeded.repeat_disagreement(run('ref_s', SFT)['evidence']) == pytest.approx(0.0002, rel=0.01)
    assert Verifier(0.03).repeat_disagreement(ref) is None                       # no repeat known
    assert Verifier(0.03, repeat=run('probe', WAN_DOCTOR, k=2)['evidence']).repeat_disagreement(ref) is None   # none in the window


def test_wan_full_runs_are_held_to_the_noise_of_their_mean_when_the_draws_differ():
    ref, doc = run('ref', WAN_REF), run('doctor', WAN_DOCTOR)
    vf = Verifier(0.03, repeat=doc['evidence'])
    noise = loss_noise(series(WAN_REF), 11, 30)
    tol = LOSS_NOISE_Z * noise * math.sqrt(2 / 20)
    assert 0.31 < noise < 0.32 and 0.29 < tol < 0.31                   # cv 0.315: a 20-update mean is held to 29.9 %
    for name, losses, dev in (('c1 patch', C1_PATCH, 0.2624), ('c2 workers4', C2_WORKERS4, 0.2460),
                              ('c4 alloc', C4_ALLOC, 0.1198), ('c6 workers2', C6_WORKERS2, 0.0595)):
        v = vf(run(name, losses), ref, ref)
        r = check(v)
        assert v.valid and r['status'] == 'pass', (name, r['detail'])
        assert r['evidence']['deviation'] == pytest.approx(dev, abs=5e-4) and r['evidence']['tolerance'] == pytest.approx(tol)
        assert 'the draws differ between runs' in r['detail'] and 'own repeat is 39.' in r['detail']
        assert Verifier(0.03)(run(name + '-today', losses), ref, ref).valid is False     # without the repeat: 3 %, as today
    v = vf(doc, None, ref)                                             # the reference's own repeat: 23.4 %, a valid repeat
    assert v.valid and check(v)['evidence']['deviation'] == pytest.approx(0.234, abs=1e-3)
    assert not Verifier(0.03)(doc, None, ref).valid                   # (rejected at 3 %: D2-R204d's "Repeat floor: unknown")
    broken = vf(run('broken', [1.5 * x for x in WAN_REF]), ref, ref)   # a loss 1.5x the reference's: 50 %
    assert not broken.valid and status(broken, 'loss_within_tolerance') == 'fail'


def test_a_seeded_job_keeps_the_manifest_tolerance_however_noisy_its_loss():
    # D2-05 (diffusers SD1.5): a diffusion loss, cv 0.38, whose doctor repeat agrees per update to 0.0 %: the manifest holds
    ref = run('ref', WAN_REF)
    vf = Verifier(0.03, repeat=run('same', WAN_REF)['evidence'])
    assert vf.repeat_disagreement(ref['evidence']) == 0.0
    full = check(vf(run('c', [1.05 * x for x in WAN_REF]), ref, ref))
    assert full['status'] == 'fail' and full['evidence']['tolerance'] == 0.03 and 'window mean;' not in full['detail']
    assert vf(run('ok', [1.01 * x for x in WAN_REF]), ref, ref).valid
    # sft_loop (F1-S1), its repeat within 0.02 % per update: bf16 + compile's 77 % over 20 updates fails, so does a 5 % break
    sref = run('sref', SFT)
    sv = Verifier(0.03, repeat=run('sdoc', [1.0002 * x for x in SFT])['evidence'])
    assert not sv(run('bf16+compile', [1.77 * x for x in SFT]), sref, sref).valid
    assert not sv(run('break5', [1.05 * x for x in SFT]), sref, sref).valid
    assert sv(run('bf16', [1.005 * x for x in SFT]), sref, sref).valid
    # widened by the noise alone (no repeat asked), sft_loop would pass a 5 % break at 9.3 %; the repeat is what forbids it
    assert 0.09 < LOSS_NOISE_Z * loss_noise(series(SFT), 11, 30) * math.sqrt(2 / 20) < 0.10


def test_the_magnitude_comes_from_the_reference_noise_not_from_the_repeat():
    # a smooth loss (the toy's decay) whose repeat is 10 % off at every update: the draws differ, but a mean of a noiseless
    # loss has no noise to widen by - the toy's 6 % trap (fused_ce + compile) still fails at 3 %
    smooth = receipt('ref', steps=30, data=False, window=[11, 30], loss=lambda s: 2.5 * math.exp(-0.02 * s))
    off = receipt('rep', steps=30, data=False, window=[11, 30], loss=lambda s: 2.75 * math.exp(-0.02 * s))
    vf = Verifier(0.03, repeat=off['evidence'])
    assert vf.repeat_disagreement(smooth['evidence']) == pytest.approx(0.1)
    trap = receipt('trap', steps=30, data=False, window=[11, 30], loss=lambda s: 2.5 * math.exp(-0.02 * s) * 1.06)
    r = check(vf(trap, smooth, smooth))
    assert r['status'] == 'fail' and r['evidence']['tolerance'] == 0.03


def test_probes_are_unchanged_by_the_repeat():
    ref = run('ref', WAN_REF)
    noise = loss_noise(series(WAN_REF), 11, 30)
    for vf in (Verifier(0.03), Verifier(0.03, repeat=run('doctor', WAN_DOCTOR)['evidence']),
               Verifier(0.03, repeat=run('same', WAN_REF)['evidence'])):
        r = check(vf(run('probe', WAN_DOCTOR[:2], k=2), ref, ref))    # DF8: a 2-update mean at 3 x the per-update noise
        assert r['status'] == 'pass' and r['evidence']['tolerance'] == pytest.approx(LOSS_NOISE_Z * noise)
        assert 'draws differ' not in r['detail']


def test_loss_repeat_view_and_the_lines_the_user_reads():
    ref = run('ref', WAN_REF)['evidence']
    vf = Verifier(0.03)
    rep = vf.loss_repeat(ref)
    assert rep['updates'] == 20 and rep['repeat'] is None and rep['draws_vary'] is None and not rep['paired']
    assert rep['tolerance'] == pytest.approx(LOSS_NOISE_Z * rep['noise'] * math.sqrt(2 / 20)) and 0.29 < rep['tolerance'] < 0.31
    unknown = loss_repeat_note(rep, 0.03)                              # `run` without a doctor run to read: what to do
    assert 'if runs draw differently' in unknown and '`infrarsi doctor` within a day' in unknown
    assert 'above 29.9%' in unknown and 'held to the manifest 3.0%' in unknown
    doc = loss_repeat_note(rep, 0.03, doctor=True)                      # the doctor's own line, before any search
    assert 'compares its reference with this run per update' in doc and 'more updates per run' in doc
    vf.repeat = run('doctor', WAN_DOCTOR)['evidence']
    rep = vf.loss_repeat(ref)
    assert rep['draws_vary'] is True and 0.35 < rep['repeat'] < 0.45
    vary = loss_repeat_note(rep, 0.03)
    assert vary.startswith("this job's draws differ between runs") and 'held to 29.9%' in vary and 'not the manifest 3.0%' in vary
    vf.repeat = run('same', WAN_REF)['evidence']
    agree = loss_repeat_note(vf.loss_repeat(ref), 0.03)
    assert 'runs repeat their draws' in agree and 'the manifest 3.0% holds' in agree
    assert loss_repeat_note(Verifier(0.03).loss_repeat(run('sft', SFT)['evidence']), 0.03) is not None   # cv 0.10: 9.3 %
    smooth = receipt('ref2', steps=30, data=False, window=[11, 30])    # the toy's decay: a quiet loss, nothing to say
    assert loss_repeat_note(Verifier(0.03).loss_repeat(smooth['evidence']), 0.03) is None
    paired = receipt('ref3', steps=30, window=[11, 30], loss=lambda s: WAN_REF[s - 1])   # a data digest: compared per update
    assert Verifier(0.03).loss_repeat(paired['evidence'])['paired']
    assert loss_repeat_note(Verifier(0.03).loss_repeat(paired['evidence']), 0.03) is None
    assert loss_repeat_note(None, 0.03) is None
