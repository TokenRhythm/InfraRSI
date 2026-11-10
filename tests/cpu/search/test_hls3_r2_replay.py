"""FABLE_HLS3_FIXES_R2_Q9H_M35H2 on the recorded arms (data/hls3_arms.json; its job-log notes per candidate and launch,
build_hls3_arms.with_notes), offline. Tests read only the fixture.

T-replay-inert (F10): the eight conv patches one model family wrote against one library (FABLE section 3.2), each read at
its probe as hls3 reads it - the notes of its base's run, the notes of its probe (an older arm ran no probe: its full
run's notes stand in, the note being logged once per process at the function's first call, which the -h3 probes show),
the note it names (PROOF_RULE's `evidence_note`: each cites the fallback note in its site or rationale; the field did not
exist), the order of its lines (the note logged at the first call, during the first update, after the patch installed).
Each gets the two lines a patch written to PROOF_RULE prints when it took effect, so only the note decides: the five
whose note is still reported after they installed are inert at start, the three whose note is gone are not. As recorded
(no `evidence_note`) the note is feedback only and the lines decide.
M35H2-h3 c1 / c13 / c14 under the lines such a patch would have printed by FABLE's reading of the library (c1 set the
value the job already had; c13 / c14 installed at a site the incumbent's Liger had replaced, their code never ran): inert
at start by the lines alone. F10c: the post-hoc floor on robust sigma_pair (what the search used).
T-replay-q9h-rank-2 (F11): Q9H-h3 replayed on robust sigma_pair with the round-2 components: micro-batch 2 (p 0.15, a
declared 40 GiB spend; the transfer judgment 0.55) is ranked at max(0.15, 0.55) - pi joins only once a feared move ran,
and none did in the record - with the ledger keeping 0.15.
T-replay-gate (F12): the six -h3 arms' passed probes ran two updates - nothing past the warm-up, so no speed veto. What
a gated probe (three updates) would have read: the full run's interval after its second update against the base's same
interval; a change of unknown memory direction counts as a possible spend."""
import copy
import math
import statistics
from types import SimpleNamespace

from hls3_helpers import FIX, PATCH_FILE, open_loop, replay, timing
from infrarsi.project.signals import proof_evidence
from infrarsi.search.engine import Outcome, Proposal
from infrarsi.search.hls3 import GATE_LOW

H3 = sorted(n for n in FIX if n.endswith('-h3'))
CONV = {('M35H2-code-env-opus', 'c5'): False, ('Q9U-code-env-opus-h2', 'c2'): False,     # effective (FABLE 3.2)
        ('Q9U-code-env-opus-h3', 'c1'): False,
        ('M35H2-code-env-opus-h2', 'c4'): True, ('Q9H-code-env-opus-h3', 'c2'): True,        # inert
        ('M35H2-code-env-opus-h3', 'c3'): True, ('Q27H2-code-env-opus', 'c8'): True, ('Q27U-code-env-opus-h3', 'c8'): True}
PROVEN = ['[infrarsi-patch] causal_conv1d_fn: found reference fallback -> set fused kernel',
          '[infrarsi-patch] causal_conv1d_fn: active']
SYNTH = {'c1': ['[infrarsi-patch] from_pretrained: found experts_implementation=grouped_mm -> set '
                'experts_implementation=grouped_mm', '[infrarsi-patch] from_pretrained: active'],
         'c13': ['[infrarsi-patch] experts.forward: found grouped_mm_experts_forward -> set sorted grouped_mm'],
         'c14': ['[infrarsi-patch] experts.forward: found grouped_mm_experts_forward -> set one-sync eager loop']}
MBS2 = {'gradient_accumulation_steps': 1, 'per_device_train_batch_size': 2}


def robust(name: str) -> str:
    """The arm with its robust sigma_pair (what the search used; the plain one is the promotion ledger's)."""
    key = name + '#robust'
    if key not in FIX:
        arm = copy.deepcopy(FIX[name])
        arm['noise']['sigma_pair'] = arm['noise']['robust']['sigma_pair']
        FIX[key] = arm
    return key


def site_of(arm: dict, c: dict) -> str:
    """The site the recorded hypothesis named (the -h3 arms' schema field); a review patch's is in its rationale."""
    return next((h.get('site') or '' for call in arm['calls'] if call['role'] == 'hypotheses'
                 for h in call['hypotheses'] if h.get('patch') and h['mechanism'] == c.get('rationale')), '')


def at_probe(name: str, cid: str, lines: list, *, off=(), named: bool = True) -> tuple:
    """(the gate's verdict, the class hls3 gives it, the probe's notes) for a recorded patch at its probe; `named`: the
    patch names its base's notes (`evidence_note`), else no note is named."""
    arm = FIX[name]
    cands = {c['id']: c for c in arm['candidates']}
    c = cands[cid]
    base = c.get('parent') or 'c0'
    eng, st, _ = open_loop(name, r1=True, r2=True, off=off)
    for x in dict.fromkeys(('c0', base)):                 # the base's run: its receipt's notes
        if x not in eng.s.candidates:
            eng.s.candidates[x] = SimpleNamespace(id=x, parent='c0', config=dict(cands[x]['config']), overlay={},
                                                  verdict={'valid': True, 'step_time': 1.0}, status='screened', results={},
                                                  receipt=None)
        rec = eng.s.candidates[x].receipt or {}
        eng.s.candidates[x].receipt = {**rec, 'evidence': {**(rec.get('evidence') or {}),
                                                           'events': {'perf_notes': list(cands[x].get('perf_notes') or [])}}}
    probe = next((a for a in arm.get('attempts') or [] if a['cid'] == cid and a['purpose'] == 'probe'), None)
    notes = list((probe if probe is not None else c).get('perf_notes') or [])
    code = c.get('patch_lines') or c.get('patch_code') or []
    p = Proposal(change={}, base=base, patch={PATCH_FILE: '\n'.join(code) + '\n'}, mechanism=c.get('rationale') or cid)
    eng.s.candidates[cid] = SimpleNamespace(id=cid, parent=base, config=dict(c['config']), overlay=dict(p.patch),
                                            verdict=None, status='proposed', results={}, receipt=None)
    proof = proof_evidence(list(lines))
    installed = max([x[2] for r in proof.values() for x in r['installs']], default=0.0)
    row = {'perf_notes': notes, 'patch_lines': list(lines), 'patch_proof': proof, 'steps': 3,
           'perf_note_last': {n: installed + 1.0 for n in notes}}          # (logged at the first call, after it installed)
    h = {'site': site_of(arm, c), 'evidence_note': (cands[base].get('perf_notes') or [None])[0] if named else None}
    why = st._gate(eng, cid, base, row, claim=st._claim(eng, base, p, h), h=h)
    kind = None
    if why:
        kind = st.classify(eng, p, Outcome(cid, 'infeasible', False, None, None, 'probe', 0.1, [why[0]], [why[1]]))['kind']
    return why, kind, notes


def screen_cost(name: str, cid: str) -> tuple[float, float]:
    """(wall seconds, GPU-hours) of the candidate's recorded full screen."""
    arm = FIX[name]
    x = timing(arm).get((cid, 'screen')) or {}
    gpu = sum(r.get('gpu_hours') or 0.0 for r in arm['screens'] if r['candidate_id'] == cid and r['fidelity'] == 'full')
    return float(x.get('wall_s') or 0.0), gpu


# ---------------------------------------------------------------- F10
def test_r2_f10_inert_patches_caught_at_probe():
    print('\nF10: the eight conv patches at their probe (lines of a patch that took effect; the note decides):')
    saved = []
    for (name, cid), inert in CONV.items():
        why, kind, notes = at_probe(name, cid, PROVEN)
        print(f'  {name:24s} {cid:4s} note at probe: {"yes" if notes else "no ":3s} -> '
              f'{kind or "proven, full run"}{": " + why[0][:70] if why else ""}')
        assert (kind == 'inert_at_start') is inert, (name, cid, why)
        if inert:
            assert 'still reported after it installed' in why[0] and 'causal_conv1d_fn' in why[0]
            saved.append((name, cid, *screen_cost(name, cid)))
        why_off, _, _ = at_probe(name, cid, PROVEN, off=['hls3_self_proof'])
        assert why_off is None                                      # switched off: the probe passes, as recorded
        assert at_probe(name, cid, PROVEN, named=False)[0] is None   # no note named: feedback only, the lines decide
    print('F10: M35H2-h3 c1 / c13 / c14 under the lines their patches would have printed:')
    for cid, lines in SYNTH.items():
        why, kind, notes = at_probe('M35H2-code-env-opus-h3', cid, lines, named=False)   # (they name no note)
        print(f'  M35H2-code-env-opus-h3   {cid:4s} -> {kind}: {why[0][:90]}')
        assert kind == 'inert_at_start' and 'still reported' not in why[0]   # (the conv note is not theirs)
        assert ('already there' if cid == 'c1' else 'never ran') in why[0]
        saved.append(('M35H2-code-env-opus-h3', cid, *screen_cost('M35H2-code-env-opus-h3', cid)))
    w, g = sum(x[2] for x in saved), sum(x[3] for x in saved)
    print(f'F10: full screens not run: {len(saved)} ({", ".join(f"{n[:9]} {c}" for n, c, _, _ in saved)}); '
          f'wall {round(w)} s on the -h3 arms (older arms: no launch timing), {g:.2f} GPU-h')
    assert len(saved) >= 5 and len(saved) == 8


def test_r2_f10c_the_inert_floor_on_robust_sigma():
    """F10c: |measured| < max(sigma_pair, kappa x mu / 4) on the -h3 arms' inert patches (sigma_pair robust, kappa the
    fresh prior: no pair resolved), against F8b's |measured| < sigma_pair."""
    print('\nF10c: post-hoc inert on robust sigma_pair (kappa0):')
    cases = {('M35H2-code-env-opus-h3', 'c1'): 12.0, ('M35H2-code-env-opus-h3', 'c3'): 6.0,
             ('M35H2-code-env-opus-h3', 'c13'): 6.0, ('M35H2-code-env-opus-h3', 'c14'): 5.0,
             ('Q9H-code-env-opus-h3', 'c2'): 6.0, ('Q27U-code-env-opus-h3', 'c8'): 6.0}
    got = {}
    for (name, cid), pred in cases.items():
        arm = FIX[robust(name)]
        cands = {c['id']: c for c in arm['candidates']}
        base = cands[cid].get('parent') or 'c0'
        st_t = {r['candidate_id']: r['step_time'] for r in arm['screens'] if r['fidelity'] == 'full' and r.get('valid')
                and r.get('step_time')}
        gain = math.log((st_t.get(base) or arm['reference_step_time']) / st_t[cid])
        flags = []
        for off in ((), ('hls3_self_proof',)):
            eng, st, _ = open_loop(robust(name), r1=True, r2=True, off=off)
            h = {'mechanism': 'p', 'patch': {PATCH_FILE: '# p\n'}, 'mu': math.log1p(pred / 100), 'site': None}
            eng.s.candidates[cid] = SimpleNamespace(id=cid, parent='c0', config={}, overlay={}, verdict=None,
                                                    status='screened', results={}, receipt=None)
            st._inert(eng, Proposal(change={}, base='c0', patch=h['patch']), Outcome(cid, 'screened', True, None, gain,
                                                                                      'full', 0.3, [], []), h)
            flags.append(bool(h.get('inert')))
        got[(name, cid)] = tuple(flags)
        sp = arm['noise']['sigma_pair']
        print(f'  {name:24s} {cid:4s} predicted +{pred:.0f} %, measured {100 * gain:+.2f} %, sigma_pair '
              f'{100 * sp:.3f} %: F10c {"inert" if flags[0] else "-"}, F8b alone {"inert" if flags[1] else "-"}')
    assert all(f[0] for f in got.values())                          # the floor calls every one inert
    assert {k for k, f in got.items() if f[1]} == {('Q9H-code-env-opus-h3', 'c2'), ('Q27U-code-env-opus-h3', 'c8')}


# ---------------------------------------------------------------- F11
def trajectory(off=()) -> list[dict]:
    """Per recorded decision of Q9H-h3 (robust sigma_pair): the incumbent, the top proposal, micro-batch 2's place, its
    ranked p and pi."""
    key = robust('Q9H-code-env-opus-h3')
    rows = [r for r in FIX[key]['screens'] if r['fidelity'] in ('full', 'probe')]
    out = []

    def on_step(i, eng, st, props):
        sc = sorted(((eng.value(p) / eng.expected_cost(p), p) for p in props or []), key=lambda x: -x[0])
        pos = next((j for j, (_, p) in enumerate(sc) if p.change == MBS2), None)
        out.append({'run': rows[i]['candidate_id'], 'fid': rows[i]['fidelity'], 'inc': eng._inc, 'pos': pos,
                    'p': sc[pos][1].p_feasible if pos is not None else None, 'plan': sc[pos][1].fidelity if pos is not None
                    else None, 'top': sc[0][1].mechanism, 'ratio': round(sc[pos][0], 4) if pos is not None else None,
                    'top_ratio': round(sc[0][0], 4), 'pi': (st.fear['tried'], st.fear['fitted'], round(st._pi(), 3)),
                    'entry_p': next((h['p_feasible'] for h in st.ledger.values() if h['base'] == eng._inc
                                     and h.get('change') == MBS2), None)})
    replay(key, ask=True, adopt=True, follow=True, on_step=on_step, r1=True, r2=True, off=off)
    return out


def test_r2_f11_q9h_rank_feared_spender():
    on, off = trajectory(), trajectory(off=['hls3_fear_prior'])
    print('\nF11: Q9H-h3 replayed (robust sigma_pair): micro-batch 2 at each recorded decision (place 0 = first)')
    for a, b in zip(on, off):
        print(f'  before {a["run"]:4s} {a["fid"]:5s} on {a["inc"]:3s}: F11 place {a["pos"]} p {a["p"]} '
              f'(v/c {a["ratio"]}; pi {a["pi"]}) | off: place {b["pos"]} p {b["p"]} | top: {a["top"][:44]}')
    c4 = [a for a in on if a['inc'] == 'c4']
    assert c4 and all(a['entry_p'] == 0.15 for a in c4)               # D2: the ledger keeps the proposer's p
    assert c4[0]['pi'] == (0, 0, 0.5) and c4[0]['p'] == 0.55 and c4[0]['plan'] == ('probe', 'full')
    assert c4[0]['pos'] == 1 and c4[0]['top'].startswith('turn off activation recompute')   # (recorded c5 first)
    assert all(a['pos'] == 0 for a in c4[1:])                        # then first at every decision on c4
    assert all(b['pos'] > 0 for b in off if b['inc'] == 'c4')       # without F11: never first


# ---------------------------------------------------------------- F12
def test_r2_f12_gate_calibration():
    print(f'\nF12: passed probes; what a gated probe (three updates) would have read - the full run\'s interval after its '
          f'second update against the base\'s (gate x{GATE_LOW} on a spender or a change of unknown memory effect):')
    got, recorded = {}, {}
    for name in H3:
        tm = timing(FIX[name])

        def on_run(i, eng, st, p, cand, row, name=name, tm=tm):
            x = tm.get((cand.id, 'probe')) or {}
            if row['fidelity'] != 'probe' or not row['valid'] or not x.get('intervals') or (x.get('steps_completed') or 0) < 2:
                return
            base = p.base or 'c0'
            ideas = frozenset(st._patch_ideas(p))
            n0 = len(st.gated)
            recorded[(name, cand.id)] = st._gate(eng, cand.id, base, {'intervals': x['intervals']}, ideas=ideas)
            full = (tm.get((cand.id, 'screen')) or {}).get('intervals') or []
            ref = st._base_intervals(eng, base)
            if len(full) >= 2 and len(ref) >= 2 and ref[1] > 0:
                stop = st._gate(eng, cand.id, base, {'intervals': full[:2]}, ideas=ideas)
                g = st.gated[-1] if len(st.gated) > n0 else {}
                why = ('declared spend' if g['spends'] else 'unknown memory effect' if g.get('unknown') else 'noted only') \
                    if g else None
                got[(name, cand.id)] = (round(full[1] / ref[1], 2), bool(stop), why)
            del st.gated[n0:]
        replay(name, ask=True, adopt=True, follow=True, r1=True, r2=True, on_run=on_run)
    assert recorded and not any(recorded.values())                 # two updates: nothing past the warm-up, no veto
    thrash = {('G4U-code-env-opus-h3', 'c2'), ('Q9H-code-env-opus-h3', 'c5')}      # their full runs were aborted
    for k, (ratio, stop, why) in sorted(got.items()):
        print(f'  {k[0]:24s} {k[1]:4s} x{ratio:.2f} {"STOP" if stop else "go"}{f" ({why})" if why else ""}')
    assert max(r for k, (r, _, _) in got.items() if k not in thrash) < GATE_LOW      # no valid probe stopped
    assert all(got[k][0] >= GATE_LOW for k in thrash)                               # both thrashes above it
    assert {k for k, v in got.items() if v[1]} == thrash                            # both stopped, nothing else
    assert got[('G4U-code-env-opus-h3', 'c2')] == (4.02, True, 'declared spend')
    assert got[('Q9H-code-env-opus-h3', 'c5')] == (1.83, True, 'unknown memory effect')   # decision 2
