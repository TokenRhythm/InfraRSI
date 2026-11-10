"""FABLE_HLS3_FIXES_R3 on the recorded arms (data/hls3_r3.json: the -h5 arms, the patch probes' log lines, appendix A's
per-update timestamps, appendix B's failed model calls; data/build_hls3_r3.py), offline. Tests read only the fixtures.

T-r3-scanner (F13): Q9H-h5 c2's 417-character installation line - read whole it is the installation it was, the patch
proven at its probe (its note gone, `active` printed); the round-2 reading (cut at 300) is the recorded "no line of its
own". M35H2-h5 c1 still set what was already there; the eight conv patches of round 2 judge as they did.
T-r3-parse (F14): appendix B's five failed answers - joined, each hypothesis on its own, non-schema edits rewritten.
T-r3-calib (F23): appendix A - every thrash at max(2 %, 5x its base's scatter) or more, no slow or valid run.
T-r3-replay-q9h: F21 (c5 / c8 with a deferral line read as deferring).
T-r3-replay-g4u: F23 (c1 / c3 / c8 / c13 thrash, c11 not), F15 (the recompute seed a spender at the judgment's +20 %),
F16-A (its pair probes with the start composite's members, compile first), F19 (six suspect pairs and their clearing),
F16 at c13 (the swap without Liger and fused); as recorded, the c6 round lost - with F14 it is read, and S-2 takes the
spender first (its SDPA patch declared to free 8 GiB).
T-r3-f22: M35H2-h5 - after c5 / c6 / c8 the shared micro-batch change waits: c11 / c12 / c16 / c17 / c18 not proposed."""
import copy
import json
import math
import statistics
from types import SimpleNamespace

import pytest

from hls3_helpers import FIX, PATCH_FILE, R3_FIX, open_loop, replay, timing
from infrarsi.agent.backends import claude_cli
from infrarsi.project.signals import scan_events
from infrarsi.search import proposers as P
from infrarsi.search.engine import Outcome, Proposal
from infrarsi.search.hls3 import JITTER_MIN, JITTER_X, proof_verdict, proposer_flags, scatter
from test_hls3_r2_replay import CONV, PROVEN, at_probe

MODEL = 'claude-opus-5-5'
G4U, Q9H, M35 = 'G4U-code-env-opus-h5', 'Q9H-code-env-opus-h5', 'M35H2-code-env-opus-h5'
SCHEMA = P.hls3_schema(P.HYPOTHESIS_SCHEMA, proposer_flags())
RC_OFF, COMPILE = {'gradient_checkpointing': False}, {'torch_compile': True}


def robust(name: str, lost_c6: bool = False) -> str:
    """The arm with its robust sigma_pair (what the search used); `lost_c6`: without the model call on c6 (G4U-h5's lost
    round, as recorded)."""
    key = name + ('#robust-lost' if lost_c6 else '#robust')
    if key not in FIX:
        arm = copy.deepcopy(FIX[name])
        arm['noise']['sigma_pair'] = arm['noise']['robust']['sigma_pair']
        if lost_c6:
            arm['calls'] = [c for c in arm['calls'] if not (c['role'] == 'hypotheses' and c['base'] == 'c6')]
        FIX[key] = arm
    return key


def probe(arm: str, cid: str) -> dict:
    return next(p for p in R3_FIX['probes'] if p['arm'] == arm and p['cid'] == cid)


def row_of(lines, steps=2) -> dict:
    """A probe's detail row as the session builds it from its log (signals.scan_events)."""
    ev = scan_events([(t, x) for t, x in lines])
    return {**{k: v for k, v in ev.items() if k.startswith(('patch', 'perf'))}, 'steps': steps}


# ---------------------------------------------------------------- T-r3-scanner (F13)
def gate_q9h(cid: str, lines, off=()) -> tuple:
    """(the gate's verdict, hls3's proof_fail row) for a Q9H-h5 patch at its probe as hls3 reads it: its base's notes
    from the base run's log, the recorded hypothesis' site and evidence note, the patch's own code."""
    pr = probe(Q9H, cid)
    eng, st, arm = open_loop(robust(Q9H), r1=True, r2=True, r3=True, off=off)
    cands = {c['id']: c for c in arm['candidates']}
    base = pr['base']
    for x in dict.fromkeys(('c0', base)):
        if x not in eng.s.candidates:
            eng.s.candidates[x] = SimpleNamespace(id=x, parent='c0', config=dict(cands[x]['config']), overlay={},
                                                  verdict={'valid': True, 'step_time': 1.0}, status='screened', results={},
                                                  receipt=None)
    bev = scan_events([(t, x) for t, x in pr['base_lines']])
    eng.s.candidates[base].receipt = {'evidence': {'events': {k: v for k, v in bev.items() if k.startswith('perf')}}}
    c = cands[cid]
    h = next((dict(f) for call in arm['calls'] if call['role'] in ('hypotheses', 'code_review')
              for f in call.get('hypotheses') or call.get('findings') or []
              if (f.get('mechanism') or '') == c.get('rationale') or str(c.get('rationale') or '').startswith('review:')
              and f.get('evidence_note') and (f.get('issue') or '')[:60] in str(c.get('rationale') or '')), {})
    code = c.get('patch_lines') or []
    p = Proposal(change={}, base=base, patch={PATCH_FILE: '\n'.join(code) + '\n'}, mechanism=c.get('rationale') or cid)
    eng.s.candidates[cid] = SimpleNamespace(id=cid, parent=base, config=dict(c['config']), overlay=dict(p.patch),
                                            verdict=None, status='proposed', results={}, receipt=None)
    why = st._gate(eng, cid, base, row_of(lines, pr['steps'] or 2), claim=st._claim(eng, base, p, h), h=h)
    return why, st.proof_fail.get(cid), h


def test_r3_scanner_q9h_c2_is_proven_read_whole():
    pr = probe(Q9H, 'c2')
    long = [x for _, x in pr['lines'] if '-> set' in x]
    assert long and all(len(x) > 300 for x in long) and min(x.index('-> set') for x in long) > 300   # (399 + the stamp)
    assert pr['recorded']['patch_proof']['GatedDeltaNet.causal_conv1d_fn']['installs'] == []   # what the arm read
    ev = scan_events([(t, x) for t, x in pr['lines']])
    whole = ev['patch_proof_whole']['GatedDeltaNet.causal_conv1d_fn']
    assert len(whole['installs']) == 1 and whole['installs'][0][1].startswith('fla.modules.convolution.causal_conv1d')
    assert whole['installs'][0][0] != whole['installs'][0][1] and whole['active_at'] is not None
    assert proof_verdict(ev['patch_proof_whole']) is None and proof_verdict(ev['patch_proof']).startswith('it printed no')
    why, _, h = gate_q9h('c2', pr['lines'])
    assert h.get('evidence_note') and why is None                    # proven: its full run follows (F13)
    why_off, fail, _ = gate_q9h('c2', pr['lines'], off=['hls3_scan_whole'])
    assert why_off[1] == 'patch_inert' and why_off[0].startswith('it printed no')   # round 2: the recorded misjudgment
    print(f'\nF13: Q9H-h5 c2 read whole -> proven (round 2: "{why_off[0][:70]}")')


def test_r3_scanner_m35h2_c1_and_the_round_two_conv_patches():
    pr = probe(M35, 'c1')
    ev = scan_events([(t, x) for t, x in pr['lines']])
    assert 'already there' in proof_verdict(ev.get('patch_proof_whole') or ev['patch_proof'])
    for (name, cid), inert in CONV.items():                        # round 2's eight, with every round-3 component on
        why, kind, _ = at_probe(name, cid, PROVEN)
        assert (kind == 'inert_at_start') is inert, (name, cid, why)


# ---------------------------------------------------------------- T-r3-parse (F14)
def call(name: str, attempt: str = 'first') -> dict:
    return next(x for x in R3_FIX['llm'] if x['call'] == name and x['attempt'] == attempt)


ROUND = {'hypotheses_18e63e00e994': 'c0', 'hypotheses_f22819980c1c': 'c6', 'hypotheses_1b88026a1c09': 'c12',
         'hypotheses_2e462c3009a2': 'c9'}      # the incumbent each call was on (FABLE_HLS3_FIXES_R3 appendix B)


def test_r3_parse_the_failed_answers():
    files = {'G4U': R3_FIX['files'][G4U], 'Q9H': R3_FIX['files'][Q9H]}
    got = {}
    for c in R3_FIX['llm']:
        arm = 'G4U' if c['arm'].startswith('G4U') else 'Q9H'
        plain, f0 = claude_cli.parse_stream(c['stdout'], model=MODEL, exit_code=0, status='RETURNED')
        joined, f1 = claude_cli.parse_stream(c['stdout'], model=MODEL, exit_code=0, status='RETURNED', join_text=True)
        action = joined or plain
        kept, dropped, fixed = P.accept_hypotheses((action or {}).get('hypotheses') or [], SCHEMA, files[arm])
        got[(c['call'], c['attempt'])] = (plain is not None, f1['parse_mode'], len(kept), len(dropped), len(fixed))
        print(f'  {arm} on {ROUND[c["call"]]:3s} {c["call"]} {c["attempt"]:7s}: recorded "{str(c["error"])[:40]}"; parse {f1["parse_mode"]}; '
              f'{len(kept)} kept ({len(fixed)} edits rewritten), {len(dropped)} dropped')
    assert got[('hypotheses_2e462c3009a2', 'first')][:3] == (False, 'joined', 4)   # Q9H c9 (lost, 641 s): two text
    #                                                                              blocks joined
    assert got[('hypotheses_f22819980c1c', 'first')][1:3] == ('failed', 0)       # G4U call 2's first reply was cut:
    #                                                                              its retry (with the errors) is read
    assert got[('hypotheses_2e462c3009a2', 'retry_1')][1:3] == ('joined', 5)
    assert got[('hypotheses_f22819980c1c', 'retry_1')][2:5] == (5, 0, 4)          # G4U c6 (lost, 401 s): 4 patches +
    #                                                                              pin off
    assert got[('hypotheses_18e63e00e994', 'first')][2] >= 8                       # G4U call 1: the grouped_mm patch too
    assert got[('hypotheses_1b88026a1c09', 'first')][2:4] == (7, 0)               # G4U call 3: all seven
    assert all(c['error'] for c in R3_FIX['llm'])                                # (round 2 used none of them)
    kept, _, _ = P.accept_hypotheses([{'mechanism': 'm', 'change': {'tf32': True}, 'predicted_gain_pct': 1}], SCHEMA, {})
    assert kept == [{'mechanism': 'm', 'change': {'tf32': True}, 'predicted_gain_pct': 1}]   # a valid answer as it came


# ---------------------------------------------------------------- T-r3-calib (F23)
def ev_of(rec: dict) -> dict:
    return {'per_step': [{'step': s, 't': t} for s, t in rec['per_step']]}


BASES = {('G4U-code-env-opus-h5', 'c13'): 'c12', ('G4U-code-env-opus-h5', 'c8'): 'c6',
         ('G4U-code-env-opus-h5', 'c11'): 'c6'}


def test_r3_calib_the_scatter_tells_thrash_from_slow():
    recs = {(r['arm'], r['cid']): r for r in R3_FIX['jitter']}
    rows = []
    for (arm, cid), r in recs.items():
        if r['class'] == 'valid':
            continue
        base = recs.get((arm, BASES.get((arm, cid), 'c0')))
        x, b = scatter(ev_of(r)), scatter(ev_of(base)) if base else None
        thr = max(JITTER_MIN, JITTER_X * b) if b is not None else JITTER_MIN
        rows.append((arm, cid, r['class'], x, b, x >= thr))
        print(f'  {arm:22s} {cid:4s} {r["class"]:8s} scatter {100 * x:5.2f} % (base {100 * (b or 0):.2f} %): '
              f'{"thrash" if x >= thr else "-"}')
    assert all(t for _, _, cls, _, _, t in rows if cls == 'thrash') and len([r for r in rows if r[2] == 'thrash']) == 6
    assert not any(t for _, _, cls, _, _, t in rows if cls in ('slow', 'possible'))
    valid = [scatter(ev_of(r)) for r in recs.values() if r['class'] == 'valid']
    assert max(valid) < JITTER_MIN / 4                              # every valid run far below the threshold


# ---------------------------------------------------------------- T-r3-replay-q9h (F21)
def test_r3_replay_q9h_a_deferral_line_names_why():
    """Q9H-h5 c5 / c8: their wrappers handed every call back to the reference conv (the decoder passes **kwargs); with
    the line a wrapper written to F21's rule prints, the reason says so (round 2: "not the code the job runs")."""
    for cid in ('c5', 'c8'):
        pr = probe(Q9H, cid)
        site = next(x.split(']', 1)[1].split(': found', 1)[0].strip() for _, x in pr['lines'] if ': found' in x)
        defer = f'[infrarsi-patch] {site}: deferred (kwargs non-empty: the decoder passes through its **kwargs)'
        lines = pr['lines'] + [[pr['lines'][-1][0] - 1.0, defer]]   # (printed before its last line)
        why, _, _ = gate_q9h(cid, sorted(lines))
        assert why[1] == 'patch_inert' and 'defers' in why[0] and 'kwargs non-empty' in why[0], why
        why_off, _, _ = gate_q9h(cid, pr['lines'], off=['hls3_deferred'])
        assert 'not the code the job runs' in why_off[0]            # round 2's sentence
        print(f'\nF21: Q9H-h5 {cid}: "{why[0][:150]}"')


# ---------------------------------------------------------------- T-r3-replay-g4u (F15, F16, F19, F23)
def g4u(lost_c6: bool = True, off=()) -> tuple[list, dict]:
    key = robust(G4U, lost_c6)
    seen = {}

    def on_run(i, eng, st, p, cand, row):
        if cand.id == 'c6' and 'pairs_c5' not in seen:
            seen['pairs_c5'] = len(st.suspects)
        if cand.id == 'c11' and 'c10' not in seen:                 # after c10 (the recompute seed out of memory on c6)
            seen['c10'] = {'spenders': [(h['key'], st._spender_mu(eng, h)) for h in st._spenders(eng, 'c6')],
                           'pairs': [(r['partner'], r['kind']) for r in st.pair_probes if r['base'] == 'c6'],
                           'bundles': [b for b in st.mem_bundles if b['base'] == 'c6' and b['origin'] == 'hls3-bundle']}
        if cand.id == 'c14' and 'c13' not in seen:                 # after c13 (recompute off on c12: thrash)
            seen['c13'] = {'suspects': sorted(sorted(k for m in pr for k, _ in m) for pr in st.suspects),
                           'swaps': [r for r in st.pair_probes if r['base'] == 'c12'], 'sp': eng.sigma_pair(),
                           'changes': [st.ledger[r['swap_key']]['change'] for r in st.pair_probes if r.get('swap_key')]}
    rows = replay(key, ask=True, adopt=True, follow=True, merge_probe=True, r1=True, r2=True, r3=True, on_run=on_run,
                  off=off)
    return rows, seen


def test_r3_replay_g4u_thrash_seed_pairs_and_swap():
    rows, seen = g4u()
    kind = {r['cid']: r['kind'] for r in rows if r['fidelity'] == 'full'}
    assert [kind[c] for c in ('c1', 'c3', 'c8', 'c13')] == ['thrash'] * 4 and kind['c11'] == 'slower'   # F23
    sp = dict(seen['c10']['spenders'])
    assert sp['c6|experience:gradient_checkpointing'] == pytest.approx(math.log1p(0.20), abs=1e-4)      # F15
    assert seen['c10']['pairs'] == [('torch_compile=True', 'composite'), ('use_liger_kernel=True', 'composite'),
                                    ('optim=adamw_torch_fused', 'composite')]                              # F16-A
    assert seen['pairs_c5'] == 6                                                                         # F19
    assert seen['c13']['suspects'] == [['gradient_checkpointing', 'torch_compile'], ['optim', 'torch_compile'],
                                       ['torch_compile', 'use_liger_kernel']]
    swap = [r for r in seen['c13']['swaps'] if r['status'] == 'swap']
    assert len(swap) == 1 and swap[0]['partner'] == 'torch_compile=True' and swap[0]['mu'] >= 2 * seen['c13']['sp']
    assert seen['c13']['changes'] == [{**RC_OFF, **COMPILE, 'optim': 'adamw_torch', 'use_liger_kernel': False}]
    print(f'\nF16 at c13: c12 without Liger and fused, with recompute off and compile: {seen["c13"]["changes"][0]}; '
          f'mu {swap[0]["mu"]:.4f} (2 sigma_pair {2 * seen["c13"]["sp"]:.4f})')
    _, off = g4u(off=['hls3_pair_probes', 'hls3_seed_mu', 'hls3_pair_set'])
    assert off['c10']['pairs'] == [] and off['pairs_c5'] == 0       # round 2: no pairs, no probes
    _, read = g4u(lost_c6=False)                                    # F14: the c6 round read - S-2 takes the spender first
    assert read['c10']['pairs'] == [] and read['c10']['bundles']
    assert 'SDPA' in ' '.join(read['c10']['bundles'][0]['enablers'])


def test_r3_pair_probes_stay_few_on_the_other_arms():
    counts = {}
    for name in (Q9H, M35):
        bases = {b['candidate']: b['base'] for b in FIX[name]['hls3_bundles'] if b.get('candidate')}
        out = {}

        def on_run(i, eng, st, p, cand, row, out=out):
            out['n'] = len({(r['moves'], r['base']) for r in st.pair_probes})
        replay(robust(name), ask=True, adopt=True, follow=True, merge_probe=True, r1=True, r2=True, r3=True,
               on_run=on_run, bases=bases)
        counts[name] = out.get('n', 0)
    print(f'\nF16 pair probes queued: {counts}')
    assert all(n <= 3 for n in counts.values())


# ---------------------------------------------------------------- T-r3-f22 (M35H2-h5)
def test_r3_f22_the_shared_micro_batch_change_waits():
    arm = FIX[M35]
    bases = {b['candidate']: b['base'] for b in arm['hls3_bundles'] if b.get('candidate')}
    tm = timing(arm)

    def run(off=()):
        return {r['cid']: r for r in replay(robust(M35), ask=True, adopt=True, merge_probe=True, r1=True, r2=True,
                                            r3=True, bases=bases, off=off)}
    on, off = run(), run(off=['hls3_shared_spender'])
    waits = [c for c in ('c11', 'c12', 'c16', 'c17', 'c18') if on[c]['adopted'] is None]
    assert waits == ['c11', 'c12', 'c16', 'c17', 'c18'] and on['c11']['kind'] == 'oom'
    assert off['c11']['adopted'] is not None                        # without F22 hls3 proposes c11 as recorded
    wall = sum((tm.get((c, 'probe')) or tm.get((c, 'screen')) or {}).get('wall_s') or 0.0 for c in waits)
    gpu = sum(r.get('gpu_hours') or 0.0 for r in arm['screens'] if r['candidate_id'] in waits)
    print(f'\nF22: M35H2-h5 {", ".join(waits)} not proposed: {wall:.0f} s of wall, {gpu:.2f} GPU-h')
