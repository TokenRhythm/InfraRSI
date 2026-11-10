"""FABLE_HLS3_FIXES_R1 on the recorded hls3 arms (the `-h3` arms of 2026-10-29/30, data/hls3_arms.json), offline: each
arm's runs fed in their recorded order to hls3 with the round-1 components on (hls3_helpers.replay, adopt + follow: a
recorded run that one of hls3's own proposals makes is observed as that proposal, a queued stall step none of whose
proposals ran is taken as served, and the incumbent is the recorded one). The numbers are what each record would have
saved at its recorded points - not what hls3 would have reached (its own trajectory would differ).

F1a: every probe's start cap at its base (4x the base's time to the first update, from the base's receipt), the probes
it would have stopped and the wall they spent past it; no launch that reached an update did so after its cap.
F1b: the gate's ratio on the passed probes (the probe's update intervals against its base's same ones).
F1c: the plain probes (not a stall, whole, bundle, coverage or swap step) whose move was neither unsure nor a memory
spender - the ones the spec's memory-only rule would no longer probe (a passed one saves its probe, a failed one becomes
a failed screen); the implemented rule keeps a start-risky move's probe, which every one of them was.
F2: the recorded full screens within 1 % of their base that hls3's coverage bundle on that base holds (start-safe knob
moves that do not spend memory: patches and spenders stay single).
F3: the out-of-memory tries of each spender (a knob value the recorded runs kept trying) that hls3 would still propose.
F4: no proposal hls3 makes fails the manifest's invariants (the experience composite and the explore move completed).
F5: G4U-h3's c15 (recompute off with compile on an incumbent holding Liger) is not proposed; a swap is.
F6: G4U-h3's "no activation recompute at mbs 2, made to fit by Liger" survives Liger's adoption as its rest.
F7: Q9H-h3's micro-batch 2 on c4 - its own p_feasible (0.15, a declared 40 GiB spend) holds it back, not a warning:
neither F7 nor F2 (spenders are no coverage members) moves it.
F8: Q27U-h3's c18 - another patch adding the code lines that c16 and c17 (one patch on two bases) added before both
crashed with the same TypeError - is ruled out by those lines."""
import math

from hls3_helpers import FIX, config_of, replay, timing

H3 = sorted(n for n in FIX if n.endswith('-h3'))
_CACHE: dict = {}
PLAIN = ('hls', 'hls-experience', 'hls-explore')


def r1(name: str) -> dict:
    """The arm replayed with the round-1 fixes: per recorded run its row plus F1's cap / gate / kept-probe reading, the
    coverage bundle's membership and the proposals the validator would refuse; the final strategy."""
    if name in _CACHE:
        return _CACHE[name]
    arm = FIX[name]
    tm = timing(arm)
    cands = {c['id']: c for c in arm['candidates']}
    seen, extra, refused, last = {}, [], [], {}
    val_steps = []

    def on_step(i, eng, st, props):
        seen['props'] = list(props or [])
        val = getattr(eng.s.hooks, 'validate_config', None)
        if val is not None:
            for q in props or []:
                if val(config_of(eng, q, q.base or eng.incumbent()))[1]:
                    refused.append((i, str(q.origin), q.change))
            val_steps.append(i)

    def on_run(i, eng, st, p, cand, row):
        base = p.base or 'c0'
        c = cands[cand.id]
        x = tm.get((cand.id, 'probe' if row['fidelity'] == 'probe' else 'screen')) or {}
        cfg = st._mcfg(eng, base, p.change, st._patch_ideas(p) if p.patch else ())
        rec = {'cap': st._probe_cap(eng, base), 'tf_base': (st._t_first(eng, base) or (None,))[0], 'wall': x.get('wall_s'),
               't_first': x.get('t_first_wall'), 'done': x.get('steps_completed')}
        if row['fidelity'] == 'probe':
            verdict = st._predict(eng, cfg)[0]
            rec['memory'] = verdict == 'unsure' or bool(st._floor_flag(eng, base, p.change)) or st._spends(eng, base, cfg)
            if x.get('intervals') and (x.get('steps_completed') or 0) >= 2:
                n0 = len(st.gated)
                rec['stop'] = bool(st._gate(eng, cand.id, base, {'intervals': x['intervals']}))
                rec['ratio'] = st.gated[-1]['ratio'] if len(st.gated) > n0 else None
                del st.gated[n0:]
                rec['ratio_raw'] = round(x['intervals'][0] / st._base_intervals(eng, base)[0], 2) \
                    if st._base_intervals(eng, base) else None
        if row['fidelity'] == 'full' and row.get('step_time') and eng.t(base):
            rec['gain'] = math.log(eng.t(base) / row['step_time'])
        cov = [q for q in seen.get('props') or [] if str(q.origin) == 'hls3-coverage' and (q.base or base) == base]
        ch = {k: v for k, v in c['config'].items() if eng.s.candidates[base].config.get(k) != v}
        rec['in_cov'] = bool(cov) and (p.hypothesis in (st.members.get(cov[0].hypothesis) or ()) or bool(ch) and all(
            cov[0].change.get(k) == v for k, v in ch.items()))
        rec['cov_members'] = [st.ledger[k]['change'] for k in st.members.get(cov[0].hypothesis) or () if k in st.ledger] \
            if cov else []
        extra.append(rec)
        last['st'], last['eng'] = st, eng

    rows = replay(name, ask=True, adopt=True, follow=True, on_step=on_step, on_run=on_run)
    for row, rec in zip(rows, extra):
        row.update(rec)
        row['config'] = cands[row['cid']]['config']
    world = int(arm['hardware'].get('gpus') or 8) * int(arm['hardware'].get('nodes') or 1)
    _CACHE[name] = {'rows': rows, 'st': last['st'], 'eng': last['eng'], 'refused': refused, 'world': world,
                    'launches': arm.get('attempts') or [], 'cands': cands}
    return _CACHE[name]


def caps_by_cid(r: dict) -> dict:
    return {row['cid']: row['cap'] for row in r['rows'] if row.get('cap')}


# ---------------------------------------------------------------- F1
def test_r1_f1_start_cap_on_h3_arms():
    print('\nF1a: probes stopped at the start cap (4x the base\'s time to the first update):')
    total = {}
    for name in H3:
        r = r1(name)
        killed = [(row['cid'], row['wall'], row['cap']) for row in r['rows'] if row['fidelity'] == 'probe' and row['cap']
                  and row['wall'] and (row['t_first'] is None and row['wall'] > row['cap']
                                       or row['t_first'] is not None and row['t_first'] > row['cap'])]
        saved = sum(w - c for _, w, c in killed)
        total[name] = (sorted(k for k, _, _ in killed), round(saved), round(saved * r['world'] / 3600, 2))
        print(f'  {name:24s} stopped {total[name][0]}  wall saved {total[name][1]} s  GPU-h {total[name][2]}')
        for row in r['rows']:                                       # (i) the loss of a start-failed probe <= 4 t_first
            if row['fidelity'] == 'probe' and row['cap'] and not row['done']:
                assert min(row['wall'] or 0, row['cap']) <= 4 * row['tf_base'] + 0.1, (name, row['cid'])
        caps = caps_by_cid(r)
        for x in r['launches']:                                     # (ii) no launch that reached an update came later
            c = r['cands'].get(x['cid']) or {}
            base = c.get('parent') or 'c0'
            cap = next((row['cap'] for row in r['rows'] if row['base'] == base and row.get('cap')), caps.get(x['cid']))
            if x['purpose'] in ('probe', 'screen') and x.get('t_first_wall') is not None and cap:
                assert x['t_first_wall'] <= cap, (name, x['cid'], x['t_first_wall'], cap)
    assert total['G4U-code-env-opus-h3'][0] == ['c15', 'c5'] and total['Q9H-code-env-opus-h3'][0] == ['c8']
    assert total['Q27H2-code-env-opus-h3'][0] == ['c10', 'c4']      # (c4: out of memory at 483 s, after its 446 s cap)
    assert total['Q27U-code-env-opus-h3'][0] == ['c27']             # (compile on the SAC + Liger incumbent: 3,670 s)
    assert not total['M35H2-code-env-opus-h3'][0] and not total['Q9U-code-env-opus-h3'][0]
    assert 9550 < sum(v[1] for v in total.values()) < 9700         # 883 + 755 + 3395 + 1391 + 37 + 3160 s


def test_r1_f1_probe_gate_on_h3_arms():
    print('\nF1b: passed probes, update interval against the base\'s (median ratio; >= 2.0 on a spender: no full run):')
    got = {}
    for name in H3:
        for row in r1(name)['rows']:
            if row['fidelity'] == 'probe' and row.get('ratio_raw') is not None:
                got[(name, row['cid'])] = (row['ratio_raw'], row.get('stop'))
                print(f'  {name:24s} {row["cid"]:4s} x{row["ratio_raw"]:.2f} {"STOP" if row.get("stop") else "go"}')
    assert got[('G4U-code-env-opus-h3', 'c2')] == (3.25, True)     # the full run aborted after 296 s: not run
    assert got[('Q9H-code-env-opus-h3', 'c5')][1] is False         # x1.80: below the 2.0 gate (a limit, NOTES)
    assert all(not got[(n, c)][1] for n, c in (('G4U-code-env-opus-h3', 'c9'), ('M35H2-code-env-opus-h3', 'c6'),
                                               ('M35H2-code-env-opus-h3', 'c9'), ('M35H2-code-env-opus-h3', 'c12')))
    stops = [k for k, v in got.items() if v[1]]
    assert stops == [('G4U-code-env-opus-h3', 'c2')]
    assert max(v[0] for k, v in got.items() if k not in (('G4U-code-env-opus-h3', 'c2'), ('Q9H-code-env-opus-h3', 'c5'))) \
        <= 1.31                                                     # every other passed probe: within x1.31


def test_r1_f1_kept_probes_on_h3_arms():
    """FABLE's F1c as written keeps a probe only for an unsure or memory-spending move: on the records it would have
    saved 8 passed probes (758 s) and turned 6 failed ones into failed full screens - three of them hangs that a full screen,
    with no start cap (D1), runs to the job's launch timeout. The implemented rule keeps a start-risky move's probe too (its
    feasibility below the probe rule - a plain probe that is not about memory had one, or hls3 would not have probed
    it): no failed probe becomes a screen."""
    print('\nF1c: plain probes (not a stall, whole, bundle, coverage or swap step) not kept:')
    lit = {'passed': [], 'failed': []}
    for name in H3:
        r = r1(name)
        for row in r['rows']:
            if row['fidelity'] != 'probe' or row['origin'] not in PLAIN:
                continue
            if not row['memory']:
                lit['passed' if row['status'] == 'probed' else 'failed'].append((name[:5], row['cid'], row['wall'],
                                                                                 row['kind']))
    w = sum(x[2] or 0 for x in lit['passed'])
    print(f'  as written (memory only): passed probes saved {[x[:2] for x in lit["passed"]]} ({round(w)} s); failed '
          f'probes that become failed screens {[(x[0], x[1], x[3]) for x in lit["failed"]]}')
    assert len(lit['passed']) == 8 and len(lit['failed']) == 6
    assert sum(1 for x in lit['failed'] if x[3] == 'start_cap') == 3      # Q27H2 c10, Q9H c8, Q27U c27: hangs


# ---------------------------------------------------------------- F2, F3, F4
def test_r1_f2_coverage_on_h3_arms():
    print('\nF2: recorded full screens within 1 % of their base that the coverage bundle holds:')
    merged = {}
    for name in H3:
        rows = [row for row in r1(name)['rows'] if row['fidelity'] == 'full' and row['status'] == 'screened'
                and row.get('gain') is not None and abs(row['gain']) <= 0.01]
        merged[name] = ([row['cid'] for row in rows if row['in_cov']], [row['cid'] for row in rows])
        print(f'  {name:24s} {len(merged[name][0])} of {len(merged[name][1])}: {merged[name][0]} (of {merged[name][1]})')
    assert merged['Q9H-code-env-opus-h3'][0] == ['c6', 'c7'] and merged['M35H2-code-env-opus-h3'][0] == ['c11', 'c15']
    assert merged['G4U-code-env-opus-h3'][0] == ['c8', 'c10']       # c11 / c17: other values of the workers knob
    assert sum(len(v[0]) for v in merged.values()) == 7             # (Q9H c2 / c3, M35H2 c1 / c3 / c13 / c14: patches)
    assert all(set(r1(n)['rows'][0]['cov_members'][0]) for n in H3 if r1(n)['rows'][0]['cov_members'])


SPENDERS = {'G4U-code-env-opus-h3': ('fsdp_config', 'fsdp_v2_noreshard.json'),
            'M35H2-code-env-opus-h3': ('gradient_checkpointing_kwargs', '{"every_n_layers": 4}'),
            'Q27H2-code-env-opus-h3': ('gradient_checkpointing_kwargs', '{"every_n_layers": 3}'),
            'Q27U-code-env-opus-h3': ('gradient_checkpointing', False)}


def test_r1_f3_tries_per_spender_on_h3_arms():
    print('\nF3: out-of-memory tries of each spender the recorded arm kept trying, and those hls3 would still propose:')
    saved = [0, 0.0, 0.0]
    for name, (k, v) in SPENDERS.items():
        r = r1(name)
        rows = [row for row in r['rows'] if row['kind'] == 'oom' and row['config'].get(k) == v]
        kept = [row['cid'] for row in rows if row['adopted']]
        gone = [row for row in rows if not row['adopted']]
        w = sum(row['wall'] or 0 for row in gone)
        saved = [saved[0] + len(gone), saved[1] + w, saved[2] + w * r['world'] / 3600]
        print(f'  {name:24s} {k}={v}: recorded {[row["cid"] for row in rows]}  still proposed {kept}  '
              f'(saved {len(gone)} tries, {round(w)} s, {w * r["world"] / 3600:.2f} GPU-h)')
        assert len(kept) <= 3 < len(rows), name
    print(f'  total: {saved[0]} tries, {round(saved[1])} s, {saved[2]:.1f} GPU-h')
    assert saved[0] == 16
    st = r1('Q27U-code-env-opus-h3')['st']
    n = next(x for x in st.need.values() if x['change'] == {'gradient_checkpointing': False})
    assert n['probes'] == 3 and n['need_lb'] > 0


def test_r1_f4_no_refused_proposal_on_h3_arms():
    for name in H3:
        r = r1(name)
        assert not r['refused'], (name, r['refused'][:3])
    assert 'bundle|exp' in r1('Q27U-code-env-opus-h3')['st'].completed      # the experience composite, completed
    assert any('gradient_accumulation_steps=2' in k for k in r1('M35H2-code-env-opus-h3')['st'].completed)


# ---------------------------------------------------------------- F5, F6 (G4U), F7 (Q9H), F8 (Q27U)
def test_r1_g4u_path():
    r = r1('G4U-code-env-opus-h3')
    by = {row['cid']: row for row in r['rows']}
    st = r['st']
    assert by['c15']['adopted'] is None and by['c13']['adopted'] is None and by['c14']['adopted'] is None
    pairs = [{w for w in x['pair']} for x in st.describe()['hls3']['r1']['suspect_pairs']]
    assert {'torch_compile=True', 'use_liger_kernel=True'} in pairs                     # F5a, from c5
    assert any(s['proposed'] and s['without'] == 'use_liger_kernel=True' for s in st.swaps)   # F5c
    rest = [x for x in st.remainders if x['on'] == 'c7' and x['rest'].get('gradient_checkpointing') is False]
    assert rest                                                                       # F6: recompute off, the rest
    carried = [h for h in st.ledger.values() if h['base'] == 'c9' and h.get('change') == {'gradient_checkpointing': False}]
    assert carried and carried[0]['mu'] > 2 * r['eng'].sigma_pair()
    print(f'\nG4U-h3: c13 / c14 / c15 not proposed; swap {[s["without"] for s in st.swaps if s["proposed"]]}; rest on c7 '
          f'{rest[0]["rest"]} at +{rest[0]["mu_pct"]} %')


def test_r1_q9h_rank():
    r = r1('Q9H-code-env-opus-h3')
    st = r['st']
    mbs = {'per_device_train_batch_size': 2, 'gradient_accumulation_steps': 1}
    on_c4 = [row for row in r['rows'] if row['base'] == 'c4' and row['cov_members']]
    assert on_c4 and not any(all(m.get(k) == v for k, v in mbs.items()) for m in on_c4[0]['cov_members'])
    h = next(h for h in st.ledger.values() if h['base'] == 'c4' and h.get('change') == mbs)
    assert h['p_feasible'] == 0.15 and h['mem_dir'] == 'spends' and st.warn_kept == 0   # its own feasibility, no warning
    print(f'\nQ9H-h3 on c4: micro-batch 2 (mu {h["mu"]:.4f}, p_feasible {h["p_feasible"]}, declared spend '
          f'{h.get("mem_gib")} GiB): not bundled (a spender); the coverage bundle holds {on_c4[0]["cov_members"]}')


def test_r1_q27u_dead_patch():
    r = r1('Q27U-code-env-opus-h3')
    by = {row['cid']: row for row in r['rows']}
    st = r['st']
    lines = [e for e in st.dead_elements if e[0] == 'line']
    assert lines and any('gradient_checkpointing_enable' in e[1] for e in lines)
    assert by['c18']['adopted'] is None
