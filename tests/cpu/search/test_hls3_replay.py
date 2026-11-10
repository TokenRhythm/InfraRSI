"""hls3 on the recorded arms of 2026-10-28/30, offline (hls3_helpers: the arms' runs fed in their recorded order; hls3's
own trajectory would differ, so these are counts of what it would have known and proposed at each recorded point, not
speed-ups - no claim of a real gain without real runs).

S-1: of the failed full screens each recorded arm spent, how many hls3's feasibility model would have kept off a screen
from the runs before them - deferred (predicted out of memory by measured moves: not run), or probed first (unsure, a
bound the declarations alone give, or the proposer's own feasibility below the probe rule with the probe kept: a failure
in the first update costs no screen) - and how many valid screens it would have deferred wrongly (none). With the
declarations the recorded proposers' words give (DECLARED) and without any (the runs and the knob order only): the same
deferrals either way, a declaration never rules a configuration out (Codex r1). The random arms: what the model reads
from random draws (information only: random keeps the whole table).
S-2 / S-4 / S-6: on G4U-code-env-opus-h2 (Gemma-4-26B-A4B: random's winner set compile with recompute off; neither
InfraRSI arm tried compile alone), whether hls3 would have generated recompute off together with compile from what the arm
knew - no recorded call ever said that compile frees memory.
S-10: which recorded proposals set a value whose declared spend puts the sharded-state floor above their card (probed
first, never pruned: the fixture has no accounting of the configuration's state, the only thing that prunes)."""
import collections

from hls3_helpers import FIX, LLM_ARMS, RANDOM_ARMS, declared, failed_screen, open_loop, replay, would


def counts(name: str, declarations: bool = True) -> dict:
    rows = replay(name, declarations=declarations)
    out = {}
    for label, sel in (('oom', lambda r: failed_screen(r) and r['kind'] == 'oom'),
                       ('other_failed', lambda r: failed_screen(r) and r['kind'] != 'oom'),
                       ('thrash', lambda r: r['kind'] == 'thrash'),
                       ('valid', lambda r: r['fidelity'] == 'full' and r['kind'] in ('ok', 'slower'))):
        rs = [r for r in rows if sel(r)]
        c = collections.Counter(would(r) for r in rs)
        out[label] = {'n': len(rs), 'deferred': c['deferred'], 'probed': c['probed'], 'screened': c['screened']}
    return out


def table(declarations: bool) -> dict:
    res = {name: counts(name, declarations) for name in LLM_ARMS + RANDOM_ARMS}
    print(f'\nhls3 feasibility replay, declarations {"on" if declarations else "off"} (deferred / probed / still screened):')
    for name, c in res.items():
        print(f'  {name:24s} ' + '  '.join(f'{k} {v["n"]}: {v["deferred"]}/{v["probed"]}/{v["screened"]}' for k, v in c.items()))
    return res


def test_replay_feasibility_with_the_recorded_declarations():
    res = table(True)
    llm = [res[n] for n in LLM_ARMS]
    oom = sum(c['oom']['n'] for c in llm)
    kept = sum(c['oom']['deferred'] + c['oom']['probed'] for c in llm)
    print(f'  LLM arms: {kept} of {oom} out-of-memory full screens kept off a screen '
          f'({sum(c["oom"]["deferred"] for c in llm)} deferred, {sum(c["oom"]["probed"] for c in llm)} probed first)')
    assert (oom, kept) == (14, 13)                                  # the one missed: G4U-h2's first run (c1), nothing before it
    assert res['G4U-code-env-opus-h2']['oom'] == {'n': 5, 'deferred': 0, 'probed': 4, 'screened': 1}   # (c10: declared)
    assert res['M35H2-code-env-opus']['oom'] == {'n': 4, 'deferred': 1, 'probed': 3, 'screened': 0}
    assert res['M35H2-code-env-opus-h2']['oom'] == {'n': 3, 'deferred': 2, 'probed': 1, 'screened': 0}
    assert all(c['valid']['deferred'] == 0 for c in res.values())   # no run that fitted would have been deferred
    assert res['Q9U-code-env-opus-h2']['other_failed'] == {'n': 2, 'deferred': 0, 'probed': 0, 'screened': 2}   # compile:
    alone = {n: counts(n, declarations=False) for n in res}         # the launch cap, not memory
    assert all(res[n][k]['deferred'] == alone[n][k]['deferred'] for n in res for k in res[n])   # declarations defer nothing
    rnd = [res[n] for n in RANDOM_ARMS]
    print(f'  random arms: of {sum(c["oom"]["n"] for c in rnd)} out-of-memory draws, '
          f'{sum(c["oom"]["deferred"] for c in rnd)} predicted from earlier draws, {sum(c["oom"]["probed"] for c in rnd)} '
          'flagged unsure')


def test_replay_feasibility_from_the_runs_alone():
    res = table(False)
    llm = [res[n] for n in LLM_ARMS]
    assert sum(c['oom']['deferred'] + c['oom']['probed'] for c in llm) == 12   # G4U-h2 c10 (recompute off) is missed
    assert res['G4U-code-env-opus-h2']['oom'] == {'n': 5, 'deferred': 0, 'probed': 3, 'screened': 2}
    assert all(c['valid']['deferred'] == 0 for c in res.values())
    assert all(res[n]['oom']['deferred'] + res[n]['oom']['probed'] == 0 for n in RANDOM_ARMS)   # draws differ in many
                                                                                                  # knobs: nothing to read


def stall_proposals(name: str) -> list[tuple[int, str, dict, str]]:
    """(recorded step, origin, change, base) of every hls3 bundle, stall probe and whole configuration it would have
    proposed along the arm's recorded runs."""
    seen = []

    def on_step(i, eng, st, props):
        for p in props or []:
            if str(p.origin).startswith('hls3') and (i, p.origin, p.change) not in [(s[0], s[1], s[2]) for s in seen]:
                seen.append((i, p.origin, dict(p.change), p.base))
    replay(name, ask=True, on_step=on_step)
    return seen


def test_g4u_hls3_proposes_recompute_off_with_compile_at_its_first_stall():
    """G4U-code-env-opus-h2 (hls2, 1.271x vs random's 1.590x): after micro-batch 2 (c2), recompute variants failed (c3
    slower, c4 probe OOM, c5 OOM). hls3's first stall comes after the arm's fourth full screen; its feasibility probes pair
    the out-of-memory spender (recompute off with expandable segments, c4's idea) with each enabler, compile among the
    moves whose memory effect nobody stated; and its whole-configuration step is the experience composite (recompute off,
    compile, Liger, fused AdamW). Random reached 2.128 s with compile + recompute off + micro-batch 2; the pre-check
    measured that bundle at 2.163 s (K'). Whether hls3 would get there is for a real arm."""
    seen = stall_proposals('G4U-code-env-opus-h2')
    for s in seen:
        print('  step', s[0], s[1], s[2], 'on', s[3])
    hit = [s for s in seen if s[2].get('gradient_checkpointing') is False and s[2].get('torch_compile') is True]
    assert hit and hit[0][0] == 5 and hit[0][3] == 'c2'             # on micro-batch 2, before the recorded sixth run
    probe = [s for s in hit if s[1] == 'hls3-stall-probe']
    assert probe and probe[0][2] == {'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'gradient_checkpointing': False,
                                     'torch_compile': True}
    assert any(s[1] == 'hls3-whole-experience' for s in hit)
    assert not [s for s in seen if s[2].get('torch_compile') and s[0] < 5]   # nothing before the stall paired them


def test_g4u_first_round_arm_gets_the_pair_only_in_the_composite():
    """G4U-code-env-opus (hls, 1.417x): the top blocked spender at its stalls is every-4th-layer recompute (out of memory
    on c0), so its probes pair that with other moves; recompute off with compile comes only in the experience composite
    (with Liger, which the arm later saw crash with compile: G4U-code c9) - an honest miss for the probes."""
    seen = stall_proposals('G4U-code-env-opus')
    for s in seen:
        print('  step', s[0], s[1], s[2], 'on', s[3])
    pair = [s for s in seen if s[2].get('gradient_checkpointing') is False and s[2].get('torch_compile') is True]
    assert pair and {s[1] for s in pair} == {'hls3-whole-experience'}
    assert all(s[1] != 'hls3-stall-probe' or '{"every_n_layers": 4}' in str(s[2]) for s in seen)


def test_the_card_size_arithmetic_flags_unsharded_weights_only_where_they_cannot_fit():
    """S-10 with the proposer's own number for unsharded bf16 weights (2 B x the parameters, as the G4U-h2 model said:
    "about 48 GiB of bf16 parameters"): above the card for 35B on 16 x 80 GB (sharded state 32.6 GiB + 65.2 > 77.3) -
    probed first, not pruned (a declaration never prunes; what prunes is the shared accounting of the configuration's
    own state, which these jobs' runs contradict: it put their references at 510 GiB per GPU) - and not flagged for 27B
    on 16 x 80 GB (borderline: Q27H2-rnd's three such draws ran, slowly), 26B and 9B on 8 x 141 GB. The M35H2 random arm
    drew it in 7 of 10 configurations; random keeps the whole table."""
    out = {}
    for name in ('M35H2-code-env-opus', 'Q27H2-code-env-opus', 'G4U-code-env-opus-h2', 'Q9U-code-env-opus-h2'):
        eng, st, arm = open_loop(name)
        for i, (ch, d, g, _) in enumerate(declared(arm)):
            st.ledger[f'c0|d{i}'] = {'mechanism': f'd{i}', 'change': ch, 'patch': None, 'mu': 0.0, 'sigma': 0.01,
                                     'p_feasible': 0.9, 'base': 'c0', 'status': 'superseded', 'key': f'c0|d{i}',
                                     'mem_dir': d, **({'mem_gib': g} if g is not None else {})}
        out[name] = st._floor_flag(eng, 'c0', {'fsdp_config': 'fsdp_v2_noreshard.json'})
        assert st._prune_why(eng, 'c0', {'fsdp_config': 'fsdp_v2_noreshard.json'}) is None   # nothing pruned
        print(f'  {name}: floor {st.mem.floor["floor_gib"]} GiB of {st.mem.cap}: {out[name] or "not flagged"}')
    assert 'above the 77 GiB card' in out['M35H2-code-env-opus']
    assert out['Q27H2-code-env-opus'] is None and out['G4U-code-env-opus-h2'] is None and out['Q9U-code-env-opus-h2'] is None
    draws = [c for c in FIX['M35H2-rnd']['candidates'] if c['id'] != 'c0']
    hit = [c['id'] for c in draws if c['config'].get('fsdp_config') == 'fsdp_v2_noreshard.json']
    print(f'  M35H2-rnd draws with unsharded weights: {len(hit)} of {len(draws)} ({", ".join(hit)})')
    assert len(hit) == 7 and len(draws) == 10


def test_q27_hls3_probes_partial_recompute_with_liger_at_its_first_stall():
    """Q27H2-code-env-opus (Qwen3.8-27B, 16 x 80 GB, hls 1.205x): every_n_layers 2 ran out of memory on c0 (its probe, c2)
    and fitted only at c11, after a Liger-style fused-CE patch (c10) - the pre-check's K (Liger flag + fused AdamW +
    expandable + every_n_layers 2) ran at 2.112 s. hls3's first stall (after the arm's third run) probes every_n_layers 2
    with expandable segments, fused AdamW and the Liger knob (no recorded call stated Liger's memory effect on this job:
    an unknown-direction move); the experience composite (Liger, fused AdamW, expandable) is its whole step. M35H2 (35B on
    16 x 80 GB): its probes pair every 4th layer with other moves - nothing fits there (all recompute variants ran out of
    memory in both arms); no claim."""
    seen = stall_proposals('Q27H2-code-env-opus')
    hit = [s for s in seen if s[1] == 'hls3-stall-probe' and s[2] == {
        'gradient_checkpointing_kwargs': '{"every_n_layers": 2}', 'use_liger_kernel': True}]
    assert hit and hit[0][0] == 3 and hit[0][3] == 'c0'
    m35 = stall_proposals('M35H2-code-env-opus-h2')
    assert not [s for s in m35 if 'gradient_accumulation_steps' in s[2] and 'per_device_train_batch_size' not in s[2]]


def test_slow_draws_without_memory_evidence_stay_slowdowns():
    """Codex r1 P1-4 on the record: Q27H2-rnd c5, c6 and c9 set unsharded weights with other changes and ran 1.8-1.9x
    slower than c0, with no recorded peak; c1 (unsharded weights with others) ran out of memory at 75.9 GiB of 77.3 GiB,
    but no measured move bounds c5, c6 or c9 by c1. So they are possible thrash, noted, not memory boundaries (the old
    rule made them thrash from the declared spends alone); they count as runs that fitted."""
    seen = []
    replay('Q27H2-rnd', on_step=lambda i, eng, st, props: seen.append(st))
    st = seen[-1]
    assert st.counts['possible_thrash'] == 3 and not [f for f in st.failures if f['class'] == 'thrash']
    assert all(st.mem.runs[c]['status'] == 'ok' for c in ('c5', 'c6', 'c9'))
