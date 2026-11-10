"""Harness of the hls3 replay tests (FABLE_TAKEAWAYS_20261030 section 4): the recorded arms of 2026-10-28/30
(data/hls3_arms.json, made by data/build_hls3_arms.py from qa_trials/trials_tb - tests read only the fixture) replayed
open-loop: every recorded run is fed, in its order, to an hls3 strategy as its outcome, and before each run the strategy is
asked what it would have predicted (S-1) or proposed (S-2, S-4, S-6) from the runs before it. The arm's own trajectory is
kept (hls3 would have taken another), so the counts say what hls3 would have known at each point of the recorded search,
not what it would have reached.

The proposer's memory declarations (hls3's schema field) did not exist in these runs; DECLARED transcribes what the
recorded proposers said about memory in their rationales, the transfer judgment and its memory note, quoted. No statement
about compile's memory effect was made in any recorded call, so compile is not declared."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

from hls2_helpers import SimEngine, SimSession
from infrarsi.agent.capability import search_params
from infrarsi.search import HLS3_R1, HLS3_R2, HLS3_R3
from infrarsi.search.engine import Outcome, Proposal
from infrarsi.search.hls3 import HLS3

FIX = json.loads((Path(__file__).resolve().parent / 'data' / 'hls3_arms.json').read_text(encoding='utf-8'))
R3_FIX = json.loads((Path(__file__).resolve().parent / 'data' / 'hls3_r3.json').read_text(encoding='utf-8'))
FIX.update(R3_FIX['arms'])                              # (the -h5 arms of FABLE_HLS3_FIXES_R3: build_hls3_r3.py)
LLM_ARMS = ('G4U-code-env-opus', 'G4U-code-env-opus-h2', 'Q27H2-code-env-opus', 'M35H2-code-env-opus',
            'M35H2-code-env-opus-h2', 'Q9U-code-env-opus', 'Q9U-code-env-opus-h2')
RANDOM_ARMS = ('G4U-rnd', 'Q27H2-rnd', 'M35H2-rnd', 'Q9U-rnd')
PATCH_FILE = 'train_patches.py'
EV = '{"every_n_layers": %d}'


def declared(arm: dict) -> list[tuple]:
    """(change from the reference, direction, GiB or None, the recorded words) for the knobs this arm has."""
    bf16 = 2 * float(arm['model']['params_b']) * 1e9 / 2 ** 30      # the proposer's own arithmetic for unsharded weights
    rows = [({'per_device_train_batch_size': 2, 'gradient_accumulation_steps': 1}, 'spends', 5.0,
             'G4U-h2 hypotheses on c0: "With recompute on, the extra activations and logits are only a few GiB"'),
            ({'gradient_checkpointing': False}, 'spends', None,
             'G4U-h2 transfer: recompute off needs room for the activations ("leaves room for activations at seq 4k")'),
            ({'gradient_checkpointing_kwargs': EV % 4}, 'spends', 30.0,
             'G4U-h2 hypotheses on c0: every 4th layer "costs only a few tens of GiB"'),
            ({'gradient_checkpointing_kwargs': EV % 2}, 'spends', 10.0,
             'G4U-h2 hypotheses on c0: every 2nd layer is "the lower-memory point of the same recompute curve"'),
            ({'fsdp_config': 'fsdp_v2_noreshard.json'}, 'spends', round(bf16, 1),
             'G4U-h2 hypotheses on c0: "Memory cost is about 48 GiB of bf16 parameters" (2 B x the parameters)'),
            ({'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}, 'frees', 2.0,
             'G4U-h2 hypotheses on c0: fragmentation "can trigger cudaMalloc/free retries. Expandable segments avoid the churn"'),
            ({'use_liger_kernel': True}, 'frees', 10.0,
             'G4U-h2 transfer: "about 10 GiB of fp32 logits at 4k; fused CE skips materializing them"')]
    rows += [({k: v}, 'neutral', None, 'no memory effect named (optimizer kernel, matmul precision, collectives, loader)')
             for k, v in (('optim', 'adamw_torch_fused'), ('tf32', True), ('NCCL_NVLS_ENABLE', '0'),
                          ('dataloader_num_workers', 0), ('dataloader_num_workers', 4), ('dataloader_num_workers', 8),
                          ('dataloader_pin_memory', False), ('NCCL_MIN_NCHANNELS', '16'), ('NCCL_MIN_NCHANNELS', '32'))]
    return [r for r in rows if all(k in arm['knobs'] and v in arm['knobs'][k] for k, v in r[0].items())]


class ReplayEngine(SimEngine):
    """A SimEngine that does not run anything: the open-loop replay sets its candidates, incumbent and noise."""

    def __init__(self, session, sigma_pair: float):
        super().__init__(session, None, sigma_runs=[sigma_pair / math.sqrt(2)], full_costs=[0.3])
        self.sp = sigma_pair

    def sigma_pair(self):
        return self.sp

    def sigma_run(self):
        return self.sp / math.sqrt(2)


class RecordedProposer:
    """The recorded model: on a base, the hypotheses of the recorded call made on it (each call once; none for a base the
    arm never asked on), with hls3's memory fields where the recorded words give them (`declarations`); a patch as a
    one-line marker (its code is not in the fixture)."""

    def __init__(self, arm: dict, declarations: bool = True):
        self.arm, self.calls, self.failed, self.used = arm, 0, 0, set()
        self.decl = declared(arm) if declarations else []
        self.decl_on = declarations
        kp = search_params(0.9)
        self.sigma_scale = kp['sigma_scale']

    def describe(self):
        return {'kind': 'RecordedProposer'}

    def fields(self, change: dict) -> dict:
        for ch, d, g, _ in self.decl:
            if change == ch:
                return {'mem_dir': d, **({'mem_gib': g} if g is not None else {})}
        return {}

    def hypotheses(self, engine, base, ledger=None):
        self.calls += 1
        out = []
        for i, call in enumerate(self.arm['calls']):
            if call['role'] != 'hypotheses' or call['base'] != base or i in self.used:
                continue
            self.used.add(i)
            for h in call['hypotheses']:
                ch = h.get('change') if isinstance(h.get('change'), dict) else {}
                g = float(h.get('predicted_gain_pct') or 0) / 100
                u = float(h.get('uncertainty_pct') or max(1.0, abs(g) * 50)) / 100
                patch = {PATCH_FILE: f'# {h["mechanism"]}\n'} if h.get('patch') else None
                if not ch and not patch:
                    continue
                out.append({'mechanism': h['mechanism'], 'change': dict(ch), 'patch': patch,
                            'mu': math.log1p(max(-0.9, g)), 'sigma': max(0.005, u * self.sigma_scale),
                            'p_feasible': min(1.0, max(0.0, float(h.get('p_feasible') if h.get('p_feasible') is not None
                                                                 else 0.8))),
                            **(self.recorded(h) if 'memory' in h or 'site' in h else {} if patch else self.fields(ch))})
            break
        return out

    def recorded(self, h: dict) -> dict:
        """An hls3 arm's own fields (its proposer declared memory and named code sites), parsed as hls3_fields does."""
        from infrarsi.search.proposers import hls3_fields
        return hls3_fields(h, {'memory': self.decl_on, 'sites': True, 'variance': True})


def experience(arm: dict) -> dict | None:
    """The run's experience plan as hls saw it: the transfer judgment and its start configuration as zero-size seeds."""
    tr = next((c for c in arm['calls'] if c['role'] == 'transfer'), None)
    comp = arm.get('experience_composite')
    if not tr or not comp:
        return None
    judged = {a['knob']: a for a in tr.get('assessment') or [] if a.get('knob') in arm['knobs']}
    seeds = [{'knob': n, 'mechanism': n, 'change': {n: v}, 'level': 'partial', 'gain': 0.0, 'evidence_ids': [], 'warnings': [],
              'sign': '+', 'p_feasible': (judged.get(n) or {}).get('p_feasible') or 0.7, 'valid_on': {n: arm['reference'][n]}}
             for n, v in comp['change'].items()]
    assessment = [{'knob': a['knob'], 'change': a['change'], 'sign': a.get('sign'), 'p_feasible': a.get('p_feasible'),
                   'expected_gain_pct': a.get('expected_gain_pct')}       # (read by hls3 F15 only)
                  for a in judged.values() if isinstance(a.get('change'), dict)]
    return {'seeds': seeds, 'composite': {**comp, 'warnings': []},
            'judgment': {'assessment': assessment, 'start_config': dict(tr.get('start_config') or {})}}   # (F19's set)


def validator(arm: dict):
    """ProjectHooks.validate_config's invariant check for the arm's manifest: a configuration whose workload invariants
    (the global batch) differ from the reference's is refused."""
    from infrarsi.project.manifest import evaluate
    inv = (arm.get('workload') or {}).get('invariants') or {}
    hw = arm.get('hardware') or {}

    def names(cfg: dict) -> dict:
        return {**{k: cfg.get(k) for k in arm['knobs']}, 'gpus': int(hw.get('gpus') or 8), 'nodes': int(hw.get('nodes') or 1)}
    ref = {k: evaluate(e, names(arm['reference'])) for k, e in inv.items()}

    def validate_config(config: dict):
        merged = {**arm['reference'], **config}
        try:
            cur = {k: evaluate(e, names(merged)) for k, e in inv.items()}
        except Exception as exc:                        # noqa: BLE001 - an unknown name: refused as the session would
            return {}, [f'invariant: {exc}'], []
        diff = [f'{k} {cur[k]} != reference {ref[k]}' for k in cur if cur[k] != ref[k]]
        return ({}, ['workload invariant changed: ' + '; '.join(diff)], []) if diff else (merged, [], [])
    return validate_config


def open_loop(name: str, *, declarations: bool = True, off=(), with_proposer: bool = True, r1: bool | None = None,
              r2: bool = False, r3: bool = False):
    """(engine, hls3, arm): an hls3 strategy on the arm's job, nothing run yet. `r1`: FABLE_HLS3_FIXES_R1's components on
    (default: for the hls3 arms, `-h3`, only; the older arms replay hls3 as of r4, the fixes off). `r2`: the round-2
    components (FABLE_HLS3_FIXES_R2) on - only test_hls3_r2_replay asks for them; the older replays stay as they were.
    `r3`: the round-3 components (FABLE_HLS3_FIXES_R3) on - test_hls3_r3_replay."""
    on = name.endswith('-h3') if r1 is None else r1
    if not on:
        off = tuple(off) + HLS3_R1
    if not r2:
        off = tuple(off) + HLS3_R2
    if not r3:
        off = tuple(off) + HLS3_R3
    arm = FIX[name]
    knobs = {k: v for k, v in arm['knobs'].items() if v}
    session = SimSession(arm['reference'], knobs=knobs, screens=int((arm.get('budget') or {}).get('screens') or 10),
                         t0=arm['reference_step_time'])
    sp = float(((arm.get('noise') or {}).get('sigma_pair')) or 0.01)
    eng = ReplayEngine(session, sp)
    if on and (arm.get('workload') or {}).get('invariants'):
        session.hooks.validate_config = validator(arm)  # (F4: the manifest's invariants, as ProjectHooks checks them)
    hw = arm.get('hardware') or {}
    groups = [sorted(k for k in ('per_device_train_batch_size', 'gradient_accumulation_steps') if k in knobs)]
    facts = {'cap_gib': hw.get('memory_gib'), 'world': int(hw.get('gpus') or 8) * int(hw.get('nodes') or 1),
             'params_b': (arm.get('model') or {}).get('params_b'), 'coupled': [g for g in groups if len(g) > 1]}
    kp = search_params(0.9)
    prop = RecordedProposer(arm, declarations) if with_proposer else RecordedProposer({**arm, 'calls': []})
    st = HLS3(prop, kappa0=kp['kappa0'], probe_below=kp['probe_below'], experience=experience(arm) if with_proposer else None,
              facts=facts, off=off)
    if not with_proposer and declarations:             # the declarations alone (the random arms: no model)
        for i, (ch, d, g, _) in enumerate(declared(arm)):
            st.ledger[f'c0|declared{i}'] = {'mechanism': f'declared {i}', 'change': ch, 'patch': None, 'mu': 0.0, 'sigma': 0.01,
                                            'p_feasible': 0.9, 'base': 'c0', 'status': 'superseded', 'key': f'c0|declared{i}',
                                            'mem_dir': d, **({'mem_gib': g} if g is not None else {})}
    return eng, st, arm


def outcome_of(arm: dict, row: dict) -> tuple[str, list, list]:
    c = next(x for x in arm['candidates'] if x['id'] == row['candidate_id'])
    if row['fidelity'] == 'probe':
        return ('probed' if row['valid'] else 'infeasible'), c['reasons'], c.get('failed_checks') or []
    if row.get('aborted'):
        return 'aborted', c['reasons'], []
    return ('screened' if row['valid'] else 'invalid'), c['reasons'], c.get('failed_checks') or []


def recorded_p(arm: dict, base: str, change: dict) -> float | None:
    """The feasibility hls gave this change on this base: the recorded model's p_feasible for exactly this change, else
    for a single-knob change the transfer judgment's (an experience seed, clipped to 0.1-0.9 as HLS._seed does) or a
    neighbour's 0.6; None for a random draw or a change none of them made."""
    ps = [h.get('p_feasible') for c in arm['calls'] if c['role'] == 'hypotheses' and c['base'] == base
          for h in c['hypotheses'] if h.get('change') == change and not h.get('patch')]
    p = next((float(p) for p in ps if isinstance(p, (int, float))), None)
    if p is not None or len(change) != 1 or arm['strategy'] == 'random':
        return p
    tr = next((c for c in arm['calls'] if c['role'] == 'transfer'), None) or {}
    judged = next((a.get('p_feasible') for a in tr.get('assessment') or [] if a.get('change') == change), None)
    return min(0.9, max(0.1, float(judged))) if isinstance(judged, (int, float)) else 0.6


def timing(arm: dict) -> dict:
    """(candidate, purpose) -> its recorded launch (the hls3 arms: build_hls3_arms.attempts; purpose reference, probe,
    screen)."""
    out = {}
    for x in arm.get('attempts') or []:
        out.setdefault((x['cid'], x['purpose']), x)
    return out


def stamped(x: dict | None) -> dict:
    """A launch's timing as a receipt holds it: its wall and stamped updates - the first at 0, the next ones by its
    recorded intervals, the last at wall - time to the first update (so the session's wall - span gives that time)."""
    if not x or not isinstance(x.get('wall_s'), (int, float)):
        return {}
    out = {'wall_s': x['wall_s']}
    tf = x.get('t_first_wall')
    if isinstance(tf, (int, float)) and x.get('intervals'):
        ts = [0.0]
        for d in x.get('intervals_all') or x['intervals']:   # (the -h5 arms keep every interval: F23's scatter)
            ts.append(ts[-1] + d)
        if x['wall_s'] - tf > ts[-1] + 1e-6:
            ts.append(x['wall_s'] - tf)
        out['per_step'] = [{'step': i + 1, 't': t} for i, t in enumerate(ts)]
        out['per_step'][-1]['step'] = max(len(ts), int(x.get('steps_completed') or 0))
    return out


def replay(name: str, *, declarations: bool = True, ask: bool = False, off=(), on_step=None,
           r1: bool | None = None, on_run=None, adopt: bool = False, follow: bool = False, r2: bool = False,
           r3: bool = False, merge_probe: bool = False, bases: dict | None = None, gains: dict | None = None,
           incumbents=None) -> list[dict]:
    """Feed the arm's recorded runs to hls3 in order. Before each run: hls3's memory prediction for its configuration
    (S-1), and with `ask` the proposals hls3 would make on the recorded incumbent (passed to on_step(i, proposals)).
    `on_run(i, eng, st, p, cand, row)`: called with the run's candidate in place, before hls3 observes its outcome. The
    hls3 arms' receipts carry their launch timing (stamped). `adopt` (with `ask`): a recorded run whose configuration one
    of hls3's proposals makes is observed as that proposal (its bundle, probe and need bookkeeping), and a queued stall
    step none of whose proposals ran is taken as served (the replay has no outcome for it) before hls3 is asked again;
    the row's `adopted` = that proposal's origin, None when hls3 would not have proposed the run. `follow`: the incumbent
    is the recorded one (a run on a later candidate than the incumbent: that candidate had become it), not the replay's
    own 2 sigma_pair rule. `merge_probe`: a passed probe whose full run follows is not observed on its own - the engine
    observes one outcome per proposal (the round-3 replays; the older ones observe both rows). `bases`: candidate -> the
    base it ran on where the record's parent is another (a memory bundle the session recorded under its spender's
    candidate). `gains`: candidate -> the search's own (robust) log gain where the screens' mean step times read
    otherwise; `incumbents`: the candidates the record accepted (only they become the incumbent). Returns one row per
    recorded run (rescores skipped)."""
    eng, st, arm = open_loop(name, declarations=declarations, off=off, with_proposer=ask, r1=r1, r2=r2, r3=r3)
    cands = {c['id']: c for c in arm['candidates']}
    sp = eng.sp
    tm = timing(arm)
    eng.s.candidates['c0'].receipt['evidence'].update(stamped(tm.get(('c0', 'reference'))))
    out_rows = []
    rows_in = [r for r in arm['screens'] if r['fidelity'] in ('full', 'probe')]
    for i, row in enumerate(arm['screens']):
        if row['fidelity'] not in ('full', 'probe'):
            continue
        cid = row['candidate_id']
        c = cands[cid]
        k = rows_in.index(row)
        if merge_probe and row['fidelity'] == 'probe' and row['valid'] and k + 1 < len(rows_in) and \
                rows_in[k + 1]['candidate_id'] == cid and rows_in[k + 1]['fidelity'] == 'full':
            continue                                    # (its full run's outcome is the proposal's)
        base = (bases or {}).get(cid) or c['parent'] or 'c0'
        if base not in eng.s.candidates:
            base = 'c0'
        if follow and base != eng._inc and base in eng.s.candidates and int(base[1:]) > int(eng._inc[1:]):
            eng._inc = base                             # (the recorded arm had accepted it)
        props = st.propose(eng, 16) if ask else None
        bcfg = eng.s.candidates[base].config
        change = {k: v for k, v in c['config'].items() if bcfg.get(k) != v}
        patched = bool(c['overlay_files']) and (cands.get(base) or {}).get('overlay_digest') != c.get('overlay_digest')
        patch = {PATCH_FILE: f'# {c["rationale"] or cid}\n' + ''.join(f'{ln}\n' for ln in c.get('patch_lines') or [])} \
            if patched else None
        adopted = q = None
        if adopt and props is not None:
            q = match(eng, props, base, c, patch)
            if q is None and st.queue:                  # its stall step ran elsewhere: hls3 asked again
                st.queue, st.idle = [], 0
                props = st.propose(eng, 16)
                q = match(eng, props, base, c, patch)
            adopted = None if q is None else str(q.origin)
        if on_step is not None:
            on_step(len(out_rows), eng, st, props)
        same = [k for k, h in st.ledger.items() if h['base'] == base and h.get('change') == change
                and bool(h.get('patch')) == bool(patch) and h['status'] == 'open']
        hyp = next((k for k in same if patch and st.ledger[k]['mechanism'] == c.get('rationale')), same[0] if same else None)
        p = Proposal(change=change, base=base, patch=patch, hypothesis=hyp, mechanism=c['rationale'] or cid,
                     fidelity=(row['fidelity'],) if row['fidelity'] == 'probe' else ('full',))
        if q is not None:
            p = q
        pcfg = st._mcfg(eng, base, change, st._patch_ideas(p) if patch else ())
        verdict, why, src = st._predict(eng, pcfg)
        if cid not in eng.s.candidates:
            eng.s.candidates[cid] = SimpleNamespace(id=cid, parent=base, config=dict(c['config']),
                                                    overlay=dict(patch or eng.s.candidates[base].overlay or {}),
                                                    verdict=None, status='proposed', results={}, receipt=None)
        cand = eng.s.candidates[cid]
        status, reasons, failed = outcome_of(arm, row)
        if c.get('events') is not None:
            cand.receipt = {'evidence': {'events': c['events'], 'timed_out': c.get('timed_out'),
                                         'steps_completed': c.get('steps_completed'),
                                         'memory_peak_gib': c.get('memory_peak_gib')}}
        launch = tm.get((cid, 'screen')) or tm.get((cid, 'probe'))
        if launch is not None:
            cand.receipt = cand.receipt or {'evidence': {}}
            cand.receipt['evidence'].update(stamped(launch))
        gain = None
        peak = {'peak_reserved_gib': c['peak_reserved_gib']} if c.get('peak_reserved_gib') is not None else {}
        if status == 'screened':                        # (a recorded peak as the verifier reported it)
            cand.verdict, cand.status = {'valid': True, 'step_time': row['step_time'], **peak}, 'screened'
            gain = math.log(eng.t(base) / row['step_time']) if eng.t(base) and row['step_time'] else None
            gain = (gains or {}).get(cid, gain)
        elif status == 'aborted':
            cand.status = 'aborted'
        elif status == 'invalid':
            cand.verdict, cand.status = {'valid': False, 'reasons': reasons, **peak}, 'screened'
        elif status == 'infeasible':
            cand.verdict, cand.status = {'valid': False, 'reasons': reasons, **peak}, 'infeasible'
        out = Outcome(cid, status, status in ('screened', 'probed'), row.get('step_time'), gain,
                      'probe' if row['fidelity'] == 'probe' else 'full', row.get('gpu_hours') or 0.0, list(reasons),
                      list(failed))
        if status == 'screened' and gain is not None:   # the recorded incumbent rule: a gain beyond 2 sigma_pair
            t_inc = eng.t(eng._inc)
            if t_inc and math.log(t_inc / row['step_time']) > 2 * sp and (incumbents is None or cid in incumbents):
                eng._inc = cid
        if row['fidelity'] == 'full':
            eng.s.ledger.used['screens'] += 1
        if on_run is not None:
            on_run(len(out_rows), eng, st, p, cand, row)
        st.observe(eng, p, out)
        kind = st.failures[-1]['class'] if st.failures and st.failures[-1]['candidate'] == cid else \
            ('slower' if status == 'aborted' else 'ok')
        out_rows.append({'cid': cid, 'base': base, 'fidelity': row['fidelity'], 'status': status, 'kind': kind,
                         'prediction': verdict, 'why': why, 'p_feasible': recorded_p(arm, base, change),
                         'change': change, 'patch': bool(patch), 'adopted': adopted, 'origin': c.get('origin')})
    return out_rows


def match(eng, props: list, base: str, c: dict, patch) -> object:
    """The proposal among `props` that runs the recorded candidate `c` on `base`: its configuration, with code when the
    run had a patch (the recorded mechanism preferred among patches)."""
    same = [q for q in props or [] if (q.base or base) == base and config_of(eng, q, base) == c['config']
            and bool(q.patch) == bool(patch)]
    return next((q for q in same if patch and q.mechanism == c.get('rationale')), same[0] if same else None)


def fresh(eng, st) -> tuple[list, object]:
    """(proposals, the copy that made them): hls3's proposals at this point of an open-loop replay - its queued stall
    step and what it proposes besides (a copy of the strategy, its proposer's used calls copied, with the queue served
    and no new stall step). The replay never runs hls3's own proposals, so a queued step would otherwise mask every
    later proposal. The strategy itself is untouched."""
    import copy
    prop = st.proposer
    veto = getattr(eng.s, 'start_failure_veto', None)
    st.proposer = None
    try:
        st2 = copy.deepcopy(st)
    finally:
        st.proposer = prop
    p2 = copy.copy(prop)
    p2.used = set(prop.used)
    st2.proposer, st2.queue, st2.idle = p2, [], 0
    try:
        return st2.propose(eng, 16) + list(st.queue), st2
    finally:
        eng.s.start_failure_veto = veto


def config_of(eng, q, base: str) -> dict:
    """The configuration a proposal runs (its change applied to its base, as the session applies it)."""
    return {**eng.s.candidates[q.base or base].config, **(q.change or {})}


PROBE_BELOW = search_params(0.9)['probe_below']        # hls's probe rule at the arms' capability (0.9)


def would(row: dict) -> str:
    """What hls3 would have done with this recorded run: 'deferred' (not run), 'probed' (a kept 2-update probe first: a
    failure at the start costs no screen), or 'screened' (a full run, as recorded)."""
    if row['prediction'] == 'infeasible':
        return 'deferred'
    if row['fidelity'] == 'probe' or row['prediction'] == 'unsure' or (
            row['prediction'] != 'feasible' and row['p_feasible'] is not None and row['p_feasible'] < PROBE_BELOW):
        return 'probed'
    return 'screened'


def failed_screen(row: dict) -> bool:
    """A full screen that failed at its start (out of memory, crash, compiler, collective, the launch timeout): what a
    2-update probe would have caught without a screen."""
    return row['fidelity'] == 'full' and row['kind'] in ('oom', 'crash', 'compiler', 'collective', 'start_cap')
