"""The experience layer in a project run: the plan for this job after the reference run (the store, the fingerprint,
coverage, seeds, the start composite and, with an LLM, the transfer judgment), its summary for result.json, and the
write-back of the run as an infrarsi-exp/1 pack (INFRARSI_HOME/experience/runs/<run id>.jsonl) that the next run's
store loads: its environment, one observation per valid full-fidelity screen, and the scored judgment."""
from __future__ import annotations

import datetime as _dt
import math
import os
import re
from collections import Counter
from pathlib import Path

from . import transfer
from .accounting import READS, estimate, mfu
from .pack import SCHEMA, validate, write_lines
from .registry import state_owners
from .store import UNKNOWN, Store, fingerprint, home, labels, plan as make_plan, states_of


def selection(value):
    """search.experience / --experience: 'auto' (every pack; the default), 'off', or pack-id patterns (a list, or
    comma-separated) -> None (all), False (off) or the patterns."""
    if value in (None, True, 'auto', 'on', 'all'):
        return None
    if value in (False, 'off', 'none', ''):
        return False
    return [s.strip() for s in (value if isinstance(value, list) else str(value).split(',')) if s.strip()]


def plan_for(project, receipt: dict, *, select=None, driver=None, log=print, ablate=None) -> dict | None:
    """The store's plan for this job (store.plan), with a `transfer` hook (HLS calls it once, charged to the search
    session) when an LLM is configured and the data leave knobs uncovered; None when experience is off. `ablate`
    (default search.ablate): 'accounting' turns the closed-form accounting off."""
    sel = selection(select if select is not None else project.raw['search'].get('experience'))
    if sel is False:
        return None
    store = Store.load(include=sel)
    fp = fingerprint(project, receipt)
    ref = project.reference_config()
    off = set(project.raw['search'].get('ablate') or [] if ablate is None else ablate)
    pl = make_plan(store, fp, project.knobs, ref, accounting='accounting' not in off)
    if driver is not None and transfer.needed(pl):
        past = [j for j in store.judgments if j.get('target_gpu') == fp.get('gpu')]

        def run_transfer(session):
            pl['judgment'] = transfer.judge(driver, pl, project.knobs, ref, session=session, judgments=past)
        pl['transfer'] = run_transfer
    lv = Counter(c['level'] for c in pl['coverage'].values())
    unknown = [k for k, c in pl['coverage'].items() if not c.get('mechanism')]
    members = (pl['composite'] or {}).get('members')   # U-1-13: no "start composite None" when there is none
    log(f'[infrarsi] experience: {len(store.packs)} packs; evidence for the {sum(lv.values())} knob moves of this project: '
        f'covered {lv["covered"]}, null {lv["null"]}, partial {lv["partial"]}, none {lv["none"]}'
        + (f'; start composite {members}' if members else '')
        + (f'; no mechanism known for {", ".join(_hint(k) for k in unknown)} (knobs.<name>.mechanism / states)'
           if unknown else ''))
    return pl


def _hint(knob: str) -> str:
    owners = state_owners(knob)
    return f'{knob} (a state of {", ".join(owners)}?)' if owners else knob


def accounting_for(project, receipt: dict | None):
    """The closed-form accounting of this job (experience.accounting) as view(config, measured step time) -> the
    proposer's and the doctor's numbers (memory per GPU with its terms against the card, the ideal update time, MFU),
    None for a configuration with a state it reads unknown; None without the model shape (manifest `model:`)."""
    fp = fingerprint(project, receipt)
    labs = labels(project.knobs, fp.get('framework'))

    def view(config: dict, step_time: float | None = None) -> dict | None:
        st = states_of(fp, labs, config)
        est = None if any(st.get(k) == UNKNOWN for k in READS) else estimate(fp, st)
        if est is None:
            return None
        t = est['mem_terms']
        return {'basis': f'closed-form, per GPU, +-25 %; ideal update = model FLOPs at the GPUs\' {est["tflops_basis"]} '
                         'matrix throughput (no communication, no recompute)',
                'memory_gib': est['mem_gib'], 'capacity_gib': est['capacity_gib'], 'margin_gib': est['fit_margin_gib'],
                'terms_gib': {'weights_grads_optimizer': t['states_gib'], 'activations': t['activations_gib'],
                              'logits': t['logits_gib'], 'cuda_nccl_outside_allocator': t['non_torch_gib'],
                              'allocator_slack': t['slack_gib']},
                'ideal_update_s': est['ideal_step_s'], 'tflops_basis': est['tflops_basis'],
                **({'dp_sync_lower_bound_s': est['dp_sync_s'], 'bandwidth_gbs': est['bandwidth_gbs']}
                   if est.get('dp_sync_s') else {}),
                'measured_update_s': step_time, 'mfu': mfu(est, step_time),
                'notes': est['notes']}
    return view if view(project.reference_config()) is not None else None


def _hls_part(trace: dict) -> dict:
    st = (trace.get('extra') or {}).get('strategy')
    parts = [st] + list(st.get('parts') or []) if isinstance(st, dict) else []
    return next((x for x in parts if isinstance(x, dict) and x.get('experience')), {})


def _learned(trace: dict) -> dict | None:
    """The `learned` section (F9b, experience.learned.section) of the run's strategy or of a part of it - any strategy
    that writes one."""
    st = (trace.get('extra') or {}).get('strategy')
    parts = [st] + list(st.get('parts') or []) if isinstance(st, dict) else []
    return next((x['learned'] for x in parts if isinstance(x, dict) and isinstance(x.get('learned'), dict)
                 and x['learned']), None)


def _patches(trace: dict) -> dict | None:
    """The `patches` section (search/finalist_base S2: the arm's proven and inert code patches, each with its mechanism,
    canonical site, gain, inert reason and scanner lines) of the run's strategy or of a part of it - only a strategy that
    writes one (no other strategy does: their packs are as before)."""
    st = (trace.get('extra') or {}).get('strategy')
    parts = [st] + list(st.get('parts') or []) if isinstance(st, dict) else []
    return next((x['patches'] for x in parts if isinstance(x, dict) and isinstance(x.get('patches'), dict)
                 and x['patches']), None)


def summary(pl: dict, trace: dict, sigma_pair: float | None) -> dict:
    """What result.json and the report keep: packs, coverage, the composite and its fate, the seeded entries' outcomes,
    the full screens spent on experience, and the transfer judgment with its scores."""
    exp = _hls_part(trace).get('experience') or {}
    out = {'packs': pl['packs'], 'skipped': pl['skipped'], 'noise_pct': pl['noise_pct'],
           'levels': dict(Counter(c['level'] for c in pl['coverage'].values())),
           'coverage': {n: {k: c.get(k) for k in ('mechanism', 'class', 'level', 'value', 'gain', 'n', 'evidence', 'within_pct')
                            if c.get(k) is not None} for n, c in pl['coverage'].items()},
           'composite': pl['composite'] and {k: pl['composite'].get(k) for k in ('members', 'change', 'gain', 'warnings', 'source')},
           'nearest': pl['nearest'], 'search': exp}
    j = pl.get('judgment')
    if j:
        ledger = (trace.get('extra') or {}).get('ledger') or []
        raw = [dict(h, measured=None if h.get('measured_gain_pct') is None else h['measured_gain_pct'] / 100)
               for h in ledger]                         # the view's measured_gain_pct is 100 x the log gain
        out['judgment'] = dict(j, scores=transfer.score(j, raw, sigma_pair or 0.0, exp.get('composite_fate')))
    return out


def doctor_view(pl: dict) -> dict:
    """The doctor's lines: coverage per knob with its nearest evidence, the nearest environments and their differences,
    the start composite with its source gains, the constraints in scope, InfraRSI's measured noise on this GPU class."""
    return {'packs': pl['packs'], 'skipped': pl['skipped'], 'noise_pct': pl['noise_pct'],
            'levels': dict(Counter(c['level'] for c in pl['coverage'].values())),
            'coverage': {n: {k: c.get(k) for k in ('mechanism', 'level', 'value', 'gain', 'n', 'evidence', 'within_pct', 'nearest')
                             if c.get(k) is not None} for n, c in pl['coverage'].items()},
            'nearest': pl['nearest'], 'composite': pl['composite'], 'constraints': pl['constraints'],
            'seed_warnings': {s['knob']: s['warnings'] for s in pl['seeds'] if s.get('warnings')}, 'missing': pl['missing']}


def _status(screen: dict) -> str:
    text = ' '.join(screen.get('reasons') or []).lower()
    if screen.get('valid'):
        return 'ok'
    if re.search(r'out[ -]of[ -]memory|outofmemory|\boom\b', text):   # the verifier writes "1 out-of-memory line(s)"
        return 'oom'
    return 'invalid' if 'loss' in text or 'tolerance' in text or 'workload' in text else 'crash'


def unwritable(fp: dict) -> str | None:
    """Why a run cannot be written back as experience: the facts its fingerprint lacks (None when it can)."""
    miss = [k for k, v in (('model.params_b', (fp.get('model') or {}).get('params_b')),
                           ('workload.seq_len', (fp.get('workload') or {}).get('seq_len'))) if not v]
    return f'the manifest has no {" and no ".join(miss)}' if miss else None


def write_back(project, pl: dict, trace: dict, promo: dict | None, run_id: str, sigma_pair: float | None,
               exp_summary: dict | None = None) -> Path | None:
    """This run as a pack in INFRARSI_HOME/experience/runs/: the environment (the fingerprint, with this run's measured
    noise), an observation per candidate run at full length (the geometric mean of its valid full screens and confirmation
    re-runs, the reference's own run included; mechanism states, unlabelled knobs as x- ids; the allocator peak its log
    reported as x-torch-reserved-gib (not nvidia-smi's mem_peak_gib); a
    candidate promotion turned down - speed, evaluation, resume or tests - is left out), an `oom` observation per candidate whose 2-update
    probe ran out of memory, and the transfer judgment with its scores. The store derives the single-mechanism effects when it loads the pack. None when
    the fingerprint lacks the model size or the sequence length (manifest `model.params_b`, `workload.seq_len`)."""
    fp = pl['fingerprint']
    if unwritable(fp):
        return None
    labs = labels(project.knobs, fp.get('framework'))
    cands = {c['id']: c for c in trace.get('candidates') or []}
    cfgs: dict[str, dict] = {}

    def config(cid):                                    # the parent chain of changes from the reference
        if cid not in cfgs:
            c = cands[cid]
            cfg = dict(project.reference_config() if c.get('parent') not in cands else config(c['parent']))
            for k, ab in (c.get('change') or {}).items():
                cfg[k] = ab[1] if isinstance(ab, list) and len(ab) == 2 else ab
            cfgs[cid] = cfg
        return cfgs[cid]

    # promotion turned it down: its speed did not reproduce, or its evaluation, resume or tests failed
    bad = {t['candidate'] for t in (promo or {}).get('tried') or []
           if any(i.get('status') == 'fail' for i in t.get('items') or [])}
    runs: dict[str, list] = {}
    for sc in trace.get('screens') or []:
        if sc.get('fidelity') in ('full', 'rescore') and not sc.get('aborted') and sc.get('candidate_id') in cands:
            runs.setdefault(sc['candidate_id'], []).append(sc)
    for sc in trace.get('screens') or []:              # a 2-update probe out of memory: memory evidence (fits() uses
        cid = sc.get('candidate_id')                   # OOM rows); a crashed probe says nothing about memory
        if sc.get('fidelity') == 'probe' and cid in cands and cid not in runs and _status(sc) == 'oom':
            runs[cid] = [sc]
    if 'c0' in cands and cands['c0'].get('step_time'):    # the reference run itself is not a screen
        runs.setdefault('c0', []).insert(0, {'valid': True, 'step_time': cands['c0']['step_time']})
    stamp = _dt.datetime.now(_dt.timezone.utc)
    w = project.window
    lines = [{'kind': 'pack', 'id': 'pack', 'schema': SCHEMA, 'pack_id': f'run-{run_id}', 'title': f'InfraRSI run {run_id}',
              'contributors': ['infrarsi'], 'license': 'unspecified', 'created': stamp.date().isoformat(), 'tier': 'verified',
              'summary': ''},                       # below: what was written
             {'kind': 'env', 'id': 'env', 'gpu': fp.get('gpu'), 'gpus_per_node': fp.get('gpus_per_node'), 'nodes': fp.get('nodes'),
              **({'memory_gib': fp['memory_gib']} if fp.get('memory_gib') else {}),   # an OOM here counts for this card
              # a manifest without `framework` (a hand-written one): the pack's registry wants one; `other` claims nothing
              'framework': fp.get('framework') or 'other', 'versions': fp.get('versions') or {},
              **({'interconnect': fp['interconnect']} if fp.get('interconnect') else {}),
              'model': {'name': project.name, **(fp.get('model') or {})}, 'workload': fp.get('workload'),
              **({'noise_pct': round(100 * sigma_pair, 4)} if sigma_pair else {})}]
    left_out = 0
    for cid in [c for c in cands if c in runs]:        # in the trace's order: the reference first
        scs = runs[cid]
        ok = [s['step_time'] for s in scs if s.get('valid') and s.get('step_time')]
        status = 'ok' if ok else _status(scs[0])
        if cid in bad and status == 'ok' or any(labs.get(n) is not None and v is not None and labs[n].state(v) is None
                                                for n, v in config(cid).items()):
            left_out += 1                               # turned down at promotion, or a knob value with no registry state
            continue
        c = cands[cid]
        cfg = {k: v for k, v in states_of(fp, labs, config(cid), ok=status == 'ok').items() if v != UNKNOWN}
        if c.get('overlay_digest'):
            cfg['x-patch'] = str(c['overlay_digest'])[:12]
        row = {'kind': 'observation', 'id': cid, 'env': 'env', 'method': 'infrarsi',
               **({} if scs[0].get('fidelity') == 'probe' else {'window_steps': w[1] - w[0] + 1}),
               'config': cfg, 'status': status}
        if ok:
            row['step_time_s'] = round(math.exp(sum(math.log(t) for t in ok) / len(ok)), 6)
            row['n'] = len(ok)
            if type(c.get('peak_reserved_gib')) in (int, float):
                # the log's peak is the allocator's (torch reserved), not nvidia-smi's (CUDA context and NCCL buffers
                # on top), which is what the packs' mem_peak_gib and the memory look-up's headroom mean
                row['x-torch-reserved-gib'] = round(c['peak_reserved_gib'], 2)
        lines.append(row)
    obs = [x for x in lines if x.get('kind') == 'observation']
    n_oom = sum(1 for x in obs if 'window_steps' not in x)
    lines[0]['summary'] = (f'project {project.name}: {len(obs) - n_oom} configurations at full fidelity'
                           + (f', {n_oom} out of memory at a probe' if n_oom else '')
                           + (f'; {left_out} left out (turned down at promotion, or a knob value with no mechanism '
                              'state)' if left_out else ''))
    learned = _learned(trace)                         # F9b (D4): the start failures it learned, as constraints a
    if learned:                                         # later arm's store reads back
        from .learned import constraints
        lines += constraints(learned, fp, labs, run_id)
    patches = _patches(trace)                         # finalists S2: the arm's proven / inert patches on the pack's header
    if patches:                                         # (a field, not a line kind: every tree's validate still reads the
        lines[0]['patches'] = patches                   # pack; a later finalist of another job reads the section back)
    j = (exp_summary or {}).get('judgment')
    if j and j.get('assessment') is not None:
        lines.append({'kind': 'judgment', 'id': 'transfer', 'env': 'env', 'model': j.get('model'), 'target_gpu': fp.get('gpu'),
                      'source_envs': [n['env'] for n in pl.get('nearest') or []],
                      'source_gpus': sorted({n.get('gpu') for n in pl.get('nearest') or [] if n.get('gpu')}),
                      'scores': j.get('scores'), 'assessment': j['assessment'], 'start_config': j.get('start_config')})
    errors, _ = validate(lines)
    if errors:
        pl['write_back_errors'] = errors[:6]            # the flow says why nothing was written
        return None
    path = home() / 'experience' / 'runs' / f'{run_id}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix('.part')                   # not *.jsonl*: the store never loads it
    try:                                               # published whole or not at all (U-7-3: Ctrl-C while writing)
        write_lines(part, lines)
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)
    return path
