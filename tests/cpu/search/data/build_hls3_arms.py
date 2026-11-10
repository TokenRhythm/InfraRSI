"""Builds tests/cpu/search/data/hls3_arms.json: the recorded arms the hls3 replay tests read (run by hand, never by a test
or at run time): per arm the manifest facts (knob choices, card, GPUs, parameters, invariants), the reference config,
every candidate (config rebuilt from the reference and each recorded change, status, verdict, peak, the failure text),
the screens in order (fidelity, cost), and for the LLM arms the model's hypotheses per call (with the incumbent it was
asked on), the code review's patches and the transfer judgment. For the hls3 arms (`-h3`, 2026-10-29/30): every launch
of the arm from its attempt receipts (purpose, wall, the time to the first update, the first per-update intervals, the
updates done, the error line) - the start cap and the probe speed gate of FABLE_HLS3_FIXES_R1 are replayed from them -
and the proposer's declared memory and code site; an arm whose search did not finish (no result.json: Q27U-h3 in flight,
Q27H2-h3 void) is rebuilt from its event log.

    python build_hls3_arms.py <trials_tb dir> <out json>
"""
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

ARMS = ['G4U-code-env-opus', 'G4U-code-env-opus-h2', 'G4U-rnd', 'Q27H2-code-env-opus', 'Q27H2-rnd', 'M35H2-code-env-opus',
        'M35H2-code-env-opus-h2', 'M35H2-rnd', 'Q9U-code-env-opus', 'Q9U-code-env-opus-h2', 'Q9U-rnd']
H3 = {'Q9H-code-env-opus-h3': 'Q9H-code-env-opus-h3', 'Q9U-code-env-opus-h3': 'Q9U-code-env-opus-h3',
      'G4U-code-env-opus-h3': 'G4U-code-env-opus-h3', 'M35H2-code-env-opus-h3': 'M35H2-code-env-opus-h3',
      'Q27U-code-env-opus-h3': 'Q27U-code-env-opus-h3',
      'Q27H2-code-env-opus-h3': '_void_nm/Q27H2-code-env-opus-h3_winh0diskfull_0739'}    # name -> its directory


def parse_json(text: str):
    t = text.strip()
    m = re.search(r'```(?:json)?\s*(.*?)```', t, re.S)
    if m:
        t = m.group(1)
    i = t.find('{')
    try:
        return json.loads(t[i:]) if i >= 0 else None
    except ValueError:
        return None


def last_result(d: Path):
    tries = sorted(d.glob('retry_*'), key=lambda p: int(p.name.split('_')[1]))
    for p in [*reversed(tries), d]:
        f = p / 'result.txt'
        if f.is_file():
            out = parse_json(f.read_text(encoding='utf-8', errors='replace'))
            if out is not None:
                return out
    return None


def calls(run: Path, h3: bool = False) -> list:
    out = []
    for d in sorted((run / 'llm_calls').glob('*_*')):
        if not d.is_dir():
            continue
        role = d.name.rsplit('_', 1)[0]
        rec = json.loads((d / 'receipt.json').read_text(encoding='utf-8')) if (d / 'receipt.json').is_file() else {}
        req = json.loads((d / 'request.json').read_text(encoding='utf-8')) if (d / 'request.json').is_file() else {}
        try:
            prompt = json.loads(req.get('prompt') or '{}')
        except ValueError:
            prompt = {}
        res = last_result(d) or {}
        row = {'role': role, 'end_utc': rec.get('end_utc'), 'base': (prompt.get('incumbent') or {}).get('id')}
        if role == 'hypotheses':
            row['hypotheses'] = [{'mechanism': str(h.get('mechanism'))[:200], 'change': h.get('change'),
                                  'patch': bool(h.get('patch') or h.get('edits')),
                                  'predicted_gain_pct': h.get('predicted_gain_pct'), 'uncertainty_pct': h.get('uncertainty_pct'),
                                  'p_feasible': h.get('p_feasible'), 'rationale': str(h.get('rationale') or '')[:400],
                                  **({k: h[k] for k in ('memory', 'site', 'variance', 'change_class') if h.get(k) is not None}
                                     if h3 else {})}
                                 for h in res.get('hypotheses') or [] if isinstance(h, dict)]
        elif role == 'code_review':
            row['verdict'] = res.get('verdict')
            row['findings'] = [{'file': f.get('file'), 'issue': str(f.get('issue'))[:300], 'knob': f.get('knob'),
                                'patch': bool(f.get('edits') or f.get('patch')),
                                'predicted_gain_pct': f.get('predicted_gain_pct')} for f in res.get('findings') or []]
        elif role == 'transfer':
            row['assessment'] = [{k: a.get(k) for k in ('knob', 'change', 'sign', 'p_feasible', 'expected_gain_pct', 'why')}
                                 for a in res.get('assessment') or []]
            row['start_config'] = res.get('start_config')
            row['memory_note'] = res.get('memory_note')
        out.append(row)
    return sorted(out, key=lambda r: r.get('end_utc') or '')


def _epoch(utc) -> float | None:
    try:
        return datetime.datetime.fromisoformat(str(utc).replace('Z', '+00:00')).timestamp()
    except ValueError:
        return None


def attempts(run: Path) -> list:
    """Every launch of the run, in start order: candidate, purpose, wall, the time from the launch to the end of the first
    update (the runner's clock to the job's first stamp; t_first_wall: the wall less the stamped span, free of clock
    skew), the first per-update intervals, updates done, timed out, exit code, the error line, the peak."""
    out = []
    for rec in (run / 'attempts').glob('*/*/receipt.json'):
        r = json.loads(rec.read_text(encoding='utf-8'))
        ev = r.get('evidence') if isinstance(r.get('evidence'), dict) else {}
        ps = [x for x in ev.get('per_step') or [] if isinstance(x.get('t'), (int, float))]
        start, wall = _epoch(r.get('utc_start')), ev.get('wall_s')
        events = ev.get('events') or {}
        out.append({'cid': r.get('candidate_id'), 'purpose': r.get('purpose'), 'utc_start': r.get('utc_start'),
                    'wall_s': round(wall, 2) if isinstance(wall, (int, float)) else None,
                    'gpu_s': (r.get('cost') or {}).get('gpu_seconds'),
                    't_first': round(ps[0]['t'] - start, 2) if ps and start else None,
                    't_first_wall': round(wall - (ps[-1]['t'] - ps[0]['t']), 2) if ps and isinstance(wall, (int, float))
                    else None,
                    'intervals': [round(y['t'] - x['t'], 4) for x, y in zip(ps, ps[1:])][:4],
                    'steps_completed': ev.get('steps_completed'), 'stop_after': ev.get('stop_after'),
                    'timed_out': ev.get('timed_out'), 'step_time': ev.get('step_time'), 'exit_code': ev.get('exit_code'),
                    'valid': r.get('valid'), 'error': str(events.get('error') or events.get('fatal') or '')[:300] or None,
                    'oom': events.get('oom') or None, 'peak': ev.get('memory_peak_gib'),
                    'perf_notes': list(events.get('perf_notes') or []) or None})
    return sorted(out, key=lambda a: a['utc_start'] or '')


def overlay_code(run: Path, cid: str, files: list) -> tuple:
    """(digest, the lines it adds to the reference's files): an hls3 arm candidate's overlay as its first launch ran it
    (the replay's patch text: F8 reads the code lines two crashing patches share)."""
    att = sorted((run / 'attempts').glob(f'*/{cid}-*/tree'))
    ref = sorted((run / 'attempts').glob('*/c0-reference-*/tree'))
    if not files or not att:
        return None, []
    h, added = hashlib.sha1(), []
    for f in sorted(files):
        a = att[0] / f
        if not a.is_file():
            continue
        text = a.read_text(encoding='utf-8', errors='replace')
        h.update(f.encode('utf-8') + b'\0' + text.encode('utf-8', 'replace'))
        have = set((ref[0] / f).read_text(encoding='utf-8', errors='replace').splitlines()) if ref and (ref[0] / f).is_file() \
            else set()
        added += [ln[:200] for ln in text.splitlines() if ln not in have][:120]
    return h.hexdigest()[:16], added


def with_notes(run: Path, cands: list, code: bool) -> list:
    """Each candidate's job-log notes (its final receipt: search/receipts/<id>.json, else its last launch) and, with
    `code`, the lines its overlay adds (`patch_code`)."""
    for c in cands:
        rec = run / 'search' / 'receipts' / f'{c["id"]}.json'
        r = json.loads(rec.read_text(encoding='utf-8')) if rec.is_file() else None
        if r is None:
            last = sorted((run / 'attempts').glob(f'*/{c["id"]}-*/receipt.json'))
            r = json.loads(last[-1].read_text(encoding='utf-8')) if last else None
        notes = ((((r or {}).get('evidence') or {}).get('events') or {}).get('perf_notes')) if r else None
        if notes:
            c['perf_notes'] = list(notes)
        if code and c.get('overlay_files') and 'patch_lines' not in c:
            c['patch_code'] = overlay_code(run, c['id'], c['overlay_files'])[1]
    return cands


def with_code(run: Path, cands: list) -> list:
    for c in cands:
        if c.get('overlay_files'):
            dg, lines = overlay_code(run, c['id'], c['overlay_files'])
            c['patch_lines'] = lines
            c['overlay_digest'] = c.get('overlay_digest') or dg
    return cands


def events_arm(run: Path, man: dict) -> dict:
    """An arm whose search did not finish (no result.json): reference, candidates and runs from its event log."""
    rows = [json.loads(x) for x in (run / 'search' / 'events.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    ref = next(json.loads(q.read_text(encoding='utf-8')) for q in (run / 'attempts').glob('*/c0-reference-*/receipt.json'))
    cfgs, cands, screens = {'c0': dict(ref['config'])}, {}, []
    for e in rows:
        pl = e['payload']
        if e['kind'] == 'candidate' and pl['id'] != 'c0':
            cid = pl['id']
            cfg = dict(cfgs.get(pl.get('parent') or 'c0') or cfgs['c0'])
            for k, v in (pl.get('change') or {}).items():
                cfg[k] = v[1] if isinstance(v, list) and len(v) == 2 else v
            cfgs[cid] = cfg
            cands[cid] = {'id': cid, 'parent': pl.get('parent'), 'origin': pl.get('origin'), 'status': 'proposed',
                          'valid': None, 'step_time': None, 'reasons': [], 'failed_checks': None,
                          'overlay_files': pl.get('overlay_files') or [], 'overlay_digest': None, 'rationale': '',
                          'peak_reserved_gib': None}
        elif e['kind'] in ('screen', 'trial'):
            c = cands.get(pl['candidate_id'])
            fid = pl.get('fidelity') or 'full'
            screens.append({'index': pl.get('index'), 'candidate_id': pl['candidate_id'], 'fidelity': fid,
                            'valid': pl.get('valid'), 'aborted': pl.get('aborted'), 'step_time': pl.get('step_time'),
                            'gpu_hours': pl.get('gpu_hours')})
            if c is not None:
                c.update(valid=pl.get('valid'), step_time=pl.get('step_time'),
                         reasons=[str(x)[:700] for x in (pl.get('reasons') or [])[:3]],
                         status=('aborted' if pl.get('aborted') else 'screened' if fid == 'full' else
                                 'probed' if pl.get('valid') else 'infeasible'))
    st = ref.get('step_time')
    if isinstance(st, dict):
        st = st.get('mean_window_max_rank_s')
    st = st or (ref.get('evidence') or {}).get('step_time')
    tr = next((c for c in calls(run) if c['role'] == 'transfer'), None) if (run / 'llm_calls').is_dir() else None
    start = (tr or {}).get('start_config') or {}
    return {'reference_step_time': st, 'cands': list(cands.values()), 'cfgs': cfgs, 'screens': screens,
            'composite': {'members': sorted(start), 'change': start, 'gain': None} if len(start) >= 2 else None,
            'budget': ((man.get('search') or {}).get('budget'))}


def arm(root: Path, name: str, rel: str | None = None) -> dict:
    d = root / (rel or name)
    found = list(d.glob('*__result.json'))
    if not found:
        return arm_unfinished(d, name)
    res = json.loads(found[0].read_text(encoding='utf-8'))
    run = next((d / 'repo' / '.infrarsi' / 'runs').glob('*'))
    man = yaml.safe_load((d / 'repo' / 'infrarsi.yaml').read_text(encoding='utf-8'))
    knobs = {k: (v.get('choices') or ([False, True] if v.get('type') == 'bool' else [])) for k, v in man['knobs'].items()}
    cfgs = {'c0': dict(res['reference']['config'])}
    cands = []
    for c in res['candidates']:
        cid = c['id']
        rec = run / 'search' / 'receipts' / f'{cid}.json'
        receipt = json.loads(rec.read_text(encoding='utf-8')) if rec.is_file() else None
        if receipt and isinstance(receipt.get('config'), dict):
            cfgs[cid] = dict(receipt['config'])
        elif cid != 'c0':
            cfg = dict(cfgs.get(c.get('parent') or 'c0') or {})
            for k, v in (c.get('change') or {}).items():
                cfg[k] = v[1] if isinstance(v, list) and len(v) == 2 else v
            cfgs[cid] = cfg
        ev = (receipt or {}).get('evidence') or {}
        events = ev.get('events') or {}
        cands.append({'id': cid, 'parent': c.get('parent'), 'origin': c.get('origin'), 'status': c.get('status'),
                      'valid': c.get('valid'), 'step_time': c.get('step_time'), 'peak_reserved_gib': c.get('peak_reserved_gib'),
                      'reasons': [str(x)[:700] for x in (c.get('reasons') or [])[:3]],
                      'failed_checks': c.get('failed_checks'), 'overlay_files': c.get('overlay_files') or [],
                      'overlay_digest': c.get('overlay_digest'), 'rationale': str(c.get('rationale') or '')[:200],
                      'config': cfgs[cid],
                      'events': {k: events.get(k) for k in ('oom', 'collective', 'traceback', 'error', 'fatal')
                                 if events.get(k) not in (None, '', 0)} if receipt else None,
                      'timed_out': ev.get('timed_out') if receipt else None,
                      'steps_completed': ev.get('steps_completed') if receipt else None,
                      'memory_peak_gib': ev.get('memory_peak_gib') if receipt else None})
    trace = json.loads((run / 'search' / 'trace.json').read_text(encoding='utf-8'))
    ledger = [{k: h.get(k) for k in ('mechanism', 'change', 'base', 'predicted_gain_pct', 'status', 'measured_gain_pct',
                                     'candidate', 'bundle_candidate', 'source', 'why', 'memory', 'deferred', 'failure')
               if name in H3 and h.get(k) is not None or k in ('mechanism', 'change', 'base', 'predicted_gain_pct', 'status',
                                                               'measured_gain_pct', 'candidate', 'bundle_candidate', 'source')}
              for h in (trace.get('extra') or {}).get('ledger') or []]
    return {'arm': name, 'strategy': res.get('strategy'), 'run_id': res.get('run_id'),
            'hardware': man.get('hardware'), 'model': man.get('model'), 'workload': man.get('workload'),
            'framework': man.get('framework'), 'knobs': knobs, 'reference': cfgs['c0'],
            'reference_step_time': res['reference'].get('step_time'),
            'noise': (res.get('search') or {}).get('noise'), 'budget': (res.get('search') or {}).get('budget'),
            'candidates': with_notes(run, with_code(run, cands) if name in H3 else cands, code=name not in H3),
            'screens': [{k: s.get(k) for k in ('index', 'candidate_id', 'fidelity', 'valid', 'aborted', 'step_time', 'gpu_hours')}
                        for s in res.get('screens') or []],
            'calls': calls(run, name in H3) if (run / 'llm_calls').is_dir() else [], 'ledger': ledger,
            'experience_composite': ((res.get('experience') or {}).get('composite')),
            **({'attempts': attempts(run), 'finished': True} if name in H3 else {})}


def arm_unfinished(d: Path, name: str) -> dict:
    """An hls3 arm without result.json, from its event log, receipts, attempts and model calls (no noise, no ledger)."""
    run = next((d / 'repo' / '.infrarsi' / 'runs').glob('*'))
    man = yaml.safe_load((d / 'repo' / 'infrarsi.yaml').read_text(encoding='utf-8'))
    knobs = {k: (v.get('choices') or ([False, True] if v.get('type') == 'bool' else [])) for k, v in man['knobs'].items()}
    e = events_arm(run, man)
    cands = []
    for c in e['cands']:
        rec = run / 'search' / 'receipts' / f'{c["id"]}.json'
        receipt = json.loads(rec.read_text(encoding='utf-8')) if rec.is_file() else None
        ev = (receipt or {}).get('evidence') or {}
        events = ev.get('events') or {}
        cands.append({**c, 'config': e['cfgs'][c['id']],
                      'events': {k: events.get(k) for k in ('oom', 'collective', 'traceback', 'error', 'fatal')
                                 if events.get(k) not in (None, '', 0)} if receipt else None,
                      'timed_out': ev.get('timed_out') if receipt else None,
                      'steps_completed': ev.get('steps_completed') if receipt else None,
                      'memory_peak_gib': ev.get('memory_peak_gib') if receipt else None})
    c0 = {'id': 'c0', 'parent': None, 'origin': 'start', 'status': 'screened', 'valid': True,
          'step_time': e['reference_step_time'], 'reasons': [], 'failed_checks': None, 'overlay_files': [],
          'overlay_digest': None, 'rationale': '', 'peak_reserved_gib': None, 'config': e['cfgs']['c0'], 'events': None,
          'timed_out': None, 'steps_completed': None, 'memory_peak_gib': None}
    return {'arm': name, 'strategy': (man.get('search') or {}).get('strategy'), 'run_id': run.name,
            'hardware': man.get('hardware'), 'model': man.get('model'), 'workload': man.get('workload'),
            'framework': man.get('framework'), 'knobs': knobs, 'reference': e['cfgs']['c0'],
            'reference_step_time': e['reference_step_time'], 'noise': None, 'budget': e['budget'],
            'candidates': with_notes(run, [c0] + with_code(run, cands), code=False), 'screens': e['screens'],
            'calls': calls(run, True) if (run / 'llm_calls').is_dir() else [], 'ledger': [],
            'experience_composite': e['composite'], 'attempts': attempts(run), 'finished': False}


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    data = {name: arm(root, name) for name in ARMS}
    data.update({name: arm(root, name, rel) for name, rel in H3.items()})
    out.write_text(json.dumps(data, indent=0, sort_keys=True, default=str) + '\n', encoding='utf-8')
    print('wrote', out, out.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
