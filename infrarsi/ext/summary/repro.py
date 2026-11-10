"""Tables R1-R4 of design §5.4 (the reproduction-problem summary).

R1  every REPRO_ISSUES.md entry, mapped by R1_mapping.json onto the §5.4 vocabularies (stage / category / fidelity /
    status / hours), plus the mapping's extra rows (AMENDMENT_EXT_005's finding; design §5.4 items not in the ledger)
R2  environment reproduction verdicts: the §0.3 gates computed from REPRO_RESULTS.md values (each verified to occur
    there) and from signed receipts (ENV-B protocol defaults, ENV-C lines b / c); ENV-C in AMENDMENT_EXT_002's three-line
    wording; deviation ids from DEVIATION_IDS.md
R3  method-migration ledger: R2_R3_sources.json + the F1 results of X6's summary
R4  category x environment counts of R1, the top-3 rows by (estimated) hours, the lessons list (<= 10)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import f1 as F1
from . import render as R
from .cells import cell, na, ok, pending, val
from .records import Store, src

HERE = Path(__file__).resolve().parent
MAPPING = HERE / 'R1_mapping.json'
SOURCES = HERE / 'R2_R3_sources.json'
ENVS = ('ENV-A', 'ENV-B', 'ENV-C', 'F1', 'all')
_SPLIT = re.compile(r'(?<!\\)\|')


def parse_issues(text: str) -> list[dict]:
    """Rows of the REPRO_ISSUES.md table: {n, time, env, symptom, cause, action, fidelity} (escaped pipes kept)."""
    out, seen = [], set()
    for line in text.splitlines():
        if not line.lstrip().startswith('|'):
            continue
        cells = [c.strip() for c in _SPLIT.split(line.strip())[1:-1]]
        if len(cells) < 7 or not re.fullmatch(r'\d+', cells[0]):
            continue
        n = int(cells[0])
        if n in seen:
            raise ValueError(f'REPRO_ISSUES.md: issue #{n} appears twice')
        seen.add(n)
        out.append({'n': n, 'time': cells[1], 'env': cells[2], 'symptom': cells[3], 'cause': cells[4], 'action': cells[5],
                    'fidelity': cells[6]})
    return sorted(out, key=lambda r: r['n'])


def check_mapping(m: dict) -> list[str]:
    """Vocabulary problems of a mapping (empty = every value is in the §5.4 vocabularies)."""
    voc, bad = m.get('vocabularies') or {}, []
    rows = [(f'#{k}', v) for k, v in (m.get('issues') or {}).items()] + [(x.get('id'), x) for x in m.get('extra_rows') or []]
    for name, r in rows:
        for f in ('stage', 'fidelity', 'status'):
            if r.get(f) not in voc.get(f, ()):
                bad.append(f'{name}: {f} {r.get(f)!r} not in the vocabulary')
        for c in r.get('category') or []:
            if c not in voc.get('category', ()):
                bad.append(f'{name}: category {c!r} not in the vocabulary')
        if not r.get('category'):
            bad.append(f'{name}: no category')
        for e in r.get('env') or []:
            if e not in voc.get('env', ENVS):
                bad.append(f'{name}: env {e!r} not in the vocabulary')
        h = r.get('hours')
        if h is not None and not (isinstance(h, (int, float)) and h >= 0):
            bad.append(f'{name}: hours {h!r}')
    return bad


def r1(issues: list[dict], m: dict) -> dict:
    rows, unmapped = [], []
    for it in issues:
        mp = (m.get('issues') or {}).get(str(it['n']))
        if mp is None:
            unmapped.append(it['n'])
            rows.append({'id': f'#{it["n"]}', 'n': it['n'], 'env': [it['env']], 'stage': None, 'category': [], 'ids': [],
                         'fidelity': None, 'status': None, 'symptom': it['symptom'][:160], 'cause': it['cause'][:160],
                         'action': it['action'][:160], 'hours': None, 'hours_basis': 'unmapped', 'source': 'REPRO_ISSUES.md',
                         'unmapped': True})
            continue
        rows.append({'id': f'#{it["n"]}', 'n': it['n'], **{k: mp.get(k) for k in ('env', 'stage', 'category', 'ids', 'fidelity',
                                                                                    'status', 'symptom', 'cause', 'action',
                                                                                    'hours', 'hours_basis', 'fidelity_note')},
                     'time': it['time'], 'ledger_env': it['env'], 'source': f'REPRO_ISSUES.md #{it["n"]} ({mp.get("mapping_source")})'})
    stale = sorted(set(int(k) for k in (m.get('issues') or {})) - {it['n'] for it in issues})
    for x in m.get('extra_rows') or []:
        if x.get('include', True):
            rows.append({**{k: x.get(k) for k in ('env', 'stage', 'category', 'ids', 'fidelity', 'status', 'symptom', 'cause',
                                                  'action', 'hours', 'hours_basis')}, 'id': x['id'], 'n': None,
                         'source': x.get('source')})
    return {'rows': rows, 'unmapped': unmapped, 'mapping_without_issue': stale, 'vocabulary_problems': check_mapping(m)}


def r4(r1rows: list[dict], m: dict) -> dict:
    cats = (m.get('vocabularies') or {}).get('category') or []
    counts = {c: {e: 0 for e in ENVS} for c in cats}
    for r in r1rows:
        for c in r.get('category') or []:
            for e in r.get('env') or []:
                if c in counts and e in counts[c]:
                    counts[c][e] += 1
    timed = sorted((r for r in r1rows if isinstance(r.get('hours'), (int, float))), key=lambda r: (-r['hours'], r['n'] or 999))
    return {'counts': counts, 'top3': timed[:3], 'hours_total': sum(r['hours'] for r in timed),
            'lessons': (m.get('lessons') or [])[:10]}


def _gate_env_c(v: float | None, steps, pub: dict, g: dict) -> str:
    if v is None:
        return 'pending'
    d = abs(v - pub['val_bpb'])
    if d <= g['reproduced_abs'] and steps is not None and abs(steps - pub['steps']) <= g['steps_rel'] * pub['steps']:
        return 'reproduced'
    return 'partly reproduced' if d <= g['partly_abs'] else 'not reproduced'


def r2(store: Store, root: Path, src_doc: dict, results_text: str, deviation_text: str, env_b: dict | None) -> dict:
    problems, rows = [], {}
    devs = {p: sorted(set(re.findall(rf'\b({p}-\d+)\b', deviation_text)), key=lambda x: int(x.rsplit('-', 1)[1]))
            for p in ('D-RB', 'D-OC', 'D-AR', 'D-F1')}

    def checked(item):
        if item.get('check') and item['check'] not in results_text:
            problems.append(f'R2: "{item["check"]}" not found in REPRO_RESULTS.md ({item.get("source")})')
            return False
        return True

    a = src_doc['R2']['ENV-A']
    s0, s1 = (next(x for x in a['ours'] if x['name'] == n) for n in ('S0', 'S1'))
    ok_a = checked(s0) & checked(s1)
    g = a['gate']
    valid = s0['norm_diff'] < g['max_norm_diff'] and s1['norm_diff'] < g['max_norm_diff']
    ratio = s0['t_s'] / s1['t_s']
    rep = valid and g['S0_s'][0] <= s0['t_s'] <= g['S0_s'][1] and g['S1_s'][0] <= s1['t_s'] <= g['S1_s'][1]
    verdict = 'reproduced' if rep else 'partly reproduced' if valid and g['ratio'][0] <= ratio <= g['ratio'][1] else 'not reproduced'
    rows['ENV-A'] = {'environment': a['environment'], 'published': a['published'],
                     'ours': cell(f"S0 {s0['t_s']} s (score {s0['score']}, norm_diff {s0['norm_diff']}); S1 {s1['t_s']} s "
                                  f"(score {s1['score']}, norm_diff {s1['norm_diff']}); ratio {ratio:.2f}",
                                  [s0['source'], s1['source']]),
                     'gate': cell(verdict if ok_a else 'pending', 'design §0.3',
                                  note=f"S0 in {g['S0_s']} s, S1 in {g['S1_s']} s, both norm_diff < {g['max_norm_diff']}"),
                     'deviations': devs['D-RB'], 'not_reproducible': a['not_reproducible'], 'limits': a['limits']}
    b = src_doc['R2']['ENV-B']
    okb = all([checked(x) for x in b['defaults']] + [checked(b['loop_smoke'])])
    n_ok = sum(1 for x in b['defaults'] if x['tflops'] > 0)
    env_rep = n_ok >= b['gate']['min_dense_defaults'] and b['loop_smoke']['result'] == 'passed'
    proto = []
    mb = 'pending'
    if env_b:
        for w, W in env_b['workloads'].items():
            t = W.get('T_def')
            proto.append(f'{w} B-DEF {R.fmt(t, 1)}')
        mb = env_b['method_behaviour_reproduced']['value']
    rows['ENV-B'] = {'environment': b['environment'], 'published': b['published'],
                     'ours': cell('R3 defaults ' + ', '.join(f"{x['model']} {x['tflops']}" for x in b['defaults'])
                                  + ' TFLOPS/GPU (5/5 success); protocol B-DEF: ' + (', '.join(proto) or 'pending')
                                  + f"; loop smoke {b['loop_smoke']['result']}",
                                  [x['source'] for x in b['defaults']] + [b['loop_smoke']['source']]
                                  + [s for W in (env_b or {}).get('workloads', {}).values() for s in (W.get('T_def') or {}).get('src', [])]),
                     'gate': cell(('environment reproduced' if env_rep else 'environment not reproduced') if okb else 'pending',
                                  'design §0.3', note=f"method behaviour reproduced: {mb}; expert comparison: not reproducible (declared)"),
                     'method_behaviour': mb, 'deviations': devs['D-OC'], 'not_reproducible': b['not_reproducible'],
                     'limits': b['limits']}
    c = src_doc['R2']['ENV-C']
    lines, pend = [], []
    for ln in c['lines']:
        if 'receipt' in ln:
            rr = store.receipts(root / 'autoresearch' / ln['receipt']['run']).get(ln['receipt']['attempt_id'])
            if rr is None:
                lines.append({**ln, 'value': pending(f"receipt {ln['receipt']['attempt_id']} missing"), 'verdict': 'pending'})
                pend.append(f"R2 ENV-C line ({ln['name']}): receipt {ln['receipt']['attempt_id']} missing")
                continue
            want_dar7 = ln['name'] == 'c'                # (b) is the upstream recipe, (c) the D-AR-7 one
            want = {'env': 'autoresearch', 'task_id': 'ENV-C', 'run_id': ln['receipt']['run'], 'arm': 'C-BASE',
                    'valid': True, 'judge_valid': True, 'D-AR-7': want_dar7}
            have = {**{k: rr.get(k) for k in want if k != 'D-AR-7'}, 'D-AR-7': 'D-AR-7' in (rr.get('deviations') or [])}
            bad = {k: v for k, v in have.items() if v != want[k]}
            if bad:
                problems.append(f"R2 ENV-C line ({ln['name']}): receipt {ln['receipt']['attempt_id']} is not the "
                                f"{'D-AR-7' if want_dar7 else 'upstream'} C-BASE run with a valid trusted judge: {bad}")
                lines.append({**ln, 'value': pending('receipt not usable'), 'verdict': 'pending'})
                pend.append(f"R2 ENV-C line ({ln['name']}): receipt not usable")
                continue
            v = (rr.get('metric') or {}).get('value')
            steps = (rr.get('summary') or {}).get('num_steps')
            s = src(ln['receipt']['run'], rr)
            lines.append({**ln, 'value': cell(v, s), 'steps': steps, 'verdict': _gate_env_c(v, steps, c['published'], c['gate']),
                          'abs_delta': abs(v - c['published']['val_bpb']) if v else None})
        else:
            checked(ln)
            lines.append({**ln, 'value': cell(ln['val_bpb'], ln['source']),
                          'verdict': _gate_env_c(ln['val_bpb'], ln['steps'], c['published'], c['gate']),
                          'abs_delta': abs(ln['val_bpb'] - c['published']['val_bpb'])})
    by = {ln['name']: ln['verdict'] for ln in lines}
    if any(v == 'pending' for v in by.values()):        # EXT_002's wording needs all three lines
        status = 'pending'
    elif by.get('c') == 'reproduced':
        status = c['status']
    else:
        status = f"not reproduced with D-AR-7 (line c: {by.get('c')})"
    rows['ENV-C'] = {'environment': c['environment'], 'published': c['published'], 'lines': lines, 'status': status,
                     'deviations': devs['D-AR'], 'not_reproducible': c['not_reproducible'], 'limits': c['limits']}
    if not ok_a:
        pend.append('R2 ENV-A: reproduction values not confirmed in REPRO_RESULTS.md')
    if not okb:
        pend.append('R2 ENV-B: reproduction values not confirmed in REPRO_RESULTS.md')
    return {'rows': rows, 'problems': problems, 'deviation_ids': devs, 'pending': pend}


def r3(src_doc: dict, f1rows: dict, f1sec: dict) -> list[dict]:
    out = []
    for m in src_doc['R3']:
        arm = m.get('arm')
        row = {**m}
        if arm in f1rows:
            r = f1rows[arm]
            row['status'] = F1.status(f1sec, arm)
            row['speedup_vs_UC'] = r.get('speedup_vs_UC')
            row['recovered_fraction'] = r.get('recovered_fraction')
            costs = val(r.get('costs')) if ok(r.get('costs')) else None
            if isinstance(costs, dict):
                s = costs.get('search') or {}
                lk = costs.get('locked') or {}
                gpuh = (s.get('gpu_hours') or 0) + ((lk.get('gpu_seconds') or 0) / 3600.0)
                row['cost'] = cell({'GPUh': round(gpuh, 2), 'USD': s.get('llm_usd'),
                                    'wall_h': round((s.get('wall_seconds') or 0) / 3600.0, 2)}, f'summary_F1.json:{arm}')
            else:
                row['cost'] = r.get('costs') if isinstance(r.get('costs'), dict) and r['costs'].get('status') else \
                    pending('F1 arm not summarized')
        else:
            row['speedup_vs_UC'] = na(row.get('status') or 'not migrated')
            row['recovered_fraction'] = na(row.get('status') or 'not migrated')
            row['cost'] = na('—')
        out.append(row)
    return out


def build(store: Store, root: Path, reports: Path, *, env_b: dict | None, f1sec: dict, f1rows: dict,
          mapping_path: Path | None = None) -> dict:
    mapping_path = Path(mapping_path or MAPPING)
    m = json.loads(store.text(mapping_path, role='tool data') or '{}')
    srcd = json.loads(store.text(SOURCES, role='tool data') or '{}')
    issues = parse_issues(store.text(reports / 'REPRO_ISSUES.md') or '')
    t1 = r1(issues, m)
    t2 = r2(store, root, srcd, store.text(reports / 'REPRO_RESULTS.md') or '',
            store.text(reports / 'design' / 'DEVIATION_IDS.md') or '', env_b)
    t3 = r3(srcd, f1rows, f1sec)
    t4 = r4(t1['rows'], m)
    problems = [f'R1: issue #{n} has no mapping' for n in t1['unmapped']]
    problems += [f'R1: mapping for #{n} but no such issue' for n in t1['mapping_without_issue']]
    problems += [f'R1 mapping: {x}' for x in t1['vocabulary_problems']] + t2['problems']
    return {'R1': t1, 'R2': t2, 'R3': t3, 'R4': t4, 'problems': problems, 'mapping_file': str(mapping_path),
            'n_issues': len(issues), 'pending': t2['pending']}


def markdown(sec: dict) -> str:
    L = ['# Reproduction problems — summary tables R1–R4 (design §5.4)', '',
         f"{sec['n_issues']} entries of REPRO_ISSUES.md mapped by `{Path(sec['mapping_file']).name}` (X7's proposal for O's "
         'review; #1–#18 use the design\'s own mapping). Hours are ESTIMATES of handling time from the logged timestamps; '
         'machine time lost is named in the basis column of summary.json, not added.', '']
    L += ['## R1 — issue ledger', '']
    t = []
    for r in sec['R1']['rows']:
        act = r.get('action') or ''
        ids = [i for i in r.get('ids') or [] if i not in act]
        t.append([r['id'], ', '.join(r.get('env') or []), r.get('stage'), r.get('symptom'), r.get('cause'),
                  ', '.join(r.get('category') or []), (', '.join(ids) + ': ' if ids else '') + act,
                  r.get('fidelity'), r.get('status'), r.get('hours') if r.get('hours') is not None else 'n/a'])
    L += [R.table(['#', 'env', 'stage', 'symptom', 'root cause', 'category', 'action / deviation id', 'fidelity impact',
                   'status', 'hours (est.)'], t, {9: 1}), '']
    L += ['Rows E5 and D1–D10 are not REPRO_ISSUES.md entries: E5 is the finding AMENDMENT_EXT_005 ruled into R1; D1–D10 '
          'are the design §5.4 items that were to be registered during execution (set `include: false` in the mapping '
          'to drop one).', '']
    L += ['## R2 — environment reproduction verdicts', '']
    r2rows = sec['R2']['rows']
    t = []
    for env in ('ENV-A', 'ENV-B'):
        x = r2rows[env]
        gate = R.fmt(x['gate']) + (f" ({x['gate']['note']})" if x['gate'].get('note') else '')
        t.append([env, x['environment'], x['published'], x['ours'], gate, ', '.join(x['deviations']),
                  x['not_reproducible'], x['limits']])
    c = r2rows['ENV-C']
    pub = c['published']
    ours = '; '.join(f"({ln['name']}) {ln['label']}: {R.fmt(ln['value'], 6)}"
                     + (f" / {ln.get('steps')} steps" if ln.get('steps') else '')
                     + f" — {ln['verdict']}" + (f" (|Δ| {ln['abs_delta']:.4f})" if ln.get('abs_delta') is not None else '')
                     + f": {ln['verdict_text']}" for ln in c['lines'])
    t.append(['ENV-C', c['environment'], f"README example: val_bpb {pub['val_bpb']}, {pub['steps']} steps, MFU {pub['mfu_percent']} %",
              ours, c['status'], ', '.join(c['deviations']), c['not_reproducible'], c['limits']])
    L += [R.table(['env', 'environment', 'published reference', 'ours (single runs, seed 42)', 'gate (§0.3)', 'deviation ids',
                   'not reproducible (declared)', 'limits on the conclusions'], t), '']
    L += ['## R3 — method-migration ledger (fallback to F1)', '']
    t = [[m['method'], m['original'], m.get('arm') or '—', m.get('status', ''), m['difference'], m['speedup_vs_UC'],
          m['recovered_fraction'], m['cost'], ', '.join(f'#{i}' for i in m.get('issues') or []) or '—'] for m in sec['R3']]
    L += [R.table(['method', 'original environment / why not reproducible', 'migrated as', 'status', 'difference to the method',
                   'F1 locked speedup vs U+C', 'recovered fraction', 'cost (GPUh / USD / wall h)', 'R1 issues'], t, {5: 3, 6: 3}), '']
    L += ['## R4 — category statistics', '']
    cnt = sec['R4']['counts']
    t = [[c] + [cnt[c][e] for e in ENVS] + [sum(cnt[c].values())] for c in cnt]
    L += [R.table(['category'] + list(ENVS) + ['total'], t), '',
          'A row with several categories or environments counts once in each.', '']
    L += ['Top 3 by (estimated) hours: ' + '; '.join(f"{r['id']} {r['hours']} h ({r['symptom']})" for r in sec['R4']['top3'])
          + f". Estimated total: {sec['R4']['hours_total']:.1f} h.", '']
    L += ['What we would do first next time:', '', R.bullets(f"{x['text']} ({', '.join(f'#{i}' for i in x.get('issues') or [])})"
                                                             for x in sec['R4']['lessons']), '']
    L += ['## Problems', '', R.bullets(sec['problems']), '']
    return '\n'.join(L)
