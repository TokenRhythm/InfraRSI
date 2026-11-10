"""F1 screens the harness lost after the path had measured them (AMENDMENT_EXT_009, D29; issue #37).

Record contract (the supersede step is the harness's, X5; the formal acceptance below is X6's). A superseding record is a
signed `ext-receipt-v1` for the SAME attempt as the original `runner_error` receipt, written to
`<run dir>/receipts/<candidate_id>.recollected.json` (`.unrecoverable.json` for the flag) next to the retained original:

  identity     copied from the original: schema, env, task_id, arm, run_id, attempt_id, candidate_id, purpose ('screen'),
               point, nodes, utc_start, utc_end; `cost` is the original's, exactly (nothing re-spent, nothing uncharged)
  collection   'recollected_readonly': the verdict of the unchanged judge over the read-only re-collected node files,
               whatever it is - the runner's receipt fields (valid, judge_valid, judge {judge_valid, numerics_class,
               reasons, metric_s, details}, metric, step_time, invalid_reasons, guard_rejections, guard_reports incl. the
               reports of guards.check, numerics_gates_sha256 = the frozen file's, ...); no runner_error
               'unrecoverable': no verdict (valid False, judge None, no metric) - `harness_lost_unrecovered`
  supersedes   the original attempt_id; a record carries no `superseded_by` (none of a supersede)
  recollection {'stage': 'collect', 'original_error': <the original's runner_error, verbatim>,
                'error_class': 'truncated_reply' (JSONDecodeError / UnicodeDecodeError) | 'collect_transport_timeout'
                               (RemoteError, only with 'error_class_ruling': the ruling that covers that attempt),
                'conditions': {'i_collection_runner_error', 'ii_readonly_complete_layout',
                               'iii_mtimes_within_node_windows', 'iv_pulled_files_identical'}  (booleans, all four keys),
                'node_records': {<node name>: {'exit_code': 0, 'started_unix', 'finished_unix', 'release_state':
                               'process_group_gone', 'release_finished_unix'}} for exactly the original's four nodes,
                'method': 'fixed_collector' | 'sftp',
                'files': [{'path': <LAYOUT name>, 'sha256': <lowercase hex>, 'size': int >= 0, 'mtime': unix}]  (the
                         exact 88-file LAYOUT for recollected_readonly; mtime window per O's ruling: node start ..
                         node release + 60 s, the pre-launch records and launch logs from the attempt's utc_start),
                'pulled': [{'path', 'sha256'}]  (the files the failed collect had pulled; each equal to its re-collected
                         copy)}

Acceptance re-checks what the receipts can show: identity and cost, a collection-stage runner_error original
(JSONDecodeError / RemoteError / TimeoutError / RuntimeError of a launched, fully accounted screen; never a candidate
failure or a refusal), the four nodes' exits 0 and released process groups, the exact layout, digests, sizes, mtimes
inside each node's [start, release + 60 s] window (output files), pulled-file equality, a complete judge verdict with
the frozen gates and the guards' reports, one record per attempt, no supersede of a supersede. A JSONDecodeError original
(the certain #37 class) must have a record in a formal window (`must_dispose`).

Use (EXT_009): for the one-shot arm (F-PROMPTS) the record supersedes the `runner_error` receipt - the trace is re-derived
from its receipts (`rederive_trace`, re-sealed by `arms.retrace`) and its selection is the best valid screen as finally
recorded; for the multi-step arms the screen stays in the trajectory as the method saw it (invalid, class `harness_lost`,
charged, the point stays seen) and a recovered value is `descriptive_only`.
"""
from __future__ import annotations

import copy
import hashlib
import json

from ...harness import signing
from . import parse as P

KINDS = ('recollected_readonly', 'unrecoverable')
ONE_SHOT = ('F-PROMPTS',)
IDENTITY = ('schema', 'env', 'task_id', 'arm', 'run_id', 'attempt_id', 'candidate_id', 'purpose', 'point', 'nodes',
            'utc_start', 'utc_end')
CONDITIONS = ('i_collection_runner_error', 'ii_readonly_complete_layout', 'iii_mtimes_within_node_windows',
              'iv_pulled_files_identical')
COLLECTION_ERROR_TYPES = ('JSONDecodeError', 'UnicodeDecodeError', 'RemoteError')
CERTAIN_LOSS_TYPES = ('JSONDecodeError', 'UnicodeDecodeError')           # a truncated / undecodable collect reply
ERROR_CLASS = {'JSONDecodeError': 'truncated_reply', 'UnicodeDecodeError': 'truncated_reply',
               'RemoteError': 'collect_transport_timeout'}              # the latter only with a recorded ruling (c3)
PRE_LAUNCH = ('attempt.json', 'job.json', 'fn_spec.json')               # written before any job starts
METHODS = ('fixed_collector', 'sftp')
REQUIRED_REPORTS = ('wall_span', 'inter_step_consistency', 'scalar_rank_consistency')   # guards.check ran to its end
JUDGE_KEYS = ('judge_valid', 'numerics_class', 'reasons', 'metric_s', 'details')
LAYOUT = tuple(sorted([f'signals/rank{r:02d}.jsonl' for r in range(32)] + [f'ranks/rank{r:02d}.json' for r in range(32)]
                      + [f'nodes/node{n}.{k}.json' for n in range(4) for k in ('exit', 'release', 'pid')]
                      + [f'nodes/node{n}.log' for n in range(4)] + [f'nodes/fn_node{n}.launch.log' for n in range(4)]
                      + ['attempt.json', 'job.json', 'fn_spec.json', 'materialization.json']))
LAYOUT_FILES = len(LAYOUT)             # 88 (runner.FlashNextRunner.output_files)
MTIME_SLACK_S = 60.0
MARKS = ('superseded_by', '_path', signing.SIGNATURE_FIELD)
HEX = set('0123456789abcdef')


def _clean(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != '_path'}


def _signed(r) -> bool:
    return isinstance(r, dict) and signing.verify(_clean(r))


def _canon(r: dict) -> str:
    return json.dumps({k: v for k, v in r.items() if k not in MARKS}, sort_keys=True, default=str)


def digest(r: dict) -> str:
    return hashlib.sha256(json.dumps(_clean(r), sort_keys=True, default=str).encode()).hexdigest()


def is_record(r) -> bool:
    return isinstance(r, dict) and r.get('collection') in KINDS and bool(r.get('supersedes'))


def _metric(r: dict):
    v = ((r or {}).get('metric') or {}).get('value')
    return float(v) if P.finite(v) and v > 0 else None


def file_node(path: str) -> int | None:
    """The node (launcher rank 0-3) whose job or launcher wrote a layout file (rank r on node r // 8; the
    materialization record on node 0); None for the shared pre-launch records."""
    if path.startswith(('signals/rank', 'ranks/rank')):
        return int(path.rsplit('rank', 1)[1][:2]) // 8
    if path.startswith('nodes/node'):
        return int(path[len('nodes/node')])
    if path.startswith('nodes/fn_node'):
        return int(path[len('nodes/fn_node')])
    if path == 'materialization.json':
        return 0
    return None


def _unix(iso) -> float | None:
    import datetime
    try:
        return datetime.datetime.fromisoformat(str(iso).replace('Z', '+00:00')).timestamp()
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- building (the harness's supersede step)
def supersede_record(original: dict, judged: dict, recollection: dict) -> dict:
    """The recollected_readonly record: the judged receipt's verdict, the original's identity and cost, signed."""
    out = copy.deepcopy(_clean(judged))
    for k in (signing.SIGNATURE_FIELD, 'runner_error', 'superseded_by'):
        out.pop(k, None)
    for k in IDENTITY + ('cost',):
        out[k] = copy.deepcopy(original.get(k))
    out.update(collection='recollected_readonly', supersedes=original.get('attempt_id'),
               recollection=copy.deepcopy(recollection))
    return signing.sign(json.loads(json.dumps(out, default=str)))


def unrecoverable_record(original: dict, recollection: dict) -> dict:
    """The flag record of a screen whose node files could not be re-collected under (ii)-(iv): no verdict."""
    out = copy.deepcopy(_clean(original))
    for k in MARKS:
        out.pop(k, None)
    out.update(collection='unrecoverable', supersedes=original.get('attempt_id'), valid=False, judge_valid=False,
               judge=None, recollection=copy.deepcopy(recollection))
    return signing.sign(json.loads(json.dumps(out, default=str)))


# ---------------------------------------------------------------- acceptance (X6's formal rule)
def collection_failure(original: dict) -> bool:
    """Condition (i), receipt side: a launched, fully accounted screen that failed in our harness with a transport /
    decoding error class - not a candidate failure (no judge verdict, no guard veto) and not a refusal."""
    re = original.get('runner_error')
    return (isinstance(re, dict) and re.get('kind') == 'runner_error' and re.get('type') in COLLECTION_ERROR_TYPES
            and original.get('valid') is False and original.get('judge') is None and not original.get('guard_rejections')
            and not original.get('not_launched') and original.get('purpose') == 'screen'
            and (original.get('cost') or {}).get('accounting_complete') is True)


def must_dispose(original: dict) -> bool:
    """The certain #37 class (a truncated / undecodable collect reply): a formal window needs its record."""
    return collection_failure(original) and (original.get('runner_error') or {}).get('type') in CERTAIN_LOSS_TYPES


def _node_problems(original: dict, rc: dict) -> list[str]:
    nr = rc.get('node_records') if isinstance(rc.get('node_records'), dict) else {}
    nodes = list(original.get('nodes') or [])
    p = []
    if len(nodes) != 4 or set(nr) != set(nodes):
        return [f'node_records must cover exactly the four nodes {nodes} (got {sorted(nr)})']
    for n in nodes:
        x = nr.get(n) if isinstance(nr.get(n), dict) else {}
        s, f, rel = x.get('started_unix'), x.get('finished_unix'), x.get('release_finished_unix')
        if x.get('exit_code') != 0 or x.get('release_state') != 'process_group_gone':
            p.append(f'{n}: exit {x.get("exit_code")!r} / release {x.get("release_state")!r} (need 0 / process_group_gone)')
        if not (P.finite(s) and P.finite(f) and P.finite(rel) and s <= f <= rel):
            p.append(f'{n}: start / finish / release times missing or out of order')
    return p


def _file_problems(original: dict, rc: dict) -> list[str]:
    files = rc.get('files') if isinstance(rc.get('files'), list) else []
    nr = rc.get('node_records') if isinstance(rc.get('node_records'), dict) else {}
    nodes = list(original.get('nodes') or [])
    p, by_path = [], {}
    for f in files:
        ok = (isinstance(f, dict) and f.get('path') in LAYOUT and isinstance(f.get('sha256'), str)
              and len(f['sha256']) == 64 and set(f['sha256']) <= HEX and type(f.get('size')) is int and f['size'] >= 0
              and P.finite(f.get('mtime')))
        if not ok:
            p.append(f'file entry not a layout file with sha256 / size / mtime: {str(f)[:120]}')
            continue
        by_path[f['path']] = f
    if len(files) != LAYOUT_FILES or set(by_path) != set(LAYOUT):
        p.append(f'the re-collection is not exactly the {LAYOUT_FILES}-file layout (missing '
                 f'{sorted(set(LAYOUT) - set(by_path))[:4]}, {len(files)} entries)')
    rel_all = [x.get('release_finished_unix') for x in nr.values() if isinstance(x, dict)]
    t0 = _unix(original.get('utc_start'))
    for path, f in by_path.items():                                   # (iii) O's ruling of 2026-10-21 (X5's basis)
        n = file_node(path)
        x = nr.get(nodes[n]) if n is not None and n < len(nodes) else None
        pre = path in PRE_LAUNCH or path.endswith('.launch.log')      # written before the job starts
        lo = t0 if pre else (x or {}).get('started_unix')
        hi = (x or {}).get('release_finished_unix') if x is not None else (
            max(rel_all) if len(rel_all) == 4 and all(P.finite(v) for v in rel_all) else None)
        if not (P.finite(lo) and P.finite(hi) and lo <= f['mtime'] <= hi + MTIME_SLACK_S):   # the ruled interval
            p.append(f'{path}: mtime {f["mtime"]} outside its window [{lo}, {hi} + {MTIME_SLACK_S:.0f} s]')
    for q in rc.get('pulled') if isinstance(rc.get('pulled'), list) else ['not a list']:   # (iv)
        if not (isinstance(q, dict) and q.get('path') in by_path and q.get('sha256') == by_path[q['path']]['sha256']):
            p.append(f'pulled file not byte-identical to its re-collected copy: {str(q)[:120]}')
    return p


def _verdict_problems(record: dict) -> list[str]:
    from . import judge as J
    p = []
    j = record.get('judge')
    if not isinstance(j, dict) or any(k not in j for k in JUDGE_KEYS):
        return ['no complete judge verdict (the unchanged judge must have run)']
    if j.get('judge_valid') is not record.get('judge_valid'):
        p.append('judge_valid differs from the judge verdict')
    if record.get('numerics_gates_sha256') != J.gates_sha256():
        p.append('not judged with the frozen numerics gates')
    reports = {r.get('id') for r in record.get('guard_reports') or [] if isinstance(r, dict)}
    if not set(REQUIRED_REPORTS) <= reports:
        p.append(f'guards.check did not run (reports {sorted(reports)[:6]})')
    rej = record.get('guard_rejections') or []
    if record.get('valid') is not (record.get('judge_valid') is True and not rej):
        p.append('valid is not judge_valid and no guard veto')
    v, st = _metric(record), (record.get('step_time') or {}).get('mean_window_max_rank_s')
    if record.get('valid') is True:
        if not (v is not None and v == j.get('metric_s') and v == st):
            p.append('metric, step_time and the judge metric disagree')
    elif v is not None or st is not None:
        p.append('an invalid verdict carries a metric')
    reasons = record.get('invalid_reasons')
    want = ([] if record.get('judge_valid') else [f'judge: {r}' for r in j.get('reasons') or []] or ['judge: invalid'])
    want += [f'guard:{g.get("id")}: {g.get("detail")}' for g in rej if isinstance(g, dict)]
    if not isinstance(reasons, list) or reasons != want:          # exactly what EnvRunner.receipt writes
        p.append('invalid_reasons are not the judge reasons and guard vetoes')
    if 'runner_error' in record:
        p.append('a re-collected verdict carries a runner_error')
    return p


def record_problems(original: dict | None, record: dict) -> list[str]:
    """Why `record` cannot supersede `original` (empty = admitted)."""
    p = []
    if not _signed(record):
        p.append('record: harness signature missing or invalid')
    if record.get('collection') not in KINDS:
        p.append(f'record: collection {record.get("collection")!r} not in {KINDS}')
    if 'superseded_by' in record:
        p.append('record: a superseding record that was itself superseded (none of a supersede)')
    if not isinstance(original, dict):
        return p + [f'no original receipt for attempt {record.get("supersedes")}']
    if not _signed(original):
        p.append('original: harness signature missing or invalid')
    if is_record(original) or original.get('supersedes'):
        p.append('a record never supersedes a record')
    if record.get('supersedes') != original.get('attempt_id') or record.get('attempt_id') != original.get('attempt_id'):
        p.append('record: supersedes / attempt_id do not name the original attempt')
    diff = [k for k in IDENTITY if record.get(k) != original.get(k)]
    if diff:
        p.append(f'record: identity differs from the original: {diff}')
    if record.get('purpose') != 'screen':
        p.append('record: only a screen is superseded')
    if record.get('cost') != original.get('cost'):
        p.append('record: cost is not the original cost')
    if not collection_failure(original):
        p.append('original: not a collection-stage runner_error of a launched, fully accounted screen (a candidate '
                 'failure or a refusal stands)')
    rc = record.get('recollection') if isinstance(record.get('recollection'), dict) else {}
    cond = rc.get('conditions') if isinstance(rc.get('conditions'), dict) else {}
    if any(type(cond.get(c)) is not bool for c in CONDITIONS):
        p.append(f'record: the four integrity conditions must be recorded as booleans {CONDITIONS}')
    if rc.get('stage') != 'collect' or rc.get('original_error') != original.get('runner_error'):
        p.append('record: the failure is not recorded as the original collect-stage error')
    err_type = (original.get('runner_error') or {}).get('type')
    if rc.get('error_class') != ERROR_CLASS.get(err_type) or (
            rc.get('error_class') == 'collect_transport_timeout' and not rc.get('error_class_ruling')):
        p.append(f'record: error class {rc.get("error_class")!r} is not the #37 class of a {err_type} '
                 '(a transport timeout needs its recorded ruling)')
    if cond.get('i_collection_runner_error') is not True:
        p.append('record: condition (i) not met')
    p += [f'record: {x}' for x in _node_problems(original, rc)]
    if record.get('collection') == 'recollected_readonly':
        bad = [c for c in CONDITIONS if cond.get(c) is not True]
        if bad:
            p.append(f'record: integrity conditions not met: {bad}')
        if rc.get('method') not in METHODS:
            p.append(f'record: collection method {rc.get("method")!r} not in {METHODS}')
        p += [f'record: {x}' for x in _file_problems(original, rc)]
        p += [f'record: {x}' for x in _verdict_problems(record)]
    elif record.get('collection') == 'unrecoverable':
        if all(cond.get(c) is True for c in CONDITIONS[1:]):
            p.append('record: conditions (ii)-(iv) all hold, so the screen is recoverable and must be superseded')
        raw = (record.get('metric') or {}).get('value') if isinstance(record.get('metric'), dict) else record.get('metric')
        st = (record.get('step_time') or {}).get('mean_window_max_rank_s') if isinstance(record.get('step_time'), dict)             else record.get('step_time')
        if record.get('valid') is not False or record.get('judge') is not None or raw is not None or st is not None:
            p.append('record: an unrecoverable screen has no verdict')
    return p


def index(receipts) -> tuple[dict, dict, list[str]]:
    """(originals {attempt: receipt}, records {attempt: admitted record}, problems). Every copy of an original must be
    signed and identical (a re-signed `superseded_by` mark aside); a second record for an attempt, a record without its
    original or a record that fails `record_problems` is a problem (vi)."""
    originals, recs, problems = {}, {}, []
    for r in receipts or []:
        (recs.setdefault(r.get('supersedes'), []) if is_record(r) else originals.setdefault(r.get('attempt_id'), [])).append(r)
    orig_one = {}
    for att, copies in originals.items():
        if len({_canon(c) for c in copies}) > 1 or any(not _signed(c) for c in copies):
            problems.append(f'differing receipts for the same attempt in the window: {att}')
        orig_one[att] = next((c for c in copies if not c.get('superseded_by')), copies[0])
    admitted = {}
    for att, rs in recs.items():
        if len({digest(r) for r in rs}) > 1:
            problems.append(f'attempt {att}: more than one superseding record (one supersede per attempt)')
            continue
        why = record_problems(orig_one.get(att), rs[0])
        if why:
            problems.append(f'attempt {att}: superseding record not admitted: {why}')
            continue
        admitted[att] = rs[0]
    return orig_one, admitted, problems


def undisposed(originals: dict, records: dict, arms=None) -> list[str]:
    """Attempts of the certain #37 class (must_dispose) in the given arms with no admitted record."""
    return sorted(str(a) for a, r in originals.items()
                  if must_dispose(r) and a not in records and (arms is None or r.get('arm') in arms))


# ---------------------------------------------------------------- the one-shot arm's trace, re-derived
def rederive_trace(trace: dict, records: dict) -> dict:
    """The one-shot arm's trace with every admitted recollected_readonly record applied to its screen row (verdict,
    reasons; the cost stays the original's), `collection` on every row (collected / recollected_readonly), an
    unrecoverable screen flagged void, and the running best and the selection recomputed as SearchSession.best_valid
    does (lowest step time; the start counts as index -1). Pure: sealing is the caller's."""
    tr = copy.deepcopy({k: v for k, v in trace.items() if k != signing.SIGNATURE_FIELD})
    if tr.get('arm') not in ONE_SHOT:
        raise ValueError(f'{tr.get("arm")}: only the one-shot arm is re-derived (EXT_009 Q1); a multi-step arm keeps '
                         'what it saw (Q4)')
    applied = []
    by_cid = {c.get('id'): c for c in tr.get('candidates') or []}
    for row in tr.get('screens') or []:
        rec = records.get(row.get('attempt_id'))
        if rec is not None and rec.get('collection') == 'recollected_readonly':
            row.update(original_valid=row.get('valid'), original_reasons=row.get('reasons'),
                       valid=bool(rec.get('valid')), step_time=_metric(rec) if rec.get('valid') else None,
                       reasons=list(rec.get('invalid_reasons') or [])[:4], collection='recollected_readonly',
                       supersedes=rec.get('supersedes'), record_digest=digest(rec))
            cand = by_cid.get(row.get('candidate_id'))
            if cand is not None:
                cand.update(valid=row['valid'], step_time=row['step_time'], reasons=row['reasons'],
                            peak_reserved_gib=rec.get('peak_reserved_gib', cand.get('peak_reserved_gib')),
                            collection='recollected_readonly')
            applied.append({'candidate_id': row.get('candidate_id'), 'attempt_id': row.get('attempt_id'),
                            'valid': row['valid'], 'step_time': row['step_time'], 'record_digest': row['record_digest']})
        elif rec is not None:                      # unrecoverable: the screen stays runner_error, void
            row.update(collection='collected', harness_lost_unrecovered=True, void=True, record_digest=digest(rec))
        else:
            row['collection'] = 'collected'
    start = (tr.get('start_step_time'), -1, tr.get('start') or 'c0')
    best = start if P.finite(start[0]) else None
    for row in tr.get('screens') or []:
        if row.get('valid') and P.finite(row.get('step_time')):
            cand = (row['step_time'], row.get('index'), row.get('candidate_id'))
            best = min(best, cand) if best else cand
        row['best_valid'], row['best_step_time'] = (best[2], best[0]) if best else (None, None)
    chosen = best[2] if best else (tr.get('start') or 'c0')
    if chosen != tr.get('selected'):
        if chosen == (tr.get('start') or 'c0'):
            from . import switch_space as S
            point = S.start_point()
        else:
            point = next((r.get('point') for r in records.values() if r.get('candidate_id') == chosen), None)
            if point is None:
                raise ValueError(f'no point for the re-derived selection {chosen}')
        tr.update(selected=chosen, selected_config=point, selected_overlay=None)
    tr['selected_step_time'] = best[0] if best else tr.get('start_step_time')
    extra = dict(tr.get('extra') or {})
    extra['ext009'] = {'rule': 'AMENDMENT_EXT_009 Q1/Q2: one-shot arm, re-derived from its receipts incl. the superseding '
                               'records; selection = the best valid screen as finally recorded',
                       'applied': applied, 'rederived_from_sha256': hashlib.sha256(
                           json.dumps(trace, sort_keys=True, default=str).encode()).hexdigest()}
    tr['extra'] = extra
    return tr


# ---------------------------------------------------------------- what the summary reports per arm (Q4, Q5)
def harness_view(arm: str, trace: dict, originals: dict, records: dict) -> dict:
    """Per screen row: collection (collected / recollected_readonly / rescreen) and class (None / failed / runner_error /
    infra_refusal / harness_lost / recovered); per arm: n_harness_lost, recovered values (descriptive_only for the
    multi-step arms), harness_cost_selection (a recovered valid value better than the arm's own final selection),
    n_failed_by_class, comparison_withheld."""
    one_shot = arm in ONE_SHOT
    rows, recovered, classes = [], [], {}
    for sc in (trace or {}).get('screens') or []:
        att = sc.get('attempt_id')
        rec, orig = records.get(att), originals.get(att) or {}
        row = {'index': sc.get('index'), 'candidate_id': sc.get('candidate_id'), 'attempt_id': att,
               'valid': sc.get('valid'), 'step_time': sc.get('step_time')}
        if sc.get('collection') == 'rescreen':
            row.update(collection='rescreen', rescreen_of=sc.get('rescreen_of'))
            cls = None if sc.get('valid') else 'failed'
        elif rec is not None and rec.get('collection') == 'recollected_readonly':
            value = {'attempt_id': att, 'candidate_id': sc.get('candidate_id'), 'valid': bool(rec.get('valid')),
                     'step_time': _metric(rec) if rec.get('valid') else None,
                     'numerics_class': (rec.get('judge') or {}).get('numerics_class'),
                     'record_digest': digest(rec), 'descriptive_only': not one_shot}
            recovered.append(value)
            row.update(collection='recollected_readonly', supersedes=att)
            if one_shot:
                cls = 'recovered'
            else:
                row['recovered'] = value
                cls = 'harness_lost'
        elif rec is not None:
            row.update(collection='collected', harness_lost_unrecovered=True)
            cls = 'harness_lost'
        else:
            row.update(collection='collected')
            if sc.get('valid'):
                cls = None
            elif orig.get('not_launched'):
                cls = 'infra_refusal'
            elif orig and collection_failure(orig):
                cls = 'runner_error'
            else:
                cls = 'failed'
        row['class'] = cls
        if cls:
            classes[cls] = classes.get(cls, 0) + 1
        rows.append(row)
    n_lost = classes.get('harness_lost', 0)
    sel_t = (trace or {}).get('selected_step_time')
    hcs = (not one_shot) and any(v['valid'] and v['step_time'] and sel_t and v['step_time'] < sel_t for v in recovered)
    return {'arm': arm, 'one_shot': one_shot, 'rows': rows, 'n_harness_lost': n_lost,
            'n_recovered': classes.get('recovered', 0), 'recovered': recovered, 'harness_cost_selection': bool(hcs),
            'n_failed_by_class': classes,
            'comparison_withheld': (not one_shot) and (n_lost >= 2 or bool(hcs))}
