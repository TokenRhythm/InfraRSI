"""Read-only re-collection of an F1 screen's outputs from its node files (AMENDMENT_EXT_009, D29; issue #37).

    python -m infrarsi.ext.flashnext.recollect proof --attempt-dir DIR [--via WINH3] [--out FILE]
        Q4: the zero-GPU proof of the collector. A screen that was collected cleanly (e.g. F-HT) is re-collected from its
        node files and every file of its recorded collection must match, sha256 for sha256.
    python -m infrarsi.ext.flashnext.recollect check --run-dir DIR --attempt ID [--via WINH3]
        Value-blind preflight: conditions (i)-(iv) and (vi) on a temporary re-collection. No judge, no record, nothing
        kept.
    python -m infrarsi.ext.flashnext.recollect supersede --run-dir DIR --attempt ID --uc-ref FILE [--via WINH3]
        Q1: the harness's supersede step. Any screen not covered by the rule is refused, and nothing is written. For a
        covered screen it always writes exactly one record, receipts/<candidate>.recollected.json (or
        .unrecoverable.json; exclusive create) in X6's format (supersede.py), before anything of a verdict is returned
        (value-blind). A screen is covered when its receipt is a
        `runner_error` of the collection class (#37) after the job: a truncated reply, or (O's ruling under Q3, c3
        only) the old collector's read timing out on every try. The step re-collects the node files read-only and
        checks (i)-(iv) and (vi):
        - all four hold: the files are judged by the unchanged judge (the arm's U+C reference, which must be this
          run's own c0, the frozen gates, the five vetoes), with the attempt binding as an identity veto (v), and the
          record is `recollected_readonly`;
        - (i) holds but (ii)-(iv) do not: the record is the `unrecoverable` flag; the screen stays `runner_error`,
          `harness_lost_unrecovered`.

On WINH: reads only (runner.fetch_outputs through one node's SSH; no lease, no launch, no write). The re-collected files
of a record are kept beside the run directories, in <EXEC>/ext_bench/flashnext/_recollected/<run_id>/<cand>-<attempt>/,
never inside a run directory. The only write into a run directory is the record.

(iii), O's ruling of 2026-10-21 (recorded per file as `window_basis`). Every file's mtime must lie at or before its
node's release `finished_utc` + 60 s. The lower bound is that node's `started_unix` for every file the job or its
launcher writes after the start. For the files written before any job starts (the attempt / job / fn_spec records the
runner uploads, and each node's launch log, which the node helper creates when it starts the launcher), the lower bound
is the attempt's own start, the original receipt's utc_start.

(iv) What the failed collection pulled. The old collector read the files in sorted remote-path order, in batches of 8,
and wrote each batch before the next was read. So what it left in the attempt directory is exactly the first 8*k of
those paths (it never collected logs). The present set must be such a prefix and byte-identical to the re-collected
copies. The PC side is trusted in the harness's threat model; the pulled set and its sha256s are pinned in the record.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

from ...contracts import utc_now
from ...harness import signing
from . import judge as J
from . import runner as R
from . import supersede as SU

READ_VIA = 'WINH3'
MTIME_SLACK_S = 60.0
OLD_BATCH = 8                                                   # the old collector's read_files batch
COLLECTION_ERRORS = ('JSONDecodeError', 'UnicodeDecodeError')   # #37: a truncated or undecodable reply
TIMEOUT_FORM = re.compile(r"WINH\d: transport failed: TimeoutError\('WINH\d: remote command exceeded \d+(\.\d+)?s'\)")
RULED_TRANSPORT_TIMEOUTS = {                                    # (run_id, attempt_id): the ruling
    ('f1_prompts_20261021b', 'a0fb0d485687c'): "O under AMENDMENT_EXT_009 Q3, 2026-10-21 ~21:00 (c3, old collector)"}
PRE_LAUNCH = ('attempt.json', 'job.json', 'fn_spec.json')
RECORD_SUFFIX = {'recollected_readonly': '.recollected.json', 'unrecoverable': '.unrecoverable.json'}   # X6's names
METHOD = 'fixed_collector'                                      # X6's name for runner.fetch_outputs (node lzma chunks)


class Refused(ValueError):
    """Not a screen this step may supersede, or not with these inputs (nothing written)."""


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _json(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _unix(iso) -> float | None:
    try:
        return datetime.datetime.fromisoformat(str(iso)).timestamp()
    except (TypeError, ValueError):
        return None


def reader(via: str = READ_VIA, ops_by_node: dict | None = None) -> R.FlashNextRunner:
    """A FlashNextRunner used only to read and judge: never leases, launches or writes on WINH (fetch_outputs through
    nodes[0] = `via`)."""
    via = str(via).upper()
    nodes = (via,) + tuple(n for n in ('WINH0', 'WINH1', 'WINH3', 'WINH4', 'WINH6') if n != via)[:3]
    if ops_by_node is None:
        from ..common.remote_exec import WinhNodeOps
        ops_by_node = {n: WinhNodeOps(n) for n in nodes}
    return R.FlashNextRunner(ops_by_node, nodes=nodes, runtime_dir='-', gates=J.load_gates(), out_dir=None)


def layout_of(attempt_dir: Path) -> dict:
    attempt, job = _json(Path(attempt_dir) / 'attempt.json'), _json(Path(attempt_dir) / 'job.json')
    if not isinstance(attempt, dict) or not isinstance(job, dict):
        raise Refused(f'{attempt_dir}: attempt.json / job.json missing (the node files cannot be located)')
    return R.layout_from_records(attempt, job)


def recollect(rdr: R.FlashNextRunner, attempt_dir: Path, dest: Path) -> tuple[dict, dict]:
    """Re-collect the screen whose records are in `attempt_dir` into the empty directory `dest`: (layout, fetch record)."""
    lay = layout_of(attempt_dir)
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise Refused(f'{dest}: not empty (a re-collection goes into a fresh directory)')
    dest.mkdir(parents=True, exist_ok=True)
    return lay, rdr.fetch_outputs(lay, dest)


def layout_names(lay: dict) -> list[str]:
    wanted, logs = R.FlashNextRunner.output_files(lay)
    return sorted(set(wanted.values()) | set(logs.values()))


def recollected_dir(run_dir: Path, run_id: str, candidate_id: str, attempt_id: str) -> Path:
    return Path(run_dir).parent / '_recollected' / run_id / f'{candidate_id}-{attempt_id}'


# ---------------------------------------------------------------- Q4: the zero-GPU proof
def proof(attempt_dir, *, rdr: R.FlashNextRunner | None = None, dest=None, via: str = READ_VIA) -> dict:
    """Re-collect a cleanly collected screen and compare every file of its recorded collection, sha256 for sha256."""
    attempt_dir = Path(attempt_dir)
    rdr = rdr or reader(via)
    dest = Path(dest) if dest else attempt_dir.parent.parent.parent / '_recollected' / 'proof' / (
        f'{attempt_dir.parent.parent.name}-{attempt_dir.name}-{time.strftime("%Y%m%d%H%M%S")}')
    t0 = time.time()
    lay, fetch = recollect(rdr, attempt_dir, dest)
    rows = []
    for name in layout_names(lay):
        a, b = attempt_dir / name, dest / name
        ha = _sha(a) if a.is_file() else None
        hb = _sha(b) if b.is_file() else None
        rows.append({'name': name, 'recorded_sha256': ha, 'recollected_sha256': hb, 'bytes': b.stat().st_size if hb else None,
                     'equal': ha is not None and ha == hb})
    n_eq = sum(1 for r in rows if r['equal'])
    return {'schema': 'flashnext-recollection-proof-v1', 'rule': 'AMENDMENT_EXT_009 Q4', 'utc': utc_now(),
            'attempt_dir': str(attempt_dir), 'recollected_dir': str(dest), 'read_via': rdr.nodes[0],
            'files': len(rows), 'equal': n_eq, 'all_equal': n_eq == len(rows) == SU.LAYOUT_FILES,
            'signal_bytes': sum(r['bytes'] or 0 for r in rows if r['name'].startswith('signals/')),
            'fetch': {k: fetch.get(k) for k in ('method', 'calls', 'sftp_files', 'seconds')},
            'seconds': round(time.time() - t0, 1), 'rows': rows}


# ---------------------------------------------------------------- Q1: the supersede step
def error_class(original: dict) -> str | None:
    """The #37 class of the original's runner_error. 'truncated_reply': a reply cut by the exit-status race, i.e.
    undecodable JSON. 'collect_transport_timeout': the old collector's batched read exceeding its timeout on every try,
    for the attempts O ruled on only (RULED_TRANSPORT_TIMEOUTS) and only in the exact transport-timeout form. None for
    anything else."""
    err = original.get('runner_error') if isinstance(original.get('runner_error'), dict) else {}
    if err.get('type') in COLLECTION_ERRORS:
        return 'truncated_reply'
    if (err.get('type') == 'RemoteError' and TIMEOUT_FORM.fullmatch(str(err.get('message') or ''))
            and (original.get('run_id'), original.get('attempt_id')) in RULED_TRANSPORT_TIMEOUTS):
        return 'collect_transport_timeout'
    return None


def collect_stage(original: dict) -> bool:
    """The failure came after the job: all four node leases were released (the runner releases them only after the
    launchers' exits and the verified cleanup), so only the collection (or the judge, which makes no remote call) can
    have raised."""
    rel = original.get('release_by_node') if isinstance(original.get('release_by_node'), dict) else {}
    return (len(rel) == 4 and sorted(rel) == sorted(original.get('nodes') or [])
            and all(isinstance(s, dict) and s.get('state') in ('removed', 'absent') for s in rel.values()))


def original_problems(original: dict) -> list[str]:
    """Why `original` is not a #37-class runner_error this step may supersede (receipt side of (i), and (vi))."""
    p = []
    if not signing.verify({k: v for k, v in original.items() if k != '_path'}):
        p.append('original: harness signature missing or invalid')
    if SU.is_record(original) or original.get('supersedes'):
        p.append('(vi) a record never supersedes a record')
    if not SU.collection_failure(original):
        p.append('(i) not a runner_error with complete accounting (a candidate failure or a refusal stands)')
    err = original.get('runner_error') if isinstance(original.get('runner_error'), dict) else {}
    if error_class(original) is None:
        p.append(f'(i) runner_error {err.get("type")!r} is not of the collection class (#37: {COLLECTION_ERRORS}, or a '
                 'ruled collect transport timeout)')
    if not collect_stage(original):
        p.append('(i) the failure is not after the job (the four node leases were not all released)')
    if original.get('purpose') != 'screen':
        p.append(f'purpose {original.get("purpose")!r}: only a screen is superseded')
    return p


def existing_records(run_dir: Path, attempt_id: str) -> list[Path]:
    out = []
    for f in sorted((Path(run_dir) / 'receipts').glob('*.json')):
        r = _json(f)
        if isinstance(r, dict) and SU.is_record(r) and r.get('supersedes') == attempt_id:
            out.append(f)
    return out


def node_records(dest: Path) -> list[dict]:
    """The four launchers' exit and release records, in node-rank order."""
    out = []
    for n in range(4):
        ex = _json(Path(dest) / 'nodes' / f'node{n}.exit.json') or {}
        rel = _json(Path(dest) / 'nodes' / f'node{n}.release.json') or {}
        out.append({'node': n, 'exit_code': ex.get('exit_code'), 'started_unix': ex.get('started_unix'),
                    'finished_unix': ex.get('finished_unix'), 'process_group_gone': rel.get('process_group_gone'),
                    'idle': rel.get('idle'), 'release_finished_utc': rel.get('finished_utc'),
                    'release_finished_unix': _unix(rel.get('finished_utc'))})
    return out


def node_records_by_name(nodes: list[dict], names: list[str]) -> dict:
    """X6's form: {node name: exit_code, started_unix, finished_unix, release_state, release_finished_unix}; rank r ran
    on names[r] (fn_spec order, bound to the receipt's node list)."""
    return {names[n['node']]: {'exit_code': n['exit_code'], 'started_unix': n['started_unix'],
                               'finished_unix': n['finished_unix'],
                               'release_state': 'process_group_gone' if n['process_group_gone'] is True else 'present',
                               'release_finished_unix': n['release_finished_unix'], 'idle': n['idle']}
            for n in nodes if n['node'] < len(names)}


def owner(name: str) -> int | None:
    """The node (launcher rank) whose job or launcher wrote `name`; None for the shared pre-launch records."""
    m = re.match(r'(?:signals|ranks)/rank(\d\d)\.', name)
    if m:
        return int(m.group(1)) // 8
    m = re.match(r'nodes/(?:fn_)?node(\d)\.', name)
    if m:
        return int(m.group(1))
    if name == 'materialization.json':
        return 0
    return None


def mtime_windows(names: list[str], files: dict, nodes: list[dict], attempt_start: float | None) -> list[dict]:
    """(iii) per file: its mtime, its window and the window's basis (see the module docstring)."""
    rel_all = [n['release_finished_unix'] for n in nodes if isinstance(n.get('release_finished_unix'), float)]
    rows = []
    for name in names:
        f = files.get(name) or {}
        k = owner(name)
        pre = name in PRE_LAUNCH or name.endswith('.launch.log')
        node = nodes[k] if k is not None else None
        lo = attempt_start if pre else (node or {}).get('started_unix')
        hi_base = (node or {}).get('release_finished_unix') if node is not None else (max(rel_all) if len(rel_all) == 4 else None)
        hi = hi_base + MTIME_SLACK_S if isinstance(hi_base, float) else None
        m = f.get('mtime')
        ok = all(isinstance(v, (int, float)) for v in (lo, hi, m)) and lo <= m <= hi
        rows.append({'name': name, 'path': f.get('remote'), 'sha256': f.get('sha256'), 'size': f.get('bytes'),
                     'source_bytes': f.get('source_bytes'), 'mtime': m, 'window': [lo, hi],
                     'window_basis': ('attempt utc_start' if pre else f'node {k} started_unix') +
                     f' .. node {"*" if k is None else k} release finished_utc + {MTIME_SLACK_S:.0f} s', 'within': ok})
    return rows


def old_collector_order(lay: dict) -> list[str]:
    """The local names in the order the old collector read them (sorted remote paths; logs were read separately)."""
    wanted, _ = R.FlashNextRunner.output_files(lay)
    return [wanted[p] for p in sorted(wanted)]


def pulled_identical(attempt_dir: Path, dest: Path, lay: dict) -> dict:
    """(iv) the files the failed collection left in the attempt directory: exactly a prefix of the old collector's
    order in whole batches (what a failure at batch k+1 leaves), each byte-identical to its re-collected copy."""
    order = old_collector_order(lay)
    pulled = sorted(p.relative_to(attempt_dir).as_posix() for p in Path(attempt_dir).rglob('*') if p.is_file())
    sha = {name: _sha(Path(attempt_dir) / name) for name in pulled}
    differing = [n for n in pulled if n in order and (not (Path(dest) / n).is_file() or sha[n] != _sha(Path(dest) / n))]
    outside = [n for n in pulled if n not in order]
    k = len(pulled)
    prefix = (k > 0 and k % OLD_BATCH == 0 and sorted(order[:k]) == pulled) or (0 < k == len(order) and
                                                                                  sorted(order) == pulled)
    return {'files': k, 'identical': k - len(differing) - len(outside), 'differing': differing, 'outside_layout': outside,
            'old_collector_prefix': prefix, 'batches': k // OLD_BATCH, 'sha256': sha}


def attempt_binding(original: dict, dest: Path) -> tuple[dict, list[tuple[str, str]]]:
    """The node files' records bound to the original attempt: the runner derives the job's rendezvous port from its
    attempt id (runner.master_port) and wrote the node map it leased. A mismatch is an identity veto (Q1 (v))."""
    job, spec = _json(Path(dest) / 'job.json') or {}, _json(Path(dest) / 'fn_spec.json') or {}
    b = {'job_master_port': job.get('master_port'), 'runner_master_port': R.master_port(original['attempt_id']),
         'fn_spec_nodes': spec.get('nodes'), 'receipt_nodes': original.get('nodes')}
    problems = []
    if b['job_master_port'] != b['runner_master_port']:
        problems.append(('attempt_binding', f'job.json master_port {b["job_master_port"]} is not the port of attempt '
                                            f'{original["attempt_id"]} ({b["runner_master_port"]})'))
    if b['fn_spec_nodes'] != b['receipt_nodes']:
        problems.append(('attempt_binding', f'fn_spec nodes {b["fn_spec_nodes"]} are not the receipt nodes {b["receipt_nodes"]}'))
    return b, problems


def _prepare(run_dir: Path, attempt_id: str, rdr: R.FlashNextRunner, dest: Path) -> dict:
    """Everything before the judge: the original, (i) receipt side, (vi), the re-collection into `dest`, (i) node side,
    (ii)-(iv), the binding. Raises Refused when (i) or (vi) fails."""
    orig_path = Path(run_dir) / 'receipts' / f'{attempt_id}.json'
    original = _json(orig_path)
    if not isinstance(original, dict) or original.get('attempt_id') != attempt_id:
        raise Refused(f'{orig_path}: no receipt of attempt {attempt_id}')
    why = original_problems(original)
    if existing_records(run_dir, attempt_id):
        why.append(f'(vi) attempt {attempt_id} already has a superseding record (one per attempt)')
    if why:
        raise Refused('; '.join(why))
    attempt_dir = Path(run_dir) / 'attempts' / f'{original["candidate_id"]}-{attempt_id}'
    lay, fetch = recollect(rdr, attempt_dir, dest)
    names = layout_names(lay)
    files = fetch['files']
    present = sorted(p.relative_to(dest).as_posix() for p in Path(dest).rglob('*') if p.is_file())
    nodes = node_records(dest)
    if not all(n['exit_code'] == 0 and n['process_group_gone'] is True for n in nodes):
        raise Refused(f'(i) the node records do not show a completed job: {nodes}')
    windows = mtime_windows(names, files, nodes, _unix(original.get('utc_start')))
    pulled = pulled_identical(attempt_dir, dest, lay)
    binding, binding_problems = attempt_binding(original, dest)
    cond = {
        'i_collection_runner_error': True,
        'ii_readonly_complete_layout': (fetch.get('method') == 'node_lzma' and len(names) == SU.LAYOUT_FILES
                                        and present == names and all(isinstance(files.get(n), dict) and
                                                                     files[n].get('sha256') for n in names)),
        'iii_mtimes_within_node_windows': len(windows) == SU.LAYOUT_FILES and all(w['within'] for w in windows),
        'iv_pulled_files_identical': (pulled['old_collector_prefix'] and not pulled['differing']
                                      and not pulled['outside_layout']),
    }
    recollection = {                              # X6's contract (supersede.py) + this step's evidence
        'stage': 'collect', 'original_error': original.get('runner_error'), 'conditions': cond,
        'node_records': node_records_by_name(nodes, list(original.get('nodes') or [])),
        'method': METHOD if fetch.get('method') == 'node_lzma' else fetch.get('method'),
        'files': [{'path': w['name'], 'sha256': w['sha256'], 'size': w['size'], 'mtime': w['mtime'],
                   'remote': w['path'], 'source_bytes': w['source_bytes'], 'window': w['window'],
                   'window_basis': w['window_basis'], 'within': w['within']} for w in windows],
        'pulled': [{'path': name, 'sha256': h} for name, h in sorted(pulled['sha256'].items())],
        'rule': 'AMENDMENT_EXT_009 Q1 (D29)', 'step': 'X5 harness supersede (ext/flashnext/recollect.py)',
        'utc': utc_now(), 'error_class': error_class(original),
        'error_class_ruling': RULED_TRANSPORT_TIMEOUTS.get((original.get('run_id'), attempt_id)),
        'collect_stage_evidence': {'release_by_node': original.get('release_by_node'),
                                   'accounting_complete': (original.get('cost') or {}).get('accounting_complete')},
        'pulled_check': {k: v for k, v in pulled.items() if k != 'sha256'},
        'node_records_by_rank': nodes, 'read_via': rdr.nodes[0], 'calls': fetch.get('calls'),
        'sftp_files': fetch.get('sftp_files'), 'seconds': fetch.get('seconds'), 'binding': binding,
        'binding_problems': [d for _, d in binding_problems], 'original_receipt_sha256': _sha(orig_path),
        'recollected_dir': str(dest), 'layout_files': present,
    }
    return {'original': original, 'lay': lay, 'conditions': cond, 'recollection': recollection,
            'binding_problems': binding_problems}


def check(run_dir, attempt_id: str, *, rdr: R.FlashNextRunner | None = None, via: str = READ_VIA) -> dict:
    """Value-blind preflight: (i)-(iv) and (vi) on a temporary re-collection that is deleted afterwards; no judge, no
    record. Returns the conditions and the error class only."""
    rdr = rdr or reader(via)
    tmp = Path(tempfile.mkdtemp(prefix='fnrecheck'))
    try:
        prep = _prepare(Path(run_dir), attempt_id, rdr, tmp / 'r')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    rc = prep['recollection']
    return {'attempt_id': attempt_id, 'conditions': prep['conditions'], 'error_class': rc['error_class'],
            'files': len(rc['files']),
            'pulled': {k: rc['pulled_check'][k] for k in ('files', 'identical', 'old_collector_prefix')},
            'binding_problems': rc['binding_problems']}


def reference_problems(uc: dict, run_dir: Path, original: dict) -> list[str]:
    """The arm's U+C reference: X6's identity checks, the original's node set, and this run's own c0 (the reference
    the arm ran with)."""
    from . import guards as G
    p = list(G.uc_reference_problems(uc))
    if sorted(uc.get('nodes') or []) != sorted(original.get('nodes') or []):
        p.append(f'the reference ran on {uc.get("nodes")}, the screen on {original.get("nodes")}')
    c0 = _json(Path(run_dir) / 'receipts' / 'c0.json')
    if not isinstance(c0, dict) or c0.get('attempt_id') != uc.get('attempt_id') or \
            c0.get(signing.SIGNATURE_FIELD) != uc.get(signing.SIGNATURE_FIELD):
        p.append(f'not this run\'s own U+C reference (receipts/c0.json attempt {(c0 or {}).get("attempt_id")})')
    return p


def supersede(run_dir, attempt_id: str, uc_ref, *, rdr: R.FlashNextRunner | None = None, via: str = READ_VIA,
              dest=None) -> dict:
    """Write the one record for `attempt_id` and return it (see the module docstring). Raises Refused, writing nothing,
    for a screen or a reference outside the rule; otherwise the record is created (exclusively) before it is returned."""
    run_dir = Path(run_dir)
    rdr = rdr or reader(via)
    uc = uc_ref if isinstance(uc_ref, dict) else _json(uc_ref)
    if not isinstance(uc, dict):
        raise Refused(f'{uc_ref}: no U+C reference receipt')
    original = _json(run_dir / 'receipts' / f'{attempt_id}.json') or {}
    bad_ref = reference_problems(uc, run_dir, original)
    if bad_ref:
        raise Refused(f'U+C reference: {bad_ref}')
    dest = Path(dest) if dest else recollected_dir(run_dir, original.get('run_id', 'run'), original.get('candidate_id', 'c'),
                                                   attempt_id)
    prep = _prepare(run_dir, attempt_id, rdr, dest)
    original, cond, recollection = prep['original'], prep['conditions'], prep['recollection']
    if all(cond.values()):
        from . import task as T
        judged = rdr.judge_recollected(dest, prep['lay'], original, J.reference_from_receipt(uc),
                                       T.make_task(original['arm']), extra_rejections=prep['binding_problems'])
        record = SU.supersede_record(original, judged, recollection)
    else:
        record = SU.unrecoverable_record(original, recollection)
    problems = SU.record_problems(original, record)
    if problems:
        raise Refused(f'the record would not be admitted: {problems}')
    path = run_dir / 'receipts' / f'{original["candidate_id"]}{RECORD_SUFFIX[record["collection"]]}'
    with open(path, 'x', encoding='utf-8') as f:                       # (vi): never over an existing record
        json.dump(record, f, indent=1, ensure_ascii=False)
    return {'record': record, 'path': str(path)}


def summary(record: dict) -> dict:
    rc = record.get('recollection') or {}
    return {'attempt_id': record.get('attempt_id'), 'candidate_id': record.get('candidate_id'),
            'collection': record.get('collection'), 'error_class': rc.get('error_class'), 'valid': record.get('valid'),
            'metric_s': (record.get('metric') or {}).get('value'),
            'numerics_class': (record.get('judge') or {}).get('numerics_class'),
            'guard_rejections': [g.get('id') for g in record.get('guard_rejections') or []],
            'invalid_reasons': (record.get('invalid_reasons') or [])[:4], 'conditions': rc.get('conditions'),
            'files': len(rc.get('files') or []),
            'pulled': {k: (rc.get('pulled_check') or {}).get(k) for k in ('files', 'identical', 'old_collector_prefix')},
            'cost_gpu_seconds': (record.get('cost') or {}).get('gpu_seconds')}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(prog='python -m infrarsi.ext.flashnext.recollect')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('proof')
    p.add_argument('--attempt-dir', required=True)
    p.add_argument('--via', default=READ_VIA)
    p.add_argument('--out', default=None, help='write the proof record (JSON) here')
    p = sub.add_parser('check')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--attempt', required=True)
    p.add_argument('--via', default=READ_VIA)
    p = sub.add_parser('supersede')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--attempt', required=True)
    p.add_argument('--uc-ref', required=True, help="the signed U+C reference receipt (this run's receipts/c0.json)")
    p.add_argument('--via', default=READ_VIA)
    a = ap.parse_args()
    if a.cmd == 'proof':
        res = proof(a.attempt_dir, via=a.via)
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1), encoding='utf-8')
        print(json.dumps({k: v for k, v in res.items() if k != 'rows'} | {'differing': [r['name'] for r in res['rows']
                                                                                     if not r['equal']]}, indent=1))
        sys.exit(0 if res['all_equal'] else 1)
    try:
        if a.cmd == 'check':
            print(json.dumps(check(a.run_dir, a.attempt, via=a.via), indent=1, default=str))
            sys.exit(0)
        res = supersede(a.run_dir, a.attempt, a.uc_ref, via=a.via)
    except Refused as exc:
        print(json.dumps({'refused': str(exc)}, indent=1))
        sys.exit(8)
    print(json.dumps({'path': res['path'], **summary(res['record'])}, indent=1, default=str))
