"""W5 summary CLI (CPU only; read-only on records and run code).

  python -m infrarsi.ext.summary.cli --ledger <RUN_LEDGER.json> --out <dir> [--allow-partial]
        [--records <execution/ext_bench>] [--reports <reports/external_bench_20261021>] [--g4-inputs <json>]
        [--r1-mapping <json>]

Writes ENV_A.md, ENV_B.md, ENV_C.md, F1.md, REPRO_SUMMARY.md, summary.json, g4_calibration.json, margin_<W>.json
(+ summary_F1.json from X6's step when F1 records exist) and, last, FREEZE.json (sha256 of every file read and written).
Without --allow-partial the summary is written only when nothing is pending and no problem was found (G-FIN); exit 4
otherwise (nothing written). --allow-partial writes whatever exists, every missing value marked `pending`.
--g4-inputs (AMENDMENT_EXT_005; O provides it): {"final": true at G-FIN, "refprime_pairwise_norms": [{"pair": [..],
"norm_diff": x, "source": ".."}], "locked_dprime_norms": {"<attempt id or arm>": {"norm_cand_dprime_vs_refprime": x,
"source": "<node>:<run>/results/<attempt>_node_result.json"}}, optional "s0_d_norms" / "s0_dprime_norms" to restate the
pre-arm samples}. Receipts are verified with this PC's receipt key (INFRARSI_RECEIPT_KEY or the default path); the
summary never signs anything and never creates a key.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

from ...contracts import EXECUTION_DIR
from . import env_a, env_b, env_c, f1, repro
from . import render as R
from .cells import val
from .records import PRELAUNCH, Store, abandoned, ledger_conflicts

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
AMENDMENTS = tuple(f'AMENDMENT_EXT_00{i}_FABLE51.md' for i in range(1, 10))
DESIGN = ('EXTERNAL_BENCH_DESIGN_FABLE51.md', 'DECISIONS_20261021.md', 'DEVIATION_IDS.md') + AMENDMENTS
# the reused code and frozen data that decide numbers and labels (FREEZE.json `tool`)
REUSED = ('common/trajectory.py', '../scoring/score.py', '../harness/signing.py', 'optico/margin.py', 'optico/task.py',
          'optico/workloads.py', 'optico/params.py', 'rebench_llmf/guards.py', 'rebench_llmf/block.py',
          'rebench_llmf/task.py', 'autoresearch/task.py', 'autoresearch/knobs.py', 'autoresearch/knob_space_C.json',
          'flashnext/cli.py', 'flashnext/arms.py')


def _clean(x):
    """JSON-safe copy: non-finite floats -> None, tuples -> lists, unknown objects -> str."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items() if not str(k).startswith('_')}
    if isinstance(x, (list, tuple, set)):
        return [_clean(v) for v in x]
    if x is None or isinstance(x, (bool, int, str)):
        return x
    return str(x)


def _dump(obj) -> str:
    return json.dumps(_clean(obj), indent=1, ensure_ascii=False, allow_nan=False)


def _strip_env_b(sec: dict) -> dict:
    out = dict(sec)
    out['workloads'] = {}
    for w, W in sec['workloads'].items():
        m = dict(W['margin_record'])
        out['workloads'][w] = {**{k: v for k, v in W.items() if k != 'margin_record'},
                               'margin_record': {k: m.get(k) for k in ('workload', 'estimator', 'n', 'mean', 's', 'rsd',
                                                                       'rsd_used', 'margin', 'pooled_df', 'S_w', 'excluded')}}
    return out


def incomplete_labels(a: dict, b: dict, c: dict) -> list[str]:
    """An `incomplete` label (accounting incomplete) cannot be final: it keeps the summary non-formal."""
    out = []
    labs = [(f'ENV-A {k}', r) for k, r in a['labels'].items()] + [(f'ENV-A {k} (pre-declared G4)', r.get('predeclared') or {})
                                                                   for k, r in a['labels'].items()]
    labs += [(f'ENV-B {w} {k}', r) for w, W in b['workloads'].items() for k, r in W['labels'].items()]
    labs += [(f'ENV-C {k}', r) for k, r in c['labels'].items()]
    for name, r in labs:
        if r.get('label') == 'incomplete':
            out.append(f'{name}: incomplete (accounting: {r.get("accounting_reasons")}); label if complete '
                       f'{r.get("label_if_complete")}')
    return out


def pending_labels(a: dict, b: dict, c: dict) -> list[str]:
    """Every comparison that is still `pending` (ENV-A both G4 readings, ENV-B pairs and environment label, ENV-C)."""
    labs = [(f'ENV-A {k}', r) for k, r in a['labels'].items()] + [(f'ENV-A {k} (pre-declared G4)', r.get('predeclared'))
                                                                   for k, r in a['labels'].items() if r.get('predeclared')]
    labs += [(f'ENV-B {w} {k}', r) for w, W in b['workloads'].items() for k, r in W['labels'].items()]
    labs += [('ENV-B environment label', b['env_label'])] + [(f'ENV-C {k}', r) for k, r in c['labels'].items()]
    return [f'{name}: pending ({r.get("why") or "inputs pending"})' for name, r in labs if (r or {}).get('label') == 'pending']


def accounting_pending(a: dict, b: dict, c: dict) -> list[str]:
    """A finished arm (compared or not, e.g. C-NAT-free) whose trace accounting is incomplete."""
    rows = [(f'ENV-A {k}', r) for k, r in a['rows'].items()]
    rows += [(f'ENV-B {w} {k}', r) for w, W in b['workloads'].items() for k, r in W['arms'].items()]
    rows += [(f'ENV-C {k}', r) for k, r in c['arms'].items()]
    out = []
    for name, r in rows:
        acc = (r.get('search') or {}).get('accounting')
        if acc and acc[0] is False:
            out.append(f'{name} ({r.get("run_id")}): accounting incomplete ({"; ".join(map(str, acc[1]))})')
    return out


def runner_error_problems(a: dict, b: dict, c: dict) -> list[str]:
    """Screens our harness refused or lost inside a result run (they consume budget slots without a measurement): an
    intervention is needed, except D24's pre-launch refusals (nothing launched) in a run whose screen budget did not
    bind - accepted and reported as findings."""
    rows = [(f'ENV-A {k}', r) for k, r in a['rows'].items()]
    rows += [(f'ENV-B {w} {k}', r) for w, W in b['workloads'].items() for k, r in W['arms'].items()]
    rows += [(f'ENV-C {k}', r) for k, r in c['arms'].items()]
    out = []
    for name, r in rows:
        s = r.get('search') or {}
        cel = s.get('n_runner_errors')
        if not (isinstance(cel, dict) and cel.get('v')):
            continue
        errs = dict(cel['v'])
        pre = errs.pop(PRELAUNCH, 0)
        binding = str(val(s.get('stop_reason')) or '') == 'budget:screens'
        if errs or (pre and binding):
            out.append(f'{name} ({r.get("run_id")}): screens refused or lost by our harness {cel["v"]} (intervention: decide '
                       're-run or accept' + ('; the screen budget bound, so D24 does not cover its pre-launch refusals'
                                            if pre and binding else '') + ')')
    return out


def pending_cells(obj, path: str = '') -> list[str]:
    """Paths of every pending cell / label / gate value in a section tree (R3-N4: a formal summary has none)."""
    out = []
    if isinstance(obj, dict):
        if any(obj.get(k) == 'pending' for k in ('status', 'label', 'value')):
            out.append(path or '/')
        for k, v in obj.items():
            out += pending_cells(v, f'{path}/{k}')
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out += pending_cells(v, f'{path}[{i}]')
    return out


def ledger_notes(text: str, notes: dict, env: str) -> str:
    rows = [f'{k}: {v}' for k, v in notes.items() if str(k) == env or str(k).startswith(env + '.')]
    return text + ('\n## Ledger notes (RUN_LEDGER orchestrator_notes)\n\n' + R.bullets(rows) + '\n' if rows else '')


def open_inputs(a: dict) -> list[str]:
    out = []
    cal = a['g4_calibration']
    if not cal.get('final'):
        out.append("--g4-inputs: the pairwise norms among the three ENV-A arms' cached ref'(D') models under "
                   "<run>/ref_permuted/ (score.py's compute_norm_and_max_diff, trusted CPU process; AMENDMENT_EXT_005) "
                   "and \"final\": true at G-FIN; until then limit_D' = 6.17 (provisional)")
    for arm, r in a['rows'].items():
        c = (r.get('locked') or {}).get('g4_dprime_norm')
        if isinstance(c, dict) and c.get('status') == 'pending':
            out.append(f"--g4-inputs locked_dprime_norms[{arm}]: {c.get('why')}")
    return out


def main(argv=None) -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(prog='infrarsi.ext.summary.cli', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ledger', required=True, help='results/RUN_LEDGER.json')
    ap.add_argument('--out', required=True, help='output directory (created)')
    ap.add_argument('--allow-partial', action='store_true', help='write what exists; missing values are pending')
    ap.add_argument('--records', default=None, help='the ext_bench records root (default <EXECUTION_DIR>/ext_bench)')
    ap.add_argument('--reports', default=None, help='reports/external_bench_20261021 (default: the ledger\'s grandparent)')
    ap.add_argument('--g4-inputs', default=None, help='AMENDMENT_EXT_005 inputs (see above)')
    ap.add_argument('--r1-mapping', default=None, help=f'R1 mapping (default {repro.MAPPING.name} of this module)')
    args = ap.parse_args(argv)
    ledger_path = Path(args.ledger).resolve()
    reports = Path(args.reports).resolve() if args.reports else ledger_path.parent.parent
    root = Path(args.records).resolve() if args.records else EXECUTION_DIR / 'ext_bench'
    out = Path(args.out).resolve()
    if not ledger_path.is_file():
        print(f'no ledger at {ledger_path}', file=sys.stderr)
        return 2
    store = Store(root)
    if not store.key_present:
        print('warning: no receipt key on this PC: signatures cannot be verified (the summary is not formal)', file=sys.stderr)
    ledger = json.loads(store.text(ledger_path, role='ledger'))
    g4 = json.loads(store.text(Path(args.g4_inputs), role='input')) if args.g4_inputs else None
    missing_design = [n for n in DESIGN if store.read_bytes(reports / 'design' / n, role='design') is None]
    ab = abandoned(ledger, root)
    a = env_a.build(store, root, ledger, abandoned=ab, g4_inputs=g4)
    b = env_b.build(store, root, ledger, abandoned=ab)
    c = env_c.build(store, root, ledger, abandoned=ab)
    with tempfile.TemporaryDirectory(prefix='x7f1') as tmp:
        fsec = f1.build(store, root, ledger, Path(tmp), allow_partial=args.allow_partial)
        frows = f1.rows(fsec)
        f1_file = (Path(tmp) / 'summary_F1.json').read_bytes() if (Path(tmp) / 'summary_F1.json').is_file() else None
        margin_f1 = (Path(tmp) / 'margin_F1.json').read_bytes() if (Path(tmp) / 'margin_F1.json').is_file() else None
    rp = repro.build(store, root, reports, env_b=b, f1sec=fsec, f1rows=frows,
                     mapping_path=Path(args.r1_mapping) if args.r1_mapping else None)
    pending = (a['pending'] + b['pending'] + c['pending'] + fsec['pending'] + rp['pending'] + incomplete_labels(a, b, c)
               + pending_labels(a, b, c) + accounting_pending(a, b, c))
    problems = sorted(set(store.problems + a['problems'] + b['problems'] + c['problems'] + fsec['problems'] + rp['problems']
                          + ledger_conflicts(ledger, root) + runner_error_problems(a, b, c)
                          + [f'design file missing: {n} (cited authority)' for n in missing_design]))
    if not store.key_present:
        problems.append('no receipt key: signatures not verified')
    inputs = open_inputs(a)
    notes = ledger.get('orchestrator_notes') if isinstance(ledger.get('orchestrator_notes'), dict) else {}
    docs = {'ENV_A.md': ledger_notes(env_a.markdown(a), notes, 'ENV-A'), 'ENV_B.md': ledger_notes(env_b.markdown(b), notes, 'ENV-B'),
            'ENV_C.md': ledger_notes(env_c.markdown(c), notes, 'ENV-C'), 'F1.md': ledger_notes(f1.markdown(fsec), notes, 'F1'),
            'REPRO_SUMMARY.md': repro.markdown(rp)}
    formal = not pending and not problems
    if formal:                  # R3-N4 guard: nothing pending may remain anywhere under a formal summary
        stray = pending_cells({'ENV-A': a, 'ENV-B': b, 'ENV-C': c, 'F1': {**fsec, 'rows': frows}, 'repro': rp})
        stray += [f'{n}: a table cell reads pending' for n, t in docs.items() if '| pending |' in t]
        if stray:
            problems.append(f'tool: {len(stray)} pending value(s) under an otherwise formal summary (not formal): {stray[:12]}')
            formal = False
    if not formal and not args.allow_partial:
        print(f'summary refused (exit 4): {len(pending)} pending, {len(problems)} problems; use --allow-partial to write a '
              'partial summary')
        for x in pending + problems:
            print(' -', x)
        return 4
    out.mkdir(parents=True, exist_ok=True)
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    head = (f'<!-- generated by infrarsi.ext.summary.cli at {now}; '
            f"{'FORMAL (G-FIN)' if formal else 'PARTIAL: values marked pending are not yet in the records'} -->\n\n")
    written = {}
    for name, text in docs.items():
        written[name] = head + text
    summary = {'schema': 'ext-bench-summary-v1', 'generated_utc': now, 'formal': formal, 'partial': not formal,
               'ledger': str(ledger_path), 'records_root': str(root),
               'cell_format': "{'v': value, 'src': ['<run_id>:<attempt_id>' | '<run_id>/<file>' | ruling]} or "
                              "{'v': null, 'status': 'pending' | 'n/a', 'why': ...}",
               'ENV-A': a, 'ENV-B': _strip_env_b(b), 'ENV-C': c,
               'F1': {k: v for k, v in fsec.items() if k != 'summary'} | {'rows': frows, 'labels': (fsec.get('summary') or {}).get('labels')},
               'repro': rp, 'abandoned': ab, 'pending': pending, 'problems': problems, 'open_inputs': inputs,
               'ledger_notes': notes,
               'findings': a['findings'] + b['findings'] + c['findings']}
    written['summary.json'] = _dump(summary)
    written['g4_calibration.json'] = _dump(a['g4_calibration'])
    for w, W in b['workloads'].items():
        written[f'margin_{w}.json'] = _dump(W['margin_record'])
    blobs = {k: v.encode('utf-8') for k, v in written.items()}
    if f1_file is not None:
        blobs['summary_F1.json'] = f1_file
    if margin_f1 is not None:
        blobs['margin_F1.json'] = margin_f1
    for name, data in blobs.items():
        (out / name).write_bytes(data)
    tool = sorted(p for p in HERE.iterdir() if p.suffix in ('.py', '.json')) + [(EXT / r).resolve() for r in REUSED]
    freeze = {'schema': 'ext-bench-freeze-v1', 'generated_utc': now, 'formal': formal, 'records_root': str(root),
              'amendments_cited': [{'file': n, 'sha256': (store.files.get(str((reports / 'design' / n).resolve())) or {}).get('sha256')}
                                   for n in AMENDMENTS],
              'f1_ext008': fsec.get('ext008'),                     # AMENDMENT_EXT_008's freeze record
              'ext009_superseding_records': fsec.get('ext009') or [],   # AMENDMENT_EXT_009 Q5: the superseding attempts
              'files': sorted(store.files.values(), key=lambda f: (f['role'], f['path'])),
              'receipts': {'total': sum(1 for f in store.files.values() if f['role'] == 'receipt'),
                           **{s: sum(1 for f in store.files.values() if f.get('signature') == s)
                              for s in ('verified', 'failed', 'no_key')}},
              'tool': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in tool],
              'outputs': [{'name': n, 'sha256': hashlib.sha256(d).hexdigest()} for n, d in sorted(blobs.items())]}
    (out / 'FREEZE.json').write_text(_dump(freeze), encoding='utf-8')
    print(json.dumps({'out': str(out), 'formal': formal, 'pending': len(pending), 'problems': len(problems),
                      'open_inputs': len(inputs), 'files_hashed': len(store.files),
                      'findings': a['findings'] + b['findings'] + c['findings'],
                      'labels': {'ENV-A': a['labels']['A-ISW vs A-NAT'].get('label'), 'ENV-B': b['env_label']['label'],
                                 'ENV-C': c['labels']['C-IS vs C-NAT-infra'].get('label'),
                                 'F1': ((fsec.get('summary') or {}).get('labels') or {}).get('F-A0 vs F-OPTICO', {}).get('label', 'pending')}},
                     indent=1, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
