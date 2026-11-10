"""Read-only access to the run records and RUN_LEDGER.json.

Every file the summary reads goes through `Store`, which keeps its sha256 for FREEZE.json; every ext receipt is checked
with `harness.signing.verify` (the PC's receipt key; when the key file is absent nothing is verified and no key is
created). Nothing is ever written under the records root. The A-NAT transcript (`outputs/`, `calls/`, the `turn`
events) is never read: only result / trace / receipt files and the first event's timestamp of a running run.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import fnmatch
import hashlib
import json
import re
from pathlib import Path

from ...harness import signing
from .cells import cell, finite

ENV_DIRS = {'ENV-A': 'rebench_llmf', 'ENV-B': 'optico', 'ENV-C': 'autoresearch', 'F1': 'flashnext'}
SUPERSEDE_KINDS = ('recollected_readonly', 'unrecoverable')                 # AMENDMENT_EXT_009 (X6's supersede.KINDS)
SUPERSEDE_MARKS = ('superseded_by', signing.SIGNATURE_FIELD)                 # an original's mark and its re-signature


def is_superseding(r) -> bool:
    """An EXT_009 superseding record (X6's supersede.is_record)."""
    return isinstance(r, dict) and r.get('collection') in SUPERSEDE_KINDS and bool(r.get('supersedes'))
_EVENT_KIND = re.compile(r'"kind"\s*:\s*"(llm_call|screen|rescore)"')
SIG_VERIFIED, SIG_FAILED, SIG_NO_KEY = 'verified', 'failed', 'no_key'


@dataclasses.dataclass
class ScoreTask:
    """The fields ext.common.trajectory reads (env, margin, tau, budget = the caps of AUC_budget's x axis)."""
    id: str
    env: str
    arm: str
    margin: dict
    tau: float
    budget: dict


def utc(x) -> _dt.datetime | None:
    if isinstance(x, (int, float)) and finite(x):
        return _dt.datetime.fromtimestamp(float(x), _dt.timezone.utc)
    try:
        d = _dt.datetime.fromisoformat(str(x).replace('Z', '+00:00'))
    except (TypeError, ValueError):
        return None
    return d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)


class Store:
    def __init__(self, root, *, verify: bool = True):
        self.root = Path(root)
        self.files: dict[str, dict] = {}          # path -> {'sha256', 'role', 'bytes'[, 'signature', 'attempt_id']}
        self.key_present = signing.key_path().exists()
        self.verify = verify
        self.problems: list[str] = []
        self._receipts: dict[str, dict] = {}      # run dir -> {attempt_id: receipt}
        self._superseding: dict[str, dict] = {}   # run dir -> {original attempt_id: [superseding records]}

    # ---------------------------------------------------------------- files
    def rel(self, p) -> str:
        p = Path(p)
        try:
            return p.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return str(p)

    def read_bytes(self, p, role: str = 'record') -> bytes | None:
        p = Path(p)
        if not p.is_file():
            return None
        b = p.read_bytes()
        key = str(p.resolve())
        self.files.setdefault(key, {'path': key, 'sha256': hashlib.sha256(b).hexdigest(), 'role': role, 'bytes': len(b)})
        return b

    def json(self, p, role: str = 'record'):
        b = self.read_bytes(p, role)
        if b is None:
            return None
        try:
            return json.loads(b.decode('utf-8-sig'))
        except ValueError as exc:
            self.problems.append(f'{self.rel(p)}: not JSON ({exc})')
            return None

    def text(self, p, role: str = 'ledger') -> str | None:
        b = self.read_bytes(p, role)
        return None if b is None else b.decode('utf-8-sig')

    def first_event_utc(self, p) -> str | None:
        """The 'utc' of the first line of an events.jsonl (nothing else of the file is kept or hashed)."""
        p = Path(p)
        if not p.is_file():
            return None
        with p.open('r', encoding='utf-8') as f:
            line = f.readline()
        try:
            return json.loads(line).get('utc')
        except (ValueError, AttributeError):
            return None

    def event_sequence(self, p) -> list | None:
        """[('call', call_id) | ('screen', index)] of an InfraRSI search's events.jsonl in producer order - the
        explicit call-to-row boundary (the search charges a call before the screen event that follows it; a rescore
        - the engine's confirmation run - is a trace row too, its event the row's boundary). Only the event kind and
        these two ids are kept; nothing else of the file is read into the summary."""
        b = self.read_bytes(p, role='record')
        if b is None:
            return None
        out = []
        for line in b.decode('utf-8', errors='replace').splitlines():
            if not _EVENT_KIND.search(line):               # turn / candidate / ... lines are never parsed
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if isinstance(e, dict) and e.get('kind') in ('llm_call', 'screen', 'rescore'):
                pl = e.get('payload') if isinstance(e.get('payload'), dict) else {}
                out.append(('call', pl.get('call_id')) if e['kind'] == 'llm_call' else ('screen', pl.get('index')))
        return out

    # ---------------------------------------------------------------- receipts
    def signature(self, r: dict) -> str:
        if not self.verify or not self.key_present:
            return SIG_NO_KEY
        return SIG_VERIFIED if signing.verify(r) else SIG_FAILED

    def receipt_file(self, p) -> dict | None:
        r = self.json(p, role='receipt')
        if not isinstance(r, dict) or r.get('schema') != 'ext-receipt-v1':
            return None
        info = self.files[str(Path(p).resolve())]
        info['attempt_id'] = r.get('attempt_id')
        info['signature'] = self.signature(r)
        if info['signature'] == SIG_FAILED:
            self.problems.append(f'{self.rel(p)}: receipt signature does not verify')
        return r

    def receipts(self, run_dir) -> dict[str, dict]:
        """{attempt_id: receipt} of a run directory (receipts/*.json, runner/receipts/*.json, receipt_*.json);
        identical copies of one attempt count once, differing copies are a problem (the first is kept). AMENDMENT_EXT_009:
        a superseding record (`collection` recollected_readonly / unrecoverable with `supersedes`, the same attempt id)
        is kept apart (`superseding()`), never as the attempt's receipt; the original's copies compare without their
        supersede marks (`superseded_by` and the re-signature), and the unmarked copy is the receipt."""
        run_dir = Path(run_dir)
        key = str(run_dir.resolve())
        if key in self._receipts:
            return self._receipts[key]
        out, text, sup = {}, {}, {}
        files = sorted(run_dir.glob('receipts/*.json')) + sorted(run_dir.glob('runner/receipts/*.json')) \
            + sorted(run_dir.glob('receipt_*.json'))
        for f in files:
            r = self.receipt_file(f)
            if r is None:
                continue
            if is_superseding(r):
                sup.setdefault(str(r.get('supersedes')), []).append({'record': r, 'path': str(f.resolve())})
                continue
            att = r.get('attempt_id')
            canon = json.dumps({k: v for k, v in r.items() if k not in SUPERSEDE_MARKS}, sort_keys=True, default=str)
            if att in out:
                if text[att] != canon:
                    self.problems.append(f'{self.rel(f)}: a second, different receipt for attempt {att}')
                elif out[att].get('superseded_by') and not r.get('superseded_by'):
                    out[att] = r                            # the unmarked copy is the receipt
                continue
            out[att], text[att] = r, canon
        self._receipts[key] = out
        self._superseding[key] = sup
        return out

    def superseding(self, run_dir) -> dict[str, list]:
        """{original attempt_id: [{'record', 'path'}]} - the EXT_009 superseding records of a run directory."""
        self.receipts(run_dir)
        return self._superseding.get(str(Path(run_dir).resolve()), {})

    def sig_of(self, run_dir, attempt_id) -> str | None:
        for info in self.files.values():
            if info.get('attempt_id') == attempt_id and info['path'].startswith(str(Path(run_dir).resolve())):
                return info.get('signature')
        return None


def src(run_id: str, receipt: dict | None) -> str | None:
    return f'{run_id}:{receipt.get("attempt_id")}' if isinstance(receipt, dict) else None


NORMAL_STOPS = ('finish', 'loop_finished', 'space_exhausted', 'no_valid_sample')


def findings(env: str, arm: str, rid: str, trace: dict, runner_errors: dict | None = None,
             accounting: tuple | None = None) -> list[str]:
    """Facts of a finished run that a reader of the label must see: a stop rule other than a budget / the method's own
    end, screens our harness refused or lost, incomplete accounting (which makes every label of the arm `incomplete`).
    `accounting` = (complete, reasons, d26 basis) when the caller applied D26 (ENV-B)."""
    from ..common import trajectory as TJ
    out = []
    sr = str(trace.get('stop_reason') or '')
    if sr and not (sr.startswith('budget') or sr in NORMAL_STOPS):
        out.append(f'{env} {arm} ({rid}): stop_reason {sr} after {(trace.get("used") or {}).get("screens")} screens of '
                   f'{(trace.get("budget") or {}).get("screens")}')
    n = sum((runner_errors or {}).values())
    if n:
        out.append(f'{env} {arm} ({rid}): {n} screen(s) refused or lost by our harness {dict(runner_errors)} (not '
                   'experiments; the trace still counts them as screens'
                   + ('; `prelaunch` = refused before anything was launched, D24' if (runner_errors or {}).get(PRELAUNCH)
                      else '') + ')')
    complete, why = accounting[:2] if accounting else TJ.accounting(trace)
    if not complete:
        out.append(f'{env} {arm} ({rid}): accounting incomplete ({"; ".join(why)}): its labels read `incomplete`')
    elif accounting and accounting[2]:
        out.append(f'{env} {arm} ({rid}): the trace flags accounting incomplete; complete under {accounting[2]}')
    return out


_NEAR_NOTE = re.compile(r'an LLM call ended within \d+ s of a row \(attribution ambiguous\)')
PRE_LAUNCH_PHASES = frozenset({'total', 'materialize_s', 'upload_s', 'acquire_s', 'lease_s'})
_TOL_GPUH = 1e-6                                  # GPU-hours: the ledgers add floats of ms-precision receipt seconds
_TOL_USD = 1e-6


def _nonneg(x) -> bool:
    return finite(x) and x >= 0


def charged_prelaunch_refusal(r: dict | None) -> bool:
    """D24 / D26(a): our harness refused the attempt before anything was launched and charged the held interval. Both
    receipt forms must show valid false, no judge, no metric value, only pre-launch phases (none of trial / monitor /
    iterations / collect / wait), every refused remote call in pre_launch, and a charged cost (gpu_seconds >= 0 with
    its basis). The 13:26 form: runner_error AttemptDeadline in pre_launch with at least one refused pre_launch call;
    the post-14:05 form: `not_launched: 'pre_launch'` (set only when no launch was attempted) with an `infra_refusal:`
    reason."""
    if not isinstance(r, dict):
        return False
    c = r.get('cost') if isinstance(r.get('cost'), dict) else {}
    phases = r.get('phases') if isinstance(r.get('phases'), dict) else {}
    dl = r.get('deadline') if isinstance(r.get('deadline'), dict) else {}
    calls = dl.get('refused_calls') or []
    if not (r.get('valid') is False and r.get('judge') is None and (r.get('metric') or {}).get('value') is None
            and set(phases) <= PRE_LAUNCH_PHASES and _nonneg(c.get('gpu_seconds')) and c.get('basis')
            and isinstance(calls, list) and all(isinstance(x, dict) and x.get('phase') == 'pre_launch' for x in calls)):
        return False
    if r.get('not_launched') == 'pre_launch':
        return str((r.get('invalid_reasons') or [''])[0]).startswith('infra_refusal')
    e = r.get('runner_error') if isinstance(r.get('runner_error'), dict) else {}
    return e.get('type') == 'AttemptDeadline' and 'pre_launch' in str(e.get('message') or '') and bool(calls)


PRODUCERS = {                                      # the trace each arm's producer writes (R6-N2: chosen by the arm)
    'infrarsi': ('infrarsi-search-trace-v1', lambda mode: mode != 'native'),
    'infra' + 'smith': ('infrarsi-search-trace-v1', lambda mode: mode != 'native'),     # SearchSession: GPU = receipts
    'native': ('infrarsi-search-trace-v1', lambda mode: mode == 'native'),         # OptiCo's own loop (B-NAT): lease
    'agent': ('ext-basic-agent-trace-v1', lambda mode: True),                       # basic agent (A-NAT, C-NAT-*)
}


def accounting_checks(trace: dict, receipts: dict, events: list | None = None, *, non_screen: dict | None = None,
                      producer: str = 'infrarsi', allow_incomplete: frozenset = frozenset()) -> tuple[list[str], dict]:
    """(failures, reconciliation) - the accounting checks of a finished search arm (Codex R5 / R6, D27), whatever the
    producer's flags say, against the declared basis of the arm's producer (chosen by the caller from the arm):
      producer   the trace's schema / mode must be the producer the arm expects (a mismatch is a record error);
      coverage   screens are 0..n-1; every screen has its receipt; every unique receipt of the run directory is used by
                 exactly one screen or is a declared non-screen charge (`non_screen` {attempt_id: label}: ENV-A's locked
                 re-score, F1's start reference and locked halves; reported apart, not part of the search totals); a
                 declared id must have its receipt and must not be a screen receipt;
      explicit   every receipt has a cost with gpu_seconds >= 0 and accounting_complete True - only a receipt the D26
                 cause recognition allowlisted (`allow_incomplete`, never for a producer-complete trace) may say
                 false; every LLM call record is complete with int tokens >= 0 and USD >= 0;
      finite     GPU-hours and wall >= 0 per row and final, tokens and USD too for an LLM arm (an arm with an LLM
                 budget); an arm without one must have no call record and explicit zero tokens and USD on every row
                 and in `used`; every cumulative dimension monotone and <= the final;
      GPU        InfraRSI's ledger / the basic agent: per-row cumulative = the running sum of the receipts'
                 gpu_seconds, final = their sum; the native B-NAT lease: per-row cumulative >= the running sum;
      LLM        final tokens / USD = the call log; per row = the calls before the row by an explicit boundary - B-NAT:
                 each call's `after_screen`; InfraRSI: the order of the `llm_call` / `screen` events (events.jsonl,
                 call ids = the trace's call list); no usable boundary fails (never inferred) - except the basic agent
                 (D27: A-NAT, C-NAT-free, C-NAT-infra), whose per-row attribution is reported as producer-asserted."""
    fails: list[str] = []
    schema, mode_ok = PRODUCERS[producer]
    native, rows_asserted = producer == 'native', producer == 'agent'
    scr = trace.get('screens') or []
    used = trace.get('used') or {}
    if trace.get('schema') not in (schema, schema.replace('infrarsi-', 'infra' + 'smith-')) or not mode_ok(trace.get('mode')):
        fails.append(f'trace schema / mode {trace.get("schema")} / {trace.get("mode")} is not the arm\'s producer '
                     f'({producer}: {schema})')
    calls = trace.get('llm_calls', [])
    if not isinstance(calls, list):
        fails.append('llm_calls is not a list')
        calls = []
    llm = bool((trace.get('budget') or {}).get('llm_tokens'))
    ids = [x.get('attempt_id') for x in scr]
    charge = {a: ((r.get('cost') or {}).get('gpu_seconds') if isinstance(r.get('cost'), dict) else None)
              for a, r in receipts.items()}
    non_screen = dict(non_screen or {})
    recon = {'producer': producer, 'gpu_basis': 'lease (>= the receipts)' if native else 'sum of the receipt charges (=)',
             'n_screens': len(scr), 'n_receipts': len(receipts), 'n_calls': len(calls),
             'non_screen_charges': {f'{lab} ({a})': round(charge[a] / 3600.0, 6) if _nonneg(charge.get(a)) else None
                                    for a, lab in non_screen.items()}}
    if any(x.get('index', k) != k for k, x in enumerate(scr)):
        fails.append('screen indices are not 0..n-1')
    if [k for k, a in enumerate(ids) if a not in receipts]:
        fails.append(f'screens without a receipt: {[k for k, a in enumerate(ids) if a not in receipts][:5]}')
    if sorted({a for a in ids if a is not None and ids.count(a) > 1}):
        fails.append(f'receipts used by several screens: {sorted({a for a in ids if a is not None and ids.count(a) > 1})[:5]}')
    if sorted(set(non_screen) - set(receipts)):
        fails.append(f'declared non-screen charges without a receipt: {sorted(set(non_screen) - set(receipts))[:5]}')
    if sorted(set(non_screen) & set(ids)):
        fails.append(f'declared non-screen charges that are screen receipts: {sorted(set(non_screen) & set(ids))[:5]}')
    unref = sorted(set(receipts) - set(ids) - set(non_screen))
    if unref:
        fails.append(f'receipts no screen references: {unref[:5]} ('
                     f'{sum(charge[a] for a in unref if _nonneg(charge[a])) / 3600.0:.6f} GPUh not in any row)')
    for a, r in receipts.items():
        c = r.get('cost')
        if not (isinstance(c, dict) and type(c.get('accounting_complete')) is bool and _nonneg(c.get('gpu_seconds'))):
            fails.append(f'receipt {a}: cost / completeness not explicit')
            break
        if c['accounting_complete'] is not True and a not in allow_incomplete:
            fails.append(f'receipt {a}: cost accounting_complete false (not a D26-recognised refusal)')
            break
    for k, c in enumerate(calls):
        if not (isinstance(c, dict) and c.get('accounting_complete') is True and type(c.get('llm_tokens')) is int
                and c['llm_tokens'] >= 0 and _nonneg(c.get('llm_usd'))):
            fails.append(f'LLM call record {k}: tokens / USD / completeness not explicit and complete')
            break
    rows = [x.get('cum') if isinstance(x.get('cum'), dict) else {} for x in scr]
    if not llm:                                         # R6-N3: no LLM budget = no call, explicit zero tokens and USD
        nz = [(k, d) for k, x in enumerate(rows + [used]) for d, z in (('llm_tokens', 0), ('llm_usd', 0.0))
              if not (type(x.get(d)) in ((int,) if d == 'llm_tokens' else (int, float)) and x.get(d) == z)]
        if calls or nz:
            fails.append(f'an arm without an LLM budget shows LLM use or no explicit zero (calls {len(calls)}, '
                         f'rows / final not zero: {nz[:4]})')
    dims = ('gpu_hours', 'wall_seconds') + (('llm_tokens', 'llm_usd') if llm else ())
    bad = [(k, d) for k, x in enumerate(rows + [used]) for d in dims if not _nonneg(x.get(d))]
    if bad:
        fails.append(f'non-finite or negative totals (row, dimension): {bad[:4]}')
    if fails:
        recon['mismatches'] = list(fails)
        return fails, recon
    tol = {'gpu_hours': _TOL_GPUH, 'wall_seconds': 1e-6, 'llm_tokens': 0, 'llm_usd': _TOL_USD}
    for d in dims:
        seq = [x[d] for x in rows]
        if any(b < a - tol[d] for a, b in zip(seq, seq[1:])) or (seq and seq[-1] > used[d] + tol[d]):
            fails.append(f'cumulative {d} decreases or exceeds the final {used[d]}')
    run = 0.0
    for k, x in enumerate(rows):
        run += charge[ids[k]] / 3600.0
        if x['gpu_hours'] < run - _TOL_GPUH or (not native and abs(x['gpu_hours'] - run) > _TOL_GPUH):
            fails.append(f'row {k}: cumulative {x["gpu_hours"]:.6f} GPUh vs the receipts\' running sum {run:.6f}')
            break
    if not native and abs(used['gpu_hours'] - run) > _TOL_GPUH:
        fails.append(f'final {used["gpu_hours"]:.6f} GPUh vs the receipts\' sum {run:.6f}')
    recon.update(gpu_hours_receipts=round(run, 6), gpu_hours_trace=used['gpu_hours'])
    if llm:
        tok, usd = sum(c['llm_tokens'] for c in calls), sum(c['llm_usd'] for c in calls)
        recon.update(llm_tokens_call_log=tok, llm_tokens_trace=used['llm_tokens'], llm_usd_call_log=round(usd, 6),
                     llm_usd_trace=used['llm_usd'])
        if used['llm_tokens'] != tok or abs(used['llm_usd'] - usd) > _TOL_USD:
            fails.append(f'final {used["llm_tokens"]} tokens / ${used["llm_usd"]} vs the call log {tok} / ${usd:.6f}')
        before = None                                   # per row: the calls before it, by the producer's boundary
        if native:
            if all('after_screen' in c and (c['after_screen'] is None or (type(c['after_screen']) is int
                                                                          and 0 <= c['after_screen'] < len(scr)))
                   for c in calls):
                before = [[c for c in calls if (-1 if c['after_screen'] is None else c['after_screen']) < k]
                          for k in range(len(scr))]
                recon['row_boundary'] = 'after_screen of each call'
        elif not rows_asserted and events is not None:
            ev_calls = [key for kind, key in events if kind == 'call']
            ev_rows = [key for kind, key in events if kind == 'screen']
            if ev_calls == [c.get('call_id') for c in calls] and ev_rows == list(range(len(scr))) \
                    and len(set(ev_calls)) == len(ev_calls) and None not in ev_calls:
                by_id = {c['call_id']: c for c in calls}
                before, acc = [None] * len(scr), []
                for kind, key in events:
                    if kind == 'call':
                        acc.append(by_id[key])
                    else:
                        before[key] = list(acc)
                recon['row_boundary'] = 'events.jsonl order (llm_call before screen)'
        if before is None and rows_asserted:
            recon['row_boundary'] = 'producer-asserted (D27): the trace carries per-row counters but no call-to-row boundary'
        elif before is None:
            fails.append('no explicit call-to-row boundary for the per-row LLM totals'
                         + ('' if native else ' (events.jsonl missing or its call / screen events differ from the trace)'))
        else:
            for k, x in enumerate(rows):
                t, u = sum(c['llm_tokens'] for c in before[k]), sum(c['llm_usd'] for c in before[k])
                if x['llm_tokens'] != t or abs(x['llm_usd'] - u) > _TOL_USD:
                    fails.append(f'row {k}: cumulative {x["llm_tokens"]} tokens / ${x["llm_usd"]} vs the calls before it '
                                 f'{t} / ${u:.6f}')
                    break
    recon['mismatches'] = list(fails)
    return fails, recon


def _d26_cause(trace: dict, receipts: dict) -> dict | None:
    """D26's cause recognition, only for a trace whose producer raised an accounting flag: >= 1 flagged screen and the
    flags a suffix that starts exactly where a recognised cause starts - (a) a charged D24 pre-launch refusal (every
    incomplete-cost receipt must be one), (b) exactly one producer note, "an LLM call ended within N s of a row
    (attribution ambiguous)", with exactly one near-boundary call, the shim log matching, and the flags from its
    after_screen + 1; no other note. Returns {'refusal_at': [...], 'refusal_ids': [...], 'note': str | None}, or None."""
    scr = trace.get('screens') or []
    flags = [(x.get('cum') or {}).get('accounting_complete') for x in scr]
    if not scr or any(type(f) is not bool for f in flags):
        return None
    first = next((k for k, f in enumerate(flags) if f is False), None)
    if first is None or not all(f is False for f in flags[first:]):
        return None
    unknown = {a for a, r in receipts.items()
               if not (isinstance(r.get('cost'), dict) and r['cost'].get('accounting_complete') is True)}
    if not all(charged_prelaunch_refusal(receipts[a]) for a in unknown):
        return None
    calls = trace.get('llm_calls') if isinstance(trace.get('llm_calls'), list) else []
    ex = trace.get('extra') if isinstance(trace.get('extra'), dict) else {}
    notes = [str(n) for n in ex.get('accounting_notes') or []]
    near = [c for c in calls if isinstance(c, dict) and c.get('near_row_boundary')]
    refusal_at = [k for k, x in enumerate(scr) if x.get('attempt_id') in unknown]
    if len(refusal_at) != len(unknown):                 # an allowlisted refusal must be a screen of this trace
        return None
    starts = list(refusal_at)
    if notes or near:
        if (len(notes) != 1 or not _NEAR_NOTE.fullmatch(notes[0]) or len(near) != 1
                or ex.get('calls_near_row_boundary', 1) != 1 or type(near[0].get('after_screen')) is not int
                or (ex.get('llm_log') or {}).get('matches_shim') is not True):
            return None
        starts.append(near[0]['after_screen'] + 1)
    if not starts or first != min(starts):
        return None
    return {'refusal_at': refusal_at, 'refusal_ids': sorted(unknown), 'note': notes[0] if notes else None}


def accounting_verified(trace: dict, receipts: dict, events: list | None = None, *, non_screen: dict | None = None,
                        producer: str = 'infrarsi', d26: bool = False, extra_fails: list | None = None) -> tuple:
    """(complete, reasons, D26 basis, reconciliation) of a finished search arm in any environment (D27). The accounting
    checks (accounting_checks) run for every arm: a producer-complete trace that fails them is incomplete with the
    failures named ("accounting checks failed: ..."). A producer flag stays, except - with d26 (ENV-B) - a recognised
    D26 cause (_d26_cause, run before the checks: only its refusal receipts may carry accounting_complete false) whose
    numbers hold: the checks pass and every refusal charge lies inside its own screen's increment; otherwise
    "D26 not granted: ..." names the numbers (never recomputed). extra_fails: record errors the caller found (F1's
    declarations)."""
    from ..common import trajectory as TJ
    complete, why = TJ.accounting(trace)
    cause = _d26_cause(trace, receipts) if d26 and not complete else None
    fails, recon = accounting_checks(trace, receipts, events, non_screen=non_screen, producer=producer,
                                     allow_incomplete=frozenset(cause['refusal_ids']) if cause else frozenset())
    fails = list(extra_fails or []) + fails
    recon['mismatches'] = list(fails)
    if complete:
        if fails:
            return False, [f'accounting checks failed: {"; ".join(fails[:3])}'], None, recon
        return True, [], None, recon
    if cause is None:
        return False, why + ([f'accounting checks failed: {"; ".join(fails[:3])}'] if fails else []), None, recon
    scr = trace['screens']
    gpu = 0.0
    for k in cause['refusal_at']:                         # the refusal's charge inside its own screen's increment
        g = receipts[scr[k]['attempt_id']]['cost']['gpu_seconds'] / 3600.0
        cur = (scr[k].get('cum') or {}).get('gpu_hours')
        prev = (scr[k - 1].get('cum') or {}).get('gpu_hours') if k else 0.0
        if not (_nonneg(cur) and _nonneg(prev)) or cur - prev < g - _TOL_GPUH:
            fails.append(f'row {k}: the refusal charge {g:.6f} GPUh is not in its increment '
                         f'{(cur or 0) - (prev or 0):.6f}')
        gpu += g
    recon['refusal_charges_gpuh'] = {scr[k]['candidate_id']: round(receipts[scr[k]['attempt_id']]['cost']['gpu_seconds']
                                                                   / 3600.0, 6) for k in cause['refusal_at']}
    recon['mismatches'] = list(fails)
    if fails:
        return False, why + [f'D26 not granted: the trace totals do not reconcile ({"; ".join(fails[:3])})'], None, recon
    basis = []
    if cause['refusal_at']:
        basis.append(f'D26(a): {len(cause["refusal_at"])} D24 pre-launch refusal receipt(s) charged as recorded '
                     f'({gpu:.2f} GPUh, upper bound; in the trace totals)')
    if cause['note']:
        basis.append(f'D26(b): "{cause["note"]}" (totals = the call log: {recon.get("llm_tokens_call_log")} tokens; '
                     'attribution as implemented)')
    return True, [], '; '.join(basis), recon


def accounting_d26(trace: dict, receipts: dict, events: list | None = None, *, producer: str = 'infrarsi') -> tuple:
    """ENV-B: accounting_verified with D26."""
    return accounting_verified(trace, receipts, events, producer=producer, d26=True)


def accounting_text(acc: tuple | None, d26: str | None = None) -> str:
    """The `accounting` column of a search arm: complete / complete (D26 basis) / incomplete (reasons); the per-row
    LLM attribution says producer-asserted (D27) where the producer records no call-to-row boundary."""
    if not acc:
        return 'n/a'
    ok, why = acc[0], acc[1]
    rec = acc[3] if len(acc) > 3 and isinstance(acc[3], dict) else {}
    if not ok:
        return 'incomplete (' + '; '.join(map(str, why)) + ')'
    extra = [x for x in (d26, 'per-row LLM attribution producer-asserted (D27)'
                         if str(rec.get('row_boundary') or '').startswith('producer-asserted') else None) if x]
    return 'complete' + (f" ({'; '.join(extra)})" if extra else '')


PRELAUNCH = 'prelaunch'


def harness_refused(r: dict | None) -> bool:
    """A screen our harness refused or lost (not an experiment): a `runner_error`, or X3's D24 pre-launch refusal
    (`not_launched: 'pre_launch'`; a validator rejection is `not_launched: 'invalid_proposal'`, an invalid proposal)."""
    r = r or {}
    return isinstance(r.get('runner_error'), dict) or r.get('not_launched') == 'pre_launch'


def refusal_kind(r: dict) -> str:
    """The class of a harness-refused screen: `prelaunch` for D24's pre-launch refusal (nothing was launched: an
    `infra_refusal`, not an experiment; the 13:26 form is a pre_launch AttemptDeadline, the post-14:05 form carries
    `not_launched: 'pre_launch'` and an `infra_refusal:` reason), else the error's type."""
    e = r.get('runner_error') if isinstance(r.get('runner_error'), dict) else {}
    if (r.get('not_launched') == 'pre_launch' or str((r.get('invalid_reasons') or [''])[0]).startswith('infra_refusal')
            or (e.get('type') == 'AttemptDeadline' and 'pre_launch' in str(e.get('message') or ''))):
        return PRELAUNCH
    return str(e.get('type') or e.get('kind'))


def trials_used(trace: dict | None, source) -> dict:
    """'used / budget (stop reason)': the budget's own screen counter (harness-refused screens included) over the
    screen cap, and why the run stopped."""
    tr = trace if isinstance(trace, dict) else {}
    used, cap = (tr.get('used') or {}).get('screens'), (tr.get('budget') or {}).get('screens')
    if used is None or not cap:
        return cell(None, source, note='no used / budget screens in the trace')
    return cell(f'{used} / {cap} ({tr.get("stop_reason")})', source)


# ---------------------------------------------------------------- ledger
_PATTERN = re.compile(r'^(?P<pre>[^*\s]*)\*(?P<post>[^\s(]*)\s*\((?P<items>[^)]*)\)')


def expand_run_ids(text) -> list[str]:
    """'envc_*_c (crnd, cnatfree, cnatinfra, cis)' -> the four run ids; a plain run id -> [it]."""
    if not isinstance(text, str) or not text.strip():
        return []
    m = _PATTERN.match(text.strip())
    if m:
        return [f'{m["pre"]}{x.strip()}{m["post"]}' for x in m['items'].split(',') if x.strip()]
    return [text.strip().split()[0]]


def is_pattern(rid: str) -> bool:
    return any(c in str(rid) for c in '*?[')


def _env_dir(ledger: dict, env: str) -> str:
    return str((ledger.get(env) or {}).get('env') or ENV_DIRS.get(env, env))


def abandoned(ledger: dict, root=None) -> list[dict]:
    """Every abandoned run of the ledger: {'env', 'scope', 'run_id', 'reason'[, 'pattern']}. 'pre_*_post (a, b)' expands
    to its list; a bare glob ('envc_*_d') expands to the matching run directories of the environment under `root`
    (kept as the pattern when there is no root or no match: its membership is then tested with fnmatch)."""
    out = []

    def add(env, scope, rows):
        for row in rows or []:
            for rid in expand_run_ids(row.get('run_id')):
                ids = [rid]
                if is_pattern(rid) and root is not None and (Path(root) / _env_dir(ledger, env)).is_dir():
                    ids = sorted(p.name for p in (Path(root) / _env_dir(ledger, env)).iterdir()
                                 if p.is_dir() and fnmatch.fnmatchcase(p.name, rid)) or [rid]
                for x in ids:
                    out.append({'env': env, 'scope': scope, 'run_id': x, 'reason': row.get('reason'),
                                **({'pattern': rid} if is_pattern(rid) else {})})
    a = ledger.get('ENV-A') or {}
    for arm, spec in (a.get('arms') or {}).items():
        add('ENV-A', arm, spec.get('abandoned') if isinstance(spec, dict) else None)
    add('ENV-A', 'ENV-A', a.get('abandoned'))
    b = ledger.get('ENV-B') or {}
    for w, spec in b.items():
        if isinstance(spec, dict) and re.fullmatch(r'W\d+', w):
            add('ENV-B', w, spec.get('abandoned'))
    add('ENV-B', 'ENV-B', b.get('abandoned'))
    add('ENV-C', 'ENV-C', (ledger.get('ENV-C') or {}).get('abandoned'))
    add('F1', 'F1', (ledger.get('F1') or {}).get('abandoned'))
    return out


def not_run_reason(spec) -> str | None:
    """A sanctioned final `not run` in the ledger: a string value 'not run…' or {'run_id': null, 'status': 'not run…'};
    a null run id alone means 'not started yet' (pending)."""
    if isinstance(spec, str) and spec.strip().lower().replace('_', ' ').startswith('not run'):
        return spec.strip()
    if isinstance(spec, dict) and not spec.get('run_id') and str(spec.get('status') or '').lower().replace('_', ' ').startswith('not run'):
        return str(spec['status'])
    return None


def run_id_of(spec) -> str | None:
    if isinstance(spec, dict):
        return spec.get('run_id') or None
    if isinstance(spec, str) and not not_run_reason(spec):
        return spec.strip() or None
    return None


def active_ids(ledger: dict) -> dict[tuple, str]:
    """{(env, run_id): 'slot'} of every result slot of the ledger (arms, locked runs, references, block, node refs);
    run directories are environment-scoped, so identity is (env, run_id)."""
    out = {}
    a = ledger.get('ENV-A') or {}
    for arm, spec in (a.get('arms') or {}).items():
        for k in ('run_id', 'locked'):
            rid = run_id_of((spec or {}).get(k)) if isinstance(spec, dict) else (run_id_of(spec) if k == 'run_id' else None)
            if rid:
                out[('ENV-A', rid)] = f'ENV-A {arm}' + (' locked' if k == 'locked' else '')
    for node, rid in (a.get('node_refs') or {}).items():
        if rid and rid != 'block S0':
            out[('ENV-A', rid)] = f'ENV-A node_ref {node}'
    if (a.get('block') or {}).get('run_id'):
        out[('ENV-A', a['block']['run_id'])] = 'ENV-A block'
    for w, spec in (ledger.get('ENV-B') or {}).items():
        if isinstance(spec, dict) and re.fullmatch(r'W\d+', w):
            for arm, v in spec.items():
                if arm == 'locked' and isinstance(v, dict):
                    out.update({('ENV-B', run_id_of(x)): f'ENV-B {w} locked {k}' for k, x in v.items() if run_id_of(x)})
                elif arm.startswith('B-') and run_id_of(v):
                    out[('ENV-B', run_id_of(v))] = f'ENV-B {w} {arm}'
    c = ledger.get('ENV-C') or {}
    for key in ('arms', 'locked'):
        out.update({('ENV-C', run_id_of(x)): f'ENV-C {key} {k}' for k, x in (c.get(key) or {}).items() if run_id_of(x)})
    if c.get('calibration'):
        out[('ENV-C', c['calibration'])] = 'ENV-C calibration'
    if c.get('calibration_extra'):
        out[('ENV-C', str(c['calibration_extra']).split()[0])] = 'ENV-C calibration_extra'
    if anchor_ref(c.get('anchor'))[1]:
        out[('ENV-C', anchor_ref(c.get('anchor'))[1])] = 'ENV-C anchor'
    f = ledger.get('F1') or {}
    for key in ('references', 'arms'):
        out.update({('F1', run_id_of(x)): f'F1 {k}' for k, x in (f.get(key) or {}).items() if run_id_of(x)})
    return out


def abandoned_ids(ledger: dict, env: str, root=None) -> set:
    return {x['run_id'] for x in abandoned(ledger, root) if x['env'] == env}


def in_ids(rid, ids) -> bool:
    """rid is one of `ids` (exact, or matched by an unexpanded glob pattern among them)."""
    return bool(rid) and (rid in ids or any(is_pattern(x) and fnmatch.fnmatchcase(str(rid), x) for x in ids))


def ledger_conflicts(ledger: dict, root=None) -> list[str]:
    """Abandoned runs that also sit in a result slot of the same environment (formal-blocking; the slot is not used)."""
    act = active_ids(ledger)
    out = []
    for x in abandoned(ledger, root):
        for (env, rid), slot in act.items():
            if env == x['env'] and in_ids(rid, {x['run_id']}):
                out.append(f'ledger: abandoned run {rid} ({x["reason"]}) is also the result slot {slot}')
    return sorted(set(out))


def anchor_ref(text) -> tuple[str | None, str | None]:
    """'a21767e9dbd4f (x1check_20261021, D-AR-7 baseline on WINH0)' -> (attempt id, run id)."""
    m = re.match(r'\s*([0-9a-f]{8,})\s*\(\s*([^,)\s]+)', str(text or ''))
    return (m.group(1), m.group(2)) if m else (None, None)


# ---------------------------------------------------------------- cost of a run (abandoned / pipeline-fix line)
def _used_of(obj) -> dict | None:
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get('trace'), dict) and isinstance(obj['trace'].get('used'), dict):
        return obj['trace']['used']
    for k in ('used', 'trace_used'):
        if isinstance(obj.get(k), dict):
            return obj[k]
    return None


def run_cost(store: Store, run_dir, *, world_size: int | None = None, read_events: bool = True) -> dict:
    """GPU-hours, LLM tokens / USD and wall hours of a run from its own records (first that exists): the `used` block
    of result.json / trace.json / agent/trace.json; ABANDONED.json (ENV-A); else the signed receipts' GPU seconds plus
    the LLM call log (events / shim), wall = first to last event. `basis` says which. read_events=False (ENV-A: the
    native arm's events carry its transcript) skips the event logs."""
    run_dir = Path(run_dir)
    rid = run_dir.name
    for name in ('result.json', 'trace.json', 'agent/trace.json'):
        used = _used_of(store.json(run_dir / name))
        if used:
            return {'gpu_hours': used.get('gpu_hours'), 'llm_tokens': used.get('llm_tokens'), 'llm_usd': used.get('llm_usd'),
                    'wall_hours': (used['wall_seconds'] / 3600.0) if finite(used.get('wall_seconds')) else None,
                    'screens': used.get('screens'), 'basis': f'{name} used', 'src': [f'{rid}/{name}']}
    ab = store.json(run_dir / 'ABANDONED.json')
    if isinstance(ab, dict):
        t0, t1 = utc(ab.get('first_event_utc')), utc(ab.get('last_event_utc'))
        wall = (t1 - t0).total_seconds() / 3600.0 if t0 and t1 else None
        llm_wall = ab.get('llm_wall_seconds') / 3600.0 if finite(ab.get('llm_wall_seconds')) else None
        return {'gpu_hours': (world_size * wall) if (world_size and wall is not None) else None,
                'llm_tokens': ab.get('llm_tokens'), 'llm_usd': 0.0 if ab.get('llm_tokens') is not None else None,
                'wall_hours': wall, 'screens': ab.get('scorings'),
                'llm_gpu_hours_h20': 4.0 * llm_wall if llm_wall is not None else None,
                'basis': 'ABANDONED.json; GPU-hours = world size x first-to-last event (estimate: lease times not recorded); '
                         'H20-GPUh = 4 x LLM wall (AMENDMENT_EXT_001 Q1)', 'src': [f'{rid}/ABANDONED.json']}
    recs = store.receipts(run_dir)
    gpu_s = sum(float((r.get('cost') or {}).get('gpu_seconds') or 0.0) for r in recs.values())
    tokens = usd = 0.0
    times = []
    for name in ('shim_calls.jsonl', 'events.jsonl', 'agent/events.jsonl') if read_events else ():
        p = run_dir / name
        b = store.read_bytes(p, role='record') if p.is_file() else None
        if b is None:
            continue
        for line in b.decode('utf-8', errors='replace').splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            t = utc(e.get('utc'))
            if t:
                times.append(t)
            if name == 'shim_calls.jsonl':
                tokens += float(e.get('llm_tokens') or 0.0)
                usd += float(e.get('llm_usd') or 0.0)
            elif e.get('kind') == 'llm_call' and isinstance(e.get('payload'), dict) and not (run_dir / 'shim_calls.jsonl').is_file():
                tokens += float(e['payload'].get('llm_tokens') or 0.0)
                usd += float(e['payload'].get('llm_usd') or 0.0)
    wall = (max(times) - min(times)).total_seconds() / 3600.0 if len(times) >= 2 else None
    if recs:
        gpu, gbasis = gpu_s / 3600.0, "sum of the signed receipts' GPU seconds"
    elif world_size and wall is not None:
        gpu, gbasis = world_size * wall, f'{world_size} GPUs x first-to-last logged event (occupancy estimate, no receipt)'
    else:
        gpu, gbasis = None, 'no receipt and no interval'
    return {'gpu_hours': gpu, 'llm_tokens': int(tokens) if tokens else None, 'llm_usd': usd or None,
            'wall_hours': wall, 'screens': len(recs) or None,
            'basis': f'{gbasis}; LLM from the call log; wall = first to last logged event (estimate: the run left no '
                     'used block)', 'src': [f'{rid}/receipts'] if recs else [f'{rid}/events']}
