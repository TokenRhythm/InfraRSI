"""SearchSession: candidates, validation, screening under budget, verdicts and the SearchTrace.

Shared by the LLM agent (optimize) and the non-LLM / third-party baselines so every arm produces the same trace
format for scoring (AUC_budget, CTG, InvalidAccept, ...). Selection: the best valid screened candidate by
StepTime (ties: earliest screen); the start candidate if nothing valid beats it.

hooks (ext.common.session_hooks.SessionHooks, EXT_INTERFACES §3): for external environments the config validator,
the patch guard and the overlay digest come from the hooks and the runner also receives candidate_id; with
hooks=None (every core task) nothing changes.

Multi-fidelity (revamp design C, runners that accept `steps` and `stop`): `trial(cid, steps=..., fidelity=...)` runs a
proposed candidate at a lower fidelity (probe: feasibility and memory; short: a step-time estimate). It is charged to
GPU-hours and wall clock but not to the screen count, and never sets the candidate's full-fidelity verdict.
`screen(cid, stop=...)` passes an early-abort predicate to the runner; an aborted screen is status 'aborted' and does
not count as a failure of the candidate.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import math
import re
import time
from pathlib import Path

from ..actions import config_diff, patch_guard, templates, validate_l0
from ..contracts import InfrastructureError, digest
from .budget import Budget, SearchLedger
from .judge import Verdict, default_judge, gpu_seconds, step_time, wall_seconds
from .state import Stateful

TRACE_SCHEMA = 'infrarsi-search-trace-v1'


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=str))


@dataclasses.dataclass
class Candidate:
    id: str
    parent: str | None
    kind: str                     # start | L0 | L1
    origin: str
    config: dict                  # resolved L0 config incl. 'env'
    overlay: dict                 # changed overlay files only: {name: text}
    overlay_digest: str | None
    key: str
    status: str = 'proposed'      # proposed | vetoed | screened | screen_failed | aborted | infeasible
    rationale: str = ''
    template: str | None = None
    warnings: list = dataclasses.field(default_factory=list)
    verdict: dict | None = None
    receipt: dict | None = None
    screen_index: int | None = None
    created_utc: str = dataclasses.field(default_factory=_utc)
    results: dict = dataclasses.field(default_factory=dict)      # lower-fidelity trials: {fidelity: {...}}

    def summary(self, parent_cfg: dict | None = None) -> dict:
        v = self.verdict or {}
        out = {'id': self.id, 'parent': self.parent, 'kind': self.kind, 'origin': self.origin, 'status': self.status,
               'change': config_diff(parent_cfg, self.config) if parent_cfg is not None else None,
               'overlay_files': sorted(self.overlay), 'template': self.template, 'valid': v.get('valid'),
               'step_time': v.get('step_time'), 'peak_reserved_gib': v.get('peak_reserved_gib'),
               'reasons': (v.get('reasons') or [])[:3], 'screen_index': self.screen_index}
        failed = [c['id'] for c in v.get('checks') or [] if isinstance(c, dict) and c.get('status') == 'fail']
        if failed:
            out['failed_checks'] = failed[:6]          # correctness checklist items (verify.checklist)
        if self.status == 'proposed':                  # never screened: what its lower-fidelity runs showed
            sh, pr = self.results.get('short') or {}, self.results.get('probe') or {}
            if sh:
                out['short'] = {'valid': sh.get('valid'), 'step_time': sh.get('step_time')}
            elif pr:
                out['probed'] = bool(pr.get('valid'))
        return out

    def deviations(self) -> dict:
        """U-1-21: the signed deviation from the reference (candidate / reference - 1) that the loss and gradient-norm
        checks compared with the tolerance on this candidate's full-length screen. None when the check measured none
        (unchecked, off, a value missing; the reference itself) or no screen ran every update (never screened, a failed
        probe's verdict, an aborted or crashed screen): a number over fewer updates is another comparison. Trace rows
        only: like the ledger's numerics, it stays out of what the model is shown (summary)."""
        ev = (self.receipt or {}).get('evidence') or {}
        full = type(ev.get('steps')) is int and ev.get('steps_completed') == ev['steps']
        checks = {x.get('id'): x for x in (self.verdict or {}).get('checks') or [] if full and isinstance(x, dict)}
        return {key: ((checks.get(cid) or {}).get('evidence') or {}).get('signed_deviation')
                for key, cid in (('loss_deviation', 'loss_within_tolerance'),
                                 ('grad_norm_deviation', 'grad_norm_within_tolerance'))}


@dataclasses.dataclass
class SearchTrace:
    schema: str
    run_id: str
    arm: str
    task_id: str
    mode: str
    package_digest: str | None
    driver: dict | None
    judge: str
    budget: dict
    used: dict
    accounting_complete: bool
    stop_reason: str
    selected: str
    selected_config: dict
    selected_overlay: dict
    selected_step_time: float | None
    start: str
    start_step_time: float | None
    candidates: list
    screens: list
    llm_calls: list
    skill_hits: list
    memory_hits: list
    memory_writes: list
    tool_calls: list
    invalid_proposals: dict
    verifier: dict
    started_utc: str
    ended_utc: str
    out_dir: str | None
    extra: dict

    def to_dict(self) -> dict:
        return _jsonable(dataclasses.asdict(self))


# the change needs a package that is not installed (D1-10: "FlashAttention2 has been toggled on, but ... the package
# flash_attn seems to be not installed"; transformers' "requires the X library but it was not found in your
# environment"): no base runs it while the package is missing. A plain ModuleNotFoundError may depend on other knobs.
UNRUNNABLE = re.compile(r'^(?:\w+\.)*ImportError: .*(not installed|not found in your environment)')
# hls2 (FABLE_DESIGN_SEARCH_FIX C5): the error of a proposal refused because it sets a knob value whose runs failed to
# start `start_failure_limit` times in this search (killed at the launch timeout, or a crash that is not out of memory,
# before the first update: torch.compile hit the 900 s cap three times in Q9U-opus2); out of memory depends on the base
START_FAILED = 'unrunnable value:'
# the stop reason of a search the user stopped (Ctrl-C / SIGTERM) under `search.on_interrupt: verify`: what it measured goes
# through the promotion as after any other stop
USER_STOP = 'user_stop'
# v4 F4 (`SearchSession.patch_lines_own`): the self-proof's print lines a patch's own line count leaves out, and the hooks'
# cumulative line error the own count replaces
OWN_PROOF_MARK = '[infrarsi-patch]'
OWN_LINES_ERROR = re.compile(r'diff \d+ lines > ')


def _count(x) -> int:
    try:
        return int(str(x).strip() or 0)
    except ValueError:
        return 0


class SearchSession(Stateful):
    # v4 F4 (FABLE_CODE_DIAGNOSIS_20261102; set on its session by a v4 finalist, off for every other arm): a hook patch's
    # line limit counts its own lines against the candidate it is proposed on, the whole overlay under this x the limit
    patch_lines_own = False
    patch_overlay_ceiling = 3
    # search-state checkpoints (agent/state.py): what the resuming run supplies - the task, the hooks, the runner, the
    # judge, the clock, the console line, the budget, the output directory, the reference and the constructor's settings;
    # the ledger is a part (its used amounts and its wall clock are state); everything else is saved
    CHECKPOINT_LIVE = ('task', 'task_raw', 'hooks', 'runner', 'judge', 'clock', 'progress', 'out_dir', 'budget', 'reference',
                       'model_facts', 'mode', 'layers', 'patch_line_limit', 'count_rescores', 'run_id', 'arm',
                       'stop_after_same_failures', 'package_digest', 'driver_info', 'ledger', 'on_interrupt', 'control')

    def checkpoint_parts(self) -> dict:
        return {'ledger': self.ledger}

    def __init__(self, *, task, budget: Budget, runner, judge=None, run_id: str, arm: str, out_dir=None,
                 start_config: dict, start_receipt: dict | None = None, reference_receipt: dict | None = None,
                 model_facts: dict | None = None, mode: str = 'strong', layers=('L0', 'L1'), clock=None,
                 stop_after_same_failures: int = 6, package_digest: str | None = None, driver_info: dict | None = None,
                 hooks=None, patch_line_limit: int | None = None, count_rescores: bool = False):
        self.task = task
        self.task_raw = task.raw if hasattr(task, 'raw') else task
        self.hooks = hooks
        self.patch_line_limit = patch_line_limit           # capability knob: replaces the per-mode limit for hook patches
        self.count_rescores = count_rescores               # a re-run is a trial of the screens budget (an environment arm)
        # Formal runs use the reward engine on harness-signed receipts; tests pass ReceiptJudge for synthetic ones.
        self.budget, self.runner, self.judge = budget, runner, judge or default_judge()
        self.run_id, self.arm, self.mode, self.layers = run_id, arm, mode, tuple(layers)
        self.model_facts, self.reference = model_facts, reference_receipt
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir:
            (self.out_dir / 'receipts').mkdir(parents=True, exist_ok=True)
        self.clock = clock or time.monotonic
        self.ledger = SearchLedger(budget, self.clock)
        self.candidates: dict[str, Candidate] = {}
        self.by_key: dict[str, str] = {}
        self.screens, self.llm_calls, self.events = [], [], []
        self.skill_hits, self.memory_hits, self.memory_writes, self.tool_calls = [], [], [], []
        self.invalid = {}
        self.verifier_stats = {'calls': 0, 'vetoes': 0, 'errors': 0}
        self.same_failure = (None, 0)
        self.unrunnable: dict[str, str] = {}             # [knob, value, overlay digest] -> the log line
        self.start_failure_limit: int | None = None      # hls2 (C5): set by the strategy; None counts nothing
        self.start_failures: dict[str, int] = {}         # [knob, value] -> runs that set it and failed to start
        self.unrunnable_values: dict[str, str] = {}      # [knob, value] -> why, once it reached the limit
        self.start_failure_veto = None                   # hls3 (F5b): (knob, value, run, parent) -> True keeps a value
        #                                                  at the limit runnable (every failure had a suspect partner)
        self.stop_flag: str | None = None
        self.stop_after_same_failures = stop_after_same_failures
        self.package_digest, self.driver_info = package_digest, driver_info
        self.started_utc = _utc()
        self.pair_wall_cap: int | None = None          # set by bounded admission; the locked pair runs within it
        self.pair_reserve: dict | None = None
        self.progress = None                           # log(line): one console line per finished run (U-1-14), or None
        self.on_interrupt = None                       # 'verify': the Engine ends an interrupted search as a normal one
        if hooks is None:
            rep = validate_l0(start_config, self.task_raw, model_facts)
            if not rep.ok:
                raise ValueError(f'start config invalid: {rep.errors}')
            resolved = rep.resolved
        else:
            resolved, errors, _ = hooks.validate_config(start_config)
            if errors:
                raise ValueError(f'start config invalid: {errors}')
        c0 = self._new('start', None, 'start', resolved, {}, rationale='search start (ED or FR)')
        if start_receipt is not None:
            self._record_receipt(c0, start_receipt, counted=False)
        self.ledger.prior = self._prior_screen_cost(start_receipt)

    def _prior_screen_cost(self, start_receipt: dict | None) -> tuple[float, float] | None:
        """Admission estimate before the first measured screen (keeps the scoring reserve from the start): the start
        receipt's measured cost, else the task's planned search GPU-hours spread over screens + reserve."""
        g, w = gpu_seconds(start_receipt), wall_seconds(start_receipt)
        if g is not None and g > 0:
            return g / 3600.0, w if w is not None else 0.0
        plan = (self.task_raw.get('gpu_hours_plan') or {}).get('search_incl_scoring')
        n = self.budget.screens + self.budget.scoring_reserve_screens
        if type(plan) in (int, float) and plan > 0 and n > 0:
            return float(plan) / n, self.budget.wall_seconds / n
        return None

    # ---------------------------------------------------------------- bookkeeping
    def poll_control(self, checkpoint=None):
        control = getattr(self, 'control', None)
        if control is not None:
            if checkpoint is None:
                control(self)
            else:
                control(self, checkpoint=checkpoint)

    def poll_trial_control(self):
        owner = getattr(getattr(self, 'control', None), '__self__', None)
        if callable(getattr(owner, 'before_trial', None)):
            owner.before_trial(self)

    def event(self, kind: str, payload: dict):
        e = {'i': len(self.events), 'utc': _utc(), 'wall': round(self.ledger.wall(), 3), 'kind': kind, 'payload': _jsonable(payload)}
        self.events.append(e)
        if self.out_dir:
            with (self.out_dir / 'events.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(e, ensure_ascii=False) + '\n')
        if self.progress is not None and kind in ('screen', 'trial', 'rescore'):
            try:                                        # a console line never stops the search
                self.progress(self._progress_line(kind, payload))
            except Exception:                           # noqa: BLE001
                pass

    def _progress_line(self, kind: str, row: dict) -> str:
        """U-1-14: a long search was silent between its start and the promotion; one line per finished run says which
        run it was, its verdict (with the reason when it failed) and the best so far."""
        start = self.candidates.get('c0')
        ref = (start.verdict or {}).get('step_time') if start else None

        def vs(t):
            return f' (x{ref / t:.3f} vs the reference)' if ref and t else ''
        cid = row.get('candidate_id')
        if kind == 'screen':
            head = f'run {(row.get("cum") or {}).get("screens", "?")} of {self.budget.screens}: {cid}'
        elif kind == 'trial':
            head = f'{row.get("fidelity")} run of {cid} ({row.get("steps")} updates)'
        else:
            head = f're-run of {cid}'
        if row.get('aborted'):
            verdict = 'stopped early, slower than the best so far'
        elif row.get('valid'):
            verdict = f'valid, {row.get("step_time"):.4g} s per update{vs(row.get("step_time"))}'
        elif row.get('reasons'):
            check, _, why = str(row['reasons'][0]).partition(': ')
            verdict = f'invalid ({check}): {why[:160]}' if why else f'invalid: {check[:160]}'
        else:
            verdict = 'invalid'
        best = row.get('best_valid')
        bv = self.candidates[best].verdict if best in self.candidates else None
        tail = ('; the best so far' if best == cid else f'; best so far {best}{vs((bv or {}).get("step_time"))}'
                if best and best != 'c0' else '; nothing faster than the reference so far' if best == 'c0' else '')
        return f'[infrarsi] {head}: {verdict}{tail}'

    def _new(self, kind, parent, origin, config, overlay, *, rationale='', template=None, warnings=()):
        cid = f'c{len(self.candidates)}'
        ov_dig = None
        if overlay and self.hooks is not None:
            ov_dig = self.hooks.overlay_digest(overlay)
        elif overlay:
            full = {**patch_guard.base_overlays(), **overlay}
            ov_dig = patch_guard.overlay_digest({n: patch_guard._sha(t) for n, t in full.items()})
        key = digest({'config': config, 'overlay': ov_dig})
        c = Candidate(id=cid, parent=parent, kind=kind, origin=origin, config=config, overlay=dict(overlay),
                      overlay_digest=ov_dig, key=key, rationale=rationale[:2000], template=template, warnings=list(warnings))
        self.candidates[cid] = c
        self.by_key.setdefault(key, cid)
        self.event('candidate', {'id': cid, 'kind': kind, 'parent': parent, 'origin': origin, 'template': template,
                                 'change': config_diff(self.candidates[parent].config, config) if parent else None,
                                 'overlay_files': sorted(overlay)})
        return c

    def count_invalid(self, reason: str):
        self.invalid[reason] = self.invalid.get(reason, 0) + 1

    def charge_llm(self, result, role: str, turn: int | None = None, **record):
        """Charge one call and keep its row (trace llm_calls, events.jsonl). record: arm-specific fields of the call
        (e.g. a declared prompt transform); the common fields always win."""
        self.ledger.charge_llm(result.llm_tokens, result.llm_usd, result.accounting_complete)
        row = {**record, 'turn': turn, 'role': role, 'model': result.model, 'observed_models': result.observed_models,
               'llm_tokens': result.llm_tokens, 'llm_usd': result.llm_usd, 'wall_seconds': round(result.wall_seconds, 3),
               'accounting_complete': result.accounting_complete, 'error': result.error, 'attempts': result.attempts,
               'call_id': result.meta.get('call_id'), 'selfhost_gpu_seconds': result.meta.get('selfhost_gpu_seconds'),
               'gpu_type': result.meta.get('gpu_type')}
        self.llm_calls.append(row)
        self.event('llm_call', row)

    # ---------------------------------------------------------------- proposals
    def resolve_base(self, base: str | None) -> Candidate:
        if base is None:
            best = self.best_valid()
            return self.candidates[best] if best else self.candidates['c0']
        if base not in self.candidates:
            raise KeyError(f'unknown base candidate {base!r}')
        return self.candidates[base]

    def check_config(self, proposed: dict, base: Candidate) -> tuple[str | None, dict, list, list, dict | None]:
        """propose_config's acceptance of a complete configuration on `base` (the L0 layer open, then validate_l0 or
        the hooks' rule: for a project the knob schema, the constraints and the workload invariants): (key, resolved,
        errors, warnings, derived), the key None when it is refused and in by_key when it repeats a candidate. It
        counts and records nothing, so a strategy can sample inside the constraints without its refused draws counting
        as proposals."""
        if 'L0' not in self.layers:                    # propose_config refuses these before validating (layer_closed)
            return None, {}, ['L0 actions are closed in this arm'], [], None
        if self.hooks is None:
            rep = validate_l0(proposed, self.task_raw, self.model_facts)
            ok, resolved, errors, warnings, derived = rep.ok, rep.resolved, rep.errors, rep.warnings, rep.derived
        else:
            resolved, errors, warnings = self.hooks.validate_config(proposed)
            ok, derived = not errors, None
        key = digest({'config': resolved, 'overlay': base.overlay_digest}) if ok else None
        return key, resolved, errors, warnings, derived

    def propose_config(self, *, config=None, changes=None, base=None, origin='agent', rationale='') -> dict:
        try:
            b = self.resolve_base(base)
        except KeyError as exc:
            self.count_invalid('unknown_base')
            return {'error': str(exc)}
        if 'L0' not in self.layers:
            self.count_invalid('layer_closed')
            return {'error': 'L0 actions are closed in this arm'}
        if (config is None) == (changes is None):
            self.count_invalid('config_shape')
            return {'error': 'give exactly one of "config" (complete) or "changes" (delta on base)'}
        if config is not None:
            proposed = dict(config)
        elif self.hooks is not None:
            proposed = {**b.config, **changes}            # knob deltas (None is a value for ext knobs)
        else:
            proposed = {k: v for k, v in b.config.items() if k != 'env'}
            env = dict(b.config.get('env') or {})
            for k, v in (changes or {}).items():
                if k == 'env':
                    if type(v) is not dict:
                        self.count_invalid('l0_invalid')
                        return {'error': 'env must be an object'}
                    for name, val in v.items():
                        if val is None:
                            env.pop(name, None)
                        else:
                            env[name] = val
                else:
                    proposed[k] = v
            proposed['env'] = env
        key, resolved, errors, warnings, derived = self.check_config(proposed, b)
        if key is None:
            self.count_invalid('l0_invalid')
            self.event('invalid_proposal', {'kind': 'L0', 'errors': errors})
            return {'error': 'L0 validation failed' if self.hooks is None else 'config validation failed', 'errors': errors[:8],
                    'warnings': warnings[:4]}
        if key in self.by_key:
            dup = self.candidates[self.by_key[key]]
            self.count_invalid('duplicate')
            return {'duplicate_of': dup.id, 'status': dup.status, 'verdict': dup.verdict and {k: dup.verdict.get(k) for k in ('valid', 'step_time')}}
        c = self._new('L0', b.id, origin, resolved, b.overlay, rationale=rationale, warnings=warnings)
        return {'candidate_id': c.id, 'parent': b.id, 'change': config_diff(b.config, c.config), 'warnings': warnings[:4],
                'derived': derived}

    def propose_patch(self, *, files=None, template=None, params=None, base=None, origin='agent', rationale='') -> dict:
        try:
            b = self.resolve_base(base)
        except KeyError as exc:
            self.count_invalid('unknown_base')
            return {'error': str(exc)}
        if 'L1' not in self.layers:
            self.count_invalid('layer_closed')
            return {'error': 'L1 patches are closed in this arm (L0 only)'}
        if (files is None) == (template is None):
            self.count_invalid('patch_shape')
            return {'error': 'give exactly one of "files" or "template"'}
        if self.hooks is not None:
            return self._propose_ext_patch(b, files=files, template=template, params=params, origin=origin, rationale=rationale)
        if template is not None:
            rendered, errs = templates.render(template, params or {}, {**patch_guard.base_overlays(), **b.overlay})
            if errs:
                self.count_invalid('template_invalid')
                return {'error': 'template rejected', 'errors': errs}
            l0 = templates.TEMPLATES[template].l0_overrides(params or {})
            if l0:          # the runtime applies these over the L0 config: validate the effective config now
                rep = validate_l0({**{k: v for k, v in b.config.items() if k != 'env'}, **l0, 'env': b.config.get('env') or {}},
                                  self.task_raw, self.model_facts)
                if not rep.ok:
                    self.count_invalid('template_invalid')
                    return {'error': 'template overrides make the effective config invalid', 'errors': rep.errors[:6]}
            files = rendered
        merged = {**b.overlay, **{str(k): v for k, v in (files or {}).items()}}
        report = patch_guard.apply(merged, mode='weak' if self.mode == 'weak' else 'strong')
        if not report.accepted:
            self.count_invalid('patch_rejected')
            self.event('invalid_proposal', {'kind': 'L1', 'findings': report.findings[:12]})
            return {'error': 'patch_guard rejected the overlay', 'findings': report.findings[:8], 'diff_lines': report.diff_lines,
                    'limit': report.limit}
        names = {patch_guard.zones.overlay_name(n): t for n, t in merged.items()}
        key = digest({'config': b.config, 'overlay': report.overlay_digest})
        if key in self.by_key:
            self.count_invalid('duplicate')
            return {'duplicate_of': self.by_key[key]}
        c = self._new('L1', b.id, origin, b.config, names, rationale=rationale, template=template)
        return {'candidate_id': c.id, 'parent': b.id, 'diff_lines': report.diff_lines, 'overlay_digest': c.overlay_digest}

    def _propose_ext_patch(self, b: Candidate, *, files, template, params, origin, rationale) -> dict:
        """File patch of an external environment: the hooks' templates and patch guard; the core per-mode line limit
        (strong 300 / weak 80) still applies on top of the task's own limit."""
        if template is not None:
            rendered, errs = self.hooks.render_template(template, params or {}, self.hooks.candidate_files(b.overlay))
            if errs:
                self.count_invalid('template_invalid')
                return {'error': 'template rejected', 'errors': errs}
            files = rendered
        merged = {**b.overlay, **{str(k): v for k, v in (files or {}).items()}}
        errors, warnings = self.hooks.validate_patch(merged)
        lines = self.hooks.patch_lines(merged)
        mode_limit = self.patch_line_limit or patch_guard.LINE_LIMITS['weak' if self.mode == 'weak' else 'strong']
        limit = min(int(self.hooks.patch_line_limit), int(mode_limit))
        if self.patch_lines_own:                       # v4 F4: the limit on this patch's own lines, the overlay's ceiling
            errors, lines, limit = self._own_lines_check(b, merged, errors, lines, limit)
        if lines > limit and not any('lines > limit' in e for e in errors):
            errors = errors + [f'diff {lines} lines > {self.mode} limit {limit}']
        if errors:
            self.count_invalid('patch_rejected')
            self.event('invalid_proposal', {'kind': 'L1', 'findings': errors[:12]})
            return {'error': 'patch rejected', 'findings': errors[:8], 'diff_lines': lines, 'limit': limit}
        key = digest({'config': b.config, 'overlay': self.hooks.overlay_digest(merged)})
        if key in self.by_key:
            self.count_invalid('duplicate')
            return {'duplicate_of': self.by_key[key]}
        c = self._new('L1', b.id, origin, b.config, merged, rationale=rationale, template=template, warnings=warnings)
        return {'candidate_id': c.id, 'parent': b.id, 'diff_lines': lines, 'overlay_digest': c.overlay_digest,
                'warnings': warnings[:4]}

    def _own_lines_check(self, b: Candidate, merged: dict, errors: list, lines: int, limit: int) -> tuple:
        """v4 F4 (`patch_lines_own`): the per-patch limit on the new patch's own lines - its added and removed lines
        against the files of the candidate it is proposed on, the self-proof's lines (OWN_PROOF_MARK prints) left out -
        and the whole overlay (the hooks' count against the project's files, `lines`) under `patch_overlay_ceiling` x the
        limit; the hooks' own cumulative line error is replaced by these two. (errors, the own lines, the limit)."""
        from ..ext.common.session_hooks import diff_lines
        old = self.hooks.candidate_files(b.overlay)

        def bare(text) -> str:
            return '\n'.join(ln for ln in str(text or '').splitlines() if OWN_PROOF_MARK not in ln)
        own = sum(diff_lines(bare(old.get(p, '')), bare(t)) for p, t in merged.items()
                  if type(t) is str and t != old.get(p))
        ceiling = int(self.patch_overlay_ceiling * limit)
        out = [e for e in errors if not OWN_LINES_ERROR.match(str(e))]
        if self._inherited_overlay(merged):             # v4.1: a knob variation of a candidate carries that candidate's overlay
            return out, 0, limit                        # - no line of its own, and the ceiling is for a new patch on top
        if own > limit:
            out.append(f"diff {own} lines > limit {limit} (this patch's own lines against its base)")
        if lines > ceiling:
            out.append(f'diff {lines} lines > limit {ceiling} (the whole overlay: {self.patch_overlay_ceiling} x {limit})')
        return out, own, limit

    def _inherited_overlay(self, merged: dict) -> bool:
        """v4.1 (the stacking loop): `merged` is the overlay of a candidate that exists - the incumbent's, which a knob
        variation of the incumbent carries (a strategy writes it as that configuration on the reference with the incumbent's
        files). Its lines were counted against the limit and the ceiling when that candidate was proposed; a variation of its
        knobs adds none, so it is inherited, not counted again. A new patch on top is another overlay: the limit on its own
        lines and the ceiling on the whole stay."""
        if not merged:
            return False
        dig = self.hooks.overlay_digest(merged)
        return dig is not None and any(c.overlay_digest == dig for c in self.candidates.values() if c.overlay)

    # ---------------------------------------------------------------- screening
    def _record_receipt(self, c: Candidate, receipt: dict, *, counted: bool):
        parent = self.candidates[c.parent].receipt if c.parent and self.candidates[c.parent].receipt else None
        try:
            v = self.judge(receipt, parent, self.reference)
        except Exception as exc:                       # a judge crash invalidates, never accepts
            v = Verdict(valid=False, reasons=[f'judge error: {type(exc).__name__}: {exc}'], step_time=step_time(receipt))
        c.receipt, c.verdict = receipt, v.to_dict()
        c.status = 'screened' if receipt.get('valid') is not None else 'screen_failed'
        if self.out_dir:
            (self.out_dir / 'receipts' / f'{c.id}.json').write_text(json.dumps(receipt, ensure_ascii=False, default=str), encoding='utf-8')
        return v

    def _note_unrunnable(self, c: Candidate, receipt: dict, verdict=None) -> None:
        """A one-knob change whose run says it needs a package that is not installed fails on every base with the same
        code (D1-10: flash_attention_2 without flash_attn, tried on three bases): see unrunnable_hit. With a
        start_failure_limit (hls2) a run that did not start also counts against the knob values it set (C5)."""
        ev = receipt.get('evidence') if isinstance(receipt, dict) and isinstance(receipt.get('evidence'), dict) else {}
        line = str((ev.get('events') or {}).get('error') or '')
        p = self.candidates.get(c.parent) if c.parent else None
        if self.start_failure_limit and p is not None and c.overlay == p.overlay:
            self._note_start_failure(c, p, ev, line, verdict)
        if not UNRUNNABLE.match(line) or p is None or c.overlay != p.overlay:
            return
        diff = [k for k in set(c.config) | set(p.config) if k != 'env' and c.config.get(k) != p.config.get(k)]
        if len(diff) == 1:
            self.unrunnable[json.dumps([diff[0], c.config.get(diff[0]), c.overlay_digest], default=str)] = line[:200]

    def _note_start_failure(self, c: Candidate, p: Candidate, ev: dict, line: str, verdict) -> None:
        """C5: a knob change (the parent's code) that did not reach its first update - killed at the launch timeout, or
        a crash that is not out of memory - counts once against each knob value it set; at start_failure_limit the
        value is unrunnable for the rest of the search (start_failed_hit). Out of memory, a sibling arm or an
        infrastructure refusal do not count (they depend on the base or the hosts). Out of memory as the log scanner
        reads it (signals.OOM_RE): Inductor's autotune "OutOfMemoryError: out of resource" is a config it skips, not an
        OOM, and a compile timeout whose log carries one still counts (review NM25)."""
        from ..project.signals import OOM_RE
        reasons = [str(r) for r in (getattr(verdict, 'reasons', None) or [])]
        if ev.get('steps_completed') != 0 or (reasons and reasons[0].split(':')[0] in ('occupied_sibling', 'infra_refusal')):
            return
        events = ev.get('events') or {}
        if _count(events.get('oom')) > 0 or OOM_RE.search(line) or any(OOM_RE.search(r) for r in reasons):
            return
        if ev.get('timed_out'):
            why = 'killed at the launch timeout before its first update'
        elif _count(events.get('traceback')) > 0 or line:
            why = 'crashed before its first update' + (f': {line[:160]}' if line else '')
        else:
            return
        for k in sorted(set(c.config) | set(p.config)):
            if k == 'env' or c.config.get(k) == p.config.get(k):
                continue
            key = json.dumps([k, c.config.get(k)], default=str)
            n = self.start_failures[key] = self.start_failures.get(key, 0) + 1
            if n >= self.start_failure_limit:
                veto = getattr(self, 'start_failure_veto', None)
                if veto is not None and veto(k, c.config.get(k), c, p):
                    continue                           # hls3 (F5b): it never failed without a suspect partner
                self.unrunnable_values[key] = f'{why} ({n} runs that set it)'

    def value_unrunnable(self, knob: str, value) -> str | None:
        """C5: why `knob` = `value` is unrunnable in this search (None when it is not)."""
        return self.unrunnable_values.get(json.dumps([knob, value], default=str))

    def start_failed_hit(self, cid: str) -> str | None:
        """C5: 'knob=value: why' when the proposed candidate sets a value that is unrunnable in this search and the
        measured candidate it was proposed on does not have (its descendants keep what ran); None otherwise."""
        c = self.candidates.get(cid)
        if c is None or c.verdict is not None or not self.unrunnable_values:
            return None
        b = self.candidates.get(c.parent) if c.parent else None
        while b is not None and b.verdict is None and b.parent:
            b = self.candidates.get(b.parent)
        ref = b.config if b is not None else {}
        for k in sorted(c.config):
            if k == 'env' or ref.get(k) == c.config[k]:
                continue
            why = self.value_unrunnable(k, c.config[k])
            if why:
                return f'{k}={c.config[k]}: {why}'
        return None

    def unrunnable_hit(self, cid: str) -> str | None:
        """The log line of an earlier failure that this candidate would repeat: it sets a knob to the value whose run
        said a package is not installed, with the same code (overlay); None otherwise."""
        c = self.candidates.get(cid)
        if c is None or c.verdict is not None:
            return None
        for key, line in self.unrunnable.items():
            k, v, od = json.loads(key)
            if od == c.overlay_digest and json.dumps(c.config.get(k), default=str) == json.dumps(v, default=str):
                return line
        return None

    def screen(self, cid: str, *, purpose: str = 'screen', stop=None, check_targets: bool = True) -> dict:
        self.poll_trial_control()
        if self.stop_flag:
            return {'error': f'search stopped: {self.stop_flag}'}
        c = self.candidates.get(cid)
        if c is None:
            return {'error': f'unknown candidate {cid!r}'}
        if c.status != 'proposed':
            return {'error': f'{cid} already {c.status}', 'verdict': c.verdict and {k: c.verdict.get(k) for k in ('valid', 'step_time', 'reasons')}}
        limit, cap = self._admission()
        if limit:
            self.stop_flag = f'budget:{limit}'
            self.event('stop', {'reason': self.stop_flag})
            return {'error': f'budget exhausted ({limit}); search stops'}
        t0 = self.clock()
        known, receipt, gpu_s, run_wall, wall, infra = True, None, None, None, 0.0, None
        try:
            try:
                kw = {'wall_cap': cap} if cap is not None else {}
                if self.hooks is not None:
                    kw['candidate_id'] = cid               # ext runners name the candidate directory after it
                if stop is not None:
                    kw['stop'] = stop                      # early abort at an update boundary
                receipt = self.runner.run(self.task, c.config, c.overlay or None, purpose, self.run_id, **kw)
            except InfrastructureError as exc:         # the hosts could not be reached: no verdict on the candidate
                infra, receipt, known = exc, {}, False
            except Exception as exc:                   # runner contract violation: cost unknown
                receipt, known = None, False
                err = f'runner error: {type(exc).__name__}: {exc}'
            else:
                err = None if isinstance(receipt, dict) else f'runner returned {type(receipt).__name__}, not a receipt'
            if infra is None:
                if err:
                    receipt, known = {'valid': False, 'invalid_reasons': [err], 'step_time': None}, False
                cost = receipt.get('cost')
                if isinstance(cost, dict) and cost.get('accounting_complete') is False:
                    known = False                      # e.g. the harness runner's OccupiedError record
                v = self._record_receipt(c, receipt, counted=True)
        finally:
            # Charged on every path (also when judging or persisting fails): driver-measured cost, else
            # world x blocking time with the accounting marked incomplete.
            wall = max(0.0, self.clock() - t0)
            gpu_s, run_wall = gpu_seconds(receipt), wall_seconds(receipt)
            if gpu_s is None:
                gpu_s, known = self._world(receipt) * wall, False
            self.ledger.charge_screen(gpu_s / 3600.0, run_wall if run_wall is not None else wall, known,
                                      count=infra is None)
        if infra is not None:
            return self._infrastructure_stop(cid, infra)
        aborted =bool(((receipt.get('evidence') or {}) if isinstance(receipt.get('evidence'), dict) else {}).get('stopped_early'))
        if aborted:
            c.status = 'aborted'
        c.screen_index = len(self.screens)
        best = self.best_valid()
        row = {'index': c.screen_index, 'candidate_id': cid, 'kind': c.kind, 'attempt_id': receipt.get('attempt_id'),
               'fidelity': 'full', 'aborted': aborted,
               'valid': v.valid, 'step_time': v.step_time, 'reasons': v.reasons[:4], 'gpu_hours': gpu_s / 3600.0,
               'wall_seconds': run_wall if run_wall is not None else wall, 'cum': self.ledger.snapshot(), 'fractions': self.ledger.fractions(),
               'best_valid': best, 'best_step_time': self.candidates[best].verdict['step_time'] if best else None}
        self.screens.append(row)
        self.event('screen', row)
        reason = None if v.valid else (v.reasons[0].split(':')[0] if v.reasons else 'invalid')
        # not a failure of the candidate: a sibling arm on this GPU (ext co-location), or an infrastructure refusal
        # before anything was launched (ENV-B pre-launch deadline / transport, O approval 2026-10-21, Follow-up 4)
        sibling = reason in ('occupied_sibling', 'infra_refusal') or aborted
        # the same failure: the same check and the same error in the log (an OOM, then a port clash, are two causes)
        ev = receipt.get('evidence') if isinstance(receipt.get('evidence'), dict) else {}
        err = str((ev.get('events') or {}).get('error') or '').split(':')[0]
        if not v.valid and not aborted:
            self._note_unrunnable(c, receipt, v)
        sig = None if reason is None else reason + (f' ({err})' if err else '')
        last, n = self.same_failure
        if not sibling:
            self.same_failure = (sig, n + 1 if sig and sig == last else (1 if sig else 0))
        if reason and not sibling and self.same_failure[1] >= self.stop_after_same_failures:
            self.stop_flag = f'same_failure_x{self.same_failure[1]}'
            self.event('stop', {'reason': self.stop_flag})
        elif self.ledger.exhausted():
            self.stop_flag = 'budget:' + self.ledger.exhausted()[0]
            self.event('stop', {'reason': self.stop_flag})
        if check_targets and c.status == 'screened' and v.valid and v.step_time:
            self.check_targets(cid, v.step_time)
        start_t = self.candidates['c0'].verdict and self.candidates['c0'].verdict.get('step_time')
        out = {'candidate_id': cid, 'valid': v.valid, 'step_time': v.step_time, 'reasons': v.reasons[:4],
               'vs_start': (round(math.log(start_t / v.step_time), 4) if v.valid and start_t and v.step_time else None),
               'best_valid': best, 'budget': self.budget_view()}
        if v.feedback:
            out['feedback'] = v.feedback               # failed checklist items with evidence and hints
        return out

    def check_targets(self, cid: str, step_time: float, *, reference_time=None, accounting=None, log=None) -> None:
        """Stop on a verified full screen, using measured MFU before an accounting estimate."""
        b = self.budget
        if b.target_speedup is None and b.target_mfu is None:
            return
        ref, cand = self.candidates['c0'], self.candidates[cid]
        t0 = reference_time or (ref.verdict or {}).get('step_time')
        reached = []
        if b.target_speedup is not None and t0 and step_time and t0 / step_time >= b.target_speedup:
            reached.append(f'target_speedup:{t0 / step_time:.3f}>={b.target_speedup:g}')
        if b.target_mfu is not None:
            def measured(receipt):
                value = ((receipt or {}).get('evidence') or {}).get('mfu')
                return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None
            mfu = measured(cand.receipt)
            if mfu is None and callable(accounting):
                mfu = (accounting(cand.config, step_time) or {}).get('mfu')
            if mfu is None and t0 and step_time:
                ref_mfu = measured(ref.receipt)
                if ref_mfu is not None:
                    mfu = ref_mfu * t0 / step_time
            if type(mfu) in (int, float) and math.isfinite(mfu) and mfu >= b.target_mfu:
                reached.append(f'target_mfu:{mfu:.3f}>={b.target_mfu:g}')
        for reason in reached:
            self.stop_flag = reason
            self.event('stop', {'reason': reason, 'candidate_id': cid, 'milestone': True})
            emit = log or self.progress
            if emit:
                emit(f'[infrarsi] milestone: {cid} reached {reason}')

    def trial(self, cid: str, *, steps: int, fidelity: str, stop=None, window: list | None = None,
              horizon: int | None = None, wall_cap: float | None = None, detail: bool = False) -> dict:
        """A lower-fidelity run of a proposed candidate (see the module docstring); returns the trial row. `horizon`:
        the {steps} the command gets (the reference's); the run is stopped after `steps` updates, so warm-up and decay
        stay the job's and the reference comparison holds at every update it ran. hls3 only (FABLE_HLS3_FIXES_R1 F1;
        unset, nothing changes): `wall_cap` stops the run at that wall instead of the job's launch timeout, and a run it
        stops is `start_slow` - not a start failure (C5 does not count it); `detail` adds the run's per-update intervals
        and wall to the row and the candidate's result."""
        self.poll_trial_control()
        if self.stop_flag:
            return {'error': f'search stopped: {self.stop_flag}'}
        c = self.candidates.get(cid)
        if c is None or c.status != 'proposed':
            return {'error': f'{cid} is not a proposed candidate'}
        limit, cap = self._admission()                # as screen(): what a full screen could not follow is not started
        if limit:
            self.stop_flag = 'budget:' + limit
            self.event('stop', {'reason': self.stop_flag})
            return {'error': f'budget exhausted ({limit})'}
        t0 = self.clock()
        kw = {'steps': int(horizon or steps), 'candidate_id': cid}
        if horizon and int(horizon) > int(steps):
            kw['stop_after'] = int(steps)
        if cap is not None:
            kw['wall_cap'] = cap
        if wall_cap is not None:                       # hls3 (F1a): the probe's own start cap
            kw['wall_cap'] = int(min(cap, wall_cap)) if cap is not None else int(wall_cap)
        if stop is not None:
            kw['stop'] = stop
        if window is not None:
            kw['window'] = list(window)                # the short run's scoring window (after the warm-up)
        try:
            receipt = self.runner.run(self.task, c.config, c.overlay or None, 'probe' if fidelity == 'probe' else 'short',
                                      self.run_id, **kw)
        except InfrastructureError as exc:             # no verdict on the candidate: the search stops
            wall = max(0.0, self.clock() - t0)
            self.ledger.charge_screen(self._world(None) * wall / 3600.0, wall, False, count=False)
            return self._infrastructure_stop(cid, exc)
        except Exception as exc:                       # runner contract violation: recorded, never raised
            receipt = {'valid': False, 'invalid_reasons': [f'runner error: {type(exc).__name__}: {exc}'], 'step_time': None}
        parent = self.candidates[c.parent].receipt if c.parent and self.candidates[c.parent].receipt else None
        try:
            v = self.judge(receipt, parent, self.reference)
        except Exception as exc:                       # noqa: BLE001 - a judge crash invalidates
            v = Verdict(valid=False, reasons=[f'judge error: {type(exc).__name__}: {exc}'])
        wall = max(0.0, self.clock() - t0)
        gpu_s = gpu_seconds(receipt)
        known = gpu_s is not None and not (isinstance(receipt.get('cost'), dict)
                                           and receipt['cost'].get('accounting_complete') is False)
        if gpu_s is None:
            gpu_s = self._world(receipt) * wall
        self.ledger.charge_screen(gpu_s / 3600.0, wall_seconds(receipt) or wall, known, count=False)
        ev = receipt.get('evidence') if isinstance(receipt.get('evidence'), dict) else {}
        aborted = bool(ev.get('stopped_early'))
        failed = [x['id'] for x in v.checks if x.get('status') == 'fail']
        # hls3 (F1a): stopped at the probe's own cap (the binding one), not a start failure
        slow = wall_cap is not None and (cap is None or wall_cap < cap) and bool(ev.get('timed_out'))
        extra = {}
        if slow:
            extra['start_slow'] = True
        if detail:
            ps = [r for r in ev.get('per_step') or [] if isinstance(r, dict) and isinstance(r.get('t'), (int, float))]
            extra.update(intervals=[b['t'] - a['t'] for a, b in zip(ps, ps[1:])][:8], wall_s=ev.get('wall_s'))
            events = ev.get('events') if isinstance(ev.get('events'), dict) else {}
            extra.update(perf_notes=list(events.get('perf_notes') or []),       # hls3 (F10): the probe's own notes and
                         patch_lines=list(events.get('patch_lines') or []),      # the patch's self-proof lines; their
                         patch_proof=dict(events.get('patch_proof') or {}),      # evidence per site and the notes' last
                         perf_note_last=dict(events.get('perf_note_last') or {}),   # places; (F12) recompilations seen
                         recompiles=int(events.get('recompiles') or 0))
            if events.get('patch_proof_whole'):                                # (F13: the lines read whole, when that
                extra['patch_proof_whole'] = dict(events['patch_proof_whole'])  # says more than the cut ones)
        res = {'fidelity': fidelity, 'steps': int(steps), 'valid': v.valid, 'step_time': v.step_time, 'aborted': aborted,
               'reasons': v.reasons[:3], 'failed_checks': failed, 'gpu_hours': gpu_s / 3600.0,
               'failure': ev.get('failure'), 'memory_peak_gib': ev.get('memory_peak_gib'), 'rsd_window': ev.get('rsd_window'),
               **({'memory_peak_measured_gib': ev['memory_peak_measured_gib']}          # (D3: the harness's measured
                  if ev.get('memory_peak_measured_gib') is not None else {}), **extra}   # peak, when it has one)
        c.results[fidelity] = res
        if not v.valid and not aborted and not slow:
            self._note_unrunnable(c, receipt, v)
        if not v.valid and not aborted and fidelity == 'probe':
            c.status = 'infeasible'                    # a probe failure (OOM, crash, wrong job) ends the candidate
            c.verdict = v.to_dict()
        row = {'index': len(self.screens), 'candidate_id': cid, 'kind': c.kind, 'fidelity': fidelity, 'steps': int(steps),
               'valid': v.valid, 'step_time': v.step_time, 'aborted': aborted, 'reasons': v.reasons[:4],
               'gpu_hours': gpu_s / 3600.0, 'cum': self.ledger.snapshot(), 'best_valid': self.best_valid(), **extra}
        self.screens.append(row)
        self.event('trial', row)
        if self.ledger.exhausted():
            self.stop_flag = 'budget:' + self.ledger.exhausted()[0]
        return row

    def reopen_start_slow(self, cid: str) -> bool:
        """hls3 (F1a): a candidate whose probe its own start cap stopped (`start_slow`, not a start failure) back to
        proposed - once - for its one full retry; False for any other candidate (nothing changes)."""
        c = self.candidates.get(cid)
        res = (c.results.get('probe') or {}) if c is not None else {}
        if c is None or c.status != 'infeasible' or not res.get('start_slow') or res.get('reopened'):
            return False
        res['reopened'] = True
        c.status, c.verdict = 'proposed', None
        self.event('reopen', {'candidate_id': cid, 'why': 'start_slow: one full retry'})
        return True

    def rescore(self, cid: str) -> dict:
        """One more full-length run of a screened candidate (the engine's paired confirmation of a composite near the
        noise): charged like a trial (GPU-hours and wall, not a screen), refused when the budget's reserve would be
        touched, never changes the candidate's verdict or status."""
        # A confirmation pair is one statistical measurement. Controls are handled
        # at the next engine boundary, so stopping cannot turn half a pair into failure.
        if self.stop_flag:
            return {'error': f'search stopped: {self.stop_flag}'}
        c = self.candidates.get(cid)
        if c is None or not (c.verdict or {}).get('valid'):
            return {'error': f'{cid} has no valid full verdict'}
        limit, cap = self._admission()
        if limit:
            return {'error': f'budget ({limit})'}
        t0 = self.clock()
        kw = {'candidate_id': cid}
        if cap is not None:
            kw['wall_cap'] = cap
        try:
            receipt = self.runner.run(self.task, c.config, c.overlay or None, 'rescore', self.run_id, **kw)
        except InfrastructureError as exc:             # no verdict: the search stops
            wall = max(0.0, self.clock() - t0)
            self.ledger.charge_screen(self._world(None) * wall / 3600.0, wall, False, count=self.count_rescores)
            return self._infrastructure_stop(cid, exc)
        except Exception as exc:                       # runner contract violation: recorded, never raised
            receipt = {'valid': False, 'invalid_reasons': [f'runner error: {type(exc).__name__}: {exc}'], 'step_time': None}
        parent = self.candidates[c.parent].receipt if c.parent and self.candidates[c.parent].receipt else None
        try:
            v = self.judge(receipt, parent, self.reference)
        except Exception as exc:                       # noqa: BLE001 - a judge crash invalidates
            v = Verdict(valid=False, reasons=[f'judge error: {type(exc).__name__}: {exc}'])
        wall = max(0.0, self.clock() - t0)
        gpu_s = gpu_seconds(receipt)
        known = gpu_s is not None and not (isinstance(receipt.get('cost'), dict)
                                           and receipt['cost'].get('accounting_complete') is False)
        if gpu_s is None:
            gpu_s = self._world(receipt) * wall
        self.ledger.charge_screen(gpu_s / 3600.0, wall_seconds(receipt) or wall, known, count=self.count_rescores)
        ev = receipt.get('evidence') if isinstance(receipt.get('evidence'), dict) else {}
        res = {'valid': v.valid, 'step_time': v.step_time, 'gpu_hours': gpu_s / 3600.0, 'reasons': v.reasons[:3],
               'split_half': ev.get('split_half'), 'split_half_robust': ev.get('split_half_robust')}
        c.results.setdefault('rescore', []).append(res)
        row = {'index': len(self.screens), 'candidate_id': cid, 'attempt_id': receipt.get('attempt_id'), 'kind': c.kind,
               'fidelity': 'rescore', 'valid': v.valid,
               'step_time': v.step_time, 'aborted': False, 'reasons': v.reasons[:4], 'gpu_hours': gpu_s / 3600.0,
               'cum': self.ledger.snapshot(), 'best_valid': self.best_valid()}
        self.screens.append(row)
        self.event('rescore', row)
        if self.ledger.exhausted():
            self.stop_flag = 'budget:' + self.ledger.exhausted()[0]
        return {**res, 'evidence': ev}

    def _expected_wall(self) -> float:
        est = self.ledger.mean_screen() or self.ledger.prior
        return float(est[1]) if est and est[1] else 900.0

    def _admission(self) -> tuple[str | None, int | None]:
        """(limiting dimension or None, wall cap for this attempt or None).

        Runners that enforce a per-attempt wall cap (HarnessScreenRunner) are admitted with their worst case,
        world x (cap + stop grace), while the worst case of both locked-pair members stays reserved; the attempt's cap
        shrinks to what fits, and a screen that could not reach its expected length is not started. Other runners
        (synthetic, CPU tests) are admitted with the mean measured cost plus the pair reserve."""
        bounded = getattr(self.runner, 'bounded_cap', None)
        if bounded is None:
            return self.ledger.admit_screen(), None
        ex = self.ledger.exhausted()
        if ex:
            return ex[0], None
        world, expected = self._world(None), self._expected_wall()
        pair_cap = self.runner.pair_wall_cap(expected)
        pair_gpu_h = 2 * self.runner.worst_case_gpu_hours(world, pair_cap)
        pair_wall = 2 * (pair_cap + self.runner.stop_grace_s)
        gpu_left = self.budget.gpu_hours - self.ledger.used['gpu_hours'] - pair_gpu_h
        wall_left = self.budget.wall_seconds - self.ledger.wall() - pair_wall
        cap = bounded(gpu_left, world, wall_left, expected)
        if cap is None:
            by_gpu = gpu_left * 3600.0 / max(1, world)
            return ('gpu_hours' if by_gpu <= wall_left else 'wall_seconds'), None
        self.pair_wall_cap = pair_cap
        self.pair_reserve = {'gpu_hours': pair_gpu_h, 'wall_seconds': pair_wall, 'member_wall_cap': pair_cap}
        return None, cap

    def _infrastructure_stop(self, cid: str, exc: Exception) -> dict:
        """A transport failure (InfrastructureError, after the runner's own retries): the candidate gets no verdict and
        stays proposed, nothing is counted against it, and the search stops with the reason."""
        msg = str(exc)[:300]
        self.stop_flag = f'infrastructure: {msg}'
        self.event('stop', {'reason': self.stop_flag, 'candidate_id': cid})
        return {'error': f'infrastructure failure: {msg}', 'infrastructure': True}

    def _world(self, receipt) -> int:
        try:
            w = int((receipt or {}).get('world_size'))
            if w > 0:
                return w
        except (TypeError, ValueError):
            pass
        if self.hooks is not None:
            return int(getattr(self.task, 'world_size', None) or self.task_raw.get('world_size') or 1)
        return int(self.task_raw['hardware']['gpus'])

    # ---------------------------------------------------------------- selection
    def best_valid(self) -> str | None:
        ok = [(c.verdict['step_time'], c.screen_index if c.screen_index is not None else -1, c.id) for c in self.candidates.values()
              if c.verdict and c.verdict.get('valid') and c.verdict.get('step_time')]
        return min(ok)[2] if ok else None

    def budget_view(self) -> dict:
        b, u = self.budget, self.ledger.snapshot()
        return {'screens': f'{u["screens"]}/{b.screens}', 'gpu_hours': f'{u["gpu_hours"]:.2f}/{b.gpu_hours}',
                'llm_tokens': f'{u["llm_tokens"]}/{b.llm_tokens}', 'wall_h': f'{u["wall_seconds"] / 3600:.2f}/{b.wall_seconds / 3600:.2f}'}

    def candidate_table(self, limit: int = 20) -> list[dict]:
        rows = []
        for c in list(self.candidates.values())[-limit:]:
            parent = self.candidates[c.parent].config if c.parent else None
            rows.append(c.summary(parent))
        return rows

    def finalize(self, *, selected: str | None = None, stop_reason: str, extra: dict | None = None,
                 override: bool = True) -> SearchTrace:
        """override=False keeps a valid `selected` (a noise-aware incumbent) even when a noisier candidate measured faster."""
        best = self.best_valid()
        chosen = selected if selected in self.candidates and (self.candidates[selected].verdict or {}).get('valid') else None
        if chosen is None:
            chosen = best or 'c0'
        elif override and best and self.candidates[best].verdict['step_time'] < self.candidates[chosen].verdict['step_time']:
            self.event('selection_override', {'agent_choice': selected, 'best_valid': best})
            chosen = best
        c = self.candidates[chosen]
        start = self.candidates['c0']
        extra = dict(extra or {})
        if self.pair_wall_cap:
            extra.update(pair_wall_cap=self.pair_wall_cap, pair_reserve=self.pair_reserve)
        trace = SearchTrace(
            schema=TRACE_SCHEMA, run_id=self.run_id, arm=self.arm,
            task_id=self.task_raw['id'] if self.hooks is None else getattr(self.task, 'id', self.task_raw.get('id')), mode=self.mode,
            package_digest=self.package_digest, driver=self.driver_info, judge=type(self.judge).__name__,
            budget=self.budget.as_dict(), used=self.ledger.snapshot(), accounting_complete=self.ledger.accounting_complete,
            stop_reason=stop_reason, selected=chosen, selected_config=c.config, selected_overlay=c.overlay,
            selected_step_time=(c.verdict or {}).get('step_time'), start=start.id,
            start_step_time=(start.verdict or {}).get('step_time'),
            candidates=[{**cand.summary(self.candidates[cand.parent].config if cand.parent else None),
                         'config_digest': digest(cand.config), 'overlay_digest': cand.overlay_digest,
                         'attempt_id': (cand.receipt or {}).get('attempt_id'), 'rationale': cand.rationale[:400],
                         **cand.deviations()}
                        for cand in self.candidates.values()],
            screens=self.screens, llm_calls=self.llm_calls, skill_hits=self.skill_hits, memory_hits=self.memory_hits,
            memory_writes=self.memory_writes, tool_calls=self.tool_calls, invalid_proposals=self.invalid, verifier=self.verifier_stats,
            started_utc=self.started_utc, ended_utc=_utc(), out_dir=str(self.out_dir) if self.out_dir else None, extra=extra or {})
        self.event('finalize', {'selected': chosen, 'stop_reason': stop_reason})
        if self.out_dir:
            (self.out_dir / 'trace.json').write_text(json.dumps(trace.to_dict(), indent=1, ensure_ascii=False), encoding='utf-8')
        return trace
