"""Phase E ENV-A: the current HLS Engine over the unchanged scoring adapter (D-RB-19)."""
from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

from ... import search
from ...agent.capability import infer
from ...agent.budget import SearchLedger
from ...agent.search import SearchSession
from ...experience import runs as exp_runs
from ..common.env_judge import EnvJudge
from .driver import PHASE_E_MODEL
from .hooks import EnvAHooks
from .task import S0_WORK, arm_deviations


class ScoringLedger(SearchLedger):
    """ENV-A has one fidelity: Engine confirmation/recheck runs are scorings too."""

    def charge_screen(self, gpu_hours, wall_seconds, known=True, count=True):
        super().charge_screen(gpu_hours, wall_seconds, known, count=True)


class EnvAContext(search.Context):
    """Keep the arm identity and optional test clock at the existing Context.session seam."""

    def session(self, arm: str, **kw):
        if self.options.get('clock') is not None:
            kw['clock'] = self.options['clock']
        s = super().session(self.task.arm, **kw)
        ledger = ScoringLedger(s.budget, s.clock)
        ledger.prior, ledger.t0 = s.ledger.prior, s.ledger.t0
        s.ledger = ledger
        return s


def project_view(task, base_files):
    """The product experience layer's project view: Composer/FSDP, not a Megatron surrogate.

    Use its ordinary registry and shipped/user packs, without inventing mechanism mappings or evidence.
    The schema and reference values come from the existing ENV-A knobs and starting solution.
    """
    import yaml
    cfg = yaml.safe_load(base_files['finetune.yaml']) or {}
    ref = {'tokenize.num_workers': None, 'skip_valid_copy': False}
    props = task.knob_schema['properties']
    for name in props:
        if name in ref:
            continue
        val = cfg
        for part in name.split('.'):
            val = val.get(part) if isinstance(val, dict) else None
        ref[name] = val
    knobs = {n: {'choices': list(s['enum']), 'default': ref[n]} for n, s in props.items()}
    raw = {'framework': 'composer', 'versions': {'framework': '0.22.0'},
           'hardware': {'gpu_type': 'H100', 'memory_gib': 80},
           'model': {'params_b': task.facts['model']['params'] / 1e9, 'family': 'dense'},
           'workload': {'seq_len': S0_WORK['seq_len'],
                        'global_batch_tokens': S0_WORK['global_train_batch_size'] * S0_WORK['seq_len']},
           # Fixed job semantics inform transfer feasibility; they are not extra knob choices.
           'states': dict(task.facts['fixed_semantics']), 'search': {}}
    return SimpleNamespace(raw=raw, gpus=task.world_size, nodes=1, knobs=knobs,
                           reference_config=lambda: dict(ref))


def _with_start(trace, start_trace, budget, elapsed_before_search):
    """Present one ordinary SearchTrace, including S0, while the Engine only spends the remainder.

    Preserve the unmodified Engine trace in search/trace.json and the start trace in start/trace.json;
    the arm-level trace has the original budget and cumulative charges from both. Locked scoring stays outside.
    """
    t = copy.deepcopy(trace)
    charge = start_trace['used']
    t['extra']['search_budget'] = t['budget']
    t['extra']['start_charge'] = charge
    t['budget'] = budget.as_dict()
    for c in t['candidates']:
        if c['id'] == 'c0':
            c['screen_index'] = next(s['screen_index'] for s in start_trace['candidates'] if s['id'] == 'c0')
        elif c['screen_index'] is not None:
            c['screen_index'] += len(start_trace['screens'])
    for row in t['screens']:
        row['index'] += len(start_trace['screens'])
        for k in ('screens', 'gpu_hours', 'llm_tokens', 'llm_usd'):
            row['cum'][k] += charge[k]
        row['cum']['wall_seconds'] += elapsed_before_search
        if 'fractions' in row:
            row['fractions'] = {k: row['cum'][k] / t['budget'][k] if t['budget'][k] else 0.0
                                for k in row['fractions']}
    t['screens'] = copy.deepcopy(start_trace['screens']) + t['screens']
    for k in ('screens', 'gpu_hours', 'llm_tokens', 'llm_usd'):
        t['used'][k] += charge[k]
    t['used']['wall_seconds'] += elapsed_before_search
    t['accounting_complete'] &= start_trace['accounting_complete']
    t['used']['accounting_complete'] = t['accounting_complete']
    t['started_utc'] = start_trace['started_utc']
    return t


def _publish_events(out, trace, parts):
    """Join the sequential S0/search event logs using the combined trace's row boundaries.

    Keep producer order (including calls after the final screen), UTCs and rescore event kinds.
    The CLI appends lease events after this arm-root stream is published; sublogs stay unchanged.
    """
    events, row_index = [], 0
    for source, elapsed in parts:
        for original in source:
            event = copy.deepcopy(original)
            event['i'] = len(events)
            event['wall'] += elapsed
            if event['kind'] in ('screen', 'rescore'):
                event['payload'] = trace['screens'][row_index]
                row_index += 1
            events.append(event)
    (out / 'events.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in events),
                                     encoding='utf-8')


def run_a_is2(task, driver, runner, *, run_id: str, out_dir=None, clock=None,
              dry_run: bool = False, log=None) -> dict:
    """S0 first, then search.run('hls', Context) with 59 of the 60 scorings remaining by default."""
    from .arms import locked_rescore_selection

    if task.arm != 'A-IS2':
        raise ValueError(f'{task.arm}: not A-IS2')
    runner.deviations = arm_deviations(task.arm)
    hooks = EnvAHooks(task, base_files=runner.base)
    budget = task.search_budget()
    capability = infer(getattr(driver, 'model', PHASE_E_MODEL))
    say = log or (lambda m: None)
    meta = {'arm': task.arm, 'capability': capability, 'deviations': list(arm_deviations(task.arm))}
    if dry_run:
        return {**meta, 'dry_run': True, 'strategy': 'hls', 'budget': budget.as_dict(),
                'layers': list(hooks.layers), 'experience': 'auto', 'uid': runner.uid,
                'driver': driver.describe(), 'multi_fidelity': False}
    prep = runner.prepare(run_id)
    out = Path(out_dir) if out_dir else None
    starter = SearchSession(task=task, budget=budget, start_config={}, runner=runner, judge=EnvJudge('rebench_llmf'),
                            hooks=hooks, run_id=run_id, arm=task.arm, clock=clock,
                            out_dir=out / 'start' if out else None)
    starter.screen('c0')
    start = starter.candidates['c0']
    start_trace = starter.finalize(stop_reason=starter.stop_flag or 'start_complete').to_dict()
    event_parts = [(starter.events, 0.0)]
    if out and start.receipt is not None:
        (out / 'receipts').mkdir(parents=True, exist_ok=True)
        (out / 'receipts' / 'c0.json').write_bytes((out / 'start' / 'receipts' / 'c0.json').read_bytes())
    coverage = None
    if not (start.verdict or {}).get('valid'):
        trace = start_trace
        trace['stop_reason'] = starter.stop_flag or 'invalid_start'
        locked = {'locked': None, 'reason': 'no valid start: search and locked scoring skipped'}
    else:
        exp = exp_runs.plan_for(project_view(task, runner.base), start.receipt, select='auto', driver=driver, log=say)
        # The shared payload builder omits states; carry this adapter's fixed context into transfer too.
        exp['payload']['target']['states'] = dict(exp['fingerprint']['states'])
        coverage = exp_runs.doctor_view(exp)
        elapsed = starter.ledger.wall()
        remaining = dataclasses.replace(budget, screens=max(0, budget.screens - 1),
                                        gpu_hours=max(0.0, budget.gpu_hours - start_trace['used']['gpu_hours']),
                                        wall_seconds=max(0.0, budget.wall_seconds - elapsed))
        options = {'full_steps': S0_WORK['optimizer_steps_batches'], 'warmup': 0, 'multi_fidelity': False,
                   'job_gpu_hours': None, 'capability': capability, 'prior_results': [], 'run_noise': None,
                   'noise_pairs': [], 'experience': exp, 'log': say, 'ablate': [], 'accounting': None, 'clock': clock}
        ctx = EnvAContext(task=task, budget=remaining, start_config={}, runner=runner, judge=EnvJudge('rebench_llmf'),
                          hooks=hooks, run_id=run_id, out_dir=out / 'search' if out else None,
                          start_receipt=start.receipt, reference_receipt=start.receipt, driver=driver, options=options)
        trace = _with_start(search.run('hls', ctx).to_dict(), start_trace, budget, elapsed)
        event_parts.extend((s.events, elapsed) for s in ctx.sessions)
        locked = locked_rescore_selection(runner, task, trace, run_id=run_id)
    trace['extra'].update(meta, experience=coverage)
    if out:
        _publish_events(out, trace, event_parts)
        trace['out_dir'] = str(out)
        (out / 'trace.json').write_text(json.dumps(trace, indent=1, default=str), encoding='utf-8')
        (out / 'experience.json').write_text(json.dumps(coverage, indent=1, default=str), encoding='utf-8')
    return {**meta, 'trace': trace, 'locked': locked, 'prepare': prep, 'experience': coverage}
