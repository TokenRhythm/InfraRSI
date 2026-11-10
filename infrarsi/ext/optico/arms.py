"""ENV-B arm launchers (design §3.3-§3.6): each call runs one arm of one workload on one leased node; when to run
what is the orchestrator's decision (design §3.7 / §6.3).

  B-DEF   the script defaults through OptiCoRunner (OptiCo's own start; also the G-B2 reference)
  B-ED    one trial of the planner projection (b_ed.pick)
  B-NAT   OptiCo's run_circle.sh loop (native.NativeDriver; D-OC-15 strip_think_format, D-OC-16 sanitizer)
  B-RND   SearchSession + seed-42 conditional sampler (b_rnd), same validator, default start + 20 new trials
  B-IS    agent.optimize with OptiCoHooks on the frozen A0 package, Opus 5.5 (B-ISW: the weak driver, D-OC-10)
  B-IS2   the current search (search.run: the Engine with HLS, the capability inferred from the model, the experience
          plan of OptiCo's knob space), the same task, runner, judge and budget (B-IS2-noexp: the experience off)
  locked  the arm's final selection and the default back to back on one node, order by arm index parity (§1.4);
          labels use margin_<W>.json (margin.py, AMENDMENT_EXT_002 run-to-run estimator); locked.json's within-run
          fields (diag_*) are diagnostics only
G-B2: report-only until the enabling rule holds; gb2_policy() builds the policy from the B-DEF and B-ED receipts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from ..common.env_judge import EnvJudge
from . import b_ed as bed
from . import b_rnd
from . import guards as gb
from . import params as pm
from . import workloads as wl
from .runner import OptiCoRunner
from .task import OptiCoHooks, make_task

ARM_INDEX = {'B-DEF': 0, 'B-ED': 1, 'B-NAT': 2, 'B-IS': 3, 'B-RND': 4, 'B-ISW': 5, 'B-IS2': 6, 'B-IS2-noexp': 7}
# the registry mechanisms OptiCo's knobs set (experience/data/mechanisms.yaml): what the experience reads of them
MECHANISMS = {'TP': ('tp', None), 'PP': ('pp', None), 'MICRO_BATCH_SIZE': ('micro_batch', None),
              'OVERLAP_GRAD_REDUCE': ('dp_grad_overlap', {'true': 'on', 'false': 'off'}),
              'OVERLAP_PARAM_GATHER': ('dp_param_gather_overlap', {'true': 'on', 'false': 'off'}),
              'AC': ('recompute', {'none': 'none', 'sel': 'selective', 'full': 'full'}),
              'FUSED_CROSS_ENTROPY': ('fused_cross_entropy', {'true': 'on', 'false': 'off'})}
MARGIN_FLOOR, MARGIN_K = 0.03, 3.0


def _load(path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def gb2_policy(default_receipt=None, bed_receipt=None, *, ref: float | None = None, hard: bool = False) -> gb.GB2Policy | None:
    """From receipt files (paths or dicts): both -> design §3.4 enabling rule; default only -> report-only reference."""
    d = _load(default_receipt) if isinstance(default_receipt, (str, Path)) else default_receipt
    e = _load(bed_receipt) if isinstance(bed_receipt, (str, Path)) else bed_receipt
    if d is not None and e is not None:
        return gb.policy_from_receipts(d, e)
    if d is not None:
        return gb.GB2Policy(reference_loss1=gb.receipt_loss1(d), hard=False, reference_attempt=d.get('attempt_id'),
                            rule='report only: enabling rule not evaluated (no B-ED receipt)')
    if ref is not None:
        return gb.GB2Policy(reference_loss1=ref, hard=hard, rule='explicit reference' + (' (hard)' if hard else ''))
    return None


def selection_from(path) -> dict:
    """The arm's final parameter set: a SearchSession trace.json (selected_config), a B-NAT result.json (best success
    row = the published protocol's result), or a receipt (params)."""
    p = Path(path)
    obj = _load(p)
    if 'selected_config' in obj:                  # SearchSession trace.json, or B-NAT's (= its best success row)
        if obj['selected_config'] is None:
            raise ValueError(f'{p}: the arm has no selection (stop_reason {obj.get("stop_reason")!r})')
        return dict(obj['selected_config'])
    if 'best_success_row' in obj:
        best = obj['best_success_row'] or {}
        rec = _load(p.parent / 'receipts' / f'{best["attempt_id"]}.json')
        return pm.typed(rec['params'], strict=True)
    if isinstance(obj.get('params'), dict):
        return dict(obj['params'])
    raise ValueError(f'{p}: no selection (expected trace.json, a B-NAT result.json or a receipt)')


def _runner(workload, ops, out_dir, gb2, log, **kw):
    return OptiCoRunner(ops, workload, out_dir=out_dir, gb2=gb2, log=log or (lambda m: print(m, flush=True)), **kw)


def run_trial(workload, ops, *, run_id: str, out_dir, params: dict, arm: str = 'B-DEF', purpose: str = 'b_def',
              candidate_id: str = 'c1', gb2=None, log=None) -> dict:
    task = make_task(workload, arm)
    return _runner(workload, ops, out_dir, gb2, log).run(task, params, None, purpose, run_id, candidate_id=candidate_id)


def run_b_ed(workload, ops, *, run_id: str, out_dir, gb2=None, log=None) -> dict:
    pick = bed.pick(workload)
    if pick['params'] is None:
        raise RuntimeError(f'B-ED: {pick["explanation"]}')
    r = run_trial(workload, ops, run_id=run_id, out_dir=out_dir, params=pick['params'], arm='B-ED', purpose='b_ed',
                  candidate_id='bed', gb2=gb2, log=log)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'b_ed_pick.json').write_text(json.dumps(pick, indent=1), encoding='utf-8')
    return {'receipt': r, 'pick': pick}


def _session(task, runner, hooks, budget, run_id, arm, out_dir, start, clock=None):
    from ...agent.search import SearchSession
    kw = {'clock': clock} if clock else {}
    return SearchSession(task=task, budget=budget, runner=runner, judge=EnvJudge(env='optico'), run_id=run_id, arm=arm,
                         out_dir=out_dir, start_config=start, layers=('L0',), hooks=hooks, **kw)


def run_b_rnd(workload, ops, *, run_id: str, out_dir, screens: int | None = None, wall_s: float | None = None,
              gpu_hours: float | None = None, gb2=None, seed: int = b_rnd.SEED, log=None, runner=None, clock=None) -> dict:
    w = wl.get(workload)
    task = make_task(w, 'B-RND')
    budget = task.search_budget(**{k: v for k, v in (('screens', screens), ('wall_seconds', wall_s), ('gpu_hours', gpu_hours))
                                   if v is not None})
    runner = runner or _runner(w, ops, out_dir, gb2, log)
    hooks = OptiCoHooks(task, runner=runner)
    s = _session(task, runner, hooks, budget, run_id, 'B-RND', out_dir, dict(w.defaults), clock)
    s.screen('c0')
    sampler = b_rnd.OptiCoRandomSampler(w, seed=seed)
    seen = {pm.params_key(c.config) for c in s.candidates.values()}
    stop = None
    while stop is None:
        if s.stop_flag or s.ledger.exhausted():
            stop = s.stop_flag or 'budget:' + s.ledger.exhausted()[0]
            break
        p = sampler.next(seen)
        if p is None:
            stop = 'no_valid_sample'
            break
        seen.add(pm.params_key(p))
        fb = s.propose_config(config=p, base='c0', origin='random')
        if 'candidate_id' in fb:
            s.screen(fb['candidate_id'])
    return s.finalize(stop_reason=stop, extra={'seed': seed, 'draws': sampler.draws, 'sampler': 'ext.optico.b_rnd'}).to_dict()


def run_b_is(workload, ops, *, run_id: str, out_dir, weak: bool = False, screens: int | None = None,
             wall_s: float | None = None, llm_tokens: int | None = None, llm_usd: float | None = None,
             gpu_hours: float | None = None, gb2=None, package=None, backend=None, log=None, runner=None, clock=None) -> dict:
    """B-IS (B-ISW with weak=True): InfraRSI's optimize() on the frozen A0 package, tools off, L0 = OptiCo's knobs."""
    from ...agent.optimize import optimize
    w = wl.get(workload)
    arm = 'B-ISW' if weak else 'B-IS'
    task = make_task(w, arm)
    over = {k: v for k, v in (('screens', screens), ('wall_seconds', wall_s), ('llm_tokens', llm_tokens),
                              ('llm_usd', llm_usd), ('gpu_hours', gpu_hours)) if v is not None}
    budget = task.search_budget(**over)
    runner = runner or _runner(w, ops, out_dir, gb2, log)
    if weak:
        runner.deviations = tuple(sorted(set(runner.deviations) | {'D-OC-10'}, key=lambda i: int(i.rsplit('-', 1)[1])))
    if package is None:
        from ...agent.arms import a0_package
        package = a0_package()
    if backend is None:
        from ...agent.arms import DRIVERS
        backend = DRIVERS['weak'](structured=True)[1] if weak else DRIVERS['strong']()
    hooks = OptiCoHooks(task, runner=runner, frozen_package_digest=package.digest)
    kw = {'clock': clock} if clock else {}
    trace = optimize(task, package, backend, budget, dict(w.defaults), runner, judge=EnvJudge(env='optico'), run_id=run_id,
                     arm=arm, out_dir=out_dir, mode='weak' if weak else 'strong', hooks=hooks, **kw)
    return trace.to_dict()


def project_view(workload):
    """What InfraRSI's experience layer reads of a project (fingerprint, plan), for an OptiCo dense workload on one
    8 x H100 node: Megatron-Core, the script's model shape, fp16, the global batch, and the knobs with the registry
    mechanisms they set (the others are named as having none)."""
    from types import SimpleNamespace
    from ...discover.workload import count_params
    w = wl.get(workload)
    props = pm.knob_schema(w)['properties']
    knobs = {}
    for name, sch in props.items():
        k = {'type': 'bool' if sch.get('type') == 'boolean' else 'int' if sch.get('type') == 'integer' else 'str',
             'default': w.defaults[name]}
        if sch.get('enum'):
            k['choices'] = list(sch['enum'])
        if name in MECHANISMS:
            k['mechanism'] = MECHANISMS[name][0]
            if MECHANISMS[name][1]:
                k['states'] = dict(MECHANISMS[name][1])
        if name == 'TP':                                 # the scripts: --sequence-parallel --tp-comm-overlap with TP > 1
            k['implies'] = {str(c): dict.fromkeys(('sequence_parallel', 'tp_comm_overlap'), 'on' if int(c) > 1 else 'off')
                            for c in k.get('choices') or [w.defaults[name]]}
        knobs[name] = k
    f = count_params(w.hf_like_config())
    model = {'params_b': round(f['params_total'] / 1e9, 3), 'family': 'dense', 'layers': w.num_layers, 'hidden': w.hidden,
             'ffn': w.ffn, 'heads': w.heads, 'kv_heads': w.kv_heads, 'vocab': w.vocab, 'untie': not w.tied_embeddings}
    raw = {'hardware': {'gpu_type': 'NVIDIA H100 80GB HBM3', 'memory_gib': round(wl.GPU_MEM_LIMIT_MIB / 1024, 1)},
           'framework': 'megatron', 'versions': {'framework': '0.15.0'}, 'model': model,
           'workload': {'seq_len': wl.SEQ, 'global_batch_tokens': wl.GBS * wl.SEQ}, 'search': {},
           # the scripts' fixed settings outside the knob space (run_<model>.sh: fp16, the distributed optimizer);
           # sequence parallelism and the TP communication overlap follow TP > 1 (the TP knob's implies)
           'states': {'precision': 'fp16', 'distributed_optimizer': 'on'}}
    return SimpleNamespace(raw=raw, gpus=wl.GPUS, nodes=1, knobs=knobs, name=f'optico-{w.key.lower()}',
                           window=wl.WINDOW, reference_config=lambda: dict(w.defaults))


def run_b_is2(workload, ops, *, run_id: str, out_dir, experience: bool = True, screens: int | None = None,
              wall_s: float | None = None, llm_tokens: int | None = None, llm_usd: float | None = None,
              gpu_hours: float | None = None, gb2=None, backend=None, log=None, runner=None, clock=None) -> dict:
    """B-IS2 (B-IS2-noexp with experience=False): search.run('hls') on OptiCo's task (the Engine: HLS ledger,
    value/cost scheduling, numeric stop; one fidelity, OptiCo's trial has no shorter run), the capability inferred from
    the model (Opus 5.5 through the CLI by default), the experience plan of the knob space (the H20 / H200 Megatron
    packs are a cross-environment prior here: disclosed, and the -noexp arm separates them)."""
    from ... import search
    from ...agent.capability import infer
    from ...experience import runs as exp_runs
    w = wl.get(workload)
    arm = 'B-IS2' if experience else 'B-IS2-noexp'
    task = make_task(w, arm)
    over = {k: v for k, v in (('screens', screens), ('wall_seconds', wall_s), ('llm_tokens', llm_tokens),
                              ('llm_usd', llm_usd), ('gpu_hours', gpu_hours)) if v is not None}
    budget = task.search_budget(**over)                # 21 trials with the start, every full run counted (re-runs too)
    runner = runner or _runner(w, ops, out_dir, gb2, log)
    if backend is None:
        from ...agent.arms import DRIVERS
        backend = DRIVERS['strong']()
        if hasattr(backend, 'max_retries'):            # as the product's claude-cli driver (project.llm.make_driver)
            backend.max_retries = 1
    hooks = OptiCoHooks(task, runner=runner)
    say = log or (lambda m: None)
    exp = exp_runs.plan_for(project_view(w), None, select=None if experience else False, driver=backend, log=say) \
        if experience else None
    # the start is screened in the session (no start receipt: B-IS's first counted screen), re-runs count as trials,
    # the trace is the arm's (trace.json / events.jsonl at the run root, arm B-IS2 / B-IS2-noexp)
    options = {'full_steps': wl.TRAIN_ITERS, 'warmup': wl.WINDOW[0] - 1, 'multi_fidelity': False, 'job_gpu_hours': None,
               'capability': infer(getattr(backend, 'model', None)), 'prior_results': [], 'run_noise': None,
               'noise_pairs': [], 'experience': exp, 'log': say, 'ablate': [], 'accounting': None, 'arm': arm,
               'count_rescores': True}
    ctx = search.Context(task=task, budget=budget, start_config=dict(w.defaults), runner=runner,
                         judge=EnvJudge(env='optico'), hooks=hooks, run_id=run_id, out_dir=Path(out_dir), driver=backend,
                         options=options)
    say(f'[optico] {arm} {w.key}: search hls, budget {budget.as_dict()}, capability {options["capability"]}')
    return search.run('hls', ctx).to_dict()


def within_run_margin(default_receipt: dict) -> tuple[float, float | None]:
    """DIAGNOSTIC ONLY (struck as a decision rule by AMENDMENT_EXT_002 Q1): max(3 %, 3 x RSD of the default's
    iterations 3..5 TFLOPS within one run). The label margin is margin.py's run-to-run estimator."""
    rsd = ((default_receipt.get('metric') or {}).get('raw') or {}).get('tflops_window_rsd')
    return max(MARGIN_FLOOR, MARGIN_K * float(rsd or 0.0)), rsd


def is_default_selection(params: dict, workload) -> bool:
    """True when the selection, normalized as the validator does, is the workload's default configuration -- the start
    of every ENV-B arm (B-NAT's cycle-0 row, screen c0 of B-IS / B-RND / B-ISW)."""
    w = wl.get(workload)

    def key(p):
        norm, errs, _ = pm.validate_optico_params(dict(p), w)
        try:
            return pm.params_key(pm.typed(norm if not errs else p, strict=True))
        except (ValueError, TypeError, AttributeError):
            return None
    k = key(params) if isinstance(params, dict) else None
    return k is not None and k == key(w.defaults)


def run_locked(workload, ops, *, run_id: str, out_dir, params: dict, arm: str, gb2=None, log=None, runner=None) -> dict:
    """The arm's final selection and the default back to back on one node; even arm index -> default first. A selection
    equal to the defaults (the start) runs nothing (D20 / AMENDMENT_EXT_003 Q2: no A/A block): locked.json records
    `skipped`, y = 0, speedup 1.0, InvalidAccept 0, both halves null; no receipt, so margin.py gets no sample from it."""
    w = wl.get(workload)
    native = arm == 'B-NAT'                    # OptiCo's own selection re-runs as OptiCo ran it (D-OC-13 n/a)
    norm, errs, _ = pm.validate_optico_params(params, w)
    if errs and not native:
        raise ValueError(f'locked selection invalid: {errs}')
    idx = ARM_INDEX.get(arm, 0)
    if is_default_selection(params, w):
        res = {'arm': arm, 'arm_index': idx, 'order': [], 'params': norm if not errs else params,
               'skipped': 'selection_is_default',
               'rule': 'D20 / AMENDMENT_EXT_003 Q2: the selection is the start (workload defaults); no locked run, '
                       'locked value = the start reference, y = 0, InvalidAccept 0',
               'default': None, 'candidate': None, 'speedup_vs_default': 1.0, 'y': 0.0, 'invalid_accept_locked': 0,
               'diag_within_run_margin': None, 'diag_within_run_rsd': None, 'diag_beyond_within_run_margin': None,
               'gb2': (gb2 or gb.GB2Policy()).to_dict()}
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'locked.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
        (log or (lambda m: print(m, flush=True)))(f'{run_id}: {arm} selected the defaults -> locked pair skipped (D20)')
        return res
    order = ['default', 'candidate'] if idx % 2 == 0 else ['candidate', 'default']
    task = make_task(w, arm)
    runner = runner or _runner(w, ops, out_dir, gb2, log)
    if native:
        runner.deviations = tuple(d for d in runner.deviations if d != 'D-OC-13')
    got = {}
    for who in order:
        p = dict(w.defaults) if who == 'default' else (norm if not errs else params)
        got[who] = runner.run(task, p, None, 'locked', run_id, candidate_id=f'locked-{who}', native=native)
    d, c = got['default'], got['candidate']
    td = ((d.get('metric') or {}).get('raw') or {}).get('tflops_per_gpu')
    tc = ((c.get('metric') or {}).get('raw') or {}).get('tflops_per_gpu')
    m, rsd = within_run_margin(d)
    speed = (tc / td) if d.get('valid') and c.get('valid') and td and tc else None
    res = {'arm': arm, 'arm_index': idx, 'order': order, 'params': norm if not errs else params,
           'default': {'attempt_id': d['attempt_id'], 'valid': d['valid'], 'tflops_per_gpu': td, 'iter_time_s': d['metric']['value']},
           'candidate': {'attempt_id': c['attempt_id'], 'valid': c['valid'], 'tflops_per_gpu': tc, 'iter_time_s': c['metric']['value'],
                         'invalid_reasons': c.get('invalid_reasons')},
           'speedup_vs_default': speed, 'y': math.log(speed) if speed else None,
           'invalid_accept_locked': 0 if c.get('valid') else 1,
           'diag_within_run_margin': m, 'diag_within_run_rsd': rsd, 'diag_beyond_within_run_margin': bool(speed and speed >= 1.0 + m),
           'diagnostics_note': 'diag_* use the within-run 3-sample RSD struck by AMENDMENT_EXT_002 Q1; labels use '
                               'margin_<W>.json (ext.optico.margin, run-to-run RSD of protocol default runs)',
           'gb2': (gb2 or gb.GB2Policy()).to_dict()}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'locked.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    return res
