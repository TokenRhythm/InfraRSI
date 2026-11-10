"""Builder A's dispatcher arms: RUN_MATRIX `arm` -> handler, exposed through DISPATCH_HANDLERS of infrarsi.agent,
infrarsi.rsi, infrarsi.redteam and infrarsi.baselines (imported by infrarsi.dispatch.handlers, owner D).

Search arms are built with D's `search_handler(search_fn, start='ED'|'FR')`, which supplies the task's start config,
its start receipt, the FR plan-A screen receipt (numerics reference) and the dispatcher's request-level runner (inside
the run-scoped lease). Each search_fn here:
  - screens through `HarnessScreenRunner` on the run's nodes (agent/harness_runner.py);
  - uses the task's search budget (task_specs search_budget; a RUN_MATRIX row may lower screens, LLM tokens and the
    wall cap; never raise them);
  - judges with the formal reward engine and the task's reward config (design/reward.yaml; speed is the only
    objective), except the E4 lenient arm (LenientJudge, then an offline full-reward audit);
  - runs the locked scoring pair {FR, selected} on input plan B back to back on the same nodes after the search
    (order by task index parity: even -> FR first, odd -> candidate first and then
    harness.recheck_against(candidate, FR plan B)); its GPU-hours are reported with the search's. The selection is
    promoted when its effective configuration and overlay differ from the reference's (c0 included); otherwise only
    the reference half runs and the arm falls back to the reference;
  - trusts a stored receipt only through receipt_trusted (D's HMAC, harness/signing.py), and takes the ED denominator
    only from the recorded, feasible ED run (ed_score_problems / ed_identity_problems);
  - writes execution/searches/<run_id>/{events.jsonl, trace.json, receipts/, calls/, result.json}; result.json
    carries the inputs of scoring.score_task (t_FR / t_cand locked, FR RSD, InvalidAccept, events, caps).
Non-search arms (R-arm reflection, M-arm design, red-team CPU check, weak-endpoint check) are plain handlers
fn(run, nodes, ctx) -> dict.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import time
from pathlib import Path, PurePosixPath

from ..contracts import EXECUTION_DIR, TaskSpec, utc_now
from .budget import Budget
from .harness_runner import HarnessScreenRunner, wire_config
from .judge import LenientJudge, RewardEngineJudge, Verdict, gpu_seconds, step_time, wall_seconds

PACKAGES_DIR = EXECUTION_DIR / 'packages'           # driver-side optimizer packages (A0, lineage A_k, snapshots)
LINEAGE_DIR = PACKAGES_DIR / 'lineage'
MONITOR_DIR = EXECUTION_DIR / 'monitor'
COST_LEDGER = EXECUTION_DIR / 'ledger' / 'cost_ledger.jsonl'


# ---------------------------------------------------------------- task facts, reward, budget
def task_reward(task) -> dict:
    """The reward config for formal judging: design/reward.yaml (speed only; the task's numerics tier is its tolerance)."""
    from ..reward import load_reward_cfg
    return load_reward_cfg()


def formal_judge(task) -> RewardEngineJudge:
    return RewardEngineJudge(task_reward(task))


def model_facts_for(task, remote=None, cache_dir=None) -> dict:
    """validate_l0's model facts (HF config) merged with D's parameter counts (discover.workload_facts)."""
    from ..actions import model_facts_from_hf_config
    from ..discover import workload
    cache = Path(cache_dir) if cache_dir else workload.CACHE
    facts = workload.workload_facts(task, remote=remote, cache_dir=cache)
    hf = json.loads((cache / workload.cache_name(task)).read_text(encoding='utf-8'))['config']
    mf = model_facts_from_hf_config(hf)
    total, active, expert = facts.get('params_total'), facts.get('params_active'), facts.get('params_expert')
    mf.update(params=total, active_params=active or total,
              expert_param_fraction=(expert / total) if expert and total else 0.0)
    return mf


def budget_for(task, run: dict, *, wall_multiplier: float = 1.0) -> Budget:
    b = Budget.from_task(task, run=run, wall_multiplier=wall_multiplier)
    if run.get('wall_hours_cap'):
        wall = min(b.wall_seconds, float(run['wall_hours_cap']) * 3600 * wall_multiplier)
        b = dataclasses.replace(b, wall_seconds=wall)
    return b


def attempt_cap(run: dict) -> int:
    from ..dispatch.handlers import attempt_cap as cap
    return cap(run)


def task_index(task) -> int:
    m = re.search(r'\d+', task.id)
    return int(m.group()) if m else 0


# ---------------------------------------------------------------- packages
def a0_package():
    """The frozen human package A0 (built once from skills/ and rsi/a0, then only loaded and digest-verified)."""
    from ..rsi.package import OptimizerPackage, build_a0
    path = PACKAGES_DIR / 'A0'
    if (path / 'manifest.json').exists():
        return OptimizerPackage.load(path)
    PACKAGES_DIR.mkdir(parents=True, exist_ok=True)
    return build_a0(path)


def lineage():
    from ..rsi.lineage import Lineage
    lin = Lineage(LINEAGE_DIR)
    if not lin.rows():
        lin.register(a0_package(), status='initial')
    return lin


def frozen_package(label: str):
    from ..rsi.package import OptimizerPackage
    lin = lineage()
    rec = lin.frozen(label)
    if rec is None:
        raise RuntimeError(f'package {label} is not frozen yet')
    if not lin.verify_frozen(label):
        raise RuntimeError(f'frozen package {label} changed after freeze')
    return OptimizerPackage.load(rec['path'])


def resolve_package(kind: str, ctx=None):
    from ..rsi.package import OptimizerPackage
    if kind == 'A0':
        return a0_package()
    if kind == 'current':
        return OptimizerPackage.load(lineage().current()['path'])
    if kind == 'A0_raw_dev_memory':
        from ..rsi.arms import rawmem_package
        return rawmem_package(ctx)
    return frozen_package(kind)


# ---------------------------------------------------------------- LLM drivers (module hooks; tests replace them)
def strong_driver():
    from .backends import make_backend
    return make_backend('claude')


# Route R2 verified on WIN0 (execution/weak_endpoint/STATUS.json, WEAK_ENDPOINT_REPORT.md): vLLM 0.26.0 serving
# DeepSeek-V4-Flash with TP4 on 4x H20 behind `ssh -N -L 18000:127.0.0.1:8000 -p 44685 user@116.169.116.30`; no API key;
# temperature 0, response_format json_object, chat_template_kwargs.thinking=false (thinking + JSON returns empty
# content); max-model-len 32768. API USD is 0; serving time is reported as H20 GPU-hours.
WEAK_STATUS = EXECUTION_DIR / 'weak_endpoint' / 'STATUS.json'
SELFHOST_DEFAULT = {'route': 'R2', 'model': 'deepseek-ai/DeepSeek-V4-Flash', 'base_url_local': 'http://127.0.0.1:18000/v1',
                    'engine': 'vllm', 'engine_version': '0.26.0', 'gpus_used': [4, 5, 6, 7], 'gpu_type': 'H20-96GB',
                    'max_model_len': 32768, 'extra_request_params': {'chat_template_kwargs': {'thinking': False}}}


def selfhost_settings() -> dict:
    """Route R2 settings: STATUS.json fields over the verified defaults; INFRARSI_WEAK_MODEL / _URL override."""
    st = dict(SELFHOST_DEFAULT)
    try:
        st.update({k: v for k, v in json.loads(WEAK_STATUS.read_text(encoding='utf-8')).items() if v is not None})
    except (OSError, ValueError):
        pass
    st['model'] = os.environ.get('INFRARSI_WEAK_MODEL') or st.get('served_model_name') or st['model']
    st['base_url_local'] = os.environ.get('INFRARSI_WEAK_URL') or st['base_url_local']
    st['engine_version'] = str(st.get('engine_version') or '').split('+')[0] or None
    # Only chat_template_kwargs may come from STATUS.json, and thinking is always off (thinking + JSON mode returns
    # empty content); model, temperature, output cap, streaming and JSON mode are fixed by the backend.
    ctk = dict((st.get('extra_request_params') or {}).get('chat_template_kwargs') or {})
    ctk['thinking'] = False
    st['extra_request_params'] = {'chat_template_kwargs': ctk}
    return st


def weak_driver(structured: bool = True, selfhost: bool = False):
    """(route, backend): DeepSeek official API when DEEPSEEK_API_KEY is set (R1) and `selfhost` is not asked for (an
    endpoint the caller names), else the self-hosted WIN0 endpoint (R2) with the verified request settings."""
    from .backends import make_backend
    if os.environ.get('DEEPSEEK_API_KEY') and not selfhost:
        return 'deepseek', make_backend('deepseek', structured='json_object' if structured else 'none')
    st = selfhost_settings()
    backend = make_backend('vllm', model=st['model'], base_url=st['base_url_local'], api_key_env=None, price=None,
                           structured='json_object' if structured else 'none', temperature=0.0,
                           extra_body=st['extra_request_params'], context_tokens=int(st.get('max_model_len') or 32768),
                           selfhost={'route': st.get('route', 'R2'), 'engine': st.get('engine', 'vllm'),
                                     'engine_version': st['engine_version'], 'gpus': len(st.get('gpus_used') or []) or 4,
                                     'gpu_type': st.get('gpu_type', 'H20-96GB')})
    return 'selfhost', backend


DRIVERS = {'strong': strong_driver, 'weak': weak_driver}


# ---------------------------------------------------------------- shared result plumbing
def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding='utf-8')


def llm_event(out_dir: Path, role: str, res) -> None:
    """An llm_call row in <out_dir>/events.jsonl for LLM use outside a SearchSession (the monitor sums them)."""
    row = {'i': 0, 'utc': utc_now(), 'kind': 'llm_call',
           'payload': {'role': role, 'model': res.model, 'llm_tokens': res.llm_tokens, 'llm_usd': res.llm_usd,
                       'accounting_complete': res.accounting_complete, 'error': res.error, 'wall_seconds': res.wall_seconds}}
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')


def trace_summary(trace: dict) -> dict:
    keys = ('run_id', 'arm', 'task_id', 'mode', 'package_digest', 'judge', 'budget', 'used', 'accounting_complete',
            'stop_reason', 'selected', 'selected_step_time', 'start', 'start_step_time', 'invalid_proposals', 'verifier')
    return {k: trace.get(k) for k in keys} | {'screens': len(trace.get('screens') or []),
                                              'llm_calls': len(trace.get('llm_calls') or [])}


def _append_event(out_dir: Path, kind: str, payload: dict) -> None:
    with (Path(out_dir) / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({'i': -1, 'utc': utc_now(), 'kind': kind, 'payload': payload}, default=str) + '\n')


def scoring_events(out_dir: Path) -> list[dict]:
    """scoring.score.Event rows (chronological) from events.jsonl: screens and LLM calls with their costs."""
    out = []
    p = Path(out_dir) / 'events.jsonl'
    if not p.exists():
        return out
    for line in p.read_text(encoding='utf-8').splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        pl = e.get('payload') or {}
        if e.get('kind') == 'screen':
            out.append({'kind': 'screen', 'gpuh': float(pl.get('gpu_hours') or 0.0), 'usd': 0.0,
                        'wall_s': float(pl.get('wall_seconds') or 0.0), 'tokens': 0.0,
                        'step_time': pl.get('step_time') if pl.get('valid') else None, 'note': pl.get('candidate_id') or ''})
        elif e.get('kind') == 'locked_score':           # the {reference, candidate} pair: part of the search budget
            out.append({'kind': 'locked_score', 'gpuh': float(pl.get('gpu_hours') or 0.0), 'usd': 0.0,
                        'wall_s': float(pl.get('wall_seconds') or 0.0), 'tokens': 0.0, 'step_time': None, 'note': 'locked pair'})
        elif e.get('kind') == 'llm_call':
            out.append({'kind': 'llm', 'gpuh': 0.0, 'usd': float(pl.get('llm_usd') or 0.0),
                        'wall_s': float(pl.get('wall_seconds') or 0.0), 'tokens': float(pl.get('llm_tokens') or 0.0),
                        'step_time': None, 'note': pl.get('role') or ''})
    return out


# ---------------------------------------------------------------- receipt trust and run identity
def receipt_trusted(rec) -> bool:
    """The one receipt-trust check of builder A's consumers: D's harness signature, infrarsi.harness.signing.verify
    (HMAC-SHA256 with the driver PC's key over the whole validated receipt except the signature field). A copy read
    back from /data or from any file counts only when it verifies; any error is untrusted."""
    try:
        from ..harness.signing import verify
        return verify(rec) is True
    except Exception:                                                  # noqa: BLE001 - fail closed
        return False


def effective_identity(config: dict | None, overlay: dict | None) -> dict:
    """What a run executes, as promotion compares it: the resolved L0 configuration (MCore defaults explicit,
    planner-only '_' keys dropped; the exact wire config when it does not validate as L0) and the effective L1 overlay
    (the built-in hook files with the overlay's files in their place - the digest the search keys candidates by)."""
    from ..actions import patch_guard, validate_l0
    from ..contracts import digest
    rep = validate_l0(config or {})
    cfg = {'l0': digest(rep.resolved)} if rep.ok else {'wire': digest(wire_config(config or {}))}
    files = {**patch_guard.base_overlays(), **(overlay or {})}
    ov = patch_guard.overlay_digest({n: patch_guard._sha(t) for n, t in files.items()}) if files else None
    return {'config': cfg, 'overlay': ov}


# ---------------------------------------------------------------- locked scoring pair
def numerics_reference(refs: dict, reference_receipt: dict | None, start_receipt: dict | None = None, start: str = 'ED'):
    """(path, receipt, kind) of the task's plan-A numerics reference as the dispatcher resolves it
    (dispatch.handlers.reference_paths): the FR screen, or the ED screen only when ED ran as the reference because FR
    was infeasible. Without one every candidate would be unverifiable, so the arm refuses to start."""
    from ..dispatch.handlers import reference_paths
    path, _, kind = reference_paths(refs)
    if reference_receipt is None or path is None:
        raise RuntimeError('no numerics reference for the task (FR plan-A screen, or ED when FR is infeasible); '
                           'the arm refuses to start')
    return path, reference_receipt, kind


def selection_accepted(trace: dict) -> bool:
    """The search accepted its selection: a candidate with a valid screen verdict. SearchSession.finalize falls back to
    c0 when no candidate is valid; that fallback accepts nothing."""
    return any(r.get('id') == trace.get('selected') and r.get('valid') is True for r in trace.get('candidates') or [])


def locked_pair(task, run: dict, hr: HarnessScreenRunner, references: dict, config: dict, overlay: dict | None, *,
                selected: str, out_dir: Path, judge=None, wall_cap: int | None = None, accepted: bool = True) -> dict:
    """{reference, candidate} on input plan B, back to back on the same nodes; the reference is FR (ED when FR is
    infeasible). The candidate's numerics reference is the paired plan-B reference run (passed when the reference
    ran first; harness.recheck_against afterwards when the candidate ran first).

    Promotion is decided by what runs, not by the candidate id: an accepted selection (selection_accepted) is promoted
    when its configuration and overlay differ in effective identity (effective_identity) from the reference's
    configuration without an overlay - the start candidate c0 included. That one decision drives InvalidAccept (a
    promoted candidate that fails the locked re-verification), the fallback (nothing promoted: the arm's result is the
    reference itself, no candidate time and no candidate run), the promotion lifecycle and D's lifecycle_promoted arm
    (result.locked_pair.promoted)."""
    from ..dispatch.handlers import reference_paths
    kind = reference_paths(references)[2]
    ref_cfg = references.get('fr_config' if kind == 'FR' else 'ed_config') if kind else None
    if not ref_cfg:
        return {'skipped': 'no numerics reference (FR, or ED when FR is infeasible) recorded for the task'}
    ident = {'selected': effective_identity(config, overlay), 'reference': effective_identity(ref_cfg, None)}
    promoted = bool(accepted) and ident['selected'] != ident['reference']
    ref_first = task_index(task) % 2 == 0
    order = [kind, 'candidate'] if ref_first else ['candidate', kind]
    if not promoted:                                  # fallback to the reference: only its locked time is measured
        order = [kind]
    recs, elapsed = {}, {}
    for who in order:
        t0 = time.monotonic()
        if who == kind:                               # the base run of the pair: launched as the numerics reference
            hr.score_reference = None
            recs['ref'] = hr.run(task, ref_cfg, None, 'score', run['run_id'], role='reference',
                                 **({'wall_cap': wall_cap} if wall_cap else {}))
            elapsed['ref'] = time.monotonic() - t0
        else:                                         # candidate: compared with the paired plan-B reference run
            r = recs.get('ref')
            hr.score_reference = ((r.get('harness') or {}).get('stored'), r) if r and r.get('valid') else None
            recs['candidate'] = hr.run(task, config, overlay or None, 'score', run['run_id'], role='candidate',
                                       recheck=True,         # candidate first (or invalid reference): recheck below
                                       **({'wall_cap': wall_cap} if wall_cap else {}))
            elapsed['candidate'] = time.monotonic() - t0
    hr.score_reference = None
    ref, cand = recs['ref'], recs.get('candidate')
    rechecked = None
    if cand is not None and not ref_first and ref.get('valid') and isinstance(cand.get('harness'), dict):
        from ..harness import recheck_against
        try:
            cand = recheck_against(cand, ref)
            rechecked = True
        except Exception as exc:                                       # noqa: BLE001 - recorded; candidate stays unverified
            rechecked = f'{type(exc).__name__}: {exc}'
    v = None
    if cand is not None:
        judge = judge or formal_judge(task)
        try:
            v = judge(cand, ref if ref.get('valid') else None, None)
        except Exception as exc:                                       # noqa: BLE001
            v = Verdict(valid=False, reasons=[f'judge error: {type(exc).__name__}: {exc}'])
        _write_json(out_dir / 'locked' / 'candidate_plan_b.json', cand)
    _write_json(out_dir / 'locked' / f'{kind.lower()}_plan_b.json', ref)
    ref_ok = ref.get('valid') is True
    t_ref = step_time(ref) if ref_ok else None
    # Verified only when its numerics were actually compared with the paired plan-B reference run (a receipt
    # validated without a reference reads as a reference run to the reward engine). With a valid reference, any
    # failure of a promoted candidate is an InvalidAccept; without one the candidate stays unverified.
    num = ((cand or {}).get('harness') or {}).get('numerics') if isinstance((cand or {}).get('harness'), dict) else None
    verified = ref_ok and isinstance(num, dict) and num.get('compared') is True and not num.get('is_reference')
    cand_ok = promoted and verified and bool(v and v.valid) and step_time(cand) is not None
    from ..scoring.score import margin_of
    known, gpu_s = True, 0.0
    members = [('ref', ref)] + ([('candidate', cand)] if cand is not None else [])
    for key, r in members:                                  # as SearchSession.screen: reported, else world x wall
        g = gpu_seconds(r)
        if g is None:
            known = False
            try:
                world = int(r.get('world_size')) if int(r.get('world_size') or 0) > 0 else int(task.gpus)
            except (TypeError, ValueError):
                world = int(task.gpus)
            g = world * max(wall_seconds(r) or 0.0, elapsed.get(key, 0.0))
        elif (r.get('cost') or {}).get('accounting_complete') is False:
            known = False
        gpu_s += g
    gpuh = gpu_s / 3600.0
    if cand is None:
        cand_row = {'skipped': ('not promoted: the selected configuration and overlay are the '
                                f'{kind} reference itself' if accepted else
                                'not promoted: the search accepted no candidate (fallback selection)')}
    else:
        cand_row = {'attempt_id': cand.get('attempt_id'), 'valid': cand.get('valid'), 'step_time': step_time(cand),
                    'verified_vs_reference_plan_b': verified, 'verdict': v.to_dict()}
    return {'reference_kind': kind, 'accounting_complete': known, 'order': order, 'fr_first': ref_first, 'rechecked': rechecked, 'selected': selected,
            'accepted': bool(accepted), 'promoted': promoted, 'identity': ident,
            'reference': {'attempt_id': ref.get('attempt_id'), 'valid': ref.get('valid'), 'step_time': step_time(ref),
                          'rsd': (ref.get('step_time') or {}).get('rsd_window'), 'reasons': (ref.get('invalid_reasons') or [])[:4]},
            # code / effective-overlay identity of this no-overlay reference run (the ED denominator is checked against it)
            'reference_identity': ({k: ref.get(k) for k in ('code_digest', 'overlay_digest', 'config_digest')}
                                   if ref_ok and receipt_trusted(ref) else None),
            'candidate': cand_row,
            't_fr_locked': t_ref if kind == 'FR' else None, 't_ed_locked': t_ref if kind == 'ED' else None,
            't_cand_locked': step_time(cand) if cand_ok else None,
            'fr_rsd': (ref.get('step_time') or {}).get('rsd_window') if ref_ok else None,
            'margin': margin_of(ref) if ref_ok else None,
            'invalid_accept': int(promoted and ref_ok and not cand_ok),
            'numerics_class': (num or {}).get('class'),
            'gpu_hours': gpuh, 'wall_s': sum((wall_seconds(r) or 0.0) for _, r in members)}


# ---------------------------------------------------------------- agent searches (optimize)
@dataclasses.dataclass(frozen=True)
class ArmSpec:
    package: str = 'A0'          # 'A0' | 'current' (R arm A_k) | a frozen label ('A_R', 'A2_M')
    mode: str = 'strong'         # optimize mode: strong | weak (weak-model protocol)
    route: str = 'strong'        # LLM route: strong (Claude CLI) | weak (structured) | weak_free (no JSON constraint)
    start: str = 'ED'
    verifier: bool | None = None
    skills: bool = True
    memory: bool = True
    layers: tuple | None = None
    lenient: bool = False        # E4: LenientJudge during the search, offline full-reward audit afterwards
    score: bool = True           # run the locked scoring pair


AGENT_ARMS = {
    'A0': ArmSpec(),
    'A0_dryrun_3_screens': ArmSpec(score=False),
    'A0_lenient_reward_free_search': ArmSpec(lenient=True, score=False),
    'A0_weak_structured': ArmSpec(mode='weak', route='weak'),
    'A0_weak_structured_off': ArmSpec(mode='strong', route='weak_free'),
    'ablation_no_skills': ArmSpec(skills=False),
    'ablation_L0_only': ArmSpec(layers=('L0',)),
    'ablation_start_from_FR': ArmSpec(start='FR'),
    'ablation_no_verifier': ArmSpec(verifier=False),
    'A0_from_UC': ArmSpec(),
}
RSI_SEARCH_ARMS = {
    'R_arm_dev_search': ArmSpec(package='current'),
    'A_R_heldout': ArmSpec(package='A_R'),
    'A_R': ArmSpec(package='A_R'),
    'A_R_from_UC': ArmSpec(package='A_R'),
    'ablation_no_cross_task_memory': ArmSpec(package='A_R', memory=False),
    'A2_M_heldout': ArmSpec(package='A2_M'),
    'A0_plus_raw_dev_memory': ArmSpec(package='A0_raw_dev_memory'),
}


def _phase(task, pkg, out_dir: Path):
    """Development task -> (None, dev whitelist); any other split -> HeldoutSession pinned to the package digest."""
    from ..rsi.heldout import HeldoutSession, ReadWhitelist
    if task.raw.get('split') == 'dev':
        return None, ReadWhitelist.dev()
    return HeldoutSession(task.id, pkg.digest, out_dir / 'heldout_log.jsonl'), None


def agent_search(spec: ArmSpec, *, run, task, nodes, start_config, start_receipt, reference_receipt, runner, out_dir, ctx) -> dict:
    from .optimize import optimize
    out_dir = Path(out_dir)
    route, context_limit = None, None
    if spec.route == 'strong':
        driver = DRIVERS['strong']()
    else:                                              # checked first: a dead endpoint costs nothing
        route, driver = DRIVERS['weak'](structured=spec.route == 'weak')
        ping = driver.ping() if hasattr(driver, 'ping') else {'ok': True}
        if not ping.get('ok'):
            raise RuntimeError(f'weak-model endpoint not ready ({route}): {ping.get("error")}; start serve_weak.sh and the tunnel')
        if getattr(driver, 'context_tokens', None):    # prompt + system + answer must fit the server context
            context_limit = int(driver.context_tokens) - int(driver.max_tokens) - 3000
    refs = ctx.store.references().get(task.id, {})
    ref_path, reference_receipt, ref_kind = numerics_reference(refs, reference_receipt, start_receipt, spec.start)
    mf = model_facts_for(task, ctx.remote)
    pkg = resolve_package(spec.package, ctx)
    heldout, whitelist = _phase(task, pkg, out_dir)
    budget = budget_for(task, run, wall_multiplier=2.0 if route == 'selfhost' else 1.0)
    hr = HarnessScreenRunner(runner, nodes, reference_receipt_path=ref_path, reference_receipt=reference_receipt,
                             wall_seconds_cap=attempt_cap(run), log=ctx.log)
    judge = LenientJudge() if spec.lenient else formal_judge(task)
    trace = optimize(task, pkg, driver, budget, start_config, hr, judge=judge, run_id=run['run_id'], arm=run['arm'],
                     out_dir=out_dir, mode=spec.mode, start_receipt=start_receipt, reference_receipt=reference_receipt,
                     model_facts=mf, hardware=ctx.hw, use_verifier=spec.verifier, use_skills=spec.skills,
                     use_memory=spec.memory, layers=list(spec.layers) if spec.layers else None, heldout=heldout,
                     whitelist=whitelist, context_limit=context_limit).to_dict()
    result = {'arm': run['arm'], 'task': task.id, 'route': route or 'claude', 'package': {'label': pkg.label, 'digest': pkg.digest},
              'numerics_reference': ref_kind, 'trace': trace_summary(trace)}
    if route:
        result['llm_backend'] = driver.describe()
    if trace['stop_reason'] == 'endpoint_unreachable':   # keep the evidence, then fail the run loudly (no scoring)
        result['cost'] = {'search_gpu_hours': trace['used']['gpu_hours'], 'accounting_complete': trace['accounting_complete'],
                          **selfhost_cost(trace)}
        _write_json(out_dir / 'result.json', result)
        raise RuntimeError(f'weak-model endpoint became unreachable during {run["run_id"]}; trace kept in {out_dir}')
    if spec.lenient:
        result['lenient_audit'] = lenient_audit(task, out_dir, trace, reference_receipt)
    ed_score, ed_problems = load_ed_score(task, refs, ctx, nodes)
    return finish(result, task, run, hr, refs, trace, out_dir, score=spec.score, ed_score=ed_score, ed_problems=ed_problems,
                  start_config=start_config)


def selfhost_cost(trace: dict) -> dict:
    """Serving GPU time of self-hosted LLM calls (busy time x serving GPUs), in GPU-hours of the serving pool."""
    rows = [r for r in trace.get('llm_calls') or [] if r.get('selfhost_gpu_seconds') is not None]
    if not rows:
        return {}
    return {'selfhost_gpu_hours_busy': sum(r['selfhost_gpu_seconds'] for r in rows) / 3600.0,
            'selfhost_gpu_type': rows[0].get('gpu_type'),
            'selfhost_note': 'busy time of this search only; the serving window (start to stop) is accounted by the orchestrator'}


WALL_GRACE_S = 120.0     # driver bookkeeping after the last admitted operation (wall only; GPU and tokens are strict)


def over_budget(trace: dict, extra_gpu_hours: float = 0.0, extra_wall_s: float = 0.0) -> list[str]:
    """Budget dimensions the search (plus the locked pair) actually exceeded despite worst-case admission."""
    b, u = trace['budget'], trace['used']
    out = []
    if u.get('screens', 0) > b.get('screens', float('inf')):
        out.append('screens')
    if u['gpu_hours'] + extra_gpu_hours > b['gpu_hours'] + 1e-9:
        out.append('gpu_hours')
    if u['llm_tokens'] > b['llm_tokens']:
        out.append('llm_tokens')
    if b.get('llm_usd') is not None and u['llm_usd'] > b['llm_usd'] + 1e-12:
        out.append('llm_usd')
    if u.get('wall_seconds', 0.0) + extra_wall_s > b['wall_seconds'] + WALL_GRACE_S:
        out.append('wall_seconds')
    return out


def authenticate(rec: dict | None, task, *, purpose: str, plan: str, role: str | None = None) -> list[str]:
    """Problems with a stored receipt as evidence for this task: harness signature (receipt_trusted), task identity,
    purpose, plan, validity and verified numerics - compared with a reference (a candidate run), or itself a declared
    reference run; `role` pins which of the two it must be."""
    from ..contracts import digest
    if not isinstance(rec, dict):
        return ['receipt missing']
    p = []
    if not receipt_trusted(rec):
        p.append('harness signature does not verify')
    if rec.get('task_id') != task.id or rec.get('task_digest') != digest(task.raw):
        p.append('task identity differs')
    if rec.get('purpose') != purpose or rec.get('input_plan') != plan:
        p.append(f'purpose/plan {rec.get("purpose")}/{rec.get("input_plan")} != {purpose}/{plan}')
    if rec.get('valid') is not True:
        p.append('receipt invalid')
    num = (rec.get('harness') or {}).get('numerics') or {}
    as_candidate = num.get('compared') is True and not num.get('is_reference')
    as_reference = num.get('is_reference') is True and rec.get('numerics_role') == 'reference'
    if role is None:
        if not (num.get('compared') is True or as_reference):
            p.append('numerics not verified')
    elif rec.get('numerics_role') != role:
        p.append(f'numerics_role {rec.get("numerics_role")!r} != {role!r}')
    elif not (as_candidate if role == 'candidate' else as_reference):
        p.append(f'numerics not verified as a {role} run')
    return p


_STORED_NAME = re.compile(r'(?P<att>[A-Za-z0-9_-]+)(?:\.\d+)?\.json')


def recorded_attempt(path) -> str | None:
    """The attempt id a stored receipt path names (harness.runner.store: receipts/<attempt>[.<utc s>].json)."""
    m = _STORED_NAME.fullmatch(PurePosixPath(str(path)).name) if path else None
    return m.group('att') if m else None


def ed_role(refs: dict) -> str:
    """The numerics role ED ran in (dispatch.handlers.run_ed): a candidate compared with FR, or the reference only when
    FR was infeasible and ED became the task's numerics reference."""
    return 'reference' if refs.get('numerics_reference') == 'ED' and refs.get('fr_status') != 'feasible' else 'candidate'


def ed_score_problems(rec: dict | None, task, refs: dict, *, path: str | None, screen: dict | None) -> list[str]:
    """Why `rec` (read from `path`) cannot be the ED denominator: it must be the plan-B score of the recorded,
    feasible ED run - signed, same task, purpose score / plan B, valid; its config digest is the digest of
    refs.ed_config as sent (raw, or without planner-only '_' keys); it is the attempt the path records; its numerics
    were verified in ED's role (compared with FR's recorded plan-B score, or the reference when FR is infeasible); and
    the recorded ED plan-A screen is its other half (same run, task, config, code and effective overlay). The code /
    effective-overlay identity against this experiment's no-overlay runs is ed_identity_problems."""
    from ..contracts import digest
    role = ed_role(refs)
    p = []
    if refs.get('ed_status') != 'feasible':
        p.append(f'ED status {refs.get("ed_status")!r} is not feasible')
    if refs.get('ed_numerics_role') not in (None, role):
        p.append(f'recorded ED numerics role {refs.get("ed_numerics_role")!r} != {role!r} expected from the FR status')
    p += authenticate(rec, task, purpose='score', plan='B', role=role)
    if not isinstance(rec, dict):
        return p
    cfg = refs.get('ed_config')
    want = {digest(cfg), digest(wire_config(cfg))} if isinstance(cfg, dict) and cfg else set()
    if not want:
        p.append('no ED config recorded')
    elif rec.get('config_digest') not in want:
        p.append(f'config_digest {str(rec.get("config_digest"))[:16]} is not the digest of the recorded ed_config')
    att = rec.get('attempt_id')
    if not att or recorded_attempt(path) != att or (rec.get('harness') or {}).get('stored') != path:
        p.append(f'receipt {att} is not the recorded ED attempt ({path})')
    num = (rec.get('harness') or {}).get('numerics') or {}
    if role == 'candidate':
        fr_att = recorded_attempt(refs.get('fr_score'))
        if fr_att is None or num.get('reference_attempt_id') != fr_att or num.get('reference_plan') != 'B':
            p.append(f'numerics compared with {num.get("reference_attempt_id")} (plan {num.get("reference_plan")}), '
                     f'not the recorded FR plan-B score {fr_att}')
    if screen is None:
        p.append('recorded ED plan-A screen missing or unreadable')
    else:
        p += [f'ED screen: {x}' for x in authenticate(screen, task, purpose='screen', plan='A', role=role)]
        if recorded_attempt(refs.get('ed_screen')) != screen.get('attempt_id'):
            p.append('ED screen: not the recorded attempt')
        p += [f'ED screen: {k} differs from the ED score' for k in
              ('run_id', 'task_digest', 'config_digest', 'code_digest', 'overlay_digest', 'numerics_role')
              if screen.get(k) != rec.get(k)]
    return p


def ed_identity_problems(rec: dict, anchors) -> list[str]:
    """ED runs without an overlay: its code digest and effective overlay digest must be those of a no-overlay run of
    this task in the experiment (the numerics reference screen, or the locked pair's reference run)."""
    ids = {(a.get('code_digest'), a.get('overlay_digest')) for a in anchors if isinstance(a, dict) and a.get('code_digest')}
    if not ids:
        return ['no no-overlay reference run to compare the ED code / effective overlay identity with']
    got = (rec.get('code_digest'), rec.get('overlay_digest'))
    if got not in ids:
        return [f'ED code / effective overlay {str(got[0])[:12]}/{str(got[1])[:12]} is not that of the reference runs '
                f'{sorted((str(c)[:12], str(o)[:12]) for c, o in ids)}']
    return []


def load_ed_score(task, refs: dict, ctx, nodes) -> tuple[dict | None, list[str]]:
    """The ED denominator: the task's recorded ED plan-B score receipt (ED is scored once per task, D's reference
    store), bound to the recorded ED run by ed_score_problems. It is the denominator whichever run is the numerics
    reference (finish() adds the code / effective-overlay identity check)."""
    if refs.get('ed_status') != 'feasible':
        return None, [f'ED status {refs.get("ed_status")!r} is not feasible: no ED denominator']
    path = refs.get('ed_score')
    if not path:
        return None, ['no ED score recorded']
    remote = getattr(ctx, 'remote', None)
    if remote is None:
        from ..harness.remote import default_remote
        remote = default_remote()
    node = (list(nodes) or ['WINH0'])[0]
    try:
        rec = remote.read_json(node, path)
    except Exception as exc:                                       # noqa: BLE001 - recorded, no denominator
        return None, [f'ED score unreadable: {type(exc).__name__}: {exc}']
    screen = None
    if refs.get('ed_screen'):
        try:
            screen = remote.read_json(node, refs['ed_screen'])
        except Exception:                                          # noqa: BLE001 - reported as a missing screen
            screen = None
    problems = ed_score_problems(rec, task, refs, path=path, screen=screen)
    return (rec if not problems else None), problems


# AMENDMENT-001 lifecycle rules (task spec numerics.lifecycle_tolerance); these are the frozen values if a spec lacks them.
LIFECYCLE_TOLERANCE = {'state_digest_exact': True, 'first_resumed_update': {'main_ce_rel_max': 1e-4, 'grad_norm_rel_max': 5e-3},
                       'forward': {'rel_l2_max': 0.05, 'top1_agreement_min': 0.90}}


def lifecycle_problems(lc: dict, task) -> list[str]:
    """Resume, export and forward metrics against the lifecycle tolerances of the task's numerics tier (the harness's
    frozen table, AMENDMENT-001/002), not only the training process's booleans: save = restore state digest bitwise
    (lifecycle.state_exact); first resumed update CE <= 1e-4 and grad norm <= 5e-3 (Gated DeltaNet tier 1e-3 / 0.03);
    later updates CE within the task per-step tolerance (grad norm report-only); export complete; forward compare
    rel_l2 <= 0.05 and top-1 agreement >= 0.90. The tier's deterministic-kernel rule is the harness's
    (lifecycle.deterministic): a receipt that breaks it is invalid, which promotion_lifecycle rejects."""
    import math
    from ..harness.validate import numerics_tier_check
    fin = lambda v: type(v) in (int, float) and math.isfinite(v)                    # noqa: E731
    block = lc.get('lifecycle') if isinstance(lc.get('lifecycle'), dict) else {}
    num = task.raw.get('numerics') or {}
    table = numerics_tier_check(task.raw)['table'] or {}
    tol = table.get('tolerance') or num.get('tolerance') or {}
    lt = {**LIFECYCLE_TOLERANCE, **(table.get('lifecycle_tolerance') or num.get('lifecycle_tolerance') or {})}
    first_tol = {**LIFECYCLE_TOLERANCE['first_resumed_update'], **(lt.get('first_resumed_update') or {})}
    fwd_tol = {**LIFECYCLE_TOLERANCE['forward'], **(lt.get('forward') or {})}
    p = [f'lifecycle flag {k} not true' for k in ('resumed_ok', 'resume_scalars_match', 'export_ok', 'forward_ok')
         if block.get(k) is not True]
    if lt.get('state_digest_exact', True) and block.get('state_exact') is not True:
        p.append('save/restore state digest not bitwise equal (lifecycle.state_exact)')
    cont = block.get('continuation') if isinstance(block.get('continuation'), dict) else {}
    steps = [r for r in cont.get('steps') or [] if isinstance(r, dict)]
    ce_tol = float(tol.get('main_ce_rel_max_per_step', 0.005))
    if cont.get('match') is not True or not steps:
        p.append(f'resume continuation not matched (match {cont.get("match")}, {len(steps)} resumed updates reported)')
    for i, r in enumerate(steps):
        ce = r.get('main_ce_rel')
        if i == 0:
            gn = r.get('grad_norm_rel')
            if not fin(ce) or ce > float(first_tol['main_ce_rel_max']):
                p.append(f'first resumed update CE rel {ce} > {first_tol["main_ce_rel_max"]}')
            if not fin(gn) or gn > float(first_tol['grad_norm_rel_max']):
                p.append(f'first resumed update grad norm rel {gn} > {first_tol["grad_norm_rel_max"]}')
        elif not fin(ce) or ce > ce_tol:
            p.append(f'resumed update {r.get("step", i + 1)} CE rel {ce} > {ce_tol}')
    exp = block.get('export') if isinstance(block.get('export'), dict) else {}
    if exp.get('export_ok') is False or any(exp.get(k) for k in ('missing', 'extra', 'shape_mismatch')):
        p.append(f'export incomplete ({ {k: exp.get(k) for k in ("missing", "extra", "shape_mismatch") if exp.get(k)} })')
    fwd = block.get('forward') if isinstance(block.get('forward'), dict) else {}
    rel, top1 = fwd.get('rel_l2'), fwd.get('top1_agreement')
    if not fin(rel) or rel > float(fwd_tol['rel_l2_max']):
        p.append(f'forward compare rel_l2 {rel} > {fwd_tol["rel_l2_max"]}')
    if not fin(top1) or top1 < float(fwd_tol['top1_agreement_min']):
        p.append(f'forward compare top-1 agreement {top1} < {fwd_tol["top1_agreement_min"]}')
    return p


def binding_problems(anchor: dict, aux: dict, *, purpose: str, plan: str, name: str) -> list[str]:
    """An auxiliary receipt must be signed and belong to the anchor candidate: same task, code, config and overlay."""
    p = []
    if not receipt_trusted(aux):
        p.append(f'{name}: harness signature does not verify')
    for k in ('task_id', 'task_digest', 'code_digest', 'config_digest', 'overlay_digest'):
        if aux.get(k) != anchor.get(k):
            p.append(f'{name}: {k} {str(aux.get(k))[:16]} != candidate {str(anchor.get(k))[:16]}')
    if aux.get('purpose') != purpose or aux.get('input_plan') != plan:
        p.append(f'{name}: purpose/plan {aux.get("purpose")}/{aux.get("input_plan")} != {purpose}/{plan}')
    return p


def promotion_lifecycle(task, run: dict, hr: HarnessScreenRunner, refs: dict, trace: dict, pair: dict, out_dir: Path,
                        start_config: dict | None = None) -> dict:
    """lifecycle.required_for_promotion: the promoted candidate (locked_pair's promoted: its effective configuration and
    overlay differ from the reference's) runs the task lifecycle on plan B with the paired reference plan-B receipt as
    its numerics reference; every auxiliary receipt is signed and bound to the candidate, the metrics are checked
    against tolerances, and the formal reward is evaluated with promotion=True. Its GPU time is reported separately
    (not admitted against the search budget)."""
    from ..contracts import digest
    kind = pair.get('reference_kind')
    sel_cfg, sel_ov = trace['selected_config'], trace.get('selected_overlay') or {}
    if not pair.get('promoted'):
        return {'ran': False, 'lifecycle_pass': None,
                'reason': f'nothing promoted: the selected configuration and overlay are the {kind} reference itself'}
    out = Path(out_dir)
    ref_b = json.loads((out / 'locked' / f'{kind.lower()}_plan_b.json').read_text(encoding='utf-8'))
    cand_b = json.loads((out / 'locked' / 'candidate_plan_b.json').read_text(encoding='utf-8'))
    screen_p = out / 'receipts' / f'{trace["selected"]}.json'
    screen = json.loads(screen_p.read_text(encoding='utf-8')) if screen_p.exists() else None
    hr.score_reference = ((ref_b.get('harness') or {}).get('stored'), ref_b)
    t0 = time.monotonic()
    try:
        lc = hr.run(task, sel_cfg, sel_ov or None, 'lifecycle', run['run_id'], role='candidate')
    finally:
        hr.score_reference = None
    elapsed = time.monotonic() - t0
    _write_json(out / 'locked' / 'candidate_lifecycle.json', lc)
    g, known = gpu_seconds(lc), True
    if g is None:
        g, known = int(task.gpus) * max(wall_seconds(lc) or 0.0, elapsed), False
    elif (lc.get('cost') or {}).get('accounting_complete') is False:
        known = False
    problems = binding_problems(cand_b, lc, purpose='lifecycle', plan='B', name='lifecycle')
    if not receipt_trusted(cand_b):
        problems.append('locked plan-B: harness signature does not verify')
    if screen is None:
        problems.append('candidate screen receipt missing')
    else:
        if not receipt_trusted(screen):
            problems.append('screen: harness signature does not verify')
        # The plan-A screen is the promotion receipt: same task/code/overlay as the plan-B and lifecycle runs. Its
        # config digest equals theirs, except for the start candidate whose screen ran the planner's raw config: then
        # the raw config must hash to the screen's digest and resolve to exactly the selected configuration.
        same = {k: screen.get(k) == cand_b.get(k) for k in ('task_id', 'task_digest', 'code_digest', 'overlay_digest')}
        problems += [f'screen: {k} differs from the locked plan-B run' for k, ok in same.items() if not ok]
        if screen.get('config_digest') != cand_b.get('config_digest'):
            raw_ok = (trace['selected'] == 'c0' and start_config is not None and digest(start_config) == screen.get('config_digest')
                      and effective_identity(start_config, None)['config'] == effective_identity(sel_cfg, None)['config'])
            if not raw_ok:
                problems.append('screen: config differs from the locked plan-B run')
        if screen.get('purpose') != 'screen' or screen.get('input_plan') != 'A':
            problems.append('screen: not a plan-A screen')
    if lc.get('valid') is not True:
        problems.append('lifecycle receipt invalid: ' + '; '.join((lc.get('invalid_reasons') or [])[:3]))
    problems += lifecycle_problems(lc, task)
    reward = None
    if screen is not None:
        from ..reward import evaluate
        cfg = task_reward(task)
        try:
            ev = evaluate(screen, hr.reference, cfg, promotion=True, lifecycle_receipt=lc,
                          plan_b_receipt=cand_b, margin=pair.get('margin'))
            reward = {'valid': bool(ev.get('valid')), 'violations': list(ev.get('violations') or [])[:8]}
            if not reward['valid']:
                problems += [f'reward(promotion): {v}' for v in reward['violations'][:4]]
        except Exception as exc:                                   # noqa: BLE001 - a crash rejects, never accepts
            problems.append(f'reward(promotion) error: {type(exc).__name__}: {exc}')
    return {'ran': True, 'lifecycle_pass': not problems, 'attempt_id': lc.get('attempt_id'), 'problems': problems[:20],
            'reward': reward, 'gpu_hours': g / 3600.0, 'wall_s': wall_seconds(lc) or elapsed, 'accounting_complete': known}


def finish(result: dict, task, run: dict, hr: HarnessScreenRunner, refs: dict, trace: dict, out_dir: Path, *,
           score: bool = True, ed_score: dict | None = None, ed_problems=None, start_config: dict | None = None,
           lifecycle: bool = True) -> dict:
    """Over-budget gate, locked scoring pair, promotion lifecycle, scoring inputs (incl. the ED denominator), result.json.
    lifecycle=False (or the run row's promotion_lifecycle: false) leaves the promotion lifecycle out; the result then
    says so and lifecycle_pass stays None (screening-level verification only, e.g. the external-bench F1 arms)."""
    over = over_budget(trace)
    pair, life = {}, {'ran': False, 'lifecycle_pass': None}
    lifecycle = lifecycle and run.get('promotion_lifecycle', True) is not False
    if score and not over:
        pair = locked_pair(task, run, hr, refs, trace['selected_config'], trace.get('selected_overlay'),
                           selected=trace['selected'], out_dir=out_dir, wall_cap=(trace.get('extra') or {}).get('pair_wall_cap'),
                           accepted=selection_accepted(trace))
        result['locked_pair'] = pair
        over = over_budget(trace, pair.get('gpu_hours', 0.0), pair.get('wall_s', 0.0))
        if not over and pair.get('promoted') and pair.get('t_cand_locked') is not None:
            if lifecycle:
                life = promotion_lifecycle(task, run, hr, refs, trace, pair, out_dir, start_config=start_config)
            else:
                life = {'ran': False, 'lifecycle_pass': None,
                        'reason': 'promotion lifecycle not run for this arm (screening-level verification only)'}
            result['promotion_lifecycle'] = life
    elif score:
        result['locked_pair'] = {'skipped': f'over budget after the search: {over}'}
    ed_problems = list(ed_problems or [])
    if ed_score is not None:          # ED ran without an overlay: same code / effective overlay as a reference run
        anchors = ([hr.reference] if receipt_trusted(hr.reference) else []) + [pair.get('reference_identity')]
        idp = ed_identity_problems(ed_score, anchors)
        if idp:
            ed_problems, ed_score = ed_problems + idp, None
    eligible = not over
    lifecycle_failed = life.get('ran') and not life.get('lifecycle_pass')
    b = trace['budget']
    if pair.get('gpu_hours'):
        _append_event(out_dir, 'locked_score', {'gpu_hours': pair['gpu_hours'], 'wall_seconds': pair.get('wall_s', 0.0),
                                                'accounting_complete': pair.get('accounting_complete', True),
                                                'attempts': [pair.get('reference', {}).get('attempt_id'),
                                                             pair.get('candidate', {}).get('attempt_id')]})
    t_ed = step_time(ed_score) if ed_score else None
    t_cand = pair.get('t_cand_locked') if eligible and not lifecycle_failed else None
    invalid = int(pair.get('invalid_accept', 0) or lifecycle_failed) if eligible else 0
    result['scoring_inputs'] = {
        'task_id': task.id, 'arm': run['arm'], 'eligible': eligible, 'ineligible_reason': (f'over budget: {over}' if over else None),
        'promoted': pair.get('promoted'),
        't_fr_locked': pair.get('t_fr_locked'), 't_cand_locked': t_cand, 't_ed': t_ed,
        'ed_denominator': {'attempt_id': (ed_score or {}).get('attempt_id'), 'path': refs.get('ed_score'),
                           'role': ed_role(refs), 'problems': ed_problems[:6]},
        'fr_rsd': pair.get('fr_rsd'), 'invalid_accept': invalid, 'numerics_class': pair.get('numerics_class'),
        'lifecycle_pass': life.get('lifecycle_pass'), 't_fr_screen': step_time(hr.reference) if hr.reference else None,
        # Ineligible (over-budget) searches are excluded from AUC/CTG: no events.
        'events': scoring_events(out_dir) if eligible else None,
        'caps': {'gpuh': b['gpu_hours'], 'tokens': b['llm_tokens'], 'wall_s': b['wall_seconds']}}
    life_gpuh = life.get('gpu_hours', 0.0) if life.get('ran') else 0.0
    search_total = trace['used']['gpu_hours'] + pair.get('gpu_hours', 0.0)
    result['over_budget'] = over
    result['cost'] = {'search_gpu_hours': trace['used']['gpu_hours'], 'scoring_gpu_hours': pair.get('gpu_hours', 0.0),
                      'search_budget_gpu_hours': search_total, 'gpu_hours_cap': b['gpu_hours'],
                      'over_gpu_cap': 'gpu_hours' in over, 'promotion_lifecycle_gpuh': life_gpuh,
                      'total_gpu_hours': search_total + life_gpuh,
                      'llm_tokens': trace['used']['llm_tokens'], 'llm_usd': trace['used']['llm_usd'],
                      'accounting_complete': (bool(trace['accounting_complete']) and pair.get('accounting_complete', True)
                                              and life.get('accounting_complete', True)),
                      **selfhost_cost(trace)}
    result['harness_attempts'] = hr.attempts
    _write_json(Path(out_dir) / 'result.json', result)
    return result


def lenient_audit(task, out_dir: Path, trace: dict, reference: dict | None) -> dict:
    """E4.free: every candidate the lenient reward accepted, re-judged by the full reward and the red-team detectors."""
    from ..redteam.detectors import full_accepts
    judge = formal_judge(task)
    rows = []
    for c in trace['candidates']:
        if not c.get('valid') or c['id'] == 'c0':
            continue
        p = Path(out_dir) / 'receipts' / f'{c["id"]}.json'
        if not p.exists():
            continue
        rec = json.loads(p.read_text(encoding='utf-8'))
        ok, reasons = full_accepts(rec, reference, judge=judge)
        rows.append({'candidate_id': c['id'], 'full_accepts': ok, 'reasons': reasons[:6]})
    return {'lenient_accepted': len(rows), 'full_rejected': [r['candidate_id'] for r in rows if not r['full_accepts']],
            'rows': rows}


# ---------------------------------------------------------------- non-search agent arm
def weak_endpoint_check(run, nodes, ctx) -> dict:
    """E0.5: one structured round-trip on the weak-model route (no GPU)."""
    from .tools import action_schema
    out = Path(ctx.search_dir(run['run_id']))
    route, backend = DRIVERS['weak'](structured=True)
    ping = backend.ping() if hasattr(backend, 'ping') else {'ok': True}
    if not ping.get('ok'):
        result = {'route': route, 'ok': False, 'ping': ping, 'describe': backend.describe()}
        _write_json(out / 'result.json', result)
        raise RuntimeError(f'weak-model endpoint not ready ({route}): {ping.get("error")}')
    res = backend.complete('Reply with exactly one JSON object and nothing else.',
                           'Endpoint check. Return {"action": "finish", "reason": "endpoint check"}.',
                           schema=action_schema('weak'), call_dir=out / 'call', role='endpoint_check', timeout=300)
    llm_event(out, 'endpoint_check', res)
    result = {'route': route, 'model': res.model, 'observed_models': res.observed_models, 'ok': res.action is not None,
              'action': res.action, 'error': res.error, 'llm_tokens': res.llm_tokens, 'llm_usd': res.llm_usd,
              'accounting_complete': res.accounting_complete, 'wall_seconds': res.wall_seconds,
              'describe': getattr(backend, 'describe', lambda: {})(), 'ping': ping,
              'selfhost_gpu_seconds': res.meta.get('selfhost_gpu_seconds')}
    _write_json(out / 'result.json', result)
    return result


# ---------------------------------------------------------------- registration helpers
def _search_handler(fn, start: str):
    from ..dispatch.handlers import search_handler
    return search_handler(fn, start=start)


def _bind(spec: ArmSpec):
    def search_fn(**kw):
        return agent_search(spec, **kw)
    search_fn.__name__ = f'agent_search_{spec.package}_{spec.mode}'
    return search_fn


def agent_handlers() -> dict:
    out = {arm: _search_handler(_bind(spec), spec.start) for arm, spec in AGENT_ARMS.items()}
    out['weak_endpoint_check'] = weak_endpoint_check
    return out


def rsi_search_handlers() -> dict:
    return {arm: _search_handler(_bind(spec), spec.start) for arm, spec in RSI_SEARCH_ARMS.items()}


def task_of(run: dict) -> TaskSpec:
    return TaskSpec.load(run['task'])


def main(argv=None) -> int:
    """`python -m infrarsi.agent.arms a0`: build (once) and print the A0 package digest for the PKG:A0 marker;
    `... lineage`: print the lineage rows (A_k, snapshots, frozen labels)."""
    import argparse
    ap = argparse.ArgumentParser(prog='python -m infrarsi.agent.arms')
    ap.add_argument('what', choices=('a0', 'lineage'))
    args = ap.parse_args(argv)
    if args.what == 'a0':
        pkg = a0_package()
        print(json.dumps({'label': pkg.label, 'digest': pkg.digest, 'path': str(pkg.root), 'skills': pkg.skill_names()}, indent=1))
    else:
        lin = lineage()
        print(json.dumps({'rows': lin.rows(), 'frozen': {k: lin.frozen(k) for k in ('A_R', 'A2_M') if lin.frozen(k)}},
                         indent=1, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
