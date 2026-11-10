"""ENV-A post-arm steps of AMENDMENT_EXT_004 (Fable 5.1): per-node S0 references (`node_ref`) and the single-node
magnitude block on WINH1 (`run_block`). Driver-side only; new code -- nothing the running arms ship or import is changed.

node_ref (one per node, after the arm's locked re-score, or after a priming read pass): the unchanged start (config {})
at purpose `score` (clean slate), `run_g4=False`, `candidate_id='node_ref'`, through the A-RND runner settings (uid pair
64103 / 64113, `enforce_g7=False`) on that node.

Block B (WINH1 GPUs 0-3, after the last arm's locked re-score): `S0 -> sel(A-NAT) -> sel(A-ISW) -> sel(A-RND)`, each at
purpose `score`, `run_g4=False`, `candidate_id='block'`, through the arm's own runner / run directory / uid pair with
`enforce_g7=False` (an attempt lease per item, the pair asserted unused first). S0 is the unchanged start through the
A-RND runner. Each candidate item is the arm's locked item, bound by identity: A-NAT restores the same snapshot with the
same `expected_digest` (G6), A-ISW / A-RND re-render the same selected config / overlay and must reproduce the locked
receipt's candidate digest.

Void (c): judge-invalid (score.py, a timeout, a node error), `first_cold` (data_prep > 13.4 s), or -- for the block S0
-- a time outside the WINH1 warm band [209.2, 222.2] s; re-run once at the end of the block in the original order. A
guard rejection (G1 / G2 / G3 / G6 / G8 / G9, missing outputs, a digest that is not the locked item's) is void without
re-run (`block_guard_rejection`). S0 void twice -> t_S0,B := 216.1; a node_ref void twice -> t_S0,n := 216.1 (node
factor not measured); a candidate void twice or not reached by the hard stop -> t_B := t_locked x t_S0,B /
t_S0,node(arm), labelled `(fallback)`. A re-run only ever replaces a void value. Containment or lease-release failures
stop the block (fail closed: the claims stay, nothing further starts).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
from pathlib import Path

from ..common.env_runner import candidate_digest as runner_candidate_digest
from ..common.remote_exec import assert_uid_unused, ContainmentError
from . import knobs as knobmod
from . import runner as rmod
from . import task as taskmod
from .guards import g1_reference
from .runner import RebenchLlmfRunner

WARM_ANCHOR_S = 216.1                 # WINH1 warm S0 (R1c); the fallback reference
BAND = (209.2, 222.2)                 # WINH1 warm band: 215.7 s +- 3 %
COLD_DATA_PREP_S = 13.4               # 1.5 x the warm 8.9 s data_prep
MARGIN = 0.05
BLOCK_ORDER = ('S0', 'A-NAT', 'A-ISW', 'A-RND')
REF_ARM = 'A-RND'                     # the S0 / node_ref runner settings (the non-LLM control's uid pair)
NO_RERUN_GUARDS = ('G1', 'G2', 'G3', 'G6', 'G8', 'G9', 'outputs_present', 'output_integrity', 'identity')
DEFAULT_REF = '/data/infrarsi/ext/runs/rebench_llmf/r1_20261021c/home_agent/trained_model_reference'
PRIME_DIRS = (taskmod.NODE_ENV, taskmod.NODE_FOUNDRY, taskmod.NODE_MODEL, taskmod.NODE_TASK + '/assets')


class BlockError(RuntimeError):
    """The block cannot continue safely (a lease or a uid process was retained, a node scorer mismatch)."""


def _utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---------------------------------------------------------------- node jobs without touching the shipped scorer
def _local_node_hashes() -> dict:
    return {f'{rmod.REMOTE_TOOLS}/{n}': hashlib.sha256((rmod._NODE_DIR / n).read_text(encoding='utf-8').encode('utf-8')).hexdigest()
            for n in rmod.NODE_FILES}


def checked_node_runner(ops, job: dict) -> dict:
    """winh_scorer's run step, but the node scorer is only shipped where it is missing: if the copy on the node differs
    from this checkout, refuse (the running arms execute that copy; it must not change under them)."""
    want = _local_node_hashes()
    have = ops.hash_files(sorted(want))
    differ = [p for p, h in want.items() if have.get(p) not in (None, h)]
    if differ:
        raise BlockError(f'node scorer differs from this checkout: {differ}; refusing to overwrite files running arms use')
    missing = [p for p, h in want.items() if have.get(p) is None]
    if missing:
        ops.mkdirs([rmod.REMOTE_TOOLS])
        for p in missing:
            ops.write_files({p: (rmod._NODE_DIR / p.rsplit('/', 1)[1]).read_text(encoding='utf-8')}, mode=0o644)
    return rmod.launch_scorer(ops, job)                 # launched once per attempt, as winh_scorer


def prime(ops, dirs=PRIME_DIRS, timeout: int = 1800) -> dict:
    """Cache-priming read pass (`cat > /dev/null` of every file under `dirs`); not a measurement."""
    q = ' '.join(f"'{d}'" for d in dirs)
    res = ops.sh(f'set -o pipefail; t0=$(date +%s); find {q} -type f -print0 | xargs -0 -r cat > /dev/null; '
                 f'echo primed_s=$(( $(date +%s) - t0 ))', timeout=timeout)
    return {'dirs': list(dirs), 'code': res.get('code'), 'out': (res.get('stdout') or '')[-200:]}


# ---------------------------------------------------------------- one scored item
def _expected_digest(rec: dict, bind_digest: str, config=None, overlay=None) -> str:
    # No environment result: the runner identifies the attempted inputs, not a scored candidate.
    failed = (rec.get('runner_error') and rec.get('judge') is None
              and rec.get('valid') is False and rec.get('judge_valid') is False)
    return runner_candidate_digest(config, overlay) if failed else bind_digest


def item_record(rec: dict, *, name: str, band: tuple | None = None, bind_digest: str | None = None,
                config: dict | None = None, overlay: dict | None = None) -> dict:
    """The block / node_ref view of one receipt: time, phases, validity and the void classification of (c)."""
    judge = rec.get('judge') if isinstance(rec.get('judge'), dict) else {}
    msg = judge.get('message') if isinstance(judge.get('message'), dict) else {}
    t = msg.get('execution_time')
    t = float(t) if type(t) in (int, float) and math.isfinite(t) and t > 0 else None
    phases = rec.get('phases') or {}
    dp = phases.get('data_prep')
    rejections = [g['id'] for g in rec.get('guard_rejections') or []]
    runner_error = rec.get('runner_error')
    out = {'name': name, 'attempt_id': rec.get('attempt_id'), 'node': rec.get('node'), 'gpus': rec.get('gpus'),
           'uid': rec.get('uid'), 'run_id': rec.get('run_id'), 'candidate_digest': rec.get('candidate_digest'),
           't': t, 'phases': {k: phases.get(k) for k in ('data_prep', 'train', 'convert', 'finetune_total', 'total')},
           'norm_diff': msg.get('norm_diff'), 'judge_valid': bool(rec.get('judge_valid')), 'valid': bool(rec.get('valid')),
           'guard_rejections': rejections, 'gpu_seconds': (rec.get('cost') or {}).get('gpu_seconds'),
           'first_cold': bool(type(dp) in (int, float) and dp > COLD_DATA_PREP_S), 'utc_end': rec.get('utc_end')}
    expected_digest = _expected_digest(rec, bind_digest, config, overlay) if bind_digest is not None else None
    if expected_digest is not None and rec.get('candidate_digest') != expected_digest:
        rejections.append('identity')
        out['identity'] = f'candidate digest {str(rec.get("candidate_digest"))[:16]} != locked item {bind_digest[:16]}'
    reasons, rerun = [], True
    if 'containment' in rejections or (rec.get('lease_release') or {}).get('kept'):
        out['fatal'] = 'containment failed or a lease path was retained'
    hard = [g for g in rejections if g in NO_RERUN_GUARDS]
    if hard:
        reasons.append('block_guard_rejection: ' + ','.join(hard))
        rerun = False
    if runner_error:
        reasons.append(f'runner_error: {runner_error.get("kind")}: {str(runner_error.get("message"))[:200]}')
    elif not out['judge_valid']:
        why = [k for k in ('norm_diff_too_high', 'finetune_failed', 'model_not_found', 'model_mismatch') if msg.get(k)]
        reasons.append('judge_invalid: ' + (','.join(why) or str(msg.get('error'))[:200]))
    if 'exit_ok' in rejections:
        reasons.append('judge_invalid: scoring exit or timeout')
    if out['first_cold']:
        reasons.append(f'first_cold: data_prep {dp:.1f} s > {COLD_DATA_PREP_S}')
    if band is not None:
        out['in_band'] = t is not None and band[0] <= t <= band[1]
        if t is not None and not out['in_band']:
            reasons.append(f'S0 outside the node warm band {band}: {t:.2f} s')
    out['void'] = bool(reasons)
    out['void_reasons'] = reasons
    out['rerun_allowed'] = bool(reasons) and rerun and 'fatal' not in out
    return out


def _runner(ops, arm: str, *, gpus, out_dir, reference_src, node_runner, clock=None) -> RebenchLlmfRunner:
    task = taskmod.build_env_a_task(arm=arm, action_space='knobs+patch' if arm in ('A-ISW', 'A-IS2') else 'knobs')
    kw = {'clock': clock} if clock is not None else {}
    return RebenchLlmfRunner(ops, task=task, gpus=gpus, enforce_g7=False, reference_src=reference_src, lease_mode='own',
                             out_dir=out_dir, node_runner=node_runner, **kw)


def _score(ops, runner: RebenchLlmfRunner, item: dict) -> dict:
    """One item: the uid pair must be unused, then one attempt (own lease) at purpose score without G4."""
    for u in runner.domain_uids():
        assert_uid_unused(ops, u)
    kw = {'run_g4': False, 'candidate_id': item['candidate_id']}
    if item.get('restore_snapshot'):
        kw.update(restore_snapshot=item['restore_snapshot'], expected_digest=item['expected_digest'])
    return runner.run(runner.task, item.get('config') or {}, item.get('overlay') or None, 'score', item['run_id'], **kw)


# ---------------------------------------------------------------- node_ref
def node_ref(ops, *, run_id: str, gpus=(0, 1, 2, 3), out_dir, reference_src: str = DEFAULT_REF, do_prime: bool = False,
             max_attempts: int = 2, node_runner=None, clock=None) -> dict:
    """One S0 scoring on this node (re-run once if void, as ruled); t_S0,n := 216.1 when void twice."""
    out_dir = Path(out_dir)
    runner = _runner(ops, REF_ARM, gpus=gpus, out_dir=out_dir, reference_src=reference_src,
                     node_runner=node_runner or checked_node_runner, clock=clock)
    rec = {'kind': 'node_ref', 'node': ops.node, 'run_id': run_id, 'gpus': list(gpus), 'runner_arm': REF_ARM,
           'uids': runner.domain_uids(), 'utc_start': _utc(), 'attempts': []}
    rec['prepare'] = {k: v for k, v in runner.prepare(run_id).items() if k in ('reference', 'base', 'ref_vs_base', 'uids_unused_before')}
    rejected = g1_reference(rec['prepare']).rejections
    if rejected:
        rec.update(fatal='S0 reference preparation failed: ' + '; '.join(g['detail'] for g in rejected),
                   t_S0_node=None, measured=False, gpu_seconds=0, utc_end=_utc())
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f'node_ref_{ops.node}.json').write_text(json.dumps(rec, indent=1, default=str), encoding='utf-8')
        return rec
    if do_prime:
        rec['prime'] = prime(ops)
    for _ in range(max(1, int(max_attempts))):
        r = _score(ops, runner, {'run_id': run_id, 'candidate_id': 'node_ref', 'config': {}})
        item = item_record(r, name='node_ref')
        rec['attempts'].append(item)
        if item.get('fatal'):
            rec['fatal'] = item['fatal']
            break
        if not item['void']:
            break
    good = [a for a in rec['attempts'] if not a['void']]
    if good:
        rec.update(t_S0_node=good[0]['t'], measured=True)
    elif len(rec['attempts']) >= 2:
        rec.update(t_S0_node=WARM_ANCHOR_S, measured=False, note='void twice: t_S0,n := 216.1, node factor not measured')
    else:
        rec.update(t_S0_node=None, measured=False, note='void once: re-run once before falling back')
    rec['gpu_seconds'] = sum(a.get('gpu_seconds') or 0 for a in rec['attempts'])
    rec['utc_end'] = _utc()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'node_ref_{ops.node}.json').write_text(json.dumps(rec, indent=1, default=str), encoding='utf-8')
    return rec


# ---------------------------------------------------------------- selections from the arm records
def _read(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding='utf-8'))


def _start_selection(arm: str, records: Path, res: dict) -> dict | None:
    """Decision D20: the arm's selection is the unchanged start -- recorded as skipped by the finish path, or (for a run
    whose in-memory finish path predates D20) recognisable from its trace: A-NAT's selected snapshot has the start's
    tree digest, a search's selection is its start candidate or renders the starting files."""
    from .arms import selection_is_start
    if arm == 'A-NAT':
        locked = (res.get('extra') or {}).get('locked') or {}
        sel = locked.get('selected') or {}
        if locked.get('skipped') == 'selection_is_start' or (sel.get('tree_digest') and sel['tree_digest'] == taskmod.START_TREE_DIGEST):
            return {'selected': sel.get('candidate_id'), 'node': locked.get('node') or (res.get('arm_lease') or {}).get('node')}
        return None
    locked = res.get('locked') or {}
    trace = res.get('trace') or {}
    if locked.get('skipped') == 'selection_is_start' or selection_is_start(trace):
        node = locked.get('node') or (res.get('arm_lease') or {}).get('node')
        if node is None and locked.get('attempt_id') and (records / 'receipts' / f'{locked["attempt_id"]}.json').exists():
            node = _read(records / 'receipts' / f'{locked["attempt_id"]}.json').get('node')
        return {'selected': trace.get('selected'), 'node': node}
    return None


def load_selection(arm: str, records: Path) -> dict:
    """The arm's locked item (what its per-arm locked re-score ran) from its CLI records directory. D20: a selection
    that is the unchanged start comes back as `is_start` (no block item; the block's S0 stands for it)."""
    records = Path(records)
    res = _read(records / 'result.json')
    start = _start_selection(arm, records, res)
    if start is not None:
        return {'arm': arm, 'is_start': True, 'skipped': 'selection_is_start', 'run_id': res.get('run_id') or records.name,
                'selected': start['selected'], 'node': start['node'], 't_locked': None}
    if arm == 'A-NAT':
        locked = ((res.get('extra') or {}).get('locked') or {})
        sel = locked.get('selected') or {}
        lk = locked.get('locked') or {}
        if not sel.get('snapshot_path') or not sel.get('tree_digest'):
            raise BlockError(f'{arm}: no locked selection in {records}')
        return {'arm': arm, 'run_id': res.get('run_id') or records.name, 'candidate_id': 'block', 'config': {},
                'restore_snapshot': sel['snapshot_path'], 'expected_digest': sel['tree_digest'],
                'bind_digest': sel['tree_digest'], 't_locked': lk.get('execution_time'),
                'locked_attempt_id': locked.get('attempt_id'), 'selected': sel.get('candidate_id'),
                'locked_valid_receipt': lk.get('valid'), 'node': (res.get('arm_lease') or {}).get('node')}
    trace = res.get('trace') or {}
    locked = res.get('locked') or {}
    att = locked.get('attempt_id')
    receipt = _read(records / 'receipts' / f'{att}.json') if att and (records / 'receipts' / f'{att}.json').exists() else None
    if receipt is None:
        raise BlockError(f'{arm}: locked receipt {att} not found under {records}/receipts')
    config = {k: v for k, v in (trace.get('selected_config') or {}).items() if k in knobmod.KNOBS}
    jm = (receipt.get('judge') or {}).get('message') or {}
    return {'arm': arm, 'run_id': receipt.get('run_id') or records.name, 'candidate_id': 'block', 'config': config,
            'overlay': trace.get('selected_overlay') or None, 'bind_digest': receipt.get('candidate_digest'),
            't_locked': jm.get('execution_time'), 'locked_attempt_id': att, 'selected': trace.get('selected'),
            'locked_valid_receipt': receipt.get('valid'), 'node': receipt.get('node')}


def first_scoring(arm: str, records: Path) -> dict:
    """AMENDMENT_EXT_004 (b): the arm's first search scoring as measured (never corrected): t_first, phases, first_cold."""
    name = 'score01' if arm == 'A-NAT' else 'c0'
    p = Path(records) / 'receipts' / f'{name}.json'
    if not p.exists():
        return {'arm': arm, 'receipt': name, 't_first': None, 'why': 'receipt missing'}
    rc = _read(p)
    jm = (rc.get('judge') or {}).get('message') or {}
    ph = rc.get('phases') or {}
    dp = ph.get('data_prep')
    return {'arm': arm, 'receipt': name, 'attempt_id': rc.get('attempt_id'), 'node': rc.get('node'),
            't_first': jm.get('execution_time'), 'phases': {k: ph.get(k) for k in ('data_prep', 'train', 'convert')},
            'first_cold': bool(type(dp) in (int, float) and dp > COLD_DATA_PREP_S), 'valid': rc.get('valid')}


def locked_validity(arm: str, records: Path) -> dict:
    """§0.3 (1)-(2) per AMENDMENT_EXT_004 (a): locked_valid := judge_valid and no rejection from G1/G2/G3/G4/G8/G9 (G7
    reported, not applied)."""
    records = Path(records)
    res = _read(records / 'result.json')
    if _start_selection(arm, records, res) is not None:            # D20: no locked re-score, InvalidAccept 0
        return {'arm': arm, 'locked_valid': True, 'skipped': 'selection_is_start', 'invalid_accept': 0,
                'locked_value': 'node S0 reference'}
    att = ((res.get('extra') or {}).get('locked') or {}).get('attempt_id') if arm == 'A-NAT' else (res.get('locked') or {}).get('attempt_id')
    p = records / 'receipts' / f'{att}.json'
    if not att or not p.exists():
        return {'arm': arm, 'locked_valid': None, 'why': 'locked receipt missing'}
    rc = _read(p)
    rej = [g['id'] for g in rc.get('guard_rejections') or []]
    hard = [g for g in rej if g in ('G1', 'G2', 'G3', 'G4', 'G8', 'G9')]
    return {'arm': arm, 'locked_valid': bool(rc.get('judge_valid')) and not hard, 'judge_valid': bool(rc.get('judge_valid')),
            'rejections': rej, 'g7_reported_only': 'G7' in rej, 'attempt_id': att, 'node': rc.get('node'),
            'first_cold': bool(((rc.get('phases') or {}).get('data_prep') or 0) > COLD_DATA_PREP_S)}


# ---------------------------------------------------------------- the block
def _deadline_ok(now_utc: datetime.datetime, deadline: datetime.datetime | None, need_s: float) -> bool:
    return deadline is None or now_utc + datetime.timedelta(seconds=need_s) <= deadline


def run_block(ops, selections: dict, *, out_dir, node_refs: dict, gpus=(0, 1, 2, 3), reference_src: str = DEFAULT_REF,
              deadline: datetime.datetime | None = None, node_runner=None, now=None, s0_run_id: str | None = None,
              validity: dict | None = None, search_first: dict | None = None,
              order=BLOCK_ORDER, band=BAND, output_name='block_A.json', metadata=None,
              prepare_s0=False, do_prime=False) -> dict:
    """S0 -> sel(A-NAT) -> sel(A-ISW) -> sel(A-RND) on this node; voids re-run once at the end; fallbacks as ruled.
    selections: {arm: load_selection(...)}; node_refs: {node: t_S0,node}."""
    now = now or (lambda: datetime.datetime.now(datetime.timezone.utc))
    out_dir = Path(out_dir)
    nr = node_runner or checked_node_runner
    runners = {arm: _runner(ops, arm, gpus=gpus, out_dir=out_dir, reference_src=reference_src, node_runner=nr)
               for arm in dict.fromkeys((*order[1:], REF_ARM))}
    s0 = {'arm': REF_ARM, 'run_id': s0_run_id or selections[REF_ARM]['run_id'], 'candidate_id': 'block', 'config': {}}
    plan = [('S0', s0)] + [(arm, selections[arm]) for arm in order[1:]
                           if selections.get(arm) and not selections[arm].get('is_start')]      # D20: S0 stands for it
    need = float(runners['A-RND'].worst_case_s)
    log = {'kind': 'block', 'version': 'AMENDMENT_EXT_004', 'node': ops.node, 'gpus': list(gpus), 'order': list(order),
           'band': list(band), 'deadline': deadline.isoformat() if deadline else None, 'utc_start': _utc(), 'runs': [], **(metadata or {})}
    final: dict[str, dict] = {}
    pending_rerun = []
    hard_stop = False

    def attempt(name, item, rerun):
        if not _deadline_ok(now(), deadline, need):
            log['runs'].append({'name': name, 'abandoned': 'hard stop', 'rerun': rerun, 'utc': _utc()})
            return None
        arm = item['arm']
        rec = _score(ops, runners[arm], item)
        it = item_record(rec, name=name, band=band if name == 'S0' else None,
                         bind_digest=item.get('bind_digest') if name != 'S0' else None,
                         config=item.get('config'), overlay=item.get('overlay'))
        it['rerun'] = rerun
        log['runs'].append(it)
        if it.get('fatal'):
            raise BlockError(f'{name}: {it["fatal"]} (attempt {it["attempt_id"]}); the block stops')
        return it

    try:
        if prepare_s0 and _deadline_ok(now(), deadline, need):
            log['prepare'] = runners[REF_ARM].prepare(s0['run_id'])
            rejected = g1_reference(log['prepare']).rejections
            if rejected:
                raise BlockError('S0 reference preparation failed: ' + '; '.join(g['detail'] for g in rejected))
        if do_prime and _deadline_ok(now(), deadline, need):
            log['prime'] = prime(ops)
            if log['prime'].get('code') != 0:
                raise BlockError('cache priming failed')
        for name, item in plan:
            it = attempt(name, item, False)
            if it is None:
                hard_stop = True
                break
            final[name] = it
            if it['void'] and it['rerun_allowed']:
                pending_rerun.append((name, item))
        # Re-runs once, at the end, in the original order.
        for name, item in ([] if hard_stop else pending_rerun):
            it = attempt(name, item, True)
            if it is None:
                break
            if not it['void']:
                final[name] = it                              # a re-run replaces a void value only
            else:
                final[name] = {**final[name], 'void_twice': True, 'rerun_record': it['attempt_id']}
    except (BlockError, ContainmentError) as exc:
        log['stopped'] = str(exc)
    log['values'] = _values(final, selections, node_refs, ops.node, validity or {}, arms=order[1:])
    log['search_first'] = search_first or {}
    log['block_void_reruns'] = sum(1 for r in log['runs'] if r.get('rerun') and 'abandoned' not in r)
    log['utc_end'] = _utc()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / output_name).write_text(json.dumps(log, indent=1, default=str), encoding='utf-8')
    return log


# ---------------------------------------------------------------- AMENDMENT_EXT_005 G4 calibration inputs (G-FIN)
EXT005_S0_D_NORMS = (3.41, 4.27, 3.43, 4.22, 4.28)    # the five pre-arm unchanged-S0-on-D norms (EXT_005 Facts)
EXT005_S0_DPRIME_NORM = 4.79                         # unchanged S0 on D' vs ref'(D'), X2 F2.11
G4_SENTINEL = '@@X2_G4PAIR@@'
G4_PAIR_SCRIPT = '''import json, os, pathlib, sys
sys.path.insert(0, %(tools)r)
import score_runner as sr
a, b, score_py = sys.argv[1:4]
out = {"models": {}}
try:
    for p in (a, b):
        sr.check_safe_model_dir(p)
        rec = json.load(open(os.path.join(os.path.dirname(p), "TREE_DIGEST.json")))["tree_digest"]
        out["models"][p] = {"recorded": rec, "before": sr.tree_digest_files(p)}
        if out["models"][p]["before"] != rec:
            raise ValueError("cache digest differs from TREE_DIGEST.json: " + p)
    score = sr.load_score_module(score_py, pathlib.Path("/nonexistent/home_agent"), pathlib.Path(a))
    ma, mb = score.load_model(a), score.load_model(b)              # the safe loader: safetensors only
    nd, mx = score.compute_norm_and_max_diff(ma, mb)
    del ma, mb
    for p in (a, b):
        out["models"][p]["after"] = sr.tree_digest_files(p)
        if out["models"][p]["after"] != out["models"][p]["recorded"]:
            raise ValueError("cache digest changed during the comparison: " + p)
    out.update(ok=True, norm_diff=float(nd), max_scalar_diff=float(mx))
except Exception as exc:
    out.update(ok=False, error=repr(exc)[:500])
print(%(sentinel)r + json.dumps(out))
'''
G4_TOOLS = '/data/infrarsi/ext/tools/x2g4'


def node_pair_runner(ops, a: str, b: str, *, timeout: int = 1800) -> dict:
    """One pair on the node: a root process (CUDA hidden: CPU only) loads both cached ref'(D') models through the
    scorer's safe loader and takes score.py's own compute_norm_and_max_diff; both caches' tree digests are checked
    against TREE_DIGEST.json before and after. Nothing in the run directories is written."""
    ops.mkdirs([G4_TOOLS])
    script = G4_PAIR_SCRIPT % {'tools': rmod.REMOTE_TOOLS, 'sentinel': G4_SENTINEL}
    ops.write_files({f'{G4_TOOLS}/g4_pair.py': script}, mode=0o644)
    score_py = taskmod.NODE_TASK + '/assets/score.py'
    res = ops.sh(f"CUDA_VISIBLE_DEVICES='' timeout --signal=TERM --kill-after=30 {timeout} "
                 f"{taskmod.NODE_ENV}/bin/python {G4_TOOLS}/g4_pair.py '{a}' '{b}' '{score_py}'", timeout=timeout + 60)
    out = res.get('stdout') or ''
    if G4_SENTINEL not in out:
        return {'ok': False, 'error': f'no result (code {res.get("code")}): {(res.get("stderr") or "")[-400:]}'}
    return json.loads(out.rsplit(G4_SENTINEL, 1)[1].strip().splitlines()[0])


def _locked_attempt(arm_run: str, records: Path) -> str | None:
    """The locked re-score's attempt id from the arm's records (A-NAT: extra.locked; searches: locked)."""
    res = _read(Path(records) / arm_run / 'result.json')
    for locked in ((res.get('extra') or {}).get('locked'), res.get('locked')):
        if isinstance(locked, dict) and locked.get('attempt_id'):
            return locked['attempt_id']
    return None


def g4_inputs(ops, *, runs=('anat_b', 'aisw_a', 'arnd_a'), records, gpus=(0, 1, 2, 3), out_path,
              pair_runner=None, lease_run_id: str | None = None, block_json: Path | None = None) -> dict:
    """AMENDMENT_EXT_005 inputs for the G-FIN summary (X7 `--g4-inputs`): (a) the pairwise norms among the arms' cached
    ref'(D') models, one pair at a time, (b) each arm's locked D' norm from its node_result.json, with sources and
    sha256; all under one own node lease (released after, fail closed). `final` only with all three pairs."""
    from itertools import combinations
    from ..common.remote_exec import GpuLease
    records = Path(records)
    pr = pair_runner or node_pair_runner
    lease = GpuLease(ops, env='rebench_llmf', gpus=gpus, mode='own')
    lease_run_id = lease_run_id or 'g4inputs_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
    doc = {'final': False, 'refprime_pairwise_norms': [], 'locked_dprime_norms': {}, 'node': ops.node,
           'computed': 'root process, CPU (CUDA hidden), score.py compute_norm_and_max_diff via the scorer safe loader',
           'utc_start': _utc(), 'lease_run_id': lease_run_id}
    lease.acquire(lease_run_id, 'g4inputs')
    try:
        caches = {r: f'{taskmod.NODE_RUNS}/{r}/ref_permuted/trained_model' for r in runs}
        for ra, rb in combinations(runs, 2):
            got = pr(ops, caches[ra], caches[rb])
            row = {'pair': [ra, rb], 'source': f'{ops.node}:{caches[ra]} vs {caches[rb]}', 'ok': bool(got.get('ok'))}
            if got.get('ok'):
                row.update(norm_diff=got['norm_diff'], max_scalar_diff=got.get('max_scalar_diff'),
                           tree_digests={r: (got.get('models') or {}).get(caches[r], {}).get('recorded') for r in (ra, rb)})
            else:
                row['error'] = got.get('error')
            doc['refprime_pairwise_norms'].append(row)
        for r in runs:
            att = _locked_attempt(r, records)
            p = f'{taskmod.NODE_RUNS}/{r}/results/{att}_node_result.json'
            payload = ops.read_json(p) if att else None
            sha = ops.hash_files([p]).get(p) if att else None
            g4 = (((payload or {}).get('result') or {}).get('g4') or {}) if isinstance(payload, dict) else {}
            doc['locked_dprime_norms'][att or f'{r}:missing'] = {
                'run': r, 'norm_cand_dprime_vs_refprime': g4.get('norm_cand_dprime_vs_refprime'),
                'source': f'{ops.node}:{p}', 'sha256': sha, 'g4_computed': g4.get('computed')}
    finally:
        released = lease.release()
        doc['lease'] = {'released': released, 'kept': list(lease.acquired), 'failures': list(lease.release_failures)}
    good = [r for r in doc['refprime_pairwise_norms'] if r.get('ok')]
    doc['final'] = len(good) == 3 and len(runs) == 3
    doc['preview'] = _g4_limit_preview(good, block_json, records)
    doc['utc_end'] = _utc()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(doc, indent=1, default=str), encoding='utf-8')
    return doc


def _g4_limit_preview(pairs: list, block_json, records: Path) -> dict:
    """limit_D' = 5.3 + (m' - m) as EXT_005 / X7 define it (preview; X7's summary computes the value of record):
    m = mean(the five pre-arm S0-on-D norms + the block S0 + node_ref norms), m' = mean(4.79 + the pairwise norms)."""
    m_samples = list(EXT005_S0_D_NORMS)
    src = ['EXT_005 Facts (5)']
    blk = Path(block_json) if block_json else Path(records) / 'block_a' / 'block_A.json'
    if blk.exists():
        for r in _read(blk).get('runs') or []:
            if r.get('name') == 'S0' and not r.get('void') and r.get('norm_diff') is not None:
                m_samples.append(float(r['norm_diff']))
                src.append(f'block S0 {r.get("attempt_id")}')
    for p in sorted((Path(records) / 'block_a').glob('node_ref_*.json')):
        for a in _read(p).get('attempts') or []:
            if not a.get('void') and a.get('norm_diff') is not None:
                m_samples.append(float(a['norm_diff']))
                src.append(f'{p.stem} {a.get("attempt_id")}')
    mp_samples = [EXT005_S0_DPRIME_NORM] + [float(r['norm_diff']) for r in pairs]
    m, mp = sum(m_samples) / len(m_samples), sum(mp_samples) / len(mp_samples)
    return {'m': m, 'm_samples': m_samples, 'm_sources': src, 'm_prime': mp, 'm_prime_samples': mp_samples,
            'limit_dprime': 5.3 + (mp - m), 'note': 'preview; the value of record is computed by the G-FIN summary'}


def _values(final: dict, selections: dict, node_refs: dict, block_node: str, validity: dict, *, arms=('A-NAT', 'A-ISW', 'A-RND')) -> dict:
    """Block values per AMENDMENT_EXT_004 (b)/(c): t_S0,B, t_B per arm (or its fallback), y_B, node factors, the
    magnitude comparisons with the 5 % margin."""
    s0 = final.get('S0')
    if s0 and not s0['void']:
        t_s0b, s0_note = s0['t'], 'measured'
    else:
        t_s0b, s0_note = WARM_ANCHOR_S, 'S0 void (twice) or not run: t_S0,B := 216.1'
    refs = dict(node_refs or {})
    refs[block_node] = t_s0b
    arm_names, arms = tuple(arms), {}
    for arm in arm_names:
        sel = selections.get(arm)
        if not sel:                                           # EXT_006: a native arm that ended without a valid score
            arms[arm] = {'t_B': None, 'label': 'degenerate (no valid score)', 'source': 'not in the block',
                         'locked_validity': validity.get(arm)}
            continue
        it = final.get(arm)
        t_locked = sel.get('t_locked')
        n_arm = sel.get('node')
        row = {'selected': sel.get('selected'), 'node': n_arm, 't_locked': t_locked, 'locked_attempt_id': sel.get('locked_attempt_id'),
               'block_attempt': it and it.get('attempt_id'), 'locked_validity': validity.get(arm), 'fallback': False}
        if sel.get('is_start'):                                # D20: locked value = the node's S0 reference
            t_locked = refs.get(n_arm)
            row.update(t_locked=t_locked, t_B=t_s0b, label='selection_is_start (S0)', source='the block S0 (D20)',
                       skipped='selection_is_start', invalid_accept=0)
        elif it and not it['void']:
            row.update(t_B=it['t'], label='', source='block')
        else:
            t_ref = refs.get(n_arm)
            if t_locked and t_ref:
                row.update(t_B=float(t_locked) * t_s0b / float(t_ref), label='(fallback)', fallback=True,
                           source=f'fallback: t_locked x t_S0,B / t_S0,{n_arm}')
            else:
                row.update(t_B=None, label='(fallback unavailable)',
                           source=f'no block value and no node reference for {n_arm}' if not t_ref else 'no locked time')
            row['block_void_reasons'] = it and it.get('void_reasons')
        if row.get('t_B'):
            row['y_B'] = math.log(t_s0b / row['t_B'])
            if t_locked:
                row['locked_vs_block'] = float(t_locked) / row['t_B']
        arms[arm] = row

    def compare(a, b):
        ta, tb = arms[a].get('t_B'), arms[b].get('t_B')
        if not (ta and tb):
            degenerate = [x for x in (a, b) if arms[x].get('label', '').startswith('degenerate')]
            return {'pair': f'{a} vs {b}', 'magnitude': ('no comparison (native arm degenerate under the D1 driver)'
                                                          if degenerate else 'not computable')}
        lab = (f'{a} faster by >= 5 %' if ta <= (1 - MARGIN) * tb else
               f'{b} faster by >= 5 %' if tb <= (1 - MARGIN) * ta else 'within 5 %')
        fb = any(arms[x].get('fallback') for x in (a, b))
        return {'pair': f'{a} vs {b}', 'ratio': ta / tb, 'magnitude': lab + (' (fallback)' if fb else '')}
    node_factor = {n: (t / t_s0b if t else None) for n, t in (node_refs or {}).items()}
    if arm_names == ('A-NAT2', 'A-IS2'):
        node_factor[block_node] = 1.0
    return {'t_S0_B': t_s0b, 't_S0_B_note': s0_note, 'node_refs': refs, 'node_factor': node_factor, 'arms': arms,
            'magnitude': ([compare('A-ISW', 'A-NAT'), compare('A-ISW', 'A-RND'), compare('A-NAT', 'A-RND')]
                          if arm_names == ('A-NAT', 'A-ISW', 'A-RND') else
                          [compare('A-IS2', 'A-NAT2')]),
            'note': '(3) magnitude only; the labels also need (1)(2) validity at the per-arm locked re-score and (4) '
                    'AUC_budget with node-referenced y (W5).'}
