"""F1 guards (AMENDMENT_EXT_003 Q3 hard checks, Q2 locked-pair guards): vetoes only, each with an id.

    check(parsed, point, cap_s=1800, ...) -> ext.common.guards.GuardResult      every hard check of one screen
    locked_self_mismatch(locked, own_screen) / void_pair(uc_half)               the locked-pair guards (arms.run_locked;
                                                                                always the frozen numerics_gates.json)
    uc_reference_problems(r) / ht_reference_problems(r)                         identity of the two reference screens
                                                                                (arms, hooks and the CLI refuse others)

Hard checks (ids):
  complete_verified   32 rank signal logs with the 30 train_step updates and 4 node exits 0 (released clean)
  frozen_hashes       attempt pins (training source 4ab14ced..., data manifest de1fc04b..., sources 635ad368...,
                      update supervision) and the 86-file code inventory against the variant layer's frozen_manifest.json
                      (whose unchanged files are the formal-54 freeze d6a8f3...); every rank's trainer echoes that
                      inventory; the local variant layer (switch_space.json, node launcher) matches the manifest. With
                      code_overlay (the code arm's screen, {'code/<name>': sha256}): the inventory must be the manifest's
                      with exactly those files - each a mutable file of code_space - replaced; every other file frozen
  knob_binding        the requested S_F1 point echoed by the launch record (attempt.json variant / systems policy /
                      G2 / replay / contracts / seed / run length), the launch env (job.json knob variables, fixed
                      FLASH_STOP / FLASH_RESUME / aux readback), the commands the four launchers actually ran, the
                      signals' identity (this attempt's id and run id on every row) and every rank's trainer result
                      (variant, seed, aux readback mode)
  wall_cap            first launcher start -> last node exit <= the screen's cap (1,800 s; 3,600 s for the two reference
                      screens on a cold cache); a launcher timeout (exit 124) is a veto
  oom_nccl            no OOM / NCCL error line in the collected node logs, no 'failure' signal row, no such node error
Code-arm screens only (NM15 P1; check(..., code_overlay=<hashes>), i.e. a screen whose runtime carried an overlay - the
F-A0-CODE search screens with an edit and the candidate half of its locked pair):
  timer_consistency   the production update timer (values.timing.update_seconds, the metric numerator) must account
                      for the wall time an EXTERNAL, out-of-process clock shows for the scoring window. The clock is the
                      fn_extclock watcher: a stdlib process the runner starts on each node (from fn_node<r>.sh, before
                      the frozen launcher) that stamps its own time.monotonic() when each rank's train_step signal first
                      appears. Its interpreter is not the editable trainer runtime, so an edit cannot advance it (NM18
                      P1-2: signals time_utc is written inside the trainer and can be forged in lockstep with the timer).
                      Per rank the span t_ext(30) - t_ext(16) is the wall time of updates 17..30 on that rank's node (one
                      node clock, no cross-node sync); the denominator is the slowest rank's span. A code edit could
                      neutralize a host-device sync (e.g. torch.cuda.synchronize) so GPU work falls outside the timed
                      interval while the CE / grad-norm / LR gates stay green; the external clock still spans it. A code-
                      arm screen without usable external records is invalid (fail-closed). Never runs for a switch-only
                      screen (no overlay), so those screens are byte-identical.
Reports (never veto): inter_step_consistency (not applicable, D-F1-4), scalar_rank_consistency, gates/space hashes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..common.guards import GuardResult
from . import knobs as K
from . import parse as P
from . import switch_space as S

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / 'frozen_manifest.json'
NODE_DIR = HERE / 'node'
IDS = ('complete_verified', 'frozen_hashes', 'knob_binding', 'wall_cap', 'oom_nccl')
CODE_IDS = ('timer_consistency',)      # NM15 P1: added by check() only for a code-arm screen that ran an overlay
LOCKED_IDS = ('locked_self_mismatch', 'void_pair')
CAP_S, COLD_REFERENCE_CAP_S = 1800.0, 3600.0
# Timer consistency (NM15 P1 -> NM18 P1-2: external clock). ratio = sum over the window of the per-update max-over-
# ranks values.timing.update_seconds (the metric numerator) / the slowest rank's EXTERNAL-clock span from update 16 to
# update 30 (max over ranks of t_ext(30) - t_ext(16), the fn_extclock watcher's out-of-process time.monotonic()). The
# span cancels the watcher's constant polling lag at both endpoints, so this external statistic matches the time_utc
# span to <0.01 %. Calibrated on the 46 honest 2026-10-28 window screens (NM18_calibration/calib_rows.json candidate B,
# sum max-rank T / max_r span): ratio min 0.9935, median 0.9947, max 0.9959 (the ~0.5 % shortfall is the per-update
# logging/snapshot overhead the wall clock sees and the timer does not). The watcher polls EXTCLOCK_POLL_S (~0.1 s), so
# each endpoint carries <=0.1 s of jitter on a ~116 s span (<0.09 %), far inside the margin. The threshold sits below
# the honest minimum by a wide margin and at/below the ratio any beyond-margin (>= ~5 %) timer evasion would produce, so
# honest screens pass and a neutralized sync fails even when the in-process time_utc is forged in lockstep.
TIMER_RATIO_MIN = 0.95
TIMER_CALIBRATION = {'honest_screens': 46, 'ratio_min': 0.9935, 'ratio_median': 0.9947, 'ratio_max': 0.9959,
                     'threshold': TIMER_RATIO_MIN, 'margin_below_honest_min': round(0.9935 - TIMER_RATIO_MIN, 4),
                     'clock': 'external out-of-process watcher (fn_extclock); time_utc kept as a report only',
                     'statistic': 'sum(per-update max-rank update_seconds) / max over ranks of the external-clock span '
                                  't_ext(30) - t_ext(16)', 'poll_resolution_s': 0.1, 'window': list(P.WINDOW)}
SEED, RUN_LIMIT = 42, 30
PINS = ('training_source_sha256', 'data_manifest_sha256', 'sources_sha256', 'update_supervision_sha256')
# w_family_contract.FLAGS of the formal-54 freeze (the W block's job env; a test compares with the frozen file)
W_FLAGS = {'FLASH_COMBINE_SEGSUM': '2', 'FLASH_HOST_SYNC_MEMO': '1', 'FLASH_PROJ_THREEPHASE': '0',
           'FLASH_QSA_DKDV_FASTMASK': '0', 'FLASH_QSA_TUNED': '3'}
KNOB_ENV = tuple(K.LAUNCH_ENV.values()) + ('FLASH_QSA_DKDV_FASTMASK', 'FLASH_G2_MODE', 'FLASH_AUX_READBACK_MODE') + tuple(
    k for k in W_FLAGS if k != 'FLASH_QSA_DKDV_FASTMASK')
FIXED_ENV = {'FLASH_STOP': '1024', 'FLASH_RESUME': '0', 'FLASH_AUX_READBACK_MODE': 'control'}
# The base job env of every registered recipe (formal-54 bundles: identical values in all of them); a screen's env is
# exactly: these fixed values + the knob variables (KNOB_ENV, from the point) + the per-run variables below, which are
# cross-bound to the attempt and job records. Anything else (another FLASH_* or NCCL_* variable, ...) is undeclared.
BASE_ENV = {'CUBLASLT_HEURISTICS_CACHE_CAPACITY': '65536', 'CUDA_DEVICE_MAX_CONNECTIONS': '8',
            'FLASH_DIAG_STORAGE': 'expert_adam_v_offload', 'FLASH_DIAG_TP': '8', 'FLASH_EXPERT_STREAM_MODE': 'streams',
            'FLASH_MTP_ENABLED': '1', 'FLASH_MUON_RATIO': '1', 'FLASH_TP8_CHECKPOINT_MODE': 'save',
            'PYTHONDONTWRITEBYTECODE': '1',
            'FLASH_IMPORT_SNAPSHOT_SHA256': '12f4e87e4461f9ba38c249cbf745f9871347208f28cd5c939c12b8f0612d6a43',
            'FLASH_IMPORT_WORKER_SHA256': 'ff1a766cf87aec457c993c1e9a5bcb164cd784ed6917e84f6b5a92f3b5964d24', **FIXED_ENV}
PER_RUN_ENV = ('FLASH_CHECKPOINT_STEM', 'FLASH_IMPORT_SNAPSHOT', 'FLASH_LOG_RUN_ID', 'FLASH_RUN_ATTEMPT',
               'FLASH_TOPOLOGY_CAPTURE_RUN', 'LP_CACHE_ROOT')
RUN_PREFIX = 'flash32_joint_'                 # prepare_precision_jobs naming: run_id = prefix + stem; run = run_id + '_' + attempt
EXT_CACHE = '/data/infrarsi/ext/cache/flashnext'                 # variant.FN_CACHE (ext_bind's LP_CACHE_ROOT)
FIXED_ATTEMPT = {'stop': 1024, 'precision_arm': {'id': 'bf16', 'mode': 'bf16', 'parts': [], 'scope': 'all'},
                 'allocator_policy': 'allocator-retry-events-zero-oom-v1',
                 'data_order_policy': 'frozen_luna_q50_schedule_and_pack_plan_absolute_update_no_seed_reshuffle'}
TRAINER_STATUS = 'COMPLETED_FORMAL_TP8_SEGMENT'
START_SKEW_S = 1.0


def _sha_file(p: Path) -> str | None:
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except OSError:
        return None


def _sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_manifest(path: str | Path | None = None) -> dict | None:
    try:
        m = json.loads(Path(path or MANIFEST_PATH).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return m if isinstance(m, dict) else None


# ---------------------------------------------------------------- the five hard checks
def complete_verified(parsed: dict, gid: str = 'complete_verified') -> GuardResult:
    g = GuardResult()
    c = parsed.get('completeness') or {}
    if not parsed.get('complete'):
        parts = [f'{c.get("n_ranks_ok")}/{P.RANKS} rank signal logs with updates 1..{P.UPDATES}']
        parts += [f'rank {x["rank"]}: {"; ".join(x["problems"][:2])}' for x in (c.get('rank_problems') or [])[:3]]
        parts += [f'node {x["node"]}: {"; ".join(x["problems"])}' for x in (c.get('node_problems') or [])]
        parts += list(c.get('problems') or [])
        g.reject(gid, '; '.join(parts))
    return g


def overlay_inventory(inv: dict, code_overlay: dict) -> tuple[dict, list[str]]:
    """The inventory a code-arm screen must carry: `inv` with the overlay's files replaced, and the problems of the
    overlay itself (a file outside the code space, a file the runtime does not have, a malformed hash)."""
    from . import code_space as C
    want, problems = dict(inv), []
    for path, h in sorted((code_overlay or {}).items(), key=lambda x: str(x[0])):
        name = C.name_of(path)
        if path not in C.MUTABLE_PATHS:
            problems.append(f'code overlay file {str(path)[:80]} is not a mutable file of {C.SPACE_ID}')
        elif name not in inv:
            problems.append(f'code overlay file {path} is not in the variant runtime inventory')
        elif not (isinstance(h, str) and len(h) == 64):
            problems.append(f'code overlay file {path}: malformed sha256 {str(h)[:20]!r}')
        else:
            want[name] = h
    return want, problems


def frozen_hashes(parsed: dict, manifest: dict | None = None, *, local_check: bool = True,
                  gid: str = 'frozen_hashes', code_overlay: dict | None = None) -> GuardResult:
    g = GuardResult()
    manifest = manifest if manifest is not None else load_manifest()
    if not manifest:
        return g.reject(gid, f'variant layer frozen_manifest.json missing or unreadable ({MANIFEST_PATH.name})')
    a = (parsed.get('launch') or {}).get('attempt')
    if not a:
        return g.reject(gid, 'no attempt.json: the source pins cannot be checked')
    pins = manifest.get('source_pins') or {}
    for k in PINS:
        if k not in pins:
            g.reject(gid, f'manifest has no pin {k}')
        elif a.get(k) != pins[k]:
            g.reject(gid, f'attempt {k} {str(a.get(k))[:16]} != frozen {pins[k][:16]}')
    inv = (manifest.get('variant_runtime') or {}).get('inventory')
    if not isinstance(inv, dict) or len(inv) != 86:
        g.reject(gid, 'manifest variant_runtime.inventory is not the 86-file code inventory')
    elif code_overlay is not None:                       # setting 2: exactly the declared mutable files replaced
        want, problems = overlay_inventory(inv, code_overlay)
        for p in problems:
            g.reject(gid, p)
        code = a.get('code_sha256') or {}
        if code != want:
            diff = sorted(n for n in set(code) | set(want) if code.get(n) != want.get(n))
            g.reject(gid, f'bundle code inventory differs from the frozen variant runtime with the declared code overlay '
                          f'in {len(diff)} files: {diff[:4]}')
        g.report('code_overlay', files=sorted(code_overlay), replaced=sorted(n for n in want if want[n] != inv[n]))
    elif a.get('code_sha256') != inv:
        code = a.get('code_sha256') or {}
        diff = sorted(n for n in set(code) | set(inv) if code.get(n) != inv.get(n))
        g.reject(gid, f'bundle code inventory differs from the frozen variant runtime in {len(diff)} files: {diff[:4]}')
    ranks = parsed.get('rank_results') or []
    want = a.get('code_sha256_digest')
    off = [r for r, x in enumerate(ranks) if x is not None and x.get('bundle_code_sha256') != want]
    if off:
        g.reject(gid, f'trainer of ranks {off[:6]} ran another code inventory than the bundle')
    freeze = (manifest.get('formal54') or {}).get('code_freeze_sha256')
    from . import variant as V
    if freeze != V.CODE_FREEZE_SHA256:
        g.reject(gid, f'manifest formal-54 code freeze {str(freeze)[:12]} != {V.CODE_FREEZE_SHA256[:12]}')
    if local_check:
        for p in local_layer_problems(manifest):
            g.reject(gid, p)
    g.report('frozen_manifest', sha256=_sha_file(MANIFEST_PATH), switch_space_sha256=manifest.get('switch_space_sha256'))
    return g


def local_layer_problems(manifest: dict) -> list[str]:
    """The variant layer on this PC against its frozen manifest: switch_space.json, the node launcher, and the runtime
    inventory = the formal-54 CODE_FREEZE outside the manifest's declared edits."""
    from . import variant as V
    out = []
    if manifest.get('switch_space_sha256') != S.sha256():
        out.append('switch_space.json differs from the frozen manifest')
    for name, h in (manifest.get('node_files') or {}).items():
        if _sha_file(NODE_DIR / name) != h:
            out.append(f'node/{name} differs from the frozen manifest')
    inv = (manifest.get('variant_runtime') or {}).get('inventory')
    freeze_file = V.FROZEN_DIR / 'CODE_FREEZE.json'
    if isinstance(inv, dict) and freeze_file.exists():
        common = json.loads(freeze_file.read_text(encoding='utf-8-sig')).get('common_code_sha256') or {}
        changed = set((manifest.get('variant_runtime') or {}).get('changed_files') or [])
        drift = sorted(n for n in inv if n not in changed and inv.get(n) != common.get(n))
        if drift or set(inv) != set(common):
            out.append(f'variant runtime is not the formal-54 freeze outside its declared edits: {drift[:4]}')
    return out


def expected_launch(point: dict) -> dict:
    """What a bundle built from this S_F1 point must carry: variant name, systems policy / G2 / replay, contracts and
    the knob environment (job.json), from the knob layer itself (knobs.variant_name / systems_policy / job_env)."""
    knobs = S.to_knobs(point)
    c, errors, _ = K.canonical(knobs)
    if errors:
        raise ValueError('; '.join(errors))
    name = K.variant_name(c)
    fam = c['family']
    env = {'FLASH_AUX_READBACK_MODE': 'control'}
    if fam == 'W':
        env.update(W_FLAGS)
    env.update(K.job_env(c))
    systems = K.systems_policy(c, name) if fam == 'systems' else None
    return {'variant': name, 'family': fam, 'systems_policy': systems,
            'g2_mode': ('candidate' if c['retain_expert_layers'] else 'control') if fam == 'systems' else None,
            'replay_mode': 'candidate' if fam == 'systems' else None, 'uc_contract': fam == 'U_plus_C',
            'w_family_contract': fam == 'W', 'env': {k: env.get(k) for k in KNOB_ENV}}


def env_problems(env: dict, attempt: dict, job: dict, exp: dict) -> list[str]:
    """The launch env against the requested point (knob variables), the fixed base env and the per-run variables
    (bound to the attempt / job records); every other variable is undeclared."""
    out = []
    for k in KNOB_ENV:
        if env.get(k) != exp['env'].get(k):
            out.append(f'job env {k} = {env.get(k)!r} != requested {exp["env"].get(k)!r}')
    for k, v in BASE_ENV.items():
        if env.get(k) != v:
            out.append(f'job env {k} = {env.get(k)!r} != fixed {v!r}')
    meta = job.get('metadata_directory')
    bind = {'FLASH_LOG_RUN_ID': attempt.get('plan_run_id'), 'FLASH_TOPOLOGY_CAPTURE_RUN': attempt.get('run'),
            'FLASH_RUN_ATTEMPT': f'{meta}/attempt.json' if meta else None,
            'FLASH_IMPORT_SNAPSHOT': f'{meta}/import_snapshot.json' if meta else None, 'LP_CACHE_ROOT': EXT_CACHE}
    for k, want in bind.items():
        if want is None or env.get(k) != want:
            out.append(f'job env {k} = {str(env.get(k))[:100]!r} is not bound to this attempt ({str(want)[:100]!r})')
    stem, run_id = env.get('FLASH_CHECKPOINT_STEM'), attempt.get('plan_run_id')
    if not (isinstance(stem, str) and stem and run_id == RUN_PREFIX + stem
            and attempt.get('run') == f'{run_id}_{attempt.get("attempt_id")}'):
        out.append(f'job env FLASH_CHECKPOINT_STEM = {str(stem)[:80]!r} is not bound to this run '
                   f'(run_id = {RUN_PREFIX}<stem>, run = <run_id>_<attempt_id>)')
    known = set(KNOB_ENV) | set(BASE_ENV) | set(PER_RUN_ENV)
    extra = sorted(k for k in env if k not in known)
    if extra:
        out.append(f'undeclared job env variables {extra[:8]} (only the switch variables of the space may differ)')
    return out


def _expand(value: str, rep: dict) -> str:
    value = str(value)
    for token, replacement in rep.items():
        value = value.replace(token, replacement)
    return value


def knob_binding(parsed: dict, point: dict, gid: str = 'knob_binding') -> GuardResult:
    g = GuardResult()
    try:
        exp = expected_launch(point)
    except (KeyError, TypeError, ValueError) as exc:
        return g.reject(gid, f'the requested point is not an S_F1 point: {exc}')
    launch = parsed.get('launch') or {}
    a, job = launch.get('attempt'), launch.get('job')
    if not a or not job:
        return g.reject(gid, 'attempt.json / job.json not collected: the launch record cannot be checked')
    checks = [('infra_variant', a.get('infra_variant'), exp['variant']), ('training_seed', a.get('training_seed'), SEED),
              ('run_limit', a.get('run_limit'), RUN_LIMIT), ('screen_stop', a.get('screen_stop'), RUN_LIMIT),
              ('resume', a.get('resume'), 0), ('systems_policy', a.get('systems_policy'), exp['systems_policy']),
              ('g2_mode', a.get('g2_mode'), exp['g2_mode']), ('replay_mode', a.get('replay_mode'), exp['replay_mode']),
              ('uc_contract present', a.get('uc_contract'), exp['uc_contract']),
              ('w_family_contract present', a.get('w_family_contract'), exp['w_family_contract']),
              ('run (= job name)', a.get('run'), job.get('name'))]
    checks += [(k, a.get(k), v) for k, v in FIXED_ATTEMPT.items()]
    for what, got, want in checks:
        if got != want:
            g.reject(gid, f'attempt.json {what} = {json.dumps(got, default=str)[:120]} != requested {json.dumps(want, default=str)[:120]}')
    env = job.get('env')
    if not isinstance(env, dict):
        return g.reject(gid, 'job.json env missing or malformed')
    for p in env_problems(env, a, job, exp):
        g.reject(gid, p)
    spec = launch.get('fn_spec') or {}
    nodes = parsed.get('nodes') or []
    if spec.get('master_addr') and spec.get('host_numbers') and spec.get('out_root') and job.get('command'):
        for n, nd in enumerate(nodes):
            rep = {'{node_rank}': str(n), '{master_addr}': str(spec['master_addr']),
                   '{master_port}': str(job.get('master_port')), '{output_dir}': f'{spec["out_root"]}/{job.get("name")}',
                   '{host}': str(spec['host_numbers'][n]) if n < len(spec['host_numbers']) else '?'}
            want = [_expand(v, rep) for v in job['command']]
            if nd.get('command') is None:
                g.reject(gid, f'node {n}: launch record (pid file) missing: the launched command is unproven')
            elif nd['command'] != want:
                g.reject(gid, f'node {n}: launched command differs from the bundle job command')
    else:
        g.reject(gid, 'fn_spec.json or the job command not collected: the launched commands are unproven')
    comp = parsed.get('completeness') or {}
    ident = [x for x in comp.get('rank_problems') or [] if any('attempt_id' in p or 'run_id' in p for p in x['problems'])]
    if ident:
        g.reject(gid, f'signal rows of ranks {[x["rank"] for x in ident][:6]} carry another attempt or run identity')
    results = parsed.get('rank_results') or []
    missing = [r for r in range(P.RANKS) if r >= len(results) or results[r] is None]
    if missing:
        g.reject(gid, f'trainer results of ranks {missing[:8]} not collected: the runtime echo is unproven')
    for r, x in enumerate(results):
        if x is None:
            continue
        bad = [f'{k}={x.get(k)!r}' for k, want in (('status', TRAINER_STATUS), ('attempt_id', a.get('attempt_id')),
                                                     ('rank', r), ('log_run_id', a.get('plan_run_id')),
                                                     ('variant', exp['variant']), ('training_seed', SEED),
                                                     ('data_order_policy', FIXED_ATTEMPT['data_order_policy']),
                                                     ('aux_readback_mode', 'control'), ('steps', RUN_LIMIT), ('resume', 0))
               if x.get(k) != want]
        if bad:
            g.reject(gid, f'rank {r} trainer result: {", ".join(bad[:4])}')
    return g


def wall_cap(parsed: dict, cap_s: float = CAP_S, gid: str = 'wall_cap') -> GuardResult:
    g = GuardResult()
    nodes = parsed.get('nodes') or []
    starts = [n.get('started_unix') for n in nodes if P.finite(n.get('started_unix'))]
    ends = [n.get('finished_unix') for n in nodes if P.finite(n.get('finished_unix'))]
    for i, n in enumerate(nodes):
        a, e, f = n.get('started_unix'), n.get('exit_started_unix'), n.get('finished_unix')
        if P.finite(a) and not (P.finite(e) and abs(e - a) <= START_SKEW_S):
            g.reject(gid, f'node {i}: exit record start {e!r} missing or disagrees with the launch record start {a!r}')
        if P.finite(a) and P.finite(f) and f < a:
            g.reject(gid, f'node {i}: exit {f!r} before its start {a!r}')
    span = (max(ends) - min(starts)) if len(starts) == P.NODES and len(ends) == P.NODES else None
    timeouts = [i for i, n in enumerate(nodes) if n.get('exit_code') == 124 or
                (isinstance(n.get('error'), dict) and n['error'].get('type') == 'RunTimeout')]
    if (parsed.get('runner') or {}).get('timed_out'):
        g.reject(gid, 'the runner gave up waiting for the node exits (cap + margin + grace)')
    if timeouts:
        g.reject(gid, f'launcher timeout (exit 124) on nodes {timeouts}')
    if span is None:
        g.reject(gid, 'launcher start / node exit times missing: the wall span is unproven')
    elif span > float(cap_s):
        g.reject(gid, f'first launcher start -> last node exit {span:.0f} s > cap {float(cap_s):.0f} s')
    g.report('wall_span', span_s=span, cap_s=float(cap_s))
    return g


def oom_nccl(parsed: dict, gid: str = 'oom_nccl') -> GuardResult:
    g = GuardResult()
    logs = parsed.get('logs') or {}
    for kind in ('oom', 'nccl'):
        hits = logs.get(kind) or []
        if hits:
            g.reject(gid, f'{len(hits)} {kind.upper()} line(s), first {hits[0]["file"]}:{hits[0]["line"]}: {hits[0]["text"][:160]}')
    rows = parsed.get('failure_rows') or []
    if rows:
        g.reject(gid, f'{len(rows)} failure signal row(s), first rank {rows[0].get("rank")}: {rows[0].get("values", "")[:160]}')
    for i, n in enumerate(parsed.get('nodes') or []):
        text = json.dumps(n.get('error'), default=str) if n.get('error') else ''
        if text and (P.OOM_RE.search(text) or P.NCCL_RE.search(text)):
            g.reject(gid, f'node {i} launcher error: {text[:160]}')
    return g


def timer_consistency(parsed: dict, *, threshold: float = TIMER_RATIO_MIN, gid: str = 'timer_consistency') -> GuardResult:
    """Code-arm screens only (NM15 P1 / NM18 P1-2): the update timer must account for the wall time an EXTERNAL,
    out-of-process clock shows for the scoring window. ratio = sum over the window of the per-update max-over-ranks
    values.timing.update_seconds (the metric numerator) / the slowest rank's external-clock span t_ext(30) - t_ext(16)
    (parse.ext_clock, the fn_extclock watcher's own time.monotonic(), per node). Honest ratio >= 0.9935
    (TIMER_CALIBRATION); below `threshold` the timer has missed time between updates (GPU work moved outside the timed
    interval, e.g. a neutralized host-device sync), so the screen is invalid. Fail-closed: a code-arm screen whose
    per-update trainer times or external window endpoints cannot be derived is rejected, not quietly passed. The
    in-process time_utc ratio is reported for transparency but never gates: an edit can forge it in lockstep with the
    timer (NM18 P1-2), which is exactly why the denominator is taken outside the trainer interpreter."""
    g = GuardResult()
    a, b = P.WINDOW
    lo_u, hi_u = a - 1, b                              # the window updates 17..30 run between the end of 16 and of 30
    win = [s for s in (parsed.get('steps') or []) if a <= s.get('update') <= b]
    num = [s.get('max_rank_s') for s in win]
    expected = b - a + 1
    if len(win) != expected or not all(P.finite(x) and x > 0 for x in num):
        return g.reject(gid, f'the update timer cannot be checked: {len(win)}/{expected} window updates with a finite '
                             'per-update time (a code-arm screen without a checkable timer is invalid)')
    ext_block = parsed.get('ext_clock') or {}
    ext_problems = ext_block.get('problems') or []
    if ext_problems:                                   # NM20 point 6: a malformed / mispartitioned watcher file is invalid
        return g.reject(gid, 'the external-clock records are not well-formed (out-of-process watcher containment): '
                             + '; '.join(str(x) for x in ext_problems[:4])
                             + ' (a code-arm screen with unusable external-clock records is invalid)')
    ext = ext_block.get('by_rank') or {}
    spans, missing = [], []
    for r in range(P.RANKS):
        d = ext.get(r) if r in ext else ext.get(str(r)) or {}
        t_lo = d.get(lo_u, d.get(str(lo_u)))
        t_hi = d.get(hi_u, d.get(str(hi_u)))
        if P.finite(t_lo) and P.finite(t_hi) and t_hi > t_lo:
            spans.append(t_hi - t_lo)
        else:
            missing.append(r)
    # in-process time_utc span (the OLD, forgeable denominator) - reported only, never the veto
    tu = [s.get('record_gap_max_rank_s') for s in win]
    tu_ratio = round(sum(num) / sum(tu), 6) if all(P.finite(x) and x > 0 for x in tu) else None
    if missing or len(spans) != P.RANKS:
        return g.reject(gid, f'the update timer cannot be checked against the external clock (fn_extclock out-of-process '
                             f'watcher): external window span missing for {len(missing)} rank(s) {missing[:8]} (a code-arm '
                             'screen without usable external-clock records is invalid)')
    trainer_s, record_s = sum(num), max(spans)        # slowest rank's external wall span (most conservative denominator)
    ratio = trainer_s / record_s
    g.report(gid, ratio=round(ratio, 6), threshold=float(threshold), window=[a, b], trainer_window_s=round(trainer_s, 4),
             external_clock_window_s=round(record_s, 4), external_span_min_s=round(min(spans), 4),
             external_span_max_s=round(record_s, 4), ranks=P.RANKS, in_process_time_utc_ratio=tu_ratio,
             clock='external_out_of_process', honest_min=TIMER_CALIBRATION['ratio_min'],
             honest_median=TIMER_CALIBRATION['ratio_median'])
    if ratio < threshold:
        g.reject(gid, f'the update timer misses time between updates: trainer time / external-clock span over updates '
                      f'{a}..{b} = {ratio:.4f} < {threshold} (honest screens >= {TIMER_CALIBRATION["ratio_min"]}; GPU '
                      'work moved outside the timed interval, e.g. a neutralized host-device sync)')
    return g

def reports(parsed: dict) -> GuardResult:
    g = GuardResult()
    g.report('inter_step_consistency', status='not_applicable',
             detail='the production path has no inter-step timer; D-F1-4 (path asserts, frozen hashes, knob binding, '
                    'numerics compare incl. MTP CE instead)')
    sc = parsed.get('scalar_consistency') or {}
    g.report('scalar_rank_consistency', checked_ranks=sc.get('checked_ranks'), mismatches=(sc.get('mismatches') or [])[:6])
    return g


def check(parsed: dict, point: dict, *, cap_s: float = CAP_S, manifest: dict | None = None, skip=(),
          local_check: bool = True, code_overlay: dict | None = None) -> GuardResult:
    """Every Q3 hard check of one screen (ids in IDS; `skip` = ids the runner already applied, e.g. 'wall_cap').
    code_overlay: the code arm's declared overlay file hashes (frozen_hashes, and the code-arm timer_consistency check);
    None for every switch-only screen, which therefore gets exactly the five Q3 checks as before (byte-identical)."""
    g = GuardResult()
    if 'complete_verified' not in skip:
        g.extend(complete_verified(parsed))
    if 'frozen_hashes' not in skip:
        g.extend(frozen_hashes(parsed, manifest, local_check=local_check,
                               **({'code_overlay': code_overlay} if code_overlay is not None else {})))
    if 'knob_binding' not in skip:
        g.extend(knob_binding(parsed, point))
    if 'wall_cap' not in skip:
        g.extend(wall_cap(parsed, cap_s))
    if 'oom_nccl' not in skip:
        g.extend(oom_nccl(parsed))
    if code_overlay is not None and 'timer_consistency' not in skip:   # NM15 P1: a code edit could have faked the timer
        g.extend(timer_consistency(parsed))
    g.extend(reports(parsed))
    return g


# ---------------------------------------------------------------- locked-pair guards (Q2)
def locked_self_mismatch(locked_steps: list, own_steps: list, gid: str = 'locked_self_mismatch') -> GuardResult:
    """The locked candidate must also pass the Q3 gates (the frozen file, never a supplied one) against its own screen."""
    from . import judge as J
    gates = J.load_gates()
    g = GuardResult()
    if not locked_steps or not own_steps:
        return g.reject(gid, 'per-update scalars of the locked run or of its own screen missing')
    res = J.compare(locked_steps, own_steps, gates)
    g.report(gid, numerics_class=res.get('class'), violations=(res.get('violations') or [])[:4])
    if res.get('class') not in J.CLASSES:
        g.reject(gid, 'locked candidate vs its own screen: ' + '; '.join((res.get('violations') or ['FAIL'])[:3]))
    return g


def _reference_identity_problems(receipt, *, arm: str, candidate_id: str, point: dict) -> list[str]:
    from ...harness import signing
    if not isinstance(receipt, dict):
        return ['no receipt']
    p = []
    if not signing.verify(receipt):
        p.append('harness signature missing or invalid')
    if receipt.get('schema') != 'ext-receipt-v1' or receipt.get('env') != 'flashnext':
        p.append('not a flashnext ext receipt')
    if receipt.get('valid') is not True or receipt.get('judge_valid') is not True:
        p.append('not a valid screen')
    ident = {k: receipt.get(k) for k in ('task_id', 'arm', 'purpose', 'candidate_id')}
    if ident != {'task_id': 'ENV-F1', 'arm': arm, 'purpose': 'reference', 'candidate_id': candidate_id}:
        p.append(f'not the {arm} reference screen ({ident})')
    rp = receipt.get('point')
    if not isinstance(rp, dict) or S.validate(rp)[0] != point:
        p.append(f'its point is not the {arm} point')
    nodes = receipt.get('nodes')
    if not (isinstance(nodes, list) and len(nodes) == 4 and len(set(nodes)) == 4 and all(isinstance(n, str) for n in nodes)):
        p.append(f'node set {nodes!r} is not four distinct nodes')
    return p


def uc_reference_problems(receipt) -> list[str]:
    """The U+C reference screen: signed, valid, task ENV-F1 / arm UC-REF / purpose reference / candidate uc-ref, the
    S_F1 start point, four recorded nodes (the window's node set)."""
    return _reference_identity_problems(receipt, arm='UC-REF', candidate_id='uc-ref', point=S.start_point())


def ht_reference_problems(receipt) -> list[str]:
    """F-HT: the same identity checks for arm F-HT / candidate f-ht and the registered S_match point."""
    return _reference_identity_problems(receipt, arm='F-HT', candidate_id='f-ht', point=S.registered_points()['S_match'])


def void_pair(uc_half_receipt: dict | None, gid: str = 'void_pair') -> GuardResult:
    """The locked U+C half must pass the gates against the reference screen (its receipt valid), else the pair is void."""
    g = GuardResult()
    if not isinstance(uc_half_receipt, dict) or uc_half_receipt.get('valid') is not True:
        why = (uc_half_receipt or {}).get('invalid_reasons') if isinstance(uc_half_receipt, dict) else ['no receipt']
        g.reject(gid, 'locked U+C half invalid against the reference screen: ' + '; '.join(map(str, (why or [])[:3])))
    return g
