"""Launch bundles for the flagship production path from a knob dict (X5 E1 spike), on this PC, CPU only.

The frozen formal-54 builder (`build_unified_bundle.py`, CODE_FREEZE d6a8f3b6...) takes a variant *name* and looks its
recipe up in `systems_policy.POLICIES` and the family contracts. `build_bundle(knobs, ...)` re-implements its fresh-build
path (resume 0) with the recipe taken from the knobs instead: the systems policy row is built field by field
(`knobs.systems_policy`), the job env from `knobs.job_env`. For a registered recipe the result must be byte-identical to
what the frozen builder makes (and, after the controller's bind step, to the formal-54 bundle on disk).

A systems row no recipe registered needs the frozen runtime to know its name: `custom_runtime(dst)` copies the 86 frozen
files and inserts the custom names into the nine variant-name tuples and the POLICIES table (exact-count text edits; the
registered rows and every other byte stay as they are). Bundles of custom rows carry that copy's inventory instead of the
frozen one; registered recipes always use the frozen runtime.

The build itself runs in a fresh interpreter (`python variant.py build <spec.json>`): the frozen modules are top-level
names (`family_policy`, `systems_policy`, ...) and the frozen and custom runtime copies must never meet in one process.
`materialize_local` runs the frozen worker's own `--materialize-only --local-source-root` path against the local LF mirror
of the source tree (no torch, no GPU): validate_launch, the systems/Q configure() asserts and every source transform.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
FROZEN_DIR = REPO / 'reports/rsi_infra_execution_20261018/code/uc_native_sync_ast_fix_v1'
FROZEN_RUNTIME = FROZEN_DIR / 'runtime'
LP = REPO / 'reports/low_precision_winh_20261014'
SOURCE_MIRROR = REPO / 'deliverables/WINH_ACCEL_CODEBASE_20261016/baseline/source_tree/qwen-sft'
REGISTERED_BUNDLES = Path(os.environ.get('FLASHNEXT_REGISTERED_BUNDLES', r'E:\rsi\f54v1\final\bundles'))
CODE_FREEZE_SHA256 = 'd6a8f3b6b6c95e9013ffedcc58ddd62fb37708056ad75f013a72bcaf0c3fc826'   # formal-54 SCHEDULE.json
ROOT = '/data/low_precision_winh_20261014'                   # the formal-54 root (byte-identity builds only)
FORMAL_SHELL = ROOT + '/deployment/flash_mcore_env_portable.sh'          # read-only use; sha pinned in frozen_manifest
# ext layout on WINH (EXT_INTERFACES §6: every write under /data/infrarsi)
FN_ROOT = '/data/infrarsi/ext/runs/flashnext'              # metadata/<name>/ (bundle) and runs/<name>/ (outputs)
FN_SRC = '/data/infrarsi/ext/src/flashnext_variant'        # custom runtime copy + node scripts
FN_LOCKS = '/data/infrarsi/ext/locks/flashnext'
FN_CACHE = '/data/infrarsi/ext/cache/flashnext'            # copy of the warm compile cache (LP_CACHE_ROOT)
FN_PHASE = FN_ROOT + '/phase_gate'
FORMAL_RUNS_ASSERT = "'/data/low_precision_winh_20261014/runs'"          # family_worker.py:40, the one screen-time pin
DEV = [96, 192, 256, 384, 512, 768, 1024]
# Every tuple of variant names that lists the native systems rows contains the literal 'S_match_full' (file: count).
NAME_TUPLES = {'collect_family.py': 1, 'family_policy.py': 1, 'family_runtime.py': 2, 'family_transform.py': 2,
               'family_worker.py': 3}
POLICY_ANCHOR = '_POLICY = None\n'
# Keys the wave controller's bind step adds (run_experiment_wave_v2.py:323-333).
BIND_PHASE_KEYS = {'schema', 'root', 'wave', 'group', 'members', 'synchronized_start', 'timeout_seconds'}


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def encode(x) -> bytes:                       # prepare_precision_jobs.encode
    return (json.dumps(x, sort_keys=True, indent=2) + '\n').encode()


def inventory(runtime) -> dict:
    return {p.name: sha(p) for p in sorted(Path(runtime).glob('*.py'))}


# ---------------------------------------------------------------- frozen inputs
def frozen_check() -> dict:
    """The frozen builder's own preconditions (build_unified_bundle.py:52-57), plus the CODE_FREEZE identity."""
    problems = []
    freeze_path = FROZEN_DIR / 'CODE_FREEZE.json'
    if sha(freeze_path) != CODE_FREEZE_SHA256:
        problems.append('CODE_FREEZE.json is not the formal-54 freeze')
    freeze = read(freeze_path)
    if inventory(FROZEN_RUNTIME) != freeze['common_code_sha256']:
        problems.append('frozen runtime inventory drifted')
    if sha(FROZEN_DIR / 'build_unified_bundle.py') != freeze['builder_sha256']:
        problems.append('frozen builder drifted')
    if {n: sha(REPO / n) for n in freeze['builder_input_sha256']} != freeze['builder_input_sha256']:
        problems.append('pinned builder inputs drifted')
    train = SOURCE_MIRROR / 'runs/qwen38_flash_mcore_ad_20260908_v1/train_flash_strategy_20260925.py'
    if not train.exists() or sha(train) != '4ab14ced3243dfe0398fbbd0630eb3ec758e59102541b7d186bad6af52b8d79f':
        problems.append('local source mirror missing or not the formal-54 trainer')
    return {'ok': not problems, 'problems': problems, 'code_freeze_sha256': CODE_FREEZE_SHA256}


# ---------------------------------------------------------------- custom runtime (registry extension only)
def _custom_rows():
    from . import knobs
    return knobs.custom_rows()


def custom_runtime(dst, rows=None, *, root: str | None = None) -> dict:
    """Copy the frozen runtime to `dst` and register `rows` (default: every custom E1 systems row) as variants.

    Edits, each asserted by exact count: the name 'S_match_full' in the nine name tuples gains the custom names after it;
    systems_policy.py gains one loop that adds the custom rows to POLICIES with the frozen loop's own formula
    (systems_policy.py:17-23). root (the ext run root): family_worker.py:40's run-root assert names it instead of the
    formal root, the only write-root pin a 30-step screen executes (contracts.py:85 is HF delivery, recovery_fork.py:13
    is recovery forks; neither runs in a screen). Nothing else changes."""
    rows = _custom_rows() if rows is None else rows
    names = [r['name'] for r in rows]
    dst = Path(dst)
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f'{dst} is not empty')
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(FROZEN_RUNTIME.glob('*.py')):
        shutil.copyfile(p, dst / p.name)
    inserted = ",".join(repr(n) for n in names)
    edits = {}
    for name, count in NAME_TUPLES.items():
        text = (dst / name).read_text(encoding='utf-8')
        found = text.count("'S_match_full'")
        if found != count:
            raise AssertionError(f'{name}: {found} name tuples, expected {count}')
        (dst / name).write_text(text.replace("'S_match_full'", "'S_match_full'," + inserted), encoding='utf-8')
        edits[name] = count
    text = (dst / 'systems_policy.py').read_text(encoding='utf-8')
    if text.count(POLICY_ANCHOR) != 1:
        raise AssertionError('systems_policy.py: policy anchor not unique')
    table = ',\n    '.join(f"({r['name']!r}, {'PREFIX' if r['policy']['retained_attention_layers'] else '()'}, "
                           f"{tuple(r['policy']['retained_expert_layers'])!r}, {r['policy']['qkv_subgroup']}, "
                           f"{r['policy']['shared_master_guard']})" for r in rows)
    block = ('# ext/flashnext custom rows (X5 E1 spike): native systems rows no recipe registered, same formula as above.\n'
             f'for name, prefix, experts, qkv, master in (\n    {table},\n):\n'
             '    POLICIES[name] = dict(variant=name, retained_attention_layers=list(prefix),\n'
             '        retained_expert_layers=list(experts), replay_enabled=True,\n'
             '        qsa_fastmask_v2=True, segsum=True, qkv_subgroup=qkv,\n'
             '        shared_master_guard=master, moment_policy=NATIVE,\n'
             "        recompute=['full', 'block', 0] if prefix else ['full', 'uniform', 1],\n"
             '        extra_gc_freeze=False, old_host_memo=False)\n\n')
    (dst / 'systems_policy.py').write_text(text.replace(POLICY_ANCHOR, block + POLICY_ANCHOR), encoding='utf-8')
    edits['systems_policy.py'] = len(rows)
    if root is not None:
        text = (dst / 'family_worker.py').read_text(encoding='utf-8')
        if text.count(FORMAL_RUNS_ASSERT) != 1:
            raise AssertionError('family_worker.py: run-root assert not unique')
        (dst / 'family_worker.py').write_text(text.replace(FORMAL_RUNS_ASSERT, repr(root.rstrip('/') + '/runs')),
                                              encoding='utf-8')
        edits['family_worker.py:run_root'] = 1
    frozen = inventory(FROZEN_RUNTIME)
    inv = inventory(dst)
    changed = sorted(n for n in inv if inv[n] != frozen.get(n))
    return {'runtime': str(dst), 'names': names, 'edits': edits, 'changed_files': changed, 'root': root,
            'inventory': inv, 'inventory_sha256': hashlib.sha256(encode(inv)).hexdigest()}


# ---------------------------------------------------------------- build (driver side)
def bundle_spec(knobs_dict, *, group, port, output, run_limit=30, seed=42, ids=None, runtime_dir=None,
                force_runtime=False, root: str | None = None) -> dict:
    """force_runtime (verification only): build even a registered recipe on `runtime_dir`, to show the custom runtime
    leaves registered recipes unchanged. root: the WINH root of the bundle's metadata/ and runs/ (default: the formal
    root, for byte-identity checks); any other root needs a custom runtime built with that root."""
    from . import knobs
    c, errors, warnings = knobs.canonical(knobs_dict)
    if errors:
        raise ValueError('; '.join(errors))
    name = knobs.variant_name(c)
    root = (root or ROOT).rstrip('/')
    custom = name.startswith('SX_') or force_runtime or root != ROOT
    if custom and runtime_dir is None:
        raise ValueError(f'{name} needs runtime_dir from custom_runtime()' + (' (re-rooted bundle)' if root != ROOT else ''))
    runtime = Path(runtime_dir) if custom else FROZEN_RUNTIME
    if root != ROOT:
        text = (runtime / 'family_worker.py').read_text(encoding='utf-8')
        if repr(root + '/runs') not in text:
            raise ValueError(f'{runtime}: family_worker.py does not accept the run root {root}/runs (custom_runtime(root=...))')
    return dict(variant=name, family=c['family'], knobs=c, warnings=warnings,
                policy=knobs.systems_policy(c, name) if c['family'] == 'systems' else None,
                env=knobs.job_env(c), group=group, port=int(port), run_limit=int(run_limit), seed=int(seed),
                output=str(Path(output)), runtime=str(runtime), check_frozen=not custom, ids=list(ids or []),
                lp=str(LP), frozen_dir=str(FROZEN_DIR), repo=str(REPO), root=root)


def build_bundle(knobs_dict, *, group, port, output, run_limit=30, seed=42, ids=None, runtime_dir=None,
                 force_runtime=False, root: str | None = None, python=sys.executable, timeout=600) -> dict:
    """Build one bundle under `output`; returns the builder's receipt plus the spec. ids: the uuid4 hex values the
    frozen preparation draws (run-id suffix, then attempt id), to reproduce a known bundle; else fresh random ones."""
    spec = bundle_spec(knobs_dict, group=group, port=port, output=output, run_limit=run_limit, seed=seed, ids=ids,
                       runtime_dir=runtime_dir, force_runtime=force_runtime, root=root)
    Path(output).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='fnspec') as tmp:
        path = Path(tmp) / 'spec.json'
        path.write_text(json.dumps(spec), encoding='utf-8')
        res = subprocess.run([python, '-B', '-X', 'utf8', str(Path(__file__).resolve()), 'build', str(path)],
                             capture_output=True, text=True, encoding='utf-8', timeout=timeout)
    if res.returncode:
        raise RuntimeError(f'bundle build failed ({spec["variant"]}): {res.stderr[-3000:]}')
    return dict(json.loads(res.stdout.strip().splitlines()[-1]), spec=spec)


def ids_of(bundle) -> list[str]:
    """The two uuid4 hex values a registered bundle's preparation drew: run-id suffix (8 of 32 used) and attempt id."""
    name = Path(bundle).name
    attempt = read(Path(bundle) / 'attempt.json')['attempt_id']
    run_suffix = name[:-len(attempt) - 1].rsplit('_', 1)[1]
    return [run_suffix + '0' * 24, attempt[1:]]


# ---------------------------------------------------------------- bind and compare (verification only)
def bind_like(bundle, phase_gate: dict, *, mode: str = 'performance') -> None:
    """The wave controller's bind edits (run_experiment_wave_v2.py:323-335) with a given phase_gate, so a fresh bundle
    can be compared with a bound formal-54 bundle. Never used for a launch."""
    bundle = Path(bundle)
    if set(phase_gate) != BIND_PHASE_KEYS:
        raise ValueError(f'phase_gate keys {sorted(phase_gate)} != controller keys {sorted(BIND_PHASE_KEYS)}')
    attempt, job = read(bundle / 'attempt.json'), read(bundle / 'job.json')
    assert 'phase_gate' not in attempt, 'bind only a fresh unbound bundle'
    attempt['phase_gate'] = copy.deepcopy(phase_gate)
    command = job['command']
    assert command[2].endswith('/flash_import_snapshot_parent_20260928.py')
    assert not any('--max-restarts' in x for x in command)
    command.insert(3, '--max-restarts=0')
    job.update(approved=True, main_table_eligible=False, diagnostic_kind='rsi_' + mode,
               dispatch_policy='One frozen wave; no retry or automatic successor')
    (bundle / 'attempt.json').write_bytes(encode(attempt))
    (bundle / 'job.json').write_bytes(encode(job))


def ext_bind(bundle, *, cap_s: float, phase_root: str = FN_PHASE, cache_root: str = FN_CACHE) -> dict:
    """The launch binding of an ext screen (the controller's bind, run_experiment_wave_v2.py:323-333, for one group):
    a single-group phase gate (phase_gate.py:36-48 accepts members {'A'} with synchronized_start false; its rendezvous
    files live under `phase_root`), --max-restarts=0 before the worker, approved, the screen's wall cap as the
    launcher timeout, and LP_CACHE_ROOT = the ext copy of the compile cache (flash_mcore_env_portable.sh reads
    ${LP_CACHE_ROOT:-...} for the Triton, Inductor and CUDA caches). Returns the phase-gate contract."""
    bundle = Path(bundle)
    attempt, job = read(bundle / 'attempt.json'), read(bundle / 'job.json')
    assert 'phase_gate' not in attempt, 'bind only a fresh unbound bundle'
    cap = int(cap_s)
    assert 60 <= cap <= 86400, cap
    gate = dict(schema='rsi-active-groups-phase-v2', root=phase_root, wave=attempt['run'], group=job['group'],
                members={job['group']: attempt['attempt_id']}, synchronized_start=False, timeout_seconds=cap)
    assert set(gate) == BIND_PHASE_KEYS
    attempt['phase_gate'] = gate
    command = job['command']
    assert command[2].endswith('/flash_import_snapshot_parent_20260928.py')
    assert not any('--max-restarts' in x for x in command)
    command.insert(3, '--max-restarts=0')
    job['env']['LP_CACHE_ROOT'] = cache_root
    job.update(approved=True, main_table_eligible=False, diagnostic_kind='flashnext_screen', timeout_seconds=cap,
               dispatch_policy='ext/flashnext FlashNextRunner: one attempt on four leased nodes; no retry')
    (bundle / 'attempt.json').write_bytes(encode(attempt))
    (bundle / 'job.json').write_bytes(encode(job))
    return gate


def remote_layout(bundle) -> dict:
    """Where a bundle lives and writes on WINH (from its own attempt / job / deployment files)."""
    bundle = Path(bundle)
    attempt, job, dep = read(bundle / 'attempt.json'), read(bundle / 'job.json'), read(bundle / 'deployment.json')
    return {'name': job['name'], 'run': attempt['run'], 'attempt_id': attempt['attempt_id'],
            'metadata': job['metadata_directory'], 'output_root': job['output_root'],
            'lifecycle_run_root': attempt['lifecycle_run_root'], 'job': job['metadata_directory'] + '/job.json',
            'files': [(row['local'], row['remote']) for row in dep['files']],
            'signals': [f"{attempt['lifecycle_run_root']}/{attempt['run']}.signals.rank{r}/signals.jsonl" for r in range(32)],
            'rank_receipts': [f"{attempt['lifecycle_run_root']}/{attempt['run']}.rank{r}.json" for r in range(32)]}


def compare_bundles(ours, registered, *, skip=('materialize_result.json',)) -> dict:
    """Byte comparison of every file. The only rewrite: our local parent directory -> the registered one, as it appears
    JSON-escaped in code_sources, deployment.json and family_prepare_receipt.json."""
    ours, registered = Path(ours), Path(registered)
    a = json.dumps(str(ours.parent.resolve()))[1:-1].encode()
    b = json.dumps(str(registered.parent.resolve()))[1:-1].encode()
    files_o = {p.relative_to(ours).as_posix() for p in ours.rglob('*') if p.is_file()}
    files_r = {p.relative_to(registered).as_posix() for p in registered.rglob('*') if p.is_file()}
    skipped = sorted(f for f in files_r if f in skip)
    only_o, only_r = sorted(files_o - files_r), sorted(files_r - files_o - set(skip))
    differ, same = [], 0
    for f in sorted(files_o & files_r):
        if (ours / f).read_bytes().replace(a, b) == (registered / f).read_bytes():
            same += 1
        else:
            differ.append(f)
    return {'identical': not (differ or only_o or only_r), 'same_files': same, 'differ': differ,
            'only_ours': only_o, 'only_registered': only_r, 'skipped': skipped}


def registered_bundle(entry: str) -> Path:
    d = REGISTERED_BUNDLES / entry
    return next(p for p in d.iterdir() if p.is_dir())


def remote_receipt(bundle) -> dict | None:
    """The PASS_MATERIALIZED_SOURCE receipt the formal-54 node materialization printed (materialize_result.json)."""
    path = Path(bundle) / 'materialize_result.json'
    if not path.exists():
        return None
    rec = None
    for line in read(path)['stdout'].splitlines():
        try:
            x = json.loads(line)
        except ValueError:
            continue
        if isinstance(x, dict) and x.get('status') == 'PASS_MATERIALIZED_SOURCE':
            rec = x
    return rec


# ---------------------------------------------------------------- local CPU materialization
def materialize_local(bundle, out, *, python=sys.executable, mirror=SOURCE_MIRROR, timeout=600) -> dict:
    """The frozen worker's offline path (family_worker.py:115-177): validate_launch, configure() asserts, every source
    transform, the effective trainer written to `out`. Job env applied as the node materialization does
    (materialize_precision_jobs.py:58). Returns {'ok', 'receipt' | 'error', 'effective_sha256'}."""
    bundle, out = Path(bundle), Path(out)
    job = read(bundle / 'job.json')
    env = dict(os.environ, **job['env'])
    env.update(CUDA_VISIBLE_DEVICES='', RANK='0', LOCAL_RANK='0', PYTHONDONTWRITEBYTECODE='1')
    cmd = [python, '-B', '-X', 'utf8', str(bundle / 'code/run_precision_screen.py'), '--attempt', str(bundle / 'attempt.json'),
           '--materialize-only', '--local-source-root', str(mirror), '--materialized-output', str(out)]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding='utf-8', timeout=timeout)
    receipt = None
    for line in res.stdout.splitlines():
        try:
            x = json.loads(line)
        except ValueError:
            continue
        if isinstance(x, dict) and x.get('status', '').startswith('PASS_'):
            receipt = x
    if res.returncode or receipt is None:
        tail = (res.stderr or res.stdout)[-2500:]
        return {'ok': False, 'error': tail, 'returncode': res.returncode}
    effective = out / 'effective_training.py'
    # The worker writes the text in text mode (CRLF on Windows); its effective_sha256 is of the LF text.
    lf = hashlib.sha256(effective.read_bytes().replace(b'\r\n', b'\n')).hexdigest() if effective.exists() else None
    return {'ok': True, 'receipt': receipt, 'effective_sha256': receipt['checks'].get('effective_sha256'),
            'file_sha256_lf': lf}


# ---------------------------------------------------------------- the build itself (runs in a fresh interpreter)
def _build(spec: dict) -> dict:
    """build_unified_bundle.build (fresh path, resume 0) with the recipe from `spec`, not from the variant name."""
    runtime, lp, frozen_dir, repo = (Path(spec[k]) for k in ('runtime', 'lp', 'frozen_dir', 'repo'))
    sys.path.insert(0, str(runtime))
    sys.path.insert(1, str(lp))
    if spec['ids']:
        draws = iter(spec['ids'])
        uuid.uuid4 = lambda: uuid.UUID(hex=next(draws))         # the two draws of prepare_precision_jobs.py:26-28
    from family_policy import VARIANTS, TRAIN_SOURCE_SHA, DATA_ORDER_POLICY, reference_eligible, validate_policy
    import prepare_precision_jobs
    import materialize_precision_jobs
    from prepare_precision_jobs import prepare, encode as enc
    from materialize_precision_jobs import refresh
    from uc_contract import CONTRACT as UC_CONTRACT
    from w_family_contract import CONTRACT as W_CONTRACT, FLAGS as W_FLAGS
    root = spec.get('root') or ROOT
    if root != ROOT:
        # The pinned helpers name the formal root as a module constant (prepare_precision_jobs.py:12,
        # materialize_precision_jobs.py:10); the files stay byte-identical (their pins hold), the constants are re-pointed
        # in this build process only. The source tree and the data stay where they are (read-only).
        prepare_precision_jobs.REMOTE = root
        materialize_precision_jobs.ROOT = root
    variant, family, group = spec['variant'], spec['family'], spec['group']
    run_limit, port, seed = spec['run_limit'], spec['port'], spec['seed']
    assert variant in VARIANTS and variant != 'Q', variant            # Q (lossy, historical reference keys) is not built here
    assert group in ('A', 'B') and 0 < run_limit <= 1024 and seed in (42, 43) and 1024 < port < 65536
    output = Path(spec['output'])
    inv = inventory(runtime)
    freeze = read(frozen_dir / 'CODE_FREEZE.json')
    builder_inputs = {name: sha(repo / name) for name in freeze['builder_input_sha256']}
    assert builder_inputs == freeze['builder_input_sha256'], 'pinned builder input drift'
    if spec['check_frozen']:
        assert inv == freeze['common_code_sha256'], 'frozen common source drift'
    job, _ = prepare(dict(id='bf16', mode='bf16', parts=[], scope='all'), group, run_limit, port, output,
                     approved=False, label='rsi_unified_' + variant.lower(), runtime=lp / 'low_precision_runtime_v6.py')
    local = output / job['name']
    launch, plan = read(local / 'attempt.json'), read(local / 'plan.json')
    launch['lifecycle_run_root'] = root + '/runs/' + job['name'] + '/training'
    launch['best_dev_directory'] = launch['lifecycle_run_root'] + '/best_dev'
    plan.update(keep_checkpoints=1, export_steps=[512, 1024], evaluation_steps=DEV)
    launch.update(infra_family_schema='infra-full-family-v1', infra_variant=variant, run_limit=run_limit,
                  screen_stop=run_limit, selection_steps=DEV, checkpoint_steps=[384], retain_checkpoint_steps=[384],
                  training_seed=seed, training_source_sha256=TRAIN_SOURCE_SHA, data_order_policy=DATA_ORDER_POLICY,
                  unified_family_version='unified_family_v2_fork', builder_input_sha256=builder_inputs,
                  terminal_raw_checkpoint_policy='retain_final_native_after_verified_HF',
                  allocator_policy='allocator-retry-events-zero-oom-v1',
                  plan=plan, plan_sha256=hashlib.sha256(enc(plan)).hexdigest())
    launch['snapshot_policy']['evaluation_steps'] = DEV
    from training_hf_contract import VERSION, CONTRACT as HF_REFERENCE
    from nccl_startup import CONTRACT as STARTUP
    launch.update(delivery_family_version=VERSION, hf_reference_policy=copy.deepcopy(HF_REFERENCE),
                  nccl_startup_policy=copy.deepcopy(STARTUP))
    launch['historical_scalar_reference'] = reference_eligible(launch)
    from recovery_fork import validate_fork
    validate_fork(launch)
    policy = spec['policy']
    if family == 'systems':                     # the row comes from the knobs; it is never looked up by name
        launch['systems_policy'] = copy.deepcopy(policy)
        launch.update(system_optimization='qsa_fastmask_prefix_retention_segsum', replay_mode='candidate',
                      g2_mode='candidate' if policy['retained_expert_layers'] else 'control')
    else:
        assert policy is None
    if family == 'U_plus_C':
        launch['uc_contract'] = copy.deepcopy(UC_CONTRACT)
    if family == 'W':
        launch['w_family_contract'] = copy.deepcopy(W_CONTRACT)
        job['env'].update(W_FLAGS)
    job['env'].update(spec['env'])
    if family == 'systems':
        assert job['env']['FLASH_G2_MODE'] == launch['g2_mode']
    validate_policy(launch)
    for source in runtime.glob('*.py'):
        shutil.copyfile(source, local / 'code' / source.name)
    assert inventory(runtime) == inv
    launch['code_sources'] = {name: str((local / 'code' / name).resolve()) for name in inv}
    job.update(approved=False, infra_variant=variant, training_seed=seed, screen_stop=run_limit,
               unified_family_version='unified_family_v2_fork', delivery_family_version=VERSION,
               diagnostic_kind='unified_full_family_candidate', timeout_seconds=43200 if run_limit == 1024 else 7200,
               lifecycle_run_root=launch['lifecycle_run_root'], attempt_id=launch['attempt_id'],
               collect_only_attempt=dict(attempt_id=launch['attempt_id'], steps=[1, run_limit]),
               dispatch_policy='Root validates actual capacity, source identity and active-group phase before GPU admission')
    for name, value in [('plan.json', plan), ('attempt.json', launch), ('job.json', job)]:
        (local / name).write_bytes(enc(value))
    refresh(local)
    if root != ROOT:
        # The portable shell is used where it is (read-only): it only exports env and execs the source tree's python.
        job = read(local / 'job.json')
        assert job['command'][1] == root + '/deployment/flash_mcore_env_portable.sh'
        job['command'][1] = FORMAL_SHELL
        (local / 'job.json').write_bytes(enc(job))
    assert read(local / 'attempt.json')['code_sha256'] == inv
    result = dict(status='LOCAL_UNIFIED_FAMILY_CANDIDATE_NOT_DISPATCHED', bundle=str(local.resolve()), variant=variant,
                  seed=seed, group=group, resume=0, run_limit=run_limit, approved=False, common_code_sha256=inv,
                  gpu_executed=False)
    (local / 'family_prepare_receipt.json').write_bytes(enc(result))
    return result


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != 'build':
        raise SystemExit('usage: variant.py build <spec.json>')
    out = _build(json.loads(Path(sys.argv[2]).read_text(encoding='utf-8')))
    print(json.dumps({k: out[k] for k in ('status', 'bundle', 'variant', 'group', 'run_limit', 'gpu_executed')}))
