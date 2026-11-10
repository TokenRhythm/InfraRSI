"""C_F1: the code-edit space of F1 setting 2 (arm F-A0-CODE: the S_F1 switches plus code edits).

A candidate of the code arm is an S_F1 point plus an overlay {'code/<name>': complete new file text} over MUTABLE, the
speed-path modules of the 86-file variant runtime that every bundle carries as code/<name> (family_worker.py:130-133:
the worker asserts every code file against the attempt's code_sha256 before it puts code/ first on sys.path). Frozen
and hash-checked exactly as for the switch-only arms: every other runtime file, the trainer and its train_step (sha
4ab14ced; the materializer asserts the train_step AST unchanged), the source tree, data, loss, optimizer and its state,
LR schedule, precision, seed and the layout (asserted by the trainer).

Where an overlay goes (FlashNextRunner): apply() writes a per-attempt copy of the local variant runtime with the overlay
files replaced (never the deployed FN_SRC runtime, never the local _variant_runtime); variant.build_bundle builds the
bundle from that copy, so its code/ directory - uploaded to FN_ROOT/metadata/<name>/code on the shared /data and
hash-verified, the directory the job runs from - carries the overlay; the worker re-checks every file on the node, every
rank echoes the inventory, and guards.frozen_hashes accepts exactly the manifest inventory with the overlay's files
replaced (every other file must still be the frozen one).

    validate(files, base=None) -> (errors, warnings)   the one validator (FlashNextHooks.validate_patch and the runner)
    apply(runtime_dir, files, dst) -> record          the per-attempt runtime copy (raises on any mismatch)
    record(files) -> dict                             the receipt block: overlay digest, per-file sha256, changed lines
    expected_inventory(files) -> {name: sha256}       the bundle inventory the overlay must produce
    overlay_digest(file_hashes)                       = SessionHooks.overlay_digest of the same file set (trace = receipt)

Integrity rules on an edited file, counted against its base version (the base's own uses stay allowed): no new import of
process / file / clock / lifecycle / optimizer modules, no new clock, file, process or environment access, no new
reference to the optimizer, LR, data schedule or seed, no new string that names the judge's records, and no new
assignment into a library module (torch, triton, time, os, sys, ...). Python only; at most MAX_PATCH_LINES changed lines
over all files; an edit cannot add a file (a new module next to run_precision_screen.py would shadow the source tree).
"""
from __future__ import annotations

import ast
import collections
import functools
import hashlib
import shutil
import tempfile
from pathlib import Path

from ...contracts import digest
from ..common.guards import path_problem
from ..common.session_hooks import MAX_FILE_BYTES, diff_lines
from . import variant as V

SPACE_ID = 'C_F1'
SCHEMA = 'flashnext-code-overlay-v1'
PREFIX = 'code/'
MAX_PATCH_LINES = 300            # ENV-A's patch limit (rebench_llmf.patch_guard.MAX_DIFF_LINES) = the strong policy's
# The speed-path runtime modules a code edit may replace (name -> one neutral sentence). Not mutable, on purpose: the
# variant registry and switch binding (systems_policy, family_policy, contracts), the lifecycle (family_*, phase_gate,
# run_precision_screen), the collectors, the optimizer side (master pack, shared master guard, moment storage, the
# systems / U+C / W runtime controllers that own optimizer-state offload and parameter sync), checkpoint and HF delivery,
# and the five files whose frozen hashes other runtime files assert (replay_cache.py by replay_training.py; the W
# kernels flash_qsa_fastmask_v1.py, flash_host_sync_memo_v1.py and flash_combine_segsum_v2.py by w_family_runtime.py).
MUTABLE = {
    'flash_qsa_fastmask_v2.py': 'QSA block-sparse attention kernels (Triton forward, dQ and dK/dV) with the packed '
                                'selection mask; the attention implementation of the systems_policy stack',
    'flash_qsa_tuned_launch_v1.py': 'Triton launch configuration of the three QSA kernels (reads the qsa_*_launch '
                                    'switches)',
    'flash_qsa_tuned_launch_v2.py': 'QSA launch path with one tile map per call and no host synchronisation',
    'qsa_fastmask_training.py': 'installs the QSA implementation after a first-call oracle on real training inputs',
    'speed50_segsum_kernel.py': 'fused FP32 local combine kernel of the DeepEP MoE combine forward (Triton)',
    'segsum_training.py': 'installs the segmented FP32 MoE combine after a first-input oracle on real routes',
    'replay_training.py': 'replay of the MoE dispatch / combine results in the recompute (backward) pass',
    'prefix_retention_training.py': 'keeps the attention-prefix activations of the 11 QSA layers from the forward pass '
                                    'instead of recomputing them',
    'speed50_module_recompute.py': 'the activation-checkpoint owner of the retained prefix / expert boundary',
    'speed50_prefix_stages.py': 'splits decoder layers into checkpoint stages without merging their autograd contexts',
    'speed50_block_padding.py': 'block-recompute padding forwarding of the bridge transformer block',
    'qkv_subgroup_training.py': 'QKV projection all-gather / reduce-scatter over one KV group\'s tensor-parallel ranks',
    'uc_native_fc1.py': 'native Transformer Engine fc1 backward of the native_rollbacks stack',
}
MUTABLE_PATHS = tuple(PREFIX + n for n in MUTABLE)
# The production_qsa_memo_segsum stack asserts the frozen hashes of these at startup (w_family_contract.SOURCE_SHA256,
# w_family_runtime.py:105-117): a changed version runs with the other stacks only.
W_PINNED = ('flash_qsa_tuned_launch_v1.py', 'flash_qsa_tuned_launch_v2.py')
# NM24 P2: a numeric-literal edit to a HOST statement is launch / kernel tuning and is allowed only in the modules
# below - they hold Triton kernels and launch parameters (block sizes, num_warps / num_stages) and no correctness
# gate. Every other mutable module (first-input oracles, shadow / witness / replay gates, checkpoint controllers)
# has its host statements frozen to the base apart from line endings; only its @triton.jit kernel bodies, if any,
# may change. This stops a numeric-only edit from hollowing out a gate (zero-length shadow, zeroed upstream, a
# disabled gate_active) while the kernel body is rewritten.
NUMERIC_TUNING_FILES = {
    'flash_qsa_fastmask_v2.py': 'the three QSA Triton kernels (fwd / dQ / dKdV) and their packed-mask launch; no gate',
    'flash_qsa_tuned_launch_v1.py': 'Triton launch configuration (block sizes, num_warps / num_stages) of the QSA kernels',
    'flash_qsa_tuned_launch_v2.py': 'the QSA launch path (one tile map per call); launch parameters only',
    'speed50_segsum_kernel.py': 'the fused FP32 segsum Triton kernel and its launch wrapper',
}
# Read-only context for read_file: the runtime files the mutable modules are installed by or call (the deployed variant
# copies), and the source-tree files they patch or run inside (from the local mirror of the WINH source tree).
READ_ONLY_RUNTIME = ('systems_policy.py', 'systems_transform.py', 'systems_runtime.py', 'family_transform.py',
                     'family_runtime.py', 'q_hook_adapters.py', 'boundary_ast.py', 'boundary_contract.py',
                     'native_full_accounting.py', 'replay_cache.py', 'coverage.py', 'route_digest.py', 'allocator_policy.py',
                     'uc_runtime.py', 'uc_transform.py', 'w_family_runtime.py', 'w_family_contract.py',
                     'flash_qsa_fastmask_v1.py', 'flash_host_sync_memo_v1.py', 'flash_combine_segsum_v2.py')
SOURCE_ROOT = '/data/zehua_flashnext_replay_20261013/source_tree/qwen-sft'
_BRIDGE = 'runtime/flash-mcore-ad-20260908/mcore-bridge-d87452898c8b8f850669ad168fe6166cfa1c460f/src/mcore_bridge/model'
READ_ONLY_SOURCE = ('runs/qwen38_flash_mcore_ad_20260908_v1/train_flash_strategy_20260925.py',
                    'runs/qwen38_flash_mcore_ad_20260908_v1/flash_mcore_deepep_moe_adapter_v6.py',
                    'runs/qwen38_flash_mcore_ad_20260908_v1/flash_mcore_deepep_moe_install_v6.py',
                    f'{_BRIDGE}/modules/kernels/qsa_block_sparse_attn.py', f'{_BRIDGE}/modules/kernels/qsa_kernels.py',
                    f'{_BRIDGE}/modules/gated_delta_net.py', f'{_BRIDGE}/modules/transformer_block.py',
                    f'{_BRIDGE}/modules/transformer_layer.py', f'{_BRIDGE}/gpts/qwen3_next.py')
SOURCE_KEY = 'qwen-sft/'          # read_file name of a source-tree file: qwen-sft/<path below SOURCE_ROOT>

# ---------------------------------------------------------------- integrity rules (counted against the base version)
BANNED_IMPORTS = (
    'subprocess', 'socket', 'ctypes', 'multiprocessing', 'signal', 'http', 'urllib', 'requests', 'shutil', 'importlib',
    'builtins', 'pickle', 'marshal', 'tempfile', 'glob', 'time', 'datetime', 'random', 'sys', 'os', 'io', 'pathlib',
    'torch.optim', 'megatron.core.optimizer', 'transformer_engine.pytorch.optimizers', 'apex',
    # the trainer's training-semantics modules (data, schedule, optimizer management) and the lifecycle / judge side
    'flash_night_adam_management_20260924', 'flash_strategy_recipe_20260925', 'flash_pool_exploration_adapter_20261006',
    'flash_quarter_runtime_20261002', 'flash_strategy_dev_20260925', 'family_worker', 'family_runtime',
    'family_transform', 'family_policy', 'family_retention', 'family_terminal', 'phase_gate', 'collect_family',
    'collect_uc', 'collect_w_family', 'systems_collection', 'qo_collection', 'training_hf_runtime',
    'training_hf_evidence', 'training_hf_contract', 'master_pack_training', 'master_integration',
    'master_instance_adapter', 'master_contract', 'shared_master_guard', 'scaled_resident', 'scaled_resident_training',
    'encoded_checkpoint_sidecar', 'resident_state_probe', 'recovery_fork', 'nccl_startup',
    # the trainer itself, the module that writes the per-update timer (update_seconds), the precision screen and the
    # running script (interim review 2026-10-28: an edit could otherwise patch the metric's writer)
    'train_flash_strategy_20260925', 'flash_strategy_signals_20260925', 'run_precision_screen', '__main__')
BANNED_NAMES = frozenset({
    # clocks: the metric is the trainer's own per-update timer
    'perf_counter', 'perf_counter_ns', 'monotonic', 'monotonic_ns', 'time_ns', 'process_time', 'process_time_ns',
    'thread_time', 'thread_time_ns', 'clock_gettime', 'clock_gettime_ns',
    # processes, files, environment
    'system', 'popen', 'Popen', 'fork', 'forkpty', 'kill', 'killpg', 'putenv', 'unsetenv', 'environ', 'chdir', 'unlink',
    'rmdir', 'rmtree', 'rename', 'renames', 'makedirs', 'mkdir', 'write_text', 'write_bytes', 'symlink', 'chmod', 'chown',
    'open', '__import__', '__builtins__',
    # training semantics an edit must not reach: optimizer and its state, LR, data schedule, seed
    'optimizer', 'param_groups', 'zero_grad', 'clip_grad_norm_', 'clip_grad_value_', 'lr_scheduler', 'base_lr',
    'learning_rate', 'MasterStorage', 'SecondMomentOffload', 'OwnedParamSync', 'owned_sync', 'load_schedule',
    'data_entry', 'manual_seed', 'seed'})
BUILTIN_ONLY = frozenset({'exec', 'eval', 'compile', 'globals'})     # as names only (torch.compile stays allowed)
# Reflection / dynamic-code primitives (NM15 P1): an edit cannot ADD them over the file's base count. The base
# speed-path installers legitimately use some of these (getattr/exec/compile/... recompile and patch kernels at
# install), so a flat ban is impossible; the delta-against-base model lets the base's own uses stand while refusing a
# new one. Reached as a bare name or as an attribute (builtins.getattr); counted as ('name', <token>) like BANNED_NAMES.
REFLECT_NAMES = frozenset({'getattr', 'setattr', 'delattr', 'vars', 'globals', 'eval', 'exec', 'compile',
                           '__import__', '__build_class__', 'breakpoint'})
# Dunder attributes that reach interpreter internals (obj.__globals__, type(x).__subclasses__, ...): the path from any
# object to __builtins__ / __import__ and back to banned modules. Counted as ('dunder', <attr>) when read as an
# attribute; a subscript key like obj['__globals__'] is a literal and is caught by DUNDER_STRING below instead.
DUNDER_DANGEROUS = frozenset({'__globals__', '__builtins__', '__dict__', '__class__', '__subclasses__', '__code__',
                              '__closure__', '__getattribute__', '__bases__', '__mro__', '__base__', '__self__',
                              '__func__', '__wrapped__', '__loader__', '__spec__', '__init_subclass__'})
# Any new string literal that contains '__' (counted as ('dunder_string', '__')): a dunder name built or indexed as a
# string, including the split trick '__' + 'import__'. Base literals with '__' stay allowed (counted against the base).
DUNDER_STRING = '__'
BANNED_STRINGS = ('signals', 'update_seconds', '.jsonl', 'exit.json', 'pid.json', 'release.json', 'attempt.json',
                  'job.json', 'fn_spec', 'STOP.json', 'INFRARSI', 'LP_CACHE_ROOT', 'code_sha256', 'bundle_code',
                  'infra_family', 'materializ', 'FLASH_', '/proc/', 'perf_counter', 'environ', '__import__',
                  'optimizer', 'learning_rate', 'base_lr')
LIBRARY_ROOTS = frozenset({'torch', 'triton', 'time', 'os', 'sys', 'builtins', 'megatron', 'transformer_engine',
                           'numpy', 'gc', 'threading', 'datetime', 'random', 'io', 'pathlib', 'importlib', 'json',
                           'signal', 'socket', 'subprocess', 'shutil', 'ctypes'})
# ---------------------------------------------------------------- kernel-and-constants code space (NM21 round 4)
# After three rounds of escapes from a static allowlist over arbitrary host Python (operator reflection, inherited os,
# torch.distributed.FileStore, triton.runtime.cache, scope shadowing, module laundering), the boundary itself changes:
# host Python is FROZEN. Compared statement-by-statement with the base (matched by position and name, ast.dump after
# normalizing line endings), the edited file may differ only by (a) the VALUES of numeric literals in any statement,
# (b) the bodies of @triton.jit kernels (rule B), and (c) NEW top-level statements that are themselves a @triton.jit
# kernel or `NAME = <number>`. A removed / reordered statement, any other new or changed statement, and any import
# change are refused. Rationale: Triton's code generator can call a host Python callable at compile time when its
# arguments are constexpr, and the launch (host) code is frozen except numeric literals, so a kernel body that reaches
# only triton.language, its own values and numeric constants cannot pull a host object into the compile.
KERNEL_BUILTINS = frozenset({'range', 'min', 'max', 'int', 'float', 'bool', 'abs'})
# triton.language public surface a kernel body may call as tl.<name>. Read from the installed triton when present (a GPU
# node), else this pinned set (the CPU validator has no triton). tl.math.* and tl.extra.cuda.libdevice.* are allowed as
# whole chains besides these. The set is deliberately generous (every op a speed edit might reach) - it only gates the
# tl.<name> surface, never host access, which rule B blocks structurally.
TRITON_LANG_PINNED = frozenset({
    'arange', 'zeros', 'zeros_like', 'full', 'full_like', 'cat', 'broadcast', 'broadcast_to', 'expand_dims', 'ravel',
    'reshape', 'view', 'trans', 'permute', 'split', 'join', 'interleave', 'flip', 'where', 'swizzle2d', 'program_id',
    'num_programs', 'load', 'store', 'make_block_ptr', 'advance', 'dot', 'dot_scaled', 'exp', 'exp2', 'log', 'log2',
    'sqrt', 'rsqrt', 'sin', 'cos', 'abs', 'erf', 'floor', 'ceil', 'fma', 'sigmoid', 'softmax', 'umulhi', 'fdiv',
    'div_rn', 'cdiv', 'minimum', 'maximum', 'clamp', 'max', 'min', 'sum', 'prod', 'cumsum', 'cumprod', 'xor_sum',
    'reduce', 'associative_scan', 'sort', 'argmax', 'argmin', 'histogram', 'gather', 'atomic_add', 'atomic_max',
    'atomic_min', 'atomic_and', 'atomic_or', 'atomic_xor', 'atomic_cas', 'atomic_xchg', 'int1', 'int8', 'int16',
    'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float8e4nv', 'float8e4b8', 'float8e5', 'float8e5b16',
    'float8e4b15', 'float16', 'bfloat16', 'float32', 'float64', 'void', 'pi32_t', 'constexpr', 'dtype', 'pointer_type',
    'block_type', 'function_type', 'tensor', 'range', 'static_range', 'static_assert', 'static_print', 'device_assert',
    'device_print', 'debug_barrier', 'multiple_of', 'max_contiguous', 'max_constancy', 'inline_asm_elementwise',
    'num_threads', 'cast', 'to', 'math', 'extra', 'core'})


RULES = ('Python only: a complete new version of a mutable file; no new file; at most '
         f'{MAX_PATCH_LINES} changed lines (added + removed) over all files versus their base versions',
         'counted against the file\'s base version (its own uses stay allowed), an edit adds no import of process, file, '
         'clock, lifecycle, data-schedule or optimizer modules; no clock, file, process or environment access; no '
         'reference to the optimizer, its state, the LR, the data schedule or the seed; no string that names the '
         'judge\'s records (signals, timing, attempt / job files, FLASH_* variables); no assignment into a library '
         'module (torch, triton, time, os, sys, ...)',
         'counted against the file\'s base version, an edit adds no Python reflection or dynamic-code primitive '
         '(getattr / setattr / delattr / vars / globals / eval / exec / compile / __import__), no new access to an '
         'interpreter-internal dunder attribute (__globals__, __builtins__, __dict__, __class__, __subclasses__, '
         '__code__, __closure__, ...) and no new string literal containing "__" (a dunder name built or indexed as a '
         'string): these are the ways an edit could reach a banned module or object without naming it',
         'what is computed stays the same: the per-update main CE, MTP CE, aux, grad norm and LR are judged against the '
         'start configuration\'s screen (E0 bitwise or E1); a change that fails to import or run costs its screen',
         'host-side Python is frozen: compared with the base file statement by statement, the edited file may differ '
         'only in the VALUES of numeric literals and in the bodies of @triton.jit kernels; a new top-level statement '
         'may only be a @triton.jit kernel or NAME = <number>; a removed, reordered or otherwise changed statement, and '
         'any import change, are refused. A @triton.jit body may reference only its parameters and locals, '
         'tl.<triton.language> / tl.math.* / tl.extra.cuda.libdevice.*, this file\'s kernels, module-level numeric '
         'constants and range / min / max / int / float / bool / abs; a body local may not take the name of a host '
         'module-level binding (an import, a module function / class or variable), so it cannot launder a host object '
         'into the compile-time surface. A NEW kernel\'s decorators are @triton.jit and '
         'optionally @triton.autotune with plain data only (triton.Config of numbers / strings / bools, a key list of '
         'strings): no @triton.heuristics, no lambda or other callable (pre_hook, post_hook, prune_configs_by, '
         'do_bench); its defaults, *args / **kwargs / return annotations and PEP 695 type parameters are evaluated '
         'in module scope at definition time, so they may use only literals, tl.<name> chains and module-level '
         'numeric constants (no parameter name, call, lambda, comprehension or walrus)',
         'a numeric-literal edit to a host statement is allowed only in the launch / kernel tuning files; in every '
         'other mutable file the host statements stay as in the base apart from line endings (only their '
         '@triton.jit kernel bodies may change), so a numeric edit cannot hollow out a correctness gate, oracle or '
         'shadow witness',
         'a new top-level NAME = <number> must use a fresh name: it may not rebind any name the base file already '
         'binds (an assignment, def, class, import alias or module-level loop target)',
         'the production_qsa_memo_segsum stack asserts the frozen hashes of flash_qsa_tuned_launch_v1.py and '
         'flash_qsa_tuned_launch_v2.py at startup: with either changed, only the other stacks run')


def file_sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def name_of(path: str) -> str:
    return path[len(PREFIX):] if isinstance(path, str) and path.startswith(PREFIX) else path


# ---------------------------------------------------------------- base versions (the frozen variant runtime)
def base_inventory() -> dict:
    """{name: sha256} of the 86-file variant runtime (frozen_manifest.json), the inventory every bundle carries."""
    from . import guards as G
    m = G.load_manifest()
    inv = ((m or {}).get('variant_runtime') or {}).get('inventory')
    if not isinstance(inv, dict) or len(inv) != 86:
        raise RuntimeError(f'{G.MANIFEST_PATH} missing or without the 86-file variant runtime inventory')
    missing = sorted(set(MUTABLE) - set(inv))
    if missing:
        raise RuntimeError(f'mutable files not in the variant runtime: {missing}')
    return dict(inv)


@functools.lru_cache(maxsize=1)
def _base_texts() -> tuple:
    inv = base_inventory()
    out = []
    for name in MUTABLE:
        raw = (V.FROZEN_RUNTIME / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != inv[name]:
            raise RuntimeError(f'{V.FROZEN_RUNTIME / name} is not the variant runtime file (frozen_manifest.json)')
        out.append((PREFIX + name, raw.decode('utf-8')))          # bytes as they are (CRLF files stay CRLF)
    return tuple(out)


def base_files() -> dict:
    """{'code/<name>': text} of every mutable file, as the variant runtime carries it (hash-checked)."""
    return dict(_base_texts())


@functools.lru_cache(maxsize=1)
def _read_only_texts() -> tuple:
    from . import deploy as D
    out = []
    with tempfile.TemporaryDirectory(prefix='fncs') as tmp:
        rt = D.custom_runtime_local(Path(tmp) / 'rt')          # the deployed variant copies (registry edits included)
        if rt['inventory'] != base_inventory():
            raise RuntimeError('the variant runtime does not match frozen_manifest.json')
        for name in READ_ONLY_RUNTIME:
            out.append((PREFIX + name, (Path(rt['runtime']) / name).read_bytes().decode('utf-8')))
    for rel in READ_ONLY_SOURCE:
        p = V.SOURCE_MIRROR / rel
        if p.is_file():                                         # the local mirror of the WINH source tree
            out.append((SOURCE_KEY + rel, p.read_bytes().decode('utf-8', 'replace')))
    return tuple(out)


def read_only_files() -> dict:
    return dict(_read_only_texts())


# ---------------------------------------------------------------- digests and records
def overlay_digest(file_hashes: dict) -> str:
    """Digest of the full mutable file set {'code/<name>': sha256} - SessionHooks.overlay_digest over the same files."""
    return digest(sorted([p, h] for p, h in file_hashes.items()))


def full_hashes(files: dict | None) -> dict:
    """{'code/<name>': sha256} of every mutable file of a candidate: the overlay's text, else the base version's."""
    inv = base_inventory()
    files = files or {}
    return {p: (file_sha(files[p]) if p in files else inv[name_of(p)]) for p in MUTABLE_PATHS}


def expected_inventory(files: dict | None) -> dict:
    """The bundle's code inventory with this overlay: the manifest inventory with exactly the overlay's files replaced."""
    inv = base_inventory()
    for p, t in (files or {}).items():
        inv[name_of(p)] = file_sha(t)
    return inv


def record(files: dict | None) -> dict | None:
    """The receipt / trace / locked block of an overlay (None for none): its digest (= the session's candidate overlay
    digest), per changed file its sha256 and its base sha256, changed lines, and the bundle inventory hash."""
    if not files:
        return None
    base, inv = base_files(), base_inventory()
    return {'schema': SCHEMA, 'space_id': SPACE_ID, 'space_sha256': sha256(), 'digest': overlay_digest(full_hashes(files)),
            'files': {p: file_sha(t) for p, t in sorted(files.items())},
            'base': {p: inv[name_of(p)] for p in sorted(files)},
            'changed_lines': sum(diff_lines(base.get(p, ''), t) for p, t in files.items()),
            'inventory_sha256': hashlib.sha256(V.encode(expected_inventory(files))).hexdigest()}


def loose_record(files) -> dict | None:
    """What a refused overlay carried (paths and the sha256 of each text), without any validation."""
    if not files:
        return None
    if not isinstance(files, dict):
        return {'schema': SCHEMA, 'malformed': type(files).__name__}
    return {'schema': SCHEMA, 'refused': True,
            'files': {str(p): (file_sha(t) if isinstance(t, str) else None) for p, t in sorted(files.items(), key=str)}}


def describe() -> dict:
    """The code space as it goes into the task spec (and its hash into every code-arm receipt)."""
    inv = base_inventory()
    return {'space_id': SPACE_ID, 'prefix': PREFIX, 'max_patch_lines': MAX_PATCH_LINES,
            'mutable': {PREFIX + n: inv[n] for n in MUTABLE}, 'w_pinned': list(W_PINNED),
            'rules': {'banned_imports': list(BANNED_IMPORTS), 'banned_names': sorted(BANNED_NAMES),
                      'builtin_only': sorted(BUILTIN_ONLY), 'banned_strings': list(BANNED_STRINGS),
                      'library_roots': sorted(LIBRARY_ROOTS), 'reflection': sorted(REFLECT_NAMES),
                      'dunder_attributes': sorted(DUNDER_DANGEROUS),
                      'dunder_strings': 'no new string literal containing "__" (counted against the base version)',
                      'code_space': {'boundary': 'kernel-and-constants (NM21 round 4)',
                                     'host_python': 'frozen: in the launch / kernel tuning files a top-level statement '
                                                    'may differ from the base only in numeric literal values; in every '
                                                    'other mutable file (gates, oracles, controllers) host statements '
                                                    'stay identical to the base (NM24 P2)',
                                     'numeric_tuning_files': sorted(NUMERIC_TUNING_FILES),
                                     'signature_expressions': 'defaults and annotations of a new or changed kernel '
                                                              'are evaluated in the module scope: only literals, '
                                                              'tl.<name> chains and module-level numeric constants; '
                                                              'no type parameters (NM24 P1)',
                                     'new_decorators': 'a new kernel: @triton.jit and optionally @triton.autotune '
                                                       'with plain data only; no heuristics, no callables (NM22)',
                                     'kernel_bodies': 'only the bodies of @triton.jit kernels may change',
                                     'new_statements': 'a new top-level statement may only be a @triton.jit kernel '
                                                       'or NAME = <number> with a name the base does not bind '
                                                       '(NM24 P3)',
                                     'kernel_surface': 'a kernel body may reference only its params / locals, '
                                                       'tl.<triton.language>, tl.math.*, tl.extra.cuda.libdevice.*, '
                                                       'this file\'s @triton.jit kernels, module-level numeric '
                                                       'constants and range / min / max / int / float / bool / abs',
                                     'body_shadow': 'a body local may not reuse the name of a host module-level binding '
                                                    '(import / module function / variable), so a `name: tl.constexpr = '
                                                    'name` cannot launder a host object into the compile (NM25 P1)',
                                     'kernel_builtins': sorted(KERNEL_BUILTINS),
                                     'rule': 'host Python is frozen; Triton can call a host callable at compile time '
                                             'for constexpr args, so kernel bodies must not name host objects '
                                             '(NM21 P1-1/2/3)'}}}


def sha256() -> str:
    return digest(describe())


def task_view(read_only: dict | None = None) -> dict:
    """The code-space block of the code arm's task view (file names, one sentence each, rules; no recipe names)."""
    base = base_files()
    return {'space_id': SPACE_ID,
            'mutable_files': [{'path': PREFIX + n, 'lines': len(base[PREFIX + n].splitlines()), 'description': d}
                              for n, d in MUTABLE.items()],
            'max_patch_lines': MAX_PATCH_LINES,
            'read_only_files': sorted(read_only or {}),
            'source_tree': f'{SOURCE_KEY}<path> = {SOURCE_ROOT}/<path> on WINH (read-only, frozen)',
            'frozen': 'every other runtime file, the trainer and its train_step, the source tree, data and its order, '
                      'loss, optimizer and optimizer state, LR schedule, precision, seed, layout',
            'rules': list(RULES)}


# ---------------------------------------------------------------- validation
def _import_hits(module: str) -> list[str]:
    return [b for b in BANNED_IMPORTS if module == b or module.startswith(b + '.')]


def _docstring_ids(tree) -> set:
    """ids of string statements (docstrings and bare string expressions): no effect, not counted as strings."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            out.add(id(node.value))
    return out


def _root(node):
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def surface(tree) -> collections.Counter:
    """The integrity-surface uses of one parsed file: Counter of (kind, token)."""
    c = collections.Counter()
    libs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                for hit in _import_hits(a.name):
                    c[('import', hit)] += 1
                if a.name.split('.')[0] in LIBRARY_ROOTS:
                    libs.add(a.asname or a.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for hit in _import_hits(node.module):
                c[('import', hit)] += 1
            if node.module.split('.')[0] in LIBRARY_ROOTS:
                libs.update(a.asname or a.name for a in node.names)
    strings = _docstring_ids(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id in BANNED_NAMES or node.id in BUILTIN_ONLY or node.id in REFLECT_NAMES:
                c[('name', node.id)] += 1
        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_NAMES or node.attr in REFLECT_NAMES:
                c[('name', node.attr)] += 1
            elif node.attr in DUNDER_DANGEROUS:
                c[('dunder', node.attr)] += 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in strings:
            for frag in BANNED_STRINGS:
                c[('string', frag)] += node.value.count(frag)
            if DUNDER_STRING in node.value:                 # a dunder name built or indexed as a string literal
                c[('dunder_string', DUNDER_STRING)] += node.value.count(DUNDER_STRING)
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = node.targets
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('setattr', 'delattr') \
                and node.args:
            root = _root(node.args[0]) if not isinstance(node.args[0], ast.Name) else node.args[0].id
            if root in libs:
                c[('library_patch', root)] += 1
        for t in targets:
            for el in (t.elts if isinstance(t, (ast.Tuple, ast.List)) else [t]):
                if isinstance(el, (ast.Attribute, ast.Subscript)) and _root(el) in libs:
                    c[('library_patch', _root(el))] += 1
    return +c


_KIND = {'import': 'import of {}', 'name': 'use of {}', 'string': 'string naming {!r}',
         'library_patch': 'assignment into the library module {}', 'dunder': 'access to the dunder attribute {}',
         'dunder_string': 'a string literal that can build a dunder name (contains {!r})'}


def integrity_problems(path: str, base_text: str | None, text: str) -> list[str]:
    """Uses of the integrity surface the edited file adds over its base version (empty = none added)."""
    try:
        new = surface(ast.parse(text, filename=path))
    except SyntaxError as exc:
        return [f'{path}: syntax error line {exc.lineno}: {exc.msg}']
    try:
        old = surface(ast.parse(base_text or '', filename=path))
    except SyntaxError:
        old = collections.Counter()
    out = []
    for (kind, token), n in sorted(new.items()):
        if n > old.get((kind, token), 0):
            out.append(f'{path}: adds {_KIND[kind].format(token)} ({n} uses, the base version has '
                       f'{old.get((kind, token), 0)}): code edits may not reach timing, files, processes, the '
                       'environment, the optimizer / LR / data / seed, library internals, or Python reflection and '
                       'dynamic code (getattr / eval / __import__, dunder attributes, dunder-building strings)')
    return out


def _top_segments(text: str):
    """({source segment -> top-level node}, tree); ({}, None) on a syntax error (integrity_problems reports it)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}, None
    segs = {}
    for i, node in enumerate(tree.body):
        seg = ast.get_source_segment(text, node)
        segs[seg if seg is not None else f'<stmt {i}:{node.lineno}>'] = node
    return segs, tree


def _bound_names(node) -> set:
    """Every name bound anywhere in the subtree: a Store/Del target, a def/class name, an import alias, a function or
    lambda argument, an except binder. A conservative union over nested scopes, so a legitimate closure variable is
    never seen as free; an edit cannot bind a dangerous callable without a primitive the allowlist already refuses."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split('.')[0])
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
    return out


@functools.lru_cache(maxsize=1)
def triton_lang_names() -> frozenset:
    """triton.language public names (installed triton's __all__/dir if present on a GPU node, else the pinned set)."""
    names = set(TRITON_LANG_PINNED)
    try:
        import triton.language as _tl                 # absent on the CPU validator; present on a WINH node
        names |= set(getattr(_tl, '__all__', ()) or ())
        names |= {n for n in dir(_tl) if not n.startswith('_')}
    except Exception:                                 # noqa: BLE001 - no triton here: the pinned set stands
        pass
    return frozenset(names)


def _numeric_norm(node) -> str:
    """ast.dump of a copy of `node` with every int / float literal value set to 0 (bool / str / complex untouched), so
    two statements that differ only in numeric constant VALUES compare equal (a block-size / num_warps / num_stages
    change)."""
    import copy
    dup = copy.deepcopy(node)
    for n in ast.walk(dup):
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            n.value = 0
    return ast.dump(dup)


def _sig_norm(fn) -> str:
    """_numeric_norm of a @triton.jit function with its body blanked: decorators + signature, numeric-insensitive. Equal
    base vs edit means only the body (and numeric literals) changed - the one host change rule A allows for a kernel."""
    import copy
    dup = copy.deepcopy(fn)
    dup.body = [ast.Pass()]
    for n in ast.walk(dup):
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            n.value = 0
    return ast.dump(dup)


def _stmt_key(node):
    """A kind/name key used to match a statement to the base by position and name."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return ('def', node.name)
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return ('import', ast.dump(node))
    if isinstance(node, ast.Assign):
        names = tuple(sorted(t.id for t in node.targets if isinstance(t, ast.Name)))
        return ('assign', names or ast.dump(node))
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        t = node.target
        return ('assign', (t.id,) if isinstance(t, ast.Name) else ast.dump(node))
    return ('other', ast.dump(node))


def _module_aliases(tree, module: str) -> set:
    """Module-level import aliases bound to exactly `module` (imports are frozen, so this is the same for base/edit)."""
    out = set()
    for n in tree.body:
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name == module:
                    out.add(a.asname or a.name.split('.')[0])
    return out


def _triton_decorators_ok(node, triton_aliases) -> bool:
    """True iff `node` is a @triton.jit kernel: decorated with <triton>.jit, optionally with <triton>.autotune(...) /
    <triton>.heuristics(...) stacked, and nothing else."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    jit = False
    for d in node.decorator_list:
        f = d.func if isinstance(d, ast.Call) else d
        if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in triton_aliases):
            return False
        if f.attr == 'jit':
            jit = True
        elif f.attr not in ('autotune', 'heuristics'):
            return False
    return jit


def _safe_decorator_value(node, triton_aliases) -> bool:
    """A decorator argument of a NEW @triton.jit kernel is evaluated at module import / compile time, so it must be a
    pure data expression: a numeric / string (no "__") / bool / None literal, a list / tuple / set / dict of safe
    values, or a <triton-alias>.Config(...) whose own arguments are safe. No lambda, no other call, no bare name, no
    attribute chain, no comprehension - the autotune config surface honest tuning needs and nothing a host callable can
    ride in on."""
    if _is_numeric_literal(node):
        return True
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return '__' not in node.value
        return isinstance(node.value, bool) or node.value is None
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_safe_decorator_value(e, triton_aliases) for e in node.elts)
    if isinstance(node, ast.Dict):
        return (all(k is not None and _safe_decorator_value(k, triton_aliases) for k in node.keys)
                and all(_safe_decorator_value(v, triton_aliases) for v in node.values))
    if isinstance(node, ast.Call):
        f = node.func
        if (isinstance(f, ast.Attribute) and f.attr == 'Config'
                and isinstance(f.value, ast.Name) and f.value.id in triton_aliases):
            return (all(_safe_decorator_value(a, triton_aliases) for a in node.args)
                    and all(_safe_decorator_value(k.value, triton_aliases) for k in node.keywords))
        return False
    return False


def _decorator_problems(path: str, fn, triton_aliases) -> list[str]:
    """A NEW @triton.jit kernel's decorator argument expressions run host Python at module import (@triton.autotune
    configs/keys) or at compile time (@triton.heuristics callables), so they are outside a kernel body yet still
    host-executable. @triton.heuristics is refused outright on a new kernel (its values map is a dict of host
    callables); every @triton.jit / @triton.autotune argument must be a safe decorator value (no lambda, call, name or
    attribute chain). Existing kernels are not checked here: rule A already pins their decorators to numeric-literal
    changes through _sig_norm."""
    out = []
    kname = getattr(fn, 'name', '<kernel>')
    for d in fn.decorator_list:
        head = d.func if isinstance(d, ast.Call) else d
        attr = head.attr if isinstance(head, ast.Attribute) else None
        if attr == 'heuristics':
            out.append(f'{path}: @triton.jit {kname} may not use @triton.heuristics on a new kernel (its callables run '
                       'host Python at compile time)')
            continue
        if not isinstance(d, ast.Call):
            continue
        for a in d.args:
            if not _safe_decorator_value(a, triton_aliases):
                out.append(f'{path}: @triton.jit {kname} has a decorator argument that is not a numeric / string '
                           'literal or a triton.Config(...) of them (no lambda, call, name or attribute chain)')
        for kw in d.keywords:
            if not _safe_decorator_value(kw.value, triton_aliases):
                out.append(f'{path}: @triton.jit {kname} has a decorator keyword {kw.arg or "**"!r} that is not a '
                           'numeric / string literal or a triton.Config(...) of them (no lambda, call, name or '
                           'attribute chain)')
    return out


def _is_numeric_literal(v) -> bool:
    return (isinstance(v, ast.Constant) and type(v.value) in (int, float)) or (
        isinstance(v, ast.UnaryOp) and isinstance(v.op, (ast.UAdd, ast.USub))
        and isinstance(v.operand, ast.Constant) and type(v.operand.value) in (int, float))


def _is_numeric_const_assign(node) -> bool:
    return (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name) and _is_numeric_literal(node.value))


def _numeric_const_names(tree) -> set:
    """Module-level names bound ONLY to numeric literals (a kernel may reference these as compile-time constants)."""
    pure = {}
    for n in tree.body:
        if _is_numeric_const_assign(n):
            pure.setdefault(n.targets[0].id, True)
        elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    pure[t.id] = False                 # bound to something non-numeric: not a pure numeric constant
    return {k for k, v in pure.items() if v}


def _attr_chain(node):
    """(innermost non-Attribute node, [attribute names, outermost last]) for an Attribute (or any node -> ([], ...))."""
    attrs, cur = [], node
    while isinstance(cur, ast.Attribute):
        attrs.append(cur.attr)
        cur = cur.value
    attrs.reverse()
    return cur, attrs


def _ok_tl_chain(chain, names) -> bool:
    """A tl-rooted attribute chain is allowed iff it is tl.<name in triton.language>, tl.math.*, or
    tl.extra.cuda.libdevice.* ."""
    if not chain:
        return False
    if chain[0] == 'math':
        return True
    if chain[:3] == ['extra', 'cuda', 'libdevice'] and len(chain) >= 4:
        return True
    return len(chain) == 1 and chain[0] in names


class _ModuleBindings(ast.NodeVisitor):
    """Every name bound in MODULE scope: a Store / Del target (recursively through tuple / list / starred unpacking),
    a def / class name, an import alias, an `except ... as`, and `match` capture names - collected through module-level
    compound statements (if / for / while / with / try / match, NM29 P2), but not into a function / class / lambda body,
    which is a separate scope. A def / class / lambda's own decorators, default arguments and annotations ARE module
    scope, so they are still visited (only the body is skipped)."""

    def __init__(self) -> None:
        self.names: set = set()

    def _skip_body(self, node) -> None:
        for field, value in ast.iter_fields(node):
            if field == 'body':
                continue                                   # a separate scope; its bindings are not module names
            for child in (value if isinstance(value, list) else [value]):
                if isinstance(child, ast.AST):
                    self.visit(child)

    def visit_FunctionDef(self, n):
        self.names.add(n.name)
        self._skip_body(n)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, n):
        self.names.add(n.name)
        self._skip_body(n)

    def visit_Lambda(self, n):
        self._skip_body(n)

    def visit_Name(self, n):
        if isinstance(n.ctx, (ast.Store, ast.Del)):
            self.names.add(n.id)

    def visit_Import(self, n):
        for a in n.names:
            self.names.add((a.asname or a.name).split('.')[0])

    def visit_ImportFrom(self, n):
        for a in n.names:
            self.names.add(a.asname or a.name)

    def visit_ExceptHandler(self, n):
        if n.name:
            self.names.add(n.name)
        self.generic_visit(n)

    def visit_MatchAs(self, n):
        if n.name:
            self.names.add(n.name)
        self.generic_visit(n)

    def visit_MatchStar(self, n):
        if n.name:
            self.names.add(n.name)

    def visit_MatchMapping(self, n):
        if n.rest:
            self.names.add(n.rest)
        self.generic_visit(n)


def _module_bound_names(tree) -> set:
    """Module-scope names the file binds: assignment / ann / aug targets (recursive through nested tuple / list /
    starred unpacking), def / class names, import aliases, for / with / except / match binders, and the same inside
    module-level if / for / while / with / try / match bodies - not inside a function / class / lambda body (NM29 P2).
    A new ``NAME = <number>`` may not rebind one of these (NM24 P3), and a kernel body may not shadow one (NM29 P1)."""
    v = _ModuleBindings()
    for n in tree.body:
        v.visit(n)
    return v.names


def _safe_sig_expr(node, tl_aliases, tl_names, numeric_consts) -> bool:
    """A @triton.jit signature expression (a default / annotation / return), evaluated in the MODULE scope when the
    kernel is defined, is safe iff it is built only from numeric / string (no "__") / bool / None literals, lists /
    tuples / sets / dicts of safe values, tl.<name> / tl.math.* / tl.extra.cuda.libdevice.* attribute chains, and
    module-level numeric constants. A bare name (parameter OR global), a call, a lambda, a comprehension, a walrus and
    a starred element are all refused - a parameter or body local does not exist yet when the signature runs."""
    if _is_numeric_literal(node):
        return True
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return '__' not in node.value
        return isinstance(node.value, bool) or node.value is None
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_safe_sig_expr(e, tl_aliases, tl_names, numeric_consts) for e in node.elts)
    if isinstance(node, ast.Dict):
        return (all(k is not None and _safe_sig_expr(k, tl_aliases, tl_names, numeric_consts) for k in node.keys)
                and all(_safe_sig_expr(v, tl_aliases, tl_names, numeric_consts) for v in node.values))
    if isinstance(node, ast.Attribute):
        root, chain = _attr_chain(node)
        return isinstance(root, ast.Name) and root.id in tl_aliases and _ok_tl_chain(chain, tl_names)
    if isinstance(node, ast.Name):
        return node.id in numeric_consts
    return False


def _signature_problems(path: str, fn, tl_aliases, numeric_consts) -> list[str]:
    """NM24 P1: every signature expression of a NEW or CHANGED @triton.jit kernel - positional / keyword-only defaults,
    all parameter annotations including *args / **kwargs, and the return annotation - is evaluated in the module (host)
    scope when the kernel is defined, so it is checked here against the host-safe surface (literals, tl.<name>, module
    numeric constants), NOT with the body's parameter / local namespace. PEP 695 type parameters are refused."""
    tl_names = triton_lang_names()
    kname = getattr(fn, 'name', '<kernel>')
    out = []
    if getattr(fn, 'type_params', None):
        out.append(f'{path}: @triton.jit {kname} may not declare type parameters (PEP 695)')
    a = fn.args
    exprs = [d for d in a.defaults] + [d for d in (a.kw_defaults or []) if d is not None]
    exprs += [p.annotation for p in a.posonlyargs + a.args + a.kwonlyargs if p.annotation]
    if a.vararg and a.vararg.annotation:
        exprs.append(a.vararg.annotation)
    if a.kwarg and a.kwarg.annotation:
        exprs.append(a.kwarg.annotation)
    if fn.returns is not None:
        exprs.append(fn.returns)
    for e in exprs:
        if not _safe_sig_expr(e, tl_aliases, tl_names, numeric_consts):
            out.append(f'{path}: @triton.jit {kname} has a signature expression (a default / annotation / return) that '
                       'is not a literal, a tl.<name> chain or a module-level numeric constant; a signature runs in '
                       'module scope at definition time, so parameters and body locals do not count here')
    return out


def _kernel_body_problems(path: str, fn, tl_aliases, kernel_fns, numeric_consts, host_names=frozenset()) -> list[str]:
    """Rule B: a @triton.jit body (new or changed) may reference only its parameters and locals, the tl alias(es) as
    tl.<name>/tl.math.*/tl.extra.cuda.libdevice.*, @triton.jit functions of this file, module-level numeric constants,
    and the builtins range/min/max/int/float/bool/abs. Method / attribute access on a kernel value is allowed (e.g.
    x.to(tl.int64), x.dtype) but never a _-prefixed or dunder attribute; no import, no global/nonlocal, no nested
    def/lambda/class, no string literal containing '__'. Host names, host modules and host method chains cannot be
    named, so the compiler cannot reach a host callable."""
    tl_names = triton_lang_names()
    allowed = set(KERNEL_BUILTINS) | set(tl_aliases) | set(kernel_fns) | set(numeric_consts)
    args = fn.args
    for a in args.posonlyargs + args.args + args.kwonlyargs:
        allowed.add(a.arg)
    if args.vararg:
        allowed.add(args.vararg.arg)
    if args.kwarg:
        allowed.add(args.kwarg.arg)
    checked = list(fn.body)      # NM24 P1: the BODY only - signature expressions run in module scope and are
    #                              checked by _signature_problems, which does NOT see parameters or body locals
    # NM25 P1: a body-local store may not re-enable a HOST module-level name (an import alias, a module def / class or a
    # module variable - `host_names`). Triton evaluates `name: tl.constexpr = <host>` at compile time and then calls
    # through the constexpr value, so a body local shadowing torch / os / ... would launder the host object past the
    # allowlist (NM24's `torch: tl.constexpr = torch` then `torch.save(...)`). Such a store is left out of `allowed`, so
    # the shadowing name's loads and attribute reaches are refused as host reaches. File kernels and module-level numeric
    # constants are legitimately referenceable (excluded from host_names by the caller), so an honest local is unaffected.
    for s in checked:
        for x in ast.walk(s):
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store) and x.id not in host_names:
                allowed.add(x.id)
    inner = set()
    for s in checked:
        for x in ast.walk(s):
            if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Attribute):
                inner.add(id(x.value))
    out = []
    kname = getattr(fn, 'name', '<kernel>')
    for s in checked:
        for x in ast.walk(s):
            if isinstance(x, (ast.Import, ast.ImportFrom)):
                out.append(f'{path}: @triton.jit {kname} may not import')
            elif isinstance(x, (ast.Global, ast.Nonlocal)):
                out.append(f'{path}: @triton.jit {kname} may not use global / nonlocal')
            elif isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                out.append(f'{path}: @triton.jit {kname} may not define a nested function / lambda / class')
            elif isinstance(x, ast.Constant) and isinstance(x.value, str) and '__' in x.value:
                out.append(f'{path}: @triton.jit {kname} may not use a string containing "__"')
            elif isinstance(x, ast.Attribute):
                if id(x) in inner:
                    continue                           # an inner link of a chain; the outermost node carries the check
                root, chain = _attr_chain(x)
                if isinstance(root, ast.Name) and root.id in tl_aliases:
                    if not _ok_tl_chain(chain, tl_names):
                        out.append(f'{path}: @triton.jit {kname} reaches {root.id}.{".".join(chain)} (only tl.<name in '
                                   'triton.language>, tl.math.* or tl.extra.cuda.libdevice.* are allowed)')
                elif isinstance(root, ast.Name) and root.id in (set(kernel_fns) | set(numeric_consts)):
                    out.append(f'{path}: @triton.jit {kname} reads an attribute of {root.id!r} (a kernel / numeric '
                               'constant has no attribute surface)')
                elif isinstance(root, ast.Name) and root.id not in allowed:
                    out.append(f'{path}: @triton.jit {kname} reaches {root.id}.{".".join(chain)} (host names are not '
                               'visible to a kernel body)')
                elif any(att.startswith('_') for att in chain):
                    out.append(f'{path}: @triton.jit {kname} reaches a _-prefixed / dunder attribute ".{chain[-1]}"')
            elif isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load):
                if x.id not in allowed:
                    out.append(f'{path}: @triton.jit {kname} uses the name {x.id!r} (a kernel body may name only its '
                               'parameters and locals, the tl alias, this file\'s kernels, module-level numeric '
                               'constants, and range / min / max / int / float / bool / abs)')
    return out


FROZEN_FILE_MSG = ('this file holds correctness gates / controllers, so its host statements must stay exactly as in '
                   'the base, numeric values included (numeric edits are allowed only in the launch / kernel tuning '
                   'files: ' + ', '.join(sorted(NUMERIC_TUNING_FILES)) + '); only @triton.jit kernel bodies may change')
KERNEL_SPACE_MSG = ('host-side Python must stay as in the base apart from numeric constant values; only the bodies of '
                    '@triton.jit kernels may change, and a new top-level statement may only be a @triton.jit kernel or '
                    'NAME = <number>')


def kernel_space_problems(path: str, base_text: str | None, text: str) -> list[str]:
    """Rule A (NM21 round 4). Match the edited file's top-level statements to the base's by position and name; accept a
    statement that is identical, identical except numeric literal values, or - if a @triton.jit kernel with the same
    numeric-insensitive signature and decorators - a body change (rule B); accept a NEW statement only when it is a
    @triton.jit kernel (rule B) or NAME = <number>; refuse a removed / reordered statement, any other change, and any
    import change. Line endings are normalized first so an ending-only flip is not a change."""
    base_text = (base_text or '').replace('\r\n', '\n').replace('\r', '\n')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    try:
        base_tree, new_tree = ast.parse(base_text), ast.parse(text)
    except SyntaxError:
        return []                                      # a syntax error is reported by integrity_problems
    triton_aliases = _module_aliases(base_tree, 'triton')
    tl_aliases = _module_aliases(base_tree, 'triton.language')
    kernel_fns = {n.name for n in new_tree.body if _triton_decorators_ok(n, triton_aliases)}
    numeric_consts = _numeric_const_names(new_tree)
    is_tuning = name_of(path) in NUMERIC_TUNING_FILES          # NM24 P2: numeric host edits only in tuning files
    base_bound = _module_bound_names(base_tree)                # NM24 P3: names a new constant may not rebind
    # NM25 P1: the module-level host names a kernel body may not re-enable by shadowing them with a body local. A file
    # kernel or a module-level numeric constant is legitimately referenceable, so it is not a host name.
    host_names = (_module_bound_names(new_tree) | base_bound) - set(kernel_fns) - set(numeric_consts)
    base_body, new_body = base_tree.body, new_tree.body
    base_keys = [_stmt_key(s) for s in base_body]
    out, bi = [], 0
    for s in new_body:
        k = _stmt_key(s)
        if bi < len(base_body) and base_keys[bi] == k:
            b = base_body[bi]
            bi += 1
            if ast.dump(b) == ast.dump(s):
                continue                               # structurally identical (formatting / comments / endings)
            if is_tuning and _numeric_norm(b) == _numeric_norm(s):
                continue                               # a tuning file: identical except numeric literal VALUES
            if (not is_tuning and _numeric_norm(b) == _numeric_norm(s)
                    and not (_triton_decorators_ok(b, triton_aliases) and _triton_decorators_ok(s, triton_aliases))):
                out.append(f'{path}: top-level {k[1]!r} changes a numeric value: {FROZEN_FILE_MSG}')
                continue
            if (_triton_decorators_ok(b, triton_aliases) and _triton_decorators_ok(s, triton_aliases)
                    and _sig_norm(b) == _sig_norm(s)):
                out += _signature_problems(path, s, tl_aliases, numeric_consts)
                out += _kernel_body_problems(path, s, tl_aliases, kernel_fns, numeric_consts, host_names)
                continue
            out.append(f'{path}: top-level {k[1]!r} changed: {KERNEL_SPACE_MSG}')
        elif k in base_keys[bi:]:
            out.append(f'{path}: top-level {k[1]!r} is reordered, or a base statement before it was removed: '
                       f'{KERNEL_SPACE_MSG}')
        elif _triton_decorators_ok(s, triton_aliases):
            out += _decorator_problems(path, s, triton_aliases)
            out += _signature_problems(path, s, tl_aliases, numeric_consts)
            out += _kernel_body_problems(path, s, tl_aliases, kernel_fns, numeric_consts, host_names)
        elif _is_numeric_const_assign(s):
            if s.targets[0].id in base_bound:          # NM24 P3: a new constant may not rebind a base name
                out.append(f'{path}: new top-level {s.targets[0].id!r} = <number> rebinds a name the base file '
                           f'already defines (use a fresh constant name): {KERNEL_SPACE_MSG}')
            continue
        else:
            out.append(f'{path}: new top-level statement {k[1]!r} is not a @triton.jit kernel or NAME = <number>: '
                       f'{KERNEL_SPACE_MSG}')
    if bi != len(base_body):
        removed = [base_keys[i][1] for i in range(bi, len(base_body))]
        out.append(f'{path}: base top-level statement(s) {removed[:4]} were removed: {KERNEL_SPACE_MSG}')
    seen, uniq = set(), []
    for e in out:
        if e not in seen:
            seen.add(e)
            uniq.append(e)
    return uniq


def validate(files, base: dict | None = None) -> tuple[list, list]:
    """(errors, warnings) of a code-arm overlay {'code/<name>': complete new text}; errors = invalid_proposal, no node.
    base: {'code/<name>': base text} (default: the variant runtime's)."""
    if type(files) is not dict or not files:
        return ['a code edit is a non-empty object {"code/<name>": complete new file text}'], []
    bad = [p for p, t in files.items() if type(p) is not str or type(t) is not str]
    if bad:
        return [f'non-string path or text: {[str(p) for p in bad][:4]}'], []
    paths = [f'{p!r}: {why}' for p in sorted(files) for why in [path_problem(p)] if why]
    if paths:
        return [f'non-canonical paths: {"; ".join(paths[:4])}'], []
    base = base_files() if base is None else base
    errors, warnings = [], []
    outside = sorted(p for p in files if p not in MUTABLE_PATHS)
    if outside:
        errors.append(f'not mutable: {outside[:6]} (mutable: {list(MUTABLE_PATHS)}; an edit cannot add a file)')
    total, changed = 0, False
    for path, text in sorted(files.items()):
        if path not in MUTABLE_PATHS:
            continue
        try:
            size = len(text.encode('utf-8'))
        except UnicodeEncodeError as exc:
            errors.append(f'{path}: not encodable as UTF-8 ({exc.reason})')
            continue
        if size > MAX_FILE_BYTES:
            errors.append(f'{path}: larger than {MAX_FILE_BYTES} bytes')
            continue
        if '\x00' in text:
            errors.append(f'{path}: NUL character')
            continue
        old = base.get(path)
        total += diff_lines(old or '', text)
        changed |= text != old
        errors += integrity_problems(path, old, text)
        errors += kernel_space_problems(path, old, text)
        if old is not None and text != old:
            if ('\r\n' in old) != ('\r\n' in text):
                warnings.append(f'{path}: line endings differ from the base version (it runs either way)')
            if name_of(path) in W_PINNED:
                warnings.append(f'{path}: the production_qsa_memo_segsum stack asserts this file\'s frozen hash at '
                                'startup; with this change only the other stacks run')
    if total > MAX_PATCH_LINES:
        errors.append(f'diff {total} lines > limit {MAX_PATCH_LINES}')
    if not changed and not outside:
        errors.append('patch identical to the base files')
    return errors, warnings


def failure_tail(local, max_chars: int = 1500, max_lines: int = 14) -> dict | None:
    """The Python error a failed code-arm screen ended with, from its collected node logs (nodes/node<r>.log, their last
    256 KiB): the first rank traceback of a node that did not stop for a peer (exit code other than 0 / 125), else of
    any node; its last lines with the bundle's code/ paths shortened. None when no traceback was collected."""
    import json
    import re
    nodes = Path(local) / 'nodes'
    order = []
    for r in range(4):
        try:
            code = json.loads((nodes / f'node{r}.exit.json').read_text(encoding='utf-8')).get('exit_code')
        except (OSError, ValueError, AttributeError):
            code = None
        order.append((0 if code not in (0, 125) else 1, r))
    prefix = re.compile(r'^\[rank(\d+)\]:\s?')
    meta = re.compile(r'/data/infrarsi/ext/runs/flashnext/metadata/[^/\s"]+/code/')
    for _, r in sorted(order):
        try:
            lines = (nodes / f'node{r}.log').read_text(encoding='utf-8', errors='replace').splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines):
            m = prefix.match(line)
            body = line[m.end():] if m else line
            if not body.startswith('Traceback (most recent call last):'):
                continue
            rank = m.group(1) if m else None
            block = [body]
            for nxt in lines[i + 1:i + 400]:
                m2 = prefix.match(nxt)
                if rank is not None and (not m2 or m2.group(1) != rank):
                    break                                       # another rank's output: this traceback is complete
                text = nxt[m2.end():] if (rank is not None and m2) else nxt
                block.append(meta.sub('code/', text))
                if rank is None and text and not text[0].isspace() and not text.startswith(_TB_HEADS):
                    break                                       # unprefixed output: the exception line ends it
            ends = [j for j, x in enumerate(block) if x and not x[0].isspace() and not x.startswith(_TB_HEADS)]
            if ends:
                block = block[:ends[-1] + 1]                    # up to the (last) exception line
            tail = '\n'.join(block[-max_lines:])
            return {'node': r, 'rank': int(rank) if rank is not None else None, 'text': tail[-max_chars:]}
    return None


_TB_HEADS = ('Traceback (most recent call last):', 'During handling of the above exception',
             'The above exception was the direct cause')


def is_code_task(task) -> bool:
    """A task of the code arm: action space knobs+patch over exactly this code space."""
    return (getattr(task, 'action_space', 'knobs') == 'knobs+patch'
            and tuple(getattr(task, 'mutable_globs', ()) or ()) == MUTABLE_PATHS
            and int(getattr(task, 'max_patch_lines', 0) or 0) == MAX_PATCH_LINES)


# ---------------------------------------------------------------- the per-attempt runtime copy
def apply(runtime_dir, files: dict, dst) -> dict:
    """Copy the variant runtime `runtime_dir` (must be the frozen_manifest.json inventory) to the empty or new directory
    `dst` and write the overlay files there (exact UTF-8 bytes of each text); the copy must then be exactly
    expected_inventory(files). Returns record(files) + {'runtime': dst, 'changed_files'}. Never writes runtime_dir."""
    errors, _ = validate(files)
    if errors:
        raise ValueError('code overlay refused: ' + '; '.join(errors[:4]))
    src, dst = Path(runtime_dir), Path(dst)
    inv = base_inventory()
    if V.inventory(src) != inv:
        raise RuntimeError(f'{src}: not the frozen variant runtime (frozen_manifest.json)')
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f'{dst} is not empty')
    if dst.resolve() == src.resolve():
        raise ValueError('the per-attempt copy cannot be the runtime itself')
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.glob('*.py')):
        shutil.copyfile(p, dst / p.name)
    for path, text in sorted(files.items()):
        (dst / name_of(path)).write_bytes(text.encode('utf-8'))
    got, want = V.inventory(dst), expected_inventory(files)
    if got != want:
        bad = sorted(n for n in set(got) | set(want) if got.get(n) != want.get(n))
        raise RuntimeError(f'per-attempt runtime copy is not the overlay inventory: {bad[:4]}')
    if V.inventory(src) != inv:                                     # the source stays the frozen runtime
        raise RuntimeError(f'{src} changed while the per-attempt copy was made')
    return {**record(files), 'runtime': str(dst), 'changed_files': sorted(n for n in got if got[n] != inv[n])}
