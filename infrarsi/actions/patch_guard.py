"""L1 patch guard: only the six overlay files may change; static AST rules; diff line limits (design 1.8).

`apply(overlay_files)` checks a proposed overlay ({file name: full new text}) against the pristine overlays
and returns a PatchReport. Runtime detectors (invariants, numerics, InterStep, dtype census) remain the final
judge; this guard rejects what is visible statically: imports of frozen modules, timer/receipt symbols,
monkeypatching of timing / precision / optimizer / collective functions, file and env writes, and introspection
that reaches frozen objects.

G19 (hook contract, runtime/overlays/__init__.py): schedule.forward_backward calls run(**kwargs) exactly once,
unconditionally, and returns its result; a replaced forward/step/... is a wrapper defined in the file that calls the
captured original exactly once (directly or through checkpoint), forwards its own inputs unchanged and returns the
result untouched; host_sync.before_update/after_update (outside the update timer) and the helpers they call run no
synchronisation, collective, optimizer step or model compute; data_path.to_device only transfers the batch (no
slicing, indexing, arithmetic, dtype change or writes on values derived from the arrays). G17 also covers RNG
reseeding, mapping mutators on optimizer/model config containers and in-place ops on parameters or batch data.
"""
from __future__ import annotations

import ast
import dataclasses
import difflib
import hashlib
import json
import shutil
from pathlib import Path

from . import zones

LINE_LIMITS = {'strong': 300, 'weak': 80}
MAX_FILE_BYTES = 65536

BANNED_MODULES = {'importlib', 'imp', 'builtins', 'ctypes', 'subprocess', 'socket', 'runpy', 'marshal', 'shutil',
                  'tempfile', 'multiprocessing', 'pickle', 'dill', 'cloudpickle', 'code', 'codeop', 'pty', 'urllib',
                  'http', 'requests', 'paramiko', 'signal', 'atexit', 'sitecustomize', 'site', 'pkgutil', 'inspect',
                  'zipimport', 'types', 'sysconfig', 'faulthandler', 'tracemalloc', 'sqlite3', 'dbm', 'shelve',
                  'zipfile', 'tarfile', 'gzip', 'bz2', 'lzma', 'mmap', 'fcntl', 'resource'}
# Store targets that alter timing, precision, memory statistics, collectives, loss/clip math or the environment, banned
# whatever object they are reached through (aliases, getattr chains, function arguments, sys.modules ...).
SENSITIVE_STORE_ATTRS = {
    'perf_counter', 'perf_counter_ns', 'monotonic', 'monotonic_ns', 'time', 'time_ns', 'process_time', 'process_time_ns',
    'thread_time', 'thread_time_ns', 'sleep', 'synchronize', 'elapsed_time', 'record', 'Event', 'allow_tf32',
    'allow_bf16_reduced_precision_reduction', 'allow_fp16_reduced_precision_reduction', 'deterministic', 'benchmark',
    'max_memory_allocated', 'max_memory_reserved', 'memory_allocated', 'memory_reserved', 'memory_stats', 'memory_summary',
    'reset_peak_memory_stats', 'mem_get_info', 'all_reduce', 'reduce_scatter', 'reduce_scatter_tensor', 'all_gather',
    'all_gather_into_tensor', 'all_to_all', 'all_to_all_single', 'broadcast', 'barrier', 'clip_grad_norm', 'clip_grad_norm_',
    'clip_grad_by_total_norm_fp32', 'get_grad_norm_fp32', 'switch_load_balancing_loss_func', 'sequence_load_balancing_loss_func',
    'z_loss_func', 'vocab_parallel_cross_entropy', 'cross_entropy', 'fused_vocab_parallel_cross_entropy', 'environ',
    'putenv', 'set_float32_matmul_precision', 'set_default_dtype', 'manual_seed'}
DUNDER_OK = {'__name__', '__doc__', '__qualname__', '__module__', '__init__', '__file__'}
SYS_BANNED_NAMES = {'modules', '_getframe', 'settrace', 'setprofile', 'meta_path', 'path_hooks', 'path_importer_cache'}
FILE_WRITE_CALLS = {'torch.save', 'numpy.save', 'numpy.savez', 'numpy.savez_compressed', 'numpy.savetxt', 'json.dump',
                    'yaml.dump', 'yaml.safe_dump', 'operator.setitem', 'operator.delitem', 'torch.jit.save',
                    'torch.onnx.export', 'torch.export.save', 'numpy.memmap', 'numpy.lib.format.open_memmap'}
# Builtins that may only be called directly (never bound to another name or passed around).
VALUE_BANNED_NAMES = {'open': 'G10', 'exec': 'G07', 'eval': 'G07', 'compile': 'G07', 'globals': 'G07', 'locals': 'G07',
                      'vars': 'G07', 'breakpoint': 'G07', 'input': 'G07'}
ENV_READ_METHODS = {'get', 'copy', 'keys', 'items', 'values'}
RNG_CALLS = {'random.seed', 'random.setstate', 'numpy.random.seed', 'numpy.random.set_state', 'torch.manual_seed', 'torch.seed',
             'torch.set_rng_state', 'torch.cuda.manual_seed', 'torch.cuda.manual_seed_all', 'torch.cuda.set_rng_state',
             'torch.cuda.set_rng_state_all', 'torch.random.manual_seed'}
MAPPING_MUTATORS = {'update', 'setdefault', 'pop', 'popitem', 'clear'}
SEMANTIC_CONTAINERS = {'param_groups', 'defaults', 'config', 'ddp_config', 'optimizer_config', 'transformer_config'}
WRAPPED_ATTRS = {'forward', 'step', 'backward', 'zero_grad', 'finish_grad_sync', 'start_grad_sync', 'start_param_sync'}
INPLACE_TENSOR_OPS = {'zero_', 'fill_', 'copy_', 'add_', 'sub_', 'mul_', 'div_', 'normal_', 'uniform_', 'clamp_', 'set_',
                      'masked_fill_', 'index_fill_', 'scatter_', 'fill_diagonal_', 'random_', 'bernoulli_', 'exponential_',
                      'resize_', 'detach_', 'fill', 'put', 'sort', 'resize'}
PARAM_WORDS = {'data', 'weight', 'bias', 'grad', 'main_grad', 'parameters', 'named_parameters', 'labels', 'tokens',
               'position_ids', 'loss_mask', 'cu_seqlens'}
OUTSIDE_TIMER_FILE = 'host_sync.py'      # all its hooks run outside the timed update
HOOK_NAMES = {'schedule.py': ('model_config_overrides', 'microbatch_order', 'forward_backward'),
              'recompute_policy.py': ('model_config_overrides', 'after_model_build'),
              'comm_groups.py': ('parallel_init_kwargs', 'ddp_config_overrides'),
              'state_placement.py': ('optimizer_config_overrides', 'after_optimizer_build'),
              'host_sync.py': ('on_train_start', 'before_update', 'after_update', 'on_train_end'),
              'data_path.py': ('prefetch', 'to_device')}
BANNED_CALLS = {'__import__', 'exec', 'eval', 'compile', 'globals', 'locals', 'vars', 'breakpoint', 'input'}
# Our frozen artifacts, matched against identifiers, attribute names, imported names and string constants.
FROZEN_SYMBOLS = {'receipt', 'receipts', 'receipt_writer', 'write_receipt', 'signals', 'signal_writer', 'write_signals',
                  'heartbeat', 'timers', 'timer', 'invariants', 'gates', 'frozen_zone_digest', 'signals_digest',
                  'reward', 'harness', 'stop_requested', 'lease'}
FROZEN_STRING_PARTS = ('receipt', 'heartbeat', 'signals/', 'stop.json', 'invariant', 'frozen_zone', 'lease')
INTROSPECTION_ATTRS = {'__globals__', '__code__', '__closure__', '__subclasses__', '__builtins__', '__bases__', '__mro__',
                       'f_globals', 'f_locals', 'f_back', 'f_code', 'gi_frame', 'tb_frame', '_getframe', 'settrace',
                       'setprofile', 'get_objects', 'get_referrers', 'get_referents', 'currentframe', 'stack'}
# Monkeypatching these dotted prefixes changes timing, precision, memory statistics, optimizer/LR/loss math,
# routing, or collectives: all frozen semantics.
BANNED_STORE_PREFIXES = ('time.', 'datetime.', 'torch.cuda.', 'torch.backends.', 'torch.distributed.', 'torch.optim.',
                         'torch.nn.utils.', 'torch.nn.functional.cross_entropy', 'torch.autograd.', 'torch.random.',
                         'torch.manual_seed', 'torch.set_', 'os.environ', 'os.putenv', 'sys.', 'builtins.',
                         'megatron.core.optimizer', 'megatron.core.optimizer_param_scheduler', 'megatron.core.timers',
                         'megatron.core.distributed.finalize_model_grads', 'megatron.core.tensor_parallel.cross_entropy',
                         'megatron.core.transformer.moe.router', 'megatron.core.transformer.moe.moe_utils',
                         'megatron.core.fusions.fused_cross_entropy', 'megatron.core.parallel_state.get_',
                         'megatron.core.rerun_state_machine', 'transformer_engine.pytorch.cross_entropy', 'numpy.random',
                         'random.')
PRECISION_CALLS = ('torch.set_float32_matmul_precision', 'torch.set_default_dtype', 'torch.set_default_tensor_type',
                   'torch.backends.', 'torch.set_flush_denormal', 'torch.use_deterministic_algorithms')
FILE_WRITE_ATTRS = {'write_text', 'write_bytes', 'unlink', 'rmdir', 'rename', 'touch', 'mkdir', 'symlink_to',
                    'hardlink_to', 'chmod', 'tofile', 'export_chrome_trace', 'export_stacks', 'export_memory_timeline',
                    'FileHandler', 'RotatingFileHandler', 'TimedRotatingFileHandler', 'WatchedFileHandler', 'basicConfig',
                    '_dump_snapshot', '_save_segment_usage', '_save_memory_usage', '_record_memory_history',
                    'savefig', 'to_csv', 'to_parquet', 'to_pickle', 'lchmod', 'link_to', 'rmtree', 'copytree', 'copyfile',
                    'renames'}   # 'replace' is handled by arity: Path.replace(target) vs str.replace(old, new)
# os: only these read-only functions and constants may be referenced (any other os.X, called or aliased, is G10).
OS_READ_ONLY = {'getenv', 'getpid', 'getppid', 'cpu_count', 'sched_getaffinity', 'getcwd', 'listdir', 'scandir', 'stat',
                'lstat', 'uname', 'times', 'getloadavg', 'urandom', 'fspath', 'fsencode', 'fsdecode', 'get_terminal_size',
                'sysconf', 'getuid', 'geteuid', 'getgid', 'getegid', 'getlogin', 'access', 'walk', 'readlink', 'statvfs',
                'strerror', 'get_exec_path', 'path', 'environ', 'sep', 'linesep', 'pathsep', 'name', 'devnull', 'curdir',
                'pardir', 'extsep', 'altsep', 'defpath', 'PathLike', 'R_OK', 'F_OK', 'X_OK', 'W_OK'}
OS_BANNED = {'os.system', 'os.popen', 'os.remove', 'os.unlink', 'os.rename', 'os.replace', 'os.rmdir', 'os.removedirs',
             'os.kill', 'os.killpg', 'os.fork', 'os.putenv', 'os.unsetenv', 'os.chdir', 'os.symlink', 'os.link',
             'os.environ.update', 'os.environ.setdefault', 'os.environ.pop', 'os.environ.clear', 'os._exit'}
# Attributes / dict keys whose assignment changes training semantics (trainable set, router, objective, batch,
# gradient provenance, precision) - G17 on any store, setattr or param_group['...'] write.
SEMANTIC_ATTRS = {'requires_grad', 'moe_router_topk', 'moe_expert_capacity_factor', 'num_moe_experts', 'moe_aux_loss_coeff',
                  'moe_z_loss_coeff', 'moe_router_load_balancing_type', 'clip_grad', 'lr', 'min_lr', 'max_lr', 'weight_decay',
                  'betas', 'eps', 'adam_beta1', 'adam_beta2', 'adam_eps', 'seq_length', 'global_batch_size', 'num_microbatches',
                  'calculate_per_token_loss', 'zero_grad_buffer', 'zero_grad', 'grad_scale_func', 'loss_func', 'compute_loss',
                  'fp8', 'params_dtype', 'main_params_dtype', 'exp_avg_dtype', 'exp_avg_sq_dtype', 'bf16', 'fp16',
                  'grad_reduce_in_fp32', 'accumulate_allreduce_grads_in_fp32', 'mtp_loss_scaling_factor', 'data', 'grad',
                  'main_grad', 'params', 'param_groups'}
DOWNCAST_DTYPES = {'bfloat16', 'float16', 'half', 'float8_e4m3fn', 'float8_e5m2', 'float8_e4m3fnuz', 'float8_e5m2fnuz'}
STATE_WORDS = {'exp_avg', 'exp_avg_sq', 'main_param', 'main_params', 'master', 'master_param', 'fp32_from_float16_groups',
               'main_grad', 'state_dict', 'shard_fp32_from_float16_groups'}


@dataclasses.dataclass
class PatchReport:
    accepted: bool
    mode: str
    findings: list
    files: dict
    diff_lines: int
    limit: int
    overlay_files: dict             # every overlay after the patch: name -> sha256 (RunRequest.overlay['files'])
    overlay_digest: str
    frozen_zone_digest_before: str | None = None
    frozen_zone_digest_after: str | None = None
    training_frozen_zone_digest: str | None = None      # the receipt invariant's definition, unchanged by construction
    staging_dir: str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def codes(self) -> set:
        return {f['code'] for f in self.findings}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def overlay_digest(files: dict[str, str]) -> str:
    """Digest of {name: sha256} for a complete overlay set."""
    rows = sorted(files.items())
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode('utf-8')).hexdigest()


def base_overlays(base_dir: str | Path | None = None) -> dict[str, str]:
    base = Path(base_dir) if base_dir else zones.PACKAGE_ROOT / zones.OVERLAY_DIR
    out = {}
    for name in zones.MUTABLE_OVERLAYS:
        p = base / name
        if p.is_file():
            out[name] = p.read_text(encoding='utf-8')
    return out


def diff_counts(old: str, new: str) -> tuple[int, int]:
    added = removed = 0
    for line in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm='', n=0):
        if line.startswith(('+++', '---', '@@')):
            continue
        if line.startswith('+'):
            added += 1
        elif line.startswith('-'):
            removed += 1
    return added, removed


def _dotted(node) -> str | None:
    """'a.b.c' for attribute chains, also through getattr(x, 'const') steps; None if the root is not a name."""
    parts = []
    while True:
        if isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'getattr' and len(node.args) >= 2
              and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)):
            parts.append(node.args[1].value)
            node = node.args[0]
        else:
            break
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return '.'.join(reversed(parts))
    return None


def _docstring_ids(tree: ast.AST) -> set[int]:
    """Docstrings are prose: they may mention receipts or invariants without touching them."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                out.add(id(first.value))
    return out


class _Checker(ast.NodeVisitor):
    def __init__(self, file: str, skip_constants: set[int] = frozenset()):
        self.file, self.findings, self.aliases = file, [], {}
        self.state_words, self.downcasts = [], []
        self.skip_constants = skip_constants
        self.module_roots: set[str] = set()       # names bound to imported modules (or attributes of them)
        self.call_funcs: set[int] = set()         # ids of nodes used directly as a call's function

    def add(self, code: str, node, detail: str):
        self.findings.append({'code': code, 'file': self.file, 'line': getattr(node, 'lineno', 0), 'detail': detail})

    def resolve(self, dotted: str | None) -> str | None:
        if not dotted:
            return dotted
        head, _, rest = dotted.partition('.')
        full = self.aliases.get(head, head)
        return full + ('.' + rest if rest else '')

    def _check_module(self, mod: str, node):
        top = mod.split('.')[0]
        if top in BANNED_MODULES:
            self.add('G06', node, f'banned module import {mod}')
        if mod == 'infrarsi' or mod.startswith('infrarsi.'):
            if not any(mod == a or mod.startswith(a + '.') for a in zones.ALLOWED_INFRARSI_IMPORTS):
                self.add('G04', node, f'import of frozen-zone module {mod}')
        parts = set(mod.lower().split('.'))
        if parts & FROZEN_SYMBOLS:
            self.add('G08', node, f'import of timer/receipt module {mod}')

    def visit_Import(self, node):
        for a in node.names:
            self._check_module(a.name, node)
            bound = (a.asname or a.name).split('.')[0]
            self.aliases[bound] = a.name if a.asname else a.name.split('.')[0]
            self.module_roots.add(bound)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.level:
            if node.level > 1:
                self.add('G05', node, 'relative import outside runtime/overlays')
            elif node.module is None:
                for a in node.names:
                    if a.name + '.py' not in zones.MUTABLE_OVERLAYS:
                        self.add('G05', node, f'relative import of non-overlay {a.name}')
            elif node.module + '.py' not in zones.MUTABLE_OVERLAYS:
                self.add('G05', node, f'relative import of non-overlay {node.module}')
        else:
            self._check_module(node.module or '', node)
        for a in node.names:
            if a.name.lower() in FROZEN_SYMBOLS:
                self.add('G08', node, f'import of timer/receipt symbol {a.name}')
            if node.module == 'os' and not node.level and a.name not in OS_READ_ONLY:
                self.add('G10', node, f'from os import {a.name}: not in the read-only os allowlist')
            if (node.module == 'sys' and a.name in SYS_BANNED_NAMES) or a.name == '*':
                self.add('G13', node, f'from {node.module} import {a.name}')
            if node.module and not node.level:
                self.aliases[a.asname or a.name] = f'{node.module}.{a.name}'
                self.module_roots.add(a.asname or a.name)
        self.generic_visit(node)

    def visit_Assign(self, node):
        # x = time / x = torch.cuda / x = getattr(torch, 'cuda'): x becomes an alias of that module path
        src = self.resolve(_dotted(node.value)) if isinstance(node.value, (ast.Name, ast.Attribute, ast.Call)) else None
        roots = {self.resolve(r).split('.')[0] for r in self.module_roots}
        if src and src.split('.')[0] in roots:
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self.aliases[t.id] = src
                    self.module_roots.add(t.id)
        self.generic_visit(node)

    def visit_Name(self, node):
        low = node.id.lower()
        if low in FROZEN_SYMBOLS:
            self.add('G08', node, f'reference to frozen symbol {node.id}')
        if node.id.startswith('__') and node.id.endswith('__') and node.id not in DUNDER_OK:
            self.add('G13', node, f'dunder name {node.id}')
        if node.id in ('setattr', 'delattr') and id(node) not in self.call_funcs:
            self.add('G09', node, f'{node.id} used as a value (dynamic attribute write)')
        if node.id == 'getattr' and id(node) not in self.call_funcs:
            self.add('G13', node, 'getattr used as a value (dynamic attribute lookup)')
        if node.id in VALUE_BANNED_NAMES and isinstance(node.ctx, ast.Load) and id(node) not in self.call_funcs:
            self.add(VALUE_BANNED_NAMES[node.id], node, f'builtin {node.id} used as a value')
        if node.id in STATE_WORDS:
            self.state_words.append(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr.lower() in FROZEN_SYMBOLS:
            self.add('G08', node, f'reference to frozen attribute {node.attr}')
        if node.attr in INTROSPECTION_ATTRS:
            self.add('G13', node, f'introspection attribute {node.attr}')
        elif node.attr.startswith('__') and node.attr.endswith('__') and node.attr not in DUNDER_OK:
            self.add('G13', node, f'dunder attribute {node.attr}')
        dotted = self.resolve(_dotted(node))
        if dotted in ('sys.modules', 'sys._getframe', 'gc.get_objects', 'gc.get_referrers', 'inspect.currentframe'):
            self.add('G13', node, f'introspection {dotted}')
        if node.attr in FILE_WRITE_ATTRS:
            self.add('G10', node, f'file-system write attribute .{node.attr}')
        if node.attr == 'open' and isinstance(node.ctx, ast.Load) and id(node) not in self.call_funcs:
            self.add('G10', node, '.open used as a value (its mode can no longer be checked)')
        if node.attr == 'replace' and isinstance(node.ctx, ast.Load) and id(node) not in self.call_funcs:
            self.add('G10', node, '.replace used as a value (Path.replace moves files)')
        if dotted and dotted.startswith('os.') and dotted.split('.')[1] not in OS_READ_ONLY:
            self.add('G10', node, f'{dotted} is not in the read-only os allowlist')
        if dotted == 'os.environ' and not self._env_read(node):
            self.add('G10', node, 'os.environ used other than os.environ.get/copy/keys/items/values')
        if node.attr in STATE_WORDS:
            self.state_words.append(node.attr)
        if node.attr in DOWNCAST_DTYPES and dotted and dotted.startswith('torch.'):
            self.downcasts.append(node)
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._check_store(node, dotted)
            if node.attr in SEMANTIC_ATTRS:
                self.add('G17', node, f'assignment to semantic attribute .{node.attr}')
            if node.attr in SENSITIVE_STORE_ATTRS:
                self.add('G09', node, f'assignment to sensitive attribute .{node.attr} (timing/precision/memory/collective/loss/env)')
        self.generic_visit(node)

    def _env_read(self, node) -> bool:
        parent = getattr(node, '_pg_parent', None)
        call = getattr(parent, '_pg_parent', None)
        return (isinstance(parent, ast.Attribute) and parent.attr in ENV_READ_METHODS and isinstance(call, ast.Call)
                and call.func is parent)

    def visit_AugAssign(self, node):
        dotted = self.resolve(_dotted(node.target))
        if dotted and (dotted.startswith('os.environ') or any(dotted == q.rstrip('.') or dotted.startswith(q) for q in BANNED_STORE_PREFIXES)):
            self.add('G10' if dotted.startswith('os.environ') else 'G09', node, f'in-place update of {dotted}')
        self.generic_visit(node)

    def visit_Subscript(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            dotted = self.resolve(_dotted(node.value))
            if dotted and (dotted == 'os.environ' or dotted.startswith('sys.')):
                self.add('G10', node, f'write to {dotted}[...]')
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value in SEMANTIC_ATTRS:
                self.add('G17', node, f"write to semantic key [{key.value!r}]")
        self.generic_visit(node)

    def _check_store(self, node, dotted: str | None):
        if dotted and any(dotted == p.rstrip('.') or dotted.startswith(p) for p in BANNED_STORE_PREFIXES):
            self.add('G09', node, f'monkeypatch of frozen function/flag {dotted}')

    def visit_Constant(self, node):
        if isinstance(node.value, str) and id(node) not in self.skip_constants:
            low = node.value.lower()
            if any(p in low for p in FROZEN_STRING_PARTS) or low in FROZEN_SYMBOLS:
                self.add('G08', node, f'string references a frozen artifact: {node.value[:60]!r}')
            if node.value in STATE_WORDS:
                self.state_words.append(node.value)
        self.generic_visit(node)

    def visit_keyword(self, node):
        if node.arg in STATE_WORDS:
            self.state_words.append(node.arg)
        self.generic_visit(node)

    def visit_Call(self, node):
        self.call_funcs.add(id(node.func))
        fn = self.resolve(_dotted(node.func))
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if name in BANNED_CALLS:
            self.add('G07', node, f'banned builtin {name}()')
        if name in ('setattr', 'delattr'):
            target = self.resolve(_dotted(node.args[0])) if node.args else None
            attr = node.args[1] if len(node.args) > 1 else None
            if not (isinstance(attr, ast.Constant) and isinstance(attr.value, str)):
                self.add('G09', node, f'{name} with a non-constant attribute name')
            else:
                full = f'{target}.{attr.value}' if target else attr.value
                if attr.value.lower() in FROZEN_SYMBOLS or attr.value in INTROSPECTION_ATTRS:
                    self.add('G08', node, f'{name} on frozen symbol {attr.value}')
                if attr.value in SEMANTIC_ATTRS:
                    self.add('G17', node, f'{name} on semantic attribute {attr.value}')
                if attr.value in SENSITIVE_STORE_ATTRS:
                    self.add('G09', node, f'{name} on sensitive attribute {attr.value}')
                if target:
                    self._check_store(node, full)
        if isinstance(node.func, ast.Attribute) and node.func.attr in ('requires_grad_', 'zero_grad_buffer_', 'set_moe_router_topk'):
            self.add('G17', node, f'call .{node.func.attr}() changes the trainable set or gradient provenance')
        if name == 'getattr' and len(node.args) > 1:
            if not (isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)):
                self.add('G13', node, 'getattr with a computed attribute name')
            elif node.args[1].value in INTROSPECTION_ATTRS or (node.args[1].value.startswith('__') and node.args[1].value not in DUNDER_OK):
                self.add('G13', node, f'getattr introspection {node.args[1].value}')
            elif node.args[1].value in FILE_WRITE_ATTRS or node.args[1].value in ('open', 'replace'):
                self.add('G10', node, f'getattr of file-system attribute {node.args[1].value!r}')
        if fn in ('operator.attrgetter', 'operator.methodcaller'):
            self.add('G13', node, f'dynamic attribute access {fn}()')
        if fn in FILE_WRITE_CALLS:
            self.add('G10', node, f'file write call {fn}()')
        if fn in RNG_CALLS:
            self.add('G17', node, f'RNG reseed {fn}() changes the data/initialisation stream')
        if isinstance(node.func, ast.Attribute):
            chain = _attr_chain(node.func.value)
            if node.func.attr in MAPPING_MUTATORS and chain & SEMANTIC_CONTAINERS:
                self.add('G17', node, f'.{node.func.attr}() on {sorted(chain & SEMANTIC_CONTAINERS)} mutates optimizer/model config')
            if node.func.attr in INPLACE_TENSOR_OPS and chain & PARAM_WORDS:
                self.add('G17', node, f'in-place .{node.func.attr}() on parameters/gradients/batch data')
        if fn == 'os.open':
            self.add('G10', node, 'low-level os.open')
        elif name == 'open' or fn in ('io.open', 'builtins.open', 'codecs.open') or (
                isinstance(node.func, ast.Attribute) and node.func.attr == 'open'):
            mode = 'r'
            module_level = name == 'open' or fn in ('io.open', 'builtins.open', 'codecs.open')
            pos = 1 if module_level else 0                 # open(file, mode) vs Path(...).open(mode) / x.open(mode)
            if len(node.args) > pos:
                mode = node.args[pos].value if isinstance(node.args[pos], ast.Constant) else None
            for kw in node.keywords:
                if kw.arg == 'mode':
                    mode = kw.value.value if isinstance(kw.value, ast.Constant) else None
            if not isinstance(mode, str) or any(ch in mode for ch in 'wax+'):
                self.add('G10', node, f'file open for writing or with unknown mode ({mode!r})')
        if isinstance(node.func, ast.Attribute) and node.func.attr in FILE_WRITE_ATTRS:
            self.add('G10', node, f'file-system write .{node.func.attr}()')
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'replace' and len(node.args) == 1 and not node.keywords:
            self.add('G10', node, '.replace(target) with one argument is Path.replace (moves a file); str.replace takes two')
        if fn and (fn in OS_BANNED or fn.startswith(('os.exec', 'os.spawn', 'os.posix_spawn'))):
            self.add('G10', node, f'banned os call {fn}')
        if fn and any(fn == p.rstrip('.') or fn.startswith(p) for p in PRECISION_CALLS):
            self.add('G09', node, f'precision/determinism flag call {fn}')
        if isinstance(node.func, ast.Attribute) and node.func.attr in ('half', 'bfloat16') and not node.args:
            self.downcasts.append(node)
        self.generic_visit(node)


def check_source(name: str, text: str) -> tuple[list[dict], ast.AST | None]:
    findings = []
    if type(text) is not str:
        return [{'code': 'G02', 'file': name, 'line': 0, 'detail': 'content must be text'}], None
    raw = text.encode('utf-8', errors='replace')
    if len(raw) > MAX_FILE_BYTES or '\x00' in text:
        return [{'code': 'G02', 'file': name, 'line': 0, 'detail': f'file too large or contains NUL ({len(raw)} bytes)'}], None
    try:
        tree = ast.parse(text, filename=name)
    except (SyntaxError, ValueError, RecursionError) as exc:
        return [{'code': 'G03', 'file': name, 'line': getattr(exc, 'lineno', 0) or 0, 'detail': f'syntax error: {exc}'}], None
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._pg_parent = parent
    ck = _Checker(name, _docstring_ids(tree))
    ck.visit(tree)
    findings += ck.findings
    findings += _contract_findings(name, tree)
    findings += _container_findings(name, tree)
    if ck.downcasts and ck.state_words:
        node = ck.downcasts[0]
        findings.append({'code': 'G14', 'file': name, 'line': getattr(node, 'lineno', 0),
                         'detail': f'low-precision cast in a file touching optimizer state ({sorted(set(ck.state_words))[:4]})'})
    return findings, tree


CONTAINER_CTORS = {'dict', 'list', 'set', 'OrderedDict', 'defaultdict', 'deque', 'Counter'}
CONTAINER_MUTATORS = {'update', 'setdefault', 'pop', 'popitem', 'clear', 'append', 'extend', 'insert', 'remove', 'reverse',
                      'sort', 'discard', 'add', 'appendleft', 'extendleft', 'popleft', 'rotate', '__setitem__', '__delitem__'}


def _is_container_ctor(value) -> bool:
    if isinstance(value, (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp)):
        return True
    if isinstance(value, ast.Call):
        f = value.func
        last = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else '')
        return last in CONTAINER_CTORS
    return False


def _local_containers(tree: ast.AST) -> set[str]:
    """Names whose every binding in the file is a container this overlay builds itself (literal, comprehension or
    dict()/list()/... call). Function parameters, loop variables, with-targets and anything else are external."""
    good, bad = set(), set()
    for node in ast.walk(tree):
        bindings = []                                   # (target, value or None)
        if isinstance(node, ast.Assign):
            bindings = [(t, node.value) for t in node.targets]
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
            bindings = [(node.target, node.value)]
        elif isinstance(node, ast.AugAssign):
            bindings = [(node.target, None)]
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            bindings = [(node.target, None)]
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            bindings = [(node.optional_vars, None)]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = node.args
            bad |= {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs} | {x.arg for x in (a.vararg, a.kwarg) if x}
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            continue
        for target, value in bindings:
            plain = isinstance(target, ast.Name)
            for n in ast.walk(target):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    (good if plain and value is not None and _is_container_ctor(value) else bad).add(n.id)
    return good - bad


def _literal_key_names(tree: ast.AST) -> set[str]:
    """Loop names bound only to the strings of a literal tuple/list/set (for k in ('exp_avg', 'exp_avg_sq'): ...),
    provided none of them is a semantic key."""
    good, bad = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)) and isinstance(node.target, ast.Name):
            it = node.iter
            ok = isinstance(it, (ast.Tuple, ast.List, ast.Set)) and it.elts and all(
                isinstance(e, ast.Constant) and isinstance(e.value, str) and e.value not in SEMANTIC_ATTRS for e in it.elts)
            (good if ok else bad).add(node.target.id)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                bad |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    return good - bad


def _container_findings(name: str, tree: ast.AST) -> list[dict]:
    """G17: writes into containers the overlay did not create itself (optimizer param groups and state, model and
    runtime config, ctx.plan, hook arguments, ...) need a literal, non-semantic key; mutator methods on them are banned."""
    out = []
    local, literal = _local_containers(tree), _literal_key_names(tree)
    add = lambda node, detail: out.append({'code': 'G17', 'file': name, 'line': getattr(node, 'lineno', 0), 'detail': detail})
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)):
            if isinstance(node.value, ast.Name) and node.value.id in local:
                continue
            key = node.slice
            safe = (isinstance(key, ast.Constant) and isinstance(key.value, str)) or (isinstance(key, ast.Name) and key.id in literal)
            if not safe:
                add(node, f'write with a computed key into {ast.unparse(node.value)[:60]} (not a container this overlay created)')
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Subscript):
            base = node.target.value
            if not (isinstance(base, ast.Name) and base.id in local):
                add(node, f'in-place update of {ast.unparse(node.target)[:60]} (not a container this overlay created)')
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in CONTAINER_MUTATORS:
            base = node.func.value
            if isinstance(base, ast.Name) and base.id in local:
                continue
            if isinstance(base, ast.Constant) or isinstance(base, (ast.List, ast.Dict, ast.Set)):
                continue
            add(node, f'.{node.func.attr}() on {ast.unparse(base)[:60]} (not a container this overlay created)')
    return out


def _attr_chain(node) -> set[str]:
    """Attribute and call names along an expression's receiver chain (a.b().c[0].d -> {a, b, c, d})."""
    out = set()
    while True:
        if isinstance(node, ast.Attribute):
            out.add(node.attr)
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                out.add(node.slice.value)
            node = node.value
        elif isinstance(node, ast.Name):
            out.add(node.id)
            return out
        else:
            return out


_CONDITIONAL = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp, ast.BoolOp, ast.ListComp, ast.SetComp, ast.DictComp,
                ast.GeneratorExp, ast.Lambda, ast.Match, ast.ExceptHandler, ast.FunctionDef, ast.AsyncFunctionDef)
CALL_THROUGH = {'checkpoint'}     # torch.utils.checkpoint.checkpoint(orig, ...), tensor_parallel.checkpoint(orig, ...)
SYNC_OR_WORK_CALLS = {'synchronize', 'step', 'all_reduce', 'reduce_scatter', 'all_gather', 'all_to_all', 'broadcast', 'barrier',
                      'backward', 'all_gather_into_tensor', 'reduce_scatter_tensor', 'all_to_all_single', 'send', 'recv',
                      'isend', 'irecv', 'batch_isend_irecv', 'wait', 'forward', 'finish_grad_sync', 'start_grad_sync',
                      'start_param_sync', 'zero_grad'}


def _uses(fn: ast.AST, names: set[str]) -> list[ast.Call]:
    """Calls inside fn that invoke one of `names`: directly, or passed first to a call-through helper (checkpointing)."""
    out = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in names:
            out.append(node)
        elif (node.args and isinstance(node.args[0], ast.Name) and node.args[0].id in names
              and (_dotted(node.func) or '').rsplit('.', 1)[-1] in CALL_THROUGH):
            out.append(node)
    return out


def _is_conditional(node: ast.AST, fn: ast.AST) -> bool:
    p = getattr(node, '_pg_parent', None)
    while p is not None and p is not fn:
        if isinstance(p, _CONDITIONAL):
            return True
        p = getattr(p, '_pg_parent', None)
    return False


def _own_nodes(fn: ast.AST):
    """Nodes of fn's body, not descending into nested functions, lambdas or classes."""
    body = getattr(fn, 'body', [])
    stack = list(body) if isinstance(body, list) else [body]          # a Lambda's body is one expression
    while stack:
        n = stack.pop()
        yield n
        stack.extend(c for c in ast.iter_child_nodes(n) if not isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                                                ast.Lambda, ast.ClassDef)))


RESULT_SAFE_METHODS = {'record_stream', 'size', 'dim', 'numel', 'is_contiguous', 'data_ptr', 'get', 'keys', 'items',
                       'values', 'detach', 'clone', 'element_size'}
BENIGN_CALLS = {'len', 'isinstance', 'type', 'id'}


def _params(fn) -> tuple[str | None, str | None, set[str]]:
    a = fn.args
    names = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
    return (a.vararg.arg if a.vararg else None), (a.kwarg.arg if a.kwarg else None), names


def _forwards_inputs(call: ast.Call, fn, *, skip_first: bool = False, only_kwargs: bool = False) -> bool:
    """The call passes the wrapper's own inputs unchanged: positional arguments are its parameters or *vararg,
    keywords are name=name or **kwarg (call-through helpers may add constant keywords such as use_reentrant)."""
    var, kw, names = _params(fn)
    args = call.args[1:] if skip_first else call.args
    if only_kwargs and (args or len(call.keywords) != 1):
        return False
    for x in args:
        if isinstance(x, ast.Starred):
            if not (isinstance(x.value, ast.Name) and x.value.id == var):
                return False
        elif not (isinstance(x, ast.Name) and x.id in names):
            return False
    for k in call.keywords:
        if k.arg is None:
            if not (isinstance(k.value, ast.Name) and k.value.id == kw):
                return False
        elif skip_first and isinstance(k.value, ast.Constant):
            continue
        elif not (isinstance(k.value, ast.Name) and k.value.id == k.arg and k.arg in names):
            return False
    return True


def _touches(node, names: set[str]) -> bool:
    return bool(_attr_chain(node) & names) if names else False


def _mutation_problem(fn, protected: set[str], result: set[str]) -> str | None:
    """Inputs (protected) and the original's result may not be rebound, written into or modified in place; the result
    may not be handed to other code before it is returned."""
    for n in _own_nodes(fn):
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                if isinstance(t, ast.Name) and t.id in protected:
                    return f'rebinds input {t.id}'
                if isinstance(t, (ast.Subscript, ast.Attribute)) and _touches(t.value, protected | result):
                    return 'writes into an input or the result'
        elif isinstance(n, ast.Delete) and any(_touches(t, protected | result) for t in n.targets):
            return 'deletes from an input or the result'
        elif isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute) and _touches(f.value, protected | result):
                m = f.attr
                if m in MAPPING_MUTATORS or m in INPLACE_TENSOR_OPS or (m.endswith('_') and not m.startswith('_')) or m in (
                        'append', 'extend', 'insert', 'remove', 'reverse', 'sort', 'backward'):
                    return f'modifies an input or the result (.{m})'
                if _touches(f.value, result) and m not in RESULT_SAFE_METHODS:
                    return f'calls .{m}() on the original result'
            name = f.id if isinstance(f, ast.Name) else None
            if name not in BENIGN_CALLS and any(_touches(a.value if isinstance(a, ast.Starred) else a, result)
                                                for a in list(n.args) + [k.value for k in n.keywords]):
                return 'passes the original result to other code before returning it'
    return None


def _pass_through_problem(fn: ast.AST, names: set[str], *, only_kwargs: bool = False) -> str | None:
    """None when fn calls one of `names` exactly once on every path, forwarding its own inputs unchanged, and returns
    that result untouched; else why not."""
    uses = _uses(fn, names)
    if len(uses) != 1 or _is_conditional(uses[0], fn):
        return f'{len(uses)} call(s) of the original, {sum(_is_conditional(u, fn) for u in uses)} conditional'
    call = uses[0]
    through = not (isinstance(call.func, ast.Name) and call.func.id in names)          # checkpoint(orig, ...)
    if not _forwards_inputs(call, fn, skip_first=through, only_kwargs=only_kwargs):
        return 'the original must receive the wrapper inputs unchanged (*args/**kwargs or name=name)'
    ancestors, p = set(), getattr(call, '_pg_parent', None)
    while p is not None and p is not fn:
        ancestors.add(id(p))
        p = getattr(p, '_pg_parent', None)
    parent = getattr(call, '_pg_parent', None)
    result = {t.id for t in parent.targets if isinstance(t, ast.Name)} if isinstance(parent, ast.Assign) else set()
    returns = []
    for n in _own_nodes(fn):
        if isinstance(n, (ast.Return, ast.Raise)) and id(n) not in ancestors and (n.lineno, n.col_offset) < (call.lineno, call.col_offset):
            return 'returns or raises before calling the original'
        if isinstance(n, ast.Return):
            returns.append(n)
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.NamedExpr)) and n is not parent:
            tg = n.targets if isinstance(n, ast.Assign) else [n.target]
            if any(isinstance(t, ast.Name) and t.id in result for t in tg):
                return 'reassigns the original result'
    if not returns:
        return 'does not return the original result'
    if any(not (r.value is call or isinstance(r.value, ast.Name) and r.value.id in result) for r in returns):
        return 'must return the original result unchanged'
    var, kw, params = _params(fn)
    protected = params | {x for x in (var, kw) if x}
    protected -= names                                                     # captured originals are not inputs
    for n in _own_nodes(fn):                                               # no extra training work around the original
        if isinstance(n, ast.Call) and n is not call and isinstance(n.func, ast.Attribute) and (
                n.func.attr in WRAPPED_ATTRS or n.func.attr in ('zero_grad_buffer', 'zero_grad_buffer_')):
            return f'calls training machinery .{n.func.attr}() besides the original'
    return _mutation_problem(fn, protected, result)


def _function_aliases(tree: ast.AST, top: dict) -> dict:
    """name -> module-level function for every `alias = func` (or `alias = other_alias`, `alias = lambda ...`)
    assignment anywhere in the file (scope-insensitive on purpose: conservative)."""
    out = dict(top)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                src = node.value
                target = out.get(src.id) if isinstance(src, ast.Name) else (src if isinstance(src, ast.Lambda) else None)
                for t in node.targets:
                    if target is not None and isinstance(t, ast.Name) and out.get(t.id) is not target:
                        out[t.id] = target
                        changed = True
    return out


def _closure(top: dict, root: ast.AST) -> list[ast.AST]:
    """root plus the module-level functions it (transitively) calls by name, following `alias = func` bindings."""
    seen, todo = {id(root): root}, [root]
    while todo:
        for node in ast.walk(todo.pop()):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in top:
                f = top[node.func.id]
                if id(f) not in seen:
                    seen[id(f)] = f
                    todo.append(f)
    return list(seen.values())


TRANSFER_METHODS = {'to', 'pin_memory', 'cuda', 'contiguous', 'items', 'keys', 'values', 'get', 'record_stream', 'is_pinned',
                    'data_ptr', 'size', 'dim', 'numel', 'element_size', 'copy'}
META_ATTRS = {'shape', 'dtype', 'ndim', 'device', 'nbytes', 'is_cuda', 'size', 'dim', 'numel', 'is_pinned', 'data_ptr'}
TRANSFER_FUNCS = {'from_numpy', 'as_tensor', 'tensor', 'copy_', 'empty_like', 'dict', 'list', 'tuple', 'len', 'isinstance', 'zip',
                  'enumerate', 'sorted', 'id', 'print'}


def _names_in(node) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} if node is not None else set()


def _meta_only(expr, tainted: set) -> bool:
    """expr reads only metadata of tainted values (x.shape, x.dtype, x.size(), len(x)) or is free of them."""
    if not _names_in(expr) & tainted or isinstance(expr, ast.Constant):
        return True
    if isinstance(expr, ast.Name):
        return False
    if isinstance(expr, ast.Attribute):
        return expr.attr in META_ATTRS
    if isinstance(expr, ast.Call):
        f = expr.func
        return (isinstance(f, ast.Attribute) and f.attr in META_ATTRS) or (isinstance(f, ast.Name) and f.id == 'len')
    if isinstance(expr, ast.Subscript):
        return _meta_only(expr.value, tainted)
    return False


def _to_device_findings(top: dict, td: ast.FunctionDef):
    """Taint the batch (the arrays parameter and everything derived from it); only transfers may touch tainted values."""
    parts = _closure(top, td)
    tainted = {a.arg for a in td.args.args[1:2]}
    keys = set()
    for _ in range(8):                                  # fixed point over assignments and loop targets
        before = len(tainted) + len(keys)
        for part in parts:
            for node in ast.walk(part):
                if isinstance(node, (ast.For, ast.comprehension)):
                    src = node.iter
                    if _names_in(src) & tainted:
                        tgt = node.target
                        if isinstance(src, ast.Call) and isinstance(src.func, ast.Attribute) and src.func.attr == 'items' \
                                and isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2:
                            keys |= _names_in(tgt.elts[0])
                            tainted |= _names_in(tgt.elts[1])
                        elif isinstance(src, ast.Call) and isinstance(src.func, ast.Attribute) and src.func.attr == 'keys' \
                                or isinstance(src, ast.Name):
                            keys |= _names_in(tgt)
                        else:
                            tainted |= _names_in(tgt)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in top and any(
                        _names_in(a) & tainted for a in list(node.args) + [k.value for k in node.keywords]):
                    fa = top[node.func.id].args
                    tainted |= {a.arg for a in fa.args + fa.kwonlyargs}
                elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
                    if _names_in(node.value) & tainted:
                        for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                            tainted |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        if len(tainted) + len(keys) == before:
            break
    for part in parts:
        for node in ast.walk(part):
            if isinstance(node, ast.Subscript) and _names_in(node.value) & tainted:
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    yield node, 'to_device may not write into the batch'
                elif not (isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)
                          or isinstance(node.slice, ast.Name) and node.slice.id in keys):
                    yield node, 'to_device may only look batch arrays up by key (no slicing or indexing)'
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)) and _names_in(node.value) & tainted:
                yield node, 'to_device may not set attributes on batch tensors'
            elif (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and _names_in(node.value) & tainted
                  and node.attr not in TRANSFER_METHODS | META_ATTRS | {'copy_'}
                  and not (isinstance(getattr(node, '_pg_parent', None), ast.Call) and node._pg_parent.func is node)):
                yield node, f'to_device may not take .{node.attr} of a batch value as a value'
            elif isinstance(node, ast.Compare) and _names_in(node) & tainted:
                none_test = all(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops) and any(
                    isinstance(c, ast.Constant) and c.value is None for c in [node.left] + node.comparators)
                if not none_test and not all(_meta_only(o, tainted) for o in [node.left] + node.comparators):
                    yield node, 'to_device may compare batch metadata (shape/dtype/None) only'
            elif isinstance(node, (ast.BinOp, ast.UnaryOp, ast.AugAssign)) and _names_in(node) & tainted:
                yield node, 'to_device may not compute on batch values'
            elif isinstance(node, ast.Call):
                f = node.func
                last = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else '')
                on_tainted = isinstance(f, ast.Attribute) and _names_in(f.value) & tainted
                args_tainted = any(_names_in(a) & tainted for a in list(node.args) + [k.value for k in node.keywords])
                if isinstance(f, ast.Name) and f.id in tainted:
                    yield node, f'to_device may not call {f.id}(), which was derived from batch values'
                    continue
                if on_tainted and last == 'copy_':
                    if not (len(node.args) >= 1 and _names_in(node.args[0]) & tainted):
                        yield node, 'to_device may only copy batch values into a buffer, never into the batch'
                elif on_tainted and last not in TRANSFER_METHODS:
                    yield node, f'to_device may only transfer batch tensors (.{last}() is not a transfer)'
                elif args_tainted and not on_tainted and last not in TRANSFER_FUNCS \
                        and not (isinstance(f, ast.Name) and f.id in top):
                    yield node, f'to_device may only pass batch arrays to transfer functions ({last or "call"}())'
                if (on_tainted or args_tainted) and any(k.arg == 'dtype' for k in node.keywords):
                    yield node, 'to_device may not change dtypes'
                if on_tainted and last == 'to' and any(isinstance(a, ast.Attribute) and (_dotted(a) or '').startswith('torch.')
                                                       and a.attr not in ('device',) for a in node.args):
                    yield node, 'to_device may not change dtypes (.to(dtype))'
            elif isinstance(node, ast.Slice):
                yield node, 'to_device may not slice (values and shapes must be preserved)'


def _contract_findings(name: str, tree: ast.AST) -> list[dict]:
    """G19: hook semantics the runtime relies on (runtime/overlays/__init__.py contract)."""
    out = []
    add = lambda node, detail: out.append({'code': 'G19', 'file': name, 'line': getattr(node, 'lineno', 0), 'detail': detail})
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    top = {n.name: n for n in getattr(tree, 'body', []) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    top = _function_aliases(tree, top)
    # schedule.forward_backward must call run(**kwargs) exactly once, unconditionally, and return its result.
    fb = top.get('forward_backward') if name == 'schedule.py' else None
    if fb is not None and not isinstance(fb, ast.Lambda):
        run_name = fb.args.args[1].arg if len(fb.args.args) > 1 else 'run'
        problem = _pass_through_problem(fb, {run_name}, only_kwargs=True)
        if problem:
            add(fb, f'forward_backward must call {run_name}(**kwargs) exactly once and return its result ({problem})')
    # Replacing forward/step/... must wrap the captured original: one unconditional call whose result is returned.
    captured: dict[str, tuple[str, str]] = {}      # name -> (receiver source, attribute) it holds the original of
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute) and node.value.attr in WRAPPED_ATTRS:
            for t in node.targets:
                if isinstance(t, ast.Name):
                    if t.id in captured and captured[t.id] != (ast.unparse(node.value.value), node.value.attr):
                        captured[t.id] = ('?', '?')        # one name holding different originals: matches nothing
                    else:
                        captured[t.id] = (ast.unparse(node.value.value), node.value.attr)
    for fn in funcs.values():                       # parameters defaulting to a captured original (closures in loops)
        for a, d in list(zip(fn.args.args[len(fn.args.args) - len(fn.args.defaults):], fn.args.defaults)) + list(
                zip(fn.args.kwonlyargs, fn.args.kw_defaults)):
            if isinstance(d, ast.Name) and d.id in captured:
                captured[a.arg] = captured[d.id]
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            tg = node.targets if isinstance(node, ast.Assign) else [node.target]
            targets = [(t, node.value) for t in tg if isinstance(t, ast.Attribute) and t.attr in WRAPPED_ATTRS]
        for t, value in targets:
            receiver = ast.unparse(t.value)
            own = {n for n, (recv, attr) in captured.items() if attr == t.attr and recv == receiver}
            if isinstance(node, ast.Assign) and isinstance(value, ast.Name) and value.id in own:
                continue                            # restoring the original
            wrapper = funcs.get(value.id) if isinstance(node, ast.Assign) and isinstance(value, ast.Name) else None
            if wrapper is None:
                add(node, f'.{t.attr} replaced by something other than a wrapper function defined here')
                continue
            if _uses(wrapper, set(captured) - own):
                add(node, f'.{t.attr} wrapper {wrapper.name}() calls an original captured from another attribute or object')
                continue
            problem = _pass_through_problem(wrapper, own)
            if problem:
                add(node, f'.{t.attr} wrapper {wrapper.name}() must call the captured original exactly once and return its result ({problem})')
    # Every host_sync hook runs outside the update timer, and any function of the file can be reached from them
    # (aliases, partials, containers), so the whole module is scanned rather than a call graph: no collective,
    # synchronisation, optimizer step or model compute may be referenced at all (called directly or bound first).
    for hook_name in HOOK_NAMES.get(name, ()):
        if isinstance(top.get(hook_name), ast.Lambda):
            add(top[hook_name], f'hook {hook_name} must be a def function (not a lambda)')
    if name == OUTSIDE_TIMER_FILE:
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                d = _dotted(node) or ''
                if d != 'ctx.step' and (node.attr in SYNC_OR_WORK_CALLS or d.startswith(('torch.distributed', 'dist.'))):
                    add(node, f'{name} hooks run outside the update timer and may not use {d or node.attr}')
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = node.module if isinstance(node, ast.ImportFrom) else ','.join(a.name for a in node.names)
                names = {a.name for a in node.names}
                if 'torch.distributed' in (mod or '') or names & SYNC_OR_WORK_CALLS:
                    add(node, f'{name} hooks run outside the update timer and may not import {mod or sorted(names)}')
    if isinstance(fb, ast.Lambda) or isinstance(top.get('to_device'), ast.Lambda):
        return out
    # data_path.to_device (and helpers it calls) must preserve values, dtypes and shapes of the batch.
    td = top.get('to_device') if name == 'data_path.py' else None
    if td is not None:
        for node, detail in _to_device_findings(top, td):
            add(node, detail)
    return out


def allowed_override_keys() -> dict[str, set]:
    """The runtime's per-hook key whitelist (infrarsi.runtime.overlays.ALLOWED, owner T); {} if not shipped."""
    try:
        from ..runtime.overlays import ALLOWED
    except ImportError:
        return {}
    return {k: set(v) for k, v in ALLOWED.items()}


def _override_findings(name: str, tree: ast.AST) -> list[dict]:
    """G18: a *_overrides / parallel_init_kwargs hook returning a dict literal may only use the runtime's ALLOWED keys
    (statically visible part of the check the runtime repeats; computed dicts are left to the runtime)."""
    allowed = allowed_override_keys()
    out, module = [], name[:-3]
    for node in getattr(tree, 'body', []):
        if not isinstance(node, ast.FunctionDef) or f'{module}.{node.name}' not in allowed:
            continue
        keys = allowed[f'{module}.{node.name}']
        for sub in ast.walk(node):
            if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Dict):
                for k in sub.value.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str) and k.value not in keys:
                        out.append({'code': 'G18', 'file': name, 'line': sub.lineno,
                                    'detail': f'{module}.{node.name} may not return key {k.value!r} (allowed: {sorted(keys)})'})
    return out


def _signatures(tree: ast.AST | None) -> dict[str, int]:
    out = {}
    if tree is None:
        return out
    for node in getattr(tree, 'body', []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith('_'):
            out[node.name] = len(node.args.posonlyargs) + len(node.args.args)
    return out


def apply(overlay_files: dict[str, str], *, mode: str = 'strong', base_dir: str | Path | None = None,
          code_root: str | Path | None = None, staging_dir: str | Path | None = None,
          materialize_to: str | Path | None = None) -> PatchReport:
    """Check (and optionally materialize) a proposed overlay set.

    overlay_files: {overlay name: complete new text}; names may be bare ('host_sync.py') or package-relative.
    code_root + staging_dir: copy the package tree, write the overlays into the copy and require the frozen-zone
    digest of the copy to equal the original's. materialize_to: write the complete overlay set (6 files) there.
    """
    if mode not in LINE_LIMITS:
        raise ValueError(f'mode {mode!r}')
    limit = LINE_LIMITS[mode]
    base = base_overlays(base_dir)
    findings, files, merged = [], {}, dict(base)
    if type(overlay_files) is not dict or not overlay_files:
        findings.append({'code': 'G01', 'file': '', 'line': 0, 'detail': 'overlay must be a non-empty {file: text} object'})
        overlay_files = {}
    for raw_name, text in overlay_files.items():
        name = zones.overlay_name(str(raw_name))
        if name is None:
            findings.append({'code': 'G01', 'file': str(raw_name), 'line': 0,
                             'detail': f'not a mutable overlay; allowed: {", ".join(zones.MUTABLE_OVERLAYS)}'})
            continue
        f, tree = check_source(name, text)
        findings += f
        old = base.get(name, '')
        if type(text) is str:
            added, removed = diff_counts(old, text)
            files[name] = {'sha256': _sha(text), 'base_sha256': _sha(old) if name in base else None,
                           'added': added, 'removed': removed, 'changed': text != old}
            merged[name] = text
            if name in base and tree is not None:
                try:
                    base_sig = _signatures(ast.parse(old))
                except SyntaxError:
                    base_sig = {}
                new_sig = _signatures(tree)
                # The runtime falls back to the pristine hook when a file omits it; a changed signature breaks the call.
                for fn, nargs in base_sig.items():
                    if fn in new_sig and new_sig[fn] != nargs:
                        findings.append({'code': 'G12', 'file': name, 'line': 0,
                                         'detail': f'hook {fn}() positional arity {nargs} -> {new_sig[fn]}'})
            if tree is not None:
                findings += _override_findings(name, tree)
    diff_lines = sum(v['added'] + v['removed'] for v in files.values())
    if diff_lines > limit:
        findings.append({'code': 'G11', 'file': '', 'line': 0, 'detail': f'diff {diff_lines} lines > {mode} limit {limit}'})
    if files and not any(v['changed'] for v in files.values()):
        findings.append({'code': 'G15', 'file': '', 'line': 0, 'detail': 'overlay identical to the pristine overlays'})
    overlay_shas = {n: _sha(t) for n, t in sorted(merged.items())}
    report = PatchReport(accepted=False, mode=mode, findings=findings, files=files, diff_lines=diff_lines, limit=limit,
                         overlay_files=overlay_shas, overlay_digest=overlay_digest(overlay_shas))
    if code_root is not None and staging_dir is not None and not findings:
        src, dst = Path(code_root), Path(staging_dir)
        report.frozen_zone_digest_before = zones.frozen_zone_digest(src)
        if dst.exists():
            raise FileExistsError(f'staging dir exists: {dst}')
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        (dst / zones.OVERLAY_DIR).mkdir(parents=True, exist_ok=True)
        for name, text in merged.items():
            (dst / zones.OVERLAY_DIR / name).write_text(text, encoding='utf-8', newline='\n')
        report.frozen_zone_digest_after = zones.frozen_zone_digest(dst)
        report.staging_dir = str(dst)
        if report.frozen_zone_digest_after != report.frozen_zone_digest_before:
            findings.append({'code': 'G16', 'file': '', 'line': 0, 'detail': 'frozen-zone digest changed in the staged tree'})
        report.training_frozen_zone_digest = zones.training_frozen_zone_digest(dst)
        if report.training_frozen_zone_digest != zones.training_frozen_zone_digest(src):
            findings.append({'code': 'G16', 'file': '', 'line': 0, 'detail': 'training frozen-zone digest changed in the staged tree'})
    report.accepted = not findings
    if report.accepted and materialize_to is not None:
        out = Path(materialize_to)
        out.mkdir(parents=True, exist_ok=True)
        for name, text in merged.items():
            (out / name).write_text(text, encoding='utf-8', newline='\n')
    return report


def static_scan(text: str, name: str = 'overlay.py') -> list[dict]:
    """AST rules only (no diff/limit/name checks); used by red-team tooling."""
    return check_source(name, text)[0]
