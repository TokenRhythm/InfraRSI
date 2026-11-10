"""What a training job's code does with its settings: the framework calls that set its speed, and what controls them.

Whatever a codebase calls its settings - argparse, click, a dataclass, a yaml or OmegaConf config, an env var, a module
constant - its speed is set by a short list of framework calls (SINKS): torch.compile, DataLoader(num_workers=),
AdamW(fused=), `allow_tf32 = ...`, autocast(dtype=, enabled=), DDP's no_sync / gradient_as_bucket_view / static_graph /
bucket_cap_mb, gradient checkpointing, Accelerator(mixed_precision=), from_pretrained(attn_implementation=), FSDP's
sharding, a DeepSpeed config dict. `trace` finds these calls in the job's source and follows each argument and each
enclosing condition back through local variables, `self` fields and function parameters - across the job's modules and
into the library functions it hands its settings to - to what sets them: `args.fused_adam`, `cfg['train']['workers']`,
`os.environ['FP8']`, a module constant. A Use gives the mechanism the call sets and, for each value of the setting, the
state it produces (`--compile 1`: compile on), so `init` drafts the setting as a knob mapped to the registry whatever
its name; `fixed` holds the state of the calls no setting controls (`num_workers=0` at train.py:82). Read from source,
never imported or run.
"""
from __future__ import annotations

import ast
import builtins
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

UNKNOWN = object()
DTYPES = {'bfloat16': 'bf16', 'bf16': 'bf16', 'float16': 'fp16', 'half': 'fp16', 'fp16': 'fp16', 'float32': 'fp32',
          'float': 'fp32', 'fp32': 'fp32'}
# a chain no option, config key or env var matches is reported only when its root names settings (not model.config)
SETTINGS_ROOT = re.compile(r'(?i)^(args?|arguments|opts?|options|cfg|conf|config|hparams|hp|params|flags|settings|'
                           r'\w+_(args|cfg|config|conf|opts))$')


@dataclass(frozen=True)
class Spec:
    arg: str | None = None            # keyword of the setting; None (and no pos): the call itself is the lever
    mechanism: str | None = None      # registry mechanism (None: a speed setting the registry has no id for)
    mode: str = 'value'               # value: the argument's value is the state (through smap); truth: on when truthy;
                                      # occurs: on when the call runs
    on: str | None = None
    off: str | None = None
    smap: tuple | str = ()            # (value, state) pairs; 'dtype': a torch dtype
    role: str = 'switch'              # how init drafts the choices of a setting whose definition gives none
    pos: int | None = None            # the argument's position when it is not passed by keyword


ATTN_HF = (('sdpa', 'auto'), ('flash_attention_2', 'flash'), ('eager', 'unfused'))
MIXED = (('bf16', 'bf16'), ('fp16', 'fp16'), ('no', 'fp32'), ('fp8', 'float8'))
LIGHTNING = (('bf16-mixed', 'bf16'), ('bf16', 'bf16'), ('16-mixed', 'fp16'), ('16', 'fp16'), ('32-true', 'fp32'),
             ('32', 'fp32'))
SHARDING = (('FULL_SHARD', 'full_shard'), ('SHARD_GRAD_OP', 'shard_grad_op'), ('HYBRID_SHARD', 'hybrid_shard'),
            ('NO_SHARD', 'none'))
CKPT = Spec(None, 'recompute', 'occurs', 'full', 'none')
CALLS = {
    'compile': (Spec(None, 'compile', 'occurs', 'on', 'off'), Spec('disable', 'compile', 'truth', 'off', 'on')),
    'AdamW': (Spec('fused', 'optimizer', 'truth', 'adamw_fused', 'adamw'),),
    'Adam': (Spec('fused', 'optimizer', 'truth', 'adamw_fused', 'adamw'),),
    'AdamW8bit': (Spec(None, 'optimizer', 'occurs', 'adamw8bit', None),),
    'DataLoader': (Spec('num_workers', 'data_workers', role='workers'), Spec('pin_memory', None, 'truth'),
                   Spec('persistent_workers', None, 'truth'), Spec('prefetch_factor', None, role='prefetch')),
    'DistributedDataParallel': (Spec('gradient_as_bucket_view', 'grad_as_bucket_view', 'truth', 'on', 'off'),
                                Spec('static_graph', 'ddp_static_graph', 'truth', 'on', 'off'),
                                Spec('bucket_cap_mb', 'ddp_bucket_mb', role='bucket')),
    'no_sync': (Spec(None, 'grad_sync_every_micro', 'occurs', 'off', 'on'),),
    'gradient_checkpointing_enable': (CKPT,), 'enable_gradient_checkpointing': (CKPT,), 'checkpoint': (CKPT,),
    'checkpoint_sequential': (CKPT,),
    'autocast': (Spec('dtype', 'precision', smap='dtype'), Spec('enabled', 'precision', 'truth', None, 'fp32')),
    'Accelerator': (Spec('mixed_precision', 'precision', smap=MIXED), Spec('gradient_accumulation_steps', None, role='ga'),
                    Spec('dynamo_backend', 'compile', smap=(('inductor', 'on'), ('INDUCTOR', 'on'), ('no', 'off'),
                                                            ('NO', 'off')))),
    'from_pretrained': (Spec('attn_implementation', 'attention_backend', smap=ATTN_HF),),
    'set_float32_matmul_precision': (Spec(None, 'tf32', smap=(('high', 'on'), ('medium', 'on'), ('highest', 'off')),
                                          pos=0),),
    'enable_xformers_memory_efficient_attention': (Spec(None, 'attention_backend', 'occurs', 'xformers', None),),
    'FullyShardedDataParallel': (Spec('sharding_strategy', 'fsdp_sharding', smap=SHARDING),),
    'fully_shard': (Spec('reshard_after_forward', 'reshard_after_forward', 'truth', 'on', 'off'),),
    'Trainer': (Spec('precision', 'precision', smap=LIGHTNING), Spec('accumulate_grad_batches', None, role='ga')),
}
CALLS['DDP'], CALLS['FSDP'] = CALLS['DistributedDataParallel'], CALLS['FullyShardedDataParallel']
# the module a sink's name must come from (re.compile is not torch.compile), and whether a method of a local value
# counts (model.compile() does; trainer.checkpoint() is a save)
QUALIFIED = {'compile': (('torch',), True), 'checkpoint': (('torch.utils.checkpoint',), False),
             'checkpoint_sequential': (('torch.utils.checkpoint',), False), 'autocast': (('torch',), False),
             'set_float32_matmul_precision': (('torch',), False), 'Accelerator': (('accelerate',), False),
             'Trainer': (('lightning', 'pytorch_lightning'), False), 'no_sync': ((), True)}
ASSIGNS = {'allow_tf32': Spec(None, 'tf32', 'truth', 'on', 'off'), 'benchmark': Spec(None, 'cudnn_benchmark', 'truth', 'on', 'off')}
DICT_KEYS = {('zero_optimization', 'stage'): Spec(None, 'zero_stage', role='zero'),
             ('offload_optimizer', 'device'): Spec(None, 'cpu_offload', smap=(('cpu', 'optimizer'), ('nvme', 'optimizer'),
                                                                               ('none', 'none'))),
             ('bf16', 'enabled'): Spec(None, 'precision', 'truth', 'bf16', None),
             ('fp16', 'enabled'): Spec(None, 'precision', 'truth', 'fp16', None),
             ('train_micro_batch_size_per_gpu',): Spec(None, 'micro_batch', role='mbs')}
CASTS = {'bool': bool, 'int': int, 'float': float, 'str': str}
LAUNCHER_ENV = {'WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'NODE_RANK', 'MASTER_ADDR',
                'MASTER_PORT', 'CUDA_VISIBLE_DEVICES'}       # set by the launcher: the topology, not a setting


class Fn:
    """How a setting's value becomes an expression's value, as steps over the value (UNKNOWN: cannot tell); `lits` are
    the literals the code compares the setting with or looks it up by (its likely values); `opaque`: the expression
    is built from the setting by a call init cannot follow (a DataLoader of 4 workers is not the number 4)."""
    __slots__ = ('steps', 'lits', 'opaque')

    def __init__(self, steps=(), lits=(), opaque=False):
        self.steps, self.lits, self.opaque = tuple(steps), tuple(lits), opaque

    def then(self, step, lits=()):
        return Fn(self.steps + (step,), self.lits + tuple(lits), self.opaque)

    def __call__(self, v):
        for s in self.steps:
            if v is UNKNOWN:
                break
            try:
                v = s(v)
            except (TypeError, ValueError, KeyError, AttributeError):    # the code's own operation on a foreign value
                return UNKNOWN
        return v

    @property
    def ident(self) -> bool:
        return not self.steps


IDENT = Fn()


def opaque(lits=()):
    return Fn((lambda v: UNKNOWN,), lits, True)


def truth(x):
    return UNKNOWN if x is UNKNOWN else bool(x)


def deciding(flows) -> bool:
    """Whether flows carry a setting's own value - not an object built from one, not the launcher's topology."""
    return any(not f.opaque and not (s[0] == 'env' and s[1] in LAUNCHER_ENV) for s, f in flows)


def dtype_state(v):
    """'bf16' for torch.bfloat16, 'bfloat16' or 'bf16' (a dtype as code writes it), else None."""
    return DTYPES.get(v.split('.')[-1].lower()) if isinstance(v, str) else None


def lit(node):
    """The value of a literal expression - a constant, a tuple or list of them, a torch attribute ('torch.bfloat16') -
    else UNKNOWN."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        vals = [lit(e) for e in node.elts]
        return UNKNOWN if any(v is UNKNOWN for v in vals) else tuple(vals)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'torch':
        return f'torch.{node.attr}'
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return -node.operand.value
    return UNKNOWN


def compare(op, a, b):
    ops = {ast.Eq: lambda: a == b, ast.NotEq: lambda: a != b, ast.In: lambda: a in b, ast.NotIn: lambda: a not in b,
           ast.Is: lambda: a is b, ast.IsNot: lambda: a is not b, ast.Lt: lambda: a < b, ast.LtE: lambda: a <= b,
           ast.Gt: lambda: a > b, ast.GtE: lambda: a >= b}
    try:
        return ops[type(op)]()
    except (KeyError, TypeError):
        return UNKNOWN


def lits_of(v):
    return tuple(v) if isinstance(v, tuple) else (v,)


def env_read(node):
    """(name, default) of os.environ.get('X', d) / os.getenv('X', d) / os.environ['X'], else None."""
    if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) and \
            isinstance(node.args[0].value, str) and ast.unparse(node.func).endswith(('environ.get', 'getenv')):
        d = lit(node.args[1]) if len(node.args) > 1 else None
        return node.args[0].value, (None if d is UNKNOWN else d)
    if isinstance(node, ast.Subscript) and ast.unparse(node.value).endswith('environ') and \
            isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.slice.value, None
    return None


def chain(node):
    """('a', 'b', 'c') for a.b.c, cfg['train']['workers'], getattr(args, 'x', d), cfg.get('x', d); else None."""
    parts = []
    while True:
        if isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            parts.append(node.slice.value)
            node = node.value
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'getattr' and \
                len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
            parts.append(node.args[1].value)
            node = node.args[0]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'get' and \
                node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            parts.append(node.args[0].value)
            node = node.func.value
        elif isinstance(node, ast.Name):
            return (node.id, *reversed(parts))
        else:
            return None


def qualname(func, imports) -> str | None:
    """The dotted name a call reaches through the module's imports (torch.utils.checkpoint.checkpoint); '' for a method
    of a local value (model.compile()); None for a bare name not imported (a local def or a builtin)."""
    attrs, node = [], func
    while isinstance(node, ast.Attribute):
        attrs.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ''
    if node.id in imports:
        mod, orig, _ = imports[node.id]
        base = f'{mod}.{orig}' if orig and mod else (orig or mod)
        return '.'.join([base, *reversed(attrs)])
    return '' if attrs else None


def statements(stmts, conds, visit):
    """visit(node, conditions) for each statement under stmts (not nested defs), with the (test, branch) pairs of the
    `if`s it runs under."""
    for s in stmts:
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(s, ast.If):
            visit(s.test, conds)
            statements(s.body, conds + [(s.test, True)], visit)
            statements(s.orelse, conds + [(s.test, False)], visit)
        elif isinstance(s, (ast.For, ast.AsyncFor, ast.While)):
            visit(s.iter if isinstance(s, (ast.For, ast.AsyncFor)) else s.test, conds)
            statements(s.body, conds, visit)
            statements(s.orelse, conds, visit)
        elif isinstance(s, (ast.With, ast.AsyncWith)):
            for it in s.items:
                visit(it.context_expr, conds)
            statements(s.body, conds, visit)
        elif isinstance(s, ast.Try) or type(s).__name__ == 'TryStar':
            statements(s.body, conds, visit)
            for h in s.handlers:
                statements(h.body, conds, visit)
            statements(s.orelse, conds, visit)
            statements(s.finalbody, conds, visit)
        else:
            visit(s, conds)


SHARED = (ast.expr_context, ast.operator, ast.boolop, ast.cmpop, ast.unaryop)   # one instance for all nodes


def marks(node) -> set:
    """The ids of the nodes under node (not the Load / Add / Eq instances every expression shares)."""
    return {id(x) for x in ast.walk(node) if not isinstance(x, SHARED)}


def own_nodes(body):
    """The nodes of a body, not those of the functions, lambdas and classes it defines."""
    stack = list(body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        yield n
        stack.extend(ast.iter_child_nodes(n))


def nodes(node, conds, out):
    """The calls, assignments and dict literals under an expression or simple statement, with their conditions (an
    `a if t else b` adds t to its branches)."""
    if isinstance(node, ast.IfExp):
        nodes(node.test, conds, out)
        nodes(node.body, conds + [(node.test, True)], out)
        nodes(node.orelse, conds + [(node.test, False)], out)
        return
    if isinstance(node, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return
    if isinstance(node, (ast.Call, ast.Dict, ast.Assign)):
        out.append((node, conds))
    for c in ast.iter_child_nodes(node):
        nodes(c, conds, out)


@dataclass
class Module:
    path: Path
    tree: ast.Module
    imports: dict
    consts: dict                                         # module-level NAME = literal
    repo: bool                                           # the job's own file (its constants are settings)


@dataclass
class Use:
    """A setting that controls a speed call: `source` ('chain', ('args', 'compile')) | ('env', NAME, default) |
    ('const', NAME, path, value); `fn` maps the setting's value to the call's argument (value, truth) or to whether
    the call runs (occurs)."""
    spec: Spec
    mode: str
    source: tuple
    fn: Fn
    on: str | None
    off: str | None
    where: tuple                                         # (path, line)
    what: str                                            # the call, for notes
    gated: bool = False                                  # another setting decides whether the call runs at all

    @property
    def mechanism(self):
        return self.spec.mechanism

    def state(self, v):
        """The mechanism state this value of the setting gives; None when it cannot tell."""
        if self.gated:
            return None                                  # if args.enable: torch.compile(m, disable=not args.fast)
        x = self.fn(v)
        if x is UNKNOWN:
            return None
        if self.mode in ('occurs', 'truth'):
            t = truth(x)
            return self.on if t is True else self.off if t is False else None
        return smap_state(self.spec, x)


def smap_state(spec, x):
    if spec.smap == 'dtype':
        return dtype_state(x)
    if spec.smap:
        return dict(spec.smap).get(str(x).split('.')[-1] if not isinstance(x, bool) else x)
    return str(x) if isinstance(x, (int, str)) and not isinstance(x, bool) else None


@dataclass
class Trace:
    uses: list = field(default_factory=list)
    fixed: dict = field(default_factory=dict)            # mechanism -> (state, path, line): set in code, no setting
    # source -> how the code uses it: speed (a speed call's argument or condition), log, pass (into a local or a traced
    # function, whose own uses count), other (anything else: what is trained, data, loss, ...)
    contexts: dict = field(default_factory=dict)


LOGS = re.compile(r'(?i)^(print|log|info|debug|warning|warn|error|exception|critical|meta|echo|add_scalar|log_\w*|'
                  r'set_postfix|set_description)$')


class Scope:
    """A function (or a module body) as the tracer reads it: its local assignments and what its parameters receive."""

    def __init__(self, tracer, mod, fn, cls):
        self.tracer, self.mod, self.fn, self.cls = tracer, mod, fn, cls
        self.cls_key = (str(mod.path), cls.name) if cls is not None else None
        self.assigns = {}
        for s in own_nodes(fn.body if fn is not None else mod.tree.body):
            if isinstance(s, ast.Assign):
                for t in s.targets:
                    if isinstance(t, ast.Name):
                        self.assigns.setdefault(t.id, []).append(s.value)
                    elif isinstance(t, ast.Tuple) and isinstance(s.value, ast.Tuple) and len(t.elts) == len(s.value.elts):
                        for a, b in zip(t.elts, s.value.elts):
                            if isinstance(a, ast.Name):
                                self.assigns.setdefault(a.id, []).append(b)
            elif isinstance(s, (ast.AnnAssign, ast.NamedExpr)) and isinstance(s.target, ast.Name) and s.value is not None:
                self.assigns.setdefault(s.target.id, []).append(s.value)
        self.busy = set()

    @property
    def binds(self):
        return self.tracer.binds.get(id(self.fn), {}) if self.fn is not None else {}

    def name_flows(self, name, depth):
        if name in self.busy:
            return []
        self.busy.add(name)
        try:
            out = [x for x in self.binds.get(name, {}).values()]
            values = self.assigns.get(name, [])
            for a in values:
                out += self.flow(a, depth + 1)
            only_const = all(lit(a) is not UNKNOWN for a in values) and (self.fn is None or not values)
            if not out and only_const and name in self.mod.consts and self.mod.repo:
                out = [(('const', name, str(self.mod.path), self.mod.consts[name]), IDENT)]
            return out
        finally:
            self.busy.discard(name)

    def chains(self, ch, depth):
        """The settings a chain names here: through a parameter bound to the job's args, `t = cfg['train']`, a `self`
        field holding the settings; else the chain itself (unless its root is a module or a builtin)."""
        root, rest = ch[0], ch[1:]
        if root in self.mod.imports or (hasattr(builtins, root) and root not in self.assigns):
            return []
        out = []
        if root == 'self' and self.cls_key and rest:
            for s, f in self.tracer.fields.get(self.cls_key, {}).get(rest[0], {}).values():
                if s[0] == 'chain' and f.ident and rest[1:]:
                    out.append(('chain', s[1] + rest[1:]))
        for s, f in self.binds.get(root, {}).values():
            if s[0] == 'chain' and f.ident:
                out.append(('chain', s[1] + rest))
        for a in self.assigns.get(root, []):
            sub = chain(a)
            if sub is not None and depth < 4 and sub[0] != root:
                out += self.chains(sub + rest, depth + 1)
        return out or [('chain', ch)]

    def flow(self, e, depth=0):
        """[(source, Fn)]: the settings an expression's value depends on, and how."""
        if depth > 8 or e is None:
            return []
        env = env_read(e)
        if env is not None:
            return [(('env', *env), IDENT)]
        if isinstance(e, ast.Constant):
            return []
        if isinstance(e, ast.Name):
            return self.name_flows(e.id, depth)
        if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) and e.value.id == 'self' and self.cls_key:
            got = list(self.tracer.fields.get(self.cls_key, {}).get(e.attr, {}).values())
            if got:
                return got
        ch = chain(e)
        if ch is not None and len(ch) > 1:
            return [(s, IDENT) for s in self.chains(ch, depth)]
        if isinstance(e, ast.Call):
            name = getattr(e.func, 'id', None) or getattr(e.func, 'attr', None)
            if name in CASTS and len(e.args) == 1 and not e.keywords:
                return [(s, f.then(CASTS[name])) for s, f in self.flow(e.args[0], depth + 1)]
            if name == 'getattr' and len(e.args) >= 2 and isinstance(e.args[0], ast.Name) and e.args[0].id == 'torch':
                return [(s, f.then(lambda v: f'torch.{v}')) for s, f in self.flow(e.args[1], depth + 1)]   # a dtype by name
            if isinstance(e.func, ast.Attribute) and name in ('lower', 'upper', 'strip') and not e.args:
                return [(s, f.then(lambda v, m=name: getattr(str(v), m)())) for s, f in self.flow(e.func.value, depth + 1)]
            inner = list(e.args) + [k.value for k in e.keywords]
            return [(s, opaque(f.lits)) for a in inner for s, f in self.flow(a, depth + 1)]
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.Not):
            return [(s, f.then(lambda v: not v)) for s, f in self.flow(e.operand, depth + 1)]
        if isinstance(e, ast.BoolOp):
            out, rest = [], [lit(v) for v in e.values[1:]]
            flows = [self.flow(v, depth + 1) for v in e.values]
            for i, fl in enumerate(flows):                # another setting decides too (not boundary = step % accum)
                others = any(deciding(g) for j, g in enumerate(flows) if j != i)
                for s, f in fl:
                    if isinstance(e.op, ast.Or) and i == 0 and len(rest) == 1 and rest[0] is not UNKNOWN:
                        out.append((s, f.then(lambda x, d=rest[0]: x or d)))      # args.workers or 4
                    elif not others:
                        out.append((s, f.then(bool)))     # the other operands are runtime state (not boundary)
                    elif isinstance(e.op, ast.And):
                        out.append((s, f.then(lambda x: UNKNOWN if x else False)))
                    else:
                        out.append((s, f.then(lambda x: True if x else UNKNOWN)))
            return out
        if isinstance(e, ast.Compare) and len(e.ops) == 1:
            op, left, right = e.ops[0], e.left, e.comparators[0]
            lv, rv = lit(left), lit(right)
            out = []
            if rv is not UNKNOWN:
                out += [(s, f.then(lambda x, op=op, rv=rv: compare(op, x, rv), lits_of(rv))) for s, f in self.flow(left, depth + 1)]
            if lv is not UNKNOWN:
                out += [(s, f.then(lambda x, op=op, lv=lv: compare(op, lv, x), lits_of(lv))) for s, f in self.flow(right, depth + 1)]
            if lv is UNKNOWN and rv is UNKNOWN:
                out += [(s, opaque(f.lits)) for side in (left, right) for s, f in self.flow(side, depth + 1)]
            return out
        if isinstance(e, ast.IfExp):
            b, o = lit(e.body), lit(e.orelse)
            tests = self.flow(e.test, depth + 1)
            out = [(s, Fn((lambda v, f=f, b=b, o=o: UNKNOWN if truth(f(v)) is UNKNOWN else (b if truth(f(v)) else o),), f.lits))
                   for s, f in tests]
            for branch, pol in ((e.body, True), (e.orelse, False)):
                for s, g in self.flow(branch, depth + 1):
                    t = next((f for s2, f in tests if s2 == s), None)
                    out.append((s, g) if t is None else
                               (s, Fn((lambda v, g=g, t=t, pol=pol: g(v) if truth(t(v)) is pol else UNKNOWN,), g.lits + t.lits)))
            return out
        if isinstance(e, ast.Subscript) and isinstance(e.value, ast.Dict):
            table = {lit(k): lit(v) for k, v in zip(e.value.keys, e.value.values) if k is not None}
            if UNKNOWN not in table:
                return [(s, f.then(lambda x, m=table: m[x] if x in m else UNKNOWN, tuple(table))) for s, f in
                        self.flow(e.slice, depth + 1)]
        return [(s, opaque(f.lits)) for c in ast.iter_child_nodes(e) for s, f in self.flow(c, depth + 1)]

    def keyword(self, call, name, pos=None):
        """The expression passed as `name` (or at `pos`), also through **dict(...) / **{...} / **name bound to one."""
        for k in call.keywords:
            if name is not None and k.arg == name:
                return k.value
        if pos is not None and len(call.args) > pos and not any(isinstance(a, ast.Starred) for a in call.args[:pos + 1]):
            return call.args[pos]
        for k in call.keywords:
            if k.arg is None and name is not None:
                for key, v in self.dict_items(k.value):
                    if key == name:
                        return v
        return None

    def settings_object(self, e):
        """The chain of an argument that hands over the job's settings as a whole (`args`, `self.cfg`, `cfg['train']`):
        a name the code reads settings from; None for a single value."""
        ch = chain(e)
        if ch is None or ch[0] in self.mod.imports or not SETTINGS_ROOT.match(ch[-1]):
            return None
        return ch

    def dict_items(self, e, depth=0):
        if isinstance(e, ast.Name) and depth < 3:
            return [kv for a in self.assigns.get(e.id, []) for kv in self.dict_items(a, depth + 1)]
        if isinstance(e, ast.Dict):
            return [(lit(k), v) for k, v in zip(e.keys, e.values) if k is not None]
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id == 'dict':
            return [(k.arg, k.value) for k in e.keywords if k.arg]
        return []


class Tracer:
    def __init__(self, root: Path, limit: int = 160):
        self.root, self.limit = root, limit
        self.mods, self.binds, self.fields = {}, {}, {}
        self.reached = {}                                # id(fn or module tree) -> (module, fn, cls, depth)
        self.classes = set()                             # (path, class name) instantiated by the job
        self.followed = set()                            # id of the calls whose callee is traced

    def module(self, path: Path):
        from .detect import _imports, _tree
        key = str(path)
        if key not in self.mods:
            tree = _tree(path)
            if tree is None:
                self.mods[key] = None
            else:
                consts = {t.id: lit(s.value) for s in tree.body if isinstance(s, ast.Assign) and lit(s.value) is not UNKNOWN
                          for t in s.targets if isinstance(t, ast.Name)}
                self.mods[key] = Module(path, tree, _imports(tree), consts, self.inside(path))
        return self.mods[key]

    def reach(self, mod, fn, cls, depth):
        key = id(fn) if fn is not None else id(mod.tree)
        foreign = sum(not m.repo for m, *_ in self.reached.values())
        if key not in self.reached and len(self.reached) < self.limit and (mod.repo or foreign < 40):
            self.reached[key] = (mod, fn, cls, depth)    # (a library's: 40 at most - transformers' Trainer has 200)

    def inside(self, path: Path) -> bool:
        return path.resolve().is_relative_to(self.root.resolve())

    def unit(self, mod, node):
        """(module, function or __init__, its class, a class being instantiated)."""
        if isinstance(node, ast.ClassDef):
            init = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '__init__'), None)
            return mod, init, node, True
        cls = next((c for c in mod.tree.body if isinstance(c, ast.ClassDef) and node in c.body), None)
        return mod, node, cls, False

    def callee(self, call, scope, library: bool):
        """What a call reaches in readable source - a function of the job, a method of its class, a class it builds -
        and a library's function or class only when `library` (the call hands over the job's settings)."""
        from .detect import find_def, module_file
        f, imp, mod = call.func, scope.mod.imports, scope.mod
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == 'self' and scope.cls is not None:
            m = next((n for n in scope.cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == f.attr), None)
            return (mod, m, scope.cls, False) if m is not None else None
        if isinstance(f, ast.Name):
            local = next((n for n in mod.tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                          and n.name == f.id), None)
            if local is not None:
                return self.unit(mod, local)
            if hasattr(builtins, f.id):
                return None
            targets = [imp[f.id][::2] + (imp[f.id][1],)] if f.id in imp and imp[f.id][1] is not None else \
                [(m, lv, f.id) for m, lv in imp['*']]       # (module, level, name): an import, else the star imports
        elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in imp:
            m, sub, lv = imp[f.value.id]
            targets = [(f'{m}.{sub}' if sub and m else sub or m, lv, f.attr)]
        else:
            return None
        for m, lv, name in targets:
            if not lv and m.split('.')[0] in sys.stdlib_module_names:
                continue
            src = module_file(m, mod.path, self.root, lv)
            if src is None or (not library and not self.inside(src)):
                continue
            hit = find_def(src, name, self.root)
            target = self.module(hit[1]) if hit else None
            if target is not None:
                return self.unit(target, hit[0])
        return None

    def bind(self, call, scope, fn, method):
        """What a call passes to fn's parameters, as flows of the caller; True when that adds something."""
        params = [p.arg for p in fn.args.posonlyargs + fn.args.args]
        if method and params and params[0] in ('self', 'cls'):
            params = params[1:]
        named = set(params) | {p.arg for p in fn.args.kwonlyargs}
        pairs = [(params[i], a) for i, a in enumerate(call.args) if i < len(params) and
                 not any(isinstance(b, ast.Starred) for b in call.args[:i + 1])]
        for k in call.keywords:
            if k.arg in named:
                pairs.append((k.arg, k.value))
            elif k.arg is None:
                pairs += [(key, v) for key, v in scope.dict_items(k.value) if key in named]
        b, grew = self.binds.setdefault(id(fn), {}), False
        for p, a in pairs:
            whole = scope.settings_object(a)               # the settings as a whole: the callee reads from it
            items = scope.flow(a) + ([(s, IDENT) for s in scope.chains(whole, 0) if s[0] == 'chain'] if whole else [])
            for i, (s, f) in enumerate(items):
                key = (id(a), s, i)
                if key not in b.setdefault(p, {}):
                    b[p][key] = (s, f)
                    grew = True
        return grew

    def calls(self, scope, depth):
        """Follow the calls of a scope: the job's own functions and classes always, a library's when it is handed the
        settings (DiffSynth's launch_training_task(..., args=args)), not torch's for a single value; a class's methods
        once it is built."""
        grew = False
        for node in own_nodes(scope.fn.body if scope.fn is not None else scope.mod.tree.body):
            if not isinstance(node, ast.Call):
                continue
            handed = any(scope.settings_object(a) for a in list(node.args) + [k.value for k in node.keywords])
            got = self.callee(node, scope, library=handed and depth < 2)
            if got is None:
                continue
            mod, fn, cls, built = got
            d = depth + (0 if mod.repo else 1)
            if built and (str(mod.path), cls.name) not in self.classes:
                self.classes.add((str(mod.path), cls.name))
                for m in cls.body if mod.repo else ():     # the job's class: forward() and the rest run too; a
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):      # library's: what __init__ calls
                        self.reach(mod, m, cls, d)
                grew = True
            if fn is not None:
                self.reach(mod, fn, cls, d)
                self.followed.add(id(node))
                grew |= self.bind(node, scope, fn, cls is not None)
        return grew

    def collect_fields(self):
        for mod, fn, cls, _ in list(self.reached.values()):
            if cls is None or fn is None:
                continue
            scope = Scope(self, mod, fn, cls)
            got = self.fields.setdefault(scope.cls_key, {})
            for s in ast.walk(fn):
                if isinstance(s, ast.Assign):
                    for t in s.targets:
                        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'self':
                            for i, (src, f) in enumerate(scope.flow(s.value)):
                                got.setdefault(t.attr, {})[(id(s), src, i)] = (src, f)


def trace(entry: Path, root: Path, cli=()) -> Trace:
    """The speed calls of the job whose entry script is `entry`, and the settings that control them; `cli`: the entry
    functions whose parameters are command-line options (click, typer, fire)."""
    tr = Tracer(root)
    mod = tr.module(entry)
    out = Trace()
    if mod is None:
        return out
    tr.reach(mod, None, None, 0)
    for n in mod.tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            tr.reach(mod, n, None, 0)
            for a in (n.args.args + n.args.kwonlyargs) if n.name in cli else ():
                tr.binds.setdefault(id(n), {}).setdefault(a.arg, {})[('cli', a.arg)] = (('chain', ('cli', a.arg)), IDENT)
    for _ in range(4):                                   # bindings and fields to a fixed point (bounded)
        grew = False
        for m, fn, cls, depth in list(tr.reached.values()):
            grew |= tr.calls(Scope(tr, m, fn, cls), depth)
        tr.collect_fields()
        if not grew:
            break
    for m, fn, cls, _ in list(tr.reached.values()):
        scope = Scope(tr, m, fn, cls)
        classify(scope, out, *sinks(scope, out), tr.followed)
    return out


TRANSPARENT = set(CASTS) | {'dict', 'list', 'tuple'}  # calls a setting passes through as itself (dict(static_graph=x))


def classify(scope, out: Trace, regions: set, levers: set, followed: set):
    """How each setting is used here (Trace.contexts): a setting is a speed setting only when every use is - init may
    then draft it under any name (`--fast` gating torch.compile), but not --train_text_encoder, which also decides
    what is trained though it gates a text encoder's gradient checkpointing. `regions`: the speed arguments of speed
    calls; `levers`: the calls and assignments that apply a lever as a whole (torch.compile(m), allow_tf32 = ...)."""
    body = scope.fn.body if scope.fn is not None else scope.mod.tree.body
    parent = {}
    for n in own_nodes(body):
        for c in ast.iter_child_nodes(n):
            parent[id(c)] = n

    def name(call):
        return getattr(call.func, 'id', None) or getattr(call.func, 'attr', None) or ''

    def lever_or_same(e):                                # torch.compile(m) if x else m; no_sync() if x else nullcontext()
        for a, b in ((e.body, e.orelse), (e.orelse, e.body)):
            if id(a) in levers and (lit(b) is not UNKNOWN or (isinstance(b, ast.Call) and name(b) in ('nullcontext', 'enable_grad')) or
                                    (isinstance(a, ast.Call) and a.args and ast.unparse(a.args[0]) == ast.unparse(b))):
                return True
        return False                                     # not get_peft_model(m) if x else torch.compile(m)

    def quiet(s):                                        # a statement that only applies a lever, or logs
        if isinstance(s, ast.If):
            return all(quiet(x) for x in s.body + s.orelse)
        if isinstance(s, ast.Pass) or id(s) in levers:
            return True
        v = s.value if isinstance(s, (ast.Expr, ast.Assign, ast.AnnAssign, ast.AugAssign)) else None
        if isinstance(v, ast.Call):                      # model = torch.compile(model); te.gradient_checkpointing_enable()
            return id(v) in levers or bool(LOGS.match(name(v)))
        return isinstance(v, ast.IfExp) and lever_or_same(v)

    def context(n):
        x = n
        while id(x) in parent:
            if id(x) in regions:
                return 'speed'
            p = parent[id(x)]
            if isinstance(p, ast.IfExp) and x is p.test and (id(p.body) in levers or id(p.orelse) in levers):
                return 'speed' if lever_or_same(p) else 'other'   # torch.compile(m) if fast else m; not ... else lora(m)
            if isinstance(p, (ast.If, ast.While)) and x is p.test:
                return 'speed' if all(quiet(s) for s in p.body + p.orelse) else 'other'
            if isinstance(p, ast.Call) and x is not p.func and LOGS.match(name(p)):
                return 'log'                             # an f-string counts by its consumer: print, not load_dataset
            if isinstance(p, ast.Call) and x is not p.func:
                if id(p) in followed:
                    return 'pass'                        # into a traced function: its own uses count
                if name(p) not in TRANSPARENT:
                    return 'other'                       # an argument of anything else: AdamW(params_with_te, ...)
            if isinstance(p, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) and x is p.value and \
                    all(isinstance(t, ast.Name) for t in (p.targets if isinstance(p, ast.Assign) else [p.target])):
                return 'pass'                            # into a local: its own uses count
            if isinstance(p, ast.stmt):
                return 'other'
            x = p
        return 'speed' if id(x) in regions else 'other'

    for n in own_nodes(body):
        p = parent.get(id(n))
        if isinstance(p, (ast.Attribute, ast.Subscript)) and p.value is n or isinstance(p, ast.Call) and p.func is n:
            continue                                     # a part of a longer chain, or a called name
        if not (isinstance(n, (ast.Attribute, ast.Subscript)) or (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))
                or env_read(n) is not None):
            continue
        srcs = {s for s, f in scope.flow(n) if not f.opaque}   # the setting's value, not an object built from it
        if srcs:
            ctx = context(n)
            for s in srcs:
                out.contexts.setdefault(s, set()).add(ctx)


def sinks(scope, out: Trace) -> tuple[set, set]:
    """The speed calls of a scope as Uses; returns classify's (speed regions, levers)."""
    found, regions, levers = [], set(), set()
    statements(scope.fn.body if scope.fn is not None else scope.mod.tree.body, [], lambda n, c: nodes(n, c, found))
    for node, conds in found:
        where = (scope.mod.path, getattr(node, 'lineno', 0))
        if isinstance(node, ast.Call):
            name = getattr(node.func, 'id', None) or getattr(node.func, 'attr', None)
            q = QUALIFIED.get(name)
            if name not in CALLS:
                continue
            if q is not None:
                qual = qualname(node.func, scope.mod.imports)
                if not ((qual and qual.startswith(q[0]) and q[0]) or (q[1] and qual == '')):
                    continue
            specs = CALLS[name]
            if name == 'compile' and scope.keyword(node, 'disable') is not None:
                specs = [s for s in specs if s.arg == 'disable']      # torch.compile(m, disable=not args.fast)
            elif name == 'autocast' and (en := scope.keyword(node, 'enabled')) is not None and lit(en) is not True:
                specs = [s for s in specs if s.arg != 'dtype']        # its dtype counts only when it is on
            for sp in specs:
                if sp.arg is None and sp.pos is None:
                    levers.add(id(node))
                    gated(scope, out, sp, sp.on, sp.off, conds, where, f'{name}()')
                    continue
                arg = scope.keyword(node, sp.arg, sp.pos)
                if arg is None:
                    continue
                regions |= marks(arg)
                on = sp.on
                if name == 'autocast' and sp.arg == 'enabled':   # on: the call's dtype (torch's CUDA default fp16)
                    dt = scope.keyword(node, 'dtype')
                    on = dtype_state(lit(dt)) if dt is not None else 'fp16'
                setting(scope, out, sp, arg, on, conds, where, f'{name}({sp.arg or ""}=)')
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                text = ast.unparse(t)
                sp = ASSIGNS.get(text.split('.')[-1]) if isinstance(t, ast.Attribute) else None
                if sp is not None and (sp.mechanism != 'cudnn_benchmark' or 'cudnn' in text):
                    regions |= marks(node.value)
                    levers.add(id(node))
                    setting(scope, out, sp, node.value, sp.on, conds, where, f'{text} =', assign=True)
        elif isinstance(node, ast.Dict):
            for path, v in dict_paths(node):
                for keys, sp in DICT_KEYS.items():
                    if path[-len(keys):] == keys:
                        regions |= marks(v)
                        setting(scope, out, sp, v, sp.on, conds, where, '.'.join(path))
    return regions, levers


def dict_paths(d, prefix=()):
    for k, v in zip(d.keys, d.values):
        if isinstance(k, ast.Constant) and isinstance(k.value, str):
            yield prefix + (k.value,), v
            if isinstance(v, ast.Dict):
                yield from dict_paths(v, prefix + (k.value,))


def setting(scope, out, sp, arg, on, conds, where, what, assign=False):
    """A speed call's argument: the settings it depends on; a literal under conditions on settings is on exactly when
    they hold (`if args.tf32: allow_tf32 = True`); a literal under none is the job's state of the mechanism."""
    flows = scope.flow(arg)
    tests = [(pol, s2, c) for test, pol in conds for s2, c in scope.flow(test) if deciding([(s2, c)])]
    for s, f in flows:
        gate = any(s2 != s for _, s2, _ in tests)        # another setting may keep the call from running
        own = [(c, pol) for pol, s2, c in tests if s2 == s]
        if own:                                          # its own condition: the value counts where the call runs
            f = Fn((lambda v, f=f, own=own: f(v) if all(truth(c(v)) is pol for c, pol in own) else UNKNOWN,), f.lits)
        out.uses.append(Use(sp, 'value' if sp.mode == 'value' else 'truth', s, f, on, sp.off, where, what, gate))
    if flows:
        return
    v = lit(arg)
    if v is UNKNOWN:
        return
    st = (on if v else sp.off) if sp.mode == 'truth' else smap_state(sp, v)
    # not running a call says nothing of its mechanism (AdamW(fused=True) under `if name == 'fused'` may be SGD
    # otherwise); not running an assignment leaves torch's default
    if not gated(scope, out, sp, st, sp.off if assign else None, conds, where, what) and sp.mechanism and st:
        out.fixed.setdefault(sp.mechanism, (st, *where))


def gated(scope, out, sp, on, off, conds, where, what):
    """Uses for the settings in the conditions a call runs under; False when none (it always runs). A condition on
    one setting decides alone only when the others are not settings (`if args.no_sync and not boundary`): with another
    setting in the way (`if args.fast:` / `if args.inductor:`) a value that holds it says nothing (UNKNOWN)."""
    per = [(pol, scope.flow(test)) for test, pol in conds]
    found = False
    for i, (pol, flows) in enumerate(per):
        others = any(deciding(fl) for j, (_, fl) in enumerate(per) if j != i)
        for s, f in flows:
            def runs(v, f=f, pol=pol, others=others):
                t = truth(f(v))
                return t if t is UNKNOWN else False if t is not pol else UNKNOWN if others else True
            out.uses.append(Use(sp, 'occurs', s, Fn((runs,), f.lits), on, off, where, what))
            found = True
    if not found and sp.arg is None and sp.pos is None and sp.mechanism and on:
        out.fixed.setdefault(sp.mechanism, (on, *where))
    return found
