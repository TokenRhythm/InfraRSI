"""ENV-C infra-only contract checks (design §4.2): they only add rejections; the trusted judge's val_bpb is the score.

(1) static     train.py vs the baseline (AST): the frozen constant assignments, GPTConfig, build_model_config,
               setup_optimizer (parameter groups), the LR/momentum/WD schedules and their application, the prepare.py
               imports (never re-bound), the make_dataloader call, seeds 42, precision (set_float32_matmul_precision
               "high", bf16 autocast, no TF32/reduced-precision flag writes), the 300 s rule with the timer after step 10,
               the fast-fail rule and the progress-line format (its frequency may change).
(2) runtime    run_meta.json (written by tools/run_train.py after the run) and the judge's view of final_model.pt:
               constants, model config, parameter count, optimizer parameter groups, matmul precision, step/token
               consistency and the timed-region accounting; steps and tokens themselves are the target.
(4) losses     printed loss of the first 10 steps within 1e-2 relative of the baseline's at the same step (same seed,
               same data order; pre-declared [estimate], design §4.2 / risk 8).
Codex W9 H-02 (protected callables stay bound to their checked definitions):
  static       no rebinding of the protected names (GPTConfig, GPT, the frozen functions, the prepare names): one
               definition each, no stores/deletes on their attributes or items, no setattr/delattr/exec/eval/compile/
               __import__/globals/locals/vars, getattr/hasattr only with a literal attribute, no frame, code-object or
               class-internals attributes (__code__, __dict__, f_globals, ...), no sys.modules / gc.get_referrers routes and
               no import of importlib, ctypes, inspect, types, marshal, pickle, runpy, builtins (or the wrapper's modules)
  attestation  run_train.py hashes the code objects the protected callables are bound to at the end of the run
               (node/code_attest.py, outside the timed region); the trusted root step compiles the same names from the
               candidate's train.py, the frozen prepare.py and the baseline: live must equal the candidate's definitions,
               the strict ones also the baseline's, the prepare names must be the prepare module's objects, and the model
               instance must not override a protected method.
Rejection ids: C-static, C-runtime, C-loss10.
"""
from __future__ import annotations

import ast
import math
from functools import lru_cache

from ..common.guards import GuardResult
from . import logs
from .node import code_attest

FROZEN_CONSTANTS = ('ASPECT_RATIO', 'HEAD_DIM', 'WINDOW_PATTERN', 'TOTAL_BATCH_SIZE', 'EMBEDDING_LR', 'UNEMBEDDING_LR',
                    'MATRIX_LR', 'SCALAR_LR', 'WEIGHT_DECAY', 'ADAM_BETAS', 'WARMUP_RATIO', 'WARMDOWN_RATIO', 'FINAL_LR_FRAC', 'DEPTH')
PREPARE_NAMES = ('MAX_SEQ_LEN', 'TIME_BUDGET', 'Tokenizer', 'make_dataloader', 'evaluate_bpb')
FROZEN_FUNCTIONS = ('build_model_config', 'get_lr_multiplier', 'get_muon_momentum', 'get_weight_decay')
SEED_CALLS = ('manual_seed', 'manual_seed_all', 'seed', 'set_rng_state', 'set_state')
PRECISION_FLAGS = ('allow_tf32', 'allow_bf16_reduced_precision_reduction', 'allow_fp16_reduced_precision_reduction')
LOSS_TOLERANCE = 1e-2
PROTECTED = ('GPTConfig', 'GPT') + FROZEN_FUNCTIONS          # with PREPARE_NAMES and the module name 'prepare'
# Bound to the candidate's own (statically visible) definitions: one binding each, no stores on them (round 2).
PROTECTED_BOUND = ('MuonAdamW', 'adamw_step_fused', 'muon_step_fused')
# Attributes no object may have assigned or deleted (Codex W9 round 2): the optimizer and module call paths.
STORE_BANNED_ATTRS = ('step', 'zero_grad', 'param_groups', 'state', 'defaults', '__class__', '__dict__', 'forward',
                      '__call__', 'setup_optimizer', '_step_adamw', '_step_muon', 'adamw_step_fused', 'muon_step_fused',
                      '__init__', '__new__') + FROZEN_FUNCTIONS
BANNED_NAMES = ('setattr', 'delattr', 'exec', 'eval', 'compile', '__import__', 'globals', 'locals', 'vars', '__builtins__',
                'breakpoint')
BANNED_MODULES = ('importlib', 'imp', 'runpy', 'ctypes', 'builtins', 'inspect', 'types', 'marshal', 'pickle', 'dill',
                  'cloudpickle', 'code', 'codeop', 'copyreg', 'code_attest', 'run_train',
                  'operator', '_operator')                  # round 3 (R3-01): attrgetter / methodcaller / setitem / delitem
# Round 3 (R3-01): dunder attributes are reflective routes (an instance __dict__ edited through any function); only
# these plain ones may be used, and a dunder string may not be passed to any call.
ALLOWED_DUNDERS = ('__init__', '__name__', '__qualname__', '__doc__', '__version__', '__file__', '__module__')
BANNED_ATTRS = ('__code__', '__globals__', '__closure__', '__defaults__', '__kwdefaults__', '__dict__', '__class__',
                '__bases__', '__mro__', '__subclasses__', '__builtins__', '__setattr__', '__delattr__', '__getattribute__',
                '__import__', '__func__', '__self__', '__wrapped__', '__loader__', '__spec__', 'f_globals', 'f_locals',
                'f_builtins', 'f_back', 'f_code', 'tb_frame', 'gi_frame', 'cr_frame', 'ag_frame', 'get_referrers',
                'get_referents', 'get_objects', '_global_optimizer_post_hooks', '_global_optimizer_pre_hooks',
                '_optimizer_step_post_hooks', '_optimizer_step_pre_hooks', '_patch_step_function', 'profile_hook_step')
BANNED_SYS_ATTRS = ('modules', '_getframe', 'settrace', 'setprofile', 'meta_path', 'path_hooks', 'path_importer_cache')


def _dump(node) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _dotted(node) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return '.'.join([node.id] + parts[::-1])
    return None


def _module_assigns(tree) -> dict[str, list]:
    out: dict[str, list] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.setdefault(t.id, []).append(node.value)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name):
            out.setdefault(node.target.id, []).append(node)
    return out


def _bindings(tree, names: set) -> list[tuple[str, int, str]]:
    """Every binding of `names` in the module (assignment targets anywhere, imports, defs, loop/with targets, global,
    nonlocal, except-as and match captures)."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in names:
            found.append((node.id, node.lineno, 'store'))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names:
            found.append((node.name, node.lineno, 'def'))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound = (a.asname or a.name).split('.')[0]
                if bound in names:
                    found.append((bound, node.lineno, f'import from {getattr(node, "module", None)}'))
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            found += [(n, node.lineno, type(node).__name__.lower()) for n in node.names if n in names]
        elif isinstance(node, ast.ExceptHandler) and node.name in names:
            found.append((node.name, node.lineno, 'except as'))
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and getattr(node, 'name', None) in names:
            found.append((node.name, node.lineno, 'match capture'))
        elif isinstance(node, ast.MatchMapping) and node.rest in names:
            found.append((node.rest, node.lineno, 'match capture'))
    return found


def _dunder(name) -> bool:
    return isinstance(name, str) and len(name) > 4 and name.startswith('__') and name.endswith('__')


def _root(node) -> str | None:
    """The name an attribute/subscript chain starts from (GPT.setup_optimizer -> GPT, x[0].y -> x)."""
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def aliases(tree, roots: set) -> set:
    """Names that refer to a guarded root (or into one): `X = GPT`, `X = GPT.setup_optimizer`, `X, Y = GPT, 1`,
    `import prepare as P`, `def f(c=GPT)`, chained to a fixed point. Scopes are ignored (strict)."""
    found = set(roots)

    def refers(expr) -> bool:
        return isinstance(expr, (ast.Name, ast.Attribute, ast.Subscript, ast.Starred)) and _root(expr) in found

    while True:
        new = set()
        for node in ast.walk(tree):
            pairs = []
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, (ast.Tuple, ast.List)) and isinstance(node.value, (ast.Tuple, ast.List)) \
                            and len(t.elts) == len(node.value.elts):
                        pairs += list(zip(t.elts, node.value.elts))
                    else:
                        pairs.append((t, node.value))
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
                pairs.append((node.target, node.value))
            elif isinstance(node, (ast.For, ast.comprehension)):
                if isinstance(node.iter, (ast.Tuple, ast.List)) and any(refers(e) for e in node.iter.elts):
                    pairs.append((node.target, node.iter.elts[0]))
                    pairs += [(node.target, e) for e in node.iter.elts]
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                pairs.append((node.optional_vars, node.context_expr))
            elif isinstance(node, ast.Import):
                new |= {a.asname or a.name.split('.')[0] for a in node.names if a.name.split('.')[0] in found}
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                args = node.args
                pos = args.posonlyargs + args.args
                pairs += [(ast.Name(id=a.arg), d) for a, d in zip(pos[len(pos) - len(args.defaults):], args.defaults)]
                pairs += [(ast.Name(id=a.arg), d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
            for target, value in pairs:
                if isinstance(target, ast.Name) and target.id not in found and refers(value):
                    new.add(target.id)
        if not new - found:
            return found
        found |= new


def rebinding(tree, base_tree) -> list[str]:
    """Codex W9 H-02: routes that would rebind a protected callable/class behind the static comparison."""
    why = []
    guarded = aliases(tree, set(PROTECTED) | set(PROTECTED_BOUND) | set(PREPARE_NAMES) | {'prepare'})
    for name in PROTECTED + PROTECTED_BOUND:
        n_base = len(_bindings(base_tree, {name}))
        where = [f'line {line} ({kind})' for n, line, kind in _bindings(tree, {name})]
        if len(where) > n_base:
            why.append(f'{name} bound {len(where)} times ({", ".join(where[:4])}); only its definition is allowed')
    for node in ast.walk(tree):
        line = getattr(node, 'lineno', '?')
        if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
            root = _root(node)
            verb = 'deletion of' if isinstance(node.ctx, ast.Del) else 'assignment to'
            if root in guarded:
                why.append(f'line {line}: {verb} {ast.unparse(node)[:80]} (protected {root})')
            elif isinstance(node, ast.Attribute) and node.attr in STORE_BANNED_ATTRS:
                why.append(f'line {line}: {verb} {ast.unparse(node)[:80]} (attribute {node.attr} of any object)')
            elif isinstance(node, ast.Attribute) and root is None:
                why.append(f'line {line}: {verb} an attribute of a computed object ({ast.unparse(node)[:80]})')
        if isinstance(node, ast.ImportFrom) and any(a.name == '*' for a in node.names):
            why.append(f'line {line}: star import from {node.module} (it can rebind protected names)')
        if isinstance(node, ast.Name):
            if node.id in BANNED_NAMES:
                why.append(f'line {line}: {node.id} is not allowed (dynamic rebinding route)')
            elif isinstance(node.ctx, (ast.Store, ast.Del)) and node.id.startswith('__') and node.id.endswith('__'):
                why.append(f'line {line}: binding the dunder name {node.id} is not allowed')
        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_ATTRS or (_dunder(node.attr) and node.attr not in ALLOWED_DUNDERS):
                why.append(f'line {line}: attribute {node.attr} is not allowed ({ast.unparse(node)[:80]})')
            elif node.attr in BANNED_SYS_ATTRS and isinstance(node.value, ast.Name) and node.value.id == 'sys':
                why.append(f'line {line}: sys.{node.attr} is not allowed')
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or '']
            for m in mods:
                if m.split('.')[0] in BANNED_MODULES:
                    why.append(f'line {line}: import of {m} is not allowed')
            if isinstance(node, ast.ImportFrom):
                top = (node.module or '').split('.')[0]
                banned = BANNED_SYS_ATTRS if top == 'sys' else (BANNED_ATTRS if top == 'gc' else ())
                bad = [a.name for a in node.names if a.name in banned or (banned and a.name == '*')]
                if bad:
                    why.append(f'line {line}: from {top} import {", ".join(bad)} is not allowed')
        if isinstance(node, ast.Call):
            strings = [a.value for a in list(node.args) + [k.value for k in node.keywords]
                       if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            bad = [s for s in strings if _dunder(s) and s not in ALLOWED_DUNDERS]
            if bad:
                why.append(f'line {line}: dunder string {bad[0]!r} passed to {ast.unparse(node.func)[:60]} (reflection)')
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('getattr', 'hasattr'):
            attr = node.args[1] if len(node.args) > 1 else None
            if not (isinstance(attr, ast.Constant) and isinstance(attr.value, str)):
                why.append(f'line {line}: {node.func.id} needs a literal attribute name')
            elif attr.value in BANNED_ATTRS or (attr.value in BANNED_SYS_ATTRS and _root(node.args[0]) == 'sys'):
                why.append(f'line {line}: {node.func.id}(..., {attr.value!r}) is not allowed')
    return why


def _top(tree, kind, name):
    for node in tree.body:
        if isinstance(node, kind) and node.name == name:
            return node
    return None


def _method(tree, cls: str, name: str):
    c = _top(tree, ast.ClassDef, cls)
    return next((f for f in (c.body if c else []) if isinstance(f, ast.FunctionDef) and f.name == name), None)


def _without_prints(fn):
    fn = ast.parse(ast.unparse(fn)).body[0]
    fn.body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                                          and isinstance(s.value.func, ast.Name) and s.value.func.id == 'print')]
    return fn


def _calls(tree, attr: str) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and (
        (isinstance(n.func, ast.Attribute) and n.func.attr == attr) or (isinstance(n.func, ast.Name) and n.func.id == attr))]


def _has_stmt(tree, stmt) -> bool:
    want = _dump(stmt)
    return any(_dump(n) == want for n in ast.walk(tree) if isinstance(n, type(stmt)))


def baseline_statements(base_tree) -> dict:
    """Statements of the baseline that must appear verbatim in a candidate (wherever they are)."""
    want, loop = {}, None
    for node in ast.walk(base_tree):
        if isinstance(node, ast.While):
            loop = node
            break
    for node in ast.walk(loop):
        if isinstance(node, ast.If):
            src = ast.unparse(node.test)
            if src == 'step > 10':
                want['timer_after_step_10'] = node
            elif src == 'step > 10 and total_training_time >= TIME_BUDGET':
                want['budget_stop'] = node
            elif src.startswith('math.isnan(train_loss_f)'):
                want['fast_fail'] = node
        elif isinstance(node, ast.For) and ast.unparse(node.iter) == 'optimizer.param_groups':
            want['schedule_application'] = node
        elif isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'progress':
            want['progress'] = node
    for node in base_tree.body:
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'autocast_ctx':
            want['bf16_autocast'] = node
    fmt = [n for n in ast.walk(base_tree) if isinstance(n, ast.JoinedStr) and 'step ' in ast.unparse(n) and 'loss:' in ast.unparse(n)]
    want['progress_line_format'] = fmt[0]
    want['make_dataloader_train'] = next(c for c in _calls(base_tree, 'make_dataloader'))
    return want


@lru_cache(maxsize=4)
def _baseline(base_text: str):
    b = ast.parse(base_text)
    return b, baseline_statements(b)


def static(base_text: str, cand_text: str) -> GuardResult:
    g = GuardResult()
    try:
        b, want = _baseline(base_text)
        c = ast.parse(cand_text)
    except SyntaxError as exc:
        return g.reject('C-static', f'train.py does not parse: line {exc.lineno}: {exc.msg}')
    ba, ca = _module_assigns(b), _module_assigns(c)
    for name in FROZEN_CONSTANTS:
        if [_dump(v) for v in ca.get(name, [])] != [_dump(v) for v in ba.get(name, [])]:
            g.reject('C-static', f'constant {name} changed (baseline: {ast.unparse(ba[name][0]) if ba.get(name) else None})')
    binds = _bindings(c, set(FROZEN_CONSTANTS))
    for name in FROZEN_CONSTANTS:
        where = [f'line {line} ({kind})' for n, line, kind in binds if n == name]
        if len(where) > 1:
            g.reject('C-static', f'{name} bound {len(where)} times ({", ".join(where[:4])}); only the baseline assignment is allowed')
    if _calls(c, 'globals'):
        g.reject('C-static', 'globals() is not allowed (frozen constants must stay literal module assignments)')
    for why in rebinding(c, b):
        g.reject('C-static', why)
    for kind, name in ((ast.ClassDef, 'GPTConfig'),) + tuple((ast.FunctionDef, f) for f in FROZEN_FUNCTIONS):
        nb, nc = _top(b, kind, name), _top(c, kind, name)
        if nc is None or _dump(nc) != _dump(nb):
            g.reject('C-static', f'{name} changed or missing')
    sb, sc = _method(b, 'GPT', 'setup_optimizer'), _method(c, 'GPT', 'setup_optimizer')
    if sc is None or _dump(_without_prints(sc)) != _dump(_without_prints(sb)):
        g.reject('C-static', 'GPT.setup_optimizer (parameter groups, LR scaling) changed')
    imported = {a.name for n in c.body if isinstance(n, ast.ImportFrom) and n.module == 'prepare' for a in n.names if a.asname is None}
    if set(PREPARE_NAMES) - imported:
        g.reject('C-static', f'must import {sorted(set(PREPARE_NAMES) - imported)} from prepare unchanged')
    rebound = [x for x in _bindings(c, set(PREPARE_NAMES)) if not x[2].startswith('import from prepare')]
    for name, line, kind in rebound:
        g.reject('C-static', f'prepare name {name} re-bound at line {line} ({kind})')
    for node in ast.walk(c):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                d = _dotted(t) or ''
                if d.startswith('prepare.'):
                    g.reject('C-static', f'assignment to {d} at line {node.lineno}')
                if d.split('.')[-1] in PRECISION_FLAGS:
                    g.reject('C-static', f'precision flag write {d} at line {node.lineno}')
    for label, stmt in want.items():
        if label == 'make_dataloader_train':
            calls = [x for x in _calls(c, 'make_dataloader') if _dump(x) == _dump(stmt)]
            if len(_calls(c, 'make_dataloader')) != 1 or len(calls) != 1:
                g.reject('C-static', 'make_dataloader must be called exactly once, as make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")')
        elif not _has_stmt(c, stmt):
            g.reject('C-static', f'baseline statement missing or changed: {label} ({ast.unparse(stmt).splitlines()[0][:100]})')
    seeds_b = sorted(ast.unparse(x) for a in SEED_CALLS for x in _calls(b, a))
    seeds_c = sorted(ast.unparse(x) for a in SEED_CALLS for x in _calls(c, a))
    if seeds_c != seeds_b:
        g.reject('C-static', f'seeding calls changed: {seeds_c} != {seeds_b}')
    prec_b = sorted(ast.unparse(x) for x in _calls(b, 'set_float32_matmul_precision'))
    prec_c = sorted(ast.unparse(x) for x in _calls(c, 'set_float32_matmul_precision'))
    if prec_c != prec_b:
        g.reject('C-static', f'set_float32_matmul_precision calls changed: {prec_c}')
    return g


# ---------------------------------------------------------------- runtime
def _const_eval(node):
    return eval(compile(ast.Expression(node), '<const>', 'eval'), {'__builtins__': {}})   # trusted baseline literals only


def expected_constants(base_text: str) -> dict:
    a = _module_assigns(ast.parse(base_text))
    out = {}
    for name in FROZEN_CONSTANTS:
        v = _const_eval(a[name][0])
        out[name] = list(v) if isinstance(v, tuple) else v
    return out


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=0.0) or a == b
    return a == b


def _group_key(g: dict) -> tuple:
    return tuple(str(g.get(k)) for k in ('kind', 'n_tensors', 'numel', 'initial_lr', 'betas', 'eps', 'ns_steps', 'beta2'))


def runtime(meta: dict | None, judge: dict | None, expected: dict) -> GuardResult:
    """expected: {'constants', 'time_budget', 'max_seq_len', 'config', 'num_params', 'optimizer_groups' (or None)}."""
    g = GuardResult()
    if not isinstance(meta, dict) or meta.get('schema') != 'ext-autoresearch-run-meta-v1':
        return g.reject('C-runtime', 'run_meta.json missing or unreadable')
    if meta.get('save_error'):
        g.reject('C-runtime', f'save tail failed: {meta["save_error"]}')
    consts, counters = meta.get('constants') or {}, meta.get('counters') or {}
    for name, want in expected['constants'].items():
        if not _same(consts.get(name), want):
            g.reject('C-runtime', f'{name} = {consts.get(name)!r} at run time (baseline {want!r})')
    for name, want in (('TIME_BUDGET', expected['time_budget']), ('MAX_SEQ_LEN', expected['max_seq_len'])):
        if consts.get(name) != want:
            g.reject('C-runtime', f'{name} = {consts.get(name)!r} (prepare.py: {want})')
    if meta.get('config') != expected['config']:
        g.reject('C-runtime', f'model config {meta.get("config")} != baseline {expected["config"]}')
    if counters.get('num_params') != expected['num_params']:
        g.reject('C-runtime', f'num_params {counters.get("num_params")} != {expected["num_params"]}')
    if isinstance(judge, dict) and judge.get('num_params') is not None and judge['num_params'] != expected['num_params']:
        g.reject('C-runtime', f'saved weights hold {judge["num_params"]} parameters')
    if expected.get('optimizer_groups') is not None:
        got = sorted(_group_key(x) for x in meta.get('optimizer_groups') or [])
        want = sorted(_group_key(x) for x in expected['optimizer_groups'])
        if got != want:
            g.reject('C-runtime', 'optimizer parameter groups differ from the baseline run')
    else:
        g.report('C-runtime', optimizer_groups='not compared (no reference run recorded)')
    if meta.get('float32_matmul_precision') != 'high':
        g.reject('C-runtime', f'float32 matmul precision {meta.get("float32_matmul_precision")!r} != high')
    step, tokens, tbs = counters.get('step'), counters.get('total_tokens'), expected['constants'].get('TOTAL_BATCH_SIZE')
    if type(step) is not int or step <= 10 or tokens != step * tbs:
        g.reject('C-runtime', f'steps/tokens inconsistent: step={step!r} total_tokens={tokens!r} (TOTAL_BATCH_SIZE {tbs})')
    t_train, wall = counters.get('total_training_time'), (meta.get('wrapper') or {}).get('train_py_wall_s')
    if not (type(t_train) in (int, float) and t_train >= expected['time_budget']):
        g.reject('C-runtime', f'timed training {t_train!r} s < TIME_BUDGET {expected["time_budget"]}')
    elif not (type(wall) in (int, float) and wall >= t_train):
        g.reject('C-runtime', f'train.py wall {wall!r} s < its timed training {t_train} s')
    g.report('C-runtime', steps=step, total_tokens=tokens, training_seconds=t_train, train_py_wall_s=wall)
    return g


def losses(log_text: str, reference: list[float], tol: float = LOSS_TOLERANCE, veto: bool = True) -> GuardResult:
    g = GuardResult()
    printed = logs.first_losses(log_text, len(reference))
    if not printed:
        g.reject('C-loss10', 'no progress line printed for steps 0-9')
        return g
    devs = {s: (abs(v - reference[s]) / abs(reference[s]) if math.isfinite(v) else math.inf) for s, v in sorted(printed.items())}
    worst = max(devs.values())
    if worst > tol and veto:
        s = max(devs, key=devs.get)
        g.reject('C-loss10', f'step {s} loss {printed[s]} vs baseline {reference[s]}: relative deviation {worst:.4g} > {tol}')
    g.report('C-loss10', worst=worst if math.isfinite(worst) else 'inf', tol=tol, steps=len(devs),
             deviations={str(k): round(v, 6) if math.isfinite(v) else 'inf' for k, v in devs.items()})
    return g


# ---------------------------------------------------------------- code attestation (Codex W9 H-02)
def attestation(live: dict | None, expect: dict | None, counters: dict | None = None) -> GuardResult:
    """live: run_meta.json `code_attestation` (the candidate's process, end of run); expect: the trusted root step's
    judge/code_expected.json (both from node/code_attest.py, same interpreter); counters: run_meta.json `counters` -
    every logical training step must have gone through the optimizer class's hooked step (Codex W9 round 3, R3-01)."""
    g = GuardResult()
    if not isinstance(live, dict) or not live:
        return g.reject('C-runtime', 'run_meta.json carries no code attestation')
    if not isinstance(expect, dict) or expect.get('schema') != 'ext-autoresearch-code-expect-v1':
        return g.reject('C-runtime', 'the trusted code expectation is missing or unreadable')
    cand, base = expect.get('candidate') or {}, expect.get('baseline') or {}
    for name, part in (('candidate', cand), ('baseline', base)):
        if 'error' in part or not part:
            return g.reject('C-runtime', f'trusted compile of the {name} failed: {part.get("error", "empty")}')
    if live.get('python') != expect.get('python'):
        g.reject('C-runtime', f'attestation interpreters differ: {live.get("python")} vs {expect.get("python")}')
    # Round 2: the same checks at the first optimizer step (the point of use) and at the end of the run.
    first = live.get('first_step')
    if not isinstance(first, dict) or first.get('error'):
        g.reject('C-runtime', 'no attestation at the first optimizer step '
                              f'(step hook: {live.get("step_hook")}; {(first or {}).get("error") if isinstance(first, dict) else "none"})')
    for where, snap in (('at the first optimizer step', first), ('at run end', live)):
        if isinstance(snap, dict) and not snap.get('error'):
            for detail in _snapshot(snap, cand, base, where):
                g.reject('C-runtime', detail)
    steps, opts = live.get('optimizer_steps'), live.get('optimizers_stepped')
    if not (type(steps) is int and steps > 0):
        g.reject('C-runtime', f'no optimizer step was observed by the step hook (steps {steps!r})')
    elif opts != 1:
        g.reject('C-runtime', f'{opts} different optimizers took steps (the baseline steps one)')
    elif live.get('same_optimizer') is not True:
        g.reject('C-runtime', 'the optimizer at run end is not the one that took the first step')
    logical = (counters or {}).get('step')
    if type(steps) is int and steps > 0:
        if type(logical) is not int:
            g.reject('C-runtime', f'no logical training-step counter to compare the {steps} hooked optimizer steps with')
        elif steps != logical:
            g.reject('C-runtime', f'{steps} optimizer steps went through the class step method, the run made {logical} '
                                  'training steps (a step was taken around the hooked method)')
    if live.get('override_steps') not in (0, None) or 'override_steps' not in live:
        g.reject('C-runtime', f'{live.get("override_steps", "unknown")} hooked steps ran while the optimizer instance '
                              'overrode a step method')
    g.report('C-attest', checked=len(cand), python=live.get('python'), optimizer_steps=steps, training_steps=logical)
    return g


def _snapshot(snap: dict, cand: dict, base: dict, where: str) -> list[str]:
    out, bad = [], []
    for key, want in sorted(cand.items()):
        if want == 'missing' or snap.get(key) != want:
            bad.append(f'{key} ({_short(snap.get(key))} {where}, {_short(want)} in the source)')
        elif key in code_attest.STRICT and base.get(key) != want:
            bad.append(f'{key} (differs from the baseline)')
    if bad:
        out.append('protected callables not bound to their checked definitions: ' + '; '.join(bad[:4]))
    bound = snap.get('bound_to_prepare') or {}
    loose = sorted(n for n in code_attest.PREPARE_BOUND if bound.get(n) is not True)
    if loose:
        out.append(f'train.py names not bound to the prepare module objects {where}: {loose}')
    if snap.get('instance_overrides'):
        out.append(f'the model instance overrides protected methods {where}: {snap["instance_overrides"]}')
    if snap.get('model_class_is_GPT') is not True:
        out.append(f'the trained model is a {snap.get("model_class")!r} {where}, not the GPT class of train.py')
    opt = snap.get('optimizer') or {}
    if not opt.get('present'):
        out.append(f'no optimizer {where}')
    elif opt.get('class') != code_attest.OPTIMIZER_CLASS or opt.get('class_is_train_def') is not True:
        out.append(f'the optimizer {where} is a {opt.get("class")!r}, not the {code_attest.OPTIMIZER_CLASS} class of train.py')
    elif opt.get('instance_overrides'):
        out.append(f'the optimizer instance overrides {opt["instance_overrides"]} {where}')
    return out


def _short(v) -> str:
    if isinstance(v, str) and len(v) == 64:
        return v[:10]
    if isinstance(v, dict):
        return '{' + ','.join(sorted(v)) + '}'
    return str(v)[:60]
