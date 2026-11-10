"""ENV-C code attestation (Codex W9 H-02): hashes of the code objects the protected callables are bound to.

  live    imported by tools/run_train.py in the candidate's process: at the first optimizer step (StepWatch, a global
          torch optimizer post-step hook registered as soon as train.py has imported torch; step 0 is outside the
          timed region) and after train.py returned - what the candidate's globals, the prepare module, the model
          and the stepping optimizer are bound to at the point of use and at the end of the run (Codex W9 round 2)
  expect  run as root by run.sh before the training (`python -I code_attest.py expect ...`): the same names compiled,
          never executed, from the candidate's train.py, the frozen prepare.py and the baseline train.py
A hash covers co_code, the constants (nested code objects by their own hash, sets in sorted order), co_names,
co_varnames, the free/cell variables, the argument counts, co_flags and the exception table - not file names or line
numbers, so moving a definition does not change it. Decorators of the installed environment (functools.wraps
wrappers such as torch.no_grad on prepare.evaluate_bpb) are looked through via __wrapped__ and listed under
`wrappers`; a wrapper whose code is not library code of the interpreter's installation is reported as foreign.
infrarsi.ext.autoresearch.infra_only_check.attestation compares live with expect: the strict callables must equal
the baseline's, GPT.setup_optimizer the candidate's own statically checked definition (it may differ from the
baseline in print lines only). Standard library only; also imported on the driver for the name lists.
"""
import hashlib
import json
import os
import sys
import types

TRAIN_FUNCS = ('build_model_config', 'get_lr_multiplier', 'get_muon_momentum', 'get_weight_decay')
TRAIN_CLASSES = ('GPTConfig',)                      # every method written in the class body
TRAIN_METHODS = {'GPT': ('setup_optimizer',)}
# The optimizer's implementation may change in source (design §4.2: fused/foreach ops, same math), so these are bound
# to the candidate's own definitions, not to the baseline's: no runtime swap of the update code is possible.
TRAIN_BOUND_FUNCS = ('adamw_step_fused', 'muon_step_fused')
TRAIN_BOUND_CLASSES = ('MuonAdamW',)                # every method written in the class body
OPTIMIZER_CLASS = 'MuonAdamW'                       # what GPT.setup_optimizer constructs (frozen)
OPTIMIZER_ATTRS = ('step', 'zero_grad', '_step_adamw', '_step_muon', '__call__')   # never instance attributes
PREPARE_FUNCS = ('make_dataloader', 'evaluate_bpb')
PREPARE_CLASSES = ('Tokenizer',)
PREPARE_BOUND = ('MAX_SEQ_LEN', 'TIME_BUDGET', 'Tokenizer', 'make_dataloader', 'evaluate_bpb')
STRICT = tuple('train:' + n for n in TRAIN_FUNCS + TRAIN_CLASSES)     # equal to the baseline's, not only to the source
CO_OPTIMIZED = 0x0001


def canon(c):
    if isinstance(c, types.CodeType):
        return 'code:' + code_hash(c)
    if isinstance(c, (tuple, list)):
        return type(c).__name__ + '(' + ','.join(canon(x) for x in c) + ')'
    if isinstance(c, (frozenset, set)):                # set order depends on the per-process string hash seed
        return type(c).__name__ + '{' + ','.join(sorted(canon(x) for x in c)) + '}'
    if isinstance(c, float) and c != c:
        return 'float:nan'
    return type(c).__name__ + ':' + repr(c)


def code_hash(co):
    h = hashlib.sha256()
    parts = [co.co_code, repr(co.co_names), repr(co.co_varnames), repr(co.co_freevars), repr(co.co_cellvars),
             repr((co.co_argcount, co.co_posonlyargcount, co.co_kwonlyargcount, co.co_flags)),
             getattr(co, 'co_exceptiontable', b''), canon(tuple(co.co_consts))]
    for p in parts:
        b = p if isinstance(p, bytes) else p.encode('utf-8')
        h.update(str(len(b)).encode() + b':' + b)
    return h.hexdigest()


def _code_of(obj):
    if isinstance(obj, (classmethod, staticmethod)):
        obj = obj.__func__
    if isinstance(obj, types.MethodType):
        obj = obj.__func__
    co = getattr(obj, '__code__', None)
    return co if isinstance(co, types.CodeType) else None


def _library(filename):
    """Code of the interpreter's installation (stdlib, site-packages), which no candidate can write."""
    if not isinstance(filename, str) or filename.startswith('<'):
        return False
    f = os.path.normcase(os.path.abspath(filename))
    roots = {os.path.normcase(os.path.abspath(r)).rstrip(os.sep) + os.sep for r in (sys.prefix, sys.base_prefix, sys.exec_prefix) if r}
    return any(f.startswith(r) for r in roots)


def _unwrap(obj, wrappers):
    """Look through decorators (functools.wraps sets __wrapped__); each wrapper is appended to `wrappers`."""
    if isinstance(obj, (classmethod, staticmethod)):
        obj = obj.__func__
    for _ in range(8):
        inner = getattr(obj, '__wrapped__', None)
        if inner is None:
            break
        co = _code_of(obj)
        wrappers.append('%s@%s' % (co.co_name, co.co_filename) if co is not None else type(obj).__name__)
        obj = inner
    return obj


def _foreign(wrappers):
    bad = [w for w in wrappers if '@' not in w or not _library(w.split('@', 1)[1])]
    return 'foreign wrapper: ' + ', '.join(bad) if bad else None


def _methods_live(cls, filename, wrapped):
    """{code name: hash} of the functions in the class dict that were written in `filename`."""
    out = {}
    for v in vars(cls).values():
        ws = []
        co = _code_of(_unwrap(v, ws))
        if co is not None and co.co_filename == filename and not co.co_name.startswith('<'):
            out[co.co_name] = _foreign(ws) or code_hash(co)
            if ws:
                wrapped[co.co_name] = ws
    return dict(sorted(out.items()))


def _func_live(obj, filename, name, wrapped=None):
    ws = []
    obj = _unwrap(obj, ws)
    co = _code_of(obj) if isinstance(obj, types.FunctionType) else None
    if co is None:
        return 'not a function: ' + type(obj).__name__
    if co.co_filename != filename or co.co_name != name:
        return 'foreign code: %s in %s' % (co.co_name, co.co_filename)
    if ws and wrapped is not None:
        wrapped[name] = ws
    return _foreign(ws) or code_hash(co)


def _optimizer_live(opt, g):
    if opt is None:
        return {'present': False}
    inst = vars(opt) if hasattr(opt, '__dict__') else {}
    return {'present': True, 'class': type(opt).__name__, 'class_is_train_def': type(opt) is g.get(OPTIMIZER_CLASS),
            'instance_overrides': sorted(k for k in inst if k in OPTIMIZER_ATTRS)}


def live(g, prepare_module, *, train_path, prepare_path, model=None, optimizer=None):
    """The attestation of a run: g = train.py's globals (runpy), prepare_module = sys.modules['prepare'], optimizer =
    the stepping optimizer (default: g['optimizer'])."""
    out, wrapped = {'python': sys.version.split()[0]}, {}
    for name in TRAIN_FUNCS + TRAIN_BOUND_FUNCS:
        out['train:' + name] = _func_live(g.get(name), train_path, name, wrapped) if name in g else 'missing'
    for name in TRAIN_CLASSES + TRAIN_BOUND_CLASSES:
        cls = g.get(name)
        out['train:' + name] = _methods_live(cls, train_path, wrapped) if isinstance(cls, type) else 'missing'
    for cname, methods in TRAIN_METHODS.items():
        cls = g.get(cname)
        for m in methods:
            out['train:%s.%s' % (cname, m)] = (_func_live(getattr(cls, m, None), train_path, m, wrapped)
                                               if isinstance(cls, type) else 'missing')
    for name in PREPARE_FUNCS:
        out['prepare:' + name] = _func_live(getattr(prepare_module, name, None), prepare_path, name, wrapped)
    for name in PREPARE_CLASSES:
        cls = getattr(prepare_module, name, None)
        out['prepare:' + name] = _methods_live(cls, prepare_path, wrapped) if isinstance(cls, type) else 'missing'
    out['wrappers'] = wrapped                       # audit: library decorators looked through
    out['bound_to_prepare'] = {n: (n in g and prepare_module is not None and hasattr(prepare_module, n)
                                   and g[n] is getattr(prepare_module, n)) for n in PREPARE_BOUND}
    m = getattr(model, '_orig_mod', model)
    protected = {x for ms in TRAIN_METHODS.values() for x in ms}
    out['instance_overrides'] = sorted(k for k in (vars(m) if m is not None and hasattr(m, '__dict__') else {}) if k in protected)
    out['model_class'] = (type(m).__name__ if m is not None else None)
    out['model_class_is_GPT'] = m is not None and type(m) is g.get('GPT')
    out['optimizer'] = _optimizer_live(optimizer if optimizer is not None else g.get('optimizer'), g)
    return out


class StepWatch:
    """Global optimizer post-step hook (torch.optim.optimizer.register_optimizer_step_post_hook): the bindings are
    attested at the first optimizer step - where the update code is used - and every step is counted with the
    optimizer that took it. install() registers it as soon as train.py has imported torch (a one-shot import hook,
    removed right after), so run_train.py never imports torch before train.py sets its environment."""

    def __init__(self, train_path, prepare_path):
        self.train_path, self.prepare_path = train_path, prepare_path
        self.steps, self.first, self.first_ref, self.stepped = 0, None, None, set()
        self.override_steps = 0                 # hooked steps taken while the instance overrode a step method
        self.registered, self.error, self._real_import = False, None, None

    def __call__(self, opt, args=None, kwargs=None):
        self.steps += 1
        self.stepped.add(id(opt))
        inst = getattr(opt, '__dict__', None)
        if isinstance(inst, dict) and any(k in inst for k in OPTIMIZER_ATTRS):
            self.override_steps += 1
        if self.first is None:
            g = getattr(sys.modules.get('__main__'), '__dict__', None) or {}    # train.py's globals under runpy
            try:
                self.first = live(g, sys.modules.get('prepare'), train_path=self.train_path, prepare_path=self.prepare_path,
                                  model=g.get('model'), optimizer=opt)
            except Exception as e:                  # noqa: BLE001 - recorded; a missing snapshot rejects the run
                self.first = {'error': '%s: %s' % (type(e).__name__, str(e)[:300])}
            try:
                import weakref
                self.first_ref = weakref.ref(opt)
            except TypeError:
                self.first_ref = None

    def register(self, optimizer_module):
        self.registered = True
        try:
            optimizer_module.register_optimizer_step_post_hook(self)
        except Exception as e:                      # noqa: BLE001
            self.error = '%s: %s' % (type(e).__name__, str(e)[:300])

    def install(self):
        import builtins
        real = self._real_import = builtins.__import__

        def _import(*a, **k):
            mod = real(*a, **k)
            opt_mod = sys.modules.get('torch.optim.optimizer')
            # torch.optim.optimizer enters sys.modules before it has run: wait for its hook registry
            if not self.registered and opt_mod is not None and hasattr(opt_mod, 'register_optimizer_step_post_hook'):
                builtins.__import__ = real
                self.register(opt_mod)
            return mod
        builtins.__import__ = _import

    def uninstall(self):
        import builtins
        if self._real_import is not None and not self.registered:
            builtins.__import__ = self._real_import

    def summary(self, end_optimizer):
        same = self.first_ref is not None and end_optimizer is not None and self.first_ref() is end_optimizer
        return {'first_step': self.first, 'optimizer_steps': self.steps, 'optimizers_stepped': len(self.stepped),
                'same_optimizer': same, 'override_steps': self.override_steps,
                'step_hook': 'registered' if self.registered and not self.error else (self.error or 'not registered')}


def _source_codes(path):
    """{name: module-level function code, class name: {method name: code}} of a source file (compiled, not run)."""
    with open(path, 'rb') as f:
        src = f.read()
    mod = compile(src, path, 'exec', dont_inherit=True)
    funcs, classes = {}, {}
    for c in mod.co_consts:                         # later definitions win, as when the module runs
        if not isinstance(c, types.CodeType) or c.co_name.startswith('<'):
            continue
        if c.co_flags & CO_OPTIMIZED:               # a function body; a class body runs unoptimized, like a module
            funcs[c.co_name] = c
        else:
            classes[c.co_name] = {x.co_name: x for x in c.co_consts
                                  if isinstance(x, types.CodeType) and not x.co_name.startswith('<')}
    return funcs, classes


def expect_from(train_path, prepare_path):
    out = {}
    if train_path:
        funcs, classes = _source_codes(train_path)
        for name in TRAIN_FUNCS + TRAIN_BOUND_FUNCS:
            out['train:' + name] = code_hash(funcs[name]) if name in funcs else 'missing'
        for name in TRAIN_CLASSES + TRAIN_BOUND_CLASSES:
            out['train:' + name] = ({k: code_hash(v) for k, v in sorted(classes[name].items())} if name in classes
                                    else 'missing')
        for cname, methods in TRAIN_METHODS.items():
            for m in methods:
                body = classes.get(cname) or {}
                out['train:%s.%s' % (cname, m)] = code_hash(body[m]) if m in body else 'missing'
    if prepare_path:
        funcs, classes = _source_codes(prepare_path)
        for name in PREPARE_FUNCS:
            out['prepare:' + name] = code_hash(funcs[name]) if name in funcs else 'missing'
        for name in PREPARE_CLASSES:
            out['prepare:' + name] = ({k: code_hash(v) for k, v in sorted(classes[name].items())} if name in classes
                                      else 'missing')
    return out


def _cli(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('expect',))
    ap.add_argument('--train', required=True)
    ap.add_argument('--prepare', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    res = {'schema': 'ext-autoresearch-code-expect-v1', 'python': sys.version.split()[0]}
    for key, tp, pp in (('candidate', a.train, a.prepare), ('baseline', a.base, None)):
        try:
            res[key] = expect_from(tp, pp)
        except Exception as e:                      # noqa: BLE001 - e.g. a candidate that does not compile
            res[key] = {'error': '%s: %s' % (type(e).__name__, str(e)[:300])}
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1, sort_keys=True)


if __name__ == '__main__':
    _cli(sys.argv[1:])
