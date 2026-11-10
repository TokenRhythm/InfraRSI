import json
import runpy
import sys

import pytest

from infrarsi.ext.autoresearch import infra_only_check as ic, source
from infrarsi.ext.autoresearch.node import code_attest
from infrarsi.ext.autoresearch.task import expected_runtime
from x1_ar_helpers import GROUPS, run_meta, train_log

BASE = source.base_train()

VIOLATIONS = {
    'lr constant': ('MATRIX_LR = 0.04', 'MATRIX_LR = 0.05'),
    'batch constant': ('TOTAL_BATCH_SIZE = 2**19', 'TOTAL_BATCH_SIZE = 2**18'),
    'depth': ('DEPTH = 8 ', 'DEPTH = 6 '),
    'rebinding': ('t_start = time.time()', 'MATRIX_LR = 0.04 * 1.1\nt_start = time.time()'),
    'gptconfig default': ('    n_embd: int = 768', '    n_embd: int = 1024'),
    'build_model_config': ('    num_heads = model_dim // HEAD_DIM', '    num_heads = model_dim // 64'),
    'setup_optimizer group': ("momentum=0.95, ns_steps=5, beta2=0.95", "momentum=0.9, ns_steps=5, beta2=0.95"),
    'lr schedule': ('        return 1.0\n    else:', '        return 1.2\n    else:'),
    'seed': ('torch.manual_seed(42)', 'torch.manual_seed(7)'),
    'extra seed': ('torch.cuda.manual_seed(42)', 'torch.cuda.manual_seed(42)\ntorch.manual_seed(42)'),
    'precision call': ('torch.set_float32_matmul_precision("high")', 'torch.set_float32_matmul_precision("medium")'),
    'tf32 flag': ('device = torch.device("cuda")', 'device = torch.device("cuda")\ntorch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True'),
    'autocast dtype': ('dtype=torch.bfloat16)\nH100', 'dtype=torch.float16)\nH100'),
    'timer start': ('    if step > 10:\n        total_training_time += dt', '    if step > 5:\n        total_training_time += dt'),
    'budget stop': ('    if step > 10 and total_training_time >= TIME_BUDGET:', '    if step > 10 and total_training_time >= TIME_BUDGET * 1.2:'),
    'fast fail': ('train_loss_f > 100:', 'train_loss_f > 1000:'),
    'progress format': ('| loss: {debiased_smooth_loss:.6f}', '| loss: {debiased_smooth_loss:.3f}'),
    'dataloader args': ('make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")',
                        'make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train", buffer_size=10)'),
    'second dataloader': ('x, y, epoch = next(train_loader)  # prefetch first batch',
                          'x, y, epoch = next(train_loader)  # prefetch first batch\nextra = make_dataloader(tokenizer, 8, MAX_SEQ_LEN, "train")'),
    'own evaluate_bpb': ('# Final eval', 'def evaluate_bpb(*a):\n    return 0.5\n# Final eval'),
    'prepare import': ('from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb',
                       'from prepare import MAX_SEQ_LEN, Tokenizer, make_dataloader, evaluate_bpb\nTIME_BUDGET = 400'),
    'prepare monkeypatch': ('# Final eval', 'import prepare\nprepare.EVAL_TOKENS = 1\n# Final eval'),
    'globals': ('# Final eval', 'globals()["MATRIX_LR"] = 1.0\n# Final eval'),
    'schedule application': ('        group["lr"] = group["initial_lr"] * lrm', '        group["lr"] = group["initial_lr"] * lrm * 1.1'),
    # Codex W9 H-02: routes that rebind a protected callable behind the static comparison
    'rebind setup_optimizer': ('t_start = time.time()', 'GPT.setup_optimizer = lambda self, **kw: None\nt_start = time.time()'),
    'setattr route': ('t_start = time.time()', 'setattr(GPT, "setup_optimizer", None)\nt_start = time.time()'),
    'delete method': ('t_start = time.time()', 'del GPT.setup_optimizer\nt_start = time.time()'),
    'rebind schedule': ('t_start = time.time()', 'get_lr_multiplier = lambda progress: 1.0\nt_start = time.time()'),
    'exec route': ('t_start = time.time()', 'exec("MATRIX_LR2 = 1")\nt_start = time.time()'),
    'eval route': ('t_start = time.time()', 'f = eval("get_lr_multiplier")\nt_start = time.time()'),
    'computed getattr': ('t_start = time.time()', 'f = getattr(GPT, "setup_" + "optimizer")\nt_start = time.time()'),
    'code object swap': ('t_start = time.time()', 'get_weight_decay.__code__ = get_muon_momentum.__code__\nt_start = time.time()'),
    'importlib reload': ('t_start = time.time()', 'import importlib, prepare\nimportlib.reload(prepare)\nt_start = time.time()'),
    'sys.modules route': ('t_start = time.time()', 'import sys\nsys.modules["prepare"].TIME_BUDGET = 400\nt_start = time.time()'),
    'frame globals': ('t_start = time.time()', 'import sys\nsys._getframe().f_globals["MATRIX_LR"] = 1.0\nt_start = time.time()'),
    'class attribute': ('t_start = time.time()', 'GPTConfig.n_layer = 12\nt_start = time.time()'),
    'except-as rebinding': ('t_start = time.time()', 'try:\n    pass\nexcept Exception as GPT:\n    pass\nt_start = time.time()'),
    'dunder builtins': ('t_start = time.time()', '__builtins__ = {}\nt_start = time.time()'),
    'gc referrers': ('t_start = time.time()', 'refs = gc.get_referrers(GPT)\nt_start = time.time()'),
    'pickle import': ('t_start = time.time()', 'import pickle\nt_start = time.time()'),
    'vars route': ('t_start = time.time()', 'vars()["get_weight_decay"] = None\nt_start = time.time()'),
    # Codex W9 round 2: optimizer and alias routes
    'optimizer.step swap': ('t_start = time.time()', 'optimizer.step = lambda *a, **k: None\nt_start = time.time()'),
    'optimizer param_groups': ('t_start = time.time()', 'optimizer.param_groups = []\nt_start = time.time()'),
    'optimizer zero_grad': ('t_start = time.time()', 'optimizer.zero_grad = None\nt_start = time.time()'),
    'model forward': ('t_start = time.time()', 'model.forward = None\nt_start = time.time()'),
    'alias of GPT': ('t_start = time.time()', 'X = GPT\nX.extra_flag = 1\nt_start = time.time()'),
    'alias of MuonAdamW': ('t_start = time.time()', 'Opt = MuonAdamW\nOpt.step = None\nt_start = time.time()'),
    'chained alias': ('t_start = time.time()', 'A = GPT\nB = A\nB.anything = 1\nt_start = time.time()'),
    'default-arg alias': ('t_start = time.time()', 'def f(c=GPT):\n    c.anything = 1\nt_start = time.time()'),
    'computed root': ('t_start = time.time()', 'type(model).forward = None\nt_start = time.time()'),
    'computed root other attr': ('t_start = time.time()', 'type(model).helper = 1\nt_start = time.time()'),
    'prepare module alias': ('t_start = time.time()', 'import prepare as P\nP.TIME_BUDGET = 400\nt_start = time.time()'),
    'star import': ('t_start = time.time()', 'from os.path import *\nt_start = time.time()'),
    'hook registry': ('t_start = time.time()', 'torch.optim.optimizer._global_optimizer_post_hooks.clear()\nt_start = time.time()'),
    'rebind fused step': ('t_start = time.time()', 'adamw_step_fused = None\nt_start = time.time()'),
    'store on fused step': ('t_start = time.time()', 'muon_step_fused.cache = None\nt_start = time.time()'),
    # Codex W9 round 3, R3-01: reflection through operator / dunder strings / dunder methods
    'R3-01 operator swap': ('t_start = time.time()', 'import operator\nd = operator.attrgetter("__dict__")(optimizer)\noperator.setitem(d, "step", None)\noperator.delitem(d, "step")\nt_start = time.time()'),
    'operator from-import': ('t_start = time.time()', 'from operator import setitem\nt_start = time.time()'),
    'dunder string to a call': ('t_start = time.time()', 'f = functools.partial(getattr, optimizer, "__dict__")\nt_start = time.time()'),
    'dunder method __setitem__': ('t_start = time.time()', 'x = {}\nx.__setitem__("a", 1)\nt_start = time.time()'),
    'dunder __getstate__': ('t_start = time.time()', 's = optimizer.__getstate__()\nt_start = time.time()'),
}

ALLOWED = {
    'device batch': ('DEVICE_BATCH_SIZE = 128', 'DEVICE_BATCH_SIZE = 64'),
    'compile mode': ('model = torch.compile(model, dynamic=False)', 'model = torch.compile(model, dynamic=False, mode="max-autotune")'),
    'fused norm': ('def norm(x):\n    return F.rms_norm(x, (x.size(-1),))', 'def norm(x):\n    return F.rms_norm(x, (x.size(-1),), eps=None)'),
    'gc policy': ('    elif (step + 1) % 5000 == 0:', '    elif (step + 1) % 2500 == 0:'),
    'print less often': ('    print(f"\\rstep {step:05d}', '    if step < 20 or step % 10 == 0:\n        print(f"\\rstep {step:05d}'),
    'prefetch wrapper': ('train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")',
                         'def prefetch(it):\n    yield from it\ntrain_loader = prefetch(make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train"))'),
    'literal getattr': ('t_start = time.time()', 'n_sm = getattr(torch.cuda, "device_count")()\nt_start = time.time()'),
    'sys stdout': ('t_start = time.time()', 'import sys\nsys.stdout.flush()\nt_start = time.time()'),
    'helper class': ('t_start = time.time()', 'class Prefetcher:\n    def __init__(self, it):\n        self.it = it\nt_start = time.time()'),
    'tensor slice write': ('t_start = time.time()', 'buf = torch.zeros(4)\nbuf.view(-1)[:2] = 1\nt_start = time.time()'),
    'model instance alias call': ('t_start = time.time()', 'm2 = model\nn_params = sum(p.numel() for p in m2.parameters())\nt_start = time.time()'),
}


def _apply(old, new):
    assert BASE.count(old) == 1, old
    return BASE.replace(old, new)


@pytest.mark.parametrize('name', sorted(VIOLATIONS))
def test_static_rejects_semantic_changes(name):
    g = ic.static(BASE, _apply(*VIOLATIONS[name]))
    assert not g.ok and set(g.ids()) == {'C-static'}, name


@pytest.mark.parametrize('name', sorted(ALLOWED))
def test_static_allows_infra_changes(name):
    g = ic.static(BASE, _apply(*ALLOWED[name]))
    assert g.ok, (name, g.rejections)


def test_static_baseline_and_syntax():
    assert ic.static(BASE, BASE).ok
    assert 'does not parse' in ic.static(BASE, 'def x(:\n').rejections[0]['detail']


def _expected():
    e = expected_runtime({'reference': {'optimizer_groups': GROUPS}})
    assert e['constants']['TOTAL_BATCH_SIZE'] == 524288 and e['constants']['ADAM_BETAS'] == [0.8, 0.95]
    return e


def test_runtime_check():
    exp = _expected()
    assert ic.runtime(run_meta(), {'num_params': 50332176}, exp).ok
    assert ic.runtime(run_meta(steps=1200), None, exp).ok               # more steps is the goal
    meta = run_meta()
    meta['constants']['MATRIX_LR'] = 0.05
    assert 'MATRIX_LR' in ic.runtime(meta, None, exp).rejections[0]['detail']
    for over, needle in ((dict(config={**source.R2['model_config'], 'n_layer': 6}), 'model config'),
                         (dict(optimizer_groups=GROUPS[:1]), 'optimizer parameter groups'),
                         (dict(float32_matmul_precision='highest'), 'precision'),
                         (dict(save_error='OSError: disk'), 'save tail failed')):
        assert any(needle in r['detail'] for r in ic.runtime(run_meta(**over), None, exp).rejections), needle
    meta = run_meta()
    meta['counters']['total_tokens'] += 1
    assert 'steps/tokens inconsistent' in ic.runtime(meta, None, exp).rejections[0]['detail']
    meta = run_meta()
    meta['counters']['total_training_time'] = 250.0
    assert 'timed training' in ic.runtime(meta, None, exp).rejections[0]['detail']
    meta = run_meta()
    meta['wrapper']['train_py_wall_s'] = 200.0
    assert 'wall' in ic.runtime(meta, None, exp).rejections[0]['detail']
    assert ic.runtime(None, None, exp).rejections[0]['detail'] == 'run_meta.json missing or unreadable'
    assert 'saved weights hold' in ic.runtime(run_meta(), {'num_params': 10}, exp).rejections[0]['detail']
    loose = dict(exp, optimizer_groups=None)
    assert ic.runtime(run_meta(optimizer_groups=[]), None, loose).ok


def test_first_losses_check():
    ref = source.R2['first_losses']
    assert ic.losses(train_log(), ref).ok
    near = [v * 1.009 for v in ref]
    assert ic.losses(train_log(near), ref).ok
    far = list(ref)
    far[5] *= 1.02
    g = ic.losses(train_log(far), ref)
    assert not g.ok and 'step 5' in g.rejections[0]['detail'] and g.reports[0]['worst'] == pytest.approx(0.02, rel=1e-3)
    assert not ic.losses('no lines', ref).ok
    assert ic.losses(train_log(far), ref, veto=False).ok                   # report-only mode keeps the numbers


# ---------------------------------------------------------------- code attestation (Codex W9 H-02, rounds 1 and 2)
MINI_PREPARE = """import contextlib

MAX_SEQ_LEN = 2048
TIME_BUDGET = 300


class Tokenizer:
    def __init__(self, enc):
        self.enc = enc

    @classmethod
    def from_directory(cls, d):
        return cls(d)

    def encode(self, s):
        return [ord(c) for c in s]


def make_dataloader(tok, b, t, split):
    yield from ()


@contextlib.contextmanager                  # a library decorator (functools.wraps), like torch.no_grad on the real one
def evaluate_bpb(model, tok, b):
    yield 1.0
"""

MINI_TRAIN = """from dataclasses import dataclass
from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb
import fakeoptim


@dataclass
class GPTConfig:
    n_layer: int = 8


def build_model_config(depth):
    return GPTConfig(n_layer=depth)


def get_lr_multiplier(p):
    return 1.0 if p < 0.5 else 0.5


def get_muon_momentum(step):
    return 0.95


def get_weight_decay(p):
    return 0.2 * (1 - p)


def adamw_step_fused(p, g):
    return p - 0.1 * g


def muon_step_fused(p, g):
    return p - 0.2 * g


class MuonAdamW:
    def __init__(self, groups):
        self.groups = groups

    def step(self):
        for g in self.groups:
            g["p"] = adamw_step_fused(g["p"], 1.0) if g["kind"] == "adamw" else muon_step_fused(g["p"], 1.0)
        fakeoptim.fire(self)                  # torch fires the global post-step hooks from the (patched) class step


class GPT:
    def __init__(self, config):
        self.config = config

    def setup_optimizer(self, lr=0.04):
        return MuonAdamW([{"kind": "adamw", "p": 1.0, "lr": lr}, {"kind": "muon", "p": 1.0, "lr": lr}])


model = GPT(build_model_config(8))
optimizer = model.setup_optimizer()
step = 0
while step < 3:
    {BEFORE}optimizer.step()
    step += 1
"""


class FakeOptimModule:
    """Stands in for torch.optim.optimizer: a global post-step hook registry, fired by the mini optimizer's class step
    the way torch's patched Optimizer.step fires it."""

    def __init__(self):
        self.hooks = []

    def register_optimizer_step_post_hook(self, hook):
        self.hooks.append(hook)

    def fire(self, opt):
        for h in self.hooks:
            h(opt, (), {})


def _attest(tmp_path, monkeypatch, tail='', base=None, before='', hook=True):
    """Runs the mini train.py as run_train.py does (StepWatch installed, runpy) and returns (live, expected)."""
    d = tmp_path / 'repo'
    d.mkdir(parents=True)
    train_text = MINI_TRAIN.replace('{BEFORE}', before) + tail
    (d / 'prepare.py').write_text(MINI_PREPARE, encoding='utf-8')
    (d / 'train.py').write_text(train_text, encoding='utf-8')
    (tmp_path / 'base_train.py').write_text(base or MINI_TRAIN.replace('{BEFORE}', ''), encoding='utf-8')
    monkeypatch.syspath_prepend(str(d))
    monkeypatch.delitem(sys.modules, 'prepare', raising=False)
    fake = FakeOptimModule()
    monkeypatch.setitem(sys.modules, 'fakeoptim', fake)
    if hook:
        monkeypatch.setitem(sys.modules, 'torch.optim.optimizer', fake)
    else:
        monkeypatch.delitem(sys.modules, 'torch.optim.optimizer', raising=False)
    train = str(d / 'train.py')
    watch = code_attest.StepWatch(train, str(d / 'prepare.py'))
    watch.install()
    try:
        g = runpy.run_path(train, run_name='__main__')
    finally:
        watch.uninstall()
    live = {**code_attest.live(g, sys.modules['prepare'], train_path=train, prepare_path=str(d / 'prepare.py'),
                               model=g['model'], optimizer=g.get('optimizer')), **watch.summary(g.get('optimizer')),
            '_counters': {'step': g.get('step')}}
    out = tmp_path / 'code_expected.json'
    code_attest._cli(['expect', '--train', train, '--prepare', str(d / 'prepare.py'), '--base', str(tmp_path / 'base_train.py'),
                      '--out', str(out)])
    return live, json.loads(out.read_text(encoding='utf-8'))


def att(live, exp):
    """The runner's call: run_meta counters travel next to the attestation."""
    return ic.attestation(live, exp, counters=live.get('_counters'))


def test_attestation_live_matches_the_compiled_source(tmp_path, monkeypatch):
    live, exp = _attest(tmp_path, monkeypatch)
    assert att(live, exp).ok, att(live, exp).rejections
    assert set(exp['candidate']) == {f'train:{n}' for n in code_attest.TRAIN_FUNCS + code_attest.TRAIN_CLASSES
                                     + code_attest.TRAIN_BOUND_FUNCS + code_attest.TRAIN_BOUND_CLASSES} | {
        'train:GPT.setup_optimizer'} | {f'prepare:{n}' for n in code_attest.PREPARE_FUNCS + code_attest.PREPARE_CLASSES}
    assert exp['candidate']['prepare:Tokenizer'].keys() == {'__init__', 'from_directory', 'encode'}
    assert exp['candidate']['train:MuonAdamW'].keys() == {'__init__', 'step'}
    assert live['train:GPTConfig'] == {} and all(live['bound_to_prepare'].values())
    assert list(live['wrappers']) == ['evaluate_bpb'] and live['wrappers']['evaluate_bpb'][0].startswith('helper@')
    assert live['optimizer_steps'] == 3 and live['optimizers_stepped'] == 1 and live['same_optimizer'] is True
    assert live['step_hook'] == 'registered' and live['first_step']['optimizer'] == {
        'present': True, 'class': 'MuonAdamW', 'class_is_train_def': True, 'instance_overrides': []}
    import builtins
    assert builtins.__import__ is __import__                            # the one-shot import hook is gone


@pytest.mark.parametrize('tail,why', [
    ('GPT.setup_optimizer = lambda self, lr=0.04: {}\n', 'train:GPT.setup_optimizer'),
    ('model.setup_optimizer = lambda **k: None\n', 'overrides protected methods'),
    ('import prepare\nprepare.make_dataloader = lambda *a: iter(())\n', 'prepare:make_dataloader'),
    ('get_lr_multiplier = lambda p: 1.0\n', 'train:get_lr_multiplier'),
    ('Tokenizer.encode = lambda self, s: []\n', 'prepare:Tokenizer'),
    ('class Fake:\n    pass\nmodel = Fake()\n', 'not the GPT class'),
    ('import functools\ndef deco(f):\n    @functools.wraps(f)\n    def w(*a, **k):\n        return f(*a, **k)\n    return w\n'
     'get_weight_decay = deco(get_weight_decay)\n', 'foreign wrapper'),
    # round 2: the two attacks of the review, after the first steps (past the first-loss window)
    ('optimizer.step = lambda: None\n', "the optimizer instance overrides ['step'] at run end"),
    ('X = MuonAdamW\nX.step = lambda self: None\n', 'train:MuonAdamW'),
    ('X = GPT\nX.setup_optimizer = lambda self, lr=0.04: None\n', 'train:GPT.setup_optimizer'),
    ('adamw_step_fused = lambda p, g: p\n', 'train:adamw_step_fused'),
    ('optimizer = MuonAdamW([])\n', 'not the one that took the first step'),
    ('opt2 = MuonAdamW([])\nfakeoptim.fire(opt2)\n', '2 different optimizers took steps'),
    ('class Other:\n    def step(self):\n        pass\noptimizer = Other()\n', "the optimizer at run end is a 'Other'"),
])
def test_attestation_catches_runtime_rebinding(tmp_path, monkeypatch, tail, why):
    live, exp = _attest(tmp_path, monkeypatch, tail)
    g = att(live, exp)
    assert not g.ok and any(why in r['detail'] for r in g.rejections), g.rejections


def test_attestation_at_the_first_step_catches_early_swaps(tmp_path, monkeypatch):
    """A swap made before the first step shows in the first-step snapshot even if it is undone before the end."""
    swap = 'if step == 0:\n        optimizer.step = lambda: MuonAdamW.step(optimizer)\n    '
    live, exp = _attest(tmp_path, monkeypatch, before=swap, tail='del optimizer.step\n')
    g = att(live, exp)
    assert not g.ok and any("overrides ['step'] at the first optimizer step" in r['detail'] for r in g.rejections), g.rejections
    assert live['optimizer']['instance_overrides'] == []                # the end alone would have passed


def test_attestation_needs_the_step_hook(tmp_path, monkeypatch):
    live, exp = _attest(tmp_path, monkeypatch, hook=False)
    g = att(live, exp)
    assert not g.ok and live['optimizer_steps'] == 0 and live['step_hook'] == 'not registered'
    assert any('no attestation at the first optimizer step' in r['detail'] for r in g.rejections)
    assert any('no optimizer step was observed' in r['detail'] for r in g.rejections)


def test_attestation_strict_functions_equal_the_baseline(tmp_path, monkeypatch):
    base = MINI_TRAIN.replace('{BEFORE}', '').replace('return 0.2 * (1 - p)', 'return 0.1 * (1 - p)')
    live, exp = _attest(tmp_path, monkeypatch, base=base)
    g = att(live, exp)
    assert not g.ok and 'train:get_weight_decay (differs from the baseline)' in g.rejections[0]['detail']
    base = MINI_TRAIN.replace('{BEFORE}', '').replace('return {"lr": lr', 'print(1)\n        return {"lr": lr')
    live, exp = _attest(tmp_path / 'b', monkeypatch, base=base)
    assert att(live, exp).ok                                 # setup_optimizer: checked against its own source
    base = MINI_TRAIN.replace('{BEFORE}', '').replace('return p - 0.1 * g', 'return p - 0.1 * g * 1.0')
    live, exp = _attest(tmp_path / 'c', monkeypatch, base=base)
    assert att(live, exp).ok                                 # the optimizer code may differ from the baseline


def test_attestation_needs_both_sides():
    assert 'no code attestation' in ic.attestation(None, {}).rejections[0]['detail']
    assert 'missing or unreadable' in ic.attestation({'python': '3'}, None).rejections[0]['detail']
    bad = {'schema': 'ext-autoresearch-code-expect-v1', 'python': '3', 'candidate': {'error': 'SyntaxError: x'}, 'baseline': {}}
    assert 'trusted compile of the candidate failed' in ic.attestation({'python': '3'}, bad).rejections[0]['detail']


R3_SWAP = ('if step == 1:\n        vars(optimizer)["step"] = {fn}\n    if step == 2:\n        del vars(optimizer)["step"]\n    ')


def test_r3_01_swap_between_the_snapshots_is_caught(tmp_path, monkeypatch):
    """Codex W9 round 3, R3-01: an instance `step` installed after the first snapshot and removed before the end.
    Bypassing the class method: fewer hooked steps than training steps. Calling it: a hooked step under an override."""
    live, exp = _attest(tmp_path, monkeypatch, before=R3_SWAP.format(fn='lambda: None'))
    assert live['first_step']['optimizer']['instance_overrides'] == [] and live['optimizer']['instance_overrides'] == []
    g = att(live, exp)
    assert not g.ok and any('2 optimizer steps went through the class step method, the run made 3 training steps' in r['detail']
                            for r in g.rejections), g.rejections
    live, exp = _attest(tmp_path / 'b', monkeypatch, before=R3_SWAP.format(fn='lambda: MuonAdamW.step(optimizer)'))
    g = att(live, exp)
    assert live['optimizer_steps'] == 3 and live['override_steps'] == 1
    assert not g.ok and any('1 hooked steps ran while the optimizer instance overrode a step method' in r['detail']
                            for r in g.rejections), g.rejections
    assert not ic.attestation({**live, 'override_steps': 0}, exp).ok      # no counters: no comparison possible, rejected
