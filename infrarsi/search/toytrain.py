"""ToyTrain: a CPU simulator of a training job's update time and memory, for comparing search strategies (revamp C).

A task is (model class x GPU class x schema x hidden bottleneck x noise) with exactly one seed. A candidate is a knob
config plus patch mechanisms (the mutable file `patches.txt`, one mechanism per line). `ToyRunner` returns signed
project receipts (the real Verifier judges them), advances a virtual clock by load + executed updates, supports
fidelities (`steps`) and early stop (`stop(rows)` after each update), and `ToySim.truth()` gives the noiseless value
for metrics. Noise uses common random numbers keyed by (task seed, config, steps, attempt index), so two strategies
that run the same candidate see the same draw.

Properties that matter for search (Fable consult 5.1): interacting mechanisms (micro-batch needs memory from
recompute/TP; TP-comm overlap needs sequence parallelism; DP overlap only matters with DP > 1), a memory cliff,
divisibility, crash pairs found by a 1-2 update probe, a hang that only a full run reveals, per-update and per-run
noise with warm-up drift, decoy toggles, and patch-only bottlenecks.
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import random

from ..contracts import digest, new_attempt_id, utc_now
from ..harness import signing
from ..project.signals import robust_rsd, split_half
from ..verify import SCHEMA

GPUS = 8
GBS, SEQ, VOCAB = 64, 4096, 151936
MODELS = {'1b': dict(params=1.3e9, active=1.3e9, layers=24, hidden=2048, moe=False),
          '7b': dict(params=7.0e9, active=7.0e9, layers=32, hidden=4096, moe=False),
          '30b-moe': dict(params=30e9, active=3.3e9, layers=48, hidden=2048, moe=True)}
HW = {'h100': dict(tflops=989e12, mem=78.0, hbm=3.35e12, link=450e9, tp_share=0.035),
      'h20': dict(tflops=148e12, mem=95.0, hbm=4.0e12, link=450e9, tp_share=0.012),
      'l20x': dict(tflops=240e12, mem=138.0, hbm=4.8e12, link=64e9, tp_share=0.15)}
LAYOUT = {'tp': [1, 2, 4, 8], 'pp': [1, 2, 4], 'mbs': [1, 2, 4, 8], 'recompute': ['full', 'selective', 'none'],
          'offload': [False, True]}
MOE_LAYOUT = {'ep': [1, 2, 4, 8], 'dispatcher': ['allgather', 'alltoall']}
TOGGLES = ('fused_adam', 'fused_ce', 'flash_attn', 'overlap_grad', 'seq_parallel', 'tp_overlap', 'compile',
           'expandable_segments', 'cuda_graphs', 'pin_memory', 'nccl_tree', 'deterministic')
PATCHES = {'remove_host_sync': 0.75, 'fuse_optimizer_kernel': 0.8, 'prefetch_data': 0.9, 'reorder_allreduce': 0.6}
# T4 regime: the toggles that help on their own; pairs of them may conflict (same resource), t x ANTAGONISM_COST each
GOOD8 = ('fused_adam', 'fused_ce', 'flash_attn', 'overlap_grad', 'pin_memory', 'expandable_segments', 'cuda_graphs',
         'compile')
ANTAGONISM_COST = 1.03
_OPTIMA: dict = {}
WINDOWS = {30: (11, 30), 10: (4, 10), 2: (1, 2)}


@dataclasses.dataclass(frozen=True)
class ToyTask:
    model: str
    gpu: str
    schema: str          # layout | toggle | both
    hidden: str          # none | hostsync
    noise: str           # low | high
    antagonism: float | None = None   # T4 regime: share of antagonistic GOOD8 pairs (None: the T1-T3 simulator)
    gain_scale: float | None = None   # T5 regime: the good toggles' log gains x s
    run_sd: float | None = None       # T5 regime: between-run noise (sd of a run's log offset)
    drift: bool = False               # the run offset differs between the halves of the window (split halves see it)
    cv: float | None = None           # per-update noise (None: 0.01 low / 0.03 high)

    @property
    def base_name(self) -> str:
        return f'{self.model}-{self.gpu}-{self.schema}-{self.hidden}-{self.noise}'

    @property
    def name(self) -> str:
        out = self.base_name if self.antagonism is None else f'{self.base_name}-a{self.antagonism:g}'
        if self.gain_scale is not None or self.run_sd is not None:
            out += f'-s{self.gain_scale if self.gain_scale is not None else 1:g}-n{self.run_sd if self.run_sd is not None else 0:g}'
        return out + ('-drift' if self.drift else '') + (f'-cv{self.cv:g}' if self.cv is not None else '')

    @property
    def seed(self) -> int:
        """From the base name: the regimes of one task share their random draws (paired, nested conflict sets)."""
        return int(hashlib.sha256(self.base_name.encode()).hexdigest()[:8], 16)


def tasks(n: int | None = None) -> list[ToyTask]:
    """The task grid (diversity instead of seeds), deterministically sub-sampled to n when given."""
    grid = [ToyTask(*t) for t in itertools.product(MODELS, HW, ('layout', 'toggle', 'both'), ('none', 'hostsync'), ('low', 'high'))]
    if n is None or n >= len(grid):
        return grid
    return sorted(grid, key=lambda t: hashlib.sha256(t.name.encode()).hexdigest())[:n]


def _u(*key) -> float:
    return int(hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12], 16) / 16 ** 12


class ToySim:
    def __init__(self, task: ToyTask):
        self.task, self.m, self.hw = task, MODELS[task.model], HW[task.gpu]
        self.cv, self.run_sd = (0.01, 0.015) if task.noise == 'low' else (0.03, 0.03)
        if task.run_sd is not None:
            self.run_sd = task.run_sd
        if task.cv is not None:
            self.cv = task.cv
        self.gs = task.gain_scale if task.gain_scale is not None else 1.0     # x ** 1.0 == x: T1-T4 unchanged
        self.host_frac = 0.12 if task.hidden == 'hostsync' else 0.02
        self.patch_ok = {p: _u(task.seed, 'patch', p) < q for p, q in PATCHES.items()}
        self.inert: frozenset = frozenset()   # (toybench --inert, hls3 F10) working patches whose hook never runs
        self.t4 = task.antagonism is not None
        self.antagonistic = {frozenset((a, b)) for a, b in itertools.combinations(GOOD8, 2)
                             if self.t4 and _u(task.seed, 'antagonism', a, b) < task.antagonism}

    def trap(self, config: dict) -> bool:
        """T4 verifier trap: fused cross-entropy under torch.compile changes the loss curve (the run is valid and
        fast, the verifier must reject it)."""
        c = self.full(config)
        return self.t4 and bool(c['fused_ce'] and c['compile'])

    # ------------------------------------------------------------ space
    def knobs(self) -> dict:
        k = {}
        if self.task.schema in ('layout', 'both'):
            k.update(LAYOUT)
            if self.m['moe']:
                k.update(MOE_LAYOUT)
        if self.task.schema in ('toggle', 'both'):
            k.update({t: [False, True] for t in TOGGLES})
        return k

    def _base(self) -> dict:
        return {'tp': 2, 'pp': 1, 'mbs': 1, 'recompute': 'full', 'offload': False, 'ep': 8 if self.m['moe'] else 1,
                'dispatcher': 'allgather', **{t: False for t in TOGGLES}}

    def _layout(self) -> dict:
        """The first runnable layout (model parallel just enough to fit, full recompute, mbs 1, toggles off)."""
        if not hasattr(self, '_first'):
            self._first = self._base()
            for tp, pp in ((2, 1), (4, 1), (4, 2), (8, 2), (8, 4)):
                c = dict(self._base(), tp=tp, pp=pp)
                if self._truth_full(c, ())[0] is not None:
                    self._first = c
                    break
        return self._first

    def reference(self) -> dict:
        """The user's current config: the first runnable layout, restricted to the schema's knobs."""
        return {k: self._layout()[k] for k in self.knobs()}

    def full(self, config: dict) -> dict:
        """Knobs the schema does not expose keep the first runnable layout (toggles off)."""
        return {**self._layout(), **config}

    # ------------------------------------------------------------ the model
    def memory(self, c: dict) -> float:
        m, tp, pp = self.m, c['tp'], c['pp']
        dp = GPUS // (tp * pp)
        dense = m['params'] * (0.1 if m['moe'] else 1.0)
        expert = m['params'] - dense
        w = dense / (tp * pp) + expert / (pp * max(1, c['ep']))
        states = 0.0 if c['offload'] else 12 * w / dp
        rc = {'full': 2 / 34, 'selective': 0.45, 'none': 1.0}[c['recompute']]
        act = 34 * SEQ * m['hidden'] * c['mbs'] * rc * (m['layers'] / pp) / (tp if c['seq_parallel'] else max(1, tp / 2))
        act *= min(pp, max(1, GBS // (dp * c['mbs'])))
        logits = c['mbs'] * SEQ * VOCAB * 4 / tp
        return (6 * w + states + act + logits) / 2 ** 30 + 3.0

    def truth(self, config: dict, patches=(), intended: bool = False) -> tuple[float | None, str | None]:
        """(noiseless seconds per update, None) or (None, failure kind); `intended`: as if its inert patches ran."""
        if self.inert and not intended:                 # (hls3 F10 toy: an inert patch changes nothing)
            patches = tuple(x for x in patches if x not in self.inert)
        return self._truth_full(self.full(config), patches)

    def _truth_full(self, c: dict, patches) -> tuple[float | None, str | None]:
        m, hw = self.m, self.hw
        tp, pp, mbs = c['tp'], c['pp'], c['mbs']
        if GPUS % (tp * pp):
            return None, 'divisibility'
        dp = GPUS // (tp * pp)
        if GBS % (dp * mbs):
            return None, 'divisibility'
        if m['moe'] and dp % c['ep'] and c['ep'] % dp:
            return None, 'divisibility'
        if c['tp_overlap'] and not (c['seq_parallel'] and tp > 1):
            return None, 'crash: tp_overlap needs sequence parallelism'
        if c['compile'] and c['offload']:
            return None, 'crash: compile with optimizer offload'
        bad_patch = [p for p in patches if p in PATCHES and not self.patch_ok[p]]
        if bad_patch:
            return None, f'crash: patch {bad_patch[0]} breaks the run'
        if self.t4 and 'reorder_allreduce' in patches and c['overlap_grad']:
            return None, 'crash: reorder_allreduce conflicts with overlap_grad'
        mem = self.memory(c)
        if mem > hw['mem']:
            return None, 'oom'
        ga = GBS // (dp * mbs)
        rc = {'full': 4 / 3, 'selective': 1.1, 'none': 1.0}[c['recompute']]
        eff = 0.45 * {1: 0.8, 2: 0.9, 4: 0.97, 8: 1.0}[mbs] * {1: 1.0, 2: 0.97, 4: 0.93, 8: 0.88}[tp]
        eff *= 1.12 ** self.gs if c['flash_attn'] else 1.0
        t_comp = 6 * m['active'] * GBS * SEQ * rc / (GPUS * hw['tflops'] * eff)
        t_comp *= (0.92 ** self.gs if c['compile'] else 1.0) * (0.98 ** self.gs if c['fused_ce'] else 1.0) * \
            (1.05 if c['deterministic'] else 1.0)
        t_comp *= 1.01 if c['seq_parallel'] else 1.0
        t_bubble = t_comp * (pp - 1) / (ga + pp - 1)
        t_tp = t_comp * hw['tp_share'] * (tp - 1) * (0.4 if c['tp_overlap'] else 1.0)
        grads = 2 * m['params'] / (tp * pp)
        tree = ((0.97 if dp >= 4 else 1.02) if self.t4 else 1.02) if c['nccl_tree'] else 1.0
        t_dp = (grads * 2 * (dp - 1) / dp / hw['link']) * (0.25 ** self.gs if c['overlap_grad'] else 1.0) * tree
        if 'reorder_allreduce' in patches:
            t_dp *= 0.7
        t_opt = 16 * m['params'] / (tp * pp * dp) / hw['hbm'] * 40 * (0.6 ** self.gs if c['fused_adam'] else 1.0)
        if 'fuse_optimizer_kernel' in patches:
            t_opt *= 0.7
        if c['offload']:
            t_opt += 16 * m['params'] / (tp * pp * dp) / 25e9
        t_data = 0.03 * t_comp * (0.5 ** self.gs if c['pin_memory'] else 1.0) * (0.3 if 'prefetch_data' in patches else 1.0)
        host = (0.01 if 'remove_host_sync' in patches else self.host_frac) * t_comp * (0.4 ** self.gs if c['cuda_graphs'] else 1.0)
        t_a2a = 0.0
        if m['moe']:
            t_a2a = 0.08 * t_comp * (0.6 if c['dispatcher'] == 'alltoall' else 1.0) * (1 + 0.1 * math.log2(c['ep']))
        t = t_comp + t_bubble + t_tp + t_dp + t_opt + t_data + host + t_a2a
        if mem > 0.88 * hw['mem'] and not c['expandable_segments']:
            t *= 1.07 ** self.gs                          # allocator retries near the cap
        if self.antagonistic:
            on = [g for g in GOOD8 if c[g]]
            t *= ANTAGONISM_COST ** sum(1 for a, b in itertools.combinations(on, 2) if frozenset((a, b)) in self.antagonistic)
        return t, None

    def hang_step(self, config: dict) -> int | None:
        c = self.full(config)
        return 15 if c['cuda_graphs'] and c['pp'] > 1 else None

    def load_s(self, config: dict) -> float:
        return 30 + 10 * self.m['params'] / 1e9 + (90 if self.full(config)['compile'] else 0)

    def optimum(self) -> tuple[float, dict, tuple]:
        """Exact noiseless optimum over the schema's knobs with every working patch applied. Toggles whose effect never
        depends on anything else are fixed at their good level; the coupled ones are enumerated with the layouts."""
        if self.t4:
            key = (self.task.name, tuple(sorted(self.inert))) if self.inert else self.task.name   # (--inert: its own)
            if key not in _OPTIMA:
                _OPTIMA[key] = self._optimum_t4()
            return _OPTIMA[key]
        good = tuple(p for p in PATCHES if self.patch_ok[p])
        knobs = self.knobs()
        fixed = {t: v for t, v in {'fused_adam': True, 'fused_ce': True, 'flash_attn': True, 'overlap_grad': True,
                                   'pin_memory': True, 'expandable_segments': True, 'nccl_tree': False,
                                   'deterministic': False}.items() if t in knobs}
        free = [n for n in knobs if n not in fixed]
        best = (math.inf, None)
        for values in itertools.product(*[knobs[n] for n in free]):
            c = {**fixed, **dict(zip(free, values))}
            if self.hang_step(c) is not None:
                continue
            t, _ = self.truth(c, good)
            if t is not None and t < best[0]:
                best = (t, c)
        return best[0], best[1], good


    def _optimum_t4(self) -> tuple[float, dict, tuple]:
        """T4: every GOOD8 toggle is enumerated (antagonism, the trap and the patch conflict couple them),
        deterministic stays off, nccl_tree follows dp (its only interaction), the conflicting patch is tried on and off;
        layouts that fail with every toggle off (divisibility, memory) are skipped: toggles do not change either."""
        good = tuple(p for p in PATCHES if self.patch_ok[p])
        plans = [good] + ([tuple(p for p in good if p != 'reorder_allreduce')] if 'reorder_allreduce' in good else [])
        knobs = self.knobs()
        toggles = [g for g in GOOD8 if g in knobs]
        other = [n for n in knobs if n not in toggles and n not in ('nccl_tree', 'deterministic')]
        best = (math.inf, None, good)
        for values in itertools.product(*[knobs[n] for n in other]):
            c0 = {**dict(zip(other, values)), **({'deterministic': False} if 'deterministic' in knobs else {})}
            f = self.full({**c0, **{g: False for g in toggles}})
            if self._truth_full(f, ())[0] is None:     # divisibility, memory, sp/tp-overlap: no toggle changes these
                continue
            if 'nccl_tree' in knobs:
                c0['nccl_tree'] = GPUS // (f['tp'] * f['pp']) >= 4
            for bits in itertools.product((False, True), repeat=len(toggles)):
                c = {**c0, **dict(zip(toggles, bits))}
                if self.hang_step(c) is not None or self.trap(c):
                    continue
                for pl in plans:
                    t, _ = self.truth(c, pl)
                    if t is not None and t < best[0]:
                        best = (t, c, pl)
        return best


# ---------------------------------------------------------------- a memory-coupled job (hls3; FABLE_TAKEAWAYS T2, T3)
MEM_KNOBS = {'micro_batch': [1, 2, 4], 'recompute': ['full', 'partial', 'none'], 'fused_loss': [False, True],
             'compile': [False, True], 'expandable_segments': [False, True], 'keep_params': [False, True],
             'fused_optimizer': [False, True], 'loader_workers': [0, 2, 4]}
MEM_REFERENCE = {'micro_batch': 1, 'recompute': 'full', 'fused_loss': False, 'compile': False, 'expandable_segments': False,
                 'keep_params': False, 'fused_optimizer': False, 'loader_workers': 2}
MEM_CARDS = {'c80': 79.0, 'c96': 94.5, 'c141': 139.8}            # GiB a card offers
MEM_MODELS = {'small': (0.10, 0.20), 'medium': (0.22, 0.32), 'large': (0.34, 0.44)}   # sharded state / card
MEM_GPUS = 8
MEM_LOAD_S = 80.0               # loading dominates a probe: a probe costs more than half a full run (as on the real jobs)


@dataclasses.dataclass(frozen=True)
class MemTask:
    """A memory-coupled job for hls3's toybench: the largest speed moves (less recompute, a larger micro-batch, unsharded
    parameters) spend memory and fit only with a move that frees it (a fused loss, compiled activations, the expandable
    allocator), which alone gains little or is slower; a cliff near the cap (valid, 1.5-4x slower); optionally two
    enablers that crash together; the proposer does not know one enabler's memory effect. Every size and gain is drawn
    per task from its seed within ranges, and which enabler is hidden too: no model's numbers."""
    model: str          # small | medium | large: the sharded training state's share of the card
    card: str           # c80 | c96 | c141
    tight: bool         # the reference near 70 % of the card (else about 55 %)
    antagonism: bool    # compile and the fused loss crash together
    noise: str = 'low'
    enabler: bool = False   # the undeclared enabler (hls3 F16): no recompute fits only with compile, which the model
    #                         declares neutral and believes mildly faster; near the cap the updates scatter (F23)

    @property
    def base_name(self) -> str:
        return f'mem-{self.model}-{self.card}-{"tight" if self.tight else "loose"}-{"anta" if self.antagonism else "free"}-{self.noise}' \
            + ('-enab' if self.enabler else '')

    name = base_name
    schema, hidden, drift = 'mem', 'none', False

    @property
    def gpu(self) -> str:
        return self.card

    @property
    def seed(self) -> int:
        return int(hashlib.sha256(self.base_name.encode()).hexdigest()[:8], 16)


def mem_tasks(n: int | None = None) -> list[MemTask]:
    """The memory-coupled grid (36 tasks), deterministically sub-sampled to n when given."""
    grid = [MemTask(m, c, t, a) for m in MEM_MODELS for c in MEM_CARDS for t in (False, True) for a in (False, True)]
    if n is None or n >= len(grid):
        return grid
    return sorted(grid, key=lambda t: hashlib.sha256(t.name.encode()).hexdigest())[:n]


def enabler_tasks(n: int | None = None) -> list[MemTask]:
    """The undeclared-enabler grid (18 tasks; hls3 F16, FABLE_HLS3_FIXES_R3): the largest memory spender (no recompute)
    fits only together with compile - which frees activations, runs slower alone, and which the simulated model declares
    memory-neutral and never pairs with it - while the declared freeing moves (fused loss, expandable segments) together
    still leave it out of memory; near the card the updates scatter. Every size and gain drawn per task from its seed."""
    grid = [MemTask(m, c, t, False, enabler=True) for m in MEM_MODELS for c in MEM_CARDS for t in (False, True)]
    if n is None or n >= len(grid):
        return grid
    return sorted(grid, key=lambda t: hashlib.sha256(t.name.encode()).hexdigest())[:n]


class MemSim:
    """ToySim's interface (ToyRunner, SimProposer, toybench) over a MemTask; memory in GiB per GPU."""

    def __init__(self, task: MemTask):
        self.task = task
        r = lambda key, lo, hi: lo + (hi - lo) * _u(task.seed, 'mem', key)          # noqa: E731
        self.cv, self.run_sd = (0.01, 0.015) if task.noise == 'low' else (0.03, 0.03)
        self.cap = MEM_CARDS[task.card]
        self.hw = {'mem': self.cap}
        self.state = r('state', *MEM_MODELS[task.model]) * self.cap
        self.frag = r('frag', 0.03, 0.06) * self.cap
        a0, l0 = r('act', 0.05, 0.09), r('logits', 0.04, 0.08)
        target = r('ref', 0.66, 0.74) if task.tight else r('ref', 0.50, 0.58)
        k = max(0.3, (target * self.cap - self.state - self.frag) / ((a0 + l0) * self.cap))
        self.act, self.logits = k * a0 * self.cap, k * l0 * self.cap
        self.rc = {'full': 1.0, 'partial': r('rc_partial', 2.0, 3.0), 'none': r('rc_none', 4.0, 6.5)}
        self.compiled = r('compiled', 0.5, 0.7)     # compiled activations' share
        self.keep = r('keep', 0.6, 1.0) * self.state    # unsharded parameters kept between forward and backward
        self.g = {'partial': r('g_partial', 0.07, 0.12), 'none': r('g_none', 0.18, 0.28), 2: r('g_mb2', 0.05, 0.10),
                  4: r('g_mb4', 0.08, 0.14), 'fused_loss': r('g_fused', 0.0, 0.03), 'compile': -r('g_compile', 0.03, 0.08),
                  'keep_params': r('g_keep', 0.02, 0.05), 'fused_optimizer': r('g_opt', 0.005, 0.015)}
        self.t0 = r('t0', 1.5, 2.5)
        self.hidden_memory = ('compile',) if _u(task.seed, 'mem', 'hide') < 0.5 else ('fused_loss',)
        self.crash_pair = ('compile', 'fused_loss') if task.antagonism else None
        self.declared_neutral = self.optimistic = ()    # (the undeclared-enabler family: what the model believes)
        self.cliff_sd = 0.0                             # (... the scatter of the updates near the cap)
        if getattr(task, 'enabler', False):
            self.compiled = r('compiled_enab', 0.4, 0.6)
            rn = self.rc['none']
            lo = (self.cap - self.state - self.logits / 3 - 0.25 * self.frag) / rn   # above: no recompute with the fused
            hi = (0.9 * self.cap - self.state - self.logits - self.frag) / (self.compiled * rn)   # loss and the allocator
            self.act = lo + (hi - lo) * r('act_enab', 0.35, 0.65) if hi > lo else self.act   # out of memory; below: with
            self.hidden_memory = self.declared_neutral = self.optimistic = ('compile',)    # compile it fits off the cliff
            self.cliff_sd = r('cliff_sd', 0.05, 0.10)
        self.patch_ok: dict = {}
        self.memory_facts = {'cap_gib': self.cap, 'world': MEM_GPUS, 'params_b': round(self.state * MEM_GPUS / 16 * 2 ** 30 / 1e9, 3),
                             'coupled': []}
        # (the candidates' extended prior, simulated: the cost each knob acts on, and the memory terms below)
        self.families = {'micro_batch': 'activations', 'recompute': 'activations', 'compile': 'kernel',
                         'fused_loss': 'logits', 'expandable_segments': 'allocator', 'keep_params': 'communication',
                         'fused_optimizer': 'optimizer', 'loader_workers': 'input'}

    def mem_terms(self, proposer) -> list[dict]:
        """The memory terms a simulated model writes for the candidates' prior (esls): this sim's own structure, each
        term known with the model's probability q and its constant off by LogNormal(0, rho / 2); a knob whose memory
        effect the task hides contributes no factor (the model does not know it frees memory)."""
        from .proposers import _gauss
        hidden = set(getattr(self, 'hidden_memory', ()) or ())

        def c(name, x):
            return round(x * math.exp(0.5 * proposer.rho * _gauss(self.task.seed, proposer.seed, 'term', name)), 4)
        rc = f"(1 if recompute == 'full' else {c('rc_partial', self.rc['partial'])} if recompute == 'partial' " \
             f"else {c('rc_none', self.rc['none'])})"
        act = f"{c('act', self.act)} * micro_batch * {rc}" + \
            ('' if 'compile' in hidden else f" * ({c('compiled', self.compiled)} if compile else 1)")
        logits = f"{c('logits', self.logits)} * micro_batch" + ('' if 'fused_loss' in hidden else ' * (0.333 if fused_loss else 1)')
        terms = [{'name': 'activations', 'gib_expr': act}, {'name': 'logits', 'gib_expr': logits},
                 {'name': 'fragmentation', 'gib_expr': f"{c('frag', self.frag)} * (0.25 if expandable_segments else 1)"},
                 {'name': 'kept parameters', 'gib_expr': f"{c('keep', self.keep)}", 'when': {'keep_params': True}}]
        return [t for t in terms if proposer.knows(f'term:{t["name"]}')]

    def knobs(self) -> dict:
        return {k: list(v) for k, v in MEM_KNOBS.items()}

    def reference(self) -> dict:
        return dict(MEM_REFERENCE)

    def full(self, config: dict) -> dict:
        return {**MEM_REFERENCE, **config}

    def memory(self, c: dict) -> float:
        c = self.full(c)
        act = self.act * c['micro_batch'] * self.rc[c['recompute']] * (self.compiled if c['compile'] else 1.0)
        logits = self.logits * c['micro_batch'] * (1 / 3 if c['fused_loss'] else 1.0)
        return self.state + act + logits + (0.25 if c['expandable_segments'] else 1.0) * self.frag + \
            (self.keep if c['keep_params'] else 0.0)

    def truth(self, config: dict, patches=()) -> tuple[float | None, str | None]:
        c = self.full(config)
        if self.crash_pair and all(c[k] for k in self.crash_pair):
            return None, 'crash: ' + ' with '.join(self.crash_pair) + ' fails at start'
        mem = self.memory(c)
        if mem > self.cap:
            return None, 'oom'
        lg = self.g.get(c['recompute'], 0.0) + self.g.get(c['micro_batch'], 0.0) + \
            sum(self.g[k] for k in ('fused_loss', 'compile', 'keep_params', 'fused_optimizer') if c[k]) + \
            {0: -0.02, 2: 0.0, 4: 0.004}[c['loader_workers']]
        t = self.t0 * math.exp(-lg)
        if mem > 0.92 * self.cap:                  # the cliff: the allocator frees and retries every update (T3)
            t *= 1.5 + 30 * (mem / self.cap - 0.92)
        return t, None

    def jitter(self, config: dict) -> float:
        """The spread (log sd) of an update's time on top of the noise: on the cliff, in the undeclared-enabler family."""
        return self.cliff_sd if self.cliff_sd and self.memory(config) > 0.92 * self.cap else 0.0

    def oom_text(self, config: dict) -> str:
        """PyTorch's message for this configuration's out-of-memory failure (what the run's log would say)."""
        mem = self.memory(config)
        tried = max(0.5, 0.2 * (mem - self.state))
        free = max(0.05, self.cap - (mem - tried))
        return (f'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate {tried:.2f} GiB. GPU 0 has a total capacity '
                f'of {self.cap:.2f} GiB of which {free:.2f} GiB is free. Including non-PyTorch memory, this process has '
                f'{self.cap - free:.2f} GiB memory in use.')

    def hang_step(self, config: dict) -> int | None:
        return None

    def trap(self, config: dict) -> bool:
        return False

    def load_s(self, config: dict) -> float:
        return MEM_LOAD_S + (90.0 if self.full(config)['compile'] else 0.0)

    def optimum(self) -> tuple[float, dict, tuple]:
        best = (math.inf, None)
        knobs = self.knobs()
        for values in itertools.product(*knobs.values()):
            c = dict(zip(knobs, values))
            t, _ = self.truth(c)
            if t is not None and t < best[0]:
                best = (t, c)
        return best[0], best[1], ()


# ---------------------------------------------------------------- the coupling family (wcs; FABLE_ARCH_REVIEW section 4.3)
NOISE_KNOBS = {'matmul_precision': ['default', 'relaxed'], 'pin_host_memory': [False, True], 'deterministic': [False, True],
               'prefetch_factor': [2, 4], 'collective_channels': ['default', '16', '32'], 'logging_interval': [1, 10]}
NOISE_REFERENCE = {'matmul_precision': 'default', 'pin_host_memory': False, 'deterministic': False, 'prefetch_factor': 2,
                   'collective_channels': 'default', 'logging_interval': 1}


@dataclasses.dataclass(frozen=True)
class CoupleTask(MemTask):
    """A memory-coupled job with the noise knobs of a real table (six knobs within +-1.5 %, one harmful) and one pair
    that only pays together (the collective channels with the fused loss): the landscape of the recorded arms - a few
    big coupled knobs among many that do nothing - for wcs's surrogate and its coverage screen."""

    @property
    def base_name(self) -> str:
        return 'couple-' + MemTask.base_name.fget(self)

    name = base_name


def couple_tasks(n: int | None = None) -> list[CoupleTask]:
    grid = [CoupleTask(m, c, t, a) for m in MEM_MODELS for c in MEM_CARDS for t in (False, True) for a in (False, True)]
    if n is None or n >= len(grid):
        return grid
    return sorted(grid, key=lambda t: hashlib.sha256(t.name.encode()).hexdigest())[:n]


class CoupleSim(MemSim):
    """MemSim with the noise knobs: each within +-1.5 % (drawn per task), `deterministic` -2 %, and the collective
    channels '32' +4..8 % only with the fused loss on (else 0); memory unchanged by them."""

    def __init__(self, task: CoupleTask):
        super().__init__(task)
        r = lambda key, lo, hi: lo + (hi - lo) * _u(task.seed, 'couple', key)      # noqa: E731
        self.noise_g = {'matmul_precision': r('mp', -0.005, 0.015), 'pin_host_memory': r('pin', -0.005, 0.01),
                        'deterministic': -0.02, 'prefetch_factor': r('pf', -0.005, 0.005), 'logging_interval': r('li', 0.0, 0.012)}
        self.channels_g = {'16': r('ch16', 0.0, 0.01), '32': r('ch32', 0.04, 0.08)}

    def knobs(self) -> dict:
        return {**MEM_KNOBS, **{k: list(v) for k, v in NOISE_KNOBS.items()}}

    def reference(self) -> dict:
        return {**MEM_REFERENCE, **NOISE_REFERENCE}

    def full(self, config: dict) -> dict:
        return {**MEM_REFERENCE, **NOISE_REFERENCE, **config}

    def truth(self, config: dict, patches=()) -> tuple[float | None, str | None]:
        t, why = super().truth(config, patches)
        if t is None:
            return t, why
        c = self.full(config)
        lg = sum(g for k, g in self.noise_g.items() if c[k] != NOISE_REFERENCE[k])
        ch = c['collective_channels']
        if ch != 'default' and c['fused_loss']:
            lg += self.channels_g[ch]
        return t * math.exp(-lg), None


def make_sim(task):
    if isinstance(task, CoupleTask):
        return CoupleSim(task)
    return MemSim(task) if isinstance(task, MemTask) else ToySim(task)


def _patches(overlay: dict | None) -> tuple:
    text = (overlay or {}).get('patches.txt') or ''
    return tuple(sorted({ln.strip() for ln in text.splitlines() if ln.strip()}))


class ToyRunner:
    """ScreenRunner over a ToySim: signed project receipts, a virtual clock, fidelities and early stop."""
    stop_grace_s = 0.0

    def __init__(self, sim: ToySim, clock, *, lr_decay: bool = False):
        self.sim, self.clock = sim, clock
        self.lr_decay = lr_decay                 # the lr decays linearly to the horizon ({steps}), as HF schedules do
        self.attempts: dict[str, int] = {}
        self.log: list[dict] = []

    def describe(self) -> dict:
        return {'kind': 'ToyRunner', 'task': self.sim.task.name}

    def run(self, task, config: dict, overlay: dict | None, purpose: str, run_id: str, **kw) -> dict:
        sim = self.sim
        horizon = int(kw.get('steps') or 30)                       # the {steps} the job gets
        steps = min(horizon, int(kw.get('stop_after') or horizon))   # the updates it runs (a probe: stopped after them)
        proofs = [x[len('proof:'):] for x in _patches(overlay) if x.startswith('proof:')]   # hls3 (F10): patches that
        defers = {x[len('defer:'):] for x in _patches(overlay) if x.startswith('defer:')}   # print their two lines (F21:
        patches = tuple(x for x in _patches(overlay) if not x.startswith(('proof:', 'defer:')))   # and `deferred`)
        key = digest({'c': sim.full(config), 'p': patches})
        n = self.attempts.get(f'{key}:{steps}', 0)
        self.attempts[f'{key}:{steps}'] = n + 1
        truth, why = sim.truth(config, patches)
        hang = sim.hang_step(config)
        trapped = sim.trap(config)
        a, b = WINDOWS.get(steps, (max(2, steps // 3 + 1), steps))
        if sim.task.drift:      # common part sd run_sd / sqrt 2, one part per window half sd run_sd: the run mean keeps sd
            mid = (a + b) // 2  # run_sd and the split half has the sd of a pair of runs (as measured on real jobs)
            common = sim.run_sd / math.sqrt(2) * _gauss(sim.task.seed, key, steps, n, 'run')
            half = [math.exp(common + sim.run_sd * _gauss(sim.task.seed, key, steps, n, h)) for h in ('half1', 'half2')]
            offset_of = lambda s: half[s > mid]                                              # noqa: E731
        else:
            offset = math.exp(sim.run_sd * _gauss(sim.task.seed, key, steps, n, 'run'))
            offset_of = lambda s: offset                                                     # noqa: E731
        rows, t, exit_code, events, stopped = [], 0.0, 0, {}, False
        load = sim.load_s(config)
        jit = sim.jitter(config) if truth is not None and hasattr(sim, 'jitter') else 0.0   # (scattered updates)
        if truth is None:
            exit_code, done = 1, 0
            events = {'oom': 1} if why == 'oom' else {'traceback': 1}
            if why == 'oom' and hasattr(sim, 'oom_text'):   # the memory-coupled job's log names the failed allocation
                events['error'] = sim.oom_text(config)
            spent = 10.0 if why == 'divisibility' else load + 5.0      # a bad layout fails before loading the model
        else:
            done = 0
            for s in range(1, steps + 1):
                if hang and s >= hang:
                    exit_code, why = None, 'hang'
                    break
                warm = {1: 1.35, 2: 1.15, 3: 1.05}.get(s, 1.0)
                dt = truth * offset_of(s) * warm * math.exp(sim.cv * _gauss(sim.task.seed, key, steps, n, s))
                if jit:
                    dt *= math.exp(jit * _gauss(sim.task.seed, key, steps, n, s, 'jitter'))
                t += dt
                rows.append({'step': s, 't': t, 'loss': round(2.5 * math.exp(-0.02 * s) * (1.06 if trapped else 1.0), 6),
                             'grad_norm': round(1.0 + 0.5 * math.exp(-0.05 * s), 6),
                             'lr': 1e-4 * min(1.0, s / 10) * ((1 - (s - 1) / horizon) if self.lr_decay else 1.0),
                             'tokens': GBS * SEQ, 'dt': dt})
                done = s
                if kw.get('stop') is not None and s < steps and kw['stop'](rows):
                    stopped, exit_code = True, -15
                    break
            spent = load + t + (1800.0 if why == 'hang' else 0.0)       # a hang burns the timeout
            dead = getattr(sim, 'inert', None) or ()                     # (an inert patch: installed, never ran)
            lines = [ln for p in sorted(proofs) if p in patches and done
                     for ln in (f'[infrarsi-patch] {p}: found off -> set on',)
                     + ((f'[infrarsi-patch] {p}: deferred (its hook hands every call back)',)
                        if p in dead and p in defers else ())
                     + (() if p in dead else (f'[infrarsi-patch] {p}: active',))]
            if lines:
                events['patch_lines'] = lines
        self.clock.advance(spent)
        win = [r['dt'] for r in rows if a <= r['step'] <= b]
        st = sum(win) / len(win) if exit_code == 0 and len(win) == b - a + 1 else None
        rsd = (math.sqrt(sum((x - st) ** 2 for x in win) / (len(win) - 1)) / st) if st and len(win) > 1 else None
        ev = {'steps': horizon, 'stop_after': steps if steps < horizon else None, 'steps_completed': done,
              'planned_stop': steps < horizon and exit_code == 0 and done >= steps,
              'exit_code': exit_code, 'timed_out': why == 'hang', 'window': [a, b],
              'per_step': [{k: r[k] for k in ('step', 'loss', 'grad_norm', 'lr', 'tokens', 't')} for r in rows],
              'step_time': st, 'clock_step_time': st, 'reported_step_time': None, 'rsd_window': rsd, 'rsd_robust': robust_rsd(win) if st else None,
              'split_half': split_half(rows, [a, b]) if st else None,
              'split_half_robust': split_half(rows, [a, b], robust=True) if st else None, 'events': {**events, 'nan_inf_steps': []}, 'memory_peak_gib': sim.memory(sim.full(config)) if truth else None,
              'memory_cap_gib': sim.hw['mem'], 'meta': {}, 'frozen': {'checked': False}, 'static_workload': {'gbs': GBS},
              'stopped_early': stopped, 'failure': why}
        receipt = {'schema': SCHEMA, 'project': sim.task.name, 'task_id': sim.task.name, 'run_id': run_id,
                   'attempt_id': new_attempt_id(), 'candidate_id': kw.get('candidate_id'), 'purpose': purpose,
                   'config': dict(config), 'overlay': dict(overlay or {}), 'world_size': GPUS, 'evidence': ev,
                   'valid': exit_code == 0, 'invalid_reasons': [] if exit_code == 0 else [why or 'stopped'],
                   'step_time': {'mean_window_max_rank_s': st}, 'metric': {'name': 'step_time_s', 'value': st, 'direction': 'min'},
                   'cost': {'gpu_seconds': GPUS * spent, 'wall_seconds': spent, 'accounting_complete': True},
                   'utc_end': utc_now()}
        self.log.append({'candidate_id': kw.get('candidate_id'), 'steps': horizon, 'stop_after': steps if steps < horizon else None,
                         'done': done, 'why': why, 'stopped': stopped,
                         'gpu_h': GPUS * spent / 3600, 'truth': truth})
        return signing.sign(receipt)


def _gauss(*key) -> float:
    u1, u2 = max(1e-12, _u(*key, 'a')), _u(*key, 'b')
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def toy_task_spec(sim: ToySim, budget: dict) -> dict:
    """The ext.common.EnvTask spec of a ToySim (knob schema, patches.txt as the mutable file)."""
    props = {k: ({'type': 'boolean'} if v == [False, True] else {'enum': v}) for k, v in sim.knobs().items()}
    return {'schema': 'infrarsi-ext-task-v1', 'id': sim.task.name, 'env': 'toytrain', 'split': 'toy',
            'metric_name': 'step_time_s', 'action_space': 'knobs+patch', 'knob_schema': {'type': 'object', 'properties': props},
            'frozen_files': {}, 'mutable_globs': ['patches.txt'], 'max_patch_lines': 20, 'budget': budget,
            'margin': {'kind': 'rsd', 'floor': 0.03, 'k': 3}, 'tau': math.log(1.10), 'world_size': GPUS,
            'facts': {'model': sim.task.model, 'gpu': sim.task.gpu, 'schema': sim.task.schema,
                      'patch_mechanisms': sorted(PATCHES)}}


def random_seed_for(task: ToyTask, arm: str) -> int:
    return random.Random(f'{task.name}:{arm}').randrange(2 ** 31)
