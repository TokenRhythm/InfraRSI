"""B-RND: random points of OptiCo's parameter set (design §3.4/§3.5), seed 42, the same validator as every non-native arm.

Conditional sampling in the style of baselines/random_l0 + baselines/space (a key is drawn only where it can matter;
every categorical is uniform over its own choices; the final point must pass validate_optico_params and be new):
  1. layout (TP, PP, MBS): uniform over the tuples for which some valid pipeline split exists (TP*PP | 8, TP divides
     heads/KV groups/FFN, GBS % (MBS*DP) == 0);
  2. split (PP > 1 only): mode uniform over the modes that have a valid split for this PP -- 'even' (when
     NUM_LAYERS % PP == 0), 'first', 'last', 'both' -- then the value(s) uniform over that mode's valid splits;
  3. AC uniform over none/sel/full/offload among those valid for the layout (offload needs PP 1); AC_LAYERS for full
     uniform over the common divisors of the stage sizes that also pass the script's (NUM_LAYERS / PP) % AC_LAYERS
     check; for offload uniform over the divisors of NUM_LAYERS below NUM_LAYERS;
  4. OVERLAP_GRAD_REDUCE uniform; OVERLAP_PARAM_GATHER drawn only when grad-reduce is on (Megatron needs it); both off
     with offload (the script drops them); FUSED_CROSS_ENTROPY uniform.
Draws that are invalid or already seen cost no trial; the sampler gives up after MAX_DRAWS.
"""
from __future__ import annotations

import itertools
import random

from . import params as pm
from . import workloads as wl

SEED = 42
MAX_DRAWS = 5000


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


class OptiCoRandomSampler:
    def __init__(self, workload, seed: int = SEED):
        self.w = wl.get(workload)
        self.seed = seed
        self.rng = random.Random(seed)
        self.draws = 0
        n = self.w.num_layers
        self.splits = {}                     # pp -> {mode: [(first, last, stages)]}
        for pp in pm.PP_CHOICES:
            modes = {'even': [], 'first': [], 'last': [], 'both': []}
            if pp == 1:
                modes['even'].append((None, None, [n]))
            else:
                opts = [None] + list(range(1, n))
                for f, l in itertools.product(opts, opts):
                    stages, errs = pm.stage_layers(n, pp, f, l)
                    if errs:
                        continue
                    mode = 'even' if f is None and l is None else ('first' if l is None else ('last' if f is None else 'both'))
                    modes[mode].append((f, l, stages))
            self.splits[pp] = {m: v for m, v in modes.items() if v}
        self.layouts = []
        for tp, pp, mbs in itertools.product(pm.TP_CHOICES, pm.PP_CHOICES, pm.MBS_CHOICES):
            if not self.splits[pp]:
                continue
            probe = self.w.knobs(TP=tp, PP=pp, MICRO_BATCH_SIZE=mbs, AC='none', AC_LAYERS=1, OVERLAP_GRAD_REDUCE=False,
                                 OVERLAP_PARAM_GATHER=False)
            mode = next(iter(self.splits[pp]))
            f, l, _ = self.splits[pp][mode][0]
            probe.update(FIRST_PP_NUM_LAYERS=f, LAST_PP_NUM_LAYERS=l)
            if not pm.validate_optico_params(probe, self.w)[1]:
                self.layouts.append((tp, pp, mbs))

    def draw(self) -> dict:
        r, n = self.rng, self.w.num_layers
        self.draws += 1
        tp, pp, mbs = r.choice(self.layouts)
        first = last = None
        stages = [n // pp] * pp
        if pp > 1:
            mode = r.choice(sorted(self.splits[pp]))
            first, last, stages = r.choice(self.splits[pp][mode])
        acs = [a for a in pm.AC_CHOICES if a != 'offload' or pp == 1]
        ac = r.choice(acs)
        k = 1
        if ac == 'full':                     # 1 always qualifies, so the list is never empty
            k = r.choice([d for d in _divisors(min(stages)) if all(s % d == 0 for s in stages) and (n // pp) % d == 0])
        elif ac == 'offload':
            k = r.choice([d for d in _divisors(n) if d < n])
        ogr = opg = False
        if ac != 'offload':
            ogr = r.choice([False, True])
            if ogr:
                opg = r.choice([False, True])
        fce = r.choice([False, True])
        return {'MICRO_BATCH_SIZE': mbs, 'TP': tp, 'PP': pp, 'OVERLAP_GRAD_REDUCE': ogr, 'OVERLAP_PARAM_GATHER': opg,
                'AC': ac, 'AC_LAYERS': k, 'FUSED_CROSS_ENTROPY': fce, 'FIRST_PP_NUM_LAYERS': first, 'LAST_PP_NUM_LAYERS': last}

    def next(self, seen=()) -> dict | None:
        """A new valid parameter set (normalized), or None after MAX_DRAWS draws without one."""
        seen = set(seen)
        for _ in range(MAX_DRAWS):
            cand = self.draw()
            norm, errs, _ = pm.validate_optico_params(cand, self.w)
            if errs:
                continue
            key = pm.params_key(norm)
            if key in seen:
                continue
            return norm
        return None

    def sequence(self, n: int, seen=()) -> list[dict]:
        out, keys = [], set(seen)
        for _ in range(n):
            p = self.next(keys)
            if p is None:
                break
            keys.add(pm.params_key(p))
            out.append(p)
        return out
