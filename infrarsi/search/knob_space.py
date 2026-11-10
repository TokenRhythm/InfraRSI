"""Non-LLM search over a knob space (any hooks task: projects and external environments): random and TPE.

Both propose complete knob configs through the shared SearchSession (validation, duplicates, budget, verdicts), so
their traces are comparable with every other strategy. They propose only what the session's own check
(SearchSession.check_config: the knob schema, constraints, workload invariants) accepts as new, so a refused draw costs
no run and no refused-proposal count. Configurations are drawn in a random order without replacement and checked once
each, up to CHECK_CAP of them; when the whole space was examined and every feasible one proposed, the search ends
`strategy_done`. TPE's refused suggestions are told FAIL, which it ignores: its random start is feasible, and after
MAX_DRAWS_PER_SCREEN fruitless suggestions in a row its next trial is a feasible configuration it has not proposed.
"""
from __future__ import annotations

import itertools
import math
import random

# the session's check takes 40-150 us per configuration (the toy; 12 knobs with 5 constraint and invariant
# expressions): at most this many configurations are checked, each once and only as far as the search draws, so 12-45 s
# in the worst case once per search; of a larger space this many, drawn without replacement, are examined
CHECK_CAP = 300_000
# TPE can suggest measured or refused configurations indefinitely (a stalled one found nothing new in 2000 asks, a
# working one needs about 10 per new configuration), and each repeat told its value stays in the study and slows every
# later ask (10 ms at 500 trials, 70 ms at 4000): after this many in a row its next trial is a feasible one it has not
# proposed
MAX_DRAWS_PER_SCREEN = 100


def choices(task) -> dict[str, list]:
    """Finite choices per knob from the knob schema (enum, booleans, small integer ranges, 8-point number grids)."""
    out = {}
    for name, s in ((task.knob_schema or {}).get('properties') or {}).items():
        if s.get('enum'):
            out[name] = list(s['enum'])
        elif s.get('type') == 'boolean':
            out[name] = [False, True]
        elif s.get('type') == 'integer' and 'minimum' in s and 'maximum' in s:
            lo, hi = int(s['minimum']), int(s['maximum'])
            if hi - lo <= 64:
                out[name] = list(range(lo, hi + 1))
            elif lo < 0:                                 # a range through negatives: an even 16-point grid
                out[name] = sorted({lo + round((hi - lo) * i / 15) for i in range(16)})
            else:                                        # geometric from the first positive value (a 0 kept as is)
                a = max(lo, 1)
                out[name] = sorted({int(round(a * (hi / a) ** (i / 15))) for i in range(16)} | {lo, hi})
        elif s.get('type') == 'number' and 'minimum' in s and 'maximum' in s:
            lo, hi = float(s['minimum']), float(s['maximum'])
            out[name] = [lo + (hi - lo) * i / 7 for i in range(8)]
    return out


def _stop(session) -> str | None:
    poll = getattr(session, 'poll_control', None)
    if poll is not None:
        poll()
    if session.stop_flag:
        return session.stop_flag
    ex = session.ledger.exhausted()
    return 'budget:' + ex[0] if ex else None


def _feasible(session, space: dict, rng: random.Random):
    """(draws, whole): the configurations of the space that the session would accept as new, lazily, in a uniformly
    random order without replacement (an index of the product, decoded by mixed radix in `space` order and checked
    once; one that TPE proposed meanwhile is skipped), and whether the draws examine the whole space rather than
    CHECK_CAP configurations of it."""
    c0, names, n = session.candidates['c0'], list(space), math.prod(len(v) for v in space.values())

    def indices():
        if n <= CHECK_CAP:
            yield from rng.sample(range(n), n)
            return
        seen = set()                                     # sample()'s own method for a few of many: range(n) of a space
        while len(seen) < CHECK_CAP:                     # over 2**63 configurations has no len()
            i = rng.randrange(n)
            if i not in seen:
                seen.add(i)
                yield i

    def draws():
        for i in indices():
            cfg = {}
            for name in names:
                i, j = divmod(i, len(space[name]))
                cfg[name] = space[name][j]
            key = session.check_config(dict(cfg), c0)[0]
            if key is not None and key not in session.by_key:
                yield cfg
    return draws(), n <= CHECK_CAP


def random_search(session, *, seed: int = 42) -> str:
    rng = random.Random(seed)
    space = choices(session.task)
    if not space:
        return 'no_knobs'
    draws, whole = _feasible(session, space, rng)
    while True:
        stop = _stop(session)
        if stop:
            return stop
        cfg = next(draws, None)                          # it checks configurations: the budget can run out meanwhile
        if cfg is None:                                  # every configuration examined, or CHECK_CAP of them
            return _stop(session) or ('strategy_done' if whole else 'no_valid_sample')
        session.screen(session.propose_config(config=cfg, base='c0', origin='random')['candidate_id'])


def tpe_search(session, *, seed: int = 42, n_startup_trials: int = 4) -> str:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError('the tpe strategy needs optuna (pip install optuna); or use --strategy random') from exc
    from optuna.distributions import CategoricalDistribution
    from optuna.trial import TrialState, create_trial
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = choices(session.task)
    if not space:
        return 'no_knobs'
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials))
    dists = {k: CategoricalDistribution(v) for k, v in space.items()}

    def penalty() -> float:
        vals = [c.verdict['step_time'] for c in session.candidates.values() if c.verdict and c.verdict.get('valid') and c.verdict.get('step_time')]
        return 2.0 * max(vals) if vals else 1e6

    start = session.candidates['c0']
    if start.verdict and all(start.config.get(k) in v for k, v in space.items()):
        val = start.verdict['step_time'] if start.verdict.get('valid') and start.verdict.get('step_time') else penalty()
        study.add_trial(create_trial(params={k: start.config[k] for k in space}, distributions=dists, value=val))
    draws, whole = _feasible(session, space, random.Random(seed))
    for cfg in itertools.islice(draws, max(0, n_startup_trials - len(study.trials))):
        study.enqueue_trial(cfg)                         # a feasible random start (TPE's own draws would mostly be refused)
    misses = 0
    while True:
        stop = _stop(session)
        if stop:
            return stop
        if misses >= MAX_DRAWS_PER_SCREEN:               # also how the search ends: at most this many asks at the end
            cfg = next(draws, None)                      # it checks configurations: the budget can run out meanwhile
            if cfg is None:                              # every configuration examined, or CHECK_CAP of them
                return _stop(session) or ('strategy_done' if whole else 'no_new_valid_sample')
            study.enqueue_trial(cfg)
            misses = 0
        trial = study.ask()
        cfg = {k: trial.suggest_categorical(k, v) for k, v in space.items()}
        key = session.check_config(cfg, start)[0]
        if key is not None and key not in session.by_key:
            fb = session.propose_config(config=cfg, base='c0', origin='tpe')
            res = session.screen(fb['candidate_id'])
            if 'error' in res:
                study.tell(trial, state=TrialState.FAIL)
            else:
                study.tell(trial, res['step_time'] if res['valid'] and res['step_time'] else penalty())
            misses = 0
            continue
        dup = session.candidates[session.by_key[key]] if key in session.by_key else None
        if dup is not None and dup.verdict:              # proposed before: told that configuration's value
            ok = dup.verdict.get('valid') and dup.verdict.get('step_time')
            study.tell(trial, dup.verdict['step_time'] if ok else penalty())
        else:                                            # refused (no run, no count) or never measured
            study.tell(trial, state=TrialState.FAIL)     # FAIL: a penalty saved no asks and found slower configurations
        misses += 1                                      # refused ones too: only-refused suggestions would never end


def space_size(task) -> float:
    return math.prod(len(v) for v in choices(task).values()) if choices(task) else 0
