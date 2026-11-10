"""random_L0 (design E2, config track): random points of the L0 space from the ED start, seed 42.

Points are drawn with the conditional sampler (keys only where they can matter); draws rejected by validate_l0
cost no screen. Same budget and stop rules as every arm.
"""
from __future__ import annotations

import random

from ..agent.search import SearchTrace
from .common import new_session, screen_start, space_for
from .space import sample_config

MAX_DRAWS_PER_SCREEN = 2000


def run_random(task, budget, start_config: dict, runner, *, judge=None, model_facts=None, seed: int = 42,
               run_id: str = 'random_L0', out_dir=None, start_receipt=None, reference_receipt=None, clock=None) -> SearchTrace:
    rng = random.Random(seed)
    session = new_session(task=task, budget=budget, runner=runner, judge=judge, run_id=run_id, arm='random_L0', out_dir=out_dir,
                          start_config=start_config, start_receipt=start_receipt, reference_receipt=reference_receipt,
                          model_facts=model_facts, clock=clock)
    space = space_for(task, model_facts)
    screen_start(session)
    stop, draws = None, 0
    while stop is None:
        if session.stop_flag or session.ledger.exhausted():
            stop = session.stop_flag or 'budget:' + session.ledger.exhausted()[0]
            break
        cid = None
        for _ in range(MAX_DRAWS_PER_SCREEN):
            draws += 1
            _, config = sample_config(lambda name, choices: rng.choice(choices), space, model_facts)
            fb = session.propose_config(config=config, base='c0', origin='random')
            if 'candidate_id' in fb:
                cid = fb['candidate_id']
                break
        if cid is None:
            stop = 'no_valid_sample'
            break
        session.screen(cid)
    return session.finalize(stop_reason=stop, extra={'seed': seed, 'draws': draws, 'space_keys': sorted(space)})
