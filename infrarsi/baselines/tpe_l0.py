"""tpe_L0 (design E2, config track): Optuna TPESampler(seed=42) over the L0 space, minimizing StepTime.

Define-by-run over the conditional sampler. The ED start enters as the first (completed or enqueued) trial.
Points rejected by validate_l0 are told FAIL (no screen spent); screened but invalid candidates are told
2x the worst valid StepTime so TPE learns infeasible regions.
"""
from __future__ import annotations

from ..agent.search import SearchTrace
from .common import new_session, screen_start, space_for
from .space import config_to_trial_params, sample_config

MAX_ASKS_PER_SCREEN = 500


def run_tpe(task, budget, start_config: dict, runner, *, judge=None, model_facts=None, seed: int = 42,
            run_id: str = 'tpe_L0', out_dir=None, start_receipt=None, reference_receipt=None, clock=None,
            n_startup_trials: int = 4) -> SearchTrace:
    import optuna
    from optuna.distributions import CategoricalDistribution
    from optuna.trial import TrialState, create_trial

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    session = new_session(task=task, budget=budget, runner=runner, judge=judge, run_id=run_id, arm='tpe_L0', out_dir=out_dir,
                          start_config=start_config, start_receipt=start_receipt, reference_receipt=reference_receipt,
                          model_facts=model_facts, clock=clock)
    space = space_for(task, model_facts)
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials)
    study = optuna.create_study(direction='minimize', sampler=sampler)

    def penalty() -> float:
        vals = [c.verdict['step_time'] for c in session.candidates.values() if c.verdict and c.verdict.get('valid') and c.verdict.get('step_time')]
        return 2.0 * max(vals) if vals else 1e6

    screen_start(session)
    start = session.candidates['c0']
    params, choices = config_to_trial_params(start.config, space, model_facts)
    if start.verdict:
        v = start.verdict['step_time'] if start.verdict.get('valid') and start.verdict.get('step_time') else penalty()
        study.add_trial(create_trial(params=params, distributions={k: CategoricalDistribution(c) for k, c in choices.items()}, value=v))
    stop, asks, fails_in_row = None, 0, 0
    while stop is None:
        if session.stop_flag or session.ledger.exhausted():
            stop = session.stop_flag or 'budget:' + session.ledger.exhausted()[0]
            break
        trial = study.ask()
        asks += 1
        _, config = sample_config(lambda name, ch: trial.suggest_categorical(name, ch), space, model_facts)
        fb = session.propose_config(config=config, base='c0', origin='tpe')
        if 'candidate_id' in fb:
            res = session.screen(fb['candidate_id'])
            if 'error' in res:
                study.tell(trial, state=TrialState.FAIL)
            else:
                study.tell(trial, res['step_time'] if res['valid'] and res['step_time'] else penalty())
            fails_in_row = 0
            continue
        if 'duplicate_of' in fb:                    # re-tell the known result; no screen
            dup = session.candidates[fb['duplicate_of']]
            if dup.verdict and dup.verdict.get('valid') and dup.verdict.get('step_time'):
                study.tell(trial, dup.verdict['step_time'])
            elif dup.verdict:
                study.tell(trial, penalty())
            else:
                study.tell(trial, state=TrialState.FAIL)
        else:
            study.tell(trial, state=TrialState.FAIL)
        fails_in_row += 1
        if fails_in_row >= MAX_ASKS_PER_SCREEN:
            stop = 'no_new_valid_sample'
    return session.finalize(stop_reason=stop, extra={'seed': seed, 'sampler': 'optuna.TPESampler', 'optuna': optuna.__version__,
                                                     'asks': asks, 'space_keys': sorted(space)})
