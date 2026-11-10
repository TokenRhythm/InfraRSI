"""Search-state checkpoints of a running search (agent/state.py has the protocol and the files): what is captured after each
observed run and at the end of the search, and how it is put back onto the objects of a resuming run.

    Writer(directory, flow=...)      the Engine's `checkpoint` hook: after a proposal chosen with a model call (the engine's
                                     `_pending`: a resumed search runs it, the paid answer is not asked again), after an
                                     observed run that spent something (a probe, a short or full run, a re-run, a model
                                     call) and at the end of the search, it writes
                                     <directory>/checkpoint.pkl (+ .prev.pkl, .json). Passive: it reads the session, the
                                     engine and the strategy and changes none of them
    capture(session, engine, strategy)   the state: session (candidates with config, overlay files, receipts, verdicts;
                                     screens, events, the model-call log, the ledger's used amounts and wall clock),
                                     engine (incumbent, noise pairs, tried keys, confirmations ...), the strategy's own
                                     attributes and parts (the proposer's counters, the budget driver's attempts and dollars,
                                     the call guard), every random generator's state, and the flow's own (`flow`)
    restore(state, session, engine, strategy)   the other way, onto objects built by the resuming run; the live handles stay
                                     the resuming run's

The state is pickled with the live objects as references to names (`live_objects`), so a reference to the strategy inside a
proposal's gate, or to the engine inside the ladder's memo, comes back as a reference to the new object.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..agent import state as S

def driver_chain(strategy) -> list:
    """The model driver of a strategy from the outside in: the budget driver, the call guard inside it, the backend (each
    wrapper keeps the next in `_inner`)."""
    out, o = [], getattr(strategy, '_bdriver', None)
    while o is not None and len(out) < 8:
        out.append(o)
        o = vars(o).get('_inner') if hasattr(o, '__dict__') else None
    return out


def live_objects(session, engine, strategy, extra: dict | None = None) -> dict:
    """The live objects of a run by the names they are written under in a checkpoint."""
    out = {'strategy': strategy, 'engine': engine, 'session': session, 'proposer': getattr(strategy, 'proposer', None),
           'hooks': getattr(session, 'hooks', None), 'runner': getattr(session, 'runner', None),
           'judge': getattr(session, 'judge', None), 'task': getattr(session, 'task', None)}
    out.update({f'driver{i}': d for i, d in enumerate(driver_chain(strategy))})
    out.update(extra or {})
    return {k: v for k, v in out.items() if v is not None}


def progress_stamp(session) -> tuple:
    """What a checkpoint is worth saving for: the runs and the model calls made so far."""
    u = session.ledger.used
    return (u.get('screens', 0), u.get('trials', 0), len(session.screens), len(session.llm_calls))


def capture(session, engine, strategy, *, why: str = 'observe', stop: str | None = None, sequence: int = 0,
            live: dict | None = None) -> tuple[dict, bytes]:
    """(header, body): the header is plain data for the summary file and the resume's checks, the body the state pickled."""
    state = {'session': session.state_dict(), 'engine': engine.state_dict(), 'strategy': strategy.state_dict(),
             'rng': S.rng_state()}
    objs = live_objects(session, engine, strategy, live)
    try:
        body = S.dumps(state, objs)
    except Exception as exc:                             # noqa: BLE001 - name the attribute that does not pickle
        bad = []
        for key in ('session', 'engine', 'strategy'):
            bad += S.unsavable(state[key], objs, f'{key}.') if isinstance(state[key], dict) else []
        raise S.CheckpointError(f'the state cannot be saved ({type(exc).__name__}: {str(exc)[:160]}): '
                                + ('; '.join(bad[:6]) if bad else 'no single attribute found')) from exc
    used = session.ledger.snapshot()
    inc = getattr(engine, '_inc', None)
    cand = session.candidates.get(inc) if inc else None
    header = {'saved_utc': S.utc(), 'why': why, 'stop': stop, 'sequence': sequence, 'run_id': session.run_id,
              'arm': session.arm, 'strategy': getattr(strategy, 'name', None),
              'strategy_class': f'{type(strategy).__module__}.{type(strategy).__qualname__}',
              'screens_done': used.get('screens'), 'trials': used.get('trials', 0),
              'budget_used': used, 'budget': session.budget.as_dict(), 'incumbent': inc,
              'incumbent_step_time': ((cand.verdict or {}).get('step_time') if cand is not None else None),
              'candidates': len(session.candidates), 'events': len(session.events), 'llm_calls': len(session.llm_calls),
              'package_hash': S.package_hash(), 'python': sys.version.split()[0], 'bytes': len(body)}
    return header, body


class Writer:
    """The Engine's `checkpoint` hook (Engine._checkpoint): `directory` is <run>/search/state; `flow()` returns the flow's
    own state to keep with the search's (project/resume.py); `live` names further live objects (the flow's). A chosen
    proposal ('proposed') or an observed run that spent nothing since the last checkpoint (no model call, a refused or
    repeated proposal) is not saved - the restored state proposes the same again; the end of the search always is."""

    def __init__(self, directory: str | Path, flow=None, live: dict | None = None):
        self.directory, self.flow, self.live = Path(directory), flow, live or {}
        self.sequence, self.stamp, self.errors = 0, None, []

    def __call__(self, engine, strategy, why: str, stop: str | None = None) -> None:
        session = engine.s
        stamp = progress_stamp(session)
        if why in ('observe', 'proposed') and stamp == self.stamp:
            return
        header, body = capture(session, engine, strategy, why=why, stop=stop, sequence=self.sequence + 1, live=self.live)
        S.write(self.directory, header, body, self.flow() if callable(self.flow) else self.flow)
        self.sequence, self.stamp = self.sequence + 1, stamp


def restore(state: dict, session, engine, strategy) -> None:
    """The saved state onto the objects of the resuming run (built as a fresh run builds them, so every live handle and
    every setting is theirs): session, engine, strategy and their parts, the random generators; the saved search's stop flag
    run of failures does not carry over when it is what ended the saved search, and the strategy is bound to the engine as its
    first proposal binds it."""
    session.load_state_dict(state['session'])
    engine.load_state_dict(state['engine'])
    strategy.load_state_dict(state['strategy'])
    S.set_rng_state(state.get('rng'))
    if str(session.stop_flag or '').startswith('same_failure'):      # (a run of failures that ended the saved search)
        session.same_failure = (None, 0)
    session.stop_flag = None
    if hasattr(strategy, 'engine'):
        strategy.engine = engine
    if 'ex_eng' in vars(strategy):
        strategy.ex_eng = engine
