"""Safe control boundaries, saved chat accounting, and continuation between probe/short/full trials."""
import copy

from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.agent.search import USER_STOP
from infrarsi.project.live import LiveConsole
from infrarsi.search.engine import Proposal, Strategy
from test_search import TOGGLE, setup


class One(Strategy):
    uses_engine_stop = False
    def propose(self, engine, k):
        return [Proposal(change={'fused_adam': True}, fidelity=('probe', 'short', 'full'), keep_probe=True)]
    def observe(self, engine, p, out):
        self.seen = out.status


def test_stop_between_probe_and_short_keeps_pending_state_and_resumes():
    _, s, eng = setup(TOGGLE, tokens=100000)
    console = LiveConsole(enabled=False, log=lambda _: None)
    s.control = console.poll
    real_run, starts = s.runner.run, []
    def run(*a, **kw):
        result = real_run(*a, **kw)
        starts.append(a[3])
        if len(starts) == 1:
            console.submit('/stop')
        return result
    s.runner.run = run
    saved = []
    eng.checkpoint = lambda e, st, why, stop: saved.append((why, copy.deepcopy(e._evaluation), copy.deepcopy(s.ledger.used)))
    strategy = One()
    trace = eng.run(strategy)
    assert trace.stop_reason == USER_STOP and starts == ['probe']
    assert eng._pending is not None and not hasattr(strategy, 'seen')
    assert saved[-1][1]['index'] == 1 and saved[-1][2]['trials'] == 1
    # Resume the exact pending plan without rerunning the probe or discarding its costs.
    console.stopping.clear()
    s.stop_flag = None
    eng.run(strategy, max_iters=1)
    assert starts == ['probe', 'short', 'screen']
    assert strategy.seen == 'screened' and s.ledger.used['trials'] == 2 and s.ledger.used['screens'] == 1


def test_chat_then_stop_checkpoints_consumption_and_notes_before_first_proposal():
    _, s, eng = setup(TOGGLE, tokens=100000)
    drv = ScriptedBackend([{'reply': 'Ready'}], tokens_per_call=100)
    console = LiveConsole(drv, enabled=False, log=lambda _: None)
    console.submit('/note Prefer fewer workers')
    console.submit('What is happening?')
    console.submit('/stop')
    s.control = console.poll
    saved = []
    eng.checkpoint = lambda e, st, why, stop: saved.append((s.ledger.used['llm_tokens'], list(s.live_notes), list(s.live_chat_history)))
    trace = eng.run(One())
    assert trace.stop_reason == USER_STOP
    assert saved[-1][0] == 100 and saved[-1][1] == ['Prefer fewer workers'] and saved[-1][2][0]['assistant'] == 'Ready'


def test_stop_during_proposal_does_not_start_a_trial():
    _, s, eng = setup(TOGGLE, tokens=100000)
    console = LiveConsole(enabled=False, log=lambda _: None)
    s.control = console.poll
    class StopInPropose(One):
        def propose(self, engine, k):
            console.submit('/stop')
            return super().propose(engine, k)
    trace = eng.run(StopInPropose())
    assert trace.stop_reason == USER_STOP and not s.screens


def test_interrupt_while_paused_preserves_elapsed_wall_budget():
    _, s, eng = setup(TOGGLE, tokens=100000)
    console = LiveConsole(enabled=False, log=lambda _: None)
    s.control = console.poll
    s.on_interrupt = 'verify'
    console.submit('/pause')
    saved = []
    eng.checkpoint = lambda *args: saved.append(s.ledger.wall())
    def wait(_):
        s.clock.advance(3600)
        raise KeyboardInterrupt
    console.closed.wait = wait
    eng.run(One())
    assert saved[-1] == s.ledger.wall() and saved[-1] - saved[0] == 3600


def test_stop_during_confirmation_finishes_pair_before_checkpoint():
    _, s, eng = setup(TOGGLE, tokens=100000, run_noise=.04, early_abort=False)
    console = LiveConsole(enabled=False, log=lambda _: None)
    s.control = console.poll
    original, starts = s.runner.run, []
    def run(*a, **kw):
        result = original(*a, **kw)
        starts.append((a[3], kw['candidate_id']))
        if len(starts) == 2:  # incumbent rescore: half the confirmation pair is complete
            console.submit('/stop')
        return result
    s.runner.run = run
    class Confirm(One):
        def propose(self, engine, k):
            return [Proposal(change={'flash_attn': True}, confirm=True)]
    trace = eng.run(Confirm())
    assert trace.stop_reason == USER_STOP
    assert [phase for phase, _ in starts] == ['screen', 'rescore', 'rescore']
    assert eng.confirmations[-1]['moved'] and eng._pending is None and eng._inc == 'c1'
