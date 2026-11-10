"""HLS tries a knob again without the incumbent's last change when it failed a numerics check on that incumbent
(FABLE_DF8_DECISION B, F-F-3). F1-S1 (sft_loop, 10 screens): bf16 came last as a zero-prior neighbour of the compile
incumbent, bf16 + compile broke the loss at the probe (3.07 vs 1.00), bf16 was never tried elsewhere, and random search
promoted bf16 without compile at x3.509 against HLS's x1.52. Now the failure adds one entry on the incumbent - the knob
with the incumbent's last change reverted - probe first, ahead of the untried zero-prior neighbours (feasibility 0.7
against 0.6); at most `retry` per incumbent, never for a retry's own failure, a crash, a patch or a patched incumbent."""
import dataclasses
from types import SimpleNamespace

from infrarsi.search.engine import Outcome, Proposal
from infrarsi.search.strategies import HLS

from test_composition import FakeProposer, hyp
from test_search import TOGGLE, setup


class Lineage:
    """A fake engine whose incumbent c1 carries one accepted change over its parent c0."""

    def __init__(self, inc_change=None, inc_overlay=None, c0_config=None):
        c0 = SimpleNamespace(config=c0_config or {'a': 0, 'bf16': 0, 'compile': 0}, overlay={}, parent=None)
        c1 = SimpleNamespace(config={**c0.config, **(inc_change if inc_change is not None else {'compile': 1})},
                             overlay=dict(inc_overlay or {}), parent='c0')
        self.s = SimpleNamespace(candidates={'c0': c0, 'c1': c1}, hooks=None)
        self._inc = 'c1'
        self.lines = []
        self.log = self.lines.append

    def incumbent(self):
        return self._inc

    def sigma_pair(self):
        return 0.02

    def sigma_run(self):
        return 0.015


def infeasible(cid, checks=('loss_within_tolerance',)):
    return Outcome(cid, 'infeasible', False, None, None, 'probe', 0.01, ['loss_within_tolerance: ...'], list(checks))


def retries_in(props):
    return [p for p in props if 'without' in p.mechanism]


def test_a_numerics_failure_on_a_moved_incumbent_is_retried_without_its_last_change():
    eng, hls = Lineage(), HLS(FakeProposer([hyp('bf16')]), explore=False)
    p = hls.propose(eng, 16)[0]
    hls.observe(eng, p, infeasible('c2'))
    assert hls.ledger[p.hypothesis]['status'] == 'invalid'
    again = hls.propose(eng, 16)
    assert len(again) == 1
    q = again[0]
    assert q.change == {'compile': 0, 'bf16': 1} and q.base == 'c1' and q.fidelity == ('probe', 'full')
    assert q.mechanism == 'bf16 without compile=1' and q.mu == 0.0 and q.p_feasible == 0.7
    h = hls.ledger[q.hypothesis]
    assert h['retry_of'] == p.hypothesis and h['explore'] and hls.retries == {'c1': ['bf16 without compile=1']}
    assert hls.describe()['retries'] == hls.retries
    assert eng.lines == ['[search] hls: bf16 failed loss_within_tolerance on c1: it will be tried without compile=1 (1 of 2 on c1)']
    assert not [v for v in hls.ledger_view() if 'without' in v['mechanism']]      # open: hidden like a neighbour
    hls.observe(eng, q, Outcome('c3', 'screened', True, 1.0, 0.9, 'full', 0.1, [], []))
    assert h['status'] == 'confirmed' and h['measured'] == 0.9 and hls.pairs == []  # exploration: not calibration
    shown = [v for v in hls.ledger_view() if 'without' in v['mechanism']]
    assert shown and shown[0]['change'] == {'compile': 0, 'bf16': 1} and shown[0]['status'] == 'confirmed'
    assert hls.propose(eng, 16) == []


def _failed_bf16_on_c1():
    eng, hls = Lineage(), HLS(FakeProposer([hyp('bf16')]), explore=False)
    p = hls.propose(eng, 16)[0]
    hls.observe(eng, p, infeasible('c2'))
    return eng, hls


def test_a_retry_the_incumbent_moved_past_goes_along():
    # F1c-S1-gpt56terra (DF11): bf16 failed on the compile incumbent and its retry without compile was due, but a small
    # gain (fused AdamW) moved the incumbent first; the retry was lost and bf16 on the new incumbent reverted fused
    # AdamW instead (a duplicate of the failed run): x1.525, where DeepSeek V4.1's run, in order, reached x3.503
    eng, hls = _failed_bf16_on_c1()
    assert retries_in(hls.propose(eng, 16))[0].change == {'compile': 0, 'bf16': 1}
    eng.s.candidates['c3'] = SimpleNamespace(config={**eng.s.candidates['c1'].config, 'a': 1}, overlay={}, parent='c1')
    eng._inc = 'c3'                                                      # fused AdamW's analogue moved the incumbent
    moved = retries_in(hls.propose(eng, 16))
    assert len(moved) == 1 and moved[0].base == 'c3' and moved[0].change == {'compile': 0, 'bf16': 1}
    assert moved[0].fidelity == ('probe', 'full') and moved[0].p_feasible == 0.7
    assert hls.ledger[moved[0].hypothesis]['rebased'] and hls.retries['c3'] == ['bf16 without compile=1']
    hls.observe(eng, moved[0], Outcome('c4', 'screened', True, 1.0, 0.8, 'full', 0.1, [], []))
    assert hls.ledger[moved[0].hypothesis]['status'] == 'confirmed'


def test_a_retry_already_in_the_new_incumbent_stays_behind():
    eng, hls = _failed_bf16_on_c1()
    eng.s.candidates['c3'] = SimpleNamespace(config={**eng.s.candidates['c1'].config, 'bf16': 1, 'compile': 0},
                                             overlay={}, parent='c1')
    eng._inc = 'c3'
    assert not retries_in(hls.propose(eng, 16)) and 'c3' not in hls.retries
    # DF15_r1: an env change as the session applies it - a subset already set, a removal already done
    for change, env in (({'bf16': 1, 'env': {'NCCL_ALGO': 'Ring'}}, {'NCCL_ALGO': 'Ring', 'NCCL_DEBUG': 'WARN'}),
                        ({'bf16': 1, 'env': {'NCCL_ALGO': None}}, {'NCCL_DEBUG': 'WARN'})):
        eng, hls = _failed_bf16_on_c1()
        h = next(x for x in hls.ledger.values() if x.get('retry_of'))
        h['change'] = change
        eng.s.candidates['c3'] = SimpleNamespace(config={**eng.s.candidates['c1'].config, 'bf16': 1, 'env': env},
                                                 overlay={}, parent='c1')
        eng._inc = 'c3'
        assert not retries_in(hls.propose(eng, 16)) and 'c3' not in hls.retries


def test_a_retry_is_not_retried_and_the_cap_per_incumbent_holds():
    eng = Lineage()
    hls = HLS(FakeProposer([hyp('bf16'), hyp('fp8'), hyp('x')]), explore=False, retry=2)
    for name, cid in (('bf16', 'c2'), ('fp8', 'c3')):
        hls.observe(eng, next(p for p in hls.propose(eng, 16) if p.mechanism == name), infeasible(cid))
    retries = retries_in(hls.propose(eng, 16))
    assert sorted(p.mechanism for p in retries) == ['bf16 without compile=1', 'fp8 without compile=1']
    hls.observe(eng, retries[0], infeasible('c4'))                     # a retry's own failure: nothing more
    assert [p.mechanism for p in retries_in(hls.propose(eng, 16))] == [retries[1].mechanism]
    hls.observe(eng, next(p for p in hls.propose(eng, 16) if p.mechanism == 'x'), infeasible('c5'))
    assert len(hls.retries['c1']) == 2 and not any(p.mechanism.startswith('x ') for p in hls.propose(eng, 16))
    none = HLS(FakeProposer([hyp('bf16')]), explore=False, retry=0)     # retry=0: the old behaviour
    none.observe(eng, none.propose(eng, 16)[0], infeasible('c2'))
    assert none.propose(eng, 16) == [] and none.retries == {}


def test_no_retry_for_a_crash_a_patch_the_start_a_patched_incumbent_or_the_knob_itself():
    eng = Lineage()
    hls = HLS(FakeProposer([hyp('bf16'), dict(hyp('edit'), change={}, patch={'f.py': 'x'})]), explore=False)
    props = hls.propose(eng, 16)
    hls.observe(eng, next(p for p in props if p.mechanism == 'bf16'), infeasible('c2', ['completes', 'no_errors']))
    hls.observe(eng, next(p for p in props if p.mechanism == 'edit'), infeasible('c3'))
    assert hls.retries == {} and not retries_in(hls.propose(eng, 16))   # crashed: every base; a patch: nothing to revert
    start = Lineage()
    start._inc = 'c0'                                                   # the start has no parent
    hls = HLS(FakeProposer([hyp('bf16')]), explore=False)
    hls.observe(start, hls.propose(start, 16)[0], infeasible('c2'))
    assert hls.retries == {}
    patched = Lineage(inc_change={}, inc_overlay={'f.py': 'patched'})   # the last change is a patch
    hls = HLS(FakeProposer([hyp('bf16')]), explore=False)
    hls.observe(patched, hls.propose(patched, 16)[0], infeasible('c2'))
    assert hls.retries == {}
    own = Lineage(inc_change={'bf16': 2})                               # the knob replaces the last change itself
    hls = HLS(FakeProposer([hyp('bf16')]), explore=False)
    hls.observe(own, hls.propose(own, 16)[0], infeasible('c2'))
    assert hls.retries == {}
    assert hyp('bf16')['change'] == {'bf16': 1}


def test_the_revert_restores_every_value_of_the_last_change_including_env_names():
    eng = Lineage(c0_config={'a': 0, 'bf16': 0, 'env': {'A': '1', 'B': '2'}},
                  inc_change={'a': 3, 'env': {'A': '9', 'C': '3'}})
    hls = HLS(FakeProposer([hyp('bf16')]), explore=False)
    hls.observe(eng, hls.propose(eng, 16)[0], infeasible('c2'))
    q = retries_in(hls.propose(eng, 16))[0]
    assert q.change == {'a': 0, 'env': {'A': '1', 'B': '2', 'C': None}, 'bf16': 1}


def test_an_env_knob_is_reverted_per_name_and_merged_with_the_failed_knobs_own_names():
    # DF8_r1 #2: the incumbent's last change set env NCCL_ALGO; the failed knob sets env NCCL_PROTO
    proto = {'mechanism': 'nccl proto', 'change': {'env': {'NCCL_PROTO': 'Simple'}}, 'patch': None, 'mu': 0.03,
             'sigma': 0.02, 'p_feasible': 0.9}
    eng = Lineage(c0_config={'a': 0, 'env': {'NCCL_ALGO': 'Ring'}}, inc_change={'env': {'NCCL_ALGO': 'Tree'}})
    hls = HLS(FakeProposer([proto]), explore=False)
    hls.observe(eng, hls.propose(eng, 16)[0], infeasible('c2'))
    q = retries_in(hls.propose(eng, 16))[0]
    assert q.change == {'env': {'NCCL_ALGO': 'Ring', 'NCCL_PROTO': 'Simple'}}
    algo = dict(proto, mechanism='nccl algo', change={'env': {'NCCL_ALGO': 'NVLS', 'NCCL_PROTO': 'LL'}})
    eng = Lineage(c0_config={'a': 0, 'env': {'NCCL_ALGO': 'Ring', 'B': '1'}},
                  inc_change={'a': 1, 'env': {'NCCL_ALGO': 'Tree', 'B': '2'}})
    hls = HLS(FakeProposer([algo]), explore=False)
    hls.observe(eng, hls.propose(eng, 16)[0], infeasible('c2'))
    q = retries_in(hls.propose(eng, 16))[0]                          # the knob's own NCCL_ALGO stays; B and a revert
    assert q.change == {'a': 0, 'env': {'B': '1', 'NCCL_ALGO': 'NVLS', 'NCCL_PROTO': 'LL'}}


def test_toy_the_big_knob_lost_to_a_small_accepted_change_is_recovered_first():
    """The F1-S1 shape on the toy's T4 trap (fused_ce + compile moves the loss 6 %): the incumbent carries fused_ce
    (+2 %, forced: within the noise); compile (+8 %) fails the probe on it; the retry tests compile without fused_ce,
    is taken before the other zero-prior neighbours, and takes the lead."""
    lines = []
    sim, s, eng = setup(dataclasses.replace(TOGGLE, antagonism=0.0), log=lines.append)
    small = eng.evaluate(Proposal(change={'fused_ce': True}, mu=0.02))
    assert small.status == 'screened' and eng.incumbent() == 'c0'
    eng._inc = small.cid
    hls = HLS(FakeProposer([]), explore=True)                           # the model has nothing: the neighbourhood
    p = next(p for p in hls.propose(eng, 16) if p.change == {'compile': True})
    out = eng.evaluate(p)
    assert (out.status, out.fidelity, out.failed_checks) == ('infeasible', 'probe', ['loss_within_tolerance'])
    hls.observe(eng, p, out)
    props = hls.propose(eng, 16)
    q = retries_in(props)[0]
    assert q.change == {'fused_ce': False, 'compile': True} and q.mechanism == 'explore:compile=True without fused_ce=True'
    assert len(props) > 5 and eng.decide(props)[0] is q                 # first among the zero-prior proposals
    assert lines[-1] == (f'[search] hls: explore:compile=True failed loss_within_tolerance on {small.cid}: it will be tried '
                         f'without fused_ce=True (1 of 2 on {small.cid})')
    out = eng.evaluate(q)
    assert out.status == 'screened' and out.gain > 0 and eng.incumbent() == out.cid
    c = s.candidates[out.cid]
    assert c.parent == small.cid and c.config['compile'] and not c.config['fused_ce']
    hls.observe(eng, q, out)
    assert hls.ledger[q.hypothesis]['status'] == 'confirmed' and hls.pairs == []
    assert hls.describe()['retries'] == {small.cid: ['explore:compile=True without fused_ce=True']}
