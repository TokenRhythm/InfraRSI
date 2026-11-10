"""Fixes from the 2026-10-23 Q&A on the search (reports/infrarsi_qa_20261023):
A1 the hypothesis model sees the files it may patch and may answer with search/replace edits (before, the prompt named
the mutable files but never showed them, so no real run ever proposed a patch);
A2 greedy shows the model what it already tried (H2b arm B repeated failed moves in 10 of 24 runs);
A4 a transport failure stops the search without a verdict on the candidate (H2b arm C: 16 candidates marked invalid in
0.2 s)."""
import json
import math

from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.contracts import InfrastructureError
from infrarsi.ext.common.session_hooks import SessionHooks
from infrarsi.search.proposers import LLMProposer, apply_edits, code_view
from infrarsi.project.executor import transport_failure
from infrarsi.search.engine import Outcome, Proposal
from infrarsi.search.proposers import _tokens
from infrarsi.search.strategies import HLS, Greedy, TPEKnobs

from test_search import TOGGLE, setup


def test_proposer_shows_the_code_and_applies_edits():
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    hooks = SessionHooks(s.task, base_files={'patches.txt': 'a\nb\n'}, read_only_files={'train.py': 'print(1)\n'})
    reply = {'hypotheses': [
        {'mechanism': 'kernel fix', 'change': {}, 'edits': [{'file': 'patches.txt', 'old': 'b\n', 'new': 'c\n'}],
         'predicted_gain_pct': 5},
        {'mechanism': 'stale edit', 'change': {}, 'edits': [{'file': 'patches.txt', 'old': 'zzz', 'new': 'c'}],
         'predicted_gain_pct': 5},
        {'mechanism': 'fused', 'change': {'fused_adam': True}, 'predicted_gain_pct': 3}]}
    drv = ScriptedBackend(lambda role, system, prompt: reply)
    prop = LLMProposer(drv, hooks)
    hyps = prop.hypotheses(eng, 'c0')
    calls = [c for c in drv.calls if c['role'] == 'hypotheses']     # the code review is the first call
    payload = json.loads(calls[0]['prompt'])
    assert payload['code']['mutable'] == {'patches.txt': 'a\nb\n'} and payload['code']['readable'] == {'train.py': 'print(1)\n'}
    assert 'CODE: `code` holds' in calls[0]['system']
    assert [h['mechanism'] for h in hyps] == ['kernel fix', 'fused'] and hyps[0]['patch'] == {'patches.txt': 'a\nc\n'}
    assert hyps[1]['patch'] is None and prop.describe()['unusable_edits'] == 1


def test_menu_mode_shows_no_code_and_takes_no_patch():
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    hooks = SessionHooks(s.task, base_files={'patches.txt': 'a\n'})
    reply = {'hypotheses': [{'mechanism': 'm', 'change': {'fused_adam': True}, 'patch': {'patches.txt': 'b\n'},
                             'predicted_gain_pct': 3}]}
    drv = ScriptedBackend(lambda role, system, prompt: reply)
    hyps = LLMProposer(drv, hooks, menu=True).hypotheses(eng, 'c0')
    assert 'code' not in json.loads(drv.calls[0]['prompt']) and 'CODE:' not in drv.calls[0]['system']
    assert hyps and hyps[0]['patch'] is None


def test_code_view_budget_and_edit_rules():
    class Hooks:
        read_only_files = {'r.py': 'x\n' * 10}

        def candidate_files(self, overlay):
            return {'big.py': '\n'.join(f'line {i}' for i in range(2000)), **(overlay or {})}

    v = code_view(Hooks(), None, budget_tokens=1000)
    assert v['excerpted'] == ['big.py'] and 'lines not shown' in v['mutable']['big.py']
    assert v['not_shown'] == ['r.py (10 lines)'] and 'readable' not in v
    files = {'k.py': 'a = 1\na = 1\nb = 2\n'}
    assert apply_edits(files, [{'file': 'k.py', 'old': 'a = 1', 'new': 'a = 3'}]) is None        # not unique
    assert apply_edits(files, [{'file': 'nope.py', 'old': 'a', 'new': 'b'}]) is None             # unknown file
    assert apply_edits(files, [{'file': 'k.py', 'old': 'b = 2', 'new': 'b = 5'}]) == {'k.py': 'a = 1\na = 1\nb = 5\n'}
    assert apply_edits(files, [{'file': 'k.py', 'old': 'c', 'new': 'd'}], patch={'k.py': 'c\n'}) == {'k.py': 'd\n'}


def test_greedy_shows_the_model_what_it_tried():
    sim, s, eng = setup(TOGGLE, budget=4, tokens=10 ** 6)
    reply = {'hypotheses': [{'mechanism': 'fused', 'change': {'fused_adam': True}, 'predicted_gain_pct': 30},
                            {'mechanism': 'flash', 'change': {'flash_attn': True}, 'predicted_gain_pct': 20}]}
    drv = ScriptedBackend(lambda role, system, prompt: reply)
    eng.run(Greedy(LLMProposer(drv, SessionHooks(s.task, base_files={'patches.txt': ''}))))
    calls = [c for c in drv.calls if c['role'] == 'hypotheses']     # the code review is the first call
    assert len(calls) >= 2 and json.loads(calls[0]['prompt'])['ledger'] == []
    seen = json.loads(calls[1]['prompt'])['ledger']
    assert seen[0]['mechanism'] == 'fused' and seen[0]['predicted_gain_pct'] == round(100 * math.log1p(0.3), 2)
    assert seen[0]['measured_gain_pct'] is not None and seen[0]['status'] in ('adopted', 'not adopted')


def test_transport_failure_stops_the_search_without_a_verdict(monkeypatch):
    sim, s, eng = setup(TOGGLE, budget=4, tokens=10 ** 6)

    def unreachable(*a, **kw):
        raise InfrastructureError('copy to node0 failed: Connection reset by peer')

    monkeypatch.setattr(s.runner, 'run', unreachable)
    reply = {'hypotheses': [{'mechanism': 'fused', 'change': {'fused_adam': True}, 'predicted_gain_pct': 30,
                             'p_feasible': 0.95}]}
    hls = HLS(LLMProposer(ScriptedBackend(lambda role, system, prompt: reply), SessionHooks(s.task, base_files={'patches.txt': ''})))
    tr = eng.run(hls).to_dict()
    assert tr['stop_reason'].startswith('infrastructure: copy to node0 failed')
    entry = next(h for h in hls.ledger.values() if h['mechanism'] == 'fused')
    assert entry['status'] == 'open' and entry.get('measured') is None      # no verdict, nothing for the cross-run ledger
    assert s.ledger.used['screens'] == 0 and not hls.pairs
    assert {c.status for cid, c in s.candidates.items() if cid != 'c0'} == {'proposed'}


def test_excerpt_of_a_huge_single_line_file_stays_within_the_budget():
    class Hooks:
        read_only_files = {}

        def candidate_files(self, overlay):
            return {'table.py': 'TABLE = ' + repr(list(range(30000))) + chr(10) + 'X = 1' + chr(10)}

    v = code_view(Hooks(), None, budget_tokens=2000)
    assert v['excerpted'] == ['table.py'] and _tokens(v['mutable']['table.py']) <= 2000


def test_only_ssh_own_words_on_exit_255_mean_a_transport_failure():
    ssh_said = [(0.0, 'ssh: connect to host node1 port 22: Connection refused')]
    job_said = [(0.0, '1729650000.123456' + chr(9) + 'RuntimeError: socket connection refused')]
    dropped = [(0.0, '1729650000.123456' + chr(9) + 'iteration 3 | loss 2.1'), (1.0, 'Connection to node0 closed by remote host.')]
    assert transport_failure(255, ssh_said) and transport_failure(255, dropped) and transport_failure(255, []) == 'no output'
    assert transport_failure(255, job_said) is None and transport_failure(1, ssh_said) is None


def test_engine_tpe_records_an_infra_outcome_as_a_failed_trial():
    sim, s, eng = setup(TOGGLE)
    tpe = TPEKnobs()
    p = tpe.propose(eng, 1)[0]
    tpe.observe(eng, p, Outcome(None, 'infra', False, None, None, 'full', 0.0, ['infrastructure failure'], []))
    states = [t.state.name for t in tpe.study.trials]
    assert 'FAIL' in states and all(t.value is None for t in tpe.study.trials if t.state.name == 'FAIL')


def test_a_killed_node_without_output_is_not_a_transport_failure_but_ssh_words_still_are():
    said = [(0.0, 'ssh: connect to host worker1 port 22: Connection refused')]
    assert transport_failure(255, [], silent_counts=False) is None
    assert transport_failure(255, said, silent_counts=False)


def test_trials_and_rescores_keep_incomplete_retry_accounting(monkeypatch):
    sim, s, eng = setup(TOGGLE, budget=4)
    cid = s.propose_config(changes={'fused_adam': True}, base='c0', origin='t')['candidate_id']
    monkeypatch.setattr(s.runner, 'run', lambda *a, **kw: {'valid': True, 'cost': {'gpu_seconds': 36.0,
                                                                                   'accounting_complete': False}})
    s.trial(cid, steps=2, fidelity='probe')
    assert s.ledger.accounting_complete is False

