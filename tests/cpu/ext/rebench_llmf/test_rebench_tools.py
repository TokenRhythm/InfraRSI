"""A-NAT basic-agent score() EnvTool + locked_rescore glue: published fields, selection = best valid score with its
snapshot identity, experiment charge, and the B-05 locked re-score of the selected snapshot (never the final tree)."""
from infrarsi.ext.common.basic_agent import EnvTool, best_screen
from infrarsi.ext.rebench_llmf import locked_rescore, make_tools

import rebench_helpers as H


class SeqRunner:
    """A runner stand-in returning a scripted sequence of receipts (matches EnvRunner.run's signature)."""

    def __init__(self, task, receipts):
        self.task = task
        self._receipts = list(receipts)
        self.calls = []

    def run(self, task, config, overlay, purpose, run_id, **kw):
        self.calls.append((purpose, kw))
        return self._receipts.pop(0)


class FakeAgent:
    """Minimal BasicAgent stand-in: records the experiments the score tool charges (SearchTrace-like rows)."""

    def __init__(self):
        self.recorded, self.screens = [], []

    def record_experiment(self, receipt, verdict, *, candidate_id, kind='experiment', extra=None):
        v = verdict.to_dict()
        self.recorded.append({'candidate_id': candidate_id, 'kind': kind, 'valid': receipt.get('valid')})
        self.screens.append({'index': len(self.screens), 'candidate_id': candidate_id, 'valid': v['valid'],
                             'step_time': v.get('step_time')})


def _rec(t, *, tree, purpose='screen', **kw):
    """A signed receipt of a reuse-home scoring whose solution tree has digest `tree`."""
    r, _ = H.runner(H.node_result(t, tree_digest=tree, **kw), arm='A-NAT', g4=H.g4_ok())
    if purpose == 'score':
        return r.run(t, None, None, 'score', 'run', run_g4=True, restore_snapshot=f'/s/{tree}.tar', expected_digest=tree)
    return r.run(t, None, None, purpose, 'run', reuse_home=True)


def _score_tool(tools):
    tool = next(d for d in tools if d.name == 'score')
    assert isinstance(tool, EnvTool) and tool.counts_as_experiment
    return tool.fn


def test_score_returns_published_fields_and_keeps_snapshot_identity():
    t = H.task('A-NAT')
    runner = SeqRunner(t, [_rec(t, tree='1' * 64, execution_time=216.1), _rec(t, tree='2' * 64, execution_time=120.0)])
    tools, state = make_tools(runner, 'A-NAT-run')
    score = _score_tool(tools)
    agent = FakeAgent()
    out1 = score({}, agent)
    assert set(out1) >= {'execution_time', 'norm_diff', 'score', 'valid', 'n_scored'}
    assert out1['execution_time'] == 216.1 and out1['n_scored'] == 1
    assert score({}, agent)['execution_time'] == 120.0
    assert len(state.scores) == 2 and state.best['execution_time'] == 120.0
    assert state.best['tree_digest'] == '2' * 64 and state.best['snapshot_path'] == f'/run/snapshots/{"2" * 64}.tar'
    assert all(c == ('screen', {'reuse_home': True, 'candidate_id': f'score0{i + 1}'}) for i, c in enumerate(runner.calls))
    assert len(agent.recorded) == 2 and agent.recorded[0]['kind'] == 'A-NAT'    # charged as experiments


def test_selection_matches_the_trace_rule_guard_rejected_scores_are_not_selected():
    """state.best follows the same rule as the trace's `selected` (best_screen: valid receipt, lowest time,
    earliest on ties); a score.py-valid scoring that a guard rejected (here G8) is not selectable."""
    t = H.task('A-NAT')
    runner = SeqRunner(t, [_rec(t, tree='1' * 64, execution_time=200.0),
                           _rec(t, tree='2' * 64, execution_time=90.0, g8=H.g8_escape()),
                           _rec(t, tree='3' * 64, execution_time=150.0),
                           _rec(t, tree='4' * 64, execution_time=150.0)])
    tools, state = make_tools(runner, 'r')
    agent = FakeAgent()
    for _ in range(4):
        _score_tool(tools)({}, agent)
    assert state.scores[1]['judge_valid'] is True and state.scores[1]['valid'] is False
    assert state.best['tree_digest'] == '3' * 64                                  # earliest of the 150 s ties
    assert best_screen({'screens': agent.screens})['candidate_id'] == state.best['candidate_id'] == 'score03'


def test_invalid_score_not_promoted():
    t = H.task('A-NAT')
    runner = SeqRunner(t, [_rec(t, tree='1' * 64, valid_judge=False, trained=False),
                           _rec(t, tree='2' * 64, execution_time=200.0)])
    tools, state = make_tools(runner, 'r')
    score = _score_tool(tools)
    bad = score({}, FakeAgent())
    assert bad['valid'] is False and 'message' in bad
    score({}, FakeAgent())
    assert state.best['execution_time'] == 200.0 and state.best['tree_digest'] == '2' * 64


def test_score_works_without_agent_record():
    t = H.task('A-NAT')
    runner = SeqRunner(t, [_rec(t, tree='1' * 64, execution_time=150.0)])
    tools, state = make_tools(runner, 'r')
    out = _score_tool(tools)({}, None)              # no agent -> still scores and records to state
    assert out['execution_time'] == 150.0 and len(state.scores) == 1


def test_locked_rescore_restores_the_selected_snapshot_never_the_final_tree():
    t = H.task('A-NAT')
    best, later = '2' * 64, '3' * 64
    runner = SeqRunner(t, [_rec(t, tree='1' * 64, execution_time=200.0), _rec(t, tree=best, execution_time=120.0),
                           _rec(t, tree=later, valid_judge=False, trained=False),   # the agent's final, worse edit
                           _rec(t, tree=best, purpose='score', execution_time=121.0)])
    tools, state = make_tools(runner, 'r')
    for _ in range(3):
        _score_tool(tools)({}, FakeAgent())
    out = locked_rescore(runner, 'r', state)
    purpose, kw = runner.calls[-1]
    assert purpose == 'score' and kw['run_g4'] is True and 'reuse_home' not in kw
    assert kw['restore_snapshot'] == f'/run/snapshots/{best}.tar' and kw['expected_digest'] == best
    assert out['locked']['valid'] is True and out['locked']['digest_matches_selection'] is True
    assert out['locked']['execution_time'] == 121.0 and out['selected']['tree_digest'] == best
    assert 'final' not in str(out)                                         # one locked line, no final-tree score


def test_locked_rescore_with_a_different_digest_is_not_valid():
    t = H.task('A-NAT')
    runner = SeqRunner(t, [_rec(t, tree='2' * 64, execution_time=120.0), _rec(t, tree='9' * 64, purpose='score')])
    tools, state = make_tools(runner, 'r')
    _score_tool(tools)({}, FakeAgent())
    out = locked_rescore(runner, 'r', state)
    assert out['locked']['digest_matches_selection'] is False and out['locked']['valid'] is False


def test_no_valid_score_means_no_locked_run():
    t = H.task('A-NAT')
    runner = SeqRunner(t, [_rec(t, tree='1' * 64, valid_judge=False, trained=False)])
    tools, state = make_tools(runner, 'r')
    _score_tool(tools)({}, FakeAgent())
    out = locked_rescore(runner, 'r', state)
    assert out['locked'] is None and [c[0] for c in runner.calls] == ['screen']


def test_score_tool_is_admitted_by_the_runner_worst_case():
    """H-05: the experiment tool asks the runner whether one more scoring's worst case fits (not the mean cost)."""
    seen = []

    class AdmitRunner(SeqRunner):
        def admits(self, ledger, budget, world=None):
            seen.append((ledger, budget))
            return 'wall_seconds'
    t = H.task('A-NAT')
    tools, _ = make_tools(AdmitRunner(t, []), 'r')
    tool = next(d for d in tools if d.name == 'score')
    agent = type('A', (), {'ledger': 'L', 'budget': 'B'})()
    assert tool.admit(agent) == 'wall_seconds' and seen == [('L', 'B')]
    assert next(d for d in make_tools(SeqRunner(t, []), 'r')[0]).admit is None


def test_score_runs_under_the_admitted_cap():
    """H-05 (R2): after admission the scoring is run with wall_cap = the runner's worst case (the node job is bounded
    to what the admission reserved)."""
    class CapRunner(SeqRunner):
        def worst_case(self):
            return 1270.0
    t = H.task('A-NAT')
    runner = CapRunner(t, [_rec(t, tree='1' * 64, execution_time=150.0)])
    tools, _ = make_tools(runner, 'r')
    _score_tool(tools)({}, FakeAgent())
    assert runner.calls[-1][1]['wall_cap'] == 1270.0
