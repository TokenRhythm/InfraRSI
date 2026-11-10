"""ENV-A `score()` tool for the native basic agent A-NAT (design §2.5, EXT_INTERFACES §5), plus the locked re-score.

`make_tools(runner, run_id)` returns a `common.basic_agent.EnvTool` list for `BasicAgent(env_tools=...)`:
  * `score` (counts_as_experiment): runs the agent's current (edited, in place) solution through the ENV-A runner --
    on the node the solution tree is snapshot (a tar named by its content digest) before the task's own score.py
    scores it (the authority), and scores.jsonl is appended (D-RB-9) -- charges the experiment budget via
    `agent.record_experiment`, and returns only the published-protocol fields (execution_time, norm_diff, score).
`submit` is BasicAgent's built-in terminal tool. After the loop `locked_rescore(runner, run_id, state)` performs the
§1.4 locked re-scoring of the SELECTED solution (Codex W9 B-05): the best valid score -- the same rule as the
trace's `selected` (valid receipt, lowest execution_time, earliest on ties) -- is restored from its snapshot into a
fresh home and re-scored with G4; the locked receipt's candidate_digest must equal the snapshot's tree digest (else
G6 rejects it). The final working tree is never scored. The runner holds the node for the whole run (lease mode
'held'), so each score takes only the per-GPU claim the run already owns.
"""
from __future__ import annotations

from ..common.basic_agent import EnvTool
from ..common.env_judge import EnvJudge
from .guards import judge_metrics
from .task import START_TREE_DIGEST


class ScoreState:
    """Per-run A-NAT bookkeeping: every score() call with its snapshot identity, and the selected (best valid) one."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.scores: list[dict] = []
        self.best: dict | None = None

    def record(self, receipt: dict, verdict=None, *, candidate_id: str | None = None) -> dict:
        m = judge_metrics(receipt.get('judge'))
        v = verdict.to_dict() if hasattr(verdict, 'to_dict') else (dict(verdict) if verdict else {})
        snap = receipt.get('snapshot') if isinstance(receipt.get('snapshot'), dict) else {}
        valid = bool(v['valid']) if 'valid' in v else bool(receipt.get('valid'))
        row = {'n': len(self.scores) + 1, 'candidate_id': candidate_id, 'valid': valid,
               'judge_valid': bool(receipt.get('judge_valid')), 'execution_time': m.get('execution_time'),
               'norm_diff': m.get('norm_diff'), 'score': m.get('score'), 'attempt_id': receipt.get('attempt_id'),
               'guard_rejections': [r['id'] for r in receipt.get('guard_rejections', [])],
               'tree_digest': receipt.get('tree_digest') or snap.get('tree_digest'),
               'snapshot_path': snap.get('path'), 'snapshot_sha256': snap.get('sha256')}
        self.scores.append(row)
        t = row['execution_time']
        if valid and type(t) in (int, float) and t > 0 and row['snapshot_path'] and row['tree_digest']:
            if self.best is None or t < self.best['execution_time']:          # ties: the earlier score stays
                self.best = row
        return row


def make_tools(runner, run_id: str, *, state: ScoreState | None = None):
    st = state or ScoreState(run_id)
    judge = EnvJudge('rebench_llmf')

    def score(args, agent):
        cid = f'score{len(st.scores) + 1:02d}'
        # Codex W9 R2 H-05: the scoring runs under the cap it was admitted with (the runner's declared worst case),
        # so the node job is bounded to what the admission reserved.
        cap = runner.worst_case() if hasattr(runner, 'worst_case') else None
        receipt = runner.run(runner.task, None, None, 'screen', run_id, reuse_home=True, candidate_id=cid,
                             **({'wall_cap': cap} if cap else {}))
        verdict = judge(receipt, None, None)
        if agent is not None and hasattr(agent, 'record_experiment'):
            agent.record_experiment(receipt, verdict, candidate_id=cid, kind=runner.task.arm or 'A-NAT')
        row = st.record(receipt, verdict, candidate_id=cid)
        out = {'execution_time': row['execution_time'], 'norm_diff': row['norm_diff'], 'score': row['score'],
               'valid': row['judge_valid'], 'n_scored': row['n']}
        if not row['judge_valid']:
            out['message'] = 'invalid: ' + str((receipt.get('judge') or {}).get('message', receipt.get('invalid_reasons')))[:200]
        return out

    tools = [EnvTool(name='score', description='{"tool":"score","note":"<what this solution tries>"}  score your current '
                     'finetune.sh: returns execution_time (s), norm_diff and the log-seconds score. Every score is '
                     'recorded; you are judged on the best valid solution.',
                     properties={}, fn=score, counts_as_experiment=True,
                     # Codex W9 H-05: admitted only when one scoring's worst case still fits the wall / GPU budget
                     admit=(lambda agent: runner.admits(agent.ledger, agent.budget)) if hasattr(runner, 'admits') else None)]
    return tools, st


def locked_rescore(runner, run_id: str, state: ScoreState | None) -> dict:
    """The §1.4 locked re-scoring (purpose 'score', clean slate, with G4) of the selected snapshot: the decision
    authority for A-NAT. Without a valid score nothing is re-scored (the final working tree never is)."""
    best = state.best if state else None
    if best is None:
        return {'locked': None, 'selected': None, 'reason': 'no valid score: nothing to re-score'}
    sel = {k: best[k] for k in ('candidate_id', 'n', 'execution_time', 'score', 'tree_digest', 'snapshot_path')}
    if best.get('tree_digest') == START_TREE_DIGEST:          # decision D20: the untouched start is not re-scored
        return {'locked': None, 'skipped': 'selection_is_start', 'selected': sel, 'node': getattr(runner, 'node', None),
                'locked_value': 'node S0 reference', 'invalid_accept': 0, 'rule': 'D20'}
    receipt = runner.run(runner.task, None, None, 'score', run_id, run_g4=True, restore_snapshot=best['snapshot_path'],
                         expected_digest=best['tree_digest'], candidate_id='locked')
    m = judge_metrics(receipt.get('judge'))
    digest_ok = receipt.get('candidate_digest') == best['tree_digest']
    return {'locked': {'valid': bool(receipt.get('valid')) and digest_ok, 'execution_time': m.get('execution_time'),
                       'score': m.get('score'), 'norm_diff': m.get('norm_diff'),
                       'guard_rejections': [r['id'] for r in receipt.get('guard_rejections', [])],
                       'candidate_digest': receipt.get('candidate_digest'), 'digest_matches_selection': digest_ok},
            'selected': {k: best[k] for k in ('candidate_id', 'n', 'execution_time', 'score', 'tree_digest', 'snapshot_path')},
            'attempt_id': receipt.get('attempt_id')}
