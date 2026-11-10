"""A-NAT: the METR-style basic agent on ENV-A (design §2.5, EXT_INTERFACES §5; AMENDMENT_EXT_001; Codex W9 R1).

setup_home builds the agent's run home: a root-owned home dir (0755) holding the root-owned, read-only env.sh (its exact
expected content, M-01) and train_data.jsonl, and the starting solution (base llm-foundry + finetune.sh/yaml) owned by
the arm's uid 64101 (B-02); the agent's HOME (offline HF cache) and a short private TMPDIR are owned by that uid too.
`runner.prepare` then installs the reference (root-owned 0700, safetensors only) and the judge's base-model copy. The
agent gets the task's own main instructions (path-substituted) plus a note about the score/submit tools and the budget;
its bash/python tools run as uid 64101 on GPUs 0-3 (ShellExecutor; every process of the uid is killed after each
call). The score tool scores the current solution in place through the ENV-A runner (score.py the authority) and
keeps its snapshot; after the loop the selected (best valid) snapshot is restored and locked-re-scored with G4.
`context-check` (AMENDMENT Q1) renders one request at the 72K limit and sends it to the endpoint.
"""
from __future__ import annotations

import collections
import hashlib
import json

from ..common.basic_agent import BasicAgent, ShellExecutor
from ..common.remote_exec import new_token, q
from . import source
from . import tools as tools_mod
from .task import (ARM_UIDS, BASE_MODEL_REV, NODE_ENV, NODE_FOUNDRY, NODE_MODEL, NODE_RUNS, NODE_TASK, arm_deviations,
                   arm_uid, env_sh_text)

CONTEXT_TOKENS = 72_000           # AMENDMENT Q1 (keeps prompt + 4096 output < 98_304 at the worst real bytes/token)
KEEP_LAST = 40
OFFLINE = {'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
A_NAT_UID = ARM_UIDS['A-NAT']
LOOP_CONSECUTIVE = 12             # AMENDMENT_EXT_006: one (tool, args, output) signature >= 12 times in a row ...
LOOP_WINDOW, LOOP_WINDOW_COUNT = 60, 30   # ... or >= 30 times within the last 60 turns -> stop (degenerate_loop)


class LoopDetector:
    """AMENDMENT_EXT_006 stop-only loop detector: a turn's signature is (tool, arguments without the free-text note,
    the output shown to the agent without its wall-time field). The agent never sees it; it only ends the run."""

    def __init__(self, consecutive: int = LOOP_CONSECUTIVE, window: int = LOOP_WINDOW, window_count: int = LOOP_WINDOW_COUNT):
        self.consecutive, self.window, self.window_count = int(consecutive), int(window), int(window_count)
        self.recent = collections.deque(maxlen=self.window)
        self.last, self.run, self.turns = None, 0, 0
        self.n_identical_max, self.window_max, self.tripped = 0, 0, None

    @staticmethod
    def signature(action: dict, result) -> str:
        action = action if isinstance(action, dict) else {}
        args = {k: v for k, v in action.items() if k not in ('tool', 'note')}
        out = {k: v for k, v in result.items() if k != 'seconds'} if isinstance(result, dict) else result
        return hashlib.sha256(json.dumps([action.get('tool'), args, out], sort_keys=True, default=str).encode('utf-8')).hexdigest()

    def observe(self, action: dict, result, turn: int | None = None) -> bool:
        """Record one dispatched turn; True once the run must stop."""
        s = self.signature(action, result)
        self.turns += 1
        self.run = self.run + 1 if s == self.last else 1
        self.last = s
        self.recent.append(s)
        in_window = self.recent.count(s)
        self.n_identical_max, self.window_max = max(self.n_identical_max, self.run), max(self.window_max, in_window)
        if self.tripped is None and (self.run >= self.consecutive or in_window >= self.window_count):
            self.tripped = {'rule': f'>= {self.consecutive} in a row' if self.run >= self.consecutive
                            else f'>= {self.window_count} in the last {self.window}',
                            'turn': turn, 'consecutive': self.run, 'in_window': in_window,
                            'tool': (action or {}).get('tool') if isinstance(action, dict) else None, 'signature': s[:16]}
        return self.tripped is not None

    def summary(self) -> dict:
        return {'rule': {'consecutive': self.consecutive, 'window': self.window, 'window_count': self.window_count},
                'turns_observed': self.turns, 'n_identical_max': self.n_identical_max, 'window_max': self.window_max,
                'tripped': self.tripped}


class LoopStopAgent(BasicAgent):
    """The METR-style basic agent, unchanged for the model (same prompt, tools and outputs), plus the stop-only loop
    detector of AMENDMENT_EXT_006: after a dispatched turn that trips it the run ends with stop_reason
    `degenerate_loop`, and the arm continues on its normal path (best valid score so far -> locked re-score + G4)."""

    def __init__(self, *args, loop_detector: LoopDetector | None = None, **kw):
        super().__init__(*args, **kw)
        self.loop = loop_detector or LoopDetector()

    def _dispatch(self, a: dict, turn: int):
        result, terminal = super()._dispatch(a, turn)
        if self.loop.observe(a, result, turn) and self.stop_reason is None:
            self.stop_reason = 'degenerate_loop'
            self._log('loop_detector', self.loop.summary())         # the transcript the model sees is untouched
        return result, terminal


def layout(run_id: str, runs_root: str = NODE_RUNS) -> dict:
    run_dir = f'{runs_root}/{run_id}'
    short = hashlib.sha256(run_id.encode('utf-8')).hexdigest()[:10]
    return {'run_dir': run_dir, 'home': f'{run_dir}/home_agent', 'solution': f'{run_dir}/home_agent/solution',
            'home_root': f'{run_dir}/home_root', 'tool': f'{run_dir}/tool',
            'ref': f'{run_dir}/ref/trained_model_reference',
            'tmp': f'/tmp/isx2nat-{short}'}          # short TMPDIR: AF_UNIX paths of the agent's own runs fit 108 bytes


def setup_home(ops, run_id: str, *, uid: int = A_NAT_UID, runs_root: str = NODE_RUNS, base_solution: dict | None = None) -> dict:
    """The agent's starting home (see the module doc). Root writes everything first and hands the uid's parts over
    last; nothing the uid owns is written by root afterwards."""
    L = layout(run_id, runs_root)
    ops.mkdirs([L['run_dir']], mode=0o711)
    ops.mkdirs([f'{L["run_dir"]}/results', f'{L["run_dir"]}/snapshots', L['tool']], mode=0o700)
    ops.mkdirs([L['home'], L['solution']], mode=0o755)
    r = ops.sh(f'set -e; cp -a {q(NODE_FOUNDRY)}/. {q(L["solution"])}/ && '
               f'cp {q(NODE_TASK + "/assets/train_data.jsonl")} {q(L["home"] + "/train_data.jsonl")}', timeout=300)
    if r.get('code') not in (0, None):
        raise RuntimeError(f'A-NAT base solution setup failed: {(r.get("stderr") or "")[-400:]}')
    base = base_solution if base_solution is not None else source.base_solution()
    ops.write_files({f'{L["solution"]}/finetune.sh': base['finetune.sh'],
                     f'{L["solution"]}/finetune.yaml': base['finetune.yaml']}, mode=0o644)
    ops.write_files({f'{L["home"]}/env.sh': env_sh_text(L['home'])}, mode=0o644)
    cache = f'{L["home_root"]}/.cache/huggingface/hub/models--deepseek-ai--deepseek-coder-1.3b-base'
    ops.mkdirs([L['home_root'], f'{cache}/refs', f'{cache}/snapshots/{BASE_MODEL_REV}'])
    ops.write_files({f'{cache}/refs/main': BASE_MODEL_REV}, mode=0o644)
    ops.symlinks({f'{cache}/snapshots/{BASE_MODEL_REV}/{fn}': f'{NODE_MODEL}/{fn}' for fn in source.MODEL_FILES})
    ops.sh(f'set -e; chmod +x {q(L["solution"] + "/finetune.sh")}; '
           f'chown -R 0:0 {q(L["home"])}; chmod 755 {q(L["home"])}; chmod 644 {q(L["home"] + "/env.sh")} '
           f'{q(L["home"] + "/train_data.jsonl")}; '
           f'chown -R -h {uid}:{uid} {q(L["solution"])} {q(L["home_root"])}; chmod 700 {q(L["solution"])} {q(L["home_root"])}; '
           f'rm -rf {q(L["tmp"])}; mkdir -m 700 {q(L["tmp"])}; chown {uid}:{uid} {q(L["tmp"])}', timeout=300)
    return L


def build_agent(task, driver, runner, ops, L: dict, *, run_id: str, raw_instructions: str | None = None,
                context_tokens: int = CONTEXT_TOKENS, keep_last: int = KEEP_LAST, out_dir=None, clock=None):
    """Wire the ShellExecutor (the arm uid) + score tool + BasicAgent; does not run the loop (caller does / dry-run)."""
    shell = ShellExecutor(ops, home=L['home_root'], cwd=L['solution'], work_dir=L['tool'], token=new_token('isx2-nat'),
                          path_dirs=(f'{NODE_ENV}/bin', '/usr/local/cuda-12.8/bin'), gpus=tuple(runner.gpus),
                          extra_env={**OFFLINE, 'XDG_CACHE_HOME': f'{L["home_root"]}/.cache', 'TMPDIR': L['tmp']},
                          uid=runner.uid)
    tools, state = tools_mod.make_tools(runner, run_id)
    text = source.task_text(ops, task, L['home'], raw_instructions=raw_instructions)
    kw = {'clock': clock} if clock is not None else {}
    agent = LoopStopAgent(driver=driver, executor=shell, task_text=text, env_tools=tools, budget=task.search_budget(),
                       run_id=run_id, env='rebench_llmf', task_id=task.id, arm=task.arm or 'A-NAT', out_dir=out_dir,
                       context_tokens=context_tokens, keep_last=keep_last, **kw)
    return agent, state


def run_native(task, driver, runner, ops, *, run_id: str, out_dir=None, raw_instructions=None,
               context_tokens: int = CONTEXT_TOKENS, keep_last: int = KEEP_LAST, clock=None, dry_run: bool = False,
               restart_of: str | None = None) -> dict:
    """Set up, prepare the judge side, build the agent, and (unless dry_run) run the loop -- which the loop detector
    may end (degenerate_loop) -- then locked-re-score the selected snapshot. The trace's `selected` is the same
    candidate the locked re-score restores. restart_of: the abandoned run this one restarts (AMENDMENT_EXT_006)."""
    arm = task.arm or 'A-NAT'
    expected_uid = arm_uid(arm)
    if runner.uid != expected_uid:
        raise ValueError(f'{arm} runs as uid {expected_uid}, runner has {runner.uid}')
    L = setup_home(ops, run_id, uid=runner.uid, **({'base_solution': runner.base} if arm == 'A-NAT2' else {}))
    agent, state = build_agent(task, driver, runner, ops, L, run_id=run_id, raw_instructions=raw_instructions,
                               context_tokens=context_tokens, keep_last=keep_last, out_dir=out_dir, clock=clock)
    if dry_run:
        return {'dry_run': True, 'arm': arm, 'layout': L, 'uid': runner.uid, 'task_text_chars': len(agent.task_text),
                'tools': sorted(agent.tools), 'budget': task.budget, 'context_tokens': context_tokens,
                'keep_last': keep_last, 'driver': agent.driver.describe() if hasattr(agent.driver, 'describe') else None,
                'loop_detector': agent.loop.summary()['rule'], 'restart_of': restart_of,
                'deviations': sorted(set(runner.deviations) | set(arm_deviations(arm)))}
    prep = runner.prepare(run_id)
    if arm == 'A-NAT2':
        # D-RB-18: the untouched S0 is the first charged score, before any model or shell call.
        # Use the existing tool's admission, snapshot and accounting; show the result as turn zero.
        action = {'tool': 'score', 'note': 'S0 reference before search'}
        result, _ = agent._dispatch(action, 0)
        entry = {'turn': 0, 'action': action, 'result': result}
        agent.turns.append(entry)
        agent._log('turn', {'tool': 'score', **entry})
    agent.run()
    locked = tools_mod.locked_rescore(runner, run_id, state)
    sel = state.best
    return agent.trace(selected=sel and sel['candidate_id'], selected_value=sel and sel['execution_time'],
                       extra={'arm': arm, 'layout': L, 'uid': runner.uid, 'prepare': prep, 'locked': locked,
                              'scores': state.scores, 'loop_detector': agent.loop.summary(), 'restart_of': restart_of,
                              'n_identical_max': agent.loop.n_identical_max,
                              'deviations': sorted(set(runner.deviations) | set(arm_deviations(arm)))})


def context_check_prompt(context_tokens: int = CONTEXT_TOKENS) -> tuple[str, str]:
    """AMENDMENT Q1 pre-start check: a (system, prompt) rendered to ~context_tokens estimate-units of log/JSON-like
    filler, to confirm the server accepts it with usage.prompt_tokens < 94_000. `estimate_tokens` = UTF-8 bytes/3.5."""
    system = ('You are scoring an LLM-foundry finetuning solution. Reply with exactly one JSON object '
              '{"tool":"score","note":"..."} and nothing else.')
    filler_unit = ('{"turn": 0, "action": {"tool": "bash", "cmd": "sed -n 1,40p finetune.sh"}, '
                   '"result": {"exit_code": 0, "stdout": "time python scripts/data_prep/convert_finetuning_dataset.py '
                   '--dataset=json --splits=train --data_files ../train_data.jsonl --out_root finetuning_tokens '
                   '--skip-preprocessing --max_seq_len 4096 --tokenizer ${BASE_MODEL}\\nreal\\t0m9.4s"}}\n')
    target_bytes = int(context_tokens * 3.5) - len(system.encode('utf-8')) - 200
    body = (filler_unit * (target_bytes // len(filler_unit) + 1))[:max(0, target_bytes)]
    prompt = '# TASK\nReduce finetune.sh runtime.\n# HISTORY\n' + body + '\n# YOUR TURN\nReply with one JSON tool call.'
    return system, prompt


def context_check(driver, *, context_tokens: int = CONTEXT_TOKENS, prompt_token_limit: int = 94_000, timeout: float = 600) -> dict:
    """Send the 72K request once; pass iff the call is accepted and usage.prompt_tokens < prompt_token_limit."""
    system, prompt = context_check_prompt(context_tokens)
    res = driver.complete(system, prompt, schema=None, role='context_check', timeout=timeout,
                          max_output_tokens=256, token_allowance=10 ** 9)
    usage = {}
    for a in (res.meta or {}).get('attempts') or []:
        if isinstance(a.get('usage'), dict):
            usage = a['usage']
    prompt_tokens = usage.get('prompt_tokens')
    accepted = res.error is None or not res.meta.get('unreachable', False)
    ok = (res.error is None) and type(prompt_tokens) is int and prompt_tokens < prompt_token_limit
    return {'ok': bool(ok), 'accepted': bool(accepted), 'prompt_tokens': prompt_tokens, 'limit': prompt_token_limit,
            'error': res.error, 'estimate_units': context_tokens, 'wall_seconds': round(res.wall_seconds, 2)}
