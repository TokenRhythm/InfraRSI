"""Basic agent of the native-method arms (EXT_INTERFACES §5; design §2.5, §4.3): a METR-style tool loop.

One JSON object per model turn, with an optional "note" (its reasoning, replayed next turn):
  {"tool":"bash","cmd":"...","timeout":<=1800}   {"tool":"python","code":"...","timeout":<=1800}
  {"tool":"<environment tool>", ...}             {"tool":"submit","note":"..."}   (ends the run)
The driver is any agent.backends backend (claude_cli for Opus 5.5, openai_compat for the self-hosted weak model); every
call is a fresh request replaying the whole trajectory, cut from the middle above `context_tokens` (default 150K) while
keeping the instructions and the last 40 turns. Accounting is InfraRSI's (agent.budget.SearchLedger: llm_tokens,
llm_usd, wall clock with model time included, experiments = environment tools flagged counts_as_experiment, their
GPU-hours); any exhausted dimension stops the loop and the environment glue submits the best result.

Tool commands run on the leased node as nobody (remote_exec.nobody_cmd) in the run's agent home, launched detached
with the run's INFRARSI_TOKEN, so an SSH drop never re-runs a command; whatever a call leaves running is killed when
it returns (no background jobs survive a tool call). Transcript, tool calls and full outputs are written to out_dir
(execution/ext_bench/<env>/<run_id>/agent/). `screens` in the trace have the SearchTrace row shape (trajectory.py).
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import math
import time
from pathlib import Path
from typing import Callable

from ...agent.backends.base import parse_json_object
from ...agent.budget import Budget, SearchLedger
from ...agent.context import estimate_tokens
from ...agent.jsonschema_lite import validate
from ...agent.optimize import MIN_CALL_WALL_S, output_allowance, usd_allowance, usd_kw
from .remote_exec import NOBODY, NodeOps, nobody_cmd, q, uid_kill

TRACE_SCHEMA = 'ext-basic-agent-trace-v1'
BASH_TIMEOUT_MAX = 1800
OUTPUT_HEAD = OUTPUT_TAIL = 8000
CONTEXT_TOKENS = 150_000
KEEP_LAST_TURNS = 40


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def truncate(text: str, head: int = OUTPUT_HEAD, tail: int = OUTPUT_TAIL) -> str:
    if text is None or len(text) <= head + tail:
        return text or ''
    return text[:head] + f'\n[... {len(text) - head - tail} characters omitted ...]\n' + text[-tail:]


@dataclasses.dataclass
class EnvTool:
    """An environment tool: fn(args, agent) -> JSON-able result shown to the model."""
    name: str
    description: str                       # one line for the system prompt (with an example call)
    properties: dict                       # JSON-schema properties of the arguments
    fn: Callable
    required: tuple = ()
    counts_as_experiment: bool = False     # charged against the experiment budget (checked before the call)
    terminal: bool = False                 # ends the loop after it ran (submit-like)
    admit: Callable | None = None          # admit(agent) -> None or the limiting dimension (worst-case admission,
                                           # e.g. EnvRunner.admits); default: the ledger's mean measured cost


# ---------------------------------------------------------------- remote shell
class ShellExecutor:
    """bash/python tool calls as an unprivileged uid on the node. work_dir (root-owned, 0700) keeps each call's script,
    stdout, stderr and exit code; python() sources go to code_dir (root-owned, group = the arm's gid, 0750/0640: the arm
    reads, never writes, other arms see nothing); the command runs in `cwd` with HOME=`home`. Nothing is written into the
    agent's home by root.

    Containment (Codex W9 B-02): `uid` is the arm's trust domain (one numeric uid per arm, no passwd entry, used by
    nothing else on the node); after every call, in a finally, every process of that uid is killed and verified gone
    (remote_exec.uid_kill; ContainmentError if one survives). The default uid 65534 (nobody) is kept for callers that
    have not moved to a trust-domain uid; it is never killed by uid (system daemons may run as nobody), only by the
    INFRARSI_TOKEN label, which an escaped process can drop - such callers do not get B-02 containment."""

    def __init__(self, ops: NodeOps, *, home: str, cwd: str, work_dir: str, token: str, path_dirs=(), gpus=(),
                 extra_env: dict | None = None, kill_leftovers: bool = True, clock=time.monotonic, sleep=time.sleep,
                 poll_s: float = 1.0, code_dir: str | None = None, uid: int = 65534):
        self.ops, self.home, self.cwd, self.work = ops, home, cwd, work_dir.rstrip('/')
        self.code_dir = (code_dir or self.work + '_code').rstrip('/')
        self.uid = int(uid)
        self.token, self.path_dirs, self.gpus, self.extra_env = token, tuple(path_dirs), tuple(gpus), dict(extra_env or {})
        self.kill_leftovers, self.clock, self.sleep, self.poll_s = kill_leftovers, clock, sleep, poll_s
        self.n = 0

    def _script(self, base: str, cmd: str, timeout: int) -> str:
        run = nobody_cmd(cmd, home=self.home, cwd=self.cwd, path_dirs=self.path_dirs, gpus=self.gpus, token=self.token,
                         extra=self.extra_env, timeout=timeout, uid=self.uid)
        return ('#!/bin/bash\n'
                f'mkdir {q(base + ".started")} 2>/dev/null || exit 0      # a repeated launch is a no-op\n'
                f'( {run} ) > {q(base + ".out")} 2> {q(base + ".err")} < /dev/null\n'
                f'echo $? > {q(base + ".rc.tmp")} && mv {q(base + ".rc.tmp")} {q(base + ".rc")}\n')

    def cleanup(self) -> list:
        """Kill everything the arm has running: all processes of its uid (verified; raises ContainmentError otherwise),
        or, for the legacy uid 65534, the processes carrying the token (and the root wrapper, which carries it too)."""
        if self.uid == NOBODY:
            return self.ops.kill_token(self.token) if self.kill_leftovers else []
        killed = self.ops.kill_token(self.token)          # the root-run wrapper scripts of this executor
        return sorted(set(killed) | set(uid_kill(self.ops, self.uid).get('killed') or []))

    def bash(self, cmd: str, timeout: float) -> dict:
        self.n += 1
        timeout = int(max(1, min(BASH_TIMEOUT_MAX, timeout or 600)))
        base = f'{self.work}/call_{self.n:04d}'
        t0 = self.clock()
        rc, killed = None, []
        try:
            self.ops.write_files({base + '.sh': self._script(base, cmd, timeout)}, mode=0o700)
            self.ops.launch(base + '.sh', f'{self.work}/launch.log', token=self.token)   # the wrapper carries the token too
            deadline, wait = t0 + timeout + 60, min(0.25, self.poll_s)
            while self.clock() < deadline:
                raw = self.ops.read_files([base + '.rc'])[base + '.rc']
                if raw is not None:
                    rc = int(raw.decode().strip() or -1)
                    break
                self.sleep(wait)
                wait = min(self.poll_s * 4, wait * 2)
        finally:
            killed = self.cleanup()                       # also when polling failed: nothing of this call survives
        outs = self.ops.read_files([base + '.out', base + '.err'], max_bytes=4 * (OUTPUT_HEAD + OUTPUT_TAIL))
        dec = {k: (v or b'').decode('utf-8', 'replace') for k, v in outs.items()}
        return {'exit_code': rc, 'timed_out': rc in (124, 137) or rc is None, 'seconds': round(self.clock() - t0, 3),
                'stdout': dec[base + '.out'], 'stderr': dec[base + '.err'], 'killed_leftovers': killed}

    def python(self, code: str, timeout: float) -> dict:
        path = f'{self.code_dir}/py_{self.n + 1:04d}.py'
        self.ops.mkdirs([self.code_dir], mode=0o750, gid=self.uid)
        self.ops.write_files({path: code}, mode=0o640, gid=self.uid)
        return self.bash(f'python {q(path)}', timeout)


# ---------------------------------------------------------------- the loop
class BasicAgent:
    def __init__(self, *, driver, executor: ShellExecutor | None, task_text: str, env_tools: list, budget: Budget,
                 run_id: str, env: str, task_id: str, arm: str, out_dir=None, clock=time.monotonic,
                 context_tokens: int = CONTEXT_TOKENS, keep_last: int = KEEP_LAST_TURNS, max_turns: int = 100000,
                 max_model_errors: int = 6, preamble: str = ''):
        self.driver, self.exec, self.task_text, self.budget = driver, executor, task_text, budget
        self.tools = {t.name: t for t in env_tools}
        for name in ('bash', 'python', 'submit'):
            if name in self.tools:
                raise ValueError(f'environment tool name {name!r} is reserved')
        self.run_id, self.env, self.task_id, self.arm = run_id, env, task_id, arm
        self.out = Path(out_dir) if out_dir else None
        if self.out:
            (self.out / 'outputs').mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.ledger = SearchLedger(budget, clock)
        self.context_tokens, self.keep_last = int(context_tokens), int(keep_last)
        self.max_turns, self.max_model_errors = max_turns, max_model_errors
        self.preamble = preamble
        self.turns: list[dict] = []
        self.screens, self.experiments, self.llm_calls = [], [], []
        self.stop_reason: str | None = None
        self.started_utc = _utc()
        self.system = self._system()

    # ------------------------------------------------------------ prompts
    def _system(self) -> str:
        rows = [self.preamble or 'You are an autonomous agent working on a remote Linux machine through tools.',
                'Reply with exactly one JSON object per turn and nothing else (no markdown). Tools:',
                f'{{"tool":"bash","cmd":"<shell command>","timeout":600,"note":"<optional reasoning>"}}  timeout <= {BASH_TIMEOUT_MAX} s',
                '{"tool":"python","code":"<python source>","timeout":600,"note":"..."}  runs with the environment\'s python']
        rows += [t.description for t in self.tools.values()]
        rows += ['{"tool":"submit","note":"<why you stop>"}  ends your run (the best recorded result counts)',
                 'Each call starts a fresh shell in your working directory (cd and exports do not persist); processes still '
                 'running when a call returns are stopped; long outputs are cut to their first and last '
                 f'{OUTPUT_HEAD} characters. There is no internet access. The budget block of every turn shows what is left.']
        return '\n'.join(rows)

    def _schema_error(self, a: dict) -> str | None:
        tool = a.get('tool')
        note = {'note': {'type': 'string', 'maxLength': 4000}}
        if tool == 'bash':
            props, req = {'tool': {}, 'cmd': {'type': 'string', 'minLength': 1, 'maxLength': 100000}, 'timeout': {'type': 'number'}, **note}, ['cmd']
        elif tool == 'python':
            props, req = {'tool': {}, 'code': {'type': 'string', 'minLength': 1, 'maxLength': 200000}, 'timeout': {'type': 'number'}, **note}, ['code']
        elif tool == 'submit':
            props, req = {'tool': {}, **note}, []
        elif tool in self.tools:
            t = self.tools[tool]
            props, req = {'tool': {}, **t.properties, **note}, list(t.required)
        else:
            return f'unknown tool {tool!r}; tools: bash, python, {", ".join(self.tools)}, submit'
        errs = validate(a, {'type': 'object', 'properties': props, 'required': ['tool', *req], 'additionalProperties': False})
        return '; '.join(errs[:4]) if errs else None

    def budget_view(self) -> dict:
        b, u = self.budget, self.ledger.snapshot()
        return {'experiments': f'{u["screens"]}/{b.screens}', 'llm_tokens': f'{u["llm_tokens"]}/{b.llm_tokens}',
                'wall_h': f'{u["wall_seconds"] / 3600:.2f}/{b.wall_seconds / 3600:.2f}',
                'gpu_hours': f'{u["gpu_hours"]:.2f}/{b.gpu_hours}'}

    def _turn_text(self, t: dict, short: bool = False) -> str:
        res = json.dumps(t['result'], ensure_ascii=False, default=str)
        if short:
            res = res[:600] + (' ...' if len(res) > 600 else '')
        return f'## turn {t["turn"]}\naction: {json.dumps(t["action"], ensure_ascii=False)}\nresult: {res}'

    def render(self) -> str:
        head = ['# TASK', self.task_text, '# BUDGET', json.dumps(self.budget_view())]
        tail = ['# YOUR TURN', 'Reply with exactly one JSON tool call.']
        limit = self.context_tokens - estimate_tokens(self.system) - 2000
        parts = [self._turn_text(t) for t in self.turns]
        text = '\n'.join(head + ['# HISTORY'] + parts + tail)
        if estimate_tokens(text) <= limit:
            return text
        # Cut from the middle: the instructions stay, then the last keep_last turns; older kept turns shrink first.
        kept = self.turns[-self.keep_last:]
        omitted = len(self.turns) - len(kept)
        for short_upto in range(0, len(kept) + 1):
            parts = [self._turn_text(t, short=i < short_upto) for i, t in enumerate(kept)]
            note = [f'[{omitted} earlier turns omitted]'] if omitted else []
            text = '\n'.join(head + ['# HISTORY'] + note + parts + tail)
            if estimate_tokens(text) <= limit:
                return text
        while len(kept) > 1 and estimate_tokens(text) > limit:
            kept, omitted = kept[1:], omitted + 1
            text = '\n'.join(head + ['# HISTORY', f'[{omitted} earlier turns omitted]'] + [self._turn_text(t, short=True) for t in kept] + tail)
        return text

    # ------------------------------------------------------------ accounting
    def charge_llm(self, res, turn: int) -> None:
        self.ledger.charge_llm(res.llm_tokens, res.llm_usd, res.accounting_complete)
        row = {'turn': turn, 'role': 'agent', 'model': res.model, 'observed_models': res.observed_models, 'llm_tokens': res.llm_tokens,
               'llm_usd': res.llm_usd, 'wall_seconds': round(res.wall_seconds, 3), 'accounting_complete': res.accounting_complete,
               'error': res.error, 'attempts': res.attempts, 'call_id': res.meta.get('call_id'),
               'selfhost_gpu_seconds': res.meta.get('selfhost_gpu_seconds'), 'gpu_type': res.meta.get('gpu_type')}
        self.llm_calls.append(row)
        self._log('llm_call', row)

    def experiment_admitted(self, tool: EnvTool | None = None) -> str | None:
        """None if one more experiment fits, else the dimension: the tool's own worst-case admission when it has one
        (Codex W9 H-05), else count, GPU-hours and wall at the mean measured cost."""
        if tool is not None and tool.admit is not None:
            return tool.admit(self)
        return self.ledger.admit_screen()

    def wall_left(self) -> float:
        return self.budget.wall_seconds - self.ledger.wall()

    def record_experiment(self, receipt: dict, verdict, *, candidate_id: str, kind: str = 'experiment', extra: dict | None = None) -> dict:
        """Charge an experiment (cost from the signed receipt) and append a SearchTrace-shaped screen row."""
        cost = receipt.get('cost') if isinstance(receipt.get('cost'), dict) else {}
        gpu_s, wall_s = cost.get('gpu_seconds'), cost.get('wall_seconds')
        known = cost.get('accounting_complete') is True and type(gpu_s) in (int, float)
        self.ledger.charge_screen((gpu_s or 0.0) / 3600.0, wall_s if type(wall_s) in (int, float) else 0.0, known)
        v = verdict.to_dict() if hasattr(verdict, 'to_dict') else dict(verdict)
        valid_rows = [s for s in self.screens if s['valid'] and s['step_time']] + ([{'candidate_id': candidate_id, 'step_time': v.get('step_time')}]
                                                                                if v.get('valid') and v.get('step_time') else [])
        best = min(valid_rows, key=lambda s: s['step_time']) if valid_rows else None
        row = {'index': len(self.screens), 'candidate_id': candidate_id, 'kind': kind, 'attempt_id': receipt.get('attempt_id'),
               'valid': bool(v.get('valid')), 'step_time': v.get('step_time'), 'reasons': (v.get('reasons') or [])[:4],
               'gpu_hours': (gpu_s or 0.0) / 3600.0, 'wall_seconds': wall_s, 'cum': self.ledger.snapshot(),
               'fractions': self.ledger.fractions(), 'best_valid': best and best['candidate_id'],
               'best_step_time': best and best['step_time'], **(extra or {})}
        self.screens.append(row)
        self.experiments.append({'candidate_id': candidate_id, 'receipt': receipt, 'verdict': v})
        if self.out:
            d = self.out / 'receipts'
            d.mkdir(parents=True, exist_ok=True)
            (d / f'{candidate_id}.json').write_text(json.dumps(receipt, ensure_ascii=False, default=str), encoding='utf-8')
        self._log('screen', row)
        return row

    def _log(self, kind: str, payload: dict) -> None:
        if self.out:
            e = {'utc': _utc(), 'wall': round(self.ledger.wall(), 3), 'kind': kind, 'payload': payload}
            with (self.out / 'events.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(e, ensure_ascii=False, default=str) + '\n')

    # ------------------------------------------------------------ tools
    def _dispatch(self, a: dict, turn: int) -> tuple[dict, bool]:
        tool = a['tool']
        left = self.budget.wall_seconds - self.ledger.wall()
        if tool == 'submit':
            return {'submitted': True}, True
        if tool in ('bash', 'python'):
            if self.exec is None:
                return {'error': 'no shell in this environment'}, False
            timeout = min(float(a.get('timeout') or 600), BASH_TIMEOUT_MAX, max(1.0, left))
            r = self.exec.bash(a['cmd'], timeout) if tool == 'bash' else self.exec.python(a['code'], timeout)
            if self.out:
                (self.out / 'outputs' / f'turn_{turn:05d}.json').write_text(json.dumps(r, ensure_ascii=False), encoding='utf-8')
            return {'exit_code': r['exit_code'], 'timed_out': r['timed_out'], 'seconds': r['seconds'],
                    'stdout': truncate(r['stdout']), 'stderr': truncate(r['stderr'])}, False
        t = self.tools[tool]
        if t.counts_as_experiment:
            limit = self.experiment_admitted(t)
            if limit:
                self.stop_reason = f'budget:{limit}'
                return {'error': f'experiment budget exhausted ({limit}); the run stops'}, True
        try:
            res = t.fn({k: v for k, v in a.items() if k not in ('tool', 'note')}, self)
        except Exception as exc:                     # noqa: BLE001 - a tool bug is reported to the model, logged
            res = {'error': f'tool {tool} failed: {type(exc).__name__}: {str(exc)[:500]}'}
        if self.out:
            (self.out / 'outputs' / f'turn_{turn:05d}.json').write_text(json.dumps(res, ensure_ascii=False, default=str), encoding='utf-8')
        return res, t.terminal

    def run(self) -> dict:
        errors_in_row, turn = 0, 0
        while self.stop_reason is None:
            ex = self.ledger.exhausted()
            if ex:
                self.stop_reason = 'budget:' + ex[0]
                break
            if turn >= self.max_turns:
                self.stop_reason = 'max_turns'
                break
            left = self.budget.wall_seconds - self.ledger.wall()
            if left < MIN_CALL_WALL_S:
                self.stop_reason = 'budget:wall_seconds'
                break
            turn += 1
            prompt = self.render()
            cap = output_allowance(self.budget, self.ledger, self.driver, self.system + prompt)
            if cap is None:
                self.stop_reason = 'budget:llm_tokens'
                break
            ok, usd_left = usd_allowance(self.budget, self.ledger, self.driver, self.system + prompt, cap)
            if not ok:                                  # Codex W9 H-06: the USD cap is enforced before the call
                self.stop_reason = 'budget:llm_usd'
                break
            res = self.driver.complete(self.system, prompt, schema=None, call_dir=(self.out / 'calls' / f'{turn:05d}') if self.out else None,
                                       role='agent', timeout=max(1.0, left), max_output_tokens=cap,
                                       token_allowance=self.budget.llm_tokens - self.ledger.used['llm_tokens'], **usd_kw(usd_left))
            self.charge_llm(res, turn)
            if res.meta.get('unreachable'):
                self.stop_reason = 'endpoint_unreachable'
                break
            action = res.action
            if action is None and res.text and not res.error:
                action, _ = parse_json_object(res.text)
            problem = res.error if res.error and action is None else (None if action is not None else 'reply is not a JSON object')
            if problem is None:
                problem = self._schema_error(action)
            if problem:
                errors_in_row += 1
                entry = {'turn': turn, 'action': action or {'raw': (res.text or '')[:400]}, 'result': {'error': f'rejected: {problem}'}}
                self.turns.append(entry)
                self._log('model_error', entry)
                if errors_in_row >= self.max_model_errors:
                    self.stop_reason = 'model_errors'
                continue
            errors_in_row = 0
            result, terminal = self._dispatch(action, turn)
            entry = {'turn': turn, 'action': action, 'result': result}
            self.turns.append(entry)
            self._log('turn', {'turn': turn, 'tool': action.get('tool'), 'action': action, 'result': result})
            if terminal and self.stop_reason is None:
                self.stop_reason = 'submit' if action['tool'] == 'submit' else f'tool:{action["tool"]}'
        return self.trace()

    def trace(self, *, selected: str | None = None, selected_value: float | None = None, extra: dict | None = None) -> dict:
        t = {'schema': TRACE_SCHEMA, 'run_id': self.run_id, 'arm': self.arm, 'env': self.env, 'task_id': self.task_id,
             'driver': getattr(self.driver, 'describe', lambda: {'kind': type(self.driver).__name__})(),
             'budget': self.budget.as_dict(), 'used': self.ledger.snapshot(), 'accounting_complete': self.ledger.accounting_complete,
             'stop_reason': self.stop_reason or 'unknown', 'turns': len(self.turns), 'screens': self.screens,
             'llm_calls': self.llm_calls, 'selected': selected, 'selected_value': selected_value, 'started_utc': self.started_utc,
             'ended_utc': _utc(), 'out_dir': str(self.out) if self.out else None, 'extra': extra or {}}
        if self.out:
            (self.out / 'trace.json').write_text(json.dumps(t, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
        return t


def keep_revert_tools(ops: NodeOps, *, repo: str, user_root: str, head_dir: str, files=('train.py',), uid: int = 65534) -> list:
    """Directory-copy keep/revert (D-AR-6, for nodes without git): keep() makes the working files the new head (a
    root-owned copy the agent cannot touch), revert() restores them from the head; the head starts as the files at
    setup time. Files inside the agent's tree (user_root) are read and written without following symlinks."""
    names = [str(f) for f in files]
    ops.mkdirs([head_dir], mode=0o700)
    state = {'keeps': 0, 'reverts': 0}

    def snapshot() -> list:
        missing = []
        for n in names:
            data = ops.read_user_file(f'{repo}/{n}', user_root, max_bytes=4 << 20)
            if data is None:
                missing.append(n)
            else:
                ops.write_files({f'{head_dir}/{n}': data}, mode=0o600)
        return missing

    if snapshot():
        raise RuntimeError('keep/revert setup: working files missing or not regular files')

    def keep(args, agent):
        bad = snapshot()
        state['keeps'] += not bad
        return {'kept': not bad, 'head_version': state['keeps'], 'files': names, **({'error': f'not regular files: {bad}'} if bad else {})}

    def revert(args, agent):
        heads = ops.read_files([f'{head_dir}/{n}' for n in names])
        for n in names:
            ops.write_user_file(f'{repo}/{n}', user_root, heads[f'{head_dir}/{n}'] or b'', uid=uid, gid=uid, mode=0o644)
        state['reverts'] += 1
        return {'reverted': True, 'head_version': state['keeps'], 'files': names}

    return [EnvTool('keep', '{"tool":"keep","note":"..."}  make the current ' + ', '.join(names) + ' the new head (an improvement)',
                    {}, keep),
            EnvTool('revert', '{"tool":"revert","note":"..."}  restore ' + ', '.join(names) + ' from the head (discard the change)',
                    {}, revert)]


def best_screen(trace: dict) -> dict | None:
    """The valid screen with the lowest metric (ties: earliest)."""
    ok = [s for s in trace.get('screens') or [] if s.get('valid') and type(s.get('step_time')) in (int, float)
          and math.isfinite(s['step_time'])]
    return min(ok, key=lambda s: (s['step_time'], s['index'])) if ok else None
