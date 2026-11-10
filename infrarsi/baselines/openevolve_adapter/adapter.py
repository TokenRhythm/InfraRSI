"""openevolve_L0L1 (design E2, implementation track): OpenEvolve (Apache-2.0, fetched to third_party_src/openevolve,
run unmodified) evolves one candidate "program" = an L0 config + L1 overlay texts. The evaluator is our harness +
reward through the same SearchSession as every arm (same budget, mutable/frozen zones, patch guard, judge); the LLM
is the common driver behind the local OpenAI-compatible shim.

OpenEvolve evaluates in worker processes, so the evaluator is a thin HTTP client and every screen runs in this
(parent) process, serialized, on the one SearchSession.
"""
from __future__ import annotations

import ast
import json
import math
import pprint
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ...actions import load_schema, zones
from ...agent.search import SearchSession, SearchTrace
from ..claude_openai_shim import ShimServer
from ...agent.context import estimate_tokens
from ...agent.optimize import MIN_CALL_WALL_S, output_allowance
from ..common import new_session, screen_start

OPENEVOLVE_ROOT = Path(__file__).resolve().parents[3] / 'third_party_src' / 'openevolve'
INVALID_SCORE = -1.0
DEVIATIONS = [
    'Evaluator = InfraRSI harness + judge through the shared SearchSession (same screens/GPUh/wall/token budget, same validity rules) instead of a user evaluate() script.',
    'The evolved "program" is data (L0_CONFIG and OVERLAYS literals parsed with ast.literal_eval), never executed by the evaluator.',
    'L0 keys validated by the L0 schema and overlays by patch_guard (300-line strong limit, frozen zone) before any screen; rejected programs score -1 without a screen.',
    'LLM = claude-opus-5-5 through the local OpenAI-compatible shim over the tools-disabled Claude Code CLI; temperature/top_p/seed are ignored by the CLI.',
    'evaluator.max_retries = 0, cascade_evaluation = False, parallel_evaluations = 1 (a screen is a full training run; retries would double-spend GPU).',
    'iterations = remaining screen budget; evaluations after any budget dimension is exhausted return the invalid score without screening.',
    'combined_score = log(t_start / t) on the screen StepTime (higher is better); invalid candidates -1.',
    'L2 kernels closed; starting point = ED config with pristine overlays (same start as the agent).',
]


def build_program(task_id: str, config: dict, overlays: dict[str, str] | None = None) -> str:
    keys = sorted(load_schema()['keys'])
    header = (f'# InfraRSI candidate for task {task_id}. Edit only inside the EVOLVE block. This file is data: it is parsed,\n'
              f'# never executed. L0_CONFIG keys must be L0 schema keys ({len(keys)} keys) plus optional "env" (whitelist);\n'
              f'# OVERLAYS maps overlay file names {list(zones.MUTABLE_OVERLAYS)} to their complete new text.\n')
    return (header + '# EVOLVE-BLOCK-START\n'
            f'L0_CONFIG = {pprint.pformat(config, width=100, sort_dicts=True)}\n'
            f'OVERLAYS = {pprint.pformat(dict(overlays or {}), width=100)}\n'
            '# EVOLVE-BLOCK-END\n')


def parse_program(code: str) -> tuple[dict, dict]:
    tree = ast.parse(code)
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('L0_CONFIG', 'OVERLAYS'):
                found[node.targets[0].id] = ast.literal_eval(node.value)
    if 'L0_CONFIG' not in found:
        raise ValueError('L0_CONFIG literal not found')
    cfg, ov = found['L0_CONFIG'], found.get('OVERLAYS', {}) or {}
    if not isinstance(cfg, dict) or not isinstance(ov, dict) or not all(isinstance(v, str) for v in ov.values()):
        raise ValueError('L0_CONFIG must be a dict and OVERLAYS a {file: text} dict')
    return cfg, ov


class EvalService:
    """Serializes evaluations onto one SearchSession."""

    def __init__(self, session: SearchSession):
        self.session, self.lock = session, threading.Lock()
        self.log = []

    def evaluate(self, code: str) -> dict:
        with self.lock:
            s = self.session
            if s.stop_flag or s.ledger.exhausted():
                return {'combined_score': INVALID_SCORE, 'valid': 0.0, 'budget_exhausted': 1.0}
            try:
                cfg, ov = parse_program(code)
            except (ValueError, SyntaxError) as exc:
                s.count_invalid('program_unparseable')
                return self._out({'combined_score': INVALID_SCORE, 'valid': 0.0}, f'unparseable program: {exc}')
            fb = s.propose_config(config=cfg, base='c0', origin='openevolve')
            cid = fb.get('candidate_id') or fb.get('duplicate_of')
            if cid is None:
                return self._out({'combined_score': INVALID_SCORE, 'valid': 0.0}, f'L0 rejected: {fb.get("errors") or fb.get("error")}')
            if ov:
                pfb = s.propose_patch(files=ov, base=cid, origin='openevolve')
                cid = pfb.get('candidate_id') or pfb.get('duplicate_of')
                if cid is None:
                    return self._out({'combined_score': INVALID_SCORE, 'valid': 0.0}, f'patch rejected: {pfb.get("findings") or pfb.get("error")}')
            c = s.candidates[cid]
            if c.status == 'proposed':
                res = s.screen(cid)
                if 'error' in res:
                    return self._out({'combined_score': INVALID_SCORE, 'valid': 0.0, 'budget_exhausted': 1.0}, res['error'])
            v = c.verdict or {}
            t0 = (s.candidates['c0'].verdict or {}).get('step_time')
            if v.get('valid') and v.get('step_time') and t0:
                return self._out({'combined_score': math.log(t0 / v['step_time']), 'valid': 1.0, 'step_time': v['step_time']}, None)
            return self._out({'combined_score': INVALID_SCORE, 'valid': 0.0}, '; '.join(v.get('reasons', [])[:3]) or 'invalid')

    def _out(self, metrics: dict, error: str | None) -> dict:
        self.log.append({'metrics': metrics, 'error': error})
        if error:
            self.session.event('openevolve_eval', {'error': error[:500]})
        return metrics


class EvalServer:
    def __init__(self, service: EvalService, host: str = '127.0.0.1', port: int = 0):
        svc = service

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))).decode('utf-8'))
                raw = json.dumps(svc.evaluate(body.get('code', ''))).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f'http://{host}:{port}/evaluate'

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class RemoteEvaluator:
    """Picklable evaluate(program_path) for OpenEvolve workers; screens run in the parent EvalServer."""

    def __init__(self, url: str, timeout: float = 4 * 3600):
        self.url, self.timeout = url, timeout

    def __call__(self, program_path: str) -> dict:
        code = Path(program_path).read_text(encoding='utf-8')
        req = urllib.request.Request(self.url, data=json.dumps({'code': code}).encode('utf-8'),
                                     headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))


def system_message(task_raw: dict) -> str:
    keys = load_schema()['keys']
    l0 = '; '.join(f'{k}: {v.get("values", v.get("range", "bool" if v["type"] == "bool" else v["type"]))}' for k, v in keys.items())
    return (f'You optimize Megatron-Core 0.18.2 training throughput for task {task_raw["id"]} '
            f'({task_raw["model"]["repo_id"]}, {task_raw["data"]["key"]}, seq {task_raw["workload"]["seq_len"]}, '
            f'{task_raw["hardware"]["gpus"]} GPUs). The program is data: L0_CONFIG (Megatron-Core config keys) and OVERLAYS '
            f'(complete replacement texts of runtime/overlays/{{schedule,recompute_policy,comm_groups,state_placement,host_sync,data_path}}.py, '
            f'<= 300 changed lines in total, no imports of infrarsi modules, timers or receipts, no monkeypatching of timing, '
            f'precision, optimizer, LR, loss, router or collectives, no file or env writes). Score = log(t_start / StepTime) of a '
            f'fresh training screen; invalid candidates (changed batch, data, loss, precision, trainable set or out-of-tolerance '
            f'numerics) score -1. L0 keys and legal values: {l0}. Env whitelist: {sorted(load_schema()["env"])}.')


def run_openevolve(task, budget, start_config: dict, runner, *, llm_backend, judge=None, model_facts=None,
                   run_id: str = 'openevolve_L0L1', out_dir=None, start_receipt=None, reference_receipt=None, clock=None,
                   openevolve_root: Path = OPENEVOLVE_ROOT, seed: int = 42) -> SearchTrace:
    out = Path(out_dir) if out_dir else None
    session = new_session(task=task, budget=budget, runner=runner, judge=judge, run_id=run_id, arm='openevolve_L0L1',
                          out_dir=out, start_config=start_config, start_receipt=start_receipt,
                          reference_receipt=reference_receipt, model_facts=model_facts, clock=clock, layers=('L0', 'L1'))
    screen_start(session)
    raw = session.task_raw
    def admit(text: str) -> str | None:
        """Same gate as the agent: any exhausted dimension, an unaffordable input, or too little wall left."""
        ex = session.ledger.exhausted()
        if ex:
            return ex[0]
        if output_allowance(budget, session.ledger, llm_backend, text) is None:
            return 'llm_tokens'                         # input + overhead + reserved output must fit
        if budget.wall_seconds - session.ledger.wall() < MIN_CALL_WALL_S:
            return 'wall_seconds'
        return None

    shim = ShimServer(llm_backend, usage_cb=lambda res: session.charge_llm(res, 'openevolve'), budget_check=admit,
                      call_timeout=lambda: max(1.0, budget.wall_seconds - session.ledger.wall()),
                      max_output=lambda text: output_allowance(budget, session.ledger, llm_backend, text),
                      log_path=(out / 'shim_calls.jsonl') if out else None, calls_dir=(out / 'calls') if out else None).start()
    server = EvalServer(EvalService(session))
    if str(openevolve_root) not in sys.path:
        sys.path.insert(0, str(openevolve_root))
    try:
        from openevolve.api import run_evolution
        from openevolve.config import Config, LLMModelConfig
        config = Config()
        config.random_seed = seed
        config.llm.models = [LLMModelConfig(name=shim.served_name, api_base=shim.url, api_key='infrarsi-local', weight=1.0,
                                            max_tokens=16000, timeout=1800, retries=1, retry_delay=5)]
        config.llm.evaluator_models = []
        config.llm.api_base = shim.url
        config.prompt.system_message = system_message(raw)
        config.evaluator.max_retries = 0
        config.evaluator.cascade_evaluation = False
        config.evaluator.parallel_evaluations = 1
        config.evaluator.use_llm_feedback = False
        config.evaluator.timeout = int(budget.wall_seconds)
        config.max_code_length = 200000
        iterations = max(0, budget.screens - session.ledger.used['screens'])
        code = build_program(raw['id'], session.candidates['c0'].config, {})
        run_evolution(initial_program=code, evaluator=RemoteEvaluator(server.url), config=config, iterations=iterations,
                      output_dir=str(out / 'openevolve') if out else None, cleanup=out is None)
        stop = session.stop_flag or ('budget:' + session.ledger.exhausted()[0] if session.ledger.exhausted() else 'iterations')
    finally:
        server.stop()
        shim.stop()
    return session.finalize(stop_reason=stop, extra={'deviations': DEVIATIONS, 'openevolve_root': str(openevolve_root),
                                                     'shim_calls': shim.calls})
