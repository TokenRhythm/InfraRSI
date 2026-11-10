"""A local OpenAI-compatible endpoint for the system tests: POST /v1/chat/completions, POST /v1/responses, GET /v1/models.

    with FakeLLM(script) as llm:                     # llm.base_url = http://127.0.0.1:<port>/v1
        driver = make_driver({'backend': 'openai-compat', 'model': llm.model, 'base_url': llm.base_url})

`script(req)` is called per request with req = {'n', 'path', 'headers', 'body', 'system', 'prompt'} and returns:
    a dict          the JSON object the model answers with (a normal reply, finish_reason stop)
    a str           the raw text of the reply (prose, broken JSON, ...)
    Reply(...)      an HTTP status with a hand-built payload (errors, odd shapes)
    Cut(obj)        the object's text cut in half, finish_reason 'length' (a reasoning model at its output cap)
    Delay(s, then)  answer `then` after `s` seconds (a slow model: the call's time budget)
    Served(obj, m)  the reply with another served model name
Every request is recorded in `llm.requests`. `by_role(...)` builds a script that answers InfraRSI's own prompts.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass
class Reply:
    status: int
    payload: dict


@dataclass
class Cut:
    obj: object


@dataclass
class Delay:
    seconds: float
    then: object = None


@dataclass
class Served:
    obj: object
    model: str


def _usage(body: dict, text: str, api: str) -> dict:
    n_in = len(json.dumps(body.get('messages') or body.get('input') or '')) // 4
    n_out = max(1, len(text) // 4)
    if api == 'responses':
        return {'input_tokens': n_in, 'output_tokens': n_out, 'total_tokens': n_in + n_out}
    return {'prompt_tokens': n_in, 'completion_tokens': n_out, 'total_tokens': n_in + n_out}


class FakeLLM:
    def __init__(self, script=None, *, model: str = 'fake-model'):
        self.model = model
        self.script = script or (lambda req: {'ok': True})
        self.requests: list[dict] = []
        self._server = None
        self._thread = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------ lifecycle
    def start(self) -> 'FakeLLM':
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *a):                   # quiet
                pass

            def _send(self, status: int, payload: dict):
                data = json.dumps(payload).encode('utf-8')
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.endswith('/models'):
                    return self._send(200, {'object': 'list', 'data': [{'id': owner.model, 'object': 'model'}]})
                self._send(404, {'error': {'message': f'no route {self.path}'}})

            def do_POST(self):
                n = int(self.headers.get('Content-Length') or 0)
                raw = self.rfile.read(n) if n else b''
                try:
                    body = json.loads(raw.decode('utf-8'))
                except ValueError:
                    return self._send(400, {'error': {'message': 'bad json'}})
                api = 'responses' if self.path.endswith('/responses') else 'chat'
                if api == 'responses':
                    system = body.get('instructions') or ''
                    inputs = body.get('input') or []
                    prompt = inputs[-1]['content'] if inputs else ''
                else:
                    msgs = body.get('messages') or []
                    system = next((m['content'] for m in msgs if m.get('role') == 'system'), '')
                    prompt = next((m['content'] for m in reversed(msgs) if m.get('role') == 'user'), '')
                with owner._lock:
                    req = {'n': len(owner.requests), 'path': self.path, 'headers': dict(self.headers), 'body': body,
                           'system': system, 'prompt': prompt, 'api': api}
                    owner.requests.append(req)
                reply = owner.script(req)
                status, payload = owner.render(reply, body, api)
                self._send(status, payload)

        self._server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.stop()

    @property
    def base_url(self) -> str:
        return f'http://127.0.0.1:{self._server.server_address[1]}/v1'

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    # ------------------------------------------------------------ payloads
    def render(self, reply, body: dict, api: str, model: str | None = None) -> tuple[int, dict]:
        model = model or self.model
        if isinstance(reply, Delay):
            time.sleep(reply.seconds)
            return self.render(reply.then if reply.then is not None else {'ok': True}, body, api, model)
        if isinstance(reply, Reply):
            return reply.status, reply.payload
        if isinstance(reply, Served):
            return self.render(reply.obj, body, api, reply.model)
        cut = isinstance(reply, Cut)
        obj = reply.obj if cut else reply
        text = obj if isinstance(obj, str) else json.dumps(obj)
        if cut:
            text = text[:max(1, len(text) // 2)]
        usage = _usage(body, text, api)
        if api == 'responses':
            payload = {'id': 'resp_1', 'object': 'response', 'model': model, 'status': 'incomplete' if cut else 'completed',
                       'output': [{'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': text}]}],
                       'usage': usage}
            if cut:
                payload['incomplete_details'] = {'reason': 'max_output_tokens'}
            return 200, payload
        return 200, {'id': 'chatcmpl-1', 'object': 'chat.completion', 'model': model,
                     'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text},
                                  'finish_reason': 'length' if cut else 'stop'}], 'usage': usage}


# ---------------------------------------------------------------- InfraRSI's own prompts
PREFLIGHT = 'Reply with the JSON object {"ok": true}'
TRANSFER = 'You judge how measured training-speed effects transfer'
REVIEW = 'You review the code of one training job'
HYPOTHESES = 'You generate optimization hypotheses'
MENU = 'rate'                                            # the menu prompt of capability < 0.4 (rated moves)


def role_of(req: dict) -> str:
    s, p = req.get('system') or '', req.get('prompt') or ''
    if PREFLIGHT in p:
        return 'preflight'
    if s.startswith(TRANSFER):
        return 'transfer'
    if s.startswith(REVIEW):
        return 'code_review'
    if s.startswith(HYPOTHESES):
        return 'hypotheses'
    return 'other'


def hypothesis(change: dict, gain_pct: float, *, mechanism: str = 'test', p_feasible: float = 0.9, patch=None, edits=None,
               change_class: str = 'execution', uncertainty_pct: float = 5.0) -> dict:
    return {'mechanism': mechanism, 'change': dict(change), 'patch': patch, 'edits': edits, 'predicted_gain_pct': gain_pct,
            'uncertainty_pct': uncertainty_pct, 'p_feasible': p_feasible, 'change_class': change_class,
            'rationale': f'{mechanism}: scripted'}


def by_role(*, hypotheses=None, review=None, transfer=None, other=None):
    """A script answering by role: `hypotheses` is a list of hypothesis lists (one per call, the last one repeated;
    [] ends the search's asking) or a callable(req) -> list; review / transfer / other likewise (dicts or callables)."""
    state = {'h': 0}

    def script(req):
        role = role_of(req)
        if role == 'preflight':
            return {'ok': True}
        if role == 'hypotheses':
            if callable(hypotheses):
                return {'hypotheses': hypotheses(req)}
            hs = hypotheses or []
            batch = hs[min(state['h'], len(hs) - 1)] if hs else []
            state['h'] += 1
            return {'hypotheses': batch}
        if role == 'code_review':
            if callable(review):
                return review(req)
            return review or {'verdict': 'no_patch', 'summary': 'scripted: nothing that only code can remove', 'findings': []}
        if role == 'transfer':
            if callable(transfer):
                return transfer(req)
            return transfer or {'assessment': [], 'start_config': {}, 'memory_note': 'scripted'}
        if callable(other):
            return other(req)
        return other or {'ok': True}
    return script
