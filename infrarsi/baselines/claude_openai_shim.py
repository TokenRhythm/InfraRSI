"""Local OpenAI-compatible endpoint that forwards to the Claude CLI backend (or any agent backend) with accounting.

Lets OpenAI-client tools (OpenEvolve) use the same driver as every other arm: the same tools-disabled, fresh-session
Claude Code CLI call, the same model check and the same cost books. Sampling parameters (temperature, top_p, seed)
cannot be set through the CLI and are recorded but ignored.

  POST /v1/chat/completions   {model, messages, ...} -> OpenAI chat.completion (usage from the CLI result)
  GET  /v1/models
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def collapse_messages(messages: list[dict]) -> tuple[str, str]:
    """(system prompt, transcript prompt) from OpenAI messages; history is replayed explicitly in one prompt."""
    system = '\n\n'.join(str(m.get('content', '')) for m in messages if m.get('role') == 'system')
    turns = [m for m in messages if m.get('role') != 'system']
    if len(turns) == 1 and turns[0].get('role') == 'user':
        return system, str(turns[0].get('content', ''))
    return system, '\n\n'.join(f'[{m.get("role", "user")}]\n{m.get("content", "")}' for m in turns)


class ShimServer:
    def __init__(self, backend, *, host: str = '127.0.0.1', port: int = 0, usage_cb=None, budget_check=None,
                 log_path: str | Path | None = None, calls_dir: str | Path | None = None, served_name: str | None = None,
                 call_timeout=None, max_output=None):
        """budget_check(request_text) -> limiting dimension or None; call_timeout() -> seconds for the next call.
        Admission, the backend call and the usage callback run under one lock, so concurrent requests cannot pass
        admission before earlier usage is charged."""
        self.backend, self.usage_cb, self.budget_check = backend, usage_cb, budget_check
        self.call_timeout = call_timeout
        self.max_output = max_output              # (request text) -> output tokens the call may produce
        self.served_name = served_name or getattr(backend, 'model', 'claude')
        self.log_path = Path(log_path) if log_path else None
        self.calls_dir = Path(calls_dir) if calls_dir else None
        self.lock = threading.Lock()
        self.calls = 0
        shim = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):                      # quiet
                pass

            def _send(self, code: int, obj: dict):
                raw = json.dumps(obj).encode('utf-8')
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path.rstrip('/').endswith('/models'):
                    return self._send(200, {'object': 'list', 'data': [{'id': shim.served_name, 'object': 'model', 'owned_by': 'infrarsi-shim'}]})
                return self._send(404, {'error': {'message': 'not found'}})

            def do_POST(self):
                if not self.path.rstrip('/').endswith('/chat/completions'):
                    return self._send(404, {'error': {'message': 'not found'}})
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))).decode('utf-8'))
                except ValueError:
                    return self._send(400, {'error': {'message': 'invalid JSON body'}})
                code, obj = shim.handle(body)
                return self._send(code, obj)

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.thread = None

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f'http://{host}:{port}/v1'

    def start(self) -> 'ShimServer':
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def handle(self, body: dict) -> tuple[int, dict]:
        messages = body.get('messages') or []
        if not isinstance(messages, list) or not messages:
            return 400, {'error': {'message': 'messages required'}}
        system, prompt = collapse_messages(messages)
        with self.lock:
            if self.budget_check:
                reason = self.budget_check((system or '') + ' ' + prompt)
                if reason:
                    return 429, {'error': {'message': f'budget exhausted: {reason}', 'type': 'budget'}}
            self.calls += 1
            n = self.calls
            call_dir = (self.calls_dir / f'{n:04d}') if self.calls_dir else None
            kw = {'timeout': self.call_timeout()} if self.call_timeout else {}
            if self.max_output:
                cap = self.max_output((system or '') + ' ' + prompt)
                if cap is None:
                    return 429, {'error': {'message': 'budget exhausted: llm_tokens', 'type': 'budget'}}
                kw['max_output_tokens'] = cap
            res = self.backend.complete(system or 'You are a helpful assistant.', prompt, call_dir=call_dir, role='shim',
                                        expect_json=False, **kw)
            if self.usage_cb:
                self.usage_cb(res)
        bd = res.meta.get('token_breakdown') or {}
        completion = bd.get('output_tokens', 0) if bd else 0
        total = res.llm_tokens or 0
        row = {'utc': time.time(), 'call': n, 'requested_model': body.get('model'), 'served': res.model,
               'llm_tokens': res.llm_tokens, 'llm_usd': res.llm_usd, 'accounting_complete': res.accounting_complete,
               'error': res.error, 'ignored_params': {k: body[k] for k in ('temperature', 'top_p', 'seed', 'max_tokens') if k in body}}
        if self.log_path:
            with self.log_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(row) + '\n')
        if res.error or res.text is None:
            return 502, {'error': {'message': f'backend error: {res.error}', 'type': 'backend'}}
        return 200, {'id': 'chatcmpl-' + uuid.uuid4().hex[:24], 'object': 'chat.completion', 'created': int(time.time()),
                     'model': res.model, 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': res.text},
                                                      'finish_reason': 'stop'}],
                     'usage': {'prompt_tokens': max(0, total - completion), 'completion_tokens': completion, 'total_tokens': total}}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='OpenAI-compatible shim over the Claude CLI backend')
    ap.add_argument('--model', default='claude-opus-5-5')
    ap.add_argument('--effort', default='high')
    ap.add_argument('--port', type=int, default=18100)
    ap.add_argument('--log', type=Path, default=Path('shim_calls.jsonl'))
    ap.add_argument('--calls-dir', type=Path, default=None)
    args = ap.parse_args(argv)
    from ..agent.backends.claude_cli import ClaudeCLIBackend
    shim = ShimServer(ClaudeCLIBackend(model=args.model, effort=args.effort), port=args.port, log_path=args.log, calls_dir=args.calls_dir)
    print(f'serving {shim.url}', flush=True)
    shim.httpd.serve_forever()


if __name__ == '__main__':
    main()
