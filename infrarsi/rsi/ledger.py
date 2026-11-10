"""Cost books: LLM (tokens, USD) and GPU (GPU-hours by device type) per run, arm, task and phase.

Append-only JSONL. The R arm's development cost (searches + reflections) is reported all-in; held-out comparisons
match deployment budgets only (FINAL_PLAN 3.6). Global LLM caps: USD 1,500 total; stop P1 LLM runs at USD 1,200.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from collections import defaultdict
from pathlib import Path

LLM_USD_CAP = 1500.0
LLM_USD_STOP_P1 = 1200.0
BOOKS = ('llm', 'gpu')
PHASES = ('search', 'reflect', 'scoring', 'design', 'verifier', 'probe', 'lifecycle', 'other')


class CostLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, *, book: str, run_id: str, arm: str, task_id: str | None, phase: str, llm_tokens: int = 0,
            llm_usd: float = 0.0, model: str | None = None, gpu_hours: float = 0.0, gpu_type: str | None = None,
            wall_seconds: float = 0.0, accounting_complete: bool = True, source: str = '') -> dict:
        if book not in BOOKS or phase not in PHASES:
            raise ValueError(f'book={book!r} phase={phase!r}')
        for name, v in (('llm_tokens', llm_tokens), ('llm_usd', llm_usd), ('gpu_hours', gpu_hours), ('wall_seconds', wall_seconds)):
            if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
                raise ValueError(f'{name}={v!r}')
        row = {'utc': _dt.datetime.now(_dt.timezone.utc).isoformat(), 'book': book, 'run_id': run_id, 'arm': arm,
               'task_id': task_id, 'phase': phase, 'llm_tokens': int(llm_tokens), 'llm_usd': float(llm_usd), 'model': model,
               'gpu_hours': float(gpu_hours), 'gpu_type': gpu_type, 'wall_seconds': float(wall_seconds),
               'accounting_complete': bool(accounting_complete), 'source': source}
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        return row

    def add_trace(self, trace: dict, *, gpu_type: str = 'H100-80GB') -> list[dict]:
        """Book a finished SearchTrace: one LLM row and one GPU row."""
        u = trace['used']
        common = dict(run_id=trace['run_id'], arm=trace['arm'], task_id=trace['task_id'], phase='search',
                      accounting_complete=trace['accounting_complete'], source='trace')
        models = sorted({c.get('model') for c in trace.get('llm_calls', []) if c.get('model')})
        return [self.add(book='llm', llm_tokens=u['llm_tokens'], llm_usd=u['llm_usd'], model=','.join(models) or None,
                         wall_seconds=u['wall_seconds'], **common),
                self.add(book='gpu', gpu_hours=u['gpu_hours'], gpu_type=gpu_type, **common)]

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines() if line.strip()]

    def totals(self, by: tuple[str, ...] = ('book',)) -> dict:
        out = defaultdict(lambda: {'llm_tokens': 0, 'llm_usd': 0.0, 'gpu_hours': 0.0, 'wall_seconds': 0.0, 'rows': 0, 'accounting_complete': True})
        for r in self.rows():
            k = tuple(r.get(x) for x in by)
            t = out[k]
            for f in ('llm_tokens', 'llm_usd', 'gpu_hours', 'wall_seconds'):
                t[f] += r[f]
            t['rows'] += 1
            t['accounting_complete'] &= r['accounting_complete']
        return {'|'.join(map(str, k)): v for k, v in out.items()}

    def llm_usd_total(self) -> float:
        return sum(r['llm_usd'] for r in self.rows() if r['book'] == 'llm')

    def check_caps(self, priority: str = 'P0') -> str | None:
        """None if LLM spending may continue for runs of this priority, else the reason to stop."""
        usd = self.llm_usd_total()
        if usd >= LLM_USD_CAP:
            return f'LLM cost {usd:.2f} >= cap {LLM_USD_CAP}'
        if priority.startswith('P1') and usd >= LLM_USD_STOP_P1:
            return f'LLM cost {usd:.2f} >= P1 stop {LLM_USD_STOP_P1}'
        return None
