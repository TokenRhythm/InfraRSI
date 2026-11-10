"""E2 baseline arms on the dispatcher: random_L0, tpe_L0 (config track, no LLM), optico_style_L0 (Claude), and
openevolve_L0L1 (Claude through the OpenAI-compatible shim). Same ED start, task search budget, formal judge,
HarnessScreenRunner and locked scoring pair as the agent (infrarsi.agent.arms). The external-bench F1 fallback
adds F-PROMPTS (baselines/prompts_oneshot.py): one recommendation call, three screens, the locked pair, no lifecycle."""
from __future__ import annotations

from pathlib import Path

from ..agent import arms as A
from ..agent.harness_runner import HarnessScreenRunner


def _search(kind: str):
    def search_fn(*, run, task, nodes, start_config, start_receipt, reference_receipt, runner, out_dir, ctx) -> dict:
        out_dir = Path(out_dir)
        refs = ctx.store.references().get(task.id, {})
        mf = A.model_facts_for(task, ctx.remote)
        budget = A.budget_for(task, run)                 # a zero-token row cap forbids LLM calls, not screens
        ref_path, reference_receipt, _ = A.numerics_reference(refs, reference_receipt, start_receipt)
        hr = HarnessScreenRunner(runner, nodes, reference_receipt_path=ref_path, reference_receipt=reference_receipt,
                                 wall_seconds_cap=A.attempt_cap(run), log=ctx.log)
        common = dict(judge=A.formal_judge(task), model_facts=mf, run_id=run['run_id'], out_dir=out_dir,
                      start_receipt=start_receipt, reference_receipt=reference_receipt)
        seed = int(run.get('seed') or 42)
        if kind == 'random_L0':
            from .random_l0 import run_random
            trace = run_random(task, budget, start_config, hr, seed=seed, **common)
        elif kind == 'tpe_L0':
            from .tpe_l0 import run_tpe
            trace = run_tpe(task, budget, start_config, hr, seed=seed, **common)
        elif kind == 'optico_style_L0':
            from .optico_style import run_optico_style
            trace = run_optico_style(task, budget, start_config, hr, llm_backend=A.DRIVERS['strong'](), **common)
        else:
            from .openevolve_adapter import run_openevolve
            trace = run_openevolve(task, budget, start_config, hr, llm_backend=A.DRIVERS['strong'](), seed=seed, **common)
        trace = trace.to_dict()
        result = {'arm': run['arm'], 'task': task.id, 'trace': A.trace_summary(trace)}
        if kind == 'optico_style_L0':
            ex = trace.get('extra') or {}
            result['optico'] = {k: ex.get(k) for k in ('n_refusals', 'prompt_transform', 'llm_calls', 'pool_rows')}
            if trace['stop_reason'] == 'refusals':      # D-OC-15 not enough (AMENDMENT_EXT_001 Q5): keep the evidence,
                msg = (f'{run["run_id"]}: {ex.get("n_refusals")} LLM refusals despite D-OC-15 strip_think_format; search '
                       f'stopped (stop_reason refusals), not scored; escalate (AMENDMENT_EXT_001 Q5). Trace in {out_dir}')
                A._write_json(out_dir / 'result.json', result)     # then fail the run loudly, as endpoint_unreachable
                ctx.log('ESCALATE ' + msg)
                raise RuntimeError(msg)
        ed_score, ed_problems = A.load_ed_score(task, refs, ctx, nodes)
        return A.finish(result, task, run, hr, refs, trace, out_dir, ed_score=ed_score, ed_problems=ed_problems,
                        start_config=start_config)
    search_fn.__name__ = f'baseline_{kind}'
    return search_fn


ARMS = ('random_L0', 'tpe_L0', 'optico_style_L0', 'openevolve_L0L1')


def dispatch_handlers() -> dict:
    from .prompts_oneshot import ARM as PROMPTS_ARM, prompts_search
    out = {arm: A._search_handler(_search(arm), 'ED') for arm in ARMS}
    out[PROMPTS_ARM] = A._search_handler(prompts_search, 'ED')     # start slot = the task start (U+C on F1)
    return out
