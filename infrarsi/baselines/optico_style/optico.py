"""optico_style_L0 (design E2, config track): OptiCo's four agents and message pool on the InfraRSI harness.

OptiCo-public (github.com/TangZhe96/OptiCo-public, commit recorded in FETCH_RECORD.json) has no LICENSE: it is run
and cited, never redistributed. Its prompt texts are therefore read at run time from third_party_src/OptiCo-public
(prompt/*.txt and the ChangePrompt wrappers of script/analyzer/analyzer.py, extracted with ast, never executed);
only the substitutions listed in DEVIATIONS are ours. The same loop runs on an external environment's hooks
(DEVIATIONS_HOOKS) and, as the strategy `optico` of `infrarsi run` (search/optico.py), on a project manifest's knobs
through ProjectHooks (DEVIATIONS_PROJECT; its prompts from INFRARSI_OPTICO_ROOT when that is set).

Agents: monitor (our harness screen -> one pool row), loc_error (LLM, on failure), give_sugg (LLM with the CoT
file), change_para (LLM -> "KEY: VALUE" lines). Message pool = the full performance table (OptiCo default).

D-OC-15 (F-OPTICO and B-NAT, DECISIONS_20261021 D7, AMENDMENT_EXT_001 Q5): every request goes out through
ext.optico.prompt_variants.strip_think_format, the drop-only rule the B-NAT shim applies; refusals are counted, a
refused request is re-sent unchanged as analyzer.api_infer does, and the third refusal in a run stops it.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import time
from pathlib import Path

from ...actions import l0_space, validate_l0
from ...agent.backends.base import is_refusal
from ...agent.search import SearchTrace
from ...agent.context import estimate_tokens
from ...agent.optimize import MIN_CALL_WALL_S, call_allowances, max_output_tokens, output_allowance
from ...ext.optico.prompt_variants import strip_think_format
from ..common import new_session, screen_start

OPTICO_ROOT = Path(__file__).resolve().parents[3] / 'third_party_src' / 'OptiCo-public'
SYSTEM = 'You are a reasoning assistant.'          # OptiCo ChangePrompt.raw() prefix; the CLI transport needs a system prompt
H100_PEAK_BF16 = 989e12
PROMPT_TRANSFORM = 'strip_think_format'            # D-OC-15, the same function as B-NAT's shim (no other rewording)
MAX_REFUSALS = 3                                   # AMENDMENT_EXT_001 Q5: the third refusal in a run stops it (escalate)
RETRY_WAIT_S = 10.0                                # analyzer.api_infer(wait_time=10) before a failed request is re-sent
_sleep = time.sleep

# OptiCo shell parameter -> L0 key (AC and AC_LAYERS map onto recompute / activation offload, see to_config()).
OPTICO_KEYS = {'MICRO_BATCH_SIZE': 'micro_batch_size', 'TP': 'tensor_model_parallel_size', 'PP': 'pipeline_model_parallel_size',
               'EP': 'expert_model_parallel_size', 'OVERLAP_GRAD_REDUCE': 'overlap_grad_reduce',
               'OVERLAP_PARAM_GATHER': 'overlap_param_gather', 'FUSED_CROSS_ENTROPY': 'cross_entropy_loss_fusion'}
DEVIATIONS = [
    'Training launcher: OptiCo run_*.sh/Pai-Megatron-Patch replaced by the InfraRSI harness (fresh process, fixed input plan, receipts); Megatron-Core 0.18.2 + mcore-bridge.',
    'Parameter set: the same L0 space as every config-track arm. OptiCo names kept for its parameters (MICRO_BATCH_SIZE, TP, PP, EP, OVERLAP_GRAD_REDUCE, OVERLAP_PARAM_GATHER, AC, AC_LAYERS, FUSED_CROSS_ENTROPY); all other L0 keys and the env whitelist appended under their Megatron-Core / env names.',
    'FIRST_PP_NUM_LAYERS / LAST_PP_NUM_LAYERS dropped: uneven pipeline layouts are not in L0.',
    'AC mapping: none -> no recompute; sel -> selective (recompute_modules core_attn); full -> full/uniform with AC_LAYERS; offload -> activation cpu_offloading with AC_LAYERS layers.',
    'Performance table columns: TFLOPS/GpuUtil are not measured by the harness; StepTime_s, TokensPerSec, MFU (6*N_active*tokens/(StepTime*GPUs*989 TFLOP/s) when model facts allow, else NULL), MemMaxGiB and MemUsage% (peak reserved / 80 GiB) are reported; GpuUtilAve is NULL.',
    'Objective: OptiCo maximizes MFU; at fixed work this equals minimizing StepTime, which the harness measures.',
    'Start point: the task ED config (same start as every arm), not the OptiCo default scripts.',
    'LLM: claude-opus-5-5 through the common tools-disabled Claude Code CLI transport (OptiCo used gpt-5.2/o3/GLM-4.6 via an OpenAI client, temperature 0.7, max_tokens 8192); temperature is not settable; a one-line system prompt is added.',
    'Iterations: bounded by the shared budget (16 screens, 1.5M tokens, wall and GPU-hours) instead of NUMBER_OF_CYCLES.',
    'Configs rejected by the L0 validator are not run; they enter the pool as failed rows with the validator message (no screen spent), as for every arm.',
    'RETRY verdicts from loc_error do not re-run the identical config (screens are deterministic fresh processes and the session forbids duplicate screens); the cause is recorded and the loop continues with give_sugg.',
    'Exact-duplicate parameter sets are not re-screened; they enter the pool as rows marked duplicate.',
    'D-OC-15 strip_think_format (DECISIONS_20261021 D7, AMENDMENT_EXT_001 Q5; the B-NAT rule): claude-opus-5-5 refuses OptiCo\'s give_sugg prompt verbatim ([reasoning_extraction]: it demands the step-by-step reasoning inside <think> tags), so before every LLM call the lines that demand <think>-tagged reasoning are dropped (ext.optico.prompt_variants.strip_think_format; drop-only, no rewording; the parsers already take split("</think>")[-1]). Each call records the lines dropped and the SHA-256 of the original and of the sent prompt. A refused call is re-sent unchanged 10 s later, as analyzer.api_infer re-sends a failed request; refusals are counted (n_refusals) and the third in a run stops it (stop_reason refusals, escalated).',
]

# With hooks (an external environment, e.g. ext/flashnext F-OPTICO): these replace the harness / L0 items above; the
# LLM, iteration, RETRY, duplicate and D-OC-15 items of DEVIATIONS are kept as they are.
DEVIATIONS_HOOKS = [
    "Training launcher: OptiCo run_*.sh/Pai-Megatron-Patch replaced by the external environment's own runner and judge "
    "(fresh process per screen, signed ext receipts).",
    "Parameter set: the environment's switch / knob table (the codec of its hooks) is the OptiCo parameter table; "
    "OptiCo's FIXED_PARA block carries the environment's fixed facts and the parameter domain, because OptiCo's prompts "
    "name no domain for parameters it does not know.",
    "Performance table columns: StepTime_s from the environment judge, TokensPerSec = physical tokens per update / "
    "StepTime, MemMaxGiB and MemUsage% from the receipt's peak reserved memory when it carries one; MFU and GpuUtilAve "
    "are NULL.",
    'Objective: OptiCo maximizes MFU; at fixed work this equals minimizing StepTime, which the environment measures.',
    "Start point: the environment's start (same for every arm), screened before the arm; its receipt is the first pool row.",
    "Configs rejected by the environment's validator are not run; they enter the pool as failed rows with the validator "
    "message (no screen spent), as for every arm.",
    "Iterations: bounded by the arm's shared budget (screens, LLM tokens and USD, wall and GPU-hours of the task) "
    "instead of NUMBER_OF_CYCLES.",
] + [d for d in DEVIATIONS if d.startswith(('LLM:', 'RETRY', 'Exact-duplicate', 'D-OC-15 '))]

# The project strategy optico (search/optico.py: `infrarsi run --strategy optico`, or search.strategy: optico, on a
# project manifest, with ProjectHooks): the items of DEVIATIONS_HOOKS written for an external environment are replaced;
# the objective and the RETRY, duplicate and D-OC-15 items are kept as they are.
DEVIATIONS_PROJECT = [
    "Training launcher: OptiCo run_*.sh/Pai-Megatron-Patch replaced by the project's own launch command, run by the "
    "InfraRSI project runner (a fresh process per run, signed receipts) and judged by the project verifier (the "
    "correctness checklist against the reference run).",
    "Parameter set: the manifest's knobs are the OptiCo parameter table (KnobCodec: each value written as the knob's "
    "choices are listed - the empty string as '', a comma as / - and read back onto them); OptiCo's FIXED_PARA block "
    "(ProjectHooks.fixed_parameters_text) carries the job's fixed facts from the manifest, the workload invariants, "
    "constraints and loss tolerance every run is held to, and the parameter domain, because OptiCo's prompts name no "
    "domain for parameters it does not know.",
    "Performance table columns: StepTime_s from the project verifier; TokensPerSec = the manifest's "
    "workload.global_batch_tokens / StepTime (NULL without it, never derived); MemMaxGiB = the peak memory the run's "
    "signals report and MemUsage% = MemMaxGiB / hardware.memory_gib (NULL without them); MFU and GpuUtilAve are NULL.",
    'Objective: OptiCo maximizes MFU; at fixed work this equals minimizing StepTime, which the project verifier measures.',
    "Start point: the flow's reference run of the user's current configuration (the start of every strategy, outside "
    "the search budget); its receipt is the first pool row and no screen is spent on it.",
    "Configs rejected by the project's validator (knob schema, constraints, workload invariants) are not run; they enter "
    "the pool as failed rows with the validator message (no screen spent), as for every strategy.",
    "Iterations: bounded by the run's search budget (search.budget or --screens: full runs, LLM tokens and USD, wall and "
    "GPU-hours), the same as every strategy's, instead of NUMBER_OF_CYCLES: each request (a refusal's re-send too) is "
    "admitted as the search proposer's calls are - its answer cap and the tokens, USD and wall time left passed to the "
    f"backend, none started in the last {MIN_CALL_WALL_S:g} s of the wall budget - and the run stops when one does not "
    "fit; six proposals in a row refused or repeating a row stop it (no_new_config), as do three refusals (refusals: "
    "reported in the run log and REPORT, not escalated).",
    "Recommendation: OptiCo's best row is not adopted as it stands: the flow's promotion re-runs the fastest valid "
    "candidates back to back with the reference (the locked pair, then the resume, eval and test checks the manifest "
    "configures), as for every strategy.",
    "LLM: the run's model (manifest llm block or --llm), which must answer in free text (claude-cli, or an "
    "OpenAI-compatible endpoint with llm.structured none), instead of OptiCo's gpt-5.2/o3/GLM-4.6 through an OpenAI "
    "client at temperature 0.7, max_tokens 8192; temperature is the backend's; a one-line system prompt is added; every "
    "call is kept under the run directory's llm_calls/.",
] + [d for d in DEVIATIONS if d.startswith(('RETRY', 'Exact-duplicate', 'D-OC-15 '))]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _sha_text(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def _wait(clock, seconds: float) -> None:
    """Wall time passing before a re-send: a test's VirtualClock is advanced, a real run sleeps."""
    if hasattr(clock, 'advance'):
        clock.advance(seconds)
    else:
        _sleep(seconds)


def _declined(session, driver, text: str) -> str:
    """The budget that left no room for a request agent.optimize.call_allowances() declined (in its order of checks)."""
    b, led = session.ledger.budget, session.ledger
    if output_allowance(b, led, driver, text) is None:
        return 'budget:llm_tokens'
    return 'budget:wall_seconds' if float(b.wall_seconds) - led.wall() < MIN_CALL_WALL_S else 'budget:llm_usd'


def _joined(node) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out = []
        for part in node.values:
            if isinstance(part, ast.Constant):
                out.append(part.value)
            else:
                out.append('{' + ast.unparse(part.value) + '}')
        return ''.join(out)
    raise ValueError(f'unexpected node {type(node).__name__}')


def load_materials(root: str | Path = OPTICO_ROOT) -> dict:
    root = Path(root)
    files = {k: root / 'prompt' / f'{k}.txt' for k in ('give_sugg', 'give_sugg_cot', 'change_para', 'loc_error')}
    analyzer = root / 'script' / 'analyzer' / 'analyzer.py'
    texts = {k: p.read_text(encoding='utf-8') for k, p in files.items()}
    tree = ast.parse(analyzer.read_text(encoding='utf-8'))
    wrappers = {}
    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ChangePrompt']:
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('give_sugg', 'change_para', 'loc_error')]:
            parts = {}
            for st in ast.walk(fn):
                if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name) \
                        and st.targets[0].id in ('instruction', 'user_prompt'):
                    parts[st.targets[0].id] = _joined(st.value)
            wrappers[fn.name] = parts
    for k in ('give_sugg', 'change_para', 'loc_error'):
        if set(wrappers.get(k, {})) != {'instruction', 'user_prompt'}:
            raise ValueError(f'OptiCo analyzer wrapper {k} not found; the fetched commit changed')
    provenance = {str(p.relative_to(root)).replace('\\', '/'): _sha(p) for p in [*files.values(), analyzer]}
    return {'texts': texts, 'wrappers': wrappers, 'provenance': provenance}


def compose(materials: dict, kind: str, input_text: str) -> str:
    """ChangePrompt(<kind>) exactly as analyzer.py builds it, with {self.input_text} filled."""
    w = materials['wrappers'][kind]
    user = w['user_prompt'].replace('{self.input_text}', input_text)
    if kind == 'give_sugg':
        return w['instruction'] + materials['texts']['give_sugg_cot'] + '\n\n' + user
    return w['instruction'] + user


def _fmt(v) -> str:
    if v is None:
        return 'None'
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, list):
        return json.dumps(v)
    return str(v)


def _parse(v: str):
    s = v.strip().strip('"').strip("'")
    low = s.lower()
    if low in ('none', 'null', ''):
        return None
    if low in ('true', 'false'):
        return low == 'true'
    if s.startswith('['):
        try:
            return json.loads(s.replace("'", '"'))
        except ValueError:
            return s
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


class ParamCodec:
    """L0 config <-> OptiCo-style 'KEY: VALUE' parameter file."""

    def __init__(self, task, model_facts):
        self.moe = bool((model_facts or {}).get('num_experts'))
        space = l0_space(task, model_facts, include_env=True)
        mapped = set(OPTICO_KEYS.values()) | {'recompute_granularity', 'recompute_method', 'recompute_num_layers', 'recompute_modules',
                                              'cpu_offloading', 'cpu_offloading_num_layers', 'moe_layer_recompute'}
        self.extra = [k for k in space if not k.startswith('env:') and k not in mapped]
        self.env = [k[4:] for k in space if k.startswith('env:')]

    def names(self) -> list[str]:
        base = ['MICRO_BATCH_SIZE', 'TP', 'PP'] + (['EP'] if self.moe else []) + ['OVERLAP_GRAD_REDUCE', 'OVERLAP_PARAM_GATHER',
                                                                                 'AC', 'AC_LAYERS', 'FUSED_CROSS_ENTROPY']
        return base + self.extra + [f'env:{e}' for e in self.env]

    def to_lines(self, config: dict) -> str:
        c = validate_l0(config).resolved
        gran = c['recompute_granularity']
        ac = 'offload' if c['cpu_offloading'] else {'full': 'full', 'selective': 'sel'}.get(gran, 'none')
        ac_layers = c['recompute_num_layers'] if ac == 'full' else (c['cpu_offloading_num_layers'] if ac == 'offload' else 0)
        rows = []
        for name in self.names():
            if name == 'AC':
                v = ac
            elif name == 'AC_LAYERS':
                v = ac_layers
            elif name in OPTICO_KEYS:
                v = c[OPTICO_KEYS[name]]
            elif name.startswith('env:'):
                v = (c.get('env') or {}).get(name[4:])
            else:
                v = c[name]
            rows.append(f'{name}: {_fmt(v)}')
        return '\n'.join(rows)

    def to_config(self, lines: dict) -> dict:
        cfg, env = {}, {}
        for name, raw in lines.items():
            v = _parse(raw)
            if name in OPTICO_KEYS:
                cfg[OPTICO_KEYS[name]] = v
            elif name.startswith('env:'):
                if v is not None:
                    env[name[4:]] = v
            elif name not in ('AC', 'AC_LAYERS'):
                cfg[name] = v
        ac = str(lines.get('AC', 'none')).strip().lower()
        layers = _parse(str(lines.get('AC_LAYERS', '1'))) or 1
        cfg.update(recompute_granularity=None, recompute_method=None, recompute_num_layers=None, recompute_modules=None, cpu_offloading=False)
        if ac == 'full':
            cfg.update(recompute_granularity='full', recompute_method='uniform', recompute_num_layers=layers)
        elif ac == 'sel':
            cfg.update(recompute_granularity='selective', recompute_modules=['core_attn'])
        elif ac == 'offload':
            cfg.update(cpu_offloading=True, cpu_offloading_num_layers=layers)
        if env:
            cfg['env'] = env
        return cfg


def parse_change_para(response: str, current: dict) -> dict:
    """OptiCo analyzer.run_analyzer('change_para') parsing, applied to the current parameter dict."""
    para_new = response.split('</think>')[-1].split('The modified parameters are:')[-1].strip().replace(',', '\n')
    found = dict(re.findall(r'([^:\n]+):([^\n]+)', para_new))
    found = {k.strip(): v.strip() for k, v in found.items()}
    return {k: found.get(k, v) for k, v in current.items()}


def parse_tail(response: str, marker: str) -> str:
    return response.split('</think>')[-1].split(marker)[-1].strip()


def _md_table(header: list[str], rows: list[list]) -> str:
    """OptiCo convert_table2md compaction of a pandas markdown table."""
    lines = ['|' + '|'.join(header) + '|', '|' + '|'.join('-' for _ in header) + '|']
    lines += ['|' + '|'.join(str(x) for x in r) + '|' for r in rows]
    return '\n'.join(lines)


class KnobCodec:
    """Generic codec for an external environment (hooks without param_codec) and for a project manifest (strategy
    optico): the task's knob_schema properties as OptiCo 'KEY: VALUE' lines; values parsed back onto each knob's enum
    (strings, booleans, None). A value is written as domain_text() lists the knob's choices - the empty string as '' and
    a comma as / (OptiCo's change_para parser splits replies on commas, as for SwitchCodec's launch values) - and a reply
    naming a choice that way, quoted or not, is read back as that choice; None, null or nothing for a knob whose choices
    hold the empty string but not None is the empty string. (No recorded arm used this codec before the project strategy:
    F-OPTICO has its hooks' SwitchCodec, the L0 track ParamCodec.)"""

    def __init__(self, task):
        self.props = dict(((getattr(task, 'knob_schema', None) or {}).get('properties')) or {})

    def names(self) -> list[str]:
        return list(self.props)

    @staticmethod
    def fmt(v) -> str:
        if isinstance(v, str):
            return "''" if v == '' else v.replace(',', '/')
        return _fmt(v)

    def to_lines(self, config: dict) -> str:
        return '\n'.join(f'{k}: {self.fmt((config or {}).get(k))}' for k in self.props)

    def parse(self, name: str, raw):
        """One reply value of knob `name` (see the class docstring); a value outside the choices is kept as written, so
        that the validator refuses it with its own message."""
        enum = (self.props.get(name) or {}).get('enum') or []
        text = str(raw).strip()
        quoted = text[1:-1] if len(text) > 2 and text[0] == text[-1] and text[0] in '"\'' else None
        for form in (text, quoted):
            hit = [c for c in enum if form is not None and form == self.fmt(c)]
            if hit:
                return hit[0]
        v = _parse(raw)
        if enum and v not in enum and str(v) in enum:
            v = str(v)
        if v is None and '' in enum and None not in enum:
            v = ''
        return v

    def to_config(self, lines: dict) -> dict:
        return {name: self.parse(name, raw) for name, raw in (lines or {}).items()}

    def domain_text(self) -> str:
        """Each parameter's allowed values as to_lines() writes them (OptiCo's prompts name no domain for parameters
        they do not know), with the knob's own description when the manifest gives one."""
        rows, comma = [], False
        for k, p in self.props.items():
            if p.get('enum'):
                vals = ' | '.join(self.fmt(c) for c in p['enum'])
                comma |= any(isinstance(c, str) and ',' in c for c in p['enum'])
            elif p.get('type') == 'boolean':
                vals = 'true | false'
            else:
                lo, hi = p.get('minimum'), p.get('maximum')
                vals = {'integer': 'an integer', 'number': 'a number', 'string': 'a string'}.get(p.get('type'), 'a value') \
                    + (f' from {lo}' if lo is not None else '') + (f' to {hi}' if hi is not None else '')
            rows.append(f'{k}: {vals}' + (f' ({p["description"]})' if p.get('description') else ''))
        if comma:
            rows.append('(a comma inside a value is written /)')
        return '\n'.join(rows)


def run_optico_style(task, budget, start_config: dict, runner, *, llm_backend, judge=None, model_facts=None,
                     run_id: str = 'optico_style_L0', out_dir=None, start_receipt=None, reference_receipt=None, clock=None,
                     optico_root: str | Path = OPTICO_ROOT, max_unproductive: int = 6, hooks=None,
                     arm: str = 'optico_style_L0', session=None, deviations: list | None = None) -> SearchTrace:
    """hooks (an ext environment's SessionHooks, e.g. ext/flashnext F-OPTICO): validator, codec (hooks.param_codec(), else
    KnobCodec over the task's knob schema) and fixed-parameter text (hooks.fixed_parameters_text()) come from the
    environment; hooks=None runs the L0 config track exactly as before. Hooks that know the job's tokens per update or
    GPU memory (hooks.tokens_per_update(), hooks.gpu_memory_gib(); a project's ProjectHooks) give the TokensPerSec and
    MemUsage% columns, NULL where they return None.
    session: a SearchSession the caller made (the project strategy optico passes the flow's, whose progress lines and
    interrupt report then cover it); its requests, refusal re-sends included, are admitted as the search proposer's are
    (agent.optimize.call_allowances: the answer cap, the tokens, USD and wall time left as backend keywords, none started
    in the last MIN_CALL_WALL_S seconds; the search stops when one does not fit). None makes one here as before, with
    the output cap as the only admission. deviations: the list the trace records (the project strategy's
    DEVIATIONS_PROJECT); None: DEVIATIONS, or DEVIATIONS_HOOKS with hooks."""
    mats = load_materials(optico_root)
    managed = session is not None
    if session is None:
        session = new_session(task=task, budget=budget, runner=runner, judge=judge, run_id=run_id, arm=arm,
                              out_dir=out_dir, start_config=start_config, start_receipt=start_receipt,
                              reference_receipt=reference_receipt, model_facts=model_facts, clock=clock, hooks=hooks)
    raw = session.task_raw
    mf = model_facts or {}
    mem_cap = 80                                       # GiB per GPU MemUsage% divides by (H100 80 GB) unless hooks say
    if hooks is None:
        codec = ParamCodec(task, model_facts)
        world = int(raw['hardware']['gpus'])
        tokens = int(raw['workload']['physical_tokens_per_update'])
        fixed = '\n'.join([f'Model Type: {raw["model"].get("family")}', f'Model Size: {raw["model"].get("key")}',
                           f'NUM_LAYERS: {mf.get("num_layers")}'] + ([f'NUM_EXPERTS: {mf.get("num_experts")}'] if codec.moe else []) +
                          ['Precison: bf16', f'Global Batch Size: {raw["workload"]["packs_per_update"]}',
                           f'Sequence Length: {raw["workload"]["seq_len"]}', f'Number of Nodes: {raw["hardware"]["nodes"]}',
                           f'Number of GPUs: {world}', 'GPU memory limit per card: 81920'])
    else:
        codec = hooks.param_codec() if callable(getattr(hooks, 'param_codec', None)) else KnobCodec(task)
        world = int(getattr(task, 'world_size', None) or raw.get('world_size') or 1)
        facts = getattr(task, 'facts', None) or raw.get('facts') or {}
        tokens = int(((facts.get('data') or {}).get('physical_tokens_per_update')) or 0)
        if callable(getattr(hooks, 'tokens_per_update', None)):     # a project: the manifest's, else unknown (NULL)
            tokens = int(hooks.tokens_per_update() or 0)
        if callable(getattr(hooks, 'gpu_memory_gib', None)):        # a project: hardware.memory_gib, else unknown
            mem_cap = hooks.gpu_memory_gib()
        fixed = (hooks.fixed_parameters_text() if callable(getattr(hooks, 'fixed_parameters_text', None))
                 else json.dumps(facts, ensure_ascii=False, default=str))
    header = codec.names() + ['Result', 'StepTime_s', 'TokensPerSec', 'MFU', 'MemMaxGiB', 'MemUsage%', 'GpuUtilAve', 'Log Director', 'Reasons for failure']
    pool: list[list] = []
    n = {'calls': 0, 'requests': 0, 'lines_dropped': 0, 'refusals': 0}

    def llm(kind: str, text: str) -> str | None:
        """One analyzer request, ChangePrompt(kind) through api_infer: D-OC-15 applied, a refusal re-sent unchanged."""
        if session.stop_flag or session.ledger.exhausted():
            return None
        original = compose(mats, kind, text)
        prompt, dropped = strip_think_format(original)
        notes = getattr(session, 'live_notes', None)
        if notes:
            prompt += '\nUser guidance (advisory; preserve all workload and verification constraints):\n' + json.dumps(notes, ensure_ascii=False)
        n['requests'] += 1
        n['lines_dropped'] += dropped
        record = {'prompt_transform': PROMPT_TRANSFORM, 'lines_dropped': dropped,
                  'prompt_sha256_original': _sha_text(original), 'prompt_sha256_sent': _sha_text(prompt)}
        attempt = 0
        # A refusal is re-sent unchanged after RETRY_WAIT_S (analyzer.api_infer re-sends a failed request up to
        # max_retries=10; the run's MAX_REFUSALS binds first). Other errors are handled as before: one on give_sugg or
        # change_para stops the run; on loc_error the failed row keeps its raw reason and the loop goes on.
        while True:
            if managed:                                # every attempt as the search proposer's call (see the docstring)
                kw = call_allowances(session, llm_backend, SYSTEM + prompt, max_output_tokens(llm_backend))
                if kw is None:
                    session.stop_flag = _declined(session, llm_backend, SYSTEM + prompt)
                    session.event('stop', {'reason': session.stop_flag, 'request': f'optico_{kind}', 'attempt': attempt + 1})
                    return None
            else:
                cap = output_allowance(budget, session.ledger, llm_backend, SYSTEM + prompt)
                if cap is None:
                    session.stop_flag = 'budget:llm_tokens'    # same admission as the agent: output reserved in full
                    return None
                kw = {'max_output_tokens': cap}
            n['calls'] += 1
            attempt += 1
            call_dir = (Path(out_dir) / 'calls' / f'{n["calls"]:04d}_{kind}') if out_dir else None
            res = llm_backend.complete(SYSTEM, prompt, call_dir=call_dir, role=f'optico_{kind}', expect_json=False, **kw)
            refused = is_refusal(res)
            session.charge_llm(res, f'optico_{kind}', **record, attempt=attempt, refusal=refused)
            if not refused:
                return None if res.error else (res.text or '')
            n['refusals'] += 1
            if n['refusals'] >= MAX_REFUSALS:              # D-OC-15 not enough: stop, the arm escalates
                session.stop_flag = 'refusals'
                session.event('stop', {'reason': 'refusals', 'n_refusals': n['refusals'], 'escalate': True})
                return None
            _wait(clock, RETRY_WAIT_S)
            if session.stop_flag or session.ledger.exhausted():
                return None

    def row_for(cid: str, params_text: str, it: int, note: str | None = None) -> list:
        c = session.candidates.get(cid) if cid else None
        values = dict(line.split(': ', 1) for line in params_text.splitlines())
        v = (c.verdict if c else None) or {}
        if c is None or not v:
            return [values[k] for k in codec.names()] + ['fail', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', it, note or 'NULL']
        ok = bool(v.get('valid'))
        t = v.get('step_time')
        act = mf.get('active_params') or mf.get('params')
        mfu = round(6 * act * tokens / (t * world * H100_PEAK_BF16), 4) if ok and t and act else 'NULL'
        mem = v.get('peak_reserved_gib')
        return [values[k] for k in codec.names()] + [
            'success' if ok else 'fail', round(t, 4) if ok and t else 'NULL',
            round(tokens / t, 1) if ok and t and tokens else 'NULL', mfu,
            round(mem, 2) if mem else 'NULL', round(100 * mem / mem_cap, 1) if mem and mem_cap else 'NULL', 'NULL', it,
            note if note else ('NULL' if ok else '; '.join(v.get('reasons', [])[:2]))]

    screen_start(session)
    current = codec.to_lines(session.candidates['c0'].config)
    pool.append(row_for('c0', current, 1))
    it, unproductive, stop = 1, 0, None
    while stop is None:
        session.poll_control()
        if session.stop_flag or session.ledger.exhausted():
            stop = session.stop_flag or 'budget:' + session.ledger.exhausted()[0]
            break
        it += 1
        if pool[-1][len(codec.names())] == 'fail':                      # loc_error agent
            log = pool[-1][-1]
            cause = llm('loc_error', mats['texts']['loc_error'].replace('{LOG}', str(log)).replace('{PARAMETERS}', current))
            if cause is not None:
                pool[-1][-1] = parse_tail(cause, 'The conclusion is:').replace('\n', ' ')[:300]
        table = _md_table(header, pool)
        sugg = llm('give_sugg', mats['texts']['give_sugg'].replace('{FIXED_PARA}', fixed).replace('{TABLE}', table)
                   .replace('{PARAMETERS}', current))
        if sugg is None:
            stop = session.stop_flag or 'llm_error_or_budget'
            break
        suggestion = parse_tail(sugg, 'The suggestions are:')
        names = ', '.join(codec.names())
        resp = llm('change_para', mats['texts']['change_para'].replace('{TABLE}', table).replace('{PARAMETERS}', current)
                   .replace('{SUGGESTIONS}', suggestion).replace('{PARAMETER_NAME}', names))
        if resp is None:
            stop = session.stop_flag or 'llm_error_or_budget'
            break
        cur_dict = dict(line.split(': ', 1) for line in current.splitlines())
        new_dict = parse_change_para(resp, cur_dict)
        new_text = '\n'.join(f'{k}: {v}' for k, v in new_dict.items())
        fb = session.propose_config(config=codec.to_config(new_dict), base='c0', origin='optico')
        if 'candidate_id' in fb:
            session.screen(fb['candidate_id'])
            pool.append(row_for(fb['candidate_id'], new_text, it))
            unproductive = 0
        elif 'duplicate_of' in fb:
            pool.append(row_for(None, new_text, it, note=f'duplicate of iteration with {fb["duplicate_of"]}; not re-run'))
            unproductive += 1
        else:
            pool.append(row_for(None, new_text, it, note='config rejected before launch: ' + '; '.join(fb.get('errors') or [fb.get('error', '')])[:240]))
            unproductive += 1
        current = new_text
        if unproductive >= max_unproductive:
            stop = 'no_new_config'
    if deviations is None:
        deviations = DEVIATIONS if hooks is None else DEVIATIONS_HOOKS
    return session.finalize(stop_reason=stop, extra={'deviations': deviations,
                                                     'optico_provenance': mats['provenance'],
                                                     'pool_rows': len(pool), 'llm_calls': n['calls'], 'pool_header': header,
                                                     'n_refusals': n['refusals'],
                                                     'prompt_transform': {'name': PROMPT_TRANSFORM, 'deviation': 'D-OC-15',
                                                                          'requests': n['requests'],
                                                                          'lines_dropped': n['lines_dropped']}})
