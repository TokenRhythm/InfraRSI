"""optimize(task, package, driver, budget, start_config, runner) -> SearchTrace (FINAL_PLAN 3.3, design 1.6).

One LLM call per turn returns one JSON action; the driver executes it and replays the explicit history next turn.
Budgets (screens, LLM tokens of all roles, wall clock, GPU-hours) stop the search when any is exhausted; failed
screens count. Weak mode applies the weak-model protocol (FINAL_PLAN 7.2): explicit menu, config first, validated
templates before free patches (<= 80 lines), L2 closed, <= 32K context, <= 3 retries.

hooks (ext.common.session_hooks.SessionHooks, EXT_INTERFACES §3) port the loop to an external environment: the task
view, config validator, patch guard, templates, open layers, receipt digest, action schema/menu (incl. the read-only
read_file action), the verifier checklist and the phase guard (frozen package, per-search memory copy) come from the
hooks; the package's prompts are used as they are, with the environment's menu and contract in {ACTION_MENU}.
With hooks=None nothing changes.
"""
from __future__ import annotations

import difflib
import json
import math
import shutil
import tempfile
from pathlib import Path

from ..actions import patch_guard, templates
from . import policy as policy_mod
from . import roles as roles_mod
from .budget import Budget
from .context import (RollingHistory, SkillIndex, estimate_tokens, profile_ranges, receipt_digest, receipt_view, run_tools,
                      task_tags)
from .memory import MemoryDB
from .roles import Roles
from .search import SearchSession, SearchTrace
from .tools import action_schema, normalize


MIN_CALL_WALL_S = 30.0          # no model call is started with less wall budget than this left
MIN_OUTPUT_TOKENS = 256         # no model call is started unless at least this many output tokens can be reserved
DEFAULT_CALL_OVERHEAD_TOKENS = 4000    # backends that do not declare their per-call overhead (CLI system context)
MIN_CALL_USD = 0.05             # with a USD cap, no model call is started with less than this left (Codex W9 H-06)


def output_allowance(budget, ledger, driver, text: str) -> int | None:
    """Output tokens a call on `driver` with input `text` may produce within the token budget (its maximum is reserved
    in full); None when not even MIN_OUTPUT_TOKENS fit. Shared by the agent and the baseline LLM paths."""
    overhead = int(getattr(driver, 'call_overhead_tokens', DEFAULT_CALL_OVERHEAD_TOKENS))
    left = budget.llm_tokens - ledger.used['llm_tokens'] - input_reserve(text) - overhead
    return min(max_output_tokens(driver), left) if left >= MIN_OUTPUT_TOKENS else None


def usd_allowance(budget, ledger, driver=None, text: str = '', out_tokens: int | None = None) -> tuple[bool, float | None]:
    """(admitted, max_usd) of the next model call (Codex W9 H-06). Without a USD cap: (True, None) - nothing changes.
    With one, the USD left is passed to the backend as max_usd (claude_cli enforces it as --max-budget-usd) and the call
    is refused below MIN_CALL_USD; a backend that prices tokens itself (a `price` table, no native dollar limit, e.g.
    openai_compat) is admitted only when its worst case fits: every attempt charged like an unknown-usage attempt
    (each input byte a token at the miss price, the whole output cap at the output price). Failed or unknown calls
    are still charged conservatively by the backends."""
    cap = getattr(budget, 'llm_usd', None)
    if cap is None:
        return True, None
    left = max(0.0, float(cap) - float(ledger.used['llm_usd']))
    if left < MIN_CALL_USD:
        return False, left
    price = getattr(driver, 'price', None)
    if isinstance(price, dict) and price.get('output') is not None:
        out = int(out_tokens or max_output_tokens(driver))
        attempts = 1 + int(getattr(driver, 'max_retries', 0) or 0)
        worst = attempts * (len(text.encode('utf-8')) * float(price.get('input_miss') or 0.0) + out * float(price['output'])) / 1e6
        if worst > left:
            return False, left
    return True, left


def usd_kw(max_usd: float | None) -> dict:
    """The max_usd keyword for a backend call; empty without a USD cap (unchanged calls)."""
    return {} if max_usd is None else {'max_usd': max_usd}


def call_allowances(session, driver, text: str, out_tokens: int) -> dict | None:
    """A search's next model call within its budget, as keywords for driver.complete: the answer cap that fits the
    tokens left after the input and the call overhead (output_allowance), the tokens, USD and wall time left; None
    when the call does not fit (no LLM tokens in the budget included). The search proposer and the transfer judgment."""
    b, led = session.ledger.budget, session.ledger
    out, wall = output_allowance(b, led, driver, text), float(b.wall_seconds) - led.wall()
    if out is None or wall < MIN_CALL_WALL_S:
        return None
    out = min(int(out_tokens), out)
    ok, usd_left = usd_allowance(b, led, driver, text, out)
    if not ok:
        return None
    return {'max_output_tokens': out, 'token_allowance': int(b.llm_tokens) - int(led.used['llm_tokens']),
            'timeout': wall, **usd_kw(usd_left)}


def input_reserve(text: str) -> int:
    """Input tokens reserved for admission: the byte-based estimate plus 25% (tokenizers differ; the exact count is
    only known afterwards, and an overrun makes the whole search ineligible)."""
    return math.ceil(estimate_tokens(text) * 1.25)


def max_output_tokens(driver) -> int:
    """The backend's per-call output cap (reserved in full before a call is admitted)."""
    for name in ('max_output_tokens', 'max_tokens', 'tokens_per_call'):
        v = getattr(driver, name, None)
        if type(v) is int and v > 0:
            return v
    return 16000


def _validation_error(error: str | None) -> bool:
    """Re-asks are for replies that failed validation, never for transport or HTTP failures."""
    return bool(error) and error.startswith(('schema', 'invalid structured JSON', 'response is not a JSON object'))


def _task_view(raw: dict, model_facts: dict | None, hardware: dict | None) -> dict:
    w, m, d, o = raw['workload'], raw['model'], raw['data'], raw['objective']
    return {'task_id': raw['id'], 'purpose': raw.get('purpose'), 'model': {'repo_id': m['repo_id'], 'family': m.get('family'),
            'size_gb': m.get('size_gb'), 'bridge_type': m.get('bridge_type')},
            'data': {'key': d['key'], 'kind': d.get('kind'), 'seq_len': w['seq_len']},
            'workload': {'packs_per_update': w['packs_per_update'], 'physical_tokens_per_update': w['physical_tokens_per_update'],
                         'screen_steps': w['screen_steps'], 'scoring_window': w['scoring_window']},
            'hardware': {'gpus': raw['hardware']['gpus'], 'nodes': raw['hardware']['nodes'], 'gpu_type': raw['hardware'].get('gpu_type'),
                         'profile': hardware},
            'objective': {'optimizer': o['optimizer'], 'clip_grad': o['clip_grad'], 'loss': o['loss'], 'precision': o['precision']},
            'numerics': raw.get('numerics'), 'model_facts': model_facts,
            'fixed': 'GBS, seq, steps, data order, loss, LR schedule, precision, router top-k/capacity are task constants'}


def _memory_tags(raw: dict, model_facts: dict | None) -> dict:
    hw, m, d, w = raw['hardware'], raw['model'], raw['data'], raw['workload']
    return {'hardware': [str(hw.get('gpu_type', '')).lower(), f'gpus{hw["gpus"]}', f'nodes{hw["nodes"]}'],
            'model': [m.get('family'), m.get('bridge_type'), 'moe' if (model_facts or {}).get('num_experts') else 'dense'],
            'data': [d.get('kind'), f'seq{int(w["seq_len"]) // 1024}k']}


def _overlay_diff(base: dict, new: dict, limit: int = 4000) -> str:
    pristine = patch_guard.base_overlays()
    out = []
    for name, text in sorted(new.items()):
        old = base.get(name, pristine.get(name, ''))
        out += list(difflib.unified_diff(old.splitlines(), text.splitlines(), f'a/{name}', f'b/{name}', lineterm='', n=1))
    s = '\n'.join(out)
    return s if len(s) <= limit else s[:limit] + '\n... (truncated)'


def _ext_prompts(roles, pkg, mode: str, pol, hooks, tmpl_menu: list) -> None:
    """The package's proposer/verifier prompts with the environment's action menu, contract and checklist filled in."""
    key = 'weak_proposer.md' if mode == 'weak' else 'proposer.md'
    roles.proposer_system = roles_mod.fill(pkg.prompts[key], ACTION_MENU=hooks.action_menu(mode), MODE=mode, LAYERS=','.join(pol.layers),
                                           PATCH_LINE_LIMIT=min(pol.patch_line_limit, int(hooks.patch_line_limit)),
                                           TEMPLATES=json.dumps(tmpl_menu, separators=(',', ':')), CHECKLIST='')
    roles.verifier_system = roles_mod.fill(pkg.prompts['verifier.md'], MODE=mode,
                                           CHECKLIST='\n'.join(f'- {x}' for x in hooks.verifier_checklist()))
    for name, text in (('proposer', roles.proposer_system), ('verifier', roles.verifier_system)):
        if roles_mod.system_arg_units(text) > roles_mod.MAX_SYSTEM_ARG_UTF16_UNITS:
            raise ValueError(f'{name} system prompt exceeds the argv budget ({roles_mod.system_arg_units(text)} UTF-16 units)')


def optimize(task, package, driver, budget: Budget, start_config: dict, runner, *, judge=None, run_id: str = 'search',
             arm: str = 'A0', out_dir=None, mode: str = 'strong', start_receipt: dict | None = None,
             reference_receipt: dict | None = None, model_facts: dict | None = None, hardware: dict | None = None,
             verifier_driver=None, summarizer_driver=None, use_verifier: bool | None = None, use_skills: bool = True,
             use_memory: bool = True, layers=None, heldout=None, whitelist=None, clock=None,
             context_limit: int | None = None, hooks=None, capability: float | None = None,
             sessions: list | None = None, progress=None, checker=None, control=None) -> SearchTrace:
    """capability (revamp design D): a value in [0, 1] replaces the strong/weak split - the menu form follows
    agent.capability.mode_for(c) and the numeric scaffolding (context, history, patch size, retries, config-first
    gates, templates first) is interpolated between the old weak (0) and strong (1) values."""
    from ..rsi.package import OptimizerPackage          # local: rsi imports agent modules
    from . import capability as cap_mod
    cap_over = {}
    if capability is not None:
        mode = cap_mod.mode_for(capability)
        cap_over = cap_mod.policy_overrides(capability)
    if mode not in ('strong', 'weak'):
        raise ValueError(f'mode {mode!r}')
    pkg = package if isinstance(package, OptimizerPackage) else OptimizerPackage.load(package)
    raw = task.raw if hasattr(task, 'raw') else task
    # Phase guard from the task's own split (G3/G4): development tasks run under the dev whitelist; every other split
    # (held-out, flagship, cross-GPU) only with a HeldoutSession pinned to the frozen package (read-only memory).
    from ..rsi.heldout import ReadWhitelist
    split = raw.get('split')
    if hooks is not None:
        # External environment: the frozen package runs read-only under the hooks' guard; no held-out session.
        if heldout is not None or whitelist is not None:
            raise PermissionError(f'{getattr(task, "id", raw.get("id"))}: an ext task runs under its hooks, not a held-out session')
        hooks.split_guard(pkg)
        if layers and set(layers) - set(hooks.layers):
            raise ValueError(f'layers {tuple(layers)} exceed the environment layers {tuple(hooks.layers)}')
        layers = tuple(layers) if layers else tuple(hooks.layers)
    elif split == 'dev':
        if heldout is not None:
            raise PermissionError(f'{raw["id"]} is a development task; a held-out session does not apply')
        (whitelist or ReadWhitelist.dev()).check_task(raw['id'])
    else:
        if heldout is None:
            raise PermissionError(f'{raw["id"]} is a {split!r} task: optimize() needs a HeldoutSession on the frozen package')
        if whitelist is not None:
            whitelist.check_task(raw['id'])
        heldout.check(pkg, raw['id'])
    pol = policy_mod.load(pkg.policy_text, mode, verifier=use_verifier, layers=list(layers) if layers else None, **cap_over)
    if context_limit:                    # endpoint with a small context (self-hosted weak model): prompt + answer must fit
        pol.max_context_tokens = min(pol.max_context_tokens, int(context_limit))
    out = Path(out_dir) if out_dir else None
    session = SearchSession(task=task, budget=budget, runner=runner, judge=judge, run_id=run_id, arm=arm, out_dir=out,
                            start_config=start_config, start_receipt=start_receipt, reference_receipt=reference_receipt,
                            model_facts=model_facts, mode=mode, layers=pol.layers, clock=clock,
                            stop_after_same_failures=pol.search.get('stop_after_same_failures', 6),
                            package_digest=pkg.digest, driver_info=getattr(driver, 'describe', lambda: {'kind': type(driver).__name__})(),
                            hooks=hooks, patch_line_limit=pol.patch_line_limit if capability is not None else None)
    session.control = control
    session.progress = progress                          # the project flow's console line per finished run (U-1-14)
    if sessions is not None:                             # the caller's: an interrupted run reports this session
        sessions.append(session)
    if hooks is None:
        tmpl_menu = [t for t in templates.menu() if templates.render(t['id'], {}, None)[0] is not None]
    else:
        tmpl_menu = list(hooks.templates())
    roles = Roles(pkg.prompts, mode, pol, driver, verifier_driver=verifier_driver, summarizer_driver=summarizer_driver,
                  calls_dir=(out / 'calls') if out else None, template_menu=tmpl_menu)
    if hooks is not None:
        _ext_prompts(roles, pkg, mode, pol, hooks, tmpl_menu)
    skills = SkillIndex(pkg.skills_dir if use_skills else None)
    read_only = heldout is not None
    mem = None
    if use_memory and pkg.memory_path.exists():
        if read_only:
            mem = MemoryDB(pkg.memory_path, read_only=True)
        else:
            # Never mutate the package: writes go to a per-search working copy; rsi.reflect merges them into A_{k+1}.
            work = (out / 'memory_working.db') if out else Path(tempfile.mkdtemp(prefix='infrarsi_mem_')) / 'memory.db'
            shutil.copyfile(pkg.memory_path, work)
            mem = MemoryDB(work)
    if hooks is None:
        tags = task_tags(raw, model_facts)
        view = _task_view(raw, model_facts, hardware)
        skill_query = ' '.join(tags) + ' ' + str(raw.get('purpose', '')) + ' parallel layout memory recompute'
        mem_tags = _memory_tags(raw, model_facts)
    else:
        tags, view, skill_query, mem_tags = hooks.skill_tags(), hooks.task_view(), hooks.skill_query(), hooks.memory_tags()
    digest_receipt = receipt_digest if hooks is None else hooks.receipt_digest


    auto_skills = skills.retrieve(skill_query, tags, k=pol.retrieve_top_k, max_tokens=pol.skill_tokens)
    for h in auto_skills:
        session.skill_hits.append({'turn': 0, 'source': 'auto', 'skill': h['skill'], 'section': h['section'], 'score': h['score']})
    mem_hits = mem.query(mem_tags, k=pol.memory_hits) if mem else []
    for h in mem_hits:
        session.memory_hits.append({'turn': 0, 'source': 'auto', 'entry_id': h['id'], 'score': h['score'], 'task_id': h.get('task_id')})
    start_payload = {'package': pkg.digest, 'mode': mode, 'policy': vars(pol), 'auto_skills': [(h['skill'], h['section']) for h in auto_skills],
                     'memory_hits': [h['id'] for h in mem_hits], 'heldout': heldout is not None}
    if hooks is not None:
        start_payload['hooks'] = hooks.describe()
    session.event('start', start_payload)

    if session.candidates['c0'].status == 'proposed' and pol.search.get('screen_start_if_missing', True):
        session.screen('c0')

    history = RollingHistory(keep=pol.keep_last_turns)
    last_skills: list = []
    templates_tried = 0
    state = {'model_errors': 0, 'turn': 0}
    authors: dict[str, str] = {}                         # candidate -> which debate model wrote it (`checker` picks its reviewer)

    out_cap = max_output_tokens(driver)
    overhead = int(getattr(driver, 'call_overhead_tokens', DEFAULT_CALL_OVERHEAD_TOKENS))

    def allowance(text: str) -> int | None:
        """Output tokens the next call may produce: the budget left after its (estimated) input and the backend's
        per-call overhead, capped by the backend's maximum; None when not even a minimal answer fits."""
        left = budget.llm_tokens - session.ledger.used['llm_tokens'] - input_reserve(text) - overhead
        return min(out_cap, left) if left >= MIN_OUTPUT_TOKENS else None

    def affordable(text: str) -> bool:
        """A call is admitted only with its maximum output reserved (input + overhead + output within the budget)."""
        return allowance(text) is not None

    def remaining_wall() -> float:
        return budget.wall_seconds - session.ledger.wall()

    def call_timeout() -> float:
        """Never longer than the wall budget left (the main loop stops before a call when < MIN_CALL_WALL_S remain)."""
        return max(1.0, remaining_wall())

    def usd(drv, text: str) -> tuple[bool, float | None]:
        return usd_allowance(budget, session.ledger, drv, text, allowance(text))

    def summarizer(prev: str, lines: list[str]):
        if pol.summarizer != 'llm':
            return None
        if remaining_wall() < MIN_CALL_WALL_S or not affordable(roles.summarizer_system + json.dumps(lines, ensure_ascii=False) + prev):
            return None                                   # fall back to the deterministic fold
        ok, usd_left = usd(roles.summarizer_driver, roles.summarizer_system + json.dumps(lines, ensure_ascii=False) + prev)
        if not ok:
            return None                                   # the USD cap: deterministic fold
        res, text = roles.summarize(prev, lines, timeout=call_timeout(), max_output_tokens=allowance(
            roles.summarizer_system + json.dumps(lines, ensure_ascii=False) + prev), **usd_kw(usd_left))
        session.charge_llm(res, 'summarizer', state['turn'])
        if res.meta.get('unreachable'):                 # terminal: the next proposer call must not be sent
            session.stop_flag = 'endpoint_unreachable'
            return None
        return text

    def l1_gate() -> str | None:
        if 'L1' not in pol.layers:
            return 'L1 patches are closed in this arm'
        if hooks is not None and 'L0' not in pol.layers:
            return None                                   # patch-only environment: there are no configs to try first
        if pol.config_first_min_screens <= 0 and pol.l0_stall_screens <= 0:
            return None
        l0 = [s for s in session.screens if s['kind'] in ('L0', 'start')]
        stall, best = 0, None
        for s in l0:
            if s['valid'] and s['step_time'] and (best is None or s['step_time'] < best):
                best, stall = s['step_time'], 0
            else:
                stall += 1
        if pol.config_first_min_screens and len(l0) >= pol.config_first_min_screens:
            return None
        if pol.l0_stall_screens and stall >= pol.l0_stall_screens:
            return None
        return (f'L1 locked (config first): {len(l0)} L0 screens so far; unlocks after {pol.config_first_min_screens} '
                f'or {pol.l0_stall_screens} L0 screens in a row without a new best (now {stall})')

    def render() -> str:
        best = session.best_valid()
        head = {'budget_used': session.budget_view(), 'start': session.candidates['c0'].summary(None),
                'best_valid': best and session.candidates[best].summary(None), 'layers': pol.layers,
                'l1_gate': l1_gate() if mode == 'weak' else None,
                'templates': [t['id'] for t in tmpl_menu] if 'L1' in pol.layers else []}
        task_block = view if hooks is None else hooks.task_view()      # ext views may change every turn (ENV-B table)
        fixed = ['# TASK', json.dumps(task_block, ensure_ascii=False, default=str), '# STATE', json.dumps(head, ensure_ascii=False, default=str),
                 '# CANDIDATES (latest 20)', json.dumps(session.candidate_table(20), ensure_ascii=False, default=str)]
        # Top-k sections within skill_tokens (protocol: top-2, <= 4K tokens); an explicit retrieval replaces the automatic one.
        sk = (last_skills or auto_skills)[: pol.retrieve_top_k]
        skill_txt = '\n\n'.join(f'## {h["skill"]} / {h["section"]}\n{h["text"]}' for h in sk)
        mem_txt = json.dumps([{k: h.get(k) for k in ('tags', 'slow', 'fast', 'outcome', 'failure_reason', 'lesson')} for h in mem_hits],
                             ensure_ascii=False, default=str)
        tail = ['# YOUR TURN', 'Choose exactly one action. Budget is shared by every model call and screen.']
        for drop in range(3):
            parts = list(fixed)
            if skill_txt and drop < 2:
                parts += ['# SKILLS (retrieved)', skill_txt]
            if mem_hits and drop < 1:
                parts += ['# MEMORY (cross-task)', mem_txt]
            used = estimate_tokens('\n'.join(parts + tail))
            room = pol.max_context_tokens - used - 200
            if room > 500 or drop == 2:
                if estimate_tokens(history.render(10 ** 9)) > room:
                    history.fold(summarizer)
                parts += ['# HISTORY', history.render(max(room, 200))]
                return '\n'.join(parts + tail)
        return '\n'.join(fixed + tail)

    def dispatch(a: dict) -> dict:
        nonlocal last_skills, templates_tried
        kind = a.get('action')
        if kind == 'read_env':
            return {'env': view if hooks is None else hooks.task_view(), 'start_config': session.candidates['c0'].config,
                    'skills_catalogue': skills.catalogue()}
        if kind == 'read_file' and hooks is not None:
            files = None
            if a.get('candidate_id'):
                c = session.candidates.get(a['candidate_id'])
                if c is None:
                    return {'error': f'unknown candidate {a["candidate_id"]!r}'}
                files = hooks.candidate_files(c.overlay)
            return hooks.read_file(a.get('path'), a.get('start') or 1, a.get('end'), files=files)
        if kind in ('read_receipt', 'profile_ranges'):
            c = session.candidates.get(a.get('candidate_id', ''))
            if c is None or c.receipt is None:
                return {'error': f'no receipt for {a.get("candidate_id")!r}'}
            if kind == 'profile_ranges':
                if hooks is not None:
                    return {'error': 'profile_ranges is not available in this environment; use read_receipt'}
                return profile_ranges(c.receipt)
            out = digest_receipt(c.receipt, c.verdict, max_tokens=pol.receipt_digest_tokens, fields=a.get('fields'))
            if pkg.tools and (hooks is None or hooks.tools_enabled):
                parent = session.candidates[c.parent].receipt if c.parent else None
                out['analyses'] = run_tools(pkg.tools, receipt_view(c.receipt, parent, raw))
                session.tool_calls.append({'turn': state['turn'], 'candidate_id': c.id, 'tools': sorted(pkg.tools),
                                           'errors': sorted(k for k, v in out['analyses'].items() if isinstance(v, dict) and 'tool_error' in v)})
            return out
        if kind == 'retrieve_skill':
            hits = skills.retrieve(a.get('query', ''), tags, k=pol.retrieve_top_k, max_tokens=pol.skill_tokens, skill=a.get('skill'))
            for h in hits:
                session.skill_hits.append({'turn': state['turn'], 'source': 'retrieve_skill', 'skill': h['skill'], 'section': h['section'], 'score': h['score']})
            last_skills = hits
            return {'sections': [{'skill': h['skill'], 'section': h['section']} for h in hits],
                    'note': 'text appears under SKILLS next turn' if hits else 'no matching skill sections'}
        if kind == 'propose_config':
            if mode == 'weak' and a.get('config') is not None:
                return {'error': 'weak mode: use "changes" on a base candidate'}
            return session.propose_config(config=a.get('config'), changes=a.get('changes'), base=a.get('base'), rationale=a.get('rationale', ''))
        if kind == 'propose_patch':
            if mode == 'weak':
                gate = l1_gate()
                if gate:
                    session.count_invalid('l1_locked')
                    return {'error': gate}
                if a.get('free_patch') and pol.templates_before_free_patch and tmpl_menu and templates_tried == 0:
                    session.count_invalid('template_first')
                    return {'error': 'apply a validated patch template before a free patch', 'templates': [t['id'] for t in tmpl_menu]}
            elif 'L1' not in pol.layers:
                session.count_invalid('layer_closed')
                return {'error': 'L1 patches are closed in this arm'}
            if a.get('template'):
                templates_tried += 1
            return session.propose_patch(files=a.get('files'), template=a.get('template'), params=a.get('params'),
                                         base=a.get('base'), rationale=a.get('rationale', ''))
        if kind == 'run_screen':
            cid = a.get('candidate_id', '')
            c = session.candidates.get(cid)
            if c is None:
                return {'error': f'unknown candidate {cid!r}'}
            if pol.verifier and c.status == 'proposed' and c.kind != 'start':
                if checker is not None:                   # the debate: not the model that wrote this candidate
                    roles.verifier_driver = checker(authors.get(cid, 'main'))[0]
                parent = session.candidates[c.parent]
                diff = _overlay_diff if hooks is None else hooks.overlay_diff
                review = {'candidate': c.summary(parent.config), 'rationale': c.rationale,
                          'parent_receipt': digest_receipt(parent.receipt, parent.verdict, max_tokens=700) if parent.receipt else None,
                          'overlay_diff': diff(parent.overlay, c.overlay) if c.overlay != parent.overlay else None,
                          'validation_warnings': c.warnings}
                if not affordable(roles.verifier_system + roles.verify_prompt(review)):
                    session.stop_flag = 'budget:llm_tokens'
                    return {'error': 'LLM token budget exhausted before verification; search stops'}
                ok, usd_left = usd(roles.verifier_driver, roles.verifier_system + roles.verify_prompt(review))
                if not ok:
                    session.stop_flag = 'budget:llm_usd'
                    return {'error': 'LLM USD budget exhausted before verification; search stops'}
                res = roles.verify(review, timeout=call_timeout(),
                                   max_output_tokens=allowance(roles.verifier_system + roles.verify_prompt(review)),
                                   **usd_kw(usd_left))
                session.charge_llm(res, 'verifier', state['turn'])
                session.verifier_stats['calls'] += 1
                if res.meta.get('unreachable'):
                    session.stop_flag = 'endpoint_unreachable'
                    return {'error': f'{res.error}; search stops'}
                if res.action is None:
                    session.verifier_stats['errors'] += 1          # verifier can only veto; failures do not block
                elif res.action.get('verdict') == 'veto':
                    session.verifier_stats['vetoes'] += 1
                    c.status = 'vetoed'
                    session.event('veto', {'candidate_id': cid, 'reasons': res.action.get('reasons')})
                    return {'candidate_id': cid, 'vetoed': True, 'reasons': res.action.get('reasons', [])[:4]}
            fb = session.screen(cid)
            if 'error' not in fb:
                fields = ('id', 'verdict', 'step_time', 'memory', 'events') if hooks is None else ('id', 'verdict', 'metric', 'summary', 'guards')
                fb['receipt'] = digest_receipt(c.receipt, c.verdict, max_tokens=min(600, pol.receipt_digest_tokens), fields=fields)
            return fb
        if kind == 'write_memory':
            entry = dict(a.get('entry') or {})
            entry['tags'] = {**mem_tags, **(entry.get('tags') or {})}
            if mem is None or read_only:
                session.memory_writes.append({'turn': state['turn'], 'noop': True, 'reason': 'held-out (read-only)' if read_only else 'memory off'})
                if heldout is not None:
                    heldout.log_noop(entry)
                return {'stored': False, 'note': 'write_memory is a no-op in this phase'}
            try:
                rid = mem.add(entry, source=run_id, task_id=raw['id'])
            except (KeyError, ValueError) as exc:
                return {'error': f'memory entry rejected: {exc}'}
            session.memory_writes.append({'turn': state['turn'], 'noop': False, 'entry_id': rid, 'entry': entry})
            return {'stored': rid is not None, 'entry_id': rid}
        if kind == 'finish':
            return {'finish': True}
        return {'error': f'unknown action {kind!r}'}

    schema = action_schema(mode) if hooks is None else hooks.action_schema(mode)
    stop = None
    finish_choice = None
    while True:
        session.poll_control()
        if session.stop_flag:
            stop = session.stop_flag
            break
        ex = session.ledger.exhausted()
        if ex:
            stop = 'budget:' + ex[0]
            break
        if state['turn'] >= pol.max_turns:
            stop = 'max_turns'
            break
        if remaining_wall() < MIN_CALL_WALL_S:
            stop = 'budget:wall_seconds'
            break
        state['turn'] += 1
        prompt = render()
        if session.stop_flag:                          # e.g. the summarizer lost the endpoint while rendering
            stop = session.stop_flag
            break
        if not affordable(roles.proposer_system + prompt):
            stop = 'budget:llm_tokens'
            break
        ok, usd_left = usd(roles.driver, roles.proposer_system + prompt)
        if not ok:
            stop = 'budget:llm_usd'
            break
        res = roles.propose(prompt, schema, timeout=call_timeout(), max_output_tokens=allowance(roles.proposer_system + prompt),
                            token_allowance=budget.llm_tokens - session.ledger.used['llm_tokens'], **usd_kw(usd_left))
        session.charge_llm(res, 'proposer', state['turn'])
        if res.meta.get('unreachable'):     # operator problem, not a model error: stop at once, no retries
            session.event('endpoint_unreachable', {'turn': state['turn'], 'error': res.error})
            stop = 'endpoint_unreachable'
            break
        retries = 0
        while (res.error and _validation_error(res.error) and mode == 'weak' and res.attempts <= 1
               and retries < pol.max_retries and not session.ledger.exhausted() and affordable(roles.proposer_system + prompt)
               and remaining_wall() >= MIN_CALL_WALL_S):
            retry_prompt = prompt + f'\n\nYOUR PREVIOUS REPLY WAS REJECTED ({res.error}). Reply with exactly one valid JSON action.'
            ok, usd_left = usd(roles.driver, roles.proposer_system + retry_prompt)
            if not ok:
                break                                   # the USD cap: no re-ask (the reply stays a model error)
            retries += 1
            res = roles.propose(retry_prompt, schema, timeout=call_timeout(),
                                max_output_tokens=allowance(roles.proposer_system + retry_prompt),
                                token_allowance=budget.llm_tokens - session.ledger.used['llm_tokens'], **usd_kw(usd_left))
            session.charge_llm(res, 'proposer_retry', state['turn'])
            if res.meta.get('unreachable'):
                break                                   # handled right below; never re-ask after endpoint loss
        if res.meta.get('unreachable'):     # lost during a retry: same fail-fast stop
            session.event('endpoint_unreachable', {'turn': state['turn'], 'error': res.error})
            stop = 'endpoint_unreachable'
            break
        if res.error or res.action is None:
            state['model_errors'] += 1
            history.add(state['turn'], None, {'error': res.error})
            session.event('model_error', {'turn': state['turn'], 'error': res.error})
            if state['model_errors'] >= pol.search.get('max_model_errors_in_row', 6):
                stop = 'model_errors'
                break
            continue
        state['model_errors'] = 0
        action = normalize(res.action, mode) if hooks is None else hooks.normalize(res.action, mode)
        feedback = dispatch(action)
        if str(action.get('action')).startswith('propose_') and isinstance(feedback, dict) and feedback.get('candidate_id'):
            authors[feedback['candidate_id']] = (res.meta or {}).get('author', 'main')
        history.add(state['turn'], res.action, feedback)
        session.event('turn', {'turn': state['turn'], 'action': res.action, 'feedback': feedback})
        if action.get('action') == 'finish':
            finish_choice, stop = action.get('candidate_id'), 'finish'
            break
    if mem is not None:
        mem.close()
    extra = {'turns': state['turn'], 'policy': vars(pol), 'heldout': heldout is not None,
             'auto_skills': [(h['skill'], h['section']) for h in auto_skills],
             'templates_available': [t['id'] for t in tmpl_menu]}
    if hooks is not None:
        extra['hooks'] = hooks.describe()
    return session.finalize(selected=finish_choice, stop_reason=stop or 'unknown', extra=extra)
