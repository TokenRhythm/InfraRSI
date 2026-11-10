"""The startup questions of `infrarsi optimize` and `infrarsi run`: a compact editable summary before any GPU
time. Enter starts with the displayed settings; more opens advanced settings and walk asks every item.

Asked only on a terminal (stdin and stdout), and never for a suite or `--resume`; `--yes` / `--no-interactive` skip them, and a
script, CI or a pipe is not asked anything. What a question changes is either a command-line setting (`a.<flag>`, as if given
on the command line) or a field of infrarsi.yaml, written back once at the end (the file is only touched when an item was
changed; its first version is kept next to it as infrarsi.yaml.bak).
"""
from __future__ import annotations

import copy
import dataclasses
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

import yaml

from .manifest import ManifestError, Project

API_BACKENDS = ('anthropic', 'openai', 'openai-compat', 'vllm', 'deepseek')
HELP = 'Enter keeps the value shown; `a` accepts the rest; `q` quits; `?` explains this item; `none` clears an optional one.'


class Quit(Exception):
    """The user stopped at the questions: nothing is run."""


UNSET = object()                                        # a manifest field to remove (a knob's added `choices` taken back)


def wanted(a, *, suite: bool = False, resume: bool = False, stdin=None, stdout=None) -> bool:
    """Whether to ask: an explicit --interactive / --no-interactive (--yes) decides, else a terminal on both ends; never for a
    suite or a resumed run, whose settings are already fixed."""
    if suite or resume or getattr(a, 'yes', False):
        return False
    flag = getattr(a, 'interactive', None)
    if flag is not None:
        return bool(flag)
    out, err = stdin or sys.stdin, stdout or sys.stdout
    return bool(getattr(out, 'isatty', lambda: False)() and getattr(err, 'isatty', lambda: False)())


@dataclasses.dataclass
class Item:
    label: str
    show: Callable[[Any], str]                          # the value the run would use now, in words
    parse: Callable[[Any, str], Any]                    # the answer into a value (ValueError: asked again, with why)
    store: Callable[[Any, Any], None]
    explain: str | Callable[[Any], str] = ''           # what `?` prints (a function of the interview, when it depends on the job)
    when: Callable[[Any], bool] = lambda ctx: True
    optional: bool = False


class Interview:
    """The questions for one project: `project` as it stands with the answers so far, `changes` the manifest fields changed."""

    def __init__(self, project: Project, a, *, read=None, write=None, llm_spec=None):
        self.project, self.a = project, a
        self.read, self.say = read or input, write or print
        self.changes: dict[tuple, Any] = {}
        self.accepted = False
        self.llm_spec = llm_spec                        # what the model would be without an answer (optimize detects it)
        self.knobs0 = copy.deepcopy(project.knobs)      # as they were: a later answer to the knob question starts from them

    # ------------------------------------------------------------ the answers
    def change(self, path: tuple, value) -> None:
        """A manifest field; ValueError (asked again) when the manifest would no longer load."""
        raw = copy.deepcopy(self.project.raw)
        cur = raw
        for key in path[:-1]:
            cur = cur.setdefault(key, {})
        if value is UNSET:
            cur.pop(path[-1], None)
        else:
            cur[path[-1]] = value
        try:
            self.project = Project(raw, self.project.path)
        except ManifestError as exc:
            raise ValueError(str(exc)) from None
        self.changes[path] = value

    def flag(self, name: str, value) -> None:
        setattr(self.a, name, value)

    def budget(self) -> dict:
        return self.project.raw['search']['budget']

    # ------------------------------------------------------------ asking
    def line(self, prompt: str) -> str:
        try:
            return self.read(prompt)
        except EOFError:                                # stdin closed: no consent to run anything
            raise Quit from None

    def ask(self, item: Item, n: int, total: int) -> None:
        while not self.accepted:
            reply = self.line(f'[{n}/{total}] {item.label} [{item.show(self)}]: ').strip()
            low = reply.lower()
            if not reply:
                return
            if low in ('a', 'all'):
                self.accepted = True
                return
            if low in ('q', 'quit', 'exit'):
                raise Quit
            if reply == '?':
                self.say('  ' + ((item.explain(self) if callable(item.explain) else item.explain) or HELP))
                continue
            try:
                value = None if item.optional and low in ('none', '-') else item.parse(self, reply)
                item.store(self, value)
                return
            except ValueError as exc:
                self.say(f'  ! {exc}')

    def walk(self) -> None:
        """The optional full walkthrough, followed by run()'s editable summary."""
        n = 0
        for item in questions():                        # one at a time: an answer can open items (the debate, the API)
            if self.accepted:
                break
            if item.when(self):
                n += 1
                self.ask(item, n, sum(1 for i in questions() if i.when(self)))
    def run(self, *, walk=False) -> None:
        """A short, editable summary; the full checklist is opt-in."""
        self.say('🚀 InfraRSI measures and verifies changes to make your training job faster.')
        self.say('⚠️ Start with a codebase or launch command that already runs successfully. '
                 'Try it once yourself before optimizing; use --command or pass your launch script.')
        self.say('📋 How to use this checklist: review the six settings, press Enter to start, '
                 'or type a number to edit. `more` shows advanced settings; `walk` walks through every item; `q` quits.')
        self.say('While editing: ' + HELP)
        self.say('💾 Continue a saved search: infrarsi run --resume RUN_DIR --screens TOTAL. '
                 'The budget includes work already done. During search, Ctrl-G opens chat.')
        self.say('Budget: full runs is the only default limit. Optional time, token and dollar caps '
                 'stop the search when any configured cap is reached; existing manifest limits are kept.')
        advanced = bool(walk)
        if walk:
            self.walk()
        basic = {'Knobs the search may change', 'Code files the search may patch', 'Loss tolerance',
                 'Search budget: full runs', 'Model', 'Requirements in your own words'}
        while True:
            items = [i for i in questions() if i.when(self) and (advanced or i.label in basic)]
            self.say('')
            self.say('These settings will be used:' + (' (advanced)' if advanced else ''))
            for n, item in enumerate(items, 1):
                self.say(f'  {n:2d}. {item.label}: {item.show(self)}')
            if not advanced:
                caps = [f'{i.label.removeprefix("Search budget: ")}={i.show(self)}'
                        for i in questions() if i.label.startswith('Search budget:') and i.label not in basic]
                self.say('  Optional caps: ' + ', '.join(caps))
            reply = self.line('Start? [Enter: yes; number: edit; more: advanced; walk: all questions; q: quit] ').strip().lower()
            if not reply or reply in ('a', 'all'):
                return
            if reply in ('q', 'quit', 'n', 'no'):
                raise Quit
            if reply in ('more', 'less'):
                advanced = reply == 'more'
                continue
            if reply == 'walk':
                self.accepted = False
                self.walk()
                advanced = True
                continue
            if reply.isdigit() and 1 <= int(reply) <= len(items):
                self.accepted = False
                self.ask(items[int(reply) - 1], int(reply), len(items))
                continue
            self.say('  ! type Enter, a number from the list, more, walk, or q')

    def save(self) -> Path | None:
        """The changed manifest fields into infrarsi.yaml (its comments at the top kept, the first version as .bak); None when
        no item of the manifest changed."""
        if not self.changes:
            return None
        path = self.project.path
        text = path.read_text(encoding='utf-8')
        doc = yaml.safe_load(text) or {}
        for keys, value in self.changes.items():
            cur = doc
            for key in keys[:-1]:
                if not isinstance(cur.get(key), dict):
                    cur[key] = {}
                cur = cur[key]
            if value is UNSET:
                cur.pop(keys[-1], None)
            else:
                cur[keys[-1]] = value
        head = ''.join(ln for ln in re.findall(r'^.*\n?', text, re.M) if ln.startswith('#')) if text.lstrip().startswith('#') else ''
        backup = path.with_name(path.name + '.bak')
        if not backup.exists():
            backup.write_text(text, encoding='utf-8')
        path.write_text(head + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding='utf-8')
        return path


# ---------------------------------------------------------------- parsing an answer
def _int(text: str, *, low: int = 1) -> int:
    try:
        v = int(text)
    except ValueError:
        raise ValueError('a whole number, please') from None
    if v < low:
        raise ValueError(f'at least {low}')
    return v


def _float(text: str) -> float:
    try:
        return float(text.rstrip('%')) / (100 if text.endswith('%') else 1)
    except ValueError:
        raise ValueError('a number, please') from None


def _tolerance(ctx, text: str) -> float:
    v = _float(text)
    if not 0 <= v < 1:
        raise ValueError('a fraction in [0, 1), e.g. 0.03 or 3%')
    return v


def _names(text: str) -> list[str]:
    return [x.strip() for x in re.split(r'[,\s]+', text) if x.strip()]


def _window(ctx, text: str) -> list[int]:
    m = re.fullmatch(r'\s*(\d+)\s*(?:-|\.\.|\s)\s*(\d+)\s*', text)
    if not m:
        raise ValueError('two update numbers, e.g. 11-30')
    return [int(m[1]), int(m[2])]


def _target(name: str):
    def parse(ctx, text: str):
        from ..agent.budget import target_value
        return target_value(name, _float(text))
    return parse


def _tokens(ctx, text: str):
    from ..search.token_governor import parse_tokens
    return parse_tokens(text)


def _hours(ctx, text: str):
    from ..search.token_governor import parse_duration_hours
    return parse_duration_hours(text)


def _usd(ctx, text: str) -> float:
    v = _float(text.lstrip('$'))
    if v <= 0:
        raise ValueError('more than zero, or none')
    return v


def _spec(text: str, *, what: str) -> str:
    """backend[:model], the name of a backend this version knows; the endpoint and the key are checked by the run, once the
    questions after this one have given them."""
    from .llm import BACKENDS
    backend, _, model = text.partition(':')
    if backend not in BACKENDS:
        raise ValueError(f'{what}: unknown backend {backend!r} (one of: {", ".join(BACKENDS)})')
    if backend in ('openai', 'openai-compat', 'vllm') and not model:
        raise ValueError(f'{what}: {backend} needs a model name ({backend}:<model>)')
    return text


def _set_flag(name: str):
    return lambda ctx, value: ctx.flag(name, value)


def _set_field(*path):
    return lambda ctx, value: ctx.change(tuple(path), value)


def _set_or_clear(name: str, *paths):
    """A flag; `none` also takes the value out of the manifest, wherever it is set (a flag can only override it, not clear it);
    `search.stop` before `search.budget`: the manifest fills the budget from it."""
    def store(ctx, value) -> None:
        ctx.flag(name, value)
        if value is not None:
            return
        for path in paths:
            cur = ctx.project.raw
            for key in path:
                cur = cur.get(key) if isinstance(cur, dict) else None
            if cur is not None:
                ctx.change(tuple(path), None)
            if path[0] == 'llm':                        # (the model's settings as optimize detected them: shown, and used)
                ctx.llm_spec = ctx.llm_spec and {k: v for k, v in ctx.llm_spec.items() if k != path[-1]}
                ctx.flag('llm_cleared', [*(getattr(ctx.a, 'llm_cleared', None) or ()), path[-1]])
    return store


def _set_list(*path):
    """A list field: `none` empties it."""
    return lambda ctx, value: ctx.change(tuple(path), list(value or []))


# ---------------------------------------------------------------- the knobs
def _knobs_shown(ctx) -> str:
    free = [k for k, v in ctx.project.knobs.items() if not (v.get('choices') and len(v['choices']) == 1)]
    return (f'{len(free)} detected parameters: ' + ', '.join(free)) if free else 'none adjustable'


def _knobs_detail(ctx) -> str:
    ref = ctx.project.reference_config()
    def values(v):
        return v['choices'] if v.get('choices') else 'true/false' if v.get('type') == 'bool' else f'{v.get("min")}..{v.get("max")}'
    rows = [f'{k}={ref.get(k)!r} of {values(v)}' for k, v in ctx.project.knobs.items()]
    return ('These are detected configuration parameters, not a limit on edits or code patches. Add more under knobs in infrarsi.yaml. The search may change these (name=current of values): ' + '; '.join(rows)
            + '. -a,b keeps a and b at their current values; only a,b lets only these change')


def _knobs_store(ctx, answer) -> None:
    mode, names = answer
    ref = ctx.project.reference_config()
    free = [k for k in ctx.knobs0 if (k not in names if mode == 'drop' else k in names)]
    for k, was in ctx.knobs0.items():
        if k not in free:
            ctx.change(('knobs', k, 'choices'), [ref.get(k)])
            ctx.change(('knobs', k, 'narrowed'), True)
        elif 'choices' in ctx.project.knobs[k] and ctx.project.knobs[k].get('choices') != was.get('choices'):
            ctx.change(('knobs', k, 'choices'), was['choices'] if 'choices' in was else UNSET)   # (this answer lets it change again)
            ctx.change(('knobs', k, 'narrowed'), UNSET)


def _knobs_parse(ctx, text: str):
    only = text.lower().startswith('only ')
    names = _names(text[5:] if only else text.lstrip('-'))
    if not only and not text.startswith('-'):
        raise ValueError('-name1,name2 keeps these at their current values; only name1,name2 lets only these change')
    wrong = [n for n in names if n not in ctx.project.knobs]
    if wrong:
        raise ValueError(f'not a knob of this job: {", ".join(wrong)} (knobs: {", ".join(ctx.project.knobs)})')
    return ('only' if only else 'drop', names)


# ---------------------------------------------------------------- the model and the debate
def _llm_now(ctx) -> dict:
    from .llm import parse
    spec = ctx.llm_spec if ctx.llm_spec is not None else parse(ctx.project.raw.get('llm'))
    if getattr(ctx.a, 'llm', None) is not None:
        from .llm import override
        spec = override(ctx.project.raw.get('llm'), ctx.a.llm)
    return dict(spec or {})


def _llm_shown(ctx) -> str:
    s = _llm_now(ctx)
    return f'{s["backend"]}' + (f':{s["model"]}' if s.get('model') else '') if s.get('backend') and s['backend'] != 'none' else 'none (a search without a model)'


def _is_api(ctx) -> bool:
    return _llm_now(ctx).get('backend') in API_BACKENDS


def _debate(ctx) -> dict:
    from .llm import debate_settings
    return debate_settings(ctx.project.raw.get('debate'), enabled=getattr(ctx.a, 'debate', None), second=getattr(ctx.a, 'second_llm', None),
                           verifier=getattr(ctx.a, 'verifier_llm', None))


def _debate_on(ctx) -> bool:
    return bool(_debate(ctx)['enabled'])


def _yes_no(ctx, text: str) -> bool:
    low = text.lower()
    if low in ('y', 'yes', 'on', 'true', '1'):
        return True
    if low in ('n', 'no', 'off', 'false', '0'):
        return False
    raise ValueError('yes or no')


def _store_debate(ctx, value) -> None:
    ctx.flag('debate', value)
    if value and not _debate(ctx).get('second'):         # asked at once: a debate without a second model does not start
        ctx.say('  a second model is needed: the next item asks for it')


def _requirements_shown(ctx) -> str:
    from .requirements import manifest_texts
    got = list(getattr(ctx.a, 'interactive_requirements', None) or []) + list(getattr(ctx.a, 'requirements', None) or [])
    n = len(manifest_texts(ctx.project))
    return (f'{len(got)} given' if got else 'none') + (f', {n} in infrarsi.yaml' if n else '')


def _requirements_store(ctx, value) -> None:
    pieces = [value]
    while True:
        more = ctx.line('    another requirement (Enter: done): ').strip()
        if more.lower() in ('q', 'quit'):
            raise Quit
        if not more:
            break
        pieces.append(more)
    ctx.flag('interactive_requirements', list(getattr(ctx.a, 'interactive_requirements', None) or []) + pieces)


def questions() -> list[Item]:
    """The items, in the order they are asked; every one has a value already, so Enter is always an answer."""
    def budget(key, unit=''):
        def show(ctx):
            given = {'screens': 'screens', 'wall_hours': 'wall_hours', 'llm_tokens': 'budget_tokens', 'llm_usd': 'budget_usd'}[key]
            v = getattr(ctx.a, given, None)
            v = ctx.budget().get(key) if v is None else v
            return 'no limit' if v is None else f'{v:,}{unit}' if isinstance(v, int) else f'{v:g}{unit}'
        return show

    return [
        Item('Config file', lambda c: str(c.project.raw['config'].get('file') or 'none'), lambda c, t: t or None, _set_field('config', 'file'),
             'the yaml or json file whose values are your job\'s current settings', optional=True),
        Item('GPUs per node', lambda c: str(c.project.gpus), lambda c, t: _int(t), _set_field('hardware', 'gpus')),
        Item('Nodes', lambda c: str(c.project.nodes), lambda c, t: _int(t), _set_field('hardware', 'nodes'),
             when=lambda c: c.project.raw['execution'].get('where') != 'ssh'),
        Item('Updates per run', lambda c: str(c.project.steps), lambda c, t: _int(t, low=2), _set_field('workload', 'steps'),
             'how many training updates each trial run makes; keep it short enough to run many, long enough to warm up'),
        Item('Timed updates', lambda c: '{}-{}'.format(*c.project.window), _window, _set_field('workload', 'window'),
             'the updates whose time is measured; earlier ones are warm-up'),
        Item('Knobs the search may change', _knobs_shown, _knobs_parse, _knobs_store, _knobs_detail),
        Item('Code files the search may patch', lambda c: ', '.join(c.project.raw['code']['mutable']) or 'none',
             lambda c, t: _names(t), _set_list('code', 'mutable'), 'paths or globs, comma-separated; none: no code changes', optional=True),
        Item('Files that must never change', lambda c: ', '.join(c.project.raw['code']['frozen']) or 'none',
             lambda c, t: _names(t), _set_list('code', 'frozen'), 'paths or globs, comma-separated (loss, data, evaluation ...)', optional=True),
        Item('What must stay the same (invariants)',
             lambda c: ', '.join(f'{k} = {v}' for k, v in (c.project.raw['workload'].get('invariants') or {}).items()) or 'none declared',
             lambda c, t: _invariant(c, t), _set_field('workload', 'invariants'),
             'name=expression over the knobs (e.g. global_batch=micro_batch*accum*gpus); a candidate that changes it never runs'),
        Item('Loss tolerance', lambda c: f'{getattr(c.a, "tolerance", None) if getattr(c.a, "tolerance", None) is not None else c.project.tolerance:g}',
             _tolerance, _set_flag('tolerance'), 'how far the loss may move from your current job\'s: 0.03 is 3 %'),
        Item('Search strategy', lambda c: str(getattr(c.a, 'strategy', None) or c.project.raw['search'].get('strategy') or 'auto'),
             lambda c, t: _strategy(c, t), _set_flag('strategy'), 'auto picks one for the job and the model'),
        Item('Target speed-up', lambda c: str(_shown_target(c, 'target_speedup')), _target('target_speedup'), _set_or_clear('target_speedup', ('search', 'stop', 'target_speedup'), ('search', 'budget', 'target_speedup')),
             'stop the search once a verified run is this many times faster, e.g. 1.5; none: until the budget ends', optional=True),
        Item('Target MFU', lambda c: str(_shown_target(c, 'target_mfu')), _target('target_mfu'), _set_or_clear('target_mfu', ('search', 'stop', 'target_mfu'), ('search', 'budget', 'target_mfu')),
             'stop once the model FLOPs utilization reaches this, e.g. 45 or 0.45; none: no such stop', optional=True),
        Item('Search budget: full runs', budget('screens'), lambda c, t: _int(t), _set_flag('screens')),
        Item('Search budget: wall clock (hours)', budget('wall_hours', ' h'), _hours, _set_or_clear('wall_hours', ('search', 'budget', 'wall_hours')), 'hours, 90m, 3600s, or none: no time limit', optional=True),
        Item('Search budget: LLM tokens', budget('llm_tokens'), _tokens, _set_or_clear('budget_tokens', ('search', 'budget', 'llm_tokens')), 'an integer, 500k, 2M, or none: no token limit', optional=True),
        Item('Search budget: LLM dollars', budget('llm_usd', ' USD'), _usd, _set_or_clear('budget_usd', ('search', 'budget', 'llm_usd')), 'none: no dollar limit', optional=True),
        Item('Model', _llm_shown, lambda c, t: _spec(t, what='model'), _set_flag('llm'), 'backend[:model]: claude-cli, codex-cli, deepseek, '
             'openai:gpt-..., openai-compat:..., vllm:..., none'),
        Item('API base URL', lambda c: str(getattr(c.a, 'api_base', None) or _llm_now(c).get('base_url') or 'the backend\'s default'),
             lambda c, t: t, _set_or_clear('api_base', ('llm', 'base_url')), when=_is_api, optional=True),
        Item('API key variable', lambda c: str(getattr(c.a, 'api_key_env', None) or _llm_now(c).get('api_key_env') or 'the backend\'s default'),
             lambda c, t: _key_env(c, t), _set_or_clear('api_key_env', ('llm', 'api_key_env')), 'the name of the environment variable holding the key, never the key', when=_is_api, optional=True),
        Item('Dual-model debate at proposals', lambda c: 'on' if _debate_on(c) else 'off', _yes_no, _store_debate,
             'a second model also proposes, and the main model judges both without knowing whose is which; about three times the tokens per '
             'proposal; off by default'),
        Item('Second model (for the debate)', lambda c: str(_debate(c).get('second') or 'none'), lambda c, t: _spec(t, what='second model'), _set_flag('second_llm'),
             'backend[:model], as for the main model', when=_debate_on),
        Item('Patch checker (for the debate)', lambda c: {None: 'the model that did not write the patch', 'none': 'none'}.get(_debate(c).get('verifier'), str(_debate(c).get('verifier'))),
             lambda c, t: 'none' if t.lower() == 'none' else _spec(t, what='patch checker'), _set_flag('verifier_llm'),
             'checks every code patch before it runs, at most two rounds', when=_debate_on),
        Item('Requirements in your own words', _requirements_shown, lambda c, t: t, _requirements_store,
             'e.g. "do not change the optimizer": any language; they can only narrow the search, and you see what they meant before it starts'),
    ]


def _invariant(ctx, text: str) -> dict:
    if '=' not in text:
        raise ValueError('name=expression over the knobs')
    name, expr = text.split('=', 1)
    return {**(ctx.project.raw['workload'].get('invariants') or {}), name.strip(): expr.strip()}


def _strategy(ctx, text: str) -> str:
    from .. import search
    if text == 'auto':                                  # (decided later, with the model the run ends up with)
        return text
    try:
        return search.resolve(text, has_llm=True, knobs=ctx.project.knobs)
    except ValueError as exc:
        raise ValueError(str(exc)) from None


def _policy(text: str) -> str:
    if text not in ('verify', 'stop'):
        raise ValueError('verify or stop')
    return text


def _key_env(ctx, name: str) -> str:
    if not os.environ.get(name):
        ctx.say(f'  ! the environment variable {name} is not set in this shell')
    return name


def _shown_target(ctx, name: str):
    v = getattr(ctx.a, name, None)
    v = ctx.budget().get(name) if v is None else v
    return 'no limit' if v is None else f'{v:g}'


class RequirementsConfirm:
    """The questions' part of project/requirements.decide: what the compiled requirements mean, and what to do with them."""

    def __init__(self, interview: Interview):
        self.qa = interview

    def __call__(self, record: dict) -> str:
        from .requirements import blocking
        n = len(blocking(record))
        typed = bool((record.get('source') or {}).get('interactive'))
        go_on = n and not typed and self._partial()     # the policy the run was started with stands, for what it already held
        prompt = (f'Requirements: [Enter] go on without the {n} that cannot be honoured; e) edit the text; q) quit: ' if go_on
                  else f'Requirements: s) go on without the {n} that cannot be honoured; e) edit the text; [Enter / q] quit: ' if n
                  else 'Requirements: [Enter] go on with these; e) edit the text; q) quit: ')
        while True:
            low = self.qa.line(prompt).strip().lower()
            if go_on and not low:
                return 'skip'
            if low in ('', 'q', 'quit') and n:
                return 'quit'
            if not low:
                self._keep(record)
                return 'go'
            if low in ('q', 'quit'):
                return 'quit'
            if low in ('e', 'edit'):
                return 'edit'
            if low in ('s', 'skip') and n:
                self._keep(record)
                return 'skip'
            self.qa.say('  ! one of the letters above')

    def _partial(self) -> bool:
        return bool(getattr(self.qa.a, 'requirements_partial', False)
                    or (self.qa.project.raw.get('requirements') or {}).get('on_unsupported') == 'continue')

    def _keep(self, record: dict) -> None:
        from .requirements import RequirementsError, append_texts
        typed = (record.get('source') or {}).get('interactive') or []
        if not typed or self.qa.project.path is None:
            return
        low = self.qa.line('Keep these requirements in infrarsi.yaml for later runs? [y/N]: ').strip().lower()
        if low in ('q', 'quit'):
            raise Quit
        if low in ('y', 'yes'):
            try:
                append_texts(self.qa.project.path, typed)
            except RequirementsError as exc:
                self.qa.say(f'  ! {exc}')

    def new_text(self, texts: list) -> list:
        """The requirements typed again ("edit"): the old ones are shown and replaced."""
        self.qa.say('The text was: ' + ' | '.join(texts))
        pieces = []
        while True:
            more = self.qa.line('    requirement (Enter: done): ').strip()
            if more.lower() in ('q', 'quit'):
                raise Quit
            if not more:
                break
            pieces.append(more)
        return pieces or list(texts)
