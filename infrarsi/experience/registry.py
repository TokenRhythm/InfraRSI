"""The infrarsi-exp/1 registries (data/mechanisms.yaml, data/gpus.yaml): a mechanism is one idea that exists in several
codebases, with a transfer class (kernel: the sign transfers across GPU classes; balance: a size needs the same GPU class
and layout) and the knobs that set it; a GPU class maps the names nvidia-smi prints to the numbers a transfer judgment
compares. `label` maps a project knob to (mechanism, state of each knob value)."""
from __future__ import annotations

import dataclasses
import re
from functools import lru_cache
from pathlib import Path

import yaml

DATA = Path(__file__).with_name('data')
TRUE = {'1', 'true', 'yes', 'on'}


def norm(s) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(s).lower()).strip('_')


GENERIC_STATES = {'on', 'off', 'none', 'auto', 'true', 'false', '0', '1', 'default'}


def state_owners(name: str, limit: int = 3) -> list[str]:
    """The registry mechanisms with a state named like this knob, in registry order (`bf16` -> precision, ...): a hint
    for a knob no alias maps - its mechanism and states are the user's to declare (F1-S1: an unmapped `bf16` switch left
    the library's precision evidence unused, x1.52; declared as precision, x3.52)."""
    n = norm(name)
    if n in GENERIC_STATES:
        return []
    return [m for m, d in mechanisms().items() if isinstance(d.get('states'), list) and n in [norm(s) for s in d['states']]][:limit]


@lru_cache(maxsize=None)
def mechanisms() -> dict:
    return yaml.safe_load((DATA / 'mechanisms.yaml').read_text(encoding='utf-8')) or {}


@lru_cache(maxsize=None)
def gpus() -> dict:
    return yaml.safe_load((DATA / 'gpus.yaml').read_text(encoding='utf-8')) or {}


@lru_cache(maxsize=None)
def phases() -> dict:
    """data/phases.yaml: the class of each step timer (per framework) and of a phase name, and the phase classes each
    mechanism can shorten ('all', 'residual' or a list)."""
    return yaml.safe_load((DATA / 'phases.yaml').read_text(encoding='utf-8')) or {}


def phase_class(name: str, framework: str | None = None) -> str | None:
    """The class of a logged phase: the framework's timer table, else the first class whose word is in the name."""
    p = phases()
    if name in ((p.get('timers') or {}).get(framework or '') or {}):
        return p['timers'][framework][name]
    low = re.sub(r'[\s_]+', '-', str(name).lower())
    return next((c for c, words in (p.get('names') or {}).items() if any(w in low for w in words)), None)


def mechanism_phases(mech: str | None) -> list | str | None:
    """The phase classes a mechanism can shorten: 'all', ['residual'], a list of classes, or None (not mapped)."""
    v = (phases().get('mechanisms') or {}).get(mech) if mech else None
    return v if v in (None, 'all') or isinstance(v, list) else [v]


def known(mech: str) -> bool:
    return mech in mechanisms() or mech.startswith('x-')


def gpu_class(name) -> str | None:
    """'NVIDIA L20X', 'H200', 'H200-141GB' -> 'H200-141GB'; None when no class matches."""
    if not name:
        return None
    n = norm(name)
    for cls, g in gpus().items():
        if n == norm(cls) or n in {norm(a) for a in g.get('aliases') or []}:
            return cls
    for cls in gpus():                                  # 'h200' inside a longer name, but 'h20' not inside 'h200'
        if re.search(rf'(?<![a-z0-9]){norm(cls.split("-")[0])}(?![0-9])', n):
            return cls
    return None


def _truthy(v) -> bool:
    return v is True or (not isinstance(v, bool) and str(v).strip().lower() in TRUE)


@dataclasses.dataclass(frozen=True)
class Label:
    mechanism: str
    cls: str                                            # kernel | balance | (x-: unknown)
    states: dict                                        # knob value (as str) -> mechanism state; empty: str(value) is the state
    own: bool = False                                   # the framework's own flag / env var for the mechanism (a registry
                                                        # alias), not a project switch that may set more (e.g. what it requires)
    allowed: tuple = ()                                 # the registry's states (empty: any, e.g. an int mechanism)
    implies: dict = dataclasses.field(default_factory=dict)   # knob value (as str) -> {mechanism: state} it also sets

    def state(self, value) -> str | None:
        """The mechanism state of a knob value; None when unknown or not one of the registry's states."""
        if value is None:
            return None
        key = str(value).lower() if isinstance(value, bool) else str(value)
        st = self.states.get(key, key if not self.states else None)
        return None if st is not None and self.allowed and st not in self.allowed and not st.startswith('x-') else st


def _alias_states(alias: dict, knob: dict, mech: dict) -> dict:
    """knob value -> state for a knob that matches a registry alias."""
    choices = [str(c).lower() if isinstance(c, bool) else str(c) for c in (knob.get('choices') or [])]
    if knob.get('type') == 'bool' or not choices:
        choices = choices or ['true', 'false']
    if 'present' in alias:                              # a switch flag: on -> the alias's state, off -> the default
        return {c: (alias['present'] if _truthy(c) else str(mech.get('default'))) for c in choices}
    if 'map' in alias:
        return {c: str(alias['map'][c]) for c in choices if c in alias['map']}
    return {}                                           # value: the knob's value is the state


def label(name: str, knob: dict, framework: str | None = None) -> Label | None:
    """The mechanism a manifest knob sets: its explicit `mechanism` (with optional `states`: value -> state), else the
    registry alias of its flag / env var / config path, else its name when it is a mechanism id; None when unknown.
    The knob's `implies` (value -> {mechanism: state}) is what a value also sets on other mechanisms: a fixed coupling
    in the job's scripts, e.g. OptiCo's sequence parallelism with TP > 1."""
    lab = _label(name, knob, framework)
    imp = knob.get('implies') or {}
    if lab is None or not imp:
        return lab
    key = lambda v: str(v).lower() if isinstance(v, bool) else str(v)   # noqa: E731 - as Label.state keys a value
    return dataclasses.replace(lab, implies={key(v): {str(m): str(s) for m, s in d.items()} for v, d in imp.items()})


def _label(name: str, knob: dict, framework: str | None) -> Label | None:
    reg = mechanisms()

    def allowed(m):
        st = (reg.get(m) or {}).get('states')
        return tuple(str(s) for s in st) if isinstance(st, list) else ()
    if knob.get('mechanism'):
        m = str(knob['mechanism'])
        states = {str(k).lower() if isinstance(k, bool) else str(k): str(v) for k, v in (knob.get('states') or {}).items()}
        return Label(m, (reg.get(m) or {}).get('class', 'x-'), states, allowed=allowed(m))
    target = {'flag': knob.get('flag'), 'env': knob.get('env'), 'config': knob.get('path') or knob.get('override')}
    for m, d in reg.items():
        for fw, aliases in (d.get('aliases') or {}).items():
            if fw not in ('any', framework) and framework is not None:
                continue
            for a in aliases:
                for kind in ('flag', 'env', 'config'):
                    if a.get(kind) and target[kind] and (target[kind] == a[kind] if kind != 'config'
                                                         else str(target[kind]).split('.')[-1] == str(a[kind]).split('.')[-1]):
                        return Label(m, d.get('class', 'x-'), _alias_states(a, knob, d), own=True, allowed=allowed(m))
    m = norm(name)
    if m in reg:                                        # named after the mechanism: its values by what they mean
        d, regs = reg[m], allowed(m)
        choices = [str(c).lower() if isinstance(c, bool) else str(c) for c in (knob.get('choices') or [])] or \
            (['false', 'true'] if knob.get('type') == 'bool' else [])
        if (choices and set(choices) <= {'false', 'true', '0', '1'} and not set(choices) <= set(regs) and len(regs) == 2
                and d.get('default') is not None):         # a switch: true / 1 turns the mechanism on
            off = 'off' if set(regs) == {'off', 'on'} else str(d['default'])
            on = next(s for s in regs if s != off)
            return Label(m, d.get('class', 'x-'), {'true': on, '1': on, 'false': off, '0': off}, allowed=regs)
        return Label(m, d.get('class', 'x-'), {}, allowed=regs)   # its own state names; another value has none
    return None
