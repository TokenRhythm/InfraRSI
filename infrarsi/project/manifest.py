"""infrarsi.yaml: the one description of a training project InfraRSI optimizes (revamp design B).

    schema: infrarsi-project-v1
    name: my-sft
    workdir: .                                   # relative to this file
    launch: {command: "torchrun --nproc_per_node {gpus} train.py --config {config} --max-steps {steps} {args}",
             env: {}, setup: null, timeout_s: 3600}
    config: {file: configs/sft.yaml}             # the user's current config = the reference (yaml or json)
    knobs:                                       # target: path | flag | override | const | env
      micro_batch_size: {type: int, choices: [1, 2, 4], path: train.micro_batch_size}   # key in config.file
      compile: {type: bool, flag: --compile}                                          # --flag [value] in {args}
      fsdp_ac: {type: bool, override: fsdp_config.activation_checkpointing}           # a.b=value in {args} (Hydra, OmegaConf)
      DEVICE_BATCH_SIZE: {type: int, choices: [64, 128], const: DEVICE_BATCH_SIZE, file: train.py}   # NAME = value in a script
      NCCL_ALGO: {type: str, choices: [Ring, Tree], env: NCCL_ALGO}
    constraints: ["gpus % (tp * pp) == 0"]       # expressions over knobs and `gpus`; rejected before any run
    code: {mutable: ["src/kernels/*.py"], frozen: ["src/loss.py"], readable: ["train.py"], max_patch_lines: 300}
    signals: {source: log}                       # log (built-in key/value scanner, optional patterns) | probe
                                                 # phases: {fwd: 'fwd (?P<v>[0-9.]+)ms'}: per-update phase times (ms, or a
                                                 # `unit` group); framework megatron reads its timer block by default
    workload: {steps: 30, window: [11, 30], invariants: {global_batch: "micro_batch_size * grad_accum * gpus / (tp * pp)"}}
    checkpoint: {save: "--save {ckpt_dir} --save-at {save_at}", resume: "--load {ckpt_dir}", save_at: 10}
    hardware: {gpus: 8, memory_gib: 80}
    execution: {where: local}                    # local | ssh (hosts, remote_dir) | slurm (srun prefix); python: {python}
    verify: {tolerance: 0.03, required: [completes, same_workload, loss_within_tolerance], disabled: [], tests: [],
             eval: "{python} eval.py --load {ckpt_dir}"}   # eval on the saved checkpoint at promotion (prints eval_loss)
    search: {strategy: auto, budget: {screens: 16, wall_hours: null, gpu_hours: null, llm_tokens: null, llm_usd: null},
             job_gpu_hours: 5000, multi_fidelity: true,   # job_gpu_hours: what the production job still runs (value/cost
                                                          # stop; unset: the budget ends the search)
             run_noise: 0.02}   # between-run noise floor; unset: learned from earlier promotions' reference re-runs
    llm: {backend: null, model: null}

Everything but `name` and `launch.command` has a default. `Project.task()` is the ext.common.EnvTask the shared
search machinery runs on (knob schema, frozen-file hashes, mutable globs, budget).
"""
from __future__ import annotations

import ast
import copy
import fnmatch
import hashlib
import json
import math
import operator
import os
import re
from pathlib import Path

import yaml

SCHEMA = 'infrarsi-project-v1'
KNOB_TYPES = ('int', 'float', 'bool', 'str')
TARGETS = ('path', 'flag', 'override', 'const', 'env')
DEFAULT_EXCLUDE = ('.git', '.hg', '.svn', '__pycache__', '.infrarsi', '.venv', 'venv', 'node_modules', 'wandb', '*.pt',
                   '*.pth', '*.safetensors', '*.bin', '*.ckpt', '*.npy', '*.arrow', '*.parquet', 'checkpoints', 'outputs')
DEFAULTS = {
    'workdir': '.',
    'launch': {'command': None, 'env': {}, 'setup': None, 'timeout_s': 3600},
    'config': {'file': None},
    'knobs': {},
    'constraints': [],
    'code': {'mutable': [], 'frozen': [], 'readable': [], 'exclude': list(DEFAULT_EXCLUDE), 'max_patch_lines': 300},
    'signals': {'source': 'log', 'patterns': {}, 'ignore': r'(?i)\b(?:val(?:idation)?|eval(?:uation)?|test)\b',
                'count_steps': False, 'phases': {}, 'phases_in_prompt': True},
    'workload': {'steps': 30, 'window': None, 'invariants': {}},
    'checkpoint': None,
    'hardware': {'gpus': 1, 'memory_gib': None, 'gpu_type': None, 'nodes': 1},
    'execution': {'where': 'local', 'hosts': [], 'remote_dir': None, 'slurm_args': '', 'python': None,
                  'master_addr': None},       # ssh: node 0's address as the other nodes reach it ({master_addr})
    'verify': {'tolerance': 0.03, 'required': None, 'disabled': [], 'tests': [], 'eval': None},
    'search': {'strategy': 'auto', 'budget': {'screens': 16, 'wall_hours': None, 'gpu_hours': None,
                                              'llm_tokens': None, 'llm_usd': None},
               'job_gpu_hours': None, 'multi_fidelity': True, 'short_steps': None, 'probe_steps': 2, 'ablate': [],
               'phase_cap': False,        # research: HLS caps a prediction at its phases' share (P0-2 step 2)
               'on_interrupt': 'verify'},  # Ctrl-C / SIGTERM during the search: verify (re-verify the best so far) | stop
    'llm': {'backend': None, 'model': None, 'capability': None},
    # the dual-model debate at the proposal calls (agent/debate.py), off unless asked for; second / verifier: backend[:model]
    # or an llm-style mapping; verifier: null = the model that did not write the patch, 'none' = no patch check
    'debate': {'enabled': False, 'second': None, 'verifier': None, 'verify_rounds': 2},
}


class ManifestError(ValueError):
    pass


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


# ---------------------------------------------------------------- safe expressions (constraints, invariants)
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
        ast.GtE: operator.ge, ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}
_FUNCS = {'min': min, 'max': max, 'abs': abs, 'int': int, 'ceil': math.ceil, 'floor': math.floor}
MAX_EXPONENT = 64               # a constraint's `**` never needs more; a huge exponent would stall the load for minutes


def evaluate(expr: str, names: dict):
    """Arithmetic / comparison / boolean expression over `names` (knob values, gpus, nodes); nothing else."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and (n.value is None or isinstance(n.value, (int, float, str, bool))):
            return n.value
        if isinstance(n, ast.Name):
            if n.id not in names:
                raise ManifestError(f'unknown name {n.id!r} in {expr!r}')
            return names[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            left, right = ev(n.left), ev(n.right)
            if type(n.op) is ast.Pow and isinstance(right, (int, float)) and abs(right) > MAX_EXPONENT:
                raise ManifestError(f'exponent {right} too large in {expr!r} (at most {MAX_EXPONENT})')
            return _BIN[type(n.op)](left, right)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.Not)):
            return -ev(n.operand) if isinstance(n.op, ast.USub) else not ev(n.operand)
        if isinstance(n, ast.BoolOp):                   # short-circuit as Python does: `tp == 0 or gpus % tp == 0`
            want = isinstance(n.op, ast.Or)
            for v in n.values:
                if bool(ev(v)) == want:
                    return want
            return not want
        if isinstance(n, ast.Compare) and all(type(o) in _CMP for o in n.ops):
            left = ev(n.left)
            for op, c in zip(n.ops, n.comparators):
                right = ev(c)
                if not _CMP[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(n, ast.IfExp):
            return ev(n.body) if ev(n.test) else ev(n.orelse)
        if isinstance(n, (ast.Tuple, ast.List)):
            return [ev(e) for e in n.elts]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ManifestError(f'unsupported expression {ast.dump(n)[:80]} in {expr!r}')
    try:
        tree = ast.parse(str(expr), mode='eval')
    except SyntaxError as exc:
        raise ManifestError(f'cannot parse {expr!r}: {exc.msg}') from None
    return ev(tree)


def expr_names(expr: str) -> set[str]:
    """The names an expression reads (not the functions it calls: min, max, ...); none when it does not parse."""
    try:
        tree = ast.parse(str(expr), mode='eval')
    except SyntaxError:
        return set()
    calls = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and id(n) not in calls}


# ---------------------------------------------------------------- dotted paths in yaml/json configs
def get_path(doc, path: str, missing=None):
    cur = doc
    for part in str(path).split('.'):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return missing
    return cur


def set_path(doc: dict, path: str, value) -> None:
    parts = str(path).split('.')
    cur = doc
    for part in parts[:-1]:
        if isinstance(cur, list) and part.isdigit():
            cur = cur[int(part)]
            continue
        if not isinstance(cur.get(part), (dict, list)):
            cur[part] = {}
        cur = cur[part]
    if isinstance(cur, list) and parts[-1].isdigit():
        cur[int(parts[-1])] = value
    else:
        cur[parts[-1]] = value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class Project:
    """A loaded, validated manifest; paths are resolved against the manifest's directory."""

    def __init__(self, raw: dict, path: str | Path | None = None):
        self.path = Path(path).resolve() if path else None
        base = self.path.parent if self.path else Path.cwd()
        self.raw = _merge(DEFAULTS, raw or {})
        self.root = (base / self.raw['workdir']).resolve()
        errs = self.problems()
        if errs:
            raise ManifestError('; '.join(errs))

    @classmethod
    def load(cls, path: str | Path) -> 'Project':
        p = Path(path)
        if p.is_dir():
            p = p / 'infrarsi.yaml'
        if not p.is_file():
            raise ManifestError(f'{p} does not exist (`infrarsi init` drafts it)')
        try:
            doc = yaml.safe_load(p.read_text(encoding='utf-8'))
        except yaml.YAMLError as exc:
            raise ManifestError(f'{p} is not valid YAML: {exc}') from None
        if not isinstance(doc, dict):
            raise ManifestError(f'{p}: not a mapping')
        if doc.get('schema', SCHEMA) != SCHEMA:
            raise ManifestError(f'{p}: schema {doc.get("schema")!r} != {SCHEMA}')
        return cls(doc, p)

    # ------------------------------------------------------------ fields
    @property
    def name(self) -> str:
        return str(self.raw.get('name') or self.root.name)

    @property
    def knobs(self) -> dict:
        return self.raw['knobs'] or {}

    @property
    def gpus(self) -> int:
        """GPUs per node (what `{gpus}` renders, e.g. torchrun --nproc_per_node {gpus})."""
        return int(self.raw['hardware'].get('gpus') or 1)

    @property
    def nodes(self) -> int:
        """Nodes a run spans: every SSH host, else hardware.nodes."""
        e = self.raw['execution']
        return len(e.get('hosts') or []) or 1 if e.get('where') == 'ssh' else int(self.raw['hardware'].get('nodes') or 1)

    @property
    def world(self) -> int:
        return self.gpus * self.nodes

    @property
    def steps(self) -> int:
        return int(self.raw['workload']['steps'])

    @property
    def save_at(self) -> int | None:
        """The update after which a promotion's save run stops and saves (checkpoint.save_at, else half the steps); None
        without a checkpoint block. One rule for the save run's length, the {save_at} placeholder and the resume check:
        the runner used to halve the save run's own length and saved a quarter in."""
        ck = self.raw.get('checkpoint')
        return int(ck.get('save_at') or max(1, self.steps // 2)) if ck else None

    @property
    def window(self) -> list[int]:
        w = self.raw['workload'].get('window')
        if w:
            return [int(w[0]), int(w[1])]
        return [max(2, self.steps // 3 + 1), self.steps]

    @property
    def tolerance(self) -> float:
        return float(self.raw['verify']['tolerance'])

    @property
    def config_file(self) -> Path | None:
        f = self.raw['config'].get('file')
        return (self.root / f) if f else None

    def problems(self) -> list[str]:
        r, errs = self.raw, []
        from ..agent.budget import target_value
        budget, stop = r['search']['budget'], r['search'].get('stop') or {}
        if not isinstance(stop, dict):
            errs.append('search.stop must be a mapping')
            stop = {}
        for name in ('target_speedup', 'target_mfu'):
            try:
                budget[name] = target_value(name, budget.get(name) if budget.get(name) is not None else stop.get(name))
            except ValueError as exc:
                errs.append(str(exc))
        if r['search'].get('on_interrupt') not in ('verify', 'stop'):
            errs.append(f"search.on_interrupt must be 'verify' or 'stop', got {r['search'].get('on_interrupt')!r}")
        req = r.get('requirements')                     # optional and not in DEFAULTS: without it the raw manifest is unchanged
        if req is not None:
            errs += self._requirements_problems(req)
        d = r['debate'] if isinstance(r.get('debate'), dict) else {}
        if not isinstance(r.get('debate'), dict) or type(d.get('enabled')) is not bool:
            errs.append('debate must be a mapping whose `enabled` is true or false')
        if d.get('verify_rounds') not in (0, 1, 2) or type(d.get('verify_rounds')) is not int:
            errs.append(f'debate.verify_rounds must be 0, 1 or 2 (a patch is checked at most twice), got {d.get("verify_rounds")!r}')
        if not (r.get('launch') or {}).get('command'):
            errs.append('launch.command is required')
        if not self.root.is_dir():
            errs.append(f'workdir {self.root} does not exist')
        for name, k in (r.get('knobs') or {}).items():
            if not isinstance(k, dict):
                errs.append(f'knob {name}: must be a mapping')
                continue
            if k.get('type', 'str') not in KNOB_TYPES:
                errs.append(f'knob {name}: type {k.get("type")!r} not in {KNOB_TYPES}')
            if sum(t in k for t in TARGETS) != 1:
                errs.append(f'knob {name}: exactly one target of {TARGETS} required')
            if 'path' in k and not self.config_file:
                errs.append(f'knob {name}: a path target needs config.file')
            if 'const' in k and not (self.root / str(k.get('file') or '')).is_file():
                errs.append(f'knob {name}: a const target needs `file`, the script that defines {k["const"]}')
            if k.get('type') != 'bool' and not k.get('choices') and not ('min' in k and 'max' in k):
                errs.append(f'knob {name}: give choices or min/max')
        cf = self.config_file
        if cf is not None and not cf.is_file():
            errs.append(f'config.file {cf} does not exist')
        if cf is not None and cf.suffix.lower() not in ('.yaml', '.yml', '.json'):
            errs.append(f'config.file {cf.name}: only yaml or json configs are supported (use flag/env knobs otherwise)')
        w = r['workload']
        if type(w.get('steps')) is not int or w['steps'] < 2:
            errs.append('workload.steps must be an int >= 2')
        elif w.get('window') and not (1 <= int(w['window'][0]) <= int(w['window'][1]) <= w['steps']):
            errs.append(f'workload.window {w["window"]} must lie within 1..{w["steps"]}')
        tol = r['verify'].get('tolerance')
        if not (type(tol) in (int, float) and 0 <= tol < 1):
            errs.append('verify.tolerance must be a number in [0, 1)')
        if r['signals'].get('source') not in ('log', 'probe'):
            errs.append('signals.source must be log or probe')
        for name, rx in (r['signals'].get('phases') or {}).items():
            try:
                groups = re.compile(str(rx)).groupindex
            except re.error as exc:
                errs.append(f'signals.phases.{name}: not a regular expression ({exc})')
                continue
            if 'v' not in groups:
                errs.append(f'signals.phases.{name}: the pattern needs a named group v for the value, e.g. '
                            f"'{name} (?P<v>[\\d.]+) ?ms'")
        if r['execution'].get('where') not in ('local', 'ssh', 'slurm'):
            errs.append('execution.where must be local, ssh or slurm')
        if r['execution'].get('where') == 'ssh' and not r['execution'].get('hosts'):
            errs.append('execution.hosts is required for ssh')
        if r.get('checkpoint') and not all(r['checkpoint'].get(k) for k in ('save', 'resume')):
            errs.append('checkpoint needs save and resume argument templates')
        ck = r.get('checkpoint') or {}
        if 'save_at' in ck and not (type(ck['save_at']) is int and type(w.get('steps')) is int and 1 <= ck['save_at'] < w['steps']):
            errs.append(f'checkpoint.save_at must be an int from 1 to workload.steps - 1 (got {ck["save_at"]!r}): the resumed '
                        'run needs updates after the checkpoint')
        from ..search import ABLATIONS
        bad = [x for x in r['search'].get('ablate') or [] if x not in ABLATIONS]
        if bad or not isinstance(r['search'].get('ablate') or [], list):
            errs.append(f'search.ablate: a list of {sorted(ABLATIONS)} (got {r["search"].get("ablate")!r})')
        if not isinstance(r['search'].get('phase_cap', False), bool):
            errs.append(f'search.phase_cap: true or false (got {r["search"]["phase_cap"]!r})')
        if not errs:                                    # a typo would otherwise compare 'error: ...' with 'error: ...'
            doc = self.base_config()
            ref = self.reference_config(doc)
            names = self.names(ref)
            exprs = [(f'workload.invariants.{n}', e) for n, e in (w.get('invariants') or {}).items()]
            for where, expr in exprs + [(f'constraint {e!r}', e) for e in r.get('constraints') or []]:
                try:
                    evaluate(expr, names)                   # an expression may handle an unset (None) value itself
                    continue
                except (ManifestError, ArithmeticError, TypeError, ValueError) as exc:
                    err = exc
                unknown = sorted(n for n in expr_names(expr) if n in self.knobs and ref.get(n) is None)
                if unknown:                             # U-7-1: a typo in a config path reads as "no value" too
                    typo = [f'{self.config_file.name} has no key {self.knobs[n]["path"]} (knobs.{n}.path)' for n in unknown
                            if 'path' in self.knobs[n] and isinstance(doc, dict)
                            and get_path(doc, self.knobs[n]['path'], missing=...) is ...]
                    errs.append(f'{where} uses {", ".join(unknown)}, whose current value is unknown: '
                                + (f'{"; ".join(typo)} - check the path, or give ' if typo else 'give ')
                                + f'{"each" if len(unknown) > 1 else "it"} a default (the value the job runs with now)')
                elif where.startswith('workload'):
                    errs.append(f'{where}: {err}')
                elif isinstance(err, ManifestError):    # a constraint naming no knob (one the user removed), unparseable
                    errs.append(f'{where}: {err} (the knobs, gpus and nodes can be used: remove the constraint or declare '
                                'the knob)')
        return errs

    @staticmethod
    def _requirements_problems(req) -> list[str]:
        """The `requirements:` block (project/requirements.py): the user's words, and what a run does with an item it cannot honour."""
        if not isinstance(req, dict):
            return ['requirements must be a mapping (text, on_unsupported)']
        errs = [f'requirements: unknown key {k} (text, on_unsupported)' for k in req if k not in ('text', 'on_unsupported')]
        text = req.get('text')
        pieces = [text] if isinstance(text, str) else text
        if not isinstance(pieces, list) or not pieces or not all(isinstance(t, str) and t.strip() for t in pieces):
            errs.append('requirements.text must be a non-empty string or a list of them')
        if req.get('on_unsupported', 'stop') not in ('stop', 'continue'):
            errs.append("requirements.on_unsupported must be 'stop' or 'continue'")
        return errs

    # ------------------------------------------------------------ configs
    def base_config(self) -> dict | None:
        cf = self.config_file
        if cf is None:
            return None
        text = cf.read_text(encoding='utf-8')
        try:
            return json.loads(text) if cf.suffix.lower() == '.json' else (yaml.safe_load(text) or {})
        except (ValueError, yaml.YAMLError) as exc:
            raise ManifestError(f'config.file {cf} cannot be parsed: {exc}') from None

    _UNREAD = object()                                  # reference_config(doc): not given, unlike None (a config holding null)

    def reference_config(self, doc=_UNREAD) -> dict:
        """The user's current knob values: from the config file (path knobs, and override knobs whose key it has) or
        the script (const knobs), else the knob's `default`; an override absent everywhere stays unset (None), and one
        the config sets to null stays null (not replaced by the default). `doc`: the config file's content when the
        caller has read it already (None when the file holds null)."""
        doc = self.base_config() if doc is self._UNREAD else doc
        out = {}
        for name, k in self.knobs.items():
            v = get_path(doc, k['path']) if 'path' in k and doc is not None else None
            if 'override' in k and doc is not None:
                v = get_path(doc, k['override'], missing=...)
                if v is None:
                    out[name] = None
                    continue
                v = None if v is ... else v
            if 'const' in k and v is None:
                v = read_const(self.root / k['file'], k['const'])
            if v is None:
                v = k.get('default')
            if v is None and k.get('type') == 'bool' and 'override' not in k:
                v = False                               # an absent flag
            out[name] = v
        return out

    def names(self, config: dict) -> dict:
        return {**{k: config.get(k) for k in self.knobs}, 'gpus': self.gpus, 'nodes': self.nodes}

    def crlf_scripts(self) -> list[str]:
        """Shell scripts in the workdir that launch.command or launch.setup runs and that have Windows (CRLF) line
        endings: bash on a POSIX host reads every line with a trailing \\r (`...\\r: No such file or directory`)."""
        out = []
        for word in ' '.join(str(self.raw['launch'].get(k) or '') for k in ('command', 'setup')).split():
            f = (self.root / word.strip('\'"')).resolve()
            if f.suffix == '.sh' and f.is_relative_to(self.root) and f.is_file() and b'\r\n' in f.read_bytes():
                out.append(f.relative_to(self.root).as_posix())
        return sorted(set(out))

    def posix_hosts(self) -> bool:
        """The job runs on POSIX hosts: over ssh, under srun, or locally on a POSIX machine."""
        return self.raw['execution'].get('where') in ('ssh', 'slurm') or os.name != 'nt'

    def constraint_errors(self, config: dict) -> list[str]:
        errs = []
        for expr in self.raw.get('constraints') or []:
            try:
                if not evaluate(expr, self.names(config)):
                    errs.append(f'constraint violated: {expr}')
            except (ManifestError, ArithmeticError, TypeError, ValueError) as exc:
                errs.append(f'constraint {expr!r} cannot be evaluated: {exc}')
        return errs

    def invariants(self, config: dict) -> dict:
        """Workload invariants of a config (e.g. the global batch), compared with the reference's by the verifier."""
        out = {}
        for name, expr in (self.raw['workload'].get('invariants') or {}).items():
            try:
                out[name] = evaluate(expr, self.names(config))
            except (ManifestError, ArithmeticError, TypeError, ValueError) as exc:
                out[name] = f'error: {exc}'
        return out

    def knob_schema(self) -> dict:
        props = {}
        for name, k in self.knobs.items():
            t = k.get('type', 'str')
            s = {'type': {'int': 'integer', 'float': 'number', 'bool': 'boolean', 'str': 'string'}[t]}
            if k.get('choices'):
                s['enum'] = list(k['choices'])
            if 'min' in k:
                s['minimum'] = k['min']
            if 'max' in k:
                s['maximum'] = k['max']
            if k.get('description'):
                s['description'] = str(k['description'])[:300]
            props[name] = s
        return {'type': 'object', 'properties': props, 'additionalProperties': False}

    # ------------------------------------------------------------ files
    def excluded(self, rel: str) -> bool:
        parts = rel.split('/')
        return any(fnmatch.fnmatchcase(p, g) for g in self.raw['code']['exclude'] for p in parts)

    def files(self, globs) -> list[str]:
        """Relative POSIX paths of the regular files under the workdir that match one of `globs`."""
        out = []
        for g in globs or ():
            for p in self.root.glob(g):
                if p.is_file():
                    rel = p.relative_to(self.root).as_posix()
                    if not self.excluded(rel):
                        out.append(rel)
        return sorted(set(out))

    def frozen_hashes(self, root: Path | None = None) -> dict:
        base = root or self.root
        return {rel: sha256(base / rel) for rel in self.files(self.raw['code']['frozen'])}

    def read_text(self, rel: str) -> str | None:
        p = self.root / rel
        try:
            return p.read_text(encoding='utf-8') if p.is_file() and p.stat().st_size <= 262144 else None
        except UnicodeDecodeError:
            return None

    # ------------------------------------------------------------ the shared task contract
    def task(self):
        from ..ext.common.env_task import EnvTask
        mutable = list(self.raw['code']['mutable'] or [])
        if not self.knobs and not mutable:
            raise ManifestError('nothing to search: declare knobs or code.mutable files')
        space = 'knobs+patch' if self.knobs and mutable else ('patch' if mutable else 'knobs')
        b = self.raw['search']['budget']
        screens = int(b.get('screens') or 16)
        # The engine uses finite numbers for arithmetic; None means no user cap.
        from ..agent.budget import UNLIMITED
        wall = float(b['wall_hours']) * 3600 if b.get('wall_hours') is not None else UNLIMITED
        gpu_hours = b.get('gpu_hours')
        if gpu_hours is None:
            gpu_hours = UNLIMITED
        spec = {'schema': 'infrarsi-ext-task-v1', 'id': self.name, 'env': 'project', 'split': 'project',
                'metric_name': 'step_time_s', 'action_space': space, 'knob_schema': self.knob_schema() if self.knobs else {},
                'frozen_files': self.frozen_hashes(), 'mutable_globs': mutable,
                'max_patch_lines': int(self.raw['code'].get('max_patch_lines') or 300),
                'budget': {'screens': screens, 'llm_tokens': int(b['llm_tokens']) if b.get('llm_tokens') is not None else UNLIMITED, 'wall_seconds': wall,
                           'gpu_hours': float(gpu_hours), 'llm_usd': b.get('llm_usd'),
                           'target_speedup': b.get('target_speedup'), 'target_mfu': b.get('target_mfu')},
                'margin': {'kind': 'rsd', 'floor': 0.03, 'k': 3}, 'tau': math.log(1.10), 'world_size': self.world,
                'facts': {'hardware': dict(self.raw['hardware']), 'workload': {'steps': self.steps, 'window': self.window},
                          'project': self.name, 'tolerance': self.tolerance},
                'manifest': self.raw, 'manifest_path': str(self.path) if self.path else None}
        return EnvTask.from_dict(spec)

    def describe(self) -> dict:
        return {'name': self.name, 'root': str(self.root), 'knobs': sorted(self.knobs), 'gpus': self.gpus, 'world': self.world,
                'steps': self.steps, 'window': self.window, 'tolerance': self.tolerance,
                'mutable': self.raw['code']['mutable'], 'frozen': self.raw['code']['frozen'],
                'signals': self.raw['signals']['source'], 'execution': self.raw['execution']['where']}


CONST_RE = r'^(?P<lead>\s*){name}\s*(?::[^=]+)?=\s*(?P<value>[^#\n]*?)(?P<tail>\s*(?:#.*)?)$'


def read_const(path: Path, name: str):
    """The literal value of a module-level `NAME = <literal>` assignment (None when absent or not a literal)."""
    import re
    try:
        text = Path(path).read_text(encoding='utf-8')
    except OSError:
        return None
    m = re.search(CONST_RE.format(name=re.escape(name)), text, re.M)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group('value').strip())
    except (ValueError, SyntaxError):
        return None


def write_const(text: str, name: str, value) -> str:
    """Replace the first module-level assignment of NAME by `NAME = repr(value)`, keeping its trailing comment."""
    import re
    rx = re.compile(CONST_RE.format(name=re.escape(name)), re.M)
    if not rx.search(text):
        raise ManifestError(f'{name} is not assigned in the script')
    return rx.sub(lambda m: f'{m.group("lead")}{name} = {value!r}{m.group("tail")}', text, count=1)


def env_for(project: Project, config: dict) -> dict:
    """Environment variables of the env-target knobs (None or False unsets nothing and sets nothing)."""
    out = {str(k): str(v) for k, v in (project.raw['launch'].get('env') or {}).items()}
    for name, k in project.knobs.items():
        if 'env' in k:
            v = config.get(name)
            if v is None:
                continue
            out[str(k['env'])] = ('1' if v else '0') if isinstance(v, bool) else str(v)
    return out


def args_for(project: Project, config: dict) -> list[str]:
    """Command-line flags of the flag-target knobs: bool flags appear when true (style store_true) unless the knob
    says `style: value` (then `--flag true|false`); other knobs render as `--flag value` (None: omitted)."""
    out = []
    for name, k in project.knobs.items():
        if 'override' in k and config.get(name) is not None:
            v = config[name]
            out.append(f'{k["override"]}={str(v).lower() if isinstance(v, bool) else v}')
            continue
        if 'flag' not in k:
            continue
        v = config.get(name)
        if v is None:
            continue
        if isinstance(v, bool) and k.get('style', 'switch') == 'switch':
            if v:
                out.append(str(k['flag']))
            elif k.get('off_flag'):
                out.append(str(k['off_flag']))
            continue
        out += [str(k['flag']), (str(v).lower() if isinstance(v, bool) else str(v))]
    return out

