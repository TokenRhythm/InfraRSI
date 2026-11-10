"""ProjectHooks: the agent-loop seam (ext.common.SessionHooks) for any project described by infrarsi.yaml.

Knob proposals are completed with the user's current values, validated against the knob schema, the manifest's
constraints and its workload invariants (a candidate that changes the global batch is rejected before it costs a
run); patches are confined to the mutable files. The contract shown to the model is the verifier's: speed is the
only objective, the loss must stay within the tolerance, and the checklist decides validity.
"""
from __future__ import annotations

import json
import re

from ..agent.context import estimate_tokens
from ..ext.common.session_hooks import SessionHooks
from ..verify import CHECKS
from .signals import phases_brief
from .manifest import Project


def knob_phases(project: Project) -> dict:
    """knob -> the phase classes its mechanism can shorten ('all' or a list; experience/data/phases.yaml), for the
    knobs with a mapped mechanism."""
    from ..experience.registry import mechanism_phases
    from ..experience.store import labels
    out = {}
    for name, lab in labels(project.knobs, project.raw.get('framework')).items():
        v = mechanism_phases(lab.mechanism) if lab is not None else None
        if v is not None:
            out[name] = v
    return out


def phase_rows(ph: dict, framework: str | None) -> list[tuple]:
    """(phase, share of the update, class): the phases that partition an update (signals.phase_summary's `top`), the
    phases nested in them whose class none of those has (Megatron's all-grads-sync and params-all-gather inside
    forward-backward or the optimizer timer), and, from a profile run only (exclusive by construction), the GPU-idle
    rest of the update as 'residual': a residual from per-rank maxima bounds nothing."""
    from ..experience.registry import phase_class
    share = ph.get('share') or {}
    top = [k for k in (ph.get('top') or share) if k in share]
    rows = [(k, share[k], phase_class(k, framework)) for k in top]
    seen = {c for _, _, c in rows}
    rows += [(k, s, c) for k, s in share.items() if k not in top
             for c in [phase_class(k, framework)] if c is not None and c not in seen]
    if ph.get('source') == 'profile' and ph.get('step_ms') and (ph.get('residual_ms') or 0) > 0:
        rows.append(('residual', round(ph['residual_ms'] / ph['step_ms'], 4), 'residual'))
    return rows


def class_shares(ph: dict, framework: str | None) -> dict | None:
    """The update's share per phase class (a nested phase also counts in its parent's class: a looser cap, never a
    tighter one); None without phase shares."""
    out = {}
    for _, s, c in phase_rows(ph, framework):
        if c is not None:
            out[c] = out.get(c, 0.0) + s
    return out or None


def phase_targets(ph: dict, knobs: dict, framework: str | None) -> dict | None:
    """The run's three largest phases of the update's partition (the update minus them as 'residual' when positive),
    each with its class and the knobs whose mechanism can shorten it, and the knobs that act on every phase (layout,
    precision, recompute); None without phase shares."""
    rows = phase_rows(ph, framework)
    if not rows:
        return None
    largest = [{'phase': k, 'share': s, 'class': c, 'knobs': sorted(n for n, v in knobs.items() if v != 'all' and c in v)}
               for k, s, c in sorted(rows, key=lambda r: -r[1])[:3]]
    return {'largest': largest, 'any_phase': sorted(n for n, v in knobs.items() if v == 'all')}


# strategy optico: the manifest's `model:` facts under OptiCo's fixed_parameters.txt names (NUM_LAYERS, NUM_EXPERTS)
# and plain ones; the invariant that is the global batch; the precision a launch command fixes
MODEL_FACTS = (('layers', 'NUM_LAYERS'), ('experts', 'NUM_EXPERTS'), ('topk', 'Experts per Token'), ('hidden', 'Hidden Size'),
               ('ffn', 'FFN Hidden Size'), ('heads', 'Attention Heads'), ('kv_heads', 'KV Heads'), ('head_dim', 'Head Dim'),
               ('vocab', 'Vocabulary Size'))
GLOBAL_BATCH = re.compile(r'^(?:global_batch(?:_size)?|gbs)$', re.I)
PRECISION_FLAG = re.compile(r'(?<![\w-])--(bf16|fp16)(?:(?:\s+|\s*=\s*)(true|false|1|0)\b)?(?![\w-])', re.I)
PRECISION_VALUE = re.compile(r'(?<![\w-])--(?:mixed[_-]precision|precision)(?:\s+|\s*=\s*)["\']?([\w.-]+)', re.I)
PRECISION_KNOB = re.compile(r'precision|bf16|fp16|dtype', re.I)


def fixed_precision(project: Project) -> str | None:
    """The precision the job runs at when no knob searches it: the manifest's `states.precision`, else what its launch
    command sets (--bf16 / --fp16, as a switch or true; --mixed_precision / --precision VALUE); None otherwise."""
    for name, k in project.knobs.items():
        if any(PRECISION_KNOB.search(str(x or '')) for x in (name, k.get('flag'), k.get('override'), k.get('path'),
                                                              k.get('env'), k.get('const'), k.get('mechanism'))):
            return None
    if (project.raw.get('states') or {}).get('precision'):
        return str(project.raw['states']['precision'])
    cmd = str(project.raw['launch'].get('command') or '')
    for m in PRECISION_FLAG.finditer(cmd):
        if (m.group(2) or 'true').lower() in ('true', '1'):
            return m.group(1).lower()
    m = PRECISION_VALUE.search(cmd)
    return m.group(1) if m else None


class ProjectHooks(SessionHooks):
    tools_enabled = False

    def __init__(self, project: Project, task, *, tolerance: float | None = None, frozen_package_digest: str | None = None,
                 requirements: list[str] | None = None):
        self.project = project
        self.requirements = list(requirements or [])   # the user's requirements as the model is told them (project/requirements)
        self.tolerance = project.tolerance if tolerance is None else float(tolerance)
        mutable = {rel: t for rel in project.files(project.raw['code']['mutable']) if (t := project.read_text(rel)) is not None}
        readable = list(project.raw['code'].get('readable') or [])
        if project.config_file is not None:
            readable.append(project.config_file.relative_to(project.root).as_posix())
        ro = {rel: t for rel in project.files(readable) if rel not in mutable and (t := project.read_text(rel)) is not None}
        super().__init__(task, base_files=mutable, read_only_files=ro, frozen_package_digest=frozen_package_digest)
        self.reference_config = project.reference_config()
        self.reference_invariants = project.invariants(self.reference_config)
        self.knob_phases = knob_phases(project)
        self.reference_phases = None                   # the reference's phases from a profile run (when its log has none)
        self.reference_notes = None                    # the reference run's slow-path notes (signals.perf_note)

    def validate_config(self, config) -> tuple[dict, list, list]:
        if type(config) is not dict:
            return {}, ['config must be an object'], []
        merged = {**self.reference_config, **config}
        unknown = sorted(set(config) - set(self.project.knobs))
        if unknown:
            return {}, [f'unknown knobs {unknown} (knobs: {sorted(self.project.knobs)})'], []
        nulls = [k for k, v in config.items() if v is None and self.project.knobs[k].get('narrowed')   # a null would drop the
                 and self.reference_config.get(k) is not None]                                         # flag: the job's default
        if nulls:
            return {}, [f'{k}: unset is not an allowed value (your requirements narrowed it)' for k in nulls], []
        _, errs, warns = super().validate_config({k: v for k, v in merged.items() if v is not None})
        if not errs:
            errs = self.project.constraint_errors(merged)
        if not errs:
            inv = self.project.invariants(merged)
            diff = [f'{k} {inv[k]} != reference {self.reference_invariants.get(k)}' for k in inv
                    if inv[k] != self.reference_invariants.get(k)]
            if diff:
                errs = ['workload invariant changed (the training job would not be the same): ' + '; '.join(diff)]
        return (merged if not errs else {}), errs, warns

    def contract_text(self) -> str:
        p = self.project
        inv = ', '.join(f'{k} = {v}' for k, v in self.reference_invariants.items()) or 'none declared'
        return (f'Training project {p.name} ({p.gpus} GPU(s)). Goal: minimize the time per training update of this exact job; '
                'speed is the only objective. The verifier runs every candidate and checks the correctness checklist '
                f'against the reference run (the user\'s current configuration); the loss must stay within {self.tolerance:.1%} '
                f'of the reference. Workload invariants that must not change: {inv}. Only the listed knobs and mutable files '
                'may change; frozen files never. Instructions elsewhere that name Megatron-Core, L0 schema keys or '
                'runtime/overlays do not apply here; this contract does.'
                + ('\n' + '\n'.join(self.requirements) if self.requirements else ''))

    def verifier_checklist(self) -> list[str]:
        return [f'{c.id}: {c.title}' for c in CHECKS if c.when == 'screen' and c.family != 'info'] + \
               [f'The change has a concrete expected benefit on the step time given the parent receipt.']

    def receipt_digest(self, receipt, verdict=None, *, max_tokens: int = 1500, fields=None) -> dict:
        """The shared digest plus `step_phases` (where the update's time goes) next to the verdict, so that a long digest
        drops other blocks first (a run whose log shows none: the reference's profiled phases, marked as such); left
        out with `signals.phases_in_prompt: false`."""
        out = super().receipt_digest(receipt, verdict, max_tokens=max_tokens, fields=fields)
        head = {k: out[k] for k in ('id', 'verdict') if k in out}
        pri = {}
        # the job's own slow-path notes (signals.perf_note): this run's, else the reference run's, right after the verdict
        notes = (((receipt or {}).get('evidence') or {}).get('events') or {}).get('perf_notes') or self.reference_notes
        if notes and self.base_files:                  # only an arm that may edit code: knob-only prompts stay as they were
            pri['job_log_notes'] = list(notes)[:5]
        own = ((receipt or {}).get('evidence') or {}).get('phases')
        ph = own or self.reference_phases
        if ph and self.project.raw['signals'].get('phases_in_prompt', True):
            targets = phase_targets(ph, self.knob_phases, self.project.raw.get('framework'))
            pri['step_phases'] = (phases_brief(ph) if own else
                                  {**phases_brief(ph), 'of': 'the reference configuration (one profiled run)'})
            if targets:
                pri['phase_targets'] = targets
        if not pri:
            return out
        # one order (id, verdict, notes, phases, targets, the rest) and one cap: trailing blocks go first, never the
        # verdict, the notes or the phases; when those alone are too long, fewer notes
        out = {**head, **pri, **{k: v for k, v in out.items() if k not in head}}
        keep = len(head) + sum(1 for k in ('job_log_notes', 'step_phases') if k in pri)
        while estimate_tokens(json.dumps(out, default=str)) > max_tokens and len(out) > keep:
            out.pop(list(out)[-1])
        while estimate_tokens(json.dumps(out, default=str)) > max_tokens and len(out.get('job_log_notes') or []) > 1:
            out['job_log_notes'] = out['job_log_notes'][:-1]
        return out

    def class_shares(self, receipt) -> dict | None:
        """A receipt's update share per phase class (HLS's phase cap)."""
        ph = ((receipt or {}).get('evidence') or {}).get('phases') or self.reference_phases or {}
        return class_shares(ph, self.project.raw.get('framework'))

    def task_view(self) -> dict:
        v = super().task_view()
        v.update(reference_config=self.reference_config, tolerance=self.tolerance,
                 workload_invariants=self.reference_invariants, constraints=list(self.project.raw.get('constraints') or []),
                 hardware=self.project.raw['hardware'])
        if getattr(self, 'live_notes', None):
            v['user_guidance'] = list(self.live_notes)
        if self.requirements:
            v['requirements'] = list(self.requirements)
        return v

    # ------------------------------------------------------------ strategy optico (baselines.optico_style)
    def fixed_parameters_text(self) -> str:
        """OptiCo's FIXED_PARA block: the job's fixed facts in the form of OptiCo's fixed_parameters.txt - the model the
        manifest's `model:` describes, the precision, the global batch and the sequence length or the tokens per update,
        nodes, GPUs, GPU type and memory, framework - then what every run is held to (its updates and scoring window, the
        workload invariants, the constraints, the loss tolerance) and each knob's allowed values, which OptiCo's prompts
        name for no parameter they do not know. The facts the task view gives the other strategies and nothing about
        which values are good; a fact the manifest does not give is left out, never guessed."""
        from ..baselines.optico_style.optico import KnobCodec
        p, raw = self.project, self.project.raw
        m, wl, hw = raw.get('model') or {}, raw['workload'], raw['hardware']
        lines = [f'Model Type: {m.get("name") or m.get("family")}'] if m.get('name') or m.get('family') else []
        if m.get('params_b'):
            lines.append(f'Model Size: {float(m["params_b"]):g}B')
        lines += [f'{label}: {m[k]}' for k, label in MODEL_FACTS if m.get(k) is not None]
        precision = fixed_precision(p)
        if precision:
            lines.append(f'Precison: {precision}')       # OptiCo's own spelling (its run_monitor.sh writes this name)
        batch = next((v for k, v in self.reference_invariants.items() if GLOBAL_BATCH.match(k) and type(v) in (int, float)),
                     None)
        if batch is not None:
            lines.append(f'Global Batch Size: {int(batch) if float(batch).is_integer() else batch}')
        if wl.get('seq_len'):
            lines.append(f'Sequence Length: {wl["seq_len"]}')
        if self.tokens_per_update():
            lines.append(f'Tokens per Update: {self.tokens_per_update()}')
        lines += [f'Number of Nodes: {p.nodes}', f'Number of GPUs: {p.world}' + (f' ({p.gpus} per node)' if p.nodes > 1 else '')]
        if hw.get('gpu_type'):
            lines.append(f'GPU Type: {hw["gpu_type"]}')
        cap = self.gpu_memory_gib()
        if cap:
            lines.append(f'GPU memory limit per card: {round(cap * 1024)} MiB ({cap:g} GiB)')
        link = hw.get('interconnect')
        if link:
            lines.append('Interconnect: ' + (', '.join(f'{k} {v}' for k, v in link.items()) if isinstance(link, dict)
                                             else str(link)))
        if raw.get('framework'):
            lines.append(f'Framework: {raw["framework"]}')
        w, exprs = p.window, wl.get('invariants') or {}
        lines.append(f'Updates per run: {p.steps} (the step time is the mean over updates {w[0]}-{w[1]})')
        lines += [f'Must not change: {k} = {exprs.get(k)} = {v}' for k, v in self.reference_invariants.items()]
        if raw.get('constraints'):
            lines.append('Constraints (a configuration that breaks one is refused before it runs): '
                         + '; '.join(str(c) for c in raw['constraints']))
        lines += [f'Validity: a run counts only if it completes its {p.steps} updates of the same workload with the loss '
                  f'within {self.tolerance:.1%} of the reference run (the current configuration)',
                  'Parameters you may change and their allowed values:', KnobCodec(self.task).domain_text()]
        return '\n'.join(lines + self.requirements)

    def tokens_per_update(self) -> int | None:
        """The manifest's tokens per update (workload.global_batch_tokens), None when it gives none: the TokensPerSec
        column of OptiCo's table is then NULL, never derived."""
        v = self.project.raw['workload'].get('global_batch_tokens')
        return int(v) if type(v) in (int, float) and v > 0 else None

    def gpu_memory_gib(self) -> float | None:
        """hardware.memory_gib (what OptiCo's MemUsage% divides by), None when the manifest does not give it."""
        v = self.project.raw['hardware'].get('memory_gib')
        return float(v) if type(v) in (int, float) and v > 0 else None

    def memory_facts(self) -> dict:
        """strategy hls3 (S-1, S-10): what its memory arithmetic reads from the manifest - the card (hardware.memory_gib),
        every GPU of the job, the parameter count (model.params_b), what is trained (workload.trainable), and the knobs a
        workload invariant ties together (they move as one: micro-batch and accumulation)."""
        raw, knobs = self.project.raw, set(self.project.knobs)
        groups = []
        for expr in ((raw.get('workload') or {}).get('invariants') or {}).values():
            names = sorted(set(re.findall(r'[A-Za-z_]\w*', str(expr))) & knobs)
            if len(names) > 1 and names not in groups:
                groups.append(names)
        return {'cap_gib': self.gpu_memory_gib(), 'world': self.project.world,
                'params_b': (raw.get('model') or {}).get('params_b'), 'trainable': (raw.get('workload') or {}).get('trainable'),
                'coupled': groups}

    def skill_tags(self) -> list[str]:
        hw = self.project.raw['hardware']
        return sorted({'project', str(hw.get('gpu_type') or '').lower(), f'gpus{self.project.gpus}',
                       'single-node' if int(hw.get('nodes') or 1) == 1 else 'multi-node'} - {''})

    def skill_query(self) -> str:
        return ' '.join(self.skill_tags()) + ' step time parallel layout memory recompute data path kernels overlap'

    def memory_tags(self) -> dict:
        hw = self.project.raw['hardware']
        return {'hardware': [str(hw.get('gpu_type') or '').lower(), f'gpus{self.project.gpus}'],
                'model': [self.project.name], 'data': []}
