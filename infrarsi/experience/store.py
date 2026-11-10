"""The experience store: the infrarsi-exp/1 packs (the shipped seeds, the packs under INFRARSI_HOME/experience/, among
them InfraRSI's own written-back runs) in memory, and what the search needs from them for one job (experience
decision, sections 1-2):

    fingerprint(project, receipt)   the target: GPU class, nodes, framework, versions, model size, workload, layout and
                                    the reference run's mechanism states (Megatron's argument dump)
    Store.coverage(fp, knobs, ref)  per knob: covered | null | partial | none, with the nearest evidence
    plan(store, fp, knobs, ref)     the seed rows, the start composite, the feasibility check and the LLM payload
    measured(framework)             mechanisms the shipped packs measured faster on a framework (what `init` drafts)
    unreached(framework, knobs)     levers measured on the framework that no knob of a project sets (`init` names them)

Coverage is per mechanism: kernel-class effects transfer across GPU classes in sign (size within about 2x); balance-class
effects need the same GPU class, nodes, model size bucket and layout for a numeric prior. Experience orders the search;
it never accepts, rejects or stops anything (that stays with the engine's measured noise and the verifier)."""
from __future__ import annotations

import fnmatch
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path

from .pack import derive_effects, full_config, gain, read_lines, validate
from .registry import gpu_class, gpus, label, mechanisms, norm, state_owners

SEED = Path(__file__).with_name('seed')
TIER = {'verified': 0, 'measured': 1, 'reported': 2}
FLOOR = {'infrarsi': 0.01, 'paired_runs': 0.01, 'independent_runs': 0.02}
LEVELS = ('covered', 'null', 'partial', 'none')
HARD = ('crash', 'oom', 'invalid')                 # constraint symptoms that make a change probe-first ('slow' only warns)
HEADROOM = {'megatron': 2.5, 'fsdp': 7.0, 'hf_trainer': 7.0}   # GiB a run needs above its measured peak
# the states that set the memory regime (as accounting.READS): a measurement matches a configuration only when they agree
SHAPE = ('tp', 'pp', 'cp', 'ep', 'etp', 'micro_batch', 'precision', 'recompute', 'bf16_main_grads', 'bf16_moments',
         'distributed_optimizer', 'sequence_parallel', 'zero_stage', 'fsdp_sharding', 'fsdp_version', 'hsdp',
         'reshard_after_forward', 'megatron_fsdp', 'cpu_offload', 'qlora', 'loss_impl', 'moe_layer_recompute', 'nvls',
         'fused_cross_entropy')
VERSION_KEYS = {'transformer_engine': 'te', 'transformer-engine': 'te', 'megatron_core': 'framework',
                'megatron-core': 'framework', 'mcore': 'framework', 'pytorch': 'torch'}
CONFIDENCE = {'high': 0, 'medium': 1, 'low': 2}
LAYOUT = ('tp', 'pp', 'cp', 'ep')
UNKNOWN = '?'
ESTIMATE = 'memory (estimate)'                  # check()'s warning from the closed-form accounting
ARGS = {'num_layers': 'layers', 'hidden_size': 'hidden', 'ffn_hidden_size': 'ffn', 'num_attention_heads': 'heads',
        'num_query_groups': 'kv_heads', 'padded_vocab_size': 'vocab', 'vocab_size': 'vocab_raw', 'seq_length': 'seq_len',
        'global_batch_size': 'gbs', 'micro_batch_size': 'mbs', 'tensor_model_parallel_size': 'tp',
        'pipeline_model_parallel_size': 'pp', 'context_parallel_size': 'cp', 'expert_model_parallel_size': 'ep',
        'num_experts': 'experts', 'kv_channels': 'head_dim', 'untie_embeddings_and_output_weights': 'untie',
        'moe_router_topk': 'topk'}
SHAPE_KEYS = ('layers', 'hidden', 'ffn', 'heads', 'kv_heads', 'head_dim', 'vocab', 'experts', 'topk', 'untie')
ARG_RE = re.compile(r'^\s*(\w+) \.{3,} (\S+)\s*$')
ONOFF = {'True': 'on', 'False': 'off'}
DTYPE = {'torch.float32': 'fp32', 'torch.bfloat16': 'bf16'}
STATE_ARGS = {   # Megatron argument -> (mechanism, printed value -> state; None: the value is the state)
    'gradient_accumulation_fusion': ('gradient_accumulation_fusion', ONOFF),
    'use_distributed_optimizer': ('distributed_optimizer', ONOFF),
    'use_precision_aware_optimizer': ('precision_aware_optimizer', ONOFF), 'overlap_grad_reduce': ('dp_grad_overlap', ONOFF),
    'overlap_param_gather': ('dp_param_gather_overlap', ONOFF), 'cross_entropy_loss_fusion': ('fused_cross_entropy', ONOFF),
    'cross_entropy_fusion_impl': ('cross_entropy_fusion_impl', None), 'sequence_parallel': ('sequence_parallel', ONOFF),
    'tp_comm_overlap': ('tp_comm_overlap', ONOFF), 'manual_gc': ('manual_gc', ONOFF),
    'check_for_nan_in_loss_and_grad': ('nan_check', ONOFF), 'main_grads_dtype': ('bf16_main_grads', DTYPE),
    'exp_avg_dtype': ('bf16_moments', DTYPE), 'ddp_average_in_collective': ('ddp_average_in_collective', ONOFF),
    'attention_backend': ('attention_backend', {'AttnBackend.auto': 'auto', 'AttnBackend.flash': 'flash',
                                                'AttnBackend.fused': 'cudnn', 'AttnBackend.unfused': 'unfused'}),
    'recompute_granularity': ('recompute', {'None': 'none', 'selective': 'selective', 'full': 'full'}),
    'empty_unused_memory_level': ('empty_unused_memory', None), 'num_workers': ('data_workers', None),
    'micro_batch_size': ('micro_batch', None), 'tensor_model_parallel_size': ('tp', None),
    'pipeline_model_parallel_size': ('pp', None), 'context_parallel_size': ('cp', None),
    'expert_model_parallel_size': ('ep', None), 'expert_tensor_parallel_size': ('etp', None)}


def home() -> Path:
    return Path(os.environ.get('INFRARSI_HOME') or Path.home() / '.infrarsi')


def arg_dump(text: str) -> dict:
    """Megatron's argument dump (`  num_layers ........ 32`, one line per argument): name -> printed value."""
    raw = {}
    for line in text.splitlines():
        m = ARG_RE.match(line.split('\t')[-1])
        if m:
            raw.setdefault(m.group(1), m.group(2))
    return raw


def megatron_args(raw: dict) -> dict:
    out = {}
    for arg, field in ARGS.items():
        v = raw.get(arg)
        if v not in (None, 'None'):
            out[field] = (v == 'True') if v in ('True', 'False') else int(v) if v.isdigit() else v
    return out


def megatron_states(raw: dict) -> dict:
    """The reference run's mechanism states (its fixed launch flags included)."""
    out = {}
    for arg, (mech, conv) in STATE_ARGS.items():
        v = raw.get(arg)
        st = None if v is None else conv.get(v) if conv else (None if v == 'None' else v)
        if st is not None:
            out[mech] = st
    if raw.get('fp8') not in (None, 'None'):
        out['precision'] = f'fp8_{raw.get("fp8_recipe", "delayed")}'
    elif raw.get('fp16') == 'True' or raw.get('bf16') == 'True':
        out['precision'] = 'fp16' if raw.get('fp16') == 'True' else 'bf16'
    return out


def versions(v: dict, framework: str | None) -> dict:
    """Version keys as the packs' scopes use them: te, framework (the framework's own package), torch, cudnn dotted
    (cuDNN's integer 90100 -> 9.1.0, 8902 -> 8.9.2)."""
    out = {}
    for k, x in (v or {}).items():
        key, s = VERSION_KEYS.get(str(k).lower(), str(k).lower()), str(x)
        if key == 'cudnn' and s.isdigit():
            n = int(s)
            s = f'{n // 10000}.{n // 100 % 100}.{n % 100}' if n >= 90000 else f'{n // 1000}.{n // 100 % 10}.{n % 100}'
        out[key] = s
        if key == framework:                    # e.g. deepspeed: its own version is the framework version
            out.setdefault('framework', s)
    return out


def band(bw) -> str | None:
    """A multi-node fabric's bandwidth bucket (all-reduce bus bandwidth, GB/s; textbook design P2-2): the H20 hosts' TCP
    near 3, unicom's TCP 6-21 after socket tuning, InfiniBand above 100; None when not measured."""
    if not isinstance(bw, (int, float)) or bw <= 0:
        return None
    return '<5' if bw < 5 else '5-25' if bw < 25 else '25-100' if bw < 100 else '>=100'


def _same_band(env: dict, fp: dict) -> bool:
    """Two multi-node environments' bandwidth buckets agree, or one of them was not measured (no claim then)."""
    a, b = (band((x.get('interconnect') or {}).get('bandwidth_gbs')) for x in (env, fp))
    return a is None or b is None or a == b


def interconnect(v) -> dict:
    """hardware.interconnect: {intra, inter} or just the inter-node fabric (tcp, ib, roce); {} when not given."""
    return dict(v) if isinstance(v, dict) else {'inter': str(v)} if v else {}


def params_b(a: dict) -> float | None:
    """Parameters of a Llama / Qwen-style decoder from its shape (SwiGLU MLP, GQA), in billions."""
    try:
        h, layers, ffn, heads = a['hidden'], a['layers'], a['ffn'], a['heads']
    except KeyError:
        return None
    hd = a.get('head_dim') or h // heads
    attn = h * heads * hd * 2 + h * (a.get('kv_heads') or heads) * hd * 2
    mlp = 3 * h * ffn * (a.get('experts') or 1)
    emb = (a.get('vocab') or a.get('vocab_raw') or 0) * h * (2 if a.get('untie', True) else 1)
    return round((layers * (attn + mlp) + emb) / 1e9, 2)


def fingerprint(project, receipt: dict | None = None) -> dict:
    """The target environment from the manifest (`model:`, `versions:` and `workload.seq_len / data` are optional
    sections) and, when given, the reference run's log (Megatron's argument dump)."""
    hw, raw = project.raw['hardware'], project.raw
    dump = {}
    log = (receipt or {}).get('log')
    if log and Path(log).is_file():
        dump = arg_dump(Path(log).read_text(encoding='utf-8', errors='replace'))
    a = megatron_args(dump)
    model = dict(raw.get('model') or {})
    if model.get('params_b') is None:
        model['params_b'] = params_b(a)
    model.setdefault('family', 'moe' if a.get('experts') else 'dense')
    shape = {**a, 'vocab': a.get('vocab') or a.get('vocab_raw')}
    for k in SHAPE_KEYS:                        # the shape the accounting needs (manifest values win)
        if shape.get(k) is not None and model.get(k) is None:
            model[k] = shape[k]
    wl = raw.get('workload') or {}
    seq = wl.get('seq_len') or a.get('seq_len')
    workload = {'seq_len': seq}
    gbt = wl.get('global_batch_tokens') or (a['gbs'] * seq if a.get('gbs') and seq else None)
    if gbt:                                     # tokens per update: the manifest's, else the argument dump's
        workload['global_batch_tokens'] = int(gbt)
    data = wl.get('data') or ('mock' if dump.get('mock_data') == 'True' else None)
    if data:
        workload['data'] = data
    if wl.get('trainable'):                     # full / lora-r<N>: what the memory look-up compares
        workload['trainable'] = str(wl['trainable'])
    return {'gpu': gpu_class(hw.get('gpu_type')) or hw.get('gpu_type'), 'memory_gib': hw.get('memory_gib'),
            'gpus_per_node': project.gpus, 'nodes': project.nodes, 'interconnect': interconnect(hw.get('interconnect')),
            'framework': raw.get('framework'), 'versions': versions(raw.get('versions'), raw.get('framework')),
            'model': model, 'workload': workload,
            'layout': {k: str(a[k]) for k in ('tp', 'pp', 'cp', 'ep') if a.get(k) is not None},
            # the job's fixed mechanism states: the argument dump's, else what a project states (`states:`; an
            # environment adapter whose settings are outside its knobs, ext/optico project_view)
            'states': {**{k: str(v) for k, v in (raw.get('states') or {}).items()}, **megatron_states(dump)}}


def labels(knobs: dict, framework: str | None) -> dict:
    return {name: label(name, k, framework) for name, k in (knobs or {}).items()}


def x_id(name: str) -> str:
    return 'x-' + norm(name).replace('_', '-')


def states_of(fp: dict, labs: dict, config: dict, *, ok: bool = False) -> dict:
    """Mechanism states of a knob configuration: the reference run's states with the knobs' states over them; a knob
    without a mechanism is its own one (x-<name>). A project switch (a knob that is not the framework's own flag for
    its mechanism) may also set what its state requires: those mechanisms are unknown, or, for a run that worked (ok),
    the required states. What a knob value implies (Label.implies) is set, unless a knob of the configuration sets it."""
    out = dict(fp.get('states') or {})
    implied, fixed = {}, {}
    for name, v in (config or {}).items():
        lab = labs.get(name)
        if lab is None:
            out[x_id(name)] = str(v).lower() if isinstance(v, bool) else str(v)
            continue
        st = lab.state(v)
        if st is None:                              # a value the registry does not know: unknown, not the reference's
            if v is not None:
                out[lab.mechanism] = UNKNOWN
            continue
        out[lab.mechanism] = st
        fixed.update(lab.implies.get(str(v).lower() if isinstance(v, bool) else str(v)) or {})
        if not lab.own:
            implied.update(((mechanisms().get(lab.mechanism) or {}).get('requires') or {}).get(st) or {})
    own = {labs[n].mechanism for n in (config or {}) if labs.get(n) is not None}
    out.update({m: s for m, s in fixed.items() if m not in own})
    for m, s in implied.items():
        if out.get(m) != str(s):
            out[m] = str(s) if ok else UNKNOWN
    return out


def _state(states: dict, mech: str) -> str | None:
    """A mechanism's state, the registry default when not given."""
    v = states.get(mech, (mechanisms().get(mech) or {}).get('default'))
    return None if v is None else str(v)


def _within(a, b, factor: float = 2.0) -> bool:
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and a > 0 and b > 0 and max(a, b) / min(a, b) <= factor


def _choices(knob: dict) -> list:
    return list(knob.get('choices') or ([False, True] if knob.get('type') == 'bool' else []))


def _env_brief(env: dict) -> dict:
    m, w = env.get('model') or {}, env.get('workload') or {}
    return {'gpu': env.get('gpu'), 'nodes': env.get('nodes'), 'model': m.get('name'), 'params_b': m.get('params_b'),
            'seq_len': w.get('seq_len'), 'pack': env.get('pack')}


def differences(fp: dict, env: dict) -> list[str]:
    """The target's differences from an environment of the store, computed (not judged): GPU class with its numbers,
    scale, model, workload, framework version; what the target does not know is not a difference."""
    out = []
    ga, gb = env.get('gpu'), fp.get('gpu')
    if ga != gb and gb is not None:
        A, B = gpus().get(ga) or {}, gpus().get(gb) or {}
        nums = [f'{k} {A.get(k)} -> {B.get(k)}' for k in ('memory_gib', 'bf16_tflops_measured', 'bf16_tflops_peak',
                                                          'hbm_tbps', 'intra_link_gbps') if A.get(k) != B.get(k)]
        out.append(f'gpu {ga} -> {gb}' + (f' ({", ".join(nums)})' if nums else ''))
    m, n = env.get('model') or {}, fp.get('model') or {}
    w, v = env.get('workload') or {}, fp.get('workload') or {}
    for name, a, b in (('nodes', env.get('nodes'), fp.get('nodes')), ('gpus_per_node', env.get('gpus_per_node'), fp.get('gpus_per_node')),
                       ('model family', m.get('family', 'dense'), n.get('family', 'dense')),
                       ('params_b', m.get('params_b'), n.get('params_b')), ('seq_len', w.get('seq_len'), v.get('seq_len')),
                       ('global_batch_tokens', w.get('global_batch_tokens'), v.get('global_batch_tokens')),
                       ('data', w.get('data'), v.get('data')),
                       ('inter-node link', (env.get('interconnect') or {}).get('inter'), (fp.get('interconnect') or {}).get('inter')),
                       ('inter-node bandwidth GB/s', (env.get('interconnect') or {}).get('bandwidth_gbs'),
                        (fp.get('interconnect') or {}).get('bandwidth_gbs')),
                       ('framework version', (env.get('versions') or {}).get('framework'), (fp.get('versions') or {}).get('framework'))):
        if a != b and b is not None:
            out.append(f'{name} {a} -> {b}')
    return out


def _level(ev: list[dict]) -> dict | None:
    """The coverage level of one move from its evidence (nearest first). Readable signs that disagree in the matched
    scope (within the best tier: a harvest record never overrides InfraRSI's own measurement) give no numeric prior;
    outside it the nearest readable record gives the sign (half its size; the transfer judgment may change the sign)."""
    joint = [r for r in ev if r.get('joint') and r['readable']]   # measured only together with other changes
    ev = [r for r in ev if not r.get('joint')]
    cov = [r for r in ev if r['level'] == 'covered']
    rows = [r for r in cov if r['readable']]
    if rows:
        top = [r for r in rows if r['tier'] == rows[0]['tier']]
        if len({r['gain'] > 0 for r in top}) > 1:
            return {'level': 'partial', 'gain': 0.0, 'n': len(top), 'conflict': True}
        return {'level': 'covered', 'gain': rows[0]['gain'], 'n': len(rows)}
    if cov:
        return {'level': 'null', 'gain': 0.0, 'n': len(cov), 'within_pct': round(100 * max(abs(r['gain']) for r in cov), 2)}
    rows = [r for r in ev if r['readable']]
    if rows:
        return {'level': 'partial', 'gain': rows[0]['gain'], 'n': len(rows)}
    return {'level': 'partial', 'gain': 0.0, 'n': len(joint), 'joint': True} if joint else None


class Store:
    def __init__(self, packs: list[list[dict]]):
        self.envs, self.effects, self.observations, self.constraints, self.rules = {}, [], [], [], []
        self.packs, self.skipped, self.judgments = [], [], []
        for lines in packs:
            head = lines[0] if lines and lines[0].get('kind') == 'pack' else {}
            pid, tier = head.get('pack_id', 'unnamed'), head.get('tier', 'reported')
            self.packs.append({'pack_id': pid, 'tier': tier, 'title': head.get('title'), 'lines': len(lines)})
            for x in lines:
                k = x.get('kind')
                if k == 'env':
                    self.envs[f'{pid}/{x["id"]}'] = dict(x, gpu=gpu_class(x.get('gpu')) or x.get('gpu'), pack=pid, tier=tier)
                elif k == 'observation':
                    self.observations.append(dict(x, env=f'{pid}/{x["env"]}', pack=pid))
                elif k in ('constraint', 'rule', 'judgment'):
                    {'constraint': self.constraints, 'rule': self.rules, 'judgment': self.judgments}[k].append(dict(x, pack=pid))
            given = [x for x in lines if x.get('kind') == 'effect']
            pairs = {tuple(g.get('evidence') or ()) for g in given}
            for x in given + [e for e in derive_effects(lines) if tuple(e['evidence']) not in pairs]:
                self.effects.append(dict(x, env=f'{pid}/{x["env"]}', pack=pid, tier=x.get('tier', tier)))
        self.by_mech: dict[str, list] = {}
        for e in self.effects:
            for m in e.get('change') or {}:
                self.by_mech.setdefault(m, []).append(e)

    @classmethod
    def load(cls, include=None, extra=()) -> 'Store':
        """The shipped seeds, the packs under INFRARSI_HOME/experience/ and `extra`; `include`: pack-id patterns to
        keep (fnmatch), None for all. A pack that does not validate, or repeats a loaded pack id, is skipped."""
        paths = sorted(SEED.glob('*.jsonl')) + sorted((home() / 'experience').glob('**/*.jsonl*')) + [Path(p) for p in extra]
        packs, skipped, seen = [], [], set()
        for p in paths:
            try:
                lines = read_lines(p)
            except (OSError, ValueError) as e:
                skipped.append((str(p), [str(e)[:200]]))
                continue
            errors, _ = validate(lines)
            pid = lines[0].get('pack_id') if lines else None
            if errors or pid in seen:
                skipped.append((str(p), errors[:3] or [f'pack id {pid} already loaded']))
                continue
            seen.add(pid)
            if include is None or any(fnmatch.fnmatchcase(pid, pat) for pat in include):
                packs.append(lines)
        st = cls(packs)
        st.skipped = skipped
        return st

    # ------------------------------------------------------------ matching
    @staticmethod
    def scope_level(fp: dict, env: dict, cls: str, base: dict | None = None, skip=()) -> str | None:
        """'covered' when the effect's environment matches the target for this mechanism class, 'partial' when only the
        framework matches, None otherwise; `skip`: the mechanisms the effect itself changes (not compared as layout)."""
        if env.get('framework') != fp.get('framework'):
            return None
        same = ((env.get('model') or {}).get('family', 'dense') == (fp.get('model') or {}).get('family', 'dense')
                and _within((env.get('workload') or {}).get('seq_len'), (fp.get('workload') or {}).get('seq_len')))
        if cls != 'kernel':                 # balance (and unclassified): same GPU class and count, size bucket, layout;
            lay = fp.get('layout') or {}       # across nodes also the same fabric (an unknown one matches nothing)
            inter = (env.get('interconnect') or {}).get('inter')
            same = (same and env.get('gpu') == fp.get('gpu') and env.get('nodes') == fp.get('nodes')
                    and env.get('gpus_per_node') == fp.get('gpus_per_node')
                    and ((env.get('nodes') or 1) <= 1 or inter is not None and inter == (fp.get('interconnect') or {}).get('inter')
                         and _same_band(env, fp))
                    and _within((env.get('model') or {}).get('params_b'), (fp.get('model') or {}).get('params_b'))
                    and all(str((base or {}).get(k, '1')) == str(lay.get(k, '1')) for k in ('tp', 'pp', 'cp', 'ep') if k not in skip))
        return 'covered' if same else 'partial'

    @staticmethod
    def readable(e: dict, env: dict) -> bool:
        g = gain(e)
        if g is None or e.get('method') in ('estimate', 'reported') or e.get('tier') == 'reported':
            return False
        noise = e.get('noise_pct', env.get('noise_pct'))
        floor = 2 * noise / 100 if isinstance(noise, (int, float)) else FLOOR.get(e.get('method'), 0.02)
        return abs(g) >= floor

    @staticmethod
    def distance(fp: dict, env: dict, base: dict | None = None, ref_states: dict | None = None) -> float:
        d = float(env.get('gpu') != fp.get('gpu')) + float(env.get('nodes') != fp.get('nodes'))
        for a, b in (((env.get('model') or {}).get('params_b'), (fp.get('model') or {}).get('params_b')),
                     ((env.get('workload') or {}).get('seq_len'), (fp.get('workload') or {}).get('seq_len'))):
            if _within(a, b, math.inf):
                d += abs(math.log(a / b))
        if ref_states:                      # an effect measured on a base like the target's reference is nearer
            d += 0.25 * sum(1 for k, v in (base or {}).items() if k in ref_states and str(v) != ref_states[k])
        return round(d, 4)

    def evidence(self, fp: dict, mech: str, cls: str, frm: str, to: str, ref_states: dict | None = None,
                 own: bool = False) -> list[dict]:
        """Effects for the move frm -> to (or its reverse, inverted) with their level and readability, nearest first:
        covered before partial, readable first, InfraRSI's own before harvests, then by distance. An effect of several
        changes at once is `joint`: partial context, never a size; so is one with companions (`with`, what the state
        requires) when the knob is the framework's own flag (`own`), which does not set them."""
        out, reg = [], mechanisms().get(mech) or {}
        for e in self.by_mech.get(mech, ()):
            ch = e.get('change') or {}
            g = gain(e)
            if g is None:
                continue
            a, b = (str(x) for x in ch[mech])
            sign = 1 if (a, b) == (frm, to) else -1 if (a, b) == (to, frm) else 0
            if not sign:
                continue
            env = self.envs.get(e['env']) or {}
            base = e.get('base') or {}
            lvl = self.scope_level(fp, env, cls, base, skip=set(ch) | set(e.get('with') or {}))
            if lvl and (len(ch) > 1 or own and e.get('with') or ref_states and any(
                    m in ref_states and ref_states[m] != str(ab[0 if sign > 0 else 1]) for m, ab in (e.get('with') or {}).items())):
                lvl = 'partial'             # with other changes, or from another state of its companion mechanism
            if lvl == 'covered' and any(_state(base, k) != _state(ref_states or {}, k) for k in reg.get('depends_on') or ()):
                lvl = 'partial'             # its size depends on a state the effect was measured in another value of
            if lvl:
                out.append({'level': lvl, 'gain': sign * g, 'readable': self.readable(e, env), 'effect': e, 'env': env,
                            'tier': TIER.get(e.get('tier'), 2), 'distance': self.distance(fp, env, base, ref_states),
                            'joint': len(ch) > 1 or own and bool(e.get('with'))})
        out.sort(key=lambda r: (r['level'] != 'covered', not r['readable'], r['tier'], r['distance']))
        return out

    # ------------------------------------------------------------ queries
    def coverage(self, fp: dict, knobs: dict, reference: dict, labs: dict | None = None, ref_states: dict | None = None) -> dict:
        """knob -> {mechanism, class, level, value, gain (log, of the move reference -> value), n, evidence, nearest}:
        per knob the best-covered other choice (then the largest gain)."""
        labs = labels(knobs, fp.get('framework')) if labs is None else labs
        ref_states = states_of(fp, labs, reference) if ref_states is None else ref_states
        out = {}
        for name, k in knobs.items():
            lab = labs.get(name)
            if lab is None:
                out[name] = {'mechanism': None, 'class': None, 'level': 'none'}
                continue
            frm, rows = lab.state(reference.get(name)), []
            for c in _choices(k):
                to = lab.state(c)
                if c == reference.get(name) or frm is None or to is None or to == frm:
                    continue
                ev = self.evidence(fp, lab.mechanism, lab.cls, frm, to, ref_states, own=lab.own)
                row = _level(ev)
                if row:
                    src = [r for r in ev if bool(r.get('joint')) == bool(row.get('joint'))]   # what the level used
                    rows.append(dict(row, value=c, to=to, nearest=_env_brief(src[0]['env']),
                                     evidence=[f'{r["effect"]["pack"]}/{r["effect"]["id"]}' for r in src[:3]]))
            best = {'mechanism': lab.mechanism, 'class': lab.cls, 'level': 'none', 'from': frm}
            if rows:
                best.update(min(rows, key=lambda r: (LEVELS.index(r['level']), -r['gain'])))
                if lab.mechanism.startswith('x-'):   # not registered: shown to the LLM, never a prior (protocol 3)
                    best.update(level='partial', gain=0.0, unregistered=True)
            out[name] = best
        return out

    def nearest(self, fp: dict, k: int = 3) -> list[dict]:
        same = [eid for eid, e in self.envs.items() if e.get('framework') == fp.get('framework')]
        rows = sorted((self.distance(fp, self.envs[eid]), eid) for eid in same)[:k]
        return [{'env': eid, 'distance': d, **_env_brief(self.envs[eid]), 'differences': differences(fp, self.envs[eid])}
                for d, eid in rows]

    def fits(self, fp: dict, states: dict) -> dict | None:
        """The memory look-up: the nearest observation of the same framework, model size (within 1.3x), sequence
        length, GPU count and shape (tp, pp, cp, micro-batch, precision, recompute, main grads) -> fits iff its peak plus
        the framework's headroom is within the target's memory. An OOM on a smaller card says nothing; None: no match."""
        cap = fp.get('memory_gib') or (gpus().get(fp.get('gpu')) or {}).get('memory_gib')
        pb, seq = (fp.get('model') or {}).get('params_b'), (fp.get('workload') or {}).get('seq_len')
        if not cap or not pb:
            return None
        want = full_config(states)
        trainable = (fp.get('workload') or {}).get('trainable')
        world = (fp.get('gpus_per_node') or 1) * (fp.get('nodes') or 1)
        best = None
        for o in self.observations:
            env = self.envs.get(o['env']) or {}
            p2 = (env.get('model') or {}).get('params_b')
            if (env.get('framework') != fp.get('framework') or not _within(p2, pb, 1.3)
                    or (env.get('workload') or {}).get('seq_len') != seq or (env.get('nodes') or 1) != (fp.get('nodes') or 1)
                    or (env.get('gpus_per_node') or 1) * (env.get('nodes') or 1) != world):
                continue
            if o.get('status') == 'oom':
                ocap = env.get('memory_gib') or (gpus().get(env.get('gpu')) or {}).get('memory_gib')
                if not ocap or cap > ocap:
                    continue
            elif o.get('status') != 'ok' or not o.get('mem_peak_gib'):
                continue                                # a crashed run's peak says nothing
            have = full_config(o.get('config'))
            if any(have.get(k) != want.get(k) for k in SHAPE) or (env.get('workload') or {}).get('trainable') != trainable:
                continue                                # LoRA vs full fine-tuning: another memory regime
            # the nearest size first; at one distance the same configuration (beyond the memory shape: compile, kernels)
            # wins, so a candidate's own OOM is not hidden by the reference's peak
            key = (abs(math.log(p2 / pb)), have != want)
            if best is None or key < best[0]:
                best = (key, o)
        if best is None:
            return None
        o = best[1]
        ok = o.get('status') != 'oom' and o['mem_peak_gib'] + HEADROOM.get(fp.get('framework'), 10.0) <= cap
        return {'fits': ok, 'mem_peak_gib': o.get('mem_peak_gib'), 'status': o.get('status'), 'capacity_gib': cap,
                'evidence': f'{o["pack"]}/{o["id"]}'}

    def matching_constraints(self, fp: dict, states: dict) -> list[dict]:
        full = full_config(states)
        return [c for c in self.constraints if _scope_matches(c.get('scope') or {}, fp, full)
                and all(full.get(m) == str(s) for m, s in (c.get('when') or {}).items())]

    def noise_pct(self, gpu: str | None) -> float | None:
        """The median run noise InfraRSI measured on this GPU class (information for the doctor; never used)."""
        xs = sorted(e['noise_pct'] for e in self.envs.values() if e.get('gpu') == gpu and e.get('tier') == 'verified'
                    and isinstance(e.get('noise_pct'), (int, float)))
        return xs[len(xs) // 2] if xs else None


def _version_ok(have: str | None, spec: str) -> bool:
    if not have:
        return False
    v = tuple(int(x) for x in re.findall(r'\d+', have)[:3])
    for part in spec.split(','):
        m = re.match(r'\s*(>=|<=|==|<|>)\s*([\d.]+)', part)
        if not m:
            continue
        w = tuple(int(x) for x in m.group(2).split('.'))
        n = len(w)
        if not {'>=': v[:n] >= w, '<=': v[:n] <= w, '==': v[:n] == w, '<': v[:n] < w, '>': v[:n] > w}[m.group(1)]:
            return False
    return True


def _scope_matches(scope: dict, fp: dict, full: dict) -> bool:
    """A record's scope against the target: mechanism keys against its states, dotted keys against the fingerprint
    (`versions.torch`, `workload.data`); a version range (">=0.13,<0.14"), a list of allowed values or {min, max}."""
    for key, want in scope.items():
        if key in mechanisms():
            have = full.get(key)
        else:
            have = fp
            for part in key.split('.'):
                have = have.get(part) if isinstance(have, dict) else None
        if isinstance(want, str) and re.match(r'^\s*(>=|<=|==|<|>)', want):
            if not _version_ok(have, want):
                return False
        elif isinstance(want, list):
            if have not in want:
                return False
        elif isinstance(want, dict):
            if not isinstance(have, (int, float)) or not want.get('min', -math.inf) <= have <= want.get('max', math.inf):
                return False
        elif have != want:
            return False
    return True


def payload_constraints(store: Store, fp: dict) -> list[dict]:
    """The constraints whose environment scope (framework, versions, workload; not the mechanism states) matches."""
    full = full_config(fp.get('states') or {})
    return [c for c in store.constraints
            if _scope_matches({k: v for k, v in (c.get('scope') or {}).items() if k not in mechanisms()}, fp, full)]


def payload(store: Store, fp: dict, cov: dict, *, budget_tokens: int = 3000) -> dict:
    """What the LLM sees of the store: coverage per knob, the nearest environments with their computed differences,
    effects for the knobs the store does not cover, and the constraints and rules whose scope matches the target;
    trimmed to about `budget_tokens`."""
    target = {k: fp.get(k) for k in ('gpu', 'memory_gib', 'gpus_per_node', 'nodes', 'framework', 'versions', 'model',
                                     'workload', 'layout')}
    coverage = {n: {k: (round(100 * v, 2) if k == 'gain' else v) for k, v in c.items()
                    if k in ('mechanism', 'level', 'value', 'gain', 'n', 'conflict', 'within_pct', 'evidence')}
                for n, c in cov.items()}
    open_ = {c['mechanism'] for c in cov.values() if c.get('mechanism') and c['level'] in ('partial', 'none')}
    effects = []
    for m in sorted(open_):
        for e in store.by_mech.get(m, ()):
            env = store.envs.get(e['env']) or {}
            g = gain(e)
            if env.get('framework') != fp.get('framework') or g is None:
                continue
            effects.append((store.distance(fp, env), {
                'id': f'{e["pack"]}/{e["id"]}', 'mechanism': m, 'with': e.get('with'),
                'change': e['change'][m] if len(e['change']) == 1 else e['change'],      # a joint effect: all of it
                'gain_pct': round(100 * g, 2), 'readable': store.readable(e, env), **_env_brief(env),
                'layout': {k: e.get('base', {}).get(k) for k in ('tp', 'pp') if e.get('base', {}).get(k)}}))
    effects = [x for _, x in sorted(effects, key=lambda t: t[0])][:40]
    full = full_config(fp.get('states') or {})
    cons = [{k: c.get(k) for k in ('id', 'when', 'symptom', 'message', 'fix')} for c in payload_constraints(store, fp)]
    hits = [r for r in store.rules if _scope_matches({k: v for k, v in (r.get('scope') or {}).items() if k != 'gpu'}, fp, full)
            and fp.get('gpu') in ((r.get('scope') or {}).get('gpu') or [fp.get('gpu')])]
    hits.sort(key=lambda r: (-len(r.get('scope') or {}), CONFIDENCE.get(r.get('confidence'), 3)))
    rules = [{'id': f'{r["pack"]}/{r["id"]}', 'text': str(r.get('text', ''))[:300], 'confidence': r.get('confidence')}
             for r in hits[:15]]
    out = {'target': target, 'coverage': coverage, 'nearest': store.nearest(fp), 'effects': effects, 'constraints': cons,
           'rules': rules}
    while len(json.dumps(out, default=str)) > 4 * budget_tokens and (out['effects'] or out['rules']):
        out['effects' if len(out['effects']) > len(out['rules']) else 'rules'].pop()
    return out


def _learned_starts(store: Store, fp: dict, knobs: dict, labs: dict) -> list[dict]:
    from .learned import start_failures
    return start_failures(store, fp, knobs, labs)


def plan(store: Store, fp: dict, knobs: dict, reference: dict, *, accounting: bool = True) -> dict:
    """Everything the search and the doctor take from the store for one job:

    seeds      one row per covered or partial knob (the move, its source gain, evidence, warnings); HLS turns them into
               ledger entries with the engine's noise (decision 2.1)
    known_ok   the moves the store measured within noise in scope (null): zero prior, but known to run
    composite  the <= 4 covered moves with the largest positive gains and no warning, as one start candidate
    check      config -> warnings: version-matched constraints with a hard symptom the reference does not trigger, and
               the memory look-up (where it has no match: the closed-form accounting, unless `accounting` is off); a
               warning makes a hypothesis probe-first (p_feasible <= 0.2), never rejects it"""
    labs = labels(knobs, fp.get('framework'))
    ref_states = states_of(fp, labs, reference)
    lay = dict(fp.get('layout') or {})              # the argument dump's layout, else the knobs' reference states
    lay.update({k: ref_states[k] for k in LAYOUT if k not in lay and ref_states.get(k) not in (None, UNKNOWN)})
    fp = dict(fp, layout=lay)
    cov = store.coverage(fp, knobs, reference, labs=labs, ref_states=ref_states)
    at_ref = {(c['pack'], c['id']) for c in store.matching_constraints(fp, ref_states)}   # ids are local to a pack
    from .accounting import READS, estimate

    def accounted(st: dict) -> dict | None:
        return None if not accounting or any(st.get(k) == UNKNOWN for k in READS) else estimate(fp, st)
    acc = accounted(ref_states)
    # the estimate warns only where it agrees that the reference (which ran) fits: else it is off for this job
    acc_warns = bool(acc and acc['capacity_gib'] and acc['mem_gib'] <= acc['capacity_gib'])

    def check(config: dict) -> list[str]:
        st = states_of(fp, labs, config)
        out = [f'{c["id"]} ({c["symptom"]}): {c.get("message", "")}'[:240] for c in store.matching_constraints(fp, st)
               if (c['pack'], c['id']) not in at_ref and c.get('symptom') in HARD]
        m = store.fits(fp, st)
        if m is not None and not m['fits']:
            out.append(f'memory: {m["evidence"]} ' + (f'peaks at {m["mem_peak_gib"]} GiB of {m["capacity_gib"]}'
                                                       if m.get('mem_peak_gib') else 'ran out of memory'))
        elif m is None and acc_warns:                  # no measured neighbour: the closed-form estimate (+-25 %)
            est = accounted(st)
            if est and est['mem_gib'] > est['capacity_gib']:
                out.append(f'{ESTIMATE}: about {est["mem_gib"]:.0f} GiB of {est["capacity_gib"]:.0f} '
                           '(closed-form accounting, +-25 %)')
        return out

    layout_knobs = [n for n, lab in labs.items() if lab is not None and lab.mechanism in LAYOUT]
    seeds, deps = [], {}
    for name, c in cov.items():
        if c['level'] in ('covered', 'partial') and c.get('gain'):
            # the evidence is for this knob's move from its reference value (and, balance class, at this layout; and at
            # the states its size depends on): on an incumbent that changed any of them, HLS does not seed it again
            dep = (mechanisms().get(c['mechanism']) or {}).get('depends_on') or ()
            deps[name] = {n for n, lab in labs.items() if lab is not None and lab.mechanism in dep}
            on = [name] + (layout_knobs if c['class'] != 'kernel' else []) + sorted(deps[name])
            seeds.append({'knob': name, 'mechanism': c['mechanism'], 'change': {name: c['value']}, 'level': c['level'],
                          'gain': c['gain'], 'evidence_ids': c.get('evidence') or [],
                          'warnings': check({**reference, name: c['value']}), 'valid_on': {k: reference.get(k) for k in on}})
    members = []
    for s in sorted(seeds, key=lambda s: -s['gain']):
        hard = [w for w in s['warnings'] if not str(w).startswith(ESTIMATE)]   # an estimate: the composite is probed
        if (s['level'] == 'covered' and s['gain'] > 0 and not hard and len(members) < 4
                and not any(deps[s['knob']] & set(m['change']) or deps[m['knob']] & set(s['change']) for m in members)):
            members.append(s)                       # not two moves where one changes a state the other's size depends on
    composite = None
    if len(members) >= 2:
        change = {k: v for s in members for k, v in s['change'].items()}
        composite = {'members': [s['knob'] for s in members], 'change': change,
                     'gain': sum(s['gain'] for s in members), 'warnings': check({**reference, **change})}
    return {'fingerprint': fp, 'coverage': cov, 'seeds': seeds, 'composite': composite, 'check': check, 'accounting': acc,
            'soft': lambda w: str(w).startswith(ESTIMATE),     # an estimate's warning: probe-first, never a drop
            'known_ok': [{n: c['value']} for n, c in cov.items() if c['level'] == 'null'],
            'constraints': [c['id'] for c in payload_constraints(store, fp)],
            'start_failures': _learned_starts(store, fp, knobs, labs),     # (F9b: what earlier arms learned)
            'at_reference': sorted(f'{p}/{i}' for p, i in at_ref),
            'nearest': store.nearest(fp), 'noise_pct': store.noise_pct(fp.get('gpu')), 'payload': payload(store, fp, cov),
            'packs': [p['pack_id'] for p in store.packs], 'skipped': store.skipped,
            'missing': [k for k, v in (('model.params_b', (fp.get('model') or {}).get('params_b')),
                                       ('workload.seq_len', (fp.get('workload') or {}).get('seq_len'))) if not v]}


# ---------------------------------------------------------------- what `init` drafts from the shipped packs
@lru_cache(maxsize=1)
def seed_store() -> Store:
    """The shipped packs only: what `init` drafts does not depend on the packs of the machine it runs on."""
    return Store([read_lines(p) for p in sorted(SEED.glob('*.jsonl'))])


def _single_moves(store: Store, framework: str | None):
    """(mechanism, [from, to], speed-up, effect, env) of the store's one-mechanism effects on this framework."""
    for e in store.effects:
        env = store.envs.get(e['env']) or {}
        ch = e.get('change') or {}
        if env.get('framework') == framework and len(ch) == 1 and isinstance(e.get('speedup'), (int, float)):
            (m, move), = ch.items()
            yield m, move, e['speedup'], e, env


@lru_cache(maxsize=None)
def measured(framework: str | None, min_speedup: float = 1.005) -> frozenset:
    """The mechanisms the shipped packs measured faster by one change on this framework. M8 searched without bf16 main
    grads (+6.6 % on Megatron in the H200 packs): the flags `init` drafts from a framework's own arguments follow these."""
    return frozenset(m for m, _, s, _, _ in _single_moves(seed_store(), framework) if s >= min_speedup)


DIFFUSION_ARCH = re.compile(r'dit|unet')                # the packs' x-arch of image and video models


def _alias(mech: str, framework: str | None, state) -> str | None:
    """The framework's own flag or env var that sets this state of the mechanism (`--main-grads-dtype bf16`); None when
    the registry knows none (only a patch reaches it)."""
    for fw in (framework, 'any'):
        for a in ((mechanisms().get(mech) or {}).get('aliases') or {}).get(fw) or []:
            where = a['flag'] if a.get('flag') else f'env {a["env"]}' if a.get('env') else f'config {a["config"]}' if a.get('config') else None
            sep = ' ' if a.get('flag') else '='
            if where is None:
                continue
            if 'present' in a:
                if str(a['present']) == str(state):
                    return where
            elif 'map' in a:
                key = next((k for k, v in a['map'].items() if str(v) == str(state)), None)
                if key is not None:
                    return f'{where}{sep}{key}'
            else:
                return f'{where}{sep}{state}'
    return None


def unreached(framework: str | None, knobs: dict, *, diffusion: bool = False, moe: bool = False,
              min_speedup: float = 1.03, limit: int = 4, current: dict | None = None) -> list[dict]:
    """The largest levers the shipped packs measured on jobs of this framework and kind (image / video models or not;
    MoE or not) that no knob of the project sets: F-D2-R205-4 and M8, a stack's largest lever was not a knob and nothing
    said so. A move counts from the state the job runs - its knob's reference value, else what its code sets
    (`current`, sinks.py), else the mechanism's default (a mechanism with none, e.g. the ZeRO stage, is not known for
    this job) - unless a knob of the project may be a state of it (an unmapped `bf16` switch); a move is reached only
    when a knob's value gives its faster state (Wan's DIFFSYNTH_ATTENTION_IMPLEMENTATION does not give cuDNN-first
    attention, x1.39). Each: the mechanism, its best single-change speed-up with model, GPU and evidence, the
    framework's flag or env var for the faster state (None: a patch), the candidate knobs."""
    labs = labels(knobs, framework)
    reach, now = {}, dict(current or {})                # mechanism -> states a knob gives (None: any); the job's state
    for n, lab in labs.items():
        if lab is None:
            continue
        k = knobs[n]
        vals = k.get('choices') or ([False, True] if k.get('type') == 'bool' else [])
        if not vals:
            reach[lab.mechanism] = None                 # a range (min / max): any state; workers [0, 2, 4, 8] not 16
        elif reach.get(lab.mechanism, ()) is not None:
            reach[lab.mechanism] = set(reach.get(lab.mechanism) or ()) | ({lab.state(v) for v in vals} - {None})
        for v in vals:
            for m2, s2 in lab.implies.get(str(v).lower() if isinstance(v, bool) else str(v), {}).items():
                if reach.get(m2, ()) is not None:
                    reach[m2] = set(reach.get(m2) or ()) | {s2}
        if lab.state(k.get('default')) is not None:
            now[lab.mechanism] = lab.state(k.get('default'))
    best = {}
    for m, move, s, e, env in _single_moves(seed_store(), framework):
        model = env.get('model') or {}
        if ((m in reach and (reach[m] is None or str(move[1]) in reach[m])) or m.startswith('x-') or s < min_speedup
                or (model.get('family') == 'moe' and not moe)
                or bool(DIFFUSION_ARCH.search(str(model.get('x-arch') or ''))) != diffusion):
            continue
        maybe = [n for n, lab in labs.items() if lab is None and m in state_owners(n)]
        start = now.get(m, (mechanisms().get(m) or {}).get('default'))
        if (start is None or str(move[0]) != str(start)) and not maybe:
            continue                                    # a move from a state this job is not known to run (zero 3 -> 2)
        if (m == 'precision' or m.startswith('fp8')) and not maybe:
            continue                                    # numerics, not speed alone; Megatron's FP8 also takes --fp8-format
        if m not in best or s > best[m]['speedup']:
            best[m] = {'mechanism': m, 'speedup': round(s, 3), 'move': [str(x) for x in move], 'model': model.get('name'),
                       'gpu': env.get('gpu'), 'evidence': e.get('id'), 'maybe': maybe}
    out = sorted(best.values(), key=lambda r: -r['speedup'])[:limit]
    for r in out:
        r['alias'] = _alias(r['mechanism'], framework, r['move'][1])
        need = ((mechanisms().get(r['mechanism']) or {}).get('requires') or {}).get(r['move'][1]) or {}
        partners = [p for p in (_alias(m2, framework, s2) for m2, s2 in need.items()) if p]
        if r['alias'] and partners:                     # bf16 main grads only with the precision-aware optimizer
            r['alias'] += ' with ' + ' and '.join(partners)
    return out
