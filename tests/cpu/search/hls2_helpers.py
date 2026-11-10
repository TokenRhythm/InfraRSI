"""Harness of the hls2 tests (FABLE_DESIGN_SEARCH_FIX, 2026-10-29): an Engine with its real arithmetic (value, cost,
plan, decide, run) over a simulated session, and a replay of the recorded Q9U-code-env-opus run (strategy hls, ten
screens, the Liger knob never tested) from data/q9u_code_env_opus.json: the model's recorded hypotheses per incumbent
stage, the recorded noise per decision and run costs, and the measured effects of the moves (the design's section 4.2:
recorded where this job measured them, the other arms' values or zero where it did not)."""
import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

from infrarsi.agent.capability import search_params
from infrarsi.agent.search import SearchSession
from infrarsi.search.engine import Engine, Outcome
from infrarsi.search.proposers import UNTESTED_STATES

FIXTURE = json.loads((Path(__file__).resolve().parent / 'data' / 'q9u_code_env_opus.json').read_text(encoding='utf-8'))
PATCH_FILE = 'train_patches.py'
BASE_TEXT = 'import torch\n\n\ndef _setup():\n    pass\n\n\n_setup()\n'


def snippet(move: str) -> str:
    """A patch appends a function and its call (as the models' patches of train_patches.py did)."""
    return f'\n\ndef _m_{move}():\n    pass\n\n\n_m_{move}()\n'


class SimSession:
    """What HLS and the Engine read of a SearchSession, with the session's own start-failure accounting (C5)."""
    _note_start_failure = SearchSession._note_start_failure
    start_failed_hit = SearchSession.start_failed_hit
    value_unrunnable = SearchSession.value_unrunnable

    def __init__(self, start_config: dict, *, knobs: dict, screens: int = 10, t0: float = 1.0, c0_gpu_hours: float = 0.3,
                 base_files: dict | None = None):
        self.candidates = {'c0': SimpleNamespace(id='c0', parent=None, config=dict(start_config), overlay={},
                                                 verdict={'valid': True, 'step_time': t0}, status='screened', results={},
                                                 receipt={'cost': {'gpu_seconds': c0_gpu_hours * 3600}, 'evidence': {}})}
        files = dict(base_files or {PATCH_FILE: BASE_TEXT})
        self.hooks = SimpleNamespace(candidate_files=lambda overlay: {**files, **(overlay or {})}, base_files=files)
        self.task = SimpleNamespace(knob_schema={'properties': {
            k: ({'type': 'boolean'} if all(isinstance(x, bool) for x in v) else {'enum': v}) for k, v in knobs.items()}})
        self.screens, self.invalid, self.by_key, self.stop_flag = [], {}, {}, None
        self.budget = SimpleNamespace(screens=screens)
        self.ledger = SimpleNamespace(used={'screens': 0, 'gpu_hours': 0.0},
                                      exhausted=lambda: ['screens'] if self.ledger.used['screens'] >= screens else [])
        self.start_failure_limit, self.start_failures, self.unrunnable_values = None, {}, {}

    def count_invalid(self, reason):
        self.invalid[reason] = self.invalid.get(reason, 0) + 1

    def finalize(self, *, selected, stop_reason, extra, override=False):
        return SimpleNamespace(selected=selected, stop_reason=stop_reason, extra=extra)


class SimEngine(Engine):
    """The Engine (decide, value, cost, plan, key, run) on a SimSession; evaluate() reads `effect(base, config,
    overlay)` -> (log gain vs the base, failure) with failure None | 'aborted' | 'timeout' | 'oom' | 'crash' (hls3).
    Noise: sigma_run per decision (the last value repeats), costs: full runs in order, probes at probe_cost."""

    def __init__(self, session: SimSession, effect, *, sigma_runs=(0.01,), full_costs=(0.3,), probe_cost=None,
                 validate=None, log=None, job_gpu_hours=None):
        super().__init__(session, full_steps=30, job_gpu_hours=job_gpu_hours, run_noise=None, log=log)
        self.effect, self.validate = effect, validate
        self.sigma_runs, self.full_costs = list(sigma_runs), list(full_costs)
        self.probe_cost = probe_cost if probe_cost is not None else 0.35 * self.full_costs[0]
        self.n_full = 0
        self.order: list[tuple] = []                     # (mechanism, change, outcome status, gain, new incumbent)

    def sigma_run(self) -> float:
        return self.sigma_runs[min(len(self.decisions), len(self.sigma_runs) - 1)]

    def sigma_pair(self) -> float:
        return self.sigma_run() * math.sqrt(2)

    def _out(self, p, cid, status, gain=None, gpu=0.0, reasons=(), failed=(), fidelity='full'):
        o = Outcome(cid, status, status == 'screened', None, gain, fidelity, gpu, list(reasons), list(failed))
        self.order.append((p.mechanism, dict(p.change or {}), status, gain, self._inc))
        return o

    def evaluate(self, p):
        s = self.s
        if tuple(p.fidelity) != ('probe',):
            self.tried.add(self.key(p))
        base = 'c0' if p.absolute else (p.base or self._inc)
        b = s.candidates[base]
        cfg = dict(p.change) if p.absolute else {**b.config, **(p.change or {})}
        bad = self.validate(cfg) if self.validate else None
        if bad:
            s.count_invalid('l0_invalid')
            return self._out(p, None, 'rejected', reasons=[bad])
        overlay = {**b.overlay, **(p.patch or {})}
        key = json.dumps([sorted(cfg.items(), key=str), sorted(overlay.items())], default=str)
        if key in s.by_key:
            s.count_invalid('duplicate')
            return self._out(p, None, 'duplicate', reasons=['already tried'])
        cid = f'c{len(s.candidates)}'
        c = SimpleNamespace(id=cid, parent=base, config=cfg, overlay=overlay, verdict=None, status='proposed', results={},
                            receipt=None)
        s.candidates[cid], s.by_key[key] = c, cid
        dead = s.start_failed_hit(cid) if s.unrunnable_values else None
        if dead:
            from infrarsi.agent.search import START_FAILED
            s.count_invalid('start_failed')
            return self._out(p, None, 'rejected', reasons=[f'{START_FAILED} {dead}'])
        gain, failure = self.effect(b, cfg, overlay)
        gpu = 0.0
        for fid in self.plan(p):
            if fid == 'probe':
                gpu += self.probe_cost
                s.ledger.used['gpu_hours'] += self.probe_cost
                s.screens.append({'fidelity': 'probe', 'gpu_hours': self.probe_cost, 'valid': failure not in ('timeout', 'oom')})
                if failure in ('timeout', 'oom', 'crash'):
                    self._fail(c, failure)
                    c.status = 'infeasible'
                    return self._out(p, cid, 'infeasible', gpu=gpu, reasons=[self._why(failure)], failed=['completes'],
                                     fidelity='probe')
                continue
            cost = self.full_costs[min(self.n_full, len(self.full_costs) - 1)]
            self.n_full += 1
            gpu += cost
            s.ledger.used['screens'] += 1
            s.ledger.used['gpu_hours'] += cost
            if failure in ('timeout', 'oom', 'crash'):
                s.screens.append({'fidelity': 'full', 'gpu_hours': cost, 'valid': False})
                self._fail(c, failure)
                c.status, c.verdict = 'screened', {'valid': False}
                return self._out(p, cid, 'invalid', gpu=gpu, reasons=[self._why(failure)], failed=['completes'])
            t_base = self.t(base)
            st = t_base * math.exp(-gain)
            if failure == 'aborted':
                s.screens.append({'fidelity': 'full', 'gpu_hours': cost, 'valid': False, 'aborted': True})
                c.status = 'aborted'
                return self._out(p, cid, 'aborted', gpu=gpu, reasons=['aborted: slower than the incumbent'])
            s.screens.append({'fidelity': 'full', 'gpu_hours': cost, 'valid': True})
            c.status, c.verdict = 'screened', {'valid': True, 'step_time': st}
            g = math.log(self.t(self._inc) / st)
            if g > 2 * self.sigma_pair() or p.confirm and g > 1.645 * self.sigma_run():
                self._inc = cid                         # (a near-threshold composite: its confirmation agrees, no noise)
            return self._out(p, cid, 'screened', gain=gain, gpu=gpu)
        return self._out(p, cid, 'invalid', reasons=['empty evaluation plan'])

    @staticmethod
    def _why(failure):
        return ('completes: killed at the timeout after 0/30 updates' if failure == 'timeout' else
                'completes: exit code 1, 0/30 updates; the log: AssertionError: a node was invalid' if failure == 'crash'
                else 'completes: exit code 1, 0/30 updates; the log: torch.OutOfMemoryError: CUDA out of memory')

    def _fail(self, c, failure):
        """The session's start-failure accounting, as SearchSession._note_unrunnable calls it for a knob change."""
        p = self.s.candidates[c.parent]
        if self.s.start_failure_limit and c.overlay == p.overlay:
            ev = {'steps_completed': 0, 'timed_out': failure == 'timeout', 'events': {'oom': int(failure == 'oom')}}
            self.s._note_start_failure(c, p, ev, '', SimpleNamespace(reasons=[self._why(failure)]))


# ---------------------------------------------------------------- the recorded run (FABLE_DESIGN_SEARCH_FIX section 4)
KNOB_MOVES = {('gradient_checkpointing', False): 'gc', ('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True'): 'exp',
              ('use_liger_kernel', True): 'liger', ('fsdp_config', 'fsdp_v2_noreshard.json'): 'fsdp',
              ('optim', 'adamw_torch_fused'): 'optim', ('torch_compile', True): 'compile', ('tf32', True): 'tf32',
              ('NCCL_NVLS_ENABLE', '0'): 'nvls', ('dataloader_num_workers', 4): 'workers4',
              ('dataloader_num_workers', 0): 'workers0', ('dataloader_num_workers', 8): 'workers8',
              ('dataloader_pin_memory', False): 'pin'}
# log gains of single moves (recorded on this job where measured: micro-batch 2 +7.72 %, recompute off +21.33 %, the
# conv patch +7.24 %, no-reshard +3.15 %, the small patches; the other arms' values otherwise; zero where never measured)
TABLE = {'exp': 0.0, 'mbs2': 0.0772, 'optim': -0.002, 'tf32': -0.0003, 'nvls': -0.003, 'workers4': -0.001,
         'workers0': -0.010, 'workers8': 0.0, 'pin': 0.0, 'compile': None, 'gdn_compile': None, 'layer_compile': None,
         'flce': -0.0037, 'lm_head': 0.0013, 'clip': 0.0007, 'nccl': 0.0005, 'bf16': 0.0015, 'ctas': -0.0029,
         'adam_local': 0.0, 'skip_fp32': 0.0, 'cudnn': 0.0, 'prefetch': 0.0, 'gc_freeze': 0.0, 'skip100': 0.0,
         'liger_ce': 0.0, 'reg_compile': 0.0, 'sel_ckpt': 0.0, 'part_ckpt': 0.0, 'sdpa': 0.0}
OOM_MOVES = {'gdn_compile', 'layer_compile'}         # assumed (code-opus c10: out of memory); compile: the 900 s cap
MOOT = {'flce': ['liger'], 'skip_fp32': ['liger'], 'liger_ce': ['liger'], 'sel_ckpt': ['gc'], 'part_ckpt': ['gc'],
        'cudnn': ['conv']}


def moves(cfg: dict, overlay: dict | None) -> set:
    out = {m for (k, v), m in KNOB_MOVES.items() if cfg.get(k) == v}
    if cfg.get('per_device_train_batch_size') == 2:
        out.add('mbs2')
    return out | set(re.findall(r'^def _m_(\w+)\(', (overlay or {}).get(PATCH_FILE, ''), re.M))


def move_gain(m: str, a: set):
    if m == 'gc':
        return 0.2133 if 'mbs2' in a else 0.19
    if m == 'liger':
        return 0.085 if 'gc' in a else 0.059
    if m == 'fsdp':
        return 0.040 if 'liger' in a else 0.0315
    if m == 'conv':
        return 0.088 if 'liger' in a else 0.0724
    return TABLE[m]


def q9u_effect(base, cfg, overlay):
    """(log gain of the candidate vs its base, failure): new moves read on the candidate's state, removed ones on the
    base's (the design's replay, hls_replay2.cand_gain); recompute off + micro-batch 2 + no-reshard without Liger or
    expandable segments thrashes the allocator (random c1, pre-check R1001)."""
    a, b = moves(cfg, overlay), moves(base.config, base.overlay)
    new, gone = a - b, b - a
    if any(m == 'compile' for m in new):
        return None, 'timeout'
    if new & OOM_MOVES:
        return None, 'oom'
    if {'gc', 'mbs2', 'fsdp'} <= a and not {'liger', 'exp'} & a:
        return -0.10, 'aborted'
    g = sum(move_gain(m, a) for m in new) - sum(move_gain(m, b) or 0.0 for m in gone)
    return g, 'aborted' if 'flce' in new or g <= -0.01 else None


def q9u_valid(cfg):
    return None if cfg['per_device_train_batch_size'] * cfg['gradient_accumulation_steps'] == 2 else \
        'config validation failed: workload invariant global_batch changes'


def stage_of(a: set) -> str:
    """The recorded call whose hypotheses the model gives on a base with these moves (the design's assumption: on an
    unrecorded base, the nearest recorded base's)."""
    if 'gc' not in a:
        return 'c1' if 'mbs2' in a else 'c0'
    return 'c2' if 'conv' not in a else 'c3' if 'fsdp' not in a else 'c8'


def moot(mv: list, a: set) -> bool:
    return (all(m in a for m in mv) or len(mv) > 1 and any(m in a for m in mv)
            or len(mv) == 1 and any(x in a for x in MOOT.get(mv[0], ())) or 'mbs1' in mv and 'mbs2' not in a)


class ReplayProposer:
    """The model of the recorded run: on each incumbent the recorded call of its stage (ideas already in the incumbent
    or measured left out), the code review's patch first until it is measured; numbers as LLMProposer turns them into
    priors (capability 0.9)."""

    def __init__(self, sigma_scale: float):
        self.sigma_scale, self.calls, self.failed, self.views = sigma_scale, 0, 0, []

    def describe(self):
        return {'kind': 'ReplayProposer'}

    def hypotheses(self, engine, base, ledger=None):
        self.calls += 1
        self.views.append((base, json.loads(json.dumps(ledger, default=str))))
        c = engine.s.candidates[base]
        a = moves(c.config, c.overlay)
        done = {h.get('mechanism') for h in ledger or [] if h.get('status') not in UNTESTED_STATES}
        text = engine.s.hooks.candidate_files(c.overlay)[PATCH_FILE]
        out = []
        for h in [FIXTURE['review']] + FIXTURE['calls'][stage_of(a)]:
            mv = h.get('moves') or []
            if h['mechanism'] in done or moot(mv, a):
                continue
            g = float(h['pct']) / 100
            u = float(h.get('unc') or max(1.0, abs(g) * 50)) / 100
            patch = {PATCH_FILE: text + snippet(mv[0])} if h.get('patch', h is FIXTURE['review']) else None
            out.append({'mechanism': h['mechanism'], 'change': dict(h.get('change') or {}), 'patch': patch,
                        'mu': math.log1p(max(-0.9, g)), 'sigma': max(0.005, u * self.sigma_scale),
                        'p_feasible': min(1.0, max(0.0, float(h.get('p', 0.8)))), 'rationale': '',
                        'change_class': h.get('change_class')})
        return out


def q9u_experience() -> dict:
    """The run's experience plan: no data for any knob (coverage none), the transfer judgment, and its start
    configuration as four zero-size seeds and the composite (experience.transfer.apply)."""
    judged = {a['knob']: a for a in FIXTURE['judgment']}
    members = FIXTURE['composite']['members']
    seeds = [{'knob': n, 'mechanism': n, 'change': dict(judged[n]['change']), 'level': 'partial', 'gain': 0.0,
              'evidence_ids': [], 'warnings': [], 'sign': '+', 'p_feasible': judged[n]['p_feasible'],
              'valid_on': {n: FIXTURE['reference'][n]}} for n in members]
    comp = dict(FIXTURE['composite'], warnings=[])
    return {'seeds': seeds, 'composite': comp, 'judgment': {'assessment': [dict(a) for a in FIXTURE['judgment']]}}


def q9u_variant_experience() -> dict:
    """The plan with more of the experience paths: a covered seed (data), a known-to-run move and a memory check."""
    exp = q9u_experience()
    exp['seeds'].append({'knob': 'dataloader_pin_memory', 'mechanism': 'pinned_h2d', 'level': 'covered', 'gain': 0.012,
                         'change': {'dataloader_pin_memory': False}, 'evidence_ids': ['pack/pin'], 'warnings': []})
    exp['known_ok'] = [{'tf32': True}]
    exp['check'] = lambda cfg: (['memory: no recompute at micro-batch 2 is near the card']
                                if cfg.get('per_device_train_batch_size') == 2 and cfg.get('gradient_checkpointing') is False
                                else [])
    return exp


# the equivalence scenarios (test_hls_equivalence): hls before the fix (golden) and now must agree on all of them
SCENARIOS = {'recorded': (10, q9u_experience), 'long': (24, q9u_experience), 'variant': (16, q9u_variant_experience)}


def run_scenario(cls, name: str) -> dict:
    """Everything a strategy decided, showed the model (each call's ledger as a digest) and recorded on one scenario."""
    screens, exp = SCENARIOS[name]
    eng, st, tr = q9u_replay(lambda prop, **kw: cls(prop, **{**kw, 'experience': exp()}), screens=screens)
    views = [[base, hashlib.sha256(json.dumps(view, sort_keys=True).encode()).hexdigest()]
             for base, view in st.proposer.views]
    return json.loads(json.dumps({'decisions': eng.decisions, 'order': eng.order, 'ledger': st.ledger_view(),
                                  'model_views': views, 'describe': st.describe(),
                                  'residual': st.residual(eng), 'selected': tr.selected, 'stop': tr.stop_reason,
                                  'invalid': eng.s.invalid, 'stop_numbers': tr.extra['stop_numbers'],
                                  'calls': st.proposer.calls}, default=str))


def q9u_replay(make_strategy, screens: int = 10):
    """Run a strategy (factory(proposer, kappa0, probe_below, experience)) on the recorded job: (engine, strategy,
    trace)."""
    kp = search_params(FIXTURE['capability'])
    session = SimSession(FIXTURE['reference'], knobs=FIXTURE['knobs'], screens=screens, t0=FIXTURE['c0']['step_time'],
                         c0_gpu_hours=FIXTURE['c0']['gpu_hours'])
    full = [s['gpu_hours'] for s in FIXTURE['screens'] if s['fidelity'] == 'full']
    probe = next(s['gpu_hours'] for s in FIXTURE['screens'] if s['fidelity'] == 'probe')
    eng = SimEngine(session, q9u_effect, sigma_runs=[d['sigma_run'] for d in FIXTURE['decisions']], full_costs=full,
                    probe_cost=probe, validate=q9u_valid)
    strategy = make_strategy(ReplayProposer(kp['sigma_scale']), kappa0=kp['kappa0'], probe_below=kp['probe_below'],
                             experience=q9u_experience())
    return eng, strategy, eng.run(strategy)
