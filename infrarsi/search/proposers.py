"""Hypothesis generators for the search strategies: the LLM (real projects) and a simulated LLM (ToyTrain).

A hypothesis is {'mechanism', 'change': {knob: value}, 'patch': {file: text} | None, 'mu': predicted log gain vs the
base, 'sigma', 'p_feasible', 'rationale'}. Strategies decide what to test and when to stop; the proposer only supplies
candidate mechanisms and predictions (the LLM as a prior, the algorithm as the search).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re

from ..agent.context import estimate_tokens
from ..agent.state import Stateful
from .fold import HISTORY_CHARS, LEDGER_CHARS, fold_history, fold_ledger
from .token_governor import TokenGovernor


def _u(*key) -> float:
    return int(hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12], 16) / 16 ** 12


def _gauss(*key) -> float:
    u1, u2 = max(1e-12, _u(*key, 'a')), _u(*key, 'b')
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def mechanism_id(change: dict, patch: dict | None = None) -> str:
    parts = [f'{k}={v}' for k, v in sorted(change.items(), key=str)]
    if patch:
        parts += [f'patch:{line}' for text in patch.values() for line in text.split() if line]
    return ','.join(parts)


class SimProposer:
    """Simulated LLM on a ToySim (Fable consult §5.3): it knows each mechanism with probability q, predicts
    gain = true x b x LogNormal(0, rho) (b > 1 over-predicts), proposes plausible distractors at rate 1 - q, and has
    a fixed, deterministic answer for every (task, base, mechanism). `pairs` adds the two-knob moves it knows."""

    def __init__(self, sim, *, q: float = 0.7, b: float = 1.5, rho: float = 0.5, pairs: bool = True, seed: int = 0,
                 memory: bool = False):
        self.sim, self.q, self.b, self.rho, self.pairs, self.seed = sim, q, b, rho, pairs, seed
        self.memory = memory                    # hls3: declare each move's memory direction and GiB (as `memory` asks)
        self.ask_proof = False                  # hls3 (F10): asked to make each patch print its two self-proof lines
        self.proof_rate = 0.9                   # ... how often the simulated model follows that rule
        self.ask_defer = False                  # hls3 (F21): ... and its `deferred` line when its hook hands calls back
        self.defer_rate = 0.9
        self.prior_extra = None                 # candidates (esls): the prior's extra fields asked (families, terms)
        self.last_prior = None
        self.calls = 0

    def describe(self) -> dict:
        return {'kind': 'SimProposer', 'q': self.q, 'b': self.b, 'rho': self.rho, 'pairs': self.pairs,
                **({'memory': True} if self.memory else {})}

    def _memory(self, cfg: dict, change: dict, mech: str) -> dict:
        """hls3's memory declaration: the move's true per-GPU memory change with a 30 % error, the direction right when
        the model knows the mechanism (else 'unknown'); a knob whose memory effect the task hides (`hidden_memory`) is
        never declared - the model does not know that it frees memory."""
        if set(change) & set(getattr(self.sim, 'declared_neutral', ()) or ()):
            return {'mem_dir': 'neutral'}               # (the undeclared-enabler family: the model believes it neutral)
        if set(change) & set(getattr(self.sim, 'hidden_memory', ()) or ()) or not self.knows(f'mem:{mech}'):
            return {}
        try:
            m0, m1 = self.sim.memory(self.sim.full(cfg)), self.sim.memory(self.sim.full({**cfg, **change}))
        except ZeroDivisionError:                      # a layout with no data-parallel rank: nothing to declare
            return {}
        d = m1 - m0
        if abs(d) < 0.25:
            return {'mem_dir': 'neutral'}
        return {'mem_dir': 'spends' if d > 0 else 'frees',
                'mem_gib': round(abs(d) * math.exp(0.3 * _gauss(self.sim.task.seed, self.seed, 'mem', mech)), 2)}

    def _base(self, engine, base: str):
        c = engine.s.candidates[base]
        from .toytrain import _patches
        return dict(c.config), _patches(c.overlay)

    def knows(self, mech: str) -> bool:
        return _u(self.sim.task.seed, self.seed, 'knows', mech) < self.q

    def predict(self, base_key, mech: str, true_gain: float | None) -> tuple[float, float, float]:
        n1, n2 = _gauss(self.sim.task.seed, self.seed, base_key, mech, 1), _gauss(self.sim.task.seed, self.seed, base_key, mech, 2)
        if true_gain is None:                                  # infeasible: the model may still find it attractive
            return abs(0.04 + 0.03 * n1) * self.b, 0.05, 0.3
        if true_gain > 0:
            mu = true_gain * self.b * math.exp(self.rho * n1)
        else:
            mu = true_gain + 0.03 * self.rho * n2              # small losses are sometimes misread as small gains
        return mu, 0.01 + self.rho * abs(mu) * 0.5, 0.9

    def hypotheses(self, engine, base: str, ledger: list | None = None) -> list[dict]:
        self.calls += 1
        sim = self.sim
        cfg, patches = self._base(engine, base)
        t0, _ = sim.truth(cfg, patches)
        if t0 is None:
            return []
        knobs = sim.knobs()
        singles = [{k: v} for k in knobs for v in knobs[k] if v != cfg.get(k)]
        moves = [(m, None) for m in singles]
        hidden = set(getattr(sim, 'hidden_memory', ()) or ())   # (MemSim) an enabler whose effect the model does not know
        if self.pairs:
            moves += [({**a, **b}, None) for a, b in itertools.combinations(singles, 2) if set(a).isdisjoint(b)
                      and not (set(a) | set(b)) & hidden]
        moves += [({}, p) for p in sorted(sim.patch_ok) if p not in patches]
        base_key = repr(sorted(cfg.items(), key=str)) + repr(tuple(x for x in patches
                                                                   if not x.startswith(('proof:', 'defer:'))))
        # (hls3 F10: a patch's printing marker is no other base: the same draws as without it)
        out, unknown = [], []
        for change, patch in moves:
            mech = mechanism_id(change, {'patches.txt': patch} if patch else None)
            pl = tuple(sorted(set(patches) | ({patch} if patch else set())))
            meant = {'intended': True} if patch and getattr(sim, 'inert', None) else {}   # (the gain if its hook ran)
            t1, _ = sim.truth({**cfg, **change}, pl, **meant)
            t0m = sim.truth(cfg, patches, **meant)[0] if meant else t0
            g = math.log(t0m / t1) if t1 and t0m else None
            if not self.knows(mech):
                unknown.append((change, patch, mech, g))
                continue
            if len(change) == 2 and g is not None:
                # a known pair is worth proposing only when it is better than both of its halves (an enabler pair)
                halves = []
                for k, v in change.items():
                    th, _ = sim.truth({**cfg, k: v}, pl)
                    halves.append(math.log(t0 / th) if th else -1.0)
                if g <= max(halves) + 0.005:
                    continue
            mu, sigma, pf = self.predict(base_key, mech, g)
            if len(change) == 1 and set(change) & set(getattr(sim, 'optimistic', ()) or ()):   # (the enabler family: it
                mu = max(mu, abs(0.01 + 0.01 * _gauss(self.sim.task.seed, self.seed, base_key, mech, 4)))   # believes it
            out.append(self._hyp(change, patch, patches, mech, mu, sigma, pf))                   # mildly faster)
        n_dis = int(round((1 - self.q) * 0.3 * len(out)))
        ranked = sorted(unknown, key=lambda x: _u(self.sim.task.seed, self.seed, base_key, 'distractor', x[2]))
        for change, patch, mech, g in ranked[:n_dis]:
            mu = abs(0.03 + 0.02 * _gauss(self.sim.task.seed, self.seed, base_key, mech, 3)) * self.b
            out.append(self._hyp(change, patch, patches, mech, mu, 0.03, 0.7))
        if self.memory:
            for h in out:
                if h['change'] and not h['patch']:
                    h.update(self._memory(cfg, h['change'], h['mechanism']))
        return out

    def _hyp(self, change, patch, base_patches, mech, mu, sigma, pf) -> dict:
        files = None
        if patch:
            from .toytrain import _u
            proof = {f'proof:{patch}'} if self.ask_proof and _u(self.sim.task.seed, 'proof', patch, sorted(base_patches)) \
                < self.proof_rate else set()           # (hls3 F10: the toy's patches.txt names the patches that print)
            defer = {f'defer:{patch}'} if proof and self.ask_defer and _u(self.sim.task.seed, 'defer', patch,
                                                                          sorted(base_patches)) < self.defer_rate else set()
            files = {'patches.txt': '\n'.join(sorted(set(base_patches) | {patch} | proof | defer)) + '\n'}
        return {'mechanism': mech, 'change': dict(change), 'patch': files, 'mu': mu, 'sigma': sigma, 'p_feasible': pf,
                'rationale': 'simulated'}

    @staticmethod
    def merge_patches(base_files: dict, patches: list) -> dict:
        """The toy's patches.txt is a set of patch names, one per line: composites are the union."""
        lines = sorted({ln for p in patches for t in (p or {}).values() for ln in str(t).split()})
        return {'patches.txt': '\n'.join(lines) + '\n'}

    def knob_prior(self, engine, base: str) -> dict:
        """wcs's prior call, simulated (the answer in PRIOR_SYSTEM's format): for each knob value the model knows (q), its
        single-move gain over the reference x b x LogNormal(0, rho) with a spread, its memory as `_memory` declares it
        (a hidden enabler stays unknown); the crash pair as a conflict when known; three whole configurations - every
        value it rates positive (aggressive), the three best (balanced), the best (safe) - and the values it does not
        know left unknown. Deterministic per task."""
        self.calls += 1
        sim = self.sim
        cfg, patches = self._base(engine, base)
        t0, _ = sim.truth(cfg, patches)
        knobs = sim.knobs()
        levels, rated = [], []
        for k, vals in knobs.items():
            for v in vals:
                if v == cfg.get(k):
                    continue
                mech = mechanism_id({k: v})
                row = {'knob': k, 'value': v}
                if self.knows(mech) and t0:
                    t1, _ = sim.truth({**cfg, k: v}, patches)
                    g = math.log(t0 / t1) if t1 else None
                    mu, sigma, pf = self.predict('prior', mech, g)
                    row.update(gain_pct=round(100 * math.expm1(mu), 3), sd_pct=round(100 * sigma, 3), p_runs=pf,
                               why='simulated')
                    rated.append((mu, k, v))
                else:
                    row.update(gain_pct=None, sd_pct=None, why='unknown')
                if self.memory:
                    m = self._memory(cfg, {k: v}, mech)
                    row['memory'] = {'direction': m.get('mem_dir', 'unknown'), 'gib': m.get('mem_gib')}
                levels.append(row)
        inter = []
        pair = getattr(sim, 'crash_pair', None)
        if pair and self.knows('pair:' + '+'.join(pair)) and all(p in knobs for p in pair):
            inter.append({'members': [{'knob': p, 'value': True} for p in pair], 'kind': 'conflicts', 'why': 'simulated'})
        pos = sorted((r for r in rated if r[0] > 0), key=lambda r: -r[0])
        best = {}
        for mu, k, v in pos:
            best.setdefault(k, (mu, v))
        configs = []
        for name, n in (('aggressive', len(best)), ('balanced', 3), ('safe', 1)):
            ch = {k: v for k, (mu, v) in sorted(best.items(), key=lambda t: -t[1][0])[:n]}
            if ch:
                configs.append({'name': name, 'change': ch, 'why': 'simulated'})
        out = {'levels': levels, 'interactions': inter, 'configs': configs, 'notes': 'simulated prior'}
        if self.prior_extra:                    # (candidates: each value's family; the sim's memory terms as it knows
            fams = getattr(sim, 'families', None) or {}            # them - a hidden enabler's factor left out)
            for row in levels:
                row['family'] = fams.get(row['knob'])
            terms = getattr(sim, 'mem_terms', None)
            if callable(terms):
                out['mem_terms'] = terms(self)
        self.last_prior = out
        return out

    def factors(self, engine, base: str) -> list[dict]:
        """Two-level DOE factors: boolean knobs it knows, with the dependent pair bundled when known."""
        cfg, _ = self._base(engine, base)
        knobs = self.sim.knobs()
        flags = [k for k, v in knobs.items() if v == [False, True]]
        out, used = [], set()
        if {'seq_parallel', 'tp_overlap'} <= set(flags) and self.knows('bundle:seq_parallel+tp_overlap'):
            out.append({'name': 'sp+tp_overlap', 'plus': {'seq_parallel': not cfg.get('seq_parallel'),
                                                          'tp_overlap': not cfg.get('tp_overlap')}})
            used |= {'seq_parallel', 'tp_overlap'}
        for k in flags:
            if k not in used:
                out.append({'name': k, 'plus': {k: not cfg.get(k)}})
        return out


HYPOTHESIS_SCHEMA = {'type': 'object', 'required': ['hypotheses'], 'properties': {'hypotheses': {'type': 'array', 'items': {
    'type': 'object', 'required': ['mechanism', 'change', 'predicted_gain_pct'],
    'properties': {'mechanism': {'type': 'string'}, 'change': {'type': 'object'},
                   'patch': {'type': ['object', 'null', 'string']},          # "" for none (D1-20, Sonnet)
                   'edits': {'type': ['array', 'null'], 'items': {
                       'type': 'object', 'required': ['file', 'old', 'new'],
                       'properties': {'file': {'type': 'string'}, 'old': {'type': 'string'}, 'new': {'type': 'string'}}}},
                   'predicted_gain_pct': {'type': 'number'},
                   'uncertainty_pct': {'type': 'number'}, 'p_feasible': {'type': 'number'},
                   'change_class': {'type': ['string', 'null']},
                   'rationale': {'type': 'string'}}}}}}
CHANGE_CLASSES = ('execution', 'numerics', 'algorithmic')


def change_class(v) -> str | None:
    """The proposer's label of what a change touches (reported with the loss deviation it caused, never a gate)."""
    v = str(v or '').strip().lower()
    return v if v in CHANGE_CLASSES else None

HYPOTHESIS_SYSTEM = """You generate optimization hypotheses for InfraRSI, which makes one existing training job faster.
Speed (time per update of the fixed workload) is the only objective; a verifier rejects any change that alters the
job (workload, data, model, optimizer steps) or moves the loss beyond the tolerance, and failed checks are shown to you.
For the current incumbent, list the mechanisms worth testing that are NOT already tested (see the ledger), each as a
concrete change of the listed knobs, a patch of the mutable files, or both, with your predicted speed gain in percent,
your uncertainty in percent, and the probability that it runs at all (memory, crashes).
Prefer one mechanism per hypothesis; give two-knob changes only when one knob enables the other (e.g. less recompute
needs a smaller micro-batch). Be calibrated: the ledger shows how your earlier predictions compared with measurements.
Label each with `change_class`: "execution" when only how the same arithmetic runs changes (overlap, data loading,
recompute, CUDA graphs, allocator: the loss should not move), "numerics" when the algorithm stays but the arithmetic
changes (precision, fused or other kernels, another parallel layout or micro-batch split: small loss differences),
"algorithmic" when what is computed changes (the verifier rejects it).
Reply with exactly one JSON object: {"hypotheses":[{"mechanism","change","patch","edits","predicted_gain_pct",
"uncertainty_pct","p_feasible","change_class","rationale"}]} with at most 12 hypotheses (`change` is {} for a pure
patch)."""

# when the strategy calibrates the predictions (HLS's kappa) the model is told not to, and is shown measurements without
# the predictions made for them: shown its misses it discounts, and kappa shrinks the discounted values again (H2b arm A)
SELF_CALIBRATION = 'Be calibrated: the ledger shows how your earlier predictions compared with measurements.'
SEARCH_CALIBRATES = """Predict each gain on its merits for this job, with the measurements as evidence about it; do not
discount predictions because earlier ones missed: InfraRSI calibrates their scale itself."""


def unpaired(row: dict) -> dict:
    """A measured ledger row without the prediction that was made for it."""
    return {k: v for k, v in row.items() if k != 'predicted_gain_pct'} if row.get('measured_gain_pct') is not None else row


PATCH_RULES = """either as `edits`, a list of {"file", "old", "new"} where `old` is an exact excerpt of the shown text that
occurs once in the file, or as `patch` = {"<path of the file>": "<its complete new text>"}, not both. A `patch` with
a file's complete text only for a file NOT listed in `code.edits_only`: those files are long, and a reply that writes
one out in full is cut at the output limit and lost - change them with `edits`. Keep a patch small and keep the
computation the same: frozen files, the model, the data, the loss and the optimizer's arithmetic must not change, and
code that measures, logs, limits or simulates time, memory or the workload (timers, sleeps, log lines, counters, caps)
is not a cost: never change it."""

CODE_RULE = """
CODE: `code` holds the current text of the files you may patch (`mutable`) and of files you may only read
(`readable`); long files are shown as head and tail. When time goes where no knob reaches (a slow loop or kernel, a
host-device sync, unfused element-wise work, data loading in a mutable file), propose a patch, """ + PATCH_RULES + """
`code_review`, when present, is the review of this code at the start of the search: its patches are queued already,
do not repeat them."""

REVIEW_SYSTEM = """You review the code of one training job for InfraRSI, which makes the job faster without changing
it: speed (time per update of the fixed workload) is the only objective. Read the files you may patch (`code.mutable`)
and the read-only ones (`code.readable`) next to the reference run's receipt and the knobs the search already changes.
List the costs in the code that make an update slower than it needs to be: host-device syncs, per-step allocations or
cache flushes, slow kernels or attention implementations chosen in code, Python loops over tensors, unfused
element-wise work, data work on the critical path, needless copies. For each give the file, the issue, the knob that
already reaches it (null when none does) and, when only code reaches it, the change, """ + PATCH_RULES + """
With a change give your predicted speed gain in percent, your uncertainty in percent and the probability that it runs
(null without a change).
Then decide: verdict "patch" when at least one finding needs a code change worth a test run, else "no_patch" (the
search then changes knobs only), and say why in `summary`. Reply with exactly one JSON object: {"verdict", "summary",
"findings": [{"file", "issue", "knob", "edits", "patch", "predicted_gain_pct", "uncertainty_pct", "p_feasible",
"change_class"}]} (change_class: "execution", "numerics" or "algorithmic", as for hypotheses)."""

REVIEW_SCHEMA = {'type': 'object', 'required': ['verdict', 'findings'], 'properties': {
    'verdict': {'type': 'string', 'enum': ['patch', 'no_patch']}, 'summary': {'type': 'string'},
    'findings': {'type': 'array', 'items': {
        'type': 'object', 'required': ['file', 'issue'],
        'properties': {'file': {'type': 'string'}, 'issue': {'type': 'string'}, 'knob': {'type': ['string', 'null']},
                       'edits': HYPOTHESIS_SCHEMA['properties']['hypotheses']['items']['properties']['edits'],
                       'patch': {'type': ['object', 'null', 'string']}, 'predicted_gain_pct': {'type': ['number', 'null']},
                       'uncertainty_pct': {'type': ['number', 'null']}, 'p_feasible': {'type': ['number', 'null']},
                       'change_class': {'type': ['string', 'null']}}}}}}

CODE_TOKENS = 12000                 # file text in one hypothesis prompt: mutable files first, then readable ones
# a file whose text takes more than this share of the answer cap is patched with `edits` only (D2-R204d: Qwen3-30B wrote
# the 230-line train_wan.py, 3,600 tokens, out in full in five hypotheses of one reply: 67 KB, cut at 16k tokens, three
# calls lost); with the 16,000-token code cap: 2,000 tokens, about 7 KB or 150 lines
FULL_TEXT_SHARE = 8


def _tokens(text: str) -> int:
    return estimate_tokens(json.dumps(text, ensure_ascii=False))       # as the text travels: inside the JSON payload


def _excerpt(text: str, budget: int) -> str:
    """Head and tail of `text` within `budget` tokens: whole lines while they fit, an over-long first line cut."""
    lines = text.splitlines()
    half = max(0, budget - 40) // 2

    def take(seq):
        out, used = [], 0
        for ln in seq:
            t = _tokens(ln) + 1
            if used + t > half:
                if not out:                            # one huge line (a generated table): keep its start only
                    while ln and _tokens(ln) + 2 > half:
                        ln = ln[:len(ln) // 2]
                    if ln:
                        out.append(ln + ' ...')
                break
            out.append(ln)
            used += t
        return out

    head = take(lines)
    tail = take(reversed(lines[len(head):]))[::-1] if len(head) < len(lines) else []
    hidden = len(lines) - len(head) - len(tail)
    return '\n'.join(head + [f'... [{hidden} lines not shown] ...'] + tail)


def _tighten(code: dict, files: dict, cap: int) -> dict:
    """`code` with every mutable file whose text takes more than the share of a `cap`-token answer in edits_only (the
    same object when nothing is added)."""
    eo = set(code.get('edits_only') or [])
    more = eo | {p for p, t in files.items() if _tokens(t or '') > cap // FULL_TEXT_SHARE}
    return code if more == eo else {**code, 'edits_only': sorted(more)}


def code_view(hooks, overlay: dict | None, budget_tokens: int = CODE_TOKENS, full_text_tokens: int | None = None) -> dict | None:
    """The files the model may patch (the incumbent's version) and read, within a token budget: whole files while they
    fit, then one head-and-tail excerpt, the rest by name and size. `edits_only` names the mutable files whose text
    takes more than `full_text_tokens` (a whole-file `patch` of one would not fit the answer: edits only). None when
    nothing is mutable."""
    mutable = hooks.candidate_files(overlay) if hasattr(hooks, 'candidate_files') else {}
    if not mutable:
        return None
    readable = {k: v for k, v in (getattr(hooks, 'read_only_files', None) or {}).items() if k not in mutable}
    out = {'mutable': {}, 'readable': {}, 'excerpted': [], 'not_shown': [],
           'edits_only': sorted(p for p, t in mutable.items() if full_text_tokens and _tokens(t or '') > full_text_tokens)}
    left = int(budget_tokens)
    for kind, files in (('mutable', mutable), ('readable', readable)):
        for path in sorted(files):
            text = files[path] or ''
            need = _tokens(text)
            if need <= left:
                out[kind][path], left = text, left - need
            elif left >= 400:
                out[kind][path] = _excerpt(text, left)
                out['excerpted'].append(path)
                left = 0
            else:
                out['not_shown'].append(f'{path} ({len(text.splitlines())} lines)')
    return {k: v for k, v in out.items() if v}


def apply_edits(files: dict, edits, patch: dict | None = None) -> dict | None:
    """{file: new text} from search/replace edits on the current files (after a whole-file `patch` when both are given);
    None when an edit does not apply (unknown file, or `old` not found exactly once)."""
    cur = {**files, **(patch or {})}
    touched = dict(patch or {})
    for e in edits if isinstance(edits, list) else []:
        if not isinstance(e, dict) or not all(isinstance(e.get(k), str) for k in ('file', 'old', 'new')):
            return None
        text = cur.get(e['file'])
        if text is None or not e['old'] or text.count(e['old']) != 1:
            return None
        cur[e['file']] = touched[e['file']] = text.replace(e['old'], e['new'])
    return touched or None


def patch_of(h: dict, files: dict) -> tuple[dict | None, bool]:
    """(the {file: new text} patch of a hypothesis or review finding, False when its edits do not apply). A `patch` is
    {path: complete text}, or one of the shapes models also send for it: {"file", "old", "new"} (an edit) and {"file",
    "patch" | "new" | "text" | "content"} (a whole file); `edits` win over a `patch` of the same file (trial C1), a
    `patch` of another file is kept, and files whose text does not change are left out (C4b: edits with old == new)."""
    if isinstance(h.get('patch'), str) and h['patch'].strip():
        return None, False                              # a diff or a description: not a patch InfraRSI can apply
    edits = list(h['edits']) if isinstance(h.get('edits'), list) else []
    raw = h.get('patch') if isinstance(h.get('patch'), dict) else {}
    patch = dict(raw) or None
    f = raw.get('file')
    if isinstance(f, str) and all(isinstance(v, str) for v in raw.values()):
        patch = None
        if isinstance(raw.get('old'), str) and isinstance(raw.get('new'), str):
            if not any(isinstance(e, dict) and e.get('file') == f for e in edits):
                edits.append({'file': f, 'old': raw['old'], 'new': raw['new']})
        else:
            text = [raw[k] for k in ('patch', 'new', 'text', 'content') if isinstance(raw.get(k), str)]
            patch = {f: text[0]} if len(raw) == 2 and text else dict(raw)
    if edits:
        out = apply_edits(files, edits)
        if out is None:
            return None, False
        patch = {**(patch or {}), **out}
    return {k: v for k, v in (patch or {}).items() if files.get(k) != v} or None, True


EDITS_HINT = ('give a code change as `patch` = {"<file>": "<its complete new text>"} or as `edits` = [{"file", "old", '
              '"new"}] with `old` an exact excerpt that occurs once in the file - no other shape')
_PLACEHOLDER = re.compile(r'^\([^\n]{0,80}\)$|^(?:\.{3}|…)$|^see\b[^\n]{0,80}$', re.I)   # "(see patch)", "..."


def relaxed_schema(schema: dict) -> dict:
    """F14: the answer's own shape - {"hypotheses": [...]} - each hypothesis validated on its own afterwards
    (accept_hypotheses), so one bad hypothesis no longer costs the answer."""
    out = json.loads(json.dumps(schema))
    out['properties']['hypotheses'] = {'type': 'array'}
    return out


def salvage_edits(h: dict, files: dict) -> dict | None:
    """F14b: the {file: new text} of a hypothesis whose `edits` are not {file, old, new}: each edit names a file of
    the incumbent's and how it changes it (`op`); its text is its own (`new`, `text`, `content`, `code`), else - none, or
    a placeholder such as "(see patch)" - the hypothesis's `patch` text (one edit at most takes it); `append` / `prepend`
    add the text to the file's current text, any other operation replaces the file whole; a proper {file, old, new} edit
    among them applies as such. None when an edit names no file of the incumbent's or no text is found."""
    edits = h.get('edits')
    if not isinstance(edits, list) or not edits:
        return None
    raw = h['patch'] if isinstance(h.get('patch'), str) and h['patch'].strip() else None
    cur, out, used = dict(files or {}), {}, False
    for e in edits:
        if not isinstance(e, dict) or not isinstance(e.get('file'), str) or e['file'] not in cur:
            return None
        f = e['file']
        if isinstance(e.get('old'), str) and isinstance(e.get('new'), str):
            if not e['old'] or cur[f].count(e['old']) != 1:
                return None
            cur[f] = out[f] = cur[f].replace(e['old'], e['new'])
            continue
        body = next((e[k] for k in ('new', 'text', 'content', 'code') if isinstance(e.get(k), str) and e[k].strip()
                     and not _PLACEHOLDER.match(e[k].strip())), None)
        if body is None:
            if raw is None or used:
                return None
            body, used = raw, True
        op = str(e.get('op') or e.get('operation') or e.get('action') or e.get('mode') or '').strip().lower()
        text = body if body.endswith('\n') else body + '\n'
        if op.startswith('append'):
            text = cur[f] + ('' if not cur[f] or cur[f].endswith('\n') else '\n') + text
        elif op.startswith('prepend'):
            text = text + cur[f]
        cur[f] = out[f] = text
    return out or None


def patch_over_edits(h: dict, files: dict) -> dict | None:
    """F14b: a hypothesis whose only problems are its `edits` while its `patch` is a usable {file: complete text} object
    (Codex HLS5 r1): that patch - an edit naming a file the patch gives whole ("(see patch)") is the patch's - with the
    edits of other files salvaged from their text (salvage_edits); None when the patch is no such object (a string, an
    edit's {"file", ...} shape) or an edit of another file cannot be read. `edits` that are no list at all (`true`, a
    string) are no edits: the patch alone (Codex HLS5 r2: never an exception - a sibling hypothesis survives)."""
    patch = h.get('patch')
    if not (isinstance(patch, dict) and patch and 'file' not in patch
            and all(isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in patch.items())):
        return None
    edits = h.get('edits') if isinstance(h.get('edits'), list) else []
    rest = [e for e in edits if not (isinstance(e, dict) and isinstance(e.get('file'), str) and e['file'] in patch)]
    if not rest:
        return dict(patch)
    extra = salvage_edits({**h, 'edits': rest, 'patch': None}, files)
    return {**extra, **patch} if extra else None


def accept_hypotheses(items, schema: dict, files: dict | None = None) -> tuple[list, list, list]:
    """F14b: each hypothesis validated on its own against the item schema - a valid one kept as it is; one whose
    only problems are its `edits` kept with its usable `patch` object (patch_over_edits), else rewritten from the edits'
    text or its `patch` text (salvage_edits: a `patch` wins over non-schema `edits`); any other dropped. Returns (kept,
    dropped [{index, mechanism, problems}], rewritten [{index, mechanism, files}])."""
    from ..agent.jsonschema_lite import validate
    item = schema['properties']['hypotheses']['items']
    kept, dropped, fixed = [], [], []
    for i, h in enumerate(items if isinstance(items, list) else []):
        path = f'$.hypotheses[{i}]'
        probs = validate(h, item, path)
        mech = str(h.get('mechanism') or '')[:160] if isinstance(h, dict) else None
        if probs and isinstance(h, dict) and all(p.startswith(path + '.edits') for p in probs):
            new = patch_over_edits(h, files or {}) or salvage_edits(h, files or {})
            h2 = {**h, 'edits': None, 'patch': new} if new else None
            if h2 is not None and not validate(h2, item, path):
                kept.append(h2)
                fixed.append({'index': i, 'mechanism': mech, 'files': sorted(new)})
                continue
        if probs:
            dropped.append({'index': i, 'mechanism': mech, 'problems': probs[:4]})
            continue
        kept.append(h)
    return kept, dropped, fixed


def explain_reply(res) -> str:
    """F14c: a retry's note - the exact reason the reply could not be used (where its JSON broke, every complaint of
    the schema), not only that it could not."""
    err = str(getattr(res, 'error', None) or '')
    text = res.text if isinstance(getattr(res, 'text', None), str) else ''
    if err.startswith('schema: '):
        why = "it did not have the answer's shape: " + err[len('schema: '):][:1500]
    elif not text.strip():
        why = 'it was empty'
    else:
        why = 'it was not one JSON object'
        body = text.strip()
        m = re.match(r'^```(?:json)?\s*\n(.*?)(?:\n```\s*)?$', body, re.S)
        try:
            json.loads(m.group(1) if m else body)
        except ValueError as exc:
            why += (f' ({getattr(exc, "msg", str(exc))[:120]} at character {getattr(exc, "pos", "?")} of {len(body)}; '
                    f'it began {body[:60]!r})')
    return (f'\n\nYour previous reply could not be used: {why}. Reply again with only the JSON object the instructions ask '
            f'for, whole and in one piece; {EDITS_HINT}.')


PHASES_RULE = """
PHASES: `incumbent.receipt.step_phases` splits the incumbent's time per update into measured phases (ms and share of
the update, largest first; `residual_ms` is what they do not cover). Prefer mechanisms that shrink a large phase or the
exposed part of one; a change aimed at a phase that is already small cannot gain more than that phase.
`incumbent.receipt.phase_targets` names, for the largest phases, the knobs of this job whose mechanism can shorten
them, and the knobs that act on every phase (layout, precision, recompute)."""

EXPERIENCE_RULE = """
EXPERIENCE: `experience` holds what other runs measured for these mechanisms (coverage per knob, the nearest
environments and their differences from this one, effects, known failures, rules). Ledger entries with source
"experience" are already queued from that data: do not repeat them; use the rest as evidence for your predictions."""

ACCOUNTING_RULE = """
ACCOUNTING: `accounting` is a closed-form estimate for the incumbent (per GPU, +-25 %): its memory and the terms of it
against the card's capacity, and the ideal update time of its model FLOPs at the GPUs' matrix throughput (mfu = ideal /
measured). A change that adds more memory than `margin_gib` needs a probe before it can count; the gap between the
ideal and the measured update is communication, input, kernels and recompute together (the phases, when given, say
which), not room that only GEMM-side changes can take."""


MENU_RULE = """
MENU MODE: do not invent changes. Rate moves from the `menu` list in the input: copy a move's `change` exactly, and give
your predicted gain, uncertainty and feasibility for each move you think is worth testing (at most the requested number)."""

# hls2 (FABLE_DESIGN_SEARCH_FIX C2): the ledger rows carry the incumbent they were proposed on and say whether they are
# still queued; an idea of an earlier incumbent that was never tested is not queued and must be proposed again
LEDGER_STATES_RULE = """
LEDGER STATES: each ledger row names the incumbent it was proposed on (`base`). "queued": open on the current incumbent,
it will be tested - do not repeat it. "superseded": proposed on an earlier incumbent and never tested (`why` says why);
it is NOT queued - propose it again for the current incumbent if it is still worth testing. "unrunnable": that knob
value failed to start twice in this search - do not propose it again."""
UNTESTED_STATES = ('open', 'queued', 'superseded', 'deferred')   # ledger rows (as the model sees them) no run measured

# hls3 (FABLE_TAKEAWAYS_20261030 section 4, search/hls3.py): what its components ask of the model and explain to it, each
# only while its component is on (proposer_flags); hls and hls2 send none of it
MEMORY_DECLARE_RULE = """
MEMORY: give each hypothesis `memory`: {"direction": "frees" | "spends" | "neutral" | "unknown", "gib": the GiB per GPU it
frees or adds relative to the incumbent, null when you cannot say} - the search reads what fits from it and from the runs."""
MEMORY_VIEW_RULE = """
MEMORY VIEW: `memory_view` holds the card's capacity, the least memory per GPU the training state needs sharded over every
GPU, what each run measured (its peak, or for an out-of-memory run the allocation that failed, the memory free then and the
least it was short by), the incumbent's moves the search predicts will not fit ("deferred": they run once a change frees
enough - propose such a change, or the move together with one) and values pruned as impossible on this card."""
FAILURES_RULE = """
FAILURES: `failures` lists this search's failed runs by class: out of memory, compiler, collective, killed at the launch
timeout (start_cap), hang, crash, numerics, and thrash (it ran but at least 1.5x slower than its base near the memory cap:
the edge of what fits, not a slow move). `failed_together` lists changes that failed in one run: each is run separately
before they are combined again."""
VARIANCE_RULE = """
VARIANCE: `variance` says how far the reference's mean update is above its median and how many updates of the window were
stalls (at least `stall_factor` x the median). The search compares medians; mark a hypothesis whose gain comes from
removing such stalls, not from shortening the typical update, with "variance": true - it is judged by the mean and tried
early."""
REBASE_RULE = """
RE-BASING: a change accepted on an earlier incumbent may be redundant on this one (the same cost removed twice), and a move
rejected on an earlier incumbent may pay on this one: you may propose a revert, or such a move again."""
SITE_RULE = """
SITE: before a patch, find its site - the function or kernel path whose time it removes and the evidence that it dominates
(a log note, a phase, an allocation size) - give it as `site`, and the smallest patch that removes that cost (the search
weighs the expected gain per changed line)."""
INERT_RULE = """
INERT PATCHES: `inert_patches` lists patches that measured no effect although they were predicted to: the hook may never
have run. A new patch at such a site must log one greppable line when its code runs (a print at the hook), or leave that
site; without it, it waits until the other moves ran."""
PROOF_RULE = """
PROOF: every patch prints two greppable lines - when it installs, `[infrarsi-patch] <site>: found <what it found> -> set
<what it set>`, and the first time the patched code runs, `[infrarsi-patch] <site>: active` - and quotes in
`evidence_note` the job-log note it removes, if any, as the log prints it (`job_log_notes`). Its probe runs a few
updates; if the patched code first runs later, give `active_from_update` (the update, 1 to 8: code that first runs later
cannot be proven). After its probe a patch is inert - no full run; it comes back in `inert_patches` with the reason, the
lines it printed and its code - when a line is missing, when found equals set, or when the note it quotes is still
reported after it installed."""
PROOF_DEFER_RULE = """
Print `active` only after the new code path has run - not on entering a wrapper that may still hand the call back. When
your replacement hands a call back to the original (an argument, a shape or a mode it does not handle), print once per
process `[infrarsi-patch] <site>: deferred (<which argument or condition>)`: an installation whose calls all went
back is inert with that reason, and a note still reported after `active` means it defers on some calls."""
WARMUP_RULE = """
WARM-UP: a probe's updates after the second are compared with its base's same updates; much slower ones, on a
configuration that may spend memory, end the move without a full run. If your change needs more updates to reach its
speed (compiling, autotuning), give `warmup_updates`: how many (1 to 6; a larger value is not supported and is ignored)."""
LIBRARY_RULE = """
LIBRARY: `library_source` holds read-only excerpts of the library code this job imports, at the names its job-log notes
and the patch sites give: what the code you patch calls or replaces (its decorators, fallbacks, the arguments its callers
pass). It is the environment's text, not this job's files: read it, never patch it."""
COVERAGE_RULE = """
COVERAGE: moves predicted below the pair noise are tested together as one bundle (`coverage` lists such bundles and their
fate); a member of a bundle that did not move the incumbent is measured together, not alone."""


# What a strategy may ask of its LLMProposer (`features`; a strategy class's proposer_features(), search/__init__._engine):
# each only while asked - hls and hls2 ask none. These four serve any strategy that probes code (hls3 asks them by its
# components; the others above - memory, memory_view, failures, together, variance, sites, rebase, inert, warmup,
# coverage - are hls3's own fields and rules, by the same mechanism).
FEATURES = {
    'parse': 'F14: the answer read robustly - a reply split over the CLI\'s resume turn joined, each hypothesis validated '
             'on its own (non-schema edits rewritten), one re-ask with the exact errors when none is usable; the dropped '
             'ones are shown to the next call (`dropped_hypotheses`)',
    'self_proof': 'F10: each patch prints its `found -> set` and `active` lines and quotes the note it removes '
                  '(`evidence_note`, `active_from_update`); the strategy judges them after the probe (search/proof.py)',
    'deferred': 'F21: ... and its `deferred (<why>)` line when its replacement hands a call back (with self_proof)',
    'library_source': 'D7: read-only excerpts of the library source the job imports next to the code (the strategy\'s '
                      'context may give its own; else at the names the job-log notes give: search/library.py)',
}


# wcs (FABLE_ARCH_REVIEW_20261031 section 4, search/wcs.py): the one prior call over the whole knob table, and the patch-
# only mode of the hypothesis call (the knob side is the search's own model)
PRIOR_SYSTEM = """You give InfraRSI a prior over one training job's knob table before any run. The job (task, reference
configuration, its receipt) and the table (every knob with its values; the reference's value marked) are given. For
every value other than the reference's, estimate what setting it alone (other knobs as in the reference) would do:
`gain_pct` - the expected speed gain in percent of the reference's update time (negative when slower; null when you
cannot say), `sd_pct` - your one-sigma uncertainty of that gain, `memory` - {"direction": "frees" | "spends" | "neutral" |
"unknown", "gib": the GiB per GPU it frees or adds, null when you cannot say}, `warmup_updates` - the updates its first
run spends warming up (compiling, autotuning; null when none), `p_runs` - the probability that it runs at all alone,
`why` - one line. Say "unknown" (null) rather than guess: the search measures what you do not know.
Then `interactions`: sets of two to four values that "enables" each other (one frees the memory the other needs; a gain
that only appears together), "conflicts" (they crash or fail to start together) or are "redundant" (the same cost
removed twice). For an "enables" set give `gain_pct`: the gain the members bring together beyond their own single
gains (when a value only pays with another, put the benefit here, not on the value) and `sd_pct`. Then `configs`: up to
four complete configurations worth running first, each as the knob values that differ from the reference - the most
aggressive you expect to fit, a balanced one, a safe one, and one you are unsure of but want tested - each with
`gain_pct`, its expected gain over the reference, and `sd_pct`; the best of them runs on the first screen. Reply with
exactly one JSON object: {"levels":[{"knob","value","gain_pct","sd_pct","memory":{"direction","gib"},"warmup_updates",
"p_runs","why"}],"interactions":[{"members":[{"knob","value"}],"kind","gain_pct","sd_pct","why"}],
"configs":[{"name","change":{"knob":"value"},"gain_pct","sd_pct","why"}],"notes":""}."""
PRIOR_SCHEMA = {'type': 'object', 'required': ['levels'], 'properties': {
    'levels': {'type': 'array', 'items': {'type': 'object', 'required': ['knob', 'value'], 'properties': {
        'knob': {'type': 'string'}, 'gain_pct': {'type': ['number', 'null']}, 'sd_pct': {'type': ['number', 'null']},
        'memory': {'type': ['object', 'null']}, 'warmup_updates': {'type': ['integer', 'null']},
        'p_runs': {'type': ['number', 'null']}, 'why': {'type': ['string', 'null']}}}},
    'interactions': {'type': ['array', 'null'], 'items': {'type': 'object', 'properties': {
        'members': {'type': 'array', 'items': {'type': 'object', 'properties': {'knob': {'type': 'string'}}}},
        'kind': {'type': ['string', 'null']}, 'gain_pct': {'type': ['number', 'null']}, 'sd_pct': {'type': ['number', 'null']},
        'why': {'type': ['string', 'null']}}}},
    'configs': {'type': ['array', 'null'], 'items': {'type': 'object', 'properties': {
        'name': {'type': ['string', 'null']}, 'change': {'type': 'object'}, 'gain_pct': {'type': ['number', 'null']},
        'sd_pct': {'type': ['number', 'null']}, 'why': {'type': ['string', 'null']}}}},
    'notes': {'type': ['string', 'null']}}}
PATCH_ONLY_RULE = """
PATCHES ONLY: the knobs are searched by another method; propose only code patches of the mutable files (`change` must be
{}), at most 3, the largest expected gain per changed line first; a hypothesis without a patch is dropped."""


def hls3_schema(schema: dict, flags: dict, *, review: bool = False) -> dict:
    """`schema` (hypotheses, or the review's findings) with the fields a proposer's features ask for (FEATURES; hls3's own
    fields too); the same object without."""
    extra = {}
    if flags.get('memory'):
        extra['memory'] = {'type': ['object', 'null'], 'properties': {'direction': {'type': ['string', 'null']},
                                                                      'gib': {'type': ['number', 'null']}}}
    if flags.get('variance') and not review:
        extra['variance'] = {'type': ['boolean', 'null']}
    if flags.get('sites'):
        extra['site'] = {'type': ['string', 'null']}
    from .hls3 import WARMUP_MAX                            # (the supported declarations: Codex HLS4R2 r2)
    from .proof import PROBE_STEPS_MAX
    if flags.get('self_proof'):
        extra['evidence_note'] = {'type': ['string', 'null']}
        extra['active_from_update'] = {'type': ['integer', 'null'], 'minimum': 1, 'maximum': PROBE_STEPS_MAX}
    if flags.get('warmup'):
        extra['warmup_updates'] = {'type': ['integer', 'null'], 'minimum': 1, 'maximum': WARMUP_MAX}
    if not extra:
        return schema
    out = json.loads(json.dumps(schema))
    out['properties']['findings' if review else 'hypotheses']['items']['properties'].update(extra)
    return out


def hls3_fields(h: dict, flags: dict) -> dict:
    """The feature fields of one hypothesis or finding as the strategy reads them: mem_dir / mem_gib (declared memory),
    variance (a stall remover), site, evidence_note / active_from_update (self_proof), warmup_updates."""
    out = {}
    m = h.get('memory')
    if flags.get('memory') and isinstance(m, dict):
        d = str(m.get('direction') or '').strip().lower()
        if d in ('frees', 'spends', 'neutral'):
            out['mem_dir'] = d
            g = m.get('gib')
            if isinstance(g, (int, float)) and not isinstance(g, bool) and math.isfinite(g):
                out['mem_gib'] = abs(float(g))
    if flags.get('variance') and h.get('variance') is True:
        out['variance'] = True
    if flags.get('sites') and isinstance(h.get('site'), str) and h['site'].strip():
        out['site'] = h['site'].strip()[:200]
    if flags.get('self_proof') and isinstance(h.get('evidence_note'), str) and h['evidence_note'].strip():
        out['evidence_note'] = h['evidence_note'].strip()[:300]
    from .hls3 import WARMUP_MAX
    from .proof import PROBE_STEPS_MAX
    for key, flag, hi in (('active_from_update', 'self_proof', PROBE_STEPS_MAX), ('warmup_updates', 'warmup', WARMUP_MAX)):
        v = h.get(key)
        if not flags.get(flag) or v is None:
            continue
        if isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= hi:
            out[key] = v
        else:                                          # (unsupported: ignored, and said - it exempts from nothing)
            out.setdefault('rejected', []).append(f'{key}={v!r}: supported 1 to {hi}; ignored')
    return out


feature_schema, feature_fields = hls3_schema, hls3_fields    # (the names for a strategy other than hls3)


def author_field(item: dict) -> dict:
    """Which model of a debate wrote this item ({} outside a debate): its patch is checked by the other model."""
    from ..agent.debate import AUTHOR_KEY
    return {'author': item[AUTHOR_KEY]} if item.get(AUTHOR_KEY) else {}


def _allowances(s, driver, text: str, out_tokens: int) -> dict | None:
    from ..agent.optimize import call_allowances, max_output_tokens
    kw = call_allowances(s, driver, text, out_tokens)
    if (kw is not None and TokenGovernor(s.ledger).zone == 'wind_down'
            and kw['max_output_tokens'] < min(1024, out_tokens, max_output_tokens(driver))):
        return None
    return kw


def _say_error(engine, res, what: str) -> None:
    """A failed call (endpoint down, authentication, a refused request) in the progress output: the search goes on
    without the model's proposals this round."""
    if res.error and not res.action:
        (getattr(engine, 'log', None) or (lambda m: None))(f'[search] {what} failed: {str(res.error)[:300]}; this round '
                                                           'goes on without the model\'s proposals')


def cut_at_limit(res) -> bool:
    """The reply stopped at the output limit before its JSON ended, in the words of each backend (openai_compat marks
    the attempt `cut`; anthropic_api 'hit max_tokens'; the CLI passes the API's 'output token maximum' through)."""
    rows = (res.meta or {}).get('attempts') if isinstance(res.meta, dict) else None
    if any(isinstance(r, dict) and r.get('cut') for r in rows or []):
        return True
    e = str(res.error or '')
    return 'cut at the output limit' in e or 'hit max_tokens' in e or 'output token maximum' in e


def written_in_full(text: str | None, files: dict) -> list[str]:
    """The files whose text a (cut) reply carries as a whole-file patch: their first lines, JSON-escaped as the reply
    would have them (a head too short to be telling is not matched)."""
    heads = {p: json.dumps(t[:160], ensure_ascii=False)[1:-1] for p, t in files.items() if t}
    return sorted(p for p, h in heads.items() if len(h) >= 40 and h in (text or ''))


class LLMProposer(Stateful):
    """Hypotheses from an LLM backend (agent.backends), one call per refresh; charged to the session's ledger.
    Capability knob: max_hypotheses per call, menu mode (the model only rates enumerated single-knob moves) and an
    inflation of its stated uncertainty (sigma_scale). search_calibrates: the strategy calibrates the predictions (HLS),
    so the model is asked for undiscounted ones and sees earlier runs without their predictions; Greedy, which uses the
    predictions as they are, keeps the self-calibrating prompt."""

    # search-state checkpoints (agent/state.py): the model handles (the budget driver / backend, the hooks, the accounting
    # view) and the strategy's context hook are the resuming run's; the call counts, the review and the caches are saved
    CHECKPOINT_LIVE = ('driver', 'hooks', 'accounting', 'context', '_review')
    patch_hypotheses = None     # v4 (FABLE_V4_BUDGET_20261102): the patches a patch-only call asks for, set by a v4
    #                             finalist from the spec's search.caps.patches_per_ask; None: three, as before
    bounded_ledger = False      # v4.1: the strategy passes its whole ledger and the call folds what does not fit
    #                             (search/fold.py, set by a finalist); False: the last 40 entries, as before

    def __init__(self, driver, hooks, *, max_tokens: int = 8000, max_hypotheses: int = 12, menu: bool = False,
                 sigma_scale: float = 1.0, prior: list | None = None, experience: dict | None = None,
                 search_calibrates: bool = False, code_tokens: int = CODE_TOKENS, code_max_tokens: int = 16000,
                 accounting=None, ledger_states: bool = False, features: dict | None = None, hls3: dict | None = None):
        self.driver, self.hooks, self.max_tokens = driver, hooks, max_tokens
        self.ledger_states = bool(ledger_states)       # hls2: the prompt explains queued / superseded / unrunnable rows
        # the features the strategy asked for (FEATURES, and hls3's own fields; `hls3`: the same, by its older name)
        self.flags = {k: True for k, v in {**(features or {}), **(hls3 or {})}.items() if v}
        self.context = None                            # (engine, base) -> the strategy's extra input fields (it sets it)
        self.lib_cache: dict = {}                      # D7: names -> the library excerpts read for them
        # answers with code edits are long: Opus overran 8000 tokens and the CLI split the answer into two turns (C3c)
        self.code_max_tokens = max(int(max_tokens), int(code_max_tokens))
        self.code_tokens = int(code_tokens)
        self.unusable_edits = 0                         # hypotheses dropped because an edit did not apply
        self.dropped: list[dict] = []                   # F14: hypotheses of an answer not used, and why
        self.rewritten: list[dict] = []                 # ... non-schema edits rewritten as whole-file patches
        self.joined = 0                                 # ... answers read from their joined text blocks
        self.prior = list(prior or [])                 # measured outcomes of earlier runs on this GPU type
        self.experience = experience                   # experience.store.payload: the store's view of this job
        self.accounting = accounting                   # experience.runs.accounting_for: (config, step time) -> view
        self.max_hypotheses, self.menu, self.sigma_scale = int(max_hypotheses), bool(menu), float(sigma_scale)
        self.search_calibrates = bool(search_calibrates)
        self.calls = 0
        self.failed = 0                                # hypothesis calls without a usable answer (HLS may ask again)
        self.review: dict | None = None                # the code review, once the model has seen the code
        self.review_patches: list[dict] = []           # its patches no strategy has tested yet
        self.review_files: dict = {}                   # the files they were written against
        self.patch_only = False                        # wcs: the hypothesis calls ask for patches only (knobs: its own model)
        self.prior_calls = 0                           # wcs: prior calls made (one per run)
        self.prior_extra = None                        # candidates (esls): (rule, top-level properties, level properties)
        #                                                appended to the prior call; None: wcs's call as it is
        self.last_prior = None                         # the prior call's answer as the model gave it (its extra fields)
        self.calls_left: int | None = None             # wcs: the model calls it may still make (None: no allowance)
        self.driver_calls = 0                          # every model call made (the review and the re-asks included)
        self.patch_checked: dict = {}                  # the debate's patch check: patch and base digest -> the patch kept

    @property
    def hls3(self) -> dict:
        """The features asked for (`flags`), by their older name."""
        return self.flags

    def describe(self) -> dict:
        return {'kind': 'LLMProposer', 'driver': getattr(self.driver, 'describe', lambda: {})() if self.driver else None,
                'unusable_edits': self.unusable_edits, 'failed_calls': self.failed,
                **({'code_review': self.review} if self.review else {}),
                **({'parse': {'joined': self.joined, 'dropped': len(self.dropped), 'rewritten': len(self.rewritten)}}
                   if self.flags.get('parse') else {}),
                **({'patch_only': True, 'prior_calls': self.prior_calls, 'driver_calls': self.driver_calls}
                   if self.patch_only or self.prior_calls else {})}

    def _spend(self) -> bool:
        """wcs: one model call against the allowance (`calls_left`); False when it is spent - the call is not made."""
        if self.calls_left is not None and self.calls_left <= 0:
            return False
        if self.calls_left is not None:
            self.calls_left -= 1
        self.driver_calls += 1
        return True

    def knob_prior(self, engine, base: str) -> dict | None:
        """wcs: one call for the model's prior over the whole knob table (PRIOR_SYSTEM): the answer as the model gave it
        (search/wcs.Prior parses it), None without a model, without tokens or without a usable answer. Charged to the
        session as role 'prior'."""
        s = engine.s
        if self.driver is None or s.ledger.used['llm_tokens'] >= s.ledger.budget.llm_tokens:
            return None
        from .knob_space import choices
        c = s.candidates[base]
        digest = self.hooks.receipt_digest(c.receipt, c.verdict, max_tokens=1500) if c.receipt else None
        table = {k: {'reference': c.config.get(k), 'values': list(v)} for k, v in choices(s.task).items()}
        payload = {'task': self.hooks.task_view(), 'reference': {'id': base, 'config': c.config, 'receipt': digest},
                   'knob_table': table, 'budget_left': s.budget_view()}
        if self.experience:
            payload['experience'] = self.experience
        acc = self.accounting(c.config, (c.verdict or {}).get('step_time')) if self.accounting else None
        if acc:
            payload['accounting'] = acc
        facts = getattr(self.hooks, 'memory_facts', None)
        if callable(facts):
            payload['memory_facts'] = facts()
        system = PRIOR_SYSTEM + (PHASES_RULE if (digest or {}).get('step_phases') else '') + \
            (EXPERIENCE_RULE if self.experience else '') + (ACCOUNTING_RULE if acc else '')
        schema = PRIOR_SCHEMA
        if self.prior_extra:                           # (candidates: more fields asked; wcs: none - its call unchanged)
            rule, top, level = self.prior_extra
            system += rule
            schema = json.loads(json.dumps(PRIOR_SCHEMA))
            schema['properties'].update(top or {})
            schema['properties']['levels']['items']['properties'].update(level or {})
        prompt = json.dumps(payload, ensure_ascii=False, default=str)
        kw = _allowances(s, self.driver, system + prompt, self.max_tokens)
        if kw is None:
            return None
        self.prior_calls += 1
        if not self._spend():
            return None
        self.calls += 1
        res = self.driver.complete(system, prompt, schema=schema, role='prior', **kw)
        s.charge_llm(res, 'prior')
        _say_error(engine, res, 'prior call')
        if not isinstance(res.action, dict):
            self.failed += 1
            return None
        self.last_prior = res.action
        return res.action

    def _hls3_rules(self, code: bool) -> str:
        """The rules of the hls3 fields this proposer asks for or shows (empty for hls and hls2)."""
        f = self.hls3
        return ((MEMORY_DECLARE_RULE if f.get('memory') else '') + (MEMORY_VIEW_RULE if f.get('memory_view') else '')
                + (FAILURES_RULE if f.get('failures') or f.get('together') else '')
                + (VARIANCE_RULE if f.get('variance') else '') + (REBASE_RULE if f.get('rebase') else '')
                + (SITE_RULE if f.get('sites') and code else '') + (PROOF_RULE if f.get('self_proof') and code else '')
                + (INERT_RULE if f.get('inert') and code else '') + (WARMUP_RULE if f.get('warmup') else '')
                + (COVERAGE_RULE if f.get('coverage') else ''))

    def _answer_cap(self, s) -> int:
        """The most a code answer of this call can be: the code cap, the backend's own cap, the tokens left - the cap
        the edits_only share is taken of (DF16_r1: with an 8,000-token backend a 1,300-token file stayed whole)."""
        from ..agent.optimize import max_output_tokens
        left = int(s.ledger.budget.llm_tokens) - int(s.ledger.used['llm_tokens'])
        return max(1, min(self.code_max_tokens, max_output_tokens(self.driver), left))


    def _library(self, engine, base: str, ctx: dict) -> dict | None:
        """D7 (the `library_source` feature): the strategy's own excerpts (its context's `library_source`, None for
        none), else those at the names the job-log notes give (search/library.source); None with the feature off."""
        if not self.flags.get('library_source'):
            return None
        if 'library_source' in (ctx or {}):
            return ctx['library_source'] or None
        from .library import source
        return source(engine, base, (), self.lib_cache)

    def _hls3_rules(self, code: bool) -> str:
        """The rules of the feature fields this proposer asks for or shows (empty for hls and hls2)."""
        f = self.flags
        return ((MEMORY_DECLARE_RULE if f.get('memory') else '') + (MEMORY_VIEW_RULE if f.get('memory_view') else '')
                + (FAILURES_RULE if f.get('failures') or f.get('together') else '')
                + (VARIANCE_RULE if f.get('variance') else '') + (REBASE_RULE if f.get('rebase') else '')
                + (SITE_RULE if f.get('sites') and code else '')
                + (PROOF_RULE + (PROOF_DEFER_RULE if f.get('deferred') else '') if f.get('self_proof') and code else '')
                + (INERT_RULE if f.get('inert') and code else '') + (WARMUP_RULE if f.get('warmup') else '')
                + (COVERAGE_RULE if f.get('coverage') else ''))

    def _answer_cap(self, s) -> int:
        """The most a code answer of this call can be: the code cap, the backend's own cap, the tokens left - the cap
        the edits_only share is taken of (DF16_r1: with an 8,000-token backend a 1,300-token file stayed whole)."""
        from ..agent.optimize import max_output_tokens
        left = int(s.ledger.budget.llm_tokens) - int(s.ledger.used['llm_tokens'])
        return max(1, min(self.code_max_tokens, max_output_tokens(self.driver), left))

    def _review(self, engine, base: str, code: dict, digest) -> list[dict]:
        """The code review before the first hypotheses: one call that lists the costs only code can remove and decides
        whether the search patches at all (verdict no_patch: later calls show no code). Returns the patches it found as
        hypotheses; a review without a usable answer leaves the code visible to the hypothesis calls."""
        s, c = engine.s, engine.s.candidates[base]
        payload = {'task': self.hooks.task_view(), 'reference': {'id': base, 'config': c.config, 'receipt': digest},
                   'code': code}
        system = REVIEW_SYSTEM + (PHASES_RULE if (digest or {}).get('step_phases') else '')
        if self.flags.get('sites'):                    # hls3 (S-8): the site first, the gain per changed line
            system += SITE_RULE
        if self.flags.get('self_proof'):               # F10: a finding's patch proves at its probe it took effect
            system += PROOF_RULE + (PROOF_DEFER_RULE if self.flags.get('deferred') else '')   # (F21: and its deferrals)
        if self.flags.get('warmup'):                   # hls3 (F12): a change that warms up longer says so
            system += WARMUP_RULE
        if self.flags.get('memory'):                   # hls3 (S-1): a finding's patch may free memory too
            system += MEMORY_DECLARE_RULE.replace('each hypothesis', 'each finding with a change')
        lib = self._library(engine, base, (self.context(engine, base) or {}) if self.context is not None else {})
        if lib:                                        # D7: the library's text the job's notes name
            payload['library_source'] = lib
            system += LIBRARY_RULE
        body = json.dumps(payload, ensure_ascii=False, default=str)
        kw = _allowances(s, self.driver, system + body, self.code_max_tokens)
        if kw is None:                                 # the call does not fit what the budget has left
            return []
        files = self.hooks.candidate_files(c.overlay) if hasattr(self.hooks, 'candidate_files') else {}
        while True:                                    # the tokens left after the input cap it lower still - and the
            tight = _tighten(payload['code'], files, kw['max_output_tokens'])   # names it adds lengthen the input:
            if tight is payload['code']:               # until nothing more is added (it only grows)
                break
            payload['code'] = tight
            body = json.dumps(payload, ensure_ascii=False, default=str)
            kw = _allowances(s, self.driver, system + body, self.code_max_tokens)
            if kw is None:
                return []
        if not self._spend():
            return []
        res = self.driver.complete(system, body, schema=hls3_schema(REVIEW_SCHEMA, self.flags, review=True),
                                   role='code_review', **kw)
        s.charge_llm(res, 'code_review')
        _say_error(engine, res, 'code review')
        a = res.action if isinstance(res.action, dict) else {}
        knobs = set((s.task.knob_schema or {}).get('properties') or {})
        verdict = a.get('verdict') if a.get('verdict') in ('patch', 'no_patch') else None
        findings, out = [], []
        for f in (a.get('findings') or [])[:12]:
            if not isinstance(f, dict):
                continue
            # only a knob of this job reaches a finding (C1b: Qwen3 wrote "knob": "None" for "no knob")
            row = {'file': str(f.get('file'))[:200], 'issue': str(f.get('issue'))[:300],
                   'knob': f['knob'] if isinstance(f.get('knob'), str) and f['knob'] in knobs else None, 'patch': None}
            if verdict == 'patch' and (f.get('edits') or f.get('patch')) and not row['knob']:
                patch, ok = patch_of(f, files)
                row['patch'] = 'queued' if patch else 'no change' if ok else 'edit did not apply'
                self.unusable_edits += 0 if ok else 1
                if patch:
                    g = float(f.get('predicted_gain_pct') or 0) / 100
                    u = float(f.get('uncertainty_pct') or max(1.0, abs(g) * 50)) / 100
                    out.append({'mechanism': f'review: {row["file"]}: {row["issue"]}'[:200], 'change': {}, 'patch': patch,
                                'mu': math.log1p(max(-0.9, g)), 'sigma': max(0.005, u * self.sigma_scale),
                                'p_feasible': min(1.0, max(0.0, float(f['p_feasible'] if isinstance(f.get('p_feasible'), (int, float))
                                                                      else 0.8))), 'rationale': row['issue'],
                                'change_class': change_class(f.get('change_class')), **hls3_fields(f, self.flags),
                                **author_field(f)})
            findings.append(row)
        self.review = {'verdict': verdict, 'summary': str(a.get('summary') or res.error or '')[:600], 'findings': findings}
        queued = [r['file'] for r in findings if r['patch'] == 'queued']
        (getattr(engine, 'log', None) or (lambda m: None))('[search] code review: ' + (
            'no usable answer; the hypothesis calls keep seeing the code' if verdict is None else
            f'{verdict}, {len(findings)} findings, {len(queued)} patches queued'
            + (f' ({", ".join(queued)})' if queued else '') + f': {self.review["summary"][:200]}'))
        return out

    def _review_patches(self, c, ledger) -> list[dict]:
        """The review's patches no strategy has tested yet (the ledger shows no outcome for them), rebased onto this
        incumbent; strategies keep only measured hypotheses across an incumbent change, and the hypothesis calls are told
        not to repeat the review's patches (QA13 F1)."""
        from .strategies import merge_patches
        done = {h.get('mechanism') for h in ledger or [] if isinstance(h, dict) and h.get('status') not in UNTESTED_STATES}
        self.review_patches = [h for h in self.review_patches if h['mechanism'] not in done]
        files = self.hooks.candidate_files(c.overlay) if hasattr(self.hooks, 'candidate_files') else {}
        out = []
        for h in self.review_patches:
            cur = {f: files.get(f, '') for f in h['patch']}
            patch = merge_patches(self.review_files, [cur, h['patch']])
            if patch is not None and patch != cur:
                out.append(dict(h, patch=patch))
        return out

    def hypotheses(self, engine, base: str, ledger: list | None = None) -> list[dict]:
        """The next hypotheses from the model, their code patches through the patch check when the debate has one."""
        from .patch_check import check_patches
        return check_patches(self, engine, base, self._hypotheses(engine, base, ledger))

    def _hypotheses(self, engine, base: str, ledger: list | None = None) -> list[dict]:
        s = engine.s
        if self.driver is None or s.ledger.used['llm_tokens'] >= s.ledger.budget.llm_tokens:   # no LLM or no LLM tokens:
            return []                                    # the strategy explores on its own
        governor = TokenGovernor(s.ledger, self.max_hypotheses)
        c = s.candidates[base]
        digest = self.hooks.receipt_digest(c.receipt, c.verdict, max_tokens=1500) if c.receipt else None
        cap = self._answer_cap(s)
        code_tokens = min(self.code_tokens, governor.code_tokens)
        code = None if self.menu or code_tokens <= 0 else code_view(self.hooks, c.overlay, code_tokens,
                                                                   cap // FULL_TEXT_SHARE)
        if code and self.review is None:
            self.review_files = self.hooks.candidate_files(c.overlay) if hasattr(self.hooks, 'candidate_files') else {}
            self.review_patches = self._review(engine, base, code, digest)
        reviewed = self._review_patches(c, ledger) if code else []
        governor = TokenGovernor(s.ledger, self.max_hypotheses)  # the review may have moved the next call into a tighter zone
        if s.ledger.exhausted() or governor.exhausted:
            return reviewed
        if code and governor.code_tokens < code_tokens:
            code = code_view(self.hooks, c.overlay, governor.code_tokens, self._answer_cap(s) // FULL_TEXT_SHARE)
        shown = (fold_ledger(list(ledger or []), chars=governor.history_limit(LEDGER_CHARS)) if self.bounded_ledger
                 else (ledger or [])[-governor.history_limit(40):])
        payload = {'task': self.hooks.task_view(), 'incumbent': {'id': base, 'config': c.config,
                   'patched_files': sorted(c.overlay or {}), 'receipt': digest},
                   'ledger': shown, 'budget_left': s.budget_view(), 'max_hypotheses': governor.max_hypotheses}
        if self.prior:
            prior = self.prior[:governor.history_limit(30)]
            payload['earlier_runs'] = [unpaired(r) for r in prior] if self.search_calibrates else prior
        if self.experience:
            payload['experience'] = self.experience
        acc = self.accounting(c.config, (c.verdict or {}).get('step_time')) if self.accounting else None
        if acc:
            payload['accounting'] = acc
        if self.review and self.review['verdict'] == 'no_patch':
            code = None                                # the review found nothing only code can remove: knobs only
        if code:
            payload['code'] = code                     # the model can only patch what it can read
            if self.review:
                payload['code_review'] = self.review
        menu = None
        if self.menu:
            from .knob_space import choices
            menu = [{'change': {k: v}} for k, vals in choices(s.task).items() for v in vals if v != c.config.get(k)]
            payload['menu'] = menu
        if self.context is not None:                   # the strategy's fields (hls3: failures, the memory view, ...)
            payload.update(self.context(engine, base) or {})
        if governor.history_scale < 1 and isinstance(payload.get('measured_history'), list):
            payload['measured_history'] = fold_history(payload['measured_history'],
                                                       chars=governor.history_limit(HISTORY_CHARS),
                                                       max_rows=governor.history_limit(40))
        elif governor.history_scale < 1 and isinstance(payload.get('measured_history'), dict):
            measured = payload['measured_history']
            measured['rows'] = fold_history(measured.get('rows') or [], chars=governor.history_limit(HISTORY_CHARS),
                                             max_rows=governor.history_limit(40))
        lib = self._library(engine, base, payload) if code else None   # D7: the library's text only next to the code
        if lib:
            payload['library_source'] = lib
        else:
            payload.pop('library_source', None)
        if self.flags.get('parse') and self.dropped:   # F14: the hypotheses of its answers not used, and why
            payload['dropped_hypotheses'] = list(self.dropped[-6:])
        prompt = json.dumps(payload, ensure_ascii=False, default=str)
        if estimate_tokens(prompt) > 60000:
            payload['ledger'] = (fold_ledger(list(ledger or []), chars=LEDGER_CHARS // 4) if self.bounded_ledger
                                 else payload['ledger'][-10:])
            prompt = json.dumps(payload, ensure_ascii=False, default=str)
        self.calls += 1
        system = (HYPOTHESIS_SYSTEM.replace(SELF_CALIBRATION, SEARCH_CALIBRATES) if self.search_calibrates else HYPOTHESIS_SYSTEM)
        system += (CODE_RULE if code else '') + (PHASES_RULE if (digest or {}).get('step_phases') else '') + \
            (EXPERIENCE_RULE if self.experience else '') + (ACCOUNTING_RULE if acc else '') + (MENU_RULE if menu else '')
        if self.ledger_states:
            system += LEDGER_STATES_RULE
        system += self._hls3_rules(bool(code)) + (LIBRARY_RULE if payload.get('library_source') else '')
        if self.patch_only:                            # wcs: patches only (no code to patch: no call)
            if not code:
                return reviewed
            system += PATCH_ONLY_RULE
            payload['max_hypotheses'] = min(governor.max_hypotheses, 3)
            if self.patch_hypotheses:                  # v4: the strategy's patches per ask, when the spec set them
                payload['max_hypotheses'] = min(governor.max_hypotheses, int(self.patch_hypotheses))
            prompt = json.dumps(payload, ensure_ascii=False, default=str)
        schema = hls3_schema(HYPOTHESIS_SCHEMA, self.flags)
        parse = bool(self.flags.get('parse'))          # F14: one hypothesis at a time, split replies joined
        ask = {'schema': relaxed_schema(schema), 'join_text': True, 'retry_note': explain_reply} if parse else \
            {'schema': schema}
        out_cap = self.code_max_tokens if code else self.max_tokens
        kw = _allowances(s, self.driver, system + prompt, out_cap)
        if kw is None:                                 # the call does not fit what the budget has left
            return reviewed
        files = self.hooks.candidate_files(c.overlay) if hasattr(self.hooks, 'candidate_files') else {}
        while code:                                    # as in the review: until the share of the last cap holds
            tight = _tighten(code, files, kw['max_output_tokens'])
            if tight is code:
                break
            code = payload['code'] = tight
            prompt = json.dumps(payload, ensure_ascii=False, default=str)
            kw = _allowances(s, self.driver, system + prompt, out_cap)
            if kw is None:
                return reviewed
        if not self._spend():
            return reviewed
        res = self.driver.complete(system, prompt, role='hypotheses', **ask, **kw)
        s.charge_llm(res, 'hypotheses')
        if not res.action and code and cut_at_limit(res):
            # cut at the output limit with code in the prompt (D2-R204d: five whole-file patches of train_wan.py in one
            # reply): one smaller ask - every file by edits, half the hypotheses - with the cut named; a second cut ends it
            wrote = written_in_full(res.text, files)
            n = max(1, governor.max_hypotheses // 2)
            payload['code'] = {**code, 'edits_only': sorted(code.get('mutable') or {})}
            payload['max_hypotheses'] = n
            payload['cut_reply'] = ('your previous reply was cut at the output limit before its JSON ended'
                                    + (f': it wrote {", ".join(wrote)} out in full' if wrote else '')
                                    + f'; reply with at most {n} hypotheses and every code change as `edits` (no file\'s full text)')
            (getattr(engine, 'log', None) or (lambda m: None))(
                '[search] hypothesis call: the reply was cut at the output limit'
                + (f' (it wrote {", ".join(wrote)} out in full as a patch)' if wrote else '')
                + f'; asking once more for at most {n} hypotheses with edits only')
            prompt = json.dumps(payload, ensure_ascii=False, default=str)
            kw = _allowances(s, self.driver, system + prompt, out_cap)
            if kw is not None and self._spend():
                res = self.driver.complete(system, prompt, role='hypotheses', **ask, **kw)
                s.charge_llm(res, 'hypotheses')
        _say_error(engine, res, 'hypothesis call')
        if not res.action:
            self.failed += 1
            return reviewed
        items = res.action.get('hypotheses') or []
        if parse:
            items = self._accept(engine, res, items, schema, files, system, payload, out_cap, ask)
        out = list(reviewed)
        allowed = [m['change'] for m in menu] if menu is not None else None
        for h in items[:governor.max_hypotheses]:
            if not isinstance(h, dict) or not isinstance(h.get('change'), dict):
                continue
            if allowed is not None and h['change'] not in allowed:
                continue
            g = float(h.get('predicted_gain_pct') or 0) / 100
            u = float(h.get('uncertainty_pct') or max(1.0, abs(g) * 50)) / 100
            patch = None
            if allowed is None and code:
                patch, ok = patch_of(h, files)
                if not ok:
                    self.unusable_edits += 1
                    continue
            if not h['change'] and not patch:
                continue                                 # nothing to test (a patch that changes nothing)
            if self.patch_only and (h['change'] or not patch):
                continue                                 # wcs: a knob hypothesis is not what was asked
            out.append({'mechanism': str(h.get('mechanism') or mechanism_id(h['change']))[:200], 'change': h['change'],
                        'patch': None if allowed is not None else patch, 'mu': math.log1p(max(-0.9, g)),
                        'sigma': max(0.005, u * self.sigma_scale),
                        'p_feasible': min(1.0, max(0.0, float(h.get('p_feasible', 0.8)))), 'rationale': str(h.get('rationale', ''))[:400],
                        'change_class': change_class(h.get('change_class')), **hls3_fields(h, self.flags), **author_field(h)})
        return out

    def _accept(self, engine, res, items, schema, files, system, payload, out_cap, ask) -> list:
        """F14b / F14c: the answer's hypotheses one by one (accept_hypotheses); when not one of them is usable, the
        model is asked once more with each one's exact errors (charged as any call). The dropped ones are kept for the
        strategy to report back (`dropped`), the rewritten ones too (`rewritten`)."""
        s = engine.s
        if (res.meta or {}).get('parse_mode') == 'joined':
            self.joined += 1
        kept, dropped, fixed = accept_hypotheses(items, schema, files)
        if not kept and dropped:
            errors = [f'hypothesis {d["index"]} ({d["mechanism"]!r}): ' + '; '.join(d['problems']) for d in dropped[:8]]
            again = json.dumps({**payload, 'previous_reply_errors': errors + [EDITS_HINT]}, ensure_ascii=False,
                               default=str)
            kw = _allowances(s, self.driver, system + again, out_cap)
            if kw is not None and self._spend():
                res = self.driver.complete(system, again, role='hypotheses', **ask, **kw)
                s.charge_llm(res, 'hypotheses')
                _say_error(engine, res, 'hypothesis call (asked again with its errors)')
                if res.action:
                    kept, d2, f2 = accept_hypotheses(res.action.get('hypotheses') or [], schema, files)
                    dropped, fixed = dropped + d2, fixed + f2
        self.dropped += [{'call': self.calls, **d} for d in dropped]
        self.rewritten += [{'call': self.calls, **f} for f in fixed]
        return kept

    def factors(self, engine, base: str) -> list[dict]:
        """Two-level DOE factors: boolean knobs, and knobs with exactly two choices one of which the base uses (the
        plus level is the other value) - the same two-level knobs that choose the default strategy."""
        cfg = engine.s.candidates[base].config
        props = (engine.s.task.knob_schema or {}).get('properties') or {}
        out = []
        project = getattr(getattr(self, 'hooks', None), 'project', None)
        for k, v in props.items():
            if v.get('type') == 'boolean' and not (project and project.knobs[k].get('narrowed')):
                out.append({'name': k, 'plus': {k: not cfg.get(k)}})
            elif len(v.get('enum') or []) == 2 and cfg.get(k) in v['enum']:
                out.append({'name': k, 'plus': {k: next(x for x in v['enum'] if x != cfg.get(k))}})
        return out



# ---------------------------------------------------------------- opro / ltune (possible_search_algorithm/10 section 04)
# The configuration call: K whole configurations from the task, the knob table, the legal-table rules (the task view's
# workload invariants and constraints) and the history of every measured configuration, best first (opro's rounds,
# OPRO); or K diverse whole configurations to be raced (ltune, lambda-Tune). Attached to LLMProposer and SimProposer
# below (an addition: no existing method changes).
CONFIG_SYSTEM = """You propose complete configurations of one training job's knobs for InfraRSI, which makes the job
faster without changing what it computes. Given: the job (task view: its knobs and their values, the workload invariants
and the constraints every configuration must satisfy, the hardware), the reference configuration and its receipt, the
knob table, and `history`: every configuration measured so far, best first - its knob values that differ from the
reference, its outcome (the measured speed over the reference; or how it failed: out of memory with the GiB it was short,
thrash at the memory edge, a slow start, aborted as slower than the best so far, a crash) and its measured memory peak
when one was read - followed by configurations ruled out before running, with why. Propose `k` complete configurations
that are not in the history, each as the knob values that differ from the reference (a value must be one of the knob's
values in the table), with `gain_pct` (its expected speed gain over the reference, in percent), `sd_pct` (your
one-sigma uncertainty), `peak_gib` (the GiB per GPU you expect it to peak at; null when you cannot say) and `why` (one
line). Keep what the history shows to work, fix what made the failures fail (add the value that frees the memory a
spender needs, rather than dropping the spender), and change several knobs together when they only pay together."""
CONFIG_ROUND_RULE = """
Order them by your expected gain, the best first; at least one should differ from the best measured configuration in
two or more knobs. Reply with exactly one JSON object: {"configs":[{"change":{"knob":"value"},"gain_pct","sd_pct",
"peak_gib","why"}],"notes":""}."""
CONFIG_RACE_RULE = """
They will be raced (short runs first, the survivors extended): make them diverse - each a different way to make the job
faster (a different bottleneck or mechanism), each complete with the values that make its spenders fit - the most
promising first. Reply with exactly one JSON object: {"configs":[{"change":{"knob":"value"},"gain_pct","sd_pct",
"peak_gib","why"}],"notes":""}."""
CONFIG_SCHEMA = {'type': 'object', 'required': ['configs'], 'properties': {
    'configs': {'type': 'array', 'items': {'type': 'object', 'properties': {
        'change': {'type': 'object'}, 'gain_pct': {'type': ['number', 'null']}, 'sd_pct': {'type': ['number', 'null']},
        'peak_gib': {'type': ['number', 'null']}, 'why': {'type': ['string', 'null']}}}},
    'notes': {'type': ['string', 'null']}}}


def _llm_config_batch(self, engine, base: str, *, k: int = 5, history=None, purpose: str = 'round'):
    """opro / ltune: one call for `k` whole configurations (CONFIG_SYSTEM; `purpose` 'race': diverse, to be raced);
    the answer as the model gave it, None without a model, without tokens, past the allowance or without a usable
    answer. Charged to the session as role 'configs'."""
    s = engine.s
    if self.driver is None or s.ledger.used['llm_tokens'] >= s.ledger.budget.llm_tokens:
        return None
    from .knob_space import choices
    c = s.candidates[base]
    digest = self.hooks.receipt_digest(c.receipt, c.verdict, max_tokens=1500) if c.receipt else None
    table = {kk: {'reference': c.config.get(kk), 'values': list(v)} for kk, v in choices(s.task).items()}
    payload = {'task': self.hooks.task_view(), 'reference': {'id': base, 'config': c.config, 'receipt': digest},
               'knob_table': table, 'k': int(k), 'history': list(history or []), 'budget_left': s.budget_view()}
    if self.experience:
        payload['experience'] = self.experience
    facts = getattr(self.hooks, 'memory_facts', None)
    if callable(facts):
        payload['memory_facts'] = facts()
    system = CONFIG_SYSTEM + (CONFIG_RACE_RULE if purpose == 'race' else CONFIG_ROUND_RULE)
    prompt = json.dumps(payload, ensure_ascii=False, default=str)
    kw = _allowances(s, self.driver, system + prompt, self.max_tokens)
    if kw is None or not self._spend():
        return None
    self.calls += 1
    res = self.driver.complete(system, prompt, schema=CONFIG_SCHEMA, role='configs', **kw)
    s.charge_llm(res, 'configs')
    _say_error(engine, res, 'configuration call')
    if not isinstance(res.action, dict):
        self.failed += 1
        return None
    self.last_configs = res.action
    return res.action


def _sim_config_batch(self, engine, base: str, *, k: int = 5, history=None, purpose: str = 'round'):
    """The configuration call, simulated (deterministic per task and call): the values the model knows (q) rated by
    their predicted single-move gains (as knob_prior rates them), the values it declares to free memory (`_memory`, with
    `memory`); a value that two or more failed configurations of the history held is left out, and nothing in the
    history is proposed again (the history read as feedback). A round (opro) builds on the best measured configuration
    of the history (the reference before any): it with each positive value it lacks, the best ones first, it with every
    freeing value, then the prior's three (every positive value, the best three, the best); a race (ltune) proposes
    diverse ones: every positive value with the freeing values, every positive value, the best three, the best, the
    second best alone, every positive value it does not declare to spend memory. Seeded random subsets of the positive
    values fill the batch. Its stated gain: the sum of its values' ratings."""
    import random
    self.calls += 1
    sim = self.sim
    cfg, patches = self._base(engine, base)
    t0, _ = sim.truth(cfg, patches)
    knobs = sim.knobs()
    rated, frees, spends = {}, [], set()
    for kk, vals in knobs.items():
        for v in vals:
            if v == cfg.get(kk):
                continue
            mech = mechanism_id({kk: v})
            if self.knows(mech) and t0:
                t1, _ = sim.truth({**cfg, kk: v}, patches)
                g = math.log(t0 / t1) if t1 else None
                rated[(kk, v)] = self.predict('prior', mech, g)[0]
            d = self._memory(cfg, {kk: v}, mech).get('mem_dir') if self.memory else None
            if d == 'frees':
                frees.append((kk, v))
            elif d == 'spends':
                spends.add((kk, v))

    def key(ch):
        return json.dumps(sorted(ch.items()), default=str)
    seen, bad, best_ch = set(), {}, None
    for row in history or []:
        ch = row.get('change') if isinstance(row, dict) else None
        if not isinstance(ch, dict):
            continue
        seen.add(key(ch))
        out = str(row.get('outcome') or '')
        if best_ch is None and out.startswith('x') and not row.get('code_patches'):
            best_ch = dict(ch)                   # (the rows are best first)
        if out.startswith(('out of memory', 'thrash', 'failed', 'a slow start')):
            for kv in ch.items():
                bad[json.dumps(kv, default=str)] = bad.get(json.dumps(kv, default=str), 0) + 1

    def ok(kk, v):
        return bad.get(json.dumps((kk, v), default=str), 0) < 2
    best = {}
    for (kk, v), mu in sorted(rated.items(), key=lambda t: -t[1]):
        if mu > 0 and ok(kk, v):
            best.setdefault(kk, (mu, v))
    ranked = sorted(best.items(), key=lambda t: -t[1][0])
    fix = {kk: v for kk, v in frees if ok(kk, v)}
    top = [dict((kk, v) for kk, (mu, v) in ranked[:n]) for n in (len(ranked), 3, 1)]
    safe = {kk: v for kk, (mu, v) in ranked if (kk, v) not in spends}
    if purpose == 'race':
        pool = [{**top[0], **fix}, safe] + top + ([{ranked[1][0]: ranked[1][1][1]}] if len(ranked) > 1 else [])
    else:
        b = best_ch or {}
        pool = [{**b, kk: v} for kk, (mu, v) in ranked if b.get(kk) != v] + [{**b, **fix}, {**b, **safe}] + top
    rng = random.Random(f'{sim.task.seed}|{self.seed}|configs|{self.calls}')
    for _ in range(4 * int(k)):
        if not ranked:
            break
        sub = [t for t in ranked if rng.random() < 0.5] or [rng.choice(ranked)]
        pool.append(dict((kk, v) for kk, (mu, v) in sub))
    out = []
    for ch in pool:
        ch = {kk: v for kk, v in ch.items() if v != cfg.get(kk)}
        if ch and key(ch) not in seen:
            seen.add(key(ch))
            out.append({'change': ch, 'gain_pct': round(100 * math.expm1(sum(rated.get((kk, v), 0.0) for kk, v in ch.items())), 2),
                        'sd_pct': 5.0, 'peak_gib': None, 'why': 'simulated'})
        if len(out) >= int(k):
            break
    self.last_configs = {'configs': out}
    return self.last_configs


LLMProposer.config_batch = _llm_config_batch
SimProposer.config_batch = _sim_config_batch
