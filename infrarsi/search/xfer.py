"""xfer: transfer-first over the experience store (FABLE_PICK_C6_20261031 section 2: Stoll et al.'s best-first transfer,
Feurer et al.'s RGPE ranking-loss weights), knob side and code side, on the candidates' base (search/llmopt).

Sources - read once, at planning, from the arm's INFRARSI_HOME and nothing else (D4: a private copy holding the other
environments' records): the run packs the experience plan loaded, one source per numeric environment (the pack's `env`
numbers: card GiB, nodes, GPUs per node, parameters, framework; never a name). A source whose numbers are this job's own
is left out (it cannot be told apart from this job). Per source:

    winner   its packs' best valid knob-only configuration (the largest gain over its own pack's reference; runs that carried
             patches are left out: their speed is not their knobs'), mapped onto this table by knob name through the
             experience registry's mechanism states - a value the table lacks: the reference's; a value whose arithmetic
             lower bound alone (the memory ledger's at z = 2, on the reference) exceeds the card: the reference's -,
             completed to a legal row
    effects  its single-move effects (two of its observations one mechanism apart, readable above twice its noise): their
             sum over a move's changed knobs is its prediction of that move here
    weight   the similarity of its numbers to this job's - same card GiB x2, same nodes x1.5, |log parameter ratio| <= 0.7
             x1.5; another framework: no source -, then after each full screen x max(F_MIN, 1 + agreements -
             disagreements): the signs of its prediction of each pair (the new run, an earlier measured run with the same
             patches) against the measured difference beyond 2 sigma_pair (RGPE's ranking loss, smallest form); never
             below W_MIN

The transfer brief - HOME/experience/transfer_brief.json, written offline from the other environments' finished arms
(reports/.../scripts/build_transfer_brief.py), read once, at planning: their proven patches (mechanism, canonical site,
measured gain) and inert ones (why), each with its environment's numbers. An entry weighs its environment's similarity x
the RGPE factor of the source with those numbers (1 without one). No brief, or one that does not parse: ordinary asks
only. No usable source: the knob side starts from the prior's configuration. The notes say which.

Screens (B the screen budget; one proposal per step):
    1       the top source's winner, probed first below 0.7 feasibility; a failure (its probe, or a full run that did not
            become the incumbent): the next source's; at most SOURCE_SCREENS source full runs. No usable source: the
            prior's best configuration (wcs's first screen)
    2       the prior's best configuration, once, when it differs from the incumbent in two or more big knobs and the time
            model rates it above the incumbent; else code
    3..B-2  code: wcs's track (patches only, the unchanged self-proof at the probe, a proven patch a pseudo-knob on every
            later configuration); the first ask is the port ask - the brief's proven patches, best weighted gain first, one
            per site, and its inert ones as `do_not_repeat`, an additive field of the patch call, sent with the code even
            when the code review said no_patch (for that request only; done only once it really went out) - the model
            ports at most 3; then ordinary asks, one whenever no patch is pending and a full screen ran since the last, CODE_ASKS in all,
            3 patches each, within search/llmopt's 10 calls / 8 USD; the port's patches are queued by their entry's weighted
            gain, stacked by the chain as they prove themselves
    B-1     the tail bundle on the incumbent
    B       the best unscreened source winner the time model puts within 2 sigma of the incumbent, else the draw
    a step with nothing of its own to run: the next source winner within 2 sigma, the tail, the draw

No short rung: a probe-first plan runs its full rung on its own candidate in the same evaluation (nothing resubmitted).

Out-of-memory families (the coordinator's rule after two GPU arms of other strategies in which 7-8 of 10 full screens ran
out of memory in the family of earlier failures that strict dominance never caught): after a measured memory failure (out of
memory or thrash at the card, at any rung) the values the memory ledger blames for it mark a family - of the failed
configuration's values off the reference, the fewest that carry half of the GiB the ledger's fit puts on them (each at
least 2 % of the card); when the ledger blames none (or is off), of the values the prior does not declare freeing or
neutral, the largest core it shares with another failed run, else all of them. A run that fitted holding a family whole
never clears it: it waives the family only for configurations that also hold its compensating values (its other values
off the reference not known to spend memory), until a failed run holds the family with them too. Any later configuration
holding all the values of a family (a source winner mapped from another environment, the prior's, the tail's, a draw)
has its feasibility capped at OOM_P = 0.5, below the probe threshold, so the ladder probes it first; with no probe left in
the allowance (or probes off) it is not run at all - never a full run without a probe -, the draw drawing again outside
the families and the tail ending.

    xfer_knob    the source winners (screen 1, B, the fill); off: the prior's configuration first
    xfer_rgpe    the weight update after each full screen; off: the similarity weights stay
    xfer_port    the port ask; off: ordinary asks only
    xfer_prior   screen 2's prior configuration; off: code
    xfer_code    wcs's code track; off: no asks
    xfer_tail    the tail bundle at B-1; off: code
    xfer_ledger  the memory ledger as the feasibility where the runs do not decide; off: wcs's
    xfer_probe   probe first below 0.7; off: full runs only (a code proof keeps its probe)

Nothing here names a knob, a model or a library: every value comes from the table, the registry, the packs, the brief,
the model's answers and the runs.
"""
from __future__ import annotations

import json
import math
import re

from . import XFER_COMPONENTS
from .ladder import LOWFI_SHARE
from .ledger import vk as lvk                    # (wcs's key form, memoised: a draw scores the whole table)
from .llmopt import CALLS_MAX, USD_MAX, LLMOptBase
from .wcs import Surrogate, vk
from .wcs import proposer_flags as wcs_flags

SOURCE_SCREENS = 2              # source winners run in full on the first screens, at most
CODE_ASKS = 6                   # the code track's asks on this instance (wcs keeps its three)
SIM_CARD, SIM_NODES, SIM_PARAMS = 2.0, 1.5, 1.5   # the similarity's factors: same card GiB, same nodes, near size
PARAMS_LOG = 0.7                # near size: |log parameter ratio| at most this
CARD_TOL = 0.5                  # two cards are the same within this many GiB
F_MIN, W_MIN = 0.25, 0.01       # the RGPE update's factor and a weight, at least
WITHIN_Z = 2.0                  # a source winner is worth a late screen within this many sigma of the incumbent
PORT_MAX, INERT_MAX = 12, 10    # the port ask shows at most this many proven entries (one per site) and inert ones
NOISE_DEFAULT = 0.02            # a pack without its noise: an effect is readable from twice this (the store's floor)
OOM_P = 0.5                     # an out-of-memory family member's feasibility at most this: below the probe threshold
BLAME_MIN, BLAME_SHARE = 0.02, 0.5   # a value the ledger blames carries at least this share of the card; the fewest that
#                                      carry this share of the GiB it puts on a failed run's suspects are blamed
MEM_FAIL = ('oom', 'thrash')    # the measured memory failures (at any rung) an out-of-memory family is read from
FITTED = ('ok', 'slower', 'numerics')   # the run kinds that fitted in memory (wcs's _record_memory: status ok)
BRIEF_SCHEMA = 'infrarsi-transfer-brief/1'
BRIEF_FILE = 'transfer_brief.json'
PORT_ID = r'\[port\s+(p\d+)\]'  # a port answer's `mechanism` names the entry it ports
_NONE = object()
PORT_RULE = (
    'PORT ASK: `proven` lists code patches that measured faster on other training jobs, the best weighted gain first, one '
    'per site: what each did (`mechanism`), the function it patched (`site`, with the source\'s own words in `where`), its '
    'measured gain there (`gain_pct`), how much it counts here (`weight`: that job\'s card, nodes and size against this '
    'one\'s, and how well that job\'s measured effects ranked this job\'s runs), what its proof line printed there '
    '(`proof`), that job\'s numbers (`source`) and in how many runs one like it was proven (`runs`). Port at most 3 of '
    'them to this job\'s mutable files, in this order unless a site does not exist here or this job already runs it fast: '
    'the same mechanism at this job\'s own call site, as a patch with its proof lines. Begin each `mechanism` with '
    '`[port <id>]` of the entry it ports. `do_not_repeat` lists variants that did not take effect, gave no measurable '
    'gain, were slower or failed on other jobs, with why: do not propose them again as they were.')
PORT_VERDICT = 'no_patch (the code is shown again for the port ask only)'   # the code review's no_patch, for that request


# ---------------------------------------------------------------- environments
def _num(x) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0 else None


def env_numbers(env) -> dict | None:
    """An environment block's numbers - card GiB, nodes, GPUs per node, parameters (billions) - and framework, from a
    pack's `env` line, an experience fingerprint or a brief entry's `source`; None for anything else."""
    if not isinstance(env, dict):
        return None
    model = env.get('model') if isinstance(env.get('model'), dict) else {}
    card = _num(env.get('card_gib', env.get('memory_gib')))
    params = _num(env.get('params_b', model.get('params_b')))
    nodes, gpn = _num(env.get('nodes')), _num(env.get('gpus_per_node'))
    fw = env.get('framework')
    return {'card_gib': card, 'nodes': int(nodes) if nodes else None, 'gpus_per_node': int(gpn) if gpn else None,
            'params_b': params, 'framework': str(fw) if fw else 'other'}


def same_env(a, b) -> bool:
    """Two environments with the same numbers (card within CARD_TOL GiB, parameters within 1 %, nodes, GPUs per node
    where both give them, framework)."""
    if not a or not b or a['framework'] != b['framework'] or a['nodes'] != b['nodes']:
        return False
    if a['gpus_per_node'] and b['gpus_per_node'] and a['gpus_per_node'] != b['gpus_per_node']:
        return False
    for k, close in (('card_gib', lambda x, y: abs(x - y) <= CARD_TOL), ('params_b', lambda x, y: abs(math.log(x / y)) <= 0.01)):
        if (a[k] is None) != (b[k] is None) or a[k] is not None and not close(a[k], b[k]):
            return False
    return True


def similarity(env, target) -> float:
    """The source's weight before any run: 0 for another framework, else 1 x SIM_CARD (same card GiB) x SIM_NODES (same
    nodes) x SIM_PARAMS (|log parameter ratio| <= PARAMS_LOG)."""
    if not env or not target or env['framework'] != target['framework']:
        return 0.0
    w = 1.0
    if env['card_gib'] and target['card_gib'] and abs(env['card_gib'] - target['card_gib']) <= CARD_TOL:
        w *= SIM_CARD
    if env['nodes'] and target['nodes'] and env['nodes'] == target['nodes']:
        w *= SIM_NODES
    if env['params_b'] and target['params_b'] and abs(math.log(env['params_b'] / target['params_b'])) <= PARAMS_LOG:
        w *= SIM_PARAMS
    return w


def _env_key(env: dict) -> tuple:
    return (round(env['card_gib'] or 0.0, 1), env['nodes'] or 0, env['gpus_per_node'] or 0,
            round(env['params_b'] or 0.0, 2), env['framework'])


def _shown_env(env: dict) -> dict:
    return {k: env[k] for k in ('card_gib', 'nodes', 'params_b') if env.get(k) is not None}


def sources_from_packs(packs, target) -> tuple[list[dict], list[str]]:
    """The sources of a list of packs (each a list of pack lines) for the target environment (the module docstring): one
    per numeric environment other than the target's, weighed by similarity; with notes on what was left out."""
    from ..experience.pack import derive_effects, gain
    groups: dict = {}
    notes: list[str] = []
    own = other_fw = 0
    for lines in packs or []:
        if not isinstance(lines, list) or not lines or not isinstance(lines[0], dict):
            continue
        pid = str(lines[0].get('pack_id') or '')
        try:
            effects = derive_effects(lines)
        except Exception:                                # noqa: BLE001 - a pack that cannot be paired gives no effects
            effects = []
        for env in (x for x in lines if isinstance(x, dict) and x.get('kind') == 'env'):
            num = env_numbers(env)
            if num is None:
                continue
            if same_env(num, target):
                own += 1                                 # (this job's own numbers: never a source)
                continue
            eid = env.get('id')
            g = groups.setdefault(_env_key(num), {'env': num, 'packs': set(), 'cands': [], 'effects': {}})
            g['packs'].add(pid)
            noise = _num(env.get('noise_pct'))
            floor = 2 * (noise / 100.0 if noise else NOISE_DEFAULT)
            for e in effects:
                ch = e.get('change') if isinstance(e.get('change'), dict) else {}
                if e.get('env') != eid or e.get('with') or len(ch) != 1:
                    continue
                ge = gain(e)
                (m, ab), = ch.items()
                if ge is None or abs(ge) < floor or not (isinstance(ab, list) and len(ab) == 2):
                    continue
                g['effects'].setdefault((str(m), str(ab[0]), str(ab[1])), []).append(ge)
                g['effects'].setdefault((str(m), str(ab[1]), str(ab[0])), []).append(-ge)
            obs = [x for x in lines if isinstance(x, dict) and x.get('kind') == 'observation' and x.get('env') == eid]
            ref = next((o for o in obs if o.get('id') == 'c0' and o.get('status') == 'ok' and _num(o.get('step_time_s'))), None)
            if ref is None:
                continue
            for o in obs:
                cfg = o.get('config') if isinstance(o.get('config'), dict) else {}
                t = _num(o.get('step_time_s'))
                if o is ref or o.get('status') != 'ok' or t is None or 'x-patch' in cfg:
                    continue
                g['cands'].append((math.log(ref['step_time_s'] / t), pid, str(o.get('id')),
                                   {str(k): str(v) for k, v in cfg.items()}))
    out = []
    for key in sorted(groups, key=str):
        g = groups[key]
        w0 = similarity(g['env'], target)
        if w0 <= 0:
            other_fw += 1
            continue
        best = sorted(g['cands'], key=lambda c: (-c[0], c[1], c[2]))
        best = best[0] if best and best[0][0] > 0 else None
        out.append({'env': g['env'], 'w0': w0, 'w': w0, 'packs': sorted(g['packs']),
                    'winner': None if best is None else {'gain': best[0], 'pack': best[1], 'obs': best[2], 'states': best[3]},
                    'effects': {k: sum(v) / len(v) for k, v in g['effects'].items()}})
    if own:
        notes.append(f'{own} pack environment(s) with this job\'s own numbers left out')
    if other_fw:
        notes.append(f'{other_fw} source environment(s) of another framework left out')
    return out, notes


def brief_entries(brief) -> tuple[list[dict], list[dict], str | None]:
    """(proven, inert, None) of a transfer brief, each entry checked on its own (a malformed one dropped); ([], [], why)
    when it is not a brief."""
    if not isinstance(brief, dict) or brief.get('schema') != BRIEF_SCHEMA:
        return [], [], 'not a transfer brief (its schema)'
    out: dict = {'proven': [], 'inert': []}
    for kind in out:
        rows = brief.get(kind)
        if not isinstance(rows, list):
            return [], [], f'its `{kind}` is not a list'
        for r in rows:
            if not isinstance(r, dict) or not isinstance(r.get('mechanism'), str) or not r['mechanism'].strip():
                continue
            env = env_numbers(r.get('source'))
            g = r.get('gain_pct')
            g = float(g) if isinstance(g, (int, float)) and not isinstance(g, bool) and math.isfinite(g) else None
            if env is None or kind == 'proven' and (g is None or g <= 0):
                continue
            site = r.get('site') if isinstance(r.get('site'), str) and r['site'].strip() else None
            proof = r.get('proof') if isinstance(r.get('proof'), dict) else None
            out[kind].append({'id': str(r.get('id') or f'{kind[0]}{len(out[kind]) + 1:03d}')[:12],
                              'mechanism': ' '.join(r['mechanism'].split())[:200], 'site': site[:80] if site else None,
                              'where': ' '.join(str(r.get('site_text') or '').split())[:160] or None, 'gain_pct': g,
                              'reason': ' '.join(str(r.get('reason') or '').split())[:200] or None,
                              'proof': {k: str(v)[:120] for k, v in proof.items() if k in ('found', 'set')} if proof else None,
                              'env': env})
    return out['proven'], out['inert'], None


# ---------------------------------------------------------------- the strategy
class XFer(LLMOptBase):
    name = 'xfer'

    @classmethod
    def proposer_features(cls, off=()) -> dict:
        """wcs's flags, the code track's by `xfer_code`."""
        return wcs_flags(['wcs_code'] if 'xfer_code' in set(off or ()) else [])

    def __init__(self, proposer, *, off=(), calls_max: int = CALLS_MAX, usd_max: float = USD_MAX, facts=None,
                 accounting=None, experience=None, kappa0: float = 0.5, seed: int = 42, lowfi_share: float = LOWFI_SHARE,
                 name: str | None = None, code_asks: int = CODE_ASKS, packs=None, brief=None, env=None):
        off = set(off or ())
        self.xon = {c: c not in off for c in XFER_COMPONENTS}
        super().__init__(proposer, calls_max=calls_max, usd_max=usd_max, off_wcs=[] if self.xon['xfer_code'] else ['wcs_code'],
                         ledger=self.xon['xfer_ledger'], short=False, lowfi_share=lowfi_share, facts=facts,
                         accounting=accounting, experience=experience, kappa0=kappa0, seed=seed)
        if name:
            self.name = name
        self.code_asks = int(code_asks)                  # (this instance's asks; wcs's own default stays three)
        if not self.xon['xfer_probe']:
            self.ladder.probe_below = 0.0                # (no knob configuration is probed first)
        self._packs_in, self._brief_in, self._env_in = packs, brief, env
        self.target: dict | None = None                  # this job's environment numbers
        self.labels: dict = {}                           # knob -> its registry label (mechanism, states)
        self.sources: list[dict] = []
        self.brief_proven: list[dict] = []               # the brief's entries (weighed at the port ask); not wcs's
        self.brief_inert: list[dict] = []                # `inert` (this run's own inert patches, shown to the model)
        self.brief_used = False
        self.notes: list[str] = []
        self.phase = 'source'                            # source -> prior -> code
        self.src_fulls = 0                               # source winners run in full in the source phase
        self.src_ok = False                              # one of them became the incumbent
        self.src_tried = False                           # a source winner was proposed (then screen 2's rule decides)
        self.by_src: dict[str, dict] = {}                # proposal id -> its source
        self.port_tried = False                          # the port ask was made (once: no retry)
        self.port_done = False                           # ... and its request, with the transfer field, really went out
        self.port: dict = {}                             # the port ask: what it showed, what came back
        self._port_payload: dict | None = None           # (the patch call's extra field, only during the port ask)
        self.ported: dict[str, str] = {}                 # patch id -> the brief entry it ports
        self.weights: list[dict] = []                    # each RGPE update
        self.steps: list[dict] = []
        self.oom_gate = {'probed': 0, 'dropped': 0}      # out-of-memory family members probed first / not run in full
        self._blame: list = []                           # [(the values blamed for a memory failure, its run)] (_blamed)
        self._blame_stamp = None
        self._family_out = False                         # (a redraw outside the families: their members weigh 0)
        self._refused = False                            # (the last knob proposal was refused: a family member unprobed)

    # ------------------------------------------------------------ planning (once)
    def _init(self, engine) -> None:
        super()._init(engine)
        self.target = self._target_env(engine)
        self.labels = self._labels(engine)
        packs, brief, why = self._packs_in, self._brief_in, None
        if packs is None or brief is None:
            if self.exp:                                 # (the arm's own home, only with the experience plan on)
                if packs is None:
                    packs = self._home_packs()
                if brief is None:
                    brief, why = self._home_brief()
            else:
                why = why or 'no experience plan: no home read'
        try:
            self.sources, notes = sources_from_packs(packs or [], self.target)
        except Exception as exc:                         # noqa: BLE001 - unreadable packs: no sources, not a dead arm
            self.sources, notes = [], [f'the packs could not be read: {str(exc)[:160]}']
        self.notes += notes
        self._map_sources(engine)
        if brief is None:
            self.notes.append(f'no transfer brief ({why or "none given"}): the code side runs ordinary asks')
        else:
            proven, inert, bad = brief_entries(brief)
            if bad:
                self.notes.append(f'the transfer brief is unusable ({bad}): the code side runs ordinary asks')
            else:
                own = [e for e in proven + inert if same_env(e['env'], self.target)]
                if own:
                    self.notes.append(f'{len(own)} brief entries with this job\'s own numbers left out')
                self.brief_proven = [e for e in proven if not same_env(e['env'], self.target)]
                self.brief_inert = [e for e in inert if not same_env(e['env'], self.target)]
                self.brief_used = bool(self.brief_proven)
                if not self.brief_proven:
                    self.notes.append('the transfer brief has no proven patch for this job: the code side runs ordinary asks')
        if not any(s.get('cfg') for s in self.sources):
            self.notes.append('no usable source configuration: the knob side starts from the prior\'s configuration')

    def _target_env(self, engine) -> dict:
        """This job's numbers: given, else the experience plan's fingerprint, else the memory facts and the manifest."""
        if self._env_in is not None:
            return env_numbers(self._env_in) or env_numbers({})
        fp = (self.exp or {}).get('fingerprint') if isinstance(self.exp, dict) else None
        fp = fp if isinstance(fp, dict) else {}
        raw = getattr(getattr(engine.s, 'task', None), 'raw', None)
        man = raw.get('manifest') if isinstance(raw, dict) and isinstance(raw.get('manifest'), dict) else {}
        card = fp.get('memory_gib') or self.facts.get('cap_gib')
        if not card and fp.get('gpu'):
            from ..experience.registry import gpus
            card = (gpus().get(fp.get('gpu')) or {}).get('memory_gib')
        return env_numbers({'card_gib': card, 'nodes': fp.get('nodes'), 'gpus_per_node': fp.get('gpus_per_node'),
                            'params_b': (fp.get('model') or {}).get('params_b') or self.facts.get('params_b'),
                            'framework': fp.get('framework') or man.get('framework')})

    def _labels(self, engine) -> dict:
        """knob -> the experience registry's label of its manifest definition (mechanism and the state of each value);
        a knob without one is its own mechanism, x-<name>, its value as text (as the packs write it)."""
        raw = getattr(getattr(engine.s, 'task', None), 'raw', None)
        man = raw.get('manifest') if isinstance(raw, dict) and isinstance(raw.get('manifest'), dict) else {}
        defs = man.get('knobs') if isinstance(man.get('knobs'), dict) else {}
        if not defs:
            return {}
        try:
            from ..experience.store import labels
            got = labels({k: d for k, d in defs.items() if k in self.knobs and isinstance(d, dict)}, man.get('framework'))
        except Exception:                                # noqa: BLE001 - no registry: every knob its own mechanism
            return {}
        return {k: lab for k, lab in got.items() if lab is not None}

    def _mech(self, k: str) -> str:
        lab = self.labels.get(k)
        if lab is not None:
            return lab.mechanism
        from ..experience.store import x_id
        return x_id(k)

    def _state(self, k: str, v) -> str | None:
        lab = self.labels.get(k)
        if lab is None:
            return str(v).lower() if isinstance(v, bool) else str(v)
        return lab.state(v)

    def _home_packs(self) -> list:
        """The run packs of HOME/experience that the experience plan loaded (all of them when it does not say)."""
        from ..experience.pack import read_lines, validate
        from ..experience.store import home
        root = home() / 'experience'
        keep = self.exp.get('packs') if isinstance(self.exp, dict) and isinstance(self.exp.get('packs'), list) else None
        keep = set(keep) if keep is not None else None
        out, seen = [], set()
        for p in sorted(root.glob('**/*.jsonl*')) if root.is_dir() else []:
            try:
                lines = read_lines(p)
                if not lines or validate(lines)[0]:
                    continue
            except Exception:                            # noqa: BLE001 - an unreadable pack is skipped
                continue
            pid = lines[0].get('pack_id')
            if pid in seen or keep is not None and pid not in keep:
                continue
            seen.add(pid)
            out.append(lines)
        return out

    def _home_brief(self) -> tuple[dict | None, str | None]:
        from ..experience.store import home
        p = home() / 'experience' / BRIEF_FILE
        if not p.is_file():
            return None, 'none in the home'
        try:
            return json.loads(p.read_text(encoding='utf-8')), None
        except Exception as exc:                         # noqa: BLE001 - a brief that does not parse: no brief
            return None, f'it does not parse: {str(exc)[:120]}'

    def _map_sources(self, engine) -> None:
        """Each source's winner on this table (the module docstring); the same configuration twice: the heavier source's."""
        seen = set()
        for i, s in enumerate(sorted(self.sources, key=lambda s: (-s['w0'], -((s['winner'] or {}).get('gain') or 0.0)))):
            s['index'] = i
            s['state'], s['cfg'], s['map_notes'] = 'unusable', None, []
            if not s['winner']:
                s['map_notes'] = ['no valid knob-only configuration faster than its reference']
                continue
            try:
                cfg, notes = self._map_winner(engine, s['winner']['states'])
            except Exception as exc:                     # noqa: BLE001 - a winner that cannot be mapped: no winner
                cfg, notes = None, [f'not mapped: {str(exc)[:120]}']
            s['map_notes'] = notes
            key = self._ckey(cfg, ()) if cfg is not None else None
            if cfg is None or key in seen:
                if cfg is not None:
                    s['map_notes'].append('the same configuration as a heavier source\'s')
                continue
            seen.add(key)
            s['cfg'], s['state'] = cfg, 'open'
        self.sources.sort(key=lambda s: s['index'])

    def _map_winner(self, engine, states: dict) -> tuple[dict | None, list[str]]:
        cfg, changed, notes = {k: self.ref.get(k) for k in self.knobs}, [], []
        for k, vals in self.knobs.items():
            st = states.get(self._mech(k))
            if st is None or self._state(k, self.ref.get(k)) == st:
                continue                                 # (not set by the source, or the reference's value)
            v = next((x for x in vals if self._state(k, x) == st), _NONE)
            if v is _NONE:
                notes.append(f'{k}: no value of the table for {st}: the reference\'s')
                continue
            cfg[k] = v
            changed.append(k)
        led = self.mledger
        for k in list(changed):                          # a value whose lower bound alone is above the card: dropped
            if led is not None and led.cap:
                lb = led.lower_bound({**{kk: self.ref.get(kk) for kk in self.knobs}, k: cfg[k]})
                if lb > led.cap:
                    notes.append(f'{k}={cfg[k]}: its lower bound alone ({lb:.0f} GiB) is above the card: the reference\'s')
                    cfg[k] = self.ref.get(k)
                    changed.remove(k)
        if not changed:
            return None, notes + ['nothing off the reference']
        legal = self._complete(engine, cfg, set(changed))
        if legal is None:
            return None, notes + ['no legal configuration']
        legal = {k: legal.get(k) for k in self.knobs}
        if self._ckey(legal, ()) == self._ckey(self.ref, ()):
            return None, notes + ['completed to the reference']
        return legal, notes

    # ------------------------------------------------------------ weights
    def _factor(self, env: dict) -> float:
        """The RGPE factor so far of the source with these numbers (1 without one)."""
        s = next((s for s in self.sources if same_env(s['env'], env)), None)
        return s['w'] / s['w0'] if s is not None and s['w0'] > 0 else 1.0

    def _entry_weight(self, e: dict) -> float:
        return similarity(e['env'], self.target) * self._factor(e['env'])

    def _predict(self, src: dict, a: dict, b: dict) -> float | None:
        """The source's additive prediction of the log gain of moving configuration a to b (None: it measured none of the
        changed knobs' moves)."""
        total, known = 0.0, False
        for k in self.knobs:
            va, vb = a.get(k), b.get(k)
            if vk(va) == vk(vb):
                continue
            sa, sb = self._state(k, va), self._state(k, vb)
            if sa is None or sb is None or sa == sb:
                continue
            g = src['effects'].get((self._mech(k), sa, sb))
            if g is not None:
                total, known = total + g, True
        return total if known else None

    def _rgpe(self, engine, cid: str) -> None:
        """After a full screen: each source's weight x max(F_MIN, 1 + agreements - disagreements) over the pairs (this run,
        an earlier measured run with the same patches) whose measured difference is beyond 2 sigma_pair and which the
        source predicts; at least W_MIN."""
        new = self.runs.get(cid) or {}
        if new.get('y') is None or new.get('bound') or not self.sources:
            return
        ids = sorted(k for k, v in (new.get('cfg') or {}).items() if isinstance(k, str) and k.startswith('patch:') and v)
        thr = 2 * engine.sigma_pair()
        pairs = []
        for oid, r in self.runs.items():
            if oid == cid or r.get('y') is None or r.get('bound') or (oid != 'c0' and (r.get('fidelity') != 'full'
                                                                                     or r.get('status') != 'screened')):
                continue
            rids = sorted(k for k, v in (r.get('cfg') or {}).items() if isinstance(k, str) and k.startswith('patch:') and v)
            d = new['y'] - r['y']
            if rids == ids and abs(d) > thr:
                pairs.append((r['cfg'], d, oid))
        rec = {'after': cid, 'screens_done': engine.full_screens(), 'pairs': len(pairs), 'sources': []}
        for s in self.sources:
            agree = disagree = 0
            for cfg, d, _ in pairs:
                p = self._predict(s, cfg, new['cfg'])
                if p is None or p == 0.0:
                    continue
                agree += (p > 0) == (d > 0)
                disagree += (p > 0) != (d > 0)
            f = max(F_MIN, 1.0 + agree - disagree)
            s['w'] = max(W_MIN, s['w'] * f)
            rec['sources'].append({'source': s['index'], 'agree': agree, 'disagree': disagree, 'factor': f,
                                   'weight': round(s['w'], 4)})
        self.weights.append(rec)

    # ------------------------------------------------------------ out-of-memory families: probed before any full run
    def _off_ref(self, cfg: dict) -> frozenset:
        return frozenset((k, lvk(cfg.get(k))) for k in self.knobs if lvk(cfg.get(k)) != lvk(self.ref.get(k)))

    def _blamed(self) -> list[tuple]:
        """[(the values blamed for a memory failure, its run, the contexts that waive them)]: per run that ran out of memory
        or thrashed at the card, at any rung (probe, short, full), of its suspects - its values off the reference - the
        fewest the memory ledger blames: by their fitted GiB (at least BLAME_MIN of the card each; the fit has every run's
        peak, so a value that fitted cheaply elsewhere carries little), the largest first, until they carry BLAME_SHARE of
        what it puts on the suspects; when the ledger blames none (or is off), of the suspects the prior does not declare
        to free memory or to be neutral, the largest core the run shares with another failed run (the same family failing
        twice), else all of them. A run that fitted holding a family whole never clears it (XFER_r1 1): it waives the
        family only for configurations that also hold its compensating values - its values off the reference outside the
        family that are not known to spend memory (the ledger fits them below BLAME_MIN of the card; without the ledger,
        the prior does not declare them spenders) -, a context void once a failed run holds the family together with it
        (the compensation did not hold there); a fitted run with no such value waives nothing. Rebuilt once per
        observation."""
        led = self.mledger
        stamp = (len(self.runs), len(led.obs) if led is not None else 0)
        if self._blame_stamp == stamp:
            return self._blame
        spend: dict = {}
        if led is not None and led.basis:
            try:
                spend = {(b['knob'], b['vk']): t for b, t in zip(led.basis, led.fit()) if b['kind'] == 'value'}
            except Exception:                            # noqa: BLE001 - a fit that fails: the core or every suspect
                spend = {}
        floor = BLAME_MIN * ((led.cap if led is not None else None) or self.mem.cap or 100.0)
        fitted = [self._off_ref(r.get('cfg') or {}) for r in self.runs.values() if r.get('kind') in FITTED]
        sus = {cid: self._off_ref(r.get('cfg') or {}) for cid, r in self.runs.items() if r.get('kind') in MEM_FAIL}
        levels = self.prior.levels if self.prior else {}
        innocent = {kv for kv, e in levels.items() if e.get('mem_dir') in ('frees', 'neutral')}

        def spends(kv) -> bool:                          # known to add memory: it cannot be what compensated
            return spend.get(kv, 0.0) >= floor if spend else (levels.get(kv) or {}).get('mem_dir') == 'spends'
        sets = {}
        for cid, s in sus.items():
            hot = sorted(((spend.get(kv, 0.0), kv) for kv in s if spend.get(kv, 0.0) >= floor), key=lambda t: (-t[0], t[1]))
            total, got, mass = sum(g for g, _ in hot), [], 0.0
            for g, kv in hot:
                if mass >= BLAME_SHARE * total:
                    break
                got.append(kv)
                mass += g
            if not got:                                  # the ledger blames none: the prior's spenders and unknowns,
                mine = s - innocent                      # the largest core shared with another failed run, else all
                cores = [mine & (o - innocent) for oc, o in sus.items() if oc != cid and mine & (o - innocent)]
                got = list(max(cores, key=lambda c: (len(c), sorted(c)))) if cores else list(mine or s)
            b = frozenset(got)
            if not b or b in sets:                       # (the same family from another run: the same waivers)
                continue
            waive = set()
            for f in fitted:                             # a fitted run holding the family: a context-specific exception
                if b <= f:
                    ctx = frozenset(kv for kv in f - b if not spends(kv))
                    if ctx and not any(b | ctx <= r for r in sus.values()):
                        waive.add(ctx)
            sets[b] = (b, cid, tuple(sorted(waive, key=lambda c: (len(c), sorted(c)))))
        out = sorted(sets.values(), key=lambda t: (len(t[0]), sorted(t[0]), t[1]))
        self._blame, self._blame_stamp = out, stamp
        return out

    def _family(self, cfg: dict) -> tuple | None:
        """(the run, the values) when the configuration holds every value blamed for a measured memory failure and no
        context that waives them (_blamed); None otherwise."""
        blamed = self._blamed()
        if blamed:
            held = {(k, lvk(cfg.get(k))) for k in self.knobs}
            for b, cid, waive in blamed:
                if b <= held and not any(ctx <= held for ctx in waive):
                    return cid, b
        return None

    @staticmethod
    def _words_kv(b) -> str:
        return ', '.join(f'{k}={v}' for k, v in sorted(b))

    def _pf(self, cfg: dict, ids=()) -> tuple[float, str]:
        """search/llmopt's feasibility, at most OOM_P for a member of an out-of-memory family (it holds a value blamed
        for a measured memory failure): the ladder probes it before any full run (0 while redrawing outside them)."""
        p, why = super()._pf(cfg, ids)
        if p <= 0.0:
            return p, why
        hit = self._family(cfg)
        if hit is None:
            return p, why
        note = f'it holds {self._words_kv(hit[1])} as {hit[0]} did, which ran out of memory: a probe before any full run'
        if self._family_out:
            return 0.0, note
        return min(p, OOM_P), (why + '; ' if why else '') + note

    def _knob_q(self, engine, inc, cfg: dict, *, kind: str, plan: tuple, p: float | None = None, why: str = '',
                origin: str | None = None):
        """The candidates' proposal of a whole configuration; None for an out-of-memory family member whose plan has no
        probe (no probe left in the allowance, or probes off): never a full run without one."""
        if self._family(cfg) is not None:
            if tuple(plan or ('full',))[0] != 'probe':
                self._refused = True
                self.oom_gate['dropped'] += 1
                return None
            self.oom_gate['probed'] += 1
        return super()._knob_q(engine, inc, cfg, kind=kind, plan=plan, p=p, why=why, origin=origin)

    def _tail_step(self, engine, inc, left):
        """search/llmopt's tail; a bundle refused as an unprobed out-of-memory family member ends the tail."""
        self._refused = False
        q = super()._tail_step(engine, inc, left)
        if q is None and self._refused:
            self.tail.done = True
        return q

    def _fallback(self, engine, inc, short: bool = False):
        """search/llmopt's draw; a draw refused as an unprobed out-of-memory family member is drawn again outside the
        families."""
        self._refused = False
        q = super()._fallback(engine, inc, short=short)
        if q is None and self._refused:
            self._family_out = True
            try:
                q = super()._fallback(engine, inc, short=short)
            finally:
                self._family_out = False
        return q

    # ------------------------------------------------------------ the knob side
    def _ranked_sources(self, *, within: tuple | None = None) -> list[dict]:
        """The open sources by weight (then their winner's gain); `within` (engine, inc): only those whose time model
        reaches the incumbent within WITHIN_Z sigma."""
        out = [s for s in self.sources if s.get('state') == 'open' and s.get('cfg')]
        out.sort(key=lambda s: (-s['w'], -s['winner']['gain'], s['index']))
        if within is not None:
            engine, inc = within
            keep = []
            for s in out:
                mu, sd = self._mu(engine, inc, s['cfg'], tuple(self.chain))
                if mu + WITHIN_Z * sd >= 0.0:
                    keep.append(s)
            out = keep
        return out

    def _source_q(self, engine, inc, s: dict, *, phase: str):
        """A proposal of the source's winner (its feasibility now; probe first below the threshold, within the
        allowance); None when it ran already or the runs rule it out."""
        ids = tuple(self.chain)
        if self._ckey(s['cfg'], ids) in self.tried or self._ckey(s['cfg'], ()) in self._ran_full():
            s['state'] = 'tried'
            return None
        p, why = self._pf(s['cfg'], ids)
        if p <= 0.0:
            s['state'] = 'ruled out'
            s['map_notes'].append(f'ruled out by the runs: {why}'[:200])
            return None
        plan = self.ladder.plan(engine, p, warmup=self._warmup(s['cfg']) or None, short=False)
        q = self._knob_q(engine, inc, s['cfg'], kind='source', plan=plan, p=p,
                         why=f'the winner of source {s["index"]} (weight {s["w"]:.2f})' + (f'; {why}' if why else ''))
        if q is None:                                    # an out-of-memory family member and no probe left: not in full
            s['state'] = 'oom family, no probe'
            return None
        s['state'], s['phase'], s['hid'] = 'proposed', phase, q.hypothesis
        self.by_src[q.hypothesis] = s
        self.src_tried = True
        return q

    def _source_step(self, engine, inc):
        if not self.xon['xfer_knob'] or self.src_ok or self.src_fulls >= SOURCE_SCREENS:
            return None
        for s in self._ranked_sources():
            q = self._source_q(engine, inc, s, phase='source')
            if q is not None:
                return q
        return None

    def _late_source(self, engine, inc):
        """The best unscreened source winner within WITHIN_Z sigma of the incumbent (the last screen, a fill)."""
        if not self.xon['xfer_knob']:
            return None
        for s in self._ranked_sources(within=(engine, inc)):
            q = self._source_q(engine, inc, s, phase='late')
            if q is not None:
                return q
        return None

    def _prior_step(self, engine, inc):
        """The prior's best configuration (by its stated gain, else the time model's; legal, not run, not ruled out): on
        the first screen when no source was tried; after the source screens only when it differs from the incumbent in
        two or more big knobs and the time model rates it above the incumbent."""
        first = not self.src_tried
        if not self.on['wcs_prior'] or not self.prior or not self.prior.configs or not first and not self.xon['xfer_prior']:
            return None
        mean = self.surr.mean()
        y_inc = Surrogate.value(self._incumbent_feats(engine, inc), mean)
        ids = tuple(self.chain)
        ranked = []
        for i, c in enumerate(self.prior.configs):
            cfg = {**{k: self.ref.get(k) for k in self.knobs}, **{k: v for k, v in c['change'].items() if k in self.knobs}}
            if not self._legal(engine, cfg):
                continue
            mu = Surrogate.value(self._feats(cfg, ids), mean) - y_inc
            ranked.append((c['mu'] if c.get('mu') is not None else mu, -i, mu, cfg, c.get('name') or f'config {i + 1}'))
        for _, _, mu, cfg, name in sorted(ranked, key=lambda t: (-t[0], -t[1])):
            if self._ckey(cfg, ids) in self.tried or self._ckey(cfg, ()) in self._ran_full():
                continue
            p, why = self._pf(cfg, ids)
            if p <= 0.0:
                continue
            if not first:
                base = self._knob_cfg(engine, inc)
                big = self._big_knobs()
                d = sum(1 for k in big if vk(cfg.get(k)) != vk(base.get(k)))
                if d < 2 or mu <= 0.0:
                    self.steps.append({'screens_done': engine.full_screens(), 'pick': 'prior skipped',
                                       'big_knobs_differ': d, 'predicted_gain_pct': round(100 * math.expm1(mu), 2)})
                    return None
            plan = self.ladder.plan(engine, p, warmup=self._warmup(cfg) or None, short=False)
            return self._knob_q(engine, inc, cfg, kind='prior', plan=plan, p=p,
                                why=f"the prior's '{name}' configuration" + (f'; {why}' if why else ''))
        return None

    # ------------------------------------------------------------ the code side: the port ask
    def context(self, engine, base) -> dict:
        """wcs's extra fields of the patch call, and during the port ask the transfer brief's view (`transfer`)."""
        out = super().context(engine, base)
        if self._port_payload:
            out = {**out, 'transfer': self._port_payload}
        return out

    def _site_key(self, e: dict) -> str:
        return (e.get('site') or '').split('.')[0].lower() or f'#{e["id"]}'

    def _port_view(self) -> dict:
        """The port ask's field: the proven entries by weighted gain, one per site (the best; `runs`: how many entries of
        that site were proven), at most PORT_MAX; the inert ones by weight - those of a shown site first -, one per site
        and reason, at most INERT_MAX."""
        rows = sorted(((self._entry_weight(e) * e['gain_pct'], self._entry_weight(e), e) for e in self.brief_proven),
                      key=lambda r: (-r[0], r[2]['id']))
        shown, groups = [], {}
        for wg, w, e in rows:
            if w <= 0.0:
                continue
            key = self._site_key(e)
            if key in groups:
                groups[key]['runs'] += 1
                continue
            row = {'id': e['id'], 'mechanism': e['mechanism'], 'site': e['site'], 'where': e['where'],
                   'gain_pct': round(e['gain_pct'], 2), 'weight': round(w, 3), 'proof': e['proof'],
                   'source': _shown_env(e['env']), 'runs': 1}
            shown.append({k: v for k, v in row.items() if v is not None})
            groups[key] = shown[-1]                      # (a later entry of the site counts in its `runs`)
        shown = shown[:PORT_MAX]
        sites = {self._site_key(r) for r in shown}
        inert = sorted(((self._entry_weight(e), e) for e in self.brief_inert), key=lambda r: (self._site_key(r[1]) not in sites,
                                                                                         -r[0], r[1]['id']))
        dnr, seen = [], set()
        for w, e in inert:
            key = (self._site_key(e), str(e['reason'] or '').split(':', 1)[0])
            if w <= 0.0 or key in seen:
                continue
            seen.add(key)
            dnr.append({k: v for k, v in {'mechanism': e['mechanism'], 'site': e['site'], 'reason': e['reason'],
                                          'source': _shown_env(e['env'])}.items() if v is not None})
            if len(dnr) >= INERT_MAX:
                break
        return {'rule': PORT_RULE, 'proven': shown, 'do_not_repeat': dnr}

    def _port_due(self, engine, left: int) -> bool:
        n = engine.full_screens()
        return (self.xon['xfer_port'] and self.brief_used and not self.port_tried and not self.queue and left >= 1
                and n >= 1 and self.last_ask_at != n and self.asks < self.code_asks and self._can_code()
                and self._budget_ok(engine))

    def _lift_review_veto(self, prop):
        """xfer's port request only (XFER_r1 2): the LLMProposer's one-off code review - its first patch call, without the
        transfer field - may answer no_patch, after which the proposer would drop the code and not send the patch call;
        for this one request the verdict reads PORT_VERDICT (the review's no_patch, and why the code is shown anyway: what
        the model sees in `code_review`), so the port request goes out with the code. Returns what puts the review back as
        it answered, for every later ask (wcs's own behaviour, and every other strategy's, unchanged)."""
        held = {}

        def lift(r):
            if isinstance(r, dict) and r.get('verdict') == 'no_patch':
                held['review'] = r
                prop.review = {**r, 'verdict': PORT_VERDICT}
        if getattr(prop, 'review', None) is not None:    # (reviewed already: the verdict lifted for this request)
            lift(prop.review)
        elif callable(getattr(prop, '_review', None)) and hasattr(prop, 'review'):
            real = prop._review

            def review_then_port(engine, base, code, digest):
                out = real(engine, base, code, digest)
                lift(getattr(prop, 'review', None))
                return out
            prop._review = review_then_port              # (an instance attribute over the method, for this call only)

        def restore():
            prop.__dict__.pop('_review', None)
            if 'review' in held:
                prop.review = held['review']
        return restore

    def _spy_dispatch(self, flag: dict):
        """What tells whether the port's patch request really went out: a `hypotheses` request the budget driver passed
        to the backend (one it refuses - the budget spent - counts in its `refused`). Returns its undoing; without a
        budget driver (a simulated or scripted model) the call itself is the request (flag['direct'])."""
        bd = self._bdriver
        if bd is None:
            flag['direct'] = True
            return lambda: None
        real = bd.complete

        def spy(system, prompt, **kw):
            n0 = bd.refused
            res = real(system, prompt, **kw)
            if kw.get('role') == 'hypotheses' and bd.refused == n0:
                flag['sent'] = True
            return res
        bd.__dict__['complete'] = spy                   # (an instance attribute over the method, for this call only)
        return lambda: bd.__dict__.pop('complete', None)

    def _port_ask(self, engine, inc) -> bool:
        """The port ask (the module docstring): one patch call with the brief's view - sent with the code even after the
        code review's no_patch (_lift_review_veto) -; at most 3 of its patches queued, the port's first by their entry's
        weighted gain, then the code review's (the proposer's own first call) to fill. Done (`port_done`, the entries
        `shown`) only when that request really went out; made once either way (no retry: what stopped it - no code to
        patch, the model budget - does not come back), the code review's patches it brought back queued either way (as
        the ordinary ask queues them). False when it showed nothing (every entry weighs 0): the step's ordinary ask goes
        ahead."""
        view = self._port_view()
        self.port_tried = True
        if not view['proven']:
            self.notes.append('the port ask had no proven patch to show (every entry weighs 0): an ordinary ask instead')
            return False
        self.asks += 1
        self.last_ask_at = engine.full_screens()
        self._allowance()
        self._port_payload = view
        flag: dict = {}
        restore_review = self._lift_review_veto(self.proposer)
        restore_driver = self._spy_dispatch(flag)
        raised = False
        try:
            hyps = self.proposer.hypotheses(engine, inc, ledger=self.model_view())
        except Exception as exc:                         # noqa: BLE001 - a failed call: nothing queued, not a dead arm
            (getattr(engine, 'log', None) or (lambda m: None))(f'[search] {self.name}: the port call failed: {str(exc)[:200]}')
            hyps, raised = [], True
        finally:
            self._port_payload = None
            restore_driver()
            restore_review()
        self.llm_calls = self._llm_used()
        sent = bool(flag.get('sent') or flag.get('direct') and not raised)
        self.port_done = sent
        review = {h.get('mechanism') for h in getattr(self.proposer, 'review_patches', None) or [] if isinstance(h, dict)}
        usable = [h for h in hyps or [] if isinstance(h, dict) and h.get('patch') and not (h.get('change') or {})]
        # (not sent: nothing was shown - the review's patches alone come back, queued as the ordinary ask would)
        weight = {r['id']: r['gain_pct'] * r['weight'] for r in view['proven']} if sent else {}
        port, rev = [], []
        for i, h in enumerate(usable):
            m = re.search(PORT_ID, str(h.get('mechanism') or ''), re.I)
            eid = m.group(1).lower() if m else None
            eid = eid if eid in weight else None
            (rev if h.get('mechanism') in review and eid is None else port).append((-(weight.get(eid) or 0.0), eid is None, i, eid, h))
        port.sort(key=lambda t: t[:3])
        queued = []
        for _, _, _, eid, h in (port + rev)[:self.patches_per_ask]:
            q = self._patch_proposal(engine, inc, h)
            if q is None:
                continue
            self.queue.append(q)
            pid = q.hypothesis.split('|', 1)[1]
            if eid:
                self.ported[pid] = eid
            queued.append({'patch': pid, 'entry': eid, 'mechanism': str(h.get('mechanism') or '')[:120]})
        if not sent:
            self.port = {'screens_done': engine.full_screens(), 'sent': False, 'shown': [], 'answers': 0, 'queued': queued}
            self.notes.append('the port request did not go out (no code to patch, the model budget, or a failed call): '
                              'ordinary asks only')
            self.steps.append({'screens_done': engine.full_screens(), 'pick': 'port ask not sent', 'queued': len(queued)})
            return True
        self.port = {'screens_done': engine.full_screens(), 'sent': True, 'shown': [r['id'] for r in view['proven']],
                     'do_not_repeat': len(view['do_not_repeat']), 'answers': len(usable), 'queued': queued}
        self.steps.append({'screens_done': engine.full_screens(), 'pick': 'port ask', 'queued': len(queued)})
        return True

    def _code(self, engine, inc, left, ask: bool = True) -> list:
        """search/llmopt's code track; its first ask is the port ask when the brief has proven patches."""
        if ask and self._port_due(engine, left) and self._port_ask(engine, inc):
            ask = False                                  # (the port ask was this step's ask)
        return super()._code(engine, inc, left, ask=ask)

    # ------------------------------------------------------------ proposals
    def propose(self, engine, k):
        if self.prior is None:
            self._init(engine)
        self.engine = engine
        left = self._left(engine)
        if left <= 0:
            return []
        inc = engine.incumbent()
        while self.pending:
            q = self.pending.pop(0)
            if engine._fresh(q):
                return [q]
        q = self._step(engine, inc, left)
        return [q] if q is not None else []

    def _note(self, engine, pick: str, q):
        if q is not None:
            self.steps.append({'screens_done': engine.full_screens(), 'pick': pick, 'id': q.hypothesis})
        return q

    def _step(self, engine, inc, left):
        if self.phase == 'source':                       # screen 1 (and 2 after a failed source full run)
            q = self._source_step(engine, inc)
            if q is not None:
                return self._note(engine, 'source', q)
            self.phase = 'prior'
        if self.phase == 'prior':                        # once: the prior's configuration (first, or by screen 2's rule)
            self.phase = 'code'
            q = self._prior_step(engine, inc)
            if q is not None:
                return self._note(engine, 'prior', q)
        if left == 1:                                    # the last screen: a source winner within 2 sigma, else the draw
            q = self._late_source(engine, inc)
            if q is not None:
                return self._note(engine, 'last: source', q)
            q = self._fallback(engine, inc)
            if q is not None:
                return self._note(engine, 'last: draw', q)
        if left == 2 and self.xon['xfer_tail']:          # the screen before: the tail bundle
            q = self._tail_step(engine, inc, left)
            if q is not None:
                return self._note(engine, 'tail', q)
        if self.on['wcs_code']:                          # code: asks only while a patch can still get its full screen
            n0 = self.asks
            q = self._code_step(engine, inc, left, ask=left >= 3)
            if self.asks != n0 and not (self.steps and str(self.steps[-1].get('pick')).startswith('port ask')):
                self.steps.append({'screens_done': engine.full_screens(), 'pick': 'code ask'})
            if q is not None:
                return self._note(engine, 'code', q)
        q = self._late_source(engine, inc)               # nothing of its own: a source winner, the tail, the draw
        if q is not None:
            return self._note(engine, 'fill: source', q)
        if self.xon['xfer_tail']:
            q = self._tail_step(engine, inc, left)
            if q is not None:
                return self._note(engine, 'fill: tail', q)
        return self._note(engine, 'fill: draw', self._fallback(engine, inc))

    # ------------------------------------------------------------ outcomes
    def observe(self, engine, p, out):
        cls = super().observe(engine, p, out)
        hid = p.hypothesis or ''
        s = self.by_src.get(hid)
        if s is not None and s.get('state') == 'proposed':
            if out.status == 'duplicate':
                s['state'] = 'repeat'
            elif out.fidelity == 'full' and out.status in ('screened', 'aborted', 'invalid'):   # its full rung ran: a
                s['state'], s['outcome'] = 'screened', out.status                                 # screen, counted
                if out.status == 'screened' and out.cid:
                    s['y'] = (self.runs.get(out.cid) or {}).get('y')
                if s.get('phase') == 'source':
                    self.src_fulls += 1
                    self.src_ok = bool(out.cid) and engine.incumbent() == out.cid
            elif out.fidelity == 'probe':
                s['state'], s['outcome'] = 'probe failed', out.status
            else:
                s['state'], s['outcome'] = 'not run', out.status
        if self.xon['xfer_rgpe'] and out.fidelity == 'full' and out.status == 'screened' and out.cid:
            self._rgpe(engine, out.cid)
        return cls

    # ------------------------------------------------------------ report
    def describe(self) -> dict:
        view, learned, prop = self.candidate_view()

        def src(s):
            return {'source': s.get('index'), 'env': _shown_env(s['env']), 'packs': len(s['packs']),
                    'weight_start': round(s['w0'], 3), 'weight': round(s['w'], 4), 'state': s.get('state'),
                    **({'outcome': s['outcome']} if s.get('outcome') else {}),
                    **({'winner_gain_pct': round(100 * math.expm1(s['winner']['gain']), 2)} if s.get('winner') else {}),
                    **({'change': self._words_of(s['cfg'])} if s.get('cfg') else {}),
                    **({'measured_gain_pct': round(100 * math.expm1(s['y']), 2)} if s.get('y') is not None else {}),
                    'effects': len(s['effects']) // 2, **({'notes': s['map_notes'][:4]} if s.get('map_notes') else {})}
        ported = {pid: {'entry': eid, 'status': (self.patches.get(pid) or {}).get('status'),
                        **({'gain_pct': self.patches[pid]['gain_pct']} if (self.patches.get(pid) or {}).get('gain_pct')
                           is not None else {})} for pid, eid in self.ported.items()}
        out = {'name': self.name, 'proposer': prop,
               'xfer': {'name': self.name, 'off': sorted(k for k, v in self.xon.items() if not v), **view,
                        **self.llmopt_view(), 'target_env': _shown_env(self.target or {}) if self.target else None,
                        'sources': [src(s) for s in self.sources], 'weights': self.weights[-12:],
                        'source_full_runs': self.src_fulls,
                        'oom_families': {**self.oom_gate, 'blamed': [
                            f'{self._words_kv(b)} ({cid})' + ''.join(f'; waived with {self._words_kv(c)}' for c in waive)
                            for b, cid, waive in self._blame][:12]},
                        'brief': {'proven': len(self.brief_proven), 'inert': len(self.brief_inert), 'used': self.brief_used},
                        'port': {**self.port, 'tried': self.port_tried, 'done': self.port_done, 'ported': ported},
                        'notes': self.notes, 'steps': self.steps[-60:]}}
        if learned:
            out['learned'] = learned                     # D4: written back for a later arm
        return out
