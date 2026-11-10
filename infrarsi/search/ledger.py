"""The memory ledger: a censored linear model of a configuration's peak memory per GPU (possible_search_algorithm/20
section 2, "Memory Ledger") - a shared component any Engine strategy can switch on (esls and rpp use it; wcs may later).

Every run is a memory observation of its configuration c, in GiB per GPU:

    exact      a peak: a passed probe, a short run, a valid full run (D3's measured peak first, else the log's)
    lower      out of memory: the need is at least what was in use plus what the failed allocation asked for (hls3's
               parse_oom need_gib; without the numbers, at least the card) - a right-censored observation, never a peak
    interval   the cliff (thrash): the need lies between NEAR x card and the card

The model m(c) = sum_j theta_j phi_j(c) has three kinds of basis functions:

    intercept  1 - the reference's memory (its measured peak, when the reference has one, is its first observation)
    value      an indicator per knob value other than the reference's; prior mean the GiB the prior declared for it
               (signed by its direction: spends +, frees -, neutral 0), spread half of it; an unknown direction 0 with a
               wide spread
    term       a formula the prior wrote (`gib_expr`: arithmetic over the knob values and the manifest's numeric facts,
               evaluated by the manifest's safe evaluator; `when`: the knob values it applies under), coefficient prior
               N(1, 0.5^2) - the shape from the model, the scale from the runs
    patch      an indicator per code patch a configuration carries (wcs's pseudo-knob patch:<id>), prior mean the GiB
               its idea declared (signed), else 0 with a wide spread; a configuration is its knob values plus the patches
               it carries (`ids`): a run's own result never stands for the same knobs under another overlay

The fit is the MAP under Gaussian priors and a Tobit likelihood (exact: normal; lower: log Phi; interval: the log of a
Phi difference) by damped Newton steps with a dense Cholesky (pure Python). P_feas(c) = Phi((margin x card - mean) / sd)
with the Laplace posterior's spread; a configuration that ran is its own result. Nothing here names a knob, a model or a
library: every name comes from the prior's answer, the manifest and the table.

    led = MemoryLedger(cap, knobs=table, ref=reference, facts=numeric_facts)
    led.add_value_terms(levels); led.add_formula_terms(terms)
    led.add_patch(pid, mem_dir, gib)                       # a code patch as a pseudo-knob
    led.observe_run(cfg, kind, peak=..., oom=..., ids=())  # or observe_exact / observe_lower / observe_interval
    led.predict(cfg, ids=()) -> (mean, sd); led.p_feasible(cfg, ids=()); led.lower_bound(cfg, ids=()); led.ran_status
"""
from __future__ import annotations

import ast
import json
import math

NEAR = 0.92                     # the allocator thrashes from about this share of the card (the cliff)
THRASH_LO = 0.90                # a thrashing configuration's need lies in [THRASH_LO x card, card]
SD_OBS = 0.02                   # an observation's spread, as a share of the card (the peak's run-to-run noise)
SD_UNKNOWN = 0.12               # a value of unknown memory direction: 0 +- this share of the card
SD_DECLARED = 0.5               # a declared GiB: +- half of it (the recorded declarations: a third or more off)
SD_DECLARED_MIN = 0.02          # ... and at least this share of the card
SD_TERM = 0.5                   # a formula term's coefficient: 1 +- this
INTERACT = 0.5                  # the additive model's own error for a configuration of two or more changed values:
#                                 this x sqrt(sum over their pairs of |effect_i| x |effect_j|) GiB (effects interact:
#                                 what one value frees often scales with what another spends)
TERMS_MAX = 8                   # formula terms kept at most
NEWTON_ITERS = 40
PATCH = 'patch:'                # a carried code patch as a pseudo-knob (wcs's form)


_VK: dict = {}                  # (type, value) -> its key: the keys of a table's values, built once


def vk(v) -> str:
    """A knob value as a comparable key (wcs's form: json, sorted keys), memoised for hashable values (a draw over a
    large table builds millions of keys; the same values every time)."""
    try:
        key = (type(v), v)
        out = _VK.get(key)
    except TypeError:                                   # (an unhashable value: built each time)
        return json.dumps(v, sort_keys=True, default=str)
    if out is None:
        out = _VK[key] = json.dumps(v, sort_keys=True, default=str)
    return out


def _phi(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _Phi(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2))


def _tail(z: float) -> tuple[float, float, float]:
    """(-log Phi(z), lam = phi(z) / Phi(z), lam (z + lam)) for any z: far in the lower tail by the Mills ratio's
    series (an observation many spreads away from the fit still pulls it, with its full gradient)."""
    if z > -30.0:
        P = _Phi(z)
        if P > 0.0:
            lam = _phi(z) / P
            return -math.log(P), lam, max(0.0, lam * (z + lam))
    x = -z
    r = 1.0 - 1.0 / (x * x) + 3.0 / x ** 4             # Phi(z) ~ phi(z) r / x
    lam = x / r
    return 0.5 * z * z + 0.5 * math.log(2 * math.pi) + math.log(x / r), lam, max(0.0, lam * (z + lam))


def _cholesky_solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    """a x = b for a symmetric positive definite a (None when it is not)."""
    n = len(b)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = a[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 1e-12:
                    return None
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
    return x


class MemoryLedger:
    """The censored memory model of the module docstring. `knobs`: knob -> its values (the table); `ref`: the
    reference's value per knob; `facts`: numeric facts a formula term may use (the manifest's: card, GPUs, parameters,
    ...); `margin`: the share of the card a configuration must stay below to count as fitting."""

    def __init__(self, cap_gib: float | None, *, knobs: dict, ref: dict, facts: dict | None = None, margin: float = NEAR,
                 sd_obs: float = SD_OBS):
        self.cap = float(cap_gib) if isinstance(cap_gib, (int, float)) and cap_gib > 0 else None
        self.knobs = {k: list(v) for k, v in (knobs or {}).items()}
        self.ref = dict(ref or {})
        self.facts = {k: v for k, v in (facts or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        self.margin = float(margin)
        self.sd_obs = max(0.5, sd_obs * self.cap) if self.cap else 2.0
        scale = self.cap or 100.0
        self.basis: list[dict] = [{'name': 'reference', 'kind': 'intercept', 'mu': 0.6 * scale, 'sd': scale}]
        self.obs: list[tuple] = []                      # (cfg, kind, a, b, source)
        self.ran: dict[str, dict] = {}                  # configuration key -> {'status', 'peak'}
        self.dropped_terms: list[dict] = []
        self._fit: tuple | None = None                  # (theta, H^-1) of the last fit
        self._phi: dict[str, list] = {}                 # configuration key -> its basis values (the basis is fixed
        self.fits = 0                                   # once the terms are added)
        self._pred: dict[str, tuple] = {}               # configuration key -> (mean, sd) under the current fit
        self.patch_ids: set = set()                     # the code patches with an indicator

    # ------------------------------------------------------------ the basis
    @staticmethod
    def _full(cfg: dict, ids=()) -> dict:
        """The configuration with the patches it carries as pseudo-knobs (patch:<id> true)."""
        if not ids:
            return cfg
        return {**cfg, **{f'{PATCH}{pid}': True for pid in ids}}

    def _key(self, cfg: dict) -> str:
        return json.dumps(sorted((k, vk(cfg.get(k, self.ref.get(k)))) for k in self.knobs)
                          + sorted(k for k, v in cfg.items() if isinstance(k, str) and k.startswith(PATCH) and v),
                          default=str)

    def add_patch(self, pid: str, mem_dir: str | None = None, gib=None) -> bool:
        """An indicator for a code patch (once): its prior mean the GiB its idea declared, signed by the direction; an
        unknown direction 0 with the wide spread."""
        if pid in self.patch_ids:
            return False
        scale = self.cap or 100.0
        g = float(gib) if isinstance(gib, (int, float)) and not isinstance(gib, bool) and gib > 0 else None
        if mem_dir in ('spends', 'frees') and g:
            mu, sd = (g if mem_dir == 'spends' else -g), max(SD_DECLARED * g, SD_DECLARED_MIN * scale)
        elif mem_dir == 'neutral':
            mu, sd = 0.0, SD_DECLARED_MIN * scale
        else:
            mu, sd = 0.0, SD_UNKNOWN * scale
        self.basis.append({'name': f'{PATCH}{pid}', 'kind': 'patch', 'pid': pid, 'mu': mu, 'sd': sd, 'declared': mem_dir})
        self.patch_ids.add(pid)
        self._fit, self._phi, self._pred = None, {}, {}
        return True

    def add_value_terms(self, levels: dict) -> int:
        """One indicator per knob value other than the reference's, from the prior's levels ((knob, vk) -> {'mem_dir',
        'mem_gib'}; wcs.Prior.levels): its mean the declared GiB, signed; unknown: 0 with a wide spread."""
        scale = self.cap or 100.0
        n = 0
        for (k, key), e in (levels or {}).items():
            if k not in self.knobs or any(b['kind'] == 'value' and b['knob'] == k and b['vk'] == key for b in self.basis):
                continue
            d, g = (e or {}).get('mem_dir'), (e or {}).get('mem_gib')
            if d == 'spends' and g:
                mu, sd = float(g), max(SD_DECLARED * float(g), SD_DECLARED_MIN * scale)
            elif d == 'frees' and g:
                mu, sd = -float(g), max(SD_DECLARED * float(g), SD_DECLARED_MIN * scale)
            elif d == 'neutral':
                mu, sd = 0.0, SD_DECLARED_MIN * scale
            else:
                mu, sd = 0.0, SD_UNKNOWN * scale
            self.basis.append({'name': f'{k}={key}', 'kind': 'value', 'knob': k, 'vk': key, 'mu': mu, 'sd': sd,
                               'declared': d})
            n += 1
        self._fit, self._phi = None, {}
        return n

    def ensure_values(self) -> int:
        """An indicator (unknown direction) for every non-reference value the prior did not give."""
        levels = {(k, vk(v)): {} for k, vals in self.knobs.items() for v in vals if vk(v) != vk(self.ref.get(k))}
        return self.add_value_terms(levels)

    def _names(self, cfg: dict) -> dict:
        out = dict(self.facts)
        for k in self.knobs:
            v = cfg.get(k, self.ref.get(k))
            if isinstance(v, (int, float, str, bool)) or v is None:
                out[k] = v
        return out

    def _term_value(self, term: dict, cfg: dict) -> float:
        """A formula term's GiB on a configuration: 0 outside its `when`; the expression by the manifest's safe evaluator,
        cached by the values of the knobs it reads."""
        from ..project.manifest import evaluate
        key = tuple(vk(cfg.get(k, self.ref.get(k))) for k in term['reads'])
        if key in term['cache']:
            return term['cache'][key]
        v = 0.0
        when = term.get('when') or {}
        if all(vk(cfg.get(k, self.ref.get(k))) in {vk(x) for x in (a if isinstance(a, list) else [a])}
               for k, a in when.items()):
            v = evaluate(term['expr'], self._names(cfg))
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f'{term["expr"]!r} is not a number')
            v = float(v)
        term['cache'][key] = v
        return v

    def add_formula_terms(self, terms) -> int:
        """The prior's formula terms ([{'name', 'gib_expr', 'when'}]): each kept when it evaluates to a finite number of
        GiB, not above ten cards, on the reference and on every value of every knob (one knob changed at a time); else
        dropped and counted. Returns how many were kept."""
        kept = 0
        scale = self.cap or 100.0
        for t in (terms or [])[:2 * TERMS_MAX]:
            if not isinstance(t, dict) or not isinstance(t.get('gib_expr'), str) or kept >= TERMS_MAX:
                if isinstance(t, dict):
                    self.dropped_terms.append({'name': str(t.get('name'))[:60], 'why': 'no expression' if kept < TERMS_MAX
                                               else 'too many terms'})
                continue
            when = t.get('when') if isinstance(t.get('when'), dict) else {}
            when = {k: v for k, v in when.items() if k in self.knobs}
            term = {'name': str(t.get('name') or f'term {kept + 1}')[:60], 'kind': 'term', 'expr': t['gib_expr'][:400],
                    'when': when, 'mu': 1.0, 'sd': SD_TERM, 'cache': {}}
            try:
                names = {n.id for n in ast.walk(ast.parse(term['expr'], mode='eval')) if isinstance(n, ast.Name)}
                term['reads'] = sorted((names | set(when)) & set(self.knobs))
                vals = [self._term_value(term, self.ref)]
                for k, choices in self.knobs.items():
                    for v in choices:
                        vals.append(self._term_value(term, {**self.ref, k: v}))
                if any(abs(x) > 10 * scale for x in vals):
                    raise ValueError('beyond ten cards')
            except Exception as exc:                     # noqa: BLE001 - a term the evaluator refuses is dropped
                self.dropped_terms.append({'name': term['name'], 'why': str(exc)[:120]})
                continue
            self.basis.append(term)
            kept += 1
        self._fit, self._phi = None, {}
        return kept

    def phi(self, cfg: dict) -> list[float]:
        """The basis functions of a configuration (knob values; the missing ones at the reference's)."""
        key = self._key(cfg)
        if key in self._phi:
            return self._phi[key]
        out = []
        for b in self.basis:
            if b['kind'] == 'intercept':
                out.append(1.0)
            elif b['kind'] == 'value':
                out.append(1.0 if vk(cfg.get(b['knob'], self.ref.get(b['knob']))) == b['vk'] else 0.0)
            elif b['kind'] == 'patch':
                out.append(1.0 if cfg.get(f'{PATCH}{b["pid"]}') else 0.0)
            else:
                try:
                    out.append(self._term_value(b, cfg))
                except Exception:                        # noqa: BLE001 - a value the term cannot read: no contribution
                    out.append(0.0)
        self._phi[key] = out
        return out

    # ------------------------------------------------------------ observations
    def observe_exact(self, cfg: dict, gib: float, source: str = '', *, ids=()) -> None:
        if isinstance(gib, (int, float)) and math.isfinite(gib) and gib > 0:
            self.obs.append((dict(self._full(cfg, ids)), 'exact', float(gib), None, source))
            self._fit = None

    def observe_lower(self, cfg: dict, gib: float | None, source: str = '', *, ids=()) -> None:
        y = float(gib) if isinstance(gib, (int, float)) and math.isfinite(gib) and gib > 0 else self.cap
        if y:
            self.obs.append((dict(self._full(cfg, ids)), 'lower', y, None, source))
            self._fit = None

    def observe_interval(self, cfg: dict, lo: float | None = None, hi: float | None = None, source: str = '', *,
                         ids=()) -> None:
        if not self.cap:
            return
        lo = THRASH_LO * self.cap if lo is None else float(lo)
        hi = self.cap if hi is None else float(hi)
        if hi > lo:
            self.obs.append((dict(self._full(cfg, ids)), 'interval', lo, hi, source))
            self._fit = None

    def observe_run(self, cfg: dict, kind: str, *, peak=None, oom: dict | None = None, source: str = '',
                    ids=()) -> str | None:
        """A run's memory evidence by hls3's classify kind: ok / slower (a peak: exact), oom (lower: need_gib, else
        in-use plus requested, else the card), thrash (interval). Returns the observation kind recorded (None: none).
        `ids`: the code patches the run carried (its identity and its features)."""
        cfg = self._full(cfg, ids)
        key = self._key(cfg)
        if kind in ('ok', 'slower', 'numerics'):
            if isinstance(peak, (int, float)) and peak > 0:
                self.observe_exact(cfg, peak, source)
                self.ran[key] = {'status': 'ok', 'peak': float(peak)}
                return 'exact'
            self.ran.setdefault(key, {'status': 'ok', 'peak': None})
            return None
        if kind == 'oom':
            o = oom or {}
            need = o.get('need_gib')
            if need is None and isinstance(o.get('tried_gib'), (int, float)) and isinstance(o.get('in_use_gib'), (int, float)):
                need = o['tried_gib'] + o['in_use_gib']
            self.observe_lower(cfg, need, source)
            self.ran[key] = {'status': 'oom', 'peak': None}
            return 'lower'
        if kind == 'thrash':
            self.observe_interval(cfg, source=source)
            self.ran[key] = {'status': 'thrash', 'peak': peak if isinstance(peak, (int, float)) else None}
            return 'interval'
        return None

    # ------------------------------------------------------------ the fit
    def _objective(self, theta, rows):
        """(negative log posterior, gradient, Hessian) at theta; rows: (phi, kind, a, b)."""
        n = len(theta)
        f = 0.0
        g = [0.0] * n
        H = [[0.0] * n for _ in range(n)]
        for j, b in enumerate(self.basis):
            prec = 1.0 / (b['sd'] ** 2)
            d = theta[j] - b['mu']
            f += 0.5 * prec * d * d
            g[j] += prec * d
            H[j][j] += prec
        s = self.sd_obs
        for x, kind, a, bb in rows:
            m = sum(xi * ti for xi, ti in zip(x, theta))
            if kind == 'exact':
                r = (m - a) / s
                f += 0.5 * r * r
                dm, d2 = r / s, 1.0 / (s * s)
            elif kind == 'lower':                       # -log Phi((m - a) / s)
                t, lam, k2 = _tail((m - a) / s)
                f += t
                dm, d2 = -lam / s, k2 / (s * s)
            else:                                       # -log(Phi((hi - m)/s) - Phi((lo - m)/s))
                za, zb = (a - m) / s, (bb - m) / s
                D = _Phi(-za) - _Phi(-zb) if za > 0 else _Phi(zb) - _Phi(za)   # (the accurate side of the difference)
                if D > 1e-12:
                    pa, pb = _phi(za), _phi(zb)
                    f -= math.log(D)
                    dm = (pb - pa) / (s * D)
                    d2 = max(0.0, (zb * pb - za * pa) / (s * s * D) + ((pb - pa) / D) ** 2 / (s * s))
                elif zb <= 0:                           # far above the interval: m <= hi's censored form
                    t, lam, k2 = _tail(zb)
                    f += t
                    dm, d2 = lam / s, k2 / (s * s)
                else:                                   # far below it: m >= lo's
                    t, lam, k2 = _tail(-za)
                    f += t
                    dm, d2 = -lam / s, k2 / (s * s)
            for i in range(n):
                if x[i]:
                    g[i] += dm * x[i]
                    for j in range(i + 1):
                        if x[j]:
                            H[i][j] += d2 * x[i] * x[j]
        for i in range(n):
            for j in range(i):
                H[j][i] = H[i][j]
        return f, g, H

    def fit(self) -> list[float]:
        """The MAP coefficients (damped Newton); cached until the next observation."""
        if self._fit is not None:
            return self._fit[0]
        rows = [(self.phi(c), k, a, b) for c, k, a, b, _ in self.obs]
        theta = [b['mu'] for b in self.basis]
        f, g, H = self._objective(theta, rows)
        for _ in range(NEWTON_ITERS):
            step = _cholesky_solve(H, g)
            if step is None:
                break
            t, done = 1.0, False
            while t > 1e-6:
                cand = [ti - t * si for ti, si in zip(theta, step)]
                f2, g2, H2 = self._objective(cand, rows)
                if f2 <= f + 1e-12:
                    done = abs(f - f2) < 1e-9
                    theta, f, g, H = cand, f2, g2, H2
                    break
                t *= 0.5
            else:
                break
            if done:
                break
        n = len(theta)
        cov = []
        for i in range(n):                              # the Laplace posterior's covariance: H^-1, column by column
            e = [0.0] * n
            e[i] = 1.0
            col = _cholesky_solve(H, e)
            cov.append(col if col is not None else [0.0] * n)
        self._fit = (theta, cov)
        self._pred = {}
        self.fits += 1
        return theta

    # ------------------------------------------------------------ predictions
    def predict(self, cfg: dict, *, ids=()) -> tuple[float, float]:
        """(mean, sd) of the configuration's peak memory per GPU (GiB): the fit's mean; its spread: the fit's, the
        observation noise and, for two or more changed values, the additive model's own error (INTERACT). `ids`: the
        code patches it carries."""
        cfg = self._full(cfg, ids)
        theta = self.fit()
        key = self._key(cfg)
        got = self._pred.get(key)
        if got is not None:
            return got
        cov = self._fit[1]
        x = self.phi(cfg)
        mu = sum(a * b for a, b in zip(x, theta))
        var = sum(x[i] * x[j] * cov[i][j] for i in range(len(x)) if x[i] for j in range(len(x)) if x[j])
        eff = [abs(t) for b, xi, t in zip(self.basis, x, theta) if b['kind'] in ('value', 'patch') and xi]
        tot = sum(eff)
        pairs = (tot * tot - sum(e * e for e in eff)) / 2.0         # sum over the pairs of |e_i| |e_j|
        out = (mu, math.sqrt(max(0.0, var) + self.sd_obs ** 2 + INTERACT ** 2 * max(0.0, pairs)))
        self._pred[key] = out
        return out

    def ran_status(self, cfg: dict, *, ids=()) -> str | None:
        """The run's own result for these knob values under exactly these patches (another overlay: none)."""
        r = self.ran.get(self._key(self._full(cfg, ids)))
        return r['status'] if r else None

    def p_feasible(self, cfg: dict, *, ids=()) -> float:
        """P(the configuration fits below margin x card); a configuration that ran (with the same patches): 1 when it
        fitted, 0 when it ran out of memory or thrashed. Without a card: 1."""
        st = self.ran_status(cfg, ids=ids)
        if st == 'ok':
            return 1.0
        if st in ('oom', 'thrash'):
            return 0.0
        if not self.cap:
            return 1.0
        mu, sd = self.predict(cfg, ids=ids)
        return _Phi((self.margin * self.cap - mu) / sd)

    def lower_bound(self, cfg: dict, z: float = 2.0, *, ids=()) -> float:
        mu, sd = self.predict(cfg, ids=ids)
        return mu - z * sd

    def order(self, cfgs: list[dict]) -> list[dict]:
        return sorted(cfgs, key=lambda c: self.predict(c)[0])

    def f_hat(self, cfgs: list[dict]) -> float:
        cfgs = list(cfgs)
        return sum(self.p_feasible(c) for c in cfgs) / len(cfgs) if cfgs else 1.0

    def describe(self) -> dict:
        theta = self.fit() if self.basis else []
        rows = [{'name': b['name'], 'kind': b['kind'], 'prior_gib' if b['kind'] != 'term' else 'prior_coef': round(b['mu'], 3),
                 'fitted': round(t, 3)} for b, t in zip(self.basis, theta)]
        rows.sort(key=lambda r: -abs(r['fitted'] - (r.get('prior_gib', r.get('prior_coef')) or 0.0)))
        kinds = {}
        for _, k, _, _, _ in self.obs:
            kinds[k] = kinds.get(k, 0) + 1
        return {'capacity_gib': self.cap, 'margin': self.margin, 'basis': len(self.basis),
                'terms': [b['name'] for b in self.basis if b['kind'] == 'term'], 'dropped_terms': self.dropped_terms,
                'observations': kinds, 'fits': self.fits, 'moved_most': rows[:8]}


def for_wcs(st) -> MemoryLedger:
    """The ledger of a strategy built on wcs (its `knobs`, `ref`, `facts`, `prior`, `mem` and `proposer`): the prior's
    declared memory per value, its formula terms (the proposer's last prior answer, `mem_terms`), an unknown-direction
    indicator for every other value, and the reference's measured peak as the first exact observation."""
    facts = {k: v for k, v in (getattr(st, 'facts', None) or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    led = MemoryLedger(getattr(st.mem, 'cap', None), knobs=st.knobs, ref=st.ref, facts=facts)
    led.add_value_terms(st.prior.levels if getattr(st, 'prior', None) else {})
    led.ensure_values()
    answer = getattr(getattr(st, 'proposer', None), 'last_prior', None)
    if isinstance(answer, dict) and isinstance(answer.get('mem_terms'), list):
        led.add_formula_terms(answer['mem_terms'])
    peak = (st.mem.runs.get('c0') or {}).get('peak')
    if isinstance(peak, (int, float)) and peak > 0:
        led.observe_exact(dict(st.ref), peak, 'c0')
    return led
