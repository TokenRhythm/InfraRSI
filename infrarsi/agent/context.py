"""Model-visible context: receipt digests (<= 1.5K tokens), profile ranges, top-k skill retrieval, rolling history.

Nothing here exposes paths, other searches, held-out material or raw logs; the agent sees only its own task,
its own candidates' digests, the package's skills/memory, and its own history.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import yaml

REQUIRED_FRONT_MATTER = ('name', 'description', 'license', 'when_to_use', 'metadata')
REQUIRED_SECTIONS = ('trigger conditions', 'required observations', 'allowed actions', 'failure signals',
                     'verification', 'success case', 'failure case')
_WORD = re.compile(r'[a-z0-9_]+')
# Retrieval prefers actionable sections (rules) over boilerplate; real cases rank between.
SECTION_WEIGHTS = {'rules': 1.3, 'allowed actions': 0.9, 'success case': 0.7, 'failure case': 0.7, 'failure signals': 0.6,
                   'trigger conditions': 0.5, 'required observations': 0.4, 'verification': 0.5, '_intro': 0.6}


def estimate_tokens(text: str) -> int:
    """Conservative token estimate (~3.5 UTF-8 bytes per token for English/JSON)."""
    return math.ceil(len(text.encode('utf-8')) / 3.5)


def _dumps(obj) -> str:
    return json.dumps(obj, separators=(',', ':'), ensure_ascii=False, default=str)


def _r(x, n=4):
    return round(x, n) if type(x) is float else x


def _numerics_view(receipt: dict) -> dict | None:
    num = receipt.get('numerics') or (receipt.get('harness') or {}).get('numerics')
    if not num:
        return None
    # AMENDMENT-001: only gated quantities are shown. Per-step grad-norm deviations (rel_max_all_steps, per_step_rel,
    # the old grad_norm_rel_max) are report-only noise the agent must not optimise against.
    keep = ('class', 'compared', 'allowed_ok', 'passed', 'violations', 'main_ce_rel_max', 'main_ce_rel_mean_window')
    out = {k: num.get(k) for k in keep if k in num}
    gn = num.get('grad_norm') if isinstance(num.get('grad_norm'), dict) else {}
    gated = {k: gn.get(k) for k in ('first_update_rel', 'log_bias_window', 'n_window') if k in gn}
    if gated:
        out['grad_norm_gated'] = gated
    sc = num.get('self_check') if isinstance(num.get('self_check'), dict) else {}
    if sc.get('repeated_pairs'):
        out['repeated_pairs'] = sc['repeated_pairs'][:6]
    if (receipt.get('harness') or {}).get('failure_kind'):
        out['failure_kind'] = receipt['harness']['failure_kind']
    return out


def receipt_digest(receipt: dict | None, verdict: dict | None = None, *, max_tokens: int = 1500, fields=None) -> dict:
    """Compact, prioritized view of a receipt; lower-priority blocks are dropped to fit max_tokens."""
    if not receipt:
        return {'receipt': None}
    st = receipt.get('step_time') or {}
    win = receipt.get('window') or [None, None]
    rows = [r for r in receipt.get('per_step') or [] if win[0] is None or win[0] <= r.get('step', 0) <= win[1]]
    mem = receipt.get('memory') or {}
    ev = receipt.get('events') or {}
    per_rank = mem.get('per_rank') or []
    resv = [r.get('peak_reserved_gib') for r in per_rank if isinstance(r, dict) and r.get('peak_reserved_gib') is not None]
    blocks = [
        ('id', {'attempt_id': receipt.get('attempt_id'), 'purpose': receipt.get('purpose'), 'valid': receipt.get('valid'),
                'invalid_reasons': (receipt.get('invalid_reasons') or [])[:4],
                'steps': f'{receipt.get("steps_completed")}/{receipt.get("steps")}'}),
        ('verdict', verdict and {k: _r(verdict.get(k)) for k in ('valid', 'reasons', 'step_time', 'reward', 'peak_reserved_gib',
                                                                 'numerics_class', 'feedback') if k != 'feedback' or verdict.get(k)}),
        ('step_time', {k: _r(st.get(k)) for k in ('mean_window_max_rank_s', 'median_window_max_rank_s', 'rsd_window', 'inter_step_mean_s')}),
        ('memory', {'max_peak_reserved_gib': _r(mem.get('max_peak_reserved_gib')), 'max_peak_alloc_gib': _r(mem.get('max_peak_alloc_gib')),
                    'reserved_min_max_over_ranks': [_r(min(resv)), _r(max(resv))] if resv else None}),
        ('events', {'alloc_retries': ev.get('alloc_retries'), 'alloc_ooms': ev.get('alloc_ooms'), 'nccl_errors': ev.get('nccl_errors'),
                    'nan_inf_steps': ev.get('nan_inf_steps'), 'exceptions': [str(e)[:300] for e in (ev.get('exceptions') or [])[:2]]}),
        ('numerics', _numerics_view(receipt)),
        ('phases', {k: _r(v, 1) for k, v in (receipt.get('phases') or {}).items()}),
        ('window_steps', {'max_rank_whole_s': [_r(r.get('max_rank_whole_s')) for r in rows][:40],
                          'slowest_rank_counts': dict(Counter(r.get('slowest_rank') for r in rows).most_common(4))}),
        ('scalars_last', rows[-1] and {k: _r(rows[-1].get(k), 6) for k in ('main_ce', 'aux_loss', 'grad_norm', 'lr')} if rows else None),
    ]
    if fields:
        keep = set(fields) | {'id', 'verdict'}
        blocks = [b for b in blocks if b[0] in keep]
    out = {k: v for k, v in blocks if v is not None}
    while estimate_tokens(_dumps(out)) > max_tokens and len(out) > 2:
        out.pop(list(out)[-1])
    return out


def receipt_view(receipt: dict | None, parent: dict | None = None, task_raw: dict | None = None, *,
                 hbm_cap_gib: float = 78.0, _nested: bool = False) -> dict | None:
    """Fixed-shape JSON view that package analysis tools (rsi.pure_analysis) read as `data`."""
    if not receipt:
        return None
    win = receipt.get('window') or [None, None]
    rows = [r for r in receipt.get('per_step') or [] if win[0] is None or win[0] <= r.get('step', 0) <= win[1]]
    mem = receipt.get('memory') or {}
    per_rank = [r.get('peak_reserved_gib') for r in mem.get('per_rank') or [] if isinstance(r, dict)]
    st = receipt.get('step_time') or {}
    ev = receipt.get('events') or {}
    w = (task_raw or {}).get('workload', {})
    view = {
        'valid': receipt.get('valid'),
        'task': {'gpus': receipt.get('world_size'), 'seq_len': w.get('seq_len'), 'packs_per_update': w.get('packs_per_update'),
                 'tokens_per_update': w.get('physical_tokens_per_update'), 'hbm_cap_gib': hbm_cap_gib},
        'step_time': {k: st.get(k) for k in ('mean_window_max_rank_s', 'median_window_max_rank_s', 'rsd_window', 'inter_step_mean_s')},
        'window_steps': [r.get('max_rank_whole_s') for r in rows if type(r.get('max_rank_whole_s')) in (int, float)][:256],
        'slowest_ranks': [r.get('slowest_rank') for r in rows if type(r.get('slowest_rank')) is int][:256],
        'memory': {'max_peak_reserved_gib': mem.get('max_peak_reserved_gib'), 'max_peak_alloc_gib': mem.get('max_peak_alloc_gib'),
                   'reserved_per_rank': [x for x in per_rank if type(x) in (int, float)][:256]},
        'events': {k: ev.get(k) for k in ('alloc_retries', 'alloc_ooms', 'nccl_errors', 'nan_inf_steps')},
        'phases': {k: v for k, v in (receipt.get('phases') or {}).items() if type(v) in (int, float)},
        'profile': receipt.get('profile') if isinstance(receipt.get('profile'), dict) else None,
    }
    if not _nested:
        view['parent'] = receipt_view(parent, None, task_raw, hbm_cap_gib=hbm_cap_gib, _nested=True) if parent else None
    return json.loads(json.dumps(view, allow_nan=False, default=str))


def run_tools(tools: dict[str, str], view: dict | None) -> dict:
    """Run package analysis tools on a receipt view; errors are reported per tool, never raised."""
    from ..rsi.pure_analysis import run
    out = {}
    if view is None:
        return out
    for name, source in sorted(tools.items()):
        try:
            out[name] = run(source, view)
        except Exception as exc:                      # tool bugs must not break the search
            out[name] = {'tool_error': f'{type(exc).__name__}: {str(exc)[:160]}'}
    return out


def profile_ranges(receipt: dict | None) -> dict:
    """Per-phase timing ranges; harness-provided profile (from signals) if present, else whole-update only.

    Phase maxima over ranks are not additive: the slowest rank of fwd_bwd need not be the slowest of optim.
    """
    if not receipt:
        return {'profile': None}
    if isinstance(receipt.get('profile'), dict):
        return {'source': 'harness signals', 'phases': receipt['profile'],
                'note': 'phase max-rank values are not additive; whole_update is the only accepted metric'}
    win = receipt.get('window') or [None, None]
    xs = [r['max_rank_whole_s'] for r in receipt.get('per_step') or []
          if 'max_rank_whole_s' in r and (win[0] is None or win[0] <= r['step'] <= win[1])]
    if not xs:
        return {'profile': None}
    xs_sorted = sorted(xs)
    return {'source': 'receipt per_step (no phase breakdown attached)',
            'whole_update_max_rank_s': {'min': _r(xs_sorted[0]), 'p50': _r(xs_sorted[len(xs) // 2]), 'max': _r(xs_sorted[-1]),
                                        'tail_last3_over_mean': _r(sum(xs[-3:]) / 3 / (sum(xs) / len(xs)))},
            'slowest_rank_counts': dict(Counter(r.get('slowest_rank') for r in receipt.get('per_step') or []).most_common(6))}


# ------------------------------------------------------------------ skills

def parse_skill(text: str) -> tuple[dict, list[dict]]:
    """Split a SKILL.md into (front matter, sections [{heading, text}]). Raises ValueError if malformed."""
    if not text.startswith('---'):
        raise ValueError('missing front matter')
    parts = text.split('\n---', 1)
    if len(parts) != 2:
        raise ValueError('unterminated front matter')
    front = yaml.safe_load(parts[0][3:])
    if not isinstance(front, dict):
        raise ValueError('front matter is not a mapping')
    body = parts[1].split('\n', 1)[1] if '\n' in parts[1] else ''
    sections, cur = [], {'heading': '_intro', 'lines': []}
    for line in body.splitlines():
        if line.startswith('## '):
            sections.append(cur)
            cur = {'heading': line[3:].strip(), 'lines': []}
        else:
            cur['lines'].append(line)
    sections.append(cur)
    out = [{'heading': s['heading'], 'text': '\n'.join(s['lines']).strip()} for s in sections]
    return front, [s for s in out if s['text']]


def check_skill(text: str, dirname: str | None = None) -> list[str]:
    try:
        front, sections = parse_skill(text)
    except (ValueError, yaml.YAMLError) as exc:
        return [f'unparseable: {exc}']
    errs = [f'front matter missing {k}' for k in REQUIRED_FRONT_MATTER if k not in front]
    if dirname and front.get('name') != dirname:
        errs.append(f'name {front.get("name")!r} != directory {dirname!r}')
    if not isinstance(front.get('metadata'), dict) or not front['metadata'].get('source'):
        errs.append('metadata.source missing')
    heads = [s['heading'].lower() for s in sections]
    for req in REQUIRED_SECTIONS:
        if not any(h.startswith(req) for h in heads):
            errs.append(f'section missing: {req}')
    return errs


class SkillIndex:
    def __init__(self, skills_dir: str | Path | None):
        self.sections, self.skills = [], {}
        if skills_dir is None or not Path(skills_dir).exists():
            return
        for p in sorted(Path(skills_dir).glob('*/SKILL.md')):
            front, secs = parse_skill(p.read_text(encoding='utf-8'))
            name = front.get('name', p.parent.name)
            meta = front.get('metadata') or {}
            tags = {str(t).lower() for t in meta.get('tags', [])}
            self.skills[name] = {'description': front.get('description', ''), 'when_to_use': front.get('when_to_use', ''), 'tags': sorted(tags)}
            head_words = _WORD.findall((front.get('description', '') + ' ' + front.get('when_to_use', '')).lower())
            name_words = _WORD.findall(name.replace('-', ' ').lower()) * 3
            for s in secs:
                words = _WORD.findall((s['heading'] + ' ' + s['text']).lower())
                self.sections.append({'skill': name, 'heading': s['heading'], 'text': s['text'], 'tags': tags,
                                      'tf': Counter(words + head_words + name_words), 'len': len(words) + 1})
        n = max(1, len(self.sections))
        df = Counter(w for s in self.sections for w in set(s['tf']))
        self.idf = {w: math.log(1 + n / c) for w, c in df.items()}

    def catalogue(self) -> list[dict]:
        return [{'skill': k, 'description': v['description'][:240]} for k, v in self.skills.items()]

    def retrieve(self, query: str, tags=(), *, k: int = 2, max_tokens: int = 4000, skill: str | None = None) -> list[dict]:
        if k <= 0 or not self.sections:
            return []
        q = Counter(_WORD.findall(query.lower()))
        tagset = {str(t).lower() for t in tags}
        scored = []
        for i, s in enumerate(self.sections):
            if skill and s['skill'] != skill:
                continue
            weight = SECTION_WEIGHTS.get(next((p for p in SECTION_WEIGHTS if s['heading'].lower().startswith(p)), ''), 1.0)
            bm = sum(self.idf.get(w, 0) * (s['tf'][w] * 2.2) / (s['tf'][w] + 1.2 * (0.25 + 0.75 * s['len'] / 120)) for w in q if w in s['tf'])
            score = weight * bm + 1.5 * len(tagset & s['tags'])
            if score > 0:
                scored.append((score, -i, s))
        scored.sort(key=lambda x: (-x[0], x[1]))
        out, used = [], 0
        for score, _, s in scored[:k]:
            budget = max_tokens - used
            if budget <= 50:
                break
            text = s['text']
            while estimate_tokens(text) > budget:
                text = text[: int(len(text) * 0.8)]
            used += estimate_tokens(text)
            out.append({'skill': s['skill'], 'section': s['heading'], 'score': round(score, 3), 'text': text})
        return out


def task_tags(task_raw: dict, model_facts: dict | None = None) -> list[str]:
    hw, w, m, d = task_raw['hardware'], task_raw['workload'], task_raw['model'], task_raw['data']
    tags = {f'gpus{hw["gpus"]}', f'nodes{hw["nodes"]}', str(hw.get('gpu_type', '')).lower(), m.get('family', ''),
            d.get('kind', ''), f'seq{w["seq_len"] // 1024}k', 'multi-node' if int(hw['nodes']) > 1 else 'single-node'}
    mf = model_facts or {}
    tags.add('moe' if mf.get('num_experts') else 'dense')
    if mf.get('linear_attention'):
        tags.add('hybrid-attention')
    if int(w['seq_len']) >= 32768:
        tags.add('long-context')
    return sorted(t for t in tags if t)


# ------------------------------------------------------------------ history

class RollingHistory:
    """Last `keep` turns verbatim; older turns folded into a one-line-per-turn (or LLM) summary."""

    def __init__(self, keep: int = 8):
        self.keep, self.turns, self.summary = keep, [], ''

    def add(self, turn: int, action: dict | None, feedback, note: str | None = None):
        self.turns.append({'turn': turn, 'action': action, 'feedback': feedback, 'note': note})

    @staticmethod
    def one_line(t: dict) -> str:
        a = t['action'] or {}
        kind = a.get('action', 'model_error')
        fb = t['feedback'] if isinstance(t['feedback'], dict) else {'msg': str(t['feedback'])[:160]}
        bits = [f't{t["turn"]} {kind}']
        for k in ('candidate_id', 'base', 'template'):
            if a.get(k):
                bits.append(f'{k}={a[k]}')
        for k in ('candidate_id', 'error', 'valid', 'step_time', 'verdict', 'status'):
            if k in fb and fb[k] not in (None, ''):
                bits.append(f'-> {k}={str(fb[k])[:100]}')
        return ' '.join(bits)

    def fold(self, summarizer=None):
        old = self.turns[:-self.keep] if self.keep else self.turns
        if not old:
            return
        self.turns = self.turns[len(old):]
        lines = [self.one_line(t) for t in old]
        text = summarizer(self.summary, lines) if summarizer else None
        self.summary = text if text else '\n'.join(filter(None, [self.summary] + lines))

    def render(self, max_tokens: int) -> str:
        parts = []
        if self.summary:
            parts.append('Earlier turns (summary):\n' + self.summary)
        for t in self.turns:
            parts.append(f'turn {t["turn"]}: {_dumps(t["action"])}\n  -> {_dumps(t["feedback"])[:6000]}')
        text = '\n'.join(parts)
        while estimate_tokens(text) > max_tokens and len(parts) > 1:
            parts.pop(1 if self.summary else 0)
            text = '\n'.join(parts)
        return text
