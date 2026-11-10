"""The dual-model debate (off by default; the manifest's `debate:` block, or `--second-llm` / `--debate`).

At the calls that decide what to try next (DEBATE_ROLES: the prior over the knobs, the hypotheses, the configurations, the
code review, the revision of the prior, the transfer judgment), the main model answers as it always does and the second
model answers the same request without seeing that answer. The main model then judges the two answers, anonymised
(blind): they are labelled A and B at random, in a canonical form, with the two models' names taken out of their free
text. Which model wrote which stays in `debates.jsonl` of the run and never reaches a prompt.

The strategy sees one reply, from one `complete` call: the answer the judge chose, or the items it selected from the two;
the tokens and dollars of all three calls are on it and charge the run's budget. Nothing else changes: a failed main
answer comes back as it did (its retries are the proposer's), a failed second answer or a failed judgement leaves the main
model's answer in place, and the verifier of every run still decides what is valid.

Code patches (DebateDriver.checker): the search's patches are checked by a verifier model before they cost a run, at most
`rounds` checks with one revision between them (search/patch_check.py).
"""
from __future__ import annotations

import dataclasses
import json
import random
import re
from pathlib import Path

from .backends.base import LLMResult

# role -> the list of the reply the judge selects items from; None: it picks one answer whole (a table, a verdict)
DEBATE_ROLES = {'hypotheses': 'hypotheses', 'configs': 'configs', 'prior': None, 'revise': None, 'code_review': None,
                'transfer': None, 'proposer': None}
AUTHOR_FIELDS = {'hypotheses': 'hypotheses', 'code_review': 'findings'}   # the lists whose items may carry a patch
# what the names of the models are NOT taken out of: code and knob values (the proposer's and the agent's), which a job's own
# model name may be part of; everything around them is text
KEEP_AS_WRITTEN = ('change', 'patch', 'knob', 'value', 'values', 'file', 'path', 'old', 'new', 'files', 'content', 'config', 'changes',
                   'params', 'template', 'start_config', 'gib_expr', 'when', 'tags', 'slow', 'fast')
EDIT_BODY = ('text', 'code')                            # an edit's text in the forms the proposer also accepts: only in an edit
MIN_CALL_USD = 0.05                                     # (agent/optimize.py: no model call is started below this)
AUTHOR_KEY = '_author'                                  # on an item of the reply: which model wrote it, for the patch check
_COMMON_WORDS = {'chat', 'mini', 'turbo', 'flash', 'latest', 'large', 'small', 'default', 'preview', 'high', 'code', 'reasoner',
                 'instruct', 'base', 'free'}
_VENDOR_WORDS = {'claude': ('claude', 'anthropic'), 'anthropic': ('claude', 'anthropic'), 'codex': ('codex', 'openai', 'chatgpt', 'gpt'),
                 'gpt': ('openai', 'chatgpt', 'gpt'), 'openai': ('openai', 'chatgpt', 'gpt'), 'deepseek': ('deepseek',),
                 'qwen': ('qwen', 'alibaba', 'tongyi'), 'gemini': ('gemini', 'google'), 'grok': ('grok', 'xai'), 'llama': ('llama', 'meta')}
ADJUDICATE_SCHEMA = {'type': 'object', 'required': ['winner'], 'properties': {
    'analysis': {'type': ['string', 'null']}, 'winner': {'type': 'string', 'enum': ['A', 'B']},
    'selected': {'type': ['array', 'null'], 'items': {'type': 'string'}}}}
JUDGE_RULES = """You judge between two independent proposals for the next step of a search that makes one training job faster
without changing what it computes. Both answer the same request, shown under `request`. You do not know who wrote either
one; they are labelled at random, and neither label says anything about quality. Judge only their technical merit:
- would the change shorten the time per update, given what the request says was measured (the receipts, the phases, the
  history, the failures)?
- is it feasible and is it correct: the same workload and data, the same optimizer steps, the loss within the tolerance, no
  frozen file touched?
- what would running it teach the search next?
Prefer a smaller well-grounded change to a larger speculative one; do not prefer a proposal for its length or its stated
confidence. Reply with exactly one JSON object and nothing else.
"""
JUDGE_ITEMS = """Proposal items are labelled A1, A2, ... and B1, B2, ...; order the ones worth running, the best first, in
`selected` (from one proposal or both; leave out the ones not worth a run). {"analysis": "<your comparison, at most 200
words>", "winner": "A" or "B" (the proposal you find stronger as a whole), "selected": ["A2", "B1", ...]}.
"""
JUDGE_WHOLE = """Pick the proposal to follow as a whole. {"analysis": "<your comparison, at most 200 words>", "winner": "A" or
"B"}.
"""


def identity_words(*drivers) -> list[str]:
    """The words that would name one of `drivers`: its model's name and the parts of it, and its vendor's."""
    words: set[str] = set()
    for d in drivers:
        if d is None:
            continue
        name = str(getattr(d, 'model', '') or '').lower()
        kind = str(getattr(d, 'kind', '') or '').lower()
        if name and name != 'default':
            words.add(name)
            words.update(p for p in re.split(r'[-_/:.\s]+', name)
                         if (len(p) >= 4 and p.isalpha() and p not in _COMMON_WORDS) or p == 'gpt')
        for hint, names in _VENDOR_WORDS.items():
            if hint in name or hint in kind:
                words.update(names)
    return sorted(words, key=len, reverse=True)


def scrub(value, pattern):
    return pattern.sub('[model]', value) if pattern is not None and isinstance(value, str) else value


def blinded(reply, pattern):
    """The reply in canonical form for the judge: keys sorted, the models' names taken out of every string but code and knob
    values (a job's own model name may legitimately be part of those). A null is kept: in a configuration it says something."""
    if isinstance(reply, dict):
        edit = isinstance(reply.get('file'), str)
        return {k: (v if (k in KEEP_AS_WRITTEN or (edit and k in EDIT_BODY)) and not _edit_object(k, v) else blinded(v, pattern))
                for k, v in sorted(reply.items()) if k != AUTHOR_KEY}
    if isinstance(reply, list):
        return [blinded(x, pattern) for x in reply]
    return scrub(reply, pattern)


def _edit_object(key: str, value) -> bool:
    """A `patch` written as one edit ({file, old, new, ...}): its fields go through as an edit's do, unlike a {path: text} map."""
    return key == 'patch' and isinstance(value, dict) and isinstance(value.get('file'), str)


def canonical(reply) -> str:
    return json.dumps(blinded(reply, None), ensure_ascii=False, sort_keys=True, default=str)


class DebateDriver:
    """The main model's backend with a second model and a blind judgement at the proposal calls (module docstring); every
    other call is the main model's, and every attribute the backend has reads and writes through to it. Its state - the
    labels' generator, the counters - is rebuilt by the run that holds it (`attach`), so a checkpoint does not carry it."""

    def __init__(self, main, second, *, verifier=None, rounds: int = 2, seed: int = 42, log=None):
        if rounds not in (0, 1, 2):
            raise ValueError(f'the patch check is limited to two rounds, got {rounds!r}')
        self.__dict__.update(_inner=main, second=second, verifier=verifier, rounds=int(rounds), seq=0, path=None,
                             _rng=random.Random(seed), log=log or (lambda m: None), stats=_new_stats(),
                             _names=_identity_pattern(main, second, verifier))

    def __getattr__(self, name):
        return getattr(self.__dict__['_inner'], name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            setattr(self.__dict__['_inner'], name, value)

    def describe(self) -> dict:
        d = getattr(self._inner, 'describe', lambda: {})() or {}
        return {**d, 'debate': {'second': model_name(self.second), 'verifier': model_name(self.verifier) if self.verifier
                                else 'the other model' if self.rounds else None, 'rounds': self.rounds}}

    def models(self) -> list:
        """The models besides the main one, for the one small test call before any GPU time."""
        return [m for m in (self.second, self.verifier) if m is not None]

    def attach(self, run_dir, log=None) -> None:
        """Where this run keeps its calls and the debate log; the counters go on from an earlier part of a resumed run."""
        run_dir = Path(run_dir)
        for name, model in (('second', self.second), ('verifier', self.verifier)):
            if model is not None and getattr(model, 'out_root', False) is None:
                model.out_root = run_dir / 'llm_calls' / name
        self.__dict__['path'] = run_dir / 'debates.jsonl'
        self.__dict__['log'] = log or self.__dict__['log']
        if self.path.is_file():
            if not self.path.read_bytes().endswith(b'\n'):  # (what a killed process left: the next row starts on its own line)
                with self.path.open('ab') as f:
                    f.write(b'\n')
            rows = logged(self.path)
            self.__dict__['seq'] = sum(1 for r in rows if r.get('event') == 'debate')
            self.__dict__['stats'] = summarize(rows)

    # ------------------------------------------------------------ the proposal calls
    def complete(self, system, prompt, *, schema=None, call_dir=None, role='proposer', **kw):
        if role not in DEBATE_ROLES:
            return self._inner.complete(system, prompt, schema=schema, call_dir=call_dir, role=role, **kw)
        first = self._inner.complete(system, prompt, schema=schema, call_dir=call_dir, role=role, **kw)
        if not isinstance(first.action, dict):           # as without a debate: the proposer's own retries deal with it
            return first
        rec = {'event': 'debate', 'seq': self.seq, 'role': role, 'main_model': model_name(self._inner)}
        self.__dict__['seq'] = self.seq + 1
        left = _admitted(self.second, system + prompt, _left(kw, [first]))
        if left is None:                                 # nothing left for a second opinion: the main answer stands
            return self._done(rec, first, first.action, [first], 'no_budget')
        rival = self.second.complete(system, prompt, schema=schema, role=role, **left)
        if not isinstance(rival.action, dict):
            rec['second_error'] = str(rival.error)[:300]
            self.log(f'[search] debate: the second model gave no usable answer ({str(rival.error)[:160]}); the main model\'s stands')
            return self._done(rec, first, first.action, [first, rival], 'second_failed')
        rec['second_model'] = model_name(self.second)
        if canonical(first.action) == canonical(rival.action):
            return self._done(rec, first, _stamped(first.action, AUTHOR_FIELDS.get(role), 'main'), [first, rival], 'same_answer')
        return self._judge(system, prompt, role, kw, first, rival, rec)

    def _judge(self, system, prompt, role, kw, first, rival, rec) -> LLMResult:
        """The main model judges the two answers blind; returns the reply the search gets."""
        main_is_a = self._rng.random() < 0.5
        replies = {('A' if main_is_a else 'B'): (first.action, 'main'), ('B' if main_is_a else 'A'): (rival.action, 'second')}
        field = DEBATE_ROLES[role]
        view = {'request': {'instructions': system, 'input': prompt}}      # (the same for both: it names neither author)
        for label in 'AB':
            view[f'proposal_{label}'] = self._view(replies[label][0], field, label)
        rules = JUDGE_RULES + (JUDGE_ITEMS if field else JUDGE_WHOLE)
        body = json.dumps(view, ensure_ascii=False, sort_keys=True, default=str)
        left = _admitted(self._inner, rules + body, _left(kw, [first, rival]), judge=True)
        verdict = None
        if left is not None:
            verdict = self._inner.complete(rules, body, schema=ADJUDICATE_SCHEMA, role='debate_judge', **left)
        used = [first, rival] + ([verdict] if verdict is not None else [])
        pick = None if verdict is None or not isinstance(verdict.action, dict) else verdict.action.get('winner')
        rec.update(labels={'A': replies['A'][1], 'B': replies['B'][1]})
        if pick not in ('A', 'B'):                       # no judgement: the main model's answer stands
            rec['judge_error'] = str(getattr(verdict, 'error', None) or 'not sent: no budget left')[:300]
            self.log(f'[search] debate: no usable judgement ({rec["judge_error"][:160]}); the main model\'s answer stands')
            return self._done(rec, first, first.action, used, 'judge_failed')
        reply, picked = self._selected(replies, role, field, pick, verdict.action.get('selected'))
        rec.update(winner=replies[pick][1], analysis=str(verdict.action.get('analysis') or '')[:1500], picked=picked)
        return self._done(rec, first, reply, used, 'judged')

    def _view(self, reply: dict, field, label: str) -> dict:
        out = blinded(reply, self._names)
        if field and isinstance(out.get(field), list):
            out[field] = [{**x, 'id': f'{label}{i + 1}'} if isinstance(x, dict) else {'id': f'{label}{i + 1}', 'value': x}
                          for i, x in enumerate(out[field])]
        return out

    def _selected(self, replies: dict, role: str, field, pick: str, selected) -> tuple[dict, list]:
        """The reply the search gets and the ids the judge's choice used: the items selected (from one proposal or both, in
        the judge's order), around them the winner's other fields; the winner whole without a usable list."""
        winner, author = replies[pick]
        whole = _stamped(winner, AUTHOR_FIELDS.get(role), author), []
        if not field or not isinstance(selected, list):
            return whole
        if selected == []:                               # the judge found none worth a run: none is run
            return {**winner, field: []}, []
        items, used = [], []
        for sid in selected:
            m = re.fullmatch(r'\s*([AB])(\d+)\s*', str(sid))
            if not m or f'{m[1]}{int(m[2])}' in used:
                continue
            rows = replies[m[1]][0].get(field) or []
            i = int(m[2]) - 1
            if 0 <= i < len(rows) and isinstance(rows[i], dict):
                items.append({**rows[i], AUTHOR_KEY: replies[m[1]][1]})
                used.append(f'{m[1]}{i + 1}')
        return ({**winner, field: items}, used) if items else whole

    def _done(self, rec: dict, first: LLMResult, action: dict, used: list, how: str) -> LLMResult:
        rec['outcome'] = how
        rec['tokens'] = sum(r.llm_tokens for r in used if type(r.llm_tokens) is int)
        self._write(rec)
        s = self.stats
        s['debates'] += 1
        s['outcomes'][how] = s['outcomes'].get(how, 0) + 1
        if rec.get('winner'):
            s['won'][rec['winner']] = s['won'].get(rec['winner'], 0) + 1
        if how == 'judged':
            self.log(f'[search] debate ({rec["role"]}): the judge preferred the {rec["winner"]} model\'s proposal')
        tokens = [r.llm_tokens for r in used]
        usd = [r.llm_usd for r in used]
        return dataclasses.replace(
            first, action=action, text=json.dumps(action, ensure_ascii=False, default=str),
            llm_tokens=sum(t for t in tokens if type(t) is int) if any(type(t) is int for t in tokens) else None,
            llm_usd=sum(u for u in usd if type(u) in (int, float)) if any(type(u) in (int, float) for u in usd) else None,
            wall_seconds=sum(r.wall_seconds or 0.0 for r in used),
            accounting_complete=all(r.accounting_complete and type(r.llm_tokens) is int and type(r.llm_usd) in (int, float)
                                    for r in used),
            observed_models=sorted({m for r in used for m in r.observed_models or ()}),
            meta={**(first.meta or {}), 'debate': how, 'author': rec.get('winner', 'main')})

    def _write(self, rec: dict) -> None:
        if self.path is None:
            return
        try:
            with self.path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
        except OSError:
            pass                                         # (a record of the debate never stops the search)

    # ------------------------------------------------------------ the code patches (search/patch_check.py)
    def checker(self, author):
        """(the model that checks a patch of `author`, the model that revises it): an explicit verifier, else the model that
        did not write it; the author revises its own patch."""
        check = self.verifier or (self._inner if author == 'second' else self.second)
        return check, (self.second if author == 'second' else self._inner)

    def record_check(self, rec: dict) -> None:
        self._write({'event': 'patch_check', **rec})
        s = self.stats['patch_checks']
        s[rec['outcome']] = s.get(rec['outcome'], 0) + 1

    def summary(self) -> dict:
        return self.stats


def _new_stats() -> dict:
    return {'debates': 0, 'outcomes': {}, 'won': {}, 'patch_checks': {}}


def logged(path) -> list[dict]:
    """The rows of a `debates.jsonl` (a line a killed process did not finish is left out)."""
    rows = []
    for line in Path(path).read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def summarize(rows: list) -> dict:
    """The counters of a `debates.jsonl`."""
    s = _new_stats()
    for r in rows:
        if r.get('event') == 'debate':
            s['debates'] += 1
            s['outcomes'][r.get('outcome')] = s['outcomes'].get(r.get('outcome'), 0) + 1
            if r.get('winner'):
                s['won'][r['winner']] = s['won'].get(r['winner'], 0) + 1
        elif r.get('event') == 'patch_check':
            s['patch_checks'][r.get('outcome')] = s['patch_checks'].get(r.get('outcome'), 0) + 1
    return s


def model_name(d) -> str | None:
    """The name to show for a model: its backend's when the model is 'default' (a CLI left on its own default)."""
    if d is None:
        return None
    name = getattr(d, 'model', None)
    return str(name if name and name != 'default' else getattr(d, 'kind', None) or 'model')


def _identity_pattern(*drivers):
    words = identity_words(*drivers)
    return re.compile('|'.join(rf'\b{re.escape(w)}\b' for w in words), re.I) if words else None


def _admitted(driver, text: str, kw: dict | None, *, judge: bool = False) -> dict | None:
    """`kw` for a call of `driver` with input `text`, its answer capped by the tokens left after that input and the call's
    overhead (what agent/optimize.output_allowance admits for any call); None when the call does not fit."""
    if kw is None:
        return None
    out = dict(kw)
    if judge:                                            # (the answer's own retry rules: not the judgement's)
        for key in ('retry_note', 'join_text'):
            out.pop(key, None)
        out['max_output_tokens'] = min(int(out.get('max_output_tokens') or 2048), 2048)
    if out.get('token_allowance') is not None:
        from .optimize import DEFAULT_CALL_OVERHEAD_TOKENS, MIN_OUTPUT_TOKENS, input_reserve
        overhead = int(getattr(driver, 'call_overhead_tokens', DEFAULT_CALL_OVERHEAD_TOKENS))
        room = int(out['token_allowance']) - input_reserve(text) - overhead
        if room < MIN_OUTPUT_TOKENS:
            return None
        if out.get('max_output_tokens') is not None:
            out['max_output_tokens'] = min(int(out['max_output_tokens']), room)
    return out


def _left(kw: dict, spent: list) -> dict | None:
    """The allowances of the call after `spent`: what the call was given less what they cost; None when a call cannot be
    made any more (the dollars or the wall clock are gone)."""
    out = dict(kw)
    if out.get('token_allowance') is not None:
        out['token_allowance'] = int(out['token_allowance']) - sum(r.llm_tokens or 0 for r in spent)
    if out.get('max_usd') is not None:
        out['max_usd'] = float(out['max_usd']) - sum(r.llm_usd or 0.0 for r in spent)
        if out['max_usd'] < MIN_CALL_USD:
            return None
    if out.get('timeout') is not None:
        out['timeout'] = float(out['timeout']) - sum(r.wall_seconds or 0.0 for r in spent)
        if out['timeout'] <= 1.0:
            return None
    return out


def _stamped(reply: dict, field, author: str) -> dict:
    """The reply whole, each item of its list that may carry a patch marked with its author (for the patch check)."""
    if not field or not isinstance(reply.get(field), list):
        return reply
    return {**reply, field: [{**x, AUTHOR_KEY: author} if isinstance(x, dict) else x for x in reply[field]]}
