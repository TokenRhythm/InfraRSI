"""Hypotheses within the output limit when a mutable file is long (FABLE_DF16_DECISION B, F-D2-R204d-3): D2-R204d (Wan,
Qwen3-30B, thinking off) wrote the 230-line train_wan.py (3,600 tokens) out in full as the `patch` of five hypotheses in
one reply - 67 KB, cut at the 16k-token limit, the same again on the doubled-cap retry, three calls and the round's
proposals lost. Now the code view names the mutable files whose text takes more than an eighth of the answer cap
(`edits_only`), the patch rules say a whole-file `patch` is for the others, and a reply that is still cut gets one
smaller ask: every file by edits, half the hypotheses, the cut named. A short file's whole-file patch keeps working."""
import json

from infrarsi.agent.backends.base import LLMResult
from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.ext.common.session_hooks import SessionHooks
from infrarsi.search.proposers import (CODE_RULE, FULL_TEXT_SHARE, PATCH_RULES, REVIEW_SYSTEM, LLMProposer, _tokens,
                                         code_view, cut_at_limit, written_in_full)

from test_search import TOGGLE, setup

BIG = ''.join(f'def layer_{i}(x):\n    return torch.nn.functional.gelu(x) * {i} + x.sum(dim=-1, keepdim=True)\n\n'
              for i in range(140))                                     # ~11 KB, ~3,300 tokens: train_wan.py's size
SMALL = 'x = 1\ntorch.cuda.synchronize()\ny = 2\n'
FILES = {'big.py': BIG, 'small.py': SMALL}
EDIT = {'file': 'big.py', 'old': 'def layer_7(x):\n', 'new': 'def layer_7(x):  # fused\n'}
REVIEW = {'verdict': 'patch', 'summary': 'two costs', 'findings': []}
CUT_ERROR = ('the reply was cut at the output limit (16384 tokens) before its JSON ended; no larger limit is left '
             '(llm.max_tokens 16384, the token or USD allowance): a reasoning model\'s thinking counts: raise llm.max_tokens '
             'or limit its reasoning (llm.extra_body)')


def cut_text(path='big.py'):
    """A reply cut mid-file while writing `path` out in full as a patch (what D2-R204d's answer_0.txt holds)."""
    return ('{\n  "hypotheses": [\n    {\n      "mechanism": "execution:optimize_data_loading",\n      "change": {},\n'
            '      "patch": {\n        "' + path + '": "' + json.dumps(FILES[path])[1:-1][:4000])


class Driver:
    """A backend whose replies are scripted: a dict is an answer, 'CUT' a reply cut at the output limit as openai_compat
    reports one (the attempt rows carry `cut`, the error its words)."""
    kind, model, call_overhead_tokens, max_output_tokens = 'fake', 'fake', 0, 16000

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def describe(self):
        return {'kind': self.kind, 'model': self.model}

    def complete(self, system, prompt, *, schema=None, role='proposer', **kw):
        self.calls.append({'role': role, 'system': system, 'prompt': json.loads(prompt), 'kw': kw})
        item = self.replies.pop(0)
        if item == 'CUT':
            return LLMResult(action=None, text=cut_text(), llm_tokens=26000, llm_usd=0.0, wall_seconds=100.0,
                             accounting_complete=True, error=CUT_ERROR, model='fake', observed_models=['fake'], attempts=2,
                             meta={'attempts': [{'attempt': 0, 'cut': True}, {'attempt': 1, 'cut': True}]})
        return LLMResult(action=item, text=json.dumps(item), llm_tokens=1000, llm_usd=0.0, wall_seconds=1.0,
                         accounting_complete=True, error=None, model='fake', observed_models=['fake'], meta={})


def proposer(drv, **kw):
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    logs = []
    eng.log = logs.append
    return LLMProposer(drv, SessionHooks(s.task, base_files=dict(FILES)), **kw), eng, logs


MID = 'y = 1  # padding\n' * 330                                  # ~1,400 tokens: under 16,000 / 8, over 8,000 / 8


def _shares_hold(drv, files):
    for call in drv.calls:                                            # every call: the share of its own answer cap
        cap, eo = call['kw']['max_output_tokens'], set(call['prompt']['code'].get('edits_only') or [])
        assert all(p in eo for p, t in files.items() if _tokens(t) > cap // FULL_TEXT_SHARE), (call['role'], cap, eo)


def test_the_share_is_of_the_call_s_own_answer_cap():
    # DF16_r1 #2: with an 8,000-token backend (or few tokens left) a 1,400-token file is edits-only in the review and the
    # hypothesis prompts, though an eighth of the nominal 16,000-token code cap would keep it whole
    files = {'mid.py': MID, 'small.py': SMALL}
    assert 1000 < _tokens(MID) <= 2000

    class Small(Driver):
        max_output_tokens = 8000
    for drv, tokens in ((Small([REVIEW, {'hypotheses': []}]), 10 ** 6), (Driver([REVIEW, {'hypotheses': []}]), 12000)):
        sim, s, eng = setup(TOGGLE, tokens=tokens)
        p = LLMProposer(drv, SessionHooks(s.task, base_files=dict(files)))
        p.hypotheses(eng, 'c0')
        assert [c['role'] for c in drv.calls] == ['code_review', 'hypotheses']
        assert all('mid.py' in c['prompt']['code']['edits_only'] for c in drv.calls)
        assert all('small.py' not in c['prompt']['code']['edits_only'] for c in drv.calls)
        _shares_hold(drv, files)


def _text_of(tokens):
    """A file whose text takes `tokens` tokens as it travels in the payload."""
    t = 'x' * int(tokens * 3.5)
    while _tokens(t) > tokens:
        t = t[:-1]
    while _tokens(t) < tokens:
        t += 'x'
    return t


def test_the_share_holds_after_the_names_lengthen_the_prompt():
    # DF16_r2: naming files edits-only lengthens the prompt, the allowance shrinks again and a file under the first
    # share crosses the second: the calls tighten until nothing more is added (24 files of 301 tokens, one of 241, a
    # 14,507-token budget: the first share 241, the last one lower)
    files = {f'training/model/part_{i:02d}.py': _text_of(301) for i in range(24)}
    files['training/model/utils.py'] = _text_of(241)
    drv = Driver([REVIEW, {'hypotheses': []}])
    sim, s, eng = setup(TOGGLE, tokens=14507)
    LLMProposer(drv, SessionHooks(s.task, base_files=dict(files))).hypotheses(eng, 'c0')
    assert drv.calls and all('code' in c['prompt'] for c in drv.calls)
    _shares_hold(drv, files)


def test_the_code_view_names_the_files_to_patch_by_edits_only():
    class Hooks:
        read_only_files = {}

        def candidate_files(self, overlay):
            return dict(FILES)

    v = code_view(Hooks(), None, budget_tokens=12000, full_text_tokens=16000 // FULL_TEXT_SHARE)
    assert v['edits_only'] == ['big.py'] and v['mutable'] == FILES and 'excerpted' not in v    # shown whole, patched by edits
    assert _tokens(BIG) > 2000 >= _tokens(SMALL)
    assert 'edits_only' not in code_view(Hooks(), None, budget_tokens=12000)                   # the share is the caller's
    for rule in (PATCH_RULES, CODE_RULE, REVIEW_SYSTEM):
        assert 'NOT listed in `code.edits_only`' in rule and 'change them with `edits`' in rule
    assert written_in_full(cut_text(), FILES) == ['big.py'] and written_in_full('{"hypotheses": []}', FILES) == []
    cut = LLMResult(action=None, text='', llm_tokens=1, llm_usd=0, wall_seconds=1, accounting_complete=True, error=CUT_ERROR,
                    model='m', observed_models=[], meta={})
    assert cut_at_limit(cut)
    assert cut_at_limit(LLMResult(action=None, text='', llm_tokens=1, llm_usd=0, wall_seconds=1, accounting_complete=True,
                                  error='invalid structured JSON: the reply hit max_tokens', model='m', observed_models=[]))
    assert not cut_at_limit(LLMResult(action=None, text='', llm_tokens=1, llm_usd=0, wall_seconds=1, accounting_complete=True,
                                      error='schema: hypotheses is required', model='m', observed_models=[]))


def test_a_cut_reply_gets_one_smaller_ask_with_edits_only_and_the_answer_counts():
    answer = {'hypotheses': [
        {'mechanism': 'fused gelu', 'change': {}, 'edits': [EDIT], 'predicted_gain_pct': 4, 'change_class': 'execution'},
        {'mechanism': 'no sync', 'change': {}, 'patch': {'small.py': 'x = 1\ny = 2\n'}, 'predicted_gain_pct': 2},
        {'mechanism': 'fused adam', 'change': {'fused_adam': True}, 'predicted_gain_pct': 3}]}
    drv = Driver([REVIEW, 'CUT', answer])
    prop, eng, logs = proposer(drv)
    hyps = prop.hypotheses(eng, 'c0')
    assert [c['role'] for c in drv.calls] == ['code_review', 'hypotheses', 'hypotheses']
    first, again = drv.calls[1]['prompt'], drv.calls[2]['prompt']
    assert first['code']['edits_only'] == ['big.py'] and first['max_hypotheses'] == 12 and 'cut_reply' not in first
    assert drv.calls[0]['prompt']['code']['edits_only'] == ['big.py']                    # the review sees the same view
    assert 'edits_only' in drv.calls[1]['system']
    assert again['code']['edits_only'] == ['big.py', 'small.py'] and again['max_hypotheses'] == 6
    assert again['cut_reply'].startswith('your previous reply was cut at the output limit')
    assert 'it wrote big.py out in full' in again['cut_reply'] and 'at most 6 hypotheses' in again['cut_reply']
    assert again['code']['mutable'] == FILES and again['incumbent'] == first['incumbent']
    assert [h['mechanism'] for h in hyps] == ['fused gelu', 'no sync', 'fused adam']
    assert hyps[0]['patch'] == {'big.py': BIG.replace(EDIT['old'], EDIT['new'])}            # the long file by its edit
    assert hyps[1]['patch'] == {'small.py': 'x = 1\ny = 2\n'} and hyps[2]['patch'] is None   # the short one whole, as before
    assert prop.failed == 0 and prop.calls == 1 and prop.unusable_edits == 0
    assert eng.s.ledger.used['llm_tokens'] == 1000 + 26000 + 1000                        # every call charged
    cut_lines = [m for m in logs if m.startswith('[search] hypothesis call: the reply was cut')]
    assert cut_lines == ['[search] hypothesis call: the reply was cut at the output limit (it wrote big.py out in full as a '
                         'patch); asking once more for at most 6 hypotheses with edits only']
    assert not any('failed' in m for m in logs)


def test_a_second_cut_ends_the_call_and_is_counted_once():
    drv = Driver([REVIEW, 'CUT', 'CUT'])
    prop, eng, logs = proposer(drv)
    assert prop.hypotheses(eng, 'c0') == []
    assert [c['role'] for c in drv.calls] == ['code_review', 'hypotheses', 'hypotheses'] and prop.failed == 1
    assert sum(m.startswith('[search] hypothesis call: the reply was cut') for m in logs) == 1
    assert any(m.startswith('[search] hypothesis call failed: the reply was cut at the output limit') for m in logs)
    assert eng.s.ledger.used['llm_tokens'] == 1000 + 2 * 26000 and not drv.replies     # both cut attempts charged, no third


def test_without_code_in_the_prompt_a_cut_reply_is_just_a_failed_call():
    drv = Driver(['CUT', {'hypotheses': []}])
    prop, eng, logs = proposer(drv, menu=True)                           # menu mode shows no code: no patch can be the cause
    assert prop.hypotheses(eng, 'c0') == [] and [c['role'] for c in drv.calls] == ['hypotheses'] and prop.failed == 1
    assert not any('asking once more' in m for m in logs)
    # a reply cut for another reason (a schema error, not the output limit) with code shown: no re-ask either
    bad = ScriptedBackend(lambda role, system, prompt: REVIEW if role == 'code_review' else 'not json')
    prop, eng, logs = proposer(bad)
    assert prop.hypotheses(eng, 'c0') == [] and [c['role'] for c in bad.calls] == ['code_review', 'hypotheses']
