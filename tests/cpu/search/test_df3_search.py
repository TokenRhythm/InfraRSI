"""Trial D1-10 (2026-10-24): `attn=flash_attention_2` failed on three bases with the same ImportError (flash_attn not
installed): 3 of 10 screens. A one-knob change whose run could not import a package is not proposed again; the
change-class table's repeat floor includes the promoted candidate's own fresh run (compiled runs repeat to 1e-6)."""
from types import SimpleNamespace

from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.ext.common.session_hooks import SessionHooks
from infrarsi.harness import signing
from infrarsi.project.flow import own_repeat
from infrarsi.project.report import _numerics
from infrarsi.search.engine import Proposal
from infrarsi.search.proposers import LLMProposer, patch_of
from test_search import TOGGLE, setup

IMPORT = 'ImportError: FlashAttention2 has been toggled on, but ... the package flash_attn seems to be not installed.'


def _failing_with(monkeypatch, s, line):
    run = s.runner.run

    def failing(*a, **kw):
        r = run(*a, **kw)
        if not r['valid']:
            r['evidence'].setdefault('events', {})['error'] = line
            signing.sign(r)
        return r
    monkeypatch.setattr(s.runner, 'run', failing)


def test_a_change_that_needs_a_missing_package_is_not_tried_on_another_base(monkeypatch):
    sim, s, eng = setup(TOGGLE)
    _failing_with(monkeypatch, s, IMPORT)
    eng.evaluate(Proposal(change={'tp_overlap': True}, base='c0'))      # crashes in the toy; the log says why
    good = eng.evaluate(Proposal(change={'fused_adam': True}, base='c0'))
    again = eng.evaluate(Proposal(change={'tp_overlap': True}, base=good.cid))
    assert again.status == 'rejected' and s.invalid.get('unrunnable') == 1
    # the same change with other code (a patch: another overlay) may no longer need the package: not refused
    fb = s.propose_config(changes={'tp_overlap': True}, base=good.cid)   # the engine made it, did not run it
    cid = fb.get('candidate_id') or fb['duplicate_of']
    assert 'flash_attn' in s.unrunnable_hit(cid)
    s.candidates[cid].overlay_digest = 'patched'
    assert s.unrunnable_hit(cid) is None


def test_a_plain_import_error_marks_nothing(monkeypatch):
    # "No module named ..." may depend on other knobs (a backend that imports another package): feedback only
    sim, s, eng = setup(TOGGLE)
    _failing_with(monkeypatch, s, "ModuleNotFoundError: No module named 'transformer_engine'")
    eng.evaluate(Proposal(change={'tp_overlap': True}, base='c0'))
    assert s.unrunnable == {}


def test_a_two_knob_failure_marks_neither_knob(monkeypatch):
    sim, s, eng = setup(TOGGLE)
    _failing_with(monkeypatch, s, IMPORT)
    eng.evaluate(Proposal(change={'tp_overlap': True, 'compile': True}, base='c0'))
    assert s.unrunnable == {} and 'candidate_id' in s.propose_config(changes={'compile': True}, base='c0')


def test_the_repeat_floor_includes_the_promoted_candidates_own_run():
    r = {'tolerance': 0.001, 'search': {'doctor_repeat_loss_deviation': 0.0, 'ledger': [
        {'mechanism': 'workers', 'change_class': 'execution', 'loss_deviation_pct': 0.0142, 'loss_deviation_base_pct': 0.0001}]},
         'promotion': {'tried': [{'reference_repeat_loss_deviation': 0.0, 'candidate_repeat_loss_deviation': 2e-6}]}}
    text = '\n'.join(_numerics(r))
    assert 'Repeat floor: 0.0002 %, the largest of 3 repeat(s)' in text and '| 0.0001 % | within |' in text
    judge = SimpleNamespace(screen_checks=lambda a, b: [{'id': 'loss_within_tolerance', 'evidence': {'deviation': 2e-6}}])
    assert own_repeat(judge, SimpleNamespace(valid=True), {'evidence': {}}, {'evidence': {}}) == 2e-6
    assert own_repeat(judge, SimpleNamespace(valid=False), {'evidence': {}}, {'evidence': {}}) is None


def test_an_empty_string_patch_is_no_patch():
    # D1-20 (Sonnet): "patch": "" next to a knob change failed the schema and the whole answer was dropped
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    hyp = {'hypotheses': [{'mechanism': 'fused', 'change': {'fused_adam': True}, 'patch': '', 'edits': [],
                           'predicted_gain_pct': 3}]}
    drv = ScriptedBackend(lambda role, system, prompt: hyp if role == 'hypotheses' else {'verdict': 'no_patch', 'findings': []})
    hyps = LLMProposer(drv, SessionHooks(s.task)).hypotheses(eng, 'c0')
    assert [h['change'] for h in hyps] == [{'fused_adam': True}]
    assert patch_of({'patch': ''}, {}) == (None, True) and patch_of({'patch': '--- a/x.py\n+++ b/x.py'}, {}) == (None, False)
