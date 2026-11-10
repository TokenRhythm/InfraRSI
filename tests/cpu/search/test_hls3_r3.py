"""FABLE_HLS3_FIXES_R3 (reports/infrarsi_newmodels_20261028/design/FABLE_HLS3_FIXES_R3.md): a unit test per fix, each
component switched off by its ablate name gives hls3 as it was (round 2: pc_code_hls4). F13 a patch's proof lines read
whole; F14 the hypothesis answer read hypothesis by hypothesis; F15 an experience seed's spender mu; F16 the pair probes of
a memory-blocked spender; F17 coverage members resolved by their bundle; F18 a patch carried over a knob-only incumbent
change; F19 the pair set of the judgment's start composite; F21 the `deferred` proof line; F23 thrash read from scattered
updates. And the shared pieces, inert for every other strategy: the scanner's whole-line evidence, the detailed trial,
the CLI's joined text blocks and retry note, the carry switch, the toy's markers and its undeclared-enabler family."""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from infrarsi.project.signals import proof_evidence, scan_events
from infrarsi.search import HLS3_COMPONENTS, HLS3_R1, HLS3_R2, HLS3_R3
from infrarsi.search import proposers as P
from infrarsi.search.engine import Outcome, Proposal
from infrarsi.search.hls3 import HLS3, JITTER_MIN, PAIR_SET, PAIR_SET_MAX, proof_verdict, proposer_flags, scatter
from infrarsi.search.strategies import HLS, HLS2
from test_composition import FakeProposer
from test_hls3 import G4U_OOM, KNOBS, START, engine, h, observe, run
from test_hls3_r1 import by_origin
from test_hls3_r2 import GOOD, NOTE, OTHER, PATCH, patch_q, probe_row, scan, with_events

R3 = list(HLS3_R3)
SITE_LONG = 'Mixer.scan_fn'
LONG = (f'[infrarsi-patch] {SITE_LONG}: found ' + ', '.join(f'mod_{i}.scan_fn=<function scan_fn at 0x7f84{i:08x}>'
                                                            for i in range(5)) + ' (reference) -> set kernels.scan (fused)')
COMPILER = ('completes: exit code 1, 0/30 updates; the log: torch._inductor.exc.InductorError: LoweringException: '
            'an op of the compiled graph failed to lower')
WHOLE = {'evidence': {'per_step': []}}


def stamp(eng, cid, gaps, n0: int = 1) -> None:
    """The candidate's receipt with stamped updates n0, n0 + 1, ... separated by `gaps`."""
    ts = [0.0]
    for g in gaps:
        ts.append(ts[-1] + g)
    c = eng.s.candidates[cid]
    c.receipt = {**(c.receipt or {}), 'evidence': {'per_step': [{'step': n0 + i, 't': t} for i, t in enumerate(ts)]}}


def jumpy(n: int, x: float) -> list:
    """n update intervals spread evenly over 1.0 +- 2x in a scrambled order: their MAD / median is about x."""
    return [1.0 + x * (((i * 37) % n) / (n - 1) * 4 - 2) for i in range(n)]


# ---------------------------------------------------------------- registration, flags
def test_r3_components_registered():
    assert len(R3) == 13 and not set(R3) & (set(HLS3_R1) | set(HLS3_R2))
    assert {HLS3_COMPONENTS[c][0] for c in R3} == {'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F21', 'F22',
                                                     'F23', 'F9a', 'F9b', 'D7'}
    f = proposer_flags()
    assert f['parse'] and f['deferred'] and not proposer_flags(['hls3_parse'])['parse']
    assert not proposer_flags(['hls3_deferred'])['deferred'] and not proposer_flags(['hls3_self_proof'])['deferred']
    on = HLS3(FakeProposer([]))
    assert on.carry_patches and not HLS3(FakeProposer([]), off=['hls3_carry_patches']).carry_patches
    assert not HLS(FakeProposer([])).carry_patches and not HLS2(FakeProposer([])).carry_patches   # hls / hls2: never
    assert 'r3' not in HLS3(FakeProposer([])).describe()['hls3']    # nothing happened yet: no section


# ---------------------------------------------------------------- F13: the proof lines read whole
def test_r3_f13_the_scanner_reads_a_long_line_whole():
    assert len(LONG) > 300 and LONG.index("-> set") > 300
    lines = [(100.0 + i, LONG) for i in range(8)] + [(200.0, f'[infrarsi-patch] {SITE_LONG}: active')]
    ev = scan_events(lines)
    assert ev['patch_proof'][SITE_LONG]['installs'] == []           # cut at 300 characters: no `-> set` (round 2)
    whole = ev['patch_proof_whole'][SITE_LONG]
    assert len(whole['installs']) == 1 and whole['installs'][0][1] == 'kernels.scan (fused)'
    assert whole['longest'] == len(LONG) and whole['active_at'] is not None
    assert all(len(x) <= 300 for x in ev['patch_lines']) and len(ev['patch_lines']) <= 20   # the display lines as before
    assert proof_verdict(ev['patch_proof']).startswith('it printed no') and proof_verdict(ev['patch_proof_whole']) is None
    short = scan_events([(1.0, GOOD[0]), (2.0, GOOD[1])])
    assert 'patch_proof_whole' not in short                         # a short line reads the same: nothing more


def test_r3_f13_the_ranks_addresses_fold_but_a_wrapper_of_one_name_stays_two_objects():
    ranks = [f'[infrarsi-patch] m.f: found mod.f=<function f at 0x7f8441a579c{i}> -> set k.f' for i in range(8)]
    ev = proof_evidence(ranks + ['[infrarsi-patch] m.f: active'], whole=True)
    assert ev['m.f']['installs'] == [['mod.f=<function f at 0x<1>>', 'k.f', 0.0]] and proof_verdict(ev) is None
    assert len(proof_evidence(ranks, whole=False)['m.f']['installs']) == 8      # (round 2: one per rank, the cap)
    wrap = ['[infrarsi-patch] m.g: found <function g at 0x7f8441a579c0> -> set <function g at 0x7f8441a57bb0>',
            '[infrarsi-patch] m.g: active']                           # functools.wraps keeps the name: two objects
    assert proof_verdict(proof_evidence(wrap, whole=True)) is None
    two = proof_evidence(wrap, whole=True)['m.g']['installs'][0][:2]
    assert two == ['<function g at 0x<1>>', '<function g at 0x<2>>']
    same = ['[infrarsi-patch] m.h: found <function h at 0x7f8441a579c0> -> set <function h at 0x7f8441a579c0>']
    assert 'already there' in proof_verdict(proof_evidence(same, whole=True))
    mask = proof_evidence(['[infrarsi-patch] m.k: found 0x00ff00ff -> set 0xff00ff00'], whole=True)
    assert mask['m.k']['installs'][0][:2] == ['0x00ff00ff', '0xff00ff00']   # a value, not an address: kept


def test_r3_f13_a_site_with_colons_keeps_its_name():
    lines = [(1.0, '[infrarsi-patch] ns::Kernel.forward: found reference -> set fused'),
             (2.0, '[infrarsi-patch] ns::Kernel.forward: active')]
    ev = scan_events(lines)
    assert ev['patch_proof_whole']['ns::Kernel.forward']['installs'][0][:2] == ['reference', 'fused']
    assert not any(r['installs'] for r in ev['patch_proof'].values())   # (cut parse: the site ends at the first ':')
    assert proof_evidence([x for _, x in lines], whole=True) == {
        'ns::Kernel.forward': {'installs': [['reference', 'fused', 0.0]], 'active_at': 1.0}}


def test_r3_f13_the_gate_reads_the_whole_line_and_reports_what_it_read():
    eng = engine(KNOBS, START)
    with_events(eng, 'c0', perf_notes=[NOTE])
    lines = [(100.0, LONG), (101.0, LONG), (200.0, f'[infrarsi-patch] {SITE_LONG}: active')]
    row = {**{k: v for k, v in scan_events(lines).items() if k.startswith(('patch', 'perf'))}, 'steps': 3}
    on = HLS3(FakeProposer([scan(evidence_note=NOTE)]), explore=False)
    q = patch_q(on, eng)
    run(eng, 'c1', {}, status='probed', fidelity='probe')
    assert q.gate(eng, 'c1', 'c0', row) is None                     # proven: installed, active, the note gone
    off = HLS3(FakeProposer([scan(evidence_note=NOTE)]), explore=False, off=['hls3_scan_whole'])
    q = patch_q(off, eng)
    why, check = q.gate(eng, 'c1', 'c0', row)
    assert check == 'patch_inert' and why.startswith('it printed no')   # round 2: the line cut, its pair lost
    assert 'parsed' not in off.proof_fail['c1']
    bad = LONG.replace(' -> set ', ' then ')                        # a long line with no pair even whole
    row2 = {**{k: v for k, v in scan_events([(100.0, bad)]).items() if k.startswith(('patch', 'perf'))}, 'steps': 3}
    q = patch_q(on, eng)
    why, _ = q.gate(eng, 'c1', 'c0', row2)
    assert why.startswith('it printed no') and f'was {len(bad)} characters' in why and 'read as none' in why
    parsed = on.proof_fail['c1']['parsed'][SITE_LONG]
    assert parsed == {'found': None, 'set': None, 'active': False, 'line_chars': len(bad)}


def test_r3_f13_the_inert_feedback_carries_the_pair_it_read():
    eng = engine(KNOBS, START)
    with_events(eng, 'c0', perf_notes=[NOTE])
    hls = HLS3(FakeProposer([scan(evidence_note=NOTE)]), explore=False)
    q = patch_q(hls, eng)
    run(eng, 'c1', {}, status='probed', fidelity='probe')
    lines = [(100.0, LONG)]                                         # installed, never active
    row = {**{k: v for k, v in scan_events(lines).items() if k.startswith(('patch', 'perf'))}, 'steps': 3}
    why, check = q.gate(eng, 'c1', 'c0', row)
    assert 'no `active` line' in why
    hls.observe(eng, q, Outcome('c1', 'infeasible', False, None, None, 'probe', 0.1, [why], [check]))
    fb = hls.context(eng, 'c0')['inert_patches'][-1]
    assert fb['parsed'][SITE_LONG]['set'] == 'kernels.scan (fused)' and fb['parsed'][SITE_LONG]['active'] is False
    assert fb['parsed'][SITE_LONG]['found'].startswith('mod_0.scan_fn=<function scan_fn at')


# ---------------------------------------------------------------- F14: the answer read hypothesis by hypothesis
FILES = {'train_patches.py': '"""Speed changes of the job."""\n'}
SCHEMA = P.hls3_schema(P.HYPOTHESIS_SCHEMA, {'memory': True, 'sites': True})


def test_r3_f14_each_hypothesis_is_validated_on_its_own():
    code = 'import gc\ngc.freeze()\n'
    items = [{'mechanism': 'knob', 'change': {'a': 1}, 'predicted_gain_pct': 2},
             {'mechanism': 'whole file', 'change': {}, 'predicted_gain_pct': 5, 'patch': code,
              'edits': [{'file': 'train_patches.py', 'op': 'replace_file'}]},
             {'mechanism': 'append, see patch', 'change': {}, 'predicted_gain_pct': 1, 'patch': code,
              'edits': [{'file': 'train_patches.py', 'op': 'append', 'content': '(see patch)'}]},
             {'mechanism': 'append its own text', 'change': {}, 'predicted_gain_pct': 1,
              'patch': 'append to train_patches.py: freeze the collector', 'edits': [{'file': 'train_patches.py',
                                                                                     'op': 'append', 'text': code}]},
             {'mechanism': 'no change object', 'change': 'a=1', 'predicted_gain_pct': 1},
             {'mechanism': 'unknown file', 'change': {}, 'predicted_gain_pct': 1, 'patch': code,
              'edits': [{'file': 'other.py', 'op': 'replace_file'}]}]
    kept, dropped, fixed = P.accept_hypotheses(items, SCHEMA, FILES)
    assert [x['mechanism'] for x in kept] == ['knob', 'whole file', 'append, see patch', 'append its own text']
    assert kept[0] is items[0]                                      # a valid hypothesis as it came
    assert kept[1]['patch'] == {'train_patches.py': code} and kept[1]['edits'] is None
    assert kept[2]['patch'] == kept[3]['patch'] == {'train_patches.py': FILES['train_patches.py'] + code}
    assert [d['index'] for d in dropped] == [4, 5] and 'change' in dropped[0]['problems'][0]
    assert [f['index'] for f in fixed] == [1, 2, 3] and fixed[0]['files'] == ['train_patches.py']
    for k in (1, 2, 3):                                             # the rewritten ones apply as patches do
        assert P.patch_of(kept[k], FILES) == (kept[k]['patch'], True)
    valid = {'hypotheses': [items[0], {'mechanism': 'p', 'change': {}, 'predicted_gain_pct': 1,
                                       'edits': [{'file': 'train_patches.py', 'old': '"""', 'new': '"""x'}]}]}
    assert P.salvage_edits({'edits': [{'file': 'train_patches.py', 'old': '"""', 'new': ''}]}, FILES) is None   # (twice)
    assert P.accept_hypotheses(valid['hypotheses'], SCHEMA, FILES)[0] == valid['hypotheses']   # all valid: as they came
    assert P.relaxed_schema(SCHEMA)['properties']['hypotheses'] == {'type': 'array'}
    assert SCHEMA['properties']['hypotheses']['items']['properties']['memory']          # (the original untouched)


class Recorder:
    """A ScriptedBackend keeping the schema and extra arguments of each hypothesis call."""

    def __init__(self, script):
        from infrarsi.agent.backends.fake import ScriptedBackend
        self.inner, self.sent = ScriptedBackend(script), []

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def complete(self, system, prompt, *, schema=None, role=None, **kw):
        if role == 'hypotheses':
            self.sent.append({'schema': schema, 'prompt': prompt, **{k: kw.get(k) for k in ('join_text', 'retry_note')}})
        return self.inner.complete(system, prompt, schema=schema, role=role, **kw)


def test_r3_f14_the_proposer_keeps_the_good_hypotheses_and_asks_again_only_when_none_is_left():
    from test_search import TOGGLE, setup
    good = {'mechanism': 'fused adam', 'change': {'fused_adam': True}, 'predicted_gain_pct': 3}
    bad = {'mechanism': 'bad', 'change': 'fused_adam=1', 'predicted_gain_pct': 3}
    answers = {'mixed': {'hypotheses': [good, bad]}, 'none': {'hypotheses': [bad]}}

    def script(name):
        seq = list(name)
        return lambda role, system, prompt: {'verdict': 'no_patch', 'findings': []} if role == 'code_review' else \
            answers[seq.pop(0)] if seq else answers['mixed']
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    used = dict(s.ledger.used)
    drv = Recorder(script(['mixed']))
    prop = P.LLMProposer(drv, s.hooks, search_calibrates=True, hls3={'parse': True})
    out = prop.hypotheses(eng, 'c0', ledger=[])
    assert [x['mechanism'] for x in out] == ['fused adam'] and len(drv.sent) == 1   # one call: the bad one dropped
    assert drv.sent[0]['schema']['properties']['hypotheses'] == {'type': 'array'} and drv.sent[0]['join_text']
    assert drv.sent[0]['retry_note'] is P.explain_reply
    assert prop.dropped[0]['mechanism'] == 'bad' and prop.dropped[0]['call'] == 1
    s.ledger.used.update(used)
    drv = Recorder(script(['none', 'mixed']))
    prop = P.LLMProposer(drv, s.hooks, search_calibrates=True, hls3={'parse': True})
    out = prop.hypotheses(eng, 'c0', ledger=[])
    assert [x['mechanism'] for x in out] == ['fused adam'] and len(drv.sent) == 2   # none usable: asked once more
    errors = json.loads(drv.sent[1]['prompt'])['previous_reply_errors']
    assert errors[0].startswith("hypothesis 0 ('bad'): $.hypotheses[0].change: expected object") and P.EDITS_HINT in errors
    s.ledger.used.update(used)
    drv = Recorder(script(['mixed']))
    old = P.LLMProposer(drv, s.hooks, search_calibrates=True, hls3={'memory': True})   # F14 off: round 2
    assert old.hypotheses(eng, 'c0', ledger=[]) == [] and old.failed == 1          # the whole answer refused
    assert drv.sent[0]['schema']['properties']['hypotheses'].get('items') and drv.sent[0]['join_text'] is None


def test_r3_f14_the_retry_note_names_the_exact_error():
    cut = SimpleNamespace(error='invalid structured JSON action', text='{"hypotheses": [{"mechanism": "x", "chan')
    note = P.explain_reply(cut)
    assert 'not one JSON object' in note and 'at character' in note and P.EDITS_HINT in note
    shape = SimpleNamespace(error='schema: $: missing hypotheses', text='{}')
    assert "did not have the answer's shape: $: missing hypotheses" in P.explain_reply(shape)
    assert 'it was empty' in P.explain_reply(SimpleNamespace(error='invalid structured JSON action', text=''))


def _cli(monkeypatch, streams, prompts):
    from infrarsi.agent.backends import claude_cli
    it = iter(streams)

    class FakePopen:
        def __init__(self, argv, stdin=None, stdout=None, stderr=None, cwd=None, env=None, **kw):
            self.pid, self.returncode, self._out = 4242, None, stdout

        def communicate(self, data=None, timeout=None):
            prompts.append(data.decode('utf-8'))
            self._out.write(next(it).encode('utf-8'))
            self.returncode = 0
            return b'', b''
    monkeypatch.setattr(claude_cli.subprocess, 'Popen', FakePopen)


def split_stream(text: str, at: int) -> str:
    """A recorded CLI stream (FABLE_HLS3_FIXES_R3 appendix B: Q9H call 4) whose two text blocks - split over the CLI's
    resume turn - hold `text` cut at `at`; the result event holds only the second, as the CLI wrote it."""
    from hls3_helpers import R3_FIX
    events = [json.loads(x) for x in R3_FIX['llm'][4]['stdout'].splitlines() if x.strip()]
    blocks, out = [text[:at], text[at:]], []
    for e in events:
        m = e.get('message') or {}
        if e.get('type') == 'assistant' and any(b.get('type') == 'text' for b in m.get('content') or []):
            e = {**e, 'message': {**m, 'content': [{'type': 'text', 'text': blocks.pop(0)}]}}
        elif e.get('type') == 'result':
            e = {**e, 'result': text[at:]}
        out.append(e)
    return '\n'.join(json.dumps(x) for x in out) + '\n'


def test_r3_f14_the_cli_joins_a_reply_split_over_its_resume_turn(tmp_path, monkeypatch):
    from infrarsi.agent.backends import claude_cli
    answer = json.dumps({'hypotheses': [{'mechanism': 'x' * 50, 'change': {'a': 1}, 'predicted_gain_pct': 1}]})
    raw = split_stream(answer, 40)
    action, f = claude_cli.parse_stream(raw, model='claude-opus-5-5', exit_code=0, status='RETURNED')
    assert action is None and f['validation_error'] == 'invalid structured JSON action'    # off: round 2
    action, f = claude_cli.parse_stream(raw, model='claude-opus-5-5', exit_code=0, status='RETURNED', join_text=True)
    assert action == json.loads(answer) and f['parse_mode'] == 'joined' and f['result_text'] == answer
    whole = split_stream(answer, 0)                                 # one block: the same answer either way
    assert claude_cli.parse_stream(whole, model='claude-opus-5-5', exit_code=0, status='RETURNED', join_text=True)[1][
        'parse_mode'] == 'strict'
    b = claude_cli.ClaudeCLIBackend(executable=sys.executable, model='claude-opus-5-5', timeout_seconds=60, max_retries=1)
    prompts = []
    _cli(monkeypatch, [split_stream('{"hypotheses": [', 5), raw], prompts)
    r = b.complete('system', 'prompt', call_dir=tmp_path / 'c1', schema={'type': 'object'}, join_text=True,
                   retry_note=P.explain_reply)
    assert r.action == json.loads(answer) and r.attempts == 2 and r.meta['parse_mode'] == 'joined'
    assert 'not one JSON object (Expecting' in prompts[1] and P.EDITS_HINT in prompts[1]   # the exact error named
    prompts = []
    _cli(monkeypatch, [split_stream('{"hypotheses": [', 5), raw], prompts)
    r = b.complete('system', 'prompt', call_dir=tmp_path / 'c2', schema={'type': 'object'})
    assert r.error == 'invalid structured JSON action' and 'could not be used (invalid structured JSON action)' in prompts[1]


# ---------------------------------------------------------------- F15: a seed's spender mu
def seeded(**kw):
    exp = {'seeds': [{'knob': 'n', 'mechanism': 'n', 'change': {'n': 4}, 'level': 'partial', 'gain': 0.0, 'sign': '+',
                      'evidence_ids': [], 'warnings': [], 'p_feasible': 0.75, 'valid_on': {'n': 1}}],
           'judgment': {'assessment': [{'knob': 'n', 'change': {'n': 4}, 'sign': '+', 'p_feasible': 0.75,
                                        'expected_gain_pct': kw.pop('judged', 20)}]}}
    return HLS3(FakeProposer(kw.pop('hyps', [])), explore=False, experience=exp, **kw)


def test_r3_f15_a_seed_that_ran_out_of_memory_is_a_spender_at_the_judged_gain():
    eng = engine(KNOBS, START)
    hls = seeded(off=['hls3_pair_probes'])
    hls.propose(eng, 16)
    key = 'c0|experience:n'
    assert hls.ledger[key]['mu'] == 0.0
    observe(hls, eng, 'c1', {'n': 4}, hypothesis=key, reasons=[G4U_OOM])
    assert hls.ledger[key]['failure'] == 'oom' and hls.ledger[key]['mu'] == 0.0   # the ledger keeps its zero
    assert [x['key'] for x in hls._spenders(eng, 'c0')] == [key]
    assert hls.seed_mu[key] == pytest.approx(math.log1p(0.20), abs=1e-5)
    eng = engine(KNOBS, START)
    hls = seeded(off=['hls3_pair_probes'], hyps=[h('n4 again', {'n': 4}, mu=0.3, mem_dir='spends')])
    hls.propose(eng, 16)
    observe(hls, eng, 'c1', {'n': 4}, hypothesis=key, reasons=[G4U_OOM])
    assert hls.ledger[key]['llm_mu'] == 0.3 and hls._spenders(eng, 'c0')   # (the model's prediction: its twin's)
    assert hls.seed_mu[key] == pytest.approx(0.3)                  # the proposer's prediction of the move: the most said
    eng = engine(KNOBS, START)
    hls = HLS3(FakeProposer([]), explore=False, off=['hls3_pair_probes'], experience={'seeds': seeded().exp['seeds']})
    hls.propose(eng, 16)
    observe(hls, eng, 'c1', {'n': 4}, hypothesis=key, reasons=[G4U_OOM])
    k = hls._kappa_for(hls.ledger[key], hls.kappa())
    assert [x['key'] for x in hls._spenders(eng, 'c0')] == [key] and k * hls.seed_mu[key] >= 2 * eng.sigma_pair()
    eng = engine(KNOBS, START)
    off = seeded(off=['hls3_pair_probes', 'hls3_seed_mu'])
    off.propose(eng, 16)
    observe(off, eng, 'c1', {'n': 4}, hypothesis=key, reasons=[G4U_OOM])
    assert off._spenders(eng, 'c0') == [] and not off.seed_mu      # round 2: a zero prior is no spender


# ---------------------------------------------------------------- F16: pair probes of a memory-blocked spender
def test_r3_f16a_a_composite_that_died_at_its_start_gives_the_spender_its_partners():
    hyps = [h('n4', {'n': 4}, mu=0.12, mem_dir='spends'), h('a', {'a': 1}, mu=0.005), h('z', {'z': 1}, mu=0.004),
            h('combo', {'n': 4, 'a': 1, 'z': 1}, mu=0.15, p=0.6)]
    eng = engine(KNOBS, START)
    hls = HLS3(FakeProposer(hyps), explore=False, off=['hls3_coverage', 'hls3_explore'])
    hls.propose(eng, 16)
    observe(hls, eng, 'c1', {'n': 4, 'a': 1, 'z': 1}, hypothesis='c0|combo', reasons=[COMPILER])
    assert hls.composites and hls.composites[0]['kind'] == 'compiler'
    observe(hls, eng, 'c2', {'n': 4}, hypothesis='c0|n4', reasons=[G4U_OOM])
    props = hls.propose(eng, 16)
    assert [p.origin for p in props] == ['hls3-pair-probe'] * 2 and all(p.fidelity == ('probe',) and p.keep_probe
                                                                         and not p.abortable for p in props)
    assert [p.change for p in props] == [{'n': 4, 'a': 1}, {'n': 4, 'z': 1}]   # the composite's members, the likelier first
    assert [r['kind'] for r in hls.pair_probes] == ['composite'] * 2
    k = hls._kappa_for(hls.ledger['c0|n4'], hls.kappa())
    assert all(p.mu >= k * 0.12 for p in props)                     # valued at the spender (and its partner)
    first, second = props
    hls.observe(eng, first, run(eng, 'c3', first.change, status='probed', failed=(), fidelity='probe'))
    bid = first.hypothesis[len('probe|'):]
    assert hls.pending[bid].fidelity == ('full',) and hls.pending[bid].mu == pytest.approx(first.mu)
    assert hls.pair_probes[0]['status'] == 'probe passed'
    n_before = hls.need[json.dumps({'n': 4})]['probes']
    hls.observe(eng, second, run(eng, 'c4', second.change, reasons=[G4U_OOM], fidelity='probe', status='infeasible'))
    assert hls.need[json.dumps({'n': 4})]['probes'] == n_before + 1 and hls.pair_probes[1]['status'] == 'oom'   # (F3)
    assert len(hls.pair_ran) == 2
    assert hls.describe()['hls3']['r3']['pair_probes'][0]['partner'] == 'a=1'
    eng = engine(KNOBS, START)
    off = HLS3(FakeProposer(hyps), explore=False, off=['hls3_coverage', 'hls3_explore', 'hls3_pair_probes'])
    off.propose(eng, 16)
    observe(off, eng, 'c1', {'n': 4, 'a': 1, 'z': 1}, hypothesis='c0|combo', reasons=[COMPILER])
    observe(off, eng, 'c2', {'n': 4}, hypothesis='c0|n4', reasons=[G4U_OOM])
    assert not [p for p in off.propose(eng, 16) if p.origin == 'hls3-pair-probe'] and not off.composites


def test_r3_f16b_undeclared_partners_and_not_when_a_bundle_covers_the_spender():
    knobs = {**KNOBS, 'w': [0, 1], 'v': [0, 1], 'u': [0, 1]}
    start = {**START, 'w': 0, 'v': 0, 'u': 0}
    hyps = [h('n4', {'n': 4}, mu=0.12, mem_dir='spends'), h('z', {'z': 1}, mu=0.01),
            h('a neutral', {'a': 1}, mu=0.005, mem_dir='neutral'), h('w frees', {'w': 1}, mu=0.0, mem_dir='frees', mem_gib=1.0),
            h('v spends', {'v': 1}, mu=0.02, mem_dir='spends'), h('u slow', {'u': 1}, mu=0.001)]

    def setup(hyp_list, **kw):
        eng = engine(knobs, start)
        hls = HLS3(FakeProposer(hyp_list), explore=False, **{'off': ['hls3_coverage', 'hls3_explore'], **kw})
        hls.propose(eng, 16)
        observe(hls, eng, 'c1', {'n': 4}, hypothesis='c0|n4', reasons=[G4U_OOM])
        return eng, hls, hls.propose(eng, 16)
    eng, hls, props = setup(hyps)
    pair = [p for p in props if p.origin == 'hls3-pair-probe']
    assert [p.change for p in pair] == [{'n': 4, 'z': 1}, {'n': 4, 'a': 1}]   # two at most, the likelier first; not the
    assert [r['kind'] for r in hls.pair_probes] == ['undeclared'] * 2       # declared enabler (S-2's), nor a spender
    eng, hls, props = setup(hyps)
    observe(hls, eng, 'c2', {'n': 4}, hypothesis='c0|n4', reasons=[G4U_OOM], base='c0')   # its own run again: no new pair
    assert len(hls.pair_probes) == 2                                         # (each pair once on a base, two per spender)
    covered = [h('n4', {'n': 4}, mu=0.12, mem_dir='spends'), h('z', {'z': 1}, mu=0.01),
               h('w frees', {'w': 1}, mu=0.0, mem_dir='frees', mem_gib=6.0)]
    eng, hls, props = setup(covered)
    assert not hls.pair_probes and [p.origin for p in props if p.origin == 'hls3-bundle']   # S-2 covers 4.53 GiB short


def test_r3_f16_a_pair_probe_is_exempt_from_its_own_pair_and_swaps_another():
    knobs = {**KNOBS, 'w': [0, 1]}
    hyps = [h('n4', {'n': 4}, mu=0.3, mem_dir='spends'), h('a', {'a': 1}, mu=0.01), h('w1', {'w': 1}, mu=0.05)]
    eng = engine(knobs, {**START, 'w': 0})
    eng.s.candidates['c0'].verdict = {'valid': True, 'step_time': 1.0}
    hls = HLS3(FakeProposer(hyps), explore=False, off=['hls3_coverage', 'hls3_explore'])
    hls.propose(eng, 16)
    observe(hls, eng, 'c1', {'w': 1}, hypothesis='c0|w1', status='screened', gain=0.05, st=0.95, failed=())
    eng._inc = 'c1'                                                 # w came in with c1
    hls.propose(eng, 16)
    pair = lambda *m: frozenset(frozenset({x}) for x in m)          # noqa: E731
    hls.suspects[pair(('n', '4'), ('a', '1'))] = {'run': 'c7', 'kind': 'compiler', 'strikes': 1}   # the pair it tests
    hls.suspects[pair(('a', '1'), ('w', '1'))] = {'run': 'c8', 'kind': 'compiler', 'strikes': 1}   # w: the incumbent's
    observe(hls, eng, 'c2', {'n': 4}, hypothesis='c1|n4', base='c1', reasons=[G4U_OOM])
    props = hls.propose(eng, 16)
    swap = [p for p in props if p.origin == 'hls3-swap']
    assert len(swap) == 1 and swap[0].change == {'n': 4, 'a': 1, 'w': 0} and hls.pair_probes[0]['status'] == 'swap'
    assert swap[0].mu == pytest.approx(hls.pair_probes[0]['mu']) and swap[0].mu >= 2 * eng.sigma_pair()   # less w's gain
    hls2 = HLS3(FakeProposer(hyps[:2]), explore=False, off=['hls3_coverage', 'hls3_explore'])
    eng = engine(knobs, {**START, 'w': 1})
    hls2.propose(eng, 16)
    hls2.suspects[pair(('n', '4'), ('a', '1'))] = {'run': 'c7', 'kind': 'compiler', 'strikes': 2}
    observe(hls2, eng, 'c1', {'n': 4}, hypothesis='c0|n4', reasons=[G4U_OOM])
    assert [p.change for p in hls2.propose(eng, 16) if p.origin == 'hls3-pair-probe'] == [{'n': 4, 'a': 1}]


# ---------------------------------------------------------------- F17: coverage members resolved by their bundle
def test_r3_f17_members_of_a_bundle_measured_within_the_noise_are_resolved_by_it():
    hyps = [h('a', {'a': 1}, mu=0.005), h('z', {'z': 1}, mu=0.004), h('mb', {'mb': 2}, mu=0.006), h('n4', {'n': 4}, mu=0.08)]

    def setup(**kw):
        eng = engine(KNOBS, START)
        eng.s.candidates['c0'].verdict = {'valid': True, 'step_time': 1.0}
        hls = HLS3(FakeProposer(hyps), explore=False, **kw)
        cov = by_origin(hls.propose(eng, 16))['hls3-coverage'][0]
        return eng, hls, cov
    eng, hls, cov = setup()
    hls.observe(eng, cov, run(eng, 'c1', cov.change, status='screened', gain=-0.001, st=1.001, failed=()))
    m = [hls.ledger[k] for k in hls.members[cov.hypothesis]]
    assert {x['status'] for x in m} == {'unresolved'} and all(x['why'].startswith('bundled, measured together: -0.1 %')
                                                               for x in m)
    singles = [q.change for q in hls.propose(eng, 16) if len(q.change or {}) == 1]
    assert {'a': 1} not in singles and {'z': 1} not in singles and {'mb': 2} not in singles
    assert [r['mechanism'] for r in hls.describe()['hls3']['r3']['captured']] == [x['mechanism'] for x in m]
    eng, hls, cov = setup()                                         # rejected at its probe: not measured, back as before
    hls.observe(eng, cov, run(eng, 'c1', cov.change, reasons=[COMPILER], fidelity='probe', status='infeasible'))
    assert {hls.ledger[k]['status'] for k in hls.members[cov.hypothesis]} == {'open'} and not hls.captured
    eng, hls, cov = setup()                                         # a gain within the noise: one of them may help - back
    hls.observe(eng, cov, run(eng, 'c1', cov.change, status='screened', gain=0.004, st=0.996, failed=()))   # to singles
    assert {hls.ledger[k]['status'] for k in hls.members[cov.hypothesis]} == {'open'} and not hls.captured
    eng, hls, cov = setup(off=['hls3_cov_unresolved'])              # round 2: back to their single screens
    hls.observe(eng, cov, run(eng, 'c1', cov.change, status='screened', gain=-0.001, st=1.001, failed=()))
    assert {hls.ledger[k]['status'] for k in hls.members[cov.hypothesis]} == {'open'}


def test_r3_f17_a_new_prediction_above_the_noise_runs_a_captured_move_alone():
    from test_hls_reask import Script
    first = [h('a', {'a': 1}, mu=0.005), h('z', {'z': 1}, mu=0.004)]
    again = [h('a weak', {'a': 1}, mu=0.004), h('z strong', {'z': 1}, mu=0.2)]
    eng = engine(KNOBS, START)
    eng.s.candidates['c0'].verdict = {'valid': True, 'step_time': 1.0}
    hls = HLS3(Script([first, again]), explore=False)
    cov = by_origin(hls.propose(eng, 16))['hls3-coverage'][0]
    hls.observe(eng, cov, run(eng, 'c1', cov.change, status='screened', gain=-0.002, st=1.002, failed=()))
    hls._ask(eng, 'c0')
    assert hls.ledger['c0|a weak']['status'] == 'unresolved' and hls.ledger['c0|z strong']['status'] == 'open'


# ---------------------------------------------------------------- F18: a patch follows a knob-only incumbent change
def test_r3_f18_an_open_patch_follows_a_knob_only_change_of_the_incumbent():
    hyps = [h('kernel', {}, patch={'f.py': 'fast\n'}, mu=0.004), h('n2', {'n': 2}, mu=0.1)]

    def setup(overlay=None, **kw):
        eng = engine(KNOBS, START)
        eng.s.candidates['c0'].verdict = {'valid': True, 'step_time': 1.0}
        hls = HLS3(FakeProposer(hyps), explore=False, **{'off': ['hls3_coverage'], **kw})
        hls.propose(eng, 16)
        observe(hls, eng, 'c1', {'n': 2}, hypothesis='c0|n2', status='screened', gain=0.1, st=0.905, failed=())
        if overlay is not None:
            eng.s.candidates['c1'].overlay = overlay
        eng._inc = 'c1'
        hls.propose(eng, 16)
        return hls
    hls = setup()
    e = next(v for v in hls.ledger.values() if v['base'] == 'c1' and v.get('patch'))
    assert e['status'] == 'open' and e['carried_from'] == 'c0' and hls.carried_patches == 1
    assert hls.describe()['hls3']['r3']['carried_patches'] == 1
    hls = setup(overlay={'f.py': 'other\n'})                        # the code changed: not carried (round 2's rule)
    assert not any(v['base'] == 'c1' and v.get('patch') and v.get('carried_from') for v in hls.ledger.values())
    assert 'not carried' in hls.ledger['c0|kernel']['why']
    hls = setup(off=['hls3_coverage', 'hls3_carry_patches'])
    assert not any(v['base'] == 'c1' and v.get('patch') and v.get('carried_from') for v in hls.ledger.values())


# ---------------------------------------------------------------- F19: the judgment's composite sets the pair set
def test_r3_f19_pairs_are_suspect_up_to_the_start_composite():
    knobs = {**KNOBS, 'w': [0, 1]}

    def setup(start_config, **kw):
        eng = engine(knobs, {**START, 'w': 0})
        exp = {'judgment': {'assessment': [], 'start_config': start_config}}
        hls = HLS3(FakeProposer([h('four', {'n': 4, 'a': 1, 'z': 1, 'w': 1}, mu=0.1, p=0.6)]), explore=False,
                   experience=exp, **{'off': ['hls3_coverage', 'hls3_explore'], **kw})
        hls.propose(eng, 16)
        observe(hls, eng, 'c1', {'n': 4, 'a': 1, 'z': 1, 'w': 1}, hypothesis='c0|four', reasons=[COMPILER])
        return hls
    four = {'n': 4, 'a': 1, 'z': 1, 'w': 1}
    hls = setup(four)
    assert hls._pair_set() == 4 and len(hls.suspects) == 6
    assert hls.describe()['hls3']['r3']['pair_set'] == 4
    assert setup({**four, 'mb': 2})._pair_set() == 5 and setup({f'k{i}': 1 for i in range(9)})._pair_set() == PAIR_SET_MAX
    assert setup({'n': 4})._pair_set() == PAIR_SET                    # (never below round 1's three)
    assert len(setup(four, off=['hls3_coverage', 'hls3_explore', 'hls3_pair_set']).suspects) == 0   # round 2: a set of 4


# ---------------------------------------------------------------- F21: the `deferred` line
def test_r3_f21_a_deferred_line_and_the_two_explanations_of_a_note_still_reported():
    deferred = [GOOD[0], '[infrarsi-patch] mixer.fast_scan_fn: deferred (kwargs non-empty)']
    assert proof_verdict(deferred, deferred=True) == ('mixer.fast_scan_fn: `deferred (kwargs non-empty)` and no `active` '
                                                      'line - the replacement handed every call back to the original')
    assert proof_verdict(deferred) == 'mixer.fast_scan_fn: no `active` line - the patched code never ran within the probe'
    assert proof_evidence(deferred)['mixer.fast_scan_fn']['deferred'][0] == 'kwargs non-empty'
    eng = engine(KNOBS, START)
    with_events(eng, 'c0', perf_notes=[NOTE])
    hls = HLS3(FakeProposer([scan(evidence_note=NOTE)]), explore=False)
    q = patch_q(hls, eng)
    run(eng, 'c1', {}, status='probed', fidelity='probe')
    why = q.gate(eng, 'c1', 'c0', probe_row(GOOD, [NOTE]))[0]       # active, and the note still reported
    assert why.endswith('the replacement ran at least once and the note was still reported: it defers on some calls')
    why = q.gate(eng, 'c1', 'c0', probe_row(deferred, [NOTE]))[0]   # no active, a deferral printed
    assert 'either the replaced object is not the one called, or the replacement defers to the original' in why
    assert why.endswith('(it printed `deferred (kwargs non-empty)`)')
    off = HLS3(FakeProposer([scan(evidence_note=NOTE)]), explore=False, off=['hls3_deferred'])
    q = patch_q(off, eng)
    assert q.gate(eng, 'c1', 'c0', probe_row(GOOD, [NOTE]))[0].endswith('the code it replaced is not the code the job runs')
    assert P.PROOF_DEFER_RULE in P.LLMProposer(None, SimpleNamespace(), hls3={'self_proof': True,
                                                                               'deferred': True})._hls3_rules(True)
    assert P.PROOF_DEFER_RULE not in P.LLMProposer(None, SimpleNamespace(), hls3={'self_proof': True})._hls3_rules(True)


def test_r3_f21_the_toys_patches_print_their_deferral_when_asked():
    from infrarsi.search.toytrain import ToySim, tasks
    from test_search import TOGGLE, setup
    sim = ToySim(tasks(36)[0])
    sp = P.SimProposer(sim)
    HLS3(sp)
    assert sp.ask_proof and sp.ask_defer
    sp.proof_rate = sp.defer_rate = 1.0
    assert sp._hyp({}, 'prefetch_data', (), 'patch:prefetch_data', 0.05, 0.01, 0.9)['patch'] == {
        'patches.txt': 'defer:prefetch_data\nprefetch_data\nproof:prefetch_data\n'}
    other = P.SimProposer(sim)
    HLS3(other, off=['hls3_deferred'])
    assert other.ask_proof and not other.ask_defer
    sim, s, _ = setup(TOGGLE)
    good = next(p for p, ok in sorted(sim.patch_ok.items()) if ok)
    sim.inert = frozenset({good})
    cfg = sim.reference()
    r = s.runner.run(s.task, cfg, {'patches.txt': f'defer:{good}\n{good}\nproof:{good}\n'}, 'probe', 'r', candidate_id='c9',
                     steps=30, stop_after=2)
    assert r['evidence']['events']['patch_lines'] == [f'[infrarsi-patch] {good}: found off -> set on',
                                                      f'[infrarsi-patch] {good}: deferred (its hook hands every call back)']
    r2 = s.runner.run(s.task, cfg, {'patches.txt': f'{good}\nproof:{good}\n'}, 'probe', 'r', candidate_id='c8', steps=30,
                      stop_after=2)
    assert r2['evidence']['events']['patch_lines'] == [f'[infrarsi-patch] {good}: found off -> set on']   # as before


# ---------------------------------------------------------------- F23: thrash from scattered updates
def test_r3_f23_the_scatter_of_a_runs_updates():
    ev = lambda gaps, n0=1: {'per_step': [{'step': n0 + i, 't': t} for i, t in    # noqa: E731
                                          enumerate([0.0] + [sum(gaps[:k + 1]) for k in range(len(gaps))])]}
    assert scatter(ev([1.0] * 29)) == 0.0
    assert scatter(ev(jumpy(29, 0.05))) == pytest.approx(0.05, rel=0.25)
    assert scatter(ev([1.0] * 3)) is None                           # too few updates to read
    long_ = [9.0] * 10 + [1.0] * 19                                 # the first ten (warm-up) not read from 20 updates on
    assert scatter(ev(long_)) == 0.0
    assert scatter({'per_step': []}) is None and scatter(None) is None


def test_r3_f23_a_slower_spending_run_with_scattered_updates_is_thrash():
    hyps = [h('n4', {'n': 4}, mu=0.12, mem_dir='spends'), h('z neutral', {'z': 1}, mu=0.03, mem_dir='neutral')]

    def setup(**kw):
        eng = engine(KNOBS, START)
        eng.s.candidates['c0'].verdict = {'valid': True, 'step_time': 1.0}
        stamp(eng, 'c0', [1.0 + 0.001 * (i % 3) for i in range(29)])
        hls = HLS3(FakeProposer(hyps), explore=False, **{'off': ['hls3_coverage', 'hls3_explore', 'hls3_pair_probes'], **kw})
        hls.propose(eng, 16)
        return eng, hls

    def slow(eng, hls, cid, change, key, gaps):
        out = run(eng, cid, change, status='aborted', failed=())
        eng.t_partial = lambda c: 1.15
        stamp(eng, cid, gaps, n0=1)
        p = Proposal(change=change, base='c0', hypothesis=key)
        cls = hls.classify(eng, p, out)
        hls.observe(eng, p, out)
        return cls
    eng, hls = setup()
    cls = slow(eng, hls, 'c1', {'n': 4}, 'c0|n4', jumpy(15, 0.07))
    assert cls['kind'] == 'thrash' and cls['jitter']['thrash'] and cls['jitter']['scatter_pct'] > 100 * JITTER_MIN
    e = hls.ledger['c0|n4']
    assert e['status'] == 'invalid' and e['failure'] == 'thrash' and hls.mem.runs['c1']['status'] == 'thrash'
    assert [x['key'] for x in hls._spenders(eng, 'c0')] == ['c0|n4']   # a spender now (S-2, F16)
    assert hls.failures[-1]['class'] == 'thrash' and 'scattered' in hls.failures[-1]['detail']
    cls = slow(eng, hls, 'c2', {'z': 1}, 'c0|z neutral', jumpy(15, 0.07))   # declared neutral: scattered, no spend
    assert cls['kind'] == 'slower' and cls['jitter']['scattered'] and not cls['jitter']['spends']
    eng, hls = setup()
    stamp(eng, 'c0', jumpy(29, 0.02))                               # the base scatters itself: 5x its own needed
    assert slow(eng, hls, 'c1', {'n': 4}, 'c0|n4', jumpy(15, 0.07))['kind'] == 'slower'
    eng, hls = setup(off=['hls3_coverage', 'hls3_explore', 'hls3_pair_probes', 'hls3_jitter_thrash'])
    assert slow(eng, hls, 'c1', {'n': 4}, 'c0|n4', jumpy(15, 0.07))['kind'] == 'slower'   # round 2
    assert not hls.jitter


# ---------------------------------------------------------------- shared: the toy's undeclared-enabler family
def test_r3_shared_the_toy_undeclared_enabler_family():
    from infrarsi.search.toytrain import MemSim, enabler_tasks, mem_tasks
    ts = enabler_tasks()
    assert len(ts) == 18 and all(t.name.endswith('-enab') for t in ts) and len({t.seed for t in ts}) == 18
    assert all(not t.name.endswith('-enab') for t in mem_tasks()) and len(mem_tasks()) == 36   # the 36 as they were
    for t in ts:
        s = MemSim(t)
        assert s.memory({'recompute': 'none'}) > s.cap                                  # the spender alone: out of memory
        assert s.memory({'recompute': 'none', 'compile': True}) <= 0.9 * s.cap            # with the enabler it fits
        assert s.memory({'recompute': 'none', 'fused_loss': True, 'expandable_segments': True}) > s.cap   # declared ones: no
        assert s.declared_neutral == ('compile',) and s.jitter({'recompute': 'none', 'compile': True}) == 0.0
    sp = P.SimProposer(MemSim(ts[0]), memory=True)
    assert sp._memory({}, {'compile': True}, 'compile') == {'mem_dir': 'neutral'}
    plain = MemSim(mem_tasks()[0])
    assert plain.cliff_sd == 0.0 and plain.declared_neutral == () and plain.jitter({'micro_batch': 4}) == 0.0
