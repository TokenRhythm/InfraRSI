"""Codex HLS5 r1 (reviews/HLS5_r1_out.md): a regression test per finding in the shared round-3 pieces, each failing on
the code before its fix (qa_trials/_pc_code_hls5_r3) and passing now. Finding 6 (a transport retry's stale peak) is in
tests/cpu/project/test_mempeak.py.

1. F13: a canonical found / set compared whole - folding an address and then cutting at 120 characters made a real
   change "set what was already there".
2. F13: an inherited wrapper keeping the name of what it replaced is the base's on every run - its raw addresses made
   it new on the next execution, and its `active` line proved a patch that printed nothing.
3. F10: several nodes' logs one after the other - the latest report of a note against the earliest installation, not
   the last line read.
4. D7: only files that resolve inside the job interpreter's own site-packages roots, not a path that names one.
5. F14: a usable whole-file `patch` object kept when only its `edits` are malformed."""
import json
import os
import sys
from types import SimpleNamespace

import pytest

from infrarsi.project import libsource
from infrarsi.project.signals import perf_note, proof_evidence, scan_events
from infrarsi.search import proof
from infrarsi.search import proposers as P
from test_shared_features import LIB, ArgvExecutor

INSTALL = '[infrarsi-patch] m.f: found {} -> set {}'
ACTIVE = '[infrarsi-patch] m.f: active'


# ---------------------------------------------------------------- 1. F13: compared whole, cut only for display
@pytest.mark.parametrize('pad', [130, 300])
def test_r1_a_change_past_the_display_length_is_a_change(pad):
    obj = 'mod.f=<function f at 0x7f00000a10c0> ' + 'x' * pad      # the same object, found and set: one label
    lines = [INSTALL.format(obj + ' (reference path)', obj + ' (fused path)'), ACTIVE]
    ev = proof_evidence(lines, whole=True)
    assert proof.proof_verdict(ev) is None                         # a real change, and it ran: proven
    f, v, _ = ev['m.f']['installs'][0]
    assert f != v and f.startswith('mod.f=<function f at 0x<1>>')
    same = proof_evidence([INSTALL.format(obj + ' (path)', obj + ' (path)'), ACTIVE], whole=True)
    why = proof.proof_verdict(same)
    assert 'it set what was already there' in why and len(why) < 2 * proof.SHOW + 120   # shown cut, compared whole
    assert all(len(x['found'] or '') <= proof.SHOW + 3 for x in proof.parsed(ev, {}).values())


# ---------------------------------------------------------------- 2. F13: an inherited wrapper is the base's
def test_r1_an_inherited_wrapper_is_the_bases_whatever_addresses_its_run_gave():
    def run(a, b):                                                 # functools.wraps: the wrapper keeps the name
        return [INSTALL.format(f'<function g at {a}>', f'<function g at {b}>'), ACTIVE]
    base = proof_evidence(run('0x7f0000000a01', '0x7f0000000b01'), whole=True)
    cand = proof_evidence(run('0x7f0000000c77', '0x7f0000000d88'), whole=True)   # the next execution's addresses
    assert proof.proof_verdict(base) is None                       # in its own run the wrapper proves itself
    why = proof.proof_verdict(cand, base)                          # a later patch on it that printed nothing
    assert why is not None and why.startswith('it printed no') and "its base's" in why
    assert proof.new_installs(cand, base) == []
    one = proof_evidence([INSTALL.format('<function h at 0x7f0000000e01>', '<function h at 0x7f0000000e01>')],
                         whole=True)
    assert 'already there' in proof.proof_verdict(one)            # one object found and set: still nothing changed


# ---------------------------------------------------------------- 3. F10: node logs one after the other
NOTE_LINE = ('`causal_conv1d_fn` is falling back to its reference PyTorch implementation because `causal_conv1d` is '
             'not installed. This is correct but much slower.')


def test_r1_appended_node_logs_keep_the_latest_note_and_the_earliest_installation():
    node0 = [(10.0, INSTALL.format('ref', 'fused')), (11.0, ACTIVE), (20.0, NOTE_LINE)]   # the fallback after install
    node1 = [(5.0, NOTE_LINE), (10.5, INSTALL.format('ref', 'fused')), (11.5, ACTIVE)]    # (its start-up warning)
    note = perf_note(NOTE_LINE)
    h = {'evidence_note': NOTE_LINE[:80]}
    verdicts = []
    for lines in (node0 + node1, sorted(node0 + node1)):            # as the runner appends them, and in time order
        ev = scan_events(lines)
        assert ev['perf_note_last'][note] == pytest.approx(20.0, abs=1e-3)
        assert ev['patch_proof']['m.f']['installs'][0][2] == pytest.approx(10.0, abs=1e-3)
        why, named = proof.proof_reason({**ev, 'steps': 3}, {}, [note], '', h)
        verdicts.append(why)
        assert named == note and why and 'still reported after it installed' in why
    assert verdicts[0] == verdicts[1]
    early = [(9.0, INSTALL.format('ref', 'fused'))]                 # a node that installed earlier: its place counts
    assert scan_events(node0 + early)['patch_proof']['m.f']['installs'][0][2] == pytest.approx(9.0, abs=1e-3)


# ---------------------------------------------------------------- 4. D7: inside the interpreter's own roots only
@pytest.fixture(scope='module')
def jobpy(tmp_path_factory):
    from jobenv import make_venv
    root = tmp_path_factory.mktemp('job')
    py, sp = make_venv(root / 'venv')
    (sp / 'fakelib').mkdir()
    (sp / 'fakelib' / 'conv.py').write_text(LIB, encoding='utf-8')
    return py, sp, root


def test_r1_library_source_is_read_only_inside_the_interpreters_site_roots(jobpy, tmp_path):
    py, sp, root = jobpy
    private = 'def causal_conv1d_fn(x):\n    return "a private file"\n'
    named = tmp_path / 'work' / 'not-site-packages' / 'private.py'   # a path that names site-packages
    fake_root = tmp_path / 'work' / 'site-packages' / 'private2.py'  # a directory of that name, not a root
    for f in (named, fake_root):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(private, encoding='utf-8')
    mods_map = {'fakelib.conv': str(sp / 'fakelib' / 'conv.py'), 'private': str(named), 'private2': str(fake_root)}
    outside = tmp_path / 'outside' / 'secret.py'                    # a link inside the root to a file outside it
    outside.parent.mkdir()
    outside.write_text(private, encoding='utf-8')
    try:
        os.symlink(outside, sp / 'linked_secret.py')
        mods_map['linked_secret'] = str(sp / 'linked_secret.py')
    except (OSError, NotImplementedError):                          # (no symlink privilege on this machine)
        pass
    mods = tmp_path / 'modules'
    mods.mkdir()
    (mods / '1.json').write_text(json.dumps(mods_map), encoding='utf-8')
    got = libsource.excerpts(ArgvExecutor(), py, str(mods), ['causal_conv1d_fn'])
    assert [g['module'] for g in got] == ['fakelib.conv']           # the library's, never the private files'
    assert all('private' not in g['text'] for g in got)
    if 'linked_secret' in mods_map:
        os.remove(sp / 'linked_secret.py')


# ---------------------------------------------------------------- 5. F14: the patch object over malformed edits
def test_r1_a_whole_file_patch_is_kept_when_only_its_edits_are_malformed():
    schema = P.hls3_schema(P.HYPOTHESIS_SCHEMA, {'parse': True})
    h = {'mechanism': 'route the conv to the fused kernel', 'change': {}, 'predicted_gain_pct': 8,
         'patch': {'train_patches.py': 'import fused\nfused.install()\n'},
         'edits': [{'file': 'train_patches.py', 'text': '(see patch)'}]}
    kept, dropped, fixed = P.accept_hypotheses([h], schema, {'train_patches.py': '# patches\n'})
    assert not dropped and kept[0]['patch'] == {'train_patches.py': 'import fused\nfused.install()\n'}
    assert kept[0]['edits'] is None
    assert fixed == [{'index': 0, 'mechanism': h['mechanism'], 'files': ['train_patches.py']}]
    both = {**h, 'edits': [{'file': 'train_patches.py', 'text': '(see patch)'},
                           {'file': 'train.py', 'op': 'append', 'text': 'import train_patches'}]}
    kept, dropped, _ = P.accept_hypotheses([both], schema, {'train_patches.py': '# patches\n', 'train.py': 'main()\n'})
    assert kept[0]['patch'] == {'train_patches.py': 'import fused\nfused.install()\n',
                                'train.py': 'main()\nimport train_patches\n'}   # another file's edit read from its text
    patch, ok = P.patch_of(kept[0], {'train_patches.py': '# patches\n', 'train.py': 'main()\n'})
    assert ok and set(patch) == {'train_patches.py', 'train.py'}
    unusable = {**h, 'patch': {'train_patches.py': '   '}}         # no usable patch object: dropped, as before
    assert P.accept_hypotheses([unusable], schema, {'train_patches.py': ''})[1]


def test_r1_findings_hold_for_the_shared_entry_point():
    """The same verdicts through proof.check, the gate any strategy calls (finding 2: an inherited wrapper)."""
    def events(a, b):
        return scan_events([(1.0, INSTALL.format(f'<function g at {a}>', f'<function g at {b}>')), (2.0, ACTIVE)])
    base = SimpleNamespace(overlay={}, receipt={'evidence': {'events': events('0x7f0000000a01', '0x7f0000000b01')}})
    eng = SimpleNamespace(s=SimpleNamespace(hooks=None, candidates={'c1': base, 'c0': base}))
    row = {**events('0x7f0000000c77', '0x7f0000000d88'), 'steps': 3}
    rec = proof.check(eng, 'c1', row, {'train_patches.py': 'x = 1\n'}, {})
    assert rec and rec['reason'].startswith('it printed no')


# ---------------------------------------------------------------- Codex HLS5 r2
def _wrapped():
    """One process's objects: an original function, a wrapper of it and a wrapper of that one (functools.wraps: all
    three reprs carry the original's name, only their addresses differ)."""
    import functools

    def g(x):
        return 1

    @functools.wraps(g)
    def w1(x):
        return g(x) + 1

    @functools.wraps(w1)
    def w2(x):
        return w1(x) + 1
    return g, w1, w2


def test_r2_a_wrapper_stacked_on_an_inherited_one_of_the_same_name_is_its_own():
    keep = []                                           # (every object alive: no address reused within this process)
    g, w1, _ = keep_ = _wrapped()
    keep.append(keep_)
    assert repr(g).split(' at ')[0] == repr(w1).split(' at ')[0]   # one name, two objects
    base = proof_evidence([INSTALL.format(repr(g), repr(w1)), ACTIVE], whole=True)          # original -> wrapper 1
    g2, w12, w22 = keep_ = _wrapped()                   # the next execution: other addresses
    keep.append(keep_)
    assert (w12(0), w22(0)) == (2, 3)                   # (both wrappers work)
    lines = [INSTALL.format(repr(g2), repr(w12)), INSTALL.format(repr(w12), repr(w22)), ACTIVE]   # inherited, then new
    cand = proof_evidence(lines, whole=True)
    new = proof.new_installs(cand, base)
    assert len(new) == 1 and new[0][2].endswith('[layer 2]')
    assert proof.proof_verdict(cand, base) is None      # its own installation, and it ran: proven
    alone = proof_evidence([INSTALL.format(repr(g2), repr(w12)), ACTIVE], whole=True)
    why = proof.proof_verdict(alone, base)              # the inherited one alone still proves nothing new
    assert why is not None and why.startswith('it printed no')
    assert proof.proof_verdict(proof_evidence([INSTALL.format(repr(g2), repr(w12)), ACTIVE], whole=True)) is None
    ranks = []                                          # four processes, each its own addresses: one installation per
    for _ in range(4):                                  # layer
        a, b, c = keep_ = _wrapped()
        keep.append(keep_)
        ranks += [INSTALL.format(repr(a), repr(b)), INSTALL.format(repr(b), repr(c))]
    ev = proof_evidence(ranks + [ACTIVE], whole=True)
    assert len(ev['m.f']['installs']) == 2 and len(proof.new_installs(ev, base)) == 1
    s = scan_events([(float(i), x) for i, x in enumerate(lines)])   # the scanner of a run's log reads it the same
    assert len(proof.new_installs(proof.proof_of(s, True), base)) == 1


def test_r2_malformed_edits_never_raise_and_a_sibling_survives():
    schema = P.hls3_schema(P.HYPOTHESIS_SCHEMA, {'parse': True})
    files = {'patches.txt': ''}
    usable = {'mechanism': 'a patch with junk edits', 'change': {}, 'predicted_gain_pct': 4,
              'patch': {'patches.txt': 'a\n'}, 'edits': True}
    listed = {'mechanism': 'an edit naming a list', 'change': {}, 'predicted_gain_pct': 4,
              'patch': {'patches.txt': 'b\n'}, 'edits': [{'file': ['patches.txt'], 'text': 'b'}]}
    texty = {**listed, 'mechanism': 'edits as a string', 'edits': 'see the patch'}
    sibling = {'mechanism': 'fused adam', 'change': {'fused_adam': True}, 'predicted_gain_pct': 3}
    kept, dropped, fixed = P.accept_hypotheses([usable, listed, texty, sibling], schema, files)
    assert [k['mechanism'] for k in kept] == ['a patch with junk edits', 'edits as a string', 'fused adam']
    assert kept[0]['patch'] == {'patches.txt': 'a\n'} and kept[0]['edits'] is None
    assert [d['mechanism'] for d in dropped] == ['an edit naming a list']
    from test_hls3_r3 import Recorder
    from test_search import TOGGLE, setup
    _, s, eng = setup(TOGGLE, tokens=10 ** 6)
    drv = Recorder(lambda role, system, prompt: {'verdict': 'patch', 'findings': []} if role == 'code_review'
                   else {'hypotheses': [usable, listed, sibling]})
    out = P.LLMProposer(drv, s.hooks, search_calibrates=True, features={'parse': True}).hypotheses(eng, 'c0', ledger=[])
    assert {h['mechanism'] for h in out} == {'a patch with junk edits', 'fused adam'}   # the call survives, both kept


# ---------------------------------------------------------------- Codex HLS5 r3
def test_r3_an_identity_line_never_makes_a_layer():
    keep = []
    g, w1, _ = keep_ = _wrapped()
    keep.append(keep_)
    base = proof_evidence([INSTALL.format(repr(g), repr(w1)), ACTIVE], whole=True)          # original -> wrapper 1
    g2, w12, _ = keep_ = _wrapped()
    keep.append(keep_)
    lines = [INSTALL.format(repr(g2), repr(w12)), INSTALL.format(repr(w12), repr(w12)), ACTIVE]   # inherited, then an
    cand = proof_evidence(lines, whole=True)                                                    # idempotent assignment
    why = proof.proof_verdict(cand, base)
    assert why is not None and 'already there' in why                 # as round 2: it set what was there
    assert not any('[layer' in v for r in cand.values() for _, v, *_ in r['installs'])
    ev = {**scan_events([(float(i), x) for i, x in enumerate(lines)]), 'steps': 3}
    bev = scan_events([(1.0, INSTALL.format(repr(g), repr(w1))), (2.0, ACTIVE)])
    eng = SimpleNamespace(s=SimpleNamespace(hooks=None, candidates={
        'c1': SimpleNamespace(overlay={}, receipt={'evidence': {'events': bev}})}))
    rec = proof.check(eng, 'c1', ev, {'train_patches.py': 'x = 1\n'}, {})
    assert rec and 'already there' in rec['reason']                    # the shared gate rejects it too
    ident = INSTALL.format('<function h at 0x7f0000000e01>', '<function h at 0x7f0000000e01>')
    base_id = proof_evidence([ident, ACTIVE], whole=True)                # a base whose line is an identity
    again = INSTALL.format('<function h at 0x7f0000000f02>', '<function h at 0x7f0000000f02>')
    cand_id = proof_evidence([again, again, ACTIVE], whole=True)         # printed again, twice, on the next execution
    assert proof.new_installs(cand_id, base_id) == []
    assert proof.proof_verdict(cand_id, base_id) is not None
