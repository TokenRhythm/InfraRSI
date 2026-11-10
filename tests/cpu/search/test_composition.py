"""Composition of within-noise gains (Fable decision 2.A-2.D): bundle bookkeeping and the split of failed bundles,
carrying unresolved positives to a new incumbent, the paired confirmation of near-threshold composites, rescore
accounting, and the per-file three-way merge of code patches."""
import math
from types import SimpleNamespace

import pytest

from infrarsi.search.engine import Outcome, Proposal
from infrarsi.search.strategies import HLS, merge3, merge_patches
from test_search import TOGGLE, setup


class FakeProposer:
    def __init__(self, hyps):
        self.hyps = hyps

    def describe(self):
        return {}

    def hypotheses(self, engine, base, ledger=None):
        return [dict(h) for h in self.hyps]


class FakeEngine:
    def __init__(self):
        cands = {'c0': SimpleNamespace(config={'a': 0, 'b': 0, 'c': 0, 'd': 0, 'x': 0}, overlay={}),
                 'c9': SimpleNamespace(config={'a': 0, 'b': 0, 'c': 0, 'd': 0, 'x': 1}, overlay={})}
        self.s = SimpleNamespace(candidates=cands, hooks=None)
        self._inc = 'c0'

    def incumbent(self):
        return self._inc

    def sigma_pair(self):
        return 0.02

    def sigma_run(self):
        return 0.015


def hyp(name, mu=0.03):
    return {'mechanism': name, 'change': {name: 1}, 'patch': None, 'mu': mu, 'sigma': 0.02, 'p_feasible': 0.9}


def screen(hls, eng, mech, gain, cid):
    p = next(p for p in hls.propose(eng, 16) if p.mechanism == mech)
    hls.observe(eng, p, Outcome(cid, 'screened', True, 1.0, gain, 'full', 0.1, [], []))


def bundle_of(hls, eng):
    return next((p for p in hls.propose(eng, 16) if p.origin == 'hls-bundle' and p.hypothesis not in hls.pending), None)


def test_moved_bundle_retires_its_members():
    eng, hls = FakeEngine(), HLS(FakeProposer([hyp(n) for n in 'abcd']), explore=False)
    for i, n in enumerate('abcd'):
        screen(hls, eng, n, 0.01 + 0.001 * i, f'c{i + 1}')          # positive, below 2 sigma_pair: unresolved
    b = bundle_of(hls, eng)
    assert b.change == {'a': 1, 'b': 1, 'c': 1, 'd': 1} and b.confirm and b.mu == pytest.approx(0.046)
    eng._inc = 'c7'
    hls.observe(eng, b, Outcome('c7', 'screened', True, 1.0, 0.05, 'full', 0.1, [], []))
    assert {h['status'] for h in hls.ledger.values()} == {'bundled'} and hls.bundles == 1


def test_failed_bundle_is_split_and_not_repeated():
    eng, hls = FakeEngine(), HLS(FakeProposer([hyp(n) for n in 'abcd']), explore=False)
    for i, n in enumerate('abcd'):
        screen(hls, eng, n, 0.01 + 0.001 * i, f'c{i + 1}')
    b = bundle_of(hls, eng)
    hls.observe(eng, Proposal(), Outcome(None, 'duplicate', False, None, None, 'none', 0.0, [], []))
    hls.observe(eng, b, Outcome(None, 'duplicate', False, None, None, 'none', 0.0, [], []))
    assert hls.bundles == 0                                          # duplicates are not evaluations
    hls.observe(eng, b, Outcome('c5', 'screened', True, 1.0, -0.01, 'full', 0.1, [], []))
    halves = sorted(sorted(p.change) for p in hls.pending.values())
    assert halves == [['a', 'c'], ['b', 'd']]                        # alternating by measured gain (d > c > b > a)
    nxt = bundle_of(hls, eng)
    assert sorted(nxt.change) == ['b', 'c', 'd']                     # the failed set's weakest member left out
    props = hls.propose(eng, 16)
    assert sum(1 for p in props if p.origin == 'hls-bundle') == 3    # the 3-member set and the two halves


def test_failed_pair_is_excluded():
    eng, hls = FakeEngine(), HLS(FakeProposer([hyp('a'), hyp('b')]), explore=False)
    screen(hls, eng, 'a', 0.01, 'c1')
    screen(hls, eng, 'b', 0.012, 'c2')
    b = bundle_of(hls, eng)
    hls.observe(eng, b, Outcome('c3', 'invalid', False, None, None, 'full', 0.1, ['oom'], ['completes']))
    assert bundle_of(hls, eng) is None and not hls.pending
    assert hls.ledger['c0|a']['failed_checks'] == ['completes']


def test_unresolved_positive_is_carried_to_a_new_incumbent():
    eng, hls = FakeEngine(), HLS(FakeProposer([hyp('a'), hyp('b')]), explore=False)
    screen(hls, eng, 'a', 0.01, 'c1')
    eng._inc = 'c9'                                                  # another change moved the incumbent
    props = hls.propose(eng, 16)
    carried = hls.ledger['c9|a']
    assert carried['status'] == 'unresolved' and carried['rebased'] and carried['measured'] == 0.01
    assert [p.mechanism for p in props if p.origin == 'hls'] == ['b']   # 'a' is not measured again
    assert len(hls.pairs) == 1                                       # no new calibration data


def test_carry_off_keeps_the_old_behaviour():
    eng, hls = FakeEngine(), HLS(FakeProposer([hyp('a'), hyp('b')]), explore=False, carry=False)
    screen(hls, eng, 'a', 0.01, 'c1')
    eng._inc = 'c9'
    props = hls.propose(eng, 16)
    assert sorted(p.mechanism for p in props if p.origin == 'hls') == ['a', 'b']


def test_rescore_is_charged_like_a_trial():
    sim, s, eng = setup(TOGGLE)
    out = eng.evaluate(Proposal(change={'fused_adam': True}, mu=0.02, abortable=False))
    c = s.candidates[out.cid]
    screens, trials, gpu = s.ledger.used['screens'], s.ledger.used.get('trials', 0), s.ledger.used['gpu_hours']
    verdict, status = dict(c.verdict), c.status
    r = s.rescore(out.cid)
    assert r['valid'] and r['step_time']
    assert s.ledger.used['screens'] == screens and s.ledger.used['trials'] == trials + 1
    assert s.ledger.used['gpu_hours'] > gpu and c.verdict == verdict and c.status == status
    assert len(c.results['rescore']) == 1 and s.screens[-1]['fidelity'] == 'rescore'
    s.ledger.used['gpu_hours'] = s.budget.gpu_hours * 0.95
    assert 'budget' in s.rescore(out.cid)['error'] and not s.stop_flag


def test_confirm_moves_only_when_the_pair_agrees(monkeypatch):
    sim, s, eng = setup(TOGGLE)
    out = eng.evaluate(Proposal(change={'fused_adam': True}, mu=0.02, abortable=False))
    cid, t_inc = out.cid, eng.t('c0')
    t_c = eng.t(cid)
    d1 = 2.5 * eng.sigma_run()
    eng._inc = 'c0'
    monkeypatch.setattr(s, 'rescore', lambda c: {'valid': True, 'step_time': t_inc if c == 'c0' else t_inc * math.exp(-d1)})
    assert eng.confirm(cid, d1) and eng.incumbent() == cid
    assert eng.t(cid) == pytest.approx(math.sqrt(t_c * t_inc * math.exp(-d1)))
    eng._inc = 'c0'
    eng.t_conf.clear()
    monkeypatch.setattr(s, 'rescore', lambda c: {'valid': True, 'step_time': t_inc if c == 'c0' else t_inc * math.exp(d1)})
    assert not eng.confirm(cid, d1) and eng.incumbent() == 'c0'
    assert [r['moved'] for r in eng.confirmations] == [True, False]


def test_merge_patches_three_way():
    base = 'a\nb\nc\nd\ne\n'
    x, y, z = 'a\nB\nc\nd\ne\n', 'a\nb\nc\nD\ne\n', 'a\nb2\nc\nd\ne\n'
    assert merge_patches({'f.py': base}, [{'f.py': x}, {'f.py': y}]) == {'f.py': 'a\nB\nc\nD\ne\n'}
    assert merge_patches({'f.py': base}, [{'f.py': x}, {'f.py': z}]) is None
    assert merge_patches({'f.py': base}, [{'f.py': x}, {'f.py': x}, {'g.py': 'new\n'}]) == {'f.py': x, 'g.py': 'new\n'}


def test_bundled_rows_reach_the_cross_run_priors(tmp_path, monkeypatch):
    from infrarsi.project.flow import prior_results, remember
    from infrarsi.project.manifest import Project
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    p = Project({'name': 'p', 'launch': {'command': '{python} t.py {args}'}, 'knobs': {'k': {'type': 'bool', 'flag': '--k'}},
                 'workload': {'steps': 4, 'window': [2, 4]}}, tmp_path / 'infrarsi.yaml')
    remember(p, {'extra': {'ledger': [{'mechanism': 'a', 'status': 'bundled', 'measured_gain_pct': 1.2}]}})
    assert [r['status'] for r in prior_results(p)] == ['bundled']


REAL = '''import torch


def make_loader(ds, batch_size):
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=2,
                                       pin_memory=False, shuffle=True)


def make_optimizer(model, lr):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)


def train_step(model, batch, opt):
    loss = model(**batch).loss
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss.item()
'''
LOADER = REAL.replace('num_workers=2,\n                                       pin_memory=False',
                      'num_workers=8,\n                                       pin_memory=True')
FUSED = REAL.replace('weight_decay=0.1)', 'weight_decay=0.1, fused=True)')
NOSYNC = REAL.replace('    return loss.item()', '    return loss.detach()')
SGD = REAL.replace('torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)', 'torch.optim.SGD(model.parameters(), lr=lr)')


def test_merge_of_real_patch_text():
    both = merge_patches({'train.py': REAL}, [{'train.py': LOADER}, {'train.py': FUSED}, {'train.py': NOSYNC}])
    want = LOADER.replace('weight_decay=0.1)', 'weight_decay=0.1, fused=True)').replace('    return loss.item()',
                                                                                        '    return loss.detach()')
    assert both == {'train.py': want}
    assert merge_patches({'train.py': REAL}, [{'train.py': FUSED}, {'train.py': SGD}]) is None     # same line, two edits
    # rebase: an edit written against the old base, applied onto an incumbent that already carries another edit
    assert merge_patches({'train.py': REAL}, [{'train.py': LOADER}, {'train.py': FUSED}])['train.py'] == \
        LOADER.replace('weight_decay=0.1)', 'weight_decay=0.1, fused=True)')
    assert merge_patches({'train.py': REAL}, [{'train.py': FUSED}, {'train.py': FUSED}]) == {'train.py': FUSED}


def test_bundle_leaves_out_a_conflicting_patch():
    eng = FakeEngine()
    eng.s.hooks = SimpleNamespace(candidate_files=lambda overlay: {'train.py': REAL, **(overlay or {})})
    hyps = [dict(hyp('a'), patch={'train.py': FUSED}), dict(hyp('b'), patch={'train.py': SGD}),
            dict(hyp('c'), patch={'train.py': LOADER})]
    hls = HLS(FakeProposer(hyps), explore=False)
    for i, (n, g) in enumerate((('a', 0.012), ('b', 0.011), ('c', 0.010))):
        screen(hls, eng, n, g, f'c{i + 1}')
    b = bundle_of(hls, eng)
    assert sorted(b.change) == ['a', 'c'] and b.patch == {'train.py': LOADER.replace('weight_decay=0.1)',
                                                                                      'weight_decay=0.1, fused=True)')}


# ---------------------------------------------------------------- review R3b (gpt-6.1-sol)
def test_merge3_does_not_guess_among_repeated_lines():
    base = 'x = 0\nx = 0\nx = 0\n'
    assert merge3(base, 'x = 1\nx = 0\nx = 0\n', 'x = 0\nx = 2\nx = 0\n') is None   # the diff cannot place the edits
    assert merge3('a\nb\nc\n', 'A\nb\nc\n', 'a\nB\nc\n') is None                   # adjacent lines conflict (as diff3)
    assert merge3('a\nx\nb\nx\nc\n', 'A\nx\nb\nx\nc\n', 'a\nx\nb\nx\nC\n') == 'A\nx\nb\nx\nC\n'
    src = 'def f():\n    pass\n\n\ndef g():\n    pass\n\n\ndef h():\n    return 1\n'
    ins = src.replace('\n\n\ndef g', '\n\n\ndef new():\n    pass\n\n\ndef g')            # an insertion among blank lines
    assert merge3(src, ins, src.replace('return 1', 'return 2')) == ins.replace('return 1', 'return 2')


def test_bundle_leaves_out_another_value_of_the_same_knob():
    eng = FakeEngine()
    hls = HLS(FakeProposer([dict(hyp('tp1'), change={'tp': 1}), dict(hyp('tp4'), change={'tp': 4}), hyp('c')]),
              explore=False)
    for i, (n, g) in enumerate((('tp1', 0.012), ('tp4', 0.011), ('c', 0.010))):
        screen(hls, eng, n, g, f'c{i + 1}')
    b = bundle_of(hls, eng)
    assert b.change == {'tp': 1, 'c': 1} and hls.members[b.hypothesis] == ['c0|tp1', 'c0|c']
    eng._inc = 'c7'
    hls.observe(eng, b, Outcome('c7', 'screened', True, 1.0, 0.05, 'full', 0.1, [], []))
    assert hls.ledger['c0|tp4']['status'] == 'unresolved'                    # never applied: not retired
    assert hls.ledger['c0|c']['bundle_candidate'] == 'c7'


def test_duplicate_half_is_not_proposed_again():
    eng, hls = FakeEngine(), HLS(FakeProposer([hyp(n) for n in 'abcd']), explore=False)
    for i, n in enumerate('abcd'):
        screen(hls, eng, n, 0.01 + 0.001 * i, f'c{i + 1}')
    hls.observe(eng, bundle_of(hls, eng), Outcome('c5', 'screened', True, 1.0, -0.01, 'full', 0.1, [], []))
    half = next(iter(hls.pending.values()))
    hls.observe(eng, half, Outcome(None, 'duplicate', False, None, None, 'none', 0.0, ['already tried'], []))
    assert half.hypothesis not in hls.pending and hls.bundles == 1
    assert all(p.hypothesis != half.hypothesis for p in hls.propose(eng, 16))


def test_rescore_that_spends_the_rest_of_the_budget_stops_the_search(monkeypatch):
    sim, s, eng = setup(TOGGLE)
    out = eng.evaluate(Proposal(change={'fused_adam': True}, mu=0.02, abortable=False))
    monkeypatch.setattr(s, '_admission', lambda: (None, None))            # admitted on the estimate ...
    s.ledger.used['gpu_hours'] = s.budget.gpu_hours * 0.99999              # ... but the run itself uses up the budget
    assert s.rescore(out.cid)['valid'] and (s.stop_flag or '').startswith('budget:')


def test_merge3_rejects_an_edit_the_diff_could_place_on_another_repeated_line():
    L, X = '    nn.Linear(16, 16),\n', '    nn.Linear(16, 16, bias=False),\n'
    base = 'model = nn.Sequential(\n' + L * 6 + ')\n'
    rows = base.splitlines(keepends=True)
    rows[3] = '    nn.ReLU(),\n'                                                     # the third layer
    a = base.replace(L, X, 1).replace(')\n', ').cuda()\n')
    assert merge3(base, a, ''.join(rows)) is None                                   # not a ReLU on the fourth layer
    five = 'model = nn.Sequential(\n' + L * 5 + ')\n'
    rows = five.splitlines(keepends=True)
    rows[1] = X
    del rows[5]                                                                     # first layer changed, last deleted
    gelu = 'model = nn.Sequential(\n' + L * 3 + '    nn.GELU(),\n' + L * 2 + ')\n'   # a GELU after the third layer
    assert merge3(five, ''.join(rows), gelu) is None and merge3(five, gelu, ''.join(rows)) is None
    marked = 'model = nn.Sequential(\n' + L * 3 + '    # head\n' + L * 3 + ')\n'    # a unique line between the edits
    one = marked.replace(L, X, 1)
    assert merge3(marked, one, marked.replace(')\n', ').cuda()\n')) == one.replace(')\n', ').cuda()\n')
