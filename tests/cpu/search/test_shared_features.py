"""The round-3 pieces that serve any strategy that opts in, not hls3 alone (the user's decisions 2026-10-31: the knob-side
search of hls3 is replaced by a new strategy; F13 / F14 / F21 and D7 serve its code track, D4 any arm): the proposer's
`features` (proposers.FEATURES) asked by a strategy class's proposer_features() - F14 robust parsing, the F10 / F13 / F21
patch self-proof (search/proof.py, for any strategy's Proposal gate), D7 read-only library source (search/library.py,
project/libsource.py) - and D4, what a strategy learned of its starts written back for a later arm (experience/learned.py).
hls and hls2 ask for none of it: their proposer is the one they had."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hls3_helpers import FIX
from infrarsi.project import libsource
from infrarsi.project.signals import scan_events
from infrarsi.search import _engine, library, proof
from infrarsi.search import proposers as P
from infrarsi.search.engine import Strategy
from infrarsi.search.hls3 import HLS3, proposer_flags
from infrarsi.search.strategies import DOE, HLS, HLS2, Greedy
from test_composition import FakeProposer
from test_hls3_r3 import Recorder
from test_hls3_r3_replay import Q9H, gate_q9h, probe, row_of

SHARED = {'parse', 'self_proof', 'deferred', 'library_source'}


class Mine(HLS):
    """A strategy other than hls3 that opts in to every shared feature."""
    name = 'mine'

    @classmethod
    def proposer_features(cls, off=()) -> dict:
        return {k: k not in set(off) for k in SHARED}


# ---------------------------------------------------------------- opting in
def test_the_features_are_asked_for_by_the_strategy_class():
    assert set(P.FEATURES) == SHARED
    for cls in (Strategy, HLS, HLS2, DOE, Greedy):
        assert cls.proposer_features() == {} and cls.proposer_features(['hls3_parse']) == {}
    f = HLS3.proposer_features()
    assert f == proposer_flags() and all(f[k] for k in SHARED)
    assert not HLS3.proposer_features(['hls3_parse'])['parse']
    assert not HLS3.proposer_features(['hls3_library_source'])['library_source']
    p = P.LLMProposer(None, SimpleNamespace(), features=Mine.proposer_features())
    assert p.flags == {k: True for k in SHARED} and p.hls3 is p.flags              # (its older name)
    assert p._hls3_rules(True) == P.PROOF_RULE + P.PROOF_DEFER_RULE and p._hls3_rules(False) == ''
    items = P.feature_schema(P.HYPOTHESIS_SCHEMA, p.flags)['properties']['hypotheses']['items']['properties']
    assert {'evidence_note', 'active_from_update'} <= set(items) and 'memory' not in items   # hls3's own fields: not asked
    assert P.LLMProposer(None, SimpleNamespace()).flags == {}
    assert P.LLMProposer(None, SimpleNamespace(), features=Mine.proposer_features(['self_proof'])).flags == \
        {'parse': True, 'deferred': True, 'library_source': True}


def test_the_factory_gives_each_strategy_the_features_its_class_asks_for(monkeypatch):
    seen = []

    class Stop(Exception):
        pass

    def fake(*a, **kw):
        seen.append(kw['features'])
        raise Stop
    monkeypatch.setattr(P, 'LLMProposer', fake)
    ctx = SimpleNamespace(driver=object(), hooks=SimpleNamespace(), start_receipt=None,
                          options={'full_steps': 10, 'job_gpu_hours': 1.0, 'ablate': ['hls3_parse']})
    for name in ('hls', 'hls2', 'doe+hls', 'greedy', 'hls3'):
        with pytest.raises(Stop):
            _engine(name)(ctx)
    assert seen == [{}, {}, {}, {}, proposer_flags(['hls3_parse'])]


# ---------------------------------------------------------------- F14 for any strategy
def _script(seq):
    good = {'mechanism': 'fused adam', 'change': {'fused_adam': True}, 'predicted_gain_pct': 3}
    bad = {'mechanism': 'bad', 'change': 'fused_adam=1', 'predicted_gain_pct': 3}
    answers = {'mixed': {'hypotheses': [good, bad]}}
    seq = list(seq)
    return lambda role, system, prompt: {'verdict': 'no_patch', 'findings': []} if role == 'code_review' else \
        answers[seq.pop(0) if seq else 'mixed']


def test_f14_robust_parsing_for_a_strategy_that_opts_in():
    from test_search import TOGGLE, setup
    _, s, eng = setup(TOGGLE, tokens=10 ** 6)
    drv = Recorder(_script(['mixed', 'mixed']))
    prop = P.LLMProposer(drv, s.hooks, search_calibrates=True, features=Mine.proposer_features())
    assert [x['mechanism'] for x in prop.hypotheses(eng, 'c0', ledger=[])] == ['fused adam']   # the bad one dropped
    assert drv.sent[0]['join_text'] and drv.sent[0]['retry_note'] is P.explain_reply
    prop.hypotheses(eng, 'c0', ledger=[])
    shown = json.loads(drv.sent[1]['prompt'])['dropped_hypotheses']      # the next call is told, with no context
    assert shown[0]['mechanism'] == 'bad' and shown[0]['call'] == 1 and prop.context is None
    assert prop.describe()['parse'] == {'joined': 0, 'dropped': 2, 'rewritten': 0}
    _, s, eng = setup(TOGGLE, tokens=10 ** 6)
    drv = Recorder(_script(['mixed', 'mixed']))
    old = P.LLMProposer(drv, s.hooks, search_calibrates=True)            # hls / hls2: the answer whole, as before
    assert old.hypotheses(eng, 'c0', ledger=[]) == [] and old.failed == 1 and 'parse' not in old.describe()
    old.hypotheses(eng, 'c0', ledger=[])
    assert 'dropped_hypotheses' not in json.loads(drv.sent[1]['prompt']) and drv.sent[0]['join_text'] is None


# ---------------------------------------------------------------- F10 / F13 / F21 for any strategy
def recorded(name: str, cid: str) -> tuple[dict, dict]:
    """A recorded candidate and the hypothesis it was proposed from (as test_hls3_r3_replay.gate_q9h finds it)."""
    arm = FIX[name]
    c = next(x for x in arm['candidates'] if x['id'] == cid)
    h = next((dict(f) for call in arm['calls'] if call['role'] in ('hypotheses', 'code_review')
              for f in call.get('hypotheses') or call.get('findings') or []
              if (f.get('mechanism') or '') == c.get('rationale') or str(c.get('rationale') or '').startswith('review:')
              and f.get('evidence_note') and (f.get('issue') or '')[:60] in str(c.get('rationale') or '')), {})
    return c, h


def at_probe(cid: str, extra=()) -> tuple:
    """(engine, base, row, patch, h) of a Q9H-h5 patch at its probe, for proof.check: its base's run's events, its own
    probe's detail row (with `extra` lines), the patch's code and hypothesis - no strategy."""
    pr = probe(Q9H, cid)
    c, h = recorded(Q9H, cid)
    bev = scan_events([(t, x) for t, x in pr['base_lines']])
    cand = SimpleNamespace(overlay={}, receipt={'evidence': {'events': {k: v for k, v in bev.items() if k.startswith('perf')
                                                                         or k.startswith('patch')}}})
    eng = SimpleNamespace(s=SimpleNamespace(hooks=None, candidates={pr['base']: cand, 'c0': cand}))
    lines = sorted(pr['lines'] + [list(x) for x in extra])
    return eng, pr['base'], row_of(lines, pr['steps'] or 2), {'train_patches.py': '\n'.join(c.get('patch_lines') or [])}, h


def test_the_patch_proof_for_any_strategys_gate():
    eng, base, row, patch, h = at_probe('c2')                     # F13: proven when its 399-character line is read whole
    assert h.get('evidence_note') and proof.check(eng, base, row, patch, h, whole=True) is None
    cut = proof.check(eng, base, row, patch, h, whole=False)
    assert cut['reason'].startswith('it printed no') and 'parsed' not in cut and cut['lines']
    assert gate_q9h('c2', probe(Q9H, 'c2')['lines'])[0] is None    # hls3's gate: the same verdict
    for cid in ('c5', 'c8'):                                       # F21: a deferral line names why
        pr = probe(Q9H, cid)
        site = next(x.split(']', 1)[1].split(': found', 1)[0].strip() for _, x in pr['lines'] if ': found' in x)
        defer = (pr['lines'][-1][0] - 1.0, f'[infrarsi-patch] {site}: deferred (kwargs non-empty)')
        eng, base, row, patch, h = at_probe(cid, [defer])
        rec = proof.check(eng, base, row, patch, h)
        assert 'kwargs non-empty' in rec['reason'] and rec['parsed'][site]['deferred'] == 'kwargs non-empty'
        assert rec['reason'] == gate_q9h(cid, sorted(pr['lines'] + [list(defer)]))[0][0]   # hls3's reason, the same text
        assert 'not the code the job runs' in proof.check(eng, base, row, patch, h, deferred=False)['reason']


# ---------------------------------------------------------------- D7: read-only library source
class ArgvExecutor:
    """An executor whose probe runs the command's python with the quoted arguments as its argv (no shell: any OS), as
    the ssh executor runs it with the job's python on its host."""

    def __init__(self):
        self.quoted, self.probes = [], 0

    def quote(self, s: str) -> str:
        self.quoted.append(s)
        return f'@@{len(self.quoted) - 1}@@'

    def python(self, configured):
        return configured or sys.executable

    def probe(self, cmd: str, *, cwd: str, env: dict, timeout_s: float = 30):
        self.probes += 1
        py, rest = cmd.split(' -c ', 1)
        argv = [self.quoted[int(t.strip('@'))] if t.startswith('@@') else t for t in rest.split()]
        return subprocess.run([py, '-c'] + argv, capture_output=True, text=True, cwd=cwd, timeout=timeout_s).stdout


LIB = '''import functools


def _cached(f):
    return f


@_cached
def causal_conv1d_fn(x, weight, bias=None, **kwargs):
    """The reference fallback."""
    if kwargs:
        return x
    return x


class GatedMixer:
    def forward(self, x, **kwargs):
        return causal_conv1d_fn(x, None, **kwargs)

    def big(self):
''' + ''.join(f'        v{i} = {i}\n' for i in range(200))


@pytest.fixture(scope='module')
def jobpy(tmp_path_factory):
    """The job's interpreter (a real virtual environment) with a library in its site-packages and a file there the job
    does not import: (its python, its site-packages)."""
    from jobenv import make_venv
    py, sp = make_venv(tmp_path_factory.mktemp('job') / 'venv')
    (sp / 'fakelib').mkdir()
    (sp / 'fakelib' / '__init__.py').write_text('', encoding='utf-8')
    (sp / 'fakelib' / 'conv.py').write_text(LIB, encoding='utf-8')
    (sp / 'other.py').write_text('def causal_conv1d_fn():\n    return "not imported"\n', encoding='utf-8')
    return py, sp


@pytest.fixture
def env(tmp_path, jobpy):
    """A job's environment: the library in its interpreter's site-packages it imported, a file of its own it imported,
    and the module list of its reference run (the modules dir)."""
    sp = jobpy[1]
    (tmp_path / 'job').mkdir()
    (tmp_path / 'job' / 'train.py').write_text('def causal_conv1d_fn():\n    return "the job\'s own"\n', encoding='utf-8')
    mods = tmp_path / 'run' / 'modules'
    mods.mkdir(parents=True)
    (mods / '123.json').write_text(json.dumps({'fakelib': str(sp / 'fakelib' / '__init__.py'),
                                               'fakelib.conv': str(sp / 'fakelib' / 'conv.py'),
                                               'train': str(tmp_path / 'job' / 'train.py')}), encoding='utf-8')
    return mods


def test_d7_the_excerpts_are_the_named_regions_of_imported_library_files_read_with_the_jobs_python(env, jobpy):
    exe, py = ArgvExecutor(), jobpy[0]
    got = libsource.excerpts(exe, py, str(env), ['fakelib.conv.causal_conv1d_fn', 'GatedMixer.forward',
                                                 'causal_conv1d_fn', 'GatedMixer.big'])
    by = {(g['name'], g['qualname']): g for g in got}
    conv = by[('fakelib.conv.causal_conv1d_fn', 'causal_conv1d_fn')]
    assert conv['text'].startswith('@_cached\ndef causal_conv1d_fn(x, weight, bias=None, **kwargs):') and \
        conv['module'] == 'fakelib.conv' and conv['file'].endswith('conv.py')
    assert by[('GatedMixer.forward', 'GatedMixer.forward')]['text'].strip().startswith('def forward(self, x, **kwargs)')
    assert ('causal_conv1d_fn', 'causal_conv1d_fn') not in by          # the same region once; not other.py, not train.py
    assert all('site-packages' in g['file'] and 'other.py' not in g['file'] for g in got)
    big = by[('GatedMixer.big', 'GatedMixer.big')]
    assert len(big['text'].splitlines()) == libsource.PER_NAME + 1 and big['text'].endswith('more lines')
    assert sum(min(len(g['text'].splitlines()), libsource.PER_NAME) for g in got) <= libsource.MAX_LINES
    small = libsource.excerpts(exe, py, str(env), ['GatedMixer.big', 'GatedMixer.forward'], max_lines=50)
    assert len(small) == 1 and len(small[0]['text'].splitlines()) == 51   # 50 lines in all, the rest counted
    assert libsource.excerpts(SimpleNamespace(quote=str, probe=lambda *a, **k: 'no marker'), 'py', 'd', ['x']) is None


def _engine_at(env, exe, notes, py=sys.executable):
    c0 = SimpleNamespace(overlay={}, receipt={'evidence': {'modules_dir': str(env), 'events': {'perf_notes': notes}}})
    runner = SimpleNamespace(executor=exe, project=SimpleNamespace(raw={'execution': {'python': py}}))
    return SimpleNamespace(s=SimpleNamespace(runner=runner, hooks=None, candidates={'c0': c0}))


def test_d7_any_strategy_reads_it_at_the_names_of_the_notes_and_sites(env, jobpy):
    exe = ArgvExecutor()
    eng = _engine_at(env, exe, ['`causal_conv1d_fn` is falling back to its reference PyTorch implementation'], jobpy[0])
    assert library.names(eng, 'c0', ['GatedMixer.forward']) == ['causal_conv1d_fn', 'GatedMixer.forward']
    cache = {}
    got = library.source(eng, 'c0', ['GatedMixer.forward'], cache)
    assert got['note'] == libsource.NOTE and 'read-only environment text' in got['note']
    assert [g['qualname'] for g in got['excerpts']] == ['causal_conv1d_fn', 'GatedMixer.forward']
    assert library.source(eng, 'c0', ['GatedMixer.forward'], cache) == got and exe.probes == 1   # once per set of names
    assert library.source(_engine_at(env, SimpleNamespace(), ['`causal_conv1d_fn`']), 'c0') is None   # no probe
    assert library.source(_engine_at(env, exe, ['no code name here']), 'c0') is None


def test_d7_the_proposer_shows_it_next_to_the_code_for_a_strategy_that_opts_in(monkeypatch):
    from test_search import TOGGLE, setup
    lib = {'note': libsource.NOTE, 'names': ['causal_conv1d_fn'], 'excerpts': [{'name': 'causal_conv1d_fn', 'text': 'x'}]}
    asked = []
    monkeypatch.setattr(library, 'source', lambda engine, base, sites=(), cache=None: asked.append(base) or lib)
    for features, ctx, has in (({'library_source': True}, None, True),                       # its own read
                               ({'library_source': True}, {'library_source': None}, False),  # the strategy's: none
                               ({}, {'library_source': lib}, False)):                        # not opted in
        _, s, eng = setup(TOGGLE, tokens=10 ** 6)
        drv = Recorder(lambda role, system, prompt: {'verdict': 'patch', 'findings': []} if role == 'code_review'
                       else {'hypotheses': []})
        prop = P.LLMProposer(drv, s.hooks, features=features)
        if ctx is not None:
            prop.context = lambda engine, base, ctx=ctx: dict(ctx)
        prop.hypotheses(eng, 'c0', ledger=[])
        calls = {c['role']: c for c in drv.inner.calls}
        payload = json.loads(calls['hypotheses']['prompt'])
        assert ('library_source' in payload) is has and (P.LIBRARY_RULE in calls['hypotheses']['system']) is has
        assert (P.LIBRARY_RULE in calls['code_review']['system']) is has
        if has:
            assert payload['library_source'] == lib and 'code' in payload
    assert asked == ['c0', 'c0']                                    # the review's call and the hypotheses' call


# ---------------------------------------------------------------- D4: what an arm learned, for any strategy
def test_d4_any_strategy_writes_back_what_it_learned_and_a_later_arm_starts_from_it(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'experience'))
    from expjob import KNOBS, REF, megatron_log, project
    from test_transfer_and_write_back import trace
    from infrarsi.experience.learned import SOURCE, section
    from infrarsi.experience.pack import read_lines, validate
    from infrarsi.experience.runs import plan_for, summary, write_back
    from infrarsi.experience.store import Store, fingerprint, plan
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    proj = project()
    pl = plan_for(proj, {'log': str(megatron_log(tmp_path / 'train.log'))}, log=lambda m: None)
    learned = section(
        failed_together=[{'members': [{'micro_batch_size': 2}, {'recompute_activations': False}], 'kind': 'start_cap',
                          'run': 'c4', 'separately': {'micro_batch_size=2': 'valid'}}],
        pairs=[{'members': [{'overlap_grad_reduce': True}, {'NCCL_NVLS_ENABLE': '0'}], 'kind': 'compiler', 'run': 'c3',
                'strikes': 2}],
        alone=[{'members': [{'fp8': '1'}], 'kind': 'start'}])
    tr = trace()
    tr['extra']['strategy'] = {'name': 'mine', 'learned': learned}  # a strategy other than hls3
    path = write_back(proj, pl, tr, None, 'r1', 0.0012, summary(pl, tr, 0.0012))
    lines = read_lines(path)
    cons = [x for x in lines if x['kind'] == 'constraint' and x.get('source') == SOURCE]
    assert validate(lines)[0] == [] and [c['learned'] for c in cons] == ['failed_together', 'pairs', 'alone']
    assert 'hls3' not in json.dumps(cons) and cons[0]['evidence'] == ['run-r1/c4'] and cons[0]['symptom'] == 'crash'
    st = Store.load(include=['run-*'])
    again = plan(st, fingerprint(proj, {'log': str(tmp_path / 'train.log')}), KNOBS, REF)
    starts = {x['learned']: x['members'] for x in again['start_failures']}
    assert starts == {'failed_together': [{'micro_batch_size': 2}, {'recompute_activations': False}],
                      'pairs': [{'overlap_grad_reduce': True}, {'NCCL_NVLS_ENABLE': '0'}], 'alone': [{'fp8': '1'}]}
    h = HLS3(FakeProposer([]), experience=again)                   # an arm that reads the same store starts from them
    pairs = {frozenset(k for m in pr for k, _ in m) for pr, info in h.suspects.items() if info.get('from_store')}
    assert pairs == {frozenset({'micro_batch_size', 'recompute_activations'}),
                     frozenset({'overlap_grad_reduce', 'NCCL_NVLS_ENABLE'})}
    assert len(h.from_store) == 2 and h.describe()['hls3']['r3']['from_store']
    off = HLS3(FakeProposer([]), experience=again, off=['hls3_writeback'])
    assert not off.suspects and not off.from_store                  # (switched off: as before)
    tr['extra']['strategy'] = {'name': 'hls', 'experience': {'seeded': 2}}   # a strategy that learned nothing: as before
    path = write_back(proj, pl, tr, None, 'r2', 0.0012, summary(pl, tr, 0.0012))
    assert not [x for x in read_lines(path) if x['kind'] == 'constraint']


def test_d4_hls3_writes_the_same_section():
    h = HLS3(FakeProposer([]))
    h.together = [{'members': [frozenset({('mbs', '2')}), frozenset({('compile', 'true')})], 'kind': 'start_cap',
                   'candidate': 'c5', 'separately': {frozenset({('mbs', '2')}): 'valid'}}]
    h.alone_fails = {('liger', 'true')}
    d = h.describe()
    assert d['learned'] == {'failed_together': [{'members': [{'mbs': 2}, {'compile': True}], 'kind': 'start_cap',
                                                 'run': 'c5', 'separately': {'mbs=2': 'valid'}}],
                            'alone': [{'members': [{'liger': True}], 'kind': 'start'}]}
    assert 'writeback' not in d['hls3'] and 'learned' not in HLS3(FakeProposer([]), off=['hls3_writeback']).describe()
