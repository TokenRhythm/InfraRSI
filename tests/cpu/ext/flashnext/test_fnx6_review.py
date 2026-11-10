"""Regression tests for the Codex review round of X6's code (GPT-5.6 Sol, xhigh, 2026-10-21; findings 1-14): each case is
a concrete input that the reviewed code mishandled."""
import copy
import json
import shutil

import pytest

import x6_helpers as H
from infrarsi.agent.backends import ScriptedBackend
from infrarsi.agent.budget import VirtualClock
from infrarsi.ext.flashnext import arms
from infrarsi.ext.flashnext import cli
from infrarsi.ext.flashnext import guards as G
from infrarsi.ext.flashnext import judge as J
from infrarsi.ext.flashnext import parse as P
from infrarsi.ext.flashnext import switch_space as S
from infrarsi.ext.flashnext import task as T
from infrarsi.ext.flashnext import variant as V
from infrarsi.ext.flashnext.hooks import FlashNextHooks
from infrarsi.harness import signing

START = S.start_point()
SMATCH = S.registered_points()['S_match']


@pytest.fixture(scope='module')
def uc(tmp_path_factory):
    return H.uc_reference_receipt(tmp_path_factory.mktemp('uc'), clock=VirtualClock())


def _screen(tmp_path, point=SMATCH, prof=None, **kw):
    return H.write_screen(tmp_path / 'scr', prof or H.profile_of(H.uc_a()), point, **kw)


def _ids(g):
    return [r['id'] for r in g.rejections]


# ---------------------------------------------------------------- 1: collect never raises
@pytest.mark.parametrize('mutate', [
    lambda root: _edit(root / 'attempt.json', lambda a: a.update(plan='torn-shape')),
    lambda root: _edit(root / 'attempt.json', lambda a: a.update(resume='0')),
    lambda root: _edit(root / 'job.json', lambda j: j.update(env=['FLASH_STOP=1024'])),
    lambda root: _edit(root / 'job.json', lambda j: j.update(command={'a': 1})),
    lambda root: (root / 'attempt.json').write_text('[1, 2, 3]'),
    lambda root: (root / 'fn_spec.json').write_text('"x"'),
    lambda root: (root / 'nodes' / 'node1.exit.json').write_text('[]'),
    lambda root: (root / 'ranks' / 'rank03.json').write_text('"x"'),
])
def test_collect_never_raises(tmp_path, mutate):
    root = _screen(tmp_path)
    mutate(root)
    p = P.collect(root)                                                  # must not raise
    v = J.evaluate(p, H.stored_ref(H.uc_a()))
    g = G.check(p, SMATCH, local_check=False)
    assert not (p['complete'] and v['judge_valid'] and g.ok)


def _edit(path, fn):
    obj = json.loads(path.read_text())
    fn(obj)
    path.write_text(json.dumps(obj))


# ---------------------------------------------------------------- 2: duplicate singleton artifacts
@pytest.mark.parametrize('dup', ['attempt.json', 'nodes/node0.exit.json', 'job.json'])
def test_duplicate_singletons_make_the_screen_incomplete(tmp_path, dup):
    root = _screen(tmp_path)
    (root / 'stale').mkdir()
    shutil.copy(root / dup, root / 'stale' / dup.split('/')[-1])
    p = P.collect(root)
    assert p['complete'] is False and any('copies under the collection' in x for x in p['completeness']['problems'])


def test_duplicate_trainer_results(tmp_path):
    root = _screen(tmp_path)
    run = json.loads((root / 'attempt.json').read_text())['run']
    shutil.copy(root / 'ranks' / 'rank05.json', root / f'{run}.rank5.json')
    p = P.collect(root)
    assert p['complete'] is False and p['rank_results'][5] is None
    assert 'knob_binding' in _ids(G.check(p, SMATCH, local_check=False))


# ---------------------------------------------------------------- 3: reference provenance
def test_forged_references_are_refused(tmp_path):
    cand = P.collect(_screen(tmp_path))
    good = H.stored_ref(H.uc_a())
    assert J.evaluate(cand, good)['judge_valid'] is True
    forged = [{'per_step': good['per_step'], 'source': {'kind': 'caller_forged'}},
              {**good, 'source': {**good['source'], 'kind': 'forged'}},
              {**good, 'source': {**good['source'], 'path': None}},
              {**good, 'per_step': copy.deepcopy(cand['per_step'])[:-1] + [dict(cand['per_step'][-1], main_ce=1.0)]},
              {**cand, 'source': {'kind': 'receipt'}}]
    for ref in forged:
        v = J.evaluate(cand, ref)
        assert v['judge_valid'] is False and any(r.startswith('reference') or 'no numerics reference' in r
                                                 for r in v['reasons']), v['reasons']


def test_a_stored_reference_whose_file_changed_is_refused(tmp_path):
    st = H.status_of(H.uc_a())
    path = tmp_path / 'STATUS.json'
    path.write_text(json.dumps(st))
    ref = P.from_status_cell(path, 'A')
    cand = P.collect(_screen(tmp_path))
    assert J.evaluate(cand, ref)['judge_valid'] is True
    st['collections']['A']['summary']['rows'][3]['rank0']['losses']['main_ce'] *= 1.01
    path.write_text(json.dumps(st))
    v = J.evaluate(cand, ref)
    assert v['judge_valid'] is False and any('STATUS file changed' in r for r in v['reasons'])
    with pytest.raises(ValueError, match='pinned'):
        J.stored_reference(path)


def test_a_screen_reference_is_re_collected(tmp_path):
    ref_dir = _screen(tmp_path / 'ref', START)
    ref = P.collect(ref_dir)
    cand = P.collect(_screen(tmp_path / 'c'))
    assert J.evaluate(cand, ref)['numerics_class'] == 'E0'
    _edit_rank0(ref_dir, lambda rows: rows[5]['values']['losses'].update(main_ce=0.5))
    v = J.evaluate(cand, ref)
    assert v['judge_valid'] is False and any('differ from their source' in r for r in v['reasons'])


def _edit_rank0(root, fn):
    path = root / 'signals' / 'rank00.jsonl'
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    fn(rows)
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')


# ---------------------------------------------------------------- 4: injected gates
def test_relaxed_gates_cannot_admit_a_screen(tmp_path):
    cand = P.collect(_screen(tmp_path, prof=H.scale_profile(H.profile_of(H.uc_a()), 0.9, ce_rel=0.01)))
    loose = J.load_gates()
    loose['tolerance'] = dict(loose['tolerance'], main_ce_rel_max_per_step=0.1, main_ce_rel_mean_over_window=0.1)
    v = J.evaluate(cand, H.stored_ref(H.uc_a()), loose)
    assert v['judge_valid'] is False and v['numerics_class'] == 'FAIL'
    assert any('differ from the frozen numerics_gates.json' in r for r in v['reasons'])
    assert v['details']['gates']['sha256'] == J.gates_sha256() and v['details']['gates']['supplied_differing_sha256']
    assert J.evaluate(P.collect(_screen(tmp_path / 'b')), H.stored_ref(H.uc_a()), J.load_gates())['judge_valid'] is True


# ---------------------------------------------------------------- 5, 6: aux present, bitwise E0
def test_aux_must_be_present_even_when_the_reference_is_zero():
    gates = J.load_gates()
    ref = [{'update': u, 'main_ce': 0.5 + u * 1e-3, 'mtp_ce': 0.6 + u * 1e-3, 'aux_loss': 0.0, 'mtp_aux': 0.0,
            'grad_norm': 1.0 + u * 1e-2, 'lr': 1e-7 * u} for u in range(1, 31)]
    cand = [{k: v for k, v in p.items() if k not in ('aux_loss', 'mtp_aux')} for p in ref]
    res = J.compare(cand, ref, gates)
    assert res['class'] == 'FAIL' and any('aux loss missing' in v for v in res['violations'])
    assert any('MTP aux loss missing' in v for v in res['violations'])


def test_e0_is_bitwise():
    gates = J.load_gates()
    ref = [{'update': u, 'main_ce': 0.5 + u * 1e-3, 'mtp_ce': 0.6 + u * 1e-3, 'aux_loss': 0.03, 'mtp_aux': 0.0,
            'grad_norm': 1.0 + u * 1e-2, 'lr': 1e-7 * u} for u in range(1, 31)]
    cand = [dict(p, mtp_aux=-0.0) for p in ref]
    assert J.compare(ref, ref, gates)['class'] == 'E0'
    res = J.compare(cand, ref, gates)
    assert res['class'] == 'E1' and res['bitwise_all'] is False


# ---------------------------------------------------------------- 7: knob binding
@pytest.mark.parametrize('kw,needle', [
    ({'attempt_overrides': {'stop': 2048}}, 'stop'),
    ({'attempt_overrides': {'precision_arm': {'id': 'fp8'}}}, 'precision_arm'),
    ({'attempt_overrides': {'allocator_policy': 'other'}}, 'allocator_policy'),
    ({'attempt_overrides': {'data_order_policy': 'WRONG'}}, 'data_order_policy'),
    ({'attempt_overrides': {'run': 'another_run'}}, 'run (= job name)'),
    ({'env_overrides': {'NCCL_ALGO': 'Tree'}}, 'undeclared job env'),
    ({'env_overrides': {'FLASH_UNDECLARED': '1'}}, 'undeclared job env'),
    ({'env_overrides': {'FLASH_LOG_RUN_ID': 'other'}}, 'FLASH_LOG_RUN_ID'),
    ({'env_overrides': {'LP_CACHE_ROOT': '/data/low_precision_winh_20261014/cache'}}, 'LP_CACHE_ROOT'),
    ({'env_overrides': {'FLASH_DIAG_STORAGE': 'none'}}, 'FLASH_DIAG_STORAGE'),
    ({'env_overrides': {'FLASH_IMPORT_WORKER_SHA256': 'x'}}, 'FLASH_IMPORT_WORKER_SHA256'),
    ({'rank_result_overrides': {r: {'rank': 0} for r in range(32)}}, 'rank='),
    ({'rank_result_overrides': {4: {'log_run_id': 'x'}}}, 'log_run_id'),
])
def test_knob_binding_fixed_fields_env_and_rank_identity(tmp_path, kw, needle):
    g = G.knob_binding(P.collect(_screen(tmp_path, **kw)), SMATCH)
    assert any(needle in r['detail'] for r in g.rejections), g.rejections


def test_rank_result_data_order(tmp_path):
    over = {7: {'infra_family': {'variant': G.expected_launch(SMATCH)['variant'], 'training_seed': 42,
                                 'data_order_policy': 'WRONG', 'bundle_code_sha256': G.load_manifest()['variant_runtime']['inventory']}}}
    g = G.knob_binding(P.collect(_screen(tmp_path, rank_result_overrides=over)), SMATCH)
    assert any('rank 7' in r['detail'] and 'data_order_policy' in r['detail'] for r in g.rejections)


def test_formal_bundle_env_differs_only_by_the_ext_cache(registered):
    for entry, name in [('A1_03_U_plus_C', 'U_plus_C'), ('A1_04_S_match', 'S_match'), ('A1_06_W', 'W')]:
        d = V.registered_bundle(entry)
        rec = P.launch_records(V.read(d / 'attempt.json'), V.read(d / 'job.json'), None)
        probs = G.env_problems(rec['job']['env'], rec['attempt'], rec['job'], G.expected_launch(S.registered_points()[name]))
        assert probs == [f"job env LP_CACHE_ROOT = 'None' is not bound to this attempt ({G.EXT_CACHE!r})"], probs


# ---------------------------------------------------------------- 8: wall-cap clock
def test_wall_clock_starts_at_the_launch_record(tmp_path):
    root = _screen(tmp_path, wall_s=900.0)
    ex = json.loads((root / 'nodes' / 'node0.exit.json').read_text())
    ex['started_unix'] = ex['started_unix'] + 1000.0                   # later than the launch record says
    (root / 'nodes' / 'node0.exit.json').write_text(json.dumps(ex))
    g = G.wall_cap(P.collect(root))
    assert any('disagrees with the launch record' in r['detail'] for r in g.rejections)
    root = _screen(tmp_path / 'b')
    ex = json.loads((root / 'nodes' / 'node2.exit.json').read_text())
    ex.update(finished_unix=ex['started_unix'] - 5.0)
    (root / 'nodes' / 'node2.exit.json').write_text(json.dumps(ex))
    assert any('before its start' in r['detail'] for r in G.wall_cap(P.collect(root)).rejections)


# ---------------------------------------------------------------- 9: F-PROMPTS start receipt
def test_start_receipt_must_be_the_uc_reference(uc):
    h = FlashNextHooks(T.make_task('F-PROMPTS'))
    assert h.start_receipt_problems(uc, START) == []
    no_point = {k: v for k, v in uc.items() if k != 'point'}
    no_point.pop(signing.SIGNATURE_FIELD)
    signing.sign(no_point)
    assert any('its point is not the UC-REF point' in p for p in h.start_receipt_problems(no_point, START))
    other = H.copy_receipt(uc, arm='F-A0', purpose='screen', candidate_id='c3')
    assert any('not the UC-REF reference screen' in p for p in h.start_receipt_problems(other, START))


# ---------------------------------------------------------------- 10: the own screen anchoring locked_self_mismatch
def _prompts(tmp_path, uc, runner, clock):
    drv = ScriptedBackend([{'configs': [{'config': {k: v for k, v in SMATCH.items() if v is not None}, 'rationale': 'x'},
                                        {'config': {'tp': 4}, 'rationale': 'y'}, {'config': {'tp': 8}, 'rationale': 'z'}]}],
                          clock=clock)
    arms.run_f_prompts(runner, run_id='fp', out_dir=tmp_path / 'fp', uc_ref=uc, backend=drv, clock=clock)
    return tmp_path / 'fp' / 'trace.json'


def test_locked_pair_needs_an_authentic_own_screen(tmp_path, uc):
    clock = VirtualClock()
    r = H.FakeRunner(tmp_path / 'run', clock=clock)
    path = _prompts(tmp_path, uc, r, clock)
    own_path = tmp_path / 'fp' / 'receipts' / 'c1.json'
    own = json.loads(own_path.read_text())
    tampered = copy.deepcopy(own)
    tampered['judge']['details']['per_step'][10]['main_ce'] *= 1.003
    own_path.write_text(json.dumps(tampered))
    n0 = len(r.launched)
    with pytest.raises(ValueError, match='signature'):
        arms.run_locked(r, arm='F-A0', run_id='lk', out_dir=tmp_path / 'lk', trace=path, uc_ref=uc)
    resigned = H.copy_receipt(own, candidate_id='c9')
    with pytest.raises(ValueError, match='not the search screen of c1'):
        arms.run_locked(r, arm='F-A0', run_id='lk', out_dir=tmp_path / 'lk', trace=path, uc_ref=uc, own_receipt=resigned)
    assert len(r.launched) == n0                                         # nothing launched


# ---------------------------------------------------------------- 11, 13: pending voids, Q5 budgets, one node set
def test_summary_is_refused_while_a_void_pair_waits_and_for_nonformal_windows(tmp_path, uc):
    clock = VirtualClock()
    r = H.FakeRunner(tmp_path / 'run', clock=clock, uc_locked_fail=1)
    path = _prompts(tmp_path, uc, r, clock)
    tr = json.loads(path.read_text())
    void = arms.run_locked(r, arm='F-PROMPTS', run_id='lk', out_dir=tmp_path / 'lk', trace=path, uc_ref=uc)
    assert void['status'] == 'void'
    rec = lambda: arms.load_receipts([r.out_dir]) + [uc]                # noqa: E731
    with pytest.raises(ValueError, match='pending its re-run'):
        arms.summarize(uc_ref=uc, ht=None, arms={'F-PROMPTS': {'trace': tr, 'locked': void}}, receipts=rec(),
                       allow_nonformal=True)
    ok = arms.run_locked(r, arm='F-PROMPTS', run_id='lk', out_dir=tmp_path / 'lk', trace=path, uc_ref=uc, attempt=2)
    one = {'F-PROMPTS': {'trace': tr, 'locked': ok}}
    with pytest.raises(ValueError, match='arms missing from the window'):          # a partial window is never formal
        arms.summarize(uc_ref=uc, ht=None, arms=one, receipts=rec())
    s = arms.summarize(uc_ref=uc, ht=None, arms=one, receipts=rec(), allow_nonformal=True)
    reasons = ' | '.join(s['nonformal_reasons'])
    assert s['formal'] is False and 'F-HT receipt missing' in reasons and 'arms missing' in reasons
    moved = dict(ok, nodes=['WINH0', 'WINH1', 'WINH3', 'WINH4'])
    s = arms.summarize(uc_ref=uc, ht=None, arms={'F-PROMPTS': {'trace': tr, 'locked': moved}}, receipts=rec(),
                       allow_nonformal=True)
    assert any('node sets differ' in x for x in s['nonformal_reasons'])
    small = dict(tr, budget=dict(tr['budget'], screens=2))
    s = arms.summarize(uc_ref=uc, ht=None, arms={'F-PROMPTS': {'trace': small, 'locked': ok}}, receipts=rec(),
                       allow_nonformal=True)
    assert any('Q5 table' in x for x in s['nonformal_reasons'])
    forged = dict(ok, t_locked=ok['t_locked'] * 0.5, candidate_half=dict(ok['candidate_half'], step_time_s=ok['t_locked'] * 0.5))
    s = arms.summarize(uc_ref=uc, ht=None, arms={'F-PROMPTS': {'trace': tr, 'locked': forged}}, receipts=rec(),
                       allow_nonformal=True)
    assert any('differs from its receipt' in x for x in s['nonformal_reasons'])


def test_cli_exit_codes_for_void_overrides_and_nodes(tmp_path, monkeypatch, uc, capsys):
    clock = VirtualClock()
    fake = H.FakeRunner(tmp_path / 'run', clock=clock, uc_locked_fail=1)
    monkeypatch.setattr(arms, 'make_runner', lambda nodes, **kw: fake)
    ucp = tmp_path / 'uc_ref.json'
    ucp.write_text(json.dumps(uc))
    path = _prompts(tmp_path, uc, fake, clock)
    base = ['--uc-ref', str(ucp), '--out', str(tmp_path / 'o')]
    assert cli.main(['locked', 'F-PROMPTS', '--run-id', 'lk', '--from', str(path)] + base) == 3     # void: re-run later
    assert cli.main(['locked', 'F-PROMPTS', '--run-id', 'lk', '--from', str(path)] + base) == 7     # attempt 1 ran
    assert cli.main(['locked', 'F-PROMPTS', '--run-id', 'lk', '--from', str(path), '--attempt', '2'] + base) == 0
    assert cli.main(['locked', 'F-PROMPTS', '--run-id', 'lk', '--from', str(path), '--attempt', '2'] + base) == 7
    with pytest.raises(SystemExit):
        cli.main(['locked', 'F-PROMPTS', '--run-id', 'lk', '--from', str(path), '--attempt', '3'] + base)
    assert cli.main(['arm', 'F-RND', '--run-id', 'r', '--screens', '2'] + base) == 6                 # override: refused
    assert cli.main(['arm', 'F-RND', '--run-id', 'r', '--nodes', 'WINH0,WINH1,WINH3,WINH4'] + base) == 5
    assert cli.main(['arm', 'F-RND', '--run-id', 'r', '--screens', '2', '--nonformal'] + base) == 0
    capsys.readouterr()


# ---------------------------------------------------------------- 12: margin membership
def test_margin_lists_inconsistent_receipts_as_anomalies(tmp_path, uc):
    r = H.FakeRunner(tmp_path / 'h')
    half = r.run(T.make_task('F-A0'), START, None, 'locked', 'lk', candidate_id='locked-uc',
                 reference_parsed=J.reference_from_receipt(uc))
    wrong = H.copy_receipt(half, point=SMATCH)
    m = arms.margin([uc, half, wrong])
    assert m['n'] == 2 and [x['kind'] for x in m['S_UC']] == ['uc_reference', 'locked_uc_half']
    assert m['excluded'] == [] and len(m['anomalies']) == 1 and m['anomalies'][0]['attempt_id'] == half['attempt_id']


# ---------------------------------------------------------------- round 2 (verification round) findings
def test_r2_env_binding_of_stem_and_import_hashes(tmp_path):
    for env, needle in [({'FLASH_CHECKPOINT_STEM': 'a'}, 'FLASH_CHECKPOINT_STEM'),
                        ({'FLASH_IMPORT_SNAPSHOT_SHA256': '0' * 64}, 'FLASH_IMPORT_SNAPSHOT_SHA256'),
                        ({'FLASH_IMPORT_WORKER_SHA256': '0' * 64}, 'FLASH_IMPORT_WORKER_SHA256')]:
        g = G.knob_binding(P.collect(_screen(tmp_path / needle, env_overrides=env)), SMATCH)
        assert any(needle in r['detail'] for r in g.rejections), (needle, g.rejections)


def test_r2_exit_record_start_is_required(tmp_path):
    root = _screen(tmp_path)
    for n in range(4):
        _edit(root / 'nodes' / f'node{n}.exit.json', lambda ex: ex.pop('started_unix'))
    g = G.wall_cap(P.collect(root))
    assert len([r for r in g.rejections if 'missing or disagrees' in r['detail']]) == 4


def test_r2_locked_self_check_has_no_gate_override():
    import inspect
    assert 'gates' not in inspect.signature(G.locked_self_mismatch).parameters
    assert 'gates' not in inspect.signature(arms.run_locked).parameters


def test_r2_every_path_checks_the_uc_reference_identity(tmp_path, uc, monkeypatch):
    clock = VirtualClock()
    r = H.FakeRunner(tmp_path / 'run', clock=clock)
    ht = arms.run_ht(r, run_id='ht', out_dir=tmp_path / 'ht', uc_ref=uc)['receipt']
    assert G.ht_reference_problems(ht) == [] and G.uc_reference_problems(ht)
    n0 = len(r.launched)
    for fn in (arms.run_f_rnd, arms.run_ht):
        with pytest.raises(ValueError, match=r'not the U\+C reference screen receipt'):
            fn(r, run_id='x', out_dir=tmp_path / 'x', uc_ref=ht)
    path = _prompts(tmp_path, uc, r, clock)
    n1 = len(r.launched)
    with pytest.raises(ValueError, match=r'not the U\+C reference screen receipt'):
        arms.run_locked(r, arm='F-A0', run_id='lk', out_dir=tmp_path / 'lk', trace=path, uc_ref=ht)
    assert len(r.launched) == n1 and n1 > n0
    monkeypatch.setattr(arms, 'make_runner', lambda nodes, **kw: r)
    (tmp_path / 'ht.json').write_text(json.dumps(ht))
    assert cli.main(['arm', 'F-RND', '--run-id', 'q', '--uc-ref', str(tmp_path / 'ht.json'), '--out', str(tmp_path / 'q')]) == 5
    nodeless = {k: v for k, v in uc.items() if k not in ('nodes', signing.SIGNATURE_FIELD)}
    signing.sign(nodeless)
    (tmp_path / 'nodeless.json').write_text(json.dumps(nodeless))
    assert cli.main(['arm', 'F-RND', '--run-id', 'q', '--uc-ref', str(tmp_path / 'nodeless.json'), '--nodes',
                     'WINH0,WINH1,WINH3,WINH4', '--out', str(tmp_path / 'q')]) == 5


def test_r2_locked_attempts_are_1_then_one_rerun(tmp_path, uc):
    clock = VirtualClock()
    r = H.FakeRunner(tmp_path / 'run', clock=clock)
    path = _prompts(tmp_path, uc, r, clock)
    with pytest.raises(ValueError, match='only a void pair is re-run'):
        arms.run_locked(r, arm='F-A0', run_id='lk', out_dir=tmp_path / 'lk', trace=path, uc_ref=uc, attempt=3)
    with pytest.raises(ValueError, match='re-runs a void attempt 1'):
        arms.run_locked(r, arm='F-A0', run_id='lk', out_dir=tmp_path / 'lk', trace=path, uc_ref=uc, attempt=2)
    ok = arms.run_locked(r, arm='F-A0', run_id='lk', out_dir=tmp_path / 'lk', trace=path, uc_ref=uc)
    assert ok['status'] == 'ok'
    with pytest.raises(ValueError, match='already ran'):
        arms.run_locked(r, arm='F-A0', run_id='lk', out_dir=tmp_path / 'lk', trace=path, uc_ref=uc)
    with pytest.raises(ValueError, match='re-runs a void attempt 1'):            # attempt 1 was not void
        arms.run_locked(r, arm='F-A0', run_id='lk', out_dir=tmp_path / 'lk', trace=path, uc_ref=uc, attempt=2)
    assert arms.final_locked([tmp_path / 'lk' / 'locked.json'])['status'] == 'ok'
    with pytest.raises(ValueError):
        arms.final_locked([dict(ok, attempt=2)])
    with pytest.raises(ValueError):
        arms.final_locked([ok, dict(ok, attempt=2)])
    with pytest.raises(ValueError):
        arms.final_locked([dict(ok, status='void'), dict(ok, attempt=2), dict(ok, attempt=3)])


def test_r2_a_margin_file_must_match_the_window_receipts(tmp_path, uc):
    clock = VirtualClock()
    r = H.FakeRunner(tmp_path / 'run', clock=clock)
    path = _prompts(tmp_path, uc, r, clock)
    ok = arms.run_locked(r, arm='F-PROMPTS', run_id='lk', out_dir=tmp_path / 'lk', trace=path, uc_ref=uc)
    window = {'F-PROMPTS': {'trace': json.loads(path.read_text()), 'locked': ok}}
    receipts = arms.load_receipts([r.out_dir]) + [uc]
    s = arms.summarize(uc_ref=uc, ht=None, arms=window, receipts=receipts, margin_rec={'margin': 0.0},
                       allow_nonformal=True)
    assert any('margin file' in x for x in s['nonformal_reasons'])
    assert s['margin']['margin'] is None or s['margin']['margin'] >= 0.03                   # recomputed, not the file
    good = arms.margin(receipts)
    s = arms.summarize(uc_ref=uc, ht=None, arms=window, receipts=receipts, margin_rec=good, allow_nonformal=True)
    assert not any('margin file' in x for x in s['nonformal_reasons'])


def test_r2_locked_numbers_must_rederive_from_the_receipts(tmp_path, uc):
    clock = VirtualClock()
    r = H.FakeRunner(tmp_path / 'run', clock=clock)
    path = _prompts(tmp_path, uc, r, clock)
    tr = json.loads(path.read_text())
    ok = arms.run_locked(r, arm='F-PROMPTS', run_id='lk', out_dir=tmp_path / 'lk', trace=path, uc_ref=uc)
    receipts = arms.load_receipts([r.out_dir, tmp_path / 'fp']) + [uc]
    by = {x['attempt_id']: x for x in receipts}
    assert arms.locked_receipt_problems('F-PROMPTS', ok, by, trace=tr, uc_ref=uc) == []
    for change, needle in [({'y_pair': ok['y_pair'] + 0.1}, "['y_pair'] differ"),
                           ({'t_uc_pair': ok['t_uc_pair'] * 1.1}, "['t_uc_pair'] differ"),
                           ({'speedup_vs_uc': 1.5, 'numerics_class': 'E1'}, "['speedup_vs_uc', 'numerics_class'] differ"),
                           ({'order': ['candidate', 'uc']}, 'parity order'),
                           ({'point': START}, 'not the arm\'s final selection'),
                           ({'own_screen_attempt': uc['attempt_id']}, 'own search screen')]:
        got = arms.locked_receipt_problems('F-PROMPTS', dict(ok, **change), by, trace=tr, uc_ref=uc)
        assert any(needle in x for x in got), (change, got)
    assert any('is arm' in x for x in arms.formal_problems(uc_ref=uc, ht=None, arms={'F-A0': {'trace': tr, 'locked': ok}},
                                                           receipts=receipts))


def test_r2_a_rerun_is_formal_only_after_a_void_its_receipts_confirm(tmp_path, uc):
    clock = VirtualClock()
    r = H.FakeRunner(tmp_path / 'run', clock=clock, uc_locked_fail=1)
    path = _prompts(tmp_path, uc, r, clock)
    tr = json.loads(path.read_text())
    void = arms.run_locked(r, arm='F-PROMPTS', run_id='lk', out_dir=tmp_path / 'lk', trace=path, uc_ref=uc)
    second = arms.run_locked(r, arm='F-PROMPTS', run_id='lk', out_dir=tmp_path / 'lk', trace=path, uc_ref=uc, attempt=2)
    receipts = arms.load_receipts([r.out_dir, tmp_path / 'fp']) + [uc]

    def reasons(**v):
        return arms.formal_problems(uc_ref=uc, ht=None, arms={'F-PROMPTS': {'trace': tr, **v}}, receipts=receipts)
    assert not [x for x in reasons(locked=second, locked_history=[void, second]) if x.startswith(('F-PROMPTS', '(attempt'))]
    assert any('without its attempt 1' in x for x in reasons(locked=second, locked_history=[second]))
    assert any('no locked result accounts for' in x for x in reasons(locked=second, locked_history=[second]))
    forged = dict(void, status='ok')                   # a valid attempt 1 relabelled void to buy a second try is caught
    assert any(x.startswith('(attempt 1)') for x in reasons(locked=second, locked_history=[dict(void, status='void',
                                                                                                 uc_half=second['uc_half'])]))
    assert any('differ from what its receipts give' in x for x in reasons(locked=forged, locked_history=[forged]))


def test_r2_every_launched_screen_of_the_window_is_on_the_reference_node_set(tmp_path, uc):
    clock = VirtualClock()
    r = H.FakeRunner(tmp_path / 'run', clock=clock)
    path = _prompts(tmp_path, uc, r, clock)
    tr = json.loads(path.read_text())
    ok = arms.run_locked(r, arm='F-PROMPTS', run_id='lk', out_dir=tmp_path / 'lk', trace=path, uc_ref=uc)
    receipts = arms.load_receipts([r.out_dir, tmp_path / 'fp']) + [uc]

    def node_reasons(recs):
        return [x for x in arms.formal_problems(uc_ref=uc, ht=None, arms={'F-PROMPTS': {'trace': tr, 'locked': ok}},
                                                receipts=recs) if 'another node set' in x]
    assert node_reasons(receipts) == []
    elsewhere = H.copy_receipt(next(x for x in receipts if x.get('purpose') == 'screen' and not x.get('not_launched')),
                               nodes=['WINH0', 'WINH2', 'WINH5', 'WINH7'], attempt_id='a-elsewhere')
    assert node_reasons(receipts + [elsewhere])


# ---------------------------------------------------------------- round 3 (second verification round) findings
def _window_one_arm(tmp_path, uc):
    clock = VirtualClock()
    r = H.FakeRunner(tmp_path / 'run', clock=clock)
    path = _prompts(tmp_path, uc, r, clock)
    ok = arms.run_locked(r, arm='F-PROMPTS', run_id='lk', out_dir=tmp_path / 'fp', trace=path, uc_ref=uc)
    return r, path, ok


def test_r3_formal_traces_are_sealed_and_bound_to_their_screen_receipts(tmp_path, uc):
    r, path, ok = _window_one_arm(tmp_path, uc)
    tr = json.loads(path.read_text())
    assert signing.verify(tr)                                            # sealed by run_f_prompts
    receipts = arms.load_receipts([r.out_dir, tmp_path / 'fp'])
    by = {x['attempt_id']: x for x in receipts}
    assert arms.trace_problems('F-PROMPTS', tr, by) == []
    edited = json.loads(json.dumps(tr))
    edited['screens'][0]['step_time'] *= 0.6                              # a faster search screen typed into the trace
    assert any('not sealed' in x for x in arms.trace_problems('F-PROMPTS', edited, by))
    signing.sign(edited)                                                 # even if it were re-sealed: the receipt disagrees
    assert any('differs from its receipt' in x for x in arms.trace_problems('F-PROMPTS', edited, by))
    cheap = json.loads(json.dumps(tr))
    cheap['screens'][0]['gpu_hours'] = 0.0
    signing.sign(cheap)
    assert any('differs from its receipt' in x for x in arms.trace_problems('F-PROMPTS', cheap, by))
    fewer = json.loads(json.dumps(tr))
    fewer['screens'] = []
    signing.sign(fewer)
    assert any('does not list' in x for x in arms.trace_problems('F-PROMPTS', fewer, by))


def test_r3_locked_halves_are_one_back_to_back_run(tmp_path, uc):
    r, path, ok = _window_one_arm(tmp_path, uc)
    tr = json.loads(path.read_text())
    assert signing.verify(ok) and ok['run_id'] == 'lk'
    other = arms.run_locked(r, arm='F-PROMPTS', run_id='lk-other', out_dir=tmp_path / 'other', trace=path, uc_ref=uc)
    receipts = arms.load_receipts([r.out_dir, tmp_path / 'fp'])
    by = {x['attempt_id']: x for x in receipts}
    assert arms.locked_receipt_problems('F-PROMPTS', ok, by, trace=tr, uc_ref=uc) == []
    mixed = dict(ok, candidate_half=other['candidate_half'])             # a candidate half from another locked run
    signing.sign(mixed)
    got = arms.locked_receipt_problems('F-PROMPTS', mixed, by, trace=tr, uc_ref=uc)
    assert any("is not the locked-candidate run of locked run 'lk'" in x for x in got), got
    uc_half = {k: v for k, v in by[ok['uc_half']['attempt_id']].items() if k != '_path'}
    late = H.copy_receipt(uc_half, utc_start='2099-01-01T00:00:00+00:00', utc_end='2099-01-01T00:20:00+00:00')
    got = arms.locked_receipt_problems('F-PROMPTS', ok, dict(by, **{late['attempt_id']: late}), trace=tr, uc_ref=uc)
    assert any('back to back' in x for x in got), got                     # F-PROMPTS (index 0) runs U+C first
    costless = dict(ok, cost={'gpu_seconds': 0.0, 'wall_seconds': 0.0})
    assert any('not sealed' in x for x in arms.locked_receipt_problems('F-PROMPTS', costless, by, trace=tr, uc_ref=uc))
    signing.sign(costless)
    assert any("['cost'] differ" in x for x in arms.locked_receipt_problems('F-PROMPTS', costless, by, trace=tr, uc_ref=uc))


def test_r3_differing_copies_of_one_receipt_reach_the_formal_check(tmp_path, uc):
    r, path, ok = _window_one_arm(tmp_path, uc)
    rec_dir = tmp_path / 'fp' / 'receipts'
    c1 = json.loads((rec_dir / 'c1.json').read_text())
    (rec_dir / 'zz_copy.json').write_text(json.dumps(dict(c1, step_time={'mean_window_max_rank_s': 1.0})))
    receipts = arms.load_receipts([tmp_path / 'fp', r.out_dir])
    assert len([x for x in receipts if x['attempt_id'] == c1['attempt_id']]) == 2       # both copies kept
    window = {'F-PROMPTS': {'trace': json.loads(path.read_text()), 'locked': ok}}
    got = arms.formal_problems(uc_ref=uc, ht=None, arms=window, receipts=receipts + [uc])
    assert any('differing receipts for the same attempt' in x and c1['attempt_id'] in x for x in got)
    half = json.loads(json.dumps(next(x for x in receipts if x['attempt_id'] == ok['uc_half']['attempt_id'])))
    half.pop('_path')
    (rec_dir / 'zz_half.json').write_text(json.dumps(dict(half, metric=dict(half['metric'], value=1.0))))
    m = arms.margin(arms.load_receipts([tmp_path / 'fp', r.out_dir]) + [uc])
    assert any(a['why'].startswith('a second, different receipt') for a in m['anomalies'])
