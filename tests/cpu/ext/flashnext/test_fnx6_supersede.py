"""AMENDMENT_EXT_009 (D29, issue #37): screens the harness lost after the path measured them. The fake runner judges a
screen and then returns the `runner_error` receipt our collection produced (the judged receipt is kept, as a read-only
re-collection would rebuild it); the supersede step is simulated with supersede.supersede_record."""
import json

import pytest

import x6_helpers as H
from infrarsi.agent.backends import ScriptedBackend
from infrarsi.agent.budget import VirtualClock
from infrarsi.ext.flashnext import arms
from infrarsi.ext.flashnext import cli
from infrarsi.ext.flashnext import supersede as SU
from infrarsi.ext.flashnext import switch_space as S
from infrarsi.harness import signing

START = S.start_point()
SMATCH = S.registered_points()['S_match']
W = S.registered_points()['W']
B = S.registered_points()['B']


@pytest.fixture(scope='module')
def uc(tmp_path_factory):
    return H.uc_reference_receipt(tmp_path_factory.mktemp('uc'), clock=VirtualClock())


def recollection(original, **cond):
    """What the harness's step (X5, recollect.py) records after a clean read-only re-collection (all four conditions
    hold): node times relative to the attempt's start; pre-launch records and launch logs just after it."""
    t0 = SU._unix(original['utc_start'])
    files = [{'path': p, 'sha256': f'{i:064x}', 'size': 1000 + i,
              'mtime': t0 + (5.0 if (p in SU.PRE_LAUNCH or p.endswith('.launch.log')) else 500.0)}
             for i, p in enumerate(SU.LAYOUT)]
    return {'stage': 'collect', 'original_error': original['runner_error'], 'method': 'fixed_collector',
            'error_class': SU.ERROR_CLASS[original['runner_error']['type']],
            'conditions': {c: cond.get(c, True) for c in SU.CONDITIONS},
            'node_records': {n: {'exit_code': 0, 'started_unix': t0 + 10.0, 'finished_unix': t0 + 1000.0,
                                 'release_state': 'process_group_gone', 'release_finished_unix': t0 + 1010.0}
                             for n in original['nodes']},
            'files': files, 'pulled': [{'path': f['path'], 'sha256': f['sha256']} for f in files[:8]]}


def _prompts(tmp_path, uc, runner, clock, configs):
    drv = ScriptedBackend([{'configs': [{'config': {k: v for k, v in c.items() if v is not None}, 'rationale': 'r'}
                                        for c in configs]}], clock=clock)
    arms.run_f_prompts(runner, run_id='fp', out_dir=tmp_path / 'fp', uc_ref=uc, backend=drv, clock=clock)
    return tmp_path / 'fp'


def _supersede(run_dir, runner, cid, judged=None, **cond):
    """The harness's supersede step (X5): re-collect read-only, judge unchanged, write the signed record."""
    original = json.loads((run_dir / 'receipts' / f'{cid}.json').read_text())
    rec = SU.supersede_record(original, judged or runner.lost[original['attempt_id']], recollection(original, **cond))
    (run_dir / 'receipts' / f'{cid}.recollected.json').write_text(json.dumps(rec))
    return original, rec


def _receipts(*dirs):
    return arms.load_receipts(list(dirs))


def _resigned(rec, **ch):
    x = {k: v for k, v in rec.items() if k != signing.SIGNATURE_FIELD}
    x.update(ch)
    return signing.sign(json.loads(json.dumps(x)))


# ---------------------------------------------------------------- the record contract
def test_record_contract(tmp_path, uc):
    clock = VirtualClock()
    r = H.FakeRunner(tmp_path / 'run', clock=clock, lose={'c1': 'json'})
    d = _prompts(tmp_path, uc, r, clock, [SMATCH, W, B])
    original, rec = _supersede(d, r, 'c1')
    assert SU.collection_failure(original) and SU.must_dispose(original)
    assert original['runner_error']['type'] == 'JSONDecodeError'
    assert SU.record_problems(original, rec) == [] and rec['valid'] is True and rec['cost'] == original['cost']
    assert all(rec[k] == original[k] for k in SU.IDENTITY) and rec['supersedes'] == original['attempt_id']
    rc = rec['recollection']
    bad_files = [dict(f, sha256='z' * 64) if i == 0 else f for i, f in enumerate(rc['files'])]
    late = [dict(f, mtime=f['mtime'] + 99999.0) if f['path'] == 'signals/rank05.jsonl' else f for f in rc['files']]
    nodes_bad = {n: dict(x, exit_code=1) for n, x in rc['node_records'].items()}
    cases = [(_resigned(rec, utc_end='2099-01-01T00:00:00+00:00'), 'identity differs'),
             (_resigned(rec, cost=dict(rec['cost'], gpu_seconds=1.0)), 'cost is not the original cost'),
             (_resigned(rec, recollection=dict(rc, conditions=dict(rc['conditions'], iv_pulled_files_identical=False))),
              'integrity conditions not met'),
             (_resigned(rec, recollection=dict(rc, files=rc['files'][:87])), '88-file layout'),
             (_resigned(rec, recollection=dict(rc, files=bad_files)), 'not a layout file'),
             (_resigned(rec, recollection=dict(rc, files=late)), 'outside its window'),
             (_resigned(rec, recollection=dict(rc, node_records=nodes_bad)), 'need 0 / process_group_gone'),
             (_resigned(rec, recollection=dict(rc, pulled=[{'path': 'job.json', 'sha256': '0' * 64}])), 'not byte-identical'),
             (_resigned(rec, recollection=dict(rc, method='scp')), 'collection method'),
             (_resigned(rec, recollection=dict(rc, stage='launch')), 'collect-stage error'),
             (_resigned(rec, recollection=dict(rc, error_class='collect_transport_timeout')), 'error class'),
             (dict(rec, valid=False), 'signature'),
             (_resigned(rec, valid=False), 'valid is not judge_valid'),
             (_resigned(rec, judge=None), 'no complete judge verdict'),
             (_resigned(rec, numerics_gates_sha256='0' * 64), 'frozen numerics gates'),
             (_resigned(rec, guard_reports=[]), 'guards.check did not run'),
             (_resigned(rec, superseded_by='another-record'), 'none of a supersede')]
    for bad, needle in cases:
        assert any(needle in x for x in SU.record_problems(original, bad)), (needle, SU.record_problems(original, bad))
    # never a candidate failure (Codex #2): a runner_error of another class, or a judged failure, stands
    assertion = _resigned(original, runner_error=dict(original['runner_error'], type='AssertionError'))
    assert not SU.collection_failure(assertion) and any('candidate' in x for x in SU.record_problems(assertion, rec))
    judged_fail = _resigned(original, judge={'numerics_class': 'FAIL'})
    assert any('candidate' in x for x in SU.record_problems(judged_fail, rec))
    # a record built from the unjudged original is refused (Codex #3)
    assert any('no complete judge verdict' in x
               for x in SU.record_problems(original, SU.supersede_record(original, original, recollection(original))))
    assert any('a record never supersedes a record' in x for x in SU.record_problems(rec, _resigned(rec, supersedes=rec['attempt_id'])))
    # one record per attempt; every copy of the original signed (Codex #7)
    other = _resigned(rec, recollection=dict(rc, method='sftp'))
    _, admitted, probs = SU.index(_receipts(d) + [other])
    assert original['attempt_id'] not in admitted and any('more than one superseding record' in x for x in probs)
    broken_copy = dict(original, harness_signature=dict(original['harness_signature'], mac='0' * 64))
    _, _, probs = SU.index(_receipts(d) + [broken_copy])
    assert any('differing receipts for the same attempt' in x for x in probs)
    # an unrecoverable flag needs a failed condition among (ii)-(iv) (Codex #5)
    flag_ok = SU.unrecoverable_record(original, recollection(original, ii_readonly_complete_layout=False))
    flag_bad = SU.unrecoverable_record(original, recollection(original))
    assert SU.record_problems(original, flag_ok) == []
    assert any('recoverable and must be superseded' in x for x in SU.record_problems(original, flag_bad))


# ---------------------------------------------------------------- the one-shot arm (Q1, Q2)
def test_one_shot_arm_is_retraced_selected_and_locked_on_the_record(tmp_path, uc, capsys):
    clock = VirtualClock()
    r = H.FakeRunner(tmp_path / 'run', clock=clock, lose={'c1': 'json'})
    d = _prompts(tmp_path, uc, r, clock, [SMATCH, W, B])
    as_run = json.loads((d / 'trace.json').read_text())
    assert as_run['selected'] == 'c2' and as_run['screens'][0]['valid'] is False      # W won while c1 was lost
    # a certain #37 loss without its record keeps the window non-formal and the trace un-retraced (Codex #1)
    window0 = {'F-PROMPTS': {'trace': as_run, 'locked': None}}
    assert any('without their EXT_009 record' in x
               for x in arms.formal_problems(uc_ref=uc, ht=None, arms=window0, receipts=_receipts(d, r.out_dir) + [uc]))
    assert cli.main(['retrace', '--run-dir', str(d)]) == 8
    capsys.readouterr()
    original, rec = _supersede(d, r, 'c1')
    originals, records, probs = SU.index(_receipts(d, r.out_dir))
    assert probs == [] and set(records) == {original['attempt_id']}
    p = arms.trace_problems('F-PROMPTS', as_run, originals, True, records=records)
    assert any('does not apply' in x for x in p)
    assert cli.main(['retrace', '--run-dir', str(d)]) == 0
    capsys.readouterr()
    tr = json.loads((d / 'trace.json').read_text())
    assert signing.verify(tr) and json.loads((d / 'trace.as_run.json').read_text()) == as_run
    row = tr['screens'][0]
    assert (row['collection'], row['valid'], row['supersedes']) == ('recollected_readonly', True, original['attempt_id'])
    assert row['step_time'] == rec['metric']['value'] and row['gpu_hours'] == as_run['screens'][0]['gpu_hours']
    assert row['record_digest'] == SU.digest(rec)
    assert tr['selected'] == 'c1' and tr['selected_config'] == SMATCH and tr['screens'][1]['collection'] == 'collected'
    assert tr['used'] == as_run['used'] and tr['extra']['ext009']['applied'][0]['attempt_id'] == original['attempt_id']
    assert arms.retrace(d) == tr                                                         # idempotent, from the as-run trace
    assert arms.trace_problems('F-PROMPTS', tr, originals, True, records=records) == []
    point, own, cid = arms.selection_from(d / 'trace.json')
    assert cid == 'c1' and own['collection'] == 'recollected_readonly' and arms.own_screen_problems(own, cid, point) == []
    lk = arms.run_locked(r, arm='F-PROMPTS', run_id='lk', out_dir=d, trace=d / 'trace.json', uc_ref=uc)
    assert lk['status'] == 'ok' and lk['locked_valid'] and lk['own_screen_attempt'] == original['attempt_id']
    assert lk['own_screen_digest'] == SU.digest(rec) and lk['own_screen_collection'] == 'recollected_readonly'
    receipts = _receipts(d, r.out_dir) + [uc]
    originals, records, _ = SU.index(receipts)
    assert arms.locked_receipt_problems('F-PROMPTS', lk, originals, trace=tr, uc_ref=uc, records=records) == []
    window = {'F-PROMPTS': {'trace': tr, 'locked': lk}}
    reasons = arms.formal_problems(uc_ref=uc, ht=None, arms=window, receipts=receipts)
    assert not [x for x in reasons if x.startswith('F-PROMPTS') or 'superseding' in x or 'differing receipts' in x
                or 'EXT_009' in x], reasons
    ht = arms.run_ht(H.FakeRunner(tmp_path / 'htr', clock=clock), run_id='ht', out_dir=tmp_path / 'ht', uc_ref=uc)['receipt']
    s = arms.summarize(uc_ref=uc, ht=ht, arms=window, receipts=receipts, allow_nonformal=True)
    assert s['ext009']['disclosure'] == arms.EXT009_DISCLOSURE and 'value-blind' in s['ext009']['disclosure']
    assert s['ext009']['collection_values'] == ['collected', 'recollected_readonly', 'rescreen']
    srow = s['rows']['F-PROMPTS']
    assert [x['collection'] for x in srow['screens']] == ['recollected_readonly', 'collected', 'collected']
    assert srow['harness']['n_recovered'] == 1 and srow['harness']['n_harness_lost'] == 0
    reading = srow['recovered_fraction_reading']
    assert 'found the best human recipe' in reading and 'vs F-HT' in reading and 'floor' in reading      # Codex #10
    edited = dict(tr, selected='c2')                                                      # a hand edit is caught twice
    p = arms.trace_problems('F-PROMPTS', edited, originals, True, records=records)
    assert any('not sealed' in x for x in p) and any('not the best valid screen' in x for x in p)


def test_the_locked_pair_is_anchored_on_the_exact_record(tmp_path, uc):
    """Codex #6: a later, different record for the attempt (or one the trace did not apply) cannot anchor the pair."""
    clock = VirtualClock()
    r = H.FakeRunner(tmp_path / 'run', clock=clock, lose={'c1': 'json'})
    d = _prompts(tmp_path, uc, r, clock, [SMATCH, W, B])
    original, rec = _supersede(d, r, 'c1')
    arms.retrace(d)
    swapped = _resigned(rec, recollection=dict(rec['recollection'], method='sftp'))     # admitted alone, other digest
    (d / 'receipts' / 'c1.recollected.json').write_text(json.dumps(swapped))
    point, own, cid = arms.selection_from(d / 'trace.json')
    assert own is None                                                                  # the row names the first record
    with pytest.raises(ValueError, match='cannot anchor the locked pair'):
        arms.run_locked(r, arm='F-PROMPTS', run_id='lk', out_dir=d, trace=d / 'trace.json', uc_ref=uc)
    originals, records, _ = SU.index(_receipts(d, r.out_dir))
    tr = json.loads((d / 'trace.json').read_text())
    assert any('applied another record' in x for x in arms.trace_problems('F-PROMPTS', tr, originals, True, records=records))


def test_an_unrecoverable_screen_stays_runner_error_and_is_flagged(tmp_path, uc):
    clock = VirtualClock()
    r = H.FakeRunner(tmp_path / 'run', clock=clock, lose={'c1': 'json'})
    d = _prompts(tmp_path, uc, r, clock, [SMATCH, W, B])
    original = json.loads((d / 'receipts' / 'c1.json').read_text())
    flag = SU.unrecoverable_record(original, recollection(original, ii_readonly_complete_layout=False))
    (d / 'receipts' / 'c1.unrecoverable.json').write_text(json.dumps(flag))
    tr = arms.retrace(d)
    row = tr['screens'][0]
    assert (row['collection'], row['valid'], row['harness_lost_unrecovered'], row['void']) == ('collected', False, True, True)
    assert tr['selected'] == 'c2'
    hv = SU.harness_view('F-PROMPTS', tr, *SU.index(_receipts(d, r.out_dir))[:2])
    assert hv['n_failed_by_class'] == {'harness_lost': 1} and hv['recovered'] == []
    assert hv['rows'][0]['collection'] == 'collected' and hv['rows'][0]['class'] == 'harness_lost'


# ---------------------------------------------------------------- the multi-step arms (Q4)
def _rnd(tmp_path, uc, lose, name='frnd'):
    clock = VirtualClock()
    r = H.FakeRunner(tmp_path / name, clock=clock, lose=lose)
    tr = arms.run_f_rnd(r, run_id=name, out_dir=tmp_path / name / 'out', uc_ref=uc, clock=clock, screens=5)
    return r, tr, tmp_path / name / 'out'


def test_a_multi_step_arm_keeps_what_it_saw(tmp_path, uc):
    r, tr, d = _rnd(tmp_path, uc, {'c2': 'json', 'c4': 'json'})
    assert [s['valid'] for s in tr['screens']][1] is False and len(r.launched) == 5        # charged, point stays seen
    for cid in ('c2', 'c4'):
        _supersede(d, r, cid)
    with pytest.raises(ValueError, match='only the one-shot arm'):
        arms.retrace(d)
    originals, records, probs = SU.index(_receipts(d, r.out_dir))
    assert probs == [] and len(records) == 2
    assert arms.trace_problems('F-RND', tr, originals, True, records=records) == []        # bound to the originals
    claimed = json.loads(json.dumps(tr))
    claimed['screens'][1]['collection'] = 'recollected_readonly'
    assert any('multi-step arm keeps what it saw' in x
               for x in arms.trace_problems('F-RND', claimed, originals, True, records=records))
    hv = SU.harness_view('F-RND', tr, originals, records)
    assert hv['n_harness_lost'] == 2 and all(v['descriptive_only'] for v in hv['recovered'])
    lost_rows = [x for x in hv['rows'] if x['class'] == 'harness_lost']
    assert len(lost_rows) == 2 and {x['collection'] for x in lost_rows} == {'recollected_readonly'}      # Codex #9
    assert hv['comparison_withheld']
    assert tr['selected_step_time'] == min(s['step_time'] for s in tr['screens'] if s['valid'])  # recovered values unused
    window = {a: {'trace': tr, 'locked': None} for a in ('F-RND', 'F-A0')}
    s = arms.summarize(uc_ref=uc, ht=None, arms=window, receipts=_receipts(d, r.out_dir) + [uc], allow_nonformal=True)
    lab = s['labels']['F-A0 vs F-RND']
    assert lab['label'] == 'no comparison (harness)' and set(lab['withheld_by']) == {'F-A0', 'F-RND'}
    assert 'computed' in lab and 'random_control' not in s


def test_harness_cost_selection_withholds_the_pair(tmp_path, uc):
    r, tr, d = _rnd(tmp_path, uc, {'c3': 'json'}, name='hcs')
    att = tr['screens'][2]['attempt_id']
    best = tr['selected_step_time']
    judged = {k: v for k, v in r.lost[att].items() if k != signing.SIGNATURE_FIELD}
    v = best * 0.5
    judged['metric'] = dict(judged['metric'], value=v)
    judged['step_time'] = {'mean_window_max_rank_s': v}
    judged['judge'] = dict(judged['judge'], metric_s=v)
    _supersede(d, r, 'c3', judged=signing.sign(judged))
    originals, records, probs = SU.index(_receipts(d, r.out_dir))
    assert probs == []
    hv = SU.harness_view('F-RND', tr, originals, records)
    assert hv['n_harness_lost'] == 1 and hv['harness_cost_selection'] and hv['comparison_withheld']
    r2, tr2, d2 = _rnd(tmp_path, uc, {'c3': 'json'}, name='one')                          # one loss, no cost: reported only
    _supersede(d2, r2, 'c3')
    o2, rec2, _ = SU.index(_receipts(d2, r2.out_dir))
    hv2 = SU.harness_view('F-RND', tr2, o2, rec2)
    if not hv2['harness_cost_selection']:
        assert hv2['n_harness_lost'] == 1 and not hv2['comparison_withheld']


def test_codex_round2_partials(tmp_path, uc):
    """EXT_009 review round 2: exact reasons, materialization.json's window, raw no-metric for the flag, any
    superseded_by key on a record."""
    clock = VirtualClock()
    r = H.FakeRunner(tmp_path / 'run', clock=clock, lose={'c1': 'json'})
    d = _prompts(tmp_path, uc, r, clock, [SMATCH, W, B])
    original, rec = _supersede(d, r, 'c1')
    rc = rec['recollection']
    extra = _resigned(rec, invalid_reasons=list(rec.get('invalid_reasons') or []) + ['fabricated'])
    assert any('invalid_reasons are not' in x for x in SU.record_problems(original, extra))
    late = [dict(f, mtime=f['mtime'] + 1e6) if f['path'] == 'materialization.json' else f for f in rc['files']]
    assert any('materialization.json' in x for x in SU.record_problems(original, _resigned(rec, recollection=dict(rc, files=late))))
    flag = SU.unrecoverable_record(original, recollection(original, iv_pulled_files_identical=False))
    assert SU.record_problems(original, flag) == []
    zero = _resigned(flag, metric=dict(flag['metric'], value=0.0), step_time={'mean_window_max_rank_s': 1.0})
    assert any('has no verdict' in x for x in SU.record_problems(original, zero))
    for marker in ('', None):
        assert any('none of a supersede' in x for x in SU.record_problems(original, _resigned(rec, superseded_by=marker)))


def test_codex_round3_lows(tmp_path, uc):
    """EXT_009 review round 3: invalid_reasons compared raw (a list, exactly); the mtime window is the ruled closed
    interval (no lower slack; O-approved basis: pre-launch records and launch logs from the attempt's utc_start)."""
    clock = VirtualClock()
    r = H.FakeRunner(tmp_path / 'run', clock=clock, lose={'c1': 'json'})
    d = _prompts(tmp_path, uc, r, clock, [SMATCH, W, B])
    original, rec = _supersede(d, r, 'c1')
    assert any('invalid_reasons are not' in x for x in SU.record_problems(original, _resigned(rec, invalid_reasons=None)))
    rc = rec['recollection']
    start0 = rc['node_records'][original['nodes'][0]]['started_unix']
    early = [dict(f, mtime=start0 - 0.5) if f['path'] == 'materialization.json' else f for f in rc['files']]
    assert any('materialization.json' in x for x in SU.record_problems(original, _resigned(rec, recollection=dict(rc, files=early))))
    t0 = SU._unix(original['utc_start'])
    at_start = [dict(f, mtime=t0) if f['path'] in SU.PRE_LAUNCH else f for f in rc['files']]
    assert SU.record_problems(original, _resigned(rec, recollection=dict(rc, files=at_start))) == []
