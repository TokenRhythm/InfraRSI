"""ext/flashnext parser and judge (X6; AMENDMENT_EXT_003 Q3, G-F1 (2)).

The 20 stored formal-54 cells are re-derived exactly: the window metric (to <= 1e-9) from the stored rows and from 32
synthesized rank signal logs, and the numerics class against the U+C A cell (every exact recipe E0, the lossy Q family
FAIL). Then the gate edges (E1, MTP CE, grad norm, LR, aux, repeated scalars) and every parse failure mode.
"""
import hashlib
import json
import math

import pytest

import x6_helpers as H
from infrarsi.ext.common.env_runner import jsonable
from infrarsi.ext.flashnext import judge as J
from infrarsi.ext.flashnext import parse as P
from infrarsi.ext.flashnext import switch_space as S

CELLS = H.cells()
IDS = [f'{c["wave"]}-{c["group"]}-{c["variant"]}' for c in CELLS]
START = S.start_point()
SMATCH = S.registered_points()['S_match']


def _screen(tmp_path, profile=None, point=SMATCH, **kw):
    return H.write_screen(tmp_path / 'scr', profile or H.profile_of(H.uc_a()), point, **kw)


def _ref():
    return H.stored_ref(H.uc_a())


# ---------------------------------------------------------------- the 20 stored cells
def test_fixture_is_an_exact_extract_of_the_stored_cells():
    assert len(CELLS) == 20 and len({(c['wave'], c['group']) for c in CELLS}) == 20
    if not H.F54.exists():
        pytest.skip('formal-54 final tree not on this machine')
    for c in CELLS:
        raw = (H.F54 / c['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == c['status_sha256']
        st = json.loads(raw.decode('utf-8-sig'))
        summ = st['collections'][c['group']]['summary']
        assert summ['variant'] == c['variant'] and summ['attempt_id'] == c['attempt_id']
        assert [r['max_rank_whole_seconds'] for r in summ['rows']] == [r[1] for r in c['rows']]
        assert P.from_status_cell(st, c['group'])['per_step'] == H.parsed_cell(c)['per_step']


@pytest.mark.parametrize('c', CELLS, ids=IDS)
def test_stored_cell_metric_is_rederived(c):
    p = H.parsed_cell(c)
    assert p['complete'] and p['metric']['n_window'] == 14
    assert abs(p['metric']['value'] - c['primary_window_17_30_mean_max32']) <= 1e-9


@pytest.mark.parametrize('c', CELLS, ids=IDS)
def test_32_rank_signals_of_a_stored_cell_parse_to_its_metric(tmp_path, c):
    root = H.write_screen(tmp_path / 's', H.profile_of(c), START)
    p = P.collect(root)
    assert p['complete'], p['completeness']
    assert abs(p['metric']['value'] - c['primary_window_17_30_mean_max32']) <= 1e-9
    assert [s['max_rank_s'] for s in p['steps']] == [r[1] for r in c['rows']]          # per-update max over 32 ranks
    assert p['per_step'] == H.parsed_cell(c)['per_step']                                # rank 0's scalars
    assert p['scalar_consistency']['mismatches'] == [] and p['scalar_consistency']['checked_ranks'] == 32


@pytest.mark.parametrize('c', CELLS, ids=IDS)
def test_numerics_class_against_the_uc_a_cell(c):
    v = J.evaluate(H.parsed_cell(c), _ref())
    if c['variant'] in H.LOSSY:
        assert v['numerics_class'] == 'FAIL' and v['judge_valid'] is False
        assert any('main CE rel dev' in r for r in v['reasons'])
        assert v['metric_s'] == pytest.approx(c['primary_window_17_30_mean_max32'], abs=1e-9)   # recorded, never gated
    else:
        assert v['numerics_class'] == 'E0' and v['judge_valid'] is True, v['reasons']
        assert v['metric_s'] == pytest.approx(c['primary_window_17_30_mean_max32'], abs=1e-9)


def test_the_ruled_cells_are_covered():
    by = {}
    for c in CELLS:
        by.setdefault(c['variant'], []).append(J.evaluate(H.parsed_cell(c), _ref())['numerics_class'])
    for name in ('S_match', 'S0', 'B', 'W', 'U_plus_C', 'S_match_full'):
        assert by[name] and set(by[name]) == {'E0'}, name
    for name in ('Q', 'Q_full', 'Qo'):                   # no Qo_full cell among the 20
        assert by[name] and set(by[name]) == {'FAIL'}, name
    assert sum(len(v) for v in by.values()) == 20


def test_q_fails_on_main_ce_and_mtp_ce():
    q = next(c for c in CELLS if c['variant'] == 'Q')
    n = J.evaluate(H.parsed_cell(q), _ref())['details']['numerics']
    assert n['main_ce_rel_max'] == pytest.approx(6.912e-3, rel=1e-3) and n['main_ce_rel_mean_window'] > 0.002
    assert n['mtp_ce_rel_max'] > 0.005 and any(v.startswith('MTP CE rel dev') for v in n['violations'])


# ---------------------------------------------------------------- gate edges on synthetic screens
@pytest.mark.parametrize('kw,cls,needle', [
    ({}, 'E0', None),
    ({'ce_rel': 1e-4}, 'E1', None),
    ({'ce_rel': 6e-3}, 'FAIL', 'main CE rel dev'),
    ({'ce_rel': 3e-3}, 'FAIL', 'main CE window mean'),
    ({'mtp_rel': 1e-3}, 'E1', None),
    ({'mtp_rel': 3e-3}, 'FAIL', 'MTP CE window mean'),
    ({'mtp_rel': 6e-3}, 'FAIL', 'MTP CE rel dev'),
    ({'gn_rel': 0.02}, 'E1', None),
    ({'gn_rel': 0.04}, 'FAIL', 'grad norm rel dev at update 1'),
    ({'lr_add': 1e-15}, 'FAIL', 'LR differs'),
])
def test_gate_edges(tmp_path, kw, cls, needle):
    prof = H.scale_profile(H.profile_of(H.uc_a()), 0.8, **kw)
    v = J.evaluate(P.collect(_screen(tmp_path, prof)), _ref())
    assert v['numerics_class'] == cls, v['details']['numerics'].get('violations')
    assert v['judge_valid'] is (cls != 'FAIL')
    if needle:
        assert any(needle in r for r in v['reasons']), v['reasons']
    if cls != 'FAIL':
        assert v['metric_s'] == pytest.approx(0.8 * H.uc_a()['primary_window_17_30_mean_max32'], rel=1e-12)


def test_grad_norm_window_bias_is_gated(tmp_path):
    prof = H.profile_of(H.uc_a())
    for p in prof[1:]:
        p['gn'] *= 1.2                              # update 1 identical, a 20 % bias afterwards
    v = J.evaluate(P.collect(_screen(tmp_path, prof)), _ref())
    assert v['numerics_class'] == 'FAIL' and any('grad norm window log bias' in r for r in v['reasons'])


def test_repeated_adjacent_scalars_and_vanished_mtp_aux_are_invalid(tmp_path):
    prof = H.profile_of(H.uc_a())
    prof[5]['main_ce'], prof[5]['gn'] = prof[4]['main_ce'], prof[4]['gn']
    v = J.evaluate(P.collect(_screen(tmp_path, prof)), _ref())
    assert v['numerics_class'] == 'FAIL' and any('repeated adjacent' in r for r in v['reasons'])
    prof = H.profile_of(H.uc_a())
    prof[9]['mtp_aux'] = 0.0
    v = J.evaluate(P.collect(_screen(tmp_path / 'b', prof)), _ref())
    assert v['numerics_class'] == 'FAIL' and any('MTP aux loss vanished' in r for r in v['reasons'])


def test_e0_requires_mtp_and_aux_bitwise(tmp_path):
    prof = H.profile_of(H.uc_a())
    prof[20]['mtp_aux'] *= 1 + 1e-9
    v = J.evaluate(P.collect(_screen(tmp_path, prof)), _ref())
    assert v['numerics_class'] == 'E1' and v['details']['numerics']['bitwise'] is True    # the 4 tracked: bitwise
    assert v['details']['numerics']['bitwise_all'] is False


# ---------------------------------------------------------------- references
def test_no_reference_is_never_valid(tmp_path):
    p = P.collect(_screen(tmp_path))
    for role in ('candidate', 'reference'):
        v = J.evaluate(p, None, role=role)
        assert v['judge_valid'] is False and 'no numerics reference' in v['reasons'][-1]


def test_reference_from_a_signed_receipt(tmp_path):
    from infrarsi.agent.budget import VirtualClock
    uc = H.uc_reference_receipt(tmp_path, clock=VirtualClock())
    v = J.evaluate(P.collect(_screen(tmp_path)), uc)
    assert v['judge_valid'] and v['numerics_class'] == 'E0' and v['details']['reference']['kind'] == 'receipt'
    forged = dict(uc, valid=True)
    forged['metric'] = dict(uc['metric'], value=1.0)                    # changed after signing
    v = J.evaluate(P.collect(_screen(tmp_path)), forged)
    assert v['judge_valid'] is False and any('signature' in r for r in v['reasons'])


def test_an_incomplete_reference_is_refused(tmp_path):
    ref = P.collect(_screen(tmp_path, drop_ranks=(3,)))
    v = J.evaluate(P.collect(_screen(tmp_path / 'b')), ref)
    assert v['judge_valid'] is False and any('reference screen is not complete' in r for r in v['reasons'])


# ---------------------------------------------------------------- parse failure modes
def _broken(tmp_path, mutate):
    root = _screen(tmp_path)
    mutate(root)
    p = P.collect(root)
    v = J.evaluate(p, _ref())
    assert p['complete'] is False and p['metric']['value'] is None
    assert v['judge_valid'] is False and v['metric_s'] is None and v['reasons'][0].startswith('incomplete screen')
    return p


def _edit_rank(root, r, fn):
    path = root / 'signals' / f'rank{r:02d}.jsonl'
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    fn(rows)
    path.write_text('\n'.join(json.dumps(x) for x in rows) + '\n')


def test_missing_rank_log(tmp_path):
    p = _broken(tmp_path, lambda root: (root / 'signals' / 'rank07.jsonl').unlink())
    assert p['completeness']['n_ranks_ok'] == 31 and p['steps'] == []


def test_missing_update(tmp_path):
    p = _broken(tmp_path, lambda root: _edit_rank(root, 4, lambda rows: rows.pop()))
    assert any('rank' in str(x) for x in p['completeness']['rank_problems'])


def test_sequence_gap_and_truncated_record(tmp_path):
    _broken(tmp_path, lambda root: _edit_rank(root, 2, lambda rows: rows[3].update(sequence=99)))
    root = _screen(tmp_path / 't')
    path = root / 'signals' / 'rank05.jsonl'
    path.write_bytes(path.read_bytes()[:-20])                            # torn last record, no newline
    assert P.collect(root)['complete'] is False


def test_foreign_attempt_rows(tmp_path):
    p = _broken(tmp_path, lambda root: _edit_rank(root, 9, lambda rows: rows[0].update(attempt_id='other')))
    assert any('attempt_id' in s for x in p['completeness']['rank_problems'] for s in x['problems'])


def test_progress_contract(tmp_path):
    _broken(tmp_path, lambda root: _edit_rank(root, 0, lambda rows: rows[2]['values']['progress'].update(horizon=2236)))


def test_non_finite_timing(tmp_path):
    _broken(tmp_path, lambda root: _edit_rank(root, 1, lambda rows: rows[2]['values']['timing'].update(update_seconds=0)))


def test_node_exit_and_release(tmp_path):
    root = _screen(tmp_path, node_exit={2: 1})
    p = P.collect(root)
    assert p['complete'] is False and p['completeness']['nodes_ok'] is False
    root = _screen(tmp_path / 'r')
    (root / 'nodes' / 'node1.release.json').write_text(json.dumps({'idle': False, 'processes': '123',
                                                                   'process_group_gone': True, 'hostname': H.HOSTS['WINH3']}))
    assert P.collect(root)['complete'] is False


def test_missing_attempt_record(tmp_path):
    _broken(tmp_path, lambda root: (root / 'attempt.json').unlink())


def test_w_excludes_update_1_from_timing_only(tmp_path):
    w = S.registered_points()['W']
    p = P.collect(H.write_screen(tmp_path / 'w', H.profile_of(H.uc_a()), w))
    assert p['excluded_timing_steps'] == [1] and p['complete']
    assert p['metric']['value'] == pytest.approx(H.uc_a()['primary_window_17_30_mean_max32'], abs=1e-12)


def test_production_layout_is_found_too(tmp_path):
    """The parser also reads the production names (<run>.signals.rank<r>/signals.jsonl, <run>.rank<r>.json)."""
    root = _screen(tmp_path)
    run = json.loads((root / 'attempt.json').read_text())['run']
    for r in range(32):
        d = root / 'training' / f'{run}.signals.rank{r}'
        d.mkdir(parents=True)
        (root / 'signals' / f'rank{r:02d}.jsonl').rename(d / 'signals.jsonl')
        (root / 'ranks' / f'rank{r:02d}.json').rename(root / 'training' / f'{run}.rank{r}.json')
    p = P.collect(root)
    assert p['complete'] and all(x is not None for x in p['rank_results'])


# ---------------------------------------------------------------- the verdict and the receipt view
def test_verdict_shape_and_receipt_view(tmp_path):
    v = J.evaluate(P.collect(_screen(tmp_path)), _ref())
    assert set(v) == {'judge_valid', 'numerics_class', 'reasons', 'metric_s', 'details'}
    json.dumps(jsonable(v), allow_nan=False)
    d = v['details']
    assert d['gates']['sha256'] == J.gates_sha256() and d['gates']['tier'] == 'moe_nondeterministic'
    assert len(d['per_step']) == 30 and set(d['per_step'][0]) == {'update', 'main_ce', 'mtp_ce', 'aux_loss', 'mtp_aux',
                                                                   'grad_norm', 'lr'}
    assert d['inter_step_consistency'].startswith('not_applicable')
    view = d['view']
    assert len(view['per_step_max_rank_s']) == 30 and view['numerics']['class'] == 'E0'
    assert {'memory.peak_reserved_gib', 'timing.update_seconds', 'gradients.global_norm', 'tokens.global_physical'} <= set(view['values'])
    assert not any(k.startswith(('microbatches', 'learning_rates.groups')) for k in view['values'])
    assert view['per_rank_spread']['slowest_rank'] is not None and view['per_rank_spread']['window_mean_spread_s'] > 0
    assert d['memory']['max_peak_reserved_gib'] == pytest.approx(70.31)
    extra = J.receipt_extra(v)
    assert extra['summary'] is view and extra['memory'] == d['memory'] and extra['numerics']['class'] == 'E0'


def test_window_metric_formula():
    steps = [{'update': u, 'max_rank_s': float(u)} for u in range(1, 31)]
    m, n = P.window_metric(steps)
    assert n == 14 and m == pytest.approx(sum(range(17, 31)) / 14)
    assert P.window_metric(steps[:-1]) == (None, 13)
    assert math.isclose(P.window_metric(steps, excluded=[1])[0], m)
