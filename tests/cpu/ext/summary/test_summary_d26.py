"""D26 (O's ruling 10-21): accounting flags that hide no resource use. (a) a D24 pre-launch refusal receipt charged as
recorded counts as complete; (b) B-NAT's "an LLM call ended within N s of a row (attribution ambiguous)" counts as
complete, note shown in the arm's row; (c) S_W1 keeps B-ISW's WINH1 default runs, its numeric effect computed. Each rule
is recognised from the records, never by a blanket override (Codex R4-N1 / R4-N2): the receipts and call records carry
explicit costs, the flags form a suffix that starts where the cause starts, and the trace's totals reconcile with the
receipt charges and the call log; anything else keeps the arm `incomplete`."""
import json

import pytest
from summary_helpers import World, write

from infrarsi.ext.optico import workloads as OW
from infrarsi.ext.summary import env_b
from infrarsi.ext.summary import records as RC

NEAR = 'an LLM call ended within 5 s of a row (attribution ambiguous)'
D24 = {'kind': 'runner_error', 'type': 'AttemptDeadline', 'message': 'pre_launch deadline passed before launch (3 s late)'}
POST = ['infra_refusal: the pre-launch phase (180 s) ended before trial.sh was launched', 'runner_error: AttemptDeadline: x']


def build_b(w):
    return env_b.build(RC.Store(w.root), w.root, w.ledger, abandoned=RC.abandoned(w.ledger, w.root))


def resign(path, **changes):
    r = json.loads(path.read_text(encoding='utf-8'))
    r.pop('harness_signature', None)
    r.update(changes)
    for k in [k for k, v in changes.items() if v is DROP]:
        r.pop(k)
    write(path, RC.signing.sign(r))


DROP = object()


def _trace(w, rid):
    p = w.root / 'optico' / rid / 'trace.json'
    return p, json.loads(p.read_text(encoding='utf-8'))


def _receipt(w, rid, att):
    return json.loads((w.root / 'optico' / rid / 'receipts' / f'{att}.json').read_text(encoding='utf-8'))


def _as_ledger(w, rid, tr, *, flag_from=None):
    """Make a synthetic InfraRSI trace ledger-consistent, as agent/budget.py writes it: cumulative GPU-hours = the
    running sum of the screens' receipt charges, tokens = one complete call record per screen, flags from `flag_from`."""
    run = 0.0
    for i, s in enumerate(tr['screens']):
        g = (_receipt(w, rid, s['attempt_id'])['cost']['gpu_seconds'] or 0.0) / 3600.0      # an uncharged refusal: 0
        run += g
        s['cum'] = dict(s['cum'], gpu_hours=run, llm_tokens=1000 * min(i + 1, 3), llm_usd=0.0,
                        accounting_complete=flag_from is None or i < flag_from)
        s['gpu_hours'] = g
    tr['llm_calls'] = [{'turn': k + 1, 'call_id': f'{rid}-call{k}', 'llm_tokens': 1000, 'llm_usd': 0.0,
                        'accounting_complete': True} for k in range(3)]
    tr['used'] = dict(tr['used'], gpu_hours=run, llm_tokens=3000, llm_usd=0.0, accounting_complete=flag_from is None)
    tr['accounting_complete'] = flag_from is None
    ev = []                                              # the producer's order: a call, then the screen it preceded
    for k in range(len(tr['screens'])):
        if k < 3:
            ev.append({'kind': 'llm_call', 'payload': {'call_id': f'{rid}-call{k}'}})
        ev.append({'kind': 'screen', 'payload': {'index': k}})
    (w.root / 'optico' / rid / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in ev), encoding='utf-8')


def _with_refusal(w, rid='bis_W3', arm='B-IS', flag_from=None, **over):
    """Append a D24 pre-launch refusal screen (charged: 8 GPUs x 80 s held) to a W3 search, with the ledger's totals
    and flags (incomplete from the refusal screen on)."""
    d = w.root / 'optico' / rid
    p, tr = _trace(w, rid)
    att = f'{rid}-c9'
    f = write(d / 'receipts' / f'{att}.json', w.b_receipt('W3', rid, att, 'c9', arm=arm, purpose='screen', T=1.0,
                                                          valid=False, runner_error=D24))
    changes = {'judge': None, 'phases': {'total': 80.0},
               'deadline': {'refused_calls': [{'phase': 'pre_launch', 'func': 'launch', 'late_s': 3.0}]},
               'cost': {'gpu_seconds': 640.0, 'wall_seconds': 80.0, 'accounting_complete': False,
                        'basis': 'world_size x seconds the GPUs were held (materialize, run, judge, collect)'}}
    changes.update(over)
    resign(f, **changes)
    idx = len(tr['screens'])
    tr['screens'].append({'index': idx, 'candidate_id': 'c9', 'attempt_id': att, 'valid': False, 'step_time': None,
                          'cum': dict(tr['screens'][-1]['cum'])})
    _as_ledger(w, rid, tr, flag_from=idx if flag_from is None else flag_from)
    write(p, tr)
    return p, f


def _acc(sec, arm, w='W3'):
    s = sec['workloads'][w]['arms'][arm]['search']
    return s['accounting'][0], s.get('accounting_d26')


# ---------------------------------------------------------------- (a)
def test_d26a_a_charged_prelaunch_refusal_counts_as_complete(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    _with_refusal(w)
    sec = build_b(w)
    ok, basis = _acc(sec, 'B-IS')
    assert ok is True and 'D26(a)' in basis and '0.18 GPUh' in basis and 'in the trace totals' in basis
    rec = sec['workloads']['W3']['arms']['B-IS']['search']['accounting_reconciliation']
    assert rec['mismatches'] == [] and rec['refusal_charges_gpuh'] == {'c9': pytest.approx(640 / 3600, abs=1e-6)}
    assert sec['workloads']['W3']['labels']['B-IS vs B-NAT']['label'] != 'incomplete'
    assert any('complete under D26(a)' in f for f in sec['findings'])
    assert 'complete (D26(a)' in env_b.markdown(sec)


def test_d26a_the_post_1405_refusal_form(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    _with_refusal(w, runner_error=None, not_launched='pre_launch', invalid_reasons=POST)
    assert _acc(build_b(w), 'B-IS')[0] is True


@pytest.mark.parametrize('over', [
    {'runner_error': {'kind': 'occupied', 'message': 'OccupiedError: 1 foreign compute process'}},   # not a D24 refusal
    {'judge': {'score': 1.0}},                                                     # a judge ran: something was launched
    {'deadline': {'refused_calls': [{'phase': 'post', 'func': 'collect', 'late_s': 1.0}]}},          # refused after launch
    {'phases': {'total': 80.0, 'trial_s': 40.0}},                                  # a launch phase
    {'cost': {'gpu_seconds': None, 'wall_seconds': 80.0, 'accounting_complete': False, 'basis': 'x'}},  # not charged
    # R4-N2: the post-14:05 form needs the same evidence
    {'runner_error': None, 'not_launched': 'pre_launch', 'invalid_reasons': POST, 'judge': {'score': 1.0}},
    {'runner_error': None, 'not_launched': 'pre_launch', 'invalid_reasons': POST, 'phases': {'total': 9, 'monitor_s': 5}},
])
def test_d26a_is_recognised_from_the_receipt_not_assumed(tmp_path, over):
    w = World(tmp_path)
    w.env_b_workload('W3')
    _with_refusal(w, **over)
    sec = build_b(w)
    assert _acc(sec, 'B-IS') == (False, None)
    assert sec['workloads']['W3']['labels']['B-IS vs B-NAT']['label'] == 'incomplete'


def test_d26a_needs_every_other_flag_explained(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    _with_refusal(w, flag_from=1)                                         # flagged before the refusal screen
    assert _acc(build_b(w), 'B-IS') == (False, None)
    w = World(tmp_path / 'b')
    w.env_b_workload('W3')
    _with_refusal(w)
    f = w.root / 'optico' / 'bis_W3' / 'receipts' / 'bis_W3-c1.json'      # another receipt with an unknown cost
    resign(f, cost={**json.loads(f.read_text(encoding='utf-8'))['cost'], 'accounting_complete': False})
    assert _acc(build_b(w), 'B-IS') == (False, None)
    w = World(tmp_path / 'c')
    w.env_b_workload('W3')
    p, _ = _with_refusal(w)
    tr = json.loads(p.read_text(encoding='utf-8'))
    tr['llm_calls'][0]['accounting_complete'] = False                     # unknown LLM usage
    write(p, tr)
    assert _acc(build_b(w), 'B-IS') == (False, None)


# ---------------------------------------------------------------- R4-N1: explicit costs, finite totals, reconciliation
@pytest.mark.parametrize('case', ['receipt_no_cost', 'receipt_no_flag', 'call_no_flag', 'call_no_usd'])
def test_d26_missing_cost_or_completeness_is_not_complete(tmp_path, case):
    w = World(tmp_path)
    w.env_b_workload('W3')
    p, _ = _with_refusal(w)
    f = w.root / 'optico' / 'bis_W3' / 'receipts' / 'bis_W3-c1.json'
    if case == 'receipt_no_cost':
        resign(f, cost=DROP)
    elif case == 'receipt_no_flag':
        c = json.loads(f.read_text(encoding='utf-8'))['cost']
        c.pop('accounting_complete')
        resign(f, cost=c)
    else:
        tr = json.loads(p.read_text(encoding='utf-8'))
        tr['llm_calls'][1].pop('accounting_complete' if case == 'call_no_flag' else 'llm_usd')
        write(p, tr)
    assert _acc(build_b(w), 'B-IS') == (False, None)


def test_d26a_a_refusal_charge_missing_from_the_totals_is_named_not_recomputed(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    p, _ = _with_refusal(w)
    tr = json.loads(p.read_text(encoding='utf-8'))
    tr['screens'][-1]['cum']['gpu_hours'] = tr['screens'][-2]['cum']['gpu_hours']    # the refusal charge left out
    tr['used']['gpu_hours'] = tr['screens'][-2]['cum']['gpu_hours']
    write(p, tr)
    sec = build_b(w)
    s = sec['workloads']['W3']['arms']['B-IS']['search']
    assert s['accounting'][0] is False and s['accounting_d26'] is None
    assert any('D26 not granted' in x and 'running sum' in x for x in s['accounting'][1])
    assert s['accounting_reconciliation']['mismatches']
    w = World(tmp_path / 'z')                                             # Codex's case: used.gpu_hours = 0.0
    w.env_b_workload('W3')
    p, _ = _with_refusal(w)
    tr = json.loads(p.read_text(encoding='utf-8'))
    tr['used']['gpu_hours'] = 0.0
    write(p, tr)
    assert _acc(build_b(w), 'B-IS') == (False, None)


def test_d26_an_llm_arm_needs_finite_token_and_usd_totals_equal_to_the_call_log(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    p, _ = _with_refusal(w)
    tr = json.loads(p.read_text(encoding='utf-8'))
    tr['used']['llm_tokens'] = 2000                                       # one call's tokens missing
    write(p, tr)
    assert _acc(build_b(w), 'B-IS') == (False, None)
    tr['used']['llm_tokens'], tr['used']['llm_usd'] = None, None
    write(p, tr)
    assert _acc(build_b(w), 'B-IS') == (False, None)


# ---------------------------------------------------------------- (b)
def _near(w, *, notes=(NEAR,), near=(2,), matches=True, flags=(True, True, False), row_tokens=None, used_tokens=3000):
    """B-NAT (native OptiCo loop, lease-based GPU) with three rows and three calls: call 0 before row 0, call 1 after
    row 0, call 2 after row 1; `near` marks the near-boundary call(s); per-row tokens = the calls ended before the row."""
    p, tr = _trace(w, 'bnat_W3')
    tr['mode'] = 'native'
    for i, s in enumerate(tr['screens']):
        s['cum'] = dict(s['cum'], llm_tokens=(row_tokens or (1000, 2000, 3000))[i], llm_usd=0.0,
                        accounting_complete=flags[i])
    tr['accounting_complete'] = False
    tr['used'] = dict(tr['used'], llm_tokens=used_tokens, llm_usd=0.0, accounting_complete=False)
    tr['llm_calls'] = [{'call': k, 'after_screen': (None, 0, 1)[k], 'near_row_boundary': k in near, 'llm_tokens': 1000,
                        'llm_usd': 0.0, 'accounting_complete': True} for k in range(3)]
    tr['extra'] = {'accounting_notes': list(notes), 'calls_near_row_boundary': len(near),
                   'llm_log': {'calls': 3, 'llm_tokens': 3000, 'llm_usd': 0.0, 'matches_shim': matches}}
    write(p, tr)


def test_d26b_the_near_boundary_note_counts_as_complete_and_is_shown(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W3')
    _near(w)
    sec = build_b(w)
    ok, basis = _acc(sec, 'B-NAT')
    assert ok is True and 'D26(b)' in basis and NEAR in basis and '3000 tokens' in basis
    assert sec['workloads']['W3']['labels']['B-IS vs B-NAT']['label'] != 'incomplete'
    row = next(x for x in env_b.markdown(sec).splitlines() if x.startswith('| B-NAT |'))
    assert 'attribution ambiguous' in row


@pytest.mark.parametrize('kw', [
    {'notes': (NEAR, 'the node lease was not released (GPU-hours still running)')},     # another note: not covered
    {'notes': (NEAR, NEAR)},                                                            # R4-N2: duplicate notes
    {'near': (1, 2)},                                                                   # R4-N2: two near-boundary calls
    {'matches': False},                                                                 # the shim log does not match
    {'flags': (True, False, False)},                                                    # flagged before the call's row
    {'flags': (True, True, True)},                                                      # R4-N2: no flagged screen
    {'flags': (False, True, False)},                                                    # R4-N2: flags not a suffix
    {'row_tokens': (1000, 2000, 2000)},                                                 # R4-N1: a row omits the call
    {'used_tokens': 2000},                                                              # R4-N1: final omits a call
    {'used_tokens': None},                                                              # R4-N1: no finite token total
])
def test_d26b_is_narrow(tmp_path, kw):
    w = World(tmp_path)
    w.env_b_workload('W3')
    _near(w, **kw)
    assert _acc(build_b(w), 'B-NAT') == (False, None)


# ---------------------------------------------------------------- (c)
def _isw_on_winh1(w, default):
    p2 = {**OW.get('W1').defaults, 'MICRO_BATCH_SIZE': 2}
    sel = w.b_search('W1', 'B-ISW', 'bisw_W1', [default, None, 330.0], best_params=p2)
    w.b_locked('W1', 'B-ISW', 'lk_bisw_W1', default, 336.0, params=sel)
    for rid in ('bisw_W1', 'lk_bisw_W1'):
        for f in (w.root / 'optico' / rid / 'receipts').glob('*.json'):
            resign(f, node='WINH1')


def test_d26c_a_two_node_s_w_is_kept_and_its_effect_computed(tmp_path):
    w = World(tmp_path)
    w.env_b_workload('W1')
    _isw_on_winh1(w, 229.0)
    sec = build_b(w)
    W = sec['workloads']['W1']
    assert {x['run_id'] for x in W['margin_record']['S_w']} >= {'bisw_W1', 'lk_bisw_W1'}        # kept in S_W1
    assert len(W['s_w_nodes']['other_node_samples']) == 2
    assert W['s_w_nodes']['statement'].endswith('no numeric effect (3 % floor)')
    assert 'no numeric effect (3 % floor)' in env_b.markdown(sec)
    _isw_on_winh1(w, 260.0)                                               # a far default: the effect is stated
    W = build_b(w)['workloads']['W1']
    assert 'without them vs' in W['s_w_nodes']['statement'] and W['s_w_nodes']['margin_without_them'] == pytest.approx(0.03)


def test_a_records_margin_file_with_abandoned_defaults_is_named(tmp_path):
    # the 16:17 margin_W1.json case: the records file was written over every run incl. abandoned ones (#29)
    w = World(tmp_path)
    w.env_b_workload('W3')
    w.ledger['ENV-B']['W3']['abandoned'] = [{'run_id': 'bis_W3_old', 'reason': '#29'}]
    mine = build_b(w)['workloads']['W3']['margin_record']
    write(w.root / 'optico' / 'margin_W3.json',
          {'S_w': mine['S_w'] + [{'attempt_id': 'old-c0', 'run_id': 'bis_W3_old', 'arm': 'B-IS', 'kind': 'search_start'}],
           'estimator': mine['estimator'], 'margin': mine['margin'], 'n': mine['n'] + 1})
    p = next(x for x in build_b(w)['problems'] if 'margin_W3.json differs' in x)
    assert "from abandoned runs ['bis_W3_old']" in p and 'no numeric effect' in p and 'lacks 0' in p
