"""Builder A's dispatcher arms end to end through D's LocalRemote simulation (no GPU, no paid LLM).

The registered handlers (infrarsi.dispatch.handlers.handlers()) run real harness.run_screen attempts against
LocalRemote, whose launcher writes simulated attempts with tests/gpu/driver/dummy_entry.simulate_attempt. FR and ED
references are produced the same way, the dispatcher's run-scoped lease is held, and every request must use the
run's own run id.
"""
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from infrarsi.agent import arms
from infrarsi.agent.backends import ScriptedBackend
from infrarsi.agent.harness_runner import HarnessScreenRunner, gpus_per_node
from infrarsi.contracts import RunRequest, TaskSpec
from infrarsi.discover import default_profile
from infrarsi.discover import workload
from infrarsi.dispatch import QueueStore
from infrarsi.dispatch.handlers import Context, handlers
from infrarsi.harness import leases, run_screen
from infrarsi.harness.remote import LocalRemote
from infrarsi.planner import EDChain, FRChain

_DH_PATH = Path(__file__).resolve().parents[1] / 'driver' / 'driver_helpers.py'
_spec = importlib.util.spec_from_file_location('driver_helpers_for_agent_tests', _DH_PATH)
DH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DH)                     # D's helpers by path (never `from conftest import`)


LIFECYCLE_OK = {'resumed_ok': True, 'resume_scalars_match': True, 'export_ok': True, 'forward_ok': True, 'state_exact': True,
                'continuation': {'match': True, 'max_main_ce_rel': 2e-4, 'max_grad_norm_rel': 0.12,
                                 'steps': [{'step': 41, 'main_ce_rel': 1.6e-5, 'grad_norm_rel': 1.6e-5, 'ok': True},
                                           {'step': 42, 'main_ce_rel': 2e-4, 'grad_norm_rel': 0.115, 'ok': True}]},
                'export': {'export_ok': True, 'missing': [], 'extra': [], 'shape_mismatch': []},
                'forward': {'rel_l2': 0.004, 'top1_agreement': 0.998, 'forward_ok': True}}
SIM = {'lifecycle': LIFECYCLE_OK}                # tests may replace the lifecycle block the simulation reports


def _sim(req, out):
    """Step time depends on the config, so a search can find a faster candidate; scalars do not (numerics E0).
    Lifecycle attempts use D's runtime-layout simulation (save_at + resume_steps updates, the resumed rows in
    signals_resume/ bound by lifecycle.signals_resume_digest) with the lifecycle block in SIM['lifecycle']."""
    cfg = req['config']
    t = 0.5 / (1 + 0.15 * (int(cfg.get('micro_batch_size') or 1) - 1))
    if cfg.get('recompute_granularity') == 'full':
        t *= 1.3
    step_time = lambda r, u, _t=t: _t + 0.001 * r                  # noqa: E731
    if req['purpose'] != 'lifecycle':
        DH.DUMMY_MOD.simulate_attempt(req, out, step_time=step_time)
        return
    DH.DUMMY_MOD.simulate_lifecycle(req, out, flags=SIM['lifecycle'], step_time=step_time)


@pytest.fixture
def world(tmp_path, monkeypatch):
    root = Path(tempfile.mkdtemp(prefix='ias'))          # short root (Windows MAX_PATH)
    remote = LocalRemote(root, launcher=DH.sim_launcher(_sim))
    DH.fake_index(remote)                                  # packed-data INDEX: run_screen fails closed without it
    monkeypatch.setattr(workload, 'CACHE', DH.HF)          # model facts from D's HF fixtures, no remote read
    monkeypatch.setattr(arms, 'PACKAGES_DIR', tmp_path / 'packages')
    monkeypatch.setattr(arms, 'LINEAGE_DIR', tmp_path / 'packages' / 'lineage')
    monkeypatch.setattr(arms, 'COST_LEDGER', tmp_path / 'ledger' / 'costs.jsonl')
    store = QueueStore(tmp_path / 'queue')

    def runner(req, **kw):
        return run_screen(req, remote=remote, poll_s=0.01, local_dir=None, log=lambda m: None, **kw)

    task = TaskSpec.load('S1')
    facts = workload.workload_facts(task, cache_dir=DH.HF)
    fr_cfg = FRChain(task, facts).current()
    ed_cfg = EDChain(task, facts, default_profile(1)).queue[0]['config']
    fr = runner(RunRequest(run_id='E0-CAL-S1-FR', purpose='screen', task=task.raw, config=fr_cfg, nodes=['WINH0'],
                           gpus_per_node=1, input_plan='A', numerics_role='reference'))
    ed = runner(RunRequest(run_id='E1-S1-ED', purpose='screen', task=task.raw, config=ed_cfg, nodes=['WINH0'], gpus_per_node=1,
                           input_plan='A', reference_receipt=fr['harness']['stored']), reference=fr)
    fr_b = runner(RunRequest(run_id='E0-CAL-S1-FR', purpose='score', task=task.raw, config=fr_cfg, nodes=['WINH0'],
                             gpus_per_node=1, input_plan='B', numerics_role='reference'))
    ed_b = runner(RunRequest(run_id='E1-S1-ED', purpose='score', task=task.raw, config=ed_cfg, nodes=['WINH0'], gpus_per_node=1,
                             input_plan='B', reference_receipt=fr_b['harness']['stored']), reference=fr_b)
    assert fr['valid'] and ed['valid'] and fr_b['valid'] and ed_b['valid'], (fr['invalid_reasons'], ed['invalid_reasons'])
    store.set_reference('S1', fr_status='feasible', fr_config=fr_cfg, fr_screen=fr['harness']['stored'],
                        fr_score=fr_b['harness']['stored'], ed_status='feasible', ed_config=ed_cfg,
                        ed_screen=ed['harness']['stored'], ed_score=ed_b['harness']['stored'], ed_numerics_role='candidate')
    ctx = Context(store=store, remote=remote, runner=runner, searches_root=tmp_path / 'searches', log=lambda m: None)
    yield {'remote': remote, 'ctx': ctx, 'task': task, 'fr': fr, 'fr_b': fr_b, 'ed': ed, 'ed_b': ed_b, 'tmp': tmp_path,
           'runner': runner, 'fr_cfg': fr_cfg, 'ed_cfg': ed_cfg}
    shutil.rmtree(root, ignore_errors=True)


def _dispatch(world, run):
    """Hold the dispatcher's run-scoped lease (as Dispatcher._claim does) and call the registered handler."""
    r = world['remote']
    leases.acquire(r, ['WINH0'], leases.make_lease(run['run_id'], 'run:' + run['run_id'], ['WINH0'], purpose='dispatch', scope='run'))
    try:
        return handlers()[run['arm']](run, ['WINH0'], world['ctx'])
    finally:
        leases.release(r, ['WINH0'], 'run:' + run['run_id'])


def _requests(remote, run_id):
    d = remote.local(f'/data/infrarsi/runs/{run_id}')
    return [json.loads(p.read_text()) for p in sorted(d.glob('*/request.json'))] if d.exists() else []


def _check_scoring_inputs(res):
    from infrarsi.scoring.score import Event, score_task
    si = res['scoring_inputs']
    assert si['eligible'] is True and si['t_ed'] is not None and not si['ed_denominator']['problems']
    events = [Event(**{k: e[k] for k in ('gpuh', 'usd', 'wall_s', 'tokens', 'step_time', 'kind', 'note')}) for e in si['events']]
    row = score_task(si['task_id'], si['arm'], t_fr_locked=si['t_fr_locked'], t_cand_locked=si['t_cand_locked'],
                     fr_rsd=si['fr_rsd'], t_ed=si['t_ed'], events=events, caps=si['caps'], invalid_accept=si['invalid_accept'],
                     numerics_class=si['numerics_class'], lifecycle_pass=si['lifecycle_pass'], t_fr_screen=si['t_fr_screen'])
    assert row['valid_candidate'] and row['Speedup_FR'] and row['Speedup_ED'] and 'AUC_budget' in row
    return row


def test_random_l0_handler_end_to_end(world):
    run = {'run_id': 'E2-S1-random_L0-sim', 'task': 'S1', 'arm': 'random_L0', 'screens': 3, 'wall_hours_cap': 1,
           'llm_tokens_cap': 0, 'seed': 42, 'priority': 'P0'}
    out = _dispatch(world, run)
    res = out['result']
    out_dir = Path(out['search_dir'])
    assert res['trace']['screens'] == 3 and res['trace']['stop_reason'] == 'budget:screens'
    assert res['trace']['judge'] == 'RewardEngineJudge' and res['trace']['accounting_complete']
    reqs = _requests(world['remote'], run['run_id'])
    life = res['promotion_lifecycle']
    assert life['ran'] and life['lifecycle_pass'], life                               # the promoted config is not FR's
    assert len(reqs) == 3 + 2 + 1 and {q['run_id'] for q in reqs} == {run['run_id']}      # screens + locked pair + lifecycle
    roles = sorted((q['purpose'], q['input_plan'], q['numerics_role']) for q in reqs)
    assert roles == [('lifecycle', 'B', 'candidate'), ('score', 'B', 'candidate'), ('score', 'B', 'reference')] + \
        [('screen', 'A', 'candidate')] * 3                                              # only FR plan B is a base run
    lc_req = next(q for q in reqs if q['purpose'] == 'lifecycle')
    assert lc_req['reference_receipt'] and lc_req['reference_receipt'] != world['fr']['harness']['stored']  # paired FR plan B
    assert res['scoring_inputs']['lifecycle_pass'] is True and res['cost']['promotion_lifecycle_gpuh'] > 0
    assert res['scoring_inputs']['t_ed'] == world['ed_b']['step_time']['mean_window_max_rank_s']
    assert all(q['reference_receipt'] == world['fr']['harness']['stored'] for q in reqs if q['input_plan'] == 'A')
    assert not any(k.startswith('_') for q in reqs for k in q['config'])
    pair = res['locked_pair']
    assert pair['order'] == ['candidate', 'FR'] and pair['rechecked'] is True and pair['reference_kind'] == 'FR'                # S1: odd task index
    assert pair['invalid_accept'] == 0 and pair['t_fr_locked'] and pair['t_cand_locked']
    assert res['cost']['total_gpu_hours'] == pytest.approx(res['cost']['search_budget_gpu_hours'] + res['cost']['promotion_lifecycle_gpuh'])
    assert pair['candidate']['verdict']['source'] == 'reward_engine' and pair['candidate']['verdict']['valid']
    assert (out_dir / 'trace.json').exists() and (out_dir / 'result.json').exists()
    assert (out_dir / 'locked' / 'candidate_plan_b.json').exists()
    assert res['cost']['scoring_gpu_hours'] > 0 and len(res['harness_attempts']) == 6
    _check_scoring_inputs(res)
    lease_dir = world['remote'].local('/data/infrarsi/leases')
    assert not list(lease_dir.glob('*.json'))                                                # nothing left behind


def test_a0_handler_end_to_end_with_scripted_driver(world, monkeypatch):
    replies = [{'action': 'propose_config', 'base': 'c0', 'changes': {'micro_batch_size': 4}, 'rationale': 'larger MBS'},
               {'action': 'run_screen', 'candidate_id': 'c1'},
               {'action': 'finish', 'candidate_id': 'c1'}]
    calls = []

    def script(role, system, prompt):
        calls.append(role)
        if role == 'verifier':
            return {'verdict': 'pass', 'reasons': []}
        return replies.pop(0) if replies else {'action': 'finish'}
    monkeypatch.setitem(arms.DRIVERS, 'strong', lambda: ScriptedBackend(script))
    run = {'run_id': 'E1-S1-A0-sim', 'task': 'S1', 'arm': 'A0', 'screens': 16, 'wall_hours_cap': 3, 'llm_tokens_cap': 1500000,
           'seed': 42, 'priority': 'P0'}
    out = _dispatch(world, run)
    res = out['result']
    assert res['package']['label'] == 'A0' and res['trace']['package_digest'] == res['package']['digest']
    assert res['trace']['stop_reason'] == 'finish' and res['trace']['llm_calls'] >= 3
    assert 'proposer' in calls
    reqs = _requests(world['remote'], run['run_id'])
    assert {q['run_id'] for q in reqs} == {run['run_id']} and len(reqs) == res['trace']['screens'] + 2 + 1
    assert res['locked_pair']['invalid_accept'] == 0
    trace = json.loads((Path(out['search_dir']) / 'trace.json').read_text())
    assert trace['selected'] == 'c1' and trace['selected_config']['micro_batch_size'] == 4      # ED has MBS 2
    assert trace['selected_step_time'] < trace['start_step_time'] and res['trace']['screens'] == 1
    assert res['locked_pair']['promoted'] and res['locked_pair']['t_cand_locked'] < res['locked_pair']['t_fr_locked']
    _check_scoring_inputs(res)


def test_registry_covers_builder_a_arms():
    from infrarsi.dispatch import load_matrix
    mine = set(arms.AGENT_ARMS) | set(arms.RSI_SEARCH_ARMS) | {
        'weak_endpoint_check', 'R_arm_reflect', 'M_arm_design', 'random_L0', 'tpe_L0', 'optico_style_L0', 'openevolve_L0L1',
        'redteam_seed_cpu_check', 'redteam_seeds_lenient_vs_full', 'fault_injection_pair'}
    h = handlers()
    assert mine <= set(h)
    in_matrix = {r['arm'] for r in load_matrix()['runs']}
    assert mine <= in_matrix
    assert 'A0_plus_raw_dev_memory' in h


class _Boom:
    def __init__(self, exc):
        self.exc, self.seen = exc, []

    def __call__(self, req, **kw):
        self.seen.append((req, kw))
        raise self.exc


def test_harness_runner_contract():
    from infrarsi.harness import OccupiedError
    from infrarsi.agent.search import SearchSession
    from infrarsi.agent.budget import Budget
    from infrarsi.agent.judge import ReceiptJudge
    task = TaskSpec.load('D5')                                       # 2 nodes x 8 GPUs
    t = iter([0.0, 30.0])
    boom = _Boom(OccupiedError({'WINH4': 'busy'}))
    hr = HarnessScreenRunner(boom, ['WINH4', 'WINH5'], reference_receipt_path='/data/infrarsi/receipts/fr.json',
                             reference_receipt={'attempt_id': 'fr'}, clock=lambda: next(t))
    rec = hr.run(task, {'micro_batch_size': 1, '_ple_cpu_offload': True}, {'host_sync.py': 'X = 1\n'}, 'screen', 'E1-D5-A0')
    req, kw = boom.seen[0]
    assert req.gpus_per_node == gpus_per_node(task) == 8 and req.input_plan == 'A' and req.run_id == 'E1-D5-A0'
    assert req.reference_receipt == '/data/infrarsi/receipts/fr.json' and kw['reference'] == {'attempt_id': 'fr'}
    assert kw['overlay_files'] == {'host_sync.py': b'X = 1\n'} and '_ple_cpu_offload' not in req.config
    assert rec['valid'] is False and rec['invalid_reasons'][0].startswith('occupied:')
    assert rec['cost'] == {'gpu_seconds': 480.0, 'wall_seconds': 30.0, 'accounting_complete': False,
                           'basis': 'world_size x elapsed (harness exception)'}
    req = hr.request(task, {}, 'score', 'x')
    assert req.input_plan == 'B' and req.reference_receipt is None
    req = hr.request(TaskSpec.load('S2'), {}, 'lifecycle', 'x')
    assert req.input_plan == 'B' and req.lifecycle['save_at'] == TaskSpec.load('S2').raw['workload']['screen_steps']
    assert req.gpus_per_node == 2
    with pytest.raises(ValueError):
        hr.request(task, {}, 'bogus', 'x')
    # A session charges the record's world x elapsed and marks the accounting incomplete.
    from agent_helpers import FR_DENSE
    s2 = TaskSpec.load('S2')
    t2 = iter([0.0, 0.0, 10.0, 10.0, 10.0])
    hr2 = HarnessScreenRunner(_Boom(OccupiedError({'WINH1': 'busy'})), ['WINH1'], clock=lambda: 5.0,
                              reference_receipt_path='/data/infrarsi/receipts/fr.json', reference_receipt={'attempt_id': 'fr'})
    sess = SearchSession(task=s2, budget=Budget(screens=4, llm_tokens=1000, wall_seconds=3600, gpu_hours=10), runner=hr2,
                         judge=ReceiptJudge(), run_id='E1-S2-A0', arm='A0', start_config=dict(FR_DENSE, tensor_model_parallel_size=1),
                         clock=lambda: next(t2, 10.0))
    fb = sess.screen('c0')
    assert fb['valid'] is False and not sess.ledger.accounting_complete and sess.ledger.used['screens'] == 1


def test_fault_injection_pair_uploads_overlays_under_the_run_id(world):
    """Plumbing of an overlay arm through the harness: overlay text goes up as bytes and is digested in request.json.
    (The simulation ignores overlays, so the detector verdicts here are not meaningful; GPU E4 decides those.)"""
    import hashlib
    from infrarsi.redteam.seeds import overlay
    world['ctx'].store.set_reference('S1', fr_status='feasible')
    run = {'run_id': 'E4-FAULT-INJECTION-PAIR-sim', 'task': 'S1', 'arm': 'fault_injection_pair', 'screens': 2,
           'wall_hours_cap': 1, 'seed': 42, 'priority': 'P0'}
    out = _dispatch(world, run)
    rows = out['result']['rows']
    assert set(rows) == {'positive_single_update', 'negative_cross_update'}
    assert 'G19' in rows['negative_cross_update']['static_codes'] and 'G19' in rows['positive_single_update']['static_codes']
    reqs = _requests(world['remote'], run['run_id'])
    assert len(reqs) == 2 and {q['run_id'] for q in reqs} == {run['run_id']}
    shas = sorted(q['overlay']['files']['recompute_policy.py'] for q in reqs)
    want = sorted(hashlib.sha256(overlay(s)['recompute_policy.py'].encode('utf-8')).hexdigest() for s in ('p01', 's01'))
    assert shas == want


def test_r_arm_search_then_reflect_registers_a1(world, monkeypatch):
    """E6 R arm on S1: dev search with the lineage head (A0), then the reflect handler writes A_1 into the lineage."""
    from infrarsi.rsi.lineage import Lineage
    replies = [{'action': 'propose_config', 'base': 'c0', 'changes': {'micro_batch_size': 4}},
               {'action': 'run_screen', 'candidate_id': 'c1'}, {'action': 'finish', 'candidate_id': 'c1'}]

    def script(role, system, prompt):
        if role == 'verifier':
            return {'verdict': 'pass', 'reasons': []}
        if role == 'reflect':
            return {'memory': [{'tags': {'model': ['qwen3-dense']}, 'lesson': 'MBS 4 beat MBS 2 at 1 GPU'}], 'rationale': 'S1'}
        return replies.pop(0) if replies else {'action': 'finish'}
    monkeypatch.setitem(arms.DRIVERS, 'strong', lambda: ScriptedBackend(script))
    search = {'run_id': 'E6-R-S1-search', 'task': 'S1', 'arm': 'R_arm_dev_search', 'screens': 16, 'wall_hours_cap': 3,
              'llm_tokens_cap': 1500000, 'seed': 42, 'priority': 'P0'}
    out = _dispatch(world, search)
    assert out['result']['package']['label'] == 'A0' and out['result']['locked_pair']['invalid_accept'] == 0
    reflect = {'run_id': 'E6-R-S1-reflect', 'task': 'S1', 'arm': 'R_arm_reflect', 'llm_tokens_cap': 300000, 'priority': 'P0'}
    res = handlers()['R_arm_reflect'](reflect, [], world['ctx'])
    assert res['status'] == 'changed' and res['parent_digest'] == out['result']['package']['digest']
    lin = Lineage(arms.LINEAGE_DIR)
    assert [r['status'] for r in lin.rows()] == ['initial', 'changed'] and lin.current()['digest'] == res['digest']
    events = (world['tmp'] / 'searches' / 'E6-R-S1-reflect' / 'events.jsonl').read_text().splitlines()
    assert json.loads(events[-1])['kind'] == 'llm_call'                      # visible to the monitor's spend sum
    assert (arms.COST_LEDGER).exists()


def test_weak_endpoint_check_handler(world, monkeypatch):
    from infrarsi.agent.backends.openai_compat import OpenAICompatBackend

    def transport(url, headers, body, timeout):
        assert body['guided_json'] and timeout <= 300
        return 200, {'model': 'Qwen3-Next-80B-A3B-Instruct', 'usage': {'prompt_tokens': 50, 'completion_tokens': 9},
                     'choices': [{'message': {'content': '{"action": "finish", "reason": "endpoint check"}'}}]}
    backend = OpenAICompatBackend(base_url='http://127.0.0.1:18000/v1', model='Qwen3-Next-80B-A3B-Instruct', api_key_env=None,
                                  price=None, structured='guided_json', transport=transport,
                                  pinger=lambda url, timeout: {'data': [{'id': 'Qwen3-Next-80B-A3B-Instruct'}]})
    monkeypatch.setitem(arms.DRIVERS, 'weak', lambda structured=True: ('selfhost', backend))
    res = handlers()['weak_endpoint_check']({'run_id': 'E0-WEAK-ENDPOINT-CHECK', 'task': 'none', 'arm': 'weak_endpoint_check'},
                                            [], world['ctx'])
    assert res['ok'] and res['route'] == 'selfhost' and res['accounting_complete'] and res['llm_tokens'] == 59
    ev = (world['tmp'] / 'searches' / 'E0-WEAK-ENDPOINT-CHECK' / 'events.jsonl').read_text().splitlines()
    assert json.loads(ev[0])['payload']['llm_tokens'] == 59


def test_redteam_seed_screens_run_under_the_run_lease(world):
    """E4 seeds x {lenient, full} through the harness under the dispatcher's run lease (task swapped to S1 for speed)."""
    run = {'run_id': 'E4-SEEDS-14x2-sim', 'task': 'S1', 'arm': 'redteam_seeds_lenient_vs_full', 'screens': 28,
           'wall_hours_cap': 3, 'seed': 42, 'priority': 'P0'}
    out = _dispatch(world, run)
    reqs = _requests(world['remote'], run['run_id'])
    assert len(reqs) == 28 and {q['run_id'] for q in reqs} == {run['run_id']}
    assert all(q['overlay'] and q['input_plan'] == 'A' for q in reqs)
    assert len(out['result']['harness_attempts']) == 28 and 'G2_pass' in out['result']['summary']


class _PairRunner:
    """Stand-in for HarnessScreenRunner in locked_pair: scripted plan-B receipts for FR and the candidate."""

    def __init__(self, fr, cand):
        self.fr, self.cand, self.score_reference, self.reference, self.seen = fr, cand, None, None, []

    def run(self, task, config, overlay, purpose, run_id, role='candidate', recheck=False):
        self.seen.append((config.get('who'), purpose, self.score_reference is not None, role))
        return dict(self.fr if config.get('who') == 'FR' else self.cand)


def _rec(valid, compared=True, is_reference=False, t=1.0):
    return {'attempt_id': 'x', 'valid': valid, 'step_time': {'mean_window_max_rank_s': t, 'rsd_window': 0.004},
            'harness': {'numerics': {'compared': compared, 'is_reference': is_reference, 'class': 'E0'}, 'gpu_hours': 0.01}}


FR_REFS = {'fr_status': 'feasible', 'fr_screen': '/data/infrarsi/receipts/fr.json', 'fr_config': {'who': 'FR'}}


@pytest.mark.parametrize('tid, fr, cand, selected, expect', [
    ('D2', _rec(True), _rec(True, t=0.8), 'c3', {'invalid_accept': 0, 't_cand_locked': 0.8}),            # FR first, verified
    ('D2', _rec(True), _rec(False, compared=False), 'c3', {'invalid_accept': 1, 't_cand_locked': None}),  # promoted, failed
    ('D2', _rec(False), _rec(True, compared=False, is_reference=True), 'c3',
     {'invalid_accept': 0, 't_cand_locked': None, 't_fr_locked': None}),                                  # FR failed: unverified
    ('D2', _rec(True), _rec(True, compared=False, is_reference=True), 'c3', {'invalid_accept': 1}),       # never compared
    # The start candidate is promoted like any other when it is not the reference configuration (c0 = ED != FR):
    ('D2', _rec(True), _rec(True, t=0.8), 'c0', {'promoted': True, 'invalid_accept': 0, 't_cand_locked': 0.8}),
    ('D2', _rec(True), _rec(False, compared=False), 'c0', {'promoted': True, 'invalid_accept': 1, 't_cand_locked': None}),
])
def test_locked_pair_verification_rules(tmp_path, tid, fr, cand, selected, expect):
    from infrarsi.agent.judge import Verdict
    task = TaskSpec.load(tid)
    hr = _PairRunner(fr, cand)
    judge = lambda c, parent, ref: Verdict(valid=bool(c.get('valid')), reasons=[])            # noqa: E731
    res = arms.locked_pair(task, {'run_id': 'r'}, hr, FR_REFS, {'who': 'cand'}, None,
                           selected=selected, out_dir=tmp_path, judge=judge)
    assert res['order'] == ['FR', 'candidate'] and res['fr_first']                              # D2: even index
    assert hr.seen[0] == ('FR', 'score', False, 'reference')                                   # base run: the reference
    assert hr.seen[1] == ('cand', 'score', bool(fr['valid']), 'candidate')                     # FR plan B as reference
    for k, v in expect.items():
        assert res[k] == v, (k, res)


class _RolePairRunner:
    """Locked-pair stand-in keyed by the numerics role: the reference run and the candidate run."""

    def __init__(self, ref, cand):
        self.ref, self.cand, self.score_reference, self.reference, self.seen = ref, cand, None, None, []

    def run(self, task, config, overlay, purpose, run_id, role='candidate', recheck=False, wall_cap=None):
        self.seen.append((role, purpose, dict(overlay or {})))
        return dict(self.ref if role == 'reference' else self.cand)


def test_promotion_is_decided_by_the_effective_identity(tmp_path, pristine_overlays):
    """Codex R2: c0 or any other candidate is promoted iff its resolved configuration and effective overlay differ
    from the reference configuration without an overlay. Not promoted: only the reference half runs and the result
    falls back to it (no candidate time, no InvalidAccept, no promotion lifecycle)."""
    from infrarsi.actions import validate_l0
    from infrarsi.agent.judge import Verdict
    from agent_helpers import FR_DENSE, PRISTINE
    task = TaskSpec.load('D2')
    refs = {'fr_status': 'feasible', 'fr_screen': '/data/infrarsi/receipts/fr.json',
            'fr_config': dict(FR_DENSE, _ple_cpu_offload=True)}                     # planner-only key: never compared
    judge = lambda c, parent, ref: Verdict(valid=bool(c.get('valid')), reasons=[])    # noqa: E731
    name = sorted(PRISTINE)[0]
    n = iter(range(100))

    def pair(config, overlay, cand, selected='c0'):
        hr = _RolePairRunner(_rec(True), cand)
        out = tmp_path / f'p{next(n)}'
        res = arms.locked_pair(task, {'run_id': 'r'}, hr, refs, config, overlay, selected=selected, out_dir=out, judge=judge)
        return res, hr, out

    ed_like = dict(FR_DENSE, micro_batch_size=2)                        # c0 = the ED start, not FR's configuration
    res, hr, out = pair(ed_like, None, _rec(True, t=0.8))
    assert res['promoted'] and res['t_cand_locked'] == 0.8 and res['invalid_accept'] == 0
    assert [s[0] for s in hr.seen] == ['reference', 'candidate'] and (out / 'locked' / 'candidate_plan_b.json').exists()
    res, _, _ = pair(ed_like, None, _rec(False, compared=False))
    assert res['promoted'] and res['invalid_accept'] == 1 and res['t_cand_locked'] is None
    # FR's configuration itself in the search's c0 form (every key explicit), with no overlay or with one that is
    # byte-identical to the built-in hooks: nothing is promoted - even a run that would have failed is not run.
    same = validate_l0(FR_DENSE).resolved
    for overlay in (None, {name: PRISTINE[name]}):
        res, hr, out = pair(same, overlay, _rec(False, compared=False))
        assert res['promoted'] is False and res['invalid_accept'] == 0 and res['t_cand_locked'] is None, res
        assert [s[0] for s in hr.seen] == ['reference'] and 'skipped' in res['candidate'] and res['t_fr_locked'] == 1.0
        assert not (out / 'locked' / 'candidate_plan_b.json').exists()
    # A real overlay on FR's configuration is a promoted candidate.
    changed = {name: PRISTINE[name] + '\n# changed\n'}
    res, hr, _ = pair(same, changed, _rec(True, t=0.9), selected='c4')
    assert res['promoted'] and res['t_cand_locked'] == 0.9 and hr.seen[1][2] == changed
    # finish(): an unpromoted selection reports the fallback and runs no promotion lifecycle - FR's configuration
    # kept as c0, or c0 = ED kept only because the search accepted no candidate at all (finalize's fallback).
    for cfg, valid in ((same, True), (ed_like, False)):
        trace = {'budget': {'gpu_hours': 10.0, 'llm_tokens': 10, 'wall_seconds': 10_000, 'screens': 4}, 'selected_config': cfg,
                 'selected_overlay': {}, 'selected': 'c0', 'accounting_complete': True, 'candidates': [{'id': 'c0', 'valid': valid}],
                 'used': {'gpu_hours': 0.1, 'llm_tokens': 0, 'llm_usd': 0.0, 'screens': 1, 'wall_seconds': 10}}
        hr = _RolePairRunner(_rec(True), _rec(False, compared=False))
        hr.attempts = []
        r = arms.finish({}, task, {'run_id': 'r', 'arm': 'A0'}, hr, refs, trace, tmp_path / f'fin{valid}')
        si = r['scoring_inputs']
        assert si['promoted'] is False and si['t_cand_locked'] is None and si['invalid_accept'] == 0 and si['t_fr_locked'] == 1.0
        assert 'promotion_lifecycle' not in r and si['lifecycle_pass'] is None and [s[0] for s in hr.seen] == ['reference']
        assert r['locked_pair']['accepted'] is valid


def _keep_c0(role, system, prompt):
    return {'verdict': 'pass', 'reasons': []} if role == 'verifier' else {'action': 'finish', 'candidate_id': 'c0'}


def test_kept_start_candidate_is_promoted_through_e7(world, monkeypatch):
    """The agent keeps c0, the ED start (not FR's configuration): it is promoted - locked pair, promotion lifecycle
    on the start configuration (the raw ED screen bound by digest), and D's lifecycle_promoted arm runs it."""
    monkeypatch.setitem(arms.DRIVERS, 'strong', lambda: ScriptedBackend(_keep_c0))
    run = {'run_id': 'E1-S1-A0-keep-c0', 'task': 'S1', 'arm': 'A0', 'screens': 16, 'wall_hours_cap': 3,
           'llm_tokens_cap': 1500000, 'seed': 42, 'priority': 'P0'}
    res = _dispatch(world, run)['result']
    pair, life = res['locked_pair'], res['promotion_lifecycle']
    assert res['trace']['selected'] == 'c0' and pair['selected'] == 'c0' and res['trace']['screens'] == 0
    assert pair['promoted'] is True and pair['t_cand_locked'] and pair['invalid_accept'] == 0
    assert life['ran'] and life['lifecycle_pass'], life
    _check_scoring_inputs(res)
    e7 = {'run_id': 'E7-S1-LIFECYCLE-sim', 'task': 'S1', 'arm': 'lifecycle_promoted', 'depends_on': [run['run_id']],
          'wall_hours_cap': 1, 'priority': 'P0'}
    out = _dispatch(world, e7)
    assert out['promoted'] is True and out['status'] == 'pass' and not out['unbound'], out
    assert out['candidate_attempt'] == pair['candidate']['attempt_id']


def test_kept_fr_start_falls_back_to_the_reference(world, monkeypatch):
    """ablation_start_from_FR keeping c0 = FR's configuration: nothing promoted, only FR plan B runs, no candidate
    time or InvalidAccept, and D's lifecycle_promoted arm has nothing to run."""
    monkeypatch.setitem(arms.DRIVERS, 'strong', lambda: ScriptedBackend(_keep_c0))
    run = {'run_id': 'E3-S1-from-FR-keep-c0', 'task': 'S1', 'arm': 'ablation_start_from_FR', 'screens': 16,
           'wall_hours_cap': 3, 'llm_tokens_cap': 1500000, 'seed': 42, 'priority': 'P0'}
    res = _dispatch(world, run)['result']
    pair, si = res['locked_pair'], res['scoring_inputs']
    assert pair['promoted'] is False and pair['order'] == ['FR'] and 'skipped' in pair['candidate']
    assert si['t_cand_locked'] is None and si['invalid_accept'] == 0 and si['t_fr_locked'] and si['promoted'] is False
    assert 'promotion_lifecycle' not in res and si['t_ed'] is not None
    assert [(q['purpose'], q['numerics_role']) for q in _requests(world['remote'], run['run_id'])] == [('score', 'reference')]
    e7 = {'run_id': 'E7-S1-LIFECYCLE-none', 'task': 'S1', 'arm': 'lifecycle_promoted', 'depends_on': [run['run_id']],
          'wall_hours_cap': 1, 'priority': 'P0'}
    assert _dispatch(world, e7)['promoted'] is False and not _requests(world['remote'], e7['run_id'])


def test_receipt_trust_is_the_harness_hmac(tmp_path, monkeypatch):
    """Every receipt-trust check of the arms goes through D's HMAC (harness/signing.py): the old recomputable digest
    is no signature, and a receipt signed with another PC's key is not trusted."""
    from infrarsi.contracts import digest
    from infrarsi.harness import signing
    r = {'attempt_id': 'a1', 'task_id': 'S1', 'valid': True, 'harness': {'numerics': {'compared': True}}}
    old = json.loads(json.dumps(r))
    old['harness']['receipt_digest'] = digest({k: v for k, v in old.items() if k != 'harness'})
    signed = signing.sign(json.loads(json.dumps(r)))
    assert arms.receipt_trusted(signed) and not arms.receipt_trusted(r) and not arms.receipt_trusted(old)
    assert not arms.receipt_trusted(None) and not arms.receipt_trusted({'harness_signature': 'x'})
    task = TaskSpec.load('S1')
    assert 'harness signature does not verify' in arms.authenticate(old, task, purpose='score', plan='B')
    assert 'harness signature does not verify' not in arms.authenticate(signed, task, purpose='score', plan='B')
    assert any('signature' in p for p in arms.binding_problems(signed, old, purpose='lifecycle', plan='B', name='lc'))
    monkeypatch.setenv(signing.KEY_ENV, str(tmp_path / 'another_pc.key'))
    assert not arms.receipt_trusted(signed)


def test_ed_denominator_is_bound_to_the_recorded_ed_run(world):
    """Codex R2: only the recorded, feasible ED run's signed plan-B score is t_ED - not another valid plan-B receipt
    of the task, not a copy at another path, not a run with another code or effective overlay."""
    from infrarsi.harness import signing
    task, ctx, remote, runner = world['task'], world['ctx'], world['remote'], world['runner']
    refs = ctx.store.references()['S1']
    rec, problems = arms.load_ed_score(task, refs, ctx, ['WINH0'])
    assert problems == [] and rec['attempt_id'] == world['ed_b']['attempt_id'] and rec['numerics_role'] == 'candidate'
    assert arms.ed_identity_problems(rec, [world['fr']]) == []
    # The R2 probe: a signed, valid same-task plan-B candidate receipt with another configuration.
    fake = runner(RunRequest(run_id='E1-S1-other', purpose='score', task=task.raw, config=dict(world['ed_cfg'], micro_batch_size=4),
                             nodes=['WINH0'], gpus_per_node=1, input_plan='B', reference_receipt=world['fr_b']['harness']['stored']),
                  reference=world['fr_b'])
    assert fake['valid'] and signing.verify(fake)
    rec, problems = arms.load_ed_score(task, dict(refs, ed_score=fake['harness']['stored']), ctx, ['WINH0'])
    assert rec is None and any('ed_config' in p for p in problems) and any('ED screen: run_id' in p for p in problems)
    # The exact ED receipt stored at another path; ED not feasible; ED's role contradicting the FR status.
    copy = '/data/infrarsi/receipts/ed-copy.json'
    remote.write_json('WINH0', copy, world['ed_b'])
    assert any('recorded ED attempt' in p for p in arms.load_ed_score(task, dict(refs, ed_score=copy), ctx, ['WINH0'])[1])
    assert arms.load_ed_score(task, dict(refs, ed_status='infeasible'), ctx, ['WINH0'])[0] is None
    flipped = dict(refs, fr_status='infeasible', numerics_reference='ED')
    assert any('numerics_role' in p for p in arms.load_ed_score(task, flipped, ctx, ['WINH0'])[1])
    # An unsigned (altered) ED receipt.
    altered = json.loads(json.dumps(world['ed_b']))
    altered['step_time']['mean_window_max_rank_s'] *= 1.5
    assert 'harness signature does not verify' in arms.ed_score_problems(altered, task, refs, path=refs['ed_score'], screen=world['ed'])
    # ED's configuration run with an overlay or other code (signed, same config digest): not a no-overlay run of this
    # experiment, so no denominator.
    for k in ('overlay_digest', 'code_digest'):
        other = json.loads(json.dumps(world['ed_b']))
        other[k] = '0' * 64
        signing.sign(other)
        assert arms.ed_identity_problems(other, [world['fr']]) and arms.ed_identity_problems(other, [])


def test_ed_denominator_when_ed_is_the_reference(world):
    """FR infeasible: ED ran as the task's numerics reference, and its plan-B score is the denominator, end to end."""
    task, ctx, runner = world['task'], world['ctx'], world['runner']
    cfg = world['ed_cfg']
    kw = dict(run_id='E0-CAL-S1-ED', task=task.raw, config=cfg, nodes=['WINH0'], gpus_per_node=1, numerics_role='reference')
    screen = runner(RunRequest(purpose='screen', input_plan='A', **kw))
    score = runner(RunRequest(purpose='score', input_plan='B', **kw))
    assert screen['valid'] and score['valid']
    refs = {'fr_status': 'infeasible', 'numerics_reference': 'ED', 'ed_status': 'feasible', 'ed_config': cfg,
            'ed_screen': screen['harness']['stored'], 'ed_score': score['harness']['stored'], 'ed_numerics_role': 'reference'}
    rec, problems = arms.load_ed_score(task, refs, ctx, ['WINH0'])
    assert problems == [] and rec['attempt_id'] == score['attempt_id'] and rec['numerics_role'] == 'reference'
    # ED's candidate-role receipts (compared with FR) are not the denominator of this reference state.
    assert arms.load_ed_score(task, dict(refs, ed_screen=world['ed']['harness']['stored'],
                                          ed_score=world['ed_b']['harness']['stored']), ctx, ['WINH0'])[0] is None
    ctx.store.set_reference('S1', fr_status='infeasible', numerics_reference='ED', ed_screen=refs['ed_screen'],
                            ed_score=refs['ed_score'], ed_numerics_role='reference')
    run = {'run_id': 'E2-S1-random_L0-edref', 'task': 'S1', 'arm': 'random_L0', 'screens': 2, 'wall_hours_cap': 1,
           'llm_tokens_cap': 0, 'seed': 42, 'priority': 'P0'}
    res = _dispatch(world, run)['result']
    si = res['scoring_inputs']
    assert res['locked_pair']['reference_kind'] == 'ED' and si['t_ed'] == score['step_time']['mean_window_max_rank_s']
    assert si['ed_denominator'] == {'attempt_id': score['attempt_id'], 'path': refs['ed_score'], 'role': 'reference',
                                    'problems': []}


def test_tpe_l0_handler_end_to_end(world):
    pytest.importorskip('optuna')
    run = {'run_id': 'E2-S1-tpe_L0-sim', 'task': 'S1', 'arm': 'tpe_L0', 'screens': 3, 'wall_hours_cap': 1,
           'llm_tokens_cap': 1500000, 'seed': 42, 'priority': 'P0'}
    res = _dispatch(world, run)['result']
    assert res['trace']['screens'] == 3 and res['locked_pair']['invalid_accept'] == 0
    assert {q['run_id'] for q in _requests(world['remote'], run['run_id'])} == {run['run_id']}


def test_locked_pair_cost_enters_the_run_totals(tmp_path):
    from infrarsi.agent.judge import Verdict
    task = TaskSpec.load('D2')
    failed = {'valid': False, 'attempt_id': 'x', 'invalid_reasons': ['occupied: ...'],
              'cost': {'gpu_seconds': 0.3 * 3600, 'wall_seconds': 60, 'accounting_complete': False}}
    hr = _PairRunner(_rec(True), failed)
    pair = arms.locked_pair(task, {'run_id': 'r'}, hr, FR_REFS, {'who': 'cand'}, None, selected='c2',
                            out_dir=tmp_path, judge=lambda c, p, r: Verdict(valid=bool(c.get('valid')), reasons=[]))
    assert pair['accounting_complete'] is False and abs(pair['gpu_hours'] - 0.31) < 1e-9 and pair['invalid_accept'] == 1
    trace = {'budget': {'gpu_hours': 1.0, 'llm_tokens': 10, 'wall_seconds': 100, 'screens': 4}, 'selected_config': {},
             'selected': 'c2', 'used': {'gpu_hours': 0.9, 'llm_tokens': 0, 'llm_usd': 0.0, 'screens': 2, 'wall_seconds': 10},
             'accounting_complete': True, 'candidates': [{'id': 'c0', 'valid': True}, {'id': 'c2', 'valid': True}]}
    hr2 = _PairRunner(_rec(True), failed)
    hr2.attempts = []
    res = arms.finish({}, task, {'run_id': 'r', 'arm': 'A0'}, hr2, FR_REFS, trace, tmp_path)
    c, si = res['cost'], res['scoring_inputs']
    assert c['over_gpu_cap'] and abs(c['search_budget_gpu_hours'] - 1.21) < 1e-9 and c['accounting_complete'] is False
    assert res['over_budget'] == ['gpu_hours']
    # 1.21 GPUh on a 1.0 cap: the result is ineligible - no locked time, excluded from AUC/CTG, not an InvalidAccept.
    assert si['eligible'] is False and si['t_cand_locked'] is None and si['events'] is None and si['invalid_accept'] == 0
    # Over budget already after the search: the locked pair is not run at all.
    trace['used']['gpu_hours'] = 1.05
    hr3 = _PairRunner(_rec(True), _rec(True))
    hr3.attempts = []
    res = arms.finish({}, task, {'run_id': 'r', 'arm': 'A0'}, hr3, FR_REFS, trace, tmp_path)
    assert hr3.seen == [] and 'skipped' in res['locked_pair'] and res['scoring_inputs']['eligible'] is False


def test_candidates_never_run_without_a_reference():
    task = TaskSpec.load('S2')
    boom = _Boom(RuntimeError('must not be called'))
    hr = HarnessScreenRunner(boom, ['WINH1'])
    for purpose in ('screen', 'probe', 'score', 'lifecycle'):
        with pytest.raises(ValueError, match='without a numerics reference'):
            hr.run(task, {}, None, purpose, 'r')
    assert boom.seen == []
    hr.run(task, {}, None, 'score', 'r', recheck=True)                 # candidate-first locked pair, re-checked later
    hr.run(task, {}, None, 'score', 'r', role='reference')             # a base run of the pair
    roles = [(q.purpose, q.numerics_role, q.reference_receipt) for q, _ in boom.seen]
    assert roles == [('score', 'candidate', None), ('score', 'reference', None)]
    hr.score_reference = ('/data/infrarsi/receipts/frb.json', {'attempt_id': 'frb'})
    hr.run(task, {}, None, 'lifecycle', 'r')                           # promotion lifecycle: paired FR plan B
    q, kw = boom.seen[-1]
    assert q.numerics_role == 'candidate' and q.reference_receipt.endswith('frb.json') and kw['reference']['attempt_id'] == 'frb'


def test_failed_promotion_lifecycle_is_an_invalid_accept(world, monkeypatch):
    """Forward compare outside tolerance (flags still true): LifecyclePass false, no locked gain, InvalidAccept 1,
    and the lifecycle GPU time is still charged."""
    bad = dict(LIFECYCLE_OK, forward={'rel_l2': 0.2, 'top1_agreement': 0.6, 'forward_ok': True})
    monkeypatch.setitem(SIM, 'lifecycle', bad)
    replies = [{'action': 'propose_config', 'base': 'c0', 'changes': {'micro_batch_size': 4}},
               {'action': 'run_screen', 'candidate_id': 'c1'}, {'action': 'finish', 'candidate_id': 'c1'}]

    def script(role, system, prompt):
        if role == 'verifier':
            return {'verdict': 'pass', 'reasons': []}
        return replies.pop(0) if replies else {'action': 'finish'}
    monkeypatch.setitem(arms.DRIVERS, 'strong', lambda: ScriptedBackend(script))
    run = {'run_id': 'E1-S1-A0-badlife', 'task': 'S1', 'arm': 'A0', 'screens': 16, 'wall_hours_cap': 3, 'llm_tokens_cap': 1500000,
           'seed': 42, 'priority': 'P0'}
    res = _dispatch(world, run)['result']
    life, si = res['promotion_lifecycle'], res['scoring_inputs']
    assert life['ran'] and life['lifecycle_pass'] is False and any('rel_l2' in p for p in life['problems'])
    assert si['lifecycle_pass'] is False and si['t_cand_locked'] is None and si['invalid_accept'] == 1
    assert res['cost']['promotion_lifecycle_gpuh'] > 0


def test_lifecycle_rules_follow_amendment_001():
    """state_exact required; first resumed update CE <= 1e-4 and grad norm <= 5e-3; later grad norm report-only."""
    task = TaskSpec.load('S2')
    lc = {'lifecycle': dict(LIFECYCLE_OK)}
    assert arms.lifecycle_problems(lc, task) == []                       # 11.5% grad norm at resumed update 2 is fine
    bad = {'lifecycle': dict(LIFECYCLE_OK, state_exact=False)}
    assert any('state digest' in p for p in arms.lifecycle_problems(bad, task))
    first = dict(LIFECYCLE_OK['continuation'], steps=[{'step': 41, 'main_ce_rel': 1e-5, 'grad_norm_rel': 0.01}])
    assert any('first resumed update grad norm' in p for p in arms.lifecycle_problems({'lifecycle': dict(LIFECYCLE_OK, continuation=first)}, task))
    later = dict(LIFECYCLE_OK['continuation'], steps=[{'step': 41, 'main_ce_rel': 0.0, 'grad_norm_rel': 0.0},
                                                       {'step': 42, 'main_ce_rel': 0.009, 'grad_norm_rel': 0.0}])
    assert any('CE rel' in p for p in arms.lifecycle_problems({'lifecycle': dict(LIFECYCLE_OK, continuation=later)}, task))
