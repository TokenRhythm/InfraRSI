import json
import math

import pytest

from infrarsi.agent import Budget, FakeScreenRunner, VirtualClock, optimize
from infrarsi.agent.backends import ScriptedBackend
from infrarsi.agent.judge import ReceiptJudge
from infrarsi.rsi import OptimizerPackage
from infrarsi.rsi.heldout import HeldoutSession, ReadWhitelist
from agent_helpers import DENSE_8B, FR_DENSE, MOE_30B

PASS = {'verdict': 'pass', 'reasons': []}
FR_TP2 = dict(FR_DENSE, tensor_model_parallel_size=2, sequence_parallel=True)


def explorer(n_changes=40):
    """Proposes a fresh config each turn (micro-batch/bucket/env sweeps) and screens it."""
    state = {'i': 0, 'last': None}
    values = [{'bucket_size': b, 'env': {'NCCL_BUFFSIZE': nb}} for b in (None, 10000000, 20000000, 40000000, 80000000, 160000000)
              for nb in (4194304, 8388608, 16777216)] + [{'overlap_grad_reduce': True, 'bucket_size': b} for b in (None, 20000000, 40000000)]

    def respond(role, system, prompt):
        if role == 'verifier':
            return PASS
        if state['last'] is None:
            change = values[state['i'] % len(values)]
            state['i'] += 1
            state['last'] = 'proposed'
            return {'action': 'propose_config', 'base': 'c0', 'changes': change}
        state['last'] = None
        cid = 'c' + str(state['i'])
        return {'action': 'run_screen', 'candidate_id': cid}
    return respond


def run(task, pkg, respond, budget=None, *, clock=None, mf=DENSE_8B, start=FR_DENSE, runner=None, **kw):
    clock = clock or VirtualClock()
    runner = runner or FakeScreenRunner(task, mf, clock=clock)
    drv = ScriptedBackend(respond, clock=clock, **kw.pop('drv', {}))
    kw.setdefault('judge', ReceiptJudge())
    tr = optimize(task, pkg, drv, budget or Budget.from_task(task), start, runner, model_facts=mf, clock=clock, **kw)
    return tr.to_dict(), drv, runner


def test_screen_budget_stops_the_search(tasks, a0_package):
    tr, drv, runner = run(tasks['D2'], a0_package, explorer())
    assert tr['stop_reason'] == 'budget:screens'
    assert tr['used']['screens'] == 16 == len(tr['screens']) == len(runner.calls)
    assert tr['selected_step_time'] <= tr['start_step_time']
    assert all('cum' in s and 'fractions' in s for s in tr['screens'])


def test_token_budget_counts_every_role(tasks, a0_package):
    b = Budget.from_task(tasks['D2'], llm_tokens=20_000)
    tr, drv, _ = run(tasks['D2'], a0_package, explorer(), b, drv={'tokens_per_call': 3000})
    assert tr['stop_reason'] == 'budget:llm_tokens' and 15_000 <= tr['used']['llm_tokens'] <= 20_000 + 3000
    roles = {c['role'] for c in tr['llm_calls']}
    assert 'verifier' in roles and 'proposer' in roles


def test_wall_clock_budget(tasks, a0_package):
    b = Budget.from_task(tasks['D2'], wall_seconds=3600)
    tr, *_ = run(tasks['D2'], a0_package, explorer(), b)
    assert tr['stop_reason'] == 'budget:wall_seconds' and tr['used']['screens'] < 16


def test_gpu_hours_budget_keeps_the_scoring_reserve(tasks, a0_package):
    b = Budget.from_task(tasks['D2'], gpu_hours=6.0)
    tr, *_ = run(tasks['D2'], a0_package, explorer(), b)
    assert tr['stop_reason'] == 'budget:gpu_hours'
    mean = tr['used']['gpu_hours'] / tr['used']['screens']
    assert tr['used']['gpu_hours'] + b.scoring_reserve_screens * mean <= b.gpu_hours + mean + 1e-9


def test_failed_screens_count_and_same_failure_stop(tasks, a0_package):
    clock = VirtualClock()
    runner = FakeScreenRunner(tasks['D2'], DENSE_8B, clock=clock, fail=lambda cfg: 'NCCL watchdog timeout')
    tr, *_ = run(tasks['D2'], a0_package, explorer(), clock=clock, runner=runner)
    assert tr['stop_reason'] == 'same_failure_x6' and tr['used']['screens'] == 6 and tr['used']['gpu_hours'] > 0
    assert not any(s['valid'] for s in tr['screens']) and tr['selected'] == 'c0'


def test_invalid_proposals_cost_no_screen(tasks, a0_package):
    script = [{'action': 'propose_config', 'base': 'c0', 'changes': {'sequence_parallel': True}},
              {'action': 'propose_config', 'base': 'c0', 'changes': {'global_batch_size': 1}},
              {'action': 'propose_config', 'base': 'c9', 'changes': {}},
              {'action': 'propose_patch', 'base': 'c0', 'files': {'runtime/timers.py': 'x = 1\n'}},
              'not json',
              {'action': 'finish', 'candidate_id': 'c0'}]
    tr, drv, runner = run(tasks['D2'], a0_package, lambda r, s, p: PASS if r == 'verifier' else script.pop(0))
    assert tr['used']['screens'] == 1 and len(runner.calls) == 1                 # only the start config
    assert tr['invalid_proposals'] == {'l0_invalid': 2, 'unknown_base': 1, 'patch_rejected': 1}
    assert tr['stop_reason'] == 'finish'


def test_verifier_veto_prevents_the_screen_and_no_verifier_ablation(tasks, a0_package):
    script = [{'action': 'propose_config', 'base': 'c0', 'changes': {'micro_batch_size': 2}},
              {'action': 'run_screen', 'candidate_id': 'c1'}, {'action': 'finish'}]
    s1 = list(script)
    tr, *_ = run(tasks['D2'], a0_package, lambda r, s, p: {'verdict': 'veto', 'reasons': ['no gain']} if r == 'verifier' else s1.pop(0))
    assert tr['verifier'] == {'calls': 1, 'vetoes': 1, 'errors': 0} and tr['used']['screens'] == 1
    s2 = list(script)
    tr, *_ = run(tasks['D2'], a0_package, lambda r, s, p: s2.pop(0), use_verifier=False)
    assert tr['verifier']['calls'] == 0 and tr['used']['screens'] == 2


def test_finish_with_invalid_choice_falls_back_to_best_valid(tasks, a0_package):
    script = [{'action': 'propose_config', 'base': 'c0', 'changes': {'recompute_granularity': None, 'recompute_method': None,
                                                                     'recompute_num_layers': None}},
              {'action': 'run_screen', 'candidate_id': 'c1'},
              {'action': 'propose_config', 'base': 'c0', 'changes': {'micro_batch_size': 1, 'tensor_model_parallel_size': 1,
                                                                     'context_parallel_size': 1, 'bucket_size': 10000000}},
              {'action': 'finish', 'candidate_id': 'c2'}]                 # c2 never screened
    tr, *_ = run(tasks['D2'], a0_package, lambda r, s, p: PASS if r == 'verifier' else script.pop(0), start=FR_TP2)
    assert tr['selected'] == 'c1' and tr['selected_step_time'] < tr['start_step_time']


def test_start_receipt_is_not_charged(tasks, a0_package):
    clock = VirtualClock()
    runner = FakeScreenRunner(tasks['D2'], DENSE_8B, clock=clock)
    start_receipt = runner.run(tasks['D2'], FR_DENSE, None, 'screen', 'E1-D2-ED')
    runner.calls.clear()
    t0 = clock()
    tr, *_ = run(tasks['D2'], a0_package, lambda r, s, p: {'action': 'finish'}, clock=clock, runner=runner, start_receipt=start_receipt)
    assert tr['used']['screens'] == 0 and tr['start_step_time'] and not runner.calls and clock() == t0 + 1.0


def test_memory_writes_never_mutate_the_package(tasks, a0_package, tmp_path):
    script = [{'action': 'write_memory', 'entry': {'lesson': 'x', 'outcome': 'y'}}, {'action': 'finish'}]
    tr, *_ = run(tasks['D2'], a0_package, lambda r, s, p: script.pop(0), out_dir=tmp_path / 'run')
    assert tr['memory_writes'][0]['noop'] is False and tr['memory_writes'][0]['entry']['tags']['model']
    OptimizerPackage.load(a0_package.root)                # digest still verifies
    assert (tmp_path / 'run' / 'memory_working.db').exists()


def test_heldout_mode_is_read_only_and_frozen(tasks, a0_package, tmp_path):
    hs = HeldoutSession('H3', a0_package.digest, tmp_path / 'heldout.jsonl')
    script = [{'action': 'write_memory', 'entry': {'lesson': 'x'}}, {'action': 'finish'}]
    tr, *_ = run(tasks['H3'], a0_package, lambda r, s, p: script.pop(0), heldout=hs, start=FR_TP2)
    assert tr['memory_writes'] == [{'turn': 1, 'noop': True, 'reason': 'held-out (read-only)'}] and tr['extra']['heldout']
    log = [json.loads(x) for x in (tmp_path / 'heldout.jsonl').read_text().splitlines()]
    assert [x['event'] for x in log] == ['heldout_start', 'write_memory_noop']
    with pytest.raises(PermissionError):
        run(tasks['D2'], a0_package, lambda r, s, p: {'action': 'finish'}, heldout=hs)
    with pytest.raises(PermissionError):
        run(tasks['H3'], a0_package, lambda r, s, p: {'action': 'finish'}, heldout=HeldoutSession('H3', '0' * 64, tmp_path / 'h2.jsonl'),
            start=FR_TP2)
    with pytest.raises(PermissionError):
        run(tasks['H1'], a0_package, lambda r, s, p: {'action': 'finish'}, whitelist=ReadWhitelist.dev(), mf=MOE_30B)


def test_tools_and_skills_are_logged(tasks, a0_package):
    script = [{'action': 'read_receipt', 'candidate_id': 'c0'}, {'action': 'retrieve_skill', 'query': 'moe all-to-all dispatch'},
              {'action': 'profile_ranges', 'candidate_id': 'c0'}, {'action': 'finish'}]
    tr, drv, _ = run(tasks['D2'], a0_package, lambda r, s, p: script.pop(0))
    assert tr['tool_calls'] and not tr['tool_calls'][0]['errors']
    assert {h['source'] for h in tr['skill_hits']} == {'auto', 'retrieve_skill'}
    assert any(h['skill'] == 'moe-ep-communication' for h in tr['skill_hits'] if h['source'] == 'retrieve_skill')
    assert '# SKILLS' in drv.calls[-1]['prompt'] and 'analyses' in json.dumps([c['prompt'] for c in drv.calls])


def test_weak_mode_protocol(tasks, a0_package, pristine_overlays):
    """config first (L1 locked), templates before free patches, <=80 lines, retries, explicit menu."""
    script = [
        {'action': 'apply_patch_template', 'base': 'c0', 'template': 'host_sync.manual_gc', 'params': {}},     # locked
        'garbage', 'still garbage',                                                                          # 2 retries
        {'action': 'propose_config', 'base': 'c0', 'changes': {'micro_batch_size': 1}},                     # duplicate of c0
        {'action': 'inspect', 'what': 'receipt', 'candidate_id': 'c0'},
    ]
    for b in (10000000, 20000000, 40000000, 80000000, 160000000, None):
        script += [{'action': 'propose_config', 'base': 'c0', 'changes': {'overlap_grad_reduce': True, 'bucket_size': b}},
                   {'action': 'run_trial', 'candidate_id': f'c{len([x for x in script if x is not None and isinstance(x, dict) and x.get("action") == "propose_config"])}'}]
    script += [{'action': 'propose_patch', 'base': 'c0', 'file': 'host_sync.py', 'content': 'def on_train_start(ctx):\n    return None\n'},
               {'action': 'apply_patch_template', 'base': 'c0', 'template': 'data_path.pinned_nonblocking', 'params': {}},
               {'action': 'propose_patch', 'base': 'c0', 'file': 'data_path.py',
                'content': 'def prefetch(ctx, step, microbatches):\n    return None\n' + '\n'.join(f'Y{i} = {i}' for i in range(90))},
               {'action': 'finish'}]
    tr, drv, _ = run(tasks['D2'], a0_package, lambda r, s, p: PASS if r == 'verifier' else script.pop(0), mode='weak')
    inv = tr['invalid_proposals']
    assert inv.get('l1_locked') == 1 and inv.get('template_first') == 1 and inv.get('patch_rejected') == 1
    retries = [c for c in tr['llm_calls'] if c['role'] == 'proposer_retry']
    assert len(retries) == 2
    assert tr['extra']['policy']['max_context_tokens'] == 32000 and tr['extra']['policy']['patch_line_limit'] == 80
    assert all(len(c['prompt']) / 3.5 <= 32000 for c in drv.calls)
    assert 'Menu (pick one)' in drv.calls[0]['system']
    kinds = [c['kind'] for c in tr['candidates']]
    assert kinds.count('L1') == 1                                  # only the template candidate


def test_judge_inter_step_rule():
    from infrarsi.agent.runner import synthetic_receipt
    from infrarsi.contracts import TaskSpec
    t = TaskSpec.load('S2').raw
    parent = synthetic_receipt(t, {}, step_time_s=2.0, peak_reserved_gib=50, run_id='r', attempt_id='p')
    good = synthetic_receipt(t, {}, step_time_s=1.8, peak_reserved_gib=50, run_id='r', attempt_id='g')
    bad = synthetic_receipt(t, {}, step_time_s=1.6, peak_reserved_gib=50, run_id='r', attempt_id='b', inter_step_ratio=1.25)
    j = ReceiptJudge()
    v = j(good, parent)
    assert v.valid and v.reward > 0.09
    v = j(bad, parent)
    assert not v.valid and any('InterStep' in r for r in v.reasons)


def _signed(receipt, *, numerics_class='E1', gpu_hours=1.5):
    """A receipt as the harness validates it: harness block with numerics and GPU-hours, HMAC-signed with the test key
    (harness/signing.py)."""
    from infrarsi.harness import signing
    r = json.loads(json.dumps(receipt))
    r.pop('cost', None)
    r['harness'] = {'numerics': {'compared': True, 'class': numerics_class, 'allowed_ok': True, 'violations': [],
                                 'main_ce_rel_max': 0.001, 'main_ce_rel_mean_window': 0.0004,         # AMENDMENT-001 fields
                                 'grad_norm': {'first_update_rel': 0.002, 'log_bias_window': 0.01, 'n_window': 14,
                                               'rel_max_all_steps': 0.15, 'per_step_rel': []},
                                 'self_check': {'repeated_pairs': []}},
                    'gpu_hours': gpu_hours, 'wall_s': 900.0, 'failure_kind': 'ok'}
    return signing.sign(r)


def test_reward_engine_judge_adapter_and_harness_cost(tasks, a0_package):
    a0 = a0_package
    pytest.importorskip('infrarsi.reward')
    from infrarsi.agent.judge import RewardEngineJudge, default_judge, gpu_seconds
    from infrarsi.agent.runner import synthetic_receipt
    t = tasks['S2'].raw
    parent = _signed(synthetic_receipt(t, {}, step_time_s=2.0, peak_reserved_gib=50, run_id='r', attempt_id='p'))
    child = _signed(synthetic_receipt(t, {}, step_time_s=1.8, peak_reserved_gib=50, run_id='r', attempt_id='c'))
    j = default_judge()
    assert isinstance(j, RewardEngineJudge)
    v = j(child, parent)
    assert v.valid and v.source == 'reward_engine' and abs(v.reward - math.log(2.0 / 1.8)) < 0.01 and v.numerics_class == 'E1', v.reasons   # speed only
    assert v.checks and not v.feedback
    tampered = json.loads(json.dumps(child))
    tampered['step_time']['mean_window_max_rank_s'] = 1.0
    v = j(tampered, parent)
    assert not v.valid and v.reward is None and any('receipt_signature' in r for r in v.reasons)
    assert gpu_seconds(child) == 1.5 * 3600
    clock = VirtualClock()

    class SignedRunner(FakeScreenRunner):
        def run(self, *a):
            return _signed(super().run(*a), gpu_hours=0.25)

    runner = SignedRunner(tasks['S2'], DENSE_8B | {'params': 4e9, 'hidden_size': 2560}, clock=clock)
    tr = optimize(tasks['S2'], a0, ScriptedBackend(lambda r, s, p: {'action': 'finish'}, clock=clock),
                  Budget.from_task(tasks['S2']), FR_DENSE, runner, model_facts=DENSE_8B, clock=clock).to_dict()
    assert tr['judge'] == 'RewardEngineJudge' and tr['used']['gpu_hours'] == 0.25 and tr['screens'][0]['valid']
