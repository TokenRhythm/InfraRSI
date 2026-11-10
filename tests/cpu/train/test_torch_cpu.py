"""torch / Megatron-Core parts that run without CUDA (on a WINH node with the frozen venv; skipped elsewhere).

  CUDA_VISIBLE_DEVICES= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $VENV/bin/python -m pytest tests/cpu/train -q
"""
import copy
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from infrarsi.runtime import gates, lifecycle  # noqa: E402

MODEL_1_7B = '/data/infrarsi/models/Qwen__Qwen3-1.7B'


def test_backends_flags_and_audit():
    flags = gates.backends_flags()
    assert {'float32_matmul_precision', 'cuda.matmul.allow_tf32', 'cudnn.allow_tf32'} <= set(flags)
    changed = dict(flags, **{'cuda.matmul.allow_tf32': not flags['cuda.matmul.allow_tf32']})
    assert gates.flags_problems(flags, flags, flags) == []
    assert any('allow_tf32' in p for p in gates.flags_problems(flags, flags, changed))


def test_dtype_census_on_a_distributed_optimizer_shape():
    model = torch.nn.Linear(8, 4).to(torch.bfloat16)
    for p in model.parameters():
        p.main_grad = torch.zeros(p.shape, dtype=torch.float32)
    masters = [p.detach().float().clone() for p in model.parameters()]
    inner = torch.optim.AdamW(masters, lr=1e-5)
    for m in masters:
        m.grad = torch.ones_like(m)
    inner.step()
    leaf = type('Leaf', (), {'shard_fp32_from_float16_groups': [masters], 'optimizer': inner})()
    chained = type('Chained', (), {'chained_optimizers': [leaf]})()
    census = gates.dtype_census([model], chained)
    assert gates.census_problems(census) == []
    assert census['counts']['exp_avg'] == {'torch.float32': 2}
    for st in inner.state.values():
        st['exp_avg_sq'] = st['exp_avg_sq'].bfloat16()
    assert any('exp_avg_sq' in p for p in gates.census_problems(gates.dtype_census([model], chained)))


def test_compare_logits():
    torch.manual_seed(0)
    hf = torch.randn(16, 100) * 5
    near = hf + 1e-3 * torch.randn_like(hf)
    ok = lifecycle.compare_logits(near, hf)
    assert ok['forward_ok'] and ok['top1_agreement'] == 1.0 and ok['max_abs_logit_diff'] < 0.01
    bad = lifecycle.compare_logits(torch.randn(16, 100) * 5, hf)
    assert not bad['forward_ok']
    padded = lifecycle.compare_logits(torch.cat([near, torch.zeros(16, 4)], -1), hf)
    assert padded['vocab_compared'] == 100 and padded['forward_ok']


def test_compare_continuation_amendment_001(tasks):
    ref = [{'step': s, 'main_ce': 2.0, 'grad_norm': 1.0, 'aux_loss': None, 'lr': 1e-5, 'supervised_tokens': 10,
            'clip_calls': 1} for s in range(31, 41)]
    tol = {'main_ce_rel_max_per_step': 0.005}
    ltol = lifecycle.lifecycle_tolerance(tasks['S2'].raw)
    assert ltol['first_resumed_update'] == {'main_ce_rel_max': 1e-4, 'grad_norm_rel_max': 5e-3}
    r = lifecycle.compare_continuation(ref, copy.deepcopy(ref), tol, ltol)
    assert r['match'] and r['bitwise']
    drift = copy.deepcopy(ref)
    drift[3]['main_ce'] = 2.002                  # later updates: CE within the task tolerance
    drift[4]['grad_norm'] = 1.5                  # later-update grad norm is report-only
    r = lifecycle.compare_continuation(ref, drift, tol, ltol)
    assert r['match'] and not r['bitwise'] and r['max_grad_norm_rel'] == pytest.approx(0.5)
    for key, value in (('main_ce', 2.02), ('lr', 2e-5), ('supervised_tokens', 11), ('clip_calls', 2)):
        bad = copy.deepcopy(ref)
        bad[5][key] = value                      # CE beyond tolerance, LR / tokens / clip calls not exact
        assert not lifecycle.compare_continuation(ref, bad, tol, ltol)['match'], key
    for key, value, ok in (('grad_norm', 1.004, True), ('grad_norm', 1.006, False), ('main_ce', 2.0001, True),
                           ('main_ce', 2.0003, False)):
        first = copy.deepcopy(ref)
        first[0][key] = value                    # first resumed update: CE 1e-4, grad norm 5e-3
        assert lifecycle.compare_continuation(ref, first, tol, ltol)['match'] is ok, (key, value)
    assert lifecycle.forward_tolerance(tasks['S2'].raw, None) == {'rel_l2': 0.05, 'top1_agreement': 0.90}
    assert lifecycle.forward_tolerance(tasks['S2'].raw, {'forward_tolerance': {'rel_l2': 0.1, 'top1_agreement': 0.8}}) \
        == {'rel_l2': 0.1, 'top1_agreement': 0.8}


def test_timers_are_bound_before_overlays(monkeypatch):
    import time
    from infrarsi.runtime import timers
    clock = timers.PhaseClock()
    monkeypatch.setattr(time, 'perf_counter', lambda: 0.0)       # what a hostile overlay would do
    with clock.phase('load'):
        time.sleep(0.02)
    assert clock.snapshot()['load'] >= 0.015


@pytest.fixture(scope='module')
def gloo_world():
    """Single-process gloo group + Megatron parallel state (TP=PP=CP=EP=1) without CUDA."""
    pytest.importorskip('megatron.core')
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29731')
    for k, v in (('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')):
        os.environ.setdefault(k, v)
    dist.init_process_group('gloo', rank=0, world_size=1)
    ps.initialize_model_parallel(1, 1, create_gloo_process_groups=False)
    yield
    ps.destroy_model_parallel()
    dist.destroy_process_group()


@pytest.mark.skipif(not Path(MODEL_1_7B, 'config.json').exists(), reason='Qwen3-1.7B not on this host')
def test_bridge_builds_qwen3_1_7b_on_meta_device(gloo_world, tasks):
    """Bridge ModelConfig + get_mcore_model on the meta device; global shapes must match the HF checkpoint."""
    from infrarsi.adapters.megatron_core import config_map, model_build as mb
    from infrarsi.contracts import RunRequest
    from train_helpers import fr_config
    facts = config_map.model_facts(MODEL_1_7B)
    req = RunRequest(run_id='dev-T-train-cpu', purpose='probe', task=copy.deepcopy(tasks['S1'].raw),
                     config=fr_config(tasks['S1'].raw, facts), nodes=['WINH1'], gpus_per_node=1)
    plan = config_map.make_plan(req, facts)
    cfg, derived = mb.model_config(plan, MODEL_1_7B, {'init_model_with_meta_device': True})
    assert cfg.calculate_per_token_loss and cfg.recompute_granularity == 'full' and cfg.params_dtype == torch.bfloat16
    assert cfg.num_layers == 28 and cfg.hidden_size == 2048 and cfg.qk_layernorm
    assert not cfg.untie_embeddings_and_output_weights          # Qwen3-1.7B ties embeddings
    from mcore_bridge import get_mcore_model
    if torch.cuda.is_available():
        pytest.skip('meta build check is for CUDA-less hosts; GPU runs build the real model')
    # TE refuses to construct modules without CUDA; on this CUDA-less host route every device request to meta.
    mp = pytest.MonkeyPatch()
    mp.setattr(torch.cuda, 'is_available', lambda: True)
    mp.setattr(torch.cuda, 'current_device', lambda: 'meta')
    mp.setattr(torch.cuda, 'get_device_capability', lambda *a, **k: (9, 0))
    try:
        models = get_mcore_model(cfg)
    finally:
        mp.undo()
    assert all(p.device.type == 'meta' for m in models for p in m.parameters())
    n_mcore = sum(p.numel() for m in models for p in m.parameters())
    hf = mb.hf_param_count(MODEL_1_7B, tied=True)
    assert n_mcore == hf['numel'], (n_mcore, hf)
    assert all(p.dtype == torch.bfloat16 for m in models for p in m.parameters())


FAMILIES = [('Qwen__Qwen3-30B-A3B', 'D4', ()), ('deepseek-ai__DeepSeek-V2-Lite', 'H1', ()),
            ('LLM-Research__Meta-Llama-3.1-8B', 'H2', ()), ('Qwen__Qwen3-Next-80B-A3B-Instruct', 'D6', ('mtp.',))]


@pytest.mark.parametrize('model,task_id,exclude', FAMILIES)
def test_bridge_meta_build_other_families(gloo_world, tasks, model, task_id, exclude):
    """qwen3_moe / deepseek_v2 / llama / qwen3_next construct through the bridge with the HF parameter count."""
    model_dir = f'/data/infrarsi/models/{model}'
    if not Path(model_dir, 'DOWNLOAD_DONE.json').exists():
        pytest.skip(f'{model} not downloaded on this host')
    if torch.cuda.is_available():
        pytest.skip('meta build check is for CUDA-less hosts')
    from infrarsi.adapters.megatron_core import config_map, model_build as mb
    from infrarsi.contracts import RunRequest
    from mcore_bridge import get_mcore_model
    from train_helpers import fr_config
    facts = config_map.model_facts(model_dir)
    config = fr_config(tasks[task_id].raw, facts)
    config['expert_model_parallel_size'] = 1 if facts['num_experts'] else config.get('expert_model_parallel_size', 1)
    req = RunRequest(run_id='dev-T-train-cpu', purpose='probe', task=copy.deepcopy(tasks[task_id].raw),
                     config=config, nodes=['WINH1'], gpus_per_node=1)
    plan = config_map.make_plan(req, facts)
    cfg, derived = mb.model_config(plan, model_dir, {'init_model_with_meta_device': True})
    if facts['num_experts']:
        assert cfg.moe_aux_loss_coeff == 0.001 and cfg.moe_router_load_balancing_type == 'aux_loss'
        assert cfg.moe_router_dtype == 'fp32' and cfg.moe_expert_capacity_factor is None
    mp = pytest.MonkeyPatch()
    mp.setattr(torch.cuda, 'is_available', lambda: True)
    mp.setattr(torch.cuda, 'current_device', lambda: 'meta')
    mp.setattr(torch.cuda, 'get_device_capability', lambda *a, **k: (9, 0))
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
    mp.setattr(MoEAlltoAllTokenDispatcher, 'cuda_dtoh_stream', object())   # side stream is created lazily on CUDA
    try:
        models = get_mcore_model(cfg)
    finally:
        mp.undo()
    n_mcore = sum(p.numel() for m in models for p in m.parameters())
    hf = mb.hf_param_count(model_dir, facts['tie_word_embeddings'], exclude)
    assert n_mcore == hf['numel'], (model, n_mcore, hf, derived)


def test_tensor_checksum_is_exact_and_order_sensitive():
    import hashlib
    t = torch.randn(1000, dtype=torch.float32)
    c = lifecycle.tensor_checksum(t)
    assert c == hashlib.sha256(t.numpy().tobytes()).hexdigest() and lifecycle.tensor_checksum(t.clone()) == c
    pair = torch.tensor([1.0, 1.0])                # bit changes +2 and -1 that a weighted sum would not see
    bits = pair.view(torch.int32).clone()
    bits[0] += 2
    bits[1] -= 1
    assert lifecycle.tensor_checksum(bits.view(torch.float32)) != lifecycle.tensor_checksum(pair)
    swapped = t.clone()
    swapped[[3, 7]] = swapped[[7, 3]]
    assert lifecycle.tensor_checksum(swapped) != c
    nudged = t.clone()
    nudged[5] = torch.nextafter(nudged[5], torch.tensor(float('inf')))
    assert lifecycle.tensor_checksum(nudged) != c
    b = t.bfloat16()
    assert lifecycle.tensor_checksum(b) == lifecycle.tensor_checksum(b.clone())
    big = torch.arange(3 * (1 << 20), dtype=torch.float32)
    old = lifecycle._CHUNK
    try:
        lifecycle._CHUNK = 1 << 20                 # chunked and unchunked hashing agree
        chunked = lifecycle.tensor_checksum(big)
    finally:
        lifecycle._CHUNK = old
    assert chunked == lifecycle.tensor_checksum(big)


def test_state_digest_covers_moments_and_steps():
    model = torch.nn.Linear(8, 4)
    masters = [p.detach().float().clone() for p in model.parameters()]
    inner = torch.optim.AdamW(masters, lr=1e-5)
    for m in masters:
        m.grad = torch.ones_like(m)
    inner.step()
    leaf = type('Leaf', (), {'shard_fp32_from_float16_groups': [masters], 'optimizer': inner})()
    d1 = lifecycle.state_digest([model], leaf)
    assert d1 == lifecycle.state_digest([model], leaf) and d1['steps']['n'] == 2 and d1['exp_avg']['n'] == 2
    st = next(iter(inner.state.values()))
    st['step'] += 1                                # wrong Adam step count after a restore
    assert lifecycle.state_digest([model], leaf)['steps'] != d1['steps']
    st['step'] -= 1
    st['exp_avg_sq'][0, 0] = torch.nextafter(st['exp_avg_sq'][0, 0], torch.tensor(1.0))   # one ulp
    assert lifecycle.state_digest([model], leaf)['exp_avg_sq'] != d1['exp_avg_sq']


def test_optimizer_step_sidecar_roundtrip():
    """A group that never stepped on this rank keeps no 'step'; restoring the sidecar reproduces the counters."""
    p1, p2 = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(2))
    inner = torch.optim.AdamW([{'params': [p1]}, {'params': []}, {'params': [p2]}], lr=1e-3)
    p1.grad, p2.grad = torch.ones(3), torch.ones(2)
    inner.step()
    inner.param_groups[0]['step'] = 5            # TE FusedAdam style per-group counter
    leaf = type('Leaf', (), {'optimizer': inner})()
    saved = lifecycle.optimizer_steps(leaf)
    assert [k for k, _ in saved] == ['g', 's', 'g', 'g', 's'] and saved[0][1] == 5.0 and saved[2][1] is None
    inner.param_groups[0].pop('step')            # what a rank-0-only metadata restore does to this rank
    next(iter(inner.state.values()))['step'].fill_(0.0)
    assert lifecycle.restore_optimizer_steps(leaf, saved) == 2
    assert lifecycle.optimizer_steps(leaf) == saved


def test_missing_torch_adam_step_is_restored():
    p = torch.nn.Parameter(torch.ones(3))
    inner = torch.optim.AdamW([p], lr=1e-3)
    p.grad = torch.ones(3)
    inner.step()
    leaf = type('Leaf', (), {'optimizer': inner})()
    saved = lifecycle.optimizer_steps(leaf)
    del inner.state[p]['step']                    # a load that keeps the moments but drops the counter
    assert lifecycle.restore_optimizer_steps(leaf, saved) == 1
    assert float(inner.state[p]['step']) == 1.0 and lifecycle.optimizer_steps(leaf) == saved


def test_supervisor_heartbeat_and_phases_total_span_the_launch(tmp_path, monkeypatch, make_request):
    """phases.total covers every phase process (harness external clock); the trainer's own total is kept."""
    import json
    import infrarsi.contracts as contracts
    from infrarsi.adapters.megatron_core.supervisor import Supervisor
    from infrarsi.runtime import receipt_writer as rw
    monkeypatch.setattr(contracts.RemotePaths, 'runs', str(tmp_path))
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('WORLD_SIZE', '1')
    req = make_request('S1', {}, gpus=1)
    path = tmp_path / 'request.json'
    path.write_text(json.dumps(req.to_dict()), encoding='utf-8')
    sup = Supervisor(str(path))
    hb = json.loads((Path(req.out_dir) / 'heartbeat.json').read_text(encoding='utf-8'))
    assert hb['phase'] == 'load' and hb['step'] == 0
    rw.write_json_atomic(Path(req.out_dir) / rw.PARTS_FILE, {'final': True, 'phases': {'train': 3.0, 'total': 5.0}})
    r = sup.receipt(False)
    assert r['phases']['trainer_total'] == 5.0 and r['phases']['train'] == 3.0
    assert 0.0 <= r['phases']['total'] < 60.0 and 'supervisor_total' not in r['phases'] and not r['valid']


def test_micro_batches_expose_nothing_of_the_trainer():
    """The schedule overlay gets data_iterator / forward_step_func; neither may lead back to the trainer."""
    import contextlib
    from types import SimpleNamespace
    pytest.importorskip('megatron.core')
    from infrarsi.adapters.megatron_core import trainer as tr
    from infrarsi.adapters.megatron_core.synthetic_data import SyntheticPackedTHDDataset
    from infrarsi.runtime import overlay_loader as l1

    class Timer:
        n = 0

        @contextlib.contextmanager
        def data(self):
            Timer.n += 1
            yield
    ds = SyntheticPackedTHDDataset(None, 'A', seq_len=64, packs_total=8, vocab_size=100, kind='pt_stream')
    w = np.random.default_rng(7).integers(1, 1 << 24, size=130, dtype=np.int64)
    stub = SimpleNamespace(timer=Timer(), source=SimpleNamespace(pack=ds.pack), plan=SimpleNamespace(cp=1), cp_rank=0,
                           overlays=l1.load(None), ctx=SimpleNamespace(device=torch.device('cpu')),
                           data_key=(w, torch.from_numpy(w)))
    consumed, sums, counts = [], [], {'forward': 0, 'backward': 0}
    it = tr.micro_batches(stub, ((0, 1), (2,)), consumed, sums, counts)
    assert not hasattr(it, '__dict__') and not hasattr(it, 't')
    assert {a for a in dir(it) if not a.startswith('__')} <= {'gi_code', 'gi_frame', 'gi_running', 'gi_yieldfrom',
                                                              'gi_suspended', 'close', 'send', 'throw'}
    b = next(it)
    assert consumed == [(0, 1)] and Timer.n == 1 and tuple(b['tokens'].shape) == (1, 128) and b['_counts'] is counts
    assert b['packed_seq_params'].qkv_format == 'thd' and len(list(it)) == 1 and consumed == [(0, 1), (2,)]
    assert not hasattr(tr._forward_step, '__self__')
    tr.Trainer.verify_batches(None, sums)            # the default copy preserves every value
    bad = l1.load(None)                     # a to_device that truncates the micro-batch
    bad.modules = dict(bad.modules, data_path=SimpleNamespace(
        to_device=lambda ctx, arrays: {k: torch.from_numpy(v)[..., :4] for k, v in arrays.items()}))
    stub.overlays = bad
    with pytest.raises(l1.OverlayViolation, match='to_device'):
        next(tr.micro_batches(stub, ((0,),), [], [], counts))

    def zero_tokens(ctx, arrays):                    # same layout, different values
        out = {k: torch.from_numpy(v) for k, v in arrays.items()}
        out['tokens'] = torch.zeros_like(out['tokens'])
        return out
    bad.modules = dict(bad.modules, data_path=SimpleNamespace(to_device=zero_tokens))
    sums = []
    next(tr.micro_batches(stub, ((0,),), [], sums, counts))
    with pytest.raises(l1.OverlayViolation, match='values'):
        tr.Trainer.verify_batches(None, sums)


def test_module_pins_catch_swapped_submodules():
    from infrarsi.runtime import gates
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU()))
    check = gates.make_module_pins([model])
    assert check() == []
    model[1][0].register_forward_hook(lambda m, i, o: o)        # hooks are not structure
    assert check() == []
    original = model[1][0]
    model[1][0] = torch.nn.Identity()
    assert check() == ['models[0].1.0']
    model[1][0] = original
    model[1].__dict__['_modules'] = dict(model[1]._modules)     # a fresh children dict
    assert check() == ['models[0].1._modules']


def test_module_pins_cover_the_whole_tree():
    from infrarsi.runtime import gates
    deep = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Sequential(torch.nn.Sequential(
        torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))))))
    check = gates.make_module_pins([deep])
    holder = deep[0][0][0][0]
    original = holder[0]
    holder[0] = holder[1]                          # a same-shaped sibling at depth 5
    assert check() == ['models[0].0.0.0.0.0']
    holder[0] = original
    assert check() == []


def test_step_counter_problems_require_every_counter():
    p1, p2 = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(2))
    inner = torch.optim.AdamW([{'params': [p1]}, {'params': [p2]}], lr=1e-3)
    p1.grad, p2.grad = torch.ones(3), torch.ones(2)
    inner.step()
    leaf = type('Leaf', (), {'optimizer': inner})()
    assert lifecycle.step_counter_problems(leaf, 1) == []
    assert lifecycle.step_counter_problems(leaf, 2) != []          # counters behind the applied updates
    del inner.state[p2]['step']                                    # a deleted counter is not "absent and fine"
    assert lifecycle.step_counter_problems(leaf, 1) == ['leaf0.opt0.group1: step None != 1']

    class FusedAdam(torch.optim.AdamW):            # TE keeps one counter per non-empty param group
        pass
    q = torch.nn.Parameter(torch.ones(2))
    te = FusedAdam([{'params': [q]}, {'params': []}], lr=1e-3)
    te.param_groups[0]['step'] = 3
    leaf = type('Leaf', (), {'optimizer': te})()
    assert lifecycle.step_counter_problems(leaf, 3) == []           # the empty group needs no counter
    del te.param_groups[0]['step']
    assert lifecycle.step_counter_problems(leaf, 3) == ['leaf0.opt0.group0: step None != 3']


def test_module_pins_freeze_child_order():
    from infrarsi.runtime import gates
    layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(3)])
    model = torch.nn.Sequential(layers)
    check = gates.make_module_pins([model])
    first = layers._modules.pop('0')              # same dict, same names, same children, new order
    layers._modules['0'] = first
    assert check() == ['models[0].0 children']
    for k in ('0', '1', '2'):                    # back to the original order
        layers._modules[k] = layers._modules.pop(k)
    assert list(layers._modules) == ['0', '1', '2'] and check() == []

    class Name(str):                              # a look-alike key that would answer any comparison
        def __eq__(self, other):
            return True

        def __hash__(self):
            return str.__hash__(self)
    child = layers._modules.pop('2')
    layers._modules[Name('2')] = child
    assert check() == ['models[0].0 children']


def test_forward_step_counts_forward_and_backward():
    pytest.importorskip('megatron.core')
    from infrarsi.adapters.megatron_core import trainer as tr
    w = torch.nn.Parameter(torch.ones(4))
    counts = {'forward': 0, 'backward': 0}

    def model(input_ids, position_ids, attention_mask, labels, packed_seq_params):
        return (input_ids.float() * w).sum(-1)
    batches = iter([{'tokens': torch.ones(1, 4), 'position_ids': None, 'labels': None, 'packed_seq_params': None,
                     'loss_mask': torch.ones(1), '_counts': counts} for _ in range(3)])
    for i in range(3):
        out, loss_fn = tr._forward_step(batches, model)
        assert counts == {'forward': i + 1, 'backward': i}
        loss, n, _ = loss_fn(out)
        loss.backward()
    assert counts == {'forward': 3, 'backward': 3}
    out, _ = tr._forward_step(iter([{'tokens': torch.ones(1, 4), 'position_ids': None, 'labels': None,
                                     'packed_seq_params': None, 'loss_mask': torch.ones(1), '_counts': counts}]), model)
    assert counts == {'forward': 4, 'backward': 3}           # a forward without its backward is visible


def test_supervisor_echoes_request_identity_and_pins_the_overlay_set(tmp_path, monkeypatch, make_request):
    import json
    import infrarsi.contracts as contracts
    from infrarsi.adapters.megatron_core import supervisor as sv
    from infrarsi.runtime import overlay_loader as l1
    monkeypatch.setattr(contracts.RemotePaths, 'runs', str(tmp_path))
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('WORLD_SIZE', '1')
    req = make_request('S1', {}, gpus=1, numerics_role='reference', data_manifest='/x/manifest.json',
                       effective_overlay_digest='0' * 64)
    path = tmp_path / 'request.json'
    path.write_text(json.dumps(req.to_dict()), encoding='utf-8')
    sup = sv.Supervisor(str(path))
    ident = sup.request_identity()
    assert ident == {'request_digest': contracts.digest(json.loads(path.read_text(encoding='utf-8'))),
                     'numerics_role': 'reference', 'data_manifest': '/x/manifest.json',
                     'effective_overlay_digest': '0' * 64}
    with pytest.raises(sv.OverlayDigestMismatch):
        sup.preflight()                               # before any model or data work
    r = sup.receipt(False)
    assert r['request_digest'] == ident['request_digest'] and r['effective_overlay_digest'] == '0' * 64
    req2 = make_request('S1', {}, gpus=1, effective_overlay_digest=l1.effective_digest(None))
    assert l1.effective_digest(req2.overlay) == req2.effective_overlay_digest
