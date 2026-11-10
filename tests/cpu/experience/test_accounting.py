"""The closed-form accounting (textbook design P0-1, 2026-10-24): per-GPU memory for each framework's sharding and the
ideal update time -> MFU; and its calibration against the H20 / H200 harvests (nvidia-smi peaks of 323 shaped runs,
28 OOMs), which the constants were fitted on: P90 error <= 25 %, >= 22 of 28 OOMs caught, <= 10 % false OOMs."""
import json
from pathlib import Path

import pytest

from expjob import KNOBS, REF as MCORE_REF, megatron_log, project
from infrarsi.experience.accounting import estimate, mfu, split
from infrarsi.experience.runs import accounting_for
from infrarsi.experience.store import Store, fingerprint, plan

SEED = Path(__file__).resolve().parents[3] / 'infrarsi' / 'experience' / 'seed'
LLAMA8B = {'params_b': 8.03, 'layers': 32, 'hidden': 4096, 'ffn': 14336, 'heads': 32, 'kv_heads': 8, 'vocab': 128256}
MEGATRON = {'gpu': 'H200-141GB', 'memory_gib': 140.4, 'framework': 'megatron', 'gpus_per_node': 8, 'nodes': 1,
            'model': LLAMA8B, 'workload': {'seq_len': 4096, 'global_batch_tokens': 262144}}
REF = {'tp': '1', 'pp': '1', 'micro_batch': '1', 'precision': 'bf16', 'distributed_optimizer': 'on',
       'bf16_main_grads': 'fp32', 'recompute': 'none'}


def test_needs_the_shape():
    assert estimate({**MEGATRON, 'model': {'params_b': 8.0}}, REF) is None
    assert estimate({**MEGATRON, 'workload': {}}, REF) is None


def test_megatron_llama8b_on_h200_matches_the_h1_job():
    est = estimate(MEGATRON, REF)                              # H1 / H2b: 3.2865 s per update, 75 GiB peak
    assert 60 < est['mem_gib'] < 100 and est['fit_margin_gib'] > 30
    assert est['tflops_basis'] == 'bf16 measured' and 0.5 < mfu(est, 3.2865) < 0.9
    assert mfu(None, 3.0) is None and mfu(est, None) is None


def test_memory_moves_the_right_way():
    base = estimate(MEGATRON, REF)['mem_gib']
    assert estimate(MEGATRON, {**REF, 'tp': '2', 'sequence_parallel': 'on'})['mem_gib'] < base
    assert estimate(MEGATRON, {**REF, 'micro_batch': '2'})['mem_gib'] > base
    assert estimate(MEGATRON, {**REF, 'recompute': 'full'})['mem_terms']['activations_gib'] < \
        estimate(MEGATRON, REF)['mem_terms']['activations_gib']
    assert estimate(MEGATRON, {**REF, 'bf16_main_grads': 'bf16'})['mem_gib'] < base
    ds = {**MEGATRON, 'framework': 'deepspeed', 'gpu': 'H20-96GB', 'memory_gib': 95.1}
    z = [estimate(ds, {'zero_stage': str(k), 'micro_batch': '1'})['mem_terms']['states_gib'] for k in range(4)]
    assert z[0] > z[1] > z[2] > z[3]
    fsdp = {**ds, 'framework': 'fsdp'}
    assert estimate(fsdp, {'cpu_offload': 'all'})['mem_terms']['states_gib'] < estimate(fsdp, {})['mem_terms']['states_gib']
    lora = {**fsdp, 'workload': {**fsdp['workload'], 'x-finetune': 'lora-r16'}}
    assert estimate(lora, {})['mem_terms']['states_gib'] < estimate(fsdp, {})['mem_terms']['states_gib']


def test_moe_without_an_expert_count_takes_what_the_dense_shape_does_not_explain():
    qwen30 = {'params_b': 30.53, 'layers': 48, 'hidden': 2048, 'ffn': 6144, 'heads': 32, 'kv_heads': 4, 'vocab': 151936}
    dense, expert, active = split(qwen30, ep=8)
    assert expert > 25e9 and active is None
    est = estimate({**MEGATRON, 'gpu': 'H20-96GB', 'memory_gib': 95.1, 'model': qwen30}, {**REF, 'ep': '8'})
    assert est['flops_per_update'] is None and est['ideal_step_s'] is None and any('MoE' in n for n in est['notes'])


def _rows():
    for name in ('h20-harvest-2026-10.jsonl', 'h200-harvest-2026-10.jsonl'):
        envs = {}
        for line in (SEED / name).read_text(encoding='utf-8').splitlines():
            d = json.loads(line)
            if d.get('kind') == 'env':
                envs[d['id']] = d
            elif d.get('kind') == 'observation' and d.get('status') in ('ok', 'oom'):
                e = envs[d['env']]
                if (e.get('model') or {}).get('layers') and (d['status'] == 'oom' or d.get('mem_peak_gib')):
                    yield e, d


def test_calibration_on_the_harvests_holds():
    cap = {'H20-96GB': 95.1, 'H200-141GB': 140.4}
    err, caught, ooms, false_oom, oks = [], 0, 0, 0, 0
    for e, d in _rows():
        fp = {'gpu': e['gpu'], 'memory_gib': cap[e['gpu']], 'framework': e['framework'], 'gpus_per_node': e.get('gpus_per_node'),
              'nodes': e.get('nodes'), 'model': e['model'], 'workload': e['workload']}
        est = estimate(fp, d['config'])
        pred = est['mem_gib'] > cap[e['gpu']]
        if d['status'] == 'oom':
            ooms, caught = ooms + 1, caught + pred
        else:
            oks, false_oom = oks + 1, false_oom + pred
            err.append(abs(est['mem_gib'] - d['mem_peak_gib']) / d['mem_peak_gib'])
    err.sort()
    assert len(err) >= 300 and ooms >= 25
    assert err[int(0.9 * len(err))] <= 0.25 and caught >= 0.75 * ooms and false_oom <= 0.10 * oks


def test_the_fingerprint_keeps_the_shape_and_the_router_top_k(tmp_path):
    m = fingerprint(project(), {'log': str(megatron_log(tmp_path / 'a.log', num_experts=8, moe_router_topk=2))})['model']
    assert (m['layers'], m['hidden'], m['ffn'], m['kv_heads'], m['vocab'], m['experts'], m['topk']) == \
        (32, 4096, 14336, 8, 128256, 8, 2) and m['family'] == 'moe'
    manifest = fingerprint(project(model={'params_b': 8.0, 'layers': 30}), {'log': str(megatron_log(tmp_path / 'b.log'))})
    assert manifest['model']['layers'] == 30 and manifest['model']['hidden'] == 4096   # the manifest's values win


def test_the_plan_warns_from_the_estimate_only_without_a_measured_match_and_where_the_reference_fits(tmp_path):
    log = {'log': str(megatron_log(tmp_path / 'train.log'))}
    fp = fingerprint(project(memory=110), log)                 # the reference ~99 GiB, micro-batch 2 ~130 GiB
    pl = plan(Store([]), fp, KNOBS, MCORE_REF)
    assert pl['accounting']['mem_gib'] < 110 and not pl['check'](MCORE_REF)
    mbs2 = {**MCORE_REF, 'micro_batch_size': 2}
    assert any(w.startswith('memory (estimate): about 130 GiB of 110') for w in pl['check'](mbs2))
    assert not plan(Store([]), fp, KNOBS, MCORE_REF, accounting=False)['check'](mbs2)
    off = plan(Store([]), fingerprint(project(memory=80), log), KNOBS, MCORE_REF)   # the reference (which ran) "does not fit"
    assert not off['check'](mbs2)


def test_the_accounting_view_for_the_proposer_and_the_doctor(tmp_path):
    view = accounting_for(project(), {'log': str(megatron_log(tmp_path / 'train.log'))})
    v = view(MCORE_REF, 3.2865)
    assert 0.5 < v['mfu'] < 0.9 and v['margin_gib'] > 30 and v['measured_update_s'] == 3.2865
    assert len(json.dumps(v)) < 1200                          # the proposer's field stays near 300 tokens
    assert view({**MCORE_REF, 'micro_batch_size': 2})['memory_gib'] > v['memory_gib'] and view(MCORE_REF)['mfu'] is None
    assert accounting_for(project(), None) is None            # no argument dump and no manifest model: no shape


def test_the_doctor_lines(tmp_path, capsys):
    from infrarsi.cli.main import _print_accounting
    view = accounting_for(project(), {'log': str(megatron_log(tmp_path / 'train.log'))})
    _print_accounting({'valid': True, 'accounting': view(MCORE_REF, 3.2865), 'memory_peak_gib': 75.0})
    out = capsys.readouterr().out
    assert 'accounting (closed-form, +-25 %): about 99 GiB per GPU of 141' in out and "allocator peak 75.0 GiB" in out
    assert 'MFU 73 %' in out
    _print_accounting({'valid': True, 'accounting': None})
    assert 'it needs the model shape' in capsys.readouterr().out
    _print_accounting({'valid': True, 'accounting': None, 'accounting_off': True})
    assert capsys.readouterr().out == ''


FSDP = {**MEGATRON, 'framework': 'fsdp'}


def test_registry_names_reach_the_estimate_through_a_projects_knobs():
    knobs = {'reshard': {'type': 'str', 'choices': ['on', 'off'], 'env': 'RESHARD', 'default': 'on',
                         'mechanism': 'reshard_after_forward', 'states': {'on': 'on', 'off': 'off'}},
             'loss': {'type': 'str', 'choices': ['hf', 'liger'], 'env': 'LOSS', 'default': 'hf', 'mechanism': 'loss_impl',
                      'states': {'hf': 'hf', 'liger': 'liger'}}}
    view = accounting_for(project(framework='fsdp', knobs=knobs, model=dict(LLAMA8B),
                                  workload={'seq_len': 4096, 'global_batch_tokens': 262144}), None)
    base = view({'reshard': 'on', 'loss': 'hf'})
    assert view({'reshard': 'off', 'loss': 'hf'})['memory_gib'] > base['memory_gib'] + 10       # +2 bytes / param kept
    assert view({'reshard': 'on', 'loss': 'liger'})['terms_gib']['logits'] < base['terms_gib']['logits']
    assert base['ideal_update_s'] == estimate(FSDP, {})['ideal_step_s']       # the manifest's tokens per update
    unknown = accounting_for(project(framework='fsdp', knobs=knobs, model=dict(LLAMA8B), workload={'seq_len': 4096}), None)
    v = unknown({'reshard': 'on', 'loss': 'hf'})
    assert v['memory_gib'] == base['memory_gib'] and v['ideal_update_s'] is None and v['mfu'] is None
    assert any('tokens per update unknown' in n for n in v['notes'])


def test_the_states_name_the_backend_of_a_launcher():
    acc = {**MEGATRON, 'framework': 'accelerate'}
    ddp, z2 = estimate(acc, {})['mem_terms']['states_gib'], estimate(acc, {'zero_stage': '2'})['mem_terms']['states_gib']
    assert ddp == estimate({**MEGATRON, 'framework': 'pytorch'}, {})['mem_terms']['states_gib']   # DDP: 16 bytes / param
    assert z2 == estimate({**MEGATRON, 'framework': 'deepspeed'}, {'zero_stage': '2'})['mem_terms']['states_gib'] < ddp
    assert estimate(acc, {'fsdp_sharding': 'full_shard'})['mem_terms']['states_gib'] < z2
    assert estimate({**MEGATRON, 'framework': 'lightning'}, {'x-fsdp-version': '2'})['mem_terms']['states_gib'] < z2


def test_fsdp_sharding_modes():
    two = {**FSDP, 'nodes': 2}
    full = estimate(two, {'fsdp_sharding': 'full_shard'})['mem_terms']['states_gib']
    assert estimate(two, {'fsdp_sharding': 'shard_grad_op'})['mem_terms']['states_gib'] > full + 10
    assert estimate(two, {'fsdp_sharding': 'hybrid_shard'})['mem_terms']['states_gib'] > full   # within a node only
    assert estimate(two, {'fsdp_sharding': 'hybrid_shard'}) == estimate(two, {'hsdp': '2x8'})


def test_zero_offload_keeps_only_the_bf16_weights_on_the_gpu():
    ds = {**MEGATRON, 'framework': 'deepspeed', 'model': {**LLAMA8B, 'params_b': 32.76}}
    st = lambda z='3', **kw: estimate(ds, {'zero_stage': z, **kw})['mem_terms']['states_gib']   # noqa: E731
    assert st(cpu_offload='optimizer') == pytest.approx(32.76e9 * 2 / 8 / 2 ** 30, abs=0.01)    # the bf16 shard (HD06)
    assert st(cpu_offload='all') == 0.0 and st(z='2', cpu_offload='optimizer') == pytest.approx(32.76e9 * 2 / 2 ** 30, abs=0.01)


def test_an_estimate_does_not_keep_a_move_out_of_the_start_composite(tmp_path):
    fp = fingerprint(project(memory=115), {'log': str(megatron_log(tmp_path / 'train.log'))})
    env = {'kind': 'env', 'id': 'e', 'gpu': fp['gpu'], 'gpus_per_node': 8, 'nodes': 1, 'framework': 'megatron',
           'versions': fp['versions'], 'model': {'params_b': 8.03}, 'workload': {'seq_len': 4096}, 'noise_pct': 0.2}
    base = {'tp': '1', 'pp': '1', 'cp': '1', 'ep': '1'}
    pack = [{'kind': 'pack', 'id': 'pack', 'schema': 'infrarsi-exp/1', 'pack_id': 'p', 'title': 'p', 'license': 'CC0',
             'tier': 'measured'}, env] + [{'kind': 'effect', 'id': i, 'env': 'e', 'change': ch, 'speedup': s, 'base': base,
                                           'method': 'paired_runs'}
                                          for i, ch, s in (('mb', {'micro_batch': ['1', '2']}, 1.06),
                                                           ('bf', {'bf16_main_grads': ['fp32', 'bf16']}, 1.04),
                                                           ('ce', {'fused_cross_entropy': ['off', 'on']}, 1.03))]
    pl = plan(Store([pack]), fp, KNOBS, MCORE_REF)
    seeds = {s['knob']: s['warnings'] for s in pl['seeds']}
    assert seeds['micro_batch_size'][0].startswith('memory (estimate)')         # alone: ~130 GiB of 115
    assert set(pl['composite']['members']) == {'micro_batch_size', 'bf16_main_grads', 'cross_entropy_loss_fusion'}
    assert pl['composite']['warnings'] == []                                    # together: ~111 GiB


def test_the_argument_dump_carries_expert_tensor_parallelism(tmp_path):
    moe = dict(num_experts=8, moe_router_topk=2, tensor_model_parallel_size=2, expert_model_parallel_size=4)
    fp = fingerprint(project(), {'log': str(megatron_log(tmp_path / 'a.log', expert_tensor_parallel_size=1, **moe))})
    tp_default = fingerprint(project(), {'log': str(megatron_log(tmp_path / 'b.log', expert_tensor_parallel_size=2, **moe))})
    assert fp['states']['etp'] == '1' and tp_default['states']['etp'] == '2'
    assert estimate(fp, fp['states'])['params_local_b'] > estimate(tp_default, tp_default['states'])['params_local_b']


def test_megatron_custom_fsdp_rows_are_sharded(tmp_path):
    rows = {d['id']: (e, d) for e, d in _rows() if d['id'] in ('M63', 'HM63', 'M73')}
    for i, (e, d) in rows.items():
        cap = {'H20-96GB': 95.1, 'H200-141GB': 140.4}[e['gpu']]
        fp = {'gpu': e['gpu'], 'memory_gib': cap, 'framework': 'megatron', 'gpus_per_node': 8, 'nodes': 1, 'model': e['model'],
              'workload': e['workload']}
        est = estimate(fp, d['config'])
        assert (est['mem_gib'] > cap) == (d['status'] == 'oom'), i
        if d['status'] == 'ok':
            assert abs(est['mem_gib'] - d['mem_peak_gib']) / d['mem_peak_gib'] < 0.25, i
        plain = estimate(fp, {**d['config'], 'x-megatron-fsdp': 'none'})
        assert plain['mem_terms']['states_gib'] > est['mem_terms']['states_gib'], i


def test_megatron_expert_tensor_parallelism_defaults_to_tp_and_routing_needs_top_k():
    moe = {**MEGATRON, 'model': {'params_b': 30.53, 'layers': 48, 'hidden': 2048, 'ffn': 768, 'heads': 32, 'kv_heads': 4,
                                 'vocab': 151936, 'experts': 128, 'topk': 8}}
    tp2 = {'tp': '2', 'ep': '4', 'distributed_optimizer': 'on', 'sequence_parallel': 'on'}
    assert estimate(moe, tp2) == estimate(moe, {**tp2, 'etp': '2'})                         # ETP defaults to TP
    assert estimate(moe, {**tp2, 'etp': '1'})['params_local_b'] > estimate(moe, tp2)['params_local_b']
    assert estimate(moe, tp2)['flops_per_update'] is not None
    no_k = estimate({**moe, 'model': {k: v for k, v in moe['model'].items() if k != 'topk'}}, tp2)
    assert no_k['flops_per_update'] is None and any('top-k' in n for n in no_k['notes'])
