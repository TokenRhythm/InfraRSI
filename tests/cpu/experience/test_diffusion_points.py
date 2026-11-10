"""Experience-layer points raised by the diffusion harvest packs (reports/diffusion_kb_20261022/experience/README.md):
accelerate-launched frameworks, a move whose size depends on another state (TF32 vs precision), and the memory look-up
keeping LoRA, full fine-tuning and sharding apart."""
from infrarsi.experience.pack import validate
from infrarsi.experience.store import Store, plan


def pack(*rest, framework='accelerate'):
    return [{'kind': 'pack', 'id': 'pack', 'schema': 'infrarsi-exp/1', 'pack_id': 'diff', 'title': 'diff',
             'license': 'CC0', 'tier': 'measured'},
            {'kind': 'env', 'id': 'unet', 'gpu': 'H100-SXM-80GB', 'gpus_per_node': 8, 'nodes': 1, 'framework': framework,
             'model': {'params_b': 0.86}, 'workload': {'seq_len': 4096, 'trainable': 'full'}},
            {'kind': 'env', 'id': 'mmdit', 'gpu': 'H100-SXM-80GB', 'gpus_per_node': 8, 'nodes': 1, 'framework': framework,
             'model': {'params_b': 11.9}, 'workload': {'seq_len': 4096, 'trainable': 'lora-r32'}},
            {'kind': 'env', 'id': 'mmdit-full', 'gpu': 'H100-SXM-80GB', 'gpus_per_node': 8, 'nodes': 1,
             'framework': framework, 'model': {'params_b': 11.9}, 'workload': {'seq_len': 4096, 'trainable': 'full'}},
            *rest]


def fp(precision='bf16', trainable='lora-r32'):
    return {'gpu': 'H100-SXM-80GB', 'memory_gib': 80, 'gpus_per_node': 8, 'nodes': 1, 'framework': 'accelerate',
            'model': {'params_b': 11.9}, 'workload': {'seq_len': 4096, 'trainable': trainable},
            'states': {'precision': precision}}


def test_accelerate_is_a_pack_framework():
    errors, _ = validate(pack())
    assert errors == []


def test_tf32_measured_in_fp32_is_not_a_covered_prior_for_a_bf16_target():
    st = Store([pack(
        {'kind': 'effect', 'id': 'sd15', 'env': 'unet', 'change': {'tf32': ['off', 'on']}, 'speedup': 1.25,
         'method': 'paired_runs', 'noise_pct': 0.3, 'base': {'precision': 'fp32'}},
        {'kind': 'effect', 'id': 'flux', 'env': 'mmdit', 'change': {'tf32': ['off', 'on']}, 'speedup': 1.0007,
         'method': 'paired_runs', 'noise_pct': 0.3, 'base': {'precision': 'bf16'}})])
    bf16 = st.evidence(fp('bf16'), 'tf32', 'kernel', 'off', 'on', ref_states={'precision': 'bf16'})
    assert [(r['effect']['id'], r['level']) for r in bf16] == [('flux', 'covered'), ('sd15', 'partial')]
    fp32 = st.evidence(fp('fp32'), 'tf32', 'kernel', 'off', 'on', ref_states={'precision': 'fp32'})
    assert {r['effect']['id']: r['level'] for r in fp32} == {'sd15': 'covered', 'flux': 'partial'}


def test_memory_lookup_keeps_lora_full_and_sharding_apart():
    cfg = {'micro_batch': '1', 'precision': 'bf16', 'recompute': 'full'}
    st = Store([pack(
        {'kind': 'observation', 'id': 'full-z2', 'env': 'mmdit-full', 'status': 'ok', 'step_time_s': 1.17,
         'mem_peak_gib': 77.4, 'config': {**cfg, 'zero_stage': '2'}},
        {'kind': 'observation', 'id': 'full-z3', 'env': 'mmdit-full', 'status': 'ok', 'step_time_s': 1.55,
         'mem_peak_gib': 43.0, 'config': {**cfg, 'zero_stage': '3'}},
        {'kind': 'observation', 'id': 'lora', 'env': 'mmdit', 'status': 'ok', 'step_time_s': 0.94,
         'mem_peak_gib': 37.1, 'config': cfg})])
    assert st.fits(fp(), cfg)['evidence'] == 'diff/lora'
    assert st.fits(fp(trainable='full'), {**cfg, 'zero_stage': '3'})['evidence'] == 'diff/full-z3'
    assert st.fits(fp(trainable='full'), {**cfg, 'zero_stage': '2'})['evidence'] == 'diff/full-z2'
    assert st.fits(fp(trainable=None), cfg) is None       # a target that does not say what it trains: no match


def test_a_move_whose_size_depends_on_another_is_neither_stacked_nor_reseeded():
    st = Store([pack(
        {'kind': 'effect', 'id': 'prec', 'env': 'unet', 'change': {'precision': ['fp32', 'bf16']}, 'speedup': 2.58,
         'method': 'paired_runs', 'noise_pct': 0.3, 'base': {'tf32': 'off'}},
        {'kind': 'effect', 'id': 'tf32', 'env': 'unet', 'change': {'tf32': ['off', 'on']}, 'speedup': 1.25,
         'method': 'paired_runs', 'noise_pct': 0.3, 'base': {'precision': 'fp32'}})])
    knobs = {'mixed_precision': {'type': 'str', 'choices': ['no', 'bf16'], 'flag': '--mixed_precision', 'default': 'no'},
             'allow_tf32': {'type': 'bool', 'flag': '--allow_tf32'}}
    target = dict(fp(trainable='full'), model={'params_b': 0.86}, states={})
    pl = plan(st, target, knobs, {'mixed_precision': 'no', 'allow_tf32': False})
    seeds = {s['knob']: s for s in pl['seeds']}
    assert seeds['allow_tf32']['level'] == 'covered'                # fp32 reference: the fp32 TF32 gain applies
    assert seeds['allow_tf32']['valid_on'] == {'allow_tf32': False, 'mixed_precision': 'no'}   # not reseeded after bf16
    assert pl['composite'] is None                                  # bf16 and TF32 together are not 2.58 x 1.25
