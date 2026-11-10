"""§2.4 knob space and the finetune.yaml / finetune.sh templater."""
import random

import yaml

from infrarsi.ext.rebench_llmf import knobs as K
from infrarsi.ext.rebench_llmf import read_base_solution


def base():
    return read_base_solution()


def test_space_size_and_schema():
    assert K.SPACE_SIZE == 6144
    sch = K.knob_schema()
    assert set(sch['properties']) == set(K.KNOBS)
    # the declared values match the design exactly
    assert sch['properties']['device_train_microbatch_size']['enum'] == [4, 8, 16, 'auto']
    assert sch['properties']['tokenize.num_workers']['enum'] == [None, 1, 8, 32]
    # the full product of the nine knob domains is 6144 points
    prod = 1
    for spec in K.KNOBS.values():
        prod *= len(spec['values'])
    assert prod == 6144


def test_render_sets_declared_keys_only():
    b = base()
    yml, sh, applied = K.render(b['finetune.yaml'], b['finetune.sh'], {
        'device_train_microbatch_size': 16, 'fsdp_config.sharding_strategy': 'NO_SHARD',
        'fsdp_config.activation_checkpointing': False, 'fsdp_config.limit_all_gathers': False,
        'fsdp_config.forward_prefetch': True, 'model.use_flash_attention_2': True,
        'train_loader.num_workers': 1, 'tokenize.num_workers': 32})
    c = yaml.safe_load(yml)
    assert c['device_train_microbatch_size'] == 16
    assert c['fsdp_config']['sharding_strategy'] == 'NO_SHARD'
    assert c['fsdp_config']['activation_checkpointing'] is False
    assert c['fsdp_config']['limit_all_gathers'] is False
    assert c['fsdp_config']['forward_prefetch'] is True
    assert c['model']['use_flash_attention_2'] is True
    assert c['train_loader']['num_workers'] == 1
    assert '--num_workers 32' in sh
    assert K.check_fixed_semantics(yml, sh) == []


def test_render_preserves_fixed_semantics_for_every_single_knob():
    b = base()
    for name, spec in K.KNOBS.items():
        for v in spec['values']:
            yml, sh, _ = K.render(b['finetune.yaml'], b['finetune.sh'], {name: v})
            assert K.check_fixed_semantics(yml, sh) == [], f'{name}={v!r} broke a fixed semantic'


def test_skip_valid_copy_edits_both_files():
    b = base()
    yml, sh, _ = K.render(b['finetune.yaml'], b['finetune.sh'], {'skip_valid_copy': True})
    assert 'eval_loader' not in yaml.safe_load(yml)
    assert 'cp -r finetuning_tokens/train finetuning_tokens/valid' not in sh
    assert 'eval_loader.dataset.split=valid' not in sh
    # with skip_valid_copy false the valid copy stays
    yml2, sh2, _ = K.render(b['finetune.yaml'], b['finetune.sh'], {'skip_valid_copy': False})
    assert 'cp -r finetuning_tokens/train finetuning_tokens/valid' in sh2


def test_auto_microbatch_repair():
    norm, errs, warns = K.validate({'device_train_microbatch_size': 'auto',
                                    'fsdp_config.activation_checkpointing': True})
    assert not errs
    assert norm['fsdp_config.activation_checkpointing'] is False       # auto wins (pre-declared repair)
    assert warns and 'repair' in warns[0]


def test_validate_rejects_unknown_and_out_of_domain():
    _, errs, _ = K.validate({'foo': 1})
    assert errs and 'unknown knob' in errs[0]
    _, errs2, _ = K.validate({'device_train_microbatch_size': 99})
    assert errs2 and 'not in' in errs2[0]


def test_validate_normalizes_string_spellings():
    norm, errs, _ = K.validate({'device_train_microbatch_size': 'auto', 'tokenize.num_workers': 'None',
                                'model.use_flash_attention_2': 'true'})
    assert not errs
    assert norm['device_train_microbatch_size'] == 'auto'
    assert norm['tokenize.num_workers'] is None
    assert norm['model.use_flash_attention_2'] is True


def test_sample_is_deterministic_and_valid():
    s1 = K.sample(random.Random(42))
    s2 = K.sample(random.Random(42))
    assert s1 == s2
    _, errs, _ = K.validate(s1)
    assert not errs
    b = base()
    yml, sh, _ = K.render(b['finetune.yaml'], b['finetune.sh'], s1)
    assert K.check_fixed_semantics(yml, sh) == []
