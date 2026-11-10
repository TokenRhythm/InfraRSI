import json
import re
from pathlib import Path

import pytest

from infrarsi.ext.optico import params as pm
from infrarsi.ext.optico import task as taskmod
from infrarsi.ext.optico import workloads as wl

OPTICO = taskmod.OPTICO_ROOT


def _val(workload, **changes):
    return pm.validate_optico_params(wl.get(workload).knobs(**changes), workload)


@pytest.mark.skipif(not (OPTICO / 'script' / 'train').exists(), reason='local OptiCo-public copy not present')
@pytest.mark.parametrize('key', sorted(wl.WORKLOADS))
def test_workload_shapes_and_defaults_match_the_scripts(key):
    w = wl.WORKLOADS[key]
    text = (OPTICO / 'script' / 'train' / w.script).read_text(encoding='utf-8')
    block = text.split(f'if [ $MODEL_SIZE = {w.model_size} ]; then', 1)
    if len(block) == 1:
        block = text.split(f'elif [ $MODEL_SIZE = {w.model_size} ]; then', 1)
    body = block[1].split('elif [ $MODEL_SIZE', 1)[0].split('\nfi\n', 1)[0]
    assert int(re.search(r'NUM_LAYERS=(\d+)', body).group(1)) == w.num_layers
    assert int(re.search(r'HIDDEN_SIZE=(\d+)', body).group(1)) == w.hidden
    assert int(re.search(r'NUM_ATTN_HEADS=(\d+)', body).group(1)) == w.heads
    if w.ffn_flag:
        assert int(re.search(r'INTERMEDIATE_SIZE=(\d+)', body).group(1)) == w.ffn
    if w.gqa_flag:
        kv = re.search(r'--num-query-groups (\$\{NUM_KEY_VALUE_HEADS\}|\d+)', body).group(1)
        kv = int(re.search(r'NUM_KEY_VALUE_HEADS=(\d+)', body).group(1)) if kv.startswith('$') else int(kv)
        assert kv == w.kv_heads
    # defaults: the size block of the knobs and the global ${X:-default}
    head = text.split(f'"{w.model_size}" ]; then', 1)[1].split('fi\n', 1)[0]
    for knob in ('AC', 'AC_LAYERS', 'TP', 'PP'):
        v = re.search(rf'{knob}=\${{{knob}:-([a-z0-9]+)}}', head).group(1)
        assert str(w.defaults[knob]) == v
    assert 'GLOBAL_BATCH_SIZE=1024' in text and 'SEQ_LENGTH=4096' in text and 'TRAIN_ITERS=5' in text
    assert pm.validate_optico_params(w.defaults, w)[1] == []


@pytest.mark.skipif(not (OPTICO / 'script').exists(), reason='local OptiCo-public copy not present')
def test_pinned_frozen_manifest_matches_the_local_checkout():
    import hashlib
    pinned = taskmod.frozen_files()
    local = {p.relative_to(OPTICO).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in OPTICO.rglob('*') if p.is_file() and p.relative_to(OPTICO).parts[0] in ('script', 'prompt')}
    assert pinned == local and 'script/monitor/run_monitor.sh' in pinned and len(pinned) == 43


def test_encode_decode_round_trip_in_run_monitor_format():
    p = wl.get('W1').knobs(MICRO_BATCH_SIZE=2, OVERLAP_GRAD_REDUCE=True, FIRST_PP_NUM_LAYERS=6, LAST_PP_NUM_LAYERS=6)
    text = pm.encode_params(p)
    assert text.splitlines()[0] == 'MICRO_BATCH_SIZE: 2' and 'OVERLAP_GRAD_REDUCE: true' in text
    assert 'LAST_PP_NUM_LAYERS: 6' in text and text.endswith('\n')
    assert pm.typed(pm.decode_params(text), strict=True) == p
    # analyzer.py's write_dict_to_txt format ('KEY:value') and OptiCo's own spellings decode the same way
    raw = pm.decode_params('MICRO_BATCH_SIZE:4\nAC:SEL\nFIRST_PP_NUM_LAYERS:None\nOVERLAP_PARAM_GATHER: True\n')
    assert pm.typed(raw) == {'MICRO_BATCH_SIZE': 4, 'AC': 'sel', 'FIRST_PP_NUM_LAYERS': None, 'OVERLAP_PARAM_GATHER': True}
    w3 = wl.get('W3').defaults
    assert pm.encode_params(w3) == ('MICRO_BATCH_SIZE: 1\nTP: 1\nPP: 1\nOVERLAP_GRAD_REDUCE: false\nOVERLAP_PARAM_GATHER: false\n'
                                    'AC: none\nAC_LAYERS: 1\nFUSED_CROSS_ENTROPY: false\nFIRST_PP_NUM_LAYERS: None\n'
                                    'LAST_PP_NUM_LAYERS: None\n')        # = the R3 parameters.txt written by run_monitor.sh


def test_defaults_are_valid_and_missing_keys_take_script_defaults():
    for w in wl.WORKLOADS.values():
        norm, errs, warns = pm.validate_optico_params({}, w)
        assert norm == w.defaults and errs == [] and warns == []
    norm, errs, _ = pm.validate_optico_params({'MICRO_BATCH_SIZE': '2', 'FUSED_CROSS_ENTROPY': 'true'}, 'W3')
    assert errs == [] and norm['MICRO_BATCH_SIZE'] == 2 and norm['FUSED_CROSS_ENTROPY'] is True and norm['TP'] == 1


@pytest.mark.parametrize('workload,changes,needle', [
    ('W3', {'TP': 4}, 'num_attention_heads 14 is not divisible by TP 4'),               # 14 heads, 2 KV groups
    ('W3', {'TP': 2}, None),
    ('W1', {'TP': 8, 'PP': 1}, 'num_attention_heads 28'),
    ('W2', {'TP': 4, 'PP': 4}, 'TP*PP = 16 does not divide'),
    ('W1', {'PP': 8, 'AC': 'none'}, 'NUM_LAYERS 28 is not divisible by PP 8'),
    ('W1', {'PP': 8, 'AC': 'none', 'FIRST_PP_NUM_LAYERS': 2, 'LAST_PP_NUM_LAYERS': 2}, None),   # 2 + 6x4 + 2
    ('W3', {'AC': 'full', 'AC_LAYERS': 5}, 'multiple of the recompute layers'),
    ('W1', {'AC': 'full', 'AC_LAYERS': 7}, None),                                       # 28/4 = 7 per stage
    ('W3', {'PP': 1, 'FIRST_PP_NUM_LAYERS': 4}, 'When PP=1'),
    ('W3', {'PP': 2, 'FIRST_PP_NUM_LAYERS': 10}, 'FIRST_PP_NUM_LAYERS + LAST_PP_NUM_LAYERS must equal NUM_LAYERS'),
    ('W3', {'PP': 2, 'FIRST_PP_NUM_LAYERS': 10, 'LAST_PP_NUM_LAYERS': 14}, None),
    ('W3', {'PP': 4, 'FIRST_PP_NUM_LAYERS': 4}, 'middle stage 20 must be divisible'),   # script OK (20 % 2), Megatron 20 % 3
    ('W3', {'PP': 4, 'FIRST_PP_NUM_LAYERS': 6}, None),
    ('W3', {'PP': 4, 'FIRST_PP_NUM_LAYERS': 5, 'LAST_PP_NUM_LAYERS': 4}, 'Middle layers(15) must be divisible by (PP-2)=2'),
    ('W3', {'PP': 4, 'FIRST_PP_NUM_LAYERS': 4, 'LAST_PP_NUM_LAYERS': 4, 'AC': 'full', 'AC_LAYERS': 4},
     None),                                                                             # stages 4/8/8/4, (24/4) % 4 -> script!
    ('W3', {'PP': 2, 'AC': 'offload', 'AC_LAYERS': 2}, 'Pipeline parallelism with CPU offloading'),
    ('W3', {'AC': 'offload', 'AC_LAYERS': 24}, 'only for layers less than 24'),
    ('W3', {'OVERLAP_PARAM_GATHER': True}, 'Must use --overlap-param-gather with --overlap-grad-reduce'),
    ('W3', {'MICRO_BATCH_SIZE': 3}, 'MICRO_BATCH_SIZE 3 not in'),
    ('W3', {'AC': 'selective'}, 'AC must be one of'),
    ('W3', {'TP': 'two'}, 'TP must be an integer'),
    ('W3', {'FUSED_CROSS_ENTROPY': 'yes'}, 'must be true or false'),
    ('W3', {'GLOBAL_BATCH_SIZE': 512}, 'unknown parameter'),
])
def test_validator_rules(workload, changes, needle):
    norm, errs, _ = pm.validate_optico_params(wl.get(workload).knobs(**changes) if 'GLOBAL_BATCH_SIZE' not in changes
                                              else {**wl.get(workload).defaults, **changes}, workload)
    if workload == 'W3' and changes.get('LAST_PP_NUM_LAYERS') == 4 and changes.get('AC') == 'full':
        # (24 // 4) % 4 = 2: the script's own (integer-division) check rejects although every stage divides by 4
        assert any('multiple of the recompute layers' in e for e in errs)
        return
    if needle is None:
        assert errs == [], errs
    else:
        assert any(needle in e for e in errs), errs


def test_normalizations_are_warnings_and_dedupe():
    norm, errs, warns = _val('W3', AC='sel', AC_LAYERS=4)
    assert errs == [] and norm['AC_LAYERS'] == 1 and 'no effect' in warns[0]
    norm2, _, _ = _val('W3', AC='sel', AC_LAYERS=1)
    assert pm.params_key(norm) == pm.params_key(norm2)
    norm, errs, warns = _val('W3', AC='offload', AC_LAYERS=2, OVERLAP_GRAD_REDUCE=True, OVERLAP_PARAM_GATHER=True)
    assert errs == [] and not norm['OVERLAP_GRAD_REDUCE'] and not norm['OVERLAP_PARAM_GATHER'] and 'offload' in warns[0]


def test_stage_layers_and_layout():
    assert pm.stage_layers(28, 4, None, None) == ([7, 7, 7, 7], [])
    assert pm.stage_layers(24, 4, 6, None) == ([6, 6, 6, 6], [])
    assert pm.stage_layers(28, 8, 2, 2)[0] == [2, 4, 4, 4, 4, 4, 4, 2]
    assert pm.stage_layers(24, 2, 10, 14)[0] == [10, 14]
    assert pm.layout(wl.get('W1').defaults) == {'tp': 1, 'pp': 4, 'dp': 2, 'mbs': 1, 'microbatches': 512}
    assert pm.format_params(wl.get('W1').defaults) == 'MBS1 TP1 PP4 AC full/1'


def test_knob_schema_is_json_and_complete():
    s = pm.knob_schema('W1')
    assert set(s['properties']) == set(pm.KNOBS) and json.loads(json.dumps(s)) == s
    assert all('default' in v for v in s['properties'].values())
