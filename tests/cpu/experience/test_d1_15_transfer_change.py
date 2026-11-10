"""Trial D1-15 (Opus): the transfer judgment wrote each change as [reference, new], the form the input's effects use,
and null for a knob's unknown mechanism; the schema dropped the whole judgment."""
from infrarsi.agent.jsonschema_lite import validate
from infrarsi.experience import transfer
from infrarsi.experience.store import Store, fingerprint, plan

from expjob import KNOBS, REF, megatron_log, project


def test_a_judgment_in_the_effects_form_is_used(tmp_path):
    fp = fingerprint(project('NVIDIA A100-SXM4-80GB', 80), {'log': str(megatron_log(tmp_path / 'train.log'))})
    pl = plan(Store.load(), fp, KNOBS, REF)
    out = {'assessment': [
        {'knob': 'micro_batch_size', 'change': [1, 2], 'mechanism': None, 'sign': '+', 'flip': False, 'p_feasible': 0.65,
         'expected_gain_pct': 8, 'why': 'a larger micro-batch beats accumulation', 'evidence_ids': []},
        {'knob': 'num_workers', 'change': {'num_workers': 8}, 'mechanism': None, 'sign': 'unknown', 'flip': False,
         'p_feasible': None, 'expected_gain_pct': None, 'why': '', 'evidence_ids': []}],
        'start_config': {}, 'memory_note': ''}
    assert validate(out, transfer.TRANSFER_SCHEMA) == []
    rec = transfer.apply(pl, out, KNOBS, REF)['assessment']
    assert [(r['knob'], r['change']) for r in rec] == [('micro_batch_size', {'micro_batch_size': 2}),
                                                       ('num_workers', {'num_workers': 8})]
    assert rec[1]['p_feasible'] == 0.7 and rec[1]['expected_gain_pct'] is None


def test_a_scalar_change_is_the_new_value(tmp_path):
    # R2-03 (Qwen3-Next-80B): "change": 2, the knob's new value itself
    fp = fingerprint(project('NVIDIA A100-SXM4-80GB', 80), {'log': str(megatron_log(tmp_path / 'train.log'))})
    pl = plan(Store.load(), fp, KNOBS, REF)
    out = {'assessment': [{'knob': 'micro_batch_size', 'change': 2, 'sign': '+', 'why': ''},
                          {'knob': 'num_workers', 'change': '8', 'sign': '0', 'why': ''}]}
    assert validate(out, transfer.TRANSFER_SCHEMA) == []
    rec = transfer.apply(pl, out, KNOBS, REF)['assessment']
    assert [(r['knob'], r['change']) for r in rec] == [('micro_batch_size', {'micro_batch_size': 2}),
                                                       ('num_workers', {'num_workers': 8})]
