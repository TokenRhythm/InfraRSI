"""Whole-codebase review fixes in the experience layer (2026-10-24, C3): NCCL's automatic NVLS mode and FSDP compile
booleans have states, so write-back keeps the run's observations and says what it left out (#1); measured values must be
numbers (#2); a knob named after a mechanism maps by meaning, not by its default (#3); a candidate promotion turned down
is no positive experience (#4); a memory match needs the same optimizer sharding (#5); the harvest packs' x- ids that
the registry has since are read under the registered id (#6); fp32 jobs get no bf16 estimate (#7); the judgment schema
is the record write-back produces (#8)."""
import json

from expjob import KNOBS, megatron_log, project
from infrarsi.experience.accounting import estimate
from infrarsi.experience.pack import SCHEMA, migrate, validate
from infrarsi.experience.registry import label
from infrarsi.experience.runs import write_back
from infrarsi.experience.store import Store, fingerprint, plan, read_lines

HEAD = {'kind': 'pack', 'id': 'pack', 'schema': SCHEMA, 'pack_id': 'p', 'title': 't', 'license': 'x', 'tier': 'measured'}
ENV = {'kind': 'env', 'id': 'e', 'gpu': 'H20-96GB', 'gpus_per_node': 8, 'nodes': 1, 'framework': 'megatron',
       'model': {'params_b': 8.0}, 'workload': {'seq_len': 4096}}


def test_nvls_auto_and_fsdp_compile_booleans_have_states():
    nvls = label('NCCL_NVLS_ENABLE', {'type': 'str', 'choices': ['2', '0'], 'env': 'NCCL_NVLS_ENABLE', 'default': '2'},
                 'megatron')
    assert (nvls.mechanism, nvls.state('2'), nvls.state('0')) == ('nvls', 'on', 'off')
    comp = label('compile', {'type': 'bool', 'path': 'training.compile'}, 'fsdp')
    assert (comp.mechanism, comp.state(True), comp.state(False)) == ('compile', 'on', 'off')


def test_a_knob_named_after_a_mechanism_maps_by_meaning():
    on = label('compile', {'type': 'bool', 'flag': '--compile', 'default': True}, 'pytorch')
    assert (on.state(True), on.state(False)) == ('on', 'off')                  # not default -> default
    gc = label('manual_gc', {'type': 'str', 'choices': ['off', 'on'], 'default': 'on'}, 'pytorch')
    assert gc.state('on') == 'on'
    assert label('compile', {'type': 'str', 'choices': ['yes', 'no']}, 'pytorch').state('yes') is None   # no meaning


def _trace(change):
    return {'candidates': [{'id': 'c0', 'parent': None, 'change': None, 'valid': True, 'step_time': 3.0},
                           {'id': 'c1', 'parent': 'c0', 'change': change, 'valid': True, 'step_time': 2.5}],
            'screens': [{'candidate_id': 'c1', 'fidelity': 'full', 'valid': True, 'step_time': 2.5}], 'extra': {}}


def test_write_back_keeps_nvls_auto_runs_and_leaves_out_what_promotion_turned_down(tmp_path, monkeypatch):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    knobs = dict(KNOBS, NCCL_NVLS_ENABLE={'type': 'str', 'choices': ['2', '0'], 'env': 'NCCL_NVLS_ENABLE', 'default': '2'})
    proj = project(knobs=knobs)
    ref = proj.reference_config()
    fp = fingerprint(proj, {'log': str(megatron_log(tmp_path / 'train.log'))})
    pl = plan(Store([]), fp, knobs, ref)
    lines = read_lines(write_back(proj, pl, _trace({'manual_gc': [False, True]}), None, 'w1', 0.001))
    assert sorted(x['id'] for x in lines if x['kind'] == 'observation') == ['c0', 'c1']
    assert lines[0]['summary'].endswith('2 configurations at full fidelity')
    promo = {'tried': [{'candidate': 'c1', 'items': [{'id': 'speed_reproduces', 'status': 'pass'},
                                                     {'id': 'eval_matches', 'status': 'fail'}]}]}
    lines = read_lines(write_back(proj, pl, _trace({'manual_gc': [False, True]}), promo, 'w2', 0.001))
    assert [x['id'] for x in lines if x['kind'] == 'observation'] == ['c0']
    assert '1 left out' in lines[0]['summary']


def test_measured_values_must_be_numbers():
    obs = {'kind': 'observation', 'id': 'o1', 'env': 'e', 'config': {'compile': 'off'}, 'status': 'ok', 'step_time_s': '1.0'}
    errors, _ = validate([HEAD, ENV, obs])
    assert any('step_time_s must be a positive number' in e for e in errors)
    assert not validate([HEAD, ENV, dict(obs, step_time_s=1.0)])[0]


def test_a_memory_match_needs_the_same_optimizer_sharding():
    oom = {'kind': 'observation', 'id': 'o1', 'env': 'e', 'status': 'oom',
           'config': {'micro_batch': '2', 'recompute': 'full', 'distributed_optimizer': 'off'}}
    s = Store([[HEAD, ENV, oom]])
    fp = {k: ENV[k] for k in ('gpu', 'gpus_per_node', 'nodes', 'framework', 'model', 'workload')}
    assert s.fits(fp, oom['config'])['fits'] is False
    assert s.fits(fp, dict(oom['config'], distributed_optimizer='on')) is None


def test_registered_harvest_ids_are_read_under_the_registered_id():
    x = migrate({'kind': 'effect', 'change': {'x-fp8-attention': ['off', 'on']}, 'base': {'x-ds-optimizer': 'a', 'tp': '1'}})
    assert x['change'] == {'fp8_attention': ['off', 'on']} and x['base'] == {'x-ds-optimizer': 'a', 'tp': '1'}
    s = Store.load()
    assert s.by_mech.get('fp8_attention') and not s.by_mech.get('x-fp8-attention')
    assert not any(k.startswith('x-megatron-fsdp') for c in s.constraints for k in c.get('when') or {})


def test_fp32_jobs_get_no_bf16_estimate():
    fp = {'model': {'params_b': 8.0, 'layers': 32, 'hidden': 4096}, 'workload': {'seq_len': 4096}, 'gpus_per_node': 8,
          'nodes': 1, 'framework': 'pytorch', 'gpu': 'H100-SXM-80GB'}
    assert estimate(fp, {'precision': 'bf16'}) is not None and estimate(fp, {'precision': 'fp32'}) is None


def test_the_judgment_schema_is_the_record_write_back_produces():
    from pathlib import Path
    import infrarsi.experience as exp
    schema = json.loads((Path(exp.__file__).with_name('data') / 'schema.json').read_text(encoding='utf-8'))
    rule = next(r for r in schema['allOf'] if r['if']['properties']['kind'].get('const') == 'judgment')
    assert rule['then']['required'] == ['assessment']
    j = {'kind': 'judgment', 'id': 'transfer', 'env': 'e', 'model': 'm', 'target_gpu': 'H20-96GB', 'source_envs': [],
         'source_gpus': [], 'scores': {}, 'assessment': [{'change': {'compile': True}, 'sign': '+'}], 'start_config': {}}
    assert not validate([HEAD, ENV, j])[0]
    assert validate([HEAD, ENV, dict(j, assessment=None)])[0]


# ---------------------------------------------------------------- round 2 (Astra CF_r1)
def test_migration_keeps_ids_whose_states_do_not_fit():
    keep = migrate({'kind': 'observation', 'config': {'bf16_main_grads': 'fp32', 'x-bf16-main-grads': '1'}})
    assert keep['config'] == {'bf16_main_grads': 'fp32', 'x-bf16-main-grads': '1'}   # the canonical entry stays
    assert migrate({'kind': 'observation', 'config': {'x-hsdp': '2'}})['config'] == {'x-hsdp': '2'}
    assert migrate({'kind': 'effect', 'change': {'x-hsdp': ['none', '2x4']}})['change'] == {'hsdp': ['none', '2x4']}


def test_a_memory_match_needs_the_same_nvls_state():
    oom = {'kind': 'observation', 'id': 'o1', 'env': 'e', 'status': 'oom', 'config': {'micro_batch': '2', 'nvls': 'on'}}
    s = Store([[HEAD, ENV, oom]])
    fp = {k: ENV[k] for k in ('gpu', 'gpus_per_node', 'nodes', 'framework', 'model', 'workload')}
    assert s.fits(fp, {'micro_batch': '2', 'nvls': 'off'}) is None


def test_the_transfer_call_is_admitted_against_the_budget():
    from types import SimpleNamespace
    from infrarsi.experience import transfer

    class Driver:
        calls = []

        def complete(self, *a, **kw):
            self.calls.append(kw)
    led = SimpleNamespace(budget=SimpleNamespace(llm_tokens=0, llm_usd=None, wall_seconds=1e9),
                          used={'llm_tokens': 0, 'llm_usd': 0.0}, wall=lambda: 0.0)
    pl = {'coverage': {}, 'fingerprint': {'gpu': None}, 'payload': {k: [] for k in ('nearest', 'effects', 'constraints',
                                                                                 'rules')} | {'target': {}}}
    d = Driver()
    assert transfer.judge(d, pl, {}, {}, session=SimpleNamespace(ledger=led)) is None and not d.calls
