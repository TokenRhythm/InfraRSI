"""AMENDMENT-002 trainer receipt fields (builder X4): lifecycle.deterministic_overrides (what a deterministic
lifecycle replaced, with the requested values) and lifecycle.resumed_bitwise_updates (report only). The pure helpers
run here; the supervisor part needs torch (WINH frozen venv) and is skipped elsewhere."""
import copy
import json
from pathlib import Path

import pytest

from infrarsi.runtime import lifecycle


def test_deterministic_overrides_record_requested_and_ran():
    ov = lifecycle.deterministic_overrides({'attention_backend': 'auto'}, {'NCCL_ALGO': 'Tree', 'CUDA_DEVICE_MAX_CONNECTIONS': '1'},
                                           'flash')
    assert ov['attention_backend'] == {'requested': 'auto', 'ran': 'flash'}
    assert ov['env.NCCL_ALGO'] == {'requested': 'Tree', 'ran': 'Ring'}
    assert ov['env.CUBLAS_WORKSPACE_CONFIG'] == {'requested': None, 'ran': ':4096:8'}
    assert ov['env.NVTE_ALLOW_NONDETERMINISTIC_ALGO'] == {'requested': None, 'ran': '0'}
    assert ov['torch.use_deterministic_algorithms'] == {'requested': False, 'ran': True}
    assert ov['TransformerConfig.deterministic_mode'] == {'requested': False, 'ran': True}
    mla = lifecycle.deterministic_overrides({'attention_backend': 'auto'}, dict(lifecycle.DETERMINISTIC_ENV), 'auto')
    assert 'attention_backend' not in mla and not any(k.startswith('env.') for k in mla)   # nothing else replaced
    json.dumps(ov)                                                                          # receipt-serialisable


def test_continuation_counts_bitwise_resumed_updates():
    ref = [{'step': s, 'main_ce': 2.0, 'grad_norm': 1.0, 'aux_loss': None, 'lr': 1e-5, 'supervised_tokens': 10,
            'clip_calls': 1} for s in range(31, 41)]
    tol, ltol = {'main_ce_rel_max_per_step': 0.005}, dict(lifecycle.LIFECYCLE_TOLERANCE)
    r = lifecycle.compare_continuation(ref, copy.deepcopy(ref), tol, ltol)
    assert r['bitwise'] and r['bitwise_updates'] == r['updates'] == 10
    drift = copy.deepcopy(ref)
    drift[4]['grad_norm'] = 1.01
    r = lifecycle.compare_continuation(ref, drift, tol, ltol)
    assert r['match'] and not r['bitwise'] and r['bitwise_updates'] == 9


def test_gated_deltanet_spec_gates_reach_the_trainer_continuation():
    """The trainer reads the spec, which carries its tier's values (the harness rejects any other spec)."""
    ref = [{'step': s, 'main_ce': 2.0, 'grad_norm': 1.0, 'aux_loss': 0.01, 'lr': 1e-5, 'supervised_tokens': 10,
            'clip_calls': 1} for s in range(31, 35)]
    first = copy.deepcopy(ref)
    first[0]['main_ce'] = 2.0 * (1 + 5e-4)
    gdn = {'numerics': {'lifecycle_tolerance': {'deterministic': 'unavailable',
                                                'first_resumed_update': {'main_ce_rel_max': 1e-3, 'grad_norm_rel_max': 0.03}}}}
    tol = {'main_ce_rel_max_per_step': 0.005}
    assert lifecycle.compare_continuation(ref, first, tol, lifecycle.lifecycle_tolerance(gdn))['match']
    assert not lifecycle.compare_continuation(ref, first, tol, lifecycle.lifecycle_tolerance({}))['match']


def test_supervisor_lifecycle_block_carries_the_new_fields(tmp_path, monkeypatch, make_request):
    pytest.importorskip('torch')
    import infrarsi.contracts as contracts
    from infrarsi.adapters.megatron_core.supervisor import Supervisor
    from infrarsi.runtime import receipt_writer as rw
    monkeypatch.setattr(contracts.RemotePaths, 'runs', str(tmp_path))
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('WORLD_SIZE', '1')
    req = make_request('S1', {}, gpus=1, purpose='lifecycle', input_plan='B',
                       lifecycle={'save_at': 4, 'resume_steps': 2, 'deterministic': True})
    path = tmp_path / 'request.json'
    path.write_text(json.dumps(req.to_dict()), encoding='utf-8')
    sup = Supervisor(str(path))
    ov = lifecycle.deterministic_overrides({'attention_backend': 'auto'}, {}, 'flash')
    life = Path(req.out_dir) / 'lifecycle'
    rw.write_json_atomic(life / 'resume_parts.json', {'lifecycle': {'deterministic': True, 'deterministic_overrides': ov,
                                                                    'resumed_bitwise_updates': 2}})
    block, _ = sup.lifecycle_block()
    assert block['deterministic_overrides'] == ov and block['resumed_bitwise_updates'] == 2 and block['deterministic']
