"""A10 / A11 in the flow (textbook designs P2-2 + P1-3b2, generic phases; 2026-10-24): the measured all-reduce bandwidth
fills the interconnect, buckets the experience scope and gives the accounting its data-parallel sync bound; a job whose
log shows no phases gets them from one profiled run (kept out of the signed receipt: result.reference.phases and the
digest's fallback)."""
from types import SimpleNamespace

import pytest

from infrarsi.experience.accounting import estimate
from infrarsi.experience.store import Store, _same_band, band
from infrarsi.project import Project
from infrarsi.project import signals as sig
from infrarsi.project.flow import apply_links, profile
from infrarsi.project.hooks import ProjectHooks

from test_project import toy  # noqa: F401 - the toy project fixture

LLAMA8B = {'params_b': 8.03, 'layers': 32, 'hidden': 4096, 'ffn': 14336, 'heads': 32, 'kv_heads': 8, 'vocab': 128256}
TWO = {'gpu': 'H100-SXM-80GB', 'memory_gib': 79.6, 'framework': 'megatron', 'gpus_per_node': 8, 'nodes': 2,
       'model': LLAMA8B, 'workload': {'seq_len': 4096, 'global_batch_tokens': 524288}}
REF = {'tp': '1', 'pp': '1', 'micro_batch': '1', 'precision': 'bf16', 'distributed_optimizer': 'on', 'bf16_main_grads': 'fp32'}


def test_bandwidth_buckets_scope_multi_node_knowledge():
    assert [band(x) for x in (None, 0, 2.8, 12, 41.5, 180)] == [None, None, '<5', '5-25', '25-100', '>=100']
    ib, tcp, slow = ({'interconnect': {'inter': 'tcp' if b < 100 else 'ib', 'bandwidth_gbs': b}} for b in (180, 12, 2.8))
    assert _same_band(tcp, {'interconnect': {'inter': 'tcp', 'bandwidth_gbs': 20}}) and not _same_band(tcp, slow)
    assert _same_band(tcp, {'interconnect': {'inter': 'tcp'}})                  # not measured: no claim
    env = {'framework': 'megatron', 'gpu': 'H100-SXM-80GB', 'nodes': 2, 'gpus_per_node': 8, 'model': {'params_b': 8.0},
           'workload': {'seq_len': 4096}, 'interconnect': {'inter': 'tcp', 'bandwidth_gbs': 2.8}}
    fp = {**TWO, 'interconnect': {'inter': 'tcp', 'bandwidth_gbs': 12}, 'layout': {}}
    assert Store.scope_level(fp, env, 'balance') == 'partial'                   # H20-like TCP is not unicom's tuned TCP
    assert Store.scope_level({**fp, 'interconnect': {'inter': 'tcp', 'bandwidth_gbs': 3.5}}, env, 'balance') == 'covered'


def test_the_accounting_bounds_the_data_parallel_gradient_sync():
    assert estimate(TWO, REF)['dp_sync_s'] is None                              # no bandwidth: no number
    est = estimate({**TWO, 'interconnect': {'inter': 'ib', 'bandwidth_gbs': 100.0}}, REF)
    w = est['params_local_b'] * 1e9
    assert est['dp_sync_s'] == pytest.approx(w * (4.0 + 2.0) * 15 / 16 / 100e9, rel=1e-3)   # RS fp32 grads + AG bf16
    one = estimate({**TWO, 'nodes': 1, 'interconnect': {'bandwidth_gbs': 100.0}}, REF)
    assert one['dp_sync_s'] is None                                             # single node: NVLink, not the link
    fsdp = {**TWO, 'framework': 'fsdp', 'interconnect': {'inter': 'ib', 'bandwidth_gbs': 100.0}}
    twice, kept = estimate(fsdp, {})['dp_sync_s'], estimate(fsdp, {'reshard_after_forward': 'off'})['dp_sync_s']
    assert kept == pytest.approx(twice * (2 + 2) / (4 + 2), rel=1e-3)          # one weight all-gather, not two
    assert estimate(fsdp, {'fsdp_sharding': 'hybrid_shard'})['dp_sync_s'] is None   # not modelled: no number


def test_apply_links_adds_a_measured_bandwidth_to_the_manifests_fabric(tmp_path):
    logs = []
    p = Project({'name': 'x', 'launch': {'command': 'true'}, 'hardware': {'interconnect': 'ib'}}, tmp_path / 'infrarsi.yaml')
    apply_links(p, SimpleNamespace(link_facts=lambda: {'inter': 'ib', 'bandwidth_gbs': 41.5}), logs.append)
    assert p.raw['hardware']['interconnect'] == {'inter': 'ib', 'bandwidth_gbs': 41.5} and 'measured' in logs[-1]
    q = Project({'name': 'x', 'launch': {'command': 'true'}, 'hardware': {'interconnect': {'inter': 'tcp', 'bandwidth_gbs': 9}}},
                tmp_path / 'infrarsi.yaml')
    apply_links(q, SimpleNamespace(link_facts=lambda: {'inter': 'tcp', 'bandwidth_gbs': 41.5}), logs.append)
    assert q.raw['hardware']['interconnect']['bandwidth_gbs'] == 9                # the manifest's number wins


PROFILE = {'unit': 'ms', 'updates': 3, 'mean_ms': {'compute': 20.0, 'comm': 1.0, 'optimizer': 1.0, 'idle': 2.0, 'update': 24.0}}


class Runner:
    def __init__(self, out, ok=True):
        self.out, self.ok, self.calls = out, ok, []

    def can_profile(self):
        return self.ok

    def run(self, task, config, overlay, purpose, run_id, **kw):
        self.calls.append((purpose, kw))
        return {'cost': {'gpu_seconds': 36.0}, 'evidence': {'profile': self.out}}


def test_a_job_without_logged_phases_gets_them_from_one_profiled_run():
    r = Runner(PROFILE)
    ph, gh = profile(r, None, {}, {'evidence': {}}, 'r', lambda m: None, [])
    assert ph == sig.profile_phases(PROFILE) and ph['top'] == ['compute', 'comm', 'optimizer'] and gh == 0.01
    assert ph['share']['compute'] == pytest.approx(20 / 24, abs=1e-4) and ph['residual_ms'] == 2.0 and ph['source'] == 'profile'
    assert r.calls[0][0] == 'profile' and r.calls[0][1]['profile'] and not r.calls[0][1]['bench']
    assert profile(Runner(PROFILE), None, {}, {'evidence': {'phases': {'x': 1}}}, 'r', print, []) == (None, 0.0)
    assert profile(Runner(PROFILE), None, {}, {'evidence': {}}, 'r', print, ['profile']) == (None, 0.0)
    assert profile(Runner(PROFILE, ok=False), None, {}, {'evidence': {}}, 'r', print, []) == (None, 0.0)
    logs = []
    assert profile(Runner({'error': 'no update boundaries'}), None, {}, {'evidence': {}}, 'r', logs.append, [])[0] is None
    assert 'no update boundaries' in logs[-1]


def test_the_digest_falls_back_to_the_reference_profile(toy):
    hooks = ProjectHooks(toy, toy.task())
    receipt = {'attempt_id': 'a1', 'valid': True, 'evidence': {}}
    assert 'step_phases' not in hooks.receipt_digest(receipt, {'valid': True})
    hooks.reference_phases = sig.profile_phases(PROFILE)
    d = hooks.receipt_digest(receipt, {'valid': True})
    assert d['step_phases']['of'].startswith('the reference configuration') and 'phase_targets' in d
    assert hooks.class_shares(receipt)['residual'] == pytest.approx(2 / 24, abs=1e-4)     # the profile's idle time: a bound
