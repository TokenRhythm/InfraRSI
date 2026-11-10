"""Dispatcher-level simulation of the F1 fallback for the external-bench arm tests (F-PROMPTS, F-OPTICO; builder X4).

D's LocalRemote with the dummy entry; S1 stands in for F1's 32 GPUs: the start slot holds a 'U+C' configuration
recorded as the task's numerics reference, FR not applicable, as on F1. Step time follows MBS and recompute (as
tests/cpu/agent/test_dispatch_arms), and the unfused attention backend costs 50 %; scalars do not depend on the config
(numerics E0).
"""
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

from infrarsi.agent import arms
from infrarsi.contracts import RunRequest, TaskSpec
from infrarsi.discover import workload
from infrarsi.dispatch import QueueStore
from infrarsi.dispatch.handlers import Context, handlers
from infrarsi.harness import leases, run_screen
from infrarsi.harness.remote import LocalRemote
from infrarsi.planner import FRChain

_DH_PATH = Path(__file__).resolve().parents[1] / 'driver' / 'driver_helpers.py'
_spec = importlib.util.spec_from_file_location('driver_helpers_for_f1_world', _DH_PATH)
DH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DH)


def sim(req, out):
    cfg = req['config']
    t = 0.5 / (1 + 0.15 * (int(cfg.get('micro_batch_size') or 1) - 1))
    if cfg.get('recompute_granularity') == 'full':
        t *= 1.3
    if cfg.get('attention_backend') == 'unfused':
        t *= 1.5
    DH.DUMMY_MOD.simulate_attempt(req, out, step_time=lambda r, u, _t=t: _t + 0.001 * r)


def world_setup(tmp_path, monkeypatch, log=None):
    """Generator for a pytest fixture: yields {'remote', 'ctx', 'uc', 'uc_a', 'uc_b'}; removes the remote root after."""
    root = Path(tempfile.mkdtemp(prefix='ifp'))
    remote = LocalRemote(root, launcher=DH.sim_launcher(sim))
    DH.fake_index(remote)
    monkeypatch.setattr(workload, 'CACHE', DH.HF)
    store = QueueStore(tmp_path / 'queue')

    def runner(req, **kw):
        return run_screen(req, remote=remote, poll_s=0.01, local_dir=None, log=lambda m: None, **kw)
    task = TaskSpec.load('S1')
    uc = FRChain(task, workload.workload_facts(task, cache_dir=DH.HF)).current()       # plays U+C
    kw = dict(run_id='XB-F1-UC', task=task.raw, config=uc, nodes=['WINH0'], gpus_per_node=1, numerics_role='reference')
    uc_a = runner(RunRequest(purpose='screen', input_plan='A', **kw))
    uc_b = runner(RunRequest(purpose='score', input_plan='B', **kw))
    assert uc_a['valid'] and uc_b['valid'], (uc_a['invalid_reasons'], uc_b['invalid_reasons'])
    store.set_reference('S1', fr_status='not_applicable', ed_status='feasible', ed_config=uc,
                        ed_screen=uc_a['harness']['stored'], ed_score=uc_b['harness']['stored'], ed_numerics_role='reference',
                        numerics_reference='ED')
    ctx = Context(store=store, remote=remote, runner=runner, searches_root=tmp_path / 'searches', log=log or (lambda m: None))
    yield {'remote': remote, 'ctx': ctx, 'uc': uc, 'uc_a': uc_a, 'uc_b': uc_b}
    shutil.rmtree(root, ignore_errors=True)


def dispatch(world, backend, monkeypatch, run):
    """The registered handler of run['arm'] inside a run-scoped lease, with `backend` as the strong driver."""
    monkeypatch.setitem(arms.DRIVERS, 'strong', lambda: backend)
    r = world['remote']
    leases.acquire(r, ['WINH0'], leases.make_lease(run['run_id'], 'run:' + run['run_id'], ['WINH0'], purpose='dispatch', scope='run'))
    try:
        return handlers()[run['arm']](run, ['WINH0'], world['ctx'])
    finally:
        leases.release(r, ['WINH0'], 'run:' + run['run_id'])


def requests(world, run_id):
    d = world['remote'].local(f'/data/infrarsi/runs/{run_id}')
    return sorted(((q['purpose'], q['input_plan'], q['numerics_role']) for q in
                   (json.loads(p.read_text()) for p in d.glob('*/request.json'))), key=str) if d.exists() else []


def events(out_dir):
    return [json.loads(x) for x in (Path(out_dir) / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
