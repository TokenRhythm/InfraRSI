"""envc_crnd_d (O, urgent): under a group lease the per-attempt occupancy check considers only the attempt's own GPUs;
processes of the co-located ENV-C arm uids never count as foreign elsewhere; a refusal caused by a sibling arm on the
attempt's own GPU is `occupied_sibling` and does not count toward the same-failure stop."""
import json

import pytest

from infrarsi.ext.common.env_judge import EnvJudge
from infrarsi.ext.common.remote_exec import GpuLease, LocalNodeOps, node_lease_path, occupancy_problems
from infrarsi.ext.common.session_hooks import SessionHooks
from infrarsi.agent.search import SearchSession
from infrarsi.harness.leases import OccupiedError
from x1_common_helpers import BASE_FILES, ToyRunner, toy_task

ENVC = tuple(range(64000, 64008))
GROUP = {'attempt_id': 'run:envc_grp_d', 'driver_host': 'PC-20251205NBNV', 'driver_pid': 28116, 'gpus': [0, 2, 4, 6],
         'nodes': ['WINH0'], 'owner': 'infrarsi-ext', 'purpose': 'ext:autoresearch', 'run_id': 'envc_grp_d',
         'schema': 'infrarsi-lease-v1', 'scope': 'run', 'since': '2026-10-21T02:53:07.852877+00:00'}


def probe(gpu=4, uid=64002):
    """envc_crnd_d's refused attempts: the co-located C-NAT-infra agent's `python prof.py` on GPU 4 (root cannot read
    its environ, so token and cwd are null)."""
    return {'hostname': 'is-ddkt5zq62vhpqy3a-devmachine-0', 'utc': 'x', 'error': None,
            'gpus': [{'index': i, 'uuid': f'GPU-{i}', 'mem_used_mib': 53304 if i == gpu else 0} for i in range(8)],
            'apps': [{'pid': 2080133, 'gpu_index': gpu, 'used_mib': 53304, 'cwd': None, 'cmdline': 'python prof.py',
                      'token': None, 'uid': uid}]}


def test_the_envc_crnd_d_probe_no_longer_blocks_c_rnd():
    ok = lambda lease: lease.get('run_id') == 'envc_grp_d'                 # noqa: E731
    old = occupancy_problems(probe(), [0], lease=GROUP, lease_ok=ok)
    assert old == ['1 foreign compute processes [2080133]']               # the refusal of envc_crnd_d c3-c8
    assert occupancy_problems(probe(), [0], lease=GROUP, lease_ok=ok, own_gpus_only=True, sibling_uids=ENVC) == []
    assert occupancy_problems(probe(), [0], lease=GROUP, lease_ok=ok, sibling_uids=ENVC) == []   # sibling elsewhere
    # a process of another uid on a sibling GPU: ignored under the group lease, foreign for an exclusive (own) lease
    assert occupancy_problems(probe(uid=1000), [0], lease=GROUP, lease_ok=ok, own_gpus_only=True, sibling_uids=ENVC) == []
    assert occupancy_problems(probe(uid=1000), [0], lease=GROUP, lease_ok=ok, sibling_uids=ENVC) == [
        '1 foreign compute processes [2080133]']
    # anything on the attempt's own GPU still blocks, a sibling's process included
    mine = occupancy_problems(probe(gpu=0), [0], lease=GROUP, lease_ok=ok, own_gpus_only=True, sibling_uids=ENVC)
    assert 'compute processes on target GPUs: [(2080133, 0)]' in mine


def _node(tmp_path, p):
    ops = LocalNodeOps(tmp_path, node='WINH0', probe_fn=lambda: p)
    path = ops.local(node_lease_path('WINH0'))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(GROUP), encoding='utf-8')
    return ops


def test_held_lease_acquires_next_to_a_profiling_sibling(tmp_path):
    ops = _node(tmp_path, probe())
    lease = GpuLease(ops, env='autoresearch', gpus=[0], mode='held', held_run_id='envc_grp_d', sibling_uids=ENVC)
    lease.acquire('envc_crnd_e', 'a1')
    assert lease.report['problems'] == [] and lease.release()
    ops = _node(tmp_path / 'b', probe(gpu=0))                              # the sibling on C-RND's own GPU
    lease = GpuLease(ops, env='autoresearch', gpus=[0], mode='held', held_run_id='envc_grp_d', sibling_uids=ENVC)
    with pytest.raises(OccupiedError):
        lease.acquire('envc_crnd_e', 'a2')
    assert lease.report['sibling_only'] is True
    ops = _node(tmp_path / 'c', probe(gpu=0, uid=1000))                    # someone else on it: a real occupation
    lease = GpuLease(ops, env='autoresearch', gpus=[0], mode='held', held_run_id='envc_grp_d', sibling_uids=ENVC)
    with pytest.raises(OccupiedError):
        lease.acquire('envc_crnd_e', 'a3')
    assert lease.report['sibling_only'] is False


class SiblingToy(ToyRunner):
    sibling_uids = ENVC


def _session(tmp_path, p):
    task = toy_task(budget={'screens': 20, 'llm_tokens': 0, 'wall_seconds': 36000.0, 'gpu_hours': 10.0})
    ops = _node(tmp_path, p)
    r = SiblingToy(ops, lease_mode='held', held_run_id='envc_grp_d')
    s = SearchSession(task=task, budget=task.search_budget(), runner=r, judge=EnvJudge('toyenv'), run_id='r', arm='T-RND',
                      start_config={}, layers=('L0',), clock=r.vclock, hooks=SessionHooks(task, base_files=BASE_FILES))
    s.screen('c0')
    for i, w in enumerate((2, 4, 8, 1, 2, 4)):
        if s.stop_flag:
            break
        s.screen(s.propose_config(config={'workers': w, 'flash': i > 2})['candidate_id'])
    return s


def test_occupied_sibling_is_not_a_failure_of_the_search(tmp_path):
    s = _session(tmp_path, probe(gpu=0))                                  # 7 refusals by a sibling arm on GPU 0
    assert len(s.screens) == 7 and s.stop_flag is None and s.same_failure[1] == 0          # no same_failure_x6 stop
    assert all(x['reasons'][0].startswith('occupied_sibling:') for x in s.screens)
    s = _session(tmp_path / 'b', probe(gpu=0, uid=1000))                  # a foreign process: counted as before
    assert s.stop_flag == 'same_failure_x6' and len(s.screens) == 6
    assert all(x['reasons'][0].startswith('occupied:') for x in s.screens)
