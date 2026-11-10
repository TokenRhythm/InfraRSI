"""preflight.check_node on canned node answers: legacy host locks (present / missing, formal launcher running),
occupancy, lease, venv link and the read-only hashes (AMENDMENT_EXT_003 G-F1 (4))."""
import pytest

from infrarsi.ext.flashnext import deploy as D
from infrarsi.ext.flashnext import preflight as PF
from infrarsi.ext.flashnext import runner as R


class CannedOps:
    def __init__(self, node, *, legacy='OK', launcher_running=False, apps=(), lease=None, skew=0.0):
        self.node, self.legacy, self.launcher_running, self.apps, self.lease = node, legacy, launcher_running, apps, lease
        self.skew = skew
        self.commands = []

    def probe(self):
        return {'hostname': R.HOSTNAMES[R.node_index(self.node)], 'error': None, 'apps': list(self.apps),
                'gpus': [{'index': g, 'mem_used_mib': 0} for g in range(8)]}

    def read_json(self, path, default=None):
        return self.lease

    def sh(self, cmd, timeout=120):
        self.commands.append(cmd)
        if cmd.startswith('date '):
            import time
            return {'code': 0, 'stdout': '%.6f\n' % (time.time() + self.skew), 'stderr': ''}
        if 'legacy' in cmd or 'LEGACY_LAUNCHER_RUNNING' in cmd:
            host = R.node_index(self.node) + 1
            lines = [f'{self.legacy} {d}/host{host:02d}.lock' for d in R.LEGACY_LOCK_DIRS]
            if self.launcher_running:
                lines.append('LEGACY_LAUNCHER_RUNNING')
            return {'code': 0, 'stdout': '\n'.join(lines) + '\n', 'stderr': ''}
        if 'readlink' in cmd:
            return {'code': 0, 'stdout': '/data/sftv2aliyun/env/qwen-sft\nVENV_OK\n', 'stderr': ''}
        return {'code': 0, 'stdout': 'cuda inductor triton\n', 'stderr': ''}

    def hash_files(self, paths):
        return {p: D.REMOTE_READ_ONLY.get(p) for p in paths}


def test_clean_node_passes():
    rep = PF.check_node(CannedOps('WINH3'), 'WINH3')
    assert rep['ok'] is True and rep['legacy_locks_ok'] is True and rep['read_only_hashes_ok'] is True, rep['problems']


@pytest.mark.parametrize('kw,needle', [
    ({'legacy': 'MISSING'}, 'legacy host lock files missing'),
    ({'launcher_running': True}, 'formal-queue launcher'),
    ({'apps': [{'pid': 7, 'gpu_index': 0, 'cwd': '/data/tyc/x'}]}, 'compute processes'),
    ({'lease': {'owner': 'infrarsi-ext', 'run_id': 'x1'}}, 'node lease held')])
def test_findings_make_the_node_not_ok(kw, needle):
    rep = PF.check_node(CannedOps('WINH6', **kw), 'WINH6')
    assert rep['ok'] is False and any(needle in p for p in rep['problems']), rep['problems']


def test_the_legacy_check_names_the_nodes_own_host_files():
    ops = CannedOps('WINH1')
    PF.check_node(ops, 'WINH1')
    cmd = next(c for c in ops.commands if 'LEGACY_LAUNCHER_RUNNING' in c)
    assert all(f'{d}/host02.lock' in cmd for d in R.LEGACY_LOCK_DIRS) and '[l]aunch_lp_node[.]py' in cmd


def test_a_node_whose_clock_is_off_is_flagged():
    nodes = {'WINH0': 0.0, 'WINH1': 0.3, 'WINH3': -0.2, 'WINH4': 30.0}
    rep = PF.preflight({n: CannedOps(n, skew=s) for n, s in nodes.items()}, list(nodes))
    assert rep['ok_nodes'] == ['WINH0', 'WINH1', 'WINH3'] and rep['clock_spread_s'] == pytest.approx(30.2, abs=0.5)
    assert any('clock' in p for p in rep['nodes']['WINH4']['problems'])


def test_an_unmeasured_clock_fails_the_node():
    class NoDate(CannedOps):
        def sh(self, cmd, timeout=120):
            if cmd.startswith('date '):
                return {'code': 127, 'stdout': '', 'stderr': 'date: not found'}
            return super().sh(cmd, timeout)
    rep = PF.check_node(NoDate('WINH0'), 'WINH0')
    assert rep['ok'] is False and any('clock not measured' in p for p in rep['problems'])
