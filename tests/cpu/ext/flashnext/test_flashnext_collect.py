"""Collection of large screen outputs (incident 2026-10-21 19:48, f1_prompts_20261021b c1: runner_error JSONDecodeError,
8 of 32 rank signal logs collected).

The real WinhNodeOps call path runs over a stand-in for harness.remote.WinhRemote that has its exit-status race:
WinhRemote._exec (harness/remote.py:118) stops reading when the exit status has arrived and nothing is buffered, so
an exec reply longer than the SSH window (paramiko DEFAULT_WINDOW_SIZE, 2 MiB) can lose its last MBs, which were still
waiting for window space when the exit status overtook them. SFTP (get_bytes) is request / response and exact. The
stand-in runs the runner's node-side collection script (runner.COLLECT_PY) with this Python on mapped paths.

Real-size fixture: 32 rank signal logs of 30 train_step rows x ~145 KB (4.35 MB each, the c1 sizes; ~140 MB) that
compress about as the real ones do (lzma ~86x)."""
import base64
import hashlib
import json
import random
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from infrarsi.ext.common import node_ext
from infrarsi.ext.common.remote_exec import LocalNodeOps, NodeError, WinhNodeOps
from infrarsi.ext.flashnext import runner as R

WINDOW = 2 * 1024 * 1024                                  # paramiko DEFAULT_WINDOW_SIZE
ROW_BYTES = 145_000
NAME = 'flash32_joint_tp8_p1_lp_rsi_unified_s_match_a_00000000_atest'
META = f'/data/infrarsi/ext/runs/flashnext/metadata/{NAME}'
OUT = f'/data/infrarsi/ext/runs/flashnext/runs/{NAME}'
ATTEMPT = {'run': NAME, 'attempt_id': 'atest', 'lifecycle_run_root': OUT + '/training'}
JOB = {'name': NAME, 'metadata_directory': META, 'output_root': OUT}


class RacyRemote:
    """WinhRemote's interface over a local root. An exec reply longer than the window loses its last 1-5 MB (the exit
    status overtaking the stdout still queued behind the window); get_bytes (SFTP) is exact. `damage` replies of the
    collection script get one base64 character of their first blob changed."""

    def __init__(self, root: Path, damage: int = 0, damage_index: int = 0):
        self.local = LocalNodeOps(root, 'WINH0')
        self.replies, self.sftp_reads, self.damage, self.damage_index = [], 0, damage, damage_index
        self.rng = random.Random(7)

    def _run_collect(self, cmd: str) -> dict:
        _, _, script, arg = shlex.split(cmd)
        req = json.loads(base64.b64decode(arg))
        back = {}
        for item in req['files']:
            back[str(self.local.local(item[0]))] = item[0]
            item[0] = str(self.local.local(item[0]))
        p = subprocess.run([sys.executable, '-c', script, '-'], input=base64.b64encode(json.dumps(req).encode()).decode(),
                           capture_output=True, text=True)        # stdin: Windows caps a command line at 32 K
        out = p.stdout
        if R.COLLECT_MARK in out:
            body = json.loads(out.split(R.COLLECT_MARK, 1)[1])
            for e in body['files']:
                e['path'] = back[e['path']]
            k = self.damage_index
            if self.damage > 0 and len(body['files']) > k and body['files'][k].get('xz'):
                self.damage -= 1
                b = body['files'][k]['xz']
                body['files'][k]['xz'] = b[:40] + ('A' if b[40] != 'A' else 'B') + b[41:]
            out = R.COLLECT_MARK + json.dumps(body) + '\n'
        return {'code': p.returncode, 'stdout': out, 'stderr': p.stderr}

    def sh(self, node, cmd, timeout=120):
        parts = shlex.split(cmd)
        if parts[:1] != ['echo']:                         # mkdir / rm of the request directory
            return {'code': 0, 'stdout': '', 'stderr': ''}
        arg = parts[-1]
        req = json.loads(self.local.local(arg[1:]).read_text() if arg.startswith('@') else base64.b64decode(arg))
        try:
            if req['func'] == 'sh' and req['kw']['cmd'].startswith('python3 -c '):
                res = {'ok': True, 'value': self._run_collect(req['kw']['cmd'])}
            else:
                res = {'ok': True, 'value': self.local.call(req['func'], **req['kw'])}
        except Exception as exc:                          # noqa: BLE001 - reported like node_ext._main
            res = {'ok': False, 'error': repr(exc), 'traceback': ''}
        out = '\n' + node_ext.SENTINEL + json.dumps(res) + '\n'
        self.replies.append(len(out))
        if len(out) > WINDOW:
            out = out[:len(out) - self.rng.randint(1 << 20, 5 << 20)]
        return {'code': 0, 'stdout': out, 'stderr': ''}

    def put_bytes(self, node, data, path):
        p = self.local.local(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get_bytes(self, node, path):
        self.sftp_reads += 1
        p = self.local.local(path)
        return p.read_bytes() if p.is_file() else None


def _row(rank: int, step: int, rng: random.Random) -> bytes:
    noise = ''.join('%032x' % rng.getrandbits(128) for _ in range(105))   # ~1.7 KB of entropy: ~50 KB per log in lzma
    pad = hashlib.sha256(f'{rank}:{step}'.encode()).hexdigest()
    pad = (pad * (ROW_BYTES // len(pad) + 1))[:ROW_BYTES - len(noise)]
    return json.dumps({'attempt_id': 'atest', 'completed_updates': step, 'kind': 'train_step', 'rank': rank,
                       'run_id': NAME, 'schema': 'flash-strategy-signals-v1', 'sequence': step,
                       'time_utc': '2026-10-21T11:30:00+00:00', 'values': {'main_ce': 1.0 + step / 1000.0,
                                                                           'noise': noise, 'pad': pad}},
                      separators=(',', ':')).encode() + b'\n'


@pytest.fixture(scope='module')
def screen(tmp_path_factory):
    """A screen's outputs on a simulated /data at real size; returns (root, layout, {remote path: sha256})."""
    root = tmp_path_factory.mktemp('collect').resolve()
    if sys.platform == 'win32':
        root = Path('\\\\?\\' + str(root))                # mirrored /data paths exceed MAX_PATH
    rng = random.Random(11)
    lay = R.layout_from_records(ATTEMPT, JOB)
    files = {}
    for r, path in enumerate(lay['signals']):
        files[path] = b''.join(_row(r, s, rng) for s in range(1, 31))
    for r, path in enumerate(lay['rank_receipts']):
        files[path] = json.dumps({'rank': r, 'status': 'ok'}).encode()
    for n in range(4):
        files[f'{OUT}/node{n}.exit.json'] = json.dumps({'exit_code': 0, 'node': n}).encode()
        files[f'{OUT}/node{n}.pid.json'] = json.dumps({'pid': 4000 + n}).encode()
        files[f'{OUT}/node{n}.release.json'] = json.dumps({'idle': True}).encode()
        files[f'{OUT}/node{n}.log'] = b''.join(b'%07d training log line of node %d\n' % (i, n) for i in range(40_000))
        files[f'{META}/fn_node{n}.launch.log'] = b'launched\n'
    files.update({f'{META}/attempt.json': json.dumps(ATTEMPT).encode(), f'{META}/job.json': json.dumps(JOB).encode(),
                  f'{META}/fn_spec.json': b'{"nodes": ["WINH0", "WINH1", "WINH4", "WINH6"]}',
                  f'{OUT}/family_materialization/materialization.json': b'{"ok": true}'})
    for path, data in files.items():
        p = root / path.lstrip('/')
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return root, lay, {p: hashlib.sha256(d).hexdigest() for p, d in files.items()}


def _runner(remote, tmp):
    nodes = ('WINH0', 'WINH1', 'WINH4', 'WINH6')
    ops = {n: WinhNodeOps(n, remote=remote) for n in nodes}
    return R.FlashNextRunner(ops, nodes=nodes, runtime_dir=str(tmp / 'rt'), out_dir=tmp / 'out')


def _check_exact(root, lay, sha, local):
    wanted, logs = R.FlashNextRunner.output_files(lay)
    for path, name in wanted.items():                    # every file byte-identical to the node's
        assert hashlib.sha256((local / name).read_bytes()).hexdigest() == sha[path], name
    for path, name in logs.items():                      # logs: the last 256 KiB, as before
        assert (local / name).read_bytes() == (root / path.lstrip('/')).read_bytes()[-R.LOG_TAIL_BYTES:], name
    assert len(list((local / 'signals').glob('rank*.jsonl'))) == 32
    assert sum((local / 'signals' / f'rank{k:02d}.jsonl').stat().st_size for k in range(32)) > 130 << 20


def test_the_old_batched_read_breaks_on_a_racy_large_reply(screen):
    """The pre-fix path (runner._collect: read_files of 8 paths per call) as a >= 50 MB exec reply: truncated JSON."""
    root, lay, _ = screen
    remote = RacyRemote(root)
    ops = WinhNodeOps('WINH0', remote=remote)
    with pytest.raises((json.JSONDecodeError, NodeError)):
        ops.read_files(lay['signals'][:12], max_bytes=32 << 20)
    assert remote.replies[-1] >= 50 << 20                # ~71 MB of base64 JSON for 12 real-size rank logs


def test_collection_is_exact_at_real_size_over_the_racy_transport(screen, tmp_path):
    root, lay, sha = screen
    remote = RacyRemote(root)
    r = _runner(remote, tmp_path)
    local = tmp_path / 'collected'
    rec = r.fetch_outputs(lay, local)
    _check_exact(root, lay, sha, local)
    assert rec['method'] == 'node_lzma' and rec['sftp_files'] == 0 and remote.sftp_reads == 0
    assert 2 <= rec['calls'] <= 6                        # ~140 MB in a few compressed replies
    assert max(remote.replies) < WINDOW                  # every exec reply below the window: none can be cut
    assert rec['files']['signals/rank31.jsonl']['sha256'] == sha[lay['signals'][31]]
    assert rec['files']['nodes/node0.log']['tail_bytes'] == R.LOG_TAIL_BYTES


def test_a_file_too_large_for_one_reply_comes_over_sftp(screen, tmp_path):
    root, lay, sha = screen
    big = lay['signals'][5]
    p = root / big.lstrip('/')
    keep = p.read_bytes()
    try:
        p.write_bytes(base64.b64encode(random.Random(3).randbytes(3 << 20)))     # ~4 MB that lzma cannot shrink
        remote = RacyRemote(root)
        rec = _runner(remote, tmp_path).fetch_outputs(lay, tmp_path / 'c')
        assert rec['sftp_files'] == 1 and remote.sftp_reads == 1
        assert (tmp_path / 'c/signals/rank05.jsonl').read_bytes() == p.read_bytes()
        assert max(remote.replies) < WINDOW
    finally:
        p.write_bytes(keep)


def test_a_damaged_reply_is_asked_again_and_a_persistent_one_fails(screen, tmp_path):
    root, lay, sha = screen
    r = _runner(RacyRemote(root, damage=2), tmp_path)
    r.fetch_outputs(lay, tmp_path / 'a')                 # two damaged replies, each asked again
    _check_exact(root, lay, sha, tmp_path / 'a')
    r = _runner(RacyRemote(root, damage=10 ** 6), tmp_path)
    with pytest.raises(RuntimeError, match='failed replies in a row'):
        r.fetch_outputs(lay, tmp_path / 'b')


def test_layout_from_records_equals_the_bundle_layout():
    lay = R.layout_from_records(ATTEMPT, JOB)
    assert lay['signals'][7] == f'{OUT}/training/{NAME}.signals.rank7/signals.jsonl'
    assert lay['rank_receipts'][31] == f'{OUT}/training/{NAME}.rank31.json' and lay['job'] == META + '/job.json'


def test_a_file_missing_on_the_node_leaves_no_stale_local_copy(screen, tmp_path):
    """Codex review (HIGH): re-collecting into a directory that holds an older copy of a file the node now reports
    missing must not leave that copy for parse.collect to judge."""
    root, lay, sha = screen
    gone = root / lay['signals'][3].lstrip('/')
    keep = gone.read_bytes()
    local = tmp_path / 'again'
    (local / 'signals').mkdir(parents=True)
    (local / 'signals' / 'rank03.jsonl').write_bytes(b'stale copy of an earlier collection')
    gone.unlink()
    try:
        rec = _runner(RacyRemote(root), tmp_path).fetch_outputs(lay, local)
    finally:
        gone.write_bytes(keep)
    assert rec['files']['signals/rank03.jsonl'] is None and not (local / 'signals/rank03.jsonl').exists()
    assert len(list((local / 'signals').glob('rank*.jsonl'))) == 31


def test_progress_resets_the_failure_count(screen, tmp_path):
    """Codex review (MEDIUM): replies that each verify some files and then fail on a later one still finish."""
    root, lay, sha = screen
    remote = RacyRemote(root, damage=5, damage_index=4)            # five replies in a row fail at their 5th file
    rec = _runner(remote, tmp_path).fetch_outputs(lay, tmp_path / 'p')
    _check_exact(root, lay, sha, tmp_path / 'p')
    assert rec['calls'] >= 6


def test_each_file_is_recorded_with_its_node_mtime(screen, tmp_path):
    root, lay, sha = screen
    rec = _runner(RacyRemote(root), tmp_path).fetch_outputs(lay, tmp_path / 'm')
    f = rec['files']['signals/rank00.jsonl']
    assert f['remote'] == lay['signals'][0] and f['sha256'] == sha[lay['signals'][0]]
    assert abs(f['mtime'] - (root / lay['signals'][0].lstrip('/')).stat().st_mtime) < 1e-3
