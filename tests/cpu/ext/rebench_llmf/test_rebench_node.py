"""Node-side scorer and sanitizer helpers that are pure (no torch): the per-arm uid commands and kill loop (B-02), the
root-side model-dir checks and the safe compare seam (B-01), the tree digest / snapshot restore (B-05), component
containment and canonical paths (H-03), G8 symlink_escape. Both modules import torch only inside functions, so
importing them here is torch-free. POSIX-only pieces (dir_fd / O_NOFOLLOW installs, chown) are exercised on the
node by the Follow-up 2 probes."""
import io
import json
import os
import struct
import subprocess
import tarfile
from pathlib import Path

import pytest

from infrarsi.ext.rebench_llmf.node import sanitize as sz
from infrarsi.ext.rebench_llmf.node import score_runner as sr

POSIX = os.name == 'posix'


# ---------------------------------------------------------------- B-02: uids
def test_uid_command_drops_to_the_arm_uid_with_a_clean_env():
    env = sr.uid_env('/data/infrarsi/ext/envs/rebench_llmf/bin', '0,1,2,3', Path('/run/home_root'), 'isx2-0123456789ab', 64102)
    cmd = sr.uid_command('cd /run/home_agent/solution && ./finetune.sh', env, 64102)
    assert 'setpriv --reuid=64102 --regid=64102 --clear-groups --no-new-privs' in cmd
    assert 'env -i' in cmd and 'CUDA_VISIBLE_DEVICES=0,1,2,3' in cmd and 'USER=isx64102' in cmd
    assert 'HF_HUB_OFFLINE=1' in cmd and 'INFRARSI_TOKEN=isx2-0123456789ab' in cmd
    # short private TMPDIR under /tmp: avoids the AF_UNIX 108-byte limit and the root-owned /tmp/streaming
    assert env['TMPDIR'] == '/tmp/isx2-0123456789ab-u64102' and len(env['TMPDIR']) < 40


@pytest.mark.parametrize('uid', [65534, 0, 1000, 64000, 64100, 64106, 64110, 64116, 60000])
def test_only_env_a_uids_are_accepted(uid):
    with pytest.raises(ValueError):
        sr.check_uid(uid)
    with pytest.raises(ValueError):
        sr.kill_uid(uid)
    with pytest.raises(ValueError):
        sr.uid_env('/b', '', '/h', 't', uid)


def test_kill_uid_kills_by_uid_and_verifies(monkeypatch):
    calls, alive = [], [[11, 12], [12], []]
    monkeypatch.setattr(sr, 'uid_pids', lambda uid: alive.pop(0) if len(alive) > 1 else alive[0])
    monkeypatch.setattr(sr.subprocess, 'run', lambda args, **kw: calls.append(list(args)) or subprocess.CompletedProcess(args, 0))
    monkeypatch.setattr(sr.time, 'sleep', lambda s: None)
    res = sr.kill_uid(64101)
    assert res['clean'] is True
    assert ['pkill', '-9', '-u', '64101'] in calls and ['pkill', '-9', '-U', '64101'] in calls
    assert ['setpriv', '--reuid=64101', '--regid=64101', '--clear-groups', 'kill', '-9', '-1'] in calls
    assert not any('65534' in ' '.join(c) for c in calls)


def test_kill_uid_reports_survivors(monkeypatch):
    monkeypatch.setattr(sr, 'uid_pids', lambda uid: [7])
    monkeypatch.setattr(sr.subprocess, 'run', lambda args, **kw: subprocess.CompletedProcess(args, 0))
    monkeypatch.setattr(sr.time, 'sleep', lambda s: None)
    res = sr.kill_uid(64103, rounds=3)
    assert res['clean'] is False and res['remaining'] == [7]


def test_cleanup_active_kills_every_used_uid(monkeypatch):
    killed, cleaned, removed = [], [], []
    monkeypatch.setattr(sr, 'kill_uid', lambda u: killed.append(u))
    monkeypatch.setattr(sr, 'clean_uid_leftovers', lambda u, tmp=False: cleaned.append(u))
    monkeypatch.setattr(sr, 'remove_tree', lambda p: removed.append(p))
    monkeypatch.setitem(sr._ACTIVE, 'uids', {64102, 64112})
    monkeypatch.setitem(sr._ACTIVE, 'tmpdirs', {'/tmp/isx2-x-u64102'})
    sr._cleanup_active()
    assert killed == [64102, 64112] and cleaned == [64102, 64112] and removed == ['/tmp/isx2-x-u64102']


def test_uid_subprocess_runs_finetune_as_the_uid_and_kills_on_failure(monkeypatch):
    seen, kills = [], []
    monkeypatch.setattr(sr, 'kill_uid', lambda u: kills.append(u))
    env = sr.uid_env('/b', '0', '/h', 'tok', 64102)
    shim = sr.UidSubprocess(env, 64102)
    monkeypatch.setattr(sr.subprocess, 'run', lambda args, **kw: seen.append((args, kw)) or subprocess.CompletedProcess(args, 0))
    assert shim.check_call(['./finetune.sh'], cwd='/run/home_agent/solution', timeout=480) == 0
    assert 'setpriv --reuid=64102' in seen[-1][0][2] and seen[-1][1]['timeout'] == 480 and kills == []
    monkeypatch.setattr(sr.subprocess, 'run', lambda args, **kw: subprocess.CompletedProcess(args, 3))
    with pytest.raises(subprocess.CalledProcessError):
        shim.check_call(['./finetune.sh'], cwd='/s', timeout=480)
    assert kills == [64102]

    def slow(args, **kw):
        raise subprocess.TimeoutExpired(args, kw.get('timeout'))
    monkeypatch.setattr(sr.subprocess, 'run', slow)
    with pytest.raises(subprocess.TimeoutExpired):
        shim.check_call(['./finetune.sh'], cwd='/s', timeout=1)
    assert kills == [64102, 64102]
    assert shim.CalledProcessError is subprocess.CalledProcessError         # score.py's except clause still matches


# ---------------------------------------------------------------- H-03: paths
def test_canonical_rel_and_component_containment(tmp_path):
    assert sr.canonical_rel('scripts/train/train.py')
    for bad in ('scripts/../x', '/x', '~/x', 'a\\b', 'a//b', './a', 'a/.', 'C:/x', '', 'a\x00b'):
        assert not sr.canonical_rel(bad), bad
    sol = tmp_path / 'solution'
    (sol / 'a').mkdir(parents=True)
    (tmp_path / 'solution-evil').mkdir()
    assert sr.is_within(sol / 'a', sol) and sr.is_within(sol, sol)
    assert not sr.is_within(tmp_path / 'solution-evil', sol)                # a string prefix is not containment
    assert not sr.is_within(sol / '..' / 'solution-evil', sol)


# ---------------------------------------------------------------- B-01: root-side model checks (no deserialising)
def _safetensors(tensors: dict, *, corrupt=None) -> bytes:
    """A safetensors byte string built by hand: {name: (dtype, shape, nbytes)}."""
    hdr, off = {'__metadata__': {'format': 'pt'}}, 0
    for name, (dtype, shape, nbytes) in tensors.items():
        hdr[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [off, off + nbytes]}
        off += nbytes
    if corrupt:
        corrupt(hdr)
    raw = json.dumps(hdr).encode()
    return struct.pack('<Q', len(raw)) + raw + b'\x00' * off


def _model_dir(d: Path, *, extra=None, config=None, st=None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / 'model.safetensors').write_bytes(st or _safetensors({'w': ('BF16', [2, 3], 12), 'b': ('F32', [3], 12)}))
    (d / 'config.json').write_text(json.dumps(config or {'model_type': 'llama', 'architectures': ['LlamaForCausalLM']}))
    (d / 'generation_config.json').write_text('{"bos_token_id": 1}')
    for name, data in (extra or {}).items():
        (d / name).write_bytes(data)
    return d


def test_safetensors_header_is_bounds_checked(tmp_path):
    p = tmp_path / 'm.safetensors'
    p.write_bytes(_safetensors({'w': ('F32', [2], 8)}))
    assert sr.read_safetensors_header(p) == {'n_tensors': 1, 'bytes': p.stat().st_size}
    bad = {
        'offsets past the data': lambda h: h['w'].update(data_offsets=[0, 10 ** 9]),
        'size != shape x dtype': lambda h: h['w'].update(shape=[3]),
        'unknown dtype': lambda h: h['w'].update(dtype='OBJECT'),
        'negative dim': lambda h: h['w'].update(shape=[-2]),
    }
    for why, fn in bad.items():
        p.write_bytes(_safetensors({'w': ('F32', [2], 8)}, corrupt=fn))
        with pytest.raises(ValueError):
            sr.read_safetensors_header(p)
    p.write_bytes(b'\xff' * 8 + b'{}')
    with pytest.raises(ValueError):
        sr.read_safetensors_header(p)


def test_validate_model_dir_accepts_only_safetensors_and_clean_json(tmp_path):
    assert sr.validate_model_dir(_model_dir(tmp_path / 'ok'), 'llama')['n_tensors'] == 2
    with pytest.raises(ValueError):                                   # a pickle next to it is not acceptable
        sr.validate_model_dir(_model_dir(tmp_path / 'bin', extra={'pytorch_model.bin': b'\x80\x02}q\x00.'}), 'llama')
    with pytest.raises(ValueError):                                   # custom code
        sr.validate_model_dir(_model_dir(tmp_path / 'am', config={'model_type': 'llama', 'auto_map': {'AutoModel': 'x.Y'}}), 'llama')
    with pytest.raises(ValueError):
        sr.validate_model_dir(_model_dir(tmp_path / 'mt', config={'model_type': 'gpt2'}), 'llama')
    with pytest.raises(ValueError):
        sr.validate_model_dir(_model_dir(tmp_path / 'tok', extra={'tokenizer.json': b'{}'}), 'llama')
    bad_gen = _model_dir(tmp_path / 'gen')
    (bad_gen / 'generation_config.json').write_text('[1, 2]')
    with pytest.raises(ValueError):
        sr.validate_model_dir(bad_gen, 'llama')


def test_root_loader_refuses_any_dir_with_a_pickle(tmp_path):
    sr.check_safe_model_dir(_model_dir(tmp_path / 'ok'))
    for name in ('pytorch_model.bin', 'model.safetensors.index.json', 'training_args.bin'):
        with pytest.raises(ValueError):
            sr.check_safe_model_dir(_model_dir(tmp_path / name.replace('.', '_'), extra={name: b'x'}))
    with pytest.raises(ValueError):
        sr.check_safe_model_dir(tmp_path / 'missing')

    class FakeAuto:
        calls = []

        @classmethod
        def from_pretrained(cls, folder, **kw):
            cls.calls.append((folder, kw))
            return 'model'
    loader = sr.make_safe_loader(type('S', (), {'AutoModel': FakeAuto}))
    assert loader(tmp_path / 'ok') == 'model' and FakeAuto.calls[-1][1] == {'use_safetensors': True}


def test_safe_compare_sanitises_before_any_load_and_maps_failure_to_model_not_found(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(sr, 'kill_uid', lambda u: events.append(('kill', u)) or {'uid': u, 'clean': True})
    monkeypatch.setattr(sr, 'clean_uid_leftovers', lambda u, tmp=False: events.append(('clean', u)))
    monkeypatch.setattr(sr, 'install_copy', lambda src, dst, names: events.append(('install', str(src), str(dst))))
    monkeypatch.setattr(sr, 'remove_tree', lambda p: events.append(('remove', str(p))))
    sol = tmp_path / 'solution'
    (sol / 'trained_model').mkdir(parents=True)
    judged = tmp_path / 'judge' / 'a1_cand' / 'trained_model'

    def original(model_dir, reference_dir):
        events.append(('compare', str(model_dir)))
        if not Path(model_dir).exists():
            raise FileNotFoundError('Trained model not found')
        return {'norm_diff': 3.4, 'max_scalar_diff': 0.001}

    def judge_for(ok):
        j = sr.Judge(uid=64102, env_bin='/b', token='t', solution=sol, models=tmp_path / 'judge', attempt='a1',
                     model_type='llama', sanitize_py='/s.py', score=None)
        res = {'ok': True, 'path': str(judged), 'tree_digest': 'x', 'kind': 'st', 'n_tensors': 2} if ok else \
              {'ok': False, 'reason': "'x' is a dict, not a tensor"}
        monkeypatch.setattr(sr, 'sanitize_to_judge', lambda src, dst, **kw: events.append(('sanitize', str(src), kw['uid'])) or res)
        return j

    judged.mkdir(parents=True)
    j = judge_for(True)
    out = sr.make_safe_compare(j, original)(model_dir=sol / 'trained_model', reference_dir='/ref')
    kinds = [e[0] for e in events]
    assert kinds.index('kill') < kinds.index('sanitize') < kinds.index('install') < kinds.index('compare')
    assert ('compare', str(judged)) in events and out['norm_diff'] == 3.4       # the installed copy is compared
    assert j.g9['ok'] and j.trained == judged and j.g8['ok'] is True
    assert j.original_compare is original
    events.clear()
    j2 = judge_for(False)
    with pytest.raises(FileNotFoundError):                    # score.py catches it: model_not_found, no score
        sr.make_safe_compare(j2, original)(model_dir=sol / 'trained_model', reference_dir='/ref')
    assert j2.g9['ok'] is False and j2.trained is None and not any(e[0] == 'install' for e in events)


# ---------------------------------------------------------------- B-05: tree digest, snapshot restore
def _tree(root: Path) -> Path:
    sol = root / 'solution'
    (sol / 'scripts').mkdir(parents=True)
    (sol / 'finetune.sh').write_text('#!/bin/bash\necho hi\n')
    (sol / 'scripts' / 'a.py').write_text('x = 1\n')
    return sol


def test_tree_digest_ignores_outputs_and_pycache_but_not_content(tmp_path):
    sol = _tree(tmp_path)
    d0 = sr.tree_digest_solution(sol)
    for out in ('trained_model', 'ftout', 'finetuning_tokens'):
        (sol / out).mkdir()
        (sol / out / 'x').write_text('big')
    (sol / 'scripts' / '__pycache__').mkdir()
    (sol / 'scripts' / '__pycache__' / 'a.pyc').write_bytes(b'\x00')
    assert sr.tree_digest_solution(sol) == d0
    (sol / 'scripts' / 'a.py').write_text('x = 2\n')
    assert sr.tree_digest_solution(sol) != d0
    (sol / 'scripts' / 'a.py').write_text('x = 1\n')
    (sol / 'scripts' / 'new.py').write_text('')
    assert sr.tree_digest_solution(sol) != d0
    with pytest.raises(ValueError):
        sr.tree_digest_solution(sol, max_bytes=4)


def _tar_of(home: Path, dst: Path) -> str:
    with tarfile.open(dst, 'w') as tf:
        tf.add(home / 'solution', arcname='solution')
    return str(dst)


def test_restore_home_materialises_the_selected_tree_and_moves_the_final_tree_aside(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, 'chown_tree', lambda p, uid: None)
    monkeypatch.setattr(sr, 'write_frozen', lambda home, src, mid: (home / 'env.sh').write_text('e'))
    home = tmp_path / 'home_agent'
    sol = _tree(home)
    selected = sr.tree_digest_solution(sol)
    snap = _tar_of(home, tmp_path / 'snap.tar')
    (sol / 'finetune.sh').write_text('#!/bin/bash\necho final edit\n')        # the agent kept editing
    got = sr.restore_home(home, snap, selected, 'train.jsonl', 'm', 64101, tmp_path / 'home_agent_final_a1')
    assert got == selected and sr.tree_digest_solution(home / 'solution') == selected
    assert 'final edit' in (tmp_path / 'home_agent_final_a1' / 'solution' / 'finetune.sh').read_text()
    with pytest.raises(ValueError):
        sr.restore_home(home, snap, 'f' * 64, 'train.jsonl', 'm', 64101, tmp_path / 'aside2')


def test_snapshot_extraction_refuses_members_outside_the_home(tmp_path):
    bad = tmp_path / 'bad.tar'
    with tarfile.open(bad, 'w') as tf:
        data = b'x'
        info = tarfile.TarInfo('../escape.txt')
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    home = tmp_path / 'home'
    home.mkdir()
    with pytest.raises(tarfile.TarError):
        sr.extract_solution(str(bad), home)
    assert not (tmp_path / 'escape.txt').exists()


# ---------------------------------------------------------------- G8
def test_symlink_escape_ok_for_a_regular_dir(tmp_path):
    sol = tmp_path / 'solution'
    trained = sol / 'trained_model'
    trained.mkdir(parents=True)
    (trained / 'config.json').write_text('{}')
    (trained / 'model.safetensors').write_bytes(b'\x00\x01')
    r = sr.symlink_escape(trained, sol)
    assert r['ok'] is True and r['escapes'] == [] and r['trained_present'] is True


def test_symlink_escape_vacuous_when_no_trained_model(tmp_path):
    sol = tmp_path / 'solution'
    sol.mkdir()
    r = sr.symlink_escape(sol / 'trained_model', sol)
    assert r['ok'] is True and r['trained_present'] is False


def test_symlink_escape_rejects_a_symlinked_trained_model(tmp_path):
    sol = tmp_path / 'solution'
    sol.mkdir()
    outside = tmp_path / 'ref'
    outside.mkdir()
    (outside / 'model.safetensors').write_bytes(b'\x00')
    try:
        (sol / 'trained_model').symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip('symlinks not permitted on this host')
    r = sr.symlink_escape(sol / 'trained_model', sol)
    assert r['ok'] is False and r['escapes']


# ---------------------------------------------------------------- the unprivileged sanitizer's pure parts
def test_sanitizer_picks_weights_like_transformers_and_rejects_odd_shards(tmp_path):
    d = tmp_path / 'tm'
    d.mkdir()
    (d / 'pytorch_model.bin').write_bytes(b'x')
    assert sz.select_weights(d) == ('pt', ['pytorch_model.bin'])
    (d / 'model.safetensors').write_bytes(b'x')
    assert sz.select_weights(d) == ('st', ['model.safetensors'])               # transformers 4.40 prefers it
    e = tmp_path / 'idx'
    e.mkdir()
    (e / 'pytorch_model.bin.index.json').write_text(json.dumps({'weight_map': {'a': '../ref/x.bin'}}))
    with pytest.raises(sz.Rejected):
        sz.select_weights(e)
    (e / 'pytorch_model.bin.index.json').write_text(json.dumps({'weight_map': {'a': 's1.bin', 'b': 's2.bin'}}))
    (e / 's1.bin').write_bytes(b'x')
    with pytest.raises(sz.Rejected):                                           # s2.bin missing
        sz.select_weights(e)
    (e / 's2.bin').write_bytes(b'x')
    assert sz.select_weights(e) == ('pt', ['s1.bin', 's2.bin'])
    with pytest.raises(sz.Rejected):
        sz.select_weights(tmp_path)                                            # no weights at all


def test_sanitizer_config_checks(tmp_path):
    for bad in ({'auto_map': {}}, {'trust_remote_code': True}, {'quantization_config': {}}):
        with pytest.raises(sz.Rejected):
            sz.check_config({'model_type': 'llama', **bad})
    p = tmp_path / 'c.json'
    p.write_text('{"a": NaN}')
    with pytest.raises(sz.Rejected):
        sz._json_dict(p)
    p.write_text('[1]')
    with pytest.raises(sz.Rejected):
        sz._json_dict(p)
    p.write_text('{"model_type": "llama"}')
    assert sz.check_config(sz._json_dict(p)) == {'model_type': 'llama'}


# ---------------------------------------------------------------- Codex W9 R2 N-01: per-arm judge uids
def test_a_job_may_only_name_its_own_arms_judge():
    assert sr.judge_for(64101, 64111) == 64111 and sr.judge_for(64102, 64112) == 64112 and sr.judge_for(64103, 64113) == 64113
    for uid, judge in ((64101, 64112), (64102, 64111), (64103, 64100), (64111, 64101), (64101, 65534)):
        with pytest.raises(ValueError):
            sr.judge_for(uid, judge)


class FakeNode:
    """A node's process table {pid: uid} with the kill semantics of pkill -u/-U and `kill -9 -1` issued as a uid
    (only processes of that uid can be reached)."""

    def __init__(self, table):
        self.table = dict(table)
        self.signalled = []

    def pids(self, uid):
        return [p for p, u in self.table.items() if u == int(uid)]

    def run(self, args, **kw):
        args = [str(a) for a in args]
        target = None
        if args[:3] == ['pkill', '-9', '-u'] or args[:3] == ['pkill', '-9', '-U']:
            target = int(args[3])
        elif args[0] == 'setpriv' and args[-3:] == ['kill', '-9', '-1']:
            target = int(args[1].split('=')[1])
        if target is not None:
            for p in self.pids(target):
                self.signalled.append((p, target))
                del self.table[p]
        return subprocess.CompletedProcess(args, 0)


def test_one_arms_cleanup_cannot_signal_the_other_arms_judge_process(monkeypatch):
    """Two co-located arms on disjoint GPUs: A-ISW's judge (64112) is training ref'(D') while an A-NAT scorer job
    (arm 64101, judge 64111) finishes and runs its cleanup. Only A-NAT's pair is signalled; A-ISW's judge, candidate
    and A-RND's processes are untouched."""
    node = FakeNode({101: 64101, 111: 64111, 201: 64102, 211: 64112, 212: 64112, 311: 64113})
    monkeypatch.setattr(sr, 'uid_pids', node.pids)
    monkeypatch.setattr(sr.subprocess, 'run', node.run)
    monkeypatch.setattr(sr.time, 'sleep', lambda s: None)
    monkeypatch.setattr(sr, 'clean_uid_leftovers', lambda u, tmp=False: [])
    monkeypatch.setattr(sr, 'remove_tree', lambda p: None)
    job = {'uid': 64101, 'judge_uid': sr.judge_for(64101, 64111)}
    monkeypatch.setitem(sr._ACTIVE, 'uids', {job['uid'], job['judge_uid']})
    monkeypatch.setitem(sr._ACTIVE, 'tmpdirs', set())
    sr._cleanup_active()
    assert set(node.table) == {201, 211, 212, 311}                     # A-ISW's judge (211, 212) still running
    assert {u for _, u in node.signalled} == {64101, 64111}
    with pytest.raises(ValueError):                                     # and no job can name another arm's judge
        sr.judge_for(64101, 64112)
