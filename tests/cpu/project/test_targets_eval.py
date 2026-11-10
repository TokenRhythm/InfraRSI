"""Knob targets beyond flags and config paths (script constants, Hydra/OmegaConf overrides), init detection on
real-world layouts, the eval_matches promotion item and the cross-run ledger."""
import json
import shutil
from pathlib import Path

import pytest
import yaml

from infrarsi.project import Project, ProjectRunner, detect
from infrarsi.project.apply import plan
from infrarsi.project.flow import prior_results, run_project
from infrarsi.project.manifest import args_for, read_const, write_const
from infrarsi.verify import Verifier

TOY = Path(__file__).resolve().parents[3] / 'examples' / 'toy_cpu'


def test_write_and_read_constants(tmp_path):
    src = 'import os\nDEVICE_BATCH_SIZE = 128  # per-device batch\nCOMPILE: bool = False\nx = DEVICE_BATCH_SIZE\n'
    out = write_const(write_const(src, 'DEVICE_BATCH_SIZE', 64), 'COMPILE', True)
    assert 'DEVICE_BATCH_SIZE = 64  # per-device batch' in out and 'COMPILE = True' in out and out.endswith('x = DEVICE_BATCH_SIZE\n')
    f = tmp_path / 'train.py'
    f.write_text(out, encoding='utf-8')
    assert read_const(f, 'DEVICE_BATCH_SIZE') == 64 and read_const(f, 'NOPE') is None


def test_const_and_override_targets(tmp_path, monkeypatch):
    (tmp_path / 'train.py').write_text(
        'import sys\nSCALE = 2\nargs = dict(a.split("=") for a in sys.argv[1:] if "=" in a)\n'
        'for s in range(1, 5):\n    print(f"step {s} loss {2.0 - 0.01 * s:.4f} tokens 1024 lr 0.001", flush=True)\n'
        'print("SCALE", SCALE, "opt", args.get("optim.fused"), flush=True)\n', encoding='utf-8')
    raw = {'name': 'tgt', 'launch': {'command': '{python} train.py {args}', 'timeout_s': 60},
           'knobs': {'scale': {'type': 'int', 'choices': [2, 4], 'const': 'SCALE', 'file': 'train.py'},
                     'fused': {'type': 'bool', 'override': 'optim.fused', 'default': False}},
           'workload': {'steps': 4, 'window': [2, 4]}}
    p = Project(raw, tmp_path / 'infrarsi.yaml')
    assert p.reference_config() == {'scale': 2, 'fused': False}
    assert args_for(p, {'scale': 4, 'fused': True}) == ['optim.fused=true']
    r = ProjectRunner(p, out_dir=tmp_path / 'runs', keep_trees=True)
    rec = r.run(p.task(), {'scale': 4, 'fused': True}, None, 'screen', 'r', candidate_id='c1')
    assert 'SCALE 4 opt true' in rec['log_tail'][-1] and rec['evidence']['steps_completed'] == 4
    assert read_const(tmp_path / 'train.py', 'SCALE') == 2                    # the user's file is untouched


def test_detect_constants_and_dotlist(tmp_path):
    ar = tmp_path / 'ar'
    ar.mkdir()
    (ar / 'train.py').write_text('import torch\nTOTAL_BATCH_SIZE = 2**19\nDEVICE_BATCH_SIZE = 128  # reduce if OOM\n'
                                 'COMPILE = True\nMATRIX_LR = 0.04\nfor step in range(3):\n    pass\n', encoding='utf-8')
    m, notes = detect.draft(ar)
    k = m['knobs']['device_batch_size']
    assert k['const'] == 'DEVICE_BATCH_SIZE' and k['file'] == 'train.py' and k['choices'] == [64, 128, 256]
    assert m['knobs']['compile']['const'] == 'COMPILE' and 'matrix_lr' not in m['knobs']
    assert not m['workload']['invariants'] and any('fixed by total_batch_size' in n for n in notes)
    lf = tmp_path / 'lf'
    (lf / 'yamls' / 'pretrain').mkdir(parents=True)
    (lf / 'train.py').write_text('import sys\nfrom composer import Trainer\nfrom omegaconf import OmegaConf as om\n'
                                 'cfg = om.from_cli(sys.argv[2:])\nif __name__ == "__main__":\n    Trainer(cfg)\n',
                                 encoding='utf-8')
    (lf / 'yamls' / 'pretrain' / 'mpt.yaml').write_text(yaml.safe_dump({
        'max_duration': '4800ba', 'global_train_batch_size': 256, 'device_train_microbatch_size': 16,
        'precision': 'amp_bf16', 'fsdp_config': {'sharding_strategy': 'FULL_SHARD', 'activation_checkpointing': False}}),
        encoding='utf-8')
    m, notes = detect.draft(lf)
    assert m['framework'] == 'composer' and m['config']['file'] == 'yamls/pretrain/mpt.yaml'
    assert m['knobs']['device_train_microbatch_size']['override'] == 'device_train_microbatch_size'
    assert m['knobs']['sharding_strategy'] == {'type': 'str', 'override': 'fsdp_config.sharding_strategy',
                                               'choices': ['FULL_SHARD', 'HYBRID_SHARD', 'SHARD_GRAD_OP']}
    assert m['knobs']['activation_checkpointing']['override'] == 'fsdp_config.activation_checkpointing'
    assert 'train.py {config} max_duration={steps}ba {args}' in m['launch']['command']


def test_eval_matches_item():
    v = Verifier(0.03)
    ok = {i['id']: i for i in v.promotion({}, {}, evals=({'exit_code': 0, 'value': 2.0}, {'exit_code': 0, 'value': 2.02}))}
    assert ok['eval_matches']['status'] == 'pass'
    bad = {i['id']: i for i in v.promotion({}, {}, evals=({'exit_code': 0, 'value': 2.0}, {'exit_code': 0, 'value': 2.2}))}
    assert bad['eval_matches']['status'] == 'fail'
    none = {i['id']: i for i in v.promotion({}, {}, evals=({'exit_code': 0, 'value': 2.0}, {'exit_code': 1, 'value': None}))}
    assert none['eval_matches']['status'] == 'fail'


def test_flow_eval_and_cross_run_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv('TOY_TIME_SCALE', '0.004')
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    d = tmp_path / 'toy'
    shutil.copytree(TOY, d, ignore=shutil.ignore_patterns('.infrarsi', '__pycache__'))
    doc = yaml.safe_load((d / 'infrarsi.yaml').read_text(encoding='utf-8'))
    doc['workload'] = {'steps': 12, 'window': [5, 12], 'invariants': {'global_batch': 'micro_batch_size * grad_accum'}}
    doc['checkpoint']['save_at'] = 6
    (d / 'infrarsi.yaml').write_text(yaml.safe_dump(doc), encoding='utf-8')
    p = Project.load(d)
    res = run_project(p, strategy='hls', budget={'screens': 6}, out_root=tmp_path / 'out', log=lambda m: None)
    if res['outcome'] == 'improved':
        items = {i['id']: i['status'] for i in res['promotion']['tried'][-1]['items']}
        assert items['eval_matches'] == 'pass'
    assert res['search']['remembered'] >= 1
    rows = prior_results(p)
    assert rows and {'mechanism', 'measured_gain_pct', 'status'} <= set(rows[0])
    lines = (tmp_path / 'home' / 'ledger.jsonl').read_text(encoding='utf-8').splitlines()
    assert json.loads(lines[0])['project'] == 'toy-cpu'


def test_apply_const_and_override(tmp_path):
    (tmp_path / 'train.py').write_text('SCALE = 2  # knob\nprint(SCALE)\n', encoding='utf-8')
    (tmp_path / 'infrarsi.yaml').write_text(yaml.safe_dump({
        'name': 'tgt', 'launch': {'command': '{python} train.py {args}'},
        'knobs': {'scale': {'type': 'int', 'choices': [2, 4], 'const': 'SCALE', 'file': 'train.py'},
                  'fused': {'type': 'bool', 'override': 'optim.fused', 'default': False}},
        'code': {'mutable': ['train.py']}, 'workload': {'steps': 4, 'window': [2, 4]}}), encoding='utf-8')
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'result.json').write_text(json.dumps({
        'run_id': 'r', 'manifest': str(tmp_path / 'infrarsi.yaml'), 'outcome': 'improved',
        'reference': {'config': {'scale': 2, 'fused': False}},
        'promotion': {'promoted': 'c3', 'config': {'scale': 4, 'fused': True},
                      'overlay': {'train.py': 'SCALE = 2  # knob\nprint(SCALE, "patched")\n'}}}), encoding='utf-8')
    p = plan(run)
    assert [f['new'] for f in p['files']] == ['SCALE = 4  # knob\nprint(SCALE, "patched")\n']
    assert p['manual'] == ["override optim.fused=true (was False)"]
