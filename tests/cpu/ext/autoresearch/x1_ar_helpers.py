"""Synthetic WINH for ENV-C runner tests: frozen assets on a local root and a launcher that plays run.sh."""
import hashlib
import json
import re
from pathlib import Path

from infrarsi.agent.budget import VirtualClock
from infrarsi.ext.autoresearch import infra_only_check, source
from infrarsi.ext.autoresearch.node import code_attest
from infrarsi.ext.autoresearch.task import env_c_task
from infrarsi.ext.common.remote_exec import LocalNodeOps

GROUPS = [{'kind': 'adamw', 'n_tensors': 1, 'numel': 4194304, 'initial_lr': 0.0049, 'betas': [0.8, 0.95], 'eps': 1e-10},
          {'kind': 'muon', 'n_tensors': 24, 'numel': 25165824, 'initial_lr': 0.04, 'ns_steps': 5, 'beta2': 0.95}]


def make_assets(root: Path) -> dict:
    """Fake frozen files on the local root (data shards, tokenizer, kernel sources) and their hashes."""
    def put(path, data: bytes):
        p = root / path.lstrip('/')
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return hashlib.sha256(data).hexdigest()
    return {'data': {s: put(f'{source.NODE_DATA}/{s}', s.encode()) for s in source.DATA_SHARDS},
            'tokenizer': {f: put(f'{source.NODE_TOKENIZER}/{f}', f.encode()) for f in source.TOKENIZER_FILES},
            'kernel': {f: put(f'{source.NODE_KERNEL}/{f}', f.encode()) for f in source.KERNEL_FILES},
            'node': 'WINH0', 'measured_utc': 'test', 'reference': {'optimizer_groups': GROUPS}}


def train_log(losses=None, val_bpb=0.9977, steps=948, crash=False) -> str:
    losses = losses or source.R2['first_losses'] + [6.3, 6.2]
    head = ('Vocab size: 8,192\nModel config: {...}\nTime budget: 300s\nGradient accumulation steps: 2\n')
    lines = ''.join(f'\rstep {i:05d} (0.0%) | loss: {l:.6f} | lrm: 1.00 | dt: {320 if i else 31937}ms | tok/sec: 1,650,000 | '
                    f'mfu: 39.9% | epoch: 1 | remaining: 300s    ' for i, l in enumerate(losses))
    if crash:
        return head + lines + '\nTraceback (most recent call last):\n  File "train.py", line 548\ntorch.OutOfMemoryError: CUDA out of memory\n'
    tail = (f'\n---\nval_bpb:          {val_bpb:.6f}\ntraining_seconds: 300.1\ntotal_seconds:    354.8\npeak_vram_mb:     45060.2\n'
            f'mfu_percent:      39.59\ntotal_tokens_M:   {steps * 524288 / 1e6:.1f}\nnum_steps:        {steps}\nnum_params_M:     50.3\ndepth:            8\n')
    return head + lines + tail


def run_meta(steps=948, **over) -> dict:
    consts = dict(infra_only_check.expected_constants(source.base_train()), DEVICE_BATCH_SIZE=128, MAX_SEQ_LEN=2048, TIME_BUDGET=300)
    meta = {'schema': 'ext-autoresearch-run-meta-v1', 'wrapper': {'train_py_wall_s': 340.0, 'save_s': 2.0}, 'constants': consts,
            'counters': {'step': steps, 'total_tokens': steps * 524288, 'total_training_time': 300.1, 'grad_accum_steps': 2,
                         'num_params': 50332176, 'num_flops_per_token': 2.390784e8, 'val_bpb': 0.9977},
            'config': source.R2['model_config'], 'optimizer_groups': [dict(g) for g in GROUPS], 'float32_matmul_precision': 'high'}
    for k, v in over.items():
        meta[k] = v
    return meta


def attest(C: Path, **over) -> tuple[dict, dict]:
    """(expected, live) code attestation of a materialized attempt: the trusted step really compiles repo/train.py,
    repo/prepare.py and judge/base_train.py; the live side copies the candidate's hashes (a run that kept its
    definitions bound) unless `over` changes fields."""
    out = C / 'judge' / 'code_expected.json'
    code_attest._cli(['expect', '--train', str(C / 'repo' / 'train.py'), '--prepare', str(C / 'repo' / 'prepare.py'),
                      '--base', str(C / 'judge' / 'base_train.py'), '--out', str(out)])
    exp = json.loads(out.read_text(encoding='utf-8'))
    snap = {**{k: v for k, v in exp['candidate'].items()}, 'python': exp['python'],
            'bound_to_prepare': {n: True for n in code_attest.PREPARE_BOUND}, 'instance_overrides': [],
            'model_class': 'GPT', 'model_class_is_GPT': True,
            'optimizer': {'present': True, 'class': 'MuonAdamW', 'class_is_train_def': True, 'instance_overrides': []}}
    live = {**snap, 'first_step': dict(snap), 'optimizer_steps': 948, 'optimizers_stepped': 1, 'same_optimizer': True,
            'override_steps': 0, 'step_hook': 'registered'}
    live.update(over)
    return exp, live


class Node:
    """A LocalNodeOps whose launcher plays run.sh according to `behaviour` (per call, popped from a list)."""

    def __init__(self, tmp_path, behaviours=None, clock=None):
        self.clock = clock or VirtualClock()
        self.behaviours = list(behaviours or [])
        self.scripts = []
        self.ops = LocalNodeOps(tmp_path, launcher=self.launch)
        self.root = Path(tmp_path)

    def launch(self, ops, script, log):
        text = Path(script).read_text(encoding='utf-8')
        if Path(script).name.startswith('call_'):          # a basic-agent tool call: optional hook, exit 0
            base = Path(script).with_suffix('')
            out = self.tool_hook(text, ops) if getattr(self, 'tool_hook', None) else 'ok\n'
            Path(str(base) + '.out').write_bytes(out.encode())
            Path(str(base) + '.err').write_bytes(b'')
            Path(str(base) + '.rc').write_bytes(b'0\n')
            self.clock.advance(3)
            return {'pid': 2}
        self.scripts.append(text)
        b = self.behaviours.pop(0) if self.behaviours else {}
        C = Path(script).parent
        (C / 'out').mkdir(parents=True, exist_ok=True)
        (C / 'judge').mkdir(parents=True, exist_ok=True)
        if b.get('no_done'):
            return {'pid': 1}
        self.clock.advance(b.get('seconds', 380))
        crash = b.get('crash', False)
        (C / 'train.log').write_bytes(train_log(b.get('losses'), b.get('printed', 0.9977), crash=crash).encode())
        exit_code = 1 if crash else b.get('exit', 0)
        contained = not b.get('contain_fail')
        judged = 'INFRARSI_SAVE=1' in text and not crash and contained
        live = None
        if 'code_attest.py expect' in text:                # the trusted root step of the infra track
            _, live = attest(C, **(b.get('attest') or {}))
        if not crash:
            meta = b.get('meta') or run_meta()
            if live is not None and 'code_attestation' not in meta:
                meta = {**meta, 'code_attestation': live}
            (C / 'out' / 'run_meta.json').write_text(json.dumps(meta), encoding='utf-8')
            if judged:
                (C / 'out' / 'final_model.pt').write_bytes(b'weights' * 100)
                judge = {'schema': 'ext-autoresearch-judge-v1', 'ok': not b.get('judge_fail'), 'val_bpb': b.get('judged', 0.9977),
                         'num_params': 50332176, 'reasons': ['state_dict names, shapes or dtypes differ'] if b.get('judge_fail') else []}
                (C / 'judge' / 'judge.json').write_text(json.dumps(judge), encoding='utf-8')
                if 'judge_h2d' in text:                    # the loader check's second judge on the same weights
                    (C / 'judge_h2d').mkdir(parents=True, exist_ok=True)
                    (C / 'judge_h2d' / 'judge.json').write_text(json.dumps({**judge, 'val_bpb': b.get('judged2', judge['val_bpb'])}),
                                                                encoding='utf-8')
        if b.get('tamper'):
            (self.root / source.NODE_DATA.lstrip('/') / source.DATA_SHARDS[0]).write_bytes(b'tampered')
        times = (f'launch 100.0\nhost fake\ngpu 0, NVIDIA H100 80GB HBM3, 0 MiB, 1980 MHz, 700.00 W\nattest_exit 0\n'
                 f'train_start 100.5\ntrain_exit {exit_code}\ntrain_end {100.5 + b.get("train_s", 345.0)}\n'
                 + ('contain_ok 446.0\n' if contained else 'contain_fail 476.0\n'))
        if judged:
            times += 'judge_start 446.0\njudge_exit 0\njudge_end 480.0\n'
        (C / 'DONE.txt').write_bytes((times + 'done 481.0\n').encode())
        return {'pid': 1}


def ar_task(tmp_path, arm='C-IS'):
    return env_c_task(arm, assets=make_assets(Path(tmp_path)))


def script_c(script: str) -> str:
    return re.search(r'^# InfraRSI ENV-C attempt (\S+)', script, re.M).group(1)
