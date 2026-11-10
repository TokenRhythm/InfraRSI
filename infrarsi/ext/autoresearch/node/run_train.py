"""ENV-C candidate entry (runs as nobody in the candidate's repo directory; never imported on the driver).

Executes the candidate's train.py unchanged (runpy, as __main__) and afterwards - outside its 300 s timed region, after
its own evaluation - saves for the trusted judge and the infra-only check:
  $INFRARSI_OUT/final_model.pt   {'config': asdict(GPTConfig), 'state_dict': CPU tensors of the uncompiled model}
  $INFRARSI_OUT/run_meta.json    the run's constants, steps, tokens, timing, optimizer parameter groups and the
                                   code attestation of the protected callables (code_attest.live, Codex W9 H-02)
INFRARSI_SAVE=0 skips the weights (free track: the printed val_bpb is the score). The exit status is train.py's;
a failure while saving is written to run_meta.json / save_error.txt and never turns a crash into a success.
"""
import dataclasses
import json
import math
import os
import runpy
import sys
import time
import traceback

import code_attest                                 # tools/ (sys.path[0]); imported before the candidate's code runs

OUT = os.environ['INFRARSI_OUT']
SAVE = os.environ.get('INFRARSI_SAVE', '1') == '1'
CONSTANTS = ('TOTAL_BATCH_SIZE', 'EMBEDDING_LR', 'UNEMBEDDING_LR', 'MATRIX_LR', 'SCALAR_LR', 'WEIGHT_DECAY', 'ADAM_BETAS',
             'WARMUP_RATIO', 'WARMDOWN_RATIO', 'FINAL_LR_FRAC', 'DEPTH', 'ASPECT_RATIO', 'HEAD_DIM', 'WINDOW_PATTERN',
             'DEVICE_BATCH_SIZE', 'MAX_SEQ_LEN', 'TIME_BUDGET', 'H100_BF16_PEAK_FLOPS')
COUNTERS = ('step', 'total_tokens', 'total_training_time', 'grad_accum_steps', 'num_params', 'num_flops_per_token', 'val_bpb',
            'startup_time', 'steady_state_mfu', 'peak_vram_mb', 'tokens_per_fwdbwd')


def plain(v):
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    try:
        return plain(v.item())                     # 0-d tensors
    except Exception:                              # noqa: BLE001
        return repr(v)[:200]


def groups(optimizer):
    rows = []
    for g in getattr(optimizer, 'param_groups', []) or []:
        rows.append({'kind': g.get('kind'), 'n_tensors': len(g.get('params', [])),
                     'numel': int(sum(p.numel() for p in g.get('params', []))),
                     **{k: plain(g.get(k)) for k in ('initial_lr', 'betas', 'eps', 'ns_steps', 'beta2') if k in g},
                     'weight_decay_final': plain(g.get('weight_decay')), 'momentum_final': plain(g.get('momentum'))})
    return rows


def main():
    repo = os.getcwd()
    sys.path.insert(0, repo)
    train_path, prepare_path = os.path.join(repo, 'train.py'), os.path.join(repo, 'prepare.py')
    watch = code_attest.StepWatch(train_path, prepare_path)       # attests at the first optimizer step (point of use)
    watch.install()
    t0 = time.time()
    try:
        g = runpy.run_path(train_path, run_name='__main__')
    finally:
        watch.uninstall()
    t1 = time.time()
    meta = {'schema': 'ext-autoresearch-run-meta-v1', 'wrapper': {'train_py_wall_s': round(t1 - t0, 3), 't_start': t0, 't_end': t1}}
    try:
        meta['code_attestation'] = {**code_attest.live(g, sys.modules.get('prepare'), train_path=train_path,
                                                       prepare_path=prepare_path, model=g.get('model'),
                                                       optimizer=g.get('optimizer')),
                                    **watch.summary(g.get('optimizer'))}
    except Exception as e:                         # noqa: BLE001 - a missing attestation invalidates an infra run
        meta['code_attestation_error'] = f'{type(e).__name__}: {str(e)[:300]}'
    try:
        import torch
        meta['constants'] = {k: plain(g[k]) for k in CONSTANTS if k in g}
        meta['counters'] = {k: plain(g[k]) for k in COUNTERS if k in g}
        cfg = g.get('config')
        meta['config'] = plain(dataclasses.asdict(cfg)) if dataclasses.is_dataclass(cfg) else plain(cfg)
        meta['optimizer_groups'] = groups(g.get('optimizer'))
        meta['float32_matmul_precision'] = torch.get_float32_matmul_precision()
        meta['torch'] = torch.__version__
        if SAVE:
            model = g['model']
            model = getattr(model, '_orig_mod', model)
            sd = {k: v.detach().to('cpu').clone() for k, v in model.state_dict().items()}
            tmp = os.path.join(OUT, 'final_model.pt.tmp')
            torch.save({'config': dataclasses.asdict(cfg), 'state_dict': sd}, tmp)
            os.replace(tmp, os.path.join(OUT, 'final_model.pt'))
            meta['wrapper']['save_s'] = round(time.time() - t1, 3)
    except Exception as e:                         # noqa: BLE001 - recorded; missing outputs invalidate the run
        meta['save_error'] = f'{type(e).__name__}: {str(e)[:500]}'
        with open(os.path.join(OUT, 'save_error.txt'), 'w', encoding='utf-8') as f:
            f.write(traceback.format_exc())
    with open(os.path.join(OUT, 'run_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)


if __name__ == '__main__':
    main()
