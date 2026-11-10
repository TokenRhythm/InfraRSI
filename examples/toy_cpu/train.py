"""A CPU-only toy 'training job' for InfraRSI's tests and the quickstart: no framework, no GPU, a few seconds.

It sleeps for a modelled step time (knobs: micro-batch size, fused optimizer, data workers, precision, compile, the
attention kernel in kernels.py) and prints one log line per update. Loss and gradient norm follow a fixed curve of the
data, so the reference and a correct candidate agree; lossy precision moves them. A fake memory model OOMs large
micro-batches.
"""
import argparse
import json
import math
import os
import random
import time

import yaml

import kernels
from loss import loss_of

MEMORY_CAP_GIB = 16.0
SCALE = float(os.environ.get('TOY_TIME_SCALE', '0.01'))        # seconds per modelled time unit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--save')
    ap.add_argument('--save-at', type=int)
    ap.add_argument('--load')
    ap.add_argument('--workers', type=int)
    ap.add_argument('--compile', action='store_true')
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding='utf-8'))
    t = cfg['train']
    mbs, ga, prec = int(t['micro_batch_size']), int(t['grad_accum']), t.get('precision', 'fp32')
    workers = a.workers if a.workers is not None else int(t.get('workers', 1))
    compile_ = a.compile or bool(t.get('compile', False))
    seq = int(cfg['data']['seq_len'])
    mem = 3.0 + 2.5 * mbs * (0.6 if prec != 'fp32' else 1.0)
    if mem > MEMORY_CAP_GIB:
        raise SystemExit(f'RuntimeError: CUDA out of memory. Tried to allocate {mem:.1f} GiB (toy model)')
    if compile_:
        time.sleep(20 * SCALE)                                  # compile warm-up before step 1
    start, state = 1, {'w': 1.0}
    if a.load:
        ck = json.load(open(os.path.join(a.load, 'ckpt.json'), encoding='utf-8'))
        start, state = ck['step'] + 1, ck['state']
    probe = os.environ.get('INFRARSI_PROBE')
    pf = open(probe, 'a', encoding='utf-8') if probe else None
    per_sample = {'fp32': 1.0, 'bf16': 0.6, 'fp8': 0.45}[prec] * kernels.attention_cost(seq) * (0.85 if compile_ else 1.0)
    for step in range(start, a.steps + 1):
        rng = random.Random(1000 + step)                        # the data of this update (same for every config)
        batch = [rng.random() for _ in range(mbs * ga)]
        compute = per_sample * mbs * ga / (0.7 + 0.3 * min(mbs, 4))
        data = max(0.0, 0.6 - 0.15 * workers)
        optim = 0.3 if t.get('fused_optimizer') else 0.9
        jitter = 1 + 0.01 * math.sin(step * 1.7)
        time.sleep((compute + data + optim) * jitter * SCALE)
        loss, gn = loss_of(step, batch, prec)
        lr = float(cfg['optim']['lr']) * min(1.0, step / 10) * (1 - (step - 1) / a.steps)   # warm-up, then linear decay
        state['w'] = state['w'] * 0.999 + lr
        tokens = mbs * ga * seq
        print(f'step {step} | loss {loss:.6f} | grad_norm {gn:.6f} | lr {lr:.6e} | tokens {tokens} | mem {mem:.2f}GiB',
              flush=True)
        if pf:
            pf.write(json.dumps({'step': step, 't': time.time(), 'loss': loss, 'grad_norm': gn, 'lr': lr,
                                 'tokens': tokens, 'data': f'{hash(tuple(batch)) & 0xffffffffffff:012x}'}) + '\n')
            pf.flush()
        if a.save and a.save_at == step:
            os.makedirs(a.save, exist_ok=True)
            json.dump({'step': step, 'state': state, 'precision': prec}, open(os.path.join(a.save, 'ckpt.json'), 'w', encoding='utf-8'))
            print(f'saved checkpoint at step {step}', flush=True)
            return
    print('training done', flush=True)


if __name__ == '__main__':
    main()
