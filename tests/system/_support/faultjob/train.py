"""A stand-in training job whose faults are modes (`--mode` or FAULT_MODE): the one job the fault-injection tests drive.

One line per update: step, loss, grad norm, lr, tokens, the job's own step time (ms), memory, a data digest. The loss
depends on the step and the global batch only, so a speed change keeps it; `fast` is a real gain (0.6 x the step).
Base step time: FAULT_STEP_S seconds (default 0.03) x STEP_SCALE (a const knob). Overrides `train.<key>=<value>` are
accepted as bare arguments (an override-target knob). FAULT_SLOW=1 (an env-target knob) makes every step 1.5 x longer.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

STEP_SCALE = 1.0          # a const knob: InfraRSI rewrites this literal in the candidate tree
MODES = ('ok', 'fast', 'slow', 'fast_lossy', 'edge_lossy', 'crash', 'oom', 'hang', 'nan', 'incomplete', 'fewer_tokens',
         'skip_lr', 'frozen_write', 'fake_fast', 'partial_timer', 'silent', 'tqdm', 'crlf', 'burst', 'dup', 'zero_based',
         'noisy', 'digest_change', 'nccl_end', 'nccl_mid', 'traceback_exit0', 'mem_over', 'exit_nonzero_end',
         'fast_noresume', 'fast_reset_state', 'fast_eval_diff', 'seq_len_change')
STATE_STEP = 0.05         # the trained state advances by this per update; the loss depends on it (a lost state shows)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--mode', default=os.environ.get('FAULT_MODE', 'ok'), choices=MODES)
    ap.add_argument('--save')
    ap.add_argument('--save-at', type=int)
    ap.add_argument('--load')
    a, rest = ap.parse_known_args()
    overrides = {}
    for tok in rest:
        if '=' in tok and not tok.startswith('-'):
            k, v = tok.split('=', 1)
            overrides[k] = v
        else:
            ap.error(f'unrecognized arguments: {tok}')
    return a, overrides


def main():
    a, over = parse()
    with open(a.config, encoding='utf-8') as f:
        cfg = json.load(f)
    t = cfg['train']
    bs, accum, workers = int(t['bs']), int(t['accum']), int(t.get('workers', 1))
    opt = over.get('train.opt', t.get('opt', 'plain'))
    mode = a.mode
    base = float(os.environ.get('FAULT_STEP_S', '0.03')) * STEP_SCALE
    step_s = base * (0.6 if mode.startswith('fast') else 1.0)
    step_s *= 1.5 if (os.environ.get('FAULT_SLOW') == '1' or mode == 'slow') else 1.0
    step_s *= (0.85 if opt == 'fused' else 1.0) * (1.0 if workers > 1 else 1.1)
    seq = 256 if mode == 'seq_len_change' else 512
    tokens = bs * accum * seq
    start, state = 1, {'w': 1.0}
    if a.load and mode != 'fast_noresume':
        with open(os.path.join(a.load, 'ckpt.json'), encoding='utf-8') as f:
            ck = json.load(f)
        start, state = ck['step'] + 1, ck['state']
        if mode == 'fast_reset_state':                # the step counter resumes, the trained state does not
            state = {'w': 1.0}
    nl = '\r\n' if mode == 'crlf' else '\n'
    held = []
    rng = random.Random(0)
    last = time.perf_counter()                      # the job's own step timer covers the whole update (sleep + logging)
    for s in range(start, a.steps + 1):
        if mode == 'crash' and s == 3:
            raise RuntimeError('CUDA error: an illegal memory access was encountered')
        if mode == 'oom' and s == 2:
            print('torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of '
                  '15.99 GiB of which 1.02 GiB is free.', flush=True)
            sys.exit(1)
        if mode == 'hang' and s == 5:
            time.sleep(10 ** 6)                         # alive, silent: a stuck collective
        if mode == 'incomplete' and s > a.steps - 3:
            break
        if mode == 'frozen_write' and s == 2:
            with open('frozen.py', 'a', encoding='utf-8') as f:
                f.write('\n# touched at run time\n')
        if mode == 'traceback_exit0' and s == 3:
            print('Traceback (most recent call last):\n  File "x.py", line 1, in <module>\nValueError: recovered and went on',
                  flush=True)
        if mode == 'nccl_mid' and s == 3:             # a collective failure swallowed by the job: it goes on and exits 0
            print('[rank0]: NCCL error: ncclSystemError: System call (e.g. socket, malloc) failed.', flush=True)
        dt = step_s * (1 + (rng.uniform(-0.3, 0.3) if mode == 'noisy' else 0.01 * math.sin(s)))
        time.sleep(dt)
        now = time.perf_counter()
        dt, last = now - last, now                  # what the update took on the job's own clock
        loss = (2.5 * math.exp(-0.02 * s) + 0.05 * math.sin(s * (bs * accum))) * (1 + 0.2 * (state['w'] - 1.0))
        if mode == 'fast_lossy':
            loss *= 1.05
        if mode == 'edge_lossy':
            loss *= 1.025
        gn = 1.0 + 0.5 * math.exp(-0.05 * s)
        lr = 1e-3 * min(1.0, s / 5) * (1 - (s - 1) / a.steps) * (0.5 if mode == 'skip_lr' else 1.0)
        tok = tokens // 2 if mode == 'fewer_tokens' else tokens
        rep_ms = 1000 * dt * (0.6 if mode == 'fake_fast' else 0.85 if mode == 'partial_timer' else 1.0)
        mem = 99.0 if mode == 'mem_over' else 3.0 + 0.5 * bs
        digest = hashlib.sha1(f'{s}:{bs * accum}:{"x" if mode == "digest_change" else ""}'.encode()).hexdigest()[:12]
        loss_s = 'nan' if (mode == 'nan' and s >= 4) else f'{loss:.6f}'
        step_no = s - 1 if mode == 'zero_based' else s
        line = (f'step {step_no} | loss {loss_s} | grad_norm {gn:.6f} | lr {lr:.6e} | tokens {tok} | seq_len {seq} | '
                f'step_time {rep_ms:.2f} ms | mem {mem:.2f}GiB | data_digest {digest}')
        if mode == 'tqdm':
            bar = '#' * (s * 10 // a.steps)
            line = (f'Steps: {int(100 * s / a.steps):3d}%|{bar:<10}| {s}/{a.steps} [00:{s % 60:02d}<00:{(a.steps - s) % 60:02d}, '
                    f'{dt:.2f}s/it, lr={lr:.2e}, step_loss={loss:.4f}]')
            sys.stdout.write(line + '\r')
            sys.stdout.flush()
        elif mode == 'silent':
            pass
        elif mode == 'burst':
            held.append(line)
        else:
            sys.stdout.write(line + nl)
            sys.stdout.flush()
            if mode == 'dup':
                sys.stdout.write(line + nl)
                sys.stdout.flush()
        state['w'] = state['w'] + STATE_STEP               # the trained state after this update
        if a.save and a.save_at == s:
            os.makedirs(a.save, exist_ok=True)
            with open(os.path.join(a.save, 'ckpt.json'), 'w', encoding='utf-8') as f:
                json.dump({'step': s, 'state': state, 'mode': mode}, f)
            print(f'saved checkpoint at step {s}', flush=True)
            return
    if mode == 'burst':
        print('\n'.join(held), flush=True)
    if mode == 'tqdm':
        print(flush=True)
    if mode == 'nccl_end':
        print('[rank0]: NCCL error: ncclSystemError: System call (e.g. socket, malloc) failed.', flush=True)
    print('training done', flush=True)
    if mode == 'exit_nonzero_end':
        sys.exit(3)


if __name__ == '__main__':
    main()
