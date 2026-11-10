"""A small real PyTorch job for the GPU leg: an MLP on fixed random data with the knobs that matter on a GPU (bf16
autocast, torch.compile, a fused AdamW, the batch split) and two traps (skipping optimizer steps, a shorter sequence),
one log line per update: step, loss, grad norm, lr, tokens, the job's own step time (ms, CUDA-synchronised), memory."""
import argparse
import json
import os
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True)
ap.add_argument('--steps', type=int, default=30)
ap.add_argument('--bf16', action='store_true')
ap.add_argument('--compile', action='store_true')
ap.add_argument('--fused', action='store_true')
ap.add_argument('--half-steps', action='store_true', help='trap: an optimizer step every second update')
ap.add_argument('--no-step-counter', action='store_true', help='leave the optimizer-step counter out of the log')
ap.add_argument('--seq', type=int, default=1024, help='trap when changed: tokens per update')
ap.add_argument('--save')
ap.add_argument('--save-at', type=int)
ap.add_argument('--load')
a = ap.parse_args()
with open(a.config, encoding='utf-8') as f:
    cfg = json.load(f)['train']
bs, accum, hidden, lr0 = int(cfg['bs']), int(cfg['accum']), int(cfg.get('hidden', 4096)), float(cfg.get('lr', 1e-3))
dev = 'cuda'
torch.manual_seed(0)
layers = []
for _ in range(int(cfg.get('layers', 6))):
    layers += [torch.nn.Linear(hidden, hidden), torch.nn.GELU()]
model = torch.nn.Sequential(*layers, torch.nn.Linear(hidden, hidden)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=lr0, fused=a.fused)
step_fn = torch.compile(model) if a.compile else model
start = 1
if a.load:
    ck = torch.load(os.path.join(a.load, 'ckpt.pt'), map_location=dev)
    model.load_state_dict(ck['model'])
    opt.load_state_dict(ck['opt'])
    start = ck['step'] + 1
print(f'trainable_params {sum(p.numel() for p in model.parameters())}', flush=True)
opt_steps = 0
for step in range(start, a.steps + 1):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    lr = lr0 * min(1.0, step / 5) * (1 - (step - 1) / a.steps)
    for g in opt.param_groups:
        g['lr'] = lr
    total = 0.0
    for micro in range(accum):
        gen = torch.Generator(device=dev).manual_seed(1000 * step + micro)      # the same batches for every configuration
        x = torch.randn(bs, a.seq, hidden, device=dev, generator=gen)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.bf16):
            y = step_fn(x)
            loss = ((y.float() - x) ** 2).mean() / accum
        loss.backward()
        total += loss.item()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
    if not a.half_steps or step % 2 == 0:
        opt.step()
        opt_steps += 1
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    mem = torch.cuda.max_memory_allocated() / 2 ** 30
    counter = '' if a.no_step_counter else f' | optim_step {opt_steps}'
    print(f'step {step} | loss {total:.6f} | grad_norm {gn:.6f} | lr {lr:.6e}{counter} | tokens {bs * accum * a.seq} | '
          f'step_time {1000 * dt:.1f} ms | mem {mem:.2f}GiB', flush=True)
    if a.save and a.save_at == step:
        os.makedirs(a.save, exist_ok=True)
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step}, os.path.join(a.save, 'ckpt.pt'))
        print(f'saved checkpoint at step {step}', flush=True)
        break
print('training done', flush=True)
