"""The toy project's eval: the checkpoint's model on a fixed batch; prints eval_loss (InfraRSI's eval_matches)."""
import argparse
import json
import os
import random

from loss import loss_of

ap = argparse.ArgumentParser()
ap.add_argument('--load', required=True)
a = ap.parse_args()
ck = json.load(open(os.path.join(a.load, 'ckpt.json'), encoding='utf-8'))
rng = random.Random(7)
loss, _ = loss_of(ck['step'], [rng.random() for _ in range(64)], ck.get('precision', 'fp32'))
print(f'eval_loss {loss:.6f}', flush=True)
