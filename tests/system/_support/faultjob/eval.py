"""The stand-in job's eval: the checkpoint's loss on a fixed batch; prints eval_loss. A checkpoint saved by the
fast_eval_diff mode evaluates 10 % off."""
import argparse
import json
import math
import os

ap = argparse.ArgumentParser()
ap.add_argument('--load', required=True)
a = ap.parse_args()
with open(os.path.join(a.load, 'ckpt.json'), encoding='utf-8') as f:
    ck = json.load(f)
loss = 2.5 * math.exp(-0.02 * ck['step']) * (1.10 if ck.get('mode') == 'fast_eval_diff' else 1.0)
print(f'eval_loss {loss:.6f}', flush=True)
