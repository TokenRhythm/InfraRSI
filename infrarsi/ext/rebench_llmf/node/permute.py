"""Build the permuted-response dataset D' for the G4 data-dependence check (design §2.3 G4).

Runs on the node (no torch, no network). Each sample keeps its prompt; the `response` fields are permuted across
samples with a fixed seed (42), so the dataset has the same number of samples and ~the same token count but the
prompt->response mapping is destroyed. A model that actually learns from the data must move differently on D' than on
D; a cached / copied solution moves the same, which G4 rejects.
"""
from __future__ import annotations

import argparse
import json
import random


def build_dprime(in_path: str, out_path: str, seed: int = 42) -> dict:
    rows = []
    with open(in_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    n = len(rows)
    order = list(range(n))
    rnd = random.Random(seed)
    # A derangement-ish permutation: shuffle, then fix any accidental fixed points by rotating them.
    rnd.shuffle(order)
    for i in range(n):
        if order[i] == i:
            order[i], order[(i + 1) % n] = order[(i + 1) % n], order[i]
    responses = [r.get('response', '') for r in rows]
    tok_before = sum(len(x) for x in responses)
    with open(out_path, 'w', encoding='utf-8') as f:
        for i, r in enumerate(rows):
            new = dict(r)
            new['response'] = responses[order[i]]
            f.write(json.dumps(new, ensure_ascii=False) + '\n')
    tok_after = sum(len(responses[order[i]]) for i in range(n))
    return {'n_samples': n, 'fixed_points': sum(1 for i in range(n) if order[i] == i),
            'response_chars_before': tok_before, 'response_chars_after': tok_after, 'seed': seed}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    print(json.dumps(build_dprime(a.inp, a.out, a.seed)))
