"""Deterministic stand-ins for the data CPU tests: an invertible char-level tokenizer and tiny raw sources in the
three real formats (sharegpt jsonl, LongAlpaca csv, wikipedia parquet), with malformed and over-long rows."""
import csv
import hashlib
import json
import random

PAD, EOS, BOS, OFFSET = 0, 1, 2, 16


class FakeTokenizer:
    """One id per character (ord + 16); special ids below 16."""

    def __init__(self, bos=False):
        self.bos_id = BOS if bos else None
        self.eos_id, self.pad_id = EOS, PAD
        self.digest = hashlib.sha256(f'fake-char-v1-bos{bos}'.encode()).hexdigest()
        self.info = {'dir': 'fake', 'bos': ['<bos>', BOS] if bos else [None, None], 'eos': ['<eos>', EOS],
                     'pad': ['<pad>', PAD]}

    def encode_batch(self, texts):
        return [[OFFSET + ord(c) for c in t] for t in texts]


def expected_sample(tok, turns):
    """plain_v1 re-derived from whole-turn strings: per turn 'User: h\\n\\nAssistant: a' + eos + '\\n\\n', supervised
    = the last len(a)+1 characters of the turn string (' ' + a) and the eos."""
    ids, sup = ([tok.bos_id], [False]) if tok.bos_id is not None else ([], [])
    for h, a in turns:
        s = f'User: {h}\n\nAssistant: {a}'
        ids += [OFFSET + ord(c) for c in s] + [tok.eos_id] + [OFFSET + ord('\n')] * 2
        sup += [False] * (len(s) - len(a) - 1) + [True] * (len(a) + 2) + [False] * 2
    return ids, sup


def write_raw(root, seed=0):
    rng = random.Random(seed)
    alphabet = 'abcdefghij klmnopqrstuvwxyz,.!?\n"\'é中\U0001f642'

    def text(lo, hi):
        return ''.join(rng.choice(alphabet) for _ in range(rng.randint(lo, hi)))

    # sharegpt: 90 conversations; every 11th is too long for seq 128; three malformed rows; one empty reply
    convs, lines = [], []
    for i in range(90):
        turns = [(text(0, 12), text(0, 25)) for _ in range(rng.randint(1, 3))]
        if i % 11 == 5:
            turns = [(text(150, 200), text(10, 20))]
        convs.append(turns)
        lines.append(json.dumps({'conversation_id': f'c{i}', 'category': 'x',
                                 'conversation': [{'human': h, 'assistant': a} for h, a in turns]}))
    for i, line in ((7, 'not json'), (13, json.dumps({'conversation': []})),
                    (21, json.dumps({'conversation': [{'human': 'x', 'assistant': 3}]}))):
        lines[i], convs[i] = line, None
    convs[30] = [('q', '')]
    lines[30] = json.dumps({'conversation': [{'human': 'q', 'assistant': ''}]})
    p = root / 'swift__sharegpt' / 'common_en_70k.jsonl'
    p.parent.mkdir(parents=True)
    p.write_bytes(('\n'.join(lines) + '\n').encode('utf-8'))

    # LongAlpaca: 50 rows, quotes/commas/newlines inside fields, every 4th has an input, every 9th is long
    rows, long_turns = [], []
    for i in range(50):
        ins, inp, out = text(5, 40), (text(1, 10) if i % 4 == 0 else ''), text(0, 40)
        if i % 9 == 4:
            ins = text(200, 260)
        rows.append([ins, out, '', inp])
        long_turns.append([(f'{ins}\n\n{inp}' if inp.strip() else ins, out)])
    p = root / 'AI-ModelScope__LongAlpaca-12k' / 'LongAlpaca-12k_.csv'
    p.parent.mkdir(parents=True)
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['instruction', 'output', 'file', 'input'])
        w.writerows(rows)

    # wikipedia: 70 documents over several row groups
    import pyarrow as pa
    import pyarrow.parquet as pq
    docs = [(f'Title {i}', text(0, 120)) for i in range(70)]
    p = root / 'swift__wikipedia' / 'data' / '20220301.en' / 'train-00001-of-00041.parquet'
    p.parent.mkdir(parents=True)
    pq.write_table(pa.table({'id': [str(i) for i in range(70)], 'url': [f'u{i}' for i in range(70)],
                             'title': [t for t, _ in docs], 'text': [x for _, x in docs]}), p, row_group_size=9)
    return {'sharegpt': convs, 'longalpaca': long_turns, 'wiki': docs}
