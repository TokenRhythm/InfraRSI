"""Trusted ENV-C judge (design §4.1 D-AR-5; runs as root on the WINH node, never imported on the driver).

    python judge_bpb.py --ckpt <final_model.pt> --out <judge.json> [--batch-size 128]

Rebuilds the baseline model from model_ref.py (extracted verbatim from the baseline train.py), loads the candidate's
saved weights (torch.load weights_only=True: tensors and plain data only, no pickled code), requires the baseline
config and exactly the baseline's parameter names, shapes and dtypes, and recomputes val_bpb with evaluate_bpb of the
ORIGINAL prepare.py next to this file, compiled and under bf16 autocast as the baseline evaluates, with a fixed eval
batch size. Writes one JSON object and exits 0; the driver decides validity from it.
"""
import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                           # the original prepare.py and model_ref.py of this judge directory


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch-size', type=int, default=128)
    a = ap.parse_args()
    t0 = time.time()
    out = {'schema': 'ext-autoresearch-judge-v1', 'ok': False, 'reasons': [], 'eval_batch_size': a.batch_size,
           'utc_start': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    try:
        import torch
        import prepare
        import model_ref
        out['files'] = {'prepare.py': sha256(prepare.__file__), 'model_ref.py': sha256(model_ref.__file__),
                        'judge_bpb.py': sha256(os.path.abspath(__file__))}
        fd = os.open(a.ckpt, os.O_RDONLY | os.O_NOFOLLOW)          # written by the untrusted run: never follow a link
        with os.fdopen(fd, 'rb') as f:
            blob = f.read()
        out['ckpt'] = {'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}
        out['versions'] = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}
        torch.set_float32_matmul_precision('high')
        import io
        ck = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
        del blob
        if not isinstance(ck, dict) or not isinstance(ck.get('state_dict'), dict) or not isinstance(ck.get('config'), dict):
            out['reasons'].append('final_model.pt must hold {"config": dict, "state_dict": dict}')
        else:
            tokenizer = prepare.Tokenizer.from_directory()
            expected = dataclasses.asdict(model_ref.expected_config(tokenizer.get_vocab_size()))
            out['config_saved'], out['config_expected'] = ck['config'], expected
            if dict(ck['config']) != expected:
                out['reasons'].append('saved model config differs from the baseline config')
            with torch.device('meta'):
                model = model_ref.GPT(model_ref.GPTConfig(**expected))
            model.to_empty(device='cuda')
            model.init_weights()                   # non-persistent rotary buffers and the baseline dtypes
            ref = model.state_dict()
            sd = ck['state_dict']
            missing, unexpected = sorted(set(ref) - set(sd)), sorted(set(sd) - set(ref))
            bad = [k for k in ref if k in sd and (not torch.is_tensor(sd[k]) or tuple(sd[k].shape) != tuple(ref[k].shape)
                                                  or sd[k].dtype != ref[k].dtype)]
            out['state_dict'] = {'keys': len(sd), 'missing': missing[:20], 'unexpected': unexpected[:20],
                                 'shape_or_dtype_mismatch': [(k, str(getattr(sd[k], 'dtype', None)), str(ref[k].dtype),
                                                              list(getattr(sd[k], 'shape', [])), list(ref[k].shape)) for k in bad[:20]]}
            if missing or unexpected or bad:
                out['reasons'].append('state_dict names, shapes or dtypes differ from the baseline model')
            nonfinite = [k for k in sd if torch.is_tensor(sd[k]) and sd[k].is_floating_point() and not torch.isfinite(sd[k]).all()]
            if nonfinite:
                out['reasons'].append(f'non-finite weights in {nonfinite[:5]}')
            if not out['reasons']:
                model.load_state_dict(sd, strict=True)
                out['num_params'] = sum(p.numel() for p in model.parameters())
                model = torch.compile(model, dynamic=False)
                model.eval()
                t1 = time.time()
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    val_bpb = prepare.evaluate_bpb(model, tokenizer, a.batch_size)
                out['eval_seconds'] = round(time.time() - t1, 3)
                out['val_bpb'] = float(val_bpb)
                if not math.isfinite(out['val_bpb']):
                    out['reasons'].append('val_bpb is not finite')
                out['peak_mem_mb'] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    except Exception as e:                         # noqa: BLE001 - recorded; the driver treats it as judge-invalid
        out['reasons'].append(f'judge error: {type(e).__name__}: {str(e)[:500]}')
        out['traceback'] = traceback.format_exc()[-4000:]
    out['ok'] = not out['reasons'] and 'val_bpb' in out
    out['seconds'] = round(time.time() - t0, 3)
    tmp = a.out + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, default=str)
    os.replace(tmp, a.out)


if __name__ == '__main__':
    main()
