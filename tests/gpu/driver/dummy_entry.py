"""Dummy torchrun entry: contract-conformant heartbeat, per-rank signals and receipt without a model.

GPU use is seconds: NCCL init + one all_reduce per update + a few matmuls. request.config['dummy'] controls it:
{load_s, step_sleep_s, matmul_n, matmul_iters, allreduce_mb, oom_below_tp, ce_jitter_rel}. Honours STOP.json at
every update boundary (rank 0 decides, all ranks agree through an all_reduce). Every signal row carries
utc_start_s/utc_end_s (time.time) taken at the same points as whole_update_s (perf_counter), and inter_step_s
(start(k) - start(k-1)); both clocks are bound at import, as in runtime/timers.py. The receipt carries the invariants
of the training-process contract (runtime/receipt_writer + adapters trainer field names): complete dtype census,
LR / loss / trainable / model-config / frozen-zone digests, per-update clip calls, backends flags, effective_config,
and the data identity from INDEX.json (manifest, order digest, consumed-pack digests, supervised-token vector).
The receipt also echoes the trusted request's request_digest, numerics_role, data_manifest and
effective_overlay_digest, and reports overlay_digest as the effective L1 set computed from the default hook files of
the code tree it runs from (PYTHONPATH, else this repository) replaced by the request's overlay files - the same rule as
runtime/overlay_loader.Overlays.digest. simulate_lifecycle() writes a lifecycle attempt in the runtime layout
(signals/ uninterrupted, signals_resume/ resumed) with lifecycle.signals_resume_digest bound to the resume files.
The helpers above main() import no torch, so tests/cpu/driver reuses them (simulate_attempt) to fake an attempt on a
LocalRemote.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import socket
import statistics
import sys
import time

SCHEMA_RECEIPT = 'infrarsi-receipt-v1'
SCHEMA_HEARTBEAT = 'infrarsi-heartbeat-v1'
INDEX = '/data/infrarsi/datasets/packed/INDEX.json'
RUNS_MARK = '/data/infrarsi/runs/'
MOE_BRIDGES = ('qwen3_moe', 'qwen3_next', 'deepseek_v2', 'qwen4_exp')
HOOK_MODULES = ('schedule', 'recompute_policy', 'comm_groups', 'state_placement', 'host_sync', 'data_path')  # = runtime
_perf, _utc = time.perf_counter, time.time            # bound at import (independent clocks for the harness)
DEFAULTS = {'load_s': 1.0, 'step_sleep_s': 0.15, 'matmul_n': 2048, 'matmul_iters': 2, 'allreduce_mb': 16,
            'oom_below_tp': None, 'ce_jitter_rel': 0.0, 'step_jitter_s': 0.0}
L0_DEFAULTS = {'tensor_model_parallel_size': 1, 'pipeline_model_parallel_size': 1, 'context_parallel_size': 1,
               'micro_batch_size': 1, 'sequence_parallel': False}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(obj):
    """contracts.digest (canonical JSON sha256) without importing infrarsi."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def atomic(path, obj):
    tmp = '%s.%d.tmp' % (path, os.getpid())
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def dummy_cfg(req):
    cfg = dict(DEFAULTS)
    cfg.update((req.get('config') or {}).get('dummy') or {})
    return cfg


def is_synthetic(task):
    data = task.get('data') or {}
    return bool(data.get('synthetic')) or str(data.get('key', '')).startswith('D-SYN')


def is_moe(task):
    m = task.get('model') or {}
    return m.get('bridge_type') in MOE_BRIDGES or 'moe' in str(m.get('family', ''))


def data_facts(req, index_path=INDEX, path_map=None):
    """Supervised tokens per update, the plan order and the INDEX order digest from INDEX.json + manifest (None when
    unavailable or synthetic). path_map maps a '/data/...' path to a local one (LocalRemote file systems)."""
    if is_synthetic(req['task']):
        return None
    pm = path_map or (lambda p: p)
    try:
        with open(pm(index_path), encoding='utf-8') as f:
            e = json.load(f)['tasks'][req['task']['id']]
        with open(pm(e['manifest']), encoding='utf-8') as f:
            order = json.load(f)['plans'][req['input_plan']]['order']
        return {'sup': e['supervised_tokens_per_update'][req['input_plan']], 'order': order, 'ppu': e['packs_per_update'],
                'manifest': req.get('data_manifest') or e['manifest'], 'manifest_sha256': e['manifest_sha256'],
                'order_digest': (e.get('data_order_digest') or {}).get(req['input_plan'])}
    except (OSError, KeyError, ValueError, TypeError):
        return None


def auto_data_facts(req, out_dir):
    """data_facts for an attempt directory inside a (possibly local) /data tree: <root>/data/infrarsi/runs/..."""
    p = str(out_dir).replace('\\', '/')
    if RUNS_MARK not in p:
        return None
    root = p.split(RUNS_MARK)[0]
    return data_facts(req, path_map=lambda x: root + x if x.startswith('/data/') else x)


def lr_at(task, update):
    s = task['objective']['lr_schedule']
    k = update - 1
    return s['min'] + 0.5 * (s['peak'] - s['min']) * (1 + math.cos(math.pi * k / s['horizon_updates']))


def global_scalars(req, update, data=None):
    """Deterministic per-update scalars: identical across configs (E0), optionally jittered (E1/FAIL tests).
    Supervised tokens come from the packed-data INDEX when available (as the real trainer reports them)."""
    task, plan = req['task'], req['input_plan']
    cfg = dummy_cfg(req)
    wl = task['workload']
    phys = int(wl['physical_tokens_per_update'])
    sup = int(data['sup'][update - 1]) if data else int(phys * (0.35 + 0.05 * ((update * 7 + (3 if plan == 'B' else 0)) % 5)))
    ce = 0.6 + 1.9 * math.exp(-update / 40.0) + (0.013 if plan == 'B' else 0.0)
    jitter = float(cfg['ce_jitter_rel'])
    if jitter:
        ce *= 1 + jitter * (1 if update % 2 else -1)
    return {'main_ce': ce, 'aux_loss': (0.0123 + 0.0001 * update) if is_moe(task) else None,
            'grad_norm': 1.5 / (1 + 0.05 * update), 'lr': lr_at(task, update), 'physical_tokens': phys,
            'supervised_tokens': sup, 'optimizer_steps': update, 'clip_calls': 1}


_SCHEMA_REL = os.path.join('infrarsi', 'actions', 'schema', 'l0_megatron_core_0_18_2.json')


def l0_defaults():
    """The L0 schema defaults (as adapters/megatron_core/config_map.make_plan uses them): from the code tree on
    PYTHONPATH (WINH) or the repository this file lives in; a minimal set when neither is readable."""
    roots = [x for x in os.environ.get('PYTHONPATH', '').split(os.pathsep) if x]
    roots.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')))
    for root in roots:
        try:
            with open(os.path.join(root, _SCHEMA_REL), encoding='utf-8') as f:
                return {k: v.get('default') for k, v in json.load(f)['keys'].items()}
        except (OSError, ValueError, KeyError, AttributeError):
            continue
    return dict(L0_DEFAULTS)


def effective_config(req):
    """The layout that ran: L0 schema defaults updated with the request config (env excluded), as make_plan."""
    eff = l0_defaults()
    eff.update({k: v for k, v in (req.get('config') or {}).items() if k != 'env'})
    return eff


def microbatches_per_update(req):
    """Backward micro-batches per rank per update for the request's layout: packs_per_update / (DP x MBS)."""
    eff = effective_config(req)
    tp, pp, cp, mbs = (int(eff.get(k) or 1) for k in ('tensor_model_parallel_size', 'pipeline_model_parallel_size',
                                                      'context_parallel_size', 'micro_batch_size'))
    dp = max(1, int(req['world_size']) // (tp * pp * cp))
    return int(req['task']['workload']['packs_per_update']) // (dp * mbs)


def invariants(req, per_step, data=None):
    task, plan = req['task'], req['input_plan']
    wl = task['workload']
    n = len(per_step)
    moe = is_moe(task)
    obj_moe = (task.get('objective') or {}).get('moe') if isinstance(task.get('objective'), dict) else None
    lrs = [p['lr'] for p in per_step]
    inv = {'gbs_packs': wl['packs_per_update'], 'physical_tokens_per_update': wl['physical_tokens_per_update'],
           'supervised_tokens_total': sum(p['supervised_tokens'] for p in per_step), 'optimizer_steps': n,
           'clip_calls_total': sum(p.get('clip_calls', 1) for p in per_step),
           'lr_schedule_digest': sha(lrs), 'lr_schedule_expected_digest': sha([lr_at(task, u) for u in range(1, n + 1)]),
           'trainable_params': 1000, 'trainable_param_digest': sha(['dummy.weight', [1000]]),
           'loss_definition_digest': sha('token-mean CE'), 'frozen_zone_digest': sha('dummy-frozen-zone'),
           'model_config_digest': sha(task['model']['repo_id']), 'router_topk': 8 if moe else None,
           'capacity_factor': (obj_moe or {}).get('capacity_factor') if moe else None, 'num_experts': 128 if moe else None,
           'dtype_census': {'param': {'torch.bfloat16': 100}, 'grad': {'torch.float32': 100},
                            'master': {'torch.float32': 100}, 'exp_avg': {'torch.float32': 100},
                            'exp_avg_sq': {'torch.float32': 100}},
           'model_defined_fp32_params': [], 'backends_flags': {'cuda.matmul.allow_tf32': False, 'cudnn.allow_tf32': True},
           'supervised_tokens_per_update': {plan: [p['supervised_tokens'] for p in per_step]}}
    if data:
        ppu, order = int(data['ppu']), data['order']
        packs = [sorted(int(order[i]) for i in range(u * ppu, (u + 1) * ppu)) for u in range(n)]
        inv.update(data_order_digest=data.get('order_digest') or sha([task['id'], plan]), data_consumed_digest=sha(packs),
                   data_consumed_per_update=[sha(u) for u in packs],
                   data={'source': data['manifest'], 'manifest_digest': data.get('manifest_sha256'), 'plan': plan,
                         'packs_per_update': ppu})
    else:
        packs = [[u] for u in range(n)]
        inv.update(data_order_digest=sha([task['id'], plan]), data_consumed_digest=sha(packs),
                   data_consumed_per_update=[sha(u) for u in packs],
                   data={'source': 'synthetic', 'manifest_digest': sha({'synthetic': task['id']}), 'plan': plan})
    return inv


def signals_digest(out_dir, world, sub='signals'):
    h = hashlib.sha256()
    for r in range(world):
        with open(os.path.join(out_dir, sub, 'rank%d.jsonl' % r), 'rb') as f:
            h.update(f.read())
    return h.hexdigest()


def _code_roots():
    roots = [x for x in os.environ.get('PYTHONPATH', '').split(os.pathsep) if x]
    roots.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')))
    return roots


def effective_overlay_digest(req):
    """runtime Overlays.digest: the default hook files of the code tree, each replaced by a request overlay file."""
    files = None
    for root in _code_roots():
        d = os.path.join(root, 'infrarsi', 'runtime', 'overlays')
        if all(os.path.isfile(os.path.join(d, m + '.py')) for m in HOOK_MODULES):
            files = {}
            for m in HOOK_MODULES:
                with open(os.path.join(d, m + '.py'), 'rb') as f:
                    files[m + '.py'] = hashlib.sha256(f.read()).hexdigest()
            break
    if files is None:
        return None
    files.update(((req.get('overlay') or {}).get('files') or {}))
    return sha([[k, v] for k, v in sorted(files.items())])


def trusted_request(req, out_dir=None):
    """The request as uploaded (out_dir/request.json when present, else the dict given)."""
    path = os.path.join(out_dir, 'request.json') if out_dir else None
    if path and os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return req


def request_echo(req, out_dir=None):
    t = trusted_request(req, out_dir)
    return {'request_digest': sha(t), 'numerics_role': t.get('numerics_role'), 'data_manifest': t.get('data_manifest'),
            'effective_overlay_digest': t.get('effective_overlay_digest')}


def make_receipt(req, rank_rows, *, valid, reasons, phases, env, events=None, memory=None, out_dir=None, data=None):
    """rank_rows: {rank: [signal rows]}; rank 0 rows carry the global scalars."""
    world = req['world_size']
    n = min(len(v) for v in rank_rows.values()) if len(rank_rows) == world else 0
    per = []
    for u in range(1, n + 1):
        t = [(rank_rows[r][u - 1]['whole_update_s'], r) for r in range(world)]
        tmax, slow = max(t)
        g = rank_rows[0][u - 1]
        inter = [rank_rows[r][u - 1].get('inter_step_s') for r in range(world)]
        per.append({'step': u, 'max_rank_whole_s': tmax, 'slowest_rank': slow,
                    'max_rank_inter_step_s': max(inter) if all(x is not None for x in inter) else None,
                    **{k: g[k] for k in ('main_ce', 'aux_loss', 'grad_norm', 'lr', 'physical_tokens', 'supervised_tokens',
                                         'clip_calls')}})
    win = req.get('window')
    step_time = None
    if win and n >= win[1]:
        xs = [p['max_rank_whole_s'] for p in per if win[0] <= p['step'] <= win[1]]
        inter = [p['max_rank_inter_step_s'] for p in per if win[0] < p['step'] <= win[1]]
        m = statistics.fmean(xs)
        step_time = {'mean_window_max_rank_s': m, 'median_window_max_rank_s': statistics.median(xs),
                     'rsd_window': statistics.stdev(xs) / m if len(xs) > 1 else 0.0,
                     'inter_step_mean_s': statistics.fmean(inter) if inter and None not in inter else None}
    mem = memory or {'per_rank': {str(r): {'peak_alloc_gib': max((x['peak_alloc_gib'] for x in rows), default=0.0),
                                           'peak_reserved_gib': max((x['peak_reserved_gib'] for x in rows), default=0.0)}
                                  for r, rows in rank_rows.items()}}
    mem.setdefault('max_peak_reserved_gib', max((v['peak_reserved_gib'] for v in mem['per_rank'].values()), default=0.0))
    task = req['task']
    rec = {'schema': SCHEMA_RECEIPT, 'run_id': req['run_id'], 'attempt_id': req['attempt_id'], 'purpose': req['purpose'],
           'task_id': task['id'], 'task_digest': req['task_digest'], 'config_digest': req['config_digest'],
           'code_digest': req['code_digest'], 'overlay_digest': effective_overlay_digest(req),
           **request_echo(req, out_dir),
           'input_plan': req['input_plan'], 'world_size': world, 'nodes': req['nodes'], 'steps': req['steps'],
           'steps_completed': n, 'window': win, 'step_time': step_time, 'per_step': per,
           'invariants': invariants(req, per, data), 'memory': mem, 'effective_config': effective_config(req),
           'events': events or {'alloc_retries': 0, 'alloc_ooms': 0, 'nccl_errors': 0, 'nan_inf_steps': 0, 'exceptions': []},
           'phases': phases, 'env': env, 'lifecycle': None, 'valid': valid, 'invalid_reasons': reasons,
           'signals_digest': signals_digest(out_dir, world) if out_dir else None}
    return rec


def bind_resume_signals(out_dir, world):
    """Record the rank-ordered digest of signals_resume/ in receipt.json's lifecycle block (as the runtime does)."""
    path = os.path.join(out_dir, 'receipt.json')
    with open(path, encoding='utf-8') as f:
        rec = json.load(f)
    rec['lifecycle'] = dict(rec.get('lifecycle') or {}, signals_resume_digest=signals_digest(out_dir, world, 'signals_resume'))
    atomic(path, rec)
    return rec


LIFECYCLE_FLAGS = {'resumed_ok': True, 'resume_scalars_match': True, 'state_exact': True, 'export_ok': True,
                   'forward_ok': True, 'forward': {'rel_l2': 0.018, 'top1_agreement': 0.996, 'forward_ok': True}}
GDN_BRIDGES = ('qwen3_next', 'qwen4_exp')


def deterministic_fields(req):
    """What the trainer reports about its kernels on a lifecycle the driver asked to run deterministically
    (AMENDMENT-002): Gated DeltaNet keeps the default kernels and says so; every other family records the replaced
    settings (runtime/lifecycle.deterministic_overrides)."""
    if ((req.get('task') or {}).get('model') or {}).get('bridge_type') in GDN_BRIDGES:
        return {'deterministic': False,
                'deterministic_unavailable': 'packed-THD Gated DeltaNet has no deterministic Megatron-Core path'}
    return {'deterministic': True, 'attention_backend': 'flash',
            'deterministic_overrides': {'torch.use_deterministic_algorithms': {'requested': False, 'ran': True},
                                        'TransformerConfig.deterministic_mode': {'requested': False, 'ran': True},
                                        'attention_backend': {'requested': 'auto', 'ran': 'flash'},
                                        'env.NCCL_ALGO': {'requested': None, 'ran': 'Ring'}}}


def simulate_lifecycle(req, out_dir, *, flags=None, resume_fn=None, bind=True, **sim):
    """A lifecycle attempt in the runtime layout: signals/ runs save_at + resume_steps updates uninterrupted,
    signals_resume/ holds the resumed process's rows (resume_fn(rank, row) -> row may alter them), the receipt keeps
    the request's steps and carries the lifecycle block with signals_resume_digest (bind=False leaves it out). The
    block starts from deterministic_fields(req); keys given in flags replace them."""
    life = req['lifecycle']
    save, extra = int(life['save_at']), int(life['resume_steps'])
    world = int(req['world_size'])
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isfile(os.path.join(out_dir, 'request.json')):    # the trusted request the receipt echoes
        atomic(os.path.join(out_dir, 'request.json'), req)
    simulate_attempt(dict(req, steps=save + extra), out_dir, **sim)
    path = os.path.join(out_dir, 'receipt.json')
    with open(path, encoding='utf-8') as f:
        rec = json.load(f)
    rec['steps'] = req['steps']
    rec['lifecycle'] = {**deterministic_fields(req), **(LIFECYCLE_FLAGS if flags is None else flags), 'saved_at': save}
    atomic(path, rec)
    os.makedirs(os.path.join(out_dir, 'signals_resume'), exist_ok=True)
    for r in range(world):
        with open(os.path.join(out_dir, 'signals', 'rank%d.jsonl' % r), encoding='utf-8') as f:
            rows = [json.loads(x) for x in f if x.strip()]
        with open(os.path.join(out_dir, 'signals_resume', 'rank%d.jsonl' % r), 'w', encoding='utf-8', newline='\n') as f:
            for row in rows:
                if row['step'] > save:
                    f.write(json.dumps((resume_fn or (lambda k, x: x))(r, row), sort_keys=True) + '\n')
    return bind_resume_signals(out_dir, world) if bind else rec


def simulate_attempt(req, out_dir, *, step_time=lambda r, u: 0.5 + 0.001 * r, stop_at=None, oom=False,
                     node_exit=0, write_receipt=True, ce_scale=1.0, data='auto', clocks=True, perf_scale=1.0,
                     exit_fields=None, phases_override=None, microbatches=None, gn_scale=None):
    """Write everything a real attempt would (no torch, no processes): heartbeat, signals, receipt, exit files.

    step_time is the true (UTC) duration; perf_scale != 1 forges the process's perf counter (whole_update_s and
    phases scale, UTC does not). clocks=False drops utc_start_s/utc_end_s (pre-change trainer signals).
    microbatches(rank, update) -> int overrides the backward micro-batch count of a row (default: the layout's);
    gn_scale(update) -> float scales the grad norm of an update (numerics tests).
    data='auto' reads INDEX.json + manifest from the /data tree that holds out_dir (a LocalRemote root); pass a facts
    dict ({sup, order, ppu, manifest, manifest_sha256, order_digest}) or None explicitly to override."""
    if data == 'auto':
        data = auto_data_facts(req, out_dir)
    world = req['world_size']
    os.makedirs(os.path.join(out_dir, 'signals'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'logs'), exist_ok=True)
    steps = req['steps']
    done = steps if stop_at is None else min(stop_at, steps)
    if oom:
        done = 0
    rows = {r: [] for r in range(world)}
    t0 = t = 1_700_000_000.0
    prev = None
    mb_default = microbatches_per_update(req)
    for u in range(1, done + 1):
        start = t                                     # barrier-aligned start on every rank
        t = start + max(step_time(r, u) for r in range(world))
        g = global_scalars(req, u, data)
        g['main_ce'] *= ce_scale
        if gn_scale is not None:
            g['grad_norm'] *= gn_scale(u)
        for r in range(world):
            w = step_time(r, u) * perf_scale
            row = {'step': u, 'whole_update_s': w, 'fwd_bwd_s': 0.8 * w, 'optim_s': 0.1 * w, 'data_s': 0.01 + 0.001 * r,
                   'peak_alloc_gib': 40.0 + r * 0.01, 'peak_reserved_gib': 44.0 + r * 0.01, 'alloc_retries': 0,
                   'alloc_ooms': 0, 'inter_step_s': None if prev is None else (start - prev) * perf_scale,
                   'microbatches': mb_default if microbatches is None else microbatches(r, u)}
            if row['microbatches'] is None:
                row.pop('microbatches')
            if clocks:
                row.update(utc_start_s=start, utc_end_s=start + step_time(r, u))
            if r == 0:
                row.update(g)
            rows[r].append(row)
        prev = start
    for r in range(world):
        with open(os.path.join(out_dir, 'signals', 'rank%d.jsonl' % r), 'w', encoding='utf-8', newline='\n') as f:
            for row in rows[r]:
                f.write(json.dumps(row, sort_keys=True) + '\n')
    stopped = stop_at is not None and stop_at < steps
    reasons = ['stopped by STOP.json at update %d' % done] if stopped else (['CUDA out of memory (simulated)'] if oom else [])
    events = {'alloc_retries': 0, 'alloc_ooms': 1 if oom else 0, 'nccl_errors': 0, 'nan_inf_steps': 0,
              'exceptions': ['torch.OutOfMemoryError: CUDA out of memory (simulated)'] if oom else []}
    if write_receipt:
        train = (t - t0) * perf_scale
        phases = {'startup': 1.0, 'load': 2.0, 'build': 1.0, 'compile_first_step': 1.0, 'train': train, 'save': 0.0,
                  'export': 0.0, 'total': 5.0 + train}
        phases.update(phases_override or {})
        rec = make_receipt(req, rows, valid=not (stopped or oom), reasons=reasons, events=events, phases=phases,
                           env={'torch': 'sim', 'hostnames': ['sim']}, out_dir=out_dir, data=data)
        atomic(os.path.join(out_dir, 'receipt.json'), rec)
    atomic(os.path.join(out_dir, 'heartbeat.json'), {'schema': SCHEMA_HEARTBEAT, 'phase': 'failed' if (stopped or oom) else 'done',
                                                     'step': done, 'utc': utc(), 'pid': 0})
    for i in range(len(req['nodes'])):
        atomic(os.path.join(out_dir, 'exit_node%d.json' % i), {
            'schema': 'infrarsi-node-exit-v1', 'node_index': i, 'hostname': 'sim%d' % i,
            'exit_code': node_exit if not oom else 1, 'reason': 'stop' if stopped else ('completed' if not oom else 'child_failed'),
            'killed_by_wrapper': False, 'error': None, 'started_utc': utc(), 'finished_utc': utc(), 'wall_s': 60.0,
            'release': {'gpu_apps_after': [], 'process_group_gone': True}, **(exit_fields or {})})
        with open(os.path.join(out_dir, 'logs', 'node%d.log' % i), 'w', encoding='utf-8') as f:
            f.write('torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB (simulated)\n' if oom else 'ok\n')


# ---------------------------------------------------------------- real torchrun entry (needs torch)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--request', required=True)
    args = ap.parse_args(argv)
    t_start = _utc()
    with open(args.request, encoding='utf-8') as f:
        req = json.load(f)
    cfg = dummy_cfg(req)
    out = req['out_dir']
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local = int(os.environ['LOCAL_RANK'])
    assert world == req['world_size'], (world, req['world_size'])
    hb_path, stop_path = os.path.join(out, 'heartbeat.json'), os.path.join(out, 'STOP.json')

    def hb(phase, step=0):
        if rank == 0:
            atomic(hb_path, {'schema': SCHEMA_HEARTBEAT, 'phase': phase, 'step': step, 'utc': utc(), 'pid': os.getpid()})

    hb('load')
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(local)
    time.sleep(float(cfg['load_s']))
    t_load = _utc()
    hb('build')
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=10), device_id=torch.device('cuda', local))
    x = torch.ones(1 << 20, device='cuda')
    dist.all_reduce(x)
    torch.cuda.synchronize()
    assert int(x[0].item()) == world, 'all_reduce sanity failed'
    t_build = _utc()
    n = int(cfg['matmul_n'])
    a = torch.randn(n, n, device='cuda', dtype=torch.bfloat16)
    buf = torch.ones(int(cfg['allreduce_mb']) * (1 << 18), device='cuda')
    sig_path = os.path.join(out, 'signals', 'rank%d.jsonl' % rank)
    rows, reasons, events = [], [], {'alloc_retries': 0, 'alloc_ooms': 0, 'nccl_errors': 0, 'nan_inf_steps': 0, 'exceptions': []}
    stopped, failed = False, None
    tp = int((req.get('config') or {}).get('tensor_model_parallel_size', 1))
    data = data_facts(req) if rank == 0 else None
    n_mb = microbatches_per_update(req)
    hb('train', 0)
    prev_start = None
    try:
        with open(sig_path, 'w', encoding='utf-8', newline='\n') as sf:
            for u in range(1, int(req['steps']) + 1):
                flag = torch.tensor([1.0 if (rank == 0 and os.path.exists(stop_path)) else 0.0], device='cuda')
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                if flag.item() > 0:
                    stopped = True
                    break
                if cfg['oom_below_tp'] is not None and tp < int(cfg['oom_below_tp']) and u == 2:
                    raise torch.OutOfMemoryError('CUDA out of memory. Tried to allocate 20.00 GiB (simulated by dummy_entry)')
                torch.cuda.synchronize()
                t0, u0 = _perf(), _utc()
                c = a
                for _ in range(int(cfg['matmul_iters'])):
                    c = c @ a
                dist.all_reduce(buf)
                time.sleep(float(cfg['step_sleep_s']) + float(cfg['step_jitter_s']) * ((u * 31 + rank) % 7) / 7.0)
                torch.cuda.synchronize()
                whole, u1 = _perf() - t0, _utc()
                row = {'step': u, 'whole_update_s': whole, 'utc_start_s': u0, 'utc_end_s': u1, 'fwd_bwd_s': whole * 0.8,
                       'optim_s': whole * 0.1, 'data_s': 0.0, 'inter_step_s': None if prev_start is None else t0 - prev_start,
                       'microbatches': n_mb,
                       'peak_alloc_gib': torch.cuda.max_memory_allocated() / 2 ** 30,
                       'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2 ** 30,
                       'alloc_retries': int(torch.cuda.memory_stats().get('num_alloc_retries', 0)),
                       'alloc_ooms': int(torch.cuda.memory_stats().get('num_ooms', 0))}
                prev_start = t0
                if rank == 0:
                    row.update(global_scalars(req, u, data))
                sf.write(json.dumps(row, sort_keys=True) + '\n')
                sf.flush()
                rows.append(row)
                hb('train', u)
    except torch.OutOfMemoryError as e:
        failed = e
        events['alloc_ooms'] = 1
        events['exceptions'].append('torch.OutOfMemoryError: %s' % e)
        reasons.append('CUDA out of memory')
    t_train = _utc()
    if failed is None:
        dist.barrier()
    if rank == 0:
        rank_rows = {}
        for r in range(world):
            with open(os.path.join(out, 'signals', 'rank%d.jsonl' % r), encoding='utf-8') as f:
                rank_rows[r] = [json.loads(line) for line in f if line.strip()]
        if stopped:
            reasons.append('stopped by STOP.json at update %d' % len(rows))
        env = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'nccl': '.'.join(map(str, torch.cuda.nccl.version())),
               'hostnames': [socket.gethostname()], 'gpu_names': [torch.cuda.get_device_name(local)],
               'whitelisted_env': {k: os.environ.get(k) for k in ('CUDA_DEVICE_MAX_CONNECTIONS', 'PYTORCH_CUDA_ALLOC_CONF')},
               'site_env': {k: os.environ.get(k) for k in ('NCCL_SOCKET_IFNAME', 'GLOO_SOCKET_IFNAME')}}
        phases = {'startup': t_load - t_start, 'load': t_load - t_start, 'build': t_build - t_load, 'compile_first_step': 0.0,
                  'train': t_train - t_build, 'save': 0.0, 'export': 0.0, 'total': _utc() - t_start}
        rec = make_receipt(req, rank_rows, valid=not (stopped or failed) and len(rows) == req['steps'], reasons=reasons,
                           phases=phases, env=env, events=events, out_dir=out, data=data)
        atomic(os.path.join(out, 'receipt.json'), rec)
        hb('failed' if (stopped or failed) else 'done', len(rows))
    if failed is not None:
        os._exit(3)
    dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    sys.exit(main())
