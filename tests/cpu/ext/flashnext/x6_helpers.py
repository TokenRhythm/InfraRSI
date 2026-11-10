"""Helpers of the ext/flashnext judge / guards / arms CPU tests (builder X6; X5's runner helpers are fn_helpers.py).
No GPU, no network, no remote access.

- Receipts are signed with a throwaway key (never the PC's real one).
- f54_cells.json: the 20 stored formal-54 cells (STATUS_AT_FINISH_*.json rows: per update the max over 32 ranks and
  rank 0's scalars, as collect_family stores them). Ranks 1..31 of a synthetic screen are derived from them: one rank
  carries the stored per-update max, the others run below it; every rank carries rank 0's global scalars.
- write_screen(): a collected screen in FlashNextRunner._collect's layout (signals/, ranks/, nodes/, attempt.json, ...).
- FakeRunner: X5's runner interface (run(task, point, None, purpose, run_id, candidate_id=, wall_cap=, cap_s=, role=,
  reference_parsed=)) on synthetic outputs: validator first, then FlashNextRunner.x6_judge (parse -> judge ->
  guards.check -> receipt_extra) with the runner's rule that guards veto only what the judge accepted; signed receipts
  assembled by the ext.common EnvRunner base. Step times follow a fixed per-point model; numerics are the stored U+C
  cell's (E0) unless a point is told to drift.
The helpers file of X5's runner tests is fn_helpers.py (Cluster, fake_build, judge_ok, task, runner).
"""
from __future__ import annotations

import atexit
import contextlib
import copy
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

_SESSION = Path(tempfile.mkdtemp(prefix='isfnx6'))
atexit.register(shutil.rmtree, _SESSION, ignore_errors=True)
os.environ.setdefault('INFRARSI_RECEIPT_KEY', str(_SESSION / 'receipt_hmac.key'))
os.environ.setdefault('INFRARSI_LAUNCH_LEDGER', str(_SESSION / 'launches.jsonl'))

from infrarsi.contracts import new_attempt_id, utc_now                       # noqa: E402
from infrarsi.ext.common.env_runner import EnvRun, EnvRunner                  # noqa: E402
from infrarsi.ext.common.guards import GuardResult                           # noqa: E402
from infrarsi.ext.flashnext import guards as G                               # noqa: E402
from infrarsi.ext.flashnext import judge as J                                # noqa: E402
from infrarsi.ext.flashnext import parse as P                                # noqa: E402
from infrarsi.ext.flashnext import runner as R                               # noqa: E402
from infrarsi.ext.flashnext import switch_space as S                         # noqa: E402
from infrarsi.harness import signing                                         # noqa: E402

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / 'fixtures' / 'f54_cells.json'
F54 = Path(r'E:\rsi\f54v1\final')
LOSSY = ('Q', 'Q_full', 'Qo', 'Qo_full')
NODES = ['WINH1', 'WINH3', 'WINH4', 'WINH6']
HOSTS = {'WINH1': 'is-ddlhqgltfd2k3e5p-devmachine-0', 'WINH3': 'is-ddmenxvc7gwyrrln-devmachine-0',
         'WINH4': 'is-ddmen6iq3r252hfg-devmachine-0', 'WINH6': 'is-ddmeoiewou2jf53g-devmachine-0'}
IPS = {'WINH1': '172.27.96.243', 'WINH3': '172.27.67.6', 'WINH4': '172.27.25.20', 'WINH6': '172.27.54.119'}
COMMAND = ['bash', '/data/low_precision_winh_20261014/deployment/flash_mcore_env_portable.sh', 'parent.py',
           '--max-restarts=0', '--nnodes=4', '--nproc-per-node=8', '--node-rank={node_rank}', '--master-addr={master_addr}',
           '--master-port={master_port}', 'worker.py', '{output_dir}/run_precision_screen.py']


# ---------------------------------------------------------------- stored cells
def doc() -> dict:
    return json.loads(FIXTURE.read_text(encoding='utf-8'))


def cells() -> list[dict]:
    return doc()['cells']


def cell(wave: str, group: str) -> dict:
    return next(c for c in cells() if c['wave'] == wave and c['group'] == group)


def uc_a() -> dict:
    return cell('rsi_perf_v1_c1_p03', 'A')


def status_of(c: dict) -> dict:
    """A STATUS-like dict of one fixture cell (what parse.from_status_cell reads)."""
    rows = []
    for r in c['rows']:
        step, mx, ws, ce, mtp, aux, maux, mtpw, gn, lr = r
        rows.append({'step': step, 'max_rank_whole_seconds': mx,
                     'rank0': {'step': step, 'whole_seconds': ws, 'grad_norm': gn, 'base_lr': lr,
                               'losses': {'main_ce': ce, 'mtp_ce': mtp, 'main_aux_weighted': aux, 'mtp_aux_weighted': maux,
                                          'mtp_ce_weighted': mtpw}}})
    summ = {'rows': rows, 'variant': c['variant'], 'attempt_id': c['attempt_id'], 'name': c['name'], 'run_id': c['run_id'],
            'primary_window_17_30_mean_max32': c['primary_window_17_30_mean_max32'],
            'complete_verified': c['complete_verified'],
            'nodes': [{'node_rank': i, 'exit_code': e} for i, e in enumerate(c['node_exit_codes'])]}
    if c.get('excluded_timing_steps'):
        summ['excluded_timing_steps'] = c['excluded_timing_steps']
    return {'wave': c['wave'], 'collections': {c['group']: {'summary': summ}}}


def parsed_cell(c: dict) -> dict:
    """A stored cell parsed in memory (a candidate; as a reference the judge refuses it: no STATUS file behind it)."""
    return P.from_status_cell(status_of(c), c['group'])


def stored_ref(c: dict) -> dict:
    """A stored cell as a reference: written to a STATUS-like file, parsed from it (path + sha256 re-verifiable)."""
    d = _SESSION / 'stored'
    d.mkdir(exist_ok=True)
    path = d / f'STATUS_{c["wave"]}_{c["group"]}.json'
    if not path.exists():
        path.write_text(json.dumps(status_of(c)), encoding='utf-8')
    return P.from_status_cell(path, c['group'])


def profile_of(c: dict) -> list[dict]:
    """Per-update profile of a cell: max-rank time, rank 0's time and rank 0's scalars."""
    out = []
    for r in c['rows']:
        step, mx, ws, ce, mtp, aux, maux, mtpw, gn, lr = r
        out.append({'update': step, 'max_s': mx, 'rank0_s': ws, 'main_ce': ce, 'mtp_ce': mtp, 'aux': aux, 'mtp_aux': maux,
                    'mtp_ce_w': mtpw, 'gn': gn, 'lr': lr})
    return out


# ---------------------------------------------------------------- launch records
STEM = 'tp8_p1_lp_fn_test'


def launch_files(point: dict, *, attempt_id: str = 'atest0001', run: str | None = None, manifest=None,
                 master_port: int = 43123) -> tuple[dict, dict, dict]:
    """attempt.json / job.json / fn_spec.json of a bundle built for `point` (as the knob layer would write them; the
    production naming: run_id = flash32_joint_<stem>, run = <run_id>_<attempt_id>)."""
    exp = G.expected_launch(point)
    m = manifest or G.load_manifest()
    run_id = G.RUN_PREFIX + STEM
    run = run or f'{run_id}_{attempt_id}'
    attempt = {'attempt_id': attempt_id, 'run': run, 'plan': {'run_id': run_id}, 'infra_variant': exp['variant'],
               'training_seed': 42, 'run_limit': 30, 'screen_stop': 30, 'resume': 0, 'stop': 1024,
               **m['source_pins'], 'code_sha256': dict(m['variant_runtime']['inventory']),
               'data_order_policy': 'frozen_luna_q50_schedule_and_pack_plan_absolute_update_no_seed_reshuffle',
               'lifecycle_run_root': f'/data/infrarsi/ext/runs/flashnext/runs/{run}/training'}
    if exp['systems_policy'] is not None:
        attempt.update(systems_policy=exp['systems_policy'], g2_mode=exp['g2_mode'], replay_mode=exp['replay_mode'])
    if exp['uc_contract']:
        attempt['uc_contract'] = {'id': 'U_plus_C'}
    if exp['w_family_contract']:
        attempt['w_family_contract'] = {'id': 'W'}
    attempt.update({k: v for k, v in G.FIXED_ATTEMPT.items()})
    meta = f'/data/infrarsi/ext/runs/flashnext/metadata/{run}'
    env = dict(G.BASE_ENV)                                          # the base env of every registered recipe
    env.update({k: v for k, v in exp['env'].items() if v is not None})
    env.update(FLASH_CHECKPOINT_STEM=STEM, FLASH_LOG_RUN_ID=attempt['plan']['run_id'],
               FLASH_TOPOLOGY_CAPTURE_RUN=run, FLASH_RUN_ATTEMPT=f'{meta}/attempt.json',
               FLASH_IMPORT_SNAPSHOT=f'{meta}/import_snapshot.json', LP_CACHE_ROOT=G.EXT_CACHE)
    out_root = '/data/infrarsi/ext/runs/flashnext/runs'
    job = {'name': run, 'group': 'A', 'master_port': master_port, 'command': list(COMMAND), 'env': env,
           'metadata_directory': meta, 'output_root': f'{out_root}/{run}', 'timeout_seconds': 1800}
    spec = {'schema': 'flashnext-node-spec-v1', 'nodes': list(NODES), 'hostnames': [HOSTS[n] for n in NODES],
            'ips': [IPS[n] for n in NODES], 'host_numbers': [2, 4, 5, 7], 'master_addr': IPS[NODES[0]],
            'master_port': master_port, 'timeout_s': 1800.0, 'out_root': out_root,
            'locks': '/data/infrarsi/ext/locks/flashnext'}
    return attempt, job, spec


def expanded_command(job: dict, spec: dict, n: int) -> list[str]:
    rep = {'{node_rank}': str(n), '{master_addr}': spec['master_addr'], '{master_port}': str(job['master_port']),
           '{output_dir}': f'{spec["out_root"]}/{job["name"]}', '{host}': str(spec['host_numbers'][n])}
    out = []
    for v in job['command']:
        for a, b in rep.items():
            v = v.replace(a, b)
        out.append(v)
    return out


# ---------------------------------------------------------------- a collected screen
def _values(u: int, s: float, p: dict, rank: int) -> dict:
    return {'losses': {'main_ce': p['main_ce'], 'mtp_ce': p['mtp_ce'], 'main_aux_weighted': p['aux'],
                       'mtp_aux_weighted': p['mtp_aux'], 'mtp_ce_weighted': p['mtp_ce_w']},
            'tokens': {'global_main_supervised': 180000 + u, 'global_mtp_supervised': 179000 + u, 'global_physical': 262144,
                       'global_real': None, 'local_physical': 65536},
            'learning_rates': {'base_lr': p['lr'], 'muon_lr': None, 'source': 'live_native_TE_AdamW_groups_no_Muon',
                               'groups': {'g': [{'options': {'lr': p['lr']}}]}},
            'gradients': {'global_norm': p['gn'], 'clip_threshold': 1.0, 'clip_coefficient': min(1.0, 1.0 / (p['gn'] + 1e-6))},
            'timing': {'update_seconds': s, 'forward_backward_seconds': 0.8 * s, 'optimizer_seconds': 0.1 * s,
                       'offload': {'d2h_seconds': 0.01 * s}, 'includes_signal_write': False},
            'memory': {'entry_allocated_gib': 40.0 + 0.01 * rank, 'peak_allocated_gib': 60.0 + 0.01 * rank,
                       'peak_reserved_gib': 70.0 + 0.01 * rank},
            'routing': {'expert_load_histogram': None, 'scope': 'aux_and_valid_counts_in_microbatches'},
            'optimizer_updates': {'adapters': {'adamw': {'steps': u, 'calls': u, 'owned_entries': 7}}, 'expert_adam_calls': None},
            'progress': {'completed_updates': u, 'horizon': 1024, 'consumed_packs': 8 * u, 'next_update_first_pack': 8 * u,
                         'stream_fraction': u / 1024, 'epoch': None},
            'microbatches': [{'micro': 0}, {'micro': 1}]}


def rank_seconds(p: dict, rank: int) -> float:
    """rank 0 = its stored time; the argmax rank carries the stored max; the others slightly below the max."""
    u, mx, r0 = p['update'], p['max_s'], p['rank0_s']
    arg = 0 if r0 >= mx else 1 + (u * 7) % 31
    if rank == 0:
        return r0
    if rank == arg:
        return mx
    return mx * (1.0 - 0.0005 * (1 + (rank * 5 + u) % 13))


def _iso(epoch: float) -> str:
    """epoch seconds -> the ISO-8601 UTC stamp the frozen signal writer produces (parse._epoch reads it back)."""
    from datetime import datetime, timezone
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def write_screen(root, profile: list[dict], point: dict, *, attempt_id: str = 'atest0001', run: str | None = None,
                 t0: float = 1.7e9, wall_s: float | None = None, drop_ranks=(), node_exit=None, log_lines=None,
                 failure_rank=None, rank_result_overrides=None, attempt_overrides=None, env_overrides=None,
                 skip_rank_results=False, variant_override=None, record_overhead_s: float = 0.05,
                 timer_underreport: float = 1.0, ext_clock: bool = False, forge_time_utc: bool = False) -> Path:
    """time_utc is a per-rank monotonic clock advancing by each update's true duration + record_overhead_s (the
    logging/snapshot overhead the record clock sees and the per-update timer does not), so the honest timer ratio is a
    realistic ~0.99. timer_underreport < 1 writes a smaller values.timing.update_seconds than the true duration while the
    time_utc clock keeps advancing by the true duration - a code edit that neutralizes a host-device sync (the NM15 P1
    timer attack); the resulting ratio drops below guards.TIMER_RATIO_MIN.
    ext_clock writes the external watcher's records (extclock/nodeNN.jsonl): per rank, t_monotonic advances by the TRUE
    duration + record_overhead_s (the real wall the out-of-process watcher sees), grouped by node (rank // 8). This is
    the independent denominator guards.timer_consistency uses (NM18 P1-2). forge_time_utc advances the in-process
    time_utc in lockstep with the under-reported timer (gap == reported update_seconds), so the OLD time_utc ratio is
    ~1.0 while the external clock still spans the true interval - Codex's NM18 P1-2 forgery."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    attempt, job, spec = launch_files(point, attempt_id=attempt_id, run=run)
    attempt.update(attempt_overrides or {})
    for k, v in (env_overrides or {}).items():
        if v is None:
            job['env'].pop(k, None)
        else:
            job['env'][k] = v
    (root / 'attempt.json').write_text(json.dumps(attempt), encoding='utf-8')
    (root / 'job.json').write_text(json.dumps(job), encoding='utf-8')
    (root / 'fn_spec.json').write_text(json.dumps(spec), encoding='utf-8')
    (root / 'signals').mkdir(exist_ok=True)
    (root / 'ranks').mkdir(exist_ok=True)
    (root / 'nodes').mkdir(exist_ok=True)
    ext_rows = {n: [] for n in range(4)}           # the external watcher's per-node records (NM18 P1-2)
    for r in range(32):
        if r in drop_ranks:
            continue
        lines = []
        clock = float(t0)                          # the in-process time_utc clock for this rank (epoch seconds)
        ext = float(t0) + 1000.0                   # the external watcher's monotonic clock (own origin per node)
        for seq, p in enumerate(profile):
            true_s = rank_seconds(p, r)
            reported = true_s * timer_underreport
            clock += reported if forge_time_utc else true_s + record_overhead_s   # forge: time_utc in lockstep w/ timer
            ext += true_s + record_overhead_s      # the external watcher always sees the true wall between updates
            rec = {'schema': 'flash-training-signals-v1', 'run_id': attempt['plan']['run_id'], 'attempt_id': attempt_id,
                   'rank': r, 'sequence': seq, 'completed_updates': p['update'], 'time_utc': _iso(clock), 'kind': 'train_step',
                   'values': _values(p['update'], reported, p, r)}
            lines.append(json.dumps(rec, sort_keys=True, separators=(',', ':')))
            ext_rows[r // 8].append({'node_rank': r // 8, 'rank': r, 'update': p['update'],
                                     't_monotonic': ext, 't_wall': ext})
        if failure_rank == r:
            lines.append(json.dumps({'schema': 'flash-training-signals-v1', 'run_id': attempt['plan']['run_id'],
                                     'attempt_id': attempt_id, 'rank': r, 'sequence': len(lines),
                                     'completed_updates': len(profile), 'kind': 'failure',
                                     'values': {'error': 'NCCL error: unhandled system error'}}))
        (root / 'signals' / f'rank{r:02d}.jsonl').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        if not skip_rank_results:
            res = {'status': G.TRAINER_STATUS, 'attempt_id': attempt_id, 'rank': r, 'steps': 30, 'resume': 0,
                   'aux_readback_mode': 'control', 'log_run_id': attempt['plan']['run_id'],
                   'infra_family': {'variant': variant_override or attempt['infra_variant'], 'training_seed': 42,
                                    'bundle_code_sha256': attempt['code_sha256'],
                                    'data_order_policy': attempt['data_order_policy']}}
            res.update((rank_result_overrides or {}).get(r, {}))
            (root / 'ranks' / f'rank{r:02d}.json').write_text(json.dumps(res), encoding='utf-8')
    if ext_clock:
        (root / 'extclock').mkdir(exist_ok=True)
        for n in range(4):
            rows = [json.dumps(x, sort_keys=True) for x in ext_rows[n]]
            (root / 'extclock' / f'node{n:02d}.jsonl').write_text(('\n'.join(rows) + '\n') if rows else '',
                                                                  encoding='utf-8')
    span = wall_s if wall_s is not None else sum(p['max_s'] for p in profile) + 120.0
    for n, node in enumerate(NODES):
        code = (node_exit or {}).get(n, 0)
        err = None if code == 0 else ({'type': 'RunTimeout', 'timeout_seconds': 1800} if code == 124 else {'peer_failures': []})
        ex = {'exit_code': code, 'error': err, 'finished_utc': utc_now(), 'finished_unix': t0 + span - n,
              'started_unix': t0 + n, 'hostname': HOSTS[node], 'node': node}
        (root / 'nodes' / f'node{n}.exit.json').write_text(json.dumps(ex), encoding='utf-8')
        (root / 'nodes' / f'node{n}.release.json').write_text(json.dumps(
            {'idle': True, 'processes': '', 'process_group_gone': True, 'hostname': HOSTS[node], 'finished_utc': utc_now()}),
            encoding='utf-8')
        (root / 'nodes' / f'node{n}.pid.json').write_text(json.dumps(
            {'pid': 1000 + n, 'hostname': HOSTS[node], 'node_rank': n, 'node': node, 'host': spec['host_numbers'][n],
             'command': expanded_command(job, spec, n), 'started_utc': utc_now(), 'started_unix': t0 + n}), encoding='utf-8')
        text = (log_lines or {}).get(n, 'step 30/30 done\n')
        (root / 'nodes' / f'node{n}.log').write_text(text, encoding='utf-8')
    (root / 'run.json').write_text(json.dumps({'timed_out': False, 'runner_stop': False, 'cap_s': 1800}), encoding='utf-8')
    return root


def scale_profile(profile: list[dict], factor: float, *, ce_rel: float = 0.0, mtp_rel: float = 0.0, gn_rel: float = 0.0,
                  lr_add: float = 0.0) -> list[dict]:
    out = []
    for p in profile:
        q = dict(p, max_s=p['max_s'] * factor, rank0_s=p['rank0_s'] * factor)
        q['main_ce'] = p['main_ce'] * (1 + ce_rel)
        q['mtp_ce'] = p['mtp_ce'] * (1 + mtp_rel)
        q['gn'] = p['gn'] * (1 + gn_rel)
        q['lr'] = p['lr'] + lr_add
        out.append(q)
    return out


# ---------------------------------------------------------------- the fake four-node runner
def speed_factor(point: dict) -> float:
    """A fixed synthetic step-time model over S_F1 (no claim about the real path): stacks, retention, launches."""
    stack = point['implementation_stack']
    f = {'native_rollbacks': 1.0, 'production': 0.85, 'production_qsa_memo_segsum': 0.72, 'systems_policy': 0.74}[stack]
    if stack == 'systems_policy':
        f -= 0.04 * bool(point['retain_attention_prefix']) + 0.02 * bool(point['retain_expert_layers'])
        f -= 0.01 * bool(point['qkv_subgroup']) + 0.005 * bool(point['shared_master_guard'])
        f -= 0.003 * (point['qsa_dkdv_fastmask'] == '1')
    for s in ('qsa_fwd_launch', 'qsa_dq_launch', 'qsa_dkdv_launch'):
        if point.get(s, None) and point[s].startswith('8'):
            f += 0.004
    return f


@contextlib.contextmanager
def local_layer(enabled: bool):
    """X5's variant layer on this PC may be mid-edit (node launcher changed before frozen_manifest.json is regenerated):
    with enabled=False the screens of a test skip only that comparison (guards.local_layer_problems; tested on its own in
    test_fnx6_guards). Every other frozen-hash check stays on."""
    if enabled:
        yield
        return
    orig = G.local_layer_problems
    G.local_layer_problems = lambda manifest: []
    try:
        yield
    finally:
        G.local_layer_problems = orig


class _Ops:
    node = '+'.join(NODES)


class FakeRunner(EnvRunner):
    """X5's FlashNextRunner.run interface on synthetic outputs (no lease, no node)."""
    env = 'flashnext'
    deviations = ('D-F1-1', 'D-F1-2', 'D-F1-3', 'D-F1-4', 'D-F1-5', 'D-F1-6')
    stop_grace_s = 180.0

    def __init__(self, tmp, *, clock=None, base=None, reference=None, drift=None, fail=None, locked_drift=None,
                 uc_locked_fail=0, overhead_s: float = 120.0, cap_s: float = 1800.0, local_layer_check: bool = False,
                 lose=None):
        super().__init__(_Ops(), gpus=range(8), out_dir=Path(tmp) / 'runner', clock=clock or (lambda: 0.0))
        self.node = _Ops.node
        self.tmp, self.base = Path(tmp), base or profile_of(uc_a())
        self.reference_parsed, self.gates = reference, J.load_gates()
        self.drift = drift or {}                 # S.key(point) -> scale_profile kwargs (numerics drift)
        self.fail = fail or {}                   # S.key(point) -> 'oom' | 'exit'
        self.locked_drift = locked_drift or {}   # S.key(point) -> kwargs applied to locked runs only
        self.uc_locked_fail = uc_locked_fail     # number of locked U+C halves to break (void pairs)
        self.overhead_s, self.cap_s = overhead_s, cap_s
        self.local_layer_check = local_layer_check
        self.history, self.launched, self.n = [], [], 0
        self.lose = dict(lose or {})             # candidate id -> lost after judging (#37: the collect reply is cut)
        self.lost = {}                           # attempt id -> the receipt the judge would have given (re-collection)

    def bounded_cap(self, gpu_hours_left, world, wall_left, expected_s):
        cap = min(self.cap_s, gpu_hours_left * 3600.0 / max(1, int(world)) - self.stop_grace_s, wall_left - self.stop_grace_s)
        return int(cap) if cap >= max(60.0, float(expected_s)) else None

    def run(self, task, config, overlay, purpose, run_id, **kw):
        cid = str(kw.pop('candidate_id', None) or 'c')
        role = kw.pop('role', 'candidate')
        reference = kw.pop('reference_parsed', self.reference_parsed)
        cap = float(kw.pop('cap_s', None) or self.cap_s)
        point, errors, warnings = S.validate(config)
        attempt_id, t0, utc_start = new_attempt_id(), self.clock(), utc_now()
        if errors:
            r = {'schema': 'ext-receipt-v1', 'env': self.env, 'task_id': task.id, 'arm': task.arm, 'run_id': run_id,
                 'attempt_id': attempt_id, 'candidate_id': cid, 'purpose': purpose, 'valid': False,
                 'invalid_reasons': [f'invalid_proposal: {e}' for e in errors], 'step_time': {'mean_window_max_rank_s': None},
                 'metric': {'name': task.metric_name, 'value': None, 'direction': 'min', 'raw': {}}, 'judge': None,
                 'judge_valid': False, 'guard_rejections': [], 'guard_reports': [], 'point': config,
                 'not_launched': 'invalid_proposal', 'deviations': list(self.deviations),
                 'cost': {'gpu_seconds': 0.0, 'wall_seconds': 0.0, 'accounting_complete': True}}
            signing.sign(r)
            self.history.append(r)
            return r
        self.n += 1
        key = S.key(point)
        self.launched.append({'point': point, 'purpose': purpose, 'candidate_id': cid})
        prof = scale_profile(self.base, speed_factor(point), **self.drift.get(key, {}))
        if purpose == 'locked' and key in self.locked_drift:
            prof = scale_profile(prof, 1.0, **self.locked_drift[key])
        local = self.tmp / 'collected' / f'{cid}-{attempt_id}'
        kw_w = {}
        mode = self.fail.get(key)
        if mode == 'oom':
            kw_w.update(log_lines={0: 'RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB\n'}, node_exit={0: 1})
        elif mode == 'exit':
            kw_w.update(node_exit={2: 1}, drop_ranks=range(16, 24))
        if purpose == 'locked' and point == S.start_point() and self.uc_locked_fail > 0:
            self.uc_locked_fail -= 1
            prof = scale_profile(prof, 1.0, ce_rel=0.02)          # the U+C half leaves the gates: void pair
        write_screen(local, prof, point, attempt_id=attempt_id, **kw_w)
        # X5's FlashNextRunner._judge: its default judge (x6_judge = parse -> judge -> guards.check -> receipt_extra);
        # guards veto only what the judge accepted
        with local_layer(self.local_layer_check):
            verdict, guards, fields = R.FlashNextRunner.x6_judge(local, reference, self.gates, role, point, cap)
        if not verdict['judge_valid']:
            guards = GuardResult(reports=list(guards.reports) + [{'id': g['id'], 'would_reject': g['detail'],
                                                                  'note': 'judge invalid: not a veto'} for g in guards.rejections])
        res = EnvRun(candidate_id=cid, judge=json.loads(json.dumps(verdict, default=str)),
                     judge_valid=bool(verdict['judge_valid']), metric_value=verdict['metric_s'],
                     metric_raw={'numerics_class': verdict['numerics_class']}, judge_reasons=list(verdict['reasons']),
                     guards=guards, phases={'held_s': 30 * 10.0},
                     extra={**fields, 'collected': str(local), 'numerics_class': verdict['numerics_class']})
        held = sum(p['max_s'] for p in prof) + self.overhead_s
        if hasattr(self.clock, 'advance'):
            self.clock.advance(held)
        receipt = self.receipt(task, point, None, purpose, run_id, attempt_id, cid, res, None, utc_start=utc_start,
                               wall=held, held=held, occupancy={})
        receipt.update(point=point, switch_space_sha256=S.sha256(), numerics_role=role, nodes=list(NODES), cap_s=cap)
        signing.sign(receipt)
        if self.lose.pop(cid, None):             # the path measured it; our collection lost it (runner_error)
            self.lost[attempt_id] = receipt
            exc = json.JSONDecodeError('Unterminated string starting at', 'x' * 64, 41)
            err = self.receipt(task, point, None, purpose, run_id, attempt_id, cid, None, ('runner_error', exc),
                               utc_start=receipt['utc_start'], wall=held, held=held, occupancy={})
            err.update(point=point, switch_space_sha256=S.sha256(), numerics_role=role, nodes=list(NODES), cap_s=cap,
                       utc_end=receipt['utc_end'], cost=dict(receipt['cost'], accounting_complete=True))
            receipt = signing.sign(err)
        self._store(receipt)
        self.history.append(receipt)
        return receipt


UC_REF_TIME_FACTOR = 1.0137        # the new node set re-measures U+C (timing differs from the stored cell, numerics E0)


def uc_reference_receipt(tmp, *, clock=None) -> dict:
    """The U+C reference screen through the fake runner, judged against the stored U+C A cell."""
    from infrarsi.ext.flashnext import arms
    r = FakeRunner(Path(tmp) / 'ucref', clock=clock, base=scale_profile(profile_of(uc_a()), UC_REF_TIME_FACTOR))
    res = arms.run_uc_reference(r, run_id='ucref', out_dir=Path(tmp) / 'ucref_out', stored=stored_ref(uc_a()))
    assert res['valid'], res
    return res['receipt']


def copy_receipt(r: dict, **changes) -> dict:
    out = copy.deepcopy(r)
    out.update(changes)
    out.pop(signing.SIGNATURE_FIELD, None)
    signing.sign(out)
    return out


def finite(x) -> bool:
    return isinstance(x, float) and math.isfinite(x)
