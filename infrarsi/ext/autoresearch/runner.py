"""ENV-C EnvRunner: one candidate train.py = one fresh 300 s run on one leased H100, then the trusted judge.

Per attempt, under runs/autoresearch/<run_id>/<candidate_id>/<attempt_id>/ on the node (root-owned unless noted).
<run_id>/ is root:root 0711 - traverse-only, because several uids may share it (the calibration's four concurrent
runs) - and is never re-grouped; <candidate_id>/ and the attempt's own directories are root:<arm gid> 0750, created
with node_ext.mkdir_domain, which refuses to re-own an existing directory of another uid: the arm reads its tree,
never writes the root-owned parts, and no other uid can enter it:
  repo/train.py (the candidate: overlay text, knobs rendered on the base), repo/prepare.py (the D-AR-7 variant; 0444)
  tools/run_train.py      candidate entry (runs train.py, then saves final_model.pt + run_meta.json outside the timer)
  tools/code_attest.py    code attestation of the protected callables (Codex W9 H-02)
  home/                   HOME of the run: .cache/autoresearch/{data, tokenizer} -> frozen data and tokenizer (root),
                          writable only tmp/, hf_home/ (FA3 kernel cache, R2 recipe), caches/ (the arm uid, 0700)
  out/                    the arm uid, 0700: final_model.pt, run_meta.json
  judge/                  0700: model_ref.py, judge_bpb.py, original prepare.py, base_train.py, code_expected.json,
                          judge.json, judge.log, home/
  judge_h2d/              loader_check only (AMENDMENT_EXT_002 b): the same judge with the D-AR-7 prepare.py on the
                          same weights; the receipt reports |val_bpb(original) - val_bpb(D-AR-7)| (<= 1e-6 expected)
  run.sh, DONE.txt, train.log
Training runs as the arm's trust-domain uid (train_uid, task.ARM_UIDS; never 65534; CUDA_VISIBLE_DEVICES = the leased
GPU; offline; 600 s timeout = a failure, program.md). When it ends run.sh kills every process of that uid and starts
the judge (root, infra track) only when none survived; the driver kills the uid again in a finally after every attempt
(Codex W9 B-02), and EnvRunner once more before the lease is released. Before the training a root step compiles the
protected callables for the code attestation. Frozen files are hashed by root before launch and after the run.
wall_cap (bounded admission, Codex H-05): the training timeout shrinks to what the cap leaves after the fixed phases;
worst_case() is the attempt's conservative wall. Deviations kept from the R2 runner: D-AR-1 data mirror, D-AR-2 FA3
mirror, D-AR-3 pip venv; the trusted judge adds D-AR-5, co-location D-AR-4, the race-free loader D-AR-7. Bulky outputs
(final_model.pt, caches) are deleted after collection.
"""
from __future__ import annotations

import math
import time

from ..common import guards, node_ext
from ..common.env_runner import EnvRun, EnvRunner
from ..common.remote_exec import ContainmentError, new_token, nobody_cmd, q, uid_kill
from ..common.session_hooks import sha_text
from . import infra_only_check, judge_files, knobs, logs, source
from .task import ENVC_UIDS, JUDGE_BATCH_SIZE, TRAIN_TIMEOUT_S, check_arm_variant

JUDGE_TIMEOUT_S = 300          # measured 17-23 s per judge (cold compile cache included)
LOG_MAX_BYTES = 4_000_000
LOADER_TOLERANCE = 1e-6
# Conservative wall of the phases around the training (worst_case; measured: materialize 14-17 s, collect 10-15 s).
MATERIALIZE_S, ATTEST_S, CONTAIN_S, COLLECT_S = 120, 60, 40, 120
MIN_TRAIN_S = 400              # a training timeout below this cannot finish the 300 s budget + compile + evaluation


def kill_snippet(token: str) -> str:
    """Bash: SIGKILL every process whose environment carries INFRARSI_TOKEN=<token> (the trusted root judge's)."""
    return ('for p in /proc/[0-9]*; do if tr "\\0" "\\n" < $p/environ 2>/dev/null | grep -qx '
            f'{q("INFRARSI_TOKEN=" + token)}; then kill -9 ${{p#/proc/}} 2>/dev/null; fi; done')


def contain_snippet(uid: int) -> str:
    """Bash function `contain`: SIGKILL every process of the trust-domain uid (as that uid: kill -9 -1; as root: pkill
    by real and effective uid) until none but zombies remain, up to 30 s; non-zero when one survived."""
    u = node_ext._check_domain_uid(uid)
    return (f'contain() {{ for i in $(seq 1 30); do setpriv --reuid={u} --regid={u} --clear-groups kill -9 -1 2>/dev/null; '
            f'pkill -9 -U {u} 2>/dev/null; pkill -9 -u {u} 2>/dev/null; '
            f'ps -U {u} -u {u} -o stat= 2>/dev/null | grep -qv "^Z" || return 0; sleep 1; done; return 1; }}')


class AutoresearchRunner(EnvRunner):
    env = 'autoresearch'
    deviations = source.DEVIATIONS
    sibling_uids = ENVC_UIDS                # the co-located ENV-C arms (envc_crnd_d: an agent profiling on its GPU)

    def __init__(self, ops, *, gpus=(0,), track: str = 'infra', expected: dict | None = None, reference_losses=None,
                 train_timeout_s: int = TRAIN_TIMEOUT_S, judge_timeout_s: int = JUDGE_TIMEOUT_S, judge_batch_size: int = JUDGE_BATCH_SIZE,
                 keep_weights: bool = False, colocated: bool = False, runs_root: str = source.NODE_RUNS, poll_s: float = 10.0,
                 sleep=time.sleep, train_uid: int, loader_check: bool = False, **kw):
        super().__init__(ops, gpus=gpus, **kw)
        if len(self.gpus) != 1:
            raise ValueError('ENV-C runs on exactly one GPU')
        if track not in ('infra', 'free'):
            raise ValueError(f'track {track!r}')
        node_ext._check_domain_uid(train_uid)       # the arm's trust domain (task.ARM_UIDS), never nobody
        self.track, self.expected = track, expected
        self.reference_losses = list(reference_losses or source.R2['first_losses'])
        self.train_timeout_s, self.judge_timeout_s, self.judge_batch_size = int(train_timeout_s), int(judge_timeout_s), int(judge_batch_size)
        self.keep_weights, self.colocated, self.runs_root = keep_weights, colocated, runs_root.rstrip('/')
        self.poll_s, self.sleep = poll_s, sleep
        self.train_uid = int(train_uid)
        self.loader_check = bool(loader_check) and track == 'infra'

    def describe(self) -> dict:
        return {**super().describe(), 'track': self.track, 'train_timeout_s': self.train_timeout_s,
                'judge_batch_size': self.judge_batch_size, 'colocated': self.colocated, 'train_uid': self.train_uid}

    def domain_uids(self) -> list[int]:
        return [self.train_uid]

    def _judges(self) -> int:
        return 0 if self.track != 'infra' else (2 if self.loader_check else 1)

    def _fixed_s(self) -> float:
        """Worst-case wall of everything but the training itself."""
        return (MATERIALIZE_S + ATTEST_S + 10 + CONTAIN_S + self._judges() * (self.judge_timeout_s + 10) + 2 * self.poll_s
                + COLLECT_S)

    def worst_case(self) -> float:
        return float(self.train_timeout_s + self._fixed_s())

    def train_timeout_for(self, wall_cap) -> int | None:
        """min(protocol timeout, what the cap leaves after the fixed phases); None below MIN_TRAIN_S."""
        t = self.train_timeout_s if wall_cap is None else int(min(self.train_timeout_s, float(wall_cap) - self._fixed_s()))
        return t if t >= min(MIN_TRAIN_S, self.train_timeout_s) else None

    # ------------------------------------------------------------ candidate files
    def render(self, config: dict, overlay: dict) -> str:
        text = overlay.get('train.py', source.base_train())
        if config and {k: v for k, v in config.items() if v != knobs.DEFAULTS.get(k)}:
            text = knobs.apply(text, config)
        return text

    def layout(self, run_id: str, candidate_id: str, attempt_id: str) -> dict:
        c = f'{self.runs_root}/{run_id}/{candidate_id}/{attempt_id}'
        return {'run': f'{self.runs_root}/{run_id}', 'cand': f'{self.runs_root}/{run_id}/{candidate_id}', 'C': c,
                'repo': f'{c}/repo', 'tools': f'{c}/tools', 'home': f'{c}/home', 'out': f'{c}/out', 'judge': f'{c}/judge',
                'jhome': f'{c}/judge/home', 'judge2': f'{c}/judge_h2d', 'j2home': f'{c}/judge_h2d/home',
                'judge_cache': f'{self.runs_root}/{run_id}/.judge_cache'}

    def _home_links(self, home: str) -> dict:
        hub = f'{home}/hf_home/hub/models--{source.KERNEL_REPO.replace("/", "--")}'
        return {f'{home}/.cache/autoresearch/data': source.NODE_DATA, f'{home}/.cache/autoresearch/tokenizer': source.NODE_TOKENIZER,
                f'{hub}/snapshots/{source.KERNEL_REV}/build/{source.KERNEL_VARIANT}': source.NODE_KERNEL}

    def _env_dirs(self, home: str) -> dict:
        return {**source.NODE_CUDA_ENV, 'HF_HOME': f'{home}/hf_home', 'TORCHINDUCTOR_CACHE_DIR': f'{home}/caches/inductor',
                'TRITON_CACHE_DIR': f'{home}/caches/triton', 'TRITON_HOME': f'{home}/caches', 'XDG_CACHE_HOME': f'{home}/caches/xdg',
                'CUDA_CACHE_PATH': f'{home}/caches/nv', 'MPLCONFIGDIR': f'{home}/caches/mpl'}

    def judge_cmd(self, L: dict, jdir: str, jhome: str, token: str) -> str:
        """The trusted judge as root: cleared environment, the run's root-only compile cache."""
        jdirs = {**self._env_dirs(jhome), 'TORCHINDUCTOR_CACHE_DIR': f'{L["judge_cache"]}/inductor',
                 'TRITON_CACHE_DIR': f'{L["judge_cache"]}/triton', 'TRITON_HOME': L['judge_cache']}
        jenv = {'HOME': jhome, 'PATH': f'{source.NODE_ENV}/bin:/usr/bin:/bin', 'CUDA_VISIBLE_DEVICES': str(self.gpus[0]),
                'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1', 'TMPDIR': f'{jhome}/tmp', 'LANG': 'C.UTF-8',
                'INFRARSI_TOKEN': token, **jdirs}
        return (f'timeout --kill-after=10 {self.judge_timeout_s} env -i ' + ' '.join(f'{k}={q(v)}' for k, v in jenv.items())
                + f' {q(source.NODE_PYTHON)} {q(jdir + "/judge_bpb.py")} --ckpt {q(L["out"] + "/final_model.pt")} '
                f'--out {q(jdir + "/judge.json")} --batch-size {self.judge_batch_size}')

    def attest_cmd(self, L: dict) -> str:
        """The trusted code expectation (root, isolated python, compile only) of the protected callables."""
        return (f'timeout --kill-after=10 {ATTEST_S} env -i PATH=/usr/bin:/bin LANG=C.UTF-8 HOME={q(L["jhome"])} '
                f'{q(source.NODE_PYTHON)} -I {q(L["tools"] + "/code_attest.py")} expect --train {q(L["repo"] + "/train.py")} '
                f'--prepare {q(L["repo"] + "/prepare.py")} --base {q(L["judge"] + "/base_train.py")} '
                f'--out {q(L["judge"] + "/code_expected.json")}')

    def script(self, L: dict, train_token: str, judge_token: str, train_timeout: int | None = None) -> str:
        gpu = self.gpus[0]
        train = nobody_cmd(f'exec python {q(L["tools"] + "/run_train.py")}', home=L['home'], cwd=L['repo'],
                           path_dirs=(f'{source.NODE_ENV}/bin',), gpus=(gpu,), token=train_token,
                           extra={**self._env_dirs(L['home']), 'INFRARSI_OUT': L['out'],
                                  'INFRARSI_SAVE': '1' if self.track == 'infra' else '0'},
                           timeout=int(train_timeout or self.train_timeout_s), uid=self.train_uid)
        judge = self.judge_cmd(L, L['judge'], L['jhome'], judge_token)
        C, T = q(L['C']), q(L['C'] + '/times.txt')
        return '\n'.join([
            '#!/bin/bash',
            f'# InfraRSI ENV-C attempt {L["C"]} (generated by infrarsi.ext.autoresearch.runner; do not edit)',
            f'mkdir {q(L["C"] + "/.started")} 2>/dev/null || exit 0',
            'now() { date +%s.%N; }',
            contain_snippet(self.train_uid),
            f'echo "launch $(now)" >> {T}',
            f'echo "host $(hostname)" >> {T}',
            f'nvidia-smi --query-gpu=index,name,memory.used,clocks.max.sm,power.limit --format=csv,noheader -i {gpu} | sed "s/^/gpu /" >> {T}',
            *([f'{self.attest_cmd(L)} > {q(L["judge"] + "/code_expected.log")} 2>&1 < /dev/null',
               f'echo "attest_exit $?" >> {T}'] if self.track == 'infra' else []),
            f'echo "train_start $(now)" >> {T}',
            f'{train} > {q(L["C"] + "/train.log")} 2>&1 < /dev/null',
            f'echo "train_exit $?" >> {T}',
            f'echo "train_end $(now)" >> {T}',
            f'if contain; then echo "contain_ok $(now)" >> {T}; else echo "contain_fail $(now)" >> {T}; fi',
            f'if [ {1 if self.track == "infra" else 0} = 1 ] && grep -q "^contain_ok" {T} && [ -f {q(L["out"] + "/final_model.pt")} ] '
            f'&& [ ! -L {q(L["out"] + "/final_model.pt")} ]; then'
            f'  echo "judge_start $(now)" >> {T}',
            f'  {judge} > {q(L["judge"] + "/judge.log")} 2>&1 < /dev/null',
            f'  echo "judge_exit $?" >> {T}',
            f'  echo "judge_end $(now)" >> {T}',
            '  ' + kill_snippet(judge_token),
            *([f'  echo "judge2_start $(now)" >> {T}',
               f'  {self.judge_cmd(L, L["judge2"], L["j2home"], judge_token)} > {q(L["judge2"] + "/judge.log")} 2>&1 < /dev/null',
               f'  echo "judge2_exit $?" >> {T}',
               f'  echo "judge2_end $(now)" >> {T}',
               '  ' + kill_snippet(judge_token)] if self.loader_check else []),
            'fi',
            f'echo "done $(now)" >> {T}',
            f'mv {T} {q(L["C"] + "/DONE.txt")}', ''])

    def materialize(self, L: dict, train_text: str, prepare_variant: str = source.DEFAULT_PREPARE) -> None:
        ops, tf, u = self.ops, judge_files.trusted_files(prepare_variant), self.train_uid
        # <run_id>/ may be shared by several uids (the calibration): root:root 0711, never re-grouped. Below it, root-owned
        # with group = this attempt's uid, 0750: the uid reads its own tree and cannot rename or replace anything root
        # wrote; another uid cannot enter it (Codex W9 B-03). mkdir_domain never re-owns an existing directory.
        ops.mkdirs([L['run']], mode=0o711)
        ops.mkdir_domain([L['cand'], L['C'], L['repo'], L['tools'], L['home'], f'{L["home"]}/.cache',
                          f'{L["home"]}/.cache/autoresearch'], mode=0o750, gid=u)
        ops.mkdirs([L['judge_cache']], mode=0o700)
        ops.mkdirs([L['out'], f'{L["home"]}/tmp', f'{L["home"]}/hf_home', f'{L["home"]}/caches'], mode=0o700, uid=u, gid=u)
        ops.mkdirs([L['judge'], L['jhome'], f'{L["jhome"]}/tmp', f'{L["jhome"]}/.cache/autoresearch'], mode=0o700)
        ops.write_files({f'{L["repo"]}/train.py': train_text, f'{L["repo"]}/prepare.py': tf['repo/prepare.py'],
                         f'{L["tools"]}/run_train.py': tf['tools/run_train.py'],
                         f'{L["tools"]}/code_attest.py': tf['tools/code_attest.py']}, mode=0o444)
        ops.write_files({f'{L["judge"]}/{k.split("/", 1)[1]}': v for k, v in tf.items() if k.startswith('judge/')}, mode=0o400)
        homes = [L['home'], L['jhome']]
        if self.loader_check:                      # second judge: same files, the D-AR-7 prepare.py
            ops.mkdirs([L['judge2'], L['j2home'], f'{L["j2home"]}/tmp', f'{L["j2home"]}/.cache/autoresearch'], mode=0o700)
            ops.write_files({f'{L["judge2"]}/{k}': v for k, v in self.loader_files().items()}, mode=0o400)
            homes.append(L['j2home'])
        links = {k: v for h in homes for k, v in self._home_links(h).items()}
        ops.symlinks(links)
        for home in homes:                         # HF cache ref of the R2 recipe; the nobody home's hub dir must stay writable
            hub = f'{home}/hf_home/hub/models--{source.KERNEL_REPO.replace("/", "--")}'
            mine = home == L['home']
            ops.write_files({f'{hub}/refs/main': source.KERNEL_REV}, mode=0o644, uid=self.train_uid if mine else None,
                            gid=self.train_uid if mine else None)
        ops.sh(f'chown -hR {self.train_uid}:{self.train_uid} {q(L["home"] + "/hf_home")} && chmod 700 {q(L["home"] + "/hf_home")}', timeout=60)

    # ------------------------------------------------------------ one attempt
    def execute(self, task, config, overlay, purpose, run_id, attempt_id, candidate_id, *, wall_cap=None, **kw) -> EnvRun:
        L = self.layout(run_id, candidate_id, attempt_id)
        phases, t0 = {}, self.clock()
        variant = task.raw.get('prepare_variant', source.DEFAULT_PREPARE)
        check_arm_variant(variant, diagnostic=task.raw.get('diagnostic') is True)    # D-AR-7 for every arm (B-04)
        train_timeout = self.train_timeout_for(wall_cap)
        if train_timeout is None:                  # admission should have refused it; nothing is launched
            return EnvRun(candidate_id=candidate_id, judge=None, judge_valid=False, metric_value=None, metric_raw={},
                          judge_reasons=[f'wall cap {wall_cap} s leaves less than {MIN_TRAIN_S} s for the training run'],
                          guards=guards.GuardResult(), phases={})
        try:
            text = self.render(config, overlay)
        except knobs.KnobError as exc:
            return EnvRun(candidate_id=candidate_id, judge=None, judge_valid=False, metric_value=None, metric_raw={},
                          judge_reasons=[f'knobs could not be rendered: {exc}'], guards=guards.GuardResult(), phases={})
        files_sha = {'train.py': sha_text(text)}
        train_token, judge_token, script_token = new_token('isx-ctrain'), new_token('isx-cjudge'), new_token('isx-cscript')
        self.materialize(L, text, variant)
        self.ops.write_files({f'{L["C"]}/run.sh': self.script(L, train_token, judge_token, train_timeout)}, mode=0o700)
        expected = dict(task.frozen_files)
        if self.loader_check:
            expected.update({f'judge2/{k}': source.sha256_text(v) for k, v in self.loader_files().items()})
        frozen = {k: source.node_path(k, L['C']) for k in expected}
        before = self.ops.hash_files(list(frozen.values()))
        phases['materialize'] = round(self.clock() - t0, 3)
        t1 = self.clock()
        deadline = t1 + train_timeout + (self._fixed_s() - MATERIALIZE_S - COLLECT_S)
        done, killed = None, []
        try:
            self.ops.launch(f'{L["C"]}/run.sh', f'{L["C"]}/launch.log', token=script_token)
            while self.clock() < deadline:
                done = self.ops.read_text(f'{L["C"]}/DONE.txt')
                if done is not None:
                    break
                self.sleep(self.poll_s)
        finally:
            # Also when polling failed (Codex W9 B-02): the root script first (it must not start the judge later), the
            # trusted judge by its token, then every process of the arm's uid, verified gone (ContainmentError if not).
            killed = self._stop(script_token if done is None else None, judge_token)
        phases['run'] = round(self.clock() - t1, 3)
        t2 = self.clock()
        after = self.ops.hash_files(list(frozen.values()))
        paths = {'train.log': f'{L["C"]}/train.log', 'judge.json': f'{L["judge"]}/judge.json', 'judge.log': f'{L["judge"]}/judge.log',
                 'code_expected.json': f'{L["judge"]}/code_expected.json'}
        raw = self.ops.read_files(list(paths.values()), max_bytes=LOG_MAX_BYTES)          # root-owned files
        got = {k: (raw[p].decode('utf-8', 'replace') if raw[p] is not None else None) for k, p in paths.items()}
        for k in ('run_meta.json', 'save_error.txt'):                                     # written by the candidate (nobody)
            got[k] = self.ops.read_user_text(f'{L["out"]}/{k}', L['out'], max_bytes=1 << 20)
        info = self.ops.regular_file_info([f'{L["out"]}/final_model.pt', f'{L["C"]}/train.log', f'{L["out"]}/run_meta.json',
                                           f'{L["judge"]}/judge.json'])
        sizes = {p: (v or {}).get('size') for p, v in info.items()}
        ckpt_sha = (info.get(f'{L["out"]}/final_model.pt') or {}).get('sha256')
        if not self.keep_weights:
            self.ops.remove_tree(f'{L["out"]}/final_model.pt', root=self.runs_root)
        self.ops.remove_tree(f'{L["home"]}/caches', root=self.runs_root)
        phases['collect'] = round(self.clock() - t2, 3)
        files_sha['prepare.py'] = source.sha256_text(source.candidate_prepare(variant))
        if self.loader_check:
            raw2 = self.ops.read_files([f'{L["judge2"]}/judge.json'])[f'{L["judge2"]}/judge.json']
            got['judge2.json'] = raw2.decode('utf-8', 'replace') if raw2 is not None else None
        return self.assess(task, L, text, files_sha, done, got, sizes, ckpt_sha, before, after, frozen, phases, killed,
                           expected=expected, train_timeout=train_timeout)

    def _stop(self, script_token: str | None, judge_token: str) -> list:
        """Post-launch cleanup; every step is tried, the uid kill last. Raises ContainmentError when a process of the
        arm's uid survived or the kill could not be done (the caller keeps the GPU leased)."""
        killed, errors = [], []
        for token in ([script_token] if script_token else []) + [judge_token]:
            try:
                killed += self.ops.kill_token(token)
            except Exception as exc:                # noqa: BLE001 - recorded; the uid kill below still runs
                errors.append(f'kill_token {token}: {exc!r}'[:300])
        try:
            res = uid_kill(self.ops, self.train_uid)
            killed += [f'uid{self.train_uid}:{p}' for p in res.get('killed') or []]
        except Exception as exc:                    # noqa: BLE001
            raise ContainmentError(f'uid {self.train_uid}: {exc}; ' + '; '.join(errors)) from exc
        if errors and script_token:                 # the root script may still be alive: do not report success
            raise ContainmentError('; '.join(errors))
        return killed

    def loader_files(self) -> dict:
        """The second judge's files: the trusted judge with the D-AR-7 prepare.py."""
        tf = judge_files.trusted_files('upstream')
        return {'model_ref.py': tf['judge/model_ref.py'], 'judge_bpb.py': tf['judge/judge_bpb.py'],
                'prepare.py': source.candidate_prepare('h2d_sync')}

    def assess(self, task, L, text, files_sha, done, got, sizes, ckpt_sha, before, after, frozen, phases, killed,
               expected: dict | None = None, train_timeout: int | None = None) -> EnvRun:
        times = parse_times(done)
        train_exit = times.get('train_exit')
        timed_out = done is None or train_exit in (124, 137)
        log = got['train.log'] or ''
        summ, rows = logs.summary(log), logs.steps(log)
        judge = _json(got['judge.json'])
        meta = _json(got['run_meta.json'])
        g = guards.GuardResult()
        inv = {v: k for k, v in frozen.items()}
        g.extend(guards.frozen_files(expected or task.frozen_files, {inv[p]: s for p, s in before.items()} if isinstance(before, dict) else None,
                                     {inv[p]: s for p, s in after.items()} if isinstance(after, dict) else None))
        g.extend(guards.exit_ok(train_exit, timed_out=timed_out, what='train.py'))
        train_wall = (times.get('train_end') or 0) - (times.get('train_start') or 0) if times.get('train_end') else None
        g.extend(guards.budget(train_wall, int(train_timeout or self.train_timeout_s) + 30, what='training run'))
        if done is not None and 'contain_ok' not in times:
            g.reject('containment', f'processes of uid {self.train_uid} survived the end of the training; the judge was not run')
        required = {'train.log': sizes.get(f'{L["C"]}/train.log')}
        if self.track == 'infra':
            required.update({'out/final_model.pt': sizes.get(f'{L["out"]}/final_model.pt'), 'out/run_meta.json': sizes.get(f'{L["out"]}/run_meta.json'),
                             'judge/judge.json': sizes.get(f'{L["judge"]}/judge.json')})
            if not timed_out and train_exit == 0:
                g.extend(guards.outputs_present(required, list(required)))
            g.extend(infra_only_check.static(source.base_train(), text))
            if train_exit == 0:
                g.extend(infra_only_check.runtime(meta, judge, self.expected or task.raw['expected_runtime']))
                g.extend(infra_only_check.attestation((meta or {}).get('code_attestation'), _json(got.get('code_expected.json')),
                                                      counters=(meta or {}).get('counters')))
            if rows or train_exit == 0:
                g.extend(infra_only_check.losses(log, self.reference_losses))
        printed = summ.get('val_bpb')
        jv = judge.get('val_bpb') if isinstance(judge, dict) else None
        if self.track == 'infra':
            judge_valid = isinstance(judge, dict) and judge.get('ok') is True and _finite(jv)
            reasons = [] if judge_valid else (list((judge or {}).get('reasons') or []) or
                                              [f'trusted judge did not run (train exit {train_exit}, final_model.pt '
                                               f'{"present" if required.get("out/final_model.pt") else "missing"})'])
            value = jv
        else:
            judge_valid = train_exit == 0 and _finite(printed)
            reasons = [] if judge_valid else [f'no val_bpb printed (train exit {train_exit})']
            value = printed
        dev = abs(printed - jv) if _finite(printed) and _finite(jv) else None
        if self.track == 'infra':
            g.report('printed_vs_judge', val_bpb_printed=printed, val_bpb_judge=jv, abs_dev=dev,
                     red=dev is None or dev > 1e-6)
        if self.loader_check:
            j2 = _json(got.get('judge2.json'))
            jv2 = j2.get('val_bpb') if isinstance(j2, dict) else None
            d2 = abs(jv - jv2) if _finite(jv) and _finite(jv2) else None
            g.report('judge_loader_equivalence', val_bpb_original_loader=jv, val_bpb_h2d_sync_loader=jv2, abs_dev=d2,
                     tolerance=LOADER_TOLERANCE, ok=d2 is not None and d2 <= LOADER_TOLERANCE)
        if killed:
            g.report('leftover_processes', killed=killed)
        steady = logs.steady_stats(rows)
        meta_c = (meta or {}).get('counters') or {}
        mfu = None
        if _finite(meta_c.get('total_training_time')) and meta_c.get('step') and meta_c.get('num_flops_per_token'):
            tbs = ((meta or {}).get('constants') or {}).get('TOTAL_BATCH_SIZE')
            if tbs:
                mfu = 100 * meta_c['num_flops_per_token'] * tbs * (meta_c['step'] - 10) / meta_c['total_training_time'] / 989.5e12
        raw = {'val_bpb_printed': printed, 'val_bpb_judge': jv, 'abs_dev': dev, **{k: summ.get(k) for k in logs.SUMMARY_KEYS if k != 'val_bpb'},
               'median_tok_per_sec': steady.get('median_tok_per_sec'), 'median_dt_ms': steady.get('median_dt_ms'),
               'mfu_recomputed': round(mfu, 3) if mfu else None}
        failed = not judge_valid or train_exit != 0
        summary = {'val_bpb': value, 'val_bpb_printed': printed, 'num_steps': summ.get('num_steps'), 'tok_per_sec': steady.get('median_tok_per_sec'),
                   'mfu_percent': summ.get('mfu_percent'), 'peak_vram_mb': summ.get('peak_vram_mb'),
                   'training_seconds': summ.get('training_seconds'), 'total_seconds': summ.get('total_seconds'),
                   'first_losses': logs.first_losses(log), 'train_exit': train_exit, 'timed_out': timed_out,
                   'checks': {gid: 'ok' if gid not in g.ids() else 'rejected' for gid in
                              (['frozen_files', 'exit_ok', 'outputs_present', 'C-static', 'C-runtime', 'C-loss10'] if self.track == 'infra'
                               else ['frozen_files', 'exit_ok'])}}
        if failed:
            summary['log_tail'] = log[-2500:] if log else (got.get('judge.log') or '')[-2500:]
            if got.get('save_error.txt'):
                summary['save_error'] = got['save_error.txt'][-1500:]
        if isinstance(judge, dict) and not judge.get('ok') and got.get('judge.log'):
            summary['judge_log_tail'] = got['judge.log'][-1500:]
        if times.get('judge_end') and times.get('judge_start'):
            phases['judge'] = round(times['judge_end'] - times['judge_start'], 3)
        if train_wall is not None:
            phases['train'] = round(train_wall, 3)
        dev_ids = ((['D-AR-5'] if self.track == 'infra' else []) + (['D-AR-4'] if self.colocated else [])
                   + (['D-AR-7'] if task.raw.get('prepare_variant') == 'h2d_sync' else []))
        if self.loader_check:
            if times.get('judge2_end') and times.get('judge2_start'):
                phases['judge2'] = round(times['judge2_end'] - times['judge2_start'], 3)
        extra = {'summary': summary, 'files': files_sha, 'candidate_dir': L['C'], 'times': times, 'train_uid': self.train_uid,
                 # AMENDMENT_EXT_007 §4.4: the class of a failed run (cuda_graph / oom / timeout / other; None = exit 0) and,
                 # for C-RND, the frozen knob space every receipt is bound to (outside `summary`: never in an agent digest)
                 'failure_class': logs.failure_class(log, train_exit, timed_out),
                 **({'knob_space_sha256': task.raw['knob_space']['sha256']} if task.raw.get('knob_space') else {}),
                 'train_timeout_s': int(train_timeout or self.train_timeout_s),
                 **({'judge_h2d': _json(got.get('judge2.json'))} if self.loader_check else {}),
                 'outputs': {'final_model.pt': {'bytes': sizes.get(f'{L["out"]}/final_model.pt'), 'sha256': ckpt_sha,
                                                'kept': self.keep_weights},
                             'train.log': {'bytes': sizes.get(f'{L["C"]}/train.log'), 'sha256': sha_text(log) if log else None}},
                 'run_meta': meta, 'track': self.track, 'deviations': dev_ids}
        return EnvRun(candidate_id=L['C'].rsplit('/', 2)[-2], judge=judge, judge_valid=judge_valid, metric_value=value,
                      metric_raw=raw, judge_reasons=reasons, guards=g, phases=phases,
                      candidate_digest=_cand_digest(files_sha), extra=extra)


def parse_times(text: str | None) -> dict:
    out = {}
    for line in (text or '').splitlines():
        key, _, val = line.partition(' ')
        if key in ('gpu', 'host'):
            out[key] = val.strip()
            continue
        try:
            out[key] = int(val) if key.endswith('_exit') else float(val)
        except ValueError:
            out[key] = val.strip()
    return out


def _json(text):
    import json
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return {'__unreadable__': text[:200]}


def _finite(v) -> bool:
    return type(v) in (int, float) and math.isfinite(v) and v > 0


def _cand_digest(files_sha: dict) -> str:
    from ...contracts import digest
    return digest({'files': files_sha})
