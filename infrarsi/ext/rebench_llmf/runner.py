"""RebenchLlmfRunner: the ENV-A EnvRunner (design §2, EXT_INTERFACES §1 owner X2; Codex W9 R1/R2 fixes).

Subclasses `common.env_runner.EnvRunner`: the base holds the node (GpuLease, owner infrarsi-ext, occupancy check),
measures the held GPU-seconds, assembles / signs / stores the `ext-receipt-v1`, keeps `valid = judge_valid and not
guard_rejections` with a null metric when invalid, and -- in a finally, before any lease or claim is released --
kills every process of `domain_uids()` = the arm's own pair {candidate uid, judge uid} (Codex W9 R2 N-01: no uid is
shared between arms, so one arm's cleanup never signals another's), keeping the lease and invalidating the receipt if
one survives. `execute()` renders the candidate (driver-side §2.4 templater), ships the node scorer and
the sanitizer, runs the task's own score.py on the leased node as the authority (candidate code as the arm's
dedicated uid; root never deserialises a candidate artifact), reads back the trusted measurements and runs the ENV-A
guards G1-G9. A tree candidate (A-NAT) is identified by the content digest of its solution tree: G6 rejects a locked
re-score whose restored tree is not the selected snapshot. Admission (Codex H-05): `worst_case_s` is the real
maximum of one screen -- score.py's 480 s finetune + node slack (the node job's hard bound) + the node `timeout`'s kill
grace + the transport waits + lease / shipping / containment -- and an admitted `wall_cap` bounds the node job to
wall_cap - DRIVER_RESERVE_S (a cap below the worst case would shorten score.py's finetune timeout to fit). Driver-side:
never imports torch.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

from ...contracts import digest
from ..common.env_runner import EnvRunner, EnvRun
from ..common.guards import GuardResult
from ..common.remote_exec import assert_uid_unused
from . import guards as guardmod
from . import knobs as knobmod
from . import patch_guard as pg
from . import task as taskmod

SCHEMA_RECEIPT = 'ext-receipt-v1'
_NODE_DIR = Path(__file__).resolve().parent / 'node'
REMOTE_TOOLS = '/data/infrarsi/ext/tools/x2node'
NODE_FILES = ('score_runner.py', 'permute.py', 'sanitize.py', '__init__.py')
SCORE_TIMEOUT_S = 480        # score.py's own finetune timeout (its default; the task protocol), also the D' bound
NODE_SLACK_S = 600           # per scoring beyond its finetune(s): home / snapshot, unprivileged sanitize, installs, norms
KILL_AFTER_S = 60            # the node-side `timeout` sends KILL this long after TERM
SSH_GRACE_S = 70             # our sh() wait past the hard kill (10 s) + NodeOps' transport grace on top of it (60 s)
SETUP_S = 60                 # lease acquire, shipping the scorer, reading the result, containment, release
DRIVER_RESERVE_S = KILL_AFTER_S + SSH_GRACE_S + SETUP_S       # an attempt's wall beyond its node job's hard bound
MIN_SCORE_TIMEOUT_S = 60


def _normalized(t) -> float | None:
    if not (type(t) in (int, float) and math.isfinite(t) and t > 0):
        return None
    return (math.log(taskmod.S0_ANCHOR_S) - math.log(t)) / (math.log(taskmod.S0_ANCHOR_S) - math.log(taskmod.S1_ANCHOR_S))


def render_candidate(base: dict, knobs: dict, overlay: dict) -> tuple[str, str, dict]:
    """(finetune.sh, finetune.yaml, extra_overlay): knobs rendered first, overlay files override. Extra overlay files
    (scripts/**, llmfoundry/**) are written into the candidate solution by the node scorer."""
    sh, yml = base['finetune.sh'], base['finetune.yaml']
    if knobs:
        yml, sh, _ = knobmod.render(yml, sh, knobs)
    extra = {}
    for rel, text in (overlay or {}).items():
        if rel == 'finetune.sh':
            sh = text
        elif rel == 'finetune.yaml':
            yml = text
        else:
            extra[rel] = text
    return sh, yml, extra


def node_bound(job: dict) -> int:
    """Hard wall of one node job: the bound the admitted cap leaves, else every training at score.py's limit plus the
    node slack (a screen: 480 + 600 = 1080 s = worst_case_s - DRIVER_RESERVE_S)."""
    if job.get('bound_s'):
        return int(job['bound_s'])
    trainings = 3 if job.get('run_g4') and not job.get('g4_share') else (2 if job.get('run_g4') else 1)
    return trainings * int(job.get('score_timeout', SCORE_TIMEOUT_S)) + NODE_SLACK_S


def ship_node_files(ops) -> None:
    """Copy the node scorer + sanitizer to REMOTE_TOOLS (root-owned 0644): one file per call, since a node_ext call
    carries its payload on the command line (128 KiB per argument)."""
    ops.mkdirs([REMOTE_TOOLS])
    for name in NODE_FILES:
        ops.write_files({f'{REMOTE_TOOLS}/{name}': (_NODE_DIR / name).read_text(encoding='utf-8')}, mode=0o644)


def winh_scorer(ops, job: dict) -> dict:
    """Default node_runner: ship the scorer + sanitizer via NodeOps, run under the env Python, read back the result.
    Bounded on the node too: on `timeout`'s SIGTERM the scorer kills every uid it used before exiting."""
    ship_node_files(ops)
    return launch_scorer(ops, job)


def launch_scorer(ops, job: dict, *, wait_s: float = 15.0) -> dict:
    """The node scorer for one attempt, launched once: a transport retry of the command after a lost ssh reply finds
    the attempt's marker and, instead of a second launch, waits for the first run's result within its bound."""
    results = f'{job["run_dir"]}/results'
    job_path = f'{results}/{job["attempt_id"]}_job.json'
    ops.mkdirs([results], mode=0o700)
    ops.write_files({job_path: json.dumps(job)}, mode=0o600)
    bound = node_bound(job)
    deadline = time.monotonic() + bound + KILL_AFTER_S + 10
    res = ops.sh(f'mkdir {results}/{job["attempt_id"]}.launched 2>/dev/null || exit 75; '
                 f'timeout --signal=TERM --kill-after={KILL_AFTER_S} {bound} '
                 f'{taskmod.NODE_ENV}/bin/python {REMOTE_TOOLS}/score_runner.py {job_path}', timeout=bound + KILL_AFTER_S + 10)
    payload = ops.read_json(job['out'])
    while payload is None and res.get('code') == 75 and time.monotonic() < deadline:   # the first run still scores
        time.sleep(wait_s)
        payload = ops.read_json(job['out'])
    if not isinstance(payload, dict) or not payload.get('ok'):
        tail = (res.get('stdout', '') + res.get('stderr', ''))[-1200:]
        raise RuntimeError(f'scorer failed: {(payload or {}).get("error")}: {tail}')
    return payload['result']


class RebenchLlmfRunner(EnvRunner):
    env = 'rebench_llmf'
    deviations = taskmod.BASE_DEVIATIONS
    worst_case_s = float(SCORE_TIMEOUT_S + NODE_SLACK_S + DRIVER_RESERVE_S)   # one screen (no G4): 1270 s

    def __init__(self, ops, *, task, gpus=(0, 1, 2, 3), enforce_g7: bool = False, reference_src: str | None = None,
                 g4_share: bool = False, timeout_s: int = 1200, base_solution: dict | None = None, node_runner=None,
                 lease_mode: str = 'own', held_run_id: str | None = None, out_dir=None, clock=None, log=None):
        kw = {} if clock is None else {'clock': clock}
        super().__init__(ops, gpus=gpus, lease_mode=lease_mode, held_run_id=held_run_id, out_dir=out_dir,
                         log=log or (lambda m: None), **kw)
        self.task = task
        self.uid = taskmod.arm_uid(task.arm)
        self.judge_uid = taskmod.judge_uid(task.arm)
        self.enforce_g7 = bool(enforce_g7)
        self.reference_src = reference_src
        self.g4_share = bool(g4_share)
        self.timeout_s = int(timeout_s)
        self.base = base_solution or taskmod.read_base_solution()
        self.node_runner = node_runner or winh_scorer

    def domain_uids(self) -> list[int]:
        """Exactly this arm's pair: its candidate uid and its own judge uid (G4 ref'(D'), base conversion). Both are
        killed and verified gone before the base releases the GPUs; no other arm uses either."""
        return [self.uid, self.judge_uid]

    def describe(self) -> dict:
        return {**super().describe(), 'uid': self.uid, 'judge_uid': self.judge_uid}

    # ------------------------------------------------------------ node jobs
    def _job(self, run_id: str, attempt_id: str, **extra) -> dict:
        return {'run_id': run_id, 'attempt_id': attempt_id, 'node': self.node, 'uid': self.uid, 'judge_uid': self.judge_uid,
                'gpus': ','.join(str(g) for g in self.gpus), 'env_dir': taskmod.NODE_ENV, 'task_dir': taskmod.NODE_TASK,
                'foundry_dir': taskmod.NODE_FOUNDRY, 'model_dir': taskmod.NODE_MODEL, 'model_id': taskmod.BASE_MODEL_ID,
                'model_rev': taskmod.BASE_MODEL_REV, 'run_dir': f'{taskmod.NODE_RUNS}/{run_id}',
                'sanitize_py': f'{REMOTE_TOOLS}/sanitize.py', 'reference_src': self.reference_src,
                'score_timeout': SCORE_TIMEOUT_S,
                'out': f'{taskmod.NODE_RUNS}/{run_id}/results/{attempt_id}_node_result.json', **extra}

    def prepare(self, run_id: str) -> dict:
        """CPU-only, before an arm's GPU work (no lease): refuses to start unless the arm uid and the judge uid are
        unused on the node (no passwd entry, no process: ContainmentError), installs the safetensors-only reference
        and the judge-owned base-model safetensors copy (+ ||ref - base|| for G3), and records the uid check. Both
        uids are killed afterwards (ContainmentError if a process survives)."""
        unused = {str(u): assert_uid_unused(self.ops, u) for u in self.domain_uids()}
        try:
            out = self.node_runner(self.ops, self._job(run_id, 'prepare', mode='prepare'))
        finally:
            self.contain()
        return {**out, 'uids_unused_before': unused}

    # ------------------------------------------------------------ EnvRunner.execute
    def execute(self, task, config: dict, overlay: dict, purpose: str, run_id: str, attempt_id: str,
                candidate_id: str, *, run_g4=None, reuse_home=False, restore_snapshot: str | None = None,
                expected_digest: str | None = None, wall_cap=None, **kw) -> EnvRun:
        knobs, kerrs, kwarns = knobmod.validate(config) if config else ({}, [], [])
        cdig = self._candidate_digest(knobs, overlay)
        bad_paths = [p for p in (overlay or {}) if not pg.canonical_path(p)]
        score_timeout, bound = SCORE_TIMEOUT_S, None
        if wall_cap is not None:                                  # the whole attempt must end within the cap
            bound = int(float(wall_cap) - DRIVER_RESERVE_S)
            score_timeout = int(min(SCORE_TIMEOUT_S, bound - NODE_SLACK_S))
        why = (['invalid knobs: ' + '; '.join(kerrs)] if kerrs else []) + \
              ([f'overlay paths are not canonical relative paths: {bad_paths[:4]}'] if bad_paths else []) + \
              ([f'wall cap {wall_cap} s leaves no room for a scoring'] if score_timeout < MIN_SCORE_TIMEOUT_S else [])
        if why:
            return EnvRun(candidate_id=candidate_id, judge=None, judge_valid=False, metric_value=None, metric_raw={},
                          judge_reasons=why, guards=GuardResult(), phases={}, candidate_digest=cdig)
        do_g4 = bool(run_g4) if run_g4 is not None else (purpose == 'score')
        sh, yml, extra = render_candidate(self.base, knobs, overlay)
        job = self._job(run_id, attempt_id, purpose=purpose, finetune_sh=sh, finetune_yaml=yml,
                        finetune_sh_base=self.base['finetune.sh'], finetune_yaml_base=self.base['finetune.yaml'],
                        overlay=extra, run_g4=do_g4, g4_share=self.g4_share, reuse_home=bool(reuse_home),
                        restore_snapshot=restore_snapshot, expected_digest=expected_digest,
                        clean_slate=(purpose == 'score'), score_timeout=score_timeout, bound_s=bound)
        node = self.node_runner(self.ops, job)
        return self._to_envrun(task, node, candidate_id, cdig, do_g4, kwarns, run_id=run_id,
                               expected_digest=expected_digest, content_identity=bool(reuse_home or restore_snapshot),
                               admission={'wall_cap': wall_cap, 'score_timeout': score_timeout, 'node_bound_s': node_bound(job)})

    def _to_envrun(self, task, node: dict, candidate_id: str, cdig: str, do_g4: bool, warnings, *, run_id: str,
                   expected_digest: str | None = None, content_identity: bool = False, admission=None) -> EnvRun:
        m = guardmod.judge_metrics(node.get('judge'))
        jvalid = guardmod.judge_valid(node.get('judge'))
        exec_time = m.get('execution_time') if guardmod._num(m.get('execution_time')) else None
        guard = guardmod.evaluate(node, taskmod.frozen_expected(task, run_id), enforce_g7=self.enforce_g7,
                                  run_g4=do_g4, g5_expected=taskmod.S0_WORK)
        if node.get('timed_out') or (node.get('exit_code') not in (0, None)):
            guard.reject('exit_ok', f'scoring exit_code={node.get("exit_code")} timed_out={node.get("timed_out")}')
        outputs = node.get('outputs') if isinstance(node.get('outputs'), dict) else {}
        if jvalid and not outputs.get('trained_model'):
            guard.reject('outputs_present', 'trained_model missing after a valid scoring')
        tree = node.get('tree_digest')
        if expected_digest is not None and tree != expected_digest:
            # G6: a locked re-score must score the selected candidate (Codex W9 B-05)
            guard.reject('G6', f'scored tree {str(tree)[:16]} != selected snapshot {str(expected_digest)[:16]}')
        reasons = [] if jvalid else ['score.py: ' + str((node.get('judge') or {}).get('message', 'no finite score'))[:300]]
        phases = node.get('phases') if isinstance(node.get('phases'), dict) else {}
        extra = {'reference': node.get('reference'), 'snapshot': node.get('snapshot'), 'tree_digest': tree,
                 'frozen_before': node.get('frozen_before'), 'frozen_after': node.get('frozen_after'),
                 'env_sh': node.get('env_sh'), 'uid': node.get('uid'), 'kills': node.get('kills'),
                 'remaining_uid_processes': node.get('remaining_uid_processes'), 'clean_slate': node.get('clean_slate'),
                 'base_model_copy': node.get('base'), 'node_cost': node.get('cost'), 'env_versions': node.get('env_versions'),
                 'admission': admission, 'deviations': taskmod.ARM_DEVIATIONS.get(task.arm, ()),
                 'knob_warnings': list(warnings)}
        return EnvRun(
            candidate_id=candidate_id, judge=node.get('judge'), judge_valid=bool(jvalid),
            metric_value=exec_time if jvalid else None,
            metric_raw={'ln_seconds_score': m.get('score'), 'norm_diff': m.get('norm_diff'),
                        'max_scalar_diff': m.get('max_scalar_diff'), 'normalized_locked': _normalized(exec_time)},
            judge_reasons=reasons, guards=guard,
            candidate_digest=tree if (content_identity and tree) else cdig,
            # the base sets phases.total to the whole-execute wall; keep finetune's own total under finetune_total
            phases={**{k: phases.get(k) for k in ('data_prep', 'train', 'convert') if phases.get(k) is not None},
                    **({'finetune_total': phases['total']} if phases.get('total') is not None else {})},
            extra=extra)

    def _candidate_digest(self, knobs: dict, overlay: dict | None) -> str:
        ov = {k: hashlib.sha256(v.encode('utf-8', 'replace')).hexdigest() for k, v in (overlay or {}).items()
              if isinstance(k, str) and isinstance(v, str)}
        return digest({'knobs': knobs, 'overlay': ov})


EnvARunner = RebenchLlmfRunner        # back-compat alias
