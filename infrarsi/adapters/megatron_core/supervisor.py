"""Per-rank supervisor: the torchrun worker itself. CPU only; it never initialises CUDA.

For each phase it starts the trainer as a fresh child process (new interpreter, CUDA context and NCCL
communicators) with the request's whitelisted environment and a private rendezvous port, then polls the
children of all ranks over a gloo group every few seconds: one failed child stops every child at once.
Rank 0 writes ``receipt.json`` from the trainer's parts and the signal files, or an invalid one when the
trainer could not finish. Lifecycle requests run train (+save) -> resume (+export) -> hfcompare (rank 0).
"""
from __future__ import annotations

import datetime
import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from time import monotonic as _monotonic, sleep as _sleep

import torch
import torch.distributed as dist

import infrarsi
from infrarsi.contracts import RunRequest, digest, utc_now
from infrarsi.runtime import receipt_writer as rw
from infrarsi.runtime import lifecycle
from infrarsi.runtime import overlay_loader as l1
from infrarsi.runtime.heartbeat import Heartbeat
from infrarsi.runtime.signals import signals_digest
from infrarsi.adapters.megatron_core import config_map
from infrarsi.adapters.megatron_core.data import DataSource, DataUnavailable
from infrarsi.adapters.megatron_core.config_map import model_dir_for

POLL_S = 5.0
KILL_GRACE_S = 20.0
WALL_GRACE_S = 600.0
RUNNING, DONE, FAILED, DEADLINE = 0, 1, 2, 3


class CodeDigestMismatch(RuntimeError):
    pass


class OverlayDigestMismatch(RuntimeError):
    pass


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}


class Supervisor:
    def __init__(self, request_path: str):
        self.request_path = str(Path(request_path).resolve())
        self.raw_request = json.loads(Path(self.request_path).read_text(encoding='utf-8'))
        self.req = RunRequest.from_dict(self.raw_request)
        self.rank, self.world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
        self.local_world = int(os.environ.get('LOCAL_WORLD_SIZE', '1'))
        self.out = Path(self.req.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.started = _monotonic()
        if self.rank == 0:      # first heartbeat at launch, so phases.total spans first heartbeat -> final receipt
            Heartbeat(self.out, enabled=True).beat('load', 0)
        self.child = None
        self.deterministic = False
        self.problems: list[str] = []

    def phases(self) -> list[str]:
        if self.req.purpose == 'lifecycle':
            life = self.req.lifecycle or {}
            return ['train', 'resume'] + (['hfcompare'] if life.get('forward_compare', True) else [])
        return ['train']

    def gather_states(self, state: int) -> list[int]:
        t = torch.tensor([state], dtype=torch.int64)
        out = [torch.zeros_like(t) for _ in range(self.world)]
        dist.all_gather(out, t)
        return [int(x) for x in out]

    def kill_child(self) -> None:
        if self.child is None or self.child.poll() is not None:
            return
        self.child.terminate()
        try:
            self.child.wait(KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            self.child.kill()
            self.child.wait()

    def run_phase(self, phase: str, plan) -> bool:
        port = torch.tensor([free_port() if self.rank == 0 else 0], dtype=torch.int64)
        dist.broadcast(port, 0)
        self.child, launch_failed = None, False
        if phase != 'hfcompare' or self.rank == 0:
            env = config_map.child_env(dict(os.environ), plan)
            env.update(MASTER_PORT=str(int(port)), TORCHELASTIC_USE_AGENT_STORE='False', HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', INFRARSI_PHASE=phase, PYTHONUNBUFFERED='1')
            if phase != 'hfcompare' and self.deterministic:
                env.update(lifecycle.DETERMINISTIC_ENV)      # opt-in: bit-faithful reference vs resumed run
            if phase == 'hfcompare' and 'CUDA_VISIBLE_DEVICES' not in env:
                env['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(self.local_world))
            cmd = [sys.executable, '-m', 'infrarsi.adapters.megatron_core.generic_train',
                   '--request', self.request_path, '--phase', phase]
            try:
                self.child = subprocess.Popen(cmd, env=env)
            except Exception as exc:      # report through the same collective so no peer is left waiting
                launch_failed = True
                rw.write_json_atomic(self.out / 'failures' / f'{phase}_rank{self.rank}.json',
                                     {'rank': self.rank, 'phase': phase, 'kind': 'launch', 'error': repr(exc)[:2000],
                                      'step': None, 'utc': utc_now()})
        deadline = self.req.wall_seconds_cap + WALL_GRACE_S
        while True:
            code = None if self.child is None else self.child.poll()
            state = FAILED if launch_failed else DONE if self.child is None or code == 0                 else RUNNING if code is None else FAILED
            if self.rank == 0 and state != FAILED and _monotonic() - self.started > deadline:
                state = DEADLINE
            states = self.gather_states(state)
            if any(s in (FAILED, DEADLINE) for s in states):
                self.kill_child()
                failed = [r for r, s in enumerate(states) if s == FAILED]
                self.problems.append(f'phase {phase} failed on ranks {failed}' if failed else
                                     f'phase {phase} exceeded wall_seconds_cap + {WALL_GRACE_S:.0f}s')
                return False
            if all(s == DONE for s in states):
                return True
            _sleep(POLL_S)

    def lifecycle_block(self) -> tuple[dict, list[str]]:
        life = self.out / 'lifecycle'
        resume = _read_json(life / 'resume_parts.json').get('lifecycle', {})
        compare = _read_json(life / 'forward_compare.json')
        save = _read_json(life / 'save.json')
        block = {'saved_at': save.get('step'), 'resumed_ok': bool(resume.get('resumed_ok')),
                 'resume_scalars_match': bool(resume.get('resume_scalars_match')),
                 'export_ok': bool(resume.get('export_ok')),
                 'forward_max_abs_logit_diff': compare.get('max_abs_logit_diff'),
                 'forward_ok': bool(compare.get('forward_ok')), 'checkpoint': save,
                 'continuation': resume.get('continuation'), 'export': resume.get('export'), 'forward': compare,
                 **{k: resume.get(k) for k in ('state_exact', 'state_digest_mismatch', 'optimizer_steps_restored',
                                               'resume_tokens_match', 'deterministic', 'deterministic_unavailable',
                                               'deterministic_overrides', 'resumed_bitwise_updates',
                                               'attention_backend', 'state_digest_saved', 'state_digest_restored',
                                               'self_check')}}
        block['signals_resume_digest'] = signals_digest(self.out / 'signals_resume')      # rank-ordered concat
        need = [k for k in ('resumed_ok', 'resume_scalars_match', 'export_ok', 'forward_ok') if not block[k]]
        if not (self.req.lifecycle or {}).get('forward_compare', True):
            need = [k for k in need if k != 'forward_ok']
        if not (self.req.lifecycle or {}).get('export', True):
            need = [k for k in need if k != 'export_ok']
        return block, [f'lifecycle: {k} is false' for k in need]

    def receipt(self, ok: bool) -> dict:
        parts = _read_json(self.out / rw.PARTS_FILE)
        problems = list(self.problems)
        if not parts.get('final'):
            problems.append('trainer did not finish (no final receipt parts)')
        events = parts.setdefault('events', {})
        events.setdefault('exceptions', [])
        for path in sorted((self.out / 'failures').glob('*.json')):
            rec = _read_json(path)
            if rec and not any(e.get('rank') == rec.get('rank') and e.get('phase') == rec.get('phase')
                               for e in events['exceptions'] if isinstance(e, dict)):
                events['exceptions'].append({k: rec.get(k) for k in ('rank', 'phase', 'kind', 'error', 'step')})
                if rec.get('kind') == 'nccl':
                    events['nccl_errors'] = events.get('nccl_errors', 0) + 1
        if getattr(self, 'mismatch_detail', None):
            events['code_digest_detail'] = self.mismatch_detail
        if events.get('stopped'):
            problems.append(events['stopped'])
        if self.req.purpose == 'lifecycle':
            block, life_problems = self.lifecycle_block()
            parts['lifecycle'] = block
            problems += life_problems
            resume = _read_json(self.out / 'lifecycle' / 'resume_parts.json')
            problems += [f'resume: {p}' for p in resume.get('problems', [])]
            for k, v in (resume.get('phases') or {}).items():
                parts.setdefault('phases', {})[f'resume_{k}'] = v
        receipt = rw.assemble(self.req, self.out / 'signals', parts, extra_problems=problems)
        receipt.update(self.request_identity())
        # phases.total is the whole launch (all phase processes), which the harness checks against the node
        # wrapper's wall clock; the train process's own total stays available as trainer_total.
        phases = receipt['phases']
        if 'total' in phases:
            phases['trainer_total'] = phases.pop('total')
        phases['total'] = _monotonic() - self.started
        if not ok:
            receipt['valid'] = False
        return receipt

    def request_identity(self) -> dict:
        """Echo of the request this process loaded, for the driver's launch ledger."""
        raw = self.raw_request
        return {'request_digest': digest(raw), 'numerics_role': raw.get('numerics_role'),
                'data_manifest': raw.get('data_manifest'), 'effective_overlay_digest': raw.get('effective_overlay_digest')}

    def updates_needed(self) -> int:
        if self.req.purpose == 'lifecycle':
            life = self.req.lifecycle or {}
            return int(life.get('save_at', self.req.steps)) + int(
                life.get('resume_steps', self.req.task['workload'].get('resume_continuation_steps', 10)))
        return int(self.req.steps)

    def preflight(self):
        """Request, layout and data checks before any GPU work; a failure becomes an invalid receipt, not a crash."""
        req, life = self.req, self.req.lifecycle or {}
        if req.purpose in ('score', 'lifecycle') and req.input_plan != 'B':
            raise config_map.ConfigError(f'{req.purpose} runs use input plan B (design 1.2), got {req.input_plan}')
        if req.purpose == 'lifecycle':
            if int(life.get('save_at', req.steps)) < 1 or int(life.get('resume_steps', 1)) < 1:
                raise config_map.ConfigError(f'lifecycle save_at/resume_steps must be positive: {life}')
            if life.get('forward_compare', True) and not life.get('export', True):
                raise config_map.ConfigError('lifecycle forward_compare needs export')
        mismatch = config_map.code_digest_problem(self.req.code_digest, Path(infrarsi.__file__).resolve().parent)
        if mismatch:
            raise CodeDigestMismatch(mismatch)
        want = getattr(req, 'effective_overlay_digest', None)
        if want:
            got = l1.effective_digest(req.overlay)
            if got != want:
                raise OverlayDigestMismatch(f'effective overlay digest {got} != request {want}')
        facts = config_map.model_facts(model_dir_for(self.req.task))
        plan = config_map.make_plan(self.req, facts)
        self.deterministic = config_map.deterministic_lifecycle(self.req, facts)
        source = DataSource(self.req.task, self.req.input_plan, facts['vocab_size'],
                            data_manifest=getattr(self.req, 'data_manifest', None))
        if self.updates_needed() > source.max_updates:
            raise DataUnavailable(f'{self.updates_needed()} updates requested, data holds {source.max_updates}')
        if self.rank == 0:
            source.verify_payload()          # sha256 of every packed binary against its manifest (once per run)
        return plan

    def main(self) -> int:
        dist.init_process_group('gloo', timeout=datetime.timedelta(minutes=60))
        ok, plan, problem = True, None, None
        try:
            plan = self.preflight()
        except CodeDigestMismatch as exc:
            problem = 'code_digest_mismatch'
            self.mismatch_detail = str(exc)
        except OverlayDigestMismatch as exc:
            problem = 'overlay_digest_mismatch'
            self.mismatch_detail = str(exc)
        except Exception as exc:          # invalid layout or data: no GPU work, invalid receipt
            problem = f'preflight: {type(exc).__name__}: {exc}'
        flag = torch.tensor([0 if problem is None else 1], dtype=torch.int64)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if int(flag):
            ok = False
            self.problems.append(problem or 'preflight failed on another rank')
        if self.rank == 0:
            provisional = rw.fallback(self.req, self.out, 'provisional receipt: run in progress or killed')
            provisional.update(self.request_identity())
            rw.write_json_atomic(self.out / 'receipt.json', provisional)
        try:
            if ok:
                for phase in self.phases():
                    if not self.run_phase(phase, plan):
                        ok = False
                        break
        finally:
            self.kill_child()
            if self.rank == 0:
                receipt = self.receipt(ok)
                rw.write_json_atomic(self.out / 'receipt.json', receipt)
                print(f'INFRARSI_RECEIPT valid={receipt["valid"]} steps={receipt["steps_completed"]} '
                      f'reasons={receipt["invalid_reasons"][:5]}', flush=True)
                life = receipt.get('lifecycle') or {}
                if (self.req.purpose == 'lifecycle' and receipt['valid']
                        and not (self.req.lifecycle or {}).get('keep_artifacts')):
                    removed = lifecycle.remove_artifacts([self.out / 'ckpt', self.out / 'hf_export',
                                                          self.out / 'lifecycle' / 'mcore_logits.pt'])
                    rw.write_json_atomic(self.out / 'lifecycle' / 'retention.json',
                                         {'removed': removed, 'utc': utc_now(), 'checkpoint': life.get('checkpoint')})
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass
        return 0 if ok else 1


def main(request_path: str) -> int:
    sup = Supervisor(request_path)

    def on_term(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, on_term)
    return sup.main()
