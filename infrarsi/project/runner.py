"""ProjectRunner: one fresh run of a candidate of a project (an agent.runner.ScreenRunner).

    run(task, config, overlay, purpose, run_id, candidate_id=..., steps=..., extra_args=..., stop=...) -> receipt

The candidate tree is a copy of the workdir (minus `code.exclude`) with the candidate's patched files written in and
its config rendered (path knobs into the config file, flag knobs into {args}, env knobs into the environment). The
run's stamped output becomes the `evidence` block (project/signals.py) of an `infrarsi-project-receipt-v1`, signed
on this machine; the verifier (infrarsi.verify) decides validity from it.
Command placeholders: {python} {config} {args} {steps} {gpus} {nodes} {run_dir} {ckpt_dir} {save_at}, and on SSH
hosts {node_rank} {nnodes} {master_addr}. kw: candidate_id, steps (default workload.steps), window, extra_args
(appended to {args}, e.g. the checkpoint save/resume arguments), ckpt_dir (shared by a save and a resume run),
wall_cap (seconds), stop (early-stop predicate over the stamped lines).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import time
from pathlib import Path

import yaml

from ..contracts import InfrastructureError, digest, new_attempt_id, utc_now
from ..harness import signing
from ..verify import SCHEMA
from . import executor as ex
from . import linkbench
from . import signals as sig
from .manifest import Project, args_for, env_for, set_path, write_const
from .probe import ENV as PROBE_ENV

TOKEN = re.compile(r'''(["'])\{(config|run_dir|ckpt_dir)\}\1|\{(\w+)\}''')
PROFHOOK = '.infrarsi_profhook'         # the phase profile's sitecustomize, staged into the run tree (A11)
PROFHOOK_SRC = Path(__file__).with_name('profhook') / 'sitecustomize.py'
PROFILE_WARMUP, PROFILE_UPDATES = 5, 3    # updates before the profiled ones; updates profiled
MEMPEAK = 'mempeak'                       # D3: the run's directory of its processes' peak memory files (the same hook)
MODULES = 'modules'                       # D7: the reference run's directory of the modules its first processes imported
MEMPEAK_RE = re.compile(r'^mempeak\s+(\d+(?:\.\d+)?)\s*$', re.M)
LINKBENCH, LINKBENCH_PORT = '_infrarsi_linkbench.py', 29561
FATAL_GRACE_S = 120.0                     # a run that logged a fatal line and made no update for this long is stopped


def fatal_grace_s() -> float:
    """FATAL_GRACE_S, or INFRARSI_FATAL_GRACE_S when set (0: no fatal-line stop)."""
    try:
        return float(os.environ.get('INFRARSI_FATAL_GRACE_S', FATAL_GRACE_S))
    except ValueError:
        return FATAL_GRACE_S


class FatalWatch:
    """Stop a run that logged a fatal line (signals.FATAL_RE) when no update has arrived for `grace_s` since the later
    of that line and the last update. Such a run otherwise waits for the job's own timeout (2026-10-28: 35-60 min per
    failed candidate: output held open by leftover processes after torchrun had exited, or ranks silent after an NCCL
    error) instead of the minute or two of a run that ends when it fails. The executors pass each node's whole line list on every check: a node's list is
    known by its first line and read once; time is this machine's monotonic clock (remote stamps may be skewed). An update
    on any node counts as progress (Megatron logs its updates on the last rank)."""

    def __init__(self, parser, grace_s: float, clock=time.monotonic, ignore: str | None = None):
        self.parser, self.grace_s, self.clock = parser, float(grace_s), clock
        self.ignore = re.compile(ignore) if ignore else None
        self.seen: dict = {}                  # a node's first line -> how many of its lines were read
        self.progress_t = None                # when the last update line arrived
        self.fatal_t = None                   # when the first fatal line arrived
        self.line = None
        self.fired = False

    def __call__(self, lines, node=None) -> bool:
        if self.fired:
            return True
        if lines:
            key = node if node is not None else lines[0]   # the executor's node index (two nodes' first lines can match)
            new = lines[self.seen.get(key, 0):]
            self.seen[key] = len(lines)
            if new:
                now = self.clock()
                if self.parser.parse(new):
                    self.progress_t = now
                if self.fatal_t is None:
                    for _, text in new:
                        t = sig.plain(text)
                        if sig.FATAL_RE.search(t) and not (self.ignore and self.ignore.search(t)):
                            self.fatal_t, self.line = now, t.strip()[:300]
                            break
        if self.fatal_t is not None and self.clock() - max(self.fatal_t, self.progress_t or 0.0) >= self.grace_s:
            self.fired = True
        return self.fired


def _either(*fns):
    """One stop predicate from several (None skipped): stop when any says so; each is asked on every call, so that a
    stateful one (FatalWatch) reads every line."""
    fs = [f for f in fns if f is not None]
    if len(fs) < 2:
        return fs[0] if fs else None

    takes = [ex.takes_node(f) for f in fs]

    def stop(lines, node=None):
        hit = False
        for f, t in zip(fs, takes):
            hit = bool(f(lines, node=node) if t else f(lines)) or hit
        return hit
    return stop


EVAL_RE = re.compile(r'(?i)(?<![\w])(?P<k>eval[_ /]?loss|val(?:idation)?[_ /]?loss|loss)[\'"]?\s*[:=]?\s*(?P<v>' + sig.NUM + ')')


def fill(template: str, values: dict, raw: dict | None = None) -> str:
    """Replace known {name} placeholders only (braces of the user's command stay as they are). Path values come quoted
    for the shell; a path placeholder the template quotes itself ("{config}", '{run_dir}': a shell argument or a string
    inside an embedded program) gets the raw value, so the template's quoting stays as written."""
    def one(m):
        if m.group(2):
            name, q = m.group(2), m.group(1)
            if raw is not None and name in raw:
                return q + str(raw[name]) + q
            return q + str(values[name]) + q if name in values else m.group(0)
        return str(values[m.group(3)]) if m.group(3) in values else m.group(0)
    return TOKEN.sub(one, template)


class ProjectRunner:
    stop_grace_s = 0.0
    memory_peaks = True                     # D3: every run's processes keep their peak memory (observation only)
    infra_retry_waits = (15.0, 60.0)        # a transport failure is tried again after these pauses, then raised
    own_wait_s, own_poll_s = 300.0, 10.0    # how long the job's GPUs may take to be free of the run's own processes

    def __init__(self, project: Project, *, out_dir, executor=None, log=None, keep_trees: bool = False, skip=None):
        self.project = project
        self.out_dir = Path(out_dir)
        try:                                      # an output directory inside the project (--out runs) is not copied
            root, s = project.root.resolve(), Path(skip).resolve() if skip else None
            if s == root:                         # --out .: this run's own directory
                s = root / self.out_dir.resolve().relative_to(root).parts[0]
            self.skip = s.relative_to(root).as_posix() if s else None
        except (ValueError, IndexError):
            self.skip = None
        self.executor = executor or ex.make(project.raw['execution'])
        self.log = log or (lambda msg: None)
        self.keep_trees = keep_trees
        self.attempts: list[dict] = []
        self.link: dict = {}                      # node 0's fabric as the last reference / doctor run's shell saw it

    def describe(self) -> dict:
        return {'kind': type(self).__name__, 'project': self.project.name, 'executor': self.executor.describe()}

    # ------------------------------------------------------------ candidate tree
    def materialize(self, tree: Path, config: dict, overlay: dict) -> str | None:
        """Copy the workdir, write the patched files, render the config file; returns its relative path (or None)."""
        p = self.project
        skip = self.skip
        ex.copy_tree(p.root, tree, (lambda r: r == skip or r.startswith(skip + '/') or p.excluded(r)) if skip else p.excluded)
        for rel, text in (overlay or {}).items():
            f = ex.own_file(tree, rel)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text, encoding='utf-8')
        consts = {}
        for name, k in p.knobs.items():
            if 'const' in k and config.get(name) is not None:
                consts.setdefault(k['file'], []).append((k['const'], config[name]))
        for rel_file, items in consts.items():
            f = ex.own_file(tree, rel_file)
            text = f.read_text(encoding='utf-8')
            for const, value in items:
                text = write_const(text, const, value)
            f.write_text(text, encoding='utf-8')
        cf = p.config_file
        if cf is None:
            return None
        rel = cf.relative_to(p.root).as_posix()
        if any('path' in k for k in p.knobs.values()):
            doc = p.base_config() or {}
            for name, k in p.knobs.items():
                if 'path' in k and config.get(name) is not None:
                    set_path(doc, k['path'], config[name])
            text = json.dumps(doc, indent=1) if cf.suffix.lower() == '.json' else yaml.safe_dump(doc, sort_keys=False)
            ex.own_file(tree, rel).write_text(text, encoding='utf-8')
        return rel

    def _stamper(self) -> dict:
        """The interpreter that runs the line stamper on an SSH host (the manifest's execution.python)."""
        return {'python': self.executor.python(self.project.raw['execution'].get('python'))} if self.executor.kind == 'ssh' else {}

    def shared_dir(self, run_id: str, name: str) -> str:
        """A directory several runs of one promotion share (checkpoints), in the executor's namespace."""
        if self.executor.kind == 'ssh':
            return f'{self.executor.root()}/{run_id}/{name}'
        return str(self.out_dir / run_id / name)

    def _values(self, *, cfg_rel: str | None, steps: int, run_dir: str, ckpt_dir: str | None = None) -> tuple[dict, dict]:
        p = self.project
        q = self.executor.quote
        # bare path placeholders are quoted here (only when needed); one the template quotes itself gets the raw value
        raw = {'config': cfg_rel or '', 'run_dir': run_dir, 'ckpt_dir': ckpt_dir or f'{run_dir}/ckpt'}
        values = {'python': q(self.executor.python(p.raw['execution'].get('python'))), 'config': q(cfg_rel) if cfg_rel else '',
                  'steps': steps, 'gpus': p.gpus, 'nodes': p.nodes, 'run_dir': q(run_dir), 'ckpt_dir': q(raw['ckpt_dir']),
                  'save_at': p.save_at or max(1, steps // 2)}    # the project's rule; without a checkpoint block half this run
        return values, raw

    def setup_line(self, *, cfg_rel: str | None, steps: int, run_dir: str, ckpt_dir: str | None = None) -> str:
        """launch.setup rendered as the job's command renders it ('' without one)."""
        setup = self.project.raw['launch'].get('setup')
        if not setup:
            return ''
        values, raw = self._values(cfg_rel=cfg_rel, steps=steps, run_dir=run_dir, ckpt_dir=ckpt_dir)
        return fill(setup, values, raw)

    def command(self, config: dict, *, cfg_rel: str | None, steps: int, run_dir: str, extra_args: str = '',
                ckpt_dir: str | None = None) -> str:
        p = self.project
        q = self.executor.quote
        flags = ' '.join(q(a) for a in args_for(p, config))
        values, raw = self._values(cfg_rel=cfg_rel, steps=steps, run_dir=run_dir, ckpt_dir=ckpt_dir)
        cmd = p.raw['launch']['command']
        args = ' '.join(x for x in (flags, fill(extra_args, values, raw)) if x)
        cmd = fill(cmd, {**values, 'args': args}, raw) if '{args}' in cmd else f'{fill(cmd, values, raw)} {args}'.strip()
        setup = self.setup_line(cfg_rel=cfg_rel, steps=steps, run_dir=run_dir, ckpt_dir=ckpt_dir)
        return f'{setup} && {cmd}' if setup else cmd

    def probe_links(self, *, cwd: str, env: dict, setup: str) -> None:
        """Node 0's fabric where the job runs (its tree, its environment), by a short probe apart from the run: the job's
        command, its timeout and its accounting stay as they are. launch.setup is not replayed (it may change the staged
        tree): the NCCL variables it assigns literally count over the environment's; a fabric the setup chooses another
        way needs hardware.interconnect."""
        probe = getattr(self.executor, 'probe', None)
        if probe is None:
            return
        try:
            text = probe(f'( {ex.LINK_QUERY} )', cwd=cwd, env={**env, **ex.setup_nccl(setup)})
        except Exception:                                  # noqa: BLE001 - a look at the fabric never fails a run
            return
        if text and '__uverbs__' in text:
            self.link = ex.parse_link(text, nodes=self.project.nodes)

    def can_profile(self) -> bool:
        """The profile run prefixes the job's command with a POSIX `export`: ssh hosts or a local POSIX shell (not
        Windows, not an srun prefix)."""
        return self.executor.kind == 'ssh' or (os.name != 'nt' and not getattr(self.executor, 'prefix', ''))

    def bench_links(self, *, rel: str, adir: Path, env: dict, setup: str) -> None:
        """A multi-node job's all-reduce bus bandwidth (linkbench.py) after its reference / doctor run: on its hosts,
        launched with torch.distributed.run over every GPU, in the job's environment and the NCCL variables launch.setup
        assigns literally (the setup is not replayed); the job's run and accounting are not touched, a failure gives no
        number. Only over ssh (an srun allocation launches its own way)."""
        if self.executor.kind != 'ssh':
            return
        devices = ex.setup_devices(setup)          # the GPUs launch.setup selects, if it names them literally
        if devices is False:                       # selected some other way: the benchmark could take other GPUs
            self.log('link benchmark skipped: launch.setup selects the GPUs in a way InfraRSI cannot read')
            return
        d = adir / 'linkbench'
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(linkbench.__file__), d / LINKBENCH)
        try:
            cwd = self.executor.place(d, f'{rel}/linkbench', timeout_s=120)
            py = self.executor.quote(self.executor.python(self.project.raw['execution'].get('python')))
            cmd = (f'{py} -m torch.distributed.run --nnodes {{nnodes}} --node_rank {{node_rank}} --master_addr '
                   f'{{master_addr}} --master_port {LINKBENCH_PORT} --nproc_per_node {self.project.gpus} {LINKBENCH}')
            res = self.executor.run(cmd, cwd=cwd, env={**env, **ex.setup_nccl(setup, every=True), **(devices or {})},
                                    timeout_s=300, log_path=d / 'linkbench.log', **self._stamper())
        except Exception:                                  # noqa: BLE001 - a bandwidth number never fails a run
            return
        found = linkbench.parse(t for _, t in res['lines'])
        if found:
            self.link['bandwidth_gbs'] = found['busbw_gbs']
        elif any(linkbench.BUSY in t for _, t in res['lines'] + [x for n in res.get('node_lines') or [] for x in n]):
            self.log('link benchmark stopped: a GPU it was given holds another process (no bandwidth number)')

    # ------------------------------------------------------------ ScreenRunner
    def run(self, task, config: dict, overlay: dict | None, purpose: str, run_id: str, **kw) -> dict:
        p = self.project
        attempt_id = new_attempt_id()
        cid = str(kw.get('candidate_id') or attempt_id)
        steps = int(kw.get('steps') or p.steps)   # the horizon the command gets ({steps})
        k = int(kw['stop_after']) if kw.get('stop_after') else None   # a probe / short run: stopped after k updates
        adir = self.out_dir / run_id / f'{cid}-{purpose}-{attempt_id[:8]}'
        tree = adir / 'tree'
        t0, utc_start = time.perf_counter(), utc_now()
        config = dict(config or {})
        cfg_rel = self.materialize(tree, config, overlay or {})
        peaks = self.memory_peaks and not os.environ.get('INFRARSI_NO_MEMPEAK')
        if kw.get('profile') or (peaks and self.executor.kind == 'ssh'):            # the hook travels with the tree (A11's phase profile; D3's peaks)
            (tree / PROFHOOK).mkdir(exist_ok=True)
            shutil.copyfile(PROFHOOK_SRC, tree / PROFHOOK / 'sitecustomize.py')
        frozen_before = p.frozen_hashes(tree)
        t1 = time.perf_counter()
        rel = adir.relative_to(self.out_dir).as_posix()
        waits = list(self.infra_retry_waits)
        tries, lost_s = 0, 0.0                    # lost_s: wall time of executions the transport cut off (GPUs may have run)
        probe_s = 0.0                             # the link probes' wall time: not the job's
        before = None                             # other jobs on the GPUs before / after: a contaminated measurement
        cut_off = False                           # an execution the transport cut off (not a failed copy): it may run on
        watch = None                              # the fatal-line watch of the last execution
        peak_dir = None                           # D3: the last execution's directory of its processes' peak files
        while True:
            t_exec = None                             # the execution's own start: the look-ups before it are not the job's
            try:
                cwd = self.executor.place(tree, f'{rel}/tree')
                run_dir = cwd.rsplit('/tree', 1)[0] if cwd.endswith('/tree') else str(adir)
                env = env_for(p, config)
                probe_path = None
                if p.raw['signals']['source'] == 'probe':     # a retry writes its own file: the probe appends
                    probe_path = f'{run_dir}/probe.jsonl' if not tries else f'{run_dir}/probe-retry{tries}.jsonl'
                    env[PROBE_ENV] = probe_path
                cmd = self.command(config, cfg_rel=cfg_rel, steps=steps, run_dir=run_dir,
                                   extra_args=kw.get('extra_args') or '', ckpt_dir=kw.get('ckpt_dir'))
                setup = self.setup_line(cfg_rel=cfg_rel, steps=steps, run_dir=run_dir, ckpt_dir=kw.get('ckpt_dir'))
                sel = ex.setup_devices(setup)         # the job's GPUs: launch.setup's literal choice, else its env's
                vis = (sel or {}).get('CUDA_VISIBLE_DEVICES', env.get('CUDA_VISIBLE_DEVICES'))
                look = sel is not False               # chosen in a way InfraRSI cannot read: no look at the GPUs
                if kw.get('profile'):                 # rank 0 profiles a few steady updates: the hook first on PYTHONPATH
                    if self.executor.kind != 'ssh':   # a local run: absolute paths (the job runs inside the tree)
                        cwd, run_dir = str(Path(cwd).resolve()), str(Path(run_dir).resolve())
                    env.update({'INFRARSI_PROFILE': f'{run_dir}/profile',
                                'INFRARSI_PROFILE_WARMUP': str(kw.get('profile_warmup', PROFILE_WARMUP)),
                                'INFRARSI_PROFILE_UPDATES': str(kw.get('profile_updates', PROFILE_UPDATES))})
                    hook = self.executor.quote(f'{cwd}/{PROFHOOK}')
                    cmd = f'export PYTHONPATH={hook}"${{PYTHONPATH:+:$PYTHONPATH}}" && {cmd}'
                if peaks:                             # D3: every process keeps its peak memory (the hook, first on the path)
                    mods = None                       # D7: the reference run's imports (the library source a strategy may show)
                    if purpose in ('probe', 'short') or k:   # (stopped by a signal after a few updates: the hook reads
                        env['INFRARSI_MEMPEAK_FAST'] = '1'   # at its fast cadence throughout - Codex HLS5 r4)
                    sub = f'exec-{tries}'             # (each execution its own directories: a transport retry's earlier
                    if self.executor.kind == 'ssh':   # execution leaves its files apart - Codex HLS5 r1)
                        env['INFRARSI_MEMPEAK'] = peak_dir = f'{run_dir}/{MEMPEAK}/{sub}'
                        if purpose == 'reference':
                            env['INFRARSI_MODULES'] = mods = f'{run_dir}/{MODULES}/{sub}'
                        if not kw.get('profile'):
                            hook = self.executor.quote(f'{cwd}/{PROFHOOK}')
                            cmd = f'export PYTHONPATH={hook}"${{PYTHONPATH:+:$PYTHONPATH}}" && {cmd}'
                    else:                             # a local run: absolute paths, the environment (any shell)
                        env['INFRARSI_MEMPEAK'] = peak_dir = str(Path(run_dir).resolve() / MEMPEAK / sub)
                        if purpose == 'reference':
                            env['INFRARSI_MODULES'] = mods = str(Path(run_dir).resolve() / MODULES / sub)
                        if not kw.get('profile'):
                            have = env.get('PYTHONPATH', os.environ.get('PYTHONPATH', ''))
                            hook_dir = (Path(cwd).resolve() / PROFHOOK) if (Path(cwd) / PROFHOOK).is_dir() else PROFHOOK_SRC.parent.resolve()
                            env['PYTHONPATH'] = os.pathsep.join([str(hook_dir)] + ([have] if have else []))
                if purpose in ('reference', 'doctor'):
                    p0 = time.perf_counter()
                    self.probe_links(cwd=cwd, env=env, setup=setup)
                    probe_s += time.perf_counter() - p0
                preds = [f for f in (kw.get('stop'), (lambda rows: rows[-1]['step'] >= k) if k else None) if f]
                stop = self._row_stop((lambda rows: any(f(rows) for f in preds)) if preds else None, probe_path)
                watch = self._fatal_watch(probe_path)
                stop = _either(stop, watch)
                self.log(f'{run_id}/{cid}: {purpose} -> {cmd}')
                before = self.gpu_state(vis) if look else None   # after the copy, right before this attempt starts
                if cut_off and ex.gpu_contention(before, 'before'):   # the attempt the transport cut off may still run
                    before = self._until_free(vis, before, f'{run_id}/{cid}: {purpose}: the cut-off attempt')
                t_exec = time.perf_counter()
                try:
                    res = self.executor.run(cmd, cwd=cwd, env=env,
                                            timeout_s=float(kw.get('wall_cap') or p.raw['launch']['timeout_s']),
                                            log_path=adir / 'train.log', stop=stop, **self._stamper())
                except InfrastructureError:
                    raise
                except Exception as exc:                          # noqa: BLE001 - recorded as a failed run
                    res = {'exit_code': None, 'timed_out': False, 'stopped': False, 'wall_s': time.perf_counter() - t_exec,
                           'lines': [(time.perf_counter(), f'runner error: {type(exc).__name__}: {exc}')]}
                break
            except InfrastructureError as exc:                    # the hosts, not the candidate: try again, then raise
                if not waits:
                    raise
                tries, lost_s = tries + 1, lost_s + (time.perf_counter() - t_exec if t_exec is not None else 0.0)
                cut_off = cut_off or t_exec is not None
                wait = waits.pop(0)                       # (a console milestone, U-6-3: the wait is minutes long)
                self.log(f'[infrarsi] {run_id}/{cid}: {purpose}: {exc}; trying again in {wait:.0f} s')
                time.sleep(wait)
        t2 = time.perf_counter()
        peak = self.memory_peak(peak_dir) if peaks and peak_dir else None   # D3: after the run, apart from its
        #                                         time - the files of the execution the receipt describes, the last one
        if peak is None and peaks and peak_dir:     # stopped, os._exit, killed from outside: no process wrote a value
            self.log(f'{run_id}/{cid}: {purpose}: peak memory unknown - no process wrote it (a full run reads it at '
                     'its interpreter exit only)')
        if watch is not None and watch.fired:
            res['fatal_stop'] = {'line': watch.line, 'grace_s': watch.grace_s}
        after = settled = self.gpu_state(vis) if look else None
        if ex.gpu_contention(after, 'after') and not ex.gpu_contention(before, 'before'):
            if res.get('stopped') or res.get('timed_out'):   # a run this runner stopped: its processes end late
                settled = self._until_free(vis, after, f'{run_id}/{cid}: {purpose}: the stopped run')
            else:
                time.sleep(3)                     # utilisation covers the last second (ours) and memory frees late: look again
                settled = self.gpu_state(vis)
        gpu_warn = ex.gpu_contention(before, 'before') + ex.gpu_contention(settled, 'after')
        ecc = ex.ecc_errors(before, after)        # the first look after the run: a failed second look does not hide it
        if gpu_warn or ecc:
            self.log(f'{run_id}/{cid}: ' + '; '.join((['GPUs not free: ' + '; '.join(gpu_warn)] if gpu_warn else []) + ecc))
        frozen_after = p.frozen_hashes(tree) if isinstance(self.executor, ex.LocalExecutor) else frozen_before
        expected = task.frozen_files if hasattr(task, 'frozen_files') else p.frozen_hashes()
        frozen = {'checked': bool(expected), 'n': len(expected),
                  'changed': sorted(k for k in expected if frozen_before.get(k) != expected[k] or frozen_after.get(k) != expected[k]),
                  'missing': sorted(k for k in expected if k not in frozen_after)}
        megatron = p.raw.get('framework') == 'megatron' and not p.raw['signals'].get('phases')   # its timer block
        log_node = 0                              # the node whose log has the update lines
        last_update = None                        # the position of the last update line in node 0's output, when known
        if probe_path:
            rows, meta = sig.read_probe(self.executor.read(probe_path) or '')
        else:
            parser = sig.LogParser(p.raw['signals'].get('patterns'), p.raw['signals'].get('ignore'),
                                   bool(p.raw['signals'].get('count_steps')), phases=p.raw['signals'].get('phases'),
                                   megatron_timers=megatron)
            # the log with the update lines (those with a loss: a startup line can carry an iteration number too),
            # node 0's first; Megatron prints its updates on the last rank: the last node next
            logs = [(0, res['lines'])] + list(reversed(list(enumerate(res.get('node_lines') or [], 1))))
            parsed = []
            for i, lines in logs:
                node_rows = parser.parse(lines)
                parsed.append((i, node_rows, parser.last_update_index))
            log_node, rows, last_update = max(parsed, key=lambda x: (sum('loss' in r for r in x[1]), len(x[1])))
            meta = {}
        n = k or steps
        window = kw.get('window') or ([max(2, n // 3 + 1), n] if n != p.steps else p.window)
        scanned = res['lines']
        if res.get('stopped'):                    # what follows our own stop is its shutdown's; every node's lines
            cut = res.get('stop_index')           # before it are the job's (a worker's OOM included)
            scanned = scanned[:cut] if cut is not None else scanned
            for lines, n in zip(res.get('node_lines') or [], res.get('node_stop_index') or []):
                scanned = scanned + lines[:n]
        events = sig.scan_events(scanned, ignore=p.raw['signals'].get('ignore'))
        if log_node == 0 and not res.get('stopped'):   # the update lines and the error lines in one node's order:
            events['collective_at_end'] = sig.collective_after(res['lines'], last_update)   # None when no update line
            if events['collective_at_end'] is None:
                events.pop('collective_at_end')
        ev = sig.evidence(rows, steps=steps, window=window, exit_code=res['exit_code'], timed_out=res['timed_out'],
                          events=events, frozen=frozen,
                          meta=meta, static_workload=p.invariants(config),
                          wall_s=res['wall_s'], memory_cap_gib=p.raw['hardware'].get('memory_gib'),
                          phase_top=sig.MEGATRON_TOP if megatron and not probe_path else None)
        # the planned stop: k updates done and then stopped by us (or ended cleanly), not a run that failed by itself
        planned = (k is not None and type(ev.get('steps_completed')) is int and ev['steps_completed'] >= k
                   and (bool(res.get('stopped')) or res['exit_code'] == 0) and not res.get('failed_before_stop'))
        ev['stop_after'], ev['planned_stop'] = k, planned
        fatal_stop = res.get('fatal_stop') if not planned else None
        if fatal_stop:                            # stopped by the fatal-line watch: a failed run, not an abort for speed
            ev['fatal_stop'] = fatal_stop
        ev['stopped_early'] = bool(res.get('stopped')) and not planned and not fatal_stop   # an abort, not the planned stop
        if peak is not None:                      # D3: an extra field only - the verifier's check, the judge and the
            ev['memory_peak_measured_gib'] = peak  # receipt digest read `memory_peak_gib`, which stays as it was
        if peaks and purpose == 'reference' and mods:
            ev['modules_dir'] = mods              # D7: where the imports are (read where the job runs: search/library)
        if kw.get('profile'):                     # what rank 0's hook wrote (phases, or why not)
            try:
                ev['profile'] = json.loads(self.executor.read(f'{run_dir}/profile/phases.json') or 'null')
            except ValueError:
                ev['profile'] = {'error': 'phases.json unreadable'}
        bench_s = 0.0
        if purpose in ('reference', 'doctor') and p.nodes > 1 and res['exit_code'] == 0 and kw.get('bench', True):
            b0 = time.perf_counter()
            self.bench_links(rel=rel, adir=adir, env=env, setup=setup)
            bench_s = time.perf_counter() - b0
        if before is not None or after is not None:
            ev['gpu_check'] = {'contention': gpu_warn, 'ecc': ecc, 'persistence_off': any(
                g['persistence'] == 'Disabled' for h in before or after or [] if h for g in h['gpus'])}
        ok = (res['exit_code'] == 0 or planned) and not res['timed_out']
        reasons = [] if ok else ([f'killed at the {res["wall_s"]:.0f} s timeout'] if res['timed_out']
                                 else [f'exit code {res["exit_code"]}'])
        t = ev['step_time']
        receipt = {'schema': SCHEMA, 'project': p.name, 'task_id': getattr(task, 'id', p.name), 'run_id': run_id,
                   'attempt_id': attempt_id, 'candidate_id': cid, 'purpose': purpose, 'config': config,
                   'config_digest': digest(config), 'overlay_files': sorted(overlay or {}), 'overlay': dict(overlay or {}),
                   'overlay_digest': digest(sorted((overlay or {}).items())) if overlay else None, 'command': cmd,
                   'executor': self.executor.describe(), 'host': socket.gethostname(), 'world_size': p.world,
                   'evidence': ev, 'valid': ok, 'invalid_reasons': reasons,
                   'step_time': {'mean_window_max_rank_s': t if ok else None},
                   'metric': {'name': 'step_time_s', 'value': t if ok else None, 'direction': 'min'},
                   'phases': {'materialize': round(t1 - t0, 3), 'run': round(t2 - t1 - probe_s, 3),
                              **({'link_probe': round(probe_s, 3)} if probe_s else {}),
                              **({'link_bench': round(bench_s, 3)} if bench_s else {}),
                              'total': round(time.perf_counter() - t0, 3)},
                   'cost': {'gpu_seconds': round(p.world * (res['wall_s'] + lost_s), 3),
                            'wall_seconds': round(time.perf_counter() - t0, 3),
                            'accounting_complete': not lost_s},   # a cut-off execution is charged by its wall time
                   'log': str(adir / 'train.log'),          # node 0: the launch output (Megatron's argument dump)
                   **({'step_log': str(adir / f'node{log_node}.log')} if log_node else {}),   # the update lines
                   'log_tail': [x for _, x in res['lines'][-30:]],
                   'utc_start': utc_start, 'utc_end': utc_now()}
        receipt = json.loads(json.dumps(receipt, default=str))
        signing.sign(receipt)
        (adir / 'receipt.json').write_text(json.dumps(receipt, indent=1), encoding='utf-8')
        if not self.keep_trees and isinstance(self.executor, ex.LocalExecutor):
            shutil.rmtree(tree, ignore_errors=True)
        self.attempts.append({'attempt_id': attempt_id, 'candidate_id': cid, 'purpose': purpose, 'ok': ok, 'step_time': t,
                              **({'gpu_contention': gpu_warn} if gpu_warn else {}), **({'gpu_ecc': ecc} if ecc else {})})
        if planned and res.get('stopped'):              # U-5-8: the exit code of a planned stop is not a failure
            self.log(f'{run_id}/{cid}: {purpose} stopped as planned after {k} updates (exit={res["exit_code"]} from the '
                     f'stop) step_time={t}')
        else:
            self.log(f'{run_id}/{cid}: {purpose} exit={res["exit_code"]} step_time={t}')
        return receipt

    def memory_peak(self, d: str) -> float | None:
        """D3: an execution's peak reserved GPU memory (GiB) - the maximum over the files its processes wrote in `d`
        (its own directory, `<run>/mempeak/exec-<n>`) on every host (one read per host after the run); None when none
        wrote one (no CUDA, no torch, the hook not on the path)."""
        if isinstance(self.executor, ex.LocalExecutor):
            p = Path(d)
            texts = [f.read_text(encoding='utf-8', errors='replace') for f in sorted(p.glob('*.txt'))] if p.is_dir() else []
        else:
            try:
                texts = self.executor.on_hosts(f'cat {self.executor.quote(d)}/*.txt 2>/dev/null', timeout_s=20) or []
            except Exception:                         # noqa: BLE001 - an observation: never the run's failure
                texts = []
        vals = [float(m.group(1)) for t in texts if t for m in MEMPEAK_RE.finditer(t)]
        return round(max(vals), 3) if vals else None

    def devices(self, vis: str | None = None) -> set | None:
        """The job's GPUs on each host: CUDA_VISIBLE_DEVICES (`vis`, else the job env's: the manifest's, else this
        process's for a local run) as indices or GPU-uuid prefixes, else the first `hardware.gpus`; an explicitly empty
        selection is no GPU; None for MIG devices (their utilisation and processes are not the parent GPU's to judge)."""
        p = self.project
        vis = vis if vis is not None else env_for(p, p.reference_config()).get('CUDA_VISIBLE_DEVICES')
        if vis is None and isinstance(self.executor, ex.LocalExecutor):
            vis = os.environ.get('CUDA_VISIBLE_DEVICES')
        if vis is None:
            return set(range(p.gpus))
        ids = [x.strip() for x in str(vis).split(',') if x.strip()]
        if any(x.upper().startswith('MIG') for x in ids):
            return None
        return {int(x) if x.isdigit() else x for x in ids}

    def gpu_state(self, vis: str | None = None) -> list | None:
        """Every host's GPUs of the job now (executor.parse_smi per host; None for a host without nvidia-smi or that
        did not answer); None when the executor cannot look (srun, Windows), the job uses no GPU or a MIG slice, or no
        host has an NVIDIA GPU."""
        try:
            dev = self.devices(vis)
            if not dev:
                return None
            snap = [ex.parse_smi(o, dev) for o in self.executor.on_hosts(ex.SMI_QUERY)]
        except Exception:                                  # noqa: BLE001 - a look at the GPUs never fails a run
            return None
        return snap if any(snap) else None

    def _until_free(self, vis, snap: list | None, what: str) -> list | None:
        """Look again until no process holds the job's GPUs, for at most own_wait_s, and return the last look: after
        a run this runner stopped, and before it retries one the transport cut off, what holds them is most likely the
        run's own processes (F4-U8-same: an aborted screen's workers ran next to the next two candidates; the orphan of
        a screen whose ssh connection dropped made the retry run out of memory and slowed the next probe)."""
        t0 = time.monotonic()
        self.log(f'{what}: its GPUs are not free yet; waiting up to {self.own_wait_s:.0f} s')
        while ex.gpu_contention(snap, 'after') and time.monotonic() - t0 < self.own_wait_s:
            time.sleep(self.own_poll_s)
            new = self.gpu_state(vis) or []       # a host that did not answer keeps its last reading: no look is no
            snap = [n if n is not None else (snap[i] if i < len(snap) else None)    # free GPU
                    for i, n in enumerate(new)] if new else snap
        if not ex.gpu_contention(snap, 'after'):
            self.log(f'{what}: the GPUs were free after {time.monotonic() - t0:.0f} s')
        return snap

    def link_facts(self) -> dict:
        """The job's interconnect as the last reference / doctor run's probe saw it on node 0 (probe_links,
        executor.parse_link): {intra, inter}; {} before such a run, or when it could not look (Windows, srun)."""
        return dict(self.link)

    def _fatal_watch(self, probe_path):
        """The fatal-line watch of one execution; none for a probe-file job (its updates are not in the log) or when
        INFRARSI_FATAL_GRACE_S is 0."""
        grace = fatal_grace_s()
        if probe_path or grace <= 0:
            return None
        s = self.project.raw['signals']
        return FatalWatch(sig.LogParser(s.get('patterns'), s.get('ignore'), bool(s.get('count_steps'))), grace,
                          ignore=s.get('ignore'))

    def _row_stop(self, predicate, probe_path):
        """Adapt an early-stop predicate over per-update rows [{'step', 't', ...}] to the executor's stamped lines."""
        if predicate is None:
            return None
        sigcfg = self.project.raw['signals']
        parser = sig.LogParser(sigcfg.get('patterns'), sigcfg.get('ignore'), bool(sigcfg.get('count_steps')))

        def stop(lines, node=None):                    # node: the executor's node index (unused here)
            if probe_path:
                rows, _ = sig.read_probe(self.executor.read(probe_path) or '')
            else:
                rows = parser.parse(lines)
            return bool(rows) and predicate(rows)
        return stop

    def _with_setup(self, cmd: str, cfg_rel: str | None, cwd: str, adir: Path, ckpt_dir: str | None = None) -> str:
        """An eval or test command after launch.setup, as the job's command runs (its environment, generated files)."""
        run_dir = cwd.rsplit('/tree', 1)[0] if cwd.endswith('/tree') else str(adir)
        setup = self.setup_line(cfg_rel=cfg_rel, steps=self.project.steps, run_dir=run_dir, ckpt_dir=ckpt_dir)
        return f'{setup} && {cmd}' if setup else cmd

    def run_eval(self, config: dict, overlay: dict | None, run_id: str, candidate_id: str, ckpt_dir: str) -> dict | None:
        """The manifest's eval command (verify.eval) on a saved checkpoint: {'cmd', 'exit_code', 'value', 'tail'}; the
        value is the last eval loss (eval_loss / val_loss / loss) the command prints."""
        cmd = self.project.raw['verify'].get('eval')
        if not cmd:
            return None
        adir = self.out_dir / run_id / f'{candidate_id}-eval-{new_attempt_id()[:8]}'
        cfg_rel = self.materialize(adir / 'tree', dict(config or {}), overlay or {})
        cwd = self.executor.place(adir / 'tree', f'{adir.relative_to(self.out_dir).as_posix()}/tree')
        values = {'python': self.executor.quote(self.executor.python(self.project.raw['execution'].get('python'))),
                  'config': self.executor.quote(cfg_rel) if cfg_rel else '', 'gpus': self.project.gpus,
                  'ckpt_dir': self.executor.quote(ckpt_dir)}
        c = self._with_setup(fill(cmd, values, {'config': cfg_rel or '', 'ckpt_dir': ckpt_dir}), cfg_rel, cwd, adir, ckpt_dir)
        res = self.executor.run(c, cwd=cwd, env=env_for(self.project, config), log_path=adir / 'eval.log',
                                timeout_s=float(self.project.raw['launch']['timeout_s']), **self._stamper())
        hits = [(m.group('k').lower(), m.group('v')) for _, line in res['lines'] for m in EVAL_RE.finditer(line)]
        found = [v for k, v in hits if k != 'loss'] or [v for _, v in hits]      # eval_loss / val_loss before a bare loss
        try:
            value = float(found[-1]) if found else None
        except ValueError:
            value = None
        if not self.keep_trees and isinstance(self.executor, ex.LocalExecutor):
            shutil.rmtree(adir / 'tree', ignore_errors=True)
        return {'cmd': c, 'exit_code': res['exit_code'], 'value': value, 'tail': chr(10).join(x for _, x in res['lines'][-10:])}

    def run_tests(self, config: dict, overlay: dict | None, run_id: str, candidate_id: str) -> list[dict]:
        """The project's own test commands (verify.tests) on the candidate tree: [{'cmd', 'exit_code', 'tail'}]."""
        cmds = list(self.project.raw['verify'].get('tests') or [])
        if not cmds:
            return []
        adir = self.out_dir / run_id / f'{candidate_id}-tests-{new_attempt_id()[:8]}'
        cfg_rel = self.materialize(adir / 'tree', dict(config or {}), overlay or {})
        cwd = self.executor.place(adir / 'tree', f'{adir.relative_to(self.out_dir).as_posix()}/tree')
        values = {'python': self.executor.quote(self.executor.python(self.project.raw['execution'].get('python'))),
                  'config': self.executor.quote(cfg_rel) if cfg_rel else '', 'gpus': self.project.gpus}
        out = []
        for i, cmd in enumerate(cmds):
            c = self._with_setup(fill(cmd, values, {'config': cfg_rel or ''}), cfg_rel, cwd, adir)
            res = self.executor.run(c, cwd=cwd, env=env_for(self.project, config), log_path=adir / f'test{i}.log',
                                    timeout_s=float(self.project.raw['launch']['timeout_s']), **self._stamper())
            out.append({'cmd': c, 'exit_code': res['exit_code'], 'tail': chr(10).join(x for _, x in res['lines'][-15:])})
        if not self.keep_trees and isinstance(self.executor, ex.LocalExecutor):
            shutil.rmtree(adir / 'tree', ignore_errors=True)
        return out
