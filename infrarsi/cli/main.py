"""infrarsi: make an existing training job faster, verified.

    infrarsi [SCRIPT|DIR]        one-command autopilot: init -> doctor -> run (B-Comp v4.1) -> report -> apply
    infrarsi optimize [TARGET]   explicit alias for the one-command 5-stage autopilot shell
    infrarsi init [DIR|SCRIPT]   draft infrarsi.yaml for a training repository (review it)
    infrarsi doctor [MANIFEST]   one run of the current configuration: what the verifier can check
    infrarsi run [MANIFEST|SUITE] [--strategy S] [--llm backend[:model]] [--tolerance 0.03] [--screens N]
    infrarsi run --resume RUN_DIR [--screens N] [--budget-usd X] [--budget-tokens T] [--wall-hours H]
                   [--remeasure-reference] [--allow-code-change]    continue a run from its search-state checkpoint
    infrarsi report [RUN_DIR]    print the report of a run
    infrarsi apply [RUN_DIR] [--write] [--branch NAME]
    infrarsi finish [RUN_DIR] [--dry-run]   re-verify the best candidates of a run that stopped early, adopt the winner
    infrarsi requirements [MANIFEST] [--text T] [--file F]   compile requirements in your own words, show what they mean
    infrarsi exp validate PACK... | stats [PACK...] | export RUN_DIR|PACK --out FILE   experience packs


Each run writes RUN_DIR = .infrarsi/runs/<name>-<UTC time> (REPORT.md, result.json, run.log). `report` and `apply`
take RUN_DIR, a file in it or a project directory (its newest run); without one, the newest run under .infrarsi/runs
of the current directory. A benchmark is the same `run` over a suite file (projects x strategies): every manifest is
checked before the first run. Errors in the request print one line (set INFRARSI_DEBUG=1 for the traceback).
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import json
import math
import os
import posixpath
import re
import shlex
import signal
import socket
import sys
import threading
import time
from pathlib import Path

from infrarsi.agent.search import USER_STOP
from infrarsi.contracts import EXECUTION_DIR, RemotePaths, RunRequest, TaskSpec, utc_now

RESEARCH =('discover', 'prepare', 'baseline', 'validate', 'export', 'benchmark', 'redteam', 'dispatch', 'monitor')


class RequestError(Exception):
    """A request the command cannot act on (a missing run directory, an existing file): one line, exit 2."""


def _dump(obj, out: str | None):
    text = json.dumps(obj, indent=1, sort_keys=True, default=str)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding='utf-8')
        print(f'wrote {out}')
    else:
        print(text)


def _nodes(s: str | None) -> list[str]:
    return [n.strip() if n.strip().startswith('WINH') else f'WINH{n.strip()}' for n in (s or '').split(',') if n.strip()]


def _needs(module: str, what: str):
    if importlib.util.find_spec(module) is None:
        raise NotImplementedError(f'{what} needs {module} (owned by another builder), which is not in this tree yet')
    return importlib.import_module(module)


def cmd_discover(a):
    from infrarsi.discover import default_profile, hardware_profile, workload_facts
    out = {}
    if a.nodes:
        out['hardware'] = default_profile(len(_nodes(a.nodes))) if a.offline else hardware_profile(_nodes(a.nodes))
    if a.task:
        out['workload'] = workload_facts(a.task, refresh=a.refresh)
    if not out:
        raise SystemExit('give --nodes and/or --task')
    _dump(out, a.out)


def cmd_prepare(a):
    """Run T-data's build_index on WINH (CPU, background) for the given tasks; INDEX.json is rewritten when done."""
    _needs('infrarsi.adapters.data.build_index', 'prepare')
    from infrarsi.contracts import DESIGN_DIR
    from infrarsi.harness import codesync
    from infrarsi.harness.launch import MCORE_STACK, VENV_PYTHON
    from infrarsi.harness.remote import default_remote
    r = default_remote()
    cs = codesync.ensure_code(r, a.node)
    specs = f'{RemotePaths.root}/design/task_specs'             # frozen copies (identical bytes, DESIGN_FREEZE.json)
    r.call(a.node, 'makedirs', paths=[specs, f'{RemotePaths.root}/logs/prepare'])
    for p in sorted((DESIGN_DIR / 'task_specs').glob('*.yaml')):
        r.put_bytes(a.node, p.read_bytes(), f'{specs}/{p.name}')
    log = f'{RemotePaths.root}/logs/prepare/build_index_{int(time.time())}.log'
    tasks = f' --tasks {a.tasks}' if a.tasks else ''
    cmd = (f'cd {cs["code_root"]} && PYTHONPATH={cs["code_root"]}:{":".join(MCORE_STACK)} setsid nohup {VENV_PYTHON} '
           f'-m infrarsi.adapters.data.build_index --specs {specs}{tasks} > {log} 2>&1 < /dev/null & echo started')
    print(r.sh(a.node, cmd)['stdout'].strip(), '->', log)


def _store():
    from infrarsi.dispatch.queue import QueueStore
    return QueueStore()


def cmd_baseline(a):
    from infrarsi.discover import default_profile, workload_facts
    from infrarsi.planner import expert_default, fr_v0
    task = TaskSpec.load(a.task)
    if a.plan_only:
        facts = workload_facts(task)
        res = fr_v0(task, facts=facts) if a.arm == 'FR' else expert_default(task, facts=facts, hw=default_profile(task.nodes))
        return _dump(res, a.out)
    from infrarsi.dispatch.handlers import BUILTIN, Context
    run = {'run_id': a.run_id or f'CLI-{task.id}-{a.arm}', 'task': task.id, 'arm': a.arm, 'wall_hours_cap': a.wall_hours}
    _dump(BUILTIN[a.arm](run, _nodes(a.nodes), Context(store=_store())), a.out)


# ---------------------------------------------------------------- the one user flow (revamp design B)
def cmd_init(a):
    from infrarsi.project import detect
    from infrarsi.project.autopilot import is_script_path, resolve_script_target
    from infrarsi.project.requirements import normalize
    target_dir = a.dir
    entry = a.entry
    if Path(target_dir).is_file() and is_script_path(Path(target_dir)):
        if entry or a.command:
            raise RequestError('give a script path or --entry / --command, not both')
        root_dir, entry = resolve_script_target(Path(target_dir))
        target_dir = str(root_dir)
    if a.command:                                       # the job's own command line: its values are the knob defaults
        if entry:
            raise RequestError('give --command or --entry, not both')
        try:
            words = shlex.split(a.command)
        except ValueError as exc:
            raise RequestError(f'--command cannot be read: {exc}') from None
        ops = sorted(set(shell_ops(a.command)))
        prog = [w for w in words if not ('=' in w and w.split('=')[0].isidentifier())]       # after VAR=value
        if ops:                                         # the search's flags would go to the last command (`| tee`)
            raise RequestError(f'--command takes one command, not a pipeline or a list ({" ".join(ops)}): put it in a '
                               'script that passes "$@" to the training command and give that with --entry')
        if prog and (posixpath.basename(prog[0]) in ('bash', 'sh') or prog[0].endswith('.sh')):
            raise RequestError('the command starts a shell script: give the script with --entry (it passes "$@" to '
                               'the training command), or give the training command itself with --command')
        launch = Path(target_dir) / LAUNCH_SCRIPT
        if launch.exists() and not a.force:
            raise RequestError(f'{launch} exists (--force replaces it)')
        launch.write_text(f'#!/bin/bash\n# the job\'s command (infrarsi init --command); the search adds its flags after it '
                          f'("$@": the last value wins)\n{a.command.strip()} "$@"\n', encoding='utf-8', newline='\n')
        entry = LAUNCH_SCRIPT
    manifest, notes = detect.draft(target_dir, config=a.config, entry=entry)
    if a.command:
        notes.insert(0, f'{LAUNCH_SCRIPT} holds your command: its flags are the knob defaults; the search passes the '
                        'knob values after them.')
    if a.requirements:                                   # the user's words as they are: `run` compiles them
        try:
            manifest['requirements'] = {'text': normalize(a.requirements)}
        except ValueError as exc:
            raise RequestError(str(exc)) from None
    try:
        out = detect.write(target_dir, manifest, notes, force=a.force)
    except FileExistsError as exc:
        raise RequestError(str(exc)) from None
    print(f'wrote {out}: framework {manifest["framework"]}, {len(manifest["knobs"])} knobs, '
          f'{manifest["hardware"]["gpus"]} GPU(s)')
    for n in notes:
        print(f'  - {n}')
    print('next: review the file, then `infrarsi doctor`')



LAUNCH_SCRIPT = 'infrarsi_launch.sh'


def shell_ops(cmd: str) -> list[str]:
    """The list and pipeline operators of a command line (`|`, `||`, `&&`, `;`, `&`, `|&`) outside quotes and escapes;
    redirections (`2>&1`, `&>`) are not operators."""
    ops, quote, i = [], None, 0
    while i < len(cmd):
        c = cmd[i]
        if quote:
            if c == '\\' and quote == '"':
                i += 1
            elif c == quote:
                quote = None
        elif c == '\\':
            i += 1
        elif c in '\'"':
            quote = c
        elif c in '|;' or (c == '&' and cmd[i - 1:i] not in ('>', '<') and cmd[i + 1:i + 2] != '>'):
            op = cmd[i:i + 2] if cmd[i:i + 2] in ('||', '&&', '|&') else c
            ops.append(op)
            i += len(op) - 1
        i += 1
    return ops


def cmd_doctor(a):
    from infrarsi.contracts import InfrastructureError
    from infrarsi.project import Project
    from infrarsi.project.doctor import doctor
    from infrarsi.project.flow import configured_patch_status
    proj = Project.load(a.manifest)
    _warn_if_locked(proj.root)                           # U-7-9: doctor's one short run writes no lock of its own
    driver = _print_llm(proj)
    try:
        rep = doctor(proj, log=lambda m: None, experience=a.experience)
    except InfrastructureError as exc:
        raise SystemExit(f'doctor: the hosts could not be reached: {exc}\n'
                         'check that `ssh <host>` works without a password prompt (execution.hosts), then run again')
    # U-1-6, U-6-1: rounded numbers in words, and where the run happened when it was not this machine
    clock = {'clock': "InfraRSI's clock", 'reported': "the job's own step time"}.get(rep['step_time_source'],
                                                                                         rep['step_time_source'])
    rsd = f', spread within the run (RSD) {rep["rsd_window"]:.1%}' if rep.get('rsd_window') is not None else ''
    print(f'reference run: {"OK" if rep["valid"] else "NOT VALID"}; {_num(rep["step_time"])} s per update ({clock}){rsd}; '
          f'{rep["steps_completed"]} updates logged{_where(proj)}')
    print(f'signals seen (updates of {rep["steps_completed"]} that log each):',
          ', '.join(f'{k} {v}' for k, v in rep['signals_seen'].items()))
    n = rep.get('noise') or {}
    if n.get('sigma_pair') is not None:                 # U-1-7: one number, and what it was estimated from
        src = ('the manifest (search.run_noise)' if n['source'] != 'measured' else
               f'{n["earlier_pairs"]} earlier repeat(s) of this project and the two halves of this run'
               if n.get('earlier_pairs') else 'the two halves of this run (no earlier runs of this project)')
        print(f'noise between two runs of one configuration (sigma_pair): {n["sigma_pair"]:.3%}, from {src}; a single '
              f'run resolves gains above {n["resolvable_gain"]:.2%}')
    ph = rep.get('phases') or {}
    if ph.get('mean_ms'):
        top = sorted(ph['mean_ms'].items(), key=lambda kv: -kv[1])[:6]
        how = 'one profiled run' if rep.get('phases_source') == 'profile' else 'window means'
        print(f'where the time goes ({how}): ' + ', '.join(
            f'{k} {v:.1f} ms' + (f' ({100 * ph["share"][k]:.1f} %)' if ph.get('share') else '') for k, v in top)
            + (f'; residual {ph["residual_ms"]:.1f} ms' if ph.get('residual_ms') is not None else ''))
    print(f'code patches {configured_patch_status(proj)}{_capability_words(proj, driver)}')
    _print_requirements(proj)
    _print_accounting(rep)
    ic = rep.get('interconnect') or {}
    if ic.get('used') or ic.get('detected'):
        used = ic.get('used') if isinstance(ic.get('used'), dict) else {'inter': ic.get('used')}
        src = 'detected' if ic.get('detected') and dict(ic['detected']) == used else 'manifest'
        print('interconnect: ' + ', '.join(f'all-reduce {v} GB/s bus bandwidth' if k == 'bandwidth_gbs' else f'{k} {v}'
                                           for k, v in used.items() if v) + f' ({src})')
    for c in rep['checks']:                              # U-1-8: the whole detail (the terminal wraps it)
        print(f'  {c["status"]:9s} {c["id"]:28s} {c["detail"]}')
    for w in rep['warnings'] + [f'not valid: {r}' for r in rep['reasons']]:
        print(f'  ! {w}')
    _print_experience(rep.get('experience'))
    print(f'log: {rep["log"]}')
    if rep.get('step_log'):
        print(f'update lines (another node prints them): {rep["step_log"]}')
    if rep['valid']:                                     # U-1-10: what to do with this, as init says after its draft
        print('next: `infrarsi run` (with the manifest\'s llm block or `--llm backend:model`; `--llm none` searches '
              'without a model)')
    else:
        print('next: fix what the lines marked ! and "not valid" say, then run `infrarsi doctor` again')
    raise SystemExit(0 if rep['valid'] else 1)


def _print_requirements(proj) -> None:
    """doctor: the manifest's requirements, from the cache only (doctor makes no model call for them); nothing without any."""
    from infrarsi.project.requirements import compile as compile_requirements, manifest_texts
    texts = manifest_texts(proj)
    if not texts:
        return
    rec = compile_requirements(proj, texts, driver=None, cached_only=True)
    if rec is None:
        print(f'requirements: {len(texts)} item(s) in infrarsi.yaml, not compiled yet: `infrarsi requirements` compiles them with '
              'the model (a run does it too)')
    else:
        c = rec['counts']
        print(f'requirements: {len(texts)} item(s) in infrarsi.yaml, compiled {(rec.get("compile") or {}).get("utc")}: '
              f'{c["enforced"]} enforced, {c["already"]} already, {c["advisory"]} guidance only, {c["blocking"]} cannot be honoured')


def _num(x, nd: int = 4) -> str:
    """A measured number for a person: significant digits, not a float's repr (U-1-6)."""
    return f'{x:.{nd}g}' if isinstance(x, (int, float)) and not isinstance(x, bool) else '-'


def _where(proj) -> str:
    """' (on h1, h2 over ssh)' when the job does not run on this machine (U-6-1), else ''."""
    ex = proj.raw.get('execution') or {}
    if ex.get('where') == 'ssh':
        return f' (on {", ".join(str(h) for h in ex.get("hosts") or [])} over ssh)'
    return ' (in a slurm allocation)' if ex.get('where') == 'slurm' else ''


def _capability_words(proj, driver) -> str:
    """'; model capability 0.90 (...)' for a manifest with a model (U-3-2: the value decides menu mode and patches): the
    value a run of this manifest uses (capability_of: llm.capability, else the driver's model, a backend default
    included), printed after the reference run, where the llm block was always read."""
    from infrarsi.project.flow import capability_of
    from infrarsi.project.llm import parse
    if driver is None:
        return ''
    how = 'llm.capability' if parse(proj.raw.get('llm')).get('capability') is not None else \
        'inferred from the model name; --capability or llm.capability sets it'
    return f'; model capability {capability_of(proj, driver):.2f} ({how})'


def _calls_model(project, strategy, driver) -> bool:
    """Whether the strategy this run resolves to calls the model (tpe, random and doe do not)."""
    from infrarsi import search
    try:
        return search.resolve(strategy or project.raw['search']['strategy'], has_llm=driver is not None,
                              knobs=project.knobs) in search.LLM_STRATEGIES
    except ValueError:                                   # an unknown strategy: run_project says so
        return False


def _print_llm(proj):
    """doctor: one small call to the manifest's model, before the reference run (F-D1-16-2); returns the driver (None
    without one, or when it could not be made) for the capability line after the run."""
    from infrarsi.project.llm import UNPRICED, make_driver, parse, preflight
    spec = parse(proj.raw.get('llm'))                    # `llm: vllm:model` too
    try:
        driver = make_driver(spec)
    except (ValueError, FileNotFoundError) as exc:
        print(f'  ! llm: {exc}')
        return
    if driver is None:
        return
    name = f'{spec.get("backend")} {getattr(driver, "model", "") or ""}'.strip()
    url = getattr(driver, 'base_url', None)              # U-3-13: an unreachable host waits for the call's timeout
    print(f'llm: one test call to {name}' + (f' at {url}' if url else '') + ' ...', flush=True)
    p = preflight(driver)
    if p['ok']:
        print(f'llm: {name} answers ({p["seconds"]} s)' + (f'; {UNPRICED}' if getattr(driver, 'unpriced', False) else ''))
    elif p['refused']:                                   # U-3-11: run does not go on without the model, it stops
        print(f'  ! llm: {name} {_llm_trouble(spec, driver, p["error"])}: `run` stops before its reference run until '
              'this is fixed (or use --llm none)')
    else:
        print(f'  ! llm: {name}: a test call failed ({str(p["error"])[:240]}); each call of a run is retried')
    return driver


def _llm_trouble(spec: dict, driver, error) -> str:
    """A refused test call in words a user can act on: nothing answering at the URL is not a refusal (U-3-12), a 401 /
    403 with the key's variable unset is a missing key (U-5-1), a gateway's 403 is usually the model not being
    available from here (U-5-5). The provider's own message is kept."""
    err = str(error)[:240]
    url = getattr(driver, 'base_url', None)
    if 'endpoint unreachable' in err:
        return f'got no answer from {url or "the endpoint"} ({err}): check llm.base_url (host and port) and that the server runs'
    if spec.get('backend') in ('openai', 'openai-compat', 'vllm') and re.search(r'HTTP 40[13]\b', err):
        env = spec.get('api_key_env', 'OPENAI_API_KEY' if spec.get('backend') == 'openai' else None)
        if env and not os.environ.get(env):
            return f'refused a test call: {err} - the environment variable {env} (llm.api_key_env) is not set'
        if not env:
            return (f'refused a test call: {err} - no llm.api_key_env is set (the environment variable that holds the '
                    'key; llm.headers can carry one too)')
    hint = (' (a gateway answers 403 when this model is not available to the key or from this network: try another '
            'model)' if re.search(r'HTTP 403\b', err) else '')
    return f'refused a test call: {err}{hint}'


def _print_accounting(rep: dict):
    a = rep.get('accounting')
    if not a:
        if rep['valid'] and not rep.get('accounting_off'):
            print('accounting: no memory or MFU estimate: it needs the model shape (manifest model: params_b, layers, '
                  'hidden, ffn, heads, vocab and workload.seq_len); fp32 training and knob values without a mechanism '
                  'state are not modelled')
        return
    t = a['terms_gib']
    peak = rep.get('memory_peak_gib')
    print(f'accounting (closed-form, +-25 %): about {a["memory_gib"]:.0f} GiB per GPU of {a["capacity_gib"]} = weights, '
          f'grads and optimizer {t["weights_grads_optimizer"]:.0f} + activations {t["activations"]:.0f} + logits '
          f'{t["logits"]:.0f} + CUDA / NCCL {t["cuda_nccl_outside_allocator"]:.0f} + allocator slack {t["allocator_slack"]:.0f}'
          + (f'; the log\'s allocator peak {peak:.1f} GiB (estimate without CUDA / NCCL: '
             f'{a["memory_gib"] - t["cuda_nccl_outside_allocator"]:.0f})' if peak else ''))
    if a.get('mfu') is not None:
        print(f'MFU {100 * a["mfu"]:.0f} %: the ideal update {a["ideal_update_s"]:.3f} s at the GPUs\' '
              f'{a.get("tflops_basis") or "nominal"} matrix throughput vs {a["measured_update_s"]:.3f} s measured (the gap: '
              'communication, input, kernels, recompute)')
    if a.get('dp_sync_lower_bound_s'):
        print(f'data-parallel gradient sync: at least {a["dp_sync_lower_bound_s"]:.3f} s per update at the measured '
              f'{a["bandwidth_gbs"]} GB/s all-reduce bus bandwidth (before overlap)')
    for n in a.get('notes') or []:                       # U-4-6: the estimate does not read the log's tokens
        print(f'  note: {n}' + (' - the estimate does not read the tokens in the log: set workload.global_batch_tokens'
                                if n.startswith('tokens per update unknown') else ''))


def _print_experience(e: dict | None):
    if not e:
        return
    lv = e['levels']
    print(f'experience: {len(e["packs"])} packs; evidence for the {sum(lv.values())} knob moves of this project: covered '
          f'{lv.get("covered", 0)}, null {lv.get("null", 0)}, partial {lv.get("partial", 0)}, none {lv.get("none", 0)}' + (f'; InfraRSI-measured run noise on this GPU class '
                                                               f'{e["noise_pct"]} %' if e.get('noise_pct') else ''))
    print('  (covered: measured in an environment like this one; null: measured there, effect within the noise; partial: '
          'measured elsewhere, only together with other changes, or with conflicting results - a sign only; none: no '
          'measurement. Experience only orders the search.)')   # U-1-11
    for n in e['nearest'][:3]:
        print(f'  nearest {n["env"]} ({n.get("model")}, {n.get("gpu")}, {n.get("params_b")} B): '
              f'{"; ".join(n["differences"]) or "same fingerprint"}')
    for k, c in e['coverage'].items():
        if c['level'] in ('covered', 'partial'):
            near = c.get('nearest') or {}
            where = '' if c['level'] == 'covered' else f'; sign from the nearest evidence: {near.get("gpu")} {near.get("model")}'
            print(f'  {c["level"]:8s} {k} -> {c.get("value")}: x{math.exp(c.get("gain") or 0):.3f} (n {c.get("n")}; '
                  f'{(c.get("evidence") or ["-"])[0]}{where})')
        elif c['level'] == 'null':
            print(f'  null     {k} -> {c.get("value")}: within +-{c.get("within_pct")} % in scope (n {c.get("n")})')
    none = [k for k, c in e['coverage'].items() if c['level'] == 'none']
    if none:
        print(f'  none     {", ".join(none)}')
    unknown = [k for k, c in e['coverage'].items() if not c.get('mechanism')]
    if unknown:                                          # F-D2-R205-4: a hand-added sharding knob nobody could place
        print(f'  unknown  {", ".join(unknown)}: the experience knows no mechanism for these knobs; a knob that names '
              'one (e.g. `mechanism: zero_stage, states: {zero2: "2", zero3: "3"}`; the ids are in '
              'infrarsi/experience/data/mechanisms.yaml) gets the measured data')   # U-2-5: where the ids are
    comp = e.get('composite')
    if comp:
        warn = f'; ! {"; ".join(comp["warnings"])[:160]}' if comp.get('warnings') else ''
        print(f'  start composite: {", ".join(f"{k}={v}" for k, v in comp["change"].items())} (source speed-up '
              f'x{math.exp(comp.get("gain") or 0):.3f}){warn}')
    for k, w in (e.get('seed_warnings') or {}).items():
        print(f'  ! {k}: {"; ".join(w)[:170]} (probe first)')
    if e.get('constraints'):
        print(f'  known failures in scope (constraint ids of the packs): {", ".join(e["constraints"])}')
    for p, errs in e.get('skipped') or []:
        print(f'  ! skipped pack {p}: {errs[0]}')
    if e.get('missing'):                                 # U-1-11: only what is missing, and where it goes
        why = {'model.params_b': '`model: {params_b: <billions of parameters>}`: without it no balance-class knob (layout, '
                                 'micro-batch, precision, overlap) can be covered',
               'workload.seq_len': '`workload: {seq_len: <tokens per sample>}`: without it no knob can be covered'}
        print(f'  ! not in the log or the manifest: {", ".join(e["missing"])} - add to infrarsi.yaml '
              + '; '.join(why.get(m, m) for m in e['missing']))


def cmd_exp(a):
    from collections import Counter
    from infrarsi.experience.pack import read_lines, validate, write_lines
    from infrarsi.experience.store import Store
    if a.action == 'validate':
        bad = 0
        for f in a.paths:
            errs, warns = validate(read_lines(f))
            print(f'{f}: {len(errs)} errors, {len(warns)} warnings')
            for m in [f'error: {e}' for e in errs[:30]] + [f'warning: {w}' for w in warns[:30]]:
                print(f'  {m}')
            bad += bool(errs)
        raise SystemExit(1 if bad else 0)
    if a.action == 'stats':
        st = Store.load(extra=a.paths)
        print(f'{len(st.packs)} packs: ' + ', '.join(f'{p["pack_id"]} ({p["tier"]}, {p["lines"]} lines)' for p in st.packs))
        print(f'{len(st.envs)} environments, {len(st.observations)} observations, {len(st.effects)} effects '
              f'({sum(1 for e in st.effects if e.get("derived"))} derived from observation pairs), {len(st.constraints)} '
              f'constraints, {len(st.rules)} rules, {len(st.judgments)} judgments')
        read = Counter(m for e in st.effects for m in (e.get('change') or {})
                       if st.readable(e, st.envs.get(e['env']) or {}))
        print('readable effects per mechanism: ' + ', '.join(f'{m} {n}' for m, n in read.most_common()))
        for p, errs in st.skipped:
            print(f'skipped {p}: {errs[0]}')
        return
    src = Path(a.paths[0]) if a.paths else Path('')     # export: a run's pack (or any pack) with a shareable header
    if src.is_dir():
        res = json.loads((src / 'result.json').read_text(encoding='utf-8'))
        e = res.get('experience') or {}
        if not e.get('written') and e.get('not_written'):   # U-7-7: and where the missing facts go
            why = e['not_written']
            raise SystemExit(f'{src} wrote no experience pack: {why}'
                             + (' (model: / workload: in infrarsi.yaml; a run writes its pack once they are there)'
                                if str(why).startswith('the manifest has no') else ''))
        src = Path(e.get('written') or '')
    if not a.out or not src.is_file():
        raise SystemExit('export needs a run directory with a written pack (or a pack file) and --out FILE')
    lines = read_lines(src)
    lines[0].update({k: v for k, v in (('pack_id', a.pack_id), ('title', a.title), ('license', a.license),
                                       ('contributors', [a.contributor] if a.contributor else None)) if v})
    errs, warns = validate(lines)
    private = [w for w in warns if 'private data' in w]
    for m in errs + private:
        print(f'  {m}')
    if errs or (private and not a.force):
        raise SystemExit('not exported: fix the errors' + (' / remove the private data (or --force)' if private else ''))
    write_lines(a.out, lines)
    print(f'wrote {a.out}: {len(lines)} lines, license {lines[0].get("license")}'
          + ('' if a.license or lines[0].get('license') not in (None, '', 'unspecified')
             else ' (add --license, e.g. CC-BY-4.0, before you share the pack)'))   # U-7-7


def _suite(path: Path) -> list:
    """infrarsi-suite-v1: {projects: [manifest paths], strategies: [names]} -> [(manifest, strategy)]."""
    import yaml
    try:
        doc = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except yaml.YAMLError as exc:
        raise RequestError(f'{path} is not valid YAML: {exc}') from None
    if not isinstance(doc, dict) or doc.get('schema') != 'infrarsi-suite-v1':
        return []
    strategies = doc.get('strategies') or [None]
    return [((path.parent / p).resolve(), s) for p in doc.get('projects') or [] for s in strategies]


def cmd_run(a):
    from infrarsi.project.autopilot import ensure_manifest, is_script_path
    target = Path(a.manifest)
    if is_script_path(target):
        try:
            proj, drafted, _ = ensure_manifest(target)
        except ValueError as exc:
            raise RequestError(str(exc)) from None
        if drafted:
            _say(f'[infrarsi] auto-drafted {proj.path} from {target}')
        target = proj.path
    jobs = _suite(target) if target.is_file() and target.name != 'infrarsi.yaml' else []
    if not jobs and not a.resume:                        # (a suite or a resumed run has its settings already)
        from infrarsi.project import Project
        from infrarsi.project.interview import wanted
        if wanted(a):
            _startup_questions(Project.load(target), a)
    if jobs and (a.requirements or a.requirements_file or a.requirements_partial):
        raise RequestError('--requirements belongs to one project: put them in each infrarsi.yaml (requirements.text)')
    if a.resume and (a.requirements or a.requirements_file or a.no_requirements or a.requirements_partial):
        raise RequestError('requirements belong to the run that started: a resumed run keeps the ones in its checkpoint (see '
                           'RUN_DIR/result.json); new requirements need a new run')

    if jobs and a.strategy:                              # --strategy replaces the suite's strategies: each project once
        jobs = list(dict.fromkeys((path, a.strategy) for path, _ in jobs))
    jobs = jobs or [(target, a.strategy)]
    budget = {'screens': a.screens} if a.screens else {}
    if a.budget_usd is not None:                         # the budget a resumed run stops at is the TOTAL used (project/resume.py)
        budget['llm_usd'] = float(a.budget_usd)
    if a.budget_tokens is not None:
        budget['llm_tokens'] = int(a.budget_tokens)
    if a.wall_hours is not None:
        budget['wall_seconds'] = float(a.wall_hours) * 3600
    for name in ('target_speedup', 'target_mfu'):
        if getattr(a, name, None) is not None:
            budget[name] = getattr(a, name)
    budget = budget or None
    resume = None
    if a.resume:                                         # the project is the run's own (<root>/.infrarsi/runs/<run>)
        from infrarsi.project.resume import ResumeError, prepare
        try:
            from infrarsi.project.resume import latest_run
            source = latest_run(target, a.out) if a.resume == 'latest' else a.resume
            resume = prepare(source, remeasure=a.remeasure_reference, allow_code_change=a.allow_code_change)
        except ResumeError as exc:
            raise RequestError(str(exc)) from None
        if a.manifest in (None, '.'):
            target = Path(resume.flow.get('result', {}).get('manifest') or resume.run_dir.parents[2] / 'infrarsi.yaml')
        jobs = [(target, resume.flow['strategy'])]
    elif a.remeasure_reference or a.allow_code_change:
        raise RequestError('--remeasure-reference and --allow-code-change belong to --resume')
    ablate = None
    if a.ablate is not None:
        from infrarsi.search import ABLATIONS
        ablate = [x.strip() for x in a.ablate.split(',') if x.strip() and x.strip() != 'none']
        bad = [x for x in ablate if x not in ABLATIONS]
        if bad:
            raise RequestError(f'--ablate: unknown {", ".join(bad)} (known: {", ".join(sorted(ABLATIONS))})')
    # without -v only the milestones ([infrarsi] lines) reach the console; everything is in <run dir>/run.log
    log = print if a.verbose else (lambda m: print(m) if m.startswith('[infrarsi]') else None)
    old = None                                           # `kill` stops a run as Ctrl-C does: the job too, a report
    if hasattr(signal, 'SIGTERM') and threading.current_thread() is threading.main_thread():
        old = signal.signal(signal.SIGTERM, _interrupt)
    try:
        return _run_jobs(a, jobs, budget, ablate, log, resume)
    finally:
        if old is not None:
            signal.signal(signal.SIGTERM, old)


def _api_overrides(project, a):
    from infrarsi.project.llm import parse
    spec = parse(project.raw.get('llm'))
    for arg, field in (('api_base', 'base_url'), ('api_key_env', 'api_key_env')):
        if getattr(a, arg, None) is not None:
            spec[field] = getattr(a, arg)
    project.raw['llm'] = spec
    return spec


def _prepare_job(a, path, strategy) -> tuple:
    """A job's project and LLM driver, its strategy checked: what the job needs before any GPU time (F-SYS-10)."""
    from infrarsi import search
    from infrarsi.project import Project
    from infrarsi.project.llm import make_driver, override as llm_override
    project = Project.load(path)
    try:
        if a.llm is not None:                            # --llm over the manifest's llm block (its endpoint settings)
            project.raw['llm'] = llm_override(project.raw.get('llm'), a.llm)
        driver = make_driver(_api_overrides(project, a))
    except (ValueError, FileNotFoundError) as exc:       # an LLM the request names that cannot be used
        raise RequestError(f'llm: {exc}') from exc
    try:                                                 # the strategy, before the reference run costs its GPU time
        name = search.resolve(strategy or project.raw['search']['strategy'], has_llm=driver is not None,
                              knobs=project.knobs)
    except ValueError as exc:
        raise RequestError(str(exc)) from None
    if driver is None and name in ('greedy', 'agent'):
        raise RequestError(f'the {name} strategy needs an LLM backend (manifest llm.backend or --llm)')
    if name == 'optico':                                 # a model that answers in free text, and OptiCo's fetched prompts
        from infrarsi.search.optico import problem
        why = problem(driver)
        if why:
            raise RequestError(why)
    return project, _with_debate(project, driver, a, name)


def _retry_launch(project, a, driver, operation, *, doctor=False, enabled=True):
    """A failed baseline can be repaired interactively; unattended runs never read stdin."""
    from infrarsi.project.interview import wanted
    from infrarsi.project.recovery import recover, DiagnosisBudget
    from infrarsi.contracts import InfrastructureError
    allowance = getattr(a, 'launch_diagnosis_budget', None) or DiagnosisBudget.for_run(project, a)
    a.launch_diagnosis_budget = allowance
    while True:
        try:
            result = operation(project)
        except InfrastructureError as exc:
            if not doctor:
                raise
            result = {'valid': False, 'reasons': [str(exc)], 'step_time': None, 'steps_completed': 0}
        failed = not result['valid'] if doctor else (result.get('outcome') == 'reference_failed' or
            (result.get('outcome') == 'infrastructure_failure' and not result.get('search')
             and not (result.get('reference') or {}).get('valid')))
        if not failed or not enabled or not wanted(a):
            return project, result
        repaired = recover(project, result, driver, write=_say, allowance=allowance)
        if repaired is None:
            return project, result
        project = repaired


def _startup_questions(project, a, *, suite: bool = False, resume: bool = False, llm_spec=None):
    """The questions that confirm this run's settings item by item, on a terminal (project/interview.py); the project as it
    stands after them. What they change is set on `a` or written to infrarsi.yaml, so the rest of the command is unchanged."""
    from infrarsi.project import interview
    from infrarsi.project.manifest import ManifestError
    if not interview.wanted(a, suite=suite, resume=resume):
        return project
    if project.path is None:
        return project
    qa = interview.Interview(project, a, llm_spec=llm_spec)
    try:
        qa.run()
    except interview.Quit:
        raise RequestError('stopped at the questions: nothing was run') from None
    saved = qa.save()
    if saved:
        _say(f'[infrarsi] your changes are in {saved} (the earlier version: {saved.name}.bak)')
    a.confirm_requirements = interview.RequirementsConfirm(qa)
    try:
        return qa.project if not saved else type(project).load(saved)
    except ManifestError as exc:
        raise RequestError(str(exc)) from None


def _requirement_texts(a, project) -> dict:
    """This run's requirements by where they came from (the manifest's block, the command line, the startup questions)."""
    from infrarsi.project.requirements import manifest_texts
    cli = list(getattr(a, 'requirements', None) or [])
    if getattr(a, 'requirements_file', None):
        try:
            cli.append(Path(a.requirements_file).read_text(encoding='utf-8'))
        except OSError as exc:
            raise RequestError(f'--requirements-file: {exc}') from None
    if getattr(a, 'no_requirements', False) and (cli or getattr(a, 'requirements_partial', False)):
        raise RequestError('--no-requirements and --requirements / --requirements-file / --requirements-partial exclude each other')
    return {'manifest': [] if getattr(a, 'no_requirements', False) else manifest_texts(project), 'cli': cli,
            'interactive': list(getattr(a, 'interactive_requirements', None) or [])}


def _apply_requirements(a, project, driver, *, confirm=None) -> tuple:
    """(the project narrowed by the user's own-words requirements, their record); (project, None) without any. Compiled before
    any GPU time; `confirm`: the startup questions' asker (project/requirements.decide), None in a non-interactive run."""
    from infrarsi.project.requirements import RequirementsError, apply, compile as compile_requirements, decide, explain
    confirm = confirm or getattr(a, 'confirm_requirements', None)
    from infrarsi.project.interview import Quit
    source = _requirement_texts(a, project)
    texts = source['manifest'] + source['cli'] + source['interactive']
    if not texts:
        return project, None
    partial = bool(getattr(a, 'requirements_partial', False)
                   or (project.raw.get('requirements') or {}).get('on_unsupported') == 'continue')
    try:
        while True:
            record = compile_requirements(project, texts, driver=driver, log=_say)
            if record is None:                           # (nothing but blanks)
                return project, None
            record['source'] = source
            for line in explain(record):
                _say(line)
            final = decide(record, interactive=confirm, partial=partial)
            if final is not None:
                break
            texts = confirm.new_text(texts)              # "edit": the asker collects the text again
            source = {'manifest': [], 'cli': [], 'interactive': list(texts)}
        return apply(project, final), final
    except Quit:
        raise RequestError('stopped at the requirements: nothing was run') from None
    except (RequirementsError, ValueError) as exc:
        raise RequestError(str(exc)) from None


def _resumed_requirements(project, resume) -> tuple:
    """A resumed run keeps the requirements its checkpoint started with: their overlay and nothing else; a change of the
    manifest's text is recorded and ignored (new requirements need a new run)."""
    from infrarsi.project.requirements import RequirementsError, apply, digest, manifest_texts
    kept = resume.requirements
    now = digest(project, manifest_texts(project)) if manifest_texts(project) else None
    if kept is None and now is None:                     # no requirements then or now: nothing to record
        return project, None
    changed = now != ((kept or {}).get('digest'))
    resume.entry['requirements'] = {'kept': (kept or {}).get('digest'), 'manifest_now': now, 'changed': changed}
    if changed:
        _say('[infrarsi] resuming: the run keeps the requirements it started with; infrarsi.yaml\'s requirements now differ (new '
             'requirements need a new run)')
    try:
        return apply(project, kept) if kept else project, kept
    except RequirementsError as exc:
        raise RequestError(str(exc)) from None


def _with_debate(project, driver, a, strategy: str):
    """`driver` inside the dual-model debate when the manifest's `debate:` block or the command line turns it on (off by
    default); nothing else changes without it. The request is checked here, before any GPU time."""
    from infrarsi import search
    from infrarsi.project.llm import debate_settings, make_debate
    s = debate_settings(project.raw.get('debate'), enabled=getattr(a, 'debate', None), second=getattr(a, 'second_llm', None),
                        verifier=getattr(a, 'verifier_llm', None))
    project.raw['debate'] = s
    if not s['enabled']:
        return driver
    if driver is None:
        raise RequestError('the debate needs a main model: set llm.backend or --llm')
    if strategy not in search.LLM_STRATEGIES:
        _say(f'  ! the debate is on but strategy {strategy} calls no model: it has no effect')
        return driver
    if strategy == 'optico':                             # (its calls are free text: there is no proposal to compare)
        raise RequestError('the debate does not work with strategy optico: use another strategy, or --no-debate')
    try:
        return make_debate(driver, s)
    except (ValueError, FileNotFoundError) as exc:
        raise RequestError(f'debate: {exc}') from exc


def _preflight_debate(driver) -> None:
    """One small call to the debate's other models, before any GPU time: the debate was asked for, so a refusal stops the run."""
    from infrarsi.agent.debate import model_name
    from infrarsi.project.llm import preflight
    for m in getattr(driver, 'models', lambda: [])():
        name = model_name(m)
        _say(f'llm: one test call to {name} (debate) before any run ...')
        p = preflight(m)
        if p['refused']:
            raise RequestError(f'llm: {name} (debate) refused a test call: {str(p["error"])[:240]}; fix `debate` or use --no-debate')
        if not p['ok']:
            _say(f'llm: a test call to {name} failed ({str(p["error"])[:200]}); the search goes on and retries each call')


def _run_jobs(a, jobs, budget, ablate, log, resume=None) -> int:
    from infrarsi.project.flow import run_project
    from infrarsi.project.manifest import ManifestError
    if a.tolerance is not None and not (0 <= a.tolerance < 1):
        raise RequestError(f'--tolerance must be in [0, 1), got {a.tolerance}')
    prepared = []
    for path, strategy in jobs:                          # U-7-8: every manifest before the first run, not a typo in the
        try:                                             # fourth one found after three projects' GPU time
            prepared.append(_prepare_job(a, path, strategy))
        except Exception as exc:                         # noqa: BLE001 - a shape typo raises AttributeError: name the file
            if len(jobs) == 1:
                raise
            what = str(exc) if isinstance(exc, (ManifestError, RequestError)) else f'{type(exc).__name__}: {exc}'
            what = re.sub(r'\s*\n\s*', ' ', what).strip()    # a YAML parser's diagnostic spans lines: still one line
            raise RequestError(what if str(path) in what else f'{path}: {what}') from exc
    rows = []
    for (path, strategy), (project, driver) in zip(jobs, prepared):
        try:
            if driver is not None and _calls_model(project, strategy, driver):   # before any GPU time (F-D1-16-2)
                from infrarsi.project.llm import parse, preflight
                model = getattr(driver, 'model', None) or 'the model'
                _say(f'llm: one test call to {model} before any run ...')   # U-3-13: it can take llm.timeout_s
                p = preflight(driver)
                if p['refused']:
                    raise RequestError(f'llm: {model} {_llm_trouble(parse(project.raw.get("llm")), driver, p["error"])}; '
                                       'fix `llm` or run with --llm none')
                if not p['ok']:
                    _say(f'llm: a test call failed ({str(p["error"])[:200]}); the search goes on and retries each call')
                _preflight_debate(driver)
            project, requirements = (_resumed_requirements(project, resume) if resume is not None
                                     else _apply_requirements(a, project, driver))
            with _run_lock(project.root):
                if resume is not None:                   # `run --resume`: the run of RUN_DIR goes on from its checkpoint
                    from infrarsi.project.resume import ResumeError, resume_project
                    try:
                        res = resume_project(project, resume.run_dir, prepared=resume, driver=driver, tolerance=a.tolerance,
                                             budget=budget, log=log, capability=a.capability, experience=a.experience,
                                             ablate=ablate, on_interrupt=a.on_interrupt, live=not (a.yes or a.interactive is False))
                    except ResumeError as exc:
                        raise RequestError(str(exc)) from None
                else:
                    project, res = _retry_launch(project, a, driver, lambda p: run_project(
                        p, strategy=strategy, driver=driver, tolerance=a.tolerance, budget=budget,
                        out_root=a.out, log=log, capability=a.capability, experience=a.experience,
                        ablate=ablate, on_interrupt=a.on_interrupt, requirements=requirements,
                        live=not (a.yes or a.interactive is False)), enabled=len(jobs) == 1)
        except Exception as exc:                         # noqa: BLE001 - a suite goes on with its other projects
            if len(jobs) == 1:
                raise
            rows.append({'project': Path(path).parent.name, 'strategy': strategy, 'outcome': f'error: {exc}'[:120],
                         'speedup': None, 'screens': None, 'run_dir': None})
            _say(f'{path}: error: {type(exc).__name__}: {exc}')
            continue
        promo = res.get('promotion') or {}
        rows.append({'project': project.name, 'strategy': res.get('strategy'), 'outcome': res['outcome'],
                     'speedup': promo.get('speedup'), 'screens': ((res.get('search') or {}).get('used') or {}).get('screens'),
                     'run_dir': res['run_dir']})
        words = OUTCOME_WORDS.get('interrupted_promoted' if res['outcome'] == 'interrupted' and promo.get('promoted')
                                  else res['outcome'], '')
        _say(f'{project.name}: {res["outcome"]}' + (f', x{promo["speedup"]:.3f} (re-run)' if promo.get('speedup') else '')
             + f'{words} -> {Path(res["run_dir"]) / "REPORT.md"}')
        if len(jobs) == 1:                               # U-1-16: the next step with the run directory filled in
            _say(f'next: infrarsi report {res["run_dir"]}' + (f'; infrarsi apply {res["run_dir"]} shows the change '
                                                               '(--write writes it)' if promo.get('promoted') else ''))
            if res.get('reference') and res['outcome'] != 'interrupted':
                from infrarsi.project.report import format_console_briefing
                _say(format_console_briefing(res))
        if res['outcome'] == 'interrupted':
            return 130                                  # a suite stops with it
        if (res.get('search') or {}).get('stop_reason') == USER_STOP and len(jobs) > 1:   # (a single run ends normally)
            _say(f'stopped by you: the other {len(jobs) - len(rows)} run(s) of the suite were not started')
            return 130
    if len(rows) > 1:
        _say('\n| project | strategy | outcome | speedup | runs |\n|---|---|---|---|---|')
        for r in rows:
            sp = f'x{r["speedup"]:.3f}' if r['speedup'] else '-'
            n = '-' if r['screens'] is None else r['screens']       # U-7-8: an error row ran nothing
            _say(f'| {r["project"]} | {r["strategy"]} | {r["outcome"]} | {sp} | {n} |')
    # exit status 1 when a run failed (its reference, the hosts, an error); `no_change` is a completed search
    return 1 if any(r['outcome'] in ('reference_failed', 'not_verifiable', 'infrastructure_failure') or r['outcome'].startswith('error')
                    for r in rows) else 0


RUN_LOCK = '.infrarsi/run.lock'


@contextlib.contextmanager
def _run_lock(root: Path):
    """<project root>/.infrarsi/run.lock while a run measures (U-7-9): two runs on the job's GPUs change each other's
    step times. A run that finds a lock says so and goes on: whether that process still lives cannot be tested portably
    (os.kill terminates a process on Windows), and the lock of a run that crashed goes when the next run replaces it."""
    _warn_if_locked(root)
    path = root / RUN_LOCK
    mine = {'pid': os.getpid(), 'host': socket.gethostname(),
            'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    try:
        with contextlib.suppress(OSError):               # a read-only project tree (run --out elsewhere): no lock
            path.parent.mkdir(exist_ok=True)
            with _lock_section(root) as held:
                if held:
                    path.write_text(json.dumps(mine), encoding='utf-8')
        yield
    finally:
        with _lock_section(root) as held:
            if held and _read_lock(root) == mine:        # not the lock of a run started after this one
                with contextlib.suppress(OSError):       # Windows: another process has it open; the next run replaces it
                    path.unlink()


@contextlib.contextmanager
def _lock_section(root: Path):
    """Yields whether this process holds the section in which one run at a time writes the lock or checks and deletes
    it (else a run ending between its check and its delete removes the lock a run starting then has just written). The
    mutex is created with O_EXCL (atomic on Windows and POSIX) and lives for microseconds; while another run's stays
    fresh for the second of waiting, the caller skips that lock operation and goes on: the lock is advisory."""
    mutex, held = root / f'{RUN_LOCK}.mutex', False
    for _ in range(20):
        try:
            os.close(os.open(mutex, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            held = True
            break
        except FileExistsError:                          # another run's section, or a crashed run's mutex
            if not _remove_stale(mutex):
                time.sleep(0.05)
        except OSError:                                  # a read-only tree: no section, as there is no lock
            break
    try:
        yield held
    finally:
        if held:
            with contextlib.suppress(OSError):
                mutex.unlink()


def _remove_stale(mutex: Path) -> bool:
    """Remove a mutex over a second old (its run crashed in the section); True when the create is worth trying again at
    once. Two runs that both find it old must not remove the fresh one the first of them then creates: the removal holds
    a second O_EXCL file and checks that the mutex is still the old one (by mtime; a crashed remover stops removals)."""
    breaker = mutex.with_name(f'{mutex.name}.break')
    try:
        old = mutex.stat().st_mtime_ns
        if time.time_ns() - old < 1_000_000_000:
            return False
        os.close(os.open(breaker, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except OSError:                                      # it is gone, or another run is removing it
        return False
    try:
        with contextlib.suppress(OSError):
            if mutex.stat().st_mtime_ns == old:
                mutex.unlink()
    finally:
        with contextlib.suppress(OSError):
            breaker.unlink()
    return True


def _warn_if_locked(root: Path) -> None:
    lock = _read_lock(root)
    if lock is not None:
        _say(f'! another run may be active in this project: pid {lock.get("pid", "?")} on {lock.get("host", "?")}, '
             f'started {lock.get("started_utc", "?")} - its load changes the step times this run measures; delete '
             f'{root / RUN_LOCK} if that run is gone')


def _read_lock(root: Path) -> dict | None:
    """The project's run lock; None without one, {} when it cannot be read (a run starting now is writing it)."""
    try:
        return json.loads((root / RUN_LOCK).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return {}


def _interrupt(signum, frame):
    raise KeyboardInterrupt


# U-1-16: the outcome codes in words, on the run's last console line (improved says its speed-up itself)
OUTCOME_WORDS = {'no_change': ' (no faster configuration passed the checks: keep the current one)',
                 'interrupted': ' (stopped before the end: nothing was promoted)',
                 # stopped after its promotion had finished (in the write-back): the report recommends the candidate
                 'interrupted_promoted': ' (stopped after the promotion: the verified recommendation stands)',
                 'infrastructure_failure': ' (the hosts could not be reached: nothing was promoted)',
                 'reference_failed': ' (your current configuration did not pass the run checks: nothing was searched)',
                 'not_verifiable': ' (the logs cannot show that a change keeps the job the same: nothing was searched)'}


def _say(msg: str) -> None:
    """print() that survives a console that went away (the results are in the run directories anyway)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        try:
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
                print(msg)
                return
        except Exception:
            pass
        enc = getattr(sys.stdout, 'encoding', None) or 'utf-8'
        try:
            print(msg.encode(enc, errors='replace').decode(enc, errors='replace'))
        except (OSError, ValueError):
            pass
    except (OSError, ValueError):
        pass


def _newest_run(runs: Path) -> Path | None:
    """The directory under `runs` modified last; None when there is none or `runs` cannot be listed."""
    try:
        return max((d for d in runs.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, default=None)
    except OSError:
        return None


def _run_dir(path: str | None, needs: str) -> tuple[Path, bool]:
    """The run directory `report` / `apply` read, and whether it was picked as the newest run (U-1-18): a run directory,
    a file of one (what a shell completes to), a project directory (its newest run under .infrarsi/runs) or no
    argument (the newest run under .infrarsi/runs of the current directory, the only place searched)."""
    p = Path(path or '.')
    if p.is_file() and p.name in ('REPORT.md', 'result.json', 'run.log'):
        p = p.parent
    runs = p / '.infrarsi' / 'runs'
    picked = path is None or runs.is_dir()              # tested first: a training repository may have its own run.log
    if picked:
        p = _newest_run(runs)
        if p is None:
            raise RequestError(f'no run under {runs.as_posix()}: run `infrarsi run` first, or give a run directory')
    if (p / needs).is_file():
        return p, picked
    if (p / 'run.log').is_file():                       # U-7-3: a run that started but never wrote its report
        raise RequestError(f'{p} did not finish: no {needs} was written (the run was interrupted outside the search, '
                           f'or it is still running); its progress is in {p / "run.log"}')
    newest = None if picked else _newest_run(Path('.infrarsi') / 'runs')   # a hint, for a named path only
    raise RequestError(f'{p} is not a run directory (no {needs}); runs are under .infrarsi/runs'
                       + (f' (the newest: {newest})' if newest else ''))


def cmd_report(a):
    run, picked = _run_dir(a.run_dir, 'REPORT.md')
    if picked:                                          # which run, on stderr: stdout stays the report
        print(f'run: {run}', file=sys.stderr)
    print((run / 'REPORT.md').read_text(encoding='utf-8'))


def cmd_apply(a):
    from infrarsi.project.apply import apply
    run, picked = _run_dir(a.run_dir, 'result.json')
    if picked:
        print(f'run: {run}')
    try:
        p = apply(run, write=a.write, branch=a.branch)
    except RuntimeError as exc:                          # a merge conflict, a branch that exists: the request, one line
        raise RequestError(str(exc)) from None
    if not p['files'] and not p['manual']:
        return print(p.get('note', 'nothing to apply'))
    if p.get('changed'):                                 # U-1-23: what changes, before the files that carry it
        print('the change: ' + ', '.join(f'{k} {old!r} -> {new!r}' for k, (old, new) in p['changed'].items()))
    for rel in p.get('conflicts') or []:                 # U-7-5: say before the diff that --write will refuse
        print(f'! {rel} changed since the search and the patch does not merge with your edits: --write will refuse '
              '(the diff below is the search\'s version against your file)')
    print(p['diff'] or '(no file changes)')
    for m in p['manual']:
        print(f'  * {m}')
    copy = p.get('config_copy')                          # the config's copy, as config.file must name it
    copy_path = Path(p['root']) / copy if copy else None
    in_place = [Path(f['path']).name for f in p['files'] if Path(f['path']) != copy_path]
    if in_place:                                         # U-7-6: the config goes to a copy, patched files do not
        print(f'  * {", ".join(in_place)}: written in place (in a git repository, --branch NAME writes on a new branch)')
    if copy:                                             # U-1-24: how the applied configuration becomes the reference
        print(f'  * to measure the applied configuration with doctor / run, set config.file: {copy} in infrarsi.yaml '
              '(and give the flag / environment knobs above their new values as defaults)')
    print('written: ' + ', '.join(p['written']) if p.get('written') else 'dry run: add --write to write the files')


def cmd_optimize(a):
    """One-command 5-stage autopilot shell: init -> doctor -> run (B-Comp v4.1) -> report -> apply."""
    from infrarsi import search
    from infrarsi.contracts import InfrastructureError
    from infrarsi.project import autopilot
    from infrarsi.project.doctor import doctor
    from infrarsi.project.flow import run_project
    from infrarsi.project.llm import make_driver, parse as parse_llm, preflight

    if a.tolerance is not None and not (0 <= a.tolerance < 1):
        raise RequestError(f'--tolerance must be in [0, 1), got {a.tolerance}')
    ablate = None
    if a.ablate is not None:
        from infrarsi.search import ABLATIONS
        ablate = [x.strip() for x in a.ablate.split(',') if x.strip() and x.strip() != 'none']
        bad = [x for x in ablate if x not in ABLATIONS]
        if bad:
            raise RequestError(f'--ablate: unknown {", ".join(bad)} (known: {", ".join(sorted(ABLATIONS))})')

    # Stage 1: init (draft or load infrarsi.yaml)
    try:
        project, drafted, notes = autopilot.ensure_manifest(
            a.target, entry=a.entry, config=a.config, command=a.command, force=a.force
        )
    except ValueError as exc:
        raise RequestError(str(exc)) from None
    action_word = 'drafted' if drafted else 'loaded'
    _say(f'[infrarsi] stage 1/5 (init): {action_word} {project.path} '
         f'(framework {project.raw.get("framework")}, {len(project.knobs)} knobs, {project.gpus} GPU(s))')
    if a.verbose and notes:
        for n in notes:
            _say(f'  - {n}')
    project = _startup_questions(project, a, llm_spec=autopilot.detect_llm_spec(project, a.llm))

    # Resolve LLM backend & search strategy (defaulting to B-Comp v4.1 when an LLM is available)
    spec = autopilot.detect_llm_spec(project, a.llm)
    project.raw['llm'] = {k: v for k, v in spec.items() if k not in (getattr(a, 'llm_cleared', None) or ())}
    spec = _api_overrides(project, a)
    try:
        driver = make_driver(spec)
    except (ValueError, FileNotFoundError) as exc:
        if a.llm is not None:
            raise RequestError(f'llm: {exc}') from exc
        driver = None
    req_strategy = a.strategy
    if req_strategy in (None, 'auto'):
        manifest_strat = project.raw['search'].get('strategy')
        if manifest_strat not in (None, 'auto'):
            req_strategy = manifest_strat
        elif driver is not None:
            req_strategy = 'bcomp'
        else:
            req_strategy = 'auto'
    try:
        strategy_name = search.resolve(req_strategy, has_llm=driver is not None, knobs=project.knobs)
    except ValueError as exc:
        raise RequestError(str(exc)) from None
    if driver is None and strategy_name in ('greedy', 'agent'):
        raise RequestError(f'the {strategy_name} strategy needs an LLM backend (manifest llm.backend or --llm)')
    if strategy_name == 'optico':
        from infrarsi.search.optico import problem
        why = problem(driver)
        if why:
            raise RequestError(why)
    if driver is not None and _calls_model(project, strategy_name, driver):
        model_label = getattr(driver, 'model', None) or 'the model'
        _say(f'llm: one test call to {model_label} before any run ...')
        pf = preflight(driver)
        if pf['refused']:
            if a.llm is not None:
                raise RequestError(f'llm: {model_label} {_llm_trouble(parse_llm(project.raw.get("llm")), driver, pf["error"])}; '
                                   'fix `llm` or run with --llm none')
            _say(f'  ! llm: {model_label} refused test call ({pf["error"]}); falling back to non-LLM search')
            driver = None
            strategy_name = search.resolve('auto', has_llm=False, knobs=project.knobs)
        elif not pf['ok']:
            _say(f'llm: a test call failed ({str(pf["error"])[:200]}); the search goes on and retries each call')
    driver = _with_debate(project, driver, a, strategy_name)
    if driver is not None and _calls_model(project, strategy_name, driver):
        _preflight_debate(driver)
    project, requirements = _apply_requirements(a, project, driver)

    # Stage 2: doctor (baseline verification & noise/repeat floor calibration)
    if a.no_doctor:
        _say('[infrarsi] stage 2/5 (doctor): skipped (--no-doctor)')
    else:
        _warn_if_locked(project.root)
        try:
            project, rep = _retry_launch(project, a, driver,
                lambda p: doctor(p, log=lambda m: None, experience=a.experience), doctor=True)
        except InfrastructureError as exc:
            raise SystemExit(f'doctor: the hosts could not be reached: {exc}\n'
                             'check that `ssh <host>` works without a password prompt (execution.hosts), then run again')
        status_str = 'OK' if rep['valid'] else 'NOT VALID'
        _say(f'[infrarsi] stage 2/5 (doctor): reference run {status_str} '
             f'({_num(rep["step_time"])} s per update, {rep["steps_completed"]} updates logged{_where(project)})')
        if not rep['valid']:
            for r in rep.get('reasons') or []:
                _say(f'  ! not valid: {r}')
            return 1

    # Stage 3: run (micro search algorithm + promotion verifier)
    budget = {'screens': a.screens} if a.screens else {}
    if a.budget_usd is not None:
        budget['llm_usd'] = float(a.budget_usd)
    if a.budget_tokens is not None:
        budget['llm_tokens'] = int(a.budget_tokens)
    if a.wall_hours is not None:
        budget['wall_seconds'] = float(a.wall_hours) * 3600
    for name in ('target_speedup', 'target_mfu'):
        if getattr(a, name, None) is not None:
            budget[name] = getattr(a, name)
    budget = budget or None
    log = print if a.verbose else (lambda m: print(m) if m.startswith('[infrarsi]') else None)
    _say(f'[infrarsi] stage 3/5 (run): searching with strategy {strategy_name}')
    old = None
    if hasattr(signal, 'SIGTERM') and threading.current_thread() is threading.main_thread():
        old = signal.signal(signal.SIGTERM, _interrupt)
    try:
        with _run_lock(project.root):
            project, res = _retry_launch(project, a, driver, lambda p: run_project(
                p, strategy=strategy_name, driver=driver, tolerance=a.tolerance, budget=budget,
                out_root=a.out, log=log, capability=a.capability, experience=a.experience, ablate=ablate,
                on_interrupt=a.on_interrupt, requirements=requirements, live=not (a.yes or a.interactive is False)
            ))
    finally:
        if old is not None:
            signal.signal(signal.SIGTERM, old)

    # Stage 4: report
    run_dir = Path(res['run_dir'])
    promo = res.get('promotion') or {}
    words = OUTCOME_WORDS.get(
        'interrupted_promoted' if res['outcome'] == 'interrupted' and promo.get('promoted') else res['outcome'], ''
    )
    _say(f'[infrarsi] stage 4/5 (report): {project.name}: {res["outcome"]}'
         + (f', x{promo["speedup"]:.3f} (re-run)' if promo.get('speedup') else '')
         + f'{words} -> {run_dir / "REPORT.md"}')

    # Stage 5: apply
    return _adopt(project, res, a, 'stage 5/5 (apply)')


def _adopt(project, res: dict, a, stage: str) -> int:
    """The last stage of `optimize` and `finish`: a promoted change written (unless --dry-run), the optimized launch script,
    the console briefing; the exit status of the command."""
    from infrarsi.project import autopilot
    from infrarsi.project.apply import apply as apply_run
    run_dir = Path(res['run_dir'])
    promo = res.get('promotion') or {}
    ap_res = None
    opt_script = None
    if promo.get('promoted'):
        try:
            ap_res = apply_run(run_dir, write=not a.dry_run, branch=a.branch)
        except RuntimeError as exc:
            raise RequestError(str(exc)) from None
        opt_script = autopilot.write_optimized_launch(project, run_dir, ap_res) if not a.dry_run else None
        if ap_res.get('changed'):
            _say('the change: ' + ', '.join(f'{k} {old_v!r} -> {new_v!r}' for k, (old_v, new_v) in ap_res['changed'].items()))
        if ap_res.get('diff'):
            _say(ap_res['diff'])
        for m in ap_res.get('manual') or []:
            _say(f'  * {m}')
        if a.dry_run:
            _say(f'[infrarsi] {stage}: dry run (--dry-run: no files written)')
        else:
            written = ', '.join(ap_res.get('written') or []) or 'launch flags/env'
            script_msg = f'; optimized launch script: {opt_script}' if opt_script else ''
            _say(f'[infrarsi] {stage}: written: {written}{script_msg}')
    else:
        _say(f'[infrarsi] {stage}: nothing to apply ({res["outcome"]})')

    if res.get('reference'):
        from infrarsi.project.report import format_console_briefing
        _say(format_console_briefing(res, applied=ap_res, opt_script=opt_script, dry_run=a.dry_run))

    if res['outcome'] == 'interrupted':
        return 130
    return 1 if res['outcome'] in ('reference_failed', 'not_verifiable', 'infrastructure_failure') else 0


NO_REQUIREMENTS = 'no requirements: give --text or --file, or a requirements block in infrarsi.yaml'


def cmd_requirements(a):
    """Compile the requirements and say what they mean for this project; no run."""
    from infrarsi.project import Project
    from infrarsi.project.llm import make_driver, override
    from infrarsi.project.requirements import RequirementsError, blocking, compile as compile_requirements, explain, manifest_texts
    project = Project.load(a.manifest)
    texts = manifest_texts(project) + list(a.text or [])
    if a.file:
        try:
            texts.append(Path(a.file).read_text(encoding='utf-8'))
        except OSError as exc:
            raise RequestError(f'--file: {exc}') from None
    if not texts:
        raise RequestError(NO_REQUIREMENTS)
    try:
        if a.llm is not None:
            project.raw['llm'] = override(project.raw.get('llm'), a.llm)
        driver = make_driver(_api_overrides(project, a))
        record = compile_requirements(project, texts, driver=driver, log=_say, refresh=a.refresh)
        if record is None:
            raise RequestError(NO_REQUIREMENTS)
    except RequirementsError as exc:
        raise RequestError(str(exc)) from None
    except (ValueError, FileNotFoundError) as exc:
        raise RequestError(f'llm: {exc}') from exc
    if a.json:
        print(json.dumps(record, ensure_ascii=False, indent=1, default=str))
    else:
        for line in explain(record):
            _say(line)
    return 1 if blocking(record) else 0


def cmd_finish(a):
    """Re-verify the best candidates of a run that ended before its own promotion, and adopt the winner."""
    from infrarsi.project.finish import FinishError, finish_run, project_of
    run, picked = _run_dir(a.run_dir, 'search/receipts/c0.json')
    if picked:
        print(f'run: {run}')
    project = project_of(run)
    log = print if a.verbose else (lambda m: print(m) if m.startswith('[infrarsi]') else None)
    old = None
    if hasattr(signal, 'SIGTERM') and threading.current_thread() is threading.main_thread():
        old = signal.signal(signal.SIGTERM, _interrupt)
    try:
        with _run_lock(project.root):
            res = finish_run(run, project=project, log=log)
    except FinishError as exc:
        raise RequestError(str(exc)) from None
    finally:
        if old is not None:
            signal.signal(signal.SIGTERM, old)
    return _adopt(project, res, a, 'apply')


def cmd_agent(a):
    """A run driven from a coding-agent session (project/agent_session.py: start, next, answer, status, note, stop, resume)."""
    from infrarsi.project.agent_session import AgentError, cli
    try:
        return cli(a.rest)
    except AgentError as exc:
        raise RequestError(str(exc)) from None


def cmd_validate(a):
    from infrarsi.harness import observe, recheck_against, run_screen
    if a.collect:
        run_id, att = a.collect.split('/')
        return _dump(observe(run_id, att), a.out)
    if a.recheck:
        rec = json.loads(Path(a.recheck).read_text(encoding='utf-8'))
        ref = json.loads(Path(a.reference).read_text(encoding='utf-8'))
        return _dump(recheck_against(rec, ref), a.out)
    task = TaskSpec.load(a.task)
    cand = json.loads(Path(a.candidate).read_text(encoding='utf-8'))
    fr = json.loads(Path(a.fr).read_text(encoding='utf-8'))
    gpn = task.gpus if task.nodes == 1 else task.gpus // task.nodes
    order = ['FR', 'cand'] if (task.raw.get('dev_order') or 0) % 2 == 0 else ['cand', 'FR']   # alternate by task parity
    out = {}
    run_id = a.run_id or f'CLI-{task.id}-locked'
    for who in order:
        # FR is the pair's numerics reference; the candidate compares against it (now, or after it via recheck_against)
        fr_done = out.get('FR') if (out.get('FR') or {}).get('valid') else None     # an invalid FR is no reference
        req = RunRequest(run_id=run_id, purpose='score', task=task.raw, config=fr if who == 'FR' else cand,
                         nodes=_nodes(a.nodes), gpus_per_node=gpn, input_plan='B', wall_seconds_cap=int(a.wall_hours * 3600),
                         numerics_role='reference' if who == 'FR' else 'candidate',
                         reference_receipt=(fr_done or {}).get('harness', {}).get('stored') if who == 'cand' else None)
        out[who] = run_screen(req, reference=fr_done if who == 'cand' else None)
    if order[0] == 'cand' and out['FR']['valid']:
        out['cand'] = recheck_against(out['cand'], out['FR'])
    from infrarsi.scoring import margin_of, score_task
    t = lambda r: (r.get('step_time') or {}).get('mean_window_max_rank_s') if r['valid'] else None   # noqa: E731
    out['score'] = score_task(task.id, a.arm or 'cand', t_fr_locked=t(out['FR']), t_cand_locked=t(out['cand']),
                              fr_rsd=(out['FR'].get('step_time') or {}).get('rsd_window'))
    out['margin'] = margin_of(out['FR'])
    _dump({'order': order, 'score': out['score'], 'FR': out['FR']['attempt_id'], 'cand': out['cand']['attempt_id'],
           'cand_valid': out['cand']['valid'], 'fr_valid': out['FR']['valid']}, a.out)


def cmd_export(a):
    from infrarsi.reward import breakeven_from_receipts, evaluate
    recs = [json.loads(Path(p).read_text(encoding='utf-8')) for p in a.receipts]
    ref = json.loads(Path(a.fr).read_text(encoding='utf-8')) if a.fr else None
    rows = []
    for r in recs:
        rv = evaluate(r, ref)
        row = {'attempt_id': r['attempt_id'], 'task': r['task_id'], 'config_digest': r['config_digest'], 'valid': rv['valid'],
               'R': rv['R'], 'StepTime': rv['metrics']['StepTime'], 'violations': rv['violations']}
        if ref and rv['valid'] and a.search_gpuh is not None:
            row['breakeven'] = breakeven_from_receipts(a.search_gpuh, ref, r)
        rows.append(row)
    _dump({'schema': 'infrarsi-export-v1', 'utc': utc_now(), 'rows': rows}, a.out)


def cmd_benchmark(a):
    from infrarsi.dispatch import Dispatcher
    Dispatcher(include_p2=a.include_p2).loop(interval_s=a.interval)


def cmd_redteam(a):
    mod = _needs('infrarsi.redteam', 'redteam')
    if not hasattr(mod, 'main'):
        raise NotImplementedError('infrarsi.redteam.main is not implemented yet')
    mod.main(a.rest)


def cmd_dispatch(a):
    from infrarsi.dispatch import Dispatcher, format_plan
    store = _store()
    if a.mark:
        store.set_marker(a.mark, a.note or '')
        return print(f'marker {a.mark} set')
    if a.unmark:
        store.clear_marker(a.unmark)
        return print(f'marker {a.unmark} cleared')
    if a.unpause:
        store.unpause()
        return print('queue unpaused')
    if a.reset:
        store.update_run(a.reset, status=a.to, note=f'reset by CLI {utc_now()}')
        return print(f'{a.reset} -> {a.to}')
    if a.stop or a.confirm_stop:
        from infrarsi.harness import request_stop
        target = a.stop or a.confirm_stop
        reason = a.reason or ('confirmed monitor STOP proposal' if a.confirm_stop else 'operator stop')
        d = EXECUTION_DIR / 'queue' / 'stop'
        d.mkdir(parents=True, exist_ok=True)
        name = target.split('/')[-1]
        (d / f'{name}.json').write_text(json.dumps({'reason': reason, 'utc': utc_now(), 'by': 'cli'}), encoding='utf-8')
        if '/' in target:
            print(request_stop(f'{RemotePaths.runs}/{target}', reason, by='cli'))
        if a.confirm_stop:
            p = EXECUTION_DIR / 'alerts' / f'STOP_REQUEST_{name}.json'
            if p.exists():
                req = json.loads(p.read_text(encoding='utf-8'))
                req.update(status='confirmed', confirmed_utc=utc_now())
                p.write_text(json.dumps(req, indent=1), encoding='utf-8')
        return print(f'stop requested for {target}')
    disp = Dispatcher(include_p2=a.include_p2, enabled=set(a.enable or []), live=a.live)
    if a.status:
        return _dump(store.state(), a.out)
    if a.once:
        return print('started:', disp.step(check_downloads=a.check_downloads))
    if a.loop:
        return disp.loop(interval_s=a.interval, check_downloads=a.check_downloads)
    p = disp.plan(check_downloads=a.check_downloads)
    print(format_plan(p, disp.matrix))
    if a.out:
        _dump(p, a.out)


def cmd_monitor(a):
    from infrarsi.monitor import run_once
    while True:
        res = run_once()
        print(f'{utc_now()} emitted {len(res["emitted"])} alerts, open {len(res["open"])}, paused={bool(res["paused"])}', flush=True)
        if not a.loop:
            return
        time.sleep(a.interval)


def _version() -> str:
    """The installed distribution's version (`pip install -e .` records it); a checkout on PYTHONPATH has none."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version('infrarsi')
    except PackageNotFoundError:
        return 'unknown (not installed)'


ALL_CMDS = ('optimize', 'init', 'doctor', 'run', 'report', 'apply', 'finish', 'requirements', 'exp', 'agent', *RESEARCH)


class _Parser(argparse.ArgumentParser):
    """ArgumentParser that routes `infrarsi <script.sh|train.py|dir>` directly to `optimize` when the first
    argument is a target path or `optimize` flag rather than a subcommand name."""

    def parse_args(self, args=None, namespace=None):
        raw = [str(x) for x in (sys.argv[1:] if args is None else args)]
        if raw and raw[0] == 'agent':                    # its own parser reads the rest (an option first too: `agent -h`)
            return argparse.Namespace(cmd='agent', rest=raw[1:], fn=cmd_agent)
        if raw and raw[0] not in ALL_CMDS and raw[0] not in ('-h', '--help', '--version'):
            raw = ['optimize', *raw]
        return super().parse_args(raw, namespace)


def build() -> argparse.ArgumentParser:
    from infrarsi.agent.budget import target_value
    from infrarsi.search.token_governor import parse_duration_hours, parse_tokens
    ap = _Parser(prog='infrarsi', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                 epilog=f'research harness of the WINH UpdateBench matrix (not listed above): {", ".join(RESEARCH)}')
    ap.add_argument('--version', action='version', version=f'infrarsi {_version()}')
    sub = ap.add_subparsers(dest='cmd', required=True,
                            metavar='{optimize,init,doctor,run,report,apply,finish,requirements,exp,agent}')
    xp = 'experience packs to use: auto (all; default), off, or pack-id patterns (comma-separated, e.g. "h20-*")'
    mf = 'infrarsi.yaml or the directory that holds it (default: the current directory)'
    oi = ('what Ctrl-C / SIGTERM does during the search: verify (stop the search, re-verify the best candidates so far; a second '
          'interrupt stops that) or stop (end at once, nothing promoted) (default: search.on_interrupt, verify)')

    def question_args(p):
        p.add_argument('--interactive', action=argparse.BooleanOptionalAction, default=None,
                       help='show a short editable startup checklist (Enter starts; more shows advanced settings) (default: '
                            'on a terminal, yes; in a script or a pipe, no)')
        p.add_argument('-y', '--yes', action='store_true', help='no questions: use the settings as they are')

    def requirement_args(p):
        p.add_argument('--requirements', action='append', metavar='TEXT',
                       help='a requirement in your own words, for this run (repeatable; any language): it can only narrow the '
                            'search or add checks, see `infrarsi requirements`')
        p.add_argument('--requirements-file', metavar='FILE', help='requirements in your own words, from a text file')
        p.add_argument('--requirements-partial', action='store_true',
                       help='run even when some requirements cannot be honoured (recorded in the report)')
        p.add_argument('--no-requirements', action='store_true', help='ignore the manifest\'s requirements for this run')

    def debate_args(p):
        p.add_argument('--debate', action=argparse.BooleanOptionalAction, default=None,
                       help='dual-model debate at the proposal calls: a second model proposes too and the main model '
                            'judges both blind (default: debate.enabled, off)')
        p.add_argument('--second-llm', metavar='BACKEND[:MODEL]', help='the debate\'s second model (turns the debate on); '
                                                                       'same forms as --llm')
        p.add_argument('--verifier-llm', metavar='BACKEND[:MODEL]', help='the model that checks code patches before they run '
                                                                         '(default: the model that did not write the patch; none: no check)')
    p = sub.add_parser('optimize', help='one-command autopilot: init -> doctor -> run (B-Comp v4.1) -> report -> apply')
    p.add_argument('target', nargs='?', default='.',
                   help='a launch script (run.sh), Python training script (train.py), codebase directory, or infrarsi.yaml')
    p.add_argument('--config', help='the config file of the training job (relative to the repository root)')
    p.add_argument('--entry', help='the training entry script (relative to the repository root)')
    p.add_argument('--command', help='the command line the job runs with today, quoted (its flags become the knob defaults)')
    p.add_argument('--force', action='store_true', help='re-draft infrarsi.yaml even when one already exists')
    p.add_argument('--no-doctor', action='store_true', help='skip Stage 2 (doctor) and go straight to Stage 3 (run)')
    p.add_argument('--dry-run', action='store_true', help='preview Stage 5 (apply) diff without writing files')
    p.add_argument('--branch', help='git branch to write applied changes on (git repositories)')
    p.add_argument('--strategy', help='search strategy (default: bcomp / B-Comp v4.1 with an LLM; tpe/random without one; '
                                      'also supports bcompv4.1, hls3, hls2, hls, doe+hls, doe, tpe, random, '
                                      'greedy, agent, optico)')
    p.add_argument('--llm', help='backend[:model]: anthropic, claude-cli/claude, codex-cli/codex, deepseek, openai, openai-compat, vllm, none (default: '
                                 'auto-detected from manifest or environment)')
    p.add_argument('--tolerance', type=float, help='accepted relative loss deviation (default: verify.tolerance, 0.03)')
    p.add_argument('--screens', '--trials', dest='screens', type=int,
                   help='search budget in full runs (default: search.budget.screens, 16)')
    p.add_argument('--capability', type=float, help='0..1: how much to trust the LLM (default: inferred from the model)')
    p.add_argument('--budget-usd', type=float, help='search budget in LLM dollars (default: search.budget.llm_usd)')
    p.add_argument('--budget-tokens', '--token-budget', dest='budget_tokens', type=parse_tokens,
                   help='search token budget: integer, 500k, or 2M (default: search.budget.llm_tokens)')
    p.add_argument('--wall-hours', '--max-wallclock', dest='wall_hours', type=parse_duration_hours,
                   help='search wall-clock budget: hours, 2h, 90m, or 3600s (default: search.budget.wall_hours)')
    p.add_argument('--target-speedup', type=lambda v: target_value('target_speedup', float(v)),
                   help='stop after a valid full screen reaches this speedup over the reference (> 1)')
    p.add_argument('--target-mfu', type=lambda v: target_value('target_mfu', float(v)),
                   help='stop at this MFU: fraction or percent (0.45 or 45)')
    p.add_argument('--api-base', help='override llm.base_url for the selected API backend')
    p.add_argument('--api-key-env', help='name of the environment variable containing the API key')
    p.add_argument('--out', help='runs directory (default: .infrarsi/runs in the workdir)')
    p.add_argument('--experience', help=xp)
    p.add_argument('--ablate', help='research: comma-separated components to turn off ("none" = all on)')
    p.add_argument('--on-interrupt', choices=['verify', 'stop'], help=oi)
    debate_args(p)
    requirement_args(p)
    question_args(p)
    p.add_argument('-v', '--verbose', action='store_true', help='print every progress line, not only the milestones')
    p.set_defaults(fn=cmd_optimize)
    # the research commands are added without help=, so argparse lists only the product commands
    # U-1-2: every argument says what it is and its default
    p = sub.add_parser('init', help='draft infrarsi.yaml for a training repository')
    p.add_argument('dir', nargs='?', default='.', help='the training repository or launch/entry script (default: the current directory)')
    p.add_argument('--config', help='the config file of the training job (relative to DIR)')
    p.add_argument('--entry', help='the training entry script (relative to DIR)')
    p.add_argument('--command', help='the command line the job runs with today, quoted (its flags become the knob defaults)')
    p.add_argument('--force', action='store_true', help='replace an existing infrarsi.yaml (and infrarsi_launch.sh)')
    p.add_argument('--requirements', action='append', metavar='TEXT', help='a requirement in your own words, kept in the '
                                                                           'manifest (repeatable); `run` compiles them')
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser('doctor', help='one reference run: signals, checklist coverage, noise, experience coverage')
    p.add_argument('manifest', nargs='?', default='.', help=mf)
    p.add_argument('--experience', help=xp)
    p.set_defaults(fn=cmd_doctor)
    p = sub.add_parser('run', help='reference run, search, verified promotion, report (a project or a suite)')
    p.add_argument('manifest', nargs='?', default='.', help=mf + ', or a suite file (schema: infrarsi-suite-v1)')
    p.add_argument('--strategy', help='auto (default: doe+hls or hls with an LLM; tpe without one, random when optuna is '
                                      'not installed), hls, hls2 (hls with the search fix), hls3 (hls2 with memory-aware '
                                      'search: --ablate names its hls3_* components), doe, doe+hls, tpe, random, greedy, '
                                      'agent, optico')
    p.add_argument('--llm', help='backend[:model]: anthropic, claude-cli/claude, codex-cli/codex, deepseek, openai, openai-compat, vllm, none (default: '
                                 "the manifest's llm block; without one, no model)")
    p.add_argument('--tolerance', type=float, help='accepted relative loss deviation (default: verify.tolerance, 0.03)')
    p.add_argument('--screens', '--trials', dest='screens', type=int,
                   help='search budget in full runs (default: search.budget.screens, 16)')
    p.add_argument('--capability', type=float, help='0..1: how much to trust the LLM (default: inferred from the model)')
    p.add_argument('--budget-usd', type=float, help='search budget in LLM dollars (default: search.budget.llm_usd); with '
                                                    '--resume the run stops when the TOTAL it used reaches it')
    p.add_argument('--budget-tokens', '--token-budget', dest='budget_tokens', type=parse_tokens,
                   help='search token budget: integer, 500k, or 2M (default: search.budget.llm_tokens)')
    p.add_argument('--wall-hours', '--max-wallclock', dest='wall_hours', type=parse_duration_hours,
                   help='search wall-clock budget: hours, 2h, 90m, or 3600s (default: search.budget.wall_hours)')
    p.add_argument('--target-speedup', type=lambda v: target_value('target_speedup', float(v)),
                   help='stop after a valid full screen reaches this speedup over the reference (> 1)')
    p.add_argument('--target-mfu', type=lambda v: target_value('target_mfu', float(v)),
                   help='stop at this MFU: fraction or percent (0.45 or 45)')
    p.add_argument('--api-base', help='override llm.base_url for the selected API backend')
    p.add_argument('--api-key-env', help='name of the environment variable containing the API key')
    p.add_argument('--resume', metavar='RUN_DIR',
                   help='continue RUN_DIR (or latest for this project, with --out for a custom runs directory) from its search-state checkpoint (<run>/search/state): the amounts it used '
                        'count towards the budget now given (--screens, --budget-usd, --budget-tokens, --wall-hours, or the '
                        "manifest's), the promotion and the report are a normal run's; the manifest is the run's project's")
    p.add_argument('--remeasure-reference', action='store_true',
                   help='with --resume: run the reference configuration again first and record both (the earlier runs are '
                        'not rescaled)')
    p.add_argument('--allow-code-change', action='store_true',
                   help='with --resume: resume although this tree is not the code version that made the checkpoint (recorded)')
    p.add_argument('--out', help='runs directory (default: .infrarsi/runs in the workdir)')
    p.add_argument('--experience', help=xp)
    p.add_argument('--ablate', help='research: comma-separated components to turn off (replaces search.ablate; '
                                    '"none" = all on), e.g. probe_first, or hls3_feasibility, hls3_bundles, ... (strategy '
                                    'hls3; every hls3_* component off is hls2)')
    p.add_argument('--on-interrupt', choices=['verify', 'stop'], help=oi)
    debate_args(p)
    requirement_args(p)
    question_args(p)
    p.add_argument('-v', '--verbose', action='store_true',
                   help='print every progress line, not only the milestones (all of them are in <run dir>/run.log)')
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser('exp', help='experience packs (infrarsi-exp/1): validate, stats, export a run for sharing')
    p.add_argument('action', choices=['validate', 'stats', 'export'])
    p.add_argument('paths', nargs='*')
    p.add_argument('--out')
    p.add_argument('--pack-id')
    p.add_argument('--title')
    p.add_argument('--license')
    p.add_argument('--contributor')
    p.add_argument('--force', action='store_true')
    p.set_defaults(fn=cmd_exp)
    rd = ('the run: its directory .infrarsi/runs/<name>-<UTC time> (run prints it at the end), a file in it '
          '(REPORT.md, result.json, run.log) or a project directory (its newest run); default: the newest run under '
          '.infrarsi/runs of the current directory, the only place searched (a run written elsewhere with run --out '
          'needs its directory)')
    p = sub.add_parser('report', help='print the report of a run directory')
    p.add_argument('run_dir', nargs='?', help=rd)
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser('apply', help='show or write the recommended change')
    p.add_argument('run_dir', nargs='?', help=rd)
    p.add_argument('--write', action='store_true', help='write the files (without it: show the change only)')
    p.add_argument('--branch', help='git branch to write on (git repositories)')
    p.set_defaults(fn=cmd_apply)
    p = sub.add_parser('requirements', help='compile requirements in your own words and show what they mean; runs nothing')
    p.add_argument('manifest', nargs='?', default='.', help=mf)
    p.add_argument('--text', action='append', help='a requirement in your own words (repeatable), besides the manifest\'s')
    p.add_argument('--file', help='requirements from a text file')
    p.add_argument('--llm', help='backend[:model] that reads them (default: the manifest\'s llm block)')
    p.add_argument('--api-base', help='override llm.base_url for the selected API backend')
    p.add_argument('--api-key-env', help='name of the environment variable containing the API key')
    p.add_argument('--refresh', action='store_true', help='compile again even when a cached result exists')
    p.add_argument('--json', action='store_true', help='print the whole record as JSON')
    p.set_defaults(fn=cmd_requirements)
    p = sub.add_parser('finish', help='re-verify the best candidates of a run that stopped early, and adopt the winner')
    p.add_argument('run_dir', nargs='?', help=rd)
    p.add_argument('--dry-run', action='store_true', help='show the change without writing files')
    p.add_argument('--branch', help='git branch to write on (git repositories)')
    p.add_argument('-v', '--verbose', action='store_true', help='print every progress line, not only the milestones')
    p.set_defaults(fn=cmd_finish)
    p = sub.add_parser('agent', add_help=False,
                       help='a run driven from a coding-agent session (Claude Code, Codex): start, next, answer, status, '
                            'note, stop, resume (`infrarsi agent -h`)')
    p.add_argument('rest', nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_agent)
    p = sub.add_parser('discover')
    p.add_argument('--nodes')
    p.add_argument('--task')
    p.add_argument('--offline', action='store_true')
    p.add_argument('--refresh', action='store_true')
    p.add_argument('--out')
    p.set_defaults(fn=cmd_discover)
    p = sub.add_parser('prepare')
    p.add_argument('--tasks', help='comma-separated task ids (default: all public tasks)')
    p.add_argument('--node', default='WINH0')
    p.set_defaults(fn=cmd_prepare)
    p = sub.add_parser('baseline')
    p.add_argument('--task', required=True)
    p.add_argument('--arm', choices=['FR', 'ED'], default='FR')
    p.add_argument('--nodes')
    p.add_argument('--run-id')
    p.add_argument('--wall-hours', type=float, default=4.0)
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--out')
    p.set_defaults(fn=cmd_baseline)
    p = sub.add_parser('validate')
    p.add_argument('--task')
    p.add_argument('--candidate')
    p.add_argument('--fr')
    p.add_argument('--arm')
    p.add_argument('--nodes')
    p.add_argument('--run-id')
    p.add_argument('--wall-hours', type=float, default=2.0)
    p.add_argument('--collect', help='RUN_ID/ATTEMPT_ID to observe and collect')
    p.add_argument('--recheck')
    p.add_argument('--reference')
    p.add_argument('--out')
    p.set_defaults(fn=cmd_validate)
    p = sub.add_parser('export')
    p.add_argument('receipts', nargs='+')
    p.add_argument('--fr')
    p.add_argument('--search-gpuh', type=float)
    p.add_argument('--out')
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser('benchmark')
    p.add_argument('--include-p2', action='store_true')
    p.add_argument('--interval', type=float, default=60)
    p.set_defaults(fn=cmd_benchmark)
    p = sub.add_parser('redteam')
    p.add_argument('rest', nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_redteam)
    p = sub.add_parser('dispatch')
    p.add_argument('--dry-run', action='store_true', help='print the next runnable set (default action)')
    p.add_argument('--live', action='store_true', help='include live occupancy and leases (read-only SSH)')
    p.add_argument('--check-downloads', action='store_true')
    p.add_argument('--once', action='store_true')
    p.add_argument('--loop', action='store_true')
    p.add_argument('--interval', type=float, default=60)
    p.add_argument('--status', action='store_true')
    p.add_argument('--mark')
    p.add_argument('--unmark')
    p.add_argument('--note')
    p.add_argument('--unpause', action='store_true')
    p.add_argument('--reset')
    p.add_argument('--to', default='pending')
    p.add_argument('--stop', help='RUN_ID or RUN_ID/ATTEMPT_ID')
    p.add_argument('--confirm-stop', help='RUN_ID/ATTEMPT_ID from a monitor STOP_REQUEST')
    p.add_argument('--reason')
    p.add_argument('--include-p2', action='store_true')
    p.add_argument('--enable', action='append')
    p.add_argument('--out')
    p.set_defaults(fn=cmd_dispatch)
    p = sub.add_parser('monitor')
    p.add_argument('--loop', action='store_true')
    p.add_argument('--interval', type=float, default=300)
    p.set_defaults(fn=cmd_monitor)
    return ap


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):               # U-5-3: each line at once, also into a pipe (`| tee`, nohup)
        sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    a = build().parse_args(argv)
    from infrarsi.project.detect import InitError
    from infrarsi.project.manifest import ManifestError
    try:
        code = a.fn(a)
    except KeyboardInterrupt:                          # before or after a search (a search writes its report)
        print(f'infrarsi {a.cmd}: interrupted', file=sys.stderr)
        raise SystemExit(130)
    except NotImplementedError as e:
        print(f'not implemented: {e}', file=sys.stderr)
        raise SystemExit(3)
    except (InitError, ManifestError, RequestError) as e:          # the request itself; anything else keeps its traceback
        if os.environ.get('INFRARSI_DEBUG'):
            raise
        print(f'infrarsi {a.cmd}: {e}', file=sys.stderr)
        raise SystemExit(2)
    except OSError as e:                                 # U-6-2: Windows' path limit, in one line (still exit 1)
        if getattr(e, 'winerror', None) != 206 or os.environ.get('INFRARSI_DEBUG'):
            raise
        print(f'infrarsi {a.cmd}: a path is longer than Windows allows ({e.filename}): move the '
              'project to a shorter path, or enable long paths (LongPathsEnabled)', file=sys.stderr)
        raise SystemExit(1)
    if type(code) is int and code:                       # a run that failed (cmd_run): scripts and CI see it
        raise SystemExit(code)
