"""`infrarsi doctor`: one reference run of the current configuration and what the verifier can see in it."""
from __future__ import annotations

from pathlib import Path

import difflib
import math

from ..search.engine import pooled_sigma_pair
from ..verify import coverage
from ..verify.checklist import UNVERIFIABLE_WORKLOAD
from ..verify.verifier import loss_repeat_note
from .flow import apply_links, make_verifier, noise_pairs, profile, run_noise
from .manifest import Project, get_path
from .runner import ProjectRunner

SIGNALS = ('loss', 'grad_norm', 'lr', 'tokens', 'samples', 'data', 'step_time', 'mem_gib')


def unread_paths(project: Project) -> list[str]:
    """Path knobs whose key the config file does not have (U-7-1): a run writes the key all the same, so a typo in the
    path makes a knob that changes nothing the job reads - and nothing else would say so."""
    doc = project.base_config()
    if not isinstance(doc, dict):
        return []
    keys = []
    stack = [('', doc)]
    while stack:                                        # every leaf key of the config, dotted
        prefix, d = stack.pop()
        for k, v in d.items():
            if isinstance(v, dict):
                stack.append((f'{prefix}{k}.', v))
            else:
                keys.append(f'{prefix}{k}')
    out = []
    for name, k in project.knobs.items():
        if 'path' in k and get_path(doc, k['path'], missing=...) is ...:
            near = difflib.get_close_matches(str(k['path']), keys, n=1)
            out.append(f'knobs.{name}.path {k["path"]}: {project.config_file.name} has no such key; a run writes it, '
                       'which changes the job only if it reads that key' + (f' (did you mean {near[0]}?)' if near else ''))
    return out


def doctor(project: Project, *, out_root=None, log=print, experience=None) -> dict:
    out = Path(out_root or project.root / '.infrarsi' / 'doctor').resolve()
    task = project.task() if (project.knobs or project.raw['code']['mutable']) else None
    runner = ProjectRunner(project, out_dir=out, log=log, skip=out)
    off = set(project.raw['search'].get('ablate') or [])
    unread = unread_paths(project)                   # before the run: a config that cannot be read costs no run
    ref = runner.run(task, project.reference_config(), None, 'doctor', 'doctor', candidate_id='c0',
                     bench='bandwidth' not in off)
    verifier = make_verifier(project)
    v = verifier(ref, None, ref)
    ev = ref['evidence']
    checks = verifier.screen_checks(ev, ev)          # this run against itself: which comparisons the signals support
    rows = ev.get('per_step') or []
    seen = {k: sum(1 for r in rows if k in r) for k in SIGNALS}
    warn = []
    crlf = project.crlf_scripts() if project.posix_hosts() else []
    if crlf:
        warn.append(f'{", ".join(crlf)}: Windows (CRLF) line endings: bash on the hosts reads every line with a trailing '
                    '\\r (convert: dos2unix, or git config core.autocrlf false and check the file out again)')
    if not rows:
        warn.append('no step lines were recognized in the log: add signals.patterns or use the probe')
    elif not ev.get('logged_every_step'):
        warn.append('not every update is logged: fewer comparison points for the loss and timing checks')
    if ev.get('clock_note'):
        warn.append(f'step time not on InfraRSI\'s clock: {ev["clock_note"]}')
    if rows and not seen['loss']:
        warn.append('no loss in the log: the loss check is required')
    if rows and verifier.workload_unverifiable(ev):   # D2-R211, D2-R204b: a whole search with every candidate rejected
        warn.append(UNVERIFIABLE_WORKLOAD + '; `run` stops before the search')
    by_id = {c['id']: c for c in checks}
    if rows and 'counter' in by_id['optimizer_steps']['detail'] and by_id['optimizer_steps']['status'] == 'unchecked':
        warn.append('no optimizer-step counter in the log: a candidate that skips or merges optimizer steps is invisible to '
                    'the optimizer_steps check (log `optim_step N` per update, or use the probe)')
    timing = next((c for c in v.checks if c['id'] == 'timing_consistent'), None)   # the reference's own verdict (F-SYS-1)
    if timing and timing['status'] == 'info':
        warn.append(timing['detail'])
    if (ev.get('rsd_window') or 0) > 0.05:
        warn.append(f'noisy step times (RSD {ev["rsd_window"]:.1%}): use more updates per run (workload.steps)')
    for w in (ev.get('gpu_check') or {}).get('contention') or []:
        warn.append(f'another job shares the GPUs ({w}): its load changes the step times the search compares')
    for w in (ev.get('gpu_check') or {}).get('ecc') or []:
        warn.append(f'{w}: the GPU may be failing (check it before a long search)')
    warn += unread
    # the noise the search will start from: earlier pairs of this project on this GPU type and this run's split half
    fixed = run_noise(project)
    pairs = noise_pairs(project) if fixed is None else []
    sp = math.sqrt(2) * fixed if fixed is not None else pooled_sigma_pair(pairs,
                                                                          [ev['split_half']] if ev.get('split_half') is not None else [])
    if sp is not None and 2 * sp > 0.02:
        warn.append(f'noise between two runs (sigma_pair) {sp:.2%}: a single run resolves only gains above {2 * sp:.1%}; '
                    'smaller ones show only in combination with other changes, or with more updates per run '
                    '(workload.steps)')
    noise = {'split_half': ev.get('split_half'), 'sigma_pair': sp, 'resolvable_gain': 2 * sp if sp is not None else None,
             'source': 'manifest search.run_noise' if fixed is not None else 'measured', 'earlier_pairs': len(pairs)}
    rep = verifier.loss_repeat(ev)                   # a loss that swings between updates (a diffusion job's sampled
    note = loss_repeat_note(rep, verifier.tolerance, doctor=True)   # timestep): what an unpaired window mean can resolve
    if note:
        warn.append(note)
    from ..experience import runs as exp_runs
    links = apply_links(project, runner, log=lambda m: None)        # before the plan: it scopes by the fabric
    prof, _ = profile(runner, task, project.reference_config(), ref, 'doctor', lambda m: None, off) if v.valid else (None, 0)
    pl = exp_runs.plan_for(project, ref, select=experience, log=lambda m: None) if v.valid else None
    oom = bool((ev.get('events') or {}).get('oom'))   # out of memory: the estimate says where the memory goes
    acc = None if 'accounting' in off or not (v.valid or oom) else exp_runs.accounting_for(project, ref)   # memory, MFU
    return {'experience': exp_runs.doctor_view(pl) if pl else None,'valid': v.valid, 'reasons': v.reasons, 'step_time': ev.get('step_time'), 'step_time_source': ev.get('step_time_source'),
            'accounting': acc(project.reference_config(), ev.get('step_time')) if acc else None, 'accounting_off': 'accounting' in off,
            'interconnect': {'detected': links, 'used': project.raw['hardware'].get('interconnect')},
            'gpu_check': ev.get('gpu_check'),
            'rsd_window': ev.get('rsd_window'), 'noise': noise, 'loss_repeat': rep, 'steps_completed': ev.get('steps_completed'), 'signals_seen': seen,
            'memory_peak_gib': ev.get('memory_peak_gib'), 'coverage': coverage(checks), 'warnings': warn, 'log': ref.get('log'), 'step_log': ref.get('step_log'),
            'phases': ev.get('phases') or prof, 'phases_source': 'log' if ev.get('phases') else 'profile' if prof else None,
            'log_tail': ref.get('log_tail')[-8:], 'checks': [{k: c[k] for k in ('id', 'status', 'detail')} for c in checks]}
