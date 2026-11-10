"""Per-update signals of a project run and the evidence block the verifier reads (revamp design B).

Sources: the training log (a built-in scanner that understands the usual `key: value` / `key=value` / `'key': value`
/ `key value` forms of Megatron-LM, torchtitan, nanoGPT-style and HF Trainer logs, plus optional explicit patterns
with a named group `v`) or the probe jsonl (`infrarsi.project.probe`). Step time is measured on InfraRSI's own
clock: the arrival time of each step's log line (or the probe's timestamp), never only what the codebase reports.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics

NUM = r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?'
VAL = rf'(?P<v>{NUM}|nan|NaN|-?inf|-?Infinity)'
ALIASES = {      # longest first inside each list; matched case-insensitively, not inside a longer identifier
    'loss': ['lm loss', 'train/loss', 'train_loss', 'training loss', 'loss'],
    'grad_norm': ['grad norm', 'grad_norm', 'gradnorm', 'gnorm'],
    'lr': ['learning rate', 'learning_rate', 'lr'],
    'optim_step': ['optimizer_step', 'optim_step', 'opt_step'],
    'tokens': ['tokens per step', 'tokens_per_step', 'tokens per update', 'num_tokens', 'ntokens', 'tokens'],
    'samples': ['global batch size', 'global_batch_size', 'samples per step', 'samples_per_step'],
    'step_time': ['elapsed time per iteration (ms)', 'step time', 'step_time', 'iter time', 'iter_time',
                  'time per step', 'time_per_step'],
    'mem_gib': ['max memory reserved', 'max_memory_reserved', 'memory reserved', 'peak memory', 'peak_memory',
                'peak_mem', 'memory', 'mem'],
}
STEP_RE = re.compile(r'(?i)(?<![\w/])(?:global[_ ]step|iteration|iter|step)\s*[:=#]?\s*\[?\s*(?P<v>\d+)(?:\s*/\s*\d+)?')
MFU_RE = re.compile(r'\bmfu\b[\'\"]?[\s:=]+(\d+(?:\.\d+)?|\.\d+)\s*(%)?', re.I)
DATA_RE = re.compile(r'(?i)(?<![\w/])[\'"]?(?:data|batch)[_ ]digest[\'"]?\s*[:=]?\s*[\'"]?(?P<v>[0-9a-f]{8,64})')
OOM_RE = re.compile(r'CUDA out of memory|OutOfMemoryError(?!: out of resource)|CUBLAS_STATUS_ALLOC_FAILED|cudaErrorMemoryAllocation|'
                    r'HIP out of memory|RESOURCE_EXHAUSTED')
COLLECTIVE_RE = re.compile(r'NCCL error|ncclInternalError|ncclSystemError|ncclUnhandledCudaError|ncclRemoteError|'
                           r'Watchdog caught collective operation timeout|ProcessGroupNCCL.*(?:[Tt]imeout|[Ee]rror)|'
                           r'DistBackendError|NCCL WARN .*[Tt]imeout')
TRACEBACK_RE = re.compile(r'^\s*(?:\[rank\d+\]:\s*)?Traceback \(most recent call last\)')   # a rank prefix allowed
# a line after which a training job does not go on: an OOM, a failed collective, a CUDA fault, a DeepEP failure, a worker
# killed by a signal. Such a run did not always end (2026-10-28): on 8 GPUs, torchrun reported every worker exited within
# 90 s of an OOM and yet leftover processes kept the job's output open until the 3,600 s timeout; on 64 GPUs, ranks sat
# silent after an NCCL error until the 2,400 s timeout. The runner stops a run that logged such a line and made no update
# since (runner.FatalWatch). An exception line alone is not one: a job can log a harmless one at start-up (an optional
# import) and go on; nor is Inductor's autotune "OutOfMemoryError: out of resource" (a config it skips).
FATAL_RE = re.compile(OOM_RE.pattern + r'|NCCL error|ncclInternalError|ncclSystemError|ncclUnhandledCudaError|'
                      r'ncclRemoteError|Watchdog caught collective operation timeout|DistBackendError|CUDA error: |'
                      r'illegal memory access|Signal \d+ \(SIG[A-Z]+\) received|'
                      # DeepEP's own failures: "DeepEP error: CPU recv timeout", "DeepEP combine RDMA receiver timeout",
                      # "DeepEP timeout check failed" (a healthy run's DeepEP lines name no error, failure or timeout)
                      r'(?i:deepep.*?(?:error|fail(?:ed|ure)?|timeout|timed out))')
CKPT_RE = re.compile(r'(?i)checkpoint')                  # a checkpoint save / load line that names an iteration
# a tqdm bar (`Steps:  10%|#  | 3/30 [00:03<00:29, 1.09s/it, lr=1e-5, step_loss=0.202]`) redraws its count: diffusers
# advances it before it sets the new postfix (and sets one after each micro-batch under accumulation), so the last
# redraw of a count holds its values (its first, the time); its 0/N is no update
TQDM_RE = re.compile(r'\|\s*\d+/\d+ \[\d+:\d+(?::\d+)?<')
# the line a failure names itself by: a Python exception (`torch.OutOfMemoryError: ...`, a rank prefix allowed), a
# shell's `...: No such file or directory` or a script's argument error; torchrun's summary of a failed child is not one
ERROR_RE = re.compile(r'^(?:\[rank\d+\]:\s*)?(?P<e>(?:[A-Za-z_]\w*\.)*[A-Z]\w*(?:Error|Exception)(?::.*)?'
                      r'|.*: (?:No such file or directory|command not found|Permission denied)'
                      r'|\S+\.py: error: .+)\s*$')                # argparse: `train.py: error: argument --x: ...`
LAUNCHER_ERRORS = ('ChildFailedError', 'CalledProcessError')
SUMMARY_RE = re.compile(r'\btrain_runtime\b')            # HF Trainer's whole-run summary dict
UNIT_S ={'ms': 1e-3, 's': 1.0, 'sec': 1.0, 'us': 1e-6}
UNIT_GIB = {'gib': 1.0, 'gb': 1e9 / 2 ** 30, 'mib': 1 / 1024, 'mb': 1e6 / 2 ** 30}
CLUMP_GAP_S = 1e-3          # log lines arriving closer than this are buffered, not real step completions
# Megatron's timer block after an iteration line (--timing-log-level >= 1): `    forward-backward ....: 2505.55` (option
# max) or `: (2501.20, 2505.55)` (minmax: the max is taken). Each value is the maximum across ranks, in ms.
MEGATRON_TIMER = re.compile(r'^\s+(?P<name>[a-z][a-z0-9]*(?:-[a-z0-9]+)*) \.{3,}:\s*\(?\s*(?:[-+\d.eE]+\s*,\s*)?'
                            r'(?P<v>[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*\)?\s*$')
# the timers that partition an update: Megatron-Core stops forward-backward after finalize_model_grads, so
# all-grads-sync is nested in it (and params-all-gather in the optimizer timer without --overlap-param-gather)
MEGATRON_TOP = ('forward-backward', 'optimizer')
MEGATRON_UPDATE_TIMERS = frozenset((                 # megatron/training/training.py: training_log's timers_to_log
    'forward-backward', 'forward-compute', 'backward-compute', 'batch-generator', 'forward-recv', 'forward-send',
    'backward-recv', 'backward-send', 'forward-send-forward-recv', 'forward-send-backward-recv',
    'backward-send-forward-recv', 'backward-send-backward-recv', 'forward-backward-send-forward-backward-recv',
    'layernorm-grads-all-reduce', 'embedding-grads-all-reduce', 'all-grads-sync', 'params-all-gather',
    'optimizer-copy-to-main-grad', 'optimizer-unscale-and-check-inf', 'optimizer-clip-main-grad', 'optimizer-count-zeros',
    'optimizer-inner-step', 'optimizer-copy-main-to-model-params', 'optimizer'))
PHASE_UNIT_MS = {'ms': 1.0, 's': 1000.0, 'sec': 1000.0, 'us': 1e-3}


def _alias_re(words) -> re.Pattern:
    """A signal's aliases as a key/value pattern: the key may carry a namespace (`train/grad_norm`, `optim/lr`), the
    value may stand in brackets (DeepSpeed's `lr=[1e-05]`), a unit may follow."""
    alt = '|'.join('[ _]'.join(re.escape(part) for part in re.split(r'[ _]', w)) for w in words)
    return re.compile(r'(?i)(?<![\w/])(?:[\w.-]+/)*[\'"]?(?:' + alt + r')[\'"]?\s*[:=]?\s*\[?\s*' + VAL
                      + r'\s*(?P<unit>ms|us|sec|s|GiB|GB|MiB|MB)?\s*\]?(?![\w.])')


ANSI_RE = re.compile(r'\x1b\[[0-9;?]*[ -/]*[@-~]')          # terminal colour / cursor codes around keys or values


def plain(text: str) -> str:
    return ANSI_RE.sub('', text) if '\x1b' in text else text


BUILTIN = {k: _alias_re(v) for k, v in ALIASES.items()}


def _value(raw: str, key: str, unit: str = '', context: str = ''):
    try:
        v = float(raw)
    except ValueError:
        return raw
    if not math.isfinite(v):
        return 'nan' if raw.lower().lstrip('-').startswith('nan') else 'inf'
    unit = (unit or '').lower()
    if key == 'step_time':
        return v * (UNIT_S.get(unit) or (1e-3 if '(ms)' in context.lower() else 1.0))
    if key == 'mem_gib':
        return v * UNIT_GIB.get(unit, 1.0)
    if key in ('tokens', 'samples') and v == int(v):
        return int(v)
    return v


def _group(m: re.Match, key: str):
    """The value group of a match: `v`, else the group named after the signal, else the first group."""
    gd = m.groupdict()
    for name in ('v', key):
        if gd.get(name) is not None:
            return gd[name]
    return m.group(1) if m.groups() else None


class LogParser:
    """Turns stamped log lines [(t, text)] into per-step rows; first value seen for a step wins (a tqdm bar's: the
    last)."""

    def __init__(self, patterns: dict | None = None, ignore: str | None = None, count_steps: bool = False,
                 phases: dict | None = None, megatron_timers: bool = False):
        self.patterns = {k: re.compile(v) for k, v in (patterns or {}).items() if k != 'step'}
        self.step_re = re.compile(patterns['step']) if patterns and patterns.get('step') else STEP_RE
        self.ignore = re.compile(ignore) if ignore else None
        self.count_steps = count_steps
        self.phases = {k: re.compile(v) for k, v in (phases or {}).items()}   # name -> regex with group v (+ unit)
        self.megatron_timers = megatron_timers

    def _phases(self, text: str) -> dict:
        """Phase times (ms) a line reports: a Megatron timer line and / or the manifest's `signals.phases` patterns."""
        out = {}
        if self.megatron_timers:
            m = MEGATRON_TIMER.match(text)
            if m and m.group('name') in MEGATRON_UPDATE_TIMERS:     # checkpoint / eval / setup timers are not the update
                out[m.group('name')] = float(m.group('v'))
        for name, rx in self.phases.items():
            m = rx.search(text)
            if m and m.groupdict().get('v') is not None:
                try:
                    v = float(m.group('v'))
                except ValueError:
                    continue
                out[name] = v * PHASE_UNIT_MS.get((m.groupdict().get('unit') or 'ms').lower(), 1.0)
        return out

    def _find(self, key: str, text: str):
        rx = self.patterns.get(key) or BUILTIN.get(key)
        if rx is None:
            return None
        m = rx.search(text)
        raw = _group(m, key) if m else None
        return None if raw is None else _value(raw, key, m.groupdict().get('unit') or '', m.group(0))

    def parse(self, lines) -> list[dict]:
        rows: dict[int, dict] = {}
        current = None
        self.last_update_index = None                    # the position of the last line that carried an update
        for i, (t, text) in enumerate(lines):
            text = plain(text)
            if self.ignore and self.ignore.search(text):
                continue
            m = self.step_re.search(text)
            step = int(_group(m, 'step')) if m else None
            ph = self._phases(text) if (self.phases or self.megatron_timers) else {}
            found = {k: v for k in (*ALIASES, *self.patterns) if (v := self._find(k, text)) is not None}
            mfu = MFU_RE.search(text)
            if mfu:
                value = float(mfu[1])
                value = value / 100 if mfu[2] or value > 1 else value
                if math.isfinite(value) and 0 <= value <= 1:
                    found['mfu'] = value
            dm = DATA_RE.search(text)
            if dm and 'data' not in found:
                found['data'] = dm.group('v')
            bar = step is not None and TQDM_RE.search(text) is not None
            if (bar and step == 0) or (step is not None and not found and not ph and CKPT_RE.search(text)):
                step = None              # a bar's 0/N, 'loaded checkpoint ... at iteration 0': not updates
            if step is None and self.count_steps and 'loss' in found and not SUMMARY_RE.search(text):
                step = (max(rows) + 1) if rows else 1  # (HF Trainer's closing summary has a train_loss: not an update)
            if step is not None:
                row = rows.setdefault(step, {'step': step, 't': t})
                current = step
                self.last_update_index = i
            elif current is not None and (found or ph) and 'loss' not in found:
                row = rows[current]                    # a continuation line (a timer line too) of the current step
            else:
                continue
            for k, v in found.items():
                if bar:
                    row[k] = v
                else:
                    row.setdefault(k, v)
            for k, v in ph.items():
                row.setdefault('phases', {}).setdefault(k, v)
        return [rows[s] for s in sorted(rows)]


# a line in which the job says it runs a slow fallback (`causal_conv1d_fn` is falling back to its reference PyTorch
# implementation ... much slower; a fast path is not available; X is not installed, using Y): the proposer reads them
# the way an engineer reads the log first (2026-10-28: every Qwen3.5-9B run logged one, unseen by the search)
PERF_NOTE_RE = re.compile(r'(?i)falling back|falls back to|fall back to|much slower|slow path|fast path is not available|'
                          r'is not installed|not available,? (?:falling|using|use)')
PERF_NOTES_MAX = 5
# a patch's self-proof lines (F10: what it found and set when it installed, and that its code ran): the evidence
# per site over the whole log (installations, the first `active`), the display lines apart (only kept when present)
PATCH_LINE_RE = re.compile(r'\[infrarsi-patch\]')
PATCH_LINES_MAX = 20                                     # the display lines (feedback), not the evidence
PROOF_RE = re.compile(r'\[infrarsi-patch\]\s*(?P<site>[^:\n]+?)\s*:\s*(?P<rest>.*?)\s*$')
PROOF_FOUND_RE = re.compile(r'found\s+(?P<found>.*?)\s*->\s*set\s+(?P<set>.*)$', re.I)
PROOF_ACTIVE_RE = re.compile(r'active\b', re.I)
PROOF_SITES_MAX, PROOF_INSTALLS_MAX = 32, 8              # sites per run; distinct installations per site
RECOMPILE_RE = re.compile(r'(?i)\brecompil')               # a compiler compiling a graph again during the run
# F13: a line read whole - the site is the text before the `:` that opens the line's word (found / active / deferred),
# so a site with '::' keeps its name; F21: `deferred (<why>)`, a replacement that handed a call back to the original
PROOF_SITE_RE = re.compile(r'\[infrarsi-patch\]\s*(?P<site>[^\n]+?)\s*:\s*(?P<rest>(?:found|active|deferred)\b.*?)\s*$',
                           re.I)
PROOF_DEFERRED_RE = re.compile(r'deferred\b[\s:\-]*(?:\((?P<why>.*)\)|(?P<rest>.*))$', re.I)
PROOF_LINE_MAX = 300                                     # a display line's length (the whole line is read: F13)
ADDRESS = re.compile(r'\bat (0x[0-9a-fA-F]{6,})')       # an object's address in its repr (<function f at 0x7f84...>)
CANON_MAX = 240                                          # F13: a found / set value kept whole up to this; a longer
#                                                          one as its head and a digest of the whole (exact, bounded)


def canonical_pair(found: str, new: str) -> list[str]:
    """F13: an installation's (found, set) as it is compared - across ranks, nodes and runs - and kept: whitespace
    collapsed; each object address in a repr labelled by where it first appears in the pair (`at 0x<1>`, `at 0x<2>`,
    ...), so the same object found and set keeps one label and two objects of one name (a wrapper keeping the name of
    what it replaced) keep two, whatever addresses a process gave them; a value longer than CANON_MAX kept as its head
    and a digest of the whole. Never cut for comparison - a display cuts it (search/proof)."""
    labels: dict = {}

    def label(m):
        return f'at 0x<{labels.setdefault(m.group(1).lower(), len(labels) + 1)}>'
    out = []
    for x in (found, new):
        c = ADDRESS.sub(label, ' '.join(str(x).split()))
        if len(c) > CANON_MAX:
            c = c[:CANON_MAX] + ' ...#' + hashlib.sha256(c.lower().encode('utf-8')).hexdigest()[:16]
        out.append(c)
    return out


def add_proof(proof: dict, text: str, at: float, *, whole: bool = False, layers: dict | None = None) -> None:
    """One `[infrarsi-patch]` line into the per-site evidence: site -> {'installs': [[found, set, at], ...] (distinct;
    `at` the earliest place each was printed), 'active_at': the earliest `active` line, 'deferred': [why, at] of the
    earliest `deferred` line (F21)} - the earliest, not the first read: several nodes' logs come one after the other.
    `at`: the line's place in the log. `whole` (F13): the line is not cut, its site may hold '::', its found / set are
    compared as canonical_pair gives them, and a line longer than a display line leaves its length (`longest`).
    `layers` (F13, the run's own state: site -> {address: layer}): an installation whose found object is what an earlier
    installation at the site set is a layer above it - its set value carries `[layer n]`, so a wrapper stacked on an
    inherited one of the same name stays its own installation (Codex HLS5 r2); the layers are the same in every
    process, whatever addresses each gave its objects."""
    m = (PROOF_SITE_RE.search(text) if whole else None) or PROOF_RE.search(text)
    if not m:
        return
    site, rest = ' '.join(m.group('site').split()), m.group('rest')
    rec = proof.get(site)
    if rec is None:
        if len(proof) >= PROOF_SITES_MAX:
            return
        rec = proof[site] = {'installs': [], 'active_at': None}
    f = PROOF_FOUND_RE.match(rest)
    if f:
        if whole:                                       # F13: whole values, addresses labelled within the pair
            pair = canonical_pair(f.group('found'), f.group('set'))
            if layers is not None and pair[0].lower() != pair[1].lower():   # ... and the layer it was installed over
                held = layers.setdefault(site, {})      # (an identity line - found is set - stays one: Codex HLS5 r3)
                layer = 1 + max([held[a.lower()] for a in ADDRESS.findall(f.group('found')) if a.lower() in held] or [0])
                for a in ADDRESS.findall(f.group('set')):
                    held[a.lower()] = max(held.get(a.lower(), 0), layer)
                if layer > 1:
                    pair[1] = f'{pair[1]} [layer {layer}]'
        else:                                           # (round 2's reading: the display line's pair, 120 characters)
            pair = [' '.join(f.group('found').split())[:120], ' '.join(f.group('set').split())[:120]]
        hit = next((x for x in rec['installs'] if x[:2] == pair), None)
        if hit is not None:                             # the same installation again (another rank or node): its
            hit[2] = min(hit[2], at)                    # earliest place
        elif len(rec['installs']) < PROOF_INSTALLS_MAX:
            rec['installs'].append(pair + [at])
    elif PROOF_ACTIVE_RE.match(rest):
        if rec['active_at'] is None or at < rec['active_at']:
            rec['active_at'] = at
    else:
        d = PROOF_DEFERRED_RE.match(rest)
        if d and ('deferred' not in rec or at < rec['deferred'][1]):
            rec['deferred'] = [' '.join((d.group('why') or d.group('rest') or '').split())[:200], at]
    if whole and len(text) > PROOF_LINE_MAX:
        rec['longest'] = max(int(rec.get('longest') or 0), len(text))


def proof_evidence(lines, *, whole: bool = False) -> dict:
    """The per-site evidence (add_proof) of a list of lines, in their order (`whole`: read as F13 reads them)."""
    out: dict = {}
    layers: dict = {}                                  # (F13: the run's installation layers per site)
    for i, x in enumerate(lines or []):
        add_proof(out, plain(str(x)), float(i), whole=whole, layers=layers if whole else None)
    return out


_NOTE_PREFIX = re.compile(r'^(?:\[rank\d+\]:\s*)?(?:[WIE]\d{4} [\d:.]+ \d+ \S+\] ?)?(?:\[[\w.]+\]\s*)?')


_WARN_PREFIX = re.compile(r'^\S+:\d+: \w*Warning: ')       # Python's `file.py:12: UserWarning: ` before the text


def perf_note(text: str) -> str | None:
    """The note a line carries (rank / logger / warning-location prefixes stripped, 300 characters), or None; the echoed
    `warnings.warn(...)` source line of a Python warning is not a second note."""
    t = plain(text).strip()
    if not PERF_NOTE_RE.search(t) or 'warnings.warn(' in t:
        return None
    return _WARN_PREFIX.sub('', _NOTE_PREFIX.sub('', t))[:300]


def scan_events(lines, ignore: str | None = None) -> dict:
    """Counts of OOM, collective and traceback lines, a few samples, `error`: the first line that names a failure, and
    `fatal`: the first line after which a training job does not go on (FATAL_RE), when there is one. `ignore`: the
    manifest's signals.ignore - a line it matches (a known-harmless traceback at start-up) is not taken as the `error` or
    `fatal` line; the counts still see every line."""
    ign = re.compile(ignore) if ignore else None
    ev = {'oom': 0, 'collective': 0, 'traceback': 0, 'samples': []}
    notes: list[str] = []
    proofs: list[str] = []
    proof: dict = {}
    proof_whole: dict = {}                             # (F13: the same lines read whole)
    layers: dict = {}                                  # (... and the installation layers per site)
    note_last: dict = {}
    recompiles = 0
    for i, (t, text) in enumerate(lines):
        text = plain(text)
        at = (float(t) if isinstance(t, (int, float)) else 0.0) + i * 1e-9   # the line's place (reading time, then order)
        named = not (ign and ign.search(text))
        note = perf_note(text) if named else None
        if note and note not in notes and len(notes) < PERF_NOTES_MAX:
            notes.append(note)
        if note in notes:                              # (F10: a note reported after a patch installed - its latest
            note_last[note] = max(note_last.get(note, at), at)   # report: several nodes' logs come one after the other)
        if PATCH_LINE_RE.search(text):
            whole = text.strip()[text.strip().find('[infrarsi-patch]'):]
            line = whole[:PROOF_LINE_MAX]
            if len(proofs) < PATCH_LINES_MAX and line not in proofs:
                proofs.append(line)
            add_proof(proof, line, at)
            add_proof(proof_whole, whole, at, whole=True, layers=layers)
        if RECOMPILE_RE.search(text):
            recompiles += 1
        m = None if 'error' in ev or not named else ERROR_RE.match(text.strip())
        if m and not any(x in m.group('e') for x in LAUNCHER_ERRORS):
            ev['error'] = m.group('e').strip()[:300]
        if named and 'fatal' not in ev and FATAL_RE.search(text):
            ev['fatal'] = text.strip()[:300]
        for key, rx in (('oom', OOM_RE), ('collective', COLLECTIVE_RE), ('traceback', TRACEBACK_RE)):
            if rx.search(text):
                ev[key] += 1
                if len(ev['samples']) < 5:
                    ev['samples'].append(text.strip()[:300])
    if notes:
        ev['perf_notes'] = notes
    if proofs:
        ev['patch_lines'] = proofs
        ev['patch_proof'] = proof
        if proof_whole != proof:                       # F13: read whole, the lines say more than cut
            ev['patch_proof_whole'] = proof_whole
        ev['perf_note_last'] = {n: note_last[n] for n in notes if n in note_last}
    if recompiles:
        ev['recompiles'] = recompiles
    return ev


def collective_after(lines, last_update_index) -> int | None:
    """How many collective (NCCL) lines of one node's output come after its last update line: the job's shutdown, as
    opposed to a failure in the middle of a run that went on (F-SYS-16). Positions, not clocks: lines of other nodes or
    of a job that logs its updates only through the probe cannot be placed, and None keeps the count rule for them."""
    if last_update_index is None:
        return None
    return sum(1 for i, (_, text) in enumerate(lines) if i > last_update_index and COLLECTIVE_RE.search(plain(text)))


def read_probe(text: str) -> tuple[list[dict], dict]:
    """Probe jsonl -> (rows, meta). Rows: {'step', 't', ...signals}; meta lines: {'meta': {...}}."""
    rows, meta = {}, {}
    for line in (text or '').splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if not isinstance(d, dict):
            continue
        d = {k: ('nan' if math.isnan(v) else 'inf') if isinstance(v, float) and not math.isfinite(v) else v
             for k, v in d.items()}                    # as the log parser has them (a NaN loss is a finding)
        if isinstance(d.get('meta'), dict):
            meta.update(d['meta'])
        elif type(d.get('step')) is int:
            rows.setdefault(d['step'], d)
    return [rows[s] for s in sorted(rows)], meta


def _rebase(rows: list[dict]) -> tuple[list[dict], int]:
    """1-based steps: a log that starts at step 0 is shifted by one."""
    if rows and rows[0]['step'] == 0:
        return [dict(r, step=r['step'] + 1) for r in rows], 0
    return rows, 1


def intervals(rows: list[dict], window: list[int]) -> tuple[list[dict], list[float]]:
    """The stamped rows that bound the window's updates (the last row before it included) and the per-update intervals
    between them on the clock (an interval over several steps is spread over them)."""
    a, b = window
    stamped = [r for r in rows if isinstance(r.get('t'), (int, float))]
    inwin = [r for r in stamped if a <= r['step'] <= b]
    before = [r for r in stamped if r['step'] < a]
    pts = ([before[-1]] if before else []) + inwin
    return pts, [(q['t'] - p['t']) / (q['step'] - p['step']) for p, q in zip(pts, pts[1:])]


def robust_rsd(xs: list[float]) -> float | None:
    """1.4826 x MAD / median of per-update times: the dispersion of the typical update, which a stall that recurs at the
    same update in every run (an epoch boundary, a periodic save) does not inflate the way the sd does; None below two."""
    if len(xs) < 2:
        return None
    med = statistics.median(xs)
    return 1.4826 * statistics.median(abs(x - med) for x in xs) / med if med > 0 else None


def clock_time(rows: list[dict], window: list[int]) -> tuple[float | None, float | None, str | None]:
    """(seconds per update over the window on the stamped clock, RSD of per-update intervals, note)."""
    pts, gaps = intervals(rows, window)
    if len(pts) < 2 or pts[-1]['step'] == pts[0]['step']:
        return None, None, 'fewer than two stamped steps in the scoring window'
    mean = (pts[-1]['t'] - pts[0]['t']) / (pts[-1]['step'] - pts[0]['step'])
    if mean <= 0:
        return None, None, 'non-increasing timestamps'
    # the typical interval below a millisecond is no training update: the lines were buffered and arrived together
    # (in clumps between real gaps, or all at once at the end)
    if len(gaps) >= 3 and statistics.median(gaps) < CLUMP_GAP_S:
        return None, None, 'log lines arrive in bursts (buffered stdout): set PYTHONUNBUFFERED=1 / flush, or use the probe'
    rsd = statistics.stdev(gaps) / mean if len(gaps) >= 2 else None
    return mean, rsd, None


def split_half(rows: list[dict], window: list[int], robust: bool = False) -> float | None:
    """log(time per update of the scoring window's second half / its first half), on the clock, else from the reported
    step times: a sample of the between-run noise that needs no repeated run (on H1 and on the CPU toy its RMS matched
    the differences between repeated runs of one configuration within 10 %). With `robust`, the medians of the
    per-update intervals (or reported times) instead of their means: a stall that recurs at the same update in every
    run lands in one half's mean but not in its median (D2-05: an epoch boundary put 7 of 8 halves at -0.28 while the
    repeats of the configuration agreed within 0.5 %). One run alone cannot tell such a stall from a random one, which
    the window mean does carry, so the engine reads the robust value only once a pair of runs of one configuration is
    in its pool (DF4_r1 #2); the receipt carries both."""
    a, b = window
    m = (a + b) // 2
    if not a <= m < b:
        return None
    t1, t2 = clock_time(rows, [a, m])[0], clock_time(rows, [m + 1, b])[0]
    if t1 and t2:
        if robust:                                   # the clock is usable in both halves: the typical interval of each
            t1, t2 = statistics.median(intervals(rows, [a, m])[1]), statistics.median(intervals(rows, [m + 1, b])[1])
    else:
        h1 = [r['step_time'] for r in rows if a <= r['step'] <= m and type(r.get('step_time')) in (int, float)]
        h2 = [r['step_time'] for r in rows if m < r['step'] <= b and type(r.get('step_time')) in (int, float)]
        mid = statistics.median if robust else (lambda xs: sum(xs) / len(xs))
        t1, t2 = (mid(h1) if h1 else None), (mid(h2) if h2 else None)
    return math.log(t2 / t1) if t1 and t2 and t1 > 0 and t2 > 0 else None


def phase_summary(rows: list[dict], window: list[int], step_time: float | None, top=None) -> dict | None:
    """Where the time of an update goes, over the scoring window: mean ms per phase, its share of the update, and the
    residual = update - the top-level phases (`top`; all phases when not given). None without phase data."""
    a, b = window
    inwin = [r for r in rows if a <= r['step'] <= b]
    have = [r['phases'] for r in inwin if isinstance(r.get('phases'), dict)]
    if not have:
        return None
    names = sorted({k for ph in have for k in ph})
    mean = {k: statistics.fmean([ph[k] for ph in have if type(ph.get(k)) in (int, float)]) for k in names
            if any(type(ph.get(k)) in (int, float) for ph in have)}
    tops = [k for k in (top or mean) if k in mean]
    out = {'unit': 'ms', 'mean_ms': {k: round(v, 3) for k, v in mean.items()}, 'top': tops,
           'coverage': f'{len(have)}/{len(inwin)}'}
    if step_time:
        step_ms = 1000 * step_time
        out.update(step_ms=round(step_ms, 3), share={k: round(v / step_ms, 4) for k, v in mean.items()},
                   residual_ms=round(step_ms - sum(mean[k] for k in tops), 3))
        if out['residual_ms'] < 0:
            out['residual_note'] = 'the phases are per-rank maxima (or overlap): they can add up to more than the update'
    return out


def profile_phases(p: dict | None) -> dict | None:
    """The one-off profile's mean split (project/profhook) in phase_summary's shape: compute, exposed comm and the
    optimizer partition the update; GPU-idle time (host side, input) is the residual. None without one."""
    m = (p or {}).get('mean_ms') or {}
    if not m.get('update'):
        return None
    mean = {k: m[k] for k in ('compute', 'comm', 'optimizer') if type(m.get(k)) in (int, float)}
    return {'unit': 'ms', 'mean_ms': mean, 'top': list(mean), 'coverage': f'{p.get("updates")} profiled updates',
            'step_ms': m['update'], 'share': {k: round(v / m['update'], 4) for k, v in mean.items()},
            'residual_ms': round(float(m.get('idle') or 0.0), 3), 'source': 'profile'}


def phases_brief(ph: dict) -> dict:
    """The phase table as the model sees it: largest first, ms and share, top-level phases marked."""
    share, top = ph.get('share') or {}, set(ph.get('top') or ())
    rows = sorted((ph.get('mean_ms') or {}).items(), key=lambda kv: -kv[1])[:14]
    out = {'update_ms': ph.get('step_ms'), 'residual_ms': ph.get('residual_ms'),
           'phases': [{'name': k, 'ms': round(v, 1), 'share': share.get(k), **({'top': True} if k in top else {})}
                      for k, v in rows]}
    if ph.get('residual_note'):
        out['note'] = ph['residual_note']
    return out


def evidence(rows: list[dict], *, steps: int, window: list[int], exit_code, timed_out: bool, events: dict,
             frozen: dict, meta: dict | None = None, static_workload: dict | None = None, wall_s: float | None = None,
             memory_cap_gib: float | None = None, phase_top=None) -> dict:
    rows, base = _rebase(rows)
    last = rows[-1]['step'] if rows else 0
    clock, rsd, note = clock_time(rows, window)
    rep = [r['step_time'] for r in rows if window[0] <= r['step'] <= window[1] and type(r.get('step_time')) in (int, float)]
    reported = sum(rep) / len(rep) if rep else None
    mfus = [r['mfu'] for r in rows if window[0] <= r['step'] <= window[1]
            and type(r.get('mfu')) in (int, float) and math.isfinite(r['mfu']) and 0 <= r['mfu'] <= 1]
    if rsd is None and len(rep) >= 2 and reported:
        rsd = statistics.stdev(rep) / reported
    mem = [r['mem_gib'] for r in rows if type(r.get('mem_gib')) in (int, float)]
    nan_steps = [r['step'] for r in rows if type(r.get('loss')) is str]
    per_step = [{k: r[k] for k in ('step', 'loss', 'grad_norm', 'lr', 'optim_step', 'tokens', 'samples', 'seq_len', 'data',
                                   'step_time', 'mem_gib', 't', 'phases', 'mfu') if k in r} for r in rows]
    meta = dict(meta or {})
    if meta.get('memory_peak_gib') is not None:
        mem.append(meta['memory_peak_gib'])
    events = dict(events or {})                          # may carry collective_at_end (runner: collective_after)
    return {'steps': steps, 'steps_completed': last, 'step_base': base, 'exit_code': exit_code, 'timed_out': bool(timed_out),
            'window': list(window), 'per_step': per_step,
            **({'mfu': statistics.mean(mfus)} if mfus else {}),
            'step_time': clock if clock is not None else reported, 'step_time_source': 'clock' if clock is not None else
            ('reported' if reported is not None else None), 'clock_step_time': clock, 'clock_note': note,
            'reported_step_time': reported, 'rsd_window': rsd, 'split_half': split_half(rows, window),
            'split_half_robust': split_half(rows, window, robust=True),
            'rsd_robust': robust_rsd(intervals(rows, window)[1] if clock is not None else rep),
            'events': {**events, 'nan_inf_steps': nan_steps},
            'memory_peak_gib': max(mem) if mem else None, 'memory_cap_gib': memory_cap_gib, 'meta': meta,
            'frozen': frozen, 'static_workload': static_workload or {}, 'wall_s': wall_s,
            'logged_every_step': bool(rows) and len(rows) == last,
            'phases': phase_summary(rows, window, clock if clock is not None else reported, phase_top)}
