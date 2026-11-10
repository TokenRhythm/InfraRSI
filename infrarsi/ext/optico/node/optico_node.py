"""Node-side helper of ext/optico (stdlib only, Python >= 3.8; uploaded next to trial.sh / loop.sh in each run dir).

  python3 optico_node.py hash <src> <out.json>                     sha256 of every file under the OptiCo base tree
  python3 optico_node.py copycheck <src> <copy> <ws> <out.json> <trial|loop> <wrapper>
        candidate copy of script/ == base with /workspace -> ws (loop: plus the sysctl lines of run_circle.sh commented),
        with the D-OC-16 wrapper installed as monitor/run_monitor.sh and OptiCo's own script as run_monitor.orig.sh
  python3 optico_node.py killafter <seconds> <token>              backstop: kill the token's processes after <seconds>
  python3 optico_node.py sanitize <para_file> <output_path> <iter_cur> <report_file> <error_log> <result_file> <events>
        D-OC-16: exit 0 when every line of the parameter file is safe to export; else (exit 3) record the trial as a
        failed run the way run_monitor.sh does (report.csv row, error log for loc_error, result 'fail') and log an event
  python3 optico_node.py watch <run_dir> <out_path> <pid> <max_rows> <max_wall_s> <token>
        B-NAT watcher: every 2 s records first sight of each iteration directory, every report.csv row count and
        every parameters.txt change (stamps.jsonl); kills the loop (processes carrying <token>, never itself) when the
        report reaches max_rows rows or max_wall_s passes; exits when <pid> is gone.
"""
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import sys
import time

TOKEN_VAR = 'INFRARSI_TOKEN'


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def hash_tree(src):
    files = {}
    for base, dirs, names in os.walk(src):
        dirs.sort()
        for n in sorted(names):
            p = os.path.join(base, n)
            if os.path.islink(p) or not os.path.isfile(p):
                continue
            files[os.path.relpath(p, src).replace(os.sep, '/')] = _sha(p)
    tree = hashlib.sha256(json.dumps(sorted(files.items())).encode('utf-8')).hexdigest()
    return {'files': {k: v for k, v in files.items() if k.split('/')[0] in ('script', 'prompt')},
            'n_files': len(files), 'tree_digest': tree, 'utc': time.time()}


SYSCTL = (re.compile(rb'^sysctl -w', re.M), b'# D-OC-12 not run on shared nodes: sysctl -w')     # loop.sh's sed


MONITOR = 'script/monitor/run_monitor.sh'
MONITOR_ORIG = 'script/monitor/run_monitor.orig.sh'


def copy_check(src, copy, ws, loop=False, wrapper=None):
    """Every file of the base's script/ must be present in the copy with the bytes of the base after the declared
    edits: /workspace -> ws everywhere, for the B-NAT loop the `sysctl -w` lines of run_circle.sh commented out, and
    OptiCo's run_monitor.sh moved to run_monitor.orig.sh behind the D-OC-16 wrapper (whose bytes must equal `wrapper`).
    run_monitor.sh rewrites monitor_gpu.sh with identical content."""
    bad, n = [], 0
    root = os.path.join(src, 'script')
    for base, dirs, names in os.walk(root):
        for name in names:
            p = os.path.join(base, name)
            rel = os.path.relpath(p, src).replace(os.sep, '/')
            with open(p, 'rb') as f:
                data = f.read().replace(b'/workspace', ws.encode('utf-8'))
            if loop and rel == 'script/run_circle.sh':
                data = SYSCTL[0].sub(SYSCTL[1], data)
            want = hashlib.sha256(data).hexdigest()
            q = os.path.join(copy, MONITOR_ORIG if (wrapper and rel == MONITOR) else rel)
            got = _sha(q) if os.path.isfile(q) else None
            n += 1
            if got != want:
                bad.append(rel)
    if wrapper:
        q = os.path.join(copy, MONITOR)
        n += 1
        if not os.path.isfile(q) or _sha(q) != _sha(wrapper):
            bad.append(MONITOR + ' (D-OC-16 wrapper)')
    return {'checked': n, 'mismatch': bad}


# ---------------------------------------------------------------- D-OC-16 parameter-file sanitizer
KEY_RE = re.compile(r'^[A-Z][A-Z0-9_]*$')
VALUE_RE = re.compile(r'^[A-Za-z0-9_.,+-]*$')
# Never exported from a parameter file: shell, loader, interpreter and path variables, and every non-knob variable
# OptiCo's scripts read (`${X:-...}` overrides, file paths, fixed settings), so a line cannot redirect, re-point or
# re-configure anything (the knobs MICRO_BATCH_SIZE, TP, PP, OVERLAP_*, AC, AC_LAYERS, FUSED_CROSS_ENTROPY,
# FIRST/LAST_PP_NUM_LAYERS, EP stay allowed).
DENY_KEYS = frozenset("""
PATH HOME SHELL ENV BASH_ENV IFS PS1 PS2 PS3 PS4 PROMPT_COMMAND CDPATH GLOBIGNORE SHELLOPTS BASHOPTS HISTFILE HISTCONTROL
FUNCNEST TMPDIR TMP TEMP USER LOGNAME LANG LANGUAGE TERM PWD OLDPWD MAIL OPTIND OPTARG SUDO_USER EDITOR PAGER
OUTPUT_DIR OUTPUT_PATH REPORT_FILE PARA_FILE FIXED_PARA_FILE RESULT_FILE MONITOR_RESULT_FILE ERROR_LOG_FILE LOG_FILE
MEMORY_LOG_FILE CURRENT_LOG_DIR MEGATRON_LOG_DIR CURRENT_ERROR_LOG_FILE CURRENT_PARA_FILE SCRIPT_PATH MONITOR_GPU_SCRIPT
PROMPT_HOME_PATH MODEL_TYPE MODEL_SIZE MODEL_PATH ITER_CUR RANK WORLD_SIZE NNODES NODE_RANK GPUS_PER_NODE MASTER_ADDR
MASTER_PORT DISTRIBUTED_ARGS PR MCORE_VERSION NUMBER_OF_CYCLES KEY BASE_URL GLOBAL_BATCH_SIZE SEQ_LENGTH TRAIN_ITERS
LR_WARMUP_ITERS LR_DECAY_ITERS LR MIN_LR SP DO FL TE CP PAD_LEN DATASET_PATH PRETRAIN_CHECKPOINT_PATH TOKENIZER_MODEL_PATH
PAI_MEGATRON_PATH MEGATRON_PATH PRETRAIN_SCRIPT_PATH OUTPUT_BASEPATH SAVE_INTERVAL NAME
""".split())
DENY_PREFIXES = ('LD_', 'PYTHON', 'BASH', 'NCCL_', 'CUDA_', 'NVTE_', 'NVIDIA_', 'TORCH', 'PYTORCH_', 'TRITON_', 'OMP_',
                 'MKL_', 'HF_', 'TRANSFORMERS_', 'INFRARSI_', 'GIT_', 'SSH_', 'XDG_', 'PIP_', 'CONDA', 'VIRTUAL_ENV',
                 'SLURM_', 'PMI_', 'OPAL_', 'UCX_', 'MALLOC_', 'GLIBC_', 'LC_', 'GCONV_', 'NLSPATH', 'HOSTALIASES',
                 'RESOLV_', 'LOCALDOMAIN')
DENSE_HEADER = ('MICRO_BATCH_SIZE', 'TP', 'PP', 'OVERLAP_GRAD_REDUCE', 'OVERLAP_PARAM_GATHER', 'AC', 'AC_LAYERS',
                'FUSED_CROSS_ENTROPY', 'FIRST_PP_NUM_LAYERS', 'LAST_PP_NUM_LAYERS', 'Result', 'TFLOPS', 'MFU', 'MemAve',
                'MemMax', 'GpuUtilAve', 'GpuUtilMax', 'Log Director', 'Reasons for failure')


def _xargs(text):
    """`echo "$x" | xargs` of run_monitor.sh: shell-like word splitting with quotes and backslashes, words re-joined by
    single spaces; None when xargs would fail (unbalanced quote)."""
    lex = shlex.shlex(text, posix=True)
    lex.whitespace, lex.whitespace_split, lex.commenters = ' \t\n', True, ''   # GNU xargs: blanks are space/tab only
    try:
        return ' '.join(lex)
    except ValueError:
        return None


def check_parameters(text):
    """[(line number, key as shown, why)] for every line run_monitor.sh would export unsafely; [] means clean.
    Lines are read as `IFS=':' read -r key value` does (split at the first ':'), then trimmed as xargs does."""
    problems = []
    for n, raw in enumerate(text.split('\n'), 1):
        key_raw, _, value_raw = raw.partition(':')
        key, value = _xargs(key_raw), _xargs(value_raw)
        if key == '':
            continue                                     # the monitor skips lines whose key is empty
        shown = re.sub(r'[^A-Za-z0-9_]', '?', key if key is not None else key_raw.strip())[:48] or '?'
        if key is None or not KEY_RE.match(key):
            problems.append((n, shown, 'unsafe characters'))
        elif key in DENY_KEYS or key.startswith(DENY_PREFIXES):
            problems.append((n, shown, 'protected variable'))
        if value is None or not VALUE_RE.match(value):
            problems.append((n, shown, 'unsafe characters'))
    return problems


def _safe_cell(v):
    v = (v or '').strip()
    if not v:
        return 'NULL'
    return v if VALUE_RE.match(v) and ',' not in v else 'REJECTED'


def reject_trial(para_file, output_path, iter_cur, report_file, error_log, result_file, events, problems, text):
    """Record the trial as a failed run exactly where run_monitor.sh would: <output>/<iter>/monitor_error.txt (copied to
    the error log loc_error reads), a report.csv row with Result=fail, and 'fail' in both result files."""
    cur = os.path.join(output_path, iter_cur)
    os.makedirs(cur, exist_ok=True)
    lines = ["InfraRSI sanitizer rejected parameter line '%s' (%s)" % (key, why) for _, key, why in problems]
    lines.append('Error: the trial was not launched (InfraRSI D-OC-16): every parameter line must be KEY: value with '
                 'KEY in A-Z0-9_ (not a shell, loader, path or OptiCo-internal variable) and a value of A-Za-z0-9_.,+- only')
    with open(os.path.join(cur, 'monitor_error.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    shutil.copyfile(os.path.join(cur, 'monitor_error.txt'), error_log)
    values = {}
    for raw in text.split('\n'):
        k, _, v = raw.partition(':')
        values.setdefault(k.strip(), v)
    cells = [_safe_cell(values.get(k)) for k in DENSE_HEADER[:10]]
    cells += ['fail'] + ['NULL'] * 6 + [re.sub(r'[^A-Za-z0-9_.-]', '_', iter_cur), 'NULL']
    new = not os.path.isfile(report_file)
    with open(report_file, 'a', encoding='utf-8') as f:
        if new:
            f.write(','.join(DENSE_HEADER) + '\n')
        f.write(','.join(cells) + '\n')
    for path in (result_file, os.path.join(cur, 'monitor_result.txt')):
        with open(path, 'w', encoding='utf-8') as f:
            f.write('fail\n')
    with open(events, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'t': time.time(), 'iter': iter_cur, 'para_file': para_file,
                            'problems': [{'line': n, 'key': k, 'why': w} for n, k, w in problems]}) + '\n')


def sanitize(para_file, output_path, iter_cur, report_file, error_log, result_file, events):
    """0 = clean (launch); 3 = rejected and recorded. Any internal error is treated as a rejection (fail closed)."""
    try:
        with open(para_file, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        problems = check_parameters(text)
    except Exception as e:                       # noqa: BLE001 - fail closed
        text, problems = '', [(0, '?', 'checker error %r' % (e,))]
    if not problems:
        return 0
    reject_trial(para_file, output_path, iter_cur, report_file, error_log, result_file, events, problems, text)
    return 3


def _rows(report):
    try:
        with open(report, 'r', encoding='utf-8', errors='replace') as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip() and not ln.startswith('#')]
    except OSError:
        return 0
    return max(0, len(lines) - 1)


def _token_pids(token):
    out = []
    for name in os.listdir('/proc'):
        if not name.isdigit() or int(name) == os.getpid():
            continue
        try:
            with open('/proc/%s/environ' % name, 'rb') as f:
                env = f.read().split(b'\0')
        except OSError:
            continue
        if (TOKEN_VAR + '=' + token).encode() in env:
            out.append(int(name))
    return out


def _kill(token):
    pids = _token_pids(token)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return pids


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def watch(run_dir, out_path, pid, max_rows, max_wall_s, token):
    stamps = os.path.join(run_dir, 'stamps.jsonl')
    report = os.path.join(out_path, 'report.csv')
    para = os.path.join(out_path, 'parameters.txt')
    t0 = time.time()
    seen, rows, last_para = set(), -1, None

    def emit(**kw):
        kw['t'] = round(time.time(), 3)
        with open(stamps, 'a', encoding='utf-8') as f:
            f.write(json.dumps(kw) + '\n')

    emit(kind='watch_start', pid=pid, max_rows=max_rows, max_wall_s=max_wall_s)
    while _alive(pid):
        try:
            for name in sorted(os.listdir(out_path)):
                if name not in seen and os.path.isdir(os.path.join(out_path, name)) and name != 'cache':
                    seen.add(name)
                    emit(kind='iter_dir', name=name, rows=max(rows, 0))
        except OSError:
            pass
        n = _rows(report)
        if n != rows:
            rows = n
            emit(kind='rows', rows=n)
        try:
            with open(para, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
        except OSError:
            text = None
        if text is not None and text != last_para:
            last_para = text
            emit(kind='parameters', text=text[:4000])
        reason = None
        if max_rows > 0 and rows >= max_rows:
            reason = 'max_trials'
        elif max_wall_s > 0 and time.time() - t0 > max_wall_s:
            reason = 'max_wall'
        if reason:
            emit(kind='stop', reason=reason, rows=rows, killed=_kill(token))
            break
        time.sleep(2)
    emit(kind='watch_end', rows=max(rows, 0))


def main():
    cmd = sys.argv[1]
    if cmd == 'hash':
        res = hash_tree(sys.argv[2])
        out = sys.argv[3]
    elif cmd == 'copycheck':
        res = copy_check(sys.argv[2], sys.argv[3], sys.argv[4], loop=len(sys.argv) > 6 and sys.argv[6] == 'loop',
                         wrapper=sys.argv[7] if len(sys.argv) > 7 else None)
        out = sys.argv[5]
    elif cmd == 'sanitize':
        sys.exit(sanitize(*sys.argv[2:9]))
    elif cmd == 'killafter':
        time.sleep(float(sys.argv[2]))
        print(json.dumps({'t': time.time(), 'killed': _kill(sys.argv[3])}), flush=True)
        return
    elif cmd == 'watch':
        watch(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), sys.argv[7])
        return
    else:
        raise SystemExit('unknown command %r' % cmd)
    tmp = out + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(res, f)
    os.replace(tmp, out)


if __name__ == '__main__':
    main()
