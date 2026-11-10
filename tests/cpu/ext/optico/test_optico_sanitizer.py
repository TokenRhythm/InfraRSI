"""D-OC-16: the parameter-file sanitizer in front of OptiCo's run_monitor.sh (node/optico_node.py + node/sanitize_monitor.sh)."""
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from infrarsi.ext.optico import params as pm
from infrarsi.ext.optico import parse
from infrarsi.ext.optico import workloads as wl
from infrarsi.ext.optico.runner import NODE_DIR

_spec = importlib.util.spec_from_file_location('optico_node_t', NODE_DIR / 'optico_node.py')
node = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(node)
W3 = wl.get('W3')


def test_clean_parameter_files_pass():
    assert node.check_parameters(pm.encode_params(W3.defaults)) == []
    assert node.check_parameters(pm.encode_params(wl.get('W1').knobs(FIRST_PP_NUM_LAYERS=2, LAST_PP_NUM_LAYERS=2))) == []
    # analyzer.py's own format ('KEY:value'), blank lines, a final line without newline, quotes xargs strips
    assert node.check_parameters('MICRO_BATCH_SIZE:4\n\nAC:sel\nFUSED_CROSS_ENTROPY: "true"\nTP:  2 ') == []


@pytest.mark.parametrize('line,why', [
    ('TP: $(rm -rf /)', 'unsafe characters'),
    ('TP: 1;touch /tmp/x', 'unsafe characters'),
    ('TP: `id`', 'unsafe characters'),
    ('AC: a[$(id)]', 'unsafe characters'),                 # bash arithmetic expansion in run_<model>.sh
    ('AC: sel > /etc/passwd', 'unsafe characters'),
    ('TP: 1 2', 'unsafe characters'),                      # word splitting in the eval'd command
    ('TP: "1', 'unsafe characters'),                       # unbalanced quote (xargs fails)
    ("TP: '$HOME'", 'unsafe characters'),
    ('TP: 2\r', 'unsafe characters'),                      # CR stays in the value bash exports
    ('lower_case: 1', 'unsafe characters'),
    ('MICRO BATCH: 1', 'unsafe characters'),
    ('BASH_FUNC_x%%: () { id; }', 'unsafe characters'),
    ('PATH: /tmp', 'protected variable'),
    ('LD_PRELOAD: evil.so', 'protected variable'),
    ('PYTHONPATH: x', 'protected variable'),
    ('BASH_ENV: x', 'protected variable'),
    ('IFS: x', 'protected variable'),
    ('SHELLOPTS: xtrace', 'protected variable'),
    ('OUTPUT_PATH: x', 'protected variable'),
    ('LOG_FILE: x', 'protected variable'),
    ('SCRIPT_PATH: x', 'protected variable'),
    ('PARA_FILE: x', 'protected variable'),
    ('PR: bf16', 'protected variable'),                    # precision override (${PR:-fp16})
    ('MCORE_VERSION: 0.14.0', 'protected variable'),       # gpt3 script selects the Megatron tree with it
    ('MODEL_SIZE: 7B', 'protected variable'),
    ('CUDA_VISIBLE_DEVICES: 0', 'protected variable'),
    ('NCCL_SOCKET_IFNAME: lo', 'protected variable'),
    ('INFRARSI_OPTICO_NODE: x', 'protected variable'),
    ('PATH', 'protected variable'),                        # no colon: run_monitor.sh would export PATH= (empty)
])
def test_unsafe_lines_are_rejected(line, why):
    text = pm.encode_params(W3.defaults) + line + '\n'
    problems = node.check_parameters(text)
    assert problems and problems[0][2] == why and problems[0][0] == 11


def test_knob_names_are_not_denylisted():
    for k in pm.KNOBS + ('EP',):
        assert k not in node.DENY_KEYS and not k.startswith(node.DENY_PREFIXES)


def test_rejection_is_recorded_as_a_failed_run(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    pf = out / 'parameters.txt'
    pf.write_text(pm.encode_params(W3.knobs(MICRO_BATCH_SIZE=2)).replace('TP: 1', 'TP: $(id)'), encoding='utf-8')
    report = out / 'report.csv'
    report.write_text(','.join(parse.DENSE_HEADER) + '\n1,1,1,false,false,none,1,false,None,None,success,233.667,0.7,1,1,1,1,1,NULL\n')
    args = [str(pf), str(out), '2', str(report), str(out / 'monitor_error.txt'), str(out / 'monitor_result.txt'),
            str(tmp_path / 'sanitizer.jsonl')]
    assert node.sanitize(*args) == 3
    err = (out / 'monitor_error.txt').read_text()
    assert err.splitlines()[0] == "InfraRSI sanitizer rejected parameter line 'TP' (unsafe characters)"
    assert (out / '2' / 'monitor_error.txt').read_text() == err
    assert (out / 'monitor_result.txt').read_text().strip() == 'fail' == (out / '2' / 'monitor_result.txt').read_text().strip()
    header, rows = parse.read_report(report.read_text())
    assert header == list(parse.DENSE_HEADER) and len(rows) == 2
    row = rows[1]
    assert row['Result'] == 'fail' and row['TP'] == 'REJECTED' and row['MICRO_BATCH_SIZE'] == '2' and row['Log Director'] == '2'
    assert parse.row_judge(row)['tflops'] is None
    ev = [json.loads(x) for x in (tmp_path / 'sanitizer.jsonl').read_text().splitlines()]
    assert ev[0]['iter'] == '2' and ev[0]['problems'][0] == {'line': 2, 'key': 'TP', 'why': 'unsafe characters'}
    # a clean file launches and leaves nothing behind
    pf.write_text(pm.encode_params(W3.defaults), encoding='utf-8')
    assert node.sanitize(*args) == 0 and len((tmp_path / 'sanitizer.jsonl').read_text().splitlines()) == 1


def test_checker_errors_fail_closed(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    rc = node.sanitize(str(tmp_path / 'missing.txt'), str(out), '3', str(out / 'report.csv'), str(out / 'monitor_error.txt'),
                       str(out / 'monitor_result.txt'), str(tmp_path / 'sanitizer.jsonl'))
    assert rc == 3 and (out / 'monitor_result.txt').read_text().strip() == 'fail'
    assert parse.read_report((out / 'report.csv').read_text())[1][0]['Result'] == 'fail'


def test_copy_check_knows_the_wrapper(tmp_path):
    src, copy, ws = tmp_path / 'src', tmp_path / 'copy', '/data/r/ws'
    for base in (src, copy):
        (base / 'script' / 'monitor').mkdir(parents=True)
    (src / 'script' / 'monitor' / 'run_monitor.sh').write_bytes(b'cd /workspace/code\n')
    (src / 'script' / 'run_circle.sh').write_bytes(b'sysctl -w x=1\nbash monitor/run_monitor.sh\n')
    wrapper = NODE_DIR / 'sanitize_monitor.sh'
    (copy / 'script' / 'monitor' / 'run_monitor.orig.sh').write_bytes(b'cd /data/r/ws/code\n')
    shutil.copyfile(wrapper, copy / 'script' / 'monitor' / 'run_monitor.sh')
    (copy / 'script' / 'run_circle.sh').write_bytes(b'# D-OC-12 not run on shared nodes: sysctl -w x=1\nbash monitor/run_monitor.sh\n')
    assert node.copy_check(str(src), str(copy), ws, loop=True, wrapper=str(wrapper)) == {'checked': 3, 'mismatch': []}
    (copy / 'script' / 'monitor' / 'run_monitor.sh').write_bytes(b'#!/bin/bash\nexec bash run_monitor.orig.sh\n')
    bad = node.copy_check(str(src), str(copy), ws, loop=True, wrapper=str(wrapper))['mismatch']
    assert bad == ['script/monitor/run_monitor.sh (D-OC-16 wrapper)']


def _bash():
    for cand in (r'C:\Program Files\Git\bin\bash.exe', shutil.which('bash')):
        if cand and os.path.isfile(cand) and 'System32' not in cand:
            return cand
    return None


@pytest.mark.skipif(_bash() is None, reason='no POSIX bash on this machine')
def test_wrapper_blocks_the_monitor_and_passes_clean_files(tmp_path):
    """The wrapper itself under bash: an unsafe file never reaches run_monitor.orig.sh; a clean one does."""
    mon = tmp_path / 'script' / 'monitor'
    mon.mkdir(parents=True)
    shutil.copyfile(NODE_DIR / 'sanitize_monitor.sh', mon / 'run_monitor.sh')
    (mon / 'run_monitor.orig.sh').write_text('echo ORIGINAL_MONITOR_RAN > "$OUTPUT_PATH/ran.txt"\n', newline='\n')
    shutil.copyfile(NODE_DIR / 'optico_node.py', tmp_path / 'optico_node.py')
    out = tmp_path / 'out'
    out.mkdir()
    py = shutil.which('python') or 'python'
    shim = tmp_path / 'bin'
    shim.mkdir()
    (shim / 'python3').write_text(f'#!/bin/bash\nexec "{Path(py).as_posix()}" "$@"\n', newline='\n')

    def run(text):
        (out / 'parameters.txt').write_text(text, encoding='utf-8', newline='\n')
        env = {**os.environ, 'OUTPUT_PATH': out.as_posix(), 'ITER_CUR': '4', 'INFRARSI_OPTICO_NODE': (tmp_path / 'optico_node.py').as_posix(),
               'INFRARSI_SANITIZER_LOG': (tmp_path / 'sanitizer.jsonl').as_posix(),
               'PATH': shim.as_posix() + os.pathsep + os.environ.get('PATH', '')}
        return subprocess.run([_bash(), (mon / 'run_monitor.sh').as_posix()], env=env, capture_output=True, text=True, timeout=60)

    r = run(pm.encode_params(W3.defaults).replace('AC: none', 'AC: none;id'))
    assert r.returncode == 0 and 'trial not launched' in r.stdout and not (out / 'ran.txt').exists()
    assert (out / 'monitor_result.txt').read_text().strip() == 'fail'
    r = run(pm.encode_params(W3.defaults))
    assert r.returncode == 0 and (out / 'ran.txt').read_text().strip() == 'ORIGINAL_MONITOR_RAN'
