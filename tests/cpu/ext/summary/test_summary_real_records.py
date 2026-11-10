"""The real finished records produce sane partial tables (`--allow-partial`).

Runs the CLI in a subprocess with this PC's receipt key (the summary only verifies, it never signs or creates a key);
skipped when the records, the ledger or the key are not on this machine. Only facts that can no longer change are
asserted (A-ISW's finished run and locked re-score, WINH3's node reference, the ENV-B reference trials, the ENV-C anchor
and calibration, the abandoned runs); labels must be `pending` or a label of the vocabulary.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from infrarsi.contracts import EXECUTION_DIR

REPO = Path(__file__).resolve().parents[4]
LEDGER = REPO.parent / 'reports' / 'external_bench_20261021' / 'results' / 'RUN_LEDGER.json'
RECORDS = EXECUTION_DIR / 'ext_bench'
KEY = Path.home() / '.infrarsi' / 'receipt_hmac.key'
LABELS = {'pending', 'advantage', 'parity', 'disadvantage', 'invalid', 'no_advantage', 'incomplete', 'not run'}

pytestmark = pytest.mark.skipif(not (LEDGER.is_file() and (RECORDS / 'rebench_llmf' / 'aisw_a' / 'result.json').is_file()
                                     and KEY.is_file()), reason='real records / ledger / receipt key not on this machine')


@pytest.fixture(scope='module')
def out(tmp_path_factory):
    d = tmp_path_factory.mktemp('x7real')
    env = {k: v for k, v in os.environ.items() if k != 'INFRARSI_RECEIPT_KEY'}      # verify with the PC key
    env['PYTHONPATH'] = str(REPO)
    p = subprocess.run([sys.executable, '-m', 'infrarsi.ext.summary.cli', '--ledger', str(LEDGER), '--out', str(d),
                        '--allow-partial'], cwd=str(REPO), env=env, capture_output=True, text=True, encoding='utf-8',
                       timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    return d


def load(out, name):
    return json.loads((out / name).read_text(encoding='utf-8'))


def test_files_and_receipt_signatures(out):
    for name in ('ENV_A.md', 'ENV_B.md', 'ENV_C.md', 'F1.md', 'REPRO_SUMMARY.md', 'summary.json', 'FREEZE.json',
                 'g4_calibration.json'):
        assert (out / name).is_file(), name
    fr = load(out, 'FREEZE.json')
    assert fr['receipts']['total'] > 50 and fr['receipts']['failed'] == 0 and fr['receipts']['no_key'] == 0
    # the native arm's transcript (outputs/, calls/, its events) is never read; A-ISW / A-RND's search events are
    # (kinds and ids only: the D27 call-to-row boundary)
    paths = [f['path'].replace('\\', '/') for f in fr['files']]
    assert not any('/outputs/' in p or '/calls/' in p for p in paths)
    assert not any(p.endswith('/events.jsonl') and '/rebench_llmf/anat_' in p for p in paths)


def test_env_a_a_isw_row(out):
    s = load(out, 'summary.json')
    a = s['ENV-A']['rows']['A-ISW']
    assert a['status'] == 'finished' and a['node'] == 'WINH3'
    assert a['search']['t_best_search']['v'] == pytest.approx(108.536099)
    assert a['search']['t_best_search']['src'] == ['aisw_a:ab92eeec71df4']
    assert a['search']['t_first']['v'] == pytest.approx(255.861439) and a['search']['first_cold']['v'] is True
    assert a['search']['n_scored']['v'] == 3 and a['search']['early_finish']['v'] is True
    lk = a['locked']
    assert lk['t_locked']['v'] == pytest.approx(109.031441) and lk['judge_valid']['v'] is True
    assert lk['g4_ratio']['v'] == pytest.approx(1.0004283812269272)
    assert lk['locked_valid_predeclared']['v'] is False                      # G4 clause (i) at 5.3 (EXT_005 disclosure)
    assert lk['g4_limit_dprime']['v'] == pytest.approx(6.17)
    assert lk['locked_valid_amended'].get('status') == 'pending' or lk['locked_valid_amended']['v'] in (True, False)
    assert a['y_ref_node']['v'] == pytest.approx(216.629206)                  # N_WINH3 (AMENDMENT_EXT_004 (b))
    assert a['AUC_budget']['v'] == pytest.approx(0.6668, abs=5e-4)
    assert s['ENV-A']['node_refs']['WINH3']['src'] == ['nref_winh3_a:adc6d58bceeb2']
    for k, res in s['ENV-A']['labels'].items():
        assert res['label'] in LABELS or res['label'].startswith('no comparison'), (k, res)


def test_env_b_references_and_margins(out):
    s = load(out, 'summary.json')
    W = s['ENV-B']['workloads']
    assert W['W1']['arms']['B-DEF']['T']['v'] == pytest.approx(344.533)
    assert W['W1']['arms']['B-ED']['T']['v'] == pytest.approx(506.5, abs=0.1)
    assert W['W2']['arms']['B-ED']['T']['v'] == pytest.approx(570.5, abs=0.1)
    assert W['W3']['arms']['B-DEF']['T']['v'] == pytest.approx(223.2, abs=0.1)
    for w in ('W1', 'W2', 'W3'):
        m = W[w]['margin_record']
        assert m['estimator'] in ('run_to_run', 'pooled_fallback') and 0.03 <= m['margin'] < 0.2
        assert not any(x['run_id'] in ('bnat_w1_a', 'bis_w1_a', 'bnat_w2_a') for x in m['S_w'])   # abandoned (#29)
    assert s['ENV-B']['env_label']['label'] in LABELS or s['ENV-B']['env_label']['label'].startswith('no advantage')


def test_env_c_anchor_calibration_and_abandoned_runs(out):
    s = load(out, 'summary.json')
    c = s['ENV-C']
    assert c['anchor']['val_bpb']['v'] == pytest.approx(0.9963646606577251) and c['anchor']['steps']['v'] == 936
    assert c['calibration']['gate']['v'] is True and c['calibration']['c_base_steps']['v'] == 933
    ab = {x['run_id'] for x in s['abandoned']}
    assert {'anat_a', 'bnat_w1_a', 'bis_w1_a', 'bnat_w2_a', 'envc_calib_a', 'envc_cis_c', 'envc_crnd_c'} <= ab
    rows = [r['run_id'] for r in c['arms'].values()]
    assert not any(r and r.endswith('_c') for r in rows)                      # abandoned runs never on result rows


def test_repro_tables(out):
    s = load(out, 'summary.json')
    r1 = s['repro']['R1']
    assert r1['unmapped'] == [] and r1['vocabulary_problems'] == []
    assert len([r for r in r1['rows'] if r['n']]) == s['repro']['n_issues'] >= 30
    r2 = s['repro']['R2']['rows']
    assert r2['ENV-A']['gate']['v'] == 'reproduced'
    assert [ln['verdict'] for ln in r2['ENV-C']['lines']] == ['reproduced', 'not reproduced', 'reproduced']
    text = (out / 'REPRO_SUMMARY.md').read_text(encoding='utf-8')
    assert '## R1' in text and '## R4' in text and 'reproduced with D-AR-7' in text


def test_f1_ext008_ext009_records(out):
    # stable since 21:32: the kept references carry the superseded S_F1 sha; c1-c3 of f1_prompts_20261021b have their
    # EXT_009 superseding records; FREEZE cites them and FREEZE_EXT008.json; F1.md carries the disclosure
    s, fr = load(out, 'summary.json'), load(out, 'FREEZE.json')
    refs = {x['kind']: x for x in s['F1']['references']}
    assert refs['UC-REF']['receipt_switch_space_sha256'].startswith('ea48b5226ab5')
    assert refs['UC-REF']['current_switch_space_sha256'].startswith('e04034a364f3')
    assert fr['f1_ext008']['file'] == 'flashnext/FREEZE_EXT008.json'
    sup = {x['attempt_id'] for x in fr['ext009_superseding_records'] if x['run_id'] == 'f1_prompts_20261021b'}
    assert sup >= {'abbd3dc252602', 'a417d90dd3ac1', 'a0fb0d485687c'}
    assert not any('a second, different receipt' in p for p in s['problems'])
    assert 'completed on the nodes but were lost by our collection transport' in (out / 'F1.md').read_text(encoding='utf-8')
