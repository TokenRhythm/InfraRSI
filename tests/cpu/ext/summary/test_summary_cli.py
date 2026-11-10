"""The W5 CLI end to end on a synthetic world: formal only when complete, --allow-partial, FREEZE.json, the F1 seam."""
import json
from pathlib import Path

import pytest
from summary_helpers import LEDGER_REL, World, f1_arm, f1_uc_ref, reports_fixture, write

from infrarsi.ext.flashnext import cli as fcli
from infrarsi.ext.summary import cli
from infrarsi.ext.summary import f1 as F1

G4_FINAL = {'final': True, 'refprime_pairwise_norms': [{'pair': list(p), 'norm_diff': 4.9, 'source': 'test'} for p in
                                                         (('aisw_x', 'anat_x'), ('aisw_x', 'arnd_x'), ('anat_x', 'arnd_x'))]}


def fake_x6(calls):
    def main(args):
        calls.append(list(args))
        out = Path(args[args.index('--out') + 1])
        rows = {a: {'step_time_locked': 9.0, 'speedup_vs_UC': 1.38, 'recovered_fraction': 0.4, 'reached_1_30x': True,
                    'numerics_class': 'E0', 'InvalidAccept': 0, 'AUC_budget': 0.2, 'n_invalid_proposal': 0,
                    'locked_status': 'ok', 't_uc_pair': 12.4, 'drift_uc_pair_over_ref': 1.0,
                    'costs': {'search': {'gpu_hours': 100.0, 'llm_usd': 20.0, 'wall_seconds': 36000.0},
                              'locked': {'gpu_seconds': 3600.0}}}
                for a in F1.ARMS}
        write(out / 'summary_F1.json', {'formal': True, 'nonformal_reasons': [], 't_uc_ref': 12.4, 'rows': rows,
                                        'labels': {'F-A0 vs F-OPTICO': {'label': 'parity', 'delta_y': 0.01, 'margin': 0.03}},
                                        'claims': 'switch-space search only', 'screen_level': 'screen-level'})
        return 0
    return main


def world(tmp_path, *, f1=True):
    w = World(tmp_path)
    w.env_a_complete()
    for wk in ('W1', 'W3', 'W2'):
        w.env_b_workload(wk)
    w.env_c_complete()
    w.c_r2_receipts()
    reports_fixture(w)
    if f1:
        ex = w.root / 'flashnext'
        uc = f1_uc_ref(ex / 'ucref_x')
        write(ex / 'ht_x' / 'ht.json', {'schema': 'ext-receipt-v1', 'attempt_id': 'ht'})
        w.ledger['F1']['references'] = {'UC-REF': 'ucref_x', 'F-HT': 'ht_x'}
        for a in F1.ARMS:
            rid = a.lower().replace('-', '_') + '_x'
            f1_arm(ex / rid, a, uc)
            w.ledger['F1']['arms'][a] = rid
    w.save_ledger()
    write(tmp_path / 'g4.json', G4_FINAL)
    return w


def run(w, out, *extra):
    return cli.main(['--ledger', str(w.tmp / LEDGER_REL), '--out', str(out), '--records', str(w.root),
                     '--reports', str(w.reports), '--g4-inputs', str(w.tmp / 'g4.json'), *extra])


def test_a_complete_world_is_formal_and_frozen(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(fcli, 'main', fake_x6(calls))
    w = world(tmp_path)
    out = tmp_path / 'out'
    assert run(w, out) == 0, capsys.readouterr().out
    s = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    assert s['formal'] is True and s['pending'] == [] and s['problems'] == []
    assert s['ENV-A']['labels']['A-ISW vs A-NAT']['label'] == 'advantage'
    assert s['ENV-B']['env_label']['label'] == 'advantage'
    assert s['ENV-C']['labels']['C-IS vs C-NAT-infra']['label'] == 'advantage'
    assert s['F1']['labels']['F-A0 vs F-OPTICO']['label'] == 'parity'
    for name in ('ENV_A.md', 'ENV_B.md', 'ENV_C.md', 'F1.md', 'REPRO_SUMMARY.md', 'summary.json', 'FREEZE.json',
                 'g4_calibration.json', 'margin_W1.json', 'margin_W2.json', 'margin_W3.json', 'summary_F1.json'):
        assert (out / name).is_file(), name
    fr = json.loads((out / 'FREEZE.json').read_text(encoding='utf-8'))
    receipts = [f for f in fr['files'] if f['role'] == 'receipt']
    assert receipts and all(f['signature'] == 'verified' and len(f['sha256']) == 64 for f in receipts)
    assert fr['receipts']['verified'] == len(receipts) and fr['receipts']['failed'] == 0
    assert {o['name'] for o in fr['outputs']} >= {'summary.json', 'ENV_A.md', 'REPRO_SUMMARY.md'}
    assert any(t['path'].endswith('cli.py') for t in fr['tool'])
    assert calls and '--allow-nonformal' not in calls[0]
    g4 = json.loads((out / 'g4_calibration.json').read_text(encoding='utf-8'))
    assert g4['estimator'] == 'computed' and g4['final'] is True
    text = (out / 'F1.md').read_text(encoding='utf-8')
    assert 'Formal: yes' in text and '1.3800' in text
    # every number of the ENV-A arm rows carries a source
    isw = s['ENV-A']['rows']['A-ISW']
    for k in ('AUC_budget', 'y_B', 't_B'):
        assert isw[k]['src'], k


def test_partial_world_is_refused_unless_allow_partial(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fcli, 'main', fake_x6([]))
    w = world(tmp_path, f1=False)
    w.ledger['ENV-A']['block'] = {'run_id': None}
    w.save_ledger()
    out = tmp_path / 'out'
    assert run(w, out) == 4 and not out.exists()
    printed = capsys.readouterr().out
    assert 'block' in printed and 'F1' in printed
    assert run(w, out, '--allow-partial') == 0
    s = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    assert s['formal'] is False and any('block' in p for p in s['pending'])
    assert s['ENV-A']['labels']['A-ISW vs A-NAT']['label'] == 'pending'
    assert '<!-- generated' in (out / 'ENV_A.md').read_text(encoding='utf-8')
    assert 'PARTIAL' in (out / 'ENV_A.md').read_text(encoding='utf-8')


def test_a_tampered_receipt_makes_the_summary_non_formal(tmp_path, monkeypatch):
    monkeypatch.setattr(fcli, 'main', fake_x6([]))
    w = world(tmp_path)
    p = w.root / 'autoresearch' / 'lk_cis_x' / 'runner' / 'receipts' / 'lk_cis_x-lock-cand.json'
    r = json.loads(p.read_text(encoding='utf-8'))
    r['metric']['value'] = 0.95
    write(p, r)
    out = tmp_path / 'out'
    assert run(w, out) == 4
    assert run(w, out, '--allow-partial') == 0
    s = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    assert any('signature does not verify' in p for p in s['problems'])
    fr = json.loads((out / 'FREEZE.json').read_text(encoding='utf-8'))
    assert fr['receipts']['failed'] == 1


def test_f1_passes_allow_nonformal_only_under_allow_partial(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fcli, 'main', fake_x6(calls))
    w = world(tmp_path)
    w.ledger['F1']['arms']['F-RND'] = None
    w.save_ledger()
    out = tmp_path / 'out'
    assert run(w, out, '--allow-partial') == 0
    assert '--allow-nonformal' in calls[-1] and 'F-RND=' not in ' '.join(calls[-1])
    s = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    assert any('F1 F-RND' in p for p in s['pending'])


def test_the_f1_seam_reports_a_refusal(tmp_path):
    class Refuse:
        @staticmethod
        def main(args):
            print('summary refused: arms missing from the window')
            return 4
    w = world(tmp_path)
    from infrarsi.ext.summary import records as RC
    sec = F1.build(RC.Store(w.root), w.root, w.ledger, tmp_path / 'f1', allow_partial=False, runner=Refuse)
    assert sec['summary'] is None and any('exit 4' in p and 'arms missing' in p for p in sec['pending'])
    assert '**Pending.**' in F1.markdown(sec)


def test_no_ledger_is_a_usage_error(tmp_path):
    assert cli.main(['--ledger', str(tmp_path / 'nope.json'), '--out', str(tmp_path / 'o')]) == 2


@pytest.mark.parametrize('obj,want', [({'a': float('inf'), '_x': 1, 'b': (1, 2)}, {'a': None, 'b': [1, 2]})])
def test_json_cleaning(obj, want):
    assert json.loads(cli._dump(obj)) == want
