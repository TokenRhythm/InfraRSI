"""F1 under AMENDMENT_EXT_008 / _009 (D28 / D29): the kept references print the superseded and the current S_F1 sha;
FREEZE.json cites both amendments, flashnext/FREEZE_EXT008.json and the superseding attempts; the store keeps EXT_009
superseding records apart from the receipts and compares an original's copies without their supersede marks; X6's
formal rule on the records enters the D27 accounting; every F1 table carries EXT_009's fixed disclosure sentence and
every screen row its `collection`; a pair withheld under EXT_009 Q4 reads `no comparison (harness)`."""
import json
from pathlib import Path

from summary_helpers import LEDGER_REL, World, f1_arm, f1_uc_ref, reports_fixture, signed, write
from test_summary_cli import run, world

from infrarsi.ext.flashnext import cli as fcli
from infrarsi.ext.summary import cli, f1
from infrarsi.ext.summary import records as RC

OLD, NEW = 'ea48b522' + '0' * 56, 'e04034a3' + '0' * 56


def _orig(att, cid, **extra):
    return signed({'schema': 'ext-receipt-v1', 'env': 'flashnext', 'task_id': 'F1', 'arm': 'F-PROMPTS', 'run_id': 'fp_x',
                   'attempt_id': att, 'candidate_id': cid, 'purpose': 'screen', 'valid': False, 'judge': None,
                   'runner_error': {'kind': 'runner_error', 'type': 'JSONDecodeError', 'message': 'Unterminated string'},
                   'cost': {'gpu_seconds': 7200.0, 'wall_seconds': 225.0, 'accounting_complete': True}, **extra})


def _record(att, cid, value=8.3409):
    return signed({'schema': 'ext-receipt-v1', 'env': 'flashnext', 'task_id': 'F1', 'arm': 'F-PROMPTS', 'run_id': 'fp_x',
                   'attempt_id': att, 'candidate_id': cid, 'purpose': 'screen', 'valid': True,
                   'metric': {'name': 'step_time_s', 'value': value}, 'collection': 'recollected_readonly',
                   'supersedes': att, 'cost': {'gpu_seconds': 7200.0, 'wall_seconds': 225.0, 'accounting_complete': True}})


# ---------------------------------------------------------------- the store
def test_superseding_records_are_kept_apart_and_marked_copies_are_one_receipt(tmp_path):
    d = tmp_path / 'fp_x' / 'receipts'
    write(d / 'att1.json', _orig('att1', 'c1'))
    r = _orig('att1', 'c1')
    r.pop('harness_signature')
    write(d / 'c1.json', RC.signing.sign({**r, 'superseded_by': 'att1'}))      # the re-signed, marked copy
    write(d / 'c1.recollected.json', _record('att1', 'c1'))
    st = RC.Store(tmp_path)
    recs = st.receipts(tmp_path / 'fp_x')
    assert list(recs) == ['att1'] and not recs['att1'].get('superseded_by') and st.problems == []
    assert [x['record']['collection'] for x in st.superseding(tmp_path / 'fp_x')['att1']] == ['recollected_readonly']
    write(d / 'c1b.json', _orig('att1', 'c1', note='a different receipt'))     # still a problem
    st = RC.Store(tmp_path)
    st.receipts(tmp_path / 'fp_x')
    assert any('a second, different receipt' in p for p in st.problems)


def test_an_unadmitted_superseding_record_makes_the_f1_arm_incomplete(tmp_path):
    w = World(tmp_path)
    ex = w.root / 'flashnext'
    uc = f1_uc_ref(ex / 'ucref_x')
    f1_arm(ex / 'fp_x', 'F-PROMPTS', uc)
    write(ex / 'fp_x' / 'receipts' / 'c1.recollected.json', _record('att1', 'c1'))      # no original, no recollection
    acc = f1._accounting(RC.Store(w.root), ex / 'fp_x', ex / 'ucref_x')
    assert acc[0] is False and any('EXT_009' in x for x in acc[1]), acc[1]


# ---------------------------------------------------------------- tables
def _x6_with_harness(calls):
    def main(args):
        calls.append(list(args))
        out = Path(args[args.index('--out') + 1])
        screens = [{'index': 0, 'candidate_id': 'c1', 'attempt_id': 'att1', 'collection': 'recollected_readonly',
                    'class': 'recovered', 'valid': True, 'step_time': 8.3409, 'descriptive_only': False}]
        rows = {a: {'step_time_locked': 8.4, 'speedup_vs_UC': 1.5, 'recovered_fraction': 1.0, 'reached_1_30x': True,
                    'numerics_class': 'E0', 'InvalidAccept': 0, 'AUC_budget': 0.2, 'n_invalid_proposal': 0,
                    'locked_status': 'ok', 't_uc_pair': 12.7, 'drift_uc_pair_over_ref': 1.0,
                    'harness': {'n_harness_lost': 2 if a == 'F-RND' else 0, 'n_recovered': 1 if a == 'F-PROMPTS' else 0,
                                'recovered': [], 'harness_cost_selection': False, 'n_failed_by_class': {},
                                'comparison_withheld': a == 'F-RND'},
                    'screens': screens if a == 'F-PROMPTS' else []} for a in f1.ARMS}
        rows['F-PROMPTS']['recovered_fraction_reading'] = (
            'the selection is the registered S_match point, the F-HT point: recovered_fraction ~ 1 says the method found '
            'the best human recipe, not that it beat it (its screen 8.341 s vs F-HT 8.363 s, -0.3%, under the 3% floor)')
        write(out / 'summary_F1.json', {
            'formal': True, 'nonformal_reasons': [], 't_uc_ref': 12.699, 'rows': rows,
            'labels': {'F-A0 vs F-OPTICO': {'label': 'parity', 'delta_y': 0.01, 'margin': 0.03},
                       'F-A0 vs F-RND': {'label': f1.NO_COMPARISON, 'withheld_by': ['F-RND'], 'computed': {'label': 'parity'},
                                         'reason': 'EXT_009 Q4: an arm with n_harness_lost >= 2 or harness_cost_selection'}},
            'switch_space': {'sha256': NEW}, 'claims': 'x', 'screen_level': 'y'})
        return 0
    return main


def test_f1_tables_carry_the_disclosure_collection_and_withheld_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(fcli, 'main', _x6_with_harness([]))
    w = World(tmp_path)
    ex = w.root / 'flashnext'
    uc = f1_uc_ref(ex / 'ucref_x')
    edit = json.loads((ex / 'ucref_x' / 'uc_ref.json').read_text(encoding='utf-8'))      # the kept reference: old sha
    edit['receipt'] = signed({**{k: v for k, v in uc.items() if k != 'harness_signature'}, 'switch_space_sha256': OLD})
    write(ex / 'ucref_x' / 'uc_ref.json', edit)
    w.ledger['F1'].update(references={'UC-REF': 'ucref_x', 'F-HT': None}, switch_space_sha256=NEW,
                          switch_space_superseded_sha256=OLD)
    for a in f1.ARMS:
        f1_arm(ex / a.lower(), a, edit['receipt'])
        w.ledger['F1']['arms'][a] = a.lower()
    sec = f1.build(RC.Store(w.root), w.root, w.ledger, tmp_path / 'f1out', allow_partial=True)
    text = f1.markdown(sec)
    ref = sec['references'][0]
    assert ref['receipt_switch_space_sha256'] == OLD and ref['current_switch_space_sha256'] == NEW
    assert OLD in text and NEW in text and 'superseded by EXT_008' in text
    tables = sum(1 for line in text.splitlines() if line.startswith('|---'))
    assert text.count(f1.disclosure()) == tables >= 4                      # under every F1 table
    w.ledger['F1']['abandoned'] = [{'run_id': 'fp_old', 'reason': '#36'}]          # the pipeline-fix table too
    text = f1.markdown(f1.build(RC.Store(w.root), w.root, w.ledger, tmp_path / 'f1out2', allow_partial=True))
    assert text.count(f1.disclosure()) == sum(1 for line in text.splitlines() if line.startswith('|---'))
    assert '| collection |' in text.replace('collection | class', 'collection |') and 'recollected_readonly' in text
    assert f1.NO_COMPARISON in text and "withheld by ['F-RND']" in text
    assert 'found the best human recipe, not that it beat it' in text


# ---------------------------------------------------------------- FREEZE
def test_freeze_cites_ext008_ext009_and_the_superseding_attempts(tmp_path, monkeypatch, capsys):
    from test_summary_cli import fake_x6
    monkeypatch.setattr(fcli, 'main', fake_x6([]))
    w = world(tmp_path)
    ex = w.root / 'flashnext'
    write(ex / 'FREEZE_EXT008.json', {'space_sha256': NEW, 'superseded': OLD})
    w.ledger['F1']['freeze_ext008'] = 'reports/x/flashnext/FREEZE_EXT008.json'
    write(ex / 'f_prompts_x' / 'receipts' / 'c1.recollected.json', _record('att1', 'c1'))
    w.save_ledger()
    assert run(w, tmp_path / 'out', '--allow-partial') == 0
    fr = json.loads((tmp_path / 'out' / 'FREEZE.json').read_text(encoding='utf-8'))
    cited = {x['file']: x['sha256'] for x in fr['amendments_cited']}
    assert cited.get('AMENDMENT_EXT_008_FABLE51.md') and cited.get('AMENDMENT_EXT_009_FABLE51.md')
    assert fr['f1_ext008']['file'] == 'flashnext/FREEZE_EXT008.json' and len(fr['f1_ext008']['sha256']) == 64
    sup = fr['ext009_superseding_records']
    assert [(x['arm'], x['attempt_id'], x['collection']) for x in sup] == [('F-PROMPTS', 'att1', 'recollected_readonly')]
    assert len(sup[0]['sha256']) == 64


def test_r1_rows_36_and_37_follow_the_rulings():
    from infrarsi.ext.summary import repro
    m = json.loads(repro.MAPPING.read_text(encoding='utf-8'))['issues']
    assert (m['36']['stage'], m['36']['fidelity'], m['36']['status']) == ('method', 'scope', 'resolved')
    assert (m['37']['stage'], m['37']['category'], m['37']['fidelity'], m['37']['status']) == \
        ('score', ['self-inflicted'], 'none', 'resolved')
