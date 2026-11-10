"""Tables R1-R4 (design §5.4): REPRO_ISSUES parsing, the proposed R1 mapping, R2 gates, R4 statistics."""
import json
from pathlib import Path

import pytest
from summary_helpers import World, reports_fixture, write

from infrarsi.ext.summary import records as RC
from infrarsi.ext.summary import repro

REAL = Path(__file__).resolve().parents[4].parent / 'reports' / 'external_bench_20261021'
ISSUES = """| # | Time | Environment | Symptom | Cause | Action | Fidelity impact |
|---|---|---|---|---|---|---|
| 2 | 10-21 03:13 | all | net \\| down | policy | sftp | none |
| 1 | 10-21 03:12 | all | nodes | gone | plan | fewer |
"""


def test_parse_issues_keeps_escaped_pipes_and_sorts():
    rows = repro.parse_issues(ISSUES)
    assert [r['n'] for r in rows] == [1, 2] and rows[1]['symptom'] == 'net \\| down'
    with pytest.raises(ValueError):
        repro.parse_issues(ISSUES + '| 1 | x | y | z | c | a | f |\n')


def test_the_proposed_mapping_uses_only_the_design_vocabularies():
    m = json.loads(repro.MAPPING.read_text(encoding='utf-8'))
    assert repro.check_mapping(m) == []
    assert len(m['lessons']) <= 10
    for n in range(1, 19):                                            # the design's own mapping for #1-#18 (§5.4)
        assert m['issues'][str(n)]['mapping_source'].startswith('design')
    assert m['issues']['10']['category'] == ['upstream bug'] and m['issues']['10']['ids'] == ['D-OC-2']
    assert m['issues']['15']['category'] == ['self-inflicted', 'upstream bug']
    assert m['issues']['26']['stage'] == 'score' and 'protocol gap' in m['issues']['26']['category']
    assert (m['issues']['27']['stage'], m['issues']['27']['category'], m['issues']['27']['fidelity']) == \
        ('run', ['self-inflicted'], 'method')
    e5 = next(x for x in m['extra_rows'] if x['id'] == 'E5')
    assert (e5['stage'], e5['category'], e5['fidelity'], e5['status']) == ('method', ['protocol gap'], 'method', 'open')


@pytest.mark.skipif(not (REAL / 'REPRO_ISSUES.md').is_file(), reason='reports folder not present')
def test_every_real_issue_is_mapped():
    issues = repro.parse_issues((REAL / 'REPRO_ISSUES.md').read_text(encoding='utf-8'))
    m = json.loads(repro.MAPPING.read_text(encoding='utf-8'))
    t = repro.r1(issues, m)
    assert t['unmapped'] == [] and t['mapping_without_issue'] == [] and t['vocabulary_problems'] == []
    assert {r['n'] for r in t['rows'] if r['n']} == {i['n'] for i in issues}


def test_bad_vocabulary_is_reported():
    m = {'vocabularies': json.loads(repro.MAPPING.read_text(encoding='utf-8'))['vocabularies'],
         'issues': {'1': {'env': ['ENV-X'], 'stage': 'nowhere', 'category': ['bad luck'], 'fidelity': 'none', 'status': 'done',
                          'hours': -1}}}
    bad = repro.check_mapping(m)
    assert len(bad) == 5


def test_r4_counts_each_category_and_environment_and_ranks_hours():
    rows = [{'id': '#1', 'n': 1, 'env': ['ENV-A', 'ENV-B'], 'category': ['network', 'protocol gap'], 'hours': 0.5},
            {'id': '#2', 'n': 2, 'env': ['ENV-C'], 'category': ['network'], 'hours': 2.0},
            {'id': '#3', 'n': 3, 'env': ['all'], 'category': ['self-inflicted'], 'hours': None},
            {'id': '#4', 'n': 4, 'env': ['F1'], 'category': ['upstream bug'], 'hours': 1.0}]
    m = {'vocabularies': {'category': ['network', 'protocol gap', 'self-inflicted', 'upstream bug']},
         'lessons': [{'text': str(i)} for i in range(12)]}
    t = repro.r4(rows, m)
    assert t['counts']['network'] == {'ENV-A': 1, 'ENV-B': 1, 'ENV-C': 1, 'F1': 0, 'all': 0}
    assert t['counts']['protocol gap']['ENV-B'] == 1 and t['counts']['self-inflicted']['all'] == 1
    assert [r['id'] for r in t['top3']] == ['#2', '#4', '#1'] and t['hours_total'] == 3.5 and len(t['lessons']) == 10


def test_r2_gates_and_the_env_c_three_lines(tmp_path):
    w = World(tmp_path)
    for att, bpb, steps, dev in (('a7d6d5a710811', 1.0228085860554135, 939, ('D-AR-1', 'D-AR-2', 'D-AR-3', 'D-AR-5')),
                                 ('a21767e9dbd4f', 0.9963646606577251, 936, ('D-AR-1', 'D-AR-2', 'D-AR-3', 'D-AR-5', 'D-AR-7'))):
        write(w.root / 'autoresearch' / 'x1check_20261021' / 'receipts' / f'{att}.json',
              w.c_receipt('x1check_20261021', att, 'c0', arm='C-BASE', bpb=bpb, steps=steps, deviations=dev))
    rd = reports_fixture(w)
    src = json.loads(repro.SOURCES.read_text(encoding='utf-8'))
    results = (rd / 'REPRO_RESULTS.md').read_text(encoding='utf-8')
    store = RC.Store(w.root)
    t = repro.r2(store, w.root, src, results, (rd / 'design' / 'DEVIATION_IDS.md').read_text(encoding='utf-8'), None)
    c = {ln['name']: ln for ln in t['rows']['ENV-C']['lines']}
    assert c['b']['verdict'] == 'not reproduced' and c['b']['abs_delta'] == pytest.approx(0.0249, abs=1e-4)
    assert c['c']['verdict'] == 'reproduced' and c['c']['value']['src'] == ['x1check_20261021:a21767e9dbd4f']
    if results:
        assert c['a']['verdict'] == 'reproduced' and t['problems'] == []
        assert t['rows']['ENV-A']['gate']['v'] == 'reproduced'
        assert t['rows']['ENV-B']['gate']['v'] == 'environment reproduced'
    t2 = repro.r2(store, w.root, src, '', '', None)                 # values that are not in REPRO_RESULTS.md: flagged
    assert t2['rows']['ENV-A']['gate']['v'] == 'pending' and len(t2['problems']) >= 3


def test_r3_lists_every_method_and_marks_f1_results_pending():
    src = json.loads(repro.SOURCES.read_text(encoding='utf-8'))
    from infrarsi.ext.summary import f1
    rows = repro.r3(src, f1.rows({}), {})
    arms = [r.get('arm') for r in rows]
    assert {'F-PROMPTS', 'F-OPTICO', 'F-A0', 'F-RND', 'F-OE'} <= set(arms)
    assert all(r['speedup_vs_UC'].get('status') in ('pending', 'n/a') for r in rows)
