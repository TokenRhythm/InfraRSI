"""Trials D2-R211 (DiT on two nodes) and D2-R204b (Wan), 2026-10-24: the log had no tokens or samples per update and the
manifest no workload invariant, so same_workload (required) rejected every candidate (bf16 x1.33, workers x1.07) after
the whole search budget was spent. The run stops before the search and says what to add."""
import yaml

from infrarsi.project import Project
from infrarsi.project.flow import run_project
from infrarsi.project.report import markdown

from test_project import toy  # noqa: F401 - the toy project fixture


def test_a_workload_that_cannot_be_checked_stops_before_the_search(toy, tmp_path):
    doc = yaml.safe_load((toy.root / 'infrarsi.yaml').read_text(encoding='utf-8'))
    doc['workload'].pop('invariants')
    (toy.root / 'infrarsi.yaml').write_text(yaml.safe_dump(doc), encoding='utf-8')
    train = (toy.root / 'train.py').read_text(encoding='utf-8')
    assert ' | tokens {tokens}' in train
    (toy.root / 'train.py').write_text(train.replace(' | tokens {tokens}', ''), encoding='utf-8')   # no workload in the log
    seen = []
    res = run_project(Project.load(toy.root), strategy='random', budget={'screens': 3}, out_root=tmp_path / 'out',
                      log=seen.append)
    assert res['outcome'] == 'not_verifiable' and not res.get('candidates')
    assert 'declare workload.invariants' in res['feedback'] and any('stopped before the search' in m for m in seen)
    assert '## Nothing could be verified' in markdown(res)


def test_the_stop_follows_the_verifiers_effective_workload_result():
    from infrarsi.verify import Verifier
    bare = {'per_step': [{'step': s, 'loss': 1.0} for s in (1, 2)], 'steps_completed': 2}
    assert Verifier().workload_unverifiable(bare)
    assert not Verifier(disabled=['same_workload']).workload_unverifiable(bare)          # disabled in the manifest: off
    assert not Verifier(required=('completes', 'no_errors')).workload_unverifiable(bare)  # optional
    seq = {**bare, 'per_step': [{'step': s, 'loss': 1.0, 'seq_len': 4096} for s in (1, 2)]}   # the probe's seq_len
    assert not Verifier().workload_unverifiable(seq)
    assert not Verifier().workload_unverifiable({**bare, 'static_workload': {'global_batch_size': 512}})


def test_a_disabled_workload_check_does_not_stop_the_run(toy, tmp_path):
    doc = yaml.safe_load((toy.root / 'infrarsi.yaml').read_text(encoding='utf-8'))
    doc['workload'].pop('invariants')
    doc['verify'] = {**(doc.get('verify') or {}), 'disabled': ['same_workload']}
    (toy.root / 'infrarsi.yaml').write_text(yaml.safe_dump(doc), encoding='utf-8')
    train = (toy.root / 'train.py').read_text(encoding='utf-8')
    (toy.root / 'train.py').write_text(train.replace(' | tokens {tokens}', ''), encoding='utf-8')
    res = run_project(Project.load(toy.root), strategy='random', budget={'screens': 2}, out_root=tmp_path / 'out',
                      log=lambda m: None)
    assert res['outcome'] != 'not_verifiable' and res.get('candidates')
