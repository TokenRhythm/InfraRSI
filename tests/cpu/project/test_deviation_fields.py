"""U-8-2: the promotion rows name their loss deviations as repeats of one configuration (`reference_repeat_loss_deviation`:
the reference's re-run against the reference; `candidate_repeat_loss_deviation`: the candidate's fresh run against its
screen), not as deviations from the reference; U-1-21: the report's candidate table shows each candidate's deviation from
the reference ("loss vs ref")."""
import json
import sys
from pathlib import Path

import pytest

from infrarsi.project import Project
from infrarsi.project.flow import promote
from infrarsi.project.report import markdown
from infrarsi.verify import Verifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'verify'))   # test_verifier's receipt helper
from test_verifier import receipt  # noqa: E402

LOSS = lambda s: 2.0 - 0.01 * s                     # noqa: E731 - the helper's loss curve


def test_the_promotion_row_names_its_repeats_as_repeats(tmp_path):
    (tmp_path / 'train.py').write_text('print(1)\n', encoding='utf-8')
    p = Project({'name': 'n', 'launch': {'command': '{python} train.py {args}'}, 'knobs': {},
                 'workload': {'steps': 10, 'window': [3, 10]}}, tmp_path / 'infrarsi.yaml')
    (tmp_path / 'rc').mkdir()
    screen = receipt('c1', t=0.8, loss=lambda s: LOSS(s) * 1.02)            # 2 % from the reference
    (tmp_path / 'rc' / 'c1.json').write_text(json.dumps({**screen, 'config': {}, 'overlay': {}}), encoding='utf-8')
    runs = iter([receipt('rerun', t=1.0, loss=lambda s: LOSS(s) * 1.001),                  # the reference again
                 receipt('fresh', t=0.8, loss=lambda s: LOSS(s) * 1.02 * 1.0005)])         # the candidate again

    class Runner:
        def run(self, *a, **kw):
            return next(runs)

        def run_tests(self, *a, **kw):
            return []
    trace = {'candidates': [{'id': 'c1', 'valid': True, 'step_time': 0.8}], 'extra': {}}
    row = promote(p, None, Runner(), Verifier(), trace, receipt('ref', t=1.0), 'r', tmp_path / 'rc')['tried'][0]
    assert row['reference_repeat_loss_deviation'] == pytest.approx(0.001)
    assert row['candidate_repeat_loss_deviation'] == pytest.approx(0.0005)  # its fresh run vs its screen, not the 2 %
    assert not {'reference_loss_deviation', 'candidate_loss_deviation'} & set(row)


def test_the_candidate_table_shows_each_candidates_deviation_from_the_reference():
    cands = [{'id': 'c0', 'status': 'screened', 'valid': True, 'step_time': 1.0, 'loss_deviation': None},
             {'id': 'c1', 'status': 'screened', 'valid': True, 'step_time': 0.9, 'change': {'precision': ['fp32', 'bf16']},
              'loss_deviation': 0.018},
             {'id': 'c2', 'status': 'screened', 'valid': False, 'change': {'precision': ['fp32', 'fp8']},
              'loss_deviation': -0.035, 'failed_checks': ['loss_within_tolerance']},
             {'id': 'c3', 'status': 'aborted', 'valid': False, 'change': {'workers': [1, 2]}, 'loss_deviation': None}]
    r = {'run_id': 'p-1', 'run_dir': '/r', 'project': {'name': 'p', 'gpus': 1, 'steps': 12, 'window': [5, 12]},
         'tolerance': 0.03, 'outcome': 'no_change', 'reference': {'config': {}, 'step_time': 1.0}, 'strategy': 'random',
         'promotion': {'promoted': None, 'why': 'nothing passed', 'tried': []}, 'candidates': cands, 'screens': [],
         'search': {'used': {'screens': 3}}, 'cost': {}}
    text = markdown(r)
    assert '| candidate | change | result | step time (s) | loss vs ref | failed checks |' in text
    assert '| c0 | (the reference) | valid | 1 | - | - |' in text
    assert '| c1 | precision fp32 -> bf16 | valid | 0.9 | +1.80 % | - |' in text
    assert '| c2 | precision fp32 -> fp8 | invalid | - | -3.50 % | loss_within_tolerance |' in text
    assert '| c3 | workers 1 -> 2 | aborted (slower) | - | - | - |' in text
    assert '- **loss vs ref**: how far' in text       # explained under "How to read this report"
