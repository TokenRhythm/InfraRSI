"""DF5_r2 #1-#2 (2026-10-24): a confirmation re-run is a trace row of its own - its events.jsonl event is the row's
call-to-row boundary, and the measurement the summary resolves for it is its own, not its candidate's latest."""
import json
import math
from types import SimpleNamespace

import pytest

from infrarsi.ext.common import trajectory as TJ
from infrarsi.ext.summary.records import Store


def _row(k, cid, gpuh, fidelity='full'):
    return {'index': k, 'candidate_id': cid, 'fidelity': fidelity, 'valid': True, 'step_time': 1.0,
            'cum': {'gpu_hours': gpuh, 'llm_usd': 0.0, 'wall_seconds': 100.0 * (k + 1), 'llm_tokens': 0}}


def test_a_rescore_event_is_its_rows_boundary(tmp_path):
    ev = [{'i': 0, 'kind': 'llm_call', 'payload': {'call_id': 'k1'}}, {'i': 1, 'kind': 'screen', 'payload': {'index': 0}},
          {'i': 2, 'kind': 'candidate', 'payload': {}}, {'i': 3, 'kind': 'screen', 'payload': {'index': 1}},
          {'i': 4, 'kind': 'rescore', 'payload': {'index': 2}}, {'i': 5, 'kind': 'rescore', 'payload': {'index': 3}}]
    p = tmp_path / 'events.jsonl'
    p.write_text(''.join(json.dumps(e) + '\n' for e in ev), encoding='utf-8')
    assert Store(tmp_path).event_sequence(p) == [('call', 'k1'), ('screen', 0), ('screen', 1), ('screen', 2), ('screen', 3)]


def test_a_confirmation_measurement_belongs_to_its_own_row():
    # the r2 reproduction: TFLOPS 100, 104, 102, 106 for c0, c1, c0 (re-run), c1 (re-run); +5 % first on the fourth run
    tr = {'screens': [_row(0, 'c0', 0.25), _row(1, 'c1', 0.5), _row(2, 'c0', 0.75, 'rescore'),
                      _row(3, 'c1', 1.0, 'rescore')],
          'used': {'gpu_hours': 1.0, 'llm_usd': 0.0, 'wall_seconds': 400.0, 'llm_tokens': 0}}
    task = SimpleNamespace(env='optico', tau=math.log(1.05))
    values = {k: (1.0, {'tflops': t}) for k, t in enumerate((100.0, 104.0, 102.0, 106.0))}
    evs = TJ.events(tr, task, 1.0, ref_raw={'tflops': 100.0}, values=values)
    assert [-math.log(e.step_time) for e in evs] == pytest.approx([math.log(t / 100.0) for t in (100, 104, 102, 106)])
    c = TJ.ctg(evs, task)
    assert c['reached'] and c['gpuh'] == pytest.approx(1.0)
