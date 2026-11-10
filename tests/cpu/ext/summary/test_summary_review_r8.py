"""Regression tests for the Codex review round 8 of the W5 summary (R8-N1): the G4 D inventory's node references are
derived from the nodes the non-cut ENV-A arms ran on (the block node is the block S0's), and the rows must match those
expected identities one to one - a deleted, null, `not run (...)`, block-mapped or file-less node reference of an
applicable node, or a missing / repeated / unexpected inventory row, keeps the limit non-final."""
import pytest
from summary_helpers import World

from infrarsi.ext.summary import env_a
from infrarsi.ext.summary import records as RC

PAIRS = (('aisw_x', 'anat_x'), ('aisw_x', 'arnd_x'), ('anat_x', 'arnd_x'))
EXPECTED = ['block S0', 'node_ref WINH3', 'node_ref WINH4']
RUNS = {'A-NAT': 'anat_x', 'A-ISW': 'aisw_x', 'A-RND': 'arnd_x'}


def g4(pairs=PAIRS):
    return {'final': True, 'refprime_pairwise_norms': [{'pair': list(p), 'norm_diff': 4.9, 'source': 't'} for p in pairs]}


def build_a(w, inputs=None):
    return env_a.build(RC.Store(w.root), w.root, w.ledger, abandoned=RC.abandoned(w.ledger, w.root),
                       g4_inputs=inputs or g4())


def row(label, v=4.2):
    return {'label': label, 'source': f'src-{label}', 'norm_diff': v, 'why': None}


def test_r8n1_the_expected_identities_are_the_arms_nodes(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    cal = build_a(w)['g4_calibration']
    assert cal['d_expected'] == EXPECTED and cal['final'] is True


@pytest.mark.parametrize('change, why', [
    ('delete', 'no node_refs entry in the ledger'),
    ('not run (cut by the orchestrator)', "marked 'not run (cut by the orchestrator)' while an arm ran on WINH4"),
    ('block S0', 'mapped to the block S0, but the block ran on WINH1'),
    ('nref_ghost', 'node_ref_WINH4.json of nref_ghost not found'),
])
def test_r8n1_an_applicable_node_reference_that_is_absent_keeps_g4_non_final(tmp_path, change, why):
    w = World(tmp_path)
    w.env_a_complete()                                          # A-RND finished on WINH4: WINH4 is applicable
    if change == 'delete':
        del w.ledger['ENV-A']['node_refs']['WINH4']
    else:
        w.ledger['ENV-A']['node_refs']['WINH4'] = change
    sec = build_a(w)
    cal = sec['g4_calibration']
    assert cal['final'] is False and any(x == f'node_ref WINH4: {why}' for x in cal['d_inventory_gaps']), \
        cal['d_inventory_gaps']
    assert any("limit_D'" in p and 'not final' in p for p in sec['pending'])


def test_r8n1_a_cut_arm_takes_its_node_out_of_the_expected_set(tmp_path):
    w = World(tmp_path)
    w.env_a_complete()
    w.ledger['ENV-A']['arms']['A-RND'] = 'not run: cut (§6.4)'
    del w.ledger['ENV-A']['node_refs']['WINH4']
    cal = build_a(w, g4(pairs=PAIRS[:1]))['g4_calibration']
    assert cal['d_expected'] == ['block S0', 'node_ref WINH3'] and cal['final'] is True


def test_r8n1_the_inventory_must_match_the_expected_identities_one_to_one():
    fn = env_a.g4_calibration
    full = [row('block S0', 4.0), row('node_ref WINH3'), row('node_ref WINH4', 3.4)]
    assert fn(g4(), full, RUNS, expected=EXPECTED)['final'] is True
    cal = fn(g4(), full[:2], RUNS, expected=EXPECTED)                       # Codex's probe: WINH4 omitted
    assert cal['final'] is False and 'node_ref WINH4: absent from the D inventory' in cal['d_inventory_gaps']
    cal = fn(g4(), full + [row('node_ref WINH4', 3.5)], RUNS, expected=EXPECTED)
    assert cal['final'] is False and 'node_ref WINH4: 2 rows in the D inventory' in cal['d_inventory_gaps']
    cal = fn(g4(), full + [row('node_ref WINH9', 1.0)], RUNS, expected=EXPECTED)
    assert cal['final'] is False and 'node_ref WINH9: not an expected D sample' in cal['d_inventory_gaps']
    assert all(x['norm_diff'] != 1.0 for x in cal['samples_d'])         # an unexpected row never enters m
