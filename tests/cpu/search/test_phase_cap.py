"""The phase cap (textbook design P0-2 step 2, 2026-10-24): HLS caps an LLM prediction at what the phase classes its
knobs' mechanisms act on can give, -log(1 - their share of the update); never a patch, a knob acting on every phase
(layout, precision, recompute) or without a mapped mechanism; off unless asked."""
import math

import pytest

from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.search.proposers import LLMProposer
from infrarsi.search.strategies import HLS

from test_search import TOGGLE, setup

HYPS = {'hypotheses': [
    {'mechanism': 'overlap grads', 'change': {'overlap_grad': True}, 'predicted_gain_pct': 20},
    {'mechanism': 'fused adam', 'change': {'fused_adam': True}, 'predicted_gain_pct': 5},
    {'mechanism': 'flash', 'change': {'flash_attn': True}, 'predicted_gain_pct': 2},
    {'mechanism': 'graphs', 'change': {'cuda_graphs': True}, 'predicted_gain_pct': 9}]}


def _hls(cap: bool):
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    s.hooks.knob_phases = {'overlap_grad': ['comm'], 'fused_adam': 'all', 'flash_attn': ['compute']}
    s.hooks.class_shares = lambda receipt: {'compute': 0.94, 'comm': 0.033, 'optimizer': 0.018}
    hls = HLS(LLMProposer(ScriptedBackend(lambda role, system, prompt: HYPS), s.hooks), phase_cap=cap)
    hls._refresh(eng, 'c0')
    return {h['mechanism']: h for h in hls.ledger.values()}, hls


def test_a_prediction_is_capped_at_its_phases_share():
    led, hls = _hls(True)
    assert led['overlap grads']['mu'] == pytest.approx(-math.log(1 - 0.033))
    assert led['overlap grads']['mu_raw'] == pytest.approx(math.log1p(0.20))
    assert not any('mu_cap' in led[m] for m in ('fused adam', 'flash', 'graphs'))   # all / under its cap / unmapped
    row = next(r for r in hls.ledger_view() if r['mechanism'] == 'overlap grads')
    assert row['mu_cap_pct'] == 3.36 and row['predicted_gain_pct'] == 3.36 and row['predicted_raw_pct'] == 18.23
    assert not any(k in r for r in hls.model_view() for k in ('mu_cap_pct', 'predicted_raw_pct'))
    off, _ = _hls(False)
    assert 'mu_cap' not in off['overlap grads'] and off['overlap grads']['mu'] == pytest.approx(math.log1p(0.20))


def test_an_unmeasured_class_leaves_the_cap_off_and_a_twin_keeps_the_capped_prediction():
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    s.hooks.knob_phases = {'overlap_grad': ['comm'], 'fused_adam': ['optimizer']}
    s.hooks.class_shares = lambda receipt: {'compute': 0.8, 'comm': 0.033}         # no optimizer timer logged
    hls = HLS(LLMProposer(ScriptedBackend(lambda role, system, prompt: HYPS), s.hooks), phase_cap=True)
    hls.ledger['c0|experience:dp_grad_overlap'] = {'mechanism': 'experience:dp_grad_overlap', 'base': 'c0',
                                                   'source': 'experience', 'status': 'open', 'change': {'overlap_grad': True},
                                                   'mu': 0.02, 'sigma': 0.01, 'p_feasible': 0.9, 'patch': None}
    hls._refresh(eng, 'c0')
    led = {h['mechanism']: h for h in hls.ledger.values()}
    assert 'mu_cap' not in led['fused adam']                                   # unknown, not zero
    twin = led['experience:dp_grad_overlap']
    assert twin['llm_mu'] == pytest.approx(-math.log(1 - 0.033)) and twin['mu'] == 0.02   # the seed's prior stays


def test_the_manifest_switch(tmp_path):
    from infrarsi.project import Project
    from infrarsi.project.manifest import ManifestError
    raw = {'schema': 'infrarsi-project-v1', 'name': 'x', 'launch': {'command': 'true'}}
    assert Project(raw, tmp_path / 'infrarsi.yaml').raw['search']['phase_cap'] is False
    assert Project({**raw, 'search': {'phase_cap': True}}, tmp_path / 'infrarsi.yaml').raw['search']['phase_cap'] is True
    with pytest.raises(ManifestError, match='search.phase_cap: true or false'):
        Project({**raw, 'search': {'phase_cap': 'yes'}}, tmp_path / 'infrarsi.yaml')
