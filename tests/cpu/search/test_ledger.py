"""The memory ledger (search/ledger.py; possible_search_algorithm/20 section 2): exact, lower-bound and interval
observations in one censored linear model; the prior's per-value declarations and formula terms; P_feas; no names."""
import math
import re
from pathlib import Path

import pytest

from infrarsi.search.ledger import INTERACT, NEAR, MemoryLedger, vk

KNOBS = {'mb': [1, 2, 4], 'rc': ['full', 'none'], 'fuse': [False, True], 'small': [0, 1]}
REF = {'mb': 1, 'rc': 'full', 'fuse': False, 'small': 0}
CAP = 80.0


def truth(c):
    """A planted memory: 30 GiB of state, activations 4 GiB x mb x (5 without recompute), logits 3 GiB x mb (a third
    fused); the small knob does nothing."""
    c = {**REF, **c}
    return 30.0 + 4.0 * c['mb'] * (5.0 if c['rc'] == 'none' else 1.0) + 3.0 * c['mb'] * (1 / 3 if c['fuse'] else 1.0)


def ledger(terms=True, declared=True):
    led = MemoryLedger(CAP, knobs=KNOBS, ref=REF, facts={'world': 8, 'cap_gib': CAP})
    levels = {}
    if declared:
        levels = {('mb', vk(2)): {'mem_dir': 'spends', 'mem_gib': 7.0}, ('mb', vk(4)): {'mem_dir': 'spends', 'mem_gib': 21.0},
                  ('rc', vk('none')): {'mem_dir': 'spends', 'mem_gib': 16.0},
                  ('fuse', vk(True)): {'mem_dir': 'frees', 'mem_gib': 2.0}}
    led.add_value_terms(levels)
    led.ensure_values()
    if terms:
        kept = led.add_formula_terms([
            {'name': 'activations', 'gib_expr': "3.0 * mb * (6 if rc == 'none' else 1)"},
            {'name': 'logits', 'gib_expr': 'mb * (1.5 if fuse else 4.0)'},
            {'name': 'bad', 'gib_expr': '__import__("os").getcwd()'},
            {'name': 'unknown name', 'gib_expr': 'seq * 2'},
            {'name': 'huge', 'gib_expr': '10 ** 9'}])
        assert kept == 2 and [d['name'] for d in led.dropped_terms] == ['bad', 'unknown name', 'huge']
    led.observe_exact(REF, truth(REF))
    return led


def test_exact_lower_and_interval_observations_fit_a_planted_memory():
    led = ledger()
    for c in ({'mb': 2}, {'rc': 'none'}, {'mb': 2, 'fuse': True}):
        led.observe_exact(c, truth(c))
    led.observe_lower({'mb': 4, 'rc': 'none'}, 72.0)                 # out of memory: at least what it held + asked
    led.observe_interval({'mb': 2, 'rc': 'none'})                     # the cliff
    for c in ({'mb': 4}, {'mb': 4, 'fuse': True}, {'rc': 'none', 'fuse': True}):
        mu, sd = led.predict(c)
        assert abs(mu - truth(c)) <= max(4.0, 2 * sd), (c, mu, sd, truth(c))
    mu, _ = led.predict({'mb': 4, 'rc': 'none'})
    assert mu >= 70.0                                                  # the censored run stays above its bound
    assert led.p_feasible({'mb': 4, 'rc': 'none', 'fuse': True}) < 0.05
    assert led.p_feasible({'mb': 2, 'fuse': True}) > 0.95


def test_a_run_is_its_own_result_and_feasibility_is_monotone_in_a_spend():
    led = ledger()
    led.observe_run({'mb': 4}, 'oom', oom={'tried_gib': 8.0, 'in_use_gib': 76.0})
    assert led.p_feasible({'mb': 4}) == 0.0 and led.ran_status({'mb': 4}) == 'oom'
    led.observe_run({'mb': 2}, 'ok', peak=truth({'mb': 2}))
    assert led.p_feasible({'mb': 2}) == 1.0
    led.observe_run({'mb': 2, 'rc': 'none'}, 'thrash')
    assert led.p_feasible({'mb': 2, 'rc': 'none'}) == 0.0
    ps = [led.p_feasible({'mb': m, 'rc': 'none', 'fuse': True}) for m in (1, 2, 4)]
    assert ps[0] >= ps[1] >= ps[2]
    assert [c['mb'] for c in led.order([{'mb': 4}, {'mb': 1}, {'mb': 2}])] == [1, 2, 4]
    assert 0.0 <= led.f_hat([{'mb': m} for m in (1, 2, 4)]) <= 1.0
    d = led.describe()
    assert d['observations'] == {'exact': 2, 'lower': 1, 'interval': 1} and d['terms'] == ['activations', 'logits']


def test_without_terms_or_declarations_the_runs_still_teach_it():
    led = ledger(terms=False, declared=False)
    mu0, sd0 = led.predict({'rc': 'none'})
    assert abs(mu0 - truth(REF)) < 1.0 and sd0 > 5.0                  # unknown: the reference, widely spread
    led.observe_exact({'rc': 'none'}, truth({'rc': 'none'}))
    mu1, sd1 = led.predict({'rc': 'none'})
    assert abs(mu1 - truth({'rc': 'none'})) < 1.5 and sd1 < sd0


def test_no_card_means_no_feasibility_judgement():
    led = MemoryLedger(None, knobs=KNOBS, ref=REF)
    led.ensure_values()
    assert led.p_feasible({'mb': 4, 'rc': 'none'}) == 1.0


def test_the_module_names_no_knob_model_or_library():
    src = Path(__file__).resolve().parents[3].joinpath('infrarsi', 'search', 'ledger.py').read_text(encoding='utf-8')
    code = '\n'.join(ln for ln in src.splitlines() if not ln.strip().startswith('#'))
    for word in ('recompute', 'micro_batch', 'liger', 'compile', 'fsdp', 'deepspeed', 'megatron', 'torch', 'qwen', 'gemma'):
        assert not re.search(rf'\b{word}\b', code, re.I), word
    assert NEAR == pytest.approx(0.92) and math.isfinite(CAP)


def test_a_combination_of_measured_values_keeps_the_additive_models_own_error():
    """Each value measured alone is known; their combination is not (what one frees scales with what another spends):
    the spread of a combination carries INTERACT x sqrt(sum over pairs of |effect_i| |effect_j|)."""
    led = ledger(terms=False, declared=False)
    for c in ({'mb': 4}, {'fuse': True}):
        led.observe_exact(c, truth(c))
    _, sd_alone = led.predict({'mb': 4})
    mu, sd = led.predict({'mb': 4, 'fuse': True})
    e_mb, e_fuse = truth({'mb': 4}) - truth(REF), truth(REF) - truth({'fuse': True})
    assert sd_alone < 3.0 and sd >= INTERACT * math.sqrt(e_mb * e_fuse) - 1e-6 and sd > sd_alone
    assert abs(mu - (truth({'mb': 4}) + truth({'fuse': True}) - truth(REF))) < 1.0   # the additive mean


def test_an_interval_far_outside_its_band_fits_without_overflow():
    led = MemoryLedger(CAP, knobs=KNOBS, ref=REF)
    led.add_value_terms({('mb', vk(4)): {'mem_dir': 'spends', 'mem_gib': 400.0}})   # a wild declaration
    led.ensure_values()
    led.observe_exact(REF, 40.0)
    led.observe_interval({'mb': 4})                                    # thrash: within [0.9, 1] x card after all
    mu, sd = led.predict({'mb': 4})
    assert math.isfinite(mu) and math.isfinite(sd) and mu < 2 * CAP
