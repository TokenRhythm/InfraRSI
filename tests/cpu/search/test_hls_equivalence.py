"""hls is unchanged by hls2 (FABLE_DESIGN_SEARCH_FIX, 2026-10-29): every new behaviour is a switch that hls leaves off.
The golden (data/hls_prechange_golden.json) is what hls did before the change - made with the pre-change
strategies.py on the scenarios of hls2_helpers.SCENARIOS (the recorded Q9U-code-env-opus run for 10 screens, the same
job for 24 screens - re-asks, the neighbourhood, bundles and merges - and a variant with covered, known-to-run and
memory-checked experience): its decisions, the ledger as recorded, each model call's ledger (digest), describe and
residual must come out the same. After an intended change of hls, rewrite it with
`python tests/cpu/search/test_hls_equivalence.py --write`."""
import json
import sys
from pathlib import Path

if __name__ == '__main__':                                          # run as a script: the repo's package
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest  # noqa: E402

from hls2_helpers import SCENARIOS, run_scenario  # noqa: E402
from infrarsi.search.strategies import HLS  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / 'data' / 'hls_prechange_golden.json'


@pytest.mark.parametrize('name', sorted(SCENARIOS))
def test_hls_decides_shows_and_records_as_before(name):
    want = json.loads(GOLDEN.read_text(encoding='utf-8'))[name]
    now = run_scenario(HLS, name)
    assert sorted(now) == sorted(want)
    for key in want:                                                # one field at a time: the failure names it
        assert now[key] == want[key], key


def test_hls_prompt_is_unchanged_and_hls2_explains_the_states():
    from infrarsi.agent.backends.fake import ScriptedBackend
    from infrarsi.search.proposers import LEDGER_STATES_RULE, LLMProposer
    from test_search import TOGGLE, setup
    drv = ScriptedBackend(lambda role, system, prompt: {'verdict': 'no_patch', 'findings': []} if role == 'code_review'
                          else {'hypotheses': []})
    ledger = [{'mechanism': 'a', 'change': {'fused_adam': True}, 'status': 'refuted', 'measured_gain_pct': -0.4}]
    sim, s, eng = setup(TOGGLE, tokens=10 ** 6)
    used = dict(s.ledger.used)
    for states in (False, True):                                    # each from the same budget (the prompt shows it)
        s.ledger.used.update(used)
        LLMProposer(drv, s.hooks, search_calibrates=True, **({'ledger_states': True} if states else {})).hypotheses(
            eng, 'c0', ledger=ledger)
    hyp = [c for c in drv.calls if c['role'] == 'hypotheses']
    assert len(hyp) == 2 and hyp[0]['prompt'] == hyp[1]['prompt']
    assert 'LEDGER STATES' not in hyp[0]['system'] and hyp[1]['system'] == hyp[0]['system'] + LEDGER_STATES_RULE


if __name__ == '__main__' and '--write' in sys.argv:
    GOLDEN.write_text(json.dumps({n: run_scenario(HLS, n) for n in SCENARIOS}, indent=0, sort_keys=True,
                                 separators=(',', ':')) + '\n', encoding='utf-8', newline='\r\n')
    print('wrote', GOLDEN)
