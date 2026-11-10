"""F-F-4 (phase F, 2026-10-24): sft_loop's `bf16` switch mapped to no mechanism, so the library's precision evidence was
never used (x1.52; declared as precision by the user: x3.52), and the plan's line only said "no mechanism known for
bf16". It now names the mechanisms that have such a state."""
from types import SimpleNamespace

from infrarsi.experience import runs as exp_runs
from infrarsi.experience.registry import state_owners


def test_a_knob_named_like_a_state_gets_its_mechanisms_named():
    assert state_owners('bf16')[0] == 'precision' and len(state_owners('bf16')) <= 3
    assert state_owners('on') == [] and state_owners('grad_ckpt') == []
    assert state_owners('zbvzerobubble') == ['pp_schedule'] and state_owners('a2a_p2p') == ['cp_comm']   # DF11_r1


def test_the_plan_line_names_the_likely_mechanism():
    knobs = {'bf16': {'type': 'bool', 'flag': '--bf16', 'default': False},
             'grad_ckpt': {'type': 'bool', 'flag': '--grad-ckpt', 'default': False}}
    view = SimpleNamespace(raw={'hardware': {'gpu_type': 'NVIDIA H100 80GB HBM3', 'memory_gib': 80}, 'framework': 'pytorch',
                                'model': {'params_b': 0.5}, 'workload': {'seq_len': 8192}, 'search': {}},
                           gpus=1, nodes=1, knobs=knobs, name='t', window=[11, 30],
                           reference_config=lambda: {'bf16': False, 'grad_ckpt': False})
    lines = []
    exp_runs.plan_for(view, None, select=None, driver=None, log=lines.append)
    line = next(m for m in lines if 'no mechanism known for' in m)
    assert 'bf16 (a state of precision' in line and 'grad_ckpt (a state' not in line
