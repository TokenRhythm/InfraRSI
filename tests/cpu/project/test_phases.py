"""B3 (2026-10-23 Q&A follow-up): where the time of an update goes - Megatron's timer block, a generic `signals.phases`
map and the Probe's `phases` - kept in the receipt's evidence, shown by doctor / the report, and given to the model
next to the verdict."""
import json

import pytest

from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.project import Project
from infrarsi.project import signals as sig
from infrarsi.project.hooks import ProjectHooks
from infrarsi.project.manifest import ManifestError

from test_project import toy  # noqa: F401 - the toy project fixture

MEGATRON = [
    (1.0, ' [2026-10-23 04:39:07] iteration        6/       8 | elapsed time per iteration (ms): 2561.6 | lm loss: 6.246416E+00 |'),
    (1.1, 'max time across ranks (ms):'),
    (1.2, '    forward-backward ...............................: 2505.55'),
    (1.3, '    all-grads-sync .................................: 85.11'),
    (1.4, '    params-all-gather ..............................: 30.29'),
    (1.5, '    optimizer ......................................: 45.09'),
    (3.6, ' [2026-10-23 04:39:10] iteration        7/       8 | elapsed time per iteration (ms): 2563.9 | lm loss: 5.066039E+00 |'),
    (3.7, '    forward-backward ...............................: (2501.20, 2508.17)'),
]


def test_megatron_timer_block_becomes_phases_of_its_update():
    rows = sig.LogParser(megatron_timers=True).parse(MEGATRON)
    assert rows[0]['phases'] == {'forward-backward': 2505.55, 'all-grads-sync': 85.11, 'params-all-gather': 30.29,
                                 'optimizer': 45.09}
    assert rows[1]['phases'] == {'forward-backward': 2508.17} and rows[1]['loss'] == 5.066039
    ph = sig.phase_summary(rows, [6, 7], 2.5626, sig.MEGATRON_TOP)
    assert ph['top'] == ['forward-backward', 'optimizer'] and ph['coverage'] == '2/2'   # all-grads-sync is nested
    assert ph['residual_ms'] == pytest.approx(2562.6 - 2506.86 - 45.09) and 'residual_note' not in ph
    assert sig.LogParser().parse(MEGATRON)[0].get('phases') is None           # off unless asked for


def test_generic_phase_patterns_and_units():
    p = sig.LogParser(phases={'fwd': r'fwd (?P<v>[\d.]+)(?P<unit>ms|s)', 'opt': r'opt=(?P<v>[\d.]+)'})
    rows = p.parse([(1.0, 'step 1 | loss 2.0 | fwd 0.5s | opt=12'), (2.0, 'step 2 | loss 1.9 | fwd 480ms | opt=11')])
    assert rows[0]['phases'] == {'fwd': 500.0, 'opt': 12.0} and rows[1]['phases'] == {'fwd': 480.0, 'opt': 11.0}
    ev = sig.evidence(rows, steps=2, window=[1, 2], exit_code=0, timed_out=False, events={}, frozen={})
    assert ev['phases']['top'] == ['fwd', 'opt'] and ev['per_step'][0]['phases']['fwd'] == 500.0


def test_probe_phases_pass_through():
    text = '\n'.join(json.dumps({'step': s, 't': 10.0 + s, 'loss': 2.0, 'phases': {'data': 5.0, 'forward_backward': 900.0,
                                                                                    'optimizer': 80.0}}) for s in (1, 2, 3))
    rows, _ = sig.read_probe(text)
    ev = sig.evidence(rows, steps=3, window=[2, 3], exit_code=0, timed_out=False, events={}, frozen={})
    assert ev['phases']['mean_ms']['forward_backward'] == 900.0 and ev['phases']['residual_ms'] == 15.0


def test_the_model_sees_the_phases_next_to_the_verdict(toy):
    receipt = {'attempt_id': 'a1', 'valid': True, 'evidence': {'phases': sig.phase_summary(
        sig.LogParser(megatron_timers=True).parse(MEGATRON), [6, 7], 2.5626, sig.MEGATRON_TOP)}}
    digest = ProjectHooks(toy, toy.task()).receipt_digest(receipt, {'valid': True, 'step_time': 2.56})
    assert list(digest)[:3] == ['id', 'verdict', 'step_phases']
    assert digest['step_phases']['phases'][0] == {'name': 'forward-backward', 'ms': 2506.9, 'share': 0.9782, 'top': True}
    raw = dict(toy.raw, signals=dict(toy.raw['signals'], phases_in_prompt=False))
    assert 'step_phases' not in ProjectHooks(Project(raw, toy.path), toy.task()).receipt_digest(receipt, {'valid': True})


def test_phase_rule_only_with_phases(toy):
    from infrarsi.search.proposers import LLMProposer
    from infrarsi.search.engine import Engine
    from infrarsi.search import Context

    calls = []

    class Hooks(ProjectHooks):
        def receipt_digest(self, *a, **kw):
            return {'id': {}, 'step_phases': {'phases': [{'name': 'fwd', 'ms': 1.0}]}}

    drv = ScriptedBackend(lambda role, system, prompt: calls.append(system) or {'hypotheses': []})
    hooks = Hooks(toy, toy.task())

    class Eng:                                       # the proposer reads the session's incumbent, task and budget
        class s:
            candidates = {'c0': type('C', (), {'receipt': {'x': 1}, 'verdict': {}, 'config': {}, 'overlay': {}})()}
            task = hooks.task
            ledger = type('L', (), {'exhausted': staticmethod(lambda: []), 'used': {'llm_tokens': 0, 'llm_usd': 0.0},
                                    'budget': type('B', (), {'llm_tokens': 10 ** 6, 'llm_usd': None, 'wall_seconds': 1e9}),
                                    'wall': staticmethod(lambda: 0.0)})

            @staticmethod
            def budget_view():
                return {}

            @staticmethod
            def charge_llm(res, role):
                pass

    LLMProposer(drv, hooks).hypotheses(Eng, 'c0')
    assert 'PHASES:' in calls[0]
    assert Engine and Context                        # imported for the module's side effects only


def test_manifest_checks_phase_patterns(toy):
    raw = dict(toy.raw, signals=dict(toy.raw['signals'], phases={'fwd': 'fwd ([0-9.]+)', 'bad': '(unclosed'}))
    with pytest.raises(ManifestError) as e:
        Project(raw, toy.path)
    assert 'signals.phases.fwd: the pattern needs a named group v' in str(e.value)
    assert 'signals.phases.bad: not a regular expression' in str(e.value)


def test_phase_lines_keep_their_signals_and_their_update():
    hf = sig.LogParser(count_steps=True, phases={'fwd': r"'fwd_ms': (?P<v>[\d.]+)"})
    rows = hf.parse([(1.0, "{'loss': 2.3, 'grad_norm': 1.1, 'fwd_ms': 500}"), (2.0, "{'loss': 2.2, 'grad_norm': 1.0, 'fwd_ms': 480}")])
    assert [(r['step'], r['loss'], r['phases']) for r in rows] == [(1, 2.3, {'fwd': 500.0}), (2, 2.2, {'fwd': 480.0})]
    cont = sig.LogParser(phases={'fwd': r'fwd_ms=(?P<v>[\d.]+)'}).parse(
        [(1.0, 'step 1 loss 2.0'), (2.0, 'memory: 12GiB step_time: 1s fwd_ms=500')])
    assert cont[0]['mem_gib'] == 12.0 and cont[0]['step_time'] == 1.0 and cont[0]['phases'] == {'fwd': 500.0}


def test_checkpoint_and_eval_timers_are_not_update_phases():
    lines = MEGATRON[:6] + [(9.0, '    save-checkpoint ................................: 5000.00'),
                            (9.5, '    evaluate .......................................: 9000.00')]
    rows = sig.LogParser(megatron_timers=True).parse(lines)
    assert set(rows[0]['phases']) == {'forward-backward', 'all-grads-sync', 'params-all-gather', 'optimizer'}


def test_digest_with_phases_keeps_the_token_budget(toy):
    from infrarsi.agent.context import estimate_tokens
    ph = sig.phase_summary(sig.LogParser(megatron_timers=True).parse(MEGATRON), [6, 7], 2.5626, sig.MEGATRON_TOP)
    receipt = {'attempt_id': 'a1', 'valid': True, 'evidence': {'phases': ph}, 'metric': {'name': 'x', 'value': 1.0},
               'summary': {'note': 'y' * 800}, 'cost': {'gpu_seconds': 1.0}}
    d = ProjectHooks(toy, toy.task()).receipt_digest(receipt, {'valid': True, 'step_time': 2.56}, max_tokens=300)
    assert list(d)[:3] == ['id', 'verdict', 'step_phases'] and estimate_tokens(json.dumps(d, default=str)) <= 300



def test_the_report_says_what_was_tried_on_each_phase():
    from infrarsi.project.report import _phase_tried
    r = {'knob_phases': {'overlap_grad_reduce': ['comm'], 'micro_batch_size': 'all', 'attention_backend': ['compute']},
         'search': {'ledger': [
             {'mechanism': 'overlap grads', 'change': {'overlap_grad_reduce': True}, 'status': 'refuted', 'measured_gain_pct': 0.42},
             {'mechanism': 'mbs2', 'change': {'micro_batch_size': 2}, 'status': 'invalid', 'measured_gain_pct': None},
             {'mechanism': 'flash', 'change': {'attention_backend': 'flash'}, 'status': 'open'},
             {'mechanism': 'gc', 'change': {'manual_gc': True}, 'status': 'confirmed', 'measured_gain_pct': 1.0}]}}
    assert _phase_tried(r) == ['Tried on each phase class: comm: overlap grads (x1.004); every phase: mbs2 (invalid); '
                               'unmapped: gc (x1.010).']
    assert _phase_tried({'search': {}}) == []


def test_phase_classes_and_mechanism_sets():
    from infrarsi.experience.registry import mechanism_phases, mechanisms, phase_class, phases
    assert [phase_class(n, 'megatron') for n in ('forward-backward', 'all-grads-sync', 'optimizer-inner-step')] == \
        ['compute', 'comm', 'optimizer']
    assert [phase_class(n) for n in ('fwd', 'opt', 'data_loading', 'all_reduce_ms', 'weird')] == \
        ['compute', 'optimizer', 'data', 'comm', None]
    assert mechanism_phases('tp') == 'all' and mechanism_phases('dp_grad_overlap') == ['comm']
    assert mechanism_phases('bf16_main_grads') == 'all' and mechanism_phases('nope') is None   # H1: +4.3 % beyond optimizer
    assert set(phases()['mechanisms']) <= set(mechanisms())
    assert set(sig.MEGATRON_UPDATE_TIMERS) <= set(phases()['timers']['megatron'])


def test_the_digest_names_the_knobs_that_reach_the_largest_phases(toy):
    from infrarsi.project.hooks import phase_targets
    ph = sig.phase_summary(sig.LogParser(megatron_timers=True).parse(MEGATRON), [6, 7], 2.5626, sig.MEGATRON_TOP)
    knobs = {'overlap_grad_reduce': ['comm'], 'fused_ce': ['compute'], 'micro_batch_size': 'all', 'manual_gc': ['residual']}
    t = phase_targets(ph, knobs, 'megatron')
    assert [(x['phase'], x['class'], x['knobs']) for x in t['largest']] == [
        ('forward-backward', 'compute', ['fused_ce']), ('all-grads-sync', 'comm', ['overlap_grad_reduce']),
        ('optimizer', 'optimizer', [])] and t['any_phase'] == ['micro_batch_size']
    assert [x['phase'] for x in phase_targets(dict(ph, residual_ms=300.0), knobs, 'megatron')['largest']] == \
        ['forward-backward', 'all-grads-sync', 'optimizer']                 # a residual of per-rank maxima: no row
    host = phase_targets(dict(ph, residual_ms=300.0, source='profile'), knobs, 'megatron')['largest']   # GPU idle
    assert [x['phase'] for x in host] == ['forward-backward', 'residual', 'all-grads-sync'] and host[1]['knobs'] == ['manual_gc']
    receipt = {'attempt_id': 'a1', 'valid': True, 'evidence': {'phases': ph}}
    digest = ProjectHooks(toy, toy.task()).receipt_digest(receipt, {'valid': True, 'step_time': 2.56})
    assert list(digest)[:4] == ['id', 'verdict', 'step_phases', 'phase_targets']
    assert digest['phase_targets']['any_phase'] == ['precision']           # the toy's knobs: precision acts everywhere
