"""ext/flashnext arms end to end on the fake four-node runner (X6; AMENDMENT_EXT_003 Q2, Q5): F-RND, F-A0 (optimize with
hooks), F-OPTICO (optico_style with hooks), F-PROMPTS (run_prompts with hooks), locked pairs, margin_F1, labels, the
summary. Every screen goes through FlashNextRunner.x6_judge (parse -> judge -> guards) and is a signed receipt."""
import json
import math
import statistics
from pathlib import Path

import pytest

import x6_helpers as H
from infrarsi.agent.backends import ScriptedBackend
from infrarsi.agent.budget import VirtualClock
from infrarsi.baselines.optico_style import optico as oc
from infrarsi.ext.common import trajectory as TJ
from infrarsi.ext.flashnext import arms
from infrarsi.ext.flashnext import switch_space as S
from infrarsi.ext.flashnext import task as T
from infrarsi.harness import signing
from infrarsi.rsi import build_a0

START = S.start_point()
SMATCH = S.registered_points()['S_match']
PASS = {'verdict': 'pass', 'reasons': []}
D_F1 = ['D-F1-1', 'D-F1-2', 'D-F1-3', 'D-F1-4', 'D-F1-5', 'D-F1-6']


@pytest.fixture(scope='module')
def a0(tmp_path_factory):
    return build_a0(tmp_path_factory.mktemp('pkg') / 'A0')


@pytest.fixture(scope='module')
def uc(tmp_path_factory):
    return H.uc_reference_receipt(tmp_path_factory.mktemp('uc'), clock=VirtualClock())


def _receipts(d: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((d / 'receipts').glob('*.json'))]


def _prompts_trace(tmp_path, uc, point=SMATCH, runner=None, clock=None):
    """A real F-PROMPTS search whose only accepted recommendation is `point` (the selection when it is faster)."""
    clock = clock or VirtualClock()
    r = runner or H.FakeRunner(tmp_path / 'run', clock=clock)
    others = [{'config': {'tp': 4}, 'rationale': 'outside'}, {'config': {'tp': 8}, 'rationale': 'outside'}]
    first = {'config': {k: v for k, v in point.items() if v is not None or k == 'implementation_stack'}, 'rationale': 'x'}
    drv = ScriptedBackend([{'configs': [first] + others}], clock=clock)
    tr = arms.run_f_prompts(r, run_id='fp', out_dir=tmp_path / 'fp', uc_ref=uc, backend=drv, clock=clock)
    return tr, r, tmp_path / 'fp' / 'trace.json'


# ---------------------------------------------------------------- F-RND
def test_f_rnd_seed_42_same_validator_twelve_new_screens(tmp_path, uc):
    clock = VirtualClock()
    r = H.FakeRunner(tmp_path / 'run', clock=clock)
    tr = arms.run_f_rnd(r, run_id='frnd', out_dir=tmp_path / 'frnd', uc_ref=uc, clock=clock)
    assert tr['arm'] == 'F-RND' and len(tr['screens']) == 12 and tr['stop_reason'] == 'budget:screens'
    pts = [x['point'] for x in r.launched]
    assert pts == S.sample_order(seed=42, n=12) and START not in pts and len({S.key(p) for p in pts}) == 12
    assert tr['extra']['seed'] == 42 and tr['invalid_proposals'] == {} and tr['budget']['llm_usd'] is None
    assert tr['used']['llm_tokens'] == 0 and tr['start_step_time'] == uc['metric']['value']
    recs = [x for x in _receipts(r.out_dir) if x.get('purpose') == 'screen']
    assert len(recs) == 12 and all(signing.verify(x) and x['valid'] and x['deviations'] == D_F1 for x in recs)
    assert tr['selected'] != 'c0' and tr['selected_step_time'] < uc['metric']['value']


# ---------------------------------------------------------------- F-A0
def test_f_a0_through_optimize_with_hooks(tmp_path, uc, a0):
    clock = VirtualClock()
    r = H.FakeRunner(tmp_path / 'run', clock=clock)
    script = [{'action': 'read_env'},
              {'action': 'propose_config', 'base': 'c0', 'changes': {k: v for k, v in SMATCH.items() if v is not None},
               'rationale': 'keep the attention prefix and the expert layers'},
              {'action': 'run_screen', 'candidate_id': 'c1'},
              {'action': 'read_receipt', 'candidate_id': 'c1'},
              {'action': 'propose_config', 'base': 'c1', 'changes': {'moment_policy': 'scaled_fp16_dual_cuda'}, 'rationale': 'x'},
              {'action': 'propose_config', 'base': 'c0', 'changes': {'implementation_stack': 'native_rollbacks'}, 'rationale': 'y'},
              {'action': 'finish', 'candidate_id': 'c1', 'reason': 'done'}]
    drv = ScriptedBackend(lambda role, s, p: PASS if role == 'verifier' else script.pop(0), clock=clock)
    tr = arms.run_f_a0(r, run_id='fa0', out_dir=tmp_path / 'fa0', uc_ref=uc, package=a0, backend=drv, clock=clock)
    assert tr['arm'] == 'F-A0' and tr['stop_reason'] == 'finish' and tr['selected'] == 'c1'
    assert [s['candidate_id'] for s in tr['screens']] == ['c1'] and len(r.launched) == 1          # c0 not re-screened
    assert tr['invalid_proposals'] == {'l0_invalid': 1, 'duplicate': 1}
    assert tr['verifier']['calls'] == 1 and tr['tool_calls'] == [] and tr['skill_hits']
    assert tr['budget']['screens'] == 12 and tr['budget']['llm_usd'] == 40.0 and tr['mode'] == 'strong'
    events = [json.loads(x) for x in (tmp_path / 'fa0' / 'events.jsonl').read_text().splitlines()]
    fb = next(e['payload']['feedback'] for e in events if e['kind'] == 'turn' and e['payload']['action'].get('action') == 'read_receipt')
    assert 'analyses' not in fb and fb['numerics']['class'] == 'E0' and fb['point'] == SMATCH
    prompt = next(c['prompt'] for c in drv.calls if c['role'] == 'proposer')
    task_block = prompt.split('# TASK\n', 1)[1].split('\n# STATE', 1)[0]
    assert T.task_view_leaks(task_block) == [] and '"implementation_stack"' in task_block
    hooks = tr['extra']['hooks']
    assert hooks['tools_enabled'] is False and hooks['layers'] == ['L0'] and hooks['switch_space_sha256'] == S.sha256()


# ---------------------------------------------------------------- F-OPTICO
ANALYZER = '''
class ChangePrompt:
    def give_sugg(self, cot_path=None):
        instruction = "Synthetic give_sugg instruction.\\n"
        user_prompt = f"Question:\\n{self.input_text}"

    def change_para(self, cot=None):
        instruction = "Synthetic change_para instruction.\\n"
        user_prompt = f"Question:\\n{self.input_text}"

    def loc_error(self, cot=None):
        instruction = "Synthetic loc_error instruction.\\n"
        user_prompt = f"Question:\\n{self.input_text}"
'''
PROMPTS = {
    'give_sugg': ('Fixed parameters:\n{FIXED_PARA}\nPerformance table:\n{TABLE}\nCurrent parameters:\n{PARAMETERS}\n'
                  'Reason step by step inside <think> tags before answering.\n'
                  "End with 'The suggestions are:' and the suggestions."),
    'give_sugg_cot': 'Step 0: compare the rows of the table.\n',
    'change_para': ('Table:\n{TABLE}\nCurrent Configuration:\n{PARAMETERS}\nModification Suggestions:\n{SUGGESTIONS}\n'
                    'Parameter names: {PARAMETER_NAME}\n'
                    "End with 'The modified parameters are:' and KEY: VALUE lines."),
    'loc_error': "Log:\n{LOG}\nParameters:\n{PARAMETERS}\nEnd with 'The conclusion is:' and the cause.",
}


@pytest.fixture
def optico_root(tmp_path):
    root = tmp_path / 'OptiCo-synthetic'
    (root / 'prompt').mkdir(parents=True)
    for name, text in PROMPTS.items():
        (root / 'prompt' / f'{name}.txt').write_text(text, encoding='utf-8')
    (root / 'script' / 'analyzer').mkdir(parents=True)
    (root / 'script' / 'analyzer' / 'analyzer.py').write_text(ANALYZER, encoding='utf-8')
    return root


def test_f_optico_switch_table_is_its_parameter_table(tmp_path, uc, optico_root):
    clock = VirtualClock()
    r = H.FakeRunner(tmp_path / 'run', clock=clock)
    codec = arms.FlashNextHooks(T.make_task('F-OPTICO')).param_codec()
    edits = [{s: codec.fmt(v) for s, v in SMATCH.items()},
             {'retain_attention_prefix': 'false', 'retain_expert_layers': 'true'}]       # experts without prefix: refused
    seen = {'cp': 0, 'gs': []}

    def respond(role, system, prompt):
        if role == 'optico_give_sugg':
            seen['gs'].append(prompt)
            return 'The suggestions are:\nretain more activations'
        if role == 'optico_change_para':
            seen['cp'] += 1
            cur = dict(x.split(': ', 1) for x in prompt.split('Current Configuration:\n')[1].split('\nModification')[0].splitlines())
            cur.update(edits[seen['cp'] - 1] if seen['cp'] <= len(edits) else {})
            return 'The modified parameters are:\n' + '\n'.join(f'{k}: {v}' for k, v in cur.items())
        return 'The conclusion is:\nrejected by the validator'
    drv = ScriptedBackend(respond, clock=clock)
    tr = arms.run_f_optico(r, run_id='fo', out_dir=tmp_path / 'fo', uc_ref=uc, backend=drv, optico_root=optico_root,
                           clock=clock, screens=3)
    assert tr['arm'] == 'F-OPTICO' and [x['point'] for x in r.launched][0] == SMATCH
    assert tr['extra']['deviations'] == oc.DEVIATIONS_HOOKS and tr['extra']['prompt_transform']['lines_dropped'] > 0
    assert tr['invalid_proposals'].get('l0_invalid', 0) >= 1
    rec = next(x for x in _receipts(r.out_dir) if x.get('purpose') == 'screen')
    assert rec['deviations'] == D_F1 + ['D-OC-15'] and rec['valid']
    assert list(r.deviations) == D_F1                                     # restored after the arm
    gs = seen['gs'][0]
    assert 'implementation_stack: native_rollbacks | production | production_qsa_memo_segsum | systems_policy' in gs
    assert '4/1 | 4/2 | 8/2' in gs and T.task_view_leaks(gs) == []


# ---------------------------------------------------------------- F-PROMPTS
def test_f_prompts_one_call_three_recommendations(tmp_path, uc):
    clock = VirtualClock()
    r = H.FakeRunner(tmp_path / 'run', clock=clock)
    drv = ScriptedBackend([{'configs': [
        {'config': {k: v for k, v in SMATCH.items() if v is not None}, 'rationale': 'retain'},
        {'config': {'tp': 4}, 'rationale': 'outside the space'},
        {'config': {'implementation_stack': 'native_rollbacks'}, 'rationale': 'the start itself'}]}], clock=clock)
    tr = arms.run_f_prompts(r, run_id='fp', out_dir=tmp_path / 'fp', uc_ref=uc, backend=drv, clock=clock)
    recs = tr['extra']['prompts']['recommendations']
    assert [x['outcome'] for x in recs] == ['screened', 'refused_invalid', 'refused_duplicate']
    assert len(drv.calls) == 1 and len(tr['screens']) == 1 and tr['selected'] == 'c1' and tr['arm'] == 'F-PROMPTS'
    assert (tr['budget']['screens'], tr['budget']['llm_tokens'], tr['budget']['llm_usd']) == (3, 200000, 10.0)
    prompt = (tmp_path / 'fp' / 'prompts' / 'prompt.txt').read_text(encoding='utf-8')
    assert '# TASK' in prompt and 'implementation_stack' in prompt and '# CURRENT CONFIGURATION: MEASURED SCREEN' in prompt
    assert T.task_view_leaks(prompt) == [] and 'Megatron' not in drv.calls[0]['system']
    assert any(d.startswith('Action space:') for d in tr['extra']['deviations'])


def test_f_prompts_refuses_an_unverified_start_before_any_call(tmp_path, uc):
    drv = ScriptedBackend([{'configs': []}])
    unsigned = {k: v for k, v in uc.items() if k != signing.SIGNATURE_FIELD}
    with pytest.raises(ValueError, match='signature'):
        arms.run_f_prompts(H.FakeRunner(tmp_path), run_id='fp', out_dir=tmp_path / 'fp', uc_ref=unsigned, backend=drv)
    assert drv.calls == []


# ---------------------------------------------------------------- locked pairs (Q2)
@pytest.mark.parametrize('arm,order', [('F-PROMPTS', ['uc', 'candidate']), ('F-OPTICO', ['candidate', 'uc']),
                                       ('F-A0', ['uc', 'candidate']), ('F-RND', ['candidate', 'uc'])])
def test_locked_pair_order_by_arm_index_parity(tmp_path, uc, arm, order):
    clock = VirtualClock()
    tr, r, path = _prompts_trace(tmp_path, uc, clock=clock)
    n0 = len(r.launched)
    res = arms.run_locked(r, arm=arm, run_id='lk', out_dir=tmp_path / 'lk', trace=path, uc_ref=uc)
    assert res['order'] == order and res['status'] == 'ok' and res['locked_valid'] is True and res['invalid_accept'] == 0
    run = r.launched[n0:]
    assert [x['purpose'] for x in run] == ['locked', 'locked']
    assert [x['point'] for x in run] == [START if w == 'uc' else SMATCH for w in order]
    assert res['speedup_vs_uc'] == pytest.approx(res['t_uc_pair'] / res['t_locked'])
    assert res['y_pair'] == pytest.approx(math.log(res['t_uc_pair'] / res['t_locked'])) and res['numerics_class'] == 'E0'
    assert json.loads((tmp_path / 'lk' / 'locked.json').read_text())['arm'] == arm
    locked = [x for x in _receipts(r.out_dir) if x.get('purpose') == 'locked']
    assert all(('D-OC-15' in x['deviations']) == (arm == 'F-OPTICO') for x in locked) and list(r.deviations) == D_F1


def test_no_locked_run_when_the_selection_is_uc(tmp_path, uc):
    tr, r, path = _prompts_trace(tmp_path, uc, point={'implementation_stack': 'native_rollbacks'})
    assert tr['selected'] == 'c0'
    n0 = len(r.launched)
    res = arms.run_locked(r, arm='F-A0', run_id='lk', out_dir=tmp_path / 'lk', trace=path, uc_ref=uc)
    assert res['locked_run'] is False and res['status'] == 'no_improvement' and len(r.launched) == n0
    assert res['y_pair'] == 0.0 and res['invalid_accept'] == 0 and res['t_locked'] == uc['metric']['value']


def test_locked_self_mismatch_makes_the_selection_invalid(tmp_path, uc):
    k = S.key(SMATCH)
    clock = VirtualClock()
    r = H.FakeRunner(tmp_path / 'run', clock=clock, drift={k: {'ce_rel': 0.0015}}, locked_drift={k: {'ce_rel': -0.003}})
    tr, r, path = _prompts_trace(tmp_path, uc, runner=r, clock=clock)
    assert tr['selected'] == 'c1'
    res = arms.run_locked(r, arm='F-A0', run_id='lk', out_dir=tmp_path / 'lk', trace=path, uc_ref=uc)
    assert res['candidate_half']['valid'] is True                       # within the gates against U+C ...
    assert [g['id'] for g in res['guard_rejections']] == ['locked_self_mismatch']   # ... not against its own screen
    assert res['locked_valid'] is False and res['invalid_accept'] == 1 and res['y_pair'] == 0.0 and res['speedup_vs_uc'] == 1.0


def test_void_pair_is_rerun_once_then_reported_without_a_value(tmp_path, uc):
    clock = VirtualClock()
    r = H.FakeRunner(tmp_path / 'run', clock=clock, uc_locked_fail=1)
    tr, r, path = _prompts_trace(tmp_path, uc, runner=r, clock=clock)
    first = arms.run_locked(r, arm='F-RND', run_id='lk', out_dir=tmp_path / 'lk', trace=path, uc_ref=uc)
    assert first['status'] == 'void' and first['t_locked'] is None and 'void_pair' in [g['id'] for g in first['guard_rejections']]
    second = arms.run_locked(r, arm='F-RND', run_id='lk', out_dir=tmp_path / 'lk', trace=path, uc_ref=uc, attempt=2)
    assert second['status'] == 'ok' and (tmp_path / 'lk' / 'locked_attempt2.json').exists()
    assert arms.final_locked(sorted((tmp_path / 'lk').glob('locked*.json')))['attempt'] == 2
    r.uc_locked_fail = 2
    a = arms.run_locked(r, arm='F-RND', run_id='lk2', out_dir=tmp_path / 'lk2', trace=path, uc_ref=uc)
    b = arms.run_locked(r, arm='F-RND', run_id='lk2', out_dir=tmp_path / 'lk2', trace=path, uc_ref=uc, attempt=2)
    assert (a['status'], b['status']) == ('void', 'void_twice') and b['locked_valid'] is None


# ---------------------------------------------------------------- margin_F1 (Q5)
def _uc_half(tmp_path, uc, factor, arm, i):
    r = H.FakeRunner(tmp_path / f'h{i}', base=H.scale_profile(H.profile_of(H.uc_a()), factor))
    return r.run(T.make_task(arm), START, None, 'locked', f'lk{i}', candidate_id='locked-uc',
                 reference_parsed=arms.J.reference_from_receipt(uc))


def test_margin_run_to_run_with_n_ge_4(tmp_path, uc):
    halves = [_uc_half(tmp_path, uc, f, a, i) for i, (f, a) in enumerate([(1.0, 'F-PROMPTS'), (1.03, 'F-OPTICO'),
                                                                          (0.98, 'F-A0'), (1.05, 'F-RND')])]
    cand = halves[0] | {'candidate_id': 'locked-candidate'}
    m = arms.margin([uc] + halves + [H.copy_receipt(cand)])
    vals = [uc['metric']['value']] + [h['metric']['value'] for h in halves]
    rsd = statistics.stdev(vals) / statistics.fmean(vals)
    assert m['estimator'] == 'run_to_run' and m['n'] == 5 and m['rsd'] == pytest.approx(rsd)
    assert m['margin'] == pytest.approx(max(0.03, 3 * rsd)) and m['margin'] > 0.03
    assert {x['kind'] for x in m['S_UC']} == {'uc_reference', 'locked_uc_half'}
    assert arms.margin_for(m, 'F-A0', 'F-OPTICO') == m['margin']


def test_margin_fallback_and_exclusions(tmp_path, uc):
    halves = [_uc_half(tmp_path, uc, 1.0, 'F-A0', 0), _uc_half(tmp_path, uc, 1.02, 'F-OPTICO', 1)]
    unsigned = {k: v for k, v in halves[1].items() if k != signing.SIGNATURE_FIELD}
    m = arms.margin([uc, halves[0], unsigned])
    assert m['estimator'] == 'fallback_within_window' and m['n'] == 2 and m['margin'] is None
    assert m['excluded'][0]['why'] == 'harness signature missing or invalid'
    win = arms.window_rsd(halves[0])
    assert m['per_arm_margin'] == {'F-A0': max(0.03, 3 * win)}
    assert arms.margin_for(m, 'F-A0', 'F-RND') == max(0.03, 3 * win)


# ---------------------------------------------------------------- labels (Q5)
def _row(y, lv=True, ia=0, auc=0.5):
    return {'y_ref': y, 'locked_valid': lv, 'InvalidAccept': ia, 'AUC_budget': auc}


@pytest.mark.parametrize('a,b,want', [(_row(0.20), _row(0.10), 'advantage'), (_row(0.10), _row(0.20), 'disadvantage'),
                                      (_row(0.11), _row(0.10), 'parity'), (_row(0.20, auc=0.1), _row(0.10), 'no_advantage'),
                                      (_row(0.0, lv=False, ia=1), _row(-0.05), 'no_advantage'),
                                      (_row(0.0, lv=False, ia=1), _row(0.10), 'disadvantage'),
                                      (_row(None), _row(0.1), 'invalid')])
def test_label_rule(a, b, want):
    assert arms.label(a, b, 0.03)['label'] == want


def test_label_agrees_with_trajectory_compare_when_both_are_valid():
    task = T.make_task('F-A0')
    for ya, yb, aa, ab in [(0.2, 0.1, 0.5, 0.5), (0.1, 0.2, 0.5, 0.5), (0.11, 0.1, 0.5, 0.5), (0.2, 0.1, 0.1, 0.5)]:
        ra = {'y_locked': ya, 'y_ref': ya, 'locked_valid': True, 'invalid_accept_locked': 0, 'InvalidAccept': 0, 'AUC_budget': aa}
        rb = {'y_locked': yb, 'y_ref': yb, 'locked_valid': True, 'AUC_budget': ab}
        assert arms.label(ra, rb, 0.03)['label'] == TJ.compare(task, ra, rb, rsd=0.0)['label']


# ---------------------------------------------------------------- the summary (Q5 tables)
def test_summary_rows_and_labels(tmp_path, uc):
    clock = VirtualClock()
    r = H.FakeRunner(tmp_path / 'run', clock=clock)
    ht = arms.run_ht(r, run_id='ht', out_dir=tmp_path / 'ht', uc_ref=uc)['receipt']
    assert ht['valid'] and ht['judge']['numerics_class'] == 'E0'
    tr_p, _, path_p = _prompts_trace(tmp_path / 'p', uc, runner=r, clock=clock)
    tr_r = arms.run_f_rnd(r, run_id='frnd', out_dir=tmp_path / 'frnd', uc_ref=uc, clock=clock, screens=4)
    runs = {}
    for arm, tr, path in (('F-PROMPTS', tr_p, path_p), ('F-RND', tr_r, tmp_path / 'frnd' / 'trace.json'),
                          ('F-A0', tr_r, tmp_path / 'frnd' / 'trace.json'), ('F-OPTICO', tr_p, path_p)):
        lk = arms.run_locked(r, arm=arm, run_id=f'lk-{arm}', out_dir=tmp_path / f'lk-{arm}', trace=path, uc_ref=uc)
        runs[arm] = {'trace': tr, 'locked': lk}
    receipts = arms.load_receipts([r.out_dir]) + [uc]
    m = arms.margin(receipts)
    assert m['n'] == 5 and m['estimator'] == 'run_to_run'
    with pytest.raises(ValueError, match='not the Q5 table'):        # F-RND ran 4 screens; F-A0 reuses another trace
        arms.summarize(uc_ref=uc, ht=ht, arms=runs, receipts=receipts)
    s = arms.summarize(uc_ref=uc, ht=ht, arms=runs, receipts=receipts, allow_nonformal=True)
    assert s['formal'] is False and s['nonformal_reasons']
    t_ref, t_ht = uc['metric']['value'], ht['metric']['value']
    for arm, row in s['rows'].items():
        lk = runs[arm]['locked']
        assert row['step_time_locked'] == lk['t_locked'] and row['InvalidAccept'] == 0
        assert row['recovered_fraction'] == pytest.approx(lk['y_pair'] / math.log(t_ref / t_ht))
        assert row['reached_1_30x'] == (lk['speedup_vs_uc'] >= 1.30) and row['numerics_class'] == 'E0'
        assert row['CTG']['reached'] in (True, False) and 0.0 <= row['AUC_budget']
        assert row['deviations'][:6] == D_F1 and row['screen_level'].startswith('screen-level')
    assert set(s['labels']) == {'F-A0 vs F-OPTICO', 'F-A0 vs F-RND', 'F-OPTICO vs F-RND'}
    assert s['F-HT']['line'] == 'best known human recipe on this node set'
    assert s['F-PROMPTS matched prefix']['label'] in ('F-A0 ahead', 'F-PROMPTS ahead', 'parity')
    assert s['switch_space']['admissible_points'] == 353 and s['margin']['estimator'] == 'run_to_run'
    if s['labels']['F-A0 vs F-RND']['label'] != 'advantage':
        assert s['random_control']['F-A0'] == 'no evidence in this space that LLM search beats random'
    assert set(s['drift']['uc_halves_over_ref']) == set(runs)


# ---------------------------------------------------------------- a complete window (Q5 budgets, all four arms, F-HT)
def test_a_complete_window_is_formal_and_the_cli_summary_agrees(tmp_path, uc, a0, optico_root, capsys):
    from infrarsi.ext.flashnext import cli
    clock = VirtualClock()
    w = tmp_path / 'w'
    run = {a: H.FakeRunner(w / d, clock=clock) for a, d in (('F-PROMPTS', 'fp'), ('F-RND', 'frnd'), ('F-A0', 'fa0'),
                                                             ('F-OPTICO', 'fo'), ('F-HT', 'ht'))}
    ht = arms.run_ht(run['F-HT'], run_id='ht', out_dir=w / 'ht', uc_ref=uc)['receipt']
    _prompts_trace(w, uc, runner=run['F-PROMPTS'], clock=clock)
    arms.run_f_rnd(run['F-RND'], run_id='frnd', out_dir=w / 'frnd', uc_ref=uc, clock=clock)
    script = [{'action': 'propose_config', 'base': 'c0', 'changes': {k: v for k, v in SMATCH.items() if v is not None},
               'rationale': 'retain'}, {'action': 'run_screen', 'candidate_id': 'c1'},
              {'action': 'finish', 'candidate_id': 'c1', 'reason': 'done'}]
    drv = ScriptedBackend(lambda role, s, p: PASS if role == 'verifier' else script.pop(0), clock=clock)
    arms.run_f_a0(run['F-A0'], run_id='fa0', out_dir=w / 'fa0', uc_ref=uc, package=a0, backend=drv, clock=clock)
    codec = arms.FlashNextHooks(T.make_task('F-OPTICO')).param_codec()
    first = {s: codec.fmt(v) for s, v in SMATCH.items()}

    def respond(role, system, prompt):
        if role == 'optico_give_sugg':
            return 'The suggestions are:\nretain more activations'
        if role == 'optico_change_para':
            cur = dict(x.split(': ', 1) for x in prompt.split('Current Configuration:\n')[1].split('\nModification')[0].splitlines())
            return 'The modified parameters are:\n' + '\n'.join(f'{k}: {v}' for k, v in {**cur, **first}.items())
        return 'The conclusion is:\nx'
    arms.run_f_optico(run['F-OPTICO'], run_id='fo', out_dir=w / 'fo', uc_ref=uc, backend=ScriptedBackend(respond, clock=clock),
                      optico_root=optico_root, clock=clock)
    window = {}
    for arm, d in (('F-PROMPTS', 'fp'), ('F-RND', 'frnd'), ('F-A0', 'fa0'), ('F-OPTICO', 'fo')):
        arms.run_locked(run[arm], arm=arm, run_id=f'lk-{d}', out_dir=w / d, trace=w / d / 'trace.json', uc_ref=uc)
        window[arm] = {'trace': json.loads((w / d / 'trace.json').read_text()),
                       'locked': arms.final_locked(sorted((w / d).glob('locked*.json')))}
    receipts = arms.load_receipts([w / d for d in ('fp', 'frnd', 'fa0', 'fo', 'ht')]) + [uc]
    s = arms.summarize(uc_ref=uc, ht=ht, arms=window, receipts=receipts)
    assert s['formal'] is True and s['nonformal_reasons'] == []
    assert s['margin']['n'] == 5 and s['margin']['margin'] == arms.margin(receipts)['margin']
    assert set(s['rows']) == set(arms.ARM_INDEX) and all(r['InvalidAccept'] == 0 for r in s['rows'].values())
    (w / 'uc_ref.json').write_text(json.dumps(uc))
    base = ['summary', '--uc-ref', str(w / 'uc_ref.json'), '--ht', str(w / 'ht' / 'ht.json'), '--out', str(w)] + sum(
        (['--arm', f'{a}={w / d}'] for a, d in (('F-PROMPTS', 'fp'), ('F-RND', 'frnd'), ('F-A0', 'fa0'), ('F-OPTICO', 'fo'))), [])
    assert cli.main(base) == 0
    capsys.readouterr()
    written = json.loads((w / 'summary_F1.json').read_text())
    assert written['formal'] is True and written['labels'] == s['labels']
    assert cli.main(['margin', '--runs', str(w), '--out', str(w)]) == 0
    capsys.readouterr()
    assert cli.main(base + ['--margin', str(w / 'margin_F1.json')]) == 0          # the margin file the window gives
    capsys.readouterr()
    doctored = dict(json.loads((w / 'margin_F1.json').read_text()), margin=0.5)
    (w / 'doctored.json').write_text(json.dumps(doctored))
    assert cli.main(base + ['--margin', str(w / 'doctored.json')]) == 4           # refused: not what the receipts give
    assert 'margin file' in capsys.readouterr().out
