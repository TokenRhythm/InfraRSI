"""Requirements in the user's own words (project/requirements.py): compiled once, checked item by item, narrowing only."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from infrarsi.agent.backends.fake import ScriptedBackend
from infrarsi.cli.main import main
from infrarsi.project import Project
from infrarsi.project import llm as llm_mod
from infrarsi.project import requirements as rq
from infrarsi.project.flow import run_project
from infrarsi.project.hooks import ProjectHooks
from infrarsi.project.manifest import ManifestError
from infrarsi.search.knob_space import choices

from test_project import toy  # noqa: F401 - the toy project fixture


def model(items, *, name='scripted-model'):
    """A model that answers the compile call with `items` (and any other call with an empty JSON object)."""
    return ScriptedBackend(lambda role, system, prompt: {'items': items} if role == 'requirements' else {}, model=name, tokens_per_call=500)


def item(kind, text='x', **kw):
    return {'text': text, 'kind': kind, 'why': 'because', **kw}


def compiled(project, items, texts=('a requirement',), **kw):
    return rq.compile(project, list(texts), driver=model(items), **kw)


def statuses(rec):
    return [i['status'] for i in rec['items']]


# ---------------------------------------------------------------- nothing given, nothing changed
def test_without_requirements_the_manifest_is_untouched(toy):
    assert 'requirements' not in toy.raw


def test_without_requirements_the_prompts_are_untouched(toy):
    hooks = ProjectHooks(toy, toy.task())
    view = hooks.task_view()
    assert 'requirements' not in view and not hooks.requirements
    assert 'User requirements' not in hooks.contract_text()


def test_without_requirements_the_result_is_untouched(toy, tmp_path):
    res = run_project(toy, strategy='random', budget={'screens': 1}, out_root=tmp_path / 'out', log=lambda m: None)
    run = Path(res['run_dir'])
    assert 'requirements' not in res and not any(k.startswith('requirements') for k in res['cost'])
    assert '## Requirements' not in (run / 'REPORT.md').read_text(encoding='utf-8') and not (run / 'requirements.json').exists()


def test_an_empty_text_is_no_requirement(toy):
    assert rq.compile(toy, ['  ', ''], driver=None) is None


# ---------------------------------------------------------------- compile and check
def test_pin_knob(toy):
    rec = compiled(toy, [item('pin_knob', knob='precision')])
    assert statuses(rec) == ['enforced'] and rec['overlay']['knobs'] == {'precision': {'choices': ['fp32']}}
    p = rq.apply(toy, rec)
    assert p.knobs['precision']['choices'] == ['fp32'] and choices(p.task())['precision'] == ['fp32']
    hooks = ProjectHooks(p, p.task())
    assert hooks.validate_config({'precision': 'bf16'})[1]


def test_restrict_values(toy):
    rec = compiled(toy, [item('restrict_values', knob='micro_batch_size', allowed=[1, '2', 4])])
    assert statuses(rec) == ['enforced'] and rec['overlay']['knobs']['micro_batch_size']['choices'] == [1, 2, 4]


def test_a_requirement_cannot_exclude_the_current_value(toy):
    rec = compiled(toy, [item('restrict_values', knob='precision', allowed=['bf16'])])
    assert statuses(rec) == ['conflict'] and rec['overlay']['knobs'] == {}
    assert 'cannot change the reference' in rec['items'][0]['words']


def test_only_knobs_pins_the_others(toy):
    rec = compiled(toy, [item('only_knobs', knobs=['workers'])])
    assert statuses(rec) == ['enforced']
    assert set(rec['overlay']['knobs']) == set(toy.knobs) - {'workers'}
    p = rq.apply(toy, rec)
    assert all(len(p.knobs[k]['choices']) == 1 for k in rec['overlay']['knobs'])
    assert len(p.knobs['workers']['choices']) > 1


@pytest.mark.parametrize('expr,status', [('micro_batch_size <= 4', 'enforced'), ('micro_batch_size > 4', 'conflict'),
                                         ('nonsense_knob < 2', 'invalid'), ('micro_batch_size <', 'invalid')])
def test_constraint(toy, expr, status):
    rec = compiled(toy, [item('constraint', expr=expr)])
    assert statuses(rec) == [status]


def test_freeze_file(toy):
    rec = compiled(toy, [item('freeze_file', path='kernels.py')])
    assert statuses(rec) == ['enforced']
    p = rq.apply(toy, rec)
    assert 'kernels.py' in p.raw['code']['frozen'] and p.files(p.raw['code']['mutable']) == []
    assert 'kernels.py' not in ProjectHooks(p, p.task()).base_files


def test_freeze_file_already_guaranteed(toy):
    assert statuses(compiled(toy, [item('freeze_file', path='loss.py')])) == ['already']


@pytest.mark.parametrize('path', ['../outside.py', 'no_such_file.py', '*.py', 'a\\b.py'])
def test_freeze_file_invalid(toy, path):
    assert statuses(compiled(toy, [item('freeze_file', path=path)])) == ['invalid']


def test_only_files(toy):
    rec = compiled(toy, [item('only_files', paths=[])])
    assert statuses(rec) == ['enforced'] and rec['overlay']['code']['mutable'] == []
    assert rq.apply(toy, rec).raw['code']['mutable'] == []


def test_only_files_keeps_what_is_mutable_and_says_what_it_ignored(toy):
    rec = compiled(toy, [item('only_files', paths=['kernels.py', 'train.py'])])
    assert statuses(rec) == ['already'] and 'train.py' in rec['items'][0]['words']


def test_max_patch_lines(toy):
    assert statuses(compiled(toy, [item('max_patch_lines', n=50)])) == ['enforced']
    assert statuses(compiled(toy, [item('max_patch_lines', n=5000)], texts=['b'])) == ['already']


def test_require_check(toy):
    rec = compiled(toy, [item('require_check', check='no_errors')])
    assert statuses(rec) == ['enforced']
    assert 'no_errors' in rq.apply(toy, rec).raw['verify']['required']
    assert statuses(compiled(toy, [item('require_check', check='completes')], texts=['b'])) == ['already']
    assert statuses(compiled(toy, [item('require_check', check='nonsense')], texts=['c'])) == ['invalid']
    assert statuses(compiled(toy, [item('require_check', check='precision')], texts=['d'])) == ['invalid']   # (reported only)


def test_a_disabled_check_is_not_turned_on(toy):
    toy.raw['verify']['disabled'] = ['no_errors']
    assert statuses(compiled(toy, [item('require_check', check='no_errors')])) == ['widening']


def test_advisory_reaches_the_prompts(toy):
    rec = compiled(toy, [item('advisory', advice='Do not switch to bf16.')])
    assert statuses(rec) == ['advisory'] and rec['overlay']['knobs'] == {}
    hooks = ProjectHooks(toy, toy.task(), requirements=rec['guidance'])
    assert 'Do not switch to bf16.' in hooks.contract_text() and hooks.task_view()['requirements'] == rec['guidance']
    assert 'Do not switch to bf16.' in hooks.fixed_parameters_text()


@pytest.mark.parametrize('kind', ['setting', 'widening', 'objective', 'unclear'])
def test_what_cannot_be_honoured_blocks(toy, kind):
    rec = compiled(toy, [item(kind, flag='--tolerance')])
    assert statuses(rec) == [kind] and rec['overlay']['knobs'] == {} and len(rq.blocking(rec)) == 1 and rq.guidance(rec) == []


def test_an_objective_change_is_reported_as_not_supported(toy):
    rec = compiled(toy, [item('objective', text='省钱')])
    assert 'one objective' in rec['items'][0]['words']


def test_a_tolerance_request_is_a_setting_and_changes_nothing(toy):
    rec = compiled(toy, [item('setting', text='把容差放宽到 10%', flag='--tolerance / verify.tolerance')])
    assert '--tolerance' in rec['items'][0]['words'] and rq.apply(toy, rec).tolerance == toy.tolerance


def test_two_items_on_one_knob_intersect(toy):
    rec = compiled(toy, [item('restrict_values', knob='micro_batch_size', allowed=[1, 2]),
                         item('restrict_values', knob='micro_batch_size', allowed=[1, 4])])
    assert statuses(rec) == ['enforced', 'enforced'] and rec['overlay']['knobs']['micro_batch_size']['choices'] == [1]


def test_a_restriction_that_leaves_out_the_current_value_is_a_conflict_of_its_own(toy):
    a = item('restrict_values', knob='grad_accum', allowed=[8, 4])
    b = item('restrict_values', knob='grad_accum', allowed=[2, 4])     # (grad_accum's current value, 8, is not in b)
    rec = compiled(toy, [a, b])
    assert statuses(rec) == ['enforced', 'conflict'] and rec['overlay']['knobs']['grad_accum']['choices'] == [4, 8]


def _ranged(toy, lo=0, hi=10, ref=5):
    k = toy.raw['knobs']['workers']
    k.pop('choices', None)
    k.update(type='int', min=lo, max=hi, default=ref)
    return toy


def test_two_ranges_on_one_knob_intersect(toy):
    p = _ranged(toy)
    rec = compiled(p, [item('restrict_values', knob='workers', min=2, max=6), item('restrict_values', knob='workers', min=4, max=8)])
    assert statuses(rec) == ['enforced', 'enforced'] and rec['overlay']['knobs']['workers'] == {'min': 4, 'max': 6}
    assert ProjectHooks(rq.apply(p, rec), rq.apply(p, rec).task()).validate_config({'workers': 8})[1]


def test_a_range_on_an_unset_knob_is_not_a_crash(toy):
    p = _ranged(toy, ref=None)
    p.raw['knobs']['workers'].pop('default')
    assert statuses(compiled(p, [item('restrict_values', knob='workers', min=2, max=6)])) == ['conflict']


def test_a_null_proposal_cannot_undo_a_pin(toy):
    rec = compiled(toy, [item('pin_knob', knob='workers')])
    p = rq.apply(toy, rec)
    assert ProjectHooks(p, p.task(), requirements=rec['guidance']).validate_config({'workers': None})[1]


def test_a_null_proposal_cannot_leave_a_narrowed_range(toy):
    p = _ranged(toy)
    rec = compiled(p, [item('restrict_values', knob='workers', min=4, max=6)])
    p2 = rq.apply(p, rec)
    assert ProjectHooks(p2, p2.task(), requirements=rec['guidance']).validate_config({'workers': None})[1]


def _hooks(p, rec):
    return ProjectHooks(p, p.task(), requirements=rec['guidance'])


def test_a_null_proposal_cannot_undo_a_pin_that_held_already(toy):
    toy.raw['knobs']['workers']['choices'] = [1]
    rec = compiled(toy, [item('pin_knob', knob='workers')])
    assert statuses(rec) == ['already'] and rec['overlay']['hold'] == ['workers']
    p = rq.apply(toy, rec)
    assert p.knobs['workers']['choices'] == [1] and _hooks(p, rec).validate_config({'workers': None})[1]


def test_a_null_proposal_cannot_leave_only_knobs_through_a_singleton(toy):
    toy.raw['knobs']['workers']['choices'] = [1]
    rec = compiled(toy, [item('only_knobs', knobs=['micro_batch_size', 'grad_accum'])])
    assert statuses(rec) == ['enforced'] and rec['overlay']['hold'] == ['workers']
    assert _hooks(rq.apply(toy, rec), rec).validate_config({'workers': None})[1]


def test_a_null_proposal_cannot_leave_values_that_were_all_it_had(toy):
    rec = compiled(toy, [item('restrict_values', knob='workers', allowed=[1, 2, 4])])
    assert statuses(rec) == ['already']
    assert _hooks(rq.apply(toy, rec), rec).validate_config({'workers': None})[1]


def test_a_null_proposal_cannot_leave_a_range_that_was_all_it_had(toy):
    p = _ranged(toy)
    rec = compiled(p, [item('restrict_values', knob='workers', min=0, max=10)])
    assert statuses(rec) == ['already']
    assert _hooks(rq.apply(p, rec), rec).validate_config({'workers': None})[1]


@pytest.mark.parametrize('it', [item('pin_knob', knob='workers'), item('only_knobs', knobs=['micro_batch_size', 'grad_accum'])])
def test_a_pinned_unset_knob_with_one_value_cannot_be_set_to_it(toy, it):
    toy.raw['knobs']['workers'] = {'type': 'int', 'choices': [5], 'flag': '--workers'}
    rec = compiled(toy, [it])
    assert statuses(rec) == ['enforced'] and rec['overlay']['knobs']['workers'] == {'choices': [None]}
    hooks = _hooks(rq.apply(toy, rec), rec)
    assert hooks.validate_config({'workers': 5})[1] and not hooks.validate_config({'workers': None})[1]


def test_asking_for_both_values_of_an_on_off_knob_marks_nothing(toy):
    rec = compiled(toy, [item('restrict_values', knob='compile', allowed=[True, False])])
    assert statuses(rec) == ['already'] and not rec['overlay']['hold']
    assert not rq.apply(toy, rec).knobs['compile'].get('narrowed')


def test_guidance_alone_leaves_an_unset_knob_valid(toy):
    toy.raw['knobs']['workers'] = {'type': 'int', 'choices': [1, 2], 'override': 'train.not_in_the_config'}
    rec = compiled(toy, [item('advisory', advice='prefer low memory')])
    p = rq.apply(toy, rec)
    assert not ProjectHooks(p, p.task(), requirements=rec['guidance']).validate_config({'workers': None})[1]


def test_without_requirements_a_boolean_counts_for_the_strategy_as_before(toy):
    from infrarsi import search
    knobs = {f'b{i}': {'type': 'bool'} for i in range(3)}
    knobs['pinned'] = {'type': 'bool', 'choices': [False]}
    assert search.resolve('auto', has_llm=True, knobs=knobs) == 'doe+hls'
    knobs['pinned']['narrowed'] = True
    assert search.resolve('auto', has_llm=True, knobs=knobs) == 'hls'


def test_a_pinned_unset_knob_still_accepts_being_unset(toy):
    toy.raw['knobs']['workers'] = {'type': 'int', 'choices': [1, 2], 'override': 'train.not_in_the_config'}
    rec = compiled(toy, [item('pin_knob', knob='workers')])
    assert rec['overlay']['knobs']['workers'] == {'choices': [None]}
    p = rq.apply(toy, rec)
    assert not ProjectHooks(p, p.task(), requirements=rec['guidance']).validate_config({'workers': None})[1]
    assert ProjectHooks(p, p.task(), requirements=rec['guidance']).validate_config({'workers': 2})[1]


def test_only_files_replaces_a_glob_even_when_it_matches_the_same_files(toy):
    toy.raw['code']['mutable'] = ['*.py']
    rec = compiled(toy, [item('only_files', paths=['kernels.py'])])
    assert statuses(rec) == ['enforced'] and rq.apply(toy, rec).raw['code']['mutable'] == ['kernels.py']


def test_a_saved_overlay_cannot_widen_a_changed_range(toy):
    p = _ranged(toy)
    rec = {'digest': 'x', 'overlay': {'knobs': {'workers': {'choices': [5, 99]}}, 'constraints': [], 'code': {}, 'verify': {}}}
    with pytest.raises(rq.RequirementsError, match='widen'):
        rq.apply(p, rec)


def test_a_saved_required_check_that_is_now_disabled_is_refused(toy):
    toy.raw['verify']['disabled'] = ['same_data']
    rec = {'digest': 'x', 'overlay': {'knobs': {}, 'constraints': [], 'code': {}, 'verify': {'required': ['same_data']}}}
    with pytest.raises(rq.RequirementsError, match='disabled'):
        rq.apply(toy, rec)


def test_a_pinned_boolean_is_not_a_factor(toy):
    from infrarsi.search.proposers import LLMProposer
    rec = compiled(toy, [item('pin_knob', knob='compile')])
    p = rq.apply(toy, rec)
    task = p.task()
    prop = LLMProposer(None, ProjectHooks(p, task))
    engine = SimpleNamespace(s=SimpleNamespace(task=task, candidates={'c0': SimpleNamespace(config=p.reference_config())}))
    assert 'compile' not in [f['name'] for f in prop.factors(engine, 'c0')]


def test_more_than_sixteen_items_is_a_reply_to_ask_again(toy):
    many = [item('advisory', advice='x')] * 17
    drv = ScriptedBackend([{'items': many}, {'items': many[:3]}], tokens_per_call=100)
    rec = rq.compile(toy, ['a'], driver=drv)
    assert len(rec['items']) == 3 and rec['compile']['calls'] == 2
    with pytest.raises(rq.RequirementsError):
        rq.compile(toy, ['b'], driver=ScriptedBackend([{'items': many}, {'items': many}], tokens_per_call=100))


def test_a_rebuilt_finish_keeps_the_requirements_record(toy, tmp_path, monkeypatch):
    from infrarsi.project import finish as finish_mod
    rec = compiled(toy, [item('advisory', advice='careful')])
    res = run_project(rq.apply(toy, rec), strategy='random', budget={'screens': 1}, out_root=tmp_path / 'out', log=lambda m: None, requirements=rec)
    run = Path(res['run_dir'])
    (run / 'result.json').unlink()
    (run / 'REPORT.md').unlink()
    monkeypatch.setattr(finish_mod, 'promote', lambda *a, **k: {'promoted': None, 'tried': [], 'gpu_hours': 0.0, 'ref_offsets': [], 'why': 'scripted'})
    done = finish_mod.finish_run(run, project=Project.load(toy.root), log=lambda m: None)
    assert done['requirements']['digest'] == rec['digest'] and '## Requirements' in (run / 'REPORT.md').read_text(encoding='utf-8')


def test_the_cache_key_covers_what_the_model_reads(toy):
    a = rq.digest(toy, ['x'])
    toy.raw['knobs']['workers']['description'] = 'dataloader workers'
    assert rq.digest(toy, ['x']) != a
    b = rq.digest(toy, ['x'])
    toy.raw['knobs']['workers']['mechanism'] = 'dp_grad_overlap'
    assert rq.digest(toy, ['x']) != b


def test_finish_keeps_the_runs_requirements(toy, tmp_path, monkeypatch):
    from infrarsi.project import finish as finish_mod
    rec = compiled(toy, [item('require_check', check='user_tests')])
    res = run_project(rq.apply(toy, rec), strategy='random', budget={'screens': 1}, out_root=tmp_path / 'out', log=lambda m: None, requirements=rec)
    run = Path(res['run_dir'])
    doc = json.loads((run / 'result.json').read_text(encoding='utf-8'))
    doc.update(outcome='interrupted', promotion={'promoted': None, 'tried': []})
    (run / 'result.json').write_text(json.dumps(doc), encoding='utf-8')
    seen = {}

    def fake(project, task, runner, verifier, trace, ref, run_id, receipts, **kw):
        seen['required'] = set(verifier.required)
        return {'promoted': None, 'tried': [], 'gpu_hours': 0.0, 'ref_offsets': [], 'why': 'scripted'}
    monkeypatch.setattr(finish_mod, 'promote', fake)
    finish_mod.finish_run(run, project=Project.load(toy.root), log=lambda m: None)
    assert 'user_tests' in seen['required']


def test_a_value_that_does_not_exist_is_invalid(toy):
    rec = compiled(toy, [item('restrict_values', knob='micro_batch_size', allowed=[1, 3])])
    assert statuses(rec) == ['invalid'] and rec['overlay']['knobs'] == {}


def test_the_model_cannot_name_what_does_not_exist(toy):
    rec = compiled(toy, [item('pin_knob', knob='no_such_knob'), item('only_knobs', knobs=['workers', 'nope']), item('require_check', check='x')])
    assert statuses(rec) == ['invalid', 'invalid', 'invalid']


def test_a_reply_with_no_item_is_not_silently_accepted(toy):
    rec = compiled(toy, [])
    assert statuses(rec) == ['unclear'] and rq.blocking(rec)


# ---------------------------------------------------------------- the model answering badly
def test_a_bad_reply_is_asked_once_more(toy):
    replies = ['not json', {'items': [item('advisory', advice='x')]}]
    drv = ScriptedBackend(list(replies), tokens_per_call=100)
    rec = rq.compile(toy, ['a'], driver=drv)
    assert statuses(rec) == ['advisory'] and len(drv.calls) == 2 and rec['compile']['calls'] == 2
    assert 'previous_reply_error' in drv.calls[1]['prompt']


def test_two_bad_replies_stop(toy):
    with pytest.raises(rq.RequirementsError, match='could not read'):
        rq.compile(toy, ['a'], driver=ScriptedBackend(['bad', 'bad']))


def test_no_model_and_no_cache_stops_and_says_how_to_write_them(toy):
    with pytest.raises(rq.RequirementsError, match='code.frozen'):
        rq.compile(toy, ['a'], driver=None)


# ---------------------------------------------------------------- the cache
def test_the_second_compile_is_free(toy):
    drv = model([item('advisory', advice='x')])
    first = rq.compile(toy, ['a'], driver=drv)
    second = rq.compile(toy, ['a'], driver=drv)
    assert len(drv.calls) == 1 and first['compile']['cached'] is False and second['compile']['cached'] is True
    assert rq.compile(toy, ['a'], driver=None)['compile']['cached'] is True   # no model needed now


def test_the_cache_key_follows_the_project_not_the_model(toy, tmp_path):
    a = rq.digest(toy, ['x'])
    drv1, drv2 = model([item('advisory', advice='x')], name='one'), model([item('advisory', advice='y')], name='two')
    rq.compile(toy, ['x'], driver=drv1)
    assert rq.compile(toy, ['x'], driver=drv2)['compile']['cached'] is True and not drv2.calls
    assert rq.digest(toy, ['y']) != a
    assert rq.digest(toy, [' x  ']) == a                # (spaces do not change a requirement)
    toy.raw['knobs']['workers']['choices'] = [1, 2]
    assert rq.digest(toy, ['x']) != a


def test_a_broken_cache_is_compiled_again(toy):
    drv = model([item('advisory', advice='x')])
    rq.compile(toy, ['a'], driver=drv)
    (toy.root / rq.CACHE_DIR / f'{rq.digest(toy, ["a"])}.json').write_text('garbage', encoding='utf-8')
    assert rq.compile(toy, ['a'], driver=drv)['compile']['cached'] is False


def test_refresh_compiles_again(toy):
    drv = model([item('advisory', advice='x')])
    rq.compile(toy, ['a'], driver=drv)
    rq.compile(toy, ['a'], driver=drv, refresh=True)
    assert len(drv.calls) == 2


# ---------------------------------------------------------------- only narrowing
@pytest.mark.parametrize('overlay', [
    {'knobs': {'precision': {'choices': ['fp32', 'something-else']}}},   # a value the knob does not have
    {'knobs': {'precision': {'choices': ['bf16']}}},                     # the current value left out
    {'code': {'mutable': ['train.py']}},                                 # a file that was not mutable
    {'code': {'max_patch_lines': 100000}}])
def test_an_overlay_that_widens_is_refused(toy, overlay):
    rec = {'digest': 'x', 'overlay': {'knobs': {}, 'constraints': [], 'code': {}, 'verify': {}, **overlay}}
    with pytest.raises(rq.RequirementsError, match='widen'):
        rq.apply(toy, rec)


def test_the_overlay_has_no_way_to_touch_the_verifier(toy):
    rec = compiled(toy, [item('pin_knob', knob='precision')])
    p = rq.apply(toy, rec)
    assert p.tolerance == toy.tolerance and p.raw['verify']['disabled'] == toy.raw['verify']['disabled']
    assert p.raw['workload'] == toy.raw['workload'] and p.raw['launch'] == toy.raw['launch']


def test_apply_leaves_the_original_alone(toy):
    before = json.dumps(toy.raw, sort_keys=True, default=str)
    rq.apply(toy, compiled(toy, [item('pin_knob', knob='precision')]))
    assert json.dumps(toy.raw, sort_keys=True, default=str) == before


# ---------------------------------------------------------------- deciding
def test_nothing_blocking_goes_on(toy):
    rec = compiled(toy, [item('pin_knob', knob='precision')])
    assert rq.decide(rec)['accepted_partial'] is False


def test_blocking_items_stop_a_non_interactive_run(toy):
    rec = compiled(toy, [item('objective')])
    with pytest.raises(rq.RequirementsError, match='nothing was run'):
        rq.decide(rec)
    assert rq.decide(rec, partial=True)['accepted_partial'] is True


@pytest.mark.parametrize('choice,expect', [('go', 'ok'), ('skip', 'ok'), ('edit', None), ('quit', 'stop')])
def test_interactive_choices(toy, choice, expect):
    rec = compiled(toy, [item('pin_knob', knob='precision')])
    if expect == 'stop':
        with pytest.raises(rq.RequirementsError):
            rq.decide(rec, interactive=lambda r: choice)
    else:
        got = rq.decide(rec, interactive=lambda r: choice)
        assert (got is None) if expect is None else (got is not None)


def test_enter_never_drops_a_requirement(toy):
    rec = compiled(toy, [item('objective')])
    with pytest.raises(rq.RequirementsError):
        rq.decide(rec, interactive=lambda r: 'go')       # (Enter answers 'go' to the question)
    assert rq.decide(rec, interactive=lambda r: 'skip')['accepted_partial'] is True


def test_explain_says_what_became_of_each(toy):
    rec = compiled(toy, [item('pin_knob', 'keep precision', knob='precision'), item('advisory', 'be careful', advice='Careful.'),
                         item('objective', 'save money')])
    lines = rq.explain(rec)
    assert '1 enforced' in lines[0] and '1 guidance only' in lines[0] and '1 cannot be honoured' in lines[0]
    assert 'enforced' in lines[1] and 'guidance only' in lines[2] and 'not supported' in lines[3]


# ---------------------------------------------------------------- the manifest block
@pytest.mark.parametrize('block,msg', [('nope', 'mapping'), ({'text': ''}, 'non-empty'), ({'text': ['a', 3]}, 'non-empty'),
                                       ({'text': 'a', 'extra': 1}, 'unknown key'), ({'text': 'a', 'on_unsupported': 'x'}, 'on_unsupported')])
def test_the_block_is_checked(toy, block, msg):
    doc = yaml.safe_load((toy.root / 'infrarsi.yaml').read_text(encoding='utf-8'))
    doc['requirements'] = block
    (toy.root / 'infrarsi.yaml').write_text(yaml.safe_dump(doc), encoding='utf-8')
    with pytest.raises(ManifestError, match=msg):
        Project.load(toy.root)


def test_the_text_may_be_a_string_or_a_list(toy):
    toy.raw['requirements'] = {'text': 'one'}
    assert rq.manifest_texts(toy) == ['one']
    toy.raw['requirements'] = {'text': ['a', 'b']}
    assert rq.manifest_texts(toy) == ['a', 'b']


def test_append_texts_keeps_a_manifest_and_refuses_a_second_block(toy):
    path = toy.root / 'infrarsi.yaml'
    before = path.read_text(encoding='utf-8')
    rq.append_texts(path, ['不要动 loss'])
    assert path.read_text(encoding='utf-8').startswith(before)
    assert Project.load(toy.root).raw['requirements']['text'] == ['不要动 loss']
    with pytest.raises(rq.RequirementsError, match='already'):
        rq.append_texts(path, ['more'])


def test_too_long_text_is_refused():
    with pytest.raises(ValueError, match='characters'):
        rq.normalize(['x' * 1001])
    with pytest.raises(ValueError, match='characters'):
        rq.normalize(['x' * 900] * 6)


# ---------------------------------------------------------------- a whole run
def test_a_run_with_requirements(toy, tmp_path):
    rec = {**compiled(toy, [item('pin_knob', '不能改精度', knob='precision'), item('advisory', '小心', advice='Be careful.')]), 'source': {'cli': ['x']}}
    p = rq.apply(toy, rec)
    res = run_project(p, strategy='random', budget={'screens': 3}, out_root=tmp_path / 'out', log=lambda m: None, requirements=rec)
    run = Path(res['run_dir'])
    assert res['requirements']['digest'] == rec['digest'] and (run / 'requirements.json').is_file()
    assert res['cost']['requirements_llm_tokens'] == 500
    assert all(c.get('change') is None or 'precision' not in (c['change'] or {}) for c in res['candidates'])
    text = (run / 'REPORT.md').read_text(encoding='utf-8')
    assert '## Requirements' in text and '不能改精度' in text and 'guidance only' in text and 'outside the search budget' in text


def test_an_early_exit_carries_the_requirements(toy, tmp_path, monkeypatch):
    from infrarsi.project import runner as runner_mod
    rec = compiled(toy, [item('pin_knob', knob='precision')])

    def fail(self, *a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(runner_mod.ProjectRunner, 'run', fail)
    with pytest.raises(KeyboardInterrupt):
        run_project(rq.apply(toy, rec), strategy='random', budget={'screens': 2}, out_root=tmp_path / 'out', log=lambda m: None, requirements=rec)
    run = next((tmp_path / 'out').iterdir())
    assert json.loads((run / 'requirements.json').read_text(encoding='utf-8'))['digest'] == rec['digest']


def _fake_llm(monkeypatch, items):
    made = []

    def make(spec, **kw):
        s = llm_mod.parse(spec)
        if not s or s.get('backend') == 'none':
            return None
        d = model(items)
        made.append(d)
        return d
    monkeypatch.setattr(llm_mod, 'make_driver', make)
    return made


def test_the_command_line_compiles_applies_and_records(toy, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    made = _fake_llm(monkeypatch, [item('pin_knob', 'keep', knob='precision')])
    main(['run', str(toy.root), '--llm', 'fake:m', '--strategy', 'random', '--screens', '2', '--requirements', 'keep the precision',
          '--out', str(tmp_path / 'runs')])
    out = capsys.readouterr().out
    assert '[infrarsi] requirements: 1 item(s) from --requirements (1)' in out and 'enforced' in out
    run = next((tmp_path / 'runs').iterdir())
    r = json.loads((run / 'result.json').read_text(encoding='utf-8'))
    assert r['requirements']['source']['cli'] == ['keep the precision'] and r['project']['knobs']
    assert sum(1 for c in made[0].calls if c['role'] == 'requirements') == 1


def test_a_blocking_item_stops_before_any_run(toy, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    _fake_llm(monkeypatch, [item('objective', 'save money')])
    with pytest.raises(SystemExit) as e:
        main(['run', str(toy.root), '--llm', 'fake:m', '--strategy', 'random', '--screens', '2', '--requirements', 'save money',
              '--out', str(tmp_path / 'runs')])
    assert e.value.code == 2 and 'nothing was run' in capsys.readouterr().err and not (tmp_path / 'runs').exists()


def test_partial_goes_on_and_says_so(toy, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    _fake_llm(monkeypatch, [item('objective', 'save money'), item('pin_knob', 'keep', knob='precision')])
    main(['run', str(toy.root), '--llm', 'fake:m', '--strategy', 'random', '--screens', '2', '--requirements', 'x', '--requirements-partial',
          '--out', str(tmp_path / 'runs')])
    run = next((tmp_path / 'runs').iterdir())
    r = json.loads((run / 'result.json').read_text(encoding='utf-8'))
    assert r['requirements']['accepted_partial'] is True
    assert 'went on without item(s) 1' in (run / 'REPORT.md').read_text(encoding='utf-8')


def test_no_requirements_ignores_the_manifest_block(toy, tmp_path, monkeypatch):
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    doc = yaml.safe_load((toy.root / 'infrarsi.yaml').read_text(encoding='utf-8'))
    doc['requirements'] = {'text': 'keep the precision'}
    (toy.root / 'infrarsi.yaml').write_text(yaml.safe_dump(doc), encoding='utf-8')
    made = _fake_llm(monkeypatch, [item('pin_knob', knob='precision')])
    main(['run', str(toy.root), '--llm', 'fake:m', '--strategy', 'random', '--screens', '1', '--no-requirements', '--out', str(tmp_path / 'runs')])
    r = json.loads((next((tmp_path / 'runs').iterdir()) / 'result.json').read_text(encoding='utf-8'))
    assert 'requirements' not in r and not any(c['role'] == 'requirements' for d in made for c in d.calls)


def test_requirements_need_one_project(tmp_path, capsys):
    suite = tmp_path / 'suite.yaml'
    suite.write_text('schema: infrarsi-suite-v1\nprojects: [a/infrarsi.yaml]\n', encoding='utf-8')
    with pytest.raises(SystemExit) as e:
        main(['run', str(suite), '--requirements', 'x'])
    assert e.value.code == 2 and 'one project' in capsys.readouterr().err


@pytest.mark.parametrize('flag', [['--requirements', 'x'], ['--requirements-partial'], ['--no-requirements'], ['--requirements-file', 'f.txt']])
def test_a_resumed_run_takes_no_new_requirements(tmp_path, capsys, flag):
    with pytest.raises(SystemExit) as e:
        main(['run', '--resume', str(tmp_path / 'run'), *flag])
    assert e.value.code == 2 and 'new requirements need a new run' in capsys.readouterr().err


def test_partial_needs_one_project_too(tmp_path, capsys):
    suite = tmp_path / 'suite.yaml'
    suite.write_text('schema: infrarsi-suite-v1\nprojects: [a/infrarsi.yaml]\n', encoding='utf-8')
    with pytest.raises(SystemExit) as e:
        main(['run', str(suite), '--requirements-partial'])
    assert e.value.code == 2


def test_no_requirements_excludes_partial(toy, capsys):
    with pytest.raises(SystemExit) as e:
        main(['run', str(toy.root), '--no-requirements', '--requirements-partial', '--llm', 'none'])
    assert e.value.code == 2 and 'exclude each other' in capsys.readouterr().err


def test_the_preview_command_says_when_the_text_is_blank(toy, capsys):
    with pytest.raises(SystemExit) as e:
        main(['requirements', str(toy.root), '--text', '   ', '--llm', 'none'])
    assert e.value.code == 2 and 'no requirements' in capsys.readouterr().err


def test_init_keeps_the_text(toy, tmp_path, capsys):
    doc = yaml.safe_load((toy.root / 'infrarsi.yaml').read_text(encoding='utf-8'))
    assert 'requirements' not in doc
    main(['init', str(toy.root), '--force', '--requirements', '不要动 loss'])
    doc = yaml.safe_load((toy.root / 'infrarsi.yaml').read_text(encoding='utf-8'))
    assert doc['requirements'] == {'text': ['不要动 loss']}


def test_the_requirements_command_shows_and_caches(toy, tmp_path, monkeypatch, capsys):
    made = _fake_llm(monkeypatch, [item('pin_knob', 'keep', knob='precision')])
    main(['requirements', str(toy.root), '--llm', 'fake:m', '--text', 'keep the precision'])
    assert 'enforced' in capsys.readouterr().out and len(made) == 1
    main(['requirements', str(toy.root), '--llm', 'fake:m', '--text', 'keep the precision', '--json'])
    record = json.loads(capsys.readouterr().out)
    assert record['schema'] == rq.SCHEMA and record['compile']['cached'] is True
    assert not (toy.root / '.infrarsi' / 'runs').exists()


def test_the_requirements_command_fails_when_something_cannot_be_honoured(toy, monkeypatch, capsys):
    _fake_llm(monkeypatch, [item('objective', 'save money')])
    with pytest.raises(SystemExit) as e:
        main(['requirements', str(toy.root), '--llm', 'fake:m', '--text', 'save money'])
    assert e.value.code == 1 and 'not supported' in capsys.readouterr().out


def test_doctor_reads_only_the_cache(toy, monkeypatch, capsys):
    from infrarsi.cli.main import _print_requirements
    toy.raw['requirements'] = {'text': ['keep']}
    _print_requirements(toy)
    assert 'not compiled yet' in capsys.readouterr().out
    rq.compile(toy, ['keep'], driver=model([item('pin_knob', knob='precision')]))
    _print_requirements(toy)
    assert '1 enforced' in capsys.readouterr().out


def test_the_old_commands_parse_as_before():
    from infrarsi.cli.main import build
    assert build().parse_args(['train.py']).cmd == 'optimize'
    assert build().parse_args(['requirements']).cmd == 'requirements'
    assert build().parse_args(['run', '--resume', 'x']).requirements is None


def test_a_namespace_without_the_new_flags_reads_as_none():
    from infrarsi.cli.main import _requirement_texts
    project = SimpleNamespace(raw={})
    assert _requirement_texts(SimpleNamespace(), project) == {'manifest': [], 'cli': [], 'interactive': []}


def test_a_resumed_run_keeps_the_requirements_it_started_with(toy, tmp_path, monkeypatch):
    from test_bcomp_project import scripted
    from infrarsi.cli.main import _resumed_requirements
    from infrarsi.project import runner as runner_mod
    from infrarsi.project.resume import prepare
    monkeypatch.setenv('INFRARSI_HOME', str(tmp_path / 'home'))
    rec = compiled(toy, [item('pin_knob', knob='fused_optimizer')])
    real, seen = runner_mod.ProjectRunner.run, {'n': 0}

    def run(self, task, config, overlay, purpose, run_id, **kw):
        seen['n'] += purpose == 'screen'
        if seen['n'] == 3:
            raise KeyboardInterrupt
        return real(self, task, config, overlay, purpose, run_id, **kw)
    monkeypatch.setattr(runner_mod.ProjectRunner, 'run', run)
    drv, _, _ = scripted(toy.root)
    res = run_project(rq.apply(toy, rec), strategy='bcomp', driver=drv, budget={'screens': 8, 'llm_tokens': 10 ** 7},
                      out_root=tmp_path / 'out', log=lambda m: None, requirements=rec, on_interrupt='stop')
    r = prepare(res['run_dir'])
    assert r.requirements['digest'] == rec['digest']
    p, kept = _resumed_requirements(Project.load(toy.root), r)
    assert kept['digest'] == rec['digest'] and p.knobs['fused_optimizer']['choices'] == [False]
    assert r.entry['requirements']['changed'] is True    # (the manifest itself has no requirements block)


def test_a_resumed_run_without_requirements_ignores_a_new_manifest_block(toy):
    from infrarsi.cli.main import _resumed_requirements
    toy.raw['requirements'] = {'text': ['keep']}
    resume = SimpleNamespace(requirements=None, entry={})
    p, kept = _resumed_requirements(toy, resume)
    assert kept is None and p is toy and resume.entry['requirements']['changed'] is True


def test_the_narrowed_project_is_what_the_strategy_sees(toy):
    from infrarsi import search
    rec = compiled(toy, [item('only_knobs', knobs=['workers'])])
    p = rq.apply(toy, rec)
    assert search.resolve('auto', has_llm=False, knobs=p.knobs) in search.STRATEGIES
    assert all(len(choices(p.task())[k]) == 1 for k in p.knobs if k != 'workers')
