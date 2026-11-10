"""The SessionHooks seam of agent.optimize / agent.search (EXT_INTERFACES §3) on a toy external environment.
hooks=None behaviour is covered by the unchanged tests/cpu/agent suite."""
import json

import pytest

from infrarsi.agent import optimize
from infrarsi.agent.backends import ScriptedBackend
from infrarsi.agent.search import SearchSession
from infrarsi.ext.common.env_judge import EnvJudge
from infrarsi.ext.common.session_hooks import READ_FILE_MAX_LINES, SessionHooks
from infrarsi.rsi import build_a0
from x1_common_helpers import BASE_FILES, ToyRunner, local_ops, toy_task

PASS = {'verdict': 'pass', 'reasons': []}


@pytest.fixture(scope='module')
def a0(tmp_path_factory):
    return build_a0(tmp_path_factory.mktemp('pkg') / 'A0')


def _hooks(task=None, **kw):
    return SessionHooks(task or toy_task(), base_files=BASE_FILES, read_only_files={'README.md': 'line\n' * 900}, **kw)


def _run(tmp_path, a0, script, *, task=None, hooks=None, **kw):
    task = task or toy_task()
    runner = ToyRunner(local_ops(tmp_path))
    hooks = hooks or _hooks(task)
    drv = ScriptedBackend(lambda role, s, p: PASS if role == 'verifier' else script.pop(0), clock=runner.vclock)
    tr = optimize(task, a0, drv, task.search_budget(), {}, runner, judge=EnvJudge('toyenv'), run_id='ext-seam', arm='T-IS',
                  out_dir=tmp_path / 'run', clock=runner.vclock, hooks=hooks, **kw)
    return tr.to_dict(), drv, runner


def test_optimize_with_hooks_end_to_end(tmp_path, a0):
    fast = BASE_FILES['solution/run.py'].replace('time.sleep(1.0)', 'time.sleep(0.5)  # fast path')
    script = [{'action': 'read_env'},
              {'action': 'read_file', 'path': 'solution/run.py'},
              {'action': 'propose_config', 'base': 'c0', 'changes': {'workers': 4}},
              {'action': 'run_screen', 'candidate_id': 'c1'},
              {'action': 'propose_patch', 'base': 'c1', 'files': {'solution/run.py': fast}},
              {'action': 'read_file', 'path': 'solution/run.py', 'candidate_id': 'c2'},
              {'action': 'run_screen', 'candidate_id': 'c2'},
              {'action': 'read_receipt', 'candidate_id': 'c2'},
              {'action': 'finish', 'candidate_id': 'c2'}]
    tr, drv, runner = _run(tmp_path, a0, script)
    assert tr['task_id'] == 'ENV-T' and tr['stop_reason'] == 'finish' and tr['selected'] == 'c2'
    assert [s['step_time'] for s in tr['screens']] == [100.0, 25.0, 12.5]          # c0 screened first, then c1, c2
    assert [x['candidate_id'] for x in runner.seen] == ['c0', 'c1', 'c2']        # the runner learns the candidate id
    assert runner.seen[2]['overlay'] == {'solution/run.py': fast} and runner.seen[2]['config'] == {'workers': 4, 'flash': False}
    turns = [json.loads(line) for line in (tmp_path / 'run' / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    fb = {t['payload']['turn']: t['payload']['feedback'] for t in turns if t['kind'] == 'turn'}
    assert fb[1]['env']['env'] == 'toyenv' and fb[1]['env']['contract']
    assert fb[2]['path'] == 'solution/run.py' and 'time.sleep(1.0)' in fb[2]['text'] and fb[6]['text'].count('fast path') == 1
    assert fb[8]['metric']['value'] == 12.5 and 'guards' in fb[8]
    start = next(t for t in turns if t['kind'] == 'start')
    assert start['payload']['hooks']['env'] == 'toyenv' and tr['extra']['hooks']['layers'] == ['L0', 'L1']
    system = drv.calls[0]['system']
    assert '"action":"read_file"' in system and 'Environment contract' in system and 'Megatron-Core 0.18.2' in system
    assert 'mutable files: solution/*' in system
    verifier = next(c for c in drv.calls if c['role'] == 'verifier')['system']
    assert 'stays within the patch line limit' in verifier


def test_hooks_reject_bad_patches_and_actions(tmp_path, a0):
    long = BASE_FILES['solution/run.py'] + ''.join(f'x{i} = {i}\n' for i in range(30))
    script = [{'action': 'propose_patch', 'base': 'c0', 'files': {'score.py': 'x'}},
              {'action': 'propose_patch', 'base': 'c0', 'files': {'solution/run.py': long}},
              {'action': 'propose_patch', 'base': 'c0', 'files': {'solution/run.py': 'def broken(:\n'}},
              {'action': 'propose_patch', 'base': 'c0', 'files': {'solution/run.py': BASE_FILES['solution/run.py']}},
              {'action': 'propose_config', 'base': 'c0', 'changes': {'workers': 3}},
              {'action': 'profile_ranges', 'candidate_id': 'c0'},
              {'action': 'read_file', 'path': 'score.py'},
              {'action': 'read_file', 'path': 'README.md', 'start': 1},
              {'action': 'finish'}]
    tr, *_ = _run(tmp_path, a0, script)
    assert tr['invalid_proposals'] == {'patch_rejected': 4, 'l0_invalid': 1} and len(tr['screens']) == 1
    turns = [json.loads(x) for x in (tmp_path / 'run' / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    fb = {t['payload']['turn']: t['payload']['feedback'] for t in turns if t['kind'] == 'turn'}
    assert 'not mutable' in fb[1]['findings'][0] and 'frozen files' not in fb[1]['findings'][0]
    assert any('lines > limit 20' in f for f in fb[2]['findings']) and 'syntax error' in fb[3]['findings'][0]
    assert fb[4]['findings'] == ['patch identical to the base files'] and 'not in [1, 2, 4, 8]' in fb[5]['errors'][0]
    errors = {t['payload']['turn']: t['payload']['error'] for t in turns if t['kind'] == 'model_error'}
    assert 'unknown action' in errors[6] and 'not readable' in fb[7]['error']       # profile_ranges is not in the ext schema
    assert fb[8]['end'] - fb[8]['start'] + 1 == READ_FILE_MAX_LINES and fb[8]['total_lines'] == 900


def test_split_guard_and_layers(tmp_path, a0):
    with pytest.raises(PermissionError):
        _run(tmp_path, a0, [{'action': 'finish'}], hooks=_hooks(frozen_package_digest='0' * 64))
    with pytest.raises(PermissionError):
        _run(tmp_path, a0, [{'action': 'finish'}], heldout=object())
    with pytest.raises(ValueError):
        _hooks(toy_task(action_space='knobs', mutable_globs=[]), layers=('L1',))
    task = toy_task(action_space='patch', knob_schema={})
    tr, drv, _ = _run(tmp_path, a0, [{'action': 'propose_config', 'base': 'c0', 'changes': {'workers': 2}}, {'action': 'finish'}],
                      task=task, hooks=_hooks(task))
    assert tr['extra']['policy']['layers'] == ['L1'] and '"propose_config"' not in json.dumps(drv.calls[0]['system'])
    assert tr['invalid_proposals'] == {}                 # not in the schema: a model error, never dispatched


def test_weak_mode_menu_and_inspect_file(tmp_path, a0):
    task = toy_task(action_space='patch', knob_schema={})
    script = [{'action': 'inspect', 'what': 'file', 'path': 'solution/run.py'},
              {'action': 'propose_patch', 'base': 'c0', 'file': 'solution/run.py',
               'content': BASE_FILES['solution/run.py'].replace('1.0', '0.25')},
              {'action': 'run_trial', 'candidate_id': 'c1'}, {'action': 'finish', 'candidate_id': 'c1'}]
    tr, drv, _ = _run(tmp_path, a0, script, task=task, hooks=_hooks(task), mode='weak')
    assert tr['selected'] == 'c1' and len(tr['screens']) == 2                     # patch-only: no config-first lock
    assert 'propose_patch' in drv.calls[0]['system'] and 'apply_patch_template' not in drv.calls[0]['system']


def test_search_session_with_hooks_direct(tmp_path):
    task = toy_task()
    runner = ToyRunner(local_ops(tmp_path))
    s = SearchSession(task=task, budget=task.search_budget(), runner=runner, judge=EnvJudge(), run_id='r', arm='T-RND',
                      start_config={}, layers=('L0',), clock=runner.vclock, hooks=_hooks(task))
    assert s.candidates['c0'].config == {'workers': 1, 'flash': False}
    assert s.propose_config(config={'workers': 1})['duplicate_of'] == 'c0'
    c = s.propose_config(changes={'flash': True})['candidate_id']
    s.screen('c0')
    s.screen(c)
    tr = s.finalize(stop_reason='done').to_dict()
    assert tr['selected'] == c and tr['task_id'] == 'ENV-T' and tr['screens'][1]['cum']['gpu_hours'] > 0
    with pytest.raises(ValueError):
        SearchSession(task=task, budget=task.search_budget(), runner=runner, run_id='r', arm='x', start_config={'workers': 3},
                      hooks=_hooks(task))


def test_task_view_is_rendered_live_every_turn(tmp_path, a0):
    """O / PROPOSED_CHANGES_X3 #1: a view that changes between turns (ENV-B performance table) reaches every prompt."""
    class Live(SessionHooks):
        n = 0

        def task_view(self):
            Live.n += 1
            return {**super().task_view(), 'live_counter': f'LIVE-{Live.n}'}
    task = toy_task()
    script = [{'action': 'read_env'}, {'action': 'read_env'}, {'action': 'finish'}]
    tr, drv, _ = _run(tmp_path, a0, script, task=task, hooks=Live(task, base_files=BASE_FILES))
    counters = [p['prompt'].split('LIVE-')[1].split('"')[0] for p in drv.calls if p['role'] == 'proposer']
    assert len(counters) == 3 and len(set(counters)) == 3                     # a fresh view in every prompt
    events = [json.loads(x) for x in (tmp_path / 'run' / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    envs = [e['payload']['feedback']['env']['live_counter'] for e in events if e['kind'] == 'turn' and 'env' in e['payload']['feedback']]
    assert len(envs) == 2 and envs[0] != envs[1]                              # read_env is live as well
