"""v4.1 (tree pc_code_o2b - bcomp; the user's rules of 11-03): the three fixes of the
budget-ended re-runs. Bounded asks (search/fold.py): what a model call shows of the past is bounded - the measured history,
the patch call's ledger, the failed runs and the inert patches -, the rows that matter most whole within a budget and the
rest in one summary that keeps its information; a list that fits is shown as before. The stacking loop: a knob variation of
a candidate carries that candidate's overlay as inherited (agent/search.py `_inherited_overlay`). Search-state
checkpoints: the state is saved after every observed run, and a search continued from one goes on where it stopped. The
same file runs in both trees."""
import json
import math

import pytest

from finalist_helpers import fin
from infrarsi.agent import state as S
from infrarsi.search import STRATEGIES
from infrarsi.search import checkpoint as ckpt
from infrarsi.search import fold as F
from infrarsi.search.toytrain import mem_tasks
from test_open_scaling import NAME, _registered_script
from test_v4 import _session


# ---------------------------------------------------------------- the measured history
def _runs(st, n, patches=8):
    """n runs on the reference values: every fifth out of memory, the others measured (slow to fast), each carrying the
    first i % `patches` of `patches` patches with long mechanisms."""
    for j in range(patches):
        st.patches[f'p{j}'] = {'mechanism': f'patch {j} ' + 'x' * 90, 'status': 'proven'}
    for i in range(n):
        cfg = {**st.ref, **{f'patch:p{j}': True for j in range(i % patches)}}
        oom = i % 5 == 4
        st.runs[f'x{i}'] = {'cfg': cfg, 'kind': 'oom' if oom else 'ok', 'status': 'infeasible' if oom else 'screened',
                            'y': None if oom else 0.001 * i, 'fidelity': 'full'}


def test_a_history_that_fits_is_shown_as_it_was():
    st, eng = fin(None, screens=20)
    st._init(eng)
    _runs(st, 6, patches=2)
    st.rejected = [{'change': {'micro_batch': i}, 'why': 'planted', 'source': 'x'} for i in range(10)]
    new = st._history(eng)
    st.bounded_asks = False
    old = st._history(eng)
    assert new == old and len(old) == 6 + 10 and not any('folded' in r for r in new)
    st.rejected[:0] = [{'change': {'micro_batch': 7}, 'why': 'older', 'source': 'x'}] * 2   # (v4: the last ten only)
    old = st._history(eng)
    st.bounded_asks = True
    new = st._history(eng)
    assert new[:-1] == old and new[-1]['folded']['ended'] == {'not run': 2}


def test_the_history_beyond_its_budget_is_folded_into_one_summary():
    st, eng = fin(None, screens=20)
    st._init(eng)
    _runs(st, 100)
    rows = st._history(eng)
    whole, summ = rows[:-1], rows[-1]['folded']
    assert sum(F.size(r) for r in whole) <= F.HISTORY_CHARS and len(whole) <= st.history_max
    assert whole[0]['outcome'] == f'x{math.exp(0.098):.3f} the reference speed'           # (the fastest first)
    assert summ['rows'] == 100 - len(whole) == sum(summ['ended'].values()) and summ['ended']['out of memory'] > 0
    assert summ['measured_speed'].startswith('x1.000 to')                                  # (the slowest is folded)
    assert any(x.startswith('patch 1 ') and 'measured, best x' in x for x in summ['code_patches'])
    st.bounded_asks = False
    old = st._history(eng)
    assert len(old) == st.history_max == 80 and F.size(rows) < F.size(old) / 2


@pytest.mark.parametrize('n', [40, 160, 640])
def test_the_history_stays_within_a_constant_however_long_the_run(n):
    st, eng = fin(None, screens=20)
    st._init(eng)
    _runs(st, n, patches=12)
    st.rejected = [{'change': {'micro_batch': i}, 'why': 'planted ' + 'w' * 80, 'source': 'x'} for i in range(n // 4)]
    rows = st._history(eng)
    assert F.size(rows) <= F.HISTORY_CHARS + 2 * F.SUMMARY_ITEMS * F.LINE + 1500


# ---------------------------------------------------------------- the ledger and the records
def test_a_long_ledger_keeps_the_newest_whole_then_a_line_each_then_counts():
    rows = [{'mechanism': f'm{i} ' + 'y' * 150, 'change': {}, 'status': 'measured' if i % 3 else 'inert at its probe',
             'measured_gain_pct': i / 10, 'site': 's' * 150} for i in range(300)]
    out = F.fold_ledger(rows)
    head, whole = out[0]['folded'], out[1:]
    assert sum(map(F.size, whole)) <= F.LEDGER_CHARS and 0 < len(whole) <= F.LEDGER_ROWS and whole == rows[-len(whole):]
    k = 300 - len(whole)
    assert head['lines'][0].startswith(f'm{k - 1} ') and sum(len(x) for x in head['lines']) <= F.LINES_CHARS + F.LINE
    assert sum(head['oldest_counted'].values()) == k - len(head['lines']) and 'inert at its probe' in head['oldest_counted']
    assert F.fold_ledger(rows[:5]) == rows[:5]


def test_failed_runs_and_inert_patches_fold_with_a_line_each():
    inert = [{'candidate': f'c{i}', 'mechanism': f'm{i}', 'site': 'here', 'at': 'probe', 'reason': 'no active line',
              'printed': ['[infrarsi-patch] x: found 1 -> set 2' * 3] * 6} for i in range(12)]
    out = F.fold_records(inert, chars=F.INERT_CHARS, line=F.inert_line, label=F.inert_label, what='inert patches')
    assert out[0]['folded']['lines'][0].startswith(f'c{12 - len(out)}: m{12 - len(out)} @ here - inert at its probe')
    assert out[-1] is inert[-1] and sum(map(F.size, out[1:])) <= F.INERT_CHARS
    fails = [{'candidate': f'c{i}', 'class': 'oom', 'mechanism': 'more memory', 'detail': 'short by 2 GiB'} for i in range(4)]
    assert F.fold_records(fails, chars=F.FAILURE_CHARS, line=F.failure_line, label=F.failure_label, what='x') == fails


def test_the_patch_call_context_folds_every_failure_and_inert_patch():
    st, eng = fin(None, screens=20)
    st._init(eng)
    st.failures = [{'candidate': f'c{i}', 'class': 'crash', 'mechanism': 'm' * 120, 'detail': 'd' * 200} for i in range(30)]
    st.inert = [{'candidate': f'c{i}', 'mechanism': 'm', 'at': 'full run', 'predicted_pct': 3.0, 'measured_pct': 0.1}
                for i in range(9)]
    ctx = st.context(eng, eng.incumbent())
    assert 'folded' in ctx['failures'][0] and ctx['failures'][-1] == st.failures[-1]
    assert ctx['inert_patches'] == st.inert                                     # (nine small ones fit: as they were)
    st.bounded_asks = False
    assert st.context(eng, eng.incumbent())['failures'] == st.failures[-10:]


# ---------------------------------------------------------------- the registered strategy, end to end
def _run(tmp_path, screens, *, seen=None, options=None, run_id='toy'):
    """The tree's finalist as the CLI runs it (STRATEGIES -> run_engine) on the memory toy with test_open_scaling's
    scripted model; `seen` collects (role, payload) of every call."""
    from infrarsi.agent.backends.fake import ScriptedBackend
    from infrarsi.search import Context
    from infrarsi.search import toybench as tb
    sim = tb.make_sim(mem_tasks()[1])
    runner = tb.ToyRunner(sim, tb.VirtualClock())
    ref_cfg = sim.reference()
    probe_task = tb.EnvTask.from_dict(tb.toy_task_spec(sim, {'screens': 1, 'llm_tokens': 0, 'wall_seconds': 1e12,
                                                             'gpu_hours': 1e9}))
    ref = runner.run(probe_task, ref_cfg, None, 'reference', 'toy', candidate_id='c0', steps=tb.FULL)
    env = tb.EnvTask.from_dict(tb.toy_task_spec(sim, {'screens': screens, 'llm_tokens': 10 ** 7, 'wall_seconds': 1e12,
                                                      'gpu_hours': 1e9}))
    script = _registered_script()

    def recorded(role, system, prompt):
        if seen is not None:
            try:
                seen.append((role, json.loads(prompt)))
            except ValueError:
                seen.append((role, prompt))
        return script(role, system, prompt)
    ctx = Context(task=env, budget=env.search_budget(), start_config=ref_cfg, runner=runner,
                  judge=tb.Verifier(0.03, memory_cap_gib=sim.hw['mem']), run_id=run_id, out_dir=tmp_path,
                  hooks=tb.MemoryHooks(env, sim.memory_facts, base_files={'patches.txt': ''}), start_receipt=ref,
                  reference_receipt=ref, driver=ScriptedBackend(recorded, tokens_per_call=2000), seed=1,
                  options={'full_steps': tb.FULL, 'short_steps': tb.SHORT, 'probe_steps': tb.PROBE, 'job_gpu_hours': None,
                           **(options or {})})
    trace = STRATEGIES[NAME](ctx)
    return trace, ctx.sessions[-1]


def test_the_patch_call_shows_the_whole_ledger_folded(tmp_path, monkeypatch):
    import infrarsi.search.proposers as P
    real = P.fold_ledger
    monkeypatch.setattr(P, 'fold_ledger', lambda rows, chars=F.LEDGER_CHARS: real(rows, chars=min(chars, 600)))
    seen = []
    trace, s = _run(tmp_path, 20, seen=seen)
    led = [p['ledger'] for role, p in seen if role == 'hypotheses' and isinstance(p, dict)]
    folded = [x for x in led if x and 'folded' in x[0]]
    assert folded and all(x[0]['folded']['lines'] for x in folded) and trace.stop_reason.startswith('budget')
    assert all(len(x) < len(led[-1]) + 40 for x in led)


# ---------------------------------------------------------------- the stacking loop
def test_a_knob_variation_carries_the_incumbents_overlay_as_inherited():
    """The loop of G4U / M35H2 v4: a knob variation of the incumbent is another configuration on the reference with the
    incumbent's files; v4 counted that whole overlay as its own lines and refused every variation. v4.1: inherited."""
    s = _session(True)
    lim = min(int(s.hooks.patch_line_limit), int(s.patch_line_limit or 300))
    first = ''.join(f'a{i}\n' for i in range(lim - 2))
    r1 = s.propose_patch(files={'patches.txt': first}, base='c0')
    more = first + ''.join(f'b{i}\n' for i in range(5))
    r2 = s.propose_patch(files={'patches.txt': more}, base=r1['candidate_id'])
    assert 'candidate_id' in r2
    cv = s.propose_config(changes={'micro_batch': 2}, base='c0')['candidate_id']
    got = s.propose_patch(files={'patches.txt': more}, base=cv)
    assert 'candidate_id' in got and got['diff_lines'] == 0                       # (inherited: no line of its own)
    on_top = s.propose_patch(files={'patches.txt': more + 'c\n'}, base=cv)       # (a new overlay: counted as before)
    assert 'candidate_id' not in on_top and any('own lines' in f for f in on_top['findings'])


# ---------------------------------------------------------------- search-state checkpoints
def test_a_search_saves_its_state_and_a_resumed_search_goes_on_where_it_stopped(tmp_path):
    d = tmp_path / 'state'
    t1, s1 = _run(tmp_path / 'a', 8, options={'checkpoint': ckpt.Writer(d)}, run_id='toy-a')
    data, path = S.read(d)
    h = data['header']
    assert path.name == S.CHECKPOINT and h['why'] == 'end' and h['screens_done'] == 8 and h['strategy'] == NAME
    assert (d / S.PREVIOUS).is_file() and json.loads((d / S.SUMMARY).read_text(encoding='utf-8'))['screens_done'] == 8
    calls_before, cands_before = len(s1.llm_calls), dict(s1.candidates)

    def resume(session, engine, strategy):
        state = S.state_of_file(data, ckpt.live_objects(session, engine, strategy))
        ckpt.restore(state, session, engine, strategy)
        strategy.rebudget(engine)
    t2, s2 = _run(tmp_path / 'b', 14, options={'resume': resume, 'checkpoint': ckpt.Writer(tmp_path / 'state2')},
                  run_id='toy-a')
    assert s2.ledger.used['screens'] == 14 and t2.stop_reason.startswith('budget')
    assert set(cands_before) <= set(s2.candidates) and len(s2.candidates) > len(cands_before)
    assert all(s2.candidates[c].config == cands_before[c].config for c in cands_before)
    assert len(s2.llm_calls) >= calls_before and S.read(tmp_path / 'state2')[0]['header']['screens_done'] == 14


def _resumer(data):
    def resume(session, engine, strategy):
        state = S.state_of_file(data, ckpt.live_objects(session, engine, strategy))
        ckpt.restore(state, session, engine, strategy)
        strategy.rebudget(engine)
    return resume


def test_an_interrupted_run_keeps_its_paid_calls_and_runs_the_chosen_proposal_first(tmp_path, monkeypatch):
    """OURS_V41 r1 P1: a proposal chosen with a model call is checkpointed before its run ('proposed'); interrupted during
    that run, the resumed search runs the same proposal first, without asking the model again, the paid call kept."""
    from infrarsi.search import engine as E
    real = E.Engine.evaluate
    seen = {'prev': None, 'hit': None}

    def key(self, p):
        return len(self.s.llm_calls), p.mechanism, sorted(p.change.items(), key=str), sorted((p.patch or {}).items())

    def interrupted(self, p):
        n = len(self.s.llm_calls)
        if seen['prev'] is not None and n > seen['prev'] and len(self.s.screens) >= 3:
            seen['hit'] = key(self, p)
            raise KeyboardInterrupt
        seen['prev'] = n
        return real(self, p)
    monkeypatch.setattr(E.Engine, 'evaluate', interrupted)
    d = tmp_path / 'state'
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path / 'a', 12, options={'checkpoint': ckpt.Writer(d)}, run_id='toy-i')
    data, _ = S.read(d)
    assert seen['hit'] and data['header']['why'] == 'proposed' and data['header']['llm_calls'] == seen['hit'][0]
    first = []

    def watched(self, p):
        first.append(key(self, p))
        return real(self, p)
    monkeypatch.setattr(E.Engine, 'evaluate', watched)
    t2, s2 = _run(tmp_path / 'b', 14, options={'resume': _resumer(data)}, run_id='toy-i')
    assert first[0] == seen['hit'] and t2.stop_reason.startswith('budget') and s2.ledger.used['screens'] == 14


def test_an_unpaid_choice_writes_no_checkpoint_of_its_own(tmp_path):
    w = ckpt.Writer(tmp_path / 'state')
    t, s = _run(tmp_path / 'a', 6, options={'checkpoint': w}, run_id='toy-w')
    h = S.read(tmp_path / 'state')[0]['header']
    assert h['why'] == 'end' and w.sequence <= 2 * 6 + len(s.llm_calls) + 1


def test_the_run_command_keeps_every_v4_option():
    """OURS_V41 r1 P0: `--capability` was lost when the resume options were added; every run read it."""
    from infrarsi.cli.main import build
    a = build().parse_args(['run', '-v', '--screens', '5', '--strategy', NAME])
    assert a.capability is None and a.resume is None and a.budget_usd is None and a.screens == 5
    b = build().parse_args(['run', '--resume', 'somewhere', '--screens', '30', '--capability', '0.5', '--budget-usd', '9'])
    assert b.capability == 0.5 and b.resume == 'somewhere' and b.budget_usd == 9.0


def test_one_row_over_the_budget_is_folded_too():
    """OURS_V41 r1 P2: the first row was kept whole whatever its size."""
    big = {'mechanism': 'm', 'site': 's' * 20000}
    out = F.fold_records([{'mechanism': 'a'}, big], chars=F.LEDGER_CHARS, line=F.ledger_line, label=F.ledger_label,
                         what='ledger entries')
    assert len(out) == 1 and out[0]['folded']['lines'][0].startswith('m -> open')
    rows = [{'row': {'change': {'k': 'v' * 20000}, 'outcome': 'x1.200 the reference speed'}, 'group': 'measured', 'order': 0,
             'y': 0.18, 'label': 'measured'},
            {'row': {'change': {'k': 1}, 'outcome': 'x1.100 the reference speed'}, 'group': 'measured', 'order': 1, 'y': 0.09,
             'label': 'measured'}]
    got = F.fold_history(rows)
    assert got[0] == rows[1]['row'] and got[1]['folded']['rows'] == 1 and sum(map(F.size, got[:1])) <= F.HISTORY_CHARS
