"""v4's offline validation on the real records (FABLE_CODE_DIAGNOSIS_20261102 section 5.1 (a)-(f); V4_IMPLEMENTATION_BRIEF):
each fix's rule applied to the finished open arms under qa_trials/trials_tb/<ENV>-open-bcomp/ - their result.json,
search/events.jsonl, llm_calls/*/ and attempt trees, read only (a test skips when its records are absent). The rules are
the tree's own: FLOOR_SD, STRONG_P and the floor kinds (F1), the stack's floor (F2), finalist_base.stream_charge (F5),
OVERLAY_CEILING (F4). What a record cannot say (what the code track would have done with a screen it was handed, the
stack's merged lines) is left out and said so. `python test_v4_replays.py` prints the numbers (the Codex brief's)."""
import glob
import json
import math
import re
from pathlib import Path

import pytest

from infrarsi.search.finalist_base import FLOOR_SD, OVERLAY_CEILING, STRONG_MIN, STRONG_P, stream_charge

from infrarsi.search.bcomp import BComp

BCOMP_KINDS = sorted(BComp.fb_floor_kinds)

TB = Path(__file__).resolve().parents[4] / 'trials_tb'
ENVS = ('Q9U', 'Q9H', 'G4U', 'Q27U', 'M35H2', 'DXH', 'DOH')
SCREENS = 20                                         # the open arms' budget
ASK_LEFT = 2
MEM_FAIL = ('oom', 'thrash')
TOGETHER = ('crash', 'compiler', 'collective', 'hang', 'start_cap', 'start_slow')
MIN_CALL_USD = 0.05


# ---------------------------------------------------------------- reading the records
def result(env: str, arm: str) -> dict | None:
    got = sorted(glob.glob(str(TB / f'{env}-open-{arm}' / '*__result.json')))
    return json.loads(Path(got[-1]).read_text(encoding='utf-8')) if got else None


def run_dir(env: str, arm: str) -> Path | None:
    got = sorted(glob.glob(str(TB / f'{env}-open-{arm}' / 'repo' / '.infrarsi' / 'runs' / '*' / 'search')))
    return Path(got[-1]).parent if got else None


def need(*arms):
    if not all(result(e, a) is not None for e, a in arms):
        pytest.skip('the open records are not there')


def strategy_view(r: dict, arm: str) -> dict:
    return r['search'][arm]


def sigma_pair(r: dict) -> float:
    return float(((r['search'].get('noise') or {}).get('sigma_pair')) or 0.0)


def floor_ln(sp: float) -> float:
    return max(math.log1p(STRONG_MIN), 2 * sp)


def cid_n(cid) -> int:
    m = re.match(r'c(\d+)$', str(cid or ''))
    return int(m.group(1)) if m else 10 ** 6


# ---------------------------------------------------------------- (b) F1: the floor on bcomp's picks
def f1_replay(env: str, arm: str = 'bcomp', kinds=None) -> dict:
    """The record's knob-side picks (its `picks`, each with the proposals ledger's sd) under F1: a held kind (bcomp v4's
    tail and draw) while the floor holds - a full screen done, ASK_LEFT screens left -
    is refused when its predicted gain + FLOOR_SD sd is below max(2 %, 2 sigma_pair) or its P below STRONG_P (the record's
    final sigma_pair; the code track's budget and probes taken as open). Also: the picks predicted below 2 % (the
    diagnosis's count), and the knob picks that improved - measured more than sigma_pair over the best earlier run with the
    same patches (a pick's gain net of a patch the chain gained in between) - with those the floor would refuse."""
    r = result(env, arm)
    v = strategy_view(r, arm)
    led = {x['id']: x for x in r['search'].get('ledger') or []}
    sp = sigma_pair(r)
    thr = floor_ln(sp)
    kinds = set(kinds or BCOMP_KINDS)
    best: dict = {frozenset(): 0.0}                  # patch set -> the best measured ln gain so far (the reference: 0)
    knob, refused, below2, better, refused_better = [], [], [], [], []
    for pk in v.get('picks') or []:
        row = led.get(pk['id']) or {}
        pset = frozenset(row.get('patches') or ())
        g = pk.get('gain_vs_reference_pct')
        y = math.log1p(g / 100.0) if isinstance(g, (int, float)) else None
        if pk['kind'] not in ('patch', 'stack'):
            mu = math.log1p((pk.get('predicted_gain_pct') or 0.0) / 100.0)
            sd = (row.get('sigma_pct') or 0.0) / 100.0
            p = pk.get('p_feasible') or 0.0
            n = pk.get('screens_done') or 0
            knob.append(pk['id'])
            if pk.get('predicted_gain_pct') is not None and pk['predicted_gain_pct'] < 2.0 and pk['kind'] != 'prior':
                below2.append(pk['id'])
            held = pk['kind'] in kinds and n >= 1 and SCREENS - n >= ASK_LEFT
            out = held and (mu + FLOOR_SD * sd + 1e-12 < thr or p < STRONG_P)
            if out:
                refused.append(pk['id'])
            if y is not None and pset in best and y - best[pset] > sp:
                better.append(pk['id'])
                if out:
                    refused_better.append(pk['id'])
        if y is not None:
            best[pset] = max(best.get(pset, y), y)
    return {'sigma_pair': round(sp, 5), 'floor_pct': round(100 * math.expm1(thr), 2), 'knob_picks': len(knob),
            'refused': refused, 'below_2pct': below2, 'improved': better, 'improved_refused': refused_better}


def test_f1_the_floor_keeps_every_knob_improvement():
    need(*[(e, 'bcomp') for e in ENVS])
    for env in ENVS:
        got = f1_replay(env)
        assert not got['improved_refused'], (env, got)
        assert len(got['refused']) <= got['knob_picks']


# ---------------------------------------------------------------- (c) F2: the stacks
def f2_replay(env: str, arm: str) -> dict:
    """The record's proven patches not adopted with a positive gain (0 < gain <= sigma_pair), in the order they ran,
    stacked as F2 stacks them: a test once the pending ones' summed ln gain reaches max(2 %, 2 sigma_pair) and two or more
    are pending (the record's final sigma_pair; the merge and the line limit are not checked: no files read)."""
    r = result(env, arm)
    v = strategy_view(r, arm)
    sp = sigma_pair(r)
    thr = floor_ln(sp)
    pats = v.get('patches') or {}
    sub = sorted(((cid_n(rec.get('candidate')), pid, rec['gain_pct']) for pid, rec in pats.items()
                  if rec.get('status') == 'proven' and not rec.get('adopted') and isinstance(rec.get('gain_pct'), (int, float))
                  and rec['gain_pct'] > 0), key=lambda t: t[0])
    stacks, pending = [], []
    for _, pid, g in sub:
        pending.append((pid, g))
        if len(pending) >= 2 and sum(math.log1p(x / 100.0) for _, x in pending) + 1e-12 >= thr:
            stacks.append({'members': [p for p, _ in pending],
                           'sum_pct': round(100 * math.expm1(sum(math.log1p(x / 100.0) for _, x in pending)), 2)})
            pending = []
    return {'sigma_pair': round(sp, 5), 'floor_pct': round(100 * math.expm1(thr), 2),
            'within_noise': [(p, g) for _, p, g in sub], 'stacks': stacks,
            'left_pending_pct': round(100 * math.expm1(sum(math.log1p(x / 100.0) for _, x in pending)), 2)}


def test_f2_the_stacks_the_records_would_have_tested():
    need(*[(e, a) for e in ENVS for a in ('bcomp',)])
    got = {(e, a): f2_replay(e, a) for e in ENVS for a in ('bcomp',)}
    for g in got.values():
        assert all(len(s['members']) >= 2 and s['sum_pct'] >= g['floor_pct'] - 1e-6 for s in g['stacks'])


# ---------------------------------------------------------------- (d) F4: the line cap per patch
def _file_lines(path: Path) -> list[str]:
    return path.read_text(encoding='utf-8', errors='replace').splitlines() if path.is_file() else None


def _diff_count(a: list, b: list) -> int:
    import difflib
    return sum(1 for ln in difflib.unified_diff(a, b, lineterm='', n=0)
               if not ln.startswith(('+++', '---', '@@')) and ln[:1] in ('+', '-'))


def f4_replay(env: str, arm: str) -> dict:
    """The line-cap refusals of the arm's events (`invalid_proposal`, 'diff N lines > [strong] limit L' - N the merged
    overlay's lines against the project's files) under F4: the new patch's own lines ~ N less its base's own overlay lines
    (the base: the incumbent its sibling patches of the same ask were proposed on; its overlay read from its attempt tree
    against the reference's), passing when they are at most L and N at most OVERLAY_CEILING x L. An estimate: the new
    patch's proof lines and an edit of a chain line count as in v3."""
    rd = run_dir(env, arm)
    ev = [json.loads(ln) for ln in (rd / 'search' / 'events.jsonl').read_text(encoding='utf-8').splitlines() if ln.strip()]
    trees = {}
    for t in glob.glob(str(rd / 'attempts' / '*' / '*' / 'tree')):
        cid = Path(t).parent.name.split('-', 1)[0]
        trees.setdefault(cid, Path(t))
    mut = None
    if 'c0' in trees:
        for f in ('train_patches.py', 'train.py'):
            if (trees['c0'] / f).is_file():
                mut = f
                break
    ref = _file_lines(trees['c0'] / mut) if mut else None
    cum: dict = {}

    def overlay_lines(cid):
        if cid not in cum:
            mine = _file_lines(trees[cid] / mut) if cid in trees and mut else None
            cum[cid] = None if mine is None or ref is None else _diff_count(ref, mine)
        return cum[cid]
    out, unknown = [], 0
    for i, e in enumerate(ev):
        if e['kind'] != 'invalid_proposal':
            continue
        for f in e['payload'].get('findings') or []:
            m = re.match(r'diff (\d+) lines > (?:\w+ )?limit (\d+)', str(f))
            if not m:
                continue
            n, lim = int(m.group(1)), int(m.group(2))
            base = None
            for j in list(range(i + 1, len(ev))) + list(range(i - 1, -1, -1)):
                x = ev[j]
                if x['kind'] == 'llm_call' and (x['payload'] or {}).get('role') == 'hypotheses' and j > i:
                    continue
                if x['kind'] == 'candidate' and (x['payload'] or {}).get('origin') == 'wcs-patch':
                    base = x['payload'].get('parent')
                    break
            have = overlay_lines(base) if base else None
            if have is None:
                unknown += 1
                out.append({'n': n, 'limit': lim, 'base': base, 'own': None, 'passes': None})
                continue
            own = n - have
            out.append({'n': n, 'limit': lim, 'base': base, 'own': own,
                        'passes': own <= lim and n <= OVERLAY_CEILING * lim})
    return {'refusals': len(out), 'pass': sum(1 for x in out if x['passes']), 'fail': sum(1 for x in out
                                                                                        if x['passes'] is False),
            'unknown': unknown, 'own_lines': sorted(x['own'] for x in out if x['own'] is not None), 'rows': out}


def test_f4_most_refused_proposals_pass_per_patch_counting():
    arms = [(e, a) for e in ENVS for a in ('bcomp',)]
    need(*arms)
    got = {ea: f4_replay(*ea) for ea in arms if run_dir(*ea) is not None}
    assert sum(g['refusals'] for g in got.values()) == 47
    assert {f'{e}-{a}': g['refusals'] for (e, a), g in got.items() if g['refusals']} == {
        'Q9U-bcomp': 3, 'Q9H-bcomp': 23, 'G4U-bcomp': 5,
        'M35H2-bcomp': 13, 'DXH-bcomp': 3}
    known = sum(g['pass'] + g['fail'] for g in got.values())
    assert known >= 40 and sum(g['pass'] for g in got.values()) >= 0.8 * known
    assert all(x['own'] is None or x['own'] < x['n'] for g in got.values() for x in g['rows'])


# ---------------------------------------------------------------- (e) F5: the failed calls
def f5_replay() -> list[dict]:
    """Every failed model call of the open bcomp arms (a receipt with an error or a status other than RETURNED):
    the dollars v3 charged and v4's (stream_charge for a call that hit its wall - the cap only when the stream reported
    nothing; any other failure: its reported cost, unchanged), and whether v4 re-asks it once with a slim payload."""
    out = []
    for env in ENVS:
        for arm in ('bcomp',):
            rd = run_dir(env, arm)
            if rd is None:
                continue
            for rec in sorted(glob.glob(str(rd / 'llm_calls' / '*' / 'receipt.json'))):
                m = json.loads(Path(rec).read_text(encoding='utf-8'))
                if not m.get('error') and m.get('status') == 'RETURNED':
                    continue
                v4, how = m.get('llm_usd'), 'its reported cost (unchanged)'
                if m.get('status') == 'TIMEOUT':
                    usd, how = stream_charge(Path(rec).parent, m.get('requested_model'))
                    v4 = min(m['llm_usd'], usd) if usd is not None else m['llm_usd']
                out.append({'arm': f'{env}-{arm}', 'call': Path(rec).parent.name, 'status': m.get('status'),
                            'error': str(m.get('error'))[:60], 'v3_usd': m.get('llm_usd'), 'v4_usd': round(v4, 3),
                            'slim_reask': m.get('status') == 'TIMEOUT' and m.get('role') == 'hypotheses', 'how': how})
    return out


def test_f5_the_timed_out_calls_are_charged_their_streamed_usage():
    need(('Q9U', 'bcomp'))
    rows = f5_replay()
    timeouts = [r for r in rows if r['status'] == 'TIMEOUT']
    assert len(timeouts) == 1 and all(r['v3_usd'] == 5.0 for r in timeouts)
    assert all(1.0 < r['v4_usd'] < 4.0 for r in timeouts) and all(r['slim_reask'] for r in timeouts)
    others = [r for r in rows if r['status'] != 'TIMEOUT']
    assert all(r['v4_usd'] == round(r['v3_usd'], 3) for r in others)
    assert {r['arm'] for r in timeouts} == {'Q9U-bcomp'}


# ---------------------------------------------------------------- (f) F6: bcomp's early ends
def f6_replay(env: str) -> dict:
    """bcomp's record where it stopped: why, the screens, calls and dollars used of the budget, the last ask - and
    whether v4's F6 holds there (a call and dollars left, ASK_LEFT screens left: the code track is handed the step and
    asks again, so the arm does not end)."""
    r = result(env, 'bcomp')
    s = r['search']
    used, budget = s.get('used') or {}, s.get('budget') or {}
    v = strategy_view(r, 'bcomp')
    calls, calls_max = v.get('llm_calls'), v.get('llm_calls_max')
    usd, usd_max = float(used.get('llm_usd') or 0.0), float(v.get('llm_usd_max') or 0.0)
    screens = int(used.get('screens') or 0)
    asks = ((v.get('finalist') or {}).get('asks') or [])
    return {'stop': s.get('stop_reason'), 'screens': f'{screens} / {budget.get("screens")}',
            'calls': f'{calls} / {calls_max}', 'usd': f'{usd:.2f} / {usd_max:.2f}',
            'last_ask': {k: asks[-1].get(k) for k in ('screens_done', 'queued', 'why')} if asks else None,
            'v4_asks_again': bool(calls_max - calls >= 1 and usd_max - usd >= MIN_CALL_USD
                                  and int(budget.get('screens') or 0) - screens >= ASK_LEFT)}


def test_f6_bcomp_would_not_have_ended_with_screens_calls_and_dollars_left():
    need(('Q27U', 'bcomp'), ('G4U', 'bcomp'))
    for env, screens in (('Q27U', 11), ('G4U', 12)):
        g = f6_replay(env)
        assert g['stop'] == 'strategy_done' and g['screens'] == f'{screens} / 20' and g['v4_asks_again'], (env, g)


# ---------------------------------------------------------------- the numbers
def summary() -> dict:
    out = {'f1_bcomp': {e: f1_replay(e, 'bcomp') for e in ENVS if result(e, 'bcomp')},
           'f2': {f'{e}-{a}': f2_replay(e, a) for e in ENVS for a in ('bcomp',) if result(e, a)},
           'f4': {f'{e}-{a}': {k: v for k, v in f4_replay(e, a).items() if k != 'rows'}
                  for e in ENVS for a in ('bcomp',) if run_dir(e, a) is not None},
           'f5': f5_replay(), 'f6': {e: f6_replay(e) for e in ('Q27U', 'G4U') if result(e, 'bcomp')}}
    return out


if __name__ == '__main__':
    print(json.dumps(summary(), indent=1, default=str))
