"""Synthetic run records for the W5 summary tests: signed ext receipts (throwaway key), traces, result / locked files,
node references and the block, in the layouts the ENV-A / ENV-B / ENV-C builders write (X1-X3 reports)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path

from infrarsi.ext.autoresearch import knobs as AK
from infrarsi.ext.optico import task as OT
from infrarsi.ext.optico import workloads as OW
from infrarsi.harness import signing

T0 = dt.datetime(2026, 10, 21, 1, 0, tzinfo=dt.timezone.utc)
DAR = ('D-AR-1', 'D-AR-2', 'D-AR-3', 'D-AR-5', 'D-AR-7')
KNOB_SHA = hashlib.sha256((Path(AK.__file__).resolve().parent / AK.KNOB_SPACE_FILE).read_bytes()).hexdigest()
REAL_REPORTS = Path(__file__).resolve().parents[4].parent / 'reports' / 'external_bench_20261021'
DESIGN_FILES = ('EXTERNAL_BENCH_DESIGN_FABLE51.md', 'DECISIONS_20261021.md', 'DEVIATION_IDS.md') + tuple(
    f'AMENDMENT_EXT_00{i}_FABLE51.md' for i in range(1, 10))
LEDGER_REL = Path('reports') / 'results' / 'RUN_LEDGER.json'


def iso(sec: float) -> str:
    return (T0 + dt.timedelta(seconds=sec)).isoformat()


def write(p: Path, obj) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1), encoding='utf-8')
    return p


def signed(r: dict) -> dict:
    return signing.sign(json.loads(json.dumps(r)))


def _cum(i, wall, gpu, tok=1000, ok=True):
    return {'screens': i + 1, 'llm_tokens': tok * (i + 1), 'llm_usd': 0.0, 'gpu_hours': gpu, 'wall_seconds': wall,
            'accounting_complete': ok}


def _calls(run_id: str, n: int, tail_usd: float = 0.0) -> list[dict]:
    """One complete LLM call record (1000 tokens, $0) before each of n rows, as SearchSession / the basic agent log it;
    tail_usd > 0: a final call after the last row (0 tokens) charging it."""
    out = [{'turn': k + 1, 'role': 'proposer', 'call_id': f'{run_id}-call{k}', 'llm_tokens': 1000, 'llm_usd': 0.0,
            'accounting_complete': True} for k in range(n)]
    if tail_usd:
        out.append({'turn': n + 1, 'role': 'proposer', 'call_id': f'{run_id}-tail', 'llm_tokens': 0, 'llm_usd': tail_usd,
                    'accounting_complete': True})
    return out


def _events(d: Path, run_id: str, n: int, tail: bool = False) -> None:
    """An InfraRSI search's events.jsonl: each call before the screen it precedes (the explicit call-to-row boundary)."""
    ev = []
    for k in range(n):
        ev += [{'i': 2 * k, 'kind': 'llm_call', 'payload': {'call_id': f'{run_id}-call{k}', 'llm_tokens': 1000}},
               {'i': 2 * k + 1, 'kind': 'screen', 'payload': {'index': k}}]
    if tail:
        ev.append({'i': 2 * n, 'kind': 'llm_call', 'payload': {'call_id': f'{run_id}-tail', 'llm_tokens': 0}})
    d.mkdir(parents=True, exist_ok=True)
    (d / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in ev), encoding='utf-8')


def f1_uc_ref(ref_dir: Path, att: str = 'uc') -> dict:
    """The U+C reference record (X6: uc_ref.json = {'kind': 'UC-REF', 'receipt_attempt', 'receipt'}) with a signed
    receipt; returns the receipt (each F1 arm keeps its copy as receipts/c0.json)."""
    r = signed({'schema': 'ext-receipt-v1', 'env': 'flashnext', 'task_id': 'F1', 'arm': 'UC-REF', 'run_id': ref_dir.name,
                'attempt_id': att, 'candidate_id': 'c0', 'purpose': 'reference', 'valid': True,
                'cost': {'gpu_seconds': 7200.0, 'wall_seconds': 225.0, 'accounting_complete': True}})
    write(ref_dir / 'uc_ref.json', {'kind': 'UC-REF', 'receipt_attempt': att, 'valid': True, 'receipt': r})
    return r


def f1_arm(arm_dir: Path, arm: str, uc: dict) -> None:
    """An F1 arm directory as the F1 seam tests need it: the trace and SearchSession's copy of the start reference."""
    write(arm_dir / 'trace.json', f1_trace(arm))
    write(arm_dir / 'receipts' / 'c0.json', uc)


def f1_trace(arm: str) -> dict:
    """A minimal sealed-shape F1 arm trace with consistent zero totals (the F1 seam tests stub X6's summary)."""
    return {'arm': arm, 'schema': 'infrarsi-search-trace-v1', 'mode': 'strong', 'screens': [], 'llm_calls': [], 'budget': {},
            'used': {'screens': 0, 'gpu_hours': 0.0, 'wall_seconds': 0.0, 'llm_tokens': 0, 'llm_usd': 0.0}}


class World:
    """A records root + a reports folder + RUN_LEDGER.json under tmp."""

    def __init__(self, tmp: Path):
        self.c_selected = {}
        self.tmp = Path(tmp)
        self.root = self.tmp / 'ext_bench'
        self.reports = self.tmp / 'reports'
        self.ledger = {'schema': 'ext-bench-run-ledger-v1',
                       'ENV-A': {'env': 'rebench_llmf', 'arms': {'A-IS': {'run_id': None, 'status': 'not run (D1)'}},
                                 'node_refs': {}, 'block': {'run_id': None, 'node': 'WINH1'}},
                       'ENV-B': {'env': 'optico'}, 'ENV-C': {'env': 'autoresearch', 'node': 'WINH0', 'arms': {}, 'locked': {}},
                       'F1': {'env': 'flashnext', 'references': {'UC-REF': None, 'F-HT': None},
                              'arms': {'F-PROMPTS': None, 'F-OPTICO': None, 'F-A0': None, 'F-RND': None}}}

    def save_ledger(self) -> Path:
        return write(self.tmp / LEDGER_REL, self.ledger)

    # ------------------------------------------------------------------ ENV-A
    def a_receipt(self, run_id, att, cid, *, arm, purpose='screen', t=200.0, nd=4.0, judge_valid=True, rejections=(),
                  g4=None, node='WINH3', data_prep=8.9, digest=None, gpu_s=800.0, start=0.0, runner_error=None):
        valid = judge_valid and not rejections and runner_error is None
        reports = [{'id': 'G7', 'norm_diff': nd, 'threshold': 5.0, 'enforced': arm == 'A-ISW'}]
        if judge_valid:
            reports.append({'id': 'G3', 'trained_vs_base': 23.8, 'ref_vs_base': 23.8, 'ratio': 1.0, 'threshold': 0.5})
        if g4 is not None:
            reports.append({'id': 'G4', 'threshold': 0.5, 'refprime_cached': False, **g4})
        r = {'schema': 'ext-receipt-v1', 'env': 'rebench_llmf', 'task_id': 'ENV-A', 'arm': arm, 'run_id': run_id,
             'attempt_id': att, 'candidate_id': cid, 'purpose': purpose, 'node': node, 'gpus': [0, 1, 2, 3],
             'world_size': 4, 'valid': valid, 'judge_valid': judge_valid,
             'invalid_reasons': [f'guard:{g}: {d}' for g, d in rejections],
             'metric': {'name': 'exec_time_s', 'value': t if valid else None, 'direction': 'min', 'raw': {'norm_diff': nd}},
             'judge': {'message': {'maximum_allowed_norm_diff': 5.3, 'execution_time': t, 'norm_diff': nd},
                       'score': math.log(t) if judge_valid else None},
             'guard_rejections': [{'id': g, 'detail': d} for g, d in rejections], 'guard_reports': reports,
             'phases': {'data_prep': data_prep, 'train': t - data_prep - 27.0, 'convert': 27.0, 'finetune_total': t},
             'candidate_digest': digest or f'dg-{cid}', 'cost': {'gpu_seconds': gpu_s, 'wall_seconds': gpu_s / 4,
                                                                  'accounting_complete': True},
             'utc_start': iso(start), 'utc_end': iso(start + 300)}
        if runner_error:
            r['runner_error'] = runner_error
        return signed(r)

    def a_arm(self, arm, run_id, *, node, screens, locked=None, stop='finish', started=0.0, accounting=True,
              llm=None, ledger=True, wall_per_screen=300.0):
        """screens: [(cid, t, valid)]; locked: {'t', 'nd', 'rejections', 'g4', 'judge_valid', 'digest_ok'} or None."""
        d = self.root / 'rebench_llmf' / run_id
        scr, wall = [], 0.0
        for i, (cid, t, valid) in enumerate(screens):
            att = f'{run_id}-{cid}'
            write(d / 'receipts' / f'{att}.json', self.a_receipt(run_id, att, cid, arm=arm, t=t, judge_valid=valid, node=node,
                                                              data_prep=24.0 if i == 0 else 8.9, start=started + wall,
                                                              gpu_s=0.33 * 3600))
            wall += wall_per_screen
            scr.append({'index': i, 'candidate_id': cid, 'attempt_id': att, 'valid': valid, 'step_time': t if valid else None,
                        'cum': _cum(i, wall, 0.33 * (i + 1), tok=0 if arm == 'A-RND' else 1000, ok=accounting)})
        good = [s for s in scr if s['valid']]
        best = min(good, key=lambda s: s['step_time']) if good else None
        n_tok = 0 if arm == 'A-RND' else 1000
        trace = {'schema': 'infrarsi-search-trace-v1', 'arm': arm, 'run_id': run_id, 'screens': scr,
                 'used': {'screens': len(scr), 'llm_tokens': n_tok * len(scr), 'llm_usd': 0.0, 'gpu_hours': 0.33 * len(scr),
                          'wall_seconds': wall + 60, 'accounting_complete': accounting},
                 'budget': {'screens': 60, 'llm_tokens': 0 if arm == 'A-RND' else 1_000_000_000, 'wall_seconds': 57_600,
                            'gpu_hours': 34.0},
                 'llm_calls': [] if arm == 'A-RND' else _calls(run_id, len(scr)),
                 'stop_reason': stop, 'started_utc': iso(started), 'ended_utc': iso(started + wall + 60),
                 'selected': best and best['candidate_id'], 'accounting_complete': accounting}
        lk_summary, att_l = {}, None
        if locked is not None and best is not None:
            att_l = f'{run_id}-locked'
            digest = f'dg-{best["candidate_id"]}' if locked.get('digest_ok', True) else 'dg-other'
            r = self.a_receipt(run_id, att_l, 'locked', arm=arm, purpose='score', t=locked['t'], nd=locked.get('nd', 4.4),
                               judge_valid=locked.get('judge_valid', True), rejections=locked.get('rejections', ()),
                               g4=locked.get('g4', {'cand_move': 29.1, 'ref_move': 29.1, 'ratio': 1.0}), node=node,
                               digest=digest, start=started + wall + 120)
            write(d / 'receipts' / f'{att_l}.json', r)
            lk_summary = {'valid': r['valid'], 'execution_time': locked['t'], 'norm_diff': locked.get('nd', 4.4),
                          'guard_rejections': [g for g, _ in locked.get('rejections', ())]}
        if arm == 'A-ISW':
            _events(d, run_id, len(scr))
        llm = llm if llm is not None else ({'llm_calls': 10, 'llm_tokens_total': 50_000, 'llm_tokens_uncached': 40_000,
                                            'llm_wall_hours': 0.1, 'llm_gpu_hours_h20': 0.4, 'llm_usd': 0}
                                           if arm != 'A-RND' else {'llm_calls': 0, 'llm_tokens_total': 0})
        t8 = best['step_time'] if best else None
        if arm == 'A-NAT':
            res = {**trace, 'schema': 'ext-basic-agent-trace-v1', 'restart_of': 'anat_a', 'llm': llm, 't_best_search_8h': t8,
                   'extra': {'locked': ({'locked': {**lk_summary, 'digest_matches_selection': locked.get('digest_ok', True)},
                                         'selected': {'candidate_id': best['candidate_id']}, 'attempt_id': att_l}
                                        if att_l else {'locked': None, 'selected': None, 'reason': 'no valid score'}),
                             'loop_detector': {'rule': '>= 12 in a row or >= 30 in the last 60'},
                             'n_identical_max': 2, 'restart_of': 'anat_a'}}
        else:
            res = {'arm': arm, 'trace': trace, 'llm': llm, 't_best_search_8h': t8,
                   'locked': {'locked': lk_summary, 'selected': best and best['candidate_id'], 'attempt_id': att_l}}
        write(d / 'result.json', res)
        if ledger:
            self.ledger['ENV-A']['arms'][arm] = {'run_id': run_id, 'node': node}
        return res

    def node_ref(self, node, t, *, run_id=None, block_dir='blk_x', nd=4.2):
        run_id = run_id or f'nref_{node.lower()}'
        d = self.root / 'rebench_llmf' / block_dir
        att = f'nr-{node}'
        write(d / 'receipts' / f'{att}.json', self.a_receipt(run_id, att, 'node_ref', arm='A-RND', purpose='score', t=t,
                                                             nd=nd, node=node))
        write(d / f'node_ref_{node}.json', {'kind': 'node_ref', 'node': node, 'run_id': run_id,
                                             'attempts': [{'attempt_id': att, 't': t, 'void': False}],
                                             't_S0_node': t, 'measured': True})
        self.ledger['ENV-A']['node_refs'][node] = run_id

    def block(self, t_s0, t_by_arm, *, run_id='blk_x', fallback=(), nd_s0=4.0):
        d = self.root / 'rebench_llmf' / run_id
        runs = [{'name': 'S0', 'attempt_id': 'blk-S0', 't': t_s0, 'void': False}]
        write(d / 'receipts' / 'blk-S0.json', self.a_receipt(run_id, 'blk-S0', 'block', arm='A-RND', purpose='score', t=t_s0,
                                                             nd=nd_s0, node='WINH1'))
        arms = {}
        for arm, t in t_by_arm.items():
            if arm in fallback:
                arms[arm] = {'t_B': t, 'label': '(fallback)', 'source': 'fallback: t_locked x t_S0,B / t_S0,node'}
                continue
            att = f'blk-{arm}'
            write(d / 'receipts' / f'{att}.json', self.a_receipt(run_id, att, 'block', arm=arm, purpose='score', t=t,
                                                                 node='WINH1'))
            runs.append({'name': arm, 'attempt_id': att, 't': t, 'void': False})
            arms[arm] = {'t_B': t, 'label': '', 'source': 'block', 'block_attempt': att}
        write(d / 'block_A.json', {'kind': 'block', 'runs': runs, 'block_void_reruns': 0,
                                   'values': {'t_S0_B': t_s0, 't_S0_B_note': 'measured', 'arms': arms}})
        self.ledger['ENV-A']['block'] = {'run_id': run_id, 'node': 'WINH1'}
        self.ledger['ENV-A']['node_refs']['WINH1'] = 'block S0'

    def env_a_complete(self, *, isw_t=100.0, nat_t=150.0, rnd_t=140.0, isw_screen=110.0, nat_screen=160.0,
                       rnd_screen=150.0, isw_locked=None, nat_locked=None, rnd_locked=None):
        self.node_ref('WINH3', 216.0)
        self.node_ref('WINH4', 216.0)
        self.a_arm('A-NAT', 'anat_x', node='WINH1', screens=[('score01', 216.0, True), ('score02', nat_screen, True)],
                   locked=nat_locked or {'t': nat_t + 1}, stop='budget:wall_seconds', started=0)
        self.a_arm('A-ISW', 'aisw_x', node='WINH3', screens=[('c0', 216.0, True), ('c1', isw_screen, True)],
                   locked=isw_locked or {'t': isw_t + 1}, started=0)
        self.a_arm('A-RND', 'arnd_x', node='WINH4', screens=[('c0', 216.0, True), ('c1', rnd_screen, True)],
                   locked=rnd_locked or {'t': rnd_t + 1}, stop='budget:screens', started=0)
        self.block(216.0, {'A-NAT': nat_t, 'A-ISW': isw_t, 'A-RND': rnd_t})

    # ------------------------------------------------------------------ ENV-B
    def b_receipt(self, w, run_id, att, cid, *, arm, purpose, T, valid=True, params=None, native=None, oom=False, start=0.0,
                  runner_error=None, gpu_s=1200.0):
        wl = OW.get(w)
        r = {'schema': 'ext-receipt-v1', 'env': 'optico', 'task_id': wl.task_id, 'arm': arm, 'run_id': run_id,
             'attempt_id': att, 'candidate_id': cid, 'purpose': purpose, 'node': 'WINH4', 'gpus': list(range(8)),
             'world_size': 8, 'valid': valid, 'judge_valid': valid,
             'metric': {'name': 'iter_time_s', 'value': (10_000.0 / T) if valid else None, 'direction': 'min',
                        'raw': {'tflops_per_gpu': T if valid else None}},
             'params': dict(params or wl.defaults), 'oom': oom, 'guard_rejections': [], 'guard_reports': [],
             'cost': {'gpu_seconds': gpu_s, 'wall_seconds': 150.0, 'accounting_complete': True},
             'utc_start': iso(start), 'utc_end': iso(start + 150)}
        if native is not None:
            r['native'] = native
        if runner_error:
            r['runner_error'] = runner_error
        return signed(r)

    def b_single(self, w, arm, run_id, T, *, params=None):
        purpose, cid = ('b_def', 'def') if arm == 'B-DEF' else ('b_ed', 'bed')
        att = f'{run_id}-{cid}'
        write(self.root / 'optico' / run_id / 'receipts' / f'{att}.json',
              self.b_receipt(w, run_id, att, cid, arm=arm, purpose=purpose, T=T, params=params))
        self.ledger['ENV-B'].setdefault(w, {'node': 'WINH4', 'locked': {}})[arm] = run_id

    def b_search(self, w, arm, run_id, Ts, *, best_params=None, stop='budget:screens', accounting=True):
        """Ts: TFLOPS per screen (None = failed run); screen 0 is the default start. Ledger-consistent like the real
        producers: each screen's receipt charges 3600 GPU-s (the rows' 1.0 GPUh each); an LLM arm has one complete call
        (1000 tokens, $0) before each row - B-NAT (mode native) by `after_screen`, InfraRSI (mode strong) by the
        llm_call / screen order of events.jsonl; B-RND has no LLM use."""
        wl = OW.get(w)
        d = self.root / 'optico' / run_id
        scr, wall = [], 0.0
        for i, T in enumerate(Ts):
            cid = f'it{i + 1}' if arm == 'B-NAT' else f'c{i}'
            att = f'{run_id}-{cid}'
            params = dict(wl.defaults) if i == 0 else dict(best_params or {**wl.defaults, 'MICRO_BATCH_SIZE': 2})
            native = ({'kind': 'default' if i == 0 else 'new', 'row_index': i, 'retry_of': None, 'native_attempt': run_id}
                      if arm == 'B-NAT' else None)
            purpose = 'native' if arm == 'B-NAT' else 'screen'
            write(d / 'receipts' / f'{att}.json', self.b_receipt(w, run_id, att, cid, arm=arm, purpose=purpose, T=T or 1.0,
                                                              valid=T is not None, params=params, native=native,
                                                              oom=T is None, start=wall, gpu_s=3600.0))
            wall += 400.0
            scr.append({'index': i, 'candidate_id': cid, 'attempt_id': att, 'valid': T is not None,
                        'step_time': (10_000.0 / T) if T else None,
                        'cum': _cum(i, wall, 1.0 * (i + 1), tok=0 if arm == 'B-RND' else 1000, ok=accounting)})
        best_i = max((i for i, T in enumerate(Ts) if T), key=lambda i: Ts[i])
        sel = dict(wl.defaults) if best_i == 0 else dict(best_params or {**wl.defaults, 'MICRO_BATCH_SIZE': 2})
        budget = OT.make_task(w, arm).budget
        trace = {'schema': 'infrarsi-search-trace-v1', 'arm': arm, 'run_id': run_id, 'task_id': wl.task_id, 'screens': scr,
                 'used': {'screens': len(scr), 'llm_tokens': 0 if arm == 'B-RND' else 1000 * len(scr), 'llm_usd': 0.0,
                          'gpu_hours': 1.0 * len(scr), 'wall_seconds': wall, 'accounting_complete': accounting},
                 'budget': budget, 'stop_reason': stop, 'started_utc': iso(0), 'ended_utc': iso(wall), 'selected_config': sel,
                 'invalid_proposals': {}, 'accounting_complete': accounting, 'mode': 'native' if arm == 'B-NAT' else 'strong',
                 'llm_calls': []}
        if arm == 'B-NAT':
            trace['llm_calls'] = [{'call': k, 'after_screen': k - 1 if k else None, 'near_row_boundary': False,
                                   'llm_tokens': 1000, 'llm_usd': 0.0, 'accounting_complete': True} for k in range(len(scr))]
        elif arm != 'B-RND':
            trace['llm_calls'] = [{'turn': k + 1, 'role': 'proposer', 'call_id': f'{run_id}-call{k}', 'llm_tokens': 1000,
                                   'llm_usd': 0.0, 'accounting_complete': True} for k in range(len(scr))]
            ev = []
            for k in range(len(scr)):
                ev += [{'i': 2 * k, 'kind': 'llm_call', 'payload': {'call_id': f'{run_id}-call{k}', 'llm_tokens': 1000}},
                       {'i': 2 * k + 1, 'kind': 'screen', 'payload': {'index': k}}]
            (d / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in ev), encoding='utf-8')
        write(d / 'trace.json', trace)
        if arm == 'B-NAT':
            write(d / 'result.json', {'schema': 'infrarsi-ext-optico-native-v1', 'status': 'done', 'llm': {'refused': 0},
                                      'best_success_row': {'attempt_id': scr[best_i]['attempt_id'], 'tflops_per_gpu': Ts[best_i]}})
        self.ledger['ENV-B'].setdefault(w, {'node': 'WINH4', 'locked': {}})[arm] = run_id
        return sel

    def b_locked(self, w, arm, run_id, T_default, T_cand, *, params=None, cand_valid=True):
        wl = OW.get(w)
        d = self.root / 'optico' / run_id
        halves = {}
        for i, (who, T, valid, p) in enumerate((('default', T_default, True, wl.defaults),
                                                ('candidate', T_cand, cand_valid, params or wl.defaults))):
            att = f'{run_id}-{who}'
            write(d / 'receipts' / f'{att}.json', self.b_receipt(w, run_id, att, f'locked-{who}', arm=arm, purpose='locked',
                                                              T=T, valid=valid, params=p, start=i * 200.0))
            halves[who] = {'attempt_id': att, 'valid': valid, 'tflops_per_gpu': T if valid else None}
        write(d / 'locked.json', {'arm': arm, 'order': ['default', 'candidate'], 'default': halves['default'],
                                  'candidate': halves['candidate'], 'invalid_accept_locked': 0 if cand_valid else 1})
        self.ledger['ENV-B'].setdefault(w, {'node': 'WINH4', 'locked': {}}).setdefault('locked', {})[arm] = run_id

    def env_b_workload(self, w, *, is_T=330.0, nat_T=300.0, rnd_T=310.0, is_auc_T=330.0, nat_auc_T=300.0,
                       is_valid=True, base=228.0, tag=''):
        """B-DEF, B-ED, B-NAT, B-IS, B-RND with locked pairs; defaults at `base` (+ small noise) so S_w has n >= 4."""
        p2 = {**OW.get(w).defaults, 'MICRO_BATCH_SIZE': 2}
        self.b_single(w, 'B-DEF', f'bdef_{w}{tag}', base)
        self.b_single(w, 'B-ED', f'bed_{w}{tag}', base * 1.3, params=p2)
        self.b_locked(w, 'B-ED', f'lk_bed_{w}{tag}', base + 1.0, base * 1.3, params=p2)
        for arm, T, aucT, d0 in (('B-NAT', nat_T, nat_auc_T, -1.0), ('B-IS', is_T, is_auc_T, 2.0), ('B-RND', rnd_T, rnd_T, 0.5)):
            rid = f'{arm.lower().replace("-", "")}_{w}{tag}'
            sel = self.b_search(w, arm, rid, [base + d0, None, aucT], best_params=p2)
            self.b_locked(w, arm, f'lk_{rid}', base - d0, T, params=sel, cand_valid=is_valid or arm != 'B-IS')

    # ------------------------------------------------------------------ ENV-C
    def c_receipt(self, run_id, att, cid, *, arm, purpose='screen', bpb=0.996, valid=True, steps=936, failure=None,
                  rejections=(), start=0.0, runner_error=None, train_sha=None, deviations=DAR, knob_sha=None):
        r = {'schema': 'ext-receipt-v1', 'env': 'autoresearch', 'task_id': 'ENV-C', 'arm': arm, 'run_id': run_id,
             'deviations': list(deviations), 'files': {'train.py': train_sha or 'sha-base-train', 'prepare.py': 'sha-prep'},
             'attempt_id': att, 'candidate_id': cid, 'purpose': purpose, 'node': 'WINH0', 'gpus': [0], 'world_size': 1,
             'valid': valid, 'judge_valid': valid and not rejections,
             'metric': {'name': 'val_bpb', 'value': bpb if valid else None, 'direction': 'min', 'raw': {'num_steps': steps}},
             'summary': {'num_steps': steps, 'tok_per_sec': 1_600_000, 'mfu_percent': 39.0, 'peak_vram_mb': 45060.2},
             'guard_rejections': [{'id': g, 'detail': 'x'} for g in rejections],
             'guard_reports': [{'id': 'judge_loader_equivalence', 'abs_dev': 0.0, 'ok': True}],
             'failure_class': failure, 'cost': {'gpu_seconds': 450.0, 'wall_seconds': 450.0, 'accounting_complete': True},
             'utc_start': iso(start), 'utc_end': iso(start + 450)}
        if runner_error:
            r['runner_error'] = runner_error
        if knob_sha:
            r['knob_space_sha256'] = knob_sha
        return signed(r)

    def c_anchor(self, bpb=0.9963646606577251, steps=936, run='x1check_x', att='a0c0ffee00001'):
        write(self.root / 'autoresearch' / run / 'receipts' / f'{att}.json',
              self.c_receipt(run, att, 'c0-h2d_sync', arm='C-BASE', bpb=bpb, steps=steps))
        self.ledger['ENV-C']['anchor'] = f'{att} ({run}, D-AR-7 baseline on WINH0)'

    def c_calib(self, steps=933, run='calib_x'):
        write(self.root / 'autoresearch' / run / 'runner' / 'receipts' / 'calib0001.json',
              self.c_receipt(run, 'calib0001', 'c0', arm='C-BASE', steps=steps))
        self.ledger['ENV-C']['calibration'] = run

    def c_arm(self, arm, run_id, bpbs, *, failures=(), accounting=True, extra=None, stop='budget:wall_seconds'):
        """bpbs: val_bpb per experiment (None = crashed); experiment 0 is the start (baseline train.py)."""
        d = self.root / 'autoresearch' / run_id
        scr, wall, cands = [], 0.0, []
        knob = KNOB_SHA if arm == 'C-RND' else None
        for i, b in enumerate(bpbs):
            cid, att = f'c{i}', f'{run_id}-c{i}'
            fc = failures[i] if i < len(failures) else None
            write(d / 'runner' / 'receipts' / f'{att}.json',
                  self.c_receipt(run_id, att, cid, arm=arm, bpb=b or 1.0, valid=b is not None,
                                 failure=fc if b is None else None, start=wall,
                                 train_sha='sha-base-train' if i == 0 else f'sha-{run_id}-{cid}', knob_sha=knob))
            cands.append({'id': cid, 'config_digest': f'cfg-{run_id}-{cid}'})
            wall += 450.0
            scr.append({'index': i, 'candidate_id': cid, 'attempt_id': att, 'valid': b is not None, 'step_time': b,
                        'cum': _cum(i, wall, 0.125 * (i + 1), tok=0 if arm == 'C-RND' else 1000, ok=accounting)})
        good = [x for x in scr if x['valid']]
        sel = min(good, key=lambda x: x['step_time'])['candidate_id'] if good else None
        self.c_selected[arm] = 'sha-base-train' if sel in (None, 'c0') else f'sha-{run_id}-{sel}'
        if arm == 'C-RND':
            extra = {'knob_space_sha256': KNOB_SHA, **(extra or {})}
        name = 'agent/trace.json' if arm.startswith('C-NAT') else 'trace.json'
        write(d / name, {'schema': 'ext-basic-agent-trace-v1' if arm.startswith('C-NAT') else 'infrarsi-search-trace-v1',
                         'selected': sel, 'candidates': cands,
                         'arm': arm, 'run_id': run_id, 'screens': scr, 'stop_reason': stop, 'ended_utc': iso(wall),
                         'used': {'screens': len(scr), 'llm_tokens': 0 if arm == 'C-RND' else 1000 * len(scr),
                                  'llm_usd': 0.0 if arm == 'C-RND' else 1.0,
                                  'gpu_hours': 0.125 * len(scr), 'wall_seconds': wall, 'accounting_complete': accounting},
                         'budget': {'screens': 36, 'llm_tokens': 0 if arm == 'C-RND' else 3_000_000},
                         'llm_calls': [] if arm == 'C-RND' else _calls(run_id, len(scr), tail_usd=1.0),
                         'extra': extra or {}, 'accounting_complete': accounting})
        if arm == 'C-IS':
            _events(d, run_id, len(scr), tail=True)
        write(d / 'result.json', {'stop_reason': stop})
        self.ledger['ENV-C']['arms'][arm] = run_id

    def c_locked(self, arm, run_id, bpb_base, bpb_cand, *, cand_valid=True):
        d = self.root / 'autoresearch' / run_id
        out = {'order': ['baseline', 'candidate']}
        for i, (who, cid, b, valid) in enumerate((('baseline', 'lock-base', bpb_base, True),
                                                  ('candidate', 'lock-cand', bpb_cand, cand_valid))):
            att = f'{run_id}-{cid}'
            write(d / 'runner' / 'receipts' / f'{att}.json',
                  self.c_receipt(run_id, att, cid, arm=arm, purpose='score', bpb=b, valid=valid, start=i * 500.0,
                                 train_sha=self.c_selected.get(arm) if who == 'candidate' else 'sha-base-train'))
            out[who] = {'attempt_id': att, 'valid': valid, 'val_bpb': b if valid else None}
        write(d / 'result.json', out)
        self.ledger['ENV-C']['locked'][arm] = run_id

    def c_r2_receipts(self):
        """The two signed X1 runs that R2's ENV-C lines (b) upstream and (c) D-AR-7 cite (R2_R3_sources.json)."""
        for att, bpb, steps, dev in (('a7d6d5a710811', 1.0228085860554135, 939, DAR[:-1]),
                                     ('a21767e9dbd4f', 0.9963646606577251, 936, DAR)):
            write(self.root / 'autoresearch' / 'x1check_20261021' / 'receipts' / f'{att}.json',
                  self.c_receipt('x1check_20261021', att, 'c0', arm='C-BASE', bpb=bpb, steps=steps, deviations=dev))

    def env_c_complete(self, *, is_locked=0.990, nat_locked=0.994, rnd_locked=0.995, is_valid=True):
        self.c_anchor()
        self.c_calib()
        self.c_arm('C-IS', 'cis_x', [0.9964, 0.991])
        self.c_arm('C-NAT-infra', 'cnatinfra_x', [0.995, None], failures=('', 'cuda_graph'))
        self.c_arm('C-NAT-free', 'cnatfree_x', [0.993, 0.985])
        self.c_arm('C-RND', 'crnd_x', [0.9964, 0.9958, None], failures=('', '', 'oom'),
                   extra={'space_coverage': 0.0426, 'space_size': 47})
        for arm, rid, b, v in (('C-IS', 'lk_cis_x', is_locked, is_valid), ('C-NAT-infra', 'lk_cnatinfra_x', nat_locked, True),
                               ('C-NAT-free', 'lk_cnatfree_x', 0.986, True), ('C-RND', 'lk_crnd_x', rnd_locked, True)):
            self.c_locked(arm, rid, 0.9965, b, cand_valid=v)


def reports_fixture(world: World, *, issues: str | None = None):
    """A minimal reports folder: REPRO_ISSUES.md (the real one by default), REPRO_RESULTS.md, DEVIATION_IDS.md."""
    real = REAL_REPORTS
    rd = world.reports
    (rd / 'design').mkdir(parents=True, exist_ok=True)
    for name in DESIGN_FILES:                          # the cited authorities (a formal summary requires them)
        src = real / 'design' / name
        (rd / 'design' / name).write_text(src.read_text(encoding='utf-8') if src.is_file() else f'# {name}\n',
                                          encoding='utf-8')
    for name in ('REPRO_ISSUES.md', 'REPRO_RESULTS.md'):
        src = real / name
        text = src.read_text(encoding='utf-8') if src.is_file() else ''
        if name == 'REPRO_ISSUES.md' and issues is not None:
            text = issues
        (rd / name).write_text(text, encoding='utf-8')
    return rd
