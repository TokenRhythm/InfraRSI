"""F1 task contracts (AMENDMENT_EXT_003 Q1, Q5): one EnvTask per arm and reference screen, the facts and the task view.

Every arm: env 'flashnext', world 32, metric step_time_s (minimized), margin {'kind': 'rsd', 'floor': 0.03, 'k': 3},
tau ln 1.10, start = U+C (the S_F1 start point; its reference screen is every arm's start receipt), the same switch
space S_F1 (switch_space.json, hash in every receipt), the same judge and gate file. Budgets = the Q5 table.

Task view (every arm, Q1): HardwareProfile, workload and fixed-layout facts, the U+C reference receipt summary, the
switch table with one neutral sentence per switch (switch_space.task_view: no provenance), the gates, the budget. No
task view names a recipe or shows a stored timing (task_view_leaks() is the test).

Setting 2 (code edits): F-A0-CODE = F-A0's method, budget, judge, start, gate file and margin with the code space C_F1
(code_space.py) open on top of S_F1: action space knobs+patch, mutable globs = the speed-path runtime modules
(code/<name>), 300 changed lines; its frozen files are every other file of the base tree. Its arm index 4 is not in
ARM_INDEX (the switch-only window of setting 1 keeps exactly its four arms); even, so its locked pair runs U+C first as
F-A0's. The four switch-only arms' contracts are byte-for-byte unchanged.

The v4 extras (FABLE_FN4_RESUME_20261103, FABLE_V4_BUDGET_20261102 section 1): F-OPEN4-BCOMP = F-OPEN-BCOMP's contract
run from the v4 tree (pc_code_o2) at F-OPEN-AR's Q5 entry; its task alone carries
the v4 method's own caps as raw['manifest'] (V4_MANIFEST: where the v4 trees' finalist_base.spec_budget reads a spec's
caps). Labelled extras (V4_LABEL), outside the pre-registered wave's counts. Every other arm's contract is unchanged.

The same-money extras (user 11-03 17:4x): F-BUD-BCOMP = the same contract run from the v4.1 tree
(pc_code_o2b; the v4.2 tree was dropped by the user 11-04) until it has spent what F-OPEN-AR spent on
this job (23.71 USD; its 3 M token cap kept), 105 screens as a safety cap, their caps BUD_MANIFEST (the v4 per-screen
parity). Labelled extras (BUD_LABEL;
EXTRA_ARMS = the v4 and the same-money extras), outside the pre-registered wave's counts.
"""
from __future__ import annotations

import json
import math
import re

from ..common.env_task import SCHEMA_TASK, EnvTask
from . import code_space as C
from . import deviations as dv
from . import guards as G
from . import judge as J
from . import parse as P
from . import switch_space as S

ENV = 'flashnext'
TASK_ID = 'ENV-F1'
METRIC = 'step_time_s'
TAU = math.log(1.10)
MARGIN = {'kind': 'rsd', 'floor': 0.03, 'k': 3}
WORLD = 32
SEED = 42
ARM_INDEX = {'F-PROMPTS': 0, 'F-OPTICO': 1, 'F-A0': 2, 'F-RND': 3}
# AMENDMENT_EXT_003 Q5 table. llm_usd 0 for F-RND is recorded as the arm's cap but not passed to the ledger (a 0 USD cap
# would exhaust it at once; the arm makes no LLM call). gpu_hours = 12 x 16 worst case (32 GPUs x 1,800 s).
ARMS = {
    'F-PROMPTS': {'screens': 3, 'llm_tokens': 200_000, 'llm_usd': 10.0, 'wall_seconds': 7200.0, 'gpu_hours': 48.0,
                  'llm': True, 'method': 'Opus 5.5 CLI; one call, no retry; top-3 over S_F1; no skills'},
    'F-OPTICO': {'screens': 12, 'llm_tokens': 1_500_000, 'llm_usd': 40.0, 'wall_seconds': 18000.0, 'gpu_hours': 192.0,
                 'llm': True, 'method': 'Opus 5.5 CLI; optico_style_L0 (4 prompt classes + pool), D-OC-15; no skills'},
    'F-A0': {'screens': 12, 'llm_tokens': 1_500_000, 'llm_usd': 40.0, 'wall_seconds': 18000.0, 'gpu_hours': 192.0,
             'llm': True, 'method': 'Opus 5.5 CLI; A0 strong mode, skills tagged moe/gdn/multi-node, verifier on, tools off'},
    'F-RND': {'screens': 12, 'llm_tokens': 0, 'llm_usd': 0.0, 'wall_seconds': 18000.0, 'gpu_hours': 192.0,
              'llm': False, 'method': 'seed-42 sampler over S_F1 (switch_space.RandomSwitchSampler); not an LLM arm'},
    # setting 2: F-A0's budget, driver and package with code edits (C_F1) open on top of S_F1
    'F-A0-CODE': {'screens': 12, 'llm_tokens': 1_500_000, 'llm_usd': 40.0, 'wall_seconds': 18000.0, 'gpu_hours': 192.0,
                  'llm': True, 'method': 'Opus 5.5 CLI; A0 strong mode, switches + code edits (L0 + L1 over C_F1), skills '
                                         'tagged moe/gdn/multi-node, verifier on, tools off'},
    # the open-space wave (FABLE_OPEN_DESIGN_20261102 sections 6 F8 / 8): 20 screens, 21,600 s wall, 320 GPU-h, baseline
    # cap 40 USD / 3 M tokens; ours (bcomp) spend 20 calls / 16 USD inside (the open tree's llmopt rate). Every
    # arm starts from U+C; the search strategy is the open tree's, and C_F1 is open (code=True) only when F9 is approved.
    'F-OPEN-BCOMP': {'screens': 20, 'llm_tokens': 3_000_000, 'llm_usd': 40.0, 'wall_seconds': 21600.0, 'gpu_hours': 320.0,
                     'llm': True, 'method': 'Opus 5.5 CLI; the open-tree search over S_F1 (+ C_F1 when code is open); '
                                            '20 calls / 16 USD inside the 40 USD cap; verifier on'},
    'F-OPEN-AR': {'screens': 20, 'llm_tokens': 3_000_000, 'llm_usd': 40.0, 'wall_seconds': 21600.0, 'gpu_hours': 320.0,
                  'llm': True, 'method': 'Opus 5.5 CLI; the native agent edits the mutable files and runs one screen at a '
                                         'time over S_F1 (+ C_F1 when code is open); 40 USD / 3 M token cap'},
    'F-OPEN-RB': {'screens': 20, 'llm_tokens': 3_000_000, 'llm_usd': 40.0, 'wall_seconds': 21600.0, 'gpu_hours': 320.0,
                  'llm': True, 'method': 'Opus 5.5 CLI; the native agent edits its solution files and scores over S_F1 '
                                         '(+ C_F1 when code is open); 40 USD / 3 M token cap'},
    'F-OPTICO-20': {'screens': 20, 'llm_tokens': 3_000_000, 'llm_usd': 40.0, 'wall_seconds': 21600.0, 'gpu_hours': 320.0,
                    'llm': True, 'method': 'Opus 5.5 CLI; optico_style_L0 (4 prompt classes + pool), D-OC-15; 20 screens, '
                                           '40 USD; configuration only (no code)'},
    'F-RND-20-s42': {'screens': 20, 'llm_tokens': 0, 'llm_usd': 0.0, 'wall_seconds': 21600.0, 'gpu_hours': 320.0,
                     'llm': False, 'method': 'seed-42 sampler over S_F1 (switch_space.RandomSwitchSampler); not an LLM arm'},
    'F-RND-20-s43': {'screens': 20, 'llm_tokens': 0, 'llm_usd': 0.0, 'wall_seconds': 21600.0, 'gpu_hours': 320.0,
                     'llm': False, 'method': 'seed-43 sampler over S_F1 (switch_space.RandomSwitchSampler); not an LLM arm'},
    # the v4 extras (FABLE_FN4_RESUME_20261103): bcomp as fixed after the diagnosis, at the baselines' budget on this
    # job = F-OPEN-AR's entry exactly; the v4 caps inside it are V4_MANIFEST's (the v4 trees spend up to 40 USD, not 16)
    'F-OPEN4-BCOMP': {'screens': 20, 'llm_tokens': 3_000_000, 'llm_usd': 40.0, 'wall_seconds': 21600.0, 'gpu_hours': 320.0,
                      'llm': True, 'method': 'Opus 5.5 CLI; the v4 open-tree search (fixed after the diagnosis) over S_F1 '
                                             '(+ C_F1 when code is open); the v4 caps inside the 40 USD / 3 M token cap; '
                                             'verifier on'},
    # the same-money re-runs (user 11-03 17:4x; v4.1 tree pc_code_o2b - the user dropped v4.2 11-04): ours until
    # they have spent what F-OPEN-AR spent on this job - its 23.71 USD and its 3 M token cap (AR stopped on the tokens);
    # screens ~1.5 x the v4 cells' rate (S_est 70) as a safety cap, the money is the stop
    'F-BUD-BCOMP': {'screens': 105, 'llm_tokens': 3_000_000, 'llm_usd': 23.71, 'wall_seconds': 72000.0, 'gpu_hours': 700.0,
                    'llm': True, 'method': 'Opus 5.5 CLI; the v4.1 open-tree search over S_F1 (+ C_F1 when code is open); '
                                           'the budget is what F-OPEN-AR spent here (23.71 USD / 3 M tokens); verifier on'},
}
CODE_ARMS = ('F-A0-CODE',)
CODE_ARM_INDEX = {'F-A0-CODE': 4}       # not in ARM_INDEX: the switch-only window keeps its four arms
# the open-space wave's arms (section 6 F8): outside ARM_INDEX (the setting-1 window) and CODE_ARMS (F-A0-CODE). Each is
# run by arms.run_f_is2 (bcomp / ar / rb), arms.run_f_optico (F-OPTICO-20) or arms.run_f_rnd (F-RND-20-s42/s43),
# and its locked pair by arms.run_locked; C_F1 is open per-run (arms.run_f_is2 code=), not by arm membership. The v4
# extras (V4_ARMS) and the same-money extras (BUD_ARMS) are appended: run as F-OPEN-BCOMP, outside the
# wave's counts.
OPEN_ARMS = ('F-OPEN-BCOMP', 'F-OPEN-AR', 'F-OPEN-RB', 'F-OPTICO-20', 'F-RND-20-s42', 'F-RND-20-s43',
             'F-OPEN4-BCOMP', 'F-BUD-BCOMP')
# Keep published arm indices and locked-pair order stable when an arm is retired.
OPEN_ARM_INDEX = {'F-OPEN-BCOMP': 6, 'F-OPEN-AR': 7, 'F-OPEN-RB': 8, 'F-OPTICO-20': 9,
                  'F-RND-20-s42': 10, 'F-RND-20-s43': 11, 'F-OPEN4-BCOMP': 12, 'F-BUD-BCOMP': 14}
V4_ARMS = ('F-OPEN4-BCOMP',)
V4_LABEL = 'fixed after the diagnosis, out-of-sample on Flash-Next'
# the same-money re-runs (v4.1): extras like the v4 ones (outside the window's counts), their own caps (BUD_MANIFEST)
BUD_ARMS = ('F-BUD-BCOMP',)
BUD_LABEL = 'v4.1 same-money re-run (budget = what F-OPEN-AR spent; post hoc, after the 11-03 protocol change)'
EXTRA_ARMS = V4_ARMS + BUD_ARMS


def extra_label(arm: str) -> str:
    return V4_LABEL if arm in V4_ARMS else BUD_LABEL


# the v4 method's own caps (FABLE_V4_BUDGET_20261102 section 1), as the open4 qa_trial specs carry them in their manifest
# (make_open4_specs.py); a v4 tree reads them from the task's raw spec only with the code track open (beyond ten screens
# its v4 parts apply). llm_usd and llm_tokens are the session's own (the Q5 entry).
V4_MANIFEST = {'search': {'budget': {'llm_calls': 60, 'report_usd': 16},
                          'caps': {'fix_calls_per_screen': 2, 'fix_calls': 20, 'patches_per_ask': 4, 'code_probes': 12}},
               'llm': {'max_usd_per_call': 8, 'timeout_s': 1200}}
# the same-money caps at 105 screens (the v4 per-screen parity: 3 calls, 1 fix call, 0.6 code probes per screen), as the
# qa_trial same-money specs carry them (scripts/make_bud_ours_specs.py)
BUD_MANIFEST = {'search': {'budget': {'llm_calls': 315, 'report_usd': 23.71},
                           'caps': {'fix_calls_per_screen': 2, 'fix_calls': 105, 'patches_per_ask': 4, 'code_probes': 63}},
                'llm': {'max_usd_per_call': 8, 'timeout_s': 1200}}
CODE_CLAIMS = ('switch-space search plus code edits of the declared speed-path runtime modules (C_F1) on the fixed '
               'production layout of Qwen3.8-Flash-Next (32 H100s), judged by the path\'s own 30-update window metric and '
               'E1 numerics against U+C; data, loss, optimizer, LR schedule, seed and layout frozen; not comparable '
               'with switch-only methods that cannot edit code; not unseen (A0\'s skills carry the human recipes), one '
               'data order, screen-level only, single seed.')
# Reference screens (plan A, before the arms; not arms): one screen each through the same runner and judge.
REFERENCES = {
    'UC-REF': {'point': 'start', 'judged_against': 'the stored U+C A cell rsi_perf_v1_c1_p03 (formal-54)',
               'numerics_role': 'reference'},
    'F-HT': {'point': 'S_match', 'judged_against': 'the U+C reference screen', 'numerics_role': 'candidate'},
}
SCREEN = {'run_limit': 30, 'seed': SEED, 'window': list(P.WINDOW), 'cap_s': G.CAP_S,
          'reference_cap_s_cold_cache': G.COLD_REFERENCE_CAP_S, 'fresh_process': True}
CLAIMS = ('switch-space search on the fixed production layout of Qwen3.8-Flash-Next (32 H100s), judged by the path\'s own '
          '30-update window metric and E1 numerics against U+C; not a layout search, not InfraRSI\'s own harness, not '
          'unseen (A0\'s skills carry the human recipes), one data order (no plan B), screen-level only, single seed.')
# Names a task view must never carry (Q1): registered recipe names and the stored formal-54 timings.
RECIPE_NAMES = ('S_match_full', 'S_match', 'S0', 'U_plus_C', 'U+C', 'Q_full', 'Qo_full', 'Qo')
STORED_TIMINGS = ('12.49', '12.694', '12.758', '10.608', '10.638', '8.535', '8.797', '8.429', '8.814', '8.959', '9.084',
                  '9.437', '8.857', '7.122', '7.166', '7.258', '7.576', '7.605', '7.662', '7.894', '8.078')


def frozen_files() -> dict:
    """The task's verified base tree: the 86-file variant runtime (as bundles carry it, code/<name>) and the read-only
    files every screen executes or reads on WINH (their absolute paths without the leading '/')."""
    m = G.load_manifest()
    if not m:
        raise RuntimeError(f'{G.MANIFEST_PATH} missing: the F1 variant layer is not frozen yet')
    files = {f'code/{n}': h for n, h in sorted(m['variant_runtime']['inventory'].items())}
    files.update({p.lstrip('/'): h for p, h in sorted(m['remote_read_only'].items())})
    return files


def knob_schema() -> dict:
    """S_F1 as a JSON schema (informational: the hooks validate with switch_space.validate)."""
    tv = S.task_view()
    props = {}
    for row in tv['switches']:
        enum = list(row['values']) + ([None] if row['name'] != 'implementation_stack' else [])
        props[row['name']] = {'enum': enum, 'description': row['description'], 'default': tv['start'][row['name']]}
    return {'type': 'object', 'properties': props, 'required': ['implementation_stack'], 'additionalProperties': False}


def facts() -> dict:
    """Workload and fixed-layout facts (F1.yaml, uc_contract.CONTRACT, the trainer's asserts); no timings."""
    return {
        'model': {'name': 'Qwen3.8-Flash-Next', 'key': 'flash-next-125b', 'family': 'hybrid QSA / Gated DeltaNet / MoE',
                  'decoder_layers': 48, 'mtp_layers': 1, 'bridge_type': 'qwen4_exp'},
        'data': {'name': 'LUNA-Q50-32K', 'kind': 'packed SFT (THD), 32,768-token packs', 'packs_per_update': 8,
                 'physical_tokens_per_update': 262144, 'order': 'frozen pack plan, no reshuffle (one data order)'},
        'objective': {'optimizer': 'AdamW (native), fp32 Adam moments, expert Adam v offloaded to host memory',
                      'lr_schedule': '5e-7 -> 5e-8 cosine over 1024 updates; a 30-update screen lies in the warm-up',
                      'loss': 'main CE + 0.1 x MTP CE + router aux (0.001)', 'precision': 'bf16 parameters, fp32 masters'},
        'fixed_layout': {'tp': 8, 'ep': 32, 'cp': 1, 'etp': 1, 'dp': 4, 'micro_batches_per_update': 2, 'seq_len': 32768,
                         'sequence_parallel': True, 'moe_dispatcher': 'all-to-all + DeepEP',
                         'recompute': 'full; block recompute when the attention prefix is retained',
                         'ple_tables': 'CPU-resident', 'source': 'asserted by the production trainer (not proposable)'},
        'screen': {'updates': 30, 'scoring_window': list(P.WINDOW), 'fresh_process': True, 'seed': SEED,
                   'metric': 'mean over updates 17..30 of the per-update max over the 32 ranks of the whole-update time',
                   'wall_cap_s': G.CAP_S},
        'hardware': {'gpu_type': 'H100-80GB', 'gpus': WORLD, 'nodes': 4, 'gpus_per_node': 8},
    }


def budget_for(arm: str) -> dict:
    a = ARMS[arm]
    b = {k: a[k] for k in ('screens', 'llm_tokens', 'wall_seconds', 'gpu_hours')}
    if a['llm']:
        b['llm_usd'] = a['llm_usd']
    return b


def make_task(arm: str = 'F-A0', *, code: bool = False) -> EnvTask:
    """The EnvTask of one arm (or reference). `code=True` opens the C_F1 code space on top of S_F1 for an open-wave arm
    (F-OPEN-* / F-OPEN4-* / F-OPTICO-20 / F-RND-20-*) exactly as it is always open for F-A0-CODE; the switch-only arms and
    F-A0-CODE are unaffected by the flag (the four setting-1 specs stay byte-for-byte). A v4 extra's task adds the v4 caps
    (raw['manifest'] = V4_MANIFEST), a same-money extra's its own (BUD_MANIFEST); no other task has a manifest."""
    if arm in ARMS:
        budget, role = budget_for(arm), 'arm'
    elif arm in REFERENCES:
        cap = G.COLD_REFERENCE_CAP_S
        budget, role = {'screens': 1, 'llm_tokens': 0, 'wall_seconds': cap + 600.0, 'gpu_hours': WORLD * cap / 3600.0}, 'reference'
    else:
        raise ValueError(f'unknown F1 arm or reference {arm!r} (known: {sorted(ARMS) + sorted(REFERENCES)})')
    gates = J.load_gates()
    spec = {'schema': SCHEMA_TASK, 'id': TASK_ID, 'env': ENV, 'split': 'external', 'arm': arm, 'role': role,
            'metric_name': METRIC, 'action_space': 'knobs', 'knob_schema': knob_schema(), 'frozen_files': frozen_files(),
            'mutable_globs': [], 'max_patch_lines': 0, 'budget': budget, 'margin': dict(MARGIN), 'tau': TAU,
            'world_size': WORLD, 'facts': facts(), 'arm_index': ARM_INDEX.get(arm), 'seed': SEED,
            'start_point': S.start_point(), 'screen': dict(SCREEN),
            'switch_space': {'space_id': S.SPACE_ID, 'sha256': S.sha256(), 'admissible_points': len(S.points()),
                             'switches': list(S.SWITCHES)},
            'numerics_gates': {'file': J.GATES_PATH.name, 'sha256': J.gates_sha256(), 'tier': gates['tier']},
            'deviations': dv.for_arm(arm), 'lifecycle': dv.SCREEN_LEVEL_LABEL, 'claims': CLAIMS,
            'method': ARMS.get(arm, {}).get('method') or REFERENCES[arm]['judged_against'],
            'llm_usd_cap_recorded': ARMS.get(arm, {}).get('llm_usd')}
    spec['arm_index'] = ARM_INDEX.get(arm, CODE_ARM_INDEX.get(arm, OPEN_ARM_INDEX.get(arm)))
    if arm in CODE_ARMS or (code and arm in OPEN_ARMS):   # setting 2: the code space on top of S_F1 (the switch-only
        #                     window specs stay byte-for-byte; the flag opens it only for an open-wave arm, as F-A0-CODE)
        spec.update(action_space='knobs+patch', mutable_globs=list(C.MUTABLE_PATHS), max_patch_lines=C.MAX_PATCH_LINES,
                    frozen_files={k: v for k, v in spec['frozen_files'].items() if k not in C.MUTABLE_PATHS},
                    claims=CODE_CLAIMS if arm in CODE_ARMS else spec['claims'],
                    code_space={**C.describe(), 'sha256': C.sha256()})
    if arm in V4_ARMS:                                    # the v4 caps, where a v4 tree reads them (from_dict deep-copies)
        spec['manifest'] = V4_MANIFEST
    elif arm in BUD_ARMS:                                 # the same-money re-runs' caps
        spec['manifest'] = BUD_MANIFEST
    return EnvTask.from_dict(spec)


# ---------------------------------------------------------------- the task view
def start_summary(receipt: dict | None) -> dict | None:
    """The U+C reference receipt summary shown to every arm (measured on this node set; never a stored timing)."""
    if not isinstance(receipt, dict):
        return None
    view = receipt.get('summary') or ((receipt.get('judge') or {}).get('details') or {}).get('view') or {}
    return {'attempt_id': receipt.get('attempt_id'), 'valid': receipt.get('valid'),
            'step_time_s': (receipt.get('metric') or {}).get('value'),
            'numerics_class_vs_previous_reference_run': (receipt.get('judge') or {}).get('numerics_class'),
            'per_step_max_rank_s': view.get('per_step_max_rank_s'), 'per_rank_spread': view.get('per_rank_spread'),
            'values': view.get('values'), 'gpu_hours': round(((receipt.get('cost') or {}).get('gpu_seconds') or 0) / 3600.0, 3)}


def gates_view() -> dict:
    g = J.load_gates()
    return {'numerics': {'reference': 'the start configuration\'s screen on this node set (updates 1..30)',
                         'tolerance': g['tolerance'], 'mtp_ce': g['mtp_ce'], 'lr': g['lr'], 'aux': g['aux'],
                         'classes': g['classes'], 'allowed': g['allowed']},
            'hard_checks': ['all 32 ranks complete 30 updates and the 4 nodes exit 0', 'frozen code, data and source hashes',
                            'the screened switch values are what ran (launch record, env, every rank)',
                            f'wall cap {int(G.CAP_S)} s per screen', 'no OOM or NCCL error'],
            'proposals': 'a point outside the switch space, a duplicate of a screened point or the start itself is refused '
                         'before any GPU time (invalid_proposal); a point that fails to build or run costs its screen'}


def task_view(task: EnvTask, *, start_receipt: dict | None = None, hardware: dict | None = None) -> dict:
    """The switch-only view (every arm); the code arm's hooks add the code-space block (code_space.task_view)."""
    from ...discover import default_profile
    hw = hardware if isinstance(hardware, dict) and isinstance(hardware.get('summary'), dict) else default_profile(4)
    return {'task_id': task.id, 'env': task.env, 'arm': task.arm,
            'metric': {'name': task.metric_name, 'direction': 'min', 'definition': task.facts['screen']['metric']},
            'hardware_profile': {'source': hw.get('source'), 'summary': hw.get('summary')},
            'workload': {k: task.facts[k] for k in ('model', 'data', 'objective', 'screen')},
            'fixed_layout': task.facts['fixed_layout'],
            'start_configuration': S.start_point(),
            'start_measured': start_summary(start_receipt),
            'switch_space': S.task_view(),
            'gates': gates_view(),
            'budget': task.budget}


def task_view_leaks(view) -> list[str]:
    """Recipe names or stored formal-54 window timings in a (rendered) task view: must be empty. A timing counts only
    as a whole number (12.49 or 12.490, not 12.4912 or 112.49)."""
    text = view if isinstance(view, str) else json.dumps(view, default=str)
    hits = [n for n in RECIPE_NAMES if re.search(r'(?<![A-Za-z0-9_])' + re.escape(n) + r'(?![A-Za-z0-9_])', text)]
    hits += [f'"{n}"' for n in ('W', 'B', 'Q') if f'"{n}"' in text]
    hits += [t for t in STORED_TIMINGS if re.search(r'(?<![0-9.])' + re.escape(t) + r'0*(?![0-9])', text)]
    return hits
