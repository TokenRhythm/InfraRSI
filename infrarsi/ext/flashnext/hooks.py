"""FlashNextHooks: the SessionHooks of F1 (EXT_INTERFACES §3; AMENDMENT_EXT_003 Q1, D-F1-2, D-F1-5).

  task_view()          task.task_view: HardwareProfile, workload and fixed-layout facts, the U+C reference receipt
                       summary, the S_F1 switch table (neutral sentences), the gates, the budget
  validate_config      switch_space.validate: the S_F1 validator, identical for every arm (errors = invalid_proposal,
                       zero GPU cost); duplicates of a screened point or of the start are refused by SearchSession
  validate_patch       always refused (switches only)
  skill_tags           moe / gdn / multi-node (+ the environment ids, which match no skill)
  tools_enabled False  A0's analysis tools and profile_ranges closed (D-F1-5); read_receipt shows receipt_digest():
                       per-update max-rank times, per-rank spread, numerics deltas and class, every values.* family
Seams for the ported baselines (backward compatible: used only with hooks):
  start_receipt_problems(receipt, config)   F-PROMPTS start check: the signed, valid ext U+C reference receipt
  param_codec(), fixed_parameters_text()    F-OPTICO: the switch table is its parameter table
The code arm (setting 2, F-A0-CODE; code_space.is_code_task): layers L0 + L1; base_files = the mutable runtime modules
(code/<name>), read_only_files = the runtime and source-tree files they are installed by or patch; validate_patch =
code_space.validate (the runner applies the same validator); the task view, contract, checklist and receipt digest
gain the code-space parts. Every other arm is exactly as before (switches only, validate_patch always refuses).
"""
from __future__ import annotations

import json

from ...agent.context import estimate_tokens
from ..common.session_hooks import SessionHooks
from . import code_space as C
from . import guards as G
from . import switch_space as S
from . import task as T


def _dump(obj) -> str:
    return json.dumps(obj, separators=(',', ':'), ensure_ascii=False, default=str)


class SwitchCodec:
    """S_F1 point <-> OptiCo 'KEY: VALUE' parameter lines (F-OPTICO). Launch values are written num_warps/num_stages
    ('4/1'): OptiCo's change_para parser splits replies on commas (parse_change_para), so '4,1' would not survive it."""

    def __init__(self):
        self.switches = list(S.SWITCHES)
        self.values = {row['name']: list(row['values']) for row in S.task_view()['switches']}

    def names(self) -> list[str]:
        return list(self.switches)

    @staticmethod
    def fmt(v) -> str:
        if v is None:
            return 'None'
        if isinstance(v, bool):
            return 'true' if v else 'false'
        return str(v).replace(',', '/')

    def to_lines(self, config: dict) -> str:
        return '\n'.join(f'{s}: {self.fmt((config or {}).get(s))}' for s in self.switches)

    def parse(self, name: str, raw) -> object:
        s = str(raw).strip().strip('"').strip("'").strip()
        low = s.lower()
        if low in ('none', 'null', 'unset', ''):
            return None
        allowed = self.values.get(name, [])
        if any(isinstance(v, bool) for v in allowed):
            if low in ('true', 'false'):
                return low == 'true'
            return s
        cand = s.replace('/', ',').replace(' ', '')
        return cand if cand in allowed else s

    def to_config(self, lines: dict) -> dict:
        """Every known switch from the lines (unknown names are kept so the validator refuses them)."""
        return {name: self.parse(name, raw) for name, raw in (lines or {}).items()}

    def domain_text(self) -> str:
        rows = []
        for row in S.task_view()['switches']:
            vals = ' | '.join(self.fmt(v) for v in row['values'])
            when = 'every stack' if row['name'] == 'implementation_stack' else 'only with implementation_stack in ' + \
                ', '.join(row['applies_to_stacks'])
            default = f'; None = {self.fmt(row["default_when_unset"])}' if row.get('default_when_unset') else ''
            rows.append(f'{row["name"]}: {vals} ({when}; otherwise None{default}). {row["description"]}')
            for v, d in (row.get('value_descriptions') or {}).items():
                rows.append(f'  {v}: {d}')
        return '\n'.join(rows)


class _F1Project:
    """A minimal project-like view so the ported AR / RB adapters find the open wave's equal-budget reporting basis
    (autoresearch/rebench.report_basis reads hooks.project.raw['search']['budget']['report_usd']; the F1 CLI path has no
    manifest-backed project, so FlashNextHooks supplies it directly). Not part of the hooks' described surface."""

    def __init__(self, report_usd: float):
        self.raw = {'search': {'budget': {'report_usd': float(report_usd)}}}


class FlashNextHooks(SessionHooks):
    tools_enabled = False
    receipt_fields = ('id', 'verdict', 'metric', 'summary', 'numerics', 'guards', 'point', 'cost')

    def __init__(self, task, *, start_receipt: dict | None = None, hardware: dict | None = None,
                 report_usd: float = 16.0, **kw):
        self.code = C.is_code_task(task)
        if self.code:                                  # setting 2: switches + code edits over C_F1
            kw.setdefault('layers', ('L0', 'L1'))
            kw.setdefault('base_files', C.base_files())
            kw.setdefault('read_only_files', C.read_only_files())
            self.receipt_fields = type(self).receipt_fields + ('code_overlay',)
        else:
            kw.setdefault('layers', ('L0',))
        super().__init__(task, **kw)
        self.start_receipt = start_receipt
        self.hardware = hardware
        # the open wave's within-16-USD equal-budget basis for AR / RB (B.7 D2); ignored by the other arms. 16 is the F1
        # open default; AR / RB report_basis falls back to the strategy's 8 only if this is absent (older runs).
        self.project = _F1Project(report_usd)

    # ------------------------------------------------------------ EXT_INTERFACES §3
    def task_view(self) -> dict:
        view = {**T.task_view(self.task, start_receipt=self.start_receipt, hardware=self.hardware),
                'contract': self.contract_text()}
        if self.code:
            view['code_space'] = C.task_view(self.read_only_files)
        return view

    def validate_config(self, config) -> tuple[dict, list, list]:
        if type(config) is not dict:
            return {}, ['config must be an object of S_F1 switches'], []
        point, errors, warnings = S.validate(config)
        return (point if not errors else {}), list(errors), list(warnings)

    def validate_patch(self, files) -> tuple[list, list]:
        if self.code:
            return C.validate(files, self.base_files)
        return ['F1 has no file actions: change implementation switches with propose_config'], []

    def contract_text(self) -> str:
        if self.code:
            return self.code_contract_text()
        return ('F1 is the production training path of Qwen3.8-Flash-Next (hybrid QSA / Gated DeltaNet / MoE) on 4 nodes x 8 '
                'H100. The parallel layout, data, order, loss, optimizer, LR schedule, precision and seed are fixed by the '
                'trainer; only the implementation switches of the switch space may change (implementation_stack and the '
                'switches its stack reads; a switch the stack does not read must be left unset = null). One screen = one '
                'fresh 30-update run on the 32 GPUs; StepTime = mean over updates 17..30 of the per-update max over ranks of '
                'the whole-update time (minimize). Valid = all ranks and nodes complete and the per-update main CE, MTP CE, '
                'grad norm, LR and aux match the start configuration\'s screen within the gates (E0 bitwise or E1); '
                'InfraRSI guards (frozen hashes, switch values bound to what ran, wall cap, no OOM/NCCL) can only reject. '
                'Proposals outside the space, duplicates and the start itself are refused at no GPU cost.')

    def code_contract_text(self) -> str:
        return ('F1 setting 2 is the production training path of Qwen3.8-Flash-Next (hybrid QSA / Gated DeltaNet / MoE) on 4 '
                'nodes x 8 H100 with code edits allowed. The parallel layout, data, order, loss, optimizer and its state, LR '
                'schedule, precision and seed are fixed by the trainer and cannot be edited. A candidate is a point of the '
                'switch space (implementation_stack and the switches its stack reads; a switch the stack does not read must be '
                'left unset = null), optionally with complete new versions of the mutable runtime files code/<name> '
                '(propose_patch; the speed-path modules listed under code_space: QSA attention kernels and launches, the MoE '
                'combine kernel, replay / retention / recompute, the QKV communication path, the start stack\'s fc1 backward), '
                f'at most {C.MAX_PATCH_LINES} changed lines over all files versus their base versions. L1 here means these '
                'files only: instructions that name Megatron-Core L0 schema keys or runtime/overlays/ files do not apply. '
                'Every other file, the trainer and its train_step are frozen and hash-checked; an edit may not add clocks, '
                'file / process / environment access, references to the optimizer, LR, data schedule or seed, strings naming '
                'the judge\'s records, or assignments into library modules (torch, triton, time, os, ...); read_file shows the '
                'mutable files (base or a candidate\'s version) and the listed read-only files. One screen = one fresh '
                '30-update run on the 32 GPUs; StepTime = mean over updates 17..30 of the per-update max over ranks of the '
                'whole-update time (minimize). Valid = all ranks and nodes complete and the per-update main CE, MTP CE, grad '
                'norm, LR and aux match the start configuration\'s screen within the gates (E0 bitwise or E1); InfraRSI '
                'guards (frozen and overlay hashes, switch values bound to what ran, wall cap, no OOM/NCCL) can only reject. '
                'Proposals outside the space or the code rules, duplicates and the start itself are refused at no GPU cost; '
                'a candidate whose code fails to import or run costs its screen.')

    def verifier_checklist(self) -> list[str]:
        items = ['The proposal is a point of the switch space: implementation_stack set, every switch its stack reads given '
                 'or left to its default, every other switch null; retain_expert_layers only with retain_attention_prefix.',
                 'It is not the start configuration nor a point already screened, and it avoids points that already failed.',
                 'It has a concrete expected benefit on the whole-update step time given the parent receipt (recompute or '
                 'replay avoided, attention / MoE kernels, host synchronisation), and keeps numerics within the gates.']
        if self.code:
            items.append('A code edit runs on the stack the point selects, computes the same loss, gradients and optimizer '
                         'inputs (it changes how, not what), touches no timing, logging, file, process or environment path, '
                         'and is complete Python that imports and runs on the 32 ranks (a failing edit costs a screen).')
        return items

    # ------------------------------------------------------------ retrieval, memory
    def skill_tags(self) -> list[str]:
        return sorted({'flashnext', 'external', 'moe', 'gdn', 'multi-node'})

    def skill_query(self) -> str:
        return ('moe gdn multi-node attention kernel fusion recompute activation retention replay expert layers '
                'segmented sum combine host sync memo triton launch warps stages fast mask step time whole update')

    def memory_tags(self) -> dict:
        return {'hardware': ['h100-80gb', f'gpus{T.WORLD}', 'nodes4'], 'model': ['flashnext', 'moe', 'gdn', 'qwen4_exp'],
                'data': ['sft_packed', 'seq32k']}

    # ------------------------------------------------------------ receipts (D-F1-5)
    @staticmethod
    def view_of(receipt: dict) -> dict:
        return receipt.get('summary') or ((receipt.get('judge') or {}).get('details') or {}).get('view') or {}

    def receipt_digest(self, receipt: dict | None, verdict: dict | None = None, *, max_tokens: int = 1500,
                       fields=None) -> dict:
        if not receipt:
            return {'receipt': None}
        view = self.view_of(receipt)
        judge = receipt.get('judge') if isinstance(receipt.get('judge'), dict) else {}
        summary = {k: view.get(k) for k in ('step_time_s', 'per_rank_spread', 'per_step_max_rank_s', 'values') if view.get(k)}
        blocks = [
            ('id', {'attempt_id': receipt.get('attempt_id'), 'purpose': receipt.get('purpose'), 'valid': receipt.get('valid'),
                    'judge_valid': receipt.get('judge_valid'), 'invalid_reasons': (receipt.get('invalid_reasons') or [])[:4]}),
            ('verdict', verdict and {k: verdict.get(k) for k in ('valid', 'reasons', 'step_time', 'reward')}),
            ('metric', {'step_time_s': (receipt.get('metric') or {}).get('value'),
                        'judge_value': ((receipt.get('metric') or {}).get('raw') or {}).get('judge_value')}),
            ('numerics', view.get('numerics') or {'class': judge.get('numerics_class')}),
            ('guards', {'rejections': [{'id': g.get('id'), 'detail': str(g.get('detail'))[:200]}
                                       for g in (receipt.get('guard_rejections') or [])[:4]],
                        'reports': sorted({str(g.get('id')) for g in receipt.get('guard_reports') or []})}),
            ('point', receipt.get('point')),
            ('summary', summary),
            ('cost', {k: (receipt.get('cost') or {}).get(k) for k in ('gpu_seconds', 'wall_seconds')}),
        ]
        if self.code:                                  # which code ran: the overlay digest and its files (None: base)
            if isinstance(receipt.get('failure_tail'), dict):          # a crashed run: the error it ended with
                blocks[0][1]['failure_tail'] = receipt['failure_tail']
            ov = receipt.get('code_overlay') if isinstance(receipt.get('code_overlay'), dict) else None
            blocks.insert(6, ('code_overlay', {'digest': (ov or {}).get('digest'), 'files': sorted((ov or {}).get('files') or {}),
                                               'changed_lines': (ov or {}).get('changed_lines')} if ov else {'digest': None}))
        keep = set(fields or self.receipt_fields) | {'id', 'verdict'}
        out = {k: v for k, v in blocks if k in keep and v is not None}
        # trim inside the summary first (values, then per-step times), then whole blocks from the end
        for part in ('values', 'per_step_max_rank_s', 'per_rank_spread'):
            if estimate_tokens(_dump(out)) <= max_tokens:
                break
            if isinstance(out.get('summary'), dict):
                out['summary'].pop(part, None)
        while estimate_tokens(_dump(out)) > max_tokens and len(out) > 2:
            out.pop(list(out)[-1])
        return out

    # ------------------------------------------------------------ baseline seams
    def start_receipt_problems(self, receipt, config) -> list[str]:
        """F-PROMPTS's input must be the U+C reference screen itself: a signed, valid ext receipt of task ENV-F1, arm
        UC-REF, purpose reference, candidate uc-ref, whose point is the S_F1 start point."""
        if not isinstance(receipt, dict):
            return ['no start receipt']
        p = [f'start receipt: {x}' for x in G.uc_reference_problems(receipt)]
        point, errs, _ = S.validate(config if isinstance(config, dict) else {})
        if errs or point != S.start_point():
            p.append('the start configuration is not the S_F1 start point')
        return p

    def param_codec(self) -> SwitchCodec:
        return SwitchCodec()

    def fixed_parameters_text(self) -> str:
        f = self.task.facts
        lay = f['fixed_layout']
        lines = [f'Model Type: {f["model"]["family"]}', f'Model Size: {f["model"]["key"]}',
                 f'NUM_LAYERS: {f["model"]["decoder_layers"]} (+{f["model"]["mtp_layers"]} MTP)',
                 'Precison: bf16 parameters, fp32 masters and Adam moments',
                 f'Global Batch Size: {f["data"]["packs_per_update"]} packs', f'Sequence Length: {lay["seq_len"]}',
                 f'Number of Nodes: {f["hardware"]["nodes"]}', f'Number of GPUs: {f["hardware"]["gpus"]}',
                 'GPU memory limit per card: 81920',
                 f'Fixed layout: TP {lay["tp"]}, EP {lay["ep"]}, CP {lay["cp"]}, ETP {lay["etp"]}, DP {lay["dp"]}, '
                 f'{lay["micro_batches_per_update"]} micro-batches per update, {lay["moe_dispatcher"]}, {lay["recompute"]}',
                 'Parameters you may change (the switch space; a parameter the chosen implementation_stack does not read '
                 'must be None):', self.param_codec().domain_text()]
        hw = T.task_view(self.task)['hardware_profile']['summary']
        g = T.gates_view()['numerics']
        b = self.task.budget
        lines += [f'Hardware: {hw.get("nodes")} nodes x {hw.get("gpus_per_node")} {hw.get("gpu_name")}, NVLink inside a node, '
                  f'{hw.get("ib_ports_active_per_node")} x {hw.get("ib_rate_gbps")} Gb/s InfiniBand between nodes, '
                  f'{hw.get("host_mem_gib")} GiB host memory per node',
                  'Validity: all 32 ranks finish the 30 updates; per update main CE and MTP CE within '
                  f'{g["tolerance"]["main_ce_rel_max_per_step"]} relative (window mean {g["tolerance"]["main_ce_rel_mean_over_window"]}), '
                  f'grad norm within {g["tolerance"]["grad_norm_rel_max_first_update"]} at update 1 and a window log bias of '
                  f'{g["tolerance"]["grad_norm_log_bias_window_max"]}, LR identical, aux present, against the start configuration\'s run',
                  f'Budget: {b["screens"]} runs, {b["llm_tokens"]} LLM tokens, {b["wall_seconds"]:.0f} s wall, {b["gpu_hours"]} GPU-hours']
        return '\n'.join(lines)

    def describe(self) -> dict:
        out = {**super().describe(), 'switch_space_sha256': S.sha256(), 'start_receipt': (self.start_receipt or {}).get('attempt_id')}
        if self.code:
            out['code_space_sha256'] = C.sha256()
        return out
