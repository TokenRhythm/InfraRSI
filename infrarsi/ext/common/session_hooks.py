"""SessionHooks: the per-environment seam of agent.optimize / agent.search.SearchSession (EXT_INTERFACES §3).

With hooks=None the agent is unchanged. With hooks, the Megatron-specific pieces are replaced:
  task_view()                       the task block of every prompt (replaces optimize._task_view)
  validate_config(config)           -> (normalized, errors, warnings)   (replaces actions.validate_l0)
  validate_patch(files)             -> (errors, warnings)               (replaces the overlay patch_guard)
  templates(), layers               L1 templates (may be empty); open layers: 'L0' = knobs, 'L1' = file patches
  receipt_fields, tools_enabled     receipt digest blocks; package analysis tools (Megatron receipts) default off
  split_guard(pkg)                  replaces the dev/held-out phase guard: the frozen package runs read-only
plus the read-only action read_file(path, start, end) (<= 400 lines, mutable or documented read-only files only).
Extensions with defaults (subclasses override what their environment needs): contract_text, verifier_checklist,
action_schema/action_menu/normalize, skill_query/skill_tags/memory_tags, receipt_digest, overlay_digest/diff.
Prompts of the package are used as they are; the environment's contract and action menu fill {ACTION_MENU}.
"""
from __future__ import annotations

import ast
import difflib
import fnmatch
import hashlib
import json

from ...agent.context import estimate_tokens
from ...agent.jsonschema_lite import validate
from ...agent.tools import MEMORY_ENTRY, normalize as _normalize
from ...contracts import digest
from .guards import path_problem, touched_outside


def _act(name: str, props: dict | None = None, required: tuple = ()) -> dict:
    return {'type': 'object', 'properties': {'action': {'const': name}, **(props or {})},
            'required': ['action', *required], 'additionalProperties': False}


READ_FILE_MAX_LINES = 400
MAX_FILE_BYTES = 262144
RECEIPT_FIELDS = ('id', 'verdict', 'metric', 'summary', 'guards', 'judge', 'phases', 'cost')
_ID = {'type': 'string', 'minLength': 1, 'maxLength': 64}
_TEXT = {'type': 'string', 'maxLength': 2000}
_PATH = {'type': 'string', 'minLength': 1, 'maxLength': 200}
_LINE = {'type': 'integer', 'minimum': 1, 'maximum': 1000000}


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def diff_lines(old: str, new: str) -> int:
    """Added + removed lines of a unified diff (the patch line limit of the design)."""
    n = 0
    for line in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm='', n=0):
        if not line.startswith(('+++', '---', '@@')):
            n += line[:1] in ('+', '-')
    return n


def layers_for(action_space: str) -> tuple:
    return {'knobs': ('L0',), 'patch': ('L1',), 'knobs+patch': ('L0', 'L1')}[action_space]


class SessionHooks:
    tools_enabled: bool = False
    receipt_fields: tuple = RECEIPT_FIELDS
    banned_imports: tuple = ()            # top-level modules a patched Python file may not import (environment AST bans)

    def __init__(self, task, *, base_files: dict | None = None, read_only_files: dict | None = None,
                 frozen_package_digest: str | None = None, layers: tuple | None = None):
        """base_files {relative path: text}: verified base versions of the mutable files (diffs, read_file);
        read_only_files {relative path: text}: documented read-only files read_file may show."""
        self.task = task
        self.base_files = dict(base_files or {})
        self.read_only_files = dict(read_only_files or {})
        self.frozen_package_digest = frozen_package_digest
        self.layers = tuple(layers) if layers else layers_for(task.action_space)
        if set(self.layers) - set(layers_for(task.action_space)):
            raise ValueError(f'layers {self.layers} exceed the action space {task.action_space!r}')

    # ------------------------------------------------------------ EXT_INTERFACES §3
    def task_view(self) -> dict:
        t = self.task
        props = (t.knob_schema or {}).get('properties') or {}
        return {'task_id': t.id, 'env': t.env, 'arm': t.arm, 'metric': {'name': t.metric_name, 'direction': 'min'},
                'action_space': t.action_space, 'layers': list(self.layers),
                'knobs': {k: {x: v.get(x) for x in ('enum', 'type', 'default', 'description') if x in v} for k, v in props.items()},
                'mutable_files': list(t.mutable_globs), 'read_only_files': sorted(self.read_only_files),
                'max_patch_lines': t.max_patch_lines, 'budget': t.budget, 'facts': t.facts,
                'contract': self.contract_text()}

    def validate_config(self, config) -> tuple[dict, list, list]:
        if type(config) is not dict:
            return {}, ['config must be an object'], []
        schema = self.task.knob_schema or {}
        props = schema.get('properties') or {}
        if 'L0' not in layers_for(self.task.action_space):
            return {}, ([f'this task has no knobs (action space {self.task.action_space!r})'] if config else []), []
        errs = validate(config, {**schema, 'additionalProperties': schema.get('additionalProperties', False)})
        resolved = {k: v['default'] for k, v in props.items() if 'default' in v}
        resolved.update(config)
        return (resolved if not errs else {}), errs, []

    def validate_patch(self, files) -> tuple[list, list]:
        errors, warnings = [], []
        if type(files) is not dict or not files:
            return ['patch must be a non-empty {relative path: complete file text} object'], []
        bad = [p for p, t in files.items() if type(p) is not str or type(t) is not str]
        if bad:
            return [f'non-string path or text: {bad[:4]}'], []
        paths = [f'{p!r}: {why}' for p in sorted(files) for why in [path_problem(p)] if why]
        if paths:
            return [f'non-canonical paths: {"; ".join(paths[:4])}'], []
        outside = touched_outside(files, self.task.mutable_globs)
        if outside:
            errors.append(f'not mutable: {outside[:6]} (mutable: {list(self.task.mutable_globs)})')
        frozen = sorted(set(files) & set(self.task.frozen_files))
        if frozen:
            errors.append(f'frozen files: {frozen[:6]}')
        total, changed = 0, False
        for path, text in sorted(files.items()):
            if len(text.encode('utf-8')) > MAX_FILE_BYTES:
                errors.append(f'{path}: larger than {MAX_FILE_BYTES} bytes')
                continue
            old = self.base_files.get(path)
            if old is None:
                warnings.append(f'{path}: new file (no base version)')
            total += diff_lines(old or '', text)
            changed |= text != old
            if path.endswith('.py'):
                try:
                    tree = ast.parse(text, filename=path)
                except SyntaxError as exc:
                    errors.append(f'{path}: syntax error line {exc.lineno}: {exc.msg}')
                    continue
                errors += self.ast_bans(path, tree, text)
        if total > self.task.max_patch_lines:
            errors.append(f'diff {total} lines > limit {self.task.max_patch_lines}')
        if not changed:
            errors.append('patch identical to the base files')
        return errors, warnings

    @property
    def patch_line_limit(self) -> int:
        return int(self.task.max_patch_lines)

    def patch_lines(self, files: dict) -> int:
        """Changed lines (added + removed) of a candidate's files versus the base files."""
        return sum(diff_lines(self.base_files.get(p, ''), t) for p, t in (files or {}).items() if type(t) is str)

    def templates(self) -> list:
        return []

    def render_template(self, template: str, params: dict, files: dict) -> tuple[dict | None, list]:
        return None, [f'unknown template {template!r} (this environment has none)']

    def split_guard(self, pkg) -> None:
        """Ext tasks run the frozen package read-only: its content must still match its digest (and the pinned
        digest when given); no development whitelist applies, write_memory only reaches the per-search copy."""
        from ...rsi.package import compute_digest
        if getattr(self.task, 'split', 'external') == 'dev':
            raise PermissionError(f'{self.task.id}: an ext task never runs as a development task')
        if self.frozen_package_digest and pkg.digest != self.frozen_package_digest:
            raise PermissionError(f'{self.task.id} runs only the frozen package {self.frozen_package_digest[:12]}, '
                                  f'got {pkg.digest[:12]}')
        if compute_digest(pkg.root) != pkg.digest:
            raise PermissionError(f'package content changed after freeze ({pkg.digest[:12]})')

    def read_file(self, path, start=1, end=None, files: dict | None = None) -> dict:
        """files: the candidate's own file set when reading a candidate's version (else the base version)."""
        if type(path) is not str:
            return {'error': 'path must be a string'}
        mutable = any(fnmatch.fnmatchcase(path, g) for g in self.task.mutable_globs)
        if mutable:
            text = (files or {}).get(path, self.base_files.get(path))
        else:
            text = self.read_only_files.get(path)
        if text is None:
            return {'error': f'{path!r} is not readable (mutable: {list(self.task.mutable_globs)}; '
                             f'read-only: {sorted(self.read_only_files)})'}
        lines = text.splitlines()
        a = max(1, int(start or 1))
        b = min(len(lines), int(end) if end else a + READ_FILE_MAX_LINES - 1, a + READ_FILE_MAX_LINES - 1)
        return {'path': path, 'start': a, 'end': b, 'total_lines': len(lines), 'mutable': mutable,
                'text': '\n'.join(f'{i:5d}  {lines[i - 1]}' for i in range(a, b + 1))}

    # ------------------------------------------------------------ prompts, schema, menu
    def contract_text(self) -> str:
        """Environment rules shown in the task view and in the proposer's {ACTION_MENU} slot (subclasses: the
        environment's frozen constants and allowed changes)."""
        return (f'This is the external environment {self.task.env} (task {self.task.id}). Minimize {self.task.metric_name}; '
                'the environment judge decides validity and InfraRSI guards can only reject. Instructions above that '
                'name Megatron-Core, L0 schema keys or runtime/overlays do not apply here; this contract does.')

    def verifier_checklist(self) -> list[str]:
        return ['The change keeps the environment contract (frozen files, fixed constants, data, loss and evaluation).',
                'The candidate touches only the mutable files and stays within the patch line limit.',
                f'The change has a concrete expected benefit on {self.task.metric_name} given the parent receipt.']

    def action_schema(self, mode: str) -> dict:
        if mode == 'weak':
            acts = {'inspect': _act('inspect', {'what': {'enum': ['env', 'receipt', 'skill', 'file']}, 'candidate_id': _ID,
                                                'query': {'type': 'string', 'maxLength': 300}, 'path': _PATH,
                                                'start': _LINE, 'end': _LINE}, ('what',))}
            if 'L0' in self.layers:
                acts['propose_config'] = _act('propose_config', {'base': _ID, 'changes': {'type': 'object', 'maxProperties': 6},
                                                                  'rationale': {'type': 'string', 'maxLength': 600}}, ('base', 'changes'))
            if 'L1' in self.layers:
                if self.templates():
                    acts['apply_patch_template'] = _act('apply_patch_template', {'base': _ID, 'template': {'type': 'string'},
                                                                                 'params': {'type': 'object'}}, ('template',))
                acts['propose_patch'] = _act('propose_patch', {'base': _ID, 'file': _PATH, 'content': {'type': 'string', 'maxLength': MAX_FILE_BYTES},
                                                                'rationale': {'type': 'string', 'maxLength': 600}}, ('file', 'content'))
            acts['run_trial'] = _act('run_trial', {'candidate_id': _ID}, ('candidate_id',))
            acts['write_memory'] = _act('write_memory', {'entry': MEMORY_ENTRY}, ('entry',))
            acts['finish'] = _act('finish', {'candidate_id': _ID, 'reason': {'type': 'string', 'maxLength': 600}})
            return {'oneOf': list(acts.values())}
        acts = {'read_env': _act('read_env'),
                'read_file': _act('read_file', {'path': _PATH, 'start': _LINE, 'end': _LINE, 'candidate_id': _ID}, ('path',)),
                'read_receipt': _act('read_receipt', {'candidate_id': _ID, 'fields': {'type': 'array', 'items': {'type': 'string'},
                                                                                      'maxItems': 12}}, ('candidate_id',)),
                'retrieve_skill': _act('retrieve_skill', {'query': {'type': 'string', 'minLength': 1, 'maxLength': 500},
                                                          'skill': {'type': 'string', 'maxLength': 80}}, ('query',))}
        if 'L0' in self.layers:
            acts['propose_config'] = _act('propose_config', {'base': _ID, 'config': {'type': 'object'}, 'changes': {'type': 'object'},
                                                              'rationale': _TEXT})
        if 'L1' in self.layers:
            acts['propose_patch'] = _act('propose_patch', {'base': _ID, 'files': {'type': 'object', 'additionalProperties': {'type': 'string'}},
                                                            'template': {'type': 'string'}, 'params': {'type': 'object'}, 'rationale': _TEXT})
        acts['run_screen'] = _act('run_screen', {'candidate_id': _ID}, ('candidate_id',))
        acts['write_memory'] = _act('write_memory', {'entry': MEMORY_ENTRY}, ('entry',))
        acts['finish'] = _act('finish', {'candidate_id': _ID, 'reason': _TEXT})
        return {'oneOf': list(acts.values())}

    def action_menu(self, mode: str) -> str:
        knobs = ', '.join(sorted(((self.task.knob_schema or {}).get('properties') or {}))) or '-'
        files = ', '.join(self.task.mutable_globs) or '-'
        if mode == 'weak':
            rows = ['Return exactly ONE JSON object, nothing else. Menu (pick one):',
                    '1 {"action":"inspect","what":"env|receipt|skill|file","candidate_id":"c1","query":"...","path":"<file>"}']
            if 'L0' in self.layers:
                rows.append(f'2 {{"action":"propose_config","base":"c0","changes":{{"<knob>":value}}}}   (at most 6 knobs of: {knobs})')
            if 'L1' in self.layers:
                if self.templates():
                    rows.append('3 {"action":"apply_patch_template","base":"c2","template":"<template id>","params":{...}}')
                rows.append(f'4 {{"action":"propose_patch","base":"c2","file":"<one of: {files}>","content":"<complete file>"}}')
            rows += ['5 {"action":"run_trial","candidate_id":"c4"}', '6 {"action":"write_memory","entry":{"lesson":"..."}}',
                     '7 {"action":"finish","candidate_id":"c4","reason":"..."}']
        else:
            rows = ['Return exactly one JSON object per turn (no markdown). Actions:',
                    '{"action":"read_env"}  task contract, environment facts, start candidate',
                    f'{{"action":"read_file","path":"<file>","start":1,"end":400,"candidate_id":"c3"}}  <= {READ_FILE_MAX_LINES} lines of a '
                    'mutable file (base version, or as in a candidate) or of a documented read-only file',
                    '{"action":"read_receipt","candidate_id":"c3","fields":["metric","summary","guards"]}  digest of a screened candidate',
                    '{"action":"retrieve_skill","query":"...","skill":"<name>"}  top-2 skill sections']
            if 'L0' in self.layers:
                rows.append(f'{{"action":"propose_config","base":"c0","changes":{{"<knob>":value}},"rationale":"..."}}  knobs: {knobs}')
            if 'L1' in self.layers:
                rows.append(f'{{"action":"propose_patch","base":"c2","files":{{"<path>":"<complete new file text>"}},"rationale":"..."}}  '
                            f'mutable files: {files}; at most {self.task.max_patch_lines} changed lines vs the base')
            rows += ['{"action":"run_screen","candidate_id":"c4"}  one screen = one fresh run of the environment and its judge (costs budget)',
                     '{"action":"write_memory","entry":{"tags":{...},"outcome":"...","failure_reason":null,"lesson":"..."}}',
                     '{"action":"finish","candidate_id":"c4","reason":"..."}  stop; the best valid screened candidate is promoted']
        return '\n'.join(rows) + '\n- Environment contract: ' + self.contract_text()

    def normalize(self, action: dict, mode: str) -> dict:
        if mode == 'weak' and action.get('action') == 'inspect' and action.get('what') == 'file':
            return {'action': 'read_file', 'path': action.get('path', ''), 'start': action.get('start'), 'end': action.get('end'),
                    'candidate_id': action.get('candidate_id')}
        return _normalize(action, mode)

    # ------------------------------------------------------------ retrieval and memory tags
    def skill_tags(self) -> list[str]:
        f = self.task.facts or {}
        hw = f.get('hardware') or {}
        tags = {self.task.env, 'external', str(hw.get('gpu_type', '')).lower(), f'gpus{self.task.world_size}',
                'single-node' if int(hw.get('nodes', 1) or 1) == 1 else 'multi-node', 'dense'}
        return sorted(t for t in tags if t)

    def skill_query(self) -> str:
        return ' '.join(self.skill_tags()) + f' {self.task.metric_name} profile whole update kernels data path memory'

    def memory_tags(self) -> dict:
        f = self.task.facts or {}
        return {'hardware': [str((f.get('hardware') or {}).get('gpu_type', '')).lower(), f'gpus{self.task.world_size}'],
                'model': [self.task.env, str((f.get('model') or {}).get('name', ''))], 'data': [str((f.get('data') or {}).get('name', ''))]}

    # ------------------------------------------------------------ receipts and candidates
    def receipt_digest(self, receipt: dict | None, verdict: dict | None = None, *, max_tokens: int = 1500, fields=None) -> dict:
        if not receipt:
            return {'receipt': None}
        guards = {'rejections': (receipt.get('guard_rejections') or [])[:6], 'reports': (receipt.get('guard_reports') or [])[:6]}
        judge = receipt.get('judge')
        judge_txt = json.dumps(judge, default=str) if judge is not None else None
        blocks = [
            ('id', {'attempt_id': receipt.get('attempt_id'), 'purpose': receipt.get('purpose'), 'valid': receipt.get('valid'),
                    'judge_valid': receipt.get('judge_valid'), 'invalid_reasons': (receipt.get('invalid_reasons') or [])[:4]}),
            ('verdict', verdict and {k: verdict.get(k) for k in ('valid', 'reasons', 'step_time', 'reward', 'feedback')
                                     if k != 'feedback' or verdict.get(k)}),
            ('metric', receipt.get('metric')),
            ('summary', receipt.get('summary')),
            ('guards', guards),
            ('judge', judge if judge_txt is None or len(judge_txt) <= 3000 else {'truncated': judge_txt[:3000]}),
            ('phases', receipt.get('phases')),
            ('cost', receipt.get('cost')),
        ]
        keep = set(fields or self.receipt_fields) | {'id', 'verdict'}
        out = {k: v for k, v in blocks if k in keep and v is not None}
        while estimate_tokens(json.dumps(out, default=str)) > max_tokens and len(out) > 2:
            out.pop(list(out)[-1])
        return out

    def candidate_files(self, overlay: dict | None) -> dict:
        """The candidate's full mutable file set: base files overridden by the candidate's files."""
        return {**self.base_files, **(overlay or {})}

    def overlay_digest(self, overlay: dict | None) -> str | None:
        if not overlay:
            return None
        return digest(sorted([p, sha_text(t)] for p, t in self.candidate_files(overlay).items()))

    def overlay_diff(self, base: dict | None, new: dict | None, limit: int = 4000) -> str:
        old_files, new_files = self.candidate_files(base), self.candidate_files(new)
        out = []
        for name in sorted(set(new or {})):
            out += list(difflib.unified_diff(old_files.get(name, '').splitlines(), new_files[name].splitlines(),
                                             f'a/{name}', f'b/{name}', lineterm='', n=1))
        s = '\n'.join(out)
        return s if len(s) <= limit else s[:limit] + '\n... (truncated)'

    def ast_bans(self, path: str, tree: ast.AST, text: str) -> list[str]:
        """Environment AST bans on a patched Python file; default: the banned_imports modules."""
        errs = []
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                if m.split('.')[0] in self.banned_imports:
                    errs.append(f'{path}:{node.lineno}: import of {m} is not allowed')
        return errs

    def describe(self) -> dict:
        return {'kind': type(self).__name__, 'task_id': self.task.id, 'env': self.task.env, 'layers': list(self.layers),
                'tools_enabled': self.tools_enabled, 'frozen_package_digest': self.frozen_package_digest,
                'base_files': {p: sha_text(t) for p, t in sorted(self.base_files.items())}}
