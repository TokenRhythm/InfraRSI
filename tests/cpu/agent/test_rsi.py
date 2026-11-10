import json
import sqlite3

import pytest

from infrarsi.agent.backends import ScriptedBackend
from infrarsi.agent.context import check_skill, parse_skill
from infrarsi.agent.memory import MemoryDB
from infrarsi.agent.runner import synthetic_receipt
from infrarsi.contracts import TaskSpec
from infrarsi.rsi import OptimizerPackage, build_a0, compute_digest, copy_package, without_memory, write_manifest
from infrarsi.rsi.audit import audit_package
from infrarsi.rsi.ledger import CostLedger
from infrarsi.rsi.lineage import Lineage
from infrarsi.rsi.package import A0_SKILLS, REPO_ROOT
from infrarsi.rsi.reflect import reflect
from infrarsi.rsi.selfcheck import selfcheck


@pytest.fixture
def pkg(tmp_path, a0_package):
    copy_package(a0_package.root, tmp_path / 'A')
    return OptimizerPackage.load(tmp_path / 'A')


@pytest.fixture(scope='module')
def history():
    t = TaskSpec.load('D2').raw
    parent = synthetic_receipt(t, {}, step_time_s=5.0, peak_reserved_gib=60, run_id='r', attempt_id='p')
    child = synthetic_receipt(t, {}, step_time_s=4.5, peak_reserved_gib=62, run_id='r', attempt_id='c')
    return [(child, parent, t), parent]


def remanifest(p):
    m = json.loads((p.root / 'manifest.json').read_text())
    write_manifest(p.root, label=m['label'], parent_digest=m['parent_digest'], producer=m['producer'])


def test_a0_has_the_12_human_skills_and_passes_selfcheck(a0_package, history):
    assert a0_package.skill_names() == sorted(A0_SKILLS) and len(A0_SKILLS) == 12
    for name in A0_SKILLS:
        text = (a0_package.skills_dir / name / 'SKILL.md').read_text(encoding='utf-8')
        front, sections = parse_skill(text)
        assert front['metadata']['source'] == 'human-authored (A0)' and front['license'] == 'Apache-2.0'
        assert not check_skill(text, name)
        assert any(s['heading'].lower().startswith('success case') for s in sections)
    r = selfcheck(a0_package.root, history_receipts=history)
    assert r.ok, r.errors
    assert r.checked['tool_runs'] == 4 * 3 and r.checked['memory_entries'] == 0
    assert (a0_package.skills_dir / '_catalogue' / 'negative-results.md').exists()


def test_repo_skills_match_a0_and_are_isolated():
    for name in A0_SKILLS:
        assert not check_skill((REPO_ROOT / 'skills' / name / 'SKILL.md').read_text(encoding='utf-8'), name)
    a = audit_package(REPO_ROOT / 'skills')
    assert a.passed, a.findings


def test_a0_is_reproducible(tmp_path, a0_package):
    other = build_a0(tmp_path / 'A0b')
    assert other.digest == a0_package.digest


@pytest.mark.parametrize('breakage, needle', [
    (lambda p: (p.skills_dir / 'recompute-policy' / 'SKILL.md').write_text('no front matter', encoding='utf-8'), 'unparseable'),
    (lambda p: (p.skills_dir / 'recompute-policy' / 'SKILL.md').write_text(
        (p.skills_dir / 'recompute-policy' / 'SKILL.md').read_text(encoding='utf-8').replace('## Failure case (real)', '## Notes'),
        encoding='utf-8'), 'section missing: failure case'),
    (lambda p: [f.unlink() for f in (p.skills_dir / 'weak-model-protocol').iterdir()] and None
        or (p.skills_dir / 'weak-model-protocol').rmdir(), 'core skills removed'),
    (lambda p: (p.root / 'prompts' / 'proposer.md').write_text('no placeholder', encoding='utf-8'), 'lost placeholder'),
    (lambda p: (p.root / 'prompts' / 'proposer.md').write_text('{ACTION_MENU}' + 'x' * 30000, encoding='utf-8'), 'argv budget'),
    (lambda p: (p.root / 'policy.yaml').write_text((p.root / 'policy.yaml').read_text().replace('max_context_tokens: 32000', 'max_context_tokens: 64000'),
                                                   encoding='utf-8'), 'modes.weak.max_context_tokens'),
    (lambda p: (p.root / 'policy.yaml').write_text((p.root / 'policy.yaml').read_text() + 'reward:\n  w_t: 2\n', encoding='utf-8'), 'unknown top-level'),
    (lambda p: (p.root / 'tools' / 'broken.analysis').write_text("return data['no_such_key']\n", encoding='utf-8'), 'tool broken'),
    (lambda p: (p.root / 'tools' / 'evil.analysis').write_text("import os\nreturn 1\n", encoding='utf-8'), 'invalid source'),
    (lambda p: (p.root / 'tools' / 'x.py').write_text('print(1)\n', encoding='utf-8'), 'unexpected files in tools'),
    (lambda p: (p.root / 'notes.txt').write_text('hi', encoding='utf-8'), 'unexpected top-level'),
])
def test_selfcheck_rejects_bad_packages(pkg, history, breakage, needle):
    breakage(pkg)
    remanifest(pkg)
    r = selfcheck(pkg.root, history_receipts=history)
    assert not r.ok and any(needle in e for e in r.errors), r.errors


def test_selfcheck_detects_unmanifested_edits(pkg):
    (pkg.skills_dir / 'recompute-policy' / 'SKILL.md').write_text(
        (pkg.skills_dir / 'recompute-policy' / 'SKILL.md').read_text(encoding='utf-8') + '\nextra\n', encoding='utf-8')
    r = selfcheck(pkg.root)
    assert any('digest' in e for e in r.errors)
    with pytest.raises(ValueError):
        OptimizerPackage.load(pkg.root)


@pytest.mark.parametrize('text, kind', [
    ('On H3 use TP2.', 'task_id'),
    ('E6-AR-H5 promoted config', 'task_id'),
    ('DeepSeek-V2-Lite prefers EP8', 'model'),
    ('meta-llama-3.1-8b fits on 4 GPUs', 'model'),
    ('the llama bridge path', 'model'),
    ('D-SFT-64K needs CP', 'data'),
    ('Qwen3-8B with D-PT-8K on 16 GPUs: EP none, TP2', 'combination'),
    ('{"schema": "infrarsi-receipt-v1", "per_step": []}', 'receipt'),
])
def test_audit_catches_heldout_material(pkg, text, kind):
    p = pkg.skills_dir / 'choose-parallel-layout' / 'SKILL.md'
    p.write_text(p.read_text(encoding='utf-8') + '\n' + text + '\n', encoding='utf-8')
    a = audit_package(pkg.root)
    assert not a.passed and kind in {f['kind'] for f in a.findings}, a.findings


def test_audit_scans_memory_rows_and_heldout_receipt_values(pkg):
    db = MemoryDB(pkg.memory_path)
    db.add({'tags': {'model': ['qwen3-dense']}, 'lesson': 'fits'}, source='x', task_id='H2')
    db.close()
    a = audit_package(pkg.root)
    assert {'task_id'} <= {f['kind'] for f in a.findings} and any(f['file'] == 'memory.db' for f in a.findings)
    p = pkg.skills_dir / 'recompute-policy' / 'SKILL.md'
    p.write_text(p.read_text(encoding='utf-8') + '\nattempt a1b2c3d4e5f6\n', encoding='utf-8')
    a = audit_package(pkg.root, heldout_receipts=[{'attempt_id': 'a1b2c3d4e5f6'}])
    assert 'heldout_receipt_value' in {f['kind'] for f in a.findings}


def test_audit_allows_dev_and_generic_text(pkg):
    p = pkg.skills_dir / 'choose-parallel-layout' / 'SKILL.md'
    p.write_text(p.read_text(encoding='utf-8') + '\nQwen3-8B with D-PT-8K on 8 GPUs (dev). H100 and H20 GPUs. 64K context needs CP.\n',
                 encoding='utf-8')
    assert audit_package(pkg.root).passed


def _trace(task_id='D2', writes=()):
    return {'task_id': task_id, 'run_id': f'E6-R-{task_id}-search', 'arm': 'R', 'mode': 'strong', 'stop_reason': 'finish',
            'start': 'c0', 'start_step_time': 5.0, 'selected': 'c1', 'selected_step_time': 4.5, 'selected_config': {},
            'selected_overlay': {}, 'candidates': [], 'screens': [], 'used': {}, 'budget': {}, 'invalid_proposals': {},
            'verifier': {}, 'skill_hits': [], 'tool_calls': [], 'memory_writes': list(writes)}


def test_reflect_changed_fallback_and_lineage(tmp_path, pkg, history):
    skill = (pkg.skills_dir / 'recompute-policy' / 'SKILL.md').read_text(encoding='utf-8')
    good = {'skills': {'recompute-policy': skill.replace('## Allowed actions', '## Rules: evolved (D2)\n- Removing full recompute fitted with 20 GiB headroom.\n\n## Allowed actions')},
            'memory': [{'tags': {'model': ['qwen3-dense']}, 'lesson': 'remove recompute at >20 GiB headroom'}],
            'tools': {'headroom_flag': "m = data['memory']['max_peak_reserved_gib']\nreturn {'tight': m != None and m > 74}\n"},
            'rationale': 'trace c1'}
    lineage, ledger = Lineage(tmp_path / 'lin'), CostLedger(tmp_path / 'ledger.jsonl')
    lineage.register(pkg, status='initial')
    writes = [{'turn': 3, 'noop': False, 'entry': {'tags': {'model': ['qwen3-dense']}, 'lesson': 'search lesson'}}]
    r1 = reflect('D2', _trace(writes=writes), pkg, ScriptedBackend([good]), tmp_path / 'r1', label='A_1',
                 history_receipts=history, lineage=lineage, ledger=ledger)
    assert r1.status == 'changed' and r1.parent_digest == pkg.digest and r1.digest != pkg.digest
    a1 = OptimizerPackage.load(r1.package_dir)
    assert 'headroom_flag' in a1.tools and MemoryDB(a1.memory_path, read_only=True).count() == 2
    assert json.loads((a1.root / 'manifest.json').read_text())['parent_digest'] == pkg.digest
    bad_cases = [({'skills': {'recompute-policy': 'broken'}, 'rationale': 'x'}, 'unparseable'),
                 ({'skills': {'new-skill': skill.replace('name: recompute-policy', 'name: new-skill') + '\nH5 uses EP16\n'}, 'rationale': 'x'}, 'isolation'),
                 ({'tools': {'bad': 'return data["nope"]\n'}, 'rationale': 'x'}, 'tool bad'),
                 ({'prompts': {'proposer.md': 'lost it'}, 'rationale': 'x'}, 'placeholder'),
                 ('not json', 'no usable reflection')]
    for i, (edit, needle) in enumerate(bad_cases):
        r = reflect('D3', _trace('D3'), a1, ScriptedBackend([edit]), tmp_path / f'b{i}', label=f'A_{i + 2}',
                    history_receipts=history, lineage=lineage, ledger=ledger)
        assert r.status == 'invalid_fallback' and r.digest == a1.digest and any(needle in e for e in r.errors), r.errors
    statuses = [x['status'] for x in lineage.rows()]
    assert statuses == ['initial', 'changed'] + ['invalid_fallback'] * 5
    assert ledger.totals(('book', 'phase'))['llm|reflect']['rows'] == 6
    with pytest.raises(ValueError):
        reflect('D4', _trace('D2'), a1, ScriptedBackend([good]), tmp_path / 'wrong', label='A_x')


def test_lineage_snapshot_freeze_and_g4(tmp_path, pkg):
    lin = Lineage(tmp_path / 'lin')
    lin.register(pkg, status='initial')
    snap = lin.snapshot('A_R', pkg)
    rec = lin.freeze('A_R')
    assert rec['digest'] == pkg.digest and lin.verify_frozen('A_R')
    with pytest.raises(PermissionError):
        lin.register(pkg, status='changed')
    (snap.root / 'policy.yaml').write_text(snap.policy_text + '\n# edit after freeze\n', encoding='utf-8')
    assert not lin.verify_frozen('A_R')


def test_without_memory_ablation(tmp_path, pkg):
    db = MemoryDB(pkg.memory_path)
    db.add({'lesson': 'x'}, source='s', task_id='D2')
    db.close()
    remanifest(pkg)
    full = OptimizerPackage.load(pkg.root)
    nm = without_memory(full, tmp_path / 'nomem')
    assert MemoryDB(nm.memory_path, read_only=True).count() == 0 and nm.skill_names() == full.skill_names()
    assert nm.digest != full.digest and json.loads((nm.root / 'manifest.json').read_text())['parent_digest'] == full.digest


def test_cost_ledger_books_and_caps(tmp_path):
    led = CostLedger(tmp_path / 'l.jsonl')
    led.add(book='llm', run_id='E1-D1-A0', arm='A0', task_id='D1', phase='search', llm_tokens=100, llm_usd=1199.0)
    led.add(book='gpu', run_id='E1-D1-A0', arm='A0', task_id='D1', phase='search', gpu_hours=20.0, gpu_type='H100-80GB')
    tot = led.totals(('book', 'arm'))
    assert tot['llm|A0']['llm_usd'] == 1199.0 and tot['gpu|A0']['gpu_hours'] == 20.0
    assert led.check_caps('P1') is None
    led.add(book='llm', run_id='r', arm='R', task_id='D2', phase='reflect', llm_usd=2.0)
    assert 'P1 stop' in led.check_caps('P1') and led.check_caps('P0') is None
    with pytest.raises(ValueError):
        led.add(book='llm', run_id='r', arm='R', task_id=None, phase='search', llm_usd=-1)
    trace = {'run_id': 'x', 'arm': 'A0', 'task_id': 'S1', 'accounting_complete': True, 'llm_calls': [{'model': 'claude-opus-5-5'}],
             'used': {'llm_tokens': 10, 'llm_usd': 0.5, 'gpu_hours': 1.5, 'wall_seconds': 60.0}}
    rows = led.add_trace(trace)
    assert [r['book'] for r in rows] == ['llm', 'gpu'] and rows[0]['model'] == 'claude-opus-5-5'


def test_memory_db_query_by_tags(tmp_path):
    db = MemoryDB(tmp_path / 'm.db')
    db.add({'tags': {'model': ['moe'], 'hardware': ['gpus8']}, 'lesson': 'moe 8'}, source='s', task_id='D4')
    db.add({'tags': {'model': ['dense'], 'hardware': ['gpus8']}, 'lesson': 'dense 8'}, source='s', task_id='D1')
    db.add({'tags': {'model': ['dense'], 'hardware': ['gpus8']}, 'lesson': 'dense 8'}, source='s', task_id='D1')   # duplicate ignored
    assert db.count() == 2
    hits = db.query({'model': ['dense'], 'hardware': ['gpus8']}, k=1)
    assert hits[0]['lesson'] == 'dense 8' and hits[0]['score'] == 1.0
    ro = MemoryDB(tmp_path / 'm.db', read_only=True)
    assert ro.add({'lesson': 'x'}, source='s', task_id='D1') is None
    with pytest.raises(sqlite3.OperationalError):
        ro.con.execute("INSERT INTO meta VALUES ('a','b')")


def test_dev_sequence_snapshots_and_freeze(tmp_path, pkg):
    from infrarsi.rsi.loop import DEV_ORDER, advance
    lineage = Lineage(tmp_path / 'lin')
    lineage.register(pkg, status='initial')
    skill = (pkg.skills_dir / 'recompute-policy' / 'SKILL.md').read_text(encoding='utf-8')
    cur = pkg
    with pytest.raises(ValueError):
        advance(lineage, 'S2', _trace('S2'), cur, ScriptedBackend(['x']), tmp_path / 'out')
    for i, tid in enumerate(DEV_ORDER):
        edit = {'skills': {'recompute-policy': skill + f'\n- evolved after {tid}\n'}, 'rationale': tid} if i % 2 == 0 else 'not json'
        cur, res = advance(lineage, tid, _trace(tid), cur, ScriptedBackend([edit]), tmp_path / 'out')
        assert res.status == ('changed' if i % 2 == 0 else 'invalid_fallback')
    rows = lineage.rows()
    assert [r['label'] for r in rows if r['status'] == 'snapshot'] == ['A1', 'A_R']
    assert lineage.frozen('A_R')['digest'] == cur.digest and lineage.verify_frozen('A_R')
    a1 = [r for r in rows if r['status'] == 'snapshot' and r['label'] == 'A1'][0]
    d2 = [r for r in rows if r.get('task_id') == 'D2'][0]
    assert a1['digest'] == d2['digest']
    with pytest.raises((PermissionError, ValueError)):
        advance(lineage, 'D6', _trace('D6'), cur, ScriptedBackend(['x']), tmp_path / 'out')


def test_lenient_judge_and_backend_factory():
    from infrarsi.agent.backends import make_backend
    from infrarsi.agent.judge import LenientJudge
    t = TaskSpec.load('S2').raw
    parent = synthetic_receipt(t, {}, step_time_s=2.0, peak_reserved_gib=50, run_id='r', attempt_id='p')
    cheat = synthetic_receipt(t, {}, step_time_s=1.0, peak_reserved_gib=50, run_id='r', attempt_id='c', inter_step_ratio=2.0)
    v = LenientJudge()(cheat, parent)
    assert v.valid and v.reward > 0.6 and v.source == 'lenient'
    b = make_backend('deepseek', api_key='k')
    assert b.model == 'deepseek-flash' and b.base_url == 'https://api.deepseek.com' and b.price['output'] == 1.20
    v = make_backend('vllm', model='Qwen3-Next-80B-A3B-Instruct')
    assert v.price is None and v.structured == 'json_object' and v.api_key is None      # the setting verified on WIN0
    with pytest.raises(ValueError):
        make_backend('gpt')
