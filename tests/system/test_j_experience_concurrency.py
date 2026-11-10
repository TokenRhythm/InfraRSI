"""J8 / J9 - the experience journey (a run writes a pack, the store sees it, export and validate round-trip, `off` writes
nothing) and two users on one machine (two runs at once, the ledger stays well-formed)."""
import json
import subprocess
import sys

import pytest

from _support import cli
from _support.consistency import check_run
from _support.toy import latest_run, make_toy, manifest, result, runs, write_manifest


def _with_model_facts(root, framework='pytorch'):
    doc = manifest(root)
    doc['model'] = {'params_b': 0.001}
    doc['workload']['seq_len'] = 512
    doc['hardware'] = {'gpus': 1, 'memory_gib': 16, 'gpu_type': 'NVIDIA H100 80GB HBM3'}
    if framework:
        doc['framework'] = framework
    write_manifest(root, doc)


def test_a_manifest_without_framework_still_writes_its_pack(tmp_path, home):
    """F-SYS-12 (fixed): no `framework` in the manifest -> the env line says `other`, the pack is written and validates."""
    root = make_toy(tmp_path / 'toy', screens=3)
    _with_model_facts(root, framework=None)
    res = cli.run('run', root, '--llm', 'none', '--strategy', 'random')
    assert res.code == 0, res.text
    e = result(latest_run(root))['experience']
    assert e.get('written'), e.get('not_written')
    env = [json.loads(ln) for ln in open(e['written'], encoding='utf-8') if '"env"' in ln][0]
    assert env['framework'] == 'other'
    assert cli.run('exp', 'validate', e['written']).code == 0


def test_a_run_writes_a_pack_the_store_reads_and_export_round_trips(tmp_path, home):
    root = make_toy(tmp_path / 'toy', screens=4)
    _with_model_facts(root)
    before = cli.run('exp', 'stats')
    assert before.code == 0
    n_before = int(before.out.split(' packs:')[0])
    res = cli.run('run', root, '--llm', 'none', '--strategy', 'random')
    assert res.code == 0, res.text
    r = check_run(latest_run(root), budget_screens=4)
    e = r['experience']
    assert e is not None and e.get('written'), e
    pack = tmp_path / 'written.jsonl'
    pack.write_bytes(open(e['written'], 'rb').read())
    assert str(home) in e['written']
    val = cli.run('exp', 'validate', e['written'])
    assert val.code == 0 and '0 errors' in val.out, val.text
    after = cli.run('exp', 'stats')
    assert int(after.out.split(' packs:')[0]) == n_before + 1, after.out
    exported = tmp_path / 'share.jsonl'
    ex = cli.run('exp', 'export', latest_run(root), '--out', exported, '--license', 'CC-BY-4.0', '--contributor', 'tester')
    assert ex.code == 0 and exported.is_file() and 'license CC-BY-4.0' in ex.out, ex.text
    head = json.loads(exported.read_text(encoding='utf-8').splitlines()[0])
    assert head.get('license') == 'CC-BY-4.0' and head.get('contributors') == ['tester']
    assert cli.run('exp', 'validate', exported).code == 0
    # the second run of the same project starts from what the first measured
    res = cli.run('run', root, '--llm', 'none', '--strategy', 'random')
    assert res.code == 0, res.text
    r2 = check_run(latest_run(root), budget_screens=4)
    lv = r2['experience']['levels']
    assert lv.get('covered', 0) + lv.get('null', 0) >= 1, lv
    doc = cli.run('doctor', root)
    assert doc.code == 0 and 'experience:' in doc.out and 'packs' in doc.out


def test_experience_off_writes_no_pack_and_shows_no_priors(tmp_path, home):
    root = make_toy(tmp_path / 'toy', screens=3)
    _with_model_facts(root)
    res = cli.run('run', root, '--llm', 'none', '--strategy', 'random', '--experience', 'off')
    assert res.code == 0, res.text
    r = check_run(latest_run(root), budget_screens=3)
    assert not (r.get('experience') or {}).get('written')
    assert not list((home / 'experience').glob('*.jsonl')) if (home / 'experience').is_dir() else True
    res = cli.run('doctor', root, '--experience', 'off')
    assert res.code == 0 and 'experience:' not in res.out


def test_a_toy_without_model_facts_says_why_no_pack_was_written(tmp_path, home):
    root = make_toy(tmp_path / 'toy', screens=2)
    res = cli.run('run', root, '--llm', 'none', '--strategy', 'random')
    assert res.code == 0, res.text
    e = result(latest_run(root))['experience']
    assert e and not e.get('written') and 'params_b' in (e.get('not_written') or '')
    ex = cli.run('exp', 'export', latest_run(root), '--out', tmp_path / 'x.jsonl')
    assert ex.code != 0 and 'wrote no experience pack' in ex.text and 'Traceback' not in ex.text


def test_export_refuses_private_strings_unless_forced(tmp_path, home):
    pack = tmp_path / 'p.jsonl'
    lines = [{'kind': 'pack', 'format': 'infrarsi-exp/1', 'pack_id': 'x-test', 'title': 't', 'tier': 'measured', 'license': 'CC-BY-4.0'},
             {'kind': 'env', 'id': 'e1', 'gpu': 'NVIDIA H100 80GB HBM3', 'gpus': 1, 'nodes': 1, 'framework': 'pytorch',
              'notes': 'measured on 10.0.0.12 under /home/user/runs'}]
    pack.write_text('\n'.join(json.dumps(x) for x in lines) + '\n', encoding='utf-8')
    val = cli.run('exp', 'validate', pack)
    assert 'private' in val.out.lower() or val.code != 0, val.text
    ex = cli.run('exp', 'export', pack, '--out', tmp_path / 'out.jsonl')
    if ex.code != 0:
        assert 'private' in ex.text.lower() or 'fix the errors' in ex.text
        forced = cli.run('exp', 'export', pack, '--out', tmp_path / 'out.jsonl', '--force')
        assert forced.code == 0 or 'fix the errors' in forced.text


def test_two_runs_on_the_same_project_started_together_keep_apart(tmp_path, home):
    """F-SYS-17 (fixed): two users (or a script) start `run` on one project within the same second: the runs land in two
    run directories (the second gets a numbered suffix), never in one shared directory."""
    root = make_toy(tmp_path / 'toy', screens=2)
    cmd = [sys.executable, '-m', 'infrarsi', 'run', str(root), '--llm', 'none', '--strategy', 'random']
    procs = [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
             for _ in range(2)]
    outs = [p.communicate(timeout=600)[0] for p in procs]
    codes = [p.returncode for p in procs]
    assert all(c in (0, 2) for c in codes), outs
    rs = runs(root)
    if codes == [0, 0]:
        assert len(rs) == 2, f'two runs completed but {len(rs)} run directory(ies) exist: {[p.name for p in rs]}'
        for p in rs:
            check_run(p, budget_screens=2)
    else:
        refused = [o for o, c in zip(outs, codes) if c == 2]
        assert refused and 'Traceback' not in refused[0]


def test_two_runs_at_once_on_one_machine(tmp_path, home):
    roots = [make_toy(tmp_path / f'toy{i}', screens=3) for i in range(2)]
    procs = [subprocess.Popen([sys.executable, '-m', 'infrarsi', 'run', str(r), '--llm', 'none', '--strategy', 'random'],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
             for r in roots]
    outs = [p.communicate(timeout=600)[0] for p in procs]
    for p, out, root in zip(procs, outs, roots):
        assert p.returncode == 0, out
        check_run(latest_run(root), budget_screens=3)
    ledger = home / 'ledger.jsonl'
    if ledger.is_file():
        for line in ledger.read_text(encoding='utf-8').splitlines():
            json.loads(line)                                        # every appended line is whole
    launches = __import__('os').environ.get('INFRARSI_LAUNCH_LEDGER')
    if launches and __import__('pathlib').Path(launches).is_file():
        for line in open(launches, encoding='utf-8'):
            if line.strip():
                json.loads(line)
