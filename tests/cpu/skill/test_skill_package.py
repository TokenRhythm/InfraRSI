"""The skill `infrarsi-optimize` (agent_skill/): its SKILL.md is one both hosts load (two front-matter keys, a short
body, every reference there); what it tells an agent to type exists in the CLI; its list of checks is the verifier's;
the build copies the engine whole and the built skill runs on its own, without writing into itself - through a whole
run on the toy job; the installer puts the skill and the answering sub-agent where each host looks for them."""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import build_skill
import install_skill
from infrarsi.agent import state as engine_state
from infrarsi.agent.backends import session as S
from infrarsi.cli.main import build as build_cli
from infrarsi.project import agent_session as A
from infrarsi.verify.checklist import CHECKS
from test_bcomp_project import scripted, toy  # noqa: F401  (the toy copy and its scripted model)

REPO = Path(__file__).resolve().parents[3]
SKILL = REPO / 'agent_skill' / 'infrarsi-optimize'
DOCS = [SKILL / 'SKILL.md', *sorted((SKILL / 'references').glob('*.md'))]


def front_matter(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding='utf-8')
    assert text.startswith('---\n'), path
    head, body = text[4:].split('\n---\n', 1)
    return yaml.safe_load(head), body


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    return build_skill.build(tmp_path_factory.mktemp('built'))


def test_skill_md_is_one_both_hosts_load():
    meta, body = front_matter(SKILL / 'SKILL.md')
    assert set(meta) == {'name', 'description'}          # the two keys every host reads; others are refused by some
    assert meta['name'] == 'infrarsi-optimize' == SKILL.name and re.fullmatch(r'[a-z0-9-]{1,64}', meta['name'])
    assert not any(word in meta['name'] for word in ('claude', 'anthropic'))
    assert 40 <= len(meta['description']) <= 1024 and '<' not in meta['description']
    assert len(body.splitlines()) <= 150                 # the body is read whole whenever the skill is used
    named = set(re.findall(r'references/([a-z_-]+\.md)', body))
    assert named == {p.name for p in (SKILL / 'references').glob('*.md')}   # every reference exists, and is pointed to
    for rule in ('Numbers come from the engine', 'contract is fixed before the search', 'do not answer the engine',
                 'engine schedules', 'Failures are data'):
        assert rule in body, rule


def test_what_the_skill_tells_an_agent_to_type_exists():
    agent, top = A.build(), build_cli()
    actions = set(next(a for a in agent._actions if a.dest == 'action').choices) - {'serve'}
    subs = next(a for a in top._actions if a.dest == 'cmd').choices
    known = {opt for sp in list(subs.values()) + list(next(a for a in agent._actions if a.dest == 'action').choices.values())
             for act in sp._actions for opt in act.option_strings}
    known |= {flag for flag, _ in A.START_OPTIONS + A.RESUME_OPTIONS}
    for doc in DOCS:
        text = doc.read_text(encoding='utf-8')
        used = set(re.findall(r'infrarsi agent ([a-z]+)', text))
        assert used <= actions, (doc.name, used - actions)
        for cmd in set(re.findall(r'`infrarsi ([a-z]+)[ `]', text)):
            assert cmd in subs, (doc.name, cmd)
        flags = set(re.findall(r'(?<![\w-])(--[a-z][a-z-]+)', text)) - {'--skip-git-repo-check', '--allowedTools'}
        assert flags <= known, (doc.name, sorted(flags - known))
    everything = '\n'.join(d.read_text(encoding='utf-8') for d in DOCS)
    assert actions <= set(re.findall(r'infrarsi agent ([a-z]+)', everything))     # each action is explained somewhere


def test_the_two_commands_that_can_write_the_users_code_are_shown_only_in_their_safe_form():
    """`infrarsi finish` writes the winner unless --dry-run is given, and `infrarsi apply --branch` does nothing without
    --write: every place the skill shows them says so (an agent types what it is shown)."""
    spans = [s for d in DOCS for s in re.findall(r'`([^`\n]+)`', d.read_text(encoding='utf-8'))]
    finish = [s for s in spans if s.startswith('infrarsi finish')]
    assert finish and all('--dry-run' in s for s in finish), finish
    branch = [s for s in spans if s.startswith('infrarsi apply') and '--branch' in s]
    assert branch and all('--write' in s for s in branch), branch
    from infrarsi.project import agent_session
    import inspect
    assert 'finish {out["run_dir"]} --dry-run' in inspect.getsource(agent_session.stop)   # and the hint a command prints
    text = (SKILL / 'references' / 'answering.md').read_text(encoding='utf-8')
    assert 'fork_turns: "none"' in text and 'Do not answer the request yourself' in text   # Codex copies history by default


def test_the_contract_reference_lists_the_verifiers_checks():
    text = (SKILL / 'references' / 'contract.md').read_text(encoding='utf-8')
    rows = re.findall(r'^\| `([a-z_]+)` \| (every run|final check) \|', text, re.M)
    assert [r[0] for r in rows].count('completes') == 1 and {r[0] for r in rows} == {c.id for c in CHECKS} and len(rows) == 16
    when = {c.id: 'final check' if c.when == 'promotion' else 'every run' for c in CHECKS}
    assert dict(rows) == when


def test_the_sub_agent_definitions():
    meta, body = front_matter(SKILL / 'agents' / 'claude' / 'infrarsi-answerer.md')
    assert meta['name'] == 'infrarsi-answerer' and set(meta) == {'name', 'description', 'tools'}
    tools = [t.strip() for t in meta['tools'].split(',')]
    assert 'Read' in tools and 'Write' in tools and 'Bash' not in tools and 'answer' in body
    try:
        import tomllib
    except ImportError:                                  # Python 3.10
        import tomli as tomllib
    codex = tomllib.loads((SKILL / 'agents' / 'codex' / 'infrarsi-answerer.toml').read_text(encoding='utf-8'))
    assert codex['name'] == 'infrarsi-answerer' and codex['description'] and 'answer.json' in codex['developer_instructions']
    ui = yaml.safe_load((SKILL / 'agents' / 'openai.yaml').read_text(encoding='utf-8'))
    assert '$infrarsi-optimize' in ui['interface']['default_prompt']


def test_the_build_copies_the_engine_whole(built):
    engine = built / 'scripts' / 'engine' / 'infrarsi'
    assert build_skill.package_hash(engine) == build_skill.package_hash(REPO / 'infrarsi') == engine_state.package_hash()
    info = json.loads((built / 'scripts' / 'engine' / 'VERSION.json').read_text(encoding='utf-8'))
    assert info['package_hash'] == engine_state.package_hash() and info['version'] and info['files'] > 300
    assert (engine / 'experience' / 'seed').is_dir() and len(list((engine / 'experience' / 'seed').glob('*.jsonl'))) == 11
    assert not list(built.rglob('__pycache__')) and (built / 'SKILL.md').is_file()
    assert {p.name for p in (built / 'references').iterdir()} == {p.name for p in (SKILL / 'references').iterdir()}
    assert sum(p.stat().st_size for p in built.rglob('*') if p.is_file()) < 30e6     # the upload limit of the Skills API


def test_the_build_refuses_a_destination_that_overlaps_its_sources(tmp_path):
    for out in (REPO / 'agent_skill', SKILL / 'dist', REPO / 'infrarsi' / 'somewhere', SKILL.parent / 'infrarsi-optimize' / 'x'):
        with pytest.raises(SystemExit, match='choose another directory'):
            build_skill.build(out)
    assert (SKILL / 'SKILL.md').is_file() and (REPO / 'infrarsi' / '__init__.py').is_file()   # (nothing was deleted)


def test_the_launcher_without_its_engine_finds_an_installed_package_not_itself(tmp_path):
    """scripts/infrarsi.py is named as the package is; started as a script, its own directory comes first on the path."""
    bare = tmp_path / 'scripts'
    bare.mkdir()
    (bare / 'infrarsi.py').write_bytes((SKILL / 'scripts' / 'infrarsi.py').read_bytes())
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', S.ENV_DIR, S.ENV_MODEL)}
    p = subprocess.run([sys.executable, str(bare / 'infrarsi.py'), 'agent', '-h'], cwd=tmp_path, env={**env, 'PYTHONPATH': str(REPO)},
                       capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120)
    assert p.returncode == 0 and 'infrarsi agent start' in p.stdout, p.stderr
    try:
        import importlib.metadata
        importlib.metadata.version('infrarsi')           # (a checkout installed with pip is found without PYTHONPATH)
    except Exception:                                    # noqa: BLE001
        p = subprocess.run([sys.executable, str(bare / 'infrarsi.py'), 'agent', '-h'], cwd=tmp_path, env=env, capture_output=True,
                           text=True, encoding='utf-8', errors='replace', timeout=120)
        assert p.returncode == 2 and 'no usable InfraRSI engine' in p.stderr


def run_skill(built, *argv, cwd, home, timeout=300):
    """`python <built skill>/scripts/infrarsi.py <argv>` with no PYTHONPATH: only what the skill ships."""
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', S.ENV_DIR, S.ENV_MODEL)}
    env['INFRARSI_HOME'] = str(home)
    p = subprocess.run([sys.executable, str(built / 'scripts' / 'infrarsi.py'), *map(str, argv)], cwd=cwd, env=env,
                       capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def test_the_built_skill_runs_on_its_own(built, tmp_path):
    code, out, err = run_skill(built, 'agent', '-h', cwd=tmp_path, home=tmp_path / 'home')
    assert code == 0 and 'infrarsi agent start' in out, err
    code, out, err = run_skill(built, 'agent', 'status', cwd=tmp_path, home=tmp_path / 'home')
    assert code == 2 and 'no session under' in err
    assert not list(built.rglob('__pycache__'))          # its bytecode cache went to the user's directory
    assert list((tmp_path / 'home' / 'pycache').rglob('*.pyc'))


def test_a_whole_run_through_the_built_skill(built, toy, tmp_path):  # noqa: F811
    home = tmp_path / 'home'
    drv, calls, _ = scripted(toy)
    code, out, err = run_skill(built, 'agent', 'start', '.', '--model', 'claude-opus-5-5', '--screens', '3',
                               '--budget-tokens', '1000000', '--no-doctor', cwd=toy, home=home)
    assert code == 0, (out, err)
    try:
        st = json.loads(out)
        for _ in range(200):
            code, out, err = run_skill(built, 'agent', 'next', '--wait', '20', cwd=toy, home=home)
            st = json.loads(out)
            if st['state'] == 'request':
                r = S.find(st['session'], st['request']['id'])
                assert (r.path / 'README.md').is_file()
                Path(st['request']['answer']).write_text(
                    json.dumps(drv.script(r.role, r.text('system.txt'), r.text('prompt.txt'))), encoding='utf-8')
                code, out, err = run_skill(built, 'agent', 'answer', r.seq, '--tokens', '1200', cwd=toy, home=home)
                assert code == 0 and json.loads(out)['ok'], (out, err)
            elif st['state'] != 'running':
                break
        assert st['state'] == 'finished' and st['result']['screens'] == 3 and st['result']['strategy'] == 'bcomp', (st, err)
        code, out, err = run_skill(built, 'report', st['run_dir'], cwd=toy, home=home)
        assert code == 0 and out.startswith('# InfraRSI run')
        log = (Path(st['session']) / 'engine.log').read_text(encoding='utf-8', errors='replace')
        assert 'Traceback' not in log and 'stage 3/5 (run)' in log
        # nor did the engine it started write its cache into the skill (the profile hook is imported by the training
        # job's own Python, which caches it where it lies)
        assert not [p for p in built.rglob('__pycache__') if 'profhook' not in p.parts]
        assert (home / 'experience').is_dir() or (home / 'ledger.jsonl').is_file()   # the run's experience: the user's home
    finally:
        d = A.current(toy)
        if d is not None and A.engine_alive(d):
            A.stop(d, now=True)


def test_install_check_uninstall(built, tmp_path, capsys):
    home = tmp_path / 'userhome'
    assert install_skill.main(['--home', str(home), '--from', str(built)]) == 0
    for host, root, agent in (('claude', '.claude', 'infrarsi-answerer.md'), ('codex', '.codex', 'infrarsi-answerer.toml')):
        skill = home / root / 'skills' / 'infrarsi-optimize'
        assert (skill / 'SKILL.md').is_file() and (skill / 'scripts' / 'engine' / 'infrarsi' / '__init__.py').is_file()
        assert (home / root / 'agents' / agent).read_bytes() == (built / 'agents' / host / agent).read_bytes()
    capsys.readouterr()
    assert install_skill.main(['--home', str(home), '--check', '--host', 'codex']) == 0
    seen = json.loads(capsys.readouterr().out)
    assert seen['host'] == 'codex' and seen['skill'].endswith('infrarsi-optimize') and seen['engine']['version']
    (home / '.claude' / 'skills' / 'infrarsi-optimize' / 'stale.txt').write_text('x', encoding='utf-8')
    assert install_skill.main(['--home', str(home), '--from', str(built), '--host', 'claude']) == 0   # replaced as a whole
    assert not (home / '.claude' / 'skills' / 'infrarsi-optimize' / 'stale.txt').exists()
    installed = home / '.claude' / 'skills' / 'infrarsi-optimize'      # --from may name the installed copy itself
    assert install_skill.main(['--home', str(home), '--from', str(installed), '--host', 'claude']) == 0
    assert (installed / 'SKILL.md').is_file() and (installed / 'scripts' / 'engine' / 'infrarsi' / '__init__.py').is_file()
    assert install_skill.main(['--home', str(home), '--uninstall']) == 0
    assert not installed.exists() and not (home / '.codex' / 'agents' / 'infrarsi-answerer.toml').exists()
    project = tmp_path / 'proj'                          # a project: Codex reads its skills from .agents/skills
    assert install_skill.main(['--project', str(project), '--from', str(built)]) == 0
    assert (project / '.claude' / 'skills' / 'infrarsi-optimize' / 'SKILL.md').is_file()
    assert (project / '.agents' / 'skills' / 'infrarsi-optimize' / 'SKILL.md').is_file()
    assert (project / '.codex' / 'agents' / 'infrarsi-answerer.toml').is_file() and not (project / '.codex' / 'skills').exists()
    capsys.readouterr()
    assert install_skill.main(['--project', str(project), '--check', '--host', 'codex']) == 0
    assert json.loads(capsys.readouterr().out)['skill'].endswith(str(Path('.agents') / 'skills' / 'infrarsi-optimize'))
    linked, target = project / '.claude' / 'skills' / 'infrarsi-optimize', tmp_path / 'elsewhere'
    try:                                                 # a skill someone linked in: the link is replaced, not followed
        install_skill._remove(linked)
        shutil.copytree(built, target)
        linked.symlink_to(target, target_is_directory=True)
    except OSError:                                      # (Windows without the right to make links)
        pass
    else:
        assert install_skill.main(['--project', str(project), '--from', str(built), '--host', 'claude']) == 0
        assert not linked.is_symlink() and (linked / 'SKILL.md').is_file() and (target / 'SKILL.md').is_file()
    with pytest.raises(SystemExit, match='not a built skill'):
        install_skill.main(['--home', str(home), '--from', str(SKILL)])
    assert install_skill.main(['--project', str(tmp_path / 'p2'), '--host', 'claude']) == 0   # (built from this tree)
    assert (tmp_path / 'p2' / '.claude' / 'skills' / 'infrarsi-optimize' / 'scripts' / 'engine' / 'VERSION.json').is_file()
