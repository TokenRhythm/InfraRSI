"""C-NAT-free / C-NAT-infra (design §4.3, deviation D-AR-6): the autoresearch protocol driven by the basic agent.

The agent gets program.md verbatim (C-NAT-infra: plus the infra appendix) and a git repository on the leased node:
  <runs>/<run_id>/agent/home/repo   the arm uid: README.md, prepare.py, program.md, train.py (base), pyproject.toml,
                                    branch autoresearch/<tag> checked out (setup steps 1, 2 and 4 done by the scaffold)
  <runs>/<run_id>/agent/home        HOME (.gitconfig; ~/.cache/autoresearch -> the frozen data and tokenizer), 0700
  <runs>/<run_id>/agent/tool        root: the bash tool's scripts and outputs
The arm is one trust domain (task.ARM_UIDS: C-NAT-free 64001, C-NAT-infra 64002): the agent's shell and every training
it starts run under that uid; <run_id>/ and agent/ are root:<arm gid> 0750, so no other arm can enter them, and every
process of the uid is killed after each tool call and each experiment (Codex W9 B-02/B-03).
keep/revert is the program's own git workflow (git 2.34 is installed on WINH); keep_revert='copy' replaces it by the basic
agent's directory-copy tools keep/revert (D-AR-6 fallback). `uv run train.py > run.log 2>&1` becomes the
tool run_experiment: the driver takes the working tree's train.py, runs it through AutoresearchRunner in a fresh
candidate directory with the original prepare.py (same runner and judge as C-IS / C-RND), copies the full output to
run.log in the repository and returns the summary (infra track: valid and the reasons of the trusted checks).
"""
from __future__ import annotations

from ..common.basic_agent import BasicAgent, EnvTool, ShellExecutor, best_screen, keep_revert_tools
from ..common.env_judge import EnvJudge
from ..common.remote_exec import new_token, nobody_cmd, q
from . import source
from .hooks import APPENDIX
from .task import ARM_UIDS                                       # noqa: F401 - re-exported (C-NAT uids live in task.py)

NOTES = '''
---
Notes for this environment (scaffold of this replication):
- Your working directory is the repository {repo} on a machine with one H100 (the GPU is yours). Setup steps 1, 2 and 4
  are done: the run tag is `{tag}`, branch `autoresearch/{tag}` is checked out, and ~/.cache/autoresearch/ holds the data
  and the tokenizer. Do step 3 (read the in-scope files) and step 5 (results.tsv), then start the experiment loop right
  away: the human has confirmed the setup and is away.
- There is no `uv` here. Instead of `uv run train.py > run.log 2>&1`, call {{"tool":"run_experiment"}}: it trains the
  train.py of your working tree for the fixed 5-minute budget on the GPU (in a fresh copy with the original prepare.py),
  writes the full output to run.log in your working directory and returns the summary. Then read the results as
  described above. Do not train with bash.
- {checks}
- One experiment takes about 6-7 minutes. Your budget is {experiments} experiments and {hours:.0f} hours of wall clock in
  total, your own thinking time included; the run ends when either is used up, so keep going until then.
'''
COPY_NOTE = ('\n- git is not available here: instead of advancing the branch call {"tool":"keep"} (the current train.py becomes the '
             'kept version), and instead of git reset call {"tool":"revert"} (train.py is restored from the kept version).')
CHECKS = {'free': 'The recorded result of an experiment is the val_bpb it prints.',
          'infra': ('After every experiment the trusted checks of the appendix run; run_experiment returns "valid" and the '
                    'reasons. An invalid experiment does not count: keep only valid improvements.')}


def task_text(task, *, repo: str, tag: str, keep_revert: str = 'git') -> str:
    text = source.upstream('program.md')
    if task.raw.get('track') == 'infra':
        text = text.rstrip() + '\n\n' + APPENDIX
    return text + NOTES.format(repo=repo, tag=tag, checks=CHECKS[task.raw.get('track', 'infra')],
                               experiments=task.budget['screens'], hours=task.budget['wall_seconds'] / 3600) + (
        COPY_NOTE if keep_revert == 'copy' else '')


def layout(run_id: str, runs_root: str = source.NODE_RUNS) -> dict:
    a = f'{runs_root}/{run_id}/agent'
    return {'agent': a, 'home': f'{a}/home', 'repo': f'{a}/home/repo', 'tool': f'{a}/tool', 'code': f'{a}/code'}


def setup_home(ops, run_id: str, *, tag: str, uid: int, runs_root: str = source.NODE_RUNS,
               prepare_variant: str = source.DEFAULT_PREPARE, keep_revert: str = 'git') -> dict:
    """Create the agent's repository (upstream files + base train.py; with git: one commit, branch autoresearch/<tag>),
    owned by the arm's uid (home 0700). The run directory is root:root 0711 (traverse-only, never re-grouped); the
    agent and code directories are root:<arm gid> 0750 (no other uid enters); the tool dir is root-only."""
    L = layout(run_id, runs_root)
    ops.mkdirs([f'{runs_root}/{run_id}'], mode=0o711)
    ops.mkdir_domain([L['agent'], L['code']], mode=0o750, gid=uid)
    ops.mkdirs([L['tool']], mode=0o700)
    ops.mkdirs([L['home'], f'{L["home"]}/tmp', L['repo']], mode=0o700, uid=uid, gid=uid)
    files = {f'{L["repo"]}/{n}': source.upstream(n) for n in ('README.md', 'program.md', 'pyproject.toml')}
    files[f'{L["repo"]}/prepare.py'] = source.candidate_prepare(prepare_variant)
    files[f'{L["repo"]}/train.py'] = source.base_train()
    files[f'{L["repo"]}/.gitignore'] = 'run.log\nresults.tsv\n__pycache__/\n'
    files[f'{L["home"]}/.gitconfig'] = '[user]\n\tname = autoresearch-agent\n\temail = agent@localhost\n[init]\n\tdefaultBranch = master\n'
    ops.write_files(files, mode=0o644, uid=uid, gid=uid)
    ops.symlinks({f'{L["home"]}/.cache/autoresearch/data': source.NODE_DATA, f'{L["home"]}/.cache/autoresearch/tokenizer': source.NODE_TOKENIZER})
    res = ops.sh(f'chown -hR {int(uid)}:{int(uid)} {q(L["home"])}', timeout=120)
    if res.get('code') not in (0, None):
        raise RuntimeError(f'agent home setup failed: {res.get("stderr", "")[-500:]}')
    if keep_revert == 'git':
        # The repository is created by the arm's own uid: root never runs git inside a tree an untrusted uid owns (and
        # git 2.34 on WINH refuses a foreign owner unless safe.directory is set in the system/global config).
        git = ('git init -q && git symbolic-ref HEAD refs/heads/master && git add -A && git commit -q -m baseline && '
               f'git checkout -q -b {q("autoresearch/" + tag)}')
        res = ops.sh(nobody_cmd(git, home=L['home'], cwd=L['repo'], uid=uid, timeout=120,
                                extra={'GIT_CONFIG_GLOBAL': f'{L["home"]}/.gitconfig'}), timeout=240)
        if res.get('code') not in (0, None):
            raise RuntimeError(f'agent repository setup failed: {(res.get("stderr") or res.get("stdout") or "")[-500:]}')
    return L


def experiment_tool(task, runner, ops, L: dict, *, run_id: str, uid: int) -> EnvTool:
    judge = EnvJudge('autoresearch')
    state = {'n': 0}
    if getattr(runner, 'train_uid', None) != uid:
        raise ValueError(f'the agent (uid {uid}) and its trainings (uid {getattr(runner, "train_uid", None)}) must share the arm uid')

    def run_experiment(args: dict, agent: BasicAgent) -> dict:
        text = ops.read_user_text(f'{L["repo"]}/train.py', L['home'], max_bytes=1 << 20)     # never follows a planted link
        if text is None:
            return {'error': 'train.py is missing, not a regular file, or larger than 1 MB'}
        state['n'] += 1
        cid = f'e{state["n"]:02d}'
        if agent.out:                             # the locked re-score needs the experiment's text, not only its hash
            (agent.out / 'experiments').mkdir(parents=True, exist_ok=True)
            (agent.out / 'experiments' / f'{cid}.train.py').write_text(text, encoding='utf-8', newline='\n')
        receipt = runner.run(task, {}, {'train.py': text}, 'screen', run_id, candidate_id=cid, wall_cap=agent.wall_left())
        verdict = judge(receipt, None, None)
        agent.record_experiment(receipt, verdict, candidate_id=cid, kind='L1' if text != source.base_train() else 'start',
                                extra={'train_py_sha256': (receipt.get('files') or {}).get('train.py')})
        cand = receipt.get('candidate_dir')
        if cand:                                  # the program reads run.log in its working tree (written without following links)
            log = ops.read_files([f'{cand}/train.log'], max_bytes=8 << 20)[f'{cand}/train.log'] or b''
            ops.write_user_file(f'{L["repo"]}/run.log', L['home'], log, uid=uid, gid=uid, mode=0o644)
        s = receipt.get('summary') or {}
        out = {'experiment': cid, 'exit_code': s.get('train_exit'), 'timed_out': s.get('timed_out'),
               'summary': {k: s.get(k) for k in ('val_bpb_printed', 'training_seconds', 'total_seconds', 'peak_vram_mb',
                                                 'mfu_percent', 'num_steps')}, 'log': 'full output in run.log'}
        if receipt.get('track') == 'infra':
            out.update(valid=receipt.get('valid'), val_bpb_trusted=(receipt.get('metric') or {}).get('raw', {}).get('val_bpb_judge'),
                       invalid_reasons=(receipt.get('invalid_reasons') or [])[:6])
        return out

    return EnvTool(name='run_experiment', description='{"tool":"run_experiment","note":"<what this experiment tries>"}  trains the '
                   'current train.py for the fixed budget (about 6-7 min), writes run.log, returns the summary',
                   properties={}, fn=run_experiment, counts_as_experiment=True,
                   admit=lambda agent: runner.admits(agent.ledger, agent.budget))


def run_native(task, driver, runner, ops, *, run_id: str, out_dir=None, tag: str | None = None, clock=None,
               keep_revert: str = 'auto', uid: int | None = None, **agent_kw) -> dict:
    """One C-NAT arm: the basic agent until a budget ends; selected = its best valid experiment (published protocol).
    keep_revert: 'git' (program.md's own workflow), 'copy' (D-AR-6 fallback tools) or 'auto' (git when installed)."""
    tag = tag or run_id.replace('_', '-')[:24]
    if keep_revert == 'auto':
        keep_revert = 'git' if ops.sh('command -v git', timeout=60).get('code') in (0, None) else 'copy'
    uid = int(uid if uid is not None else task.raw['uid'])
    L = setup_home(ops, run_id, tag=tag, runs_root=runner.runs_root,
                   prepare_variant=task.raw['prepare_variant'], keep_revert=keep_revert, uid=uid)
    shell = ShellExecutor(ops, home=L['home'], cwd=L['repo'], work_dir=L['tool'], token=new_token('isx-cnat'), code_dir=L['code'], uid=uid,
                          path_dirs=(f'{source.NODE_ENV}/bin',), gpus=tuple(runner.gpus),
                          extra_env={'HF_HOME': f'{L["home"]}/hf_home', 'GIT_CONFIG_GLOBAL': f'{L["home"]}/.gitconfig'})
    kw = {'clock': clock} if clock is not None else {}
    tools = [experiment_tool(task, runner, ops, L, run_id=run_id, uid=uid)]
    if keep_revert == 'copy':
        tools += keep_revert_tools(ops, repo=L['repo'], user_root=L['home'], head_dir=f'{L["agent"]}/head', files=('train.py',),
                                   uid=uid)
    agent = BasicAgent(driver=driver, executor=shell, task_text=task_text(task, repo=L['repo'], tag=tag, keep_revert=keep_revert),
                       env_tools=tools, budget=task.search_budget(),
                       run_id=run_id, env='autoresearch', task_id=task.id, arm=task.arm, out_dir=out_dir, **kw, **agent_kw)
    agent.run()
    best = best_screen({'screens': agent.screens})
    return agent.trace(selected=best and best['candidate_id'], selected_value=best and best['step_time'],
                       extra={'tag': tag, 'layout': L, 'track': task.raw.get('track'), 'keep_revert': keep_revert, 'uid': uid})
