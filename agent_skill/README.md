# InfraRSI as a skill for Claude Code and Codex

**English** | [简体中文](README_ZH.md)

`infrarsi-optimize` lets a coding-agent session do what `infrarsi optimize` does: take a training job that runs, search
for faster settings and code changes, verify each one against the original job, and hand back only what passed. You talk
to Claude Code or Codex; the session uses its own model, so there is no API key to configure.

The skill is not a prompt that describes InfraRSI. It ships the InfraRSI engine and gives the session three jobs around
it:

| Who | Does what |
|---|---|
| The engine (this repository's `infrarsi` package, copied into the skill) | Fixes the task contract, runs and times every training run, applies the 16 checks, calibrates noise, runs the B-Comp search, re-runs the winner back to back with the original, keeps the cost ledger, reads and writes the experience library, writes the report |
| The session's agent | Adapts the project (`infrarsi.yaml`), agrees the contract with the user, keeps the run going, explains the report, applies the change when the user agrees |
| A fresh sub-agent per request | Answers each question the search asks (a prior over the settings, a code review, patches, a revision), seeing only that question |

The search is the one `infrarsi optimize --llm claude-cli` runs. The request a sub-agent answers is the same stateless
call - system prompt, prompt and answer schema as the search built them - so who answers is the only difference. On the
simulated job in `tests/cpu/search/test_session_parity.py` the two give identical prompts, candidates, runs and selection.

## Install

```bash
python agent_skill/install_skill.py --host both      # or: --host claude | --host codex | --project DIR
python agent_skill/install_skill.py --check
```

This builds the skill (its files plus a copy of the engine, about 7 MB) and puts it in `~/.claude/skills/` and
`~/.codex/skills/`, with the answering sub-agent's definition in `~/.claude/agents/` and `~/.codex/agents/`. The engine
runs with the Python of the training environment: 3.11 or newer with PyYAML (3.10 also needs `tomli`). Nothing is
downloaded.

In Codex, whose sessions run in a sandbox by default, the engine must be allowed to write `~/.infrarsi` (its experience
library, its ledger, the key it signs measurements with) and, for GPUs on other machines, to use the network: let the
one command that starts the engine run outside the sandbox when the session asks.

## Use

In a session opened in the training project: "make this training job faster" (or `/infrarsi-optimize` in Claude Code,
`$infrarsi-optimize` in Codex). The agent drafts the contract, shows you what may change and what must not, and starts
the run after your go. The search runs as a background process of its own; a session that disconnects loses nothing.
At the end you get the report, and the change is written only when you say so.

The commands the agent uses are ordinary ones and work by hand too. The skill has no `infrarsi` executable of its own:
below, `infrarsi` stands for `python "<skill dir>/scripts/infrarsi.py"` (or the `infrarsi` of a checkout installed with
`pip install -e .`).

```bash
infrarsi agent start . --model claude-opus-5-5 --screens 16
infrarsi agent next          # waits for: a request to answer, the end, a failure
infrarsi agent answer 3      # hands in <request dir>/answer.json
infrarsi agent status        # where the run stands
infrarsi agent note "..."    # guidance for later proposals
infrarsi agent stop          # stop the search, verify the best candidate so far
infrarsi agent resume        # continue a stopped or dead run from its checkpoint
```

## Layout

```
agent_skill/
  infrarsi-optimize/         the skill: SKILL.md, references/, scripts/infrarsi.py, agents/ (sub-agent definitions)
  build_skill.py             skill + engine -> agent_skill/dist/infrarsi-optimize/
  install_skill.py           build and install for Claude Code / Codex
infrarsi/agent/backends/session.py     the backend: a model call as a request the session answers
infrarsi/project/agent_session.py      `infrarsi agent ...`
```

Tests: `tests/cpu/agent/test_session_backend.py`, `tests/cpu/project/test_agent_session.py`,
`tests/cpu/search/test_session_parity.py`, `tests/cpu/skill/test_skill_package.py`.
