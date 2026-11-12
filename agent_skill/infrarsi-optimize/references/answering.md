# Answering a request

When `infrarsi agent next` returns `"state": "request"`, the engine is waiting for one answer. It has written the
question to `request.dir`: `README.md` (what to do), `system.txt`, `prompt.txt`, sometimes `schema.json`. The answer goes
to `request.answer` (normally `answer.json` in that directory).

## Why a fresh sub-agent

The search treats each answer as an independent estimate and corrects it against measurements. You are not independent:
you have seen the measurements, the user's hopes and your own earlier guesses. An answerer that has seen none of this
gives the search what it expects. So: one request, one new sub-agent, and nothing from this conversation in its prompt.

## The prompt (use it as it is)

```
Answer InfraRSI request <id>. Its directory is <dir>.
Read README.md in that directory and do exactly what it says: read system.txt and prompt.txt (and schema.json if it
is there), then write your answer to <answer file name> in that directory. Write no other file and run nothing.
When the file is written, reply with the single word: done
```

Do not add advice, numbers, file contents or your own view of the job.

The answerer must be the model you named with `--model` at the start (normally your own): the engine weighs the answers
by that name. Do not hand a request to a smaller or cheaper model.

## Claude Code

Use the Agent tool:

- `subagent_type: infrarsi-answerer` when that agent is installed (the installer puts it in `~/.claude/agents/`);
- otherwise `subagent_type: general-purpose`.

Never use a fork (it inherits this conversation), and never continue an earlier answerer with a message: each request
gets a new one. Leave the sub-agent's model as it is (it inherits yours).

## Codex

Spawn a sub-agent with the prompt above and **no history from this thread**: `spawn_agent` with
`fork_turns: "none"` (the default would copy this conversation into it), and `agent_type: "infrarsi-answerer"` when that
agent is installed (`~/.codex/agents/infrarsi-answerer.toml`), otherwise the default agent. Wait for it, then hand the
answer in. If sub-agents are not available in this session, start a fresh non-interactive run of the CLI with your own
model instead, for example:

`codex exec --skip-git-repo-check -s workspace-write -m <your model> -C "<dir>" "<the prompt above>"`

## If neither is possible

Do not answer the request yourself. Stop the run with `infrarsi agent stop` and tell the user that this session cannot
start an independent answerer; the unattended form in `workflow.md` (the engine asks the local command-line tool itself)
needs none.

## Handing in

`infrarsi agent answer <id>` checks the answer file and hands it to the engine. Add `--tokens N` when your host tells
you how many tokens the sub-agent used, and `--by <model name>` if you like.

| Result | Meaning | What you do |
|---|---|---|
| `"ok": true` | The engine took it | `infrarsi agent next` |
| `"ok": false`, `"closed": false` | The answer was not usable (not one JSON object, or not the shape asked for); `feedback.txt` in the directory says why | Start a **new** sub-agent with the same prompt plus: `Your previous answer was not accepted. Read feedback.txt in the directory first, then write a corrected answer.` Then `infrarsi agent answer <id>` again |
| `"ok": false`, `"closed": true` | The request is over (answered before, timed out, or the last attempt failed) | `infrarsi agent next`; the engine goes on without this answer |

Do not edit the answer file yourself, not even to repair its JSON.

## What the answerer may use

Only the request directory. It does not read the project, run commands or look at other requests. The engine has put
into the request what it wants the model to know (the task, the measured history, the code it may patch); an answerer
that reads more answers a different question from the one the search asked.
