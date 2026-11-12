# InfraRSI 的 Claude Code / Codex Skill

[English](README.md) | **简体中文**

`infrarsi-optimize` 让编码 agent 的会话完成 `infrarsi optimize` 做的事：拿到一个已经跑通的训练任务，搜索更快的配置和代码改动，逐个对照原任务验证，只交回通过验证的结果。你直接和 Claude Code 或 Codex 对话，用的就是会话自己的模型，不需要配置 API。

这个 Skill 不是一篇描述 InfraRSI 的提示词。它自带 InfraRSI 引擎，会话围绕引擎做三件事：

| 谁 | 做什么 |
|---|---|
| 引擎（本仓库的 `infrarsi` 包，打包时原样拷进 Skill） | 定下任务约定，启动并计时每一次训练，执行 16 项检查，标定噪声，运行 B-Comp 搜索，把胜出的候选和原任务背靠背复测，记录全部成本，读写经验库，生成报告 |
| 会话里的 agent | 适配代码库（`infrarsi.yaml`），和用户确认约定，看护运行，讲解报告，用户同意后落地改动 |
| 每道题一个全新的子代理 | 回答搜索提出的每个问题（各选项的先验估计、代码审查、补丁、修订），只看到这一道题 |

搜索过程与 `infrarsi optimize --llm claude-cli` 相同。子代理拿到的就是原来那次无状态调用：系统提示词、内容、答案格式都由搜索原样给出，区别只在由谁作答。在 `tests/cpu/search/test_session_parity.py` 的模拟任务上，两种方式的提示词、候选、每次试跑和最终选择完全一致。

## 安装

```bash
python agent_skill/install_skill.py --host both      # 也可以 --host claude、--host codex、--project 目录
python agent_skill/install_skill.py --check
```

这条命令先打包（Skill 自己的文件加一份引擎，约 7 MB），再装到 `~/.claude/skills/` 和 `~/.codex/skills/`，并把答题子代理的定义放到 `~/.claude/agents/` 和 `~/.codex/agents/`。引擎用训练环境的 Python 运行：3.11 及以上加 PyYAML（3.10 还需要 `tomli`）。不需要联网下载。

Codex 的会话默认在沙箱里运行。引擎需要能写 `~/.infrarsi`（经验库、账本、给测量结果签名的密钥）；GPU 在别的机器上时还需要联网。会话请求时，让"启动引擎"那一条命令在沙箱外运行即可。

## 使用

在训练项目里打开会话，说"帮我把这个训练任务调快"（或者在 Claude Code 里用 `/infrarsi-optimize`，在 Codex 里用 `$infrarsi-optimize`）。agent 会起草任务约定，告诉你哪些可以改、哪些不能动，你确认后才开始。搜索是一个独立的后台进程，会话断开不影响它。结束时你会拿到报告，改动只有在你同意后才写回。

agent 用的都是普通命令，手动执行也可以。Skill 本身不提供名为 `infrarsi` 的可执行文件：下面的 `infrarsi` 指 `python "<Skill 目录>/scripts/infrarsi.py"`（或者用 `pip install -e .` 装好的仓库里的 `infrarsi`）。

```bash
infrarsi agent start . --model claude-opus-5-5 --screens 16
infrarsi agent next          # 等到：有一道题要答、运行结束、出错
infrarsi agent answer 3      # 交回 <题目目录>/answer.json
infrarsi agent status        # 查看运行到哪了
infrarsi agent note "..."    # 给后面的提案一条提示
infrarsi agent stop          # 停止搜索，复验当前最优
infrarsi agent resume        # 从检查点继续一次被停止或中断的运行
```

## 目录

```
agent_skill/
  infrarsi-optimize/         Skill 本体：SKILL.md、references/、scripts/infrarsi.py、agents/（子代理定义）
  build_skill.py             Skill 加引擎，打包到 agent_skill/dist/infrarsi-optimize/
  install_skill.py           打包并安装到 Claude Code / Codex
infrarsi/agent/backends/session.py     答题后端：把一次模型调用变成会话要回答的一道题
infrarsi/project/agent_session.py      `infrarsi agent ...` 命令组
```

测试：`tests/cpu/agent/test_session_backend.py`、`tests/cpu/project/test_agent_session.py`、`tests/cpu/search/test_session_parity.py`、`tests/cpu/skill/test_skill_package.py`。
