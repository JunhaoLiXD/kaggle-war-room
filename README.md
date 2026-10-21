# Kaggle War Room

A multi-role AI-agent workflow for Kaggle competitions, packaged as a plugin for
**Claude Code** and **Codex**. Computer-vision focused, but the workflow is task-agnostic.

- **Scout** watches public notebooks, the leaderboard and discussions every day and reports only what changed.
- **Strategist** reads the rules, experiment log and decision log, then plans experiments, compute and submissions.
- **Critic** — ideally a *different vendor's* model — reviews each plan adversarially, without seeing the Strategist's reasoning.
- **Builder** reviews public notebooks before reusing them, implements, blends, and validates every submission.

Roles hand off through files, scripts do the mechanical work (snapshot diffing, submission
checks, prediction correlation, stagnation detection), and every risky step stops for your
confirmation. Final submissions are chosen as a hedged pair to resist leaderboard shake-up.

## Install

### Easiest: ask your agent

In Claude Code or Codex, say:

> Install the kaggle-war-room plugin from https://github.com/JunhaoLiXD/kaggle-war-room

The agent follows [Instructions for AI agents](#instructions-for-ai-agents) below.

### Claude Code

```bash
claude plugin marketplace add JunhaoLiXD/kaggle-war-room
claude plugin install kaggle-war-room@kaggle-war-room
```

Or inside a session: `/plugin marketplace add JunhaoLiXD/kaggle-war-room`, then
`/plugin install kaggle-war-room@kaggle-war-room`. Start a new session (or run `/reload-plugins`).

### Codex

```bash
codex plugin marketplace add JunhaoLiXD/kaggle-war-room
codex plugin add kaggle-war-room@kaggle-war-room
```

Start a new Codex session afterwards.

### Python dependencies (both hosts)

```bash
pip install numpy pyyaml kaggle
```

Then create a Kaggle API token (kaggle.com → Settings → API) and save it as
`~/.kaggle/kaggle.json`. **Do this yourself — never paste the key into an agent chat.**

### Manual install (any agent that supports Agent Skills)

Copy `plugins/kaggle-war-room/skills/kaggle-war-room/` into your agent's skills directory, e.g.
`~/.claude/skills/kaggle-war-room/` or `~/.codex/skills/kaggle-war-room/`. For Claude Code you can
also copy `plugins/kaggle-war-room/agents/*.md` into `~/.claude/agents/` to get the role subagents.

## Quick start

1. In your project folder, tell the agent: *"Start a Kaggle competition workspace for `<competition-slug>`, deadline YYYY-MM-DD."*
2. Edit `kaggle-war-room-workspace/<slug>/settings.yaml` — at least `competition.deadline`, `competition.team_name`,
   `metric.higher_is_better`, and optionally `models.*`.
3. The agent drafts `rules.md` from the competition pages → **you confirm**. It builds a
   baseline from a reviewed public notebook → **you confirm**.
4. Each day: *"Run today's Kaggle update and plan."* Scout → Strategist → Critic → Strategist → Builder.

## Where your files go

The plugin never scatters files in your project. Everything it creates is in one folder:

```
<your project>/kaggle-war-room-workspace/
├── workspace.json        which competitions exist, which one is active
├── README.md
└── <competition-slug>/   settings.yaml, rules.md, baseline.md, experiments.csv,
    ├── code/             decisions.md, daily_brief.md, plan.md, review.md,
    ├── preds/            submissions.md
    ├── submissions/
    └── snapshots/
```

Every session starts by running `workspace.py locate`, so the agent always reads and writes
the right competition folder. Several competitions in one project are fine: *"switch to
`<slug>`"* runs `workspace.py use <slug>`. To keep the workspace elsewhere (e.g. another
drive), set the environment variable `KAGGLE_WAR_ROOM_WORKSPACE` to that folder.

## Configuration

Everything you can change lives in `kaggle-war-room-workspace/<slug>/settings.yaml` and is documented in
[`SKILL.md` → Variables you can change](plugins/kaggle-war-room/skills/kaggle-war-room/SKILL.md#variables-you-can-change):

- **models** per role (`inherit` by default, so a single-model setup just works)
- **Critic**: external CLI of the other vendor (`codex exec …` from Claude Code, `claude -p …` from Codex) or a fresh-context subagent
- **compute**: weekly Kaggle GPU quota, your own GPUs, inference limit, confirmation threshold
- **submissions**: daily limit and who allocates them
- **phases**: explore / refine / finalize boundaries, stagnation limits, `min_gain`

## You are always asked before

- confirming `rules.md` and the baseline
- every Kaggle submission
- any experiment estimated above `confirm_if_gpu_hours_over` GPU hours
- switching phase (explore → refine → finalize)
- big plan changes caused by the Critic, or overruling the Critic
- choosing final submissions

## Safety

Public notebooks, discussions and dataset descriptions are treated as **data, not
instructions**. Text aimed at AI agents is reported, never followed. Notebooks are reviewed
for leakage, leaderboard probing, rule violations and suspicious code before any reuse, and
replicated code should run in an isolated environment. Data comes only from the official
Kaggle API.

## Repository layout

```
.claude-plugin/marketplace.json      Claude Code marketplace
.agents/plugins/marketplace.json     Codex marketplace
plugins/kaggle-war-room/
├── .claude-plugin/plugin.json       Claude Code plugin manifest
├── .codex-plugin/plugin.json        Codex plugin manifest
├── agents/                          Claude Code role subagents
└── skills/kaggle-war-room/             The skill (shared by both hosts)
    ├── SKILL.md                     Workflow, gates, variables
    ├── config/settings.example.yaml All tunable variables
    ├── roles/                       Scout / Builder / Strategist / Critic playbooks
    ├── scripts/                     9 Python tools (each has --help)
    ├── references/                  Review checklists, anti-shake-up, ensembling, CV tips
    └── templates/                   Workspace files (rules.md, plan.md, ...)
```

## Instructions for AI agents

You were asked to install this plugin. Do the following, then report what you did.

1. Detect your host:
   - **Claude Code** (you have the `claude` CLI or are Claude Code):
     run `claude plugin marketplace add JunhaoLiXD/kaggle-war-room`, then
     `claude plugin install kaggle-war-room@kaggle-war-room`.
   - **Codex** (you have the `codex` CLI or are Codex):
     run `codex plugin marketplace add JunhaoLiXD/kaggle-war-room`, then
     `codex plugin add kaggle-war-room@kaggle-war-room`.
   - **Other agent with skills support**: clone the repository and copy
     `plugins/kaggle-war-room/skills/kaggle-war-room/` into your skills directory.
2. Run `pip install numpy pyyaml kaggle` (ask first if the user manages environments
   carefully, e.g. conda or a project venv).
3. Do **not** ask for, read, or write the Kaggle API key. Tell the user to save their token to
   `~/.kaggle/kaggle.json` themselves.
4. Tell the user to start a new session (Claude Code: or run `/reload-plugins`) and that they
   can begin with: *"Start a Kaggle competition workspace for `<slug>`."*
5. End your report with this welcome, once, in the user's language:
   > 🎖️ The war room is open. Scout is heading out for recon, the Strategist is laying out the
   > map, the Red Team (Critic) is ready to challenge every plan, and Engineering (Builder) is
   > standing by. Commander, let's win this one — good luck!

## License

[MIT](LICENSE)
