#!/usr/bin/env python3
"""Create a competition workspace for the kaggle-war-room workflow.

Everything the skill writes goes under ONE folder in the user's project:

  <project>/kaggle-war-room-workspace/        (workspace root; reused if found upwards from cwd)
  ├── workspace.json                       index of competitions + the active one
  ├── README.md
  └── <slug>/                              WORKDIR: settings.yaml, rules.md, baseline.md,
      ├── code/  preds/  submissions/      experiments.csv, decisions.md, daily_brief.md,
      └── snapshots/                       plan.md, review.md, submissions.md

The new competition is registered in workspace.json and made active, so later scripts and
roles find it with `workspace.py locate` (no paths to remember).
Existing files are never overwritten unless --force is given.

Examples:
  python init_competition.py my-comp --deadline 2026-12-31 --task-type segmentation
  python init_competition.py my-comp --root D:/kaggle/kaggle-war-room-workspace
"""
import argparse
import datetime as dt
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _workspace as wsmod  # noqa: E402

SKILL_DIR = Path(__file__).resolve().parent.parent
TEMPLATES = [
    "rules.md", "baseline.md", "experiments.csv", "decisions.md",
    "daily_brief.md", "plan.md", "review.md", "submissions.md",
]
TASK_TYPES = ["classification", "segmentation", "detection", "other"]


def render(text, slug, today):
    return text.replace("{slug}", slug).replace("{date}", today)


def make_settings(slug, deadline, task_type, team_name):
    text = (SKILL_DIR / "config" / "settings.example.yaml").read_text(encoding="utf-8")
    text = text.replace("slug: <comp-slug>", f"slug: {slug}", 1)
    if deadline:
        text = re.sub(r"(\n  deadline: )null", rf"\g<1>{deadline}", text, count=1)
    if task_type:
        text = re.sub(r"(\n  task_type: )null", rf"\g<1>{task_type}", text, count=1)
    if team_name:
        text = re.sub(r"(\n  team_name: )null", rf'\g<1>"{team_name}"', text, count=1)
    return text


def flavor_on(settings_path):
    """`flavor: false` in settings.yaml turns the war-room touches off."""
    text = settings_path.read_text(encoding="utf-8")
    return not re.search(r"(?m)^flavor:\s*(false|no|off)\b", text, re.I)


def banner(slug, deadline):
    """Opening banner, shown once when a competition is created. ASCII only (Windows consoles)."""
    if deadline:
        days = (dt.date.fromisoformat(deadline) - dt.date.today()).days
        when = f"Deadline: {deadline}  (D-{days})"
    else:
        when = "Deadline: not set yet"
    w = 52
    lines = [
        "=" * w,
        f"  KAGGLE WAR ROOM  ::  OPERATION {slug.upper()}"[:w],
        f"  Status: OPEN        {when}"[:w],
        "-" * w,
        "  Recon ....... Scout      Command ..... Strategist",
        "  Red Team .... Critic     Engineering . Builder",
        "=" * w,
        "  First orders: confirm rules.md, then the baseline.",
        "",
    ]
    return "\n".join(lines)


def write(path, content, force, created, skipped):
    if path.exists() and not force:
        skipped.append(path.name)
        return
    path.write_text(content, encoding="utf-8", newline="")
    created.append(path.name)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("slug", help="Kaggle competition slug (from the competition URL)")
    p.add_argument("--root", help=f"workspace root (default: existing {wsmod.DIRNAME}/ found upwards from cwd, "
                                  f"else ./{wsmod.DIRNAME})")
    p.add_argument("--deadline", help="final submission deadline, YYYY-MM-DD (UTC)")
    p.add_argument("--task-type", choices=TASK_TYPES, help="task type, used to pick references")
    p.add_argument("--team-name", help="your Kaggle team name, used to find your leaderboard rank")
    p.add_argument("--force", action="store_true", help="overwrite existing files")
    args = p.parse_args(argv)

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.slug):
        p.error("slug must contain only letters, digits, '-' and '_'")
    if args.deadline:
        try:
            dt.date.fromisoformat(args.deadline)
        except ValueError:
            p.error("--deadline must be YYYY-MM-DD")

    root = Path(args.root).resolve() if args.root else (wsmod.find_root() or wsmod.default_root())
    ws = root / args.slug
    ws.mkdir(parents=True, exist_ok=True)
    for sub in ("snapshots", "preds", "code", "submissions"):
        (ws / sub).mkdir(exist_ok=True)
    wsmod.register(root, args.slug, make_active=True)

    today = dt.date.today().isoformat()
    created, skipped = [], []
    write(ws / "settings.yaml", make_settings(args.slug, args.deadline, args.task_type, args.team_name),
          args.force, created, skipped)
    for name in TEMPLATES:
        src = SKILL_DIR / "templates" / name
        write(ws / name, render(src.read_text(encoding="utf-8"), args.slug, today), args.force, created, skipped)

    if "settings.yaml" in created and flavor_on(ws / "settings.yaml"):
        print(banner(args.slug, args.deadline))
    print(f"Workspace root: {root}")
    print(f"WORKDIR (active): {ws.resolve()}")
    if created:
        print("Created:  " + ", ".join(created))
    if skipped:
        print("Kept existing (use --force to overwrite): " + ", ".join(skipped))
    print("\nNext steps:")
    print(f"  1. Edit {ws / 'settings.yaml'} (models, compute, submissions, deadline).")
    print("  2. Read the competition pages and fill rules.md (references/rules-checklist.md).")
    print("  3. Ask the user to confirm rules.md, then build the baseline.")
    if not args.deadline:
        print("  ! No --deadline given: set competition.deadline in settings.yaml so phases work.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
