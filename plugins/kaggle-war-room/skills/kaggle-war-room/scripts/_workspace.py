"""Shared helpers: locate the kaggle-war-room workspace and the active competition.

Layout (everything the skill writes lives under ONE folder in the user's project):

  <project>/kaggle-war-room-workspace/
  ├── workspace.json          index: {"active": "<slug>", "competitions": {"<slug>": {...}}}
  ├── README.md               what this folder is
  └── <slug>/                 one folder per competition (= WORKDIR)

The workspace root is found by, in order:
  1. env var KAGGLE_WAR_ROOM_WORKSPACE (path to the root folder)
  2. walking up from the current directory looking for kaggle-war-room-workspace/workspace.json
"""
import datetime as dt
import json
import os
from pathlib import Path

DIRNAME = "kaggle-war-room-workspace"
INDEX = "workspace.json"
ENV = "KAGGLE_WAR_ROOM_WORKSPACE"

ROOT_README = """# kaggle-war-room workspace

Everything the kaggle-war-room plugin creates lives in this folder. Do not move files out of it.

- `workspace.json` - index of competitions and which one is active
- `<competition-slug>/` - one folder per competition: settings.yaml, rules.md, baseline.md,
  experiments.csv, decisions.md, daily_brief.md, plan.md, review.md, submissions.md,
  code/, preds/, submissions/, snapshots/

Find the active competition folder with:
  python <SKILL_DIR>/scripts/workspace.py locate
"""


def find_root(start=None):
    env = os.environ.get(ENV)
    if env:
        p = Path(env).expanduser().resolve()
        return p if (p / INDEX).exists() else None
    cur = Path(start or Path.cwd()).resolve()
    for d in [cur, *cur.parents]:
        if (d / DIRNAME / INDEX).exists():
            return d / DIRNAME
        if d.name == DIRNAME and (d / INDEX).exists():
            return d
    return None


def default_root():
    env = os.environ.get(ENV)
    return Path(env).expanduser().resolve() if env else Path.cwd().resolve() / DIRNAME


def load_index(root):
    path = Path(root) / INDEX
    if not path.exists():
        return {"version": 1, "active": None, "competitions": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_index(root, index):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / INDEX).write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(ROOT_README, encoding="utf-8")


def register(root, slug, make_active=True):
    index = load_index(root)
    index.setdefault("competitions", {}).setdefault(slug, {"path": slug, "created": dt.date.today().isoformat()})
    if make_active or not index.get("active"):
        index["active"] = slug
    save_index(root, index)
    return index


def settings_path(workdir, value):
    """A path from settings.yaml: absolute as written, relative ones are relative to WORKDIR."""
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else Path(workdir) / p


def resolve_workdir(workdir=None, slug=None):
    """Return the competition folder: explicit --workdir, else <root>/<slug or active>."""
    if workdir:
        return Path(workdir).expanduser().resolve()
    root = find_root()
    if root is None:
        raise SystemExit(
            f"No {DIRNAME}/ found from {Path.cwd()} upwards. Run init_competition.py first "
            f"(or set {ENV}, or pass --workdir).")
    index = load_index(root)
    slug = slug or index.get("active")
    if not slug:
        raise SystemExit(f"{root / INDEX} has no active competition. Run: workspace.py use <slug>")
    if slug not in index.get("competitions", {}):
        raise SystemExit(f"Competition '{slug}' is not registered in {root / INDEX}. "
                         f"Known: {sorted(index.get('competitions', {}))}")
    return (root / index["competitions"][slug].get("path", slug)).resolve()
