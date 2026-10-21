#!/usr/bin/env python3
"""Locate and manage the kaggle-war-room workspace (the one folder holding everything the skill writes).

Run `locate` at the start of every session/role to learn WORKDIR.

  <project>/kaggle-war-room-workspace/
  ├── workspace.json     index of competitions + the active one
  └── <slug>/            WORKDIR for that competition

The root is found via env var KAGGLE_WAR_ROOM_WORKSPACE, or by searching upward from the
current directory.

Commands:
  locate            print the absolute WORKDIR of the active competition (exit 1 if none)
  root              print the absolute workspace root
  list              list registered competitions (* = active)
  use <slug>        make <slug> the active competition

Examples:
  python workspace.py locate
  python workspace.py list
  python workspace.py use my-other-comp
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _workspace as ws  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("locate", help="print the active competition's WORKDIR")
    sub.add_parser("root", help="print the workspace root")
    sub.add_parser("list", help="list competitions")
    u = sub.add_parser("use", help="set the active competition")
    u.add_argument("slug")
    args = p.parse_args(argv)

    root = ws.find_root()
    if root is None:
        print(f"No {ws.DIRNAME}/ found from {Path.cwd()} upwards. "
              f"Create one with init_competition.py <slug>.", file=sys.stderr)
        return 1
    index = ws.load_index(root)
    comps = index.get("competitions", {})

    if args.cmd == "root":
        print(root)
    elif args.cmd == "locate":
        try:
            workdir = ws.resolve_workdir()
        except SystemExit as e:
            print(e, file=sys.stderr)
            return 1
        print(workdir)
        settings = workdir / "settings.yaml"
        if settings.exists() and not re.search(r"(?m)^setup_complete:\s*true\b",
                                               settings.read_text(encoding="utf-8"), re.I):
            print("NOTE: first-run setup not finished - run configure.py (detect --write, checklist, done) "
                  "before anything else.", file=sys.stderr)
    elif args.cmd == "list":
        if not comps:
            print("(no competitions)")
        for slug, meta in sorted(comps.items()):
            mark = "*" if slug == index.get("active") else " "
            print(f"{mark} {slug}  ->  {(root / meta.get('path', slug)).resolve()}  (created {meta.get('created')})")
    elif args.cmd == "use":
        if args.slug not in comps:
            print(f"Unknown competition '{args.slug}'. Known: {sorted(comps)}", file=sys.stderr)
            return 1
        index["active"] = args.slug
        ws.save_index(root, index)
        print(f"Active competition: {args.slug} -> {(root / comps[args.slug].get('path', args.slug)).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
