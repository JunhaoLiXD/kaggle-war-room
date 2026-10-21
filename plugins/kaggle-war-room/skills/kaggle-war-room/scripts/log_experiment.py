#!/usr/bin/env python3
"""Append one experiment to <workdir>/experiments.csv, or update fields of an existing one.

Columns: exp_id,date,solution,changed_vars,cv,cv_std,lb,gpu_hours,infer_minutes,notes

exp_id is auto-assigned (E001, E002, ...) unless --exp-id is given. Use --update to fill in
fields later (typically the public LB after the user approved and made a submission).

Examples:
  python log_experiment.py --solution "effnet_b3 baseline" \
      --changed "img_size=512" --cv 0.8123 --cv-std 0.004 --gpu-hours 2.5 --infer-minutes 18 \
      --notes "replicated public notebook X v12"
  python log_experiment.py --update E004 --lb 0.815
  python log_experiment.py --show 5
"""
import argparse
import csv
import datetime as dt
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _workspace  # noqa: E402

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

COLUMNS = ["exp_id", "date", "solution", "changed_vars", "cv", "cv_std", "lb", "gpu_hours", "infer_minutes", "notes"]
NUMERIC = {"cv", "cv_std", "lb", "gpu_hours", "infer_minutes"}


def read_rows(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise SystemExit(f"{path} has unexpected columns {reader.fieldnames}; expected {COLUMNS}")
        return list(reader)


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def next_id(rows):
    nums = [int(m.group(1)) for r in rows if (m := re.fullmatch(r"E(\d+)", r.get("exp_id", "")))]
    return f"E{(max(nums) + 1 if nums else 1):03d}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", help="competition folder (default: active competition from workspace.json)")
    p.add_argument("--competition", help="use this registered competition instead of the active one")
    p.add_argument("--exp-id", help="experiment id (default: next E###)")
    p.add_argument("--update", metavar="EXP_ID", help="update fields of an existing experiment instead of appending")
    p.add_argument("--show", type=int, metavar="N", help="print the last N rows and exit")
    p.add_argument("--date", help="YYYY-MM-DD (default: today)")
    p.add_argument("--solution", help="short name of the solution / direction")
    p.add_argument("--changed", dest="changed_vars", help="what changed vs. the parent experiment, e.g. 'lr=3e-4; img=640'")
    p.add_argument("--cv", type=float)
    p.add_argument("--cv-std", type=float)
    p.add_argument("--lb", type=float, help="public LB score (often added later with --update)")
    p.add_argument("--gpu-hours", type=float, help="training GPU hours")
    p.add_argument("--infer-minutes", type=float, help="inference time incl. TTA, minutes")
    p.add_argument("--notes")
    args = p.parse_args(argv)

    path = _workspace.resolve_workdir(args.workdir, args.competition) / "experiments.csv"
    rows = read_rows(path)

    if args.show:
        for r in rows[-args.show:]:
            print(", ".join(f"{k}={r[k]}" for k in COLUMNS if r[k]))
        return 0

    fields = {k: getattr(args, k) for k in COLUMNS[1:] if getattr(args, k, None) is not None}
    for k in NUMERIC & fields.keys():
        fields[k] = f"{fields[k]:g}"
    if args.date:
        try:
            dt.date.fromisoformat(args.date)
        except ValueError:
            p.error("--date must be YYYY-MM-DD")

    if args.update:
        match = [r for r in rows if r["exp_id"] == args.update]
        if not match:
            raise SystemExit(f"experiment {args.update} not found in {path}")
        if not fields:
            p.error("--update needs at least one field to change")
        match[0].update(fields)
        write_rows(path, rows)
        print(f"Updated {args.update}: " + ", ".join(f"{k}={v}" for k, v in fields.items()))
        return 0

    if not args.solution:
        p.error("--solution is required when appending")
    exp_id = args.exp_id or next_id(rows)
    if any(r["exp_id"] == exp_id for r in rows):
        raise SystemExit(f"exp_id {exp_id} already exists; use --update")
    row = {k: "" for k in COLUMNS}
    row.update(fields)
    row["exp_id"] = exp_id
    row["date"] = args.date or dt.date.today().isoformat()
    rows.append(row)
    write_rows(path, rows)
    print(f"Logged {exp_id} to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
