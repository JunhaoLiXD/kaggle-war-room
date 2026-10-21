#!/usr/bin/env python3
"""Report the current phase and whether progress has stalled.

Reads <workdir>/settings.yaml and <workdir>/experiments.csv.

Phase (from competition.deadline and phases.*.until_days_left):
  days_left > explore.until_days_left  -> explore
  days_left > refine.until_days_left   -> refine
  otherwise                            -> finalize
The last reported phase is stored in <workdir>/.phase; a change is reported as
PHASE CHANGED (the user must confirm a phase switch).

Stagnation: walk experiments with a CV value in file order, tracking the best CV so far.
An experiment "counts as progress" if it beats the best by at least phases.min_gain
(respecting metric.higher_is_better). The streak is the number of most recent experiments
without progress. Status is STAGNANT when streak >= the current phase's stagnation_limit.
Use --solution to compute the streak within one direction only.

Also reports GPU hours logged in the last 7 days vs compute.kaggle_gpu_hours_per_week.

Examples:
  python stagnation_check.py 
  python stagnation_check.py --solution convnext --json
  python stagnation_check.py --today 2026-11-20 --no-save
"""
import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _workspace  # noqa: E402

DEFAULT_PHASES = {
    "explore": {"until_days_left": 30, "stagnation_limit": 3},
    "refine": {"until_days_left": 10, "stagnation_limit": 6},
    "finalize": {"until_days_left": 0},
    "min_gain": 0.001,
}


def load_settings(ws):
    import yaml
    path = ws / "settings.yaml"
    if not path.exists():
        raise SystemExit(f"settings.yaml not found in {ws}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def determine_phase(days_left, phases):
    if days_left is None:
        return "explore", "deadline not set - defaulting to explore; set competition.deadline"
    if days_left > phases["explore"]["until_days_left"]:
        return "explore", None
    if days_left > phases["refine"]["until_days_left"]:
        return "refine", None
    return "finalize", None


def stagnation(rows, higher, min_gain):
    best, best_id, streak, history = None, None, 0, []
    for r in rows:
        cv = fnum(r.get("cv"))
        if cv is None:
            continue
        if best is None:
            best, best_id, streak = cv, r["exp_id"], 0
            history.append((r["exp_id"], cv, "first"))
            continue
        gain = (cv - best) if higher else (best - cv)
        if gain >= min_gain:
            best, best_id, streak = cv, r["exp_id"], 0
            history.append((r["exp_id"], cv, f"+{gain:.6g} progress"))
        else:
            streak += 1
            if (cv > best) == higher and cv != best:
                best, best_id = cv, r["exp_id"]  # tiny improvement: new best, but still no real progress
            history.append((r["exp_id"], cv, f"{gain:+.6g} below min_gain"))
    return best, best_id, streak, history


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", help="competition folder (default: active competition from workspace.json)")
    p.add_argument("--competition", help="use this registered competition instead of the active one")
    p.add_argument("--solution", help="only consider experiments whose solution contains this text")
    p.add_argument("--today", help="override today's date, YYYY-MM-DD (testing)")
    p.add_argument("--no-save", action="store_true", help="do not update the stored phase")
    p.add_argument("--json", action="store_true", help="print JSON")
    args = p.parse_args(argv)

    ws = _workspace.resolve_workdir(args.workdir, args.competition)
    s = load_settings(ws)
    phases = {**DEFAULT_PHASES, **(s.get("phases") or {})}
    for k in ("explore", "refine", "finalize"):
        phases[k] = {**DEFAULT_PHASES[k], **(phases.get(k) or {})}
    min_gain = float(phases.get("min_gain", 0.001))
    higher = (s.get("metric") or {}).get("higher_is_better", True)
    compute = s.get("compute") or {}

    today = dt.date.fromisoformat(args.today) if args.today else dt.date.today()
    deadline = (s.get("competition") or {}).get("deadline")
    days_left = None
    if deadline:
        days_left = (dt.date.fromisoformat(str(deadline)) - today).days
    phase, note = determine_phase(days_left, phases)
    limit = phases[phase].get("stagnation_limit")

    exp_path = ws / "experiments.csv"
    rows = []
    if exp_path.exists():
        with open(exp_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    all_rows = rows
    if args.solution:
        rows = [r for r in rows if args.solution.lower() in (r.get("solution") or "").lower()]
    best, best_id, streak, history = stagnation(rows, higher, min_gain)

    status = "OK"
    if limit is not None and streak >= int(limit):
        status = "STAGNANT"
    elif limit is None:
        status = "N/A (no stagnation_limit in this phase)"

    week_ago = today - dt.timedelta(days=7)
    gpu_week = 0.0
    for r in all_rows:
        try:
            d = dt.date.fromisoformat(r.get("date", ""))
        except ValueError:
            continue
        if week_ago < d <= today:
            gpu_week += fnum(r.get("gpu_hours")) or 0.0
    quota = compute.get("kaggle_gpu_hours_per_week")

    phase_file = ws / ".phase"
    prev_phase = phase_file.read_text(encoding="utf-8").strip() if phase_file.exists() else None
    phase_changed = prev_phase is not None and prev_phase != phase
    if not args.no_save:
        phase_file.write_text(phase, encoding="utf-8")

    res = {
        "today": today.isoformat(), "deadline": str(deadline) if deadline else None, "days_left": days_left,
        "phase": phase, "previous_phase": prev_phase, "phase_changed": phase_changed,
        "focus": phases[phase].get("focus"), "note": note,
        "filter": args.solution, "experiments_with_cv": len([h for h in history]),
        "best_cv": best, "best_exp": best_id, "min_gain": min_gain, "higher_is_better": higher,
        "streak": streak, "stagnation_limit": limit, "status": status,
        "gpu_hours_last_7_days": round(gpu_week, 3), "kaggle_gpu_hours_per_week": quota,
        "recent": [{"exp_id": e, "cv": c, "verdict": v} for e, c, v in history[-8:]],
    }
    if args.json:
        print(json.dumps(res, indent=1))
        return 0

    tag = f"[D-{days_left}] " if days_left is not None else ""
    print(f"{tag}Phase: {phase}" + (f" ({res['focus']})" if res["focus"] else "")
          + (f" - {days_left} days left (deadline {deadline})" if days_left is not None else ""))
    if note:
        print(f"Note: {note}")
    if phase_changed:
        print(f"PHASE CHANGED: {prev_phase} -> {phase}. Ask the user to confirm the phase switch.")
    scope = f" (solution contains '{args.solution}')" if args.solution else ""
    print(f"Experiments with CV{scope}: {len(history)}; best CV {best} ({best_id})")
    print(f"Streak without >= {min_gain} gain: {streak} / limit {limit}")
    print(f"Status: {status}")
    if status == "STAGNANT":
        print("-> plan.md must include a 'Change of direction?' section.")
    if quota is not None:
        print(f"GPU hours logged in last 7 days: {gpu_week:.2f} / {quota} weekly quota")
    if history:
        print("Recent:")
        for e, c, v in history[-8:]:
            print(f"  {e}: cv={c} ({v})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
