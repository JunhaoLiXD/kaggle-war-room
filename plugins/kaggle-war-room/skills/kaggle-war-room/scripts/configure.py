#!/usr/bin/env python3
"""First-run setup for a competition: what the user must decide, what can be detected, and writing it.

Commands:
  check (default)       list every setting the user should decide or confirm, with its current
                        value and status. Exit 1 while a REQUIRED item is missing.
  detect [--write]      look up facts from the Kaggle API (deadline, metric, daily submission
                        limit, code competition, host, your username) and this machine (GPUs,
                        sample_submission.csv). --write fills only values that are still unset
                        or at their default; it never overwrites your own choices.
  set KEY=VALUE ...     write values into settings.yaml, keeping comments. VALUE is YAML:
                        models.scout=haiku  scout.host_authors="[Jane Doe, Kaggle Staff]"
                        compute.local_gpus="[{name: RTX 4090, vram_gb: 24, hours_per_day: 8}]"
  done                  mark setup complete (refused while a REQUIRED item is missing)

The agent shows the `check` list to the user in ONE message, writes the answers with `set`,
and runs `done` once the user confirmed. Personal choices (models, compute, flavor) are copied
to new competitions automatically by init_competition.py.

Examples:
  python configure.py detect --write
  python configure.py
  python configure.py set models.scout=haiku models.builder=sonnet models.strategist=opus
  python configure.py done
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _workspace  # noqa: E402

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

SKILL_DIR = Path(__file__).resolve().parent.parent
EXAMPLE = SKILL_DIR / "config" / "settings.example.yaml"

# Personal preferences copied from the previous competition by init_competition.py.
PERSONAL_KEYS = [
    "kaggle_username", "flavor",
    "models.scout", "models.builder", "models.strategist",
    "models.critic", "models.critic_model", "models.critic_command",
    "compute.local_gpus", "compute.kaggle_gpu_hours_per_week",
    "compute.confirm_if_gpu_hours_over", "compute.kaggle_accelerator",
]

# What the user decides or confirms on first run. kind: required | decide
# "when" limits an item to some situations.
ITEMS = [
    # group, key, kind, question
    ("Models & review", "models.scout", "decide",
     "Model for the Scout (cheap/fast is enough). Claude Code: haiku|sonnet|opus; Codex: a `codex exec -m` name; inherit = main session"),
    ("Models & review", "models.builder", "decide", "Model for the Builder (main coding model)"),
    ("Models & review", "models.strategist", "decide", "Model for the Strategist (strongest reasoning model)"),
    ("Models & review", "models.critic", "decide",
     "Critic mode: external (the other vendor's CLI, recommended) or subagent (fresh context on this host)"),
    ("Models & review", "models.critic_command", "decide",
     "External Critic command - from Claude Code use `codex exec ...`, from Codex use `claude -p ...`"),
    ("Compute", "compute.local_gpus", "decide",
     "Your own GPUs, e.g. [{name: RTX 4090, vram_gb: 24, hours_per_day: 8}]; [] = none"),
    ("Compute", "compute.kaggle_gpu_hours_per_week", "decide", "Weekly Kaggle GPU hours you can spend (free tier: 30)"),
    ("Compute", "compute.confirm_if_gpu_hours_over", "decide", "Ask before any single run longer than this many GPU hours"),
    ("Competition facts", "competition.deadline", "required", "Final submission deadline (YYYY-MM-DD, UTC)"),
    ("Competition facts", "metric.name", "required", "Evaluation metric (from the Evaluation page)"),
    ("Competition facts", "metric.higher_is_better", "required", "Is a higher metric better? (false for rmse, logloss, ...)"),
    ("Competition facts", "submissions.daily_limit", "required", "Daily submission limit (from the rules)"),
    ("Competition facts", "competition.code_competition", "required", "Must submissions come from a Kaggle notebook?"),
    ("Competition facts", "compute.inference_time_limit_hours", "required_if_code",
     "Inference time limit in hours (code competitions)"),
    ("Competition facts", "submissions.sample_submission", "required", "Path to sample_submission.csv (download the data first)"),
    ("You in this competition", "competition.team_name", "decide", "Your team name on the leaderboard (null = not tracked)"),
    ("You in this competition", "scout.host_authors", "decide",
     "Display names of the host's staff / Kaggle staff, so their posts are flagged [HOST]"),
    ("You in this competition", "competition.task_type", "decide", "classification | segmentation | detection | other"),
    ("You in this competition", "kaggle_username", "decide", "Your Kaggle username (owner of kernels/datasets)"),
    ("Style", "flavor", "decide", "Small war-room touches at key moments (true/false)"),
]

LOWER_IS_BETTER = re.compile(r"(?i)rmse|mse|mae|log\s*loss|logloss|cross.?entropy|error|loss|wrmse|smape|mape|distance")


# ---------------------------------------------------------------- YAML editing (keeps comments)
def _yaml():
    import yaml
    return yaml


def dump_value(value):
    yaml = _yaml()
    s = yaml.safe_dump(value, default_flow_style=True, allow_unicode=True, sort_keys=False, width=10 ** 6).strip()
    if s.endswith("\n..."):
        s = s[:-4].strip()
    elif s.endswith("..."):
        s = s[:-3].strip()
    return s


def get_key(data, dotted):
    cur = data
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _split_comment(rest):
    """Split ' value   # comment' into (value, comment_with_leading_spaces) ignoring # inside quotes."""
    quote = None
    for i, ch in enumerate(rest):
        if ch in "\"'" and (quote is None or quote == ch):
            quote = None if quote else ch
        elif ch == "#" and quote is None and (i == 0 or rest[i - 1] in " \t"):
            j = i
            while j > 0 and rest[j - 1] in " \t":
                j -= 1
            return rest[:j], rest[j:]
    return rest, ""


def set_key(text, dotted, value):
    """Set a scalar/list/flow value for `dotted` in block-style YAML text, keeping comments."""
    parts = dotted.split(".")
    lines = text.split("\n")
    stack = []
    for i, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = re.match(r"^(\s*)([A-Za-z_][\w-]*):(.*)$", line)
        if not m:
            continue
        indent, key, rest = len(m.group(1)), m.group(2), m.group(3)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = [k for _, k in stack] + [key]
        if path == parts:
            val, comment = _split_comment(rest)
            stripped = val.strip()
            if stripped == "":
                raise ValueError(f"{dotted} is a section, not a single value")
            end = i + 1
            if stripped in (">", ">-", "|", "|-", ">+", "|+"):  # drop the old block scalar body
                def deeper(j):
                    return lines[j].strip() and len(lines[j]) - len(lines[j].lstrip()) > indent
                while end < len(lines) and (deeper(end) or (not lines[end].strip() and any(
                        deeper(k) for k in range(end + 1, min(end + 3, len(lines)))))):
                    end += 1
            new = f"{m.group(1)}{key}: {dump_value(value)}"
            if comment:
                comment = comment.lstrip()
                col = len(line) - len(comment)  # column of the '#'
                new = new.ljust(col) + comment if len(new) < col else new + " " + comment
            lines[i:end] = [new]
            out = "\n".join(lines)
            if get_key(_yaml().safe_load(out), dotted) != value:
                raise ValueError(f"could not write {dotted} safely; edit settings.yaml by hand")
            return out
        stack.append((indent, key))
    raise KeyError(f"{dotted} not found in settings.yaml (only keys that exist can be set)")


def parse_value(raw):
    if raw.strip() == "":
        return None
    return _yaml().safe_load(raw)


# ---------------------------------------------------------------- status
def example_defaults():
    return _yaml().safe_load(EXAMPLE.read_text(encoding="utf-8")) or {}


def is_unset(v):
    return v is None or (isinstance(v, str) and v.startswith("<"))


def status_of(settings, defaults, key, kind, ws):
    v = get_key(settings, key)
    code = get_key(settings, "competition.code_competition")
    if kind == "required_if_code":
        if code is False:
            return "n/a", v
        if code is None and is_unset(v):
            return "needs code_competition", v
        kind = "required"
    if key == "models.critic_command" and get_key(settings, "models.critic") == "subagent":
        return "n/a", v
    if key == "submissions.sample_submission" and not is_unset(v):
        p = Path(str(v))
        if not p.is_absolute():
            p = Path(ws) / p
        return ("OK" if p.exists() else "MISSING (file not found)"), v
    if is_unset(v):
        return ("MISSING" if kind == "required" else "unset"), v
    if v == get_key(defaults, key):
        return "default - confirm", v
    return "set", v


def check(ws, settings, quiet=False):
    defaults = example_defaults()
    missing, lines, group = [], [], None
    for g, key, kind, question in ITEMS:
        st, v = status_of(settings, defaults, key, kind, ws)
        if st.startswith("MISSING"):
            missing.append(key)
        if g != group:
            lines.append(f"\n[{g}]")
            group = g
        shown = dump_value(v)
        if len(shown) > 28:
            shown = shown[:12] + "..." + shown[-13:]
        lines.append(f"  {key:<38} {shown:<28} {st:<22} {question}")
    if not quiet:
        print(f"Setup checklist for {ws.name}  (setup_complete: {bool(settings.get('setup_complete'))})")
        print("\n".join(lines))
        if missing:
            print(f"\nREQUIRED, still missing: {', '.join(missing)}")
        print("\nAsk the user about every line above in ONE message (show current values, recommend), "
              "write answers with `configure.py set KEY=VALUE ...`, then `configure.py done`.")
    return missing


# ---------------------------------------------------------------- detection
def detect_local_gpus():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        if "," in line:
            name, mem = [s.strip() for s in line.split(",", 1)]
            try:
                gpus.append({"name": name, "vram_gb": round(float(mem) / 1024), "hours_per_day": None})
            except ValueError:
                pass
    return gpus


def find_sample_submissions(start, max_depth=4):
    found = []
    start = Path(start)
    skip = {".git", "node_modules", "snapshots", "kernels", "research", "__pycache__", ".venv", "venv"}
    for root, dirs, files in os.walk(start):
        depth = len(Path(root).relative_to(start).parts)
        dirs[:] = [d for d in dirs if d not in skip and depth < max_depth]
        for f in files:
            if re.match(r"(?i)sample_?submission.*\.csv$", f):
                found.append(Path(root) / f)
    return found


def detect(ws, settings):
    """Return {key: (value, note)} proposals. Kaggle lookups are skipped if the API is unavailable."""
    import _kaggle as K
    props, notes = {}, []
    slug = (settings.get("competition") or {}).get("slug")
    try:
        api = K.connect()
    except SystemExit as e:
        api = None
        notes.append(f"Kaggle API unavailable ({e}); competition facts not detected")
    except Exception as e:
        api = None
        notes.append(f"Kaggle API unavailable ({e}); competition facts not detected")
    if api is not None:
        try:
            resp = api.competitions_list(search=slug, page_size=20)
            comps = K.get(resp, "competitions") if not isinstance(resp, list) else resp
            comp = next((c for c in comps or [] if str(K.get(c, "ref") or "").rstrip("/").split("/")[-1] == slug), None)
        except Exception as e:
            comp = None
            notes.append(f"competition lookup failed: {e}")
        if comp is not None:
            d = K.get(comp, "deadline")
            if d is not None:
                props["competition.deadline"] = (str(K.iso(d))[:10], "from Kaggle")
            metric = K.get(comp, "evaluation_metric", "evaluationMetric")
            if metric:
                props["metric.name"] = (metric, "from Kaggle")
                props["metric.higher_is_better"] = (not bool(LOWER_IS_BETTER.search(metric)),
                                                    "guessed from the metric name - confirm")
            mds = K.get(comp, "max_daily_submissions", "maxDailySubmissions")
            if mds:
                props["submissions.daily_limit"] = (int(mds), "from Kaggle")
            ko = K.get(comp, "is_kernels_submissions_only", "isKernelsSubmissionsOnly")
            if ko is not None:
                props["competition.code_competition"] = (bool(ko), "from Kaggle")
            host = K.get(comp, "host_name", "hostName") or K.get(comp, "organization_name", "organizationName")
            if host:
                notes.append(f"Host: {host} - ask the user which staff names to put in scout.host_authors "
                             "(pinned posts are flagged anyway)")
        elif slug:
            notes.append(f"competition '{slug}' not found via the API (check the slug, or accept the rules first)")
        try:
            user = api.get_config_value(api.CONFIG_NAME_USER)
            if user:
                props["kaggle_username"] = (user, "from your Kaggle credentials")
        except Exception:
            pass
    gpus = detect_local_gpus()
    if gpus:
        props["compute.local_gpus"] = (gpus, "from nvidia-smi - ask the user for hours_per_day")
    else:
        notes.append("No local NVIDIA GPU detected (nvidia-smi not found) - compute.local_gpus stays [] unless the user says otherwise")
    project = Path(ws).parent.parent  # <project>/kaggle-war-room-workspace/<slug>
    samples = find_sample_submissions(project)
    if len(samples) == 1:
        props["submissions.sample_submission"] = (samples[0].resolve().as_posix(), "found in the project")
    elif samples:
        notes.append("Several sample_submission files found - ask which one: " + ", ".join(p.as_posix() for p in samples[:5]))
    else:
        notes.append("No sample_submission*.csv found in the project - download the competition data first")
    return props, notes


# ---------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", help="competition folder (default: active competition)")
    p.add_argument("--competition", help="registered competition to use instead of the active one")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("check", help="list what to decide (default)")
    d = sub.add_parser("detect", help="detect facts from Kaggle and this machine")
    d.add_argument("--write", action="store_true", help="fill values that are still unset or default")
    s = sub.add_parser("set", help="write KEY=VALUE pairs")
    s.add_argument("pairs", nargs="+", metavar="KEY=VALUE")
    sub.add_parser("done", help="mark setup complete")
    args = p.parse_args(argv)

    ws = _workspace.resolve_workdir(args.workdir, args.competition)
    path = ws / "settings.yaml"
    if not path.exists():
        raise SystemExit(f"{path} not found - run init_competition.py first")
    text = path.read_text(encoding="utf-8")
    settings = _yaml().safe_load(text) or {}
    cmd = args.cmd or "check"

    if cmd == "check":
        return 1 if check(ws, settings) else 0

    if cmd == "set":
        for pair in args.pairs:
            if "=" not in pair:
                raise SystemExit(f"expected KEY=VALUE, got {pair!r}")
            key, raw = pair.split("=", 1)
            try:
                text = set_key(text, key.strip(), parse_value(raw))
            except (KeyError, ValueError) as e:
                raise SystemExit(str(e).strip('"'))
            print(f"set {key.strip()} = {raw}")
        path.write_text(text, encoding="utf-8")
        return 0

    if cmd == "detect":
        props, notes = detect(ws, settings)
        defaults = example_defaults()
        written = []
        for key, (value, note) in props.items():
            cur = get_key(settings, key)
            free = is_unset(cur) or cur == get_key(defaults, key)
            action = "keep yours" if not free else ("write" if args.write else "would write")
            print(f"  {key:<38} {dump_value(value):<30} {note}  [{action}]")
            if args.write and free:
                text = set_key(text, key, value)
                written.append(key)
        for n in notes:
            print(f"  note: {n}")
        if written:
            path.write_text(text, encoding="utf-8")
            print(f"Wrote {len(written)} value(s). Confirm them with the user in the setup checklist.")
        return 0

    if cmd == "done":
        missing = check(ws, settings, quiet=True)
        if missing:
            raise SystemExit(f"Setup not complete - still missing: {', '.join(missing)}")
        path.write_text(set_key(text, "setup_complete", True), encoding="utf-8")
        print(f"Setup complete for {ws.name}.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
