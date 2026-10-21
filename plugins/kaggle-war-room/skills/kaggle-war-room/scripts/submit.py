#!/usr/bin/env python3
"""Closed submission loop: quota -> checked dry run -> submit after approval -> poll score -> CV vs LB.

Commands:
  status                       submissions left today, weekly GPU quota, recent submissions
  submit FILE --exp E012       validate FILE, show a DRY RUN; add --yes (only after the user
                               approved) to really submit. Code competitions: --kernel owner/slug
                               [--kernel-version N] and FILE = the output file name the kernel writes
  poll [--exp E | --ref REF]   wait (a few minutes at most) for the score of the latest pending
                               submission, write it to experiments.csv / submissions.md once, and
                               compare LB with CV. Exit code 2 = still pending: run it again later
  receipts                     list submissions made through this script

Every real submission is recorded in <WORKDIR>/submissions/receipts.jsonl. A submission
spends one of the day's slots, so `submit --yes` must never run without the user's explicit yes.

Examples:
  python submit.py status
  python submit.py submit kaggle-war-room-workspace/x/submissions/E012.csv --exp E012 -m "E012 convnext 5fold"
  python submit.py submit E012.csv --exp E012 -m "E012" --yes
  python submit.py submit submission.csv --kernel me/x-infer --kernel-version 7 --exp E012 -m "E012" --yes
  python submit.py poll --exp E012
"""
import argparse
import datetime as dt
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kaggle as K  # noqa: E402
import _workspace  # noqa: E402

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass


def receipts_path(ws):
    return Path(ws) / "submissions" / "receipts.jsonl"


# ---------------------------------------------------------------- status
def get_status(api, slug):
    out = {"competition": slug}
    try:
        lim = api.competition_get_submission_limits(slug)
        out["submissions"] = {
            "today": K.get(lim, "num_today", "numToday"),
            "total": K.get(lim, "num_total", "numTotal"),
            "allowed_now": K.get(lim, "num_allowed_now", "numAllowedNow"),
            "limited_by_total": K.get(lim, "limited_by_total", "limitedByTotal"),
        }
    except Exception as e:
        out["submissions"] = {"error": str(e)}
    try:
        q = api.quota_view()
        gpu = K.get(q, "gpu_quota", "gpuQuota")
        if gpu is not None:
            used = K.hours(K.get(gpu, "time_used", "timeUsed"))
            total = K.hours(K.get(gpu, "total_time_allowed", "totalTimeAllowed"))
            out["gpu"] = {"used_hours": round(used, 2) if used is not None else None,
                          "total_hours": round(total, 2) if total is not None else None,
                          "remaining_hours": round(max(0.0, total - used), 2) if None not in (used, total) else None,
                          "refresh": K.iso(K.get(q, "quota_refresh_time", "quotaRefreshTime"))}
    except Exception as e:
        out["gpu"] = {"error": str(e)}
    out["recent"] = recent_submissions(api, slug, 10)
    return out


def recent_submissions(api, slug, n=20):
    try:
        subs = api.competition_submissions(slug, page_size=n) or []
    except Exception as e:
        return [{"error": str(e)}]
    rows = []
    for s in subs:
        if s is None:
            continue
        rows.append({
            "ref": K.get(s, "ref"),
            "date": K.iso(K.get(s, "date")),
            "description": K.get(s, "description"),
            "file": K.get(s, "file_name", "fileName"),
            "status": K.enum_name(K.get(s, "status")),
            "public_score": K.get(s, "public_score", "publicScore"),
            "error": K.get(s, "error_description", "errorDescription"),
        })
    return rows


def cmd_status(api, ws, settings, args):
    st = get_status(api, K.slug_of(settings))
    if args.json:
        print(json.dumps(st, indent=1, default=str))
        return 0
    s = st["submissions"]
    if "error" in s:
        print(f"Submissions: unavailable ({s['error']})")
    else:
        limit = (settings.get("submissions") or {}).get("daily_limit")
        print(f"Submissions: {s['today']} used today, {s['allowed_now']} left now"
              + (f" (daily_limit in settings: {limit})" if limit else ""))
    g = st.get("gpu", {})
    if "error" in g:
        print(f"GPU quota: unavailable ({g['error']})")
    elif g:
        print(f"GPU quota: {g['used_hours']}h used / {g['total_hours']}h, {g['remaining_hours']}h left (refresh {g['refresh']})")
    print("Recent submissions:")
    for r in st["recent"][:10]:
        if "error" in r:
            print(f"  (unavailable: {r['error']})")
            break
        print(f"  {r['date']}  {r['status']:<9} public={r['public_score']}  {r['description']}")
    return 0


# ---------------------------------------------------------------- submit
def exp_row(ws, exp_id):
    import log_experiment
    rows = log_experiment.read_rows(Path(ws) / "experiments.csv")
    return next((r for r in rows if r["exp_id"] == exp_id), None)


def cmd_submit(api, ws, settings, args):
    slug = K.slug_of(settings)
    code = bool(args.kernel)
    problems = []
    if exp_row(ws, args.exp) is None:
        raise SystemExit(f"experiment {args.exp} not found in experiments.csv - log it first "
                         "(log_experiment.py), so the score can be linked to its CV.")
    if not code:
        f = Path(args.file)
        if not f.exists():
            raise SystemExit(f"file not found: {f}")
        configured = (settings.get("submissions") or {}).get("sample_submission")
        if args.sample:  # typed on the command line: relative to the current directory
            sample = Path(args.sample)
        elif configured:  # from settings.yaml: relative to WORKDIR (same rule as configure.py)
            sample = _workspace.settings_path(ws, configured)
        else:
            sample = None
        if sample is not None:
            if not sample.is_file():
                raise SystemExit(f"sample_submission not found: {sample} (fix submissions.sample_submission; "
                                 "relative paths are relative to the competition folder)")
            import check_submission
            ok, errors, warnings = check_submission.check(str(f), str(sample))
            if ok is None:
                raise SystemExit(f"could not read the files for validation: {errors[0]}")
            for w in warnings:
                print(f"check WARNING: {w}")
            if not ok:
                for e in errors:
                    print(f"check ERROR: {e}")
                raise SystemExit("submission file failed validation - not submitting")
            print("check_submission: PASS")
        else:
            problems.append("no sample_submission configured (submissions.sample_submission) - file not validated")
    st = get_status(api, slug)["submissions"]
    left = st.get("allowed_now")
    if left is not None and left <= 0:
        raise SystemExit("No submissions left right now - not submitting.")

    row = exp_row(ws, args.exp)
    message = args.message or args.exp
    print("\n== Submission plan ==")
    print(f"competition : {slug}")
    print(f"type        : {'code competition (kernel)' if code else 'file upload'}")
    if code:
        print(f"kernel      : {args.kernel}" + (f" v{args.kernel_version}" if args.kernel_version else " (latest version)"))
        print(f"output file : {args.file}")
    else:
        print(f"file        : {args.file}")
    print(f"message     : {message}")
    if row:
        print(f"experiment  : {args.exp}  CV={row.get('cv') or '?'}  ({row.get('solution')})")
    print(f"slots       : {st.get('today', '?')} used today, {left if left is not None else '?'} left -> {left - 1 if isinstance(left, int) else '?'} after this")
    for p in problems:
        print(f"note        : {p}")

    if not args.yes:
        print("\nDRY RUN - nothing was submitted. Ask the user; re-run with --yes only after they approve.")
        return 0

    started_at = K.now_iso()  # lower bound for matching a submission that comes back without a ref
    if code:
        resp = api.competition_submit_code(file_name=args.file, message=message, competition=slug,
                                           kernel=args.kernel, kernel_version=args.kernel_version, quiet=True)
    else:
        resp = api.competition_submit(args.file, message, slug, quiet=True)
    rec = {
        "started_at": started_at, "submitted_at": K.now_iso(), "exp_id": args.exp, "competition": slug,
        "message": message, "kind": "code" if code else "file", "file": args.file,
        "sha256": None if code else K.sha256(args.file),
        "kernel": args.kernel, "kernel_version": args.kernel_version,
        "ref": K.get(resp, "ref"), "server_message": K.get(resp, "message"),
        "status": "PENDING", "public_score": None, "logged": False,
    }
    K.append_jsonl(receipts_path(ws), rec)
    print(f"\nSubmitted. ref={rec['ref']} {rec['server_message'] or ''}".rstrip())
    print(f"Next: python submit.py poll --exp {args.exp}   (re-run until it is scored)")
    return 0


# ---------------------------------------------------------------- poll
AMBIGUOUS = "AMBIGUOUS"
CLOCK_SKEW = 120  # seconds of tolerance between our clock and Kaggle's


def _when(value):
    try:
        t = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def find_submission(subs, rec, claimed=()):
    """The Kaggle submission behind a receipt. With a ref: exact ref only. Without one: same
    message, made after this receipt's submit started, not claimed by another receipt - and only
    if exactly one fits (otherwise AMBIGUOUS). Never falls back to older submissions."""
    if rec.get("ref") is not None:
        return next((s for s in subs if str(s.get("ref")) == str(rec["ref"])), None)
    started = _when(rec.get("started_at") or rec.get("submitted_at"))
    if started is None:
        return None
    floor = started - dt.timedelta(seconds=CLOCK_SKEW)
    cands = [s for s in subs
             if s.get("description") == rec.get("message")
             and str(s.get("ref")) not in claimed
             and (_when(s.get("date")) or floor - dt.timedelta(days=1)) >= floor]
    if len(cands) > 1:
        return AMBIGUOUS
    return cands[0] if cands else None


def cv_lb_report(ws, settings, exp_id, lb):
    """Compare this LB with CV and with the historical LB-CV offset."""
    import log_experiment
    rows = log_experiment.read_rows(Path(ws) / "experiments.csv")
    me = next((r for r in rows if r["exp_id"] == exp_id), None)
    if me is None or K.to_float(me.get("cv")) is None:
        return [f"LB {lb} (no CV logged for {exp_id})"]
    cv = K.to_float(me["cv"])
    offset = lb - cv
    lines = [f"{exp_id}: CV {cv} -> public LB {lb}  (LB - CV = {offset:+.6g})"]
    hist = [K.to_float(r["lb"]) - K.to_float(r["cv"]) for r in rows
            if r["exp_id"] != exp_id and K.to_float(r.get("lb")) is not None and K.to_float(r.get("cv")) is not None]
    if len(hist) >= 3:
        mean = statistics.mean(hist)
        sd = statistics.pstdev(hist)
        min_gain = float((settings.get("phases") or {}).get("min_gain", 0.001))
        tol = max(2 * sd, 2 * min_gain)
        lines.append(f"historical LB - CV offset: mean {mean:+.6g}, sd {sd:.6g} over {len(hist)} submissions")
        if abs(offset - mean) > tol:
            lines.append("WARNING: CV/LB divergence - this offset is outside the usual range. "
                         "Check for CV leakage, a fold/preprocessing mismatch, or public-LB noise "
                         "before trusting either number (references/anti-shakeup.md).")
        else:
            lines.append("CV and LB agree with past submissions.")
    else:
        lines.append(f"(only {len(hist)} earlier CV/LB pairs - need 3 to judge agreement)")
    return lines


def append_submissions_md(ws, rec, cv):
    path = Path(ws) / "submissions.md"
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip().lower() == "## log")
    except StopIteration:
        return
    end = start + 1
    while end < len(lines) and not lines[end].startswith("## "):
        end += 1
    last_table = max((i for i in range(start, end) if lines[i].startswith("|")), default=None)
    if last_table is None:
        return
    marker = f"| ref {rec.get('ref')} |"
    if rec.get("ref") is not None and any(l.rstrip().endswith(marker) for l in lines[start:end]):
        return  # already written (a retried poll must not add a second row)
    row = (f"| {rec['submitted_at'][:10]} | {rec.get('exp_id') or ''} | {Path(rec.get('file') or '').name or rec.get('kernel')} "
           f"| {cv if cv is not None else ''} | {rec.get('public_score')} | yes | ref {rec.get('ref')} |")
    lines.insert(last_table + 1, row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_poll(api, ws, settings, args):
    slug = K.slug_of(settings)
    receipts = K.read_jsonl(receipts_path(ws))
    matches = [i for i, r in enumerate(receipts)
               if (args.ref is None or str(r.get("ref")) == str(args.ref))
               and (args.exp is None or r.get("exp_id") == args.exp)]
    if not matches:
        raise SystemExit("No matching submission in receipts.jsonl (submit through submit.py first).")
    def unfinished(r):  # not scored yet, or scored but not written to the logs yet
        return r.get("status") == "PENDING" or (r.get("status") == "COMPLETE" and not r.get("logged", True))

    if args.ref is None and args.exp is None:  # default: the latest submission still needing work
        matches = [i for i in matches if unfinished(receipts[i])] or matches
    idx = matches[-1]
    rec = receipts[idx]
    if not unfinished(rec):  # fully handled: never log twice
        print(f"Already scored and logged: {rec['status']} public LB = {rec.get('public_score')} "
              f"({rec.get('exp_id') or 'no exp'}, ref {rec.get('ref')}). Nothing to update.")
        return 0

    if rec.get("status") == "PENDING":
        claimed = {str(r.get("ref")) for i, r in enumerate(receipts) if i != idx and r.get("ref") is not None}
        deadline = time.time() + args.timeout * 60
        while True:
            sub = find_submission(recent_submissions(api, slug, 50), rec, claimed)
            if sub == AMBIGUOUS:
                print("AMBIGUOUS: several new submissions carry this message and Kaggle returned no ref. "
                      "Not guessing - check `submit.py status` and re-run poll with --ref <ref>.")
                return 3
            status = sub["status"] if sub else None
            if status in ("COMPLETE", "ERROR"):
                break
            if time.time() >= deadline:
                print(f"PENDING: still {status or 'not visible'} after {args.timeout:g} min. "
                      "Run `submit.py poll` again in a few minutes.")
                return 2
            time.sleep(args.interval)
        rec.update(status=status, public_score=K.to_float(sub.get("public_score")), error=sub.get("error"),
                   scored_at=K.now_iso(), ref=rec.get("ref") if rec.get("ref") is not None else sub.get("ref"),
                   logged=(status == "ERROR"))
        receipts[idx] = rec
        K.write_jsonl(receipts_path(ws), receipts)  # scored; logging below is a separate, repeatable step

    if rec["status"] == "ERROR":
        print(f"Submission FAILED: {rec.get('error') or 'no error description'}")
        print("Code competition? Download the kernel log and check paths, internet, time and memory limits.")
        return 1
    lb = rec["public_score"]
    print(f"Scored: public LB = {lb}")
    cv = None
    if rec.get("exp_id") and lb is not None:
        import log_experiment
        log_experiment.main(["--workdir", str(ws), "--update", rec["exp_id"], "--lb", repr(lb)])
        row = exp_row(ws, rec["exp_id"])
        cv = row.get("cv") if row else None
        for line in cv_lb_report(ws, settings, rec["exp_id"], lb):
            print(line)
    append_submissions_md(ws, rec, cv)
    rec["logged"] = True
    receipts[idx] = rec
    K.write_jsonl(receipts_path(ws), receipts)
    return 0


def cmd_receipts(ws, args):
    for r in K.read_jsonl(receipts_path(ws)):
        print(f"{r['submitted_at']}  {r.get('exp_id') or '-':<6} {r['status']:<9} LB={r.get('public_score')}  ref={r.get('ref')}  {r.get('message')}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", help="competition folder (default: active competition)")
    p.add_argument("--competition", help="registered competition to use instead of the active one")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status", help="quota and recent submissions")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("submit", help="dry run by default; --yes submits")
    s.add_argument("file", help="submission CSV (or, with --kernel, the output file the kernel writes)")
    s.add_argument("--exp", required=True, help="experiment id from experiments.csv (log it first)")
    s.add_argument("-m", "--message", help="submission description")
    s.add_argument("--sample", help="sample_submission.csv (default: submissions.sample_submission)")
    s.add_argument("--kernel", help="code competition: owner/kernel-slug to submit")
    s.add_argument("--kernel-version", type=int, help="code competition: kernel version (default latest)")
    s.add_argument("--yes", action="store_true", help="really submit (only after explicit user approval)")
    s = sub.add_parser("poll", help="wait for the score, log it, compare with CV")
    s.add_argument("--ref", help="submission ref (default: latest receipt)")
    s.add_argument("--exp", help="pick the latest receipt for this experiment")
    s.add_argument("--timeout", type=float, default=None,
                   help="minutes to wait before returning PENDING (default submissions.poll_timeout_minutes or 8; "
                        "kept short so it fits agent command time limits)")
    s.add_argument("--interval", type=float, default=30, help="seconds between checks")
    sub.add_parser("receipts", help="list recorded submissions")
    args = p.parse_args(argv)

    ws = _workspace.resolve_workdir(args.workdir, args.competition)
    if args.cmd == "receipts":
        return cmd_receipts(ws, args)
    settings = K.load_settings(ws)
    if args.cmd == "poll" and args.timeout is None:
        args.timeout = float((settings.get("submissions") or {}).get("poll_timeout_minutes", 8))
    api = K.connect()
    return {"status": cmd_status, "submit": cmd_submit, "poll": cmd_poll}[args.cmd](api, ws, settings, args)


if __name__ == "__main__":
    sys.exit(main())
