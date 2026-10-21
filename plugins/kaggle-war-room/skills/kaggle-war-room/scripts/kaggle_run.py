#!/usr/bin/env python3
"""Run code on Kaggle GPUs and move artifacts as private datasets (official Kaggle API).

Typical flow (see references/kaggle-kernels.md):
  1. init train --code code/train.py --gpu          -> kernels/train/ with kernel-metadata.json
  2. push train --hours 3                            -> DRY RUN (quota, sources, cost)
     push train --hours 3 --yes                      -> really push and start the run
  3. wait train --download                           -> check for a few minutes; exit 2 = still
                                                        running, run it again later; when done,
                                                        pull outputs + log
  4. upload-dataset kernels/train/output/v1 --slug x-weights --title "x weights" [--yes]
  5. init infer --code code/infer.py --gpu --dataset me/x-weights   (internet off by default)
  6. push/wait infer, then submit.py submit submission.csv --kernel me/<slug> --yes

Everything is private by default. Pushing a kernel spends weekly GPU hours and uploading creates
a resource on the user's account, so `push --yes` and `upload-dataset --yes` must only run after
the user approved. Runs are logged to <WORKDIR>/kernels/runs.jsonl.

Commands: init, push, status, wait, output, upload-dataset, list
"""
import argparse
import json
import re
import shutil
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

META = "kernel-metadata.json"
DONE = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"}


def kdir(ws, name):
    return Path(ws) / "kernels" / name


def load_meta(ws, name):
    p = kdir(ws, name) / META
    if not p.exists():
        raise SystemExit(f"{p} not found - run: kaggle_run.py init {name} --code <file>")
    return json.loads(p.read_text(encoding="utf-8"))


def slugify(s):
    return re.sub(r"[^a-z0-9-]+", "-", s.lower()).strip("-")[:50]


def gpu_left(api):
    try:
        q = api.quota_view()
        g = K.get(q, "gpu_quota", "gpuQuota")
        used, total = K.hours(K.get(g, "time_used", "timeUsed")), K.hours(K.get(g, "total_time_allowed", "totalTimeAllowed"))
        return round(total - used, 2) if None not in (used, total) else None
    except Exception:
        return None


# ---------------------------------------------------------------- commands
def cmd_init(api, ws, settings, args):
    folder = kdir(ws, args.name)
    if (folder / META).exists() and not args.force:
        raise SystemExit(f"{folder / META} exists (use --force to overwrite)")
    code = Path(args.code)
    if not code.exists():
        raise SystemExit(f"code file not found: {code}")
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(code, folder / code.name)
    slug = K.slug_of(settings)
    owner = args.owner or K.kaggle_username(api, settings)
    meta = {
        "id": f"{owner}/{slugify(args.slug or f'{slug}-{args.name}')}",
        "title": args.title or f"{slug} {args.name}",
        "code_file": code.name,
        "language": "python",
        "kernel_type": "notebook" if code.suffix == ".ipynb" else "script",
        "is_private": True,
        "enable_gpu": bool(args.gpu),
        "enable_tpu": False,
        "enable_internet": bool(args.internet),
        "dataset_sources": args.dataset or [],
        "competition_sources": [] if args.no_competition_data else [slug],
        "kernel_sources": args.kernel_source or [],
        "model_sources": args.model_source or [],
    }
    (folder / META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Created {folder / META}  (id {meta['id']}, private, gpu={meta['enable_gpu']}, internet={meta['enable_internet']})")
    if args.internet:
        print("Note: code competitions usually require internet OFF for the scoring run (check rules.md).")
    return 0


def cmd_push(api, ws, settings, args):
    folder = kdir(ws, args.name)
    meta = load_meta(ws, args.name)
    if not meta.get("is_private", True) and not args.allow_public:
        raise SystemExit("Kernel is public in kernel-metadata.json. Keep it private, or pass --allow-public "
                         "after checking the competition's sharing rules.")
    compute = settings.get("compute") or {}
    acc = args.accelerator or (compute.get("kaggle_accelerator") if meta.get("enable_gpu") else None)
    threshold = compute.get("confirm_if_gpu_hours_over")
    left = gpu_left(api)
    print("== Kaggle run plan ==")
    print(f"kernel      : {meta['id']}  ({meta['kernel_type']}, {meta['code_file']})")
    print(f"accelerator : {acc or 'CPU'}   internet: {meta.get('enable_internet')}   private: {meta.get('is_private', True)}")
    print(f"sources     : competitions={meta.get('competition_sources')} datasets={meta.get('dataset_sources')} "
          f"kernels={meta.get('kernel_sources')} models={meta.get('model_sources')}")
    print(f"estimate    : {args.hours if args.hours is not None else '?'} GPU h   weekly GPU left: {left if left is not None else '?'} h")
    if args.hours is None and acc:
        print("note        : no --hours estimate given; the plan must estimate GPU time")
    if acc and args.hours is not None and left is not None and args.hours > left:
        print("WARNING     : estimate exceeds the remaining weekly GPU quota")
    if acc and threshold is not None and args.hours is not None and args.hours > float(threshold):
        print(f"CONFIRM     : estimate > confirm_if_gpu_hours_over ({threshold} h) - the user must approve this run")
    if not args.yes:
        print("\nDRY RUN - nothing was pushed. Ask the user; re-run with --yes only after they approve.")
        return 0
    resp = api.kernels_push(str(folder), timeout=str(int(args.timeout)) if args.timeout else None, acc=acc)
    err = K.get(resp, "error")
    invalid = {k: K.get(resp, k) for k in ("invalid_dataset_sources", "invalid_competition_sources",
                                            "invalid_kernel_sources", "invalid_model_sources") if K.get(resp, k)}
    rec = {"pushed_at": K.now_iso(), "name": args.name, "kernel": meta["id"], "ref": K.get(resp, "ref"),
           "url": K.get(resp, "url"), "version": K.get(resp, "version_number", "versionNumber"),
           "accelerator": acc, "estimate_hours": args.hours, "error": err, "invalid_sources": invalid or None,
           "status": "ERROR" if err else "QUEUED"}
    K.append_jsonl(Path(ws) / "kernels" / "runs.jsonl", rec)
    if err or invalid:
        print(f"Push FAILED: {err or ''} {json.dumps(invalid) if invalid else ''}".strip())
        return 1
    print(f"Pushed {meta['id']} v{rec['version']}  {rec['url'] or ''}".rstrip())
    print(f"Next: python kaggle_run.py wait {args.name} --download")
    return 0


def kernel_status(api, ref):
    st = api.kernels_status(ref)
    return K.enum_name(K.get(st, "status")), K.get(st, "failure_message", "failureMessage")


def update_last_run(ws, name, **fields):
    path = Path(ws) / "kernels" / "runs.jsonl"
    runs = K.read_jsonl(path)
    for r in reversed(runs):
        if r.get("name") == name:
            r.update(fields)
            break
    if runs:
        K.write_jsonl(path, runs)


def cmd_status(api, ws, settings, args):
    ref = load_meta(ws, args.name)["id"]
    status, msg = kernel_status(api, ref)
    print(f"{ref}: {status}" + (f" - {msg}" if msg else ""))
    return 0


def download(api, ws, name, ref):
    runs = [r for r in K.read_jsonl(Path(ws) / "kernels" / "runs.jsonl") if r.get("name") == name]
    version = runs[-1].get("version") if runs else None
    dest = kdir(ws, name) / "output" / (f"v{version}" if version else "latest")
    dest.mkdir(parents=True, exist_ok=True)
    res = api.kernels_output(ref, str(dest), force=True, quiet=True)
    files = res[0] if isinstance(res, (list, tuple)) and res and isinstance(res[0], list) else res
    logs = sorted(p for p in dest.rglob("*.log"))
    print(f"Downloaded outputs to {dest}" + (f" ({len(files)} files)" if isinstance(files, list) else ""))
    if logs:
        print(f"Log: {logs[0]}  (read the tail, not the whole file)")
    return dest


def cmd_wait(api, ws, settings, args):
    ref = load_meta(ws, args.name)["id"]
    deadline = time.time() + args.timeout * 60
    status, msg = kernel_status(api, ref)
    while status not in DONE:
        if time.time() >= deadline:
            print(f"RUNNING: {ref} still {status} after {args.timeout:g} min. "
                  f"Run `kaggle_run.py wait {args.name} --download` again later.")
            return 2
        time.sleep(args.interval)
        status, msg = kernel_status(api, ref)
    update_last_run(ws, args.name, status=status, finished_at=K.now_iso(), failure_message=msg)
    print(f"{ref}: {status}" + (f" - {msg}" if msg else ""))
    if args.download:
        download(api, ws, args.name, ref)
    if status != "COMPLETE":
        print("Run did not complete: download the log (output --name) and check paths, internet, memory and time limits.")
        return 1
    return 0


def cmd_output(api, ws, settings, args):
    download(api, ws, args.name, load_meta(ws, args.name)["id"])
    return 0


def cmd_upload(api, ws, settings, args):
    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"not a folder: {folder}")
    owner = args.owner or K.kaggle_username(api, settings)
    ref = f"{owner}/{slugify(args.slug)}"
    meta_path = folder / "dataset-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.update({"id": ref, "title": args.title or meta.get("title") or args.slug})
    meta.setdefault("licenses", [{"name": "CC0-1.0"}])
    files = [p for p in folder.rglob("*") if p.is_file() and p.name != "dataset-metadata.json"]
    size_mb = sum(p.stat().st_size for p in files) / 1e6
    try:
        api.dataset_status(ref)
        exists = True
    except Exception:
        exists = False
    print("== Dataset upload plan ==")
    print(f"dataset     : {ref}  ({'new version' if exists else 'new dataset'})")
    print(f"visibility  : {'PUBLIC' if args.public else 'private'}")
    print(f"content     : {len(files)} files, {size_mb:.1f} MB from {folder}")
    if args.public:
        print("WARNING     : public datasets may break competition sharing rules (e.g. models trained on "
              "competition data). Only with the user's explicit OK after checking rules.md.")
    if not files:
        raise SystemExit("folder has no files to upload")
    if not args.yes:
        print("\nDRY RUN - nothing was uploaded. Ask the user; re-run with --yes only after they approve.")
        return 0
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if exists:
        resp = api.dataset_create_version(str(folder), args.version_notes or f"update {K.now_iso()}",
                                          quiet=True, dir_mode="zip")
    else:
        resp = api.dataset_create_new(str(folder), public=bool(args.public), quiet=True, dir_mode="zip")
    err = K.get(resp, "error")
    rec = {"uploaded_at": K.now_iso(), "dataset": ref, "folder": str(folder), "new": not exists,
           "public": bool(args.public), "url": K.get(resp, "url"), "error": err}
    K.append_jsonl(Path(ws) / "kernels" / "datasets.jsonl", rec)
    if err:
        print(f"Upload FAILED: {err}")
        return 1
    print(f"Uploaded {ref}  {rec['url'] or ''}".rstrip())
    print(f"Use it in a kernel: kaggle_run.py init <name> --code <file> --dataset {ref}")
    return 0


def cmd_list(ws, args):
    for r in K.read_jsonl(Path(ws) / "kernels" / "runs.jsonl"):
        print(f"{r['pushed_at']}  {r['name']:<10} {r.get('status') or '?':<10} v{r.get('version')}  {r['kernel']}  est {r.get('estimate_hours')}h")
    for r in K.read_jsonl(Path(ws) / "kernels" / "datasets.jsonl"):
        print(f"{r['uploaded_at']}  dataset    {'ERROR' if r.get('error') else 'OK':<10} {r['dataset']}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", help="competition folder (default: active competition)")
    p.add_argument("--competition", help="registered competition to use instead of the active one")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init", help="create kernels/<name>/ with kernel-metadata.json")
    s.add_argument("name")
    s.add_argument("--code", required=True, help="script (.py) or notebook (.ipynb) to run")
    s.add_argument("--title")
    s.add_argument("--slug", help="kernel slug (default <competition>-<name>)")
    s.add_argument("--owner", help="Kaggle username (default: kaggle_username or credentials)")
    s.add_argument("--gpu", action="store_true")
    s.add_argument("--internet", action="store_true", help="enable internet (default off)")
    s.add_argument("--dataset", nargs="*", help="dataset sources owner/slug")
    s.add_argument("--kernel-source", nargs="*", help="kernel sources owner/slug")
    s.add_argument("--model-source", nargs="*", help="model sources")
    s.add_argument("--no-competition-data", action="store_true")
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("push", help="dry run by default; --yes pushes and starts the run")
    s.add_argument("name")
    s.add_argument("--hours", type=float, help="estimated GPU hours for this run")
    s.add_argument("--accelerator", help="e.g. NvidiaTeslaT4, NvidiaTeslaP100 (default compute.kaggle_accelerator)")
    s.add_argument("--timeout", type=float, help="max run time in seconds")
    s.add_argument("--allow-public", action="store_true")
    s.add_argument("--yes", action="store_true", help="really push (only after explicit user approval)")
    s = sub.add_parser("status")
    s.add_argument("name")
    s = sub.add_parser("wait", help="poll until the run finishes")
    s.add_argument("name")
    s.add_argument("--interval", type=float, default=60, help="seconds between checks")
    s.add_argument("--timeout", type=float, default=8,
                   help="minutes to wait before returning (default 8, fits agent command time limits; "
                        "re-run while the kernel is still running)")
    s.add_argument("--download", action="store_true", help="download outputs when done")
    s = sub.add_parser("output", help="download the latest outputs")
    s.add_argument("name")
    s = sub.add_parser("upload-dataset", help="dry run by default; --yes uploads (private)")
    s.add_argument("folder")
    s.add_argument("--slug", required=True)
    s.add_argument("--title")
    s.add_argument("--owner")
    s.add_argument("--version-notes")
    s.add_argument("--public", action="store_true", help="make it public (check sharing rules first)")
    s.add_argument("--yes", action="store_true", help="really upload (only after explicit user approval)")
    sub.add_parser("list", help="list runs and uploads")
    args = p.parse_args(argv)

    ws = _workspace.resolve_workdir(args.workdir, args.competition)
    if args.cmd == "list":
        return cmd_list(ws, args)
    settings = K.load_settings(ws)
    api = K.connect()
    return {"init": cmd_init, "push": cmd_push, "status": cmd_status, "wait": cmd_wait,
            "output": cmd_output, "upload-dataset": cmd_upload}[args.cmd](api, ws, settings, args)


if __name__ == "__main__":
    sys.exit(main())
