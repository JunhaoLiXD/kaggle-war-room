#!/usr/bin/env python3
"""Create the CV split (folds.csv) every experiment must share. numpy only, no sklearn.

Schemes (--scheme, default auto):
  kfold            shuffled K folds
  stratified       class proportions kept equal across folds (--target)
  group            each group (--group: patient, scene, video, site, ...) in exactly one fold
  stratified-group both of the above (greedy, like sklearn's StratifiedGroupKFold)
  time             contiguous blocks ordered by --time (fold 0 = earliest); validate on later
                   folds than you train on
  auto             group+target -> stratified-group, group -> group, target -> stratified, else kfold

A numeric target with many distinct values is binned into --bins quantile bins for stratification.
Writes <WORKDIR>/folds.csv (id column(s) + fold) and folds.json (scheme, seed, fold sizes and
balance). An existing folds.csv is never overwritten without --force: changing the split makes
every past experiment incomparable - record the reason in decisions.md if you must.

Examples:
  python make_folds.py data/train.csv --id image_id --target label --group patient_id
  python make_folds.py data/train.csv --id id --target y --folds 10 --seed 7
  python make_folds.py data/train.csv --id id --time date --folds 5
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _workspace  # noqa: E402

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def read_columns(path, cols):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in cols if c and c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"columns not found in {path}: {missing}. Available: {reader.fieldnames}")
        data = {c: [] for c in cols if c}
        for row in reader:
            for c in data:
                data[c].append(row[c])
    return data


def stratify_labels(values, bins):
    """Class labels for stratification; numeric targets with many values become quantile bins."""
    try:
        nums = np.array([float(v) for v in values])
        if len(set(values)) > max(bins, 20):
            edges = np.unique(np.quantile(nums, np.linspace(0, 1, bins + 1)[1:-1]))
            return [f"bin{int(b)}" for b in np.searchsorted(edges, nums, side="right")]
    except ValueError:
        pass
    return list(values)


def kfold(n, k, rng):
    folds = np.empty(n, dtype=int)
    for f, part in enumerate(np.array_split(rng.permutation(n), k)):
        folds[part] = f
    return folds


def stratified(labels, k, rng):
    labels = np.asarray(labels)
    folds = np.empty(len(labels), dtype=int)
    for cls in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == cls))
        offset = rng.integers(k)
        folds[idx] = (np.arange(len(idx)) + offset) % k
    return folds


def group_split(groups, k, rng, labels=None):
    """Assign whole groups to folds. With labels: keep class proportions balanced (greedy)."""
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    if len(uniq) < k:
        raise SystemExit(f"only {len(uniq)} groups for {k} folds - use fewer folds")
    sizes = np.bincount(inv)
    if labels is not None:
        cls, lab = np.unique(np.asarray(labels), return_inverse=True)
        counts = np.zeros((len(uniq), len(cls)))
        np.add.at(counts, (inv, lab), 1)
    else:
        counts = sizes[:, None].astype(float)
    totals = counts.sum(0)
    order = rng.permutation(len(uniq))
    if labels is not None:  # most class-skewed (and largest) groups first, as sklearn does
        skew = (counts / np.maximum(counts.sum(1, keepdims=True), 1)).std(1)
        order = order[np.lexsort((-sizes[order], -skew[order]))]
    else:
        order = order[np.argsort(-sizes[order], kind="stable")]
    fold_counts = np.zeros((k, counts.shape[1]))
    fold_sizes = np.zeros(k)
    assign = np.empty(len(uniq), dtype=int)
    for g in order:
        if labels is None:
            best = int(np.argmin(fold_sizes))
        else:
            costs = []
            for f in range(k):
                trial = fold_counts.copy()
                trial[f] += counts[g]
                costs.append(((trial / np.maximum(totals, 1)).std(0)).mean())
            costs = np.array(costs)
            ties = np.flatnonzero(np.isclose(costs, costs.min()))
            best = int(ties[np.argmin(fold_sizes[ties])])
        assign[g] = best
        fold_counts[best] += counts[g]
        fold_sizes[best] += sizes[g]
    return assign[inv]


def time_split(times, k):
    order = np.argsort(np.asarray(times), kind="stable")
    folds = np.empty(len(times), dtype=int)
    for f, part in enumerate(np.array_split(order, k)):
        folds[part] = f
    return folds


def summarize(folds, k, labels=None, groups=None):
    out = {"fold_sizes": [int((folds == f).sum()) for f in range(k)]}
    if groups is not None:
        g = np.asarray(groups)
        out["groups_per_fold"] = [int(len(np.unique(g[folds == f]))) for f in range(k)]
        per_group = {}
        for gi, f in zip(g, folds):
            per_group.setdefault(gi, set()).add(int(f))
        out["groups_split_across_folds"] = sum(1 for s in per_group.values() if len(s) > 1)
    if labels is not None:
        lab = np.asarray(labels)
        top = [c for c, _ in Counter(lab.tolist()).most_common(10)]
        out["class_share_by_fold"] = {str(c): [round(float((lab[folds == f] == c).mean()), 4) if (folds == f).any() else 0.0
                                               for f in range(k)] for c in top}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("train", help="training CSV")
    p.add_argument("--id", nargs="*", default=[], help="id column(s) copied to folds.csv (default: row index)")
    p.add_argument("--target", help="target column for stratification")
    p.add_argument("--group", help="group column (patient, scene, video, site, ...)")
    p.add_argument("--time", help="time column for a time-ordered split")
    p.add_argument("--scheme", default="auto", choices=["auto", "kfold", "stratified", "group", "stratified-group", "time"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bins", type=int, default=10, help="quantile bins for a continuous target")
    p.add_argument("--out", help="output CSV (default <WORKDIR>/folds.csv)")
    p.add_argument("--workdir", help="competition folder (default: active competition)")
    p.add_argument("--force", action="store_true", help="overwrite an existing folds file")
    args = p.parse_args(argv)

    if args.folds < 2:
        p.error("--folds must be at least 2")
    scheme = args.scheme
    if scheme == "auto":
        scheme = ("time" if args.time else "stratified-group" if args.group and args.target
                  else "group" if args.group else "stratified" if args.target else "kfold")
    need = {"stratified": ["target"], "group": ["group"], "stratified-group": ["target", "group"], "time": ["time"]}
    for col in need.get(scheme, []):
        if not getattr(args, col):
            p.error(f"--scheme {scheme} needs --{col}")

    out = Path(args.out) if args.out else _workspace.resolve_workdir(args.workdir) / "folds.csv"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists. Changing the split invalidates every past experiment; pass --force "
                         "only if that is intended, and record why in decisions.md.")

    data = read_columns(args.train, list(dict.fromkeys(args.id + [args.target, args.group, args.time])))
    n = len(next(iter(data.values()))) if data else sum(1 for _ in open(args.train, encoding="utf-8-sig")) - 1
    if n < args.folds:
        raise SystemExit(f"only {n} rows for {args.folds} folds")
    rng = np.random.default_rng(args.seed)
    labels = stratify_labels(data[args.target], args.bins) if args.target else None
    groups = data[args.group] if args.group else None

    if scheme == "kfold":
        folds = kfold(n, args.folds, rng)
    elif scheme == "stratified":
        folds = stratified(labels, args.folds, rng)
    elif scheme == "group":
        folds = group_split(groups, args.folds, rng)
    elif scheme == "stratified-group":
        folds = group_split(groups, args.folds, rng, labels)
    else:
        times = data[args.time]
        try:
            times = [float(t) for t in times]
        except ValueError:
            pass  # ISO dates sort correctly as strings
        folds = time_split(times, args.folds)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        id_cols = args.id or ["row_index"]
        w.writerow(id_cols + ["fold"])
        for i in range(n):
            ids = [data[c][i] for c in args.id] if args.id else [i]
            w.writerow(ids + [int(folds[i])])

    summary = summarize(folds, args.folds, labels, groups)
    meta = {"scheme": scheme, "folds": args.folds, "seed": args.seed, "train": str(args.train),
            "id": args.id or ["row_index"], "target": args.target, "group": args.group, "time": args.time,
            "rows": n, **summary}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    print(f"Wrote {out} ({n} rows, scheme={scheme}, folds={args.folds}, seed={args.seed})")
    print(f"fold sizes: {summary['fold_sizes']}")
    if groups is not None:
        print(f"groups per fold: {summary['groups_per_fold']}  groups split across folds: {summary['groups_split_across_folds']}")
    if labels is not None:
        worst = max((max(v) - min(v), c) for c, v in summary["class_share_by_fold"].items())
        print(f"largest class-share gap between folds: {worst[0]:.4f} (class {worst[1]})")
    if scheme == "time":
        print("Time split: validate on folds later than the training folds (forward chaining), not on random folds.")
    print("Record the scheme in baseline.md; every experiment must use this file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
