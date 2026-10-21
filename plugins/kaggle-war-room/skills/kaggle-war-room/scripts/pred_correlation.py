#!/usr/bin/env python3
"""Correlation matrix of OOF / test predictions across models, for diversity-based blend selection.

Inputs: .npy files (any shape; flattened) or .csv files (all numeric columns except
--id-col, flattened row-major). All inputs must have the same number of values and be
aligned to the same ids/order.

Outputs:
  - Pearson and Spearman correlation matrices
  - with --labels: each model's score, and a greedy diversity-first blend path: start from
    the best model, repeatedly add the candidate least correlated with the current blend
    (among models within --margin of the best score), report blended score after each step.

Metrics for --metric: auc (binary), logloss, accuracy (argmax for 2-D), rmse, mae, dice
(binary masks, threshold --threshold). Pure numpy - no sklearn/scipy needed.

Examples:
  python pred_correlation.py preds/*_oof.npy
  python pred_correlation.py preds/e01_oof.npy preds/e02_oof.npy preds/e03_oof.npy \
      --labels preds/oof_labels.npy --metric auc
  python pred_correlation.py a.csv b.csv --id-col image_id --json
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HIGHER_IS_BETTER = {"auc": True, "accuracy": True, "dice": True, "logloss": False, "rmse": False, "mae": False}


def load(path, id_col=None):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False).astype(np.float64)
    if path.suffix == ".csv":
        import csv
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        header, rows = rows[0], rows[1:]
        keep = [i for i, c in enumerate(header) if c != id_col]
        data = []
        for r in rows:
            vals = []
            for i in keep:
                try:
                    vals.append(float(r[i]))
                except ValueError:
                    break
            else:
                data.append(vals)
                continue
            raise SystemExit(f"{path}: non-numeric value in row {r[:3]}... use --id-col for the id column")
        return np.asarray(data, dtype=np.float64)
    raise SystemExit(f"unsupported file type: {path} (use .npy or .csv)")


def rankdata(x):
    """Average ranks (ties share the mean rank), like scipy.stats.rankdata."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    boundaries = np.flatnonzero(np.diff(xs)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(x)]))
    for s, e in zip(starts, ends):
        ranks[order[s:e]] = (s + e + 1) / 2.0
    return ranks


def corr_matrix(mat):
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(mat)
    return np.nan_to_num(c, nan=0.0)


def score(metric, y, p, threshold=0.5):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if metric == "auc":
        y, p = y.ravel(), p.ravel()
        pos = y == 1
        n_pos, n_neg = pos.sum(), (~pos).sum()
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        r = rankdata(p)
        return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    if metric == "logloss":
        eps = 1e-15
        if p.ndim == 2 and p.shape[1] > 1:
            pc = np.clip(p, eps, 1)
            pc = pc / pc.sum(1, keepdims=True)
            return float(-np.mean(np.log(pc[np.arange(len(pc)), y.astype(int).ravel()])))
        p, y = np.clip(p.ravel(), eps, 1 - eps), y.ravel()
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    if metric == "accuracy":
        if p.ndim == 2 and p.shape[1] > 1:
            return float(np.mean(p.argmax(1) == y.astype(int).ravel()))
        return float(np.mean((p.ravel() >= threshold) == (y.ravel() >= 0.5)))
    if metric == "rmse":
        return float(np.sqrt(np.mean((p.ravel() - y.ravel()) ** 2)))
    if metric == "mae":
        return float(np.mean(np.abs(p.ravel() - y.ravel())))
    if metric == "dice":
        a, b = p.ravel() >= threshold, y.ravel() >= 0.5
        denom = a.sum() + b.sum()
        return float(1.0 if denom == 0 else 2 * (a & b).sum() / denom)
    raise SystemExit(f"unknown metric {metric}")


def greedy_blend(names, preds, flat, y, metric, margin, threshold):
    hib = HIGHER_IS_BETTER[metric]
    scores = {n: score(metric, y, preds[n], threshold) for n in names}
    best = max(scores, key=scores.get) if hib else min(scores, key=scores.get)
    eligible = [n for n in names if (scores[best] - scores[n] if hib else scores[n] - scores[best]) <= margin]
    chosen = [best]
    blend = preds[best].copy()
    path = [{"step": 1, "added": best, "corr_with_blend": None, "blend_score": scores[best]}]
    while True:
        rest = [n for n in eligible if n not in chosen]
        if not rest:
            break
        bflat = blend.ravel()
        corrs = {n: float(np.nan_to_num(np.corrcoef(bflat, flat[n])[0, 1])) for n in rest}
        nxt = min(corrs, key=corrs.get)
        new_blend = (blend * len(chosen) + preds[nxt]) / (len(chosen) + 1)
        chosen.append(nxt)
        blend = new_blend
        path.append({"step": len(chosen), "added": nxt, "corr_with_blend": corrs[nxt],
                     "blend_score": score(metric, y, blend, threshold)})
    return scores, path


def fmt_matrix(names, m):
    short = [Path(n).stem for n in names]
    w = max(8, max(len(s) for s in short))
    lines = [" " * w + " | " + " | ".join(f"{s[:8]:>8}" for s in short)]
    for s, row in zip(short, m):
        lines.append(f"{s:<{w}} | " + " | ".join(f"{v:8.4f}" for v in row))
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("preds", nargs="+", help="prediction files (.npy/.csv); globs allowed")
    p.add_argument("--labels", help="ground-truth labels for OOF (.npy/.csv) - enables scores and blend path")
    p.add_argument("--metric", choices=sorted(HIGHER_IS_BETTER), default="auc", help="metric for scoring (default auc)")
    p.add_argument("--margin", type=float, default=0.01, help="only blend models within this score of the best")
    p.add_argument("--threshold", type=float, default=0.5, help="threshold for accuracy/dice")
    p.add_argument("--id-col", help="id column to drop from .csv inputs")
    p.add_argument("--max-points", type=int, default=2_000_000, help="subsample flattened arrays above this size")
    p.add_argument("--json", action="store_true", help="print JSON")
    args = p.parse_args(argv)

    files = []
    for pat in args.preds:
        files.extend(sorted(glob.glob(pat)) or [pat])
    files = list(dict.fromkeys(files))
    if len(files) < 2:
        p.error("need at least two prediction files")
    preds = {f: load(f, args.id_col) for f in files}
    sizes = {f: a.size for f, a in preds.items()}
    if len(set(sizes.values())) != 1:
        raise SystemExit(f"prediction sizes differ: {sizes}")

    n = next(iter(sizes.values()))
    idx = None
    if n > args.max_points:
        idx = np.random.default_rng(0).choice(n, args.max_points, replace=False)
    flat = {f: (a.ravel()[idx] if idx is not None else a.ravel()) for f, a in preds.items()}
    mat = np.vstack([flat[f] for f in files])
    pearson = corr_matrix(mat)
    spearman = corr_matrix(np.vstack([rankdata(r) for r in mat]))

    result = {"files": files, "pearson": pearson.round(6).tolist(), "spearman": spearman.round(6).tolist()}
    if args.labels:
        y = load(args.labels, args.id_col)
        if y.shape[0] != next(iter(preds.values())).shape[0]:
            raise SystemExit(f"labels have {y.shape[0]} rows, predictions have {next(iter(preds.values())).shape[0]}")
        scores, path = greedy_blend(files, preds, flat, y, args.metric, args.margin, args.threshold)
        result["metric"] = args.metric
        result["scores"] = scores
        result["blend_path"] = path

    if args.json:
        print(json.dumps(result, indent=1))
        return 0
    print("## Pearson correlation\n" + fmt_matrix(files, pearson))
    print("\n## Spearman correlation\n" + fmt_matrix(files, spearman))
    off = pearson[~np.eye(len(files), dtype=bool)]
    print(f"\nMean off-diagonal Pearson: {off.mean():.4f}  (>0.98 ~ near-duplicate models)")
    if args.labels:
        print(f"\n## Individual {args.metric} ({'higher' if HIGHER_IS_BETTER[args.metric] else 'lower'} is better)")
        for f, s in sorted(result["scores"].items(), key=lambda kv: kv[1], reverse=HIGHER_IS_BETTER[args.metric]):
            print(f"- {Path(f).stem}: {s:.6f}")
        print(f"\n## Greedy diversity-first blend path (equal weights, margin {args.margin})")
        for st in result["blend_path"]:
            c = "" if st["corr_with_blend"] is None else f" corr_with_blend={st['corr_with_blend']:.4f}"
            print(f"{st['step']}. + {Path(st['added']).stem}{c} -> blend {args.metric}={st['blend_score']:.6f}")
        print("Keep a member only if it improves blended CV by more than min_gain (see references/ensembling.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
