#!/usr/bin/env python3
"""Validate a submission CSV against sample_submission.csv before submitting.

Checks:
  - header: same column names in the same order
  - row count matches
  - id column: no duplicates, no missing ids, no extra ids
  - id order matches the sample (error with --strict-order, otherwise warning)
  - no empty / NaN / inf values (columns that are empty somewhere in the sample, e.g. RLE
    masks, may be empty)
  - columns numeric in the sample are numeric in the submission
  - optional value range (--range 0 1) for numeric prediction columns
  - optional allowed values for label columns (--allowed-values)

Exit code 0 = pass (warnings allowed), 1 = errors found, 2 = could not read files.

Examples:
  python check_submission.py submission.csv --sample data/sample_submission.csv
  python check_submission.py sub.csv --sample sample.csv --range 0 1 --strict-order
  python check_submission.py sub.csv --sample sample.csv --allowed-values label=cat,dog
"""
import argparse
import csv
import math
import sys
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))  # RLE strings can be long
NAN_TOKENS = {"nan", "na", "n/a", "null", "none", "inf", "-inf", "+inf", "infinity", "-infinity"}


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return [], []
        rows = [r for r in reader if r]  # ignore fully blank lines
    return header, rows


def is_number(s):
    try:
        v = float(s)
    except ValueError:
        return False
    return not (math.isnan(v) or math.isinf(v))


def check(sub_path, sample_path, id_col=None, strict_order=False, value_range=None,
          allowed=None, max_report=10):
    errors, warnings = [], []
    try:
        sh, srows = read_csv(sample_path)
        h, rows = read_csv(sub_path)
    except (OSError, UnicodeDecodeError) as e:
        return None, [f"cannot read files: {e}"], []

    if not h:
        return False, ["submission is empty"], []
    # header
    if h != sh:
        if [c.strip() for c in h] == [c.strip() for c in sh]:
            errors.append("header has extra whitespace around column names")
        elif set(h) == set(sh):
            errors.append(f"column order differs: expected {sh}, got {h}")
        else:
            missing = [c for c in sh if c not in h]
            extra = [c for c in h if c not in sh]
            errors.append(f"header mismatch: missing {missing}, extra {extra}")
    # row shape
    bad_len = [i + 2 for i, r in enumerate(rows) if len(r) != len(h)]
    if bad_len:
        errors.append(f"{len(bad_len)} rows have wrong number of fields (lines {bad_len[:max_report]})")
    if len(rows) != len(srows):
        errors.append(f"row count {len(rows)} != sample {len(srows)}")

    id_name = id_col or sh[0]
    if id_name not in sh:
        return False, errors + [f"id column '{id_name}' not in sample header"], warnings
    si = sh.index(id_name)
    sample_ids = [r[si] for r in srows if len(r) > si]
    if id_name in h:
        ii = h.index(id_name)
        ids = [r[ii] for r in rows if len(r) > ii]
        seen, dups = set(), []
        for x in ids:
            if x in seen:
                dups.append(x)
            seen.add(x)
        if dups:
            errors.append(f"{len(dups)} duplicate ids, e.g. {dups[:max_report]}")
        sset = set(sample_ids)
        missing = [x for x in sample_ids if x not in seen]
        extra = [x for x in ids if x not in sset]
        if missing:
            errors.append(f"{len(missing)} ids missing, e.g. {missing[:max_report]}")
        if extra:
            errors.append(f"{len(extra)} unexpected ids, e.g. {extra[:max_report]}")
        if not missing and not extra and not dups and ids != sample_ids:
            first = next(i for i, (a, b) in enumerate(zip(ids, sample_ids)) if a != b)
            msg = f"id order differs from sample (first at row {first + 2})"
            (errors if strict_order else warnings).append(msg)
    else:
        errors.append(f"id column '{id_name}' missing from submission")

    # values
    for col in sh:
        if col == id_name or col not in h:
            continue
        ci, sci = h.index(col), sh.index(col)
        svals = [r[sci] for r in srows if len(r) > sci]
        vals = [r[ci] for r in rows if len(r) > ci]
        sample_allows_empty = any(v.strip() == "" for v in svals)
        sample_numeric = bool(svals) and all(is_number(v) for v in svals if v.strip())
        empty = [i + 2 for i, v in enumerate(vals) if v.strip() == ""]
        nan = [i + 2 for i, v in enumerate(vals) if v.strip().lower() in NAN_TOKENS]
        if empty and not sample_allows_empty:
            errors.append(f"column '{col}': {len(empty)} empty values (lines {empty[:max_report]})")
        elif empty:
            warnings.append(f"column '{col}': {len(empty)} empty values (allowed: sample has empties)")
        if nan:
            errors.append(f"column '{col}': {len(nan)} NaN/inf values (lines {nan[:max_report]})")
        if sample_numeric:
            non_num = [i + 2 for i, v in enumerate(vals) if v.strip() and v.strip().lower() not in NAN_TOKENS
                       and not is_number(v)]
            if non_num:
                errors.append(f"column '{col}': {len(non_num)} non-numeric values (lines {non_num[:max_report]})")
            if value_range:
                lo, hi = value_range
                out = [i + 2 for i, v in enumerate(vals) if is_number(v) and not (lo <= float(v) <= hi)]
                if out:
                    errors.append(f"column '{col}': {len(out)} values outside [{lo}, {hi}] (lines {out[:max_report]})")
            nums = [float(v) for v in vals if is_number(v)]
            if len(nums) > 1 and len(set(nums)) == 1:
                warnings.append(f"column '{col}': all predictions are identical ({nums[0]})")
        if allowed and col in allowed:
            bad = sorted({v for v in vals if v not in allowed[col]})
            if bad:
                errors.append(f"column '{col}': values not allowed: {bad[:max_report]}")
    return not errors, errors, warnings


def parse_allowed(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--allowed-values expects col=v1,v2 - got {it!r}")
        col, vals = it.split("=", 1)
        out[col] = set(vals.split(","))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("submission", help="submission CSV to check")
    p.add_argument("--sample", required=True, help="sample_submission.csv from the competition data")
    p.add_argument("--id-col", help="id column name (default: first column of the sample)")
    p.add_argument("--strict-order", action="store_true", help="treat id order differences as errors")
    p.add_argument("--range", nargs=2, type=float, metavar=("LO", "HI"), help="allowed range for numeric columns")
    p.add_argument("--allowed-values", nargs="*", metavar="COL=V1,V2", help="allowed values for label columns")
    args = p.parse_args(argv)

    for f in (args.submission, args.sample):
        if not Path(f).exists():
            print(f"ERROR: file not found: {f}")
            return 2
    ok, errors, warnings = check(args.submission, args.sample, args.id_col, args.strict_order,
                                 args.range, parse_allowed(args.allowed_values))
    if ok is None:
        print("ERROR: " + errors[0])
        return 2
    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR: {e}")
    print("PASS" if ok else f"FAIL ({len(errors)} errors)")
    if ok:
        print("Reminder: get explicit user approval before submitting.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
