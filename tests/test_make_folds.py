"""Tests for make_folds.py (the shared CV split).

Run from the repository root:
    python -m unittest discover -s tests -v
"""
import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "plugins" / "kaggle-war-room" / "skills" / "kaggle-war-room" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import make_folds  # noqa: E402


class TestMakeFolds(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        rng = np.random.default_rng(1)
        self.train = self.dir / "train.csv"
        with open(self.train, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "label", "patient", "date", "y"])
            for i in range(1200):
                p = int(rng.integers(0, 90))
                label = "rare" if rng.random() < 0.05 else ("a" if p % 3 else "b")
                w.writerow([f"img{i}", label, f"p{p}", f"2026-{1 + i % 12:02d}-{1 + i % 28:02d}",
                            round(float(rng.normal()), 4)])
        self.rows = list(csv.DictReader(open(self.train, newline="")))

    def tearDown(self):
        self._tmp.cleanup()

    def make(self, *args, out="folds.csv"):
        out_path = self.dir / out
        with contextlib.redirect_stdout(io.StringIO()):
            code = make_folds.main([str(self.train), "--out", str(out_path), *args])
        self.assertEqual(code, 0)
        folds = {r["image_id"]: int(r["fold"]) for r in csv.DictReader(open(out_path, newline=""))}
        meta = json.loads(out_path.with_suffix(".json").read_text())
        return folds, meta

    def test_group_never_split(self):
        for scheme_args in (["--group", "patient"], ["--group", "patient", "--target", "label"]):
            folds, meta = self.make("--id", "image_id", *scheme_args, out=f"g{len(scheme_args)}.csv")
            per_patient = {}
            for r in self.rows:
                per_patient.setdefault(r["patient"], set()).add(folds[r["image_id"]])
            self.assertTrue(all(len(s) == 1 for s in per_patient.values()))
            self.assertEqual(meta["groups_split_across_folds"], 0)
        self.assertEqual(meta["scheme"], "stratified-group")

    def test_stratified_balance(self):
        folds, meta = self.make("--id", "image_id", "--target", "label")
        self.assertEqual(meta["scheme"], "stratified")
        for cls, shares in meta["class_share_by_fold"].items():
            self.assertLess(max(shares) - min(shares), 0.02, cls)
        self.assertLessEqual(max(meta["fold_sizes"]) - min(meta["fold_sizes"]), 2)

    def test_stratified_group_beats_plain_group_on_balance(self):
        _, sg = self.make("--id", "image_id", "--group", "patient", "--target", "label", out="sg.csv")
        _, g = self.make("--id", "image_id", "--group", "patient", "--scheme", "group", "--target", "label", out="g.csv")
        gap = lambda m: max(max(v) - min(v) for v in m["class_share_by_fold"].values())
        self.assertLessEqual(gap(sg), gap(g))

    def test_time_split_is_ordered(self):
        folds, meta = self.make("--id", "image_id", "--time", "date")
        self.assertEqual(meta["scheme"], "time")
        dates = {}
        for r in self.rows:
            dates.setdefault(folds[r["image_id"]], []).append(r["date"])
        for f in range(4):
            self.assertLessEqual(max(dates[f]), min(dates[f + 1]))

    def test_continuous_target_is_binned(self):
        _, meta = self.make("--id", "image_id", "--target", "y")
        self.assertTrue(all(c.startswith("bin") for c in meta["class_share_by_fold"]))

    def test_seed_reproducible_and_overwrite_guard(self):
        a, _ = self.make("--id", "image_id", "--seed", "3", out="a.csv")
        b, _ = self.make("--id", "image_id", "--seed", "3", out="b.csv")
        self.assertEqual(a, b)
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stdout(io.StringIO()):
                make_folds.main([str(self.train), "--out", str(self.dir / "a.csv"), "--id", "image_id"])

    def test_missing_column_and_too_few_groups(self):
        with self.assertRaises(SystemExit):
            make_folds.main([str(self.train), "--out", str(self.dir / "x.csv"), "--target", "nope"])
        with self.assertRaises(SystemExit):
            make_folds.main([str(self.train), "--out", str(self.dir / "y.csv"), "--group", "label", "--folds", "5"])


if __name__ == "__main__":
    unittest.main()
