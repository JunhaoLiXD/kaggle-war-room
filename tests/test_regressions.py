"""Regression tests for bugs fixed in 0.1.1, plus a --help smoke test for every script.

Run from the repository root (needs numpy and pyyaml):
    python -m unittest discover -s tests -v
"""
import contextlib
import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "plugins" / "kaggle-war-room" / "skills" / "kaggle-war-room" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import fetch_updates  # noqa: E402
import log_experiment  # noqa: E402
import pred_correlation  # noqa: E402


def run_json(main, argv):
    """Call a script's main() and parse its JSON stdout."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(argv)
    return code, json.loads(buf.getvalue())


class Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def make_preds(self, n=1000, k=3, seed=0):
        rng = np.random.default_rng(seed)
        y = (rng.random(n) < 0.3).astype(float)
        np.save(self.dir / "y.npy", y)
        paths = []
        for i in range(k):
            p = self.dir / f"m{i}.npy"
            np.save(p, np.clip(y * 0.5 + rng.normal(0.3, 0.3 + 0.05 * i, n), 0, 1))
            paths.append(str(p))
        return paths, str(self.dir / "y.npy")


class TestBlendSubsampling(Tmp):
    """Bug 1: blend path crashed when predictions were subsampled (--max-points < size)."""

    def test_subsampled_blend_path_runs(self):
        paths, labels = self.make_preds(n=1000, k=3)
        code, res = run_json(pred_correlation.main,
                             [*paths, "--labels", labels, "--max-points", "500", "--margin", "1", "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(len(res["blend_path"]), 3)
        for step in res["blend_path"][1:]:
            self.assertTrue(-1.0 <= step["corr_with_blend"] <= 1.0)

    def test_subsampling_uses_same_indices_as_no_subsampling_order(self):
        # With subsampling covering every point, results must equal the unsampled run.
        paths, labels = self.make_preds(n=400, k=3)
        _, full = run_json(pred_correlation.main, [*paths, "--labels", labels, "--margin", "1", "--json"])
        _, same = run_json(pred_correlation.main,
                           [*paths, "--labels", labels, "--margin", "1", "--max-points", "399", "--json"])
        self.assertEqual([s["added"] for s in full["blend_path"]], [s["added"] for s in same["blend_path"]])


class TestMixedShapes(Tmp):
    """Bug 2: one-column CSV (n,1) + NPY (n,) broadcast to (n,n) during blending."""

    def write_csv(self, name, values):
        path = self.dir / name
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "pred"])
            for i, v in enumerate(values):
                w.writerow([i, float(v)])
        return str(path)

    def test_single_column_csv_loads_as_1d(self):
        p = self.write_csv("c.csv", np.linspace(0, 1, 50))
        self.assertEqual(pred_correlation.load(p, "id").shape, (50,))

    def test_csv_and_npy_blend(self):
        paths, labels = self.make_preds(n=300, k=1)
        c = self.write_csv("c.csv", np.load(paths[0]) * 0.9)
        code, res = run_json(pred_correlation.main,
                             [paths[0], c, "--id-col", "id", "--labels", labels, "--margin", "1", "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(len(res["blend_path"]), 2)
        self.assertTrue(0.0 <= res["blend_path"][-1]["blend_score"] <= 1.0)

    def test_mismatched_shapes_rejected(self):
        a = self.dir / "a.npy"
        b = self.dir / "b.npy"
        np.save(a, np.zeros((100, 2)))
        np.save(b, np.zeros(200))  # same size, different shape
        with self.assertRaises(SystemExit) as cm:
            pred_correlation.main([str(a), str(b)])
        self.assertIn("shapes differ", str(cm.exception))

    def test_label_shape_checked(self):
        preds = {"a": np.zeros(10), "b": np.zeros(10)}
        self.assertIsNone(pred_correlation.check_shapes(preds, np.zeros(10)))
        self.assertIn("labels have shape", pred_correlation.check_shapes(preds, np.zeros((10, 2))))
        multi = {"a": np.zeros((10, 3)), "b": np.zeros((10, 3))}
        self.assertIsNone(pred_correlation.check_shapes(multi, np.zeros(10)))  # class ids per row


class TestDownloadFailureSummary(unittest.TestCase):
    """Bug 3: a failed notebook download ({ref, error}) crashed summarize() with KeyError 'path'."""

    def test_failed_download_reported_not_crashing(self):
        ch = {
            "current_snapshot": "t1", "previous_snapshot": "t0",
            "leaderboard": {"top_score": 0.9, "top_score_delta": 0.0, "new_in_top": []},
            "notebooks": {"new": [], "changed": [], "movers": [], "dropped_out": []},
            "topics": {"new": [], "active": []},
            "downloaded": [
                {"ref": "alice/ok", "path": "/x/v2.ipynb", "previous": "/x/v1.ipynb"},
                {"ref": "bob/broken", "error": "timeout"},
            ],
        }
        text = fetch_updates.summarize(ch)
        self.assertIn("alice/ok: /x/v2.ipynb", text)
        self.assertIn("Download failures (1)", text)
        self.assertIn("bob/broken: timeout", text)


class TestLogPrecision(Tmp):
    """Bug 4: scores were saved with ':g' (6 significant digits)."""

    def test_full_precision_kept(self):
        (self.dir / "experiments.csv").write_text(",".join(log_experiment.COLUMNS) + "\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            log_experiment.main(["--workdir", str(self.dir), "--solution", "a", "--cv", "0.987654321"])
            log_experiment.main(["--workdir", str(self.dir), "--solution", "b", "--cv", "0.9876543"])
            log_experiment.main(["--workdir", str(self.dir), "--update", "E001", "--lb", "0.123456789012"])
        rows = log_experiment.read_rows(self.dir / "experiments.csv")
        self.assertEqual(rows[0]["cv"], "0.987654321")
        self.assertEqual(rows[1]["cv"], "0.9876543")
        self.assertEqual(rows[0]["lb"], "0.123456789012")
        self.assertNotEqual(rows[0]["cv"], rows[1]["cv"])


class TestScriptsHelp(unittest.TestCase):
    def test_every_script_has_help(self):
        scripts = sorted(p for p in SCRIPTS.glob("*.py") if not p.name.startswith("_"))
        self.assertGreaterEqual(len(scripts), 9)
        for s in scripts:
            r = subprocess.run([sys.executable, str(s), "--help"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"{s.name} --help failed: {r.stderr[-300:]}")


if __name__ == "__main__":
    unittest.main()
