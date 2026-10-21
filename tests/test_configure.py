"""Tests for first-run setup (configure.py), personal-settings carry-over and the locate reminder.

Run from the repository root:
    python -m unittest discover -s tests -v
"""
import contextlib
import datetime as dt
import io
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "plugins" / "kaggle-war-room" / "skills" / "kaggle-war-room" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import _kaggle  # noqa: E402
import configure  # noqa: E402
import init_competition  # noqa: E402
import workspace  # noqa: E402

EXAMPLE = (SCRIPTS.parent / "config" / "settings.example.yaml").read_text(encoding="utf-8")


def run(main, argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main(argv)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            err.write(str(e))
    return code, out.getvalue(), err.getvalue()


class TestSetKey(unittest.TestCase):
    def test_keeps_comments_and_other_lines(self):
        t = configure.set_key(EXAMPLE, "models.scout", "haiku")
        line = next(l for l in t.splitlines() if l.strip().startswith("scout:") and "haiku" in l)
        self.assertIn("# cheap + fast is enough", line)
        self.assertEqual(len(t.splitlines()), len(EXAMPLE.splitlines()))

    def test_block_scalar_replaced_cleanly(self):
        t = configure.set_key(EXAMPLE, "models.critic_command", "claude -p --add-dir {skill_dir}")
        data = yaml.safe_load(t)
        self.assertEqual(data["models"]["critic_command"], "claude -p --add-dir {skill_dir}")
        self.assertEqual(data["compute"]["kaggle_gpu_hours_per_week"], 30)  # next section intact
        self.assertNotIn("codex exec -s read-only --skip-git-repo-check -C {workdir} -o", t.split("critic_command:")[1].split("\n")[1])

    def test_lists_dicts_and_hash_in_value(self):
        gpus = [{"name": "RTX 4090", "vram_gb": 24, "hours_per_day": 8}]
        t = configure.set_key(EXAMPLE, "compute.local_gpus", gpus)
        t = configure.set_key(t, "scout.host_authors", ["Jane Doe", "Kaggle Staff"])
        t = configure.set_key(t, "metric.name", "map@0.5 # weird")
        d = yaml.safe_load(t)
        self.assertEqual(d["compute"]["local_gpus"], gpus)
        self.assertEqual(d["scout"]["host_authors"], ["Jane Doe", "Kaggle Staff"])
        self.assertEqual(d["metric"]["name"], "map@0.5 # weird")

    def test_refuses_sections_and_unknown_keys(self):
        for key in ("models", "phases.explore.until_days_left", "no.such.key"):
            with self.assertRaises((KeyError, ValueError)):
                configure.set_key(EXAMPLE, key, 1)


class FakeApi:
    CONFIG_NAME_USER = "username"

    def __init__(self, metric="Log Loss", code=True):
        self.metric, self.code = metric, code

    def competitions_list(self, search=None, page_size=20):
        return {"competitions": [
            {"ref": "https://www.kaggle.com/competitions/other", "deadline": dt.datetime(2030, 1, 1)},
            {"ref": "https://www.kaggle.com/competitions/comp-x", "deadline": dt.datetime(2026, 12, 31, 23, 59),
             "evaluation_metric": self.metric, "max_daily_submissions": 3,
             "is_kernels_submissions_only": self.code, "host_name": "Acme Labs"},
        ]}

    def get_config_value(self, name):
        return "me"


class SetupFlow(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.project = Path(self._tmp.name)
        self.root = self.project / "kaggle-war-room-workspace"
        self.init("comp-x")
        self.ws = self.root / "comp-x"
        self._connect, self._gpus = _kaggle.connect, configure.detect_local_gpus
        _kaggle.connect = lambda: FakeApi()
        configure.detect_local_gpus = lambda: []

    def tearDown(self):
        _kaggle.connect, configure.detect_local_gpus = self._connect, self._gpus
        self._tmp.cleanup()

    def init(self, slug):
        with contextlib.redirect_stdout(io.StringIO()):
            init_competition.main([slug, "--root", str(self.root)])

    def settings(self, ws=None):
        return yaml.safe_load(((ws or self.ws) / "settings.yaml").read_text(encoding="utf-8"))

    def W(self, ws=None):
        return ["--workdir", str(ws or self.ws)]

    def test_fresh_workspace_lists_required_items(self):
        code, out, _ = run(configure.main, self.W())
        self.assertEqual(code, 1)
        for key in ("competition.deadline", "metric.name", "competition.code_competition",
                    "submissions.sample_submission", "models.scout", "scout.host_authors"):
            self.assertIn(key, out)
        self.assertIn("REQUIRED, still missing", out)
        self.assertIn("ONE message", out)

    def test_detect_write_fills_facts_but_keeps_user_values(self):
        (self.project / "data").mkdir()
        (self.project / "data" / "sample_submission.csv").write_text("id,target\n1,0\n")
        run(configure.main, self.W() + ["set", "submissions.daily_limit=2"])
        code, out, _ = run(configure.main, self.W() + ["detect", "--write"])
        self.assertEqual(code, 0)
        s = self.settings()
        self.assertEqual(str(s["competition"]["deadline"]), "2026-12-31")
        self.assertEqual(s["metric"]["name"], "Log Loss")
        self.assertFalse(s["metric"]["higher_is_better"])          # guessed from "Log Loss"
        self.assertTrue(s["competition"]["code_competition"])
        self.assertEqual(s["kaggle_username"], "me")
        self.assertEqual(s["submissions"]["daily_limit"], 2)       # user's value kept
        self.assertTrue(s["submissions"]["sample_submission"].endswith("data/sample_submission.csv"))
        self.assertIn("Acme Labs", out)
        self.assertIn("keep yours", out)

    def test_detect_without_write_changes_nothing(self):
        before = (self.ws / "settings.yaml").read_text(encoding="utf-8")
        run(configure.main, self.W() + ["detect"])
        self.assertEqual((self.ws / "settings.yaml").read_text(encoding="utf-8"), before)

    def test_inference_limit_required_only_for_code_competitions(self):
        sample = self.project / "s.csv"
        sample.write_text("id,y\n")
        base = ["competition.deadline=2026-12-31", "metric.name=AUC", "submissions.daily_limit=5",
                f"submissions.sample_submission={sample.as_posix()}"]
        run(configure.main, self.W() + ["set", *base, "competition.code_competition=false"])
        self.assertEqual(run(configure.main, self.W())[0], 0)
        run(configure.main, self.W() + ["set", "competition.code_competition=true"])
        code, out, _ = run(configure.main, self.W())
        self.assertEqual(code, 1)
        self.assertIn("compute.inference_time_limit_hours", out.split("REQUIRED, still missing")[1])

    def test_done_refused_until_complete_then_locate_is_quiet(self):
        code, _, err = run(configure.main, self.W() + ["done"])
        self.assertNotEqual(code, 0)
        self.assertFalse(self.settings()["setup_complete"])
        sample = self.project / "s.csv"
        sample.write_text("id,y\n")
        run(configure.main, self.W() + ["set", "competition.deadline=2026-12-31", "metric.name=AUC",
                                        "competition.code_competition=false",
                                        f"submissions.sample_submission={sample.as_posix()}"])
        code, out, _ = run(configure.main, self.W() + ["done"])
        self.assertEqual(code, 0)
        self.assertTrue(self.settings()["setup_complete"])

    def test_missing_sample_file_is_reported(self):
        run(configure.main, self.W() + ["set", "submissions.sample_submission=/no/such/file.csv"])
        code, out, _ = run(configure.main, self.W())
        self.assertIn("file not found", out)
        self.assertEqual(code, 1)

    def test_personal_settings_carry_over_to_next_competition(self):
        run(configure.main, self.W() + ["set", "models.scout=haiku", "models.strategist=opus", "flavor=false",
                                        "metric.name=AUC", "competition.team_name=Team A"])
        self.init("comp-y")
        s = self.settings(self.root / "comp-y")
        self.assertEqual(s["models"]["scout"], "haiku")
        self.assertEqual(s["models"]["strategist"], "opus")
        self.assertFalse(s["flavor"])
        self.assertIsNone(s["metric"]["name"])                    # competition facts are not copied
        self.assertIsNone(s["competition"]["team_name"])
        self.assertFalse(s["setup_complete"])


class TestLocateReminder(unittest.TestCase):
    def test_locate_reminds_until_setup_done(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "kaggle-war-room-workspace"
            with contextlib.redirect_stdout(io.StringIO()):
                init_competition.main(["comp-z", "--root", str(root)])
            old = os.environ.get("KAGGLE_WAR_ROOM_WORKSPACE")
            os.environ["KAGGLE_WAR_ROOM_WORKSPACE"] = str(root)
            try:
                code, out, err = run(workspace.main, ["locate"])
                self.assertEqual(code, 0)
                self.assertTrue(out.strip().endswith("comp-z"))
                self.assertIn("first-run setup not finished", err)
                p = root / "comp-z" / "settings.yaml"
                p.write_text(p.read_text(encoding="utf-8").replace("setup_complete: false", "setup_complete: true"),
                             encoding="utf-8")
                code, out, err = run(workspace.main, ["locate"])
                self.assertNotIn("first-run setup", err)
            finally:
                if old is None:
                    os.environ.pop("KAGGLE_WAR_ROOM_WORKSPACE", None)
                else:
                    os.environ["KAGGLE_WAR_ROOM_WORKSPACE"] = old


if __name__ == "__main__":
    unittest.main()
