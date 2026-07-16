"""Tests for scenario-file provenance freezing in the run orchestrator.

    .venv/bin/python -m unittest tests.test_scenario_freeze -v

The orchestrator copies the --scenario-file into the run dir so a resumed or
re-analysed run stays self-contained even if the mutable repo copy is later edited.
Covers:
- _extract_flag_value: space form, equals form, absent, malformed (flag last)
- _freeze_scenario_file: copies into run dir, no-op without the flag, warns+skips
  on a missing file, preserves basename for an absolute path
"""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import run_llm_probability_simulation_analysis as orch  # noqa: E402


class ExtractFlagValueTests(unittest.TestCase):
    def test_space_form(self):
        argv = ["--runs", "20", "--scenario-file", "scenarios_a3.py", "--new"]
        self.assertEqual(orch._extract_flag_value(argv, "--scenario-file"), "scenarios_a3.py")

    def test_equals_form(self):
        argv = ["--runs", "20", "--scenario-file=scenarios_a3.py", "--new"]
        self.assertEqual(orch._extract_flag_value(argv, "--scenario-file"), "scenarios_a3.py")

    def test_absent(self):
        argv = ["--runs", "20", "--new"]
        self.assertIsNone(orch._extract_flag_value(argv, "--scenario-file"))

    def test_flag_last_no_value(self):
        # Malformed (flag with no following value) must not raise.
        argv = ["--runs", "20", "--scenario-file"]
        self.assertIsNone(orch._extract_flag_value(argv, "--scenario-file"))


class FreezeScenarioFileTests(unittest.TestCase):
    def test_copies_file_into_run_dir(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            scenario = root / "scenarios_x.py"
            scenario.write_text("CONTEXT_SCENARIOS = {'baseline': {}}\n", encoding="utf-8")
            run_dir = root / "run_dir"
            run_dir.mkdir()
            with mock.patch.object(orch, "REPO_ROOT", root):
                frozen = orch._freeze_scenario_file(
                    ["--scenario-file", "scenarios_x.py"], run_dir)
            self.assertEqual(frozen, run_dir / "scenarios_x.py")
            self.assertTrue(frozen.exists())
            self.assertEqual(
                frozen.read_text(encoding="utf-8"),
                scenario.read_text(encoding="utf-8"))

    def test_returns_none_without_flag(self):
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d) / "run_dir"
            run_dir.mkdir()
            self.assertIsNone(orch._freeze_scenario_file(["--runs", "20"], run_dir))
            self.assertEqual(list(run_dir.iterdir()), [])

    def test_warns_and_skips_when_missing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run_dir = root / "run_dir"
            run_dir.mkdir()
            buf = io.StringIO()
            with mock.patch.object(orch, "REPO_ROOT", root), \
                    contextlib.redirect_stdout(buf):
                frozen = orch._freeze_scenario_file(
                    ["--scenario-file", "does_not_exist.py"], run_dir)
            self.assertIsNone(frozen)
            self.assertEqual(list(run_dir.iterdir()), [])
            self.assertIn("does not exist", buf.getvalue())

    def test_preserves_basename_for_absolute_path(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            other = root / "elsewhere"
            other.mkdir()
            scenario = other / "scenarios_prod.py"
            scenario.write_text("CONTEXT_SCENARIOS = {'baseline': {}}\n", encoding="utf-8")
            run_dir = root / "run_dir"
            run_dir.mkdir()
            with mock.patch.object(orch, "REPO_ROOT", root):
                frozen = orch._freeze_scenario_file(
                    ["--scenario-file", str(scenario)], run_dir)
            self.assertEqual(frozen, run_dir / "scenarios_prod.py")
            self.assertTrue(frozen.exists())


if __name__ == "__main__":
    unittest.main()
