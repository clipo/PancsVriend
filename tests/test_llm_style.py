"""Tests for the llm_style request-style parameter and scenario_file loading.

    .venv/bin/python -m unittest tests.test_llm_style -v

Covers:
- resolve_llm_style normalisation + rejection of unknown styles
- resolve_llm_request_url path rewriting per style
- build_llm_request payload shape for all four styles + legacy (None)
- apply_scenario_file: in-place swap, validation, missing file
- YAML contexts_args -> CLI flag passthrough for the new keys
- end-to-end: LLMAgent sends the style-appropriate request (requests.post mocked)
"""
import copy
import os
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import llm_runner  # noqa: E402
from llm_runner import (  # noqa: E402
    LLM_STYLES,
    MOVE_STAY_GRAMMAR,
    apply_scenario_file,
    build_llm_request,
    resolve_llm_request_url,
    resolve_llm_style,
)
from context_scenarios import CONTEXT_SCENARIOS  # noqa: E402

BASE_URL = "http://localhost:8082/v1/completions"


class TestResolveStyle(unittest.TestCase):
    def test_all_canonical_styles_accepted(self):
        for style in LLM_STYLES:
            self.assertEqual(resolve_llm_style(style), style)

    def test_none_and_empty_mean_legacy(self):
        self.assertIsNone(resolve_llm_style(None))
        self.assertIsNone(resolve_llm_style(""))
        self.assertIsNone(resolve_llm_style("  "))

    def test_normalisation(self):
        self.assertEqual(resolve_llm_style("Completions+Grammar"), "completions+grammar")
        self.assertEqual(resolve_llm_style("chat grammar"), "chat+grammar")
        self.assertEqual(resolve_llm_style("completions_grammar"), "completions+grammar")

    def test_unknown_style_rejected(self):
        for bad in ("grammar", "chatcompletions", "raw", "chat++extra"):
            with self.assertRaises(ValueError):
                resolve_llm_style(bad)


class TestResolveUrl(unittest.TestCase):
    def test_style_none_keeps_url(self):
        self.assertEqual(resolve_llm_request_url(BASE_URL, None), BASE_URL)
        chat_url = "http://localhost:8082/v1/chat/completions"
        self.assertEqual(resolve_llm_request_url(chat_url, None), chat_url)

    def test_completions_styles_target_completions(self):
        for style in ("completions", "completions+grammar"):
            self.assertEqual(resolve_llm_request_url(BASE_URL, style),
                             "http://localhost:8082/v1/completions")
            # rewrites even when the URL points at chat
            self.assertEqual(
                resolve_llm_request_url("http://localhost:8082/v1/chat/completions", style),
                "http://localhost:8082/v1/completions")

    def test_chat_styles_target_chat(self):
        for style in ("chat", "chat+grammar"):
            self.assertEqual(resolve_llm_request_url(BASE_URL, style),
                             "http://localhost:8082/v1/chat/completions")


class TestBuildRequest(unittest.TestCase):
    def _build(self, style, url=BASE_URL):
        return build_llm_request(url, style, "m", "PROMPT", 0.3)

    def test_completions_payload(self):
        url, p = self._build("completions")
        self.assertTrue(url.endswith("/v1/completions"))
        self.assertEqual(p["prompt"], "PROMPT")
        self.assertNotIn("messages", p)
        self.assertNotIn("grammar", p)
        self.assertEqual(p["max_tokens"], 5)
        self.assertEqual(p["top_k"], 0)          # pinned sampler present

    def test_completions_grammar_payload(self):
        url, p = self._build("completions+grammar")
        self.assertTrue(url.endswith("/v1/completions"))
        self.assertEqual(p["prompt"], "PROMPT")
        self.assertEqual(p["grammar"], MOVE_STAY_GRAMMAR)

    def test_chat_payload(self):
        url, p = self._build("chat")
        self.assertTrue(url.endswith("/v1/chat/completions"))
        self.assertEqual(p["messages"], [{"role": "user", "content": "PROMPT"}])
        self.assertNotIn("prompt", p)
        self.assertNotIn("grammar", p)

    def test_chat_grammar_payload(self):
        url, p = self._build("chat+grammar")
        self.assertTrue(url.endswith("/v1/chat/completions"))
        self.assertEqual(p["messages"], [{"role": "user", "content": "PROMPT"}])
        self.assertEqual(p["grammar"], MOVE_STAY_GRAMMAR)

    def test_legacy_none_raw_payload(self):
        url, p = self._build(None)
        self.assertEqual(url, BASE_URL)
        self.assertEqual(p["prompt"], "PROMPT")
        self.assertNotIn("grammar", p)

    def test_legacy_none_chat_url_infers_messages(self):
        url, p = self._build(None, url="http://localhost:8082/v1/chat/completions")
        self.assertIn("messages", p)
        self.assertNotIn("prompt", p)

    def test_grammar_is_valid_shape(self):
        # the ws line must reach llama.cpp with literal backslash escapes
        self.assertIn(r"[ \t\n]*", MOVE_STAY_GRAMMAR)
        self.assertIn("root   ::=", MOVE_STAY_GRAMMAR)


class TestScenarioFile(unittest.TestCase):
    def setUp(self):
        self._saved = copy.deepcopy(CONTEXT_SCENARIOS)

    def tearDown(self):
        CONTEXT_SCENARIOS.clear()
        CONTEXT_SCENARIOS.update(self._saved)

    def _write(self, body):
        f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
        f.write(body)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_swap_in_place(self):
        path = self._write(
            "CONTEXT_SCENARIOS = {'only_one': {'type_a': 'a', 'type_b': 'b',"
            " 'prompt_template': 'T {agent_type} {opposite_type} {context}'}}\n")
        apply_scenario_file(path)
        self.assertEqual(list(CONTEXT_SCENARIOS.keys()), ["only_one"])
        # the swap must be visible through llm_runner's imported reference too
        self.assertIs(llm_runner.CONTEXT_SCENARIOS, CONTEXT_SCENARIOS)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            apply_scenario_file("/nonexistent/scenarios_nope.py")

    def test_missing_field_rejected(self):
        path = self._write("CONTEXT_SCENARIOS = {'x': {'type_a': 'a'}}\n")
        with self.assertRaises(ValueError):
            apply_scenario_file(path)

    def test_returns_resolved_path_for_provenance(self):
        # the return value is what config.json records: it must be the file that
        # was actually loaded, not abspath() of a bare name against a foreign cwd
        cwd = os.getcwd()
        os.chdir(tempfile.gettempdir())
        try:
            resolved = apply_scenario_file("scenarios_a3.py")
        finally:
            os.chdir(cwd)
        self.assertTrue(os.path.exists(resolved), resolved)
        self.assertEqual(os.path.realpath(resolved),
                         os.path.realpath(os.path.join(REPO_ROOT, "scenarios_a3.py")))

    def test_relative_path_resolves_from_other_cwd(self):
        # the YAML pipeline runs run_all_contexts with cwd=<run_dir>; a bare
        # 'scenarios_a3.py' must still resolve (against the repo root)
        cwd = os.getcwd()
        os.chdir(tempfile.gettempdir())
        try:
            apply_scenario_file("scenarios_a3.py")
            self.assertEqual(len(CONTEXT_SCENARIOS), 6)
        finally:
            os.chdir(cwd)

    def test_shipped_a3_file_loads(self):
        apply_scenario_file(os.path.join(REPO_ROOT, "scenarios_a3.py"))
        self.assertEqual(len(CONTEXT_SCENARIOS), 6)
        for name, entry in CONTEXT_SCENARIOS.items():
            for ph in ("{agent_type}", "{opposite_type}", "{context}"):
                self.assertIn(ph, entry["prompt_template"], f"{name} missing {ph}")
            # A3 invariants: fixed text above the grid, flipped question below
            tmpl = entry["prompt_template"]
            self.assertLess(tmpl.index("ONLY one word"), tmpl.index("{context}"),
                            f"{name}: format rule must sit above the grid")
            self.assertIn("Do you want to stay or move?",
                          tmpl.split("{context}")[1], f"{name}: flipped question after grid")


class TestYamlPassthrough(unittest.TestCase):
    def test_new_keys_become_cli_flags(self):
        sys.path.insert(0, REPO_ROOT)
        from run_llm_probability_simulation_analysis import _passthrough_map_to_cli_args
        cli = _passthrough_map_to_cli_args({
            "llm_style": "completions+grammar",
            "scenario_file": "scenarios_a3.py",
            "new": True,
            "runs": 20,
        })
        joined = " ".join(cli)
        self.assertIn("--llm-style completions+grammar", joined)
        self.assertIn("--scenario-file scenarios_a3.py", joined)
        self.assertIn("--new", joined)
        self.assertNotIn("--new True", joined)   # boolean -> bare flag


class TestAgentEndToEnd(unittest.TestCase):
    """LLMAgent must send the style-appropriate request (requests.post mocked)."""

    def _run_agent(self, style):
        import config as cfg
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["payload"] = json
            resp = mock.Mock()
            resp.status_code = 200
            resp.raise_for_status = mock.Mock()
            if "messages" in json:
                resp.json = mock.Mock(return_value={
                    "choices": [{"message": {"content": "STAY"}, "text": None}]})
            else:
                resp.json = mock.Mock(return_value={"choices": [{"text": "STAY"}]})
            return resp

        agent = llm_runner.LLMAgent(0, "baseline", "m", BASE_URL, "",
                                    temperature=0.3, llm_style=style)
        grid = [[None] * cfg.GRID_SIZE for _ in range(cfg.GRID_SIZE)]
        grid[1][1] = agent
        with mock.patch.object(llm_runner.requests, "post", side_effect=fake_post):
            decision = agent.get_llm_decision(1, 1, grid, max_retries=0)
        return decision, captured

    def test_completions_grammar_end_to_end(self):
        decision, cap = self._run_agent("completions+grammar")
        self.assertIsNone(decision)   # STAY -> stay put (None)
        self.assertTrue(cap["url"].endswith("/v1/completions"))
        self.assertIn("prompt", cap["payload"])
        self.assertEqual(cap["payload"]["grammar"], MOVE_STAY_GRAMMAR)

    def test_chat_end_to_end(self):
        decision, cap = self._run_agent("chat")
        self.assertIsNone(decision)   # STAY -> stay put (None)
        self.assertTrue(cap["url"].endswith("/v1/chat/completions"))
        self.assertIn("messages", cap["payload"])
        self.assertNotIn("grammar", cap["payload"])

    def test_legacy_none_end_to_end(self):
        decision, cap = self._run_agent(None)
        self.assertIsNone(decision)   # STAY -> stay put (None)
        self.assertEqual(cap["url"], BASE_URL)
        self.assertIn("prompt", cap["payload"])

    def test_completions_plain_end_to_end(self):
        decision, cap = self._run_agent("completions")
        self.assertIsNone(decision)   # STAY -> stay put (None)
        self.assertTrue(cap["url"].endswith("/v1/completions"))
        self.assertIn("prompt", cap["payload"])
        self.assertNotIn("grammar", cap["payload"])

    def test_chat_grammar_end_to_end(self):
        decision, cap = self._run_agent("chat+grammar")
        self.assertIsNone(decision)   # STAY -> stay put (None)
        self.assertTrue(cap["url"].endswith("/v1/chat/completions"))
        self.assertIn("messages", cap["payload"])
        self.assertEqual(cap["payload"]["grammar"], MOVE_STAY_GRAMMAR)


class TestRunAllContextsThreading(unittest.TestCase):
    """--llm-style / --scenario-file given to run_all_contexts must arrive in
    run_llm_experiment's kwargs; an invalid style must be rejected at argument
    time (before any experiment starts)."""

    def setUp(self):
        self._scenarios_snapshot = copy.deepcopy(CONTEXT_SCENARIOS)
        self._cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)  # manifest writes land in the tmp dir

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()
        CONTEXT_SCENARIOS.clear()
        CONTEXT_SCENARIOS.update(self._scenarios_snapshot)

    def _run_main(self, argv):
        import run_all_contexts as rac
        captured = {}

        def fake_run_llm_experiment(**kwargs):
            captured.update(kwargs)
            return f"experiments/{kwargs['scenario']}", [{"run_id": 0}]

        with mock.patch.object(rac, "run_llm_experiment",
                               side_effect=fake_run_llm_experiment), \
                mock.patch.object(rac, "_find_resume_candidate",
                                  return_value=(None, False)), \
                mock.patch.object(sys, "argv", argv):
            rac.main()
        return captured

    def test_style_and_scenario_file_reach_run_llm_experiment(self):
        captured = self._run_main([
            "run_all_contexts.py",
            "--runs", "2",
            "--scenarios", "baseline",
            "--llm-model", "m",
            "--llm-style", "completions+grammar",
            "--scenario-file", "scenarios_a3.py",
        ])
        self.assertEqual(captured["llm_style"], "completions+grammar")
        self.assertEqual(captured["scenario_file"], "scenarios_a3.py")
        # single scenario, 2 runs: offset 0 of total 2
        self.assertEqual(captured["progress_offset"], 0)
        self.assertEqual(captured["progress_total"], 2)

    def test_default_style_is_none_legacy(self):
        captured = self._run_main([
            "run_all_contexts.py",
            "--runs", "1",
            "--scenarios", "baseline",
            "--llm-model", "m",
        ])
        self.assertIsNone(captured["llm_style"])
        self.assertIsNone(captured["scenario_file"])

    def test_invalid_style_rejected_before_running(self):
        import run_all_contexts as rac
        ran = mock.Mock()
        with mock.patch.object(rac, "run_llm_experiment", ran), \
                mock.patch.object(sys, "argv", [
                    "run_all_contexts.py",
                    "--runs", "1",
                    "--scenarios", "baseline",
                    "--llm-model", "m",
                    "--llm-style", "telepathy",
                ]):
            with self.assertRaises(SystemExit):
                rac.main()
        ran.assert_not_called()


if __name__ == "__main__":
    unittest.main()
