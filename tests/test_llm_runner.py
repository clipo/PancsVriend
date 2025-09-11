import json
import os
import requests
import pytest
import time
import llm_runner
import config as cfg
from types import SimpleNamespace
from context_scenarios import CONTEXT_SCENARIOS

class DummyResponse:
    def __init__(self, status_code, data=None, text=""):
        self.status_code = status_code
        self._data = data or {}
        self.text = text

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def test_check_llm_connection_success(monkeypatch):
    payload_captured = {}
    def fake_post(url, headers, json, timeout):
        payload_captured.update(dict(url=url, headers=headers, json=json, timeout=timeout))
        data = {"choices":[{"message":{"content":"OK"}}]}
        return DummyResponse(200, data)
    monkeypatch.setattr(requests, "post", fake_post)
    ok = llm_runner.check_llm_connection("m", "u", "k", timeout=3)
    assert ok is True
    # ensure we invoked requests.post
    assert payload_captured["timeout"] == 3
    assert payload_captured["json"]["messages"][0]["content"].startswith("Respond")


def test_check_llm_connection_http_error(monkeypatch):
    def fake_post(url, headers, json, timeout):
        return DummyResponse(404, {}, text="not found")
    monkeypatch.setattr(requests, "post", fake_post)
    assert llm_runner.check_llm_connection() is False


def test_check_llm_connection_invalid_json(monkeypatch):
    def fake_post(url, headers, json, timeout):
        # missing "choices" key
        return DummyResponse(200, {"foo": []})
    monkeypatch.setattr(requests, "post", fake_post)
    assert llm_runner.check_llm_connection() is False


def test_check_llm_connection_timeout(monkeypatch):
    def fake_post(*args, **kwargs):
        raise requests.exceptions.Timeout()
    monkeypatch.setattr(requests, "post", fake_post)
    assert llm_runner.check_llm_connection() is False


def test_check_llm_connection_conn_error(monkeypatch):
    def fake_post(*args, **kwargs):
        raise requests.exceptions.ConnectionError("fail")
    monkeypatch.setattr(requests, "post", fake_post)
    assert llm_runner.check_llm_connection() is False


def test_check_llm_connection_general_exception(monkeypatch):
    """Test general exception handling"""
    def fake_post(*args, **kwargs):
        raise ValueError("something went wrong")
    monkeypatch.setattr(requests, "post", fake_post)
    assert llm_runner.check_llm_connection() is False


def test_check_llm_connection_with_timing(monkeypatch):
    """Test connection check with timing mock"""
    def fake_post(url, headers, json, timeout):
        data = {"choices": [{"message": {"content": "OK"}}]}
        return DummyResponse(200, data)
    
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(time, "time", lambda: 1.0)
    
    result = llm_runner.check_llm_connection("model", "url", "key")
    assert result is True


def test_get_context_grid(monkeypatch):
    # force GRID_SIZE to 3
    monkeypatch.setattr(cfg, "GRID_SIZE", 3)
    # dummy agents
    class A(SimpleNamespace):
        pass
    # build grid
    grid = [
        [A(type_id=0), A(type_id=1), None],
        [None, A(type_id=0), A(type_id=1)],
        [None, None, None]
    ]
    agent = llm_runner.LLMAgent(0, scenario='baseline')
    # override context_scenarios so .prompt_template isn't used here
    agent.context_info = CONTEXT_SCENARIOS['baseline']
    s = agent.get_context_grid(1,1, grid)
    # lines should be 3 rows of 3 symbols
    rows = s.splitlines()
    assert len(rows) == 3
    # center is X
    assert rows[1].split()[1] == "X"
    # check one same-type neighbor and one opposite
    # top-left was type_id=0 → 'S'
    assert rows[0].split()[0] == "S"
    # top-middle was type_id=1 → 'O'
    assert rows[0].split()[1] == "O"
    # out of bounds replaced by '#' when GRID_SIZE small
    # but our grid covers all 3x3 so no '#'


def test_get_context_grid_out_of_bounds(monkeypatch):
    """Test context grid with out-of-bounds positions"""
    monkeypatch.setattr(cfg, "GRID_SIZE", 2)
    
    grid = [[None, None], [None, None]]
    agent = llm_runner.LLMAgent(0, scenario='baseline')
    context_str = agent.get_context_grid(0, 0, grid)  # Corner position
    
    rows = context_str.splitlines()
    # Should contain # for out-of-bounds positions
    context_flat = " ".join(rows)
    assert "#" in context_flat


def test_llm_agent_initialization():
    """Test LLMAgent initialization"""
    agent = llm_runner.LLMAgent(0, scenario='baseline', llm_model='test-model', 
                               llm_url='test-url', llm_api_key='test-key', run_id=1, step=2)
    
    assert agent.type_id == 0
    assert agent.scenario == 'baseline'
    assert agent.llm_model == 'test-model'
    assert agent.llm_url == 'test-url'
    assert agent.llm_api_key == 'test-key'
    assert agent.run_id == 1
    assert agent.step == 2
    assert agent.llm_call_count == 0
    assert agent.llm_call_time == 0.0


def test_get_llm_decision_move_response(monkeypatch):
    """Test LLM decision parsing for MOVE response"""
    monkeypatch.setattr(cfg, "GRID_SIZE", 3)
    
    # Mock successful HTTP response with MOVE decision
    call_count = [0]
    def fake_post(url, headers, json, timeout):
        call_count[0] += 1
        data = {"choices": [{"message": {"content": "MOVE to a better location"}}]}
        return DummyResponse(200, data)
    
    # Mock time.time to return different values for start/end timing
    time_values = [1.0, 1.5]  # 0.5 second difference
    time_index = [0]
    def mock_time():
        val = time_values[min(time_index[0], len(time_values) - 1)]
        time_index[0] += 1
        return val
    
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(time, "time", mock_time)
    
    # Create grid with some empty spaces
    grid = [[None, None, None], [None, None, None], [None, None, None]]
    
    agent = llm_runner.LLMAgent(0, scenario='baseline', run_id=1, step=2)
    
    # Mock random.choice to return predictable position
    import random
    original_choice = random.choice
    def mock_choice(seq):
        return (2, 2) if seq else original_choice(seq)
    monkeypatch.setattr(random, "choice", mock_choice)
    
    result = agent.get_llm_decision(1, 1, grid)
    
    assert result == (2, 2)
    assert agent.llm_call_count == 1
    assert agent.llm_call_time > 0


def test_get_llm_decision_stay_response(monkeypatch):
    """Test LLM decision parsing for STAY response"""
    monkeypatch.setattr(cfg, "GRID_SIZE", 2)  # Match grid size
    
    def fake_post(url, headers, json, timeout):
        data = {"choices": [{"message": {"content": "STAY in current position"}}]}
        return DummyResponse(200, data)
    
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(time, "time", lambda: 1.0)
    
    grid = [[None, None], [None, None]]
    agent = llm_runner.LLMAgent(0, scenario='baseline', run_id=1, step=2)
    
    result = agent.get_llm_decision(1, 1, grid)
    assert result is None  # STAY should return None


def test_get_llm_decision_unparseable_response(monkeypatch):
    """Test LLM decision with unparseable response"""
    monkeypatch.setattr(cfg, "GRID_SIZE", 2)  # Match grid size
    
    def fake_post(url, headers, json, timeout):
        data = {"choices": [{"message": {"content": "I don't understand"}}]}
        return DummyResponse(200, data)
    
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(time, "time", lambda: 1.0)
    
    grid = [[None, None], [None, None]]
    agent = llm_runner.LLMAgent(0, scenario='baseline', run_id=1, step=2)
    
    result = agent.get_llm_decision(1, 1, grid)
    assert result is None  # Unparseable should default to STAY


def test_get_llm_decision_timeout_retry(monkeypatch):
    """Test retry logic on timeout"""
    monkeypatch.setattr(cfg, "GRID_SIZE", 2)  # Match grid size
    
    call_count = [0]
    
    def fake_post(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 2:
            raise requests.exceptions.Timeout()
        # Third attempt succeeds
        data = {"choices": [{"message": {"content": "STAY"}}]}
        return DummyResponse(200, data)
    
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(time, "time", lambda: 1.0)
    monkeypatch.setattr(time, "sleep", lambda x: None)  # Skip sleep
    
    grid = [[None, None], [None, None]]
    agent = llm_runner.LLMAgent(0, scenario='baseline', run_id=1, step=2)
    
    result = agent.get_llm_decision(1, 1, grid, max_retries=5)
    assert result is None
    assert call_count[0] == 3  # Should have retried twice


def test_get_llm_decision_max_retries_exceeded(monkeypatch):
    """Test exception when max retries exceeded"""
    monkeypatch.setattr(cfg, "GRID_SIZE", 2)  # Match grid size
    
    def fake_post(*args, **kwargs):
        raise requests.exceptions.Timeout()
    
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(time, "sleep", lambda x: None)  # Skip sleep
    
    grid = [[None, None], [None, None]]
    agent = llm_runner.LLMAgent(0, scenario='baseline', run_id=1, step=2)
    
    with pytest.raises(Exception) as excinfo:
        agent.get_llm_decision(1, 1, grid, max_retries=2)
    
    assert "LLM timeout after 2 retries" in str(excinfo.value)


def test_llm_simulation_initialization():
    """Test LLMSimulation initialization"""
    # Use a valid scenario from CONTEXT_SCENARIOS instead of 'test'
    sim = llm_runner.LLMSimulation(run_id=5, scenario='baseline', llm_model='model', 
                                  llm_url='url', llm_api_key='key', random_seed=42)
    
    assert sim.run_id == 5
    assert sim.scenario == 'baseline'
    assert sim.llm_model == 'model'
    assert sim.llm_url == 'url'
    assert sim.llm_api_key == 'key'
    assert sim.total_llm_calls == 0
    assert sim.total_llm_time == 0.0


def test_llm_simulation_create_llm_agent():
    """Test _create_llm_agent method"""
    sim = llm_runner.LLMSimulation(run_id=1, scenario='baseline')
    agent = sim._create_llm_agent(0)
    
    assert isinstance(agent, llm_runner.LLMAgent)
    assert agent.type_id == 0
    assert agent.scenario == 'baseline'


def test_llm_decision_function_forwards():
    class Dummy:
        def __init__(self):
            self.called = False
        def get_llm_decision(self, r, c, g):
            self.called = True
            return ("ans", r, c, g)
    dummy = Dummy()
    grid = [["g"]]
    out = llm_runner.llm_decision_function(dummy, 2, 3, grid)
    assert dummy.called
    assert out == ("ans", 2, 3, grid)


def test_run_single_simulation_augment(monkeypatch):
    # stub LLMSimulation.run_single_simulation
    def fake_run(self, output_dir, max_steps):
        return {"foo": "bar"}
    monkeypatch.setattr(llm_runner.LLMSimulation, "run_single_simulation", fake_run)
    
    # Use valid scenario name from CONTEXT_SCENARIOS
    args = (5, "baseline", "m", "u", "k", "outdir")
    res = llm_runner.run_single_simulation(args)
    # base result plus added keys
    assert res["foo"] == "bar"
    assert res["scenario"] == "baseline"
    assert res["llm_call_count"] == 0
    assert res["avg_llm_call_time"] == 0.0


def test_check_existing_experiment_no_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exists, cnt, outdir, ids = llm_runner.check_existing_experiment("nope")
    assert exists is False
    assert cnt == 0
    assert isinstance(ids, set)


def test_check_existing_experiment_with_npz(tmp_path, monkeypatch):
    # create experiments/foo/states/states_run_2.npz
    base = tmp_path / "experiments" / "foo" / "states"
    base.mkdir(parents=True)
    f = base / "states_run_2.npz"
    f.write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    exists, cnt, outdir, ids = llm_runner.check_existing_experiment("foo")
    assert exists is True
    assert cnt == 1
    assert ids == {2}
    assert outdir == "experiments/foo"


def test_list_available_experiments(tmp_path, monkeypatch):
    # set up a fake experiment dir and config
    expdir = tmp_path / "experiments" / "bar"
    expdir.mkdir(parents=True)
    cfgfile = expdir / "config.json"
    cfgfile.write_text(json.dumps({"n_runs":5}))
    # stub check_existing_experiment
    def fake_chk(name):
        assert name == "bar"
        return True, 2, "experiments/bar", {0,1}
    monkeypatch.setattr(llm_runner, "check_existing_experiment", fake_chk)
    monkeypatch.chdir(tmp_path)
    lst = llm_runner.list_available_experiments()
    assert isinstance(lst, list) and lst, "should return non-empty list"
    rec = lst[0]
    assert rec["name"] == "bar"
    assert rec["completed"] == 2
    assert rec["total"] == 5
    assert rec["path"] == "experiments/bar"


def test_get_llm_decision_no_empty_spaces(monkeypatch):
    """Test MOVE decision when no empty spaces available"""
    monkeypatch.setattr(cfg, "GRID_SIZE", 2)
    
    def fake_post(url, headers, json, timeout):
        data = {"choices": [{"message": {"content": "MOVE"}}]}
        return DummyResponse(200, data)
    
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(time, "time", lambda: 1.0)
    
    # Create dummy agent
    class DummyAgent:
        def __init__(self, type_id):
            self.type_id = type_id
    
    # Grid with no empty spaces
    grid = [[DummyAgent(0), DummyAgent(1)], [DummyAgent(0), DummyAgent(1)]]
    agent = llm_runner.LLMAgent(0, scenario='baseline', run_id=1, step=2)
    
    result = agent.get_llm_decision(0, 0, grid)
    assert result is None  # Should stay when no empty spaces


def test_get_llm_decision_http_error_retry(monkeypatch):
    """Test retry logic on HTTP error"""
    monkeypatch.setattr(cfg, "GRID_SIZE", 2)
    
    call_count = [0]
    def fake_post(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 1:
            raise requests.HTTPError("Server error")
        # Second attempt succeeds
        data = {"choices": [{"message": {"content": "STAY"}}]}
        return DummyResponse(200, data)
    
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(time, "time", lambda: 1.0)
    monkeypatch.setattr(time, "sleep", lambda x: None)  # Skip sleep
    
    grid = [[None, None], [None, None]]
    agent = llm_runner.LLMAgent(0, scenario='baseline', run_id=1, step=2)
    
    result = agent.get_llm_decision(1, 1, grid, max_retries=3)
    assert result is None
    assert call_count[0] == 2  # Should have retried once


def test_get_llm_decision_malformed_json_response(monkeypatch):
    """Test handling of malformed JSON response"""
    monkeypatch.setattr(cfg, "GRID_SIZE", 2)
    
    def fake_post(url, headers, json, timeout):
        # Return response that will cause JSON parsing error
        response = DummyResponse(200)
        response.json = lambda: {"invalid": "structure"}  # Missing "choices"
        return response
    
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(time, "time", lambda: 1.0)
    monkeypatch.setattr(time, "sleep", lambda x: None)
    
    grid = [[None, None], [None, None]]
    agent = llm_runner.LLMAgent(0, scenario='baseline', run_id=1, step=2)
    
    with pytest.raises(Exception) as excinfo:
        agent.get_llm_decision(1, 1, grid, max_retries=2)
    
    # Should fail with KeyError or similar after exhausting retries
    assert "after 2 retries" in str(excinfo.value)


def test_check_existing_experiment_json_files(tmp_path, monkeypatch):
    """Test check_existing_experiment with JSON.gz pattern"""
    monkeypatch.chdir(tmp_path)
    
    exp_dir = tmp_path / "experiments" / "json_exp"
    exp_dir.mkdir(parents=True)
    
    # Create files matching the run_*.json.gz pattern
    (exp_dir / "run_0.json.gz").write_bytes(b"fake_data")
    (exp_dir / "run_3.json.gz").write_bytes(b"fake_data")
    
    exists, count, output_dir, ids = llm_runner.check_existing_experiment("json_exp")
    
    assert exists is True
    assert count == 2
    assert ids == {0, 3}


def test_list_available_experiments_missing_config(tmp_path, monkeypatch):
    """Test list_available_experiments with missing config file"""
    monkeypatch.chdir(tmp_path)
    
    exp_dir = tmp_path / "experiments" / "no_config_exp"
    exp_dir.mkdir(parents=True)
    # No config.json file created
    
    def mock_check(name):
        return True, 1, f"experiments/{name}", {0}
    
    monkeypatch.setattr(llm_runner, "check_existing_experiment", mock_check)
    
    result = llm_runner.list_available_experiments()
    
    assert len(result) == 1
    assert result[0]['name'] == 'no_config_exp'
    assert result[0]['total'] == 'unknown'  # Should handle missing config gracefully


def test_llm_agent_type_assignment():
    """Test that agent types are assigned correctly based on type_id"""
    # Test type_id = 0 (should get type_a)
    agent_0 = llm_runner.LLMAgent(0, scenario='baseline')
    assert agent_0.agent_type == CONTEXT_SCENARIOS['baseline']['type_a']
    assert agent_0.opposite_type == CONTEXT_SCENARIOS['baseline']['type_b']
    
    # Test type_id = 1 (should get type_b)
    agent_1 = llm_runner.LLMAgent(1, scenario='baseline')
    assert agent_1.agent_type == CONTEXT_SCENARIOS['baseline']['type_b']
    assert agent_1.opposite_type == CONTEXT_SCENARIOS['baseline']['type_a']


def test_llm_agent_with_config_defaults(monkeypatch):
    """Test LLMAgent uses config defaults when parameters not provided"""
    # Mock config values
    monkeypatch.setattr(cfg, "OLLAMA_MODEL", "default-model")
    monkeypatch.setattr(cfg, "OLLAMA_URL", "default-url")
    monkeypatch.setattr(cfg, "OLLAMA_API_KEY", "default-key")
    
    agent = llm_runner.LLMAgent(0, scenario='baseline')
    
    assert agent.llm_model == "default-model"
    assert agent.llm_url == "default-url"
    assert agent.llm_api_key == "default-key"


def test_context_grid_boundary_conditions(monkeypatch):
    """Test context grid generation at various boundary positions"""
    monkeypatch.setattr(cfg, "GRID_SIZE", 3)
    
    class DummyAgent:
        def __init__(self, type_id):
            self.type_id = type_id
    
    grid = [[DummyAgent(0), None, None], 
            [None, DummyAgent(1), None], 
            [None, None, DummyAgent(0)]]
    
    agent = llm_runner.LLMAgent(0, scenario='baseline')
    
    # Test all corner and edge positions
    positions_to_test = [(0, 0), (0, 2), (2, 0), (2, 2), (1, 0), (0, 1)]
    
    for r, c in positions_to_test:
        context_str = agent.get_context_grid(r, c, grid)
        rows = context_str.splitlines()
        
        # Should always return 3x3 grid
        assert len(rows) == 3
        assert all(len(row.split()) == 3 for row in rows)
        
        # Center should always be X
        assert rows[1].split()[1] == "X"
