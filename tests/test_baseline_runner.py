import pytest
import baseline_runner
import config as cfg
import json
from pathlib import Path


class FixedDatetime:
    @staticmethod
    def now():
        class X:
            def strftime(self, fmt):
                return "20200101_000000"

        return X()


def test_mechanical_decision_calls_random_response():
    class DummyAgent:
        def __init__(self):
            self.called = False

        def random_response(self, r, c, grid):
            self.called = True
            return ("moved", r, c, tuple(map(tuple, grid)))

    agent = DummyAgent()
    grid = [[None]]
    result = baseline_runner.mechanical_decision(agent, 2, 3, grid)
    assert agent.called
    expected_grid = tuple(map(tuple, grid))
    assert result == ("moved", 2, 3, expected_grid)


def test_run_single_simulation_invokes_run(monkeypatch, tmp_path):
    monkeypatch.setattr(baseline_runner.BaselineSimulation, "run_single_simulation",
                        lambda self, output_dir, max_steps: "RESULT_OK")
    args = (7, {"foo": "bar"}, str(tmp_path))
    out = baseline_runner.run_single_simulation(args)
    assert out == "RESULT_OK"


def test_run_baseline_experiment_without_override(monkeypatch, tmp_path):
    """
    Test that run_baseline_experiment creates the correct output directory, writes config.json without overrides,
    calls Simulation.analyze_results with correct arguments, and returns expected results.
    """
    # Change cwd to tmp_path for filesystem operations
    monkeypatch.chdir(tmp_path)
    # Setup fake datetime for deterministic experiment_name
    monkeypatch.setattr(baseline_runner, 'datetime', FixedDatetime)
    # Monkeypatch BaselineSimulation.run_single_simulation to return dummy results
    dummy_run = {'foo': 'bar'}
    monkeypatch.setattr(baseline_runner.BaselineSimulation, 'run_single_simulation', lambda self, output_dir, max_steps: dummy_run)
    # Capture analyze_results inputs
    captured = {}
    def fake_analyze(results, out_dir, n_runs):
        captured['results'] = results
        captured['out_dir'] = out_dir
        captured['n_runs'] = n_runs
        # Return convergence data with necessary keys
        conv_data = [{'converged': True, 'convergence_step': None} for _ in range(n_runs)]
        return out_dir, ['res1'] * n_runs, conv_data
    monkeypatch.setattr(baseline_runner.Simulation, 'analyze_results', staticmethod(fake_analyze))

    # Run experiment
    out_dir, res = baseline_runner.run_baseline_experiment(n_runs=3, max_steps=50,
                                                          config_override=None, parallel=False)
    # Expected directory path
    expected_dir = tmp_path / 'experiments' / 'baseline_20200101_000000'
    assert Path(expected_dir).exists(), "Output experiment directory not created"
    # Validate config.json content
    cfgfile = expected_dir / 'config.json'
    assert cfgfile.exists(), "config.json not created"
    data = json.loads(cfgfile.read_text())
    assert data['n_runs'] == 3
    assert data['max_steps'] == 50
    assert 'overrides' not in data
    assert data['timestamp'] == '20200101_000000'
    # validate analyze_results was called correctly
    assert captured['n_runs'] == 3
    # out_dir is relative to cwd (tmp_path), so resolve it
    assert Path(tmp_path, captured['out_dir']) == expected_dir
    # Each run returns dummy_run, so captured['results'] is list of three dummy_run
    assert captured['results'] == [dummy_run] * 3
    # Validate return
    assert res == ['res1'] * 3


def test_run_baseline_experiment_with_override(monkeypatch, tmp_path):
    """
    Test that run_baseline_experiment includes overrides in config.json and returns correct results.
    """
    # Change cwd to tmp_path for filesystem operations
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(baseline_runner, 'datetime', FixedDatetime)
    # Stub run_single_simulation
    monkeypatch.setattr(baseline_runner.BaselineSimulation, 'run_single_simulation', lambda self, output_dir, max_steps: {'x':1})
    # Stub analyze_results
    monkeypatch.setattr(baseline_runner.Simulation, 'analyze_results', staticmethod(lambda results, out_dir, n_runs: (
        out_dir,
        ['A'],
        # Return convergence data with required keys
        [{'converged': False, 'convergence_step': None} for _ in range(n_runs)]
    )))

    override = {'GRID_SIZE': 10, 'NUM_TYPE_A': 20}
    out_dir, res = baseline_runner.run_baseline_experiment(n_runs=1, max_steps=5,
                                                          config_override=override, parallel=False)
    expected_dir = tmp_path / 'experiments' / 'baseline_20200101_000000'
    # Validate overrides in config
    cfgfile = expected_dir / 'config.json'
    cfgdata = json.loads(cfgfile.read_text())
    assert 'overrides' in cfgdata
    assert cfgdata['overrides'] == override
    # Validate return
    assert res == ['A']


def test_baseline_simulation_initialization_no_override():
    """Test BaselineSimulation initialization without config override"""
    sim = baseline_runner.BaselineSimulation(run_id=5)
    
    assert sim.run_id == 5
    assert hasattr(sim, 'grid')
    assert hasattr(sim, 'agent_factory')
    assert hasattr(sim, 'decision_func')


def test_baseline_simulation_initialization_with_override(monkeypatch):
    """Test BaselineSimulation initialization with config override"""
    # Store original values to restore later
    original_grid_size = cfg.GRID_SIZE
    original_num_type_a = cfg.NUM_TYPE_A
    
    try:
        override = {'GRID_SIZE': 15, 'NUM_TYPE_A': 30}
        sim = baseline_runner.BaselineSimulation(run_id=3, config_override=override)
        
        # Verify config was modified
        assert cfg.GRID_SIZE == 15
        assert cfg.NUM_TYPE_A == 30
        assert sim.run_id == 3
    finally:
        # Restore original config values
        cfg.GRID_SIZE = original_grid_size
        cfg.NUM_TYPE_A = original_num_type_a


def test_run_single_simulation_with_config_override(monkeypatch, tmp_path):
    """Test run_single_simulation applies config override correctly"""
    # Mock BaselineSimulation.run_single_simulation to capture the sim instance
    captured_sim = {}
    
    def mock_run_single_simulation(self, output_dir, max_steps):
        captured_sim['sim'] = self
        captured_sim['output_dir'] = output_dir
        captured_sim['max_steps'] = max_steps
        return {"test": "result"}
    
    monkeypatch.setattr(baseline_runner.BaselineSimulation, "run_single_simulation", mock_run_single_simulation)
    
    override = {'GRID_SIZE': 12}
    args = (7, override, str(tmp_path))
    result = baseline_runner.run_single_simulation(args)
    
    assert result == {"test": "result"}
    assert captured_sim['output_dir'] == str(tmp_path)
    assert captured_sim['max_steps'] == 1000  # Default max_steps


def test_run_baseline_experiment_parallel_execution(monkeypatch, tmp_path):
    """Test parallel execution path in run_baseline_experiment"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(baseline_runner, 'datetime', FixedDatetime)
    
    # Mock Pool execution
    execution_log = []
    
    class MockPool:
        def __init__(self, processes):
            self.processes = processes
            execution_log.append(f"Pool created with {processes} processes")
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
        
        def imap(self, func, args_list):
            execution_log.append(f"imap called with {len(list(args_list))} args")
            # Reset args_list since it was consumed
            args_list = [(i, None, "experiments/baseline_20200101_000000") for i in range(2)]
            for args in args_list:
                yield {"run_id": args[0]}
    
    monkeypatch.setattr(baseline_runner, 'Pool', MockPool)
    
    # Mock analyze_results
    def fake_analyze(results, out_dir, n_runs):
        return out_dir, results, [{'converged': True, 'convergence_step': 10} for _ in range(n_runs)]
    monkeypatch.setattr(baseline_runner.Simulation, 'analyze_results', staticmethod(fake_analyze))
    
    out_dir, res = baseline_runner.run_baseline_experiment(n_runs=2, parallel=True)
    
    assert "Pool created with 2 processes" in execution_log
    assert "imap called with 2 args" in execution_log


def test_run_baseline_experiment_single_process_execution(monkeypatch, tmp_path):
    """Test single-process execution path in run_baseline_experiment"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(baseline_runner, 'datetime', FixedDatetime)
    
    # Mock run_single_simulation
    call_log = []
    def mock_run_single_simulation(args):
        call_log.append(args)
        return {"run_id": args[0], "result": "success"}
    
    monkeypatch.setattr(baseline_runner, 'run_single_simulation', mock_run_single_simulation)
    
    # Mock analyze_results
    def fake_analyze(results, out_dir, n_runs):
        return out_dir, results, [{'converged': True, 'convergence_step': 15} for _ in range(n_runs)]
    monkeypatch.setattr(baseline_runner.Simulation, 'analyze_results', staticmethod(fake_analyze))
    
    out_dir, res = baseline_runner.run_baseline_experiment(n_runs=3, parallel=False)
    
    # Verify all runs were called sequentially
    assert len(call_log) == 3
    assert call_log[0][0] == 0  # First run_id
    assert call_log[1][0] == 1  # Second run_id  
    assert call_log[2][0] == 2  # Third run_id


def test_run_baseline_experiment_config_dict_content(monkeypatch, tmp_path):
    """Test that config.json contains all expected fields"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(baseline_runner, 'datetime', FixedDatetime)
    
    # Mock config values
    monkeypatch.setattr(cfg, 'GRID_SIZE', 25)
    monkeypatch.setattr(cfg, 'NUM_TYPE_A', 125)
    monkeypatch.setattr(cfg, 'NUM_TYPE_B', 125)
    monkeypatch.setattr(cfg, 'SIMILARITY_THRESHOLD', 0.3)
    monkeypatch.setattr(cfg, 'AGENT_SATISFACTION_THRESHOLD', 0.7)
    monkeypatch.setattr(cfg, 'NO_MOVE_THRESHOLD', 10)
    
    # Stub simulation execution
    monkeypatch.setattr(baseline_runner, 'run_single_simulation', lambda args: {"stub": "result"})
    
    # Mock analyze_results
    def fake_analyze(results, out_dir, n_runs):
        return out_dir, results, [{'converged': True, 'convergence_step': 5} for _ in range(n_runs)]
    monkeypatch.setattr(baseline_runner.Simulation, 'analyze_results', staticmethod(fake_analyze))
    
    out_dir, res = baseline_runner.run_baseline_experiment(n_runs=1, max_steps=500, parallel=False)
    
    # Verify config.json content
    config_file = tmp_path / 'experiments' / 'baseline_20200101_000000' / 'config.json'
    config_data = json.loads(config_file.read_text())
    
    assert config_data['n_runs'] == 1
    assert config_data['max_steps'] == 500
    assert config_data['grid_size'] == 25
    assert config_data['num_type_a'] == 125
    assert config_data['num_type_b'] == 125
    assert config_data['similarity_threshold'] == 0.3
    assert config_data['agent_satisfaction_threshold'] == 0.7
    assert config_data['no_move_threshold'] == 10
    assert config_data['timestamp'] == '20200101_000000'


def test_run_baseline_experiment_cpu_count_limiting(monkeypatch, tmp_path):
    """Test that parallel execution respects CPU count limits"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(baseline_runner, 'datetime', FixedDatetime)
    
    # Mock cpu_count to return a small number
    monkeypatch.setattr(baseline_runner, 'cpu_count', lambda: 2)
    
    # Mock Pool to capture process count
    process_counts = []
    
    class MockPool:
        def __init__(self, processes):
            process_counts.append(processes)
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
            
        def imap(self, func, args_list):
            # Convert to list to get length, then recreate
            args_list = list(args_list)
            for args in args_list:
                yield {"stub": "result"}
    
    monkeypatch.setattr(baseline_runner, 'Pool', MockPool)
    
    # Mock analyze_results
    def fake_analyze(results, out_dir, n_runs):
        return out_dir, results, [{'converged': False, 'convergence_step': None} for _ in range(n_runs)]
    monkeypatch.setattr(baseline_runner.Simulation, 'analyze_results', staticmethod(fake_analyze))
    
    # Test with more runs than CPU count
    baseline_runner.run_baseline_experiment(n_runs=10, parallel=True)
    
    # Should use min(cpu_count, n_runs) = min(2, 10) = 2
    assert process_counts[0] == 2
    
    # Test with fewer runs than CPU count  
    baseline_runner.run_baseline_experiment(n_runs=1, parallel=True)
    
    # Should use min(cpu_count, n_runs) = min(2, 1) = 1
    assert process_counts[1] == 1


def test_run_baseline_experiment_directory_creation_error(monkeypatch, tmp_path):
    """Test handling when directory creation fails"""
    monkeypatch.setattr(baseline_runner, 'datetime', FixedDatetime)
    
    # Mock os.makedirs to raise an exception
    def mock_makedirs(path, exist_ok=False):
        raise PermissionError("Cannot create directory")
    
    monkeypatch.setattr(baseline_runner.os, 'makedirs', mock_makedirs)
    
    # Should raise the PermissionError
    with pytest.raises(PermissionError):
        baseline_runner.run_baseline_experiment(n_runs=1)


def test_run_baseline_experiment_config_write_error(monkeypatch, tmp_path):
    """Test handling when config.json write fails"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(baseline_runner, 'datetime', FixedDatetime)
    
    # Create the directory but make config write fail
    def mock_open(*args, **kwargs):
        if 'config.json' in str(args[0]):
            raise IOError("Cannot write config file")
        return open(*args, **kwargs)
    
    import builtins
    monkeypatch.setattr(builtins, 'open', mock_open)
    
    with pytest.raises(IOError):
        baseline_runner.run_baseline_experiment(n_runs=1)


def test_mechanical_decision_with_different_grid_sizes():
    """Test mechanical_decision with various grid configurations"""
    class TestAgent:
        def __init__(self):
            self.calls = []
        
        def random_response(self, r, c, grid):
            self.calls.append((r, c, len(grid), len(grid[0]) if grid else 0))
            return ("response", r, c, grid)
    
    agent = TestAgent()
    
    # Test with empty grid
    result = baseline_runner.mechanical_decision(agent, 0, 0, [])
    assert result == ("response", 0, 0, [])
    assert agent.calls[-1] == (0, 0, 0, 0)
    
    # Test with 1x1 grid
    grid_1x1 = [[None]]
    result = baseline_runner.mechanical_decision(agent, 0, 0, grid_1x1)
    assert result == ("response", 0, 0, grid_1x1)
    assert agent.calls[-1] == (0, 0, 1, 1)
    
    # Test with larger grid
    grid_3x3 = [[None, None, None], [None, None, None], [None, None, None]]
    result = baseline_runner.mechanical_decision(agent, 1, 2, grid_3x3)
    assert result == ("response", 1, 2, grid_3x3)
    assert agent.calls[-1] == (1, 2, 3, 3)


def test_run_baseline_experiment_analyze_results_integration(monkeypatch, tmp_path):
    """Test that analyze_results is called with correct parameters and processes results properly"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(baseline_runner, 'datetime', FixedDatetime)
    
    # Mock simulation results
    mock_results = [
        {"run_id": 0, "final_step": 50, "converged": True},
        {"run_id": 1, "final_step": 100, "converged": False},
        {"run_id": 2, "final_step": 75, "converged": True}
    ]
    
    def mock_run_single_simulation(args):
        run_id = args[0]
        return mock_results[run_id]
    
    monkeypatch.setattr(baseline_runner, 'run_single_simulation', mock_run_single_simulation)
    
    # Capture analyze_results call
    analyze_calls = []
    def mock_analyze_results(results, out_dir, n_runs):
        analyze_calls.append({
            'results': results,
            'out_dir': out_dir,
            'n_runs': n_runs
        })
        # Return expected format
        convergence_data = [
            {'converged': True, 'convergence_step': 50, 'final_step': 50},
            {'converged': False, 'convergence_step': None, 'final_step': 100},
            {'converged': True, 'convergence_step': 75, 'final_step': 75}
        ]
        return out_dir, results, convergence_data
    
    monkeypatch.setattr(baseline_runner.Simulation, 'analyze_results', staticmethod(mock_analyze_results))
    
    out_dir, results = baseline_runner.run_baseline_experiment(n_runs=3, parallel=False)
    
    # Verify analyze_results was called correctly
    assert len(analyze_calls) == 1
    call = analyze_calls[0]
    assert call['results'] == mock_results
    assert call['n_runs'] == 3
    assert 'baseline_20200101_000000' in call['out_dir']
    
    # Verify return values
    assert results == mock_results
    assert 'baseline_20200101_000000' in out_dir


def test_run_baseline_experiment_empty_results_handling(monkeypatch, tmp_path):
    """Test handling of empty or None results from simulations"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(baseline_runner, 'datetime', FixedDatetime)
    
    # Mock simulation that returns None/empty results
    def mock_run_single_simulation(args):
        return None
    
    monkeypatch.setattr(baseline_runner, 'run_single_simulation', mock_run_single_simulation)
    
    # Mock analyze_results to handle None results
    def mock_analyze_results(results, out_dir, n_runs):
        # Should receive list of None values
        assert all(r is None for r in results)
        assert len(results) == n_runs
        return out_dir, results, [{'converged': False, 'convergence_step': None} for _ in range(n_runs)]
    
    monkeypatch.setattr(baseline_runner.Simulation, 'analyze_results', staticmethod(mock_analyze_results))
    
    # Should not raise exception with None results
    out_dir, results = baseline_runner.run_baseline_experiment(n_runs=2, parallel=False)
    
    assert results == [None, None]


def test_baseline_simulation_inheritance():
    """Test that BaselineSimulation properly inherits from Simulation"""
    sim = baseline_runner.BaselineSimulation(run_id=1)
    
    # Should inherit from Simulation
    assert isinstance(sim, baseline_runner.Simulation)
    
    # Should have the correct agent_factory and decision_func
    assert sim.agent_factory == baseline_runner.Agent
    assert sim.decision_func == baseline_runner.mechanical_decision


def test_run_baseline_experiment_default_parameters(monkeypatch, tmp_path):
    """Test run_baseline_experiment with all default parameters"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(baseline_runner, 'datetime', FixedDatetime)
    
    # Mock to avoid actual long-running simulation
    def mock_run_single_simulation(args):
        return {"run_id": args[0]}
    monkeypatch.setattr(baseline_runner, 'run_single_simulation', mock_run_single_simulation)
    
    def mock_analyze_results(results, out_dir, n_runs):
        return out_dir, results[:5], [{'converged': True, 'convergence_step': 10} for _ in range(5)]  # Return only first 5 to speed up test
    monkeypatch.setattr(baseline_runner.Simulation, 'analyze_results', staticmethod(mock_analyze_results))
    
    # Call with defaults (but limit n_runs for test speed and use single-process to avoid pickle issues)
    out_dir, results = baseline_runner.run_baseline_experiment(n_runs=5, parallel=False)
    
    # Verify config contains default values
    config_file = Path(tmp_path) / 'experiments' / 'baseline_20200101_000000' / 'config.json'
    config_data = json.loads(config_file.read_text())
    
    assert config_data['n_runs'] == 5
    assert config_data['max_steps'] == 1000  # Default value
    assert 'overrides' not in config_data


