import os
import pandas as pd
import pytest
import numpy as np
import json
import gzip
from unittest.mock import patch
from base_simulation import Simulation
import config as cfg


def test_analyze_results_writes_csv(tmp_path):
    # Create sample results for analysis
    results = [
        {
            'run_id': 1,
            'converged': True,
            'convergence_step': 5,
            'final_step': 5,
            'metrics_history': [
                {'step': 0, 'run_id': 1, 'clusters': 2, 'switch_rate': 0.1, 'distance': 1, 'mix_deviation': 0.0, 'share': 0.5, 'ghetto_rate': 0.0},
                {'step': 1, 'run_id': 1, 'clusters': 3, 'switch_rate': 0.2, 'distance': 2, 'mix_deviation': 0.1, 'share': 0.6, 'ghetto_rate': 0.1},
            ]
        }
    ]
    output_dir = tmp_path / "output"
    os.makedirs(output_dir)

    # Run analysis
    out_dir, out_results, out_conv = Simulation.analyze_results(results, str(output_dir), len(results))

    # Check generated files
    metrics_file = output_dir / "metrics_history.csv"
    conv_file = output_dir / "convergence_summary.csv"
    assert metrics_file.exists(), "metrics_history.csv not created"
    assert conv_file.exists(), "convergence_summary.csv not created"

    # Validate contents
    df_metrics = pd.read_csv(metrics_file)
    df_conv = pd.read_csv(conv_file)
    assert len(df_metrics) == 2
    assert len(df_conv) == 1
    assert out_dir == str(output_dir)
    assert isinstance(out_results, list)
    assert isinstance(out_conv, list)


def test_load_results_from_output_precomputed(tmp_path):
    # Prepare pre-computed CSVs
    metrics = pd.DataFrame([
        {'step': 0, 'run_id': 2, 'clusters': 1},
        {'step': 1, 'run_id': 2, 'clusters': 2},
    ])
    conv = pd.DataFrame([
        {'run_id': 2, 'converged': True, 'convergence_step': 1, 'final_step': 1}
    ])
    metrics.to_csv(tmp_path / "metrics_history.csv", index=False)
    conv.to_csv(tmp_path / "convergence_summary.csv", index=False)

    results, n_runs = Simulation.load_results_from_output(str(tmp_path))
    assert n_runs == 1
    assert len(results) == 1
    res = results[0]
    assert res['run_id'] == 2
    assert res['converged']
    assert res['convergence_step'] == 1
    assert res['final_step'] == 1
    assert isinstance(res['metrics_history'], list)
    assert len(res['metrics_history']) == 2


def test_load_results_from_output_missing_dir(tmp_path):
    # Non-existent move_logs directory should raise
    with pytest.raises(FileNotFoundError):
        Simulation.load_results_from_output(str(tmp_path / "no_logs"))


def test_load_and_analyze_results_no_results(tmp_path, monkeypatch):
    # Simulate no results found
    empty_dir = tmp_path / "empty"
    os.makedirs(empty_dir)
    monkeypatch.setattr(Simulation, 'load_results_from_output', lambda x: ([], 0))
    with pytest.raises(ValueError):
        Simulation.load_and_analyze_results(str(empty_dir))


def test_analyze_results_creates_step_statistics(tmp_path):
    # Create sample results with multiple steps
    metrics_history = [
        {'step': 0, 'run_id': 1, 'clusters': 1, 'switch_rate': 0.1, 'distance': 0.5, 'mix_deviation': 0.0, 'share': 0.4, 'ghetto_rate': 0.0},
        {'step': 1, 'run_id': 1, 'clusters': 2, 'switch_rate': 0.2, 'distance': 1.0, 'mix_deviation': 0.1, 'share': 0.5, 'ghetto_rate': 0.1}
    ]
    results = [{
        'run_id': 1,
        'converged': True,
        'convergence_step': 1,
        'final_step': 1,
        'metrics_history': metrics_history
    }]
    output_dir = tmp_path / "out"
    os.makedirs(output_dir)

    # Run analysis
    Simulation.analyze_results(results, str(output_dir), len(results))

    # Files should exist
    assert (output_dir / 'metrics_history.csv').exists(), "metrics_history.csv missing"
    assert (output_dir / 'convergence_summary.csv').exists(), "convergence_summary.csv missing"
    assert (output_dir / 'step_statistics.csv').exists(), "step_statistics.csv missing"

    # Validate step_statistics columns
    df_stats = pd.read_csv(output_dir / 'step_statistics.csv')
    expected_cols = {'step', 'clusters_mean', 'clusters_std', 'switch_rate_mean'}
    assert expected_cols.issubset(set(df_stats.columns)), "Missing expected columns in step_statistics.csv"


def test_load_and_analyze_results_integration(tmp_path, monkeypatch):
    # Prepare mock load_results_from_output to return sample results
    metrics_history = [
        {'step': 0, 'run_id': 2, 'clusters': 1, 'switch_rate': 0.1, 'distance': 0.5, 'mix_deviation': 0.0, 'share': 0.3, 'ghetto_rate': 0.0}
    ]
    sample_results = [{
        'run_id': 2,
        'converged': False,
        'convergence_step': None,
        'final_step': 0,
        'metrics_history': metrics_history
    }]
    # Monkeypatch to simulate loading existing results
    monkeypatch.setattr(Simulation, 'load_results_from_output', lambda x: (sample_results, 1))

    # Run load_and_analyze_results (should call analyze_results under the hood)
    output_dir = tmp_path / "integration"
    os.makedirs(output_dir)
    out_dir, out_results, out_conv = Simulation.load_and_analyze_results(output_dir=str(output_dir))

    # Check output files
    assert (output_dir / 'metrics_history.csv').exists(), "metrics_history.csv missing after integration"
    assert (output_dir / 'convergence_summary.csv').exists(), "convergence_summary.csv missing after integration"
    assert (output_dir / 'step_statistics.csv').exists(), "step_statistics.csv missing after integration"

    # Validate returned data
    assert out_dir == str(output_dir)
    assert out_results == sample_results
    assert isinstance(out_conv, list)


# ========== COMPREHENSIVE ADDITIONAL TESTS ==========

class MockAgent:
    """Mock agent for testing"""
    def __init__(self, type_id):
        self.type_id = type_id
        self.starting_position = None
        self.position_history = []
        self.new_position = None
        self.llm_call_count = 0
        self.llm_call_time = 0.0


class MockAgentFactory:
    """Mock agent factory for testing"""
    @staticmethod
    def create_agent(type_id):
        return MockAgent(type_id)


def mock_decision_func(agent, r, c, grid):
    """Mock decision function that returns None (stay in place)"""
    return None


def mock_move_decision_func(agent, r, c, grid):
    """Mock decision function that tries to move to (0, 0)"""
    return (0, 0)


def test_simulation_initialization():
    """Test Simulation class initialization with all parameters"""
    sim = Simulation(
        run_id=42,
        agent_factory=MockAgent,
        decision_func=mock_decision_func,
        scenario='test_scenario',
        random_seed=123
    )
    
    assert sim.run_id == 42
    assert sim.scenario == 'test_scenario'
    assert sim.step == 0
    assert not sim.converged
    assert sim.convergence_step is None
    assert sim.no_move_steps == 0
    assert sim.random_seed == 123
    assert sim.agent_factory == MockAgent
    assert sim.decision_func == mock_decision_func
    assert isinstance(sim.metrics_history, list)
    assert isinstance(sim.states, list)
    assert isinstance(sim.agent_move_log, list)
    assert sim.grid.shape == (cfg.GRID_SIZE, cfg.GRID_SIZE)


def test_simulation_initialization_default_parameters(monkeypatch):
    """Test Simulation initialization with default parameters"""
    # Mock config values for consistent testing
    monkeypatch.setattr(cfg, 'GRID_SIZE', 10)
    monkeypatch.setattr(cfg, 'NUM_TYPE_A', 20)
    monkeypatch.setattr(cfg, 'NUM_TYPE_B', 20)
    monkeypatch.setattr(cfg, 'NO_MOVE_THRESHOLD', 5)
    
    sim = Simulation(
        run_id=1,
        agent_factory=MockAgent,
        decision_func=mock_decision_func
    )
    
    assert sim.scenario == 'baseline'
    assert sim.random_seed is None
    assert sim.no_move_threshold == 5


def test_populate_grid(monkeypatch):
    """Test grid population with agents"""
    monkeypatch.setattr(cfg, 'GRID_SIZE', 5)
    monkeypatch.setattr(cfg, 'NUM_TYPE_A', 3)
    monkeypatch.setattr(cfg, 'NUM_TYPE_B', 2)
    
    sim = Simulation(run_id=1, agent_factory=MockAgent, decision_func=mock_decision_func)
    
    # Count agents by type
    type_a_count = 0
    type_b_count = 0
    total_agents = 0
    
    for r in range(cfg.GRID_SIZE):
        for c in range(cfg.GRID_SIZE):
            agent = sim.grid[r][c]
            if agent is not None:
                total_agents += 1
                if agent.type_id == 0:
                    type_a_count += 1
                elif agent.type_id == 1:
                    type_b_count += 1
                
                # Check agent initialization
                assert agent.starting_position == (r, c)
                assert agent.position_history == [(r, c)]
                assert agent.new_position is None
    
    assert total_agents == 5  # NUM_TYPE_A + NUM_TYPE_B
    assert type_a_count == 3
    assert type_b_count == 2


def test_grid_to_int():
    """Test _grid_to_int method"""
    sim = Simulation(run_id=1, agent_factory=MockAgent, decision_func=mock_decision_func)
    
    # Clear grid and add specific agents
    sim.grid = np.full((cfg.GRID_SIZE, cfg.GRID_SIZE), None)
    sim.grid[0][0] = MockAgent(0)
    sim.grid[1][1] = MockAgent(1)
    
    int_grid = sim._grid_to_int()
    
    assert int_grid[0][0] == 0
    assert int_grid[1][1] == 1
    assert int_grid[2][2] == -1  # Empty cell


def test_log_state_per_move():
    """Test logging of grid states"""
    sim = Simulation(run_id=1, agent_factory=MockAgent, decision_func=mock_decision_func)
    
    initial_states_count = len(sim.states)
    sim.log_state_per_move()
    
    assert len(sim.states) == initial_states_count + 1
    assert isinstance(sim.states[-1], np.ndarray)


def test_log_agent_move():
    """Test agent move logging"""
    sim = Simulation(run_id=1, agent_factory=MockAgent, decision_func=mock_decision_func)
    
    agent = MockAgent(0)
    initial_log_count = len(sim.agent_move_log)
    
    sim.log_agent_move(agent, 0, 0, (1, 1), True, (1, 1), 'successful_move')
    
    assert len(sim.agent_move_log) == initial_log_count + 1
    
    move_entry = sim.agent_move_log[-1]
    assert move_entry['step'] == sim.step
    assert move_entry['agent_id'] == id(agent)
    assert move_entry['type_id'] == 0
    assert move_entry['current_position'] == (0, 0)
    assert move_entry['decision'] == (1, 1)
    assert move_entry['moved'] is True
    assert move_entry['new_position'] == (1, 1)
    assert move_entry['reason'] == 'successful_move'


def test_log_agent_move_verbose(capsys):
    """Test agent move logging with verbose output"""
    sim = Simulation(run_id=1, agent_factory=MockAgent, decision_func=mock_decision_func)
    
    agent = MockAgent(0)
    sim.log_agent_move(agent, 0, 0, (1, 1), True, (1, 1), 'successful_move', verbose_move_log=True)
    
    captured = capsys.readouterr()
    assert f"Agent-{id(agent)}" in captured.out
    assert "moved from (0,0) to (1, 1)" in captured.out


def test_update_agents_no_movement():
    """Test update_agents when no agents move"""
    sim = Simulation(run_id=1, agent_factory=MockAgent, decision_func=mock_decision_func)
    
    # Mock decision function returns None (stay)
    sim.decision_func = lambda agent, r, c, grid: None
    
    moved = sim.update_agents()
    assert moved is False


def test_update_agents_successful_movement(monkeypatch):
    """Test update_agents with successful movement"""
    monkeypatch.setattr(cfg, 'GRID_SIZE', 3)
    monkeypatch.setattr(cfg, 'NUM_TYPE_A', 1)
    monkeypatch.setattr(cfg, 'NUM_TYPE_B', 0)
    
    sim = Simulation(run_id=1, agent_factory=MockAgent, decision_func=mock_decision_func)
    
    # Place agent at (0,0) and clear rest of grid
    agent = MockAgent(0)
    sim.grid = np.full((3, 3), None)
    sim.grid[0][0] = agent
    
    # Mock decision function to move to (1,1)
    sim.decision_func = lambda agent, r, c, grid: (1, 1)
    
    moved = sim.update_agents()
    
    assert moved is True
    assert sim.grid[0][0] is None
    assert sim.grid[1][1] is agent
    assert agent.new_position == (1, 1)


def test_update_agents_target_occupied(monkeypatch):
    """Test update_agents when target position is occupied"""
    monkeypatch.setattr(cfg, 'GRID_SIZE', 3)
    monkeypatch.setattr(cfg, 'NUM_TYPE_A', 2)
    monkeypatch.setattr(cfg, 'NUM_TYPE_B', 0)
    
    sim = Simulation(run_id=1, agent_factory=MockAgent, decision_func=mock_decision_func)
    
    # Place agents at (0,0) and (1,1)
    agent1 = MockAgent(0)
    agent2 = MockAgent(0)
    sim.grid = np.full((3, 3), None)
    sim.grid[0][0] = agent1
    sim.grid[1][1] = agent2
    
    # Mock decision function to try to move to occupied position
    sim.decision_func = lambda agent, r, c, grid: (1, 1)
    
    moved = sim.update_agents()
    
    assert moved is False
    assert sim.grid[0][0] is agent1  # Agent didn't move
    assert agent1.new_position == (0, 0)


def test_update_agents_out_of_bounds(monkeypatch):
    """Test update_agents with out-of-bounds move"""
    monkeypatch.setattr(cfg, 'GRID_SIZE', 3)
    monkeypatch.setattr(cfg, 'NUM_TYPE_A', 1)
    monkeypatch.setattr(cfg, 'NUM_TYPE_B', 0)
    
    sim = Simulation(run_id=1, agent_factory=MockAgent, decision_func=mock_decision_func)
    
    # Place agent at (0,0)
    agent = MockAgent(0)
    sim.grid = np.full((3, 3), None)
    sim.grid[0][0] = agent
    
    # Mock decision function to move out of bounds
    sim.decision_func = lambda agent, r, c, grid: (-1, -1)
    
    moved = sim.update_agents()
    
    assert moved is False
    assert sim.grid[0][0] is agent  # Agent didn't move
    assert agent.new_position == (0, 0)


def test_run_step():
    """Test run_step method"""
    # Mock calculate_all_metrics
    with patch('base_simulation.calculate_all_metrics') as mock_metrics:
        mock_metrics.return_value = {
            'clusters': 2,
            'switch_rate': 0.1,
            'distance': 1.0,
            'mix_deviation': 0.0,
            'share': 0.5,
            'ghetto_rate': 0.0
        }
        
        sim = Simulation(run_id=1, agent_factory=MockAgent, decision_func=mock_decision_func)
        sim.decision_func = lambda agent, r, c, grid: None  # No movement
        
        initial_step = sim.step
        converged = sim.run_step()
        
        assert sim.step == initial_step + 1
        assert converged is False
        assert len(sim.metrics_history) > 0
        assert sim.metrics_history[-1]['step'] == initial_step
        assert sim.metrics_history[-1]['run_id'] == 1
        assert sim.no_move_steps == 1


def test_run_step_convergence(monkeypatch):
    """Test convergence detection in run_step"""
    monkeypatch.setattr(cfg, 'NO_MOVE_THRESHOLD', 2)
    
    with patch('base_simulation.calculate_all_metrics') as mock_metrics:
        mock_metrics.return_value = {
            'clusters': 1,
            'switch_rate': 0.0,
            'distance': 0.0,
            'mix_deviation': 0.0,
            'share': 0.5,
            'ghetto_rate': 0.0
        }
        
        sim = Simulation(run_id=1, agent_factory=MockAgent, decision_func=mock_decision_func)
        sim.decision_func = lambda agent, r, c, grid: None  # No movement
        
        # First step - no movement
        converged = sim.run_step()
        assert converged is False
        assert sim.no_move_steps == 1
        
        # Second step - still no movement, should converge
        converged = sim.run_step()
        assert converged is True
        assert sim.converged is True
        assert sim.convergence_step == 1  # Step when convergence detected


def test_run_single_simulation(tmp_path):
    """Test complete single simulation run"""
    with patch('base_simulation.calculate_all_metrics') as mock_metrics:
        mock_metrics.return_value = {
            'clusters': 1,
            'switch_rate': 0.0,
            'distance': 0.0,
            'mix_deviation': 0.0,
            'share': 0.5,
            'ghetto_rate': 0.0
        }
        
        sim = Simulation(run_id=5, agent_factory=MockAgent, decision_func=mock_decision_func)
        sim.decision_func = lambda agent, r, c, grid: None  # No movement for quick convergence
        
        result = sim.run_single_simulation(output_dir=str(tmp_path), max_steps=10)
        
        assert result['run_id'] == 5
        assert 'converged' in result
        assert 'convergence_step' in result
        assert 'final_step' in result
        assert 'metrics_history' in result
        assert 'states_per_move' in result
        assert 'total_agent_moves' in result
        assert isinstance(result['metrics_history'], list)


def test_run_single_simulation_max_steps():
    """Test simulation stopping at max_steps"""
    with patch('base_simulation.calculate_all_metrics') as mock_metrics:
        mock_metrics.return_value = {
            'clusters': 2,
            'switch_rate': 0.1,
            'distance': 1.0,
            'mix_deviation': 0.0,
            'share': 0.5,
            'ghetto_rate': 0.0
        }
        
        sim = Simulation(run_id=1, agent_factory=MockAgent, decision_func=mock_decision_func)
        
        # Mock decision to always move (prevent convergence)
        def always_move(agent, r, c, grid):
            return ((r + 1) % cfg.GRID_SIZE, (c + 1) % cfg.GRID_SIZE)
        
        sim.decision_func = always_move
        
        result = sim.run_single_simulation(max_steps=3)
        
        assert result['final_step'] <= 3
        assert not result['converged']  # Should not converge in 3 steps


def test_save_states(tmp_path):
    """Test saving grid states"""
    sim = Simulation(run_id=1, agent_factory=MockAgent, decision_func=mock_decision_func)
    
    sim.save_states(str(tmp_path))
    
    states_file = tmp_path / "states" / "states_run_1.npz"
    assert states_file.exists()
    
    # Verify content
    loaded_states = np.load(states_file)
    assert 'states' in loaded_states


def test_save_states_no_output_dir():
    """Test save_states with None output_dir"""
    sim = Simulation(run_id=1, agent_factory=MockAgent, decision_func=mock_decision_func)
    
    # Should not raise exception
    sim.save_states(None)


def test_save_agent_move_log(tmp_path):
    """Test saving agent move log"""
    sim = Simulation(run_id=1, agent_factory=MockAgent, decision_func=mock_decision_func)
    
    # Add some move log entries
    agent = MockAgent(0)
    sim.log_agent_move(agent, 0, 0, (1, 1), True, (1, 1), 'test_move')
    
    sim.save_agent_move_log(str(tmp_path))
    
    # Check CSV file
    csv_file = tmp_path / "move_logs" / "agent_moves_run_1.csv"
    assert csv_file.exists()
    
    df = pd.read_csv(csv_file)
    assert len(df) >= 1
    assert 'step' in df.columns
    assert 'agent_id' in df.columns
    
    # Check JSON.gz file
    json_file = tmp_path / "move_logs" / "agent_moves_run_1.json.gz"
    assert json_file.exists()


def test_save_agent_move_log_no_moves(tmp_path):
    """Test save_agent_move_log with empty move log"""
    sim = Simulation(run_id=1, agent_factory=MockAgent, decision_func=mock_decision_func)
    sim.agent_move_log = []  # Empty log
    
    sim.save_agent_move_log(str(tmp_path))
    
    # Should not create files for empty log
    move_logs_dir = tmp_path / "move_logs"
    assert not move_logs_dir.exists()


def test_save_agent_move_log_no_output_dir():
    """Test save_agent_move_log with None output_dir"""
    sim = Simulation(run_id=1, agent_factory=MockAgent, decision_func=mock_decision_func)
    
    # Should not raise exception
    sim.save_agent_move_log(None)


def test_analyze_results_empty_results():
    """Test analyze_results with empty results list"""
    results = []
    
    with pytest.raises(Exception):  # Should handle empty results gracefully or raise appropriate error
        Simulation.analyze_results(results, "/tmp", 0)


def test_analyze_results_single_run(tmp_path):
    """Test analyze_results with single run"""
    results = [{
        'run_id': 1,
        'converged': True,
        'convergence_step': 5,
        'final_step': 5,
        'metrics_history': [
            {'step': 0, 'run_id': 1, 'clusters': 2, 'switch_rate': 0.1, 'distance': 1.0, 'mix_deviation': 0.0, 'share': 0.5, 'ghetto_rate': 0.0},
            {'step': 1, 'run_id': 1, 'clusters': 1, 'switch_rate': 0.0, 'distance': 0.5, 'mix_deviation': 0.1, 'share': 0.6, 'ghetto_rate': 0.1}
        ]
    }]
    
    out_dir, out_results, out_conv = Simulation.analyze_results(results, str(tmp_path), 1)
    
    assert out_dir == str(tmp_path)
    assert out_results == results
    assert len(out_conv) == 1
    
    # Check files created
    assert (tmp_path / "metrics_history.csv").exists()
    assert (tmp_path / "convergence_summary.csv").exists()
    assert (tmp_path / "step_statistics.csv").exists()


def test_analyze_results_multiple_runs(tmp_path):
    """Test analyze_results with multiple runs"""
    results = [
        {
            'run_id': 1,
            'converged': True,
            'convergence_step': 2,
            'final_step': 2,
            'metrics_history': [
                {'step': 0, 'run_id': 1, 'clusters': 3, 'switch_rate': 0.2, 'distance': 1.5, 'mix_deviation': 0.05, 'share': 0.4, 'ghetto_rate': 0.0},
                {'step': 1, 'run_id': 1, 'clusters': 2, 'switch_rate': 0.1, 'distance': 1.0, 'mix_deviation': 0.1, 'share': 0.5, 'ghetto_rate': 0.1}
            ]
        },
        {
            'run_id': 2,
            'converged': False,
            'convergence_step': None,
            'final_step': 3,
            'metrics_history': [
                {'step': 0, 'run_id': 2, 'clusters': 2, 'switch_rate': 0.15, 'distance': 1.2, 'mix_deviation': 0.02, 'share': 0.45, 'ghetto_rate': 0.05},
                {'step': 1, 'run_id': 2, 'clusters': 3, 'switch_rate': 0.25, 'distance': 1.8, 'mix_deviation': 0.15, 'share': 0.55, 'ghetto_rate': 0.15}
            ]
        }
    ]
    
    out_dir, out_results, out_conv = Simulation.analyze_results(results, str(tmp_path), 2)
    
    assert len(out_conv) == 2
    assert out_conv[0]['run_id'] == 1
    assert out_conv[1]['run_id'] == 2
    
    # Verify step statistics
    step_stats = pd.read_csv(tmp_path / "step_statistics.csv")
    assert 'step' in step_stats.columns
    assert 'clusters_mean' in step_stats.columns
    assert len(step_stats) >= 1  # At least one step


def test_load_results_from_output_missing_directory():
    """Test load_results_from_output with non-existent directory"""
    with pytest.raises(FileNotFoundError):
        Simulation.load_results_from_output("/nonexistent/path")


def test_load_results_from_output_existing_analysis(tmp_path):
    """Test load_results_from_output with existing analysis files"""
    # Create sample analysis files
    metrics_data = [
        {'step': 0, 'run_id': 1, 'clusters': 2, 'switch_rate': 0.1, 'distance': 1.0, 'mix_deviation': 0.0, 'share': 0.5, 'ghetto_rate': 0.0}
    ]
    convergence_data = [
        {'run_id': 1, 'converged': True, 'convergence_step': 5, 'final_step': 5}
    ]
    
    pd.DataFrame(metrics_data).to_csv(tmp_path / "metrics_history.csv", index=False)
    pd.DataFrame(convergence_data).to_csv(tmp_path / "convergence_summary.csv", index=False)
    
    results, n_runs = Simulation.load_results_from_output(str(tmp_path))
    
    assert n_runs == 1
    assert len(results) == 1
    assert results[0]['run_id'] == 1
    assert results[0]['converged']


def test_load_results_from_output_raw_data(tmp_path):
    """Test load_results_from_output with raw move log data"""
    # Create move_logs directory with sample data
    move_logs_dir = tmp_path / "move_logs"
    move_logs_dir.mkdir()
    
    # Create sample move log CSV
    move_data = [
        {'step': 0, 'agent_id': 123, 'type_id': 0, 'moved': True, 'grid': '[[0, -1], [-1, 1]]'},
        {'step': 1, 'agent_id': 123, 'type_id': 0, 'moved': False, 'grid': '[[0, -1], [-1, 1]]'}
    ]
    pd.DataFrame(move_data).to_csv(move_logs_dir / "agent_moves_run_1.csv", index=False)
    
    # Mock cfg.GRID_SIZE for consistent testing
    with patch.object(cfg, 'GRID_SIZE', 2):
        results, n_runs = Simulation.load_results_from_output(str(tmp_path))
    
    assert n_runs == 1
    assert len(results) == 1
    assert results[0]['run_id'] == 1


def test_load_and_analyze_results_comprehensive_no_results(tmp_path, monkeypatch):
    """Test load_and_analyze_results with no results"""
    # Mock load_results_from_output to return empty results
    monkeypatch.setattr(Simulation, 'load_results_from_output', lambda x: ([], 0))
    
    with pytest.raises(ValueError, match="No simulation results found"):
        Simulation.load_and_analyze_results(str(tmp_path))


def test_load_and_analyze_results_success(tmp_path, monkeypatch):
    """Test successful load_and_analyze_results"""
    # Mock load_results_from_output
    sample_results = [{
        'run_id': 1,
        'converged': True,
        'convergence_step': 2,
        'final_step': 2,
        'metrics_history': [
            {'step': 0, 'run_id': 1, 'clusters': 1, 'switch_rate': 0.0, 'distance': 0.0, 'mix_deviation': 0.0, 'share': 0.5, 'ghetto_rate': 0.0}
        ]
    }]
    
    monkeypatch.setattr(Simulation, 'load_results_from_output', lambda x: (sample_results, 1))
    
    out_dir, out_results, out_conv = Simulation.load_and_analyze_results(str(tmp_path))
    
    assert out_dir == str(tmp_path)
    assert out_results == sample_results
    assert len(out_conv) == 1


def test_run_single_simulation_with_progress(capsys):
    """Test run_single_simulation with progress bar"""
    with patch('base_simulation.calculate_all_metrics') as mock_metrics:
        mock_metrics.return_value = {
            'clusters': 1,
            'switch_rate': 0.0,
            'distance': 0.0,
            'mix_deviation': 0.0,
            'share': 0.5,
            'ghetto_rate': 0.0
        }
        
        sim = Simulation(run_id=1, agent_factory=MockAgent, decision_func=mock_decision_func)
        sim.decision_func = lambda agent, r, c, grid: None  # No movement for quick convergence
        
        result = sim.run_single_simulation(max_steps=5, show_progress=True)
        
        # Should show some progress information
        assert result['run_id'] == 1
