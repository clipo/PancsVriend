# Base simulation class for Schelling model variants
import numpy as np
import config as cfg
from Metrics import as_int_grid, calculate_all_metrics
import os
import gzip
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from tqdm import tqdm

import run_files


class _MetricsMockAgent:
    def __init__(self, type_id):
        self.type_id = type_id


def convergence_from_step_moves(step_moves, threshold):
    """(converged, first_no_move_step, detected_step) from a {step: n_moves} map.

    THE convergence definition, in one place. A run has converged when its
    LAST `threshold` steps all had zero agent movements; the reported
    convergence step is the FIRST step of that window, and `detected_step` is
    the step at which the criterion was met (first + threshold - 1), which is
    also the last step simulated.

    This used to be written out twice with one meaning (first-of-window, in
    _load_single_run_result and llm_runner._analyze_run_status) and once with
    the other (last-of-window, in Simulation.run_step), so
    convergence_summary.csv carried both conventions depending on which path
    produced the row — 722 rows on disk with final_step - convergence_step == 0
    against 1593 with == 4. Everything routes through here now (2026-09-01).
    """
    if not step_moves or threshold < 1:
        return False, None, None
    steps = sorted(step_moves)
    window = steps[-threshold:]
    if len(window) < threshold:
        return False, None, None
    if any(step_moves[step] != 0 for step in window):
        return False, None, None
    return True, int(window[0]), int(window[-1])


def _load_single_run_result(task):
    """Rebuild one run's convergence + per-step metrics from its saved files.

    Both come through run_files, which hands back the same per-step tables
    for the per-step format and for full (per-move) logs, so this function
    does not care which one the run wrote.

    The grid's SHAPE comes from the data too. It used to come from the ambient
    cfg.GRID_SIZE, which is silently wrong whenever the config in force differs
    from the one a run was recorded under: a 20x20 config re-analysing a 10x10
    run padded 300 cells with None and returned plausible-looking metrics for a
    grid that was three quarters empty, with no error raised.
    """
    run_id, output_dir, threshold = task

    step_log = run_files.load_step_log(output_dir, run_id)
    step_moves = run_files.step_moves(step_log)
    max_step = max(step_moves) if step_moves else 0
    converged, convergence_step, _ = convergence_from_step_moves(step_moves, threshold)

    metrics_history = []
    loaded = run_files.load_step_frames(output_dir, run_id)
    if loaded is not None:
        for step, grid_array in zip(*loaded):
            try:
                step_metrics = calculate_all_metrics(np.asarray(grid_array))
                step_metrics['step'] = step
                step_metrics['run_id'] = run_id
                metrics_history.append(step_metrics)

            except Exception as e:
                print(f"Warning: Could not reconstruct metrics for step {step}, run {run_id}: {e}")
                metrics_history.append({
                    'step': step,
                    'run_id': run_id,
                    'clusters': 0,
                    'switch_rate': 0,
                    'distance': 0,
                    'mix_deviation': 0,
                    'share': 0.5,
                    'ghetto_rate': 0
                })

    if not metrics_history:
        metrics_history = [{
            'step': 0,
            'run_id': run_id,
            'clusters': 0,
            'switch_rate': 0,
            'distance': 0,
            'mix_deviation': 0,
            'share': 0.5,
            'ghetto_rate': 0
        }]

    return {
        'run_id': run_id,
        'converged': converged,
        'convergence_step': convergence_step,
        'final_step': max_step,
        'metrics_history': metrics_history
    }


class Simulation:
    def __init__(self, run_id, agent_factory, decision_func, scenario='baseline', random_seed=None,
                 initial_int_grid=None, initial_step=None, initial_no_move_steps=None,
                 full_move_log=False):
        self.run_id = run_id
        self.scenario = scenario
        self.grid = np.full((cfg.GRID_SIZE, cfg.GRID_SIZE), None)
        self.step = 0
        self.converged = False
        self.convergence_step = None
        self.no_move_steps = 0
        self.no_move_threshold = cfg.NO_MOVE_THRESHOLD
        self.metrics_history = []
        self.agent_factory = agent_factory
        self.decision_func = decision_func
        self.random_seed = random_seed
        # What the run records (see run_files). Per-step by default: one row
        # of decision counts and one grid frame per step. Full: one record and
        # one frame per agent decision — needed only when records carry LLM
        # replies (live-LLM runs pass full_move_log=True), or to regenerate
        # per-move detail for a deterministic run via FULL_MOVE_LOG=1.
        self.full_move_log = bool(full_move_log) or (
            os.environ.get('FULL_MOVE_LOG', '').lower() in ('true', '1', 'yes'))
        self.states = []
        self.agent_move_log = []       # full format: per-move records
        self.step_log = []             # per-step format: run_files.STEP_LOG_COLUMNS rows

        if self.random_seed is None:
            np.random.seed(None)
        else:
            # Seed BOTH RNG streams the simulation actually draws from:
            # numpy (grid population, agent activation order) and stdlib
            # random (value-function/mechanical decisions, destination picks
            # in llm_runner). Before 2026-08-21 a passed seed did nothing —
            # this branch is what makes run k of two batches pairable.
            import random as _random
            np.random.seed(self.random_seed)
            _random.seed(self.random_seed)

        # Initialize grid either randomly or from a provided int grid (resume)
        if initial_int_grid is not None:
            self.populate_from_int_grid(initial_int_grid)
            # Allow resuming at a specified step (next step index)
            if initial_step is not None:
                try:
                    self.step = int(initial_step)
                except Exception:
                    pass
        else:
            self.populate_grid()
        # Frame 0 is the initial grid in both formats.
        self.states.append(self._grid_to_int())
        if self.full_move_log:
            # Log a dummy "move" to record their initial state
            self.log_agent_move(None, None, None, None, False, None, 'initial_state', verbose_move_log=False)

        if initial_no_move_steps is not None:
            try:
                parsed_streak = int(initial_no_move_steps)
                if parsed_streak >= 0:
                    self.no_move_steps = parsed_streak
            except Exception:
                pass

    def populate_from_int_grid(self, int_grid):
        """Populate grid from a 2D numpy/list of ints (-1 empty, 0/1 type ids)."""
        arr = np.array(int_grid)
        assert arr.shape == (cfg.GRID_SIZE, cfg.GRID_SIZE), "initial_int_grid shape mismatch"
        agent_id = 0
        for r in range(cfg.GRID_SIZE):
            for c in range(cfg.GRID_SIZE):
                t = int(arr[r, c])
                if t >= 0:
                    agent = self.agent_factory(t)
                    self.grid[r][c] = agent
                    agent.agent_id = agent_id
                    agent_id += 1
                    agent.starting_position = (r, c)
                    agent.position_history = [(r, c)]
                    agent.new_position = None

    def populate_grid(self):
        agents = [self.agent_factory(type_id) for type_id in ([0] * cfg.NUM_TYPE_A + [1] * cfg.NUM_TYPE_B)]
        np.random.shuffle(agents)
        flat_positions = [(r, c) for r in range(cfg.GRID_SIZE) for c in range(cfg.GRID_SIZE)]
        np.random.shuffle(flat_positions)
        for agent_id, (agent, pos) in enumerate(zip(agents, flat_positions[:len(agents)])):
            r, c = pos
            self.grid[r][c] = agent
            # Stable per-run identity (index in the seeded shuffle). Keys the
            # per-decision RNG streams in value-function mode, see
            # LLMAgent._get_value_function_decision.
            agent.agent_id = agent_id
            # Assign starting position and initialize position tracking
            agent.starting_position = (r, c)
            agent.position_history = [(r, c)]  # Track all positions throughout the run
            agent.new_position = None  # Initialize new_position attribute

    def update_agents(self, verbose_move_log=False):
        all_positions = [(r, c) for r in range(cfg.GRID_SIZE) for c in range(cfg.GRID_SIZE) if self.grid[r][c]]
        np.random.shuffle(all_positions)
        moved = False
            
        if verbose_move_log:
            print(f"[Step {self.step}] Processing {len(all_positions)} agents for movement decisions")
        
        for i, (r, c) in enumerate(all_positions):
            agent = self.grid[r][c]
            if agent is not None:
                agent.new_position = None  # Reset new position for this agent
                agent.step = self.step     # keys the decision RNG (value-function mode)
            move_to = self.decision_func(agent, r, c, self.grid)
                        
            # Log each agent's move decision
            if move_to and move_to != (r, c):
                r_new, c_new = move_to
                if 0 <= r_new < cfg.GRID_SIZE and 0 <= c_new < cfg.GRID_SIZE:
                    if self.grid[r_new][c_new] is None: # Target position is empty
                        # Track the new position
                        agent.new_position = (r_new, c_new)
                        agent.position_history.append((r_new, c_new))
                        
                        # Move the agent
                        self.grid[r_new][c_new] = agent
                        self.grid[r][c] = None
                        moved = True
                                            
                        # Log the move
                        self.log_agent_move(agent, r, c, move_to, True, (r_new, c_new), 'successful_move', verbose_move_log)
                        self.log_state_per_move()

                    else:
                        # Target occupied
                        agent.new_position = (r, c)
                        self.log_agent_move(agent, r, c, move_to, False, (r, c), 'target_occupied', verbose_move_log)
                        self.log_state_per_move()

                else:
                    # Invalid move (out of bounds)
                    agent.new_position = (r, c)
                    self.log_agent_move(agent, r, c, move_to, False, (r, c), 'invalid_target', verbose_move_log)                        
                    self.log_state_per_move()

            else:
                # Agent stays in current position
                agent.new_position = (r, c)
                if move_to is None:
                    self.log_agent_move(agent, r, c, move_to, False, (r, c), 'chose_to_stay', verbose_move_log)
                    self.log_state_per_move()

                else:
                    self.log_agent_move(agent, r, c, move_to, False, (r, c), 'same_position', verbose_move_log)
                    self.log_state_per_move()
    
        if verbose_move_log:
            print(f"[Step {self.step}] Movement phase complete - {'Some' if moved else 'No'} agents moved this step")
        return moved

    def run_step(self, verbose_move_log=False):
        moved = self.update_agents(verbose_move_log=verbose_move_log)
        # One int grid per step, shared by the metrics and the frame log; the
        # metrics used to walk the object grid separately (2026-09-05).
        int_grid = self._grid_to_int()
        metrics = calculate_all_metrics(int_grid)
        metrics['step'] = self.step
        metrics['run_id'] = self.run_id
        self.metrics_history.append(metrics)
        if not self.full_move_log:
            self.states.append(int_grid)               # grid after this step
        if not moved:
            self.no_move_steps += 1
        else:
            self.no_move_steps = 0
        if self.no_move_steps >= self.no_move_threshold:
            self.converged = True
            # FIRST step of the no-move window, not the step the criterion
            # tripped on (2026-09-01) — matching the two log-reconstruction
            # paths, which have always reported first-of-window. See
            # convergence_from_step_moves. final_step stays self.step, so
            # final_step == convergence_step + NO_MOVE_THRESHOLD - 1 for every
            # converged run regardless of which path produced the row.
            self.convergence_step = self.step - (self.no_move_steps - 1)
        if not self.converged:
            self.step += 1 # Increment step only if not converged 
        return self.converged

    def _grid_to_int(self):
        # int8, not the platform int (2026-08-27): cells only ever hold -1
        # (empty), 0 or 1 (the two agent types in config.py), so 64 bits per
        # cell wasted 8x the memory and 4x the npz size. Readers are
        # dtype-agnostic (equality masks, explicit casts, np.array_equal), and
        # a hypothetical type_id > 127 would raise OverflowError here rather
        # than corrupt silently. .tolist() still yields Python ints, so the
        # move-log JSON is unchanged.
        return as_int_grid(self.grid)

    @staticmethod
    def _normalize_save_every_steps(save_every_steps):
        """Interval between INTERMEDIATE saves, or None for 'only at the end'.

        None is the default and now means NO intermediate saves (changed
        2026-08-27; it used to mean 'every step', the most expensive setting).
        Both writers rewrite their whole file from scratch — save_agent_move_log
        re-dumps the entire move log (a full grid per record) and save_states
        re-compresses every frame — so saving each step made total I/O grow as
        O(steps^2). Measured on a 10x10 grid: ~95% of run time was json.dump,
        and per-step cost rose 10x (33ms -> 320ms) going from a 200- to a
        1000-step cap. Nothing is lost by skipping them: run_single_simulation
        always saves unconditionally after the loop. Intermediate saves only
        buy crash granularity, and a run killed mid-way is re-run from scratch
        anyway. Values < 1 (and unparseable ones) also mean 'only at the end'.
        """
        if save_every_steps is None:
            return None
        try:
            value = int(save_every_steps)
        except (TypeError, ValueError):
            return None
        return value if value >= 1 else None

    @staticmethod
    def _process_pool_context():
        for method in ("spawn", "forkserver"):
            try:
                return mp.get_context(method)
            except ValueError:
                continue
        return mp.get_context()

    def run_single_simulation(self, output_dir=None, max_steps=1000, show_progress=False, save_every_steps=None):
        """Run a single simulation and optionally save agent moves."""
        save_every_steps = self._normalize_save_every_steps(save_every_steps)
        progress_bar = None
        if show_progress:
            progress_bar = tqdm(total=max_steps, desc=f"Run {self.run_id} ({self.scenario})", 
                               unit="step", leave=True, ncols=80)
        
        while not self.converged and self.step < max_steps:
            self.run_step()
            if save_every_steps is not None and self.step % save_every_steps == 0:
                self.save_states(output_dir)
                self.save_agent_move_log(output_dir)  # Save the detailed move log
            
            if progress_bar:
                progress_bar.update(1)
                
                # Update progress bar with current status
                if self.step % 10 == 0:  # Update postfix every 10 steps to avoid spam
                    progress_bar.set_postfix({
                        'converged': self.converged,
                        'no_move_steps': self.no_move_steps,
                        'moves_logged': len(self.agent_move_log) if self.full_move_log
                                        else sum(row['decisions'] for row in self.step_log)
                    })
        
        if progress_bar:
            progress_bar.close()
        self.save_states(output_dir)
        self.save_agent_move_log(output_dir)  # Save the detailed move log

        # Print summary statistics
        if self.full_move_log:
            decisions = sum(1 for entry in self.agent_move_log if entry['reason'] != 'initial_state')
            moves = sum(1 for entry in self.agent_move_log if entry['moved'])
        else:
            decisions = sum(row['decisions'] for row in self.step_log)
            moves = sum(row['moved'] for row in self.step_log)
        print(f"[Run {self.run_id}] Move summary: {moves} moves, {decisions - moves} stays, {self.step} steps")
        
        return {
            'run_id': self.run_id,
            'scenario': self.scenario,
            'converged': self.converged,
            'convergence_step': self.convergence_step,
            'final_step': self.step,
            'metrics_history': self.metrics_history,
            # COUNT only, not the frames (2026-08-27). The full per-move history
            # is persisted to states_run_<id>.npz and every consumer reads it
            # from there; returning it here also shipped it through the Pool
            # pickle (~66 MB for a 1000-step run) and the parent retains every
            # result until analyze_results — up to ~7.5 GB resident per
            # scenario, for data nothing read.
            'n_states': len(self.states),
            'initial_grid': self.states[0] if self.states else None,
            'total_agent_moves': decisions
        }

    def save_states(self, output_dir):
        """Save the grid frames: per step, or per move in full_move_log mode."""
        if output_dir is not None:
            states_dir = os.path.join(output_dir, "states")
            os.makedirs(states_dir, exist_ok=True)

            # Save grid states as numpy arrays (includes state after every individual move)
            states_array = np.array(self.states)
            np.savez_compressed(os.path.join(states_dir, f"states_run_{self.run_id}.npz"), 
                              states=states_array)
            
            # print(f"[Run {self.run_id}] Saved {len(self.states)} grid states (including after each move)")

    def save_agent_move_log(self, output_dir):
        """Save the move log: step_moves_run_<id>.csv, or the per-move JSON in full mode."""
        if output_dir is None:
            return
        if not self.full_move_log:
            if self.step_log:
                os.makedirs(os.path.join(output_dir, "move_logs"), exist_ok=True)
                pd.DataFrame(self.step_log, columns=list(run_files.STEP_LOG_COLUMNS)).to_csv(
                    run_files.step_log_path(output_dir, self.run_id), index=False)
            return
        if self.agent_move_log:
            move_logs_dir = os.path.join(output_dir, "move_logs")
            os.makedirs(move_logs_dir, exist_ok=True)
            
            # Convert to DataFrame for easy CSV export
            # df = pd.DataFrame(self.agent_move_log)
            # Save as CSV for easy analysis
            # csv_path = os.path.join(move_logs_dir, f"agent_moves_run_{self.run_id}.csv") 
            # df.to_csv(csv_path, index=False) # NOTE: uncomment to save readable csv files
            
            # Also save as compressed JSON for complete data
            json_path = os.path.join(move_logs_dir, f"agent_moves_run_{self.run_id}.json.gz")
            with gzip.open(json_path, 'wt', encoding='utf-8') as f:
                # write(dumps(...)) rather than dump(..., f): dump streams the
                # encoder token by token, issuing ~15M tiny write() calls
                # through the gzip wrapper for a long run. Serialising once and
                # writing it in one call is ~2.3x faster and byte-identical.
                # separators/default MUST be preserved — dropping separators
                # inflates the file 22.8%, and dropping default=str turns any
                # non-JSON-native value into a TypeError that would lose the
                # whole run's move log at the final save.
                f.write(json.dumps(self.agent_move_log,
                                   separators=(',', ':'), default=str))
            
            # print(f"[Run {self.run_id}] Saved {len(self.agent_move_log)} agent move entries to {move_logs_dir}")

    def log_agent_move(self, agent, r, c, move_to, moved, new_position, reason, verbose_move_log=False):
        """Record one agent decision: a per-step count, or a full record in full_move_log mode."""
        if not self.full_move_log:
            if reason not in run_files.REASONS:
                raise ValueError(f"unknown move reason {reason!r}; expected one of {run_files.REASONS}")
            if not self.step_log or self.step_log[-1]['step'] != self.step:
                self.step_log.append(run_files.new_step_row(self.step))
            row = self.step_log[-1]
            row['decisions'] += 1
            row['moved'] += int(bool(moved))
            row[reason] += 1
            status = getattr(agent, 'last_llm_parse_status', None)
            row['parse_failed'] += int(status is not None and status != 'OK')
            self._print_move(agent, r, c, move_to, moved, new_position, reason, verbose_move_log)
            return

        # Create move entry for logging
        move_entry = {
            'step': self.step,
            'agent_id': id(agent) if agent else None,
            'type_id': agent.type_id if agent else None,
            'current_position': (r, c),
            'decision': move_to,
            'moved': moved,
            'new_position': new_position,
            'reason': reason,
            'llm_call_count': getattr(agent, 'llm_call_count', 0),
            'llm_call_time': getattr(agent, 'llm_call_time', 0.0),
            'timestamp': pd.Timestamp.now().isoformat(),
            # No 'grid' here (dropped 2026-09-01). log_state_per_move() appends
            # the identical array to self.states at the same instant, and that
            # is what states_run_<id>.npz stores, so record i and frame i were
            # always the same grid — written twice. The JSON copy cost ~38% of
            # the move log and the _grid_to_int() call behind it was ~40% of
            # run time. Readers take frame i from the npz; the invariant is
            # pinned by tests/test_move_log_and_states.py.
        }

        store_llm_responses = (
            getattr(cfg, 'STORE_LLM_RESPONSES', False) or
            os.environ.get('STORE_LLM_RESPONSES', '').lower() in ('true', '1', 'yes')
        )
        if store_llm_responses:
            move_entry['llm_raw_response'] = getattr(agent, 'last_llm_response_raw', None)
            move_entry['llm_parsed_decision'] = getattr(agent, 'last_llm_parsed_decision', None)
            move_entry['llm_parse_status'] = getattr(agent, 'last_llm_parse_status', None)
        
        # Add move entry to log
        self.agent_move_log.append(move_entry)
        self._print_move(agent, r, c, move_to, moved, new_position, reason, verbose_move_log)

    def _print_move(self, agent, r, c, move_to, moved, new_position, reason, verbose_move_log):
        if verbose_move_log:
            agent_id = f"Agent-{id(agent)}"
            if moved:
                print(f"[Step {self.step}] {agent_id} (Type {agent.type_id}) moved from ({r},{c}) to {new_position}")
            elif reason == 'target_occupied':
                print(f"[Step {self.step}] {agent_id} (Type {agent.type_id}) wanted to move from ({r},{c}) to {move_to} but target was occupied - stayed")
            elif reason == 'invalid_target':
                print(f"[Step {self.step}] {agent_id} (Type {agent.type_id}) wanted to move from ({r},{c}) to {move_to} but target was out of bounds - stayed")
            elif reason == 'chose_to_stay':
                print(f"[Step {self.step}] {agent_id} (Type {agent.type_id}) at ({r},{c}) chose to stay (decision: None)")
            else:
                print(f"[Step {self.step}] {agent_id} (Type {agent.type_id}) at ({r},{c}) chose to stay (decision: {move_to})")

    def log_state_per_move(self):
        """Full mode only: one frame per agent decision, paired 1:1 with the move records."""
        if self.full_move_log:
            self.states.append(self._grid_to_int())

    # --- Resume helpers ---
    def preload_record(self, output_dir):
        """Resume from the run's saved record, so the final save keeps steps
        0..self.step-1 instead of overwriting them with the resumed part.

        Installs the saved rows/records and frames up to the seed grid (the
        last saved frame, which must equal the grid this run was seeded
        with); leaves the record untouched when the files do not line up.
        """
        frames = run_files.load_frames(output_dir, self.run_id)
        if self.full_move_log:
            records = run_files.load_move_log_json(output_dir, self.run_id) or []
            kept = [r for r in records if int(r.get('step', self.step)) < self.step]
        else:
            step_log = run_files.load_step_log(output_dir, self.run_id)
            kept = [] if step_log is None else step_log[step_log['step'] < self.step].to_dict('records')
        n_frames = len(kept) if self.full_move_log else len(kept) + 1
        if not kept or frames is None or len(frames) < n_frames \
                or not np.array_equal(frames[n_frames - 1], self._grid_to_int()):
            print(f"[Run {self.run_id}] Warning: saved record does not match the resume seed; "
                  f"only the resumed steps will be saved")
            return
        self.states = list(frames[:n_frames])
        if self.full_move_log:
            self.agent_move_log = kept
        else:
            self.step_log = kept

    def set_state_from_int_grid(self, int_grid, step=None):
        """Set the current simulation grid from a 2D array/list of ints and optionally the next step.

        int_grid: shape (GRID_SIZE, GRID_SIZE); -1 empty, otherwise type_id
        step: if provided, sets self.step to this (the next step index)
        """
        if int_grid is None:
            return
        arr = np.array(int_grid)
        if arr.shape != (cfg.GRID_SIZE, cfg.GRID_SIZE):
            raise ValueError("int_grid shape mismatch with GRID_SIZE")
        # Clear grid
        self.grid = np.full((cfg.GRID_SIZE, cfg.GRID_SIZE), None)
        for r in range(cfg.GRID_SIZE):
            for c in range(cfg.GRID_SIZE):
                t = int(arr[r, c])
                if t >= 0:
                    agent = self.agent_factory(t)
                    self.grid[r][c] = agent
                    agent.starting_position = (r, c)
                    agent.position_history = [(r, c)]
                    agent.new_position = None
        if step is not None:
            try:
                self.step = int(step)
            except Exception:
                pass

    _PLACEHOLDER_FINAL_STEP = 'unknown'

    @staticmethod
    def _read_csv_if_present(path):
        """Existing CSV as a DataFrame; empty frame if absent or unreadable."""
        if not os.path.exists(path):
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception as exc:                      # corrupt/partial file
            print(f"Warning: could not read {path} ({exc}); "
                  f"treating as empty — prior rows may be lost")
            return pd.DataFrame()

    @staticmethod
    def analyze_results(results, output_dir, n_runs):
        """Analyze simulation results and save metrics, convergence data, and step statistics.

        RESUME SAFETY (2026-08-25): rows already on disk are MERGED rather than
        overwritten. On resume, llm_runner rebuilds previously completed runs
        from their .npz state files, which carry no metrics_history and only
        placeholder convergence values (converged=True, final_step='unknown').
        Writing those straight out replaced the real history of every earlier
        run with nothing. Merge policy, keyed by run_id:
          * metrics    — a re-executed run replaces its own old rows; runs not
                         in this batch keep the rows already on disk;
          * convergence — a real new row wins over disk, but a PLACEHOLDER
                         never overwrites a real stored row.
        """
        all_metrics = []
        convergence_data = []

        for result in results:
            convergence_data.append({
                'run_id': result['run_id'],
                'converged': result['converged'],
                'convergence_step': result['convergence_step'],
                'final_step': result['final_step']
            })
            for metric in result['metrics_history']:
                all_metrics.append(metric)

        metrics_path = run_files.metrics_history_path(output_dir)
        conv_path = f"{output_dir}/convergence_summary.csv"

        # --- metrics: new rows replace their own run_id, others persist ------
        new_metrics = pd.DataFrame(all_metrics)
        disk_metrics = Simulation._read_csv_if_present(metrics_path)
        if not disk_metrics.empty and 'run_id' in disk_metrics.columns:
            if not new_metrics.empty:
                disk_metrics = disk_metrics[
                    ~disk_metrics['run_id'].isin(set(new_metrics['run_id']))]
            merged_metrics = pd.concat([disk_metrics, new_metrics], ignore_index=True)
        else:
            merged_metrics = new_metrics
        if not merged_metrics.empty and {'run_id', 'step'} <= set(merged_metrics.columns):
            merged_metrics = merged_metrics.sort_values(['run_id', 'step'],
                                                        kind='stable').reset_index(drop=True)
        merged_metrics.to_csv(metrics_path, index=False)

        # --- convergence: placeholders never clobber real stored rows --------
        by_run = {}
        for row in convergence_data:                       # placeholders first
            if row.get('final_step') == Simulation._PLACEHOLDER_FINAL_STEP:
                by_run[row['run_id']] = row
        disk_conv = Simulation._read_csv_if_present(conv_path)
        if not disk_conv.empty and 'run_id' in disk_conv.columns:
            for row in disk_conv.to_dict('records'):       # disk beats placeholder
                by_run[row['run_id']] = row
        for row in convergence_data:                       # real new rows win
            if row.get('final_step') != Simulation._PLACEHOLDER_FINAL_STEP:
                by_run[row['run_id']] = row
        merged_conv = [by_run[k] for k in sorted(by_run)]
        pd.DataFrame(merged_conv).to_csv(conv_path, index=False)

        # --- step statistics over the MERGED history -------------------------
        df = merged_metrics
        metric_cols = [c for c in df.columns if c not in ('step', 'run_id')]
        if not df.empty and 'step' in df.columns and metric_cols:
            step_stats = df.groupby('step').agg(
                {c: ['mean', 'std', 'min', 'max'] for c in metric_cols}).reset_index()
            step_stats.columns = ['_'.join(col).strip() if col[1] else col[0]
                                  for col in step_stats.columns.values]
            step_stats.to_csv(f"{output_dir}/step_statistics.csv", index=False)

        # --- one row per run: convergence step + final-step metrics ----------
        # Written here rather than in each runner so baseline_runner,
        # llm_runner and the load_and_analyze_results reload path all produce
        # it without their own call site (2026-09-01). Imported locally: the
        # module imports DissimilarityIndex and config, and base_simulation is
        # imported by tests that stub those.
        from run_summary import write_run_summary
        write_run_summary(output_dir, results)

        return output_dir, results, merged_conv

    @staticmethod
    def load_results_from_output(output_dir, force_recompute: bool = False):
        """
        Load simulation results from stored output files to feed into analyze_results function.
        
        Args:
            output_dir (str): Directory containing saved simulation outputs
            force_recompute (bool): If True, ignore existing analysis files and rebuild
                from raw logs/states when possible.
            
        Returns:
            tuple: (results, n_runs) where results is a list of result dictionaries
                  compatible with analyze_results function
        """
        results = []
        
        # Check if metrics_history.csv already exists (from previous analysis)
        metrics_file = run_files.metrics_history_path(output_dir)
        convergence_file = os.path.join(output_dir, "convergence_summary.csv")
        
        if (not force_recompute) and os.path.exists(metrics_file) and os.path.exists(convergence_file):
            print(f"Loading existing analysis files from {output_dir}")
            
            # Load pre-computed metrics and convergence data
            metrics_df = pd.read_csv(metrics_file)
            convergence_df = pd.read_csv(convergence_file)
            metrics_by_run = {
                run_id: group.to_dict('records')
                for run_id, group in metrics_df.groupby('run_id', sort=False)
            }
            convergence_first_row_by_run = {}
            for _, row in convergence_df.iterrows():
                run_id = row['run_id']
                if run_id not in convergence_first_row_by_run:
                    convergence_first_row_by_run[run_id] = row
            
            # Group metrics by run_id to reconstruct results structure
            for run_id in convergence_df['run_id'].unique():
                convergence_row = convergence_first_row_by_run[run_id]
                run_metrics = metrics_by_run.get(run_id, [])
                
                result = {
                    'run_id': run_id,
                    'converged': convergence_row['converged'],
                    'convergence_step': convergence_row['convergence_step'] if pd.notna(convergence_row['convergence_step']) else None,
                    'final_step': convergence_row['final_step'],
                    'metrics_history': run_metrics
                }
                results.append(result)
                
            n_runs = len(convergence_df)
            print(f"Loaded {n_runs} simulation runs from existing analysis files")
            
        else:
            print(f"Loading raw simulation data from {output_dir}")
            
            # Load from individual move log files
            move_logs_dir = os.path.join(output_dir, "move_logs")

            if not os.path.exists(move_logs_dir):
                raise FileNotFoundError(f"Move logs directory not found: {move_logs_dir}")

            run_ids = run_files.list_run_ids(output_dir)

            print(f"Found {len(run_ids)} simulation runs: {run_ids}")
            threshold = getattr(cfg, 'NO_MOVE_THRESHOLD', 5)
            tasks = [(run_id, output_dir, threshold) for run_id in run_ids]

            max_workers = min(len(tasks), max(1, os.cpu_count() or 1)) if tasks else 1

            if max_workers > 1 and len(tasks) > 1:
                print(f"Processing runs in parallel with {max_workers} workers")
                try:
                    with ProcessPoolExecutor(max_workers=max_workers, mp_context=Simulation._process_pool_context()) as executor:
                        results = list(executor.map(_load_single_run_result, tasks))
                except Exception as e:
                    print(f"Warning: Parallel processing failed ({e}), falling back to sequential")
                    for task in tasks:
                        run_id = task[0]
                        print(f"Loading run {run_id}...")
                        results.append(_load_single_run_result(task))
            else:
                for task in tasks:
                    run_id = task[0]
                    print(f"Loading run {run_id}...")
                    results.append(_load_single_run_result(task))
            
            n_runs = len(results)
            print(f"Loaded {n_runs} simulation runs from raw data")
        
        return results, n_runs

    @staticmethod
    def load_and_analyze_results(output_dir, force_recompute: bool = False):
        """
        Convenience function that loads stored simulation outputs and runs analysis.
        
        Args:
            output_dir (str): Directory containing saved simulation outputs
            force_recompute (bool): If True, ignore existing analysis files and rebuild
                from raw logs/states when possible.
            
        Returns:
            tuple: (output_dir, results, convergence_data) from analyze_results
        """
        print(f"Loading and analyzing results from: {output_dir}")
        
        # Load the results from stored output
        # Be robust to monkeypatched or older signatures that don't accept force_recompute
        try:
            results, n_runs = Simulation.load_results_from_output(output_dir, force_recompute=force_recompute)
        except TypeError:
            try:
                # Try positional in case only positional args are supported
                results, n_runs = Simulation.load_results_from_output(output_dir, force_recompute)
            except TypeError:
                # Fall back to legacy call with only output_dir
                results, n_runs = Simulation.load_results_from_output(output_dir)
        
        if not results:
            raise ValueError(f"No simulation results found in {output_dir}")
        
        print(f"Analyzing {n_runs} simulation runs...")
        
        # Run the analysis
        return Simulation.analyze_results(results, output_dir, n_runs)
