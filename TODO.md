# TODO

## Highest Priority (Current Focus)

### 1. Collapse different agent classes into a single parameterized agent class
- [ ] Files to refactor:
  - `Agent.py` - Base mechanical agent with utility-based decisions
  - `LLMAgent.py` - Basic LLM agent (function-based, not class-based)
  - `llm_agent_with_memory.py` - LLMAgentWithMemory class with persistent context
- [ ] Create unified `UnifiedAgent.py` with parameters:
  - `agent_type`: 'mechanical' | 'llm' | 'llm_with_memory'
  - `llm_config`: Optional LLM configuration (model, url, api_key)
  - `memory_enabled`: Boolean for memory persistence
  - `scenario`: Social context scenario
- [ ] Update all references in:
  - `baseline_runner.py`
  - `llm_runner.py`
  - `llm_runner_with_memory.py`
  - `SchellingSim.py`

### 2. Modify LLM code to store and display all prompt/response pairs
- [ ] Add prompt/response storage mechanism to LLMAgent classes
- [ ] Display LLM interactions to console with timestamps
- [ ] Store interactions in a structured format (JSON/CSV)
- [ ] Save prompt/response history to experiment directory
- [ ] Update llm_runner.py and llm_agent_with_memory.py

### 3. Remove automatic fallback to mechanical model on LLM failure
- [ ] Remove circuit breaker logic in llm_runner.py (lines 381-384, 459-461)
- [ ] Remove fallback calls to agent.best_response() on LLM errors
- [ ] Let LLM failures result in no movement instead of mechanical fallback
- [ ] Keep retry logic but remove mechanical substitution

### 4. Merge runner files into single comprehensive runner script
- [ ] Files to merge:
  - `baseline_runner.py` - Runs mechanical agent simulations
  - `llm_runner.py` - Runs LLM agent simulations
  - `llm_runner_with_memory.py` - Runs LLM agents with memory
  - `run_experiments.py` - Master orchestrator
  - All other experiment runners (comprehensive_comparison_study.py, etc.)
- [ ] Create unified `universal_runner.py` with:
  - Support for all agent types (mechanical, LLM, LLM with memory)
  - Configurable via command-line arguments
  - Single entry point for all simulation types
  - Batch and individual run capabilities

### 5. Update grid state saving to capture every agent decision
- [ ] Modify `_grid_to_int()` function to save state after each individual agent move
- [ ] Update state storage to include:
  - Agent ID that moved
  - Source position (r, c)
  - Destination position (r_new, c_new)
  - Timestamp of decision
  - LLM prompt/response if applicable
- [ ] Ensure state files don't become too large (consider compression)
- [ ] Add option to toggle between step-wise and move-wise saving

## High Priority

### Framework Improvements
- [ ] Add support for more LLM providers (Gemini, Mistral)
- [ ] Implement caching mechanism for LLM responses to reduce API costs
- [ ] Add retry logic with exponential backoff for LLM API failures
- [ ] Create unified experiment configuration system (replace multiple YAML files)

### Analysis & Visualization
- [ ] Add real-time dashboard for monitoring running experiments
- [ ] Implement convergence speed comparison across scenarios
- [ ] Add statistical power analysis for experiment design
- [ ] Create interactive web-based visualization dashboard

### Performance Optimization
- [ ] Parallelize baseline simulations across multiple cores
- [ ] Optimize memory usage for large-scale experiments
- [ ] Implement checkpointing for long-running experiments
- [ ] Add GPU support for matrix operations in metrics calculations

## Medium Priority

### Code Quality
- [ ] Add comprehensive unit tests for all modules
- [ ] Implement type hints throughout codebase
- [ ] Create CI/CD pipeline with automated testing
- [ ] Add code coverage reporting

### Documentation
- [ ] Write detailed API documentation
- [ ] Create user guide with examples
- [ ] Add architecture diagrams
- [ ] Document all configuration options

### Features
- [ ] Add support for irregular grid topologies
- [ ] Implement agent memory persistence across runs
- [ ] Add scenario builder GUI
- [ ] Support for multi-attribute segregation analysis

## Low Priority

### Research Extensions
- [ ] Implement 3D visualization of segregation patterns
- [ ] Add support for dynamic population sizes
- [ ] Create scenario templates for common research questions
- [ ] Implement agent learning mechanisms

### Infrastructure
- [ ] Dockerize the application
- [ ] Create cloud deployment scripts
- [ ] Add experiment result database
- [ ] Implement result sharing mechanism

## Completed
- [x] Basic framework implementation
- [x] LLM integration with multiple providers
- [x] Statistical analysis pipeline
- [x] PDF report generation
- [x] Interactive GUI simulation