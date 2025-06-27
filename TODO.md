# TODO

## Highest Priority (Current Focus)

### 1. Collapse different agent classes into a single parameterized agent class
- [ ] Files to refactor:
  - `llm_agent_with_memory.py` - LLMAgentWithMemory class with persistent context


### 2. Modify LLM code to store and display all prompt/response pairs
- [ ] Add prompt/
		** response storage mechanism to LLMAgent classes 
- [ ] Display LLM interactions to console with timestamps
- [ ] Store interactions in a structured format (JSON/CSV)
- [ ] Save prompt/response history to experiment directory
- [ ] Update llm_runner.py and llm_agent_with_memory.py

### Scenarios to run
python llm_runner.py --runs 100 --processes 10 --scenario 'race_white_black'

for example 

see context_scenarios.py









-----
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