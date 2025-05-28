# Schelling Segregation Simulator

This Python project simulates the spatial segregation dynamics described in Pancs & Vriend (2007).

## Features

- 2D lattice with Agent-based model
- Mechanistic and optional LLM-based agents (via Ollama)
- Multiple segregation metrics
- GUI using PyGame

## To Run

Install dependencies:

```bash
pip install pygame pygame_gui numpy
```

Run the simulation:

```bash
python SchellingSim.py
```

To enable LLM-based agents, set `USE_LLM = True` in `config.py` and configure `LLMAgent.py`.
