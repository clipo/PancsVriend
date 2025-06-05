# ===== Common Model Settings (used by baseline_runner.py & llm_runner.py) =====
GRID_SIZE = 20                           # number of cells per side of the grid
NUM_TYPE_A = 150                        # initial count of Type A agents (e.g., red)
NUM_TYPE_B = 150                         # initial count of Type B agents (e.g., blue)

# ===== Base Runner Settings (used by baseline_runner.py) =====
UTILITY_BASE = 1                         # base utility value in agents' decision function
UTILITY_SLOPE = 1                        # slope factor in the utility calculation
UTILITY_CUTOFF = 50                      # threshold utility for relocation decisions

# ===== LLM Runner Settings (used by llm_runner.py) =====
OLLAMA_MODEL = "qwen2.5-coder:32B"          # LLM model identifier for Ollama
OLLAMA_URL = "https://chat.binghamton.edu/api/chat/completions"  # API endpoint for LLM service
OLLAMA_API_KEY = "sk-571df6eec7f5495faef553ab5cb2c67a"  # authentication key for LLM API

# ===== GUI Simulation Settings (used by SchellingSim.py) =====
USE_LLM = False                          # toggle LLM-based decisions (True) vs mechanical (False)
CELL_SIZE = 20                           # pixel size of each square cell
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE + 200  # window width in px (grid area + GUI controls)
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + 100 # window height in px (grid area + footer)
FPS = 30                                  # frames per second for simulation and rendering

BG_COLOR = (255, 255, 255)               # background color (RGB white)
GRID_LINE_COLOR = (180, 180, 180)        # grid line color (RGB light gray)
COLOR_A = (255, 0, 0)                    # color for Type A agents (RGB red)
COLOR_B = (0, 0, 255)                    # color for Type B agents (RGB blue)
VACANT_COLOR = (200, 200, 200)           # color for empty cells (RGB gray)