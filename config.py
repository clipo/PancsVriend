GRID_SIZE = 20
CELL_SIZE = 20

WINDOW_WIDTH = GRID_SIZE * CELL_SIZE + 200  # add horizontal space for GUI controls
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + 100
FPS = 30

NUM_TYPE_A = 150
NUM_TYPE_B = 150

BG_COLOR = (255, 255, 255)
GRID_LINE_COLOR = (180, 180, 180)
COLOR_A = (255, 0, 0)  # Red
COLOR_B = (0, 0, 255)  # Blue
VACANT_COLOR = (200, 200, 200)  # Light Gray

USE_LLM = False

UTILITY_BASE = 1
UTILITY_SLOPE = 1
UTILITY_CUTOFF = 50

# LLM API config
OLLAMA_MODEL = "qwen2.5-coder:32B"
OLLAMA_URL = "https://chat.binghamton.edu/api/chat/completions"
OLLAMA_API_KEY = "sk-571df6eec7f5495faef553ab5cb2c67a"