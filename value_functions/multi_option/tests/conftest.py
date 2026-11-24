"""Make the multi_option modules importable in the tests, and the repo's own code when present."""

import sys  # the module search path
from pathlib import Path  # file paths

HERE = Path(__file__).resolve().parent  # .../value_functions/multi_option/tests
sys.path.insert(0, str(HERE))  # the fake server
sys.path.insert(0, str(HERE.parent))  # option_language, multi_option_path_sum, ...
import os  # environment variables
REPO = Path(os.environ.get("PANCSVRIEND_REPO", HERE.parents[2]))  # the repo root (this folder is value_functions/multi_option/tests)
for sub in ("", "value_functions", "value_functions/logprob", "prompt_refinement"):  # where the repo keeps its modules
    p = REPO / sub
    if p.is_dir() and str(p) not in sys.path:
        sys.path.append(str(p))
