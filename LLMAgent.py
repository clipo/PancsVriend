
import requests
import config as cfg
import re

def maybe_use_llm_agent(agent, r, c, grid):
    try:
        # Construct 3x3 neighborhood context
        context = []
        for dr in [-1, 0, 1]:
            row = []
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < cfg.GRID_SIZE and 0 <= nc < cfg.GRID_SIZE:
                    neighbor = grid[nr][nc]
                    if neighbor is None:
                        row.append("E")  # Empty
                    elif neighbor.type_id == agent.type_id:
                        row.append("S")  # Same
                    else:
                        row.append("O")  # Opposite
                else:
                    row.append("X")  # Out of bounds
            context.append(row)

        prompt = f"""You are an agent deciding where to move in a 3x3 grid.
Each cell contains:
- 'S' for same-type neighbor
- 'O' for opposite-type neighbor
- 'E' for empty space
- 'X' for out-of-bounds

Only respond with the coordinates of the best empty cell to move to, formatted exactly as (row, col).
If you choose not to move, respond only with: None

DO NOT provide any explanation. DO NOT include anything else.

Here is your 3x3 grid (centered on you):
{context}
"""

        payload = {
            "model": cfg.OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }

        headers = {
            "Authorization": f"Bearer {cfg.OLLAMA_API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(cfg.OLLAMA_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        print("[LLM RAW Response]", text)

        # Try to parse a move tuple (row, col)
        match = re.search(r"\((\d+),\s*(\d+)\)", text)
        if match:
            move_to = (int(match.group(1)), int(match.group(2)))
            print("[LLM Parsed Move via regex]", move_to)
            return move_to

        if "none" in text.strip().lower():
            print("[LLM Decision] No move.")
            return None

        print("[LLM Format Unrecognized]", text)
        return None

    except Exception as e:
        print("[LLM Request Failed]", e)
        return None
